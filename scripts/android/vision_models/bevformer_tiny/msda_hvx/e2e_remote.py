#!/usr/bin/env python3
"""BEVFormer-tiny frame by frame with the encoder and decoder compiled by the tinygrad fork and run through onnx-remote.

Per keyframe of a scene: backbone (int8, HTP, qnn_run_multi like ../e2e_phone.py) -> encoder (<work>/enc3.msda.onnx) ->
decoder (<work>/decoder.msda.onnx), the last two through onnx-remote (compile_v69.sh behind onnx-remote-compiler, the
hexagon worker on the cDSP: ../../../tinygrad_hexagon_bridge/openpilot_v65/e2e.sh). Each piece is fed the previous one's
phone output, and the phone's own BEV is carried to the next frame as prev_bev (rotated on the host), like e2e_msda.py.
fp32 torch runs alongside; reports per-frame cos and GT matches for both, and each piece's latency. --htp-decoder also
runs the fp16 HTP decoder on the same BEV, for comparison.

usage: e2e_remote.py --ckpt <pth> --data <nuscenes-mini> --work <work> [--scene scene-0103] [--frames 6] [--iters 10]
       [--backbone backbone6.q8] [--htp-decoder decoder.sim]
env: CLIENT (onnx-remote-client), WORKER (onnx-remote-hexagon-worker), COMPILER_PORT (the compiler service, started with
     the fork's default env: GEMM_FP16=1 HMX=1 ...), PHONE_RUN (the host's phone lock: every adb call goes through it),
     R (the HTP pieces' phone directory, default /data/local/tmp/bevformer_tiny: qnn_run_multi, the QNN libs, the backbone)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import e2e_phone as E  # noqa: E402
import model as M  # noqa: E402
from nuscenes import NuScenesMini, match, temporal_can_bus  # noqa: E402

E2E = HERE.parents[2] / "tinygrad_hexagon_bridge" / "openpilot_v65" / "e2e.sh"
if os.environ.get("PHONE_RUN"):
    E.ADB = [os.environ["PHONE_RUN"], *E.ADB]


def remote(
    work: Path, model: str, inputs: dict, iters: int
) -> tuple[list[np.ndarray], float, float]:
    """<work>/<model>.onnx through onnx-remote on the cDSP -> (float outputs, DSP ms, best RPC-inclusive ms)."""
    tmp = work / "e2e_remote"
    tmp.mkdir(exist_ok=True)
    args = []
    for k, v in inputs.items():
        v = np.ascontiguousarray(v, np.float32)
        v.tofile(tmp / f"{k}.bin")
        args += ["--input-raw", f"1:{','.join(map(str, v.shape))}:{tmp / f'{k}.bin'}"]
    out = subprocess.run(
        [
            "bash",
            str(E2E),
            os.environ["WORKER"],
            str(work / f"{model}.onnx"),
            *args,
            "--dump",
            str(tmp / "out.bin"),
            "--iters",
            str(iters),
        ],
        capture_output=True,
        text=True,
        cwd=tmp,
    )
    log = out.stdout + out.stderr
    if out.returncode or "dsp_us=" not in log:
        raise RuntimeError(f"{model}: {log[-1500:]}")
    dsp = float(re.findall(r"dsp_us=(\d+)", log)[-1]) / 1000
    rpc = float(re.search(r"best ([0-9.]+) ms", log).group(1))
    return np.fromfile(tmp / "out.bin", np.float32), dsp, rpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--backbone", default="backbone6.q8")
    ap.add_argument(
        "--htp-decoder",
        default="",
        help="also run this HTP decoder (e.g. decoder.sim) on the cDSP's BEV",
    )
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work = Path(a.work)
    bb, enc, dec = M.load_official(a.ckpt)
    ns = NuScenesMini(a.data)
    prev = {"cpu": None, "dsp": None}
    prev_abs = None
    keys = ["cpu", "dsp"] + (["htp"] if a.htp_decoder else [])
    tot = {k: [0, 0, 0] for k in keys}
    ms: dict[str, list[float]] = {
        k: []
        for k in ("backbone", "enc_dsp", "enc_rpc", "dec_dsp", "dec_rpc", "dec_htp")
    }
    for i, tok in enumerate(ns.scene_samples(a.scene)[: a.frames]):
        f = ns.frame(tok)
        can_bus = temporal_can_bus(f["can_bus_abs"], prev_abs)
        ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
        first = prev["cpu"] is None
        has_prev = torch.zeros(1) if first else torch.ones(1)
        shift = torch.zeros(2) if first else M.can_bus_shift(can_bus)
        feats = bb(f["img"])
        p_in = (
            torch.zeros(M.NQ, M.EMBED)
            if first
            else M.rotate_prev_bev(prev["cpu"], can_bus)
        )
        bev = enc(feats, p_in, has_prev, shift, can_bus, ref_cam, bev_mask)
        cls, bbox = dec(bev)
        (feats_h,), t_b = E.phone(work, a.backbone, {"img": f["img"].numpy()})
        p_h = (
            np.zeros((M.NQ, M.EMBED), np.float32)
            if first
            else M.rotate_prev_bev(torch.from_numpy(prev["dsp"]), can_bus).numpy()
        )
        enc_in = {
            "feats": feats_h,
            "prev_bev": p_h,
            "has_prev": has_prev,
            "shift": shift,
            "can_bus": can_bus,
            "ref_cam": ref_cam,
            "bev_mask": bev_mask.float(),
        }
        bev_d, te_d, te_r = remote(
            work,
            "enc3.msda",
            {k: np.asarray(v, np.float32) for k, v in enc_in.items()},
            a.iters,
        )
        bev_d = bev_d.reshape(M.NQ, M.EMBED)
        enc_cos = E.cos(
            enc(*[torch.as_tensor(np.asarray(v, np.float32)) for v in enc_in.values()]),
            bev_d,
        )
        outs, td_d, td_r = remote(work, "decoder.msda", {"bev_embed": bev_d}, a.iters)
        n = cls.numel()
        res = {
            "cpu": (cls, bbox),
            "dsp": (
                torch.from_numpy(outs[:n].reshape(cls.shape).copy()),
                torch.from_numpy(outs[n:].reshape(bbox.shape).copy()),
            ),
        }
        t_h = 0.0
        if a.htp_decoder:
            (cls_h, bbox_h), t_h = E.phone(work, a.htp_decoder, {"bev_embed": bev_d})
            res["htp"] = (torch.from_numpy(cls_h), torch.from_numpy(bbox_h))
        r = {}
        for k, (c, b) in res.items():
            boxes, scores, labels = M.decode(c, b)
            r[k] = match(boxes, scores, labels, f["gt"])
            for j in range(3):
                tot[k][j] += r[k][j]
        for k, v in zip(ms, (t_b, te_d, te_r, td_d, td_r, t_h)):
            ms[k].append(v)
        print(
            f"frame {i}: ms backbone {t_b:.1f} encoder {te_d:.1f} (rpc {te_r:.1f}) decoder {td_d:.1f} (rpc {td_r:.1f})"
            + (f" htp decoder {t_h:.1f}" if a.htp_decoder else "")
            + f" | cos vs fp32: feats {E.cos(feats, feats_h):.5f} bev {E.cos(bev, bev_d):.5f} (encoder alone {enc_cos:.6f})"
            f" cls {E.cos(cls, res['dsp'][0]):.5f} bbox {E.cos(bbox, res['dsp'][1]):.5f} | matched/pred/GT {r}",
            flush=True,
        )
        prev["cpu"], prev["dsp"], prev_abs = bev, bev_d, f["can_bus_abs"]
    for k, (tp, npred, ngt) in tot.items():
        print(
            f"{k}: {tp} matched of {ngt} GT ({npred} detections >= 0.3) over {a.frames} frames"
        )
    med = {k: float(np.median(v)) for k, v in ms.items()}
    print(
        "median ms: "
        + " ".join(f"{k} {v:.1f}" for k, v in med.items() if v)
        + f" | frame (backbone + DSP encoder + DSP decoder) {med['backbone'] + med['enc_dsp'] + med['dec_dsp']:.1f}"
    )


if __name__ == "__main__":
    main()
