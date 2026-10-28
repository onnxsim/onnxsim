#!/usr/bin/env python3
"""BEVFormer-tiny's whole frame in one phone process (frame_run): per-frame latency, pipelined
throughput, and whether the HTP and the HVX overlap.

Unlike e2e_msda.py (one process per piece, the host carrying tensors between them), frame_run chains
backbone -> encoder -> decoder in-process and carries the BEV itself: the host only prepares each
frame's inputs, which don't depend on any output --
- img.u8: the backbone's quantized NHWC input (quantize.py's qparams, backbone6.q8.json);
- tsa_ref / ref_cam / vis / can_bus / has_prev: split.host_inputs;
- rot_idx.i32: which prev_bev row lands on each BEV cell after rotate_prev_bev (torchvision's nearest
  rotate of an index image, checked exact against rotate_prev_bev itself), so the phone's prev_bev is a
  row gather of its own last BEV.

Then, on the phone (every adb call under PHONE_RUN's lock):
- `seq`: frames one at a time -> per-frame latency and outputs (GT matched like e2e_msda.py);
- `pipe`: backbone | encoder | decoder threads, two slots between stages -> throughput, latency, and
  the outputs bit-compared with seq's;
- `conc`: backbone / decoder alone, the 6 sampling calls alone, then each pair at once.

--tg-enc / --tg-dec: the encoder / decoder as tinygrad-compiled programs (.tghx: the fork's examples/dsp_graph_onnx.py
--artifact on export_msda_op.py's enc3.msda.onnx / decoder.msda.onnx) instead of the HTP pieces (frame_run's TG_ENC / TG_DEC).

usage: frame_e2e.py --ckpt <pth> --data <nuscenes-mini> --work <work> [--build build] [--modes seq,pipe,conc]
       [--tg-enc enc3.tghx] [--tg-dec decoder.tghx]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import e2e_msda as X  # noqa: E402  (sets E.R / E.ADB from R / PHONE_RUN)
import e2e_phone as E  # noqa: E402
import model as M  # noqa: E402
import split as S  # noqa: E402
from nuscenes import NuScenesMini, match, temporal_can_bus  # noqa: E402


def rot_idx(can_bus) -> np.ndarray:
    """(NQ,) int32: source row of rotate_prev_bev for each BEV cell, -1 where it fills zeros."""
    idx = (
        (torch.arange(M.NQ, dtype=torch.float32) + 1)
        .reshape(M.NQ, 1)
        .expand(M.NQ, M.EMBED)
    )
    r = M.rotate_prev_bev(idx.contiguous(), can_bus)[:, 0]
    out = (r.round().to(torch.int64) - 1).to(torch.int32).numpy()
    # exact: gathering rows reproduces rotate_prev_bev on arbitrary data
    x = torch.randn(M.NQ, M.EMBED)
    g = torch.where(
        torch.from_numpy(out)[:, None] >= 0,
        x[torch.from_numpy(out).clamp(min=0).long()],
        0,
    )
    assert torch.equal(g, M.rotate_prev_bev(x, can_bus)), (
        "rot_idx doesn't reproduce rotate_prev_bev"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--build", default=str(HERE / "build"))
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--backbone", default="backbone6.q8")
    ap.add_argument(
        "--decoder",
        default="decoder.sim",
        help="a decoder piece, or 'split' (dec_split.py export's pieces + HVX)",
    )
    ap.add_argument(
        "--pieces",
        default="msda_split",
        help="the encoder's HTP pieces: a <work> subdirectory of <piece>.sim.onnx (e.g. quantize_split.py's)",
    )
    ap.add_argument("--modes", default="seq,pipe,conc")
    ap.add_argument("--tg-enc", default="", help="a tinygrad encoder program (.tghx)")
    ap.add_argument("--tg-dec", default="", help="a tinygrad decoder program (.tghx)")
    ap.add_argument(
        "--reps",
        type=int,
        default=5,
        help="passes over the frames (seq/pipe), iterations (conc)",
    )
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work, R = Path(a.work), E.R
    qio = json.loads((work / f"{a.backbone}.json").read_text())
    bb, enc, dec = M.load_official(a.ckpt)
    ns = NuScenesMini(a.data)
    root = work / "frame_run"
    frames, gts, cpu = [], [], []
    prev, prev_abs = None, None
    for i, tok in enumerate(ns.scene_samples(a.scene)[: a.frames]):
        f = ns.frame(tok)
        can_bus = temporal_can_bus(f["can_bus_abs"], prev_abs)
        prev_abs = f["can_bus_abs"]
        ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
        first = prev is None
        has_prev = torch.zeros(1) if first else torch.ones(1)
        shift = torch.zeros(2) if first else M.can_bus_shift(can_bus)
        feats = bb(f["img"])
        p_in = torch.zeros(M.NQ, M.EMBED) if first else M.rotate_prev_bev(prev, can_bus)
        enc_in = (feats, p_in, has_prev, shift, can_bus, ref_cam, bev_mask)
        prev = enc(*enc_in)
        cpu.append((prev, *dec(prev)))
        gts.append(f["gt"])
        tsa_ref, ref_cam_h, vis = S.host_inputs(enc_in)
        q = qio["img"]
        img = np.clip(
            np.round(f["img"].numpy() / q["scale"]) + q["zero_point"], 0, 255
        ).astype(np.uint8)
        d = root / f"f{i}"
        d.mkdir(parents=True, exist_ok=True)
        f32 = lambda t: np.ascontiguousarray(np.asarray(t, np.float32))  # noqa: E731
        np.ascontiguousarray(img.transpose(0, 2, 3, 1)).tofile(d / "img.u8")
        for k, v in {
            "has_prev": has_prev,
            "can_bus": can_bus,
            "tsa_ref": tsa_ref,
            "ref_cam": ref_cam_h,
            "shift": shift,
            "bev_mask": bev_mask,
        }.items():
            f32(v).tofile(d / f"{k}.f32")
        np.ascontiguousarray(vis.numpy().astype(np.uint8)).tofile(d / "vis.u8")
        rot_idx(can_bus).tofile(d / "rot_idx.i32")
        frames.append(d)
    fq = qio["feats"]
    (work / "msda_split" / "feats_q.txt").write_text(
        f"{fq['scale']!r} {fq['zero_point']}\n"
    )

    split_dec = a.decoder == "split"
    X.push_runtime(
        work, Path(a.build), [a.backbone] + ([] if split_dec else [a.decoder])
    )
    if split_dec:  # dec_split.py export's pieces + layer 0's constants
        for p in ("dpre", "dmid0", "dmid1", "dmid2", "dmid3", "dmid4", "dpost"):
            subprocess.run(
                [
                    *E.ADB,
                    "push",
                    "-q",
                    str(work / "msda_split" / f"{p}.sim.onnx"),
                    f"{R}/msda_split/{p}.onnx",
                ],
                check=True,
            )
        subprocess.run(
            [
                *E.ADB,
                "push",
                "-q",
                str(work / "msda_split" / "dec_const"),
                f"{R}/msda_split/",
            ],
            check=True,
        )
    sh = lambda c: subprocess.run([*E.ADB, "shell", c], check=True)  # noqa: E731
    if (
        a.pieces != "msda_split"
    ):  # push_runtime pushed msda_split/; the variant goes next to it
        sh(f"mkdir -p {R}/{a.pieces}")
        for p in X.PIECES:
            subprocess.run(
                [
                    *E.ADB,
                    "push",
                    "-q",
                    str(work / a.pieces / f"{p}.sim.onnx"),
                    f"{R}/{a.pieces}/{p}.onnx",
                ],
                check=True,
            )
    subprocess.run(
        [*E.ADB, "push", "-q", str(Path(a.build) / "frame_run"), f"{R}/frame_run"],
        check=True,
    )
    subprocess.run(
        [
            *E.ADB,
            "push",
            "-q",
            str(work / "msda_split" / "feats_q.txt"),
            f"{R}/{a.pieces}/feats_q.txt",
        ],
        check=True,
    )
    # a phone / host output directory per configuration: runs of different configs can't collide
    fr = "fr_" + f"{a.backbone}_{a.decoder}".replace(".", "_")
    tg = ""
    for flag, path, name in (
        ("TG_ENC", a.tg_enc, "tg_enc.tghx"),
        ("TG_DEC", a.tg_dec, "tg_dec.tghx"),
    ):
        if path:
            subprocess.run([*E.ADB, "push", "-q", path, f"{R}/{name}"], check=True)
            tg += f"{flag}={name} "
            fr += "_" + flag.lower()
    if a.pieces != "msda_split":
        fr += "_" + a.pieces.replace(".", "_")
    sh(
        f"chmod 755 {R}/frame_run && rm -rf {R}/{fr} && mkdir -p {R}/{fr}/seq {R}/{fr}/pipe"
    )
    subprocess.run(
        [*E.ADB, "push", "-q", *[str(d) for d in frames], f"{R}/{fr}/"], check=True
    )
    env = (
        f"cd {R} && {tg}LD_LIBRARY_PATH={R} QNN_PERF=burst "
        f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
    )
    names = " ".join(f"{fr}/{d.name}" for d in frames)
    for mode in a.modes.split(","):
        out = subprocess.run(
            [
                *E.ADB,
                "shell",
                f"{env} ./frame_run {a.pieces} {a.backbone}.onnx {'split' if split_dec else a.decoder + '.onnx'} "
                f"{mode} 1 {a.reps} "
                f"{fr}/{mode} {names} 2>&1",
            ],
            capture_output=True,
            text=True,
        ).stdout
        print(
            f"== {mode}\n"
            + "\n".join(
                ln
                for ln in out.splitlines()
                if not ln.startswith(("2026", " ")) or "ms" in ln
            )
        )
        if "PASS" not in out:
            raise SystemExit(f"frame_run {mode} failed")
        if mode in ("seq", "pipe"):
            loc = root / fr / mode
            loc.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [*E.ADB, "pull", "-q", f"{R}/{fr}/{mode}/.", str(loc)], check=True
            )
            hits = {"cpu": 0, "htp": 0}
            for i, gt in enumerate(gts):
                bev_h = np.fromfile(loc / f"f{i}_bev.f32", np.float32).reshape(
                    M.NQ, M.EMBED
                )
                cls_h = np.fromfile(loc / f"f{i}_cls.f32", np.float32).reshape(900, 10)
                bbox_h = np.fromfile(loc / f"f{i}_bbox.f32", np.float32).reshape(
                    900, 10
                )
                for k, (c, b) in {
                    "cpu": cpu[i][1:],
                    "htp": (torch.from_numpy(cls_h), torch.from_numpy(bbox_h)),
                }.items():
                    boxes, scores, labels = M.decode(c, b)
                    hits[k] += match(boxes, scores, labels, gt)[0]
                print(
                    f"  frame {i}: bev cos {E.cos(cpu[i][0], bev_h):.6f} cls cos {E.cos(cpu[i][1], cls_h):.6f}"
                )
            print(
                f"  GT matched: phone {hits['htp']} / fp32 {hits['cpu']} (of {sum(len(g) for g in gts)})"
            )


if __name__ == "__main__":
    main()
