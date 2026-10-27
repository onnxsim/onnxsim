#!/usr/bin/env python3
"""BEVFormer-tiny's encoder / decoder exported with every deformable-attention span as one `onnxsim::MSDA` node.

Each "sampling grid -> GridSample -> x attention weight -> sum over points (-> average over maps)" span (split.py's
`msda_fused`, the hand kernel's contract in ../../../msda_hvx/msda_kernel.h, one level, mode MSDA_REF_PIX) becomes

  MSDA(value (NV, H*W, 256), ref (NVR, Q, R, 2), off (Q, M, NO, P, 2), attw (Q, M, NO, P)[, vis (NV, Q) uint8]) -> (Q, 256)
  attributes H, W (ints)

so a compiler can map the whole span to the hand-written kernel: the tinygrad fork's ONNX importer does (nn/onnx.py MSDA:
the hand kernel as one DSP call, the same math through GridSample elsewhere), behind onnx-remote's compile_v69.sh. Everything
else in the graph is the plain export of model.py. onnxruntime has no MSDA, so the check here is torch-only: the exported
pieces' torch modules (split.py's encoder pieces with msda_fused, the decoder with msda_fused in its cross-attention) against
model.py's own forward on the saved frames. Compare a device run with onnxruntime on the matching plain export
(<piece>.sim.onnx from export.py), same inputs.

usage: export_msda_op.py enc1|enc3|decoder --ckpt <pth> --work <work>   -> <work>/<piece>.msda.onnx (onnxsim-simplified)
       export_msda_op.py frame --work <work>   -> <work>/frame.msda.onnx: enc3.msda.onnx and decoder.msda.onnx (both exported
       first) as one graph, bev_embed kept as an output -- one program for frame_run's TG_ENC (two separate programs don't fit
       the DSP process together)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[2] / "msda_hvx")]
import model as M  # noqa: E402
import split  # noqa: E402


class MSDAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, ref, off, attw, vis, H, W):
        return split.msda_fused(value, (H, W), ref, off, attw, vis)

    @staticmethod
    def symbolic(g, value, ref, off, attw, vis, H, W):
        ins = [value, ref, off, attw] + (
            [] if vis is None or vis.node().mustBeNone() else [vis]
        )
        out = g.op("onnxsim::MSDA", *ins, H_i=H, W_i=W)
        q, c = off.type().sizes()[0], value.type().sizes()[2]
        out.setType(value.type().with_sizes([q, c]))
        return out


_calls = [0]


def sampler(value, hw, ref, off, attw, vis=None):
    # MSDA_ONLY=i,j,...: only those calls (in graph order: the encoder's TSA, SCA, TSA, ... / the decoder's layers) become MSDA
    # nodes, the rest stay the plain GridSample span -- for bisecting a device result call by call
    k, _calls[0] = _calls[0], _calls[0] + 1
    only = os.environ.get("MSDA_ONLY")
    if only and str(k) not in only.split(","):
        return split.msda_fused(value, hw, ref, off, attw, vis)
    return MSDAFn.apply(value, ref, off, attw, vis, int(hw[0]), int(hw[1]))


class EncoderMSDA(nn.Module):
    """model.Encoder's inputs and output, run as split.run_split with the MSDA node as its sampler."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask):
        rec: list = []
        out = split.run_split(
            self.enc,
            (feats, prev_bev, has_prev, shift, can_bus, ref_cam, bev_mask),
            sampler=sampler,
            record=rec if os.environ.get("DEBUG_LAYERS") else None,
        )
        # DEBUG_LAYERS=1: every sampler call's output as an extra graph output too, for locating a wrong device result
        return (out, *[r[k][1] for r in rec for k in ("tsa", "sca")]) if rec else out


def decoder_msda_forward(self, query, query_pos, value, ref, reference=False):
    """model.DecoderMSDA.forward with its sampling as one MSDA node: loc = ref + off / (W, H) is MSDA_REF_PIX with the raw
    offsets in pixels."""
    m, p = M.HEADS, self.p
    q = query + query_pos
    v = self.value_proj(value).reshape(1, M.NQ, M.EMBED)
    off = self.sampling_offsets(q).reshape(M.NUM_QUERY, m, 1, p, 2)
    w = torch.softmax(self.attention_weights(q).reshape(M.NUM_QUERY, m, p), -1).reshape(
        M.NUM_QUERY, m, 1, p
    )
    out = sampler(v, (M.BEV_H, M.BEV_W), ref.reshape(1, M.NUM_QUERY, 1, 2), off, w)
    return self.output_proj(out) + query


def worst(a, b):
    a, b = a.numpy().ravel().astype(np.float64), b.numpy().ravel().astype(np.float64)
    return float(np.abs(a - b).max()), float(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    )


def merge_frame(work: Path) -> Path:
    import onnx
    from onnx import compose

    e, d = (
        onnx.load(str(work / "enc3.msda.onnx")),
        onnx.load(str(work / "decoder.msda.onnx")),
    )
    for m in (
        e,
        d,
    ):  # (onnxsim's own metadata differs between the two, which merge_models refuses)
        del m.metadata_props[:]
    d = compose.add_prefix(d, "dec_", rename_inputs=False, rename_outputs=False)
    m = compose.merge_models(e, d, io_map=[("bev_embed", "bev_embed")])
    m.graph.output.insert(0, next(o for o in e.graph.output if o.name == "bev_embed"))
    out = work / "frame.msda.onnx"
    onnx.save(m, str(out))
    print(f"{out}: outputs {[o.name for o in m.graph.output]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("piece")
    ap.add_argument("--ckpt")
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    if a.piece == "frame":
        merge_frame(Path(a.work))
        return
    if not a.ckpt:
        raise SystemExit("--ckpt is required")
    torch.set_grad_enabled(False)
    work = Path(a.work)
    frames = [
        torch.load(p, weights_only=False)
        for p in sorted((work / "frames").glob("*.pt"))
    ]
    _, enc, dec = M.load_official(a.ckpt)
    if a.piece.startswith("enc"):
        enc.layers = enc.layers[: int(a.piece[3:])]
        stock, mod = enc, EncoderMSDA(enc)
        in_names = [
            "feats",
            "prev_bev",
            "has_prev",
            "shift",
            "can_bus",
            "ref_cam",
            "bev_mask",
        ]  # export.py's ENC_NAMES
        out_names, inputs_of = ["bev_embed"], lambda f: tuple(f["enc_in"])
    elif a.piece == "decoder":
        refs = [
            dec(f["bev"]) for f in frames
        ]  # the stock decoder, before its cross-attention is swapped
        for layer in dec.layers:
            layer.cross_attn.forward = decoder_msda_forward.__get__(layer.cross_attn)
        stock, mod = None, dec
        in_names, out_names, inputs_of = (
            ["bev_embed"],
            ["cls_scores", "bbox_preds"],
            lambda f: (f["bev"],),
        )
    else:
        raise SystemExit(f"unknown piece {a.piece}")
    # torch check: the MSDA formulation against model.py's own forward
    for i, f in enumerate(frames):
        x = inputs_of(f)
        want = stock(*x) if stock is not None else refs[i]
        got = mod(*x)
        want, got = (
            (want if isinstance(want, tuple) else (want,)),
            (got if isinstance(got, tuple) else (got,)),
        )
        print(
            f"frame {i}: "
            + ", ".join(
                f"{n} max abs {d:.2e} cos {c:.7f}"
                for n, (d, c) in zip(out_names, map(worst, want, got))
            )
        )
    x = inputs_of(frames[min(1, len(frames) - 1)])
    _calls[0] = 0  # count the export's own trace only
    out = work / (
        f"{a.piece}.msda.onnx"
        if not os.environ.get("MSDA_ONLY")
        else f"{a.piece}.msda{os.environ['MSDA_ONLY'].replace(',', '_')}.onnx"
    )
    if os.environ.get("DEBUG_LAYERS"):
        out = out.with_suffix(".dbg.onnx")
        out_names = out_names + [
            f"{k}{i}"
            for i in range(len(getattr(mod, "enc", mod).layers))
            for k in ("tsa", "sca")
        ]
    torch.onnx.export(
        mod,
        x,
        str(out),
        input_names=in_names,
        output_names=out_names,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        custom_opsets={"onnxsim": 1},
    )
    import onnx

    import onnxsim

    # simplified like export.py's pieces: the raw export computes reshape targets with Shape -> Gather -> Concat chains, which
    # a compiler that captures the graph without running it (onnx-remote's compile_v69.sh) can't evaluate
    m, _ = onnxsim.simplify(onnx.load(str(out)), check_n=0)
    onnx.save(m, str(out))
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(
        f"{out}: {len(m.graph.node)} nodes, "
        + " ".join(
            f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:12]
        )
    )


if __name__ == "__main__":
    main()
