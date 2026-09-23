#!/usr/bin/env python3
"""The FPN output conv as the producer RoiAlign actually wants: hex_conv3x3_kernel.py's vrmpybusv
3x3 conv (unchanged reduction, same `ow_tile` output-column reuse) with the backbone's
requantize + dequantize epilogue fused into its store, writing the fp32 feature map straight into
either layout:

  * `layout="hwc"` -- channels-last (H, W, C), what roialign_fast/roialign_kernel.h reads. Each
    accumulator vector is 32 channels of one pixel, so this is a contiguous 128-byte store.
  * `layout="chw"` -- NCHW, what the backbone emits today and ONNX Runtime's RoiAlign reads. The
    same 32 lanes land H*W floats apart, so this is 32 scattered scalar stores.

Epilogue (per lane, matching ONNX Runtime's QLinearConv + DequantizeLinear bit for bit; see
fpn_channels_last/capture_fpn_out.py): `t = acc + bias` (bias has the input zero point folded
in), `q = clamp(round_half_even(float(t) * M) + zp, 0, 255)`, `out = float(q - zp) * s_out`. It
runs as scalar IEEE fp32 on the DSP's scalar core (vectorization explicitly disabled): HVX fp32 on
this phone is qf32, which is not IEEE-rounded, and one-ulp drift would lose bit-exactness vs ORT.

The input must be pre-padded with the input's real zero point (not 0) -- with the zero point folded
into the bias, a zero-point border contributes exactly nothing, matching QLinearConv's padding.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hex_conv3x3_kernel import pack_weight_3x3  # noqa: E402


def pad_input_zp(a_hwc: np.ndarray, zp: int) -> np.ndarray:
    return np.pad(a_hwc, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=zp)


def ow_tile_for(iw: int) -> int:
    return next(t for t in (8, 4, 2, 1) if iw % t == 0)


def build_kernel(cin, cout, ih, iw, a_pad, wp, bias, mult, zp, s_out, layout, kernel_name, ow_tile=None):
    """a_pad: flat (ih+2)*(iw+2)*cin uint8 (zero-point padded). wp: flat pack_weight_3x3() output.
    bias: (cout,) int32 (zero point folded). Returns a flat ih*iw*cout fp32 Tensor in `layout`."""
    from tinygrad import Tensor, UOp
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import AxisType, KernelInfo, Ops

    assert layout in ("hwc", "chw") and cin % 4 == 0 and cout % 32 == 0
    ow_tile = ow_tile or ow_tile_for(iw)
    assert iw % ow_tile == 0
    nt_count, kc_count = cout // 32, cin // 4
    r_count, owt_count, iw_pad, hw = 9 * kc_count, iw // ow_tile, iw + 2, ih * iw
    i32x32 = "int __attribute__((vector_size(128)))"
    u8x128 = "unsigned char __attribute__((vector_size(128)))"
    m_lit, s_lit = float(np.float32(mult)).hex() + "f", float(np.float32(s_out)).hex() + "f"
    ostr = 1 if layout == "hwc" else hw

    def _reg_i32(shape, slot, *deps):
        ret = UOp.placeholder(shape, dtypes.int32, slot=slot, addrspace=AddrSpace.REG)
        return ret.after((ret.after(*deps) if deps else ret).store(ret.const_like(0)))

    def kernel_fn(C: UOp, A: UOp, Wp: UOp, B: UOp) -> UOp:
        oh_rng = UOp.range(ih, 0, AxisType.WEAK)
        owt_rng = UOp.range(owt_count, 1, AxisType.WEAK)
        nt_rng = UOp.range(nt_count, 2, AxisType.WEAK)
        accs = [_reg_i32((32,), t, oh_rng, owt_rng, nt_rng) for t in range(ow_tile)]
        r_rng = UOp.range(r_count, 3, AxisType.REDUCE)
        acc_addrs = [acc.after(r_rng)[0] for acc in accs]
        pos, kc = r_rng // kc_count, r_rng % kc_count
        kh, kw = pos // 3, pos % 3
        w_idx = Wp[pos * (nt_count * kc_count * 128) + nt_rng * (kc_count * 128) + kc * 128]
        a_idxs = [A[(oh_rng + kh) * (iw_pad * cin) + (owt_rng * ow_tile + t + kw) * cin + kc * 4] for t in range(ow_tile)]
        acc_lines = []
        for t in range(ow_tile):
            broadcast = ",".join([f"*(unsigned int*){{{ow_tile + 1 + t}}}"] * 32)
            acc_lines.append(f"*({i32x32}*){{{t}}} = __builtin_HEXAGON_V6_vrmpybusv_acc_128B("
                             f"*({i32x32}*){{{t}}}, ({i32x32}){{{{{broadcast}}}}}, __wv);")
        step = UOp(Ops.CUSTOM, dtypes.void, (*acc_addrs, w_idx, *a_idxs),
                   arg=f"{u8x128} __wv = *({u8x128}*){{{ow_tile}}}; " + " ".join(acc_lines))
        update = step.end(r_rng)
        final_addrs = [acc.after(update)[0] for acc in accs]

        # One CUSTOM for all ow_tile stores (see hex_conv3x3_kernel.py for why not one per column).
        pix = [oh_rng * iw + (owt_rng * ow_tile + t) for t in range(ow_tile)]
        outs = [C[p * cout + nt_rng * 32] if layout == "hwc" else C[nt_rng * (32 * hw) + p] for p in pix]
        b_ptr = B[nt_rng * 32]
        # operands: {0..T-1} = output pointers, {T..2T-1} = accumulators, {2T} = bias pointer
        # Plain C text with @OUT@/@ACC@/@BIAS@ markers; braces are then escaped for the renderer's
        # str.format() and the markers become its positional operand fields.
        # The accumulator is copied into an explicitly 128-byte-aligned local before any scalar
        # lane reads: the REG placeholder renders as a plain `int bufN[32]` (4-byte aligned) that
        # the reduction reads/writes through 128-byte vector casts, and mixing scalar lane loads
        # into that same buffer crashed qemu-hexagon outright (a copy via the same vector cast the
        # plain store path already uses keeps every scalar read on well-defined, aligned storage).
        tmpl = (f"{{ int __attribute__((aligned(128))) __a[32]; *({i32x32}*)__a = *({i32x32}*)@ACC@; "
                "const int* __b = (const int*)@BIAS@; float* __o = (float*)@OUT@; "
                "_Pragma(\"clang loop vectorize(disable) interleave(disable)\") "
                "for (int __l = 0; __l < 32; __l++) { "
                f"float __x = (float)(__a[__l] + __b[__l]) * {m_lit}; "
                "__x = (__x + 12582912.0f) - 12582912.0f; "  # round half to even (valid for |x| < 2^22)
                f"int __q = (int)__x + {zp}; __q = __q < 0 ? 0 : (__q > 255 ? 255 : __q); "
                f"__o[__l * {ostr}] = (float)(__q - {zp}) * {s_lit}; }} }}")
        tmpl = tmpl.replace("{", "{{").replace("}", "}}")
        lines = [tmpl.replace("@OUT@", f"{{{t}}}").replace("@ACC@", f"{{{ow_tile + t}}}").replace("@BIAS@", f"{{{2 * ow_tile}}}")
                 for t in range(ow_tile)]
        out_step = UOp(Ops.CUSTOM, dtypes.void, (*outs, *final_addrs, b_ptr), arg=" ".join(lines))
        return out_step.end(nt_rng, owt_rng, oh_rng).sink(arg=KernelInfo(name=kernel_name, opts_to_apply=()))

    c = Tensor.empty(ih * iw * cout, dtype="float32", device="DSP")
    return Tensor.custom_kernel(c, a_pad, wp, bias, fxn=functools.partial(kernel_fn))[0]


def reference(a_chw, zp_in, w, bias, mult, zp, s_out):
    """numpy: (H, W, C) fp32, the exact semantics above (border padded with the input zero point)."""
    c, h, wd = a_chw.shape
    a = np.pad(a_chw.transpose(1, 2, 0).astype(np.int64), ((1, 1), (1, 1), (0, 0)), constant_values=zp_in)
    acc = np.zeros((h, wd, w.shape[0]), np.int64)
    for kh in range(3):
        for kw in range(3):
            acc += a[kh:kh + h, kw:kw + wd] @ w[:, :, kh, kw].astype(np.int64).T
    x = (acc + bias).astype(np.float32) * np.float32(mult)
    q = np.clip(np.rint(x) + zp, 0, 255)
    return ((q - zp).astype(np.float32) * np.float32(s_out)).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="fpn_out_real.npz from fpn_channels_last/capture_fpn_out.py")
    p.add_argument("--levels", default="P5,P4,P3,P2")
    p.add_argument("--verify", default="P5,P4,P3,P2", help="levels to also execute under MOCKDSP=1 (qemu) and check")
    p.add_argument("--out", type=Path, default=Path("fpn_out_kernels.c"))
    args = p.parse_args()
    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")
    from tinygrad import Tensor
    from tinygrad.renderer.cstyle import ClangRenderer

    d = np.load(args.data)
    srcs = []
    orig = ClangRenderer.render

    def cap(self, uops):
        s = orig(self, uops)
        mk = s.find("/* DSP boilerplate */")
        srcs.append((s[:mk] if mk >= 0 else s).rstrip() + "\n")
        return s

    ClangRenderer.render = cap
    try:
        for lvl in args.levels.split(","):
            a_chw, w, bias = d[f"{lvl}_in"], d[f"{lvl}_w"], d[f"{lvl}_bias"]
            mult, s_out = d[f"{lvl}_scalars"]
            zp_in, zp_out = (int(v) for v in d[f"{lvl}_zps"])
            cin, ih, iw = a_chw.shape
            cout = w.shape[0]
            run = lvl in args.verify.split(",")
            a_pad = pad_input_zp(a_chw.transpose(1, 2, 0), zp_in).reshape(-1)
            wp = pack_weight_3x3(w).reshape(-1)
            outs = {}
            for layout in ("hwc", "chw"):
                name = f"fpnout_{lvl}_{layout}"
                t = build_kernel(cin, cout, ih, iw, Tensor(a_pad, device="DSP"), Tensor(wp, device="DSP"),
                                 Tensor(bias, device="DSP"), mult, zp_out, s_out, layout, name)
                if run:
                    outs[layout] = t.numpy()
                else:  # render + compile only: a full-scale P3/P2 conv under qemu is too slow to be worth it
                    from tinygrad.runtime import ops_dsp
                    call = ops_dsp.MockDSPProgram.__call__
                    ops_dsp.MockDSPProgram.__call__ = lambda self, *a, **k: 0.0
                    try:
                        t.realize()
                    finally:
                        ops_dsp.MockDSPProgram.__call__ = call
            if run:
                hwc = outs["hwc"].reshape(ih, iw, cout)
                chw = outs["chw"].reshape(cout, ih, iw)
                ref = reference(a_chw, zp_in, w, bias, mult, zp_out, s_out)
                ort_f = d[f"{lvl}_ort_f"].transpose(1, 2, 0)
                print(f"{lvl} qemu: hwc == transpose(chw): {np.array_equal(hwc, chw.transpose(1, 2, 0))}; "
                      f"hwc == numpy ref: {np.array_equal(hwc, ref)}; hwc == ORT's real fp32 output: {np.array_equal(hwc, ort_f)}")
    finally:
        ClangRenderer.render = orig
        args.out.write_text("".join(srcs))
    print(f"wrote {args.out} ({len(srcs)} kernels)")


if __name__ == "__main__":
    main()
