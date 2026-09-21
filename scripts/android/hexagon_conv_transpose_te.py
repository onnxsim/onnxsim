"""Plain-TE mask-head ConvTranspose kernels for Hexagon (no hand-written intrinsics).

Shared by `bench_tvm_hexagon_conv_transpose_generated.py` (phone) and the simulator tests. The
kernels are ordinary TE computes with a parity-plane accumulator; `hexagon_qfloat.build` turns
the vectorized multiply-accumulate chains into HVX qfloat intrinsic chains.

Modes: `f32` (fp32, fp32 accumulate), `f16w` (fp16 in/out, fp32 accumulate via widening
hf x hf -> qf32) and `f16k` (fp16 in/out, qf16 partial sums of `chunk` channels widened into an
fp32 total).
"""

from __future__ import annotations

from hexagon_qfloat import build
from tvm import te


def _placeholders(shape_info, dtype):
    n, ic, h, w, oc = shape_info
    data = te.placeholder((n, h, w, ic), name="data", dtype=dtype)
    weight = te.placeholder((2, 2, ic, oc), name="weight", dtype=dtype)
    bias = te.placeholder((oc,), name="bias", dtype=dtype)
    return data, weight, bias


def conv_transpose_module(
    mode, shape_info, target, vectors, pixel_block, unroll, chunk, use_pass, name="main", parallel=True
):
    """Plain TE 2x2/stride-2 ConvTranspose (NHWC, packed weights) for `mode`."""
    n, ic, h, w, oc = shape_info
    fp16 = mode != "f32"
    lanes = 64 if fp16 else 32
    dtype = "float16" if fp16 else "float32"
    data, weight, bias = _placeholders(shape_info, dtype)
    acc_dtype = "float32" if mode in ("f32", "f16w", "f16k") else dtype

    shape = (n, h * 2, w * 2, oc)
    plane = (n, h, w, 2, 2, oc)  # [batch, row, col, kernel row, kernel col, out channel]

    def term(idx, r):
        b, y, x, py, px, c = idx
        if fp16 and mode == "f16w":  # widen the operands, not the product
            return data[b, y, x, r].astype(acc_dtype) * weight[py, px, r, c].astype(acc_dtype)
        return data[b, y, x, r] * weight[py, px, r, c]

    if mode == "f16k":
        # Two-level reduction: qf16 partial sums over `chunk` channels, fp32 total.
        outer = te.reduce_axis((0, ic // chunk), name="rco")
        inner = te.reduce_axis((0, chunk), name="rci")
        partial = te.compute(
            (*plane, ic // chunk),
            lambda *i: te.sum(term(i[:6], i[6] * chunk + inner), axis=inner),
            name="partial",
        )
        acc = te.compute(
            plane,
            lambda *i: te.sum(partial[(*i, outer)].astype("float32"), axis=outer),
            name="acc",
        )
    else:
        reduce = te.reduce_axis((0, ic), name="rc")
        acc = te.compute(plane, lambda *i: te.sum(term(i, reduce), axis=reduce), name="acc")
    # Interleave the four parity planes into the NHWC output (bias added here).
    out = te.compute(
        (n, h * 2, w * 2, oc),
        lambda b, y, x, c: (
            acc[b, y // 2, x // 2, y % 2, x % 2, c] + bias[c].astype(acc_dtype)
        ).astype(dtype),
        name="conv_transpose",
    )

    s = te.create_schedule(out.op)
    b_ax, y_ax, x_ax, c_ax = s[out].op.axis
    yo, yp = s[out].split(y_ax, factor=2)
    xo, xp = s[out].split(x_ax, factor=2)
    xoo, xoi = s[out].split(xo, factor=pixel_block)
    co, ci = s[out].split(c_ax, factor=lanes * vectors)
    # Both x parities sit inside the accumulator tile, so one splatted activation feeds both.
    s[out].reorder(b_ax, yo, yp, xoo, co, xoi, xp, ci)
    fused = s[out].fuse(b_ax, yo)
    if parallel:
        s[out].parallel(fused)
    ci_o, ci_i = s[out].split(ci, factor=lanes)
    s[out].unroll(xoi)
    s[out].unroll(xp)
    s[out].unroll(ci_o)
    s[out].vectorize(ci_i)
    s[acc].compute_at(s[out], co)

    ab, ay, ax, apy, apx, ac = s[acc].op.axis
    aco, aci = s[acc].split(ac, factor=lanes)
    spatial = [ab, ay, ax, apy, apx, aco, aci]
    if mode == "f16k":
        (rco,) = s[acc].op.reduce_axis
        s[acc].reorder(rco, *spatial)
        s[partial].compute_at(s[acc], rco)
        p_axes = list(s[partial].op.axis)
        (rci,) = s[partial].op.reduce_axis
        pco, pci = s[partial].split(p_axes[5], factor=lanes)
        s[partial].reorder(p_axes[6], rci, *p_axes[:5], pco, pci)
        for axis in (rci, p_axes[2], p_axes[4], pco):
            s[partial].unroll(axis)
        s[partial].vectorize(pci)
    else:
        (rc,) = s[acc].op.reduce_axis
        rco, rci = s[acc].split(rc, factor=unroll)
        s[acc].reorder(rco, rci, *spatial)
        s[acc].unroll(rci)
    for axis in (ax, apx, aco):
        s[acc].unroll(axis)
    s[acc].vectorize(aci)

    module = build(s, [data, weight, bias, out], target, name=name, enable_pass=use_pass)
    return module, shape
