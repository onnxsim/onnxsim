#!/usr/bin/env python3
"""fp16 variants of the stride-2 Hexagon ConvTranspose benchmark (mask-head workload).

Compares, on a connected phone, against the fp32 kernels in
`bench_tvm_hexagon_conv_transpose.py`:

* `llvm16`      - fp16 TE compute vectorized by LLVM, fp16 accumulation.
* `llvm16acc32` - fp16 inputs, fp32 accumulation, LLVM codegen.
* `qf16`        - hand-written HVX: `vmpy.qf16.hf` chained with `vadd.qf16`; 64 MACs per
                  vector op, accumulation in qf16. (LLVM 19 lacks the `vmpy.rt.hf` intrinsic.)
* `qf32w`       - hand-written HVX: widening `vmpy.qf32.hf` (hf x hf -> qf32 pair) with
                  `vadd.qf32` accumulation, i.e. fp16 storage and fp32 accumulation.

Inputs, weights and the output are fp16 (NHWC, weights packed to [kh, kw, ic, oc]). Errors are
reported against an fp32 reference computed from the fp16-rounded operands.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import setuptools  # noqa: F401  # TVM 0.17 imports distutils during module initialization.
import tvm
from bench_tvm_hexagon_conv_transpose import (
    _configure_linker,
    _hexagon_target,
    _numpy_stride2_reference,
    _qf32_intrin,
)
from tvm import te
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker

HF_LANES = 64  # fp16 elements per 128-byte HVX vector
I32_LANES = 32


def _llvm_module(shape_info, target, channel_tile, acc32):
    n, in_channels, in_height, in_width, out_channels = shape_info
    out_height, out_width = in_height * 2, in_width * 2
    data = te.placeholder((n, in_height, in_width, in_channels), name="data", dtype="float16")
    weight = te.placeholder((2, 2, in_channels, out_channels), name="weight", dtype="float16")
    bias = te.placeholder((out_channels,), name="bias", dtype="float16")
    rc = te.reduce_axis((0, in_channels), name="rc")
    acc_dtype = "float32" if acc32 else "float16"

    def product(b, y, x, oc):
        term = data[b, y // 2, x // 2, rc] * weight[y % 2, x % 2, rc, oc]
        return term.astype(acc_dtype) if acc32 else term

    acc = te.compute(
        (n, out_height, out_width, out_channels),
        lambda b, y, x, oc: te.sum(product(b, y, x, oc), axis=rc),
        name="acc",
    )
    output = te.compute(
        acc.shape,
        lambda b, y, x, oc: (acc[b, y, x, oc] + bias[oc].astype(acc_dtype)).astype("float16"),
        name="conv_transpose_fp16",
    )
    schedule = te.create_schedule(output.op)
    schedule[acc].compute_at(schedule[output], schedule[output].op.axis[3])
    batch, height, width, channel = schedule[output].op.axis
    channel_outer, channel_inner = schedule[output].split(channel, factor=channel_tile)
    outer = schedule[output].fuse(batch, height, width, channel_outer)
    schedule[output].reorder(outer, channel_inner)
    schedule[output].vectorize(channel_inner)
    schedule[output].parallel(outer)
    return tvm.build(schedule, [data, weight, bias, output], target=target, name="main")


def _hf_module(
    mode, shape_info, target, channel_vectors, pixel_block, unroll, binds_alignment=128
):
    """Hand-written HVX ConvTranspose. `mode` is "qf16", "qf32w" or "qf16k"."""
    n, in_channels, in_height, in_width, out_channels = shape_info
    out_height, out_width = in_height * 2, in_width * 2
    span = HF_LANES * channel_vectors
    assert out_channels % span == 0 and in_width % pixel_block == 0 and in_channels % unroll == 0
    data = te.placeholder((n, in_height, in_width, in_channels), name="data", dtype="float16")
    weight = te.placeholder((2, 2, in_channels, out_channels), name="weight", dtype="float16")
    bias = te.placeholder((out_channels,), name="bias", dtype="float16")
    # qf32w keeps a low/high qf32 pair per accumulator; qf16k adds a short qf16 partial sum.
    acc_slots = {"qf16": 1, "qf32w": 2, "qf16k": 3}[mode]

    def body(ins, outs):
        ib = tvm.tir.ir_builder.create()

        def view(buf, dtype, divisor):
            size = int(np.prod([int(d) for d in buf.shape])) // divisor
            return ib.buffer_ptr(tvm.tir.decl_buffer((size,), dtype, data=buf.data))

        # fp16 buffers are addressed as int16 (scalar activations) or int32 (128-byte vectors).
        data_i16 = view(ins[0], "int16", 1)
        weight_i32, bias_i32 = view(ins[1], "int32", 2), view(ins[2], "int32", 2)
        out_i32 = view(outs[0], "int32", 2)
        acc = ib.allocate(
            "int32", (pixel_block * channel_vectors * acc_slots * I32_LANES,), name="acc", scope="local"
        )
        zero = tvm.tir.Broadcast(tvm.tir.const(0, "int32"), I32_LANES)

        def slot(p, v, half=0):
            base = ((p * channel_vectors + v) * acc_slots + half) * I32_LANES
            return tvm.tir.Ramp(base, 1, I32_LANES)

        def vec32(buf, index):
            return buf[tvm.tir.Ramp(index, 1, I32_LANES)]

        # qf16 1.0 (= hf 1.0 x hf 1.0), used to widen qf16 partial sums to qf32.
        one_hf = _qf32_intrin("int32x32", "lvsplath", tvm.tir.const(0x3C00, "int32"))
        one_qf16 = _qf32_intrin("int32x32", "vmpy.qf16.hf", one_hf, one_hf)
        oc_blocks = out_channels // span
        col_blocks = in_width // pixel_block
        with ib.for_range(0, n * in_height, name="task", kind="parallel") as task:
            batch, row = task // in_height, task % in_height
            for parity_y in range(2):
                for parity_x in range(2):
                    with ib.for_range(0, col_blocks, name="col_block") as col_block:
                        with ib.for_range(0, oc_blocks, name="oc_block") as oc_block:
                            oc0 = oc_block * span  # in fp16 elements
                            pixel0 = col_block * pixel_block
                            for p in range(pixel_block):
                                for v in range(channel_vectors):
                                    for half in range(acc_slots):
                                        acc[slot(p, v, half)] = zero
                            with ib.for_range(0, in_channels // unroll, name="rc_block") as rc_block:
                                for u in range(unroll):
                                    rc = rc_block * unroll + u
                                    weights = []
                                    for v in range(channel_vectors):
                                        w_index = ((parity_y * 2 + parity_x) * in_channels + rc) * (
                                            out_channels // 2
                                        ) + oc0 // 2 + v * I32_LANES
                                        weights.append(vec32(weight_i32, w_index))
                                    for p in range(pixel_block):
                                        d_index = (
                                            (batch * in_height + row) * in_width + pixel0 + p
                                        ) * in_channels + rc
                                        half_bits = tvm.tir.Cast("int32", data_i16[d_index]) & 0xFFFF
                                        rt = half_bits | (half_bits << 16)
                                        splat = _qf32_intrin("int32x32", "lvsplath", rt)
                                        for v in range(channel_vectors):
                                            if mode == "qf16":
                                                prod = _qf32_intrin(
                                                    "int32x32", "vmpy.qf16.hf", splat, weights[v]
                                                )
                                                acc[slot(p, v)] = _qf32_intrin(
                                                    "int32x32", "vadd.qf16", acc[slot(p, v)], prod
                                                )
                                            elif mode == "qf16k":
                                                prod = _qf32_intrin(
                                                    "int32x32", "vmpy.qf16.hf", splat, weights[v]
                                                )
                                                partial = slot(p, v, 2)
                                                acc[partial] = (
                                                    prod
                                                    if u == 0
                                                    else _qf32_intrin(
                                                        "int32x32", "vadd.qf16", acc[partial], prod
                                                    )
                                                )
                                            else:
                                                pair = _qf32_intrin(
                                                    "int32x64", "vmpy.qf32.hf", splat, weights[v]
                                                )
                                                for half, part in enumerate(
                                                    (tvm.tir.op.vectorlow, tvm.tir.op.vectorhigh)
                                                ):
                                                    acc[slot(p, v, half)] = _qf32_intrin(
                                                        "int32x32",
                                                        "vadd.qf32",
                                                        acc[slot(p, v, half)],
                                                        part("int32x32", pair),
                                                    )
                                if mode == "qf16k":
                                    # Widen the short qf16 partial sum (x qf16 1.0) into fp32 totals.
                                    for p in range(pixel_block):
                                        for v in range(channel_vectors):
                                            wide = _qf32_intrin(
                                                "int32x64", "vmpy.qf32.qf16", acc[slot(p, v, 2)], one_qf16
                                            )
                                            for half, part in enumerate(
                                                (tvm.tir.op.vectorlow, tvm.tir.op.vectorhigh)
                                            ):
                                                acc[slot(p, v, half)] = _qf32_intrin(
                                                    "int32x32",
                                                    "vadd.qf32",
                                                    acc[slot(p, v, half)],
                                                    part("int32x32", wide),
                                                )
                            for p in range(pixel_block):
                                out_row = 2 * row + parity_y
                                out_col = 2 * (pixel0 + p) + parity_x
                                for v in range(channel_vectors):
                                    if mode == "qf16":
                                        result = _qf32_intrin(
                                            "int32x32", "vconv.hf.qf16", acc[slot(p, v)]
                                        )
                                    else:
                                        pair = tvm.tir.op.vectorcombine(
                                            "int32x64", acc[slot(p, v, 0)], acc[slot(p, v, 1)]
                                        )
                                        result = _qf32_intrin("int32x32", "vconv.hf.qf32", pair)
                                    biased = _qf32_intrin(
                                        "int32x32",
                                        "vadd.hf",
                                        result,
                                        vec32(bias_i32, oc0 // 2 + v * I32_LANES),
                                    )
                                    o_index = (
                                        ((batch * out_height + out_row) * out_width + out_col)
                                        * (out_channels // 2)
                                        + oc0 // 2
                                        + v * I32_LANES
                                    )
                                    out_i32[tvm.tir.Ramp(o_index, 1, I32_LANES)] = _qf32_intrin(
                                        "int32x32", "vconv.hf.qf16", biased
                                    )
        return ib.get()

    output = te.extern(
        (n, out_height, out_width, out_channels),
        [data, weight, bias],
        body,
        name=f"conv_transpose_{mode}",
        dtype="float16",
    )
    schedule = te.create_schedule(output.op)
    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, name=t.op.name, data_alignment=binds_alignment, offset_factor=1
        )
        for t in (data, weight, bias, output)
    }
    return tvm.build(
        schedule, [data, weight, bias, output], target=target, name="main", binds=binds
    )


def _run(session, module_path, inputs, output_shape, expected, exact32, repeat):
    remote = session.load_module(session.upload(str(module_path), module_path.name))
    device = session.device
    device_inputs = [tvm.nd.array(value, device) for value in inputs]
    output = tvm.nd.empty(output_shape, "float16", device)
    remote["main"](*device_inputs, output)
    result = output.numpy().astype("float32")
    scale = float(np.max(np.abs(expected)))
    error = float(np.max(np.abs(result - expected)))
    error_vs_fp32 = float(np.max(np.abs(result - exact32)))
    times = remote.time_evaluator("main", device, number=1, repeat=repeat)(
        *device_inputs, output
    ).results
    return float(np.median(times) * 1e3), error / scale, error_vs_fp32 / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--modes", default="llvm16,llvm16acc32,qf16,qf32w")
    parser.add_argument("--vectors", default="1,2", help="hf vectors (64 oc each) per tile")
    parser.add_argument("--pixel-blocks", default="1,2")
    parser.add_argument("--unrolls", default="4")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _configure_linker()

    n, ic, oc, size = args.roi_batch, 256, 256, 14
    shape_info = (n, ic, size, size, oc)
    rng = np.random.default_rng(59)
    data = rng.normal(0, 0.1, (n, ic, size, size)).astype("float32")
    weight = rng.normal(0, 0.05, (ic, oc, 2, 2)).astype("float32")
    bias = rng.normal(0, 0.01, (oc,)).astype("float32")
    data16, weight16, bias16 = (a.astype("float16") for a in (data, weight, bias))
    # Reference on the fp16-rounded operands, and on the original fp32 operands.
    expected = _numpy_stride2_reference(
        data16.astype("float32"), weight16.astype("float32"), bias16.astype("float32")
    ).transpose(0, 2, 3, 1)
    exact32 = _numpy_stride2_reference(data, weight, bias).transpose(0, 2, 3, 1)
    inputs = [
        np.ascontiguousarray(data16.transpose(0, 2, 3, 1)),
        np.ascontiguousarray(weight16.transpose(2, 3, 0, 1)),
        bias16,
    ]
    output_shape = (n, size * 2, size * 2, oc)

    target = _hexagon_target()
    tracker = Tracker(host="127.0.0.1", port=9197)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9197,
            "rpc_server_port": 7077,
            "workspace_base": "/data/local/tmp/tvm_hexagon_deconv_fp16",
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        for mode in args.modes.split(","):
            configs = (
                [(v, 1, 1) for v in (1, 2)]
                if mode.startswith("llvm")
                else list(
                    itertools.product(
                        map(int, args.vectors.split(",")),
                        map(int, args.pixel_blocks.split(",")),
                        map(int, args.unrolls.split(",")),
                    )
                )
            )
            for vectors, pixel_block, unroll in configs:
                label = f"{mode} vec={vectors} pix={pixel_block} unroll={unroll}"
                try:
                    if mode.startswith("llvm"):
                        module = _llvm_module(
                            shape_info, target, HF_LANES * vectors, mode == "llvm16acc32"
                        )
                    else:
                        module = _hf_module(mode, shape_info, target, vectors, pixel_block, unroll)
                    path = Path("/tmp") / f"tvm_hexagon_deconv_{mode}_{vectors}_{pixel_block}_{unroll}.so"
                    module.save(str(path))
                    with launcher.create_session() as session:
                        ms, rel, rel32 = _run(
                            session, path, inputs, output_shape, expected, exact32, args.repeat
                        )
                    verdict = "" if rel < 0.05 else "  ** WRONG RESULT **"
                    print(
                        f"{label}: median={ms:.3f} ms, max_err/scale={rel:.2e} "
                        f"(vs fp32 operands {rel32:.2e}){verdict}",
                        flush=True,
                    )
                except Exception as error:  # noqa: BLE001 - keep sweeping
                    print(f"{label}: FAILED ({type(error).__name__}: {str(error).splitlines()[-1][:150]})", flush=True)
    finally:
        launcher.stop_server()
        tracker.terminate()


if __name__ == "__main__":
    main()
