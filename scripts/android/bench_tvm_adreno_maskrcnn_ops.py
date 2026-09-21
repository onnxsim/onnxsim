#!/usr/bin/env python3
"""Mask R-CNN operator timings from TVM's OpenCL code generator on an Adreno GPU.

GPU counterpart of `bench_tvm_hexagon_maskrcnn_ops.py` / `..._more_ops.py`: the same operator
shapes, but compiled with TVM's `opencl -device=adreno` codegen and scheduled automatically with
`tvm.dlight` (no hand-written GPU schedules). Kernels run over RPC on the phone's OpenCL device
and are checked against a host-CPU TVM build of the same TE compute.

Setup (see docs/tvm-hexagon-conv-transpose-handoff.md): a TVM Android runtime built with
`-DUSE_OPENCL=ON -DUSE_CPP_RPC=ON`, pushed to the phone and started with

    ./tvm_rpc server --host=0.0.0.0 --port=9190 --port-end=9199 --key=adreno

plus `adb forward tcp:9190 tcp:9190` and `TVM_NDK_CC` pointing at the NDK's aarch64 clang.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import setuptools  # noqa: F401  # TVM 0.17 imports distutils during module initialization.
import tvm
from tvm import dlight, rpc, te, topi
from tvm.contrib import ndk

ALL_OPS = (
    "conv,pool,resize,roi,deconv,qdq,add,relu,add_relu,matmul,softmax,sigmoid,box_decode,"
    "level_mapper,filter,gather,concat,split,transpose,flatten,reducemin"
)


def _gpu_target(fp16: bool):
    attrs = "opencl -device=adreno"
    return tvm.target.Target(attrs, host="llvm -mtriple=aarch64-linux-android")


def _placeholder(shape, name, dtype="float32"):
    return te.placeholder(shape, name=name, dtype=dtype)


def _random_boxes(rng, count, extent=224.0):
    xy = rng.uniform(0, extent - 40, (count, 2))
    wh = rng.uniform(8, 140, (count, 2))
    return np.concatenate([xy, np.minimum(xy + wh, extent)], axis=1).astype("float32")


def _cpu_reference(args, outputs, arrays):
    schedule = te.create_schedule([out.op for out in outputs])
    module = tvm.build(schedule, list(args) + list(outputs), target="llvm", name="ref")
    cpu = tvm.cpu(0)
    inputs = [tvm.nd.array(array, cpu) for array in arrays]
    results = [tvm.nd.empty([int(d) for d in out.shape], out.dtype, cpu) for out in outputs]
    module(*inputs, *results)
    return [result.numpy() for result in results]


def _gpu_build(args, outputs, target):
    func = te.create_prim_func(list(args) + list(outputs))
    module = tvm.IRModule({"main": func})
    rule_sets = [
        (
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.LowBatchGEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Transpose(),
            dlight.gpu.Fallback(),
        ),
        (dlight.gpu.Fallback(),),  # a specialised rule can reject shapes such as 1x1 conv
    ]
    last_error = None
    for rules in rule_sets:
        try:
            with target:
                scheduled = dlight.ApplyDefaultSchedule(*rules)(module)
            return tvm.build(scheduled, target=target)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise last_error


# ---- workloads: each returns (name, args, outputs, arrays, rtol) ----


def _bias_relu(conv, bias):
    return te.compute(conv.shape, lambda n, c, h, w: te.max(conv[n, c, h, w] + bias[c], 0.0))


def _fc(x, w, bias, k, m, batch, relu):
    rk = te.reduce_axis((0, k), name="rk")
    acc = te.compute((batch, m), lambda n, j: te.sum(x[n, rk] * w[rk, j], axis=rk))
    if relu:
        return te.compute((batch, m), lambda n, j: te.max(acc[n, j] + bias[j], 0.0))
    return te.compute((batch, m), lambda n, j: acc[n, j] + bias[j])


def _conv(rng, roi_batch):
    cases = [
        ("conv3x3_relu_1x64x56x56", (1, 64, 56, 56), (64, 64, 3, 3), 1),
        ("conv1x1_relu_1x64x56x56", (1, 64, 56, 56), (64, 64, 1, 1), 0),
        ("conv3x3_relu_1x256x14x14", (1, 256, 14, 14), (256, 256, 3, 3), 1),
        (f"conv3x3_relu_{roi_batch}x256x14x14", (roi_batch, 256, 14, 14), (256, 256, 3, 3), 1),
    ]
    workloads = []
    for name, data_shape, weight_shape, pad in cases:
        x = _placeholder(data_shape, "x")
        weight = _placeholder(weight_shape, "weight")
        bias = _placeholder((weight_shape[0],), "bias")
        conv = topi.nn.conv2d_nchw(x, weight, (1, 1), (pad, pad, pad, pad), (1, 1), "float32")
        out = _bias_relu(conv, bias)
        arrays = [
            rng.normal(size=data_shape).astype("float32"),
            rng.normal(0, 0.05, weight_shape).astype("float32"),
            rng.normal(0, 0.1, (weight_shape[0],)).astype("float32"),
        ]
        workloads.append((name, [x, weight, bias], [out], arrays, 2e-3))
    return workloads


def _pool(rng, roi_batch):
    shape = (1, 64, 112, 112)
    x = _placeholder(shape, "x")
    out = topi.nn.pool2d(x, (3, 3), (2, 2), (1, 1), (1, 1, 1, 1), "max", layout="NCHW")
    return [("maxpool_1x64x112x112", [x], [out], [rng.normal(size=shape).astype("float32")], 0)]


def _resize(rng, roi_batch):
    shape = (1, 256, 14, 14)
    x = _placeholder(shape, "x")
    out = topi.image.resize2d(
        x,
        roi=(0.0, 0.0, 0.0, 0.0),
        size=(28, 28),
        layout="NCHW",
        method="nearest_neighbor",
        coordinate_transformation_mode="half_pixel",
        rounding_method="round_prefer_floor",
    )
    return [("resize_1x256x14x14_to_28", [x], [out], [rng.normal(size=shape).astype("float32")], 0)]


def _roi(rng, roi_batch):
    shape = (1, 256, 56, 56)
    data = _placeholder(shape, "data")
    rois = _placeholder((roi_batch, 5), "rois")
    out = topi.vision.roi_align_nchw(data, rois, (7, 7), 0.25, b"avg", 2)  # model: sampling_ratio=2
    roi_array = np.zeros((roi_batch, 5), dtype="float32")
    roi_array[:, 1] = np.arange(roi_batch) * 3.0
    roi_array[:, 2] = np.arange(roi_batch) * 2.0
    roi_array[:, 3] = np.minimum(roi_array[:, 1] + 112.0, 223.0)
    roi_array[:, 4] = np.minimum(roi_array[:, 2] + 96.0, 223.0)
    arrays = [rng.normal(0, 0.1, shape).astype("float32"), roi_array]
    return [(f"roi_align_56x56_to_7x7_x{roi_batch}", [data, rois], [out], arrays, 2e-5)]


def _deconv(rng, roi_batch):
    data_shape, weight_shape = (roi_batch, 256, 14, 14), (256, 256, 2, 2)
    data = _placeholder(data_shape, "data")
    weight = _placeholder(weight_shape, "weight")
    bias = _placeholder((256,), "bias")
    deconv = topi.nn.conv2d_transpose_nchw(data, weight, (2, 2), (0, 0), "float32", (0, 0))
    out = te.compute(deconv.shape, lambda n, c, h, w: deconv[n, c, h, w] + bias[c])
    arrays = [
        rng.normal(0, 0.1, data_shape).astype("float32"),
        rng.normal(0, 0.05, weight_shape).astype("float32"),
        np.zeros((256,), dtype="float32"),
    ]
    # Same 2x2/stride-2 parity formulation as the Hexagon specialization: no zero-dilated input.
    rc = te.reduce_axis((0, 256), name="rc")
    parity = te.compute(
        (roi_batch, 256, 28, 28),
        lambda b, oc, y, x: te.sum(
            data[b, rc, y // 2, x // 2] * weight[rc, oc, y % 2, x % 2] + bias[oc] / 256.0, axis=rc
        ),
        name="conv_transpose_parity",
    )
    return [
        (f"conv_transpose_{roi_batch}x256x14x14", [data, weight, bias], [out], arrays, 2e-3),
        (
            f"conv_transpose_parity_{roi_batch}x256x14x14",
            [data, weight, bias],
            [parity],
            arrays,
            2e-3,
        ),
    ]


def _qdq(rng, roi_batch):
    shape, scale, zero_point = (1, 3, 224, 224), 0.02, 114
    data = _placeholder(shape, "data")
    quantized = te.compute(
        shape,
        lambda *idx: te.min(te.max(tvm.tir.round(data[idx] / scale) + zero_point, 0.0), 255.0).astype(
            "uint8"
        ),
        name="quantize_linear",
    )
    dequantized = te.compute(
        shape,
        lambda *idx: (quantized[idx].astype("float32") - zero_point) * scale,
        name="dequantize_linear",
    )
    return [("quantize_dequantize_1x3x224x224", [data], [quantized, dequantized],
             [rng.normal(0, 1.0, shape).astype("float32")], 1e-5)]


def _eltwise(kind):
    def build(rng, roi_batch):
        workloads = []
        for shape in [(1, 256, 56, 56), (1, 2048, 7, 7)]:
            a, b = _placeholder(shape, "a"), _placeholder(shape, "b")
            if kind == "add":
                out, args = topi.add(a, b), [a, b]
            elif kind == "add_relu":
                out, args = topi.nn.relu(topi.add(a, b)), [a, b]
            else:
                out, args = topi.nn.relu(a), [a]
            arrays = [rng.normal(size=shape).astype("float32") for _ in args]
            workloads.append((f"{kind}_{'x'.join(map(str, shape))}", args, [out], arrays, 1e-6))
        return workloads

    return build


def _matmul(rng, roi_batch):
    workloads = []
    for index, (k, m) in enumerate([(12544, 1024), (1024, 1024), (1024, 324), (1024, 81)]):
        x, w, bias = _placeholder((roi_batch, k), "x"), _placeholder((k, m), "w"), _placeholder((m,), "b")
        relu = index < 2
        out = _fc(x, w, bias, k, m, roi_batch, relu)
        arrays = [
            rng.normal(0, 0.1, (roi_batch, k)).astype("float32"),
            rng.normal(0, 0.02, (k, m)).astype("float32"),
            rng.normal(0, 0.1, (m,)).astype("float32"),
        ]
        workloads.append((f"matmul_{roi_batch}x{k}x{m}{'_relu' if relu else ''}", [x, w, bias], [out], arrays, 2e-3))
    return workloads


def _softmax(rng, roi_batch):
    x = _placeholder((roi_batch, 81), "x")
    return [(f"softmax_{roi_batch}x81", [x], [topi.nn.softmax(x, axis=1)],
             [rng.normal(0, 2, (roi_batch, 81)).astype("float32")], 1e-4)]


def _sigmoid(rng, roi_batch):
    shape = (roi_batch, 81, 28, 28)
    x = _placeholder(shape, "x")
    return [(f"sigmoid_{'x'.join(map(str, shape))}", [x], [topi.sigmoid(x)],
             [rng.normal(0, 3, shape).astype("float32")], 1e-4)]


def _box_decode(rng, roi_batch, count=1000):
    clip = float(np.log(1000.0 / 16))
    deltas, anchors = _placeholder((count, 4), "deltas"), _placeholder((count, 4), "anchors")

    def decode(k, j):
        width, height = anchors[k, 2] - anchors[k, 0], anchors[k, 3] - anchors[k, 1]
        cx, cy = anchors[k, 0] + width * 0.5, anchors[k, 1] + height * 0.5
        pcx, pcy = deltas[k, 0] / 10.0 * width + cx, deltas[k, 1] / 10.0 * height + cy
        pw = te.exp(te.min(deltas[k, 2] / 5.0, clip)) * width
        ph = te.exp(te.min(deltas[k, 3] / 5.0, clip)) * height
        return tvm.tir.Select(
            j == 0, pcx - pw * 0.5,
            tvm.tir.Select(j == 1, pcy - ph * 0.5, tvm.tir.Select(j == 2, pcx + pw * 0.5, pcy + ph * 0.5)),
        )

    out = te.compute((count, 4), decode, name="decoded")
    arrays = [rng.normal(0, 1.0, (count, 4)).astype("float32"), _random_boxes(rng, count)]
    return [(f"box_decode_{count}", [deltas, anchors], [out], arrays, 1e-4)]


def _level_mapper(rng, roi_batch, count=1000):
    boxes = _placeholder((count, 4), "boxes")

    def level(k):
        area = (boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1])
        value = te.floor(4.0 + te.log(te.sqrt(area) / 224.0 + 1e-6) / float(np.log(2.0)))
        return te.max(te.min(value, 5.0), 2.0)

    out = te.compute((count,), level, name="level")
    return [(f"level_mapper_{count}", [boxes], [out], [_random_boxes(rng, count)], 0)]


def _filter(rng, roi_batch, count=9408):
    scores, boxes = _placeholder((count,), "scores"), _placeholder((count, 4), "boxes")

    def keep(k):
        wide, tall = (boxes[k, 2] - boxes[k, 0]) >= 1e-3, (boxes[k, 3] - boxes[k, 1]) >= 1e-3
        return tvm.tir.Cast("uint8", tvm.tir.all(scores[k] > 0.05, wide, tall))

    out = te.compute((count,), keep, name="keep")
    return [(f"score_size_filter_{count}", [scores, boxes], [out],
             [rng.uniform(0, 1, count).astype("float32"), _random_boxes(rng, count)], 0)]


def _gather(rng, roi_batch, total=9408, count=1000):
    data, indices = _placeholder((total, 4), "boxes"), _placeholder((count,), "indices", "int64")
    out = topi.take(data, indices, axis=0)
    arrays = [_random_boxes(rng, total), rng.integers(0, total, count).astype("int64")]
    return [(f"gather_rows_{total}_by_{count}", [data, indices], [out], arrays, 0)]


def _concat(rng, roi_batch):
    sizes = [3 * side * side for side in (56, 28, 14, 7, 4)]
    parts = [_placeholder((size, 4), f"level{i}") for i, size in enumerate(sizes)]
    arrays = [rng.normal(size=(size, 4)).astype("float32") for size in sizes]
    return [(f"concat_rpn_levels_{sum(sizes)}x4", parts, [topi.concatenate(parts, axis=0)], arrays, 0)]


def _split(rng, roi_batch, count=1000):
    boxes = _placeholder((count, 4), "boxes")
    outs = [topi.strided_slice(boxes, [0, i], [count, i + 1], [1, 1]) for i in range(4)]
    return [(f"split_boxes_{count}", [boxes], outs, [_random_boxes(rng, count)], 0)]


def _transpose(rng, roi_batch, count=1000):
    x = _placeholder((1, count), "x")
    return [(f"transpose_1x{count}", [x], [topi.transpose(x, (1, 0))],
             [rng.normal(size=(1, count)).astype("float32")], 0)]


def _flatten(rng, roi_batch):
    shape = (roi_batch, 256, 7, 7)
    x = _placeholder(shape, "x")
    return [(f"flatten_{'x'.join(map(str, shape))}", [x], [topi.reshape(x, (roi_batch, 12544))],
             [rng.normal(size=shape).astype("float32")], 0)]


def _reducemin(rng, roi_batch, count=1000):
    x = _placeholder((count,), "x")
    return [(f"reducemin_{count}", [x], [topi.min(x, axis=0, keepdims=True)],
             [rng.normal(size=count).astype("float32")], 0)]


BUILDERS = {
    "conv": _conv, "pool": _pool, "resize": _resize, "roi": _roi, "deconv": _deconv,
    "qdq": _qdq, "add": _eltwise("add"), "relu": _eltwise("relu"), "add_relu": _eltwise("add_relu"),
    "matmul": _matmul, "softmax": _softmax, "sigmoid": _sigmoid, "box_decode": _box_decode,
    "level_mapper": _level_mapper, "filter": _filter, "gather": _gather, "concat": _concat,
    "split": _split, "transpose": _transpose, "flatten": _flatten, "reducemin": _reducemin,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--key", default="adreno")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--ops", default=ALL_OPS)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    target = _gpu_target(False)
    remote = rpc.connect(args.host, args.port, key=args.key)
    dev = remote.cl(0)
    print(f"device: {dev.device_name}", flush=True)
    rng = np.random.default_rng(61)
    for op in args.ops.split(","):
        for name, inputs, outputs, arrays, rtol in BUILDERS[op](rng, args.roi_batch):
            try:
                expected = _cpu_reference(inputs, outputs, arrays)
                library = _gpu_build(inputs, outputs, target)
                path = Path("/tmp") / f"tvm_adreno_{name}.so"
                library.export_library(str(path), fcompile=ndk.create_shared)
                remote.upload(str(path))
                module = remote.load_module(path.name)
                nd_in = [tvm.nd.array(array, dev) for array in arrays]
                nd_out = [tvm.nd.empty([int(d) for d in out.shape], out.dtype, dev) for out in outputs]
                module["main"](*nd_in, *nd_out)
                for got, want in zip(nd_out, expected):
                    if rtol:
                        np.testing.assert_allclose(got.numpy(), want, rtol=rtol, atol=rtol)
                    else:
                        np.testing.assert_array_equal(got.numpy(), want)
                single = module.time_evaluator("main", dev, number=1, repeat=args.repeat)(*nd_in, *nd_out)
                batched = module.time_evaluator("main", dev, number=20, repeat=3)(*nd_in, *nd_out)
                print(
                    f"{name}: median={np.median(single.results) * 1e3:.3f} ms, "
                    f"pipelined={np.median(batched.results) * 1e3:.3f} ms, correctness=PASS",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"{name}: FAILED ({type(error).__name__}: {str(error).splitlines()[-1][:170]})", flush=True)


if __name__ == "__main__":
    main()
