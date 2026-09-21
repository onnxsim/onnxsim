#!/usr/bin/env python3
"""Run Mask R-CNN-shaped TVM kernels on a connected Hexagon Android DSP.

This is an opt-in hardware probe, not part of the normal test suite. It takes
the static ResNet/FPN tensor shapes and pooling attributes from a Mask R-CNN
ONNX model, generates representative convolution, transpose convolution,
pooling, resize, RoIAlign, and quantize/dequantize kernels with TVM's Hexagon
target, runs them through TVM RPC, and compares against TVM/LLVM CPU or NumPy.
Weights, activations, and regions are randomized: this checks kernel codegen
and DSP execution, not end-to-end model accuracy or graph delegation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import setuptools  # noqa: F401  # TVM's Hexagon helpers expect it at import time.
import tvm
from tvm import te, topi
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.rpc.tracker import Tracker
from tvm.topi.testing import roi_align_nchw_python


def _attrs(node):
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def _shape_map(model):
    result = {}
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in values:
        result[value.name] = tuple(
            dim.dim_value if dim.dim_value else None
            for dim in value.type.tensor_type.shape.dim
        )
    for value in model.graph.initializer:
        result[value.name] = tuple(value.dims)
    return result


def _find_conv(model, shape_map, data_shape, weight_shape):
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        actual_data = shape_map.get(node.input[0])
        batch_is_dynamic = (
            actual_data is not None
            and len(actual_data) == 4
            and actual_data[0] is None
            and actual_data[1:] == data_shape[1:]
        )
        if (actual_data == data_shape or batch_is_dynamic) and shape_map.get(node.input[1]) == weight_shape:
            return node
    raise RuntimeError(f"No Conv found with data={data_shape} and weight={weight_shape}")


def _find_pool(model, shape_map, data_shape):
    for node in model.graph.node:
        if node.op_type == "MaxPool" and shape_map.get(node.input[0]) == data_shape:
            return node
    raise RuntimeError(f"No MaxPool found with input={data_shape}")


def _model_workloads(model_path: Path, roi_batch: int):
    model = onnx.load(str(model_path))
    shape_overrides = {}
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 3 and dims[0].dim_value == 3:
            shape_overrides[value.name] = (3, 224, 224)
            break
    if not shape_overrides:
        raise RuntimeError("Expected a three-dimensional [3, height, width] image input")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    # Shape inference preserves symbolic image dimensions in this fixture;
    # set the sample size and infer once more to expose the backbone shapes.
    for value in model.graph.input:
        if value.name in shape_overrides:
            for dim, size in zip(value.type.tensor_type.shape.dim, shape_overrides[value.name]):
                dim.dim_value = size
                dim.ClearField("dim_param")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = _shape_map(model)

    conv_specs = [
        ("resnet_bottleneck_1x1", (1, 64, 56, 56), (64, 64, 1, 1)),
        ("resnet_bottleneck_3x3", (1, 64, 56, 56), (64, 64, 3, 3)),
        ("fpn_3x3", (1, 256, 14, 14), (256, 256, 3, 3)),
        ("roi_mask_head_3x3", (roi_batch, 256, 14, 14), (256, 256, 3, 3)),
    ]
    convs = []
    for name, data_shape, weight_shape in conv_specs:
        node = _find_conv(model, shapes, data_shape, weight_shape)
        attrs = _attrs(node)
        pads = attrs.get("pads", [0, 0, 0, 0])
        convs.append(
            (
                name,
                data_shape,
                weight_shape,
                tuple(attrs.get("strides", [1, 1])),
                (pads[0], pads[1]),
                (pads[2], pads[3]),
            )
        )

    pools = []
    for name, data_shape in [
        ("backbone_stem_maxpool", (1, 64, 112, 112)),
        ("fpn_maxpool", (1, 256, 7, 7)),
    ]:
        node = _find_pool(model, shapes, data_shape)
        attrs = _attrs(node)
        pools.append(
            (
                name,
                data_shape,
                tuple(attrs["kernel_shape"]),
                tuple(attrs.get("strides", [1, 1])),
                tuple(attrs.get("pads", [0, 0, 0, 0])),
            )
        )

    resizes = []
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        input_shape = shapes.get(node.input[0])
        if (
            input_shape is None
            or len(input_shape) != 4
            or input_shape[1] != 256
            or input_shape[2] not in (7, 14, 28)
            or input_shape[3] != input_shape[2]
        ):
            continue
        attrs = _attrs(node)
        if attrs.get("mode", b"nearest") != b"nearest":
            continue
        size = (input_shape[2] * 2, input_shape[3] * 2)
        resizes.append(
            (
                f"fpn_resize_{input_shape[2]}_to_{size[0]}",
                (1, 256, input_shape[2], input_shape[3]),
                size,
                attrs.get("coordinate_transformation_mode", b"half_pixel").decode(),
                attrs.get("nearest_mode", b"round_prefer_floor").decode(),
            )
        )
    if not resizes:
        raise RuntimeError("No nearest-neighbor FPN Resize operators found")

    roi_aligns = {}
    for node in model.graph.node:
        if node.op_type != "RoiAlign":
            continue
        input_shape = shapes.get(node.input[0])
        if input_shape is None or len(input_shape) != 4 or input_shape[1:] not in (
            (256, 7, 7),
            (256, 14, 14),
            (256, 28, 28),
            (256, 56, 56),
        ):
            continue
        attrs = _attrs(node)
        if attrs.get("output_height") != 7 or attrs.get("output_width") != 7:
            continue
        height = input_shape[2]
        roi_aligns.setdefault(
            height,
            (
                f"roi_align_{height}x{height}_to_7x7",
                (1, 256, height, height),
                (7, 7),
                attrs["spatial_scale"],
                attrs.get("sampling_ratio", 0),
                attrs.get("mode", b"avg").decode(),
            ),
        )
    if len(roi_aligns) != 4:
        raise RuntimeError(f"Expected four FPN RoiAlign inputs, found {sorted(roi_aligns)}")
    deconv_node = next((node for node in model.graph.node if node.op_type == "ConvTranspose"), None)
    if deconv_node is None:
        raise RuntimeError("No mask-head ConvTranspose operator found")
    deconv_attrs = _attrs(deconv_node)
    deconv_weight_shape = shapes.get(deconv_node.input[1])
    if deconv_weight_shape != (256, 256, 2, 2):
        raise RuntimeError(f"Unexpected mask-head ConvTranspose weights: {deconv_weight_shape}")
    deconv = (
        "roi_mask_head_conv_transpose_2x",
        (roi_batch, 256, 14, 14),
        deconv_weight_shape,
        tuple(deconv_attrs.get("strides", [1, 1])),
        tuple(deconv_attrs.get("pads", [0, 0, 0, 0])),
        tuple(deconv_attrs.get("output_padding", [0, 0])),
    )

    initializers = {
        value.name: onnx.numpy_helper.to_array(value) for value in model.graph.initializer
    }
    qdq = None
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or len(node.input) < 3:
            continue
        scale = initializers.get(node.input[1])
        zero_point = initializers.get(node.input[2])
        data_shape = shapes.get(node.input[0])
        if (
            scale is None
            or zero_point is None
            or np.asarray(scale).ndim != 0
            or np.asarray(zero_point).dtype != np.uint8
            or data_shape is None
            or any(dim is None or dim <= 0 for dim in data_shape)
        ):
            continue
        matching_dequant = next(
            (
                candidate
                for candidate in model.graph.node
                if candidate.op_type == "DequantizeLinear"
                and candidate.input[0] == node.output[0]
            ),
            None,
        )
        if matching_dequant is None:
            continue
        qdq = (
            "activation_quantize_dequantize_uint8",
            data_shape,
            float(np.asarray(scale)),
            int(np.asarray(zero_point)),
        )
        break
    if qdq is None:
        raise RuntimeError("No static per-tensor uint8 QuantizeLinear/DequantizeLinear pair found")
    return convs, pools, resizes, list(roi_aligns.values()), deconv, qdq


def _hexagon_target():
    target = tvm.target.hexagon("v73")
    return tvm.target.Target(target, host=target)


def _conv_module(name, data_shape, weight_shape, stride, pad_before, pad_after, target):
    x = te.placeholder(data_shape, name="x", dtype="float32")
    weight = te.placeholder(weight_shape, name="weight", dtype="float32")
    bias = te.placeholder((weight_shape[0],), name="bias", dtype="float32")
    padding = tuple(pad_before) + tuple(pad_after)
    conv = topi.nn.conv2d_nchw(x, weight, stride, padding, (1, 1), "float32")
    biased = te.compute(
        conv.shape,
        lambda n, c, h, w: conv[n, c, h, w] + bias[c],
        name="bias_add",
    )
    output = te.compute(
        conv.shape,
        lambda n, c, h, w: te.max(biased[n, c, h, w], 0.0),
        name="relu",
    )
    schedule = topi.hexagon.schedule_conv2d(output, layout="NCHW")
    # Width is contiguous in NCHW. Small width vectors leave more independent
    # outer tiles for the Hexagon worker pool than a single 32-wide vector.
    n, channel, height, width = schedule[conv].op.axis
    width_outer, width_inner = schedule[conv].split(width, factor=8)
    outer = schedule[conv].fuse(n, channel, height, width_outer)
    schedule[conv].reorder(outer, width_inner, *schedule[conv].op.reduce_axis)
    schedule[conv].vectorize(width_inner)
    schedule[conv].parallel(outer)
    module = tvm.build(schedule, [x, weight, bias, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _pool_module(data_shape, kernel, stride, pads, target):
    x = te.placeholder(data_shape, name="x", dtype="float32")
    output = topi.nn.pool2d(
        x,
        kernel,
        stride,
        (1, 1),
        pads,
        "max",
        ceil_mode=False,
        layout="NCHW",
    )
    schedule = te.create_schedule(output.op)
    n, channel, height, width = schedule[output].op.axis
    width_outer, width_inner = schedule[output].split(width, factor=32)
    outer = schedule[output].fuse(n, channel, height, width_outer)
    schedule[output].reorder(outer, *schedule[output].op.reduce_axis, width_inner)
    schedule[output].vectorize(width_inner)
    schedule[output].parallel(outer)
    module = tvm.build(schedule, [x, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _resize_module(data_shape, size, coordinate_mode, rounding_mode, target):
    x = te.placeholder(data_shape, name="x", dtype="float32")
    output = topi.image.resize2d(
        x,
        roi=(0.0, 0.0, 0.0, 0.0),
        size=size,
        layout="NCHW",
        method="nearest_neighbor",
        coordinate_transformation_mode=coordinate_mode,
        rounding_method=rounding_mode,
    )
    schedule = topi.hexagon.schedule_injective(output)
    module = tvm.build(schedule, [x, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _roi_align_module(data_shape, rois_shape, pooled_size, spatial_scale, sample_ratio, mode, target):
    data = te.placeholder(data_shape, name="data", dtype="float32")
    rois = te.placeholder(rois_shape, name="rois", dtype="float32")
    output = topi.vision.roi_align_nchw(
        data, rois, pooled_size, spatial_scale, mode.encode(), sample_ratio
    )
    # TVM 0.17 has no registered Hexagon Relay schedule for RoiAlign. Its TOPI
    # compute is TE-based, so use Hexagon's injective schedule to vectorize and
    # parallelize the independent output elements.
    schedule = topi.hexagon.schedule_injective(output)
    module = tvm.build(schedule, [data, rois, output], target=target, name="main")
    return module, tuple(int(dim) for dim in output.shape)


def _conv_transpose_module(data_shape, weight_shape, stride, pads, output_padding, target):
    data = te.placeholder(data_shape, name="data", dtype="float32")
    weight = te.placeholder(weight_shape, name="weight", dtype="float32")
    bias = te.placeholder((weight_shape[1],), name="bias", dtype="float32")
    output = topi.nn.conv2d_transpose_nchw(
        data,
        weight,
        stride,
        pads,
        "float32",
        output_padding,
    )
    biased = te.compute(
        output.shape,
        lambda n, c, h, w: output[n, c, h, w] + bias[c],
        name="bias_add",
    )
    schedule = topi.hexagon.schedule_conv2d_transpose_nchw(biased)
    n, channel, height, width = schedule[output].op.axis
    width_outer, width_inner = schedule[output].split(width, factor=8)
    outer = schedule[output].fuse(n, channel, height, width_outer)
    schedule[output].reorder(outer, width_inner, *schedule[output].op.reduce_axis)
    schedule[output].vectorize(width_inner)
    schedule[output].parallel(outer)
    module = tvm.build(schedule, [data, weight, bias, biased], target=target, name="main")
    return module, tuple(int(dim) for dim in biased.shape)


def _qdq_module(data_shape, scale, zero_point, target):
    data = te.placeholder(data_shape, name="data", dtype="float32")
    quantized = te.compute(
        data_shape,
        lambda *idx: te.min(
            te.max(tvm.tir.round(data[idx] / scale) + zero_point, 0.0),
            255.0,
        ).astype("uint8"),
        name="quantize_linear",
    )
    dequantized = te.compute(
        data_shape,
        lambda *idx: (quantized[idx].astype("float32") - zero_point) * scale,
        name="dequantize_linear",
    )
    schedule = topi.hexagon.schedule_injective([quantized, dequantized])
    module = tvm.build(schedule, [data, quantized, dequantized], target=target, name="main")
    return module


def _cpu_conv(data, weight, bias, stride, pad_before, pad_after):
    data_shape, weight_shape = data.shape, weight.shape
    x = te.placeholder(data_shape, name="x", dtype="float32")
    w = te.placeholder(weight_shape, name="weight", dtype="float32")
    b = te.placeholder((weight_shape[0],), name="bias", dtype="float32")
    conv = topi.nn.conv2d_nchw(
        x, w, stride, tuple(pad_before) + tuple(pad_after), (1, 1), "float32"
    )
    biased = te.compute(
        conv.shape,
        lambda n, c, h, wi: conv[n, c, h, wi] + b[c],
        name="bias_add",
    )
    output = te.compute(
        conv.shape,
        lambda n, c, h, wi: te.max(biased[n, c, h, wi], 0.0),
        name="relu",
    )
    schedule = topi.hexagon.schedule_conv2d(output, layout="NCHW")
    _, width_inner = schedule[conv].split(conv.op.axis[3], factor=32)
    schedule[conv].vectorize(width_inner)
    module = tvm.build(schedule, [x, w, b, output], target="llvm", name="main")
    result = tvm.nd.empty(tuple(int(dim) for dim in output.shape))
    module["main"](tvm.nd.array(data), tvm.nd.array(weight), tvm.nd.array(bias), result)
    return result.numpy()


def run(args):
    convs, pools, resizes, roi_aligns, deconv, qdq = _model_workloads(args.model, args.roi_batch)
    target = _hexagon_target()
    rng = np.random.default_rng(11)
    tracker = Tracker(host=args.rpc_host, port=args.tracker_port)
    launcher = HexagonLauncher(
        args.serial,
        rpc_info={
            "rpc_tracker_host": args.rpc_host,
            "rpc_tracker_port": args.tracker_port,
            "rpc_server_port": args.server_port,
            "workspace_base": args.device_workspace,
            "adb_server_socket": None,
        },
    )
    try:
        launcher.start_server()
        with launcher.create_session() as session:
            for name, data_shape, weight_shape, stride, pad_before, pad_after in convs:
                module, output_shape = _conv_module(
                    name, data_shape, weight_shape, stride, pad_before, pad_after, target
                )
                # The TVM RPC session accepts a module through its shared object
                # export; save keeps the Hexagon ELF standalone and avoids a host link.
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(0, 0.1, data_shape).astype("float32")
                weight = rng.normal(0, 0.05, weight_shape).astype("float32")
                bias = np.zeros((weight_shape[0],), dtype="float32")
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](
                    tvm.nd.array(data, session.device),
                    tvm.nd.array(weight, session.device),
                    tvm.nd.array(bias, session.device),
                    remote_output,
                )
                actual = remote_output.numpy()
                expected = _cpu_conv(data, weight, bias, stride, pad_before, pad_after)
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
                error = float(np.max(np.abs(actual - expected)))
                print(f"PASS {name}: output={output_shape} max_abs_err={error:.8g}", flush=True)

            for name, data_shape, kernel, stride, pads in pools:
                module, output_shape = _pool_module(data_shape, kernel, stride, pads, target)
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(size=data_shape).astype("float32")
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](tvm.nd.array(data, session.device), remote_output)
                actual = remote_output.numpy()

                x = te.placeholder(data_shape, name="x", dtype="float32")
                output = topi.nn.pool2d(
                    x, kernel, stride, (1, 1), pads, "max", ceil_mode=False, layout="NCHW"
                )
                schedule = topi.hexagon.schedule_pool(output, layout="NCHW")
                cpu_module = tvm.build(schedule, [x, output], target="llvm", name="main")
                expected = tvm.nd.empty(output_shape)
                cpu_module["main"](tvm.nd.array(data), expected)
                np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=0)
                print(f"PASS {name}: output={output_shape} exact", flush=True)

            for name, data_shape, size, coordinate_mode, rounding_mode in resizes:
                module, output_shape = _resize_module(
                    data_shape, size, coordinate_mode, rounding_mode, target
                )
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(size=data_shape).astype("float32")
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](tvm.nd.array(data, session.device), remote_output)
                actual = remote_output.numpy()

                x = te.placeholder(data_shape, name="x", dtype="float32")
                output = topi.image.resize2d(
                    x,
                    roi=(0.0, 0.0, 0.0, 0.0),
                    size=size,
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode=coordinate_mode,
                    rounding_method=rounding_mode,
                )
                cpu_module = tvm.build(
                    te.create_schedule(output.op), [x, output], target="llvm", name="main"
                )
                expected = tvm.nd.empty(output_shape)
                cpu_module["main"](tvm.nd.array(data), expected)
                np.testing.assert_allclose(actual, expected.numpy(), rtol=0, atol=0)
                print(f"PASS {name}: output={output_shape} exact", flush=True)

            for name, data_shape, pooled_size, spatial_scale, sample_ratio, mode in roi_aligns:
                rois_shape = (args.roi_batch, 5)
                module, output_shape = _roi_align_module(
                    data_shape,
                    rois_shape,
                    pooled_size,
                    spatial_scale,
                    sample_ratio,
                    mode,
                    target,
                )
                local_path = Path(args.artifact_dir) / f"{name}.so"
                module.save(str(local_path))
                remote_path = session.upload(str(local_path), local_path.name)
                remote_module = session.load_module(remote_path)
                data = rng.normal(0, 0.1, data_shape).astype("float32")
                rois = np.zeros(rois_shape, dtype="float32")
                rois[:, 1] = np.arange(args.roi_batch, dtype="float32") * 3.0
                rois[:, 2] = np.arange(args.roi_batch, dtype="float32") * 2.0
                rois[:, 3] = np.minimum(rois[:, 1] + 112.0, 223.0)
                rois[:, 4] = np.minimum(rois[:, 2] + 96.0, 223.0)
                remote_output = tvm.nd.empty(output_shape, "float32", session.device)
                remote_module["main"](
                    tvm.nd.array(data, session.device),
                    tvm.nd.array(rois, session.device),
                    remote_output,
                )
                actual = remote_output.numpy()
                expected = roi_align_nchw_python(
                    data,
                    rois,
                    pooled_size,
                    spatial_scale,
                    sample_ratio,
                    mode=mode.encode(),
                )
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                error = float(np.max(np.abs(actual - expected)))
                print(
                    f"PASS {name}: output={output_shape} max_abs_err={error:.8g}", flush=True
                )

            name, data_shape, weight_shape, stride, pads, output_padding = deconv
            module, output_shape = _conv_transpose_module(
                data_shape, weight_shape, stride, pads, output_padding, target
            )
            local_path = Path(args.artifact_dir) / f"{name}.so"
            module.save(str(local_path))
            remote_path = session.upload(str(local_path), local_path.name)
            remote_module = session.load_module(remote_path)
            data = rng.normal(0, 0.1, data_shape).astype("float32")
            weight = rng.normal(0, 0.05, weight_shape).astype("float32")
            bias = np.zeros((weight_shape[1],), dtype="float32")
            remote_output = tvm.nd.empty(output_shape, "float32", session.device)
            remote_module["main"](
                tvm.nd.array(data, session.device),
                tvm.nd.array(weight, session.device),
                tvm.nd.array(bias, session.device),
                remote_output,
            )
            actual = remote_output.numpy()

            x = te.placeholder(data_shape, name="data", dtype="float32")
            w = te.placeholder(weight_shape, name="weight", dtype="float32")
            b = te.placeholder((weight_shape[1],), name="bias", dtype="float32")
            cpu_conv = topi.nn.conv2d_transpose_nchw(
                x, w, stride, pads, "float32", output_padding
            )
            expected_expr = te.compute(
                cpu_conv.shape,
                lambda n, c, h, wi: cpu_conv[n, c, h, wi] + b[c],
                name="bias_add",
            )
            cpu_schedule = topi.hexagon.schedule_conv2d_transpose_nchw(expected_expr)
            cpu_module = tvm.build(
                cpu_schedule, [x, w, b, expected_expr], target="llvm", name="main"
            )
            expected = tvm.nd.empty(output_shape)
            cpu_module["main"](tvm.nd.array(data), tvm.nd.array(weight), tvm.nd.array(bias), expected)
            np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-3, atol=2e-3)
            error = float(np.max(np.abs(actual - expected.numpy())))
            print(f"PASS {name}: output={output_shape} max_abs_err={error:.8g}", flush=True)

            name, data_shape, scale, zero_point = qdq
            qdq_module = _qdq_module(data_shape, scale, zero_point, target)
            qdq_path = Path(args.artifact_dir) / f"{name}.so"
            qdq_module.save(str(qdq_path))
            remote_path = session.upload(str(qdq_path), qdq_path.name)
            remote_qdq = session.load_module(remote_path)
            data = rng.normal(0, 1.0, data_shape).astype("float32")
            q_output = tvm.nd.empty(data_shape, "uint8", session.device)
            dq_output = tvm.nd.empty(data_shape, "float32", session.device)
            remote_qdq["main"](tvm.nd.array(data, session.device), q_output, dq_output)
            actual_q = q_output.numpy()
            actual_dq = dq_output.numpy()
            expected_q = np.clip(np.rint(data / scale) + zero_point, 0, 255).astype("uint8")
            expected_dq = (expected_q.astype("float32") - zero_point) * scale
            np.testing.assert_array_equal(actual_q, expected_q)
            dq_max_abs_err = float(np.max(np.abs(actual_dq - expected_dq)))
            np.testing.assert_allclose(actual_dq, expected_dq, rtol=1e-6, atol=1e-5)
            print(
                f"PASS {name}: shape={data_shape} quantized_exact "
                f"dequantized_max_abs_err={dq_max_abs_err:.7g}",
                flush=True,
            )
    finally:
        launcher.stop_server()
        tracker.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Mask R-CNN ONNX model")
    parser.add_argument("--serial", required=True, help="adb device serial")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--tracker-port", type=int, default=9190)
    parser.add_argument("--server-port", type=int, default=7070)
    parser.add_argument("--device-workspace", default="/data/local/tmp/tvm_hexagon")
    parser.add_argument("--artifact-dir", type=Path, default=Path("/tmp/tvm-maskrcnn"))
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
