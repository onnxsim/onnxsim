#!/usr/bin/env python3
"""Build the NanoPC-T6 / RK3588 benchmark models and stage them for upload.

Same model shapes and construction as ``build_luckfox_models.py``, retargeted
at ``rk3588`` so the identical graphs can be compared across two Rockchip SoCs.
``onnxsim`` runs before RKNN compilation in both cases, so the benchmark
exercises the same deployment path a user takes for real models.

RK3588 accepts floating-point builds (unlike RV1106), so ``--onnxsim-quantize``
is optional here rather than required.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxsim import simplify

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def _install_onnx_mapping_shim() -> None:
    """Keep RKNN-Toolkit2 2.3.2 compatible with ONNX 1.22+ (see
    ``rknn_backend._ensure_onnx_mapping_shim``)."""
    if hasattr(onnx, "mapping"):
        return
    mapping = types.ModuleType("onnx.mapping")
    table = {
        dtype: onnx.helper.tensor_dtype_to_np_dtype(dtype)
        for dtype in onnx.TensorProto.DataType.values()
        if dtype != onnx.TensorProto.UNDEFINED
    }
    mapping.TENSOR_TYPE_TO_NP_TYPE = table
    mapping.NP_TYPE_TO_TENSOR_TYPE = {value: key for key, value in table.items()}
    onnx.mapping = mapping
    sys.modules["onnx.mapping"] = mapping


_install_onnx_mapping_shim()
from rknn.api import RKNN  # noqa: E402


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32), name=name)


def _model(
    name: str, shape: tuple[int, int, int, int], depthwise: bool
) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [shape[0], 32, 1, 1])
    nodes = []
    initializers = []
    if depthwise:
        w0 = np.zeros((16, 3, 3, 3), np.float32)
        w0[:, :, 1, 1] = 0.25
        nodes.append(
            helper.make_node(
                "Conv",
                ["input", "w0"],
                ["stem"],
                name="stem",
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            )
        )
        initializers.append(_initializer("w0", w0))
        wd = np.zeros((16, 1, 3, 3), np.float32)
        wd[:, :, 1, 1] = 1.0
        nodes.append(
            helper.make_node(
                "Conv",
                ["stem", "wd"],
                ["dw"],
                name="depthwise",
                pads=[1, 1, 1, 1],
                group=16,
            )
        )
        initializers.append(_initializer("wd", wd))
        wp = np.zeros((32, 16, 1, 1), np.float32)
        for i in range(32):
            wp[i, i % 16, 0, 0] = 0.5
        nodes.append(
            helper.make_node("Conv", ["dw", "wp"], ["features"], name="pointwise")
        )
        initializers.append(_initializer("wp", wp))
    else:
        w0 = np.zeros((16, 3, 3, 3), np.float32)
        w0[:, :, 1, 1] = 0.25
        nodes.append(
            helper.make_node(
                "Conv", ["input", "w0"], ["conv"], name="conv", pads=[1, 1, 1, 1]
            )
        )
        initializers.append(_initializer("w0", w0))
        initializers.extend(
            [
                _initializer("bn_scale", np.ones(16)),
                _initializer("bn_bias", np.zeros(16)),
                _initializer("bn_mean", np.zeros(16)),
                _initializer("bn_var", np.ones(16)),
            ]
        )
        nodes.append(
            helper.make_node(
                "BatchNormalization",
                ["conv", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                ["features"],
                name="batchnorm",
                epsilon=1e-5,
            )
        )
        w1 = np.zeros((32, 16, 3, 3), np.float32)
        w1[:, :, 1, 1] = 0.125
        initializers.append(_initializer("w1", w1))
        nodes.append(
            helper.make_node(
                "Conv",
                ["features", "w1"],
                ["features2"],
                name="conv2",
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            )
        )
        nodes.append(helper.make_node("Relu", ["features2"], ["relu"], name="relu"))
        nodes.append(
            helper.make_node("GlobalAveragePool", ["relu"], ["output"], name="gap")
        )
        graph = helper.make_graph(nodes, name, [x], [y], initializer=initializers)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    nodes.append(helper.make_node("Relu", ["features"], ["relu"], name="relu"))
    nodes.append(
        helper.make_node("GlobalAveragePool", ["relu"], ["output"], name="gap")
    )
    graph = helper.make_graph(nodes, name, [x], [y], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _calibration_dataset(
    output_dir: str, stem: str, shape: tuple[int, int, int, int]
) -> str:
    """Write a deterministic calibration set as raw floats, so the dataset does
    not need OpenCV (the Luckfox builder writes PNGs and depends on cv2)."""
    calibration_dir = os.path.join(output_dir, stem + ".calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    paths = []
    for index in range(8):
        rng = np.random.default_rng(1000 + index)
        # Low-contrast samples in the same range a real 8-bit image would give,
        # so the INT8 activation scales stay representative.
        image = rng.uniform(8.0, 8.0 + 24.0 * (index + 1), size=shape).astype(
            np.float32
        )
        path = os.path.join(calibration_dir, f"{index:02d}.npy")
        np.save(path, image)
        paths.append(path)
    dataset = os.path.join(output_dir, stem + ".dataset.txt")
    with open(dataset, "w", encoding="utf-8") as f:
        f.write("\n".join(paths) + "\n")
    return dataset


def _check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"rknn {operation} failed: {ret}")


def _build(
    model: onnx.ModelProto,
    stem: str,
    output_dir: str,
    onnxsim_quantize: bool = False,
    target_platform: str = "rk3588",
) -> None:
    raw_path = os.path.join(output_dir, stem + ".onnx")
    simp_path = os.path.join(output_dir, stem + ".simplified.onnx")
    rknn_path = os.path.join(output_dir, f"{stem}.{target_platform}.rknn")
    onnx.save(model, raw_path)
    simplified, ok = simplify(model, check_n=0)
    if not ok:
        raise RuntimeError(f"onnxsim could not simplify {stem}")
    onnx.save(simplified, simp_path)
    print(
        f"{stem}: raw={len(model.graph.node)} nodes, "
        f"simplified={len(simplified.graph.node)} nodes"
    )

    load_path = simp_path
    if onnxsim_quantize:
        from onnxsim import quantize_static

        q_path = os.path.join(output_dir, stem + ".onnxsim-int8.onnx")
        shape = tuple(
            dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
        )
        calibration_data = [
            {"input": np.full(shape, (index + 1) / 8.0, np.float32)}
            for index in range(8)
        ]
        quantized = quantize_static(
            simplified,
            calibration_data=calibration_data,
            method="minmax",
            full_graph=True,
            op_types_to_exclude=["Relu", "GlobalAveragePool"],
        )
        onnx.save(quantized, q_path)
        load_path = q_path

    rknn = RKNN(verbose=False)
    try:
        config = dict(
            target_platform=target_platform,
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
        )
        if onnxsim_quantize:
            config["optimization_level"] = 3
        _check(rknn.config(**config), "config")
        _check(rknn.load_onnx(model=load_path), "load_onnx")
        if onnxsim_quantize:
            _check(rknn.build(do_quantization=False), "build")
        else:
            shape = tuple(
                dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
            )
            dataset = _calibration_dataset(output_dir, stem, shape)
            _check(rknn.build(do_quantization=True, dataset=dataset), "build")
        _check(rknn.export_rknn(rknn_path), "export_rknn")
    finally:
        rknn.release()
    print(f"{stem}: rknn={os.path.getsize(rknn_path) // 1024} KiB -> {rknn_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="bench/nanopc-rk3588")
    ap.add_argument(
        "--onnxsim-quantize",
        action="store_true",
        help="emit QDQ INT8 ONNX with onnxsim and build it with do_quantization=False",
    )
    ap.add_argument("--target-platform", default="rk3588")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _build(
        _model("conv_bn_relu_224", (1, 3, 224, 224), False),
        "conv_bn_relu_224",
        args.output_dir,
        args.onnxsim_quantize,
        args.target_platform,
    )
    _build(
        _model("depthwise_pointwise_112", (1, 3, 112, 112), True),
        "depthwise_pointwise_112",
        args.output_dir,
        args.onnxsim_quantize,
        args.target_platform,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
