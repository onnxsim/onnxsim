"""Tests for onnxsim.fusion_error (certified error of an exact-in-reals rewrite).

The pair is Conv followed by BatchNormalization (original) and the same Conv with the folded
weights ``w * gamma / sqrt(var + eps)`` and bias ``(b - mean) * gamma / sqrt(var + eps) + beta``
stored in fp32 (fused). Containment is checked against onnxruntime executions of both graphs.
"""

import numpy as np
import pytest
from onnx import numpy_helper, parser

from onnxsim import fusion_error

_EPS = 1e-5


def _conv_bn(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((3, 2, 3, 3)).astype(np.float32) * 0.5
    b = rng.standard_normal(3).astype(np.float32) * 0.1
    gamma = rng.uniform(0.5, 1.5, 3).astype(np.float32)
    beta = rng.standard_normal(3).astype(np.float32) * 0.1
    mean = rng.standard_normal(3).astype(np.float32) * 0.1
    var = rng.uniform(0.5, 2.0, 3).astype(np.float32)

    original = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 15]>
        agraph (float[1, 2, 5, 5] x) => (float[1, 3, 5, 5] y) {
            c = Conv<kernel_shape=[3, 3], pads=[1, 1, 1, 1]>(x, W, B)
            y = BatchNormalization<epsilon=1e-5>(c, gamma, beta, mean, var)
        }
        """
    )
    for name, val in {
        "W": w,
        "B": b,
        "gamma": gamma,
        "beta": beta,
        "mean": mean,
        "var": var,
    }.items():
        original.graph.initializer.append(numpy_helper.from_array(val, name))

    scale = gamma.astype(np.float64) / np.sqrt(var.astype(np.float64) + _EPS)
    w_fused = (w.astype(np.float64) * scale[:, None, None, None]).astype(np.float32)
    b_fused = ((b.astype(np.float64) - mean) * scale + beta).astype(np.float32)
    fused = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 15]>
        agraph (float[1, 2, 5, 5] x) => (float[1, 3, 5, 5] y) {
            y = Conv<kernel_shape=[3, 3], pads=[1, 1, 1, 1]>(x, W2, B2)
        }
        """
    )
    for name, val in {"W2": w_fused, "B2": b_fused}.items():
        fused.graph.initializer.append(numpy_helper.from_array(val, name))
    return original, fused


def _box():
    return {"x": (-1.0, 1.0)}


def test_conv_bn_fusion_bound_is_finite_and_nonzero():
    original, fused = _conv_bn()
    res = fusion_error.fusion_error_bound(original, fused, _box(), precision="fp32")
    out = res["outputs"]["y"]
    assert np.isfinite(out["absolute"]) and out["absolute"] > 0.0
    assert np.isfinite(out["ulps"]) and out["ulps"] > 0.0
    assert res["worst_absolute"] == out["absolute"]
    assert out["absolute"] == pytest.approx(
        out["real_difference"] + out["roundoff_original"] + out["roundoff_fused"]
    )


def test_bound_contains_onnxruntime_difference():
    ort = pytest.importorskip("onnxruntime")
    original, fused = _conv_bn()
    bound = fusion_error.fusion_error_bound(original, fused, _box())["outputs"]["y"][
        "absolute"
    ]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    run_orig = ort.InferenceSession(
        original.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    run_fused = ort.InferenceSession(
        fused.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        x = rng.uniform(-1.0, 1.0, size=(1, 2, 5, 5)).astype(np.float32)
        y0 = run_orig.run(None, {"x": x})[0].astype(np.float64)
        y1 = run_fused.run(None, {"x": x})[0].astype(np.float64)
        worst = max(worst, float(np.max(np.abs(y0 - y1))))
    assert worst <= bound


def test_unbounded_input_gives_infinite_bound():
    original, fused = _conv_bn()
    out = fusion_error.fusion_error_bound(original, fused, None)["outputs"]["y"]
    assert out["absolute"] == float("inf")
    assert out["ulps"] == float("inf")


def test_output_mismatch_is_rejected():
    original, fused = _conv_bn()
    fused.graph.output[0].name = "z"
    with pytest.raises(ValueError, match="outputs differ"):
        fusion_error.fusion_error_bound(original, fused, _box())


def test_unknown_precision_is_rejected():
    original, fused = _conv_bn()
    with pytest.raises(ValueError, match="fp8"):
        fusion_error.fusion_error_bound(original, fused, _box(), precision="fp8")
