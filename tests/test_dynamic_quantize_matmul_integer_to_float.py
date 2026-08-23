"""Tests for ``onnxsim.quantize_dynamic_matmul_integer_to_float`` (the
``dynamic_quantize_matmul_integer_to_float`` C++ pass) -- the same dynamic
quantization scheme ``test_dynamic_quantize_matmul.py`` covers, but using
ONNX Runtime's "com.microsoft" contrib op ``MatMulIntegerToFloat`` to fuse
the dequantize (and optional bias-add) step into a single node instead of a
MatMulInteger+Cast+Mul(+Add) chain.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs):
    # See test_dynamic_quantize_matmul.py's identical helper: INT8 dynamic
    # quantization rounding is a discontinuous function of its input, so a
    # value near a rounding boundary can land in the adjacent bucket from a
    # last-bit floating-point difference across platforms/onnxruntime
    # versions. Checking the aggregate relative L2 error (not a tight
    # per-element band) avoids exactly the CI flakiness that produced.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 0.1, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_matmul():
    rng = np.random.default_rng(0)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulIntegerToFloat"] == 1
    # No separate MatMulInteger/Cast/Mul/Add chain -- MatMulIntegerToFloat
    # fuses all of it into one node.
    assert ops["MatMulInteger"] == 0
    assert ops["Cast"] == 0
    assert ops["Mul"] == 0
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulIntegerToFloat"] == 1
    # The bias is passed directly as MatMulIntegerToFloat's 7th input, not a
    # separate Add node.
    assert ops["Add"] == 0

    mmitf = next(n for n in quant.graph.node if n.op_type == "MatMulIntegerToFloat")
    assert len(mmitf.input) == 7
    assert mmitf.input[6] == "B"

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_matmul_no_bias_uses_empty_placeholder():
    rng = np.random.default_rng(2)
    K, N = 16, 8
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [2, K])], [_vi("Y", [2, N])], [weight])

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    mmitf = next(n for n in quant.graph.node if n.op_type == "MatMulIntegerToFloat")
    # b_zero_point (index 5) omitted as the standard empty-string
    # placeholder; no bias, so only 6 inputs total (no trailing 7th).
    assert mmitf.input[5] == ""
    assert len(mmitf.input) == 6


def test_quantize_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8]), _vi("W", [8, 4])], [_vi("Y", [4, 4])], [])
    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_non_default_gemm_attrs():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], alpha=2.0)]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight])
    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    assert _op_counts(quant)["Gemm"] == 1


def test_quantize_skips_reduction_depth_that_would_overflow_int32():
    from onnxsim.precision_estimator import MAX_SAFE_INT32_REDUCTION_DEPTH

    k = MAX_SAFE_INT32_REDUCTION_DEPTH + 1
    weight = _f32(np.random.randn(k, 1) * 0.01, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, k])], [_vi("Y", [1, 1])], [weight])

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["MatMulIntegerToFloat"] == 0
