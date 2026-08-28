"""Tests for ``onnxsim.quantize_qoperator`` (the ``qoperator_quantize_matmul``
C++ pass) and its output-range calibration helper.

Each model is built directly with the ``onnx.parser`` text format (no torch
dependency), calibrated with random data, quantized, and then actually run
through ONNX Runtime -- both before and after quantization -- so these tests
double as a minimal end-to-end calibrate/quantize/deploy check: the quantized
graph must load and execute under a real inference engine, and its outputs
must stay close to the float baseline.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
    # Pin a low IR version so the model loads under older onnxruntime builds
    # (which cap at IR version 11), matching test_fusion_patterns.py.
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


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
    # INT8/uint8 quantization is lossy by design; see
    # test_dynamic_quantize_matmul.py's identically-named helper for why this
    # checks aggregate relative L2 error rather than a tight per-element bound.
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
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [weight],
    )

    quant = onnxsim.quantize_qoperator(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    # Unlike quantize_static's QDQ format, the MatMul node itself is replaced
    # by QLinearMatMul -- no float MatMul is left in the graph.
    assert ops["MatMul"] == 0
    assert ops["QLinearMatMul"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    # QLinearMatMul has no bias input, so the bias is added back in float
    # after dequantization, like dynamic_quantize_matmul.h does.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    model = _model(
        f"""
        g (float[3,{K}] X) => (float[3,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [weight, bias],
    )

    quant = onnxsim.quantize_qoperator(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 0
    assert ops["QLinearMatMul"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["Add"] == 1

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_weight():
    # Both MatMul operands are graph inputs (neither is a constant), so there
    # is nothing to quantize ahead of time.
    model = _model(
        """
        g (float[4,8] X, float[8,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    quant = onnxsim.quantize_qoperator(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["QLinearMatMul"] == 0


def test_quantize_skips_non_default_gemm_attrs():
    # alpha != 1 falls outside the "vanilla" Gemm shape this pass handles.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = Gemm<alpha = 2.0>(X, W)
        }
        """,
        [weight],
    )
    quant = onnxsim.quantize_qoperator(model)
    assert _op_counts(quant)["Gemm"] == 1
    assert _op_counts(quant)["QLinearMatMul"] == 0


def test_quantize_skips_old_opset():
    # QLinearMatMul needs opset >= 10.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [weight],
        opset=9,
    )
    quant = onnxsim.quantize_qoperator(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["QLinearMatMul"] == 0


def test_list_qoperator_quantizable_outputs_includes_matmul_output():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[2,8] X) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [weight],
    )
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_quantizable_outputs(model.SerializeToString())
    assert set(names) == {"Y"}


def test_calibrate_with_extra_tensor_names_includes_output():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[2,8] X) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [weight],
    )
    data = onnxsim.generate_random_calibration_data(model, num_samples=4, seed=0)
    ranges = onnxsim.calibrate(model, data, extra_tensor_names=["Y"])
    assert set(ranges.keys()) == {"X", "Y"}
    lo, hi = ranges["Y"]
    assert lo < hi
