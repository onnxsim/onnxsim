"""Tests for ``onnxsim.apply_imatrix_quantization_cpp`` -- the C++-backed
port of ``onnxsim.apply_imatrix_quantization`` (llama.cpp's "importance
matrix" applied to this repository's own plain block-wise INT4 weight
quantizer; see ``onnxsim/imatrix_quant.py``'s own module docstring for the
technique and ``onnxsim/imatrix_quant_entry.h`` for this port's own scope).
Like ``test_wanda_pruning_cpp.py``, this runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor.

Unlike Wanda (a pruning pass whose C++ port is expected to zero exactly the
same entries as its pure-Python reference), this port's own weighted
grid-search quantizer is the *same algorithm* as
``onnxsim.imatrix_quant.quantize_dequantize_int4_imatrix``, ported
scalar-loop-for-scalar-loop (see ``onnxsim/passes/imatrix_quant.h``'s own
top comment) -- so, unlike ``ApplyQuarot``/``apply_quarot_cpp`` (a
*deliberately* non-interchangeable independent construction), this port is
expected to track the pure-Python reference numerically closely (float32
protobuf round-trip precision aside), which the first test below checks
directly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.imatrix_quant import quantize_dequantize_int4_plain

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _matmul_model(w, K, N, batch="batch", opset=21):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_port_matches_python_reference_closely():
    rng = np.random.default_rng(0)
    K, N = 64, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    calibration_data = [
        {"X": rng.standard_normal((8, K)).astype(np.float32)} for _ in range(6)
    ]

    py_result = onnxsim.apply_imatrix_quantization(
        model, calibration_data=calibration_data
    )
    cpp_result = onnxsim.apply_imatrix_quantization_cpp(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(py_result)
    onnx.checker.check_model(cpp_result)

    py_w = _current_weight(py_result)
    cpp_w = _current_weight(cpp_result)
    np.testing.assert_allclose(cpp_w, py_w, rtol=1e-5, atol=1e-6)


def test_apply_imatrix_quantization_cpp_replaces_weight():
    rng = np.random.default_rng(1)
    K, N = 64, 8
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_imatrix_quantization_cpp(model, num_samples=4, seed=0)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)


def test_apply_imatrix_quantization_cpp_output_stays_close_to_float():
    rng = np.random.default_rng(2)
    K, N = 64, 16
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_imatrix_quantization_cpp(model, num_samples=8, seed=0)
    onnx.checker.check_model(q)

    x = rng.standard_normal((6, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    rel_l2 = np.linalg.norm(float_y - q_y) / max(np.linalg.norm(float_y), 1e-9)
    assert rel_l2 < 0.5


def test_apply_imatrix_quantization_cpp_gemm_with_bias():
    rng = np.random.default_rng(3)
    K, N = 64, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_imatrix_quantization_cpp(model, num_samples=4, seed=0)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))


def test_apply_imatrix_quantization_cpp_respects_skip_names():
    rng = np.random.default_rng(4)
    K, N = 64, 8
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_imatrix_quantization_cpp(
        model, num_samples=4, skip_names=["W"]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_imatrix_quantization_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_imatrix_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_imatrix_quantization_cpp_skips_reduction_dim_not_divisible_by_block_size():
    rng = np.random.default_rng(5)
    K, N = 50, 8  # not a multiple of block_size=32
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_imatrix_quantization_cpp(model, num_samples=4)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_imatrix_quantization_cpp_beats_plain_quantization_on_output_fidelity():
    # C++-side version of the core empirical claim (see
    # test_imatrix_quant.py's own Python-side version of this same test):
    # on a layer whose calibration activations make one weight-outlier
    # input channel unimportant while the rest carry real signal,
    # importance-weighted quantization should reproduce the float model's
    # real output more faithfully than the plain (unweighted) baseline.
    rng = np.random.default_rng(6)
    K, N, block_size = 64, 8, 32
    outlier_channel = 5

    w = rng.standard_normal((K, N)).astype(np.float64) * 0.3
    w[outlier_channel, :] += rng.choice([-1.0, 1.0], N) * 12.0
    w = w.astype(np.float32)
    model = _matmul_model(w, K, N)

    def _make_batch(num_rows, seed_offset):
        r = np.random.default_rng(100 + seed_offset)
        x = r.standard_normal((num_rows, K)).astype(np.float32)
        x[:, outlier_channel] = (r.standard_normal(num_rows) * 0.02).astype(np.float32)
        return {"X": x}

    calibration_data = [_make_batch(16, i) for i in range(8)]

    q_imatrix = onnxsim.apply_imatrix_quantization_cpp(
        model, calibration_data=calibration_data, block_size=block_size
    )
    w_plain_dequant = quantize_dequantize_int4_plain(w.astype(np.float64).T, block_size)
    plain_model = _matmul_model(w_plain_dequant.T.astype(np.float32), K, N)

    onnx.checker.check_model(q_imatrix)
    onnx.checker.check_model(plain_model)

    x_test = _make_batch(2000, 999)["X"]
    (float_y,) = _run(model, {"X": x_test})
    (imatrix_y,) = _run(q_imatrix, {"X": x_test})
    (plain_y,) = _run(plain_model, {"X": x_test})

    imatrix_mse = float(np.mean((float_y - imatrix_y) ** 2))
    plain_mse = float(np.mean((float_y - plain_y) ** 2))
    assert imatrix_mse < plain_mse
