"""Tests for ``onnxsim.apply_any_precision_llm`` (Any-Precision LLM, see
``onnxsim/any_precision_llm.py``) -- a nested bit-plane weight
quantization where every lower bit-width's own code is recoverable from
the highest bit-width's code by a plain integer right-shift.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.any_precision_llm import _dequantize_by_bin_mean, _nested_bitplane_codes

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _matmul_model(w, K, N, batch="batch"):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def test_nested_codes_satisfy_exact_shift_invariant():
    # The core algebraic claim: for every bit-width up to max_bits, that
    # bit-width's own code is exactly recoverable from the max-bit-width
    # code by a plain right-shift -- verified bit-for-bit, not just
    # approximately.
    rng = np.random.default_rng(0)
    values = rng.standard_normal(500)
    max_bits = 6

    max_codes = _nested_bitplane_codes(values, max_bits)
    assert max_codes.min() >= 0
    assert max_codes.max() < 2**max_bits

    for bits in range(1, max_bits + 1):
        codes_b = _nested_bitplane_codes(values, bits)
        shifted = max_codes >> (max_bits - bits)
        np.testing.assert_array_equal(codes_b, shifted)


def test_dequantize_by_bin_mean_reconstructs_each_bins_own_average():
    codes = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    values = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 100.0])
    recon = _dequantize_by_bin_mean(values, codes)
    np.testing.assert_allclose(recon, [2.0, 2.0, 20.0, 20.0, 20.0, 100.0])


def test_reconstruction_error_improves_with_more_bits():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(w, K=64, N=16)

    errors = []
    for bits in (2, 4, 6, 8):
        q = onnxsim.apply_any_precision_llm(model, bits=bits, max_bits=8)
        new_w = onnx.numpy_helper.to_array(q.graph.initializer[0])
        errors.append(
            float(np.linalg.norm(new_w.astype(np.float64) - w.astype(np.float64)))
        )

    # Strictly non-increasing (each additional bit only ever refines an
    # existing bin further, never coarsens it), and 8 bits should be
    # dramatically better than 2 bits.
    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-6
    assert errors[-1] < errors[0] * 0.3


def test_low_bit_reconstruction_matches_direct_low_bit_call():
    # Materializing bits=3 directly must equal what you get by building
    # the max_bits=8 tree and truncating -- the "quantize once, deploy at
    # any precision" property, checked end-to-end through the public API.
    rng = np.random.default_rng(2)
    w = rng.standard_normal((32, 8)).astype(np.float32)
    model = _matmul_model(w, K=32, N=8)

    direct = onnxsim.apply_any_precision_llm(model, bits=3, max_bits=3)
    via_tree = onnxsim.apply_any_precision_llm(model, bits=3, max_bits=8)
    w_direct = onnx.numpy_helper.to_array(direct.graph.initializer[0])
    w_tree = onnx.numpy_helper.to_array(via_tree.graph.initializer[0])
    # Same code assignment either way (both are 3-bit prefixes of the same
    # per-value split history), so the per-bin-mean reconstruction (which
    # only depends on the codes and the original values within each block)
    # is identical too.
    np.testing.assert_allclose(w_direct, w_tree, rtol=1e-5, atol=1e-6)


def test_output_stays_close_to_float_via_onnxruntime():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(w, K=64, N=16)
    q = onnxsim.apply_any_precision_llm(model, bits=6, max_bits=8)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.1


def test_rejects_bits_outside_range():
    model = _matmul_model(np.zeros((8, 4), dtype=np.float32), K=8, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_any_precision_llm(model, bits=9, max_bits=8)
    with pytest.raises(ValueError):
        onnxsim.apply_any_precision_llm(model, bits=0, max_bits=8)


def test_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_any_precision_llm(model)
    assert result.SerializeToString() == model.SerializeToString()
