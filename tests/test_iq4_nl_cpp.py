"""Tests for ``onnxsim.apply_iq4_nl_quantization_cpp`` -- the C++-backed
port of ``onnxsim.apply_iq4_nl_quantization`` (see
``onnxsim/passes/iq4_nl.h``). Unlike most other ``*_cpp`` ports in this
repo, IQ4_NL has no accumulation or iterative-refinement step at all (see
that header's own "ACCEPTED, PERMANENT DIVERGENCE" note), so this port is
expected to track the pure-Python port unusually closely -- but these
tests still check structural/algebraic properties and comparable (not
required to be bit-for-bit identical) reconstruction error, matching this
repo's own established contract for a ``*_cpp`` port
(``tests/test_any_precision_llm_cpp.py``, ``tests/test_quarot_cpp.py``).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.iq4_nl import _BLOCK_SIZE, IQ4_NL_CODEBOOK

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


def _current_weight(model, weight_input_index=1):
    # The C++ pass (like any_precision_llm.h's own pattern) rewires the
    # matched node's weight input to a freshly created initializer, leaving
    # the original one dangling unused in the graph -- so the *node's own
    # current input name* is the only reliable way to find the actual
    # (post-quantization) weight, not initializer list position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_iq4_nl_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((_BLOCK_SIZE * 2, 8)).astype(np.float32)
    model = _matmul_model(w, K=_BLOCK_SIZE * 2, N=8)

    q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # any_precision_llm.h's/gguf_kquant.py's own established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_iq4_nl_at_most_16_distinct_levels_per_32_element_block():
    rng = np.random.default_rng(1)
    K, N = _BLOCK_SIZE * 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    new_w = _current_weight(q)

    # Blocks are laid out over the weight's own flattened row-major storage
    # (see passes/iq4_nl.h's own scope note), i.e. contiguous groups of 32
    # along the flattened [K, N] buffer.
    flat = new_w.reshape(-1)
    for start in range(0, flat.size, _BLOCK_SIZE):
        block = flat[start : start + _BLOCK_SIZE]
        assert len(np.unique(block)) <= 16


def test_cpp_iq4_nl_reconstructed_values_match_fixed_codebook_shape():
    rng = np.random.default_rng(2)
    K, N = _BLOCK_SIZE * 2, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    new_w = _current_weight(q).astype(np.float64)

    codebook = np.asarray(IQ4_NL_CODEBOOK)
    max_abs_codebook = np.max(np.abs(codebook))
    flat_in = w.reshape(-1).astype(np.float64)
    flat_out = new_w.reshape(-1)
    for start in range(0, flat_in.size, _BLOCK_SIZE):
        block_in = flat_in[start : start + _BLOCK_SIZE]
        block_out = flat_out[start : start + _BLOCK_SIZE]
        scale = max(np.abs(block_in).max(), 1e-12) / max_abs_codebook
        ratios = block_out / scale
        nearest_dist = np.min(np.abs(ratios[:, None] - codebook[None, :]), axis=1)
        assert np.all(nearest_dist < 1e-4)


def test_cpp_iq4_nl_reduces_reconstruction_error_versus_naive_uniform_int4():
    # The core empirical claim this format exists for, checked directly
    # against the C++ port's own actual output (not just the Python port's,
    # which tests/test_iq4_nl.py already covers): a fixed non-uniform
    # codebook should reconstruct roughly-Gaussian weights more accurately
    # than a naive uniform 4-bit grid at the same block size.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE * 64, 2
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    flat_in = w64.reshape(-1)
    naive_out = np.empty_like(flat_in)
    for start in range(0, flat_in.size, _BLOCK_SIZE):
        block = flat_in[start : start + _BLOCK_SIZE]
        scale = max(np.abs(block).max(), 1e-12) / 7.0
        q_codes = np.clip(np.round(block / scale), -7, 7)
        naive_out[start : start + _BLOCK_SIZE] = q_codes * scale

    iq4_nl_mse = float(np.mean((flat_in - new_w.reshape(-1)) ** 2))
    naive_mse = float(np.mean((flat_in - naive_out) ** 2))
    assert iq4_nl_mse < naive_mse


def test_cpp_iq4_nl_behaves_similarly_to_python_port():
    # Not required to be bit-for-bit identical (see passes/iq4_nl.h's own
    # documented divergence note), but should reach a very similar
    # reconstruction error on the same input -- this scheme has no
    # accumulation/iteration-order dependence at all, so the two ports are
    # expected to track each other unusually closely among this repo's
    # *_cpp pairs.
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE * 4, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    py_q = onnxsim.apply_iq4_nl_quantization(model)
    cpp_q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    py_w = onnx.numpy_helper.to_array(py_q.graph.initializer[-1]).astype(np.float64)
    cpp_w = _current_weight(cpp_q).astype(np.float64)

    w64 = w.astype(np.float64)
    py_err = np.linalg.norm(py_w - w64)
    cpp_err = np.linalg.norm(cpp_w - w64)
    assert cpp_err < np.linalg.norm(w64) * 0.2
    assert cpp_err < py_err * 1.5 and py_err < cpp_err * 1.5


def test_cpp_iq4_nl_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE * 2, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
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
    q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_iq4_nl_ragged_last_block_matches_python_zero_padding():
    # K not a multiple of 32 -- passes/iq4_nl.h's own scope note claims this
    # ragged-last-block approach is mathematically identical to iq4_nl.py's
    # zero-pad-then-discard one, since padding zeros can never change a
    # block's own max-abs scale. Checked directly here.
    rng = np.random.default_rng(6)
    K, N = _BLOCK_SIZE + 5, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)

    py_q = onnxsim.apply_iq4_nl_quantization(model)
    cpp_q = onnxsim.apply_iq4_nl_quantization_cpp(model)
    py_w = onnx.numpy_helper.to_array(py_q.graph.initializer[-1]).astype(np.float64)
    cpp_w = _current_weight(cpp_q).astype(np.float64)

    np.testing.assert_allclose(py_w, cpp_w, atol=1e-4)


def test_cpp_iq4_nl_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_iq4_nl_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_iq4_nl_skips_non_2d_weight():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.apply_iq4_nl_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
