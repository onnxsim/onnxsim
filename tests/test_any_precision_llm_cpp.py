"""Tests for ``onnxsim.apply_any_precision_llm_cpp`` -- the C++-backed port
of ``onnxsim.apply_any_precision_llm`` (see
``onnxsim/passes/any_precision_llm.h``). Floating-point iteration order
differs from the pure-Python port (grouping by hash map, not numpy's own
reduction order), so these tests check the same structural/algebraic
properties the Python port's own tests check (exact nesting invariant,
monotonically improving reconstruction with more bits) rather than
bit-for-bit equality with the Python port.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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
    # The C++ pass (like weight_only_quantize_matmul.h's own pattern)
    # rewires the matched node's weight input to a freshly created
    # initializer, leaving the original one dangling unused in the graph --
    # so the *node's own current input name* is the only reliable way to
    # find the actual (post-quantization) weight, not initializer list
    # position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_any_precision_llm_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 8)).astype(np.float32)
    model = _matmul_model(w, K=32, N=8)

    q = onnxsim.apply_any_precision_llm_cpp(model, bits=4, max_bits=8)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)


def test_cpp_any_precision_llm_reconstruction_error_improves_with_more_bits():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(w, K=64, N=16)

    errors = []
    for bits in (2, 4, 6, 8):
        q = onnxsim.apply_any_precision_llm_cpp(model, bits=bits, max_bits=8)
        new_w = _current_weight(q)
        errors.append(
            float(np.linalg.norm(new_w.astype(np.float64) - w.astype(np.float64)))
        )

    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-3
    assert errors[-1] < errors[0] * 0.3


def test_cpp_any_precision_llm_low_bit_matches_direct_low_bit_call():
    # The "quantize once, deploy at any precision" property: truncating a
    # max_bits=8 tree down to 3 bits must give the same result as building
    # the tree directly to max_bits=3.
    rng = np.random.default_rng(2)
    w = rng.standard_normal((32, 8)).astype(np.float32)
    model = _matmul_model(w, K=32, N=8)

    direct = onnxsim.apply_any_precision_llm_cpp(model, bits=3, max_bits=3)
    via_tree = onnxsim.apply_any_precision_llm_cpp(model, bits=3, max_bits=8)
    w_direct = _current_weight(direct)
    w_tree = _current_weight(via_tree)
    np.testing.assert_allclose(w_direct, w_tree, rtol=1e-5, atol=1e-6)


def test_cpp_any_precision_llm_behaves_similarly_to_python_port():
    # Not bit-for-bit identical (different iteration/summation order -- see
    # this test file's own docstring and passes/any_precision_llm.h's own
    # documented divergence note), but should reach a similar reconstruction
    # error on the same input, the same "independently correct" contract
    # apply_quarot/apply_quarot_cpp already established.
    rng = np.random.default_rng(3)
    w = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(w, K=64, N=16)

    py_q = onnxsim.apply_any_precision_llm(model, bits=5, max_bits=8)
    cpp_q = onnxsim.apply_any_precision_llm_cpp(model, bits=5, max_bits=8)
    py_w = onnx.numpy_helper.to_array(py_q.graph.initializer[0]).astype(np.float64)
    cpp_w = _current_weight(cpp_q).astype(np.float64)

    w64 = w.astype(np.float64)
    py_err = np.linalg.norm(py_w - w64)
    cpp_err = np.linalg.norm(cpp_w - w64)
    # Both should be small and of the same order of magnitude.
    assert cpp_err < np.linalg.norm(w64) * 0.1
    assert cpp_err < py_err * 3.0 and py_err < cpp_err * 3.0


def test_cpp_any_precision_llm_gemm_with_bias():
    rng = np.random.default_rng(4)
    K, N = 32, 8
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
    q = onnxsim.apply_any_precision_llm_cpp(model, bits=4, max_bits=8)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.1


def test_cpp_any_precision_llm_rejects_bits_outside_range():
    model = _matmul_model(np.zeros((8, 4), dtype=np.float32), K=8, N=4)
    with pytest.raises(Exception):
        onnxsim.apply_any_precision_llm_cpp(model, bits=9, max_bits=8)
    with pytest.raises(Exception):
        onnxsim.apply_any_precision_llm_cpp(model, bits=0, max_bits=8)


def test_cpp_any_precision_llm_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_any_precision_llm_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_any_precision_llm_defaults_match_python():
    import inspect

    sig = inspect.signature(onnxsim.apply_any_precision_llm_cpp)
    py_sig = inspect.signature(onnxsim.apply_any_precision_llm)
    assert sig.parameters["bits"].default == py_sig.parameters["bits"].default
    assert sig.parameters["max_bits"].default == py_sig.parameters["max_bits"].default
    assert (
        sig.parameters["block_size"].default == py_sig.parameters["block_size"].default
    )
