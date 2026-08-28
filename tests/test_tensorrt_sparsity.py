"""Tests for ``onnxsim.convert_matmul_to_gemm`` -- rewrites MatMul into an
equivalent Gemm so N:M-pruned weights become eligible for ONNX Runtime's
TensorRT execution provider sparse math, see ``onnxsim/tensorrt_sparsity.py``.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_types(model):
    return [n.op_type for n in model.graph.node]


def test_2d_input_is_a_direct_swap_no_scaffold():
    K, N = 8, 4
    rng = np.random.default_rng(0)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    onnx.checker.check_model(out)
    assert _op_types(out) == ["Gemm"]

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y0,) = _run(model, {"X": x})
    (y1,) = _run(out, {"X": x})
    np.testing.assert_array_equal(y0, y1)


def test_3d_batched_input_gets_flatten_unflatten_scaffold():
    K, N = 8, 4
    rng = np.random.default_rng(1)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    onnx.checker.check_model(out)
    assert _op_types(out) == ["Reshape", "Gemm", "Shape", "Slice", "Concat", "Reshape"]
    assert sum(n.op_type == "MatMul" for n in out.graph.node) == 0

    x = rng.standard_normal((2, 5, K)).astype(np.float32)
    (y0,) = _run(model, {"X": x})
    (y1,) = _run(out, {"X": x})
    assert y0.shape == y1.shape == (2, 5, N)
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)


def test_1d_input_matches_matmuls_vector_promotion_semantics():
    # numpy/ONNX MatMul semantics: a 1-D lhs is promoted to a row vector,
    # multiplied, then the prepended 1 is dropped from the result -- the
    # rank-agnostic reshape/slice scaffold must reproduce this exactly.
    K, N = 8, 4
    rng = np.random.default_rng(2)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[{K}] X) => (float[{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    onnx.checker.check_model(out)

    x = rng.standard_normal((K,)).astype(np.float32)
    (y0,) = _run(model, {"X": x})
    (y1,) = _run(out, {"X": x})
    assert y0.shape == y1.shape == (N,)
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)


def test_unknown_rank_input_still_uses_correct_scaffold():
    # No declared shape for X at all (rank not statically known) -- shape
    # inference can't help, so this must still fall back to the general
    # (rank-agnostic) scaffold rather than guessing 2-D.
    K, N = 8, 4
    rng = np.random.default_rng(3)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[1] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    # The text parser always sets a (possibly empty) shape; clear it entirely
    # to reproduce make_tensor_value_info(..., shape=None)'s "rank not
    # statically known" value info, which is what this test exercises.
    model.graph.input[0].type.tensor_type.ClearField("shape")

    out = onnxsim.convert_matmul_to_gemm(model)
    assert "MatMul" not in _op_types(out)
    assert "Gemm" in _op_types(out)

    x = rng.standard_normal((2, 5, K)).astype(np.float32)
    (y0,) = _run(model, {"X": x})
    (y1,) = _run(out, {"X": x})
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)


def test_opset_below_13_is_a_no_op():
    K, N = 8, 4
    rng = np.random.default_rng(4)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
        opset=12,
        ir_version=8,
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    assert _op_types(out) == ["MatMul"]


def test_non_constant_weight_is_left_untouched():
    K, N = 8, 4
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{K},{N}] W_dyn) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W_dyn)
        }}
        """
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    assert _op_types(out) == ["MatMul"]


def test_non_2d_weight_is_left_untouched():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((2, 8, 4)).astype(np.float32)
    model = _model(
        """
        g (float[batch,2,8] X) => (float[batch,2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[_f32(w, "W")],
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    assert _op_types(out) == ["MatMul"]


def test_existing_gemm_nodes_are_left_alone():
    K, N = 8, 4
    rng = np.random.default_rng(6)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )

    out = onnxsim.convert_matmul_to_gemm(model)
    assert _op_types(out) == ["Gemm"]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(out.graph.initializer[0]),
        onnx.numpy_helper.to_array(model.graph.initializer[0]),
    )


def test_composes_with_nm_pruning_weight_untouched_by_conversion():
    # The real point of this pass: run it after N:M pruning and confirm the
    # pruned (2:4-sparse) weight survives byte-for-byte, and the converted
    # graph's output still matches the pruned float graph's exactly.
    K, H, N = 16, 32, 4
    rng = np.random.default_rng(7)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    converted = onnxsim.convert_matmul_to_gemm(pruned)
    onnx.checker.check_model(converted)
    assert "MatMul" not in _op_types(converted)

    pruned_w1 = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "W1")
    )
    converted_w1 = onnx.numpy_helper.to_array(
        next(t for t in converted.graph.initializer if t.name == "W1")
    )
    np.testing.assert_array_equal(pruned_w1, converted_w1)
    assert onnxsim.weight_sparsity(converted) == pytest.approx(0.5, abs=1e-9)

    x = rng.standard_normal((2, 5, K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_converted,) = _run(converted, {"X": x})
    np.testing.assert_allclose(y_pruned, y_converted, rtol=1e-6, atol=1e-6)


def test_no_matmul_nodes_is_a_no_op():
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    out = onnxsim.convert_matmul_to_gemm(model)
    assert _op_types(out) == ["Relu"]
