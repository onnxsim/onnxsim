"""Tests for ``onnxsim.apply_gguf_q5_0_quantization``/``apply_gguf_q5_1_quantization``
(see ``onnxsim/gguf_legacy_quant_5bit.py``) -- llama.cpp's legacy Q5_0/Q5_1
GGUF block formats, weight-only, represented as a float32
quantize-dequantize round trip.

Numerics are checked directly against numpy re-implementations of the
quantize/dequantize math, not via an onnxruntime round trip -- this
module never introduces any new graph nodes (the quantized weight is
folded directly into a new initializer).
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim
from onnxsim.gguf_legacy_quant import (
    quantize_dequantize_q4_0,
    quantize_dequantize_q4_1,
)
from onnxsim.gguf_legacy_quant_5bit import _BLOCK_SIZE, _MAX_CODE, _Q5_0_BIAS


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


def test_q5_0_round_trip_is_close_to_the_original_weight():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((8, _BLOCK_SIZE)) * 0.5
    dequant = onnxsim.quantize_dequantize_q5_0(values)
    assert dequant.shape == values.shape
    # A step is d = max|x| / 16, so error is half a step everywhere except
    # at the most positive element: with the fixed -16 bias, +max|x| wants
    # code 32, which clips to 31, costing a full step (Q5_0 inherits this
    # asymmetry from Q4_0 -- its signed code range is [-16, 15]).
    max_abs = np.abs(values).max(axis=-1)
    assert np.all(np.abs(dequant - values) <= max_abs[:, None] / 16.0 + 1e-3)


def test_q5_0_codes_stay_within_the_documented_range():
    # Re-derive each element's code from the dequantized output:
    # dequant = (code - 16) * d, so code = dequant / d + 16 must land in
    # [0, 31] (a 5-bit unsigned code with the format's fixed -16 bias).
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_BLOCK_SIZE * 4) * 3.0
    dequant = onnxsim.quantize_dequantize_q5_0(values)
    for block_in, block_out in zip(
        values.reshape(-1, _BLOCK_SIZE), dequant.reshape(-1, _BLOCK_SIZE)
    ):
        d = np.float64(np.float16(max(np.abs(block_in).max(), 1e-12) / _Q5_0_BIAS))
        code = np.round(block_out / d) + _Q5_0_BIAS
        assert code.min() >= 0
        assert code.max() <= _MAX_CODE
        assert len(np.unique(block_out)) <= _MAX_CODE + 1


def test_q5_0_is_symmetric_zero_maps_to_zero():
    # Q5_0 has no separate min/zero-point -- a value of exactly 0 must
    # round-trip to exactly 0 regardless of the rest of the block (since
    # code=16 maps to (16-16)*d == 0 exactly for any d).
    rng = np.random.default_rng(2)
    values = rng.standard_normal(_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_q5_0(values)
    assert dequant[0] == 0.0


def test_q5_1_round_trip_is_close_to_the_original_weight():
    rng = np.random.default_rng(3)
    values = rng.standard_normal((8, _BLOCK_SIZE)) * 0.5
    dequant = onnxsim.quantize_dequantize_q5_1(values)
    assert dequant.shape == values.shape
    span = values.max(axis=-1) - values.min(axis=-1)
    assert np.all(np.abs(dequant - values) <= span[:, None] / _MAX_CODE / 2.0 + 1e-3)


def test_q5_1_codes_stay_within_the_documented_range():
    # Q5_1 uses the 5-bit code unsigned with an explicit min:
    # dequant = code * d + m, so code = (dequant - m) / d in [0, 31].
    rng = np.random.default_rng(4)
    values = rng.standard_normal(_BLOCK_SIZE * 4) * 2.0 + 5.0
    dequant = onnxsim.quantize_dequantize_q5_1(values)
    for block_in, block_out in zip(
        values.reshape(-1, _BLOCK_SIZE), dequant.reshape(-1, _BLOCK_SIZE)
    ):
        m = np.float64(np.float16(block_in.min()))
        d = np.float64(
            np.float16(max(block_in.max() - block_in.min(), 1e-12) / _MAX_CODE)
        )
        code = np.round((block_out - m) / d)
        assert code.min() >= 0
        assert code.max() <= _MAX_CODE
        assert len(np.unique(block_out)) <= _MAX_CODE + 1


def test_q5_1_recovers_a_constant_offset_block_exactly():
    # Q5_1's explicit min lets it represent a block that is a small
    # uniform grid plus a large constant offset almost exactly -- Q5_0
    # (symmetric, no min) would waste most of its range on the offset.
    codes = np.arange(_BLOCK_SIZE) % 4  # 0..3, repeating
    values = codes.astype(np.float64) * 0.1 + 1000.0
    dequant = onnxsim.quantize_dequantize_q5_1(values)
    np.testing.assert_allclose(dequant, values, atol=1e-2)


def test_q5_0_beats_q4_0_on_the_same_weight():
    # One extra bit of code resolution should measurably help.
    rng = np.random.default_rng(5)
    values = rng.standard_normal(_BLOCK_SIZE * 16)
    err_q4_0 = float(np.mean((values - quantize_dequantize_q4_0(values)) ** 2))
    err_q5_0 = float(np.mean((values - onnxsim.quantize_dequantize_q5_0(values)) ** 2))
    assert err_q5_0 < err_q4_0


def test_q5_1_beats_q4_1_on_the_same_weight():
    rng = np.random.default_rng(6)
    values = rng.standard_normal(_BLOCK_SIZE * 16) * 0.3 + 2.0
    err_q4_1 = float(np.mean((values - quantize_dequantize_q4_1(values)) ** 2))
    err_q5_1 = float(np.mean((values - onnxsim.quantize_dequantize_q5_1(values)) ** 2))
    assert err_q5_1 < err_q4_1


def test_q5_0_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(7)
    aligned = rng.standard_normal(_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])
    dequant_aligned = onnxsim.quantize_dequantize_q5_0(aligned)
    dequant_padded = onnxsim.quantize_dequantize_q5_0(padded)
    np.testing.assert_array_equal(dequant_padded[: _BLOCK_SIZE * 2], dequant_aligned)


def test_apply_gguf_q5_0_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(8)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q5_0_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q5_0(w).astype(np.float32)
    )
    assert len(q.graph.node) == len(model.graph.node)


def test_apply_gguf_q5_1_quantization_handles_gemm_with_transb():
    rng = np.random.default_rng(9)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((N, K)).astype(np.float32)  # transB=1 layout
    model = _model(
        f"""
        g (float[2,{K}] X) => (float[2,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W)
        }}
        """,
        [_f32(w, "W")],
    )

    q = onnxsim.apply_gguf_q5_1_quantization(model)
    onnx.checker.check_model(q)

    gemm_node = next(n for n in q.graph.node if n.op_type == "Gemm")
    assert gemm_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == gemm_node.input[1])
    )
    assert new_w.shape == w.shape
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q5_1(w).astype(np.float32)
    )


def test_apply_gguf_q5_1_quantization_matches_conv_weight_when_enabled():
    rng = np.random.default_rng(10)
    w = rng.standard_normal((4, 2, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,4,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )

    q_with_conv = onnxsim.apply_gguf_q5_1_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_q5_1_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_apply_gguf_5bit_quantization_respects_skip_names():
    rng = np.random.default_rng(11)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    for apply_fn in (
        onnxsim.apply_gguf_q5_0_quantization,
        onnxsim.apply_gguf_q5_1_quantization,
    ):
        result = apply_fn(model, skip_names={"W"})
        assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_5bit_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_q5_0_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_5bit_quantization_skips_non_constant_weight():
    # W is produced by a node, not an initializer -- nothing to fold.
    K, N = _BLOCK_SIZE, 4
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{K},{N}] W) => (float[batch,{N}] Y)
        {{
          Wt = Identity(W)
          Y = MatMul(X, Wt)
        }}
        """
    )
    for apply_fn in (
        onnxsim.apply_gguf_q5_0_quantization,
        onnxsim.apply_gguf_q5_1_quantization,
    ):
        result = apply_fn(model)
        assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_5bit_quantization_skips_non_float_weight():
    K, N = _BLOCK_SIZE, 4
    w = np.zeros((K, N), dtype=np.int64)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Wf = Cast<to=1>(W)
          Y = MatMul(X, Wf)
        }}
        """,
        [onnx.numpy_helper.from_array(w, name="W")],
    )
    result = onnxsim.apply_gguf_q5_1_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
