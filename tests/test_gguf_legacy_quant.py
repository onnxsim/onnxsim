"""Tests for ``onnxsim.apply_gguf_q4_0_quantization``/``apply_gguf_q4_1_quantization``
(see ``onnxsim/gguf_legacy_quant.py``) -- llama.cpp's legacy Q4_0/Q4_1
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
from onnxsim.gguf_legacy_quant import _BLOCK_SIZE


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


def _naive_int4_round_trip(values: np.ndarray, block_size: int) -> np.ndarray:
    flat = values.reshape(-1)
    blocks = flat.reshape(-1, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / 7.0
    q = np.clip(np.round(blocks / scale[:, None]), -7, 7)
    return (q * scale[:, None]).reshape(values.shape)


def test_q4_0_round_trip_has_at_most_16_levels_per_block():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(_BLOCK_SIZE * 3)
    dequant = onnxsim.quantize_dequantize_q4_0(values)
    assert dequant.shape == values.shape
    for block in dequant.reshape(-1, _BLOCK_SIZE):
        assert len(np.unique(block)) <= 16


def test_q4_0_is_symmetric_zero_maps_to_zero():
    # Q4_0 has no separate min/zero-point -- a value of exactly 0 must
    # round-trip to exactly 0 regardless of the rest of the block (since
    # code=8 maps to (8-8)*d == 0 exactly for any d).
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_q4_0(values)
    assert dequant[0] == 0.0


def test_q4_1_recovers_a_constant_offset_block_exactly():
    # Q4_1's explicit min lets it represent a block that is a small
    # uniform grid plus a large constant offset almost exactly -- Q4_0
    # (symmetric, no min) would waste most of its range on the offset.
    codes = np.arange(_BLOCK_SIZE) % 4  # 0..3, repeating
    values = codes.astype(np.float64) * 0.1 + 1000.0
    dequant = onnxsim.quantize_dequantize_q4_1(values)
    np.testing.assert_allclose(dequant, values, atol=1e-2)


def test_q4_1_beats_q4_0_on_a_large_constant_offset_block():
    rng = np.random.default_rng(2)
    values = rng.standard_normal(_BLOCK_SIZE) * 0.1 + 1000.0
    q4_0 = onnxsim.quantize_dequantize_q4_0(values)
    q4_1 = onnxsim.quantize_dequantize_q4_1(values)
    err_q4_0 = float(np.mean((values - q4_0) ** 2))
    err_q4_1 = float(np.mean((values - q4_1) ** 2))
    assert err_q4_1 < err_q4_0


def test_legacy_quant_beats_naive_single_scale_int4_on_mixed_blocks():
    rng = np.random.default_rng(3)
    n = _BLOCK_SIZE * 64
    sub_block_scales = rng.uniform(0.1, 5.0, size=n // _BLOCK_SIZE)
    noise = rng.standard_normal(n).reshape(-1, _BLOCK_SIZE)
    values = (noise * sub_block_scales[:, None]).reshape(-1)

    q4_0 = onnxsim.quantize_dequantize_q4_0(values)
    naive = _naive_int4_round_trip(values, _BLOCK_SIZE)
    assert np.mean((values - q4_0) ** 2) <= np.mean((values - naive) ** 2) * 1.01


def test_q4_0_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(4)
    aligned = rng.standard_normal(_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])
    dequant_aligned = onnxsim.quantize_dequantize_q4_0(aligned)
    dequant_padded = onnxsim.quantize_dequantize_q4_0(padded)
    np.testing.assert_array_equal(dequant_padded[: _BLOCK_SIZE * 2], dequant_aligned)


def test_apply_gguf_q4_0_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q4_0_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    assert len(q.graph.node) == len(model.graph.node)


def test_apply_gguf_q4_1_quantization_matches_conv_weight_when_enabled():
    rng = np.random.default_rng(6)
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

    q_with_conv = onnxsim.apply_gguf_q4_1_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_q4_1_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_apply_gguf_legacy_quantization_respects_skip_names():
    rng = np.random.default_rng(7)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_gguf_q4_0_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_legacy_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_q4_0_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_legacy_quantization_skips_non_float_weight():
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
    result = onnxsim.apply_gguf_q4_0_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
