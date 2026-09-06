"""Tests for ``onnxsim.apply_gguf_ternary_quantization`` (see
``onnxsim/gguf_ternary_quant.py``) -- BitNet b1.58's published absmean
ternary quantization rule, as shipped by llama.cpp's GGUF TQ1_0/TQ2_0
tensor types, weight-only, represented as a float32 quantize-dequantize
round trip.

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
from onnxsim.gguf_ternary_quant import _BLOCK_SIZE


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


def test_hand_computed_block_matches_the_papers_own_absmean_rule():
    # BitNet b1.58's own published rule, computed by hand: tiling [1, 2, 3,
    # -4] to fill a whole 256-element block keeps the block's mean(|w|)
    # equal to mean(|[1, 2, 3, 4]|) = 2.5 exactly (tiling doesn't change a
    # mean), with no padding dilution. code = round(clip(w/d, -1, 1)) =
    # round([0.4, 0.8, 1.0, -1.0]) = [0, 1, 1, -1]; dequant = code * d.
    values = np.tile(np.array([1.0, 2.0, 3.0, -4.0]), _BLOCK_SIZE // 4)
    dequant = onnxsim.quantize_dequantize_ternary(values)
    expected_d = np.float16(np.mean(np.abs(values))).astype(np.float64)
    expected = np.tile(
        np.array([0.0, expected_d, expected_d, -expected_d]), _BLOCK_SIZE // 4
    )
    np.testing.assert_allclose(dequant, expected, atol=1e-6)


def test_round_trip_has_at_most_3_levels_per_block():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(_BLOCK_SIZE * 3)
    dequant = onnxsim.quantize_dequantize_ternary(values)
    assert dequant.shape == values.shape
    for block in dequant.reshape(-1, _BLOCK_SIZE):
        assert len(np.unique(block)) <= 3


def test_zero_maps_to_zero():
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_ternary(values)
    assert dequant[0] == 0.0


def test_all_dequantized_values_share_the_same_magnitude_per_block():
    # Every nonzero dequantized value in a block must be exactly +d or -d
    # (a single shared scale per block, per the paper's own rule).
    rng = np.random.default_rng(2)
    values = rng.standard_normal(_BLOCK_SIZE) * 3.0
    dequant = onnxsim.quantize_dequantize_ternary(values)
    nonzero = np.abs(dequant[dequant != 0.0])
    assert nonzero.size > 0
    np.testing.assert_allclose(nonzero, nonzero[0], rtol=1e-6)


def test_large_uniform_magnitude_weights_round_trip_near_exactly():
    # A block whose every |weight| already equals its own mean should
    # round-trip almost exactly (code = sign(w), d = |w|).
    rng = np.random.default_rng(3)
    signs = rng.choice([-1.0, 1.0], size=_BLOCK_SIZE)
    values = signs * 2.0
    dequant = onnxsim.quantize_dequantize_ternary(values)
    np.testing.assert_allclose(dequant, values, atol=1e-2)


def test_ternary_beats_naive_binary_sign_quantization_on_sparse_weights():
    # The paper's central motivation: many weights are near-zero and should
    # quantize to exactly 0, which a two-level {-1, +1} sign-only scheme
    # cannot represent at all -- so ternary should win on a sparse block.
    rng = np.random.default_rng(4)
    n = _BLOCK_SIZE * 8
    mask = rng.random(n) < 0.7
    values = rng.standard_normal(n)
    values[mask] = 0.0

    ternary = onnxsim.quantize_dequantize_ternary(values)

    blocks = values.reshape(-1, _BLOCK_SIZE)
    d_binary = np.maximum(np.mean(np.abs(blocks), axis=-1), 1e-12)
    signs = np.sign(blocks)
    signs[signs == 0.0] = 1.0
    binary = (signs * d_binary[:, np.newaxis]).reshape(-1)

    err_ternary = float(np.mean((values - ternary) ** 2))
    err_binary = float(np.mean((values - binary) ** 2))
    assert err_ternary < err_binary


def test_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(5)
    aligned = rng.standard_normal(_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])
    dequant_aligned = onnxsim.quantize_dequantize_ternary(aligned)
    dequant_padded = onnxsim.quantize_dequantize_ternary(padded)
    np.testing.assert_array_equal(dequant_padded[: _BLOCK_SIZE * 2], dequant_aligned)


def test_apply_gguf_ternary_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(6)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_ternary_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    assert len(q.graph.node) == len(model.graph.node)


def test_apply_gguf_ternary_quantization_matches_conv_weight_when_enabled():
    rng = np.random.default_rng(7)
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

    q_with_conv = onnxsim.apply_gguf_ternary_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_ternary_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_apply_gguf_ternary_quantization_respects_skip_names():
    rng = np.random.default_rng(8)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_gguf_ternary_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_ternary_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_ternary_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_ternary_quantization_skips_non_float_weight():
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
    result = onnxsim.apply_gguf_ternary_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
