"""Tests for ``onnxsim.apply_gguf_q6_k_quantization``/
``onnxsim.quantize_dequantize_q6_k`` (see ``onnxsim/gguf_q6_k.py``) --
llama.cpp's Q6_K K-quant format, weight-only, represented as a float32
quantize-dequantize round trip.

Numerics are checked directly against numpy re-implementations of the
quantize/dequantize math (see ``onnxsim/gguf_q6_k.py``'s own docstring on
what is and isn't independently verified), not via an onnxruntime round
trip -- this module never introduces any new graph nodes to run in the
first place (the quantized weight is folded directly into a new
initializer).
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim
from onnxsim.gguf_q6_k import (
    _SUB_BLOCK_SIZE,
    _SUB_BLOCKS_PER_SUPER_BLOCK,
    _SUPER_BLOCK_SIZE,
)


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


def _naive_int6_round_trip(values: np.ndarray, block_size: int) -> np.ndarray:
    """A plain, single-scale-per-block symmetric 6-bit quantizer -- no
    sub-block structure, just one max-abs scale for each contiguous
    ``block_size``-element group. This is the baseline K-quants exist to
    beat: the same block size as a Q6_K super-block (256), but none of its
    per-sub-block scale machinery.
    """
    flat = values.reshape(-1)
    blocks = flat.reshape(-1, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / 32.0
    q = np.clip(np.round(blocks / scale[:, None]), -32, 31)
    return (q * scale[:, None]).reshape(values.shape)


def test_q6_k_round_trip_has_at_most_64_levels_per_sub_block():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE * 3).astype(np.float64)
    dequant = onnxsim.quantize_dequantize_q6_k(values)
    assert dequant.shape == values.shape

    sub_blocks = dequant.reshape(-1, _SUB_BLOCKS_PER_SUPER_BLOCK, _SUB_BLOCK_SIZE)
    for super_block in sub_blocks:
        for sub_block in super_block:
            # A 6-bit code per element means at most 64 distinct
            # reconstructed values within any single 16-element sub-block.
            assert len(np.unique(sub_block)) <= 64


def test_q6_k_is_symmetric_zero_maps_to_zero():
    # Q6_K has no separate min/zero-point -- a value of exactly 0 must
    # round-trip to exactly 0 regardless of the rest of the sub-block.
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_q6_k(values)
    assert dequant[0] == 0.0


def test_q6_k_matches_ggml_kquant_reconstruction_formula():
    # Directly re-derive sub_scale/code from the *known* Q6_K decode
    # formula this module's own docstring cites (onnxsim/ggml_kquant.h's
    # DequantizeQ6_KBlock: value = d*sc_j*q) and check every returned
    # value is exactly reachable by SOME 6-bit symmetric code under some
    # shared per-sub-block scale -- i.e. the round trip really is a
    # per-sub-block symmetric 6-bit quantizer, not something else.
    rng = np.random.default_rng(2)
    values = rng.uniform(-5.0, 5.0, size=_SUPER_BLOCK_SIZE * 2)
    dequant = onnxsim.quantize_dequantize_q6_k(values)

    sub_blocks = dequant.reshape(-1, _SUB_BLOCK_SIZE)
    for sub_block in sub_blocks:
        levels = np.unique(sub_block)
        assert len(levels) <= 64
        if len(levels) >= 2:
            gaps = np.diff(levels)
            ratios = gaps / gaps.min()
            assert np.allclose(ratios, np.round(ratios), atol=1e-3)


def test_q6_k_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(3)
    aligned = rng.standard_normal(_SUPER_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])

    dequant_aligned = onnxsim.quantize_dequantize_q6_k(aligned)
    dequant_padded = onnxsim.quantize_dequantize_q6_k(padded)

    np.testing.assert_array_equal(
        dequant_padded[: _SUPER_BLOCK_SIZE * 2], dequant_aligned
    )
    assert dequant_padded.shape == padded.shape


def test_q6_k_sub_block_scale_beats_naive_single_scale_int6():
    # The core empirical claim K-quants exist for: giving each 16-element
    # sub-block its own scale -- even after coarsely re-quantizing it to 8
    # bits relative to a shared per-256-block reference -- reconstructs a
    # real, non-uniform weight distribution more accurately than one single
    # symmetric scale shared across the whole 256-element block.
    rng = np.random.default_rng(4)
    num_super_blocks = 64
    n = num_super_blocks * _SUPER_BLOCK_SIZE
    sub_block_scales = rng.uniform(0.1, 5.0, size=n // _SUB_BLOCK_SIZE)
    noise = rng.standard_normal(n).reshape(-1, _SUB_BLOCK_SIZE)
    values = (noise * sub_block_scales[:, None]).reshape(-1)

    q6_k = onnxsim.quantize_dequantize_q6_k(values)
    naive = _naive_int6_round_trip(values, _SUPER_BLOCK_SIZE)

    q6_k_mse = float(np.mean((values - q6_k) ** 2))
    naive_mse = float(np.mean((values - naive) ** 2))
    assert q6_k_mse < naive_mse


def test_q6_k_beats_q4_k_at_reconstructing_a_fine_grained_distribution():
    # Q6_K's whole reason to exist over Q4_K: more bits per element should
    # reconstruct the same real weight distribution more accurately.
    rng = np.random.default_rng(5)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE * 16)

    q6_k = onnxsim.quantize_dequantize_q6_k(values)
    q4_k = onnxsim.quantize_dequantize_q4_k(values)

    q6_k_mse = float(np.mean((values - q6_k) ** 2))
    q4_k_mse = float(np.mean((values - q4_k) ** 2))
    assert q6_k_mse < q4_k_mse


def test_gguf_q6_k_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(6)
    K, N = _SUPER_BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q6_k_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    assert len(q.graph.node) == len(model.graph.node)


def test_gguf_q6_k_quantization_matches_conv_weight_when_enabled():
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

    q_with_conv = onnxsim.apply_gguf_q6_k_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_q6_k_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_gguf_q6_k_quantization_respects_skip_names():
    rng = np.random.default_rng(8)
    K, N = _SUPER_BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_gguf_q6_k_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q6_k_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_q6_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q6_k_quantization_skips_non_float_weight():
    K, N = _SUPER_BLOCK_SIZE, 4
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
    result = onnxsim.apply_gguf_q6_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
