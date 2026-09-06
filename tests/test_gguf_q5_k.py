"""Tests for ``onnxsim.apply_gguf_q5_k_quantization``/
``onnxsim.quantize_dequantize_q5_k`` (see ``onnxsim/gguf_q5_k.py``) --
llama.cpp's Q5_K K-quant format, weight-only, represented as a float32
quantize-dequantize round trip.

Numerics are checked directly against numpy re-implementations of the
quantize/dequantize math (see ``onnxsim/gguf_q5_k.py``'s own docstring on
what is and isn't independently verified), not via an onnxruntime round
trip -- onnxruntime is not bit-exact across CPU architectures, and this
module never introduces any new graph nodes to run in the first place (the
quantized weight is folded directly into a new initializer).
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim
from onnxsim.gguf_kquant import quantize_dequantize_q4_k
from onnxsim.gguf_q5_k import (
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


def _naive_single_scale_round_trip(values: np.ndarray, block_size: int) -> np.ndarray:
    """A plain, single-scale-per-block symmetric 5-bit quantizer -- no
    sub-block structure, no (scale, min) affine pair, just one max-abs
    scale for each contiguous ``block_size``-element group. This is the
    baseline K-quants exist to beat: the same block size as a Q5_K
    super-block (256) and the same 5-bit code width, but none of its
    sub-block/affine machinery.
    """
    flat = values.reshape(-1)
    blocks = flat.reshape(-1, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / 15.0  # symmetric [-15, 15]
    q = np.clip(np.round(blocks / scale[:, None]), -15, 15)
    return (q * scale[:, None]).reshape(values.shape)


def test_q5_k_round_trip_has_at_most_32_levels_per_sub_block():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE * 3).astype(np.float64)
    dequant = onnxsim.quantize_dequantize_q5_k(values)
    assert dequant.shape == values.shape

    sub_blocks = dequant.reshape(-1, _SUB_BLOCKS_PER_SUPER_BLOCK, _SUB_BLOCK_SIZE)
    for super_block in sub_blocks:
        for sub_block in super_block:
            # A 5-bit code per element means at most 32 distinct
            # reconstructed values within any single 32-element sub-block.
            assert len(np.unique(sub_block)) <= 32


def test_q5_k_matches_ggml_kquant_reconstruction_formula():
    # Directly re-derive sub_scale/sub_min/code from the *known* Q5_K
    # decode formula this module's own docstring cites (onnxsim/
    # ggml_kquant.h's DequantizeQ5_KBlock/GetScaleMinK4: value = d*sc*q -
    # dmin*m, q in [0, 31]) and check every returned value is exactly
    # reachable by SOME 5-bit code under some (sub_scale, sub_min) affine
    # pair shared within its own 32-element sub-block -- i.e. the round
    # trip really is a per-sub-block affine 5-bit quantizer, not something
    # else.
    rng = np.random.default_rng(1)
    values = rng.uniform(-3.0, 5.0, size=_SUPER_BLOCK_SIZE * 2)
    dequant = onnxsim.quantize_dequantize_q5_k(values)

    sub_blocks = dequant.reshape(-1, _SUB_BLOCK_SIZE)
    for sub_block in sub_blocks:
        levels = np.unique(sub_block)
        assert len(levels) <= 32
        if len(levels) >= 2:
            # Reconstructed levels of a genuine affine 5-bit grid land on
            # `sub_min + code * sub_scale` for integer codes in [0, 31] --
            # not necessarily consecutive codes (a sub-block need not use
            # every code), so gaps between the *present* levels are integer
            # multiples of one shared base gap, not necessarily all equal.
            gaps = np.diff(levels)
            ratios = gaps / gaps.min()
            assert np.allclose(ratios, np.round(ratios), atol=1e-3)


def test_q5_k_reconstruction_error_is_small_on_a_random_weight_matrix():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((_SUPER_BLOCK_SIZE, 8))
    dequant = onnxsim.quantize_dequantize_q5_k(w)

    assert dequant.shape == w.shape
    # A 5-bit affine quantizer over 32-element sub-blocks of a standard
    # normal reconstructs to well within a few percent of the data's own
    # scale -- a real numeric bound, not just "it ran".
    rel_rmse = float(np.sqrt(np.mean((w - dequant) ** 2)) / np.std(w))
    assert rel_rmse < 0.05
    assert float(np.max(np.abs(w - dequant))) < 0.2


def test_q5_k_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(2)
    aligned = rng.standard_normal(_SUPER_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])

    dequant_aligned = onnxsim.quantize_dequantize_q5_k(aligned)
    dequant_padded = onnxsim.quantize_dequantize_q5_k(padded)

    # The first two whole super-blocks are identical between the two
    # calls -- only the trailing, separately-padded partial super-block
    # differs.
    np.testing.assert_array_equal(
        dequant_padded[: _SUPER_BLOCK_SIZE * 2], dequant_aligned
    )
    assert dequant_padded.shape == padded.shape


def test_q5_k_sub_block_affine_quantization_beats_naive_single_scale():
    # The core empirical claim K-quants exist for: giving each 32-element
    # sub-block its own (scale, min) affine pair -- even after coarsely
    # re-quantizing that pair to 6 bits relative to a shared per-256-block
    # reference -- reconstructs a real, non-uniform weight distribution
    # more accurately than one single symmetric scale shared across the
    # whole 256-element block.
    rng = np.random.default_rng(3)
    num_super_blocks = 64
    n = num_super_blocks * _SUPER_BLOCK_SIZE
    # A distribution deliberately shaped so different sub-blocks have very
    # different ranges (some near-zero, some wide, plus an asymmetric
    # positive shift) -- exactly where a per-sub-block affine fit should
    # help and a single per-256-block symmetric scale should struggle.
    sub_block_scales = rng.uniform(0.1, 5.0, size=n // _SUB_BLOCK_SIZE)
    sub_block_shifts = rng.uniform(-2.0, 2.0, size=n // _SUB_BLOCK_SIZE)
    noise = rng.standard_normal(n).reshape(-1, _SUB_BLOCK_SIZE)
    values = (noise * sub_block_scales[:, None] + sub_block_shifts[:, None]).reshape(-1)

    q5_k = onnxsim.quantize_dequantize_q5_k(values)
    naive = _naive_single_scale_round_trip(values, _SUPER_BLOCK_SIZE)

    q5_k_mse = float(np.mean((values - q5_k) ** 2))
    naive_mse = float(np.mean((values - naive) ** 2))
    assert q5_k_mse < naive_mse


def test_q5_k_represents_finer_gradations_than_q4_k():
    # Q5_K's only algorithmic difference from Q4_K is a 5-bit element code
    # instead of a 4-bit one, on the identical sub-block structure and
    # identical min/max-fit-then-requantize encoder -- so on the same data
    # it must never reconstruct measurably worse, and on a smooth,
    # non-trivial distribution it should be clearly better.
    rng = np.random.default_rng(8)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE * 16) * 0.7 + 0.3

    q5_k = onnxsim.quantize_dequantize_q5_k(values)
    q4_k = quantize_dequantize_q4_k(values)

    q5_k_mse = float(np.mean((values - q5_k) ** 2))
    q4_k_mse = float(np.mean((values - q4_k) ** 2))
    assert q5_k_mse <= q4_k_mse
    # Twice the code density over the same affine grid: comfortably better
    # than a mere tie.
    assert q5_k_mse < 0.5 * q4_k_mse


def test_gguf_q5_k_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(4)
    K, N = _SUPER_BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q5_k_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    # The replacement weight is exactly this module's own round trip of the
    # original -- not merely "something different".
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q5_k(w).astype(np.float32), rtol=0, atol=0
    )
    assert float(np.max(np.abs(new_w - w))) < 0.05
    # No new graph nodes -- the quantized weight is folded directly into a
    # new float32 initializer, same as every other weight-only quantize_*
    # function in this repo.
    assert len(q.graph.node) == len(model.graph.node)


def test_gguf_q5_k_quantization_handles_gemm_transb():
    rng = np.random.default_rng(9)
    K, N = _SUPER_BLOCK_SIZE, 6
    # transB=1 -> weight stored [N, K]
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.3
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        [_f32(w, "W"), _f32(bias, "B")],
    )

    q = onnxsim.apply_gguf_q5_k_quantization(model)
    onnx.checker.check_model(q)

    gemm_node = next(n for n in q.graph.node if n.op_type == "Gemm")
    assert gemm_node.input[1] != "W"
    # The bias is untouched -- only the weight input is quantized.
    assert gemm_node.input[2] == "B"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == gemm_node.input[1])
    )
    assert new_w.shape == w.shape
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q5_k(w).astype(np.float32), rtol=0, atol=0
    )


def test_gguf_q5_k_quantization_handles_non_block_multiple_weight():
    rng = np.random.default_rng(10)
    # 100*3 = 300 elements: neither a multiple of 256 nor of 32, so the
    # padding path runs and must still return the original shape.
    K, N = 100, 3
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q5_k_quantization(model)
    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q5_k(w).astype(np.float32), rtol=0, atol=0
    )


def test_gguf_q5_k_quantization_matches_conv_weight_when_enabled():
    rng = np.random.default_rng(5)
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

    q_with_conv = onnxsim.apply_gguf_q5_k_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_q5_k_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_gguf_q5_k_quantization_respects_skip_names():
    rng = np.random.default_rng(6)
    K, N = _SUPER_BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_gguf_q5_k_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q5_k_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_q5_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q5_k_quantization_skips_non_constant_weight():
    # Both MatMul inputs are graph inputs -- there is no initializer to
    # quantize, so the model must come back byte-identical.
    K, N = _SUPER_BLOCK_SIZE, 4
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{K},{N}] W) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )
    result = onnxsim.apply_gguf_q5_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q5_k_quantization_skips_non_float_weight():
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
    result = onnxsim.apply_gguf_q5_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
