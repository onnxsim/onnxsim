"""Tests for ``onnxsim.apply_iq4_nl_quantization``/
``onnxsim.quantize_dequantize_iq4_nl`` (see ``onnxsim/iq4_nl.py``) --
llama.cpp's IQ4_NL fixed-codebook 4-bit format, weight-only, represented as
a float32 quantize-dequantize round trip.

Numerics are checked directly against numpy re-implementations of the
quantize/dequantize math, not via an onnxruntime round trip -- onnxruntime
is not bit-exact across CPU architectures, and this module never introduces
any new graph nodes to run in the first place (the quantized weight is
folded directly into a new initializer).
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim
from onnxsim.iq4_nl import (
    _BLOCK_SIZE,
    IQ4_NL_CODEBOOK,
    _lloyd_max_gaussian_codebook,
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


def _naive_uniform_int4_round_trip(values: np.ndarray, block_size: int) -> np.ndarray:
    """A plain, single-scale-per-block, *uniform*-grid symmetric INT4
    quantizer -- 16 evenly spaced levels in ``[-scale, scale]`` instead of
    IQ4_NL's non-uniform, distribution-matched 16-level codebook. This is
    the baseline a non-linear codebook is supposed to beat: same bit width,
    same block size, same "no calibration data" property, just a linear
    instead of non-uniform grid.
    """
    flat = values.reshape(-1)
    blocks = flat.reshape(-1, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / 7.0  # symmetric [-7, 7]
    q = np.clip(np.round(blocks / scale[:, None]), -7, 7)
    return (q * scale[:, None]).reshape(values.shape)


def test_codebook_is_reproducible_from_its_own_derivation():
    # This is the crux of the honesty requirement this module's docstring
    # spells out: IQ4_NL_CODEBOOK is not a hardcoded/recalled magic
    # constant -- it is exactly what _lloyd_max_gaussian_codebook() itself
    # computes, independently re-derived here.
    rederived = _lloyd_max_gaussian_codebook()
    np.testing.assert_allclose(rederived, IQ4_NL_CODEBOOK, atol=1e-9)


def test_codebook_has_16_sorted_symmetric_levels_normalized_to_one():
    codebook = np.asarray(IQ4_NL_CODEBOOK)
    assert codebook.shape == (16,)
    assert np.all(np.diff(codebook) > 0)  # strictly sorted ascending
    np.testing.assert_allclose(codebook, -codebook[::-1], atol=1e-9)  # symmetric
    assert np.max(np.abs(codebook)) == 1.0


def test_codebook_is_non_uniform_denser_near_zero():
    # The entire point of a "non-linear" codebook: levels should be packed
    # more tightly near 0 (where a Gaussian's density is highest) than at
    # the tails, unlike a uniform grid where every gap is identical.
    codebook = np.asarray(IQ4_NL_CODEBOOK)
    gaps = np.diff(codebook)
    edge_gap = gaps[0]
    center_gap = gaps[len(gaps) // 2]
    assert center_gap < edge_gap
    # Gaps should be (anti-)symmetric and monotonically shrink toward the
    # center from either edge.
    half = gaps[: len(gaps) // 2]
    assert np.all(np.diff(half) < 0)


def test_iq4_nl_round_trip_has_at_most_16_levels_per_block():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(_BLOCK_SIZE * 5).astype(np.float64)
    dequant = onnxsim.quantize_dequantize_iq4_nl(values)
    assert dequant.shape == values.shape

    blocks = dequant.reshape(-1, _BLOCK_SIZE)
    for block in blocks:
        assert len(np.unique(block)) <= 16


def test_iq4_nl_reconstructed_values_are_exactly_scale_times_a_codebook_entry():
    rng = np.random.default_rng(1)
    values = rng.uniform(-3.0, 3.0, size=_BLOCK_SIZE * 4)
    dequant = onnxsim.quantize_dequantize_iq4_nl(values)

    codebook = np.asarray(IQ4_NL_CODEBOOK)
    max_abs_codebook = np.max(np.abs(codebook))
    blocks_in = values.reshape(-1, _BLOCK_SIZE)
    blocks_out = dequant.reshape(-1, _BLOCK_SIZE)
    for block_in, block_out in zip(blocks_in, blocks_out):
        scale = max(np.abs(block_in).max(), 1e-12) / max_abs_codebook
        ratios = block_out / scale
        # Every reconstructed element, divided by its own block's scale,
        # must land on (very nearly) one of the 16 fixed codebook values.
        nearest_codebook_dist = np.min(
            np.abs(ratios[:, None] - codebook[None, :]), axis=1
        )
        assert np.all(nearest_codebook_dist < 1e-6)


def test_iq4_nl_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(2)
    aligned = rng.standard_normal(_BLOCK_SIZE * 3)
    padded = np.concatenate([aligned, rng.standard_normal(10)])

    dequant_aligned = onnxsim.quantize_dequantize_iq4_nl(aligned)
    dequant_padded = onnxsim.quantize_dequantize_iq4_nl(padded)

    np.testing.assert_array_equal(dequant_padded[: _BLOCK_SIZE * 3], dequant_aligned)
    assert dequant_padded.shape == padded.shape


def test_iq4_nl_all_zero_block_round_trips_to_near_zero():
    # The codebook has no exact-zero level (like real IQ4_NL, whose 16
    # entries are also all nonzero -- see this module's own docstring), so
    # an all-zero block's scale floors to the epsilon guard and its
    # reconstruction lands on the nearest-to-zero codebook entry times that
    # epsilon scale -- vanishingly small, but not bit-exact zero.
    values = np.zeros(_BLOCK_SIZE * 2)
    dequant = onnxsim.quantize_dequantize_iq4_nl(values)
    np.testing.assert_allclose(dequant, values, atol=1e-10)


def test_iq4_nl_non_uniform_codebook_beats_naive_uniform_int4_on_gaussian_data():
    # The core empirical claim a non-linear codebook exists for: matching a
    # 16-level codebook's own shape to a roughly-Gaussian weight
    # distribution reconstructs it more accurately than an evenly-spaced
    # uniform 4-bit grid at the same bit width and block size.
    rng = np.random.default_rng(3)
    num_blocks = 256
    values = rng.standard_normal(num_blocks * _BLOCK_SIZE)

    iq4_nl = onnxsim.quantize_dequantize_iq4_nl(values)
    naive = _naive_uniform_int4_round_trip(values, _BLOCK_SIZE)

    iq4_nl_mse = float(np.mean((values - iq4_nl) ** 2))
    naive_mse = float(np.mean((values - naive) ** 2))
    assert iq4_nl_mse < naive_mse


def test_iq4_nl_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE * 2, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_iq4_nl_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    # No new graph nodes -- the quantized weight is folded directly into a
    # new float32 initializer, same as every other weight-only quantize_*
    # function in this repo.
    assert len(q.graph.node) == len(model.graph.node)
    # The original initializer is left in the graph, unused -- matching
    # apply_gguf_q4_k_quantization's own established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_iq4_nl_quantization_matches_conv_weight_when_enabled():
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

    q_with_conv = onnxsim.apply_iq4_nl_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_iq4_nl_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_iq4_nl_quantization_respects_skip_names():
    rng = np.random.default_rng(6)
    K, N = _BLOCK_SIZE * 2, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_iq4_nl_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_iq4_nl_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_iq4_nl_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_iq4_nl_quantization_skips_non_float_weight():
    K, N = _BLOCK_SIZE * 2, 4
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
    result = onnxsim.apply_iq4_nl_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
