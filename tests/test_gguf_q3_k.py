"""Tests for ``onnxsim.apply_gguf_q3_k_quantization``/
``onnxsim.quantize_dequantize_q3_k`` (see ``onnxsim/gguf_q3_k.py``) --
llama.cpp's Q3_K K-quant format, weight-only, represented as a float32
quantize-dequantize round trip.

Numerics are checked directly against numpy re-implementations of the
quantize/dequantize math (see ``onnxsim/gguf_q3_k.py``'s own docstring on
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
from onnxsim.gguf_q3_k import (
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


def _naive_int3_round_trip(values: np.ndarray, block_size: int) -> np.ndarray:
    """A plain, single-scale-per-block quantizer over Q3_K's own
    asymmetric ``[-4, 3]`` code range -- no sub-block structure, just one
    max-abs scale for each contiguous ``block_size``-element group. This
    is the baseline K-quants exist to beat: the same block size as a Q3_K
    super-block (256), but none of its per-sub-block scale machinery.
    """
    flat = values.reshape(-1)
    blocks = flat.reshape(-1, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=-1), 1e-12) / 4.0
    q = np.clip(np.round(blocks / scale[:, None]), -4, 3)
    return (q * scale[:, None]).reshape(values.shape)


def test_q3_k_round_trip_has_at_most_8_levels_per_sub_block():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE * 3).astype(np.float64)
    dequant = onnxsim.quantize_dequantize_q3_k(values)
    assert dequant.shape == values.shape

    sub_blocks = dequant.reshape(-1, _SUB_BLOCKS_PER_SUPER_BLOCK, _SUB_BLOCK_SIZE)
    for super_block in sub_blocks:
        for sub_block in super_block:
            # A 3-bit code per element ({-4, ..., 3}) means at most 8
            # distinct reconstructed values within any 16-element sub-block.
            assert len(np.unique(sub_block)) <= 8


def test_q3_k_is_symmetric_zero_maps_to_zero():
    # Q3_K has no separate min/zero-point (one shared d_all, a signed
    # per-sub-block scale code, a signed element code) -- a value of
    # exactly 0 must round-trip to exactly 0 regardless of the rest of the
    # sub-block.
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_q3_k(values)
    assert dequant[0] == 0.0


def test_q3_k_element_code_range_is_asymmetric():
    # The one behaviour genuinely specific to Q3_K among its K-quant
    # siblings: ggml_kquant.h's DequantizeQ3_KBlock combines 2 low bits
    # with a high-mask bit as `low - (hmask_bit ? 0 : 4)`, so the element
    # code range is the *asymmetric* set {-4, ..., 3} -- one more level
    # below zero than above. Worth checking directly rather than trusting
    # the code: -4 * sub_scale must be reachable, +4 * sub_scale (the
    # excluded 9th, symmetric level) must not be.
    rng = np.random.default_rng(12)
    # One whole super-block laid out as 16 rows of 16, so each row is
    # exactly one sub-block.
    w = rng.uniform(-0.4, 0.4, size=(_SUB_BLOCKS_PER_SUPER_BLOCK, _SUB_BLOCK_SIZE))
    # Row 0 is the boundary sub-block: its largest-magnitude element is
    # negative (-1.0, so it lands on the -4 code) and it also holds a
    # near-identical positive twin (+0.999), which a symmetric 9-value
    # range would put on +4 but Q3_K's own range must clamp to +3.
    w[0, 0] = -1.0
    w[0, 1] = 0.999
    w[0, 2:] = rng.uniform(-0.9, 0.9, size=_SUB_BLOCK_SIZE - 2)

    dequant = onnxsim.quantize_dequantize_q3_k(w)
    assert dequant.shape == w.shape

    row = dequant[0]
    levels = np.unique(row)
    # Every reconstructed value in a sub-block is an integer multiple of
    # that sub-block's own scale, so the smallest gap between distinct
    # levels *is* that scale (row 0 does contain adjacent codes).
    sub_scale = float(np.diff(levels).min())
    codes = row / sub_scale
    np.testing.assert_allclose(codes, np.round(codes), atol=1e-6)

    codes = np.round(codes).astype(int)
    assert codes.min() == -4  # the extra level below zero is used
    assert codes.max() == 3  # ...and nothing above +3 exists
    assert np.any(np.isclose(row, -4.0 * sub_scale))
    assert not np.any(np.isclose(row, 4.0 * sub_scale))
    # +0.999 was *not* reconstructed at +4*sub_scale (~ +0.9999); it was
    # clamped down to the top real code, +3.
    assert codes[1] == 3

    # And no other sub-block escapes the range either.
    for other in dequant:
        other_levels = np.unique(other)
        if len(other_levels) < 2:
            continue
        other_scale = float(np.diff(other_levels).min())
        other_codes = other / other_scale
        assert other_codes.min() >= -4.0 - 1e-6
        assert other_codes.max() <= 3.0 + 1e-6


def test_q3_k_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(3)
    aligned = rng.standard_normal(_SUPER_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])

    dequant_aligned = onnxsim.quantize_dequantize_q3_k(aligned)
    dequant_padded = onnxsim.quantize_dequantize_q3_k(padded)

    np.testing.assert_array_equal(
        dequant_padded[: _SUPER_BLOCK_SIZE * 2], dequant_aligned
    )
    assert dequant_padded.shape == padded.shape


def test_q3_k_sub_block_scale_beats_naive_single_scale_int3():
    # The core empirical claim K-quants exist for: giving each 16-element
    # sub-block its own scale -- even after coarsely re-quantizing it
    # relative to a shared per-256-block float16 reference -- reconstructs
    # a real, non-uniform weight distribution more accurately than one
    # single scale shared across the whole 256-element block.
    rng = np.random.default_rng(4)
    num_super_blocks = 64
    n = num_super_blocks * _SUPER_BLOCK_SIZE
    sub_block_scales = rng.uniform(0.1, 5.0, size=n // _SUB_BLOCK_SIZE)
    noise = rng.standard_normal(n).reshape(-1, _SUB_BLOCK_SIZE)
    values = (noise * sub_block_scales[:, None]).reshape(-1)

    q3_k = onnxsim.quantize_dequantize_q3_k(values)
    naive = _naive_int3_round_trip(values, _SUPER_BLOCK_SIZE)

    q3_k_mse = float(np.mean((values - q3_k) ** 2))
    naive_mse = float(np.mean((values - naive) ** 2))
    assert q3_k_mse < naive_mse


def test_q3_k_is_coarser_than_q6_k():
    # Q3_K's whole reason to exist over Q6_K is size, not accuracy: fewer
    # bits per element must reconstruct the same distribution *less*
    # accurately.
    rng = np.random.default_rng(5)
    values = rng.standard_normal(_SUPER_BLOCK_SIZE * 16)

    q3_k = onnxsim.quantize_dequantize_q3_k(values)
    q6_k = onnxsim.quantize_dequantize_q6_k(values)

    q3_k_mse = float(np.mean((values - q3_k) ** 2))
    q6_k_mse = float(np.mean((values - q6_k) ** 2))
    assert q6_k_mse < q3_k_mse


def test_gguf_q3_k_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(6)
    K, N = _SUPER_BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q3_k_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    assert len(q.graph.node) == len(model.graph.node)

    # The folded initializer is exactly the module's own round trip, and a
    # plausible reconstruction of the original weight.
    expected = onnxsim.quantize_dequantize_q3_k(w).astype(np.float32)
    np.testing.assert_array_equal(new_w, expected)
    assert float(np.mean((w - new_w) ** 2)) < float(np.mean(w**2))


def test_gguf_q3_k_quantization_handles_gemm_transb():
    rng = np.random.default_rng(7)
    K, N = _SUPER_BLOCK_SIZE, 8
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

    q = onnxsim.apply_gguf_q3_k_quantization(model)
    onnx.checker.check_model(q)

    gemm_node = next(n for n in q.graph.node if n.op_type == "Gemm")
    assert gemm_node.input[1] != "W"
    assert gemm_node.input[2] == "B"  # the bias is left alone
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == gemm_node.input[1])
    )
    assert new_w.shape == w.shape
    np.testing.assert_array_equal(
        new_w, onnxsim.quantize_dequantize_q3_k(w).astype(np.float32)
    )


def test_gguf_q3_k_quantization_matches_conv_weight_when_enabled():
    rng = np.random.default_rng(8)
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

    q_with_conv = onnxsim.apply_gguf_q3_k_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_q3_k_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_gguf_q3_k_quantization_respects_skip_names():
    rng = np.random.default_rng(9)
    K, N = _SUPER_BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_gguf_q3_k_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q3_k_quantization_skips_non_constant_weight():
    K, N = _SUPER_BLOCK_SIZE, 4
    # W is a graph *input*, not an initializer -- there is no constant
    # weight to quantize, so the pass must leave the model alone.
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{K},{N}] W) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )
    result = onnxsim.apply_gguf_q3_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q3_k_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_q3_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_gguf_q3_k_quantization_skips_non_float_weight():
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
    result = onnxsim.apply_gguf_q3_k_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
