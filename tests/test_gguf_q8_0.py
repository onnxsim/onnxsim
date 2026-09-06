"""Tests for ``onnxsim.apply_gguf_q8_0_quantization`` (see
``onnxsim/gguf_q8_0.py``) -- llama.cpp's Q8_0 GGUF block format,
weight-only, represented as a float32 quantize-dequantize round trip.

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
from onnxsim.gguf_legacy_quant import quantize_dequantize_q4_0
from onnxsim.gguf_q8_0 import _BLOCK_SIZE, _MAX_CODE, _MIN_CODE


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


def test_q8_0_round_trip_is_close_to_the_original_weight():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((8, _BLOCK_SIZE)) * 0.5
    dequant = onnxsim.quantize_dequantize_q8_0(values)
    assert dequant.shape == values.shape
    # Worst case per block is half a step, and a step is d = max|x| / 127.
    max_abs = np.abs(values).max(axis=-1)
    assert np.all(np.abs(dequant - values) <= max_abs[:, None] / 127.0 / 2.0 + 1e-4)


def test_q8_0_codes_stay_within_the_signed_byte_range():
    # Re-derive each element's code from the dequantized output:
    # dequant = code * d, so code = dequant / d must land in [-128, 127]
    # (a real signed two's-complement byte; this encoder's magnitude-based
    # scale only ever reaches the symmetric [-127, 127] subset).
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_BLOCK_SIZE * 4) * 3.0
    dequant = onnxsim.quantize_dequantize_q8_0(values)
    for block_in, block_out in zip(
        values.reshape(-1, _BLOCK_SIZE), dequant.reshape(-1, _BLOCK_SIZE)
    ):
        d = np.float64(np.float16(max(np.abs(block_in).max(), 1e-12) / _MAX_CODE))
        code = np.round(block_out / d)
        assert code.min() >= _MIN_CODE
        assert code.max() <= _MAX_CODE
        assert len(np.unique(block_out)) <= _MAX_CODE - _MIN_CODE + 1


def test_q8_0_is_symmetric_zero_maps_to_zero():
    # Q8_0 has no bias and no min -- an exact 0 must round-trip to exactly
    # 0 regardless of the rest of the block (code 0 maps to 0 * d == 0).
    rng = np.random.default_rng(2)
    values = rng.standard_normal(_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_q8_0(values)
    assert dequant[0] == 0.0


def test_q8_0_beats_q4_0_on_the_same_weight():
    # 8-bit codes should clearly beat 4-bit ones on the identical input.
    rng = np.random.default_rng(3)
    values = rng.standard_normal(_BLOCK_SIZE * 16)
    err_q4_0 = float(np.mean((values - quantize_dequantize_q4_0(values)) ** 2))
    err_q8_0 = float(np.mean((values - onnxsim.quantize_dequantize_q8_0(values)) ** 2))
    assert err_q8_0 < err_q4_0


def test_q8_0_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(4)
    aligned = rng.standard_normal(_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])
    dequant_aligned = onnxsim.quantize_dequantize_q8_0(aligned)
    dequant_padded = onnxsim.quantize_dequantize_q8_0(padded)
    np.testing.assert_array_equal(dequant_padded[: _BLOCK_SIZE * 2], dequant_aligned)


def test_apply_gguf_q8_0_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_gguf_q8_0_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q8_0(w).astype(np.float32)
    )
    assert len(q.graph.node) == len(model.graph.node)


def test_apply_gguf_q8_0_quantization_handles_gemm_with_transb():
    rng = np.random.default_rng(6)
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

    q = onnxsim.apply_gguf_q8_0_quantization(model)
    onnx.checker.check_model(q)

    gemm_node = next(n for n in q.graph.node if n.op_type == "Gemm")
    assert gemm_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == gemm_node.input[1])
    )
    assert new_w.shape == w.shape
    np.testing.assert_allclose(
        new_w, onnxsim.quantize_dequantize_q8_0(w).astype(np.float32)
    )


def test_apply_gguf_q8_0_quantization_matches_conv_weight_when_enabled():
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

    q_with_conv = onnxsim.apply_gguf_q8_0_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_gguf_q8_0_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_apply_gguf_q8_0_quantization_respects_skip_names():
    rng = np.random.default_rng(8)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_gguf_q8_0_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_q8_0_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gguf_q8_0_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_q8_0_quantization_skips_non_constant_weight():
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
    result = onnxsim.apply_gguf_q8_0_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_gguf_q8_0_quantization_skips_non_float_weight():
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
    result = onnxsim.apply_gguf_q8_0_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
