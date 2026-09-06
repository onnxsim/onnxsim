"""Tests for ``onnxsim.apply_fp6_llm_quantization`` (see
``onnxsim/fp6_llm.py``) -- FP6-LLM's 6-bit floating-point weight-only
quantization, represented as a float32 quantize-dequantize round trip.

Numerics are checked directly against numpy/ml_dtypes re-implementations
of the quantize/dequantize math, not via an onnxruntime round trip -- this
module never introduces any new graph nodes (the quantized weight is
folded directly into a new initializer).
"""

import ml_dtypes
import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.fp6_llm import _BLOCK_SIZE, _FP6_MAX


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


@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_max_representable_matches_hardcoded_constant(fmt):
    # Reproduces the dense-sweep verification this module's own docstring
    # describes -- the hardcoded _FP6_MAX constants are not recalled/assumed,
    # they were found this way and this test guards against silent drift
    # (e.g. an ml_dtypes version change).
    dtype = {"e3m2": ml_dtypes.float6_e3m2fn, "e2m3": ml_dtypes.float6_e2m3fn}[fmt]
    sweep = np.linspace(0, 64, 200001).astype(np.float32)
    cast = sweep.astype(dtype).astype(np.float32)
    finite_max = float(np.max(cast[np.isfinite(cast)]))
    assert finite_max == pytest.approx(_FP6_MAX[fmt])


@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_a_block_at_the_format_max_round_trips_near_exactly(fmt):
    # Every element already at the block's own chosen scale target
    # (max(|block|) == _FP6_MAX[fmt] after rescaling) round-trips exactly,
    # since _FP6_MAX[fmt] is itself an exactly representable FP6 value.
    rng = np.random.default_rng(0)
    values = rng.uniform(-1.0, 1.0, size=_BLOCK_SIZE)
    values[0] = 1.0  # forces max(|block|) == 1.0 after scaling by itself
    dequant = onnxsim.quantize_dequantize_fp6(values, fmt=fmt)
    assert dequant.shape == values.shape
    np.testing.assert_allclose(dequant[0], values[0], rtol=1e-5)


@pytest.mark.parametrize("fmt", ["e3m2", "e2m3"])
def test_zero_maps_to_zero(fmt):
    rng = np.random.default_rng(1)
    values = rng.standard_normal(_BLOCK_SIZE)
    values[0] = 0.0
    dequant = onnxsim.quantize_dequantize_fp6(values, fmt=fmt)
    assert dequant[0] == 0.0


def test_fp6_beats_naive_int4_on_a_long_tailed_block():
    # The paper's central motivation: a floating-point grid represents a
    # long-tailed (mixed small-and-large magnitude) distribution better
    # than a fixed-step integer grid at a comparable bit budget.
    rng = np.random.default_rng(2)
    small = rng.standard_normal(_BLOCK_SIZE - 4) * 0.05
    large = np.array([3.0, -3.5, 4.0, -2.5])
    values = np.concatenate([small, large])

    fp6 = onnxsim.quantize_dequantize_fp6(values, fmt="e3m2")

    scale = np.maximum(np.abs(values).max(), 1e-12) / 7.0
    naive_int4 = np.clip(np.round(values / scale), -7, 7) * scale

    err_fp6 = float(np.mean((values - fp6) ** 2))
    err_int4 = float(np.mean((values - naive_int4) ** 2))
    assert err_fp6 < err_int4


def test_e3m2_has_wider_range_than_e2m3():
    assert _FP6_MAX["e3m2"] > _FP6_MAX["e2m3"]


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        onnxsim.quantize_dequantize_fp6(np.zeros(_BLOCK_SIZE), fmt="e5m0")


def test_padding_does_not_change_already_aligned_values():
    rng = np.random.default_rng(3)
    aligned = rng.standard_normal(_BLOCK_SIZE * 2)
    padded = np.concatenate([aligned, rng.standard_normal(10)])
    dequant_aligned = onnxsim.quantize_dequantize_fp6(aligned)
    dequant_padded = onnxsim.quantize_dequantize_fp6(padded)
    np.testing.assert_array_equal(dequant_padded[: _BLOCK_SIZE * 2], dequant_aligned)


def test_apply_fp6_llm_quantization_replaces_matmul_weight():
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_fp6_llm_quantization(model)
    onnx.checker.check_model(q)

    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert new_w.shape == w.shape
    assert not np.array_equal(new_w, w)
    assert len(q.graph.node) == len(model.graph.node)


def test_apply_fp6_llm_quantization_matches_conv_weight_when_enabled():
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

    q_with_conv = onnxsim.apply_fp6_llm_quantization(model, include_conv=True)
    conv_node = next(n for n in q_with_conv.graph.node if n.op_type == "Conv")
    assert conv_node.input[1] != "W"

    q_without_conv = onnxsim.apply_fp6_llm_quantization(model, include_conv=False)
    assert q_without_conv.SerializeToString() == model.SerializeToString()


def test_apply_fp6_llm_quantization_respects_skip_names():
    rng = np.random.default_rng(6)
    K, N = _BLOCK_SIZE, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_fp6_llm_quantization(model, skip_names={"W"})
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_fp6_llm_quantization_noop_when_no_matmul_gemm_conv_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_fp6_llm_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_apply_fp6_llm_quantization_skips_non_float_weight():
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
    result = onnxsim.apply_fp6_llm_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
