"""Tests for ``onnxsim.quantize_weight_only_nvfp4`` (NVIDIA NVFP4, see
``onnxsim/nvfp4_quantization.py``) -- block-wise quantization onto E2M1's
same fixed 16-value codebook :mod:`onnxsim.mx_quantization` uses, but with
a per-block E4M3 scale (not a pure power of two) plus a per-tensor FP32
global scale, represented in the ONNX graph via ordinary
Gather/Reshape/Mul (no contrib op, no opset-21 features).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.mx_quantization import MXFP4_CODEBOOK
from onnxsim.nvfp4_quantization import (
    _E4M3_POSITIVE_GRID,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    _round_to_e4m3,
)

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _dequantize_nvfp4_by_hand(model, w_name="W", block_size=16):
    wq = next(t for t in model.graph.initializer if t.name == f"{w_name}_nvfp4_q")
    ws = next(t for t in model.graph.initializer if t.name == f"{w_name}_nvfp4_scale")
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    codebook = np.asarray(MXFP4_CODEBOOK, dtype=np.float64)

    dim0, dim1 = codes.shape
    num_blocks = scale.shape[0]
    block_size_actual = dim0 // num_blocks
    assert block_size_actual == block_size
    values = codebook[codes]  # [dim0, dim1]
    scale_full = np.repeat(scale, block_size, axis=0)  # [dim0, dim1]
    return values * scale_full


def test_nvfp4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_nvfp4_block_scale_is_not_restricted_to_powers_of_two():
    # Unlike MXFP4's E8M0 scale, NVFP4's E4M3 scale has a mantissa, so a
    # generic weight's per-block scales should include values that are not
    # an exact power of two.
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)

    ws = next(t for t in q.graph.initializer if t.name == "W_nvfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64).ravel()
    log2_scale = np.log2(scale)
    assert not np.all(np.abs(log2_scale - np.round(log2_scale)) < 1e-9)


def test_nvfp4_effective_scale_is_a_rounded_e4m3_value_times_a_shared_global_scale():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)

    ws = next(t for t in q.graph.initializer if t.name == "W_nvfp4_scale")
    effective_scale = onnx.numpy_helper.to_array(ws).astype(np.float64).ravel()

    tensor_amax = np.abs(weight.astype(np.float64)).max()
    global_scale = tensor_amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    raw_block_scale = effective_scale / global_scale
    # Every raw (pre-global-scale) block scale must itself be an exact
    # element of the E4M3 grid -- that is NVFP4's whole point.
    nearest = _round_to_e4m3(raw_block_scale)
    # `effective_scale` round-tripped through the graph's float32 initializer,
    # so `raw_block_scale` (recovered by dividing back out the exact float64
    # global_scale) only matches the original float64 block_scale to float32
    # precision (~1e-7 relative) -- comfortably tighter than the E4M3 grid's
    # own spacing (as coarse as 32 at this magnitude), so this still clearly
    # distinguishes "on the grid" from "an arbitrary unrounded value".
    assert np.allclose(nearest, raw_block_scale, rtol=1e-5, atol=1e-3)
    assert np.all(raw_block_scale <= FLOAT8_E4M3_MAX + 1e-6)


def test_nvfp4_dequantized_values_match_hand_decoded_reference():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)

    w_hand = _dequantize_nvfp4_by_hand(q, block_size=16)

    codebook = np.asarray(MXFP4_CODEBOOK)
    max_gap = np.max(np.diff(codebook))
    ws = next(t for t in q.graph.initializer if t.name == "W_nvfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, 16, axis=0)
    assert np.all(
        np.abs(w_hand - weight.astype(np.float64)) <= max_gap * scale_full / 2 + 1e-6
    )


def test_nvfp4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=4)
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_nvfp4_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 128, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_nvfp4_codes_stay_in_range():
    rng = np.random.default_rng(7)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)
    wq = next(t for t in q.graph.initializer if t.name == "W_nvfp4_q")
    codes = onnx.numpy_helper.to_array(wq)
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_nvfp4_skips_non_block_divisible_k():
    model = _matmul_model(K=24, N=8, seed=8)  # 24 is not a multiple of 16
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16)
    assert q.SerializeToString() == model.SerializeToString()


def test_nvfp4_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(9)
    w_base = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((64, 4)).astype(np.float32) * 0.1
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,16] Y, float[batch,4] H)
        {
          Y = MatMul(X, W)
          H = MatMul(X, W_other)
        }
        """,
        initializer=[_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    q = onnxsim.quantize_weight_only_nvfp4(model, block_size=16, skip_names=["W_other"])
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_nvfp4_q" in names
    assert "W_other_nvfp4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_nvfp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_nvfp4(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_e4m3_grid_is_well_formed():
    assert _E4M3_POSITIVE_GRID.shape == (127,)
    assert np.all(np.diff(_E4M3_POSITIVE_GRID) > 0)  # strictly increasing, no dupes
    assert _E4M3_POSITIVE_GRID[0] == 0.0
    assert _E4M3_POSITIVE_GRID[-1] == FLOAT8_E4M3_MAX == 448.0
    assert 1.0 in _E4M3_POSITIVE_GRID  # exactly representable (exponent field 7)


def test_round_to_e4m3_is_idempotent_and_clamps():
    rng = np.random.default_rng(10)
    values = rng.uniform(0, 1000, size=1000)
    rounded = _round_to_e4m3(values)
    assert np.all(rounded <= FLOAT8_E4M3_MAX)
    # Rounding an already-representable value must be a no-op.
    assert np.allclose(_round_to_e4m3(rounded), rounded)
