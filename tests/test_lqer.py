"""Tests for ``onnxsim.apply_lqer`` (LQER, see ``onnxsim/lqer.py``) --
adds an activation-weighted low-rank correction of a quantized layer's own
existing reconstruction error, generalizing
``onnxsim.apply_low_rank_compensation``'s plain (unweighted) SVD.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.lqer import weighted_low_rank_correction

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
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


def test_weighted_correction_matches_plain_svd_when_weights_uniform():
    # A uniform channel_weights vector cannot change which subspace
    # minimizes the (now uniformly-scaled) objective -- so the resulting
    # low-rank approximation must equal the unweighted one exactly, up to
    # the shared scale canceling out algebraically.
    rng = np.random.default_rng(0)
    residual = rng.standard_normal((20, 6))
    b_plain, a_plain = weighted_low_rank_correction(residual, None, rank=3)
    b_uniform, a_uniform = weighted_low_rank_correction(
        residual, np.full(20, 2.5), rank=3
    )
    np.testing.assert_allclose(b_plain @ a_plain, b_uniform @ a_uniform, atol=1e-4)


def test_weighted_correction_prioritizes_high_weight_channels_at_low_rank():
    # Two channels each carry an equally large, orthogonal error component
    # -- an unweighted rank-1 fit is free to pick either (or a mix); a
    # heavily-weighted channel should make the weighted fit capture that
    # channel's own error much more completely than the other's.
    k, n = 2, 4
    residual = np.zeros((k, n))
    residual[0, :] = [3.0, 0.0, 0.0, 0.0]
    residual[1, :] = [0.0, 3.0, 0.0, 0.0]

    b, a = weighted_low_rank_correction(residual, np.array([100.0, 1.0]), rank=1)
    approx = b @ a
    err0 = np.linalg.norm(residual[0] - approx[0])
    err1 = np.linalg.norm(residual[1] - approx[1])
    assert err0 < err1


def test_weighted_correction_recovers_residual_exactly_at_full_rank():
    rng = np.random.default_rng(1)
    residual = rng.standard_normal((10, 6))
    weights = rng.uniform(0.1, 5.0, size=10)
    b, a = weighted_low_rank_correction(residual, weights, rank=min(10, 6))
    np.testing.assert_allclose(b @ a, residual, atol=1e-6)


def test_lqer_reduces_reconstruction_error():
    model = _matmul_model(K=64, N=16, seed=0)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": rng.standard_normal((32, 64)).astype(np.float32)}]

    lqer_model = onnxsim.apply_lqer(
        model, quant, rank=8, calibration_data=calibration_data
    )
    onnx.checker.check_model(lqer_model)

    op_types = [n.op_type for n in lqer_model.graph.node]
    assert op_types.count("MatMul") == 3  # base + the two correction matmuls

    (float_y,) = _run(model, {"X": x})
    (rtn_y,) = _run(quant, {"X": x})
    (lqer_y,) = _run(lqer_model, {"X": x})
    rtn_err = np.linalg.norm(float_y.astype(np.float64) - rtn_y.astype(np.float64))
    lqer_err = np.linalg.norm(float_y.astype(np.float64) - lqer_y.astype(np.float64))
    assert lqer_err < rtn_err


def test_lqer_beats_plain_lorc_on_activation_biased_output_error():
    # The core empirical claim: when calibration shows most of a layer's
    # real activation energy concentrated in a small subset of input
    # channels, LQER's weighted correction should reduce the *actual*
    # output error on that same activation distribution more than plain
    # (unweighted) LoRC's -- even though both start from the identical
    # INT4 base quantization and the same rank budget.
    # K must be a multiple of 32 for quantize_weight_only_int4 to match
    # this MatMul at all (see its own docstring).
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=2)
    quant = onnxsim.quantize_weight_only_int4(model)

    rng = np.random.default_rng(3)
    # Calibration/eval inputs: only the first 8 of 64 channels carry any
    # signal, the rest are near-zero -- a sharply skewed activation profile.
    active = 8
    calib_x = np.zeros((64, K), dtype=np.float32)
    calib_x[:, :active] = rng.standard_normal((64, active)).astype(np.float32) * 5.0
    eval_x = np.zeros((16, K), dtype=np.float32)
    eval_x[:, :active] = rng.standard_normal((16, active)).astype(np.float32) * 5.0

    rank = 4
    lqer_model = onnxsim.apply_lqer(
        model, quant, rank=rank, calibration_data=[{"X": calib_x}]
    )
    lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=rank)

    (float_y,) = _run(model, {"X": eval_x})
    (lqer_y,) = _run(lqer_model, {"X": eval_x})
    (lorc_y,) = _run(lorc_model, {"X": eval_x})

    lqer_err = np.linalg.norm(float_y.astype(np.float64) - lqer_y.astype(np.float64))
    lorc_err = np.linalg.norm(float_y.astype(np.float64) - lorc_y.astype(np.float64))
    assert lqer_err < lorc_err


def test_lqer_falls_back_to_plain_svd_without_calibration_activation():
    # A layer whose activation was never captured (empty calibration data,
    # so no channel weight is ever recorded for it) should still get a
    # correction -- via the same unweighted fallback
    # low_rank_compensation.py always uses.
    model = _matmul_model(K=32, N=8, seed=4)
    quant = onnxsim.quantize_weight_only_int4(model)

    lqer_model = onnxsim.apply_lqer(model, quant, rank=4, calibration_data=[])
    lorc_model = onnxsim.apply_low_rank_compensation(model, quant, rank=4)

    lqer_w = onnx.numpy_helper.to_array(lqer_model.graph.initializer[-2])
    lorc_w = onnx.numpy_helper.to_array(lorc_model.graph.initializer[-2])
    np.testing.assert_allclose(lqer_w, lorc_w, atol=1e-4)


def test_lqer_higher_rank_never_increases_error():
    model = _matmul_model(K=64, N=16, seed=6)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": rng.standard_normal((32, 64)).astype(np.float32)}]
    (float_y,) = _run(model, {"X": x})

    errors = []
    for rank in (1, 4, 8, 16):
        lqer_model = onnxsim.apply_lqer(
            model, quant, rank=rank, calibration_data=calibration_data
        )
        (y,) = _run(lqer_model, {"X": x})
        errors.append(np.linalg.norm(float_y.astype(np.float64) - y.astype(np.float64)))

    assert all(errors[i] >= errors[i + 1] - 1e-6 for i in range(len(errors) - 1))


def test_lqer_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_lqer(model, model, rank=4)
    assert result.SerializeToString() == model.SerializeToString()
