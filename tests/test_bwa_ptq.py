"""Tests for ``onnxsim.apply_bwa_ptq`` (Binary Weight-Activation PTQ, see
``onnxsim/bwa_ptq.py``) -- Hessian-weighted two-scale binary EM, pushing an
ordinary dense float32 MatMul/Gemm weight down to exactly 1 sign bit + 1
group-select bit/element (uniformly, no salient-column structure or additive
residual the way ``onnxsim.billm`` uses).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model(K=64, N=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    return (
        _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{N}] Y)
            {{
              Y = MatMul(X, W)
            }}
            """,
            [_f32(weight, "W")],
        ),
        weight,
    )


def _gemm_transb_model(K=48, N=8, seed=2):
    rng = np.random.default_rng(seed)
    # transB=1 -> weight stored [N, K]
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.3
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    return (
        _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{N}] Y)
            {{
              Y = Gemm<transB=1>(X, W, B)
            }}
            """,
            [_f32(weight, "W"), _f32(bias, "B")],
        ),
        weight,
    )


def _calibration(K, num_samples=48, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def _bimodal_weight(K=64, N=8, seed=0):
    # Two clearly separated magnitude populations within each group so a
    # single flat scale (BiLLM's own non-salient path, or a naive one-scale
    # binarizer) represents the group poorly, but two scales should fit
    # each population almost exactly.
    rng = np.random.default_rng(seed)
    small = rng.standard_normal((K, N)).astype(np.float64) * 0.05
    mask = rng.random((K, N)) < 0.5
    large = rng.standard_normal((K, N)).astype(np.float64) * 2.0
    weight = np.where(mask, large, small)
    return weight.astype(np.float32)


def _decode_bwa_weight(model, w_name, orig_shape):
    prefix = f"{w_name}_bwa"
    by_name = {t.name: t for t in model.graph.initializer}
    sign = onnx.numpy_helper.to_array(by_name[f"{prefix}_sign"]).astype(np.float64)
    group_select = onnx.numpy_helper.to_array(by_name[f"{prefix}_group_select"]).astype(
        np.float64
    )
    scale0 = onnx.numpy_helper.to_array(by_name[f"{prefix}_scale0"]).astype(np.float64)
    scale1 = onnx.numpy_helper.to_array(by_name[f"{prefix}_scale1"]).astype(np.float64)
    scale_eff = scale0 + group_select * (scale1 - scale0)
    recon = sign * scale_eff
    assert recon.shape == orig_shape
    return recon, sign, group_select


def _plain_single_scale_reconstruction(w_kn, group_size):
    k, n = w_kn.shape
    recon = np.empty_like(w_kn, dtype=np.float64)
    for start in range(0, k, group_size):
        end = min(start + group_size, k)
        block = w_kn[start:end, :]
        scale = np.mean(np.abs(block))
        recon[start:end, :] = np.where(block >= 0.0, 1.0, -1.0) * scale
    return recon


def test_two_scale_reconstruction_beats_single_scale_on_bimodal_weight():
    K, N = 64, 8
    model, weight = _matmul_model(K=K, N=N, seed=0)
    weight = _bimodal_weight(K=K, N=N, seed=0)
    model.graph.initializer[0].CopyFrom(_f32(weight, "W"))

    calib = [{"X": _calibration(K, seed=1)}]
    quantized = onnxsim.apply_bwa_ptq(model, calibration_data=calib, group_size=32)

    recon, sign, group_select = _decode_bwa_weight(quantized, "W", weight.shape)
    bwa_err = np.linalg.norm(weight.astype(np.float64) - recon)

    naive_recon = _plain_single_scale_reconstruction(weight.astype(np.float64), 32)
    naive_err = np.linalg.norm(weight.astype(np.float64) - naive_recon)

    assert bwa_err < 0.6 * naive_err
    # Both populations should actually be in use (not degenerate to one).
    assert np.any(group_select == 0)
    assert np.any(group_select == 1)


def test_codes_are_in_valid_discrete_set():
    K, N = 40, 6
    model, weight = _matmul_model(K=K, N=N, seed=3)
    calib = [{"X": _calibration(K, seed=4)}]
    quantized = onnxsim.apply_bwa_ptq(model, calibration_data=calib, group_size=16)

    by_name = {t.name: t for t in quantized.graph.initializer}
    sign = onnx.numpy_helper.to_array(by_name["W_bwa_sign"])
    group_select = onnx.numpy_helper.to_array(by_name["W_bwa_group_select"])
    assert set(np.unique(sign).tolist()) <= {-1, 1}
    assert set(np.unique(group_select).tolist()) <= {0, 1}
    assert sign.dtype == np.int8
    assert group_select.dtype == np.int8


def test_scale0_never_exceeds_scale1():
    # Canonicalized so the encoding is deterministic: scale0 <= scale1
    # everywhere, group_select just picks which one applies.
    K, N = 64, 8
    model, weight = _matmul_model(K=K, N=N, seed=2)
    model.graph.initializer[0].CopyFrom(_f32(_bimodal_weight(K=K, N=N, seed=2), "W"))
    calib = [{"X": _calibration(K, seed=5)}]
    quantized = onnxsim.apply_bwa_ptq(model, calibration_data=calib, group_size=32)

    by_name = {t.name: t for t in quantized.graph.initializer}
    scale0 = onnx.numpy_helper.to_array(by_name["W_bwa_scale0"])
    scale1 = onnx.numpy_helper.to_array(by_name["W_bwa_scale1"])
    assert np.all(scale0 <= scale1 + 1e-6)


def test_end_to_end_float_closeness():
    K, N = 64, 8
    model, weight = _matmul_model(K=K, N=N, seed=5)
    calib = [{"X": _calibration(K, seed=6)}]
    quantized = onnxsim.apply_bwa_ptq(model, calibration_data=calib, group_size=32)

    x = np.random.default_rng(7).standard_normal((4, K)).astype(np.float32)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})

    # A ~1-bit-average binarizer is aggressively lossy by design -- not
    # remotely INT4-level closeness -- but should still be well within the
    # same order of magnitude as the float output, not noise.
    assert _rel_l2(float_out, quant_out) < 0.9


def test_gemm_transb_weight():
    K, N = 48, 8
    model, weight = _gemm_transb_model(K=K, N=N, seed=8)
    calib = [{"X": _calibration(K, seed=9)}]
    quantized = onnxsim.apply_bwa_ptq(model, calibration_data=calib, group_size=24)

    recon, sign, group_select = _decode_bwa_weight(quantized, "W", weight.shape)
    assert set(np.unique(sign).tolist()) <= {-1, 1}
    assert set(np.unique(group_select).tolist()) <= {0, 1}

    x = _calibration(K, num_samples=3, seed=10)
    (float_out,) = _run(model, {"X": x})
    (quant_out,) = _run(quantized, {"X": x})
    assert _rel_l2(float_out, quant_out) < 0.9


def test_noop_on_non_matching_layer():
    K, N = 32, 4
    model, weight = _matmul_model(K=K, N=N, seed=11)
    calib = [{"X": _calibration(K, seed=12)}]

    quantized = onnxsim.apply_bwa_ptq(
        model, calibration_data=calib, group_size=16, skip_names={"W"}
    )
    assert quantized.SerializeToString() == model.SerializeToString()

    # A layer whose weight isn't a plain constant 2-D float32 tensor (a
    # 1-D bias-shaped initializer here) shouldn't be touched at all.
    conv_like = _model(
        """
        g (float[1,4] X) => (float[1,4] Y)
        {
          Y = Add(X, Bias)
        }
        """,
        [_f32(np.zeros(4, dtype=np.float32), "Bias")],
    )
    out = onnxsim.apply_bwa_ptq(conv_like, calibration_data=calib)
    assert out.SerializeToString() == conv_like.SerializeToString()
