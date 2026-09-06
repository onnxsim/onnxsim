"""Tests for ``onnxsim.apply_gptaq`` (GPTAQ, see ``onnxsim/gptaq.py``) --
GPTQ's own per-column algorithm, applied to a small closed-form correction
of the target weight that accounts for the gap between a layer's *true*
(float-model) input activation and the *actual* (already partly quantized)
input activation ``quantized_model`` will really see at inference time.
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


def _upstream_corrupted_model(K0=32, N1=8, corruption=None, seed=0):
    # A layer's own true float input is X; `Corruption` (a fixed, chosen
    # constant, not an emergent side effect of some earlier layer's own
    # INT4 rounding) simulates whatever upstream layers' quantization
    # already did to that input by the time it reaches this one. Passing
    # `corruption=None` (the float model) makes `Y1 == X` exactly;
    # passing a real array (the "quantized_model" input) makes `Y1` the
    # *actual*, already-corrupted signal this layer really receives.
    # Isolating the corruption in a plain `Add` (not a second real INT4
    # layer) keeps the gap between the two models' `Y1` a deterministic,
    # exactly-controlled quantity, not one dependent on some upstream
    # layer's own INT4 rounding, which can differ from platform to
    # platform (different onnxruntime kernels/BLAS backends can tip a
    # marginal rounding decision either way -- exactly what made an
    # earlier version of this test flaky across CI runners).
    rng = np.random.default_rng(seed)
    w2 = rng.standard_normal((K0, N1)).astype(np.float32) * 0.5
    corruption = (
        np.zeros(K0, dtype=np.float32)
        if corruption is None
        else corruption.astype(np.float32)
    )
    return _model(
        f"""
        g (float[batch,{K0}] X) => (float[batch,{N1}] Y2)
        {{
          Y1 = Add(X, Corruption)
          Y2 = MatMul(Y1, W2)
        }}
        """,
        [_f32(w2, "W2"), _f32(corruption, "Corruption")],
    )


def _correlated_calibration(K, num_samples=64, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _full_rank_calibration(K, num_samples=96, seed=1):
    # Deliberately *not* _correlated_calibration's own low-rank-plus-noise
    # signal: that shape gives GPTQ's own Hessian (H = X^T X) a condition
    # number in the hundreds of thousands (rank-6 structure inside a
    # 32-dimensional space), which makes _inverse_hessian_cholesky's
    # Cholesky factorization genuinely ill-conditioned -- exactly the
    # regime where different BLAS/LAPACK backends (observed: ARM vs
    # x86_64) can round the last few bits differently and flip which
    # side of an INT4 rounding boundary a marginal weight lands on. A
    # plain full-rank Gaussian keeps H's condition number in the tens,
    # not hundreds of thousands (checked directly with np.linalg.cond
    # while designing this test), so GPTQ's and GPTAQ's shared linear
    # algebra stays numerically well-behaved -- and reproducibly
    # different between the two -- on every platform.
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def _int4_weight(model, node_output_name):
    # Fetch Wq by the MatMul/Gemm node's own DequantizeLinear input, not by
    # scanning for "some INT4 tensor" or guessing a name from the original
    # weight's own name: quantize_weight_only_int4 does not derive the new
    # initializer's name from the original weight's name at all, and never
    # prunes the original (now-dead) float32 weight initializer either --
    # same convention test_gptq.py's own _dequantize_int4 helper follows.
    node = next(
        n
        for n in model.graph.node
        if n.op_type in ("MatMul", "Gemm") and n.output[0] == node_output_name
    )
    dq_node = next(
        n
        for n in model.graph.node
        if n.op_type == "DequantizeLinear" and n.output[0] == node.input[1]
    )
    return next(t for t in model.graph.initializer if t.name == dq_node.input[0])


def test_gptaq_matches_gptq_exactly_with_no_upstream_quantization():
    # A single layer has no upstream corruption to correct for: the
    # quantized model's own activation at its input *is* the float model's
    # activation (both are simply the graph's own input X), so delta_x is
    # exactly zero, GPTAQ's shift is exactly zero, and its output must be
    # bit-for-bit identical to plain GPTQ's.
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    gptaq_model = onnxsim.apply_gptaq(model, quant, calibration_data=calibration_data)

    wq_gptq = _int4_weight(gptq_model, "Y")
    wq_gptaq = _int4_weight(gptaq_model, "Y")
    assert wq_gptq.raw_data == wq_gptaq.raw_data


def test_gptaq_beats_gptq_once_an_upstream_layer_is_already_corrected():
    # GPTAQ's whole point only shows up once `quantized_model` reflects a
    # real upstream correction: `float_model`'s own Y1 is exactly X, but
    # `quantized_model`'s own Y1 is X plus a fixed, deliberately large
    # `Corruption` (see `_upstream_corrupted_model`'s own docstring for why
    # this is injected via a plain Add rather than a second real INT4
    # layer). GPTQ recalibrates the MatMul against the *float* model's Y1
    # -- not what it actually receives at inference (the corrupted Y1).
    # GPTAQ recalibrates against `quantized_model`'s own actual Y1, which
    # is exactly what the layer will really see, so it should reconstruct
    # the network's true end-to-end output more closely. Verified (see the
    # commit history of this test) to hold with a comfortable margin
    # (GPTAQ's error stays well under half of GPTQ's) across hundreds of
    # independent weight/corruption/calibration seed combinations, not
    # just the one fixed below -- unlike two earlier versions of this
    # test, which each hit a *different* platform-specific exact tie in
    # CI (chaining two real INT4 layers, whose activation gap is itself a
    # by-product of discrete rounding; then, even after fixing that,
    # using _correlated_calibration's own low-rank calibration signal,
    # whose ill-conditioned Hessian made the column algorithm's shared
    # Cholesky factorization sensitive to which BLAS backend a given
    # platform happens to use). Uses `_full_rank_calibration`, not
    # `_correlated_calibration`, specifically to avoid the latter.
    K0, N1 = 32, 8
    corruption = np.random.default_rng(42).standard_normal(K0) * 0.6
    float_model = _upstream_corrupted_model(K0=K0, N1=N1, corruption=None, seed=0)
    corrupted_model = _upstream_corrupted_model(
        K0=K0, N1=N1, corruption=corruption, seed=0
    )
    x = _full_rank_calibration(K=K0, num_samples=96, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(corrupted_model)
    final_gptq = onnxsim.apply_gptq(
        float_model, quant, calibration_data=calibration_data
    )
    final_gptaq = onnxsim.apply_gptaq(
        float_model, quant, calibration_data=calibration_data
    )
    onnx.checker.check_model(final_gptq)
    onnx.checker.check_model(final_gptaq)

    (float_y,) = _run(float_model, {"X": x})
    (gptq_y,) = _run(final_gptq, {"X": x})
    (gptaq_y,) = _run(final_gptaq, {"X": x})
    assert np.all(np.isfinite(gptaq_y))
    assert _rel_l2(float_y, gptaq_y) < _rel_l2(float_y, gptq_y)


def test_gptaq_preserves_scale_and_shape():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    quant_dq = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    before_scale = onnx.numpy_helper.to_array(
        next(t for t in quant.graph.initializer if t.name == quant_dq.input[1])
    )
    gptaq_model = onnxsim.apply_gptaq(model, quant, calibration_data=calibration_data)
    gptaq_dq = next(
        n for n in gptaq_model.graph.node if n.op_type == "DequantizeLinear"
    )
    after_scale = onnx.numpy_helper.to_array(
        next(t for t in gptaq_model.graph.initializer if t.name == gptaq_dq.input[1])
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = _int4_weight(gptaq_model, "Y")
    assert list(wq.dims) == [32, 8]


def test_gptaq_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=6)
    x = _correlated_calibration(K=32, num_samples=16, rank=2, seed=7) * 3
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    gptaq_model = onnxsim.apply_gptaq(model, quant, calibration_data=calibration_data)
    wq = _int4_weight(gptaq_model, "Y")
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    assert np.all(codes >= -7) and np.all(codes <= 7)


def test_gptaq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _correlated_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    gptaq_model = onnxsim.apply_gptaq(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(gptaq_model)

    (float_y,) = _run(model, {"X": x})
    (gptaq_y,) = _run(gptaq_model, {"X": x})
    assert np.all(np.isfinite(gptaq_y))
    assert _rel_l2(float_y, gptaq_y) < 0.25


def test_gptaq_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gptaq(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
