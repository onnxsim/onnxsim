"""Tests for ``onnxsim.apply_gptq`` (GPTQ, see ``onnxsim/gptq.py``) --
sequentially quantizes each INT4-quantized MatMul/Gemm layer's input
channels one at a time, propagating each channel's rounding error into the
not-yet-quantized channels via the layer's own (calibration-data-derived)
Hessian, so correlated channels can compensate for each other's error
instead of rounding independently.
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


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # GPTQ's own motivating scenario: input channels that are *correlated*
    # (here, every channel is a linear combination of a handful of latent
    # factors) -- independent per-element or per-channel rounding can't
    # compensate for one channel's error using another's, but GPTQ's
    # off-diagonal Hessian terms (which capture exactly this correlation)
    # can.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    # Fetch Wq/Ws by the DequantizeLinear node's own input names, not by
    # scanning for "some tensor of this dtype": quantize_weight_only_int4
    # never prunes the original (now-dead) float32 weight initializer, so a
    # dtype-only scan can silently grab that instead of the real scale.
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def test_gptq_reduces_reconstruction_error_with_correlated_channels():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(gptq_model)
    w_gptq = _dequantize_int4(gptq_model)
    y_gptq = x.astype(np.float64) @ w_gptq
    gptq_err = np.linalg.norm(y_float - y_gptq)

    assert gptq_err < rtn_err


def test_gptq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _correlated_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    gptq_model = onnxsim.apply_gptq(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(gptq_model)

    (float_y,) = _run(model, {"X": x})
    (gptq_y,) = _run(gptq_model, {"X": x})
    assert np.all(np.isfinite(gptq_y))
    assert _rel_l2(float_y, gptq_y) < 0.25


def test_gptq_preserves_scale_and_shape():
    model = _matmul_model(K=32, N=8, seed=4)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=5)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    quant_dq = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    before_scale = onnx.numpy_helper.to_array(
        next(t for t in quant.graph.initializer if t.name == quant_dq.input[1])
    )
    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    gptq_dq = next(n for n in gptq_model.graph.node if n.op_type == "DequantizeLinear")
    after_scale = onnx.numpy_helper.to_array(
        next(t for t in gptq_model.graph.initializer if t.name == gptq_dq.input[1])
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = next(
        t for t in gptq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    assert list(wq.dims) == [32, 8]


def test_gptq_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=6)
    x = _correlated_calibration(K=32, num_samples=16, rank=2, seed=7) * 3
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    wq = next(
        t for t in gptq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
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


def test_gptq_gemm_transb_with_small_processing_block():
    # proc_block_size smaller than K exercises the cross-block error
    # propagation path (not just the within-block one every other test
    # here incidentally covers via K <= the default proc_block_size=128).
    rng = np.random.default_rng(8)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _correlated_calibration(K=K, num_samples=32, rank=8, seed=9)
    calibration_data = [{"X": x}]

    gptq_model = onnxsim.apply_gptq(
        model, quant, calibration_data=calibration_data, proc_block_size=32
    )
    onnx.checker.check_model(gptq_model)

    (float_y,) = _run(model, {"X": x})
    (gptq_y,) = _run(gptq_model, {"X": x})
    assert _rel_l2(float_y, gptq_y) < 0.25


def test_gptq_handles_dead_input_channel():
    # A channel with zero variance in the calibration data (H's diagonal is
    # exactly 0 there) must not blow up the Hessian inversion.
    model = _matmul_model(K=32, N=8, seed=10)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=12)
    x[:, 5] = 0.0  # dead channel
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    gptq_model = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(gptq_model)

    (gptq_y,) = _run(gptq_model, {"X": x})
    assert np.all(np.isfinite(gptq_y))


def test_gptq_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_gptq(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_gptq_captures_a_shared_activation_only_once(monkeypatch):
    # Q/K/V projections reading one shared LayerNorm output is the norm in a
    # real transformer, and every calibration-driven pass in this repo probes
    # each matched layer's activation by name. Collecting those names into a
    # plain list (rather than a set) makes a shared tensor appear once per
    # consuming layer, so the capture loop appends -- and later concatenates
    # -- the same activation once per layer: N copies for N consumers, i.e.
    # an N-times-larger Hessian to build and hold.
    #
    # The duplication is invisible in the *output* (H is scaled by a positive
    # constant, and GPTQ's own column update depends only on the ratio
    # hinv[i, j] / hinv[i, i], which that constant cancels out of), so only a
    # structural check like this one catches a regression.
    k, n = 32, 8
    rng = np.random.default_rng(0)
    model = _model(
        f"""
        g (float[batch,{k}] X) => (float[batch,{n}] Y1, float[batch,{n}] Y2)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(X, W2)
        }}
        """,
        [
            _f32(rng.standard_normal((k, n)) * 0.5, "W1"),
            _f32(rng.standard_normal((k, n)) * 0.5, "W2"),
        ],
    )

    seen = []
    original = onnxsim.gptq._add_probe_outputs

    def spy(m, names):
        seen.append(list(names))
        return original(m, names)

    monkeypatch.setattr(onnxsim.gptq, "_add_probe_outputs", spy)
    quant = onnxsim.quantize_weight_only_int4(model)
    onnxsim.apply_gptq(
        model,
        quant,
        calibration_data=[{"X": rng.standard_normal((16, k)).astype(np.float32)}],
    )

    assert seen, "apply_gptq did not probe any activation"
    for names in seen:
        assert len(names) == len(set(names)), f"duplicate probe names: {names}"


def _model_3d(k=64, n=16, seed=0):
    # [batch, seq, K] -- the activation shape of essentially every real
    # transformer, since ONNX MatMul broadcasts over leading dimensions.
    rng = np.random.default_rng(seed)
    return _model(
        f"""
        g (float[batch,seq,{k}] X) => (float[batch,seq,{n}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(rng.standard_normal((k, n)) * 0.5, "W")],
    )


def _correlated_calibration_3d(k=64, batch=8, seq=16, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((batch, seq, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, k)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((batch, seq, k)).astype(np.float32) * 0.05
    return x


def test_gptq_handles_a_3d_transformer_shaped_activation():
    # A [batch, seq, K] activation used to be filtered out entirely (the
    # capture kept only ndim == 2 arrays), so apply_gptq silently returned
    # quantized_model unchanged on exactly the model shape it exists for.
    # Flattening the leading dimensions is exact -- the layer's own
    # reconstruction objective sums over the same rows either way.
    k, n = 64, 16
    model = _model_3d(k=k, n=n, seed=0)
    x = _correlated_calibration_3d(k=k, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    corrected = onnxsim.apply_gptq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(corrected)

    assert corrected.SerializeToString() != quant.SerializeToString(), (
        "apply_gptq was a no-op on a 3-D activation"
    )

    (float_y,) = _run(model, {"X": x})
    (rtn_y,) = _run(quant, {"X": x})
    (gptq_y,) = _run(corrected, {"X": x})
    assert np.all(np.isfinite(gptq_y))
    # Correlated channels are exactly the regime GPTQ's Hessian exploits, so
    # the improvement over plain round-to-nearest is large here (measured
    # ~0.11 -> ~0.02), not a marginal difference that could flip on another
    # platform.
    assert _rel_l2(float_y, gptq_y) < 0.5 * _rel_l2(float_y, rtn_y)


def test_gptq_flattening_matches_an_equivalent_2d_calibration():
    # Flattening [batch, seq, K] -> [batch * seq, K] must be *exact*, not an
    # approximation: feeding the same rows as a 2-D batch has to produce
    # byte-identical codes.
    k, n = 32, 8
    x3 = _correlated_calibration_3d(k=k, batch=4, seq=8, rank=4, seed=3)
    model_3d = _model_3d(k=k, n=n, seed=2)
    model_2d = _matmul_model(K=k, N=n, seed=2)

    q3 = onnxsim.quantize_weight_only_int4(model_3d)
    q2 = onnxsim.quantize_weight_only_int4(model_2d)
    out3 = onnxsim.apply_gptq(model_3d, q3, calibration_data=[{"X": x3}])
    out2 = onnxsim.apply_gptq(model_2d, q2, calibration_data=[{"X": x3.reshape(-1, k)}])

    w3 = next(t for t in out3.graph.initializer if t.data_type == onnx.TensorProto.INT4)
    w2 = next(t for t in out2.graph.initializer if t.data_type == onnx.TensorProto.INT4)
    assert w3.raw_data == w2.raw_data
