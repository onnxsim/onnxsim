"""Tests for ``onnxsim.apply_adaround`` (AIMET's Adaptive Rounding, see
``onnxsim/adaround.py``) -- optimizes each INT4-quantized MatMul/Gemm
layer's own per-element rounding decision (floor vs. ceil) to minimize that
layer's real reconstruction error, instead of the round-to-nearest every
``quantize_weight_only_int4`` layer starts out with.
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


def _matmul_matmul_int4_models(K=64, N=16, batch=4, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    float_model = _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    return float_model, quant_model


def _dequantize_int4(quant_model):
    """Decodes the DequantizeLinear(Wq, Ws)-fed MatMul/Gemm's weight in
    ``quant_model`` back to a dense float array, using onnxruntime itself
    (so this stays independent of adaround.py's own internal math)."""
    # Feed a zero/one probe matrix through the model's own DequantizeLinear
    # node isn't directly possible without extracting it into its own
    # session, so instead this decodes by hand from the initializer bytes.
    dq_node = next(n for n in quant_model.graph.node if n.op_type == "DequantizeLinear")
    # Fetch Wq/Ws by the DequantizeLinear node's own input names, not by
    # scanning for "some tensor of this dtype": quantize_weight_only_int4
    # never prunes the original (now-dead) float32 weight initializer, so a
    # dtype-only scan can silently grab that instead of the real scale.
    wq = next(t for t in quant_model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in quant_model.graph.initializer if t.name == dq_node.input[1])
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
    scale_full = scale_full[tuple(slicer)]
    return codes * scale_full


def test_adaround_reduces_reconstruction_error_vs_round_to_nearest():
    float_model, quant_model = _matmul_matmul_int4_models(K=64, N=16, batch=32, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    w_float = onnx.numpy_helper.to_array(float_model.graph.initializer[0]).astype(
        np.float64
    )
    w_rtn = _dequantize_int4(quant_model)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    adaround_model = onnxsim.apply_adaround(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_iterations=200,
    )
    w_ada = _dequantize_int4(adaround_model)
    y_ada = x.astype(np.float64) @ w_ada
    ada_err = np.linalg.norm(y_float - y_ada)

    assert ada_err < rtn_err


def test_adaround_output_stays_close_to_float_via_onnxruntime():
    float_model, quant_model = _matmul_matmul_int4_models(K=64, N=16, batch=16, seed=3)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=200
    )
    onnx.checker.check_model(adaround_model)

    (float_y,) = _run(float_model, {"X": x})
    (ada_y,) = _run(adaround_model, {"X": x})
    assert np.all(np.isfinite(ada_y))
    assert _rel_l2(float_y, ada_y) < 0.25


def test_adaround_preserves_scale_and_shape():
    float_model, quant_model = _matmul_matmul_int4_models(K=32, N=8, seed=5)
    rng = np.random.default_rng(6)
    calibration_data = [{"X": rng.standard_normal((4, 32)).astype(np.float32)}]

    quant_dq = next(
        n for n in quant_model.graph.node if n.op_type == "DequantizeLinear"
    )
    before_scale = onnx.numpy_helper.to_array(
        next(t for t in quant_model.graph.initializer if t.name == quant_dq.input[1])
    )
    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=50
    )
    ada_dq = next(
        n for n in adaround_model.graph.node if n.op_type == "DequantizeLinear"
    )
    after_scale = onnx.numpy_helper.to_array(
        next(t for t in adaround_model.graph.initializer if t.name == ada_dq.input[1])
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = next(
        t
        for t in adaround_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
    )
    assert list(wq.dims) == [32, 8]


def test_adaround_codes_stay_in_range():
    float_model, quant_model = _matmul_matmul_int4_models(K=32, N=8, seed=7)
    rng = np.random.default_rng(8)
    calibration_data = [{"X": rng.standard_normal((4, 32)).astype(np.float32) * 3}]

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=100
    )
    wq = next(
        t
        for t in adaround_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
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


def test_adaround_gemm_transb_with_bias():
    rng = np.random.default_rng(9)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _model(
        f"""
        g (float[8,{K}] X) => (float[8,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    onnx.checker.check_model(quant_model)

    x = rng.standard_normal((8, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=150
    )
    onnx.checker.check_model(adaround_model)

    (float_y,) = _run(float_model, {"X": x})
    (ada_y,) = _run(adaround_model, {"X": x})
    assert _rel_l2(float_y, ada_y) < 0.25


def test_adaround_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_adaround(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def _model_3d(K=64, N=16, seed=0):
    # [batch, seq, K] -- the activation shape of essentially every real
    # transformer, since ONNX MatMul broadcasts over leading dimensions.
    rng = np.random.default_rng(seed)
    return _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(rng.standard_normal((K, N)).astype(np.float32) * 0.5, "W")],
    )


def _correlated_calibration_3d(K=64, batch=8, seq=16, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((batch, seq, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((batch, seq, K)).astype(np.float32) * 0.05
    return x.astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adaround_handles_a_3d_transformer_shaped_activation(seed):
    # A [batch, seq, K] activation used to be filtered out entirely (the
    # capture kept only ndim == 2 arrays), so apply_adaround silently
    # returned quantized_model unchanged on exactly the model shape it
    # exists for. Flattening the leading dimensions is exact -- the layer's
    # own reconstruction objective sums over the same rows either way.
    float_model = _model_3d(K=64, N=16, seed=seed)
    x = _correlated_calibration_3d(K=64, seed=seed + 100)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=[{"X": x}]
    )
    onnx.checker.check_model(adaround_model)
    assert adaround_model.SerializeToString() != quant_model.SerializeToString(), (
        "apply_adaround was a no-op on a 3-D activation"
    )

    (float_y,) = _run(float_model, {"X": x})
    (rtn_y,) = _run(quant_model, {"X": x})
    (ada_y,) = _run(adaround_model, {"X": x})
    assert np.all(np.isfinite(ada_y))
    # Correlated channels are the regime AdaRound's own per-layer objective
    # exploits, so the improvement over plain round-to-nearest is large
    # here (measured ~0.10 -> ~0.035 across seeds), not a marginal
    # difference that could flip on another platform.
    assert _rel_l2(float_y, ada_y) < 0.7 * _rel_l2(float_y, rtn_y)


def test_adaround_flattening_matches_an_equivalent_2d_calibration():
    # Flattening [batch, seq, K] -> [batch * seq, K] must be *exact*, not an
    # approximation: feeding the same rows as a 2-D batch has to produce
    # byte-identical INT4 codes.
    K, N = 32, 8
    weight = (np.random.default_rng(7).standard_normal((K, N)) * 0.5).astype(np.float32)
    model_3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    model_2d = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    x3 = _correlated_calibration_3d(K=K, batch=4, seq=8, rank=4, seed=11)

    out3 = onnxsim.apply_adaround(
        model_3d,
        onnxsim.quantize_weight_only_int4(model_3d),
        calibration_data=[{"X": x3}],
    )
    out2 = onnxsim.apply_adaround(
        model_2d,
        onnxsim.quantize_weight_only_int4(model_2d),
        calibration_data=[{"X": x3.reshape(-1, K)}],
    )
    codes3 = next(
        t for t in out3.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    codes2 = next(
        t for t in out2.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    assert codes3.raw_data == codes2.raw_data
