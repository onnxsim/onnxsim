"""Tests for ``onnxsim.apply_dsq``/``onnxsim.apply_dac`` (D2Quant's Dual-Scale
Quantizer and Deviation-Aware Correction, see ``onnxsim/d2quant.py``).

Per this repo's own numerics convention: reconstruction of the INT4 codes
this module writes is checked directly against the ONNX initializers via
numpy, with a tight *relative* tolerance -- onnxruntime's own MatMul kernel
reduction order is not bit-exact across CPU architectures, so any
onnxruntime-based check here is kept separate and loose.
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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _decode_int4_signed(t: onnx.TensorProto) -> np.ndarray:
    dims = list(t.dims)
    numel = int(np.prod(dims)) if dims else 0
    raw = np.frombuffer(t.raw_data, dtype=np.uint8)
    lo = raw & 0x0F
    hi = (raw >> 4) & 0x0F
    nibbles = np.empty(numel, dtype=np.uint8)
    nibbles[0::2] = lo[: (numel + 1) // 2]
    nibbles[1::2] = hi[: numel // 2]
    signed = nibbles.astype(np.int64)
    signed = np.where(signed >= 8, signed - 16, signed)
    return signed.reshape(dims)


# ---------------------------------------------------------------------------
# DSQ
# ---------------------------------------------------------------------------


def _swiglu_model(D=8, H=16, Dout=6, extra_up_consumer=False, gemm_down=False, seed=0):
    rng = np.random.default_rng(seed)
    w_gate = rng.standard_normal((D, H)).astype(np.float32) * 0.3
    w_up = rng.standard_normal((D, H)).astype(np.float32) * 0.3
    w_down = rng.standard_normal((H, Dout)).astype(np.float32) * 0.3
    # A few outlier-heavy columns in w_down -- DSQ's own target scenario.
    w_down[:, 0] *= 6.0

    down_body = (
        "Y = MatMul(Gated, Wdown)"
        if not gemm_down
        else "Y = Gemm<transB = 1>(Gated, WdownT)"
    )
    w_down_for_graph = w_down if not gemm_down else w_down.T.copy()
    down_init = _f32(w_down_for_graph, "WdownT" if gemm_down else "Wdown")

    extra_output = ", float[batch,{H}] UpOut".format(H=H) if extra_up_consumer else ""
    extra_line = "UpOut = Identity(UpProj)" if extra_up_consumer else ""

    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{Dout}] Y{extra_output})
        {{
          GateProj = MatMul(X, Wgate)
          UpProj = MatMul(X, Wup)
          Sig = Sigmoid(GateProj)
          Silu = Mul(GateProj, Sig)
          Gated = Mul(Silu, UpProj)
          {down_body}
          {extra_line}
        }}
        """,
        [_f32(w_gate, "Wgate"), _f32(w_up, "Wup"), down_init],
    )
    return model, w_gate, w_up, w_down


def test_apply_dsq_quantizes_down_proj_and_rescales_up_proj():
    model, w_gate, w_up, w_down = _swiglu_model(D=8, H=16, Dout=6, seed=0)
    dsq_model = onnxsim.apply_dsq(model, block_size=4, num_iterations=15)
    onnx.checker.check_model(dsq_model)

    down_node = next(
        n for n in dsq_model.graph.node if n.op_type == "MatMul" and n.output[0] == "Y"
    )
    dq_node = next(n for n in dsq_model.graph.node if n.output[0] == down_node.input[1])
    assert dq_node.op_type == "DequantizeLinear"
    attrs = {a.name: a for a in dq_node.attribute}
    assert attrs["axis"].i == 0
    assert attrs["block_size"].i == 4

    wq = next(t for t in dsq_model.graph.initializer if t.name == dq_node.input[0])
    assert wq.data_type == onnx.TensorProto.INT4
    assert list(wq.dims) == [16, 6]

    up_init = next(t for t in dsq_model.graph.initializer if t.name == "Wup")
    up_after = onnx.numpy_helper.to_array(up_init)
    assert not np.allclose(up_after, w_up)
    # Every column's rescale ratio must be a single constant down that
    # column (only the up-proj's *output channel* -- H -- is rescaled).
    ratio = up_after.astype(np.float64) / w_up.astype(np.float64)
    assert np.allclose(ratio, ratio[0:1, :], rtol=1e-5)


def test_apply_dsq_reconstruction_matches_original_weight():
    D, H, Dout, block_size = 8, 16, 6, 4
    model, _, w_up, w_down = _swiglu_model(D=D, H=H, Dout=Dout, seed=1)
    dsq_model = onnxsim.apply_dsq(model, block_size=block_size, num_iterations=15)

    down_node = next(
        n for n in dsq_model.graph.node if n.op_type == "MatMul" and n.output[0] == "Y"
    )
    dq_node = next(n for n in dsq_model.graph.node if n.output[0] == down_node.input[1])
    wq = next(t for t in dsq_model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in dsq_model.graph.initializer if t.name == dq_node.input[1])
    codes = _decode_int4_signed(wq).astype(np.float64)  # [H, Dout]
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)  # [H/block_size, Dout]
    num_blocks = H // block_size
    dequant_normalized = (
        codes.reshape(num_blocks, block_size, Dout) * scale[:, np.newaxis, :]
    ).reshape(H, Dout)

    up_init = next(t for t in dsq_model.graph.initializer if t.name == "Wup")
    up_after = onnx.numpy_helper.to_array(up_init).astype(np.float64)
    s_c_recovered = (up_after / w_up.astype(np.float64)).mean(axis=0)  # [H]

    reconstructed = dequant_normalized * s_c_recovered[:, np.newaxis]
    rel_err = np.linalg.norm(reconstructed - w_down) / np.linalg.norm(w_down)
    assert rel_err < 0.1


def test_apply_dsq_output_stays_close_to_float_via_onnxruntime():
    model, *_ = _swiglu_model(D=8, H=16, Dout=6, seed=2)
    dsq_model = onnxsim.apply_dsq(model, block_size=4, num_iterations=15)
    onnx.checker.check_model(dsq_model)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 8)).astype(np.float32)
    (float_y,) = _run(model, {"X": x}, output_names=["Y"]).values()
    (dsq_y,) = _run(dsq_model, {"X": x}, output_names=["Y"]).values()
    assert np.all(np.isfinite(dsq_y))
    assert _rel_l2(float_y, dsq_y) < 0.25


def test_apply_dsq_gemm_transb_down_proj():
    model, *_ = _swiglu_model(D=8, H=16, Dout=6, gemm_down=True, seed=4)
    dsq_model = onnxsim.apply_dsq(model, block_size=4, num_iterations=15)
    onnx.checker.check_model(dsq_model)

    dq_nodes = [n for n in dsq_model.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dq_nodes) == 1
    attrs = {a.name: a for a in dq_nodes[0].attribute}
    assert attrs["axis"].i == 1  # transB=1 stores W as [N, K] -- reduction is axis 1

    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    (float_y,) = _run(model, {"X": x}, output_names=["Y"]).values()
    (dsq_y,) = _run(dsq_model, {"X": x}, output_names=["Y"]).values()
    assert _rel_l2(float_y, dsq_y) < 0.25


def test_apply_dsq_declines_when_up_proj_has_extra_consumer():
    model, *_ = _swiglu_model(D=8, H=16, Dout=6, extra_up_consumer=True, seed=6)
    dsq_model = onnxsim.apply_dsq(model, block_size=4, num_iterations=15)
    assert dsq_model.SerializeToString() == model.SerializeToString()


def test_apply_dsq_noop_when_no_swiglu_pattern():
    model = _model(
        """
        g (float[4,16] X) => (float[4,6] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(np.random.default_rng(7).standard_normal((16, 6)), "W")],
    )
    dsq_model = onnxsim.apply_dsq(model, block_size=4)
    assert dsq_model.SerializeToString() == model.SerializeToString()


def test_apply_dsq_noop_below_opset_21():
    model, *_ = _swiglu_model(D=8, H=16, Dout=6, seed=8)
    model.opset_import[0].version = 17
    dsq_model = onnxsim.apply_dsq(model, block_size=4)
    assert dsq_model.SerializeToString() == model.SerializeToString()


# ---------------------------------------------------------------------------
# DAC
# ---------------------------------------------------------------------------


def _ln_shift_models(K=16, shift=None, with_beta=True, seed=0):
    rng = np.random.default_rng(seed)
    gamma = (np.ones(K) + rng.standard_normal(K) * 0.05).astype(np.float32)
    beta = (rng.standard_normal(K) * 0.1).astype(np.float32)
    if shift is None:
        shift = np.zeros(K, dtype=np.float32)

    beta_arg = ", Beta" if with_beta else ""
    beta_init = [_f32(beta, "Beta")] if with_beta else []

    float_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{K}] Y)
        {{
          Y = LayerNormalization<axis = -1>(X, Gamma{beta_arg})
        }}
        """,
        [_f32(gamma, "Gamma")] + beta_init,
        opset=17,
    )
    quantized_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{K}] Y)
        {{
          Xq = Add(X, Shift)
          Y = LayerNormalization<axis = -1>(Xq, Gamma{beta_arg})
        }}
        """,
        [_f32(gamma, "Gamma"), _f32(shift, "Shift")] + beta_init,
        opset=17,
    )
    return float_model, quantized_model, gamma, beta if with_beta else None


def test_apply_dac_reduces_deviation_from_shifted_layernorm_input():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[3] = 2.0
    shift[9] = -1.5
    float_model, quantized_model, gamma, beta = _ln_shift_models(
        K=K, shift=shift, seed=0
    )

    rng = np.random.default_rng(1)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]

    corrected = onnxsim.apply_dac(float_model, quantized_model, calibration_data=calib)
    onnx.checker.check_model(corrected)

    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    assert not np.allclose(beta_after, beta)
    gamma_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Gamma")
    )
    np.testing.assert_array_equal(gamma_after, gamma)  # DAC never touches gamma

    eval_x = rng.standard_normal((64, K)).astype(np.float32)
    (float_y,) = _run(float_model, {"X": eval_x}, output_names=["Y"]).values()
    (before_y,) = _run(quantized_model, {"X": eval_x}, output_names=["Y"]).values()
    (after_y,) = _run(corrected, {"X": eval_x}, output_names=["Y"]).values()
    assert _rel_l2(float_y, after_y) < _rel_l2(float_y, before_y)


def test_apply_dac_adds_bias_when_layernorm_has_none():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[2] = 3.0
    float_model, quantized_model, _, _ = _ln_shift_models(
        K=K, shift=shift, with_beta=False, seed=2
    )
    ln_node = next(
        n for n in quantized_model.graph.node if n.op_type == "LayerNormalization"
    )
    assert len(ln_node.input) == 2

    rng = np.random.default_rng(3)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]
    corrected = onnxsim.apply_dac(float_model, quantized_model, calibration_data=calib)
    onnx.checker.check_model(corrected)

    corrected_ln = next(
        n for n in corrected.graph.node if n.op_type == "LayerNormalization"
    )
    assert len(corrected_ln.input) == 3
    bias_init = next(
        t for t in corrected.graph.initializer if t.name == corrected_ln.input[2]
    )
    assert onnx.numpy_helper.to_array(bias_init)[2] != 0.0


def test_apply_dac_noop_when_no_deviation():
    K = 16
    float_model, quantized_model, _, beta = _ln_shift_models(K=K, shift=None, seed=4)

    rng = np.random.default_rng(5)
    calib = [{"X": rng.standard_normal((16, K)).astype(np.float32)} for _ in range(4)]
    corrected = onnxsim.apply_dac(float_model, quantized_model, calibration_data=calib)

    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    np.testing.assert_allclose(beta_after, beta)


def test_apply_dac_noop_when_no_layernorm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """,
        opset=17,
    )
    corrected = onnxsim.apply_dac(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert corrected.SerializeToString() == model.SerializeToString()
