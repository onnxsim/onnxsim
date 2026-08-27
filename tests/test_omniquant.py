"""Tests for ``onnxsim.apply_omniquant`` (OmniQuant, see
``onnxsim/omniquant.py``) -- grid-searches a per-block Learnable Weight
Clipping ratio, then a per-channel Learnable Equivalent Transformation
(closed-form mean-shift plus a SmoothQuant-style migrated scale), on top
of an existing ``quantize_weight_only_int4``-quantized model.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
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


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _outlier_weight_calibration(K=64, num_samples=64, outlier_dims=(3, 7), seed=1):
    # A calibration set with both a nonzero per-channel mean (LET's shift
    # should help) and a couple of channels with much larger magnitude
    # (LET's scale, and LWC's clipping, should both help).
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32) + 2.0
    for c in outlier_dims:
        x[:, c] *= 20.0
    return x


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    codes = onnx.numpy_helper.to_array(wq).astype(np.float64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def test_omniquant_reduces_reconstruction_error_with_outlier_channels():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _outlier_weight_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    oq_model = onnxsim.apply_omniquant(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(oq_model)

    # LET must have actually inserted its Sub/Mul/Add -- otherwise this is
    # just re-testing LWC-only reclipping.
    assert any(n.op_type == "Sub" for n in oq_model.graph.node)
    assert any(n.op_type == "Mul" for n in oq_model.graph.node)
    assert any(n.op_type == "Add" for n in oq_model.graph.node)

    (float_y,) = _run(model, {"X": x})
    (oq_y,) = _run(oq_model, {"X": x})
    oq_err = np.linalg.norm(y_float - oq_y.astype(np.float64))
    assert oq_err < rtn_err


def test_omniquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _outlier_weight_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    oq_model = onnxsim.apply_omniquant(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(oq_model)

    (float_y,) = _run(model, {"X": x})
    (oq_y,) = _run(oq_model, {"X": x})
    assert np.all(np.isfinite(oq_y))
    assert _rel_l2(float_y, oq_y) < 0.25


def test_omniquant_never_worse_than_plain_rtn():
    # Both of OmniQuant's grid searches always include "no change" as
    # their first candidate (clip_ratio=1.0 -- identical to plain RTN's
    # own scale formula -- for LWC, alpha=0 for LET) and only replace it
    # with a strictly-better one, so on any data (including plain
    # unstructured noise, which -- unlike a single-stage search such as
    # onnxsim.apply_awq's own -- these two compounded per-block/per-layer
    # greedy searches can still find a small overfit-to-sample "gain"
    # from) the chosen reconstruction error can never be worse than
    # plain RTN's, only equal or better.
    model = _matmul_model(K=32, N=8, seed=4)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 32)).astype(np.float32)  # zero-mean, no outliers
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    oq_model = onnxsim.apply_omniquant(model, quant, calibration_data=calibration_data)

    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y_float = x.astype(np.float64) @ w_float
    w_rtn = _dequantize_int4(quant)
    rtn_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_rtn)

    (oq_y,) = _run(oq_model, {"X": x})
    oq_err = np.linalg.norm(y_float - oq_y.astype(np.float64))
    assert oq_err <= rtn_err + 1e-6


def test_omniquant_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _outlier_weight_calibration(K=K, num_samples=32, outlier_dims=(10, 50), seed=7)
    calibration_data = [{"X": x}]

    oq_model = onnxsim.apply_omniquant(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(oq_model)

    (float_y,) = _run(model, {"X": x})
    (oq_y,) = _run(oq_model, {"X": x})
    assert _rel_l2(float_y, oq_y) < 0.25


def test_omniquant_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=8)
    x = _outlier_weight_calibration(K=32, num_samples=16, outlier_dims=(1,), seed=9)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    oq_model = onnxsim.apply_omniquant(model, quant, calibration_data=calibration_data)
    wq = next(
        t for t in oq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
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


def test_omniquant_noop_when_no_int4_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_omniquant(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
