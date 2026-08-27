"""Tests for ``onnxsim.workaround_ort_matmul_nbits_axis0_bug`` (see
``onnxsim/ort_matmul_nbits_workaround.py``) -- rewrites an ``axis=0``
block-quantized ``DequantizeLinear`` feeding a plain MatMul/Gemm into the
``axis=1`` shape ONNX Runtime's ``MatMulNBitsFusion`` transformer actually
supports, so it no longer silently miscomputes the layer under ONNX
Runtime's default graph optimization settings.

These tests specifically exercise *default* onnxruntime optimization
(the whole point of the workaround), unlike every other onnxsim test file,
which doesn't care about optimization level since it doesn't hit this bug.
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


def _matmul_model(K=32, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _run(model, feeds, disable_optimizations=False):
    so = ort.SessionOptions()
    if disable_optimizations:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_workaround_fixes_default_optimization_for_plain_matmul():
    K, N = 32, 16
    model = _matmul_model(K=K, N=N, seed=0)
    quant = onnxsim.quantize_weight_only_int4(model)

    dq_node = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    axis = next(a.i for a in dq_node.attribute if a.name == "axis")
    assert axis == 0  # the vulnerable case this workaround targets

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, K)).astype(np.float32)

    # The bug: same model, same weights, differs between default
    # optimization and optimizations disabled -- this is the ground truth
    # this test is guarding against, not a property of the fix itself.
    (y_buggy_default,) = _run(quant, {"X": x})
    (y_correct_reference,) = _run(quant, {"X": x}, disable_optimizations=True)
    assert not np.allclose(y_buggy_default, y_correct_reference, rtol=0, atol=1e-3)

    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(quant)
    onnx.checker.check_model(fixed)
    (y_fixed_default,) = _run(fixed, {"X": x})
    assert np.allclose(
        y_fixed_default.astype(np.float64),
        y_correct_reference.astype(np.float64),
        rtol=0,
        atol=1e-4,
    )


def test_workaround_inserts_axis1_dequantize_and_transpose():
    model = _matmul_model(K=32, N=16, seed=2)
    quant = onnxsim.quantize_weight_only_int4(model)
    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(quant)

    dq_nodes = [n for n in fixed.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dq_nodes) == 2  # original (now unused) + the new axis=1 one
    axes = [
        n for n in dq_nodes if any(a.name == "axis" and a.i == 1 for a in n.attribute)
    ]
    assert len(axes) == 1
    assert any(n.op_type == "Transpose" for n in fixed.graph.node)


def test_workaround_noop_on_gemm_transb1():
    # axis=1 already -- not affected by the bug, nothing to rewrite.
    rng = np.random.default_rng(3)
    K, N = 32, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    dq_node = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    axis = next(a.i for a in dq_node.attribute if a.name == "axis")
    assert axis == 1

    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(quant)
    assert fixed.SerializeToString() == quant.SerializeToString()


def test_workaround_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.workaround_ort_matmul_nbits_axis0_bug(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_workaround_handles_hqq_three_input_dequantize_linear():
    # HQQ's DequantizeLinear has a zero-point (Wq, Ws, Wz) -- confirm all
    # three tensors get transposed together, not just Wq/Ws.
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=4)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, K)).astype(np.float32)

    quant = onnxsim.quantize_weight_only_int4_hqq(model, block_size=32)
    dq_node = next(n for n in quant.graph.node if n.op_type == "DequantizeLinear")
    assert len(dq_node.input) == 3
    axis = next(a.i for a in dq_node.attribute if a.name == "axis")
    assert axis == 0

    (y_buggy_default,) = _run(quant, {"X": x})
    (y_correct_reference,) = _run(quant, {"X": x}, disable_optimizations=True)

    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(quant)
    onnx.checker.check_model(fixed)
    (y_fixed_default,) = _run(fixed, {"X": x})
    assert np.allclose(
        y_fixed_default.astype(np.float64),
        y_correct_reference.astype(np.float64),
        rtol=0,
        atol=1e-4,
    )


def test_workaround_applies_transparently_to_awq_refined_model():
    # AWQ only rewrites Wq/Ws' own values, not the DequantizeLinear's axis
    # -- confirm the workaround still finds and fixes it downstream.
    K, N = 32, 16
    model = _matmul_model(K=K, N=N, seed=6)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, K)).astype(np.float32)
    for c in (3, 7):
        x[:, c] *= 20.0
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    awq_model = onnxsim.apply_awq(model, quant, calibration_data=calibration_data)

    (y_buggy_default,) = _run(awq_model, {"X": x})
    (y_correct_reference,) = _run(awq_model, {"X": x}, disable_optimizations=True)

    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(awq_model)
    onnx.checker.check_model(fixed)
    (y_fixed_default,) = _run(fixed, {"X": x})
    assert np.allclose(
        y_fixed_default.astype(np.float64),
        y_correct_reference.astype(np.float64),
        rtol=0,
        atol=1e-4,
    )


def test_workaround_applies_transparently_to_omniquant_refined_model():
    # OmniQuant also only rewrites Wq/Ws' own values (plus, when its LET
    # transform helps, inserting Sub/Mul/Add nodes around the matched
    # MatMul, never touching the DequantizeLinear's own axis) -- confirm
    # the workaround still finds and fixes it downstream.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=8)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((32, K)).astype(np.float32) + 2.0
    for c in (3, 7):
        x[:, c] *= 20.0
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    oq_model = onnxsim.apply_omniquant(model, quant, calibration_data=calibration_data)

    (y_buggy_default,) = _run(oq_model, {"X": x})
    (y_correct_reference,) = _run(oq_model, {"X": x}, disable_optimizations=True)

    fixed = onnxsim.workaround_ort_matmul_nbits_axis0_bug(oq_model)
    onnx.checker.check_model(fixed)
    (y_fixed_default,) = _run(fixed, {"X": x})
    assert np.allclose(
        y_fixed_default.astype(np.float64),
        y_correct_reference.astype(np.float64),
        rtol=0,
        atol=1e-4,
    )
