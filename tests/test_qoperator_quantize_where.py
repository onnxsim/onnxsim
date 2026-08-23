"""Tests for ``onnxsim.quantize_qoperator_where`` (the
``qoperator_quantize_where`` C++ pass) -- the ternary-select analogue of
``test_qoperator_quantize_elementwise.py``'s ``QLinearAdd``/``QLinearMul``
coverage, using ONNX Runtime's "com.microsoft" contrib op ``QLinearWhere``
instead.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _bvi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.BOOL, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_where():
    rng = np.random.default_rng(0)
    nodes = [onnx.helper.make_node("Where", ["Cond", "A", "B"], ["C"])]
    model = _model(
        nodes,
        [_bvi("Cond", [4, 8]), _vi("A", [4, 8]), _vi("B", [4, 8])],
        [_vi("C", [4, 8])],
        [],
    )

    quant = onnxsim.quantize_qoperator_where(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Where"] == 0
    assert ops["QLinearWhere"] == 1
    assert ops["QuantizeLinear"] == 2  # one for A, one for B
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    cond = rng.random((4, 8)) > 0.5
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    feeds = {"Cond": cond, "A": a, "B": b}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_where_broadcast():
    rng = np.random.default_rng(4)
    nodes = [onnx.helper.make_node("Where", ["Cond", "A", "B"], ["C"])]
    model = _model(
        nodes,
        [_bvi("Cond", [1, 8]), _vi("A", [4, 8]), _vi("B", [4, 1])],
        [_vi("C", [4, 8])],
        [],
    )

    quant = onnxsim.quantize_qoperator_where(model, num_calibration_samples=16, seed=4)
    onnx.checker.check_model(quant)
    assert _op_counts(quant)["QLinearWhere"] == 1

    cond = rng.random((1, 8)) > 0.5
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 1)).astype(np.float32)
    feeds = {"Cond": cond, "A": a, "B": b}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_constant_operand():
    # A constant operand is left alone -- it should be quantized from its
    # own static values, not force-fed through the calibration harness.
    const = _f32(np.random.default_rng(2).standard_normal((4, 8)), "B")
    nodes = [onnx.helper.make_node("Where", ["Cond", "A", "B"], ["C"])]
    model = _model(
        nodes, [_bvi("Cond", [4, 8]), _vi("A", [4, 8])], [_vi("C", [4, 8])], [const]
    )

    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_where_quantizable_tensors(model.SerializeToString())
    assert names == []

    quant = onnxsim.quantize_qoperator_where(model)
    assert _op_counts(quant)["Where"] == 1
    assert _op_counts(quant)["QLinearWhere"] == 0


def test_quantize_skips_non_float():
    nodes = [onnx.helper.make_node("Where", ["Cond", "A", "B"], ["C"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [
            _bvi("Cond", [4]),
            onnx.helper.make_tensor_value_info("A", onnx.TensorProto.INT64, [4]),
            onnx.helper.make_tensor_value_info("B", onnx.TensorProto.INT64, [4]),
        ],
        [onnx.helper.make_tensor_value_info("C", onnx.TensorProto.INT64, [4])],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    quant = onnxsim.quantize_qoperator_where(model)
    assert _op_counts(quant)["Where"] == 1
    assert _op_counts(quant)["QLinearWhere"] == 0


def test_list_qoperator_where_quantizable_tensors():
    nodes = [onnx.helper.make_node("Where", ["Cond", "A", "B"], ["C"])]
    model = _model(
        nodes,
        [_bvi("Cond", [4, 8]), _vi("A", [4, 8]), _vi("B", [4, 8])],
        [_vi("C", [4, 8])],
        [],
    )
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_where_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"A", "B", "C"}
