"""Tests for ``onnxsim.quantize_qoperator_softmax`` (the
``qoperator_quantize_softmax`` C++ pass) -- the reduction-axis analogue of
``test_qoperator_quantize_activation.py``'s ``QLinearSigmoid``/
``QLinearLeakyRelu`` coverage, using ONNX Runtime's "com.microsoft" contrib
op ``QLinearSoftmax`` instead.
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


def test_quantize_softmax_default_axis():
    rng = np.random.default_rng(0)
    nodes = [onnx.helper.make_node("Softmax", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=0
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Softmax"] == 0
    assert ops["QLinearSoftmax"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearSoftmax")
    axis = next(a.i for a in qlop.attribute if a.name == "axis")
    opset_attr = next(a.i for a in qlop.attribute if a.name == "opset")
    assert axis == -1
    assert opset_attr == 13

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_softmax_explicit_axis():
    rng = np.random.default_rng(1)
    nodes = [onnx.helper.make_node("Softmax", ["X"], ["Y"], axis=1)]
    model = _model(nodes, [_vi("X", [2, 4, 8])], [_vi("Y", [2, 4, 8])], [])

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=1
    )
    onnx.checker.check_model(quant)
    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearSoftmax")
    axis = next(a.i for a in qlop.attribute if a.name == "axis")
    assert axis == 1

    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_softmax_pre_opset13_semantics():
    # Pre-opset-13 Softmax flattens the tensor into a 2-D matrix at `axis`
    # and reduces the trailing dimension -- entirely different semantics
    # from opset-13+'s in-place per-axis reduction. The rewrite must thread
    # the model's own opset through as QLinearSoftmax's `opset` attribute so
    # ONNX Runtime's kernel replicates the *correct* one, not silently
    # assume the newer semantics.
    rng = np.random.default_rng(2)
    nodes = [onnx.helper.make_node("Softmax", ["X"], ["Y"], axis=1)]
    model = _model(nodes, [_vi("X", [2, 3, 4])], [_vi("Y", [2, 3, 4])], [], opset=11)

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=2
    )
    onnx.checker.check_model(quant)
    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearSoftmax")
    opset_attr = next(a.i for a in qlop.attribute if a.name == "opset")
    assert opset_attr == 11

    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_multiple_independent_nodes():
    rng = np.random.default_rng(3)
    nodes = [
        onnx.helper.make_node("Softmax", ["A"], ["T1"], axis=-1),
        onnx.helper.make_node("Sigmoid", ["B"], ["T2"]),
        onnx.helper.make_node("Concat", ["T1", "T2"], ["C"], axis=0),
    ]
    model = _model(nodes, [_vi("A", [4, 8]), _vi("B", [4, 8])], [_vi("C", [8, 8])], [])

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=3
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["QLinearSoftmax"] == 1
    assert ops["Sigmoid"] == 1  # untouched by this pass

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    feeds = {"A": a, "B": b}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_non_float():
    nodes = [onnx.helper.make_node("Softmax", ["X"], ["Y"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [4])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [4])],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    quant = onnxsim.quantize_qoperator_softmax(model)
    assert _op_counts(quant)["Softmax"] == 1
    assert _op_counts(quant)["QLinearSoftmax"] == 0


def test_list_qoperator_softmax_quantizable_tensors():
    nodes = [onnx.helper.make_node("Softmax", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_softmax_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"X", "Y"}


def test_list_qoperator_softmax_quantizable_tensors_no_opset_import():
    # A model with no resolvable default-domain opset import has nothing
    # quantizable -- there is no safe "opset" attribute value to guess.
    nodes = [onnx.helper.make_node("Softmax", ["X"], ["Y"])]
    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("X", [4, 8])], [_vi("Y", [4, 8])], []
    )
    model = onnx.helper.make_model(graph, opset_imports=[], ir_version=10)
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_softmax_quantizable_tensors(model.SerializeToString())
    assert names == []
