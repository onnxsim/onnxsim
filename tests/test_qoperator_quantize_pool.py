"""Tests for ``onnxsim.quantize_qoperator_pool`` (the
``qoperator_quantize_pool`` C++ pass) -- the pooling analogue of
``test_qoperator_quantize_activation.py``'s ``QLinearSigmoid``/
``QLinearLeakyRelu`` coverage, using ONNX Runtime's "com.microsoft" contrib
ops ``QLinearAveragePool``/``QLinearGlobalAveragePool`` instead.
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


def test_quantize_average_pool():
    rng = np.random.default_rng(0)
    nodes = [
        onnx.helper.make_node(
            "AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[2, 2]
        )
    ]
    model = _model(nodes, [_vi("X", [1, 3, 4, 4])], [_vi("Y", [1, 3, 2, 2])], [])

    quant = onnxsim.quantize_qoperator_pool(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["AveragePool"] == 0
    assert ops["QLinearAveragePool"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearAveragePool")
    kernel_shape = list(
        next(a.ints for a in qlop.attribute if a.name == "kernel_shape")
    )
    strides = list(next(a.ints for a in qlop.attribute if a.name == "strides"))
    channels_last = next(a.i for a in qlop.attribute if a.name == "channels_last")
    assert kernel_shape == [2, 2]
    assert strides == [2, 2]
    assert channels_last == 0

    x = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_average_pool_with_padding_and_count_include_pad():
    rng = np.random.default_rng(1)
    nodes = [
        onnx.helper.make_node(
            "AveragePool",
            ["X"],
            ["Y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            count_include_pad=1,
        )
    ]
    model = _model(nodes, [_vi("X", [1, 2, 5, 5])], [_vi("Y", [1, 2, 5, 5])], [])

    quant = onnxsim.quantize_qoperator_pool(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearAveragePool")
    pads = list(next(a.ints for a in qlop.attribute if a.name == "pads"))
    count_include_pad = next(
        a.i for a in qlop.attribute if a.name == "count_include_pad"
    )
    assert pads == [1, 1, 1, 1]
    assert count_include_pad == 1

    x = rng.standard_normal((1, 2, 5, 5)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_global_average_pool():
    rng = np.random.default_rng(2)
    nodes = [onnx.helper.make_node("GlobalAveragePool", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, 3, 4, 4])], [_vi("Y", [1, 3, 1, 1])], [])

    quant = onnxsim.quantize_qoperator_pool(model, num_calibration_samples=16, seed=2)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["GlobalAveragePool"] == 0
    assert ops["QLinearGlobalAveragePool"] == 1

    x = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_multiple_independent_nodes():
    rng = np.random.default_rng(3)
    nodes = [
        onnx.helper.make_node("AveragePool", ["A"], ["T1"], kernel_shape=[2, 2]),
        onnx.helper.make_node("Sigmoid", ["B"], ["T2"]),
    ]
    model = _model(
        nodes,
        [_vi("A", [1, 2, 4, 4]), _vi("B", [1, 2, 3, 3])],
        [_vi("T1", [1, 2, 3, 3]), _vi("T2", [1, 2, 3, 3])],
        [],
    )

    quant = onnxsim.quantize_qoperator_pool(model, num_calibration_samples=16, seed=3)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["QLinearAveragePool"] == 1
    assert ops["Sigmoid"] == 1  # untouched by this pass

    a = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    b = rng.standard_normal((1, 2, 3, 3)).astype(np.float32)
    feeds = {"A": a, "B": b}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_dilations():
    # Standard ONNX AveragePool gained an optional `dilations` attribute in
    # opset 19; ONNX Runtime's QLinearAveragePool kernel rejects it, so a
    # node carrying it must be left untouched.
    nodes = [
        onnx.helper.make_node(
            "AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], dilations=[1, 1]
        )
    ]
    model = _model(
        nodes, [_vi("X", [1, 2, 4, 4])], [_vi("Y", [1, 2, 3, 3])], [], opset=19
    )
    quant = onnxsim.quantize_qoperator_pool(model)
    assert _op_counts(quant)["AveragePool"] == 1
    assert _op_counts(quant)["QLinearAveragePool"] == 0


def test_quantize_skips_non_float():
    nodes = [onnx.helper.make_node("GlobalAveragePool", ["X"], ["Y"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [
            onnx.helper.make_tensor_value_info(
                "X", onnx.TensorProto.FLOAT16, [1, 2, 4, 4]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT16, [1, 2, 1, 1]
            )
        ],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    quant = onnxsim.quantize_qoperator_pool(model)
    assert _op_counts(quant)["GlobalAveragePool"] == 1
    assert _op_counts(quant)["QLinearGlobalAveragePool"] == 0


def test_list_qoperator_pool_quantizable_tensors():
    nodes = [onnx.helper.make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2])]
    model = _model(nodes, [_vi("X", [1, 2, 4, 4])], [_vi("Y", [1, 2, 3, 3])], [])
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_pool_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"X", "Y"}
