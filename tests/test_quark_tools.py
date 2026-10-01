"""Tests for onnxsim.quark_tools (Quark-named post-quantization graph tools)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_tools as qt


def _run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _qdq_model():
    # weight: int8 per-channel (axis 1) DQ; activation: Q->DQ pair before MatMul
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,2] x) => (float[N,2] y)
        <int8[2,2] wq = {10, -20, 30, 40}, float[2] ws = {0.5, 0.25}, int8[2] wz = {0, 4},
         float xs = {0.1}, uint8 xz = {128}>
        {
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            w = DequantizeLinear<axis = 1>(wq, ws, wz)
            y = MatMul(xd, w)
        }
        """
    )
    return model


def test_remove_qdq_strips_pairs_and_folds_weights():
    model = _qdq_model()
    out = qt.remove_qdq(model)
    assert _ops(out) == ["MatMul"]
    assert {i.name for i in out.graph.initializer} == {"w"}
    w = numpy_helper.to_array(out.graph.initializer[0])
    expected = (np.array([[10, -20], [30, 40]], np.float32) - [0, 4]) * [0.5, 0.25]
    np.testing.assert_allclose(w, expected)
    x = np.array([[1.0, 2.0]], np.float32)
    np.testing.assert_allclose(_run(out, {"x": x})[0], x @ expected, rtol=1e-6)
    onnx.checker.check_model(out)


def test_remove_qdq_without_fold_keeps_weight_dq():
    out = qt.remove_qdq(_qdq_model(), fold_weights=False)
    assert _ops(out) == ["DequantizeLinear", "MatMul"]


def test_remove_qdq_keeps_graph_output_name_via_identity():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N] x) => (float[N] y)
        <float s = {0.1}, uint8 z = {0}>
        {
            q = QuantizeLinear(x, s, z)
            y = DequantizeLinear(q, s, z)
        }
        """
    )
    out = qt.remove_qdq(model)
    assert _ops(out) == ["Identity"]
    assert out.graph.output[0].name == "y"


def test_remove_qdq_leaves_q_feeding_non_dq():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N] x) => (uint8[N] y)
        <float s = {0.1}, uint8 z = {0}>
        {
            q = QuantizeLinear(x, s, z)
            y = Identity(q)
        }
        """
    )
    assert _ops(qt.remove_qdq(model)) == ["QuantizeLinear", "Identity"]


def test_remove_qdq_does_not_mutate_input():
    model = _qdq_model()
    before = model.SerializeToString()
    qt.remove_qdq(model)
    assert model.SerializeToString() == before


def test_shared_initializer_made_unique():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[2] x) => (float[2] y)
        <float[2] c = {1.0, 2.0}>
        {
            a = Add(x, c)
            b = Mul(a, c)
            y = Sub(b, c)
        }
        """
    )
    out = qt.convert_shared_initializer_to_unique(model)
    names = [i.name for i in out.graph.initializer]
    assert sorted(names) == ["c", "c_copy1", "c_copy2"]
    used = [n.input[1] for n in out.graph.node]
    assert len(set(used)) == 3
    x = np.array([3.0, 4.0], np.float32)
    np.testing.assert_allclose(_run(out, {"x": x})[0], _run(model, {"x": x})[0])


def test_dynamic_to_fixed_propagates_static_shapes():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[N,4] x) => (float[N,4] y)
        {
            y = Relu(x)
        }
        """
    )
    out = qt.convert_dynamic_to_fixed(model, {"x": [8, 4]})
    dims = [d.dim_value for d in out.graph.output[0].type.tensor_type.shape.dim]
    assert dims == [8, 4]
    with pytest.raises(ValueError, match="not a graph input"):
        qt.convert_dynamic_to_fixed(model, {"nope": [1, 4]})
    with pytest.raises(ValueError, match="rank"):
        qt.convert_dynamic_to_fixed(model, {"x": [4]})


def test_replace_inf_weights():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[3] x) => (float[3] y)
        {
            y = Add(x, c)
        }
        """
    )
    # inf has no text-format literal, so the tensor is attached programmatically.
    arr = np.array([np.inf, -np.inf, 1.0], np.float32)
    model.graph.initializer.append(numpy_helper.from_array(arr, "c"))
    out = qt.replace_inf_weights(model, max_value=1e30)
    got = numpy_helper.to_array(out.graph.initializer[0])
    np.testing.assert_array_equal(got, np.array([1e30, -1e30, 1.0], np.float32))
