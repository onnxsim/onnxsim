"""Pass-isolated tests for eliminate_nop_reduce / eliminate_nop_softmax.

A Reduce{Sum,Mean,Max,Min,Prod} over only size-1 axes returns its input value
(``Squeeze`` when keepdims=0), and Softmax/LogSoftmax over a size-1 group is the
constant 1/0. Each test runs the one pass under test alone (every other default
pass is skipped) so the op counts show what that pass did; ``onnxsim.simplify``'s
own random-input check still guards numerical equivalence.
"""

import collections

import onnx
import onnxsim.onnxsim_cpp2py_export as C
import pytest
from onnx import parser

import onnxsim


def _run(model, pass_name, **kwargs):
    skipped = sorted(set(C._list_optimizers()) - {pass_name})
    sim_model, check_ok = onnxsim.simplify(
        model, check_n=3, skipped_optimizers=skipped, **kwargs
    )
    assert check_ok, "simplified model failed onnxsim's equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["": {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


# --------------------------------------------------------------------------- #
# eliminate_nop_reduce
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "op", ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd"]
)
def test_reduce_keepdims1_is_removed(op):
    # Opset 18: every Reduce* takes `axes` as an input.
    model = _model(
        f"""
        g (float[2,1,4] X) => (float[2,1,4] Y)
        <int64[1] axes = {{1}}>
        {{
          a = Relu(X)
          Y = {op}(a, axes)
        }}
        """,
        opset=18,
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops[op] == 0
    assert ops["Relu"] == 1


@pytest.mark.parametrize("op", ["ReduceMax", "ReduceMin", "ReduceProd"])
def test_reduce_keepdims0_becomes_squeeze_attr_axes(op):
    # Opset 13: only ReduceSum moved axes to an input; the rest keep the attr,
    # and Squeeze takes its axes as an input.
    model = _model(
        f"""
        g (float[2,1,4,1] X) => (float[2,4] Y)
        {{
          Y = {op}<axes = [1, -1], keepdims = 0>(X)
        }}
        """
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops[op] == 0
    assert ops["Squeeze"] == 1


def test_reduce_keepdims0_squeeze_uses_attr_before_opset13():
    model = _model(
        """
        g (float[2,1,4] X) => (float[2,4] Y)
        {
          Y = ReduceSum<axes = [1], keepdims = 0>(X)
        }
        """,
        opset=11,
    )
    sim, ops = _run(model, "eliminate_nop_reduce")
    assert ops["ReduceSum"] == 0
    assert ops["Squeeze"] == 1
    (squeeze,) = [n for n in sim.graph.node if n.op_type == "Squeeze"]
    assert len(squeeze.input) == 1
    assert [a.name for a in squeeze.attribute] == ["axes"]


def test_reduce_all_axes_of_all_ones_input():
    # No axes given: reduces every axis, all of which are 1.
    model = _model(
        """
        g (float[1,1,1] X) => (float Y)
        {
          a = Relu(X)
          Y = ReduceMax<keepdims = 0>(a)
        }
        """
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops["ReduceMax"] == 0
    assert ops["Squeeze"] == 1


def test_reduce_noop_with_empty_axes_is_removed():
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y)
        <int64[0] axes = {}>
        {
          a = Relu(X)
          Y = ReduceSum<noop_with_empty_axes = 1>(a, axes)
        }
        """,
        opset=18,
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops["ReduceSum"] == 0


def test_reduce_declines_when_axis_not_one():
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,1,4] Y)
        <int64[1] axes = {1}>
        {
          Y = ReduceSum<keepdims = 1>(X, axes)
        }
        """
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops["ReduceSum"] == 1


def test_reduce_declines_when_one_of_several_axes_not_one():
    model = _model(
        """
        g (float[2,1,4] X) => (float[2,1,1] Y)
        <int64[2] axes = {1, 2}>
        {
          Y = ReduceMax<keepdims = 1>(X, axes)
        }
        """,
        opset=18,
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops["ReduceMax"] == 1


def test_reduce_declines_on_symbolic_dim():
    model = _model(
        """
        g (float[N,4] X) => (float[1,4] Y)
        <int64[1] axes = {0}>
        {
          Y = ReduceSum<keepdims = 1>(X, axes)
        }
        """
    )
    _, ops = _run(
        model,
        "eliminate_nop_reduce",
        dynamic_input_shape=True,
        test_input_shapes={"X": [3, 4]},
    )
    assert ops["ReduceSum"] == 1


@pytest.mark.parametrize("op", ["ReduceL1", "ReduceL2", "ReduceSumSquare"])
def test_reduce_declines_non_identity_kinds(op):
    # |x| / sqrt(x^2) / x^2 of the lone element is not the element itself.
    model = _model(
        f"""
        g (float[2,1,4] X) => (float[2,1,4] Y)
        {{
          Y = {op}<axes = [1], keepdims = 1>(X)
        }}
        """
    )
    _, ops = _run(model, "eliminate_nop_reduce")
    assert ops[op] == 1


# --------------------------------------------------------------------------- #
# eliminate_nop_softmax
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
def test_softmax_over_unit_last_axis_is_removed(op):
    model = _model(
        f"""
        g (float[2,3,1] X) => (float[2,3,1] Y)
        {{
          Y = {op}<axis = -1>(X)
        }}
        """
    )
    _, ops = _run(model, "eliminate_nop_softmax")
    # The ConstantOfShape(Shape(X)) replacement is itself folded to an
    # initializer by onnxsim's constant folding.
    assert ops[op] == 0


def test_softmax_explicit_inner_unit_axis():
    model = _model(
        """
        g (float[2,1,5] X) => (float[2,1,5] Y)
        {
          Y = Softmax<axis = 1>(X)
        }
        """
    )
    _, ops = _run(model, "eliminate_nop_softmax")
    assert ops["Softmax"] == 0


def test_softmax_float16():
    model = _model(
        """
        g (float16[2,3,1] X) => (float16[2,3,1] Y)
        {
          Y = Softmax<axis = -1>(X)
        }
        """
    )
    sim, ops = _run(model, "eliminate_nop_softmax")
    assert ops["Softmax"] == 0
    assert sim.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


def test_softmax_declines_when_axis_not_one():
    model = _model(
        """
        g (float[2,1,5] X) => (float[2,1,5] Y)
        {
          Y = Softmax<axis = -1>(X)
        }
        """
    )
    _, ops = _run(model, "eliminate_nop_softmax")
    assert ops["Softmax"] == 1


def test_softmax_opset11_coerces_trailing_block():
    # Before opset 13 the normalization spans dims[axis:], so [2,1,3] with
    # axis=1 is a 3-element group and must stay; [2,1,1] is a single element.
    keep = _model(
        """
        g (float[2,1,3] X) => (float[2,1,3] Y)
        {
          Y = Softmax<axis = 1>(X)
        }
        """,
        opset=11,
    )
    _, ops = _run(keep, "eliminate_nop_softmax")
    assert ops["Softmax"] == 1

    drop = _model(
        """
        g (float[2,1,1] X) => (float[2,1,1] Y)
        {
          Y = Softmax<axis = 1>(X)
        }
        """,
        opset=11,
    )
    _, ops = _run(drop, "eliminate_nop_softmax")
    assert ops["Softmax"] == 0


def test_softmax_declines_on_symbolic_axis_dim():
    model = _model(
        """
        g (float[2,N] X) => (float[2,N] Y)
        {
          Y = Softmax<axis = -1>(X)
        }
        """
    )
    _, ops = _run(
        model,
        "eliminate_nop_softmax",
        dynamic_input_shape=True,
        test_input_shapes={"X": [2, 5]},
    )
    assert ops["Softmax"] == 1
