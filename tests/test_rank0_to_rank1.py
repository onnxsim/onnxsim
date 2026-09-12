"""Tests for the ``rank0_to_rank1`` C++ pass
(onnxsim/passes/rank0_to_rank1.h).

Gives every rank-0 (scalar) graph output a trailing axis -- the onnxsim-core
counterpart of ``scripts/axera/legalize.py``'s ``rank0_to_rank1`` rule (see
``tests/test_axera_legalize.py`` for that rule's own Pulsar2-calibration
motivation). Usable from any binding via
``extra_optimizers=["rank0_to_rank1"]``.

Models are built with ``onnx.parser`` per CLAUDE.md's convention, except for
the rank-0 output's ``shape`` field, which the text format has no way to
spell explicitly ("no shape" and "shape with zero dims" both parse the same
way) -- ``ClearField``/an explicit empty ``dim`` list is set programmatically
after parsing instead, per CLAUDE.md's documented exception for that case.
"""

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=17, ir_version=10):
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _mark_rank0(model, name):
    """Declares `name` a rank-0 (scalar) value -- an explicit, zero-length
    `shape` -- for whichever graph output already carries that name."""
    for value in model.graph.output:
        if value.name == name:
            value.type.tensor_type.ClearField("shape")
            value.type.tensor_type.shape.SetInParent()


def test_reduce_mean_output_gets_keepdims_and_explicit_axes():
    """The motivating case: a training loss, `ReduceMean` with no axes
    named, over a graph output Pulsar2's calibration step cannot concatenate
    at rank 0."""
    model = _model(
        """
        g (float[2,3] x) => (float loss)
        { loss = ReduceMean(x) }
        """
    )
    _mark_rank0(model, "loss")
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["rank0_to_rank1"])
    assert ok

    out = sim_model.graph.output[0]
    assert len(out.type.tensor_type.shape.dim) == 1
    assert out.type.tensor_type.shape.dim[0].dim_value == 1

    reduce = next(n for n in sim_model.graph.node if n.op_type == "ReduceMean")
    keepdims = next(a.i for a in reduce.attribute if a.name == "keepdims")
    assert keepdims == 1
    axes_attr = next((a for a in reduce.attribute if a.name == "axes"), None)
    if axes_attr is not None:
        assert list(axes_attr.ints) == [0, 1]
    else:
        # opset >= 18: axes is reduce.input[1], a constant initializer.
        axes_init = next(
            i for i in sim_model.graph.initializer if i.name == reduce.input[1]
        )
        assert list(onnx.numpy_helper.to_array(axes_init)) == [0, 1]

    x = np.random.RandomState(0).randn(2, 3).astype(np.float32)
    (before,) = _run(model, {"x": x})
    (after,) = _run(sim_model, {"x": x})
    assert np.asarray(after).shape == (1,)
    assert np.allclose(np.asarray(before).reshape(-1), np.asarray(after))


def test_reduce_mean_output_gets_axes_as_input_at_opset18():
    """`axes` moved from a Reduce* attribute to its second input at opset
    18 -- the unnamed-axes case must add the right kind of value for the
    graph's own opset, not always the attribute form."""
    model = _model(
        """
        g (float[2,3] x) => (float loss)
        { loss = ReduceMean(x) }
        """,
        opset=18,
    )
    _mark_rank0(model, "loss")
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["rank0_to_rank1"])
    assert ok

    reduce = next(n for n in sim_model.graph.node if n.op_type == "ReduceMean")
    assert not any(a.name == "axes" for a in reduce.attribute)
    assert len(reduce.input) == 2
    axes_init = next(
        i for i in sim_model.graph.initializer if i.name == reduce.input[1]
    )
    assert list(onnx.numpy_helper.to_array(axes_init)) == [0, 1]
    keepdims = next(a.i for a in reduce.attribute if a.name == "keepdims")
    assert keepdims == 1

    x = np.random.RandomState(2).randn(2, 3).astype(np.float32)
    (before,) = _run(model, {"x": x})
    (after,) = _run(sim_model, {"x": x})
    assert np.allclose(np.asarray(before).reshape(-1), np.asarray(after))


def test_named_axes_are_left_alone():
    """Only the *default* "reduce everything" case needs naming -- a
    `ReduceMean` that already names its own axes must not be rewritten."""
    model = _model(
        """
        g (float[2,3] x) => (float loss)
        { loss = ReduceMean<axes=[0,1]>(x) }
        """
    )
    _mark_rank0(model, "loss")
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["rank0_to_rank1"])
    assert ok
    reduce = next(n for n in sim_model.graph.node if n.op_type == "ReduceMean")
    axes_attr = next(a for a in reduce.attribute if a.name == "axes")
    assert list(axes_attr.ints) == [0, 1]


def test_non_reduce_scalar_output_still_gets_reshaped():
    """A scalar output from any other op is still wrapped in a `Reshape` to
    `[1]` -- the keepdims/axes special-casing is Reduce*-only, but the rank
    fix-up applies to every rank-0 output."""
    model = _model(
        """
        g (float[2] x) => (float y)
        {
          a = ReduceSum<keepdims=0>(x)
          b = ReduceSum<keepdims=0>(x)
          y = Add(a, b)
        }
        """
    )
    _mark_rank0(model, "y")
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["rank0_to_rank1"])
    assert ok
    out = sim_model.graph.output[0]
    assert len(out.type.tensor_type.shape.dim) == 1
    assert out.type.tensor_type.shape.dim[0].dim_value == 1
    assert any(n.op_type == "Reshape" for n in sim_model.graph.node)

    x = np.random.RandomState(1).randn(2).astype(np.float32)
    (before,) = _run(model, {"x": x})
    (after,) = _run(sim_model, {"x": x})
    assert np.allclose(np.asarray(before).reshape(-1), np.asarray(after))


def test_rank1_output_is_left_alone():
    model = _model(
        """
        g (float[2,3] x) => (float[3] y)
        { y = ReduceMean<axes=[0], keepdims=0>(x) }
        """
    )
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["rank0_to_rank1"])
    assert ok
    assert [n.op_type for n in sim_model.graph.node] == ["ReduceMean"]


def test_disabled_by_default():
    """`rank0_to_rank1` is `PassType::Other`, so a plain `simplify()` call
    (no `extra_optimizers`) must leave a scalar output alone."""
    model = _model(
        """
        g (float[2,3] x) => (float loss)
        { loss = ReduceMean(x) }
        """
    )
    _mark_rank0(model, "loss")
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    assert len(sim_model.graph.output[0].type.tensor_type.shape.dim) == 0
