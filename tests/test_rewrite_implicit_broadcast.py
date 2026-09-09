"""Tests for the ``rewrite_implicit_broadcast`` C++ pass
(onnxsim/passes/rewrite_implicit_broadcast.h).

Some inference backends (small NPUs in particular, or runtimes whose
elementwise kernels require identically-shaped operands) don't support
NumPy-style ("implicit"/multidirectional) broadcasting. This pass rewrites a
broadcasting elementwise op -- e.g. ``Add``/``Mul``/``Where`` -- into the
same op applied to operands that have each been made exactly the node's
(statically-known) output shape via an explicit ``Expand``, so the op itself
never needs to broadcast anything at runtime.

It is registered as ``PassType::Other`` and therefore never runs by default
-- opt in via ``extra_optimizers=["rewrite_implicit_broadcast"]``.
"""

import numpy as np
import onnx
import onnx.reference
from onnx import parser

import onnxsim


def _model(body, initializer=(), opset=13, ir_version=10):
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


def _simplify_with_pass(model, **kwargs):
    return onnxsim.simplify(
        model, extra_optimizers=["rewrite_implicit_broadcast"], **kwargs
    )


def _operand_shapes(model, node):
    value_shapes = {}
    for vi in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        value_shapes[vi.name] = tuple(
            d.dim_value for d in vi.type.tensor_type.shape.dim
        )
    return [value_shapes[name] for name in node.input]


def test_add_broadcast_made_explicit():
    model = _model(
        """
        g (float[3,1] a, float[4] b) => (float[3,4] y)
        {
          y = Add(a, b)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = _simplify_with_pass(model, check_n=0)
    onnx.checker.check_model(sim_model)

    # Exactly one Add remains, and both its operands (now Expand outputs)
    # have the Add's own output shape -- no implicit broadcasting left.
    adds = [n for n in sim_model.graph.node if n.op_type == "Add"]
    assert len(adds) == 1
    shapes = _operand_shapes(sim_model, adds[0])
    assert shapes == [(3, 4), (3, 4)]
    assert any(n.op_type == "Expand" for n in sim_model.graph.node)

    assert check_ok
    a = np.random.rand(3, 1).astype(np.float32)
    b = np.random.rand(4).astype(np.float32)
    ref = onnx.reference.ReferenceEvaluator(model)
    (expected,) = ref.run(None, {"a": a, "b": b})
    sim_ref = onnx.reference.ReferenceEvaluator(sim_model)
    (actual,) = sim_ref.run(None, {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected)


def test_where_all_three_operands_made_explicit():
    # cond/x/y all have different (numpy-broadcastable) shapes; all three
    # participate in Where's own broadcast and should all be expanded.
    model = _model(
        """
        g (bool[1,4] cond, float[4] x, float[3,1] y) => (float[3,4] z)
        {
          z = Where(cond, x, y)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = _simplify_with_pass(model, check_n=0)
    onnx.checker.check_model(sim_model)

    wheres = [n for n in sim_model.graph.node if n.op_type == "Where"]
    assert len(wheres) == 1
    shapes = _operand_shapes(sim_model, wheres[0])
    assert shapes == [(3, 4), (3, 4), (3, 4)]

    assert check_ok
    cond = np.array([[True, False, True, False]])
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    ref = onnx.reference.ReferenceEvaluator(model)
    (expected,) = ref.run(None, {"cond": cond, "x": x, "y": y})
    sim_ref = onnx.reference.ReferenceEvaluator(sim_model)
    (actual,) = sim_ref.run(None, {"cond": cond, "x": x, "y": y})
    np.testing.assert_allclose(actual, expected)


def test_already_same_shape_operands_left_untouched():
    # Both operands already have the output's exact shape: nothing to make
    # explicit, so no Expand should be inserted at all.
    model = _model(
        """
        g (float[3,4] a, float[3,4] b) => (float[3,4] y)
        {
          y = Add(a, b)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = _simplify_with_pass(model, check_n=0)
    assert check_ok
    onnx.checker.check_model(sim_model)
    assert not any(n.op_type == "Expand" for n in sim_model.graph.node)


def test_pass_is_opt_in_only():
    # Without extra_optimizers, the pass must not run even though the graph
    # has an implicit-broadcast Add.
    model = _model(
        """
        g (float[3,1] a, float[4] b) => (float[3,4] y)
        {
          y = Add(a, b)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    onnx.checker.check_model(sim_model)
    assert not any(n.op_type == "Expand" for n in sim_model.graph.node)
