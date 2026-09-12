"""Tests for the ``neg_to_mul`` C++ pass (onnxsim/passes/neg_to_mul.h).

``Neg(x)`` becomes ``Mul(x, -1)`` -- the onnxsim-core counterpart of
``scripts/axera/legalize.py``'s ``neg_to_mul`` rule (that file's docstring
records this as the fix for the one op ``onnxsim.graph_grad`` emits that is
absent from a real NPU's op-support list). Usable from any binding via
``extra_optimizers=["neg_to_mul"]``.

Models are built with ``onnx.parser`` per CLAUDE.md's convention. Numeric
equivalence comes from ``onnxsim.simplify``'s own ``check_n``.
"""

import numpy as np
from onnx import parser

import onnxsim


def _model(body, opset=17, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _find(model, op_type):
    return next(n for n in model.graph.node if n.op_type == op_type)


def test_neg_becomes_mul_by_minus_one_and_computes_the_same_thing():
    model = _model(
        """
        g (float[2,3] x) => (float[2,3] y)
        { y = Neg(x) }
        """
    )
    x = np.random.RandomState(0).randn(2, 3).astype(np.float32)
    sim_model, ok = onnxsim.simplify(
        model, check_n=1, input_data={"x": x}, extra_optimizers=["neg_to_mul"]
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Neg" not in op_types
    mul = _find(sim_model, "Mul")
    assert mul.input[0] == "x"


def test_non_float_neg_is_left_alone():
    """The emitted `-1` constant is written via `Tensor::floats()`, which is
    only the right wire representation for `float32` -- an `int64` `Neg`
    (exact and common: negating an axis/shape value) must not be rewritten
    into a `Mul` carrying a `float32` constant against an `int64` input."""
    model = _model(
        """
        g (int64[3] x) => (int64[3] y)
        { y = Neg(x) }
        """
    )
    x = np.array([1, -2, 3], np.int64)
    sim_model, ok = onnxsim.simplify(
        model, check_n=1, input_data={"x": x}, extra_optimizers=["neg_to_mul"]
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Neg"]


def test_disabled_by_default():
    """`neg_to_mul` is `PassType::Other`, so a plain `simplify()` call (no
    `extra_optimizers`) must leave `Neg` alone."""
    model = _model(
        """
        g (float[2,3] x) => (float[2,3] y)
        { y = Neg(x) }
        """
    )
    x = np.zeros((2, 3), np.float32)
    sim_model, ok = onnxsim.simplify(model, check_n=1, input_data={"x": x})
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Neg"]
