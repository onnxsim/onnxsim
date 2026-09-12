"""Tests for the ``pow2_to_mul`` C++ pass (onnxsim/passes/pow2_to_mul.h).

``Pow(x, 2)`` becomes ``Mul(x, x)`` -- the onnxsim-core counterpart of
``scripts/axera/legalize.py``'s ``pow2_to_mul`` rule (that file's docstring
records this as the fix for a vendor compiler's fused-activation matcher,
which recognizes ``Pow(x, 2)`` as part of a larger pattern it cannot tile at
every shape). Usable from any binding via ``extra_optimizers=["pow2_to_mul"]``.

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


def test_pow_by_2_becomes_mul_by_self_and_computes_the_same_thing():
    model = _model(
        """
        g (float[2,3] x) => (float[2,3] y)
        <float exponent = {2.0}>
        { y = Pow(x, exponent) }
        """
    )
    x = np.random.RandomState(0).randn(2, 3).astype(np.float32)
    sim_model, ok = onnxsim.simplify(
        model, check_n=1, input_data={"x": x}, extra_optimizers=["pow2_to_mul"]
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Pow" not in op_types
    mul = _find(sim_model, "Mul")
    assert mul.input[0] == mul.input[1] == "x"


def test_pow_by_a_different_exponent_is_left_alone():
    """Only the exact scalar-two case fires -- `Pow(x, 3)` is not
    `Mul(x, x)`, and must not be rewritten."""
    model = _model(
        """
        g (float[4] x) => (float[4] y)
        <float exponent = {3.0}>
        { y = Pow(x, exponent) }
        """
    )
    x = np.array([1.0, 2.0, -1.0, 0.5], np.float32)
    sim_model, ok = onnxsim.simplify(
        model, check_n=1, input_data={"x": x}, extra_optimizers=["pow2_to_mul"]
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Pow"]


def test_pow_with_a_non_constant_exponent_is_left_alone():
    """The exponent must be a constant scalar -- an elementwise `Pow` with a
    runtime-input exponent (which may not even be 2 everywhere) is not this
    rule's shape."""
    model = _model(
        """
        g (float[3] x, float[3] p) => (float[3] y)
        { y = Pow(x, p) }
        """
    )
    x = np.array([1.0, 2.0, 3.0], np.float32)
    p = np.array([2.0, 2.0, 2.0], np.float32)
    sim_model, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"x": x, "p": p},
        extra_optimizers=["pow2_to_mul"],
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Pow"]


def test_disabled_by_default():
    """`pow2_to_mul` is `PassType::Other`, so a plain `simplify()` call (no
    `extra_optimizers`) must leave `Pow` alone."""
    model = _model(
        """
        g (float[2,3] x) => (float[2,3] y)
        <float exponent = {2.0}>
        { y = Pow(x, exponent) }
        """
    )
    x = np.zeros((2, 3), np.float32)
    sim_model, ok = onnxsim.simplify(model, check_n=1, input_data={"x": x})
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Pow"]
