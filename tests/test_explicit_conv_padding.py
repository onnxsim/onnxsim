"""Tests for the ``explicit_conv_padding`` C++ pass
(onnxsim/passes/explicit_conv_padding.h).

A ``Conv`` with asymmetric explicit ``pads`` gets a ``Pad`` node hoisted in
front of it and its own ``pads`` zeroed -- the onnxsim-core counterpart of
``scripts/axera/legalize.py``'s ``explicit_conv_padding`` rule (that file's
docstring records this as the fix for a causal convolution whose padding is
entirely on one side, which a vendor backend's fused convolution kernel
refuses). Usable from any binding via
``extra_optimizers=["explicit_conv_padding"]``.

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


def test_asymmetric_pads_become_a_pad_node_and_compute_the_same_thing():
    model = _model(
        """
        g (float[1,1,10] x) => (float[1,1,10] y)
        <float[1,1,3] w = {1.0, -2.0, 0.5}>
        {
          y = Conv<pads = [2, 0], kernel_shape = [3]>(x, w)
        }
        """
    )
    x = np.random.RandomState(0).randn(1, 1, 10).astype(np.float32)
    sim_model, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"x": x},
        extra_optimizers=["explicit_conv_padding"],
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Pad", "Conv"]

    conv = _find(sim_model, "Conv")
    pads_attr = next(a for a in conv.attribute if a.name == "pads")
    assert list(pads_attr.ints) == [0, 0]

    pad = _find(sim_model, "Pad")
    assert pad.input[0] == "x"


def test_symmetric_pads_are_left_alone():
    """Symmetric explicit `pads` are already the form every backend wants --
    hoisting a `Pad` node here would be pure churn, not a fix."""
    model = _model(
        """
        g (float[1,1,10] x) => (float[1,1,10] y)
        <float[1,1,3] w = {1.0, -2.0, 0.5}>
        {
          y = Conv<pads = [1, 1], kernel_shape = [3]>(x, w)
        }
        """
    )
    x = np.random.RandomState(0).randn(1, 1, 10).astype(np.float32)
    sim_model, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"x": x},
        extra_optimizers=["explicit_conv_padding"],
    )
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Conv"]


def test_disabled_by_default():
    """`explicit_conv_padding` is `PassType::Other`, so a plain `simplify()`
    call (no `extra_optimizers`) must leave the asymmetric `Conv` alone."""
    model = _model(
        """
        g (float[1,1,10] x) => (float[1,1,10] y)
        <float[1,1,3] w = {1.0, -2.0, 0.5}>
        {
          y = Conv<pads = [2, 0], kernel_shape = [3]>(x, w)
        }
        """
    )
    x = np.zeros((1, 1, 10), np.float32)
    sim_model, ok = onnxsim.simplify(model, check_n=1, input_data={"x": x})
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["Conv"]
    conv = _find(sim_model, "Conv")
    pads_attr = next(a for a in conv.attribute if a.name == "pads")
    assert list(pads_attr.ints) == [2, 0]
