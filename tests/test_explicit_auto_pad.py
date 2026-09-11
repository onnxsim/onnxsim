"""Tests for the ``explicit_auto_pad`` C++ pass
(onnxsim/passes/explicit_auto_pad.h).

``Conv``/``AveragePool``/``MaxPool`` with ``auto_pad`` in ``SAME_UPPER``,
``SAME_LOWER`` or ``VALID`` get ``auto_pad = "NOTSET"`` and the equivalent
explicit ``pads``, computed from the ONNX operator spec's own ``auto_pad``
formula -- the same rewrite ``scripts/axelera/legalize.py``'s
``explicit_auto_pad`` does at the Python/script level (that file's docstring
records this as one of the rules a real Voyager SDK compiler build enforces,
under the vendor's own ``auto_pad == "NOTSET"`` requirement); this is its
onnxsim-core counterpart, usable from any binding (Python, C, Rust, npm) via
``extra_optimizers=["explicit_auto_pad"]`` rather than only as a standalone
script.

Every model is built with the ONNX text format parser (``onnx.parser``) per
CLAUDE.md's convention for this repo's tests. ``onnxsim.simplify``'s own
``check_n`` machinery gives the numeric-equivalence check (onnxruntime, or
the onnx reference evaluator when onnxruntime is not installed) -- except
for ``VALID``, where the installed ``onnx`` package's own
``ReferenceEvaluator`` has a real bug (``onnx/reference/ops/op_conv.py``
reuses the ``SAME_UPPER`` pad formula for ``auto_pad == "VALID"`` instead of
padding with zero), so that case is checked structurally against the ONNX
spec's own definition of ``VALID`` instead of round-tripped through
``check_n``.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

import onnxsim


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
    onnx.checker.check_model(model)
    return model


def _find(model, op_type):
    return next(n for n in model.graph.node if n.op_type == op_type)


def _attr(node, name):
    return next((a for a in node.attribute if a.name == name), None)


def _auto_pad(node):
    a = _attr(node, "auto_pad")
    return a.s.decode("utf-8") if a is not None else "NOTSET"


def _pads(node):
    a = _attr(node, "pads")
    return list(a.ints) if a is not None else None


def _conv_model(auto_pad, C=4, H=10, W=10, k=3, include_kernel_shape=True):
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(C, C, k, k).astype(np.float32), "w"
    )
    kernel_attr = f"kernel_shape=[{k}, {k}], " if include_kernel_shape else ""
    return _model(
        f"""
        g (float[1,{C},{H},{W}] x) => (float[1,{C},?,?] y)
        {{ y = Conv<{kernel_attr}auto_pad = "{auto_pad}">(x, w) }}
        """,
        initializer=[w],
    )


def _simplify(model, x, check_n=1):
    return onnxsim.simplify(
        model,
        check_n=check_n,
        input_data={"x": x},
        extra_optimizers=["explicit_auto_pad"],
    )


def test_conv_same_upper_gets_explicit_pads_and_computes_the_same_thing():
    model = _conv_model("SAME_UPPER")
    x = np.random.RandomState(1).randn(1, 4, 10, 10).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok

    conv = _find(sim_model, "Conv")
    assert _auto_pad(conv) == "NOTSET"
    # 10x10 input, 3x3 kernel, stride 1 -> SAME needs 2 total, 1 on each side.
    assert _pads(conv) == [1, 1, 1, 1]


def test_conv_same_lower_splits_the_odd_pixel_the_other_way():
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(4, 4, 4, 4).astype(np.float32), "w"
    )
    model = _model(
        """
        g (float[1,4,9,9] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape=[4, 4], auto_pad="SAME_LOWER", strides=[1, 1]>(x, w) }
        """,
        initializer=[w],
    )
    x = np.random.RandomState(2).randn(1, 4, 9, 9).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok

    conv = _find(sim_model, "Conv")
    # needed = 3 total; SAME_LOWER puts the extra pixel at the start.
    assert _pads(conv) == [2, 2, 1, 1]


def test_conv_valid_becomes_zero_pads():
    # No check_n round-trip here -- see this file's module docstring for the
    # ReferenceEvaluator VALID bug this sidesteps. Structural correctness
    # follows directly from the ONNX spec's own definition of VALID (zero
    # padding), which needs no numeric check to confirm.
    model = _conv_model("VALID")
    x = np.random.RandomState(3).randn(1, 4, 10, 10).astype(np.float32)
    sim_model, ok = _simplify(model, x, check_n=0)
    assert ok

    conv = _find(sim_model, "Conv")
    assert _auto_pad(conv) == "NOTSET"
    assert _pads(conv) == [0, 0, 0, 0]


def test_conv_notset_is_left_alone():
    model = _conv_model("NOTSET")
    x = np.random.RandomState(4).randn(1, 4, 10, 10).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok
    conv = _find(sim_model, "Conv")
    assert _pads(conv) is None


def test_conv_kernel_shape_is_read_from_weight_when_attribute_is_absent():
    """Conv's own kernel_shape attribute is optional (inferrable from `W`),
    and exporters routinely omit it -- this rewrite has to do the same or it
    silently skips exactly those graphs."""
    model = _conv_model("SAME_UPPER", include_kernel_shape=False)
    x = np.random.RandomState(5).randn(1, 4, 10, 10).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok
    conv = _find(sim_model, "Conv")
    assert _auto_pad(conv) == "NOTSET"
    assert _pads(conv) == [1, 1, 1, 1]


def test_conv_with_dynamic_input_shape_is_left_alone():
    """The `auto_pad` formula needs the input's spatial shape; a dynamic
    axis means it can't be computed, so the rule declines rather than guess
    and risk changing the output shape."""
    w = numpy_helper.from_array(np.zeros((4, 4, 3, 3), np.float32), "w")
    model = _model(
        """
        g (float[1,4,H,W] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape=[3, 3], auto_pad="SAME_UPPER">(x, w) }
        """,
        initializer=[w],
    )
    sim_model, ok = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["explicit_auto_pad"]
    )
    assert ok
    conv = _find(sim_model, "Conv")
    assert _auto_pad(conv) == "SAME_UPPER"


@pytest.mark.parametrize("op_type", ["AveragePool", "MaxPool"])
def test_averagepool_and_maxpool_same_upper_are_also_rewritten(op_type):
    model = _model(
        f"""
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        {{ y = {op_type}<kernel_shape=[3, 3], auto_pad="SAME_UPPER">(x) }}
        """
    )
    x = np.random.RandomState(6).randn(1, 4, 10, 10).astype(np.float32)
    sim_model, ok = _simplify(model, x)
    assert ok

    node = _find(sim_model, op_type)
    assert _auto_pad(node) == "NOTSET"
    assert _pads(node) == [1, 1, 1, 1]


def test_disabled_by_default():
    """`explicit_auto_pad` is `PassType::Other`, so a plain `simplify()` call
    (no `extra_optimizers`) must leave a SAME-padded Conv untouched."""
    model = _conv_model("SAME_UPPER")
    x = np.random.RandomState(7).randn(1, 4, 10, 10).astype(np.float32)
    sim_model, ok = onnxsim.simplify(model, check_n=1, input_data={"x": x})
    assert ok
    conv = _find(sim_model, "Conv")
    assert _auto_pad(conv) == "SAME_UPPER"
