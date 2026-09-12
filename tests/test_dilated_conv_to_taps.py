"""Tests for the ``dilated_conv_to_taps`` C++ pass
(onnxsim/passes/dilated_conv_to_taps.h).

A widely dilated 1-D convolution becomes one 1x1 convolution per tap, summed
-- the onnxsim-core counterpart of ``scripts/axera/legalize.py``'s
``dilated_conv_to_taps`` rule (see ``tests/test_axera_legalize.py``'s own
``test_dilated_conv_becomes_one_convolution_per_tap`` for that rule's
Pulsar2 motivation). Usable from any binding via
``extra_optimizers=["dilated_conv_to_taps"]``.

Models are built with ``onnx.parser`` per CLAUDE.md's convention; the
convolution weight/bias are numpy-built ``numpy_helper.from_array``
initializers attached after parsing, per CLAUDE.md's guidance for
random/deterministic weight arrays.
"""

import numpy as np
import pytest
from onnx import numpy_helper, parser

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


def _dilated_conv_model(pads, dilation, k=3, c=4, length=16, with_bias=True):
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(c, c, k).astype(np.float32), "w"
    )
    out_len = length + pads[0] + pads[1] - ((k - 1) * dilation + 1) + 1
    initializer = [w]
    inputs = "x, w, b" if with_bias else "x, w"
    if with_bias:
        b = numpy_helper.from_array(
            np.random.RandomState(1).randn(c).astype(np.float32), "b"
        )
        initializer.append(b)
    return _model(
        f"""
        g (float[1,{c},{length}] x) => (float[1,{c},{out_len}] y)
        {{
          y = Conv<kernel_shape=[{k}], pads=[{pads[0]},{pads[1]}],
                   dilations=[{dilation}], strides=[1]>({inputs})
        }}
        """,
        initializer=initializer,
    )


@pytest.mark.parametrize("pads,dilation", [((4, 0), 2), ((2, 2), 2), ((0, 0), 3)])
def test_dilated_conv_becomes_one_conv_per_tap_and_computes_the_same_thing(
    pads, dilation
):
    model = _dilated_conv_model(pads, dilation)
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["dilated_conv_to_taps"])
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.count("Conv") == 3
    # A zero-valued Pad (the pads=(0, 0) case) is a genuine no-op, which
    # onnxsim's own default `eliminate_nop_pad` pass removes as part of the
    # same simplify() call -- expected, not this pass's own concern.
    if any(pads):
        assert "Pad" in op_types
    assert "Slice" in op_types
    for node in sim_model.graph.node:
        if node.op_type == "Conv":
            dilations = next(
                (a.ints for a in node.attribute if a.name == "dilations"), [1]
            )
            assert list(dilations) == [1]

    x = np.random.RandomState(2).randn(1, 4, 16).astype(np.float32)
    (before,) = _run(model, {"x": x})
    (after,) = _run(sim_model, {"x": x})
    assert np.allclose(before, after, atol=1e-4), np.abs(before - after).max()


def test_bias_is_added_exactly_once():
    """Splitting a Conv with a bias into per-tap Convs must add the bias
    input to exactly one tap -- adding it to every tap would multiply it by
    the number of taps once the partials are summed."""
    model = _dilated_conv_model((4, 0), 2, with_bias=True)
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["dilated_conv_to_taps"])
    assert ok
    convs_with_bias = sum(
        1 for n in sim_model.graph.node if n.op_type == "Conv" and len(n.input) == 3
    )
    assert convs_with_bias == 1

    x = np.random.RandomState(3).randn(1, 4, 16).astype(np.float32)
    (before,) = _run(model, {"x": x})
    (after,) = _run(sim_model, {"x": x})
    assert np.allclose(before, after, atol=1e-4), np.abs(before - after).max()


def test_low_dilation_is_left_alone():
    """`min_dilation` is 2 -- an ordinary (dilation=1) Conv has nothing to
    gain from this rewrite and must be left untouched."""
    model = _dilated_conv_model((1, 1), 1)
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["dilated_conv_to_taps"])
    assert ok
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]


def test_grouped_conv_is_left_alone():
    """The rewrite reads the weight as [Cout, Cin, taps] and slices the
    *input* channel-wise -- wrong for `group > 1`, whose weight is
    [Cout, Cin/groups, taps], so a grouped conv must be skipped."""
    c = 4
    k, dilation = 3, 2
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(c, 1, k).astype(np.float32), "w"
    )
    model = _model(
        f"""
        g (float[1,{c},16] x) => (float[1,{c},12] y)
        {{
          y = Conv<kernel_shape=[{k}], pads=[0,0], dilations=[{dilation}],
                   strides=[1], group={c}>(x, w)
        }}
        """,
        initializer=[w],
    )
    sim_model, ok = onnxsim.simplify(model, extra_optimizers=["dilated_conv_to_taps"])
    assert ok
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]


def test_disabled_by_default():
    """`dilated_conv_to_taps` is `PassType::Other`, so a plain `simplify()`
    call (no `extra_optimizers`) must leave a dilated Conv alone."""
    model = _dilated_conv_model((4, 0), 2)
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]
