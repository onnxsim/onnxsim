"""Voyager SDK (Axelera Metis AIPU) legalization rewrites, checked offline.

`scripts/axelera/legalize.py` holds rewrites for three of the AIPU
acceleration constraints `voyager_ops.py`/`voyager_simulator.py` scrape from
Voyager SDK's own docs and check statically. Each test here checks the two
properties that matter: the rewrite fires where it should and nowhere else,
and it does not change what the graph computes. Where the target constraint
is one `voyager_simulator.evaluate_constraints()` can check, the test also
confirms the rewrite actually flips its verdict from `"violated"` to `"ok"`
-- real (if docs-derived, not hardware-verified) evidence the rewrite did
what it claims, not just that it ran.

Numeric equivalence is checked with `onnx.reference.ReferenceEvaluator`
rather than onnxruntime, since nothing here needs a real accelerator or even
a real ONNX Runtime install.

Neither needs Docker, a device, or the (large, optional) `axelera-rt`/
`axelera-devkit` install `voyager_backend.py` wraps.
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

_AXELERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axelera"
)
if _AXELERA_DIR not in sys.path:
    sys.path.insert(0, _AXELERA_DIR)

import legalize  # noqa: E402
import voyager_simulator as sim  # noqa: E402


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


def _verdicts(model):
    cache = sim._ShapeCache(model)
    return [sim.evaluate_constraints(n, cache).verdict for n in model.graph.node]


def _assert_same_output(before, after, feeds):
    onnx.checker.check_model(after)
    before_out = ReferenceEvaluator(before).run(None, feeds)
    after_out = ReferenceEvaluator(after).run(None, feeds)
    for b, a in zip(before_out, after_out):
        assert np.allclose(b, a, atol=1e-5), np.abs(np.asarray(b) - np.asarray(a)).max()


# --- explicit_auto_pad ---


def _conv_model(auto_pad, C=4, H=10, W=10, k=3):
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(C, C, k, k).astype(np.float32), "w"
    )
    return _model(
        f"""
        g (float[1,{C},{H},{W}] x) => (float[1,{C},?,?] y)
        {{ y = Conv<kernel_shape = [{k}, {k}], auto_pad = "{auto_pad}">(x, w) }}
        """,
        initializer=[w],
    )


def test_conv_same_upper_gets_explicit_pads_and_computes_the_same_thing():
    before = _conv_model("SAME_UPPER")
    after = _conv_model("SAME_UPPER")
    assert _verdicts(before) == ["violated"]

    assert legalize.explicit_auto_pad(after) == 1
    conv = after.graph.node[0]
    assert (
        onnx.helper.get_attribute_value(
            next(a for a in conv.attribute if a.name == "auto_pad")
        )
        == b"NOTSET"
    )
    pads = list(next(a for a in conv.attribute if a.name == "pads").ints)
    # 10x10 input, 3x3 kernel, stride 1 -> SAME needs 2 total, 1 on each side.
    assert pads == [1, 1, 1, 1]
    assert _verdicts(after) == ["ok"]

    x = np.random.RandomState(1).randn(1, 4, 10, 10).astype(np.float32)
    _assert_same_output(before, after, {"x": x})


def test_conv_same_lower_splits_the_odd_pixel_the_other_way():
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(4, 4, 4, 4).astype(np.float32), "w"
    )
    before = _model(
        """
        g (float[1,4,9,9] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape = [4, 4], auto_pad = "SAME_LOWER", strides = [1, 1]>(x, w) }
        """,
        initializer=[w],
    )
    after = _model(
        """
        g (float[1,4,9,9] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape = [4, 4], auto_pad = "SAME_LOWER", strides = [1, 1]>(x, w) }
        """,
        initializer=[w],
    )
    assert legalize.explicit_auto_pad(after) == 1
    pads = list(next(a for a in after.graph.node[0].attribute if a.name == "pads").ints)
    # needed = 3 total; SAME_LOWER puts the extra pixel at the start.
    assert pads == [2, 2, 1, 1]

    x = np.random.RandomState(2).randn(1, 4, 9, 9).astype(np.float32)
    _assert_same_output(before, after, {"x": x})


def test_conv_valid_becomes_zero_pads():
    # No numeric round-trip against the un-rewritten model here: the ONNX
    # spec defines VALID as zero padding, but the installed onnx package's
    # own ReferenceEvaluator has a real bug where Conv's auto_pad=="VALID"
    # branch reuses the SAME_UPPER pad formula instead of padding with zero
    # (onnx/reference/ops/op_conv.py, the `if auto_pad in {"SAME_LOWER",
    # "SAME_UPPER", "VALID"}:` branch does not special-case VALID) --
    # verified directly: a VALID Conv over this test's 10x10/3x3/stride-1
    # shape comes back from ReferenceEvaluator as 10x10, not the spec's 8x8.
    # The rewrite's own correctness is checked structurally against the
    # spec text instead, and numerically against the *correct* zero-pad
    # reading of VALID.
    after = _conv_model("VALID")
    assert legalize.explicit_auto_pad(after) == 1
    conv = after.graph.node[0]
    pads = list(next(a for a in conv.attribute if a.name == "pads").ints)
    assert pads == [0, 0, 0, 0]
    assert (
        onnx.helper.get_attribute_value(
            next(a for a in conv.attribute if a.name == "auto_pad")
        )
        == b"NOTSET"
    )

    reference = _conv_model("NOTSET")  # explicit zero pads, spec-correct VALID
    x = np.random.RandomState(3).randn(1, 4, 10, 10).astype(np.float32)
    _assert_same_output(reference, after, {"x": x})


def test_conv_notset_is_left_alone():
    model = _conv_model("NOTSET")
    assert legalize.explicit_auto_pad(model) == 0


def test_conv_kernel_shape_is_read_from_weight_when_attribute_is_absent():
    """Voyager's own Conv rule reads the kernel from `W`'s shape, since
    exporters routinely omit the `kernel_shape` attribute -- this rewrite
    has to do the same or it silently skips exactly those graphs."""
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(4, 4, 3, 3).astype(np.float32), "w"
    )
    before = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = Conv<auto_pad = "SAME_UPPER">(x, w) }
        """,
        initializer=[w],
    )
    after = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = Conv<auto_pad = "SAME_UPPER">(x, w) }
        """,
        initializer=[w],
    )
    assert legalize.explicit_auto_pad(after) == 1
    x = np.random.RandomState(4).randn(1, 4, 10, 10).astype(np.float32)
    _assert_same_output(before, after, {"x": x})


def test_conv_with_dynamic_input_shape_is_left_alone():
    """The `auto_pad` formula needs the input's spatial shape; a dynamic
    axis means it can't be computed, so the rule has to decline rather than
    guess and risk changing the output shape."""
    w = numpy_helper.from_array(np.zeros((4, 4, 3, 3), np.float32), "w")
    model = _model(
        """
        g (float[1,4,H,W] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x, w) }
        """,
        initializer=[w],
    )
    assert legalize.explicit_auto_pad(model) == 0
    assert _verdicts(model) == ["violated"]


def test_maxpool_same_upper_is_rewritten_and_becomes_fully_ok():
    """MaxPool's only two documented rules are `auto_pad == "NOTSET"` and
    `storage_order == 0`; with no `count_include_pad`-style wrinkle, this
    rewrite alone is enough to flip the whole verdict."""
    before = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = MaxPool<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x) }
        """
    )
    after = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = MaxPool<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x) }
        """
    )
    assert legalize.explicit_auto_pad(after) == 1
    assert _verdicts(before) == ["violated"]
    assert _verdicts(after) == ["ok"]

    x = np.random.RandomState(5).randn(1, 4, 10, 10).astype(np.float32)
    _assert_same_output(before, after, {"x": x})


def test_averagepool_same_upper_is_rewritten_and_preserves_semantics():
    """AveragePool's rules also require `count_include_pad == 1` once the
    padding is non-zero -- and that attribute changes the *result* at the
    padded edge (whether the implicit zeros count toward the divisor), so
    this rewrite correctly leaves it alone rather than flipping it to chase
    a clean verdict. What it does guarantee: the `auto_pad` rule specifically
    is satisfied, and the graph still computes the same thing.
    """
    before = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = AveragePool<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x) }
        """
    )
    after = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = AveragePool<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x) }
        """
    )
    assert legalize.explicit_auto_pad(after) == 1
    before_verdict = _verdicts(before)[0]
    assert before_verdict == "violated"

    cache = sim._ShapeCache(after)
    result = sim.evaluate_constraints(after.graph.node[0], cache)
    assert not any("auto_pad" in rule for rule in result.detail)

    x = np.random.RandomState(5).randn(1, 4, 10, 10).astype(np.float32)
    _assert_same_output(before, after, {"x": x})


# --- gemm_transA_to_transpose ---


def test_gemm_transA_becomes_an_explicit_transpose_and_computes_the_same_thing():
    b = numpy_helper.from_array(
        np.random.RandomState(0).randn(4, 8).astype(np.float32), "b"
    )
    c = numpy_helper.from_array(
        np.random.RandomState(1).randn(8).astype(np.float32), "c"
    )
    before = _model(
        """
        g (float[4,10] a) => (float[10,8] y)
        { y = Gemm<transA = 1>(a, b, c) }
        """,
        initializer=[b, c],
    )
    after = _model(
        """
        g (float[4,10] a) => (float[10,8] y)
        { y = Gemm<transA = 1>(a, b, c) }
        """,
        initializer=[b, c],
    )
    assert _verdicts(before) == ["violated"]

    assert legalize.gemm_transA_to_transpose(after) == 1
    kinds = [n.op_type for n in after.graph.node]
    assert kinds == ["Transpose", "Gemm"]
    gemm = after.graph.node[1]
    assert (
        onnx.helper.get_attribute_value(
            next(a for a in gemm.attribute if a.name == "transA")
        )
        == 0
    )
    # The Gemm's own rule is now satisfied...
    assert (
        sim.evaluate_constraints(after.graph.node[1], sim._ShapeCache(after)).verdict
        == "ok"
    )
    # ...but the inserted Transpose trades one documented violation for
    # another: Voyager's scraped `Transpose` rule (`perm == [0, 1, 2, 3]`)
    # is written for 4D feature maps, and a rank-2 `perm=[1, 0]` never
    # matches it -- the docs simply have no rank-2 case. That is a real gap
    # in what this checker (and, per its docstring, possibly Voyager's own
    # docs) can bless here, not a bug in the rewrite: it is still exact.
    assert (
        sim.evaluate_constraints(after.graph.node[0], sim._ShapeCache(after)).verdict
        == "violated"
    )

    a = np.random.RandomState(2).randn(4, 10).astype(np.float32)
    _assert_same_output(before, after, {"a": a})


def test_gemm_without_transA_is_left_alone():
    b = numpy_helper.from_array(np.zeros((8, 4), np.float32), "b")
    model = _model(
        """
        g (float[4,8] a) => (float[4,4] y)
        { y = Gemm(a, b) }
        """,
        initializer=[b],
    )
    assert legalize.gemm_transA_to_transpose(model) == 0
    assert [n.op_type for n in model.graph.node] == ["Gemm"]


# --- maxpool_rowmajor_when_indices_unused ---


def test_maxpool_storage_order_cleared_when_indices_output_is_absent():
    before = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y = MaxPool<kernel_shape = [2, 2], storage_order = 1>(x) }
        """
    )
    after = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y = MaxPool<kernel_shape = [2, 2], storage_order = 1>(x) }
        """
    )
    assert legalize.maxpool_rowmajor_when_indices_unused(after) == 1
    node = after.graph.node[0]
    assert (
        onnx.helper.get_attribute_value(
            next(a for a in node.attribute if a.name == "storage_order")
        )
        == 0
    )

    x = np.random.RandomState(6).randn(1, 4, 8, 8).astype(np.float32)
    _assert_same_output(before, after, {"x": x})


def test_maxpool_storage_order_is_left_alone_when_indices_output_is_used():
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y, int64[1,4,7,7] idx)
        { y, idx = MaxPool<kernel_shape = [2, 2], storage_order = 1>(x) }
        """
    )
    assert legalize.maxpool_rowmajor_when_indices_unused(model) == 0


def test_maxpool_default_storage_order_is_left_alone():
    model = _model(
        """
        g (float[1,4,8,8] x) => (float[1,4,7,7] y)
        { y = MaxPool<kernel_shape = [2, 2]>(x) }
        """
    )
    assert legalize.maxpool_rowmajor_when_indices_unused(model) == 0


# --- legalize() driver ---


def test_legalize_reports_what_each_rule_changed():
    w = numpy_helper.from_array(np.zeros((4, 4, 3, 3), np.float32), "w")
    model = _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x, w) }
        """,
        initializer=[w],
    )
    applied = legalize.legalize(model)
    assert set(applied) == set(legalize.RULES)
    assert applied["explicit_auto_pad"] == 1
    assert applied["gemm_transA_to_transpose"] == 0
    assert applied["maxpool_rowmajor_when_indices_unused"] == 0


# --- as_custom_rewriter() ---


def _same_upper_conv_model():
    w = numpy_helper.from_array(np.zeros((4, 4, 3, 3), np.float32), "w")
    return _model(
        """
        g (float[1,4,10,10] x) => (float[1,4,?,?] y)
        { y = Conv<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x, w) }
        """,
        initializer=[w],
    )


def test_as_custom_rewriter_mutates_in_place_and_returns_none_when_changed():
    """Matches `onnxsim.simplify`'s `custom_rewriter` contract: mutate
    `model` and return `None` when something changed (as opposed to
    returning a new `ModelProto`, the contract's other allowed form)."""
    model = _same_upper_conv_model()
    rewriter = legalize.as_custom_rewriter()
    result = rewriter(model)
    assert result is None
    conv = model.graph.node[0]
    assert any(a.name == "pads" for a in conv.attribute)


def test_as_custom_rewriter_returns_false_when_nothing_changed():
    """`custom_rewriter` returning `False` tells onnxsim's fixed point
    nothing changed this round, so it can skip re-processing the model --
    this is what lets `as_custom_rewriter()` be called repeatedly (as the
    fixed point does) without looping forever re-declaring "changed"."""
    model = _model(
        """
        g (float[4,8] a) => (float[4,4] y)
        { y = Gemm(a, b) }
        """,
        initializer=[numpy_helper.from_array(np.zeros((8, 4), np.float32), "b")],
    )
    rewriter = legalize.as_custom_rewriter()
    assert rewriter(model) is False
    # Idempotent: a second call against an already-legalized model also
    # reports no change, exactly like the fixed point's steady state.
    changed_model = _same_upper_conv_model()
    rewriter(changed_model)
    assert rewriter(changed_model) is False


def test_as_custom_rewriter_honors_an_explicit_rule_subset():
    model = _same_upper_conv_model()
    rewriter = legalize.as_custom_rewriter(rules=["gemm_transA_to_transpose"])
    # SAME_UPPER only affects explicit_auto_pad, which wasn't asked for.
    assert rewriter(model) is False
    conv = model.graph.node[0]
    assert not any(a.name == "pads" for a in conv.attribute)


def test_as_custom_rewriter_used_by_onnxsim_simplify():
    """End-to-end: the adapter actually works as `onnxsim.simplify`'s
    `custom_rewriter`, with no `onnxsim` rebuild needed -- unlike the native
    C++ passes (`tests/test_explicit_auto_pad.py` and friends), which need
    the matching pass compiled into the `onnxsim` in use."""
    onnxsim = pytest.importorskip("onnxsim")

    model = _same_upper_conv_model()
    x = np.zeros((1, 4, 10, 10), np.float32)
    sim_model, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"x": x},
        custom_rewriter=legalize.as_custom_rewriter(),
    )
    assert ok
    conv = next(n for n in sim_model.graph.node if n.op_type == "Conv")
    assert not any(
        a.name == "auto_pad" and a.s == b"SAME_UPPER" for a in conv.attribute
    )
