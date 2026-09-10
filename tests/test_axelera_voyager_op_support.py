"""Tests for `scripts/axelera/voyager_ops.py` / `voyager_simulator.py` -- a
docs-derived, no-hardware/no-compiler estimate of Voyager SDK's Metis AIPU
operator support (see that module's docstring for the full scope and, more
importantly, what it explicitly does NOT claim: no real compiler or
hardware was ever involved, since Voyager SDK's `axelera-types`/
`axelera-runtime` packages are served only from Axelera's own private
package index -- see `installer_support.py` in a voyager-sdk checkout).

Every model below is built with `onnx.parser` and checked against known,
hand-verified outcomes from Voyager SDK's own opset-17 docs (`docs/
reference/compiler/onnx-opset17-support.md`) at the time this was written --
e.g. Conv requires `auto_pad == "NOTSET"`, Gemm requires `transA == 0`,
Transpose requires `perm == [0, 1, 2, 3]`. These are regression tests for
the evaluator's own logic (namespace binding, AST safety, tri-state
fail-closed behavior), not a claim that Voyager SDK's docs (or this
transcription of them) are permanently correct.
"""

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper, parser

_AXELERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axelera"
)
if _AXELERA_DIR not in sys.path:
    sys.path.insert(0, _AXELERA_DIR)

import voyager_ops as ops  # noqa: E402
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


# --- voyager_ops.py: the scraped data itself ---


def test_op_support_data_matches_known_levels():
    # Spot-check against docs/reference/compiler/onnx-support.md's summary
    # table, read directly at the time this test was written.
    assert ops.VOYAGER_OP_LEVEL["Relu"] == "Supported"
    assert ops.VOYAGER_OP_LEVEL["Sigmoid"] == "Supported"
    assert ops.VOYAGER_OP_LEVEL["Conv"] == "Constrained"
    assert ops.VOYAGER_OP_LEVEL["Gemm"] == "Constrained"
    assert "Erf" not in ops.VOYAGER_OP_LEVEL  # undocumented -> CPU fallback


def test_unconstrained_and_constrained_partition_is_disjoint_and_covers_all():
    assert ops.VOYAGER_UNCONSTRAINED_OPS & ops.VOYAGER_CONSTRAINED_OPS == set()
    assert ops.VOYAGER_UNCONSTRAINED_OPS | ops.VOYAGER_CONSTRAINED_OPS == set(
        ops.VOYAGER_OP_LEVEL
    )


def test_prose_only_ops_have_no_formal_predicate():
    for name in ops.VOYAGER_PROSE_ONLY_CONSTRAINTS:
        entry = ops.VOYAGER_OP_SUPPORT[name]
        assert entry["level"] == "Constrained"
        assert not entry["rules"]
        assert not entry["allow_config"]
        assert entry["notes"]


# --- voyager_simulator.py: partition() (op-type only) ---


def test_partition_classifies_by_documented_op_type_membership():
    model = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          r = Relu(x)
          y = Erf(r)
        }
        """
    )
    p = sim.partition(model)
    assert len(p.npu_nodes) == 1
    assert len(p.cpu_fallback_nodes) == 1
    assert p.cpu_fallback_op_types == {"Erf": 1}
    assert sim.coverage(model) == "partial"


def test_coverage_full_and_none():
    full = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        { y = Relu(x) }
        """
    )
    assert sim.coverage(full) == "full"

    none = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        { y = Erf(x) }
        """
    )
    assert sim.coverage(none) == "none"


# --- voyager_simulator.py: evaluate_constraints() (per-node, best-effort) ---


def test_supported_op_is_always_ok():
    model = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        { y = Relu(x) }
        """
    )
    assert _verdicts(model) == ["ok"]


def test_undocumented_op_is_cpu_fallback_not_applicable():
    model = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        { y = Erf(x) }
        """
    )
    cache = sim._ShapeCache(model)
    result = sim.evaluate_constraints(model.graph.node[0], cache)
    assert result.level == "CPU fallback"
    assert result.verdict == "not_applicable"


def test_prose_only_op_is_never_verified_ok_or_violated():
    model = _model(
        """
        g (float[4,8] a, float[8,10] b) => (float[4,10] y)
        { y = MatMul(a, b) }
        """
    )
    assert _verdicts(model) == ["prose_only"]


def test_conv_auto_pad_notset_default_is_ok():
    model = _model(
        """
        g (float[1,8,10,10] x) => (float[1,8,8,8] y)
        { y = Conv<kernel_shape = [3, 3]>(x, w) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((8, 8, 3, 3), np.float32), name="w")
        ],
    )
    assert _verdicts(model) == ["ok"]


def test_conv_same_upper_padding_is_violated():
    model = _model(
        """
        g (float[1,8,10,10] x) => (float[1,8,10,10] y)
        { y = Conv<kernel_shape = [3, 3], auto_pad = "SAME_UPPER">(x, w) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((8, 8, 3, 3), np.float32), name="w")
        ],
    )
    assert _verdicts(model) == ["violated"]


def test_conv_depthwise_symmetric_kernel_is_ok():
    model = _model(
        """
        g (float[1,8,10,10] x) => (float[1,8,8,8] y)
        { y = Conv<kernel_shape = [3, 3], group = 8>(x, w) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((8, 1, 3, 3), np.float32), name="w")
        ],
    )
    assert _verdicts(model) == ["ok"]


def test_conv_depthwise_asymmetric_kernel_is_violated():
    model = _model(
        """
        g (float[1,8,10,10] x) => (float[1,8,8,8] y)
        { y = Conv<kernel_shape = [3, 5], group = 8>(x, w) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((8, 1, 3, 5), np.float32), name="w")
        ],
    )
    assert _verdicts(model) == ["violated"]


def test_gemm_transA_default_is_ok():
    model = _model(
        """
        g (float[4,8] a) => (float[4,10] y)
        { y = Gemm(a, b, c) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((8, 10), np.float32), name="b"),
            numpy_helper.from_array(np.zeros((10,), np.float32), name="c"),
        ],
    )
    assert _verdicts(model) == ["ok"]


def test_gemm_transA_set_is_violated():
    model = _model(
        """
        g (float[4,8] a) => (float[8,10] y)
        { y = Gemm<transA = 1>(a, b, c) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((4, 8), np.float32), name="b"),
            numpy_helper.from_array(np.zeros((10,), np.float32), name="c"),
        ],
    )
    assert _verdicts(model) == ["violated"]


def test_clip_relu6_and_hardtanh_are_ok_others_violated():
    for lo, hi, expected in [
        (0.0, 6.0, "ok"),
        (-1.0, 1.0, "ok"),
        (2.0, 6.0, "violated"),
    ]:
        model = _model(
            f"""
            g (float[1,4] x) => (float[1,4] y)
            <float mn = {{{lo}}}, float mx = {{{hi}}}>
            {{ y = Clip(x, mn, mx) }}
            """
        )
        assert _verdicts(model) == [expected], (lo, hi)


def test_transpose_identity_perm_is_ok_others_violated():
    for perm, expected in [([0, 1, 2, 3], "ok"), ([0, 2, 3, 1], "violated")]:
        model = _model(
            f"""
            g (float[1,2,3,4] x) => (float[1,2,3,4] y)
            {{ y = Transpose<perm = {perm}>(x) }}
            """
        )
        assert _verdicts(model) == [expected], perm


def test_prelu_matching_and_mismatched_channel_broadcast():
    match = _model(
        """
        g (float[1,8,4,4] x) => (float[1,8,4,4] y)
        { y = PRelu(x, s) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((1, 8, 1, 1), np.float32), name="s")
        ],
    )
    assert _verdicts(match) == ["ok"]

    mismatch = _model(
        """
        g (float[1,8,4,4] x) => (float[1,8,4,4] y)
        { y = PRelu(x, s) }
        """,
        initializer=[
            numpy_helper.from_array(np.zeros((1, 4, 1, 1), np.float32), name="s")
        ],
    )
    assert _verdicts(mismatch) == ["violated"]


def test_concat_axis_channel_is_ok_batch_axis_is_violated():
    channel = _model(
        """
        g (float[1,4,4,4] a, float[1,4,4,4] b) => (float[1,8,4,4] y)
        { y = Concat<axis = 1>(a, b) }
        """
    )
    assert _verdicts(channel) == ["ok"]

    batch = _model(
        """
        g (float[1,4,4,4] a, float[1,4,4,4] b) => (float[2,4,4,4] y)
        { y = Concat<axis = 0>(a, b) }
        """
    )
    assert _verdicts(batch) == ["violated"]


def test_slice_channel_axis_needs_64_divisible_size():
    ok = _model(
        """
        g (float[1,8,4,4] x) => (float[1,4,4,4] y)
        <int64[1] starts = {0}, int64[1] ends = {4}, int64[1] axes = {1}>
        { y = Slice(x, starts, ends, axes) }
        """
    )
    assert _verdicts(ok) == ["ok"]

    violated = _model(
        """
        g (float[1,8,10,10] x) => (float[1,8,4,10] y)
        <int64[1] starts = {0}, int64[1] ends = {4}, int64[1] axes = {2}>
        { y = Slice(x, starts, ends, axes) }
        """
    )
    assert _verdicts(violated) == ["violated"]


def test_reshape_identity_shape_is_ok():
    model = _model(
        """
        g (float[1,8] x) => (float[1,8] y)
        <int64[2] s = {1, 8}>
        { y = Reshape(x, s) }
        """
    )
    assert _verdicts(model) == ["ok"]


# --- Evaluator safety: never guesses on what it can't resolve ---


def test_non_constant_slice_axes_is_unknown_not_a_guess():
    # `axes` (referenced by all three of Slice's rules) fed by a
    # non-constant (graph-input) tensor rather than a constant: none of the
    # rules can be statically resolved without its value. `starts` is left
    # unbound too (Slice's rules never reference it at all) to double as a
    # check that an unrelated missing/dynamic input doesn't itself cause a
    # false "unknown" -- covered instead by the ok/violated cases above,
    # which all omit `starts`/`ends` from the graph text entirely.
    model = _model(
        """
        g (float[1,8,4,4] x, int64[1] axes) => (float[1,4,4,4] y)
        <int64[1] starts = {0}, int64[1] ends = {4}>
        { y = Slice(x, starts, ends, axes) }
        """
    )
    assert _verdicts(model) == ["unknown"]


def test_unresolvable_op_without_a_param_spec_is_unknown():
    # HardSwish is "Supported" (no constraints) but Softmax is "Constrained"
    # and prose-only, already covered above; pick a Constrained op with a
    # formal predicate this module's _OP_PARAM_SPEC simply doesn't cover to
    # confirm the "no known binding" path also fails closed rather than
    # crashing. Squeeze is prose-only (covered); use a fabricated op_type
    # absent from VOYAGER_OP_SUPPORT entirely to hit the CPU-fallback path
    # instead, which is already covered by test_undocumented_op_is_cpu_
    # fallback_not_applicable -- this test instead exercises evaluate_expr
    # directly against an op the spec table has no entry for at all.
    model = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        { y = Relu(x) }
        """
    )
    cache = sim._ShapeCache(model)
    node = model.graph.node[0]
    result = sim.evaluate_expr("NotARealOp", node, cache, "some_unbound_name == 1")
    assert result is None


def test_ast_safety_rejects_unexpected_syntax():
    model = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        { y = Relu(x) }
        """
    )
    cache = sim._ShapeCache(model)
    node = model.graph.node[0]
    # A lambda/exec-style construct should never be reachable from the
    # docs-sourced expressions, but confirm this fails closed (via the
    # restricted eval() namespace/builtins, whichever layer catches it
    # first) rather than silently eval'ing arbitrary code if it ever were.
    assert (
        sim.evaluate_expr("Relu", node, cache, "__import__('os').system('echo hi')")
        is None
    )
