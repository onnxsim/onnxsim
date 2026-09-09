"""Formal check for EliminateNopMonotoneArgmax (eliminate_nop_monotone_argmax.h).

``patternMatchPredicate`` matches an ``ArgMax`` node (which always carries its
own ``axis`` attribute) whose *sole* input is the output of a node
``satisfies_monotone_condition(argmax_axis, that_node)``. Two disjoint op-kind
sets are recognized:

* ``monotone_node_no_axis_kind = {Log, Exp, Sqrt}`` -- elementwise ops with no
  axis-dependence, so the condition holds unconditionally (regardless of
  ArgMax's ``axis``).
* ``monotone_node_axis_kind = {Softmax, LogSoftmax}`` -- these mix values
  *across* one axis, so they are only monotone *along* that specific axis;
  the condition requires the inner node's own ``axis`` attribute to exactly
  equal ArgMax's ``axis`` (``node->hasAttribute(kaxis) && axis ==
  node->i(kaxis)``).

``runTransform`` fires only when the monotone node's output has exactly one
use (this ArgMax is its sole consumer): it rewires ArgMax's input to read
directly from the monotone node's *own* input, then destroys the monotone
node (``tryReplacingAllUsesWith`` + ``destroy()``).

The source carries its own caveat, reproduced here verbatim:

    Note for log and sqrt this optimization is not always right, because it
    is a undefined behavior when the input is negative

Log's mathematical domain is ``x > 0`` and Sqrt's is ``x >= 0``; ArgMax(Log(X))
== ArgMax(X) (and likewise for Sqrt) only holds when Log(X)/Sqrt(X) is even
defined, i.e. every element of X lies in that domain. Both are still strictly
increasing *on their domain*, so the algebra below is exactly as sound for
them there as for Exp/Softmax/LogSoftmax -- but ``satisfies_monotone_condition``
itself never inspects a single value, only op-kind and (for the axis-kind
ops) the static ``axis`` attribute, so the compiled pass fires for Log/Sqrt
regardless of whether X is actually positive. That is a real, acknowledged
scope gap in the pass (matching the C++ comment above): it assumes graphs
never feed invalid input to Log/Sqrt in the first place, and does not check
it. This file proves the rewrite's algebra sound as a lemma about a strictly
monotone function, and separately confirms empirically (below) that the
pass's predicate is indeed unconditional/value-independent for Log -- it is
not claiming Log/Sqrt's domain caveat away, only documenting it precisely as
the honest boundary of what's proven.

Formal content: argmax over an axis is invariant under composing with any
function that is strictly increasing along that axis (holding the other
coordinates fixed). This is modeled *once*, generally, as an uninterpreted
``f: Real -> Real`` with the hypothesis ``ForAll([a, b], a > b => f(a) >
f(b))`` -- the one property every one of Exp, Softmax-along-its-axis and
LogSoftmax-along-its-axis genuinely has (Softmax/LogSoftmax restricted to
varying only the coordinate along their own axis, the rest held fixed,
exactly mirrors this univariate abstraction). For three symbolic "positions"
along the reduced axis (``x0, x1, x2`` -- enough for "the argmax index" to be
a genuine multi-way comparison, not a degenerate 1- or 2-element case), the
index of the maximum among ``{x0, x1, x2}`` is proved equal to the index of
the maximum among ``{f(x0), f(x1), f(x2)}``, with ties broken by preferring
the lowest index (matching how ``argmax_index`` below is written; strict
monotonicity makes ``f`` an order-isomorphism -- both ``>`` and ``==`` between
any two positions are preserved -- so the tie-break outcome is preserved too,
not just the strict comparisons).
"""

import numpy as np
import onnx.numpy_helper
from _formal_verify_common import isolate, producer, prove, simplify_isolated, z3
from onnx import parser

import onnxsim


def _argmax_index(v0, v1, v2):
    # First-occurring-max convention (ties prefer the lowest index) over
    # exactly 3 positions -- a genuine 3-way comparison, not a 1- or
    # 2-element degenerate case.
    return z3.If(z3.And(v0 >= v1, v0 >= v2), 0, z3.If(v1 >= v2, 1, 2))


def test_eliminate_nop_monotone_argmax_is_sound():
    f = z3.Function("f", z3.RealSort(), z3.RealSort())
    a, b = z3.Reals("a b")
    x0, x1, x2 = z3.Reals("x0 x1 x2")

    # The one property Exp / Softmax-along-its-axis / LogSoftmax-along-its-axis
    # genuinely share: strictly increasing.
    strictly_increasing = z3.ForAll([a, b], z3.Implies(a > b, f(a) > f(b)))

    prove(
        z3.Implies(
            strictly_increasing,
            _argmax_index(x0, x1, x2) == _argmax_index(f(x0), f(x1), f(x2)),
        )
    )


def test_eliminate_nop_monotone_argmax_pass_matches_exp():
    # Exp is in monotone_node_no_axis_kind and qualifies regardless of
    # ArgMax's axis (Exp has no axis-dependence at all): the pass, run
    # alone, rewires ArgMax to read X directly and destroys Exp.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4] Y)
        {
          e = Exp(X)
          Y = ArgMax<axis = 1, keepdims = 0>(e)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_monotone_argmax")
    assert ops["Exp"] == 0
    assert ops["ArgMax"] == 1
    argmax_node = producer(sim_model, "Y")
    assert list(argmax_node.input) == ["X"]


def test_eliminate_nop_monotone_argmax_pass_matches_softmax_same_axis():
    # Softmax is in monotone_node_axis_kind and only qualifies when its own
    # axis matches ArgMax's -- here both are axis=1, so the pass fires.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4] Y)
        {
          s = Softmax<axis = 1>(X)
          Y = ArgMax<axis = 1, keepdims = 0>(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_monotone_argmax")
    assert ops["Softmax"] == 0
    assert ops["ArgMax"] == 1
    argmax_node = producer(sim_model, "Y")
    assert list(argmax_node.input) == ["X"]


def test_eliminate_nop_monotone_argmax_pass_matches_log_positive_input():
    # Log is in monotone_node_no_axis_kind (like Exp, unconditionally
    # qualifying regardless of ArgMax's axis) -- the pass fires. X is a
    # plain graph input with no initializer, so onnxsim's own numeric
    # --check (run by simplify_isolated, check_n=3, the default "random"
    # input_fill) samples it uniformly from [0, 1) -- always non-negative --
    # so Log(X) stays well-defined here and the check meaningfully confirms
    # the rewritten graph still agrees with the original.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4] Y)
        {
          l = Log(X)
          Y = ArgMax<axis = 1, keepdims = 0>(l)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_monotone_argmax")
    assert ops["Log"] == 0
    assert ops["ArgMax"] == 1
    argmax_node = producer(sim_model, "Y")
    assert list(argmax_node.input) == ["X"]


def test_eliminate_nop_monotone_argmax_pass_fires_on_log_unconditionally():
    # satisfies_monotone_condition never inspects a single tensor value --
    # only node->kind() and (for the axis-kind set) the static axis
    # attribute. So the pass's *own predicate* cannot depend on whether X is
    # actually positive; it fires for Log even when X contains negative
    # entries, which is exactly the documented (and unchecked) caveat in the
    # C++ source. X is given a concrete initializer with negative entries to
    # make that observable, with skip_constant_folding=True (otherwise
    # onnxsim's constant folder -- a separate step from the optimizer-pass
    # list skipped_optimizers controls -- would just evaluate the whole
    # graph away, which would prove nothing about this pass specifically)
    # and check_n=0 (Log(negative) is undefined, so a numeric equivalence
    # check is not meaningful on this model -- this test is only about
    # whether the *pass* rewrites the graph, not about numeric correctness
    # of doing so).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4] Y)
        {
          l = Log(X)
          Y = ArgMax<axis = 1, keepdims = 0>(l)
        }
        """
    )
    x = np.array([[-1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * 4, dtype=np.float32)
    model.graph.initializer.append(onnx.numpy_helper.from_array(x, name="X"))

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        skip_constant_folding=True,
        skipped_optimizers=isolate("eliminate_nop_monotone_argmax"),
    )
    assert check_ok  # trivially true: check_n=0 runs no numeric samples
    ops = {n.op_type for n in sim_model.graph.node}
    assert "Log" not in ops
    argmax_node = producer(sim_model, "Y")
    assert list(argmax_node.input) == ["X"]


def test_eliminate_nop_monotone_argmax_declines_different_axis():
    # Softmax's own axis (1) differs from ArgMax's axis (0):
    # satisfies_monotone_condition requires exact equality for the
    # axis-kind set, so the predicate declines and both nodes survive.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[8] Y)
        {
          s = Softmax<axis = 1>(X)
          Y = ArgMax<axis = 0, keepdims = 0>(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_monotone_argmax")
    assert ops["Softmax"] == 1
    assert ops["ArgMax"] == 1


def test_eliminate_nop_monotone_argmax_declines_multi_use():
    # Exp's output is consumed both by ArgMax and directly as a second
    # graph output, so monotone_node->output()->uses().size() == 1 fails --
    # runTransform declines and both nodes survive, chained exactly as
    # before.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4] Y, float[4,8] E)
        {
          e = Exp(X)
          Y = ArgMax<axis = 1, keepdims = 0>(e)
          E = Identity(e)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_monotone_argmax")
    assert ops["Exp"] == 1
    assert ops["ArgMax"] == 1
    argmax_node = producer(sim_model, "Y")
    assert list(argmax_node.input) == ["e"]


def test_eliminate_nop_monotone_argmax_declines_non_monotone_op():
    # Relu is in neither monotone_node_no_axis_kind nor
    # monotone_node_axis_kind -- correctly so, since Relu is flat (constant
    # at 0) for all non-positive inputs and hence not globally strictly
    # monotone: ArgMax(Relu(X)) can genuinely differ from ArgMax(X) whenever
    # more than one entry along the axis is non-positive (all such entries
    # tie at 0 under Relu, even though they differ, and may differ from X's
    # own true maximum). satisfies_monotone_condition simply returns false
    # for kRelu, so the pass, run alone, must leave both nodes untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4] Y)
        {
          r = Relu(X)
          Y = ArgMax<axis = 1, keepdims = 0>(r)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_monotone_argmax")
    assert ops["Relu"] == 1
    assert ops["ArgMax"] == 1
