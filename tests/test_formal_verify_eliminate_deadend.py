"""Formal check for EliminateDeadEnd (eliminate_deadend.h).

Like ``eliminate_duplicate_initializer.h`` (see
``test_formal_verify_eliminate_duplicate_initializer.py``), this pass is a
``FullGraphBasedPass`` (whole-graph analysis via ``runPass(Graph&)``), not a
``PredicateBasedPass``. It has been referenced repeatedly by other tests in
this suite (grep ``eliminate_deadend`` across ``tests/test_formal_verify_*``)
as the pass that would normally sweep up a node a rewrite leaves dangling
(0 uses) instead of destroying outright -- this file verifies that sweep
itself.

What it does, straight from the header: it walks the graph's nodes in
**reverse** order and destroys any node with no uses at all -- checked via
the cheap ``hasUsesInCurrentGraph()`` in the overwhelmingly common case, or
the more expensive ``hasUses()`` (which also accounts for a value captured
by a nested If/Loop/Scan subgraph body) only when
``GraphMayHaveCapturedValues(graph)`` says the graph could possibly contain
such a capturing op at all. The reverse order matters operationally: if node
M is node N's only consumer and M itself turns out to have no uses, M is
destroyed (dropping its inputs' use counts, including N's) *before* the
sweep reaches N -- so a single ``runPass`` invocation can cascade-remove an
entire chain of newly-dead nodes, not just directly-dead ones. See the
module comment in ``_formal_verify_common.py`` and e.g.
``test_formal_verify_eliminate_consecutive_idempotent_ops.py`` /
``test_formal_verify_fuse_consecutive_slices.py`` for real examples of the
"left dangling, eliminate_deadend would normally clean this up" pattern this
pass exists to handle.

Formal content, and why the proof here is intentionally thin (matching
``eliminate_duplicate_initializer``'s precedent for a genuinely "thin,
definitional" pass): this is not an algebraic identity about any operator's
semantics -- it is a claim about graph reachability/dependency structure. A
node whose output(s) are consumed by *nothing* -- no other node, no graph
output, and (accounting for ``hasUses()``) no captured use inside a nested
subgraph body either -- contributes nothing observable to what the graph
computes, so removing it cannot change the result. This is modeled below as
a small acyclic chain of named, uninterpreted-unary-function-defined values
-- ``v1 = f1(input)``, ``v2 = f2(v1)``, graph output ``= f3(v2)`` -- plus a
``dead = f4(v1)`` value that nothing downstream reads. The graph's output
expression simply never mentions ``dead`` syntactically, so two versions of
that expression -- one where ``dead``'s defining equation is additionally
asserted to exist, one where it is not -- are trivially, syntactically
identical; removing ``dead``'s definition (deleting the dead node) changes
nothing about the output. This is really an observation about *syntactic*
non-dependency rather than a nontrivial semantic derivation -- there is no
operator algebra to prove here, unlike almost every other pass in this
suite -- and that is expected and stated honestly, as
``eliminate_duplicate_initializer``'s own docstring does for its own
comparably thin claim. The negative control below (a deliberately wrong
model where the output expression is made to actually reference ``dead``)
confirms the "removing it changes nothing" argument is not vacuously true
regardless of what is being removed. A second proof extends the same
argument transitively to ``dead2 = f5(dead)`` -- a node that is only "dead"
*because* ``dead`` itself is dead -- tying directly to the cascading
reverse-iteration removal the real pass performs.

Given how thin the algebra is, the real engineering content -- what the
differential tests below actually exercise against the real compiled pass --
is entirely in the reachability/dependency logic itself: that a single
``runPass`` invocation removes a whole dead chain at once (the cascade case,
confirmed empirically below to be the single most important behavior to
verify, since a naive one-node-per-invocation removal would produce a
weaker, order-dependent result), that a live chain is left untouched
alongside a directly-dead node, that an all-live graph triggers no changes
at all, and that a value captured only inside a nested If subgraph body is
correctly *not* treated as dead even though it has no direct use in the
outer graph's own node list.

On single-invocation cascading, confirmed empirically (see this file's
development notes) rather than assumed: ``EliminateDeadEnd`` is constructed
with ``PassEfficiency::Complete`` (see the header), and
``CountBasedPassAnalysis::fixedPointOptimizationNeeded()``
(``pass.h``) only requests a repeat run for ``PassEfficiency::Partial``
passes -- so with only ``eliminate_deadend`` active (as ``isolate()``
arranges), ``FixedPointPassManager::run`` (``pass_manager.cc``) invokes
``runPass`` on it exactly once. The cascade test below observing all three
chained dead nodes gone after one ``simplify_isolated`` call is therefore
real evidence of the reverse-iteration cascade the header describes, not an
artifact of the pass being silently re-run by the fixed-point driver.

On the captured-value case: confirmed empirically below that it is not too
complex to construct via ``onnx.parser`` after all (the ``If`` node's
attribute-graph syntax needs no trailing ``()`` after the ``<...>`` -- see
``third_party/onnx/tests/cpp/parser_test.cc``'s ``IfNodeTest`` for the
working grammar) -- so this file includes that case rather than skipping it.
"""

from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import parser

import onnxsim


def _model(body, opset=16, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def test_eliminate_deadend_is_sound():
    f1 = z3.Function("f1", z3.RealSort(), z3.RealSort())
    f2 = z3.Function("f2", z3.RealSort(), z3.RealSort())
    f3 = z3.Function("f3", z3.RealSort(), z3.RealSort())
    f4 = z3.Function("f4", z3.RealSort(), z3.RealSort())
    inp = z3.Real("input")

    # v1 = f1(input), v2 = f2(v1), output = f3(v2): a small acyclic chain the
    # graph output actually depends on.
    v1 = f1(inp)
    v2 = f2(v1)
    output = f3(v2)

    # dead = f4(v1): a value defined from something live (v1), but nothing
    # downstream reads dead itself -- it is a genuine dead end, not merely
    # unreachable from the input.
    dead = f4(v1)

    # The claim: whether or not dead's defining equation is additionally
    # asserted, the graph's output expression is unaffected. `output` simply
    # never mentions `dead`/f4 syntactically, so this holds for any value of
    # `dead` at all -- stated here as an implication so the hypothesis
    # (dead's definition holding) is explicit, mirroring this suite's usual
    # "Implies(hypothesis, claim)" shape even though the hypothesis does no
    # work in discharging it.
    prove(z3.Implies(dead == f4(v1), output == f3(f2(f1(inp)))))


def test_eliminate_deadend_is_sound_cascade():
    # The cascade case: dead2 = f5(dead) is only a dead end *because* dead
    # itself is one -- neither is mentioned by the live output expression,
    # transitively. This is the algebraic counterpart of the real pass's
    # reverse-iteration cascade: removing dead2 first, then dead, still
    # leaves the live output expression untouched either way.
    f1 = z3.Function("f1", z3.RealSort(), z3.RealSort())
    f2 = z3.Function("f2", z3.RealSort(), z3.RealSort())
    f3 = z3.Function("f3", z3.RealSort(), z3.RealSort())
    f4 = z3.Function("f4", z3.RealSort(), z3.RealSort())
    f5 = z3.Function("f5", z3.RealSort(), z3.RealSort())
    inp = z3.Real("input")

    v1 = f1(inp)
    v2 = f2(v1)
    output = f3(v2)
    dead = f4(v1)
    dead2 = f5(dead)

    prove(
        z3.Implies(
            z3.And(dead == f4(v1), dead2 == f5(dead)),
            output == f3(f2(f1(inp))),
        )
    )


def test_eliminate_deadend_negative_control_needs_no_actual_dependency():
    # Sanity check that the proof above isn't vacuously true regardless of
    # what "dead" is: construct a deliberately WRONG model where the output
    # expression is made to actually reference dead (as if a dead node's
    # value leaked into a live computation), and confirm Z3 finds this is
    # NOT the same expression as the honest, dead-free output -- i.e. the
    # "removing an unused definition changes nothing" argument only works
    # because the real `output` genuinely never mentions `dead`.
    f1 = z3.Function("f1", z3.RealSort(), z3.RealSort())
    f2 = z3.Function("f2", z3.RealSort(), z3.RealSort())
    f3 = z3.Function("f3", z3.RealSort(), z3.RealSort())
    f4 = z3.Function("f4", z3.RealSort(), z3.RealSort())
    inp = z3.Real("input")

    v1 = f1(inp)
    v2 = f2(v1)
    output_without_dead = f3(v2)
    dead = f4(v1)
    output_actually_depends_on_dead = f3(v2) + dead  # bogus: not a dead end

    solver = z3.Solver()
    solver.add(z3.Not(output_without_dead == output_actually_depends_on_dead))
    assert solver.check() == z3.sat


def test_eliminate_deadend_pass_matches_single_dead_node():
    # A live chain (v1 -> v2 -> Y, reaching the graph output) alongside one
    # directly-dead node (dead, consumed by nothing): the compiled pass,
    # run alone, should remove exactly `dead` and leave the live chain
    # untouched.
    model = _model(
        """
        g (float[2,2] X) => (float[2,2] Y)
        {
          v1 = Relu(X)
          v2 = Sigmoid(v1)
          Y = Tanh(v2)
          dead = Neg(v1)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_deadend")
    assert ops == {"Relu": 1, "Sigmoid": 1, "Tanh": 1}
    assert [n.output[0] for n in sim_model.graph.node] == ["v1", "v2", "Y"]


def test_eliminate_deadend_pass_matches_cascade():
    # The key behavior: a chain of THREE dead-end nodes (dead1 consumed only
    # by dead2, dead2 only by dead3, dead3 by nothing), alongside a live
    # v1 -> Y chain. A naive single-pass-single-node removal would only
    # catch dead3 (the sole directly-0-use node before any removal) on one
    # invocation; the real pass's reverse node-order iteration instead
    # removes all three in this ONE simplify_isolated call, confirming the
    # header's cascade claim empirically -- and, per this pass's
    # PassEfficiency::Complete (see this file's module docstring), the
    # fixed-point driver never re-invokes it to catch stragglers, so this
    # result reflects genuinely a single runPass call.
    model = _model(
        """
        g (float[2,2] X) => (float[2,2] Y)
        {
          v1 = Relu(X)
          Y = Sigmoid(v1)
          dead1 = Neg(v1)
          dead2 = Abs(dead1)
          dead3 = Sqrt(dead2)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_deadend")
    assert ops == {"Relu": 1, "Sigmoid": 1}
    assert [n.output[0] for n in sim_model.graph.node] == ["v1", "Y"]


def test_eliminate_deadend_declines_when_everything_is_live():
    # Every node's output reaches a graph output (v1 feeds both Y1 and Y2):
    # nothing is a dead end, so the pass -- run alone -- makes no changes at
    # all.
    model = _model(
        """
        g (float[2,2] X) => (float[2,2] Y1, float[2,2] Y2)
        {
          v1 = Relu(X)
          Y1 = Sigmoid(v1)
          Y2 = Tanh(v1)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_deadend")
    assert ops == {"Relu": 1, "Sigmoid": 1, "Tanh": 1}
    assert [n.output[0] for n in sim_model.graph.node] == ["v1", "Y1", "Y2"]


def test_eliminate_deadend_keeps_value_captured_by_nested_if_subgraph():
    # v1's only use is inside the `then_branch`/`else_branch` subgraph
    # bodies of an If node (a captured outer value) -- it has NO use in the
    # outer graph's own direct node list, so hasUsesInCurrentGraph() alone
    # would (wrongly) call it a dead end. GraphMayHaveCapturedValues(graph)
    # sees the If op and makes the pass fall back to the more expensive,
    # subgraph-aware hasUses() instead, which correctly finds the captured
    # use and keeps v1. A separate, genuinely-unused `dead` node (reachable
    # from nothing, not even a subgraph) is included alongside it and
    # confirmed removed regardless -- i.e. the capture-awareness doesn't
    # just make the pass universally conservative once an If is present.
    #
    # The If node's attribute-graph syntax needs no trailing `()` after the
    # closing `>` -- confirmed against
    # third_party/onnx/tests/cpp/parser_test.cc's IfNodeTest grammar after
    # an initial guess with a trailing `()` failed to parse.
    model = _model(
        """
        g (float[4] X, bool cond) => (float[4] Y)
        {
          v1 = Relu(X)
          dead = Neg(X)
          Y = If (cond) <
              then_branch = g1 () => (float[4] z) { z = Sigmoid(v1) },
              else_branch = g2 () => (float[4] z) { z = Neg(v1) }
              >
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_deadend")
    assert ops == {"Relu": 1, "If": 1}
    assert [n.output[0] for n in sim_model.graph.node] == ["v1", "Y"]


def test_eliminate_deadend_is_a_real_default_pass():
    # Sanity check underpinning every "left dangling ... eliminate_deadend
    # would normally clean this up" comment elsewhere in this test suite:
    # eliminate_deadend is a genuine default onnxsim pass, and isolate()
    # (used throughout this suite, including above) skips it whenever a
    # *different* single pass is isolated -- e.g. isolate("eliminate_identity")
    # lists eliminate_deadend among the passes to skip, exactly as
    # simplify_isolated's own module docstring describes.
    assert "eliminate_deadend" in onnxsim.onnxsim_cpp2py_export._list_optimizers()
    assert "eliminate_deadend" in isolate("eliminate_identity")
