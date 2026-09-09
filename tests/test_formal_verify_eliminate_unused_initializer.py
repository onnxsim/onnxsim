"""Formal check for EliminateUnusedInitializer (eliminate_unused_initializer.h).

Like ``eliminate_duplicate_initializer`` (its closest sibling: read that file's
module docstring first, this one follows the same shape), this pass is a
``FullGraphBasedPass`` (whole-graph analysis via ``runPass(Graph&)``), not a
``PredicateBasedPass``. It computes, once per graph, the set of every
initializer name; walks every node's inputs (recursively descending into any
nested subgraph attribute too, via ``DescendOnGraphAttributesUnconstrained`` --
relevant for ``If``/``Loop``/``Scan`` bodies) and every graph output, erasing
from that set any initializer name actually referenced anywhere; and finally
deletes whatever names remain -- from the initializer list, and (if the same
name also happens to appear in the graph's input list) from the input list
too. The header states its two safety conditions plainly: "condition 1: A is
not used as any node's input; condition 2: A is not an output."

Formal content, and why the proof here is intentionally thin (same honest
framing as ``eliminate_duplicate_initializer``'s docstring): this is not an
algebraic identity about some operator's semantics -- there is no operator
algebra involved at all. An initializer erased by this pass is, by
construction, referenced by *nothing*: no node input anywhere in the graph
(including nested subgraphs), no graph output. Zero syntactic dependents means
removing it from the initializer list cannot change what any node computes or
what any graph output evaluates to, because nothing in the graph's computation
ever reads it in the first place. This is modeled below as literally as
possible: the graph's node structure and output are built as Z3 terms from
named, uninterpreted tensors and an uninterpreted ``Add``, and the "unused
initializer" is a value that is accepted as a parameter but never mentioned
anywhere inside that term -- i.e. syntactic non-dependency, not a derived
algebraic equivalence. Proving the output is invariant under any change to
that unmentioned value is close to a tautology, deliberately so: unlike
``eliminate_duplicate_initializer``'s content-equality/substitution argument,
there is no substitution reasoning to do here at all. The negative control
below (a variant output expression that actually *does* reference the
"unused" value) confirms the claim is not vacuously true regardless of
reference -- invariance genuinely fails once the value is actually used.

Given how thin the algebra is, the real engineering content of this pass --
and what the differential tests below actually exercise against the real
compiled pass -- is entirely in getting the two safety conditions right:
(a) an initializer consumed by any node anywhere, including inside a nested
``If``/``Loop``/``Scan`` subgraph, must survive; (b) an initializer that is
itself a graph output (no producing node -- a ``Value`` initialized straight
from a constant tensor and output verbatim) must survive; (c) a genuinely
unreferenced initializer must be dropped, from the initializer list and, if
present, the input list.

A surprise found empirically while developing this file's differential
tests, on point (c)'s input-list interaction: this pass has *no* protection
at all for an initializer whose name also happens to be a graph input -- in
sharp contrast to ``eliminate_duplicate_initializer``, which explicitly skips
any initializer named in the input set (never touching it either as a
duplicate or as a merge target). Here, the header's two conditions
("not a node input", "not an output") say nothing about graph inputs: if an
initializer is genuinely unused by that definition, it is erased regardless
of whether its name is also a graph input, and its ``graph.input`` entry is
erased right along with it (see ``eliminate_unused_initializer.h``'s own
``graph.eraseInput`` call). Reaching this cleanly through
``onnxsim.simplify()`` needs the same workaround as
``eliminate_duplicate_initializer``'s own input-name test, and for the same
underlying reason: onnxsim's own Python-level preprocessing
(``remove_initializer_from_input`` in ``onnx_simplifier.py``) unconditionally
strips any ``graph.input`` entry duplicating an initializer name *before* the
model reaches the C++ optimizer, so by default the duplicate input entry is
already gone before this pass has a chance to touch it either way. Passing
``mutable_initializer=True`` disables that stripping and is what lets the
differential test below actually observe the pass erasing the input entry
itself; since ``mutable_initializer`` is not part of ``simplify_isolated``'s
signature, that one test calls ``onnxsim.simplify`` directly (reusing
``isolate()`` for the skip list) instead of going through the shared helper,
exactly as ``eliminate_duplicate_initializer``'s own input-name test does.

A second, unrelated surprise found while developing these tests: onnxsim's
constant-folding stage (entirely separate from ``skipped_optimizers`` --
see ``onnxsim/constant_folding.cpp``) also invokes this exact same compiled
onnx-optimizer pass directly at the end of every fold call, to sweep up
initializers that folding itself leaves dangling (issue #174) -- independent
of whether the pass is in the active optimizer-pass list. This never changes
what the tests below observe (it is the identical real pass, so results
agree either way, and every model here is built with a genuinely-unused
initializer regardless of folding), but it does mean the pass under test
effectively runs even when ``skipped_optimizers`` tries to exclude it, so
long as constant folding itself is enabled (the default). Confirmed
empirically (throwaway script, not committed) by disabling constant folding
(``skip_constant_folding=True``) and isolating only this pass via
``skipped_optimizers``: behavior was identical to the default path.

The recursive subgraph-descent case (an initializer referenced only from
inside a nested ``If`` node's ``then_branch``) *was* reachable and is
exercised below, via ``onnx.parser``'s inline subgraph-attribute syntax
(``then_branch = g () => (...) { ... }`` inside the node's ``< ... >``
attribute list, no positional call parens after it -- confirmed empirically;
naively adding trailing ``()`` after the attribute list, as an ordinary node
call, is a parse error).
"""

import numpy as np
import onnx
from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import helper, numpy_helper, parser

import onnxsim


def _tensor(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)


def test_eliminate_unused_initializer_is_sound():
    # Uninterpreted stand-ins for the graph's node structure: B and C are
    # genuinely used (via D = Add(B, C), then G = Add(X, D)); the "unused
    # initializer" under test never appears in that structure at all.
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    C = z3.Function("C", z3.IntSort(), z3.RealSort())
    X = z3.Function("X", z3.IntSort(), z3.RealSort())
    Add = z3.Function("Add", z3.RealSort(), z3.RealSort(), z3.RealSort())
    i = z3.Int("i")

    def graph_output(unused_val):
        # `unused_val` stands for "whatever value the unused initializer A
        # might have held" -- accepted as a parameter here purely so the two
        # calls below can range over every possible such value, but never
        # actually used inside this expression. That is exactly condition 1
        # from the header comment ("A is not used as any node's input"): the
        # node structure (D, then G) is built entirely from B, C and X.
        d = Add(B(i), C(i))
        g = Add(X(i), d)
        return g

    # Since graph_output's term never mentions unused_val, its value cannot
    # depend on it: the graph's output is identical for every possible value
    # the removed initializer A could have held, so deleting A's initializer
    # entry changes nothing any node computes or any output evaluates to.
    u1, u2 = z3.Reals("u1 u2")
    prove(graph_output(u1) == graph_output(u2))


def test_eliminate_unused_initializer_negative_control_needs_reference():
    # If the output expression actually DID reference the "unused" value --
    # i.e. it were not actually unused -- the invariance claim above must NOT
    # be valid, or the "proof" would be vacuously true regardless of whether
    # anything depends on the value at all. Build a variant that plugs the
    # value into the output (`Add(g, unused_val)`) and confirm Z3 finds a sat
    # counterexample to invariance, not unsat.
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    C = z3.Function("C", z3.IntSort(), z3.RealSort())
    X = z3.Function("X", z3.IntSort(), z3.RealSort())
    Add = z3.Function("Add", z3.RealSort(), z3.RealSort(), z3.RealSort())
    i = z3.Int("i")

    def graph_output_referencing(referenced_val):
        d = Add(B(i), C(i))
        g = Add(X(i), d)
        return Add(g, referenced_val)  # now genuinely depends on the value

    u1, u2 = z3.Reals("u1 u2")
    solver = z3.Solver()
    solver.add(z3.Not(graph_output_referencing(u1) == graph_output_referencing(u2)))
    assert solver.check() == z3.sat


def test_eliminate_unused_initializer_pass_matches():
    # A is genuinely orphaned: referenced by no node and no graph output. B
    # and C are consumed together with the runtime input X (not with each
    # other alone), which keeps onnxsim's own constant-folding stage from
    # folding D = Add(B, C) away into a fresh initializer before this pass
    # ever runs (a pure Add(initializer, initializer) would constant-fold
    # entirely, defeating the point of a test about B/C staying untouched --
    # confirmed empirically while developing this test, the same class of
    # pitfall `eliminate_duplicate_initializer`'s own shape/dtype tests
    # avoid). The compiled pass, run alone, removes A and leaves B/C in
    # place.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          D = Add(X, B)
          G = Add(D, C)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("A", np.array(values).reshape(2, 2)),
            _tensor("B", np.array(values).reshape(2, 2)),
            _tensor("C", np.array(values).reshape(2, 2)),
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_unused_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["B", "C"]
    assert ops["Add"] == 2


def test_eliminate_unused_initializer_declines_when_all_used():
    # Every initializer is actually consumed (B and C, both via Add against
    # the runtime input X): nothing is unused, so the pass makes no changes.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          D = Add(X, B)
          G = Add(D, C)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("B", np.array(values).reshape(2, 2)),
            _tensor("C", np.array(values).reshape(2, 2)),
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_unused_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["B", "C"]
    assert ops["Add"] == 2


def test_eliminate_unused_initializer_protects_output_named_initializer():
    # Y is a graph output produced directly by an initializer (no producing
    # node) -- condition 2 from the header ("A is not an output") must
    # protect it. Z is a separate, genuinely-unreferenced initializer
    # included so the pass still has real work to do alongside Y surviving.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Y, float[2,2] W)
        {
          W = Add(X, B)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("Y", np.array(values).reshape(2, 2)),
            _tensor("B", np.array(values).reshape(2, 2)),
            _tensor("Z", np.array(values).reshape(2, 2)),  # genuinely unused
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_unused_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["Y", "B"]
    assert [o.name for o in sim_model.graph.output] == ["Y", "W"]
    assert ops["Add"] == 1


def test_eliminate_unused_initializer_removes_orphan_named_as_input():
    # A is genuinely orphaned (no node input, no output) AND its name also
    # appears in graph.input -- unlike eliminate_duplicate_initializer, this
    # pass has no exclusion for that case (the header's two conditions say
    # nothing about graph inputs): A is erased, and per the pass's own
    # `graph.eraseInput` call, its graph.input entry is erased right along
    # with it. B and C are consumed via the runtime input X so they survive
    # untouched, showing the pass still discriminates correctly alongside A's
    # removal.
    #
    # `mutable_initializer=True` is required to observe this cleanly:
    # onnxsim's default preprocessing strips any graph.input entry that
    # duplicates an initializer name *before* the C++ optimizer ever runs
    # (see this file's module docstring) -- with the default
    # `mutable_initializer=False`, A's duplicate input entry would already be
    # gone before this pass runs, leaving the "does the pass itself erase the
    # input entry" question unanswered (confirmed empirically while
    # developing this test). This also means `simplify_isolated` (which
    # doesn't expose `mutable_initializer`) can't be used here -- call
    # `onnxsim.simplify` directly instead, reusing `isolate()` for the skip
    # list, exactly as `eliminate_duplicate_initializer`'s own input-name
    # test does.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          D = Add(X, B)
          G = Add(D, C)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("A", np.array(values).reshape(2, 2)),
            _tensor("B", np.array(values).reshape(2, 2)),
            _tensor("C", np.array(values).reshape(2, 2)),
        ]
    )
    model.graph.input.append(
        helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 2])
    )

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_unused_initializer"),
        mutable_initializer=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"

    assert [i.name for i in sim_model.graph.initializer] == ["B", "C"]
    assert [i.name for i in sim_model.graph.input] == ["X"]  # A's input erased too


def test_eliminate_unused_initializer_recursive_subgraph_descent():
    # InnerA is referenced ONLY from inside a nested If node's then_branch
    # subgraph -- not by any node in the outer graph's own node list. A naive
    # implementation that only checked the outer graph's direct node inputs
    # would incorrectly consider InnerA dead; the real pass's
    # `DescendOnGraphAttributesUnconstrained` walk correctly recognizes it as
    # used and leaves it (and the outer graph's initializer list) untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        agraph (float[2,2] X, bool Cond) => (float[2,2] Out)
        {
          Out = If (Cond) <
              then_branch = then_g () => (float[2,2] ThenOut) { ThenOut = Add(X, InnerA) },
              else_branch = else_g () => (float[2,2] ElseOut) { ElseOut = Identity(X) }
              >
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.append(_tensor("InnerA", np.array(values).reshape(2, 2)))
    sim_model, _ = simplify_isolated(model, "eliminate_unused_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["InnerA"]
