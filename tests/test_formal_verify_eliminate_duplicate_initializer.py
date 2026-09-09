"""Formal check for EliminateDuplicateInitializer (eliminate_duplicate_initializer.h).

This pass is structurally different from most others in this suite: it is a
``FullGraphBasedPass`` (whole-graph analysis via ``runPass(Graph&)``), not a
``PredicateBasedPass`` (per-node ``patternMatchPredicate``/``runTransform``).
It scans every initializer in the graph, groups them by byte-for-byte content
equality (``CSETensorHash``/``CSETensorEqual``, see ``cse_util.h``), keeps the
first initializer encountered in each group as the group's canonical
representative, and rewires every other (duplicate) initializer's uses onto
that representative before deleting the now-unused duplicates -- except an
initializer is skipped entirely (never considered, as either a duplicate or a
canonical target) if its name also appears as a graph-level input name or a
graph-level output name.

Formal content, and why the proof here is intentionally thin: this is not an
algebraic identity about some operator's semantics (unlike almost every other
pass in this suite) -- it is a substitution argument under value equality.
If two initializers ``A`` and ``B`` denote byte-for-byte identical tensor
content (same shape, same dtype, same values everywhere), then ``A`` and
``B`` are, semantically, the *same* tensor value stored under two different
graph names, so any consumer produces the same result fed either one. This is
modeled below as: an uninterpreted "content" function ``T : Int -> Real``,
two distinct named tensors ``A``, ``B`` each constrained to equal ``T``
pointwise (standing in for "same underlying content, two initializer
entries"), and an arbitrary uninterpreted ``consumer``. The content-equality
hypothesis does essentially all the work here -- there is no nontrivial
algebra to prove, unlike e.g. the Concat-splice or Transpose-composition
proofs elsewhere in this suite. The negative control below (unconstrained,
independent ``A``/``B``, no hypothesis) confirms the claim is not vacuously
true regardless of that hypothesis.

Given how thin the algebra is, the real engineering content of this pass --
and what the differential tests below actually exercise against the real
compiled pass -- is entirely in (a) getting the content-equality check right
(same shape, same dtype, same bytes -- confirmed empirically below, including
that *dtype* and *shape* both participate, not just the flat byte/value
sequence) and (b) the two exclusion rules (initializer-name-is-also-an-input,
initializer-name-is-also-an-output).

A subtlety in exercising the input-name exclusion through onnxsim, found
empirically: onnxsim's own Python-level preprocessing
(``remove_initializer_from_input`` in ``onnx_simplifier.py``) unconditionally
strips any ``graph.input`` entry whose name duplicates an initializer name
*before* the model ever reaches the C++ optimizer -- so by default this
pass's own ``input_set`` guard is never actually exercised via
``onnxsim.simplify()``/``simplify_isolated``: the duplicate input entry is
already gone by the time ``EliminateInitializer`` runs, and the
would-be-protected initializer participates in deduplication normally.
Passing ``mutable_initializer=True`` (a public ``simplify()`` parameter,
plumbed straight to the C++ core) disables that stripping and is what makes
this test file's input-exclusion test able to reach the pass's own guard at
all; it is *not* part of ``simplify_isolated``'s signature, so that one test
calls ``onnxsim.simplify`` directly (reusing ``isolate()`` for the skip
list) instead of going through the shared helper. No such preprocessing
exists for graph outputs, so the output-name exclusion test goes through
``simplify_isolated`` normally.

On ``simplify_isolated``/``isolate`` working for a ``FullGraphBasedPass``:
confirmed empirically below (every differential test here uses it except the
one that needs ``mutable_initializer``) -- ``isolate()`` builds its skip list
from ``C._list_optimizers()``/``skipped_optimizers`` purely by pass name, and
onnx-optimizer's pass-skipping applies uniformly regardless of whether a pass
is predicate- or full-graph-based, so nothing pass-family-specific needed
adjusting here.
"""

import numpy as np
import onnx
from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import helper, numpy_helper, parser

import onnxsim


def _tensor(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)


def test_eliminate_duplicate_initializer_is_sound():
    T = z3.Function("T", z3.IntSort(), z3.RealSort())
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")

    # "A and B are two initializer entries holding the same tensor content":
    # both equal the shared, uninterpreted content function T at every index.
    same_content = z3.ForAll([j], z3.And(A(j) == T(j), B(j) == T(j)))
    prove(z3.Implies(same_content, consumer(A(i)) == consumer(B(i))))


def test_eliminate_duplicate_initializer_negative_control_needs_hypothesis():
    # Without the content-equality hypothesis, A and B are two independent,
    # fully unconstrained uninterpreted functions: the claim must NOT be
    # valid then, or the "proof" above would be vacuously true regardless of
    # what same_content says. Z3 should find a sat counterexample negating
    # the claim, not unsat.
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i = z3.Int("i")

    solver = z3.Solver()
    solver.add(z3.Not(consumer(A(i)) == consumer(B(i))))
    assert solver.check() == z3.sat


def test_eliminate_duplicate_initializer_pass_matches():
    # A and B are byte-for-byte identical (same shape, dtype, values) and
    # each genuinely used (by a separate Add): the compiled pass merges them,
    # keeping the first-encountered one (A) as the survivor and rewiring F's
    # use of B onto A.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          E = Add(X, A)
          F = Add(X, B)
          G = Add(E, F)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("A", np.array(values).reshape(2, 2)),
            _tensor("B", np.array(values).reshape(2, 2)),
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_duplicate_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["A"]
    assert ops["Add"] == 3
    (e_node,) = [n for n in sim_model.graph.node if list(n.output) == ["E"]]
    (f_node,) = [n for n in sim_model.graph.node if list(n.output) == ["F"]]
    assert list(e_node.input) == ["X", "A"]
    assert list(f_node.input) == ["X", "A"]  # B's use rewired onto A


def test_eliminate_duplicate_initializer_declines_different_values():
    # Same shape and dtype, but different values at one position: not
    # byte-equal, so CSETensorCompare's raw_data memcmp fails and both
    # initializers survive untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          E = Add(X, A)
          F = Add(X, B)
          G = Add(E, F)
        }
        """
    )
    model.graph.initializer.extend(
        [
            _tensor("A", np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)),
            _tensor("B", np.array([1.0, 2.0, 3.0, 5.0]).reshape(2, 2)),
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_duplicate_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["A", "B"]
    assert ops["Add"] == 3


def test_eliminate_duplicate_initializer_declines_different_shape():
    # Same flat values/bytes, but a different declared shape ([4] vs [2,2]):
    # CSETensorCompare's `lhs->sizes() != rhs->sizes()` check fails before
    # the byte comparison ever runs, so shape participates in equality, not
    # just the flat value sequence -- both initializers survive untouched.
    #
    # Each initializer is consumed by an Add against a *runtime* (graph
    # input) tensor rather than folded through a Reshape/Cast -- onnxsim's
    # own constant-folding stage (separate from, and unaffected by,
    # `skipped_optimizers`) would otherwise fold a Reshape/Cast-of-constant
    # into a fresh, differently-shaped/typed initializer before this pass
    # ever runs, which would defeat the point of this test (confirmed
    # empirically while developing this test).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X, float[2,2] Y) => (float[4] E, float[2,2] F)
        {
          E = Add(X, A)
          F = Add(Y, B)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("A", np.array(values)),  # shape [4]
            _tensor("B", np.array(values).reshape(2, 2)),  # shape [2, 2]
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_duplicate_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["A", "B"]
    assert ops["Add"] == 2


def test_eliminate_duplicate_initializer_declines_different_dtype():
    # Same numeric values, but different element dtypes (int32 vs float32):
    # CSETensorCompare's `lhs->elem_type() != rhs->elem_type()` check fails,
    # so dtype participates in equality too -- both survive untouched. As in
    # the different-shape test above, each is consumed against a runtime
    # input (not via a Cast-of-constant) to avoid onnxsim's constant folding
    # normalizing the dtype mismatch away before this pass runs.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (int32[2,2] X, float[2,2] Y) => (int32[2,2] E, float[2,2] F)
        {
          E = Add(X, A)
          F = Add(Y, B)
        }
        """
    )
    values = [1, 2, 3, 4]
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.array(values, dtype=np.int32).reshape(2, 2), name="A"
            ),
            _tensor("B", np.array(values, dtype=np.float32).reshape(2, 2)),
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_duplicate_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["A", "B"]
    assert ops["Add"] == 2


def test_eliminate_duplicate_initializer_declines_input_named_initializer():
    # A's name also appears as a graph input (the "optional input with a
    # default value" pattern the header comment describes) -- A must be
    # skipped entirely: not merged away, and not usable as a merge target
    # for others. B and C are a separate content-identical pair (unrelated
    # to any input name) included so the pass still has real work to do;
    # if it did, that would show up as B/C merging normally while A alone
    # stays untouched.
    #
    # `mutable_initializer=True` is required to reach this guard at all:
    # onnxsim's default preprocessing strips any graph.input entry that
    # duplicates an initializer name *before* the C++ optimizer ever runs
    # (see this file's module docstring) -- with the default
    # `mutable_initializer=False`, A's duplicate input entry would already be
    # gone by the time this pass runs, and A would merge normally instead of
    # being protected, defeating the point of this test (confirmed
    # empirically while developing this test). This also means
    # `simplify_isolated` (which doesn't expose `mutable_initializer`) can't
    # be used here -- call `onnxsim.simplify` directly instead, reusing
    # `isolate()` for the skip list.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Ea, float[2,2] Eb, float[2,2] Ec)
        {
          Ea = Add(X, A)
          Eb = Add(X, B)
          Ec = Add(X, C)
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
        skipped_optimizers=isolate("eliminate_duplicate_initializer"),
        mutable_initializer=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"

    assert [i.name for i in sim_model.graph.initializer] == ["A", "B"]
    (ea_node,) = [n for n in sim_model.graph.node if list(n.output) == ["Ea"]]
    (eb_node,) = [n for n in sim_model.graph.node if list(n.output) == ["Eb"]]
    (ec_node,) = [n for n in sim_model.graph.node if list(n.output) == ["Ec"]]
    assert list(ea_node.input) == ["X", "A"]  # untouched: A is input-excluded
    assert list(eb_node.input) == ["X", "B"]
    assert list(ec_node.input) == ["X", "B"]  # C's use rewired onto B


def test_eliminate_duplicate_initializer_declines_output_named_initializer():
    # Y is a graph output produced directly by an initializer (no producing
    # node) -- Y must be skipped entirely, the same way an input-named
    # initializer is: not merged away, and not usable as a merge target. Z1
    # and Z2 are a separate content-identical pair so the pass still has
    # real work to do alongside Y being left alone.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Y, float[2,2] W1, float[2,2] W2)
        {
          W1 = Add(X, Z1)
          W2 = Add(X, Z2)
        }
        """
    )
    values = [1.0, 2.0, 3.0, 4.0]
    model.graph.initializer.extend(
        [
            _tensor("Y", np.array(values).reshape(2, 2)),
            _tensor("Z1", np.array(values).reshape(2, 2)),
            _tensor("Z2", np.array(values).reshape(2, 2)),
        ]
    )
    sim_model, ops = simplify_isolated(model, "eliminate_duplicate_initializer")
    assert [i.name for i in sim_model.graph.initializer] == ["Y", "Z1"]
    assert [o.name for o in sim_model.graph.output] == ["Y", "W1", "W2"]
    (w1_node,) = [n for n in sim_model.graph.node if list(n.output) == ["W1"]]
    (w2_node,) = [n for n in sim_model.graph.node if list(n.output) == ["W2"]]
    assert list(w1_node.input) == ["X", "Z1"]
    assert list(w2_node.input) == ["X", "Z1"]  # Z2's use rewired onto Z1
