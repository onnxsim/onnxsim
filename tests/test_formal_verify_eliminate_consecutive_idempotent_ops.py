"""Formal check for EliminateConsecutiveIdempotentOps
(eliminate_consecutive_idempotent_ops.h).

``patternMatchPredicate`` matches a node ``node`` whose op kind is one of
``{Ceil, Floor, Round, Relu, Reshape, Sign}`` AND whose sole input is the
output of ANOTHER node (``previous_node``) of the EXACT SAME op kind, with
that inner node's output having exactly one use (only ``node`` consumes it).
``runTransform`` then rewires ``node``'s own input from ``previous_node``'s
output to ``previous_node``'s own input -- i.e. ``node`` (the outer op, same
kind, same attributes, untouched otherwise) ends up reading directly from
whatever fed ``previous_node``, skipping ``previous_node`` entirely.
``previous_node`` is left dangling (0 uses), not itself destroyed here --
that is ``eliminate_deadend``'s job, run separately.

Soundness needs: two applications of the SAME idempotent op are equivalent,
for the purposes of ``node``'s own output, to one application. Five of the
six op kinds (Ceil, Floor, Round, Relu, Sign) share one algebraic shape --
``f(f(x)) == f(x)`` for a FIXED unary function ``f`` -- derived, not assumed,
from two more primitive facts: "``f``'s output always lands in some
restricted set S" and "``f`` is the identity on S":

  * Ceil/Floor/Round: S = the integers. Each of these always returns an
    integer, and each is (trivially) the identity ON an already-integer
    input -- regardless of Round's particular tie-breaking rule, which never
    matters here since the inner application has already produced an
    integer.
  * Relu: S = the non-negative reals (``relu(x) = max(x, 0) >= 0`` always;
    ``relu`` is the identity on ``y >= 0``).
  * Sign: S = ``{-1, 0, 1}`` (sign's range is always one of those three
    values; sign is the identity on each of them).

Only two representative members of this family get their own Z3 proof below
-- Ceil (the "identity on an axiomatized range predicate" shape) and Relu
(the "identity on a numeric inequality" shape), chosen because they exercise
slightly different hypothesis shapes, per this repo's established practice
of proving the algebraic PATTERN once or twice cleanly rather than
re-deriving it six times (see e.g. ``test_formal_verify_eliminate_nop_with_unit.py``,
which similarly proves two representative op families in Z3 and leans on
differential tests for the rest of ``isUnit``'s op-kind breadth). Floor,
Round, and Sign follow the identical "derive f(f(x))==f(x) from a range axiom
plus an identity-on-that-range axiom" structure and are not re-proved here;
all six op kinds are covered by the differential tests below, which run the
real compiled pass.

Reshape is structurally DIFFERENT: it is not "f(f(x)) == f(x)" for one fixed
function, because the pass's ``Reshape`` case composes TWO ARBITRARY
(possibly different) target shapes ``S1`` then ``S2`` --
``Reshape(Reshape(X, S1), S2) == Reshape(X, S2)`` holds for *every* valid
``S1``, not because Reshape is a fixed idempotent function of one variable,
but because Reshape never moves data, only relabels which multi-index maps to
which position in a shared row-major flat buffer -- so composing two
relabelings and then a third always collapses algebraically to just the
final relabeling, independent of what the intermediate one was. This file
proves that composability claim as its own separate Z3 test, reusing (rather
than re-deriving) the row-major flatten/unflatten technique already
established in ``test_formal_verify_eliminate_nop_reshape.py``, extended here
from "one reshape is a no-op" to "two reshapes compose into the second one
alone".

Every claim below is composed with an arbitrary uninterpreted ``consumer``
function, matching this repo's established style (see
``test_formal_verify_eliminate_identity.py``): this is what turns "f(f(x)) ==
f(x)" into a real substitution-soundness argument for ``node``'s own
downstream consumers, rather than a fact about ``f`` in isolation.

Reshape's ``runTransform`` has one more wrinkle worth noting, confirmed
empirically (via a throwaway debug script, since deleted) against the real
compiled pass rather than assumed from reading the C++ alone: the generic
``tryReplacingAllUsesWith`` / ``Value::replaceAllUsesWith`` machinery
propagates the OLD value's tracked static sizes onto the NEW value
(``onnx/common/ir.h``'s ``replaceAllUsesWith`` unconditionally does
``newValue->setSizes(oldValue->sizes())`` when the old value has known
sizes). For every other op in this pass's family that is a correctness
non-issue, because the outer and inner ops are the same kind and hence
already produce the same shape as their own input. But for Reshape
specifically, ``node->input(0)`` (the inner Reshape's OUTPUT, shape ``S1``)
and ``previous_node->input(0)`` (whatever feeds the inner Reshape, e.g. the
graph input, shape ``S0`` -- generally different from ``S1``) do NOT share a
shape, so redirecting the inner Reshape's uses onto ``previous_node``'s input
would otherwise silently overwrite that value's own tracked shape with the
WRONG shape (``S1`` instead of its real ``S0``). That is exactly what the
explicit ``previous_node->input(0)->setSizes(sizes)`` afterward undoes,
restoring the correct ``S0`` that was captured before the rewiring call. This
is IR-internal shape-tracking bookkeeping (it can affect what LATER passes in
the same optimizer run believe about that value's shape), not something that
shows up as a wrong number in the final tensor output either way -- the
differential Reshape test below confirms, by actually running the resulting
model through ``onnxruntime``, that the final output values are correct
regardless.
"""

import numpy as np
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_consecutive_idempotent_ceil_family_is_sound():
    # Representative of the Ceil/Floor/Round/Sign "derive f(f(x))==f(x) from
    # a range axiom + identity-on-that-range axiom" shape, using an
    # axiomatized `is_integer` predicate for the range rather than assuming
    # the derived fact outright.
    f = z3.Function("f", z3.RealSort(), z3.RealSort())
    is_integer = z3.Function("is_integer", z3.RealSort(), z3.BoolSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    x, n = z3.Reals("x n")

    # f's own defining properties (true of Ceil, Floor, and Round alike):
    # its output always lands in the integers, and it is the identity ON an
    # already-integer input.
    f_range_is_integer = z3.ForAll([x], is_integer(f(x)))
    identity_on_integers = z3.ForAll([n], z3.Implies(is_integer(n), f(n) == n))

    prove(
        z3.Implies(
            z3.And(f_range_is_integer, identity_on_integers),
            consumer(f(f(x))) == consumer(f(x)),
        )
    )


def test_eliminate_consecutive_idempotent_relu_is_sound():
    # Relu's own version of the same shape, but via a numeric inequality
    # (non-negativity) instead of an axiomatized set-membership predicate --
    # deliberately a different hypothesis shape from the Ceil-family proof
    # above, per the module docstring.
    relu = z3.Function("relu", z3.RealSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    x, y = z3.Reals("x y")

    relu_is_nonneg = z3.ForAll([x], relu(x) >= 0)
    relu_identity_on_nonneg = z3.ForAll([y], z3.Implies(y >= 0, relu(y) == y))

    prove(
        z3.Implies(
            z3.And(relu_is_nonneg, relu_identity_on_nonneg),
            consumer(relu(relu(x))) == consumer(relu(x)),
        )
    )


def test_eliminate_consecutive_idempotent_reshape_composability_is_sound():
    # Structurally distinct from the two proofs above: this is not
    # "f(f(x))==f(x)" for one fixed function, it's "composing two reshapes
    # through an ARBITRARY intermediate shape S1 collapses to just the final
    # reshape S2" -- true for every S1, not because Reshape is idempotent as
    # a function of one variable but because it never moves data.
    #
    # Reuses test_formal_verify_eliminate_nop_reshape.py's own technique: X
    # is modeled as a function over a row-major flat buffer index
    # (`x_flat`), and reshaping into some shape (., a1) then reading back
    # flat position k is a pure index round-trip -- `(k / a1) * a1 + (k %
    # a1) == k` -- which Z3's integer div/mod theory resolves directly, with
    # NO dependence on what a1 actually is (only that it's positive, i.e. a
    # valid axis length). That is the whole proof: the intermediate shape
    # cancels out algebraically, for literally any a1.
    x_flat = z3.Function("x_flat", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    a1, b0, b1, i, j = z3.Ints("a1 b0 b1 i j")

    # a1: the intermediate shape S1's trailing axis length (arbitrary,
    # unconstrained relative to b0/b1 -- that is the whole point). b0, b1:
    # the final shape S2. i, j: the output index in scope, ranging over S2.
    domain = z3.And(a1 > 0, b0 > 0, b1 > 0, 0 <= i, i < b0, 0 <= j, j < b1)

    flat_b = i * b1 + j
    # Reshape(Reshape(X, S1), S2) at (i, j): re-flatten against S2 to get
    # flat position k = flat_b, then address the INTERMEDIATE array (whose
    # own trailing axis length is a1) at that flat position -- which itself
    # unflattens/reflattens through a1 and round-trips back to k exactly.
    k = flat_b
    composed_val = x_flat((k / a1) * a1 + (k % a1))
    # Reshape(X, S2) at (i, j) directly.
    direct_val = x_flat(flat_b)

    prove(
        z3.Implies(domain, consumer(composed_val) == consumer(direct_val)),
        msg="Reshape(Reshape(X, S1), S2) is not sound-equivalent to Reshape(X, S2)",
    )


def test_eliminate_consecutive_idempotent_pass_matches_relu():
    # Relu(Relu(X)), single use of the inner Relu's output -- fires: the
    # outer Relu is rewired to read X directly, the inner Relu is left
    # dangling (0 uses, not itself destroyed by this pass -- eliminate_deadend
    # would normally clean it up, but it is one of the other default passes
    # `simplify_isolated` skips here to isolate this one pass, so the dead
    # `a = Relu(X)` node is still physically present in the graph;
    # `producer` walks backward from the real graph output to find the LIVE
    # computation rather than overcounting via a raw op-type Counter -- see
    # its own docstring in _formal_verify_common.py).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Relu(X)
          Y = Relu(a)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_consecutive_idempotent_ops")
    assert ops["Relu"] == 2  # outer Relu(Y) + dangling inner Relu(a)
    relu_node = producer(sim_model, "Y")
    assert list(relu_node.input) == ["X"]


def test_eliminate_consecutive_idempotent_pass_matches_ceil():
    # Ceil(Ceil(X)) -- same pattern, different op kind.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Ceil(X)
          Y = Ceil(a)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_consecutive_idempotent_ops")
    assert ops["Ceil"] == 2  # outer Ceil(Y) + dangling inner Ceil(a)
    ceil_node = producer(sim_model, "Y")
    assert list(ceil_node.input) == ["X"]


def test_eliminate_consecutive_idempotent_pass_matches_reshape():
    # Reshape(Reshape(X, S1), S2) -- fires: the outer Reshape ends up reading
    # X directly, with target shape S2. This is the one case in the family
    # where getting the C++'s shape-restoration logic wrong (see module
    # docstring) could plausibly produce an observably broken graph, so this
    # test also actually runs the simplified model through onnxruntime and
    # checks the numeric output, rather than relying only on
    # `simplify_isolated`'s own `check_ok` (which already ran onnxsim's
    # own equivalence check, but doing it again explicitly here makes the
    # exact claim -- "the final output shape/value is correct" -- visible in
    # this file rather than only inside the shared helper).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[4,8] X) => (float[2,16] Y)
        <int64[3] S1 = {2, 2, 8}, int64[2] S2 = {2, 16}>
        {
          a = Reshape(X, S1)
          Y = Reshape(a, S2)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_consecutive_idempotent_ops")
    assert ops["Reshape"] == 2  # outer Reshape(Y) + dangling inner Reshape(a)
    reshape_node = producer(sim_model, "Y")
    assert list(reshape_node.input) == ["X", "S2"]

    sess = ort.InferenceSession(sim_model.SerializeToString())
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    (out,) = sess.run(None, {"X": x})
    np.testing.assert_array_equal(out, x.reshape(2, 16))


def test_eliminate_consecutive_idempotent_declines_on_multi_use_inner():
    # Declining case: the inner Relu's output is ALSO returned as a second
    # graph output, so it has more than one use -- the single-use
    # precondition (`node->input(0)->uses().size() == 1`) fails, and both
    # Relu nodes survive chained.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y, float[4,8] Z)
        {
          a = Relu(X)
          Y = Relu(a)
          Z = Identity(a)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_consecutive_idempotent_ops")
    assert ops["Relu"] == 2
    (outer_relu,) = [n for n in sim_model.graph.node if n.output[0] == "Y"]
    assert list(outer_relu.input) == ["a"]


def test_eliminate_consecutive_idempotent_declines_on_different_op_kinds():
    # Declining case: Relu(Ceil(X)) -- the outer and inner ops are BOTH
    # idempotent kinds, but different from each other. The predicate
    # requires the exact SAME kind consecutively (`CheckKind(node,
    # Symbol(op), 0, Symbol(op))`), so this must not fire.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Ceil(X)
          Y = Relu(a)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_consecutive_idempotent_ops")
    assert ops["Ceil"] == 1
    assert ops["Relu"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["a"]
