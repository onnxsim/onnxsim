"""Formal check for RewriteWhere (rewrite_where.h):
``Where(Not(b), x, y) -> Where(b, y, x)``.

``patternMatchPredicate`` matches any ``Where`` node whose first input (the
condition) is produced by a ``Not`` node -- nothing else is checked there.
``runTransform`` only fires when that ``Not``'s output has exactly one use
(consumed solely by this ``Where``): it rewires the ``Where``'s condition
input directly to the ``Not``'s own input ``b`` (skipping the ``Not``
entirely), swaps the ``Where``'s second and third inputs (``x``/``y``
become ``y``/``x``), and destroys the now-dead ``Not`` node. If the ``Not``
output has more than one use, the pass declines outright (returns ``false``,
leaving both nodes untouched) -- rewiring would otherwise silently change
what those other consumers see.

Soundness is a pure boolean identity, not a tensor-shape argument: ``Where``
is a per-element ternary selector, so modeling ``Where(c, x, y)`` directly as
Z3's native ``If(c, x, y)`` -- which is definitionally exactly what ONNX's
``Where`` computes elementwise -- makes the claim ``If(Not(b), x, y) ==
If(b, y, x)`` a direct case-split proof: "``x`` when ``b`` is false, else
``y``" is exactly "``y`` when ``b`` is true, else ``x``". Composing this
pointwise identity with an arbitrary uninterpreted ``consumer`` function and
an arbitrary tensor index (as in test_formal_verify_eliminate_identity.py)
states the full substitution-safety claim: the rewritten graph node (reading
``b`` directly, ``x``/``y`` swapped) produces, at every element, the exact
same value the original node (reading ``Not(b)``, ``x``/``y`` in their
original order) did -- for *any* downstream consumer, not just the
one-hop-Where check below.
"""

from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser


def test_rewrite_where_is_sound():
    # b/x/y stand for one arbitrary element of the (elementwise) B/X/Y
    # tensors -- Where broadcasts this identity across every index, so
    # proving it for one symbolic element proves it for the whole tensor.
    b = z3.Bool("b")
    x = z3.Real("x")
    y = z3.Real("y")
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    # Pointwise Where identity: Where(Not(b), x, y) == Where(b, y, x).
    where_identity = z3.If(z3.Not(b), x, y) == z3.If(b, y, x)
    prove(where_identity)

    # Full substitution-safety claim: composed with an arbitrary downstream
    # consumer, the rewritten node's output matches the original's -- for
    # every consumer, not just the one-hop-Where check in
    # test_rewrite_where_pass_matches below.
    original = z3.If(z3.Not(b), x, y)
    rewritten = z3.If(b, y, x)
    prove(z3.Implies(where_identity, consumer(original) == consumer(rewritten)))


def test_rewrite_where_negative_control_no_swap_is_unsound():
    # Sanity check that the proof is genuine, not vacuous: dropping the
    # x/y swap (i.e. Where(Not(b), x, y) == Where(b, x, y), unswapped)
    # must NOT hold for every b, x, y -- confirm Z3 finds a real
    # counterexample rather than reporting the (wrong) claim valid.
    b = z3.Bool("b")
    x = z3.Real("x")
    y = z3.Real("y")
    solver = z3.Solver()
    unswapped_claim = z3.If(z3.Not(b), x, y) == z3.If(b, x, y)
    solver.add(z3.Not(unswapped_claim))
    assert solver.check() == z3.sat


def test_rewrite_where_negative_control_missing_not_is_unsound():
    # Likewise, dropping the Not entirely (Where(b, x, y) == Where(b, y, x))
    # must not hold for every b, x, y.
    b = z3.Bool("b")
    x = z3.Real("x")
    y = z3.Real("y")
    solver = z3.Solver()
    missing_not_claim = z3.If(b, x, y) == z3.If(b, y, x)
    solver.add(z3.Not(missing_not_claim))
    assert solver.check() == z3.sat


def test_rewrite_where_pass_matches():
    # Differential check: the real compiled pass, run alone, rewrites
    # Where(Not(B), X, Y) to a Where reading B directly (Not is gone) with
    # X and Y swapped -- verified by inspecting the surviving node's actual
    # inputs, in order, not merely counting op types.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (bool[4] B, float[4] X, float[4] Y) => (float[4] Z)
        {
            nb = Not(B)
            Z = Where(nb, X, Y)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "rewrite_where")
    assert ops["Not"] == 0
    where_node = producer(sim_model, "Z")
    assert where_node.op_type == "Where"
    assert list(where_node.input) == ["B", "Y", "X"]


def test_rewrite_where_declines_when_not_has_other_uses():
    # Not's output is consumed by both the Where and a second graph
    # output -- the single-use precondition fails, so the pass, run
    # alone, must leave Not and Where untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (bool[4] B, float[4] X, float[4] Y) => (float[4] Z, bool[4] NB)
        {
            nb = Not(B)
            Z = Where(nb, X, Y)
            NB = Identity(nb)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "rewrite_where")
    assert ops["Not"] == 1
    where_node = producer(sim_model, "Z")
    assert where_node.op_type == "Where"
    assert list(where_node.input) == ["nb", "X", "Y"]


def test_rewrite_where_declines_when_condition_is_not_a_not():
    # Where's condition is a plain boolean input, not the output of a Not
    # node -- patternMatchPredicate declines trivially.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (bool[4] B, float[4] X, float[4] Y) => (float[4] Z)
        {
            Z = Where(B, X, Y)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "rewrite_where")
    assert ops["Where"] == 1
    where_node = producer(sim_model, "Z")
    assert list(where_node.input) == ["B", "X", "Y"]
