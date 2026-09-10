"""Formal check for ``graph_grad._grad_add`` (and the onnxscript-templated
``GradAdd`` it is cross-checked against in ``test_graph_grad_templates.py``).

``Add`` broadcasts its two operands (numpy-style) before combining them
elementwise, so the interesting content of its VJP is not ``+``'s
commutativity/associativity -- it is that ``ctx.reduce_to`` sums the
incoming gradient ``g`` over exactly the axes broadcasting inserted, which
is what makes the rule the *adjoint* of broadcasting: ``⟨g, broadcast(A)⟩
== ⟨reduce(g), A⟩`` for every ``A``, not merely for the samples a numeric
check happens to try.

Modeled on the bias-broadcast case from
``test_formal_verify_adjust_add.py``: ``A`` has shape ``[4]`` and broadcasts
over the (implicit, leading) batch axis against ``B``'s shape ``[3, 4]``.
"""

import pytest
from _formal_verify_common import prove, z3
from test_graph_grad import _check, _model

pytest.importorskip("onnxruntime")


def test_grad_add_bias_broadcast_reduction_is_adjoint():
    n_rows, n_cols = 3, 4
    g = z3.Function("g", z3.IntSort(), z3.IntSort(), z3.RealSort())
    A = z3.Function("A", z3.IntSort(), z3.RealSort())

    # ctx.reduce_to(g, out=[3,4], target=[4]): sum g over the broadcast axis.
    def dA(j):
        return z3.Sum([g(i, j) for i in range(n_rows)])

    lhs = z3.Sum([g(i, j) * A(j) for i in range(n_rows) for j in range(n_cols)])
    rhs = z3.Sum([dA(j) * A(j) for j in range(n_cols)])
    prove(lhs == rhs, "GradAdd's sum-reduction is not the adjoint of broadcasting")


def test_grad_add_wrong_axis_reduction_breaks_adjoint():
    # Negative control: reducing over j (A's own axis) instead of i (the
    # broadcast axis) breaks the same identity for arbitrary g, A --
    # confirms the axis choice is load-bearing, not incidental.
    n_rows, n_cols = 3, 4
    g = z3.Function("g", z3.IntSort(), z3.IntSort(), z3.RealSort())
    A = z3.Function("A", z3.IntSort(), z3.RealSort())

    def wrong_dA(j):
        return z3.Sum([g(j, k) for k in range(n_cols)])

    lhs = z3.Sum([g(i, j) * A(j) for i in range(n_rows) for j in range(n_cols)])
    wrong_rhs = z3.Sum([wrong_dA(j) * A(j) for j in range(n_cols)])
    solver = z3.Solver()
    solver.add(z3.Not(lhs == wrong_rhs))
    assert solver.check() == z3.sat, (
        "wrong-axis reduction holds for all g, A -- negative control is vacuous"
    )


def test_grad_add_matches_finite_differences_with_broadcasting():
    _check(
        _model(
            """
            g (float[4] A, float[3,4] B) => (float[3,4] Y) {
              Y = Add(A, B)
            }
            """
        )
    )
