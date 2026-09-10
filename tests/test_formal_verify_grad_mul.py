"""Formal check for ``graph_grad._grad_mul``.

``Mul`` is bilinear, so unlike the transcendental rules elsewhere in
``graph_grad.py`` (``Sigmoid``, ``Sqrt``, ...), its VJP is *exact* algebra --
the product rule ``d(a*b) = da*b + a*db`` -- not a linearization of
something Z3 can't reason about. That makes it provable directly over reals
with no shape modeling at all (broadcasting undone the same way as
``_grad_add``, already proved in ``test_formal_verify_grad_add.py``): the
elementwise core is what's checked here.

The claim is the defining adjoint property of a VJP: for the rule's
``da = g*b``, ``db = g*a`` to be correct, they must satisfy
``g * (va*b + a*vb) == da*va + db*vb`` for *every* direction ``(va, vb)``,
not just the one direction a numeric check happens to perturb along.
"""

import pytest
from _formal_verify_common import prove, z3
from test_graph_grad import _check, _model

pytest.importorskip("onnxruntime")


def test_grad_mul_is_product_rule():
    a, b, g = z3.Reals("a b g")
    da_rule, db_rule = g * b, g * a  # _grad_mul, before ctx.reduce_to
    va, vb = z3.Reals("va vb")
    prove(
        z3.ForAll(
            [va, vb],
            g * (va * b + a * vb) == da_rule * va + db_rule * vb,
        ),
        "GradMul is not the adjoint of Mul's bilinear Jacobian",
    )


def test_grad_mul_using_only_one_factor_is_unsound():
    # Negative control: a rule that used `g` alone for both da and db
    # (dropping the other operand entirely -- the kind of copy-paste bug
    # this proof would catch) does not satisfy the same identity.
    a, b, g = z3.Reals("a2 b2 g2")
    wrong_da, wrong_db = g, g
    va, vb = z3.Reals("va2 vb2")
    solver = z3.Solver()
    solver.add(
        z3.Not(
            z3.ForAll(
                [va, vb],
                g * (va * b + a * vb) == wrong_da * va + wrong_db * vb,
            )
        )
    )
    assert solver.check() == z3.sat, (
        "dropping the other operand still satisfies the adjoint identity -- "
        "negative control is vacuous"
    )


def test_grad_mul_matches_finite_differences_with_broadcasting():
    _check(
        _model(
            """
            g (float[4] A, float[3,4] B) => (float[3,4] Y) {
              Y = Mul(A, B)
            }
            """
        )
    )
