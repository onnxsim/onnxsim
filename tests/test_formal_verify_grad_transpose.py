"""Formal check for ``graph_grad._grad_transpose``.

``Transpose`` is a pure permutation of axes, so its VJP has to route each
element of the incoming gradient ``g`` back through the *inverse* of the
permutation the forward pass applied. The rule (``graph_grad.py:1200-1209``)
computes that inverse with a small loop:

.. code-block:: python

    inverse = [0] * rank
    for position, axis in enumerate(perm):
        inverse[axis] = position

and emits ``Transpose(g, perm=inverse)``. Getting this backwards -- reusing
``perm`` itself, say -- would be silently wrong for any permutation that
isn't its own inverse (a 3-cycle, unlike a plain axis swap).

The claim proved below is exactly that construction's correctness as a
*function*, not for one example: for every valid permutation of a 3-axis
tensor (representative of any rank -- the algorithm never looks past
``enumerate(perm)``), the array it builds satisfies ``perm[inverse[m]] == m``
for every axis ``m`` -- i.e. it really is a right-inverse of ``perm`` under
composition, which is precisely the property that makes
``Transpose(Transpose(x, perm), perm=inverse)`` reconstruct ``x``. Finite
domain (3 distinct values in ``{0, 1, 2}``), so Z3 decides it exhaustively
rather than sampling permutations.

The differential check ties this to the real compiled rule, via
``test_graph_grad.py``'s finite-difference machinery, using a 3-cycle
specifically -- the case a self-inverse assumption would get wrong.
"""

import pytest
from _formal_verify_common import prove, z3
from test_graph_grad import _check, _model

pytest.importorskip("onnxruntime")


def _select(k, *vals):
    """``vals[k]`` for a symbolic ``k`` -- a lookup table as nested ``If``s."""
    expr = vals[-1]
    for i in range(len(vals) - 2, -1, -1):
        expr = z3.If(k == i, vals[i], expr)
    return expr


def test_grad_transpose_inverse_construction_is_a_right_inverse():
    p0, p1, p2 = z3.Ints("p0 p1 p2")
    perm_is_valid = z3.And(
        z3.Or(p0 == 0, p0 == 1, p0 == 2),
        z3.Or(p1 == 0, p1 == 1, p1 == 2),
        z3.Or(p2 == 0, p2 == 1, p2 == 2),
        z3.Distinct(p0, p1, p2),
    )

    # graph_grad._grad_transpose's own algorithm: inverse[m] is the position
    # k holding value m in perm, i.e. the k with perm[k] == m.
    def inverse_of(m):
        return z3.If(p0 == m, 0, z3.If(p1 == m, 1, 2))

    inv0, inv1, inv2 = inverse_of(0), inverse_of(1), inverse_of(2)

    # perm[inverse[m]] == m for every m -- inverse really is perm's
    # (right-)inverse under composition.
    claim = z3.And(
        _select(inv0, p0, p1, p2) == 0,
        _select(inv1, p0, p1, p2) == 1,
        _select(inv2, p0, p1, p2) == 2,
    )
    prove(
        z3.Implies(perm_is_valid, claim),
        "_grad_transpose's inverse-construction algorithm is not a true "
        "functional inverse of perm",
    )


def test_grad_transpose_reusing_perm_as_its_own_inverse_is_unsound():
    # Negative control: using perm itself instead of its inverse (the bug
    # this rule exists to avoid) does not satisfy the same composition
    # identity for every permutation of 3 elements -- confirms the proof
    # above is testing something real, not an identity that holds anyway.
    p0, p1, p2 = z3.Ints("p0 p1 p2")
    perm_is_valid = z3.And(
        z3.Or(p0 == 0, p0 == 1, p0 == 2),
        z3.Or(p1 == 0, p1 == 1, p1 == 2),
        z3.Or(p2 == 0, p2 == 1, p2 == 2),
        z3.Distinct(p0, p1, p2),
    )
    wrong_claim = z3.And(
        _select(p0, p0, p1, p2) == 0,
        _select(p1, p0, p1, p2) == 1,
        _select(p2, p0, p1, p2) == 2,
    )
    solver = z3.Solver()
    solver.add(perm_is_valid)
    solver.add(z3.Not(wrong_claim))
    assert solver.check() == z3.sat, (
        "perm-as-its-own-inverse satisfies the composition identity for "
        "every permutation -- negative control is vacuous"
    )


def test_grad_transpose_matches_finite_differences_for_a_3cycle():
    # perm = [1, 2, 0] is a 3-cycle (its inverse, [2, 0, 1], is not itself)
    # -- exactly the case the proof above is about, exercised against the
    # real compiled rule.
    _check(
        _model(
            """
            g (float[2,3,4] A) => (float[3,4,2] Y) {
              Y = Transpose <perm = [1, 2, 0]> (A)
            }
            """
        )
    )
