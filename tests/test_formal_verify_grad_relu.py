"""Formal check for ``graph_grad._grad_relu``.

``Relu`` is piecewise-affine: locally the identity where ``x > 0``, locally
the zero map where ``x < 0``, with no single correct derivative at the kink
``x == 0`` -- the rule picks the subgradient ``0`` there (a documented
convention, matching the straight-through masks ``adaround.py`` already
builds), not something a proof can call "the" right answer. So what's
provable is the directional derivative on each open branch, not at the
kink itself.
"""

import pytest
from _formal_verify_common import prove, z3
from test_graph_grad import _check, _model

pytest.importorskip("onnxruntime")


def test_grad_relu_matches_branch_slope_away_from_kink():
    x, g = z3.Reals("x g")
    dx_rule = z3.If(x > 0, g, z3.RealVal(0))  # _grad_relu: g * (x > 0)
    prove(z3.Implies(x > 0, dx_rule == g), "GradRelu wrong on the positive branch")
    prove(z3.Implies(x < 0, dx_rule == 0), "GradRelu wrong on the negative branch")


def test_grad_relu_with_flipped_mask_is_unsound():
    # Negative control: a mask that fired on the wrong branch (x < 0 instead
    # of x > 0 -- e.g. a flipped comparison) does not match Relu's actual
    # slope on either branch.
    x, g = z3.Reals("x2 g2")
    wrong_rule = z3.If(x < 0, g, z3.RealVal(0))
    solver = z3.Solver()
    solver.add(z3.Not(z3.Implies(x > 0, wrong_rule == g)))
    assert solver.check() == z3.sat, (
        "flipped-mask rule still matches the positive branch -- negative control is vacuous"
    )


def test_grad_relu_matches_finite_differences():
    _check(
        _model(
            """
            g (float[3,4] A) => (float[3,4] Y) {
              Y = Relu(A)
            }
            """
        )
    )
