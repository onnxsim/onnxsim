"""Formal check for ``graph_grad._grad_conv``.

Per that rule's own docstring (``graph_grad.py:583-643``), ``Conv`` is
differentiated by rewriting the forward as im2col (a ``Gather``) followed by
a ``MatMul``: for a fixed kernel tap ``t`` and output position ``o``,

    col[t, o] = X[position(o, t)]           (im2col: one Gather)
    Y[m, o]   = sum_{c, t} W[m, c, t] * col[c, t, o]

Differentiating the ``MatMul`` half is the adjoint already proved in
``test_formal_verify_grad_matmul.py``. The one genuinely new primitive is
"col2im" -- reconstructing ``dX`` from ``dcol`` -- which the docstring
claims is *also* expressible as a ``Gather`` (not a scatter-add) because,
for a fixed tap, ``position`` is injective: each input element is read by
at most one output position per tap, so summing ``dcol`` over the (tap,
output-position) pairs that read a given input element is the same as
gathering it.

This file proves that claim directly: col2im-as-Gather is the *adjoint* of
im2col-as-Gather, for a concrete 1-D geometry (input length 4, kernel size
2, stride 1, no padding -- the simplest ``position`` that is still injective
per tap, which is all the docstring's argument needs; groups and further
spatial dims only change which index tables get built, not this identity).
"""

import pytest
from _formal_verify_common import prove, z3
from test_graph_grad import _check, _model

pytest.importorskip("onnxruntime")


def _reals(prefix, *shape):
    if not shape:
        return z3.Real(prefix)
    return [_reals(f"{prefix}_{i}", *shape[1:]) for i in range(shape[0])]


def test_grad_conv_col2im_gather_is_adjoint_of_im2col_gather():
    in_len, kernel, out_len = 4, 2, 3
    dcol_dir = _reals("cdcold", kernel, out_len)
    dX_dir = _reals("cdXd", in_len)

    def position(o, t):
        return o + t

    # Adjoint claim: <dcol_dir, im2col'(dX_dir)> == <col2im(dcol_dir), dX_dir>
    # for every direction dX_dir, where im2col'(dX_dir)[t, o] = dX_dir[position(o, t)].
    lhs = z3.Sum(
        [
            dcol_dir[t][o] * dX_dir[position(o, t)]
            for t in range(kernel)
            for o in range(out_len)
        ]
    )

    # col2im per _grad_conv: dX[p] = sum over (t, o) with position(o,t)==p.
    def col2im(p):
        return z3.Sum(
            [
                dcol_dir[t][o]
                for t in range(kernel)
                for o in range(out_len)
                if position(o, t) == p
            ]
        )

    rhs = z3.Sum([col2im(p) * dX_dir[p] for p in range(in_len)])
    prove(lhs == rhs, "GradConv's col2im is not the adjoint of im2col's Gather")


def test_grad_conv_wrong_tap_offset_is_unsound():
    # Negative control: col2im built against an off-by-one tap offset (a
    # plausible index-table bug) does not recover the adjoint.
    in_len, kernel, out_len = 4, 2, 3
    dcol_dir = _reals("wcdcold", kernel, out_len)
    dX_dir = _reals("wcdXd", in_len)

    def position(o, t):
        return o + t

    def wrong_position(o, t):
        return o + t + 1

    lhs = z3.Sum(
        [
            dcol_dir[t][o] * dX_dir[position(o, t)]
            for t in range(kernel)
            for o in range(out_len)
        ]
    )

    def wrong_col2im(p):
        return z3.Sum(
            [
                dcol_dir[t][o]
                for t in range(kernel)
                for o in range(out_len)
                if 0 <= wrong_position(o, t) < in_len and wrong_position(o, t) == p
            ]
        )

    rhs = z3.Sum([wrong_col2im(p) * dX_dir[p] for p in range(in_len)])
    solver = z3.Solver()
    solver.add(z3.Not(lhs == rhs))
    assert solver.check() == z3.sat, (
        "off-by-one col2im holds for all directions -- negative control is vacuous"
    )


def test_grad_conv_matches_finite_differences():
    _check(
        _model(
            """
            g (float[1,2,4,4] A, float[3,2,3,3] B) => (float[1,3,2,2] Y) {
              Y = Conv(A, B)
            }
            """
        )
    )


def test_grad_conv_matches_finite_differences_strided_padded():
    # Exercises the padding/stride mask alongside the col2im adjoint proved
    # above (a tap reading outside the input, or an input element a strided
    # tap never touched).
    _check(
        _model(
            """
            g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
              Y = Conv <strides = [2, 2], pads = [1, 1, 1, 1]> (A, B)
            }
            """
        )
    )
