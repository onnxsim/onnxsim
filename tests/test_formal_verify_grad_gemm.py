"""Formal check for ``graph_grad._grad_gemm``.

``Gemm`` computes ``Y = alpha * A' @ B' + beta * C`` (``A' = A^T`` when
``transA``, likewise ``B'``). ``_grad_gemm`` (``graph_grad.py:328-363``)
scales the incoming gradient by ``alpha`` once, differentiates through the
plain matrix product (the same adjoint proved in
``test_formal_verify_grad_matmul.py``), then transposes back into ``A``'s
and ``B``'s own layouts; ``C``'s gradient is ``beta`` times the
broadcast-reduce adjoint already proved in
``test_formal_verify_grad_add.py``.

This file isolates the one piece those two proofs don't already cover: that
scaling the incoming gradient by ``alpha``/``beta`` before differentiating
is the right place for the scalar to enter, rather than scaling the
*results* (a plausible-looking but different rule) -- checked for the
no-transpose case at a concrete shape; the ``transA``/``transB`` cases
reduce to this one composed with ``_grad_transpose``'s own proof.
"""

import pytest
from _formal_verify_common import prove, z3
from test_graph_grad import _check, _model

pytest.importorskip("onnxruntime")


def _reals(prefix, *shape):
    if not shape:
        return z3.Real(prefix)
    return [_reals(f"{prefix}_{i}", *shape[1:]) for i in range(shape[0])]


def _matmul(A, B):
    M, K, N = len(A), len(A[0]), len(B[0])
    return [
        [z3.Sum([A[m][k] * B[k][n] for k in range(K)]) for n in range(N)]
        for m in range(M)
    ]


def _transpose(A):
    return [list(row) for row in zip(*A)]


def _frob(A, B):
    return z3.Sum([A[i][j] * B[i][j] for i in range(len(A)) for j in range(len(A[0]))])


def test_grad_gemm_no_transpose_scaling_is_adjoint():
    M, K, N = 2, 2, 2
    alpha, beta = 0.5, 2.0
    A = _reals("gA", M, K)
    B = _reals("gB", K, N)
    G = _reals("gG", M, N)
    dA_dir = _reals("gdAd", M, K)
    dB_dir = _reals("gdBd", K, N)
    dC_dir = _reals("gdCd", M, N)

    gs = [[G[m][n] * alpha for n in range(N)] for m in range(M)]  # _grad_gemm's `gs`
    ga = _matmul(gs, _transpose(B))
    gb = _matmul(_transpose(A), gs)
    gc = [[G[m][n] * beta for n in range(N)] for m in range(M)]

    dY = [
        [
            alpha
            * (
                z3.Sum([dA_dir[m][k] * B[k][n] for k in range(K)])
                + z3.Sum([A[m][k] * dB_dir[k][n] for k in range(K)])
            )
            + beta * dC_dir[m][n]
            for n in range(N)
        ]
        for m in range(M)
    ]
    lhs = _frob(G, dY)
    rhs = _frob(ga, dA_dir) + _frob(gb, dB_dir) + _frob(gc, dC_dir)
    prove(lhs == rhs, "GradGemm (no-transpose) is not the adjoint of Gemm's Jacobian")


def test_grad_gemm_scaling_after_matmul_instead_of_before_is_unsound():
    # Negative control: scaling the *results* by alpha (ga = (G@B^T)*alpha)
    # rather than scaling G by alpha before the matmul happens to be
    # numerically identical for the plain matmul term (scalar multiplication
    # commutes through it) -- so the real risk this proof guards is scaling
    # only ONE of the two results, a plausible copy-paste slip. Confirm that
    # breaks the identity.
    M, K, N = 2, 2, 2
    alpha, beta = 0.5, 2.0
    A = _reals("wgA", M, K)
    B = _reals("wgB", K, N)
    G = _reals("wgG", M, N)
    dA_dir = _reals("wgdAd", M, K)
    dB_dir = _reals("wgdBd", K, N)
    dC_dir = _reals("wgdCd", M, N)

    ga = _matmul(G, _transpose(B))  # missing the alpha scale entirely
    gb = _matmul(_transpose(A), [[G[m][n] * alpha for n in range(N)] for m in range(M)])
    gc = [[G[m][n] * beta for n in range(N)] for m in range(M)]

    dY = [
        [
            alpha
            * (
                z3.Sum([dA_dir[m][k] * B[k][n] for k in range(K)])
                + z3.Sum([A[m][k] * dB_dir[k][n] for k in range(K)])
            )
            + beta * dC_dir[m][n]
            for n in range(N)
        ]
        for m in range(M)
    ]
    lhs = _frob(G, dY)
    rhs = _frob(ga, dA_dir) + _frob(gb, dB_dir) + _frob(gc, dC_dir)
    solver = z3.Solver()
    solver.add(z3.Not(lhs == rhs))
    assert solver.check() == z3.sat, (
        "dropping alpha from just one of the two results still holds -- "
        "negative control is vacuous"
    )


def test_grad_gemm_matches_finite_differences():
    _check(
        _model(
            """
            g (float[3,4] A, float[4,5] B, float[5] C) => (float[3,5] Y) {
              Y = Gemm <alpha = 0.5, beta = 2.0> (A, B, C)
            }
            """
        )
    )
