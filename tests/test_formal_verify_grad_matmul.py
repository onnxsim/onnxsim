"""Formal check for ``graph_grad._grad_matmul``.

``_grad_matmul`` (``graph_grad.py:305-325``) computes ``dA = G @ B^T`` and
``dB = A^T @ G`` (each then reduced back over whatever batch axes
broadcasting replicated -- the same broadcast-reduce adjoint already proved
in ``test_formal_verify_grad_add.py``, so this file isolates the core 2-D
matrix-product adjoint instead).

The claim is the defining adjoint property of ``MatMul``'s Jacobian: for
``Y = A @ B``, the VJP is correct iff, for *every* direction
``(dA_dir, dB_dir)``,

    ⟨G, dA_dir @ B + A @ dB_dir⟩_F  ==  ⟨dA, dA_dir⟩_F + ⟨dB, dB_dir⟩_F

(Frobenius inner product), which follows from the standard trace identity
``⟨G, X @ B⟩ = ⟨G @ B^T, X⟩`` and ``⟨G, A @ X⟩ = ⟨A^T @ G, X⟩``. Proved for a
concrete small shape (``M=K=N=2``) -- ``ctx.shape`` gives the rule a
concrete static shape too, never a symbolic rank, so this is exactly the
scale the real rule reasons about, just fully expanded into scalar
arithmetic for Z3.
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


def test_grad_matmul_is_adjoint():
    M, K, N = 2, 2, 2
    A = _reals("A", M, K)
    B = _reals("B", K, N)
    G = _reals("G", M, N)
    dA_dir = _reals("dAd", M, K)
    dB_dir = _reals("dBd", K, N)

    dA = _matmul(G, _transpose(B))  # _grad_matmul's rule
    dB = _matmul(_transpose(A), G)

    dY = [
        [
            z3.Sum([dA_dir[m][k] * B[k][n] for k in range(K)])
            + z3.Sum([A[m][k] * dB_dir[k][n] for k in range(K)])
            for n in range(N)
        ]
        for m in range(M)
    ]
    lhs = _frob(G, dY)
    rhs = _frob(dA, dA_dir) + _frob(dB, dB_dir)
    prove(lhs == rhs, "GradMatMul is not the adjoint of MatMul's Jacobian")


def test_grad_matmul_without_transposing_is_unsound():
    # Negative control: dA = G @ B (forgetting to transpose B) does not
    # satisfy the adjoint identity for arbitrary A, B, G.
    M, K, N = 2, 2, 2
    A = _reals("nA", M, K)
    B = _reals("nB", K, N)
    G = _reals("nG", M, N)
    dA_dir = _reals("ndAd", M, K)
    dB_dir = _reals("ndBd", K, N)

    wrong_dA = _matmul(G, B)  # missing the transpose
    dB = _matmul(_transpose(A), G)
    dY = [
        [
            z3.Sum([dA_dir[m][k] * B[k][n] for k in range(K)])
            + z3.Sum([A[m][k] * dB_dir[k][n] for k in range(K)])
            for n in range(N)
        ]
        for m in range(M)
    ]
    lhs = _frob(G, dY)
    rhs = _frob(wrong_dA, dA_dir) + _frob(dB, dB_dir)
    solver = z3.Solver()
    solver.add(z3.Not(lhs == rhs))
    assert solver.check() == z3.sat, (
        "un-transposed rule holds for all A, B, G -- negative control is vacuous"
    )


def test_grad_matmul_matches_finite_differences():
    _check(
        _model(
            """
            g (float[3,4] A, float[4,5] B) => (float[3,5] Y) {
              Y = MatMul(A, B)
            }
            """
        )
    )


def test_grad_matmul_matches_finite_differences_with_broadcast_batch():
    # Exercises the batch-broadcast reduction alongside the core matmul
    # adjoint proved above (A's batch dim broadcasts against B's).
    _check(
        _model(
            """
            g (float[1,3,4] A, float[2,4,5] B) => (float[2,3,5] Y) {
              Y = MatMul(A, B)
            }
            """
        )
    )
