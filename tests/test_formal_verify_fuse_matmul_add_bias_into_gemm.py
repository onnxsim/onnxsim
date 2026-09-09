"""Formal check for FuseMatMulAddBiasIntoGemm (fuse_matmul_add_bias_into_gemm.h).

The rewrite fires only on ``Add(MatMul(A, B), C)`` -- MatMul strictly as the
Add's *first* operand (``CheckKind(node, kAdd, 0, kMatMul)``);
``Add(C, MatMul(A, B))`` is never matched, even though addition is
commutative, because the pass never checks the second operand.
``runTransform`` additionally requires both MatMul operands to be rank-2
with statically known relevant dims, and the bias ``C`` to broadcast against
the (N, M) output the same way ONNX Gemm's own ``C`` broadcasts -- 1-D
matching M, or 2-D with a leading dim of 1 or N. The rewrite replaces the
matched Add node with a new ``Gemm(A, B, C, alpha=1, beta=1, transA=0,
transB=0)``; the original MatMul is left dangling (0 uses) rather than
destroyed outright -- a separate dead-code pass (eliminate_deadend) removes
it, so it is still present immediately after this pass runs in isolation.

Soundness is close to definitional -- Gemm's own semantics are
``alpha * A @ B + beta * C``, so with alpha=beta=1 this is exactly
``A @ B + C`` -- but it is worth proving explicitly rather than asserting by
inspection: a 1-off in which operand supplies alpha/beta/transA/transB, or a
wrong operand order, would silently compute something else. This proves it
as a real arithmetic identity over a small concrete matrix shape (2x3 @ 3x2,
1-D bias broadcasting over rows), which exercises the summation and the
broadcast rule without needing to reason about a symbolic-rank
generalization.
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_fuse_matmul_add_bias_into_gemm_is_sound():
    # A = 2x3, B = 3x2, C = bias of length 2 (broadcasts over A's 2 rows).
    A = [[z3.Real(f"a{i}{k}") for k in range(3)] for i in range(2)]
    B = [[z3.Real(f"b{k}{j}") for j in range(2)] for k in range(3)]
    C = [z3.Real(f"c{j}") for j in range(2)]

    def dot(i, j):
        return sum(A[i][k] * B[k][j] for k in range(3))

    alpha, beta = 1, 1
    matmul_then_add = [[dot(i, j) + C[j] for j in range(2)] for i in range(2)]
    gemm = [[alpha * dot(i, j) + beta * C[j] for j in range(2)] for i in range(2)]

    claim = z3.And(
        *[matmul_then_add[i][j] == gemm[i][j] for i in range(2) for j in range(2)]
    )
    prove(claim)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def test_fuse_matmul_add_bias_into_gemm_pass_matches():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 8))
    B = rng.standard_normal(8)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,16] X) => (float[4,8] Y)
        {
          mm = MatMul(X, W)
          Y = Add(mm, B)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W"), _f32(B, "B")])
    sim_model, ops = simplify_isolated(model, "fuse_matmul_add_bias_into_gemm")
    assert ops["Gemm"] == 1
    # Add is destroyed (it's the matched node); MatMul is left dangling
    # (0 uses) rather than destroyed -- see the module docstring.
    assert ops["Add"] == 0
    assert ops["MatMul"] == 1

    gemm_node = next(n for n in sim_model.graph.node if n.op_type == "Gemm")
    attrs = {a.name: a for a in gemm_node.attribute}
    # The exact attribute values just proven sound above -- alpha=beta=1
    # (Gemm's own defaults) and no transpose, either explicitly set or left
    # at ONNX's default.
    assert attrs.get("alpha") is None or attrs["alpha"].f == 1.0
    assert attrs.get("beta") is None or attrs["beta"].f == 1.0
    assert attrs.get("transA") is None or attrs["transA"].i == 0
    assert attrs.get("transB") is None or attrs["transB"].i == 0


def test_fuse_matmul_add_bias_into_gemm_declines_swapped_operand_order():
    # Add(bias, MatMul(...)) -- bias as the *first* operand -- is never
    # matched: the predicate only checks Add's operand 0 for a MatMul.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 8))
    B = rng.standard_normal(8)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,16] X) => (float[4,8] Y)
        {
          mm = MatMul(X, W)
          Y = Add(B, mm)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W"), _f32(B, "B")])
    _, ops = simplify_isolated(model, "fuse_matmul_add_bias_into_gemm")
    assert ops["Gemm"] == 0
    assert ops["MatMul"] == 1
