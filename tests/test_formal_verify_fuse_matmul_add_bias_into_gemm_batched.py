"""Formal check for FuseMatMulAddBiasIntoGemmBatched (opt-in; onnxsim's own
``onnxsim/passes/fuse_matmul_add_bias_into_gemm_batched.h``), the batched
(rank >= 3) sibling of the default ``fuse_matmul_add_bias_into_gemm``
already proved in test_formal_verify_fuse_matmul_add_bias_into_gemm.py.

It matches ``Add(MatMul(X, W), bias)`` **in either operand order**
(``Add(bias, MatMul(X, W))`` too) -- unlike the non-batched sibling, which
only matches MatMul as Add's first operand. X must be rank >= 3 with a
static trailing (contraction) dim; the leading "batch" dims may be dynamic.
W must be a constant rank-2 ``[K, N]``; bias must be exactly 1-D, ``[N]`` or
``[1]``. The rewrite reshapes X to 2-D (``X2 = Reshape(X, [-1, K])``), runs
a single ``Gemm(X2, W, bias, alpha=1, beta=1, transA=0, transB=0)``, and
reshapes the result back to the original leading dims plus ``N``.

Soundness combines two facts:

1. Row-major ``Reshape`` never moves data, it only relabels a flat buffer's
   index -- modeled here with an uninterpreted 1-D buffer function so both
   X's ``[B, M, K]`` indexing and X2's ``[B*M, K]`` indexing are shown to
   read the *identical* buffer element, for literally all integer indices
   (this holds unconditionally, with no side condition needed -- it is pure
   index arithmetic, the same flattening identity a C-order reshape always
   satisfies).
2. Gemm's row-wise formula is exactly MatMul-then-Add's per-row formula --
   the same arithmetic identity already proved in
   test_formal_verify_fuse_matmul_add_bias_into_gemm.py -- applied to the
   flattened row ``b*M + m`` instead of a single un-batched row.
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

_K = 2  # concrete contraction dim -- enough to exercise the dot-product sum
_N = 2  # concrete output-channel dim


def test_fuse_matmul_add_bias_into_gemm_batched_is_sound():
    buf = z3.Function("buf", z3.IntSort(), z3.RealSort())  # X's flat row-major buffer
    W = [[z3.Real(f"w{k}{n}") for n in range(_N)] for k in range(_K)]
    bias = [z3.Real(f"bias{n}") for n in range(_N)]
    b, m, M = z3.Ints("b m M")

    def x_at(bb, mm, k):
        # X[b, m, k], read from X's own row-major flat buffer: shape [*, M, K].
        return buf(bb * M * _K + mm * _K + k)

    def x2_at(r, k):
        # X2 = Reshape(X, [-1, K])'s row r -- the *same* buffer, no data moved.
        return buf(r * _K + k)

    def dot(row_at, n):
        return sum(row_at(k) * W[k][n] for k in range(_K))

    batched = [dot(lambda k: x_at(b, m, k), n) + bias[n] for n in range(_N)]
    gemm_row = b * M + m
    gemm = [dot(lambda k: x2_at(gemm_row, k), n) + bias[n] for n in range(_N)]

    claim = z3.And(*[batched[n] == gemm[n] for n in range(_N)])
    prove(claim)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def test_fuse_matmul_add_bias_into_gemm_batched_pass_matches():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 5))
    B = rng.standard_normal(5)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,8] X) => (float[2,3,5] Y)
        {
          mm = MatMul(X, W)
          Y = Add(mm, B)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W"), _f32(B, "B")])
    sim_model, ops = simplify_isolated_extra(
        model, "fuse_matmul_add_bias_into_gemm_batched"
    )
    assert ops["Gemm"] == 1
    assert ops["Reshape"] == 2  # X flattened in, output un-flattened back out
    # Add is destroyed (it's the matched node); MatMul is left dangling
    # (0 uses) rather than destroyed -- the same pattern as the non-batched
    # sibling pass (see test_formal_verify_fuse_matmul_add_bias_into_gemm.py).
    assert ops["Add"] == 0
    assert ops["MatMul"] == 1

    gemm_node = next(n for n in sim_model.graph.node if n.op_type == "Gemm")
    attrs = {a.name: a for a in gemm_node.attribute}
    assert attrs.get("alpha") is None or attrs["alpha"].f == 1.0
    assert attrs.get("beta") is None or attrs["beta"].f == 1.0
    assert attrs.get("transA") is None or attrs["transA"].i == 0
    assert attrs.get("transB") is None or attrs["transB"].i == 0


def test_fuse_matmul_add_bias_into_gemm_batched_matches_swapped_operand_order():
    # Unlike the non-batched fuse_matmul_add_bias_into_gemm (which only
    # matches Add(MatMul(...), bias), not the swapped order), this pass
    # matches Add(bias, MatMul(...)) too.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 5))
    B = rng.standard_normal(5)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,8] X) => (float[2,3,5] Y)
        {
          mm = MatMul(X, W)
          Y = Add(B, mm)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W"), _f32(B, "B")])
    _, ops = simplify_isolated_extra(model, "fuse_matmul_add_bias_into_gemm_batched")
    assert ops["Gemm"] == 1
    assert ops["Add"] == 0
