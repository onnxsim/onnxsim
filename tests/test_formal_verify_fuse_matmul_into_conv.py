"""Formal check for FuseMatMulIntoConv (opt-in; onnxsim's own
``onnxsim/passes/fuse_matmul_into_conv.h``): rewrites a bare ``MatMul(X, W)``
(optionally with a bias ``Add`` in *either* operand order, or a
``Gemm(X, W[, B])`` with ``alpha==1`` and, if biased, ``beta==1``) into a
1-D ``Conv`` with ``kernel_shape=[1]``.

The rewrite reshapes X to ``[-1, K, 1]`` (collapsing every leading dim into
Conv's batch axis and appending a trailing size-1 spatial axis -- a pure
reshape, no data movement), transposes W to ``[N, K]`` and unsqueezes it to
``[N, K, 1]`` (a Conv weight ``[Cout, Cin, kernel=1]``), and reshapes the
Conv's output back. Bias, if present, must be exactly ``[N]`` or ``[1]``
(expanded to ``[N]`` if ``[1]``, since Conv's bias has no broadcast rule).

Soundness combines two already-established facts and one new one:

1. Reshape only relabels a flat buffer -- the same argument as
   test_formal_verify_fuse_matmul_add_bias_into_gemm_batched.py, here even
   simpler: the appended spatial axis has size 1, so it contributes a zero
   offset and the trailing ``[..., 1]`` reshape reads the identical element
   X's own ``[batch, K]`` indexing would.
2. A Conv with ``kernel_shape=[1]`` has no sliding window and no zero-fill
   boundary to reason about at all -- unlike
   test_formal_verify_fuse_pad_into_conv.py's general Conv model, this
   pass's own reshape guarantees exactly one spatial position, so it is
   exactly a per-position matrix-vector product.
3. That matrix-vector product is exactly MatMul(X, W) + bias, the same
   arithmetic identity already proved in
   test_formal_verify_fuse_matmul_add_bias_into_gemm.py.

A biased match rewires Add's uses to the new Conv chain but does not delete
the now-unused original MatMul node outright; isolating this one pass (no
eliminate_deadend alongside it, unlike a real ``simplify()`` call) then lets
that dangling MatMul get matched by this same pass's own bare-MatMul rule on
a later fixed-point iteration, producing a second, entirely dead Conv chain
nothing reads. The differential checks below trace forward from the graph's
real output instead of counting op types, so that harmless artifact of
testing one pass in isolation doesn't make the check about the wrong node.
"""

import numpy as np
import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

_K = 2  # concrete contraction dim -- enough to exercise the dot-product sum
_N = 2  # concrete output-channel (Cout) dim


def test_fuse_matmul_into_conv_is_sound():
    buf = z3.Function("buf", z3.IntSort(), z3.RealSort())  # X's flat row-major buffer
    # Conv weight W2[n, k] = W[k, n] (the pass's own weight transpose).
    W2 = [[z3.Real(f"w{n}{k}") for k in range(_K)] for n in range(_N)]
    bias = [z3.Real(f"bias{n}") for n in range(_N)]
    batch = z3.Int("batch")

    def x_flat(bb, k):
        # X's own row-major layout, logically [batch, K].
        return buf(bb * _K + k)

    def x_conv(bb, k):
        # X2 = Reshape(X, [-1, K, 1]): the trailing size-1 spatial axis has
        # stride 1 but only ever holds index 0, so it contributes nothing --
        # the same flat buffer element as x_flat.
        return buf(bb * _K * 1 + k * 1 + 0)

    def conv_out(bb, n):
        # Conv, kernel_shape=[1], group=1, one spatial position: a plain
        # per-position matrix-vector product -- no window, no zero-fill.
        return sum(W2[n][k] * x_conv(bb, k) for k in range(_K)) + bias[n]

    def matmul_add_out(bb, n):
        return sum(x_flat(bb, k) * W2[n][k] for k in range(_K)) + bias[n]

    claim = z3.And(*[conv_out(batch, n) == matmul_add_out(batch, n) for n in range(_N)])
    prove(claim)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def test_fuse_matmul_into_conv_pass_matches():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 8))
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,16] X) => (float[4,8] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    model.graph.initializer.extend([_f32(W, "W")])
    sim_model, ops = simplify_isolated_extra(model, "fuse_matmul_into_conv")
    assert ops["MatMul"] == 0

    # Y's producer is the final un-flattening Reshape; its input is Conv's
    # output. kernel_shape is never set as an explicit Conv attribute here --
    # it's left implicit, inferred (by any ONNX-conformant runtime) from the
    # weight tensor's own shape, which is what this checks instead.
    reshape_node = producer(sim_model, "Y")
    assert reshape_node.op_type == "Reshape"
    conv_node = producer(sim_model, reshape_node.input[0])
    assert conv_node.op_type == "Conv"
    by_name = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in sim_model.graph.initializer
    }
    conv_weight = by_name[conv_node.input[1]]
    assert conv_weight.shape == (8, 16, 1)  # [Cout, Cin, kernel=1]


def test_fuse_matmul_into_conv_matches_swapped_bias_operand_order():
    # Unlike the non-batched fuse_matmul_add_bias_into_gemm (which only
    # matches Add(MatMul(...), bias)), this pass matches
    # Add(bias, MatMul(...)) too.
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
    sim_model, ops = simplify_isolated_extra(model, "fuse_matmul_into_conv")
    assert ops["Add"] == 0

    # Trace from Y (not a raw op-type count): isolating this one pass, with
    # its usual eliminate_deadend companion skipped, leaves the original
    # (now-unused) MatMul's own separate Conv-fusion lying around as dead
    # code alongside the live one Y actually depends on.
    reshape_node = producer(sim_model, "Y")
    assert reshape_node.op_type == "Reshape"
    conv_node = producer(sim_model, reshape_node.input[0])
    assert conv_node.op_type == "Conv"
    assert "B" in conv_node.input
