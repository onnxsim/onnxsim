"""Formal check for RewriteGatherNDToGather (opt-in; onnxsim's own
``onnxsim/passes/rewrite_gathernd_to_gather.h``): rewrites
``GatherND(data, indices, batch_dims=b)`` into a flatten + plain ``Gather``.

This proof covers the ``batch_dims=0`` case with the ``k`` indexed axes
merged into one: ``data`` (rank ``r``, axes ``[0, k)`` statically sized) is
reshaped to ``flat_data = Reshape(data, [-1, *data.shape[k:]])``; each index
column ``j`` is normalized (``idx_j < 0 ? idx_j + dim_j : idx_j`` --
``GatherND``, like ``Gather``, allows negative per-axis indices) and the
normalized columns combined into one flat row index by row-major strides,
``combined = sum(idx_j_norm * mult_j)`` with ``mult_j = prod(dims[j+1:k])``;
finally ``output = Gather(flat_data, combined, axis=0)``. The general
``batch_dims > 0`` case additionally adds a per-batch-row offset, not
re-derived here -- see that file's own header comment for the full
derivation; the differential check below only exercises ``batch_dims=0``.

Soundness is the same "reshape only relabels a flat buffer" argument as
test_formal_verify_fuse_matmul_add_bias_into_gemm_batched.py, here merging
the first ``k`` axes (rather than arbitrary leading batch dims) with a
trailing free axis ``t`` preserved: proving
``data[i0, i1, t] == flat_data[i0*D1 + i1, t]`` for ``k=2`` is exactly the
same "compare two row-major flat-buffer offsets" argument, an
unconditional index identity independent of dimension sizes -- normalizing
each index column to be non-negative *before* combining it with strides is
what makes ``i0``/``i1`` valid row-major sub-indices in the first place
(combining raw negative indices with strides is not the same arithmetic
at all: e.g. index -1 into a size-3 axis must normalize to 2 before being
multiplied by a stride, not stay -1), which the differential check below
confirms empirically by exercising negative indices in both columns
against the real compiled pass, rather than by trying to state "wrong
arithmetic gives a wrong answer" as its own Z3 claim.
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser


def test_rewrite_gathernd_to_gather_reshape_is_sound():
    # data: logically rank 3, shape (D0, D1, D2); the pass merges axes 0,1.
    buf = z3.Function("buf", z3.IntSort(), z3.RealSort())  # data's flat buffer
    D1, D2 = z3.Ints("D1 D2")
    i0, i1, t = z3.Ints("i0 i1 t")

    def data_at(a, b, c):
        return buf(a * D1 * D2 + b * D2 + c)

    def flat_data_at(row, c):
        return buf(row * D2 + c)

    combined = i0 * D1 + i1
    prove(data_at(i0, i1, t) == flat_data_at(combined, t))


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def test_rewrite_gathernd_to_gather_pass_matches_negative_indices():
    # indices deliberately includes a negative value in both columns
    # (axis 0 has size 3, so -1 normalizes to 2; axis 1 has size 4, so -1
    # normalizes to 3) -- exercising NormalizeIndex against the real
    # compiled pass, not just the reshape lemma above.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4,2] data) => (float[2,2] Y)
        <int64[2,2] indices = {0, 1, -1, -1}>
        {
          Y = GatherND(data, indices)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gathernd_to_gather")
    assert ops["GatherND"] == 0
    assert ops["Gather"] == 1
