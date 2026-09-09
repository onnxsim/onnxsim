"""Formal check for EliminateNopTranspose (eliminate_nop_transpose.h).

The rewrite requires the ``perm`` attribute to be *present* -- a Transpose
with no ``perm`` attribute (whose ONNX default semantics reverse every axis,
which only happens to be a no-op for rank <= 1) is never matched by this
pass at all, regardless of rank -- and to be exactly the identity
permutation ``[0, 1, ..., n-1]`` (``is_nop_transpose``). It then rewires
every use of the Transpose's output directly to its input
(``tryReplacingAllUsesWith``), the same rewiring EliminateIdentity uses.

Soundness is the special case of fuse_consecutive_transposes' own
permutation algebra (see test_formal_verify_fuse_consecutive_transposes.py)
where perm is the identity: scattering a symbolic index through the
identity permutation returns that index unchanged, so an uninterpreted
tensor read at the "transposed" index equals the read at the original
index, for every index -- a genuine universal proof, not a finite sample.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser

_RANK = 3
_IDENTITY_PERM = list(range(_RANK))


def _scatter(idx, perm):
    # transpose(T, perm)[idx] reads T at the index built by scattering idx
    # through perm: result[perm[i]] = idx[i] (ONNX/numpy Transpose semantics).
    out = [None] * len(perm)
    for i, p in enumerate(perm):
        out[p] = idx[i]
    return out


def test_eliminate_nop_transpose_is_sound():
    tensor = z3.Function("tensor", *([z3.IntSort()] * _RANK), z3.RealSort())
    j = [z3.Int(f"j{i}") for i in range(_RANK)]
    prove(tensor(*_scatter(j, _IDENTITY_PERM)) == tensor(*j))


def test_eliminate_nop_transpose_pass_matches():
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,4] X) => (float[2,3,4] Y)
        {
          t = Transpose<perm = [0, 1, 2]>(X)
          Y = Relu(t)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_transpose")
    assert ops["Transpose"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_transpose_declines_missing_perm():
    # A Transpose with no perm attribute is never matched by this pass, even
    # though ONNX's default (reverse every axis) happens to be a no-op here
    # only for rank <= 1 -- eliminate_nop_transpose.h's own
    # patternMatchPredicate requires hasAttribute(kperm) and does not
    # special-case the missing-attribute default at all.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,4] X) => (float[4,3,2] Y)
        {
          Y = Transpose(X)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_transpose")
    assert ops["Transpose"] == 1
