"""Formal check for FuseConsecutiveTransposes (fuse_consecutive_transposes.h).

Two consecutive Transpose nodes collapse into one whose perm is
``compose_transposes(t1, t2)[i] = t1[t2[i]]``, where ``t1`` is the earlier
(inner) transpose's perm and ``t2`` the later (outer) one's -- see that
function in fuse_consecutive_transposes.h, reproduced exactly (not just
approximated) in ``_compose_transposes`` below.

Soundness here is pure permutation algebra, independent of tensor shape or
dtype: modeling the tensor as an uninterpreted function from a symbolic
integer index tuple to a real value lets Z3 prove the fused single transpose
reads the exact same value at every index as the two original transposes
applied in sequence -- for every rank-4 index, a genuine universal proof
(free variables in a Z3 validity check are implicitly universally
quantified), not just a finite sample of concrete indices.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser

_RANK = 4
_T1 = [1, 0, 2, 3]  # inner transpose (applied first): swap axes 0 and 1
_T2 = [0, 2, 1, 3]  # outer transpose (applied second): swap axes 1 and 2


def _compose_transposes(t1, t2):
    # Mirrors fuse_consecutive_transposes.h's own compose_transposes exactly.
    return [t1[t2[i]] for i in range(len(t1))]


def _scatter(idx, perm):
    # transpose(T, perm)[idx] reads T at the index built by scattering idx
    # through perm: result[perm[i]] = idx[i] (ONNX/numpy Transpose semantics:
    # output.dims[i] = input.dims[perm[i]]).
    out = [None] * len(perm)
    for i, p in enumerate(perm):
        out[p] = idx[i]
    return out


def test_fuse_consecutive_transposes_is_sound():
    tf = _compose_transposes(_T1, _T2)

    tensor = z3.Function("tensor", *([z3.IntSort()] * _RANK), z3.RealSort())
    j = [z3.Int(f"j{i}") for i in range(_RANK)]

    two_step = tensor(*_scatter(_scatter(j, _T2), _T1))
    fused = tensor(*_scatter(j, tf))
    prove(two_step == fused)


def test_fuse_consecutive_transposes_pass_matches():
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,4,5] X) => (float[3,4,2,5] Y)
        {{
          t = Transpose<perm = {_T1}>(X)
          Y = Transpose<perm = {_T2}>(t)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_transposes")
    assert ops["Transpose"] == 1
    fused_node = next(n for n in sim_model.graph.node if n.op_type == "Transpose")
    perm = next(a.ints for a in fused_node.attribute if a.name == "perm")
    assert list(perm) == _compose_transposes(_T1, _T2)
