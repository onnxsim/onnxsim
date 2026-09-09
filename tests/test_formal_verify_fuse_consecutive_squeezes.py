"""Formal check for FuseConsecutiveSqueezes (fuse_consecutive_squeezes.h).

Two consecutive Squeeze nodes collapse into one whose axes list is
``compose_squeezes(axes_1, axes_2)``, where ``axes_1`` is the earlier
(inner) squeeze's axes and ``axes_2`` the later (outer) one's -- see that
function in fuse_consecutive_squeezes.h, reproduced exactly (not just
approximated) in ``_compose_squeezes`` below.

Squeeze's ``axes`` (pre-opset-13: the ``axes`` attribute; opset 13+: the
second input, which ``GetValueFromAttrOrInput`` resolves via
``GetValueFromInput`` -> ``FetchConstantTensor``, so it must be a
*compile-time* constant -- an initializer or a ``Constant`` node output --
for the pass to see it at all) are always expressed relative to the tensor
as it stands *at that point in the graph*. That is where composing two
Squeezes gets interesting, unlike composing two Transposes
(test_formal_verify_fuse_consecutive_transposes.py): ``axes_1`` is already
relative to the original (pre-either-squeeze) tensor, but ``axes_2`` is
relative to the *once-squeezed* (rank-reduced) tensor that ``axes_1``
produced. Combining them into one axes list relative to the *original*
tensor means re-indexing every entry of ``axes_2`` through the axes that
``axes_1`` already removed -- shifting each one by however many of
``axes_1``'s entries land at or before it. Getting that shift-by-how-many-
already-removed-axes-come-before-this-one arithmetic wrong is a real "index
remapping" bug class; ``compose_squeezes`` in fuse_consecutive_squeezes.h
reads (with ``sorted_axes_1 = sorted(axes_1)``):

    ret = sorted_axes_1                       # already original-relative
    for i in axes_2:                          # each already-squeezed-relative
        for prev_num, a in enumerate(sorted_axes_1):
            if a - prev_num > i:               # i lands before original axis a
                ret.append(i + prev_num)
                break
        else:                                  # i lands after every axes_1 entry
            ret.append(i + len(sorted_axes_1))
    ret.sort()

reproduced verbatim (as a direct transliteration, not just approximated) in
``_compose_squeezes`` below. The predicate also declines (returns false,
leaving both Squeeze nodes in place) whenever either axes list contains a
negative value -- ``compose_squeezes`` explicitly bails on that -- so both
axes lists are modeled here as containing only non-negative axes, matching
that implicit side condition.

Soundness is proven by modeling the tensor as an uninterpreted function from
a symbolic integer index tuple to a real value, and defining
``_insert_removed`` -- the inverse of squeeze's own index-shift: given an
index into the *squeezed* tensor, reconstruct the corresponding index into
the *pre-squeeze* tensor by inserting a ``0`` at each removed axis (the only
valid coordinate there, since a squeezed axis must have had dim 1) and
threading the given index's components through every other (kept) position.
Applying it twice (once per squeeze, composed) versus once with the fused
axes list must read the exact same value at every remaining index -- for
every index into the twice-squeezed tensor, a genuine universal proof (free
variables in a Z3 validity check are implicitly universally quantified), not
just a finite sample of concrete indices. As with the transpose proof, the
two axes lists are concrete (matching the pass's own worked example in its
header comment) rather than fully symbolic, so the index arithmetic stays
tractable for Z3.
"""

import collections

from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import numpy_helper, parser

import onnxsim

_RANK0 = 7  # X's rank, before either squeeze
_AXES_1 = [1, 4]  # inner squeeze's axes (relative to X)
_AXES_2 = [0, 4]  # outer squeeze's axes (relative to the once-squeezed tensor)


def _compose_squeezes(axes_1, axes_2):
    # Mirrors fuse_consecutive_squeezes.h's own compose_squeezes exactly (see
    # the module docstring for the annotated original).
    sorted_axes_1 = sorted(axes_1)
    ret = list(sorted_axes_1)
    for i in axes_2:
        for prev_num, a in enumerate(sorted_axes_1):
            if a - prev_num > i:
                ret.append(i + prev_num)
                break
        else:
            ret.append(i + len(sorted_axes_1))
    return sorted(ret)


def _insert_removed(idx, sorted_removed_axes, rank):
    # Squeeze(T, axes=sorted_removed_axes)[idx] reads T at the index built by
    # inserting a 0 (the only valid coordinate on a removed, dim-1 axis) at
    # each removed position, and taking idx's components, in order, at every
    # other (kept) position -- the inverse of squeeze's own index-shift.
    it = iter(idx)
    return [0 if p in sorted_removed_axes else next(it) for p in range(rank)]


def _ints_literal(xs):
    # ONNX text-format tensor literal, e.g. [1, 4] -> "{1, 4}" (as opposed to
    # the "[1, 4]" square-bracket syntax used for int-list *attributes*).
    return "{" + ", ".join(map(str, xs)) + "}"


def test_fuse_consecutive_squeezes_is_sound():
    composed = _compose_squeezes(_AXES_1, _AXES_2)
    rank1 = _RANK0 - len(_AXES_1)  # once-squeezed tensor's rank
    rank2 = rank1 - len(_AXES_2)  # twice-squeezed tensor's rank
    assert rank2 == _RANK0 - len(composed)

    tensor = z3.Function("tensor", *([z3.IntSort()] * _RANK0), z3.RealSort())
    z = [z3.Int(f"z{i}") for i in range(rank2)]

    two_step = tensor(
        *_insert_removed(
            _insert_removed(z, sorted(_AXES_2), rank1), sorted(_AXES_1), _RANK0
        )
    )
    fused = tensor(*_insert_removed(z, composed, _RANK0))
    prove(two_step == fused)


def test_fuse_consecutive_squeezes_pass_matches():
    # X's shape and the two axes lists are exactly fuse_consecutive_squeezes.h's
    # own header-comment worked example: [1,1,2,3,1,5,1] --Squeeze(axes=[1,4])->
    # [1,2,3,5,1] --Squeeze(axes=[0,4])-> [2,3,5], fusing to Squeeze(axes=[0,1,4,6]).
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,1,2,3,1,5,1] X) => (float[2,3,5] Z)
        <int64[{len(_AXES_1)}] axes1 = {_ints_literal(_AXES_1)},
         int64[{len(_AXES_2)}] axes2 = {_ints_literal(_AXES_2)}>
        {{
          y = Squeeze(X, axes1)
          Z = Squeeze(y, axes2)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_squeezes")
    assert ops["Squeeze"] == 1
    (fused_node,) = [n for n in sim_model.graph.node if n.op_type == "Squeeze"]
    axes_init = next(
        init for init in sim_model.graph.initializer if init.name == fused_node.input[1]
    )
    assert list(numpy_helper.to_array(axes_init)) == _compose_squeezes(_AXES_1, _AXES_2)


def test_fuse_consecutive_squeezes_declines_non_constant_axes():
    # The outer Squeeze's axes come from an Add of two initializers rather
    # than being an initializer (or Constant node) themselves. The value is
    # still deterministically [0, 4] at runtime -- the model computes the
    # same thing either way, so onnxsim's own numeric --check still passes --
    # but FetchConstantTensor only recognizes a Constant node or an
    # initializer directly, not an arbitrary constant-foldable expression, so
    # compose_squeezes can't retrieve axes_2 and runTransform declines,
    # leaving both Squeeze nodes (and the Add) in place. This bypasses
    # simplify_isolated (which only controls the *optimizer pass* list) and
    # calls onnxsim.simplify directly with skip_constant_folding=True --
    # onnxsim's constant folding is a separate step that runs before the
    # optimizer passes regardless of skipped_optimizers, and would otherwise
    # fold the Add away before fuse_consecutive_squeezes ever saw it.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,1,2,3,1,5,1] X) => (float[2,3,5] Z)
        <int64[{len(_AXES_1)}] axes1 = {_ints_literal(_AXES_1)},
         int64[2] axes2_a = {{0, 3}},
         int64[2] axes2_b = {{0, 1}}>
        {{
          y = Squeeze(X, axes1)
          axes2 = Add(axes2_a, axes2_b)
          Z = Squeeze(y, axes2)
        }}
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_consecutive_squeezes"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Squeeze"] == 2
    assert ops["Add"] == 1
