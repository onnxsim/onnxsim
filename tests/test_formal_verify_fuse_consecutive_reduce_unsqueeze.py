"""Formal check for FuseConsecutiveReduceUnsqueeze
(fuse_consecutive_reduce_unsqueeze.h).

``reduction_operators`` is a fixed set of ONNX's axis-reduction family
(``ReduceL1``, ``ReduceL2``, ``ReduceLogSum``, ``ReduceLogSumExp``,
``ReduceMax``, ``ReduceMean``, ``ReduceMin``, ``ReduceProd``, ``ReduceSum``,
``ReduceSumSquare``) that all share the exact same ``axes``/``keepdims``
attribute shape and the exact same "does a reduced axis disappear or become
size-1" behavior controlled by ``keepdims`` -- they differ only in what value
each kept position computes, which is irrelevant to this pass.
``patternMatchPredicate`` matches ``Unsqueeze(ReduceX(...))`` where
``ReduceX`` is one of those kinds, has a ``keepdims`` attribute explicitly
present, and that attribute is currently ``0`` (the reduction currently DROPS
the reduced axes entirely rather than keeping them as size-1).

``runTransform`` requires the reduction op's output to have exactly one use
(only this Unsqueeze consumes it -- the same single-consumer precondition
seen throughout this fusion-pass family, since the transform mutates the
reduction op in place and would otherwise change what every other consumer
sees). It fetches both the Unsqueeze's own ``axes`` and the reduction op's
own ``axes`` (``GetValueFromAttrOrInput``, so each must be a compile-time
constant) and declines unless they are EXACTLY EQUAL AS FETCHED (``axes !=
prev_axes`` bails). This is a stricter, non-normalizing comparison than some
sibling passes: ``GetValueFromAttrOrInput`` -> ``GetValueFromAttr`` /
``GetValueFromInput`` (``pass_util.h``) do no negative-axis normalization at
all before this comparison -- confirmed empirically (see below) with a case
where the two axes lists are semantically the same axis (``-1`` vs. the
literal positive equivalent for a rank-3 tensor) but spelled differently: the
pass declines, exactly as reading the C++ predicts. So "the axes match"
really means "the two lists are byte-for-byte identical *as written* in the
graph", not "the two lists denote the same axes once normalized".

When the axes match, ``runTransform`` rewires the Unsqueeze's OWN consumers
directly to the reduction op's OWN output (skipping the Unsqueeze), then
MUTATES the reduction op itself: sets its ``keepdims`` attribute to ``1``,
and manually copies the Unsqueeze's own output ``sizes``/``elemType`` onto
the reduction op's output (the reduction op's own shape-inference metadata,
computed back when ``keepdims=0``, is stale for the new ``keepdims=1``
shape). It destroys only the Unsqueeze node (``NodeDestroyType::DestroyOne``)
-- the reduction op survives, mutated in place. This is DIFFERENT from most
passes in this suite (which either rewire-and-leave-both-nodes-alive, or
destroy the outer node while leaving an inner one dangling for
``eliminate_deadend`` to sweep up): here there is no dangling node at all,
confirmed empirically below -- after a successful fuse, exactly one node
(the same reduction node, by construction, now with ``keepdims=1``) survives
with the graph's original output name, and the Unsqueeze is gone outright
(not merely 0-use).

Formal content: ``ReduceX(input, axes, keepdims=0)`` followed by
``Unsqueeze(_, axes)`` with the IDENTICAL axes list is equivalent to
``ReduceX(input, axes, keepdims=1)`` directly, because of how ONNX itself
defines ``keepdims``: ``keepdims=0`` computes the reduction and then DROPS
the reduced axes from the output shape entirely (rank decreases by
``len(axes)``); ``keepdims=1`` computes the EXACT SAME per-element reduction
values but KEEPS those axes in the output shape as size-1 dims AT THEIR
ORIGINAL POSITIONS, rather than dropping them. So Unsqueeze-ing the
``keepdims=0`` result back in at exactly the axes that were dropped is
definitionally the same operation as never having dropped them in the first
place (``keepdims=1``).

The model below is the same "insert/remove a size-1 axis" technique as
``fuse_consecutive_squeezes``/``fuse_consecutive_unsqueezes``
(``_remove_inserted`` here, reused from
``test_formal_verify_fuse_consecutive_unsqueezes.py``'s helper of the same
name/shape -- it drops an index's components at each inserted/dim-1 position,
the inverse of squeeze's own ``_insert_removed``), but the soundness argument
is structured as an explicit two-step chain rather than a single self-evident
substitution, since (unlike squeezes-composing-with-squeezes) there is a real
semantic fact of ONNX's ``Reduce*`` operator family being invoked, not just
index bookkeeping:

1. (ONNX's own definition of ``keepdims``, modeled as an axiom below, since
   the actual per-element reduction computation is irrelevant here and would
   otherwise need one uninterpreted function per op kind for no benefit): for
   every full-rank index, the ``keepdims=1`` result at that index equals the
   ``keepdims=0`` result at the index obtained by dropping the components at
   the reduced-axis positions -- i.e. the exact same computed values, merely
   kept at their original positions instead of squeezed out.
2. (Unsqueeze's own semantics): reading ``Unsqueeze(keepdims=0 result,
   axes)`` at a full-rank index reads the (smaller) ``keepdims=0`` result at
   that same dropped-components index.

Chaining (1) and (2) through the identical index-projection (using the SAME
``axes`` on both sides -- modeling the pass's own literal-equality
requirement) gives ``unsqueeze_of_keepdims0(idx) == direct_keepdims1(idx)``
for every full-rank ``idx``, composed with an arbitrary uninterpreted
``consumer`` for substitution safety, matching this suite's established
style (see e.g. ``test_formal_verify_eliminate_consecutive_idempotent_ops.py``).
A throwaway negative-control script (since deleted) confirmed Z3 actually
exercises this index arithmetic rather than trivially discharging the claim:
corrupting either the reinsertion axes or the axiom's own axes to a different
list makes the claim `sat` with a genuine counterexample.

Everything above the "Formal content" heading that describes the real
compiled pass's behavior (single node surviving with the graph's original
output name, no dangling node, the non-normalizing literal axes comparison)
was confirmed against the actual compiled pass via a throwaway debug script
(since deleted), not just assumed from reading the header comment.
"""

from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser

_RANK0 = 4  # X's rank, before the reduction
_AXES = [1, 3]  # reduced (and re-Unsqueeze'd) axes, relative to X


def _remove_inserted(idx, sorted_inserted_axes, rank):
    # Unsqueeze(T, axes=sorted_inserted_axes)[idx] reads T at the index built
    # by dropping idx's components at each inserted position (an inserted,
    # dim-1 axis only ever has index 0 there) and taking the rest, in order.
    # Reused from test_formal_verify_fuse_consecutive_unsqueezes.py's helper
    # of the same name/shape; this pass's own reduced-axis positions play the
    # same "dim-1 axis whose only valid coordinate is dropped" role that an
    # Unsqueeze's own inserted axes do there -- ReduceX(keepdims=0) is
    # exactly "Unsqueeze but in reverse: the dim-1 axes never got inserted at
    # all", and the Unsqueeze this pass fuses away re-inserts them at those
    # same positions.
    return [idx[p] for p in range(rank) if p not in sorted_inserted_axes]


def _ints_literal(xs):
    # ONNX text-format tensor literal, e.g. [1, 3] -> "{1, 3}" (as opposed to
    # the "[1, 3]" square-bracket syntax used for int-list *attributes*).
    return "{" + ", ".join(map(str, xs)) + "}"


def test_fuse_consecutive_reduce_unsqueeze_is_sound():
    rank_reduced = _RANK0 - len(_AXES)  # keepdims=0 result's rank

    # keepdims=0 result: an uninterpreted function of the reduced-rank index
    # -- SOME reduction value at each kept position; which reduction
    # (sum/mean/max/...) is irrelevant, since every reduction_operators kind
    # shares this exact axes/keepdims shape-handling logic.
    reduce_keepdims0 = z3.Function(
        "reduce_keepdims0", *([z3.IntSort()] * rank_reduced), z3.RealSort()
    )
    # keepdims=1 result: a SEPARATE uninterpreted function of the full-rank
    # index, related to reduce_keepdims0 only via the axiom below -- modeling
    # "the compiled reduction op, reconfigured to keepdims=1" as its own
    # opaque computation, not something trivially equal to the first function
    # by construction.
    reduce_keepdims1 = z3.Function(
        "reduce_keepdims1", *([z3.IntSort()] * _RANK0), z3.RealSort()
    )
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    sorted_axes = sorted(_AXES)
    a = [z3.Int(f"a{i}") for i in range(_RANK0)]  # bound inside the axiom
    idx = [z3.Int(f"idx{i}") for i in range(_RANK0)]  # free, universally proven over

    # Step 1: ONNX's own definition of keepdims -- keepdims=1 computes the
    # exact same per-element values as keepdims=0, just kept at their
    # original positions instead of squeezed out.
    keepdims_semantics = z3.ForAll(
        a,
        reduce_keepdims1(*a)
        == reduce_keepdims0(*_remove_inserted(a, sorted_axes, _RANK0)),
    )

    # Step 2: Unsqueeze(keepdims=0 result, axes) read at a full-rank index.
    unsqueeze_of_keepdims0 = reduce_keepdims0(
        *_remove_inserted(idx, sorted_axes, _RANK0)
    )
    # The direct keepdims=1 read at that same index.
    direct_keepdims1 = reduce_keepdims1(*idx)

    prove(
        z3.Implies(
            keepdims_semantics,
            consumer(unsqueeze_of_keepdims0) == consumer(direct_keepdims1),
        )
    )


def test_fuse_consecutive_reduce_unsqueeze_pass_matches_reducesum():
    # ReduceSum(X, axes, keepdims=0) -> Unsqueeze(_, axes), identical axes:
    # fuses. Only the ReduceSum node survives (same node, mutated), reading
    # straight from X, with keepdims flipped to 1 and the graph's original
    # output name ("Z", formerly the Unsqueeze's) now on the reduction node.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[2,3,4,5] X) => (float[2,1,4,1] Z)
        <int64[{len(_AXES)}] axes = {_ints_literal(_AXES)}>
        {{
          r = ReduceSum <keepdims=0> (X, axes)
          Z = Unsqueeze(r, axes)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_reduce_unsqueeze")
    assert ops["Unsqueeze"] == 0
    assert ops["ReduceSum"] == 1
    (node,) = sim_model.graph.node
    assert node.op_type == "ReduceSum"
    assert list(node.input) == ["X", "axes"]
    assert list(node.output) == ["Z"]
    keepdims_attr = next(a for a in node.attribute if a.name == "keepdims")
    assert keepdims_attr.i == 1
    (out,) = sim_model.graph.output
    assert [d.dim_value for d in out.type.tensor_type.shape.dim] == [2, 1, 4, 1]


def test_fuse_consecutive_reduce_unsqueeze_pass_matches_reducemax():
    # Same shape of fusion, different reduction_operators member -- confirms
    # the predicate/transform genuinely isn't kind-specific.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[2,3,4,5] X) => (float[2,1,4,1] Z)
        <int64[{len(_AXES)}] axes = {_ints_literal(_AXES)}>
        {{
          r = ReduceMax <keepdims=0> (X, axes)
          Z = Unsqueeze(r, axes)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_reduce_unsqueeze")
    assert ops["Unsqueeze"] == 0
    assert ops["ReduceMax"] == 1
    (node,) = sim_model.graph.node
    keepdims_attr = next(a for a in node.attribute if a.name == "keepdims")
    assert keepdims_attr.i == 1


def test_fuse_consecutive_reduce_unsqueeze_declines_on_multi_use_reduction():
    # Declining case: the reduction op's output is ALSO returned as a second
    # graph output, so it has more than one use. The pass would otherwise
    # have to change what that other consumer sees (keepdims=0 -> 1 changes
    # the output's rank), so runTransform's own single-use check bails --
    # both nodes survive, keepdims still 0.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[2,3,4,5] X) => (float[2,4] R, float[2,1,4,1] Z)
        <int64[{len(_AXES)}] axes = {_ints_literal(_AXES)}>
        {{
          R = ReduceSum <keepdims=0> (X, axes)
          Z = Unsqueeze(R, axes)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_reduce_unsqueeze")
    assert ops["ReduceSum"] == 1
    assert ops["Unsqueeze"] == 1
    reduce_node = producer(sim_model, "R")
    keepdims_attr = next(a for a in reduce_node.attribute if a.name == "keepdims")
    assert keepdims_attr.i == 0
    unsqueeze_node = producer(sim_model, "Z")
    assert list(unsqueeze_node.input) == ["R", "axes"]


def test_fuse_consecutive_reduce_unsqueeze_declines_on_axes_mismatch():
    # Declining case: the Unsqueeze's own axes ([2], a single axis) do not
    # exactly equal the reduction's own axes ([1, 3]) -- both nodes survive
    # unchanged.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[2,3,4,5] X) => (float[2,1,4,1,1] Z)
        <int64[{len(_AXES)}] axes_r = {_ints_literal(_AXES)}, int64[1] axes_u = {{2}}>
        {{
          r = ReduceSum <keepdims=0> (X, axes_r)
          Z = Unsqueeze(r, axes_u)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_reduce_unsqueeze")
    assert ops["ReduceSum"] == 1
    assert ops["Unsqueeze"] == 1
    reduce_node = producer(sim_model, "r")
    keepdims_attr = next(a for a in reduce_node.attribute if a.name == "keepdims")
    assert keepdims_attr.i == 0
    unsqueeze_node = producer(sim_model, "Z")
    assert list(unsqueeze_node.input) == ["r", "axes_u"]


def test_fuse_consecutive_reduce_unsqueeze_declines_on_negative_vs_positive_axes():
    # Declining case, more subtle than a genuine axes mismatch: the two axes
    # lists denote the SAME axis semantically (X has rank 3, so axis -1 and
    # axis 2 are the same axis), but are spelled differently. GetValueFromAttrOrInput
    # does no negative-axis normalization before the pass's `axes != prev_axes`
    # comparison, so this declines exactly like a genuine mismatch would --
    # confirmed against the real compiled pass, not assumed from reading the
    # header (see module docstring).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[2,3,4] X) => (float[2,3,1] Z)
        <int64[1] axes_r = {-1}, int64[1] axes_u = {2}>
        {
          r = ReduceSum <keepdims=0> (X, axes_r)
          Z = Unsqueeze(r, axes_u)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_reduce_unsqueeze")
    assert ops["ReduceSum"] == 1
    assert ops["Unsqueeze"] == 1
    reduce_node = producer(sim_model, "r")
    keepdims_attr = next(a for a in reduce_node.attribute if a.name == "keepdims")
    assert keepdims_attr.i == 0


def test_fuse_consecutive_reduce_unsqueeze_predicate_declines_on_existing_keepdims():
    # Declining case: the reduction op already has keepdims=1. patternMatchPredicate's
    # own `prev_node->i(kkeepdims) == 0` check should decline outright -- a
    # sanity check that the predicate correctly restricts to the keepdims=0
    # starting state (an Unsqueeze after an already-keepdims=1 reduction is
    # not this pass's business, whatever else might be true of that graph).
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        g (float[2,3,4,5] X) => (float[2,1,4,1,1] Z)
        <int64[{len(_AXES)}] axes = {_ints_literal(_AXES)}>
        {{
          r = ReduceSum <keepdims=1> (X, axes)
          Z = Unsqueeze(r, axes)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_reduce_unsqueeze")
    assert ops["ReduceSum"] == 1
    assert ops["Unsqueeze"] == 1
    reduce_node = producer(sim_model, "r")
    keepdims_attr = next(a for a in reduce_node.attribute if a.name == "keepdims")
    assert keepdims_attr.i == 1
