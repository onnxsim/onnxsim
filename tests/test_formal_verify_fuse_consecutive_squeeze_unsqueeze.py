"""Formal check for FuseConsecutiveSqueezeUnsqueeze
(fuse_consecutive_squeeze_unsqueeze.h).

Unlike ``fuse_consecutive_squeezes``/``fuse_consecutive_unsqueezes`` (two
nodes of the *same* kind, generally *different* axes lists, fused into one
node with a *composed* axes list), this pass matches a Squeeze immediately
followed by an Unsqueeze, or an Unsqueeze immediately followed by a Squeeze
(``patternMatchPredicate``), fetches both nodes' ``axes`` (``kaxes`` -- an
attribute pre-opset-13, the second input from opset 13 on --
``GetValueFromAttrOrInput``, so each must be a compile-time constant),
normalizes negative entries against a shared "reference rank" (the
Squeeze's own input rank, which by construction equals the Unsqueeze's own
output rank -- ``ReferenceRank``, recovered from whichever value carries
static shape info; declines when it is unknown *and* a negative axis is
present, same pattern as ``fuse_consecutive_unsqueezes``), sorts both lists,
and -- crucially -- declines *unless they are exactly equal*
(``axes != prev_axes`` bails: this pass does no index remapping at all).
When they match, the whole pair collapses to nothing: every consumer of the
*outer* node's output is rewired straight to the *inner* node's own input
(``tryReplacingAllUsesWith(node->output(), prev->input(0))``), and only the
outer node is destroyed (``NodeDestroyType::DestroyOne``) -- confirmed
empirically below, the inner node is left dangling, unlike
``fuse_consecutive_squeezes`` (which explicitly destroys the now-unused
inner node too) and *like* ``fuse_consecutive_unsqueezes`` (which also
leaves its dead inner node in place).

Formal content: Squeeze removes a set of size-1 axes, by their *index
position* on the tensor it reads; Unsqueeze inserts size-1 axes back, also
by index position. ``_insert_removed`` below (Squeeze's own index
reconstruction, reproduced from ``test_formal_verify_fuse_consecutive_squeezes.py``)
and ``_remove_inserted`` (Unsqueeze's, reproduced from
``test_formal_verify_fuse_consecutive_unsqueezes.py``) are exact inverses of
each other *on the smaller ("squeezed"/pre-unsqueeze) index space* when
given the *same* axes list -- ``_remove_inserted(_insert_removed(idx, axes,
rank), axes, rank) == idx`` unconditionally, no shift/remap arithmetic and
no side condition needed, because insert always places a 0 at exactly the
positions remove then discards. This is the whole soundness argument: it is
what makes the pass's decision to require the *same* axes list (rather than
doing the composition arithmetic the sibling passes do) sufficient. Note this
is *not* the same as saying "reading Z at an arbitrary full-rank index equals
reading X there" for literally every index -- it only needs to hold at the
indices a real (size-1-there) tensor can ever have, and those are exactly
the ones ``_insert_removed`` produces, which is exactly the index set both
soundness tests below quantify over.

Both node orders are covered, since the predicate matches them as genuinely
separate cases (``CheckKind(node, kUnsqueeze, 0, kSqueeze)`` and
``CheckKind(node, kSqueeze, 0, kUnsqueeze)``) even though the underlying
index argument is symmetric:

* Squeeze then Unsqueeze: X (rank ``_RANK0``) -> Squeeze(axes) -> y (rank
  ``rank1``) -> Unsqueeze(axes) -> Z (rank ``_RANK0`` again). For a symbolic
  index ``idx`` into y (length ``rank1``), ``w = _insert_removed(idx, axes,
  _RANK0)`` is the corresponding index into X *and* Z (both full rank); the
  claim is ``Z[w] == X[w]``, which unwinds via the two nodes' own read
  semantics (``y[i] = X[_insert_removed(i, axes, _RANK0)]``,
  ``Z[j] = y[_remove_inserted(j, axes, _RANK0)]``) straight into the round
  trip lemma.
* Unsqueeze then Squeeze: X (rank ``rank1``) -> Unsqueeze(axes) -> y (rank
  ``_RANK0``) -> Squeeze(axes) -> Z (rank ``rank1`` again, X's own rank
  here). For a symbolic index ``idx`` into X directly (length ``rank1``),
  the claim is ``Z[idx] == X[idx]``, which unwinds the same way
  (``y[j] = X[_remove_inserted(j, axes, _RANK0)]``,
  ``Z[i] = y[_insert_removed(i, axes, _RANK0)]``) into the exact same round
  trip lemma, applied in the same insert-then-remove order.

Each soundness test models the relevant tensor as an uninterpreted function
from a symbolic integer index tuple to a real value and proves the composed
(two-node) read equals the direct read at every index, composed with an
arbitrary uninterpreted ``consumer`` (matching
``test_formal_verify_eliminate_identity.py``'s style: this is what makes it
a real "any downstream computation sees the same thing" argument, not just
restating ``x == x``) -- a genuine universal proof (free variables in a Z3
validity check are implicitly universally quantified), not a finite sample.
As with the sibling files, the axes list is concrete (matching the pass's
own header-comment worked example, ``axes=[1, 3]`` on a rank-4 tensor)
rather than fully symbolic, so the index arithmetic stays tractable for Z3.

One thing confirmed empirically against the actual compiled pass (rather
than assumed from reading the header) before writing the differential tests
below: a model that plugs the Squeeze/Unsqueeze pair directly between a
graph input and a graph output has both endpoints of the rewire count as
graph boundary values, and ``tryReplacingAllUsesWith``
(``onnxoptimizer/pass.h``) unconditionally declines whenever *both* the
value being replaced and its replacement are graph inputs/outputs
(``areTwoValuesBothInputOrOutput``) -- an IR representational limit
unrelated to this pass's own axes-matching algebra, but enough to make the
pass silently no-op on the naive "X straight into Y straight out" model this
file's differential tests would otherwise reach for. Every differential test
below that expects a fuse therefore wraps X and Z in a plain ``Identity`` on
each side, exactly like other formal-verify files in this suite route
"the value under test" through a non-boundary node when the rewrite being
checked would otherwise touch the graph's own input/output values directly.
"""

import collections

from _formal_verify_common import isolate, producer, prove, simplify_isolated, z3
from onnx import parser

import onnxsim

_RANK0 = 4  # X's rank before Squeeze / after Unsqueeze -- matches the header
# comment's own worked example: X shape [2,1,3,1], axes=[1,3].
_AXES = [1, 3]


def _insert_removed(idx, sorted_removed_axes, rank):
    # Squeeze(T, axes=sorted_removed_axes)[idx] reads T at the index built by
    # inserting a 0 (the only valid coordinate on a removed, dim-1 axis) at
    # each removed position, and taking idx's components, in order, at every
    # other (kept) position. Reproduced from
    # test_formal_verify_fuse_consecutive_squeezes.py.
    it = iter(idx)
    return [0 if p in sorted_removed_axes else next(it) for p in range(rank)]


def _remove_inserted(idx, sorted_inserted_axes, rank):
    # Unsqueeze(T, axes=sorted_inserted_axes)[idx] reads T at the index built
    # by dropping idx's components at each inserted position (an inserted,
    # dim-1 axis only ever has index 0 there) and taking the rest, in order.
    # Reproduced from test_formal_verify_fuse_consecutive_unsqueezes.py.
    return [idx[p] for p in range(rank) if p not in sorted_inserted_axes]


def _ints_literal(xs):
    # ONNX text-format tensor literal, e.g. [1, 3] -> "{1, 3}" (as opposed to
    # the "[1, 3]" square-bracket syntax used for int-list *attributes*).
    return "{" + ", ".join(map(str, xs)) + "}"


def test_fuse_consecutive_squeeze_unsqueeze_round_trip_lemma():
    # The algebraic core both node orders rely on (see module docstring):
    # applying _insert_removed and then _remove_inserted *with the same axes
    # list* is an exact, unconditional round trip on the smaller
    # ("squeezed"/pre-unsqueeze) index space -- unlike the sibling files' own
    # two-DIFFERENT-axes-lists compositions, no shifting/remapping arithmetic
    # is needed at all, because insert always places a 0 at exactly the
    # positions remove then discards right back out.
    rank1 = _RANK0 - len(_AXES)
    idx = [z3.Int(f"i{i}") for i in range(rank1)]
    round_tripped = _remove_inserted(
        _insert_removed(idx, sorted(_AXES), _RANK0), sorted(_AXES), _RANK0
    )
    prove(z3.And(*(a == b for a, b in zip(round_tripped, idx))))


def test_fuse_consecutive_squeeze_unsqueeze_is_sound_squeeze_then_unsqueeze():
    # X (rank _RANK0) --Squeeze(axes)--> y (rank rank1) --Unsqueeze(axes)--> Z
    # (rank _RANK0 again). For idx into y, w = _insert_removed(idx, axes,
    # _RANK0) is the corresponding index into X *and* Z; unwinding each
    # node's own read semantics reduces Z[w] == X[w] to the round trip lemma.
    rank1 = _RANK0 - len(_AXES)
    tensor = z3.Function("tensor", *([z3.IntSort()] * _RANK0), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    idx = [z3.Int(f"i{i}") for i in range(rank1)]

    w = _insert_removed(idx, sorted(_AXES), _RANK0)  # X's (and Z's) own index
    # y[idx] = X[w] (Squeeze's read); Z[w] = y[_remove_inserted(w, axes, _RANK0)].
    composed = tensor(
        *_insert_removed(
            _remove_inserted(w, sorted(_AXES), _RANK0), sorted(_AXES), _RANK0
        )
    )
    direct = tensor(*w)
    prove(consumer(composed) == consumer(direct))


def test_fuse_consecutive_squeeze_unsqueeze_is_sound_unsqueeze_then_squeeze():
    # X (rank rank1) --Unsqueeze(axes)--> y (rank _RANK0) --Squeeze(axes)-->
    # Z (rank rank1 again, X's own rank here). For idx into X directly,
    # unwinding each node's own read semantics reduces Z[idx] == X[idx] to
    # the exact same round trip lemma, applied in the same insert-then-remove
    # order as the other node order above.
    rank1 = _RANK0 - len(_AXES)
    tensor = z3.Function("tensor", *([z3.IntSort()] * rank1), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    idx = [z3.Int(f"i{i}") for i in range(rank1)]  # index into X directly

    # y[w] = X[_remove_inserted(w, axes, _RANK0)] (Unsqueeze's read);
    # Z[idx] = y[_insert_removed(idx, axes, _RANK0)] (Squeeze's read).
    composed = tensor(
        *_remove_inserted(
            _insert_removed(idx, sorted(_AXES), _RANK0), sorted(_AXES), _RANK0
        )
    )
    direct = tensor(*idx)
    prove(consumer(composed) == consumer(direct))


def test_fuse_consecutive_squeeze_unsqueeze_pass_matches_squeeze_then_unsqueeze():
    # The header comment's own worked example: X shape [2,1,3,1] --Squeeze
    # (axes=[1,3])-> [2,3] --Unsqueeze(axes=[1,3])-> [2,1,3,1] == X. X and Z
    # are each routed through a plain Identity (see module docstring: without
    # it, both ends of the rewire are graph boundary values and
    # tryReplacingAllUsesWith declines for an unrelated IR reason). Confirmed
    # empirically: the pass destroys the *outer* node (Unsqueeze) and leaves
    # the *inner* one (Squeeze) dangling, rewiring Z0's producer straight to X.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,1,3,1] X0) => (float[2,1,3,1] Z0)
        <int64[{len(_AXES)}] axes = {_ints_literal(_AXES)}>
        {{
          X = Identity(X0)
          y = Squeeze(X, axes)
          Z = Unsqueeze(y, axes)
          Z0 = Identity(Z)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_squeeze_unsqueeze")
    # The outer node (Unsqueeze) is destroyed; the inner one (Squeeze) is left
    # dangling (its output "y" has no more uses) rather than also destroyed.
    assert ops["Unsqueeze"] == 0
    assert ops["Squeeze"] == 1
    fused_node = producer(sim_model, "Z0")
    assert fused_node.op_type == "Identity"
    assert fused_node.input[0] == "X", "should read straight from X, not through y/Z"


def test_fuse_consecutive_squeeze_unsqueeze_pass_matches_unsqueeze_then_squeeze():
    # The other order, also from the header comment: X shape [2,3]
    # --Unsqueeze(axes=[1,3])-> [2,1,3,1] --Squeeze(axes=[1,3])-> [2,3] == X.
    # Confirmed empirically: here the outer node (Squeeze) is destroyed and
    # the inner one (Unsqueeze) is left dangling -- the destroyed/dangling
    # roles swap with the node order, since "outer" always means the node
    # that matched the predicate (the consumer of the other).
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3] X0) => (float[2,3] Z0)
        <int64[{len(_AXES)}] axes = {_ints_literal(_AXES)}>
        {{
          X = Identity(X0)
          y = Unsqueeze(X, axes)
          Z = Squeeze(y, axes)
          Z0 = Identity(Z)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_squeeze_unsqueeze")
    assert ops["Squeeze"] == 0
    assert ops["Unsqueeze"] == 1
    fused_node = producer(sim_model, "Z0")
    assert fused_node.op_type == "Identity"
    assert fused_node.input[0] == "X", "should read straight from X, not through y/Z"


def test_fuse_consecutive_squeeze_unsqueeze_declines_axes_mismatch():
    # Squeeze removes axes [1, 3] but Unsqueeze re-inserts at [1, 2] instead:
    # axes != prev_axes, so runTransform bails before ever calling
    # tryReplacingAllUsesWith and both nodes survive, still chained.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,1,3,1] X) => (float[2,1,1,3] Z)
        <int64[2] axes_sq = {1, 3},
         int64[2] axes_un = {1, 2}>
        {
          y = Squeeze(X, axes_sq)
          Z = Unsqueeze(y, axes_un)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_squeeze_unsqueeze")
    assert ops["Squeeze"] == 1
    assert ops["Unsqueeze"] == 1
    fused_node = producer(sim_model, "Z")
    assert fused_node.op_type == "Unsqueeze"
    assert fused_node.input[0] == "y", "should decline and leave the two nodes chained"


def test_fuse_consecutive_squeeze_unsqueeze_declines_negative_axis_unknown_shape():
    # X's rank is genuinely unresolvable by shape inference: Base (known
    # shape [1,2,3,1]) is squeezed by sq_axes = Add(sq_a, sq_b), which is
    # deterministic ([0] at runtime) but not a compile-time constant --
    # FetchConstantTensor only recognizes a Constant node or an initializer
    # directly -- the same trick fuse_consecutive_unsqueezes' own
    # dynamic-shape test uses. ReferenceRank can recover neither the
    # Squeeze's input rank nor the Unsqueeze's output rank (both are X's
    # unresolvable rank), and axes=[-1] contains a negative entry, so
    # NormalizeAxes declines and both nodes stay chained (y -> Z). Bypasses
    # simplify_isolated (which only controls the *optimizer pass* list) and
    # calls onnxsim.simplify directly with skip_constant_folding=True, since
    # constant folding runs before the optimizer passes regardless of
    # skipped_optimizers and would otherwise fold the Add/Squeeze away first.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,2,3,1] Base) => (float[2,3,1] Z0)
        <int64[1] sq_a = {0}, int64[1] sq_b = {0},
         int64[1] axes = {-1}>
        {
          sq_axes = Add(sq_a, sq_b)
          X = Squeeze(Base, sq_axes)
          y = Squeeze(X, axes)
          Z = Unsqueeze(y, axes)
          Z0 = Identity(Z)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_consecutive_squeeze_unsqueeze"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Squeeze"] == 2
    assert ops["Unsqueeze"] == 1
    fused_node = producer(sim_model, "Z0")
    assert fused_node.op_type == "Identity"
    assert fused_node.input[0] == "Z", "should decline and leave the two nodes chained"


def test_fuse_consecutive_squeeze_unsqueeze_negative_axis_still_fuses_with_known_shape():
    # Control for the decline test above: the exact same negative axis (-1,
    # relative to rank 3) declines only because the rank is unresolvable --
    # with a statically known shape (X a plain graph input routed through
    # Identity, no Squeeze-by-non-constant-axes involved), it normalizes fine
    # and the pass still fuses.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,1] X0) => (float[2,3,1] Z0)
        <int64[1] axes = {-1}>
        {
          X = Identity(X0)
          y = Squeeze(X, axes)
          Z = Unsqueeze(y, axes)
          Z0 = Identity(Z)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_squeeze_unsqueeze")
    assert ops["Unsqueeze"] == 0
    assert ops["Squeeze"] == 1
    fused_node = producer(sim_model, "Z0")
    assert fused_node.op_type == "Identity"
    assert fused_node.input[0] == "X", "should fuse despite the negative axis"
