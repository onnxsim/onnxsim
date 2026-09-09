"""Formal check for FuseConsecutiveUnsqueezes (fuse_consecutive_unsqueezes.h).

onnxsim registers its own copy of this pass
(``onnxsim/passes/fuse_consecutive_unsqueezes.h``, wrapped in
``namespace onnxsim_passes``) via ``RegisterOrReplace``, overwriting the
upstream onnx-optimizer entry of the same name
(``third_party/onnx-optimizer/onnxoptimizer/passes/fuse_consecutive_unsqueezes.h``)
in the pass registry that ``getPassName()`` keys into -- so onnxsim's version
is the one that actually runs (same override relationship as
``fuse_add_bias_into_conv``, see that pass's own formal-verify file). The two
implementations differ in exactly one place: upstream's
``patternMatchPredicate`` additionally requires
``GetInputsOfPreNode(node, 0)[0]->has_sizes()`` -- the *inner* Unsqueeze's
input must have a statically known rank -- before it will even attempt the
fusion. onnxsim's version drops that requirement from the predicate and
instead only bails, deep inside ``runTransform``, when the rank is unknown
**and** either axes list actually contains a negative value:

    if (!prev->input(0)->has_sizes()) {
      for (int64_t a : axes_of_prev) { if (a < 0) return false; }
      for (int64_t a : axes) { if (a < 0) return false; }
    }
    const auto dims = prev->input(0)->sizes();   // may be empty/meaningless
    ... AddYIfNegative(axis, dims.size() + ...) ...   // now a no-op either way

This matters for dynamic-shape graphs -- the header comment gives detection
models feeding ``NonMaxSuppression`` through ``Unsqueeze(Unsqueeze(x))``
chains whose input has no static shape as the motivating case. When every
axis is already non-negative, ``AddYIfNegative`` is a no-op regardless of
what ``dims.size()`` actually is, so the fusion arithmetic below doesn't
care that the rank is unknown; a negative axis, on the other hand, can only
be normalized to a positive one by adding a rank we don't have, so onnxsim's
version has no choice but to decline exactly like upstream would.

Once both axes lists are fetched (via ``GetValueFromAttrOrInput``, so each
must be a compile-time constant -- an attribute for opset<=12, or a
Constant/initializer input for opset 13+) and normalized, the fused axes
list is built by (with ``axes_of_prev`` = the *inner* Unsqueeze's axes,
``axes`` = the *outer* one's, both already sorted):

    for (auto& n : axes_of_prev) {     // n ranges over inner axes, by ref
      for (const auto& m : axes) {     // m ranges over outer axes
        if (m <= n) { n++; }
      }
    }
    fused_axes = sort(axes_of_prev + axes)   // axes_of_prev now shifted

reproduced verbatim (as a direct transliteration -- note ``n`` mutates
*during* the inner loop, so a later ``m`` is compared against the
already-shifted ``n``, not the original value; getting this sequential
mutation right, not just "count how many outer axes are <= the original
inner axis", is exactly the kind of index-arithmetic bug this file exists to
catch) in ``_compose_unsqueezes`` below. This is the *opposite* composition
direction from ``fuse_consecutive_squeezes``
(test_formal_verify_fuse_consecutive_squeezes.py, this file's closest
template): Squeeze *removes* axes, so composing two squeezes shifts the
outer squeeze's (already-once-reduced-rank-relative) axes *down* to account
for positions the inner squeeze already removed; Unsqueeze *inserts* new
(dim-1) axes, so here it is the *inner* Unsqueeze's axes that must shift
*up*, by however many of the outer Unsqueeze's insertions land at or before
each one -- the outer unsqueeze's insertions push everything at or after
them one position to the right.

Soundness is proven the same way as the squeeze file, with the inverse
index-transformation: squeeze's ``_insert_removed`` reconstructs a
pre-squeeze index by *inserting* a 0 at each removed axis; unsqueeze's
``_remove_inserted`` below does the reverse -- given an index into the
*twice-unsqueezed* (larger-rank) output, it *drops* the components at the
inserted-axis positions (an Unsqueeze's output is always index 0 there, by
construction) to recover the index into the smaller-rank input. Applying it
twice (once per Unsqueeze, composed) versus once with the fused axes list
must read the exact same value at every index into the *final*
(twice-unsqueezed, largest-rank) tensor -- a genuine universal proof (free
variables in a Z3 validity check are implicitly universally quantified), not
a finite sample. As with the squeeze proof, the two axes lists are concrete
(chosen to interact non-trivially -- the inner axis 3 and outer axis 2 both
shift because of overlaps with the other list) rather than fully symbolic,
so the index arithmetic stays tractable for Z3.

Two more things confirmed empirically against the actual compiled pass
(rather than assumed from reading the header) before writing the
differential tests below:

* Unlike ``fuse_consecutive_squeezes`` -- which explicitly calls
  ``orig_input->node()->destroy()`` once the inner Squeeze's output has no
  more uses -- neither this pass nor its upstream counterpart ever destroys
  the now-dead inner Unsqueeze node; both only set
  ``destroy_current = NodeDestroyType::DestroyZero`` (meaning "don't even
  destroy the outer node being rewritten in place") and rewire the *outer*
  node's input straight to the original tensor. So after a successful fuse,
  the graph still contains **two** Unsqueeze nodes -- the original inner one
  (now dead: nothing reads its output) and the rewritten outer one (now
  reading straight from the original input, with the fused axes). Isolating
  this pass via ``skipped_optimizers`` also skips ``eliminate_deadend`` (a
  default pass), so that dead node is never swept away either -- the tests
  below use ``producer()`` (see ``_formal_verify_common.py``) to walk back
  from the live graph output rather than assuming a node-type count of 1.
* Both the pre-13 attribute form (``Unsqueeze<axes=[...]>(X)``) and the
  opset-13+ input form (``Unsqueeze(X, axes)``) parse through
  ``onnx.parser`` and drive the real pass identically (unlike some other
  ops, Unsqueeze's opset-13 signature turned out not to need any special
  handling here); the basic differential test below uses the input form for
  consistency with the squeeze file's own choice.

The dynamic-shape relaxation test needs a value whose rank is genuinely
unresolvable by onnxsim's shape inference (``has_sizes()`` false), not just
one with an unknown *value*. A plain unranked graph input doesn't work --
ONNX's own checker requires every *main-graph* input/output to declare a
``shape`` field (even an all-``dim_param`` one), so ``ClearField("shape")``
on a graph input fails ``onnx.checker`` inside ``onnxsim.simplify`` with
"Field 'shape' ... is required but missing." (confirmed empirically). The
construction used instead: ``Squeeze`` a tensor of otherwise-known shape by
an axes value that is deterministic but not a compile-time constant --
``Add`` of two initializers, the same trick
``fuse_consecutive_squeezes``'s own ``declines_non_constant_axes`` test uses
to defeat ``FetchConstantTensor`` -- which leaves the squeezed output's rank
genuinely unresolvable (removing an unknown-statically-but-fixed-at-runtime
number of axes) even though its runtime value never varies. Like that
squeeze test, this bypasses ``simplify_isolated`` (which only controls the
*optimizer pass* list) and calls ``onnxsim.simplify`` directly with
``skip_constant_folding=True``, since onnxsim's constant folding is a
separate step that runs before the optimizer passes regardless of
``skipped_optimizers`` and would otherwise fold the ``Add``/``Squeeze`` away
-- restoring a statically-known shape -- before ``fuse_consecutive_unsqueezes``
ever saw it. A control test (static shape, same negative axis) confirms the
decline is really caused by the unknown rank and not some other artifact --
with a known shape, the same negative axis normalizes fine and the pass
still fuses.
"""

import collections

from _formal_verify_common import isolate, producer, prove, simplify_isolated, z3
from onnx import numpy_helper, parser

import onnxsim

_RANK0 = 3  # X's rank, before either unsqueeze
_AXES_OF_PREV = [1, 3]  # inner Unsqueeze's axes (relative to the once-unsqueezed rank)
_AXES = [0, 2]  # outer Unsqueeze's axes (relative to the twice-unsqueezed rank)


def _compose_unsqueezes(axes_of_prev, axes):
    # Mirrors fuse_consecutive_unsqueezes.h's own fused-axes loop exactly
    # (see the module docstring for the annotated original), including the
    # sequential mutation of each inner axis as later outer axes are checked
    # against its already-shifted value.
    axes_of_prev = sorted(axes_of_prev)
    axes_sorted = sorted(axes)
    shifted = []
    for n in axes_of_prev:
        for m in axes_sorted:
            if m <= n:
                n += 1
        shifted.append(n)
    return sorted(shifted + list(axes_sorted))


def _remove_inserted(idx, sorted_inserted_axes, rank):
    # Unsqueeze(T, axes=sorted_inserted_axes)[idx] reads T at the index built
    # by dropping idx's components at each inserted position (an inserted,
    # dim-1 axis only ever has index 0 there) and taking the rest, in order
    # -- the inverse of squeeze's own _insert_removed.
    return [idx[p] for p in range(rank) if p not in sorted_inserted_axes]


def _ints_literal(xs):
    # ONNX text-format tensor literal, e.g. [1, 3] -> "{1, 3}" (as opposed to
    # the "[1, 3]" square-bracket syntax used for int-list *attributes*).
    return "{" + ", ".join(map(str, xs)) + "}"


def test_fuse_consecutive_unsqueezes_is_sound():
    composed = _compose_unsqueezes(_AXES_OF_PREV, _AXES)
    rank1 = _RANK0 + len(_AXES_OF_PREV)  # once-unsqueezed tensor's rank
    rank2 = rank1 + len(_AXES)  # twice-unsqueezed (final) tensor's rank
    assert rank2 == _RANK0 + len(composed)

    tensor = z3.Function("tensor", *([z3.IntSort()] * _RANK0), z3.RealSort())
    z = [z3.Int(f"z{i}") for i in range(rank2)]

    two_step = tensor(
        *_remove_inserted(
            _remove_inserted(z, sorted(_AXES), rank2), sorted(_AXES_OF_PREV), rank1
        )
    )
    fused = tensor(*_remove_inserted(z, composed, rank2))
    prove(two_step == fused)


def test_fuse_consecutive_unsqueezes_pass_matches():
    # X: [2,4,5] --Unsqueeze(axes=[1,3])-> [2,1,4,1,5]
    #      --Unsqueeze(axes=[0,2])-> [1,2,1,1,4,1,5],
    # fusing to a single Unsqueeze(X, axes=_compose_unsqueezes(...)) == [0,2,3,5].
    composed = _compose_unsqueezes(_AXES_OF_PREV, _AXES)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,4,5] X) => (float[1,2,1,1,4,1,5] Z)
        <int64[{len(_AXES_OF_PREV)}] axes1 = {_ints_literal(_AXES_OF_PREV)},
         int64[{len(_AXES)}] axes2 = {_ints_literal(_AXES)}>
        {{
          y = Unsqueeze(X, axes1)
          Z = Unsqueeze(y, axes2)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_unsqueezes")
    # Both Unsqueeze nodes are still present (see module docstring: this
    # pass, unlike fuse_consecutive_squeezes, never destroys the now-dead
    # inner node) -- walk back from the live output instead of counting.
    assert ops["Unsqueeze"] == 2
    fused_node = producer(sim_model, "Z")
    assert fused_node.op_type == "Unsqueeze"
    assert fused_node.input[0] == "X", "should read straight from X, not the dead 'y'"
    axes_init = next(
        init for init in sim_model.graph.initializer if init.name == fused_node.input[1]
    )
    assert list(numpy_helper.to_array(axes_init)) == composed


def _dynamic_shape_model(axes_of_prev, axes):
    # Base: known shape [1,2,4,5,1]. sq_axes = Add(sq_a, sq_b) is always
    # [0, 4] at runtime (removing the two size-1 axes, leaving [2,4,5], the
    # same shape as the static test's X) but is not a compile-time constant
    # -- FetchConstantTensor only recognizes a Constant node or an
    # initializer directly -- so X's rank is genuinely unresolvable by shape
    # inference, even though its value never varies. See module docstring.
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,2,4,5,1] Base) => (float[1,2,1,1,4,1,5] Z)
        <int64[2] sq_a = {{0, 4}}, int64[2] sq_b = {{0, 0}},
         int64[{len(axes_of_prev)}] axes1 = {_ints_literal(axes_of_prev)},
         int64[{len(axes)}] axes2 = {_ints_literal(axes)}>
        {{
          sq_axes = Add(sq_a, sq_b)
          X = Squeeze(Base, sq_axes)
          y = Unsqueeze(X, axes1)
          Z = Unsqueeze(y, axes2)
        }}
        """
    )


def test_fuse_consecutive_unsqueezes_relaxes_unknown_shape():
    # The onnxsim-specific relaxation: X's rank is unresolvable (see
    # _dynamic_shape_model), but every axis in both lists is already
    # non-negative, so onnxsim's version still fuses -- unlike upstream,
    # whose patternMatchPredicate requires has_sizes() up front and would
    # never even attempt this rewrite.
    composed = _compose_unsqueezes(_AXES_OF_PREV, _AXES)
    model = _dynamic_shape_model(_AXES_OF_PREV, _AXES)
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_consecutive_unsqueezes"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Unsqueeze"] == 2  # the dead inner node is left in place, as above
    fused_node = producer(sim_model, "Z")
    assert fused_node.op_type == "Unsqueeze"
    assert fused_node.input[0] == "X", "should fuse despite X's unresolvable rank"
    axes_init = next(
        init for init in sim_model.graph.initializer if init.name == fused_node.input[1]
    )
    assert list(numpy_helper.to_array(axes_init)) == composed


def test_fuse_consecutive_unsqueezes_declines_negative_axis_unknown_shape():
    # Same unknown-rank X, but axes_of_prev now contains a negative entry
    # (-1, i.e. the same last axis as the relaxation test's rank-1 tensor,
    # spelled negatively). Normalizing it needs prev's input rank, which is
    # unavailable, so onnxsim's version bails too -- both Unsqueeze nodes
    # stay in place, still chained (y -> Z), exactly like the upstream
    # predicate would have declined to match at all.
    model = _dynamic_shape_model([1, -1], _AXES)
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_consecutive_unsqueezes"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops["Unsqueeze"] == 2
    assert ops["Squeeze"] == 1
    assert ops["Add"] == 1
    fused_node = producer(sim_model, "Z")
    assert fused_node.op_type == "Unsqueeze"
    assert fused_node.input[0] == "y", "should decline and leave the two nodes chained"


def test_fuse_consecutive_unsqueezes_negative_axis_still_fuses_with_known_shape():
    # Control for the two dynamic-shape tests above: the exact same negative
    # axis (-1, relative to rank1 = _RANK0 + len(_AXES_OF_PREV) = 5, i.e.
    # axis 3) declines only because the rank is unresolvable -- with a
    # statically known shape (X a plain graph input, no Squeeze-by-
    # non-constant-axes involved), it normalizes to the same axis 3 used
    # throughout this file and the pass still fuses, producing the same
    # composed axes as the fully-non-negative basic test.
    composed = _compose_unsqueezes(_AXES_OF_PREV, _AXES)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,4,5] X) => (float[1,2,1,1,4,1,5] Z)
        <int64[2] axes1 = {{1, -2}},
         int64[{len(_AXES)}] axes2 = {_ints_literal(_AXES)}>
        {{
          y = Unsqueeze(X, axes1)
          Z = Unsqueeze(y, axes2)
        }}
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_unsqueezes")
    assert ops["Unsqueeze"] == 2
    fused_node = producer(sim_model, "Z")
    assert fused_node.op_type == "Unsqueeze"
    assert fused_node.input[0] == "X"
    axes_init = next(
        init for init in sim_model.graph.initializer if init.name == fused_node.input[1]
    )
    assert list(numpy_helper.to_array(axes_init)) == composed
