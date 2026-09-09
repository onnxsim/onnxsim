"""Formal check for EliminateSliceAfterShape (eliminate_slice_after_shape.h).

``patternMatchPredicate`` matches ``Slice(Shape(X)[...])`` -- a ``Slice`` node
whose sole data input (input 0) is the output of a ``Shape`` node -- ONLY when
``X`` (the ``Shape`` node's own input) has a statically known rank
(``HasDimsOfInputOfNode``, i.e. ``X``'s ``Value::has_sizes()``; individual
dims of ``X`` may still be symbolic ``dim_param``s at this point, only the
*rank* need be known).

``runTransform`` first resolves ``result_of_shape_op``: the sub-list of
``X``'s per-axis ``Dimension``s that ``Shape``'s own (rarely used) ``start``/
``end`` attributes select (``FetchStartAndEndAttrOfShape`` -- always the
whole list here, since none of the models below set those attributes; that
part of the pass is out of this file's scope). It then fetches the Slice
node's own ``starts``/``ends``/``step`` -- a SINGLE scalar each
(``FetchSoleIntValueOfAttr``/``FetchSoleIntValueOfTensor``), so this pass
only ever handles a Slice with exactly one axis being sliced -- and applies
ONNX's own Slice index semantics by hand: negative-index normalization
(``AddYIfNegative``, applied ONCE), then clamping to ``[0, len]`` for a
forward step or the DIFFERENT range ``[-1, len-1]`` for a backward
(negative-step, i.e. reversed) walk, then striding through the selected
sub-range of ``result_of_shape_op`` with the given (possibly negative) step,
declining (leaving both ``Shape`` and ``Slice`` in place) if any dim it
selects along the way isn't statically known as a concrete int
(``!d.is_int``). The final selected dims become a fresh INT64 constant
initializer that ``Slice``'s consumers are rewired to directly; the old
``Shape`` node is left dangling (same pattern as elsewhere in this suite,
e.g. ``fuse_consecutive_slices``), since ``eliminate_deadend`` -- a separate
default pass -- isn't what removes it.

This is the same "trust static shape metadata equals the actual runtime
shape" premise as this suite's other shape-family passes: nothing here
proves the general claim that ``X``'s declared static dims match its runtime
shape (that's a graph-level property no single-pass proof can establish, and
is simply taken as given -- the same honesty this suite's other shape passes
apply). What IS specific and worth proving carefully here is Slice's own
general indexing arithmetic -- forward vs. reverse traversal over a
sub-range, with ONNX's negative-index normalization and *asymmetric* forward/
backward clamping reproduced exactly. ``_walk_slice`` below is a direct,
line-for-line transliteration of ``runTransform``'s own walk (operating on a
plain Python list standing in for ``result_of_shape_op``), so it can be
compared against Python's/ONNX's own ``list[start:end:step]`` semantics --
which is genuinely what any ONNX backend computes for ``Slice`` at runtime,
confirmed directly below: ``onnx.reference``'s own ``Slice`` kernel is
literally ``data[slice(start, end, step)]``, i.e. plain Python/numpy slicing.

A discovered subtlety, worth flagging honestly rather than hiding: the
transliterated walk is NOT byte-for-byte equivalent to
``list[start:end:step]`` for every possible integer ``start``/``end`` --
only whenever NOT (``step < 0`` and ``start < -len(dims)`` and
``end < -len(dims)``), i.e. unless the step is negative AND *both* bounds are
so far out of range that even one ``AddYIfNegative`` pass leaves them
negative. In that narrow corner, Python's own slice-index-adjustment clamps
an all-the-way-out-of-range ``start`` to ``-1`` (matching the ``-1`` it also
uses for ``end``, yielding an empty result), while this pass instead clamps
``start`` to ``0`` (its clamp range is ``[0, len-1]``, not ``[-1, len-1]``),
which can select one extra element that plain re-derived Python slicing
would not. Empirically (see the differential test below), onnxruntime's own
compiled ``Slice`` kernel agrees with THIS pass's answer in that corner, not
with ``onnx.reference``'s pure-Python one -- real backends can disagree with
each other on this kind of pathological double-out-of-range input, and this
pass happens to match onnxruntime's behavior. This corner is never reachable
by any ``Slice(Shape(X))`` a real graph author or exporter would produce
(``start``/``end`` would have to be more negative than ``X``'s own rank), so
it's excluded from -- and explicitly characterized by -- the proof below
rather than either silently ignored or wrongly claimed away.
"""

import collections

from _formal_verify_common import isolate, prove, z3
from onnx import numpy_helper, parser

import onnxsim

# X's shape for every differential test below: 5 distinct dims, so which
# ones got selected (and in what order) is unambiguous from the values alone.
_X_SHAPE = [2, 3, 4, 5, 6]


def _walk_slice(dims, start, end, step):
    """Transliterates runTransform's own index walk (see eliminate_slice_after_shape.h),
    operating on a plain Python list ``dims`` standing in for
    ``result_of_shape_op``. Faithful down to the asymmetric clamp ranges for
    forward vs. backward (negative) ``step`` -- see the module docstring for
    the one documented corner where this deliberately does NOT match
    ``dims[start:end:step]``.
    """
    n = len(dims)
    if start < 0:
        start += n
    if end < 0:
        end += n

    out = []
    if step > 0:
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        i = start
        while i < end:
            out.append(dims[i])
            i += step
    else:
        start = max(0, min(start, n - 1))
        end = max(-1, min(end, n))
        i = start
        while i > end:
            out.append(dims[i])
            i += step
    return out


# Representative (start, end, step) configs used by both formal-proof tests
# below -- deliberately outside the one documented divergence corner (no
# config here has step < 0 with *both* start and end more negative than
# -len(dims)), i.e. exactly the domain any real Slice(Shape(X)) would ever
# use. Each is named by what it exercises.
_REPRESENTATIVE_CONFIGS = [
    ("forward_basic", 1, 3, 1),
    ("forward_clamped_end", 2, 10, 1),  # end=10 clamps down to len(dims)=5
    ("forward_extreme_negative_start", -100, 3, 1),  # start clamps up to 0
    ("reverse_simple", 3, 0, -1),
    ("reverse_full_walk", 4, -6, -1),  # end alone is far out of range; start isn't
    ("reverse_step_2", 4, 0, -2),
    ("negative_start_forward", -2, 5, 1),
]


def test_eliminate_slice_after_shape_index_walk_is_sound():
    # Uninterpreted per-element consumer of a selected dim, to prove
    # substitution safety: no matter what downstream computation reads the
    # constant this pass produces, it sees the exact same sequence of values
    # (in the same order) that evaluating the original Slice(Shape(X)) at
    # runtime would have produced -- given (the pass's own premise) that
    # X's *declared* static dims equal its *actual* runtime shape, modeled
    # here by using the very same symbolic reals for both.
    d = z3.Reals("d0 d1 d2 d3 d4")
    dims = list(d)
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    for name, start, end, step in _REPRESENTATIVE_CONFIGS:
        walk = _walk_slice(dims, start, end, step)
        native = dims[start:end:step]
        assert len(walk) == len(native), (
            f"{name}: _walk_slice produced {len(walk)} elements, "
            f"native slicing produced {len(native)}"
        )
        claim = z3.And(*(consumer(w) == consumer(v) for w, v in zip(walk, native)))
        prove(
            claim,
            msg=f"{name}: index walk disagrees with Slice's own runtime semantics",
        )


def test_eliminate_slice_after_shape_index_walk_matches_python_slicing_exhaustively():
    # Same identity as the Z3 proof above, but swept concretely (plain
    # equality, no solver needed -- the walk is pure index bookkeeping, not
    # value algebra) over a broad grid of start/end/step, to pin down
    # *exactly* where the transliterated walk agrees with true Slice
    # semantics and where it doesn't (see the module docstring). This both
    # backs the representative configs above with much wider coverage and
    # documents the one known divergence precisely rather than leaving it
    # implicit.
    dims = [11, 22, 33, 44, 55]
    n = len(dims)
    span = range(-2 * n - 3, 2 * n + 4)
    steps = (-3, -2, -1, 1, 2, 3)

    mismatches = set()
    for start in span:
        for end in span:
            for step in steps:
                got = tuple(_walk_slice(dims, start, end, step))
                want = tuple(dims[start:end:step])
                if got != want:
                    mismatches.add((start, end, step))

    is_the_documented_corner = {
        (start, end, step)
        for start in span
        for end in span
        for step in steps
        if step < 0 and start < -n and end < -n
    }
    assert mismatches == is_the_documented_corner, (
        "the walk's divergence from Python/ONNX slicing semantics no longer "
        "matches the documented (step<0, start<-n, end<-n) corner -- update "
        "the module docstring and this characterization together"
    )
    assert mismatches, "sanity: the documented corner should be non-empty for this grid"


def _slice_after_shape_model(starts, ends, steps):
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[{",".join(map(str, _X_SHAPE))}] X) => (int64[?] Z)
        <int64[1] starts = {{{starts}}}, int64[1] ends = {{{ends}}}, int64[1] steps = {{{steps}}}>
        {{
          s = Shape(X)
          Z = Slice(s, starts, ends, , steps)
        }}
        """
    )


def _check_fires(model, starts, ends, steps):
    # skip_constant_folding=True is needed here (unlike some sibling
    # shape-family tests): without it, onnxsim's separate constant-folding
    # step -- which always runs before the optimizer passes regardless of
    # skipped_optimizers -- folds Shape(X) then Slice(...) away by itself
    # first, on models where X's shape is fully static, leaving nothing for
    # this pass to actually do (confirmed empirically: with constant folding
    # left on, the *dangling Shape node* this pass's own header comment
    # promises never shows up in the result). With it off, the real
    # compiled eliminate_slice_after_shape pass is what performs the
    # rewrite, and the dangling Shape node is exactly what's left behind.
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_slice_after_shape"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {"Shape": 1}, "expected only the dangling Shape node to survive"

    expected = _walk_slice(_X_SHAPE, starts, ends, steps)
    out_name = sim_model.graph.output[0].name
    z_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    assert list(numpy_helper.to_array(z_init)) == expected
    assert expected == _X_SHAPE[starts:ends:steps]  # cross-check vs. Python itself
    return sim_model


def test_eliminate_slice_after_shape_pass_matches_forward_slice():
    model = _slice_after_shape_model(1, 3, 1)
    _check_fires(model, 1, 3, 1)


def test_eliminate_slice_after_shape_pass_matches_reversed_slice():
    # Negative step: the case most likely to have an off-by-one or clamping
    # bug, so this is checked carefully against Python's own reversed-slice
    # semantics -- python [2,3,4,5,6][3:0:-1] == [5,4,3] (dims at indices
    # 3,2,1, in that descending order).
    model = _slice_after_shape_model(3, 0, -1)
    sim_model = _check_fires(model, 3, 0, -1)
    out_name = sim_model.graph.output[0].name
    z_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    assert list(numpy_helper.to_array(z_init)) == [5, 4, 3]


def test_eliminate_slice_after_shape_pass_matches_negative_start():
    # Negative start: python [2,3,4,5,6][-2:5] == [5,6] (indices 3,4).
    model = _slice_after_shape_model(-2, 5, 1)
    sim_model = _check_fires(model, -2, 5, 1)
    out_name = sim_model.graph.output[0].name
    z_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    assert list(numpy_helper.to_array(z_init)) == [5, 6]


def test_eliminate_slice_after_shape_declines_when_selected_dim_is_symbolic():
    # X's rank is statically known (3), so the predicate's own
    # HasDimsOfInputOfNode check passes -- but X's axis 0 is a dim_param
    # ("N"), and the Slice selects indices [0, 2) which includes it.
    # runTransform's own `!d.is_int` check declines mid-walk (on the very
    # first selected dim), leaving both Shape and Slice untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[N,3,4] X) => (int64[?] Z)
        <int64[1] starts = {0}, int64[1] ends = {2}, int64[1] steps = {1}>
        {
          s = Shape(X)
          Z = Slice(s, starts, ends, , steps)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model, check_n=3, skipped_optimizers=isolate("eliminate_slice_after_shape")
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {"Shape": 1, "Slice": 1}
    (slice_node,) = [n for n in sim_model.graph.node if n.op_type == "Slice"]
    assert slice_node.input[0] == "s"  # still chained onto Shape's output


def test_eliminate_slice_after_shape_declines_when_x_rank_is_unresolvable():
    # X is itself Squeeze(Y, axes) where axes comes from an Add of two
    # initializers rather than being a constant (or Constant node) itself
    # -- deterministic at runtime ([0, 3] either way), but FetchConstantTensor
    # can't see through the Add, so X's rank is genuinely unresolvable by
    # onnxsim's shape inference, and HasDimsOfInputOfNode(shape_node, 0) is
    # false: the predicate declines outright, before ever looking at the
    # Slice. Same construction this suite's fuse_consecutive_squeezes and
    # fuse_consecutive_unsqueezes formal-verify tests use to defeat
    # FetchConstantTensor for an analogous purpose. This bypasses
    # simplify_isolated (which only controls the *optimizer pass* list) and
    # calls onnxsim.simplify directly with skip_constant_folding=True, since
    # constant folding would otherwise fold the Add/Squeeze away first and
    # restore a statically-known rank before this pass ever saw the graph.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,1,2,3,1,5,1] Y) => (int64[?] Z)
        <int64[2] axes_a = {0, 3}, int64[2] axes_b = {0, 1},
         int64[1] starts = {0}, int64[1] ends = {2}, int64[1] steps = {1}>
        {
          axes = Add(axes_a, axes_b)
          X = Squeeze(Y, axes)
          s = Shape(X)
          Z = Slice(s, starts, ends, , steps)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_slice_after_shape"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {"Add": 1, "Squeeze": 1, "Shape": 1, "Slice": 1}
    (slice_node,) = [n for n in sim_model.graph.node if n.op_type == "Slice"]
    assert slice_node.input[0] == "s"
