"""Formal check for FuseConsecutiveSlices (fuse_consecutive_slices.h).

``patternMatchPredicate`` matches ``Slice1(Slice2(X))`` -- an outer Slice
node whose sole producer on input 0 is another Slice -- only when ALL of the
following hold:

* both Slice nodes have exactly 5 inputs (the opset-10+ full form: data,
  starts, ends, axes, steps, with ``steps`` required at index 4). This means
  the pass never fires on the pre-opset-10 3-input form (data, starts, ends)
  or the axes-omitted 4-input form (data, starts, ends, axes) -- both are
  exercised below as declining cases.
* both slices' ``axes`` (input index 3) are fetchable as compile-time
  constants via ``GetValueFromInput`` -- **not** ``GetValueFromAttrOrInput``,
  since Slice's ``axes`` was never an attribute, only ever an input.
* the outer Slice's data input (i.e. the inner Slice's output) has a
  statically known shape (``has_sizes()``), needed to normalize negative
  axes via ``AddYIfNegative``.
* after that normalization, the inner and outer slices' ``axes`` lists have
  **no intersection** (``!HasIntersection(...)``) -- the two slices must
  operate on entirely disjoint sets of axes.

``runTransform`` does *not* statically concatenate the starts/ends/axes/steps
as Python-style lists. It builds four new graph-level ``Concat`` nodes (one
each for starts, ends, axes, steps), each concatenating the inner slice's
input ``i`` with the outer slice's input ``i`` (in that order: inner first,
outer second) along axis 0. It then creates one new ``Slice`` node reading
the ORIGINAL, pre-either-slice data (``slice2->input(0)``, i.e. the inner
Slice's own data input) and the four Concat outputs as its starts/ends/axes/
steps, and rewires the old outer Slice's output to the new Slice's output.
``destroy_current = NodeDestroyType::DestroyOne`` destroys only the outer
Slice node (``n``, aka ``slice1``) -- confirmed empirically below
(``test_fuse_consecutive_slices_pass_matches``) -- so the inner Slice node
(``slice2``) is left in the graph, its output now unused: a dangling node,
not cleaned up by this pass itself (that's ``eliminate_deadend``'s job, a
separate default pass). This mirrors the same dangling-producer subtlety
already found in this repo's other fusion-pass formal-verify tests (e.g.
``fuse_pad_into_conv``, ``fuse_matmul_add_bias_into_gemm``).

Formal content: two Slice operations along pairwise-disjoint axis sets
commute and combine into a single Slice restricting each axis set
independently and simultaneously -- slicing one axis never affects indices
along a different axis, so "restrict axis set A" then "restrict axis set B"
(A, B disjoint) is exactly the same as restricting A and B in one combined
operation. This is modeled concretely with a rank-2 tensor (the minimum
needed to exercise two disjoint single-axis slices, matching how e.g.
``test_formal_verify_fuse_consecutive_squeezes.py`` and
``test_formal_verify_fuse_consecutive_concats.py`` use small concrete
examples rather than fully general N-D symbolic shapes): the inner Slice
restricts axis 0 to ``[start0, end0)``, the outer Slice restricts axis 1 to
``[start1, end1)``, both with ``step=1``. The tensor is an uninterpreted Z3
function ``Int, Int -> Real``; Slice's own index-shift semantics is modeled
as ``sliced(i) = tensor(i + start)`` on the sliced axis and left unchanged
on the other axis. This scoping is honest about what's *not* proven here:
generalizing to strided slices (``step != 1``) is the same argument with an
extra ``step`` multiplication folded into the index map, and generalizing
beyond two axes/rank 2 is the same disjoint-axes argument applied
independently per axis -- neither is re-derived here.
"""

import numpy as np
from _formal_verify_common import isolate, producer, prove, simplify_isolated, z3
from onnx import numpy_helper, parser

import onnxsim


def test_fuse_consecutive_slices_is_sound():
    # Uninterpreted rank-2 tensor and an uninterpreted consumer of sliced
    # values, to prove substitution safety (the rewrite is sound no matter
    # what downstream computation reads the sliced result).
    T = z3.Function("T", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")
    start0, start1 = z3.Ints("start0 start1")  # inner axis-0 / outer axis-1 starts

    def slice2_out(i, j):
        # Inner Slice2: restricts axis 0 to start at start0 (step=1), axis 1
        # untouched.
        return T(i + start0, j)

    def slice1_out(i, j):
        # Outer Slice1, applied on top of Slice2's output: restricts axis 1
        # to start at start1 (step=1), axis 0 untouched (already sliced).
        return slice2_out(i, j + start1)

    def fused_out(i, j):
        # What runTransform's single new Slice computes: both axes'
        # constraints applied at once, directly against the original tensor
        # T (the new Slice reads slice2->input(0), not either old output).
        return T(i + start0, j + start1)

    prove(consumer(slice1_out(i, j)) == consumer(fused_out(i, j)))


def _slice_pair_model(inner_axis, outer_axis):
    # X is 2-D so both axis-0 and axis-1 slices are always in-bounds
    # regardless of which axis each Slice restricts.
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[10,10] X) => (float[?,?] Z)
        <int64[1] s2_starts = {{1}}, int64[1] s2_ends = {{5}},
         int64[1] s2_axes = {{{inner_axis}}}, int64[1] s2_steps = {{1}},
         int64[1] s1_starts = {{2}}, int64[1] s1_ends = {{8}},
         int64[1] s1_axes = {{{outer_axis}}}, int64[1] s1_steps = {{1}}>
        {{
          y = Slice(X, s2_starts, s2_ends, s2_axes, s2_steps)
          Z = Slice(y, s1_starts, s1_ends, s1_axes, s1_steps)
        }}
        """
    )


def _check_fused(sim_model, expected_starts, expected_ends, expected_axes):
    # The new Slice must read directly from the *original* data X, not from
    # either old Slice's output -- confirmed via producer() walking back from
    # the graph output.
    out_name = sim_model.graph.output[0].name
    fused = producer(sim_model, out_name)
    assert fused.op_type == "Slice"
    assert fused.input[0] == "X"

    initializers = {
        i.name: numpy_helper.to_array(i) for i in sim_model.graph.initializer
    }
    nodes_by_output = {out: n for n in sim_model.graph.node for out in n.output}

    def resolve(value_name):
        if value_name in initializers:
            return initializers[value_name]
        # Constant-folded away by onnxsim's separate constant-folding step;
        # re-derive by walking the Concat node's own two inputs.
        concat = nodes_by_output[value_name]
        assert concat.op_type == "Concat"
        inner, outer = concat.input
        return np.concatenate([initializers[inner], initializers[outer]])

    starts, ends, axes, steps = (resolve(v) for v in fused.input[1:5])
    assert list(starts) == expected_starts
    assert list(ends) == expected_ends
    assert list(axes) == expected_axes
    assert list(steps) == [1, 1]


def test_fuse_consecutive_slices_pass_matches():
    # Inner Slice2 restricts axis 1, outer Slice1 restricts axis 0: disjoint
    # axes, so runTransform fires. The inner Slice2 node is left dangling in
    # the graph (only the outer Slice1 is destroyed) since eliminate_deadend
    # is skipped by isolate() here.
    model = _slice_pair_model(inner_axis=1, outer_axis=0)
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_slices")
    assert ops["Slice"] == 2  # the new fused Slice + the dangling old inner Slice
    assert ops["Concat"] == 0  # folded away by onnxsim's separate constant folding

    dangling = [n for n in sim_model.graph.node if n.output[0] == "y"]
    assert len(dangling) == 1 and dangling[0].op_type == "Slice"

    _check_fused(sim_model, [1, 2], [5, 8], [1, 0])


def test_fuse_consecutive_slices_pass_matches_reversed_axes():
    # Same as above but with the axes swapped: inner Slice2 restricts axis
    # 0, outer Slice1 restricts axis 1 -- still disjoint, still fuses, and
    # the Concat order (inner value first, then outer value) is confirmed
    # independent of which physical axis each side owns.
    model = _slice_pair_model(inner_axis=0, outer_axis=1)
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_slices")
    assert ops["Slice"] == 2

    _check_fused(sim_model, [1, 2], [5, 8], [0, 1])


def test_fuse_consecutive_slices_pass_matches_concat_node_structure():
    # Same disjoint-axes model, but with skip_constant_folding=True (bypassing
    # simplify_isolated, which doesn't expose that knob) so the four Concat
    # nodes runTransform actually builds survive to be inspected directly,
    # rather than being folded into initializers by onnxsim's separate
    # constant-folding step.
    model = _slice_pair_model(inner_axis=1, outer_axis=0)
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_consecutive_slices"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"

    concats = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert len(concats) == 4
    # Each Concat's two inputs are [inner Slice2's input i, outer Slice1's
    # input i], in that order -- exactly runTransform's addInput sequence.
    expected_inputs = [
        ["s2_starts", "s1_starts"],
        ["s2_ends", "s1_ends"],
        ["s2_axes", "s1_axes"],
        ["s2_steps", "s1_steps"],
    ]
    actual_inputs = sorted(list(c.input) for c in concats)
    assert actual_inputs == sorted(expected_inputs)
    for c in concats:
        axis_attr = next(a.i for a in c.attribute if a.name == "axis")
        assert axis_attr == 0

    out_name = sim_model.graph.output[0].name
    fused = producer(sim_model, out_name)
    assert fused.op_type == "Slice"
    assert fused.input[0] == "X"
    for concat_output in fused.input[1:5]:
        assert concat_output in {n.output[0] for n in concats}


def test_fuse_consecutive_slices_declines_overlapping_axes():
    # Both slices restrict axis 0: HasIntersection is true, so the predicate
    # declines outright -- both original Slice nodes survive, chained.
    model = _slice_pair_model(inner_axis=0, outer_axis=0)
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_slices")
    assert ops["Slice"] == 2
    (outer,) = [n for n in sim_model.graph.node if "Z" in n.output]
    assert outer.input[0] == "y"  # still chained onto the inner Slice's output


def test_fuse_consecutive_slices_declines_inner_slice_missing_steps():
    # The inner Slice omits `steps` (4-input form: data, starts, ends, axes)
    # -- GetInputsOfPreNode(node, 0).size() == 5 fails, so the predicate
    # declines and both Slice nodes survive chained.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[10,10] X) => (float[?,?] Z)
        <int64[1] s2_starts = {1}, int64[1] s2_ends = {5}, int64[1] s2_axes = {1},
         int64[1] s1_starts = {2}, int64[1] s1_ends = {8}, int64[1] s1_axes = {0},
         int64[1] s1_steps = {1}>
        {
          y = Slice(X, s2_starts, s2_ends, s2_axes)
          Z = Slice(y, s1_starts, s1_ends, s1_axes, s1_steps)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_slices")
    assert ops["Slice"] == 2
    (outer,) = [n for n in sim_model.graph.node if "Z" in n.output]
    assert outer.input[0] == "y"


def test_fuse_consecutive_slices_declines_outer_slice_missing_steps():
    # The outer Slice itself omits `steps` -- node->inputs().size() == 5
    # fails outright, so the predicate declines without even inspecting the
    # inner Slice.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[10,10] X) => (float[?,?] Z)
        <int64[1] s2_starts = {1}, int64[1] s2_ends = {5}, int64[1] s2_axes = {1},
         int64[1] s2_steps = {1}, int64[1] s1_starts = {2}, int64[1] s1_ends = {8},
         int64[1] s1_axes = {0}>
        {
          y = Slice(X, s2_starts, s2_ends, s2_axes, s2_steps)
          Z = Slice(y, s1_starts, s1_ends, s1_axes)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_slices")
    assert ops["Slice"] == 2
