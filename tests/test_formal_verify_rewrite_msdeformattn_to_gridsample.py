"""Formal check for RewriteMSDeformAttnToGridSample (opt-in; onnxsim's own
``onnxsim/passes/rewrite_msdeformattn_to_gridsample.h``).

Unlike this suite's quantization files, this pass is a pure GRAPH-SHAPE
REWRITE: it decomposes mmdeploy/mmcv's custom ``MMCVMultiScaleDeformableAttention``
op into ordinary ONNX ops (``Split``/``Reshape``/``Transpose``/``Gather``/
``Concat``/``Mul``/``Sub``/``ReduceSum``, plus one ``GridSample`` per
feature-map level) with no rounding or quantization anywhere -- pure float32
arithmetic *restructuring*. So the right claim here is EXACT algebraic
equivalence, not a bounded-error claim (unlike ``test_formal_verify_dynamic_quantize_matmul.py``
and friends), and also unlike ``test_formal_verify_rewrite_gridsample_to_gather.py``
(whose whole job is to *reimplement* bilinear interpolation by hand via
``Gather``): this rewrite gets to delegate bilinear interpolation to a real
``GridSample`` node and never touches it. So bilinear interpolation
correctness is GridSample's OWN contract and is deliberately NOT re-proved
here -- the two things this rewrite is itself responsible for, and that this
file's Z3 proofs cover, are:

1. The coordinate transform feeding ``GridSample``. Confirmed from the
   source (``rewrite_msdeformattn_to_gridsample.h``'s header comment step 0
   and its ``runTransform``)::

       Value* sampling_grids =
           b.Sub(b.Mul(sampling_locations, b.ConstF(2.0f)), b.ConstF(1.0f));

   i.e. exactly ``grid = 2 * loc - 1`` -- matching mmcv's own
   ``sampling_grids = 2 * sampling_locations - 1`` line for line, and
   matching this task's guessed formula exactly (no surprise here).
   ``test_coordinate_transform_...`` below proves this is an exact,
   invertible affine bijection from ``[0, 1]`` (mmdeploy's own input
   convention, per the header's declared input spec) onto ``[-1, 1]``
   (``GridSample``'s own convention), endpoints included.

2. The final weighted-sum aggregation across ``(level, point)`` pairs.
   Confirmed from the source (header comment steps 3-5, ``runTransform``)::

       Value* weighted = b.Mul(concat_flat, attn_reshaped);
       Value* reduced = b.ReduceSum(weighted, b.ConstI64Vec1(3), false);

   i.e. exactly ``Mul`` then ``ReduceSum`` over the flattened ``L*P`` axis --
   an elementwise-product-then-sum, matching the mathematical definition
   ``output[...] = sum over (level, point) of attention_weights[...] *
   sampled_value[...]``. The part of this that is actually worth formalizing
   is not "does Mul-then-ReduceSum compute a sum of products" (definitional,
   not interesting) but whether the *two independently-flattened* operands
   -- ``concat_flat`` (built by ``Concat(axis=3, level_outputs)`` over the
   per-level loop, each level's own points already in order, then a Reshape
   merging trailing ``(L, P)`` -> ``L*P``) and ``attn_reshaped`` (built by
   transposing ``attention_weights`` then a Reshape merging trailing
   ``(L, P)`` -> ``L*P``) -- land on the SAME flattened index for the same
   ``(level, point)`` pair. Both merge ``L`` as the outer (more significant)
   axis and ``P`` as the inner one, so both produce flat index ``k = l*P +
   p`` for pair ``(l, p)``; ``test_weighted_sum_aggregation_...`` below
   proves the resulting elementwise-Mul-then-sum equals the reference
   double sum over ``(l, p)`` given this alignment, and a negative control
   shows a mismatched flattening (e.g. one operand transposed to ``p*L +
   l``) would NOT generally preserve the equality -- i.e. this alignment
   check is not vacuous.

3. Scope: this rewrite's own aggregation formula is correct for ANY
   ``attention_weights`` values -- it does not require (and this pass does
   not check or enforce) that weights sum to 1 across ``(level, point)`` for
   a given query/head. That property holds for a *trained* deformable-
   attention model (whose ``attention_weights`` typically come from a
   softmax over exactly that axis), but it is a fact about how the
   *original* op is used, not something ``RewriteMSDeformAttnToGridSample``
   itself relies on for correctness. ``test_aggregation_correct_regardless_of_...``
   below proves the aggregation lemma holds even under an explicit
   hypothesis that the weights do NOT sum to 1.

Bilinear interpolation itself -- ``GridSample``'s own already-correct
semantics -- is exercised only via the differential test below: a small,
from-scratch NumPy reimplementation of multi-scale deformable attention
(Zhu et al., "Deformable DETR"; mmcv's ``multi_scale_deformable_attn_pytorch``,
transcribed independently here purely to confirm understanding, NOT
imported or copied from ``tests/test_msdeformattn_to_gridsample.py``'s own
reference or from any other file in this repo) implements bilinear sampling
by hand (matching ``GridSample``'s own ``align_corners=False``/
``padding_mode="zeros"`` denormalization, per
``rewrite_gridsample_to_gather.h``'s own documented formula
``coord = ((g + 1) * dim - 1) / 2``) -- deliberately NOT delegating to
``GridSample``/onnx/onnxruntime for that step, so this differential check
does not share a code path with the rewrite it is validating. The real
compiled pass is run via ``simplify_isolated_extra`` (``check_n=0``: the
*original* graph contains a custom op with no ONNX kernel, so onnxsim's own
built-in random-input equivalence check has nothing to run the original
graph with -- mirroring ``tests/test_msdeformattn_to_gridsample.py``'s own
documented reasoning for the same thing), the rewritten graph is confirmed
to contain zero remaining custom-op nodes and exactly ``L`` ``GridSample``
nodes, and is then executed via onnxruntime (graph optimization disabled via
``ORT_DISABLE_ALL``, this suite's established precedent for making sure a
runtime does not silently fuse/rewrite the exact node sequence a proof
reasons about) and compared against the independent NumPy reference at a
tight tolerance, using genuinely non-integer sampling locations (real
bilinear interpolation across all four corners, not a trivial on-grid case).
"""

import contextlib
import math

import numpy as np
import onnx
import onnxruntime as ort
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

OP_TYPE = "MMCVMultiScaleDeformableAttention"

# --------------------------------------------------------------------------- #
# 1. Coordinate transform: sampling_locations' own [0,1] convention -> exact,
#    invertible affine map onto GridSample's own [-1,1] convention.
# --------------------------------------------------------------------------- #


def test_coordinate_transform_is_exact_bijection_onto_gridsample_range():
    loc = z3.Real("loc")
    # rewrite_msdeformattn_to_gridsample.h's runTransform:
    #   Value* sampling_grids =
    #       b.Sub(b.Mul(sampling_locations, b.ConstF(2.0f)), b.ConstF(1.0f));
    grid = 2 * loc - 1

    # Maps [0, 1] into [-1, 1] exactly, for every loc in that range.
    prove(z3.Implies(z3.And(loc >= 0, loc <= 1), z3.And(grid >= -1, grid <= 1)))
    # Endpoints land exactly on GridSample's own endpoints (no off-by-epsilon
    # slack): mmdeploy's [0,1] convention's own edges are GridSample's edges.
    prove(z3.Implies(loc == 0, grid == -1))
    prove(z3.Implies(loc == 1, grid == 1))
    # Exact inverse -- the map loses no information (not that this rewrite
    # ever needs to invert it, but an inexact/lossy transform could not
    # satisfy this): loc == (grid + 1) / 2 for every loc.
    prove(((grid + 1) / 2) == loc)


def test_coordinate_transform_negative_control_missing_shift_is_unsound():
    # Sanity check that the proof above is genuine, not vacuous: a plausible
    # *wrong* formula that forgets the "- 1" shift (e.g. a typo'd
    # `2 * sampling_locations` with no Sub at all) must NOT satisfy the same
    # endpoint/range property for every loc in [0, 1] -- confirm Z3 finds a
    # real counterexample (loc == 1 alone already breaks it: 2*1 == 2, not
    # inside [-1, 1]).
    loc = z3.Real("loc")
    wrong_grid = 2 * loc  # the "-1" shift is missing
    solver = z3.Solver()
    solver.add(loc >= 0, loc <= 1)
    solver.add(z3.Not(z3.And(wrong_grid >= -1, wrong_grid <= 1)))
    assert solver.check() == z3.sat


def test_coordinate_transform_negative_control_no_rescale_is_unsound():
    # Likewise, a formula that shifts but forgets to rescale (`loc - 1`,
    # mapping [0,1] to [-1,0]) must fail the endpoint check at loc == 1
    # (produces 0, not GridSample's own +1 upper endpoint).
    loc = z3.Real("loc")
    wrong_grid = loc - 1
    solver = z3.Solver()
    solver.add(loc == 1)
    solver.add(z3.Not(wrong_grid == 1))
    assert solver.check() == z3.sat


# --------------------------------------------------------------------------- #
# 2. Weighted-sum aggregation: Mul-then-ReduceSum over the flattened L*P axis
#    is exactly the reference sum-of-products over (level, point) pairs --
#    which requires the two independently-flattened operands (the stacked
#    per-level GridSample outputs, and attention_weights) to land on the
#    SAME flattened index k = l*P + p for a given (level, point). Concrete,
#    fixed L=2, P=2 (matching the task's own suggested small case), free
#    symbolic sampled-value/attention-weight scalars.
# --------------------------------------------------------------------------- #

_L, _P = 2, 2


def _flatten_l_major(a):
    # k = l*P + p: L outer (more significant), P inner -- how BOTH
    # concat_flat's own Reshape (merging trailing (L, P) after
    # Concat(axis=3, level_outputs), one level appended per loop iteration)
    # and attn_reshaped's own Reshape (merging trailing (L, P) after
    # Transpose([0,2,1,3,4])) flatten the level/point pair, per the header
    # comment's steps 2d/3/4.
    return [a[lvl][p] for lvl in range(_L) for p in range(_P)]


def _flatten_p_major(a):
    # The WRONG order: p*L + l (P outer, L inner) -- used only by the
    # negative control below, to show the alignment genuinely matters.
    return [a[lvl][p] for p in range(_P) for lvl in range(_L)]


def test_weighted_sum_aggregation_matches_reference_sum_of_products():
    s = [[z3.Real(f"s{lvl}{p}") for p in range(_P)] for lvl in range(_L)]
    w = [[z3.Real(f"w{lvl}{p}") for p in range(_P)] for lvl in range(_L)]

    sampled_flat = _flatten_l_major(s)  # concat_flat's own flattening
    weight_flat = _flatten_l_major(w)  # attn_reshaped's own flattening

    # weighted = Mul(concat_flat, attn_reshaped);
    # reduced  = ReduceSum(weighted, axes=[3], keepdims=0)
    mul_then_reduce_sum = z3.Sum(
        *[sampled_flat[k] * weight_flat[k] for k in range(_L * _P)]
    )
    # The mathematical definition: output = sum over (level, point) of
    # attention_weights[...] * sampled_value[...].
    reference = z3.Sum(*[s[lvl][p] * w[lvl][p] for lvl in range(_L) for p in range(_P)])

    prove(mul_then_reduce_sum == reference)


def test_weighted_sum_aggregation_negative_control_mismatched_flatten_order_is_unsound():
    # If the two operands were flattened in different (level, point) orders
    # -- e.g. one L-major (the real pass's own order) and the other
    # P-major -- Mul-then-ReduceSum would silently pair the WRONG sampled
    # value with each attention weight (e.g. index 1 would pair s[0][1]
    # with w[1][0] instead of w[0][1]). Confirm this mismatch is not
    # equivalent to the reference for some concrete s/w -- i.e. that the
    # alignment fact the lemma above depends on is a real, checkable
    # property, not a vacuously-true one.
    s = [[z3.Real(f"s{lvl}{p}") for p in range(_P)] for lvl in range(_L)]
    w = [[z3.Real(f"w{lvl}{p}") for p in range(_P)] for lvl in range(_L)]

    sampled_flat = _flatten_l_major(s)
    weight_flat = _flatten_p_major(w)  # mismatched flattening order

    mismatched = z3.Sum(*[sampled_flat[k] * weight_flat[k] for k in range(_L * _P)])
    reference = z3.Sum(*[s[lvl][p] * w[lvl][p] for lvl in range(_L) for p in range(_P)])

    solver = z3.Solver()
    solver.add(z3.Not(mismatched == reference))
    assert solver.check() == z3.sat


# --------------------------------------------------------------------------- #
# 3. Scope: the aggregation formula is exactly correct for ANY
#    attention_weights, whether or not they sum to 1 across (level, point).
#    Summing to 1 is a property of how the *original* op is trained/used
#    (a softmax over that axis), never checked or assumed by this rewrite.
# --------------------------------------------------------------------------- #


def test_aggregation_correct_regardless_of_whether_weights_sum_to_one():
    s = [[z3.Real(f"s{lvl}{p}") for p in range(_P)] for lvl in range(_L)]
    w = [[z3.Real(f"w{lvl}{p}") for p in range(_P)] for lvl in range(_L)]

    sampled_flat = _flatten_l_major(s)
    weight_flat = _flatten_l_major(w)
    mul_then_reduce_sum = z3.Sum(
        *[sampled_flat[k] * weight_flat[k] for k in range(_L * _P)]
    )
    reference = z3.Sum(*[s[lvl][p] * w[lvl][p] for lvl in range(_L) for p in range(_P)])
    weights_sum = z3.Sum(*[w[lvl][p] for lvl in range(_L) for p in range(_P)])

    # Holds under an explicit hypothesis that the weights do NOT sum to 1 --
    # e.g. all-zero weights (sum 0, a degenerate but perfectly legal input
    # this rewrite must still handle correctly) or weights summing to some
    # arbitrary value are covered just as much as normalized ones.
    prove(z3.Implies(weights_sum != 1, mul_then_reduce_sum == reference))
    # ...and it holds equally well when they DO happen to sum to 1 -- the
    # formula's correctness never depended on which case applies.
    prove(z3.Implies(weights_sum == 1, mul_then_reduce_sum == reference))


# --------------------------------------------------------------------------- #
# Differential/structural test: the real compiled pass, run end to end and
# checked against a from-scratch NumPy reference (bilinear sampling
# implemented by hand, matching GridSample's own align_corners=False/
# padding_mode="zeros" semantics -- NOT delegated to GridSample/onnx/
# onnxruntime, so this reference shares no code path with the rewrite).
# --------------------------------------------------------------------------- #


def _bilinear_sample_align_corners_false_zeros(feat, gx, gy):
    """Bilinear-sample a single-channel 2-D array at one GridSample-
    convention ``(gx, gy)`` coordinate (each in ``[-1, 1]``), with
    ``align_corners=False`` denormalization and ``padding_mode="zeros"``.

    Written entirely from scratch: denormalization formula
    ``coord = ((g + 1) * dim - 1) / 2`` is exactly ``align_corners=0``'s own
    formula documented in ``rewrite_gridsample_to_gather.h``'s
    ``Denormalize`` comment -- read to confirm understanding of ONNX
    ``GridSample``'s own semantics, not to reuse any code from it.
    """
    H, W = feat.shape
    ix = ((gx + 1.0) * W - 1.0) / 2.0
    iy = ((gy + 1.0) * H - 1.0) / 2.0
    x0 = math.floor(ix)
    x1 = x0 + 1
    y0 = math.floor(iy)
    y1 = y0 + 1
    wx1 = ix - x0
    wx0 = 1.0 - wx1
    wy1 = iy - y0
    wy0 = 1.0 - wy1

    def px(xi, yi):
        if 0 <= xi < W and 0 <= yi < H:
            return float(feat[yi, xi])
        return 0.0  # padding_mode="zeros"

    return (
        wy0 * wx0 * px(x0, y0)
        + wy0 * wx1 * px(x1, y0)
        + wy1 * wx0 * px(x0, y1)
        + wy1 * wx1 * px(x1, y1)
    )


def independent_msda_reference(
    value, spatial_shapes, sampling_locations, attention_weights
):
    """From-scratch multi-scale deformable attention reference (Zhu et al.,
    "Deformable DETR"; mmcv's ``multi_scale_deformable_attn_pytorch``).

    Deliberately not imported from, or copied out of,
    ``tests/test_msdeformattn_to_gridsample.py``'s own reference (which
    delegates bilinear sampling to a real GridSample node/evaluator) or any
    other file in this repo -- bilinear sampling here is the hand-rolled
    helper above, so this function shares no code path with either
    ``GridSample`` or the rewrite under test.
    """
    bs, num_keys, M, D = value.shape
    _, num_queries, _, L, P, _ = sampling_locations.shape
    assert sum(int(h) * int(w) for h, w in spatial_shapes) == num_keys

    # Split value's flattened (num_keys,) axis back into one raster-order
    # (H_l, W_l) feature map per level (mirrors spec step 1/1a).
    offsets = np.cumsum([int(h) * int(w) for h, w in spatial_shapes])[:-1]
    value_per_level = np.split(value, offsets, axis=1)
    feature_maps = []
    for level, (h_, w_) in enumerate(spatial_shapes):
        h_, w_ = int(h_), int(w_)
        feature_maps.append(value_per_level[level].reshape(bs, h_, w_, M, D))

    output = np.zeros((bs, num_queries, M, D), dtype=np.float64)
    for b in range(bs):
        for q in range(num_queries):
            for m in range(M):
                for level in range(L):
                    feat = feature_maps[level]
                    for p in range(P):
                        loc_x, loc_y = sampling_locations[b, q, m, level, p]
                        # Coordinate transform under test (step 0).
                        gx = 2.0 * float(loc_x) - 1.0
                        gy = 2.0 * float(loc_y) - 1.0
                        weight = float(attention_weights[b, q, m, level, p])
                        for d in range(D):
                            sampled = _bilinear_sample_align_corners_false_zeros(
                                feat[b, :, :, m, d], gx, gy
                            )
                            # Aggregation under test (steps 3-5).
                            output[b, q, m, d] += weight * sampled
    return output.reshape(bs, num_queries, M * D).astype(np.float32)


def _register_custom_op_schema(domain, since_version=1):
    op_schema = onnx.defs.OpSchema
    schema = op_schema(
        OP_TYPE,
        domain,
        since_version,
        inputs=[
            op_schema.FormalParameter("value", "T", "value"),
            op_schema.FormalParameter("spatial_shapes", "T1", "spatial_shapes"),
            op_schema.FormalParameter("level_start_index", "T1", "level_start_index"),
            op_schema.FormalParameter("sampling_locations", "T", "sampling_locations"),
            op_schema.FormalParameter("attention_weights", "T", "attention_weights"),
        ],
        outputs=[op_schema.FormalParameter("output", "T", "output")],
        type_constraints=[
            ("T", ["tensor(float)"], "Constrain to float tensors."),
            ("T1", ["tensor(int64)"], "Constrain to int64 tensors."),
        ],
        attributes=[
            op_schema.Attribute(
                "im2col_step",
                op_schema.AttrType.INT,
                "CUDA-kernel batching knob, no effect on output values",
                required=False,
            ),
        ],
    )
    onnx.defs.register_schema(schema)


@contextlib.contextmanager
def _custom_op_schema(domain=""):
    # No built-in ONNX schema exists for this custom op -- register a
    # minimal one for the duration of the test so the graph passes onnx's
    # own structural validation on the way in, mirroring
    # tests/test_msdeformattn_to_gridsample.py's own
    # ``_msda_schema``/``_register_schema``.
    _register_custom_op_schema(domain)
    try:
        yield
    finally:
        onnx.defs.deregister_schema(OP_TYPE, 1, domain)


def _model(bs, num_keys, num_queries, M, D, spatial_shapes, P, opset=20, ir_version=10):
    L = len(spatial_shapes)
    md = M * D
    body = f"""
    <
      ir_version: {ir_version},
      opset_import: ["": {opset}]
    >
    agraph (
      float[{bs},{num_keys},{M},{D}] value,
      int64[{L},2] spatial_shapes,
      int64[{L}] level_start_index,
      float[{bs},{num_queries},{M},{L},{P},2] sampling_locations,
      float[{bs},{num_queries},{M},{L},{P}] attention_weights
    ) => (float[{bs},{num_queries},{md}] Y)
    {{
      Y = {OP_TYPE}(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
    }}
    """
    return parser.parse_model(body)


def test_rewrite_msdeformattn_to_gridsample_pass_matches_independent_reference():
    rng = np.random.default_rng(0)
    # Two feature levels of distinct, small, hand-reasoned-about sizes.
    spatial_shapes = [(4, 4), (2, 2)]
    bs, M, D, P = 1, 1, 2, 2
    num_queries = 3
    L = len(spatial_shapes)
    num_keys = sum(h * w for h, w in spatial_shapes)

    value = rng.standard_normal((bs, num_keys, M, D)).astype(np.float32)
    spatial_shapes_arr = np.array(spatial_shapes, dtype=np.int64)
    level_start_index = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes])[:-1]]
    ).astype(np.int64)
    # Genuinely non-integer/off-grid sampling locations (real 4-corner
    # bilinear interpolation, not a trivial on-grid case), with a bit of
    # [0,1]-overspill so a feature map's own zero-padded border is exercised
    # too.
    sampling_locations = rng.uniform(
        -0.1, 1.1, size=(bs, num_queries, M, L, P, 2)
    ).astype(np.float32)
    # Deliberately NOT normalized to sum to 1 across (level, point) -- see
    # test_aggregation_correct_regardless_of_whether_weights_sum_to_one
    # above; this differential check's own weights confirm the same point
    # empirically end to end.
    attention_weights = rng.uniform(0.1, 2.0, size=(bs, num_queries, M, L, P)).astype(
        np.float32
    )

    with _custom_op_schema(""):
        model = _model(bs, num_keys, num_queries, M, D, spatial_shapes, P)
        sim_model, ops = simplify_isolated_extra(
            model, "rewrite_msdeformattn_to_gridsample", check_n=0
        )

    assert ops[OP_TYPE] == 0, ops
    assert ops["GridSample"] == L, ops  # exactly one GridSample per level

    feeds = {
        "value": value,
        "spatial_shapes": spatial_shapes_arr,
        "level_start_index": level_start_index,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
    }

    # Graph optimization disabled (ORT_DISABLE_ALL): this suite's
    # established precedent for making sure onnxruntime executes exactly
    # the node sequence the pass produced, rather than a fused/rewritten
    # variant of it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        sim_model.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    actual = sess.run(None, feeds)[0]

    expected = independent_msda_reference(
        value, spatial_shapes_arr, sampling_locations, attention_weights
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
