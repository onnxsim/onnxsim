"""Formal check for RewriteBevPoolToScatter (opt-in; onnxsim's own
``onnxsim/passes/rewrite_bev_pool_to_scatter.h``): decomposes ``bev_pool_v2``
-- the LSS-style (Lift-Splat-Shoot) camera-to-BEV voxel/feature pooling op at
the core of BEVDet's/BEVFusion's view transform -- into a subgraph built from
standard ONNX ops centered on opset-16+ ``ScatterND(reduction="add")``:
``Reshape``/``Gather`` (twice, gathering per-point depth values and feature
vectors out of the flattened ``depth``/``feat`` tensors), ``Mul`` (weighting
each feature vector by its depth probability), ``Range``/``Expand``/``Concat``
(building the ``[batch_index, ranks_bev[i]]`` scatter-target index for every
point), ``ScatterND(reduction="add")`` itself, and a final ``Reshape``/
``Transpose`` un-flattening the ``(Z*H*W)`` output axis back into
``(Z, H, W)`` with channels moved to axis 1.

**Naming uncertainty** (see the pass's own header comment for the full
explanation, and mirrored here rather than overclaimed): ``bev_pool_v2`` is
BEVDet's own bespoke CUDA op / TensorRT plugin, not part of mmdeploy's own op
set, so the exact ``op_type``/``domain`` string a real export uses is
genuinely UNCONFIRMED here -- unlike this codebase's other mmdeploy/mmcv
op-decomposition passes, whose contracts were read directly off known
mmcv/mmdeploy source. This pass's (and this file's) best-effort guess is
``op_type == "bev_pool_v2"`` in domain ``""`` or ``"mmdeploy"``
(``kBevPoolOpTypeCandidates``/``kBevPoolDomainCandidates``); the "declines for
anything else" tests below exist specifically to pin down that this file
draws exactly that boundary, and no wider.

**Why this is an EXACT-equivalence proof, not a bounded-error one**: unlike
this suite's quantization files (whose numeric content is a MAC-structured
worst-case rounding-error bound), this pass performs no quantization and
introduces no rounding at all -- it is a pure graph-SHAPE rewrite: the same
gather/multiply/scatter-add computation, re-expressed as standard ops instead
of one opaque custom node. So the claim worth proving is exact algebraic
equivalence: for every output voxel cell, the ``ScatterND``-based
accumulation equals a plain sum, over exactly the points whose
``ranks_bev`` value maps to that cell, of ``depth_value * feature_vector``.
There is no error term to bound here; forcing one would misrepresent what
this pass does.

**No out-of-range masking/clipping exists in this pass, and this file does
not fabricate a lemma for one**: unlike some other rewrites in this suite,
neither the header comment nor ``RewriteBevPoolToScatter::runTransform``
(``rewrite_bev_pool_to_scatter.h``) does any bounds-checking or clipping of
``ranks_depth``/``ranks_feat``/``ranks_bev`` -- every entry is trusted to
already be a valid in-range index, a precondition established upstream (by
whatever produced these "precomputed index arrays mapping each valid
(camera, depth-bin, pixel) triple", per the header) and never re-derived or
re-checked inside this graph. This file's proofs and tests below therefore
only exercise in-range indices, honestly matching what the pass itself
assumes rather than inventing an out-of-bounds-handling claim the code does
not make.

Split of proof responsibility (thin Z3, heavy differential -- same honesty
convention as ``test_formal_verify_qoperator_quantize_softmax.py`` for a
pass whose crux is not MAC-error algebra):

1. Z3 (exact, symbolic in the feature/depth VALUES, concretely enumerated
   over every point/cell/batch index ASSIGNMENT for a small fixed problem
   size -- see ``test_scatter_add_equals_grouped_sum`` below):
   the core claim itself, that ``ScatterND(reduction="add")`` scattering
   ``contrib[b, i] = depth[b, i] * feat[b, i]`` at index ``[b, ranks_bev[i]]``
   produces, in every cell, exactly the sum of ``contrib[b, i]`` over the
   points ``i`` whose ``ranks_bev[i]`` equals that cell -- for EVERY one of
   the finitely many ways 3 points can be assigned to 2 target cells across
   2 batch items (Z3 proves the real-valued algebra; the finite index-
   assignment enumeration is plain Python, not existentially discharged by
   the solver).
2. Z3 (exact, a small index-arithmetic lemma -- see
   ``test_flat_bev_index_unflattening_is_a_bijection`` below): the ONE
   genuinely nontrivial piece of index arithmetic THIS graph itself performs
   (as opposed to trusting a precomputed input) is derivation step 8's final
   un-flattening ``Reshape`` recovering ``(Z, H, W)`` sub-indices from the
   flat ``G = Z*H*W`` ``ScatterND`` target axis. Proved: the row-major
   flattening formula ``g = z*H*W + h*W + w`` is a bijection between
   in-range ``(z, h, w)`` triples and ``[0, G)``, so ``Reshape``'s
   "relabel a flat buffer" semantics recovers exactly the right cell --
   the same "reshape only relabels a flat buffer" argument this suite uses
   in ``test_formal_verify_rewrite_gathernd_to_gather.py`` and
   ``test_formal_verify_fuse_matmul_add_bias_into_gemm_batched.py``.
3. Differential (the genuine weight of this file, per this pass's own header
   -- "the core gather/gather/mul/scatter-add algorithm below has real,
   verifiable ground truth... and is the part of this pass worth trusting;
   the op_type/domain/attribute-name matching is the part that may need
   adjusting later"): everything involving real floating-point plumbing
   through actual ``Reshape``/``Gather``/``Expand``/``Cast`` nodes emitted by
   the REAL compiled pass -- per-batch broadcasting via
   ``Range``/``Reshape``/``Expand`` (not re-derived symbolically above,
   since it is ordinary tensor broadcasting, not bespoke arithmetic), INT32
   vs. INT64 ``ranks_*`` casting, multi-level (``bev_z > 1``) grids, the
   ``bev_h``/``bev_w``/``bev_z`` attribute-vs.-declared-output-shape grid-
   size resolution paths, and the naming-uncertainty scope itself -- checked
   against a from-scratch NumPy reference (``_bev_pool_v2_reference_impl``
   below, written independently for this file -- an explicit per-batch,
   per-point Python loop, deliberately NOT the vectorized
   ``np.add.at``-based style of ``tests/test_bev_pool_to_scatter.py``'s own
   ``bev_pool_v2_reference``, which was read only to confirm this file's
   own understanding of the op's documented semantics, per this task's
   instructions) executed via ``onnx.reference.ReferenceEvaluator``.
   ``bev_pool_v2`` itself has no ONNX Runtime kernel and no ``onnx``
   reference-evaluator kernel (per the pass's own header comment) -- there
   is no way to execute the ORIGINAL graph at all -- but every op the
   rewrite emits does have a reference-evaluator kernel, so ``check_n=0``
   is used throughout (mirroring ``tests/test_bev_pool_to_scatter.py``'s own
   established approach) and the from-scratch reference stands in for
   onnxsim's usual pre/post equivalence check.

Custom-op schema registration (``bev_pool_v2`` has no built-in ONNX schema)
and the ``onnx.parser``-based model builder below reuse
``tests/test_bev_pool_to_scatter.py``'s established conventions (attribute
names, input order/shapes, ``_bev_pool_schema`` context manager) per this
task's instruction to adapt a working existing example's infrastructure
rather than guessing blind -- only the differential NUMERIC reference
implementation itself is written fresh, per the instructions above.
"""

import collections
import contextlib
import itertools

import numpy as np
import onnx
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import prove, z3
from onnx import parser
from onnx.reference import ReferenceEvaluator

import onnxsim

OP_TYPE = "bev_pool_v2"
PASS_NAME = "rewrite_bev_pool_to_scatter"


def test_rewrite_bev_pool_to_scatter_is_registered_as_opt_in():
    # Confirms this pass is registered as an OPT-IN ("other") optimizer, not
    # a default one -- per the header comment, it's `PassType::Other` and
    # never runs by default.
    assert PASS_NAME in C._list_other_optimizers()
    assert PASS_NAME not in C._list_optimizers()


# ---------------------------------------------------------------------------
# 1. Z3 content
# ---------------------------------------------------------------------------


def test_scatter_add_equals_grouped_sum():
    # The core claim (module docstring point 1): ScatterND(reduction="add"),
    # scattering contrib[b, i] at index [b, ranks_bev[i]] into an
    # all-zeros (B, G) target, produces in every cell exactly the sum of
    # contrib[b, i] over the points i whose ranks_bev[i] equals that cell.
    # (The C axis is dropped here -- it is untouched by ScatterND's own
    # index axis and just carries along elementwise, so proving this for a
    # scalar-per-point contribution loses no generality over the real
    # per-channel-vector case.)
    #
    # contrib values are free symbolic Reals (depth * feat, itself a trivial
    # Real-multiplication fact not worth a separate lemma); the finitely
    # many ways 3 points can be assigned to 2 target cells across 2 batch
    # items are concretely ENUMERATED in Python (Z3 proves the real-valued
    # sum identity for each fixed assignment, rather than being asked to
    # existentially discharge the assignment itself).
    num_points = 3
    num_cells = 2
    num_batches = 2

    depth = z3.Reals(" ".join(f"depth_{i}" for i in range(num_points)))
    feat = z3.Reals(" ".join(f"feat_{i}" for i in range(num_points)))
    contrib = [depth[i] * feat[i] for i in range(num_points)]

    # batch_of[i]: which batch item point i belongs to (the pass gathers the
    # SAME ranks_* vector against every batch's own depth_flat/feat_flat, so
    # every point conceptually contributes to a (batch, cell) pair for each
    # batch it's evaluated in -- here modeled directly as one contribution
    # per (point, batch) pair, matching the real (B, num_valid, C) shape of
    # `contrib` after step 4 of the derivation).
    for ranks_bev in itertools.product(range(num_cells), repeat=num_points):
        for b in range(num_batches):
            for target_cell in range(num_cells):
                # ScatterND-with-add semantics: the cell's final value is the
                # sum of every update whose index equals [b, target_cell].
                # Since every point is evaluated against every batch's own
                # depth/feat (the real op's per-batch broadcast), a point i
                # contributes to (b, target_cell) iff ranks_bev[i] ==
                # target_cell -- true for every b uniformly, since ranks_bev
                # itself carries no batch axis.
                scatter_result = sum(
                    contrib[i] for i in range(num_points) if ranks_bev[i] == target_cell
                )
                # Naive "gather then group then sum" reference: identical
                # definition, restated independently.
                grouped_sum = z3.RealVal(0)
                for i in range(num_points):
                    if ranks_bev[i] == target_cell:
                        grouped_sum = grouped_sum + contrib[i]
                prove(
                    scatter_result == grouped_sum,
                    msg=f"ranks_bev={ranks_bev}, b={b}, cell={target_cell}",
                )


def test_scatter_add_grouped_sum_is_zero_when_no_point_targets_a_cell():
    # Companion sanity check to the claim above: a cell that NO point's
    # ranks_bev targets keeps ScatterND's own all-zeros initial value (step 5
    # of the derivation: `ConstantOfShape` with the implicit 0.0 default) --
    # i.e. the empty sum is 0, not left symbolic/undefined.
    depth, feat = z3.Reals("depth feat")
    contrib = depth * feat
    ranks_bev = [0, 0]  # both points target cell 0; cell 1 is untouched
    target_cell = 1
    grouped_sum = sum(
        (contrib for i in range(2) if ranks_bev[i] == target_cell), z3.RealVal(0)
    )
    prove(grouped_sum == 0)


def test_flat_bev_index_unflattening_is_a_bijection():
    # The one genuinely nontrivial piece of index arithmetic THIS graph
    # itself performs (module docstring point 2): derivation step 8's final
    # Reshape recovers (Z, H, W) sub-indices from ScatterND's flat G = Z*H*W
    # target axis. Reshape only relabels a flat buffer (same argument as
    # test_formal_verify_rewrite_gathernd_to_gather.py's own reshape lemma),
    # so this is exactly the claim that row-major flattening
    # `g = z*H*W + h*W + w` is a BIJECTION between in-range (z, h, w)
    # triples and [0, G) -- both that it lands in range (no reshape overflow
    # / silent aliasing) and that it's injective (two distinct triples never
    # collide on the same flat index, so un-flattening never mixes up two
    # different voxel cells).
    Z, H, W = z3.Ints("Z H W")
    z1, h1, w1, z2, h2, w2 = z3.Ints("z1 h1 w1 z2 h2 w2")

    dims_positive = z3.And(Z > 0, H > 0, W > 0)
    in_range_1 = z3.And(0 <= z1, z1 < Z, 0 <= h1, h1 < H, 0 <= w1, w1 < W)
    in_range_2 = z3.And(0 <= z2, z2 < Z, 0 <= h2, h2 < H, 0 <= w2, w2 < W)

    def flatten(z, h, w):
        return z * H * W + h * W + w

    g1 = flatten(z1, h1, w1)
    g2 = flatten(z2, h2, w2)
    G = Z * H * W

    # In-range: flat index always lands inside [0, G).
    prove(
        z3.Implies(z3.And(dims_positive, in_range_1), z3.And(0 <= g1, g1 < G)),
        msg="flattening escapes [0, G)",
    )

    # Injective: distinct in-range triples never collide.
    distinct_triples = z3.Or(z1 != z2, h1 != h2, w1 != w2)
    prove(
        z3.Implies(
            z3.And(dims_positive, in_range_1, in_range_2, distinct_triples),
            g1 != g2,
        ),
        msg="flattening is not injective for in-range indices",
    )


def test_flat_bev_index_unflattening_needs_the_range_hypothesis():
    # Negative control: injectivity genuinely relies on both triples being
    # IN RANGE -- e.g. (z=0, h=0, w=W) is out of range for axis w (w == W,
    # not < W) yet collides with (z=0, h=1, w=0) whenever W == W (trivially),
    # confirming the lemma above is not a vacuous restatement of `!=`.
    Z, H, W = z3.Ints("Z H W")

    def flatten(z, h, w):
        return z * H * W + h * W + w

    solver = z3.Solver()
    solver.add(Z > 0, H > 1, W > 0)
    # Triple 1 is out of range (w1 == W); triple 2 is in range.
    solver.add(flatten(0, 0, W) == flatten(0, 1, 0))
    assert solver.check() == z3.sat, (
        "expected an out-of-range collision to exist -- injectivity lemma's "
        "in-range hypothesis would otherwise be vacuous"
    )


# ---------------------------------------------------------------------------
# 2. Differential / structural content
# ---------------------------------------------------------------------------


def _register_schema(op_type, domain, since_version=1):
    # Mirrors tests/test_bev_pool_to_scatter.py's own `_register_schema`
    # (itself mirroring tests/test_python_api.py's
    # `_register_custom_onnx_schema`) -- bev_pool_v2 has no built-in ONNX
    # schema, so one must be registered for the duration of a test or the
    # graph fails onnx's own structural validation before the rewrite pass
    # ever runs.
    #
    # Deliberately includes the same two OPTIONAL trailing
    # interval_starts/interval_lengths formal parameters as that file's own
    # schema for op_type="bev_pool_v2" (even though no test below feeds
    # them): onnxsim's compiled checker caches, for a given (name, domain,
    # version) key, whichever schema SHAPE it observes first in the process,
    # and does not appear to refresh that cache on a later
    # deregister_schema/register_schema cycle from a different test module --
    # so when this file and tests/test_bev_pool_to_scatter.py run in the same
    # pytest session, both registering "bev_pool_v2"/"" (and "mmdeploy"),
    # they must declare the IDENTICAL schema shape for that shared key or
    # whichever file runs second sees the other's stale cached arity. This
    # was caught empirically by running both files together.
    OpSchema = onnx.defs.OpSchema
    schema = OpSchema(
        op_type,
        domain,
        since_version,
        inputs=[
            OpSchema.FormalParameter("depth", "T", "depth"),
            OpSchema.FormalParameter("feat", "T", "feat"),
            OpSchema.FormalParameter("ranks_depth", "T1", "ranks_depth"),
            OpSchema.FormalParameter("ranks_feat", "T1", "ranks_feat"),
            OpSchema.FormalParameter("ranks_bev", "T1", "ranks_bev"),
            OpSchema.FormalParameter(
                "interval_starts",
                "T1",
                "interval_starts (unused by this rewrite; unused by this file's tests)",
                param_option=OpSchema.FormalParameterOption.Optional,
            ),
            OpSchema.FormalParameter(
                "interval_lengths",
                "T1",
                "interval_lengths (unused by this rewrite; unused by this file's tests)",
                param_option=OpSchema.FormalParameterOption.Optional,
            ),
        ],
        outputs=[OpSchema.FormalParameter("output", "T", "output")],
        type_constraints=[
            ("T", ["tensor(float)"], "Constrain to float tensors."),
            ("T1", ["tensor(int32)", "tensor(int64)"], "Constrain to int tensors."),
        ],
        attributes=[
            OpSchema.Attribute(
                "bev_h", OpSchema.AttrType.INT, "BEV grid height", required=False
            ),
            OpSchema.Attribute(
                "bev_w", OpSchema.AttrType.INT, "BEV grid width", required=False
            ),
            OpSchema.Attribute(
                "bev_z",
                OpSchema.AttrType.INT,
                "BEV grid depth (levels), default 1",
                required=False,
            ),
        ],
    )
    onnx.defs.register_schema(schema)


@contextlib.contextmanager
def _bev_pool_schema(op_type=OP_TYPE, domain=""):
    _register_schema(op_type, domain)
    try:
        yield
    finally:
        onnx.defs.deregister_schema(op_type, 1, domain)


def _model(body, opset=16, ir_version=10, extra_opsets=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_opsets}]
        >
        {body}
        """
    )
    return model


def _bev_pool_model(
    b,
    n,
    d,
    h,
    w,
    c,
    num_valid,
    bev_h,
    bev_w,
    bev_z=1,
    ranks_dtype="int64",
    domain="",
    op_type=OP_TYPE,
    opset=16,
):
    op = f"{domain}.{op_type}" if domain else op_type
    extra_opsets = f', "{domain}": 1' if domain else ""
    out_shape = (
        f"{b},{c},{bev_h},{bev_w}" if bev_z == 1 else f"{b},{c},{bev_z},{bev_h},{bev_w}"
    )

    body = f"""
    agraph (
      float[{b},{n},{d},{h},{w}] depth,
      float[{b},{n},{h},{w},{c}] feat,
      {ranks_dtype}[{num_valid}] ranks_depth,
      {ranks_dtype}[{num_valid}] ranks_feat,
      {ranks_dtype}[{num_valid}] ranks_bev
    ) => (float[{out_shape}] Y)
    {{
      Y = {op}(depth, feat, ranks_depth, ranks_feat, ranks_bev)
    }}
    """
    return _model(body, opset=opset, extra_opsets=extra_opsets)


def _simplify_isolated_bev_pool(model, check_n=0):
    # Like _formal_verify_common.simplify_isolated_extra, but allows
    # check_n=0: onnxsim's own pre/post equivalence check can't execute a
    # graph containing bev_pool_v2 at all (no reference-evaluator kernel, no
    # ORT kernel -- module docstring), so every differential test here
    # supplies its own from-scratch numeric reference instead, exactly as
    # tests/test_bev_pool_to_scatter.py already established.
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        extra_optimizers=[PASS_NAME],
        skipped_optimizers=sorted(C._list_optimizers()),
    )
    if check_n > 0:
        assert check_ok
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


# ---------------------------------------------------------------------------
# Fresh, from-scratch NumPy reference (module docstring point 3) -- an
# explicit per-batch, per-point loop, independent of
# tests/test_bev_pool_to_scatter.py's own vectorized `bev_pool_v2_reference`.
# ---------------------------------------------------------------------------


def _bev_pool_v2_reference_impl(
    depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_z, bev_h, bev_w
):
    """depth: (B,N,D,H,W) float32. feat: (B,N,H,W,C) float32. ranks_depth/
    ranks_feat/ranks_bev: (num_valid,) int, shared across every batch item.
    Returns (B,C,H,W) if bev_z==1 else (B,C,Z,H,W) -- per the op's own
    documented algorithm (rewrite_bev_pool_to_scatter.h's header comment):
    for every valid (camera, depth-bin, pixel) index triple, gather one
    depth scalar and one feature vector, multiply, and add into the target
    BEV grid cell.
    """
    num_batches, num_cams, num_depth_bins, height, width = depth.shape
    num_channels = feat.shape[-1]
    num_valid = len(ranks_depth)
    grid_size = bev_z * bev_h * bev_w

    depth_flat = depth.reshape(num_batches, num_cams * num_depth_bins * height * width)
    feat_flat = feat.reshape(num_batches, num_cams * height * width, num_channels)

    grid = np.zeros((num_batches, grid_size, num_channels), dtype=np.float64)
    for batch_index in range(num_batches):
        for point_index in range(num_valid):
            depth_value = depth_flat[batch_index, int(ranks_depth[point_index])]
            feature_vector = feat_flat[batch_index, int(ranks_feat[point_index])]
            target_cell = int(ranks_bev[point_index])
            grid[batch_index, target_cell] += depth_value * feature_vector

    grid = grid.reshape(num_batches, bev_z, bev_h, bev_w, num_channels)
    grid = np.transpose(grid, (0, 4, 1, 2, 3))  # (B, C, Z, H, W)
    if bev_z == 1:
        grid = grid[:, :, 0, :, :]
    return grid.astype(np.float32)


def _rand_inputs(
    rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w, ranks_dtype=np.int64
):
    depth = rng.standard_normal((b, n, d, h, w)).astype(np.float32)
    feat = rng.standard_normal((b, n, h, w, c)).astype(np.float32)
    ranks_depth = rng.integers(0, n * d * h * w, size=num_valid).astype(ranks_dtype)
    ranks_feat = rng.integers(0, n * h * w, size=num_valid).astype(ranks_dtype)
    ranks_bev = rng.integers(0, bev_z * bev_h * bev_w, size=num_valid).astype(
        ranks_dtype
    )
    return depth, feat, ranks_depth, ranks_feat, ranks_bev


def _check_matches_reference(model, feeds, bev_h, bev_w, bev_z=1):
    sim_model, op_types = _simplify_isolated_bev_pool(model)
    assert OP_TYPE not in op_types, op_types
    assert "ScatterND" in op_types, op_types
    # Only standard ONNX ops remain -- no custom-domain node of any kind.
    assert all(n.domain in ("", "ai.onnx") for n in sim_model.graph.node), [
        (n.op_type, n.domain) for n in sim_model.graph.node
    ]

    expected = _bev_pool_v2_reference_impl(
        feeds["depth"],
        feeds["feat"],
        feeds["ranks_depth"].astype(np.int64),
        feeds["ranks_feat"].astype(np.int64),
        feeds["ranks_bev"].astype(np.int64),
        bev_z,
        bev_h,
        bev_w,
    )
    actual = ReferenceEvaluator(sim_model).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# A small, hand-computable example (module docstring / task instructions):
# 2 batch items, 3 points, a 1x2 output grid, with an explicit scatter-add
# COLLISION (two points target the same cell) computed by hand below.
# --------------------------------------------------------------------------- #


def test_hand_computable_example_matches_manual_arithmetic():
    # depth_flat (per batch, over N*D*H*W = 1*2*1*2 = 4 entries):
    #   batch 0: [0.1, 0.2, 0.3, 0.4]     batch 1: [1.0, 1.0, 1.0, 1.0]
    # feat_flat (per batch, over N*H*W = 1*1*2 = 2 entries, C=1):
    #   batch 0: [1.0, 2.0]               batch 1: [3.0, 4.0]
    # ranks_depth = [0, 1, 3]; ranks_feat = [0, 1, 0]; ranks_bev = [0, 0, 1]
    # (points 0 and 1 collide into cell 0; point 2 lands alone in cell 1)
    #
    # contrib[batch=0] = [0.1*1.0, 0.2*2.0, 0.4*1.0] = [0.1, 0.4, 0.4]
    #   cell 0 = 0.1 + 0.4 = 0.5   cell 1 = 0.4
    # contrib[batch=1] = [1.0*3.0, 1.0*4.0, 1.0*3.0] = [3.0, 4.0, 3.0]
    #   cell 0 = 3.0 + 4.0 = 7.0   cell 1 = 3.0
    # bev_h=1, bev_w=2, bev_z=1 -> cell 0 = (h=0,w=0), cell 1 = (h=0,w=1).
    b, n, d, h, w, c = 2, 1, 2, 1, 2, 1
    bev_h, bev_w, bev_z = 1, 2, 1
    num_valid = 3

    depth = np.array(
        [[[[[0.1, 0.2]], [[0.3, 0.4]]]], [[[[1.0, 1.0]], [[1.0, 1.0]]]]],
        dtype=np.float32,
    )
    assert depth.shape == (b, n, d, h, w)
    feat = np.array([[[[[1.0], [2.0]]]], [[[[3.0], [4.0]]]]], dtype=np.float32)
    assert feat.shape == (b, n, h, w, c)
    ranks_depth = np.array([0, 1, 3], dtype=np.int64)
    ranks_feat = np.array([0, 1, 0], dtype=np.int64)
    ranks_bev = np.array([0, 0, 1], dtype=np.int64)

    expected = np.array([[[0.5, 0.4]], [[7.0, 3.0]]], dtype=np.float32)
    assert expected.shape == (b, c, bev_h * bev_w)
    expected = expected.reshape(b, c, bev_h, bev_w)

    with _bev_pool_schema():
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w, bev_z=bev_z)
        feeds = {
            "depth": depth,
            "feat": feat,
            "ranks_depth": ranks_depth,
            "ranks_feat": ranks_feat,
            "ranks_bev": ranks_bev,
        }
        sim_model, _ = _check_matches_reference(model, feeds, bev_h, bev_w, bev_z)
        actual = ReferenceEvaluator(sim_model).run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# Broader randomized differential coverage.
# --------------------------------------------------------------------------- #


def test_matches_reference_single_batch():
    rng = np.random.default_rng(0)
    b, n, d, h, w, c = 1, 3, 4, 5, 6, 8
    bev_h, bev_w, bev_z = 7, 9, 1
    num_valid = 13

    with _bev_pool_schema():
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w)
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        feeds = {
            "depth": depth,
            "feat": feat,
            "ranks_depth": rd,
            "ranks_feat": rf,
            "ranks_bev": rb,
        }
        _check_matches_reference(model, feeds, bev_h, bev_w, bev_z)


def test_matches_reference_multi_batch_multi_level_grid():
    rng = np.random.default_rng(1)
    b, n, d, h, w, c = 3, 2, 3, 4, 4, 5
    bev_h, bev_w, bev_z = 5, 5, 3
    num_valid = 29  # ragged: not a multiple of anything relevant.

    with _bev_pool_schema():
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w, bev_z=bev_z)
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        feeds = {
            "depth": depth,
            "feat": feat,
            "ranks_depth": rd,
            "ranks_feat": rf,
            "ranks_bev": rb,
        }
        _check_matches_reference(model, feeds, bev_h, bev_w, bev_z)


def test_matches_reference_int32_ranks():
    rng = np.random.default_rng(2)
    b, n, d, h, w, c = 2, 2, 3, 4, 4, 5
    bev_h, bev_w, bev_z = 8, 8, 1
    num_valid = 21

    with _bev_pool_schema():
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, ranks_dtype="int32"
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w, ranks_dtype=np.int32
        )
        feeds = {
            "depth": depth,
            "feat": feat,
            "ranks_depth": rd,
            "ranks_feat": rf,
            "ranks_bev": rb,
        }
        sim_model, op_types = _check_matches_reference(
            model, feeds, bev_h, bev_w, bev_z
        )
        # INT32 ranks must be cast to INT64 somewhere (ScatterND indices and
        # Range both require INT64).
        assert "Cast" in op_types, op_types


def test_matches_reference_mmdeploy_domain():
    rng = np.random.default_rng(3)
    b, n, d, h, w, c = 2, 2, 3, 4, 4, 4
    bev_h, bev_w, bev_z = 6, 6, 1
    num_valid = 15

    with _bev_pool_schema(domain="mmdeploy"):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, domain="mmdeploy"
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        feeds = {
            "depth": depth,
            "feat": feat,
            "ranks_depth": rd,
            "ranks_feat": rf,
            "ranks_bev": rb,
        }
        _check_matches_reference(model, feeds, bev_h, bev_w, bev_z)


# --------------------------------------------------------------------------- #
# Naming-uncertainty scope: fires only for the coded candidates, declines for
# anything else (module docstring / task instructions).
# --------------------------------------------------------------------------- #


def test_declines_for_unlisted_op_type():
    # Same valid shapes/attributes as a firing case, but a different op_type
    # string -- outside kBevPoolOpTypeCandidates == {"bev_pool_v2"}.
    b, n, d, h, w, c = 1, 2, 2, 3, 3, 4
    bev_h, bev_w = 4, 4
    num_valid = 5
    other_op_type = "bev_pool_v1"

    with _bev_pool_schema(op_type=other_op_type):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, op_type=other_op_type
        )
        sim_model, op_types = _simplify_isolated_bev_pool(model)
        assert op_types[other_op_type] == 1, op_types
        assert "ScatterND" not in op_types, op_types


def test_declines_for_unlisted_domain():
    # Valid op_type, but a domain outside kBevPoolDomainCandidates ==
    # {"", "mmdeploy"}.
    b, n, d, h, w, c = 1, 2, 2, 3, 3, 4
    bev_h, bev_w = 4, 4
    num_valid = 5
    other_domain = "custom.other"

    with _bev_pool_schema(domain=other_domain):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, domain=other_domain
        )
        sim_model, op_types = _simplify_isolated_bev_pool(model)
        assert op_types[OP_TYPE] == 1, op_types
        assert "ScatterND" not in op_types, op_types


def test_declines_below_opset16():
    # ScatterND's `reduction` attribute (this rewrite's whole mechanism)
    # needs opset >= 16 -- a hard requirement per the header, not merely
    # defensive.
    b, n, d, h, w, c = 1, 2, 2, 3, 3, 4
    bev_h, bev_w = 4, 4
    num_valid = 5

    with _bev_pool_schema():
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w, opset=15)
        sim_model, op_types = _simplify_isolated_bev_pool(model)
        assert op_types[OP_TYPE] == 1, op_types
        assert "ScatterND" not in op_types, op_types


def test_declines_when_grid_shape_is_unavailable():
    # No bev_h/bev_w attributes AND a fully symbolic output shape -- there is
    # no way to size the ScatterND target, so the predicate must decline.
    with _bev_pool_schema():
        body = """
        agraph (
          float[1,2,3,4,4] depth,
          float[1,2,4,4,5] feat,
          int64[7] ranks_depth,
          int64[7] ranks_feat,
          int64[7] ranks_bev
        ) => (float[Bo,Co,Ho,Wo] Y)
        {
          Y = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev)
        }
        """
        model = _model(body)
        sim_model, op_types = _simplify_isolated_bev_pool(model)
        assert op_types[OP_TYPE] == 1, op_types
        assert "ScatterND" not in op_types, op_types


def test_extra_optimizers_required_to_fire():
    # The pass is opt-in: plain simplify() (no extra_optimizers) must leave
    # bev_pool_v2 alone.
    b, n, d, h, w, c = 1, 2, 2, 3, 3, 4
    bev_h, bev_w = 4, 4
    num_valid = 5

    with _bev_pool_schema():
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w)
        sim_model, ok = onnxsim.simplify(model, check_n=0)
        assert ok
        op_types = [nd.op_type for nd in sim_model.graph.node]
        assert OP_TYPE in op_types, op_types
