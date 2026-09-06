"""Tests for the opt-in ``rewrite_bev_pool_to_scatter`` pass
(``onnxsim/passes/rewrite_bev_pool_to_scatter.h``) -- decomposes
``bev_pool_v2``, the LSS-style (Lift-Splat-Shoot) camera-to-BEV voxel/feature
pooling op at the core of BEVDet's and BEVFusion's view transform, into a
subgraph built from standard ONNX ops centered on opset-16+
``ScatterND(reduction="add")``. This is an opt-in ``PassType::Other``
rewrite -- ``extra_optimizers=["rewrite_bev_pool_to_scatter"]``.

**Naming uncertainty** (see the pass's own header comment for the full
explanation): ``bev_pool_v2`` ships as a bespoke CUDA op / TensorRT plugin in
BEVDet's own deployment tooling rather than as part of mmdeploy's own op
set, so the exact ``op_type``/``domain`` a real export uses is not confirmed
here -- unlike this codebase's other mmdeploy/mmcv op-decomposition passes,
whose contracts were read directly off known source. This file, like the
pass itself, targets ``op_type == "bev_pool_v2"`` in domain ``""`` or
``"mmdeploy"``; the underlying gather/gather/mul/scatter-add algorithm
(independently checked against ``bev_pool_v2_reference`` below) is the part
of this pass with real, verifiable ground truth, and is what these tests
mostly exercise.

**Why this test file can't use onnxsim's usual pre/post equivalence
check**: ``bev_pool_v2`` has no ONNX Runtime kernel and no ``onnx``
reference-evaluator kernel -- there is no way to execute the *original*
graph at all, so every call below passes ``check_n=0``. Instead, each test
implements an independent NumPy reference of the op's documented algorithm
(``bev_pool_v2_reference`` below -- gather + multiply + per-batch
grouped-sum) and compares the *simplified* graph's output -- executed via
``onnx.reference.ReferenceEvaluator`` (every op this pass emits --
``Reshape``/``Gather``/``Mul``/``Range``/``Expand``/``ScatterND``/... -- has
a reference-evaluator kernel, so no optional ``onnxruntime`` dependency is
needed here) -- against that independent reference.

Since ``bev_pool_v2`` has no built-in ONNX schema either, every model built
here registers a minimal one via ``onnx.defs.register_schema`` (mirroring
``tests/test_python_api.py``'s ``_register_custom_onnx_schema`` and
``tests/test_msdeformattn_to_gridsample.py``'s ``_register_schema``) so the
graph passes onnxsim's/onnx's structural validation on the way in.

Models are built with the ONNX text format parser (``onnx.parser``), per
CLAUDE.md's convention for this repo's tests.
"""

import collections
import contextlib

import numpy as np
import onnx
import pytest
from onnx import parser
from onnx.reference import ReferenceEvaluator

import onnxsim

OP_TYPE = "bev_pool_v2"


# ---------------------------------------------------------------------------
# Custom-op schema registration (bev_pool_v2 has no built-in ONNX schema).
# ---------------------------------------------------------------------------


def _register_schema(domain, since_version=1):
    OpSchema = onnx.defs.OpSchema
    schema = OpSchema(
        OP_TYPE,
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
                "interval_starts (unused by this rewrite)",
                param_option=OpSchema.FormalParameterOption.Optional,
            ),
            OpSchema.FormalParameter(
                "interval_lengths",
                "T1",
                "interval_lengths (unused by this rewrite)",
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
def _bev_pool_schema(domain):
    # A custom op with no registered schema fails onnxsim's/onnx's own
    # structural validation before the rewrite pass ever gets a chance to
    # run -- register one for the duration of the test, exactly like
    # tests/test_python_api.py's ``test_custom_op_with_registered_schema_is_simplified``.
    _register_schema(domain)
    try:
        yield
    finally:
        onnx.defs.deregister_schema(OP_TYPE, 1, domain)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _model(body, initializer=(), opset=16, ir_version=10, extra_opsets=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_opsets}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
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
    opset=16,
    static_output_shape=True,
    with_intervals=False,
):
    """Builds a single-node graph: `depth, feat, ranks_* [, intervals] ->
    bev_pool_v2 -> Y`.

    ``static_output_shape=True`` declares Y's shape with concrete ints
    (exercising ``DetermineBevGridShape``'s output-shape-based source);
    ``False`` declares it with symbolic dims instead and attaches
    ``bev_h``/``bev_w``/``bev_z`` attributes (exercising the attribute
    fallback).
    """
    op = f"{domain}.{OP_TYPE}" if domain else OP_TYPE
    extra_opsets = f', "{domain}": 1' if domain else ""

    if static_output_shape:
        out_shape = (
            f"{b},{c},{bev_h},{bev_w}"
            if bev_z == 1
            else f"{b},{c},{bev_z},{bev_h},{bev_w}"
        )
        attrs = ""
    else:
        out_shape = "Bo,Co,Ho,Wo" if bev_z == 1 else "Bo,Co,Zo,Ho,Wo"
        attrs = f" <bev_h={bev_h}, bev_w={bev_w}, bev_z={bev_z}>"

    extra_inputs = ""
    extra_args = ""
    if with_intervals:
        extra_inputs = (
            f", {ranks_dtype}[2] interval_starts, {ranks_dtype}[2] interval_lengths"
        )
        extra_args = ", interval_starts, interval_lengths"

    body = f"""
    agraph (
      float[{b},{n},{d},{h},{w}] depth,
      float[{b},{n},{h},{w},{c}] feat,
      {ranks_dtype}[{num_valid}] ranks_depth,
      {ranks_dtype}[{num_valid}] ranks_feat,
      {ranks_dtype}[{num_valid}] ranks_bev{extra_inputs}
    ) => (float[{out_shape}] Y)
    {{
      Y = {op}{attrs}(depth, feat, ranks_depth, ranks_feat, ranks_bev{extra_args})
    }}
    """
    return _model(body, opset=opset, extra_opsets=extra_opsets)


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


# ---------------------------------------------------------------------------
# Independent NumPy reference implementation (see module docstring).
# ---------------------------------------------------------------------------


def bev_pool_v2_reference(
    depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_h, bev_w, bev_z=1
):
    """depth: (B,N,D,H,W) float. feat: (B,N,H,W,C) float. ranks_depth/
    ranks_feat/ranks_bev: (num_valid,) int, SAME for every batch item.
    Returns (B,C,H,W) if bev_z==1 else (B,C,Z,H,W), per the op's documented
    algorithm: for every valid (camera,depth-bin,pixel) triple, gather a
    depth value and a feature vector, multiply, and scatter-add into the
    output BEV grid.
    """
    b, n, d, h, w = depth.shape
    c = feat.shape[-1]
    out = np.zeros((b, bev_z * bev_h * bev_w, c), dtype=np.float32)
    for bi in range(b):
        depth_flat = depth[bi].reshape(n * d * h * w)
        feat_flat = feat[bi].reshape(n * h * w, c)
        d_vals = depth_flat[ranks_depth]
        f_vals = feat_flat[ranks_feat]
        contrib = d_vals[:, None] * f_vals
        np.add.at(out[bi], ranks_bev, contrib)
    out = out.reshape(b, bev_z, bev_h, bev_w, c).transpose(0, 4, 1, 2, 3)
    if bev_z == 1:
        out = out[:, :, 0, :, :]
    return out


def _simplify_and_check(model, feeds, bev_h, bev_w, bev_z=1, rtol=1e-4, atol=1e-4):
    sim_model, _ = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["rewrite_bev_pool_to_scatter"]
    )
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert OP_TYPE not in op_types, op_types
    assert "ScatterND" in op_types, op_types

    expected = bev_pool_v2_reference(
        feeds["depth"],
        feeds["feat"],
        feeds["ranks_depth"].astype(np.int64),
        feeds["ranks_feat"].astype(np.int64),
        feeds["ranks_bev"].astype(np.int64),
        bev_h,
        bev_w,
        bev_z,
    )
    evaluator = ReferenceEvaluator(sim_model)
    actual = evaluator.run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return sim_model, op_types


def _feeds(depth, feat, ranks_depth, ranks_feat, ranks_bev, with_intervals=False):
    feeds = {
        "depth": depth,
        "feat": feat,
        "ranks_depth": ranks_depth,
        "ranks_feat": ranks_feat,
        "ranks_bev": ranks_bev,
    }
    if with_intervals:
        dtype = ranks_depth.dtype
        feeds["interval_starts"] = np.array([0, 1], dtype=dtype)
        feeds["interval_lengths"] = np.array([1, 1], dtype=dtype)
    return feeds


# --------------------------------------------------------------------------- #
# Single/multi batch, single-level (bev_z=1, rank-4 output).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("domain", ["", "mmdeploy"])
def test_single_batch_item(domain):
    rng = np.random.default_rng(0)
    b, n, d, h, w, c = 1, 3, 4, 5, 6, 8
    bev_h, bev_w, bev_z = 7, 9, 1
    num_valid = 13

    with _bev_pool_schema(domain):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, domain=domain
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


def test_multiple_batch_items():
    rng = np.random.default_rng(1)
    b, n, d, h, w, c = 4, 2, 3, 5, 5, 6
    bev_h, bev_w, bev_z = 10, 10, 1
    num_valid = 40

    with _bev_pool_schema(""):
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w)
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


# --------------------------------------------------------------------------- #
# Multi-level BEV grid (bev_z > 1, rank-5 output).
# --------------------------------------------------------------------------- #


def test_multi_level_bev_grid():
    rng = np.random.default_rng(2)
    b, n, d, h, w, c = 2, 2, 5, 4, 4, 6
    bev_h, bev_w, bev_z = 6, 6, 4
    num_valid = 37  # ragged: not a multiple of D*H*W, N*H*W, or bev_z*bev_h*bev_w.

    with _bev_pool_schema("mmdeploy"):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, bev_z=bev_z, domain="mmdeploy"
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


# --------------------------------------------------------------------------- #
# Varying N (cameras) / D (depth bins) / C (feature channels).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,d,c", [(1, 1, 1), (2, 8, 3), (6, 2, 16), (5, 5, 5)])
def test_varying_cameras_depth_bins_channels(n, d, c):
    rng = np.random.default_rng(3)
    b, h, w = 2, 4, 4
    bev_h, bev_w, bev_z = 5, 5, 1
    num_valid = 17

    with _bev_pool_schema(""):
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w)
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


# --------------------------------------------------------------------------- #
# A genuinely ragged num_valid count, on top of what test_multi_level_bev_grid
# already exercises -- no hidden assumption that num_valid evenly tiles
# anything.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_valid", [1, 3, 11, 100])
def test_arbitrary_num_valid_counts(num_valid):
    rng = np.random.default_rng(4)
    b, n, d, h, w, c = 2, 3, 4, 4, 4, 4
    bev_h, bev_w, bev_z = 6, 7, 1

    with _bev_pool_schema(""):
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w)
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


# --------------------------------------------------------------------------- #
# INT32 ranks (the schema/predicate accept either INT32 or INT64).
# --------------------------------------------------------------------------- #


def test_int32_ranks_dtype():
    rng = np.random.default_rng(5)
    b, n, d, h, w, c = 2, 2, 3, 4, 4, 5
    bev_h, bev_w, bev_z = 8, 8, 1
    num_valid = 21

    with _bev_pool_schema(""):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, ranks_dtype="int32"
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w, ranks_dtype=np.int32
        )
        sim_model, op_types = _simplify_and_check(
            model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z
        )
        # INT32 ranks must be cast to INT64 somewhere along the way (ScatterND
        # indices and Range require INT64).
        assert "Cast" in op_types, op_types


# --------------------------------------------------------------------------- #
# bev_h/bev_w/bev_z attribute fallback (output shape declared symbolic).
# --------------------------------------------------------------------------- #


def test_bev_grid_shape_from_attributes_when_output_shape_is_symbolic():
    rng = np.random.default_rng(6)
    b, n, d, h, w, c = 2, 2, 3, 4, 4, 5
    bev_h, bev_w, bev_z = 8, 8, 1
    num_valid = 21

    with _bev_pool_schema(""):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, static_output_shape=False
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


def test_bev_grid_shape_from_attributes_multi_level():
    rng = np.random.default_rng(7)
    b, n, d, h, w, c = 1, 2, 3, 4, 4, 4
    bev_h, bev_w, bev_z = 5, 5, 3
    num_valid = 10

    with _bev_pool_schema(""):
        model = _bev_pool_model(
            b,
            n,
            d,
            h,
            w,
            c,
            num_valid,
            bev_h,
            bev_w,
            bev_z=bev_z,
            static_output_shape=False,
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        _simplify_and_check(model, _feeds(depth, feat, rd, rf, rb), bev_h, bev_w, bev_z)


# --------------------------------------------------------------------------- #
# Optional interval_starts/interval_lengths inputs are accepted but ignored.
# --------------------------------------------------------------------------- #


def test_interval_starts_and_lengths_present_but_ignored():
    rng = np.random.default_rng(8)
    b, n, d, h, w, c = 2, 2, 3, 4, 4, 5
    bev_h, bev_w, bev_z = 6, 6, 1
    num_valid = 12

    with _bev_pool_schema(""):
        model = _bev_pool_model(
            b, n, d, h, w, c, num_valid, bev_h, bev_w, with_intervals=True
        )
        depth, feat, rd, rf, rb = _rand_inputs(
            rng, b, n, d, h, w, c, num_valid, bev_z, bev_h, bev_w
        )
        feeds = _feeds(depth, feat, rd, rf, rb, with_intervals=True)
        sim_model, _ = _simplify_and_check(model, feeds, bev_h, bev_w, bev_z)

        # Garbage interval_starts/interval_lengths values must not change the
        # result at all -- they are documented as entirely unread.
        feeds2 = dict(feeds)
        feeds2["interval_starts"] = np.array([999, -999], dtype=np.int64)
        feeds2["interval_lengths"] = np.array([-1, 12345], dtype=np.int64)
        evaluator = ReferenceEvaluator(sim_model)
        actual1 = evaluator.run(None, feeds)[0]
        actual2 = evaluator.run(None, feeds2)[0]
        np.testing.assert_array_equal(actual1, actual2)


# --------------------------------------------------------------------------- #
# Negative tests: predicate must decline outside this pass's documented scope.
# --------------------------------------------------------------------------- #


def test_declines_below_opset16():
    """ScatterND's `reduction` attribute (this rewrite's whole mechanism)
    needs opset >= 16 -- a hard requirement, not merely defensive."""
    b, n, d, h, w, c = 1, 2, 2, 3, 3, 4
    bev_h, bev_w = 4, 4
    num_valid = 5

    with _bev_pool_schema(""):
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w, opset=15)
        sim_model, ok = onnxsim.simplify(
            model, check_n=0, extra_optimizers=["rewrite_bev_pool_to_scatter"]
        )
        assert ok
        op_types = [nd.op_type for nd in sim_model.graph.node]
        assert OP_TYPE in op_types, op_types
        assert "ScatterND" not in op_types, op_types


def test_declines_when_grid_shape_is_unavailable():
    """No bev_h/bev_w attributes AND a fully symbolic output shape -- there
    is no way to size the ScatterND target, so the predicate must decline."""
    with _bev_pool_schema(""):
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
        sim_model, ok = onnxsim.simplify(
            model, check_n=0, extra_optimizers=["rewrite_bev_pool_to_scatter"]
        )
        assert ok
        op_types = [nd.op_type for nd in sim_model.graph.node]
        assert OP_TYPE in op_types, op_types
        assert "ScatterND" not in op_types, op_types


def test_declines_when_depth_is_not_rank5():
    with _bev_pool_schema(""):
        body = """
        agraph (
          float[1,24,4,4] depth,
          float[1,2,4,4,5] feat,
          int64[7] ranks_depth,
          int64[7] ranks_feat,
          int64[7] ranks_bev
        ) => (float[1,5,4,4] Y)
        {
          Y = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev)
        }
        """
        model = _model(body)
        sim_model, ok = onnxsim.simplify(
            model, check_n=0, extra_optimizers=["rewrite_bev_pool_to_scatter"]
        )
        assert ok
        op_types = [nd.op_type for nd in sim_model.graph.node]
        assert OP_TYPE in op_types, op_types
        assert "ScatterND" not in op_types, op_types


def test_extra_optimizers_required_to_fire():
    """The pass is opt-in: plain simplify() must leave bev_pool_v2 alone."""
    b, n, d, h, w, c = 1, 2, 2, 3, 3, 4
    bev_h, bev_w = 4, 4
    num_valid = 5

    with _bev_pool_schema(""):
        model = _bev_pool_model(b, n, d, h, w, c, num_valid, bev_h, bev_w)
        sim_model, ok = onnxsim.simplify(model, check_n=0)
        assert ok
        op_types = [nd.op_type for nd in sim_model.graph.node]
        assert OP_TYPE in op_types, op_types
