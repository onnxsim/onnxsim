"""Tests for the opt-in ``rewrite_msdeformattn_to_gridsample`` pass.

``MMCVMultiScaleDeformableAttention`` (mmdeploy/mmcv's custom deformable-
attention op, used by BEVFormer/Deformable-DETR-style exports) has no ONNX
Runtime kernel and no ``onnx`` reference-evaluator kernel -- it is not a real
ONNX operator, just a custom op mmdeploy's exporter emits. That means
onnxsim's usual automatic ``check_n``/``input_data`` equivalence check inside
``simplify()`` (which runs the *original* graph to compare against) cannot be
used here: there is nothing that can execute the original node. So this file
does not rely on ``check_n``/``input_data`` or ``simplify()``'s returned
``check_ok`` at all (``check_n`` already defaults to 0, i.e. "don't try").

Instead, correctness is established independently: a small NumPy
implementation of mmcv's own pure-PyTorch fallback,
``multi_scale_deformable_attn_pytorch`` (transcribed faithfully from mmcv's
source -- see the reference function below and the derivation in
``onnxsim/passes/rewrite_msdeformattn_to_gridsample.h``), computes the
expected output from the same random inputs. The bilinear-sampling step
itself delegates to a real ``GridSample`` node run through
``onnx.reference.ReferenceEvaluator`` (already covered end-to-end by
``tests/test_gridsample_to_gather.py``), rather than a second hand-rolled
bilinear-sampling implementation, so this reference exercises the same
split/reshape/transpose/attention-weighting algorithm the rewrite performs
without silently sharing a bug with the C++ pass under test. The SIMPLIFIED
graph's output (which the pass rewrites into ``Split``/``Reshape``/
``Transpose``/``Gather``/``Concat``/``GridSample``/``ReduceSum`` -- ordinary
ONNX ops any evaluator can run) is then compared against that NumPy
reference with ``np.testing.assert_allclose``.

Since ``MMCVMultiScaleDeformableAttention`` has no built-in ONNX schema
either, every model built here registers a minimal one via
``onnx.defs.register_schema`` (mirroring
``tests/test_python_api.py``'s ``_register_custom_onnx_schema`` /
``test_custom_op_with_registered_schema_is_simplified``) so the graph passes
onnxsim's/onnx's structural validation on the way in; ``onnxsim.simplify``'s
default ``import_custom_schemas=True`` then bridges it into onnxsim's own
schema registry.

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

OP_TYPE = "MMCVMultiScaleDeformableAttention"

# A single reusable GridSample model/evaluator (mode="linear",
# padding_mode="zeros", align_corners=0 -- exactly what the rewrite itself
# emits per level, and exactly mmcv's own
# ``F.grid_sample(..., mode='bilinear', padding_mode='zeros',
# align_corners=False)`` call) used as the bilinear-sampling primitive for
# the NumPy reference below, instead of a second hand-rolled implementation.
_GRIDSAMPLE_MODEL = parser.parse_model(
    """
    <
      ir_version: 10,
      opset_import: ["": 20]
    >
    grid_sample_ref (float[N,C,H,W] X, float[N,Ho,Wo,2] grid) => (float[N,C,Ho,Wo] Y)
    {
      Y = GridSample <mode="linear", padding_mode="zeros", align_corners=0> (X, grid)
    }
    """
)
_GRIDSAMPLE_EVALUATOR = ReferenceEvaluator(_GRIDSAMPLE_MODEL)


def _grid_sample_bilinear_zeros(x, grid):
    return _GRIDSAMPLE_EVALUATOR.run(None, {"X": x, "grid": grid})[0]


def multi_scale_deformable_attn_reference(
    value, spatial_shapes, sampling_locations, attention_weights
):
    """Transcription of mmcv's ``multi_scale_deformable_attn_pytorch``:

    def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                             sampling_locations,
                                             attention_weights):
        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ = \\
            sampling_locations.shape
        value_list = value.split(
            [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(value_spatial_shapes):
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(
                1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(
                value_l_, sampling_grid_l_, mode='bilinear',
                padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        attention_weights = attention_weights.transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_levels * num_points)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
                  * attention_weights).sum(-1).view(
            bs, num_heads * embed_dims, num_queries)
        return output.transpose(1, 2).contiguous()
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    split_sizes = [int(h) * int(w) for h, w in spatial_shapes]
    offsets = np.cumsum(split_sizes)[:-1]
    value_list = np.split(value, offsets, axis=1)

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):
        H_, W_ = int(H_), int(W_)
        value_l_ = (
            value_list[level]
            .reshape(bs, H_ * W_, num_heads * embed_dims)
            .transpose(0, 2, 1)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level]
            .transpose(0, 2, 1, 3, 4)
            .reshape(bs * num_heads, num_queries, num_points, 2)
        )
        sampled = _grid_sample_bilinear_zeros(
            value_l_.astype(np.float32), sampling_grid_l_.astype(np.float32)
        )
        sampling_value_list.append(sampled)  # (bs*M, D, nq, P)

    attn = attention_weights.transpose(0, 2, 1, 3, 4).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    stacked = np.stack(sampling_value_list, axis=-2)  # (bs*M, D, nq, L, P)
    stacked = stacked.reshape(
        bs * num_heads, embed_dims, num_queries, num_levels * num_points
    )
    output = (stacked * attn).sum(-1)  # (bs*M, D, nq)
    output = output.reshape(bs, num_heads * embed_dims, num_queries)
    return output.transpose(0, 2, 1).astype(np.float32)  # (bs, nq, M*D)


def _register_schema(domain, since_version=1):
    OpSchema = onnx.defs.OpSchema
    schema = OpSchema(
        OP_TYPE,
        domain,
        since_version,
        inputs=[
            OpSchema.FormalParameter("value", "T", "value"),
            OpSchema.FormalParameter("spatial_shapes", "T1", "spatial_shapes"),
            OpSchema.FormalParameter(
                "level_start_index", "T1", "level_start_index"
            ),
            OpSchema.FormalParameter(
                "sampling_locations", "T", "sampling_locations"
            ),
            OpSchema.FormalParameter(
                "attention_weights", "T", "attention_weights"
            ),
        ],
        outputs=[OpSchema.FormalParameter("output", "T", "output")],
        type_constraints=[
            ("T", ["tensor(float)"], "Constrain to float tensors."),
            ("T1", ["tensor(int64)"], "Constrain to int64 tensors."),
        ],
        attributes=[
            OpSchema.Attribute(
                "im2col_step",
                OpSchema.AttrType.INT,
                "CUDA-kernel batching knob, no effect on output values",
                required=False,
            ),
        ],
    )
    onnx.defs.register_schema(schema)


@contextlib.contextmanager
def _msda_schema(domain):
    # A custom op with no registered schema fails onnxsim's/onnx's own
    # structural validation before the rewrite pass ever gets a chance to
    # run (see this file's module docstring) -- register one for the
    # duration of the test, exactly like
    # tests/test_python_api.py's ``test_custom_op_with_registered_schema_is_simplified``.
    _register_schema(domain)
    try:
        yield
    finally:
        onnx.defs.deregister_schema(OP_TYPE, 1, domain)


def _model(
    bs,
    num_keys,
    num_queries,
    M,
    D,
    spatial_shapes,
    P,
    domain="",
    spatial_shapes_dim0=None,
    opset=20,
    ir_version=10,
):
    """Builds a single-node model wrapping ``MMCVMultiScaleDeformableAttention``.

    ``bs``/``num_keys``/``num_queries`` may be an int or a symbolic-dim name
    (str) -- only ``M``/``D``/``L`` (``len(spatial_shapes)``)/``P`` need be
    static per the op's own requirements. ``spatial_shapes_dim0`` overrides
    ``spatial_shapes``'s own first axis size in the declared shape (letting a
    test declare it as an unrelated dynamic symbol while still handing a
    concrete tensor of ``len(spatial_shapes)`` rows at runtime) -- used by
    the "unknown L" negative test.
    """
    L = len(spatial_shapes)
    MD = M * D
    L_dim = spatial_shapes_dim0 if spatial_shapes_dim0 is not None else L

    opset_imports = [f'"": {opset}']
    domain_prefix = ""
    if domain:
        opset_imports.append(f'"{domain}": 1')
        domain_prefix = f"{domain}."

    body = f"""
    <
      ir_version: {ir_version},
      opset_import: [{", ".join(opset_imports)}]
    >
    agraph (
      float[{bs},{num_keys},{M},{D}] value,
      int64[{L_dim},2] spatial_shapes,
      int64[{L_dim}] level_start_index,
      float[{bs},{num_queries},{M},{L},{P},2] sampling_locations,
      float[{bs},{num_queries},{M},{L},{P}] attention_weights
    ) => (float[{bs},{num_queries},{MD}] Y)
    {{
      Y = {domain_prefix}{OP_TYPE}(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
    }}
    """
    return parser.parse_model(body)


def _rand_inputs(rng, bs, num_keys, num_queries, M, D, spatial_shapes, P):
    assert sum(h * w for h, w in spatial_shapes) == num_keys
    value = rng.standard_normal((bs, num_keys, M, D)).astype(np.float32)
    spatial_shapes_arr = np.array(spatial_shapes, dtype=np.int64)
    level_start_index = np.concatenate(
        [[0], np.cumsum([h * w for h, w in spatial_shapes])[:-1]]
    ).astype(np.int64)
    # Values in [0, 1] per the op's spec, with a little overspill on both
    # ends so zero-padding at the sampled feature maps' borders is actually
    # exercised, not just in-bounds bilinear sampling.
    sampling_locations = rng.uniform(
        -0.1, 1.1, size=(bs, num_queries, M, len(spatial_shapes), P, 2)
    ).astype(np.float32)
    attention_weights = rng.uniform(
        0.0, 1.0, size=(bs, num_queries, M, len(spatial_shapes), P)
    ).astype(np.float32)
    return value, spatial_shapes_arr, level_start_index, sampling_locations, attention_weights


def _simplify_and_check(model, feeds, rtol=1e-4, atol=1e-5):
    sim_model, _ = onnxsim.simplify(
        model,
        extra_optimizers=["rewrite_msdeformattn_to_gridsample"],
    )
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert OP_TYPE not in op_types, op_types
    assert "GridSample" in op_types, op_types

    expected = multi_scale_deformable_attn_reference(
        feeds["value"],
        feeds["spatial_shapes"],
        feeds["sampling_locations"],
        feeds["attention_weights"],
    )
    evaluator = ReferenceEvaluator(sim_model)
    actual = evaluator.run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# Multi-level (BEVFormer-style multi-scale FPN: 4 levels, distinct H_l/W_l).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("domain", ["", "mmdeploy"])
def test_multi_level(domain):
    rng = np.random.RandomState(0)
    spatial_shapes = [(16, 24), (8, 12), (4, 6), (2, 3)]
    bs, M, D, P = 2, 4, 8, 4
    num_keys = sum(h * w for h, w in spatial_shapes)
    num_queries = 5

    with _msda_schema(domain):
        model = _model(
            bs, num_keys, num_queries, M, D, spatial_shapes, P, domain=domain
        )
        value, ss, lsi, loc, attn = _rand_inputs(
            rng, bs, num_keys, num_queries, M, D, spatial_shapes, P
        )
        feeds = {
            "value": value,
            "spatial_shapes": ss,
            "level_start_index": lsi,
            "sampling_locations": loc,
            "attention_weights": attn,
        }
        _simplify_and_check(model, feeds)


# --------------------------------------------------------------------------- #
# Single level (L=1) -- the BEV self-attention case.
# --------------------------------------------------------------------------- #


def test_single_level():
    rng = np.random.RandomState(1)
    spatial_shapes = [(10, 15)]
    bs, M, D, P = 1, 8, 32, 4
    num_keys = sum(h * w for h, w in spatial_shapes)
    num_queries = 7

    with _msda_schema(""):
        model = _model(bs, num_keys, num_queries, M, D, spatial_shapes, P)
        value, ss, lsi, loc, attn = _rand_inputs(
            rng, bs, num_keys, num_queries, M, D, spatial_shapes, P
        )
        feeds = {
            "value": value,
            "spatial_shapes": ss,
            "level_start_index": lsi,
            "sampling_locations": loc,
            "attention_weights": attn,
        }
        _simplify_and_check(model, feeds)


# --------------------------------------------------------------------------- #
# Varying M/D/P.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M,D,P", [(1, 4, 1), (2, 16, 2), (8, 4, 8), (3, 5, 6)]
)
def test_varying_heads_dims_points(M, D, P):
    rng = np.random.RandomState(2)
    spatial_shapes = [(6, 6), (3, 4)]
    bs = 2
    num_keys = sum(h * w for h, w in spatial_shapes)
    num_queries = 3

    with _msda_schema(""):
        model = _model(bs, num_keys, num_queries, M, D, spatial_shapes, P)
        value, ss, lsi, loc, attn = _rand_inputs(
            rng, bs, num_keys, num_queries, M, D, spatial_shapes, P
        )
        feeds = {
            "value": value,
            "spatial_shapes": ss,
            "level_start_index": lsi,
            "sampling_locations": loc,
            "attention_weights": attn,
        }
        _simplify_and_check(model, feeds)


# --------------------------------------------------------------------------- #
# Dynamic (symbolic) bs / num_queries / num_keys -- the rewrite must not
# secretly require any of these to be statically known.
# --------------------------------------------------------------------------- #


def test_dynamic_batch_queries_keys():
    rng = np.random.RandomState(3)
    spatial_shapes = [(5, 5), (2, 3)]
    M, D, P = 2, 6, 3
    bs, num_queries = 2, 4
    num_keys = sum(h * w for h, w in spatial_shapes)

    with _msda_schema(""):
        model = _model(
            "N", "K", "Q", M, D, spatial_shapes, P
        )  # symbolic bs, num_keys, num_queries
        value, ss, lsi, loc, attn = _rand_inputs(
            rng, bs, num_keys, num_queries, M, D, spatial_shapes, P
        )
        feeds = {
            "value": value,
            "spatial_shapes": ss,
            "level_start_index": lsi,
            "sampling_locations": loc,
            "attention_weights": attn,
        }
        _simplify_and_check(model, feeds)


# --------------------------------------------------------------------------- #
# Negative tests: out-of-scope inputs must leave the node untouched.
# --------------------------------------------------------------------------- #


def test_declines_mismatched_num_heads():
    # value's M (=4) disagrees with sampling_locations'/attention_weights' M
    # (=5) -- the predicate must decline rather than build a subgraph on
    # inconsistent assumptions.
    domain = ""
    with _msda_schema(domain):
        body = """
        <
          ir_version: 10,
          opset_import: ["": 20]
        >
        agraph (
          float[1,36,4,8] value,
          int64[1,2] spatial_shapes,
          int64[1] level_start_index,
          float[1,3,5,1,4,2] sampling_locations,
          float[1,3,5,1,4] attention_weights
        ) => (float[1,3,40] Y)
        {
          Y = MMCVMultiScaleDeformableAttention(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        }
        """
        model = parser.parse_model(body)
        sim_model, _ = onnxsim.simplify(
            model, extra_optimizers=["rewrite_msdeformattn_to_gridsample"]
        )
        op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
        assert OP_TYPE in op_types, op_types
        assert "GridSample" not in op_types, op_types


def test_declines_unknown_num_levels():
    # spatial_shapes' own first axis (num_levels) is not statically known --
    # the pass cannot unroll a per-level loop of unknown length, so it must
    # decline even though every other dim is static and self-consistent.
    domain = ""
    with _msda_schema(domain):
        model = _model(
            bs=1,
            num_keys=6,
            num_queries=2,
            M=2,
            D=4,
            spatial_shapes=[(2, 3)],
            P=3,
            spatial_shapes_dim0="L",
        )
        sim_model, _ = onnxsim.simplify(
            model, extra_optimizers=["rewrite_msdeformattn_to_gridsample"]
        )
        op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
        assert OP_TYPE in op_types, op_types
        assert "GridSample" not in op_types, op_types
