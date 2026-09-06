"""Tests for onnxsim's built-in schemas for the mmdeploy/mmcv/BEVDet custom
ops decomposed by this branch's rewrite_msdeformattn_to_gridsample,
rewrite_deform_conv_to_gather, rewrite_trt_batched_nms,
rewrite_trt_batched_rotated_nms, and rewrite_bev_pool_to_scatter passes
(onnxsim/bev_custom_op_schemas.cpp).

Unlike those five passes' own test files, none of these tests pass
``extra_optimizers`` or enable any rewrite pass, and none of them call
``onnx.defs.register_schema`` themselves (the pattern every one of those five
test files' ``_model()`` helpers relies on to get past its own upfront
``onnx.checker.check_model()`` call). The point here is exactly the opposite:
proving onnxsim itself, out of the box, already knows enough about these ops
-- via the schemas ``RegisterBevCustomOpSchemas()`` registers internally --
to run real shape inference through them.

The proof technique: build a graph with one custom-op node followed by a
``Shape``/``Gather`` chain that reads one dimension of the custom op's own
output. If onnxsim's shape inference can determine that dimension from the
custom op's *inputs* via a real ``TypeAndShapeInferenceFunction``, its own
constant-folding pass collapses the ``Shape``/``Gather`` chain into a literal
-- something that could not happen if the custom op's output shape were left
unresolved (the pre-existing, already-adequate-for-passing-the-checker
status quo without any of these schemas: `RegisterCustomDefaultDomainOpSchemas`
for the default domain, and onnx::checker::check_model's own tolerance for
unknown non-default-domain ops -- see bev_custom_op_schemas.h's own comment).
So a folded-to-a-literal output is not just "the model didn't get rejected";
it is a real, non-trivial re-derivation that could only have come from these
schemas' inference functions actually running.
"""

from onnx import numpy_helper, parser

import onnxsim


def _model(body, opset=17, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "mmdeploy": 1]
        >
        {body}
        """
    )


def _folded_dim(model):
    """Simplify with no extra optimizers/rewrite passes and return the sole
    remaining output's folded scalar value, asserting the graph collapsed to
    a pure initializer (proof the Shape/Gather chain was fully constant-
    folded, not left as live nodes)."""
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    assert sim_model.graph.node == [], [n.op_type for n in sim_model.graph.node]
    assert len(sim_model.graph.initializer) == 1
    return int(numpy_helper.to_array(sim_model.graph.initializer[0])[0])


def _shape_gather_body(domain_prefix, custom_op_body, axis):
    return f"""
    agraph {custom_op_body[0]} => (int64[1] out_dim)
    {{
      attn_out = {domain_prefix}{custom_op_body[1]}
      shp = Shape(attn_out)
      idx = Constant<value = int64[1] {{{axis}}}>()
      out_dim = Gather<axis = 0>(shp, idx)
    }}
    """


def test_msdeformattn_output_shape_inferred_from_inputs():
    # M=4, D=8 -> dim 2 of the output should fold to M*D=32.
    model = _model(
        _shape_gather_body(
            "mmdeploy.",
            (
                "(float[2,100,4,8] value, int64[3,2] spatial_shapes, "
                "int64[3] level_start_index, "
                "float[2,10,4,3,2,2] sampling_locations, "
                "float[2,10,4,3,2] attention_weights)",
                "MMCVMultiScaleDeformableAttention (value, spatial_shapes, "
                "level_start_index, sampling_locations, attention_weights)",
            ),
            axis=2,
        )
    )
    assert _folded_dim(model) == 4 * 8


def test_msdeformattn_default_domain_also_inferred():
    model = _model(
        _shape_gather_body(
            "",
            (
                "(float[1,50,2,16] value, int64[2,2] spatial_shapes, "
                "int64[2] level_start_index, "
                "float[1,5,2,2,4,2] sampling_locations, "
                "float[1,5,2,2,4] attention_weights)",
                "MMCVMultiScaleDeformableAttention (value, spatial_shapes, "
                "level_start_index, sampling_locations, attention_weights)",
            ),
            axis=2,
        )
    )
    assert _folded_dim(model) == 2 * 16


def test_deform_conv2d_output_spatial_dims_from_offset():
    # offset's own (Hout, Wout) = (6, 7) should flow straight to the output.
    model = _model(
        _shape_gather_body(
            "mmdeploy.",
            (
                "(float[1,8,10,10] input, float[1,18,6,7] offset, "
                "float[4,8,3,3] weight)",
                "MMCVDeformConv2d (input, offset, weight)",
            ),
            axis=3,
        )
    )
    assert _folded_dim(model) == 7


def test_modulated_deform_conv2d_cout_from_weight():
    model = _model(
        _shape_gather_body(
            "mmdeploy.",
            (
                "(float[1,4,10,10] input, float[1,18,8,8] offset, "
                "float[1,9,8,8] mask, float[6,4,3,3] weight)",
                "MMCVModulatedDeformConv2d (input, offset, mask, weight)",
            ),
            axis=1,
        )
    )
    assert _folded_dim(model) == 6


def test_trt_batched_nms_num_detections_batch_dim():
    body = """
    agraph (float[2,50,1,4] boxes, float[2,50,3] scores) => (int64[1] out_dim)
    {
      num_det, nmsed_boxes, nmsed_scores, nmsed_classes = mmdeploy.TRTBatchedNMS
          <topK = 20, keepTopK = 10, scoreThreshold = 0.0, iouThreshold = 0.5>
          (boxes, scores)
      shp = Shape(nmsed_boxes)
      idx = Constant<value = int64[1] {1}>()
      out_dim = Gather<axis = 0>(shp, idx)
    }
    """
    model = _model(body)
    assert _folded_dim(model) == 10


def test_trt_batched_rotated_nms_nmsed_boxes_width():
    body = """
    agraph (float[1,30,1,5] boxes, float[1,30,2] scores) => (int64[1] out_dim)
    {
      num_det, nmsed_boxes, nmsed_scores, nmsed_classes = mmdeploy.TRTBatchedRotatedNMS
          <topK = 15, keepTopK = 5, scoreThreshold = 0.0, iouThreshold = 0.5>
          (boxes, scores)
      shp = Shape(nmsed_boxes)
      idx = Constant<value = int64[1] {2}>()
      out_dim = Gather<axis = 0>(shp, idx)
    }
    """
    model = _model(body)
    assert _folded_dim(model) == 5  # rotated boxes carry 5 values, not 4


def test_bev_pool_v2_grid_shape_from_attributes():
    body = """
    agraph (float[2,3,4,5,6] depth, float[2,3,5,6,7] feat,
            int64[10] ranks_depth, int64[10] ranks_feat, int64[10] ranks_bev)
           => (int64[1] out_dim)
    {
      bev = mmdeploy.bev_pool_v2 <bev_h = 8, bev_w = 9> (depth, feat, ranks_depth, ranks_feat, ranks_bev)
      shp = Shape(bev)
      idx = Constant<value = int64[1] {3}>()
      out_dim = Gather<axis = 0>(shp, idx)
    }
    """
    model = _model(body)
    assert _folded_dim(model) == 9
