/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bev_custom_op_schemas.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxsim {

namespace {

using onnx::InferenceContext;
using onnx::OpSchema;

// Every schema in this file is registered under both of these domains,
// matching the domain tolerance every rewrite pass on this branch itself
// implements (`node->has_domain() && !node->domain().empty() &&
// node->domain() != "mmdeploy"` -> decline; empty/unset and "mmdeploy" are
// both accepted).
constexpr const char* kDomains[] = {"", "mmdeploy"};

// Generous element-type constraints, deliberately -- see
// bev_custom_op_schemas.h's header comment for why a schema narrower than the
// real op's contract is worse than no schema at all. `kFloatTypes` covers every
// float precision a TensorRT/mmdeploy export plausibly uses (FP16 deployments
// are common); `kIndexTypes` covers both 32- and 64-bit index/shape tensors.
const std::vector<std::string>& FloatTypes() {
  static const std::vector<std::string> types = {"tensor(float)",
                                                 "tensor(float16)"};
  return types;
}

const std::vector<std::string>& IndexTypes() {
  static const std::vector<std::string> types = {"tensor(int32)",
                                                 "tensor(int64)"};
  return types;
}

// Registers `schema` unless a schema for the same (name, domain) pair is
// already known -- e.g. one a caller's own `onnx.defs.register_schema` (or a
// real ONNX Runtime/TensorRT build) already provided. Never fails or
// throws: registering the same schema twice (once per process, or once per
// model simplified in the same process) is a harmless no-op.
void RegisterIfAbsent(OpSchema&& schema) {
  const std::string name = schema.Name();
  const std::string domain = schema.domain();
  if (onnx::OpSchemaRegistry::Schema(name, domain) != nullptr) {
    return;
  }
  onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                       /*fail_duplicate_schema=*/false,
                       /*fail_with_exception=*/false);
}

// --------------------------------------------------------------------------
// MMCVMultiScaleDeformableAttention
// --------------------------------------------------------------------------

// Output shape (bs, num_queries, num_heads*embed_dims): `bs` from `value`'s
// own leading dim (falling back to `sampling_locations`' when `value`'s is
// unknown -- both name the same batch axis), `num_queries` from
// `sampling_locations`' 2nd dim, and the trailing dim only when both
// `value`'s num_heads (dim 2) and embed_dims (dim 3) are statically known
// (matching rewrite_msdeformattn_to_gridsample.h's own static-M/D
// requirement) -- left unset otherwise, exactly as plain shape inference
// leaves any dimension it cannot determine.
void MSDeformAttnShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumInputs() < 4) {
    return;
  }
  const onnx::TensorShapeProto* value_shape =
      onnx::hasInputShape(ctx, 0) ? &ctx.getInputType(0)->tensor_type().shape()
                                  : nullptr;
  const onnx::TensorShapeProto* sampling_shape =
      onnx::hasInputShape(ctx, 3) ? &ctx.getInputType(3)->tensor_type().shape()
                                  : nullptr;

  auto* out_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  out_shape->clear_dim();
  // dim 0: bs.
  if (value_shape != nullptr && value_shape->dim_size() == 4) {
    *out_shape->add_dim() = value_shape->dim(0);
  } else if (sampling_shape != nullptr && sampling_shape->dim_size() == 6) {
    *out_shape->add_dim() = sampling_shape->dim(0);
  } else {
    out_shape->add_dim();
  }
  // dim 1: num_queries.
  if (sampling_shape != nullptr && sampling_shape->dim_size() == 6) {
    *out_shape->add_dim() = sampling_shape->dim(1);
  } else {
    out_shape->add_dim();
  }
  // dim 2: num_heads * embed_dims, only when both are static.
  if (value_shape != nullptr && value_shape->dim_size() == 4 &&
      value_shape->dim(2).has_dim_value() &&
      value_shape->dim(3).has_dim_value()) {
    out_shape->add_dim()->set_dim_value(value_shape->dim(2).dim_value() *
                                        value_shape->dim(3).dim_value());
  } else {
    out_shape->add_dim();
  }
}

OpSchema MakeMSDeformAttnSchema(const char* domain) {
  return OpSchema()
      .SetName("MMCVMultiScaleDeformableAttention")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "Multi-scale deformable attention (BEVFormer, Deformable DETR): "
          "for each query, bilinearly samples `value` at `sampling_locations` "
          "(per head/level/point) across the multi-level feature maps "
          "described by `spatial_shapes`/`level_start_index`, and combines "
          "the samples with `attention_weights`. No ONNX Runtime kernel "
          "exists for this op; onnxsim's opt-in "
          "rewrite_msdeformattn_to_gridsample pass decomposes it into "
          "standard ops built around GridSample.")
      .Attr("im2col_step",
            "CUDA-kernel batching parameter; does not affect output values.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Input(0, "value",
             "(bs, num_keys, num_heads, embed_dims) multi-level feature "
             "values, flattened over levels along the num_keys axis.",
             "T")
      .Input(1, "spatial_shapes",
             "(num_levels, 2) rows of (H_l, W_l) per feature level.", "T1")
      .Input(2, "level_start_index",
             "(num_levels,) flat start offset of each level within "
             "num_keys.",
             "T1")
      .Input(3, "sampling_locations",
             "(bs, num_queries, num_heads, num_levels, num_points, 2) "
             "normalized [0,1] sampling coordinates.",
             "T")
      .Input(4, "attention_weights",
             "(bs, num_queries, num_heads, num_levels, num_points) "
             "per-sample weights.",
             "T")
      .Output(0, "output",
              "(bs, num_queries, num_heads*embed_dims) attended output.", "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain value/sampling_locations/attention_weights/"
                      "output to float tensors.")
      .TypeConstraint("T1", IndexTypes(),
                      "Constrain spatial_shapes/level_start_index to integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(MSDeformAttnShapeInference)
      .AllowUncheckedAttributes();
}

// --------------------------------------------------------------------------
// MMCVDeformConv2d / MMCVModulatedDeformConv2d
// --------------------------------------------------------------------------

// Output shape (N, Cout, Hout, Wout): `N` from `input`'s own leading dim,
// `Cout` from `weight`'s leading dim, and `Hout`/`Wout` read directly off
// `offset`'s own trailing two dims -- `offset` is already sized to the
// op's actual output resolution by the exporter (rewrite_deform_conv_to_
// gather.h's own header comment notes this is why the pass itself never
// needs to recompute them from stride/padding/dilation arithmetic either).
void DeformConvShapeInference(InferenceContext& ctx, size_t weight_index) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto* out_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  out_shape->clear_dim();

  const bool has_input_shape =
      onnx::hasInputShape(ctx, 0) &&
      ctx.getInputType(0)->tensor_type().shape().dim_size() == 4;
  const bool has_weight_shape =
      ctx.getNumInputs() > weight_index &&
      onnx::hasInputShape(ctx, weight_index) &&
      ctx.getInputType(weight_index)->tensor_type().shape().dim_size() == 4;
  const bool has_offset_shape =
      onnx::hasInputShape(ctx, 1) &&
      ctx.getInputType(1)->tensor_type().shape().dim_size() == 4;

  // dim 0: N.
  if (has_input_shape) {
    *out_shape->add_dim() = ctx.getInputType(0)->tensor_type().shape().dim(0);
  } else {
    out_shape->add_dim();
  }
  // dim 1: Cout.
  if (has_weight_shape) {
    *out_shape->add_dim() =
        ctx.getInputType(weight_index)->tensor_type().shape().dim(0);
  } else {
    out_shape->add_dim();
  }
  // dims 2, 3: Hout, Wout, from offset's own trailing dims.
  if (has_offset_shape) {
    const auto& offset_shape = ctx.getInputType(1)->tensor_type().shape();
    *out_shape->add_dim() = offset_shape.dim(2);
    *out_shape->add_dim() = offset_shape.dim(3);
  } else {
    out_shape->add_dim();
    out_shape->add_dim();
  }
}

OpSchema MakeDeformConv2dSchema(const char* domain) {
  return OpSchema()
      .SetName("MMCVDeformConv2d")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "Deformable convolution v1 (DCNv1): a standard convolution whose "
          "per-kernel-tap sampling location is offset by a learned "
          "per-output-pixel, per-deform-group (dy, dx) pair, bilinearly "
          "sampled with zeros padding outside the input's bounds. No ONNX "
          "Runtime kernel exists for this op; onnxsim's opt-in "
          "rewrite_deform_conv_to_gather pass decomposes it into standard "
          "ops (scoped to groups == 1).")
      .Attr("stride", "(stride_h, stride_w).", onnx::AttributeProto::INTS,
            /*required=*/false)
      .Attr("padding", "(pad_h, pad_w).", onnx::AttributeProto::INTS,
            /*required=*/false)
      .Attr("dilation", "(dilation_h, dilation_w).", onnx::AttributeProto::INTS,
            /*required=*/false)
      .Attr("groups", "Convolution groups.", onnx::AttributeProto::INT,
            /*required=*/false)
      .Attr("deform_groups",
            "Deform groups (channels-for-offset-purposes partition count; "
            "also seen spelled `deformable_groups`).",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("deformable_groups", "Alias of `deform_groups`.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Input(0, "input", "(N, Cin, H, W) input feature map.", "T")
      .Input(1, "offset",
             "(N, deform_groups*2*kh*kw, Hout, Wout) per-tap (dy, dx) "
             "offsets.",
             "T")
      .Input(2, "weight", "(Cout, Cin/groups, kh, kw) convolution weight.", "T")
      .Input(3, "bias", "(Cout,) optional bias.", "T", OpSchema::Optional)
      .Output(0, "output", "(N, Cout, Hout, Wout) output feature map.", "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain input/offset/weight/bias/output to float "
                      "tensors.")
      .TypeAndShapeInferenceFunction(
          [](InferenceContext& ctx) { DeformConvShapeInference(ctx, 2); })
      .AllowUncheckedAttributes();
}

OpSchema MakeModulatedDeformConv2dSchema(const char* domain) {
  return OpSchema()
      .SetName("MMCVModulatedDeformConv2d")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "Modulated deformable convolution v2 (DCNv2): as MMCVDeformConv2d, "
          "plus a per-tap, per-deform-group scalar modulation mask "
          "multiplied into each sampled value. No ONNX Runtime kernel "
          "exists for this op; onnxsim's opt-in rewrite_deform_conv_to_gather "
          "pass decomposes it into standard ops (scoped to groups == 1).")
      .Attr("stride", "(stride_h, stride_w).", onnx::AttributeProto::INTS,
            /*required=*/false)
      .Attr("padding", "(pad_h, pad_w).", onnx::AttributeProto::INTS,
            /*required=*/false)
      .Attr("dilation", "(dilation_h, dilation_w).", onnx::AttributeProto::INTS,
            /*required=*/false)
      .Attr("groups", "Convolution groups.", onnx::AttributeProto::INT,
            /*required=*/false)
      .Attr("deform_groups",
            "Deform groups (channels-for-offset/mask-purposes partition "
            "count; also seen spelled `deformable_groups`).",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("deformable_groups", "Alias of `deform_groups`.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Input(0, "input", "(N, Cin, H, W) input feature map.", "T")
      .Input(1, "offset",
             "(N, deform_groups*2*kh*kw, Hout, Wout) per-tap (dy, dx) "
             "offsets.",
             "T")
      .Input(2, "mask",
             "(N, deform_groups*kh*kw, Hout, Wout) per-tap modulation "
             "scalars.",
             "T")
      .Input(3, "weight", "(Cout, Cin/groups, kh, kw) convolution weight.", "T")
      .Input(4, "bias", "(Cout,) optional bias.", "T", OpSchema::Optional)
      .Output(0, "output", "(N, Cout, Hout, Wout) output feature map.", "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain input/offset/mask/weight/bias/output to "
                      "float tensors.")
      .TypeAndShapeInferenceFunction(
          [](InferenceContext& ctx) { DeformConvShapeInference(ctx, 3); })
      .AllowUncheckedAttributes();
}

// --------------------------------------------------------------------------
// TRTBatchedNMS / TRTBatchedRotatedNMS
// --------------------------------------------------------------------------

// Shared shape inference for both NMS variants: `num_detections` (N, 1),
// `nmsed_boxes` (N, keepTopK, box_width), `nmsed_scores`/`nmsed_classes`
// (N, keepTopK). `N` comes from `boxes`' own leading dim; `keepTopK` from
// the node's own attribute when statically positive, left unset otherwise
// (exactly as any dimension shape inference cannot determine is left
// unset). `num_detections`' element type defaults to INT32, matching both
// passes' own documented default for an undeclared output type -- see
// bev_custom_op_schemas.h's header comment on why this is a soft,
// best-effort annotation rather than a hard requirement.
void BatchedNmsShapeInference(InferenceContext& ctx, int64_t box_width) {
  if (ctx.getNumOutputs() != 4) {
    return;
  }
  onnx::updateOutputElemType(ctx, 0, onnx::TensorProto::INT32);
  if (onnx::hasInputShape(ctx, 0)) {
    onnx::propagateElemTypeFromInputToOutput(ctx, 0, 1);
  }
  if (onnx::hasInputShape(ctx, 1)) {
    onnx::propagateElemTypeFromInputToOutput(ctx, 1, 2);
    onnx::propagateElemTypeFromInputToOutput(ctx, 1, 3);
  }

  const onnx::TensorShapeProto_Dimension* n_dim = nullptr;
  onnx::TensorShapeProto_Dimension n_dim_storage;
  if (onnx::hasInputShape(ctx, 0) &&
      ctx.getInputType(0)->tensor_type().shape().dim_size() == 4) {
    n_dim_storage = ctx.getInputType(0)->tensor_type().shape().dim(0);
    n_dim = &n_dim_storage;
  }

  int64_t keep_top_k = -1;
  const auto* keep_top_k_attr = ctx.getAttribute("keepTopK");
  if (keep_top_k_attr != nullptr && keep_top_k_attr->has_i() &&
      keep_top_k_attr->i() > 0) {
    keep_top_k = keep_top_k_attr->i();
  }

  // num_detections: (N, 1).
  {
    auto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    shape->clear_dim();
    if (n_dim != nullptr) {
      *shape->add_dim() = *n_dim;
    } else {
      shape->add_dim();
    }
    shape->add_dim()->set_dim_value(1);
  }
  // nmsed_boxes: (N, keepTopK, box_width).
  {
    auto* shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    shape->clear_dim();
    if (n_dim != nullptr) {
      *shape->add_dim() = *n_dim;
    } else {
      shape->add_dim();
    }
    if (keep_top_k > 0) {
      shape->add_dim()->set_dim_value(keep_top_k);
    } else {
      shape->add_dim();
    }
    shape->add_dim()->set_dim_value(box_width);
  }
  // nmsed_scores, nmsed_classes: (N, keepTopK).
  for (int out_idx : {2, 3}) {
    auto* shape =
        ctx.getOutputType(out_idx)->mutable_tensor_type()->mutable_shape();
    shape->clear_dim();
    if (n_dim != nullptr) {
      *shape->add_dim() = *n_dim;
    } else {
      shape->add_dim();
    }
    if (keep_top_k > 0) {
      shape->add_dim()->set_dim_value(keep_top_k);
    } else {
      shape->add_dim();
    }
  }
}

OpSchema MakeBatchedNmsSchemaCommon(const char* name, const char* domain,
                                    const char* doc, int64_t box_width) {
  OpSchema schema;
  schema.SetName(name)
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(doc)
      .Attr("background_label_id",
            "Class index to exclude from consideration, or -1 for none.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("num_classes", "Number of classes in `scores`.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("topK", "Max boxes retained per class before merging.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("keepTopK", "Max final detections per batch item.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("scoreThreshold", "Pre-NMS score threshold.",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("iouThreshold", "Greedy-NMS IoU threshold.",
            onnx::AttributeProto::FLOAT, /*required=*/false)
      .Attr("isNormalized", "Whether box coordinates are already in [0,1].",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("clipBoxes", "Whether to clip boxes to [0,1] before NMS.",
            onnx::AttributeProto::INT, /*required=*/false)
      .Input(0, "boxes",
             box_width == 4
                 ? "(N, num_boxes, num_classes_or_1, 4) (x1,y1,x2,y2)-style "
                   "boxes."
                 : "(N, num_boxes, num_classes_or_1, 5) (cx,cy,w,h,theta) "
                   "rotated boxes.",
             "T")
      .Input(1, "scores", "(N, num_boxes, num_classes) per-class scores.", "T")
      .Output(0, "num_detections",
              "(N, 1) count of valid (non-padding) detections per batch "
              "item.",
              "T_COUNT")
      .Output(1, "nmsed_boxes",
              box_width == 4 ? "(N, keepTopK, 4) kept boxes, zero-padded."
                             : "(N, keepTopK, 5) kept rotated boxes, "
                               "zero-padded.",
              "T")
      .Output(2, "nmsed_scores", "(N, keepTopK) kept scores, zero-padded.", "T")
      .Output(3, "nmsed_classes",
              "(N, keepTopK) kept class indices (as float), padded with "
              "-1.0.",
              "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain boxes/scores/outputs to float tensors.")
      .TypeConstraint("T_COUNT", {"tensor(int32)", "tensor(int64)"},
                      "Constrain num_detections to integer tensors.")
      .TypeAndShapeInferenceFunction([box_width](InferenceContext& ctx) {
        BatchedNmsShapeInference(ctx, box_width);
      })
      .AllowUncheckedAttributes();
  return schema;
}

OpSchema MakeTRTBatchedNmsSchema(const char* domain) {
  return MakeBatchedNmsSchemaCommon(
      "TRTBatchedNMS", domain,
      "TensorRT-plugin per-class greedy NMS with a per-batch top-K merge "
      "across classes, for axis-aligned boxes. No ONNX Runtime kernel "
      "exists for this op; onnxsim's opt-in rewrite_trt_batched_nms pass "
      "decomposes it into NonMaxSuppression plus standard ops (scoped to "
      "class-agnostic boxes and a statically-known batch size).",
      /*box_width=*/4);
}

OpSchema MakeTRTBatchedRotatedNmsSchema(const char* domain) {
  return MakeBatchedNmsSchemaCommon(
      "TRTBatchedRotatedNMS", domain,
      "TensorRT-plugin per-class greedy NMS with a per-batch top-K merge "
      "across classes, for rotated (cx,cy,w,h,theta) boxes. No ONNX Runtime "
      "kernel exists for this op; onnxsim's opt-in "
      "rewrite_trt_batched_rotated_nms pass decomposes it via Sutherland-"
      "Hodgman rotated-IoU and an unrolled greedy-suppression loop (scoped "
      "to class-agnostic boxes and statically-known batch size/num_classes).",
      /*box_width=*/5);
}

// --------------------------------------------------------------------------
// bev_pool_v2
// --------------------------------------------------------------------------

// Output shape (B, C, bev_z, bev_h, bev_w) (or (B, C, bev_h, bev_w) when
// `bev_z` is absent/1 -- both ranks are real conventions, see
// rewrite_bev_pool_to_scatter.h's own `DetermineBevGridShape`). `B` from
// `depth`'s own leading dim, `C` from `feat`'s own trailing dim; the BEV
// grid dims themselves only from `bev_h`/`bev_w`/`bev_z` attributes here
// (unlike the pass itself, which also accepts the grid shape via the
// node's *own* already-declared output shape or a `bev_feat_shape`/
// `output_shape` attribute -- this schema cannot read its own node's
// output shape from within a shape-inference function called to
// *establish* that very shape, and `bev_feat_shape`/`output_shape` are a
// generic-enough attribute name that guessing their semantics here, on top
// of `bev_pool_v2`'s already-uncertain naming contract, felt like
// compounding one guess with another). When neither `bev_h` nor `bev_w` is
// present, this function leaves the output shape's BEV-grid dims unset,
// same as plain shape inference would for any dimension it cannot
// determine -- the rewrite pass's own, more permissive shape-source
// fallback chain still applies regardless of what this schema does or
// does not infer.
void BevPoolShapeInference(InferenceContext& ctx) {
  if (onnx::hasInputShape(ctx, 1)) {
    onnx::propagateElemTypeFromInputToOutput(ctx, 1, 0);
  } else if (onnx::hasInputShape(ctx, 0)) {
    onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  }

  const bool has_depth_shape =
      onnx::hasInputShape(ctx, 0) &&
      ctx.getInputType(0)->tensor_type().shape().dim_size() == 5;
  const bool has_feat_shape =
      onnx::hasInputShape(ctx, 1) &&
      ctx.getInputType(1)->tensor_type().shape().dim_size() == 5;

  int64_t bev_h = -1, bev_w = -1, bev_z = 1;
  const auto* h_attr = ctx.getAttribute("bev_h");
  const auto* w_attr = ctx.getAttribute("bev_w");
  if (h_attr != nullptr && h_attr->has_i() && h_attr->i() > 0 &&
      w_attr != nullptr && w_attr->has_i() && w_attr->i() > 0) {
    bev_h = h_attr->i();
    bev_w = w_attr->i();
    const auto* z_attr = ctx.getAttribute("bev_z");
    if (z_attr != nullptr && z_attr->has_i() && z_attr->i() > 0) {
      bev_z = z_attr->i();
    }
  }

  auto* out_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  out_shape->clear_dim();
  // dim 0: B.
  if (has_depth_shape) {
    *out_shape->add_dim() = ctx.getInputType(0)->tensor_type().shape().dim(0);
  } else {
    out_shape->add_dim();
  }
  // dim 1: C.
  if (has_feat_shape) {
    *out_shape->add_dim() = ctx.getInputType(1)->tensor_type().shape().dim(4);
  } else {
    out_shape->add_dim();
  }
  if (bev_z > 1) {
    out_shape->add_dim()->set_dim_value(bev_z);
  }
  if (bev_h > 0 && bev_w > 0) {
    out_shape->add_dim()->set_dim_value(bev_h);
    out_shape->add_dim()->set_dim_value(bev_w);
  } else {
    out_shape->add_dim();
    out_shape->add_dim();
  }
}

OpSchema MakeBevPoolSchema(const char* domain) {
  return OpSchema()
      .SetName("bev_pool_v2")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "BEVDet/BEVFusion's LSS-style camera-to-BEV voxel pooling: "
          "gathers per-(camera, depth-bin, pixel) depth/feature values via "
          "the shared `ranks_depth`/`ranks_feat` index arrays, multiplies "
          "them, and scatter-adds into the BEV grid cell named by "
          "`ranks_bev`. No ONNX Runtime kernel exists for this op; "
          "onnxsim's opt-in rewrite_bev_pool_to_scatter pass decomposes it "
          "into Gather/Mul/ScatterND(reduction=\"add\"). This op's own "
          "op_type/domain/attribute naming is BEVDet's bespoke plugin "
          "convention, not part of mainline mmdeploy/mmcv -- see this "
          "file's header comment.")
      .Attr("bev_h", "BEV grid height.", onnx::AttributeProto::INT,
            /*required=*/false)
      .Attr("bev_w", "BEV grid width.", onnx::AttributeProto::INT,
            /*required=*/false)
      .Attr("bev_z", "BEV grid depth (1 for a single-level BEV plane).",
            onnx::AttributeProto::INT, /*required=*/false)
      .Attr("bev_feat_shape",
            "Alternative (bev_h, bev_w) or (bev_z, bev_h, "
            "bev_w) form of the above.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Attr("output_shape", "Alias of `bev_feat_shape`.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Input(0, "depth",
             "(B, N, D, H, W) per-pixel, per-depth-bin probability.", "T")
      .Input(1, "feat", "(B, N, H, W, C) per-pixel camera features.", "T")
      .Input(2, "ranks_depth",
             "(num_valid,) flat index into depth[b].reshape(N*D*H*W), "
             "shared across the batch.",
             "T1")
      .Input(3, "ranks_feat",
             "(num_valid,) flat index into feat[b].reshape(N*H*W), shared "
             "across the batch.",
             "T1")
      .Input(4, "ranks_bev",
             "(num_valid,) flat target BEV-grid index, shared across the "
             "batch.",
             "T1")
      .Input(5, "interval_starts",
             "Optional CSR-style grouping metadata; not required by "
             "onnxsim's decomposition.",
             "T1", OpSchema::Optional)
      .Input(6, "interval_lengths", "Optional, paired with interval_starts.",
             "T1", OpSchema::Optional)
      .Output(0, "bev_feat",
              "(B, C, bev_z, bev_h, bev_w), or (B, C, bev_h, bev_w) when "
              "bev_z is absent/1.",
              "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain depth/feat/bev_feat to float tensors.")
      .TypeConstraint("T1", IndexTypes(),
                      "Constrain the rank/interval index tensors to integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(BevPoolShapeInference)
      .AllowUncheckedAttributes();
}

void RegisterAll() {
  // The default ("") domain is ai.onnx itself and is already known to the
  // registry. "mmdeploy" is not a standard ONNX domain, so -- exactly like
  // contrib_schemas.cpp's own `kMSDomain` registration -- it must be
  // registered with the schema registry's domain-to-version-range map
  // before any schema can be registered into it; skipping this step doesn't
  // fail loudly, it just silently makes every RegisterSchema call below a
  // no-op for that domain (found empirically: `RegisterSchema` printed
  // "domain is not known by the checker" to stderr and the schema was never
  // actually registered).
  auto& domain_range = onnx::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domain_range.Map().count("mmdeploy") == 0) {
    domain_range.AddDomainToVersion("mmdeploy", /*min_version=*/1,
                                    /*max_version=*/1);
  }

  for (const char* domain : kDomains) {
    RegisterIfAbsent(MakeMSDeformAttnSchema(domain));
    RegisterIfAbsent(MakeDeformConv2dSchema(domain));
    RegisterIfAbsent(MakeModulatedDeformConv2dSchema(domain));
    RegisterIfAbsent(MakeTRTBatchedNmsSchema(domain));
    RegisterIfAbsent(MakeTRTBatchedRotatedNmsSchema(domain));
    RegisterIfAbsent(MakeBevPoolSchema(domain));
  }
}

}  // namespace

void RegisterBevCustomOpSchemas() {
  static std::once_flag once;
  std::call_once(once, RegisterAll);
}

}  // namespace onnxsim
