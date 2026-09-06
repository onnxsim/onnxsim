/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace onnxsim {

// Import operator schemas for the mmdeploy/mmcv/BEVDet custom operators this
// branch's rewrite_msdeformattn_to_gridsample, rewrite_deform_conv_to_gather,
// rewrite_trt_batched_nms, rewrite_trt_batched_rotated_nms, and
// rewrite_bev_pool_to_scatter passes decompose:
//   MMCVMultiScaleDeformableAttention, MMCVDeformConv2d,
//   MMCVModulatedDeformConv2d, TRTBatchedNMS, TRTBatchedRotatedNMS,
//   bev_pool_v2.
//
// This is *not* required for onnxsim to simplify a model containing one of
// these ops -- two mechanisms already make that work with no schema at all:
// `RegisterCustomDefaultDomainOpSchemas` (model_prep.h) auto-registers a
// permissive placeholder for any of them exported into the default ONNX
// domain, and onnx::checker::check_model already tolerates unknown ops in a
// non-default domain (e.g. "mmdeploy") unconditionally. Both leave the op's
// own output shape/type unresolved, though ("shape inference simply flows
// past the op"), which only matters when something downstream of the op
// needs that shape *before* one of the rewrite passes above ever gets a
// chance to replace the node with real, shape-inferable ops -- an exporter
// that didn't already annotate the op's output in `value_info` is the one
// case this closes.
//
// Registering these schemas here, with a real `TypeAndShapeInferenceFunction`
// per op, lets ONNX's own shape inference compute that output shape from the
// op's inputs/attributes instead, in both the default and "mmdeploy" domains
// (matching every rewrite pass's own domain tolerance). Every schema here is
// deliberately permissive on input/output element types (float and float16;
// int32 and int64 for shape/index tensors) and calls
// `.AllowUncheckedAttributes()`: a schema that's *narrower* than the real
// op's actual contract would make previously-tolerated models fail
// `onnx::checker::check_model`, which is strictly worse than not registering
// a schema at all -- see this file's .cpp for the exact contract each
// schema encodes, and where it necessarily involved a judgment call (most
// notably `bev_pool_v2`, whose op_type/domain/attribute naming is BEVDet's
// own bespoke plugin convention, not confirmed against a real export, unlike
// the other five ops' mmcv/mmdeploy-sourced contracts).
//
// The registration is performed at most once per process and never
// overrides a schema already registered for the same (name, domain) pair
// (for example one a caller's own `onnx.defs.register_schema` already
// bridged in via `import_custom_schemas`). It is safe to call multiple
// times and from any of the simplification entry points.
void RegisterBevCustomOpSchemas();

}  // namespace onnxsim
