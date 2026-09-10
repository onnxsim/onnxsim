/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace onnxsim {

// Import operator schemas for QONNX's generic fake-quantization custom ops:
// `Quant`, `BipolarQuant`, `Trunc`, and `FloatQuant`. These are the ops
// Brevitas (the PyTorch quantization-aware-training library FINN/Xilinx
// deployments are built on) emits by default via
// `brevitas.export.export_qonnx` -- its primary/native export path, used for
// arbitrary-bit-width weight and activation quantization that plain 8-bit
// `QuantizeLinear`/`DequantizeLinear` cannot express. They live in the
// `qonnx.custom_op.general` domain (the current name) and its predecessor
// `finn.custom_op.general` (still emitted by older Brevitas/FINN versions);
// both are registered identically here.
//
// Like ONNX Runtime's `com.microsoft` contrib ops (contrib_schemas.h) and the
// mmdeploy/mmcv/BEVDet ops (bev_custom_op_schemas.h), these schemas are
// unknown to plain ONNX shape inference, so without them shape deduction
// stops dead the moment it reaches a `Quant` node -- every weight or
// activation Brevitas quantizes. Registering a `TypeAndShapeInferenceFunction`
// for each (in practice: the output is elementwise-shaped like the first
// input, exactly like the op's own "fake-quantize then dequantize back to
// float in place" semantics) lets onnx::shape_inference::InferShapes flow
// through them, which unlocks simplification of everything downstream.
//
// This intentionally stops at shape inference. Unlike `QuantizeLinear`/
// `DequantizeLinear`, which onnxsim's constant folder explicitly declines to
// fold even on constant inputs (`IsQDQ` in constant_folding.cpp) so the
// quantization boundary survives simplification, these ops are simply never
// offered to the constant folder at all: `IsOfficialOp` (constant_folding.cpp)
// only recognizes the default ONNX domain, so a `Quant`/`BipolarQuant`/
// `Trunc`/`FloatQuant` node is never a constant-folding candidate regardless
// of whether its inputs are constant -- the same "preserve the quantization
// annotation" outcome as `IsQDQ`, for free, without this file needing to
// special-case it.
//
// The registration is performed at most once per process and never overrides
// a schema that is already registered (for example one a caller's own
// `onnx.defs.register_schema` already bridged in via `import_custom_schemas`).
// It is safe to call multiple times and from any of the simplification entry
// points.
void RegisterQonnxCustomOpSchemas();

}  // namespace onnxsim
