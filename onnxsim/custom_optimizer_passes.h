/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace onnxsim {

// Register onnxsim's own optimizer passes into onnxoptimizer's global pass
// registry so they can be selected by name through Optimize/OptimizeFixed and
// appear in GetAvailablePasses / GetFuseAndEliminationPass, exactly like the
// passes that ship with onnxoptimizer.
//
// These passes are onnxsim-specific graph rewrites that used to live in
// onnxsim's onnxoptimizer fork (plus fuse_preceding_mul_into_conv, added
// directly in onnxsim). They are defined under onnxsim/passes/ and injected
// directly into onnxoptimizer's existing global pass registry
// (onnx::optimization::Optimizer::passes), so onnxoptimizer needs no change:
//
//   - fuse_mul_into_conv
//   - fuse_preceding_mul_into_conv
//   - fuse_consecutive_mul
//   - fuse_matmul_add_bias_into_gemm_batched
//   - eliminate_reshape_around_elementwise
//   - fuse_rms_norm
//   - fuse_gelu
//   - fuse_layer_norm
//   - dynamic_quantize_matmul
//   - dynamic_quantize_ternary_matmul
//   - static_quantize_matmul
//   - static_quantize_conv
//   - static_quantize_int16_matmul
//   - static_quantize_int16_conv
//   - weight_only_quantize_matmul
//   - weight_only_quantize_conv
//   - weight_only_quantize_int4_matmul
//   - weight_only_quantize_int4_conv
//   - weight_only_quantize_int16_matmul
//   - weight_only_quantize_int16_conv
//   - weight_only_quantize_int8_block_matmul
//   - weight_only_quantize_int8_block_conv
//   - qoperator_quantize_matmul
//   - qoperator_quantize_conv
//   - qoperator_quantize_elementwise
//   - qoperator_quantize_activation
//   - qoperator_quantize_concat
//   - qoperator_quantize_softmax
//   - quantize_fp16
//   - quantize_bf16
//   - quantize_fp8
//
// The registration runs at most once per process and is safe to call from any
// simplification entry point. It MUST run before the pass list is built (a
// couple of these passes are auto-included via GetFuseAndEliminationPass) and
// before OptimizeFixed is invoked, so the simplification entry points call it
// up front.
void RegisterCustomOptimizerPasses();

}  // namespace onnxsim
