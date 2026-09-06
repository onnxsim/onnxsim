/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "custom_optimizer_passes.h"

#include <memory>
#include <mutex>
#include <string>

#include "onnxoptimizer/optimize.h"
#include "passes/any_precision_llm.h"
#include "passes/cross_layer_equalization.h"
#include "passes/defuse_matmul_integer_to_float.h"
#include "passes/double_quantization.h"
#include "passes/dynamic_quantize_attention.h"
#include "passes/dynamic_quantize_matmul.h"
#include "passes/dynamic_quantize_matmul_integer_to_float.h"
#include "passes/dynamic_quantize_ternary_matmul.h"
#include "passes/eliminate_consecutive_idempotent_ops.h"
#include "passes/eliminate_loop_with_const_trip_count.h"
#include "passes/eliminate_nop_dropout.h"
#include "passes/eliminate_optional_get_element.h"
#include "passes/eliminate_optional_has_element.h"
#include "passes/eliminate_reshape_around_elementwise.h"
#include "passes/eliminate_sequence_at_construct.h"
#include "passes/eliminate_sequence_length_construct.h"
#include "passes/fuse_add_bias_into_conv.h"
#include "passes/fuse_attention.h"
#include "passes/fuse_bn_into_conv.h"
#include "passes/fuse_consecutive_mul.h"
#include "passes/fuse_consecutive_reduce.h"
#include "passes/fuse_consecutive_reshapes.h"
#include "passes/fuse_consecutive_unsqueezes.h"
#include "passes/fuse_gelu.h"
#include "passes/fuse_gqa.h"
#include "passes/fuse_layer_norm.h"
#include "passes/fuse_matmul_add_bias_into_gemm.h"
#include "passes/fuse_matmul_add_bias_into_gemm_batched.h"
#include "passes/fuse_matmul_into_conv.h"
#include "passes/fuse_mul_into_conv.h"
#include "passes/fuse_pad_into_pool.h"
#include "passes/fuse_preceding_mul_into_conv.h"
#include "passes/fuse_qkv.h"
#include "passes/fuse_reshape_family.h"
#include "passes/fuse_rms_norm.h"
#include "passes/fuse_rope.h"
#include "passes/gguf_legacy_quant.h"
#include "passes/gguf_ternary_quant.h"
#include "passes/iq4_nl.h"
#include "passes/magnitude_pruning.h"
#include "passes/qoperator_quantize_activation.h"
#include "passes/qoperator_quantize_concat.h"
#include "passes/qoperator_quantize_conv.h"
#include "passes/qoperator_quantize_elementwise.h"
#include "passes/qoperator_quantize_gemm.h"
#include "passes/qoperator_quantize_matmul.h"
#include "passes/qoperator_quantize_pool.h"
#include "passes/qoperator_quantize_softmax.h"
#include "passes/qoperator_quantize_where.h"
#include "passes/quantize_bf16.h"
#include "passes/quantize_fp16.h"
#include "passes/quantize_fp8.h"
#include "passes/quarot.h"
#include "passes/rewrite_arg_reduce_select_last_index.h"
#include "passes/rewrite_bool_where.h"
#include "passes/rewrite_gatherelements_to_gather.h"
#include "passes/rewrite_gathernd_to_gather.h"
#include "passes/rewrite_gridsample_to_gather.h"
#include "passes/static_quantize_conv.h"
#include "passes/static_quantize_int16_conv.h"
#include "passes/static_quantize_int16_matmul.h"
#include "passes/static_quantize_matmul.h"
#include "passes/weight_only_quantize_conv.h"
#include "passes/weight_only_quantize_int16_conv.h"
#include "passes/weight_only_quantize_int16_matmul.h"
#include "passes/weight_only_quantize_int4_conv.h"
#include "passes/weight_only_quantize_int4_matmul.h"
#include "passes/weight_only_quantize_int8_block_conv.h"
#include "passes/weight_only_quantize_int8_block_matmul.h"
#include "passes/weight_only_quantize_matmul.h"
#include "passes/weight_only_quantize_matmul_nbits.h"
#include "passes/weight_only_quantize_mxfp4_matmul.h"

namespace onnxsim {

namespace {

// Register onnxsim's pass T into onnxoptimizer's global registry, replacing any
// existing pass of the same name. onnxoptimizer's own registerPass<T> always
// appends to pass_names, so reusing it to overwrite a name would list -- and
// therefore run -- that pass twice. We touch the registry's public members
// directly instead: overwrite the map entry, and add the name only when it is
// new. This is what lets onnxsim ship its own version of an onnxoptimizer pass
// (e.g. a bug-fixed one) and have it win over the built-in of the same name.
template <typename T>
void RegisterOrReplace(ONNX_NAMESPACE::optimization::GlobalPassRegistry& reg) {
  auto pass = std::make_shared<T>();
  const std::string name = pass->getPassName();
  if (reg.passes.find(name) == reg.passes.end()) {
    reg.pass_names.emplace_back(name);
  }
  reg.passes[name] = std::move(pass);
}

}  // namespace

void RegisterCustomOptimizerPasses() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    namespace p = ONNX_NAMESPACE::optimization::onnxsim_passes;
    // Inject onnxsim's passes into onnxoptimizer's existing global registry,
    // overwriting any built-in of the same name. call_once keeps this a
    // one-time write, matching the registry's static, read-only-after-init use.
    auto& registry = ONNX_NAMESPACE::optimization::Optimizer::passes;

    // onnxsim-only rewrites (no built-in of the same name today).
    RegisterOrReplace<p::AnyPrecisionLlm>(registry);
    RegisterOrReplace<p::CrossLayerEqualization>(registry);
    RegisterOrReplace<p::DefuseMatMulIntegerToFloat>(registry);
    RegisterOrReplace<p::DoubleQuantization>(registry);
    RegisterOrReplace<p::DynamicQuantizeAttention>(registry);
    RegisterOrReplace<p::DynamicQuantizeMatMul>(registry);
    RegisterOrReplace<p::DynamicQuantizeMatMulIntegerToFloat>(registry);
    RegisterOrReplace<p::DynamicQuantizeTernaryMatMul>(registry);
    RegisterOrReplace<p::EliminateLoopWithConstTripCount>(registry);
    RegisterOrReplace<p::EliminateOptionalGetElement>(registry);
    RegisterOrReplace<p::EliminateOptionalHasElement>(registry);
    RegisterOrReplace<p::EliminateReshapeAroundElementwise>(registry);
    RegisterOrReplace<p::EliminateSequenceAtConstruct>(registry);
    RegisterOrReplace<p::EliminateSequenceLengthConstruct>(registry);
    RegisterOrReplace<p::FuseAttention>(registry);
    RegisterOrReplace<p::FuseConsecutiveMul>(registry);
    RegisterOrReplace<p::FuseConsecutiveReduce>(registry);
    RegisterOrReplace<p::FuseConsecutiveReshapes>(registry);
    RegisterOrReplace<p::FuseGelu>(registry);
    RegisterOrReplace<p::FuseGQA>(registry);
    RegisterOrReplace<p::FuseLayerNorm>(registry);
    RegisterOrReplace<p::FuseMatMulAddBiasIntoGemmBatched>(registry);
    RegisterOrReplace<p::FuseMatMulIntoConv>(registry);
    RegisterOrReplace<p::FuseMulIntoConv>(registry);
    RegisterOrReplace<p::FusePrecedingMulIntoConv>(registry);
    RegisterOrReplace<p::FuseReshapeFamily>(registry);
    RegisterOrReplace<p::FuseRMSNorm>(registry);
    RegisterOrReplace<p::FuseRope>(registry);
    RegisterOrReplace<p::GgufQ4_0>(registry);
    RegisterOrReplace<p::GgufQ4_1>(registry);
    RegisterOrReplace<p::GgufTernaryQuant>(registry);
    RegisterOrReplace<p::IQ4NL>(registry);
    RegisterOrReplace<p::MagnitudePruningAttention>(registry);
    RegisterOrReplace<p::MagnitudePruningConv>(registry);
    RegisterOrReplace<p::MagnitudePruningGlobal>(registry);
    RegisterOrReplace<p::MagnitudePruningMatMul>(registry);
    RegisterOrReplace<p::QOperatorQuantizeActivation>(registry);
    RegisterOrReplace<p::QOperatorQuantizeConcat>(registry);
    RegisterOrReplace<p::QOperatorQuantizeConv>(registry);
    RegisterOrReplace<p::QOperatorQuantizeElementwise>(registry);
    RegisterOrReplace<p::QOperatorQuantizeGemm>(registry);
    RegisterOrReplace<p::QOperatorQuantizeMatMul>(registry);
    RegisterOrReplace<p::QOperatorQuantizePool>(registry);
    RegisterOrReplace<p::QOperatorQuantizeSoftmax>(registry);
    RegisterOrReplace<p::QOperatorQuantizeWhere>(registry);
    RegisterOrReplace<p::QuantizeBf16Pass>(registry);
    RegisterOrReplace<p::QuantizeFp16Pass>(registry);
    RegisterOrReplace<p::QuantizeFp8Pass>(registry);
    RegisterOrReplace<p::Quarot>(registry);
    RegisterOrReplace<p::RewriteArgReduceSelectLastIndex>(registry);
    RegisterOrReplace<p::RewriteBoolWhere>(registry);
    RegisterOrReplace<p::RewriteGatherElementsToGather>(registry);
    RegisterOrReplace<p::RewriteGatherNDToGather>(registry);
    RegisterOrReplace<p::RewriteGridSampleToGather>(registry);
    RegisterOrReplace<p::StaticQuantizeConv>(registry);
    RegisterOrReplace<p::StaticQuantizeInt16Conv>(registry);
    RegisterOrReplace<p::StaticQuantizeInt16MatMul>(registry);
    RegisterOrReplace<p::StaticQuantizeMatMul>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeConv>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeInt16Conv>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeInt16MatMul>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeInt4Conv>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeInt4MatMul>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeInt8BlockConv>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeInt8BlockMatMul>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeMatMul>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeMatMulNBits>(registry);
    RegisterOrReplace<p::WeightOnlyQuantizeMXFP4MatMul>(registry);

    // onnxsim's patched versions of built-in onnxoptimizer passes. These
    // overwrite the registry entries the submodule registers under the same
    // names, so onnxsim's fixes/extensions (ConvTranspose fusions, no-op
    // opset-12 Dropout, zero-padding MaxPool, ...) apply while the fork itself
    // tracks upstream onnxoptimizer.
    RegisterOrReplace<p::EliminateConsecutiveIdempotentOps>(registry);
    RegisterOrReplace<p::EliminateNopDropout>(registry);
    RegisterOrReplace<p::FuseAddBiasIntoConv>(registry);
    RegisterOrReplace<p::FuseBNIntoConv>(registry);
    RegisterOrReplace<p::FuseConsecutiveUnsqueezes>(registry);
    RegisterOrReplace<p::FuseMatMulAddBiasIntoGemm>(registry);
    RegisterOrReplace<p::FusePadIntoPool>(registry);
    RegisterOrReplace<p::FuseQKV>(registry);
  });
}

}  // namespace onnxsim
