#include "quantize_entry.h"

#include <onnx/onnx_pb.h>

#include <memory>
#include <stdexcept>
#include <unordered_set>

#include "custom_optimizer_passes.h"
#include "model_prep.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnxoptimizer/optimize.h"
#include "passes/any_precision_llm.h"
#include "passes/dynamic_quantize_matmul_integer_to_float.h"
#include "passes/fp6_llm.h"
#include "passes/gguf_legacy_quant.h"
#include "passes/gguf_q6_k.h"
#include "passes/gguf_ternary_quant.h"
#include "passes/iq4_nl.h"
#include "passes/qoperator_quantize_gemm.h"
#include "passes/qoperator_quantize_pool.h"
#include "passes/qoperator_quantize_softmax.h"
#include "passes/qoperator_quantize_where.h"
#include "passes/quantize_bf16.h"
#include "passes/quantize_conv_common.h"
#include "passes/quantize_fp16.h"
#include "passes/quantize_fp8.h"
#include "passes/quantize_matmul_common.h"
#include "passes/quarot.h"
#include "passes/static_quantize_matmul.h"
#include "passes/zeroquant.h"

onnx::ModelProto QuantizeDynamic(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_matmul (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"dynamic_quantize_matmul",
                                      "eliminate_unused_initializer"});
}

onnx::ModelProto QuantizeDynamicMatMulIntegerToFloat(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_matmul_integer_to_float (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model,
      std::vector<std::string>{"dynamic_quantize_matmul_integer_to_float",
                                      "eliminate_unused_initializer"});
}

std::vector<std::string> ListQuantizableActivations(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::MatMulLikeInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchMatMulLike(node, info)) {
      continue;
    }
    if (info.x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    const onnx::Tensor* w_t = onnx::optimization::FetchConstantTensor(info.w);
    if (w_t == nullptr ||
        w_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      continue;
    }
    if (seen.insert(info.x->uniqueName()).second) {
      names.push_back(info.x->uniqueName());
    }
  }
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::ConvInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchConv(node, info)) {
      continue;
    }
    if (info.x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    const onnx::Tensor* w_t = onnx::optimization::FetchConstantTensor(info.w);
    if (w_t == nullptr ||
        w_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
        w_t->sizes().size() < 3) {
      continue;
    }
    if (seen.insert(info.x->uniqueName()).second) {
      names.push_back(info.x->uniqueName());
    }
  }
  return names;
}

onnx::ModelProto QuantizeStatic(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers static_quantize_matmul (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // static_quantize_matmul reads this global, the same way onnxsim's other
  // passes read `config` -- see StaticQuantizationCalibrationRanges's doc
  // comment.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"static_quantize_matmul",
                                      "static_quantize_conv",
                                      "eliminate_unused_initializer"});
}

onnx::ModelProto QuantizeStaticInt16(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers static_quantize_int16_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // Reads/writes the same calibration-ranges global QuantizeStatic uses --
  // safe as long as only one of QuantizeStatic/QuantizeStaticInt16/
  // QuantizeQOperator runs per call, which OptimizeFixed's single
  // pass-name list here ensures (see QuantizeQOperator's own comment on
  // this).
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"static_quantize_int16_matmul",
                                      "static_quantize_int16_conv",
                                      "eliminate_unused_initializer"});
}
