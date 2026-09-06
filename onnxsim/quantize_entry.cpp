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
#include "passes/gguf_legacy_quant.h"
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

onnx::ModelProto QuantizeDynamic(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_matmul (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"dynamic_quantize_matmul"});
}

onnx::ModelProto QuantizeDynamicMatMulIntegerToFloat(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_matmul_integer_to_float (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model,
      std::vector<std::string>{"dynamic_quantize_matmul_integer_to_float"});
}

onnx::ModelProto QuantizeAttentionDynamic(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_attention (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"dynamic_quantize_attention"});
}

onnx::ModelProto QuantizeTernary(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers dynamic_quantize_ternary_matmul (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"dynamic_quantize_ternary_matmul"});
}

onnx::ModelProto QuantizeWeightOnly(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_matmul",
                                      "weight_only_quantize_conv"});
}

onnx::ModelProto QuantizeWeightOnlyInt4(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_int4_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_int4_matmul",
                                      "weight_only_quantize_int4_conv"});
}

onnx::ModelProto QuantizeWeightOnlyMatMulNBits(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_matmul_nbits (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_matmul_nbits"});
}

onnx::ModelProto QuantizeWeightOnlyInt16(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_int16_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_int16_matmul",
                                      "weight_only_quantize_int16_conv"});
}

onnx::ModelProto QuantizeWeightOnlyInt8Block(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_int8_block_matmul/_conv (idempotent)
  // into onnxoptimizer's registry so OptimizeFixed can find them by name
  // below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_int8_block_matmul",
                                      "weight_only_quantize_int8_block_conv"});
}

onnx::ModelProto QuantizeWeightOnlyMXFP4(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers weight_only_quantize_mxfp4_matmul (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"weight_only_quantize_mxfp4_matmul"});
}

onnx::ModelProto ApplyDoubleQuantization(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers double_quantization (idempotent) into onnxoptimizer's registry
  // so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"double_quantization"});
}

onnx::ModelProto ApplyAnyPrecisionLlm(const onnx::ModelProto& model,
                                      int64_t bits, int64_t max_bits,
                                      int64_t block_size) {
  if (max_bits < 1) {
    throw std::invalid_argument(
        "ApplyAnyPrecisionLlm: max_bits must be >= 1, got " +
        std::to_string(max_bits));
  }
  if (bits < 1 || bits > max_bits) {
    throw std::invalid_argument(
        "ApplyAnyPrecisionLlm: bits (" + std::to_string(bits) +
        ") must be in [1, max_bits=" + std::to_string(max_bits) + "]");
  }
  PrepareSchemasForDebug(model);
  // Registers any_precision_llm (idempotent) into onnxoptimizer's registry
  // so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // any_precision_llm reads these the same way quarot reads QuarotSeed() --
  // OptimizeFixed's pass-name list has no way to carry a parameter directly.
  onnx::optimization::onnxsim_passes::AnyPrecisionLlmBits() = bits;
  onnx::optimization::onnxsim_passes::AnyPrecisionLlmMaxBits() = max_bits;
  onnx::optimization::onnxsim_passes::AnyPrecisionLlmBlockSize() = block_size;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"any_precision_llm"});
}

onnx::ModelProto ApplyQuarot(const onnx::ModelProto& model, uint64_t seed,
                             int64_t block_size, float epsilon) {
  PrepareSchemasForDebug(model);
  // Registers quarot (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quarot reads these the same way quantize_fp16 reads
  // QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no way
  // to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuarotSeed() = seed;
  onnx::optimization::onnxsim_passes::QuarotBlockSize() = block_size;
  onnx::optimization::onnxsim_passes::QuarotEpsilon() = epsilon;
  return onnx::optimization::OptimizeFixed(model,
                                           std::vector<std::string>{"quarot"});
}

onnx::ModelProto ApplyIQ4NL(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers iq4_nl (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(model,
                                           std::vector<std::string>{"iq4_nl"});
}

onnx::ModelProto ApplyGgufQ4_0(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"gguf_q4_0"});
}

onnx::ModelProto ApplyGgufQ4_1(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"gguf_q4_1"});
}

onnx::ModelProto ApplyGgufTernaryQuant(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"gguf_ternary_quant"});
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
                                      "static_quantize_conv"});
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
                                      "static_quantize_int16_conv"});
}

std::vector<std::string> ListQOperatorQuantizableOutputs(
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
    if (seen.insert(node->output()->uniqueName()).second) {
      names.push_back(node->output()->uniqueName());
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
    if (info.bias != nullptr) {
      // qoperator_quantize_conv only rewrites a Conv whose bias (if any) is a
      // constant float32 [Cout] tensor it can pre-quantize to INT32 -- see
      // that pass's doc comment. Skip listing this Conv's output otherwise,
      // since it will never actually be rewritten.
      const onnx::Tensor* b_t =
          onnx::optimization::FetchConstantTensor(info.bias);
      if (b_t == nullptr ||
          b_t->elem_type() != onnx::TensorProto_DataType_FLOAT ||
          b_t->sizes().size() != 1) {
        continue;
      }
    }
    if (seen.insert(node->output()->uniqueName()).second) {
      names.push_back(node->output()->uniqueName());
    }
  }
  return names;
}

onnx::ModelProto QuantizeQOperator(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_matmul/_conv read the same calibration-ranges global
  // static_quantize_matmul/_conv do (see StaticQuantizationCalibrationRanges's
  // doc comment) -- both are keyed by tensor name, and QOperator format's
  // ranges are a superset (activation names plus output names) of QDQ
  // format's, so sharing the map is safe as long as only one of
  // QuantizeStatic/QuantizeQOperator runs per call, which OptimizeFixed's
  // single pass-name list here ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_matmul",
                                      "qoperator_quantize_conv"});
}

std::vector<std::string> ListQOperatorElementwiseQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    if (node->kind() != onnx::kAdd && node->kind() != onnx::kMul) {
      continue;
    }
    if (node->inputs().size() != 2) {
      continue;
    }
    onnx::Value* a = node->inputs()[0];
    onnx::Value* b = node->inputs()[1];
    if (a->elemType() != onnx::TensorProto_DataType_FLOAT ||
        b->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    if (onnx::optimization::FetchConstantTensor(a) != nullptr ||
        onnx::optimization::FetchConstantTensor(b) != nullptr) {
      continue;
    }
    add_name(a->uniqueName());
    add_name(b->uniqueName());
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorElementwise(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_elementwise (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_elementwise reads the same calibration-ranges global
  // every other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, both operands' and the output's, so sharing the map is safe as
  // long as only one Quantize* entry point runs per call, which
  // OptimizeFixed's single pass-name list here ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_elementwise"});
}

std::vector<std::string> ListQOperatorActivationQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    if (node->kind() != onnx::kSigmoid &&
        node->kind() != onnx::Symbol("LeakyRelu")) {
      continue;
    }
    if (node->inputs().size() != 1) {
      continue;
    }
    onnx::Value* x = node->inputs()[0];
    if (x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    add_name(x->uniqueName());
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorActivation(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_activation (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_activation reads the same calibration-ranges global
  // every other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, so sharing the map is safe as long as only one Quantize* entry
  // point runs per call, which OptimizeFixed's single pass-name list here
  // ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_activation"});
}

std::vector<std::string> ListQOperatorConcatQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    if (node->kind() != onnx::kConcat || node->inputs().empty()) {
      continue;
    }
    bool all_quantizable = true;
    for (onnx::Value* in : node->inputs()) {
      if (in->elemType() != onnx::TensorProto_DataType_FLOAT ||
          onnx::optimization::FetchConstantTensor(in) != nullptr) {
        all_quantizable = false;
        break;
      }
    }
    if (!all_quantizable) {
      continue;
    }
    for (onnx::Value* in : node->inputs()) {
      add_name(in->uniqueName());
    }
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorConcat(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_concat (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_concat reads the same calibration-ranges global every
  // other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, so sharing the map is safe as long as only one Quantize* entry
  // point runs per call, which OptimizeFixed's single pass-name list here
  // ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_concat"});
}

namespace {

// Resolves ``model``'s own default-domain ("" / "ai.onnx") opset version
// from its opset imports, mirroring
// onnxsim_passes::DefaultDomainOpsetVersion (which operates on the IR
// Graph, not the ModelProto, so this is a separate, proto-level lookup).
// Returns 0 if it cannot be determined.
int64_t DefaultDomainOpsetVersionOf(const onnx::ModelProto& model) {
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return opset.version();
    }
  }
  return 0;
}

}  // namespace

std::vector<std::string> ListQOperatorSoftmaxQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  if (DefaultDomainOpsetVersionOf(model) <= 0) {
    return {};
  }
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    if (node->kind() != onnx::Symbol("Softmax")) {
      continue;
    }
    if (node->inputs().size() != 1) {
      continue;
    }
    onnx::Value* x = node->inputs()[0];
    if (x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    add_name(x->uniqueName());
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorSoftmax(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_softmax (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_softmax reads the same calibration-ranges global
  // every other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, so sharing the map is safe as long as only one Quantize* entry
  // point runs per call, which OptimizeFixed's single pass-name list here
  // ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_softmax"});
}

std::vector<std::string> ListQOperatorPoolQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    if (node->kind() != onnx::Symbol("AveragePool") &&
        node->kind() != onnx::Symbol("GlobalAveragePool")) {
      continue;
    }
    if (node->inputs().size() != 1) {
      continue;
    }
    if (node->hasAttribute(onnx::Symbol("dilations"))) {
      continue;
    }
    onnx::Value* x = node->inputs()[0];
    if (x->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    add_name(x->uniqueName());
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorPool(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_pool (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_pool reads the same calibration-ranges global every
  // other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, so sharing the map is safe as long as only one Quantize* entry
  // point runs per call, which OptimizeFixed's single pass-name list here
  // ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_pool"});
}

std::vector<std::string> ListQOperatorWhereQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    if (node->kind() != onnx::Symbol("Where")) {
      continue;
    }
    if (node->inputs().size() != 3) {
      continue;
    }
    onnx::Value* x = node->inputs()[1];
    onnx::Value* y = node->inputs()[2];
    if (x->elemType() != onnx::TensorProto_DataType_FLOAT ||
        y->elemType() != onnx::TensorProto_DataType_FLOAT) {
      continue;
    }
    if (onnx::optimization::FetchConstantTensor(x) != nullptr ||
        onnx::optimization::FetchConstantTensor(y) != nullptr) {
      continue;
    }
    add_name(x->uniqueName());
    add_name(y->uniqueName());
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorWhere(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_where (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_where reads the same calibration-ranges global every
  // other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, both operands' and the output's, so sharing the map is safe as
  // long as only one Quantize* entry point runs per call, which
  // OptimizeFixed's single pass-name list here ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_where"});
}

std::vector<std::string> ListQOperatorGemmQuantizableTensors(
    const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
  if (g.get() == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  std::unordered_set<std::string> seen;
  auto add_name = [&](const std::string& name) {
    if (seen.insert(name).second) {
      names.push_back(name);
    }
  };
  for (auto* node : g->nodes()) {
    onnx::optimization::onnxsim_passes::GemmQuantizableInfo info;
    if (!onnx::optimization::onnxsim_passes::MatchGemmQuantizable(node, info)) {
      continue;
    }
    add_name(info.a->uniqueName());
    add_name(node->output()->uniqueName());
  }
  return names;
}

onnx::ModelProto QuantizeQOperatorGemm(
    const onnx::ModelProto& model,
    const std::unordered_map<std::string, std::pair<float, float>>&
        activation_ranges) {
  PrepareSchemasForDebug(model);
  // Registers qoperator_quantize_gemm (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // qoperator_quantize_gemm reads the same calibration-ranges global every
  // other static/QOperator pass does (see
  // StaticQuantizationCalibrationRanges's doc comment) -- keyed by tensor
  // name, so sharing the map is safe as long as only one Quantize* entry
  // point runs per call, which OptimizeFixed's single pass-name list here
  // ensures.
  onnx::optimization::onnxsim_passes::StaticQuantizationCalibrationRanges() =
      activation_ranges;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"qoperator_quantize_gemm"});
}

onnx::ModelProto QuantizeFp16(const onnx::ModelProto& model,
                              bool keep_io_types) {
  PrepareSchemasForDebug(model);
  // Registers quantize_fp16 (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quantize_fp16 reads this the same way static_quantize_matmul.h's passes
  // read StaticQuantizationCalibrationRanges() -- OptimizeFixed's pass-name
  // list has no way to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuantizeFp16KeepIoTypes() = keep_io_types;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"quantize_fp16"});
}

onnx::ModelProto QuantizeBf16(const onnx::ModelProto& model,
                              bool keep_io_types) {
  PrepareSchemasForDebug(model);
  // Registers quantize_bf16 (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quantize_bf16 reads this the same way quantize_fp16 reads
  // QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no way
  // to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuantizeBf16KeepIoTypes() = keep_io_types;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"quantize_bf16"});
}

onnx::ModelProto QuantizeFp8(const onnx::ModelProto& model,
                             const std::string& format, bool keep_io_types) {
  onnx::optimization::onnxsim_passes::Float8Format target_format;
  if (format == "e4m3") {
    target_format = onnx::optimization::onnxsim_passes::Float8Format::kE4M3FN;
  } else if (format == "e5m2") {
    target_format = onnx::optimization::onnxsim_passes::Float8Format::kE5M2;
  } else {
    throw std::invalid_argument(
        "QuantizeFp8: format must be \"e4m3\" or \"e5m2\", got \"" + format +
        "\"");
  }
  PrepareSchemasForDebug(model);
  // Registers quantize_fp8 (idempotent) into onnxoptimizer's registry so
  // OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // quantize_fp8 reads these the same way quantize_fp16 reads
  // QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no way
  // to carry a parameter directly.
  onnx::optimization::onnxsim_passes::QuantizeFp8TargetFormat() = target_format;
  onnx::optimization::onnxsim_passes::QuantizeFp8KeepIoTypes() = keep_io_types;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"quantize_fp8"});
}
