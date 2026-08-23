// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes a standalone `Softmax` node into
// ONNX Runtime's "com.microsoft" contrib op QLinearSoftmax -- the
// reduction-axis analogue of qoperator_quantize_activation.h's
// QLinearSigmoid/QLinearLeakyRelu rewrite (see that file's doc comment for
// why these are contrib, not standard, ONNX ops).
//
// Before:
//   Y = Softmax(X, axis=ax)     X: runtime float32 tensor
// After:
//   Xq = QuantizeLinear(X, Xs, Xzp)                          -- CALIBRATED
//   Yq = QLinearSoftmax(Xq, Xs, Xzp, Ys, Yzp,
//                       axis=ax, opset=default_domain_opset) -- true int8
//   Y  = DequantizeLinear(Yq, Ys, Yzp)                       -- CALIBRATED
//
// Like qoperator_quantize_activation.h, there is only ever one operand here,
// so only its own calibrated range is needed on top of the output's --
// QOperator format computes directly in int8, so the output must be
// quantized too.
//
// `QLinearSoftmax`'s `opset` attribute tells ONNX Runtime's kernel which of
// standard ONNX's two incompatible `Softmax` axis semantics to replicate:
// pre-opset-13 `Softmax` flattens the tensor into a 2-D matrix at `axis` and
// reduces over the trailing dimension, while opset-13+ `Softmax` reduces over
// `axis` in place, same rank in and out. Rather than guess or hardcode a
// version, this pass reads the *model's own* declared default-domain (""/
// "ai.onnx") opset from its opset imports and passes that through verbatim,
// so the rewritten node reproduces the exact semantics the original `Softmax`
// node already had. A model with no resolvable default-domain opset import
// is left untouched -- there is no safe default to guess.
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.
//
// Only a Softmax with exactly 1 input, float32, is matched; a node is only
// rewritten when both its input's name and its own output's name have a
// calibrated range (set via StaticQuantizationCalibrationRanges(), see
// QuantizeQOperatorSoftmax in onnxsim.h).

#pragma once

#include <cstdint>
#include <string>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/static_quantize_matmul.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Resolves the model's own default-domain ("" / "ai.onnx") opset version
// from its opset imports. Returns 0 if it cannot be determined (no import
// for the default domain).
inline int64_t DefaultDomainOpsetVersion(Graph& graph) {
  for (const OpSetID& opset : graph.opset_versions_mutable()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return opset.version();
    }
  }
  return 0;
}

struct QOperatorQuantizeSoftmax final : public PredicateBasedPass {
  explicit QOperatorQuantizeSoftmax()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "qoperator_quantize_softmax";
  }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != Symbol("Softmax")) {
      return false;
    }
    if (n->inputs().size() != 1) {
      return false;
    }
    Value* x = n->inputs()[0];
    if (x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (DefaultDomainOpsetVersion(*n->owningGraph()) <= 0) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    return ranges.count(x->uniqueName()) != 0 &&
           ranges.count(n->output()->uniqueName()) != 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (n->kind() != Symbol("Softmax")) {
      return false;
    }
    if (n->inputs().size() != 1) {
      return false;
    }
    Value* x = n->inputs()[0];
    if (x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const int64_t opset_version = DefaultDomainOpsetVersion(graph);
    if (opset_version <= 0) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto x_range_it = ranges.find(x->uniqueName());
    const auto y_range_it = ranges.find(n->output()->uniqueName());
    if (x_range_it == ranges.end() || y_range_it == ranges.end()) {
      return false;
    }
    const int64_t axis = GetValueFromAttrWithDefault(n, kaxis, int64_t(-1));

    float x_scale_f = 1.0f;
    int32_t x_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        x_range_it->second.first, x_range_it->second.second, x_scale_f, x_zp_i);
    float y_scale_f = 1.0f;
    int32_t y_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        y_range_it->second.first, y_range_it->second.second, y_scale_f, y_zp_i);

    Tensor x_scale_t;
    x_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    x_scale_t.floats() = {x_scale_f};
    Tensor x_zp_t;
    x_zp_t.elem_type() = TensorProto_DataType_UINT8;
    x_zp_t.int32s() = {x_zp_i};
    Value* x_scale_v = graph.addInitializerAndCreateValue(x_scale_t);
    Value* x_zp_v = graph.addInitializerAndCreateValue(x_zp_t);

    Tensor y_scale_t;
    y_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    y_scale_t.floats() = {y_scale_f};
    Tensor y_zp_t;
    y_zp_t.elem_type() = TensorProto_DataType_UINT8;
    y_zp_t.int32s() = {y_zp_i};
    Value* y_scale_v = graph.addInitializerAndCreateValue(y_scale_t);
    Value* y_zp_v = graph.addInitializerAndCreateValue(y_zp_t);

    // Xq = QuantizeLinear(X, Xs, Xzp)
    Node* xq = graph.create(Symbol("QuantizeLinear"), 1);
    xq->addInput(x);
    xq->addInput(x_scale_v);
    xq->addInput(x_zp_v);
    xq->insertBefore(n);
    xq->output()->setElemType(TensorProto_DataType_UINT8);

    // Yq = QLinearSoftmax(Xq, Xs, Xzp, Ys, Yzp, axis=axis,
    //                      opset=opset_version), a "com.microsoft" contrib
    // op.
    Node* qlop = graph.create(Symbol("QLinearSoftmax"), 1);
    qlop->addInput(xq->output());
    qlop->addInput(x_scale_v);
    qlop->addInput(x_zp_v);
    qlop->addInput(y_scale_v);
    qlop->addInput(y_zp_v);
    qlop->i_(kaxis, axis);
    qlop->i_(Symbol("opset"), opset_version);
    qlop->setDomain("com.microsoft");
    qlop->insertBefore(n);
    qlop->output()->setElemType(TensorProto_DataType_UINT8);

    // Y = DequantizeLinear(Yq, Ys, Yzp)
    Node* dq = graph.create(Symbol("DequantizeLinear"), 1);
    dq->addInput(qlop->output());
    dq->addInput(y_scale_v);
    dq->addInput(y_zp_v);
    dq->insertBefore(n);
    dq->output()->setElemType(TensorProto_DataType_FLOAT);
    if (n->output()->sizes().size() > 0) {
      dq->output()->setSizes(n->output()->sizes());
    }

    bool has_ms_domain = false;
    for (const OpSetID& opset : graph.opset_versions_mutable()) {
      if (opset.domain() == "com.microsoft") {
        has_ms_domain = true;
        break;
      }
    }
    if (!has_ms_domain) {
      graph.opset_versions_mutable().emplace_back("com.microsoft", 1);
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(n->output(), dq->output());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
