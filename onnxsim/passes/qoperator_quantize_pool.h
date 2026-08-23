// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes a standalone `AveragePool` or
// `GlobalAveragePool` node into ONNX Runtime's "com.microsoft" contrib ops
// QLinearAveragePool/QLinearGlobalAveragePool -- the pooling analogue of
// qoperator_quantize_activation.h's QLinearSigmoid/QLinearLeakyRelu rewrite
// (see that file's doc comment for why these are contrib, not standard,
// ONNX ops).
//
// Before (illustrated for AveragePool; GlobalAveragePool is identical but
// for the op/QLinear* name and having no attributes of its own to carry
// over):
//   Y = AveragePool(X, kernel_shape=k, pads=p, ...)   X: runtime float32
// After:
//   Xq = QuantizeLinear(X, Xs, Xzp)                            -- CALIBRATED
//   Yq = QLinearAveragePool(Xq, Xs, Xzp, Ys, Yzp,
//                           kernel_shape=k, pads=p, ...,
//                           channels_last=0)                   -- true int8
//   Y  = DequantizeLinear(Yq, Ys, Yzp)                         -- CALIBRATED
//
// Like qoperator_quantize_activation.h, there is only ever one operand here,
// so only its own calibrated range is needed on top of the output's --
// QOperator format computes directly in int8, so the output must be
// quantized too.
//
// Every attribute the original `AveragePool` node has (kernel_shape, pads,
// strides, ceil_mode, count_include_pad, auto_pad) is copied onto the
// rewritten node unchanged via `copyAttributes` -- `QLinearAveragePool`'s
// schema mirrors standard ONNX `AveragePool`'s own attribute set exactly, so
// no per-attribute translation is needed (`GlobalAveragePool` has no
// attributes of its own, so this is a no-op for that case). Both ops
// additionally take a `channels_last` attribute standard ONNX pooling
// doesn't have; onnxsim only ever produces NCHW-layout graphs, so this is
// always explicitly set to 0 (NCHW), never carried over from anywhere.
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.
//
// Standard ONNX AveragePool gained an optional `dilations` attribute in
// opset 19; ONNX Runtime's QLinearAveragePool kernel does not accept it
// ("Unrecognized attribute: dilations for operator QLinearAveragePool" is a
// real, observed ONNX Runtime error for this exact combination) -- so an
// AveragePool node carrying that attribute is left untouched entirely
// rather than producing a node ONNX Runtime would refuse to load.
//
// Only an AveragePool/GlobalAveragePool with exactly 1 input, float32, is
// matched; a node is only rewritten when both its input's name and its own
// output's name have a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeQOperatorPool in
// onnxsim.h).

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

// Identifies whether `n` is a node this pass handles, and if so whether it
// is the (attribute-bearing) AveragePool or the (attribute-free)
// GlobalAveragePool variant.
inline bool MatchQLinearPoolKind(Node* n, bool& is_global) {
  if (n->kind() == Symbol("AveragePool")) {
    is_global = false;
    return true;
  }
  if (n->kind() == Symbol("GlobalAveragePool")) {
    is_global = true;
    return true;
  }
  return false;
}

struct QOperatorQuantizePool final : public PredicateBasedPass {
  explicit QOperatorQuantizePool()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "qoperator_quantize_pool"; }

  bool patternMatchPredicate(Node* n) override {
    bool is_global = false;
    if (!MatchQLinearPoolKind(n, is_global)) {
      return false;
    }
    if (n->inputs().size() != 1) {
      return false;
    }
    if (n->hasAttribute(kdilations)) {
      return false;  // QLinearAveragePool's kernel rejects this attribute.
    }
    Value* x = n->inputs()[0];
    if (x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    return ranges.count(x->uniqueName()) != 0 &&
           ranges.count(n->output()->uniqueName()) != 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    bool is_global = false;
    if (!MatchQLinearPoolKind(n, is_global)) {
      return false;
    }
    if (n->inputs().size() != 1) {
      return false;
    }
    if (n->hasAttribute(kdilations)) {
      return false;  // QLinearAveragePool's kernel rejects this attribute.
    }
    Value* x = n->inputs()[0];
    if (x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto x_range_it = ranges.find(x->uniqueName());
    const auto y_range_it = ranges.find(n->output()->uniqueName());
    if (x_range_it == ranges.end() || y_range_it == ranges.end()) {
      return false;
    }

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

    // Yq = QLinear{AveragePool,GlobalAveragePool}(Xq, Xs, Xzp, Ys, Yzp,
    //      <original attributes>, channels_last=0), a "com.microsoft"
    // contrib op.
    Node* qlop = graph.create(
        Symbol(is_global ? "QLinearGlobalAveragePool" : "QLinearAveragePool"),
        1);
    qlop->addInput(xq->output());
    qlop->addInput(x_scale_v);
    qlop->addInput(x_zp_v);
    qlop->addInput(y_scale_v);
    qlop->addInput(y_zp_v);
    qlop->copyAttributes(*n);
    qlop->i_(Symbol("channels_last"), int64_t(0));
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
