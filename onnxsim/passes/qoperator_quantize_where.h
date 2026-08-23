// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Statically (calibration-based) quantizes a `Where` node whose two data
// operands are both non-constant float32 tensors into ONNX Runtime's
// "com.microsoft" contrib op QLinearWhere -- the ternary-select analogue of
// qoperator_quantize_elementwise.h's QLinearAdd/QLinearMul rewrite (see
// that file's doc comment for why these are contrib, not standard, ONNX
// ops).
//
// Before:
//   Z = Where(Cond, X, Y)   -- Cond: bool; X, Y: both runtime float32
// After:
//   Xq = QuantizeLinear(X, Xs, Xzp)                     -- Xs/Xzp: CALIBRATED
//   Yq = QuantizeLinear(Y, Ys, Yzp)                     -- Ys/Yzp: CALIBRATED
//   Zq = QLinearWhere(Cond, Xq, Xs, Xzp, Yq, Ys, Yzp, Zs, Zzp)
//        -- true int8 compute; Cond passes through unquantized (it's bool)
//   Z  = DequantizeLinear(Zq, Zs, Zzp)                  -- Zs/Zzp: CALIBRATED
//
// Like qoperator_quantize_elementwise.h's QLinearAdd/QLinearMul,
// QLinearWhere has no "weight" role -- both X and Y are calibrated as
// activations, on top of the output's own calibrated range (QOperator
// format computes directly in int8, so the output must be quantized too).
// The condition operand is never quantized: QLinearWhere's schema takes it
// as a plain `tensor(bool)`, passed straight through.
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.
//
// Only a Where whose X and Y are both non-constant float32 tensors is
// matched: a constant operand is better quantized from its own static
// values than force-fed through the runtime calibration harness as if it
// varied at inference time (same reasoning qoperator_quantize_elementwise.h
// applies to Add/Mul). A node is only rewritten when X's name, Y's name,
// and the node's own output's name all have a calibrated range (set via
// StaticQuantizationCalibrationRanges(), see QuantizeQOperatorWhere in
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

struct QOperatorQuantizeWhere final : public PredicateBasedPass {
  explicit QOperatorQuantizeWhere()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "qoperator_quantize_where";
  }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != Symbol("Where")) {
      return false;
    }
    if (n->inputs().size() != 3) {
      return false;
    }
    Value* x = n->inputs()[1];
    Value* y = n->inputs()[2];
    if (x->elemType() != TensorProto_DataType_FLOAT ||
        y->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (FetchConstantTensor(x) != nullptr ||
        FetchConstantTensor(y) != nullptr) {
      return false;  // Constant operand: quantize from its own values
                     // instead.
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    return ranges.count(x->uniqueName()) != 0 &&
           ranges.count(y->uniqueName()) != 0 &&
           ranges.count(n->output()->uniqueName()) != 0;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (n->kind() != Symbol("Where")) {
      return false;
    }
    if (n->inputs().size() != 3) {
      return false;
    }
    Value* cond = n->inputs()[0];
    Value* x = n->inputs()[1];
    Value* y = n->inputs()[2];
    if (x->elemType() != TensorProto_DataType_FLOAT ||
        y->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    if (FetchConstantTensor(x) != nullptr ||
        FetchConstantTensor(y) != nullptr) {
      return false;
    }
    const auto& ranges = StaticQuantizationCalibrationRanges();
    const auto x_range_it = ranges.find(x->uniqueName());
    const auto y_range_it = ranges.find(y->uniqueName());
    const auto z_range_it = ranges.find(n->output()->uniqueName());
    if (x_range_it == ranges.end() || y_range_it == ranges.end() ||
        z_range_it == ranges.end()) {
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
    float z_scale_f = 1.0f;
    int32_t z_zp_i = 0;
    ComputeAsymmetricUint8QuantParams(
        z_range_it->second.first, z_range_it->second.second, z_scale_f, z_zp_i);

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

    Tensor z_scale_t;
    z_scale_t.elem_type() = TensorProto_DataType_FLOAT;
    z_scale_t.floats() = {z_scale_f};
    Tensor z_zp_t;
    z_zp_t.elem_type() = TensorProto_DataType_UINT8;
    z_zp_t.int32s() = {z_zp_i};
    Value* z_scale_v = graph.addInitializerAndCreateValue(z_scale_t);
    Value* z_zp_v = graph.addInitializerAndCreateValue(z_zp_t);

    // Xq = QuantizeLinear(X, Xs, Xzp)
    Node* xq = graph.create(Symbol("QuantizeLinear"), 1);
    xq->addInput(x);
    xq->addInput(x_scale_v);
    xq->addInput(x_zp_v);
    xq->insertBefore(n);
    xq->output()->setElemType(TensorProto_DataType_UINT8);

    // Yq = QuantizeLinear(Y, Ys, Yzp)
    Node* yq = graph.create(Symbol("QuantizeLinear"), 1);
    yq->addInput(y);
    yq->addInput(y_scale_v);
    yq->addInput(y_zp_v);
    yq->insertBefore(n);
    yq->output()->setElemType(TensorProto_DataType_UINT8);

    // Zq = QLinearWhere(Cond, Xq, Xs, Xzp, Yq, Ys, Yzp, Zs, Zzp), a
    // "com.microsoft" contrib op.
    Node* qlop = graph.create(Symbol("QLinearWhere"), 1);
    qlop->addInput(cond);
    qlop->addInput(xq->output());
    qlop->addInput(x_scale_v);
    qlop->addInput(x_zp_v);
    qlop->addInput(yq->output());
    qlop->addInput(y_scale_v);
    qlop->addInput(y_zp_v);
    qlop->addInput(z_scale_v);
    qlop->addInput(z_zp_v);
    qlop->setDomain("com.microsoft");
    qlop->insertBefore(n);
    qlop->output()->setElemType(TensorProto_DataType_UINT8);

    // Z = DequantizeLinear(Zq, Zs, Zzp)
    Node* dq = graph.create(Symbol("DequantizeLinear"), 1);
    dq->addInput(qlop->output());
    dq->addInput(z_scale_v);
    dq->addInput(z_zp_v);
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
