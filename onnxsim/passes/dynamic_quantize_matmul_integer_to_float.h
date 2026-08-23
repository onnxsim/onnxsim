// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// The same "dynamic quantization" rewrite dynamic_quantize_matmul.h
// applies (see that file's doc comment for the full rationale: W is
// quantized once from its static values, X is quantized at runtime by
// DynamicQuantizeLinear, opset/shape/overflow restrictions are identical),
// but the dequantize step is a single ONNX Runtime "com.microsoft" contrib
// op, MatMulIntegerToFloat, instead of three-to-four separate standard-ONNX
// nodes:
//
// Before (illustrated for MatMul; a "vanilla" Gemm is handled the same way):
//   Y = MatMul(X, W)         W constant, 2-D, float32
// After:
//   Xq, Xs, Xzp = DynamicQuantizeLinear(X)
//   Y = MatMulIntegerToFloat(Xq, Wq, Xs, Ws, Xzp, [,Bias])  -- true int8
//       compute, dequantizes and (optionally) adds the bias in one op
//
// dynamic_quantize_matmul.h instead builds this as
//   Acc = MatMulInteger(Xq, Wq, Xzp)              -- int32
//   Y   = Cast<float>(Acc) * (Xs * Ws)             -- Cast + 2x Mul
//   Y   = Y + Bias                                 -- optional Add
// -- four to five nodes where this rewrite needs two (DynamicQuantizeLinear
// plus MatMulIntegerToFloat itself), since MatMulIntegerToFloat's own
// schema already has a `bias` input added directly to the dequantized
// result, with no separate scale-multiply or Add node required.
//
// A rewritten node is only reachable through ONNX Runtime's "com.microsoft"
// contrib domain, so this pass also adds that domain (version 1) to the
// model's opset imports the first time it fires, if not already present.

#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct DynamicQuantizeMatMulIntegerToFloat final : public PredicateBasedPass {
  explicit DynamicQuantizeMatMulIntegerToFloat()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dynamic_quantize_matmul_integer_to_float";
  }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 11) {
      return false;  // DynamicQuantizeLinear needs opset >= 11.
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t k =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    return IsSafeInt32ReductionDepth(k);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    if (info.x->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }
    const int64_t k =
        info.weight_transposed ? w_t->sizes()[1] : w_t->sizes()[0];
    if (!IsSafeInt32ReductionDepth(k)) {
      return false;
    }

    Tensor w_q;
    Tensor w_scale;
    QuantizeWeightPerChannelKN(*w_t, info.weight_transposed, w_q, w_scale);

    // Xq, Xs, Xzp = DynamicQuantizeLinear(X)
    Node* dql = graph.create(Symbol("DynamicQuantizeLinear"), 3);
    dql->addInput(info.x);
    dql->insertBefore(n);
    Value* x_q = dql->outputs()[0];
    Value* x_scale = dql->outputs()[1];
    Value* x_zp = dql->outputs()[2];
    x_q->setElemType(TensorProto_DataType_UINT8);
    x_scale->setElemType(TensorProto_DataType_FLOAT);
    x_zp->setElemType(TensorProto_DataType_UINT8);

    Value* w_q_value = graph.addInitializerAndCreateValue(w_q);
    Value* w_scale_value = graph.addInitializerAndCreateValue(w_scale);

    // Y = MatMulIntegerToFloat(Xq, Wq, Xs, Ws, Xzp, [,Bias]), a
    // "com.microsoft" contrib op -- b_zero_point (index 5) is omitted
    // (symmetric weight quantization, i.e. always 0), represented as the
    // standard ONNX empty-string placeholder for a middle-position optional
    // input.
    Node* mmitf = graph.create(Symbol("MatMulIntegerToFloat"), 1);
    mmitf->addInput(x_q);
    mmitf->addInput(w_q_value);
    mmitf->addInput(x_scale);
    mmitf->addInput(w_scale_value);
    mmitf->addInput(x_zp);
    Node* undef = graph.create(kUndefined, 1);
    undef->insertBefore(n);
    undef->output()->setUniqueName("");
    mmitf->addInput(undef->output());
    if (info.bias != nullptr) {
      mmitf->addInput(info.bias);
    }
    mmitf->setDomain("com.microsoft");
    mmitf->insertBefore(n);
    mmitf->output()->setElemType(TensorProto_DataType_FLOAT);

    if (n->output()->sizes().size() > 0) {
      mmitf->output()->setSizes(n->output()->sizes());
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
        tryReplacingAllUsesWith(n->output(), mmitf->output());
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
