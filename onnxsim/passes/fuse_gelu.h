// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Fuses the exact-erf GELU decomposition
//   Y = 0.5 * X * (1 + Erf(X / sqrt(2)))
// -- however the two commutative Mul/Add operands are ordered, and whichever
// of X's two uses is written first -- into a single ``Gelu`` node (the
// ``approximate="none"`` default, i.e. exactly this formula). ``Gelu`` was
// only added to the default (`""`) domain at opset 20, so this only fires
// when the graph's opset already supports it; an older-opset graph is left
// with the decomposition (a caller wanting the fusion can convert the model
// to opset >= 20 first via ``Simplify``'s ``target_opset_version``).
//
// Formula constants must equal the nearest representation in their source
// tensor type; nearby values describe a different computation.

#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct FuseGelu final : public PredicateBasedPass {
  explicit FuseGelu()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_gelu"; }

  static bool ExtractScalarConstant(Value* v, double& out) {
    const Tensor* t = FetchConstantTensor(v);
    if (t == nullptr) {
      return false;
    }
    int64_t numel = 1;
    for (const auto& s : t->sizes()) {
      numel *= s;
    }
    if (numel != 1) {
      return false;
    }
    if (t->elem_type() == TensorProto_DataType_FLOAT) {
      out = t->is_raw_data()
                ? ReadRawDataHostOrder<float>(t->data<float>(), 1)[0]
                : t->floats()[0];
      return true;
    }
    if (t->elem_type() == TensorProto_DataType_DOUBLE) {
      out = t->is_raw_data()
                ? ReadRawDataHostOrder<double>(t->data<double>(), 1)[0]
                : t->doubles()[0];
      return true;
    }
    return false;
  }

  static bool IsScalarConstantExact(Value* v, double target) {
    const Tensor* t = FetchConstantTensor(v);
    if (t == nullptr) {
      return false;
    }
    double val = 0.0;
    if (!ExtractScalarConstant(v, val)) {
      return false;
    }
    return t->elem_type() == TensorProto_DataType_FLOAT
               ? static_cast<float>(val) == static_cast<float>(target)
               : val == target;
  }

  // Given a 2-input commutative node, returns the non-constant operand in
  // `other` when exactly one operand matches the scalar constant `target`.
  static bool MatchOneConstOperand(Node* n, double target, Value*& other) {
    if (n->inputs().size() != 2) {
      return false;
    }
    Value* a = n->input(0);
    Value* b = n->input(1);
    const bool a_is = IsScalarConstantExact(a, target);
    const bool b_is = IsScalarConstantExact(b, target);
    if (a_is == b_is) {
      return false;  // need exactly one match (rules out both/neither).
    }
    other = a_is ? b : a;
    return true;
  }

  struct Match {
    Value* x = nullptr;
    // t3, t2, t1, t0's nodes, closest-to-output first (i.e. already in
    // reverse-dependency order) -- see FuseLayerNorm::Match::dead_chain's
    // comment for why runTransform destroys these itself instead of
    // leaving them for a later eliminate_deadend round.
    std::vector<Node*> dead_chain;
  };

  // Finds the exact-erf GELU chain feeding the final ``0.5 *`` Mul `n`.
  static bool TryMatch(Node* n, Match& m) {
    if (n->kind() != kMul) {
      return false;
    }
    Value* t3 = nullptr;
    if (!MatchOneConstOperand(n, 0.5, t3)) {
      return false;
    }
    Node* t3_node = t3->node();
    if (t3_node->kind() != kMul || t3_node->inputs().size() != 2) {
      return false;
    }
    // Try both operand orders for which side is X vs. the (1 + Erf(..)) term.
    for (const auto& order :
         {std::make_pair(t3_node->input(0), t3_node->input(1)),
          std::make_pair(t3_node->input(1), t3_node->input(0))}) {
      Value* cand_x = order.first;
      Value* t2 = order.second;
      Node* t2_node = t2->node();
      if (t2_node->kind() != kAdd) {
        continue;
      }
      Value* t1 = nullptr;
      if (!MatchOneConstOperand(t2_node, 1.0, t1)) {
        continue;
      }
      Node* t1_node = t1->node();
      if (t1_node->kind() != Symbol("Erf") || t1_node->inputs().size() != 1) {
        continue;
      }
      Node* t0_node = t1_node->input(0)->node();
      if (t0_node->kind() != kDiv || t0_node->inputs().size() != 2) {
        continue;
      }
      // Div is not commutative: numerator must be the same X, denominator
      // must be sqrt(2).
      if (t0_node->input(0) != cand_x) {
        continue;
      }
      if (!IsScalarConstantExact(t0_node->input(1), std::sqrt(2.0))) {
        continue;
      }
      m.x = cand_x;
      m.dead_chain = {t3_node, t2_node, t1_node, t0_node};
      return true;
    }
    return false;
  }

  bool patternMatchPredicate(Node* n) override {
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 20) {
      return false;  // Gelu is only in the default domain from opset 20.
    }
    Match m;
    return TryMatch(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!TryMatch(n, m)) {
      return false;
    }
    Node* gelu = graph.create(Symbol("Gelu"), 1);
    gelu->addInput(m.x);
    gelu->insertBefore(n);
    gelu->output()->setElemType(n->output()->elemType());
    if (n->output()->has_sizes()) {
      gelu->output()->setSizes(n->output()->sizes());
    }
    n->output()->replaceAllUsesWith(gelu->output());
    // `n` is destroyed by the caller below, but only *after* runTransform
    // returns, so it still counts as a use of its own inputs right now --
    // sever them first (see FuseLayerNorm's identical comment for why).
    n->removeAllInputs();
    // Drop the rest of the now-dead decomposition ourselves rather than
    // waiting for a later eliminate_deadend round.
    for (Node* dead : m.dead_chain) {
      if (!dead->hasUsesInCurrentGraph()) {
        dead->destroy();
      }
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
