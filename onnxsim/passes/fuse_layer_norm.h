// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Fuses the textbook last-axis LayerNorm decomposition
//   mean = ReduceMean(X, axes=[-1])
//   diff = X - mean
//   var  = ReduceMean(diff^2, axes=[-1])          // diff^2 via Mul or Pow(.,2)
//   Y    = diff / Sqrt(var + epsilon) * scale + bias
// -- however the commutative Add/Mul operands are ordered -- into a single
// ``LayerNormalization`` node (axis=-1). That op was only added to the
// default (`""`) domain at opset 17, so this only fires when the graph's
// opset already supports it.
//
// Only the last axis is handled (the common case: normalizing over a
// trailing feature dimension), matched via each ReduceMean's ``axes``
// attribute -- so a graph already using opset >= 18's ReduceMean, which
// moved ``axes`` from an attribute to an optional input, is left alone
// (the same conservative "recognize the common, unambiguous shape, leave
// everything else as-is" policy the quantization passes use). ``scale`` and
// ``bias`` must be constant 0-D/1-D tensors whose size matches X's
// statically-known last dimension, since ``LayerNormalization`` requires
// their shape to exactly equal the normalized dimensions.

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

struct FuseLayerNorm final : public PredicateBasedPass {
  explicit FuseLayerNorm()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_layer_norm"; }

  struct Match {
    Value* x = nullptr;
    Value* scale = nullptr;
    Value* bias = nullptr;
    double epsilon = 1e-5;
    // The decomposition's nodes, closest-to-output first (i.e. already in
    // reverse-dependency order): once `n` itself is gone, each becomes
    // unused exactly when the one before it in this list is destroyed. Kept
    // so runTransform can drop them immediately instead of waiting for a
    // later eliminate_deadend round -- see its doc comment.
    std::vector<Node*> dead_chain;
  };

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

  // Splits a 2-input commutative node into (the operand that is a scalar
  // constant, the other operand); fails unless exactly one operand qualifies.
  static bool MatchScalarConstOperand(Node* n, Value*& constv, Value*& other) {
    if (n->inputs().size() != 2) {
      return false;
    }
    Value* a = n->input(0);
    Value* b = n->input(1);
    double dummy = 0.0;
    const bool a_is = ExtractScalarConstant(a, dummy);
    const bool b_is = ExtractScalarConstant(b, dummy);
    if (a_is == b_is) {
      return false;
    }
    constv = a_is ? a : b;
    other = a_is ? b : a;
    return true;
  }

  // Same, but for the operand being a constant tensor of rank <= 1 (a
  // per-channel vector or scalar), used for LayerNorm's scale/bias.
  static bool MatchVectorConstOperand(Node* n, Value*& constv, Value*& other) {
    if (n->inputs().size() != 2) {
      return false;
    }
    Value* a = n->input(0);
    Value* b = n->input(1);
    auto is_vec = [](Value* v) {
      const Tensor* t = FetchConstantTensor(v);
      return t != nullptr && t->sizes().size() <= 1;
    };
    const bool a_is = is_vec(a);
    const bool b_is = is_vec(b);
    if (a_is == b_is) {
      return false;
    }
    constv = a_is ? a : b;
    other = a_is ? b : a;
    return true;
  }

  // A ReduceMean whose single reduction axis is unambiguously the last axis
  // of its input, with `keepdims` at its default (1) so its output still
  // broadcasts against the input for the surrounding Sub/Div.
  static bool ReducesLastAxisKeepdims(Node* reduce_n) {
    if (reduce_n->kind() != kReduceMean || reduce_n->inputs().size() != 1) {
      return false;
    }
    const auto axes =
        GetValueFromAttrWithDefault<std::vector<int64_t>>(reduce_n, kaxes, {});
    if (axes.size() != 1) {
      return false;
    }
    const int64_t keepdims =
        GetValueFromAttrWithDefault<int64_t>(reduce_n, kkeepdims, int64_t(1));
    if (keepdims != 1) {
      return false;
    }
    if (axes[0] == -1) {
      return true;
    }
    Value* in = reduce_n->input(0);
    return in->has_sizes() && !in->sizes().empty() &&
           axes[0] == static_cast<int64_t>(in->sizes().size()) - 1;
  }

  static bool TryMatch(Node* n, Match& m) {
    if (n->kind() != kAdd) {
      return false;
    }
    Value* bias_v = nullptr;
    Value* scaled = nullptr;
    if (!MatchVectorConstOperand(n, bias_v, scaled)) {
      return false;
    }
    Node* scaled_node = scaled->node();
    if (scaled_node->kind() != kMul) {
      return false;
    }
    Value* scale_v = nullptr;
    Value* inv = nullptr;
    if (!MatchVectorConstOperand(scaled_node, scale_v, inv)) {
      return false;
    }
    Node* inv_node = inv->node();
    if (inv_node->kind() != kDiv || inv_node->inputs().size() != 2) {
      return false;
    }
    Value* diff = inv_node->input(0);
    Node* std_node = inv_node->input(1)->node();
    if (std_node->kind() != kSqrt || std_node->inputs().size() != 1) {
      return false;
    }
    Node* add_eps_node = std_node->input(0)->node();
    if (add_eps_node->kind() != kAdd) {
      return false;
    }
    Value* eps_v = nullptr;
    Value* var = nullptr;
    if (!MatchScalarConstOperand(add_eps_node, eps_v, var)) {
      return false;
    }
    double eps = 0.0;
    if (!ExtractScalarConstant(eps_v, eps)) {
      return false;
    }
    if (static_cast<double>(static_cast<float>(eps)) != eps) {
      return false;
    }
    Node* var_node = var->node();
    if (!ReducesLastAxisKeepdims(var_node)) {
      return false;
    }
    Value* sq = var_node->input(0);
    Node* diff_node = diff->node();
    if (diff_node->kind() != kSub || diff_node->inputs().size() != 2) {
      return false;
    }
    Value* x = diff_node->input(0);
    Node* mean_node = diff_node->input(1)->node();
    if (!ReducesLastAxisKeepdims(mean_node) || mean_node->input(0) != x) {
      return false;
    }
    // sq = diff^2, via Mul(diff, diff) or Pow(diff, 2).
    Node* sq_node = sq->node();
    if (sq_node->kind() == kMul) {
      if (sq_node->inputs().size() != 2 || sq_node->input(0) != diff ||
          sq_node->input(1) != diff) {
        return false;
      }
    } else if (sq_node->kind() == kPow) {
      if (sq_node->inputs().size() != 2 || sq_node->input(0) != diff ||
          !IsScalarConstantExact(sq_node->input(1), 2.0)) {
        return false;
      }
    } else {
      return false;
    }

    // LayerNormalization requires scale/bias to exactly match X's normalized
    // (here: last) dimension -- only fire when that is statically known.
    if (!x->has_sizes() || x->sizes().empty() || !x->sizes().back().is_int) {
      return false;
    }
    const int64_t channels = x->sizes().back().dim;
    const Tensor* scale_t = FetchConstantTensor(scale_v);
    const Tensor* bias_t = FetchConstantTensor(bias_v);
    if (scale_t == nullptr || bias_t == nullptr) {
      return false;
    }
    const int64_t scale_sz = scale_t->sizes().empty() ? 1 : scale_t->sizes()[0];
    const int64_t bias_sz = bias_t->sizes().empty() ? 1 : bias_t->sizes()[0];
    if (scale_sz != channels || bias_sz != channels) {
      return false;
    }

    m.x = x;
    m.scale = scale_v;
    m.bias = bias_v;
    m.epsilon = eps;
    // Closest-to-output first, so each node is already unused by the time
    // runTransform destroys it -- see the field's own comment.
    m.dead_chain = {scaled_node, inv_node, std_node,  add_eps_node,
                    var_node,    sq_node,  diff_node, mean_node};
    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    const int opset = getOpsetVersion(*n->owningGraph());
    if (opset != 0 && opset < 17) {
      return false;  // LayerNormalization is only in the default domain from
                     // opset 17.
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
    Node* ln = graph.create(Symbol("LayerNormalization"), 1);
    ln->addInput(m.x);
    ln->addInput(m.scale);
    ln->addInput(m.bias);
    ln->i_(kaxis, -1);
    ln->f_(kepsilon, static_cast<float>(m.epsilon));
    ln->insertBefore(n);
    ln->output()->setElemType(n->output()->elemType());
    if (n->output()->has_sizes()) {
      ln->output()->setSizes(n->output()->sizes());
    }
    n->output()->replaceAllUsesWith(ln->output());
    // `n` itself is destroyed by the caller (destroy_current below), but only
    // *after* runTransform returns -- so `n` still counts as a use of its own
    // inputs right now. Sever them first, or the top of `dead_chain` would
    // still look used and this whole cascade would stop before it starts.
    n->removeAllInputs();
    // The decomposition nodes `n` and its inputs fed are now unreachable from
    // any live output, but nothing destroys them automatically -- a
    // PredicateBasedPass only ever removes the one node it matched on (see
    // pass.cc's _runPassInternal). Drop them here, in dependency order,
    // rather than waiting for a later eliminate_deadend round: each becomes
    // unused (in the current graph) exactly when the entry before it in
    // `dead_chain` is destroyed, by construction (see that field's comment),
    // so this is a plain linear sweep, not a search.
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
