// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <unordered_set>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct EliminateConsecutiveIdempotentOps final : public PredicateBasedPass {
  explicit EliminateConsecutiveIdempotentOps()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_consecutive_idempotent_ops";
  }

  bool patternMatchPredicate(Node* node) override {
    static const std::unordered_set<std::string> idempotent_ops = {
        "Ceil", "Floor", "Round", "Relu", "Reshape", "Sign"};
    for (const auto& op : idempotent_ops) {
      // TODO: support uses().size() > 1 for ops except Reshape
      if (CheckKind(node, Symbol(op), 0, Symbol(op)) &&
          node->input(0)->uses().size() == 1) {
        return true;
      }
    }
    return false;
  }

  // Unlike Ceil/Floor/Round/Relu/Sign (genuinely idempotent: f(f(x)) ==
  // f(x)), Reshape(Reshape(X, s1), s2) is only equivalent to Reshape(X, s2)
  // when s2's `0` entries (under the default allowzero=0, meaning "copy this
  // dim from the node's input") can't be disturbed by dropping the
  // intermediate reshape -- see fuse_consecutive_reshapes.h's file comment
  // for the full rationale. Upstream onnx-optimizer's version of this pass
  // fuses unconditionally, which silently changes which shape a `0` entry
  // copies from.
  static bool ReshapeFusionSafe(Node* node) {
    if (node->hasAttribute(Symbol("allowzero")) &&
        node->i(Symbol("allowzero")) == 1) {
      return true;
    }
    const Tensor* new_shape_tensor = FetchConstantTensor(node->inputs()[1]);
    if (!new_shape_tensor || new_shape_tensor->elem_type() !=
                                 ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
    const auto new_shape = ParseTensorData<int64_t>(new_shape_tensor);
    return std::none_of(new_shape.begin(), new_shape.end(),
                        [](int64_t d) { return d == 0; });
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (node->kind() == kReshape && !ReshapeFusionSafe(node)) {
      return false;
    }
    Node* previous_node = node->input(0)->node();
    const bool had_sizes = previous_node->input(0)->has_sizes();
    std::vector<Dimension> sizes = previous_node->input(0)->sizes();
    bool replacing_success =
        tryReplacingAllUsesWith(node->input(0), previous_node->input(0));
    if (replacing_success) {
      if (node->kind() == kReshape) {
        // Restore both dimensions and their presence: setSizes({}) would
        // incorrectly turn an unknown rank into a scalar (rank 0).
        if (had_sizes) {
          previous_node->input(0)->setSizes(sizes);
        } else {
          previous_node->input(0)->wipeSizes();
        }
      }
      return true;
    }
    return false;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
