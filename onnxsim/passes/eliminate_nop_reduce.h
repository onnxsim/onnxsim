// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// A Reduce* over axes whose extents are all statically 1 combines exactly one
// element per output position, so for Sum, Mean, Max, Min and Prod the result
// is the input value itself:
//
//   keepdims=1:  Y = ReduceX(X, axes)   ->   Y = X
//   keepdims=0:  Y = ReduceX(X, axes)   ->   Y = Squeeze(X, axes)
//
// The same holds when the reduction has no axes at all (a rank-0 input, or an
// empty `axes` under noop_with_empty_axes=1), where the node is a plain
// identity regardless of keepdims.
//
// ReduceL1/L2/SumSquare/LogSum/LogSumExp are deliberately left alone: they
// apply a non-identity elementwise map (|x|, x^2, log) to the lone element,
// so the node is not a no-op.

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/fuse_consecutive_reduce.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct EliminateNopReduce final : public PredicateBasedPass {
  explicit EliminateNopReduce()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "eliminate_nop_reduce"; }

  static bool IsIdentityOnSingletonKind(NodeKind k) {
    static const std::unordered_set<NodeKind> kKinds{
        kReduceSum, kReduceMean, kReduceMax, kReduceMin, kReduceProd};
    return kKinds.count(k) != 0;
  }

  // Resolves `node`'s axes and checks each one is a statically-1 dimension of
  // its input. On success `axes` is sorted/deduped (empty for the identity
  // cases). Rank must be known whenever there are axes to check.
  static bool MatchUnitAxes(Node* node, std::vector<int64_t>& axes) {
    Value* x = node->input(0);
    const bool have_rank = x->has_sizes();
    const int64_t rank =
        have_rank ? static_cast<int64_t>(x->sizes().size()) : 0;

    bool is_identity = false;
    if (!FuseConsecutiveReduce::ResolveAxes(node, have_rank, rank, axes,
                                            is_identity)) {
      return false;
    }
    if (is_identity) {
      axes.clear();
      return true;
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    if (axes.empty()) {
      return true;  // rank-0 input: nothing to reduce
    }
    if (!have_rank) {
      return false;
    }
    for (int64_t a : axes) {
      if (a < 0 || a >= rank) {
        return false;
      }
      const Dimension& d = x->sizes()[static_cast<size_t>(a)];
      if (!d.is_int || d.dim != 1) {
        return false;
      }
    }
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    if (!IsIdentityOnSingletonKind(node->kind()) || node->inputs().empty()) {
      return false;
    }
    std::vector<int64_t> axes;
    return MatchUnitAxes(node, axes);
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    std::vector<int64_t> axes;
    if (!MatchUnitAxes(node, axes)) {
      return false;
    }

    int64_t keepdims = 1;
    GetValueFromAttr(node, kkeepdims, keepdims);

    if (axes.empty() || keepdims != 0) {
      if (!tryReplacingAllUsesWith(node->output(), node->input(0))) {
        return false;
      }
      destroy_current = NodeDestroyType::DestroyOne;
      return true;
    }

    // keepdims=0: drop exactly the reduced (size-1) axes.
    Node* squeeze = graph.create(kSqueeze, 1);
    squeeze->addInput(node->input(0));
    const int opset = getOpsetVersion(graph);
    if (opset != 0 && opset < 13) {
      squeeze->is_(kaxes, std::vector<int64_t>(axes));
    } else {
      Tensor axes_t;
      axes_t.elem_type() = TensorProto_DataType_INT64;
      axes_t.sizes().push_back(static_cast<int64_t>(axes.size()));
      axes_t.int64s().assign(axes.begin(), axes.end());
      squeeze->addInput(graph.addInitializerAndCreateValue(std::move(axes_t)));
    }
    squeeze->output()->copyMetadata(node->output());
    squeeze->insertBefore(node);
    if (!tryReplacingAllUsesWith(node, squeeze)) {
      squeeze->destroy();
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
