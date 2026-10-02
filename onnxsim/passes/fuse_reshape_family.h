// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Reshape, Squeeze, Unsqueeze and Flatten all only rearrange a tensor's
// shape -- none of them ever change its total element count. So for any two
// adjacent ops drawn from that family:
//   Y = <family op 1>(X, ...)
//   Z = <family op 2>(Y, ...)
// Z is exactly Reshape(X, shape(Z)), and shape(Z) needs at most one free
// (not statically known) dimension to be expressible as a constant Reshape
// target -- the free slot can always be written as Reshape's `-1` sentinel,
// since the total element count flowing through the whole family chain is
// invariant regardless of which intermediate tensor it's computed from.
//
// Squeeze<->Squeeze, Unsqueeze<->Unsqueeze, Squeeze<->Unsqueeze and
// Reshape<->Reshape pairs already have their own dedicated fusions
// (fuse_consecutive_squeezes, fuse_consecutive_unsqueezes,
// fuse_consecutive_squeeze_unsqueeze, fuse_consecutive_reshapes /
// eliminate_consecutive_idempotent_ops) that work by composing each pair's
// own axis/shape rule and so still fuse when no shape is statically known.
// This pass instead leans on already-computed shape inference, so it only
// covers the mixed pairs those don't -- anything involving Flatten, plus
// Reshape next to Squeeze/Unsqueeze -- and only fires once the *output*
// shape of the pair has at most one unresolved dimension.
#include <algorithm>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseReshapeFamily final : public PredicateBasedPass {
  explicit FuseReshapeFamily()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "fuse_reshape_family"; }

  static bool IsFamily(uint32_t kind) {
    return kind == kReshape || kind == kSqueeze || kind == kUnsqueeze ||
           kind == kFlatten;
  }

  // Pairs already covered by a dedicated, shape-agnostic fusion elsewhere
  // (see the file comment above) -- left alone here so this pass only picks
  // up the mixed pairs those don't.
  static bool HandledElsewhere(uint32_t outer, uint32_t inner) {
    if (outer == kReshape && inner == kReshape) return true;
    if (outer == kSqueeze && inner == kSqueeze) return true;
    if (outer == kUnsqueeze && inner == kUnsqueeze) return true;
    if (outer == kSqueeze && inner == kUnsqueeze) return true;
    if (outer == kUnsqueeze && inner == kSqueeze) return true;
    return false;
  }

  bool patternMatchPredicate(Node* node) override {
    if (!IsFamily(node->kind()) || node->inputs().empty()) {
      return false;
    }
    Node* prev = node->input(0)->node();
    return IsFamily(prev->kind()) &&
           !HandledElsewhere(node->kind(), prev->kind());
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (!n->output()->has_sizes()) {
      // Nothing to collapse to without a target shape.
      return false;
    }
    const auto& dims = n->output()->sizes();
    if (dims.empty()) {
      // has_sizes() with zero dims is how a rank genuinely known to be 0
      // (a scalar) and "shape inference couldn't determine a rank at all"
      // both surface here -- e.g. Unsqueeze's shape inference can report
      // this for a symbolic-shaped input even though its output can never
      // actually be a scalar. Treat it as untrustworthy rather than risk
      // emitting a Reshape to the wrong (empty) target shape.
      return false;
    }
    std::vector<int64_t> shape;
    shape.reserve(dims.size());
    bool seen_unknown = false;
    for (const auto& d : dims) {
      if (d.is_int) {
        shape.push_back(d.dim);
        continue;
      }
      if (seen_unknown) {
        // A second free dimension can't be encoded by Reshape's single -1.
        return false;
      }
      seen_unknown = true;
      shape.push_back(-1);
    }

    // A resolved `0` can only be baked as a literal Reshape entry once
    // allowzero=1 (opset >= 14) makes it mean "this dim is 0" instead of
    // the default "copy this dim from the input" -- see
    // fuse_consecutive_reshapes.h's file comment for the full rationale.
    const bool has_zero = std::any_of(shape.begin(), shape.end(),
                                      [](int64_t d) { return d == 0; });
    if (has_zero && seen_unknown) {
      // allowzero=1 makes a literal zero mean zero, but ONNX forbids using
      // that spelling together with Reshape's inferred-dimension sentinel.
      return false;
    }
    if (has_zero && getOpsetVersion(graph) < 14) {
      return false;
    }

    Node* prev = PrevNode(n, 0);
    Value* new_input = prev->input(0);

    Tensor shape_t;
    shape_t.sizes().push_back(static_cast<int64_t>(shape.size()));
    shape_t.elem_type() = TensorProto_DataType_INT64;
    shape_t.int64s() = shape;
    Value* shape_v = graph.addInitializerAndCreateValue(shape_t);

    Node* new_reshape = graph.create(kReshape, 1);
    new_reshape->addInput(new_input);
    new_reshape->addInput(shape_v);
    if (has_zero) {
      new_reshape->i_(Symbol("allowzero"), 1);
    }
    new_reshape->insertBefore(n);

    // tryReplacingAllUsesWith redirects n's output uses to new_reshape's
    // output and copies n's output's sizes/elem type onto it.
    if (!tryReplacingAllUsesWith(n, new_reshape)) {
      new_reshape->destroy();
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    if (prev->output()->uses().empty()) {
      prev->destroy();
    }
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
