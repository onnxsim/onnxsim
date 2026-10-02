// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Y = Conv(X, W[, B])
//   Z = Y * S
// After:
//   Z = Conv(X, W', [B'])   with  W' = W * S,  B' = B * S
//
// S must be a constant that scales the Conv output per output channel: either a
// single element (scalar broadcast) or a tensor whose only non-unit dimension
// is the channel axis of the Conv output (axis 1), e.g. [C], [C, 1, 1] or [1,
// C, 1, 1]. The scale is folded into the convolution weights (and bias, if
// present) by creating Mul nodes on the constant weights, which the constant
// folder then materialises; the leftover multiplicative op disappears.
//
// This recovers the fusion of an affine scale that a framework exported as a
// standalone Mul -- most commonly the multiplicative half of a BatchNorm
// written as ``Conv -> Mul -> Add``. Once the Mul is folded in, the following
// bias Add is picked up by fuse_add_bias_into_conv, collapsing the whole affine
// tail into the Conv. Only Conv (not ConvTranspose) is handled, because its
// weight layout puts the output channel on axis 0.

#include <numeric>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseMulIntoConv final : public PredicateBasedPass {
  explicit FuseMulIntoConv()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_mul_into_conv"; }

  // Index (0 or 1) of the Mul operand that is a foldable Conv output -- a Conv
  // used only by this Mul and whose weight is constant -- or -1 if neither is.
  static int ConvOperandIndex(Node* mul) {
    for (int i = 0; i < 2; ++i) {
      Value* v = mul->inputs()[i];
      if (v->node()->kind() == kConv && v->uses().size() == 1 &&
          IsConstantTensor(v->node(), 1)) {
        return i;
      }
    }
    return -1;
  }

  bool patternMatchPredicate(Node* n) override {
    if (n->kind() != kMul || n->inputs().size() != 2) {
      return false;
    }
    const int conv_idx = ConvOperandIndex(n);
    if (conv_idx < 0) {
      return false;
    }
    return IsConstantTensor(n->inputs()[1 - conv_idx]);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const int conv_idx = ConvOperandIndex(n);
    if (conv_idx < 0) {
      return false;
    }
    Value* conv_out = n->inputs()[conv_idx];
    Value* scale = n->inputs()[1 - conv_idx];
    Node* conv = conv_out->node();
    Value* weight = conv->inputs()[1];

    const Tensor* weight_t = FetchConstantTensor(weight);
    const Tensor* scale_t = FetchConstantTensor(scale);
    if (weight_t == nullptr || scale_t == nullptr) {
      return false;
    }
    // The scale is multiplied against the constant weight, so the element types
    // must agree.
    if (weight_t->elem_type() != scale_t->elem_type()) {
      return false;
    }

    // Conv weight is [C_out, C_in / groups, k...]; the output channel is axis
    // 0.
    const auto& w_sizes = weight_t->sizes();
    if (w_sizes.size() < 2) {
      return false;
    }
    const int64_t C = w_sizes[0];
    const int64_t w_rank = static_cast<int64_t>(w_sizes.size());

    // Validate that `scale` scales the Conv output per output channel: a scalar
    // (single element) or a tensor whose only non-unit axis equals C and, when
    // right-aligned against the Conv output rank (== w_rank), lands on the
    // channel axis (index 1).
    const auto& s_sizes = scale_t->sizes();
    int64_t s_numel = 1;
    for (const int64_t d : s_sizes) {
      s_numel *= d;
    }
    bool per_channel;
    if (s_numel == 1) {
      per_channel = false;  // scalar: broadcasts to every channel
    } else if (s_numel == C) {
      const int64_t s_rank = static_cast<int64_t>(s_sizes.size());
      int64_t non_unit_axis = -1;
      for (int64_t i = 0; i < s_rank; ++i) {
        if (s_sizes[i] != 1) {
          if (non_unit_axis != -1) {
            return false;  // more than one non-unit axis
          }
          non_unit_axis = i;
        }
      }
      if (non_unit_axis < 0 || w_rank - s_rank + non_unit_axis != 1) {
        return false;  // the C-sized axis is not the channel axis
      }
      per_channel = true;
    } else {
      return false;
    }

    // If the Conv has a bias it must be constant too, so it can be scaled.
    const bool has_bias = conv->inputs().size() == 3;
    if (has_bias) {
      Value* bias = conv->inputs()[2];
      const Tensor* bias_t = FetchConstantTensor(bias);
      if (bias_t == nullptr || bias_t->elem_type() != scale_t->elem_type()) {
        return false;
      }
    }

    // --- All checks passed; build the scaled weight (and bias). ---

    // `bias_scale` is the [C] (or scalar) scale used for the bias;
    // `weight_scale` is that scale broadcast to the weight rank [C, 1, ...] for
    // the weight.
    Value* bias_scale = scale;
    Value* weight_scale = scale;
    if (!per_channel && has_bias && s_sizes.size() > 1) {
      // A singleton tensor is scalar-valued for broadcasting, but Conv's
      // optional bias must remain rank 1.  Flatten shapes such as [1,1,1,1]
      // before multiplying them into the [C] bias.
      Node* reshape = graph.create(kReshape, 1);
      reshape->addInput(scale);
      Tensor shape_t;
      shape_t.elem_type() = TensorProto_DataType_INT64;
      shape_t.sizes().push_back(1);
      shape_t.int64s().push_back(1);
      reshape->addInput(graph.addInitializerAndCreateValue(shape_t));
      reshape->insertBefore(conv);
      bias_scale = reshape->output();
    }
    if (per_channel) {
      // Flatten the scale to [C].
      Value* scale_1d = scale;
      if (static_cast<int64_t>(s_sizes.size()) != 1) {
        Node* reshape = graph.create(kReshape, 1);
        reshape->addInput(scale);
        Tensor shape_t;
        shape_t.elem_type() = TensorProto_DataType_INT64;
        shape_t.sizes().push_back(1);
        shape_t.int64s().push_back(C);
        reshape->addInput(graph.addInitializerAndCreateValue(shape_t));
        reshape->insertBefore(conv);
        scale_1d = reshape->output();
      }
      bias_scale = scale_1d;
      // Broadcast [C] -> [C, 1, ..., 1] so it multiplies the weight per
      // channel.
      weight_scale = scale_1d;
      if (w_rank > 1) {
        std::vector<int64_t> axes(w_rank - 1);
        std::iota(axes.begin(), axes.end(), 1);
        Node* unsqueeze = graph.create(kUnsqueeze, 1);
        unsqueeze->addInput(scale_1d);
        const int opset = getOpsetVersion(graph);
        if (opset >= 13 || opset == 0) {
          Tensor axes_t;
          axes_t.elem_type() = TensorProto_DataType_INT64;
          axes_t.sizes().push_back(static_cast<int64_t>(axes.size()));
          axes_t.int64s() = axes;
          unsqueeze->addInput(graph.addInitializerAndCreateValue(axes_t));
        } else {
          unsqueeze->is_(kaxes, std::move(axes));
        }
        unsqueeze->insertBefore(conv);
        weight_scale = unsqueeze->output();
      }
    }

    Node* mul_w = graph.create(kMul, 1);
    mul_w->addInput(weight);
    mul_w->addInput(weight_scale);
    mul_w->insertBefore(conv);
    conv->replaceInput(1, mul_w->output());

    if (has_bias) {
      Value* bias = conv->inputs()[2];
      Node* mul_b = graph.create(kMul, 1);
      mul_b->addInput(bias);
      mul_b->addInput(bias_scale);
      mul_b->insertBefore(conv);
      conv->replaceInput(2, mul_b->output());
    }

    // The scale does not change the Conv output shape/type; carry the Mul's
    // inferred metadata onto the Conv output.
    if (conv_out->sizes().size() == 0 && n->output()->sizes().size() > 0) {
      conv_out->setSizes(n->output()->sizes());
    }
    if (n->output()->elemType() != TensorProto_DataType_UNDEFINED) {
      conv_out->setElemType(n->output()->elemType());
    }

    const bool replacing_success = tryReplacingAllUsesWith(n, conv);
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
