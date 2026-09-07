// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   W = <initializer / Constant>   # dims [2,3,4]
//   Y = Reshape(W, {6,4})
// After:
//   W' = <initializer / Constant>  # dims [6,4], byte-for-byte the same data
//   (consumers read W' directly; the Reshape node is gone)
//
// Reshape, Squeeze, Unsqueeze and Flatten only rearrange a tensor's *shape* --
// the element sequence they produce is exactly the element sequence they were
// given (see fuse_reshape_family.h, which fuses two adjacent members of that
// family into one). Applying one to a constant is therefore a pure metadata
// change: the result is the same tensor data under a new dims list, which this
// pass computes from the constant's own (statically known, exact) dims rather
// than from shape inference.
//
// onnxsim's constant folding (constant_folding.cpp's GetConstantNodes) already
// removes these nodes in the default pipeline -- an initializer is a constant
// there, so `Reshape(W, const_shape)` is foldable and its result is
// materialized as a new initializer. This pass is not a replacement for that;
// it covers what folding cannot or should not do:
//   - it runs with constant folding switched off (--skip-constant-folding),
//     where the reshape would otherwise survive on a graph that is entirely
//     static;
//   - it needs no model executor and never round-trips the weight through one,
//     so a multi-GB weight costs a dims rewrite instead of a full execute +
//     materialize;
//   - it preserves an external-data (data_location EXTERNAL) initializer as an
//     external reference -- reshaping does not change the byte range it points
//     at -- where folding would have to load and inline the data.
//
// Only a constant this node is the *sole* consumer of is rewritten. Otherwise
// the original tensor stays alive for its other consumers and the rewritten
// copy is added alongside it, which trades one node for a second copy of the
// weight -- a bad deal exactly on the big-weight models this pass is meant to
// help, and one the caller already opted out of by disabling constant folding.
// With the sole-use restriction the old initializer is left unused and
// eliminate_unused_initializer (a Nop pass, so always in the default set)
// drops it, leaving the graph strictly smaller.
//
// The rewritten tensor is always a fresh Tensor rather than an in-place edit
// of the existing one: the tensor-content digest cache
// (onnxoptimizer/passes/tensor_content_hash.h) memoizes per Tensor::tensor_id()
// across pass runs on a resident Graph -- as onnxsim's own OptAndShape fixed
// point does, passing clear_tensor_digest_cache=false -- and stays valid only
// because no pass mutates a retained tensor in place. A copy/assignment mints a
// fresh tensor_id(), so a rewritten tensor simply misses the cache.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct EliminateReshapeFamilyOnConstant final : public PredicateBasedPass {
  explicit EliminateReshapeFamilyOnConstant()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_reshape_family_on_constant";
  }

  static bool IsFamily(uint32_t kind) {
    return kind == kReshape || kind == kSqueeze || kind == kUnsqueeze ||
           kind == kFlatten;
  }

  // Product of `dims`, or -1 if any entry is negative (which no real tensor's
  // dims list has -- a `-1` only ever appears in a Reshape *target* shape).
  static int64_t ElementCount(const std::vector<int64_t>& dims) {
    int64_t total = 1;
    for (const int64_t d : dims) {
      if (d < 0) {
        return -1;
      }
      total *= d;
    }
    return total;
  }

  // Reshape(data, shape): the target shape must be a constant INT64 tensor.
  static bool ReshapeSizes(const Node* n, const std::vector<int64_t>& in_dims,
                           std::vector<int64_t>& out) {
    std::vector<int64_t> shape;
    if (!GetValueFromInput(n, 1, shape)) {
      return false;
    }
    const bool allowzero =
        n->hasAttribute(Symbol("allowzero")) && n->i(Symbol("allowzero")) == 1;
    int64_t infer_at = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      const int64_t d = shape[i];
      if (d == 0 && !allowzero) {
        // Under the default allowzero=0 a `0` means "copy dimension i from the
        // data input" -- and here the data input is the constant itself, so its
        // own dims answer it exactly.
        if (i >= in_dims.size()) {
          return false;
        }
        out.push_back(in_dims[i]);
      } else if (d == -1) {
        // ONNX allows at most one inferred dimension.
        if (infer_at >= 0) {
          return false;
        }
        infer_at = static_cast<int64_t>(i);
        out.push_back(1);  // placeholder, resolved below
      } else if (d < 0) {
        return false;
      } else {
        out.push_back(d);
      }
    }
    const int64_t total = ElementCount(in_dims);
    if (total < 0) {
      return false;
    }
    if (infer_at >= 0) {
      const int64_t rest = ElementCount(out);
      // rest == 0 leaves the inferred dimension undefined (any value satisfies
      // it), which is not something to guess at.
      if (rest <= 0 || total % rest != 0) {
        return false;
      }
      out[infer_at] = total / rest;
    }
    // A target shape whose element count disagrees with the data is an invalid
    // Reshape; leave such a graph exactly as it is rather than "fixing" it.
    return ElementCount(out) == total;
  }

  // Squeeze(data, axes?) -- axes is an attribute before opset 13 and an
  // (optional) input from 13 on; GetValueFromAttrOrInput reads either.
  static bool SqueezeSizes(const Node* n, const std::vector<int64_t>& in_dims,
                           std::vector<int64_t>& out) {
    const int64_t rank = static_cast<int64_t>(in_dims.size());
    if (!n->hasAttribute(kaxes) && n->inputs().size() < 2) {
      // No axes given at all: every size-1 dimension is squeezed.
      std::copy_if(in_dims.begin(), in_dims.end(), std::back_inserter(out),
                   [](int64_t d) { return d != 1; });
      return true;
    }
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(n, kaxes, 1, axes) || axes.empty()) {
      // Present but not statically known, or an explicitly empty axes list --
      // which the spec reads as "squeeze nothing" but implementations have
      // been known to treat as the omitted-axes case ("squeeze every 1").
      // Not a difference to bake into a weight.
      return false;
    }
    std::set<int64_t> squeezed;
    for (int64_t axis : axes) {
      axis = AddYIfNegative(axis, rank);
      // Squeezing a dimension that is not 1 is invalid; leave it alone.
      if (axis < 0 || axis >= rank || in_dims[axis] != 1) {
        return false;
      }
      squeezed.insert(axis);
    }
    for (int64_t i = 0; i < rank; ++i) {
      if (squeezed.count(i) == 0) {
        out.push_back(in_dims[i]);
      }
    }
    return true;
  }

  // Unsqueeze(data, axes) -- axes is required, as an attribute before opset 13
  // and an input from 13 on. Its entries index into the *output*'s rank.
  static bool UnsqueezeSizes(const Node* n, const std::vector<int64_t>& in_dims,
                             std::vector<int64_t>& out) {
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(n, kaxes, 1, axes) || axes.empty()) {
      return false;
    }
    const int64_t out_rank = static_cast<int64_t>(in_dims.size() + axes.size());
    std::set<int64_t> inserted;
    for (int64_t axis : axes) {
      axis = AddYIfNegative(axis, out_rank);
      if (axis < 0 || axis >= out_rank) {
        return false;
      }
      inserted.insert(axis);
    }
    // Duplicate axes make the op invalid (and would leave a dim unaccounted
    // for below).
    if (inserted.size() != axes.size()) {
      return false;
    }
    size_t next = 0;
    for (int64_t i = 0; i < out_rank; ++i) {
      out.push_back(inserted.count(i) != 0 ? 1 : in_dims[next++]);
    }
    return next == in_dims.size();
  }

  // Flatten(data) -> [prod(dims[:axis]), prod(dims[axis:])], axis defaulting
  // to 1 and allowed to be negative (or equal to the rank).
  static bool FlattenSizes(const Node* n, const std::vector<int64_t>& in_dims,
                           std::vector<int64_t>& out) {
    const int64_t rank = static_cast<int64_t>(in_dims.size());
    int64_t axis =
        GetValueFromAttrWithDefault(n, kaxis, static_cast<int64_t>(1));
    axis = AddYIfNegative(axis, rank);
    if (axis < 0 || axis > rank) {
      return false;
    }
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < rank; ++i) {
      (i < axis ? outer : inner) *= in_dims[i];
    }
    out = {outer, inner};
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    return IsFamily(node->kind()) && !node->inputs().empty() &&
           IsConstantTensor(node->input(0));
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Value* data = n->input(0);
    // See this file's comment on the sole-use restriction: rewriting a shared
    // constant would leave a second copy of it in the graph.
    if (data->uses().size() != 1) {
      return false;
    }
    const Tensor* tensor = FetchConstantTensor(data);
    if (tensor == nullptr) {
      return false;
    }
    // Replacing the output with an initializer / Constant value renames it,
    // which is not allowed for a value the graph itself hands back.
    const auto& graph_outputs = graph.outputs();
    if (std::find(graph_outputs.begin(), graph_outputs.end(), n->output()) !=
        graph_outputs.end()) {
      return false;
    }

    const std::vector<int64_t>& in_dims = tensor->sizes();
    std::vector<int64_t> dims;
    bool resolved = false;
    if (n->kind() == kReshape) {
      resolved = ReshapeSizes(n, in_dims, dims);
    } else if (n->kind() == kSqueeze) {
      resolved = SqueezeSizes(n, in_dims, dims);
    } else if (n->kind() == kUnsqueeze) {
      resolved = UnsqueezeSizes(n, in_dims, dims);
    } else if (n->kind() == kFlatten) {
      resolved = FlattenSizes(n, in_dims, dims);
    }
    if (!resolved) {
      return false;
    }

    std::vector<Dimension> value_sizes;
    value_sizes.reserve(dims.size());
    for (const int64_t d : dims) {
      value_sizes.emplace_back(d);
    }
    // The data is carried over untouched -- only the dims list changes -- so
    // this holds for raw_data, the typed data fields and an EXTERNAL
    // data_location's byte range alike.
    Tensor reshaped = *tensor;
    reshaped.sizes() = dims;

    Value* replacement = nullptr;
    if (data->node()->kind() == kConstant) {
      // Keep a Constant node a Constant node: onnxsim deliberately does not
      // turn a computed value into graph weight data (see
      // extract_constant_to_initializer's entry in SimplifyImpl's
      // always_disabled_passes, and GetConstantNodes' impure_outputs).
      Node* folded = graph.create(kConstant, 1);
      folded->t_(kvalue, std::move(reshaped));
      folded->output()->setElemType(tensor->elem_type());
      folded->output()->setSizes(value_sizes);
      folded->insertBefore(n);
      replacement = folded->output();
    } else {
      reshaped.setName(graph.getNextUniqueName());
      replacement = graph.addInitializerAndCreateValue(std::move(reshaped));
    }
    // Safe without tryReplacingAllUsesWith's check: the graph-output guard
    // above already established that n's output is not one.
    n->output()->replaceAllUsesWith(replacement);
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
