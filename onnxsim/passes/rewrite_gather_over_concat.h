// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Rewrites `Gather(Concat(x_0, ..., x_{n-1}, axis=ca), indices, axis=ga)`
// into `Gather(x_i, indices - offset_i, axis=ga)` whenever `ga == ca`
// (`Gather` selects along the very axis `Concat` joined its inputs on) and
// `indices` is a compile-time constant whose every (canonicalized) value
// falls inside the same single Concat input's segment `x_i`. This drops the
// `Gather`'s dependency on the `Concat` -- and on every one of its *other*
// inputs -- entirely: producing the `Gather`'s output no longer needs the
// full concatenated tensor materialized, only the one input segment the
// constant indices actually select from. If the `Concat` has no other
// consumers left afterward, a later dead-code-elimination pass can then
// drop the whole `Concat` node and any of its now-unreferenced other
// inputs' subgraphs -- this is the common case this pass targets: a
// `Concat` of independently-computed (or partly constant, partly dynamic)
// branches immediately followed by a `Gather` that, statically, only ever
// needed one of them.
//
// A single `Gather` node can only ever be rewired to depend on one upstream
// value, so when the constant indices span *more than one* Concat input
// segment this pass declines outright rather than attempt anything more
// elaborate (e.g. splitting into several smaller per-segment `Gather`s
// recombined with a new `Concat`) -- that would trade one dependency for
// several, and is not obviously a win in general.
//
// This is `PassType::Other` (a graph-shape rewrite, not itself a
// size/op-count reduction -- the payoff shows up only once a later
// dead-code-elimination pass acts on the now-possibly-unused `Concat`/its
// other inputs) and never runs by default. Opt in with
// `extra_optimizers=["rewrite_gather_over_concat"]` (Python) or
// `--enable-optimization rewrite_gather_over_concat` (CLI).
//
// Scope (the predicate declines outside this):
//  - `Gather` in the default (empty) domain, exactly 2 inputs, 1 output;
//    its data input produced by a `Concat` with at least 2 inputs.
//  - `Gather`'s data input (== `Concat`'s output) has statically-known rank
//    `r` -- needed to normalize a negative `axis` on either node -- and
//    `ga == ca` once both are normalized into `[0, r)`.
//  - Every one of `Concat`'s `n` inputs has rank `r` and a statically-known
//    size along axis `ca`.
//  - `indices` (`Gather`'s second input) is a non-empty compile-time
//    constant tensor (a `Constant` node or an initializer) of `INT32` or
//    `INT64` elements, and every value in it -- after wrapping any negative
//    value into `[0, total)`, `total` the sum of the `n` per-input axis-`ca`
//    sizes -- lies in range and falls inside the *same* single input's
//    segment.

#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct RewriteGatherOverConcat final : public PredicateBasedPass {
  explicit RewriteGatherOverConcat()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_gather_over_concat";
  }

  // Resolves which single Concat input segment (if any) `indices` -- a
  // known-constant tensor, already read out as plain integers -- selects
  // from entirely, given `axis_sizes` (the static size of each Concat input
  // along the shared Gather/Concat axis, in input order). Returns false
  // (declining the rewrite) when `indices` is empty, any value is out of
  // `[-total, total)` range, or the resolved values span more than one
  // segment. On success, `segment` is the winning input's index and
  // `local_indices` holds `indices`, each value shifted down by that
  // segment's cumulative offset so it is valid as a `Gather` index directly
  // into that one input.
  static bool ResolveSingleSegment(const std::vector<int64_t>& indices,
                                   const std::vector<int64_t>& axis_sizes,
                                   size_t& segment,
                                   std::vector<int64_t>& local_indices) {
    if (indices.empty()) {
      return false;
    }
    std::vector<int64_t> offsets(axis_sizes.size());
    int64_t total = 0;
    for (size_t i = 0; i < axis_sizes.size(); ++i) {
      offsets[i] = total;
      total += axis_sizes[i];
    }

    bool have_segment = false;
    size_t resolved_segment = 0;
    local_indices.clear();
    local_indices.reserve(indices.size());
    for (int64_t idx : indices) {
      const int64_t norm = AddYIfNegative(idx, total);
      if (norm < 0 || norm >= total) {
        return false;
      }
      // Linear scan for the segment `norm` falls into -- n (Concat's input
      // count) is small in practice, and this only runs at
      // simplification time, never per-inference.
      size_t seg = 0;
      while (seg + 1 < axis_sizes.size() &&
             norm >= offsets[seg] + axis_sizes[seg]) {
        ++seg;
      }
      if (!have_segment) {
        resolved_segment = seg;
        have_segment = true;
      } else if (seg != resolved_segment) {
        return false;
      }
      local_indices.push_back(norm - offsets[seg]);
    }
    segment = resolved_segment;
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    if (!CheckKind(node, "Gather", 0, "Concat")) {
      return false;
    }
    if (node->inputs().size() != 2 || node->outputs().size() != 1) {
      return false;
    }
    // Leave a same-named op in a non-ai.onnx domain (e.g. a vendor/plugin
    // "Gather") alone.
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    Node* concat = node->input(0)->node();
    if (concat->inputs().size() < 2) {
      return false;
    }

    Value* data = node->input(0);
    if (!data->has_sizes()) {
      return false;
    }
    const int64_t r = static_cast<int64_t>(data->sizes().size());

    int64_t ga = GetValueFromAttrWithDefault<int64_t>(node, kaxis, int64_t(0));
    int64_t ca = concat->i(kaxis);
    ga = AddYIfNegative(ga, r);
    ca = AddYIfNegative(ca, r);
    if (ga < 0 || ga >= r || ca < 0 || ca >= r || ga != ca) {
      return false;
    }

    Value* indices = node->input(1);
    const Tensor* idx_tensor = FetchConstantTensor(indices);
    if (idx_tensor == nullptr ||
        (idx_tensor->elem_type() != TensorProto_DataType_INT32 &&
         idx_tensor->elem_type() != TensorProto_DataType_INT64)) {
      return false;
    }

    std::vector<int64_t> axis_sizes;
    axis_sizes.reserve(concat->inputs().size());
    for (Value* in : concat->inputs()) {
      if (!in->has_sizes() || static_cast<int64_t>(in->sizes().size()) != r ||
          !in->sizes()[ca].is_int) {
        return false;
      }
      axis_sizes.push_back(in->sizes()[ca].dim);
    }

    const std::vector<int64_t> idx_values = GetIntsFromValue(indices);
    size_t segment;
    std::vector<int64_t> local_indices;
    return ResolveSingleSegment(idx_values, axis_sizes, segment, local_indices);
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* concat = node->input(0)->node();
    Value* data = node->input(0);
    const int64_t r = static_cast<int64_t>(data->sizes().size());
    const int64_t ca = AddYIfNegative(concat->i(kaxis), r);

    std::vector<int64_t> axis_sizes;
    axis_sizes.reserve(concat->inputs().size());
    for (Value* in : concat->inputs()) {
      axis_sizes.push_back(in->sizes()[ca].dim);
    }

    Value* indices = node->input(1);
    const Tensor* idx_tensor = FetchConstantTensor(indices);
    const std::vector<int64_t> idx_values = GetIntsFromValue(indices);

    size_t segment;
    std::vector<int64_t> local_indices;
    if (!ResolveSingleSegment(idx_values, axis_sizes, segment, local_indices)) {
      return false;
    }

    Tensor new_idx_tensor;
    new_idx_tensor.elem_type() = idx_tensor->elem_type();
    new_idx_tensor.sizes() = idx_tensor->sizes();
    if (idx_tensor->elem_type() == TensorProto_DataType_INT64) {
      new_idx_tensor.int64s() = local_indices;
    } else {
      std::vector<int32_t> local_indices_i32(local_indices.begin(),
                                             local_indices.end());
      new_idx_tensor.int32s() = std::move(local_indices_i32);
    }
    Value* new_indices =
        graph.addInitializerAndCreateValue(std::move(new_idx_tensor));

    Value* selected_input = concat->input(segment);
    node->replaceInput(0, selected_input);
    node->replaceInput(1, new_indices);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
