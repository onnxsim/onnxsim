// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Fuses `Concat(Gather(x, s_0, axis=ga), ..., Gather(x, s_{k-1}, axis=ga),
// axis=ca)` -- where `s_0, ..., s_{k-1}` are, in that exact order, *all* of a
// single `Split(idx, axis=sa)` node's outputs -- into one `Gather(x, idx,
// axis=ga)`, dropping the `Split`, every one of the `k` per-chunk `Gather`s,
// and the `Concat` outright (a later dead-code-elimination pass drops
// whichever of those become unreferenced, e.g. any that also feed something
// else).
//
// This is the mirror image of `rewrite_gather_over_concat` (which pushes a
// `Gather` *into* one segment of a `Concat` it reads from): here, `Split`
// partitions `idx` into `k` pieces along its own axis `sa`; gathering `x`
// with each piece (same `x`, same `axis=ga`, for every piece) and
// concatenating the `k` results back together along axis `ca = ga + sa`
// reconstructs -- element for element, since `Split` followed by `Concat`
// along the same axis is an identity, regardless of the individual chunk
// sizes -- exactly `Gather(x, idx, axis=ga)`. Unlike
// `rewrite_gather_over_concat`, this holds for *any* `idx` values: nothing
// here depends on `idx` being a compile-time constant, or on any shape
// being statically known unless one of `ga`/`sa`/`ca` is itself given as a
// negative (relative-to-rank) attribute value, in which case only the one
// corresponding rank is needed to resolve it.
//
// Since this always trades `k + 2` nodes (`Split`, `k` `Gather`s, `Concat`)
// for exactly 1 (assuming none of the intermediates has an outside
// consumer, the common case), it is `PassType::Fuse` and runs by default,
// unlike the opt-in `Other`-classified Gather-family rewrites next to it.
//
// Scope (the predicate declines outside this):
//  - `Concat`, in the default (empty) domain, with at least 1 input and
//    exactly 1 output.
//  - Every one of `Concat`'s inputs is produced by a distinct `Gather` node
//    (default domain, exactly 2 inputs, 1 output), all sharing the exact
//    same data input `x` and the exact same (unnormalized -- compared
//    as-is, since every `Gather` shares the same `x` and so the same rank
//    frame) `axis` attribute `ga`.
//  - Those `Gather`s' indices inputs are, in Concat-input order, exactly
//    the full, in-order output list of one common `Split` node (default
//    domain) -- i.e. `Concat` consumes every `Split` output, none twice,
//    none reordered.
//  - `ca == ga + sa` once all three are normalized into their respective
//    non-negative ranges (`ga` against `x`'s rank, `sa` against `idx`'s
//    rank, `ca` against the output's rank `=
//    rank(x) - 1 + rank(idx)` -- each rank fetched only when the
//    corresponding attribute is actually negative and needs it).
//  - `split` does not carry the `split_large_gather` marker (see
//    gather_split_concat_markers.h) -- otherwise this pass would
//    immediately undo that opt-in, size-limited-backend rewrite.
#include <cstdint>
#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/gather_split_concat_markers.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct FuseSplitGatherConcat final : public PredicateBasedPass {
  explicit FuseSplitGatherConcat()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_split_gather_concat";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kConcat || node->outputs().size() != 1) {
      return false;
    }
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    ArrayRef<Value*> concat_inputs = node->inputs();
    if (concat_inputs.empty()) {
      return false;
    }

    Node* first_gather = concat_inputs[0]->node();
    if (first_gather->kind() != Symbol("Gather") ||
        first_gather->inputs().size() != 2 ||
        first_gather->outputs().size() != 1) {
      return false;
    }
    if (first_gather->has_domain() && !first_gather->domain().empty()) {
      return false;
    }
    Value* data = first_gather->input(0);
    const int64_t ga =
        GetValueFromAttrWithDefault<int64_t>(first_gather, kaxis, int64_t(0));

    Node* split = first_gather->input(1)->node();
    if (split->kind() != Symbol("Split") ||
        split->outputs().size() != concat_inputs.size()) {
      return false;
    }
    // See gather_split_concat_markers.h: a Split split_large_gather
    // deliberately introduced to keep each Gather under a size-limited
    // backend's indices-count limit must not be immediately re-fused, or
    // the two passes would perpetually undo each other.
    if (split->has_doc_string() &&
        split->docString() == kSizeLimitedGatherSplitMarker) {
      return false;
    }
    if (split->has_domain() && !split->domain().empty()) {
      return false;
    }

    for (size_t i = 0; i < concat_inputs.size(); ++i) {
      Node* g = concat_inputs[i]->node();
      if (g->kind() != Symbol("Gather") || g->inputs().size() != 2 ||
          g->outputs().size() != 1) {
        return false;
      }
      if (g->has_domain() && !g->domain().empty()) {
        return false;
      }
      if (g->input(0) != data || g->input(1) != split->outputs()[i]) {
        return false;
      }
      // Every Gather shares the same data input, so raw (unnormalized)
      // comparison is valid -- they are already in the same rank frame.
      if (GetValueFromAttrWithDefault<int64_t>(g, kaxis, int64_t(0)) != ga) {
        return false;
      }
    }

    const int64_t sa =
        GetValueFromAttrWithDefault<int64_t>(split, kaxis, int64_t(0));
    const int64_t ca = node->i(kaxis);
    Value* idx = split->input(0);

    int64_t ga_norm = ga;
    if (ga < 0) {
      if (!data->has_sizes()) {
        return false;
      }
      ga_norm = ga + static_cast<int64_t>(data->sizes().size());
    }
    int64_t sa_norm = sa;
    if (sa < 0) {
      if (!idx->has_sizes()) {
        return false;
      }
      sa_norm = sa + static_cast<int64_t>(idx->sizes().size());
    }
    int64_t ca_norm = ca;
    if (ca < 0) {
      if (!data->has_sizes() || !idx->has_sizes()) {
        return false;
      }
      const int64_t r_out = static_cast<int64_t>(data->sizes().size()) - 1 +
                            static_cast<int64_t>(idx->sizes().size());
      ca_norm = ca + r_out;
    }
    return ca_norm == ga_norm + sa_norm;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* first_gather = node->input(0)->node();
    Value* data = first_gather->input(0);
    Node* split = first_gather->input(1)->node();
    Value* idx = split->input(0);
    const int64_t ga =
        GetValueFromAttrWithDefault<int64_t>(first_gather, kaxis, int64_t(0));

    Node* new_gather = graph.create(Symbol("Gather"), 1);
    new_gather->addInput(data);
    new_gather->addInput(idx);
    new_gather->i_(kaxis, ga);
    new_gather->insertBefore(node);
    new_gather->output()->setElemType(data->elemType());
    if (!node->output()->sizes().empty()) {
      new_gather->output()->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), new_gather->output());
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
