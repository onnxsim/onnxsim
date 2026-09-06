// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// The mirror image of `fuse_split_gather_concat` (which collapses exactly
// this shape back into one `Gather`): rewrites `Gather(x, idx, axis=ga)`,
// when `idx`'s total element count exceeds `kMaxIndicesPerGather`, into
// `Concat(Gather(x, s_0, axis=ga), ..., Gather(x, s_{k-1}, axis=ga),
// axis=ga)` where `s_0, ..., s_{k-1} = Split(idx, axis=0)`, each chunk sized
// so its own total element count stays at or under the limit. Some
// accelerators (NPU/DSP backends with a fixed-size gather/lookup descriptor
// table, for instance) cannot execute a single `Gather` whose indices tensor
// is arbitrarily large; this trades one such `Gather` for several smaller
// ones a size-limited backend can actually run, recombined with a `Concat`
// that is itself a plain memory-layout op every backend supports.
//
// `kMaxIndicesPerGather` is a single named, conservative placeholder --
// deliberately not a runtime-configurable parameter (real hardware limits
// vary too widely to guess generically here) -- so a downstream fork
// targeting a specific accelerator's actual limit can just change this one
// constant and rebuild.
//
// This is `PassType::Other` (it trades 1 node for several, only ever a
// packaging change for a size-limited backend, never a size/op-count
// reduction) and never runs by default. Opt in with
// `extra_optimizers=["split_large_gather"]` (Python) or
// `--enable-optimization split_large_gather` (CLI).
//
// Scope (the predicate declines outside this):
//  - `Gather`, in the default (empty) domain, exactly 2 inputs, 1 output.
//  - `indices` (`Gather`'s second input) has statically-known rank and
//    sizes on every axis, and its total element count exceeds
//    `kMaxIndicesPerGather`.
//  - Splitting is always along `indices`' own axis 0: axis 0's size must
//    exceed 1, and the per-"row" element count (the product of `indices`'
//    other axes) must itself be at or under `kMaxIndicesPerGather` --
//    otherwise no split along axis 0 alone could bring any one chunk under
//    the limit, and this pass declines rather than pick a different axis.
//  - The resulting chunk count stays at or under `kMaxChunks`, guarding
//    against turning one oversized `Gather` into a pathologically large
//    number of tiny ones when `indices`' axis-0 size vastly exceeds the
//    limit relative to its other axes.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/gather_split_concat_markers.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct SplitLargeGather final : public PredicateBasedPass {
  // See this file's header comment: a documented, easily-bumped constant,
  // not a runtime parameter, since no single default fits every backend.
  static constexpr int64_t kMaxIndicesPerGather = 1LL << 16;
  static constexpr int64_t kMaxChunks = 1024;

  explicit SplitLargeGather()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "split_large_gather"; }

  // Computes the axis-0 chunk sizes `indices` (rank >= 1, all dims known)
  // should be split into so each chunk's total element count stays at or
  // under `kMaxIndicesPerGather`, given `per_row` (the product of every
  // axis but 0). Returns an empty vector when out of scope: `dim0 <= 1`
  // (nothing to split), `per_row` alone already exceeds the limit (no split
  // along axis 0 can help), or the resulting chunk count would exceed
  // `kMaxChunks`.
  static std::vector<int64_t> ComputeChunkSizes(int64_t dim0, int64_t per_row) {
    if (dim0 <= 1 || per_row > kMaxIndicesPerGather) {
      return {};
    }
    const int64_t chunk_rows =
        std::max<int64_t>(1, kMaxIndicesPerGather / per_row);
    std::vector<int64_t> chunk_sizes;
    int64_t remaining = dim0;
    while (remaining > 0) {
      if (static_cast<int64_t>(chunk_sizes.size()) >= kMaxChunks) {
        return {};
      }
      const int64_t c = std::min(chunk_rows, remaining);
      chunk_sizes.push_back(c);
      remaining -= c;
    }
    return chunk_sizes;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("Gather") || node->inputs().size() != 2 ||
        node->outputs().size() != 1) {
      return false;
    }
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    Value* indices = node->input(1);
    if (!indices->has_sizes() || indices->sizes().empty()) {
      return false;
    }
    int64_t total = 1;
    for (const Dimension& d : indices->sizes()) {
      if (!d.is_int) {
        return false;
      }
      total *= d.dim;
    }
    if (total <= kMaxIndicesPerGather) {
      return false;
    }

    const int64_t dim0 = indices->sizes()[0].dim;
    const int64_t per_row = total / dim0;
    return !ComputeChunkSizes(dim0, per_row).empty();
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* data = node->input(0);
    Value* indices = node->input(1);
    const int64_t ga =
        GetValueFromAttrWithDefault<int64_t>(node, kaxis, int64_t(0));

    int64_t total = 1;
    for (const Dimension& d : indices->sizes()) {
      total *= d.dim;
    }
    const int64_t dim0 = indices->sizes()[0].dim;
    const int64_t per_row = total / dim0;
    const std::vector<int64_t> chunk_sizes = ComputeChunkSizes(dim0, per_row);
    if (chunk_sizes.empty()) {
      return false;
    }
    const size_t k = chunk_sizes.size();

    Node* split = graph.create(Symbol("Split"), static_cast<int>(k));
    split->i_(kaxis, int64_t(0));
    // See gather_split_concat_markers.h: keeps fuse_split_gather_concat (a
    // default-on pass) from immediately fusing this deliberate,
    // size-limited-backend split back into one oversized Gather.
    split->setDocString(kSizeLimitedGatherSplitMarker);
    split->addInput(indices);
    const int opset = getOpsetVersion(*node->owningGraph());
    if (opset == 0 || opset >= 13) {
      Tensor split_sizes_t;
      split_sizes_t.elem_type() = TensorProto_DataType_INT64;
      split_sizes_t.sizes().push_back(static_cast<int64_t>(k));
      split_sizes_t.int64s() = chunk_sizes;
      split->addInput(
          graph.addInitializerAndCreateValue(std::move(split_sizes_t)));
    } else {
      split->is_(Symbol("split"), std::vector<int64_t>(chunk_sizes));
    }
    split->insertBefore(node);
    for (Value* out : split->outputs()) {
      out->setElemType(indices->elemType());
    }

    Node* concat = graph.create(kConcat, 1);
    concat->i_(kaxis, ga);
    for (Value* chunk : split->outputs()) {
      Node* g = graph.create(Symbol("Gather"), 1);
      g->addInput(data);
      g->addInput(chunk);
      g->i_(kaxis, ga);
      g->insertBefore(node);
      g->output()->setElemType(data->elemType());
      concat->addInput(g->output());
    }
    concat->insertBefore(node);
    concat->output()->setElemType(data->elemType());
    if (!node->output()->sizes().empty()) {
      concat->output()->setSizes(node->output()->sizes());
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), concat->output());
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
