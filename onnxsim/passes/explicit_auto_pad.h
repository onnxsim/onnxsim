// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// `Conv`/`AveragePool`/`MaxPool` with `auto_pad` in `SAME_UPPER`,
// `SAME_LOWER` or `VALID` get `auto_pad = "NOTSET"` and the equivalent
// explicit `pads`, computed from the ONNX operator spec's own `auto_pad`
// formula: `VALID` is zero padding by definition; `SAME_UPPER`/`SAME_LOWER`
// split the padding the output needs to reach `ceil(input / stride)`, the
// extra pixel (if any) going to the end (`SAME_UPPER`) or the start
// (`SAME_LOWER`). The two forms are defined to agree, so this changes
// nothing about what the op computes, only how the padding amount is
// spelled.
//
// Needs the input's spatial shape to be statically known -- the formula is
// defined in terms of it -- so a node whose input shape isn't resolved (a
// dynamic axis) is left alone rather than guessed at. `Conv`'s
// `kernel_shape` attribute is optional (inferrable from the weight tensor
// `W`, which exporters routinely omit it in favor of), so this reads the
// kernel from `W`'s shape when the attribute is absent -- the same place a
// compiler that itself requires `auto_pad == "NOTSET"` would have to read
// it from.
//
// A generic, target-agnostic legalization: many inference backends accept
// only explicit padding and refuse (or silently mishandle) `SAME_*`/`VALID`
// `auto_pad`. This pass carries no knowledge of any particular target; a
// target-specific legalizer decides whether it needs this rewrite and opts
// into it (see e.g. `scripts/axelera/legalize.py`, whose own docstring
// records a real compiler that documents `auto_pad == "NOTSET"` as a hard
// requirement).
//
// This is a pure graph-shape rewrite -- not a node-count reduction -- so it
// is `PassType::Other` and never runs by default. Opt in with
// `extra_optimizers=["explicit_auto_pad"]` (Python) or
// `--enable-optimization explicit_auto_pad` (CLI).

#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace explicit_auto_pad_detail {

// The statically-known spatial (i.e. all but the leading N, C) dims of a
// rank `nd + 2` tensor, or false if the rank doesn't match or any spatial
// dim is symbolic/unknown.
inline bool StaticSpatialShape(const Value* v, size_t nd,
                               std::vector<int64_t>& out) {
  if (!v->has_sizes() || v->sizes().size() != nd + 2) {
    return false;
  }
  out.clear();
  out.reserve(nd);
  for (size_t i = 2; i < nd + 2; ++i) {
    const Dimension& d = v->sizes()[i];
    if (!d.is_int) {
      return false;
    }
    out.push_back(d.dim);
  }
  return true;
}

// ONNX's own `auto_pad` formula (see the `Conv` operator spec), as explicit
// `(begin, end)` pads per spatial axis.
inline void SamePads(const std::string& auto_pad,
                     const std::vector<int64_t>& spatial_in,
                     const std::vector<int64_t>& kernel,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& dilations,
                     std::vector<int64_t>& begin, std::vector<int64_t>& end) {
  const size_t nd = kernel.size();
  begin.assign(nd, 0);
  end.assign(nd, 0);
  if (auto_pad == "VALID") {
    return;
  }
  for (size_t i = 0; i < nd; ++i) {
    const int64_t out_size =
        (spatial_in[i] + strides[i] - 1) / strides[i];  // ceil division
    int64_t needed = (out_size - 1) * strides[i] +
                     ((kernel[i] - 1) * dilations[i] + 1) - spatial_in[i];
    if (needed < 0) {
      needed = 0;
    }
    if (auto_pad == "SAME_UPPER") {
      begin[i] = needed / 2;
      end[i] = needed - begin[i];
    } else {  // SAME_LOWER
      end[i] = needed / 2;
      begin[i] = needed - end[i];
    }
  }
}

}  // namespace explicit_auto_pad_detail

struct ExplicitAutoPad final : public PredicateBasedPass {
  explicit ExplicitAutoPad()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "explicit_auto_pad"; }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kConv && node->kind() != Symbol("AveragePool") &&
        node->kind() != Symbol("MaxPool")) {
      return false;
    }
    const std::string auto_pad = GetValueFromAttrWithDefault<std::string>(
        node, Symbol("auto_pad"), std::string("NOTSET"));
    return auto_pad != "NOTSET";
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    using explicit_auto_pad_detail::SamePads;
    using explicit_auto_pad_detail::StaticSpatialShape;

    const std::string auto_pad = GetValueFromAttrWithDefault<std::string>(
        node, Symbol("auto_pad"), std::string("NOTSET"));

    std::vector<int64_t> kernel;
    if (node->hasAttribute(kkernel_shape)) {
      kernel = node->is(kkernel_shape);
    } else if (node->kind() == kConv && node->inputs().size() > 1) {
      const Value* w = node->input(1);
      if (!w->has_sizes()) {
        return false;
      }
      for (size_t i = 2; i < w->sizes().size(); ++i) {
        const Dimension& d = w->sizes()[i];
        if (!d.is_int) {
          return false;
        }
        kernel.push_back(d.dim);
      }
    }
    if (kernel.empty()) {
      return false;
    }

    std::vector<int64_t> spatial_in;
    if (!StaticSpatialShape(node->input(0), kernel.size(), spatial_in)) {
      return false;
    }

    std::vector<int64_t> strides(kernel.size(), 1);
    if (node->hasAttribute(kstrides)) {
      strides = node->is(kstrides);
    }
    std::vector<int64_t> dilations(kernel.size(), 1);
    if (node->hasAttribute(kdilations)) {
      dilations = node->is(kdilations);
    }
    if (strides.size() != kernel.size() || dilations.size() != kernel.size()) {
      return false;
    }

    std::vector<int64_t> begin, end;
    SamePads(auto_pad, spatial_in, kernel, strides, dilations, begin, end);

    node->s_(Symbol("auto_pad"), std::string("NOTSET"));
    std::vector<int64_t> pads(begin);
    pads.insert(pads.end(), end.begin(), end.end());
    node->is_(kpads, std::move(pads));
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
