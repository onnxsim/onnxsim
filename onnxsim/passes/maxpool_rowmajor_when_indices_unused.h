// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// A `MaxPool` with `storage_order = 1` (column-major) gets it cleared to 0
// (row-major) whenever the optional `Indices` output isn't actually
// consumed -- the attribute only orders that output (whether it's read out
// row-major or column-major), so with no `Indices` consumer the two
// settings compute the identical `Y`.
//
// A generic, target-agnostic legalization: some inference backends only
// implement (or only accelerate) `MaxPool` with `storage_order == 0`. This
// pass carries no knowledge of any particular target; a target-specific
// legalizer decides whether it needs this rewrite and opts into it (see
// e.g. `scripts/axelera/legalize.py`, whose own docstring records a real
// compiler that documents `storage_order == 0` as a hard requirement).
//
// This is a pure graph-shape rewrite -- so it is `PassType::Other` and
// never runs by default. Opt in with
// `extra_optimizers=["maxpool_rowmajor_when_indices_unused"]` (Python) or
// `--enable-optimization maxpool_rowmajor_when_indices_unused` (CLI).

#include <cstdint>
#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct MaxPoolRowMajorWhenIndicesUnused final : public PredicateBasedPass {
  explicit MaxPoolRowMajorWhenIndicesUnused()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "maxpool_rowmajor_when_indices_unused";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("MaxPool")) {
      return false;
    }
    if (GetValueFromAttrWithDefault<int64_t>(node, kstorage_order,
                                             int64_t(0)) == 0) {
      return false;
    }
    // Indices is produced (a second output was declared) and consumed;
    // its ordering is observable, so leave storage_order alone.
    return !(node->outputs().size() > 1 && !node->outputs()[1]->uses().empty());
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    node->i_(kstorage_order, int64_t(0));
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
