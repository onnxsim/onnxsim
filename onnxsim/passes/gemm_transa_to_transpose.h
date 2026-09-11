// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// A `Gemm` with `transA = 1` gets an explicit `Transpose` on `A` instead,
// with `transA` cleared. `A` is required to be rank 2 by the `Gemm` spec, so
// `perm=[1, 0]` is always the right transpose -- `transA = 1` means
// "transpose A before the matmul", and inserting the `Transpose` node ahead
// of the `Gemm` is the same operation, spelled as two nodes instead of one
// attribute. Some backends document `transA == 0` as a hard requirement
// (e.g. Axelera Voyager SDK's Metis compiler -- see
// `scripts/axelera/legalize.py`'s Python counterpart of this rule).
//
// This is a pure graph-shape rewrite -- not a node-count reduction -- so it
// is `PassType::Other` and never runs by default. Opt in with
// `extra_optimizers=["gemm_transA_to_transpose"]` (Python) or
// `--enable-optimization gemm_transA_to_transpose` (CLI).

#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct GemmTransAToTranspose final : public PredicateBasedPass {
  explicit GemmTransAToTranspose()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "gemm_transA_to_transpose";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kGemm || node->inputs().empty()) {
      return false;
    }
    return GetValueFromAttrWithDefault<int64_t>(node, ktransA, int64_t(0)) != 0;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* a = node->input(0);

    Node* transpose = graph.create(kTranspose, 1);
    transpose->addInput(a);
    transpose->is_(kperm, std::vector<int64_t>{1, 0});
    transpose->insertBefore(node);
    transpose->output()->setElemType(a->elemType());
    if (a->has_sizes() && a->sizes().size() == 2) {
      std::vector<Dimension> swapped{a->sizes()[1], a->sizes()[0]};
      transpose->output()->setSizes(std::move(swapped));
    }

    node->replaceInput(0, transpose->output());
    node->i_(ktransA, int64_t(0));
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
