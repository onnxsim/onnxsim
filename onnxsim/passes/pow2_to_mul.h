// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// `Pow(x, 2)` becomes `Mul(x, x)` -- exact for floats, one fewer
// transcendental op, and a way past a backend whose fused-activation
// matcher recognizes `Pow(x, 2)` as part of a larger pattern it cannot
// lower at every shape (see `scripts/axera/legalize.py`'s Python
// counterpart of this rule for the concrete case that motivated it: a
// vendor compiler that fuses this exact shape into an activation kernel
// with no tiling support, so the unfused arithmetic is the only form that
// builds).
//
// Scoped to a `float` base and a constant, single-element exponent equal to
// exactly `2.0` -- `Pow`'s exponent can be a full tensor (elementwise
// power), and this rule only ever fires for the scalar-two case; anything
// else is left alone rather than approximated.
//
// This is a pure graph-shape rewrite -- not a node-count reduction -- so it
// is `PassType::Other` and never runs by default. Opt in with
// `extra_optimizers=["pow2_to_mul"]` (Python) or
// `--enable-optimization pow2_to_mul` (CLI).

#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct Pow2ToMul final : public PredicateBasedPass {
  explicit Pow2ToMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "pow2_to_mul"; }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("Pow") || node->inputs().size() != 2 ||
        node->input(0)->elemType() != TensorProto_DataType_FLOAT) {
      return false;
    }
    float exponent = 0.0f;
    return GetValueFromInput(node, 1, exponent) && exponent == 2.0f;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* mul = graph.create(kMul, 1);
    mul->addInput(node->input(0));
    mul->addInput(node->input(0));
    mul->output()->setElemType(node->output()->elemType());
    if (node->output()->has_sizes()) {
      mul->output()->setSizes(node->output()->sizes());
    }
    mul->insertBefore(node);

    node->output()->replaceAllUsesWith(mul->output());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
