// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// `Neg(x)` becomes `Mul(x, -1)` -- exact, and a way past a backend that
// implements elementwise `Mul` but not `Neg` (e.g. a hand-rolled
// reverse-mode autodiff pass that differentiates a subtraction produces
// exactly one `Neg` per occurrence, and some NPU op-support lists have
// `Mul` but not `Neg` -- see `scripts/axera/legalize.py`'s Python
// counterpart of this rule for the concrete case that motivated it).
//
// Scoped to `float32` only: the `-1` constant this emits is populated via
// `Tensor::floats()` (ONNX's `float_data` field), which is only the right
// wire representation for that one element type -- `float64`/`float16`/
// integer `Neg` would need a different tensor field per type, not handled
// here.
//
// This is a pure graph-shape rewrite -- not a node-count reduction -- so it
// is `PassType::Other` and never runs by default. Opt in with
// `extra_optimizers=["neg_to_mul"]` (Python) or
// `--enable-optimization neg_to_mul` (CLI).

#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct NegToMul final : public PredicateBasedPass {
  explicit NegToMul()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "neg_to_mul"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == Symbol("Neg") && node->inputs().size() == 1 &&
           node->input(0)->elemType() == TensorProto_DataType_FLOAT;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Tensor minus_one;
    minus_one.elem_type() = TensorProto_DataType_FLOAT;
    minus_one.floats().push_back(-1.0f);
    Value* minus_one_v = graph.addInitializerAndCreateValue(minus_one);

    Node* mul = graph.create(kMul, 1);
    mul->addInput(node->input(0));
    mul->addInput(minus_one_v);
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
