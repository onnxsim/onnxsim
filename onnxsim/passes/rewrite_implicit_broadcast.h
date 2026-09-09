// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Rewrites a multidirectional ("NumPy-style" implicit) broadcasting
// elementwise op -- one or more of whose operands has a shape that differs
// from the node's own (statically-known) output shape -- into the same op
// applied to operands that have all already been made exactly the output's
// shape via an explicit `Expand`:
//
//   Y = Add(A, B)              // A: [3,1], B: [4], Y: [3,4] -- implicit
//   ->
//   A' = Expand(A, [3,4])
//   B' = Expand(B, [3,4])
//   Y  = Add(A', B')           // A', B', Y: all [3,4] -- no broadcasting
//                              // is needed to evaluate this Add at all
//
// Some inference backends and accelerators -- particularly small NPUs, or
// runtimes whose elementwise kernels are written to require identically-
// shaped operands -- either don't support NumPy-style broadcasting at all,
// or only support a narrower form of it (e.g. unidirectional/per-channel).
// For those targets, a graph that still contains an implicitly-broadcasting
// Add/Mul/Where/... simply fails to lower or run. Making every broadcast
// explicit via `Expand` sidesteps that gap entirely: after this rewrite, an
// elementwise op's operands are always identically shaped, so a backend
// never needs to implement (or even recognize) broadcasting to run it --
// only elementwise ops between same-shaped tensors and plain `Expand`
// (a broadcast-to-shape copy), which is broadly supported on its own.
//
// This is emphatically *not* a size/op-count optimization -- it grows the
// graph (one `Expand` per operand that isn't already the output's shape) and
// can increase the amount of data actually materialized at runtime (a
// broadcast operand is realized at its full expanded size rather than
// reused in place), which is why it is `PassType::Other` and never runs by
// default. Opt in with `extra_optimizers=["rewrite_implicit_broadcast"]`
// (Python) or `--enable-optimization rewrite_implicit_broadcast` (CLI) when
// targeting a backend that needs it.
//
// Scope (the predicate declines outside this):
//  - One of the ops the ONNX spec itself documents as multidirectional-
//    broadcasting: the binary/variadic arithmetic, comparison, and logical
//    family (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `Mod`, `Max`, `Min`, `Mean`,
//    `Sum`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual`, `Equal`,
//    `And`, `Or`, `Xor`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`) plus
//    `Where` (whose `condition` operand also participates in the same
//    broadcast as its two data operands). In the default (empty) domain
//    only -- a same-named op in a vendor/plugin domain is left alone.
//  - The node's output has exactly one value, with a fully static
//    (rank *and* every dimension concrete) shape -- needed to spell out the
//    exact target shape `Expand`'s second input requires. A dynamic output
//    shape is not handled; nothing here computes a broadcast shape at
//    runtime.
//  - At least one operand's own shape does not already exactly equal the
//    output's shape -- otherwise there is nothing to make explicit and the
//    predicate declines (this is also what keeps the pass `Complete`: once
//    every operand of a node matches, re-running never fires on it again).
//    An operand with an unknown or dynamic shape is conservatively always
//    wrapped in `Expand` (it cannot be proven to already match), which is
//    always safe -- `Expand`ing an operand that already has the target
//    shape is a value-preserving identity, just extra work a later CSE/nop
//    pass, or the backend itself, can still fold away.
//  - `Expand`'s tensor-shape-input form is available from opset 8 onward
//    (its only form in this fork's supported opset range) -- a defensive
//    floor, since every op in the scope list above already implies that
//    opset or later.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct RewriteImplicitBroadcast final : public PredicateBasedPass {
  explicit RewriteImplicitBroadcast()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_implicit_broadcast";
  }

  static bool IsMultidirectionalBroadcastOp(Symbol kind) {
    static const std::vector<Symbol> kOps = {
        Symbol("Add"),          Symbol("Sub"),         Symbol("Mul"),
        Symbol("Div"),          Symbol("Pow"),         Symbol("Mod"),
        Symbol("Max"),          Symbol("Min"),         Symbol("Mean"),
        Symbol("Sum"),          Symbol("Greater"),     Symbol("GreaterOrEqual"),
        Symbol("Less"),         Symbol("LessOrEqual"), Symbol("Equal"),
        Symbol("And"),          Symbol("Or"),          Symbol("Xor"),
        Symbol("BitwiseAnd"),   Symbol("BitwiseOr"),   Symbol("BitwiseXor"),
        Symbol("Where"),
    };
    for (const Symbol& s : kOps) {
      if (kind == s) {
        return true;
      }
    }
    return false;
  }

  // True only when `v`'s shape is fully known and identical (same rank,
  // same concrete dims) to `target` -- the condition under which wrapping
  // `v` in `Expand` would be a no-op, so this pass leaves it alone.
  static bool AlreadyMatchesTargetShape(const Value* v,
                                        const std::vector<Dimension>& target) {
    if (!v->has_sizes()) {
      return false;
    }
    const std::vector<Dimension>& s = v->sizes();
    if (s.size() != target.size()) {
      return false;
    }
    for (size_t i = 0; i < s.size(); ++i) {
      if (!s[i].is_int || s[i].dim != target[i].dim) {
        return false;
      }
    }
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    if (!IsMultidirectionalBroadcastOp(node->kind())) {
      return false;
    }
    // Leave a same-named op in a non-ai.onnx domain (e.g. a vendor/plugin
    // "Add") alone.
    if (node->has_domain() && !node->domain().empty()) {
      return false;
    }
    if (node->inputs().empty() || node->outputs().size() != 1) {
      return false;
    }
    const Value* out = node->output();
    if (!out->has_sizes()) {
      return false;
    }
    for (const Dimension& d : out->sizes()) {
      if (!d.is_int) {
        return false;
      }
    }

    bool any_needs_expand = false;
    for (Value* in : node->inputs()) {
      if (!AlreadyMatchesTargetShape(in, out->sizes())) {
        any_needs_expand = true;
        break;
      }
    }
    if (!any_needs_expand) {
      return false;
    }

    // Defensive: every op in scope already implies opset >= 8 by its own
    // introduction, so this can only ever be a no-op check.
    const int opset = getOpsetVersion(*node->owningGraph());
    return opset == 0 || opset >= 8;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* out = node->output();
    const std::vector<Dimension> out_sizes = out->sizes();
    std::vector<int64_t> target_shape;
    target_shape.reserve(out_sizes.size());
    for (const Dimension& d : out_sizes) {
      target_shape.push_back(d.dim);
    }

    // Built lazily and shared across every operand of this node that needs
    // one -- Where's condition/x/y operands, say, all Expand to the exact
    // same target shape.
    Value* shape_const = nullptr;
    auto ShapeConst = [&]() -> Value* {
      if (shape_const == nullptr) {
        Tensor t;
        t.elem_type() = TensorProto_DataType_INT64;
        t.sizes().push_back(static_cast<int64_t>(target_shape.size()));
        for (int64_t d : target_shape) {
          t.int64s().push_back(d);
        }
        shape_const = graph.addInitializerAndCreateValue(std::move(t));
      }
      return shape_const;
    };

    bool changed = false;
    const size_t num_inputs = node->inputs().size();
    for (size_t i = 0; i < num_inputs; ++i) {
      Value* in = node->input(i);
      if (AlreadyMatchesTargetShape(in, out_sizes)) {
        continue;
      }
      Node* expand = graph.create(Symbol("Expand"), 1);
      expand->addInput(in);
      expand->addInput(ShapeConst());
      expand->insertBefore(node);
      expand->output()->setElemType(in->elemType());
      expand->output()->setSizes(out_sizes);
      node->replaceInput(i, expand->output());
      changed = true;
    }
    return changed;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
