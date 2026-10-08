// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Softmax/LogSoftmax normalize over a group of elements; when that group is a
// single element the result does not depend on the input value:
//
//   Softmax(X)    = exp(x) / exp(x) = 1
//   LogSoftmax(X) = x - log(exp(x)) = 0
//
// so the node is replaced by a constant of X's shape,
// `ConstantOfShape(Shape(X))`, which constant folding collapses to an
// initializer when X's shape is static. (Strictly, a non-finite input would
// give NaN in a naive implementation; the constant is the mathematical value.)
//
// What counts as "a single element" depends on the opset:
//   - opset >= 13: the normalization runs over `axis` alone, so dim[axis]==1.
//   - opset <  13: the input is first coerced to 2-D at `axis` and normalized
//     over the whole trailing block, so dims[axis:] must all be 1.
//
// Only FLOAT/DOUBLE/FLOAT16/BFLOAT16 are handled (the types Softmax defines).

#include <string>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct EliminateNopSoftmax final : public PredicateBasedPass {
  explicit EliminateNopSoftmax()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "eliminate_nop_softmax"; }

  // Little-endian raw bytes of a single 1 (or 0) of `elem_type`; empty when
  // the type isn't one this pass supports.
  static std::string ConstantBytes(int32_t elem_type, bool one) {
    switch (elem_type) {
      case TensorProto_DataType_FLOAT:
        return one ? std::string("\x00\x00\x80\x3f", 4) : std::string(4, '\0');
      case TensorProto_DataType_DOUBLE:
        return one ? std::string("\x00\x00\x00\x00\x00\x00\xf0\x3f", 8)
                   : std::string(8, '\0');
      case TensorProto_DataType_FLOAT16:
        return one ? std::string("\x00\x3c", 2) : std::string(2, '\0');
      case TensorProto_DataType_BFLOAT16:
        return one ? std::string("\x80\x3f", 2) : std::string(2, '\0');
      default:
        return std::string();
    }
  }

  static bool IsUnitDim(const Dimension& d) { return d.is_int && d.dim == 1; }

  static bool MatchSingleElementGroup(Node* node, const Graph& graph) {
    Value* x = node->input(0);
    if (!x->has_sizes() || x->sizes().empty() ||
        ConstantBytes(x->elemType(), true).empty()) {
      return false;
    }
    const auto& sizes = x->sizes();
    const int64_t rank = static_cast<int64_t>(sizes.size());
    const int opset = getOpsetVersion(graph);
    const bool coerce_2d = opset != 0 && opset < 13;

    int64_t axis = coerce_2d ? 1 : -1;
    GetValueFromAttr(node, kaxis, axis);
    if (axis < -rank || axis >= rank) {
      return false;
    }
    axis = AddYIfNegative(axis, rank);

    for (int64_t i = axis; i < (coerce_2d ? rank : axis + 1); ++i) {
      if (!IsUnitDim(sizes[static_cast<size_t>(i)])) {
        return false;
      }
    }
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    if ((node->kind() != kSoftmax && node->kind() != kLogSoftmax) ||
        node->inputs().size() != 1) {
      return false;
    }
    return MatchSingleElementGroup(node, *node->owningGraph());
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (!MatchSingleElementGroup(node, graph)) {
      return false;
    }
    Value* x = node->input(0);
    const int32_t elem_type = x->elemType();

    Node* shape = graph.create(Symbol("Shape"), 1);
    shape->addInput(x);
    shape->output()->setElemType(TensorProto_DataType_INT64);
    shape->insertBefore(node);

    Tensor value;
    value.elem_type() = elem_type;
    value.sizes().push_back(1);
    value.set_raw_data(ConstantBytes(elem_type, node->kind() == kSoftmax));

    Node* fill = graph.create(Symbol("ConstantOfShape"), 1);
    fill->addInput(shape->output());
    fill->t_(Symbol("value"), std::move(value));
    fill->output()->copyMetadata(node->output());
    fill->insertBefore(node);

    if (!tryReplacingAllUsesWith(node, fill)) {
      fill->destroy();
      shape->destroy();
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
