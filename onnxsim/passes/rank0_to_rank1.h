// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Gives every rank-0 (scalar) graph output a trailing axis, so it becomes
// rank-1 with a single element -- a port of scripts/axera/legalize.py's
// rank0_to_rank1 rule, kept for the compiler/runtime that motivated it
// (Pulsar2's calibration step concatenates each output tensor across
// calibration samples, and a rank-0 tensor cannot be concatenated), but
// generic: any consumer of an ONNX graph that cannot handle a scalar output
// -- a calibration/batching pipeline is the common case -- has the same
// need, and nothing here is Pulsar2- or AX650N-specific.
//
// This is graph-output-driven, not node-pattern-driven (there is no fixed
// node kind or shape to match against -- the trigger is "this graph output
// is declared rank 0", wherever it comes from), which is why it is a
// FullGraphBasedPass rather than a PredicateBasedPass, the same split
// float16_to_float32.h documents for itself.
//
// Two things change for each affected output:
//  1. If the value is produced directly by a ReduceMean/Sum/Max/Min/Prod
//     node, that node is switched to `keepdims=1` (instead of reducing the
//     axis away) and, when its axes are not already named explicitly, given
//     an explicit axes list covering every axis of its input -- because
//     ONNX's own "no axes named" default (reduce every axis) is not always
//     what a real runtime does: a bare `ReduceMean` observed reducing only
//     the last axis of its input, rather than every axis, on the motivating
//     target. This has no effect on what the *node* computes when axes are
//     already given, and reduces every axis either way when they are not --
//     naming them only makes explicit what the default already meant.
//  2. Regardless of what produced it, the graph now needs a rank-1
//     `[1]`-shaped value under the *original* output name: the actual
//     producer's output is renamed internally, and a `Reshape(..., [1])`
//     node takes its place, its output carrying the original name and
//     redirecting every prior consumer of the value (the graph-output slot,
//     and any other internal use) to the reshaped result. `keepdims=1`
//     already leaves a same-numel tensor (a size-1 dim per reduced axis
//     instead of none), so this reshape is exact regardless of the
//     producer's actual rank.

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct Rank0ToRank1 final : public FullGraphBasedPass {
  explicit Rank0ToRank1()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::None) {}
  std::string getPassName() const override { return "rank0_to_rank1"; }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  // The Reduce* kinds scripts/axera/legalize.py's rank0_to_rank1 special-
  // cases -- deliberately not every reduction op ONNX defines (e.g.
  // ReduceL1/L2/LogSum(Exp)/SumSquare are left out there too), to stay a
  // faithful, behavior-preserving port rather than a broadened rewrite.
  static bool IsHandledReduceKind(NodeKind k) {
    static const std::unordered_set<NodeKind> kKinds{
        Symbol("ReduceMean"), Symbol("ReduceSum"), Symbol("ReduceMax"),
        Symbol("ReduceMin"), Symbol("ReduceProd")};
    return kKinds.count(k) != 0;
  }

  static Value* ConstI64Vec1(Graph& graph, int64_t v) {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes().push_back(1);
    t.int64s().push_back(v);
    return graph.addInitializerAndCreateValue(std::move(t));
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    // Snapshot: redirecting a graph output's producer below mutates
    // return_node()'s own inputs (graph.outputs() itself), which would
    // invalidate a live iteration over it.
    std::vector<Value*> orig_outputs(graph.outputs().begin(),
                                     graph.outputs().end());

    for (Value* v : orig_outputs) {
      if (!v->has_sizes() || !v->sizes().empty()) {
        continue;  // not a declared-rank-0 output
      }

      Node* producer = v->node();
      if (IsHandledReduceKind(producer->kind()) &&
          producer->inputs().size() >= 1) {
        producer->i_(kkeepdims, int64_t{1});

        Value* x = producer->input(0);
        const bool named =
            producer->hasAttribute(kaxes) || producer->inputs().size() > 1;
        if (!named && x->has_sizes() && !x->sizes().empty()) {
          std::vector<int64_t> axes(x->sizes().size());
          for (size_t i = 0; i < axes.size(); ++i) {
            axes[i] = static_cast<int64_t>(i);
          }
          // `axes` moved from a Reduce* attribute to its second input at
          // opset 18 (scripts/axera/legalize.py's own `_set_axes` --
          // uniformly for every Reduce* kind, not per-op like
          // fuse_consecutive_reduce.h's more precise ReduceSum-at-13
          // handling, since real graphs in this project are always one
          // opset end to end). getOpsetVersion returns 0 when no opset
          // import is found, which onnxsim's own opset-lookup helpers treat
          // as "assume unversioned/legacy" -- the attribute form.
          const int opset = PredicateBasedPass::getOpsetVersion(graph);
          if (opset >= 18) {
            Tensor axes_t;
            axes_t.elem_type() = TensorProto_DataType_INT64;
            axes_t.sizes().push_back(static_cast<int64_t>(axes.size()));
            axes_t.int64s().assign(axes.begin(), axes.end());
            producer->addInput(
                graph.addInitializerAndCreateValue(std::move(axes_t)));
          } else {
            producer->is_(kaxes, std::move(axes));
          }
        }
      }

      // Snapshot uses() before creating the Reshape node, so the reshape's
      // own new use of `v` is never among the uses redirected to it.
      auto use_list = v->uses();

      const std::string original_name = v->uniqueName();
      v->setUniqueName(graph.getNextUniqueName());

      Node* reshape = graph.create(kReshape, 1);
      reshape->addInput(v);
      reshape->addInput(ConstI64Vec1(graph, 1));
      reshape = graph.appendNode(reshape);
      reshape->output()->setUniqueName(original_name);
      reshape->output()->setElemType(v->elemType());
      reshape->output()->setSizes({Dimension(int64_t{1})});

      for (auto& use : use_list) {
        use.user->replaceInput(use.offset, reshape->output());
      }
    }

    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
