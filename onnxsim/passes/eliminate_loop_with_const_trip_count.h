// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

// Unrolls a `Loop` node into its constituent iterations when the trip count
// is a compile-time constant (and, if a break condition is also present,
// that condition is provably true on every iteration). This is the `Loop`
// analogue of onnxoptimizer's eliminate_if_with_const_cond: both exist to
// turn a control-flow construct that most downstream ONNX consumers either
// can't ingest at all, or ingest only in restricted form, into plain
// feed-forward ops once the control decision is already known at
// graph-simplification time. A `Loop` over a statically-known range -- the
// ONNX shape emitted for a Python `for i in range(N): ...` with no `break`
// -- is a common case: TVM's Relax ONNX frontend, for one, has no `Loop`
// support at all.
//
// Two forms of `Loop` are unrolled:
//
//  1. Trip-count loop, with or without a break condition, where the break
//     condition (if present) can never fire: the initial `cond` input is a
//     constant `true` and the body's `cond_out` output either forwards the
//     body's own `cond` input unchanged (so by induction it stays `true`
//     forever) or is itself a constant `true`. The body then runs exactly
//     `trip_count` times.
//  2. Any loop (trip count constant or not) whose initial `cond` is a
//     constant `false`: per the Loop spec the condition is checked before
//     every iteration including the first, so the body never runs at all --
//     the loop's outputs are just its initial loop-carried inputs.
//
// Only loop-carried dependencies are handled; a `Loop` with scan-outputs is
// left alone (scan-output unrolling needs to synthesize an
// Unsqueeze+Concat per output, which is a reasonable follow-up but adds
// opset-version-dependent node construction this pass doesn't need for the
// common trip-count/accumulator case).
struct EliminateLoopWithConstTripCount final : public PredicateBasedPass {
  explicit EliminateLoopWithConstTripCount()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_loop_with_const_trip_count";
  }

  // Upper bound on how many iterations this pass will unroll, to keep a
  // constant-but-huge trip count from blowing up the graph.
  static constexpr int64_t kMaxUnrollIterations = 1024;

  // True if `cond_out` (a loop body's `cond_out` output) is guaranteed to
  // evaluate to `true` on every iteration: either it is a direct passthrough
  // of the body's own `cond` input (so it never actually changes across
  // iterations), or it is itself a compile-time constant `true`.
  static bool IsProvablyAlwaysTrue(const Value *cond_out,
                                   const Value *body_cond_in) {
    if (cond_out == body_cond_in) {
      return true;
    }
    const Tensor *t = FetchConstantTensor(cond_out);
    if (t == nullptr) {
      return false;
    }
    const auto data = ParseTensorData<bool>(t);
    return !data.empty() && data[0];
  }

  bool patternMatchPredicate(Node *node) override {
    if (node->kind() != kLoop || !node->hasAttribute(kbody)) {
      return false;
    }
    const auto &inputs = node->inputs();
    if (inputs.size() < 2) {
      return false;
    }
    const size_t n = inputs.size() - 2;
    if (node->outputs().size() != n) {
      // Has scan-outputs; not handled by this pass.
      return false;
    }

    const Value *m_value = inputs[0];
    const Value *cond_value = inputs[1];
    const bool has_m = m_value->node()->kind() != kUndefined;
    const bool has_cond = cond_value->node()->kind() != kUndefined;
    if (!has_m && !has_cond) {
      return false;  // malformed: Loop requires at least one of the two
    }

    // Case 2 from the file comment: constant-false initial condition means
    // zero iterations regardless of the trip count (which need not even be
    // constant).
    if (has_cond) {
      const Tensor *cond_tensor = FetchConstantTensor(cond_value);
      if (cond_tensor != nullptr) {
        const auto cond_data = ParseTensorData<bool>(cond_tensor);
        if (!cond_data.empty() && !cond_data[0]) {
          return true;
        }
      }
    }

    // Case 1: need a concrete, boundable trip count.
    if (!has_m) {
      return false;
    }
    const Tensor *m_tensor = FetchConstantTensor(m_value);
    if (m_tensor == nullptr) {
      return false;
    }
    const auto m_data = ParseTensorData<int64_t>(m_tensor);
    if (m_data.empty() || m_data[0] < 0 || m_data[0] > kMaxUnrollIterations) {
      return false;
    }

    const auto body = node->g(kbody);
    if (body->inputs().size() != 2 + n || body->outputs().size() != 1 + n) {
      return false;  // unexpected body signature
    }
    // copyAttributes does not rewrite lexical captures inside nested graph
    // attributes. The IR tracks subgraph-bearing nodes sparsely, so use its
    // graph walk instead of scanning every body node and attribute.
    bool has_nested_graph = false;
    body->forSelfAndEachSubGraph([&](Graph *graph) {
      has_nested_graph = has_nested_graph || graph != body.get();
    });
    if (has_nested_graph) {
      return false;
    }

    if (has_cond) {
      const Tensor *cond_tensor = FetchConstantTensor(cond_value);
      if (cond_tensor == nullptr) {
        return false;
      }
      const auto cond_data = ParseTensorData<bool>(cond_tensor);
      if (cond_data.empty() || !cond_data[0]) {
        return false;  // constant-false already handled above; anything
                       // else non-true here means "not provably true"
      }
      if (!IsProvablyAlwaysTrue(body->outputs()[0], body->inputs()[1])) {
        return false;
      }
    }
    // has_cond == false: per the Loop spec, when `cond` is omitted the body
    // runs unconditionally for exactly `trip_count` iterations and cond_out
    // is ignored, so there is nothing further to prove.
    return true;
  }

  bool runTransform(Node *loop_node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    auto &parent_graph = graph;
    const auto &loop_inputs = loop_node->inputs();
    const size_t n = loop_inputs.size() - 2;

    // Re-derive which of the two matched cases this is (see
    // patternMatchPredicate): a constant-false initial cond forces zero
    // iterations regardless of the trip count.
    int64_t trip_count = 0;
    const Value *cond_value = loop_inputs[1];
    bool zero_by_cond = false;
    if (cond_value->node()->kind() != kUndefined) {
      const Tensor *cond_tensor = FetchConstantTensor(cond_value);
      if (cond_tensor != nullptr) {
        const auto cond_data = ParseTensorData<bool>(cond_tensor);
        if (!cond_data.empty() && !cond_data[0]) {
          zero_by_cond = true;
        }
      }
    }
    if (!zero_by_cond) {
      const Tensor *m_tensor = FetchConstantTensor(loop_inputs[0]);
      trip_count = ParseTensorData<int64_t>(m_tensor)[0];
    }

    std::vector<Value *> carry(n);
    for (size_t j = 0; j < n; ++j) {
      carry[j] = loop_inputs[2 + j];
    }

    if (trip_count > 0) {
      const auto body = loop_node->g(kbody);

      std::unordered_map<std::string, Value *> unique_name_to_value_in_parent;
      for (auto *x : parent_graph.nodes()) {
        for (auto *x_output : x->outputs()) {
          unique_name_to_value_in_parent[x_output->uniqueName()] = x_output;
        }
      }

      for (int64_t iter = 0; iter < trip_count; ++iter) {
        Node *iter_const = parent_graph.create(kConstant, 1);
        {
          Tensor t;
          t.elem_type() = TensorProto_DataType_INT64;
          t.int64s().push_back(iter);
          iter_const->t_(kvalue, t);
        }
        iter_const->output()->setElemType(TensorProto_DataType_INT64);
        iter_const->output()->setSizes({});
        iter_const->insertBefore(loop_node);

        Node *cond_const = parent_graph.create(kConstant, 1);
        {
          Tensor t;
          t.elem_type() = TensorProto_DataType_BOOL;
          t.int32s().push_back(1);
          cond_const->t_(kvalue, t);
        }
        cond_const->output()->setElemType(TensorProto_DataType_BOOL);
        cond_const->output()->setSizes({});
        cond_const->insertBefore(loop_node);

        std::unordered_map<std::string, Value *> value_dict;
        value_dict[body->inputs()[0]->uniqueName()] = iter_const->output();
        value_dict[body->inputs()[1]->uniqueName()] = cond_const->output();
        for (size_t j = 0; j < n; ++j) {
          value_dict[body->inputs()[2 + j]->uniqueName()] = carry[j];
        }

        for (auto *node : body->nodes()) {
          auto *new_node =
              parent_graph.create(node->kind(), node->outputs().size());
          new_node->insertBefore(loop_node);
          new_node->copyAttributes(*node);
          for (const auto *input : node->inputs()) {
            const auto &unique_name = input->uniqueName();
            auto vit = value_dict.find(unique_name);
            if (vit != value_dict.end()) {
              new_node->addInput(vit->second);
              continue;
            }
            if (input->node()->kind() == kCaptured) {
              auto it = unique_name_to_value_in_parent.find(unique_name);
              if (it == unique_name_to_value_in_parent.end()) {
                // a value from the parent graph of parent_graph
                auto *captured_node = parent_graph.create(kCaptured, 1);
                captured_node->output()->setUniqueName(unique_name);
                new_node->addInput(captured_node->output());
              } else {
                new_node->addInput(it->second);
              }
            } else if (input->node()->kind() == kParam) {
              ONNX_ASSERT(body->is_constant_initializer(input));
              const Tensor &initializer_subgraph =
                  *body->getInitializer(unique_name);
              // Copy the tensor under a fresh name: the body is inlined once
              // per iteration, so reusing its original initializer name
              // verbatim would add several distinctly-owned initializers
              // under the same name to parent_graph, one per iteration.
              Tensor initializer_parent_graph = initializer_subgraph;
              initializer_parent_graph.setName(
                  parent_graph.getNextUniqueName());
              new_node->addInput(parent_graph.addInitializerAndCreateValue(
                  initializer_parent_graph));
            } else {
              ONNX_ASSERTM(false,
                           "loop body input not in value_dict can only be "
                           "captured, param, or a body input/carried value");
            }
          }
          for (size_t i = 0; i < node->outputs().size(); ++i) {
            const auto *output_in_subgraph = node->outputs()[i];
            auto *output_in_parent_graph = new_node->outputs()[i];
            value_dict[output_in_subgraph->uniqueName()] =
                output_in_parent_graph;
          }
        }

        // Resolve this iteration's loop-carried outputs (body outputs
        // [1 .. n], skipping cond_out at index 0 -- it is never needed here:
        // either it was already proven constant-true, or the outer `cond`
        // input was omitted and cond_out is spec-ignored).
        const auto &body_outputs = body->outputs();
        for (size_t j = 0; j < n; ++j) {
          const auto *output_in_subgraph = body_outputs[1 + j];
          const auto &unique_name = output_in_subgraph->uniqueName();
          Value *resolved = nullptr;
          auto it = value_dict.find(unique_name);
          if (it != value_dict.end()) {
            resolved = it->second;
          } else if (output_in_subgraph->node()->kind() == kCaptured) {
            auto parent_it = unique_name_to_value_in_parent.find(unique_name);
            if (parent_it == unique_name_to_value_in_parent.end()) {
              auto *captured_node = parent_graph.create(kCaptured, 1);
              captured_node->insertBefore(loop_node);
              captured_node->output()->setUniqueName(unique_name);
              resolved = captured_node->output();
            } else {
              resolved = parent_it->second;
            }
          } else if (output_in_subgraph->node()->kind() == kParam) {
            ONNX_ASSERT(body->is_constant_initializer(output_in_subgraph));
            const Tensor &initializer_subgraph =
                *body->getInitializer(unique_name);
            Tensor initializer_parent_graph = initializer_subgraph;
            initializer_parent_graph.setName(parent_graph.getNextUniqueName());
            resolved = parent_graph.addInitializerAndCreateValue(
                initializer_parent_graph);
          } else {
            ONNX_ASSERTM(false,
                         "loop body output not in value_dict can only be "
                         "captured or param");
          }
          carry[j] = resolved;
        }
      }
    }

    for (size_t j = 0; j < n; ++j) {
      loop_node->outputs()[j]->replaceAllUsesWith(carry[j]);
    }
    destroy_current = DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
