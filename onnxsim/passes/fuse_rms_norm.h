// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before (the textbook RMSNorm forward -- e.g. LLaMA/Mistral/Qwen-family
// exports; see test_mnn_llm_export.py's `_RMSNorm`):
//   variance = X.pow(2).mean(-1, keepdim=True)  # ReduceMean(Pow(X,2)|Mul(X,X))
//   Y = weight * (X * rsqrt(variance + eps))    # or weight * (X / sqrt(...))
// After:
//   Y = RMSNormalization<axis, epsilon>(X, weight)
//
// RMSNormalization is a standard ONNX op (opset 23, onnx/defs/nn/defs.cc)
// with a portable function-body fallback, so replacing the hand-unrolled
// decomposition with it is a pure node-count win everywhere, and it
// additionally lets any backend that *does* carry a fused/accelerated
// RMSNorm kernel dispatch straight to it -- the same role the
// Attention/RotaryEmbedding ops play for their own subgraphs. This collapses
// the 6-7 node chain (Pow-or-Mul, ReduceMean, Add, Sqrt, Reciprocal-or-Div,
// Mul, Mul) most exporters emit into a single node; a 32-layer LLaMA-style
// model has ~65 of these (2 per block plus the final norm).
//
// Handles the exporter spellings of `X * rsqrt(v)` seen in practice:
//   Div(X, Sqrt(v))                 -- matches RMSNormalization's own
//                                       reference decomposition exactly
//   Mul(X, Reciprocal(Sqrt(v)))
//   Mul(X, Div(one=1, Sqrt(v)))
// with the weight-Mul and the X-operand of each inner op allowed in either
// order. `v` must be `Add(ReduceMean(square, axes), eps)` with a single
// reduced axis (attribute, or opset >= 18 input) and keepdims=1, where
// `square` is `Mul(X, X)` or `Pow(X, 2)` over the *same* X used above, and
// every intermediate value along the chain is used exactly once.
//
// The whole matched chain (everything strictly between X/weight and the
// outer Mul) is torn down explicitly here rather than left for a later
// dead-code-elimination pass: onnx-optimizer's ``FixedPointPassManager``
// only re-sweeps its full pass list (giving ``eliminate_deadend`` another
// look) when some *other* pass in the list reports
// ``PassEfficiency::Partial`` fixed-point work during the same sweep. On a
// graph where nothing else in the default pass set happens to fire in that
// round (plausible on a small/isolated subgraph), a `Complete`-efficiency
// fuse pass -- this one included -- can otherwise leave its own orphaned
// nodes behind for a full extra `Simplify()` iteration, or indefinitely.
// Node::destroy() requires each destroyed node's outputs to already have
// zero uses, so the chain must come down innermost-consumer first: detach
// the outer Mul's edge into the chain, then destroy in the order
// MatchOuter/MatchDenominator recorded it (mirrors the explicit
// ``bn->removeInput`` + destroy teardown fuse_bn_into_conv.h uses for the
// same reason).
//
// Only fires at opset >= 23 (RMSNormalization's introducing version); pass
// `target_opset_version=23` (or higher) to onnxsim to upgrade first if the
// exported model predates it -- true of essentially all LLM exports today.
//
// Also matches the HuggingFace-style variant that upcasts to fp32 around
// the reduction (Qwen2/Qwen3/LLaMA `*RMSNorm.forward`, and what TensorRT
// Edge-LLM's dynamo export emits for every norm):
//   XU = Cast(X, to=FLOAT)
//   N  = XU * rsqrt(mean(XU^2) + eps)      # any spelling listed above
//   Y  = weight * Cast(N, to=T)            # T = X's own element type
// which is exactly RMSNormalization's reference body with
// `stash_type=FLOAT` (Cast X to the stash type, normalize, Cast back to T,
// then multiply by scale), so it fuses to
//   Y = RMSNormalization<axis, epsilon, stash_type=1>(X, weight).
// Requires X's and weight's element types to be known and equal to the
// outer Cast's target, and the upcast value to feed nothing but the chain.
//
// Half-precision models: the eps / 2 / 1 constants of an fp16 or bf16 chain
// are read in that type too. When the *plain* variant runs entirely in fp16
// (no upcast), the fused node carries an explicit stash_type=FLOAT -- the
// op's default, and what the HF reference RMSNorm does -- so the reduction
// is computed in fp32. That is deliberately not bit-faithful to the fp16
// chain: `Pow(x, 2)` in fp16 overflows to inf for |x| > ~255 in any runtime
// that really computes it in fp16 (torch, GPU kernels; onnxruntime's CPU
// provider happens to upcast internally), and real LLM residual streams
// exceed that (onnx-community/Qwen2.5-0.5B-Instruct's fp16 export reaches
// ~1,700 by layer 3; run in torch fp16 it generates repeated garbage --
// scripts/nvidia/README.md). Where the fp16 chain did not
// overflow or underflow the two agree to fp16 rounding; where it did, the
// fused node gives the finite, intended value, so onnxsim's equivalence
// check reports a mismatch for exactly those inputs.
//
// RMSNormalization reduces over *every* axis from `axis` to the last one,
// so only a ReduceMean over the last axis (-1, or rank-1 when X's rank is
// known) is fused; a single inner axis would otherwise silently widen into
// a reduction over all trailing axes.

#include <algorithm>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/float16_to_float32.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseRMSNorm final : public PredicateBasedPass {
  explicit FuseRMSNorm()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_rms_norm"; }

  struct Match {
    Value* x = nullptr;
    Value* scale = nullptr;
    int64_t axis = -1;
    float epsilon = 0.0f;
    // 0 = no stash_type attribute (plain variant); otherwise the
    // TensorProto element type the upcast variant normalizes in.
    int32_t stash_type = 0;
    // Every node strictly between (X, scale) and the outer Mul, ordered
    // innermost-consumer first (i.e. the node whose output the outer Mul
    // directly used comes first) so destroying them in this order always
    // has a zero-use output to destroy.
    std::vector<Node*> dead_chain;
  };

  static bool FetchScalarAsFloat(Value* v, float& out) {
    if (FetchSoleValueOfTensor(v, out)) {
      return true;
    }
    double d;
    if (FetchSoleValueOfTensor(v, d)) {
      out = static_cast<float>(d);
      return static_cast<double>(out) == d;
    }
    // fp16/bf16 models carry their eps / 2 / 1 constants in the model's own
    // type; FetchSoleValueOfTensor<T> only matches an exact elem_type.
    Float16 h;
    if (FetchSoleValueOfTensor(v, h)) {
      out = Float16BitsToFloat(h.bits);
      return true;
    }
    BFloat16 b;
    if (FetchSoleValueOfTensor(v, b)) {
      out = static_cast<float>(b);
      return true;
    }
    return false;
  }

  static bool IsConstantTwo(Value* v) {
    float f;
    if (FetchScalarAsFloat(v, f)) {
      return f == 2.0f;
    }
    int64_t i;
    if (FetchSoleValueOfTensor(v, i)) {
      return i == 2;
    }
    return false;
  }

  static bool IsLastAxis(Value* x, int64_t axis) {
    if (axis == -1) {
      return true;
    }
    if (!x->has_sizes()) {
      return false;
    }
    int64_t rank = static_cast<int64_t>(x->sizes().size());
    return axis == rank - 1;
  }

  // Matches `Add(ReduceMean(square, axes), eps)` feeding `sqrt_out`'s Sqrt,
  // where `square` is `Mul(X, X)` or `Pow(X, 2)`. Every intermediate along
  // the chain must be single-use. On success, appends [sqrt, add, mean,
  // square] to `chain` (innermost first).
  static bool MatchDenominator(Value* sqrt_out, Value*& x, int64_t& axis,
                               float& epsilon, std::vector<Node*>& chain) {
    if (!CheckKind(sqrt_out, kSqrt) || sqrt_out->uses().size() != 1) {
      return false;
    }
    Node* sqrt = sqrt_out->node();
    Value* add_out = sqrt->input(0);
    if (!CheckKind(add_out, kAdd) || add_out->uses().size() != 1) {
      return false;
    }
    Node* add = add_out->node();
    if (add->inputs().size() != 2) {
      return false;
    }
    Value* mean_out = nullptr;
    Value* eps_v = nullptr;
    for (int i = 0; i < 2; ++i) {
      if (CheckKind(add->input(i), kReduceMean)) {
        mean_out = add->input(i);
        eps_v = add->input(1 - i);
      }
    }
    if (mean_out == nullptr || mean_out->uses().size() != 1 ||
        !FetchScalarAsFloat(eps_v, epsilon)) {
      return false;
    }

    Node* mean = mean_out->node();
    if (GetValueFromAttrWithDefault(mean, kkeepdims, (int64_t)1) != 1) {
      return false;
    }
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(mean, kaxes, (size_t)1, axes) ||
        axes.size() != 1) {
      return false;
    }
    axis = axes[0];

    Value* square_out = mean->input(0);
    if (square_out->uses().size() != 1) {
      return false;
    }
    Node* square = square_out->node();
    bool square_ok = false;
    if (CheckKind(square, kMul) && square->inputs().size() == 2 &&
        square->input(0) == square->input(1)) {
      x = square->input(0);
      square_ok = true;
    } else if (CheckKind(square, kPow) && square->inputs().size() == 2 &&
               IsConstantTwo(square->input(1))) {
      x = square->input(0);
      square_ok = true;
    }
    if (!square_ok || !IsLastAxis(x, axis)) {
      return false;
    }

    chain.push_back(sqrt);
    chain.push_back(add);
    chain.push_back(mean);
    chain.push_back(square);
    return true;
  }

  // Matches the normalized value `norm` = `X * rsqrt(v)` / `X / sqrt(v)`
  // (see the header comment for the accepted spellings), trying both operand
  // orders of the inner commutative ops. On success, sets `x`/`axis`/`eps`
  // and fills `chain` with every node from `norm`'s producer inward,
  // innermost-consumer first.
  static bool MatchNorm(Value* norm, Value*& x, int64_t& axis, float& eps,
                        std::vector<Node*>& chain) {
    if (norm->uses().size() != 1) {
      return false;
    }
    Node* norm_node = norm->node();

    // Div(X, RMS): the exact spelling of RMSNormalization's own reference
    // decomposition (`Normalized = Div(X, RMS)`).
    if (CheckKind(norm_node, kDiv) && norm_node->inputs().size() == 2) {
      Value* x_direct = norm_node->input(0);
      Value* x_denom;
      std::vector<Node*> c;
      if (MatchDenominator(norm_node->input(1), x_denom, axis, eps, c) &&
          x_denom == x_direct) {
        c.insert(c.begin(), norm_node);
        x = x_direct;
        chain = std::move(c);
        return true;
      }
    }

    // Mul(X, 1/RMS), 1/RMS = Reciprocal(RMS) or Div(one, RMS).
    if (CheckKind(norm_node, kMul) && norm_node->inputs().size() == 2) {
      for (int j = 0; j < 2; ++j) {
        Value* x_direct = norm_node->input(j);
        Value* inv = norm_node->input(1 - j);
        if (inv->uses().size() != 1) {
          continue;
        }
        Node* inv_node = inv->node();
        Value* sqrt_out = nullptr;
        if (CheckKind(inv_node, "Reciprocal") &&
            inv_node->inputs().size() == 1) {
          sqrt_out = inv_node->input(0);
        } else if (CheckKind(inv_node, kDiv) &&
                   inv_node->inputs().size() == 2) {
          float one;
          if (!FetchScalarAsFloat(inv_node->input(0), one) || one != 1.0f) {
            continue;
          }
          sqrt_out = inv_node->input(1);
        } else {
          continue;
        }
        Value* x_denom;
        std::vector<Node*> c;
        if (MatchDenominator(sqrt_out, x_denom, axis, eps, c) &&
            x_denom == x_direct) {
          c.insert(c.begin(), inv_node);
          c.insert(c.begin(), norm_node);
          x = x_direct;
          chain = std::move(c);
          return true;
        }
      }
    }
    return false;
  }

  // Upcast variant: `norm_t` = Cast(N, to=T) where N is MatchNorm over
  // XU = Cast(X, to=FLOAT) and X is of type T. On success, `x` is the
  // *original* X, and `chain` also covers both Casts (outer Cast first,
  // upcast Cast last, so it is destroyed after every one of its uses).
  static bool MatchUpcastNorm(Value* norm_t, Value* scale, Value*& x,
                              int64_t& axis, float& eps,
                              std::vector<Node*>& chain) {
    if (norm_t->uses().size() != 1 || !CheckKind(norm_t, kCast)) {
      return false;
    }
    Node* down = norm_t->node();
    if (!down->hasAttribute(kto)) {
      return false;
    }
    const int64_t t = down->i(kto);
    Value* xu;
    std::vector<Node*> c;
    if (!MatchNorm(down->input(0), xu, axis, eps, c) || !CheckKind(xu, kCast)) {
      return false;
    }
    Node* up = xu->node();
    if (!up->hasAttribute(kto) ||
        up->i(kto) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return false;
    }
    Value* x_orig = up->input(0);
    if (x_orig->elemType() != t || scale->elemType() != t) {
      return false;
    }
    // XU must feed nothing outside the matched chain.
    for (const Use& u : xu->uses()) {
      if (std::find(c.begin(), c.end(), u.user) == c.end()) {
        return false;
      }
    }
    c.insert(c.begin(), down);
    c.push_back(up);
    x = x_orig;
    chain = std::move(c);
    return true;
  }

  // Matches the outer `weight * norm` (either operand order), where `norm`
  // is MatchNorm's plain form or MatchUpcastNorm's Cast-wrapped form.
  static bool MatchOuter(Node* n, Match& out) {
    if (!CheckKind(n, kMul) || n->inputs().size() != 2) {
      return false;
    }
    for (int i = 0; i < 2; ++i) {
      Value* norm = n->input(i);
      Value* scale = n->input(1 - i);
      Value* x;
      int64_t axis;
      float eps;
      std::vector<Node*> chain;
      if (MatchNorm(norm, x, axis, eps, chain)) {
        // A half-precision X gets an explicit stash_type=FLOAT -- see the
        // header comment on why that is intended.
        const bool half = x->elemType() == TensorProto_DataType_FLOAT16 ||
                          x->elemType() == TensorProto_DataType_BFLOAT16;
        out = Match{x,
                    scale,
                    axis,
                    eps,
                    half ? TensorProto_DataType_FLOAT : 0,
                    std::move(chain)};
        return true;
      }
      if (MatchUpcastNorm(norm, scale, x, axis, eps, chain)) {
        out = Match{x,
                    scale,
                    axis,
                    eps,
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                    std::move(chain)};
        return true;
      }
    }
    return false;
  }

  bool patternMatchPredicate(Node* n) override {
    if (getOpsetVersion(*n->owningGraph()) < 23) {
      return false;
    }
    Match m;
    return MatchOuter(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!MatchOuter(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());

    Node* rms = graph.create(Symbol("RMSNormalization"), n->outputs().size());
    rms->addInput(m.x);
    rms->addInput(m.scale);
    rms->i_(kaxis, m.axis);
    rms->f_(kepsilon, m.epsilon);
    if (m.stash_type != 0) {
      rms->i_(Symbol("stash_type"), m.stash_type);
    }
    for (int i = 0; i < static_cast<int>(n->outputs().size()); ++i) {
      rms->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    rms->insertBefore(n);
    if (!tryReplacingAllUsesWith(n, rms)) {
      return false;
    }

    // `n` itself is destroyed by the driving iterator below, but everything
    // upstream of it is not, so detach n's one remaining edge into the dead
    // chain (dropping its innermost node to zero uses) and tear the rest
    // down explicitly, innermost first.
    Value* chain_out = m.dead_chain.front()->outputs()[0];
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      if (n->input(i) == chain_out) {
        n->removeInput(i);
        break;
      }
    }
    for (Node* dead : m.dead_chain) {
      dead->destroy();
    }

    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
