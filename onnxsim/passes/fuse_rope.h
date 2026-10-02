// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Fuses the "rotate_half" rotary position embedding (RoPE) application --
// the pattern HuggingFace-style LLaMA/Mistral/Qwen-family exports apply to Q
// and K separately, right after their head-split and before the QK^T
// matmul (see test_mnn_llm_export.py's `_Attention` for a real traced
// example) -- into a single standard ONNX `RotaryEmbedding` node (opset 23,
// onnx/defs/nn/defs.cc; see fuse_rms_norm.h for the sibling native-op fusion
// this mirrors).
//
// Before, for each of Q and K independently (`X` stands for whichever; both
// sides share the same `cos`/`sin`, computed once):
//   def rotate_half(x):
//       x1, x2 = x[..., :D/2], x[..., D/2:]
//       return concat([-x2, x1], axis=-1)
//   emb = concat([angle, angle], axis=-1)   -- `angle` duplicated, once
//   cos, sin = emb.cos()[:, None], emb.sin()[:, None]   -- broadcast over heads
//   X_embed = X * cos + rotate_half(X) * sin
// After, for each of Q and K:
//   cos_half = Cos(angle)          -- half-width, computed once per side here
//   sin_half = Sin(angle)          --   (a later eliminate_common_subexpression
//                                        pass merges Q's and K's copies)
//   X_embed = RotaryEmbedding(X, cos_half, sin_half)
//
// `RotaryEmbedding`'s own reference algorithm (see its schema doc) computes
// exactly `x1*cos_half - x2*sin_half` / `x2*cos_half + x1*sin_half`
// concatenated back together, which is only equal to the hand-rolled
// `X*cos_full + rotate_half(X)*sin_full` above when `cos_full`/`sin_full`
// are genuinely `concat([cos_half, cos_half])`/`concat([sin_half,
// sin_half])` -- i.e. when `cos = Cos(concat([angle, angle]))` and `sin =
// Sin(concat([angle, angle]))` share the *same* `angle` Value on both sides
// of the duplicating Concat (checked by reference-identity, not just
// structural shape), which is exactly what a real trace of the HuggingFace
// formula above produces (`emb = torch.cat((angle, angle), dim=-1)`). If
// this can't be confirmed, the match declines rather than guessing --
// wiring an ordinary (non-duplicated) `cos`/`sin` into `RotaryEmbedding`
// would silently compute something else.
//
// `cos`/`sin`'s own `[:, None]` broadcast-over-heads unsqueeze (axis 1, to
// broadcast against X's rank-4 `[B, num_heads, S, D]` shape) is matched and
// discarded rather than carried into the fused node: `RotaryEmbedding`'s own
// reference algorithm already broadcasts a rank-3 `[B, S, D/2]` cos/sin
// cache over X's head axis internally when X is rank 4, which is exactly
// this same broadcast, so the pre-existing Unsqueeze would just be
// redundant on the new cos_half/sin_half (rank 3, matching `angle`'s own
// shape) -- unsqueezing to any *other* axis is a different broadcast this
// pass does not attempt to reproduce and declines on.
//
// Q's and K's applications are matched and fused independently (each
// anchored on its own outer Add), but since they read the same shared
// cos/sin/Concat/angle chain, whichever of the two is transformed second in
// a given pass sweep also tears down that now-fully-unused shared chain --
// see MaybeAppendSharedChain's own comment for why leaving it for a later
// dead-code-elimination pass to notice is not reliable within
// onnx-optimizer's fixed-point loop (mirrors fuse_rms_norm.h's own note on
// the same concern for its unrelated chain).
//
// Only fires at opset >= 23 (RotaryEmbedding's introducing version); pass
// `target_opset_version=23` (or higher) to onnxsim to upgrade first if the
// exported model predates it -- true of essentially all LLM exports today.
//
// Does not match `interleaved=1`-style RoPE (alternating elements rather
// than first/second half), partial rotation (`rotary_embedding_dim` <
// head_size), or the position_ids-indexed cache-table form (`cos`/`sin`
// gathered by `position_ids` from a precomputed table rather than computed
// fresh per call) -- left for follow-up.

#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Matches `sliced` as one half of `rotate_half`'s split: `Slice(x, start,
// end, axis=-1[, step=1])` where `x`'s own last dimension is known (via
// shape inference) and even. `want_first_half` selects which half: the first
// half must be exactly `[0, D/2)`; the second half must start at `D/2` and
// extend at least through `D` (real exporters commonly over-shoot with
// `end=INT64_MAX` rather than `D` exactly, which Slice's own semantics clip
// to the actual dimension size, so this accepts any `end >= D`).
inline bool MatchRopeHalfSlice(Value* sliced, Value*& x, bool want_first_half,
                               int64_t& head_size, std::vector<Node*>& chain) {
  if (!CheckKind(sliced, kSlice) || sliced->uses().size() != 1) {
    return false;
  }
  Node* sl = sliced->node();
  if (sl->inputs().size() < 4) {
    return false;
  }
  Value* data = sl->input(0);
  if (!data->has_sizes() || data->sizes().empty()) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(data->sizes().size());
  const Dimension& last_dim = data->sizes()[static_cast<size_t>(rank - 1)];
  if (!last_dim.is_int || last_dim.dim <= 0 || last_dim.dim % 2 != 0) {
    return false;
  }
  head_size = last_dim.dim;
  const int64_t half = head_size / 2;

  int64_t start = 0, end = 0, axis = 0;
  if (!GetValueFromInput(sl, 1, start, 0) ||
      !GetValueFromInput(sl, 2, end, 0) || !GetValueFromInput(sl, 3, axis, 0)) {
    return false;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis != rank - 1) {
    return false;
  }
  if (sl->inputs().size() >= 5) {
    int64_t step = 1;
    if (!GetValueFromInput(sl, 4, step, 0) || step != 1) {
      return false;
    }
  }
  if (want_first_half) {
    if (start != 0 || end != half) {
      return false;
    }
  } else {
    if (start != half || end < head_size) {
      return false;
    }
  }
  x = data;
  chain.push_back(sl);
  return true;
}

// Matches `rotated` as `rotate_half(x) = Concat([Neg(x2), x1], axis=-1)`
// where `x1`/`x2` are the first/second-half `Slice`s of the *same* `x` (see
// MatchRopeHalfSlice). Appends [concat, neg, slice_x2, slice_x1] to `chain`
// (innermost/closest-to-consumer first, so this order is always a valid
// destroy order given each node here is single-use by construction).
inline bool MatchRotateHalf(Value* rotated, Value*& x,
                            std::vector<Node*>& chain) {
  if (!CheckKind(rotated, kConcat) || rotated->uses().size() != 1) {
    return false;
  }
  Node* concat_n = rotated->node();
  if (concat_n->inputs().size() != 2) {
    return false;
  }
  Value* neg_out = concat_n->input(0);
  Value* x1_out = concat_n->input(1);
  if (!CheckKind(neg_out, kNeg) || neg_out->uses().size() != 1) {
    return false;
  }
  Node* neg_n = neg_out->node();
  if (neg_n->inputs().size() != 1) {
    return false;
  }
  Value* x2_out = neg_n->input(0);

  Value* x_from_x1 = nullptr;
  Value* x_from_x2 = nullptr;
  int64_t head_size1 = 0, head_size2 = 0;
  std::vector<Node*> x1_chain, x2_chain;
  if (!MatchRopeHalfSlice(x1_out, x_from_x1, /*want_first_half=*/true,
                          head_size1, x1_chain) ||
      !MatchRopeHalfSlice(x2_out, x_from_x2, /*want_first_half=*/false,
                          head_size2, x2_chain)) {
    return false;
  }
  if (x_from_x1 != x_from_x2 || head_size1 != head_size2) {
    return false;
  }
  x = x_from_x1;
  chain.push_back(concat_n);
  chain.push_back(neg_n);
  for (Node* nd : x2_chain) chain.push_back(nd);
  for (Node* nd : x1_chain) chain.push_back(nd);
  return true;
}

// Matches `cos_bcast`/`sin_bcast` (the two operands actually multiplied
// against X and rotate_half(X)) back to a shared, once-computed `angle`:
//   cos_bcast = Unsqueeze(Cos(Concat([angle, angle], axis=-1)), axes=[1])
//   sin_bcast = Unsqueeze(Sin(Concat([angle, angle], axis=-1)), axes=[1])
// with the Concat's two inputs required to be the *same* Value (reference
// identity, not just equal shapes) -- see this file's top comment for why
// that specific check is what makes the fusion mathematically exact. Unlike
// every other Match* here, `cos_bcast`/`sin_bcast`/their producers are NOT
// required to be single-use: they are shared between Q's and K's
// independent RoPE applications by construction, and this pass tears them
// down itself once both sides no longer need them (MaybeAppendSharedChain).
inline bool MatchRopeCosSin(Value* cos_bcast, Value* sin_bcast, Value*& angle,
                            Node*& cos_unsq, Node*& sin_unsq, Node*& cos_n,
                            Node*& sin_n, Node*& concat_n) {
  if (!CheckKind(cos_bcast, kUnsqueeze) || !CheckKind(sin_bcast, kUnsqueeze)) {
    return false;
  }
  cos_unsq = cos_bcast->node();
  sin_unsq = sin_bcast->node();
  if (cos_unsq->inputs().size() < 2 || sin_unsq->inputs().size() < 2) {
    return false;
  }
  std::vector<int64_t> cos_axes, sin_axes;
  if (!GetValueFromInput(cos_unsq->input(1), cos_axes) ||
      cos_axes != std::vector<int64_t>{1}) {
    return false;
  }
  if (!GetValueFromInput(sin_unsq->input(1), sin_axes) ||
      sin_axes != std::vector<int64_t>{1}) {
    return false;
  }
  Value* cos_full = cos_unsq->input(0);
  Value* sin_full = sin_unsq->input(0);
  if (!CheckKind(cos_full, "Cos") || !CheckKind(sin_full, "Sin")) {
    return false;
  }
  cos_n = cos_full->node();
  sin_n = sin_full->node();
  if (cos_n->inputs().size() != 1 || sin_n->inputs().size() != 1) {
    return false;
  }
  Value* concat_a = cos_n->input(0);
  Value* concat_b = sin_n->input(0);
  if (concat_a != concat_b || !CheckKind(concat_a, kConcat)) {
    return false;
  }
  concat_n = concat_a->node();
  if (concat_n->inputs().size() != 2 ||
      concat_n->input(0) != concat_n->input(1)) {
    return false;  // Must be a literal self-duplicate: concat([h, h]).
  }
  angle = concat_n->input(0);
  return true;
}

struct RopeMatch {
  Value* x = nullptr;      // Q or K's pre-rotation activation, rank-4.
  Value* angle = nullptr;  // Shared half-width angle: concat([angle,angle])
                           // is what Cos/Sin actually read.
  Node* cos_unsq = nullptr;
  Node* sin_unsq = nullptr;
  Node* cos_n = nullptr;
  Node* sin_n = nullptr;
  Node* concat_n = nullptr;
  // Everything strictly between (x, cos_bcast, sin_bcast) and `n` (the outer
  // Add), ordered innermost (closest to `n`) first -- does not include `n`
  // itself (destroyed by the pass driver, matching every other fuse pass in
  // this codebase) and does not include the shared cos/sin/Concat chain
  // (see MaybeAppendSharedChain).
  std::vector<Node*> dead_chain;
};

// One side's full match: `n = Add(Mul(x, cos_bcast), Mul(rotate_half(x),
// sin_bcast))`, in either operand order, with the two Muls in either operand
// order internally too.
inline bool MatchRopeApply(Node* n, RopeMatch& m) {
  if (!CheckKind(n, kAdd) || n->inputs().size() != 2) {
    return false;
  }
  for (int i = 0; i < 2; ++i) {
    Value* mul_a = n->input(i);
    Value* mul_b = n->input(1 - i);
    if (!CheckKind(mul_a, kMul) || mul_a->uses().size() != 1 ||
        !CheckKind(mul_b, kMul) || mul_b->uses().size() != 1) {
      continue;
    }
    Node* mul_a_n = mul_a->node();
    Node* mul_b_n = mul_b->node();
    if (mul_a_n->inputs().size() != 2 || mul_b_n->inputs().size() != 2) {
      continue;
    }
    for (int j = 0; j < 2; ++j) {
      Value* rotated_candidate = mul_b_n->input(j);
      Value* sin_bcast = mul_b_n->input(1 - j);
      Value* x_from_rotate = nullptr;
      std::vector<Node*> rotate_chain;
      if (!MatchRotateHalf(rotated_candidate, x_from_rotate, rotate_chain)) {
        continue;
      }
      for (int k = 0; k < 2; ++k) {
        Value* x_candidate = mul_a_n->input(k);
        Value* cos_bcast = mul_a_n->input(1 - k);
        if (x_candidate != x_from_rotate) {
          continue;
        }
        Value* angle = nullptr;
        Node *cos_unsq = nullptr, *sin_unsq = nullptr;
        Node *cos_n = nullptr, *sin_n = nullptr, *concat_n = nullptr;
        if (!MatchRopeCosSin(cos_bcast, sin_bcast, angle, cos_unsq, sin_unsq,
                             cos_n, sin_n, concat_n)) {
          continue;
        }
        // `RotaryEmbedding`'s schema needs `num_heads` for a 3-D X, which
        // this pass never sets (it always emits the 4-D form). X is
        // virtually always already rank 4 here in practice -- RoPE applies
        // right after Q/K's own head-split transpose -- but decline rather
        // than emit an underspecified node on the off chance it isn't.
        if (!x_candidate->has_sizes() || x_candidate->sizes().size() != 4) {
          continue;
        }
        m.x = x_candidate;
        m.angle = angle;
        m.cos_unsq = cos_unsq;
        m.sin_unsq = sin_unsq;
        m.cos_n = cos_n;
        m.sin_n = sin_n;
        m.concat_n = concat_n;
        m.dead_chain.push_back(mul_a_n);
        m.dead_chain.push_back(mul_b_n);
        for (Node* nd : rotate_chain) m.dead_chain.push_back(nd);
        return true;
      }
    }
  }
  return false;
}

// Appends the shared cos/sin/Concat chain to `dead_chain` if -- after this
// match's own mul_a_n/mul_b_n are destroyed -- it has no other remaining
// consumer, i.e. this side is the *last* of (at most) two RoPE applications
// (Q's and K's) reading it. Checked via live use counts, not match order:
// whichever of Q's/K's Add nodes this pass transforms second in a given
// sweep is the one that observes the other side's Mul(s) already gone and
// tears the shared chain down. This explicit handling -- rather than
// leaving the now-dead chain for a later dead-code-elimination pass to
// notice -- mirrors fuse_rms_norm.h's own reasoning: onnx-optimizer's
// FixedPointPassManager only re-sweeps the full pass list (giving
// eliminate_deadend another look) when some *other* pass in the list
// reports Partial-efficiency work in the same round, so a Complete-efficiency
// fuse pass' own orphaned nodes can otherwise survive in the "simplified"
// output for an extra iteration, or indefinitely if nothing else in the
// default pass set happens to fire that round.
inline bool IsGraphOutput(const Graph& graph, const Value* value) {
  for (const Value* output : graph.outputs()) {
    if (output == value) {
      return true;
    }
  }
  return false;
}

inline bool NodeHasGraphOutput(const Graph& graph, const Node* node) {
  for (const Value* output : node->outputs()) {
    if (IsGraphOutput(graph, output)) {
      return true;
    }
  }
  return false;
}

inline void MaybeAppendSharedChain(const Graph& graph, const RopeMatch& m,
                                   std::vector<Node*>& dead_chain) {
  if (NodeHasGraphOutput(graph, m.cos_unsq) ||
      NodeHasGraphOutput(graph, m.sin_unsq) ||
      NodeHasGraphOutput(graph, m.cos_n) ||
      NodeHasGraphOutput(graph, m.sin_n) ||
      NodeHasGraphOutput(graph, m.concat_n)) {
    return;
  }
  Value* cos_bcast = m.cos_unsq->output();
  Value* sin_bcast = m.sin_unsq->output();
  auto sole_use_is = [](Value* v, Node* only_user) {
    return v->uses().size() == 1 && v->uses()[0].user == only_user;
  };
  const bool cos_now_unused = sole_use_is(cos_bcast, m.dead_chain[0]);
  const bool sin_now_unused = sole_use_is(sin_bcast, m.dead_chain[1]);
  if (!cos_now_unused || !sin_now_unused) {
    return;
  }
  dead_chain.push_back(m.cos_unsq);
  dead_chain.push_back(m.sin_unsq);
  Value* cos_full = m.cos_n->output();
  Value* sin_full = m.sin_n->output();
  if (!sole_use_is(cos_full, m.cos_unsq) ||
      !sole_use_is(sin_full, m.sin_unsq)) {
    return;
  }
  dead_chain.push_back(m.cos_n);
  dead_chain.push_back(m.sin_n);
  dead_chain.push_back(m.concat_n);
}

struct FuseRope final : public PredicateBasedPass {
  explicit FuseRope()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_rope"; }

  bool patternMatchPredicate(Node* n) override {
    if (getOpsetVersion(*n->owningGraph()) < 23) {
      return false;
    }
    RopeMatch m;
    return MatchRopeApply(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    RopeMatch m;
    if (!MatchRopeApply(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());
    for (const Node* dead : m.dead_chain) {
      if (NodeHasGraphOutput(graph, dead)) {
        return false;
      }
    }
    MaybeAppendSharedChain(graph, m, m.dead_chain);

    Node* cos_half = graph.create(Symbol("Cos"), 1);
    cos_half->addInput(m.angle);
    cos_half->insertBefore(n);

    Node* sin_half = graph.create(Symbol("Sin"), 1);
    sin_half->addInput(m.angle);
    sin_half->insertBefore(n);

    Node* rope = graph.create(Symbol("RotaryEmbedding"), 1);
    rope->addInput(m.x);
    rope->addInput(cos_half->output());
    rope->addInput(sin_half->output());
    rope->insertBefore(n);
    rope->output()->copyMetadata(n->output());

    if (!tryReplacingAllUsesWith(n, rope)) {
      return false;
    }

    // `n` (the outer Add) still holds its own two input edges into
    // dead_chain[0]/[1] (mul_a_n/mul_b_n) until detached here -- unlike
    // fuse_attention.h/fuse_rms_norm.h's own anchors, BOTH of `n`'s inputs
    // are part of the dead chain here (neither is an untouched, reused
    // value), so both must be dropped before the destroy loop below can
    // reach them at zero uses.
    n->removeAllInputs();
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
