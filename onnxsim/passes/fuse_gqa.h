// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Fuses a "hand-written" causal grouped-query/multi-query attention block
// (fewer K/V heads than Q heads, each K/V head shared by a contiguous group
// of Q heads via the standard HuggingFace `repeat_kv` -- Unsqueeze->Expand->
// Reshape -- broadcast, plus an additive causal mask) into ONNX Runtime's
// "com.microsoft" `GroupQueryAttention` contrib op.
//
// Before (Q/K/V bias optional, matched via MatchAttentionProjection; K/V's
// repeat_kv wrapping matched via MatchRepeatKV; see fuse_attention.h for the
// non-GQA sibling this reuses several matchers from):
//   Q = Reshape(Linear(X, Wq[, Bq]), [B,S,NH,D]).Transpose([0,2,1,3])
//   K0 = Reshape(Linear(X, Wk[, Bk]), [B,S,NKV,D]).Transpose([0,2,1,3])
//   V0 = Reshape(Linear(X, Wv[, Bv]), [B,S,NKV,D]).Transpose([0,2,1,3])
//   K = Reshape(Expand(Unsqueeze(K0, 2), [B,NKV,n_rep,S,D]), [B,NH,S,D])
//   V = Reshape(Expand(Unsqueeze(V0, 2), [B,NKV,n_rep,S,D]), [B,NH,S,D])
//   scores = (Q @ K.Transpose([0,1,3,2])) / sqrt(head_size)   -- or the
//            pre-scaled MatchScaledQKMatMul shape; see fuse_attention.h
//   masked = scores + causal_mask     -- causal_mask a *constant* tensor
//            (see VerifyCausalMaskConstant for why this must be provable,
//            not merely assumed, and what that costs in practical coverage)
//   attn   = Softmax(masked, axis=-1)
//   ctx    = (attn @ V).Transpose([0,2,1,3]).Reshape([B,S,NH*D])
//   Y      = Linear(ctx, Wout[, Bout])
// After:
//   Y = GroupQueryAttention(Linear(X,Wq[,Bq]), Linear(X,Wk[,Bk]),
//                           Linear(X,Wv[,Bv]), <skipped>, <skipped>,
//                           seqlens_k, total_sequence_length,
//                           num_heads=NH, kv_num_heads=NKV,
//                           scale=1/sqrt(head_size))
//   Y = Linear(Y, Wout[, Bout])            -- unchanged, not part of the fuse
//
// Unlike `Attention` (fuse_attention.h's own target), `GroupQueryAttention`'s
// `query`/`key`/`value` inputs are each already-projected, rank-3
// `[batch, seq, hidden]` tensors -- the op does its own internal head-split
// and repeat-broadcast -- so this fusion feeds it the Q/K/V *projection*
// outputs directly (no weight merging needed, and Q/K/V's own projection
// chain -- MatMul[+Add] -- is left completely alone) and tears down
// everything from the head-split onward instead.
//
// **`GroupQueryAttention` always applies causal masking internally** (its
// own schema doc: "implements causal grouped-query attention"; confirmed
// empirically -- its output matches a manual causal-masked reference to
// float32 precision and diverges by ~3 from a manual *bidirectional*
// reference on the same random inputs). There is no attribute to disable
// this. Consequently:
//   - A bidirectional (unmasked, or non-causal-masked) GQA/MQA block cannot
//     be fused by this pass at all -- no ONNX Runtime contrib op supports
//     that combination (`Attention`/`MultiHeadAttention` both require equal
//     Q/K/V head counts; verified `MultiHeadAttention` rejects mismatched
//     hidden sizes at runtime). This pass simply never matches such a block.
//   - Because a real GQA export's causal mask is essentially always a
//     *runtime* input (built from `position_ids`/padding at inference time,
//     not a compile-time constant), and because wiring a mask through as
//     `attention_bias` on top of `GroupQueryAttention`'s own unconditional
//     causal masking would be a genuine *decline*-vs-*guess* judgment call
//     this pass declines rather than makes (an insufficiently-restrictive
//     runtime mask could be silently overridden by the stricter built-in
//     causal masking), this pass only fires when the additive mask is a
//     provable **constant** matching the causal pattern exactly (see
//     VerifyCausalMaskConstant). This is a real, deliberate scope
//     narrowing versus most real-world exports -- left as a documented
//     follow-up (recognizing a standard causal-mask-*construction*
//     subgraph, rather than requiring the mask already be folded to a
//     constant, would widen this considerably).
//   - `seqlens_k`/`total_sequence_length` (mandatory KV-cache bookkeeping
//     inputs `GroupQueryAttention`'s schema requires even for a plain,
//     no-cache forward pass) are synthesized here as constants (`S-1` per
//     batch row, and `S`), which requires `batch_size`/`sequence_length` to
//     be statically known -- declines otherwise.
//
// Only self-attention (Q, K, V all reading the *same* source activation),
// equal Q/K/V head_size, and Q's own head count an exact positive multiple
// of K/V's (the `repeat_kv` structural precondition) is handled.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/fuse_attention.h"
#include "passes/quantize_conv_common.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Matches `wrapped` as the HuggingFace `repeat_kv` broadcast of `raw`:
//   wrapped = Reshape(Expand(Unsqueeze(raw, axes=[2]), shape), merge_shape)
// `raw` (shape [B, NKV, S, D]) is duplicated `n_rep` times along a freshly
// inserted axis and flattened back into the heads axis, producing `wrapped`
// (shape [B, NKV*n_rep, S, D]) -- exactly HuggingFace's own reference
// `repeat_kv(x, n_rep) = x[:,:,None].expand(B,NKV,n_rep,S,D).reshape(B,
// NKV*n_rep,S,D)`, which is what `GroupQueryAttention`'s own internal
// broadcast semantics assume (each of the NKV raw heads maps to a
// *contiguous* run of `n_rep` heads in `wrapped`, not an interleaved/tiled
// one). `n_rep` is derived from shape metadata (wrapped's heads dim divided
// by raw's), not decoded from the Expand/Reshape's own (possibly dynamic)
// target-shape inputs -- see MatchKTransposeSwapChain in fuse_attention.h
// for why comparing shapes this way is sufficient and this file's own
// top comment for why static shapes are required here regardless.
inline bool MatchRepeatKV(Value* wrapped, Value*& raw, int64_t& n_rep,
                          std::vector<Node*>& chain) {
  if (!CheckKind(wrapped, kReshape) || wrapped->uses().size() != 1) {
    return false;
  }
  Node* merge_rs = wrapped->node();
  if (merge_rs->inputs().size() != 2) {
    return false;
  }
  Value* expanded = merge_rs->input(0);
  if (!CheckKind(expanded, "Expand") || expanded->uses().size() != 1) {
    return false;
  }
  Node* expand_n = expanded->node();
  if (expand_n->inputs().size() != 2) {
    return false;
  }
  Value* unsq_out = expand_n->input(0);
  if (!CheckKind(unsq_out, kUnsqueeze) || unsq_out->uses().size() != 1) {
    return false;
  }
  Node* unsq_n = unsq_out->node();
  if (unsq_n->inputs().size() < 2) {
    return false;
  }
  std::vector<int64_t> axes;
  if (!GetValueFromInput(unsq_n->input(1), axes) ||
      axes != std::vector<int64_t>{2}) {
    return false;
  }
  raw = unsq_n->input(0);
  if (!raw->has_sizes() || !wrapped->has_sizes() || raw->sizes().size() != 4 ||
      wrapped->sizes().size() != 4) {
    return false;
  }
  const auto& r = raw->sizes();
  const auto& w = wrapped->sizes();
  auto same_dim = [](const Dimension& a, const Dimension& b) {
    return a.is_int == b.is_int &&
           (a.is_int ? a.dim == b.dim : a.param == b.param);
  };
  if (!same_dim(r[0], w[0]) || !same_dim(r[2], w[2]) || !same_dim(r[3], w[3])) {
    return false;
  }
  if (!r[1].is_int || !w[1].is_int || r[1].dim <= 0 ||
      w[1].dim % r[1].dim != 0) {
    return false;
  }
  n_rep = w[1].dim / r[1].dim;
  if (n_rep < 1) {
    return false;
  }
  chain.push_back(merge_rs);
  chain.push_back(expand_n);
  chain.push_back(unsq_n);
  return true;
}

// Verifies `mask` is a *constant* float32 tensor of shape [1, 1, seq_len,
// seq_len] matching the standard additive causal pattern exactly: 0.0 on
// and below the diagonal (position j can attend to position i for j <= i),
// negative infinity strictly above it. `GroupQueryAttention`
// already applies exactly this masking internally and unconditionally (see
// this file's top comment), so a real causal mask is redundant to pass
// through explicitly -- but *some* mask must be present and provably
// exactly this pattern for the fusion to be sound, since a graph with no
// mask at all (bidirectional) or a differently-shaped one must not be
// silently reinterpreted as causal.
inline bool VerifyCausalMaskConstant(Value* mask, int64_t seq_len) {
  const Tensor* t = FetchConstantTensor(mask);
  if (t == nullptr || t->elem_type() != TensorProto_DataType_FLOAT) {
    return false;
  }
  const auto& sizes = t->sizes();
  if (sizes.size() != 4 || sizes[0] != 1 || sizes[1] != 1 ||
      sizes[2] != seq_len || sizes[3] != seq_len) {
    return false;
  }
  const std::vector<float> data = ReadFloatTensorFlat(*t);
  if (static_cast<int64_t>(data.size()) != seq_len * seq_len) {
    return false;
  }
  for (int64_t i = 0; i < seq_len; ++i) {
    for (int64_t j = 0; j < seq_len; ++j) {
      const float v = data[static_cast<size_t>(i * seq_len + j)];
      if (j <= i) {
        if (v != 0.0f) {
          return false;
        }
      } else if (!std::isinf(v) || v >= 0.0f) {
        return false;
      }
    }
  }
  return true;
}

struct GQAMatch {
  AttentionProjectionInfo q, k, v;
  // Q/K/V's own projection *output* Values (rank 3, [B,S,hidden]) --
  // GroupQueryAttention's query/key/value inputs, wired in directly.
  Value* q_proj_out = nullptr;
  Value* k_proj_out = nullptr;
  Value* v_proj_out = nullptr;
  int64_t num_heads = 0;     // Q's head count.
  int64_t kv_num_heads = 0;  // K/V's (shared) head count; num_heads must be
                             // an exact positive multiple of this.
  double scale = 0.0;
  int64_t batch_size = 0;
  int64_t seq_len = 0;
  // Every node strictly upstream of `n` down to, but not including, the
  // QKV *projection outputs* (unlike fuse_attention.h's own dead_chain,
  // which also consumes the projections themselves into a merged weight,
  // GroupQueryAttention reads each projection's output directly, so
  // MatchAttentionProjection's own `chain` for q/k/v is deliberately never
  // added here), ordered innermost (closest to `n`) first.
  std::vector<Node*> dead_chain;
};

inline bool MatchGQA(Node* n, GQAMatch& m) {
  if (!CheckKind(n, kReshape) || n->inputs().size() != 2 ||
      n->outputs().size() != 1) {
    return false;
  }
  Value* transposed_back = n->input(0);
  if (!CheckKind(transposed_back, kTranspose) ||
      transposed_back->uses().size() != 1) {
    return false;
  }
  Node* tr_back = transposed_back->node();
  std::vector<int64_t> perm_back;
  if (!GetValueFromAttr(tr_back, kperm, perm_back) ||
      perm_back != std::vector<int64_t>{0, 2, 1, 3}) {
    return false;
  }
  Value* v_matmul_out = tr_back->input(0);
  if (!CheckKind(v_matmul_out, kMatMul) || v_matmul_out->uses().size() != 1) {
    return false;
  }
  Node* v_mm = v_matmul_out->node();
  if (v_mm->inputs().size() != 2) {
    return false;
  }
  Value* softmax_out = nullptr;
  Value* v_t = nullptr;
  for (int i = 0; i < 2; ++i) {
    if (CheckKind(v_mm->input(i), kSoftmax)) {
      softmax_out = v_mm->input(i);
      v_t = v_mm->input(1 - i);
    }
  }
  if (softmax_out == nullptr || softmax_out->uses().size() != 1) {
    return false;
  }
  Node* softmax = softmax_out->node();
  if (softmax->inputs().size() != 1) {
    return false;
  }
  const int64_t sm_axis =
      GetValueFromAttrWithDefault(softmax, kaxis, int64_t(-1));
  if (sm_axis != -1 && sm_axis != 3) {
    return false;
  }

  // V-side: unwrap repeat_kv first (required -- if this doesn't match,
  // there's no GQA/MQA broadcast here and fuse_attention.h's plain
  // Attention fusion, if applicable, is the right pass instead).
  std::vector<Node*> v_repeat_chain;
  Value* v_raw = nullptr;
  int64_t v_n_rep = 0;
  if (!MatchRepeatKV(v_t, v_raw, v_n_rep, v_repeat_chain)) {
    return false;
  }
  std::vector<Node*> v_head_chain;
  int64_t v_heads = 0, v_head_size = 0;
  if (!MatchAttentionHeadSplit(v_raw, v_heads, v_head_size, m.v_proj_out,
                               {0, 2, 1, 3}, v_head_chain) ||
      !MatchAttentionProjection(m.v_proj_out, m.v)) {
    return false;
  }

  // scores = softmax's input, but wrapped in an additive causal mask:
  // masked = Add(unmasked_scores, causal_mask).
  Value* masked = softmax->input(0);
  if (masked->uses().size() != 1 || !CheckKind(masked, kAdd)) {
    return false;
  }
  Node* mask_add = masked->node();
  if (mask_add->inputs().size() != 2) {
    return false;
  }
  Value* q_side = nullptr;
  Value* k_side = nullptr;
  std::vector<Node*> scale_chain;
  Value* causal_mask = nullptr;
  bool scores_matched = false;
  for (int i = 0; i < 2; ++i) {
    Value* scores_candidate = mask_add->input(i);
    Value* mask_candidate = mask_add->input(1 - i);
    if (scores_candidate->uses().size() != 1) {
      continue;
    }
    std::vector<Node*> candidate_scale_chain;
    Value* q_candidate = nullptr;
    Value* k_candidate = nullptr;
    double scale_candidate = 0.0;
    if (!MatchScaledQKMatMul(scores_candidate, q_candidate, k_candidate,
                             scale_candidate, candidate_scale_chain)) {
      continue;
    }
    q_side = q_candidate;
    k_side = k_candidate;
    m.scale = scale_candidate;
    scale_chain = candidate_scale_chain;
    causal_mask = mask_candidate;
    scores_matched = true;
    break;
  }
  if (!scores_matched) {
    return false;
  }

  std::vector<Node*> q_head_chain;
  int64_t q_heads = 0, q_head_size = 0;
  if (!MatchAttentionHeadSplit(q_side, q_heads, q_head_size, m.q_proj_out,
                               {0, 2, 1, 3}, q_head_chain) ||
      !MatchAttentionProjection(m.q_proj_out, m.q)) {
    return false;
  }

  // K-side: the QK^T matmul needs K^T, i.e. K's last two axes swapped --
  // here that's a single direct Transpose([0,1,3,2]) over the *already
  // repeat_kv-wrapped* K (unlike fuse_attention.h's non-GQA K matching,
  // where the swap is fused directly into K's own head-split transpose).
  if (!CheckKind(k_side, kTranspose) || k_side->uses().size() != 1) {
    return false;
  }
  Node* k_swap = k_side->node();
  std::vector<int64_t> k_swap_perm;
  if (!GetValueFromAttr(k_swap, kperm, k_swap_perm) ||
      k_swap_perm != std::vector<int64_t>{0, 1, 3, 2}) {
    return false;
  }
  Value* k_repeated = k_swap->input(0);
  std::vector<Node*> k_repeat_chain;
  Value* k_raw = nullptr;
  int64_t k_n_rep = 0;
  if (!MatchRepeatKV(k_repeated, k_raw, k_n_rep, k_repeat_chain)) {
    return false;
  }
  std::vector<Node*> k_head_chain;
  int64_t k_heads = 0, k_head_size = 0;
  if (!MatchAttentionHeadSplit(k_raw, k_heads, k_head_size, m.k_proj_out,
                               {0, 2, 1, 3}, k_head_chain) ||
      !MatchAttentionProjection(m.k_proj_out, m.k)) {
    return false;
  }

  if (k_heads != v_heads || k_n_rep != v_n_rep || q_head_size != k_head_size ||
      q_head_size != v_head_size) {
    return false;
  }
  if (q_heads != k_heads * k_n_rep) {
    return false;  // Should be implied by MatchRepeatKV's own arithmetic,
                   // but verify explicitly rather than trust it silently.
  }
  m.num_heads = q_heads;
  m.kv_num_heads = k_heads;

  if (m.q.input != m.k.input || m.q.input != m.v.input) {
    return false;  // Self-attention only.
  }
  // Unlike fuse_attention.h's own Attention fusion, GroupQueryAttention
  // reads q_proj_out/k_proj_out/v_proj_out directly (never `m.q.input`
  // itself), so no rank-3-recovery on `m.q.input` is needed here -- only
  // q_proj_out's own (always rank-3, by MatchAttentionProjection's own
  // contract) shape matters, for reading batch_size/sequence_length.
  if (!m.q_proj_out->has_sizes() || m.q_proj_out->sizes().size() != 3) {
    return false;
  }
  const auto& qp_sizes = m.q_proj_out->sizes();
  if (!qp_sizes[0].is_int || !qp_sizes[1].is_int || qp_sizes[0].dim <= 0 ||
      qp_sizes[1].dim <= 0) {
    return false;  // batch_size/sequence_length must be statically known --
                   // seqlens_k/total_sequence_length are synthesized as
                   // compile-time constants below.
  }
  m.batch_size = qp_sizes[0].dim;
  m.seq_len = qp_sizes[1].dim;

  if (!VerifyCausalMaskConstant(causal_mask, m.seq_len)) {
    return false;
  }

  const bool any_bias =
      m.q.bias != nullptr || m.k.bias != nullptr || m.v.bias != nullptr;
  const bool all_bias =
      m.q.bias != nullptr && m.k.bias != nullptr && m.v.bias != nullptr;
  if (any_bias && !all_bias) {
    return false;  // Not required by the op, but keeps this consistent with
                   // fuse_attention.h's own Attention fusion; an all/none
                   // mismatch is unusual enough to not be worth a further
                   // special case here.
  }

  m.dead_chain.push_back(tr_back);
  m.dead_chain.push_back(v_mm);
  m.dead_chain.push_back(softmax);
  m.dead_chain.push_back(mask_add);
  for (Node* nd : v_repeat_chain) m.dead_chain.push_back(nd);
  for (Node* nd : v_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : scale_chain) m.dead_chain.push_back(nd);
  for (Node* nd : q_head_chain) m.dead_chain.push_back(nd);
  m.dead_chain.push_back(k_swap);
  for (Node* nd : k_repeat_chain) m.dead_chain.push_back(nd);
  for (Node* nd : k_head_chain) m.dead_chain.push_back(nd);
  return true;
}

struct FuseGQA final : public PredicateBasedPass {
  explicit FuseGQA()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_gqa"; }

  bool patternMatchPredicate(Node* n) override {
    GQAMatch m;
    return MatchGQA(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    GQAMatch m;
    if (!MatchGQA(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());

    Tensor seqlens_k_t;
    seqlens_k_t.elem_type() = TensorProto_DataType_INT32;
    seqlens_k_t.sizes() = {m.batch_size};
    seqlens_k_t.int32s().assign(static_cast<size_t>(m.batch_size),
                                static_cast<int32_t>(m.seq_len - 1));
    Value* seqlens_k_v = graph.addInitializerAndCreateValue(seqlens_k_t);

    Tensor total_seq_t;
    total_seq_t.elem_type() = TensorProto_DataType_INT32;
    total_seq_t.sizes() = {};
    total_seq_t.int32s().assign(1, static_cast<int32_t>(m.seq_len));
    Value* total_seq_v = graph.addInitializerAndCreateValue(total_seq_t);

    // past_key/past_value are skipped (this fusion never has a KV cache to
    // carry over -- see this file's top comment): ONNX represents a skipped
    // *middle* optional input as an edge from a dedicated kUndefined-kind
    // node, which the encoder serializes back to an empty input name
    // regardless of where in the node list it sits.
    Node* undef = graph.create(kUndefined, 1);
    undef->insertBefore(n);
    undef->output()->setUniqueName("");

    // GroupQueryAttention's schema requires at least 3 outputs (output,
    // present_key, present_value) even though only "output" is used here --
    // creating it with just 1 fails ONNX Runtime's own schema validation
    // ("has output size 1 not in range [min=3, max=4]"), caught by CI on a
    // real run rather than by the checker (which didn't flag it). The two
    // extra outputs are simply left unused; nothing references them.
    Node* gqa = graph.create(Symbol("GroupQueryAttention"), 3);
    gqa->addInput(m.q_proj_out);
    gqa->addInput(m.k_proj_out);
    gqa->addInput(m.v_proj_out);
    gqa->addInput(undef->output());  // past_key (unused)
    gqa->addInput(undef->output());  // past_value (unused)
    gqa->addInput(seqlens_k_v);
    gqa->addInput(total_seq_v);
    gqa->insertBefore(n);
    gqa->i_(Symbol("num_heads"), m.num_heads);
    gqa->i_(Symbol("kv_num_heads"), m.kv_num_heads);
    gqa->f_(Symbol("scale"), static_cast<float>(m.scale));
    gqa->setDomain("com.microsoft");

    bool has_ms_domain = false;
    for (const OpSetID& opset : graph.opset_versions_mutable()) {
      if (opset.domain() == "com.microsoft") {
        has_ms_domain = true;
        break;
      }
    }
    if (!has_ms_domain) {
      graph.opset_versions_mutable().emplace_back("com.microsoft", 1);
    }

    Value* n_shape = n->input(1);
    Node* reshape_out = graph.create(Symbol("Reshape"), 1);
    reshape_out->addInput(gqa->outputs()[0]);
    reshape_out->addInput(n_shape);
    reshape_out->insertBefore(n);
    reshape_out->output()->copyMetadata(n->output());

    if (!tryReplacingAllUsesWith(n, reshape_out)) {
      return false;
    }

    n->removeInput(0);
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
