// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Fuses a "hand-written" multi-head self-attention block -- the pattern a
// typical from-scratch eager attention implementation (e.g. a
// HuggingFace-style `nn.Linear` Q/K/V projection + reshape/transpose +
// scaled dot-product + softmax + `nn.Linear` output projection, *not*
// PyTorch's built-in `nn.MultiheadAttention`, whose legacy TorchScript
// export emits a very different ~70-node trace full of dynamic-shape
// bookkeeping this pass does not attempt to recognize) -- into a single
// ONNX Runtime "com.microsoft" contrib op, `Attention`.
//
// Before (Q/K/V bias optional; each projection may be a single Gemm(x, W,
// B[, transB=1]) node or a bare MatMul(x, W) optionally followed by a
// separate Add(B, .) node -- see MatchProjection):
//   Q = Reshape(Linear(X, Wq[, Bq]), [B,S,H,D]).Transpose([0,2,1,3])
//   K = Reshape(Linear(X, Wk[, Bk]), [B,S,H,D]).Transpose([0,2,3,1])
//   V = Reshape(Linear(X, Wv[, Bv]), [B,S,H,D]).Transpose([0,2,1,3])
//   scores = (Q @ K) / sqrt(head_size)      -- or `* (1/sqrt(head_size))`,
//            or `(Q * sqrt(1/sqrt(head_size))) @ (K * sqrt(1/sqrt(head_size)))`
//            -- see MatchScaledQKMatMul for why this third, pre-scaled shape
//            is also handled: it is what `torch.nn.functional.
//            scaled_dot_product_attention`'s own ONNX export decomposes
//            into, in both PyTorch's legacy TorchScript and newer dynamo
//            exporters, once the dynamic Shape/Slice/Cast/Sqrt/... chain
//            computing that scale from Q's runtime shape is constant-folded
//            (which happens automatically, earlier in the same simplify()
//            fixed point this pass itself runs inside, whenever the model's
//            shapes are static).
//   attn   = Softmax(scores, axis=-1)
//   ctx    = (attn @ V).Transpose([0,2,1,3]).Reshape([B,S,H*D])
//   Y      = Linear(ctx, Wout[, Bout])
// After:
//   Wqkv = Concat(Wq, Wk, Wv, axis=1)      -- computed once, here
//   Bqkv = Concat(Bq, Bk, Bv)              -- only when Q/K/V all have bias
//   Y = Attention(X, Wqkv[, Bqkv], num_heads=H, scale=1/sqrt(head_size),
//                 qkv_hidden_sizes=[Nq, Nk, Nv])
//   Y = Linear(Y, Wout[, Bout])            -- unchanged, not part of the fuse
//
// This collapses roughly a dozen shape/matmul/softmax nodes per attention
// block into one `Attention` node (the output projection is left as its own
// Linear -- `Attention`'s own schema only covers the QKV side). Unlike
// `fuse_rms_norm.h`'s target (`RMSNormalization`, standard ONNX), `Attention`
// is a "com.microsoft" contrib op, so this pass adds that domain (version 1)
// to the model's opset imports the first time it fires, if not already
// present, and the result needs a "com.microsoft"-aware runtime to execute.
//
// K's head-split transpose usually lands directly at permutation [0,2,3,1]
// (X.Transpose([0,2,1,3]) is the more common overall convention, but K
// additionally needs its last two axes swapped for the Q@K^T dot product to
// be well-formed, and most exporters -- including SDPA's legacy TorchScript
// export -- fold that swap directly into K's own head-split Transpose rather
// than emitting a separate one; see MatchAttentionHeadSplit). SDPA's *dynamo*
// export instead head-splits K the same way as Q/V (perm [0,2,1,3]) and
// spells the "swap the last two axes for K^T" step separately, as a
// standalone `Reshape -> Transpose([0,2,1]) -> Reshape` 3-D round trip
// (collapse batch/head into one leading dim, swap the remaining two, restore
// the 4-D shape) rather than folding it into one 4-D permutation --
// MatchKTransposeSwapChain recognizes this alternate spelling, verifying it
// is shape-equivalent to the direct form via shape inference rather than by
// decoding the two Reshapes' (possibly dynamic) target-shape computations.
// Only self-attention (Q, K, V all reading the *same* source activation) with
// no attention mask, no past/present KV cache, and a Softmax over the last
// axis (`axis=-1` or the input's last dimension index -- both are safe here
// regardless of the pre-/post-opset-13 Softmax axis-semantics split, since
// the matched score tensor is always exactly rank 4 by construction, and
// "flatten everything before the last axis" degenerates to the same
// per-row reduction as "reduce the last axis in place" for a fixed rank)
// is handled. Q and K must have the same per-head size (required for the
// Q@K^T dot product to be well-formed); V's hidden size may differ, per
// `Attention`'s own documented `qkv_hidden_sizes` semantics.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// A single Q/K/V Linear-style projection matched feeding `out`. `out` is
// always required to be single-use: every node this struct's `chain` records
// is destroyed once the fusion fires, so nothing else may still depend on it.
struct AttentionProjectionInfo {
  Value* input = nullptr;
  Value* weight = nullptr;  // constant 2-D float32
  Value* bias = nullptr;    // constant 1-D float32, or nullptr
  bool weight_transposed = false;
  // Node(s) strictly between (input, weight[, bias]) and `out`, ordered
  // innermost (closest to `out`'s consumer) first: just the standalone
  // MatMul node when `out` is a separate bias-Add's own output (the Add
  // node itself IS `out->node()`, so it is included too, ahead of the
  // MatMul it consumes); a single entry (`out->node()` itself) when `out`
  // is already the Gemm/MatMul.
  std::vector<Node*> chain;
};

// Matches a Linear-style projection whose result is `out`: a vanilla Gemm(x,
// W[, B]) or bare MatMul(x, W) node directly (`MatchMatMulLike`), or a
// MatMul(x, W) followed by a separate Add(B, .) bias node (the shape a plain
// `nn.Linear` exported via `MatMul` rather than `Gemm` produces). `weight`
// and, if present, `bias` must be constant float32 tensors. `out` must be
// single-use -- the caller is always going to tear down its producer chain.
inline bool MatchAttentionProjection(Value* out,
                                     AttentionProjectionInfo& info) {
  if (out->uses().size() != 1) {
    return false;
  }
  Node* node = out->node();
  if (CheckKind(node, kAdd) && node->inputs().size() == 2) {
    for (int i = 0; i < 2; ++i) {
      Value* mm_out = node->input(i);
      Value* bias = node->input(1 - i);
      if (mm_out->uses().size() != 1 || !CheckKind(mm_out, kMatMul)) {
        continue;
      }
      MatMulLikeInfo mm_info;
      if (!MatchMatMulLike(mm_out->node(), mm_info)) {
        continue;
      }
      const Tensor* bias_t = FetchConstantTensor(bias);
      if (bias_t == nullptr ||
          bias_t->elem_type() != TensorProto_DataType_FLOAT ||
          bias_t->sizes().size() != 1) {
        continue;
      }
      const Tensor* w_t = FetchConstantTensor(mm_info.w);
      if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
          w_t->sizes().size() != 2) {
        continue;
      }
      info.input = mm_info.x;
      info.weight = mm_info.w;
      info.bias = bias;
      info.weight_transposed = false;  // kMatMul only, never transposed.
      info.chain = {node, mm_out->node()};
      return true;
    }
    return false;
  }
  MatMulLikeInfo mm_info;
  if (!MatchMatMulLike(node, mm_info)) {
    return false;
  }
  const Tensor* w_t = FetchConstantTensor(mm_info.w);
  if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
      w_t->sizes().size() != 2) {
    return false;
  }
  if (mm_info.bias != nullptr) {
    const Tensor* bias_t = FetchConstantTensor(mm_info.bias);
    if (bias_t == nullptr ||
        bias_t->elem_type() != TensorProto_DataType_FLOAT ||
        bias_t->sizes().size() != 1) {
      return false;
    }
  }
  info.input = mm_info.x;
  info.weight = mm_info.w;
  info.bias = mm_info.bias;
  info.weight_transposed = mm_info.weight_transposed;
  info.chain = {node};
  return true;
}

// Reads a scalar constant as a double regardless of whether it is stored as
// float32 or float64 -- FetchSoleValueOfTensor<T> requires an exact
// elem_type match (see quantize_matmul_common.h/pass_util.h), and the
// scaling constant a real export emits (e.g. `1/sqrt(head_size)`) is
// ordinary float32, not double, so asking for T=double alone would always
// silently fail. Mirrors fuse_rms_norm.h's own FetchScalarAsFloat.
inline bool FetchScalarAsDouble(Value* v, double& out) {
  if (FetchSoleValueOfTensor(v, out)) {
    return true;
  }
  float f;
  if (FetchSoleValueOfTensor(v, f)) {
    out = static_cast<double>(f);
    return true;
  }
  return false;
}

// Matches `scores` (Softmax's sole input, already known single-use by the
// caller) as a scaled Q@K^T dot product, in either of two shapes real
// exporters produce:
//
//  - "post-scaled": scores = Div(QK, c) or Mul(QK, c), where QK is itself a
//    plain MatMul -- the shape a hand-rolled `(Q @ K) / sqrt(head_size)`
//    attention implementation emits.
//  - "pre-scaled": scores = MatMul(Mul(q_side, c), Mul(k_side, c)), i.e. the
//    same combined scale split as `sqrt(c)` onto each operand *before* the
//    dot product rather than applied to its result once -- see this file's
//    top-of-file doc comment for why this is `scaled_dot_product_attention`'s
//    own decomposition. Both Muls' scalar operands must fetch to the same
//    value (within a small relative tolerance, since real graphs sometimes
//    compute the same folded constant via two syntactically distinct nodes
//    rather than sharing one Value*); an asymmetric split isn't this shape.
//
// On success sets `q_side`/`k_side` to the two (not yet head-split-matched)
// operands and `scale` to the *combined* scale factor (already squared back
// up in the pre-scaled case, so the caller never needs to know which shape
// matched). Every node this consumed -- the Div/Mul wrapper and the inner
// MatMul in the post-scaled case, or the qk MatMul and its two Mul operands
// in the pre-scaled case -- is appended to `dead_chain` **in a valid destroy
// order** (each consumer before the producer(s) feeding it: `dead_chain`'s
// own contract, since the two shapes need opposite relative orders here --
// post-scaled destroys the Div/Mul *before* the MatMul it wraps, but
// pre-scaled destroys the MatMul *before* the two Muls feeding *it* -- a
// single fixed push order in the caller can't satisfy both, so this function
// owns it instead of returning the qk MatMul node separately).
inline bool MatchScaledQKMatMul(Value* scores, Value*& q_side, Value*& k_side,
                                double& scale, std::vector<Node*>& dead_chain) {
  Node* scores_node = scores->node();

  Value* qk = nullptr;
  if (CheckKind(scores_node, kDiv) && scores_node->inputs().size() == 2) {
    double divisor = 0.0;
    if (FetchScalarAsDouble(scores_node->input(1), divisor) && divisor != 0.0) {
      qk = scores_node->input(0);
      scale = 1.0 / divisor;
    }
  } else if (CheckKind(scores_node, kMul) &&
             scores_node->inputs().size() == 2) {
    for (int i = 0; i < 2; ++i) {
      double mult = 0.0;
      if (FetchScalarAsDouble(scores_node->input(1 - i), mult)) {
        qk = scores_node->input(i);
        scale = mult;
        break;
      }
    }
  }
  if (qk != nullptr) {
    // ORT's fused Attention/GQA operators reserve an attribute value of zero
    // for their default 1/sqrt(head_size) scale. It cannot encode a source
    // computation whose effective multiplier rounds to float zero.
    if (static_cast<float>(scale) == 0.0f) {
      return false;
    }
    if (qk->uses().size() != 1 || !CheckKind(qk, kMatMul)) {
      return false;
    }
    Node* mm = qk->node();
    if (mm->inputs().size() != 2) {
      return false;
    }
    dead_chain.push_back(scores_node);  // consumer of mm's output: first.
    dead_chain.push_back(mm);
    q_side = mm->input(0);
    k_side = mm->input(1);
    return true;
  }

  if (!CheckKind(scores_node, kMatMul) || scores_node->inputs().size() != 2) {
    return false;
  }
  double c[2] = {0.0, 0.0};
  Value* operand[2] = {nullptr, nullptr};
  Node* muls[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; ++i) {
    Value* in = scores_node->input(i);
    if (in->uses().size() != 1 || !CheckKind(in, kMul)) {
      return false;
    }
    Node* mul = in->node();
    if (mul->inputs().size() != 2) {
      return false;
    }
    bool found = false;
    for (int j = 0; j < 2; ++j) {
      double v = 0.0;
      if (FetchScalarAsDouble(mul->input(1 - j), v)) {
        operand[i] = mul->input(j);
        c[i] = v;
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
    muls[i] = mul;
  }
  if (std::fabs(c[0] - c[1]) >
      1e-6 * std::max(std::fabs(c[0]), std::fabs(c[1]))) {
    return false;
  }
  const double combined_scale = c[0] * c[1];
  if (static_cast<float>(combined_scale) == 0.0f) {
    return false;
  }
  // scores_node (the qk MatMul) consumes both muls' outputs: destroy it
  // first, then its two (now-unused) producers.
  dead_chain.push_back(scores_node);
  dead_chain.push_back(muls[0]);
  dead_chain.push_back(muls[1]);
  q_side = operand[0];
  k_side = operand[1];
  scale = combined_scale;
  return true;
}

// Matches `transposed = Transpose(Reshape(proj_out, [B, S, num_heads,
// head_size]), want_perm)`, both single-use, extracting num_heads/head_size
// from the Reshape's constant target-shape input. Appends [transpose,
// reshape] to `chain` (innermost/closest-to-consumer first).
inline bool MatchAttentionHeadSplit(Value* transposed, int64_t& num_heads,
                                    int64_t& head_size, Value*& proj_out,
                                    const std::vector<int64_t>& want_perm,
                                    std::vector<Node*>& chain) {
  if (!CheckKind(transposed, kTranspose) || transposed->uses().size() != 1) {
    return false;
  }
  Node* tr = transposed->node();
  std::vector<int64_t> perm;
  if (!GetValueFromAttr(tr, kperm, perm) || perm != want_perm) {
    return false;
  }
  Value* reshaped = tr->input(0);
  if (!CheckKind(reshaped, kReshape) || reshaped->uses().size() != 1) {
    return false;
  }
  Node* rs = reshaped->node();
  if (rs->inputs().size() != 2) {
    return false;
  }
  std::vector<int64_t> shape;
  if (!GetValueFromInput(rs->input(1), shape) || shape.size() != 4) {
    return false;
  }
  num_heads = shape[2];
  head_size = shape[3];
  if (num_heads <= 0 || head_size <= 0) {
    return false;
  }
  proj_out = rs->input(0);
  chain.push_back(tr);
  chain.push_back(rs);
  return true;
}

// Matches the alternate spelling `scaled_dot_product_attention`'s *dynamo*
// exporter uses for K's "swap the last two (seq, head_size) axes" step:
// rather than a single 4-D Transpose straight to perm [0,2,3,1] (the direct
// form MatchAttentionHeadSplit's own want_perm looks for), it head-splits K
// exactly like Q/V (Transpose perm [0,2,1,3]) and then swaps the last two
// axes as a *separate* step that round-trips through 3-D:
//   Reshape(Transpose(Reshape(headsplit_out, [B*H,S,D]), [0,2,1]), [B,H,D,S])
// which is mathematically identical to `Transpose(headsplit_out, [0,1,3,2])`
// -- and, composed with headsplit_out's own [0,2,1,3] transpose from
// proj_out, to `Transpose(proj_out, [0,2,3,1])` directly, i.e. exactly what
// the direct form already looks for. Rather than re-deriving that arithmetic
// from the two Reshapes' target-shape inputs (which may themselves be
// dynamic, Shape-derived values rather than literal constants), this
// confirms the same identity a cheaper way: shape inference -- run earlier
// in the same fixed-point loop this pass itself runs inside -- has already
// resolved every intermediate value's static sizes whenever they're static
// at all, so directly comparing them is sufficient regardless of how the two
// Reshapes computed their target shapes internally. Two independent shape
// checks are both required, not just the final one: a row-major Reshape's
// semantics are only pinned down uniquely by which *trailing* dimensions it
// leaves untouched, so this checks that the first Reshape's output keeps
// `headsplit_out`'s own last two dims (S, D) unchanged in its own last two
// positions (proving the merged leading dimension can only be B*H, in that
// order -- the standard row-major flatten of exactly those two axes and no
// other grouping, e.g. not S folded in instead) *and* that the final
// Reshape's output exactly matches `headsplit_out`'s shape with its last two
// dims exchanged (proving the corresponding unmerge/split recovers B and H
// individually, in the same order, rather than some other factorization of
// the same product). Checking only the final shape is not enough on its
// own: a chain that merges a *different* pair of leading dims (or merges in
// the opposite order) can still land on the same final 4-D shape by
// coincidence while actually computing a different permutation of the
// underlying data. Declines (rather than guessing) whenever any shape isn't
// statically known.
inline bool MatchKTransposeSwapChain(Value* k_side, int64_t& num_heads,
                                     int64_t& head_size, Value*& proj_out,
                                     std::vector<Node*>& chain) {
  if (!CheckKind(k_side, kReshape) || k_side->uses().size() != 1) {
    return false;
  }
  Node* outer_rs = k_side->node();
  if (outer_rs->inputs().size() != 2) {
    return false;
  }
  Value* swapped = outer_rs->input(0);
  if (!CheckKind(swapped, kTranspose) || swapped->uses().size() != 1) {
    return false;
  }
  Node* swap_tr = swapped->node();
  std::vector<int64_t> swap_perm;
  if (!GetValueFromAttr(swap_tr, kperm, swap_perm) ||
      swap_perm != std::vector<int64_t>{0, 2, 1}) {
    return false;
  }
  Value* flattened = swap_tr->input(0);
  if (!CheckKind(flattened, kReshape) || flattened->uses().size() != 1) {
    return false;
  }
  Node* inner_rs = flattened->node();
  if (inner_rs->inputs().size() != 2) {
    return false;
  }
  Value* headsplit_out = inner_rs->input(0);
  std::vector<Node*> headsplit_chain;
  if (!MatchAttentionHeadSplit(headsplit_out, num_heads, head_size, proj_out,
                               {0, 2, 1, 3}, headsplit_chain)) {
    return false;
  }
  if (!headsplit_out->has_sizes() || !flattened->has_sizes() ||
      !k_side->has_sizes() || headsplit_out->sizes().size() != 4 ||
      flattened->sizes().size() != 3 || k_side->sizes().size() != 4) {
    return false;
  }
  const auto& h_sizes = headsplit_out->sizes();
  const auto& f_sizes = flattened->sizes();
  const auto& k_sizes = k_side->sizes();
  auto same_dim = [](const Dimension& a, const Dimension& b) {
    return a.is_int == b.is_int &&
           (a.is_int ? a.dim == b.dim : a.param == b.param);
  };
  // First Reshape must merge exactly headsplit_out's leading two dims,
  // leaving (S, D) as-is in the trailing two positions.
  if (!same_dim(h_sizes[2], f_sizes[1]) || !same_dim(h_sizes[3], f_sizes[2])) {
    return false;
  }
  if (!same_dim(h_sizes[0], k_sizes[0]) || !same_dim(h_sizes[1], k_sizes[1]) ||
      !same_dim(h_sizes[2], k_sizes[3]) || !same_dim(h_sizes[3], k_sizes[2])) {
    return false;
  }
  chain.push_back(outer_rs);
  chain.push_back(swap_tr);
  chain.push_back(inner_rs);
  for (Node* nd : headsplit_chain) chain.push_back(nd);
  return true;
}

// `Attention`'s schema requires a rank-3 X input ([batch_size,
// sequence_length, input_hidden_size]). Q/K/V's shared source is usually
// exactly the model's own rank-3 activation, but when an earlier fixed-point
// iteration has already rewritten a MatMul+Add projection into Gemm (Gemm
// requires 2-D operands, so onnx-optimizer's own fuse_matmul_add_into_gemm
// flattens a batch/sequence-dim input to 2-D with a Reshape first) before
// this pass gets its turn -- which for a `torch.nn.functional.
// scaled_dot_product_attention` export is close to guaranteed: its own scale
// computation needs an earlier constant-folding pass to resolve before
// MatchScaledQKMatMul's pattern is even structurally satisfiable, so
// Gemm-conversion (which has no such prerequisite) reliably wins that race
// -- `input` ends up rank-2 instead of rank-3.
//
// Recovers the original rank-3 activation in that specific case: when
// `input` is exactly `Reshape(x3, ...)` with `x3` rank-3 and the same
// trailing (hidden-size) dimension `input` itself has, returns `x3`. This is
// mathematically transparent to wire into Attention's X input in place of
// `input` -- fuse_matmul_add_into_gemm's flattening Reshape only ever merges
// the leading (batch/sequence) dims and keeps the last one, which is exactly
// what both Gemm's row-wise application and Attention's own per-token QKV
// projection already treat those leading dims as, so there is nothing to
// numerically "correct for", just a different, already-live Value* for the
// same quantity. Falls through to returning `input` unchanged -- rank-3,
// rank-2, or otherwise -- whenever this specific shape isn't matched; the
// caller still validates the final rank itself.
inline Value* RecoverRank3AttentionInput(Value* input) {
  if (input->has_sizes() && input->sizes().size() == 3) {
    return input;
  }
  if (!input->has_sizes() || input->sizes().size() != 2 ||
      !CheckKind(input, kReshape)) {
    return input;
  }
  Node* rs = input->node();
  if (rs->inputs().size() != 2) {
    return input;
  }
  Value* x3 = rs->input(0);
  if (!x3->has_sizes() || x3->sizes().size() != 3) {
    return input;
  }
  const Dimension& last3 = x3->sizes()[2];
  const Dimension& last2 = input->sizes()[1];
  if (last3.is_int != last2.is_int ||
      (last3.is_int && last3.dim != last2.dim) ||
      (!last3.is_int && last3.param != last2.param)) {
    return input;
  }
  return x3;
}

struct AttentionMatch {
  AttentionProjectionInfo q, k, v;
  int64_t num_heads = 0;
  double scale = 0.0;
  // Every node strictly upstream of `n` (the ctx-producing Reshape the pass
  // anchors on) down to, but not including, the QKV inputs/weights,
  // ordered innermost (closest to `n`) first, so destroying them in this
  // order always has a zero-use output to destroy. Does not include `n`
  // itself -- `n` is fully replaced by the fused Attention node's output
  // and destroyed by the pass driver instead (see FuseAttention::
  // runTransform), the same idiom fuse_rms_norm.h uses for its own anchor.
  std::vector<Node*> dead_chain;
};

// Anchors on `n` = the Reshape that produces `ctx` (the attention context,
// [B,S,H*D]) -- i.e. Reshape(Transpose(V_mm_out, [0,2,1,3]), [B,S,H*D]) --
// rather than on the (untouched, never matched or torn down) output
// projection that consumes `ctx`. This keeps the output projection, whatever
// shape it takes, entirely out of scope: only `n`'s own two inputs and
// everything transitively upstream of the Transpose are examined.
inline bool MatchAttention(Node* n, AttentionMatch& m) {
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
    return false;  // Must reduce the last of the 4 (B,H,S,S) axes.
  }
  Value* scores = softmax->input(0);
  if (scores->uses().size() != 1) {
    return false;
  }

  std::vector<Node*> v_head_chain;
  Value* v_proj_out = nullptr;
  int64_t v_heads = 0, v_head_size = 0;
  if (!MatchAttentionHeadSplit(v_t, v_heads, v_head_size, v_proj_out,
                               {0, 2, 1, 3}, v_head_chain) ||
      !MatchAttentionProjection(v_proj_out, m.v)) {
    return false;
  }

  Value* q_side = nullptr;
  Value* k_side = nullptr;
  std::vector<Node*> scale_chain;
  if (!MatchScaledQKMatMul(scores, q_side, k_side, m.scale, scale_chain)) {
    return false;
  }

  std::vector<Node*> q_head_chain;
  Value* q_proj_out = nullptr;
  int64_t q_heads = 0, q_head_size = 0;
  if (!MatchAttentionHeadSplit(q_side, q_heads, q_head_size, q_proj_out,
                               {0, 2, 1, 3}, q_head_chain) ||
      !MatchAttentionProjection(q_proj_out, m.q)) {
    return false;
  }

  std::vector<Node*> k_head_chain;
  Value* k_proj_out = nullptr;
  int64_t k_heads = 0, k_head_size = 0;
  const bool k_direct_split = MatchAttentionHeadSplit(
      k_side, k_heads, k_head_size, k_proj_out, {0, 2, 3, 1}, k_head_chain);
  if ((!k_direct_split &&
       !MatchKTransposeSwapChain(k_side, k_heads, k_head_size, k_proj_out,
                                 k_head_chain)) ||
      !MatchAttentionProjection(k_proj_out, m.k)) {
    return false;
  }

  if (q_heads != k_heads || q_heads != v_heads || q_head_size != k_head_size) {
    return false;
  }
  m.num_heads = q_heads;

  if (m.q.input != m.k.input || m.q.input != m.v.input) {
    return false;  // Self-attention only: Q/K/V share the same source.
  }
  m.q.input = m.k.input = m.v.input = RecoverRank3AttentionInput(m.q.input);
  // `Attention`'s schema requires a rank-3 X input ([batch_size,
  // sequence_length, input_hidden_size]) -- if RecoverRank3AttentionInput
  // could not find one (see its own doc comment for when/why this happens),
  // firing anyway would emit a shape-invalid Attention node (rejected at
  // load time by a real runtime) rather than a wrong-but-loadable one, so
  // this declines conservatively instead of guessing further.
  if (!m.q.input->has_sizes() || m.q.input->sizes().size() != 3) {
    return false;
  }

  const bool any_bias =
      m.q.bias != nullptr || m.k.bias != nullptr || m.v.bias != nullptr;
  const bool all_bias =
      m.q.bias != nullptr && m.k.bias != nullptr && m.v.bias != nullptr;
  if (any_bias && !all_bias) {
    return false;  // Attention has one merged bias input -- all or none.
  }

  m.dead_chain.push_back(tr_back);
  m.dead_chain.push_back(v_mm);
  m.dead_chain.push_back(softmax);
  for (Node* nd : v_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : m.v.chain) m.dead_chain.push_back(nd);
  for (Node* nd : scale_chain) m.dead_chain.push_back(nd);
  for (Node* nd : q_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : m.q.chain) m.dead_chain.push_back(nd);
  for (Node* nd : k_head_chain) m.dead_chain.push_back(nd);
  for (Node* nd : m.k.chain) m.dead_chain.push_back(nd);
  return true;
}

// Reads `w_t` (a constant 2-D float32 tensor, [N, K] when `transposed` else
// [K, N]) into a flat, row-major [K, N] host-byte-order buffer regardless of
// storage layout -- `Attention`'s merged weight has no transpose attribute
// of its own, so a Gemm(transB=1)-sourced weight must be physically
// transposed here, unlike quantize_matmul_common.h's
// QuantizeWeightPerChannelInPlace (which deliberately keeps the source
// layout for a different rewrite).
inline std::vector<float> ReadAttentionWeightAsKN(const Tensor& w_t,
                                                  bool transposed, int64_t& K,
                                                  int64_t& N) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  K = transposed ? dim1 : dim0;
  N = transposed ? dim0 : dim1;
  const std::vector<float> data = ReadFloatMatrix(w_t);
  std::vector<float> out(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      out[static_cast<size_t>(k * N + n)] =
          transposed ? data[static_cast<size_t>(n * K + k)]
                     : data[static_cast<size_t>(k * N + n)];
    }
  }
  return out;
}

// Reads a constant 1-D float32 tensor into a flat host-byte-order buffer.
inline std::vector<float> ReadAttentionBiasVector(const Tensor& t) {
  const int64_t numel = t.sizes()[0];
  if (t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(t.data<float>(), numel);
  }
  return t.floats();
}

struct FuseAttention final : public PredicateBasedPass {
  explicit FuseAttention()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_attention"; }

  bool patternMatchPredicate(Node* n) override {
    AttentionMatch m;
    return MatchAttention(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    AttentionMatch m;
    if (!MatchAttention(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());

    const Tensor* wq_t = FetchConstantTensor(m.q.weight);
    const Tensor* wk_t = FetchConstantTensor(m.k.weight);
    const Tensor* wv_t = FetchConstantTensor(m.v.weight);
    int64_t kq = 0, nq = 0, kk = 0, nk = 0, kv = 0, nv = 0;
    const std::vector<float> wq =
        ReadAttentionWeightAsKN(*wq_t, m.q.weight_transposed, kq, nq);
    const std::vector<float> wk =
        ReadAttentionWeightAsKN(*wk_t, m.k.weight_transposed, kk, nk);
    const std::vector<float> wv =
        ReadAttentionWeightAsKN(*wv_t, m.v.weight_transposed, kv, nv);
    if (kq != kk || kq != kv) {
      return false;  // Q/K/V must all read the same input_hidden_size.
    }

    Tensor merged_w;
    merged_w.elem_type() = TensorProto_DataType_FLOAT;
    merged_w.sizes() = {kq, nq + nk + nv};
    merged_w.floats().resize(static_cast<size_t>(kq * (nq + nk + nv)));
    for (int64_t k = 0; k < kq; ++k) {
      float* row = merged_w.floats().data() + k * (nq + nk + nv);
      std::copy(wq.begin() + k * nq, wq.begin() + (k + 1) * nq, row);
      std::copy(wk.begin() + k * nk, wk.begin() + (k + 1) * nk, row + nq);
      std::copy(wv.begin() + k * nv, wv.begin() + (k + 1) * nv, row + nq + nk);
    }
    Value* weights_v = graph.addInitializerAndCreateValue(merged_w);

    // Always synthesize Attention's (schema-optional) bias input, even when
    // Q/K/V have none, rather than omitting it: at least one real ONNX
    // Runtime CPU build (1.29.0, verified directly against a minimal
    // hand-built model, independent of this pass) segfaults executing its
    // own `Attention` kernel with only 2 inputs -- a zero-filled bias is a
    // no-op numerically and sidesteps that kernel path entirely.
    Tensor merged_b;
    merged_b.elem_type() = TensorProto_DataType_FLOAT;
    merged_b.sizes() = {nq + nk + nv};
    merged_b.floats().assign(static_cast<size_t>(nq + nk + nv), 0.0f);
    if (m.q.bias != nullptr) {
      const Tensor* bq_t = FetchConstantTensor(m.q.bias);
      const Tensor* bk_t = FetchConstantTensor(m.k.bias);
      const Tensor* bv_t = FetchConstantTensor(m.v.bias);
      const std::vector<float> bq = ReadAttentionBiasVector(*bq_t);
      const std::vector<float> bk = ReadAttentionBiasVector(*bk_t);
      const std::vector<float> bv = ReadAttentionBiasVector(*bv_t);
      if (static_cast<int64_t>(bq.size()) != nq ||
          static_cast<int64_t>(bk.size()) != nk ||
          static_cast<int64_t>(bv.size()) != nv) {
        return false;
      }
      std::copy(bq.begin(), bq.end(), merged_b.floats().begin());
      std::copy(bk.begin(), bk.end(), merged_b.floats().begin() + nq);
      std::copy(bv.begin(), bv.end(), merged_b.floats().begin() + nq + nk);
    }
    Value* bias_v = graph.addInitializerAndCreateValue(merged_b);

    Node* attn = graph.create(Symbol("Attention"), 1);
    attn->addInput(m.q.input);
    attn->addInput(weights_v);
    attn->addInput(bias_v);
    attn->i_(Symbol("num_heads"), m.num_heads);
    attn->f_(Symbol("scale"), static_cast<float>(m.scale));
    attn->is_(Symbol("qkv_hidden_sizes"), std::vector<int64_t>{nq, nk, nv});
    attn->setDomain("com.microsoft");
    attn->insertBefore(n);
    // No copyMetadata here: `attn`'s output is always rank-3
    // ([batch_size, sequence_length, hidden_size], per Attention's own
    // schema), which is not necessarily n's own shape -- see the Reshape
    // built below. Left for the next shape-inference pass to (re)infer.

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

    // `n`'s own target shape (its second input) is reused, unchanged, to
    // reshape `attn`'s raw output into whatever shape `n`'s consumers
    // actually expect -- almost always identical to `attn`'s own natural
    // [B,S,H*D] shape (a cheap identity reshape onnx-optimizer's own
    // eliminate_nop_reshape prunes in a later fixed-point iteration), but
    // when an earlier iteration has already collapsed `n`'s reshape target
    // together with a downstream Gemm's own input-flattening reshape into
    // one direct-to-2-D [B*S,H*D] target, substituting `attn`'s raw output
    // for every use of `n` directly (the way this used to work) would
    // silently mismatch that shape. Reusing `n`'s own target value here --
    // rather than assuming it is always the "natural" 3-D one -- is correct
    // either way, since it is simply whatever `n` already reshaped this same
    // ctx data into.
    Value* n_shape = n->input(1);
    Node* reshape_out = graph.create(Symbol("Reshape"), 1);
    reshape_out->addInput(attn->output());
    reshape_out->addInput(n_shape);
    reshape_out->insertBefore(n);
    reshape_out->output()->copyMetadata(n->output());

    // `n` is fully replaced by `reshape_out`'s output -- whatever consumed
    // `ctx` (the untouched output projection) now reads straight from it --
    // and `n` itself is destroyed by the pass driver below
    // (destroy_current = DestroyOne), the same idiom fuse_rms_norm.h and
    // fuse_add_bias_into_conv.h use for their own anchors.
    if (!tryReplacingAllUsesWith(n, reshape_out)) {
      return false;
    }

    // `n` still holds its own edge into the dead chain (its input(0), the
    // Transpose's output) until we detach it here; only then does the dead
    // chain's own destroy loop below see every node's output at zero uses
    // exactly when it is destroyed, innermost (closest to `n`) first.
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
