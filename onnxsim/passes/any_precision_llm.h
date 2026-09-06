// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Any-Precision LLM (Park et al., 2024, ICML 2024, "Any-Precision LLM:
// Low-Cost Deployment of Multiple, Different-Sized LLMs") -- C++ port of
// any_precision_llm.py's own apply_any_precision_llm. See that module's
// docstring for the full rationale (a nested bit-plane code, built once to
// a maximum bit-width via repeated within-bin bisection, where every lower
// bit-width's own code is exactly recoverable from the max-depth code by a
// plain integer right-shift -- one quantization pass serves any precision
// up to its own ceiling).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own (channel, K-block)
//                                   nested-bit-plane quantize-dequantize
//                                   round trip at AnyPrecisionLlmBits() bits
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor. Everything else is left alone.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM any_precision_llm.py: this port's own
// per-bin bisection groups indices by current code using an
// std::unordered_map (iteration order over bins is therefore
// hash-order-dependent, not numerically significant since each bin's own
// split only ever looks at that bin's own min/max) -- ties at a bin's exact
// split threshold, and floating-point summation order inside a per-bin mean,
// can each differ in the last ULP or two from the Python port's own numpy
// reduction order. As with apply_quarot/apply_quarot_cpp, this port and
// any_precision_llm.py's apply_any_precision_llm are independently-correct,
// non-interchangeable entry points -- not required to be bit-for-bit
// identical, only similarly accurate (verified in this pass's own Python
// test file: both satisfy the same exact nesting invariant, and both
// improve reconstruction error monotonically with more bits).

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Read/written by ApplyAnyPrecisionLlm (quantize_entry.cpp) immediately
// before invoking OptimizeFixed, the same "global mutable reference,
// reconfigured per call" pattern QuarotSeed()/QuarotBlockSize() already use
// for their own pass.
inline int64_t& AnyPrecisionLlmBits() {
  static int64_t bits = 4;
  return bits;
}
inline int64_t& AnyPrecisionLlmMaxBits() {
  static int64_t max_bits = 8;
  return max_bits;
}
inline int64_t& AnyPrecisionLlmBlockSize() {
  static int64_t block_size = 32;
  return block_size;
}

namespace any_precision_llm_detail {

// Mirrors any_precision_llm.py's own _nested_bitplane_codes exactly: at
// each of `max_bits` steps, every *existing* bin (grouped by its own
// current code) is bisected at the midpoint of that bin's own current
// member values, appending one more low-order bit to every code in it. A
// singleton bin (no member left to compare against) just gets a trailing
// zero bit -- the same as the Python port's own `bin_values.size() <= 1`
// case.
inline std::vector<int64_t> NestedBitplaneCodes(
    const std::vector<double>& values, int64_t max_bits) {
  const size_t n = values.size();
  std::vector<int64_t> codes(n, 0);
  for (int64_t bit = 0; bit < max_bits; ++bit) {
    std::unordered_map<int64_t, std::vector<size_t>> bins;
    for (size_t i = 0; i < n; ++i) {
      bins[codes[i]].push_back(i);
    }
    for (const auto& kv : bins) {
      const std::vector<size_t>& idxs = kv.second;
      if (idxs.size() <= 1) {
        for (size_t i : idxs) {
          codes[i] = codes[i] * 2;
        }
        continue;
      }
      double lo = values[idxs[0]];
      double hi = values[idxs[0]];
      for (size_t i : idxs) {
        lo = std::min(lo, values[i]);
        hi = std::max(hi, values[i]);
      }
      const double split = 0.5 * (lo + hi);
      for (size_t i : idxs) {
        codes[i] = codes[i] * 2 + (values[i] >= split ? 1 : 0);
      }
    }
  }
  return codes;
}

// Mirrors any_precision_llm.py's own _dequantize_by_bin_mean: reconstructs
// every element as the mean of its own bin's own original values (a proper
// vector-quantizer reconstruction, not a linear/uniform-grid one -- see
// that module's docstring for why a single affine fit does not have the
// "monotonically improves with more bits" property this scheme relies on).
inline std::vector<double> DequantizeByBinMean(
    const std::vector<double>& values, const std::vector<int64_t>& codes) {
  std::unordered_map<int64_t, std::pair<double, int64_t>> sums;
  for (size_t i = 0; i < values.size(); ++i) {
    auto& entry = sums[codes[i]];
    entry.first += values[i];
    entry.second += 1;
  }
  std::vector<double> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const auto& entry = sums[codes[i]];
    out[i] = entry.first / static_cast<double>(entry.second);
  }
  return out;
}

}  // namespace any_precision_llm_detail

struct AnyPrecisionLlm final : public PredicateBasedPass {
  explicit AnyPrecisionLlm()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "any_precision_llm"; }

  bool patternMatchPredicate(Node* n) override {
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    return w_t != nullptr && w_t->elem_type() == TensorProto_DataType_FLOAT &&
           w_t->sizes().size() == 2;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // This pass only ever replaces `n`'s weight input, never `n` itself.
    destroy_current = NodeDestroyType::DestroyZero;
    MatMulLikeInfo info;
    if (!MatchMatMulLike(n, info)) {
      return false;
    }
    const Tensor* w_t = FetchConstantTensor(info.w);
    if (w_t == nullptr || w_t->elem_type() != TensorProto_DataType_FLOAT ||
        w_t->sizes().size() != 2) {
      return false;
    }

    const int64_t bits = AnyPrecisionLlmBits();
    const int64_t max_bits = AnyPrecisionLlmMaxBits();
    const int64_t block_size = std::max<int64_t>(1, AnyPrecisionLlmBlockSize());

    const auto& sizes = w_t->sizes();
    const int64_t dim0 = sizes[0];
    const int64_t dim1 = sizes[1];
    // Logical [N, K] (output-channel-first) view, matching
    // any_precision_llm.py's own `w_nk` convention.
    const int64_t N = info.weight_transposed ? dim0 : dim1;
    const int64_t K = info.weight_transposed ? dim1 : dim0;

    const std::vector<float> data = ReadFloatMatrix(*w_t);
    // Element (row, k) of the logical [N, K] weight, regardless of storage
    // layout: `weight_transposed` means W is already stored [N, K]
    // (dim0=N, dim1=K); otherwise W is [K, N] (dim0=K, dim1=N).
    auto at = [&](int64_t row, int64_t k) -> double {
      return info.weight_transposed ? static_cast<double>(data[row * dim1 + k])
                                    : static_cast<double>(data[k * dim1 + row]);
    };

    using any_precision_llm_detail::DequantizeByBinMean;
    using any_precision_llm_detail::NestedBitplaneCodes;

    std::vector<float> out_data(static_cast<size_t>(dim0 * dim1));
    auto set_at = [&](int64_t row, int64_t k, float v) {
      if (info.weight_transposed) {
        out_data[static_cast<size_t>(row * dim1 + k)] = v;
      } else {
        out_data[static_cast<size_t>(k * dim1 + row)] = v;
      }
    };

    for (int64_t row = 0; row < N; ++row) {
      for (int64_t start = 0; start < K; start += block_size) {
        const int64_t end = std::min(start + block_size, K);
        std::vector<double> block(static_cast<size_t>(end - start));
        for (int64_t k = start; k < end; ++k) {
          block[static_cast<size_t>(k - start)] = at(row, k);
        }
        const std::vector<int64_t> max_codes =
            NestedBitplaneCodes(block, max_bits);
        std::vector<int64_t> codes_b(block.size());
        for (size_t i = 0; i < block.size(); ++i) {
          codes_b[i] = max_codes[i] >> (max_bits - bits);
        }
        const std::vector<double> recon = DequantizeByBinMean(block, codes_b);
        for (int64_t k = start; k < end; ++k) {
          set_at(row, k,
                 static_cast<float>(recon[static_cast<size_t>(k - start)]));
        }
      }
    }

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    w_out.sizes() = sizes;
    w_out.floats() = std::move(out_data);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
