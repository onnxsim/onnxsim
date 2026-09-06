// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// llama.cpp's IQ4_NL ("I-quant", fixed non-uniform 4-bit codebook) -- C++
// port of iq4_nl.py's own apply_iq4_nl_quantization. See that module's
// docstring for the full rationale and, importantly, this format's own
// codebook provenance: unlike gguf_kquant.py's Q4_K decoder (transcribed
// from and cross-checked against onnxsim/ggml_kquant.h, itself transcribed
// from GGML's own ggml-quants.c), no verified source for llama.cpp's real
// IQ4_NL codebook (`kvalues_iq4nl`) was available in this repository or
// from a runnable reference alongside this change -- this repo's own
// gguf_reconstruct.py/ggml_kquant.h decode Q4_K and the rest of the
// K-quant family, but no I-quant format, and a whole-tree grep for
// "IQ4_NL"/"kvalues_iq4nl" turns up nothing outside this change itself. Per
// this repo's norm against shipping recalled/assumed numeric constants
// without verifying them computationally first, the codebook below is
// therefore NOT llama.cpp's own table -- it is transcribed verbatim from
// iq4_nl.py's own IQ4_NL_CODEBOOK, which that module derives
// computationally (a generalized Lloyd-Max / 1-D k-means quantizer run to
// convergence against a standard normal distribution via deterministic
// numerical quadrature -- see iq4_nl.py's own top-of-file docstring and
// its `_lloyd_max_gaussian_codebook`). tests/test_iq4_nl_cpp.py checks the
// array below against onnxsim.IQ4_NL_CODEBOOK directly, so the two stay in
// sync.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 32-element-block
//                                   IQ4_NL quantize-dequantize round trip:
//                                   dequant = codebook[nearest_code] *
//                                   scale, scale = max(|block|) /
//                                   max(|codebook|)
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope any_precision_llm.h uses. Unlike
// iq4_nl.py's own apply_iq4_nl_quantization, this port does not add an
// include_conv option: several other *_cpp ports in this repo already
// establish that a C++ port need not mirror every optional knob its
// Python counterpart has (e.g. apply_any_precision_llm_cpp has no Conv
// support either).
//
// Blocks are laid out over the weight's own flattened, row-major storage
// (whatever 2-D shape/layout it already has), exactly like iq4_nl.py's own
// quantize_dequantize_iq4_nl -- NOT per (output-channel, K-block) the way
// any_precision_llm.h's nested-bitplane scheme is; IQ4_NL's own block
// structure has no notion of output channel at all, only "32 consecutive
// elements of however the tensor is stored". A ragged final block (when
// the weight's total element count is not itself a multiple of 32) is
// quantized using only its own real elements, which is mathematically
// identical to iq4_nl.py's own zero-pad-then-discard approach: appending
// zero-valued padding elements to a block can never change that block's
// own max(|.|) scale (a zero is never the largest-magnitude element unless
// the whole block is already all-zero, in which case the scale floors to
// the same epsilon either way).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM iq4_nl.py: unlike quarot.h's rotation
// matmul or any_precision_llm.h's per-bin bisection, this scheme has no
// accumulation or iterative-refinement step at all -- every element's own
// code is chosen independently, from a single per-block max-abs scale and
// a 16-way nearest-neighbor scan over a fixed codebook -- so in practice
// this port is expected to track iq4_nl.py's own float64 numpy
// implementation far more closely than any other *_cpp port in this repo
// (up to the two languages' argmin implementations breaking an exact
// codebook-distance tie differently, an occurrence this port has not
// observed in testing but does not rule out as a guarantee). As with every
// other pair in this repo, apply_iq4_nl_quantization and
// apply_iq4_nl_quantization_cpp remain independently-correct,
// non-interchangeable entry points -- this port's own tests
// (tests/test_iq4_nl_cpp.py) check structural/algebraic properties and
// comparable (not required to be bit-identical) reconstruction error
// against the Python port, matching quarot.h's/any_precision_llm.h's own
// established contract.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace iq4_nl_detail {

// Kept in sync with onnxsim/iq4_nl.py's own IQ4_NL_CODEBOOK -- see this
// file's own top-of-file comment for the derivation and honesty note.
inline const std::array<double, 16>& Codebook() {
  static const std::array<double, 16> codebook = {
      -1.0,
      -0.757290980404998,
      -0.5923241681050287,
      -0.45994900549399614,
      -0.34508882475608954,
      -0.24055736084608642,
      -0.1421631738251121,
      -0.04704566832784618,
      0.04704566832784618,
      0.1421631738251121,
      0.24055736084608642,
      0.34508882475608954,
      0.45994900549399614,
      0.5923241681050287,
      0.757290980404998,
      1.0,
  };
  return codebook;
}

constexpr int64_t kBlockSize = 32;

// Nearest codebook entry to `value` (an exact tie between two equidistant
// entries keeps the lower index, matching numpy.argmin's own
// first-occurrence tie-break over the same ascending-sorted codebook, which
// iq4_nl.py's own _quantize_dequantize_blocks relies on identically).
inline double NearestCodebookValue(double value) {
  const auto& codebook = Codebook();
  double best_dist = std::numeric_limits<double>::infinity();
  double best_value = codebook[0];
  for (double level : codebook) {
    const double dist = std::fabs(value - level);
    if (dist < best_dist) {
      best_dist = dist;
      best_value = level;
    }
  }
  return best_value;
}

// Quantize-dequantize round trip for one block of `count` (<= kBlockSize)
// contiguous elements starting at `data[0]`, written back in place. Mirrors
// iq4_nl.py's own _quantize_dequantize_blocks: scale = max(|block|) /
// max(|codebook|) (floored to avoid a divide-by-zero on an all-zero
// block), then every element snaps to whichever codebook entry (times that
// scale) is closest.
inline void QuantizeDequantizeBlock(double* data, int64_t count) {
  const auto& codebook = Codebook();
  double max_abs_codebook = 0.0;
  for (double level : codebook) {
    max_abs_codebook = std::max(max_abs_codebook, std::fabs(level));
  }
  double max_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  const double scale = std::max(max_abs, 1e-12) / max_abs_codebook;
  for (int64_t i = 0; i < count; ++i) {
    data[i] = NearestCodebookValue(data[i] / scale) * scale;
  }
}

}  // namespace iq4_nl_detail

struct IQ4NL final : public PredicateBasedPass {
  explicit IQ4NL()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "iq4_nl"; }

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

    const auto& sizes = w_t->sizes();
    const int64_t numel = sizes[0] * sizes[1];
    const std::vector<float> data = ReadFloatMatrix(*w_t);

    std::vector<double> out_data(data.begin(), data.end());
    for (int64_t start = 0; start < numel; start += iq4_nl_detail::kBlockSize) {
      const int64_t count = std::min(iq4_nl_detail::kBlockSize, numel - start);
      iq4_nl_detail::QuantizeDequantizeBlock(out_data.data() + start, count);
    }

    Tensor w_out;
    w_out.elem_type() = TensorProto_DataType_FLOAT;
    w_out.sizes() = sizes;
    std::vector<float> out_float(out_data.begin(), out_data.end());
    w_out.floats() = std::move(out_float);

    Value* w_out_v = graph.addInitializerAndCreateValue(w_out);
    n->replaceInput(1, w_out_v);
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
