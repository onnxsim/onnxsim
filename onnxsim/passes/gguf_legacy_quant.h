// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// llama.cpp's legacy GGUF "Q4_0"/"Q4_1" block quant formats -- C++ port of
// gguf_legacy_quant.py's own apply_gguf_q4_0_quantization/
// apply_gguf_q4_1_quantization. See that module's docstring for the full
// rationale and the exact reconstruction formulas this port targets
// (transcribed from, and kept consistent with, this repo's own verified
// onnxsim/ggml_legacy_quant.h decoder): Q4_0 is symmetric with no separate
// min (`dequant = (code - 8) * d`, code in [0, 15]); Q4_1 is asymmetric
// with an explicit per-block min (`dequant = code * d + m`, code in
// [0, 15]). Both use one plain 32-element block -- no super-block/
// sub-block requantization (unlike gguf_kquant.py's Q4_K) and no fixed
// codebook (unlike iq4_nl.py's IQ4_NL).
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 32-element-block
//                                   Q4_0/Q4_1 quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope any_precision_llm.h/iq4_nl.h use. Unlike
// gguf_legacy_quant.py's own apply_gguf_q4_0_quantization/
// apply_gguf_q4_1_quantization, this port does not add an include_conv
// option -- several other *_cpp ports in this repo already establish that a
// C++ port need not mirror every optional knob its Python counterpart has.
//
// Blocks are laid out over the weight's own flattened, row-major storage,
// exactly like iq4_nl.h's own block layout -- NOT per (output-channel,
// K-block). A ragged final block (when the weight's total element count
// isn't itself a multiple of 32) is quantized using only its own real
// elements, mathematically identical to the Python port's own
// zero-pad-then-discard approach for the same reason iq4_nl.h's own comment
// gives: appending zero-valued padding can never change a block's own
// max(|.|)/min/max statistics unless the whole block is already all-zero.
//
// ACCEPTED, PERMANENT DIVERGENCE FROM gguf_legacy_quant.py: as with
// iq4_nl.h, this scheme has no accumulation/iterative-refinement step --
// every block's own (scale[, min]) is computed independently from that
// block's own min/max, so this port is expected to track the Python port's
// own float64 numpy implementation closely, up to float16 round-trip and
// floating-point summation order differences. apply_gguf_q4_0_quantization/
// _q4_1_quantization and their _cpp counterparts remain independently-
// correct, non-interchangeable entry points -- this port's own tests check
// structural/algebraic properties and comparable (not bit-identical)
// reconstruction error, matching this repo's established contract for
// every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml_kquant.h"
#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_fp16.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace gguf_legacy_quant_detail {

constexpr int64_t kBlockSize = 32;
constexpr int64_t kMaxCode = 15;
constexpr int64_t kQ4_0Bias = 8;

// Round-trips a float32 value through ggml_half (IEEE754 binary16),
// reproducing the same precision loss real Q4_0/Q4_1 scale/min fields
// carry -- mirrors gguf_legacy_quant.py's own
// `.astype(np.float16).astype(np.float64)` step. Reuses this repo's own
// already-verified fp16 codec (quantize_fp16.h's FloatToFloat16Bits for
// the narrowing direction, ggml_kquant.h's Float16BitsToFloat32 -- already
// used to decode this exact family of formats -- for widening back) rather
// than a fresh hand-rolled bit-manipulation routine.
inline double RoundTripFloat16(double value) {
  return static_cast<double>(::onnxsim::tensor_pool::gguf::Float16BitsToFloat32(
      FloatToFloat16Bits(static_cast<float>(value))));
}

// Quantize-dequantize round trip for one Q4_0 block of `count` (<=
// kBlockSize) contiguous elements, written back in place. Mirrors
// gguf_legacy_quant.py's own quantize_dequantize_q4_0: d = max(|block|) / 8
// (round-tripped through float16), code = round(value / d) + 8 clamped to
// [0, 15], dequant = (code - 8) * d.
inline void QuantizeDequantizeQ4_0Block(double* data, int64_t count) {
  double max_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  const double d = RoundTripFloat16(std::max(max_abs, 1e-12) / kQ4_0Bias);
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round(data[i] / d) + static_cast<double>(kQ4_0Bias);
    code = std::min(std::max(code, 0.0), static_cast<double>(kMaxCode));
    data[i] = (code - static_cast<double>(kQ4_0Bias)) * d;
  }
}

// Quantize-dequantize round trip for one Q4_1 block, written back in
// place. Mirrors gguf_legacy_quant.py's own quantize_dequantize_q4_1:
// m = min(block), d = (max(block) - m) / 15 (both round-tripped through
// float16), code = round((value - m) / d) clamped to [0, 15],
// dequant = code * d + m.
inline void QuantizeDequantizeQ4_1Block(double* data, int64_t count) {
  double lo = data[0];
  double hi = data[0];
  for (int64_t i = 1; i < count; ++i) {
    lo = std::min(lo, data[i]);
    hi = std::max(hi, data[i]);
  }
  const double m = RoundTripFloat16(lo);
  const double d = RoundTripFloat16(std::max(hi - lo, 1e-12) / kMaxCode);
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round((data[i] - m) / d);
    code = std::min(std::max(code, 0.0), static_cast<double>(kMaxCode));
    data[i] = code * d + m;
  }
}

}  // namespace gguf_legacy_quant_detail

// Shared implementation for both GgufQ4_0/GgufQ4_1 -- matches
// MatMul/vanilla-Gemm the same way IQ4NL does, then quantizes the flattened
// weight block-by-block via whichever `BlockFn` the subclass provides.
template <void (*BlockFn)(double*, int64_t), const char* kName>
struct GgufLegacyQuantBase : public PredicateBasedPass {
  explicit GgufLegacyQuantBase()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return kName; }

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
    for (int64_t start = 0; start < numel;
         start += gguf_legacy_quant_detail::kBlockSize) {
      const int64_t count =
          std::min(gguf_legacy_quant_detail::kBlockSize, numel - start);
      BlockFn(out_data.data() + start, count);
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

inline constexpr char kGgufQ4_0Name[] = "gguf_q4_0";
inline constexpr char kGgufQ4_1Name[] = "gguf_q4_1";

using GgufQ4_0 =
    GgufLegacyQuantBase<gguf_legacy_quant_detail::QuantizeDequantizeQ4_0Block,
                        kGgufQ4_0Name>;
using GgufQ4_1 =
    GgufLegacyQuantBase<gguf_legacy_quant_detail::QuantizeDequantizeQ4_1Block,
                        kGgufQ4_1Name>;

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
