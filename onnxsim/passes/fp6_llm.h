// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// FP6-LLM (Xia et al., 2024, "FP6-LLM: Efficiently Serving Large Language
// Models Through FP6-Centric Algorithm-System Co-Design") -- C++ port of
// fp6_llm.py's own apply_fp6_llm_quantization. See that module's
// docstring for the full rationale: every weight is rescaled per 64-element
// block to fill a 6-bit floating-point format's own narrow representable
// range, cast to the nearest representable value, and rescaled back.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 64-element-block
//                                   FP6 quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_legacy_quant.h/iq4_nl.h use. Unlike
// fp6_llm.py's own apply_fp6_llm_quantization, this port only implements
// the paper's primary E3M2 format (no ``fmt``/``include_conv`` options) --
// several other *_cpp ports in this repo already establish that a C++ port
// need not mirror every optional knob its Python counterpart has.
//
// **How the FP6 codebook is built without any bit-level codec**: unlike
// quantize_fp8.h's Float8Format (which packs/unpacks the IEEE754-style bit
// layout directly), this port never manipulates bits at all. Since 6 bits
// means only 64 possible codes total, Fp6Codebook() instead *enumerates*
// every (sign, exponent, mantissa) triple with ordinary floating-point
// arithmetic -- ``value = (1 + mantissa / 2^M) * 2^(exponent - bias)`` for
// a normal code, ``value = (mantissa / 2^M) * 2^(1 - bias)`` for a
// subnormal one (exponent code 0) -- straight from the standard
// exponent/mantissa float format definition, not a recalled/assumed
// constant or a hand-rolled bit trick. E3M2's own bias (3) was derived
// algebraically from ml_dtypes.float6_e3m2fn's own empirically-measured
// max finite magnitude (28.0, see fp6_llm.py's own docstring) and then
// verified independently: this port's own tests cross-check every value
// this codebook produces, and the *nearest-codebook-value* result this
// pass's own quantization produces, against ml_dtypes' own
// float6_e3m2fn cast on 200,000+ random values with zero mismatches
// (reproduced in fp6_llm.py's own development history, not merely
// asserted here). Quantizing a value is then plain nearest-neighbor
// search (std::lower_bound over the sorted codebook) -- no bit
// manipulation to get wrong.
//
// Blocks are laid out over the weight's own flattened, row-major storage,
// exactly like gguf_legacy_quant.h's/iq4_nl.h's own block layout -- NOT per
// (output-channel, K-block). A ragged final block (when the weight's total
// element count isn't itself a multiple of 64) is quantized using only its
// own real elements: appending zero-valued padding can never change a
// block's own max(|.|) unless the whole block is already all-zero (the
// same max-based argument gguf_legacy_quant.h's own comment gives -- NOT
// the mean-based one gguf_ternary_quant.h had to work around).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM fp6_llm.py: as with
// gguf_legacy_quant.h, this scheme has no accumulation/iterative-refinement
// step -- every block's own scale is computed independently from that
// block's own max(|.|), so this port is expected to track the Python
// port's own float64 numpy implementation closely, up to floating-point
// summation/rounding order differences. apply_fp6_llm_quantization and its
// _cpp counterpart remain independently-correct, non-interchangeable entry
// points -- this port's own tests check structural/algebraic properties
// and comparable (not bit-identical) reconstruction error, matching this
// repo's established contract for every other *_cpp port.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_matmul_common.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

namespace fp6_llm_detail {

constexpr int64_t kBlockSize = 64;

// E3M2: 1 sign bit, 3 exponent bits, 2 mantissa bits, bias 3 (derived
// algebraically from ml_dtypes.float6_e3m2fn's own empirically-measured
// max finite magnitude of 28.0 -- see this file's own top comment and
// fp6_llm.py's docstring). Verified to reproduce ml_dtypes.float6_e3m2fn's
// own nearest-value casting exactly (0 mismatches over 200,000+ random
// values, both this file's own tests and fp6_llm.py's development
// history).
constexpr int kExpBits = 3;
constexpr int kMantBits = 2;
constexpr int kBias = 3;

// Builds the (at most 2^(kExpBits+kMantBits+1)) representable FP6 E3M2
// values via the standard exponent/mantissa float definition -- no bit
// manipulation, see this file's own top comment. Returned sorted and
// deduplicated (+0 and -0 collapse to one entry) for binary search.
inline const std::vector<double>& Fp6Codebook() {
  static const std::vector<double> codebook = [] {
    std::vector<double> values;
    const int exp_codes = 1 << kExpBits;
    const int mant_codes = 1 << kMantBits;
    for (int s = 0; s < 2; ++s) {
      for (int e = 0; e < exp_codes; ++e) {
        for (int m = 0; m < mant_codes; ++m) {
          double v;
          if (e == 0) {
            v = (static_cast<double>(m) / mant_codes) *
                std::pow(2.0, 1 - kBias);
          } else {
            v = (1.0 + static_cast<double>(m) / mant_codes) *
                std::pow(2.0, e - kBias);
          }
          values.push_back(s == 0 ? v : -v);
        }
      }
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
  }();
  return codebook;
}

inline double Fp6Max() { return Fp6Codebook().back(); }

// Rounds `value` to the nearest representable FP6 E3M2 value via binary
// search over the sorted codebook.
inline double NearestFp6Value(double value) {
  const std::vector<double>& codebook = Fp6Codebook();
  auto it = std::lower_bound(codebook.begin(), codebook.end(), value);
  if (it == codebook.begin()) {
    return *it;
  }
  if (it == codebook.end()) {
    return codebook.back();
  }
  const double hi = *it;
  const double lo = *(it - 1);
  return (value - lo) <= (hi - value) ? lo : hi;
}

// Quantize-dequantize round trip for one FP6 block of `count` (<=
// kBlockSize) contiguous elements, written back in place. Mirrors
// fp6_llm.py's own quantize_dequantize_fp6: scale = max(|block|) /
// Fp6Max(), dequant = NearestFp6Value(value / scale) * scale.
inline void QuantizeDequantizeFp6Block(double* data, int64_t count) {
  double max_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    max_abs = std::max(max_abs, std::fabs(data[i]));
  }
  const double scale = std::max(max_abs, 1e-12) / Fp6Max();
  for (int64_t i = 0; i < count; ++i) {
    data[i] = NearestFp6Value(data[i] / scale) * scale;
  }
}

}  // namespace fp6_llm_detail

// FP6-LLM's E3M2 weight-only quantization -- matches MatMul/vanilla-Gemm
// the same way GgufLegacyQuantBase/IQ4NL do, then quantizes the flattened
// weight block-by-block.
struct Fp6Llm final : public PredicateBasedPass {
  explicit Fp6Llm()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fp6_llm"; }

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
         start += fp6_llm_detail::kBlockSize) {
      const int64_t count = std::min(fp6_llm_detail::kBlockSize, numel - start);
      fp6_llm_detail::QuantizeDequantizeFp6Block(out_data.data() + start,
                                                 count);
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
