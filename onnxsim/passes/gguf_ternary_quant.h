// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// BitNet b1.58's published absmean ternary weight quantization, as shipped
// by llama.cpp's GGUF TQ1_0/TQ2_0 tensor types -- C++ port of
// gguf_ternary_quant.py's own apply_gguf_ternary_quantization. See that
// module's docstring for the full rationale: every weight is restricted to
// one of {-1, 0, +1} times one shared per-256-element-block scale
// ``d = mean(|block|)`` (round-tripped through float16, matching
// llama.cpp's own storage), ``code = round(clip(value / d, -1, 1))``,
// ``dequant = code * d``.
//
// Before (illustrated for MatMul; a "vanilla" Gemm -- transA=0, alpha=1,
// beta=1 -- is handled the same way, its bias C left untouched):
//   Y = MatMul(X, W) [+ bias]      W constant, [K, N], float32
// After:
//   Y = MatMul(X, W') [+ bias]     W' same shape/dtype as W, every element
//                                   replaced by its own 256-element-block
//                                   ternary quantize-dequantize round trip
//
// Only the common, unambiguous shape is handled: a MatMul, or a Gemm with
// transA=0, alpha=1 and beta=1, whose weight (input 1) is a constant 2-D
// float32 tensor -- the same scope gguf_legacy_quant.h/iq4_nl.h use. Unlike
// gguf_ternary_quant.py's own apply_gguf_ternary_quantization, this port
// does not add an include_conv option -- several other *_cpp ports in this
// repo already establish that a C++ port need not mirror every optional
// knob its Python counterpart has.
//
// Blocks are laid out over the weight's own flattened, row-major storage,
// exactly like gguf_legacy_quant.h's/iq4_nl.h's own block layout -- NOT per
// (output-channel, K-block). A ragged final block (when the weight's total
// element count isn't itself a multiple of 256) is quantized using only
// its own real elements: QuantizeDequantizeTernaryBlock always divides by
// its own `count` argument, so a ragged block's mean is computed over just
// its real elements -- matching gguf_ternary_quant.py's own real-count
// division, NOT the zero-pad-then-average-over-the-full-block-size
// approach gguf_legacy_quant.h's max/min-based scale can get away with
// (a *mean*-based scale is not invariant to zero-padding: averaging over
// the padded size would silently dilute it -- see gguf_ternary_quant.py's
// own docstring for the same argument in more detail).
//
// ACCEPTED, PERMANENT DIVERGENCE FROM gguf_ternary_quant.py: as with
// gguf_legacy_quant.h, this scheme has no accumulation/iterative-refinement
// step -- every block's own scale is computed independently from that
// block's own mean(|.|), so this port is expected to track the Python
// port's own float64 numpy implementation closely, up to float16
// round-trip and floating-point summation order differences.
// apply_gguf_ternary_quantization and its _cpp counterpart remain
// independently-correct, non-interchangeable entry points -- this port's
// own tests check structural/algebraic properties and comparable (not
// bit-identical) reconstruction error, matching this repo's established
// contract for every other *_cpp port.

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

namespace gguf_ternary_quant_detail {

constexpr int64_t kBlockSize =
    256;  // llama.cpp's own TQ1_0/TQ2_0 super-block size

// Round-trips a float32 value through ggml_half (IEEE754 binary16),
// reproducing the same precision loss real TQ1_0/TQ2_0 scale fields carry
// -- mirrors gguf_ternary_quant.py's own
// `.astype(np.float16).astype(np.float64)` step. Reuses this repo's own
// already-verified fp16 codec (quantize_fp16.h's FloatToFloat16Bits for
// the narrowing direction, ggml_kquant.h's Float16BitsToFloat32 for
// widening back) rather than a fresh hand-rolled bit-manipulation routine
// -- the same reuse gguf_legacy_quant.h's own RoundTripFloat16 makes.
inline double RoundTripFloat16(double value) {
  return static_cast<double>(::onnxsim::tensor_pool::gguf::Float16BitsToFloat32(
      FloatToFloat16Bits(static_cast<float>(value))));
}

// Quantize-dequantize round trip for one ternary block of `count` (<=
// kBlockSize) contiguous elements, written back in place. Mirrors
// gguf_ternary_quant.py's own quantize_dequantize_ternary: d = mean(|block|)
// (round-tripped through float16), code = round(clip(value / d, -1, 1)) in
// {-1, 0, 1}, dequant = code * d.
inline void QuantizeDequantizeTernaryBlock(double* data, int64_t count) {
  double sum_abs = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    sum_abs += std::fabs(data[i]);
  }
  const double d =
      RoundTripFloat16(std::max(sum_abs / static_cast<double>(count), 1e-12));
  for (int64_t i = 0; i < count; ++i) {
    double code = std::round(data[i] / d);
    code = std::min(std::max(code, -1.0), 1.0);
    data[i] = code * d;
  }
}

}  // namespace gguf_ternary_quant_detail

// llama.cpp's BitNet b1.58 TQ1_0/TQ2_0 ternary weight quantization --
// matches MatMul/vanilla-Gemm the same way GgufLegacyQuantBase/IQ4NL do,
// then quantizes the flattened weight block-by-block.
struct GgufTernaryQuant final : public PredicateBasedPass {
  explicit GgufTernaryQuant()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "gguf_ternary_quant"; }

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
         start += gguf_ternary_quant_detail::kBlockSize) {
      const int64_t count =
          std::min(gguf_ternary_quant_detail::kBlockSize, numel - start);
      gguf_ternary_quant_detail::QuantizeDequantizeTernaryBlock(
          out_data.data() + start, count);
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
