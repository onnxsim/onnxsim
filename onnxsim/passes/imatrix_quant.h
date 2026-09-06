// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// llama.cpp's "importance matrix" (imatrix) applied to this repository's own
// plain block-wise INT4 weight quantizer -- C++ port of the pure numeric
// mechanism in onnxsim.imatrix_quant.quantize_dequantize_int4_imatrix (see
// onnxsim/imatrix_quant.py's own module docstring for the full rationale:
// why mean-square activation is a sound per-channel importance proxy, and
// why this is a weighted grid-search extension of this repository's own
// already-verified plain block quantizer rather than a new GGUF block
// format). imatrix_quant_entry.h/.cpp is the calibration-gathering and
// graph-rewriting side that calls into this header; this header holds only
// the pure numeric quantizer, kept dependency-free and easy to test in
// isolation, the same split quantize_matmul_common.h's own
// TryQuantizeWeightBlockwiseInt4InPlace/ReadWeightNK keep from their own
// callers.
//
// QuantizeDequantizeInt4Imatrix below is a direct scalar-loop translation
// of quantize_dequantize_int4_imatrix's own vectorized numpy
// implementation -- same base scale (`max(|block|) / 7`, i.e. exactly
// TryQuantizeWeightBlockwiseInt4InPlace's own plain per-block scale with
// `clip_ratio = 1`), same candidate-scale grid
// (`base_scale * linspace(scale_lo, scale_hi, num_scale_candidates)`), same
// per-(row, block) selection rule (minimize
// `sum(importance_j * (w_j - dequant_j) ** 2)`). The Python side verifies
// this exact algorithm against a scalar-loop reference and against the
// plain (unweighted) baseline under uniform importance (see
// tests/test_imatrix_quant.py); this port needs no separate numerical
// derivation of its own since it is that same scalar loop, not a
// re-derivation, and tests/test_imatrix_quant_cpp.py cross-checks this
// C++ output against onnxsim.imatrix_quant.quantize_dequantize_int4_imatrix
// directly.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace onnxsim_passes {

// `w_nk`: row-major [n, k], output channel first. `importance_k`: length
// `k`, per-input-channel importance (mean-square activation magnitude).
// `k` must be a multiple of `block_size` (checked by the caller -- see
// imatrix_quant_entry.cpp's own candidate matching). `num_scale_candidates`
// candidate scale factors are evaluated per (output row, block), evenly
// spaced across [`scale_lo`, `scale_hi`] (mirrors `np.linspace`); with
// `num_scale_candidates == 1` only `scale_lo` is tried.
//
// :returns: `w_nk`'s importance-weighted INT4 quantize-dequantize round
//         trip, row-major [n, k], float32.
inline std::vector<float> QuantizeDequantizeInt4Imatrix(
    const std::vector<float>& w_nk, int64_t n, int64_t k,
    const std::vector<double>& importance_k, int64_t block_size,
    int64_t num_scale_candidates, double scale_lo, double scale_hi) {
  const int64_t num_blocks = k / block_size;
  std::vector<float> out(static_cast<size_t>(n * k));
  std::vector<double> block(static_cast<size_t>(block_size));

  for (int64_t r = 0; r < n; ++r) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      const int64_t base = r * k + b * block_size;
      double max_abs = 0.0;
      for (int64_t j = 0; j < block_size; ++j) {
        const double v =
            static_cast<double>(w_nk[static_cast<size_t>(base + j)]);
        block[static_cast<size_t>(j)] = v;
        max_abs = std::max(max_abs, std::fabs(v));
      }
      const double base_scale = std::max(max_abs, 1e-12) / 7.0;

      double best_scale = base_scale;
      double best_err = std::numeric_limits<double>::infinity();
      for (int64_t c = 0; c < num_scale_candidates; ++c) {
        const double factor =
            num_scale_candidates == 1
                ? scale_lo
                : scale_lo + (scale_hi - scale_lo) * static_cast<double>(c) /
                                 static_cast<double>(num_scale_candidates - 1);
        const double trial_scale = std::max(base_scale * factor, 1e-12);
        double err = 0.0;
        for (int64_t j = 0; j < block_size; ++j) {
          double code = std::round(block[static_cast<size_t>(j)] / trial_scale);
          code = std::clamp(code, -7.0, 7.0);
          const double diff =
              block[static_cast<size_t>(j)] - code * trial_scale;
          err += importance_k[static_cast<size_t>(b * block_size + j)] * diff *
                 diff;
        }
        if (err < best_err) {
          best_err = err;
          best_scale = trial_scale;
        }
      }

      for (int64_t j = 0; j < block_size; ++j) {
        double code = std::round(block[static_cast<size_t>(j)] / best_scale);
        code = std::clamp(code, -7.0, 7.0);
        out[static_cast<size_t>(base + j)] =
            static_cast<float>(code * best_scale);
      }
    }
  }
  return out;
}

}  // namespace onnxsim_passes
