// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Node-matching and weight-quantization helpers shared by
// dynamic_quantize_matmul.h and static_quantize_matmul.h: both passes target
// the same MatMul / "vanilla" Gemm shape and quantize the constant weight to
// INT8 per output channel from its static values; they differ only in how
// the *activation* is quantized (a runtime-computed DynamicQuantizeLinear vs.
// a calibrated, fixed QuantizeLinear/DequantizeLinear pair) and, downstream
// of that, in the rest of the graph they build. ReadInt8Matrix is also used
// by defuse_matmul_integer_to_float.h, which reads back a weight one of
// these passes quantized.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// The largest reduction depth (K, the number of terms MatMulInteger's int32
// accumulator sums for one output element) for which that accumulator cannot
// overflow. Both operands' worst case is fixed by the surrounding
// quantization scheme, not by the actual weight data:
// QuantizeWeightPerChannelKN / QuantizeWeightPerChannelInPlace always scale a
// channel so its largest-magnitude element quantizes to exactly +-127, and
// DynamicQuantizeLinear / QuantizeLinear produce uint8 (max 255). So the
// worst possible per-term product is 127 * 255, and K of them can sum to at
// most K * 127 * 255 -- once that exceeds INT32_MAX the accumulator can wrap
// around, silently corrupting the result. This bound is exact and
// independent of the weight's actual values.
inline int64_t MaxSafeInt32ReductionDepth() {
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
         (127 * 255);
}

// Whether a reduction depth of `k` is safe from int32 accumulator overflow
// under the worst-case bound above (see MaxSafeInt32ReductionDepth).
inline bool IsSafeInt32ReductionDepth(int64_t k) {
  return k <= MaxSafeInt32ReductionDepth();
}

// The pieces of a MatMul/vanilla-Gemm node these passes care about.
struct MatMulLikeInfo {
  Value* x = nullptr;     // activation (not required to be constant)
  Value* w = nullptr;     // weight; must be a constant 2-D float32 tensor
  Value* bias = nullptr;  // Gemm's optional C input; nullptr for MatMul
  bool weight_transposed = false;  // Gemm transB == 1: W stored as [N, K]
};

// Recognizes a MatMul, or a Gemm with transA=0 and (when it has a bias)
// beta=1 and always alpha=1, filling `info`. Attributes are read with ONNX's
// documented defaults when absent.
inline bool MatchMatMulLike(Node* n, MatMulLikeInfo& info) {
  if (n->kind() == kMatMul) {
    if (n->inputs().size() != 2) {
      return false;
    }
    info.x = n->inputs()[0];
    info.w = n->inputs()[1];
    return true;
  }
  if (n->kind() == kGemm) {
    const size_t num_inputs = n->inputs().size();
    if (num_inputs != 2 && num_inputs != 3) {
      return false;
    }
    const int64_t transA = GetValueFromAttrWithDefault(n, ktransA, int64_t(0));
    const int64_t transB = GetValueFromAttrWithDefault(n, ktransB, int64_t(0));
    const double alpha = GetValueFromAttrWithDefault(n, kalpha, 1.0);
    const double beta = GetValueFromAttrWithDefault(n, kbeta, 1.0);
    if (transA != 0 || alpha != 1.0) {
      return false;
    }
    if (num_inputs == 3) {
      if (beta != 1.0) {
        return false;
      }
      info.bias = n->inputs()[2];
    }
    info.x = n->inputs()[0];
    info.w = n->inputs()[1];
    info.weight_transposed = transB != 0;
    return true;
  }
  return false;
}

// Reads `w_t` (a 2-D float32 constant) into a flat row-major, host-byte-order
// float buffer, regardless of whether it is stored as raw bytes (which are
// little-endian on the wire regardless of host -- see endian_read.h) or a
// typed float array (already host-order, decoded by protobuf itself).
inline std::vector<float> ReadFloatMatrix(const Tensor& w_t) {
  const auto& sizes = w_t.sizes();
  const int64_t numel = sizes[0] * sizes[1];
  RequireMaterializedValues(
      w_t.name(),
      w_t.is_raw_data() ? static_cast<int64_t>(w_t.raw().size() / sizeof(float))
                        : static_cast<int64_t>(w_t.floats().size()),
      numel);
  if (w_t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(w_t.data<float>(), numel);
  }
  return w_t.floats();
}

// Reads `q_t` (a 2-D INT8 constant) into a flat row-major vector<int8_t>,
// regardless of whether it is stored as raw bytes (int8_t has no
// byte-order concerns, but this still goes through ReadRawDataHostOrder for
// consistency with every other raw_data read site -- see endian_read.h) or
// ONNX's typed field for sub-32-bit integer types, which is int32_data (a
// signed 8-bit code always round-trips through it unchanged).
inline std::vector<int8_t> ReadInt8Matrix(const Tensor& q_t) {
  const auto& sizes = q_t.sizes();
  const int64_t numel = sizes[0] * sizes[1];
  RequireMaterializedValues(
      q_t.name(),
      q_t.is_raw_data() ? static_cast<int64_t>(q_t.raw().size())
                        : static_cast<int64_t>(q_t.int32s().size()),
      numel);
  if (q_t.is_raw_data()) {
    return ReadRawDataHostOrder<int8_t>(
        reinterpret_cast<const int8_t*>(q_t.raw().data()), numel);
  }
  const std::vector<int32_t>& codes = q_t.int32s();
  std::vector<int8_t> out(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    out[static_cast<size_t>(i)] =
        static_cast<int8_t>(codes[static_cast<size_t>(i)]);
  }
  return out;
}

// Quantizes `w_t` (a 2-D float32 constant, laid out as [N, K] when
// `transposed` else [K, N]) per output channel: `q_out` holds the weight in
// the non-transposed [K, N] layout MatMulInteger's B operand needs, and
// `scale_out`[j] is the symmetric INT8 scale for output channel j
// (max(|w[:, j]|) / 127, or 1.0 for an all-zero channel so no scale is 0).
// Used by the dynamic-quantization pass, whose replacement MatMulInteger
// always consumes B in [K, N] layout.
inline void QuantizeWeightPerChannelKN(const Tensor& w_t, bool transposed,
                                       Tensor& q_out, Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t K = transposed ? dim1 : dim0;
  const int64_t N = transposed ? dim0 : dim1;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  // element (k, n) of the logical [K, N] weight, regardless of storage
  // layout.
  auto at = [&](int64_t k, int64_t n) {
    return transposed ? data[n * K + k] : data[k * N + n];
  };

  std::vector<float> scale(static_cast<size_t>(N), 0.0f);
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      scale[n] = std::max(scale[n], std::fabs(at(k, n)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {K, N};
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int8_t> codes(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      const float q = std::round(at(k, n) / scale[n]);
      codes[k * N + n] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {N};
  scale_out.floats() = std::move(scale);
}

// Quantizes `w_t` (a 2-D float32 constant) per output channel *in its own
// layout* (no transpose): `channel_axis` (0 or 1) is the axis of `w_t` that
// indexes the output channel. `q_out` has the SAME shape as `w_t`, and
// `scale_out`[j] is the symmetric INT8 scale for channel j. Used by the
// static-quantization pass, whose DequantizeLinear replaces the weight input
// in place -- the surrounding MatMul/Gemm node (and therefore the layout it
// expects of its weight input) is left untouched, unlike the dynamic pass's
// MatMulInteger replacement.
inline void QuantizeWeightPerChannelInPlace(const Tensor& w_t,
                                            int64_t channel_axis, Tensor& q_out,
                                            Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t C = channel_axis == 0 ? dim0 : dim1;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  auto channel_of = [&](int64_t i, int64_t j) {
    return channel_axis == 0 ? i : j;
  };

  std::vector<float> scale(static_cast<size_t>(C), 0.0f);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const int64_t c = channel_of(i, j);
      scale[c] = std::max(scale[c], std::fabs(at(i, j)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {dim0, dim1};
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int8_t> codes(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const int64_t c = channel_of(i, j);
      const float q = std::round(at(i, j) / scale[c]);
      codes[i * dim1 + j] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C};
  scale_out.floats() = std::move(scale);
}

// An explicit per-channel INT8 zero-point (all zeros, shape [C]) for a
// symmetric-quantized weight's DequantizeLinear.
//
// The zero_point input is optional per the ONNX spec (and symmetric
// quantization is always 0), so the static-quantize passes used to omit it --
// but runtimes that fuse the QDQ pattern into an integer kernel require scale
// and zero_point to have the same shape: onnxruntime's QGemm fusion rejects
// the fused MatMul+Add outright ("zero point and scale of input b should have
// the same shape size"), and the VitisAI EP aborts while partitioning it.
// Spelling out the zeros keeps the numerics identical and both runtimes happy.
inline void MakeSymmetricInt8WeightZeroPoint(int64_t channels, Tensor& zp_out) {
  zp_out.elem_type() = TensorProto_DataType_INT8;
  zp_out.sizes() = {channels};
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  zp_out.set_raw_data(WriteRawDataLittleEndian(
      std::vector<int8_t>(static_cast<size_t>(channels), 0)));
}

// Same as QuantizeWeightPerChannelInPlace, but INT16 (max(|w[:, j]|) / 32767
// per channel) instead of INT8 -- used by weight_only_quantize_int16_matmul.h
// for the outlier-heavy channels INT8's coarser step (1/127 relative) resolves
// poorly (see docs/int16-quantization.md and precision_estimator.py's own
// max_outlier_ratio-driven recommendation, which names INT16 as the fix).
inline void QuantizeWeightPerChannelInPlaceInt16(const Tensor& w_t,
                                                 int64_t channel_axis,
                                                 Tensor& q_out,
                                                 Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t C = channel_axis == 0 ? dim0 : dim1;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  auto channel_of = [&](int64_t i, int64_t j) {
    return channel_axis == 0 ? i : j;
  };

  std::vector<float> scale(static_cast<size_t>(C), 0.0f);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const int64_t c = channel_of(i, j);
      scale[c] = std::max(scale[c], std::fabs(at(i, j)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 32767.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT16;
  q_out.sizes() = {dim0, dim1};
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int16_t> codes(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const int64_t c = channel_of(i, j);
      const float q = std::round(at(i, j) / scale[c]);
      codes[i * dim1 + j] =
          static_cast<int16_t>(std::clamp(q, -32767.0f, 32767.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C};
  scale_out.floats() = std::move(scale);
}

// Attempts a *ternary* decomposition of `w_t` (a 2-D float32 constant, laid
// out as [N, K] when `transposed` else [K, N]): each output column may use
// only one of {-s, 0, +s} for a single per-column scale `s` -- the weight
// representation BitNet b1.58 (https://github.com/microsoft/BitNet) and
// similar ternary-weight models use, which a generic ONNX export still
// stores as a dense float32 initializer. Returns false (leaving `q_out`/
// `scale_out` unspecified) when any element of any column is not within
// `rtol` of its column's {-s, 0, +s}, or when every column is entirely zero
// (carries no ternary signal, so there is nothing to gain by rewriting it).
// On success, `q_out` holds the weight in the non-transposed [K, N] layout
// MatMulInteger's B operand needs (int8, values in {-1, 0, 1}), and
// `scale_out`[j] is output channel j's scale (an all-zero column gets an
// arbitrary scale of 1.0, matching its all-zero codes).
inline bool TryQuantizeWeightTernaryKN(const Tensor& w_t, bool transposed,
                                       Tensor& q_out, Tensor& scale_out,
                                       float rtol = 1e-5f) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t K = transposed ? dim1 : dim0;
  const int64_t N = transposed ? dim0 : dim1;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t k, int64_t n) {
    return transposed ? data[n * K + k] : data[k * N + n];
  };

  std::vector<float> scale(static_cast<size_t>(N), 0.0f);
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      scale[n] = std::max(scale[n], std::fabs(at(k, n)));
    }
  }
  bool any_nonzero = false;
  for (const float s : scale) {
    any_nonzero |= s > 0.0f;
  }
  if (!any_nonzero) {
    return false;
  }
  for (float& s : scale) {
    if (s == 0.0f) {
      s = 1.0f;  // All-zero column: arbitrary scale, codes are all 0 anyway.
    }
  }

  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why. Built into a local vector first (rather than
  // q_out directly) so a not-ternary bail-out below leaves q_out completely
  // untouched, not just its int32_data half-populated.
  std::vector<int8_t> codes(static_cast<size_t>(K * N));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      const float normalized = at(k, n) / scale[n];
      const float code = std::round(normalized);
      if (std::fabs(code) > 1.0f || std::fabs(normalized - code) > rtol) {
        return false;  // Not ternary: bail before mutating the caller's q_out.
      }
      codes[k * N + n] = static_cast<int8_t>(code);
    }
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {K, N};
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {N};
  scale_out.floats() = std::move(scale);
  return true;
}

// Block-wise INT4 quantization of `w_t` (a 2-D float32 constant) *in its own
// layout* (no transpose), mirroring QuantizeWeightPerChannelInPlace's
// per-channel INT8 scheme but along the *reduction* axis instead of the
// output-channel one: `channel_axis` (0 or 1) is the output-channel axis, so
// the reduction axis (`K`, the one MatMulInteger/MatMul actually sums over)
// is the other one. `K` is split into `K / block_size` consecutive blocks,
// each with its own scale per output channel -- the scheme GPTQ/AWQ-style
// weight-only INT4 quantization and ONNX opset 21's blocked
// QuantizeLinear/DequantizeLinear both use (see block_size). Finer blocks
// trade more scale-tensor overhead for lower quantization error than a
// single per-channel scale.
//
// Returns false (nothing written) when `K` is not evenly divisible by
// `block_size` -- the common, unambiguous case only; a ragged last block is
// left to a future extension rather than handled approximately here.
// Otherwise `q_out` has the SAME shape as `w_t` (int4, values in [-7, 7]),
// and `scale_out` also has `w_t`'s shape except the reduction axis is
// replaced by the block count `K / block_size` -- exactly the shape ONNX's
// DequantizeLinear(axis=<reduction axis>, block_size=block_size) expects for
// its `x_scale` input.
inline bool TryQuantizeWeightBlockwiseInt4InPlace(const Tensor& w_t,
                                                  int64_t channel_axis,
                                                  int64_t block_size,
                                                  Tensor& q_out,
                                                  Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t reduction_axis = 1 - channel_axis;
  const int64_t K = reduction_axis == 0 ? dim0 : dim1;
  if (block_size <= 0 || K % block_size != 0) {
    return false;
  }
  const int64_t num_blocks = K / block_size;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  // scale_out's shape is w_t's shape with the reduction axis divided by
  // block_size; this maps element (i, j) of w_t to its (block, channel)
  // group's flat index in that smaller tensor.
  const int64_t scale_dim0 = reduction_axis == 0 ? num_blocks : dim0;
  const int64_t scale_dim1 = reduction_axis == 1 ? num_blocks : dim1;
  auto scale_index = [&](int64_t i, int64_t j) {
    const int64_t si = reduction_axis == 0 ? i / block_size : i;
    const int64_t sj = reduction_axis == 1 ? j / block_size : j;
    return si * scale_dim1 + sj;
  };

  std::vector<float> scale(static_cast<size_t>(scale_dim0 * scale_dim1), 0.0f);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      float& s = scale[static_cast<size_t>(scale_index(i, j))];
      s = std::max(s, std::fabs(at(i, j)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 7.0f : 1.0f;
  }

  // Unlike INT8 (which, like this function, now also goes to raw_data via
  // WriteRawDataLittleEndian -- see endian_read.h -- just one whole byte per
  // element instead of a nibble), ONNX's wire format packs INT4 two values
  // per byte in raw_data -- low nibble first, i.e. byte[i] = (code[2i] &
  // 0xF) | ((code[2i+1] & 0xF) << 4), matching onnx.numpy_helper's
  // _pack_4bitx2 -- and onnxsim's vendored onnx-optimizer ir_pb_converter has
  // no INT4-aware packing path of its own for the typed int32_data field, so
  // this builds the packed bytes directly instead of going through a
  // whole-byte-per-element helper. `K % block_size == 0` (checked above)
  // makes `dim0 * dim1`
  // always even here (block_size is even), so no odd-count padding byte is
  // needed.
  const int64_t numel = dim0 * dim1;
  std::vector<int8_t> codes(static_cast<size_t>(numel));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const float s = scale[static_cast<size_t>(scale_index(i, j))];
      const float q = std::round(at(i, j) / s);
      codes[static_cast<size_t>(i * dim1 + j)] =
          static_cast<int8_t>(std::clamp(q, -7.0f, 7.0f));
    }
  }
  std::string packed(static_cast<size_t>((numel + 1) / 2), '\0');
  for (int64_t i = 0; i < numel; ++i) {
    const uint8_t nibble =
        static_cast<uint8_t>(codes[static_cast<size_t>(i)]) & 0x0F;
    uint8_t& byte =
        reinterpret_cast<uint8_t&>(packed[static_cast<size_t>(i / 2)]);
    if (i % 2 == 0) {
      byte = nibble;
    } else {
      byte = static_cast<uint8_t>(byte | (nibble << 4));
    }
  }

  q_out.elem_type() = TensorProto_DataType_INT4;
  q_out.sizes() = {dim0, dim1};
  q_out.set_raw_data(std::move(packed));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {scale_dim0, scale_dim1};
  scale_out.floats() = std::move(scale);
  return true;
}

// Reads `w_t` (a 2-D float32 constant, [K, N] or, when `transposed`,
// [N, K]) into a flat row-major [N, K] (output channel first) buffer --
// matches TryQuantizeWeightBlockwiseInt4InPlace's own channel_axis
// convention, used by callers (magnitude_pruning.h, quarot.h) that need the
// weight in a fixed output-channel-first layout regardless of its own
// on-disk transposed-or-not storage.
inline std::vector<float> ReadWeightNK(const Tensor& w_t, bool transposed) {
  const int64_t dim0 = w_t.sizes()[0];
  const int64_t dim1 = w_t.sizes()[1];
  const int64_t rows = transposed ? dim0 : dim1;
  const int64_t cols = transposed ? dim1 : dim0;
  const std::vector<float> data = ReadFloatMatrix(w_t);
  std::vector<float> w_nk(static_cast<size_t>(rows * cols));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const float v = data[static_cast<size_t>(i * dim1 + j)];
      if (transposed) {
        w_nk[static_cast<size_t>(i * cols + j)] = v;
      } else {
        w_nk[static_cast<size_t>(j * cols + i)] = v;
      }
    }
  }
  return w_nk;
}

// Same block-wise scheme as TryQuantizeWeightBlockwiseInt4InPlace, but INT8
// (values in [-127, 127], scale = max(|w|) / 127 per block) instead of
// INT4 -- a middle ground between QuantizeWeightPerChannelInPlace's single
// per-channel INT8 scale (coarser, no block overhead) and INT4's block-wise
// scheme (finer, but a much narrower 4-bit code range): the same per-channel
// bit width as quantize_weight_only, with block-wise quantization's better
// resolution for channels a single per-channel scale would under-resolve.
// Used by weight_only_quantize_int8_block_matmul.h.
//
// Like INT4, this packs into raw_data (one whole byte per element, via
// WriteRawDataLittleEndian -- see endian_read.h) rather than the typed
// int32_data field, which -- though `int32_data` is a spec-legal field for
// INT8/UINT8/INT16/etc. too -- costs far more than 1 byte per element there
// (every varint-encoded entry needing the sign bit takes the field's full 10
// bytes). Unlike INT4's nibble-packing, INT8 needs no bit-packing of its
// own: each code is already a whole byte.
inline bool TryQuantizeWeightBlockwiseInt8InPlace(const Tensor& w_t,
                                                  int64_t channel_axis,
                                                  int64_t block_size,
                                                  Tensor& q_out,
                                                  Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t dim0 = sizes[0];
  const int64_t dim1 = sizes[1];
  const int64_t reduction_axis = 1 - channel_axis;
  const int64_t K = reduction_axis == 0 ? dim0 : dim1;
  if (block_size <= 0 || K % block_size != 0) {
    return false;
  }
  const int64_t num_blocks = K / block_size;

  const std::vector<float> data = ReadFloatMatrix(w_t);
  auto at = [&](int64_t i, int64_t j) { return data[i * dim1 + j]; };
  const int64_t scale_dim0 = reduction_axis == 0 ? num_blocks : dim0;
  const int64_t scale_dim1 = reduction_axis == 1 ? num_blocks : dim1;
  auto scale_index = [&](int64_t i, int64_t j) {
    const int64_t si = reduction_axis == 0 ? i / block_size : i;
    const int64_t sj = reduction_axis == 1 ? j / block_size : j;
    return si * scale_dim1 + sj;
  };

  std::vector<float> scale(static_cast<size_t>(scale_dim0 * scale_dim1), 0.0f);
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      float& s = scale[static_cast<size_t>(scale_index(i, j))];
      s = std::max(s, std::fabs(at(i, j)));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {dim0, dim1};
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int8_t> codes(static_cast<size_t>(dim0 * dim1));
  for (int64_t i = 0; i < dim0; ++i) {
    for (int64_t j = 0; j < dim1; ++j) {
      const float s = scale[static_cast<size_t>(scale_index(i, j))];
      const float q = std::round(at(i, j) / s);
      codes[i * dim1 + j] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {scale_dim0, scale_dim1};
  scale_out.floats() = std::move(scale);
  return true;
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
