// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Node-matching and weight-quantization helpers for Conv, mirroring
// quantize_matmul_common.h's role for MatMul/Gemm: both quantize a constant
// weight to INT8 per output channel from its static values. Conv's weight
// layout ([Cout, Cin/groups, k...]) always puts the output channel on axis 0
// (unlike MatMul/Gemm, whose weight can be transposed), so there is no
// transposed-layout case to handle here.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// The pieces of a Conv node these passes care about.
struct ConvInfo {
  Value* x = nullptr;     // activation (not required to be constant)
  Value* w = nullptr;     // weight; must be a constant float32 tensor, rank
                          // >= 3 ([Cout, Cin/groups, k...])
  Value* bias = nullptr;  // Conv's optional B input; nullptr if absent
};

// Recognizes a Conv node, filling `info`. Kernel/stride/pad/dilation/group/
// auto_pad attributes are untouched by the caller -- Conv itself is never
// replaced, so they need no special handling.
inline bool MatchConv(Node* n, ConvInfo& info) {
  if (n->kind() != kConv) {
    return false;
  }
  const size_t num_inputs = n->inputs().size();
  if (num_inputs != 2 && num_inputs != 3) {
    return false;
  }
  info.x = n->input(0);
  info.w = n->input(1);
  if (num_inputs == 3) {
    info.bias = n->input(2);
  }
  return true;
}

// Reads `t` (a float32 constant of any rank) into a flat row-major,
// host-byte-order buffer, regardless of whether it is stored as raw bytes
// (which are little-endian on the wire regardless of host -- see
// endian_read.h) or a typed float array (already host-order, decoded by
// protobuf itself).
inline std::vector<float> ReadFloatTensorFlat(const Tensor& t) {
  int64_t numel = 1;
  for (const auto& s : t.sizes()) {
    numel *= s;
  }
  RequireMaterializedValues(
      t.name(),
      t.is_raw_data() ? static_cast<int64_t>(t.raw().size() / sizeof(float))
                      : static_cast<int64_t>(t.floats().size()),
      numel);
  if (t.is_raw_data()) {
    return ReadRawDataHostOrder<float>(t.data<float>(), numel);
  }
  return t.floats();
}

// Quantizes `w_t` (a constant float32 Conv weight, [Cout, Cin/groups, k...])
// per output channel (axis 0 -- Conv's layout gives no other choice):
// `q_out` has the same shape as `w_t`, and `scale_out`[c] is the symmetric
// INT8 scale for output channel c (max(|w[c, ...]|) / 127, or 1.0 for an
// all-zero channel so no scale is 0).
inline void QuantizeConvWeightPerOutputChannel(const Tensor& w_t, Tensor& q_out,
                                               Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t C = sizes[0];
  int64_t inner = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    inner *= sizes[i];
  }
  const std::vector<float> data = ReadFloatTensorFlat(w_t);

  std::vector<float> scale(static_cast<size_t>(C), 0.0f);
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      scale[c] = std::max(scale[c], std::fabs(data[c * inner + j]));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = sizes;
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int8_t> codes(static_cast<size_t>(C * inner));
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      const float q = std::round(data[c * inner + j] / scale[c]);
      codes[c * inner + j] =
          static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C};
  scale_out.floats() = std::move(scale);
}

// Same as QuantizeConvWeightPerOutputChannel, but INT16
// (max(|w[c, ...]|) / 32767 per channel) instead of INT8 -- used by
// weight_only_quantize_int16_conv.h. See
// QuantizeWeightPerChannelInPlaceInt16 in quantize_matmul_common.h for why.
inline void QuantizeConvWeightPerOutputChannelInt16(const Tensor& w_t,
                                                    Tensor& q_out,
                                                    Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t C = sizes[0];
  int64_t inner = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    inner *= sizes[i];
  }
  const std::vector<float> data = ReadFloatTensorFlat(w_t);

  std::vector<float> scale(static_cast<size_t>(C), 0.0f);
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      scale[c] = std::max(scale[c], std::fabs(data[c * inner + j]));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 32767.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT16;
  q_out.sizes() = sizes;
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int16_t> codes(static_cast<size_t>(C * inner));
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      const float q = std::round(data[c * inner + j] / scale[c]);
      codes[c * inner + j] =
          static_cast<int16_t>(std::clamp(q, -32767.0f, 32767.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C};
  scale_out.floats() = std::move(scale);
}

// Block-wise INT4 quantization of `w_t` (a constant float32 Conv weight,
// [Cout, Cin/groups, k...]), mirroring
// quantize_matmul_common.h's TryQuantizeWeightBlockwiseInt4InPlace but for
// Conv's multi-axis weight: Conv has no single reduction axis the way
// MatMul/Gemm does (every one of ``Cin/groups`` and the kernel dims
// contributes to one output pixel's sum), so instead of blocking along one
// of Conv's own axes (which would put an independent scale on every kernel
// spatial position -- little compression benefit when the kernel is, say,
// 3x3), this flattens everything but the output channel into one sequence
// of length `inner = Cin/groups * prod(k...)` and blocks *that*, exactly
// like a MatMul weight would be. `q_out`/`scale_out` are therefore 2-D
// ([Cout, inner] and [Cout, inner / block_size]) even though `w_t` is not --
// the caller reshapes `q_out`'s DequantizeLinear output back to `w_t`'s
// original shape (see weight_only_quantize_int4_conv.h).
//
// Returns false (nothing written) when `inner` is not evenly divisible by
// `block_size` -- the common, unambiguous case only, matching the MatMul
// version.
inline bool TryQuantizeConvWeightBlockwiseInt4Flat(const Tensor& w_t,
                                                   int64_t block_size,
                                                   Tensor& q_out,
                                                   Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t C = sizes[0];
  int64_t inner = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    inner *= sizes[i];
  }
  if (block_size <= 0 || inner % block_size != 0) {
    return false;
  }
  const int64_t num_blocks = inner / block_size;

  const std::vector<float> data = ReadFloatTensorFlat(w_t);

  std::vector<float> scale(static_cast<size_t>(C * num_blocks), 0.0f);
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      float& s = scale[static_cast<size_t>(c * num_blocks + j / block_size)];
      s = std::max(s, std::fabs(data[static_cast<size_t>(c * inner + j)]));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 7.0f : 1.0f;
  }

  // Pack two int4 codes per byte, low nibble first -- see
  // TryQuantizeWeightBlockwiseInt4InPlace's identical packing for why this
  // bypasses q_out.int32s() (ONNX's wire format for INT4 packs raw_data;
  // onnxsim's vendored onnx-optimizer has no INT4-aware packing for the
  // typed int32_data field). `inner % block_size == 0` with an even
  // block_size makes `C * inner` always even, so no odd-count padding byte
  // is needed.
  const int64_t numel = C * inner;
  std::vector<int8_t> codes(static_cast<size_t>(numel));
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      const float s =
          scale[static_cast<size_t>(c * num_blocks + j / block_size)];
      const float q = std::round(data[static_cast<size_t>(c * inner + j)] / s);
      codes[static_cast<size_t>(c * inner + j)] =
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
  q_out.sizes() = {C, inner};
  q_out.set_raw_data(std::move(packed));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C, num_blocks};
  scale_out.floats() = std::move(scale);
  return true;
}

// Same flatten-then-block scheme as TryQuantizeConvWeightBlockwiseInt4Flat,
// but INT8 (values in [-127, 127], scale = max(|w|) / 127 per block) instead
// of INT4 -- see TryQuantizeWeightBlockwiseInt8InPlace in
// quantize_matmul_common.h for why this exists alongside the per-channel
// INT8 scheme (QuantizeConvWeightPerOutputChannel) and the INT4 block-wise
// one. Used by weight_only_quantize_int8_block_conv.h. `q_out`/`scale_out`
// are 2-D ([C, inner] and [C, inner / block_size]) even though `w_t` is not
// -- like the INT4 version, the caller reshapes the DequantizeLinear output
// back to `w_t`'s original shape.
inline bool TryQuantizeConvWeightBlockwiseInt8Flat(const Tensor& w_t,
                                                   int64_t block_size,
                                                   Tensor& q_out,
                                                   Tensor& scale_out) {
  const auto& sizes = w_t.sizes();
  const int64_t C = sizes[0];
  int64_t inner = 1;
  for (size_t i = 1; i < sizes.size(); ++i) {
    inner *= sizes[i];
  }
  if (block_size <= 0 || inner % block_size != 0) {
    return false;
  }
  const int64_t num_blocks = inner / block_size;

  const std::vector<float> data = ReadFloatTensorFlat(w_t);

  std::vector<float> scale(static_cast<size_t>(C * num_blocks), 0.0f);
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      float& s = scale[static_cast<size_t>(c * num_blocks + j / block_size)];
      s = std::max(s, std::fabs(data[static_cast<size_t>(c * inner + j)]));
    }
  }
  for (float& s : scale) {
    s = s > 0.0f ? s / 127.0f : 1.0f;
  }

  q_out.elem_type() = TensorProto_DataType_INT8;
  q_out.sizes() = {C, inner};
  // raw_data, not int32s() -- see WriteRawDataLittleEndian's doc comment in
  // endian_read.h for why.
  std::vector<int8_t> codes(static_cast<size_t>(C * inner));
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t j = 0; j < inner; ++j) {
      const float s =
          scale[static_cast<size_t>(c * num_blocks + j / block_size)];
      const float q = std::round(data[static_cast<size_t>(c * inner + j)] / s);
      codes[c * inner + j] =
          static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
  }
  q_out.set_raw_data(WriteRawDataLittleEndian(codes));

  scale_out.elem_type() = TensorProto_DataType_FLOAT;
  scale_out.sizes() = {C, num_blocks};
  scale_out.floats() = std::move(scale);
  return true;
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
