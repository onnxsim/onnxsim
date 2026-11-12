// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Endian-safe reads of a Tensor's raw_data, for every onnxsim pass that reads
// a constant's numeric values directly -- e.g. to quantize a weight, or to
// match a fusion pattern's literal scalar operands (0.5, sqrt(2), ...).
//
// ONNX fixes the wire byte order of TensorProto::raw_data to little-endian on
// every host (see onnxsim::dlpack::kRawDataIsHostOrder's doc comment,
// dlpack_dtype.h, for the identical concern at the DLPack constant-folding
// boundary). onnx-optimizer's in-memory Tensor (onnx/common/ir.h), used
// throughout onnxsim's own passes, copies that wire byte range into its own
// raw_data storage completely unchanged (see ir_pb_converter.cc's
// importTensor) -- so on a little-endian host Tensor::data<T>() already
// returns host-order values, but on a big-endian host it returns the wire
// bytes verbatim, misread as if they were already native-order. A typed
// field (Tensor::floats(), ::doubles(), ...) needs no such swap either way:
// protobuf decodes those into host-order scalars itself, which is why every
// read site here only ever swaps the is_raw_data() branch.

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "dlpack_dtype.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Throws unless a constant tensor actually carries its `numel` values. An
// external-data initializer that was never loaded keeps its shape but has no
// bytes, and reading it would run past an empty buffer (a SEGV, not an error).
inline void RequireMaterializedValues(const std::string& name, int64_t have,
                                      int64_t numel) {
  if (have < numel) {
    throw std::runtime_error(
        "onnxsim: constant tensor '" + name + "' has " + std::to_string(have) +
        " of " + std::to_string(numel) +
        " values loaded; load its external data before optimizing");
  }
}

// Copies `numel` elements starting at `raw` (a Tensor::data<T>() pointer into
// raw_data storage) into a host-order vector<T>, swapping each element's
// bytes only on a big-endian host -- a plain memcpy on a little-endian one,
// where raw_data already agrees with the host.
template <typename T>
inline std::vector<T> ReadRawDataHostOrder(const T* raw, int64_t numel) {
  std::vector<T> out(static_cast<size_t>(numel));
  std::memcpy(out.data(), raw, out.size() * sizeof(T));
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                      out.size() * sizeof(T), sizeof(T));
  }
  return out;
}

// The write-side counterpart of ReadRawDataHostOrder: packs a host-order
// vector<T> into a little-endian-on-wire byte string, suitable for
// Tensor::set_raw_data(). Every onnxsim quantization pass that builds a small
// (1/2-byte-per-element) integer or float-bit-pattern tensor -- INT8/UINT8/
// INT16/UINT16, or FLOAT16/BFLOAT16/FLOAT8's raw bit patterns -- should
// prefer this over Tensor::int32s(): ONNX's wire format allows either
// (TensorProto.int32_data is a valid, spec-legal field for all of these
// types, and onnxsim's own ir_pb_converter serializes whichever one the
// in-memory Tensor is carrying -- see is_raw_data() there), but int32_data is
// a `repeated int32` protobuf field, individually varint-encoded: every
// value takes at least 1 byte, values needing the sign bit (anything
// negative, since this is a plain `int32` field, not zigzag-coded `sint32`)
// take the full 10 bytes every such value's varint encoding requires,
// regardless of the logical type's own width. raw_data instead packs each
// element at its true byte width with no per-element framing, matching what
// onnx's own reference tensor builder (onnx.numpy_helper.from_array, which
// always uses raw_data, INT8 and FLOAT16 included) produces. A plain memcpy
// on a little-endian host, byte-swapped on a big-endian one -- the same
// convention ReadRawDataHostOrder uses in reverse.
template <typename T>
inline std::string WriteRawDataLittleEndian(const std::vector<T>& values) {
  std::string out(values.size() * sizeof(T), '\0');
  std::memcpy(out.data(), values.data(), out.size());
  if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
    onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(out.data()),
                                      out.size(), sizeof(T));
  }
  return out;
}

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
