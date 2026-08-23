/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dequantization for the four GGML "K-quant" block formats gguf_dtype.h's
 * IsKQuant covers (Q4_K, Q5_K, Q6_K, Q8_0): decodes a tensor's native, still-
 * packed GGML block bytes (see gguf_dtype.h's KQuantBlockBytes) into plain
 * host-order float32 values.
 *
 * Pure, dependency-free (no protobuf, no onnx headers) like gguf_dtype.h
 * itself -- operates on raw bytes and the same integer ggml_type codes, so
 * this builds and unit-tests standalone.
 *
 * Every block layout and dequantization formula here is transcribed
 * verbatim from GGML's own reference implementation
 * (https://github.com/ggml-org/ggml -- ggml-common.h's block_q*_K/block_q8_0
 * struct layouts, ggml-quants.c's dequantize_row_q*_K/dequantize_row_q8_0
 * and their get_scale_min_k4 helper). Byte order: like
 * onnx::TensorProto::raw_data and every GGUF file (see tensor_pool.h's file
 * comment), a block's multi-byte fields are little-endian on disk regardless
 * of host byte order -- this file's Float16BitsToFloat32 argument is always
 * reconstructed via explicit byte-at-a-time reads (ReadLE16), never a
 * reinterpret_cast of the block struct, so decoding is correct on a
 * big-endian host too (this repo tests on s390x). A block's single-byte
 * fields (the packed quant codes, the int8 Q6_K scales) need no such care --
 * one byte has no endianness.
 */
#ifndef ONNXSIM_GGML_KQUANT_H_
#define ONNXSIM_GGML_KQUANT_H_

#include <cstdint>
#include <cstring>
#include <vector>

#include "gguf_dtype.h"

namespace onnxsim {
namespace tensor_pool {
namespace gguf {

// Reconstructs a little-endian uint16_t from two bytes, regardless of host
// byte order (unlike a reinterpret_cast, which would misread the value on a
// big-endian host).
inline uint16_t ReadLE16(const uint8_t* p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

// IEEE754 half-precision (the wire format of GGML's `ggml_half`, and ONNX's
// FLOAT16) -> single-precision. Standard bit-manipulation conversion:
// normal/subnormal/zero/inf/nan all handled, mirroring
// passes/quantize_fp16.h's FloatToFloat16Bits in the opposite direction.
inline float Float16BitsToFloat32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/-0
    } else {
      // Subnormal half: normalize by shifting the mantissa left until its
      // implicit leading bit (0x400) appears, decrementing the exponent
      // once per shift.
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf or nan
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// GGML's get_scale_min_k4: unpacks Q4_K/Q5_K's 12-byte `scales` array (eight
// 6-bit scale values and eight 6-bit min values, for the super-block's 8
// sub-blocks) into sub-block `j`'s (scale, min) pair. The packing spreads
// each 6-bit value's high 2 bits into a byte another sub-block's low nibble
// otherwise wouldn't use -- see GGML's own comment-free reference; this is
// transcribed bit-for-bit rather than re-derived.
inline void GetScaleMinK4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
  }
}

// Decodes one 34-byte Q8_0 block (32 elements) into `out`.
inline void DequantizeQ8_0Block(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const auto* qs = reinterpret_cast<const int8_t*>(block + 2);
  for (int j = 0; j < 32; ++j) {
    out[j] = static_cast<float>(qs[j]) * d;
  }
}

// Decodes one 144-byte Q4_K block (256 elements) into `out`.
inline void DequantizeQ4_KBlock(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const float dmin = Float16BitsToFloat32(ReadLE16(block + 2));
  const uint8_t* scales = block + 4;  // 12 bytes
  const uint8_t* q = block + 4 + 12;  // 128 bytes

  int is = 0;
  float* y = out;
  for (int j = 0; j < 256; j += 64) {
    uint8_t sc, m;
    GetScaleMinK4(is + 0, scales, &sc, &m);
    const float d1 = d * sc;
    const float m1 = dmin * m;
    GetScaleMinK4(is + 1, scales, &sc, &m);
    const float d2 = d * sc;
    const float m2 = dmin * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
    q += 32;
    is += 2;
  }
}

// Decodes one 176-byte Q5_K block (256 elements) into `out`.
inline void DequantizeQ5_KBlock(const uint8_t* block, float* out) {
  const float d = Float16BitsToFloat32(ReadLE16(block));
  const float dmin = Float16BitsToFloat32(ReadLE16(block + 2));
  const uint8_t* scales = block + 4;        // 12 bytes
  const uint8_t* qh = block + 4 + 12;       // 32 bytes
  const uint8_t* ql = block + 4 + 12 + 32;  // 128 bytes

  int is = 0;
  uint8_t u1 = 1, u2 = 2;
  float* y = out;
  for (int j = 0; j < 256; j += 64) {
    uint8_t sc, m;
    GetScaleMinK4(is + 0, scales, &sc, &m);
    const float d1 = d * sc;
    const float m1 = dmin * m;
    GetScaleMinK4(is + 1, scales, &sc, &m);
    const float d2 = d * sc;
    const float m2 = dmin * m;
    for (int l = 0; l < 32; ++l) {
      *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
    }
    for (int l = 0; l < 32; ++l) {
      *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
    }
    ql += 32;
    is += 2;
    u1 = static_cast<uint8_t>(u1 << 2);
    u2 = static_cast<uint8_t>(u2 << 2);
  }
}

// Decodes one 210-byte Q6_K block (256 elements) into `out`.
inline void DequantizeQ6_KBlock(const uint8_t* block, float* out) {
  const uint8_t* ql = block;        // 128 bytes
  const uint8_t* qh = block + 128;  // 64 bytes
  const auto* sc = reinterpret_cast<const int8_t*>(block + 128 + 64);  // 16

  const float d = Float16BitsToFloat32(ReadLE16(block + 128 + 64 + 16));

  float* y = out;
  for (int n = 0; n < 256; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int8_t q1 =
          static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) -
          32;
      const int8_t q2 =
          static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) -
          32;
      const int8_t q3 =
          static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) -
          32;
      const int8_t q4 =
          static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) -
          32;
      y[l + 0] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

// Decodes `raw` (an IsKQuant(ggml_type) tensor's native block bytes, exactly
// `numel / KQuantBlockElements(ggml_type) * KQuantBlockBytes(ggml_type)`
// bytes long) into `numel` host-order float32 values, appended to `out`
// (not cleared first). Returns false, leaving `out` untouched, if
// `ggml_type` is not one of the four IsKQuant types, `numel` is not a
// multiple of its block size, or `raw_size` does not match the expected
// byte length for `numel` elements (a corrupt/truncated buffer).
inline bool DequantizeGgmlKQuant(const uint8_t* raw, size_t raw_size,
                                 uint32_t ggml_type, int64_t numel,
                                 std::vector<float>* out) {
  if (!IsKQuant(ggml_type) || numel < 0) {
    return false;
  }
  const size_t block_elems = KQuantBlockElements(ggml_type);
  const size_t block_bytes = KQuantBlockBytes(ggml_type);
  const uint64_t unumel = static_cast<uint64_t>(numel);
  if (unumel % block_elems != 0) {
    return false;
  }
  const uint64_t num_blocks = unumel / block_elems;
  if (num_blocks * block_bytes != raw_size) {
    return false;
  }

  const size_t out_start = out->size();
  out->resize(out_start + unumel);
  float* dst = out->data() + out_start;
  const uint8_t* src = raw;
  for (uint64_t b = 0; b < num_blocks; ++b) {
    switch (ggml_type) {
      case GGML_TYPE_Q8_0:
        DequantizeQ8_0Block(src, dst);
        break;
      case GGML_TYPE_Q4_K:
        DequantizeQ4_KBlock(src, dst);
        break;
      case GGML_TYPE_Q5_K:
        DequantizeQ5_KBlock(src, dst);
        break;
      case GGML_TYPE_Q6_K:
        DequantizeQ6_KBlock(src, dst);
        break;
      default:
        return false;  // Unreachable: IsKQuant already filtered ggml_type.
    }
    src += block_bytes;
    dst += block_elems;
  }
  return true;
}

}  // namespace gguf
}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_GGML_KQUANT_H_
