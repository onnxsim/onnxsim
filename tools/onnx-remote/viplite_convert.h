// Tensor conversions used by the VIPLite worker (remote_viplite_worker.cpp): IEEE half <-> float and affine/fixed-point
// quantize/dequantize over whole buffers. Kept free of any VIPLite types so it is unit-tested on the host (viplite_convert_test.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace viplite_convert {

inline float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else {  // subnormal
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; --exp; }
      bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f; std::memcpy(&f, &bits, 4); return f;
}

inline uint16_t float_to_half(float f) {
  uint32_t x; std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1u))) ++half;  // round to nearest, ties to even
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // ties to even; a carry into the exponent is the correct result
  return static_cast<uint16_t>(half);
}

// Bulk converters. They hoist the format, scale and range out of the loop so the compiler can vectorize the 1-2 byte formats that real
// networks use.
struct QuantParams { float scale, zero; float lo, hi; };  // real -> q: clamp(rint(x / scale + zero))

template <typename T>
void quantize_bulk(const float* src, T* dst, uint64_t n, const QuantParams& q) {
  const float inv = 1.0f / q.scale;
  for (uint64_t i = 0; i < n; ++i) {
    float v = std::nearbyintf(src[i] * inv + q.zero);
    dst[i] = static_cast<T>(std::min(q.hi, std::max(q.lo, v)));
  }
}

template <typename T>
void dequantize_bulk(const T* src, float* dst, uint64_t n, const QuantParams& q) {
  for (uint64_t i = 0; i < n; ++i) dst[i] = (static_cast<float>(src[i]) - q.zero) * q.scale;
}

}  // namespace viplite_convert
