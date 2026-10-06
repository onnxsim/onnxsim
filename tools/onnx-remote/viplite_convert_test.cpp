// Host unit test for viplite_convert.h (the VIPLite worker's half-float and quantize/dequantize routines). Needs no VIPLite SDK or device.
#include "viplite_convert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

using namespace viplite_convert;

static int g_failures = 0;
#define CHECK(cond)                                                                                  \
  do {                                                                                               \
    if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; }      \
  } while (0)
#define CHECK_EQ(a, b)                                                                               \
  do {                                                                                               \
    const auto va = (a); const auto vb = (b);                                                        \
    if (!(va == vb)) { std::printf("FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b, (long long)va, (long long)vb); ++g_failures; } \
  } while (0)

static float bits_to_float(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

static void test_half_roundtrip() {
  // Every finite half decodes and re-encodes to itself; NaNs stay NaN; the sign of zero survives.
  for (uint32_t h = 0; h < 0x10000; ++h) {
    const float f = half_to_float(static_cast<uint16_t>(h));
    const bool is_nan = (h & 0x7c00) == 0x7c00 && (h & 0x3ff);
    if (is_nan) { CHECK(std::isnan(f)); CHECK((float_to_half(f) & 0x7c00) == 0x7c00 && (float_to_half(f) & 0x3ff)); continue; }
    CHECK_EQ(float_to_half(f), static_cast<uint16_t>(h));
  }
}

static void test_half_known_values() {
  CHECK_EQ(float_to_half(1.0f), 0x3c00);
  CHECK_EQ(float_to_half(-2.0f), 0xc000);
  CHECK_EQ(float_to_half(0.0f), 0x0000);
  CHECK_EQ(float_to_half(-0.0f), 0x8000);
  CHECK_EQ(float_to_half(65504.0f), 0x7bff);                       // largest finite half
  CHECK_EQ(float_to_half(65520.0f), 0x7c00);                       // rounds up to infinity
  CHECK_EQ(float_to_half(1e6f), 0x7c00);
  CHECK_EQ(float_to_half(-1e6f), 0xfc00);
  CHECK_EQ(float_to_half(std::numeric_limits<float>::infinity()), 0x7c00);
  CHECK_EQ(float_to_half(std::ldexp(1.0f, -14)), 0x0400);          // smallest normal
  CHECK_EQ(float_to_half(std::ldexp(1.0f, -24)), 0x0001);          // smallest subnormal
  CHECK_EQ(float_to_half(std::ldexp(1.0f, -26)), 0x0000);          // underflows to zero
  // Round to nearest, ties to even (IEEE 754).
  CHECK_EQ(float_to_half(1.0f + std::ldexp(1.0f, -11)), 0x3c00);   // halfway between 0x3c00 and 0x3c01 -> even (0x3c00)
  CHECK_EQ(float_to_half(1.0f + 3 * std::ldexp(1.0f, -11)), 0x3c02);  // halfway between 0x3c01 and 0x3c02 -> even (0x3c02)
  CHECK_EQ(float_to_half(1.0f + std::ldexp(1.0f, -11) + std::ldexp(1.0f, -20)), 0x3c01);  // just above halfway -> up
  CHECK_EQ(float_to_half(std::ldexp(1.0f, -25)), 0x0000);          // halfway between 0 and the smallest subnormal -> even (0)
  CHECK_EQ(float_to_half(3 * std::ldexp(1.0f, -25)), 0x0002);      // halfway between 1 and 2 subnormal ulps -> even (2)
  CHECK_EQ(float_to_half(2047.5f + 0.0f), 0x67ff + 1);             // 2047.5 is halfway between 2047 (0x67ff) and 2048 (0x6800) -> even (0x6800)
  CHECK_EQ(float_to_half(bits_to_float(0x7fc00000u)) & 0x7c00, 0x7c00);  // NaN stays NaN
}

static void test_quantize_affine() {
  const QuantParams q{0.5f, 128.0f, 0.0f, 255.0f};  // uint8, scale 0.5, zero point 128
  const float in[] = {0.0f, 1.0f, -64.0f, -100.0f, 100.0f, 0.25f, 0.75f};
  uint8_t out[7];
  quantize_bulk(in, out, 7, q);
  CHECK_EQ(out[0], 128);
  CHECK_EQ(out[1], 130);
  CHECK_EQ(out[2], 0);     // -64 / 0.5 + 128 = 0 exactly
  CHECK_EQ(out[3], 0);     // below range saturates
  CHECK_EQ(out[4], 255);   // above range saturates
  CHECK_EQ(out[5], 128);   // 128.5 -> tie -> even (128)
  CHECK_EQ(out[6], 130);   // 129.5 -> tie -> even (130)
  // Non-finite inputs must not be undefined behaviour: NaN -> low end, +/-inf saturate.
  const float odd[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  uint8_t o2[3];
  quantize_bulk(odd, o2, 3, q);
  CHECK_EQ(o2[0], 0);
  CHECK_EQ(o2[1], 255);
  CHECK_EQ(o2[2], 0);
}

static void test_quantize_ties_and_signed() {
  const QuantParams q{1.0f, 0.0f, -128.0f, 127.0f};  // int8, scale 1
  const float in[] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 200.0f, -200.0f, 126.5f, 127.5f};
  int8_t out[10];
  quantize_bulk(in, out, 10, q);
  const int8_t want[] = {0, 2, 2, 0, -2, -2, 127, -128, 126, 127};
  for (int i = 0; i < 10; ++i) CHECK_EQ(out[i], want[i]);
}

static void test_dequantize_and_roundtrip() {
  const QuantParams q{0.0935506672f, 164.0f, 0.0f, 255.0f};  // a real scale/zero point from the A733's YOLOv5n output
  uint8_t all[256]; float real[256]; uint8_t back[256];
  for (int i = 0; i < 256; ++i) all[i] = static_cast<uint8_t>(i);
  dequantize_bulk(all, real, 256, q);
  CHECK(real[164] == 0.0f);
  CHECK(std::fabs(real[165] - 0.0935506672f) < 1e-7f);
  quantize_bulk(real, back, 256, q);
  for (int i = 0; i < 256; ++i) CHECK_EQ(back[i], all[i]);  // quantize(dequantize(q)) == q for every code
  // int8 per-channel style (zero point 0) and int16 too
  const QuantParams s8{0.02f, 0.0f, -128.0f, 127.0f};
  int8_t a8[256]; float r8[256]; int8_t b8[256];
  for (int i = 0; i < 256; ++i) a8[i] = static_cast<int8_t>(i - 128);
  dequantize_bulk(a8, r8, 256, s8);
  quantize_bulk(r8, b8, 256, s8);
  for (int i = 0; i < 256; ++i) CHECK_EQ(b8[i], a8[i]);
}

static void test_dynamic_fixed_point() {
  // DFP with fixed_point_pos 8 is scale 2^-8, zero point 0 (worker's quant_params).
  const QuantParams q{std::ldexp(1.0f, -8), 0.0f, -32768.0f, 32767.0f};
  const float in[] = {1.0f, -1.0f, 200.0f, -200.0f, 0.001953125f};  // last is exactly half an ulp: tie -> even (0)
  int16_t out[5];
  quantize_bulk(in, out, 5, q);
  CHECK_EQ(out[0], 256);
  CHECK_EQ(out[1], -256);
  CHECK_EQ(out[2], 32767);   // 200 * 256 = 51200 saturates
  CHECK_EQ(out[3], -32768);
  CHECK_EQ(out[4], 0);
  float back[2]; const int16_t codes[] = {256, -128};
  dequantize_bulk(codes, back, 2, q);
  CHECK(back[0] == 1.0f && back[1] == -0.5f);
}

int main() {
  test_half_roundtrip();
  test_half_known_values();
  test_quantize_affine();
  test_quantize_ties_and_signed();
  test_dequantize_and_roundtrip();
  test_dynamic_fixed_point();
  if (g_failures) { std::printf("%d check(s) failed\n", g_failures); return 1; }
  std::printf("viplite_convert tests passed\n");
  return 0;
}
