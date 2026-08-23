/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++20 ggml_kquant_test.cpp -o t && ./t
 *
 * Dependency-free unit test for ggml_kquant.h's GGML K-quant dequantization
 * (Q8_0, Q4_K, Q5_K, Q6_K), mirroring gguf_dtype_test.cpp's style.
 *
 * Every block layout/formula this decodes was cross-checked against an
 * independent from-scratch Python transcription of GGML's own reference
 * (ggml-quants.c's dequantize_row_q8_0/q4_K/q5_K/q6_K) over full random
 * blocks -- 0 error. The cases here are smaller, hand-verifiable vectors
 * (chosen so most of a block's bytes are zero and its non-zero bytes'
 * contribution can be checked by hand), meant to catch a regression in this
 * file specifically, not to re-derive GGML's spec from scratch.
 */
#include "ggml_kquant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace onnxsim::tensor_pool::gguf;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void CheckNear(float got, float want, const std::string& what,
               float tol = 1e-4f) {
  Check(std::fabs(got - want) <= tol, what + " (got " + std::to_string(got) +
                                          ", want " + std::to_string(want) +
                                          ")");
}

void WriteLE16(std::vector<uint8_t>& buf, uint16_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFF));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

// Encodes `f` as an IEEE754 half-precision bit pattern -- a plain reference
// encoder (round-to-nearest not implemented; every value used below is
// exactly representable in fp16) used only to BUILD test input, the inverse
// direction of the Float16BitsToFloat32 function under test.
uint16_t EncodeF16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

void TestFloat16BitsToFloat32() {
  struct Row {
    uint16_t bits;
    float expected;
  };
  const Row rows[] = {
      {0x3C00, 1.0f},  {0x4000, 2.0f}, {0xC000, -2.0f},  {0x0000, 0.0f},
      {0x8000, -0.0f}, {0x3E00, 1.5f}, {0xC280, -3.25f},
  };
  for (const auto& r : rows) {
    CheckNear(Float16BitsToFloat32(r.bits), r.expected,
              "Float16BitsToFloat32(0x" + std::to_string(r.bits) + ")");
  }
  Check(std::isinf(Float16BitsToFloat32(0x7C00)), "fp16 +inf");
  Check(std::isinf(Float16BitsToFloat32(0xFC00)), "fp16 -inf");
  Check(std::isnan(Float16BitsToFloat32(0x7E00)), "fp16 nan");
  // Smallest subnormal (2^-24) and largest subnormal (2^-14 * (1 - 2^-10)).
  CheckNear(Float16BitsToFloat32(0x0001), 5.960464e-8f, "fp16 min subnormal",
            1e-12f);
  CheckNear(Float16BitsToFloat32(0x03FF), 6.097555e-5f, "fp16 max subnormal",
            1e-9f);

  // Round trip through the local EncodeF16 reference encoder.
  const float rt[] = {1.5f, -3.25f, 100.0f, 0.0f, -0.0f};
  for (float f : rt) {
    CheckNear(Float16BitsToFloat32(EncodeF16(f)), f, "fp16 round-trip");
  }
}

void TestQ8_0() {
  // d = 2.0, qs[i] = i - 16 (so values span the full int8-ish range this
  // test cares about), expected[i] = (i - 16) * 2.0.
  std::vector<uint8_t> block;
  WriteLE16(block, EncodeF16(2.0f));
  for (int i = 0; i < 32; ++i) {
    block.push_back(static_cast<uint8_t>(static_cast<int8_t>(i - 16)));
  }
  Check(block.size() == KQuantBlockBytes(GGML_TYPE_Q8_0), "Q8_0 block size");

  float out[32];
  DequantizeQ8_0Block(block.data(), out);
  bool ok = true;
  for (int i = 0; i < 32; ++i) {
    if (std::fabs(out[i] - static_cast<float>(i - 16) * 2.0f) > 1e-4f) {
      ok = false;
    }
  }
  Check(ok, "Q8_0 block decodes to (i - 16) * 2.0");

  // Through the dispatcher too, for 2 concatenated blocks (64 elements).
  std::vector<uint8_t> two_blocks = block;
  two_blocks.insert(two_blocks.end(), block.begin(), block.end());
  std::vector<float> dispatched;
  Check(DequantizeGgmlKQuant(two_blocks.data(), two_blocks.size(),
                             GGML_TYPE_Q8_0, 64, &dispatched),
        "DequantizeGgmlKQuant(Q8_0, 2 blocks) succeeds");
  Check(dispatched.size() == 64, "DequantizeGgmlKQuant(Q8_0) output size");
  if (dispatched.size() == 64) {
    CheckNear(dispatched[0], -32.0f, "Q8_0 dispatcher block 0 elem 0");
    CheckNear(dispatched[32], -32.0f, "Q8_0 dispatcher block 1 elem 0");
  }
}

// Builds a Q4_K block where only sub-block 0 (the first 32 outputs, is=0,
// the `j < 4` branch of GetScaleMinK4 -- a direct 6-bit mask, no bit-
// spreading from neighboring bytes) is non-trivial; every other sub-block's
// scale/min bytes are left 0, so its (scale, min) is (0, 0) and its outputs
// are trivially 0 regardless of quant code -- keeping this hand-verifiable
// without re-deriving GetScaleMinK4's full byte-spreading scheme.
void TestQ4_K() {
  std::vector<uint8_t> block(KQuantBlockBytes(GGML_TYPE_Q4_K), 0);
  const uint16_t d_bits = EncodeF16(1.0f);
  const uint16_t dmin_bits = EncodeF16(0.5f);
  block[0] = static_cast<uint8_t>(d_bits & 0xFF);
  block[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
  block[2] = static_cast<uint8_t>(dmin_bits & 0xFF);
  block[3] = static_cast<uint8_t>((dmin_bits >> 8) & 0xFF);
  uint8_t* scales = block.data() + 4;
  scales[0] = 10;  // sub-block 0's scale (direct 6-bit mask, j=0 < 4)
  scales[4] = 3;   // sub-block 0's min (direct 6-bit mask, j=0 < 4)
  uint8_t* qs = block.data() + 4 + 12;
  qs[0] = 0x05;  // element 0: low nibble 5, high nibble 0 (sub-block 1, d2=0)

  // d1 = d * scale = 1.0 * 10 = 10.0; m1 = dmin * min = 0.5 * 3 = 1.5.
  // expected[0] = d1 * (qs[0] & 0xF) - m1 = 10.0 * 5 - 1.5 = 48.5.
  // expected[32] (sub-block 1, d2=m2=0 since scales[1]/scales[5] are 0) = 0.
  float out[256];
  DequantizeQ4_KBlock(block.data(), out);
  CheckNear(out[0], 48.5f, "Q4_K sub-block 0 element 0");
  CheckNear(out[1], -1.5f, "Q4_K sub-block 0 element 1 (qs[1]=0 -> -m1)");
  CheckNear(out[32], 0.0f, "Q4_K sub-block 1 (zeroed scale/min) is 0");
  CheckNear(out[64], 0.0f, "Q4_K sub-block 2 (zeroed scale/min) is 0");
}

// Same trick as TestQ4_K: only sub-block 0 is non-trivial (u1 = 1 selects
// qh's bit 0 for the high 5th bit); every other sub-block's scale/min is 0.
void TestQ5_K() {
  std::vector<uint8_t> block(KQuantBlockBytes(GGML_TYPE_Q5_K), 0);
  const uint16_t d_bits = EncodeF16(1.0f);
  const uint16_t dmin_bits = EncodeF16(0.5f);
  block[0] = static_cast<uint8_t>(d_bits & 0xFF);
  block[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
  block[2] = static_cast<uint8_t>(dmin_bits & 0xFF);
  block[3] = static_cast<uint8_t>((dmin_bits >> 8) & 0xFF);
  uint8_t* scales = block.data() + 4;
  scales[0] = 10;
  scales[4] = 3;
  uint8_t* qh = block.data() + 4 + 12;
  uint8_t* ql = block.data() + 4 + 12 + 32;
  qh[0] = 0x01;  // bit 0 set -> element 0's high bit (u1 = 1) is set
  ql[0] = 0x05;  // low nibble 5

  // element 0: (ql[0]&0xF) + (qh[0]&1 ? 16 : 0) = 5 + 16 = 21.
  // expected[0] = d1 * 21 - m1 = 10.0 * 21 - 1.5 = 208.5.
  float out[256];
  DequantizeQ5_KBlock(block.data(), out);
  CheckNear(out[0], 208.5f, "Q5_K sub-block 0 element 0 (high bit set)");
  CheckNear(out[32], 0.0f, "Q5_K sub-block 1 (zeroed scale/min) is 0");
}

// Q6_K has no "zeroed sub-block is trivially 0" shortcut (every sub-block's
// scale is read from `scales`, and the -32 bias means a zero quant code
// still produces a non-zero dequantized value) -- direct single-element
// verification instead: element 0 depends only on ql[0], qh[0] (bits 0-1),
// scales[0], and d.
void TestQ6_K() {
  std::vector<uint8_t> block(KQuantBlockBytes(GGML_TYPE_Q6_K), 0);
  uint8_t* ql = block.data();
  uint8_t* qh = block.data() + 128;
  int8_t* scales = reinterpret_cast<int8_t*>(block.data() + 128 + 64);
  const uint16_t d_bits = EncodeF16(2.0f);
  block[128 + 64 + 16] = static_cast<uint8_t>(d_bits & 0xFF);
  block[128 + 64 + 16 + 1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);

  scales[0] = 5;  // sc[is=0 + 0] for element 0
  ql[0] = 0x03;   // element 0's low 4 bits: 3
  qh[0] = 0x00;   // element 0's high 2 bits (bits 0-1 of qh[0]): 0

  // q1 = (ql[0]&0xF) | (((qh[0]>>0)&3)<<4) - 32 = 3 - 32 = -29.
  // expected[0] = d * scales[0] * q1 = 2.0 * 5 * -29 = -290.0.
  float out[256];
  DequantizeQ6_KBlock(block.data(), out);
  CheckNear(out[0], -290.0f, "Q6_K element 0");
}

void TestDequantizeGgmlKQuantRejectsBadInput() {
  std::vector<uint8_t> q8_block(KQuantBlockBytes(GGML_TYPE_Q8_0), 0);
  std::vector<float> out;

  // Not a multiple of the block size.
  Check(!DequantizeGgmlKQuant(q8_block.data(), q8_block.size(), GGML_TYPE_Q8_0,
                              33, &out),
        "rejects non-block-aligned numel");

  // raw_size doesn't match numel's expected byte length.
  Check(!DequantizeGgmlKQuant(q8_block.data(), q8_block.size() - 1,
                              GGML_TYPE_Q8_0, 32, &out),
        "rejects mismatched raw_size");

  // Not a K-quant type at all (F32 -- a raw type, not a block-quantized one).
  Check(!DequantizeGgmlKQuant(q8_block.data(), q8_block.size(), GGML_TYPE_F32,
                              32, &out),
        "rejects non-K-quant ggml_type");

  // Negative numel.
  Check(!DequantizeGgmlKQuant(q8_block.data(), q8_block.size(), GGML_TYPE_Q8_0,
                              -1, &out),
        "rejects negative numel");

  Check(out.empty(), "no partial output written on any rejected call");
}

}  // namespace

int main() {
  TestFloat16BitsToFloat32();
  TestQ8_0();
  TestQ4_K();
  TestQ5_K();
  TestQ6_K();
  TestDequantizeGgmlKQuantRejectsBadInput();

  if (g_failures == 0) {
    std::printf("ggml_kquant_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "ggml_kquant_test: %d failure(s)\n", g_failures);
  return 1;
}
