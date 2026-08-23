/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++17 gguf_dtype_test.cpp -o t && ./t
 *
 * Dependency-free unit test for the ONNX<->GGML dtype mapping, mirroring
 * tensor_pool_dtype_test.cpp.
 */
#include "gguf_dtype.h"

#include <cstdio>
#include <string>

using namespace onnxsim::tensor_pool::gguf;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Row {
  int32_t onnx;
  uint32_t ggml;
  size_t bytes;
};

}  // namespace

int main() {
  const Row rows[] = {
      {ONNX_FLOAT, GGML_TYPE_F32, 4},     {ONNX_FLOAT16, GGML_TYPE_F16, 2},
      {ONNX_BFLOAT16, GGML_TYPE_BF16, 2}, {ONNX_DOUBLE, GGML_TYPE_F64, 8},
      {ONNX_INT8, GGML_TYPE_I8, 1},       {ONNX_INT16, GGML_TYPE_I16, 2},
      {ONNX_INT32, GGML_TYPE_I32, 4},     {ONNX_INT64, GGML_TYPE_I64, 8},
  };

  for (const auto& row : rows) {
    uint32_t ggml = 0xFFFFFFFF;
    Check(FromOnnx(row.onnx, &ggml) && ggml == row.ggml,
          "FromOnnx(" + std::to_string(row.onnx) +
              ") == " + std::to_string(row.ggml));
    Check(ElementSize(row.ggml) == row.bytes,
          "ElementSize(" + std::to_string(row.ggml) +
              ") == " + std::to_string(row.bytes));
    Check(IsRaw(row.ggml), "IsRaw(" + std::to_string(row.ggml) + ")");
    int32_t onnx = -1;
    Check(ToOnnx(row.ggml, &onnx) && onnx == row.onnx,
          "ToOnnx(FromOnnx(" + std::to_string(row.onnx) + "))) round-trips");
  }

  // Quantized types this mapping does NOT decode (legacy Q4_0/Q4_1/Q5_0/
  // Q5_1/Q8_1, Q2_K/Q3_K, Q8_K, IQ*_XXS/BF16-adjacent codes, ...) or
  // unrecognized types are rejected, not guessed at. Deliberately excludes
  // 8/12/13/14 (Q8_0/Q4_K/Q5_K/Q6_K), which ARE supported -- see the
  // k_quant_rows loop below.
  const uint32_t quantized[] = {2, 3, 6, 7, 9, 10, 11, 15, 16, 20, 29, 9999};
  for (uint32_t q : quantized) {
    int32_t onnx = -1;
    Check(!ToOnnx(q, &onnx),
          "ToOnnx rejects quantized/unknown type " + std::to_string(q));
    Check(!IsRaw(q),
          "IsRaw is false for quantized/unknown type " + std::to_string(q));
    Check(!IsKQuant(q),
          "IsKQuant is false for unsupported type " + std::to_string(q));
    Check(ElementSize(q) == 0,
          "ElementSize is 0 for quantized/unknown type " + std::to_string(q));
    uint64_t nbytes = 0xFFFFFFFFFFFFFFFFull;
    Check(!TryTotalBytes(q, 256, &nbytes),
          "TryTotalBytes rejects unsupported type " + std::to_string(q));
  }

  // The four K-quant types this mapping DOES decode -- Q4_K/Q5_K/Q6_K
  // (super-blocks of 256 elements) and Q8_0 (blocks of 32). Each round-trips
  // through ToOnnx/FromOnnx to its private ONNXSIM_GGML_* code (unlike every
  // OTHER private dtype a caller might construct, which FromOnnx correctly
  // rejects -- see the unsupported_onnx loop below), is NOT IsRaw (its
  // sizing is block-based, not per-element), and TryTotalBytes agrees with
  // KQuantBlockElements/KQuantBlockBytes's block math.
  struct KQuantRow {
    uint32_t ggml;
    int32_t onnx;
    size_t block_elems;
    size_t block_bytes;
  };
  const KQuantRow k_quant_rows[] = {
      {GGML_TYPE_Q8_0, ONNXSIM_GGML_Q8_0, 32, 34},
      {GGML_TYPE_Q4_K, ONNXSIM_GGML_Q4_K, 256, 144},
      {GGML_TYPE_Q5_K, ONNXSIM_GGML_Q5_K, 256, 176},
      {GGML_TYPE_Q6_K, ONNXSIM_GGML_Q6_K, 256, 210},
  };
  for (const auto& row : k_quant_rows) {
    Check(IsKQuant(row.ggml), "IsKQuant(" + std::to_string(row.ggml) + ")");
    Check(!IsRaw(row.ggml), "!IsRaw(" + std::to_string(row.ggml) + ")");
    Check(ElementSize(row.ggml) == 0,
          "ElementSize is 0 for K-quant type " + std::to_string(row.ggml));
    Check(KQuantBlockElements(row.ggml) == row.block_elems,
          "KQuantBlockElements(" + std::to_string(row.ggml) + ")");
    Check(KQuantBlockBytes(row.ggml) == row.block_bytes,
          "KQuantBlockBytes(" + std::to_string(row.ggml) + ")");

    int32_t onnx = -1;
    Check(ToOnnx(row.ggml, &onnx) && onnx == row.onnx,
          "ToOnnx(" + std::to_string(row.ggml) +
              ") == " + std::to_string(row.onnx));
    uint32_t ggml_back = 0;
    Check(FromOnnx(row.onnx, &ggml_back) && ggml_back == row.ggml,
          "FromOnnx(ToOnnx(" + std::to_string(row.ggml) + "))) round-trips");

    // 3 blocks' worth of elements -> exactly 3 blocks' worth of bytes; not
    // a multiple of the block size -> rejected, not rounded/truncated.
    uint64_t nbytes = 0;
    Check(TryTotalBytes(row.ggml, row.block_elems * 3, &nbytes) &&
              nbytes == row.block_bytes * 3,
          "TryTotalBytes(" + std::to_string(row.ggml) + ", 3 blocks)");
    Check(!TryTotalBytes(row.ggml, row.block_elems + 1, &nbytes),
          "TryTotalBytes rejects non-block-aligned nelems for " +
              std::to_string(row.ggml));
  }

  // ONNX dtypes with no raw ggml counterpart (unsigned ints, BOOL, STRING,
  // UNDEFINED -- ggml has no unsigned or bool tensor type at all).
  const int32_t unsupported_onnx[] = {2 /*UINT8*/, 4 /*UINT16*/, 9 /*BOOL*/,
                                      8 /*STRING*/, ONNX_UNDEFINED};
  for (int32_t d : unsupported_onnx) {
    uint32_t ggml = 0;
    Check(!FromOnnx(d, &ggml),
          "FromOnnx rejects unsupported ONNX dtype " + std::to_string(d));
  }

  if (g_failures == 0) {
    std::printf("gguf_dtype_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "gguf_dtype_test: %d failure(s)\n", g_failures);
  return 1;
}
