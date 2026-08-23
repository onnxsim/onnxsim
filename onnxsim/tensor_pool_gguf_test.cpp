/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone: g++ -std=c++20 tensor_pool_gguf_test.cpp tensor_pool.cpp \
 *               tensor_pool_gguf.cpp -o t && ./t
 *
 * Dependency-free unit test for TensorPool's GGUF read/write
 * (tensor_pool_gguf.cpp), mirroring tensor_pool_test.cpp's safetensors
 * coverage plus GGUF-specific cases: ne[] dimension order, alignment
 * padding, and skipping quantized/unsupported tensor types.
 */
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>

#include "gguf_dtype.h"
#include "tensor_pool.h"

using namespace onnxsim::tensor_pool;
using namespace onnxsim::tensor_pool::gguf;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::string TempPath(const std::string& suffix) {
  return "/tmp/onnxsim_tensor_pool_gguf_test_" + std::to_string(::getpid()) +
         suffix;
}

template <typename T>
void WriteLE(std::ostream& out, T v) {
  static_assert(std::is_trivially_copyable_v<T>);
  unsigned char raw[sizeof(T)];
  std::memcpy(raw, &v, sizeof(T));
  if constexpr (std::endian::native == std::endian::big) {
    std::reverse(std::begin(raw), std::end(raw));
  }
  out.write(reinterpret_cast<const char*>(raw), sizeof(T));
}

void TestRoundTrip() {
  TensorPool pool;
  // A non-square 2x3 tensor specifically to catch a dimension-order bug:
  // ggml's ne[] is innermost-dimension-first, the reverse of onnx's
  // outermost-first shape, and getting that backwards wouldn't corrupt the
  // bytes (still contiguous) but would silently transpose the logical shape.
  pool.Add("weight", ONNX_FLOAT, {2, 3}, std::string(24, '\x11'));
  pool.Add("bias", ONNX_INT64, {4}, std::string(32, '\x22'));
  pool.Add("scalar", ONNX_INT8, {}, std::string(1, '\x33'));

  std::string path = TempPath("_roundtrip.gguf");
  std::map<std::string, std::string> metadata = {
      {"general.architecture", "onnxsim-test"}};
  pool.SaveGGUF(path, metadata);

  TensorPool loaded;
  auto skipped = loaded.LoadGGUF(path);
  Check(skipped.empty(), "nothing skipped for an all-raw-dtype pool");
  Check(loaded.size() == 3, "all three tensors loaded");

  const Entry* w = loaded.Find("weight");
  Check(w != nullptr, "weight present");
  if (w != nullptr) {
    Check(w->dtype == ONNX_FLOAT, "weight dtype round-trips");
    Check(w->shape == std::vector<int64_t>({2, 3}),
          "weight shape round-trips in the RIGHT order (not transposed to "
          "{3, 2})");
    Check(w->data == std::string(24, '\x11'), "weight bytes round-trip");
  }

  const Entry* b = loaded.Find("bias");
  Check(b != nullptr && b->dtype == ONNX_INT64 &&
            b->shape == std::vector<int64_t>({4}) &&
            b->data == std::string(32, '\x22'),
        "bias round-trips");

  const Entry* s = loaded.Find("scalar");
  Check(s != nullptr && s->shape.empty() && s->data == std::string(1, '\x33'),
        "rank-0 (scalar) tensor round-trips");

  std::remove(path.c_str());
}

void TestEmptyMetadataOmitted() {
  // No `string_metadata` argument at all -- must still produce a loadable
  // file with metadata_kv_count == 0, not e.g. write a garbage/absent count.
  TensorPool pool;
  pool.Add("x", ONNX_FLOAT, {1}, std::string(4, '\0'));
  std::string path = TempPath("_no_metadata.gguf");
  pool.SaveGGUF(path);
  TensorPool loaded;
  Check(loaded.LoadGGUF(path).empty() && loaded.size() == 1,
        "a file saved with no metadata still loads correctly");
  std::remove(path.c_str());
}

void TestAlignmentPadding() {
  // Two tensors whose sizes are not multiples of the default 32-byte
  // alignment -- exercises the inter-tensor padding path, not just the
  // header-to-data-section padding every other test already exercises.
  TensorPool pool;
  pool.Add("a", ONNX_INT8, {5}, std::string(5, 'a'));  // 5 bytes, not aligned
  pool.Add("b", ONNX_INT8, {7}, std::string(7, 'b'));  // 7 bytes, not aligned
  std::string path = TempPath("_align.gguf");
  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  uint64_t data_section_start = 0;
  pool.SaveGGUF(path, {}, &offsets, &data_section_start);

  Check(data_section_start % kDefaultAlignment == 0,
        "data section starts aligned");
  Check(offsets.at("a").first % kDefaultAlignment == 0,
        "tensor 'a' starts aligned");
  Check(offsets.at("b").first % kDefaultAlignment == 0,
        "tensor 'b' starts aligned (after 'a's unaligned 5-byte length)");
  Check(offsets.at("a").second - offsets.at("a").first == 5,
        "'a' spans exactly its 5 bytes, not the alignment padding");

  TensorPool loaded;
  Check(loaded.LoadGGUF(path).empty(), "nothing skipped");
  const Entry* a = loaded.Find("a");
  const Entry* b = loaded.Find("b");
  Check(a != nullptr && a->data == std::string(5, 'a'), "'a' round-trips");
  Check(b != nullptr && b->data == std::string(7, 'b'), "'b' round-trips");
  std::remove(path.c_str());
}

// Writes a minimal, hand-built GGUF v3 file with one raw F32 tensor and one
// tensor of a made-up "quantized" ggml_type (id 2 == GGML_TYPE_Q4_0) this
// pool cannot represent, to exercise LoadGGUF's skip-and-report path -- the
// case that matters most in practice, since a real quantized LLM's .gguf is
// mostly tensors like this.
void TestSkipsUnsupportedDtype() {
  std::string path = TempPath("_quantized.gguf");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    auto write_string = [&](const std::string& s) {
      WriteLE<uint64_t>(out, s.size());
      out.write(s.data(), static_cast<std::streamsize>(s.size()));
    };

    WriteLE<uint32_t>(out, kMagic);
    WriteLE<uint32_t>(out, kSupportedVersion);
    WriteLE<uint64_t>(out, 2);  // tensor_count
    WriteLE<uint64_t>(out, 0);  // metadata_kv_count

    // Tensor 0: "raw", F32, ne=[2] (8 bytes), offset 0.
    write_string("raw");
    WriteLE<uint32_t>(out, 1);              // n_dimensions
    WriteLE<uint64_t>(out, 2);              // ne[0]
    WriteLE<uint32_t>(out, GGML_TYPE_F32);  // type
    WriteLE<uint64_t>(out, 0);              // offset

    // Tensor 1: "quant", ggml_type 2 (Q4_0), ne=[32] -- offset chosen past
    // 'raw's aligned 32-byte slot. This test never decodes its bytes (can't
    // -- that's the whole point), so the payload content doesn't matter.
    write_string("quant");
    WriteLE<uint32_t>(out, 1);                  // n_dimensions
    WriteLE<uint64_t>(out, 32);                 // ne[0]
    WriteLE<uint32_t>(out, 2);                  // GGML_TYPE_Q4_0
    WriteLE<uint64_t>(out, kDefaultAlignment);  // offset (aligned)

    // Pad to the data section (header ends right here; align up to 32).
    uint64_t header_end = static_cast<uint64_t>(out.tellp());
    uint64_t data_start = (header_end + kDefaultAlignment - 1) /
                          kDefaultAlignment * kDefaultAlignment;
    for (uint64_t i = header_end; i < data_start; ++i) out.put('\0');

    // 'raw' tensor data: 2 x F32 = 8 bytes.
    WriteLE<float>(out, 1.5f);
    WriteLE<float>(out, -2.5f);
    // Pad up to the 'quant' tensor's offset (kDefaultAlignment from
    // data_start).
    uint64_t cur = static_cast<uint64_t>(out.tellp()) - data_start;
    for (uint64_t i = cur; i < kDefaultAlignment; ++i) out.put('\0');
    // 'quant' tensor's Q4_0 block data: 18 bytes/block * 1 block for 32
    // elements (block size 32) -- exact bytes don't matter, just presence.
    for (int i = 0; i < 18; ++i) out.put('\xAB');
  }

  TensorPool pool;
  auto skipped = pool.LoadGGUF(path);
  Check(skipped.size() == 1 && skipped[0] == "quant",
        "the quantized tensor is skipped and reported by name");
  Check(pool.size() == 1, "only the raw tensor made it into the pool");
  const Entry* raw = pool.Find("raw");
  Check(raw != nullptr && raw->dtype == ONNX_FLOAT &&
            raw->shape == std::vector<int64_t>({2}),
        "the raw tensor itself loaded correctly despite its sibling being "
        "skipped");
  if (raw != nullptr) {
    float v0, v1;
    std::memcpy(&v0, raw->data.data(), 4);
    std::memcpy(&v1, raw->data.data() + 4, 4);
    if constexpr (std::endian::native == std::endian::big) {
      auto swap4 = [](float f) {
        unsigned char b[4];
        std::memcpy(b, &f, 4);
        std::reverse(std::begin(b), std::end(b));
        std::memcpy(&f, b, 4);
        return f;
      };
      v0 = swap4(v0);
      v1 = swap4(v1);
    }
    Check(v0 == 1.5f && v1 == -2.5f, "the raw tensor's bytes are correct");
  }
  Check(pool.Find("quant") == nullptr,
        "the skipped tensor is definitely not silently present with garbage "
        "data");
  std::remove(path.c_str());
}

void TestLoadGGUFMmapRoundTrip() {
  // Mirrors TestRoundTrip, but through the mmap-backed loader -- same
  // dimension-order and byte-content checks, to confirm the mapped path
  // parses identically to the seek+read path, not just that it doesn't
  // crash.
  TensorPool pool;
  pool.Add("weight", ONNX_FLOAT, {2, 3}, std::string(24, '\x11'));
  pool.Add("bias", ONNX_INT64, {4}, std::string(32, '\x22'));
  pool.Add("scalar", ONNX_INT8, {}, std::string(1, '\x33'));

  std::string path = TempPath("_mmap_roundtrip.gguf");
  pool.SaveGGUF(path, {{"general.architecture", "onnxsim-test"}});

  TensorPool loaded;
  auto skipped = loaded.LoadGGUFMmap(path);
  Check(skipped.empty(), "mmap: nothing skipped for an all-raw-dtype pool");
  Check(loaded.size() == 3, "mmap: all three tensors loaded");

  const Entry* w = loaded.Find("weight");
  Check(w != nullptr, "mmap: weight present");
  if (w != nullptr) {
    Check(w->dtype == ONNX_FLOAT, "mmap: weight dtype round-trips");
    Check(w->shape == std::vector<int64_t>({2, 3}),
          "mmap: weight shape round-trips in the RIGHT order (not "
          "transposed to {3, 2})");
    Check(w->data == std::string(24, '\x11'), "mmap: weight bytes round-trip");
  }

  const Entry* b = loaded.Find("bias");
  Check(b != nullptr && b->dtype == ONNX_INT64 &&
            b->shape == std::vector<int64_t>({4}) &&
            b->data == std::string(32, '\x22'),
        "mmap: bias round-trips");

  const Entry* s = loaded.Find("scalar");
  Check(s != nullptr && s->shape.empty() && s->data == std::string(1, '\x33'),
        "mmap: rank-0 (scalar) tensor round-trips");

  std::remove(path.c_str());
}

void TestLoadGGUFMmapSkipsUnsupportedDtype() {
  // Same hand-built file as TestSkipsUnsupportedDtype (one raw F32 tensor,
  // one made-up quantized type this pool can't represent), loaded through
  // LoadGGUFMmap -- confirms the mapped path skips-and-reports exactly like
  // the seek+read path rather than e.g. faulting trying to decode it.
  std::string path = TempPath("_mmap_quantized.gguf");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    auto write_string = [&](const std::string& s) {
      WriteLE<uint64_t>(out, s.size());
      out.write(s.data(), static_cast<std::streamsize>(s.size()));
    };

    WriteLE<uint32_t>(out, kMagic);
    WriteLE<uint32_t>(out, kSupportedVersion);
    WriteLE<uint64_t>(out, 2);  // tensor_count
    WriteLE<uint64_t>(out, 0);  // metadata_kv_count

    write_string("raw");
    WriteLE<uint32_t>(out, 1);
    WriteLE<uint64_t>(out, 2);
    WriteLE<uint32_t>(out, GGML_TYPE_F32);
    WriteLE<uint64_t>(out, 0);

    write_string("quant");
    WriteLE<uint32_t>(out, 1);
    WriteLE<uint64_t>(out, 32);
    WriteLE<uint32_t>(out, 2);  // GGML_TYPE_Q4_0
    WriteLE<uint64_t>(out, kDefaultAlignment);

    uint64_t header_end = static_cast<uint64_t>(out.tellp());
    uint64_t data_start = (header_end + kDefaultAlignment - 1) /
                          kDefaultAlignment * kDefaultAlignment;
    for (uint64_t i = header_end; i < data_start; ++i) out.put('\0');

    WriteLE<float>(out, 1.5f);
    WriteLE<float>(out, -2.5f);
    uint64_t cur = static_cast<uint64_t>(out.tellp()) - data_start;
    for (uint64_t i = cur; i < kDefaultAlignment; ++i) out.put('\0');
    for (int i = 0; i < 18; ++i) out.put('\xAB');
  }

  TensorPool pool;
  auto skipped = pool.LoadGGUFMmap(path);
  Check(skipped.size() == 1 && skipped[0] == "quant",
        "mmap: the quantized tensor is skipped and reported by name");
  Check(pool.size() == 1, "mmap: only the raw tensor made it into the pool");
  const Entry* raw = pool.Find("raw");
  Check(raw != nullptr && raw->dtype == ONNX_FLOAT &&
            raw->shape == std::vector<int64_t>({2}),
        "mmap: the raw tensor itself loaded correctly despite its sibling "
        "being skipped");
  Check(pool.Find("quant") == nullptr,
        "mmap: the skipped tensor is definitely not silently present with "
        "garbage data");
  std::remove(path.c_str());
}

void TestLoadGGUFMmapRejectsBadMagic() {
  std::string path = TempPath("_mmap_bad_magic.gguf");
  std::ofstream out(path, std::ios::binary);
  WriteLE<uint32_t>(out, 0xDEADBEEF);
  WriteLE<uint32_t>(out, kSupportedVersion);
  out.close();
  TensorPool pool;
  bool threw = false;
  try {
    pool.LoadGGUFMmap(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "mmap: bad magic is rejected");
  std::remove(path.c_str());
}

void TestLoadGGUFMmapMissingFileFallsBackAndThrows() {
  // No such file: TryMmapFile fails, so this exercises the fallback to
  // LoadGGUF, which should throw its own "cannot open" error rather than
  // silently succeeding.
  TensorPool pool;
  bool threw = false;
  try {
    pool.LoadGGUFMmap(TempPath("_mmap_missing.gguf"));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "mmap: loading a missing file throws (via read fallback)");
}

void TestRejectsBadMagicAndVersion() {
  {
    std::string path = TempPath("_bad_magic.gguf");
    std::ofstream out(path, std::ios::binary);
    WriteLE<uint32_t>(out, 0xDEADBEEF);
    WriteLE<uint32_t>(out, kSupportedVersion);
    out.close();
    TensorPool pool;
    bool threw = false;
    try {
      pool.LoadGGUF(path);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "bad magic is rejected");
    std::remove(path.c_str());
  }
  {
    std::string path = TempPath("_bad_version.gguf");
    std::ofstream out(path, std::ios::binary);
    WriteLE<uint32_t>(out, kMagic);
    WriteLE<uint32_t>(out, 1);  // unsupported version
    out.close();
    TensorPool pool;
    bool threw = false;
    try {
      pool.LoadGGUF(path);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "unsupported version is rejected");
    std::remove(path.c_str());
  }
}

void TestRejectsUnrepresentableDtype() {
  // TensorPool never actually holds a dtype gguf can't represent (see
  // gguf_dtype.h), so this exercises SaveGGUF's defensive check via a
  // dtype value that isn't a real OnnxDtype at all.
  TensorPool pool;
  pool.Add("bad", 8 /* ONNX STRING */, {1}, std::string("x"));
  std::string path = TempPath("_bad_dtype.gguf");
  bool threw = false;
  try {
    pool.SaveGGUF(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "saving a dtype with no raw ggml_type throws");
}

void TestKQuantRoundTripAndDecode() {
  // A Q8_0 block: d = 2.0 (fp16 0x4000 -- sign 0, biased exponent 16, zero
  // mantissa), qs[i] = i - 16 -- same vector ggml_kquant_test.cpp
  // hand-verifies, reused here to check TensorPool's own plumbing (pooling,
  // SaveGGUF/LoadGGUF byte-exact round trip, DequantizeToFloat), not the
  // decode math itself.
  const uint16_t d_bits = 0x4000;
  std::string block(34, '\0');
  block[0] = static_cast<char>(d_bits & 0xFF);
  block[1] = static_cast<char>((d_bits >> 8) & 0xFF);
  for (int i = 0; i < 32; ++i) {
    block[2 + i] = static_cast<char>(static_cast<int8_t>(i - 16));
  }

  TensorPool pool;
  pool.Add("w", ONNXSIM_GGML_Q8_0, {32}, std::string(block));

  std::string path = TempPath("_kquant.gguf");
  pool.SaveGGUF(path);

  TensorPool loaded;
  auto skipped = loaded.LoadGGUF(path);
  Check(skipped.empty(), "K-quant tensor is not skipped");
  const Entry* w = loaded.Find("w");
  Check(w != nullptr, "K-quant tensor is present");
  if (w != nullptr) {
    Check(w->dtype == ONNXSIM_GGML_Q8_0, "K-quant dtype round-trips");
    Check(w->shape == std::vector<int64_t>({32}), "K-quant shape round-trips");
    Check(w->data == block, "K-quant native block bytes round-trip bit-exact");

    std::vector<float> decoded;
    Check(loaded.DequantizeToFloat("w", &decoded),
          "DequantizeToFloat succeeds for a K-quant entry");
    Check(decoded.size() == 32, "DequantizeToFloat output size");
    bool values_ok = decoded.size() == 32;
    for (size_t i = 0; values_ok && i < 32; ++i) {
      values_ok =
          std::fabs(decoded[i] - static_cast<float>(static_cast<int>(i) - 16) *
                                     2.0f) < 1e-4f;
    }
    Check(values_ok, "DequantizeToFloat decodes to (i - 16) * 2.0");
  }

  // DequantizeToFloat is specifically for K-quant entries -- false (not a
  // silent raw copy) for an ordinary raw-dtype one.
  std::vector<float> not_kquant;
  pool.Add("raw", ONNX_FLOAT, {1}, std::string(4, '\0'));
  Check(!pool.DequantizeToFloat("raw", &not_kquant),
        "DequantizeToFloat rejects a raw-dtype entry");
  Check(!loaded.DequantizeToFloat("missing", &not_kquant),
        "DequantizeToFloat rejects a name not in the pool");

  std::remove(path.c_str());
}

void TestLoadGGUFRejectsMisalignedKQuantTensor() {
  // A Q8_0 tensor whose element count (5) is not a multiple of its block
  // size (32) -- malformed; must throw rather than silently truncate or
  // misread the following tensor's bytes.
  std::string path = TempPath("_kquant_misaligned.gguf");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    auto write_string = [&](const std::string& s) {
      WriteLE<uint64_t>(out, s.size());
      out.write(s.data(), static_cast<std::streamsize>(s.size()));
    };
    WriteLE<uint32_t>(out, kMagic);
    WriteLE<uint32_t>(out, kSupportedVersion);
    WriteLE<uint64_t>(out, 1);  // tensor_count
    WriteLE<uint64_t>(out, 0);  // metadata_kv_count

    write_string("bad");
    WriteLE<uint32_t>(out, 1);
    WriteLE<uint64_t>(out, 5);  // ne[0] -- not a multiple of 32
    WriteLE<uint32_t>(out, GGML_TYPE_Q8_0);
    WriteLE<uint64_t>(out, 0);  // offset
  }

  TensorPool pool;
  bool threw = false;
  try {
    pool.LoadGGUF(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "loading a non-block-aligned K-quant tensor throws");
  std::remove(path.c_str());
}

}  // namespace

int main() {
  TestRoundTrip();
  TestEmptyMetadataOmitted();
  TestAlignmentPadding();
  TestSkipsUnsupportedDtype();
  TestRejectsBadMagicAndVersion();
  TestRejectsUnrepresentableDtype();
  TestLoadGGUFMmapRoundTrip();
  TestLoadGGUFMmapSkipsUnsupportedDtype();
  TestLoadGGUFMmapRejectsBadMagic();
  TestLoadGGUFMmapMissingFileFallsBackAndThrows();
  TestKQuantRoundTripAndDecode();
  TestLoadGGUFRejectsMisalignedKQuantTensor();

  if (g_failures == 0) {
    std::printf("tensor_pool_gguf_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_gguf_test: %d failure(s)\n", g_failures);
  return 1;
}
