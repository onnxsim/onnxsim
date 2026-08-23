/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises tensor_pool_gguf_bridge.h's onnx::TensorProto <-> TensorPool GGUF
 * glue, mirroring tensor_pool_bridge_test.cpp. Needs onnx configured, so it's
 * built against the fully-configured `onnxsim` CMake target rather than
 * standalone (same as tensor_pool_bridge_test.cpp).
 */
#include "tensor_pool_gguf_bridge.h"

#include <onnx/onnx_pb.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "gguf_dtype.h"

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
  return "/tmp/onnxsim_tensor_pool_gguf_bridge_test_" +
         std::to_string(::getpid()) + suffix;
}

std::string ReadFileRange(const std::string& path, uint64_t offset,
                          uint64_t length) {
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(offset));
  std::string out(length, '\0');
  in.read(out.data(), static_cast<std::streamsize>(length));
  return out;
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

onnx::TensorProto MakeInitializer(const std::string& name,
                                  onnx::TensorProto::DataType dtype,
                                  const std::vector<int64_t>& dims,
                                  const std::string& raw) {
  onnx::TensorProto t;
  t.set_name(name);
  t.set_data_type(dtype);
  for (int64_t d : dims) t.add_dims(d);
  t.set_raw_data(raw);
  return t;
}

void TestExportModelWithGGUF() {
  onnx::ModelProto model;
  auto* graph = model.mutable_graph();
  // Non-square shape, same reasoning as tensor_pool_gguf_test.cpp's
  // TestRoundTrip: catches a ne[]-vs-shape order bug that byte comparison
  // alone would miss.
  *graph->add_initializer() = MakeInitializer(
      "weight", onnx::TensorProto::FLOAT, {2, 3}, std::string(24, '\x11'));
  *graph->add_initializer() = MakeInitializer("bias", onnx::TensorProto::INT64,
                                              {2}, std::string(16, '\x22'));

  auto* node = graph->add_node();
  node->set_name("const_node");
  auto* attr = node->add_attribute();
  attr->set_name("value");
  *attr->mutable_t() = MakeInitializer("const_tensor", onnx::TensorProto::INT8,
                                       {4}, std::string(4, '\x33'));

  std::string path = TempPath("_export.gguf");
  TensorPool pool;
  size_t n = ExportModelWithGGUF(model, path, pool,
                                 {{"general.architecture", "onnxsim-test"}});
  Check(n == 3, "all three tensors exported");

  for (const auto& name : {"weight", "bias", "const_tensor"}) {
    const onnx::TensorProto* t = nullptr;
    for (const auto& init : graph->initializer()) {
      if (init.name() == name) t = &init;
    }
    if (t == nullptr && graph->node(0).attribute(0).t().name() == name) {
      t = &graph->node(0).attribute(0).t();
    }
    Check(t != nullptr, std::string(name) + ": found in model");
    if (t == nullptr) continue;
    Check(t->data_location() == onnx::TensorProto::EXTERNAL,
          std::string(name) + ": is EXTERNAL after export");
    Check(!t->has_raw_data(),
          std::string(name) + ": no inline raw_data left after export");

    std::string location, offset_str, length_str;
    for (const auto& e : t->external_data()) {
      if (e.key() == "location") location = e.value();
      if (e.key() == "offset") offset_str = e.value();
      if (e.key() == "length") length_str = e.value();
    }
    Check(location == path, std::string(name) + ": location == gguf path");
    Check(!offset_str.empty() && !length_str.empty(),
          std::string(name) + ": offset/length are set");

    uint64_t offset = std::stoull(offset_str);
    uint64_t length = std::stoull(length_str);
    std::string on_disk = ReadFileRange(path, offset, length);
    const Entry* pooled = pool.Find(name);
    Check(pooled != nullptr && on_disk == pooled->data,
          std::string(name) +
              ": bytes at the recorded external_data offset match the pool");
  }

  // Independently loadable as GGUF with no onnxsim/onnx involvement.
  TensorPool reloaded;
  Check(reloaded.LoadGGUF(path).empty() && reloaded.size() == 3,
        "the exported file is independently loadable as GGUF");
  const Entry* w = reloaded.Find("weight");
  Check(w != nullptr && w->shape == std::vector<int64_t>({2, 3}),
        "the reloaded 'weight' tensor kept its shape (not transposed)");

  std::remove(path.c_str());
}

void TestUnnamedAttributeTensor() {
  // Same regression as tensor_pool_bridge_test.cpp's test of the same name:
  // an attribute tensor with no .name() must still round-trip, keyed by
  // ForEachTensor's positional fallback.
  onnx::ModelProto model;
  auto* node = model.mutable_graph()->add_node();
  node->set_name("n0");
  auto* attr = node->add_attribute();
  attr->set_name("value");
  auto* t = attr->mutable_t();
  t->set_data_type(onnx::TensorProto::FLOAT);
  t->add_dims(1);
  t->set_raw_data(std::string(4, '\x77'));

  std::string path = TempPath("_unnamed_attr.gguf");
  TensorPool pool;
  size_t n = ExportModelWithGGUF(model, path, pool);
  Check(n == 1, "the unnamed attribute tensor is exported");

  const auto& exported_t = model.graph().node(0).attribute(0).t();
  Check(exported_t.data_location() == onnx::TensorProto::EXTERNAL,
        "unnamed attribute tensor is EXTERNAL after export");
  bool has_offset = false, has_length = false;
  for (const auto& e : exported_t.external_data()) {
    has_offset |= (e.key() == "offset" && !e.value().empty());
    has_length |= (e.key() == "length" && !e.value().empty());
  }
  Check(has_offset && has_length,
        "unnamed attribute tensor got real offset/length, not left "
        "half-adopted");
  std::remove(path.c_str());
}

void TestImportModelWithGGUFHydrateAll() {
  onnx::ModelProto original;
  auto* g = original.mutable_graph();
  *g->add_initializer() = MakeInitializer("w1", onnx::TensorProto::FLOAT, {3},
                                          std::string(12, '\x44'));
  *g->add_initializer() = MakeInitializer("w2", onnx::TensorProto::INT32, {2},
                                          std::string(8, '\x55'));

  std::string path = TempPath("_import.gguf");
  TensorPool export_pool;
  ExportModelWithGGUF(original, path, export_pool);

  onnx::ModelProto fresh;
  auto* fg = fresh.mutable_graph();
  auto* w1 = fg->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(onnx::TensorProto::FLOAT);
  w1->add_dims(3);
  auto* w2 = fg->add_initializer();
  w2->set_name("w2");
  w2->set_data_type(onnx::TensorProto::INT32);
  w2->add_dims(2);

  TensorPool import_pool;
  std::vector<std::string> skipped;
  size_t matched =
      ImportModelWithGGUF(fresh, path, import_pool, true, &skipped);
  Check(matched == 2, "both initializers matched on import");
  Check(skipped.empty(), "nothing skipped -- every tensor was a raw dtype");

  Check(fg->initializer(0).has_raw_data() &&
            fg->initializer(0).raw_data() == std::string(12, '\x44'),
        "w1 hydrated to the original bytes");
  Check(fg->initializer(0).data_location() == onnx::TensorProto::DEFAULT,
        "w1 is DEFAULT-located after hydration");
  Check(fg->initializer(1).has_raw_data() &&
            fg->initializer(1).raw_data() == std::string(8, '\x55'),
        "w2 hydrated to the original bytes");

  std::remove(path.c_str());
}

void TestImportModelWithGGUFLazy() {
  onnx::ModelProto original;
  *original.mutable_graph()->add_initializer() = MakeInitializer(
      "lazy", onnx::TensorProto::FLOAT, {1}, std::string(4, '\x66'));
  std::string path = TempPath("_lazy.gguf");
  TensorPool export_pool;
  ExportModelWithGGUF(original, path, export_pool);

  onnx::ModelProto fresh;
  auto* init = fresh.mutable_graph()->add_initializer();
  init->set_name("lazy");
  init->set_data_type(onnx::TensorProto::FLOAT);
  init->add_dims(1);

  TensorPool import_pool;
  size_t matched = ImportModelWithGGUF(fresh, path, import_pool,
                                       /*hydrate_all=*/false);
  Check(matched == 1, "lazy import still reports the match");
  Check(!fresh.graph().initializer(0).has_raw_data(),
        "lazy import leaves raw_data unset");
  Check(fresh.graph().initializer(0).data_location() ==
            onnx::TensorProto::EXTERNAL,
        "lazy import marks the tensor EXTERNAL for on-demand hydration");

  bool hydrated = HydrateTensorProto(
      "lazy", *fresh.mutable_graph()->mutable_initializer(0), import_pool);
  Check(hydrated &&
            fresh.graph().initializer(0).raw_data() == std::string(4, '\x66'),
        "on-demand HydrateTensorProto recovers the bytes");

  std::remove(path.c_str());
}

// Hand-builds a GGUF file with one raw F32 tensor ("w1") and one Q4_0
// (quantized) tensor ("w2"), then imports it against a model declaring both
// as initializers -- the scenario a real quantized-LLM checkpoint puts a
// caller in for nearly every weight.
void TestImportModelWithGGUFSkipsQuantized() {
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

    write_string("w1");
    WriteLE<uint32_t>(out, 1);
    WriteLE<uint64_t>(out, 2);  // ne[0]
    WriteLE<uint32_t>(out, GGML_TYPE_F32);
    WriteLE<uint64_t>(out, 0);  // offset

    write_string("w2");
    WriteLE<uint32_t>(out, 1);
    WriteLE<uint64_t>(out, 32);  // ne[0]
    WriteLE<uint32_t>(out, 2);   // GGML_TYPE_Q4_0
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

  onnx::ModelProto model;
  auto* w1 = model.mutable_graph()->add_initializer();
  w1->set_name("w1");
  w1->set_data_type(onnx::TensorProto::FLOAT);
  w1->add_dims(2);
  auto* w2 = model.mutable_graph()->add_initializer();
  w2->set_name("w2");
  w2->set_data_type(onnx::TensorProto::FLOAT);  // placeholder; never filled
  w2->add_dims(32);

  TensorPool pool;
  std::vector<std::string> skipped;
  size_t matched = ImportModelWithGGUF(model, path, pool, true, &skipped);
  Check(matched == 1, "only w1 (the raw tensor) is matched");
  Check(skipped.size() == 1 && skipped[0] == "w2",
        "w2 is reported as skipped, so a caller can tell it's missing "
        "weights rather than silently getting a half-loaded model");
  Check(model.graph().initializer(0).has_raw_data(), "w1 was hydrated");
  Check(!model.graph().initializer(1).has_raw_data(),
        "w2 was left untouched -- definitely not filled with garbage bytes "
        "reinterpreted from its quantized encoding");

  std::remove(path.c_str());
}

// Hand-builds a GGUF file with one Q8_0 tensor (d = 2.0, qs[i] = i - 16 --
// the same hand-verified vector ggml_kquant_test.cpp/tensor_pool_gguf_test.cpp
// use), imported against a model whose matching initializer is declared as
// INT32 -- deliberately the WRONG dtype, to confirm HydrateTensorProtoFromGGUF
// forces it to FLOAT (the only meaningful type for a dequantized result)
// rather than leaving whatever the caller's graph happened to declare.
void TestImportModelWithGGUFDecodesKQuant() {
  std::string path = TempPath("_kquant_import.gguf");
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

    write_string("w");
    WriteLE<uint32_t>(out, 1);
    WriteLE<uint64_t>(out, 32);  // ne[0]
    WriteLE<uint32_t>(out, GGML_TYPE_Q8_0);
    WriteLE<uint64_t>(out, 0);  // offset

    uint64_t header_end = static_cast<uint64_t>(out.tellp());
    uint64_t data_start = (header_end + kDefaultAlignment - 1) /
                          kDefaultAlignment * kDefaultAlignment;
    for (uint64_t i = header_end; i < data_start; ++i) out.put('\0');
    WriteLE<uint16_t>(out, 0x4000);  // d = 2.0 (fp16)
    for (int i = 0; i < 32; ++i) {
      out.put(static_cast<char>(static_cast<int8_t>(i - 16)));  // qs[i]
    }
  }

  onnx::ModelProto model;
  auto* w = model.mutable_graph()->add_initializer();
  w->set_name("w");
  w->set_data_type(onnx::TensorProto::INT32);  // deliberately wrong
  w->add_dims(32);

  TensorPool pool;
  std::vector<std::string> skipped;
  size_t matched = ImportModelWithGGUF(model, path, pool, true, &skipped);
  Check(matched == 1, "the K-quant tensor is matched");
  Check(skipped.empty(), "nothing skipped -- Q8_0 is a supported K-quant type");

  const auto& hydrated = model.graph().initializer(0);
  Check(hydrated.data_type() == onnx::TensorProto::FLOAT,
        "K-quant hydration forces data_type to FLOAT, overriding the "
        "caller's (wrong) declared INT32");
  Check(hydrated.has_raw_data() &&
            hydrated.raw_data().size() == 32 * sizeof(float),
        "hydrated raw_data is 32 float32 values");
  bool values_ok = hydrated.raw_data().size() == 32 * sizeof(float);
  if (values_ok) {
    float floats[32];
    std::memcpy(floats, hydrated.raw_data().data(), sizeof(floats));
    for (int i = 0; i < 32 && values_ok; ++i) {
      float want = static_cast<float>(i - 16) * 2.0f;
      values_ok = std::fabs(floats[i] - want) < 1e-4f;
    }
  }
  Check(values_ok, "hydrated values decode to (i - 16) * 2.0");

  std::remove(path.c_str());
}

}  // namespace

int main() {
  TestExportModelWithGGUF();
  TestUnnamedAttributeTensor();
  TestImportModelWithGGUFHydrateAll();
  TestImportModelWithGGUFLazy();
  TestImportModelWithGGUFSkipsQuantized();
  TestImportModelWithGGUFDecodesKQuant();

  if (g_failures == 0) {
    std::printf("tensor_pool_gguf_bridge_test: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "tensor_pool_gguf_bridge_test: %d failure(s)\n",
               g_failures);
  return 1;
}
