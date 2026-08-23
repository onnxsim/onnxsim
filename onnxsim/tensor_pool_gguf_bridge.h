/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * GGUF counterpart of tensor_pool_bridge.h's ExportModelWithSafetensors /
 * ImportModelWithSafetensors -- same design, a different container format.
 * See that header's top comment for the shared rationale (moving an
 * initializer's raw_data into a pool with no copy, then patching in a real,
 * spec-compliant onnx EXTERNAL reference once the file's offsets are known)
 * and its NOT-covered-here note, which applies equally: this is meant for
 * the load/save boundary, not as onnxsim's internal in-flight
 * representation during Simplify().
 *
 * The GGUF-specific thing worth calling out here is gguf_dtype.h's scope:
 * most of GGML's block-quantized types (Q4_0, every IQ*_ variant, ...) have
 * no ONNX raw-data equivalent at all, so ImportModelWithGGUF's
 * `skipped_out` matters more here than ImportModelWithSafetensors's
 * equivalent ever would in practice. The K-quant family this mapping DOES
 * cover (Q4_K/Q5_K/Q6_K/Q8_0 -- what a real quantized checkpoint, e.g.
 * Unsloth's GGUF exports, actually uses for the bulk of its weights) is
 * hydrated as ordinary float32, via HydrateTensorProtoFromGGUF's decode --
 * ImportModelWithGGUF is the intended way to pull a third-party GGUF
 * checkpoint's weight *values* into an ONNX graph you already have, by
 * initializer name (it needs no embedded onnxsim model, unlike
 * LoadModelFromGGUF below). Still, a checkpoint using one of the
 * unsupported quantized families will leave the matching initializers
 * un-hydrated, and a caller needs to know which, rather than silently
 * getting a model that's missing some of its weights.
 */
#ifndef ONNXSIM_TENSOR_POOL_GGUF_BRIDGE_H_
#define ONNXSIM_TENSOR_POOL_GGUF_BRIDGE_H_

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "dlpack_dtype.h"
#include "gguf_dtype.h"
#include "tensor_pool.h"
// Reuses AdoptFromTensorProto / HydrateTensorProto / detail::ForEachTensor --
// none of that is safetensors-specific, so there is nothing GGUF-specific to
// redefine here.
#include "tensor_pool_bridge.h"

namespace onnxsim {
namespace tensor_pool {

// GGUF counterpart of tensor_pool_bridge.h's HydrateTensorProto: same
// contract (returns false, leaving `tensor` untouched, if `name` isn't
// pooled) EXCEPT a K-quant-native entry (Q4_K/Q5_K/Q6_K/Q8_0 -- see
// gguf_dtype.h's IsKQuant) is decoded to plain float32 raw_data instead of
// copied verbatim, and `tensor`'s data_type is forced to FLOAT to match --
// overriding whatever data_type the caller's initializer previously
// declared, since the decoded values are only meaningful as float32 (the
// caller's own declared dtype reflects whatever *their* graph expects,
// which has no bearing on what a third-party GGUF file's weights actually
// are once dequantized). An ordinary raw-dtype entry (FLOAT, FLOAT16, ...)
// is hydrated exactly like HydrateTensorProto (data_type left as the
// caller's initializer already declared).
inline bool HydrateTensorProtoFromGGUF(const std::string& name,
                                       onnx::TensorProto& tensor,
                                       const TensorPool& pool) {
  const Entry* entry = pool.Find(name);
  if (entry == nullptr) return false;

  uint32_t ggml_type;
  if (gguf::FromOnnx(entry->dtype, &ggml_type) && gguf::IsKQuant(ggml_type)) {
    std::vector<float> floats;
    if (!pool.DequantizeToFloat(name, &floats)) {
      return false;  // Unreachable: FromOnnx+IsKQuant already validated dtype.
    }
    tensor.set_data_location(onnx::TensorProto::DEFAULT);
    tensor.clear_external_data();
    tensor.set_data_type(onnx::TensorProto::FLOAT);
    // TensorProto::raw_data is always little-endian on the wire (see
    // tensor_pool.h's file comment); `floats` holds true host-order values
    // (DequantizeToFloat already did the little-endian-safe decode), so
    // this needs the same conditional swap passes/endian_read.h's
    // ReadRawDataHostOrder does in the opposite direction.
    std::string raw(floats.size() * sizeof(float), '\0');
    std::memcpy(raw.data(), floats.data(), raw.size());
    if constexpr (!onnxsim::dlpack::kRawDataIsHostOrder) {
      onnxsim::dlpack::SwapElementBytes(reinterpret_cast<uint8_t*>(raw.data()),
                                        raw.size(), sizeof(float));
    }
    tensor.set_raw_data(std::move(raw));
    return true;
  }
  return HydrateTensorProto(name, tensor, pool);
}

// Rewrite `model` so every eligible tensor becomes an EXTERNAL reference into
// a freshly-written `gguf_path`, and write that file. Returns the number of
// tensors externalized. `pool` ends up holding every externalized tensor.
// `string_metadata` is forwarded to TensorPool::SaveGGUF verbatim (e.g.
// {"general.architecture", "onnxsim"}).
inline size_t ExportModelWithGGUF(
    onnx::ModelProto& model, const std::string& gguf_path, TensorPool& pool,
    const std::map<std::string, std::string>& string_metadata = {}) {
  // See ExportModelWithSafetensors's identical comment: keyed by the pool
  // key ForEachTensor assigned, not by t->name(), which may be empty for an
  // unnamed attribute tensor.
  std::vector<std::pair<std::string, onnx::TensorProto*>> exported;
  detail::ForEachTensor(*model.mutable_graph(),
                        [&](const std::string& name, onnx::TensorProto& t) {
                          if (AdoptFromTensorProto(name, t, pool)) {
                            exported.emplace_back(name, &t);
                          }
                        });

  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  uint64_t data_section_start = 0;
  pool.SaveGGUF(gguf_path, string_metadata, &offsets, &data_section_start);

  for (auto& [pool_key, t] : exported) {
    const auto& [begin, end] = offsets.at(pool_key);
    t->set_data_location(onnx::TensorProto::EXTERNAL);
    t->clear_external_data();
    auto set_kv = [&](const std::string& k, const std::string& v) {
      auto* e = t->add_external_data();
      e->set_key(k);
      e->set_value(v);
    };
    set_kv("location", gguf_path);
    set_kv("offset", std::to_string(data_section_start + begin));
    set_kv("length", std::to_string(end - begin));
  }
  return exported.size();
}

// Load `gguf_path` into `pool` and, for every graph initializer whose name
// matches a pooled tensor -- a raw-dtype tensor, or a K-quant one (see
// gguf_dtype.h's IsKQuant), either hydrate it in place (hydrate_all, the
// default -- a K-quant match is decoded to float32, see
// HydrateTensorProtoFromGGUF) or leave it as a lazy EXTERNAL reference
// (hydrate_all=false; see tensor_pool_bridge.h's caveat on this mode --
// note a K-quant match left lazy this way still needs
// HydrateTensorProtoFromGGUF, not the plain HydrateTensorProto that caveat
// points to, when it's eventually hydrated on demand).
// Returns the number of initializers matched (hydrated or marked lazy).
//
// `model`'s own graph is exactly what determines which of the file's
// tensors get pulled in -- this is the intended way to bring a third-party
// GGUF checkpoint's weight *values* into an ONNX graph you already have
// (e.g. exported by another tool), by initializer name: build/load that
// graph first, then call this. It does NOT require (or use) an embedded
// onnxsim model the way ImportModelWithSafetensors/LoadModelFromGGUF do --
// see this header's file comment.
//
// When `skipped_out` is non-null, it receives the names of GGUF tensors
// that were present in the file but skipped because their ggml_type has no
// representation TensorPool can hold at all (TensorPool::LoadGGUF's own
// return value, passed through) -- check this against your model's
// initializer names to find out which weights this import could NOT bring
// in, rather than discovering it only when a later pass trips over an
// initializer that's still empty.
inline size_t ImportModelWithGGUF(
    onnx::ModelProto& model, const std::string& gguf_path, TensorPool& pool,
    bool hydrate_all = true, std::vector<std::string>* skipped_out = nullptr) {
  std::vector<std::string> skipped = pool.LoadGGUF(gguf_path);
  if (skipped_out != nullptr) *skipped_out = std::move(skipped);

  size_t matched = 0;
  for (auto& init : *model.mutable_graph()->mutable_initializer()) {
    const Entry* entry = pool.Find(init.name());
    if (entry == nullptr) continue;
    ++matched;
    if (hydrate_all) {
      HydrateTensorProtoFromGGUF(init.name(), init, pool);
    } else {
      init.set_data_location(onnx::TensorProto::EXTERNAL);
      init.clear_external_data();
      auto* e = init.add_external_data();
      e->set_key("location");
      e->set_value(gguf_path);
    }
  }
  return matched;
}

// GGUF counterparts of tensor_pool_bridge.h's SaveModelAsSafetensors /
// LoadModelFromSafetensors -- see that header's top comment for the
// self-describing-archive design (EmbedModel/ExtractModel, which these
// reuse unchanged; embedding a model doesn't care which codec eventually
// serializes the pool).

// Convenience wrapper: EmbedModel then pool.SaveGGUF(path, string_metadata).
// `model` ends up mutated exactly as EmbedModel leaves it -- reload with
// LoadModelFromGGUF rather than reusing `model`.
inline void SaveModelAsGGUF(
    onnx::ModelProto& model, const std::string& path, TensorPool& pool,
    const std::map<std::string, std::string>& string_metadata = {}) {
  EmbedModel(model, path, pool);
  pool.SaveGGUF(path, string_metadata);
}

// Convenience wrapper: pool.LoadGGUF(path) then ExtractModel. Returns false
// if `path`'s pool has no embedded model. `skipped_out`, when non-null,
// receives the names of any OTHER pooled tensors skipped as quantized/
// unsupported (kEmbeddedModelKey itself is always a raw I8 blob, so it is
// never one of them) -- same meaning as ImportModelWithGGUF's.
inline bool LoadModelFromGGUF(const std::string& path, onnx::ModelProto* model,
                              TensorPool& pool, bool hydrate_all = true,
                              std::vector<std::string>* skipped_out = nullptr) {
  std::vector<std::string> skipped = pool.LoadGGUF(path);
  if (skipped_out != nullptr) *skipped_out = std::move(skipped);
  return ExtractModel(pool, model, hydrate_all);
}

// GGUF counterpart of tensor_pool_bridge.h's SaveModelAsSafetensorsStandalone
// -- see that function's comment (and the "standalone archives" section
// above it) for the real-offset design and why every weight tensor is still
// written to disk exactly once despite the model blob needing two passes.
inline void SaveModelAsGGUFStandalone(
    onnx::ModelProto& model, const std::string& path, TensorPool& pool,
    const std::map<std::string, std::string>& string_metadata = {}) {
  auto adopted = AdoptAllWithPlaceholderOffsets(model, path, pool);
  CheckNoEmbeddedModelKeyCollision(pool);

  std::string placeholder_bytes;
  if (!model.SerializeToString(&placeholder_bytes)) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsGGUFStandalone: failed to serialize the "
        "model");
  }
  pool.Add(kEmbeddedModelKey, ONNX_INT8,
           {static_cast<int64_t>(placeholder_bytes.size())},
           std::string(placeholder_bytes));

  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  uint64_t data_section_start = 0;
  pool.SaveGGUF(path, string_metadata, &offsets, &data_section_start);

  PatchStandaloneOffsets(adopted, offsets, data_section_start);

  std::string final_bytes;
  if (!model.SerializeToString(&final_bytes)) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsGGUFStandalone: failed to re-serialize the "
        "model");
  }
  if (final_bytes.size() != placeholder_bytes.size()) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsGGUFStandalone: internal error -- "
        "re-serializing with real offsets changed the model's size (" +
        std::to_string(placeholder_bytes.size()) + " -> " +
        std::to_string(final_bytes.size()) + " bytes)");
  }

  const auto& model_range = offsets.at(kEmbeddedModelKey);
  PatchFileBytes(path, data_section_start + model_range.first, final_bytes);
  pool.Add(kEmbeddedModelKey, ONNX_INT8,
           {static_cast<int64_t>(final_bytes.size())}, std::move(final_bytes));
}

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_GGUF_BRIDGE_H_
