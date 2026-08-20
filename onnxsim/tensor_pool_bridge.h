/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bridges onnx::TensorProto <-> TensorPool, and ties TensorPool's
 * safetensors file format to onnx's own external-data mechanism
 * (TensorProto's data_location/external_data fields) so a model's weights
 * can live in one canonical, ecosystem-standard .safetensors file instead of
 * being embedded in the .onnx file itself or dumped in onnx's ad hoc raw
 * external-data format.
 *
 * Two directions:
 *   * ExportModelWithSafetensors -- turns an ordinary, fully in-memory model
 *     into one whose initializers are EXTERNAL references into a
 *     freshly-written safetensors file. Each initializer's raw_data is moved
 *     into the pool via TensorProto::release_raw_data() (no copy), then
 *     every reference is patched with the real, absolute file offset/length
 *     the tensor's bytes ended up at -- so the result is simultaneously a
 *     valid safetensors file (openable by the `safetensors` Python package,
 *     HF transformers/diffusers, etc. with no onnxsim involved) AND correct,
 *     spec-compliant classic onnx external data (openable by vanilla
 *     onnx/onnxruntime, which only look at offset/length and don't care
 *     what bytes -- here, the JSON header -- precede them in the file).
 *   * ImportModelWithSafetensors -- the reverse: loads a .safetensors file
 *     into a pool (one read, zero-copy views; see tensor_pool.h) and, by
 *     default, hydrates every matching initializer back to an ordinary
 *     in-memory (non-EXTERNAL) TensorProto, so the resulting model works
 *     with every existing onnxsim pass with no caveats. Hydration is one
 *     copy per tensor -- unavoidable, because onnx::TensorProto physically
 *     owns raw_data as a plain std::string (no borrowed/Cord-backed bytes
 *     type) -- but it's the *only* copy: the pool itself loads zero-copy,
 *     and only the tensors a caller actually asks to hydrate ever pay it
 *     (pass hydrate_all=false to leave the rest as lazy EXTERNAL pool
 *     references and hydrate individual ones on demand via
 *     HydrateTensorProto -- see that function's caveat first, though).
 *
 * A third pair, EmbedModel/ExtractModel plus the format-specific
 * SaveModelAsSafetensors/LoadModelFromSafetensors below, goes one step
 * further: instead of a small .onnx file with EXTERNAL references pointing
 * at a separate weights file, the graph ITSELF is embedded inside the
 * weights file too, as one more pooled entry (name "model.onnx", a raw
 * byte blob) alongside the tensors -- one self-describing archive, not two
 * files. The embedded copy's own initializers use a SYMBOLIC EXTERNAL
 * reference (location = the archive path, no offset/length -- the same
 * convention hydrate_all=false above already uses), resolved by name
 * lookup into the archive's own pool, not by byte offset. That sidesteps a
 * real chicken-and-egg problem real offsets would have here (the model's
 * serialized size depends on the offsets, which depend on where the model
 * blob itself lands in the file, which depends on the model's serialized
 * size) at the cost of the extracted blob not being standalone-loadable by
 * a vanilla onnx tool on its own -- only onnxsim's own
 * LoadModelFromSafetensors/LoadModelFromGGUF (via ExtractModel) know to
 * resolve it. Giving the embedded copy real, byte-accurate offsets instead
 * (so an extracted model.onnx is standalone-usable elsewhere too) is a
 * tracked follow-up, not implemented here.
 *
 * A fourth, independent piece, PoolExternalData, has nothing to do with the
 * safetensors/GGUF archive formats above: it pools a model's *existing*
 * classic onnx external data (the ordinary "<name>.onnx" + "<name>.data"
 * pair onnx.save(..., save_as_external_data=True) produces) by
 * memory-mapping the data file(s) instead of reading them into a heap
 * buffer -- see onnxsim.h's LoadModelPooled, which combines this with
 * onnx-optimizer's own loadModel to give onnxsim's `simplify()` its default
 * path-based model-loading step.
 *
 * NOT covered here: keeping initializers EXTERNAL as onnxsim's *internal*
 * in-flight representation during Simplify()'s fixed point, to dodge
 * ModelProto copy costs mid-pipeline. That would collide with existing logic
 * that already treats EXTERNAL as "we don't have the bytes, skip this
 * tensor" -- onnxsim.cpp's IsAllZeroTensor and its integer-tensor extraction
 * helper, and dlpack_bridge.h's FromTensorProtoBorrowing, all bail out on
 * EXTERNAL tensors today. Using TensorPool as a lazy, on-demand backing
 * store for those call sites (hydrate a tensor from the pool only when a
 * pass actually needs its bytes, instead of skipping it outright) is a
 * natural follow-up, but it means auditing and updating each of those call
 * sites, not just adding a pool -- so hydrate_all=false below is a low-level
 * building block for that future work, not something to wire into Simplify()
 * as-is.
 */
#ifndef ONNXSIM_TENSOR_POOL_BRIDGE_H_
#define ONNXSIM_TENSOR_POOL_BRIDGE_H_

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mmap_file.h"
#include "tensor_pool.h"
#include "tensor_pool_dtype.h"
#include "tensor_pool_hash.h"

namespace onnxsim {
namespace tensor_pool {

namespace detail {

template <typename Fn>
void ForEachTensor(onnx::GraphProto& graph, Fn&& fn);

// The parsed "location"/"offset"/"length" triple a classic-EXTERNAL
// TensorProto's external_data carries (onnx's own "raw external data in a
// file" convention -- the same one PoolExternalData below reads and
// ExportModelWithSafetensors's callers write). Reimplemented here, rather
// than reused from onnx-optimizer's own equivalent (model_util.cc's
// ExternalDataInfo), so this header keeps needing only onnx's protobuf
// headers -- see this header's top comment.
struct ExternalDataRef {
  std::string location;
  int64_t offset = -1;  // -1 means "from the start of the file"
  int64_t length = -1;  // -1 means "to the end of the file"

  explicit ExternalDataRef(const onnx::TensorProto& tensor) {
    for (const auto& e : tensor.external_data()) {
      if (e.key() == "location") {
        location = e.value();
      } else if (e.key() == "offset") {
        offset = std::stoll(e.value());
      } else if (e.key() == "length") {
        length = std::stoll(e.value());
      }
    }
  }
};

// Recurse into one node attribute: its own tensor (`t`), its repeated
// `tensors`, and any subgraph(s) it carries (`g` / `graphs`, e.g. an `If`
// branch or a `Loop` body). `path` is a stable, human-readable fallback name
// stem for tensors that have no `.name()` of their own (attribute tensors
// frequently don't).
template <typename Fn>
void ForEachTensorInAttr(onnx::AttributeProto& attr, const std::string& path,
                         Fn&& fn) {
  if (attr.has_t()) {
    auto* t = attr.mutable_t();
    fn(t->name().empty() ? path + "/t" : t->name(), *t);
  }
  for (int i = 0; i < attr.tensors_size(); ++i) {
    auto* t = attr.mutable_tensors(i);
    fn(t->name().empty() ? path + "/tensors" + std::to_string(i) : t->name(),
       *t);
  }
  if (attr.has_g()) ForEachTensor(*attr.mutable_g(), fn);
  for (auto& g : *attr.mutable_graphs()) ForEachTensor(g, fn);
}

// Recurse into every TensorProto a graph carries -- initializers and node
// attribute tensors, recursively through subgraphs -- calling fn(name,
// tensor) for each. `name` is the tensor's own `.name()` when non-empty,
// else a positional fallback, so every tensor gets a distinct pool key even
// when unnamed.
template <typename Fn>
void ForEachTensor(onnx::GraphProto& graph, Fn&& fn) {
  for (int i = 0; i < graph.initializer_size(); ++i) {
    auto* init = graph.mutable_initializer(i);
    fn(init->name().empty() ? "initializer" + std::to_string(i) : init->name(),
       *init);
  }
  for (int ni = 0; ni < graph.node_size(); ++ni) {
    auto* node = graph.mutable_node(ni);
    std::string node_path =
        node->name().empty() ? "node" + std::to_string(ni) : node->name();
    for (int ai = 0; ai < node->attribute_size(); ++ai) {
      ForEachTensorInAttr(*node->mutable_attribute(ai),
                          node_path + "/attr" + std::to_string(ai), fn);
    }
  }
}

}  // namespace detail

// Move `tensor`'s inline raw_data into `pool` under `name` via
// TensorProto::release_raw_data() (no copy). Returns false, leaving `tensor`
// untouched, when there's nothing eligible to move: no raw_data (a typed
// repeated field, or an empty tensor), already EXTERNAL, or STRING (no raw
// byte layout at all).
//
// CAUTION: on success, `tensor` is left in an intermediate, not-yet-
// externalized state -- its bytes are gone but data_location is still
// DEFAULT, not EXTERNAL -- because the pool doesn't know the tensor's final
// file offset until the whole pool has been written (offsets depend on every
// other tensor's size too). Callers must follow up by marking it EXTERNAL
// with the real location/offset/length once that's known; do not leave (or
// serialize) a model with tensors in this half-adopted state.
// ExportModelWithSafetensors below does this correctly; use it rather than
// calling AdoptFromTensorProto directly unless you need to compose your own
// export sequence.
inline bool AdoptFromTensorProto(const std::string& name,
                                 onnx::TensorProto& tensor, TensorPool& pool) {
  if (tensor.data_location() == onnx::TensorProto::EXTERNAL) return false;
  if (tensor.data_type() == onnx::TensorProto::STRING) return false;
  if (!tensor.has_raw_data()) return false;

  std::vector<int64_t> shape(tensor.dims().begin(), tensor.dims().end());
  std::unique_ptr<std::string> bytes(tensor.release_raw_data());
  pool.Add(name, tensor.data_type(), std::move(shape), std::move(*bytes));
  return true;
}

// Fill `tensor`'s raw_data by copying `name`'s bytes out of `pool` (this copy
// is unavoidable: onnx::TensorProto physically owns raw_data as a plain
// std::string) and clear any EXTERNAL / external_data state, so the result is
// an ordinary, self-contained in-memory tensor safe for every existing
// onnxsim pass. Returns false, leaving `tensor` untouched, if `name` isn't in
// `pool`.
inline bool HydrateTensorProto(const std::string& name,
                               onnx::TensorProto& tensor,
                               const TensorPool& pool) {
  const Entry* entry = pool.Find(name);
  if (entry == nullptr) return false;
  tensor.set_data_location(onnx::TensorProto::DEFAULT);
  tensor.clear_external_data();
  tensor.set_raw_data(entry->data.data(), entry->data.size());
  return true;
}

// Pool every classic-EXTERNAL tensor (onnx's own external-data convention --
// see detail::ExternalDataRef above) in `model` -- initializers and node-
// attribute tensors, recursing into subgraphs (detail::ForEachTensor) -- by
// memory-mapping the file(s) its external_data entries point to
// (TryMmapFile, see mmap_file.h) rather than reading each tensor's slice
// into a fresh heap buffer the way onnx's own external-data loader
// (loadExternalDataForModel) does. Distinct tensors that share one file (the
// common `all_tensors_to_one_file=True` case) share one mapping, so a model
// with thousands of initializers in one data file costs one mmap() call, not
// one per tensor. `base_dir` resolves a relative `location` exactly like
// onnx's own loader (join, don't reinterpret) -- pass the directory the
// model's own .onnx file lives in.
//
// hydrate_all (default true, matching plain onnx.load / onnx's own external-
// data loader) copies each pooled tensor's bytes back into its own raw_data
// before returning (via HydrateTensorProto) -- with this default, every
// pooled tensor in `model` ends up an ordinary, self-contained in-memory
// tensor, functionally identical to what onnx's own external-data loader
// produces, and `pool` is populated but not required by the caller
// afterwards. With hydrate_all=false, a pooled tensor is left EXTERNAL, its
// external_data untouched (so it still names the real on-disk
// file/offset/length onnx's own loader would use) -- `model` never carries
// that tensor's bytes at all (no large raw_data), but `pool` now holds a
// zero-copy Entry for it too, addressable by name, for a caller that knows
// how to consume one (see this header's top comment for which onnxsim call
// sites do not, yet).
//
// Hydration, when requested, is a distinct final pass over every tensor this
// call pooled -- run only after every one of them has been mapped/read into
// `pool` successfully, never interleaved into the mapping loop itself. That
// makes this call atomic with respect to `model`: if pooling fails partway
// (a later tensor's file is missing, say), no earlier tensor has been
// mutated either -- `model` is left exactly as it was passed in, not a mix
// of already-hydrated and still-EXTERNAL tensors depending on graph
// traversal order.
//
// A file that fails to memory-map (no mmap support on this platform, or the
// mmap()/MapViewOfFile() call itself fails -- see TryMmapFile) falls back to
// an ordinary seek+read for just the tensors that file backs, matching
// onnx's own external-data loader's behavior exactly, just without the mmap
// benefit for that one file.
//
// Returns the number of tensors pooled. Throws std::runtime_error if a
// referenced file cannot be opened at all (mapped or not), or if a tensor's
// offset/length range falls outside its file.
inline size_t PoolExternalData(onnx::ModelProto& model,
                               const std::string& base_dir, TensorPool& pool,
                               bool hydrate_all = true) {
  // Keyed by the resolved absolute path, so tensors sharing one data file
  // (the common case) share one mapping/read instead of one per tensor.
  std::map<std::string, std::pair<std::shared_ptr<const char[]>, uint64_t>>
      mapped_files;
  // Pooled this call, kept alongside each TensorProto* so the hydration pass
  // below need not re-walk (or re-match by name against) the graph -- same
  // bookkeeping ExportModelWithSafetensors and
  // AdoptAllWithPlaceholderOffsets use for their own two-phase sequences.
  std::vector<std::pair<std::string, onnx::TensorProto*>> pooled_tensors;

  detail::ForEachTensor(*model.mutable_graph(), [&](const std::string& name,
                                                    onnx::TensorProto& t) {
    if (t.data_location() != onnx::TensorProto::EXTERNAL) return;
    detail::ExternalDataRef ref(t);
    if (ref.location.empty()) return;  // nothing to pool

    const std::string resolved =
        (std::filesystem::path(base_dir) / ref.location).string();

    auto it = mapped_files.find(resolved);
    if (it == mapped_files.end()) {
      it = mapped_files.emplace(resolved, TryMmapFile(resolved)).first;
    }
    auto& [mapping, file_size] = it->second;

    std::vector<int64_t> shape(t.dims().begin(), t.dims().end());

    if (mapping) {
      const uint64_t begin =
          ref.offset >= 0 ? static_cast<uint64_t>(ref.offset) : 0;
      const uint64_t len = ref.length >= 0 ? static_cast<uint64_t>(ref.length)
                                           : file_size - begin;
      if (begin > file_size || len > file_size - begin) {
        throw std::runtime_error(
            "PoolExternalData: '" + resolved + "' tensor '" + name +
            "' external_data range exceeds the file's size");
      }
      std::shared_ptr<const char[]> owner(mapping, mapping.get() + begin);
      pool.Add(name, t.data_type(), shape, std::move(owner),
               std::string_view(mapping.get() + begin, len));
    } else {
      // No mmap support / mmap failed: fall back to an ordinary
      // seek+read, matching onnx's own external-data loader
      // (loadExternalDataForTensor) exactly for just this tensor.
      std::ifstream in(resolved, std::ios::binary);
      if (!in) {
        throw std::runtime_error("PoolExternalData: cannot open '" + resolved +
                                 "'");
      }
      in.seekg(0, std::ios::end);
      const int64_t full_size = static_cast<int64_t>(in.tellg());
      const int64_t begin = ref.offset >= 0 ? ref.offset : 0;
      const int64_t len = ref.length >= 0 ? ref.length : full_size - begin;
      in.seekg(begin, std::ios::beg);
      std::string bytes(static_cast<size_t>(len), '\0');
      in.read(bytes.data(), static_cast<std::streamsize>(len));
      if (!in) {
        throw std::runtime_error("PoolExternalData: failed to read '" +
                                 resolved + "' for tensor '" + name + "'");
      }
      pool.Add(name, t.data_type(), shape, std::move(bytes));
    }
    pooled_tensors.emplace_back(name, &t);
  });

  // Final phase: every tensor above pooled successfully (no throw reached
  // here otherwise), so it is now safe to hydrate them all -- or, with
  // hydrate_all=false, to leave every one of them exactly as pooling found
  // it (still EXTERNAL, no raw_data), never partially mutated.
  if (hydrate_all) {
    for (auto& [name, t] : pooled_tensors) {
      HydrateTensorProto(name, *t, pool);
    }
  }

  return pooled_tensors.size();
}

// Rewrite `model` so every eligible tensor (graph initializers and node-
// attribute tensors, recursing through subgraphs) becomes an EXTERNAL
// reference into a freshly-written `safetensors_path`, and write that file.
// Returns the number of tensors externalized. `pool` is an out-parameter: it
// ends up holding every externalized tensor, so a caller can inspect or
// reuse them (e.g. across several models sharing weights) without re-reading
// the file this just wrote.
inline size_t ExportModelWithSafetensors(onnx::ModelProto& model,
                                         const std::string& safetensors_path,
                                         TensorPool& pool) {
  // Keyed by the *pool key* ForEachTensor assigned (name, or a positional
  // fallback for an unnamed attribute tensor) -- NOT by `t->name()`, which
  // may be empty. Using t->name() to look the tensor back up after pooling
  // would silently drop every unnamed attribute tensor: it'd be adopted (its
  // bytes moved into the pool) but never re-marked EXTERNAL, leaving it
  // half-adopted (see AdoptFromTensorProto's caution note).
  std::vector<std::pair<std::string, onnx::TensorProto*>> exported;
  detail::ForEachTensor(*model.mutable_graph(),
                        [&](const std::string& name, onnx::TensorProto& t) {
                          if (AdoptFromTensorProto(name, t, pool)) {
                            exported.emplace_back(name, &t);
                          }
                        });

  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  pool.SaveSafetensors(safetensors_path, &offsets);
  const uint64_t prefix = HeaderPrefixSize(safetensors_path);

  for (auto& [pool_key, t] : exported) {
    const auto& [begin, end] = offsets.at(pool_key);
    t->set_data_location(onnx::TensorProto::EXTERNAL);
    t->clear_external_data();
    auto set_kv = [&](const std::string& k, const std::string& v) {
      auto* e = t->add_external_data();
      e->set_key(k);
      e->set_value(v);
    };
    set_kv("location", safetensors_path);
    set_kv("offset", std::to_string(prefix + begin));
    set_kv("length", std::to_string(end - begin));
  }
  return exported.size();
}

// Load `safetensors_path` into `pool` and, for every graph initializer whose
// name matches a pooled tensor, either hydrate it in place (hydrate_all, the
// default -- an ordinary in-memory tensor ready for any onnxsim pass) or
// leave it as a lazy EXTERNAL reference (hydrate_all=false) for a caller
// that will call HydrateTensorProto itself on just the tensors it ends up
// needing. See this header's top comment for why hydrate_all=false is not
// yet safe to feed straight into onnxsim's Simplify(). Returns the number of
// initializers matched.
inline size_t ImportModelWithSafetensors(onnx::ModelProto& model,
                                         const std::string& safetensors_path,
                                         TensorPool& pool,
                                         bool hydrate_all = true) {
  pool.LoadSafetensors(safetensors_path);
  size_t matched = 0;
  for (auto& init : *model.mutable_graph()->mutable_initializer()) {
    const Entry* entry = pool.Find(init.name());
    if (entry == nullptr) continue;
    ++matched;
    if (hydrate_all) {
      HydrateTensorProto(init.name(), init, pool);
    } else {
      init.set_data_location(onnx::TensorProto::EXTERNAL);
      init.clear_external_data();
      auto* e = init.add_external_data();
      e->set_key("location");
      e->set_value(safetensors_path);
    }
  }
  return matched;
}

// Export `pool`'s per-tensor ContentHash (see tensor_pool.h) into
// `model`'s graph-level metadata_props, one entry per pooled tensor keyed
// "<algorithm>:<tensor name>" (e.g. "blake3:conv1.weight"), plus one
// "<algorithm>:__all__" entry -- a digest, under the same HashAlgorithm, of
// every "<name>\0<hex digest>\n" line concatenated in name-sorted order --
// so a consumer can check the whole set with a single recomputation
// instead of re-deriving and comparing every metadata_props entry itself.
// A downstream verifier needs only tensor_pool_hash.h's ContentDigest (or
// an equivalent reimplementation of its framing) and the tensor's own
// dtype/shape/bytes -- not onnxsim or TensorPool -- to recompute and check
// either value. Call this AFTER `pool` holds its final tensor bytes (e.g.
// right after ExportModelWithSafetensors, which is also the point at which
// `model`'s initializers already carry the file references this hash lets
// a caller confirm weren't corrupted or substituted); calling it earlier
// hashes whatever `pool` happens to contain at the time.
inline void AttachContentHashMetadata(const TensorPool& pool,
                                      onnx::ModelProto& model) {
  const std::string algo_name(HashAlgorithmName(pool.GetHashAlgorithm()));
  auto* graph = model.mutable_graph();
  std::string concat;
  for (const auto& [name, hex] : pool.AllContentHashes()) {
    auto* kv = graph->add_metadata_props();
    kv->set_key(algo_name + ":" + name);
    kv->set_value(hex);
    concat += name;
    concat.push_back('\0');
    concat += hex;
    concat.push_back('\n');
  }
  auto* all_kv = graph->add_metadata_props();
  all_kv->set_key(algo_name + ":__all__");
  all_kv->set_value(ToHex(RawDigest(pool.GetHashAlgorithm(), concat)));
}

// --- self-describing single-file archives (see this header's top comment) -

// Pool key under which a self-describing archive's own graph is stored.
inline const std::string kEmbeddedModelKey = "model.onnx";  // NOLINT

// TensorPool::Add silently overwrites an existing entry of the same name,
// which would be actively dangerous here: a model with a real initializer
// named "model.onnx" (unlikely, but not forbidden by the ONNX spec) would
// have that weight silently clobbered by the embedded-model blob with no
// error. Call this right before adding kEmbeddedModelKey and it throws
// instead.
inline void CheckNoEmbeddedModelKeyCollision(const TensorPool& pool) {
  if (pool.Find(kEmbeddedModelKey) != nullptr) {
    throw std::runtime_error(
        "TensorPool: a tensor is already named '" + kEmbeddedModelKey +
        "', which collides with the reserved key this archive format "
        "embeds the graph itself under; rename that tensor before "
        "embedding");
  }
}

// Move every eligible tensor's bytes out of `model` into `pool` (as
// AdoptFromTensorProto does), mark each as a symbolic EXTERNAL reference
// into `archive_path`, then serialize the now-byte-free `model` and add
// those bytes to `pool` under kEmbeddedModelKey. After this, `pool` (once
// saved to `archive_path`) is a complete, self-describing archive; `model`
// itself is mutated into the same byte-free, symbolically-external state
// AdoptFromTensorProto leaves individual tensors in -- reload it via
// ExtractModel/LoadModelFromSafetensors/LoadModelFromGGUF rather than using
// `model` directly afterwards.
//
// The embedded copy's EXTERNAL references are symbolic (see this header's
// top comment) -- for real, byte-accurate offsets that make the extracted
// model.onnx blob standalone-loadable elsewhere, use
// SaveModelAsSafetensorsStandalone / SaveModelAsGGUFStandalone instead.
inline void EmbedModel(onnx::ModelProto& model, const std::string& archive_path,
                       TensorPool& pool) {
  detail::ForEachTensor(*model.mutable_graph(),
                        [&](const std::string& name, onnx::TensorProto& t) {
                          if (!AdoptFromTensorProto(name, t, pool)) return;
                          t.set_data_location(onnx::TensorProto::EXTERNAL);
                          t.clear_external_data();
                          auto* e = t.add_external_data();
                          e->set_key("location");
                          e->set_value(archive_path);
                        });
  CheckNoEmbeddedModelKeyCollision(pool);

  std::string bytes;
  if (!model.SerializeToString(&bytes)) {
    throw std::runtime_error(
        "TensorPool: EmbedModel: failed to serialize the model");
  }
  int64_t nbytes = static_cast<int64_t>(bytes.size());
  pool.Add(kEmbeddedModelKey, ONNX_INT8, {nbytes}, std::move(bytes));
}

// Reverse of EmbedModel: find kEmbeddedModelKey in `pool`, parse it into
// `*model`, and hydrate its initializers from `pool` by name (or, with
// hydrate_all=false, leave them as the symbolic EXTERNAL references
// EmbedModel wrote, for on-demand HydrateTensorProto -- same caveat as
// ImportModelWithSafetensors's hydrate_all=false). Returns false, leaving
// `*model` untouched, if `pool` has no embedded model. Throws
// std::runtime_error if the embedded bytes aren't a valid ModelProto.
inline bool ExtractModel(const TensorPool& pool, onnx::ModelProto* model,
                         bool hydrate_all = true) {
  const Entry* blob = pool.Find(kEmbeddedModelKey);
  if (blob == nullptr) return false;
  if (!model->ParseFromArray(blob->data.data(),
                             static_cast<int>(blob->data.size()))) {
    throw std::runtime_error("TensorPool: ExtractModel: embedded '" +
                             kEmbeddedModelKey +
                             "' is not a valid serialized ModelProto");
  }
  if (hydrate_all) {
    for (auto& init : *model->mutable_graph()->mutable_initializer()) {
      HydrateTensorProto(init.name(), init, pool);
    }
  }
  return true;
}

// Convenience wrapper: EmbedModel then pool.SaveSafetensors(path). `model`
// ends up mutated exactly as EmbedModel leaves it (see that function's
// note) -- reload with LoadModelFromSafetensors rather than reusing `model`.
inline void SaveModelAsSafetensors(onnx::ModelProto& model,
                                   const std::string& path, TensorPool& pool) {
  EmbedModel(model, path, pool);
  pool.SaveSafetensors(path);
}

// Convenience wrapper: pool.LoadSafetensors(path) then ExtractModel. Returns
// false if `path`'s pool has no embedded model (e.g. it's a plain weights-
// only safetensors file with no onnxsim-authored archive on top).
inline bool LoadModelFromSafetensors(const std::string& path,
                                     onnx::ModelProto* model, TensorPool& pool,
                                     bool hydrate_all = true) {
  pool.LoadSafetensors(path);
  return ExtractModel(pool, model, hydrate_all);
}

// --- standalone archives: real, byte-accurate self-referencing offsets ----
//
// EmbedModel's embedded copy is symbolic -- fine for onnxsim's own
// round-trip (ExtractModel resolves by name, never touches offset/length),
// but useless to a vanilla onnx/onnxruntime given just the extracted
// model.onnx blob. Making the embedded copy's offsets byte-accurate has a
// real chicken-and-egg problem: the model's serialized size depends on the
// digit-string offset/length values patched into it, which depend on where
// the model blob itself ends up in the archive, which depends on the
// model's serialized size.
//
// Broken with a fixed-width field: every offset/length value is written as
// exactly kOffsetFieldWidth decimal digits (zero-padded -- ordinary decimal
// parsers, including onnx's own external-data reader, skip leading zeros
// with no issue), so re-patching the *real* value in later never changes
// the model's serialized byte length versus an all-zero placeholder. That
// makes the whole thing a single clean pass: adopt tensors with placeholder
// offsets -> serialize -> pool it -> ask TensorPool for the real layout ->
// patch the *same-length* real values in -> re-serialize (guaranteed
// identical length) -> overwrite just the already-written model blob's
// bytes in place. Every weight tensor is still written to disk exactly
// once; only the small model blob is touched twice.

inline constexpr int kOffsetFieldWidth = 20;  // covers up to ~10**20 bytes

inline std::string FixedWidthDecimal(uint64_t v) {
  std::string s = std::to_string(v);
  if (s.size() > static_cast<size_t>(kOffsetFieldWidth)) {
    throw std::runtime_error("TensorPool: offset/length " + s +
                             " exceeds the reserved field width (" +
                             std::to_string(kOffsetFieldWidth) + " digits)");
  }
  return std::string(static_cast<size_t>(kOffsetFieldWidth) - s.size(), '0') +
         s;
}

// First half of the standalone-embed algorithm: adopt every eligible
// tensor's bytes into `pool` and mark each EXTERNAL with `archive_path` and
// fixed-width placeholder (all-zero) offset/length, ready for
// PatchStandaloneOffsets once the pool's real layout is known. Returns
// each adopted tensor's pool key alongside its TensorProto*, exactly like
// ExportModelWithSafetensors's internal bookkeeping, so the second half can
// find and re-patch them without a second graph traversal.
inline std::vector<std::pair<std::string, onnx::TensorProto*>>
AdoptAllWithPlaceholderOffsets(onnx::ModelProto& model,
                               const std::string& archive_path,
                               TensorPool& pool) {
  std::vector<std::pair<std::string, onnx::TensorProto*>> adopted;
  detail::ForEachTensor(*model.mutable_graph(), [&](const std::string& name,
                                                    onnx::TensorProto& t) {
    if (!AdoptFromTensorProto(name, t, pool)) return;
    t.set_data_location(onnx::TensorProto::EXTERNAL);
    t.clear_external_data();
    auto set_kv = [&](const std::string& k, const std::string& v) {
      auto* e = t.add_external_data();
      e->set_key(k);
      e->set_value(v);
    };
    set_kv("location", archive_path);
    set_kv("offset", FixedWidthDecimal(0));
    set_kv("length", FixedWidthDecimal(0));
    adopted.emplace_back(name, &t);
  });
  return adopted;
}

// Second half: overwrite each adopted tensor's placeholder offset/length
// (in place, same field width) with its real value from `offsets`
// (relative to the data section) and `data_section_start` (absolute) --
// both exactly what TensorPool::SaveSafetensors's/SaveGGUF's own
// data_offsets_out/data_section_start_out out-params provide.
inline void PatchStandaloneOffsets(
    std::vector<std::pair<std::string, onnx::TensorProto*>>& adopted,
    const std::map<std::string, std::pair<uint64_t, uint64_t>>& offsets,
    uint64_t data_section_start) {
  for (auto& [name, t] : adopted) {
    const auto& [begin, end] = offsets.at(name);
    for (auto& e : *t->mutable_external_data()) {
      if (e.key() == "offset") {
        e.set_value(FixedWidthDecimal(data_section_start + begin));
      } else if (e.key() == "length") {
        e.set_value(FixedWidthDecimal(end - begin));
      }
    }
  }
}

// Overwrite `bytes.size()` bytes of the already-written file at `path`,
// starting at absolute byte `offset`, without touching anything else in the
// file (no truncation, no resize) -- the final step of the standalone-embed
// algorithm, patching the model blob's placeholder bytes with the real,
// offset-patched serialization in place.
inline void PatchFileBytes(const std::string& path, uint64_t offset,
                           const std::string& bytes) {
  std::fstream out(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!out) {
    throw std::runtime_error("TensorPool: cannot reopen '" + path +
                             "' to patch the embedded model's bytes");
  }
  out.seekp(static_cast<std::streamoff>(offset));
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    throw std::runtime_error(
        "TensorPool: failed to patch the embedded model's bytes into '" + path +
        "'");
  }
}

// Standalone counterpart of SaveModelAsSafetensors: same one-file result,
// but the embedded copy's initializers carry real, byte-accurate
// external_data offset/length into `path` -- extracting the "model.onnx"
// tensor's bytes and saving them separately produces an ordinary, fully
// standalone .onnx file loadable by vanilla onnx/onnxruntime, no onnxsim
// involved. Every weight tensor is written to disk exactly once; the model
// blob itself is written, then patched in place (same algorithm as
// SaveModelAsGGUFStandalone) -- see this section's top comment.
inline void SaveModelAsSafetensorsStandalone(onnx::ModelProto& model,
                                             const std::string& path,
                                             TensorPool& pool) {
  auto adopted = AdoptAllWithPlaceholderOffsets(model, path, pool);
  CheckNoEmbeddedModelKeyCollision(pool);

  std::string placeholder_bytes;
  if (!model.SerializeToString(&placeholder_bytes)) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsSafetensorsStandalone: failed to serialize "
        "the model");
  }
  pool.Add(kEmbeddedModelKey, ONNX_INT8,
           {static_cast<int64_t>(placeholder_bytes.size())},
           std::string(placeholder_bytes));

  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  pool.SaveSafetensors(path, &offsets);
  uint64_t data_section_start = HeaderPrefixSize(path);

  PatchStandaloneOffsets(adopted, offsets, data_section_start);

  std::string final_bytes;
  if (!model.SerializeToString(&final_bytes)) {
    throw std::runtime_error(
        "TensorPool: SaveModelAsSafetensorsStandalone: failed to "
        "re-serialize the model");
  }
  if (final_bytes.size() != placeholder_bytes.size()) {
    // Should be unreachable given FixedWidthDecimal's fixed width; a hard
    // check rather than a silently-corrupt file if the invariant ever
    // breaks (e.g. a future edit to what EmbedModel-style code patches).
    throw std::runtime_error(
        "TensorPool: SaveModelAsSafetensorsStandalone: internal error -- "
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

#endif  // ONNXSIM_TENSOR_POOL_BRIDGE_H_
