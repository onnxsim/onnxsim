/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * TensorPool: a ref-counted, name-keyed store of tensor byte buffers, with a
 * HuggingFace safetensors (https://github.com/huggingface/safetensors) file
 * as its serialization format.
 *
 * Motivation (reducing onnx::TensorProto data copies): onnx::TensorProto
 * physically owns its `raw_data` bytes as a plain std::string, so ordinary
 * protobuf copies (ModelProto assignment, onnx-optimizer's Import/Export
 * round trip, a Python trampoline hop) each deep-copy every initializer's
 * payload -- see onnxsim.cpp's OptimizeFixed move-through-the-round-trip fix
 * for issue #633, which works around exactly this for one specific call
 * site. TensorPool generalizes the same idea for the two places a real copy
 * is otherwise unavoidable:
 *
 *   1. Loading weights from disk: onnx's own external-data loader (and a
 *      naive safetensors reader) memcpy's each tensor's slice out of the
 *      file into a fresh TensorProto.raw_data. LoadSafetensors instead reads
 *      the file into ONE owned buffer and hands out std::shared_ptr-aliased
 *      views into it -- one copy total (the disk read), not one per tensor,
 *      and every Entry a caller Finds() shares that same buffer for free.
 *   2. Saving weights back out: SaveSafetensors writes each entry straight
 *      from its own buffer -- it never concatenates the pool into one
 *      in-memory blob first.
 *
 * TensorPool itself is dependency-free of onnx/protobuf (only the *integer*
 * ONNX dtype codes from tensor_pool_dtype.h, plus the vendored BLAKE3 and
 * self-contained SHA-256 backing ContentHash/SetHashAlgorithm below -- see
 * tensor_pool_hash.h), so it builds and unit-tests standalone -- mirrors
 * the dlpack_dtype.h / dlpack_bridge.h split. The
 * onnx::TensorProto <-> TensorPool glue lives in tensor_pool_bridge.h; see
 * that header's comment for how (and where) it's safe to plug a pool into
 * onnxsim's existing passes -- notably, several of them (onnxsim.cpp's
 * IsAllZeroTensor and integer-tensor extraction, dlpack_bridge.h's
 * FromTensorProtoBorrowing) already special-case and skip
 * onnx::TensorProto::EXTERNAL tensors, so a pool reference is only safe to
 * hand them once it has been hydrated back to an ordinary in-memory tensor.
 *
 * File format (https://github.com/huggingface/safetensors#format):
 *   [8 bytes] header length N, u64 little-endian
 *   [N bytes] UTF-8 JSON header: a flat object of
 *     {"<name>": {"dtype": "<code>", "shape": [...],
 *                 "data_offsets": [begin, end]}, ...},
 *     plus an optional "__metadata__" entry (arbitrary string->string map,
 *     skipped by this reader)
 *   [rest of file] raw tensor bytes, concatenated at the offsets the header
 *     describes, relative to the end of the header
 * This is onnx's own "raw external data in a file" layout with a JSON
 * manifest bolted on: a tensor is addressed the same way (an offset + length
 * range in a file), just keyed by name via the header instead of by
 * per-tensor location/offset/length entries living in the TensorProto
 * itself. That means a TensorPool's file is BOTH a standard safetensors file
 * (openable by the `safetensors` Python package, HF `transformers`/
 * `diffusers`, etc., with no onnxsim involved) AND usable, unmodified, as the
 * backing file for onnx's own classic external-data mechanism, by pointing
 * each TensorProto's external_data offset past the header (see
 * tensor_pool_bridge.h's ExportModelWithSafetensors).
 *
 * Byte order: like onnx::TensorProto::raw_data (which ONNX's spec fixes to
 * little-endian on every host) and every safetensors file actually produced
 * in the wild, an Entry's `data` is *always* little-endian bytes, regardless
 * of host byte order. TensorPool never interprets or swaps them -- only the
 * file's own 8-byte header-length prefix, which this pool reads/writes byte
 * by byte, needs (and gets) explicit little-endian handling for correctness
 * on a big-endian host (this repo tests on s390x; see dlpack_dtype.h's
 * kRawDataIsHostOrder for the *other* boundary, where such bytes are
 * eventually swapped into host order for arithmetic).
 */
#ifndef ONNXSIM_TENSOR_POOL_H_
#define ONNXSIM_TENSOR_POOL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensor_pool_hash.h"

namespace onnxsim {
namespace tensor_pool {

// A named tensor's data as a zero-copy view into whatever storage TensorPool
// keeps alive on its behalf. `owner` keeps `data` alive (it may be an
// aliasing shared_ptr into a much larger buffer, e.g. one Entry per tensor in
// a loaded safetensors file all alias the same underlying read); copying an
// Entry is a shared_ptr refcount bump, never a byte copy.
struct Entry {
  // An OnnxDtype from tensor_pool_dtype.h, OR -- for a tensor LoadGGUF/
  // LoadGGUFMmap pooled from a GGML K-quant source (Q4_K/Q5_K/Q6_K/Q8_0) --
  // one of gguf_dtype.h's private ONNXSIM_GGML_* codes, in which case `data`
  // holds the tensor's native, still-packed GGML block bytes rather than a
  // raw per-element layout. See gguf_dtype.h's file comment and
  // TensorPool::DequantizeToFloat below.
  int32_t dtype = 0;
  std::vector<int64_t> shape;
  std::shared_ptr<const char[]> owner;
  std::string_view data;  // dtype's raw little-endian bytes; aliases *owner

  size_t nbytes() const { return data.size(); }
};

class TensorPool {
 public:
  // Store `bytes` under `name`, taking ownership without copying (the string
  // is moved in and becomes the pool's sole owner of that allocation).
  // Overwrites any existing entry of the same name.
  void Add(const std::string& name, int32_t dtype, std::vector<int64_t> shape,
           std::string&& bytes);

  // Store bytes already owned elsewhere via a shared_ptr -- e.g. a view
  // produced by another pool's LoadSafetensors, or a sub-range of an entry
  // this pool already holds. No copy either way.
  void Add(const std::string& name, int32_t dtype, std::vector<int64_t> shape,
           std::shared_ptr<const char[]> owner, std::string_view data);

  const Entry* Find(const std::string& name) const;
  bool Erase(const std::string& name);
  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }

  // Content-hash algorithm used by ContentHash/AllContentHashes below.
  // Defaults to BLAKE3 -- cryptographic and, per its own published
  // benchmarks, faster than SHA-256, so it's a safe default for both an
  // internal dedup/lookup key and (see tensor_pool_bridge.h's
  // AttachContentHashMetadata) an externally-verifiable integrity value.
  // Override to kSha256 to interoperate with a downstream verifier that
  // only knows SHA-256; see tensor_pool_hash.h's file comment for the
  // tradeoff. Affects only hashes computed after the call.
  void SetHashAlgorithm(HashAlgorithm algo) { hash_algorithm_ = algo; }
  HashAlgorithm GetHashAlgorithm() const { return hash_algorithm_; }

  // Hex-encoded digest of `name`'s dtype + shape + bytes (see
  // tensor_pool_hash.h's ContentDigest) under the pool's current
  // HashAlgorithm. Two entries -- in this pool or, run through the same
  // HashAlgorithm, anywhere else -- with equal ContentHash have identical
  // dtype, shape, and raw bytes (modulo the underlying hash's own collision
  // resistance). Throws std::out_of_range if `name` isn't in the pool.
  std::string ContentHash(const std::string& name) const;

  // ContentHash for every entry, in the pool's iteration (name-sorted)
  // order -- e.g. for AttachContentHashMetadata to export deterministically.
  std::map<std::string, std::string> AllContentHashes() const;

  // Write every entry to a .safetensors file at `path`, in the pool's
  // iteration (name-sorted) order. Each tensor is written straight from its
  // own Entry::data -- never concatenated into one in-memory blob first --
  // so saving performs no tensor-data copies of its own beyond the OS's
  // ordinary write() buffering. Throws std::runtime_error if any pooled
  // dtype has no safetensors representation (see tensor_pool_dtype.h), or on
  // I/O failure.
  //
  // When `data_offsets_out` is non-null, it is filled with each written
  // tensor's [begin, end) byte range *relative to the end of the header*
  // (exactly the file's own "data_offsets" header field) -- callers that
  // need the tensor's *absolute* file offset (e.g. to fill in a
  // TensorProto's external_data, as tensor_pool_bridge.h's
  // ExportModelWithSafetensors does) add HeaderPrefixSize(path)'s result.
  void SaveSafetensors(const std::string& path,
                       std::map<std::string, std::pair<uint64_t, uint64_t>>*
                           data_offsets_out = nullptr) const;

  // Replace this pool's contents with every tensor described by the
  // .safetensors file at `path`. Reads the whole file into one owned buffer
  // and gives every Entry a std::shared_ptr that aliases *that one buffer*
  // (via shared_ptr's aliasing constructor) -- one copy total (the disk
  // read), not one per tensor. Throws std::runtime_error on I/O failure, a
  // malformed header, or a tensor whose data_offsets fall outside the file.
  void LoadSafetensors(const std::string& path);

  // Like LoadSafetensors, but memory-maps `path` (mmap() on POSIX,
  // MapViewOfFile() on Windows) instead of reading it into a heap buffer:
  // every Entry's data is a direct view into the OS page cache backing the
  // file, so this makes *zero* tensor-data copies -- not even
  // LoadSafetensors's one whole-file read -- and pages are faulted in
  // lazily, on first touch, rather than the whole file being made resident
  // up front. The mapping itself is kept alive by the same shared_ptr-
  // aliasing scheme as LoadSafetensors (see Entry::owner): it lives for as
  // long as any Entry (or view derived from one) this call produced is
  // reachable, and is unmapped automatically once the last one is dropped.
  //
  // Falls back to an ordinary LoadSafetensors call -- silently, with no
  // memory-mapping benefit but identical results -- when this platform has
  // no memory-mapping support (e.g. Emscripten/wasm) or the underlying
  // mmap()/MapViewOfFile() call fails (e.g. `path` is empty, or isn't a
  // regular seekable file). Throws std::runtime_error on I/O failure, a
  // malformed header, or a tensor whose data_offsets fall outside the file
  // -- same as LoadSafetensors.
  //
  // Caveat inherent to memory-mapping, unlike LoadSafetensors's own private
  // copy: the mapped bytes alias `path` on disk for as long as the mapping
  // lives. Modifying or truncating the file out from under a live mapping is
  // undefined behavior (a SIGBUS on POSIX if a mapped page is truncated
  // away; stale or torn bytes if its content is rewritten in place) -- only
  // use this for a file this process (or a well-behaved peer) won't mutate
  // while the pool is alive.
  void LoadSafetensorsMmap(const std::string& path);

  // --- GGUF (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
  // format, version 3 -- implemented in tensor_pool_gguf.cpp, a separate
  // translation unit from the safetensors codec above, since the two file
  // formats share nothing but the TensorPool storage they read into/from.
  // See gguf_dtype.h's file comment for an important scope note: most of
  // GGML's block-quantized types (Q4_0, every IQ*_ variant, ...) have no
  // ONNX raw-data equivalent and are never pooled -- LoadGGUF skips and
  // reports them rather than writing garbage. The K-quant family
  // (Q4_K/Q5_K/Q6_K/Q8_0) -- what a real quantized checkpoint (e.g.
  // Unsloth's GGUF exports) actually uses for the bulk of its weights -- IS
  // pooled, holding its native, still-packed block bytes (see
  // gguf_dtype.h's IsKQuant); DequantizeToFloat below decodes an entry like
  // that to plain float32 values.

  // Write every entry to a GGUF file at `path`. Every dtype TensorPool can
  // hold has a raw ggml_type counterpart (see gguf_dtype.h), so -- unlike
  // LoadGGUF -- this never has to skip an entry; it throws instead if some
  // future dtype addition broke that invariant. `string_metadata` is
  // written verbatim as top-level GGUF string metadata entries (e.g.
  // {"general.architecture", "onnxsim"}); pass {} for none.
  //
  // When `data_offsets_out` is non-null, it is filled with each written
  // tensor's [begin, end) byte range *relative to the start of the tensor-
  // data section* (GGUF's own per-tensor "offset" field, plus nbytes).
  // When `data_section_start_out` is non-null, it receives that section's
  // absolute byte offset from the start of the file. Together they're
  // enough for a caller (see tensor_pool_gguf_bridge.h) to point a
  // TensorProto's external_data straight at the file, the same way
  // SaveSafetensors's data_offsets_out plus HeaderPrefixSize does for
  // safetensors. Throws std::runtime_error on I/O failure.
  void SaveGGUF(const std::string& path,
                const std::map<std::string, std::string>& string_metadata = {},
                std::map<std::string, std::pair<uint64_t, uint64_t>>*
                    data_offsets_out = nullptr,
                uint64_t* data_section_start_out = nullptr) const;

  // Replace this pool's contents with every tensor in the GGUF file at
  // `path` whose ggml_type this pool can represent -- either a *raw*,
  // unquantized type (stored as-is) or one of the four K-quant types
  // gguf_dtype.h's IsKQuant covers (stored as its native, still-packed
  // block bytes -- see DequantizeToFloat to decode one). Every other
  // quantized type (Q4_0, every IQ*_ variant, ...) has no representation
  // this pool can hold at all. Unlike LoadSafetensors, this does NOT read
  // the whole file into memory: only the (small) header/metadata/tensor-
  // info section is read up front, and each *included* tensor's bytes are
  // read with their own targeted seek + read -- loading a large quantized
  // checkpoint this way costs only the bytes of the tensors this pool can
  // actually use, not the whole file. Returns the names of tensors that
  // were present in the file but skipped because their ggml_type has no
  // representation this pool can hold (empty if every tensor was loaded).
  // Throws std::runtime_error on I/O failure, an unrecognized magic/
  // version, a malformed header, or a K-quant tensor whose element count
  // is not a multiple of its quantization block size.
  std::vector<std::string> LoadGGUF(const std::string& path);

  // Like LoadGGUF, but memory-maps `path` (mmap() on POSIX, MapViewOfFile()
  // on Windows -- the same TryMmapFile helper LoadSafetensorsMmap uses, see
  // mmap_file.h) instead of doing its own seek + read per included tensor.
  // The header/metadata/tensor-info section is still parsed exactly as
  // LoadGGUF parses it (just from the mapped bytes instead of an ifstream);
  // only what happens per tensor differs: an *included* tensor's Entry
  // becomes a direct view into the mapped file via the usual shared_ptr-
  // aliasing scheme (see Entry::owner) rather than a copied std::string, and
  // a *skipped* tensor's bytes are never even faulted into memory, since
  // mapping the file only reserves address space -- pages are faulted in
  // lazily, on first touch, so an untouched skipped tensor's pages are
  // simply never read at all (better than LoadGGUF's own already-selective
  // seek + read, which at least issues no I/O for skipped tensors but still
  // pays a syscall per included one).
  //
  // Falls back to an ordinary LoadGGUF call -- silently, with no memory-
  // mapping benefit but identical results -- when this platform has no
  // memory-mapping support (e.g. Emscripten/wasm) or the underlying mmap()/
  // MapViewOfFile() call fails. Returns the same skipped-tensor-names list,
  // and throws under the same conditions, as LoadGGUF.
  //
  // Same caveat as LoadSafetensorsMmap: the mapped bytes alias `path` on
  // disk for as long as the mapping lives, so modifying or truncating the
  // file out from under a live mapping is undefined behavior.
  std::vector<std::string> LoadGGUFMmap(const std::string& path);

  // Decodes `name`'s entry to plain float32 values, appended to `out` (not
  // cleared first) -- the only way to get usable numeric values out of an
  // entry LoadGGUF/LoadGGUFMmap pooled from a K-quant (Q4_K/Q5_K/Q6_K/Q8_0)
  // source, since its `data` otherwise holds native, still-packed GGML
  // block bytes, not per-element values (see gguf_dtype.h's IsKQuant).
  // Returns false, leaving `out` untouched, if `name` isn't in the pool or
  // its dtype is not one of those four K-quant codes (including an
  // ordinary raw dtype, e.g. FLOAT/FLOAT16 -- those need no decoding at
  // all; read Entry::data directly, the same way every other TensorPool
  // consumer already does).
  bool DequantizeToFloat(const std::string& name,
                         std::vector<float>* out) const;

 private:
  std::map<std::string, Entry> entries_;
  HashAlgorithm hash_algorithm_ = HashAlgorithm::kBlake3;
};

// Bytes of `path`'s safetensors preamble (the 8-byte length prefix plus the
// JSON header itself) -- i.e. where the raw tensor data segment begins, and
// so what SaveSafetensors's per-tensor data_offsets are relative to. Reads
// only the 8-byte length prefix, not the whole file. Throws
// std::runtime_error on I/O failure.
uint64_t HeaderPrefixSize(const std::string& path);

}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_TENSOR_POOL_H_
