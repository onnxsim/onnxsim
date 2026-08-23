// GGUF codec for TensorPool. See tensor_pool.h's declarations of
// SaveGGUF/LoadGGUF and gguf_dtype.h's file comment (dtype scope, notably
// that quantized types are never pooled) for the design rationale; this file
// is "just" the wire-format read/write.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <istream>
#include <optional>
#include <stdexcept>
#include <streambuf>

#include "ggml_kquant.h"
#include "gguf_dtype.h"
#include "mmap_file.h"
#include "tensor_pool.h"

namespace onnxsim {
namespace tensor_pool {

// This file is entirely GGUF-specific, so bring gguf::'s names (kMagic,
// GGUF_METADATA_VALUE_TYPE_*, ToOnnx/FromOnnx, ...) into scope for the whole
// translation unit -- both the helpers in the anonymous namespace below and
// the TensorPool::SaveGGUF/LoadGGUF definitions after it, which are siblings
// of that anonymous namespace, not nested inside it, so a `using` directive
// placed inside the anonymous namespace would not have reached them.
using namespace gguf;  // NOLINT

namespace {

uint64_t AlignUp(uint64_t v, uint64_t align) {
  if (align == 0) return v;
  uint64_t rem = v % align;
  return rem == 0 ? v : v + (align - rem);
}

// --- little-endian primitive readers/writers -------------------------------
// Duplicated locally rather than shared with tensor_pool.cpp's safetensors
// codec's WriteU64LE/ReadU64LE, mirroring gguf_dtype.h's choice to duplicate
// (not #include) tensor_pool_dtype.h's ONNX dtype numbers: keeps the two
// format codecs independently readable and buildable, at the cost of a few
// duplicated lines.

void WriteU32LE(std::ostream& out, uint32_t v) {
  char buf[4];
  for (int i = 0; i < 4; ++i) {
    buf[i] = static_cast<char>(v & 0xFF);
    v >>= 8;
  }
  out.write(buf, 4);
}

void WriteU64LE(std::ostream& out, uint64_t v) {
  char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<char>(v & 0xFF);
    v >>= 8;
  }
  out.write(buf, 8);
}

void WriteGGUFString(std::ostream& out, const std::string& s) {
  WriteU64LE(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

[[noreturn]] void FailRead(const std::string& path, const std::string& what) {
  throw std::runtime_error("TensorPool::LoadGGUF: '" + path + "': " + what);
}

uint32_t ReadU32LE(std::istream& in, const std::string& path) {
  unsigned char buf[4];
  in.read(reinterpret_cast<char*>(buf), 4);
  if (!in) FailRead(path, "unexpected end of file");
  uint32_t v = 0;
  for (int i = 3; i >= 0; --i) v = (v << 8) | buf[i];
  return v;
}

uint64_t ReadU64LE(std::istream& in, const std::string& path) {
  unsigned char buf[8];
  in.read(reinterpret_cast<char*>(buf), 8);
  if (!in) FailRead(path, "unexpected end of file");
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | buf[i];
  return v;
}

// A single GGUF string is length-prefixed with no cap of its own; bound it
// generously so a corrupt/malicious length (e.g. near UINT64_MAX) fails fast
// with a clear error instead of attempting a huge allocation.
constexpr uint64_t kMaxStringLen = 1ull << 30;  // 1 GiB

std::string ReadGGUFString(std::istream& in, const std::string& path) {
  uint64_t len = ReadU64LE(in, path);
  if (len > kMaxStringLen) {
    FailRead(path, "implausibly long string (len=" + std::to_string(len) +
                       "), file is likely corrupt");
  }
  std::string s(len, '\0');
  if (len > 0) {
    in.read(s.data(), static_cast<std::streamsize>(len));
    if (!in) FailRead(path, "unexpected end of file while reading a string");
  }
  return s;
}

size_t FixedMetadataScalarSize(uint32_t value_type) {
  switch (value_type) {
    case GGUF_METADATA_VALUE_TYPE_UINT8:
    case GGUF_METADATA_VALUE_TYPE_INT8:
    case GGUF_METADATA_VALUE_TYPE_BOOL:
      return 1;
    case GGUF_METADATA_VALUE_TYPE_UINT16:
    case GGUF_METADATA_VALUE_TYPE_INT16:
      return 2;
    case GGUF_METADATA_VALUE_TYPE_UINT32:
    case GGUF_METADATA_VALUE_TYPE_INT32:
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
      return 4;
    case GGUF_METADATA_VALUE_TYPE_UINT64:
    case GGUF_METADATA_VALUE_TYPE_INT64:
    case GGUF_METADATA_VALUE_TYPE_FLOAT64:
      return 8;
    default:
      return 0;  // STRING / ARRAY: no fixed size, handled by the caller
  }
}

// Skip one metadata value of `value_type` (the tag itself already consumed
// by the caller). TensorPool doesn't model GGUF's arbitrary metadata, so
// every value is discarded -- except an unsigned integer scalar's numeric
// value is decoded and returned, the one piece SaveGGUF... no, LoadGGUF's
// "general.alignment" probe below needs.
std::optional<uint64_t> SkipMetadataValue(std::istream& in,
                                          const std::string& path,
                                          uint32_t value_type) {
  if (value_type == GGUF_METADATA_VALUE_TYPE_STRING) {
    ReadGGUFString(in, path);
    return std::nullopt;
  }
  if (value_type == GGUF_METADATA_VALUE_TYPE_ARRAY) {
    uint32_t elem_type = ReadU32LE(in, path);
    uint64_t len = ReadU64LE(in, path);
    if (elem_type == GGUF_METADATA_VALUE_TYPE_ARRAY) {
      FailRead(path, "nested metadata arrays are not supported");
    }
    if (elem_type == GGUF_METADATA_VALUE_TYPE_STRING) {
      for (uint64_t i = 0; i < len; ++i) ReadGGUFString(in, path);
    } else {
      size_t elem_size = FixedMetadataScalarSize(elem_type);
      if (elem_size == 0) {
        FailRead(path, "unknown metadata array element type " +
                           std::to_string(elem_type));
      }
      in.seekg(static_cast<std::streamoff>(elem_size * len), std::ios::cur);
      if (!in) FailRead(path, "truncated metadata array");
    }
    return std::nullopt;
  }
  size_t size = FixedMetadataScalarSize(value_type);
  if (size == 0) {
    FailRead(path, "unknown metadata value type " + std::to_string(value_type));
  }
  unsigned char buf[8] = {};
  in.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(size));
  if (!in) FailRead(path, "truncated metadata value");
  switch (value_type) {
    case GGUF_METADATA_VALUE_TYPE_UINT8:
      return static_cast<uint64_t>(buf[0]);
    case GGUF_METADATA_VALUE_TYPE_UINT16:
      return static_cast<uint64_t>(buf[0]) |
             (static_cast<uint64_t>(buf[1]) << 8);
    case GGUF_METADATA_VALUE_TYPE_UINT32: {
      uint64_t v = 0;
      for (int i = 3; i >= 0; --i) v = (v << 8) | buf[i];
      return v;
    }
    case GGUF_METADATA_VALUE_TYPE_UINT64: {
      uint64_t v = 0;
      for (int i = 7; i >= 0; --i) v = (v << 8) | buf[i];
      return v;
    }
    default:
      return std::nullopt;  // signed ints / floats / bool: value not needed
  }
}

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> ne;  // ggml order: ne[0] is the innermost dimension
  uint32_t ggml_type = 0;
  uint64_t offset = 0;  // relative to the start of the tensor-data section
};

// Implausible-but-cheap-to-check bounds on counts read from the file,
// rejected before any loop/allocation sized by them -- a corrupt or
// adversarial file shouldn't be able to make this reader allocate or loop
// proportionally to an attacker-chosen 64-bit count.
constexpr uint64_t kMaxTensorCount = 10'000'000;
constexpr uint64_t kMaxMetadataCount = 10'000'000;
constexpr uint32_t kMaxDims =
    8;  // ggml's own GGML_MAX_DIMS is 4; generous slack

// A read-only std::streambuf backed directly by mapped file bytes, so
// LoadGGUFMmap can reuse every header/metadata/tensor-info parsing helper
// above (ReadU32LE, ReadGGUFString, SkipMetadataValue, ...) unchanged --
// they only need a std::istream, not specifically an ifstream. Unlike the
// default std::streambuf, this overrides seekoff/seekpos: LoadGGUF's parsing
// relies on in.tellg() (to find where the tensor-info section ends) working
// correctly, which the base class does not support on its own.
class MemStreamBuf : public std::streambuf {
 public:
  MemStreamBuf(const char* base, size_t size)
      : begin_(base), end_(base + size) {
    char* p = const_cast<char*>(begin_);
    setg(p, p, const_cast<char*>(end_));
  }

 protected:
  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if ((which & std::ios_base::in) == 0) return pos_type(off_type(-1));
    const char* base;
    switch (dir) {
      case std::ios_base::beg:
        base = begin_;
        break;
      case std::ios_base::cur:
        base = gptr();
        break;
      case std::ios_base::end:
        base = end_;
        break;
      default:
        return pos_type(off_type(-1));
    }
    const char* new_pos = base + off;
    if (new_pos < begin_ || new_pos > end_) return pos_type(off_type(-1));
    setg(const_cast<char*>(begin_), const_cast<char*>(new_pos),
         const_cast<char*>(end_));
    return pos_type(new_pos - begin_);
  }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(off_type(pos), std::ios_base::beg, which);
  }

 private:
  const char* begin_;
  const char* end_;
};

}  // namespace

void TensorPool::SaveGGUF(
    const std::string& path,
    const std::map<std::string, std::string>& string_metadata,
    std::map<std::string, std::pair<uint64_t, uint64_t>>* data_offsets_out,
    uint64_t* data_section_start_out) const {
  if (data_offsets_out != nullptr) data_offsets_out->clear();

  // Validate every entry's dtype up front so a partially-written file is
  // never left behind on a rejected tensor (mirrors SaveSafetensors).
  for (const auto& [name, entry] : entries_) {
    uint32_t ggml_type;
    if (!gguf::FromOnnx(entry.dtype, &ggml_type)) {
      throw std::runtime_error("TensorPool::SaveGGUF: tensor '" + name +
                               "' has an ONNX dtype (" +
                               std::to_string(entry.dtype) +
                               ") with no raw ggml_type representation");
    }
  }

  // Each tensor's [begin, end) relative to the data section's start. This
  // depends only on entries_ (name-sorted iteration order and each entry's
  // byte length), not on anything written yet, so it can be computed in one
  // pure pass before any I/O.
  std::map<std::string, std::pair<uint64_t, uint64_t>> offsets;
  {
    uint64_t running = 0;
    for (const auto& [name, entry] : entries_) {
      running = AlignUp(running, kDefaultAlignment);
      uint64_t begin = running;
      uint64_t end = begin + entry.nbytes();
      offsets[name] = {begin, end};
      running = end;
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("TensorPool::SaveGGUF: cannot open '" + path +
                             "' for writing");
  }

  WriteU32LE(out, kMagic);
  WriteU32LE(out, kSupportedVersion);
  WriteU64LE(out, entries_.size());
  WriteU64LE(out, string_metadata.size());
  for (const auto& [key, value] : string_metadata) {
    WriteGGUFString(out, key);
    WriteU32LE(out, GGUF_METADATA_VALUE_TYPE_STRING);
    WriteGGUFString(out, value);
  }

  for (const auto& [name, entry] : entries_) {
    uint32_t ggml_type = 0;
    gguf::FromOnnx(entry.dtype, &ggml_type);  // already validated above
    WriteGGUFString(out, name);
    WriteU32LE(out, static_cast<uint32_t>(entry.shape.size()));
    // GGML's ne[] wants the innermost dimension first -- the reverse of
    // onnx's row-major, outermost-first shape order (see LoadGGUF's mirror
    // comment on the way back).
    for (auto it = entry.shape.rbegin(); it != entry.shape.rend(); ++it) {
      WriteU64LE(out, static_cast<uint64_t>(*it));
    }
    WriteU32LE(out, ggml_type);
    WriteU64LE(out, offsets.at(name).first);
  }

  static const char kZeroBuf[64] = {};
  auto write_padding = [&](uint64_t n) {
    while (n > 0) {
      uint64_t chunk = std::min<uint64_t>(n, sizeof(kZeroBuf));
      out.write(kZeroBuf, static_cast<std::streamsize>(chunk));
      n -= chunk;
    }
  };

  uint64_t header_end = static_cast<uint64_t>(out.tellp());
  uint64_t data_section_start = AlignUp(header_end, kDefaultAlignment);
  write_padding(data_section_start - header_end);

  uint64_t write_pos = 0;  // relative to data_section_start
  for (const auto& [name, entry] : entries_) {
    const auto& [begin, end] = offsets.at(name);
    write_padding(begin - write_pos);
    if (!entry.data.empty()) {
      out.write(entry.data.data(),
                static_cast<std::streamsize>(entry.data.size()));
    }
    write_pos = end;
  }

  if (!out) {
    throw std::runtime_error("TensorPool::SaveGGUF: write failed for '" + path +
                             "'");
  }

  if (data_offsets_out != nullptr) *data_offsets_out = std::move(offsets);
  if (data_section_start_out != nullptr) {
    *data_section_start_out = data_section_start;
  }
}

std::vector<std::string> TensorPool::LoadGGUF(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("TensorPool::LoadGGUF: cannot open '" + path +
                             "'");
  }

  uint32_t magic = ReadU32LE(in, path);
  if (magic != kMagic) {
    FailRead(path, "not a GGUF file (bad magic)");
  }
  uint32_t version = ReadU32LE(in, path);
  if (version != kSupportedVersion) {
    FailRead(path, "GGUF version " + std::to_string(version) +
                       " is not supported (only version " +
                       std::to_string(kSupportedVersion) + " is)");
  }
  uint64_t tensor_count = ReadU64LE(in, path);
  uint64_t metadata_kv_count = ReadU64LE(in, path);
  if (tensor_count > kMaxTensorCount) {
    FailRead(path, "implausible tensor_count (" + std::to_string(tensor_count) +
                       "), likely corrupt");
  }
  if (metadata_kv_count > kMaxMetadataCount) {
    FailRead(path, "implausible metadata_kv_count (" +
                       std::to_string(metadata_kv_count) + "), likely corrupt");
  }

  uint64_t alignment = kDefaultAlignment;
  for (uint64_t i = 0; i < metadata_kv_count; ++i) {
    std::string key = ReadGGUFString(in, path);
    uint32_t value_type = ReadU32LE(in, path);
    std::optional<uint64_t> numeric = SkipMetadataValue(in, path, value_type);
    if (key == "general.alignment" && numeric.has_value() && *numeric != 0) {
      alignment = *numeric;
    }
  }

  std::vector<TensorInfo> infos;
  infos.reserve(tensor_count);
  for (uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo info;
    info.name = ReadGGUFString(in, path);
    uint32_t n_dims = ReadU32LE(in, path);
    if (n_dims > kMaxDims) {
      FailRead(path, "tensor '" + info.name +
                         "' has an implausible dimension count (" +
                         std::to_string(n_dims) + ")");
    }
    info.ne.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) info.ne[d] = ReadU64LE(in, path);
    info.ggml_type = ReadU32LE(in, path);
    info.offset = ReadU64LE(in, path);
    infos.push_back(std::move(info));
  }

  uint64_t header_end = static_cast<uint64_t>(in.tellg());
  uint64_t data_section_start = AlignUp(header_end, alignment);

  in.seekg(0, std::ios::end);
  uint64_t file_size = static_cast<uint64_t>(in.tellg());

  entries_.clear();
  std::vector<std::string> skipped;
  for (const auto& info : infos) {
    int32_t onnx_dtype;
    if (!gguf::ToOnnx(info.ggml_type, &onnx_dtype)) {
      skipped.push_back(info.name);
      continue;
    }
    uint64_t nelems = 1;
    for (uint64_t d : info.ne) nelems *= d;
    uint64_t nbytes;
    if (!gguf::TryTotalBytes(info.ggml_type, nelems, &nbytes)) {
      // A K-quant tensor whose element count isn't block-aligned -- a
      // malformed/truncated file, since GGML itself requires block
      // alignment for every real tensor of this type.
      FailRead(path, "tensor '" + info.name +
                         "' element count is not a multiple of its "
                         "quantization block size");
    }
    uint64_t abs_offset = data_section_start + info.offset;
    if (abs_offset > file_size || nbytes > file_size - abs_offset) {
      FailRead(path, "tensor '" + info.name +
                         "' data range extends past the end of the file");
    }

    std::string bytes(nbytes, '\0');
    if (nbytes > 0) {
      in.seekg(static_cast<std::streamoff>(abs_offset));
      in.read(bytes.data(), static_cast<std::streamsize>(nbytes));
      if (!in) {
        FailRead(path, "truncated while reading tensor '" + info.name + "'");
      }
    }

    // Reverse of SaveGGUF's reversal: ggml's ne[] is innermost-dimension-
    // first, onnx's shape is outermost-first, for the same contiguous bytes.
    std::vector<int64_t> shape(info.ne.rbegin(), info.ne.rend());
    Add(info.name, onnx_dtype, std::move(shape), std::move(bytes));
  }
  return skipped;
}

std::vector<std::string> TensorPool::LoadGGUFMmap(const std::string& path) {
  auto [mapped, file_size] = TryMmapFile(path);
  if (!mapped) {
    // No mapping support on this platform, or the mmap()/MapViewOfFile()
    // call itself failed -- fall back to the ordinary seek + read path
    // rather than surfacing a spurious failure for what may be a perfectly
    // valid file.
    return LoadGGUF(path);
  }

  MemStreamBuf buf(mapped.get(), static_cast<size_t>(file_size));
  std::istream in(&buf);

  uint32_t magic = ReadU32LE(in, path);
  if (magic != kMagic) {
    FailRead(path, "not a GGUF file (bad magic)");
  }
  uint32_t version = ReadU32LE(in, path);
  if (version != kSupportedVersion) {
    FailRead(path, "GGUF version " + std::to_string(version) +
                       " is not supported (only version " +
                       std::to_string(kSupportedVersion) + " is)");
  }
  uint64_t tensor_count = ReadU64LE(in, path);
  uint64_t metadata_kv_count = ReadU64LE(in, path);
  if (tensor_count > kMaxTensorCount) {
    FailRead(path, "implausible tensor_count (" + std::to_string(tensor_count) +
                       "), likely corrupt");
  }
  if (metadata_kv_count > kMaxMetadataCount) {
    FailRead(path, "implausible metadata_kv_count (" +
                       std::to_string(metadata_kv_count) + "), likely corrupt");
  }

  uint64_t alignment = kDefaultAlignment;
  for (uint64_t i = 0; i < metadata_kv_count; ++i) {
    std::string key = ReadGGUFString(in, path);
    uint32_t value_type = ReadU32LE(in, path);
    std::optional<uint64_t> numeric = SkipMetadataValue(in, path, value_type);
    if (key == "general.alignment" && numeric.has_value() && *numeric != 0) {
      alignment = *numeric;
    }
  }

  std::vector<TensorInfo> infos;
  infos.reserve(tensor_count);
  for (uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo info;
    info.name = ReadGGUFString(in, path);
    uint32_t n_dims = ReadU32LE(in, path);
    if (n_dims > kMaxDims) {
      FailRead(path, "tensor '" + info.name +
                         "' has an implausible dimension count (" +
                         std::to_string(n_dims) + ")");
    }
    info.ne.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) info.ne[d] = ReadU64LE(in, path);
    info.ggml_type = ReadU32LE(in, path);
    info.offset = ReadU64LE(in, path);
    infos.push_back(std::move(info));
  }

  uint64_t header_end = static_cast<uint64_t>(in.tellg());
  uint64_t data_section_start = AlignUp(header_end, alignment);

  entries_.clear();
  std::vector<std::string> skipped;
  for (const auto& info : infos) {
    int32_t onnx_dtype;
    if (!gguf::ToOnnx(info.ggml_type, &onnx_dtype)) {
      skipped.push_back(info.name);
      continue;
    }
    uint64_t nelems = 1;
    for (uint64_t d : info.ne) nelems *= d;
    uint64_t nbytes;
    if (!gguf::TryTotalBytes(info.ggml_type, nelems, &nbytes)) {
      FailRead(path, "tensor '" + info.name +
                         "' element count is not a multiple of its "
                         "quantization block size");
    }
    uint64_t abs_offset = data_section_start + info.offset;
    if (abs_offset > file_size || nbytes > file_size - abs_offset) {
      FailRead(path, "tensor '" + info.name +
                         "' data range extends past the end of the file");
    }

    const char* base = mapped.get() + abs_offset;
    std::shared_ptr<const char[]> owner(mapped, base);
    std::string_view data(base, nbytes);

    // Reverse of SaveGGUF's reversal: ggml's ne[] is innermost-dimension-
    // first, onnx's shape is outermost-first, for the same contiguous bytes.
    std::vector<int64_t> shape(info.ne.rbegin(), info.ne.rend());
    Add(info.name, onnx_dtype, std::move(shape), std::move(owner), data);
  }
  return skipped;
}

bool TensorPool::DequantizeToFloat(const std::string& name,
                                   std::vector<float>* out) const {
  const Entry* entry = Find(name);
  if (entry == nullptr) {
    return false;
  }
  uint32_t ggml_type;
  if (!FromOnnx(entry->dtype, &ggml_type) || !IsKQuant(ggml_type)) {
    return false;
  }
  int64_t numel = 1;
  for (int64_t d : entry->shape) numel *= d;
  return DequantizeGgmlKQuant(
      reinterpret_cast<const uint8_t*>(entry->data.data()), entry->data.size(),
      ggml_type, numel, out);
}

}  // namespace tensor_pool
}  // namespace onnxsim
