/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure, dependency-free mapping between ONNX tensor element types and GGML
 * tensor types (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
 * plus the small set of GGUF wire-format constants (magic, version,
 * alignment, metadata value-type codes) needed to read/write the container
 * itself. Operates on the *integer* ONNX dtype codes and the *integer* GGML
 * type codes, not the onnx headers, so tensor_pool_gguf.cpp builds and unit-
 * tests on its own -- mirrors tensor_pool_dtype.h's split for safetensors,
 * for the same reason.
 *
 * IMPORTANT SCOPE NOTE: only GGML's plain, unquantized element types map to
 * an ONNX dtype (F32, F16, BF16, F64, I8, I16, I32, I64 -- storage where one
 * element is one fixed-width value, exactly what onnx::TensorProto::raw_data
 * and safetensors both assume). GGML's block-quantized types (Q4_0, every
 * IQ*_ variant, ...) -- which is what most real-world .gguf files (quantized
 * LLM checkpoints) actually store most of their tensors as -- have NO onnx
 * raw-data equivalent: each "element" is really part of a shared block of
 * packed sub-byte codes plus one or more scale/zero-point values, decoded by
 * dequantization math this mapping does not implement. ToOnnx() rejects them
 * (returns false) rather than guess; TensorPool's GGUF loader
 * (tensor_pool_gguf.cpp) skips such tensors and reports them, rather than
 * materializing garbage bytes as if they were raw floats.
 *
 * EXCEPTION -- the "K-quant" family this mapping DOES cover (Q4_K, Q5_K,
 * Q6_K, plus the simpler per-32-element Q8_0): these are the block formats
 * real-world quantized checkpoints (e.g. Unsloth's GGUF exports) actually
 * use for the bulk of their weights, unlike the rarer legacy Q4_0-family and
 * IQ*-family types this mapping still does not cover. Each maps to one of
 * the ONNXSIM_GGML_* codes below -- private, onnxsim-fork-only integer
 * values, NOT part of the ONNX spec (TensorProto's `data_type` field is
 * plain `int32`, not the enum type, precisely so a private code like this
 * needs no protobuf schema change to round-trip through it -- see
 * onnx.in.proto). A tensor tagged with one of these codes carries its
 * *native, still-packed* GGML block bytes verbatim (see KQuantBlockBytes) --
 * genuinely quantized, not a raw per-element layout ToOnnx()'s ordinary
 * contract promises. Nothing outside onnxsim's own GGUF loading path should
 * ever see one of these codes on a live onnx::TensorProto: ggml_kquant.h's
 * DequantizeGgmlKQuant, wired into tensor_pool_bridge.h's
 * HydrateTensorProto, always decodes to ONNX_FLOAT before a pooled K-quant
 * entry reaches a TensorProto a caller can observe.
 */
#ifndef ONNXSIM_GGUF_DTYPE_H_
#define ONNXSIM_GGUF_DTYPE_H_

#include <cstddef>
#include <cstdint>

namespace onnxsim {
namespace tensor_pool {
namespace gguf {

// --- container format constants -------------------------------------------

inline constexpr uint32_t kMagic =
    0x46554747;  // "GGUF" read as a little-endian u32
// Only version 3 (the current format, and what every actively-maintained
// GGUF writer -- llama.cpp included -- has produced since 2023) is
// supported. Version 1 used 32-bit tensor/metadata counts and a different
// string encoding; version 2 is functionally version 3 minus a couple of
// later-added metadata conventions this reader doesn't depend on anyway, but
// is rare enough in the wild that it isn't worth the extra branch -- treated
// as unsupported alongside version 1.
inline constexpr uint32_t kSupportedVersion = 3;
inline constexpr uint64_t kDefaultAlignment = 32;

// gguf_metadata_value_type.
enum MetadataValueType : uint32_t {
  GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  GGUF_METADATA_VALUE_TYPE_STRING = 8,
  GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

// ggml_type -- only the stable low IDs this mapping cares about are named;
// every other value (the quantized/k-quant/IQ* family, and any type newer
// than this list) is handled generically as "unsupported" by ToOnnx/IsRaw
// below without needing its own enumerator. GGML never reassigns an existing
// ID (only appends and occasionally retires one), so these are stable across
// ggml/llama.cpp versions.
enum GgmlType : uint32_t {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_BF16 = 30,
};

// --- ONNX dtype <-> ggml_type (raw types only; see the file comment) ------

// Stable ONNX TensorProto::DataType wire numbers, duplicated (not shared via
// an #include) so this header stays independent of tensor_pool_dtype.h and
// buildable on its own, exactly like that header is independent of
// dlpack_dtype.h's copy of the same numbers.
enum OnnxDtype : int32_t {
  ONNX_UNDEFINED = 0,
  ONNX_FLOAT = 1,
  ONNX_INT8 = 3,
  ONNX_INT16 = 5,
  ONNX_INT32 = 6,
  ONNX_INT64 = 7,
  ONNX_FLOAT16 = 10,
  ONNX_DOUBLE = 11,
  ONNX_BFLOAT16 = 16,
};

// Private, onnxsim-fork-only dtype codes an Entry::dtype (or a TensorProto
// this loader briefly produces before HydrateTensorProto decodes it away --
// see the file comment) may hold for a still-packed GGML K-quant tensor.
// Chosen from a range (10000+) with no plausible collision against ONNX's
// own DataType enum (defined up to 26 as of this writing -- see
// onnx.in.proto); never reassign or reuse one of these four values.
enum OnnxsimPrivateDtype : int32_t {
  ONNXSIM_GGML_Q4_K = 10001,
  ONNXSIM_GGML_Q5_K = 10002,
  ONNXSIM_GGML_Q6_K = 10003,
  ONNXSIM_GGML_Q8_0 = 10004,
};

// Bytes of one element. 0 for a ggml_type this mapping doesn't cover
// (quantized types report 0 here too -- computing their true per-block size
// isn't needed since they're never pooled, only skipped).
inline size_t ElementSize(uint32_t ggml_type) {
  switch (ggml_type) {
    case GGML_TYPE_I8:
      return 1;
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I16:
      return 2;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
      return 4;
    case GGML_TYPE_F64:
    case GGML_TYPE_I64:
      return 8;
    default:
      return 0;
  }
}

// True for a ggml_type with a fixed-width, unquantized, one-value-per-
// element layout -- the only kind ElementSize/nelems*ElementSize sizing
// applies to. False for every quantized/K-quant/IQ* type (including the
// four IsKQuant covers below, which need TryTotalBytes instead -- see that
// function) and anything this mapping doesn't recognize.
inline bool IsRaw(uint32_t ggml_type) { return ElementSize(ggml_type) != 0; }

// True for one of the four block-quantized types this mapping covers (see
// the file comment's EXCEPTION paragraph) -- Q4_K/Q5_K/Q6_K (super-blocks of
// 256 elements) or Q8_0 (blocks of 32). False for every other ggml_type,
// including every quantized family this mapping does NOT decode (Q4_0,
// every IQ*_ variant, ...).
inline bool IsKQuant(uint32_t ggml_type) {
  switch (ggml_type) {
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_0:
      return true;
    default:
      return false;
  }
}

// Elements per block for an IsKQuant ggml_type. 0 for anything else.
inline size_t KQuantBlockElements(uint32_t ggml_type) {
  switch (ggml_type) {
    case GGML_TYPE_Q8_0:
      return 32;
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
      return 256;  // GGML's QK_K super-block size.
    default:
      return 0;
  }
}

// Bytes per block for an IsKQuant ggml_type -- the packed on-disk size of
// one KQuantBlockElements(ggml_type)-element block (a 2-byte fp16 scale plus
// the packed sub-byte quant codes for Q8_0; two 2-byte fp16 super-block
// scales, a 12-byte packed 6-bit scale/min table, and the packed quant codes
// for Q4_K/Q5_K/Q6_K -- see block_q4_K/block_q5_K/block_q6_K/block_q8_0 in
// ggml's ggml-common.h, and ggml_kquant.h's DequantizeGgmlKQuant for the
// exact byte layout each decodes). 0 for anything else.
inline size_t KQuantBlockBytes(uint32_t ggml_type) {
  switch (ggml_type) {
    case GGML_TYPE_Q8_0:
      return 34;  // 2 (d) + 32 (qs)
    case GGML_TYPE_Q4_K:
      return 144;  // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
    case GGML_TYPE_Q5_K:
      return 176;  // 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs)
    case GGML_TYPE_Q6_K:
      return 210;  // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
    default:
      return 0;
  }
}

// Map a ggml_type to its ONNX dtype code -- ONNXSIM_GGML_* (see that enum's
// doc comment) for one of the four IsKQuant types, an ordinary ONNX_* code
// for a raw type. Returns false (leaving `*out` untouched) for a quantized
// type this mapping doesn't cover (Q4_0, IQ*, ...) or one it doesn't
// recognize at all.
inline bool ToOnnx(uint32_t ggml_type, int32_t* out) {
  switch (ggml_type) {
    case GGML_TYPE_F32:
      *out = ONNX_FLOAT;
      return true;
    case GGML_TYPE_F16:
      *out = ONNX_FLOAT16;
      return true;
    case GGML_TYPE_BF16:
      *out = ONNX_BFLOAT16;
      return true;
    case GGML_TYPE_F64:
      *out = ONNX_DOUBLE;
      return true;
    case GGML_TYPE_I8:
      *out = ONNX_INT8;
      return true;
    case GGML_TYPE_I16:
      *out = ONNX_INT16;
      return true;
    case GGML_TYPE_I32:
      *out = ONNX_INT32;
      return true;
    case GGML_TYPE_I64:
      *out = ONNX_INT64;
      return true;
    case GGML_TYPE_Q4_K:
      *out = ONNXSIM_GGML_Q4_K;
      return true;
    case GGML_TYPE_Q5_K:
      *out = ONNXSIM_GGML_Q5_K;
      return true;
    case GGML_TYPE_Q6_K:
      *out = ONNXSIM_GGML_Q6_K;
      return true;
    case GGML_TYPE_Q8_0:
      *out = ONNXSIM_GGML_Q8_0;
      return true;
    default:
      return false;
  }
}

// The exact byte length of an IsRaw or IsKQuant ggml_type tensor holding
// `nelems` total elements -- generalizes the simple `nelems *
// ElementSize(ggml_type)` a raw type alone would need, to also cover a
// K-quant type's block-based sizing (`nelems / KQuantBlockElements *
// KQuantBlockBytes`). Returns false (leaving `*out_nbytes` untouched) for a
// ggml_type this mapping doesn't cover at all, or a K-quant type whose
// `nelems` is not an exact multiple of its block size (a malformed or
// truncated tensor -- every real K-quant tensor's element count is block-
// aligned, since GGML itself requires each row to be).
inline bool TryTotalBytes(uint32_t ggml_type, uint64_t nelems,
                          uint64_t* out_nbytes) {
  if (IsRaw(ggml_type)) {
    *out_nbytes = nelems * static_cast<uint64_t>(ElementSize(ggml_type));
    return true;
  }
  if (IsKQuant(ggml_type)) {
    const uint64_t block_elems =
        static_cast<uint64_t>(KQuantBlockElements(ggml_type));
    if (nelems % block_elems != 0) {
      return false;
    }
    *out_nbytes = (nelems / block_elems) *
                  static_cast<uint64_t>(KQuantBlockBytes(ggml_type));
    return true;
  }
  return false;
}

// Inverse of ToOnnx. Returns false for an ONNX dtype with no raw ggml_type
// counterpart (STRING, COMPLEX64/128, UNDEFINED, the unsigned int family --
// ggml has no unsigned integer types at all, quantized or otherwise -- or
// one of onnxsim's own custom dtypes this mapping doesn't cover). The four
// ONNXSIM_GGML_* codes DO round-trip here (unlike every other private dtype
// a caller might construct): a pooled K-quant entry that still holds its
// native, un-decoded block bytes (see the file comment) can always be
// written back out to a real GGUF file bit-for-bit, the same as any other
// pooled dtype -- TensorPool::SaveGGUF relies on that.
inline bool FromOnnx(int32_t onnx_dtype, uint32_t* out) {
  switch (onnx_dtype) {
    case ONNX_FLOAT:
      *out = GGML_TYPE_F32;
      return true;
    case ONNX_FLOAT16:
      *out = GGML_TYPE_F16;
      return true;
    case ONNX_BFLOAT16:
      *out = GGML_TYPE_BF16;
      return true;
    case ONNX_DOUBLE:
      *out = GGML_TYPE_F64;
      return true;
    case ONNX_INT8:
      *out = GGML_TYPE_I8;
      return true;
    case ONNX_INT16:
      *out = GGML_TYPE_I16;
      return true;
    case ONNX_INT32:
      *out = GGML_TYPE_I32;
      return true;
    case ONNX_INT64:
      *out = GGML_TYPE_I64;
      return true;
    case ONNXSIM_GGML_Q4_K:
      *out = GGML_TYPE_Q4_K;
      return true;
    case ONNXSIM_GGML_Q5_K:
      *out = GGML_TYPE_Q5_K;
      return true;
    case ONNXSIM_GGML_Q6_K:
      *out = GGML_TYPE_Q6_K;
      return true;
    case ONNXSIM_GGML_Q8_0:
      *out = GGML_TYPE_Q8_0;
      return true;
    default:
      return false;
  }
}

}  // namespace gguf
}  // namespace tensor_pool
}  // namespace onnxsim

#endif  // ONNXSIM_GGUF_DTYPE_H_
