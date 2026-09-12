// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Retypes an all-float16 graph to float32 -- the widening mirror of
// quantize_fp16.h's QuantizeFp16Pass, and a port of
// scripts/axera/legalize.py's float16_to_float32 (the "port scripts/axera/
// legalize.py rules to onnxsim core" migration this project has been
// running, alongside pow2_to_mul.h/explicit_conv_padding.h/neg_to_mul.h;
// see docs/axera-legalize-onnxsim-core-migration.md). It exists for the
// Axera AX650N Pulsar2 compiler, which takes float32 only: an fp16 export
// has its constants in three places, and converting only one leaves a graph
// that mixes precisions, which onnxruntime rejects outright at load time.
//
// Unlike QuantizeFp16Pass, this is unconditional and total, not a partial/
// mixed-precision transform with a keep_io_types choice: every FLOAT16
// declaration anywhere in the graph -- constant tensors (initializers and
// Constant nodes), a Cast node's `to` attribute, and every Value's own
// elemType (graph inputs, graph outputs, and interior node outputs alike)
// -- becomes FLOAT, full stop. That is a safe, always-correct rewrite
// exactly because it is total: unlike QuantizeFp16Pass (which cannot assume
// every float32-declared output actually becomes float16, since a mixed-
// precision graph may deliberately keep some values at float32, e.g. past a
// Cast(to=FLOAT) placed mid-graph on purpose) this pass's precondition is
// that the *whole* graph is already float16, so every FLOAT16-declared
// Value's true new type is unambiguously FLOAT once its producers are
// retyped -- no guessing, no wiping-to-UNDEFINED-and-letting-a-later-pass-
// re-infer needed.
//
// float16->float32 is also exact (every float16 value has a precise float32
// representation), unlike float32->float16 (QuantizeFp16Pass's
// FloatToFloat16Bits rounds and clamps) -- so this pass never loses
// precision or changes a single computed value, only how it is spelled.
//
// Only the top-level graph is converted -- nodes inside control-flow
// subgraphs (If/Loop/Scan bodies) are left untouched, the same known
// limitation quantize_fp16.h documents for itself.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/endian_read.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

// Converts the bit pattern of an IEEE 754 binary16 (float16) value to the
// float32 value it exactly represents -- the widening direction is always
// exact, unlike quantize_fp16.h's FloatToFloat16Bits (which rounds).
inline float Float16BitsToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exp16 = (bits >> 10) & 0x1Fu;
  uint32_t mant16 = bits & 0x3FFu;

  uint32_t out_bits;
  if (exp16 == 0) {
    if (mant16 == 0) {
      out_bits = sign;  // +-0
    } else {
      // Subnormal float16: shift the mantissa left until its leading 1
      // lands in the implicit-bit position, adjusting the exponent to
      // match, then rebias into float32's exponent range.
      int32_t shift = 0;
      while ((mant16 & 0x400u) == 0) {
        mant16 <<= 1;
        shift += 1;
      }
      mant16 &= 0x3FFu;
      const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - shift);
      out_bits = sign | (exp32 << 23) | (mant16 << 13);
    }
  } else if (exp16 == 0x1Fu) {
    // Inf/NaN: float16's max exponent maps to float32's max exponent, the
    // mantissa carried through unchanged (0 for Inf, nonzero for NaN).
    out_bits = sign | (0xFFu << 23) | (mant16 << 13);
  } else {
    const uint32_t exp32 = exp16 - 15u + 127u;
    out_bits = sign | (exp32 << 23) | (mant16 << 13);
  }

  float out;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

// Reads a constant float16 tensor's bit patterns, widened to a flat float32
// buffer -- the read-side counterpart of quantize_fp16.h's
// ConvertFloatTensorToFp16, direction reversed. float16 has no dedicated
// typed TensorProto field (see WriteRawDataLittleEndian's doc comment in
// endian_read.h): onnx.numpy_helper.from_array always uses raw_data for it,
// but int32_data (each entry the 16-bit pattern zero-extended) is also
// spec-legal, so both are handled here.
inline std::vector<float> ReadFloat16TensorFlat(const Tensor& t) {
  int64_t numel = 1;
  for (const auto& s : t.sizes()) {
    numel *= s;
  }
  std::vector<uint16_t> bits;
  if (t.is_raw_data()) {
    // Tensor::data<T>() is only explicitly instantiated for the handful of
    // types ir.h's own define_data() macro lists (float/double/int32_t/
    // int64_t/uint64_t) -- uint16_t is not among them, so raw_data is read
    // by reinterpreting raw() directly instead, the same way it is stored
    // (ConvertFloat16TensorToFloat32 below writes uint16_t bit patterns via
    // WriteRawDataLittleEndian, never through data<uint16_t>() either).
    bits = ReadRawDataHostOrder<uint16_t>(
        reinterpret_cast<const uint16_t*>(t.raw().data()), numel);
  } else {
    bits.reserve(static_cast<size_t>(numel));
    for (int32_t v : t.int32s()) {
      bits.push_back(static_cast<uint16_t>(v));
    }
  }
  std::vector<float> out(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) {
    out[i] = Float16BitsToFloat(bits[i]);
  }
  return out;
}

// Converts a constant float16 tensor of any rank/shape to float32, keeping
// the same shape.
inline Tensor ConvertFloat16TensorToFloat32(const Tensor& t) {
  const std::vector<float> data = ReadFloat16TensorFlat(t);
  Tensor out;
  out.elem_type() = TensorProto_DataType_FLOAT;
  out.sizes() = t.sizes();
  out.set_raw_data(WriteRawDataLittleEndian(data));
  return out;
}

struct Float16ToFloat32Pass final : public FullGraphBasedPass {
  explicit Float16ToFloat32Pass()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::None) {}
  std::string getPassName() const override { return "float16_to_float32"; }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    // 1. Every constant float16 tensor (a true initializer, or a Constant
    // node's embedded value -- FetchConstantTensor covers both uniformly)
    // becomes float32, once per unique value.
    std::unordered_set<std::string> seen;
    std::vector<Value*> candidates;
    for (Node* n : graph.nodes()) {
      for (Value* in : n->inputs()) {
        if (!seen.insert(in->uniqueName()).second) {
          continue;
        }
        const Tensor* t = FetchConstantTensor(in);
        if (t != nullptr && t->elem_type() == TensorProto_DataType_FLOAT16) {
          candidates.push_back(in);
        }
      }
    }
    for (Value* old_v : candidates) {
      const Tensor* t = FetchConstantTensor(old_v);
      if (t == nullptr) {
        continue;  // Defensive: shouldn't happen, nothing else touches these.
      }
      Tensor f32_t = ConvertFloat16TensorToFloat32(*t);
      Value* new_v = graph.addInitializerAndCreateValue(f32_t);
      tryReplacingAllUsesWith(old_v, new_v);
    }

    // 2. Every Cast node's `to` attribute, wherever it names FLOAT16.
    for (Node* n : graph.nodes()) {
      if (n->kind() == kCast && n->hasAttribute(kto) &&
          n->i(kto) == static_cast<int64_t>(TensorProto_DataType_FLOAT16)) {
        n->i_(kto, static_cast<int64_t>(TensorProto_DataType_FLOAT));
      }
    }

    // 3. Every Value's own declared elemType -- graph inputs, graph
    // outputs, and every node's output alike. Safe to relabel directly
    // (rather than wipe, as QuantizeFp16Pass does for its narrowing,
    // partial-conversion case) because this pass's precondition is a
    // wholly float16 graph: once step 1 retypes every constant that feeds
    // it, a Value still declared FLOAT16 has no other correct type left to
    // have.
    std::unordered_set<Value*> retyped;
    for (Value* v : graph.inputs()) {
      if (v->elemType() == TensorProto_DataType_FLOAT16) {
        v->setElemType(TensorProto_DataType_FLOAT);
        retyped.insert(v);
      }
    }
    for (Value* v : graph.outputs()) {
      if (v->elemType() == TensorProto_DataType_FLOAT16) {
        v->setElemType(TensorProto_DataType_FLOAT);
        retyped.insert(v);
      }
    }
    for (Node* n : graph.nodes()) {
      for (Value* out : n->outputs()) {
        if (out->elemType() == TensorProto_DataType_FLOAT16) {
          out->setElemType(TensorProto_DataType_FLOAT);
          retyped.insert(out);
        }
      }
    }

    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
