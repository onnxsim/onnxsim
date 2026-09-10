/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qonnx_schemas.h"

#include <mutex>
#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxsim {

namespace {

using onnx::OpSchema;

// `Quant`/`BipolarQuant`/`Trunc`/`FloatQuant` are registered under both of
// these domains: `qonnx.custom_op.general` is the current name (the `qonnx`
// package, https://github.com/fastmachinelearning/qonnx), and
// `finn.custom_op.general` is its predecessor, still emitted by Brevitas
// versions built against the original FINN compiler's own custom-op package
// before the QONNX split. A model exported by either version is common in
// the wild, so both are covered identically rather than guessing which one a
// given file used.
constexpr const char* kDomains[] = {"qonnx.custom_op.general",
                                    "finn.custom_op.general"};

// Deliberately generous, matching bev_custom_op_schemas.cpp's own reasoning:
// a schema narrower than what real exports actually produce would make a
// previously-tolerated model fail onnx::checker::check_model, which is worse
// than not registering a schema at all. Brevitas quantizes float32, float16
// and bfloat16 activations/weights; `double` is included for completeness
// with ONNX's own float family.
const std::vector<std::string>& FloatTypes() {
  static const std::vector<std::string> types = {
      "tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(double)"};
  return types;
}

// Registers `schema` unless a schema for the same (name, domain) pair is
// already known. Never fails or throws: registering the same schema twice
// (once per process, or once per model simplified in the same process) is a
// harmless no-op.
void RegisterIfAbsent(OpSchema&& schema) {
  const std::string name = schema.Name();
  const std::string domain = schema.domain();
  if (onnx::OpSchemaRegistry::Schema(name, domain) != nullptr) {
    return;
  }
  onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                       /*fail_duplicate_schema=*/false,
                       /*fail_with_exception=*/false);
}

// Every op here is a "fake-quantize then immediately dequantize back to
// float" node: it quantizes its first input to an integer (or minifloat)
// grid parameterized by the remaining inputs/attributes and returns the
// result of dequantizing that straight back, in place -- same shape, same
// element type as the input being quantized.
// `propagateShapeAndTypeFromFirstInput` (used the same way for
// `com.microsoft`'s QLinearSigmoid/QLinearLeakyRelu/ QLinearSoftmax in
// contrib_schemas.cpp) is exactly that contract.
OpSchema MakeQuantSchema(const char* domain) {
  return OpSchema()
      .SetName("Quant")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "QONNX/FINN generic integer fake-quantizer, as emitted by "
          "Brevitas's default `export_qonnx` path: quantizes `X` to a "
          "`bitwidth`-wide integer grid using `scale`/`zeropoint` (in the "
          "same `(round(X/scale) + zeropoint)` convention as ONNX's own "
          "QuantizeLinear/DequantizeLinear), then dequantizes the result "
          "straight back to `X`'s own float dtype. Unlike standard "
          "`QuantizeLinear`, `bitwidth` is a runtime input rather than "
          "implied by the output's integer dtype, so it represents "
          "arbitrary (not just 8-bit) precision, including values learned "
          "during quantization-aware training.")
      .Attr("signed",
            "1 if the quantized grid is signed (e.g. int4), 0 if unsigned "
            "(e.g. uint4).",
            onnx::AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("narrow",
            "1 to exclude the most-negative signed code from the grid "
            "(narrow range), 0 to include it.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("rounding_mode",
            "Rounding rule used to map X/scale + zeropoint onto the integer "
            "grid (e.g. \"ROUND\", \"CEIL\", \"FLOOR\").",
            onnx::AttributeProto::STRING, std::string("ROUND"))
      .Input(0, "X", "Tensor to quantize.", "T")
      .Input(1, "scale", "Quantization scale, per-tensor or broadcastable.",
             "T")
      .Input(2, "zeropoint",
             "Quantization zero-point in the integer-code domain, stored as "
             "`T` (rather than an integer dtype) so it can itself be a "
             "trained value.",
             "T")
      .Input(3, "bitwidth",
             "Bit-width of the integer grid, as a scalar float tensor.", "T")
      .Output(0, "Xq",
              "`X`, fake-quantized to `bitwidth` bits and dequantized back.",
              "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain X/scale/zeropoint/bitwidth/Xq to float "
                      "tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput)
      .AllowUncheckedAttributes();
}

OpSchema MakeBipolarQuantSchema(const char* domain) {
  return OpSchema()
      .SetName("BipolarQuant")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "QONNX/FINN 1-bit fake-quantizer: `Xq = scale * sign(X)` (with "
          "`sign(0) = +1`), the fixed {-1, +1} special case `Quant` would "
          "need `bitwidth = 1`, `signed = 1` for, split into its own op "
          "because a single-bit grid has no separate zero-point or rounding "
          "mode to speak of.")
      .Input(0, "X", "Tensor to quantize.", "T")
      .Input(1, "scale", "Quantization scale, per-tensor or broadcastable.",
             "T")
      .Output(0, "Xq", "`X`, fake-quantized to {-scale, +scale}.", "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain X/scale/Xq to float tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput)
      .AllowUncheckedAttributes();
}

OpSchema MakeTruncSchema(const char* domain) {
  return OpSchema()
      .SetName("Trunc")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "QONNX/FINN bit-width-reducing truncation: dequantizes `X` from "
          "`input_bit_width` bits with `scale`/`zeropoint`, right-shifts to "
          "`output_bit_width` bits, and re-dequantizes. Brevitas emits this "
          "for the accumulator truncation an average-pool (or any "
          "quantized reduction) needs to stay within its declared output "
          "bit-width.")
      .Attr("rounding_mode",
            "Rounding rule used when discarding the low bits (e.g. "
            "\"ROUND\", \"FLOOR\").",
            onnx::AttributeProto::STRING, std::string("ROUND"))
      .Input(0, "X", "Tensor to truncate.", "T")
      .Input(1, "scale", "Quantization scale, per-tensor or broadcastable.",
             "T")
      .Input(2, "zeropoint",
             "Quantization zero-point in the integer-code domain, stored as "
             "`T`.",
             "T")
      .Input(3, "input_bit_width",
             "Bit-width `X` is already quantized to, as a scalar float "
             "tensor.",
             "T")
      .Input(4, "output_bit_width",
             "Bit-width to truncate down to, as a scalar float tensor.", "T")
      .Output(0, "Xq", "`X`, truncated to `output_bit_width` bits.", "T")
      .TypeConstraint(
          "T", FloatTypes(),
          "Constrain X/scale/zeropoint/*_bit_width/Xq to float tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput)
      .AllowUncheckedAttributes();
}

OpSchema MakeFloatQuantSchema(const char* domain) {
  return OpSchema()
      .SetName("FloatQuant")
      .SetDomain(domain)
      .SinceVersion(1)
      .SetDoc(
          "QONNX/FINN generic minifloat fake-quantizer: quantizes `X` to a "
          "floating-point grid with `exponent_bitwidth` exponent bits, "
          "`mantissa_bitwidth` mantissa bits, `exponent_bias`, and a "
          "saturation point `max_val`, then dequantizes the result back to "
          "`X`'s own dtype -- the minifloat counterpart of `Quant` (e.g. for "
          "FP8/FP4-style Brevitas exports), added to QONNX after the "
          "original integer-only `Quant`/`BipolarQuant`/`Trunc` set.")
      .Attr("signed", "1 if the minifloat grid is signed, 0 if unsigned.",
            onnx::AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("narrow",
            "1 to exclude the grid's most-negative code (narrow range), 0 "
            "to include it.",
            onnx::AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("rounding_mode",
            "Rounding rule used to map X onto the minifloat grid (e.g. "
            "\"ROUND\").",
            onnx::AttributeProto::STRING, std::string("ROUND"))
      .Input(0, "X", "Tensor to quantize.", "T")
      .Input(1, "scale", "Quantization scale, per-tensor or broadcastable.",
             "T")
      .Input(2, "exponent_bitwidth",
             "Exponent bit-width of the minifloat grid, as a scalar float "
             "tensor.",
             "T")
      .Input(3, "mantissa_bitwidth",
             "Mantissa bit-width of the minifloat grid, as a scalar float "
             "tensor.",
             "T")
      .Input(4, "exponent_bias",
             "Exponent bias of the minifloat grid, as a scalar float "
             "tensor.",
             "T")
      .Input(5, "max_val",
             "Saturation magnitude of the minifloat grid, as a scalar float "
             "tensor.",
             "T")
      .Output(0, "Xq", "`X`, fake-quantized to the minifloat grid.", "T")
      .TypeConstraint("T", FloatTypes(),
                      "Constrain every float input/output to float tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput)
      .AllowUncheckedAttributes();
}

void RegisterAll() {
  // Both custom domains must be known to the schema registry before any
  // schema in them can be registered -- see bev_custom_op_schemas.cpp's own
  // `RegisterAll` for what happens if this step is skipped (RegisterSchema
  // silently no-ops instead of failing loudly).
  auto& domain_range = onnx::OpSchemaRegistry::DomainToVersionRange::Instance();
  for (const char* domain : kDomains) {
    if (domain_range.Map().count(domain) == 0) {
      domain_range.AddDomainToVersion(domain, /*min_version=*/1,
                                      /*max_version=*/1);
    }
  }

  for (const char* domain : kDomains) {
    RegisterIfAbsent(MakeQuantSchema(domain));
    RegisterIfAbsent(MakeBipolarQuantSchema(domain));
    RegisterIfAbsent(MakeTruncSchema(domain));
    RegisterIfAbsent(MakeFloatQuantSchema(domain));
  }
}

}  // namespace

void RegisterQonnxCustomOpSchemas() {
  static std::once_flag once;
  std::call_once(once, RegisterAll);
}

}  // namespace onnxsim
