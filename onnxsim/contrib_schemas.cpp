/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "contrib_schemas.h"

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/defs/data_propagators.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxsim {

namespace {

constexpr const char* kMSDomain = "com.microsoft";

using onnx::DataPropagationContext;
using onnx::InferenceContext;
using onnx::OpSchema;

// Attach a data-propagation function to `Reshape`. ONNX ships data-propagation
// functions for the shape family (Shape, Gather, Concat, Slice, Squeeze,
// Unsqueeze, Add/Sub/Mul, ...) but not for Reshape, so a shape tensor threaded
// through a Reshape -- e.g. `Shape(x) -> Reshape(., [-1]) -> Gather(...)` --
// loses its propagated value there and downstream shape arithmetic stops
// folding.
//
// A Reshape only rearranges a tensor's dims; it never changes the number of
// elements or their row-major order. Data propagation tracks a shape tensor's
// *value* as a flat, ordered list of its elements, so that list is invariant
// under a Reshape and can be copied straight through -- the same reasoning, and
// the same helper (PropagateShapeDataFromInputToOutput), that ONNX uses for
// Squeeze/Unsqueeze, which likewise only add or remove size-1 axes.
//
// The schema objects are owned by the registry and Schema() returns a pointer
// into that storage, so we const_cast to augment them in place. Data
// propagation only runs when explicitly enabled (onnxsim's partial-shape pass),
// so this never affects ordinary shape inference.
void RegisterReshapeDataPropagation() {
  std::unordered_set<const OpSchema*> augmented;
  for (int ver = 1; ver <= 64; ++ver) {
    const OpSchema* schema =
        onnx::OpSchemaRegistry::Schema("Reshape", ver, onnx::ONNX_DOMAIN);
    if (schema == nullptr || augmented.count(schema)) {
      continue;
    }
    augmented.insert(schema);
    const_cast<OpSchema*>(schema)->PartialDataPropagationFunction(
        [](DataPropagationContext& ctx) {
          onnx::PropagateShapeDataFromInputToOutput(ctx, 0);
        });
  }
}

// Shape/type inference for the element-wise binary quantized ops (QLinearAdd,
// QLinearMul). Inputs are laid out as
//   A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point
// so the two data tensors that determine the output shape are inputs 0 and 3.
void QLinearBinaryShapeInference(InferenceContext& ctx) {
  // The output is quantized to the same element type as the first operand.
  onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (onnx::hasInputShape(ctx, 0) && onnx::hasInputShape(ctx, 3)) {
    onnx::bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(3)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  }
}

// Shape/type inference for QLinearConcat. Inputs are
//   Y_scale, Y_zero_point, (T, T_scale, T_zero_point)+
// The output element type follows Y_zero_point (input 1) and the shape is the
// concatenation of the data tensors (inputs 2, 5, 8, ...) along `axis`.
void QLinearConcatShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 1, 0);

  const auto* axis_attr = ctx.getAttribute("axis");
  if (axis_attr == nullptr || !axis_attr->has_i()) {
    return;
  }
  int64_t axis = axis_attr->i();

  std::vector<size_t> data_indices;
  for (size_t i = 2; i < ctx.getNumInputs(); i += 3) {
    data_indices.push_back(i);
  }
  if (data_indices.empty()) {
    return;
  }

  // Every data tensor must have a known rank and the ranks must agree.
  int rank = -1;
  for (size_t idx : data_indices) {
    if (!onnx::hasInputShape(ctx, idx)) {
      return;
    }
    const int cur_rank =
        ctx.getInputType(idx)->tensor_type().shape().dim_size();
    if (rank == -1) {
      rank = cur_rank;
    } else if (rank != cur_rank) {
      // Inconsistent ranks: leave the output shape unset rather than guessing.
      return;
    }
  }
  if (rank <= 0) {
    return;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return;
  }

  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape->clear_dim();
  for (int i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  bool axis_dim_known = true;
  int64_t axis_dim_sum = 0;
  for (size_t idx : data_indices) {
    const auto& shape = ctx.getInputType(idx)->tensor_type().shape();
    for (int d = 0; d < rank; ++d) {
      const auto& dim = shape.dim(d);
      if (d == axis) {
        if (dim.has_dim_value()) {
          axis_dim_sum += dim.dim_value();
        } else {
          axis_dim_known = false;
        }
        continue;
      }
      // Non-axis dimensions must match across inputs; keep the most specific
      // information we can (a concrete value, otherwise a symbolic name).
      auto* out_dim = output_shape->mutable_dim(d);
      if (!out_dim->has_dim_value() && dim.has_dim_value()) {
        out_dim->set_dim_value(dim.dim_value());
      } else if (!out_dim->has_dim_value() && !out_dim->has_dim_param() &&
                 dim.has_dim_param()) {
        out_dim->set_dim_param(dim.dim_param());
      }
    }
  }
  if (axis_dim_known) {
    output_shape->mutable_dim(axis)->set_dim_value(axis_dim_sum);
  }
}

// Shape/type inference for QLinearGlobalAveragePool. Per ONNX Runtime's own
// doc ("the output tensor has the same rank as the input, with the N and C
// value keep[ing] its value, while the other dimensions are all 1"), the
// output shape is fully determined by the input shape and rank -- unlike
// QLinearAveragePool below, whose true output shape additionally depends on
// kernel_shape/strides/pads/ceil_mode/auto_pad arithmetic this function
// does not replicate.
void QLinearGlobalAveragePoolShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!onnx::hasInputShape(ctx, 0)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const int rank = input_shape.dim_size();
  if (rank < 2) {
    return;
  }
  int64_t channels_last = 0;
  const auto* channels_last_attr = ctx.getAttribute("channels_last");
  if (channels_last_attr != nullptr && channels_last_attr->has_i()) {
    channels_last = channels_last_attr->i();
  }
  const int channel_dim = channels_last != 0 ? rank - 1 : 1;

  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape->clear_dim();
  for (int i = 0; i < rank; ++i) {
    if (i == 0 || i == channel_dim) {
      *output_shape->add_dim() = input_shape.dim(i);
    } else {
      output_shape->add_dim()->set_dim_value(1);
    }
  }
}

// Shape/type inference for QLinearWhere. Inputs are laid out as
//   condition, X, x_scale, x_zero_point, Y, y_scale, y_zero_point,
//   z_scale, z_zero_point
// so the output's element type follows X (input 1) and its shape is the
// 3-way broadcast of condition (0), X (1), and Y (4) -- exactly mirroring
// ONNX Runtime's own QLinearWhere inference function.
void QLinearWhereShapeInference(InferenceContext& ctx) {
  onnx::propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (!onnx::hasNInputShapes(ctx, 9)) {
    return;
  }
  std::vector<const onnx::TensorShapeProto*> shapes;
  shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
  shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
  shapes.push_back(&ctx.getInputType(4)->tensor_type().shape());
  onnx::multidirectionalBroadcastShapeInference(
      shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

// Shape/type inference for QGemm. The output's element type follows
// `y_zero_point` (input 8) when present -- meaning the output is itself
// quantized -- else it is plain float32 (QGemm's schema allows an
// unquantized float output when `y_scale`/`y_zero_point` are omitted; this
// onnxsim pass never omits them, always producing the quantized-output
// form, but the schema itself supports both). The output shape is the
// standard Gemm (M, N) computed from A's and B's 2-D shapes and the
// transA/transB attributes.
void QGemmShapeInference(InferenceContext& ctx) {
  if (ctx.getNumInputs() == 9 && ctx.getInputType(8) != nullptr) {
    onnx::propagateElemTypeFromInputToOutput(ctx, 8, 0);
  } else {
    onnx::updateOutputElemType(ctx, 0, onnx::TensorProto::FLOAT);
  }
  if (!onnx::hasInputShape(ctx, 0) || !onnx::hasInputShape(ctx, 3)) {
    return;
  }
  const auto& a_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto& b_shape = ctx.getInputType(3)->tensor_type().shape();
  if (a_shape.dim_size() != 2 || b_shape.dim_size() != 2) {
    return;
  }
  int64_t trans_a = 0;
  const auto* trans_a_attr = ctx.getAttribute("transA");
  if (trans_a_attr != nullptr && trans_a_attr->has_i()) {
    trans_a = trans_a_attr->i();
  }
  int64_t trans_b = 0;
  const auto* trans_b_attr = ctx.getAttribute("transB");
  if (trans_b_attr != nullptr && trans_b_attr->has_i()) {
    trans_b = trans_b_attr->i();
  }
  auto* output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  output_shape->clear_dim();
  *output_shape->add_dim() = a_shape.dim(trans_a != 0 ? 1 : 0);
  *output_shape->add_dim() = b_shape.dim(trans_b != 0 ? 0 : 1);
}

// Registers `schema` unless an equivalent schema is already known. Duplicate
// registration is turned into a no-op instead of an error so the function stays
// safe to run alongside a build that already provides these schemas.
void RegisterIfAbsent(OpSchema&& schema) {
  const std::string name = schema.Name();
  if (onnx::OpSchemaRegistry::Schema(name, kMSDomain) != nullptr) {
    return;
  }
  onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                       /*fail_duplicate_schema=*/false,
                       /*fail_with_exception=*/false);
}

OpSchema MakeQLinearBinarySchema(const char* name) {
  return OpSchema()
      .SetName(name)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized element-wise binary op contributed by ONNX Runtime.")
      .Input(0, "A", "First quantized operand.", "T")
      .Input(1, "A_scale", "Scale of A.", "tensor(float)")
      .Input(2, "A_zero_point", "Zero point of A.", "T", OpSchema::Optional)
      .Input(3, "B", "Second quantized operand.", "T")
      .Input(4, "B_scale", "Scale of B.", "tensor(float)")
      .Input(5, "B_zero_point", "Zero point of B.", "T", OpSchema::Optional)
      .Input(6, "C_scale", "Scale of the output C.", "tensor(float)")
      .Input(7, "C_zero_point", "Zero point of the output C.", "T",
             OpSchema::Optional)
      .Output(0, "C", "Quantized result.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer tensors.")
      .TypeAndShapeInferenceFunction(QLinearBinaryShapeInference);
}

OpSchema MakeQLinearUnarySchema(const char* name, bool has_alpha) {
  OpSchema schema;
  schema.SetName(name)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized element-wise unary op contributed by ONNX Runtime.")
      .Input(0, "X", "Quantized input.", "T")
      .Input(1, "X_scale", "Scale of X.", "tensor(float)")
      .Input(2, "X_zero_point", "Zero point of X.", "T", OpSchema::Optional)
      .Input(3, "Y_scale", "Scale of the output Y.", "tensor(float)")
      .Input(4, "Y_zero_point", "Zero point of the output Y.", "T",
             OpSchema::Optional)
      .Output(0, "Y", "Quantized output.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput);
  if (has_alpha) {
    schema.Attr("alpha", "Coefficient of leakage.", onnx::AttributeProto::FLOAT,
                0.01f);
  }
  return schema;
}

OpSchema MakeQLinearConcatSchema() {
  return OpSchema()
      .SetName("QLinearConcat")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized concatenation contributed by ONNX Runtime.")
      .Attr("axis", "Axis to concatenate on.", onnx::AttributeProto::INT,
            /*required=*/true)
      .Input(0, "Y_scale", "Scale of the output Y.", "TF")
      .Input(1, "Y_zero_point", "Zero point of the output Y.", "T8")
      .Input(2, "inputs",
             "Repeated (tensor, scale, zero_point) triples to concatenate.",
             "TV", OpSchema::Variadic, /*is_homogeneous=*/false)
      .Output(0, "Y", "Concatenated quantized result.", "T8")
      .TypeConstraint("T8", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain quantized tensors to 8-bit integers.")
      .TypeConstraint("TF", {"tensor(float)"}, "Constrain scales to float.")
      .TypeConstraint("TV", {"tensor(uint8)", "tensor(int8)", "tensor(float)"},
                      "Constrain the variadic inputs.")
      .TypeAndShapeInferenceFunction(QLinearConcatShapeInference);
}

// Same layout/attribute set ONNX Runtime itself registers for
// "com.microsoft" QLinearSoftmax: `X`/`Y` share a single `T` type constraint
// (8-bit signed or unsigned), the output's shape and element type simply
// follow the input's (Softmax never changes shape), and the `opset`
// attribute is required -- it tells the runtime kernel which of standard
// ONNX's two incompatible `Softmax` axis semantics (pre-13 flattening vs.
// 13+ in-place per-axis reduction) to replicate. Unlike
// MakeQLinearUnarySchema's `Y_zero_point` (optional, since QLinearSigmoid/
// QLinearLeakyRelu allow a runtime-implied default), QLinearSoftmax's
// `y_zero_point` is required.
OpSchema MakeQLinearSoftmaxSchema() {
  return OpSchema()
      .SetName("QLinearSoftmax")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "QLinearSoftmax computes the normalized exponential values for "
          "the given input: Softmax(input, axis) = Exp(input) / "
          "ReduceSum(Exp(input), axis=axis, keepdims=1).")
      .Attr("axis", "Apply softmax to elements for dimensions axis.",
            onnx::AttributeProto::INT, static_cast<int64_t>(-1))
      .Attr("opset",
            "Opset version of the standard-ONNX Softmax whose axis "
            "semantics this node replicates.",
            onnx::AttributeProto::INT)
      .Input(0, "X", "The input tensor.", "T")
      .Input(1, "X_scale", "Scale of quantized input X. Must be a scalar.",
             "tensor(float)")
      .Input(2, "x_zero_point",
             "Zero point of quantized input X. Must be a scalar.", "T",
             OpSchema::Optional)
      .Input(3, "y_scale", "Scale of quantized output Y. Must be a scalar.",
             "tensor(float)")
      .Input(4, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "T")
      .Output(0, "Y", "Output data tensor.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output types to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(onnx::propagateShapeAndTypeFromFirstInput);
}

// Same layout/attribute set ONNX Runtime itself registers for
// "com.microsoft" QLinearAveragePool: every attribute standard ONNX
// AveragePool has (kernel_shape required; auto_pad/ceil_mode/
// count_include_pad/pads/strides optional, matching AveragePool's own
// defaults), plus a `channels_last` attribute AveragePool itself doesn't
// have. Type/shape inference only propagates the element type -- the true
// output shape depends on the same kernel/stride/pad arithmetic standard
// AveragePool's own inference function implements, which this schema does
// not replicate (qoperator_quantize_pool.h's rewrite doesn't need it either:
// it copies the original node's already-known output shape onto the
// trailing DequantizeLinear directly).
OpSchema MakeQLinearAveragePoolSchema() {
  return OpSchema()
      .SetName("QLinearAveragePool")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "QLinearAveragePool consumes an input tensor X and applies "
          "average pooling across the tensor according to kernel sizes, "
          "stride sizes and pad lengths, computing on dequantized values "
          "and requantizing the result.")
      .Attr("auto_pad",
            "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or "
            "VALID (deprecated, kept for parity with standard ONNX "
            "AveragePool).",
            onnx::AttributeProto::STRING, "NOTSET")
      .Attr("ceil_mode",
            "Whether to use ceil or floor (default) to compute the output "
            "shape.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("channels_last", "Works on NHWC layout or not. Default not.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("count_include_pad",
            "Whether to include pad pixels when calculating values for "
            "the edges. Default 0, doesn't count include pad.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("kernel_shape", "The size of the kernel along each axis.",
            onnx::AttributeProto::INTS, /*required=*/true)
      .Attr("pads",
            "Padding for the beginning and ending along each spatial "
            "axis. Defaults to 0 along start and end of each spatial axis "
            "when absent.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Attr("strides",
            "Stride along each spatial axis. Defaults to 1 along each "
            "spatial axis when absent.",
            onnx::AttributeProto::INTS, /*required=*/false)
      .Input(0, "X", "Input data tensor from the previous operator.", "T")
      .Input(1, "x_scale", "Scale of quantized input X. Must be a scalar.",
             "tensor(float)")
      .Input(2, "x_zero_point",
             "Zero point of quantized input X. Must be a scalar.", "T",
             OpSchema::Optional)
      .Input(3, "y_scale", "Scale of quantized output Y. Must be a scalar.",
             "tensor(float)")
      .Input(4, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "T",
             OpSchema::Optional)
      .Output(0, "Y", "Output data tensor from average pooling.", "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        onnx::propagateElemTypeFromInputToOutput(ctx, 0, 0);
      });
}

// Same layout ONNX Runtime itself registers for "com.microsoft"
// QLinearGlobalAveragePool: unlike QLinearAveragePool, both zero-points are
// required (not optional), there are no kernel_shape/strides/pads/etc.
// attributes (it always pools over every spatial position), and the output
// shape is simple enough (same rank, N/C kept, every other dim collapsed to
// 1) that QLinearGlobalAveragePoolShapeInference computes it exactly.
OpSchema MakeQLinearGlobalAveragePoolSchema() {
  return OpSchema()
      .SetName("QLinearGlobalAveragePool")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc(
          "QLinearGlobalAveragePool consumes an input tensor X and applies "
          "average pooling across the values in the same channel. This is "
          "equivalent to AveragePool with kernel size equal to the "
          "spatial dimensions of the input tensor.")
      .Attr("channels_last", "Works on NHWC layout or not. Default not.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "X", "Input data tensor from the previous operator.", "T")
      .Input(1, "x_scale", "Scale of quantized input X. Must be a scalar.",
             "tensor(float)")
      .Input(2, "x_zero_point",
             "Zero point of quantized input X. Must be a scalar.", "T")
      .Input(3, "y_scale", "Scale of quantized output Y. Must be a scalar.",
             "tensor(float)")
      .Input(4, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "T")
      .Output(0, "Y",
              "Output data tensor from pooling across the input "
              "tensor.",
              "T")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(QLinearGlobalAveragePoolShapeInference);
}

// Same layout ONNX Runtime itself registers for "com.microsoft"
// QLinearWhere: unlike every other schema in this file, every input is
// required (no OpSchema::Optional anywhere) -- ONNX Runtime's own doc
// strings for a couple of these inputs are copy-paste typos ("X" is
// documented as "Y's zero point.", verbatim, in ORT's own source); this
// registration keeps the exact same names/types/order but writes correct
// descriptions.
OpSchema MakeQLinearWhereSchema() {
  return OpSchema()
      .SetName("QLinearWhere")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Return elements, either from X or Y, depending on condition.")
      .Input(0, "condition", "When True (nonzero), yield X, otherwise yield Y.",
             "B")
      .Input(1, "X", "First quantized operand.", "T")
      .Input(2, "x_scale", "Scale of X.", "TF")
      .Input(3, "x_zero_point", "Zero point of X.", "T")
      .Input(4, "Y", "Second quantized operand.", "T")
      .Input(5, "y_scale", "Scale of Y.", "TF")
      .Input(6, "y_zero_point", "Zero point of Y.", "T")
      .Input(7, "z_scale", "Scale of the output Z.", "TF")
      .Input(8, "z_zero_point", "Zero point of the output Z.", "T")
      .Output(0, "Z",
              "Tensor of shape equal to the broadcasted shape of "
              "condition, X, and Y.",
              "T")
      .TypeConstraint("B", {"tensor(bool)"}, "Constrain condition to bool.")
      .TypeConstraint("TF", {"tensor(float)"}, "Constrain scales to float.")
      .TypeConstraint("T", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain input and output to 8-bit integer "
                      "tensors.")
      .TypeAndShapeInferenceFunction(QLinearWhereShapeInference);
}

// Same layout ONNX Runtime itself registers for "com.microsoft" QGemm --
// the fully-general quantized Gemm: unlike QLinearMatMul (used by
// qoperator_quantize_matmul.h for the "vanilla" transA=0/alpha=1 case),
// QGemm keeps `transA`/`transB`/`alpha` as attributes of its own, so it
// needs no forced weight-transpose or activation restriction the way
// QLinearMatMul does. `C` (bias), `y_scale`, and `y_zero_point` are all
// optional -- an omitted `C` means "as if C is a scalar 0", and omitted
// `y_scale`/`y_zero_point` means the output stays float32; onnxsim's own
// qoperator_quantize_gemm.h rewrite always supplies all three (matching
// every other pass in this family's "always fully quantize" convention),
// but the schema itself supports the leaner cases too.
OpSchema MakeQGemmSchema() {
  return OpSchema()
      .SetName("QGemm")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Quantized Gemm.")
      .Attr("transA", "Whether A should be transposed.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("transB", "Whether B should be transposed.",
            onnx::AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("alpha", "Scalar multiplier for the product of A and B.",
            onnx::AttributeProto::FLOAT, 1.0f)
      .Input(0, "A",
             "Input tensor A. Shape (M, K) if transA is 0, else (K, M).", "TA")
      .Input(1, "a_scale", "Scale of quantized input A. Must be a scalar.", "T")
      .Input(2, "a_zero_point", "Zero point of quantized input A.", "TA")
      .Input(3, "B",
             "Input tensor B. Shape (K, N) if transB is 0, else (N, K).", "TB")
      .Input(4, "b_scale",
             "Scale of quantized input B. A scalar (per-tensor) or 1-D "
             "tensor of N elements (per-column).",
             "T")
      .Input(5, "b_zero_point",
             "Zero point of quantized input B. Same shape as b_scale.", "TB")
      .Input(6, "C",
             "Optional bias tensor, unidirectionally broadcastable to "
             "(M, N). Its type is int32 and must already be quantized "
             "with zero_point = 0 and scale = alpha * a_scale * b_scale.",
             "TC", OpSchema::Optional)
      .Input(7, "y_scale",
             "Scale of quantized output Y. Must be a scalar. If omitted "
             "(along with y_zero_point), the output is float32.",
             "T", OpSchema::Optional)
      .Input(8, "y_zero_point",
             "Zero point of quantized output Y. Must be a scalar.", "TYZ",
             OpSchema::Optional)
      .Output(0, "Y", "Output tensor of shape (M, N).", "TY")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain scales to float.")
      .TypeConstraint("TA", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain A and its zero point to 8-bit tensors.")
      .TypeConstraint("TB", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain B and its zero point to 8-bit tensors.")
      .TypeConstraint("TC", {"tensor(int32)"},
                      "Constrain C to 32-bit integer tensors.")
      .TypeConstraint("TYZ", {"tensor(uint8)", "tensor(int8)"},
                      "Constrain the output zero point to 8-bit tensors.")
      .TypeConstraint("TY", {"tensor(float)", "tensor(uint8)", "tensor(int8)"},
                      "Constrain the output to float32 or 8-bit tensors.")
      .TypeAndShapeInferenceFunction(QGemmShapeInference);
}

void RegisterAll() {
  // The custom domain must be known to the schema registry before any schema
  // in it can be registered.
  auto& domain_range = onnx::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domain_range.Map().count(kMSDomain) == 0) {
    domain_range.AddDomainToVersion(kMSDomain, /*min_version=*/1,
                                    /*max_version=*/1);
  }

  RegisterIfAbsent(MakeQLinearBinarySchema("QLinearAdd"));
  RegisterIfAbsent(MakeQLinearBinarySchema("QLinearMul"));
  RegisterIfAbsent(
      MakeQLinearUnarySchema("QLinearSigmoid", /*has_alpha=*/false));
  RegisterIfAbsent(
      MakeQLinearUnarySchema("QLinearLeakyRelu", /*has_alpha=*/true));
  RegisterIfAbsent(MakeQLinearConcatSchema());
  RegisterIfAbsent(MakeQLinearSoftmaxSchema());
  RegisterIfAbsent(MakeQLinearAveragePoolSchema());
  RegisterIfAbsent(MakeQLinearGlobalAveragePoolSchema());
  RegisterIfAbsent(MakeQLinearWhereSchema());
  RegisterIfAbsent(MakeQGemmSchema());

  // Augment the standard Reshape schema with a data-propagation function so
  // shape tensors can flow through a Reshape during partial shape evaluation.
  RegisterReshapeDataPropagation();
}

}  // namespace

void RegisterContribOpSchemas() {
  static std::once_flag once;
  std::call_once(once, RegisterAll);
}

}  // namespace onnxsim
