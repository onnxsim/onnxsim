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
