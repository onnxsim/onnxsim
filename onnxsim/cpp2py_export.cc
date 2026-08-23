/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dlpack_bridge.h"
#include "function_rewriter.h"
#include "model_info.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/optimize.h"
#include "onnxsim.h"
#include "tensor_pool.h"
#include "tensor_pool_bridge.h"
#include "tensor_pool_gguf_bridge.h"

namespace py = nanobind;
using namespace nanobind::literals;

// nanobind type casters converting Python ``onnx`` protobuf messages (any
// object exposing ``SerializeToString``) to/from the corresponding C++ proto
// via the protobuf wire format. This mirrors ONNX's own
// ``ONNX_DEFINE_TYPE_CASTER`` so bindings can accept/return real
// ``onnx.*Proto`` objects instead of pre-serialized bytes. ``AttributeProto``
// marshals attribute defaults;
// ``TypeProto``/``NodeProto``/``TensorProto`` marshal the custom-operator shape
// inference bridge (see ``RunPythonNodeInference``).
namespace nanobind {
namespace detail {
#define ONNXSIM_PROTO_CASTER(ProtoType, PyName)                                \
  template <>                                                                  \
  struct type_caster<onnx::ProtoType> {                                        \
    NB_TYPE_CASTER(onnx::ProtoType, const_name(PyName))                        \
    bool from_python(handle src, uint8_t, cleanup_list*) noexcept {            \
      try {                                                                    \
        if (!nanobind::hasattr(src, "SerializeToString")) {                    \
          return false;                                                        \
        }                                                                      \
        auto serialized =                                                      \
            nanobind::cast<nanobind::bytes>(src.attr("SerializeToString")());  \
        return onnx::ParseProtoFromBytes(&value, serialized.c_str(),           \
                                         serialized.size());                   \
      } catch (const nanobind::python_error&) {                                \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    static handle from_cpp(const onnx::ProtoType& proto, rv_policy,            \
                           cleanup_list*) noexcept {                           \
      try {                                                                    \
        const std::string serialized = proto.SerializeAsString();              \
        auto py_proto = nanobind::module_::import_("onnx").attr(#ProtoType)(); \
        py_proto.attr("ParseFromString")(                                      \
            nanobind::bytes(serialized.c_str(), serialized.size()));           \
        return py_proto.release();                                             \
      } catch (...) {                                                          \
        return handle();                                                       \
      }                                                                        \
    }                                                                          \
  };

ONNXSIM_PROTO_CASTER(AttributeProto, "onnx.AttributeProto")
ONNXSIM_PROTO_CASTER(TypeProto, "onnx.TypeProto")
ONNXSIM_PROTO_CASTER(NodeProto, "onnx.NodeProto")
ONNXSIM_PROTO_CASTER(TensorProto, "onnx.TensorProto")
ONNXSIM_PROTO_CASTER(FunctionProto, "onnx.FunctionProto")

#undef ONNXSIM_PROTO_CASTER
}  // namespace detail
}  // namespace nanobind

namespace {

using onnx::OpSchema;

// A formal parameter (input/output) as marshalled from the Python ``onnx``
// module: (name, description, type_str, option, is_homogeneous, min_arity).
// ``option`` is the integer value of onnx's FormalParameterOption enum
// (Single=0, Optional=1, Variadic=2).
using PyFormalParameter =
    std::tuple<std::string, std::string, std::string, int, bool, int>;
// An attribute: (name, description, type, required, default_value). ``type`` is
// the integer value of onnx's AttributeProto::AttributeType enum. When
// ``default_value`` has a defined type the attribute is optional with that
// default; when its type is UNDEFINED, ``required`` decides.
using PyAttribute =
    std::tuple<std::string, std::string, int, bool, onnx::AttributeProto>;
// A type constraint: (type_param_str, allowed_type_strs, description).
using PyTypeConstraint =
    std::tuple<std::string, std::vector<std::string>, std::string>;

// Ensure ``domain`` exists in the schema registry's domain-to-version range and
// that ``version`` falls inside it, so a schema with that since_version can be
// registered. The default ONNX domain ("") is always present; custom domains
// coming from user-registered schemas usually are not, and onnx refuses to
// register a schema whose domain/version is outside the known range.
void EnsureDomainVersion(const std::string& domain, int version) {
  auto& range = onnx::OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto& map = range.Map();
  auto it = map.find(domain);
  if (it == map.end()) {
    range.AddDomainToVersion(domain, /*min_version=*/std::min(version, 1),
                             /*max_version=*/std::max(version, 1));
  } else {
    const int lo = std::min(it->second.first, version);
    const int hi = std::max(it->second.second, version);
    if (lo != it->second.first || hi != it->second.second) {
      range.UpdateDomainToVersion(domain, lo, hi);
    }
  }
}

// The default ONNX domain is stored as the empty string in the schema registry;
// "ai.onnx" is an accepted spelling of the same domain.
std::string NormalizeDomain(const std::string& domain) {
  return domain == "ai.onnx" ? std::string() : domain;
}

// C++ shape/type inference trampoline for a custom operator whose *real*
// inference function lives in the Python ``onnx`` module (registered by the
// user via ``onnx.defs.register_schema`` +
// ``set_type_and_shape_inference_function``). That function is native code
// inside the ``onnx`` library and cannot be called directly from onnxsim's
// separately linked copy, so instead we reconstruct the node and its input
// types from onnxsim's ``InferenceContext`` and hand them to
// ``onnx.shape_inference.infer_node_outputs``, which runs the Python inference
// function and returns the output types. The results are written back into the
// context so onnxsim's own shape inference (and constant folding) can use them.
//
// onnxsim's ``InferenceContext`` is positional (it exposes input/output types
// by index, not by name), so a synthetic node is built with placeholder names
// ``in0.. / out0..``; attribute values are read by the names the schema
// declares. This is invoked during ``InferShapes``, which onnxsim always runs
// while holding the GIL (it is driven synchronously from the Python
// ``simplify`` binding);
// ``gil_scoped_acquire`` is nonetheless taken to be safe. Any failure is
// swallowed so a misbehaving custom inference never aborts simplification.
void RunPythonNodeInference(onnx::InferenceContext& ctx,
                            const std::string& op_type,
                            const std::string& domain, int since_version,
                            const std::vector<std::string>& attr_names) {
  py::gil_scoped_acquire gil;
  try {
    const size_t num_inputs = ctx.getNumInputs();
    const size_t num_outputs = ctx.getNumOutputs();

    onnx::NodeProto node;
    node.set_op_type(op_type);
    node.set_domain(domain);
    for (size_t i = 0; i < num_inputs; ++i) {
      node.add_input("in" + std::to_string(i));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
      node.add_output("out" + std::to_string(i));
    }
    for (const auto& name : attr_names) {
      const onnx::AttributeProto* attr = ctx.getAttribute(name);
      if (attr != nullptr) {
        *node.add_attribute() = *attr;
      }
    }

    py::dict input_types;
    py::dict input_data;
    for (size_t i = 0; i < num_inputs; ++i) {
      const std::string key = "in" + std::to_string(i);
      const onnx::TypeProto* type = ctx.getInputType(i);
      if (type != nullptr) {
        input_types[key.c_str()] = py::cast(*type);
      }
      const onnx::TensorProto* data = ctx.getInputData(i);
      if (data != nullptr) {
        input_data[key.c_str()] = py::cast(*data);
      }
    }

    py::object schema = py::module_::import_("onnx.defs")
                            .attr("get_schema")(op_type, since_version, domain);
    py::object result_obj =
        py::module_::import_("onnx.shape_inference")
            .attr("infer_node_outputs")(schema, py::cast(node), input_types,
                                        input_data);
    py::dict result = py::cast<py::dict>(result_obj);

    for (size_t i = 0; i < num_outputs; ++i) {
      const std::string key = "out" + std::to_string(i);
      if (!result.contains(key.c_str())) {
        continue;
      }
      onnx::TypeProto* out_type = ctx.getOutputType(i);
      if (out_type != nullptr) {
        py::object value = result[key.c_str()];
        *out_type = py::cast<onnx::TypeProto>(value);
      }
    }
  } catch (...) {
    // Best-effort: leave the outputs uninferred on any failure so onnxsim's
    // shape inference simply flows past this operator, as it did before.
  }
}

}  // namespace

struct PyModelExecutor : public ModelExecutor {
  using ModelExecutor::ModelExecutor;

  // Adapts the DLPack executor boundary to the Python protocol, which still
  // exchanges tensors as serialized TensorProto bytes (the Python side --
  // onnxruntime's Python API, onnx's reference evaluator -- speaks TensorProto,
  // not DLPack). So this adapter pays a protobuf round trip that the C++/C-ABI
  // executors avoid; Python is not the zero-copy target. A future
  // dlpack-native Python executor could bypass it via __dlpack__.
  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
    std::vector<py::bytes> inputs_bytes;
    inputs_bytes.reserve(inputs.size());
    for (const DLManagedTensor* in : inputs) {
      const std::string str =
          onnxsim::dlpack::ToTensorProto(in->dl_tensor).SerializeAsString();
      inputs_bytes.emplace_back(str.data(), str.size());
    }
    std::string model_str = model.SerializeAsString();
    auto output_bytes =
        _PyRun(py::bytes(model_str.data(), model_str.size()), inputs_bytes);
    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(output_bytes.size());
    for (const py::bytes& x : output_bytes) {
      onnx::TensorProto tp;
      tp.ParseFromString(std::string(x.c_str(), x.size()));
      // Owning conversion: the parsed proto is a temporary, so the managed
      // tensor must keep it alive itself.
      outputs.emplace_back(
          onnxsim::dlpack::FromTensorProtoOwning(std::move(tp)));
    }
    return outputs;
  }

  virtual std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const = 0;
};

struct PyModelExecutorTrampoline : public PyModelExecutor {
  NB_TRAMPOLINE(PyModelExecutor, 1);

  /* Inherit the constructors */
  // using PyModelExecutor::PyModelExecutor;

  /* Trampoline (need one for each virtual function) */
  std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const override {
    NB_OVERRIDE_PURE_NAME(
        "Run", _PyRun, /* Name of function in C++ (must match Python name) */
        model_bytes, inputs_bytes /* Argument(s) */
    );
  }
};

// Bridges the C++ ``GraphRewriter`` interface to a Python implementation, in
// the same shape as ``PyModelExecutor``: the model is serialized to the
// protobuf wire format, handed to Python as ``bytes``, and the rewritten model
// is parsed back from the returned ``bytes``. This keeps onnxsim itself free of
// any dependency on the Python rewriting library (onnxscript etc.); the caller
// supplies the Python ``Run`` implementation.
struct PyGraphRewriter : public GraphRewriter {
  using GraphRewriter::GraphRewriter;

  bool _Run(onnx::ModelProto& model) const override {
    std::string model_str = model.SerializeAsString();
    auto output_bytes = _PyRun(py::bytes(model_str.data(), model_str.size()));
    // An empty ``bytes`` is the "model unchanged" sentinel: the Python rewriter
    // reported that it rewrote nothing, so leave ``model`` alone instead of
    // parsing an identical ModelProto back out of the returned bytes.
    if (output_bytes.size() == 0) {
      return false;
    }
    model.ParseFromString(
        std::string(output_bytes.c_str(), output_bytes.size()));
    return true;
  }

  virtual py::bytes _PyRun(const py::bytes& model_bytes) const = 0;
};

struct PyGraphRewriterTrampoline : public PyGraphRewriter {
  NB_TRAMPOLINE(PyGraphRewriter, 1);

  py::bytes _PyRun(const py::bytes& model_bytes) const override {
    NB_OVERRIDE_PURE_NAME(
        "Run", _PyRun, /* Name of function in C++ (must match Python name) */
        model_bytes    /* Argument(s) */
    );
  }
};

NB_MODULE(onnxsim_cpp2py_export, m) {
  m.doc() = "ONNX Simplifier";

  using namespace py::literals;

  // Compute the model metrics (op counts, size, MACs, memory access, peak
  // footprint) in C++ so the Python ``model_info`` can delegate the counting to
  // a single implementation. The symbolic metrics are returned as
  // coefficient/monomial polynomials -- each is a list of (coeff, [dim_name,
  // ...]) terms -- which the Python side rebuilds into sympy expressions,
  // keeping the public API (and its exact symbolic output) unchanged. Pass
  // ``run_shape_inference=False`` when the caller already inferred shapes (e.g.
  // the function-expanded graph, inferred with data propagation).
  m.def(
      "_model_metrics",
      [](const py::bytes& model_bytes, bool run_shape_inference) {
        onnx::ModelProto model;
        onnx::ParseProtoFromBytes(&model, model_bytes.c_str(),
                                  model_bytes.size());
        const ModelInfo info = GetModelInfo(model, run_shape_inference);
        auto to_poly = [](const onnxsim::SymExpr& expr) {
          std::vector<std::pair<int64_t, std::vector<std::string>>> poly;
          for (const auto& [monomial, coeff] : expr.terms())
            poly.emplace_back(coeff, monomial);
          return poly;
        };
        return std::make_tuple(info.op_nums, info.model_size,
                               to_poly(info.macs), to_poly(info.mem_access),
                               to_poly(info.memory_footprint));
      },
      "model_bytes"_a, "run_shape_inference"_a = true);

  // Dynamically quantizes MatMul/Gemm weights to INT8 (per output channel,
  // symmetric) and activations to uint8 at runtime via DynamicQuantizeLinear
  // -- see QuantizeDynamic in onnxsim.h. Pure graph rewrite: no ModelExecutor
  // or calibration data needed.
  m.def(
      "quantize_dynamic",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeDynamic(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Dynamically quantizes MatMul/Gemm nodes whose weight is structurally
  // ternary ({-s, 0, +s} per output column, e.g. BitNet b1.58) into the same
  // DynamicQuantizeLinear/MatMulInteger shape as quantize_dynamic, but with a
  // lossless ternary weight encoding instead of a rounded approximation --
  // see QuantizeTernary in onnxsim.h.
  m.def(
      "quantize_ternary",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeTernary(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Weight-only quantizes MatMul/Gemm/Conv weights to INT8 (per output
  // channel, symmetric) via a single DequantizeLinear -- activations are
  // never touched, so no calibration data or ModelExecutor is needed. See
  // QuantizeWeightOnly in onnxsim.h.
  m.def(
      "quantize_weight_only",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnly(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Block-wise INT4 weight-only quantizes MatMul/Gemm/Conv weights (one
  // symmetric scale per 32-element block of the reduction dimension, per
  // output channel) via a single DequantizeLinear(block_size=32) --
  // activations are never touched, so no calibration data or ModelExecutor
  // is needed. See QuantizeWeightOnlyInt4 in onnxsim.h.
  m.def(
      "quantize_weight_only_int4",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnlyInt4(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // INT16 weight-only quantizes MatMul/Gemm/Conv weights (one symmetric
  // scale per output channel, INT16's finer step than INT8's) -- activations
  // are never touched, so no calibration data or ModelExecutor is needed.
  // See QuantizeWeightOnlyInt16 in onnxsim.h.
  m.def(
      "quantize_weight_only_int16",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnlyInt16(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Block-wise INT8 weight-only quantizes MatMul/Gemm/Conv weights (one
  // symmetric scale per 32-element block of the flattened reduction
  // dimension, per output channel) via a single DequantizeLinear(axis=...,
  // block_size=32) -- activations are never touched, so no calibration data
  // or ModelExecutor is needed. See QuantizeWeightOnlyInt8Block in
  // onnxsim.h.
  m.def(
      "quantize_weight_only_int8_block",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnlyInt8Block(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Lists the activation tensor names quantize_static could quantize --
  // see ListQuantizableActivations in onnxsim.h.
  m.def(
      "list_quantizable_activations",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQuantizableActivations(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes MatMul/Gemm/Conv: weights to
  // INT8
  // (per output channel, symmetric, ahead of time) and activations to uint8
  // via a QuantizeLinear/DequantizeLinear pair with a *fixed* scale/zero-point
  // derived from `activation_ranges` (tensor name -> (min, max), typically
  // from list_quantizable_activations plus running the float model over
  // calibration data) -- see QuantizeStatic in onnxsim.h.
  m.def(
      "quantize_static",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeStatic(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Same as quantize_static, but a "W8A16" scheme: the weight stays INT8,
  // while the activation is quantized to uint16 instead of uint8 (an 8x
  // finer calibrated affine step) -- see QuantizeStaticInt16 in onnxsim.h.
  m.def(
      "quantize_static_int16",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeStaticInt16(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the *output* tensor names quantize_qoperator could additionally
  // quantize, on top of list_quantizable_activations' input names -- see
  // ListQOperatorQuantizableOutputs in onnxsim.h.
  m.def(
      "list_qoperator_quantizable_outputs",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorQuantizableOutputs(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes MatMul/Gemm into the
  // "QOperator" format (QLinearMatMul) rather than quantize_static's QDQ
  // format -- needs a calibrated range for both the activation and the
  // node's own output (see list_qoperator_quantizable_outputs) since
  // QLinearMatMul computes directly in int8, with no float intermediate --
  // see QuantizeQOperator in onnxsim.h.
  m.def(
      "quantize_qoperator",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeQOperator(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the tensor names quantize_qoperator_elementwise could quantize --
  // both operands and the output of every qualifying Add/Mul node -- see
  // ListQOperatorElementwiseQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_elementwise_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorElementwiseQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes elementwise Add/Mul into ONNX
  // Runtime's "com.microsoft" QLinearAdd/QLinearMul contrib ops -- needs a
  // calibrated range for both operands and the node's own output (see
  // list_qoperator_elementwise_quantizable_tensors) since these compute
  // directly in int8, with no float intermediate -- see
  // QuantizeQOperatorElementwise in onnxsim.h.
  m.def(
      "quantize_qoperator_elementwise",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            QuantizeQOperatorElementwise(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the tensor names quantize_qoperator_activation could quantize --
  // the input and output of every qualifying Sigmoid/LeakyRelu node -- see
  // ListQOperatorActivationQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_activation_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorActivationQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes standalone Sigmoid/LeakyRelu
  // into ONNX Runtime's "com.microsoft" QLinearSigmoid/QLinearLeakyRelu
  // contrib ops -- needs a calibrated range for both the input and the
  // node's own output (see list_qoperator_activation_quantizable_tensors)
  // since these compute directly in int8, with no float intermediate -- see
  // QuantizeQOperatorActivation in onnxsim.h.
  m.def(
      "quantize_qoperator_activation",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            QuantizeQOperatorActivation(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the tensor names quantize_qoperator_concat could quantize -- every
  // input plus the output of every qualifying Concat node -- see
  // ListQOperatorConcatQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_concat_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorConcatQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes Concat into ONNX Runtime's
  // "com.microsoft" QLinearConcat contrib op -- needs a calibrated range for
  // every input and the node's own output (see
  // list_qoperator_concat_quantizable_tensors) since this computes directly
  // in int8, with no float intermediate -- see QuantizeQOperatorConcat in
  // onnxsim.h.
  m.def(
      "quantize_qoperator_concat",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeQOperatorConcat(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Converts every float32 weight (and, by default, every internal
  // activation) to float16 -- no calibration data needed, since float16 is
  // still a floating-point format, not an integer scheme. With
  // keep_io_types (the default true), the graph's own external input/output
  // types stay float32 via boundary Cast nodes. See QuantizeFp16 in
  // onnxsim.h.
  m.def(
      "quantize_fp16",
      [](const py::bytes& model_proto_bytes, bool keep_io_types) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeFp16(model, keep_io_types);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "keep_io_types"_a = true);

  // Converts every float32 weight (and, by default, every internal
  // activation) to bfloat16 -- the same calibration-free, whole-graph
  // conversion as quantize_fp16 above, just to a different narrow
  // floating-point format (bfloat16 keeps float32's full exponent range, so
  // there is no clamping concern). See QuantizeBf16 in onnxsim.h.
  m.def(
      "quantize_bf16",
      [](const py::bytes& model_proto_bytes, bool keep_io_types) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeBf16(model, keep_io_types);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "keep_io_types"_a = true);

  // Converts every float32 weight (and, by default, every internal
  // activation) to an 8-bit floating-point format -- the same
  // calibration-free, whole-graph conversion as quantize_fp16/quantize_bf16
  // above, just to a much narrower floating-point format. `format` selects
  // "e4m3" (E4M3FN, the default) or "e5m2" (E5M2); both convert with
  // saturation (clamping) rather than producing an infinity/NaN for an
  // out-of-range magnitude. See QuantizeFp8 in onnxsim.h.
  m.def(
      "quantize_fp8",
      [](const py::bytes& model_proto_bytes, const std::string& format,
         bool keep_io_types) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeFp8(model, format, keep_io_types);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "format"_a = "e4m3", "keep_io_types"_a = true);

  m.def(
       "simplify",
       [](std::shared_ptr<PyModelExecutor> executor,
          const py::bytes& model_proto_bytes,
          std::optional<std::vector<std::string>> skip_optimizers,
          bool constant_folding, bool shape_inference,
          size_t tensor_size_threshold, std::optional<int> target_opset_version,
          std::shared_ptr<GraphRewriter> rewriter,
          bool initializers_as_constants, bool include_inline_functions,
          bool mutable_initializer,
          std::optional<std::unordered_map<std::string, std::vector<int64_t>>>
              overwrite_input_shapes,
          std::optional<std::vector<std::string>> unused_output) -> py::bytes {
         // force env initialization to register opset
         InitEnv();
         ONNX_NAMESPACE::ModelProto model;
         ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                             model_proto_bytes.size());
         auto const result = Simplify(
             *executor, model, skip_optimizers, constant_folding,
             shape_inference, tensor_size_threshold, target_opset_version,
             rewriter.get(), initializers_as_constants,
             include_inline_functions, mutable_initializer,
             overwrite_input_shapes, unused_output);
         std::string out;
         result.SerializeToString(&out);
         return py::bytes(out.data(), out.size());
       },
       "executor"_a, "model_bytes"_a, "skip_optimizers"_a.none(),
       "constant_folding"_a = true, "shape_inference"_a = true,
       "tensor_size_threshold"_a, "target_opset_version"_a.none(),
       "rewriter"_a.none(), "initializers_as_constants"_a = true,
       "include_inline_functions"_a = false, "mutable_initializer"_a = true,
       "overwrite_input_shapes"_a.none(), "unused_output"_a.none())
      .def(
          "simplify_path",
          [](std::shared_ptr<PyModelExecutor> executor,
             const std::string& in_path, const std::string& out_path,
             std::optional<std::vector<std::string>> skip_optimizers,
             bool constant_folding, bool shape_inference,
             size_t tensor_size_threshold,
             std::optional<int> target_opset_version,
             std::shared_ptr<GraphRewriter> rewriter,
             bool initializers_as_constants, bool include_inline_functions,
             bool mutable_initializer,
             std::optional<
                 std::unordered_map<std::string, std::vector<int64_t>>>
                 overwrite_input_shapes,
             std::optional<std::vector<std::string>> unused_output) -> bool {
            // force env initialization to register opset
            InitEnv();
            SimplifyPath(*executor, in_path, out_path, skip_optimizers,
                         constant_folding, shape_inference,
                         tensor_size_threshold, target_opset_version,
                         rewriter.get(), initializers_as_constants,
                         include_inline_functions, mutable_initializer,
                         overwrite_input_shapes, unused_output);
            return true;
          },
          "executor"_a, "in_path"_a, "out_path"_a, "skip_optimizers"_a.none(),
          "constant_folding"_a = true, "shape_inference"_a = true,
          "tensor_size_threshold"_a, "target_opset_version"_a.none(),
          "rewriter"_a.none(), "initializers_as_constants"_a = true,
          "include_inline_functions"_a = false, "mutable_initializer"_a = true,
          "overwrite_input_shapes"_a.none(), "unused_output"_a.none())
      .def("_list_optimizers",
           []() {
             py::list ret;
             for (const auto& p :
                  onnx::optimization::GetFuseAndEliminationPass()) {
               ret.append(p);
             }
             return ret;
           })
      // Whether onnxsim's internal (statically linked) schema registry already
      // knows an operator, at any opset version, in ``domain``. Used to skip
      // operators that do not need importing from the Python ``onnx`` module.
      .def(
          "_has_schema",
          [](const std::string& op_type, const std::string& domain) -> bool {
            return onnx::OpSchemaRegistry::Schema(
                       op_type, NormalizeDomain(domain)) != nullptr;
          },
          "op_type"_a, "domain"_a)
      // Register a single operator schema into onnxsim's internal schema
      // registry. onnxsim links its own copy of ONNX, so its registry is
      // separate from the one the Python ``onnx`` module uses; this bridges a
      // schema (e.g. one a user added via ``onnx.defs.register_schema``) across
      // that boundary so the model passes ``check_model`` (GitHub issue #326).
      //
      // When ``has_inference_function`` is set, the Python schema carries a
      // type/shape inference function; a C++ trampoline is attached that calls
      // it back through ``onnx.shape_inference.infer_node_outputs`` during
      // onnxsim's shape inference (see ``RunPythonNodeInference``). Otherwise
      // the schema is registered without one and shape inference simply flows
      // past the operator. Registration never raises: a malformed or duplicate
      // schema is reported to stderr and ignored, matching the other
      // schema-registration paths in onnxsim.
      .def(
          "_register_schema",
          [](const std::string& name, const std::string& domain,
             int since_version, const std::string& doc,
             const std::vector<PyFormalParameter>& inputs,
             const std::vector<PyFormalParameter>& outputs,
             const std::vector<PyAttribute>& attributes,
             const std::vector<PyTypeConstraint>& type_constraints,
             bool has_inference_function) {
            if (since_version < 1) {
              since_version = 1;
            }
            const std::string dom = NormalizeDomain(domain);
            EnsureDomainVersion(dom, since_version);

            OpSchema schema;
            schema.SetName(name)
                .SetDomain(dom)
                .SinceVersion(since_version)
                .SetDoc(doc);

            int idx = 0;
            for (const auto& p : inputs) {
              schema.Input(
                  idx++, std::get<0>(p), std::get<1>(p), std::get<2>(p),
                  static_cast<OpSchema::FormalParameterOption>(std::get<3>(p)),
                  std::get<4>(p), std::get<5>(p));
            }
            idx = 0;
            for (const auto& p : outputs) {
              schema.Output(
                  idx++, std::get<0>(p), std::get<1>(p), std::get<2>(p),
                  static_cast<OpSchema::FormalParameterOption>(std::get<3>(p)),
                  std::get<4>(p), std::get<5>(p));
            }
            for (const auto& a : attributes) {
              const onnx::AttributeProto& default_value = std::get<4>(a);
              if (default_value.type() != onnx::AttributeProto::UNDEFINED) {
                schema.Attr(OpSchema::Attribute(std::get<0>(a), std::get<1>(a),
                                                default_value));
              } else {
                schema.Attr(OpSchema::Attribute(
                    std::get<0>(a), std::get<1>(a),
                    static_cast<onnx::AttributeProto::AttributeType>(
                        std::get<2>(a)),
                    std::get<3>(a)));
              }
            }
            for (const auto& tc : type_constraints) {
              schema.TypeConstraint(std::get<0>(tc), std::get<1>(tc),
                                    std::get<2>(tc));
            }

            if (has_inference_function) {
              // Capture what the trampoline needs to reach back into the Python
              // ``onnx`` registry (which owns the real inference function) and
              // to read the node's attributes by name.
              std::vector<std::string> attr_names;
              attr_names.reserve(attributes.size());
              for (const auto& a : attributes) {
                attr_names.push_back(std::get<0>(a));
              }
              const int ver = since_version;
              schema.TypeAndShapeInferenceFunction(
                  [name, dom, ver, attr_names](onnx::InferenceContext& ctx) {
                    RunPythonNodeInference(ctx, name, dom, ver, attr_names);
                  });
            }

            onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/0,
                                 /*fail_duplicate_schema=*/false,
                                 /*fail_with_exception=*/false);
          },
          "name"_a, "domain"_a, "since_version"_a, "doc"_a, "inputs"_a,
          "outputs"_a, "attributes"_a, "type_constraints"_a,
          "has_inference_function"_a);

  py::class_<PyModelExecutor, PyModelExecutorTrampoline>(m, "ModelExecutor")
      .def(py::init<>())
      .def("Run", &PyModelExecutor::_PyRun);

  // The abstract C++ base shared by every rewriter kind. It carries no Python
  // constructor; ``simplify``/``simplify_path`` accept any subclass.
  py::class_<GraphRewriter>(m, "_GraphRewriterBase");

  // The Python-callable rewriter (an ``onnxscript.rewriter`` rule set, etc.).
  py::class_<PyGraphRewriter, GraphRewriter, PyGraphRewriterTrampoline>(
      m, "GraphRewriter")
      .def(py::init<>())
      .def("Run", &PyGraphRewriter::_PyRun);

  // The data-driven rewriter: a list of (pattern, replacement) FunctionProto
  // pairs. Being pure data, the same rules work from every binding, not just
  // Python. Returned as the base ``GraphRewriter`` -- the concrete type stays
  // private to the onnxsim core so this extension never references its vtable.
  m.def(
      "make_function_proto_rewriter",
      [](std::vector<std::pair<onnx::FunctionProto, onnx::FunctionProto>> rules)
          -> std::shared_ptr<GraphRewriter> {
        std::vector<onnxsim::FunctionRewriteRule> converted;
        converted.reserve(rules.size());
        for (auto& pair : rules) {
          converted.push_back(onnxsim::FunctionRewriteRule{
              std::move(pair.first), std::move(pair.second)});
        }
        return onnxsim::MakeFunctionProtoRewriter(std::move(converted));
      },
      "rules"_a);

  // Standalone safetensors/GGUF archive export/import: a model's graph and
  // weights packaged together in one ecosystem-standard file (see
  // onnxsim/tensor_pool_bridge.h and tensor_pool_gguf_bridge.h's *Standalone
  // functions for the real-offset design). Exchanged as bytes for the model
  // (like ``simplify``) and a real path for the archive itself, since the
  // archive is inherently file-based.
  m.def(
      "export_safetensors",
      [](const py::bytes& model_bytes, const std::string& out_path) {
        onnx::ModelProto model;
        ParseProtoFromBytes(&model, model_bytes.c_str(), model_bytes.size());
        onnxsim::tensor_pool::TensorPool pool;
        onnxsim::tensor_pool::SaveModelAsSafetensorsStandalone(model, out_path,
                                                               pool);
      },
      "model_bytes"_a, "out_path"_a);

  m.def(
      "import_safetensors",
      [](const std::string& in_path) -> py::bytes {
        onnx::ModelProto model;
        onnxsim::tensor_pool::TensorPool pool;
        if (!onnxsim::tensor_pool::LoadModelFromSafetensors(in_path, &model,
                                                            pool)) {
          throw std::runtime_error(
              "safetensors file has no embedded onnxsim model (a plain "
              "weights-only archive is not importable as a graph)");
        }
        const std::string out = model.SerializeAsString();
        return py::bytes(out.data(), out.size());
      },
      "in_path"_a);

  m.def(
      "export_gguf",
      [](const py::bytes& model_bytes, const std::string& out_path) {
        onnx::ModelProto model;
        ParseProtoFromBytes(&model, model_bytes.c_str(), model_bytes.size());
        onnxsim::tensor_pool::TensorPool pool;
        onnxsim::tensor_pool::SaveModelAsGGUFStandalone(model, out_path, pool);
      },
      "model_bytes"_a, "out_path"_a);

  m.def(
      "import_gguf",
      [](const std::string& in_path) -> py::bytes {
        onnx::ModelProto model;
        onnxsim::tensor_pool::TensorPool pool;
        if (!onnxsim::tensor_pool::LoadModelFromGGUF(in_path, &model, pool)) {
          throw std::runtime_error(
              "gguf file has no embedded onnxsim model (a plain weights-only "
              "archive is not importable as a graph)");
        }
        const std::string out = model.SerializeAsString();
        return py::bytes(out.data(), out.size());
      },
      "in_path"_a);

  // Hydrates `model`'s initializers, by name, from any GGUF file --
  // including a plain third-party weights-only checkpoint with no embedded
  // onnxsim model (unlike import_gguf, which requires one). A K-quant
  // tensor (Q4_K/Q5_K/Q6_K/Q8_0 -- what most real quantized checkpoints,
  // e.g. Unsloth's GGUF exports, actually use for the bulk of their
  // weights) is decoded to float32; see
  // ImportModelWithGGUF/HydrateTensorProtoFromGGUF in
  // tensor_pool_gguf_bridge.h. Returns (updated model bytes, names of GGUF
  // tensors present in the file but skipped because their ggml_type has no
  // representation TensorPool can hold at all, e.g. a legacy Q4_0 or IQ*-
  // family tensor -- NOT tensors simply absent from `model`'s
  // initializers, which this silently leaves alone rather than reporting).
  m.def(
      "import_gguf_weights",
      [](const py::bytes& model_bytes, const std::string& gguf_path)
          -> std::tuple<py::bytes, std::vector<std::string>> {
        onnx::ModelProto model;
        ParseProtoFromBytes(&model, model_bytes.c_str(), model_bytes.size());
        onnxsim::tensor_pool::TensorPool pool;
        std::vector<std::string> skipped;
        onnxsim::tensor_pool::ImportModelWithGGUF(model, gguf_path, pool,
                                                  /*hydrate_all=*/true,
                                                  &skipped);
        const std::string out = model.SerializeAsString();
        return {py::bytes(out.data(), out.size()), std::move(skipped)};
      },
      "model_bytes"_a, "gguf_path"_a);
}
