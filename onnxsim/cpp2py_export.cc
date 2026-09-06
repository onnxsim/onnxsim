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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "custom_optimizer_passes.h"
#include "dlpack_bridge.h"
#include "function_rewriter.h"
#include "memory_planning.h"
#include "model_info.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/optimize.h"
#include "onnxsim.h"
#include "precision_estimator.h"
#include "tensor_pool.h"
#include "tensor_pool_bridge.h"
#include "tensor_pool_gguf_bridge.h"
#include "xnnpack_codegen.h"

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
// A full operator schema as read back from onnxsim's internal registry:
// (name, domain, since_version, doc, inputs, outputs, attributes,
// type_constraints, has_type_and_shape_inference_function). Same shape
// ``_register_schema`` accepts, so it round-trips through the Python ``onnx``
// module's own ``OpSchema``/``register_schema`` in the opposite direction.
using PySchema =
    std::tuple<std::string, std::string, int, std::string,
               std::vector<PyFormalParameter>, std::vector<PyFormalParameter>,
               std::vector<PyAttribute>, std::vector<PyTypeConstraint>, bool>;

// Recursively marshal a MemoryPlan (see memory_planning.h) into the same
// (offsets, arena_bytes, naive_bytes, unplanned, subgraph_reserved_bytes,
// subgraph_plans) tuple shape at every level, so the Python
// ``memory_planning`` module can rebuild nested ``MemoryPlan`` dataclasses
// without any special-casing for depth. ``subgraph_plans`` -- a
// ``py::dict`` rather than a plain ``std::map`` -- is itself a Python dict
// mapping each subgraph's key to its own recursively-marshalled tuple.
py::object MemoryPlanToPyTuple(const onnxsim::MemoryPlan& plan) {
  py::dict subgraphs;
  for (const auto& [key, sub] : plan.subgraph_plans) {
    subgraphs[key.c_str()] = MemoryPlanToPyTuple(sub);
  }
  return py::cast(std::make_tuple(plan.offsets, plan.arena_bytes,
                                  plan.naive_bytes, plan.unplanned,
                                  plan.subgraph_reserved_bytes, subgraphs));
}

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

  // The maximum default-domain ("" / "ai.onnx") opset version this build's
  // compiled-in onnx schema registry knows about -- i.e. what
  // target_opset_version="latest" resolves to. Exposed so the Python side can
  // resolve "latest" against the same registry ConvertOpsetVersion itself
  // uses, rather than guessing from the (possibly differently-versioned)
  // pip-installed `onnx` package.
  m.def("max_default_domain_opset_version", []() {
    return onnx::OpSchemaRegistry::DomainToVersionRange::Instance()
        .Map()
        .at("")
        .second;
  });

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

  // Compute a static activation-memory plan (see memory_planning.h): a byte
  // offset for every tensor whose size is concretely known, packed into one
  // shared arena by reusing space from tensors whose liveness has ended,
  // plus one independently-computed nested plan per control-flow (If/Loop/
  // Scan) subgraph body. Returned via MemoryPlanToPyTuple's recursive
  // (offsets, arena_bytes, naive_bytes, unplanned, subgraph_reserved_bytes,
  // subgraph_plans) tuple shape so the Python ``memory_planning`` module can
  // rebuild a nested ``MemoryPlan`` dataclass tree, mirroring how
  // ``_model_metrics`` hands its polynomials back to ``model_info`` for the
  // sympy rebuild.
  m.def(
      "_memory_plan",
      [](const py::bytes& model_bytes, bool run_shape_inference) {
        onnx::ModelProto model;
        onnx::ParseProtoFromBytes(&model, model_bytes.c_str(),
                                  model_bytes.size());
        const onnxsim::GraphView view =
            GetGraphView(model, run_shape_inference);
        const onnxsim::MemoryPlan plan =
            onnxsim::ComputeActivationMemoryPlan(view);
        return MemoryPlanToPyTuple(plan);
      },
      "model_bytes"_a, "run_shape_inference"_a = true);

  // Emit a standalone C source file reconstructing `model` as an XNNPACK
  // Subgraph (see xnnpack_codegen.h for scope/layout convention). Unlike
  // _memory_plan, this needs no shape-inference flag: GenerateXnnpackC always
  // runs it internally, since generated code must bake in concrete shapes.
  m.def(
      "_generate_xnnpack_c",
      [](const py::bytes& model_bytes, const std::string& function_prefix) {
        onnx::ModelProto model;
        onnx::ParseProtoFromBytes(&model, model_bytes.c_str(),
                                  model_bytes.size());
        return onnxsim::xnnpack_backend::GenerateXnnpackC(model,
                                                          function_prefix);
      },
      "model_bytes"_a, "function_prefix"_a);

  // Data-free Cross-Layer Equalization preprocessing (not itself a
  // quantization scheme) -- see CrossLayerEqualize in onnxsim.h. Pure graph
  // rewrite: no ModelExecutor or calibration data needed.
  m.def(
      "cross_layer_equalize",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = CrossLayerEqualize(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

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

  // Same rewrite as quantize_dynamic, but the dequantize step is a single
  // ONNX Runtime "com.microsoft" contrib op (MatMulIntegerToFloat) instead
  // of quantize_dynamic's separate MatMulInteger+Cast+Mul(+Add) node chain
  // -- see QuantizeDynamicMatMulIntegerToFloat in onnxsim.h. Pure graph
  // rewrite: no ModelExecutor or calibration data needed.
  m.def(
      "quantize_dynamic_matmul_integer_to_float",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeDynamicMatMulIntegerToFloat(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Dynamically quantizes an existing "com.microsoft" Attention node (see
  // fuse_attention.h -- this does not fuse attention itself) into its
  // quantized counterpart, QAttention -- see QuantizeAttentionDynamic in
  // onnxsim.h. Pure graph rewrite: no ModelExecutor or calibration data
  // needed.
  m.def(
      "quantize_attention_dynamic",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeAttentionDynamic(model);
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

  // Block-wise INT4 weight-only quantizes MatMul/Gemm weights into ONNX
  // Runtime's own com.microsoft::MatMulNBits contrib op -- a vendor-specific
  // (ORT-only) counterpart to quantize_weight_only_int4's portable standard-
  // ONNX output. See QuantizeWeightOnlyMatMulNBits in onnxsim.h.
  m.def(
      "quantize_weight_only_matmul_nbits",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnlyMatMulNBits(model);
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

  // OCP Microscaling MXFP4 weight-only quantizes every MatMul/vanilla-Gemm
  // whose weight is a constant float32 tensor whose reduction dimension is
  // divisible by 32, via a Gather-a-codebook-then-scale dequant chain (no
  // native ONNX MX tensor type). Activations are never touched, so no
  // calibration data is needed. See QuantizeWeightOnlyMXFP4 in onnxsim.h.
  m.def(
      "quantize_weight_only_mxfp4",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnlyMXFP4(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // QLoRA-style double quantization: quantizes every already-present
  // DequantizeLinear node's own (large enough) constant scale tensor to
  // UINT8 with a per-tensor meta-scale. See ApplyDoubleQuantization in
  // onnxsim.h.
  m.def(
      "apply_double_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyDoubleQuantization(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // Magnitude pruning (Han et al., 2015): zeros the least-magnitude entries
  // of every MatMul/vanilla-Gemm/Conv/com.microsoft::Attention layer's
  // constant weight, independently per output row/filter. Data-free.
  // `n`/`m` are `None` (unstructured, ranked by `sparsity`) or both given
  // together (N:M semi-structured). `global_sparsity` pools every matched
  // layer's importance into one whole-model ranking; incompatible with
  // `n`/`m`. Same `n`/`m`/`global_sparsity` shape as apply_wanda_pruning's
  // own binding above. See PruneMagnitude in onnxsim.h.
  m.def(
      "prune_magnitude",
      [](const py::bytes& model_proto_bytes, double sparsity,
         std::optional<int64_t> n, std::optional<int64_t> m,
         bool global_sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            PruneMagnitude(model, sparsity, n, m, global_sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "sparsity"_a, "n"_a.none(), "m"_a.none(),
      "global_sparsity"_a = false);

  // Structured (channel) pruning: removes whole output channels from
  // MatMul/vanilla-Gemm and Conv layers -- real structural pruning, not
  // just value-only zeroing. See ApplyStructuredPruning in
  // structured_pruning_entry.h. `importance_norm` ("l1"/"l2") and
  // `global_sparsity` mirror pruning.py's own `apply_structured_pruning`
  // parameters of the same names exactly -- see that function's own
  // declaration comment.
  m.def(
      "apply_structured_pruning",
      [](const py::bytes& model_proto_bytes, double sparsity,
         const std::string& importance_norm,
         bool global_sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyStructuredPruning(
            model, sparsity, importance_norm, global_sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "sparsity"_a, "importance_norm"_a = "l2",
      "global_sparsity"_a = false);

  // The calibration-driven (Wanda-style) upgrade of apply_structured_pruning
  // above -- same executor-as-first-argument shape as `simplify`'s own
  // binding, since this is the first calibration-driven (not purely
  // data-free) structured-pruning entry point: the executor is what
  // actually runs `model_bytes` over `calibration_data` to capture
  // per-channel activation norms. `calibration_data` is
  // `List[Dict[str, onnx.TensorProto]]` -- one {graph input name: tensor}
  // map per calibration batch, crossing via the very same
  // `onnx::TensorProto` nanobind caster (ONNXSIM_PROTO_CASTER above) every
  // other proto crosses this boundary with -- see
  // ApplyStructuredWandaPruning/WandaCalibrationStats in
  // structured_pruning_entry.cpp for the full calibration-crossing design
  // (including exactly where the name -> ModelExecutor::Run-positional
  // reordering happens). See ApplyStructuredWandaPruning in
  // structured_pruning_entry.h.
  m.def(
      "apply_structured_wanda_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity, double epsilon, const std::string& importance_norm,
         bool global_sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyStructuredWandaPruning(
            model, *executor, calibration_data, sparsity, epsilon,
            importance_norm, global_sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a,
      "epsilon"_a = 1e-8, "importance_norm"_a = "l2",
      "global_sparsity"_a = false);

  // Attention-head pruning: removes whole attention heads (or, for
  // grouped-query attention, whole KV groups) from every matched fused
  // self-attention block. See ApplyAttentionHeadPruning in onnxsim.h.
  m.def(
      "apply_attention_head_pruning",
      [](const py::bytes& model_proto_bytes, double sparsity,
         const std::string& importance_norm) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyAttentionHeadPruning(model, sparsity, importance_norm);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "sparsity"_a, "importance_norm"_a = "l2");

  // The calibration-driven (Wanda-style) upgrade of
  // apply_attention_head_pruning above -- same executor-as-first-argument
  // shape, and the same `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) crossing convention, as apply_structured_wanda_
  // pruning's own binding above (see that binding's own comment for the
  // full calibration-crossing design). See ApplyAttentionHeadWandaPruning
  // in structured_pruning_entry.h.
  m.def(
      "apply_attention_head_wanda_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity, double epsilon,
         const std::string& importance_norm) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyAttentionHeadWandaPruning(model, *executor, calibration_data,
                                           sparsity, epsilon, importance_norm);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a,
      "epsilon"_a = 1e-8, "importance_norm"_a = "l2");

  // SparseGPT (Frantar & Alistarh, 2023) unstructured/N:M pruning: zeros
  // the least-important entries of every matched MatMul/vanilla-Gemm/
  // com.microsoft::Attention merged-QKV-weight/2-D Conv (ordinary/
  // depthwise/general-grouped) layer's constant FLOAT32/FLOAT16/BFLOAT16
  // weight, using a sequential, Hessian-error-compensating algorithm
  // (GPTQ's own Cholesky-factored inverse Hessian reformulation) rather
  // than a one-shot static importance score -- unlike every pass above,
  // this never changes any tensor's shape, only individual weight entries'
  // own values. Same executor-as-first-argument, `calibration_data`
  // (List[Dict[str, onnx.TensorProto]]) crossing convention as
  // apply_structured_wanda_pruning's own binding above (see that binding's
  // own comment for the full calibration-crossing design). `n`/`m` are
  // `None` (unstructured, ranked by `sparsity`) or both given together
  // (N:M semi-structured). See ApplySparseGptPruning in
  // structured_pruning_entry.h for the full scope, now at full parity with
  // pruning.py's own `apply_sparsegpt_pruning` (itself now a thin alias for
  // this port), Conv included.
  m.def(
      "apply_sparsegpt_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity, std::optional<int64_t> n, std::optional<int64_t> m,
         double percdamp, int64_t proc_block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplySparseGptPruning(model, *executor, calibration_data, sparsity,
                                  n, m, percdamp, proc_block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a,
      "n"_a.none(), "m"_a.none(), "percdamp"_a = 0.01,
      "proc_block_size"_a = 128);

  // Wanda pruning (Sun et al., 2023): the calibration-driven upgrade of
  // magnitude pruning's data-free baseline, zeroing the least-important
  // entries of every matched layer's constant 2-D FLOAT32/FLOAT16/BFLOAT16
  // weight to an unstructured or N:M sparsity pattern using
  // ``|W_ij| * ||X_j||_2`` (weight magnitude times its reduction-dimension
  // entry's calibrated activation L2-norm) as the importance metric -- a
  // one-shot static score, unlike apply_sparsegpt_pruning's own sequential
  // Hessian-error-compensating algorithm. Same candidate set as
  // apply_sparsegpt_pruning's own binding above except widened to
  // FLOAT32/FLOAT16/BFLOAT16, PLUS every 2-D Conv node's constant 4-D
  // weight (ordinary/depthwise/general-grouped alike) -- TRUE parity with
  // pruning.py's own apply_wanda_pruning on every one of those candidate
  // families (deliberately NOT aliased to that function despite the Conv
  // parity, though -- a separate, pre-existing MatMul-family calibration
  // rank-handling gap unrelated to Conv blocks it; see ApplyWandaPruning's
  // own declaration comment in structured_pruning_entry.h for the full
  // writeup). Same executor-as-first-argument, `calibration_data`
  // (List[Dict[str, onnx.TensorProto]]) crossing convention. See
  // ApplyWandaPruning in structured_pruning_entry.h for the full scope and
  // the data-free magnitude fallback an unobserved layer gets (unlike
  // SparseGPT, which has none).
  // `global_sparsity` pools every matched layer's importance into one
  // whole-model ranking, mirroring apply_structured_wanda_pruning's own
  // `sparsity`-only mode's structural analogue; incompatible with `n`/`m`.
  m.def(
      "apply_wanda_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity, std::optional<int64_t> n, std::optional<int64_t> m,
         double epsilon, bool global_sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyWandaPruning(model, *executor, calibration_data, sparsity, n,
                              m, epsilon, global_sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a,
      "n"_a.none(), "m"_a.none(), "epsilon"_a = 1e-8,
      "global_sparsity"_a = false);

  // MoE expert-intermediate-channel pruning: removes intermediate
  // (`inter_size`) channels from every expert of a matched
  // `com.microsoft::MoE` node at once -- real structural pruning, data-free.
  // Whole-expert pruning (shrinking `num_experts` itself) is NOT ported --
  // see ApplyMoeExpertChannelPruning in structured_pruning_entry.h.
  m.def(
      "apply_moe_expert_channel_pruning",
      [](const py::bytes& model_proto_bytes, double sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyMoeExpertChannelPruning(model, sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "sparsity"_a);

  // QMoE expert-channel pruning: removes intermediate (inter_size) channels
  // from every expert of a matched com.microsoft::QMoE node -- the
  // quantized-weight counterpart of apply_structured_pruning. See
  // ApplyQMoEExpertChannelPruning in onnxsim.h.
  m.def(
      "apply_qmoe_expert_channel_pruning",
      [](const py::bytes& model_proto_bytes, double sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyQMoEExpertChannelPruning(model, sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "sparsity"_a);

  // MoE whole-expert pruning: the calibration-driven complementary
  // technique to apply_moe_expert_channel_pruning above -- drops whole
  // experts (shrinks `num_experts`) ranked by mean router gate weight over
  // `calibration_data`, same executor-as-first-argument shape as
  // apply_structured_wanda_pruning (this is likewise NOT purely
  // data-free -- it runs `model_bytes` over `calibration_data` to capture
  // router activations). See ApplyMoeWholeExpertPruning in
  // structured_pruning_entry.h.
  m.def(
      "apply_moe_whole_expert_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyMoeWholeExpertPruning(
            model, *executor, calibration_data, sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a);

  // QMoE whole-expert pruning: the quantized-weight counterpart of
  // apply_moe_whole_expert_pruning above -- same calibration-driven
  // ranking (mean router gate weight, the exact same
  // MoeRouterGateCalibrationStats helper -- `router_probs` is QMoE's own
  // second input too, upstream of and oblivious to its quantized
  // fc1/fc2), same executor-as-first-argument shape. See
  // ApplyQMoEWholeExpertPruning in structured_pruning_entry.h.
  m.def(
      "apply_qmoe_whole_expert_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyQMoEWholeExpertPruning(
            model, *executor, calibration_data, sparsity);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a);

  // Transformer block (depth) pruning: drops whole redundant pre-norm
  // transformer residual sub-blocks wholesale -- a GENUINELY DIFFERENT
  // KIND of pass from every calibration-driven binding above (graph
  // surgery: nodes deleted and consumers rewired, not tensors resized in
  // place). Same executor-as-first-argument shape and `calibration_data`
  // (List[Dict[str, onnx.TensorProto]]) crossing convention as
  // apply_structured_wanda_pruning/apply_moe_whole_expert_pruning above.
  // `num_blocks_to_drop` (Optional[int], via nanobind's std::optional
  // caster -- same crossing already used for e.g. GQA head/kv-head
  // overrides elsewhere in this file) takes priority over `sparsity` when
  // given, mirroring pruning.py's own `apply_transformer_block_pruning`
  // keyword-argument precedence exactly. See ApplyTransformerBlockPruning
  // in structured_pruning_entry.h.
  m.def(
      "apply_transformer_block_pruning",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double sparsity,
         std::optional<int64_t> num_blocks_to_drop) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyTransformerBlockPruning(
            model, *executor, calibration_data, sparsity, num_blocks_to_drop);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "sparsity"_a,
      "num_blocks_to_drop"_a = std::nullopt);

  // Embedding vocabulary pruning: shrinks a matched token-embedding
  // table's vocabulary axis (plus, where a tied/untied lm_head exists, its
  // own vocab-logits projection too) down to a caller-supplied explicit
  // keep/drop set. Unlike every `apply_*` binding above, the pruned model
  // this returns does not accept the original model's own token ids -- see
  // EmbeddingVocabPruningResult/ApplyEmbeddingVocabPruning in
  // structured_pruning_entry.h. Returned as a
  // (model_bytes, matched, kept_token_ids, lm_head_pruned) tuple rather
  // than bare bytes -- reconstructed into the real, public
  // `onnxsim.pruning.EmbeddingPruningResult` dataclass by the Python
  // wrapper (onnx_simplifier.py's own
  // apply_embedding_vocab_pruning_cpp), which also derives `id_map` from
  // `kept_token_ids` (trivial: `{tok: i for i, tok in
  // enumerate(kept_token_ids)}`) rather than this needing to also cross
  // the nanobind boundary as a separate map.
  m.def(
      "apply_embedding_vocab_pruning",
      [](const py::bytes& model_proto_bytes,
         std::optional<std::vector<int64_t>> keep_token_ids,
         std::optional<std::vector<int64_t>> drop_token_ids,
         std::optional<std::string> input_name)
          -> std::tuple<py::bytes, bool, std::vector<int64_t>, bool> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyEmbeddingVocabPruning(
            model, keep_token_ids, drop_token_ids, input_name);
        std::string out;
        result.model.SerializeToString(&out);
        return {py::bytes(out.data(), out.size()), result.matched,
                result.kept_token_ids, result.lm_head_pruned};
      },
      "model_bytes"_a, "keep_token_ids"_a.none(), "drop_token_ids"_a.none(),
      "input_name"_a.none());

  // The importance-ranked variant -- see EmbeddingVocabPruningResult/
  // ApplyEmbeddingVocabMagnitudePruning in structured_pruning_entry.h.
  // Same return shape as apply_embedding_vocab_pruning above.
  m.def(
      "apply_embedding_vocab_magnitude_pruning",
      [](const py::bytes& model_proto_bytes, double sparsity,
         std::optional<std::vector<int64_t>> protect_token_ids,
         std::optional<std::string> input_name)
          -> std::tuple<py::bytes, bool, std::vector<int64_t>, bool> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyEmbeddingVocabMagnitudePruning(
            model, sparsity, protect_token_ids, input_name);
        std::string out;
        result.model.SerializeToString(&out);
        return {py::bytes(out.data(), out.size()), result.matched,
                result.kept_token_ids, result.lm_head_pruned};
      },
      "model_bytes"_a, "sparsity"_a = 0.5, "protect_token_ids"_a.none(),
      "input_name"_a.none());

  // Any-Precision LLM (Park et al., 2024, ICML 2024): nested bit-plane
  // weight-only quantization, one quantization pass serving any bit-width
  // up to max_bits. See ApplyAnyPrecisionLlm in onnxsim.h.
  m.def(
      "apply_any_precision_llm",
      [](const py::bytes& model_proto_bytes, int64_t bits, int64_t max_bits,
         int64_t block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyAnyPrecisionLlm(model, bits, max_bits, block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "bits"_a = 4, "max_bits"_a = 8, "block_size"_a = 32);

  // QuaRot (Ashkboos et al., 2024): rotation preprocessing plus INT4
  // round-to-nearest quantization of both the weight and the activation of
  // every MatMul/vanilla-Gemm layer. Data-free. See ApplyQuarot in
  // onnxsim.h.
  m.def(
      "apply_quarot",
      [](const py::bytes& model_proto_bytes, uint64_t seed, int64_t block_size,
         float epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyQuarot(model, seed, block_size, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "seed"_a, "block_size"_a = 32, "epsilon"_a = 1e-12f);

  // llama.cpp's IQ4_NL: fixed 16-entry non-uniform-codebook weight-only 4-bit
  // quantization, one scale per 32-element block. Data-free. See ApplyIQ4NL
  // in onnxsim.h.
  m.def(
      "apply_iq4_nl",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyIQ4NL(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // llama.cpp's legacy GGUF Q4_0/Q4_1 block formats: weight-only 4-bit
  // quantization, one plain 32-element block per scale(/min). Data-free.
  // See ApplyGgufQ4_0/ApplyGgufQ4_1 in onnxsim.h.
  m.def(
      "apply_gguf_q4_0_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ4_0(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
  m.def(
      "apply_gguf_q4_1_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ4_1(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // BitNet b1.58's published absmean ternary weight quantization, as
  // shipped by llama.cpp's GGUF TQ1_0/TQ2_0 tensor types: weight-only,
  // one shared {-1, 0, +1} scale per 256-element block. Data-free. See
  // ApplyGgufTernaryQuant in onnxsim.h.
  m.def(
      "apply_gguf_ternary_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufTernaryQuant(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);

  // FP6-LLM's E3M2 6-bit floating-point weight-only quantization, one
  // scale per 64-element block. Data-free. See ApplyFp6Llm in onnxsim.h.
  m.def(
      "apply_fp6_llm",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyFp6Llm(model);
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

  // Lists the tensor names quantize_qoperator_softmax could quantize -- the
  // input and output of every qualifying Softmax node -- see
  // ListQOperatorSoftmaxQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_softmax_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorSoftmaxQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes standalone Softmax into ONNX
  // Runtime's "com.microsoft" QLinearSoftmax contrib op -- needs a
  // calibrated range for both the input and the node's own output (see
  // list_qoperator_softmax_quantizable_tensors) since this computes
  // directly in int8, with no float intermediate -- see
  // QuantizeQOperatorSoftmax in onnxsim.h.
  m.def(
      "quantize_qoperator_softmax",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeQOperatorSoftmax(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the tensor names quantize_qoperator_pool could quantize -- the
  // input and output of every qualifying AveragePool/GlobalAveragePool node
  // -- see ListQOperatorPoolQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_pool_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorPoolQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes standalone AveragePool/
  // GlobalAveragePool into ONNX Runtime's "com.microsoft"
  // QLinearAveragePool/QLinearGlobalAveragePool contrib ops -- needs a
  // calibrated range for both the input and the node's own output (see
  // list_qoperator_pool_quantizable_tensors) since these compute directly
  // in int8, with no float intermediate -- see QuantizeQOperatorPool in
  // onnxsim.h.
  m.def(
      "quantize_qoperator_pool",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeQOperatorPool(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the tensor names quantize_qoperator_where could quantize -- both
  // operands and the output of every qualifying Where node -- see
  // ListQOperatorWhereQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_where_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorWhereQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes Where into ONNX Runtime's
  // "com.microsoft" QLinearWhere contrib op -- needs a calibrated range for
  // both operands and the node's own output (see
  // list_qoperator_where_quantizable_tensors) since this computes directly
  // in int8, with no float intermediate -- see QuantizeQOperatorWhere in
  // onnxsim.h.
  m.def(
      "quantize_qoperator_where",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeQOperatorWhere(model, activation_ranges);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "activation_ranges"_a);

  // Lists the tensor names quantize_qoperator_gemm could quantize -- the
  // activation and output of every qualifying Gemm node -- see
  // ListQOperatorGemmQuantizableTensors in onnxsim.h.
  m.def(
      "list_qoperator_gemm_quantizable_tensors",
      [](const py::bytes& model_proto_bytes) -> std::vector<std::string> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        return ListQOperatorGemmQuantizableTensors(model);
      },
      "model_bytes"_a);

  // Statically (calibration-based) quantizes Gemm into ONNX Runtime's
  // "com.microsoft" QGemm contrib op -- the fully-general analogue of
  // quantize_qoperator's QLinearMatMul rewrite (handles any transA/transB/
  // alpha) -- needs a calibrated range for the activation and the node's
  // own output (see list_qoperator_gemm_quantizable_tensors) since this
  // computes directly in int8, with no float intermediate -- see
  // QuantizeQOperatorGemm in onnxsim.h.
  m.def(
      "quantize_qoperator_gemm",
      [](const py::bytes& model_proto_bytes,
         const std::unordered_map<std::string, std::pair<float, float>>&
             activation_ranges) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeQOperatorGemm(model, activation_ranges);
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

  // Static, calibration-free INT8-quantization risk analysis -- the single
  // C++ implementation (onnxsim/precision_estimator.{h,cpp}) that both this
  // Python binding and the WASM UI (scripts/convertmodel/interface.cpp) call
  // into, so the algorithm exists in exactly one place rather than two
  // (Python used to carry its own parallel implementation). The Python-facing
  // ``onnxsim.precision_estimator`` module is a thin wrapper that reconstructs
  // its public dataclasses from the tuples returned here; see that module's
  // docstring. Each weight-estimate tuple is (node_name, op_type,
  // reduction_depth, num_channels, int32_accumulator_safe, float32_cast_exact,
  // max_outlier_ratio, outlier_risk, activation_producer_op,
  // activation_range_lo, activation_range_hi, recommendation) -- the last
  // three fields are None together (no known range) or all present. Each
  // attention-estimate tuple is (node_name, num_query_heads, num_kv_heads,
  // head_dim, default_scale, actual_scale, scale_matches_default,
  // recommendation).
  m.def(
      "_estimate_model_quantization_drop",
      [](const py::bytes& model_proto_bytes) {
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const onnxsim::ModelQuantizationEstimate est =
            onnxsim::EstimateModelQuantizationDrop(model);

        using WeightTuple =
            std::tuple<std::string, std::string, int64_t, int64_t, bool, bool,
                       double, bool, std::optional<std::string>,
                       std::optional<double>, std::optional<double>,
                       std::string>;
        std::vector<WeightTuple> weight_estimates;
        weight_estimates.reserve(est.weight_estimates.size());
        for (const auto& w : est.weight_estimates) {
          weight_estimates.emplace_back(
              w.node_name, w.op_type, w.reduction_depth, w.num_channels,
              w.int32_accumulator_safe, w.float32_cast_exact,
              w.max_outlier_ratio, w.outlier_risk,
              w.activation_producer_op.empty()
                  ? std::nullopt
                  : std::optional<std::string>(w.activation_producer_op),
              w.has_activation_range
                  ? std::optional<double>(w.activation_range_lo)
                  : std::nullopt,
              w.has_activation_range
                  ? std::optional<double>(w.activation_range_hi)
                  : std::nullopt,
              w.recommendation);
        }

        using AttentionTuple =
            std::tuple<std::string, std::optional<int64_t>,
                       std::optional<int64_t>, std::optional<int64_t>,
                       std::optional<double>, std::optional<double>,
                       std::optional<bool>, std::string>;
        std::vector<AttentionTuple> attention_estimates;
        attention_estimates.reserve(est.attention_estimates.size());
        for (const auto& a : est.attention_estimates) {
          attention_estimates.emplace_back(
              a.node_name,
              a.has_num_query_heads ? std::optional<int64_t>(a.num_query_heads)
                                    : std::nullopt,
              a.has_num_kv_heads ? std::optional<int64_t>(a.num_kv_heads)
                                 : std::nullopt,
              a.has_head_dim ? std::optional<int64_t>(a.head_dim)
                             : std::nullopt,
              std::isnan(a.default_scale)
                  ? std::nullopt
                  : std::optional<double>(a.default_scale),
              std::isnan(a.actual_scale)
                  ? std::nullopt
                  : std::optional<double>(a.actual_scale),
              a.scale_matches_default < 0
                  ? std::nullopt
                  : std::optional<bool>(a.scale_matches_default != 0),
              a.recommendation);
        }

        return std::make_tuple(est.total_nodes_analyzed, est.unsafe_nodes,
                               est.outlier_risk_nodes, est.worst_outlier_ratio,
                               est.estimated_relative_error, est.risk_level,
                               weight_estimates, attention_estimates);
      },
      "model_bytes"_a);

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
          std::optional<std::vector<std::string>> unused_output,
          std::optional<std::vector<std::string>> extra_optimizers)
           -> py::bytes {
         // force env initialization to register opset
         InitEnv();
         ONNX_NAMESPACE::ModelProto model;
         ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                             model_proto_bytes.size());
         // ``model`` is this lambda's own local, parsed fresh from
         // ``model_proto_bytes`` and never read again after this call --
         // exactly the case SimplifyConsumeInput's doc comment calls out as
         // safe, and it does not touch the caller's own Python object (which
         // was only ever serialized *from*, not aliased). See
         // bench/RESULTS_synthetic_decoder_oom.md for why this matters.
         auto const result = SimplifyConsumeInput(
             *executor, model, skip_optimizers, constant_folding,
             shape_inference, tensor_size_threshold, target_opset_version,
             rewriter.get(), initializers_as_constants,
             include_inline_functions, mutable_initializer,
             overwrite_input_shapes, unused_output, extra_optimizers);
         std::string out;
         result.SerializeToString(&out);
         return py::bytes(out.data(), out.size());
       },
       "executor"_a, "model_bytes"_a, "skip_optimizers"_a.none(),
       "constant_folding"_a = true, "shape_inference"_a = true,
       "tensor_size_threshold"_a, "target_opset_version"_a.none(),
       "rewriter"_a.none(), "initializers_as_constants"_a = true,
       "include_inline_functions"_a = false, "mutable_initializer"_a = true,
       "overwrite_input_shapes"_a.none(), "unused_output"_a.none(),
       "extra_optimizers"_a.none())
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
             std::optional<std::vector<std::string>> unused_output,
             std::optional<std::vector<std::string>> extra_optimizers) -> bool {
            // force env initialization to register opset
            InitEnv();
            SimplifyPath(
                *executor, in_path, out_path, skip_optimizers, constant_folding,
                shape_inference, tensor_size_threshold, target_opset_version,
                rewriter.get(), initializers_as_constants,
                include_inline_functions, mutable_initializer,
                overwrite_input_shapes, unused_output, extra_optimizers);
            return true;
          },
          "executor"_a, "in_path"_a, "out_path"_a, "skip_optimizers"_a.none(),
          "constant_folding"_a = true, "shape_inference"_a = true,
          "tensor_size_threshold"_a, "target_opset_version"_a.none(),
          "rewriter"_a.none(), "initializers_as_constants"_a = true,
          "include_inline_functions"_a = false, "mutable_initializer"_a = true,
          "overwrite_input_shapes"_a.none(), "unused_output"_a.none(),
          "extra_optimizers"_a.none())
      .def("_list_optimizers",
           []() {
             py::list ret;
             for (const auto& p :
                  onnx::optimization::GetFuseAndEliminationPass()) {
               ret.append(p);
             }
             return ret;
           })
      // The counterpart to _list_optimizers: pass names valid for
      // extra_optimizers specifically -- registered but not already part of
      // the default fuse/elimination set (typically PassType::Other, e.g.
      // fuse_matmul_add_bias_into_gemm_batched). Registers onnxsim's own
      // custom passes first so this is accurate even if called before any
      // simplify()/simplify_path() call has done so.
      .def("_list_other_optimizers",
           []() {
             onnxsim::RegisterCustomOptimizerPasses();
             const auto default_passes =
                 onnx::optimization::GetFuseAndEliminationPass();
             const std::unordered_set<std::string> default_set(
                 default_passes.begin(), default_passes.end());
             py::list ret;
             for (const auto& p : onnx::optimization::GetAvailablePasses()) {
               if (!default_set.count(p)) {
                 ret.append(p);
               }
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
          "has_inference_function"_a)
      // The counterpart to ``_register_schema``: read back every operator
      // schema onnxsim's internal (statically linked) registry knows about --
      // its built-in ONNX Runtime contrib-op schemas (see
      // ``contrib_schemas.cpp``) plus anything a caller previously imported
      // via ``_register_schema`` -- in the same tuple shape that function
      // accepts. This lets Python code (e.g. ``export_onnx_schemas``) push
      // onnxsim's schemas into the separate registry the ``onnx`` Python
      // module uses, so tools built on ``onnx.defs``/``onnx.checker``
      // recognize them without onnxsim.
      .def("_get_all_schemas", []() {
        std::vector<PySchema> ret;
        for (const auto& schema :
             onnx::OpSchemaRegistry::get_all_schemas_with_history()) {
          std::vector<PyFormalParameter> inputs;
          inputs.reserve(schema.inputs().size());
          for (const auto& p : schema.inputs()) {
            inputs.emplace_back(p.GetName(), p.GetDescription(), p.GetTypeStr(),
                                static_cast<int>(p.GetOption()),
                                p.GetIsHomogeneous(), p.GetMinArity());
          }
          std::vector<PyFormalParameter> outputs;
          outputs.reserve(schema.outputs().size());
          for (const auto& p : schema.outputs()) {
            outputs.emplace_back(p.GetName(), p.GetDescription(),
                                 p.GetTypeStr(),
                                 static_cast<int>(p.GetOption()),
                                 p.GetIsHomogeneous(), p.GetMinArity());
          }
          std::vector<PyAttribute> attributes;
          attributes.reserve(schema.attributes().size());
          for (const auto& kv : schema.attributes()) {
            const auto& attr = kv.second;
            attributes.emplace_back(attr.name, attr.description,
                                    static_cast<int>(attr.type), attr.required,
                                    attr.default_value);
          }
          std::vector<PyTypeConstraint> type_constraints;
          type_constraints.reserve(schema.typeConstraintParams().size());
          for (const auto& tc : schema.typeConstraintParams()) {
            type_constraints.emplace_back(tc.type_param_str,
                                          tc.allowed_type_strs, tc.description);
          }
          const char* doc = schema.doc();
          ret.emplace_back(schema.Name(), schema.domain(),
                           schema.since_version(),
                           doc ? std::string(doc) : std::string(),
                           std::move(inputs), std::move(outputs),
                           std::move(attributes), std::move(type_constraints),
                           schema.has_type_and_shape_inference_function());
        }
        return ret;
      });

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

  // Python-facing view of TensorPool (see tensor_pool.h): the named,
  // ref-counted tensor store that ``load_model`` below populates as it
  // resolves a model's external weights. Returned alongside the model so a
  // caller can inspect what was actually loaded -- e.g. verify a tensor's
  // ContentHash, or hydrate one on demand after a ``hydrate_all=False``
  // load -- without re-deriving it from the model's own initializers.
  //
  // Caveat inherited from mmap_file.h's TryMmapFile (see its own doc
  // comment): for a classic-external-data load, an entry's bytes may alias
  // a live memory mapping of the file on disk, so on Windows that file
  // can't be deleted or moved while this pool object is still alive (POSIX
  // has no such restriction). `bytes`/`dtype`/`shape`/`content_hash` below
  // all return independent copies, so extracting what's needed and then
  // dropping the pool is always safe.
  py::class_<onnxsim::tensor_pool::TensorPool>(m, "TensorPool")
      .def("__len__", &onnxsim::tensor_pool::TensorPool::size)
      .def("__contains__",
           [](const onnxsim::tensor_pool::TensorPool& pool,
              const std::string& name) { return pool.Find(name) != nullptr; })
      .def("names",
           [](const onnxsim::tensor_pool::TensorPool& pool) {
             std::vector<std::string> names;
             names.reserve(pool.size());
             for (const auto& [name, entry] : pool) names.push_back(name);
             return names;
           })
      .def("dtype",
           [](const onnxsim::tensor_pool::TensorPool& pool,
              const std::string& name) -> int32_t {
             const auto* entry = pool.Find(name);
             if (entry == nullptr) {
               throw std::out_of_range("TensorPool: no entry named '" + name +
                                       "'");
             }
             return entry->dtype;
           })
      .def("shape",
           [](const onnxsim::tensor_pool::TensorPool& pool,
              const std::string& name) -> std::vector<int64_t> {
             const auto* entry = pool.Find(name);
             if (entry == nullptr) {
               throw std::out_of_range("TensorPool: no entry named '" + name +
                                       "'");
             }
             return entry->shape;
           })
      .def("bytes",
           [](const onnxsim::tensor_pool::TensorPool& pool,
              const std::string& name) -> py::bytes {
             const auto* entry = pool.Find(name);
             if (entry == nullptr) {
               throw std::out_of_range("TensorPool: no entry named '" + name +
                                       "'");
             }
             return py::bytes(entry->data.data(), entry->data.size());
           })
      .def("content_hash", &onnxsim::tensor_pool::TensorPool::ContentHash,
           "name"_a);

  // Standalone safetensors/GGUF archive export/import: a model's graph and
  // weights packaged together in one ecosystem-standard file (see
  // onnxsim/tensor_pool_bridge.h and tensor_pool_gguf_bridge.h's *Standalone
  // functions for the real-offset design). Exchanged as bytes for the model
  // (like ``simplify``) and a real path for the archive itself, since the
  // archive is inherently file-based.
  //
  // `tensor_bytes` carries each eligible tensor's raw_data separately from
  // `model_bytes` (whose Python caller has already stripped those same
  // fields before serializing) -- avoids paying a full protobuf encode
  // (Python) + decode (here) of the tensor data on top of the copies
  // AdoptAllWithPlaceholderOffsets/the archive write already make; see that
  // function's doc comment. Converting each py::bytes to a std::string is
  // the one necessary copy of that tensor's bytes crossing into C++.
  m.def(
      "export_safetensors",
      [](const py::bytes& model_bytes,
         std::map<std::string, py::bytes>& tensor_bytes,
         const std::string& out_path) {
        onnx::ModelProto model;
        ParseProtoFromBytes(&model, model_bytes.c_str(), model_bytes.size());
        std::map<std::string, std::string> external_bytes;
        for (auto& [name, b] : tensor_bytes) {
          external_bytes.emplace(name, std::string(b.c_str(), b.size()));
        }
        onnxsim::tensor_pool::TensorPool pool;
        onnxsim::tensor_pool::SaveModelAsSafetensorsStandalone(
            model, out_path, pool, &external_bytes);
      },
      "model_bytes"_a, "tensor_bytes"_a, "out_path"_a);

  // Always loads lazily (hydrate_all=false) for the same reason
  // load_model's binding does -- see that binding's comment. Returns the
  // TensorPool too so the Python wrapper can hydrate tensor-by-tensor
  // itself instead of paying a second full-model serialize/parse here.
  m.def(
      "import_safetensors",
      [](const std::string& in_path)
          -> std::tuple<py::bytes, onnxsim::tensor_pool::TensorPool> {
        onnx::ModelProto model;
        onnxsim::tensor_pool::TensorPool pool;
        if (!onnxsim::tensor_pool::LoadModelFromSafetensors(
                in_path, &model, pool, /*hydrate_all=*/false)) {
          throw std::runtime_error(
              "safetensors file has no embedded onnxsim model (a plain "
              "weights-only archive is not importable as a graph)");
        }
        const std::string out = model.SerializeAsString();
        return {py::bytes(out.data(), out.size()), std::move(pool)};
      },
      "in_path"_a);

  m.def(
      "export_gguf",
      [](const py::bytes& model_bytes,
         std::map<std::string, py::bytes>& tensor_bytes,
         const std::string& out_path) {
        onnx::ModelProto model;
        ParseProtoFromBytes(&model, model_bytes.c_str(), model_bytes.size());
        std::map<std::string, std::string> external_bytes;
        for (auto& [name, b] : tensor_bytes) {
          external_bytes.emplace(name, std::string(b.c_str(), b.size()));
        }
        onnxsim::tensor_pool::TensorPool pool;
        onnxsim::tensor_pool::SaveModelAsGGUFStandalone(
            model, out_path, pool, /*string_metadata=*/{}, &external_bytes);
      },
      "model_bytes"_a, "tensor_bytes"_a, "out_path"_a);

  m.def(
      "import_gguf",
      [](const std::string& in_path)
          -> std::tuple<py::bytes, onnxsim::tensor_pool::TensorPool> {
        onnx::ModelProto model;
        onnxsim::tensor_pool::TensorPool pool;
        if (!onnxsim::tensor_pool::LoadModelFromGGUF(in_path, &model, pool,
                                                     /*hydrate_all=*/false)) {
          throw std::runtime_error(
              "gguf file has no embedded onnxsim model (a plain weights-only "
              "archive is not importable as a graph)");
        }
        const std::string out = model.SerializeAsString();
        return {py::bytes(out.data(), out.size()), std::move(pool)};
      },
      "in_path"_a);

  // Unified model loader: dispatches on `path`'s extension between plain
  // ONNX (`.onnx`, or anything else -- classic external data resolved via
  // LoadModelWithTensorPool's mmap'd TensorPool, see that function's doc
  // comment for the rationale) and onnxsim's own self-describing archives
  // (`.safetensors` / `.gguf`, resolved the same way import_safetensors/
  // import_gguf above do). Always returns the TensorPool it resolved into
  // (empty for a model with no external weights) alongside the model
  // bytes, unlike import_safetensors/import_gguf, which discard theirs --
  // see the TensorPool binding above for why that's useful.
  //
  // Always loads with hydrate_all=false at this layer -- deliberately,
  // *not* a caller-facing option here. Measured: hydrating in C++ and then
  // crossing the FFI boundary re-serializes and re-parses the *whole*
  // model, tensor bytes included, on top of the mmap/copy work hydration
  // itself already did -- on a 190MB/2000-tensor model that made the
  // "hydrate_all=True" case ~3.6x SLOWER than plain onnx.load(), not
  // faster, while this lazy load alone takes ~3ms (mmap only, no copies).
  // onnxsim.load_model's Python wrapper is the one that offers a
  // hydrate_all option, implemented by copying tensor-by-tensor straight
  // from the returned TensorPool (one copy per tensor, same as any loader
  // must eventually pay) instead of round-tripping the whole model.
  m.def(
      "load_model",
      [](const std::string& path)
          -> std::tuple<py::bytes, onnxsim::tensor_pool::TensorPool> {
        onnx::ModelProto model;
        onnxsim::tensor_pool::TensorPool pool;
        std::string ext;
        {
          auto pos = path.find_last_of('.');
          if (pos != std::string::npos) {
            ext = path.substr(pos);
            for (char& c : ext) {
              c = static_cast<char>(
                  std::tolower(static_cast<unsigned char>(c)));
            }
          }
        }
        if (ext == ".safetensors") {
          if (!onnxsim::tensor_pool::LoadModelFromSafetensors(
                  path, &model, pool, /*hydrate_all=*/false)) {
            throw std::runtime_error(
                "safetensors file has no embedded onnxsim model (a plain "
                "weights-only archive is not importable as a graph)");
          }
        } else if (ext == ".gguf") {
          if (!onnxsim::tensor_pool::LoadModelFromGGUF(path, &model, pool,
                                                       /*hydrate_all=*/false)) {
            throw std::runtime_error(
                "gguf file has no embedded onnxsim model (a plain "
                "weights-only archive is not importable as a graph)");
          }
        } else {
          onnxsim::tensor_pool::LoadModelWithTensorPool(path, &model, pool,
                                                        /*hydrate_all=*/false);
        }
        const std::string out = model.SerializeAsString();
        return {py::bytes(out.data(), out.size()), std::move(pool)};
      },
      "path"_a);

  // Hydrates `model`'s initializers, by name, from any GGUF file --
  // including a plain third-party weights-only checkpoint with no embedded
  // onnxsim model (unlike import_gguf, which requires one). A K-quant
  // tensor (Q4_K/Q5_K/Q6_K/Q8_0 -- what most real quantized checkpoints,
  // e.g. Unsloth's GGUF exports, actually use for the bulk of their
  // weights) is decoded to float32; see
  // ImportModelWithGGUFToPool/HydrateTensorProtoFromGGUF in
  // tensor_pool_gguf_bridge.h. Returns (byte-free model bytes, the matched
  // tensors' already-decoded bytes as a TensorPool, names of GGUF tensors
  // present in the file but skipped because their ggml_type has no
  // representation TensorPool can hold at all, e.g. a legacy Q4_0 or IQ*-
  // family tensor -- NOT tensors simply absent from `model`'s
  // initializers, which this silently leaves alone rather than reporting).
  // Splitting the matched tensors out into a TensorPool, rather than
  // writing them into `model` and returning the whole thing serialized,
  // avoids a full protobuf encode (here) + decode (Python) of the
  // (potentially huge) newly-hydrated tensor data on top of the copies
  // ImportModelWithGGUFToPool already makes.
  m.def(
      "import_gguf_weights",
      [](const py::bytes& model_bytes, const std::string& gguf_path)
          -> std::tuple<py::bytes, onnxsim::tensor_pool::TensorPool,
                        std::vector<std::string>> {
        onnx::ModelProto model;
        ParseProtoFromBytes(&model, model_bytes.c_str(), model_bytes.size());
        onnxsim::tensor_pool::TensorPool matched;
        std::vector<std::string> skipped;
        onnxsim::tensor_pool::ImportModelWithGGUFToPool(model, gguf_path,
                                                        matched, &skipped);
        const std::string out = model.SerializeAsString();
        return {py::bytes(out.data(), out.size()), std::move(matched),
                std::move(skipped)};
      },
      "model_bytes"_a, "gguf_path"_a);

  // Reads a GGUF file's architecture hyperparameters (general.architecture,
  // <arch>.block_count, <arch>.attention.head_count, <arch>.rope.freq_base,
  // ...) and per-tensor name/shape/ggml_type list, WITHOUT reading any
  // tensor byte data -- see tensor_pool.h's GGUFMetadata/ReadGGUFMetadata
  // doc comments. This is the piece TensorPool::LoadGGUF/import_gguf_weights
  // above never surfaced: they parse the same header section but only ever
  // look at general.alignment before moving on to loading tensor *values*.
  // Returns {"kv": {key: int|float|str|bool, ...},
  //          "tensors": [{"name": str, "shape": [int, ...],
  //                       "ggml_type": int}, ...]}. ARRAY-typed metadata
  // values (e.g. tokenizer.ggml.tokens) are omitted from "kv" entirely --
  // see GGUFMetadata's doc comment for why.
  m.def(
      "read_gguf_metadata",
      [](const std::string& path) -> py::dict {
        onnxsim::tensor_pool::GGUFMetadata meta =
            onnxsim::tensor_pool::ReadGGUFMetadata(path);

        py::dict kv;
        for (const auto& [key, value] : meta.kv) {
          switch (value.kind) {
            case onnxsim::tensor_pool::GGUFMetadataValue::Kind::kInt:
              kv[key.c_str()] = value.int_value;
              break;
            case onnxsim::tensor_pool::GGUFMetadataValue::Kind::kFloat:
              kv[key.c_str()] = value.float_value;
              break;
            case onnxsim::tensor_pool::GGUFMetadataValue::Kind::kString:
              kv[key.c_str()] = value.string_value;
              break;
            case onnxsim::tensor_pool::GGUFMetadataValue::Kind::kBool:
              kv[key.c_str()] = value.bool_value;
              break;
          }
        }

        py::list tensors;
        for (const auto& t : meta.tensors) {
          py::dict entry;
          entry["name"] = t.name;
          py::list shape;
          for (int64_t d : t.shape) shape.append(d);
          entry["shape"] = shape;
          entry["ggml_type"] = t.ggml_type;
          tensors.append(entry);
        }

        py::dict out;
        out["kv"] = kv;
        out["tensors"] = tensors;
        return out;
      },
      "path"_a);
}
