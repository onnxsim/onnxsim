/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
#include <stdexcept>
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
  // ``attention_tiled=False`` reserves each attention node's S x S score matrix
  // as transient workspace in the plan (see memory_planning.h); the default
  // leaves the plan unchanged.
  m.def(
      "_memory_plan",
      [](const py::bytes& model_bytes, bool run_shape_inference,
         bool attention_tiled) {
        onnx::ModelProto model;
        onnx::ParseProtoFromBytes(&model, model_bytes.c_str(),
                                  model_bytes.size());
        const onnxsim::GraphView view =
            GetGraphView(model, run_shape_inference);
        const onnxsim::MemoryPlan plan =
            onnxsim::ComputeActivationMemoryPlan(view, attention_tiled);
        return MemoryPlanToPyTuple(plan);
      },
      "model_bytes"_a, "run_shape_inference"_a = true,
      "attention_tiled"_a = true);

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Dynamically quantizes an existing "com.microsoft" Attention node (see
  // fuse_attention.h -- this does not fuse attention itself) into its
  // quantized counterpart, QAttention -- see QuantizeAttentionDynamic in
  // onnxsim.h. Pure graph rewrite: no ModelExecutor or calibration data
  // needed.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Dynamically quantizes MatMul/Gemm nodes whose weight is structurally
  // ternary ({-s, 0, +s} per output column, e.g. BitNet b1.58) into the same
  // DynamicQuantizeLinear/MatMulInteger shape as quantize_dynamic, but with a
  // lossless ternary weight encoding instead of a rounded approximation --
  // see QuantizeTernary in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Weight-only quantizes MatMul/Gemm/Conv weights to INT8 (per output
  // channel, symmetric) via a single DequantizeLinear -- activations are
  // never touched, so no calibration data or ModelExecutor is needed. See
  // QuantizeWeightOnly in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Block-wise INT4 weight-only quantizes MatMul/Gemm/Conv weights (one
  // symmetric scale per 32-element block of the reduction dimension, per
  // output channel) via a single DequantizeLinear(block_size=32) --
  // activations are never touched, so no calibration data or ModelExecutor
  // is needed. See QuantizeWeightOnlyInt4 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Block-wise INT4 weight-only quantizes MatMul/Gemm weights into ONNX
  // Runtime's own com.microsoft::MatMulNBits contrib op -- a vendor-specific
  // (ORT-only) counterpart to quantize_weight_only_int4's portable standard-
  // ONNX output. See QuantizeWeightOnlyMatMulNBits in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // INT16 weight-only quantizes MatMul/Gemm/Conv weights (one symmetric
  // scale per output channel, INT16's finer step than INT8's) -- activations
  // are never touched, so no calibration data or ModelExecutor is needed.
  // See QuantizeWeightOnlyInt16 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Block-wise INT8 weight-only quantizes MatMul/Gemm/Conv weights (one
  // symmetric scale per 32-element block of the flattened reduction
  // dimension, per output channel) via a single DequantizeLinear(axis=...,
  // block_size=32) -- activations are never touched, so no calibration data
  // or ModelExecutor is needed. See QuantizeWeightOnlyInt8Block in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // OCP Microscaling MXFP4 weight-only quantizes every MatMul/vanilla-Gemm
  // whose weight is a constant float32 tensor whose reduction dimension is
  // divisible by 32, via a Gather-a-codebook-then-scale dequant chain (no
  // native ONNX MX tensor type). Activations are never touched, so no
  // calibration data is needed. See QuantizeWeightOnlyMXFP4 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QLoRA-style double quantization: quantizes every already-present
  // DequantizeLinear node's own (large enough) constant scale tensor to
  // UINT8 with a per-tensor meta-scale. See ApplyDoubleQuantization in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Magnitude pruning (Han et al., 2015): zeros the least-magnitude entries
  // of every MatMul/vanilla-Gemm/Conv/com.microsoft::Attention layer's
  // constant weight, independently per output row/filter. Data-free.
  // `n`/`m` are `None` (unstructured, ranked by `sparsity`) or both given
  // together (N:M semi-structured). `global_sparsity` pools every matched
  // layer's importance into one whole-model ranking; incompatible with
  // `n`/`m`. Same `n`/`m`/`global_sparsity` shape as apply_wanda_pruning's
  // own binding above. See PruneMagnitude in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Structured (channel) pruning: removes whole output channels from
  // MatMul/vanilla-Gemm and Conv layers -- real structural pruning, not
  // just value-only zeroing. See ApplyStructuredPruning in
  // structured_pruning_entry.h. `importance_norm` ("l1"/"l2") and
  // `global_sparsity` mirror pruning.py's own `apply_structured_pruning`
  // parameters of the same names exactly -- see that function's own
  // declaration comment.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Attention-head pruning: removes whole attention heads (or, for
  // grouped-query attention, whole KV groups) from every matched fused
  // self-attention block. See ApplyAttentionHeadPruning in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // The calibration-driven (Wanda-style) upgrade of
  // apply_attention_head_pruning above -- same executor-as-first-argument
  // shape, and the same `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) crossing convention, as apply_structured_wanda_
  // pruning's own binding above (see that binding's own comment for the
  // full calibration-crossing design). See ApplyAttentionHeadWandaPruning
  // in structured_pruning_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's "importance matrix" (imatrix): weight-only-quantizes every
  // matched MatMul/vanilla-Gemm node's constant 2-D FLOAT32 weight to INT4
  // (folded as a float32 quantize-dequantize round trip, no new graph
  // nodes), using real calibration activations to bias each weight block's
  // scale search toward minimizing importance-weighted squared error. Same
  // executor-as-first-argument, `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) crossing convention as apply_wanda_pruning's own
  // binding above. See ApplyImatrixQuantization in imatrix_quant_entry.h
  // for the full scope and onnxsim/imatrix_quant.py for the technique this
  // ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_imatrix_quantization",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t block_size, int64_t num_scale_candidates, double scale_lo,
         double scale_hi,
         const std::vector<std::string>& skip_names) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const std::unordered_set<std::string> skip_names_set(skip_names.begin(),
                                                             skip_names.end());
        const auto result = ApplyImatrixQuantization(
            model, *executor, calibration_data, block_size,
            num_scale_candidates, scale_lo, scale_hi, skip_names_set);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "block_size"_a = 32,
      "num_scale_candidates"_a = 41, "scale_lo"_a = 0.4, "scale_hi"_a = 1.6,
      "skip_names"_a = std::vector<std::string>());
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Outlier Suppression Gamma Migration (Wei et al., 2022): folds the
  // SmoothQuant-style per-channel migration scale directly into every
  // matched LayerNormalization's gamma/bias (adding zero runtime nodes),
  // compensated by scaling every downstream MatMul/vanilla-Gemm
  // consumer's weight rows by the same scale -- a lossless
  // pre-conditioning transform ahead of a separate W8A8 quantizer, never
  // a quantization scheme itself. Same executor-as-first-argument,
  // `calibration_data` (List[Dict[str, onnx.TensorProto]]) crossing
  // convention as apply_imatrix_quantization's own binding above. See
  // ApplyOutlierSuppression in outlier_suppression_entry.h for the full
  // scope and onnxsim/outlier_suppression.py for the technique this
  // ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_outlier_suppression",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double alpha, double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyOutlierSuppression(
            model, *executor, calibration_data, alpha, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "alpha"_a = 0.5,
      "epsilon"_a = 1e-5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LLM.int8() (Dettmers et al., 2022): decomposes every matched
  // MatMul/vanilla-Gemm node into a float32 outlier part plus a
  // vector-wise INT8 part computed via MatMulInteger (per-row activation
  // scales at runtime, per-output-channel weight scales offline,
  // uint8 activation at zero-point 128). Same
  // executor-as-first-argument, `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) crossing convention as
  // apply_outlier_suppression's own binding above. See ApplyLlmInt8 in
  // llm_int8_entry.h for the full scope and onnxsim/llm_int8.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_llm_int8",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double outlier_threshold, double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyLlmInt8(model, *executor, calibration_data,
                                         outlier_threshold, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "outlier_threshold"_a = 6.0, "epsilon"_a = 1e-8);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // SpQR (Dettmers et al., 2023): outlier-aware block-wise INT4
  // quantization -- per-element outliers (by Hessian-diagonal-weighted
  // sensitivity) are excluded from their own block's scale and stored as
  // an exact sparse correction. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_llm_int8's own binding
  // above. See ApplySpqr in spqr_entry.h for the full scope and
  // onnxsim/spqr.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_spqr",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t block_size, double outlier_fraction) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplySpqr(model, *executor, calibration_data,
                                      block_size, outlier_fraction);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "block_size"_a = 16,
      "outlier_fraction"_a = 0.01);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // PB-LLM (Shang et al., 2024): a structured mixed-precision binarizer --
  // per matched layer, the `salient_ratio` fraction of input channels with
  // the highest Hessian-diagonal-weighted magnitude stay INT8, every other
  // channel is binarized to ~1 bit/element. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_llm_int8's own binding
  // above. See ApplyPbLlm in pb_llm_entry.h for the full scope and
  // onnxsim/pb_llm.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_weight_only_pb_llm",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double salient_ratio) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyPbLlm(model, *executor, calibration_data, salient_ratio);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "salient_ratio"_a = 0.15);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // SqueezeLLM (Kim et al., 2023): sensitivity-weighted per-group codebook
  // (a real GatherND-based graph rewrite, not folded to a single
  // initializer) plus a dense-and-sparse outlier correction. Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_llm_int8's own binding above. See ApplySqueezeLlm in
  // squeezellm_entry.h for the full scope and onnxsim/squeezellm.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_weight_only_squeezellm",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t block_size, int64_t bits, double outlier_fraction,
         int64_t num_kmeans_iterations) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplySqueezeLlm(model, *executor, calibration_data, block_size,
                            bits, outlier_fraction, num_kmeans_iterations);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "block_size"_a = 32,
      "bits"_a = 4, "outlier_fraction"_a = 0.0045,
      "num_kmeans_iterations"_a = 20);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // BiLLM (Huang et al., 2024, ICML): a genuine ~1-bit-average weight
  // binarizer -- Hessian-guided salient-column selection, a two-level
  // binary residual approximation for salient columns, plain flat binary
  // for the rest, and OBC-style forward error compensation (reusing
  // apply_gptq's own Cholesky-factored-inverse-Hessian mechanism). Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_llm_int8's own binding above. See ApplyBillm in billm_entry.h for
  // the full scope and onnxsim/billm.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_billm",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t block_size, double percdamp,
         int64_t max_salient_search) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyBillm(model, *executor, calibration_data, block_size, percdamp,
                       max_salient_search);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "block_size"_a = 128,
      "percdamp"_a = 0.01, "max_salient_search"_a = 30);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // KV-cache quantization (KIVI/KVQuant): per-channel static INT8 for a
  // matched Concat(past, new, axis=seq) stream's own Key-style values,
  // per-token dynamic INT8 (data-free) for Value-style ones -- see
  // onnxsim.kv_cache_quantization's own docstring for the split criterion
  // (present output name contains ".value", or listed explicitly in
  // `value_output_names`). Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_llm_int8's own binding
  // above. See ApplyKvCacheQuantization in kv_cache_quantization_entry.h
  // for the full scope and onnxsim/kv_cache_quantization.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_kv_cache",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         const std::vector<std::string>& value_output_names) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyKvCacheQuantization(
            model, *executor, calibration_data, value_output_names);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "value_output_names"_a = std::vector<std::string>());
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // OWQ (Lee, Park, Kim, Kim and Sung, 2023, AAAI 2024): rescues the top
  // `outlier_fraction` OBS-salient columns of an already-
  // quantize_weight_only_int4-quantized layer back to exact float32
  // precision via an additive Gather/MatMul/Add correction, reusing
  // apply_gptq's own Cholesky-factored-inverse-Hessian mechanism --
  // `quantized_model`'s own INT4 codes are never modified. Same
  // two-model executor-as-first-argument shape as apply_gptq's own binding
  // below; `calibration_data` (List[Dict[str, onnx.TensorProto]]) is keyed
  // to the float model's own graph inputs. See ApplyOwq in owq_entry.h for
  // the full scope and onnxsim/owq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_owq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double outlier_fraction, double percdamp) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyOwq(float_model, quantized_model, *executor, calibration_data,
                     outlier_fraction, percdamp);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "outlier_fraction"_a = 0.01, "percdamp"_a = 0.01);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // GEAR (Kang et al., 2024): low-rank-plus-sparse residual compensation
  // layered on top of onnxsim.kv_cache_quantization's own static
  // per-channel INT8 base quantization, applied only to a freshly-produced
  // KV-cache token. Same executor-as-first-argument, `calibration_data`
  // crossing convention as apply_llm_int8's own binding above. See
  // ApplyGear in gear_entry.h for the full scope and onnxsim/gear.py for
  // the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gear",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t rank, double outlier_fraction) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGear(model, *executor, calibration_data, rank,
                                      outlier_fraction);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "rank"_a = 4,
      "outlier_fraction"_a = 0.05);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // RotateKV (Su et al., 2025): fits a per-stream orthogonal rotation from
  // a matched KV-cache stream's own calibration-activation covariance and
  // applies it to both the stream's fresh Key and its compensating Query,
  // exact by construction for any orthogonal R. Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_llm_int8's own binding above. See ApplyRotateKv in
  // rotatekv_entry.h for the full scope and onnxsim/rotatekv.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_rotatekv",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyRotateKv(model, *executor, calibration_data);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // GPTQ (Frantar et al., 2022): sequential, Hessian-compensated INT4
  // rounding for every quantize_weight_only_int4-quantized MatMul/Gemm
  // layer shared (by node output name) between a float model and its
  // quantized counterpart, reusing that scheme's own per-block scales and
  // changing only which integer each element rounds to. Same
  // executor-as-first-argument shape as every other calibration-driven
  // binding, except the first two arguments are the float and quantized
  // model bytes respectively; `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) is keyed to the float model's own graph inputs.
  // See ApplyGptq in gptq_entry.h for the full scope and
  // onnxsim/gptq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gptq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double percdamp, int64_t proc_block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyGptq(float_model, quantized_model, *executor, calibration_data,
                      percdamp, proc_block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "percdamp"_a = 0.01, "proc_block_size"_a = 128);

  // GPTQ's column loop alone (GptqColumns in gptq_entry.h) on float64
  // arrays, for onnxsim.gptq._gptq_quantize_columns: w_nk [N, K],
  // scale_blocks [N, G], hinv [K, K], optional per-element code bounds
  // lo / hi [N, K]. Returns the [N, K] codes.
  using F64Array = py::ndarray<const double, py::c_contig, py::device::cpu>;
  m.def(
      "gptq_columns",
      [](F64Array w_nk, F64Array scale_blocks, int64_t quant_block_size,
         F64Array hinv, int64_t proc_block_size, std::optional<F64Array> lo,
         std::optional<F64Array> hi, double qmin, double qmax,
         int num_threads) {
        if (w_nk.ndim() != 2 || scale_blocks.ndim() != 2 || hinv.ndim() != 2) {
          throw std::invalid_argument("gptq_columns: arrays must be 2-D");
        }
        const int64_t n = static_cast<int64_t>(w_nk.shape(0));
        const int64_t k = static_cast<int64_t>(w_nk.shape(1));
        const int64_t groups = static_cast<int64_t>(scale_blocks.shape(1));
        if (static_cast<int64_t>(scale_blocks.shape(0)) != n ||
            static_cast<int64_t>(hinv.shape(0)) != k ||
            static_cast<int64_t>(hinv.shape(1)) != k || quant_block_size <= 0 ||
            proc_block_size <= 0 || (k - 1) / quant_block_size >= groups ||
            lo.has_value() != hi.has_value()) {
          throw std::invalid_argument("gptq_columns: inconsistent shapes");
        }
        for (const auto* b : {&lo, &hi}) {
          if (b->has_value() &&
              ((*b)->ndim() != 2 || static_cast<int64_t>((*b)->shape(0)) != n ||
               static_cast<int64_t>((*b)->shape(1)) != k)) {
            throw std::invalid_argument("gptq_columns: lo / hi must be [N, K]");
          }
        }
        auto* codes = new double[static_cast<size_t>(n * k)];
        py::capsule owner(
            codes, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        {
          py::gil_scoped_release release;
          GptqColumns(w_nk.data(), n, k, scale_blocks.data(), groups,
                      quant_block_size, hinv.data(), proc_block_size,
                      lo ? lo->data() : nullptr, hi ? hi->data() : nullptr,
                      qmin, qmax, num_threads, codes);
        }
        return py::ndarray<py::numpy, double, py::ndim<2>>(
            codes, {static_cast<size_t>(n), static_cast<size_t>(k)}, owner);
      },
      "w_nk"_a, "scale_blocks"_a, "quant_block_size"_a, "hinv"_a,
      "proc_block_size"_a, "lo"_a = py::none(), "hi"_a = py::none(),
      "qmin"_a = -7.0, "qmax"_a = 7.0, "num_threads"_a = 4);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // GPTAQ (Li, Yin, Lee, Xiao, Panda, 2025): a small, closed-form
  // asymmetric-calibration correction layered on top of apply_gptq's own
  // binding above -- probes BOTH the float model AND the quantized model
  // at each candidate layer's own activation input, folding the gap
  // between the two (the upstream quantization's own accumulated
  // corruption) into GPTQ's own per-column procedure via one small, exact
  // pre-computation (`Shift`), then runs GPTQ's own column algorithm
  // unchanged on `W + Shift`. Same two-model executor-as-first-argument
  // shape as apply_gptq's own binding above; `calibration_data` (List[
  // Dict[str, onnx.TensorProto]]) is run through BOTH models (unlike
  // apply_gptq's own binding, which only ever probes the float model).
  // See ApplyGptaq in gptaq_entry.h for the full scope and
  // onnxsim/gptaq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gptaq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double percdamp, int64_t proc_block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyGptaq(float_model, quantized_model, *executor,
                       calibration_data, percdamp, proc_block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "percdamp"_a = 0.01, "proc_block_size"_a = 128);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QuantEase (Behdin, Acharya, Gupta, Song, Zhu and Keerthi, 2023): plain
  // cyclic coordinate descent over the same per-row reconstruction
  // objective apply_gptq's own binding above minimizes via a single greedy
  // Hessian-compensated sweep -- repeatedly sweeps every column of every
  // quantize_weight_only_int4-quantized MatMul/Gemm layer shared (by node
  // output name) between a float model and its quantized counterpart,
  // moving each column to the closed-form unconstrained optimum of the
  // objective itself (needing only the layer's own Hessian, never its
  // inverse), rounded to the nearest grid point, for `num_epochs` full
  // sweeps. Same two-model executor-as-first-argument shape as apply_gptq's
  // own binding above. See ApplyQuantease in quantease_entry.h for the full
  // scope and onnxsim/quantease.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_quantease",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_epochs) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyQuantease(float_model, quantized_model, *executor,
                           calibration_data, num_epochs);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_epochs"_a = 4);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LQER (Zhang et al., 2024): activation-weighted low-rank correction of
  // an INT4-quantized MatMul/Gemm layer's own existing reconstruction
  // error -- an activation-weighted generalization of
  // apply_low_rank_compensation's own plain (unweighted) SVD. Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_gptq's own binding above. See ApplyLqer in lqer_entry.h for the
  // full scope and onnxsim/lqer.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_lqer",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t rank, double eps) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyLqer(float_model, quantized_model, *executor,
                                      calibration_data, rank, eps);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "rank"_a = 8, "eps"_a = 1e-6);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Norm Tweaking (Li, Xu, Ni, Chen, Ye, Sun, 2023): recalibrates a
  // LayerNormalization node's own scale/bias in place so its output
  // distribution's per-channel mean/standard deviation matches the float
  // model's own, correcting for the distribution shift a quantized
  // upstream layer introduces. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_gptq's own binding
  // above. See ApplyNormTweaking in norm_tweaking_entry.h for the full
  // scope and onnxsim/norm_tweaking.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_norm_tweaking",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double eps) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyNormTweaking(float_model, quantized_model,
                                              *executor, calibration_data, eps);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "eps"_a = 1e-6);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // D2Quant's Deviation-Aware Correction (DAC) (Yan et al., 2026): folds a
  // measured, quantization-induced per-channel mean-shift deviation
  // directly into the following LayerNormalization's own bias. Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_gptq's own binding above. See ApplyDac in dac_entry.h for the
  // full scope and onnxsim/d2quant.py's own apply_dac for the technique
  // this ports (that module's own apply_dsq is a separate technique,
  // already ported elsewhere as "apply_dsq" above).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_dac",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double min_expected_error_reduction,
         double correction_threshold) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyDac(float_model, quantized_model, *executor, calibration_data,
                     min_expected_error_reduction, correction_threshold);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "min_expected_error_reduction"_a = 0.5,
      "correction_threshold"_a = 1e-12);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AWQ (Lin et al., 2023): grid-searched per-channel weight rescaling
  // for every quantize_weight_only_int4-quantized MatMul/Gemm layer
  // shared (by node output name) between a float model and its quantized
  // counterpart, re-quantizing from scratch at each grid point and
  // keeping the exponent with the lowest reconstruction error. Same
  // two-model executor-as-first-argument shape as apply_gptq's own
  // binding above; `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) is keyed to the float model's own graph inputs.
  // See ApplyAwq in awq_entry.h for the full scope and onnxsim/awq.py
  // for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_awq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_alpha_steps) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyAwq(float_model, quantized_model, *executor,
                                     calibration_data, num_alpha_steps);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_alpha_steps"_a = 20);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AdaRound (Nagel et al., 2020): Nagel et al.'s rectified-sigmoid
  // relaxation of each weight element's floor/ceil rounding decision,
  // optimized by a hand-rolled Adam loop to minimize a layer's own
  // reconstruction error against real calibration activations. Same
  // two-model executor-as-first-argument shape as apply_gptq's own
  // binding above (candidates are processed independently, so `executor`
  // is invoked once, up front, the same as apply_gptq's own);
  // `calibration_data` (List[Dict[str, onnx.TensorProto]]) is keyed to
  // the float model's own graph inputs. `beta_start`/`beta_end` are the
  // two ends of apply_adaround's own `beta_range` tuple, split into
  // separate parameters here since this binding layer has no tuple type.
  // See ApplyAdaround in adaround_entry.h for the full scope (including
  // its own accepted numerical scope -- an iterative optimization, not a
  // closed-form computation) and onnxsim/adaround.py for the technique
  // this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_adaround",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_iterations, double learning_rate, double reg_param,
         double warm_start, double beta_start, double beta_end) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyAdaround(float_model, quantized_model, *executor,
                          calibration_data, num_iterations, learning_rate,
                          reg_param, warm_start, beta_start, beta_end);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_iterations"_a = 300, "learning_rate"_a = 0.1,
      "reg_param"_a = 0.01, "warm_start"_a = 0.2, "beta_start"_a = 20.0,
      "beta_end"_a = 2.0);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AdaQuant (Hubara, Nahshan, Hanani, Banner, Soudry, 2020/2021): a
  // per-layer joint Adam optimization over the same rectified-sigmoid
  // weight-rounding relaxation apply_adaround's own binding above uses,
  // PLUS the activation's own (scale, zero_point) for
  // onnxsim.quantize_static's W8A8 QDQ scheme (not
  // quantize_weight_only_int4's blocked-INT4 one). Same two-model
  // executor-as-first-argument shape as apply_adaround's own binding
  // above (candidates are processed independently, so `executor` is
  // invoked once, up front); `beta_start`/`beta_end` are the two ends of
  // apply_adaquant's own `beta_range` tuple, split the same way. See
  // ApplyAdaquant in adaquant_entry.h for the full scope and
  // onnxsim/adaquant.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_adaquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_iterations, double weight_learning_rate,
         double activation_learning_rate, double reg_param, double warm_start,
         double beta_start, double beta_end) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyAdaquant(
            float_model, quantized_model, *executor, calibration_data,
            num_iterations, weight_learning_rate, activation_learning_rate,
            reg_param, warm_start, beta_start, beta_end);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_iterations"_a = 300,
      "weight_learning_rate"_a = 0.1, "activation_learning_rate"_a = 0.01,
      "reg_param"_a = 0.01, "warm_start"_a = 0.2, "beta_start"_a = 20.0,
      "beta_end"_a = 2.0);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // OmniQuant (Shao et al., 2023): grid-searched Learnable Weight Clipping
  // plus a closed-form-shift/grid-searched-scale Learnable Equivalent
  // Transformation, applied to every quantize_weight_only_int4-quantized
  // MatMul/Gemm layer shared (by node output name) between a float model
  // and its quantized counterpart. Same two-model executor-as-first-
  // argument shape as apply_adaround's own binding above (candidates are
  // processed independently, so `executor` is invoked once, up front); a
  // BOUNDED GRID SEARCH, not an iterative Adam optimization, unlike
  // apply_adaround's own. See ApplyOmniquant in omniquant_entry.h for the
  // full scope and onnxsim/omniquant.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_omniquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_clip_steps, int64_t num_alpha_steps,
         double min_clip_ratio) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyOmniquant(
            float_model, quantized_model, *executor, calibration_data,
            num_clip_steps, num_alpha_steps, min_clip_ratio);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_clip_steps"_a = 20, "num_alpha_steps"_a = 20,
      "min_clip_ratio"_a = 0.5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AffineQuant (Ma et al., 2024, ICLR): OmniQuant's own LWC plus a
  // block-diagonal Learnable Equivalent Transformation (a per-block
  // orthogonal rotation on top of OmniQuant's own diagonal scale/shift).
  // Same shape as apply_omniquant's own binding above, plus
  // `affine_block_size`. See ApplyAffinequant in affinequant_entry.h for
  // the full scope and onnxsim/affinequant.py for the technique this
  // ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_affinequant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_clip_steps, int64_t num_alpha_steps, double min_clip_ratio,
         int64_t affine_block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyAffinequant(
            float_model, quantized_model, *executor, calibration_data,
            num_clip_steps, num_alpha_steps, min_clip_ratio, affine_block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_clip_steps"_a = 20, "num_alpha_steps"_a = 20,
      "min_clip_ratio"_a = 0.5, "affine_block_size"_a = 8);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // BRECQ (Li et al., 2021, ICLR): jointly optimizes every
  // quantize_weight_only_int4-quantized MatMul/Gemm layer inside a
  // caller-delimited block against the block's own final output
  // reconstruction error, Fisher-diagonal weighted -- extends
  // apply_adaround's own rectified-sigmoid relaxation and Adam loop to a
  // jointly optimized block of layers. Same two-model executor-as-first-
  // argument shape as apply_adaround's own binding above; `blocks` is a
  // list of `(block_input_name, block_output_name)` string pairs;
  // `beta_start`/`beta_end` are the two ends of apply_brecq's own
  // `beta_range` tuple, split the same way apply_adaround's own binding
  // splits it. See ApplyBrecq in brecq_entry.h for the full scope and
  // onnxsim/brecq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_brecq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::pair<std::string, std::string>> blocks,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_iterations, double learning_rate, double reg_param,
         double warm_start, double beta_start, double beta_end,
         double fisher_eps) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyBrecq(float_model, quantized_model, *executor, blocks,
                       calibration_data, num_iterations, learning_rate,
                       reg_param, warm_start, beta_start, beta_end, fisher_eps);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "blocks"_a, "calibration_data"_a, "num_iterations"_a = 300,
      "learning_rate"_a = 0.1, "reg_param"_a = 0.01, "warm_start"_a = 0.2,
      "beta_start"_a = 20.0, "beta_end"_a = 2.0, "fisher_eps"_a = 1e-3);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Sensitivity-based mixed-precision weight quantization: quantizes
  // every matched MatMul/vanilla-Gemm layer to block-wise INT8 or INT4,
  // chosen per layer from a calibration-driven Hessian-diagonal (or
  // full-Hessian) sensitivity score, budgeted so the top
  // `high_bits_fraction` most-sensitive layers get INT8. Single-model
  // executor-as-first-argument shape (same `calibration_data` crossing
  // convention as apply_llm_int8's own binding above -- see that binding
  // for the shape). See ApplyMixedPrecisionQuantization in
  // mixed_precision_entry.h for the full scope and
  // onnxsim/mixed_precision.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_mixed_precision_quantization",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double high_bits_fraction, int64_t block_size,
         const std::string& sensitivity_metric) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyMixedPrecisionQuantization(
            model, *executor, calibration_data, high_bits_fraction, block_size,
            sensitivity_metric);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "high_bits_fraction"_a = 0.2, "block_size"_a = 32,
      "sensitivity_metric"_a = "hessian_diag");
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QoQ's SmoothAttention: migrates Key's per-channel quantization
  // difficulty into Query (which stays float) for every decomposed
  // attention subgraph (MatMul(Q,Kt) -> [Mul/Div] -> [Add] -> Softmax ->
  // MatMul(_,V)) -- a provably-lossless calibrated scale migration, no
  // quantization happens here at all. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_llm_int8's own
  // binding above. See ApplySmoothAttention in smooth_attention_entry.h
  // for the full scope and onnxsim/qoq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_smooth_attention",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplySmoothAttention(model, *executor, calibration_data, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "epsilon"_a = 1e-5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // PTQ4ViT's twin uniform quantization: splits a matched Softmax/GELU
  // output's value range at a calibration-searched threshold into two
  // independently-quantized sub-ranges, doubling usable resolution where
  // the real distribution concentrates its mass. Same executor-as-first-
  // argument, `calibration_data` crossing convention as apply_llm_int8's
  // own binding above. See ApplyPtq4Vit in ptq4vit_entry.h for the full
  // scope and onnxsim/ptq4vit.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_ptq4vit_quantization",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t n_levels) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyPtq4Vit(model, *executor, calibration_data, n_levels);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "n_levels"_a = 256);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AutoRound (Cheng et al., 2023): jointly optimizes AdaRound's own
  // rectified-sigmoid rounding relaxation AND a second, per-(output
  // channel, block) clip-ratio parameter that lets the effective scale
  // move during the same Adam optimization -- closing the one gap between
  // apply_adaround's own binding above (fixed-scale, rounding-only) and
  // AutoRound proper. Same two-model executor-as-first-argument shape as
  // apply_adaround's own binding above (candidates are processed
  // independently, so `executor` is invoked once, up front);
  // `beta_start`/`beta_end` and `clip_ratio_min`/`clip_ratio_max` are the
  // two ends of apply_autoround's own `beta_range`/`clip_ratio_range`
  // tuples, split the same way apply_adaround's own binding splits
  // `beta_range`. Always runs AdaRound's own fixed-scale optimization too
  // and keeps whichever candidate has the lower measured reconstruction
  // error, so a layer's scale is only ever rewritten when doing so actually
  // helps. See ApplyAutoround in autoround_entry.h for the full scope and
  // onnxsim/autoround.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_autoround",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_iterations, double learning_rate,
         double clip_learning_rate, double reg_param, double warm_start,
         double beta_start, double beta_end, double clip_ratio_min,
         double clip_ratio_max) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyAutoround(
            float_model, quantized_model, *executor, calibration_data,
            num_iterations, learning_rate, clip_learning_rate, reg_param,
            warm_start, beta_start, beta_end, clip_ratio_min, clip_ratio_max);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_iterations"_a = 300, "learning_rate"_a = 0.1,
      "clip_learning_rate"_a = 0.03, "reg_param"_a = 0.01, "warm_start"_a = 0.2,
      "beta_start"_a = 20.0, "beta_end"_a = 2.0, "clip_ratio_min"_a = 0.5,
      "clip_ratio_max"_a = 1.5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // FlexRound (Lee et al., 2023, ICML): "learnable-division rounding" --
  // the fourth onnxsim-native PTQ technique alongside apply_adaround's own
  // binding above, apply_gptq's, and apply_awq's, each pulling a different
  // lever on the same target scheme. Reparametrizes the divisor itself
  // (`S = scale * S2 * s3`, both `S2`/`s3` learnable and log-space
  // parametrized) rather than AdaRound's own additive rounding
  // perturbation. Same two-model executor-as-first-argument shape as
  // apply_adaround's own binding above (candidates are processed
  // independently, so `executor` is invoked once, up front); never
  // rewrites a scale initializer (unlike apply_autoround's own binding
  // above), only the matched layer's own codes. See ApplyFlexround in
  // flexround_entry.h for the full scope and onnxsim/flexround.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_flexround",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_iterations, double learning_rate,
         double log_clip) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyFlexround(
            float_model, quantized_model, *executor, calibration_data,
            num_iterations, learning_rate, log_clip);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_iterations"_a = 300, "learning_rate"_a = 0.05,
      "log_clip"_a = 4.0);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // FOEM ("First-Order Error Matters", 2025): extends apply_gptq's own
  // binding above with an additional first-order-drift compensation term
  // -- alongside GPTQ's own Hessian-compensated rounding error, also
  // charges forward a damped fraction of how far each column's own
  // pre-quantization value has already drifted from the weight's true
  // original column, due to every earlier column's own forward
  // propagation. `foem_beta == 0.0` recovers plain GPTQ exactly. Same
  // two-model executor-as-first-argument shape as apply_gptq's own binding
  // above; `percdamp`/`proc_block_size` mirror apply_gptq's own parameters
  // of the same names and defaults. See ApplyFoem in foem_entry.h for the
  // full scope and onnxsim/foem.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_foem",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double percdamp, int64_t proc_block_size,
         double foem_beta) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyFoem(float_model, quantized_model, *executor, calibration_data,
                      percdamp, proc_block_size, foem_beta);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "percdamp"_a = 0.01, "proc_block_size"_a = 128,
      "foem_beta"_a = 0.005);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // SliM-LLM (Huang, Shao, Dong, Luo, Qiao et al., 2024): salience-driven
  // mixed-precision quantization picking a bit-width per GROUP within a
  // layer's own weight (rather than per whole layer). Single-model,
  // node-inserting rewrite (unlike apply_gptq's own binding above, there
  // is no separate "already quantized" model -- this pass builds and
  // inserts its own DequantizeLinear-based dequantization subgraph),
  // same executor-as-first-argument, `calibration_data` crossing
  // convention as apply_llm_int8's own binding. See ApplySlimLlm in
  // slim_llm_entry.h for the full scope and onnxsim/slim_llm.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_slim_llm",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double target_bits, int64_t low_bits, int64_t high_bits,
         int64_t group_size, double percdamp) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplySlimLlm(model, *executor, calibration_data, target_bits,
                         low_bits, high_bits, group_size, percdamp);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "target_bits"_a = 3.0, "low_bits"_a = 2, "high_bits"_a = 4,
      "group_size"_a = 32, "percdamp"_a = 0.01);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // MoEQuant (Hu, Chen et al., 2025): Expert-Balanced Self-Sampling (EBSS)
  // plus Affinity-Guided Quantization (AGQ) calibration methodology for
  // com.microsoft::MoE nodes' per-expert weights, reusing apply_gptq's own
  // column-update machinery as-is via a precomputed, per-expert Hessian --
  // NOT a new quantization algorithm (see this module's own docstring).
  // Single-model, node-structure-preserving (only the matched MoE node's
  // own fc1/fc2 weight initializers are rewritten in place -- simulated
  // ("fake") quantization, same graph/dtype/shape as the input). Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_llm_int8's own binding. See ApplyMoequant in moequant_entry.h
  // for the full scope (including its own two accepted numerical
  // divergences) and onnxsim/moequant.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_moequant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t quant_block_size, double percdamp, int64_t proc_block_size,
         bool ebss, uint64_t seed) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyMoequant(model, *executor, calibration_data, quant_block_size,
                          percdamp, proc_block_size, ebss, seed);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "quant_block_size"_a = 32, "percdamp"_a = 0.01, "proc_block_size"_a = 128,
      "ebss"_a = true, "seed"_a = 0);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Qronos: a sequential, whole-model generalization of apply_gptq that
  // additionally accounts for the error already baked into a layer's
  // activations because upstream layers were quantized first, not just
  // this layer's own rounding -- processes layers in the float model's
  // own node order, re-probing the progressively-corrected quantized
  // model before each subsequent layer (so `executor` is invoked once
  // per matched layer here, not once up front for all of them like every
  // other calibration-driven binding above). Same two-model
  // executor-as-first-argument shape as apply_gptq's own binding above;
  // `calibration_data` (List[Dict[str, onnx.TensorProto]]) is keyed to
  // the float model's own graph inputs. See ApplyQronos in
  // qronos_entry.h for the full scope and onnxsim/qronos.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_qronos",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double percdamp, int64_t proc_block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result =
            ApplyQronos(float_model, quantized_model, *executor,
                        calibration_data, percdamp, proc_block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "percdamp"_a = 0.01, "proc_block_size"_a = 128);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // TesseraQ: "Progressive Adaptive Rounding" (PAR) -- an AdaRound-style
  // rectified-sigmoid rounding relaxation, optimized by a hand-rolled
  // Adam loop jointly with each weight block's own dequantization scale
  // (in log-space), with a coarse-to-fine element-by-element hardening
  // schedule across `par_rounds` rounds. Same two-model
  // executor-as-first-argument shape as apply_gptq's own binding above
  // (candidates are processed independently, so `executor` is invoked
  // once, up front, the same as apply_gptq's own); `calibration_data`
  // (List[Dict[str, onnx.TensorProto]]) is keyed to the float model's own
  // graph inputs. See ApplyTesseraq in tesseraq_entry.h for the full
  // scope (including its own accepted numerical scope -- an iterative
  // optimization, not a closed-form computation) and
  // onnxsim/tesseraq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_tesseraq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& float_model_bytes, const py::bytes& quantized_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_bits, int64_t num_iterations, int64_t par_rounds,
         double learning_rate, double scale_learning_rate, double reg_param,
         double warm_start, double beta_start, double beta_end) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_bytes.c_str(),
                            quantized_bytes.size());
        const auto result = ApplyTesseraq(
            float_model, quantized_model, *executor, calibration_data, num_bits,
            num_iterations, par_rounds, learning_rate, scale_learning_rate,
            reg_param, warm_start, beta_start, beta_end);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "float_model_bytes"_a, "quantized_model_bytes"_a,
      "calibration_data"_a, "num_bits"_a = 4, "num_iterations"_a = 400,
      "par_rounds"_a = 4, "learning_rate"_a = 0.1,
      "scale_learning_rate"_a = 0.01, "reg_param"_a = 0.01,
      "warm_start"_a = 0.2, "beta_start"_a = 20.0, "beta_end"_a = 2.0);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QuaRot+GPTQ (Ashkboos et al., 2024): the real QuaRot paper's optional,
  // tighter weight quantizer -- rotates every matched MatMul/vanilla-Gemm
  // node's activation by a fresh per-layer random orthogonal matrix and
  // quantizes both operands to INT4, the weight via ApplyGptq's own
  // Hessian-compensated column algorithm (evaluated in the rotated
  // activation space) instead of round-to-nearest. Unlike apply_gptq/
  // apply_awq's own two-model bindings above, there is only one model here
  // (this pass derives its own rotation and quantizes from scratch, like
  // ApplyQuarot's own data-free pass); `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) is keyed to that single model's own graph inputs.
  // See ApplyQuarotGptq in quarot_gptq_entry.h for the full scope and
  // onnxsim/quarot.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_quarot_gptq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         uint64_t seed, int64_t block_size, double percdamp,
         int64_t proc_block_size, float epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyQuarotGptq(model, *executor, calibration_data, seed,
                            block_size, percdamp, proc_block_size, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "seed"_a = 0,
      "block_size"_a = 32, "percdamp"_a = 0.01, "proc_block_size"_a = 128,
      "epsilon"_a = 1e-12);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // GPTVQ (Van Baalen et al., 2024): a genuine combination of
  // apply_gptq's own sequential, Hessian-compensated correction with a
  // k-means-fit vector codebook -- small groups of consecutive
  // input-channel columns of every matched MatMul/vanilla-Gemm node's
  // constant 2-D FLOAT32 weight are jointly quantized against the
  // codebook, then each group's resulting per-column residual is
  // propagated into every not-yet-quantized column exactly like
  // apply_gptq's own per-column correction. Rewires only the matched
  // node's weight input (Gather+Reshape[+Transpose]); the node itself,
  // including any bias, is left otherwise unchanged. Same
  // executor-as-first-argument, `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) crossing convention, and `skip_names` (List[str])
  // crossing convention as apply_imatrix_quantization's own binding
  // above. See ApplyGptvq in gptvq_entry.h for the full scope (including
  // its own permanent RNG divergence from the Python reference for the
  // k-means codebook fit) and onnxsim/gptvq.py for the technique this
  // ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gptvq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         uint64_t seed, int64_t vector_dim, int64_t num_centroids,
         int64_t num_iterations, double percdamp,
         const std::vector<std::string>& skip_names) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const std::unordered_set<std::string> skip_names_set(skip_names.begin(),
                                                             skip_names.end());
        const auto result =
            ApplyGptvq(model, *executor, calibration_data, seed, vector_dim,
                       num_centroids, num_iterations, percdamp, skip_names_set);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "seed"_a = 0,
      "vector_dim"_a = 2, "num_centroids"_a = 256, "num_iterations"_a = 10,
      "percdamp"_a = 0.01, "skip_names"_a = std::vector<std::string>());
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // SmoothQuant migration (Xiao et al., 2022): rescales every matched
  // MatMul/vanilla-Gemm node's constant 2-D FLOAT32 weight columns by the
  // per-channel migration scale `s` in place and inserts a `Mul` node
  // dividing that layer's activation input by the same `s` -- a lossless
  // pre-conditioning transform ahead of a separate W8A8 quantizer, never a
  // quantization scheme itself. Same executor-as-first-argument,
  // `calibration_data` (List[Dict[str, onnx.TensorProto]]) crossing
  // convention as apply_imatrix_quantization's own binding above. See
  // ApplySmoothQuant in smoothquant_entry.h for the full scope and
  // onnxsim/smoothquant.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_smoothquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double alpha, double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplySmoothQuant(model, *executor, calibration_data,
                                             alpha, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "alpha"_a = 0.5,
      "epsilon"_a = 1e-5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // FPTQ (Li, Zhang, Li, Yao, Zhang, Chu, Sun, Du and Xie, 2023): shares
  // apply_smoothquant's own binding above core mechanism (rescale a
  // matched MatMul/vanilla-Gemm layer's weight columns by a per-channel
  // migration scale `s`, dividing the activation by the same `s` via a
  // new `Mul` node), replacing SmoothQuant's own power-law scale with
  // FPTQ's own logarithmic-equalization scale on layers whose activation
  // has an outlier channel at least `outlier_ratio_threshold` times the
  // layer's own typical (geometric-mean) channel scale. Same
  // executor-as-first-argument, `calibration_data` crossing convention as
  // apply_smoothquant's own binding above. See ApplyFptq in fptq_entry.h
  // for the full scope and onnxsim/fptq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_fptq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double alpha, double outlier_ratio_threshold,
         double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyFptq(model, *executor, calibration_data, alpha,
                                      outlier_ratio_threshold, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "alpha"_a = 0.5,
      "outlier_ratio_threshold"_a = 10.0, "epsilon"_a = 1e-5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // EasyQuant (Wu, Judd, Isaev, Micikevicius, 2020): plain W8A8 (INT8
  // weight + INT8 activation) quantizer using coordinate-descent GRID
  // SEARCH (no gradients/Hessian) -- alternates a per-output-channel
  // weight-scale grid search (independent per channel, minimizing that
  // channel's own quantized-output MSE) with a single activation-scale
  // grid search (maximizing the whole layer's quantized-output cosine
  // similarity) for `num_iterations` rounds, then folds the optimized
  // weight into a new float32 initializer and inserts
  // `Div`/`Round`/`Clip`/`Mul` nodes quantize-dequantize round-tripping
  // the activation input. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_smoothquant's own
  // binding above (single-model, unlike apply_gptq's own two-model
  // shape). See ApplyEasyquant in easyquant_entry.h for the full scope
  // and onnxsim/easyquant.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_easyquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t num_iterations, int64_t num_candidates,
         double search_span) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyEasyquant(model, *executor, calibration_data, num_iterations,
                           num_candidates, search_span);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "num_iterations"_a = 3, "num_candidates"_a = 21, "search_span"_a = 0.5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // RPTQ (Yuan et al., 2023): clusters every matched MatMul/vanilla-Gemm
  // node's input channels by their own calibration abs-max (a plain
  // Lloyd's-algorithm k-means) and permutes them -- plus the weight's
  // matching K-axis rows -- so same-cluster channels sit contiguously,
  // via a new `Gather` before the node. An exact reordering, not a
  // quantization -- see ApplyRptqReorder in rptq_entry.h for the full
  // scope and onnxsim/rptq.py for the technique this ports. Returned as a
  // (model_bytes, layers) pair rather than bare bytes, mirroring
  // apply_embedding_vocab_pruning's own precedent above: `layers` is a
  // list of (x_name, w_name, gather_output, permutation, cluster_bounds)
  // tuples, reconstructed into the real, public
  // `onnxsim.rptq.RptqLayerInfo` dict by the Python wrapper
  // (onnx_simplifier.py's own apply_rptq_reorder_cpp).
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_rptq_reorder",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t seed, int64_t num_clusters)
          -> std::tuple<
              py::bytes,
              std::vector<std::tuple<
                  std::string, std::string, std::string, std::vector<int64_t>,
                  std::vector<std::pair<int64_t, int64_t>>>>> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyRptqReorder(model, *executor, calibration_data,
                                             seed, num_clusters);
        std::string out;
        result.model.SerializeToString(&out);
        std::vector<std::tuple<std::string, std::string, std::string,
                               std::vector<int64_t>,
                               std::vector<std::pair<int64_t, int64_t>>>>
            layers;
        layers.reserve(result.layers.size());
        for (const auto& l : result.layers) {
          layers.emplace_back(l.x_name, l.w_name, l.gather_output,
                              l.permutation, l.cluster_bounds);
        }
        return {py::bytes(out.data(), out.size()), layers};
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "seed"_a = 0,
      "num_clusters"_a = 4);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LoRA adapter injection: splices a trainable low-rank `X @ A @ B` branch
  // around every eligible MatMul/Gemm/Conv weight (2-D float32
  // MatMul/Gemm; 4-D float32, kernel_shape==[1,1], group==1 Conv), leaving
  // the base weight itself untouched -- `A` is Kaiming-normal-initialized
  // from `seed`, `B` starts at all zeros, so injection is a numeric no-op
  // until trained. Data-free, single-model, no executor needed. See
  // InjectLora in lora_entry.h for the full scope (including its one
  // documented divergence: the `A` initializer's RNG stream does not
  // reproduce numpy's PCG64 bit-for-bit for the same seed -- immaterial
  // here since `B` is always zero) and onnxsim/lora.py's inject_lora for
  // the technique this ports. Returned as a (model_bytes, targets) pair,
  // mirroring apply_rptq_reorder's own precedent just above: `targets` is
  // a list of (weight_name, node_output, op_type, lora_a_name,
  // lora_b_name, rank, alpha) tuples, reconstructed into the real, public
  // `onnxsim.lora.LoraTarget`/`LoraAdapter` by the Python wrapper
  // (onnx_simplifier.py's own inject_lora_cpp).
  m.def(
      "inject_lora",
      [](const py::bytes& model_proto_bytes, int64_t rank,
         std::optional<double> alpha, std::vector<std::string> target_op_types,
         bool restrict_target_names, std::vector<std::string> target_names,
         int64_t seed)
          -> std::tuple<py::bytes,
                        std::vector<std::tuple<
                            std::string, std::string, std::string, std::string,
                            std::string, int64_t, std::optional<double>>>> {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        InjectLoraOptions options;
        options.rank = rank;
        options.has_alpha = alpha.has_value();
        options.alpha = alpha.has_value() ? static_cast<float>(*alpha) : 0.0f;
        options.target_op_types = std::move(target_op_types);
        options.restrict_target_names = restrict_target_names;
        options.target_names = std::move(target_names);
        options.seed = static_cast<uint64_t>(seed);
        const auto result = InjectLora(model, options);
        std::string out;
        result.model.SerializeToString(&out);
        std::vector<
            std::tuple<std::string, std::string, std::string, std::string,
                       std::string, int64_t, std::optional<double>>>
            targets;
        targets.reserve(result.adapter.targets.size());
        for (const LoraTarget& t : result.adapter.targets) {
          targets.emplace_back(
              t.weight_name, t.node_output, t.op_type, t.lora_a_name,
              t.lora_b_name, t.rank,
              t.has_alpha ? std::optional<double>(t.alpha) : std::nullopt);
        }
        return {py::bytes(out.data(), out.size()), targets};
      },
      "model_bytes"_a, "rank"_a = 8, "alpha"_a = std::nullopt,
      "target_op_types"_a = std::vector<std::string>{"MatMul", "Gemm", "Conv"},
      "restrict_target_names"_a = false,
      "target_names"_a = std::vector<std::string>{}, "seed"_a = 0);

  // SpinQuant (Liu et al., 2024), "R1-only" variant: fits a single dense
  // [K, K] rotation per matched MatMul/vanilla-Gemm layer as the
  // eigenvector basis of that layer's own calibration-activation
  // covariance (a closed-form substitute for SpinQuant's own learned,
  // Cayley-manifold-optimized rotation), conjugates the weight by it, then
  // block-wise INT4-quantizes the result. See ApplySpinquant in
  // spinquant_entry.h for the full scope and onnxsim/spinquant.py for the
  // technique this ports. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_spqr's own binding
  // above.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_spinquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t block_size) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplySpinquant(model, *executor, calibration_data, block_size);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "block_size"_a = 32);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // ParoQuant (Liang et al., 2025): combines a SmoothQuant-style per-
  // -channel scale with many independent, cheap 2x2 (pairwise, Givens)
  // rotations on fixed adjacent-channel pairs within each quantization
  // block -- each angle grid-searched against its own block's INT4
  // reconstruction error -- instead of onnxsim.apply_spinquant's single
  // dense rotation, then block-wise INT4-quantizes the result. See
  // ApplyParoquant in paroquant_entry.h for the full scope and
  // onnxsim/paroquant.py for the technique this ports. Same executor-as-
  // -first-argument, `calibration_data` crossing convention as
  // apply_spqr's own binding above.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_paroquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t block_size, double alpha, int64_t num_angle_steps,
         double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyParoquant(model, *executor, calibration_data, block_size,
                           alpha, num_angle_steps, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "block_size"_a = 32,
      "alpha"_a = 0.5, "num_angle_steps"_a = 9, "epsilon"_a = 1e-5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // DuQuant (Lin et al., 2024): a calibration-ranked permutation that
  // redistributes each matched MatMul/vanilla-Gemm layer's worst outlier
  // input channels one-per-block across the quantization grouping,
  // composed with an independent Haar-random orthogonal rotation applied
  // within each block, then INT4 round-to-nearest quantization of BOTH the
  // weight (offline, block-wise) and the activation (data-free, per-token,
  // at graph-run time). Same executor-as-first-argument, `calibration_data`
  // crossing convention as apply_spqr's own binding above. See ApplyDuquant
  // in duquant_entry.h for the full scope (including its own accepted
  // per-node RNG divergence) and onnxsim/duquant.py for the technique this
  // ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_duquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t seed, int64_t block_size, double outlier_fraction,
         double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyDuquant(model, *executor, calibration_data, seed, block_size,
                         outlier_fraction, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "seed"_a = 0,
      "block_size"_a = 32, "outlier_fraction"_a = 0.05, "epsilon"_a = 1e-12);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Outlier Suppression+ (Wei et al., 2023): per-channel shifting ahead
  // of SmoothQuant's own per-channel scale -- recenters each activation
  // channel around zero, rescales the weight columns in place, inserts a
  // `Sub`+`Mul` pair before the layer and an `Add` after it restoring the
  // shift's constant contribution -- a lossless pre-conditioning
  // transform ahead of a separate W8A8 quantizer, never a quantization
  // scheme itself. Same executor-as-first-argument, `calibration_data`
  // (List[Dict[str, onnx.TensorProto]]) crossing convention as
  // apply_outlier_suppression's own binding above. See
  // ApplyOutlierSuppressionPlus in outlier_suppression_plus_entry.h for
  // the full scope and onnxsim/outlier_suppression_plus.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_outlier_suppression_plus",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double alpha, double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyOutlierSuppressionPlus(
            model, *executor, calibration_data, alpha, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "alpha"_a = 0.5,
      "epsilon"_a = 1e-5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // MoE expert-intermediate-channel pruning: removes intermediate
  // (`inter_size`) channels from every expert of a matched
  // `com.microsoft::MoE` node at once -- real structural pruning, data-free.
  // Whole-expert pruning (shrinking `num_experts` itself) is NOT ported --
  // see ApplyMoeExpertChannelPruning in structured_pruning_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QMoE expert-channel pruning: removes intermediate (inter_size) channels
  // from every expert of a matched com.microsoft::QMoE node -- the
  // quantized-weight counterpart of apply_structured_pruning. See
  // ApplyQMoEExpertChannelPruning in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // MoE whole-expert pruning: the calibration-driven complementary
  // technique to apply_moe_expert_channel_pruning above -- drops whole
  // experts (shrinks `num_experts`) ranked by mean router gate weight over
  // `calibration_data`, same executor-as-first-argument shape as
  // apply_structured_wanda_pruning (this is likewise NOT purely
  // data-free -- it runs `model_bytes` over `calibration_data` to capture
  // router activations). See ApplyMoeWholeExpertPruning in
  // structured_pruning_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QMoE whole-expert pruning: the quantized-weight counterpart of
  // apply_moe_whole_expert_pruning above -- same calibration-driven
  // ranking (mean router gate weight, the exact same
  // MoeRouterGateCalibrationStats helper -- `router_probs` is QMoE's own
  // second input too, upstream of and oblivious to its quantized
  // fc1/fc2), same executor-as-first-argument shape. See
  // ApplyQMoEWholeExpertPruning in structured_pruning_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // The importance-ranked variant -- see EmbeddingVocabPruningResult/
  // ApplyEmbeddingVocabMagnitudePruning in structured_pruning_entry.h.
  // Same return shape as apply_embedding_vocab_pruning above.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Any-Precision LLM (Park et al., 2024, ICML 2024): nested bit-plane
  // weight-only quantization, one quantization pass serving any bit-width
  // up to max_bits. See ApplyAnyPrecisionLlm in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QuaRot (Ashkboos et al., 2024): rotation preprocessing plus INT4
  // round-to-nearest quantization of both the weight and the activation of
  // every MatMul/vanilla-Gemm layer. Data-free. See ApplyQuarot in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's IQ4_NL: fixed 16-entry non-uniform-codebook weight-only 4-bit
  // quantization, one scale per 32-element block. Data-free. See ApplyIQ4NL
  // in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's legacy GGUF Q4_0/Q4_1 block formats: weight-only 4-bit
  // quantization, one plain 32-element block per scale(/min). Data-free.
  // See ApplyGgufQ4_0/ApplyGgufQ4_1 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's legacy GGUF Q5_0/Q5_1 block formats: weight-only 5-bit
  // quantization, one plain 32-element block per scale(/min) -- the same
  // scheme as Q4_0/Q4_1 above, one bit wider. Data-free. See
  // ApplyGgufQ5_0/ApplyGgufQ5_1 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q5_0_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ5_0(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q5_1_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ5_1(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's GGUF Q8_0 block format: weight-only signed 8-bit
  // quantization, one plain 32-element block sharing a single fp16 scale
  // (no bias, no min). Data-free. See ApplyGgufQ8_0 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q8_0_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ8_0(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's GGUF Q2_K K-quant format: weight-only 2-bit quantization,
  // a 256-element super-block split into 16 sub-blocks of 16, each with
  // its own asymmetric (scale, min) pair re-quantized to 4-bit codes.
  // Data-free. See ApplyGgufQ2K in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q2_k_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ2K(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's GGUF Q3_K K-quant format: weight-only 3-bit quantization,
  // a 256-element super-block split into 16 sub-blocks of 16, each with
  // a 6-bit scale code times one shared super-block scale, times the
  // format's own asymmetric 3-bit element code. Data-free. See
  // ApplyGgufQ3K in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q3_k_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ3K(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's GGUF Q4_K K-quant format: weight-only 4-bit quantization,
  // a 256-element super-block split into 8 sub-blocks of 32, each with
  // its own asymmetric (scale, min) pair re-quantized to 6-bit codes.
  // Data-free. See ApplyGgufQ4K in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q4_k_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ4K(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's GGUF Q5_K K-quant format: identical to Q4_K above except
  // a 5-bit (vs 4-bit) element code. Data-free. See ApplyGgufQ5K in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q5_k_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ5K(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // BitNet b1.58's published absmean ternary weight quantization, as
  // shipped by llama.cpp's GGUF TQ1_0/TQ2_0 tensor types: weight-only,
  // one shared {-1, 0, +1} scale per 256-element block. Data-free. See
  // ApplyGgufTernaryQuant in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // FP6-LLM's E3M2 6-bit floating-point weight-only quantization, one
  // scale per 64-element block. Data-free. See ApplyFp6Llm in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // llama.cpp's Q6_K K-quant format: weight-only, one 8-bit sub-block
  // scale (times a shared float16 super-block scale) per 16-element
  // sub-block of a 256-element super-block. Data-free. See ApplyGgufQ6K
  // in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_gguf_q6_k_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyGgufQ6K(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AngelSlim's LeptoQuant: outlier-aware block FP8 weight quantization,
  // one FLOAT8E4M3FN scale per 128x128 tile, grid-searched over an
  // outlier fraction. Data-free. See ApplyLeptoquant in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_leptoquant",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyLeptoquant(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // bitsandbytes' NF4: weight-only quantization with a fixed 16-value
  // non-uniform codebook, one scale per 64-element (output-channel,
  // K-block) group. Data-free. See ApplyNF4 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_weight_only_nf4",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyNF4(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // IF4: per (output-channel, 16-element K-block) choice between a plain
  // INT4 grid and MXFP4's own E2M1 codebook, whichever reconstructs that
  // block with lower MSE. Data-free. See ApplyIF4 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_weight_only_if4",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyIF4(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // NVIDIA's NVFP4: shares MXFP4's E2M1 codebook, with a two-level
  // (per-tensor global scale, E4M3-rounded per-block scale) rule.
  // Data-free. See ApplyNVFP4Quantization in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_weight_only_nvfp4",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyNVFP4Quantization(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // DeepSeek-V3-style fine-grained block FP8 weight quantization: one
  // real FLOAT8E4M3FN round trip per 128x128 tile. Weight side only --
  // see ApplyDeepSeekFp8 in onnxsim.h for the activation-quantization
  // scope-narrowing note. Data-free.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_deepseek_fp8",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyDeepSeekFp8(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // K-means per-layer codebook weight quantization (Han et al., 2015,
  // "Deep Compression"): a 16-centroid codebook fit per layer via
  // Lloyd's algorithm. Data-free. See ApplyKMeansQuantization in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_kmeans_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyKMeansQuantization(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // HQQ (Half-Quadratic Quantization): asymmetric affine INT4 with an
  // IRLS-refined zero-point per (output-channel, 32-element K-block)
  // group. Data-free. See ApplyHQQ in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_hqq",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyHQQ(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // I-BERT's own i-GELU polynomial approximation of Erf: a
  // nonlinear-activation rewrite, not a weight quantizer -- matches any
  // standalone Erf node. Data-free. See ApplyIBertGelu in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_ibert_gelu",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyIBertGelu(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // I-BERT's own integer-friendly Softmax exp-approximation: matches any
  // standalone Softmax node. Data-free. See ApplyIBertSoftmax in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_ibert_softmax",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyIBertSoftmax(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AdpQ: calibration-free salient/non-salient weight split via a
  // median/MAD-based adaptive threshold. Data-free. See ApplyADPQ in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_adpq",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyADPQ(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // ICQuant: per-block single-outlier exact reconstruction plus a
  // symmetric 7-level-per-side grid for the rest. Data-free. See
  // ApplyICQuant in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_icquant",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyICQuant(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // OliVe: outlier-victim pair quantization. Data-free. See ApplyOlive
  // in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_olive",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyOlive(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // AQLM: additive/residual multi-codebook quantization via greedy
  // residual k-means. Data-free. See ApplyAQLM in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_aqlm",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyAQLM(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Drop-by-Drop: additive multi-bitwidth codebook quantization via 4
  // importance-weighted greedy residual k-means stages. Data-free. See
  // ApplyDropByDrop in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_drop_by_drop",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyDropByDrop(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LO-BCQ: block-clustered quantization -- blocks are clustered by their
  // own [mean, std] feature vector, then each cluster fits its own small
  // codebook. Data-free. See ApplyLoBcq in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_lo_bcq",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyLoBcq(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // QuIP#: rotation-based incoherence processing plus E8-lattice vector
  // quantization, folded into a single replacement weight initializer.
  // Data-free. See ApplyQuipSharp in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_quip_sharp",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyQuipSharp(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Attention computation quantization: per-token dynamic INT8 for Q/K/V,
  // fixed-scale UINT8 for the Softmax output. Data-free. New graph nodes
  // (not a fold-to-initializer -- no constant weight is involved). See
  // ApplyAttentionQuantization in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_attention_quantization",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyAttentionQuantization(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // ZeroQuant (Yao et al., 2022): group-wise INT8 weight quantization paired
  // with per-token dynamic INT8 activation quantization, executed as a real
  // int8 x int8 MatMulInteger. Data-free. New graph nodes (not a
  // fold-to-initializer -- the activation's own per-token scale is a
  // runtime value). See ApplyZeroQuant in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_zeroquant",
      [](const py::bytes& model_proto_bytes, int64_t block_size,
         float epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyZeroQuant(model, block_size, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "block_size"_a = 32, "epsilon"_a = 1e-12f);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // IntactKV (Liu et al., 2024): splits a KV-cache stream's own fixed-length
  // leading pivot prefix into its own always-exact stream. Data-free.
  // Companion pass, not a quantizer -- see ApplyIntactKv in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_intactkv",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyIntactKv(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // KBVQ-MoE (Xu et al., 2026): KLT-shared-basis plus per-expert
  // bias-corrected vector quantization for a com.microsoft::MoE router
  // group's own experts. Data-free. See ApplyKbvqMoe in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_kbvq_moe",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyKbvqMoe(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LLM-FP4 (Liu et al., 2023): searched (exponent/mantissa split,
  // per-block real-valued scale) FP4 weight-only quantization. Data-free.
  // See QuantizeWeightOnlyLlmFp4 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_weight_only_llm_fp4",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = QuantizeWeightOnlyLlmFp4(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LLM-FP4 activation quantization, data-free per-token variant (Liu et
  // al., 2023): completes W4A4 for every quantize_weight_only_llm_fp4-
  // quantized layer by inserting a per-token, data-free FP4 quantize/
  // dequantize round-trip on that layer's own activation input, reusing
  // that same layer's own already-baked codebook. NOT the paper's own
  // per-channel-migration design -- see
  // ApplyLlmFp4ActivationQuantization in llm_fp4_activation_entry.h for
  // the full "Honesty note" and onnxsim/llm_fp4.py for the technique.
  // Data-free.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_llm_fp4_activation_quantization",
      [](const py::bytes& model_proto_bytes, double epsilon) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyLlmFp4ActivationQuantization(model, epsilon);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "epsilon"_a = 1e-12);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // LLM-FP4 activation quantization, calibrated per-tensor variant (Liu
  // et al., 2023): the paper's own quantizer half (the migration half is
  // the caller's job, via apply_smoothquant/apply_outlier_suppression
  // run first) -- fits one real-valued per-tensor scale from calibration
  // data and bakes it into the graph as a constant before inserting a
  // static FP4 quantize/dequantize round-trip. Same
  // executor-as-first-argument, `calibration_data` crossing convention
  // as apply_llm_int8's own binding above. See
  // ApplyLlmFp4ActivationQuantizationPerTensor in
  // llm_fp4_activation_entry.h for the full scope and
  // onnxsim/llm_fp4.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_llm_fp4_activation_quantization_per_tensor",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         std::optional<std::vector<double>> clip_ratios) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyLlmFp4ActivationQuantizationPerTensor(
            model, *executor, calibration_data, clip_ratios);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a,
      "clip_ratios"_a = std::nullopt);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Binary Weight-Activation PTQ (Song et al., 2025, ACL Findings), weight
  // side only (W(1+1)): binarizes every matched MatMul/vanilla-Gemm layer
  // to exactly 1 sign bit + 1 group-select bit/element via Hessian-
  // weighted two-scale binary EM. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_llm_int8's own
  // binding above. See ApplyBwaPtq in bwa_ptq_entry.h for the full scope
  // and onnxsim/bwa_ptq.py for the technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_bwa_ptq",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t group_size, int64_t max_em_iters) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyBwaPtq(model, *executor, calibration_data,
                                        group_size, max_em_iters);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "group_size"_a = 128,
      "max_em_iters"_a = 10);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Pruning-recovery fine-tuning: for every surviving MatMul/vanilla-Gemm
  // layer present (by node output name) in both `original_model` and
  // `pruned_model`, re-solves its weight (and bias) as a closed-form
  // ridge-regression fit against `original_model`'s own real
  // activations. Same two-model executor-as-first-argument shape as
  // apply_gptq's own binding above; `calibration_data` (List[Dict[str,
  // onnx.TensorProto]]) is keyed to `original_model`'s own graph inputs.
  // See ApplyPruningFinetune in finetune_entry.h for the full scope and
  // onnxsim/finetune.py for the technique this ports.
  m.def(
      "apply_pruning_finetune",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& original_model_bytes,
         const py::bytes& pruned_model_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         double reg_param) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto original_model;
        ParseProtoFromBytes(&original_model, original_model_bytes.c_str(),
                            original_model_bytes.size());
        ONNX_NAMESPACE::ModelProto pruned_model;
        ParseProtoFromBytes(&pruned_model, pruned_model_bytes.c_str(),
                            pruned_model_bytes.size());
        const auto result =
            ApplyPruningFinetune(original_model, pruned_model, *executor,
                                 calibration_data, reg_param);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "original_model_bytes"_a, "pruned_model_bytes"_a,
      "calibration_data"_a, "reg_param"_a = 1e-2);

  // QServe's QoQ quantization (Lin et al., 2024): progressive
  // (INT8-then-INT4) block-wise weight quantization. Data-free. See
  // ApplyQoq in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_qoq",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyQoq(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // D2Quant's Dual-Scale Quantizer (Yan et al., 2026): a per-column
  // auxiliary scale for SwiGLU/GLU down-projection weights, absorbed into
  // the paired up-projection's own weight. Data-free. See ApplyDsq in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_dsq",
      [](const py::bytes& model_proto_bytes) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplyDsq(model);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // DAQ (Delta-Aware Quantization): data-free, but takes two full model
  // byte buffers (a base and a fine-tuned checkpoint, matched by node
  // output name) rather than one -- see ApplyDaq in daq_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_daq",
      [](const py::bytes& base_model_bytes,
         const py::bytes& post_trained_model_bytes, const std::string& metric,
         const std::vector<std::string>& skip_names) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto base_model;
        ParseProtoFromBytes(&base_model, base_model_bytes.c_str(),
                            base_model_bytes.size());
        ONNX_NAMESPACE::ModelProto post_trained_model;
        ParseProtoFromBytes(&post_trained_model,
                            post_trained_model_bytes.c_str(),
                            post_trained_model_bytes.size());
        const std::unordered_set<std::string> skip_names_set(skip_names.begin(),
                                                             skip_names.end());
        const auto result =
            ApplyDaq(base_model, post_trained_model, metric, skip_names_set);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "base_model_bytes"_a, "post_trained_model_bytes"_a, "metric"_a = "cosine",
      "skip_names"_a = std::vector<std::string>{});
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // SVDQuant / Nunchaku (Li, Lin, Zhang, et al., 2024): an optional
  // SmoothQuant-style migration (ApplySmoothQuant, reused directly -- see
  // svdquant_entry.h's own top-of-file comment), then for every matched
  // MatMul/vanilla-Gemm layer a low-rank/residual split of the weight via
  // truncated SVD -- keeps the rank-`rank` dominant/outlier structure in a
  // full-precision low-rank branch and block-wise INT4-quantizes only the
  // (now much more uniform) residual. Same executor-as-first-argument,
  // `calibration_data` crossing convention as apply_spqr's own binding
  // above (only actually used when `smooth_alpha` has a value). See
  // ApplySvdquant in svdquant_entry.h for the full scope (including its own
  // accepted Jacobi-SVD divergence) and onnxsim/svdquant.py for the
  // technique this ports.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_svdquant",
      [](std::shared_ptr<PyModelExecutor> executor,
         const py::bytes& model_proto_bytes,
         std::vector<std::unordered_map<std::string, onnx::TensorProto>>
             calibration_data,
         int64_t rank, int64_t block_size,
         std::optional<double> smooth_alpha) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result = ApplySvdquant(model, *executor, calibration_data,
                                          rank, block_size, smooth_alpha);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "executor"_a, "model_bytes"_a, "calibration_data"_a, "rank"_a = 32,
      "block_size"_a = 32, "smooth_alpha"_a = 0.5);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Low-Rank Compensation (LoRC): data-free, but takes two full model byte
  // buffers (a float model and its own INT4-quantized counterpart, matched
  // by node output name) rather than one -- see ApplyLowRankCompensation
  // in low_rank_compensation_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "apply_low_rank_compensation",
      [](const py::bytes& float_model_bytes,
         const py::bytes& quantized_model_bytes, int64_t rank) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto float_model;
        ParseProtoFromBytes(&float_model, float_model_bytes.c_str(),
                            float_model_bytes.size());
        ONNX_NAMESPACE::ModelProto quantized_model;
        ParseProtoFromBytes(&quantized_model, quantized_model_bytes.c_str(),
                            quantized_model_bytes.size());
        const auto result =
            ApplyLowRankCompensation(float_model, quantized_model, rank);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "float_model_bytes"_a, "quantized_model_bytes"_a, "rank"_a = 8);
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Embedding-output binarization: data-free, single-model, no
  // ModelExecutor -- targets a whole graph OUTPUT declaration rather than a
  // matched node. `output_name` empty stands in for Python's own
  // `output_name=None` sentinel. See ApplyEmbeddingQuantizationBinary in
  // embedding_quantization_entry.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
  m.def(
      "quantize_embedding_binary",
      [](const py::bytes& model_proto_bytes,
         const std::string& output_name) -> py::bytes {
        InitEnv();
        ONNX_NAMESPACE::ModelProto model;
        ParseProtoFromBytes(&model, model_proto_bytes.c_str(),
                            model_proto_bytes.size());
        const auto result =
            ApplyEmbeddingQuantizationBinary(model, output_name);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out.data(), out.size());
      },
      "model_bytes"_a, "output_name"_a = "");
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes MatMul/Gemm into the
  // "QOperator" format (QLinearMatMul) rather than quantize_static's QDQ
  // format -- needs a calibrated range for both the activation and the
  // node's own output (see list_qoperator_quantizable_outputs) since
  // QLinearMatMul computes directly in int8, with no float intermediate --
  // see QuantizeQOperator in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_elementwise could quantize --
  // both operands and the output of every qualifying Add/Mul node -- see
  // ListQOperatorElementwiseQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes elementwise Add/Mul into ONNX
  // Runtime's "com.microsoft" QLinearAdd/QLinearMul contrib ops -- needs a
  // calibrated range for both operands and the node's own output (see
  // list_qoperator_elementwise_quantizable_tensors) since these compute
  // directly in int8, with no float intermediate -- see
  // QuantizeQOperatorElementwise in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_activation could quantize --
  // the input and output of every qualifying Sigmoid/LeakyRelu node -- see
  // ListQOperatorActivationQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes standalone Sigmoid/LeakyRelu
  // into ONNX Runtime's "com.microsoft" QLinearSigmoid/QLinearLeakyRelu
  // contrib ops -- needs a calibrated range for both the input and the
  // node's own output (see list_qoperator_activation_quantizable_tensors)
  // since these compute directly in int8, with no float intermediate -- see
  // QuantizeQOperatorActivation in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_concat could quantize -- every
  // input plus the output of every qualifying Concat node -- see
  // ListQOperatorConcatQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes Concat into ONNX Runtime's
  // "com.microsoft" QLinearConcat contrib op -- needs a calibrated range for
  // every input and the node's own output (see
  // list_qoperator_concat_quantizable_tensors) since this computes directly
  // in int8, with no float intermediate -- see QuantizeQOperatorConcat in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_softmax could quantize -- the
  // input and output of every qualifying Softmax node -- see
  // ListQOperatorSoftmaxQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes standalone Softmax into ONNX
  // Runtime's "com.microsoft" QLinearSoftmax contrib op -- needs a
  // calibrated range for both the input and the node's own output (see
  // list_qoperator_softmax_quantizable_tensors) since this computes
  // directly in int8, with no float intermediate -- see
  // QuantizeQOperatorSoftmax in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_pool could quantize -- the
  // input and output of every qualifying AveragePool/GlobalAveragePool node
  // -- see ListQOperatorPoolQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes standalone AveragePool/
  // GlobalAveragePool into ONNX Runtime's "com.microsoft"
  // QLinearAveragePool/QLinearGlobalAveragePool contrib ops -- needs a
  // calibrated range for both the input and the node's own output (see
  // list_qoperator_pool_quantizable_tensors) since these compute directly
  // in int8, with no float intermediate -- see QuantizeQOperatorPool in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_where could quantize -- both
  // operands and the output of every qualifying Where node -- see
  // ListQOperatorWhereQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes Where into ONNX Runtime's
  // "com.microsoft" QLinearWhere contrib op -- needs a calibrated range for
  // both operands and the node's own output (see
  // list_qoperator_where_quantizable_tensors) since this computes directly
  // in int8, with no float intermediate -- see QuantizeQOperatorWhere in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Lists the tensor names quantize_qoperator_gemm could quantize -- the
  // activation and output of every qualifying Gemm node -- see
  // ListQOperatorGemmQuantizableTensors in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Statically (calibration-based) quantizes Gemm into ONNX Runtime's
  // "com.microsoft" QGemm contrib op -- the fully-general analogue of
  // quantize_qoperator's QLinearMatMul rewrite (handles any transA/transB/
  // alpha) -- needs a calibrated range for the activation and the node's
  // own output (see list_qoperator_gemm_quantizable_tensors) since this
  // computes directly in int8, with no float intermediate -- see
  // QuantizeQOperatorGemm in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Converts every float32 weight (and, by default, every internal
  // activation) to float16 -- no calibration data needed, since float16 is
  // still a floating-point format, not an integer scheme. With
  // keep_io_types (the default true), the graph's own external input/output
  // types stay float32 via boundary Cast nodes. See QuantizeFp16 in
  // onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Converts every float32 weight (and, by default, every internal
  // activation) to bfloat16 -- the same calibration-free, whole-graph
  // conversion as quantize_fp16 above, just to a different narrow
  // floating-point format (bfloat16 keeps float32's full exponent range, so
  // there is no clamping concern). See QuantizeBf16 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

  // Converts every float32 weight (and, by default, every internal
  // activation) to an 8-bit floating-point format -- the same
  // calibration-free, whole-graph conversion as quantize_fp16/quantize_bf16
  // above, just to a much narrower floating-point format. `format` selects
  // "e4m3" (E4M3FN, the default) or "e5m2" (E5M2); both convert with
  // saturation (clamping) rather than producing an infinity/NaN for an
  // out-of-range magnitude. See QuantizeFp8 in onnxsim.h.
#ifdef ONNXSIM_HAS_NON_CORE_FEATURES
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
#endif  // ONNXSIM_HAS_NON_CORE_FEATURES

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
         // docs/bench/RESULTS_synthetic_decoder_oom.md for why this matters.
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
