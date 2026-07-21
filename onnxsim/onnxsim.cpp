#include "onnxsim.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <unordered_map>

#ifndef NO_BUILTIN_ORT
#include "onnxruntime/core/common/endian.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#endif
#include "contrib_schemas.h"
#include "onnx/common/file_utils.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/passes/logging.h"

struct Config {
  std::vector<std::string> optimizer_passes;
  // default value is max
  size_t tensor_size_threshold = -1;
};

Config config;

bool IsOfficialOp(const std::string& domain, const std::string& op) {
  if (domain != "ai.onnx" && domain != "ai.onnx.ml" && !domain.empty()) {
    return false;
  }
  // these experimental ops were in onnx default domain but are no
  // longer supported by onnx now.
  static std::set<std::string> experimental_ops = {"ATen",
                                                   "Affine",
                                                   "ConstantFill",
                                                   "Crop",
                                                   "DynamicSlice",
                                                   "GRUUnit",
                                                   "GivenTensorFill",
                                                   "ImageScaler",
                                                   "ParametricSoftplus",
                                                   "Scale",
                                                   "ScaledTanh"};
  return experimental_ops.find(op) == experimental_ops.end();
}

bool IsDeterministic(const std::string& domain, const std::string& op,
                     int opset_version) {
  // Query the determinism attribute of the operator schema instead of
  // maintaining a hardcoded list of non-deterministic ops. See
  // https://github.com/onnx/onnx/pull/7176.
  //
  // The ONNX operator schema registry stores the default ONNX domain as an
  // empty string.
  const std::string& lookup_domain = domain == "ai.onnx" ? "" : domain;
  const auto* schema =
      onnx::OpSchemaRegistry::Schema(op, opset_version, lookup_domain);
  if (schema == nullptr) {
    // Unknown op. Assume it is not deterministic.
    return false;
  }
  // Only fold ops that are known to be deterministic. Ops whose determinism
  // cannot be statically determined (e.g. context-dependent functions) are
  // treated as non-deterministic to be safe.
  return schema->GetNodeDeterminism() ==
         onnx::OpSchema::NodeDeterminism::Deterministic;
}

bool IsQDQ(const std::string& domain, const std::string& op) {
  if (domain == "ai.onnx" || domain.empty()) {
    return op == "QuantizeLinear" || op == "DequantizeLinear";
  }
  return false;
}

auto FindInitializerByName(const onnx::ModelProto& model,
                           const std::string& name) {
  for (const auto& initializer : model.graph().initializer()) {
    if (initializer.name() == name) {
      return initializer;
    }
  }
  throw std::invalid_argument("no initializer " + name);
}

auto FindValueInfoProtoByName(const onnx::ModelProto& model,
                              const std::string& name) {
  for (const auto& vi : model.graph().value_info()) {
    if (vi.name() == name) {
      return vi;
    }
  }
  for (const auto& initializer : model.graph().initializer()) {
    if (initializer.name() == name) {
      onnx::ValueInfoProto vi;
      for (const auto& dim : initializer.dims()) {
        vi.mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim()
            ->set_dim_value(dim);
      }
      vi.mutable_type()->mutable_tensor_type()->set_elem_type(
          initializer.data_type());
      vi.set_name(name);
      return vi;
    }
  }
  throw std::invalid_argument("no value info " + name);
}

#ifndef NO_BUILTIN_ORT
onnx::TensorProto TensorToTensorProto(const Ort::Value& tensor) {
  onnx::TensorProto tensor_proto;
  for (const auto& dim : tensor.GetTensorTypeAndShapeInfo().GetShape()) {
    tensor_proto.add_dims(dim);
  }
  onnx::TensorProto::DataType onnx_dtype =
      (onnx::TensorProto::DataType)tensor.GetTensorTypeAndShapeInfo()
          .GetElementType();
  tensor_proto.set_data_type(onnx_dtype);

  switch (onnx_dtype) {
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)                   \
  case onnx::TensorProto::onnx_dtype: {                                   \
    const auto* dptr = tensor.GetTensorData<cpp_type>();                  \
    for (size_t i = 0;                                                    \
         i < tensor.GetTensorTypeAndShapeInfo().GetElementCount(); i++) { \
      tensor_proto.add_##storage_dtype##_data(dptr[i]);                   \
    }                                                                     \
    break;                                                                \
  }

    CASE_DTYPE(FLOAT, float, float)
    CASE_DTYPE(DOUBLE, double, double)
    CASE_DTYPE(INT64, int64, int64_t)
    CASE_DTYPE(UINT64, uint64, uint64_t)
    CASE_DTYPE(INT32, int32, int32_t)
    CASE_DTYPE(UINT8, int32, uint8_t)
    CASE_DTYPE(INT8, int32, int8_t)
    CASE_DTYPE(UINT16, int32, uint16_t)
    CASE_DTYPE(INT16, int32, int16_t)
    CASE_DTYPE(BOOL, int32, int8_t)
#undef CASE_DTYPE
    default:
      throw std::invalid_argument("Unknown dtype " +
                                  std::to_string(tensor_proto.data_type()));
  }
  return tensor_proto;
}

Ort::Value TensorProtoToTensor(const onnx::TensorProto& tensor_proto) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto tensor = Ort::Value::CreateTensor(
      allocator, tensor_proto.dims().data(), tensor_proto.dims_size(),
      (ONNXTensorElementDataType)tensor_proto.data_type());
  if (tensor_proto.has_raw_data()) {
    if (onnxruntime::endian::native == onnxruntime::endian::big) {
      throw std::invalid_argument("only little endian is supported");
    }
    memcpy(tensor.GetTensorMutableData<void>(), tensor_proto.raw_data().data(),
           tensor_proto.raw_data().size());
  } else {
    switch (tensor_proto.data_type()) {
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)         \
  case onnx::TensorProto::onnx_dtype: {                         \
    std::vector<cpp_type> vec;                                  \
    for (const auto& x : tensor_proto.storage_dtype##_data()) { \
      vec.push_back(x);                                         \
    }                                                           \
    memcpy(tensor.GetTensorMutableData<void>(), vec.data(),     \
           vec.size() * sizeof(cpp_type));                      \
    break;                                                      \
  }
      CASE_DTYPE(FLOAT, float, float)
      CASE_DTYPE(DOUBLE, double, double)
      CASE_DTYPE(INT64, int64, int64_t)
      CASE_DTYPE(UINT64, uint64, uint64_t)
      CASE_DTYPE(INT32, int32, int32_t)
      CASE_DTYPE(UINT8, int32, uint8_t)
      CASE_DTYPE(INT8, int32, int8_t)
      CASE_DTYPE(UINT16, int32, uint16_t)
      CASE_DTYPE(INT16, int32, int16_t)
      CASE_DTYPE(BOOL, int32, int8_t)
#undef CASE_DTYPE
      default:
        throw std::invalid_argument("Unknown dtype " +
                                    std::to_string(tensor_proto.data_type()));
    }
  }
  return tensor;
}

std::shared_ptr<Ort::Env> GetEnv() {
  static std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>();
  return env;
}

struct CppModelExecutor : public ModelExecutor {
  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    std::transform(
        model.graph().input().begin(), model.graph().input().end(),
        std::back_inserter(input_name_ptrs),
        [](const onnx::ValueInfoProto& x) { return x.name().c_str(); });
    std::transform(
        model.graph().output().begin(), model.graph().output().end(),
        std::back_inserter(output_name_ptrs),
        [](const onnx::ValueInfoProto& x) { return x.name().c_str(); });
    Ort::SessionOptions sess_opts;
    sess_opts.SetLogSeverityLevel(3);
    sess_opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    std::string model_str = model.SerializeAsString();
    Ort::Session session(*GetEnv(), model_str.data(), model_str.size(),
                         sess_opts);
    Ort::RunOptions run_opts;
    run_opts.SetRunLogSeverityLevel(3);
    std::vector<Ort::Value> input_tensors;
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(input_tensors), TensorProtoToTensor);
    auto output_tensors = session.Run(
        run_opts, input_name_ptrs.data(), input_tensors.data(),
        input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());

    std::vector<onnx::TensorProto> output_tps;
    std::transform(output_tensors.begin(), output_tensors.end(),
                   std::back_inserter(output_tps), TensorToTensorProto);
    return output_tps;
  }
};

std::shared_ptr<const ModelExecutor> GetBuiltinModelExecutor() {
  static std::shared_ptr<const ModelExecutor> executor =
      std::make_shared<CppModelExecutor>();
  return executor;
}

void InitEnv() { GetEnv(); }
#else
void InitEnv() {
  // do nothing
}
#endif

std::vector<onnx::TensorProto> RunOp(const ModelExecutor& executor,
                                     onnx::ModelProto& model,
                                     const onnx::NodeProto& op) {
  std::vector<std::string> input_names;
  std::vector<onnx::TensorProto> input_tps;
  std::set<std::string> initializer_names;

  onnx::ModelProto op_model;
  op_model.set_ir_version(model.ir_version());
  for (const auto& x : model.opset_import()) {
    *op_model.add_opset_import() = x;
  }
  *op_model.mutable_graph()->add_node() = op;

  for (const auto& input : op.input()) {
    if (std::find(input_names.begin(), input_names.end(), input) !=
        input_names.end()) {
      continue;
    }
    // skip "" which represents the unset optional input
    if (input.empty()) {
      continue;
    }
    if (initializer_names.find(input) != initializer_names.end()) {
      continue;
    }
    auto in_tp = FindInitializerByName(model, input);
    if (in_tp.dims().size() == 1 && in_tp.dims()[0] == 0) {
      initializer_names.insert(input);
      *op_model.mutable_graph()->add_initializer() = in_tp;
      continue;
    }
    input_names.push_back(input);
    input_tps.push_back(in_tp);
  }

  for (const auto& x : input_names) {
    // skip "" which represents the unset optional input
    if (x.empty()) {
      continue;
    }
    *op_model.mutable_graph()->add_input() = FindValueInfoProtoByName(model, x);
  }
  for (const auto& x : op.output()) {
    onnx::ValueInfoProto vi;
    // In principle output ValueInfoProto must have type. But it is not checked.
    vi.set_name(x);
    *op_model.mutable_graph()->add_output() = vi;
  }

  using namespace ONNX_NAMESPACE::optimization;
  VLOG(1) << "Running node: " << op;
  auto output_tps = executor._Run(op_model, input_tps);
  for (size_t i = 0; i < op.output_size(); i++) {
    output_tps[i].set_name(op.output(i));
  }
  return output_tps;
}

void RunOpAndAddInitializer(const ModelExecutor& executor,
                            onnx::ModelProto& model,
                            const onnx::NodeProto& op) {
  const auto output_tps = RunOp(executor, model, op);
  for (const auto& output_tp : output_tps) {
    *model.mutable_graph()->add_initializer() = output_tp;
  }
}

bool HasSubgraph(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.type() == onnx::AttributeProto::GRAPH ||
        attr.type() == onnx::AttributeProto::GRAPHS) {
      return true;
    }
  }
  return false;
}

size_t size_of_dtype(onnx::TensorProto::DataType dtype) {
  switch (dtype) {
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8:
      return 1;
    case onnx::TensorProto::DataType::TensorProto_DataType_BFLOAT16:
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT16:
      return 2;
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32:
      return 4;
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX64:
      return 8;
    case onnx::TensorProto::DataType::TensorProto_DataType_COMPLEX128:
      return 16;
    // Don't know the size of string.. Just return 16.
    case onnx::TensorProto::DataType::TensorProto_DataType_STRING:
      return 16;
    default:
    case onnx::TensorProto::DataType::TensorProto_DataType_UNDEFINED:
      throw std::invalid_argument("Undefined or unknown datatype");
  }
  throw std::invalid_argument("Unknown datatype " + std::to_string(dtype));
}

bool ProduceLargeTensor(const onnx::ModelProto& model,
                        const onnx::NodeProto& node, size_t threshold) {
  std::set<std::string> large_tensor_ops{"Tile", "ConstantOfShape", "Expand"};
  if (large_tensor_ops.find(node.op_type()) == large_tensor_ops.end()) {
    return false;
  }
  for (const auto& value_info : model.graph().value_info()) {
    if (value_info.name() == node.output(0)) {
      size_t size = size_of_dtype(static_cast<onnx::TensorProto::DataType>(
          value_info.type().tensor_type().elem_type()));
      for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
        size *= dim.dim_value();
      }
      if (size <= threshold) {
        return false;
      }
    }
  }
  // If the output is not in value_info, we assume it is large.
  // There is a possibility that value_info is presented by the shape inference
  // later and `ProduceLargeTensor` is called again and returns false at that
  // time.
  return true;
}

std::pair<std::vector<onnx::NodeProto>, std::vector<onnx::NodeProto>>
GetConstantNodes(const onnx::ModelProto& model) {
  // tensor with empty name("") represents the empty value of an optional input
  // so "" should be treated as a name of a constant tensor.
  std::vector<std::string> const_names{""};
  std::vector<onnx::NodeProto> const_nodes;
  std::vector<onnx::NodeProto> non_const_nodes;
  std::transform(
      model.graph().initializer().begin(), model.graph().initializer().end(),
      std::back_inserter(const_names), [](const auto& x) { return x.name(); });
  // Map each domain to its imported opset version so the correct operator
  // schema can be looked up. The default ONNX domain is normalized to an empty
  // string, which is how the schema registry stores it.
  std::unordered_map<std::string, int> domain_to_version;
  for (const auto& opset : model.opset_import()) {
    const std::string& domain =
        opset.domain() == "ai.onnx" ? "" : opset.domain();
    domain_to_version[domain] = opset.version();
  }
  auto opset_version_of = [&domain_to_version](const std::string& domain) {
    const std::string& key = domain == "ai.onnx" ? "" : domain;
    auto iter = domain_to_version.find(key);
    return iter == domain_to_version.end() ? 0 : iter->second;
  };
  // node is already topo sorted
  for (const auto& node : model.graph().node()) {
    // clang-format off
    if (IsOfficialOp(node.domain(), node.op_type()) &&
        IsDeterministic(node.domain(), node.op_type(),
                        opset_version_of(node.domain())) &&
        !IsQDQ(node.domain(), node.op_type()) &&
        !HasSubgraph(node) &&
        !ProduceLargeTensor(model, node, config.tensor_size_threshold) &&
        // clang-format on
        std::all_of(node.input().begin(), node.input().end(),
                    [&const_names](const auto& x) {
                      return std::find(const_names.begin(), const_names.end(),
                                       x) != const_names.end();
                    })) {
      const_names.insert(const_names.end(), node.output().begin(),
                         node.output().end());
      const_nodes.push_back(node);
    } else {
      non_const_nodes.push_back(node);
    }
  }
  return {const_nodes, non_const_nodes};
}

onnx::ModelProto _InferShapes(const onnx::ModelProto& model) {
  onnx::ModelProto result;
  result.CopyFrom(model);
  onnx::shape_inference::InferShapes(result);
  return result;
}

onnx::ModelProto _FoldConstant(const ModelExecutor& executor,
                               const onnx::ModelProto& model) {
  const auto& tmp = model;
  {
    onnx::ModelProto model;
    model.CopyFrom(tmp);
    auto [const_nodes, non_const_nodes] = GetConstantNodes(model);
    (void)non_const_nodes;
    // Outputs of const nodes that were successfully folded into initializers.
    std::set<std::string> folded_outputs;
    for (const auto& x : const_nodes) {
      try {
        RunOpAndAddInitializer(executor, model, x);
        for (const auto& output : x.output()) {
          folded_outputs.insert(output);
        }
      } catch (const std::exception& e) {
        std::cerr << "WARNING: failed to run \"" << x.op_type() <<
          "\" op (name is \"" << x.name() << "\"), skip..." << std::endl;
      }
    }
    // Rebuild the node list in its original topological order, dropping only
    // the const nodes that were successfully folded into initializers. A const
    // node that failed to fold must keep its original position: appending it to
    // the end can place it after a non-const consumer (e.g. a Loop reading a
    // SequenceEmpty output), which breaks topological sorting and makes the
    // resulting model fail onnx's checker (issues #238, #335, #352).
    google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
    original_nodes.Swap(model.mutable_graph()->mutable_node());
    for (auto& node : original_nodes) {
      const bool folded = node.output_size() > 0 &&
                          folded_outputs.count(node.output(0)) > 0;
      if (!folded) {
        *model.mutable_graph()->add_node() = std::move(node);
      }
    }
    return model;
  }
}

onnx::ModelProto Optimize(const onnx::ModelProto& model) {
  return onnx::optimization::OptimizeFixed(model, config.optimizer_passes);
}

template <typename T>
std::function<T(const T&)> FixedPointFn(const std::function<T(const T&)>& f1,
                                        const std::function<T(const T&)>& f2,
                                        size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](const T& x) {
    size_t _max_iters = max_iters;
    T tmp1 = f1(x);
    T tmp2 = f2(tmp1);
    T& y1 = tmp1;
    T& y2 = tmp2;
    while (_max_iters-- > 0) {
      if (google::protobuf::util::MessageDifferencer::Equals(y1, y2)) {
        if (converged) {
          *converged = true;
        }
        return y2;
      }
      y1 = f1(y2);
      if (google::protobuf::util::MessageDifferencer::Equals(y1, y2)) {
        if (converged) {
          *converged = true;
        }
        return y1;
      }
      y2 = f2(y1);
    }

    if (converged) {
      *converged = false;
    }
    return y2;
  };
}

template <typename T>
std::function<T(const T&)> FixedPointFn(const std::function<T(const T&)>& f1,
                                        const std::function<T(const T&)>& f2,
                                        size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

onnx::ModelProto Identity(const onnx::ModelProto& model) { return model; }

void Check(const onnx::ModelProto& model) { onnx::checker::check_model(model); }

onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold) {
  // Make shape inference aware of ONNX Runtime's quantized contrib operators
  // (QLinearAdd and friends) so shape deduction does not stop at them.
  onnxsim::RegisterContribOpSchemas();

  Check(model);

  config.tensor_size_threshold = tensor_size_threshold;
  config.optimizer_passes.clear();
  // The onnxoptimizer pass "eliminate_if_with_const_cond" folds an `If` whose
  // condition is a compile-time constant by inlining the taken branch. It
  // segfaults when that branch's output is produced by an initializer instead
  // of a node -- a state onnxsim itself reaches when the earlier
  // "extract_constant_to_initializer" pass turns a subgraph `Constant` into a
  // subgraph initializer. The pass builds its output map only from subgraph
  // node outputs, so the branch output is missing and it dereferences a null
  // Value. Always skip it so simplification degrades to leaving the `If` in
  // place rather than aborting the whole process. Actually folding a
  // constant-condition `If` is tracked separately (GitHub issue #275).
  // See GitHub issue #452.
  static const std::set<std::string> always_skip_passes = {
      "eliminate_if_with_const_cond"};
  // skip_optimizers == nullopt means skiping all optimizers, so
  // config.optimizer_passes is empty
  if (skip_optimizers) {
    std::vector<std::string> passes;
    const auto all_passes = onnx::optimization::GetFuseAndEliminationPass();
    for (const auto& pass : all_passes) {
      if (always_skip_passes.count(pass) > 0) {
        continue;
      }
      if (std::find(skip_optimizers->begin(), skip_optimizers->end(), pass) ==
          skip_optimizers->end()) {
        passes.push_back(pass);
      }
    }
    config.optimizer_passes = passes;
  }

  std::function<onnx::ModelProto(const onnx::ModelProto&)> FoldConstant;
  if (constant_folding) {
    FoldConstant = [&executor](const onnx::ModelProto& model) {
      return _FoldConstant(executor, model);
    };
  } else {
    FoldConstant = Identity;
  }
  auto InferShapes = shape_inference ? _InferShapes : Identity;

  int fixed_point_iters =
      std::getenv("ONNXSIM_FIXED_POINT_ITERS")
          ? std::atoi(std::getenv("ONNXSIM_FIXED_POINT_ITERS"))
          : 50;

  auto OptAndShape = FixedPointFn(std::function{InferShapes},
                                  std::function{Optimize}, fixed_point_iters);
  bool converged = false;
  auto OptAndShapeAndFold =
      FixedPointFn(std::function{OptAndShape}, std::function{FoldConstant},
                   fixed_point_iters, &converged);
  auto sim_model = OptAndShapeAndFold(model);
  Check(sim_model);
  if (!converged) {
    std::cout << "WARNING: the simplification stopped because of timeout. "
                 "Please set environment variable `ONNXSIM_FIXED_POINT_ITERS` "
                 "to a number higher than "
              << fixed_point_iters << "if you want further simplification."
              << std::endl;
  }
  return sim_model;
}

void SimplifyPath(const ModelExecutor& executor, const std::string& in_path,
                  const std::string& out_path,
                  std::optional<std::vector<std::string>> skip_optimizers,
                  bool constant_folding, bool shape_inference,
                  size_t tensor_size_threshold) {
  onnx::ModelProto model;
  onnx::optimization::loadModel(&model, in_path, true);

  model = Simplify(executor, model, skip_optimizers, constant_folding,
                   shape_inference, tensor_size_threshold);

  onnx::optimization::saveModel(&model, out_path, true, "");
}
