#include "onnxsim.h"

#include <google/protobuf/arena.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ONNXSIM_HAS_ORT
// Both ways CMake obtains ONNX Runtime (an official release, or
// cmake/build_ort.cmake's own from-source ExternalProject build) now install
// their public headers flat (e.g. onnxruntime_cxx_api.h directly under the
// include root) -- see ONNXSIM_ORT_FLAT_HEADERS in CMakeLists.txt /
// cmake/build_ort.cmake. Byte order is handled with C++20 std::endian in
// dlpack_dtype.h, so this does not depend on ORT's internal
// core/common/endian.h (which neither ships).
#include "onnxruntime_cxx_api.h"
#endif
#include "contrib_schemas.h"
#include "custom_optimizer_passes.h"
#include "dlpack_bridge.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/graph_shape_inference.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/logging.h"
#include "profiler.h"
#include "sym_shape_infer.h"
#include "sym_value_eval.h"
#include "tensor_pool_bridge.h"

struct Config {
  std::vector<std::string> optimizer_passes;
  // default value is max
  size_t tensor_size_threshold = -1;
  // Whether graph initializers are treated as constant tensors. When false,
  // constant folding does not seed its constant set with initializer names, so
  // nodes rooted only at initializers are left unfolded (see GetConstantNodes),
  // and the onnx optimizer is told to leave initializer-backed weights alone
  // too (see Optimize). Constant nodes are always constant.
  bool initializers_as_constants = true;
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

// Correct the determinism metadata of operators ONNX mis-annotates, so the
// ordinary ``IsDeterministic`` check below can fold them.
//
// ``OpSchema::GetNodeDeterminism`` infers a *function* op's determinism from
// the ops in its function body, and reports ``NonDeterministic`` for a body
// that contains a subgraph-carrying op (``Loop``/``If``/``Scan``) and
// ``Unknown`` for a context-dependent function -- neither of which means the op
// is actually random. ``Range`` is the canonical victim: its body is a ``Loop``
// (opset < 27) or a context-dependent function (opset >= 27), so it is reported
// non-deterministic even though its output is a pure function of its inputs. It
// is then never constant-folded, which in turn strands whole static subgraphs
// built on top of it -- e.g. the ``Range -> Slice -> Reshape -> Expand ->
// Unsqueeze -> Concat`` attention-mask construction (and the neighbouring
// ``ScatterND`` chains) in Swin-style models, leaving hundreds of constant
// nodes that other simplifiers fold away.
//
// Rather than second-guess the determinism query in ``IsDeterministic``, fix
// the source data: mark these genuinely-deterministic ops ``Deterministic`` on
// their registered schemas (every version in the registry's history). The
// registry returns pointers into its own storage, so this updates the metadata
// in place.
void FixupSchemaDeterminism() {
  static std::once_flag once;
  std::call_once(once, [] {
    // Deterministic default-domain ops whose schema determinism ONNX infers
    // (incorrectly, for folding purposes) from a function body.
    static const std::set<std::string> deterministic_ops = {"Range"};
    for (const auto& schema :
         onnx::OpSchemaRegistry::get_all_schemas_with_history()) {
      if (!schema.domain().empty() || !deterministic_ops.count(schema.Name())) {
        continue;
      }
      const onnx::OpSchema* registered = onnx::OpSchemaRegistry::Schema(
          schema.Name(), schema.since_version(), schema.domain());
      if (registered != nullptr) {
        const_cast<onnx::OpSchema*>(registered)
            ->SetNodeDeterminism(
                onnx::OpSchema::NodeDeterminism::Deterministic);
      }
    }
  });
}

bool IsDeterministic(const std::string& domain, const std::string& op,
                     int opset_version) {
  // Query the determinism attribute of the operator schema instead of
  // maintaining a hardcoded list of non-deterministic ops. See
  // https://github.com/onnx/onnx/pull/7176. Operators ONNX mis-annotates for
  // constant-folding purposes (e.g. ``Range``) have their metadata corrected by
  // FixupSchemaDeterminism(), which Simplify() runs before folding.
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

// Returns a reference into `model`'s own initializer list rather than a copy:
// callers that only read the tensor (the common case) avoid deep-copying its
// raw_data bytes just to look it up.
const onnx::TensorProto& FindInitializerByName(const onnx::ModelProto& model,
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

#ifdef ONNXSIM_HAS_ORT
// The TensorProto<->Ort::Value converters that used to live here have moved to
// dlpack_bridge.h and now exchange data through DLManagedTensor:
//   * inputs: onnxsim::dlpack::BorrowAsOrtValue wraps the feed buffer with the
//     borrowing CreateTensor overload -- no copy in;
//   * outputs: onnxsim::dlpack::FromOrtValue moves ORT's own output allocation
//     into the returned tensor -- no copy out (and no per-element add_*_data).

std::shared_ptr<Ort::Env> GetEnv() {
  static std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>();
  return env;
}

// Turn on ONNX Runtime's own per-operator session profiler when
// ONNXSIM_ORT_PROFILE is set, and return whether it was enabled. This is
// separate from onnxsim's span profiler (ONNXSIM_PROFILE): it makes each
// constant-folding session dump ONNX Runtime's detailed per-kernel Chrome
// trace. The variable names a file prefix (ONNX Runtime writes one
// ``<prefix>_<timestamp>.json`` per session); the truthy shorthands select a
// default prefix, mirroring ONNXSIM_PROFILE.
bool EnableOrtProfilingFromEnv(Ort::SessionOptions& sess_opts) {
  // Merging (ONNXSIM_MERGE_ORT_PROFILE) also needs the per-session traces, and
  // writes them to an intermediate prefix that Finish() folds in and deletes.
  const bool merging = onnxsim::Profiler::Instance().merge_ort_traces();
  const char* env = std::getenv("ONNXSIM_ORT_PROFILE");
  if (env == nullptr && !merging) {
    return false;
  }
  std::string prefix;
  if (merging) {
    prefix = "onnxsim_ort_merge_tmp";
  } else {
    prefix = env;
    if (prefix.empty() || prefix == "1" || prefix == "true" || prefix == "on" ||
        prefix == "yes") {
      prefix = "onnxsim_ort_profile";
    }
  }
#ifdef _WIN32
  // ORTCHAR_T is wchar_t on Windows; widen the (ASCII) prefix for the API.
  std::wstring wprefix(prefix.begin(), prefix.end());
  sess_opts.EnableProfiling(wprefix.c_str());
#else
  sess_opts.EnableProfiling(prefix.c_str());
#endif
  return true;
}

struct CppModelExecutor : public ModelExecutor {
  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
    // The RunOps call site already profiles each fold group's session run as a
    // single ``OrtSession`` span (see RunOps); for the built-in executor break
    // that down further into ``OrtSessionInit`` (building the session, where
    // ONNX Runtime loads the graph and usually the dominant cost) and
    // ``OrtSessionRun`` (the inference). All ProfiledScopes are no-ops unless
    // ONNXSIM_PROFILE is set, so this adds nothing otherwise.
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
    // This executor exists only to run constant-folding's throwaway
    // fold-group sub-models (see RunOps above) -- never a full-size
    // correctness check -- and each session here runs exactly once. So:
    //  - Memory-pattern planning, which pays off across *repeated* Run()
    //    calls, buys nothing for a session used once; skip planning it.
    //  - A fresh intra-op thread pool sized to the machine's CPU count is
    //    spun up (and joined) on every session construction, which happens
    //    once per fold group per fixed-point round -- often hundreds of times
    //    per large model. For the shape/index ops typical of a fold group
    //    that spin-up/join is pure overhead, and it is where most of a fold
    //    session's time goes (see the ``OrtSessionInit`` span above).
    //    Running single-threaded skips it.
    sess_opts.DisableMemPattern();
    sess_opts.SetIntraOpNumThreads(1);
    sess_opts.SetInterOpNumThreads(1);
    const bool ort_profiling = EnableOrtProfilingFromEnv(sess_opts);
    std::string model_str = model.SerializeAsString();
    Ort::Session session{nullptr};
    {
      onnxsim::ProfiledScope init_scope("OrtSessionInit");
      session = Ort::Session(*GetEnv(), model_str.data(), model_str.size(),
                             sess_opts);
    }
    Ort::RunOptions run_opts;
    run_opts.SetRunLogSeverityLevel(3);
    // Borrow each feed's buffer directly into an Ort::Value -- no copy. The
    // DLManagedTensors are owned by the caller (RunOps) and outlive this call,
    // so the borrowed pointers stay valid through session.Run.
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(inputs.size());
    for (const DLManagedTensor* in : inputs) {
      input_tensors.push_back(onnxsim::dlpack::BorrowAsOrtValue(in->dl_tensor));
    }
    std::vector<Ort::Value> output_tensors;
    {
      onnxsim::ProfiledScope run_scope("OrtSessionRun");
      output_tensors =
          session.Run(run_opts, input_name_ptrs.data(), input_tensors.data(),
                      input_tensors.size(), output_name_ptrs.data(),
                      output_name_ptrs.size());
    }
    if (ort_profiling) {
      // Flush ONNX Runtime's profiling trace for this session to disk. When
      // merging, hand its path to the profiler so Finish() folds it into the
      // onnxsim trace (and deletes the intermediate file).
      Ort::AllocatorWithDefaultOptions allocator;
      Ort::AllocatedStringPtr profile_file =
          session.EndProfilingAllocated(allocator);
      if (onnxsim::Profiler::Instance().merge_ort_traces() &&
          profile_file != nullptr) {
        onnxsim::Profiler::Instance().AddOrtTracePath(profile_file.get());
      }
    }

    // Hand ORT's own output buffers out as DLManagedTensors: FromOrtValue moves
    // each Ort::Value into the managed tensor, so nothing is copied here (the
    // one unavoidable copy happens when RunOps bakes the result into the
    // model's initializers as raw_data).
    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(output_tensors.size());
    for (auto& v : output_tensors) {
      outputs.emplace_back(onnxsim::dlpack::FromOrtValue(std::move(v)));
    }
    return outputs;
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

// Fold a group of const nodes together by building one sub-model that produces
// all of their outputs, running it through `executor` in a single Session, and
// returning the resulting tensors (named, in group order). Folding many nodes
// per Session collapses what used to be one Session construction per node into
// one per group, which is the dominant cost of constant folding. `ops` must be
// in topological order (as produced by GetConstantNodes); a group may therefore
// contain nodes that consume the outputs of earlier nodes in the same group --
// such tensors stay internal to the sub-model instead of becoming feeds.
//
// `deferred_producers` maps the output name of every "deferred" node (a
// ConstantOfShape or Constant->Expand that is logically constant but was not
// materialized because it would produce a large tensor, see GetConstantNodes)
// to the node itself. When an input of a grouped node is such a deferred
// output, the producing node is inlined into the sub-model instead of being
// looked up as an already-materialized initializer. This is applied
// transitively, so a whole chain (e.g. ConstantOfShape -> Expand -> Reshape)
// runs together inside the executor: the large intermediate tensors are
// computed transiently and only the (smaller) grouped outputs are returned to
// be stored as initializers.
std::vector<onnx::TensorProto> RunOps(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    const std::vector<const onnx::NodeProto*>& ops,
    const std::unordered_map<std::string, const onnx::NodeProto*>&
        deferred_producers) {
  std::vector<std::string> input_names;
  // Pointers borrow directly from `model`'s own initializers -- `model` is
  // const and not touched again until every pointer here has been consumed
  // (by the DLPack bridge below, within this same call), so this avoids
  // deep-copying each constant input's raw_data just to feed the executor.
  std::vector<const onnx::TensorProto*> input_tps;
  // Names already emitted as a feed or an initializer of the sub-model, so a
  // constant shared by several grouped nodes is added exactly once.
  std::set<std::string> seen_inputs;

  // Build the throwaway sub-model on an arena. RunOp is called once per
  // foldable node -- often thousands of times across a fixed-point run -- and
  // each call copies initializers and nodes into `op_model`, so the message is
  // a deep tree of nested sub-messages (NodeProto, TensorProto, ValueInfoProto,
  // dims, ...). Without an arena, destroying it walks that whole tree freeing
  // each sub-message individually; on an arena the entire tree is released in
  // one bulk free when `arena` goes out of scope. `Create` propagates the arena
  // pointer to every `add_*`/`mutable_*` sub-message -- that propagation is
  // what makes the teardown cheap. (Older protobuf spelled this
  // arena-propagating form `Arena::CreateMessage`, but that alias was
  // deprecated in protobuf 5.x and removed in 6.x -- the floor the bundled ONNX
  // now requires -- so `Create` is the modern equivalent for message types.)
  // The sub-model is strictly local: it is never Swap'd or moved into `model`,
  // and the executor returns its outputs in a separate std::vector that does
  // not live on this arena, so the arena can be torn down on return without
  // dangling anything the caller keeps.
  google::protobuf::Arena arena;
  onnx::ModelProto& op_model =
      *google::protobuf::Arena::Create<onnx::ModelProto>(&arena);
  // Spans the sub-model-construction phase below (building op_model: copying
  // each grouped node and its constant inputs into the throwaway sub-model)
  // -- not lexically scoped via ProfiledScope's RAII since input_names/
  // input_tps/output_names, populated in this phase, are also read by the
  // DLPack-bridging and output-materialization phases after it ends. See
  // ONNXSIM_PROFILE's own doc comment for what "the tensor copying inside
  // constant folding" actually covers.
  onnxsim::Profiler::Instance().Begin("BuildSubModel");
  op_model.set_ir_version(model.ir_version());
  for (const auto& x : model.opset_import()) {
    *op_model.add_opset_import() = x;
  }

  // Outputs produced by a node in the group: these are computed inside the
  // sub-model, so a grouped node consuming one must not treat it as an external
  // constant feed.
  std::set<std::string> internal_outputs;
  for (const auto* op : ops) {
    for (const auto& output : op->output()) {
      internal_outputs.insert(output);
    }
  }

  // Post-order traversal: emit every deferred producer before its consumer, and
  // each grouped node in topological order, so the sub-model stays
  // topologically sorted. Each node is included at most once even when several
  // consumers share it.
  std::set<const onnx::NodeProto*> included;
  std::function<void(const onnx::NodeProto&)> include_node =
      [&](const onnx::NodeProto& node) {
        if (!included.insert(&node).second) {
          return;
        }
        for (const auto& input : node.input()) {
          // skip "" which represents the unset optional input
          if (input.empty()) {
            continue;
          }
          // Produced by another node in the group: it is an intermediate of the
          // sub-model, not an external input.
          if (internal_outputs.find(input) != internal_outputs.end()) {
            continue;
          }
          auto deferred_iter = deferred_producers.find(input);
          if (deferred_iter != deferred_producers.end()) {
            // Produced by a deferred node: inline it rather than treating the
            // (unmaterialized) output as an external constant input.
            include_node(*deferred_iter->second);
            continue;
          }
          if (!seen_inputs.insert(input).second) {
            continue;
          }
          const auto& in_tp = FindInitializerByName(model, input);
          if (in_tp.dims().size() == 1 && in_tp.dims()[0] == 0) {
            *op_model.mutable_graph()->add_initializer() = in_tp;
            continue;
          }
          input_names.push_back(input);
          input_tps.push_back(&in_tp);
        }
        *op_model.mutable_graph()->add_node() = node;
      };
  for (const auto* op : ops) {
    include_node(*op);
  }

  for (const auto& x : input_names) {
    // skip "" which represents the unset optional input
    if (x.empty()) {
      continue;
    }
    *op_model.mutable_graph()->add_input() = FindValueInfoProtoByName(model, x);
  }
  // Mark every grouped output as a graph output so the single Run materializes
  // all of them. `output_names` records them in graph-output order, which is
  // the order the executor returns the tensors in.
  std::vector<std::string> output_names;
  for (const auto* op : ops) {
    for (const auto& x : op->output()) {
      onnx::ValueInfoProto vi;
      // In principle output ValueInfoProto must have type. But it is not
      // checked.
      vi.set_name(x);
      *op_model.mutable_graph()->add_output() = vi;
      output_names.push_back(x);
    }
  }
  onnxsim::Profiler::Instance().End();  // BuildSubModel

  using namespace ONNX_NAMESPACE::optimization;
  VLOG(1) << "Running " << ops.size() << " node(s) as one batch";
  // Constant folding's actual work is running each fold group's sub-model
  // through the model executor -- an ONNX Runtime session. Profile that run so
  // it shows up in the trace nested under FoldConstant. This is the one spot
  // common to every executor (the built-in ONNX Runtime one and the Python
  // trampoline that Python's simplify() injects), so the session run is
  // profiled regardless of binding. The ProfiledScope is a no-op unless
  // ONNXSIM_PROFILE is set.
  // Bridge to the DLPack executor boundary. Each feed borrows its initializer's
  // buffer (no copy); `input_tps` is fully built above and not mutated again,
  // so the borrowed pointers stay valid. `input_dls` owns the managed tensors
  // and must outlive the executor call (the executor borrows them). Outputs
  // come back as DLManagedTensors and are baked into TensorProto raw_data here
  // -- the single, unavoidable copy, since folded results become model
  // initializers.
  std::vector<DLManagedTensorPtr> input_dls;
  std::vector<const DLManagedTensor*> input_ptrs;
  {
    onnxsim::ProfiledScope dlpack_input_scope("DLPackInputBridge");
    input_dls.reserve(input_tps.size());
    for (const auto* tp : input_tps) {
      input_dls.emplace_back(onnxsim::dlpack::FromTensorProtoBorrowing(*tp));
    }
    input_ptrs.reserve(input_dls.size());
    for (const auto& p : input_dls) input_ptrs.push_back(p.get());
  }

  std::vector<DLManagedTensorPtr> output_dls;
  {
    onnxsim::ProfiledScope session_scope("OrtSession");
    output_dls = executor.Run(op_model, input_ptrs);
  }
  std::vector<onnx::TensorProto> output_tps;
  onnxsim::ProfiledScope dlpack_output_scope("DLPackOutputCopy");
  output_tps.reserve(output_dls.size());
  for (size_t i = 0; i < output_dls.size(); i++) {
    output_tps.push_back(onnxsim::dlpack::ToTensorProto(
        output_dls[i]->dl_tensor,
        i < output_names.size() ? output_names[i] : std::string()));
  }
  return output_tps;
}

void RunOpsAndAddInitializers(
    const ModelExecutor& executor, onnx::ModelProto& model,
    const std::vector<const onnx::NodeProto*>& ops,
    const std::unordered_map<std::string, const onnx::NodeProto*>&
        deferred_producers) {
  const auto output_tps = RunOps(executor, model, ops, deferred_producers);
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

// The result of partitioning a graph's nodes for constant folding.
struct ConstantNodePartition {
  // Nodes whose output is materialized into an initializer by folding.
  std::vector<onnx::NodeProto> const_nodes;
  // Nodes kept in the graph (folded consumers reference their outputs via
  // initializers, or they are genuinely non-constant runtime nodes).
  std::vector<onnx::NodeProto> non_const_nodes;
  // Outputs of "deferred" nodes: ConstantOfShape/Expand nodes whose inputs are
  // all constant but that were not folded into an initializer because they
  // would produce a large tensor. They stay in the graph (in non_const_nodes),
  // yet their outputs are treated as constant so downstream constant nodes stay
  // foldable; RunOps inlines the producing node into the sub-model it executes,
  // so the large intermediate tensor is computed transiently and never stored.
  std::set<std::string> deferred_outputs;
};

ConstantNodePartition GetConstantNodes(const onnx::ModelProto& model) {
  // tensor with empty name("") represents the empty value of an optional input
  // so "" should be treated as a name of a constant tensor.
  //
  // A hash set, not a vector: every node's every input is looked up against
  // this set below (``std::all_of`` over ``const_names``), and the set grows
  // by every initializer up front plus every folded node's outputs as the
  // scan proceeds. A vector + linear ``std::find`` makes that lookup
  // O(constants seen so far) per input, i.e. O(nodes * initializers) overall
  // on a model with many weights; a hash set makes each lookup O(1) average.
  std::unordered_set<std::string> const_names{""};
  ConstantNodePartition partition;
  auto& const_nodes = partition.const_nodes;
  auto& non_const_nodes = partition.non_const_nodes;
  // Seed the constant set with the initializer names, unless the caller asked
  // for initializers to be treated as non-constant. In that case a node whose
  // inputs are (only) initializers is not foldable, so its weights are left in
  // the graph untouched; a node fed by a Constant node still folds because ""
  // and Constant outputs remain in the constant set.
  if (config.initializers_as_constants) {
    for (const auto& x : model.graph().initializer()) {
      const_names.insert(x.name());
    }
  }
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
    const bool foldable =
        IsOfficialOp(node.domain(), node.op_type()) &&
        IsDeterministic(node.domain(), node.op_type(),
                        opset_version_of(node.domain())) &&
        !IsQDQ(node.domain(), node.op_type()) && !HasSubgraph(node) &&
        std::all_of(
            node.input().begin(), node.input().end(),
            [&const_names](const auto& x) { return const_names.count(x) > 0; });
    if (!foldable) {
      non_const_nodes.push_back(node);
      continue;
    }
    if (!ProduceLargeTensor(model, node, config.tensor_size_threshold)) {
      // Ordinary constant folding: the output is materialized as an
      // initializer and the node is dropped.
      const_names.insert(node.output().begin(), node.output().end());
      const_nodes.push_back(node);
      continue;
    }
    // Large-tensor op. ConstantOfShape and the foldable Expand (the
    // "Constant -> Expand" pattern) are folded lazily: the node is kept in the
    // graph but its output is still treated as constant so consumers keep
    // folding, and RunOps inlines it into the executor's sub-model at fold time
    // rather than materializing the large tensor as an initializer. Other
    // large-tensor ops (e.g. Tile) remain fully excluded from folding.
    if (node.op_type() == "ConstantOfShape" || node.op_type() == "Expand") {
      const_names.insert(node.output().begin(), node.output().end());
      partition.deferred_outputs.insert(node.output().begin(),
                                        node.output().end());
    }
    non_const_nodes.push_back(node);
  }
  return partition;
}

// Recursively collect the names of every tensor consumed as a node input,
// descending into subgraphs (e.g. the branches of "If" or the body of "Loop").
// Because ONNX subgraphs can reference tensors from the enclosing scope, an
// initializer in the main graph may be used only by a node inside a subgraph.
// Collecting names recursively ensures such initializers are not mistaken for
// unused ones (issue #174).
void CollectUsedTensorNames(const onnx::GraphProto& graph,
                            std::set<std::string>& used) {
  for (const auto& node : graph.node()) {
    for (const auto& input : node.input()) {
      if (!input.empty()) {
        used.insert(input);
      }
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectUsedTensorNames(attr.g(), used);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectUsedTensorNames(subgraph, used);
      }
    }
  }
  // Graph outputs must be kept even if no node consumes them.
  for (const auto& output : graph.output()) {
    used.insert(output.name());
  }
}

// Remove initializers of the main graph that are no longer referenced by any
// node (including nodes in subgraphs). Constant folding replaces a subgraph of
// const ops (e.g. a Transpose on a weight) with a freshly computed initializer,
// but leaves the original operand initializers in place. Without cleanup those
// dangling weights are duplicated in the graph, which can push the model past
// the 2GB protobuf limit before the onnx optimizer gets a chance to remove
// them (issue #174).
// Takes `model` by value and mutates it in place rather than copying into a
// separate `result`: every caller already owns a private, uniquely-held copy
// by this point and passes it via std::move, so this is a cheap move-in, not
// a deep copy of the (potentially huge) initializer bytes.
onnx::ModelProto EliminateUnusedInitializer(onnx::ModelProto model) {
  std::set<std::string> used;
  CollectUsedTensorNames(model.graph(), used);
  // Keep initializers that double as graph inputs (their default value);
  // dropping them would silently turn them into required inputs.
  for (const auto& input : model.graph().input()) {
    used.insert(input.name());
  }

  google::protobuf::RepeatedPtrField<onnx::TensorProto> kept;
  for (auto& initializer : *model.mutable_graph()->mutable_initializer()) {
    if (used.count(initializer.name()) > 0) {
      *kept.Add() = std::move(initializer);
    }
  }
  model.mutable_graph()->mutable_initializer()->Swap(&kept);

  return model;
}

// Mutates the model in place; ``onnx::shape_inference::InferShapes`` already
// works in place, so no extra ModelProto copy is made (the previous ``const&``
// signature forced a defensive ``CopyFrom`` because the input could not be
// mutated).
void _InferShapes(onnx::ModelProto& model) {
  onnx::shape_inference::InferShapes(model);
}

// Build a lookup from tensor name to its type, gathering shapes from every
// place a shape can be declared: value_info (populated by shape inference),
// graph inputs and graph outputs. Pointers reference `model`, so the map must
// not outlive it and `model` must not be mutated while the map is in use.
std::unordered_map<std::string, const onnx::TypeProto*> BuildTypeMap(
    const onnx::ModelProto& model) {
  std::unordered_map<std::string, const onnx::TypeProto*> type_map;
  auto add = [&type_map](const onnx::ValueInfoProto& vi) {
    if (vi.has_type()) {
      type_map[vi.name()] = &vi.type();
    }
  };
  for (const auto& vi : model.graph().value_info()) add(vi);
  for (const auto& vi : model.graph().input()) add(vi);
  for (const auto& vi : model.graph().output()) add(vi);
  return type_map;
}

// Fetch the element type and a fully static shape of `name` from `type_map`.
// Returns false unless the tensor has a known integer (INT64/INT32) element
// type and a shape whose every dimension is a fixed value. A rank-0 (scalar)
// tensor yields an empty `dims` (element count 1).
bool GetStaticIntTensorInfo(
    const std::unordered_map<std::string, const onnx::TypeProto*>& type_map,
    const std::string& name, onnx::TensorProto::DataType& elem_type,
    std::vector<int64_t>& dims) {
  auto iter = type_map.find(name);
  if (iter == type_map.end() || !iter->second->has_tensor_type()) {
    return false;
  }
  const auto& tensor_type = iter->second->tensor_type();
  elem_type = static_cast<onnx::TensorProto::DataType>(tensor_type.elem_type());
  if (elem_type != onnx::TensorProto::INT64 &&
      elem_type != onnx::TensorProto::INT32) {
    return false;
  }
  if (!tensor_type.has_shape()) {
    // Rank is unknown.
    return false;
  }
  dims.clear();
  for (const auto& dim : tensor_type.shape().dim()) {
    if (!dim.has_dim_value()) {
      return false;
    }
    dims.push_back(dim.dim_value());
  }
  return true;
}

// --- Native symbolic shape evaluation (issue #532, milestones M1/M2/M3) ------
//
// The ONNX data-propagation path below stalls at any arithmetic over a dynamic
// dim symbol: it carries a value as a TensorShapeProto whose entries are a
// concrete int or an *opaque* dim_param string, so a Reshape target like
// `[batch, 1024, 128]`, or a `Div`/`Where`/`Equal` over the symbol, cannot be
// evaluated. The dependency-free evaluator in sym_value_eval / sym_shape_infer
// keeps each dynamic dim as a `SymExpr` and computes the shape algebra. These
// helpers adapt an `onnx::ModelProto` into the evaluator's plain structs and
// run M2 (symbolic activation shapes) then M1 (symbolic value evaluation) over
// it.

// Read one little-endian `T` out of `p`. ONNX defines TensorProto::raw_data as
// little-endian on every host, so this is a plain byte-wise decode rather than
// a memcpy into a host integer (which would be correct only on a little-endian
// machine -- see docs/big-endian.md).
template <typename T>
T ReadLittleEndian(const char* p) {
  std::make_unsigned_t<T> v = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    v |= static_cast<std::make_unsigned_t<T>>(static_cast<unsigned char>(p[i]))
         << (8 * i);
  }
  return static_cast<T>(v);
}

// Convert an integer TensorProto (rank 0 or 1, INT64/INT32, inline data) to a
// SymTensor of concrete values. Returns nullopt for other dtypes/ranks or data
// kept in an external file.
std::optional<onnxsim::SymTensor> IntTensorToSymTensor(
    const onnx::TensorProto& tp) {
  if (tp.data_location() == onnx::TensorProto::EXTERNAL) return std::nullopt;
  const auto dt = tp.data_type();
  if (dt != onnx::TensorProto::INT64 && dt != onnx::TensorProto::INT32)
    return std::nullopt;
  if (tp.dims_size() > 1) return std::nullopt;  // rank 0 (scalar) or 1 only
  const bool scalar = tp.dims_size() == 0;
  std::vector<int64_t> vals;
  if (tp.has_raw_data()) {
    const std::string& raw = tp.raw_data();
    if (dt == onnx::TensorProto::INT64) {
      const size_t n = raw.size() / sizeof(int64_t);
      vals.resize(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = ReadLittleEndian<int64_t>(raw.data() + i * sizeof(int64_t));
    } else {
      const size_t n = raw.size() / sizeof(int32_t);
      vals.resize(n);
      for (size_t i = 0; i < n; ++i)
        vals[i] = ReadLittleEndian<int32_t>(raw.data() + i * sizeof(int32_t));
    }
  } else if (dt == onnx::TensorProto::INT64) {
    vals.assign(tp.int64_data().begin(), tp.int64_data().end());
  } else {
    vals.assign(tp.int32_data().begin(), tp.int32_data().end());
  }
  const int64_t expect = scalar ? 1 : tp.dims(0);
  if (static_cast<int64_t>(vals.size()) != expect) return std::nullopt;
  onnxsim::SymTensor t;
  t.scalar = scalar;
  for (int64_t v : vals) t.data.emplace_back(v);
  return t;
}

// A TypeProto's shape as a SymShape: dim_value -> SymExpr(v), a non-empty
// dim_param -> its Symbol, an otherwise-unknown dim -> a fresh distinct symbol
// (so the rank is preserved). Returns nullopt when the rank itself is unknown.
std::optional<onnxsim::SymShape> TypeProtoToSymShape(
    const onnx::TypeProto& type, int64_t& fresh) {
  if (!type.has_tensor_type() || !type.tensor_type().has_shape())
    return std::nullopt;
  onnxsim::SymShape shape;
  for (const auto& dim : type.tensor_type().shape().dim()) {
    if (dim.has_dim_value())
      shape.push_back(onnxsim::SymExpr(dim.dim_value()));
    else if (!dim.dim_param().empty())
      shape.push_back(onnxsim::SymExpr::Symbol(dim.dim_param()));
    else
      shape.push_back(
          onnxsim::SymExpr::Symbol("seedunk_" + std::to_string(fresh++)));
  }
  return shape;
}

// One node in the evaluator's plain form. A node from a non-default domain gets
// an empty op_type so no handler matches it (its outputs stay unevaluated).
onnxsim::SymNode ToSymNode(const onnx::NodeProto& node) {
  onnxsim::SymNode n;
  const std::string& domain = node.domain();
  n.op_type = (domain.empty() || domain == "ai.onnx") ? node.op_type() : "";
  n.input.assign(node.input().begin(), node.input().end());
  n.output.assign(node.output().begin(), node.output().end());
  for (const auto& attr : node.attribute()) {
    onnxsim::SymAttr a;
    a.name = attr.name();
    switch (attr.type()) {
      case onnx::AttributeProto::INT:
        a.i = attr.i();
        break;
      case onnx::AttributeProto::INTS:
        a.ints.assign(attr.ints().begin(), attr.ints().end());
        break;
      case onnx::AttributeProto::TENSOR:
        if (auto t = IntTensorToSymTensor(attr.t())) a.t = std::move(*t);
        break;
      default:
        break;
    }
    n.attribute.push_back(std::move(a));
  }
  return n;
}

// Run M2 (symbolic activation-shape inference) then M1 (symbolic value
// evaluation) over `model`, returning every shape-data tensor the evaluator
// could resolve as a SymTensor (its entries possibly still symbolic).
std::map<std::string, onnxsim::SymTensor> EvaluateModelSymbolicValues(
    const onnx::ModelProto& model) {
  int64_t fresh = 0;
  std::vector<onnxsim::SymNode> nodes;
  nodes.reserve(model.graph().node_size());
  for (const auto& node : model.graph().node())
    nodes.push_back(ToSymNode(node));

  std::map<std::string, onnxsim::SymTensor> initializers;
  std::map<std::string, onnxsim::SymShape> shapes_seed;
  for (const auto& init : model.graph().initializer()) {
    if (auto t = IntTensorToSymTensor(init)) initializers[init.name()] = *t;
    onnxsim::SymShape s;  // an initializer's own shape is fully static
    for (int64_t d : init.dims()) s.emplace_back(d);
    shapes_seed[init.name()] = std::move(s);
  }
  auto seed = [&](const onnx::ValueInfoProto& vi) {
    if (shapes_seed.count(vi.name()))
      return;  // keep the concrete initializer shape
    if (auto s = TypeProtoToSymShape(vi.type(), fresh))
      shapes_seed[vi.name()] = std::move(*s);
  };
  for (const auto& vi : model.graph().input()) seed(vi);
  for (const auto& vi : model.graph().value_info()) seed(vi);
  for (const auto& vi : model.graph().output()) seed(vi);

  onnxsim::ShapeGraph sg;
  sg.node = nodes;
  sg.value_info = shapes_seed;
  sg.initializer = initializers;

  onnxsim::SymGraph vg;
  vg.node = std::move(nodes);
  vg.initializer = std::move(initializers);
  vg.shape = onnxsim::InferSymbolicShapes(sg);
  return onnxsim::EvaluateSymbolicValues(vg);
}

// Partial shape evaluation (issue #139) via ONNX data propagation.
//
// The plain constant folder only folds a node when *all* of its inputs are
// constant, so shape-computing ops like `Shape` are never folded: their input
// is an activation. Yet those ops depend solely on shapes, which shape
// inference knows -- fully or partially -- even when some dimensions stay
// dynamic.
//
// ONNX shape inference can *propagate* those partially known values: with data
// propagation enabled it fills a DataValueMap mapping each tensor to a
// TensorShapeProto whose entries are either a concrete dim_value or a symbolic
// dim_param. Ops across the shape family (Shape, Gather, Slice, Concat,
// Squeeze/Unsqueeze, Cast, Add/Sub/Mul, ...) participate, so a chain like
//   Shape([batch, C, H, W]) -> Gather([1, 2, 3])  ==>  [C, H, W]
// is propagated end to end and comes out fully concrete even though the batch
// dimension stays dynamic (the mask-rcnn pattern from issue #139).
//
// This pass rewrites every node whose lone output has a fully concrete
// propagated value into a `Constant` node. Downstream ops then fold through the
// ordinary constant folder, and now-dead nodes are removed by the optimizer.
void _EvalPartialShape(onnx::ModelProto& model) {
  // This pass runs shape inference with *data propagation* (lenient options)
  // purely to discover foldable shape values; it must not otherwise change the
  // model. InferShapes mutates value_info and output types in place, so
  // snapshot those annotations and restore them on the paths that fold nothing,
  // leaving the model byte-for-byte unchanged (the old code returned the
  // untouched input there). The snapshot is metadata only -- no tensor weights
  // -- so it is cheap, unlike the full-model ``CopyFrom`` it replaces.
  // Restoring also keeps this pass's data-propagation value_info out of the
  // model, which matters: it differs from the regular shape-inference pass's
  // value_info, and leaving it behind could make the outer fixed point
  // oscillate.
  auto saved_value_info = model.graph().value_info();
  auto saved_output = model.graph().output();
  auto restore = [&]() {
    *model.mutable_graph()->mutable_value_info() = saved_value_info;
    *model.mutable_graph()->mutable_output() = saved_output;
  };

  onnx::shape_inference::DataValueMap data_map;
  try {
    const onnx::ShapeInferenceOptions options(/*check_type=*/false,
                                              /*error_mode=*/0,
                                              /*enable_data_propagation=*/true);
    onnx::shape_inference::InferShapes(
        model, onnx::OpSchemaRegistry::Instance(), options, &data_map);
  } catch (const std::exception&) {
    // If shape inference fails we simply have no propagated values to exploit.
    restore();
    return;
  }

  // An empty data_map is not a dead end anymore: the native symbolic evaluator
  // (issue #532) runs further below and can resolve chains ONNX data
  // propagation could not, so fall through instead of returning. The loops over
  // data_map simply add nothing, and the final `folded_values && reshape_fixes`
  // empty check restores the model if the symbolic pass also finds nothing.

  const auto type_map = BuildTypeMap(model);

  // Maps the output of a foldable node to the constant tensor it produces. Each
  // such node is rewritten into a `Constant` node holding this value.
  std::unordered_map<std::string, onnx::TensorProto> folded_values;

  for (const auto& node : model.graph().node()) {
    // Shape-family ops are single-output; only replace a node when its lone
    // output is fully known, so dropping it can never orphan a second output.
    if (node.output_size() != 1) {
      continue;
    }
    const std::string& output = node.output(0);
    auto data_iter = data_map.find(output);
    if (data_iter == data_map.end()) {
      continue;
    }

    // Every element must be statically known. Data propagation represents an
    // unknown element as a dimension with neither dim_value nor dim_param, so
    // requiring dim_value on every entry both proves the value is concrete and
    // filters out activations whose rank alone is known.
    const onnx::TensorShapeProto& value = data_iter->second;
    bool fully_known = true;
    std::vector<int64_t> values;
    for (const auto& dim : value.dim()) {
      if (!dim.has_dim_value()) {
        fully_known = false;
        break;
      }
      values.push_back(dim.dim_value());
    }
    if (!fully_known) {
      continue;
    }

    // Build the constant tensor with the output's real dtype and shape. The
    // propagated data is a flat sequence, so require a fully static shape whose
    // element count matches what was propagated.
    onnx::TensorProto::DataType elem_type;
    std::vector<int64_t> dims;
    if (!GetStaticIntTensorInfo(type_map, output, elem_type, dims)) {
      continue;
    }
    int64_t element_count = 1;
    for (int64_t d : dims) {
      element_count *= d;
    }
    if (element_count != static_cast<int64_t>(values.size())) {
      continue;
    }

    onnx::TensorProto tp;
    tp.set_data_type(elem_type);
    for (int64_t d : dims) {
      tp.add_dims(d);
    }
    if (elem_type == onnx::TensorProto::INT64) {
      for (int64_t v : values) {
        tp.add_int64_data(v);
      }
    } else {
      for (int64_t v : values) {
        tp.add_int32_data(static_cast<int32_t>(v));
      }
    }
    folded_values.emplace(output, std::move(tp));
  }

  // Data propagation for `Reshape` (single dynamic dim). The fully-known folder
  // above only rewrites a node whose propagated value is entirely concrete, so
  // a shape tensor that keeps one symbolic entry -- e.g. `[batch, 1024, 128]`
  // on a graph with a dynamic batch, or `[?, 768]` with a dynamic sequence
  // length -- is left alone, and with it the whole `Shape -> Gather -> Concat`
  // subgraph that computes it at runtime. Those single-dynamic-dim reshapes
  // dominate transformer and speech graphs.
  //
  // When a Reshape's shape input propagates to a value with exactly one unknown
  // entry and all other entries positive constants, materialize the shape as a
  // constant with the unknown slot set to -1. ONNX Reshape infers the -1 dim
  // from the total element count, so for every input the result is identical to
  // the runtime-computed shape, while the shape-producing subgraph becomes dead
  // and is removed by the optimizer. (Correctness is still gated by onnxsim's
  // own equivalence check.)
  struct ReshapeShapeFix {
    std::string shape_name;
    onnx::TensorProto shape_tensor;
  };
  std::unordered_map<std::string, ReshapeShapeFix> reshape_fixes;
  for (const auto& node : model.graph().node()) {
    if (node.op_type() != "Reshape" || node.input_size() < 2 ||
        node.output_size() != 1) {
      continue;
    }
    auto data_iter = data_map.find(node.input(1));
    if (data_iter == data_map.end()) {
      continue;
    }
    const onnx::TensorShapeProto& shape_value = data_iter->second;
    if (shape_value.dim_size() == 0) {
      continue;
    }
    int unknown = 0;
    bool usable = true;
    std::vector<int64_t> shape;
    shape.reserve(shape_value.dim_size());
    for (const auto& dim : shape_value.dim()) {
      if (dim.has_dim_value()) {
        // A non-positive entry is a literal 0 (copy-dim) or an already
        // materialized -1; leave those for the ordinary folder.
        if (dim.dim_value() <= 0) {
          usable = false;
          break;
        }
        shape.push_back(dim.dim_value());
      } else {
        ++unknown;
        shape.push_back(-1);
      }
    }
    // Exactly one unknown dim maps to Reshape's single -1 sentinel. Zero
    // unknowns is handled by the fully-known folder above; two or more cannot
    // be expressed with a single -1.
    if (!usable || unknown != 1) {
      continue;
    }
    onnx::TensorProto tp;
    tp.set_data_type(onnx::TensorProto::INT64);
    tp.add_dims(static_cast<int64_t>(shape.size()));
    for (int64_t v : shape) {
      tp.add_int64_data(v);
    }
    reshape_fixes.emplace(
        node.output(0),
        ReshapeShapeFix{node.output(0) + "_dp_shape", std::move(tp)});
  }

  // Native symbolic evaluation (issue #532). ONNX data propagation above stops
  // wherever the shape algebra crosses a dynamic-dim symbol; the SymExpr
  // evaluator resolves those chains. Merge whatever it finds that the ONNX path
  // did not already cover into the same two rewrite maps, so the shared rewrite
  // loop below handles both uniformly. Correctness stays gated by check_n.
  {
    const auto sym_values = EvaluateModelSymbolicValues(model);
    // Fully concrete symbolic values fold to a `Constant`, exactly like the
    // ONNX fully-known folder above (same dtype/shape/element-count checks).
    for (const auto& node : model.graph().node()) {
      if (node.output_size() != 1) continue;
      const std::string& output = node.output(0);
      if (folded_values.count(output) || reshape_fixes.count(output)) continue;
      auto sym_iter = sym_values.find(output);
      if (sym_iter == sym_values.end()) continue;
      std::vector<int64_t> flat;
      bool all_concrete = true;
      for (const auto& e : sym_iter->second.data) {
        if (e.is_symbolic()) {
          all_concrete = false;
          break;
        }
        flat.push_back(e.to_int());
      }
      if (!all_concrete) continue;
      onnx::TensorProto::DataType elem_type;
      std::vector<int64_t> dims;
      if (!GetStaticIntTensorInfo(type_map, output, elem_type, dims)) continue;
      int64_t element_count = 1;
      for (int64_t d : dims) element_count *= d;
      if (element_count != static_cast<int64_t>(flat.size())) continue;
      onnx::TensorProto tp;
      tp.set_data_type(elem_type);
      for (int64_t d : dims) tp.add_dims(d);
      if (elem_type == onnx::TensorProto::INT64) {
        for (int64_t v : flat) tp.add_int64_data(v);
      } else {
        for (int64_t v : flat) tp.add_int32_data(static_cast<int32_t>(v));
      }
      folded_values.emplace(output, std::move(tp));
    }
    // A Reshape whose target has exactly one symbolic entry (plus positive
    // constants) becomes `[-1, ...]` -- the same rewrite as the ONNX data-prop
    // Reshape path above, but reached through SymExpr arithmetic.
    for (const auto& node : model.graph().node()) {
      if (node.op_type() != "Reshape" || node.input_size() < 2 ||
          node.output_size() != 1) {
        continue;
      }
      if (reshape_fixes.count(node.output(0))) continue;
      auto sym_iter = sym_values.find(node.input(1));
      if (sym_iter == sym_values.end()) continue;
      const onnxsim::SymTensor& shape_value = sym_iter->second;
      if (shape_value.scalar || shape_value.data.empty()) continue;
      int unknown = 0;
      bool usable = true;
      std::vector<int64_t> shape;
      shape.reserve(shape_value.data.size());
      for (const auto& e : shape_value.data) {
        if (e.is_symbolic()) {
          ++unknown;
          shape.push_back(-1);
        } else {
          const int64_t v = e.to_int();
          if (v <=
              0) {  // a literal 0 (copy) or -1 is left for the ordinary folder
            usable = false;
            break;
          }
          shape.push_back(v);
        }
      }
      if (!usable || unknown != 1) continue;
      onnx::TensorProto tp;
      tp.set_data_type(onnx::TensorProto::INT64);
      tp.add_dims(static_cast<int64_t>(shape.size()));
      for (int64_t v : shape) tp.add_int64_data(v);
      reshape_fixes.emplace(
          node.output(0),
          ReshapeShapeFix{node.output(0) + "_sym_shape", std::move(tp)});
    }
  }

  if (folded_values.empty() && reshape_fixes.empty()) {
    restore();
    return;
  }

  // Rewrite each foldable node into a `Constant` node in the same position,
  // keeping the graph topologically sorted. Emitting a `Constant` node (rather
  // than injecting an initializer) leaves the value in producer form, so the
  // ordinary constant folder and optimizer decide how to materialize it.
  google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
  original_nodes.Swap(model.mutable_graph()->mutable_node());
  for (auto& node : original_nodes) {
    auto iter = node.output_size() == 1 ? folded_values.find(node.output(0))
                                        : folded_values.end();
    if (iter != folded_values.end()) {
      onnx::NodeProto* constant = model.mutable_graph()->add_node();
      constant->set_name(node.name());
      constant->set_op_type("Constant");
      constant->add_output(iter->first);
      onnx::AttributeProto* attr = constant->add_attribute();
      attr->set_name("value");
      attr->set_type(onnx::AttributeProto::TENSOR);
      *attr->mutable_t() = std::move(iter->second);
      continue;
    }
    auto fix_iter = node.output_size() == 1 ? reshape_fixes.find(node.output(0))
                                            : reshape_fixes.end();
    if (fix_iter != reshape_fixes.end()) {
      // Emit the materialized shape as a Constant just before the Reshape
      // (preserving topological order), then repoint the Reshape's shape input
      // at it. The original shape-producing subgraph is now unused and is
      // cleaned up by the optimizer's dead-node elimination.
      onnx::NodeProto* shape_const = model.mutable_graph()->add_node();
      shape_const->set_op_type("Constant");
      shape_const->add_output(fix_iter->second.shape_name);
      onnx::AttributeProto* attr = shape_const->add_attribute();
      attr->set_name("value");
      attr->set_type(onnx::AttributeProto::TENSOR);
      *attr->mutable_t() = std::move(fix_iter->second.shape_tensor);

      onnx::NodeProto* reshape = model.mutable_graph()->add_node();
      *reshape = std::move(node);
      reshape->set_input(1, fix_iter->second.shape_name);
      continue;
    }
    *model.mutable_graph()->add_node() = std::move(node);
  }
}

// Whether every element of `tensor` is zero. Only the storage forms that can be
// inspected locally are accepted: a tensor whose data lives in an external file
// is reported as "not provably zero" rather than loaded.
bool IsAllZeroTensor(const onnx::Tensor& tensor) {
  if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    return false;
  }
  if (tensor.elem_type() == onnx::TensorProto::STRING ||
      tensor.elem_type() == onnx::TensorProto::UNDEFINED) {
    return false;
  }
  if (tensor.is_raw_data()) {
    const std::string& raw = tensor.raw();
    return std::all_of(raw.begin(), raw.end(), [](char c) { return c == 0; });
  }
  auto all_zero = [](const auto& field) {
    return std::all_of(field.begin(), field.end(),
                       [](auto value) { return value == 0; });
  };
  const size_t element_count =
      tensor.floats().size() + tensor.int32s().size() + tensor.int64s().size() +
      tensor.doubles().size() + tensor.uint64s().size();
  if (element_count == 0) {
    return false;
  }
  return all_zero(tensor.floats()) && all_zero(tensor.int32s()) &&
         all_zero(tensor.int64s()) && all_zero(tensor.doubles()) &&
         all_zero(tensor.uint64s());
}

// Whether `value` is provably an all-zero tensor, following the chain of ops
// that produced it, walking onnx::Value/Node. Only shape-manipulating ops are
// traversed: they move elements around without changing their values, so an
// all-zero input implies an all-zero output. Ops whose output could be empty
// (Slice, Split, Gather) stay sound too, since an empty tensor is vacuously
// all zeros. ``value``'s producer being the graph's single kUndefined
// placeholder node (see ir_pb_converter.cc) means "this optional input was
// not provided".
bool IsAllZeroGraphValue(
    onnx::Value* value,
    const std::unordered_map<std::string, const onnx::Tensor*>&
        initializer_by_name,
    std::unordered_map<const onnx::Value*, bool>& memo) {
  if (value->node()->kind() == onnx::kUndefined) {
    return false;
  }
  auto memo_iter = memo.find(value);
  if (memo_iter != memo.end()) {
    return memo_iter->second;
  }
  // Insert a pessimistic answer up front: it both memoizes the miss and
  // breaks any cycle a malformed graph might contain.
  memo.emplace(value, false);

  auto init_iter = initializer_by_name.find(value->uniqueName());
  if (init_iter != initializer_by_name.end()) {
    const bool result = IsAllZeroTensor(*init_iter->second);
    memo[value] = result;
    return result;
  }

  onnx::Node* producer = value->node();
  if (producer->has_domain() && !producer->domain().empty() &&
      producer->domain() != "ai.onnx") {
    return false;
  }

  static const onnx::Symbol kConstant("Constant");
  static const onnx::Symbol kConstantOfShape("ConstantOfShape");
  static const onnx::Symbol kCast("Cast");
  static const onnx::Symbol kCastLike("CastLike");
  static const onnx::Symbol kIdentity("Identity");
  static const onnx::Symbol kReshape("Reshape");
  static const onnx::Symbol kTranspose("Transpose");
  static const onnx::Symbol kSqueeze("Squeeze");
  static const onnx::Symbol kUnsqueeze("Unsqueeze");
  static const onnx::Symbol kFlatten("Flatten");
  static const onnx::Symbol kTile("Tile");
  static const onnx::Symbol kExpand("Expand");
  static const onnx::Symbol kSlice("Slice");
  static const onnx::Symbol kSplit("Split");
  static const onnx::Symbol kGather("Gather");
  static const onnx::Symbol kGatherElements("GatherElements");
  static const onnx::Symbol kGatherND("GatherND");
  static const onnx::Symbol kConcat("Concat");
  static const onnx::Symbol kValue("value");
  static const onnx::Symbol kValueFloat("value_float");
  static const onnx::Symbol kValueInt("value_int");
  static const onnx::Symbol kValueFloats("value_floats");
  static const onnx::Symbol kValueInts("value_ints");
  static const onnx::Symbol kTo("to");

  bool result = false;
  const onnx::Symbol kind = producer->kind();
  if (kind == kConstant) {
    if (producer->hasAttribute(kValue)) {
      result = IsAllZeroTensor(producer->t(kValue));
    } else if (producer->hasAttribute(kValueFloat)) {
      result = producer->f(kValueFloat) == 0;
    } else if (producer->hasAttribute(kValueInt)) {
      result = producer->i(kValueInt) == 0;
    } else if (producer->hasAttribute(kValueFloats)) {
      const auto& floats = producer->fs(kValueFloats);
      result = !floats.empty() && std::all_of(floats.begin(), floats.end(),
                                              [](double f) { return f == 0; });
    } else if (producer->hasAttribute(kValueInts)) {
      const auto& ints = producer->is(kValueInts);
      result = !ints.empty() && std::all_of(ints.begin(), ints.end(),
                                            [](int64_t i) { return i == 0; });
    }
  } else if (kind == kConstantOfShape) {
    result =
        !producer->hasAttribute(kValue) || IsAllZeroTensor(producer->t(kValue));
  } else if (kind == kCast || kind == kCastLike) {
    const bool to_string = producer->hasAttribute(kTo) &&
                           producer->i(kTo) == onnx::TensorProto::STRING;
    result = !to_string && IsAllZeroGraphValue(producer->inputs()[0],
                                               initializer_by_name, memo);
  } else if (kind == kIdentity || kind == kReshape || kind == kTranspose ||
             kind == kSqueeze || kind == kUnsqueeze || kind == kFlatten ||
             kind == kTile || kind == kExpand || kind == kSlice ||
             kind == kSplit || kind == kGather || kind == kGatherElements ||
             kind == kGatherND) {
    result =
        IsAllZeroGraphValue(producer->inputs()[0], initializer_by_name, memo);
  } else if (kind == kConcat) {
    const auto& inputs = producer->inputs();
    result = !inputs.empty() &&
             std::all_of(inputs.begin(), inputs.end(), [&](onnx::Value* input) {
               return IsAllZeroGraphValue(input, initializer_by_name, memo);
             });
  }

  memo[value] = result;
  return result;
}

// The graph's single placeholder Value standing in for "this optional input
// was not provided" (see ir_pb_converter.cc) -- every kUndefined-producer
// input aliases this same Value. Only scanned when actually needed (an
// all-zero RNN initial state was found), so the common case of a graph with
// no LSTM/RNN/GRU nodes pays nothing for it.
onnx::Value* FindUndefinedGraphValue(onnx::Graph& graph) {
  for (onnx::Node* node : graph.nodes()) {
    if (node->kind() == onnx::kUndefined) {
      return node->outputs()[0];
    }
  }
  return nullptr;
}

// Unset the recurrent initial states of RNN/GRU/LSTM nodes that are provably
// all zeros (issue #314).
//
// paddle2onnx (like several other converters) materializes the zero initial
// hidden/cell state of an LSTM as a *batch-dependent* subgraph, because the
// state's shape is [num_directions, batch_size, hidden_size]:
//   Shape(x) -> Slice -> Concat([batch,1,1]) -> Tile(zeros) -> Transpose
//            -> Slice -> LSTM(initial_h, initial_c)
// When the model has a dynamic batch dimension none of that can be constant
// folded, so the simplified model keeps a Shape/Slice/Concat/Tile chain that
// downstream converters (onnx2ncnn in the issue) reject outright.
//
// The ONNX spec says initial_h/initial_c default to zero when omitted, so an
// input that is provably all zeros can simply be unset. The subgraph feeding it
// then becomes dead and is removed by the ordinary dead-end elimination pass.
// Only the initial states are unset; the equally zero-defaulting B and P inputs
// are left alone because they are plain initializers, so dropping them removes
// no operator while risking a needless behaviour change in consumers that read
// them.
void EliminateZeroRnnInitialState(onnx::Graph& graph) {
  std::unordered_map<std::string, const onnx::Tensor*> initializer_by_name;
  const auto& initializers = graph.initializers();
  const auto& initializer_names = graph.initializer_names();
  initializer_by_name.reserve(initializers.size());
  for (size_t i = 0; i < initializers.size(); ++i) {
    initializer_by_name[initializer_names[i]] = initializers[i].get();
  }
  std::unordered_map<const onnx::Value*, bool> memo;
  onnx::Value* undefined = nullptr;

  static const onnx::Symbol kLSTM("LSTM");
  static const onnx::Symbol kRNN("RNN");
  static const onnx::Symbol kGRU("GRU");

  for (onnx::Node* node : graph.nodes()) {
    // Recurse first, so recurrent ops inside If/Loop/Scan bodies are handled
    // too.
    for (onnx::Symbol attr : node->attributeNames()) {
      if (node->kindOf(attr) == onnx::AttributeKind::g) {
        EliminateZeroRnnInitialState(*node->g(attr));
      } else if (node->kindOf(attr) == onnx::AttributeKind::gs) {
        for (const auto& subgraph : node->gs(attr)) {
          EliminateZeroRnnInitialState(*subgraph);
        }
      }
    }

    if (node->has_domain() && !node->domain().empty() &&
        node->domain() != "ai.onnx") {
      continue;
    }
    const onnx::Symbol kind = node->kind();
    int last_state_index;
    if (kind == kLSTM) {
      last_state_index = 6;
    } else if (kind == kRNN || kind == kGRU) {
      last_state_index = 5;
    } else {
      continue;
    }

    const auto& inputs = node->inputs();
    for (int i = 5;
         i <= last_state_index && i < static_cast<int>(inputs.size()); i++) {
      if (IsAllZeroGraphValue(inputs[i], initializer_by_name, memo)) {
        if (undefined == nullptr) {
          undefined = FindUndefinedGraphValue(graph);
        }
        node->replaceInput(i, undefined);
      }
    }
    // Trailing empty inputs carry no information; drop them so the node ends
    // up in the same shape a converter would have emitted without the state.
    while (!node->inputs().empty() &&
           node->inputs().back()->node()->kind() == onnx::kUndefined) {
      node->removeInput(node->inputs().size() - 1);
    }
  }
}

// Estimate the number of bytes the outputs of `node` occupy once materialized,
// using shapes gathered by shape inference (`vi_map` maps a tensor name to its
// value_info). Outputs whose dtype or shape is not fully known contribute
// nothing; the caller falls back to a node-count budget to stay bounded when no
// size information is available.
size_t EstimateOutputBytes(
    const std::unordered_map<std::string, const onnx::ValueInfoProto*>& vi_map,
    const onnx::NodeProto& node) {
  size_t total = 0;
  for (const auto& output : node.output()) {
    auto iter = vi_map.find(output);
    if (iter == vi_map.end() || !iter->second->type().has_tensor_type()) {
      continue;
    }
    const auto& tensor_type = iter->second->type().tensor_type();
    if (tensor_type.elem_type() == onnx::TensorProto::UNDEFINED) {
      continue;
    }
    if (!tensor_type.has_shape()) {
      continue;
    }
    size_t size;
    try {
      size = size_of_dtype(
          static_cast<onnx::TensorProto::DataType>(tensor_type.elem_type()));
    } catch (const std::exception&) {
      // Unknown dtype: treat as unsized and rely on the node-count budget.
      continue;
    }
    bool known = true;
    for (const auto& dim : tensor_type.shape().dim()) {
      if (!dim.has_dim_value()) {
        known = false;
        break;
      }
      size *= dim.dim_value();
    }
    if (known) {
      total += size;
    }
  }
  return total;
}

// Fold the const nodes in `const_nodes[begin, end)` as a single batch,
// appending the resulting initializers to `model` and recording the folded
// output names in `folded_outputs`. On failure the batch is bisected and each
// half retried, so a single un-runnable node does not stop the rest of the
// group from folding; the lower half is folded first and adds its initializers,
// so the upper half can still read any values it depends on. A batch of one
// that fails is skipped with a warning, matching the original per-node
// behaviour.
void FoldGroup(const ModelExecutor& executor, onnx::ModelProto& model,
               const std::vector<onnx::NodeProto>& const_nodes, size_t begin,
               size_t end,
               const std::unordered_map<std::string, const onnx::NodeProto*>&
                   deferred_producers,
               std::set<std::string>& folded_outputs) {
  if (begin >= end) {
    return;
  }
  std::vector<const onnx::NodeProto*> ops;
  ops.reserve(end - begin);
  for (size_t k = begin; k < end; k++) {
    ops.push_back(&const_nodes[k]);
  }
  try {
    RunOpsAndAddInitializers(executor, model, ops, deferred_producers);
    for (size_t k = begin; k < end; k++) {
      for (const auto& output : const_nodes[k].output()) {
        folded_outputs.insert(output);
      }
    }
  } catch (const std::exception& e) {
    if (end - begin == 1) {
      const auto& x = const_nodes[begin];
      std::cerr << "WARNING: failed to run \"" << x.op_type()
                << "\" op (name is \"" << x.name() << "\"), skip... "
                << e.what() << std::endl;
      return;
    }
    const size_t mid = begin + (end - begin) / 2;
    FoldGroup(executor, model, const_nodes, begin, mid, deferred_producers,
              folded_outputs);
    FoldGroup(executor, model, const_nodes, mid, end, deferred_producers,
              folded_outputs);
  }
}

// Takes `model` by value rather than `const&`: both call sites pass an
// rvalue (std::move of a local they immediately overwrite with the return
// value), so this is a cheap move-construction -- a pointer/buffer swap --
// not a deep copy of the model's initializer bytes. `model` is then this
// function's own uniquely-owned working copy, mutated in place throughout.
onnx::ModelProto _FoldConstant(const ModelExecutor& executor,
                               onnx::ModelProto model) {
  ConstantNodePartition partition;
  {
    onnxsim::ProfiledScope analysis_scope("GetConstantNodes");
    partition = GetConstantNodes(model);
  }
  const auto& const_nodes = partition.const_nodes;
  // Map each deferred node's output to the producing node so RunOps can
  // inline it into the sub-model executed when folding a downstream consumer.
  // The pointers stay valid for the loop below: folding only appends
  // initializers to the graph and never touches its node list.
  std::unordered_map<std::string, const onnx::NodeProto*> deferred_producers;
  if (!partition.deferred_outputs.empty()) {
    for (const auto& node : model.graph().node()) {
      for (const auto& output : node.output()) {
        if (partition.deferred_outputs.count(output) > 0) {
          deferred_producers.emplace(output, &node);
        }
      }
    }
  }
  // Look up each tensor's inferred shape so batches can be capped by the
  // bytes they would materialize (see below). Pointers reference `model`,
  // which is not mutated (only appended to) while the map is in use.
  std::unordered_map<std::string, const onnx::ValueInfoProto*> vi_map;
  for (const auto& vi : model.graph().value_info()) {
    vi_map[vi.name()] = &vi;
  }
  // Fold the const nodes in batches: one Session per batch instead of one per
  // node. `const_nodes` is topologically sorted, so a batch is a contiguous
  // slice and a later batch reads any earlier batch's outputs as ordinary
  // initializers. Two budgets bound ORT's peak memory: a batch is closed once
  // its outputs would exceed kBatchByteBudget or it reaches kBatchMaxNodes.
  // Nodes that consume a deferred (large-tensor) output are folded on their
  // own so the large intermediate is materialized transiently for just that
  // node, exactly as in the per-node path.
  constexpr size_t kBatchByteBudget = size_t(256) << 20;  // 256 MiB
  constexpr size_t kBatchMaxNodes = 1024;
  auto consumes_deferred = [&](const onnx::NodeProto& node) {
    if (partition.deferred_outputs.empty()) {
      return false;
    }
    for (const auto& input : node.input()) {
      if (partition.deferred_outputs.count(input) > 0) {
        return true;
      }
    }
    return false;
  };
  // Outputs of const nodes that were successfully folded into initializers.
  std::set<std::string> folded_outputs;
  const size_t num_const_nodes = const_nodes.size();
  for (size_t i = 0; i < num_const_nodes;) {
    if (consumes_deferred(const_nodes[i])) {
      FoldGroup(executor, model, const_nodes, i, i + 1, deferred_producers,
                folded_outputs);
      i++;
      continue;
    }
    size_t j = i;
    size_t bytes = 0;
    while (j < num_const_nodes && j - i < kBatchMaxNodes &&
           !consumes_deferred(const_nodes[j])) {
      const size_t node_bytes = EstimateOutputBytes(vi_map, const_nodes[j]);
      if (j > i && bytes + node_bytes > kBatchByteBudget) {
        break;
      }
      bytes += node_bytes;
      j++;
    }
    FoldGroup(executor, model, const_nodes, i, j, deferred_producers,
              folded_outputs);
    i = j;
  }
  // Rebuild the node list in its original topological order, dropping only
  // the const nodes that were successfully folded into initializers. A const
  // node that failed to fold must keep its original position: appending it to
  // the end can place it after a non-const consumer (e.g. a Loop reading a
  // SequenceEmpty output), which breaks topological sorting and makes the
  // resulting model fail onnx's checker (issues #238, #335, #352).
  {
    onnxsim::ProfiledScope rebuild_scope("RebuildNodeList");
    google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
    original_nodes.Swap(model.mutable_graph()->mutable_node());
    for (auto& node : original_nodes) {
      const bool folded =
          node.output_size() > 0 && folded_outputs.count(node.output(0)) > 0;
      if (!folded) {
        *model.mutable_graph()->add_node() = std::move(node);
      }
    }
  }
  // Drop initializers left dangling by folding so the intermediate model does
  // not balloon in size (issue #174).
  onnxsim::ProfiledScope eliminate_scope("EliminateUnusedInitializerScope");
  return EliminateUnusedInitializer(std::move(model));
}

// ``model`` is ``onnx::ModelProto&`` (not const): the call site below passes a
// mutable lvalue that is about to be move-assigned over
// (``model = Optimize(model)``), so its pre-call contents are dead once this
// returns. That lets OptimizeFixed move each initializer's raw bytes through
// the ModelProto<->Graph round trip instead of copying them -- the dominant
// cost of this call for weight-heavy models (onnxsim issue #633) -- via the
// moving ImportModelProto/ExportModelProto overloads added to onnxsim's own
// onnx fork.
onnx::ModelProto Optimize(onnx::ModelProto& model) {
  // Make onnxsim's own optimizer passes available to onnxoptimizer's registry
  // (idempotent) so config.optimizer_passes may name them.
  onnxsim::RegisterCustomOptimizerPasses();
  // Mirror the initializer treatment into the onnx optimizer so its
  // value-baking passes (fuse_bn_into_conv, ...) respect it too. The setting is
  // thread-local in the optimizer; restore it afterwards so we do not leak it.
  const bool prev = onnx::optimization::InitializersAsConstants();
  onnx::optimization::SetInitializersAsConstants(
      config.initializers_as_constants);
  auto result =
      onnx::optimization::OptimizeFixed(model, config.optimizer_passes);
  onnx::optimization::SetInitializersAsConstants(prev);
  return result;
}

// A 128-bit fingerprint of a model, used by FixedPointFn to detect when an
// iteration stopped changing the model without keeping a second full ModelProto
// around just for the comparison. Two models with the same fingerprint are
// treated as equal; the odds of a false match are ~2^-128 per comparison, and a
// false match would only stop simplification one round early (the model stays
// valid), never produce an incorrect model.
struct ModelFingerprint {
  uint64_t h1;
  uint64_t h2;
  bool operator==(const ModelFingerprint& other) const {
    return h1 == other.h1 && h2 == other.h2;
  }
};

// Mixes ``n`` bytes at ``data`` into the two rolling hash accumulators (FNV-1a
// and a splitmix-style mix), 8 bytes (one machine word) at a time rather than
// one byte at a time -- cutting the number of loop iterations, and with them
// most of the loop/branch overhead, by roughly 8x versus a byte-at-a-time
// mix, for identical byte coverage and order.
void MixBytes(const char* data, size_t n, uint64_t& h1, uint64_t& h2) {
  const size_t n_words = n / sizeof(uint64_t);
  for (size_t i = 0; i < n_words; ++i) {
    // memcpy, not a reinterpret_cast dereference: ``data`` is not guaranteed
    // 8-byte aligned (it may be a chunk of a streaming buffer, or a
    // std::string's internal buffer), and an unaligned uint64_t load is
    // undefined behavior in C++. Any decent compiler lowers this fixed-size
    // memcpy to a single unaligned load.
    uint64_t word;
    std::memcpy(&word, data + i * sizeof(uint64_t), sizeof(uint64_t));
    h1 = (h1 ^ word) * 1099511628211ULL;
    h2 = (h2 + word) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
  // Fewer than 8 trailing bytes: fall back to a byte-at-a-time mix so every
  // byte still participates in the hash.
  for (size_t i = n_words * sizeof(uint64_t); i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(data[i]);
    h1 = (h1 ^ c) * 1099511628211ULL;
    h2 = (h2 + c) * 0x9E3779B97F4A7C15ULL;
    h2 ^= h2 >> 29;
  }
}

// A protobuf output stream that hashes bytes as they are handed out by the
// serializer instead of collecting them into a buffer. Fingerprint() used to
// call ``model.SerializeAsString()`` -- which allocates a buffer the size of
// the whole serialized model (hundreds of MB of initializer weight data on a
// large model) and serializes into it -- and then made a second, separate
// pass over that buffer to hash it. Handing the serializer a small, reused
// scratch buffer instead avoids that large allocation (and the deallocation
// moments later): each time the serializer fills it, this stream
// mixes it into the running hash and hands the same buffer back out, so the
// whole model is hashed in bounded extra memory regardless of its size.
class HashingOutputStream final
    : public google::protobuf::io::ZeroCopyOutputStream {
 public:
  bool Next(void** data, int* size) override {
    FlushPending();
    *data = buf_.data();
    *size = static_cast<int>(buf_.size());
    pending_ = static_cast<int>(buf_.size());
    return true;
  }
  void BackUp(int count) override { pending_ -= count; }
  int64_t ByteCount() const override { return byte_count_; }

  // Call after serialization completes to mix in the final partial buffer.
  void Finish() { FlushPending(); }

  uint64_t h1() const { return h1_; }
  uint64_t h2() const { return h2_; }
  size_t total_bytes() const { return static_cast<size_t>(byte_count_); }

 private:
  void FlushPending() {
    if (pending_ <= 0) {
      return;
    }
    MixBytes(buf_.data(), static_cast<size_t>(pending_), h1_, h2_);
    byte_count_ += pending_;
    pending_ = 0;
  }

  // 64 KiB: large enough that the per-Next() call overhead is negligible next
  // to the memory-bandwidth-bound hashing work, small enough to stay resident
  // in L1/L2 cache across the mix. Heap-allocated, not a fixed-size array
  // member: this object lives on the call stack of Fingerprint(), which is
  // itself invoked from deep inside the (possibly nested) fixed-point
  // machinery, and the WASM build's stack is only tens of KiB total -- a
  // 64 KiB stack-resident array here reliably overflowed it (issue caught by
  // the WASM build's own smoke test, "Aborted(stack overflow ...)" on the
  // very first Simplify() call). std::vector's control block is a few
  // pointer-sized words on the stack; the buffer itself is heap-backed.
  std::vector<char> buf_ = std::vector<char>(1 << 16);
  int pending_ = 0;
  int64_t byte_count_ = 0;
  uint64_t h1_ = 1469598103934665603ULL;  // FNV-1a offset basis
  uint64_t h2_ = 0;
};

ModelFingerprint Fingerprint(const onnx::ModelProto& model) {
  // ModelProto contains no protobuf ``map<>`` fields, so serialization order is
  // stable and equal models serialize to identical bytes.
  //
  // Fingerprint() runs at every step of every (possibly nested) fixed point,
  // so on a model with hundreds of MB of initializer weight data -- which
  // dominates the serialized size but is untouched by most rounds (shape
  // inference, the onnx-optimizer passes) -- it still has to read all of that
  // data, every round, just to prove nothing changed. Profiling a 265 MB
  // model showed this was a multi-second fraction (over half of total wall
  // time on that model) of total simplification time, so it is worth
  // streaming through a small buffer (see HashingOutputStream) rather than
  // materializing the whole serialized model first.
  HashingOutputStream stream;
  // SerializeToZeroCopyStream only returns false if a Next() call on the
  // stream fails (out of disk space, a broken pipe, ...); HashingOutputStream
  // writes to memory and its Next() always succeeds, so this cannot fail.
  model.SerializeToZeroCopyStream(&stream);
  stream.Finish();
  uint64_t h1 = stream.h1();
  uint64_t h2 = stream.h2();
  h2 ^= stream.total_bytes();
  return {h1, h2};
}

// Alternately apply ``f1`` and ``f2`` until the model stops changing (a joint
// fixed point) or ``max_iters`` alternations elapse. Each application produces
// a fresh model, so ``model`` is move-assigned in place and only a single
// ModelProto is held live across the loop; convergence is detected by comparing
// the fingerprints of consecutive states rather than keeping the previous
// ModelProto for a ``MessageDifferencer::Equals`` call. This mirrors the
// original consecutive-pair comparison exactly -- it stops as soon as the last
// applied function left the model unchanged -- while roughly halving the number
// of full model copies held at once (which matters because these fixed points
// nest).
// The transforms mutate the model in place (``std::function<void(T&)>``), so a
// transform that already works in place (e.g. ``_InferShapes``) makes no copy
// at all, and one that must build a fresh model (e.g. ``Optimize``, whose
// underlying ``OptimizeFixed`` returns a new proto) move-assigns it back. The
// returned function likewise mutates in place, so it composes when these fixed
// points nest and a single ModelProto is threaded through the whole thing.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    // Profiled separately from the transforms it gates: on large models this
    // convergence check is not free (see Fingerprint()'s comment), so its cost
    // should be visible in its own right rather than silently inflating
    // whichever of f1/f2's spans happens to run next. A no-op unless
    // ONNXSIM_PROFILE is set.
    auto fingerprint = [](const onnx::ModelProto& m) {
      onnxsim::ProfiledScope scope("Fingerprint");
      return Fingerprint(m);
    };
    size_t _max_iters = max_iters;
    f1(model);
    ModelFingerprint fp_prev = fingerprint(model);
    f2(model);
    ModelFingerprint fp_cur = fingerprint(model);
    while (_max_iters-- > 0) {
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f1(model);
      fp_prev = fp_cur;
      fp_cur = fingerprint(model);
      if (fp_cur == fp_prev) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      f2(model);
      fp_prev = fp_cur;
      fp_cur = fingerprint(model);
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<void(T&)>& f1,
                                     const std::function<void(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// Same convergence algorithm as the ``Fingerprint``-based ``FixedPointFn``
// above, specialized for transforms that can cheaply report whether they
// changed the model themselves (``f1``/``f2`` return true iff they did),
// instead of hashing the whole serialized model after every call. This skips
// ``Fingerprint()`` entirely, which matters most here because this is the
// innermost, most-frequently-run fixed point (``OptAndShape`` below runs
// every simplification round). It is only as safe as ``f1``/``f2``'s own
// signal: both onnx-optimizer's per-pass transform counts and onnx's
// InferShapes value-change count are exact for onnxsim's pass list (no
// pass with an ``Empty``/uncounted analysis type is used), so a ``false``
// return means that call provably made no change -- not just "probably".
// A false negative here (missing a real change) would still not produce an
// incorrect model: it only makes this inner loop stop one round early, and
// the fingerprint-based fixed point one level up (which wraps every call
// to ``OptAndShape`` as a whole) re-drives it from the top if that whole
// call's net effect changed anything, same as a ``Fingerprint`` hash
// collision would.
template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<bool(T&)>& f1,
                                     const std::function<bool(T&)>& f2,
                                     size_t max_iters, bool* converged) {
  return [f1, f2, max_iters, converged](T& model) -> void {
    size_t _max_iters = max_iters;
    f1(model);
    bool last_changed = f2(model);
    while (_max_iters-- > 0) {
      if (!last_changed) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      last_changed = f1(model);
      if (!last_changed) {
        if (converged) {
          *converged = true;
        }
        return;
      }
      last_changed = f2(model);
    }

    if (converged) {
      *converged = false;
    }
  };
}

template <typename T>
std::function<void(T&)> FixedPointFn(const std::function<bool(T&)>& f1,
                                     const std::function<bool(T&)>& f2,
                                     size_t max_iters) {
  return FixedPointFn(f1, f2, max_iters, nullptr);
}

// A no-op in-place transform (mutates nothing), used when shape inference or
// constant folding is disabled.
void Identity(onnx::ModelProto&) {}

// Recursively collect the op types of operators that live in ONNX's *default*
// domain but have no registered schema. These are custom operators -- most
// commonly TensorRT plugins such as ``BatchedNMS_TRT`` or ``EfficientNMS_TRT``
// -- that were exported into the default domain instead of a vendor-specific
// one.
//
// Custom ops that already live in a non-default domain (e.g. ``com.microsoft``
// or ``TRT``) are intentionally ignored: onnx::checker::check_model already
// tolerates unknown ops in non-standard domains, which is exactly the manual
// workaround reported in GitHub issue #220.
void CollectCustomDefaultDomainOps(const onnx::GraphProto& graph,
                                   int default_opset_version,
                                   std::set<std::string>& custom_ops) {
  for (const auto& node : graph.node()) {
    const std::string& domain = node.domain();
    const bool is_default_domain = domain.empty() || domain == "ai.onnx";
    if (is_default_domain &&
        onnx::OpSchemaRegistry::Schema(node.op_type(), default_opset_version,
                                       /*domain=*/"") == nullptr) {
      custom_ops.insert(node.op_type());
    }
    // Recurse into subgraphs held in node attributes (If/Loop/Scan bodies).
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectCustomDefaultDomainOps(attr.g(), default_opset_version,
                                      custom_ops);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectCustomDefaultDomainOps(subgraph, default_opset_version,
                                      custom_ops);
      }
    }
  }
}

// Register a permissive placeholder schema for every default-domain custom op
// found in ``model``. Without a schema, onnx::checker::check_model rejects the
// model with "No Op registered for <op> with domain_version of <n>" and
// simplification never even starts (GitHub issues #107 and #220). The
// placeholder accepts any number of inputs/outputs of any tensor type and any
// attributes, so the checker passes and the op is preserved untouched through
// simplification. It carries no shape/type inference function, so shape
// inference simply flows past the op as before.
void RegisterCustomDefaultDomainOpSchemas(const onnx::ModelProto& model) {
  int default_opset_version = 1;
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      default_opset_version =
          std::max(default_opset_version, static_cast<int>(opset.version()));
    }
  }

  std::set<std::string> custom_ops;
  CollectCustomDefaultDomainOps(model.graph(), default_opset_version,
                                custom_ops);

  for (const auto& op_type : custom_ops) {
    onnx::OpSchema schema;
    schema.SetName(op_type)
        .SetDomain("")
        .SinceVersion(1)
        .SetDoc(
            "Placeholder schema registered by onnxsim for a custom operator "
            "(e.g. a TensorRT plugin) exported into the default ONNX domain, "
            "so "
            "that the model passes validation and is simplified with the "
            "operator preserved unchanged.")
        .Input(0, "inputs", "Variadic inputs of the custom operator.", "T",
               onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
               /*min_arity=*/0)
        .Output(0, "outputs", "Variadic outputs of the custom operator.", "T",
                onnx::OpSchema::Variadic, /*is_homogeneous=*/false,
                /*min_arity=*/0)
        .TypeConstraint("T", onnx::OpSchema::all_tensor_types(),
                        "Allow inputs and outputs of any tensor type.")
        // Custom ops carry arbitrary, plugin-specific attributes; accept them
        // all rather than trying to enumerate them.
        .AllowUncheckedAttributes();
    // Never fail or throw: a duplicate registration (e.g. simplifying two
    // models that use the same custom op in one process) is a harmless no-op.
    onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/1,
                         /*fail_duplicate_schema=*/false,
                         /*fail_with_exception=*/false);
  }
}

// Collect the names of every node in `graph`, descending into subgraphs held in
// node attributes (If/Loop/Scan bodies). Used to de-duplicate the names later
// assigned to nameless nodes.
void CollectNodeNames(const onnx::GraphProto& graph,
                      std::set<std::string>& names) {
  for (const auto& node : graph.node()) {
    if (!node.name().empty()) {
      names.insert(node.name());
    }
    for (const auto& attr : node.attribute()) {
      if (attr.has_g()) {
        CollectNodeNames(attr.g(), names);
      }
      for (const auto& subgraph : attr.graphs()) {
        CollectNodeNames(subgraph, names);
      }
    }
  }
}

// Give a unique, deterministic name to every node that has none. Nodes without
// a name survive simplification unnamed -- either because they were nameless in
// the input model or because an onnx-optimizer pass created a replacement node
// without setting a name -- which trips up downstream tooling that keys on node
// names (issue #269). Each generated name is derived from the op type plus a
// running counter and de-duplicated against every name already present in the
// graph (including names generated earlier in this pass). Subgraphs are handled
// recursively so nodes inside If/Loop/Scan bodies are named too.
void AssignMissingNodeNames(onnx::GraphProto& graph,
                            std::set<std::string>& used_names,
                            size_t& counter) {
  for (auto& node : *graph.mutable_node()) {
    if (node.name().empty()) {
      std::string name;
      do {
        name = node.op_type() + "_" + std::to_string(counter++);
      } while (used_names.count(name) > 0);
      used_names.insert(name);
      node.set_name(name);
    }
    for (auto& attr : *node.mutable_attribute()) {
      if (attr.has_g()) {
        AssignMissingNodeNames(*attr.mutable_g(), used_names, counter);
      }
      for (auto& subgraph : *attr.mutable_graphs()) {
        AssignMissingNodeNames(subgraph, used_names, counter);
      }
    }
  }
}

// Assign names to any nodes left nameless after simplification (issue #269).
void AssignMissingNodeNames(onnx::ModelProto& model) {
  std::set<std::string> used_names;
  CollectNodeNames(model.graph(), used_names);
  size_t counter = 0;
  AssignMissingNodeNames(*model.mutable_graph(), used_names, counter);
}

void Check(const onnx::ModelProto& model) { onnx::checker::check_model(model); }

// Return the opset version the model imports for the default ONNX domain
// (represented as either the empty string or "ai.onnx"), or std::nullopt when
// the model does not import the default domain at all (a model made purely of
// custom-domain operators).
std::optional<int> DefaultOpsetVersion(const onnx::ModelProto& model) {
  for (const auto& opset : model.opset_import()) {
    if (opset.domain().empty() || opset.domain() == "ai.onnx") {
      return static_cast<int>(opset.version());
    }
  }
  return std::nullopt;
}

// Mirrors onnx_simplifier.py's remove_initializer_from_input: an initializer
// that also appears in graph.input is treated as a runtime input by
// onnxoptimizer's is_constant_initializer, which blocks value-baking fusions
// (fuse_bn_into_conv, ...) that only work on non-input initializers -- e.g.
// the plain Conv+BN chains of the opset-8 resnet101-v1-7 were left completely
// unsimplified without this. Removing it from the input list lets it fold
// like any other constant. Kept in sync with the Python function; see that
// one's own comment for the opset-6 floor (Cast's `to` attribute's INT
// encoding wasn't valid before opset 6, so bumping an old-IR/old-opset model
// to IR 4 here would let a later fusion emit a node the opset rejects).
void RemoveInitializerFromInput(onnx::ModelProto& model) {
  constexpr int kMinOpsetForInitializerFold = 6;
  if (model.ir_version() < 4) {
    const auto opset = DefaultOpsetVersion(model);
    if (!opset || *opset < kMinOpsetForInitializerFold) {
      return;
    }
  }
  std::set<std::string> initializer_names;
  for (const auto& init : model.graph().initializer()) {
    initializer_names.insert(init.name());
  }
  auto* inputs = model.mutable_graph()->mutable_input();
  bool removed_any = false;
  for (int i = inputs->size() - 1; i >= 0; --i) {
    if (initializer_names.count((*inputs)[i].name()) > 0) {
      inputs->erase(inputs->begin() + i);
      removed_any = true;
    }
  }
  if (removed_any && model.ir_version() < 4) {
    model.set_ir_version(4);
  }
}

// ONNX TensorProto element types onnxoptimizer's tensor-value hashing
// (cse_util.h) knows how to hash. Any other type makes
// eliminate_common_subexpression/eliminate_duplicate_initializer raise
// "no supported data type: <N>". Enumerates the *supported* types (rather
// than the unsupported ones) so an element type added to ONNX in the future
// is treated as unhashable by default instead of silently crashing. Mirrors
// onnx_simplifier.py's _CSE_HASHABLE_ELEM_TYPES.
bool IsCSEHashableElemType(int32_t elem_type) {
  switch (elem_type) {
    case onnx::TensorProto::UNDEFINED:
    case onnx::TensorProto::BOOL:
    case onnx::TensorProto::INT8:
    case onnx::TensorProto::INT16:
    case onnx::TensorProto::INT32:
    case onnx::TensorProto::INT64:
    case onnx::TensorProto::UINT8:
    case onnx::TensorProto::UINT16:
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
    case onnx::TensorProto::FLOAT:
    case onnx::TensorProto::DOUBLE:
    case onnx::TensorProto::FLOAT16:
    case onnx::TensorProto::BFLOAT16:
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::COMPLEX128:
    case onnx::TensorProto::STRING:
      return true;
    default:
      return false;
  }
}

// Mirrors onnx_simplifier.py's _has_cse_unhashable_tensor /
// _iter_tensor_data_types: walks every tensor CSE might hash -- initializers,
// t/ts node attributes, recursing into subgraphs -- looking for an element
// type IsCSEHashableElemType rejects.
bool GraphHasCSEUnhashableTensor(const onnx::GraphProto& graph) {
  for (const auto& init : graph.initializer()) {
    if (!IsCSEHashableElemType(init.data_type())) {
      return true;
    }
  }
  for (const auto& node : graph.node()) {
    for (const auto& attr : node.attribute()) {
      if (attr.has_t() && !IsCSEHashableElemType(attr.t().data_type())) {
        return true;
      }
      for (const auto& t : attr.tensors()) {
        if (!IsCSEHashableElemType(t.data_type())) {
          return true;
        }
      }
      if (attr.has_g() && GraphHasCSEUnhashableTensor(attr.g())) {
        return true;
      }
      for (const auto& g : attr.graphs()) {
        if (GraphHasCSEUnhashableTensor(g)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Mirrors onnx_simplifier.py's overwrite_input_shapes loop: for each named
// graph input, overwrite its shape's dims with the given values, skipping
// any non-positive entry (which means "keep the original, possibly dynamic,
// dimension" rather than hardcoding an invalid size such as 0 -- GitHub
// issue #237). Throws std::runtime_error if a name isn't a graph input
// (matching onnx_simplifier.py's RuntimeError for the same case).
void ApplyInputShapeOverwrite(
    onnx::ModelProto& model,
    const std::unordered_map<std::string, std::vector<int64_t>>&
        overwrite_input_shapes) {
  for (const auto& [name, shape] : overwrite_input_shapes) {
    bool found = false;
    for (auto& input : *model.mutable_graph()->mutable_input()) {
      if (input.name() != name) {
        continue;
      }
      found = true;
      auto* dims = input.mutable_type()
                       ->mutable_tensor_type()
                       ->mutable_shape()
                       ->mutable_dim();
      for (int i = 0; i < dims->size() && i < static_cast<int>(shape.size());
           ++i) {
        if (shape[i] > 0) {
          dims->Mutable(i)->set_dim_value(shape[i]);
        }
      }
    }
    if (!found) {
      throw std::runtime_error("The model doesn't have input named \"" + name +
                               "\"");
    }
  }
}

// Mirrors onnx_simplifier.py's remove_unused_output: drops the named graph
// outputs. Downstream dead-end elimination cleans up nodes that only fed
// them. Throws std::runtime_error if a name isn't a graph output (matching
// onnx_simplifier.py's RuntimeError for the same case).
void RemoveUnusedOutputs(onnx::ModelProto& model,
                         const std::vector<std::string>& unused_output) {
  for (const auto& name : unused_output) {
    bool found = false;
    for (const auto& output : model.graph().output()) {
      if (output.name() == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("The model doesn't have output named \"" + name +
                               "\"");
    }
  }
  auto* outputs = model.mutable_graph()->mutable_output();
  google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> kept;
  for (auto& output : *outputs) {
    if (std::find(unused_output.begin(), unused_output.end(), output.name()) ==
        unused_output.end()) {
      *kept.Add() = std::move(output);
    }
  }
  outputs->Swap(&kept);
}

// Convert the default ONNX domain of the model to target_version using onnx's
// own version converter. Only the default ONNX domain opset is changed;
// custom/other-domain opset imports are left untouched. Returns the model
// unchanged when it is already at the requested version or does not import the
// default domain.
onnx::ModelProto ConvertOpsetVersion(const onnx::ModelProto& model,
                                     int target_version) {
  const auto current = DefaultOpsetVersion(model);
  if (!current || *current == target_version) {
    return model;
  }
  return onnx::version_conversion::ConvertVersion(model, target_version);
}

// Shared schema setup for the single-pass debug helpers below, mirroring the
// head of ``Simplify``: teach shape inference about ONNX Runtime's quantized
// contrib ops and register permissive placeholders for custom ops exported into
// the default ONNX domain, so neither shape inference nor a later checker
// rejects the model. Unlike ``Simplify`` these helpers do not ``Check`` the
// model up front -- the point of running a step in isolation is to debug a
// model that may not yet be fully valid.
void PrepareSchemasForDebug(const onnx::ModelProto& model) {
  onnxsim::RegisterContribOpSchemas();
  FixupSchemaDeterminism();
  RegisterCustomDefaultDomainOpSchemas(model);
}

onnx::ModelProto InferShapesOnce(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnx::ModelProto out = model;
  _InferShapes(out);
  return out;
}

onnx::ModelProto PropagateDataOnce(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  onnx::ModelProto out = model;
  _EvalPartialShape(out);
  return out;
}

onnx::ModelProto FoldConstantOnce(const ModelExecutor& executor,
                                  const onnx::ModelProto& model,
                                  size_t tensor_size_threshold,
                                  bool initializers_as_constants) {
  PrepareSchemasForDebug(model);
  // ``_FoldConstant`` reads these globals (tensor-size cap and the
  // initializers-as-constants policy), just as ``Simplify`` sets them.
  config.tensor_size_threshold = tensor_size_threshold;
  config.initializers_as_constants = initializers_as_constants;
  config.optimizer_passes.clear();
  onnx::ModelProto out = model;
  // Mirror ``Simplify``'s FoldConstant step: partial shape evaluation first
  // turns Shape/Gather-on-shape into constants that the ordinary constant
  // folder can then propagate.
  _EvalPartialShape(out);
  out = _FoldConstant(executor, std::move(out));
  return out;
}

// Records the options this Simplify() call actually used (skip_optimizers
// reflects any auto-added unhashable-tensor protections) as string
// key/value pairs in the model's metadata_props, namespaced under
// "onnxsim:" so they cannot collide with the model's own metadata. Lets a
// downstream consumer see how a model was simplified without having to have
// kept the original call around. Replaces any pre-existing "onnxsim:*"
// entries (e.g. left by a previous simplify() call) rather than
// accumulating duplicates across repeated simplification.
void RecordSimplifyOptionsMetadata(
    onnx::ModelProto& model,
    const std::optional<std::vector<std::string>>& skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, bool initializers_as_constants,
    bool include_inline_functions, bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output) {
  auto join = [](const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) {
        out += ",";
      }
      out += v[i];
    }
    return out;
  };

  std::vector<std::pair<std::string, std::string>> options;
  options.emplace_back("onnxsim:skip_optimizers",
                       skip_optimizers ? join(*skip_optimizers) : "<all>");
  options.emplace_back("onnxsim:constant_folding",
                       constant_folding ? "true" : "false");
  options.emplace_back("onnxsim:shape_inference",
                       shape_inference ? "true" : "false");
  options.emplace_back("onnxsim:tensor_size_threshold",
                       std::to_string(tensor_size_threshold));
  options.emplace_back(
      "onnxsim:target_opset_version",
      target_opset_version ? std::to_string(*target_opset_version) : "");
  options.emplace_back("onnxsim:initializers_as_constants",
                       initializers_as_constants ? "true" : "false");
  options.emplace_back("onnxsim:include_inline_functions",
                       include_inline_functions ? "true" : "false");
  options.emplace_back("onnxsim:mutable_initializer",
                       mutable_initializer ? "true" : "false");
  if (overwrite_input_shapes) {
    std::string s;
    bool first = true;
    for (const auto& [name, shape] : *overwrite_input_shapes) {
      if (!first) {
        s += ";";
      }
      first = false;
      s += name + ":";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
          s += ",";
        }
        s += std::to_string(shape[i]);
      }
    }
    options.emplace_back("onnxsim:overwrite_input_shapes", s);
  }
  if (unused_output) {
    options.emplace_back("onnxsim:unused_output", join(*unused_output));
  }

  auto* props = model.mutable_metadata_props();
  google::protobuf::RepeatedPtrField<onnx::StringStringEntryProto> kept;
  for (auto& prop : *props) {
    if (prop.key().rfind("onnxsim:", 0) != 0) {
      *kept.Add() = std::move(prop);
    }
  }
  props->Swap(&kept);
  for (const auto& [key, value] : options) {
    auto* entry = props->Add();
    entry->set_key(key);
    entry->set_value(value);
  }
}

onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output) {
  // Register onnxsim's own optimizer passes into onnxoptimizer's registry
  // before the pass list is built below: fuse_consecutive_mul,
  // fuse_mul_into_conv, fuse_preceding_mul_into_conv and
  // eliminate_reshape_around_elementwise are auto-selected via
  // GetFuseAndEliminationPass, and fuse_matmul_add_bias_into_gemm_batched is
  // named explicitly. The call is idempotent.
  onnxsim::RegisterCustomOptimizerPasses();
  // Make shape inference aware of ONNX Runtime's quantized contrib operators
  // (QLinearAdd and friends) so shape deduction does not stop at them.
  onnxsim::RegisterContribOpSchemas();
  // Correct the determinism metadata of ops ONNX mis-annotates (e.g. Range) so
  // constant folding does not skip them.
  FixupSchemaDeterminism();
  // Register permissive placeholder schemas for custom ops exported into the
  // default ONNX domain (e.g. TensorRT plugins such as BatchedNMS_TRT) so the
  // checker below does not reject the model (GitHub issues #107, #220).
  RegisterCustomDefaultDomainOpSchemas(model);

  Check(model);

  config.tensor_size_threshold = tensor_size_threshold;
  config.initializers_as_constants = initializers_as_constants;
  config.optimizer_passes.clear();
  // onnxsim already folds ``Gather`` on a ``Shape`` into constants on its own:
  // ``_EvalPartialShape`` (above) plus data-propagating shape inference resolve
  // the statically-known dimensions and leave the genuinely dynamic ones
  // untouched. The onnx-optimizer ``eliminate_shape_gather`` pass is therefore
  // redundant here, and it aborts the whole process on graphs where a Gather
  // index cannot be statically resolved to an axis (common in dynamic-shape
  // detection models such as FasterRCNN). Always drop it from the pass list.
  std::vector<std::string> always_disabled_passes = {"eliminate_shape_gather"};
  // When initializers are treated as non-constant, keep ``Constant`` nodes in
  // producer form: ``extract_constant_to_initializer`` would rewrite every
  // Constant into an initializer, which -- being non-constant now -- would then
  // block onnxsim's own constant folding of genuinely-constant subgraphs. The
  // value-baking passes themselves already leave initializer weights alone via
  // ``IsConstantTensor`` (see the onnx-optimizer changes), so only this
  // representation-changing pass needs dropping.
  if (!initializers_as_constants) {
    always_disabled_passes.push_back("extract_constant_to_initializer");
  }
  auto is_disabled = [](const std::vector<std::string>& list,
                        const std::string& pass) {
    return std::find(list.begin(), list.end(), pass) != list.end();
  };
  // onnxoptimizer's common-subexpression / duplicate-initializer passes hash
  // tensor values and crash with "no supported data type: <N>" on element
  // types they cannot hash, such as the float8 zero points produced by
  // NVIDIA ModelOpt fp8 QDQ models (GitHub issue #348). When such a tensor
  // is present, transparently skip those two passes so the rest of
  // simplification still runs, instead of crashing outright. Only relevant
  // when some optimizer is actually going to run (skip_optimizers ==
  // nullopt already disables all of them).
  if (skip_optimizers && GraphHasCSEUnhashableTensor(model.graph())) {
    static const std::vector<std::string> kTensorValueHashingOptimizers = {
        "eliminate_common_subexpression", "eliminate_duplicate_initializer"};
    for (const auto& opt : kTensorValueHashingOptimizers) {
      if (!is_disabled(*skip_optimizers, opt)) {
        skip_optimizers->push_back(opt);
      }
    }
  }
  // skip_optimizers == nullopt means skiping all optimizers, so
  // config.optimizer_passes is empty
  if (skip_optimizers) {
    std::vector<std::string> passes;
    const auto all_passes = onnx::optimization::GetFuseAndEliminationPass();
    for (const auto& pass : all_passes) {
      if (!is_disabled(*skip_optimizers, pass) &&
          !is_disabled(always_disabled_passes, pass)) {
        passes.push_back(pass);
      }
    }
    // Opt into the batched MatMul+bias -> Gemm rewrite. onnx-optimizer
    // registers it as PassType::Other (so it is absent from
    // GetFuseAndEliminationPass), because it is a graph-shape rewrite rather
    // than a pure node reduction. Transformer linear layers apply a 2-D weight
    // to rank-3 activations, which the 2-D-only
    // ``fuse_matmul_add_bias_into_gemm`` cannot fuse; converting them to
    // ``Gemm`` lets runtimes dispatch their tuned GEMM kernels. The reshape
    // scaffolding it introduces around chains of element-wise ops is then
    // cancelled by ``eliminate_reshape_around_elementwise`` (a Nop pass in the
    // default set), which keeps the Gemms but drops the now-inverse reshape
    // pairs so the node count does not regress. Still honour an explicit
    // ``--skip-optimization`` for it.
    const std::string batched_gemm = "fuse_matmul_add_bias_into_gemm_batched";
    if (!is_disabled(*skip_optimizers, batched_gemm) &&
        !is_disabled(always_disabled_passes, batched_gemm)) {
      passes.push_back(batched_gemm);
    }
    config.optimizer_passes = passes;
  }

  // Every transform mutates the model in place.
  using ModelFn = std::function<void(onnx::ModelProto&)>;
  ModelFn FoldConstant;
  if (constant_folding) {
    FoldConstant = [&executor](onnx::ModelProto& model) {
      // Partial shape evaluation (issue #139) turns Shape/Gather-on-shape into
      // constants that the ordinary constant folder can then propagate.
      _EvalPartialShape(model);
      model = _FoldConstant(executor, std::move(model));
    };
  } else {
    FoldConstant = Identity;
  }
  // ``perform_optimization=False`` (skip_optimizers == nullopt) means the
  // caller wants the graph structure left alone, so the state elimination --
  // which relies on dead-end elimination to clean up behind it -- runs with the
  // optimizer or not at all.
  const bool optimize = skip_optimizers.has_value();

  int fixed_point_iters =
      std::getenv("ONNXSIM_FIXED_POINT_ITERS")
          ? std::atoi(std::getenv("ONNXSIM_FIXED_POINT_ITERS"))
          : 50;

  // Optionally profile every fixed-point function. Turned on by pointing
  // ``ONNXSIM_PROFILE`` at an output file (``ONNXSIM_PROFILE=1`` uses the
  // default name), mirroring ``ONNXSIM_FIXED_POINT_ITERS`` so it works from
  // every binding without a signature change. ``Profiled`` wraps a transform so
  // each invocation records its wall/CPU duration and peak RSS; the wrappers
  // nest, so the fixed points show up as parent spans of the transforms they
  // drive. When profiling is off ``Profiled`` returns the function unchanged
  // and
  // ``ProfiledScope`` is a no-op, so there is zero overhead.
  if (const char* profile_env = std::getenv("ONNXSIM_PROFILE")) {
    std::string profile_path = profile_env;
    if (profile_path == "1" || profile_path == "true" || profile_path == "on" ||
        profile_path == "yes") {
      profile_path = "onnxsim_profile.json";
    }
    if (!profile_path.empty()) {
      onnxsim::Profiler::Instance().Enable(profile_path);
    }
  }
  // Optionally break the per-round Optimize() cost down further, into each
  // PredicateBasedPass's "matching" (patternMatchPredicate) vs "modifying"
  // (runTransform) phase, per pass name -- see onnxoptimizer/pass.h's
  // PassPhaseTiming for scope and overhead. Exploratory diagnostic for
  // onnxsim issue #633; the summary prints to stderr after Pipeline runs.
  const bool profile_pass_phases = [] {
    const char* env = std::getenv("ONNXSIM_PROFILE_PASS_PHASES");
    return env != nullptr && std::string(env) != "0" &&
           std::string(env) != "false";
  }();
  if (profile_pass_phases) {
    onnx::optimization::ResetPassPhaseTimings();
    onnx::optimization::ResetPassTotalTimings();
    onnx::optimization::ResetCSEHashCompareTiming();
    onnx::optimization::ResetCSEPassTiming();
    onnx::optimization::ResetDeadendPassTiming();
    onnx::optimization::SetPassPhaseProfilingEnabled(true);
  }
  // Optionally merge ONNX Runtime's own per-session profiles into the onnxsim
  // trace -- the binding-agnostic counterpart of Python's
  // ``merge_ort_profile``, so the C ABI, Rust and WASM bindings can produce one
  // unified trace too. Enable the profiler if it is not already (there must be
  // a trace to merge into), then have it collect and splice ONNX Runtime's
  // traces at Finish().
  if (const char* merge_env = std::getenv("ONNXSIM_MERGE_ORT_PROFILE")) {
    std::string v = merge_env;
    if (!v.empty() && v != "0" && v != "false" && v != "off" && v != "no") {
      if (!onnxsim::Profiler::Instance().enabled()) {
        onnxsim::Profiler::Instance().Enable("onnxsim_profile.json");
      }
      onnxsim::Profiler::Instance().SetMergeOrtTraces(true);
    }
  }
  // Ensure the trace is flushed and the sampler thread is stopped even if a
  // transform throws mid-pipeline. Finish() is idempotent, so the explicit call
  // on the success path below is a harmless no-op the second time.
  struct ProfilerFinishGuard {
    ~ProfilerFinishGuard() { onnxsim::Profiler::Instance().Finish(); }
  } profiler_finish_guard;
  auto Profiled = [](const char* name, ModelFn fn) -> ModelFn {
    if (!onnxsim::Profiler::Instance().enabled()) {
      return fn;
    }
    return [name, fn = std::move(fn)](onnx::ModelProto& model) {
      onnxsim::ProfiledScope scope(name);
      fn(model);
    };
  };
  // Fully Graph-resident OptAndShape, built on onnx::InferShapesOnGraph
  // (onnx/common/graph_shape_inference.h) and onnx-optimizer's
  // OptimizeGraphFixed (onnxoptimizer/optimize.h): both run directly against
  // one onnx::Graph held for the whole inner fixed point, so this does
  // exactly one Import and one Export for the *entire* fixed point, however
  // many rounds it takes to converge -- instead of one Import+Export per
  // round. This is onnxsim issue #633's "remaining option 2" (eliminating
  // the round trip itself, not just its per-round cost); see #637 and #638
  // for the validation, profiling and follow-up fix that took this from
  // opt-in to the default.
  //
  // Remaining known scope gap: InferShapesOnGraph's own v1 scope leaves
  // control-flow ops (If/Loop/Scan) and function-body ops uninferred (safe:
  // exactly as if they had no registered schema, never wrong -- see that
  // function's doc comment), which can converge to less complete shape info
  // than a from-scratch protobuf InferShapes call for models using those
  // ops.
  ModelFn OptAndShape = Profiled(
      "OptAndShape",
      [optimize, shape_inference, fixed_point_iters](onnx::ModelProto& model) {
        std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
        if (g.get() == nullptr) {
          // Same fallback as Optimizer::optimize(): if we can't parse the
          // model, leave it untouched.
          return;
        }
        // Scope onnx-optimizer's tensor-hash caches (TensorContentDigest,
        // the typed-field path, and CSETensorHash's raw_data-branch cache
        // -- see onnxoptimizer/passes/{tensor_content_hash,cse_util}.h) to
        // this whole OptAndShape call, instead of the OptimizeGraphFixed
        // call below clearing them every round (its own default): `g` is
        // one resident Graph held across every round of the fixed point
        // underneath, and no round mutates a retained tensor's content in
        // place (only replaces it wholesale, which mints its own fresh
        // tensor_id() and simply misses these caches) -- so a hash computed
        // for a tensor that survives unchanged from one round to the next
        // stays valid, and correct, letting eliminate_duplicate_initializer
        // and eliminate_common_subexpression skip rehashing it on every one
        // of the ~dozens of rounds a deep repeated-block model can take.
        // Dominated in practice by the raw_data cache: real exported models
        // are raw_data-heavy, and measured 98%+ of those hash calls were
        // otherwise recomputing a value already seen earlier in the same
        // run (see onnxsim issue #633). OptimizeGraphChanged below passes
        // clear_tensor_digest_cache=false to keep both caches alive across
        // its own repeated OptimizeGraphFixed calls.
        onnx::optimization::ClearTensorContentDigestCache();
        onnx::optimization::ClearRawHashCache();
        using GraphFnChanged = std::function<bool(onnx::Graph&)>;
        GraphFnChanged InferShapesOnGraphChanged =
            shape_inference
                ? GraphFnChanged([](onnx::Graph& graph) {
                    onnxsim::ProfiledScope scope("InferShapes");
                    return onnx::InferShapesOnGraph(graph);
                  })
                : GraphFnChanged([](onnx::Graph&) { return false; });
        GraphFnChanged OptimizeGraphChanged = [optimize](onnx::Graph& graph) {
          onnxsim::ProfiledScope scope("Optimize");
          onnxsim::RegisterCustomOptimizerPasses();
          if (optimize) {
            // Unset all-zero recurrent initial states (issue #314) so the
            // subgraph computing them becomes dead and the passes below can
            // remove it. Not reflected in the "changed" signal below, but
            // that is safe: see FixedPointFn's bool-returning overload
            // comment -- any resulting dead-end elimination is itself
            // reflected in the report.
            EliminateZeroRnnInitialState(graph);
          }
          const bool prev = onnx::optimization::InitializersAsConstants();
          onnx::optimization::SetInitializersAsConstants(
              config.initializers_as_constants);
          std::map<std::string, unsigned int> report;
          onnx::optimization::OptimizeGraphFixed(
              graph, config.optimizer_passes, &report,
              /*clear_tensor_digest_cache=*/false);
          onnx::optimization::SetInitializersAsConstants(prev);
          for (const auto& pass_count : report) {
            if (pass_count.second != 0) {
              return true;
            }
          }
          return false;
        };
        FixedPointFn(InferShapesOnGraphChanged, OptimizeGraphChanged,
                     fixed_point_iters)(*g);
        onnx::ModelProto out = onnx::PrepareOutput(model);
        onnx::ExportModelProto(&out, g, /*consume_tensor_data=*/true);
        // OptimizeGraphFixed never sees model-local functions (they live
        // on ModelProto, not Graph), so carry them over unchanged --
        // mirroring Optimizer::optimize()'s own AddFunctionsToModel.
        for (const auto& function_proto : model.functions()) {
          *out.add_functions() = function_proto;
        }
        model = std::move(out);
      });
  bool converged = false;
  ModelFn Pipeline;
  if (rewriter) {
    // Run the user-supplied rewriter (e.g. an onnxscript.rewriter rule set) as
    // the outermost stage of the fixed point: each round drives shape
    // inference, onnxoptimizer and constant folding to a fixed point, then
    // rewrites, then repeats until the whole thing stops changing. This lets a
    // rewrite expose new optimizer/folding opportunities and vice versa. The
    // rewriter runs on the ``ModelProto`` directly, so no internal-graph
    // conversion is involved.
    ModelFn OptAndShapeAndFold = Profiled(
        "OptAndShapeAndFold",
        FixedPointFn(OptAndShape, Profiled("FoldConstant", FoldConstant),
                     fixed_point_iters));
    ModelFn RewriteInPlace =
        Profiled("Rewrite", [rewriter](onnx::ModelProto& model) {
          // ``_Run`` rewrites in place and returns whether it changed anything;
          // when it reports no change it leaves ``model`` untouched, so no copy
          // is made. The fixed point's fingerprint comparison then sees the
          // unchanged model and converges without an extra ModelProto
          // round-trip.
          rewriter->_Run(model);
        });
    Pipeline =
        Profiled("Pipeline", FixedPointFn(OptAndShapeAndFold, RewriteInPlace,
                                          fixed_point_iters, &converged));
  } else {
    Pipeline = Profiled(
        "Pipeline",
        FixedPointFn(OptAndShape, Profiled("FoldConstant", FoldConstant),
                     fixed_point_iters, &converged));
  }
  // The fixed points mutate in place, so make one working copy of the (const)
  // input model and simplify it in place.
  onnx::ModelProto sim_model = model;
  // Matches onnx_simplifier.py's default (mutable_initializer=False): fold
  // initializers that also appear as graph inputs like any other constant.
  if (!mutable_initializer) {
    RemoveInitializerFromInput(sim_model);
  }
  // Overwrite named input shapes, if requested, before shape inference or
  // any transform runs -- mirrors onnx_simplifier.py's overwrite_input_shapes
  // loop, applied here instead of by the Python wrapper.
  if (overwrite_input_shapes) {
    ApplyInputShapeOverwrite(sim_model, *overwrite_input_shapes);
  }
  // Drop the named graph outputs, if requested, before the fixed point so
  // dead-end elimination cleans up nodes that only fed them.
  if (unused_output) {
    RemoveUnusedOutputs(sim_model, *unused_output);
  }
  // Optionally inline the model's local (model-defined) functions into the main
  // graph up front, so the optimizer, shape inference and constant folding
  // below see through them into a flat op graph. Done before the opset
  // conversion and fixed point so everything downstream operates on the inlined
  // graph; schema-defined functions are left untouched. Off by default, so a
  // model with functions is otherwise simplified exactly as before.
  if (include_inline_functions) {
    onnx::inliner::InlineLocalFunctions(sim_model);
  }
  // Optionally convert the model to a different opset version (of the default
  // ONNX domain) first, so the simplification below can clean up any redundant
  // nodes the version converter introduces.
  if (target_opset_version) {
    sim_model = ConvertOpsetVersion(sim_model, *target_opset_version);
  }
  {
    // A single root span so the profiled fixed points nest under one box in the
    // flame graph (a no-op when profiling is disabled).
    onnxsim::ProfiledScope root("Simplify");
    Pipeline(sim_model);
  }
  // Flush the profiling trace and print the per-function summary. Safe to call
  // unconditionally: Finish() is a no-op unless profiling was enabled above.
  onnxsim::Profiler::Instance().Finish();
  if (profile_pass_phases) {
    onnx::optimization::SetPassPhaseProfilingEnabled(false);
    std::vector<std::pair<std::string, onnx::optimization::PassPhaseTiming>>
        rows(onnx::optimization::GetPassPhaseTimings().begin(),
             onnx::optimization::GetPassPhaseTimings().end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return (a.second.match_ms + a.second.transform_ms) >
             (b.second.match_ms + b.second.transform_ms);
    });
    std::cerr << "\nonnxsim pass-phase profiling (PredicateBasedPass "
                 "matching vs modifying)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << std::left << std::setw(38) << "pass" << std::right
              << std::setw(12) << "match(ms)" << std::setw(10) << "calls"
              << std::setw(14) << "modify(ms)" << std::setw(10) << "calls"
              << std::setw(12) << "total(ms)" << "\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n";
    double total_match_ms = 0, total_transform_ms = 0;
    for (const auto& [name, t] : rows) {
      std::cerr << std::left << std::setw(38) << name << std::right
                << std::setw(12) << std::fixed << std::setprecision(2)
                << t.match_ms << std::setw(10) << t.match_calls << std::setw(14)
                << t.transform_ms << std::setw(10) << t.transform_calls
                << std::setw(12) << (t.match_ms + t.transform_ms) << "\n";
      total_match_ms += t.match_ms;
      total_transform_ms += t.transform_ms;
    }
    std::cerr << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "TOTAL matching: " << total_match_ms
              << "ms, TOTAL modifying: " << total_transform_ms << "ms\n";

    // Coarser companion table: total wall time inside EVERY pass's
    // runPass(Graph&) call (both kinds), which is what actually accounts
    // for the full Optimize() cost -- see PassTotalTiming's comment.
    std::vector<std::pair<std::string, onnx::optimization::PassTotalTiming>>
        total_rows(onnx::optimization::GetPassTotalTimings().begin(),
                   onnx::optimization::GetPassTotalTimings().end());
    std::sort(total_rows.begin(), total_rows.end(),
              [](const auto& a, const auto& b) {
                return a.second.total_ms > b.second.total_ms;
              });
    std::cerr << "\nonnxsim pass-phase profiling (total runPass() time, all "
                 "pass kinds)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << std::left << std::setw(38) << "pass" << std::right
              << std::setw(14) << "total(ms)" << std::setw(10) << "calls"
              << "\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n";
    double grand_total_ms = 0;
    for (const auto& [name, t] : total_rows) {
      std::cerr << std::left << std::setw(38) << name << std::right
                << std::setw(14) << std::fixed << std::setprecision(2)
                << t.total_ms << std::setw(10) << t.calls << "\n";
      grand_total_ms += t.total_ms;
    }
    std::cerr << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "GRAND TOTAL (sum of all pass runPass() calls): "
              << grand_total_ms << "ms\n";

    // CSETensorHash/CSETensorCompare breakdown -- the actual hashing and
    // equality-checking work inside eliminate_duplicate_initializer and
    // eliminate_common_subexpression's hash-map lookups. Splits raw_data
    // (real exported-model weights; a byte hash / a single memcmp) from
    // typed-field (BLAKE3-digest-backed; rarer in practice) so it's clear
    // which one, if either, actually accounts for those two passes' cost.
    const auto& cse_t = onnx::optimization::GetCSEHashCompareTiming();
    std::cerr << "\nonnxsim CSETensorHash / CSETensorCompare breakdown "
                 "(inside eliminate_duplicate_initializer / "
                 "eliminate_common_subexpression's hash-map lookups)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << std::left << std::setw(24) << "" << std::right << std::setw(14)
              << "total(ms)" << std::setw(10) << "calls" << "\n"
              << std::left << std::setw(24) << "raw_data hash" << std::right
              << std::setw(14) << std::fixed << std::setprecision(2)
              << cse_t.raw_hash_ms << std::setw(10) << cse_t.raw_hash_calls
              << "\n"
              << std::left << std::setw(24) << "typed-field hash" << std::right
              << std::setw(14) << cse_t.typed_hash_ms << std::setw(10)
              << cse_t.typed_hash_calls << "\n"
              << std::left << std::setw(24) << "raw_data compare" << std::right
              << std::setw(14) << cse_t.raw_compare_ms << std::setw(10)
              << cse_t.raw_compare_calls << "\n"
              << std::left << std::setw(24) << "typed-field compare"
              << std::right << std::setw(14) << cse_t.typed_compare_ms
              << std::setw(10) << cse_t.typed_compare_calls << "\n"
              << std::left << std::setw(24) << "node hash (CSENodeHash)"
              << std::right << std::setw(14) << cse_t.node_hash_ms
              << std::setw(10) << cse_t.node_hash_calls << "\n"
              << std::left << std::setw(24) << "  of which attrs+sort"
              << std::right << std::setw(14) << cse_t.node_hash_attrsort_ms
              << std::setw(10) << cse_t.node_hash_attrsort_calls << "\n"
              << std::left << std::setw(24) << "node equal (CSEEqual)"
              << std::right << std::setw(14) << cse_t.node_equal_ms
              << std::setw(10) << cse_t.node_equal_calls << "\n"
              << std::left << std::setw(24) << "  of which attrs+sort"
              << std::right << std::setw(14) << cse_t.node_equal_attrsort_ms
              << std::setw(10) << cse_t.node_equal_attrsort_calls << "\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "TOTAL hash: "
              << (cse_t.raw_hash_ms + cse_t.typed_hash_ms + cse_t.node_hash_ms)
              << "ms, TOTAL compare: "
              << (cse_t.raw_compare_ms + cse_t.typed_compare_ms +
                  cse_t.node_equal_ms)
              << "ms\n";

    // Actual cache hit rate: see cse_util.h's g_raw_hash_cache, keyed by
    // Tensor::tensor_id().
    if (cse_t.raw_hash_calls > 0) {
      std::cerr << "raw_data hash cache: " << cse_t.raw_hash_cache_hits << "/"
                << cse_t.raw_hash_calls << " calls ("
                << (100.0 * cse_t.raw_hash_cache_hits / cse_t.raw_hash_calls)
                << "%) were hits (" << cse_t.raw_hash_cache_hit_ms << "ms) vs "
                << cse_t.raw_hash_cache_misses << " misses ("
                << cse_t.raw_hash_cache_miss_ms << "ms).\n";
    }

    // eliminate_common_subexpression's own outer-loop breakdown: filtering
    // (hasUses/IsSupportedByCSE), the hash_map.emplace() lookup itself
    // (overlaps with node_hash_ms/node_equal_ms above -- same work, two
    // views), and replacing a found duplicate's uses.
    const auto& cse_pass_t = onnx::optimization::GetCSEPassTiming();
    std::cerr << "\nonnxsim eliminate_common_subexpression outer-loop "
                 "breakdown (across "
              << cse_pass_t.calls << " calls, " << cse_pass_t.nodes_seen
              << " nodes seen, " << cse_pass_t.nodes_filtered_out
              << " filtered out, " << cse_pass_t.nodes_replaced
              << " replaced)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "  filter (hasUses/IsSupportedByCSE): " << cse_pass_t.filter_ms
              << "ms\n"
              << "  lookup (hash_map.emplace):          "
              << cse_pass_t.lookup_ms << "ms\n"
              << "  replace (tryReplacingAllUsesWith):   "
              << cse_pass_t.replace_ms << "ms\n";

    // eliminate_deadend's own breakdown: the liveness check (hasUses) vs
    // actually unlinking/freeing a dead node (destroyCurrent).
    const auto& dead_t = onnx::optimization::GetDeadendPassTiming();
    std::cerr << "\nonnxsim eliminate_deadend breakdown (across "
              << dead_t.calls << " calls, " << dead_t.nodes_seen
              << " nodes seen, " << dead_t.nodes_removed << " removed)\n"
              << "-------------------------------------------------------"
                 "------------------------------------\n"
              << "  hasUses():      " << dead_t.has_uses_ms << "ms\n"
              << "  destroyCurrent(): " << dead_t.destroy_ms << "ms\n";
  }
  // Simplification (and some onnx-optimizer passes) can leave nodes without a
  // name; assign unique names to them so downstream tools that key on node
  // names keep working (issue #269).
  AssignMissingNodeNames(sim_model);
  RecordSimplifyOptionsMetadata(sim_model, skip_optimizers, constant_folding,
                                shape_inference, tensor_size_threshold,
                                target_opset_version, initializers_as_constants,
                                include_inline_functions, mutable_initializer,
                                overwrite_input_shapes, unused_output);
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

size_t LoadModelPooled(const std::string& path, onnx::ModelProto* model,
                       onnxsim::tensor_pool::TensorPool& pool,
                       uint64_t hydrate_threshold_bytes) {
  onnx::optimization::loadModel(model, path, /*load_external_data=*/false);
  const std::string base_dir =
      std::filesystem::path(path).parent_path().string();
  return onnxsim::tensor_pool::PoolExternalData(*model, base_dir, pool,
                                                hydrate_threshold_bytes);
}

// Tensors at or under this size are hydrated eagerly on load (see
// LoadModelPooled below); larger ones stay lazy, pool-only references for as
// long as Simplify()'s fixed point never actually needs their bytes (every
// value-reading onnxsim/onnx-optimizer pass treats an EXTERNAL tensor as
// unavailable rather than misreading its absent data -- see
// PoolExternalData's own comment). Matches onnx_simplifier.py's
// DEFAULT_TENSOR_SIZE_THRESHOLDHOLD ("a very very large threshold"): both
// exist to draw the same "this single tensor is unusually huge" line, just
// for two different decisions (there: whether to keep a constant-folded
// output; here: whether to eagerly materialize an input weight).
constexpr uint64_t kSimplifyPathHydrateThresholdBytes =
    1536ULL * 1024 * 1024;  // 1.5GB

void SimplifyPath(
    const ModelExecutor& executor, const std::string& in_path,
    const std::string& out_path,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output) {
  const bool debug_timing = std::getenv("ONNXSIM_DEBUG_PATH_TIMING") != nullptr;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto elapsed_ms = [](auto t0, auto t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  onnx::ModelProto model;
  // Declared here, not inside the load block below, so it stays alive
  // (keeping every mmap'd tensor's mapping alive) across the Simplify() call
  // -- a tensor PoolExternalData left EXTERNAL for being over
  // kSimplifyPathHydrateThresholdBytes is hydrated from *this* pool, on
  // demand, right before saveModel below, not eagerly here.
  onnxsim::tensor_pool::TensorPool pool;
  {
    const auto t0 = now();
    // The default model-loading step for onnxsim's primary path-based entry
    // point: reads the graph structure without materializing external data,
    // then memory-maps every classic-EXTERNAL tensor's data file straight
    // into a TensorPool instead of onnx-optimizer's own external-data loader
    // reading each tensor's slice into a fresh heap buffer. Tensors at or
    // under kSimplifyPathHydrateThresholdBytes are hydrated eagerly here;
    // larger ones stay lazy through the Simplify() call below.
    LoadModelPooled(in_path, &model, pool, kSimplifyPathHydrateThresholdBytes);
    if (debug_timing) {
      std::cerr << "SimplifyPath: loadModel " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }

  {
    const auto t0 = now();
    model =
        Simplify(executor, model, skip_optimizers, constant_folding,
                 shape_inference, tensor_size_threshold, target_opset_version,
                 rewriter, initializers_as_constants, include_inline_functions,
                 mutable_initializer, overwrite_input_shapes, unused_output);
    if (debug_timing) {
      std::cerr << "SimplifyPath: Simplify " << elapsed_ms(t0, now()) << "ms\n";
    }
  }

  {
    const auto t0 = now();
    // Flush any tensor LoadModelPooled left EXTERNAL (too large to
    // materialize eagerly, and never touched -- or already dropped by dead-
    // initializer elimination -- during Simplify()) back into ordinary
    // in-memory form, so the saved output below is always fully self-
    // contained no matter whether out_path's directory matches in_path's
    // (an EXTERNAL tensor's external_data location is relative to where it
    // was originally loaded from). See HydrateAllFromPool's own comment.
    onnxsim::tensor_pool::HydrateAllFromPool(model, pool);
    if (debug_timing) {
      std::cerr << "SimplifyPath: HydrateAllFromPool " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }

  // Prefer a single self-contained inline file over external data: onnx's
  // own checker (onnx.checker.check_model, called by every Python caller's
  // correctness check) is drastically slower validating an external-data
  // model than an equivalent inline one -- 18.6s vs ~4s measured on an
  // 833MB model, apparently from re-reading/re-validating the sibling data
  // file rather than working off the already-parsed in-memory tensors. Only
  // fall back to external data when the model does not fit in a single
  // protobuf message (the 2GB limit `saveModel`'s unconditional external-
  // data write used to sidestep unconditionally).
  constexpr size_t kProtobufSizeLimit = (size_t(2) << 30) - 9999;
  bool needs_external_data;
  {
    const auto t0 = now();
    needs_external_data = model.ByteSizeLong() >= kProtobufSizeLimit;
    if (debug_timing) {
      std::cerr << "SimplifyPath: ByteSizeLong " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }
  {
    const auto t0 = now();
    onnx::optimization::saveModel(&model, out_path, needs_external_data, "");
    if (debug_timing) {
      std::cerr << "SimplifyPath: saveModel " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }
}
