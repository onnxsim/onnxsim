#include "constant_folding.h"

#include <google/protobuf/arena.h>
#include <onnx/onnx_pb.h>

#ifdef ONNXSIM_HAS_ORT
#include "onnxruntime_cxx_api.h"
#endif
#include <algorithm>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "custom_optimizer_passes.h"
#include "dlpack_bridge.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/ir_pb_converter_internal.h"
#include "onnx/defs/schema.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/passes/logging.h"
#include "profiler.h"

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
        deferred_producers,
    const std::unordered_map<std::string, const onnx::NodeProto*>&
        constant_node_producers) {
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
          auto constant_iter = constant_node_producers.find(input);
          if (constant_iter != constant_node_producers.end()) {
            // Produced by a Constant node (pre-existing, or created by an
            // earlier fold batch/round because that fold was not purely
            // initializer-derived): its value has no initializer to look up,
            // so inline the (zero-input, so trivially includable) Constant
            // node itself instead.
            include_node(*constant_iter->second);
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

// Builds a `Constant` NodeProto whose sole output holds `output_tp` as its
// `value` attribute (verbatim, raw_data included).
onnx::NodeProto MakeConstantNode(const onnx::TensorProto& output_tp) {
  onnx::NodeProto node;
  node.set_op_type("Constant");
  node.add_output(output_tp.name());
  onnx::AttributeProto* attr = node.add_attribute();
  attr->set_name("value");
  attr->set_type(onnx::AttributeProto::TENSOR);
  *attr->mutable_t() = output_tp;
  return node;
}

// Materializes a successfully-folded batch's outputs into `model`: an output
// in `impure_outputs` (see ConstantNodePartition's own doc comment) becomes a
// fresh `Constant` node instead of a plain initializer. `new_constant_nodes`
// owns every such node for the rest of this fold pass (stable references,
// unlike a std::vector, since later batches keep pointers into it via
// `constant_node_producers`) and collects them for the caller to splice into
// the model's node list once folding is done (see _FoldConstant's
// RebuildNodeList step). `constant_node_producers` is grown with each new
// Constant node's output name so a later batch that consumes it can inline it
// via RunOps' own lookup, exactly like a pre-existing Constant node.
void RunOpsAndAddInitializers(
    const ModelExecutor& executor, onnx::ModelProto& model,
    const std::vector<const onnx::NodeProto*>& ops,
    const std::unordered_map<std::string, const onnx::NodeProto*>&
        deferred_producers,
    const std::set<std::string>& impure_outputs,
    std::unordered_map<std::string, const onnx::NodeProto*>&
        constant_node_producers,
    std::deque<onnx::NodeProto>& new_constant_nodes) {
  const auto output_tps =
      RunOps(executor, model, ops, deferred_producers, constant_node_producers);
  for (const auto& output_tp : output_tps) {
    if (impure_outputs.count(output_tp.name()) == 0) {
      *model.mutable_graph()->add_initializer() = output_tp;
      continue;
    }
    new_constant_nodes.push_back(MakeConstantNode(output_tp));
    constant_node_producers[output_tp.name()] = &new_constant_nodes.back();
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
  // Nodes whose output is materialized (into an initializer or a fresh
  // Constant node, see impure_outputs below) by folding.
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
  // Outputs of const_nodes whose value does not trace back purely to graph
  // initializers -- transitively, through a chain of other purely-initializer
  // folds -- because somewhere upstream it consumes a Constant node's embedded
  // value or a deferred output. These are materialized as a fresh Constant node
  // rather than an initializer, so a value the graph actually computed stays
  // visually distinct from literal weight data; see RunOpsAndAddInitializers.
  std::set<std::string> impure_outputs;
};

// Whether `node` (assumed op_type "Constant") was marked transient by another
// onnxsim pass -- see kTransientConstantAttr's own comment.
bool IsTransientConstant(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.name() == kTransientConstantAttr) {
      return true;
    }
  }
  return false;
}

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
  // Subset of const_names whose value traces back purely to graph
  // initializers -- transitively, through other purely-initializer folds --
  // as opposed to a Constant node's embedded value or a deferred (large-
  // tensor) output. See ConstantNodePartition::impure_outputs.
  std::unordered_set<std::string> pure_names{""};
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
      pure_names.insert(x.name());
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
    // A Constant node's output is already a fully-resolved value -- leave it
    // exactly as-is (never re-materialize it into an initializer or a new
    // Constant node) rather than folding it like any other node. This keeps
    // the pass idempotent: a Constant node this same function created on a
    // previous round (because a fold was not purely initializer-derived, see
    // impure_outputs) would otherwise be trivially "foldable" (zero inputs)
    // and get flattened straight back into an initializer, erasing the
    // provenance distinction on the very next fixed-point iteration. Its
    // output still joins const_names so downstream nodes recognize it as a
    // constant input -- but never pure_names, so any consumer is correctly
    // treated as not purely initializer-derived either. Exception: a node
    // marked transient (kTransientConstantAttr) is another onnxsim pass's own
    // intermediate representation, not a provenance-worthy Constant, so it
    // falls through to the ordinary foldable path below instead.
    const bool is_default_domain =
        node.domain().empty() || node.domain() == "ai.onnx";
    if (is_default_domain && node.op_type() == "Constant" &&
        !IsTransientConstant(node)) {
      const_names.insert(node.output().begin(), node.output().end());
      non_const_nodes.push_back(node);
      continue;
    }
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
      // Ordinary constant folding: the output is materialized (as an
      // initializer, or a Constant node if not purely initializer-derived)
      // and the node is dropped.
      const bool pure = std::all_of(
          node.input().begin(), node.input().end(),
          [&pure_names](const auto& x) { return pure_names.count(x) > 0; });
      const_names.insert(node.output().begin(), node.output().end());
      if (pure) {
        pure_names.insert(node.output().begin(), node.output().end());
      } else {
        partition.impure_outputs.insert(node.output().begin(),
                                        node.output().end());
      }
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
               const std::set<std::string>& impure_outputs,
               std::unordered_map<std::string, const onnx::NodeProto*>&
                   constant_node_producers,
               std::deque<onnx::NodeProto>& new_constant_nodes,
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
    RunOpsAndAddInitializers(executor, model, ops, deferred_producers,
                             impure_outputs, constant_node_producers,
                             new_constant_nodes);
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
              impure_outputs, constant_node_producers, new_constant_nodes,
              folded_outputs);
    FoldGroup(executor, model, const_nodes, mid, end, deferred_producers,
              impure_outputs, constant_node_producers, new_constant_nodes,
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
  // Map each pre-existing, non-transient Constant node's output to the node
  // itself, seeding the lookup RunOps uses to inline a Constant node's
  // embedded value instead of looking it up as an initializer -- including a
  // transient one (kTransientConstantAttr): unlike a genuine Constant node,
  // it is not special-cased out of GetConstantNodes and so flows through the
  // ordinary foldable path, but that fold can still fail and leave it
  // in place (FoldGroup's catch-and-skip), so a later batch may need to
  // inline it same as any other Constant. Safe to seed unconditionally here
  // (unlike the Graph-native counterpart below): this map's pointers alias
  // `model.graph().node()`, which nothing in this function's batch loop
  // frees or reallocates -- only RebuildNodeList, strictly after the loop,
  // drops folded entries. Grown as folding creates new Constant nodes for
  // impure outputs (see RunOpsAndAddInitializers); new_constant_nodes owns
  // those (a std::deque, not a std::vector, so the pointers this map holds
  // into it stay valid as more are appended).
  std::unordered_map<std::string, const onnx::NodeProto*>
      constant_node_producers;
  for (const auto& node : model.graph().node()) {
    const bool is_default_domain =
        node.domain().empty() || node.domain() == "ai.onnx";
    if (is_default_domain && node.op_type() == "Constant") {
      for (const auto& output : node.output()) {
        constant_node_producers.emplace(output, &node);
      }
    }
  }
  std::deque<onnx::NodeProto> new_constant_nodes;
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
                partition.impure_outputs, constant_node_producers,
                new_constant_nodes, folded_outputs);
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
              partition.impure_outputs, constant_node_producers,
              new_constant_nodes, folded_outputs);
    i = j;
  }
  // Rebuild the node list in its original topological order, dropping only
  // the const nodes that were successfully folded into initializers. A const
  // node that failed to fold must keep its original position: appending it to
  // the end can place it after a non-const consumer (e.g. a Loop reading a
  // SequenceEmpty output), which breaks topological sorting and makes the
  // resulting model fail onnx's checker (issues #238, #335, #352). Newly
  // created Constant nodes (impure folds) are prepended ahead of everything
  // else: they have no inputs of their own, so any position before their
  // first use is topologically valid, and the front trivially satisfies that
  // for every consumer regardless of where it sits in `original_nodes`.
  //
  // Every pre-existing Constant node kept as-is (GetConstantNodes' own
  // passthrough) is prepended alongside them, for the same reason: a
  // Constant node has no inputs, so moving it earlier is always safe, and a
  // model can carry one positioned after one of its own uses -- tolerated by
  // ONNX's checker (this is not the initializer list, whose order genuinely
  // doesn't matter) but not by the stricter topological-order check the
  // ModelProto<->Graph round trip inside Optimize() runs. Such a node used
  // to be silently normalized by extract_constant_to_initializer (which
  // converted it to an order-independent initializer) before that pass was
  // always disabled.
  {
    onnxsim::ProfiledScope rebuild_scope("RebuildNodeList");
    google::protobuf::RepeatedPtrField<onnx::NodeProto> original_nodes;
    original_nodes.Swap(model.mutable_graph()->mutable_node());
    google::protobuf::RepeatedPtrField<onnx::NodeProto> kept_constants;
    google::protobuf::RepeatedPtrField<onnx::NodeProto> kept_others;
    for (auto& node : original_nodes) {
      const bool folded =
          node.output_size() > 0 && folded_outputs.count(node.output(0)) > 0;
      if (folded) {
        continue;
      }
      const bool is_default_domain =
          node.domain().empty() || node.domain() == "ai.onnx";
      if (is_default_domain && node.op_type() == "Constant") {
        *kept_constants.Add() = std::move(node);
      } else {
        *kept_others.Add() = std::move(node);
      }
    }
    for (auto& node : new_constant_nodes) {
      *model.mutable_graph()->add_node() = std::move(node);
    }
    for (auto& node : kept_constants) {
      *model.mutable_graph()->add_node() = std::move(node);
    }
    for (auto& node : kept_others) {
      *model.mutable_graph()->add_node() = std::move(node);
    }
  }
  // Drop initializers left dangling by folding so the intermediate model does
  // not balloon in size (issue #174).
  onnxsim::ProfiledScope eliminate_scope("EliminateUnusedInitializerScope");
  return EliminateUnusedInitializer(std::move(model));
}

// --- Graph-native counterpart of GetConstantNodes/FoldGroup/_FoldConstant ---
//
// Ports the ORT-execution-based constant folder above to walk onnx::Graph
// directly, reached without a ModelProto <-> Graph round trip. Each fold
// batch still serializes to a small, throwaway sub-ModelProto right at the
// executor boundary (ORT has no Graph-native session API) -- exactly what
// RunOps already did, just built from Graph Node/Value/Tensor objects
// (reusing addAttribute/encodeTensor from ir_pb_converter_internal.h,
// graph_shape_inference.cc's own pattern) instead of NodeProto copies.
// Like the ModelProto path, _FoldConstantOnGraph runs onnx-optimizer's
// eliminate_unused_initializer pass directly (not just via
// config.optimizer_passes) at the end of every call: that pass list is
// empty when the caller asked to skip optimization entirely
// (perform_optimization=False), so nothing else is guaranteed to sweep up
// an initializer folding leaves dangling in that mode.

struct ConstantNodePartitionGraph {
  // Nodes whose output is materialized (into an initializer or a fresh
  // Constant node, see impure_outputs below) by folding, in topological
  // order.
  std::vector<onnx::Node*> const_nodes;
  // Unique names of "deferred" nodes' outputs -- see ConstantNodePartition's
  // own doc comment; same semantics, ported verbatim.
  std::unordered_set<std::string> deferred_outputs;
  // Outputs of const_nodes that do not trace back purely to graph
  // initializers -- see ConstantNodePartition::impure_outputs, same
  // semantics, ported verbatim.
  std::unordered_set<std::string> impure_outputs;
};

bool HasSubgraphAttr(onnx::Node* node) {
  for (onnx::Symbol name : node->attributeNames()) {
    const auto kind = node->kindOf(name);
    if (kind == onnx::AttributeKind::g || kind == onnx::AttributeKind::gs) {
      return true;
    }
  }
  return false;
}

// Graph-native counterpart of ProduceLargeTensor: reads the node's own
// output Value directly instead of looking it up in a separate value_info
// map (a Value already carries its own current shape/type in the IR).
bool ProduceLargeTensorOnGraph(onnx::Node* node, size_t threshold) {
  static const std::set<std::string> large_tensor_ops{"Tile", "ConstantOfShape",
                                                      "Expand"};
  if (large_tensor_ops.find(node->kind().toString()) ==
      large_tensor_ops.end()) {
    return false;
  }
  if (node->outputs().empty()) {
    return true;
  }
  onnx::Value* out = node->outputs()[0];
  if (out->elemType() == onnx::TensorProto::UNDEFINED || !out->has_sizes()) {
    return true;
  }
  size_t size;
  try {
    size = size_of_dtype(
        static_cast<onnx::TensorProto::DataType>(out->elemType()));
  } catch (const std::exception&) {
    return true;
  }
  for (const auto& d : out->sizes()) {
    if (!d.is_int) {
      return true;
    }
    size *= static_cast<size_t>(d.dim);
  }
  return size > threshold;
}

// Graph-native counterpart of EstimateOutputBytes.
size_t EstimateOutputBytesOnGraph(onnx::Node* node) {
  size_t total = 0;
  for (onnx::Value* out : node->outputs()) {
    if (out->elemType() == onnx::TensorProto::UNDEFINED || !out->has_sizes()) {
      continue;
    }
    size_t size;
    try {
      size = size_of_dtype(
          static_cast<onnx::TensorProto::DataType>(out->elemType()));
    } catch (const std::exception&) {
      continue;
    }
    bool known = true;
    for (const auto& d : out->sizes()) {
      if (!d.is_int) {
        known = false;
        break;
      }
      size *= static_cast<size_t>(d.dim);
    }
    if (known) {
      total += size;
    }
  }
  return total;
}

// Graph-native counterpart of IsTransientConstant.
bool IsTransientConstantOnGraph(onnx::Node* node) {
  static const onnx::Symbol kTransientAttr(kTransientConstantAttr);
  return node->hasAttribute(kTransientAttr);
}

// Graph-native counterpart of GetConstantNodes.
ConstantNodePartitionGraph GetConstantNodesOnGraph(
    onnx::Graph& g, const std::vector<onnx::Node*>& node_ptrs) {
  // Reference node every genuine Constant encountered below is moved just
  // before, in scan order -- see that branch's own comment for why.
  onnx::Node* const front_anchor =
      node_ptrs.empty() ? nullptr : node_ptrs.front();
  std::unordered_set<std::string> const_names{""};
  // Subset of const_names whose value traces back purely to graph
  // initializers -- see GetConstantNodes' own pure_names for the full
  // rationale, ported verbatim.
  std::unordered_set<std::string> pure_names{""};
  ConstantNodePartitionGraph partition;
  if (config.initializers_as_constants) {
    for (const auto& name : g.initializer_names()) {
      const_names.insert(name);
      pure_names.insert(name);
    }
  }
  std::unordered_map<std::string, int> domain_to_version;
  for (const onnx::OpSetID& opset : g.opset_versions_mutable()) {
    const std::string& domain =
        opset.domain() == "ai.onnx" ? "" : opset.domain();
    domain_to_version[domain] = static_cast<int>(opset.version());
  }
  auto opset_version_of = [&domain_to_version](const std::string& domain) {
    const std::string& key = domain == "ai.onnx" ? "" : domain;
    auto iter = domain_to_version.find(key);
    return iter == domain_to_version.end() ? 0 : iter->second;
  };
  for (onnx::Node* node : node_ptrs) {
    if (node->kind() == onnx::kUndefined || node->kind() == onnx::kCaptured) {
      continue;
    }
    const std::string domain =
        node->has_domain() ? node->domain() : std::string();
    const std::string op_type = node->kind().toString();
    // Leave Constant nodes untouched, transient ones excepted -- see
    // GetConstantNodes' own comment on its identical special case for the
    // full rationale (idempotence: a Constant node created by a previous
    // round for an impure fold must not be trivially re-folded straight back
    // into an initializer) and kTransientConstantAttr's own comment.
    const bool is_default_domain = domain.empty() || domain == "ai.onnx";
    if (is_default_domain && node->kind() == onnx::kConstant &&
        !IsTransientConstantOnGraph(node)) {
      // A Constant node has no inputs, so moving it earlier is always safe;
      // do so unconditionally rather than checking whether it is already
      // valid. A model can otherwise carry one positioned after one of its
      // own uses -- tolerated by ONNX's checker (this is not the initializer
      // list, whose order genuinely doesn't matter) but not by the stricter
      // topological-order check the ModelProto<->Graph round trip inside
      // Optimize() runs -- and such a node used to be silently normalized by
      // extract_constant_to_initializer (which converted it to an
      // order-independent initializer) before that pass was always disabled.
      // Moved just before `front_anchor` (in scan order, so relative order
      // among moved Constants is preserved) rather than the graph's true
      // first node, which may itself shift as earlier Constants in this same
      // scan are moved.
      if (front_anchor != nullptr && node != front_anchor) {
        node->moveBefore(front_anchor);
      }
      for (onnx::Value* out : node->outputs()) {
        const_names.insert(out->uniqueName());
      }
      continue;
    }
    const bool foldable =
        IsOfficialOp(domain, op_type) &&
        IsDeterministic(domain, op_type, opset_version_of(domain)) &&
        !IsQDQ(domain, op_type) && !HasSubgraphAttr(node) &&
        std::all_of(
            node->inputs().begin(), node->inputs().end(),
            [&const_names](onnx::Value* v) {
              const std::string name =
                  v->node()->kind() == onnx::kUndefined ? "" : v->uniqueName();
              return const_names.count(name) > 0;
            });
    if (!foldable) {
      continue;
    }
    if (!ProduceLargeTensorOnGraph(node, config.tensor_size_threshold)) {
      const bool pure = std::all_of(
          node->inputs().begin(), node->inputs().end(),
          [&pure_names](onnx::Value* v) {
            const std::string name =
                v->node()->kind() == onnx::kUndefined ? "" : v->uniqueName();
            return pure_names.count(name) > 0;
          });
      for (onnx::Value* out : node->outputs()) {
        const_names.insert(out->uniqueName());
        if (pure) {
          pure_names.insert(out->uniqueName());
        } else {
          partition.impure_outputs.insert(out->uniqueName());
        }
      }
      partition.const_nodes.push_back(node);
      continue;
    }
    if (op_type == "ConstantOfShape" || op_type == "Expand") {
      for (onnx::Value* out : node->outputs()) {
        const_names.insert(out->uniqueName());
        partition.deferred_outputs.insert(out->uniqueName());
      }
    }
  }
  return partition;
}

// Graph-native counterpart of RunOps: builds the throwaway sub-model from
// Graph Node/Value/Tensor objects. Each constant feed is encoded into its
// own owned TensorProto (encodeTensor copies its bytes once) and handed to
// the DLPack bridge via FromTensorProtoOwning, which takes ownership of the
// moved proto -- safe regardless of the source onnx::Tensor's lifetime,
// unlike borrowing a pointer into a temporary.
struct RunOpsOnGraphResult {
  std::vector<onnx::TensorProto> tensors;
  // Parallel to `tensors`: the original graph Value each entry replaces
  // (the node output RunOps computed a constant for).
  std::vector<onnx::Value*> values;
};

RunOpsOnGraphResult RunOpsOnGraph(
    const ModelExecutor& executor, onnx::Graph& g,
    const std::vector<onnx::Node*>& ops,
    const std::unordered_map<std::string, onnx::Node*>& deferred_producers,
    const std::unordered_map<std::string, onnx::Node*>& constant_node_producers,
    int64_t ir_version) {
  std::vector<std::string> input_names;
  std::vector<const onnx::Tensor*> input_tensors;
  std::set<std::string> seen_inputs;

  google::protobuf::Arena arena;
  onnx::ModelProto& op_model =
      *google::protobuf::Arena::Create<onnx::ModelProto>(&arena);
  // Inherited from the model being simplified (not onnx::Version::IR_VERSION,
  // this library's own compiled-in constant): onnxsim's vendored onnx tracks
  // upstream's IR_VERSION ahead of what any released onnxruntime actually
  // supports, and this throwaway sub-model still has to load in a real
  // onnxruntime session for its one Run() call -- the original model's own
  // ir_version is what the caller's onnxruntime install already proved it
  // could handle.
  op_model.set_ir_version(ir_version);
  for (const onnx::OpSetID& opset : g.opset_versions_mutable()) {
    auto* x = op_model.add_opset_import();
    x->set_domain(opset.domain());
    x->set_version(opset.version());
  }

  std::set<std::string> internal_outputs;
  for (onnx::Node* op : ops) {
    for (onnx::Value* out : op->outputs()) {
      internal_outputs.insert(out->uniqueName());
    }
  }

  std::set<onnx::Node*> included;
  std::function<void(onnx::Node*)> include_node = [&](onnx::Node* node) {
    if (!included.insert(node).second) {
      return;
    }
    for (onnx::Value* input : node->inputs()) {
      if (input->node()->kind() == onnx::kUndefined) {
        continue;  // unset optional input
      }
      const std::string& name = input->uniqueName();
      if (internal_outputs.count(name) > 0) {
        continue;
      }
      auto deferred_iter = deferred_producers.find(name);
      if (deferred_iter != deferred_producers.end()) {
        include_node(deferred_iter->second);
        continue;
      }
      auto constant_iter = constant_node_producers.find(name);
      if (constant_iter != constant_node_producers.end()) {
        // Produced by a Constant node (pre-existing, or created by an
        // earlier fold batch/round because that fold was not purely
        // initializer-derived): its value has no initializer to look up, so
        // inline the (zero-input, so trivially includable) Constant node
        // itself instead.
        include_node(constant_iter->second);
        continue;
      }
      if (!seen_inputs.insert(name).second) {
        continue;
      }
      const onnx::Tensor* init = g.getInitializer(name);
      if (init == nullptr) {
        // Mirrors FindInitializerByName's own throwing behavior: `name`
        // was classified constant during GetConstantNodesOnGraph's static
        // partition, but its producer never actually materialized an
        // initializer for it (e.g. an earlier batch's fold for it failed
        // and was skipped, see FoldGroupOnGraph's catch clause). Throwing
        // here lets the same catch-and-bisect/skip machinery handle this
        // node too, instead of crashing.
        throw std::invalid_argument("no initializer " + name);
      }
      if (init->sizes().size() == 1 && init->sizes()[0] == 0) {
        onnx::TensorProto* p = op_model.mutable_graph()->add_initializer();
        p->set_name(name);
        onnx::encodeTensor(*p, *init);
        continue;
      }
      input_names.push_back(name);
      input_tensors.push_back(init);
    }
    onnx::NodeProto* np = op_model.mutable_graph()->add_node();
    np->set_op_type(node->kind().toString());
    if (node->has_domain()) {
      np->set_domain(node->domain());
    }
    for (onnx::Value* input : node->inputs()) {
      np->add_input(
          input->node()->kind() == onnx::kUndefined ? "" : input->uniqueName());
    }
    for (onnx::Value* output : node->outputs()) {
      np->add_output(output->uniqueName());
    }
    for (onnx::Symbol attr_name : node->attributeNames()) {
      onnx::addAttribute(*np, *node, attr_name, /*consume_tensor_data=*/false);
    }
  };
  for (onnx::Node* op : ops) {
    include_node(op);
  }

  for (size_t i = 0; i < input_names.size(); i++) {
    onnx::ValueInfoProto* vi = op_model.mutable_graph()->add_input();
    vi->set_name(input_names[i]);
    auto* tensor_type = vi->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(input_tensors[i]->elem_type());
    for (int64_t d : input_tensors[i]->sizes()) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(d);
    }
  }
  std::vector<std::string> output_names;
  std::vector<onnx::Value*> output_values;
  for (onnx::Node* op : ops) {
    for (onnx::Value* out : op->outputs()) {
      op_model.mutable_graph()->add_output()->set_name(out->uniqueName());
      output_names.push_back(out->uniqueName());
      output_values.push_back(out);
    }
  }

  std::vector<DLManagedTensorPtr> input_dls;
  std::vector<const DLManagedTensor*> input_ptrs;
  input_dls.reserve(input_tensors.size());
  for (const auto* t : input_tensors) {
    onnx::TensorProto tp;
    onnx::encodeTensor(tp, *t);
    input_dls.emplace_back(
        onnxsim::dlpack::FromTensorProtoOwning(std::move(tp)));
  }
  input_ptrs.reserve(input_dls.size());
  for (const auto& p : input_dls) input_ptrs.push_back(p.get());

  std::vector<DLManagedTensorPtr> output_dls;
  {
    // Profiled under the same "OrtSession" span name as the ModelProto-based
    // RunOps (see its own comment): the one spot common to every executor
    // binding, so a trace can find the actual session-run time regardless of
    // which folding path produced it.
    onnxsim::ProfiledScope session_scope("OrtSession");
    output_dls = executor.Run(op_model, input_ptrs);
  }
  RunOpsOnGraphResult result;
  result.tensors.reserve(output_dls.size());
  for (size_t i = 0; i < output_dls.size(); i++) {
    result.tensors.push_back(onnxsim::dlpack::ToTensorProto(
        output_dls[i]->dl_tensor,
        i < output_names.size() ? output_names[i] : std::string()));
  }
  // executor.Run() returns one tensor per graph output, in the same order
  // output_values (built alongside output_names above) lists them in.
  result.values = output_values;
  result.values.resize(result.tensors.size());
  return result;
}

// Graph-native counterpart of FoldGroup: splices each successfully-folded
// output directly into `g` -- as a new initializer (Graph::
// addInitializerAndCreateValue) when it is in `impure_outputs` (see
// ConstantNodePartitionGraph's own doc comment), a fresh Constant node
// otherwise -- rewires the folded node's uses onto it, and removes the
// folded node -- no NodeProto rebuild needed. `constant_node_producers` is
// grown with each new Constant node so a later batch/round that consumes it
// can inline it via RunOpsOnGraph's own lookup, exactly like a pre-existing
// Constant node.
void FoldGroupOnGraph(
    const ModelExecutor& executor, onnx::Graph& g,
    const std::vector<onnx::Node*>& const_nodes, size_t begin, size_t end,
    const std::unordered_map<std::string, onnx::Node*>& deferred_producers,
    const std::unordered_set<std::string>& impure_outputs,
    std::unordered_map<std::string, onnx::Node*>& constant_node_producers,
    size_t& num_folded, int64_t ir_version) {
  if (begin >= end) {
    return;
  }
  std::vector<onnx::Node*> ops(const_nodes.begin() + begin,
                               const_nodes.begin() + end);
  // Only RunOpsOnGraph itself -- which merely evaluates `ops` and does not
  // touch `g` -- is retried via bisection on failure below: at this point
  // nothing in `g` has been mutated yet, so re-deriving `ops` from
  // `const_nodes[begin, mid)`/`[mid, end)` and re-running is always safe.
  // The splice-into-`g` loop that follows is deliberately kept outside this
  // try/catch (see its own comment below for why).
  RunOpsOnGraphResult result;
  try {
    result = RunOpsOnGraph(executor, g, ops, deferred_producers,
                           constant_node_producers, ir_version);
  } catch (const std::exception& e) {
    if (end - begin == 1) {
      onnx::Node* node = const_nodes[begin];
      std::cerr << "WARNING: failed to run \"" << node->kind().toString()
                << "\" op (name is \"" << node->name() << "\"), skip... "
                << e.what() << std::endl;
      return;
    }
    const size_t mid = begin + (end - begin) / 2;
    FoldGroupOnGraph(executor, g, const_nodes, begin, mid, deferred_producers,
                     impure_outputs, constant_node_producers, num_folded,
                     ir_version);
    FoldGroupOnGraph(executor, g, const_nodes, mid, end, deferred_producers,
                     impure_outputs, constant_node_producers, num_folded,
                     ir_version);
    return;
  }

  // Every op in this batch folded successfully (RunOpsOnGraph throws,
  // rather than partially populating its result, on any failure) -- decode
  // each returned TensorProto (ToTensorProto always emits raw_data, see
  // dlpack_bridge.h) into a Tensor, add it as a new initializer or Constant
  // node, and rewire the Value it replaces onto it. A multi-output node's
  // outputs are independent Values here, each replaced on its own; the
  // owning node is only destroyed (never touched again after) once every
  // one of its outputs has been replaced.
  //
  // Deliberately not wrapped in the try/catch above (nor any other): unlike
  // RunOpsOnGraph, this loop mutates `g` as it goes (replaceAllUsesWith,
  // Node::destroy), so a failure partway through leaves some of `ops`
  // already spliced/destroyed. The old code caught exceptions from this
  // whole function body in one try/catch, so a failure here after some
  // nodes were already destroyed fell into the same bisect-and-retry path
  // as a RunOpsOnGraph failure -- but that retry re-derives `ops` from
  // `const_nodes[begin, end)` again, which still holds pointers to the
  // now-destroyed (freed) nodes: a real use-after-free, observed as this
  // exact code taking down the whole process nondeterministically (a clean
  // assertion failure one run, a segfault or heap-corruption-flavored
  // exception the next) on real-world models (see the Detectron2 export
  // path's own tests). The fix has two parts: (1) this loop no longer
  // retries on failure at all -- seeing the defensive check below, it
  // should now never throw in the first place; (2) the check itself, so
  // there is nothing left here for a stale retry to have papered over.
  std::unordered_map<onnx::Node*, size_t> remaining_outputs;
  for (onnx::Node* node : ops) {
    remaining_outputs[node] = node->outputs().size();
  }
  for (size_t i = 0; i < result.tensors.size(); i++) {
    const onnx::TensorProto& tp = result.tensors[i];
    onnx::Value* old_value = result.values[i];
    onnx::Node* owner = old_value->node();
    onnx::Tensor t;
    t.setName(tp.name());
    t.elem_type() = tp.data_type();
    for (int64_t d : tp.dims()) t.sizes().push_back(d);
    if (tp.has_raw_data()) {
      t.set_raw_data(tp.raw_data());
    }
    onnx::Value* new_value;
    if (impure_outputs.count(tp.name()) == 0) {
      new_value = g.addInitializerAndCreateValue(t);
    } else {
      // Not purely initializer-derived: materialize as a Constant node
      // instead of an initializer. Prepended at the very front of the
      // graph (rather than inserted at the folded node's old position):
      // a Constant node has no inputs of its own, so the front is always
      // topologically valid regardless of where its consumers -- or, for
      // a multi-batch fold, nodes not yet visited in this same call --
      // currently sit.
      onnx::Node* constant = g.create(onnx::kConstant, 1);
      constant->t_(onnx::kvalue, t);
      std::vector<onnx::Dimension> sizes;
      sizes.reserve(tp.dims_size());
      for (int64_t d : tp.dims()) {
        sizes.emplace_back(d);
      }
      constant->output()->setSizes(sizes);
      constant->output()->setElemType(tp.data_type());
      constant->output()->setUniqueName(tp.name());
      g.prependNode(constant);
      constant_node_producers[tp.name()] = constant;
      new_value = constant->output();
    }
    old_value->replaceAllUsesWith(new_value);
    if (--remaining_outputs[owner] == 0) {
      // If `owner` is itself a (transient) Constant node seeded into
      // constant_node_producers, drop its entry before destroying it --
      // see the seeding loop's own comment on why transient nodes are
      // seeded despite being foldable: this is the other half of that,
      // keeping the map from ever pointing at freed memory.
      constant_node_producers.erase(tp.name());
      // Defensive: every one of owner's outputs has its own entry in
      // result.tensors, so by the time remaining_outputs[owner] reaches 0
      // each has already individually gone through replaceAllUsesWith
      // above and should have zero uses left -- Node::destroy() asserts
      // exactly that (onnx::Node::eraseOutput's
      // outputs_[i]->uses().empty()). That assumption has been observed to
      // not hold on some real-world models for reasons not fully
      // root-caused (see this function's own comment above); asserting on
      // it via destroy() takes the whole process down, nondeterministically
      // (a clean exception one run, a segfault the next -- signs of memory
      // corruption once it happens, not just a benign logic bug). Checking
      // it explicitly here instead means the fold degrades safely: `owner`
      // is left in the graph as dead code (unreachable, since nothing
      // holds a live use for a value none of the checks above found any
      // uses for) for eliminate_deadend/eliminate_unused_initializer to
      // sweep up later, instead of corrupting the graph.
      const bool all_outputs_unused =
          std::all_of(owner->outputs().begin(), owner->outputs().end(),
                     [](onnx::Value* v) { return v->uses().empty(); });
      if (all_outputs_unused) {
        owner->destroy();
      }
    }
  }
  num_folded += end - begin;
}

// Graph-native counterpart of _FoldConstant. Returns whether anything
// folded, mirroring OptimizeGraphChanged's own bool-returning convention so
// the outer fixed point can detect convergence without a fingerprint
// comparison. Always sweeps unused initializers at the end (see this
// section's own top-of-file comment for why).
bool _FoldConstantOnGraph(const ModelExecutor& executor, onnx::Graph& g,
                          int64_t ir_version) {
  std::vector<onnx::Node*> node_ptrs(g.nodes().begin(), g.nodes().end());
  ConstantNodePartitionGraph partition;
  {
    onnxsim::ProfiledScope analysis_scope("GetConstantNodes");
    partition = GetConstantNodesOnGraph(g, node_ptrs);
  }
  const auto& const_nodes = partition.const_nodes;
  if (const_nodes.empty()) {
    // Nothing to fold this call, but a dangling initializer could already be
    // sitting in `g` from a previous round or the input model itself -- see
    // this function's own doc comment for why this sweep cannot be skipped.
    onnx::optimization::EliminateUnusedInitializer()
        .eliminate_unused_initializer(g);
    return false;
  }

  std::unordered_map<std::string, onnx::Node*> deferred_producers;
  if (!partition.deferred_outputs.empty()) {
    for (onnx::Node* node : node_ptrs) {
      for (onnx::Value* out : node->outputs()) {
        if (partition.deferred_outputs.count(out->uniqueName()) > 0) {
          deferred_producers.emplace(out->uniqueName(), node);
        }
      }
    }
  }
  // Map each pre-existing, non-transient Constant node's output to the node
  // itself, seeding the lookup RunOpsOnGraph uses to inline a Constant node's
  // embedded value instead of looking it up as an initializer. Grown as
  // folding creates new Constant nodes for impure outputs (see
  // FoldGroupOnGraph) -- including a transient one (kTransientConstantAttr):
  // unlike a genuine Constant node, it is not special-cased out of
  // GetConstantNodesOnGraph and so flows through the ordinary foldable path,
  // but that fold can still fail and leave it in place
  // (FoldGroupOnGraph's catch-and-skip), so a later batch may need to inline
  // it same as any other Constant. A transient node's fold *succeeding*
  // destroys it (FoldGroupOnGraph's `owner->destroy()`), which would
  // otherwise leave this map holding a dangling Node* -- FoldGroupOnGraph
  // erases the entry itself right before destroying the node, so seeding it
  // here is safe.
  std::unordered_map<std::string, onnx::Node*> constant_node_producers;
  for (onnx::Node* node : node_ptrs) {
    const std::string domain =
        node->has_domain() ? node->domain() : std::string();
    const bool is_default_domain = domain.empty() || domain == "ai.onnx";
    if (is_default_domain && node->kind() == onnx::kConstant) {
      for (onnx::Value* out : node->outputs()) {
        constant_node_producers.emplace(out->uniqueName(), node);
      }
    }
  }

  constexpr size_t kBatchByteBudget = size_t(256) << 20;  // 256 MiB
  constexpr size_t kBatchMaxNodes = 1024;
  auto consumes_deferred = [&](onnx::Node* node) {
    if (partition.deferred_outputs.empty()) {
      return false;
    }
    for (onnx::Value* input : node->inputs()) {
      if (partition.deferred_outputs.count(input->uniqueName()) > 0) {
        return true;
      }
    }
    return false;
  };

  size_t num_folded = 0;
  const size_t num_const_nodes = const_nodes.size();
  for (size_t i = 0; i < num_const_nodes;) {
    if (consumes_deferred(const_nodes[i])) {
      FoldGroupOnGraph(executor, g, const_nodes, i, i + 1, deferred_producers,
                       partition.impure_outputs, constant_node_producers,
                       num_folded, ir_version);
      i++;
      continue;
    }
    size_t j = i;
    size_t bytes = 0;
    while (j < num_const_nodes && j - i < kBatchMaxNodes &&
           !consumes_deferred(const_nodes[j])) {
      const size_t node_bytes = EstimateOutputBytesOnGraph(const_nodes[j]);
      if (j > i && bytes + node_bytes > kBatchByteBudget) {
        break;
      }
      bytes += node_bytes;
      j++;
    }
    FoldGroupOnGraph(executor, g, const_nodes, i, j, deferred_producers,
                     partition.impure_outputs, constant_node_producers,
                     num_folded, ir_version);
    i = j;
  }
  // Drop initializers left dangling by folding so the graph does not balloon
  // in size (issue #174) -- see this function's own doc comment for why this
  // cannot be left to config.optimizer_passes alone.
  onnx::optimization::EliminateUnusedInitializer().eliminate_unused_initializer(
      g);
  return num_folded > 0;
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
