#include "onnxsim.h"

#include <google/protobuf/io/zero_copy_stream.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <string>
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
#include "bev_custom_op_schemas.h"
#include "constant_folding.h"
#include "contrib_schemas.h"
#include "custom_optimizer_passes.h"
#include "gemm_fusion_backend.h"
#include "model_info.h"
#include "model_prep.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/graph_shape_inference.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/printer.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"
#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/logging.h"
#include "partial_shape_eval.h"
#include "profiler.h"
#include "quantize_entry.h"

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
// "onnxsim." -- the same dotted convention as the rest of onnxsim's own
// metadata (AnnotateModelInfo's "onnxsim.macs" and friends, and the Python
// side's ``model_info.METADATA_PREFIX``) -- so they cannot collide with the
// model's own metadata. Lets a downstream consumer see how a model was
// simplified without having to have kept the original call around. Replaces
// any pre-existing entries under these exact keys (e.g. left by a previous
// simplify() call) rather than accumulating duplicates across repeated
// simplification; unlike a prefix wipe, this leaves other "onnxsim.*"
// entries (compute metrics, the structural diff recorded by
// RecordSimplifyDiffMetadata) untouched.
void RecordSimplifyOptionsMetadata(
    onnx::ModelProto& model,
    const std::optional<std::vector<std::string>>& skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, bool initializers_as_constants,
    bool include_inline_functions, bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output,
    const std::optional<std::vector<std::string>>& extra_optimizers) {
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
  options.emplace_back("onnxsim.skip_optimizers",
                       skip_optimizers ? join(*skip_optimizers) : "<all>");
  options.emplace_back("onnxsim.constant_folding",
                       constant_folding ? "true" : "false");
  options.emplace_back("onnxsim.shape_inference",
                       shape_inference ? "true" : "false");
  options.emplace_back("onnxsim.tensor_size_threshold",
                       std::to_string(tensor_size_threshold));
  options.emplace_back(
      "onnxsim.target_opset_version",
      target_opset_version ? std::to_string(*target_opset_version) : "");
  options.emplace_back("onnxsim.initializers_as_constants",
                       initializers_as_constants ? "true" : "false");
  options.emplace_back("onnxsim.include_inline_functions",
                       include_inline_functions ? "true" : "false");
  options.emplace_back("onnxsim.mutable_initializer",
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
    options.emplace_back("onnxsim.overwrite_input_shapes", s);
  }
  if (unused_output) {
    options.emplace_back("onnxsim.unused_output", join(*unused_output));
  }
  if (extra_optimizers) {
    options.emplace_back("onnxsim.extra_optimizers", join(*extra_optimizers));
  }

  // Every key this function could ever write, not just the ones present in
  // `options` this call -- ``overwrite_input_shapes``/``unused_output``/
  // ``extra_optimizers`` are conditional, so a stale value one of them left
  // behind on a previous simplify() call (that did set it) must still be
  // cleared on a later call that doesn't, or it would look like it's still
  // in effect.
  static const std::unordered_set<std::string> known_option_keys = {
      "onnxsim.skip_optimizers",          "onnxsim.constant_folding",
      "onnxsim.shape_inference",          "onnxsim.tensor_size_threshold",
      "onnxsim.target_opset_version",     "onnxsim.initializers_as_constants",
      "onnxsim.include_inline_functions", "onnxsim.mutable_initializer",
      "onnxsim.overwrite_input_shapes",   "onnxsim.unused_output",
      "onnxsim.extra_optimizers"};

  auto* props = model.mutable_metadata_props();
  google::protobuf::RepeatedPtrField<onnx::StringStringEntryProto> kept;
  for (auto& prop : *props) {
    if (!known_option_keys.count(prop.key())) {
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

// Number of nodes currently in `graph`'s main node list. Only used to feed
// the profiler's per-round "node reduction" counter (see RecordNodeCount
// call sites below); callers already guard this behind
// Profiler::Instance().enabled() so the O(n) walk never runs when profiling
// is off.
size_t CountGraphNodes(const onnx::Graph& graph) {
  return static_cast<size_t>(
      std::distance(graph.nodes().begin(), graph.nodes().end()));
}

// Builds the map InferShapesOnGraph (via OptAndShapeOnGraph/
// _EvalPartialShapeOnGraph below) needs to infer through a model-local
// function call: onnx::Graph carries no notion of model.functions() itself
// (see graph_shape_inference.h's own doc comment), so any caller with a
// ModelProto in hand builds this once, the same way
// shape_inference::InferShapes(ModelProto&, ...) does internally for the
// protobuf-based path. Key format (domain:name, or domain:name:overload for
// an IR>=10 function overload) matches what every ONNX helper expects --
// see graph_shape_inference.h's own doc comment.
onnx::shape_inference::ModelLocalFunctionsMap BuildModelLocalFunctionsMap(
    const onnx::ModelProto& model) {
  onnx::shape_inference::ModelLocalFunctionsMap functions;
  functions.reserve(static_cast<size_t>(model.functions_size()));
  for (const auto& fn : model.functions()) {
    std::string id = fn.domain() + ":" + fn.name();
    if (!fn.overload().empty()) {
      id += ":" + fn.overload();
    }
    functions.emplace(std::move(id), &fn);
  }
  return functions;
}

// Runs InferShapesOnGraph + OptimizeGraphFixed to their own inner fixed
// point directly on `g`, with no ModelProto conversion at either end -- the
// Graph-resident core shared by OptAndShape's ModelFn (which wraps this in
// one Import before and one Export after, see below) and the fully
// Graph-native outer Pipeline (which shares one Import/Export across the
// *whole* outer fixed point instead, see Simplify's !rewriter branch).
// `model_local_functions` is forwarded unchanged to InferShapesOnGraph (see
// that function's own doc comment) -- Graph carries no notion of
// model-local functions itself, so a caller whose model has any must build
// this map from its own ModelProto.functions() and pass it in; omitted
// (the default, empty), a model-local function call's output is left
// untouched, exactly as if the op had no registered schema.
// Returns whether anything changed.
bool OptAndShapeOnGraph(onnx::Graph& g, bool optimize, bool shape_inference,
                        size_t fixed_point_iters,
                        const onnx::shape_inference::ModelLocalFunctionsMap&
                            model_local_functions = {}) {
  // See OptAndShape's own doc comment for why these caches are scoped to
  // one call of this function rather than cleared every round underneath.
  onnx::optimization::ClearTensorContentDigestCache();
  onnx::optimization::ClearRawHashCache();
  using GraphFnChanged = std::function<bool(onnx::Graph&)>;
  bool any_changed = false;
  GraphFnChanged InferShapesOnGraphChanged =
      shape_inference ? GraphFnChanged([&any_changed, &model_local_functions](
                                           onnx::Graph& graph) {
        onnxsim::ProfiledScope scope("InferShapes");
        const bool c =
            onnx::InferShapesOnGraph(graph, onnx::ShapeInferenceOptions(),
                                     nullptr, model_local_functions);
        any_changed |= c;
        return c;
      })
                      : GraphFnChanged([](onnx::Graph&) { return false; });
  GraphFnChanged OptimizeGraphChanged = [optimize,
                                         &any_changed](onnx::Graph& graph) {
    onnxsim::ProfiledScope scope("Optimize");
    onnxsim::RegisterCustomOptimizerPasses();
    if (optimize) {
      // Unset all-zero recurrent initial states (issue #314) so the
      // subgraph computing them becomes dead and the passes below can
      // remove it. Not reflected in the "changed" signal below, but that
      // is safe: see FixedPointFn's bool-returning overload comment -- any
      // resulting dead-end elimination is itself reflected in the report.
      EliminateZeroRnnInitialState(graph);
    }
    const bool prev = onnx::optimization::InitializersAsConstants();
    onnx::optimization::SetInitializersAsConstants(
        config.initializers_as_constants);
    std::map<std::string, unsigned int> report;
    onnx::optimization::OptimizeGraphFixed(graph, config.optimizer_passes,
                                           &report,
                                           /*clear_tensor_digest_cache=*/false);
    onnx::optimization::SetInitializersAsConstants(prev);
    bool changed = false;
    for (const auto& pass_count : report) {
      if (pass_count.second != 0) {
        changed = true;
        break;
      }
    }
    any_changed |= changed;
    if (onnxsim::Profiler::Instance().enabled()) {
      onnxsim::Profiler::Instance().RecordNodeCount("Optimize",
                                                    CountGraphNodes(graph));
    }
    return changed;
  };
  FixedPointFn(InferShapesOnGraphChanged, OptimizeGraphChanged,
               fixed_point_iters)(g);
  return any_changed;
}

// Shared body of Simplify()/SimplifyConsumeInput() below, differing only in
// how ``sim_model`` -- the mutable working copy the fixed point runs on --
// gets built from ``model``. ``mutable_model`` is null (and ``model`` is
// deep-copied into ``sim_model``, as always) for the plain, input-preserving
// path; when non-null, it aliases ``model`` and this instead does the same
// move-based ModelProto -> Graph -> ModelProto round trip already used for
// OptAndShape's own resident Graph above (and Optimizer::optimize()'s own
// consuming overload in onnxoptimizer/optimize.h) -- see
// SimplifyConsumeInput's own doc comment for when that's safe to ask for.
static onnx::ModelProto SimplifyImpl(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    onnx::ModelProto* mutable_model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output,
    const std::optional<std::vector<std::string>>& extra_optimizers) {
  // Register onnxsim's own optimizer passes into onnxoptimizer's registry
  // before the pass list is built below: fuse_attention, fuse_consecutive_mul,
  // fuse_mul_into_conv, fuse_preceding_mul_into_conv, fuse_rms_norm and
  // eliminate_reshape_around_elementwise are auto-selected via
  // GetFuseAndEliminationPass, and fuse_matmul_add_bias_into_gemm_batched is
  // named explicitly. The call is idempotent.
  onnxsim::RegisterCustomOptimizerPasses();
  // Make shape inference aware of ONNX Runtime's quantized contrib operators
  // (QLinearAdd and friends) so shape deduction does not stop at them.
  onnxsim::RegisterContribOpSchemas();
  // Likewise for the mmdeploy/mmcv/BEVDet custom ops this branch's
  // rewrite_*_to_* passes decompose (MMCVMultiScaleDeformableAttention,
  // MMCVDeformConv2d/MMCVModulatedDeformConv2d, TRTBatchedNMS/
  // TRTBatchedRotatedNMS, bev_pool_v2).
  onnxsim::RegisterBevCustomOpSchemas();
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
  // Always keep ``Constant`` nodes in producer form: onnxsim's own constant
  // folder (constant_folding.cpp's GetConstantNodes/GetConstantNodesOnGraph)
  // now leaves a ``Constant`` node untouched rather than baking it into an
  // initializer, on the theory that a Constant node's value is not "graph
  // weight data" the way an initializer is -- and materializes any fold that
  // is not purely initializer-derived (directly or transitively) as a fresh
  // Constant node of its own, so that distinction stays visible in the
  // output model. ``extract_constant_to_initializer`` would erase it right
  // back by rewriting every Constant into an initializer -- including ones
  // onnxsim itself just created -- so it is always dropped from the pass
  // list, not just when initializers are treated as non-constant (where it
  // additionally has the correctness problem that its output, being
  // non-constant, would block further folding of genuinely-constant
  // subgraphs). The value-baking passes themselves already leave initializer
  // weights alone via ``IsConstantTensor`` (see the onnx-optimizer changes),
  // so only this representation-changing pass needs dropping.
  std::vector<std::string> always_disabled_passes = {
      "eliminate_shape_gather", "extract_constant_to_initializer"};
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
    // extra_optimizers is the general form of the batched_gemm carve-out
    // above: a caller-named pass -- typically PassType::Other, since
    // GetFuseAndEliminationPass already covers every Fuse/Nop pass -- runs in
    // addition to the default set. skip_optimizers/always_disabled_passes
    // still apply, so an explicit --skip-optimization wins, and a name
    // already present (e.g. the caller redundantly names a default-set pass)
    // is not duplicated. An unknown name is deliberately not filtered out
    // here: OptimizeGraphFixed's Optimizer construction looks every entry of
    // ``passes`` up in the global pass registry and throws on one that does
    // not exist, which is the desired behavior -- see extra_optimizers' own
    // doc comment in onnxsim.h.
    if (extra_optimizers) {
      for (const auto& pass : *extra_optimizers) {
        if (!is_disabled(*skip_optimizers, pass) &&
            !is_disabled(always_disabled_passes, pass) &&
            std::find(passes.begin(), passes.end(), pass) == passes.end()) {
          passes.push_back(pass);
        }
      }
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
      if (onnxsim::Profiler::Instance().enabled()) {
        onnxsim::Profiler::Instance().RecordNodeCount(
            "FoldConstant", static_cast<size_t>(model.graph().node_size()));
      }
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

  // Which runtime fuse_matmul_add_bias_into_gemm(_batched) should assume will
  // execute the simplified model -- see gemm_fusion_backend.h. Mirrors
  // ONNXSIM_FIXED_POINT_ITERS/ONNXSIM_PROFILE so it works from every binding
  // without a signature change; set unconditionally (not just when the env
  // var is present) so this call never inherits a setting left over from an
  // unrelated prior Simplify() call in the same process.
  onnxsim::SetGemmFusionBackend(
      std::getenv("ONNXSIM_GEMM_FUSION_BACKEND")
          ? onnxsim::ParseGemmFusionBackend(
                std::getenv("ONNXSIM_GEMM_FUSION_BACKEND"))
          : onnxsim::GemmFusionBackend::kOrtCpu);

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
  // Baseline point for the "node reduction per loop" plot: the node count
  // before any fixed-point round has run.
  if (onnxsim::Profiler::Instance().enabled()) {
    onnxsim::Profiler::Instance().RecordNodeCount(
        "Initial", static_cast<size_t>(model.graph().node_size()));
  }
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
        // run (see onnxsim issue #633).
        OptAndShapeOnGraph(*g, optimize, shape_inference, fixed_point_iters,
                           BuildModelLocalFunctionsMap(model));
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
  // Set from `model.ir_version()` right before `Pipeline` runs at the bottom
  // of this function -- fold's throwaway sub-models need it (see
  // RunOpsOnGraph's own comment). Declared here, alongside `converged`, and
  // not inside the `else` branch below that builds `Pipeline`'s closures:
  // those closures (and the reference to this variable they capture) are
  // still called from `Pipeline(sim_model)` well after that branch's own
  // scope -- and this variable's lifetime, were it declared there instead --
  // has ended.
  int64_t fold_ir_version = 0;
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
          if (onnxsim::Profiler::Instance().enabled()) {
            onnxsim::Profiler::Instance().RecordNodeCount(
                "Rewrite", static_cast<size_t>(model.graph().node_size()));
          }
        });
    Pipeline =
        Profiled("Pipeline", FixedPointFn(OptAndShapeAndFold, RewriteInPlace,
                                          fixed_point_iters, &converged));
  } else {
    // Fully Graph-resident outer pipeline: OptAndShapeOnGraph and the
    // Graph-native constant folder (_EvalPartialShapeOnGraph +
    // _FoldConstantOnGraph) both run directly against one onnx::Graph held
    // for the *entire* outer fixed point, so this whole Simplify() call
    // does exactly one Import and one Export -- instead of one Import+
    // Export per outer round, which #637/#638 already eliminated at the
    // *inner* (OptAndShape) level but which this pipeline still paid once
    // per outer round until now. Only available without a rewriter:
    // onnxscript's rewriter only understands ModelProto (see the `if
    // (rewriter)` branch above), so a supplied rewriter still needs the
    // ModelProto-based OptAndShape/FoldConstant path.
    using GraphFn = std::function<void(onnx::Graph&)>;
    using GraphFnChanged = std::function<bool(onnx::Graph&)>;
    // Built once from the outer, pre-simplification `model` (SimplifyImpl's
    // own parameter, not `sim_model` or any later ModelFn's own `model`
    // parameter) and captured by value into both closures below: neither
    // OptAndShapeOnGraph nor _EvalPartialShapeOnGraph's own InferShapesOnGraph
    // call otherwise has any way to reach model.functions(), and this map
    // needs to outlive every round of the fixed point these two closures
    // drive. Still correct when include_inline_functions has already
    // inlined `sim_model`'s own function calls away by the time this runs
    // (see that option's own doc comment) -- the map is simply unreferenced
    // by any node in that case, not wrong.
    const onnx::shape_inference::ModelLocalFunctionsMap model_local_functions =
        BuildModelLocalFunctionsMap(model);
    GraphFnChanged OptAndShapeOnGraphChanged =
        [optimize, shape_inference, fixed_point_iters,
         model_local_functions](onnx::Graph& graph) {
          onnxsim::ProfiledScope scope("OptAndShape");
          return OptAndShapeOnGraph(graph, optimize, shape_inference,
                                    fixed_point_iters, model_local_functions);
        };
    // `fold_ir_version` is declared above, alongside `converged`: see its own
    // comment for why it cannot live in this block despite only being read
    // here and set (from `model.ir_version()`) in the `Pipeline` lambda
    // below.
    GraphFnChanged FoldConstantOnGraphChanged =
        constant_folding ? GraphFnChanged([&executor, &fold_ir_version,
                                           model_local_functions](
                                              onnx::Graph& graph) {
          onnxsim::ProfiledScope scope("FoldConstant");
          const bool a = _EvalPartialShapeOnGraph(graph, model_local_functions);
          const bool b = _FoldConstantOnGraph(executor, graph, fold_ir_version);
          if (onnxsim::Profiler::Instance().enabled()) {
            onnxsim::Profiler::Instance().RecordNodeCount(
                "FoldConstant", CountGraphNodes(graph));
          }
          return a || b;
        })
                         : GraphFnChanged([](onnx::Graph&) { return false; });
    GraphFn PipelineOnGraph =
        FixedPointFn(OptAndShapeOnGraphChanged, FoldConstantOnGraphChanged,
                     fixed_point_iters, &converged);
    Pipeline =
        Profiled("Pipeline", [PipelineOnGraph, &fold_ir_version,
                              target_opset_version](onnx::ModelProto& model) {
          std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
          if (g.get() == nullptr) {
            // Same fallback as Optimizer::optimize(): if we can't parse the
            // model, leave it untouched.
            return;
          }
          // Convert the default ONNX domain's opset first, directly on the
          // resident graph via ConvertVersionOnGraph -- so the simplification
          // below can clean up any redundant nodes the version converter
          // introduces, same rationale as the old ModelProto-level
          // ConvertOpsetVersion call this replaces -- but without paying a
          // second Import/Export pair for it (unlike ConvertOpsetVersion, which
          // always does its own; still used by the `rewriter` branch above,
          // which has no resident Graph to share this one with).
          if (target_opset_version) {
            onnxsim::ProfiledScope opset_scope("ConvertOpsetVersion");
            onnx::version_conversion::ConvertVersionOnGraph(
                g, *target_opset_version);
          }
          fold_ir_version = model.ir_version();
          PipelineOnGraph(*g);
          onnx::ModelProto out = onnx::PrepareOutput(model);
          onnx::ExportModelProto(&out, g, /*consume_tensor_data=*/true);
          // OptimizeGraphFixed never sees model-local functions (they live on
          // ModelProto, not Graph), so carry them over unchanged -- mirroring
          // Optimizer::optimize()'s own AddFunctionsToModel.
          for (const auto& function_proto : model.functions()) {
            *out.add_functions() = function_proto;
          }
          model = std::move(out);
        });
  }
  // The fixed points mutate in place, so make one working copy of the (const)
  // input model and simplify it in place. When the caller has told us
  // ``model``'s tensor data can be consumed (``mutable_model`` non-null), do
  // the same move-based Import/Export round trip as OptAndShape's own
  // resident Graph above, instead of a deep copy -- this is what actually
  // avoids the extra ~1x-model-size peak documented in
  // bench/RESULTS_synthetic_decoder_oom.md, since it was traced to exactly
  // this copy, not (as first suspected) anything on the Python side.
  onnx::ModelProto sim_model;
  if (mutable_model != nullptr) {
    std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(*mutable_model));
    if (g.get() == nullptr) {
      // Same fallback as the plain path: if we can't parse it, just copy.
      sim_model = model;
    } else {
      sim_model = onnx::PrepareOutput(model);
      onnx::ExportModelProto(&sim_model, g, /*consume_tensor_data=*/true);
      for (const auto& function_proto : model.functions()) {
        *sim_model.add_functions() = function_proto;
      }
    }
  } else {
    sim_model = model;
  }
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
  //
  // Before inlining, tally how many top-level-graph nodes call each function
  // (matched by (domain, op_type) against the function's own (domain, name)) so
  // RecordSimplifyDiffMetadata's block below can record which functions
  // disappeared and how many call sites each had -- InlineLocalFunctions clears
  // model.functions() entirely, so this is the last point that information is
  // available. Calls inside subgraphs or other functions are not counted, only
  // the top-level graph (matching DiffGraphs' own top-level-only scope).
  std::vector<std::string> inlined_function_labels;
  if (include_inline_functions) {
    std::unordered_map<std::string, size_t> call_counts;
    for (const auto& fn : sim_model.functions()) {
      call_counts.emplace(fn.domain() + "::" + fn.name(), 0);
    }
    if (!call_counts.empty()) {
      for (const auto& node : sim_model.graph().node()) {
        auto it = call_counts.find(node.domain() + "::" + node.op_type());
        if (it != call_counts.end()) {
          ++it->second;
        }
      }
    }
    onnx::inliner::InlineLocalFunctions(sim_model);
    inlined_function_labels.reserve(call_counts.size());
    for (const auto& [label, count] : call_counts) {
      inlined_function_labels.push_back(label + ":" + std::to_string(count));
    }
    std::sort(inlined_function_labels.begin(), inlined_function_labels.end());
  }
  // Optionally convert the model to a different opset version (of the default
  // ONNX domain) first, so the simplification below can clean up any redundant
  // nodes the version converter introduces. Wrapped in its own profiled span --
  // sibling to, not nested inside, the "Simplify" root below -- so its cost is
  // visible in the flame graph / profile_pass_phases output instead of showing
  // up as unaccounted time before the first pass.
  //
  // Only needed here for the `rewriter` branch, which is ModelProto-based
  // throughout (onnxscript's rewriter only understands ModelProto) and so has
  // no resident Graph to fold this into. The no-rewriter branch does the
  // equivalent conversion itself, directly on its own resident Graph, inside
  // Pipeline above -- see its ConvertVersionOnGraph call.
  if (target_opset_version && rewriter) {
    onnxsim::ProfiledScope opset_scope("ConvertOpsetVersion");
    // std::move: sim_model is about to be overwritten by the result anyway,
    // so let ConvertOpsetVersion's by-value parameter move-construct instead
    // of copying -- required for it to reach ConvertVersion's non-copying
    // overload (see model_prep.cpp).
    sim_model =
        ConvertOpsetVersion(std::move(sim_model), *target_opset_version);
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
  const std::vector<std::string> assigned_node_names =
      AssignMissingNodeNames(sim_model);
  DropIncompleteValueInfo(sim_model);
  RecordSimplifyOptionsMetadata(
      sim_model, skip_optimizers, constant_folding, shape_inference,
      tensor_size_threshold, target_opset_version, initializers_as_constants,
      include_inline_functions, mutable_initializer, overwrite_input_shapes,
      unused_output, extra_optimizers);
  RecordSimplifyDiffMetadata(sim_model, model);
  // Nodes onnxsim itself had to name (no author-given name survived), and
  // functions inlined away (see the tally taken just before
  // InlineLocalFunctions, above) -- two more places a name or a structural
  // grouping present in the input model is gone from the output, alongside
  // the fusion/folding changes RecordSimplifyDiffMetadata already covers.
  RecordCappedListMetadata(sim_model, "onnxsim.assigned_node_names",
                           assigned_node_names, 20);
  RecordCappedListMetadata(sim_model, "onnxsim.inlined_functions",
                           inlined_function_labels, 20);
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

onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output,
    const std::optional<std::vector<std::string>>& extra_optimizers) {
  return SimplifyImpl(executor, model, /*mutable_model=*/nullptr,
                      skip_optimizers, constant_folding, shape_inference,
                      tensor_size_threshold, target_opset_version, rewriter,
                      initializers_as_constants, include_inline_functions,
                      mutable_initializer, overwrite_input_shapes,
                      unused_output, extra_optimizers);
}

onnx::ModelProto SimplifyConsumeInput(
    const ModelExecutor& executor, onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version, const GraphRewriter* rewriter,
    bool initializers_as_constants, bool include_inline_functions,
    bool mutable_initializer,
    const std::optional<std::unordered_map<std::string, std::vector<int64_t>>>&
        overwrite_input_shapes,
    const std::optional<std::vector<std::string>>& unused_output,
    const std::optional<std::vector<std::string>>& extra_optimizers) {
  return SimplifyImpl(executor, model, /*mutable_model=*/&model,
                      skip_optimizers, constant_folding, shape_inference,
                      tensor_size_threshold, target_opset_version, rewriter,
                      initializers_as_constants, include_inline_functions,
                      mutable_initializer, overwrite_input_shapes,
                      unused_output, extra_optimizers);
}

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
    const std::optional<std::vector<std::string>>& unused_output,
    const std::optional<std::vector<std::string>>& extra_optimizers) {
  const bool debug_timing = std::getenv("ONNXSIM_DEBUG_PATH_TIMING") != nullptr;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto elapsed_ms = [](auto t0, auto t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  onnx::ModelProto model;
  {
    const auto t0 = now();
    onnx::optimization::loadModel(&model, in_path, true);
    if (debug_timing) {
      std::cerr << "SimplifyPath: loadModel " << elapsed_ms(t0, now())
                << "ms\n";
    }
  }

  {
    const auto t0 = now();
    // ``model`` is this function's own local, about to be overwritten by the
    // result and never read again beforehand -- exactly the case
    // SimplifyConsumeInput's doc comment calls out as safe, and the one this
    // investigation was written for (see
    // bench/RESULTS_synthetic_decoder_oom.md): it avoids the ~1x-model-size
    // deep copy ``Simplify()`` would otherwise make of ``model`` to get a
    // mutable working copy, cutting the peak RSS of simplifying a large
    // external-data model roughly in half.
    model = SimplifyConsumeInput(
        executor, model, skip_optimizers, constant_folding, shape_inference,
        tensor_size_threshold, target_opset_version, rewriter,
        initializers_as_constants, include_inline_functions,
        mutable_initializer, overwrite_input_shapes, unused_output,
        extra_optimizers);
    if (debug_timing) {
      std::cerr << "SimplifyPath: Simplify " << elapsed_ms(t0, now()) << "ms\n";
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
