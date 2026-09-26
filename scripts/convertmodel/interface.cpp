#include "bias_correction_entry.h"
#include "lora_entry.h"
#include "model_info.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "onnxoptimizer/optimize.h"
#include "onnxsim.h"
#include "precision_estimator.h"
#include "qat_entry.h"
#include "tensor_pool.h"
#include "tensor_pool_bridge.h"
#include "tensor_pool_gguf_bridge.h"

// Version strings baked in by CMake (read from VERSION and
// third_party/onnx-optimizer/VERSION_NUMBER). Fall back to "unknown" so the
// file still compiles outside the CMake build (e.g. an IDE/static check).
#ifndef ONNXSIM_VERSION_STRING
#define ONNXSIM_VERSION_STRING "unknown"
#endif
#ifndef ONNX_OPTIMIZER_VERSION_STRING
#define ONNX_OPTIMIZER_VERSION_STRING "unknown"
#endif

// In a hookable WASM build (ONNXSIM_WASM_HOOKABLE_EXECUTOR), constant folding
// can be delegated to a page-provided executor via this bridge. The ORT-web
// variant is compiled without ONNXSIM_HAS_ORT; the default WASM build keeps
// its built-in ORT fallback.
#ifdef ONNXSIM_WASM_HOOKABLE_EXECUTOR
#include "js_model_executor.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace em = emscripten;

namespace {

// Where the C++ profiler writes its Chrome trace inside Emscripten's in-memory
// filesystem when the page asks for a profile. Read back into a string and
// removed by ReadAndClearProfileTrace() so nothing accumulates across runs.
constexpr const char *kProfileTracePath = "onnxsim_profile.json";

// Turn onnxsim's simplification profiler on for the next Simplify() call. The
// profiler is driven by environment variables (so every binding gets it for
// free); setenv() feeds the same libc environ that onnxsim.cpp reads via
// std::getenv(). ONNXSIM_MERGE_ORT_PROFILE additionally folds ONNX Runtime's
// per-session constant-folding profiles into the same trace.
void EnableProfiling() {
  setenv("ONNXSIM_PROFILE", kProfileTracePath, 1);
  setenv("ONNXSIM_MERGE_ORT_PROFILE", "1", 1);
}

void DisableProfiling() {
  unsetenv("ONNXSIM_PROFILE");
  unsetenv("ONNXSIM_MERGE_ORT_PROFILE");
}

// Read the trace the profiler wrote (empty string if it is missing) and delete
// the file so the MEMFS does not grow run over run.
std::string ReadAndClearProfileTrace() {
  std::string trace;
  std::ifstream ifs(kProfileTracePath, std::ios::binary);
  if (ifs) {
    std::ostringstream ss;
    ss << ifs.rdbuf();
    trace = ss.str();
  }
  std::remove(kProfileTracePath);
  return trace;
}

// Where the standalone safetensors/gguf export/import functions below stage
// the archive inside Emscripten's in-memory filesystem, mirroring
// kProfileTracePath. On export, the TensorPool bridge bakes this path into
// the archive as the embedded model's external_data "location" (see
// SaveModelAsSafetensorsStandalone / SaveModelAsGGUFStandalone), so it is a
// fixed internal name rather than the user-facing download filename --
// onnxsim's own loaders resolve the embedded model by pool name, not by this
// path, so the mismatch with whatever name the browser saves the download
// under (or the name of a file the user uploads for import) is harmless.
constexpr const char *kSafetensorsArchivePath = "onnxsim_export.safetensors";
constexpr const char *kGGUFArchivePath = "onnxsim_export.gguf";

// Read `path` back into a string and delete it, same pattern as
// ReadAndClearProfileTrace above. Returns an empty string if the file could
// not be opened.
std::string ReadAndDeleteFile(const std::string &path) {
  std::string data;
  // Seek to measure the size and read directly into a pre-sized string,
  // rather than streaming through an ostringstream (which grows its own
  // internal buffer geometrically and copies out of it into the returned
  // string on .str()) -- one read into the final buffer instead of several
  // reallocate-and-copy passes. Matters here because a safetensors/gguf
  // archive embeds the whole model and can be tens of MB.
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (ifs) {
    const std::streamoff size = ifs.tellg();
    if (size > 0) {
      data.resize(static_cast<size_t>(size));
      ifs.seekg(0);
      ifs.read(&data[0], size);
    }
  }
  std::remove(path.c_str());
  return data;
}

// Stage `data` at `path` inside the MEMFS, for the import functions below to
// hand to a TensorPool loader that only knows how to read from a path.
// Returns false (and leaves nothing behind) on a write failure.
bool WriteFile(const std::string &path, const std::string &data) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs)
    return false;
  ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(ofs);
}

// Serialize `model` into the shared static buffer and hand JS a typed-memory
// view over it (null on failure). The buffer is `static` so the view stays
// valid after this function returns; the worker copies it out immediately
// (toBase64 / a fresh Uint8Array), and calls are sequential, so a single shared
// buffer is safe -- matching how the other bindings here return model bytes.
em::val SerializeModel(const onnx::ModelProto &model) {
  static std::string result;
  if (!model.SerializeToString(&result)) {
    std::cerr << "Serialize failed" << std::endl;
    return em::val::null();
  }
  return em::val(em::typed_memory_view(
      result.size(), reinterpret_cast<uint8_t *>(result.data())));
}

// Materialize the raw little-endian bytes of `tensor` into `out`, reading
// either its `raw_data` or the typed repeated fields
// (float_data/int64_data/...). The wasm target is little-endian, so writing
// native bytes yields the little-endian layout onnxruntime-web's typed arrays
// expect. Returns false for a dtype we do not pack (STRING / COMPLEX / the
// narrow float variants stored out of band); the caller reports it. The bridged
// dtypes match CppModelExecutor / the ONNX backend node tests.
bool TensorProtoToRawBytes(const onnx::TensorProto &tensor, std::string &out) {
  using TP = onnx::TensorProto;
  out.clear();
  if (tensor.has_raw_data()) {
    out = tensor.raw_data();
    return true;
  }
  auto append = [&out](const void *p, size_t n) {
    out.append(reinterpret_cast<const char *>(p), n);
  };
  switch (tensor.data_type()) {
  case TP::FLOAT:
    for (float v : tensor.float_data())
      append(&v, sizeof(v));
    return true;
  case TP::DOUBLE:
    for (double v : tensor.double_data())
      append(&v, sizeof(v));
    return true;
  case TP::INT64:
    for (int64_t v : tensor.int64_data())
      append(&v, sizeof(v));
    return true;
  case TP::UINT64:
    for (uint64_t v : tensor.uint64_data())
      append(&v, sizeof(v));
    return true;
  case TP::INT32:
    for (int32_t v : tensor.int32_data())
      append(&v, sizeof(v));
    return true;
  case TP::UINT32:
    // uint32 values live in uint64_data per the TensorProto encoding.
    for (uint64_t v : tensor.uint64_data()) {
      uint32_t w = static_cast<uint32_t>(v);
      append(&w, sizeof(w));
    }
    return true;
  case TP::INT16:
  case TP::UINT16:
  case TP::FLOAT16:
  case TP::BFLOAT16:
    // Stored widened in int32_data; pack the low 16 bits.
    for (int32_t v : tensor.int32_data()) {
      uint16_t w = static_cast<uint16_t>(v);
      append(&w, sizeof(w));
    }
    return true;
  case TP::INT8:
  case TP::UINT8:
  case TP::BOOL:
    for (int32_t v : tensor.int32_data()) {
      uint8_t b = static_cast<uint8_t>(v);
      append(&b, sizeof(b));
    }
    return true;
  default:
    return false;
  }
}

// Highest opset version this build supports for the default ONNX domain
// (ai.onnx), read from the linked onnx at runtime. Returns 1 if the registry
// somehow lacks the default domain. Used to give parsed graphs a sensible
// default opset import.
int HighestDefaultOpset() {
  const auto &ranges =
      onnx::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  auto it = ranges.find(""); // "" is the default ONNX domain (ai.onnx)
  int max_opset = (it != ranges.end()) ? it->second.second : 0;
  return max_opset > 0 ? max_opset : 1;
}

// Make sure `model` carries an opset import for the default ONNX domain and for
// every domain its local functions live in, so a model assembled from parsed
// text (which may omit the prolog, or reference local functions) still
// validates and can be simplified / visualized. Existing opset imports are left
// untouched; only missing domains are added. A local function's own domain is
// imported at the version the function declares for that domain, defaulting to
// 1 for a custom domain that declares none.
void EnsureOpsetImports(onnx::ModelProto &model) {
  std::set<std::string> present;
  for (const auto &op : model.opset_import()) {
    present.insert(op.domain());
  }
  if (!present.count("")) {
    auto *op = model.add_opset_import();
    op->set_domain(""); // default ONNX domain (ai.onnx)
    op->set_version(HighestDefaultOpset());
    present.insert("");
  }
  for (const auto &fn : model.functions()) {
    const std::string &dom = fn.domain();
    if (present.count(dom))
      continue;
    int64_t ver = 1;
    for (const auto &op : fn.opset_import()) {
      if (op.domain() == dom) {
        ver = op.version();
        break;
      }
    }
    auto *op = model.add_opset_import();
    op->set_domain(dom);
    op->set_version(ver);
    present.insert(dom);
  }
}

} // namespace

// Returns an object { model: Uint8Array, trace: string }. `trace` is the
// profiler's Chrome trace JSON when `profile` is true, otherwise "". Returning
// an object (rather than the bare model view) lets the worker hand the trace to
// the in-page flame-graph viewer without a second round-trip.
em::val onnxsimplify_export(const std::string &data, em::val skip_optimizers,
                            bool constant_folding, bool shape_inference,
                            size_t tensor_size_threshold,
                            int target_opset_version, bool profile,
                            bool annotate, bool graph_diff) {
  InitEnv();

  std::cerr << "LOG_THRESHOLD: " << std::getenv("LOG_THRESHOLD") << std::endl;
  onnx::ModelProto xmodel;
  std::cerr << "parsing message" << std::endl;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }

  if (profile) {
    EnableProfiling();
  }

  std::cerr << "simplify begin" << std::endl;
  onnx::ModelProto optimized;
  try {
    optimized = Simplify(
#ifdef ONNXSIM_WASM_HOOKABLE_EXECUTOR
        // Runs each fold group through the page's onnxruntime-web. Its Run
        // blocks on a JS Promise, so this whole function is Asyncified and
        // returns a Promise to JS (the worker awaits it).
        *GetJsModelExecutor(),
#else
        *GetBuiltinModelExecutor(),
#endif
        xmodel, em::vecFromJSArray<std::string>(skip_optimizers),
        constant_folding, shape_inference, tensor_size_threshold,
        // A target opset version of <= 0 means "leave the opset unchanged".
        target_opset_version > 0 ? std::make_optional(target_opset_version)
                                 : std::nullopt);
  } catch (const std::exception &e) {
    std::cerr << "simplify error: " << e.what() << std::endl;
    if (profile) {
      DisableProfiling();
      ReadAndClearProfileTrace();
    }
    return em::val::null();
  }
  std::cerr << "simplify end" << std::endl;

  // Print the before/after diff (op counts + model size) to stdout so the page
  // surfaces it in the log, mirroring the Python CLI's "here is the difference"
  // output. std::cout is routed to the worker's `print` handler.
  std::cout << "Finish! Here is the difference:\n"
            << FormatSimplifyingInfo(xmodel, optimized) << std::flush;

  // Optional, more detailed node/value-level diff -- which nodes/values were
  // removed, added, or changed (matched by output tensor name). Off by
  // default (it can be long for a big model), controlled by the page's
  // "graph diff" toggle.
  if (graph_diff) {
    std::cout << FormatGraphDiff(xmodel, optimized) << std::flush;
  }

  // Collect the trace right after Simplify() (the profiler has flushed it by
  // now) and turn profiling back off, so a later check/serialize failure
  // cannot leave the environment armed for the next request.
  std::string trace;
  if (profile) {
    DisableProfiling();
    trace = ReadAndClearProfileTrace();
  }

  // Bake onnxsim's MAC/FLOP model-info metrics into the model's
  // metadata_props (mirrors Python model_info.annotate_metadata) so the page's
  // "Run inference" panel can read and display them. On by default.
  if (annotate) {
    try {
      AnnotateModelInfo(optimized);
    } catch (const std::exception &e) {
      std::cerr << "annotate model info failed: " << e.what() << std::endl;
    }
  }

  try {
    std::cerr << "checking model" << std::endl;
    onnx::checker::check_model(optimized);
  } catch (const onnx::checker::ValidationError &e) {
    std::cerr << "model check failed: " << e.what() << std::endl;
    return em::val::null();
  }

  std::cerr << "serializing model" << std::endl;
  static std::string result;
  if (!optimized.SerializeToString(&result)) {
    std::cerr << "Serialize failed" << std::endl;
    return em::val::null();
  }
  std::cerr << "model simplify ended" << std::endl;
  em::val out = em::val::object();
  out.set("model",
          em::val(em::typed_memory_view(
              result.size(), reinterpret_cast<uint8_t *>(result.data()))));
  out.set("trace", em::val(trace));
  return out;
}

em::val onnxoptimizer_optimize(const std::string &data, em::val passes_ary,
                               bool annotate) {
  std::vector<std::string> passes = em::vecFromJSArray<std::string>(passes_ary);
  onnx::ModelProto xmodel;
  std::cerr << "parsing message" << std::endl;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto optimized;
  try {
    optimized = onnx::optimization::Optimize(xmodel, passes);
  } catch (const std::exception &e) {
    std::cerr << "optimize error: " << e.what() << std::endl;
    return em::val::null();
  }
  if (annotate) {
    try {
      AnnotateModelInfo(optimized);
    } catch (const std::exception &e) {
      std::cerr << "annotate model info failed: " << e.what() << std::endl;
    }
  }
  std::cerr << "serializing model" << std::endl;
  static std::string result;
  if (!optimized.SerializeToString(&result)) {
    std::cerr << "Serialize failed" << std::endl;
    return em::val::null();
  }
  return em::val(em::typed_memory_view(
      result.size(), reinterpret_cast<uint8_t *>(result.data())));
}

em::val onnxoptimizer_optimize_fixed(const std::string &data,
                                     em::val passes_ary, bool annotate) {
  std::vector<std::string> passes = em::vecFromJSArray<std::string>(passes_ary);
  onnx::ModelProto xmodel;
  std::cerr << "parsing message" << std::endl;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto optimized;
  try {
    optimized = onnx::optimization::OptimizeFixed(xmodel, passes);
  } catch (const std::exception &e) {
    std::cerr << "optimize error: " << e.what() << std::endl;
    return em::val::null();
  }
  if (annotate) {
    try {
      AnnotateModelInfo(optimized);
    } catch (const std::exception &e) {
      std::cerr << "annotate model info failed: " << e.what() << std::endl;
    }
  }
  std::cerr << "serializing model" << std::endl;
  static std::string result;
  if (!optimized.SerializeToString(&result)) {
    std::cerr << "Serialize failed" << std::endl;
    return em::val::null();
  }
  return em::val(em::typed_memory_view(
      result.size(), reinterpret_cast<uint8_t *>(result.data())));
}

// Annotate a model's MAC/FLOP model-info metrics into its metadata_props
// without simplifying/optimizing it, and return the annotated bytes. The page
// uses this to give the *original* uploaded model the same onnxsim.* metrics
// the converted model gets, so the "Run inference" panel can report throughput
// (GFLOP/s) for both and compare their inference speed. Execution is unaffected
// — only metadata_props are added — so the annotated bytes run identically.
// Returns null on a parse/serialize failure (annotation itself is best-effort:
// a model whose shapes cannot be inferred simply gains no metrics).
em::val onnxsim_annotate_model_info(const std::string &data) {
  onnx::ModelProto xmodel;
  std::cerr << "parsing message" << std::endl;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    AnnotateModelInfo(xmodel);
  } catch (const std::exception &e) {
    std::cerr << "annotate model info failed: " << e.what() << std::endl;
  }
  std::cerr << "serializing model" << std::endl;
  static std::string result;
  if (!xmodel.SerializeToString(&result)) {
    std::cerr << "Serialize failed" << std::endl;
    return em::val::null();
  }
  return em::val(em::typed_memory_view(
      result.size(), reinterpret_cast<uint8_t *>(result.data())));
}

// Export `data` (a serialized onnx::ModelProto) as a single standalone
// safetensors archive: every initializer's bytes move into the archive with
// real, byte-accurate offsets -- openable by the `safetensors` Python package
// / HF tooling with no onnxsim involved -- and the graph itself is embedded
// alongside them (tensor_pool_bridge.h's SaveModelAsSafetensorsStandalone), so
// the one file this returns is both the model's weights and its graph. Used
// by the page's download-format selector for the ".onnx.safetensors" option.
// Returns null on a parse/export failure.
em::val onnxsim_export_safetensors(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  onnxsim::tensor_pool::TensorPool pool;
  try {
    onnxsim::tensor_pool::SaveModelAsSafetensorsStandalone(
        xmodel, kSafetensorsArchivePath, pool);
  } catch (const std::exception &e) {
    std::cerr << "safetensors export failed: " << e.what() << std::endl;
    std::remove(kSafetensorsArchivePath);
    return em::val::null();
  }
  static std::string result;
  result = ReadAndDeleteFile(kSafetensorsArchivePath);
  if (result.empty()) {
    std::cerr << "failed to read back the exported safetensors file"
              << std::endl;
    return em::val::null();
  }
  return em::val(em::typed_memory_view(
      result.size(), reinterpret_cast<uint8_t *>(result.data())));
}

// GGUF counterpart of onnxsim_export_safetensors above; see
// tensor_pool_gguf_bridge.h's SaveModelAsGGUFStandalone. Used by the page's
// download-format selector for the ".onnx.gguf" option.
em::val onnxsim_export_gguf(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  onnxsim::tensor_pool::TensorPool pool;
  try {
    onnxsim::tensor_pool::SaveModelAsGGUFStandalone(xmodel, kGGUFArchivePath,
                                                    pool);
  } catch (const std::exception &e) {
    std::cerr << "gguf export failed: " << e.what() << std::endl;
    std::remove(kGGUFArchivePath);
    return em::val::null();
  }
  static std::string result;
  result = ReadAndDeleteFile(kGGUFArchivePath);
  if (result.empty()) {
    std::cerr << "failed to read back the exported gguf file" << std::endl;
    return em::val::null();
  }
  return em::val(em::typed_memory_view(
      result.size(), reinterpret_cast<uint8_t *>(result.data())));
}

// Import counterpart of onnxsim_export_safetensors: `data` is the whole raw
// bytes of a standalone safetensors archive (as produced by that function,
// or any other tool following the same self-describing-archive convention --
// tensor_pool_bridge.h's EmbedModel/ExtractModel -- an embedded "model.onnx"
// entry alongside the tensors). Stages it into the MEMFS, loads it back into
// an ordinary, fully in-memory onnx::ModelProto via LoadModelFromSafetensors
// (hydrate_all=true, so every initializer is hydrated -- no lingering
// EXTERNAL references into the now-deleted staged file), and returns its
// serialized bytes. Used by the page's file picker to accept a
// ".onnx.safetensors" upload. Returns null if the archive has no embedded
// model (e.g. a plain weights-only safetensors file with no onnxsim-authored
// graph) or on any other stage/load/parse failure.
em::val onnxsim_import_safetensors(const std::string &data) {
  if (!WriteFile(kSafetensorsArchivePath, data)) {
    std::cerr << "failed to stage the uploaded safetensors file" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto xmodel;
  onnxsim::tensor_pool::TensorPool pool;
  bool ok = false;
  try {
    ok = onnxsim::tensor_pool::LoadModelFromSafetensors(kSafetensorsArchivePath,
                                                        &xmodel, pool);
  } catch (const std::exception &e) {
    std::cerr << "safetensors import failed: " << e.what() << std::endl;
    std::remove(kSafetensorsArchivePath);
    return em::val::null();
  }
  std::remove(kSafetensorsArchivePath);
  if (!ok) {
    std::cerr << "safetensors file has no embedded onnxsim model (a plain "
                 "weights-only archive is not importable as a graph)"
              << std::endl;
    return em::val::null();
  }
  return SerializeModel(xmodel);
}

// GGUF counterpart of onnxsim_import_safetensors above; see
// tensor_pool_gguf_bridge.h's LoadModelFromGGUF. Used by the page's file
// picker to accept a ".onnx.gguf" upload. `skipped` (any other pooled
// tensors LoadModelFromGGUF could not hydrate, e.g. a quantized weight) is
// logged rather than surfaced structurally -- best-effort, matching how the
// rest of this file reports non-fatal issues via std::cerr.
em::val onnxsim_import_gguf(const std::string &data) {
  if (!WriteFile(kGGUFArchivePath, data)) {
    std::cerr << "failed to stage the uploaded gguf file" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto xmodel;
  onnxsim::tensor_pool::TensorPool pool;
  bool ok = false;
  std::vector<std::string> skipped;
  try {
    ok = onnxsim::tensor_pool::LoadModelFromGGUF(kGGUFArchivePath, &xmodel,
                                                 pool, true, &skipped);
  } catch (const std::exception &e) {
    std::cerr << "gguf import failed: " << e.what() << std::endl;
    std::remove(kGGUFArchivePath);
    return em::val::null();
  }
  std::remove(kGGUFArchivePath);
  if (!ok) {
    std::cerr << "gguf file has no embedded onnxsim model (a plain "
                 "weights-only archive is not importable as a graph)"
              << std::endl;
    return em::val::null();
  }
  if (!skipped.empty()) {
    std::cerr << "gguf import: " << skipped.size()
              << " tensor(s) left un-hydrated (quantized/unsupported dtype)"
              << std::endl;
  }
  return SerializeModel(xmodel);
}

// Report the versions of the libraries baked into this module, for detailed
// bug reports. onnxsim and onnx-optimizer come from CMake (their VERSION
// files); onnx is reported as its max IR version + the highest opset it
// supports for the default (ai.onnx) domain, both read from the linked onnx at
// runtime; protobuf is decoded from GOOGLE_PROTOBUF_VERSION. Returns a plain JS
// object of strings.
em::val onnxsim_versions() {
  em::val out = em::val::object();
  out.set("onnxsim", std::string(ONNXSIM_VERSION_STRING));
  out.set("onnx_optimizer", std::string(ONNX_OPTIMIZER_VERSION_STRING));

  // onnx: IR version + highest supported opset for the default ONNX domain.
  {
    int max_opset = 0;
    const auto &ranges =
        onnx::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
    auto it = ranges.find(""); // "" is the default ONNX domain (ai.onnx)
    if (it != ranges.end()) {
      max_opset = it->second.second;
    }
    std::ostringstream os;
    os << "IR v" << static_cast<int>(onnx::IR_VERSION) << ", opset "
       << max_opset;
    out.set("onnx", os.str());
  }

  // protobuf: GOOGLE_PROTOBUF_VERSION is major*1000000 + minor*1000 + patch.
#ifdef GOOGLE_PROTOBUF_VERSION
  {
    int pv = GOOGLE_PROTOBUF_VERSION;
    std::ostringstream os;
    os << (pv / 1000000) << "." << (pv / 1000 % 1000) << "." << (pv % 1000);
    out.set("protobuf", os.str());
  }
#else
  out.set("protobuf", std::string("unknown"));
#endif
  return out;
}

// Parse an ONNX *text* representation into a model and return its bytes.
// Accepts either a whole-model text (with an `<ir_version: ..., opset_import:
// [...]>` prolog) or a bare graph body -- the textual form
// onnx.parser.parse_graph / parse_model accept. Local function definitions (one
// or more `<domain: ...> name (...) => (...) {...}` blocks after the graph) are
// parsed too: the model path captures them natively, and either path is
// normalized so the model carries an opset import for the default ONNX domain
// and for every domain its functions use, plus a valid ir_version -- so a graph
// that calls local functions still validates and can be simplified / visualized
// / run like an uploaded model. Returns { model: Uint8Array } on success or {
// error: string } describing the parse failure(s).
em::val onnxsim_parse_graph(const std::string &text) {
  em::val out = em::val::object();
  // Try whole-model text first: this is the only path that captures trailing
  // FunctionProto definitions (GraphProto has no functions field), and it also
  // accepts a bare graph because the `<...>` prolog is optional. Each
  // OnnxParser consumes its input, so a fresh parser is constructed per attempt
  // (the member Parse API is stable across onnx versions).
  onnx::ModelProto model;
  onnx::OnnxParser model_parser(text.c_str());
  onnx::Common::Status model_st = model_parser.Parse(model);
  if (!model_st.IsOK() || !model.has_graph()) {
    // Fall back to graph-only text and wrap it into a model. This path has no
    // functions (the graph syntax cannot carry them), so a model that needs
    // local functions must go through the model path above.
    onnx::GraphProto graph;
    onnx::OnnxParser graph_parser(text.c_str());
    onnx::Common::Status graph_st = graph_parser.Parse(graph);
    if (!graph_st.IsOK()) {
      out.set("error", std::string("failed to parse as a model (") +
                           model_st.ErrorMessage() + ") and as a graph (" +
                           graph_st.ErrorMessage() + ")");
      return out;
    }
    model.Clear();
    *model.mutable_graph() = graph;
  }
  // A bare graph (or a graph with local functions but no explicit prolog)
  // parses without an ir_version or opset imports; fill in sensible defaults so
  // the result validates. Local functions require IR v8+, and
  // EnsureOpsetImports adds an opset import for each function's domain.
  if (model.ir_version() == 0) {
    model.set_ir_version(onnx::IR_VERSION);
  }
  EnsureOpsetImports(model);
  em::val bytes = SerializeModel(model);
  if (bytes.isNull()) {
    out.set("error", std::string("failed to serialize the parsed model"));
    return out;
  }
  out.set("model", bytes);
  return out;
}

// Extract a serialized ModelProto's ONNX *textual* representation -- the
// inverse of onnxsim_parse_graph above: the same syntax onnx.parser.parse_model
// accepts (and onnx.printer.to_text produces on the Python side), covering the
// model's ir_version/opset imports, its graph, and any local function
// definitions. Lets the "loaded" model shown in the converter's Netron pane be
// read as text -- e.g. to inspect it, or to round-trip it through the "Parse a
// text graph" panel after editing. Returns { text: string } on success or {
// error: string } if the bytes don't parse as a model.
em::val onnxsim_extract_graph(const std::string &data) {
  em::val out = em::val::object();
  onnx::ModelProto model;
  if (!model.ParseFromArray(data.data(), data.size())) {
    out.set("error", std::string("failed to parse ModelProto"));
    return out;
  }
  try {
    out.set("text", onnx::ProtoToString(model));
  } catch (const std::exception &e) {
    out.set("error", std::string("failed to print the model as text: ") +
                          e.what());
  }
  return out;
}

// Inline the model's local (model-defined) functions into its main graph: every
// call-site of a FunctionProto listed in `model.functions` is replaced by the
// function body, and the functions are removed from the model. This flattens a
// model that uses local functions into a plain op graph, which onnx-optimizer /
// Simplify and constant folding can then see through. Schema-defined (built-in)
// functions are left alone. Returns the inlined model bytes (null on failure);
// `annotate` optionally bakes MAC/FLOP model info into metadata_props, matching
// the other convert entry points.
em::val onnxsim_inline_functions(const std::string &data, bool annotate) {
  onnx::ModelProto xmodel;
  std::cerr << "parsing message" << std::endl;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    onnx::inliner::InlineLocalFunctions(xmodel);
  } catch (const std::exception &e) {
    std::cerr << "inline functions error: " << e.what() << std::endl;
    return em::val::null();
  }
  if (annotate) {
    try {
      AnnotateModelInfo(xmodel);
    } catch (const std::exception &e) {
      std::cerr << "annotate model info failed: " << e.what() << std::endl;
    }
  }
  return SerializeModel(xmodel);
}

// Parse a serialized onnx.TensorProto (e.g. an input_N.pb / output_N.pb from an
// ONNX backend test case) into a form onnxruntime-web can consume in JS.
// Returns { dtype: int (TensorProto.DataType), name: string, dims: [int...],
// data: Uint8Array (raw little-endian) } or { error: string }.
em::val onnxsim_parse_tensor(const std::string &data) {
  em::val out = em::val::object();
  onnx::TensorProto tensor;
  if (!tensor.ParseFromArray(data.data(), data.size())) {
    out.set("error", std::string("failed to parse TensorProto"));
    return out;
  }
  static std::string raw;
  if (!TensorProtoToRawBytes(tensor, raw)) {
    std::ostringstream os;
    os << "unsupported tensor data type " << tensor.data_type()
       << " (STRING / COMPLEX and similar are not bridged)";
    out.set("error", os.str());
    return out;
  }
  out.set("dtype", static_cast<int>(tensor.data_type()));
  out.set("name", tensor.name());
  em::val dims = em::val::array();
  for (int i = 0; i < tensor.dims_size(); ++i) {
    dims.set(i, static_cast<double>(tensor.dims(i)));
  }
  out.set("dims", dims);
  out.set("data", em::val(em::typed_memory_view(
                      raw.size(), reinterpret_cast<uint8_t *>(raw.data()))));
  return out;
}

// Run a single simplification building block once (not in the fixed point) for
// debugging, returning the transformed model bytes (null on failure). Shape
// inference and data propagation need no model executor. Constant folding runs
// through the same executor Simplify uses -- in the ORT-web build that awaits
// onnxruntime-web, so (like onnxsimplify_export) onnxsim_fold_constant is
// Asyncified and returns a Promise the worker awaits.
em::val onnxsim_infer_shapes(const std::string &data) {
  InitEnv();
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(InferShapesOnce(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "shape inference error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_data_propagation(const std::string &data) {
  InitEnv();
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(PropagateDataOnce(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "data propagation error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_fold_constant(const std::string &data,
                              size_t tensor_size_threshold) {
  InitEnv();
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    onnx::ModelProto folded = FoldConstantOnce(
#ifdef ONNXSIM_WASM_HOOKABLE_EXECUTOR
        *GetJsModelExecutor(),
#else
        *GetBuiltinModelExecutor(),
#endif
        xmodel, tensor_size_threshold);
    return SerializeModel(folded);
  } catch (const std::exception &e) {
    std::cerr << "constant folding error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// True when this module was built to delegate constant folding to
// onnxruntime-web (ONNXSIM_WASM_ORT_WEB). The worker uses this to decide
// whether it must load onnxruntime-web and register a runner on the Module
// before simplifying, and whether onnxsimplify_export returns a Promise
// (Asyncify) it needs to await. In the default built-in-ORT build this is false
// and the worker's behaviour is unchanged.
bool onnxsim_needs_ort_web() {
#ifdef ONNXSIM_WASM_ORT_WEB
  return true;
#else
  return false;
#endif
}

std::vector<std::string> onnxoptimizer_passes() {
  return onnx::optimization::GetAvailablePasses();
}

std::vector<std::string> onnxoptimizer_fuse_elimination_passes() {
  return onnx::optimization::GetFuseAndEliminationPass();
}

// Converts parallel JS arrays -- ``names_ary`` (tensor names) and
// ``ranges_flat_ary`` (a flat [min0, max0, min1, max1, ...] array, twice the
// length of ``names_ary``) -- into the calibration-ranges map
// QuantizeStatic/QuantizeQOperator take. Flat parallel arrays (rather than a
// JS object/Map bound through embind) match the batched, positional
// convention already used at this C++/JS boundary elsewhere (see
// js_model_executor.cpp / docs/wasm_ort_web.md).
std::unordered_map<std::string, std::pair<float, float>>
ParseCalibrationRanges(em::val names_ary, em::val ranges_flat_ary) {
  std::vector<std::string> names = em::vecFromJSArray<std::string>(names_ary);
  std::vector<float> flat = em::vecFromJSArray<float>(ranges_flat_ary);
  std::unordered_map<std::string, std::pair<float, float>> ranges;
  ranges.reserve(names.size());
  for (size_t i = 0; i < names.size() && 2 * i + 1 < flat.size(); ++i) {
    ranges.emplace(names[i], std::make_pair(flat[2 * i], flat[2 * i + 1]));
  }
  return ranges;
}

em::val onnxsim_quantize_dynamic(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeDynamic(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_dynamic error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_ternary(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeTernary(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_ternary error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnly(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_int4(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnlyInt4(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_int4 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_fp16(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeFp16(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_fp16 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_bf16(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeBf16(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_bf16 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// `format` is "e4m3" or "e5m2" -- see QuantizeFp8 in onnxsim.h.
em::val onnxsim_quantize_fp8(const std::string &data,
                             const std::string &format) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeFp8(xmodel, format));
  } catch (const std::exception &e) {
    std::cerr << "quantize_fp8 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val
WeightPrecisionEstimateToVal(const onnxsim::WeightPrecisionEstimate &est) {
  em::val out = em::val::object();
  out.set("nodeName", est.node_name);
  out.set("opType", est.op_type);
  out.set("reductionDepth", static_cast<double>(est.reduction_depth));
  out.set("numChannels", static_cast<double>(est.num_channels));
  out.set("int32AccumulatorSafe", est.int32_accumulator_safe);
  out.set("float32CastExact", est.float32_cast_exact);
  out.set("maxOutlierRatio", est.max_outlier_ratio);
  out.set("outlierRisk", est.outlier_risk);
  out.set("activationProducerOp", est.activation_producer_op);
  out.set("hasActivationRange", est.has_activation_range);
  out.set("activationRangeLo", est.activation_range_lo);
  out.set("activationRangeHi", est.activation_range_hi);
  out.set("recommendation", est.recommendation);
  return out;
}

em::val AttentionPrecisionEstimateToVal(
    const onnxsim::AttentionPrecisionEstimate &est) {
  em::val out = em::val::object();
  out.set("nodeName", est.node_name);
  out.set("hasNumQueryHeads", est.has_num_query_heads);
  out.set("numQueryHeads", static_cast<double>(est.num_query_heads));
  out.set("hasNumKvHeads", est.has_num_kv_heads);
  out.set("numKvHeads", static_cast<double>(est.num_kv_heads));
  out.set("hasHeadDim", est.has_head_dim);
  out.set("headDim", static_cast<double>(est.head_dim));
  out.set("defaultScale", est.default_scale);
  out.set("actualScale", est.actual_scale);
  // -1 = unknown, 0 = false, 1 = true -- see AttentionPrecisionEstimate's
  // doc comment in precision_estimator.h.
  out.set("scaleMatchesDefault", est.scale_matches_default);
  out.set("recommendation", est.recommendation);
  return out;
}

// Static, calibration-free INT8-quantization risk pre-check -- see
// precision_estimator.h (a C++ port of onnxsim/precision_estimator.py's
// estimate_model_quantization_drop, kept in exact sync with it). Purely a
// read-only analysis of the model's own weights and shapes: no execution and
// no calibration data needed, so the page can run this the instant a model
// loads -- before the user has picked a quantize method or run anything
// through onnxruntime-web. Returns null on parse failure, else an object
// mirroring onnxsim.ModelQuantizationEstimate: { totalNodesAnalyzed,
// unsafeNodes, outlierRiskNodes, worstOutlierRatio, estimatedRelativeError,
// riskLevel, weightEstimates: [...], attentionEstimates: [...] } (NaN fields
// -- e.g. worstOutlierRatio when no node had a computable ratio, or
// estimatedRelativeError when riskLevel is "unsafe" -- come through as the JS
// value NaN, exactly like Python's math.nan does for the same fields).
em::val onnxsim_estimate_quantization_drop(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  onnxsim::ModelQuantizationEstimate est;
  try {
    est = onnxsim::EstimateModelQuantizationDrop(xmodel);
  } catch (const std::exception &e) {
    std::cerr << "estimate_quantization_drop error: " << e.what() << std::endl;
    return em::val::null();
  }

  em::val out = em::val::object();
  out.set("totalNodesAnalyzed", static_cast<double>(est.total_nodes_analyzed));
  em::val unsafe_nodes = em::val::array();
  for (size_t i = 0; i < est.unsafe_nodes.size(); ++i) {
    unsafe_nodes.set(i, est.unsafe_nodes[i]);
  }
  out.set("unsafeNodes", unsafe_nodes);
  em::val outlier_nodes = em::val::array();
  for (size_t i = 0; i < est.outlier_risk_nodes.size(); ++i) {
    outlier_nodes.set(i, est.outlier_risk_nodes[i]);
  }
  out.set("outlierRiskNodes", outlier_nodes);
  out.set("worstOutlierRatio", est.worst_outlier_ratio);
  out.set("estimatedRelativeError", est.estimated_relative_error);
  out.set("riskLevel", est.risk_level);

  em::val weight_estimates = em::val::array();
  for (size_t i = 0; i < est.weight_estimates.size(); ++i) {
    weight_estimates.set(i,
                         WeightPrecisionEstimateToVal(est.weight_estimates[i]));
  }
  out.set("weightEstimates", weight_estimates);

  em::val attention_estimates = em::val::array();
  for (size_t i = 0; i < est.attention_estimates.size(); ++i) {
    attention_estimates.set(
        i, AttentionPrecisionEstimateToVal(est.attention_estimates[i]));
  }
  out.set("attentionEstimates", attention_estimates);
  return out;
}

// Both list_* functions below are used by the page to discover which tensor
// names to calibrate (run the model over sample inputs and record each
// tensor's observed min/max) before calling quantize_static/quantize_qoperator.
std::vector<std::string>
onnxsim_list_quantizable_activations(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return {};
  }
  return ListQuantizableActivations(xmodel);
}

std::vector<std::string>
onnxsim_list_qoperator_quantizable_outputs(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return {};
  }
  return ListQOperatorQuantizableOutputs(xmodel);
}

// Appends a bare ValueInfoProto(name=name) graph output for every name in
// ``names_ary`` not already an output, mirroring calibration.py's calibrate()
// -- the page's browser-side calibration step needs onnxruntime-web to
// compute and return each candidate tensor's value without changing the
// graph otherwise, so it can observe the per-tensor (min, max) over the
// sample inputs it feeds through. A bare ValueInfoProto with no type/shape is
// a valid ONNX output (onnxruntime infers it from the run); this matches
// calibrate()'s own onnx.ValueInfoProto(name=name) exactly.
em::val onnxsim_add_graph_outputs(const std::string &data, em::val names_ary) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  std::vector<std::string> names = em::vecFromJSArray<std::string>(names_ary);
  onnx::GraphProto *graph = xmodel.mutable_graph();
  std::set<std::string> existing_outputs;
  for (const auto &vi : graph->output()) {
    existing_outputs.insert(vi.name());
  }
  for (const auto &name : names) {
    if (existing_outputs.insert(name).second) {
      graph->add_output()->set_name(name);
    }
  }
  return SerializeModel(xmodel);
}

em::val onnxsim_quantize_static(const std::string &data, em::val names_ary,
                                em::val ranges_flat_ary) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeStatic(
        xmodel, ParseCalibrationRanges(names_ary, ranges_flat_ary)));
  } catch (const std::exception &e) {
    std::cerr << "quantize_static error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_qoperator(const std::string &data, em::val names_ary,
                                   em::val ranges_flat_ary) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeQOperator(
        xmodel, ParseCalibrationRanges(names_ary, ranges_flat_ary)));
  } catch (const std::exception &e) {
    std::cerr << "quantize_qoperator error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// ---------------------------------------------------------------------------
// Bias correction (onnxsim/bias_correction_entry.h) -- correct_bias/
// correct_spatial_bias's graph-surgery half only, same division of labour as
// static quantization above: onnxsim_list_correctable_outputs tells the page
// which Conv/Gemm/MatMul/Resize outputs are even eligible (present, by name,
// in both the float and modified model), the page runs both models through
// onnxruntime-web on synthetic calibration data and measures each one's
// per-channel or per-position mean error itself (bias_correction_calibration.mjs),
// and onnxsim_apply_bias_corrections only splices the already-measured
// numbers in.
em::val onnxsim_list_correctable_outputs(const std::string &float_data,
                                          const std::string &modified_data) {
  onnx::ModelProto float_model, modified_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size()) ||
      !modified_model.ParseFromArray(modified_data.data(),
                                      modified_data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  em::val out = em::val::array();
  const auto candidates = ListCorrectableOutputs(float_model, modified_model);
  for (size_t i = 0; i < candidates.size(); ++i) {
    em::val entry = em::val::object();
    entry.set("name", candidates[i].output_name);
    entry.set("axis", static_cast<double>(candidates[i].axis));
    entry.set("spatial", candidates[i].spatial);
    out.set(i, entry);
  }
  return out;
}

// `corrections_ary` is a JS array of `{name, shape, data}` objects -- `shape`
// the broadcast shape already reshaped to the output's own rank (e.g.
// [1, C, 1, 1] per-channel, or [1, C, H, W] per-position), `data` its
// row-major values (a plain array or Float32Array, either works through
// vecFromJSArray) -- exactly what bias_correction_calibration.mjs measures.
em::val onnxsim_apply_bias_corrections(const std::string &data,
                                        em::val corrections_ary) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  const unsigned length = corrections_ary["length"].as<unsigned>();
  std::vector<BiasCorrectionEntry> corrections;
  corrections.reserve(length);
  for (unsigned i = 0; i < length; ++i) {
    em::val item = corrections_ary[i];
    BiasCorrectionEntry entry;
    entry.output_name = item["name"].as<std::string>();
    // Via doubles (see OptInt64ListFromVal's own comment for why int64_t
    // cannot be converted directly).
    for (double d : em::vecFromJSArray<double>(item["shape"])) {
      entry.shape.push_back(static_cast<int64_t>(d));
    }
    entry.data = em::vecFromJSArray<float>(item["data"]);
    corrections.push_back(std::move(entry));
  }
  return SerializeModel(ApplyBiasCorrections(xmodel, corrections));
}

// ---------------------------------------------------------------------------
// Block-wise QAT (onnxsim/qat_entry.h).
//
// Only the graph surgery crosses this boundary. Building the step graph is
// what cannot be done without onnxsim; *running* it is an ordinary inference
// loop -- bind the captured activations and the initial state, run the graph
// once per step, feed each state output back into its state input -- and the
// page already has a runtime for that (onnxruntime-web), so the loop stays in
// JS. qat_entry.h's "intended browser flow" comment is the contract; the
// object onnxsim_qat_build_step_graph returns names every piece of it, so a
// caller never has to guess a tensor name.

// QatOptions as a plain JS object with named fields:
//
//   { optimizer, learnScales, learnActivationScales, fakeQuant,
//     preserveSparsity, batchSize, batchSeed, shuffle }
//
// Named fields rather than eight positional arguments because they are
// independent knobs that each already have a default: an absent (or
// null/undefined) field keeps QatOptions' own, so `{}` means "full batch,
// train the weights only with Adam" -- apply_qat's default -- and a caller
// that wants one knob writes one field. `batchSize`/`batchSeed` arrive as JS
// numbers (doubles) and are truncated to the int64 fields they feed; nothing
// here is large enough for that to lose anything.
//
// `optimizer` is `"adam"` (QatOptions' own default, kept when the field is
// absent) or `"sgd_momentum"`, and picks only the block's own weight update --
// learnScales/learnActivationScales's parameters always train with Adam
// regardless, exactly as in apply_qat. Anything else is refused the same way
// an unrecognized value in any other field here would be: BuildQatStepGraph
// throws, and the binding returns null with the reason on the console.
//
// `fakeQuant: false` is the one field that changes what the two model
// arguments mean rather than adding a knob: the quantizer comes out of the
// middle and the second model's own float MatMul/Gemm weights are trained
// against the first model's activations -- apply_block_finetune, so the page
// can fine-tune a pruned or otherwise altered model and not only quantize
// one. It cannot be combined with either scale flag (there is no quantizer
// left for them to name); BuildQatStepGraph refuses that pairing and the
// binding returns null with the reason on the console, as it does for every
// other refusal.
QatOptions QatOptionsFromVal(em::val options) {
  QatOptions out;
  if (options.isUndefined() || options.isNull())
    return out;
  auto read_bool = [&options](const char *key, bool &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = v.as<bool>();
  };
  auto read_int = [&options](const char *key, int64_t &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = static_cast<int64_t>(v.as<double>());
  };
  auto read_string = [&options](const char *key, std::string &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = v.as<std::string>();
  };
  read_string("optimizer", out.optimizer);
  read_bool("learnScales", out.learn_scales);
  read_bool("learnActivationScales", out.learn_activation_scales);
  read_bool("fakeQuant", out.fake_quant);
  read_bool("preserveSparsity", out.preserve_sparsity);
  read_int("batchSize", out.batch_size);
  read_int("batchSeed", out.batch_seed);
  read_bool("shuffle", out.shuffle);
  return out;
}

// One TensorProto as { name, dtype, dims, data } -- deliberately the same
// shape onnxsim_parse_tensor returns, so the page decodes a QAT state tensor
// with the code it already has for a backend test's .pb. `data` is a view over
// `storage`, which the caller owns and must keep alive for as long as JS holds
// the view.
em::val QatTensorToVal(const onnx::TensorProto &tensor, std::string &storage) {
  em::val out = em::val::object();
  out.set("name", tensor.name());
  out.set("dtype", static_cast<int>(tensor.data_type()));
  em::val dims = em::val::array();
  for (int i = 0; i < tensor.dims_size(); ++i) {
    dims.set(i, static_cast<double>(tensor.dims(i)));
  }
  out.set("dims", dims);
  if (!TensorProtoToRawBytes(tensor, storage)) {
    // Every state tensor a step graph has is float32, so this is a "cannot
    // happen" that says so rather than handing back a silently empty buffer.
    std::ostringstream os;
    os << "unsupported tensor data type " << tensor.data_type();
    out.set("error", os.str());
    storage.clear();
  }
  out.set("data",
          em::val(em::typed_memory_view(
              storage.size(), reinterpret_cast<uint8_t *>(storage.data()))));
  return out;
}

// The plans onnxsim_qat_build_step_graph has built, alive on the C++ side and
// named to JS by an integer handle.
//
// A QatStepPlan is not a JS-shaped value: each QatTrainedLayer carries a whole
// TensorProto (the frozen weight scale) plus the block geometry
// WriteBackQatState re-derives integer codes from, and spelling all of that
// out across the boundary would be a second, hand-maintained encoding of a
// struct whose only reader is a C++ function. So the plan stays here and JS
// gets `planHandle`, hands it back to onnxsim_qat_write_back, and drops it
// with onnxsim_qat_release_plan. Nothing expires on its own: a page that
// trains block after block without releasing keeps every step graph it built.
std::map<int, QatStepPlan> &QatPlans() {
  static std::map<int, QatStepPlan> plans;
  return plans;
}

// Builds the step graph for one block of `quantized_data` against the float
// model that produced it -- BuildQatStepGraph, reachable from the page.
//
// `num_rows` is how many calibration rows the caller will bind (the leading
// dimension of every captured activation). It is a shape, not data: nothing
// here runs either model, and the activations themselves never cross this
// boundary in this direction.
//
// Returns null on a parse failure or a refused block (the reason goes to
// stderr, which the worker mirrors into the page's log, as with every other
// binding here), else:
//
//   {
//     stepGraph:  Uint8Array,                  // serialized ModelProto
//     planHandle: number,                      // for onnxsim_qat_write_back
//     state:      [{ input, output }],         // feed `output` back to `input`
//     scalars:    [string],                    // fresh scalar float per step
//     loss:       string,                      // "" when the graph has none
//     captures:   [{ input, source, dims, teacher }],
//     initialState: [{ name, dtype, dims, data }],
//     rowIndexInput: string,                   // "" unless batchSize > 0
//     rowIndexSize:  number,
//     numRows:       number,
//   }
//
// Driving the loop from that: bind every `initialState` tensor by its name and
// every capture (read `source` out of the float model, bind it as `input` --
// they differ whenever the step graph renames a tensor, e.g. under a
// minibatch, so binding by `source` would train on the wrong buffer); then per
// step feed the `scalars` (the learning rates plus Adam's two bias-correction
// factors) and, with a minibatch, `rowIndexSize` int64 row indices under
// `rowIndexInput`; read `loss` for diagnostics; and carry each state pair's
// `output` value into its `input` for the next step.
//
// Like every other model-returning binding here, `stepGraph` and each
// `initialState.data` are views over buffers reused by the next call -- copy
// them out (a fresh Uint8Array, or straight into an onnxruntime-web tensor)
// before calling back in.
em::val onnxsim_qat_build_step_graph(const std::string &float_data,
                                     const std::string &quantized_data,
                                     const std::string &block_input_name,
                                     const std::string &block_output_name,
                                     int num_rows, em::val options) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }

  QatStepPlan plan;
  try {
    plan = BuildQatStepGraph(float_model, quantized_model, block_input_name,
                             block_output_name, num_rows,
                             QatOptionsFromVal(options));
  } catch (const std::exception &e) {
    // BuildQatStepGraph refuses loudly -- an unclosed block, a node with no
    // gradient rule (named), a block with no layer of the requested scheme --
    // and the message is the actionable half, so it goes to the log.
    std::cerr << "qat_build_step_graph error: " << e.what() << std::endl;
    return em::val::null();
  }

  em::val step_graph = SerializeModel(plan.step_graph);
  if (step_graph.isNull()) {
    return em::val::null();
  }

  em::val out = em::val::object();
  out.set("stepGraph", step_graph);

  em::val state = em::val::array();
  for (size_t i = 0; i < plan.state.size(); ++i) {
    em::val entry = em::val::object();
    entry.set("input", plan.state[i].first);
    entry.set("output", plan.state[i].second);
    state.set(i, entry);
  }
  out.set("state", state);

  em::val scalars = em::val::array();
  for (size_t i = 0; i < plan.scalars.size(); ++i) {
    scalars.set(i, plan.scalars[i]);
  }
  out.set("scalars", scalars);
  out.set("loss", plan.loss_name);

  em::val captures = em::val::array();
  for (size_t i = 0; i < plan.captures.size(); ++i) {
    const QatCapture &capture = plan.captures[i];
    em::val entry = em::val::object();
    entry.set("input", capture.step_graph_input);
    entry.set("source", capture.source_tensor);
    em::val dims = em::val::array();
    for (size_t d = 0; d < capture.dims.size(); ++d) {
      dims.set(d, static_cast<double>(capture.dims[d]));
    }
    entry.set("dims", dims);
    entry.set("teacher", capture.is_teacher);
    captures.set(i, entry);
  }
  out.set("captures", captures);

  // One buffer per initial-state tensor, all of them live at once (unlike
  // onnxsim_parse_tensor's single static string, which only ever backs one
  // tensor at a time). Sized up front so the vector never reallocates while
  // the views into its elements are being made; reused, and so overwritten, by
  // the next call to this function.
  static std::vector<std::string> initial_state_raw;
  initial_state_raw.assign(plan.initial_state.size(), std::string());
  em::val initial_state = em::val::array();
  for (size_t i = 0; i < plan.initial_state.size(); ++i) {
    initial_state.set(
        i, QatTensorToVal(plan.initial_state[i], initial_state_raw[i]));
  }
  out.set("initialState", initial_state);

  out.set("rowIndexInput", plan.row_index_input);
  out.set("rowIndexSize", static_cast<double>(plan.row_index_size));
  out.set("numRows", static_cast<double>(plan.num_rows));

  static int next_plan_handle = 1;
  const int handle = next_plan_handle++;
  QatPlans().emplace(handle, std::move(plan));
  out.set("planHandle", handle);
  return out;
}

// Writes a finished loop's state back into the quantized model --
// WriteBackQatState, reachable from the page. Returns the tuned model's bytes,
// or null (with the reason on stderr) if the model will not parse, the handle
// is not a live plan, an entry is malformed, or the write-back itself refuses.
//
// `plan_handle` is the `planHandle` onnxsim_qat_build_step_graph returned.
// `final_state` is the loop's last state values as
// `{ <state input name>: { dims: [...], data: Float32Array } }` -- which is
// exactly the shape of an onnxruntime-web output tensor, so the last step's
// outputs can be handed back as they come, re-keyed from each state pair's
// `output` to its `input`. Every state tensor a step graph carries is float32,
// so that is what the values are read as.
//
// The plan is *not* released here: a caller may write back more than once
// (e.g. to compare a mid-training snapshot against the final one). Call
// onnxsim_qat_release_plan when the block is done.
em::val onnxsim_qat_write_back(const std::string &data, int plan_handle,
                               em::val final_state) {
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  auto plan = QatPlans().find(plan_handle);
  if (plan == QatPlans().end()) {
    std::cerr << "qat_write_back error: no plan with handle " << plan_handle
              << " (already released?)" << std::endl;
    return em::val::null();
  }
  if (final_state.isUndefined() || final_state.isNull()) {
    std::cerr << "qat_write_back error: final state is missing" << std::endl;
    return em::val::null();
  }

  std::map<std::string, onnx::TensorProto> state;
  em::val keys = em::val::global("Object").call<em::val>("keys", final_state);
  for (const std::string &name : em::vecFromJSArray<std::string>(keys)) {
    em::val entry = final_state[name];
    em::val dims = entry["dims"];
    em::val values = entry["data"];
    if (dims.isUndefined() || dims.isNull() || values.isUndefined() ||
        values.isNull()) {
      std::cerr << "qat_write_back error: final state entry '" << name
                << "' needs both dims and data" << std::endl;
      return em::val::null();
    }
    onnx::TensorProto &tensor = state[name];
    tensor.set_name(name);
    tensor.set_data_type(onnx::TensorProto::FLOAT);
    for (double dim : em::convertJSArrayToNumberVector<double>(dims)) {
      tensor.add_dims(static_cast<int64_t>(dim));
    }
    // raw_data, little-endian, matching what the emitter itself writes for
    // every tensor it builds (qat_graph_builder.cpp) -- one encoding on both
    // sides of the loop.
    const std::vector<float> raw =
        em::convertJSArrayToNumberVector<float>(values);
    tensor.set_raw_data(raw.data(), raw.size() * sizeof(float));
  }

  try {
    return SerializeModel(
        WriteBackQatState(quantized_model, plan->second, state));
  } catch (const std::exception &e) {
    std::cerr << "qat_write_back error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// Drops a plan onnxsim_qat_build_step_graph handed out. True if it was there.
// A plan holds its whole step graph, so a page training block after block
// should release each one as it finishes rather than at the end.
bool onnxsim_qat_release_plan(int plan_handle) {
  return QatPlans().erase(plan_handle) > 0;
}

// ---------------------------------------------------------------------------
// LoRA/QLoRA fine-tuning (onnxsim/lora_entry.h).
//
// The same split as QAT's, and for the same reason (see the comment above
// QAT's own section): building the step graph is graph surgery and belongs
// in wasm, running it is an ordinary inference loop and belongs wherever the
// runtime is. lora_entry.h's own top comment is the contract; the object
// onnxsim_lora_build_step_graph returns names every piece of it.
//
// The one shape difference from QAT is that LoRA also needs an *injection*
// binding. QAT takes an already-quantized model as input -- the quantizer has
// its own WASM entry points elsewhere (onnxsim_quantize_weight_only_int4 and
// friends) -- but LoRA's injection step (onnxsim/lora.py's inject_lora) has
// no such precedent: it is graph surgery private to this module. So the flow
// here is four calls rather than QAT's three: onnxsim_lora_inject, then
// onnxsim_lora_build_step_graph, the loop, onnxsim_lora_write_back, and
// onnxsim_lora_release_plan.

// InjectLoraOptions as a plain JS object with named fields:
//
//   { rank, hasAlpha, alpha, targetOpTypes, restrictTargetNames, targetNames,
//     seed }
//
// Absent (or null/undefined) fields keep InjectLoraOptions' own defaults, so
// `{}` means "inject every eligible MatMul/Gemm/Conv at rank 8, unscaled".
// `restrictTargetNames`/`targetNames` are the one pair where absent and
// present-but-empty are different requests, exactly as InjectLoraOptions'
// own field comment explains: `restrictTargetNames: false` (the default)
// injects everything eligible regardless of what `targetNames` holds, while
// `restrictTargetNames: true, targetNames: []` injects nothing at all -- a
// caller cannot express the second with `targetNames` alone.
InjectLoraOptions InjectLoraOptionsFromVal(em::val options) {
  InjectLoraOptions out;
  if (options.isUndefined() || options.isNull())
    return out;
  auto read_bool = [&options](const char *key, bool &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = v.as<bool>();
  };
  auto read_int = [&options](const char *key, int64_t &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = static_cast<int64_t>(v.as<double>());
  };
  auto read_strings = [&options](const char *key,
                                  std::vector<std::string> &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = em::vecFromJSArray<std::string>(v);
  };
  read_int("rank", out.rank);
  read_bool("hasAlpha", out.has_alpha);
  {
    em::val v = options["alpha"];
    if (!v.isUndefined() && !v.isNull())
      out.alpha = v.as<float>();
  }
  read_strings("targetOpTypes", out.target_op_types);
  read_bool("restrictTargetNames", out.restrict_target_names);
  read_strings("targetNames", out.target_names);
  {
    // uint64_t rather than int64_t, matching InjectLoraOptions::seed --
    // narrowed through a JS double the same way batchSize/batchSeed are
    // above, which loses nothing for a seed value in practice.
    em::val v = options["seed"];
    if (!v.isUndefined() && !v.isNull())
      out.seed = static_cast<uint64_t>(v.as<double>());
  }
  return out;
}

// LoraOptions as a plain JS object: { batchSize, batchSeed, shuffle } -- the
// same three fields QatOptions carries for its own minibatch schedule
// (QatOptionsFromVal above), and nothing else: LoRA has no scales to learn
// and no sparsity to preserve, so it needs none of QatOptions' other knobs.
LoraOptions LoraOptionsFromVal(em::val options) {
  LoraOptions out;
  if (options.isUndefined() || options.isNull())
    return out;
  auto read_bool = [&options](const char *key, bool &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = v.as<bool>();
  };
  auto read_int = [&options](const char *key, int64_t &dst) {
    em::val v = options[key];
    if (!v.isUndefined() && !v.isNull())
      dst = static_cast<int64_t>(v.as<double>());
  };
  read_int("batchSize", out.batch_size);
  read_int("batchSeed", out.batch_seed);
  read_bool("shuffle", out.shuffle);
  return out;
}

// One LoraTarget as { weightName, nodeOutput, opType, loraAName, loraBName,
// rank, hasAlpha, alpha } -- field for field, so a page can list what got
// injected (which weight, which op type, at what rank) without decoding the
// model itself. `alpha` is 0 when `hasAlpha` is false, matching
// LoraTarget::alpha's own "meaningful only when has_alpha is set" contract.
em::val LoraTargetToVal(const LoraTarget &target) {
  em::val out = em::val::object();
  out.set("weightName", target.weight_name);
  out.set("nodeOutput", target.node_output);
  out.set("opType", target.op_type);
  out.set("loraAName", target.lora_a_name);
  out.set("loraBName", target.lora_b_name);
  out.set("rank", static_cast<double>(target.rank));
  out.set("hasAlpha", target.has_alpha);
  out.set("alpha", target.has_alpha ? target.alpha : 0.0f);
  return out;
}

// A LoraAdapter as a plain array of LoraTargetToVal entries -- what
// onnxsim_lora_inject hands back and what onnxsim_lora_build_step_graph
// takes in, so JS can filter the array (train only some of the injected
// branches, per BuildLoraStepGraph's own "a caller may pass a LoraAdapter
// with a subset of targets" contract) with no C++ round trip.
em::val LoraAdapterToVal(const LoraAdapter &adapter) {
  em::val out = em::val::array();
  for (size_t i = 0; i < adapter.targets.size(); ++i) {
    out.set(i, LoraTargetToVal(adapter.targets[i]));
  }
  return out;
}

// The inverse of LoraAdapterToVal -- reads a JS array of the same shape back
// into a LoraAdapter. Throws std::invalid_argument (caught by the calling
// binding, same as every other refusal here) rather than defaulting a
// missing field: an adapter entry with, say, no `loraBName` is a caller bug
// (a hand-built array, or one filtered incorrectly) and BuildLoraStepGraph
// would fail confusingly further in rather than saying so here.
LoraAdapter LoraAdapterFromVal(em::val adapter_val) {
  if (adapter_val.isUndefined() || adapter_val.isNull()) {
    throw std::invalid_argument("lora adapter is missing");
  }
  LoraAdapter out;
  const int n = adapter_val["length"].as<int>();
  for (int i = 0; i < n; ++i) {
    em::val item = adapter_val[i];
    if (item.isUndefined() || item.isNull()) {
      throw std::invalid_argument("lora adapter entry " + std::to_string(i) +
                                  " is missing");
    }
    LoraTarget target;
    target.weight_name = item["weightName"].as<std::string>();
    target.node_output = item["nodeOutput"].as<std::string>();
    target.op_type = item["opType"].as<std::string>();
    target.lora_a_name = item["loraAName"].as<std::string>();
    target.lora_b_name = item["loraBName"].as<std::string>();
    target.rank = static_cast<int64_t>(item["rank"].as<double>());
    target.has_alpha = item["hasAlpha"].as<bool>();
    target.alpha = target.has_alpha ? item["alpha"].as<float>() : 0.0f;
    out.targets.push_back(std::move(target));
  }
  return out;
}

// The plans onnxsim_lora_build_step_graph has built, alive on the C++ side
// and named to JS by an integer handle -- LoRA's own parallel registry to
// QatPlans() above, not shared with it. A LoraStepPlan is a different C++
// type from QatStepPlan, and a page that is training a LoRA adapter and a
// QAT block in the same session (nothing here forbids it) must be able to
// release one without touching the other's handle numbering.
std::map<int, LoraStepPlan> &LoraPlans() {
  static std::map<int, LoraStepPlan> plans;
  return plans;
}

// Injects a trainable low-rank adapter branch -- InjectLora, reachable from
// the page. Unlike onnxsim_qat_build_step_graph, which takes an
// already-quantized model produced by one of the quantize_* bindings, LoRA
// has no separate injection entry point elsewhere (see this section's own
// top comment), so this binding is what produces the model
// onnxsim_lora_build_step_graph trains.
//
// Returns null on a parse failure or a refused injection (an
// onnx::checker::ValidationError from InjectLora's own final check; the
// reason goes to stderr, as with every other refusal here), else:
//
//   {
//     model:   Uint8Array,   // serialized ModelProto, the injected model
//     adapter: [{ weightName, nodeOutput, opType, loraAName, loraBName,
//                 rank, hasAlpha, alpha }],
//   }
//
// `adapter` is every target InjectLora produced, in injection order -- hand
// it to onnxsim_lora_build_step_graph unchanged to train all of them, or
// filter it first to train only some.
//
// Like every other model-returning binding here, `model` is a view over a
// buffer the next call reuses -- copy it out before calling back in.
em::val onnxsim_lora_inject(const std::string &data, em::val options) {
  onnx::ModelProto model;
  if (!model.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }

  LoraInjectionResult result;
  try {
    result = InjectLora(model, InjectLoraOptionsFromVal(options));
  } catch (const std::exception &e) {
    std::cerr << "lora_inject error: " << e.what() << std::endl;
    return em::val::null();
  }

  em::val model_bytes = SerializeModel(result.model);
  if (model_bytes.isNull()) {
    return em::val::null();
  }
  em::val out = em::val::object();
  out.set("model", model_bytes);
  out.set("adapter", LoraAdapterToVal(result.adapter));
  return out;
}

// Builds the step graph for one block of `injected_data` -- BuildLoraStepGraph,
// reachable from the page. Mirrors onnxsim_qat_build_step_graph's shape and
// contract closely (see that function's own doc comment for the fuller
// walkthrough, which applies here too); what differs is LoRA's own: one
// model rather than a float/quantized pair -- there is no separate teacher
// model argument here, because the reconstruction target is whatever the
// caller captures and binds as the block's teacher capture, ordinarily read
// from a reference model per lora_entry.h's "intended browser flow" --
// `adapter` (this run's own LoraAdapter, as onnxsim_lora_inject returned it
// or a caller-filtered subset) in place of a quantization scheme, and
// `parameters` in the returned object (see below).
//
// `num_rows` is how many calibration rows the caller will bind (the leading
// dimension of every captured activation). It is a shape, not data: nothing
// here runs the injected model or the reference model, and neither model's
// activations cross this boundary in this direction.
//
// Returns null on a parse failure or a refused block -- see
// BuildLoraStepGraph's own doc comment for the refusal cases: an unclosed
// block, a node with no gradient rule (named), an adapter with no targets, or
// a minibatch whose captures disagree on their row count -- else:
//
//   {
//     stepGraph:  Uint8Array,                  // serialized ModelProto
//     planHandle: number,                      // for onnxsim_lora_write_back
//     state:      [{ input, output }],         // feed `output` back to `input`
//     scalars:    [string],                    // "lora__lr" plus Adam's two
//                                               // bias-correction factors --
//                                               // no scale rates, unlike QAT
//     loss:       string,                      // "" when the graph has none
//     captures:   [{ input, source, dims, teacher }],
//     initialState: [{ name, dtype, dims, data }],
//     rowIndexInput: string,                   // "" unless batchSize > 0
//     rowIndexSize:  number,
//     numRows:       number,
//     parameters:    [string],                 // onnxsim_lora_write_back's
//                                               // finalState key list --
//                                               // LoraStepPlan::parameters
//   }
//
// Driving the loop from that is identical to onnxsim_qat_build_step_graph's
// own (see that function's doc comment): bind every `initialState` tensor by
// its name and every capture by its `input` (never its `source` -- they
// differ under a minibatch); feed the `scalars` and, with a minibatch,
// `rowIndexSize` int64 row indices under `rowIndexInput` each step; read
// `loss` for diagnostics; and carry each state pair's `output` into its
// `input` for the next step.
//
// Like every other model-returning binding here, `stepGraph` and each
// `initialState.data` are views over buffers reused by the next call -- copy
// them out before calling back in.
em::val onnxsim_lora_build_step_graph(const std::string &injected_data,
                                      em::val adapter_val,
                                      const std::string &block_input_name,
                                      const std::string &block_output_name,
                                      int num_rows, em::val options) {
  onnx::ModelProto injected_model;
  if (!injected_model.ParseFromArray(injected_data.data(),
                                     injected_data.size())) {
    std::cerr << "Parse failed (injected model)" << std::endl;
    return em::val::null();
  }

  LoraStepPlan plan;
  try {
    const LoraAdapter adapter = LoraAdapterFromVal(adapter_val);
    plan = BuildLoraStepGraph(injected_model, adapter, block_input_name,
                              block_output_name, num_rows,
                              LoraOptionsFromVal(options));
  } catch (const std::exception &e) {
    // BuildLoraStepGraph refuses loudly -- an unclosed block, a node with no
    // gradient rule (named), an adapter with no targets -- and the message
    // is the actionable half, so it goes to the log.
    std::cerr << "lora_build_step_graph error: " << e.what() << std::endl;
    return em::val::null();
  }

  em::val step_graph = SerializeModel(plan.step_graph);
  if (step_graph.isNull()) {
    return em::val::null();
  }

  em::val out = em::val::object();
  out.set("stepGraph", step_graph);

  em::val state = em::val::array();
  for (size_t i = 0; i < plan.state.size(); ++i) {
    em::val entry = em::val::object();
    entry.set("input", plan.state[i].first);
    entry.set("output", plan.state[i].second);
    state.set(i, entry);
  }
  out.set("state", state);

  em::val scalars = em::val::array();
  for (size_t i = 0; i < plan.scalars.size(); ++i) {
    scalars.set(i, plan.scalars[i]);
  }
  out.set("scalars", scalars);
  out.set("loss", plan.loss_name);

  em::val captures = em::val::array();
  for (size_t i = 0; i < plan.captures.size(); ++i) {
    const LoraCapture &capture = plan.captures[i];
    em::val entry = em::val::object();
    entry.set("input", capture.step_graph_input);
    entry.set("source", capture.source_tensor);
    em::val dims = em::val::array();
    for (size_t d = 0; d < capture.dims.size(); ++d) {
      dims.set(d, static_cast<double>(capture.dims[d]));
    }
    entry.set("dims", dims);
    entry.set("teacher", capture.is_teacher);
    captures.set(i, entry);
  }
  out.set("captures", captures);

  // Same per-call reuse pattern onnxsim_qat_build_step_graph's own
  // initial_state_raw uses (see that function's comment) -- a parallel
  // static buffer rather than a shared one, so a LoRA build and a QAT build
  // alive at once do not stomp on each other's views. QatTensorToVal itself
  // is reused unchanged: it encodes a bare TensorProto as { name, dtype,
  // dims, data } with no QAT-specific content, which is exactly what a LoRA
  // state tensor needs too.
  static std::vector<std::string> initial_state_raw;
  initial_state_raw.assign(plan.initial_state.size(), std::string());
  em::val initial_state = em::val::array();
  for (size_t i = 0; i < plan.initial_state.size(); ++i) {
    initial_state.set(
        i, QatTensorToVal(plan.initial_state[i], initial_state_raw[i]));
  }
  out.set("initialState", initial_state);

  out.set("rowIndexInput", plan.row_index_input);
  out.set("rowIndexSize", static_cast<double>(plan.row_index_size));
  out.set("numRows", static_cast<double>(plan.num_rows));

  em::val parameters = em::val::array();
  for (size_t i = 0; i < plan.parameters.size(); ++i) {
    parameters.set(i, plan.parameters[i]);
  }
  out.set("parameters", parameters);

  static int next_plan_handle = 1;
  const int handle = next_plan_handle++;
  LoraPlans().emplace(handle, std::move(plan));
  out.set("planHandle", handle);
  return out;
}

// Writes a finished loop's state back into the injected model --
// WriteBackLoraState, reachable from the page. Mirrors onnxsim_qat_write_back
// closely (see that function's own doc comment for `final_state`'s shape);
// the one difference is what the write-back reads out of it.
// WriteBackLoraState only looks up `plan.parameters` (each adapter tensor's
// own name) in `final_state`, unlike QAT's write-back, which also reads
// scale/zero-point state when those were trained -- but every other entry
// (the Adam moments the loop's own `state` list also carries) is simply
// unread rather than rejected, so this passes `final_state` through as
// given rather than trimming it first.
//
// `plan_handle` is the `planHandle` onnxsim_lora_build_step_graph returned.
// Returns the tuned model's bytes, or null (with the reason on stderr) if the
// model will not parse, the handle is not a live plan, an entry is malformed,
// or the write-back itself refuses (a state tensor's element count does not
// match the adapter tensor it is meant to replace).
//
// The plan is *not* released here, for the same reason as QAT's: a caller
// may write back more than once (e.g. to compare a mid-training snapshot
// against the final one). Call onnxsim_lora_release_plan when the block is
// done.
em::val onnxsim_lora_write_back(const std::string &data, int plan_handle,
                                em::val final_state) {
  onnx::ModelProto injected_model;
  if (!injected_model.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  auto plan = LoraPlans().find(plan_handle);
  if (plan == LoraPlans().end()) {
    std::cerr << "lora_write_back error: no plan with handle " << plan_handle
              << " (already released?)" << std::endl;
    return em::val::null();
  }
  if (final_state.isUndefined() || final_state.isNull()) {
    std::cerr << "lora_write_back error: final state is missing" << std::endl;
    return em::val::null();
  }

  std::map<std::string, onnx::TensorProto> state;
  em::val keys = em::val::global("Object").call<em::val>("keys", final_state);
  for (const std::string &name : em::vecFromJSArray<std::string>(keys)) {
    em::val entry = final_state[name];
    em::val dims = entry["dims"];
    em::val values = entry["data"];
    if (dims.isUndefined() || dims.isNull() || values.isUndefined() ||
        values.isNull()) {
      std::cerr << "lora_write_back error: final state entry '" << name
                << "' needs both dims and data" << std::endl;
      return em::val::null();
    }
    onnx::TensorProto &tensor = state[name];
    tensor.set_name(name);
    tensor.set_data_type(onnx::TensorProto::FLOAT);
    for (double dim : em::convertJSArrayToNumberVector<double>(dims)) {
      tensor.add_dims(static_cast<int64_t>(dim));
    }
    // raw_data, little-endian, matching what the emitter itself writes for
    // every tensor it builds (qat_graph_builder.cpp, shared by LoRA's own
    // step-graph builder) -- one encoding on both sides of the loop.
    const std::vector<float> raw =
        em::convertJSArrayToNumberVector<float>(values);
    tensor.set_raw_data(raw.data(), raw.size() * sizeof(float));
  }

  try {
    return SerializeModel(
        WriteBackLoraState(injected_model, plan->second, state));
  } catch (const std::exception &e) {
    std::cerr << "lora_write_back error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// Drops a plan onnxsim_lora_build_step_graph handed out. True if it was
// there. Mirrors onnxsim_qat_release_plan; see that function's own comment --
// nothing here expires on its own either.
bool onnxsim_lora_release_plan(int plan_handle) {
  return LoraPlans().erase(plan_handle) > 0;
}

// ---------------------------------------------------------------------------
// Data-free quantization / pruning passes already ported to C++ (see
// onnxsim.h, quantize_entry.h, pruning_entry.h, structured_pruning_entry.h,
// cross_layer_equalization_entry.h) but previously reachable only from
// Python. Each binding below is a pure graph rewrite -- model bytes in,
// model bytes out, no executor and no calibration data -- so it follows the
// exact same parse/call/SerializeModel shape as onnxsim_quantize_dynamic
// above, and never suspends on Asyncify (safe to call synchronously from
// any JS context, ORT-web or built-in-ORT build alike).

em::val onnxsim_cross_layer_equalize(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(CrossLayerEqualize(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "cross_layer_equalize error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_dynamic_matmul_integer_to_float(
    const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeDynamicMatMulIntegerToFloat(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_dynamic_matmul_integer_to_float error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_attention_dynamic(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeAttentionDynamic(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_attention_dynamic error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_int16(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnlyInt16(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_int16 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_int8_block(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnlyInt8Block(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_int8_block error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_mxfp4(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnlyMXFP4(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_mxfp4 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_matmul_nbits(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnlyMatMulNBits(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_matmul_nbits error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_double_quantization(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyDoubleQuantization(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_double_quantization error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// `bits`/`max_bits`/`block_size` mirror apply_any_precision_llm_cpp's own
// parameters of the same names exactly (see onnxsim.h).
em::val onnxsim_apply_any_precision_llm(const std::string &data, int bits,
                                        int max_bits, int block_size) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyAnyPrecisionLlm(xmodel, bits, max_bits,
                                               block_size));
  } catch (const std::exception &e) {
    std::cerr << "apply_any_precision_llm error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// `seed` arrives as a JS number (double) and is narrowed to the uint64 it
// feeds, the same way QatOptions' batchSeed is above; `block_size`/
// `epsilon` mirror apply_quarot_cpp's own parameters exactly (see onnxsim.h).
em::val onnxsim_apply_quarot(const std::string &data, double seed,
                             int block_size, float epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQuarot(xmodel, static_cast<uint64_t>(seed),
                                      block_size, epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_quarot error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_iq4_nl(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyIQ4NL(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_iq4_nl error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q4_0(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ4_0(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q4_0 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q4_1(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ4_1(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q4_1 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q5_0(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ5_0(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q5_0 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q5_1(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ5_1(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q5_1 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q8_0(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ8_0(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q8_0 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q2_k(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ2K(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q2_k error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q3_k(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ3K(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q3_k error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q4_k(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ4K(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q4_k error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q5_k(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ5K(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q5_k error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_ternary(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufTernaryQuant(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_ternary error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_fp6_llm(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyFp6Llm(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_fp6_llm error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_gguf_q6_k(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGgufQ6K(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_gguf_q6_k error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_leptoquant(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyLeptoquant(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_leptoquant error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_nf4(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyNF4(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_nf4 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_if4(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyIF4(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_if4 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_nvfp4(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyNVFP4Quantization(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_nvfp4 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_deepseek_fp8(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyDeepSeekFp8(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_deepseek_fp8 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_kmeans_quantization(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyKMeansQuantization(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_kmeans_quantization error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_hqq(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyHQQ(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_hqq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_ibert_gelu(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyIBertGelu(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_ibert_gelu error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_ibert_softmax(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyIBertSoftmax(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_ibert_softmax error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_adpq(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyADPQ(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_adpq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_icquant(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyICQuant(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_icquant error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_olive(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyOlive(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_olive error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_aqlm(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyAQLM(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_aqlm error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_drop_by_drop(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyDropByDrop(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_drop_by_drop error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_lo_bcq(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyLoBcq(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_lo_bcq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_quip_sharp(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQuipSharp(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_quip_sharp error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_attention_quantization(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyAttentionQuantization(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_attention_quantization error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// `block_size`/`epsilon` mirror apply_zeroquant_cpp's own parameters
// exactly (see onnxsim.h) -- the same way onnxsim_apply_quarot's own are.
em::val onnxsim_apply_zeroquant(const std::string &data, int block_size,
                                float epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(
        ApplyZeroQuant(xmodel, static_cast<int64_t>(block_size), epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_zeroquant error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_intactkv(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyIntactKv(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_intactkv error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_kbvq_moe(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyKbvqMoe(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_kbvq_moe error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_quantize_weight_only_llm_fp4(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(QuantizeWeightOnlyLlmFp4(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_llm_fp4 error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_qoq(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQoq(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_qoq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_dsq(const std::string &data) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyDsq(xmodel));
  } catch (const std::exception &e) {
    std::cerr << "apply_dsq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// DAQ is data-free but still needs two full models (base + fine-tuned,
// matched by node output name) -- the same two-model shape
// onnxsim_list_correctable_outputs/onnxsim_apply_bias_corrections above
// already use, minus any activation-probing arguments.
em::val onnxsim_apply_daq(const std::string &base_data,
                          const std::string &post_trained_data,
                          const std::string &metric,
                          em::val skip_names_ary) {
  onnx::ModelProto base_model, post_trained_model;
  if (!base_model.ParseFromArray(base_data.data(), base_data.size()) ||
      !post_trained_model.ParseFromArray(post_trained_data.data(),
                                         post_trained_data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  std::unordered_set<std::string> skip_names;
  if (!skip_names_ary.isUndefined() && !skip_names_ary.isNull()) {
    for (const auto &name :
        em::vecFromJSArray<std::string>(skip_names_ary)) {
      skip_names.insert(name);
    }
  }
  try {
    return SerializeModel(
        ApplyDaq(base_model, post_trained_model, metric, skip_names));
  } catch (const std::exception &e) {
    std::cerr << "apply_daq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// Low-Rank Compensation is data-free but still needs two full models
// (float + INT4-quantized, matched by node output name) -- the same
// two-model shape onnxsim_apply_daq above already uses. `rank` arrives as
// a JS number (double) and is narrowed to the int64_t it feeds, the same
// way onnxsim_apply_quarot's own `seed` is -- embind has no binding for
// int64_t itself.
em::val onnxsim_apply_low_rank_compensation(const std::string &float_data,
                                            const std::string &quantized_data,
                                            double rank) {
  onnx::ModelProto float_model, quantized_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size()) ||
      !quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyLowRankCompensation(
        float_model, quantized_model, static_cast<int64_t>(rank)));
  } catch (const std::exception &e) {
    std::cerr << "apply_low_rank_compensation error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// Embedding-output binarization is data-free, single-model, and needs no
// ModelExecutor -- it targets a whole graph OUTPUT declaration rather than
// a matched node. An absent (undefined/null) `output_name` is this port's
// own empty-string stand-in for Python's `output_name=None` sentinel
// ("resolve automatically, requiring exactly one FLOAT output").
em::val onnxsim_quantize_embedding_binary(const std::string &data,
                                          em::val output_name_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  std::string output_name;
  if (!output_name_val.isUndefined() && !output_name_val.isNull()) {
    output_name = output_name_val.as<std::string>();
  }
  try {
    return SerializeModel(
        ApplyEmbeddingQuantizationBinary(xmodel, output_name));
  } catch (const std::exception &e) {
    std::cerr << "quantize_embedding_binary error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// An absent (undefined/null) `n`/`m` is "not given" (unstructured sparsity,
// ranked by `sparsity`); a present number must be given together with the
// other one (N:M semi-structured) -- mirroring prune_magnitude_cpp's own
// keyword-argument contract exactly (see onnxsim.h).
std::optional<int64_t> OptInt64FromVal(em::val v) {
  if (v.isUndefined() || v.isNull())
    return std::nullopt;
  return static_cast<int64_t>(v.as<double>());
}

em::val onnxsim_prune_magnitude(const std::string &data, double sparsity,
                                em::val n_val, em::val m_val,
                                bool global_sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(PruneMagnitude(xmodel, sparsity,
                                         OptInt64FromVal(n_val),
                                         OptInt64FromVal(m_val),
                                         global_sparsity));
  } catch (const std::exception &e) {
    std::cerr << "prune_magnitude error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_structured_pruning(const std::string &data,
                                         double sparsity,
                                         const std::string &importance_norm,
                                         bool global_sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyStructuredPruning(xmodel, sparsity,
                                                 importance_norm,
                                                 global_sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_structured_pruning error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_attention_head_pruning(const std::string &data,
                                             double sparsity,
                                             const std::string &importance_norm) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(
        ApplyAttentionHeadPruning(xmodel, sparsity, importance_norm));
  } catch (const std::exception &e) {
    std::cerr << "apply_attention_head_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_moe_expert_channel_pruning(const std::string &data,
                                                 double sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyMoeExpertChannelPruning(xmodel, sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_moe_expert_channel_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_qmoe_expert_channel_pruning(const std::string &data,
                                                  double sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQMoEExpertChannelPruning(xmodel, sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_qmoe_expert_channel_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// An absent (undefined/null) token-id list is "not given" (std::nullopt); a
// present JS array crosses as the vector -- the same "absent vs.
// present-but-empty are different requests" shape as LoRA's
// restrictTargetNames/targetNames above. `input_name` is likewise absent or
// a string. Returns an object mirroring the C++
// EmbeddingVocabPruningResult:
//
//   { model: Uint8Array, matched: boolean, keptTokenIds: [...],
//     lmHeadPruned: boolean }
//
// (`id_map` is `{keptTokenIds[i]: i}`, trivially rebuilt by the caller --
// see structured_pruning_entry.h's own comment on why it never crosses.)
em::val
EmbeddingVocabPruningResultToVal(const EmbeddingVocabPruningResult &result) {
  em::val model_bytes = SerializeModel(result.model);
  if (model_bytes.isNull()) {
    return em::val::null();
  }
  em::val out = em::val::object();
  out.set("model", model_bytes);
  out.set("matched", result.matched);
  em::val kept = em::val::array();
  for (size_t i = 0; i < result.kept_token_ids.size(); ++i) {
    kept.set(i, static_cast<double>(result.kept_token_ids[i]));
  }
  out.set("keptTokenIds", kept);
  out.set("lmHeadPruned", result.lm_head_pruned);
  return out;
}

std::optional<std::vector<int64_t>> OptInt64ListFromVal(em::val v) {
  if (v.isUndefined() || v.isNull())
    return std::nullopt;
  // Via doubles: embind's vecFromJSArray cannot convert plain JS numbers to
  // int64_t directly ("emval::as has unknown type"), so convert as doubles
  // first -- the same narrowing QatOptions' batchSize/batchSeed already use.
  std::vector<int64_t> out;
  for (double d : em::vecFromJSArray<double>(v)) {
    out.push_back(static_cast<int64_t>(d));
  }
  return out;
}

std::optional<std::string> OptStringFromVal(em::val v) {
  if (v.isUndefined() || v.isNull())
    return std::nullopt;
  return v.as<std::string>();
}

em::val onnxsim_apply_embedding_vocab_pruning(const std::string &data,
                                              em::val keep_token_ids_val,
                                              em::val drop_token_ids_val,
                                              em::val input_name_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return EmbeddingVocabPruningResultToVal(ApplyEmbeddingVocabPruning(
        xmodel, OptInt64ListFromVal(keep_token_ids_val),
        OptInt64ListFromVal(drop_token_ids_val),
        OptStringFromVal(input_name_val)));
  } catch (const std::exception &e) {
    std::cerr << "apply_embedding_vocab_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_embedding_vocab_magnitude_pruning(
    const std::string &data, double sparsity, em::val protect_token_ids_val,
    em::val input_name_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return EmbeddingVocabPruningResultToVal(
        ApplyEmbeddingVocabMagnitudePruning(
            xmodel, sparsity, OptInt64ListFromVal(protect_token_ids_val),
            OptStringFromVal(input_name_val)));
  } catch (const std::exception &e) {
    std::cerr << "apply_embedding_vocab_magnitude_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// ---------------------------------------------------------------------------
// Calibration-driven passes (Wanda/SparseGPT/imatrix families, MoE
// whole-expert, transformer-block depth pruning -- see
// structured_pruning_entry.h and imatrix_quant_entry.h).
//
// Same division of labour as static quantization above: the page runs the
// model over sample inputs with onnxruntime-web itself and hands the
// observed activations back across this boundary; C++ only runs the
// probe-augmented model over them (via the same executor fold_constant
// uses) and applies the rewrite. `calibration_batches` is a JS array of
// batches, each batch a plain object mapping a graph input name to one
// tensor:
//
//   [{ "<inputName>": { dtype, dims, data }, ... }, ...]
//
// `dtype` is the ONNX TensorProto.DataType enum value, `dims` the shape,
// `data` a Uint8Array (or any TypedArray view -- Float32Array, BigInt64Array,
// ...) whose underlying bytes are the tensor's raw little-endian element
// data (exactly what onnxruntime-web's own tensor `.data` already holds, so
// `new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)`
// crosses with no copy on the JS side). A batch missing one of the model's
// own graph inputs throws (surfaced as null with the reason on stderr),
// exactly like the C++ entry points' own std::invalid_argument contract.
//
// Because these run the model through the executor, they suspend on Asyncify
// in the ORT-web build exactly like onnxsim_fold_constant does -- a JS
// caller must await the result when it is a Promise
// (`if (r?.then) r = await r`, see worker.js' own fold_constant case).

// Raw element size of the numeric ONNX dtypes calibration tensors can take.
// 0 means "not a plain numeric dtype" (STRING and friends) -- the caller
// refuses those rather than guessing a layout.
size_t OnnxCalibrationDtypeSize(int32_t dtype) {
  using TP = onnx::TensorProto;
  switch (dtype) {
  case TP::FLOAT:
  case TP::INT32:
  case TP::UINT32:
    return 4;
  case TP::DOUBLE:
  case TP::INT64:
  case TP::UINT64:
    return 8;
  case TP::INT16:
  case TP::UINT16:
  case TP::FLOAT16:
  case TP::BFLOAT16:
    return 2;
  case TP::INT8:
  case TP::UINT8:
  case TP::BOOL:
    return 1;
  default:
    return 0;
  }
}

// Copy a JS TypedArray/Uint8Array's underlying bytes into `out`. `data` must
// be a view (it has a `buffer`/`byteOffset`/`byteLength`); a plain JS number
// array is refused loudly rather than converted element-wise, since that
// conversion would silently reinterpret float values as bytes.
std::string JsTypedArrayToBytes(em::val data, size_t expect_bytes,
                                const std::string &name) {
  em::val buffer = data["buffer"];
  em::val byte_offset = data["byteOffset"];
  em::val byte_length = data["byteLength"];
  if (buffer.isUndefined() || buffer.isNull() || byte_offset.isUndefined() ||
      byte_offset.isNull() || byte_length.isUndefined() ||
      byte_length.isNull()) {
    throw std::invalid_argument("calibration tensor '" + name +
                                "' data must be a TypedArray/Uint8Array of "
                                "raw little-endian bytes, not a plain array");
  }
  const size_t nbytes = byte_length.as<size_t>();
  if (nbytes != expect_bytes) {
    std::ostringstream os;
    os << "calibration tensor '" << name << "' has " << nbytes
       << " bytes but dtype*dims implies " << expect_bytes;
    throw std::invalid_argument(os.str());
  }
  em::val u8 = em::val::global("Uint8Array").new_(
      buffer, byte_offset.as<size_t>(), nbytes);
  std::string out;
  out.resize(nbytes);
  if (nbytes != 0) {
    em::val dest = em::val(emscripten::typed_memory_view(
        nbytes, reinterpret_cast<uint8_t *>(out.data())));
    dest.call<void>("set", u8);
  }
  return out;
}

std::vector<std::unordered_map<std::string, onnx::TensorProto>>
ParseCalibrationBatches(em::val batches_val) {
  std::vector<std::unordered_map<std::string, onnx::TensorProto>> batches;
  if (batches_val.isUndefined() || batches_val.isNull()) {
    return batches;
  }
  const unsigned num_batches = batches_val["length"].as<unsigned>();
  batches.reserve(num_batches);
  for (unsigned b = 0; b < num_batches; ++b) {
    em::val batch_val = batches_val[b];
    if (batch_val.isUndefined() || batch_val.isNull()) {
      throw std::invalid_argument("calibration batch " + std::to_string(b) +
                                  " is missing");
    }
    std::unordered_map<std::string, onnx::TensorProto> batch;
    em::val keys =
        em::val::global("Object").call<em::val>("keys", batch_val);
    for (const std::string &name : em::vecFromJSArray<std::string>(keys)) {
      em::val entry = batch_val[name.c_str()];
      if (entry.isUndefined() || entry.isNull()) {
        throw std::invalid_argument("calibration tensor '" + name +
                                    "' is missing");
      }
      em::val dtype_val = entry["dtype"];
      em::val dims_val = entry["dims"];
      em::val data_val = entry["data"];
      if (dtype_val.isUndefined() || dtype_val.isNull() ||
          dims_val.isUndefined() || dims_val.isNull() ||
          data_val.isUndefined() || data_val.isNull()) {
        throw std::invalid_argument("calibration tensor '" + name +
                                    "' needs dtype, dims and data");
      }
      const int32_t dtype = dtype_val.as<int32_t>();
      const size_t elem_size = OnnxCalibrationDtypeSize(dtype);
      if (elem_size == 0) {
        throw std::invalid_argument("calibration tensor '" + name +
                                    "' has unsupported dtype " +
                                    std::to_string(dtype));
      }
      onnx::TensorProto tensor;
      tensor.set_name(name);
      tensor.set_data_type(
          static_cast<onnx::TensorProto::DataType>(dtype));
      int64_t numel = 1;
      for (double dim : em::convertJSArrayToNumberVector<double>(dims_val)) {
        const int64_t d = static_cast<int64_t>(dim);
        tensor.add_dims(d);
        numel *= d;
      }
      tensor.set_raw_data(JsTypedArrayToBytes(
          data_val, static_cast<size_t>(numel) * elem_size, name));
      batch.emplace(name, std::move(tensor));
    }
    batches.push_back(std::move(batch));
  }
  return batches;
}

// The executor calibration-driven passes run probe sub-models through --
// the same selection fold_constant makes (see its own call site above).
const ModelExecutor &CalibrationExecutor() {
#ifdef ONNXSIM_WASM_HOOKABLE_EXECUTOR
  return *GetJsModelExecutor();
#else
  return *GetBuiltinModelExecutor();
#endif
}

em::val onnxsim_apply_structured_wanda_pruning(
    const std::string &data, em::val calibration_batches_val, double sparsity,
    double epsilon, const std::string &importance_norm,
    bool global_sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyStructuredWandaPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity, epsilon,
        importance_norm, global_sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_structured_wanda_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_attention_head_wanda_pruning(
    const std::string &data, em::val calibration_batches_val, double sparsity,
    double epsilon, const std::string &importance_norm) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyAttentionHeadWandaPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity, epsilon,
        importance_norm));
  } catch (const std::exception &e) {
    std::cerr << "apply_attention_head_wanda_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_sparsegpt_pruning(
    const std::string &data, em::val calibration_batches_val, double sparsity,
    em::val n_val, em::val m_val, double percdamp, int proc_block_size) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplySparseGptPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity,
        OptInt64FromVal(n_val), OptInt64FromVal(m_val), percdamp,
        proc_block_size));
  } catch (const std::exception &e) {
    std::cerr << "apply_sparsegpt_pruning error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_wanda_pruning(const std::string &data,
                                    em::val calibration_batches_val,
                                    double sparsity, em::val n_val,
                                    em::val m_val, double epsilon,
                                    bool global_sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyWandaPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity,
        OptInt64FromVal(n_val), OptInt64FromVal(m_val), epsilon,
        global_sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_wanda_pruning error: " << e.what() << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_moe_whole_expert_pruning(
    const std::string &data, em::val calibration_batches_val,
    double sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyMoeWholeExpertPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_moe_whole_expert_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

em::val onnxsim_apply_qmoe_whole_expert_pruning(
    const std::string &data, em::val calibration_batches_val,
    double sparsity) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQMoEWholeExpertPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity));
  } catch (const std::exception &e) {
    std::cerr << "apply_qmoe_whole_expert_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// An absent (undefined/null) `num_blocks_to_drop` leaves the choice to
// `sparsity` (fraction of matched candidates); a present number is an
// explicit block count and takes priority -- mirroring
// apply_transformer_block_pruning_cpp's own contract exactly.
em::val onnxsim_apply_transformer_block_pruning(
    const std::string &data, em::val calibration_batches_val, double sparsity,
    em::val num_blocks_to_drop_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyTransformerBlockPruning(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), sparsity,
        OptInt64FromVal(num_blocks_to_drop_val)));
  } catch (const std::exception &e) {
    std::cerr << "apply_transformer_block_pruning error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// `scale_lo`/`scale_hi` are the two ends of apply_imatrix_quantization_cpp's
// own `scale_search_range` tuple, split into two numbers; `skip_names` is
// absent (undefined/null) or a string array of weight initializer names to
// leave unquantized. Every other parameter mirrors that wrapper exactly.
em::val onnxsim_apply_imatrix_quantization(
    const std::string &data, em::val calibration_batches_val, int block_size,
    int num_scale_candidates, double scale_lo, double scale_hi,
    em::val skip_names_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    std::unordered_set<std::string> skip_names;
    if (!skip_names_val.isUndefined() && !skip_names_val.isNull()) {
      for (const std::string &name :
           em::vecFromJSArray<std::string>(skip_names_val)) {
        skip_names.insert(name);
      }
    }
    return SerializeModel(ApplyImatrixQuantization(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), block_size,
        num_scale_candidates, scale_lo, scale_hi, skip_names));
  } catch (const std::exception &e) {
    std::cerr << "apply_imatrix_quantization error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// Outlier Suppression Gamma Migration (Wei et al., 2022): folds the
// SmoothQuant-style per-channel migration scale directly into every
// matched LayerNormalization's gamma/bias (adding zero runtime nodes),
// compensated by scaling every downstream MatMul/vanilla-Gemm consumer's
// weight rows by the same scale -- a lossless pre-conditioning transform
// ahead of a separate W8A8 quantizer. Same calibration-batch contract and
// executor as every other calibration-driven binding above. `alpha` is
// the migration strength (0.5 splits difficulty evenly on a log scale);
// `epsilon` floors the per-channel max-abs values and the scale itself.
// See ApplyOutlierSuppression in outlier_suppression_entry.h.
em::val onnxsim_apply_outlier_suppression(const std::string &data,
                                          em::val calibration_batches_val,
                                          double alpha, double epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyOutlierSuppression(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), alpha, epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_outlier_suppression error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// LLM.int8() (Dettmers et al., 2022): decomposes every matched
// MatMul/vanilla-Gemm node into a float32 outlier part plus a
// vector-wise INT8 part computed via MatMulInteger (per-row activation
// scales at runtime, per-output-channel weight scales offline, uint8
// activation at zero-point 128). Same calibration-batch contract and
// executor as every other calibration-driven binding above.
// `outlier_threshold` marks an input channel an outlier when its
// activation magnitude exceeds it anywhere in the calibration data;
// `epsilon` floors zero max-abs values before dividing. See ApplyLlmInt8
// in llm_int8_entry.h.
em::val onnxsim_apply_llm_int8(const std::string &data,
                               em::val calibration_batches_val,
                               double outlier_threshold, double epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyLlmInt8(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), outlier_threshold,
        epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_llm_int8 error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// SpQR (Dettmers et al., 2023): outlier-aware block-wise INT4
// quantization -- per-element outliers (by Hessian-diagonal-weighted
// sensitivity) are excluded from their own block's scale and stored as an
// exact sparse correction. Same calibration-batch contract and executor
// as every other calibration-driven binding above. See ApplySpqr in
// spqr_entry.h.
em::val onnxsim_apply_spqr(const std::string &data,
                           em::val calibration_batches_val,
                           int block_size, double outlier_fraction) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplySpqr(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), block_size,
        outlier_fraction));
  } catch (const std::exception &e) {
    std::cerr << "apply_spqr error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// PB-LLM (Shang et al., 2024): a structured mixed-precision binarizer --
// per matched layer, the `salient_ratio` fraction of input channels with
// the highest Hessian-diagonal-weighted magnitude stay INT8, every other
// channel is binarized to ~1 bit/element. Same calibration-batch contract
// and executor as every other calibration-driven binding above. See
// ApplyPbLlm in pb_llm_entry.h.
em::val onnxsim_quantize_weight_only_pb_llm(const std::string &data,
                                            em::val calibration_batches_val,
                                            double salient_ratio) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyPbLlm(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), salient_ratio));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_pb_llm error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// SqueezeLLM (Kim et al., 2023): sensitivity-weighted per-group codebook
// (a real GatherND-based graph rewrite, not folded to a single
// initializer) plus a dense-and-sparse outlier correction. Same
// calibration-batch contract and executor as every other
// calibration-driven binding above. See ApplySqueezeLlm in
// squeezellm_entry.h.
em::val onnxsim_quantize_weight_only_squeezellm(
    const std::string &data, em::val calibration_batches_val,
    int block_size, int bits, double outlier_fraction,
    int num_kmeans_iterations) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplySqueezeLlm(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), block_size, bits,
        outlier_fraction, num_kmeans_iterations));
  } catch (const std::exception &e) {
    std::cerr << "quantize_weight_only_squeezellm error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

// BiLLM (Huang et al., 2024, ICML): a genuine ~1-bit-average weight
// binarizer -- Hessian-guided salient-column selection, a two-level
// binary residual approximation for salient columns, plain flat binary
// for the rest, and OBC-style forward error compensation (reusing
// onnxsim_apply_gptq's own Cholesky-factored-inverse-Hessian mechanism).
// Same calibration-batch contract and executor as every other
// calibration-driven binding above. See ApplyBillm in billm_entry.h.
em::val onnxsim_apply_billm(const std::string &data,
                            em::val calibration_batches_val, int block_size,
                            double percdamp, int max_salient_search) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyBillm(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), block_size,
        percdamp, max_salient_search));
  } catch (const std::exception &e) {
    std::cerr << "apply_billm error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// KV-cache quantization (KIVI/KVQuant): per-channel static INT8 for a
// matched Concat(past, new, axis=seq) stream's own Key-style values,
// per-token dynamic INT8 (data-free) for Value-style ones. Same
// calibration-batch contract and executor as every other calibration-driven
// binding above; `value_output_names` is absent (undefined/null) or a
// string array of `present` output names to treat as Value-style instead of
// the default ".value"-in-name heuristic. See ApplyKvCacheQuantization in
// kv_cache_quantization_entry.h.
em::val onnxsim_quantize_kv_cache(const std::string &data,
                                  em::val calibration_batches_val,
                                  em::val value_output_names_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    std::vector<std::string> value_output_names;
    if (!value_output_names_val.isUndefined() &&
        !value_output_names_val.isNull()) {
      value_output_names = em::vecFromJSArray<std::string>(value_output_names_val);
    }
    return SerializeModel(ApplyKvCacheQuantization(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), value_output_names));
  } catch (const std::exception &e) {
    std::cerr << "quantize_kv_cache error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// OWQ (Lee, Park, Kim, Kim and Sung, 2023, AAAI 2024): rescues the top
// `outlier_fraction` OBS-salient columns of an already-
// quantize_weight_only_int4-quantized layer back to exact float32 precision
// via an additive Gather/MatMul/Add correction, reusing
// onnxsim_apply_gptq's own Cholesky-factored-inverse-Hessian mechanism --
// `quantized_model`'s own INT4 codes are never modified. Takes two models,
// the same shape as onnxsim_apply_gptq. See ApplyOwq in owq_entry.h.
em::val onnxsim_apply_owq(const std::string &float_data,
                          const std::string &quantized_data,
                          em::val calibration_batches_val,
                          double outlier_fraction, double percdamp) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyOwq(
        float_model, quantized_model, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), outlier_fraction,
        percdamp));
  } catch (const std::exception &e) {
    std::cerr << "apply_owq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// GEAR (Kang et al., 2024): low-rank-plus-sparse residual compensation
// layered on top of onnxsim_quantize_kv_cache's own static per-channel
// INT8 base quantization, applied only to a freshly-produced KV-cache
// token. Same calibration-batch contract and executor as every other
// calibration-driven binding above. See ApplyGear in gear_entry.h.
em::val onnxsim_apply_gear(const std::string &data,
                           em::val calibration_batches_val, int rank,
                           double outlier_fraction) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGear(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), rank,
        outlier_fraction));
  } catch (const std::exception &e) {
    std::cerr << "apply_gear error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// RotateKV (Su et al., 2025): fits a per-stream orthogonal rotation from a
// matched KV-cache stream's own calibration-activation covariance and
// applies it to both the stream's fresh Key and its compensating Query.
// Same calibration-batch contract and executor as every other
// calibration-driven binding above. See ApplyRotateKv in rotatekv_entry.h.
em::val onnxsim_apply_rotatekv(const std::string &data,
                               em::val calibration_batches_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyRotateKv(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val)));
  } catch (const std::exception &e) {
    std::cerr << "apply_rotatekv error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// SmoothQuant migration (Xiao et al., 2022): rescales every matched
// MatMul/vanilla-Gemm node's constant 2-D FLOAT32 weight columns by the
// per-channel migration scale and inserts a `Mul` dividing that layer's
// activation input by the same scale -- a lossless pre-conditioning
// transform ahead of a separate W8A8 quantizer. Same calibration-batch
// contract and executor as every other calibration-driven binding above.
// `alpha` is the migration strength (0.5 splits difficulty evenly on a log
// scale); `epsilon` floors the per-channel max-abs values and `s` itself.
// See ApplySmoothQuant in smoothquant_entry.h.
em::val onnxsim_apply_smoothquant(const std::string &data,
                                  em::val calibration_batches_val,
                                  double alpha, double epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplySmoothQuant(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), alpha, epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_smoothquant error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// GPTQ (Frantar et al., 2022): sequential, Hessian-compensated INT4
// rounding for every quantize_weight_only_int4-quantized MatMul/Gemm
// layer shared (by node output name) between a float model and its
// quantized counterpart, reusing that scheme's own per-block scales.
// Takes two models: the float model (whose graph inputs the calibration
// batches are keyed to, and which the executor runs) and the quantized
// model (which comes back with rewritten INT4 codes). Same
// calibration-batch contract and executor as every other
// calibration-driven binding above. `percdamp` is the Hessian damping
// factor; `proc_block_size` is GPTQ's own column-processing block size
// (not the quantization scale's block size, which is reused unchanged).
// See ApplyGptq in gptq_entry.h.
em::val onnxsim_apply_gptq(const std::string &float_data,
                           const std::string &quantized_data,
                           em::val calibration_batches_val, double percdamp,
                           int proc_block_size) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyGptq(
        float_model, quantized_model, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), percdamp,
        proc_block_size));
  } catch (const std::exception &e) {
    std::cerr << "apply_gptq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// AdaRound (Nagel et al., 2020): a rectified-sigmoid relaxation of each
// weight element's floor/ceil rounding decision, optimized by a
// hand-rolled Adam loop to minimize a layer's own reconstruction error
// against real calibration activations. Same two-model calibration-batch
// contract and executor as onnxsim_apply_gptq's own binding above
// (candidates are processed independently, so `executor` is invoked
// once, up front, the same as onnxsim_apply_gptq's own). `beta_start`/
// `beta_end` are the two ends of apply_adaround's own `beta_range`
// tuple, split into separate parameters here since this binding layer
// has no tuple type. See ApplyAdaround in adaround_entry.h, including
// its own accepted numerical scope note (an iterative optimization, not
// a closed-form computation).
em::val onnxsim_apply_adaround(const std::string &float_data,
                               const std::string &quantized_data,
                               em::val calibration_batches_val,
                               int num_iterations, double learning_rate,
                               double reg_param, double warm_start,
                               double beta_start, double beta_end) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyAdaround(
        float_model, quantized_model, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), num_iterations,
        learning_rate, reg_param, warm_start, beta_start, beta_end));
  } catch (const std::exception &e) {
    std::cerr << "apply_adaround error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// Qronos: a sequential, whole-model generalization of onnxsim_apply_gptq
// that additionally accounts for the error already baked into a layer's
// activations because upstream layers were quantized first, not just
// this layer's own rounding -- processes layers in the float model's
// own node order, re-probing the progressively-corrected quantized
// model before each subsequent layer. Same two-model calibration-batch
// contract and executor as onnxsim_apply_gptq's own binding above
// (`percdamp`/`proc_block_size` mean the same thing). See ApplyQronos in
// qronos_entry.h.
em::val onnxsim_apply_qronos(const std::string &float_data,
                             const std::string &quantized_data,
                             em::val calibration_batches_val, double percdamp,
                             int proc_block_size) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQronos(
        float_model, quantized_model, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), percdamp,
        proc_block_size));
  } catch (const std::exception &e) {
    std::cerr << "apply_qronos error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// TesseraQ: "Progressive Adaptive Rounding" (PAR) -- an AdaRound-style
// rectified-sigmoid rounding relaxation, optimized by a hand-rolled Adam
// loop jointly with each weight block's own dequantization scale (in
// log-space), with a coarse-to-fine element-by-element hardening
// schedule across `par_rounds` rounds. Same two-model calibration-batch
// contract and executor as onnxsim_apply_gptq's own binding above
// (candidates are processed independently, so `executor` is invoked
// once, up front, the same as onnxsim_apply_gptq's own). `beta_start`/
// `beta_end` are the two ends of apply_tesseraq's own `beta_range`
// tuple, split into separate parameters here since this binding layer
// has no tuple type. See ApplyTesseraq in tesseraq_entry.h, including
// its own accepted numerical scope note (an iterative optimization, not
// a closed-form computation).
em::val onnxsim_apply_tesseraq(const std::string &float_data,
                               const std::string &quantized_data,
                               em::val calibration_batches_val, int num_bits,
                               int num_iterations, int par_rounds,
                               double learning_rate,
                               double scale_learning_rate, double reg_param,
                               double warm_start, double beta_start,
                               double beta_end) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyTesseraq(
        float_model, quantized_model, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), num_bits,
        num_iterations, par_rounds, learning_rate, scale_learning_rate,
        reg_param, warm_start, beta_start, beta_end));
  } catch (const std::exception &e) {
    std::cerr << "apply_tesseraq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// AWQ (Lin et al., 2023): grid-searched per-channel weight rescaling for
// every quantize_weight_only_int4-quantized MatMul/Gemm layer shared (by
// node output name) between a float model and its quantized counterpart,
// re-quantizing from scratch at each grid point. Takes two models like
// onnxsim_apply_gptq above (float model first) and honors the same
// calibration-batch contract and executor. `num_alpha_steps` is the grid
// density over [0, 1] inclusive. See ApplyAwq in awq_entry.h.
em::val onnxsim_apply_awq(const std::string &float_data,
                          const std::string &quantized_data,
                          em::val calibration_batches_val,
                          int num_alpha_steps) {
  onnx::ModelProto float_model;
  if (!float_model.ParseFromArray(float_data.data(), float_data.size())) {
    std::cerr << "Parse failed (float model)" << std::endl;
    return em::val::null();
  }
  onnx::ModelProto quantized_model;
  if (!quantized_model.ParseFromArray(quantized_data.data(),
                                      quantized_data.size())) {
    std::cerr << "Parse failed (quantized model)" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyAwq(
        float_model, quantized_model, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), num_alpha_steps));
  } catch (const std::exception &e) {
    std::cerr << "apply_awq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// QuaRot+GPTQ (Ashkboos et al., 2024): the real QuaRot paper's optional,
// tighter weight quantizer -- rotates every matched MatMul/vanilla-Gemm
// node's activation by a fresh per-layer random orthogonal matrix and
// quantizes both operands to INT4, the weight via onnxsim_apply_gptq's own
// Hessian-compensated column algorithm (evaluated in the rotated
// activation space) instead of round-to-nearest. Unlike
// onnxsim_apply_gptq/onnxsim_apply_awq above, takes a single model (this
// pass derives its own rotation and quantizes from scratch, like
// onnxsim_apply_quarot's own data-free binding above); same
// calibration-batch contract and executor as every other calibration-driven
// binding. `seed` arrives as a JS number (double) and is narrowed to the
// uint64 it feeds, the same way onnxsim_apply_quarot's own does;
// `block_size`/`epsilon` mirror onnxsim_apply_quarot's own parameters,
// `percdamp`/`proc_block_size` mirror onnxsim_apply_gptq's own. See
// ApplyQuarotGptq in quarot_gptq_entry.h.
em::val onnxsim_apply_quarot_gptq(const std::string &data,
                                  em::val calibration_batches_val,
                                  double seed, int block_size,
                                  double percdamp, int proc_block_size,
                                  float epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyQuarotGptq(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val),
        static_cast<uint64_t>(seed), block_size, percdamp, proc_block_size,
        epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_quarot_gptq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// GPTVQ (Van Baalen et al., 2024): a genuine combination of
// onnxsim_apply_gptq's own sequential, Hessian-compensated correction
// with a k-means-fit vector codebook -- small groups of consecutive
// input-channel columns of every matched MatMul/vanilla-Gemm node's
// constant 2-D FLOAT32 weight are jointly quantized against the
// codebook, then each group's resulting per-column residual is
// propagated into every not-yet-quantized column exactly like
// onnxsim_apply_gptq's own per-column correction. Rewires only the
// matched node's weight input (Gather+Reshape[+Transpose]); the node
// itself, including any bias, is left otherwise unchanged. Same
// calibration-batch contract and executor as every other
// calibration-driven binding above, and the same `skip_names` (JS
// array of strings, possibly undefined/null) crossing convention as
// onnxsim_apply_imatrix_quantization's own binding. `seed` arrives as a
// JS number (double) and is narrowed to the uint64 it feeds, the same
// way onnxsim_apply_quarot_gptq's own does. See ApplyGptvq in
// gptvq_entry.h.
em::val onnxsim_apply_gptvq(const std::string &data,
                             em::val calibration_batches_val, double seed,
                             int vector_dim, int num_centroids,
                             int num_iterations, double percdamp,
                             em::val skip_names_val) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    std::unordered_set<std::string> skip_names;
    if (!skip_names_val.isUndefined() && !skip_names_val.isNull()) {
      for (const std::string &name :
           em::vecFromJSArray<std::string>(skip_names_val)) {
        skip_names.insert(name);
      }
    }
    return SerializeModel(ApplyGptvq(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val),
        static_cast<uint64_t>(seed), vector_dim, num_centroids,
        num_iterations, percdamp, skip_names));
  } catch (const std::exception &e) {
    std::cerr << "apply_gptvq error: " << e.what() << std::endl;
    return em::val::null();
  }
}

// Outlier Suppression+ (Wei et al., 2023): per-channel shifting ahead of
// SmoothQuant's own per-channel scale -- recenters each activation
// channel around zero, rescales the weight columns in place, inserts a
// `Sub`+`Mul` pair before the layer and an `Add` after it restoring the
// shift's constant contribution -- a lossless pre-conditioning transform
// ahead of a separate W8A8 quantizer. Same calibration-batch contract
// and executor as every other calibration-driven binding above. `alpha`
// is the scaling step's migration strength (applied to the *shifted*
// activation's range); `epsilon` floors the per-channel max-abs values
// and the scale itself. See ApplyOutlierSuppressionPlus in
// outlier_suppression_plus_entry.h.
em::val onnxsim_apply_outlier_suppression_plus(
    const std::string &data, em::val calibration_batches_val, double alpha,
    double epsilon) {
  onnx::ModelProto xmodel;
  if (!xmodel.ParseFromArray(data.data(), data.size())) {
    std::cerr << "Parse failed" << std::endl;
    return em::val::null();
  }
  try {
    return SerializeModel(ApplyOutlierSuppressionPlus(
        xmodel, CalibrationExecutor(),
        ParseCalibrationBatches(calibration_batches_val), alpha, epsilon));
  } catch (const std::exception &e) {
    std::cerr << "apply_outlier_suppression_plus error: " << e.what()
              << std::endl;
    return em::val::null();
  }
}

EMSCRIPTEN_BINDINGS(module) {
  function("onnxsimplify_export", &onnxsimplify_export);
  function("onnxsim_annotate_model_info", &onnxsim_annotate_model_info);
  function("onnxsim_export_safetensors", &onnxsim_export_safetensors);
  function("onnxsim_export_gguf", &onnxsim_export_gguf);
  function("onnxsim_import_safetensors", &onnxsim_import_safetensors);
  function("onnxsim_import_gguf", &onnxsim_import_gguf);
  function("onnxoptimizer_optimize", &onnxoptimizer_optimize);
  function("onnxoptimizer_optimize_fixed", &onnxoptimizer_optimize_fixed);
  em::function("onnxoptimizer_passes", &onnxoptimizer_passes);
  em::function("onnxoptimizer_fuse_elimination_passes",
               &onnxoptimizer_fuse_elimination_passes);
  em::function("onnxsim_needs_ort_web", &onnxsim_needs_ort_web);
  // Version reporting, text-graph parsing, TensorProto parsing, and the
  // single-pass debugging entry points (see the definitions above).
  em::function("onnxsim_versions", &onnxsim_versions);
  em::function("onnxsim_parse_graph", &onnxsim_parse_graph);
  em::function("onnxsim_extract_graph", &onnxsim_extract_graph);
  em::function("onnxsim_inline_functions", &onnxsim_inline_functions);
  em::function("onnxsim_parse_tensor", &onnxsim_parse_tensor);
  em::function("onnxsim_infer_shapes", &onnxsim_infer_shapes);
  em::function("onnxsim_data_propagation", &onnxsim_data_propagation);
  em::function("onnxsim_fold_constant", &onnxsim_fold_constant);

  // Calibration-free quantization methods (no activation ranges needed).
  function("onnxsim_quantize_dynamic", &onnxsim_quantize_dynamic);
  function("onnxsim_quantize_ternary", &onnxsim_quantize_ternary);
  function("onnxsim_quantize_weight_only", &onnxsim_quantize_weight_only);
  function("onnxsim_quantize_weight_only_int4",
           &onnxsim_quantize_weight_only_int4);
  function("onnxsim_quantize_fp16", &onnxsim_quantize_fp16);
  function("onnxsim_quantize_bf16", &onnxsim_quantize_bf16);
  function("onnxsim_quantize_fp8", &onnxsim_quantize_fp8);
  // Static, calibration-free INT8-quantization risk pre-check (see its own
  // doc comment above).
  em::function("onnxsim_estimate_quantization_drop",
               &onnxsim_estimate_quantization_drop);
  // Calibration-based quantization: list_* discovers which tensors to
  // calibrate, quantize_static/quantize_qoperator take the calibrated
  // ranges back as parallel [name] / [min0, max0, min1, max1, ...] arrays.
  em::function("onnxsim_list_quantizable_activations",
               &onnxsim_list_quantizable_activations);
  em::function("onnxsim_list_qoperator_quantizable_outputs",
               &onnxsim_list_qoperator_quantizable_outputs);
  function("onnxsim_add_graph_outputs", &onnxsim_add_graph_outputs);
  function("onnxsim_quantize_static", &onnxsim_quantize_static);
  function("onnxsim_quantize_qoperator", &onnxsim_quantize_qoperator);

  // Bias correction: list_correctable_outputs discovers which Conv/Gemm/
  // MatMul/Resize outputs are eligible, apply_bias_corrections splices the
  // page's own already-measured per-channel/per-position corrections in
  // (see the doc comments above, and onnxsim/bias_correction_entry.h).
  em::function("onnxsim_list_correctable_outputs",
               &onnxsim_list_correctable_outputs);
  function("onnxsim_apply_bias_corrections", &onnxsim_apply_bias_corrections);

  // Data-free quantization / pruning passes ported to C++ (see their own doc
  // comments above, and onnxsim.h): model bytes in, model bytes out, no
  // executor and no calibration data.
  function("onnxsim_cross_layer_equalize", &onnxsim_cross_layer_equalize);
  function("onnxsim_quantize_dynamic_matmul_integer_to_float",
           &onnxsim_quantize_dynamic_matmul_integer_to_float);
  function("onnxsim_quantize_attention_dynamic",
           &onnxsim_quantize_attention_dynamic);
  function("onnxsim_quantize_weight_only_int16",
           &onnxsim_quantize_weight_only_int16);
  function("onnxsim_quantize_weight_only_int8_block",
           &onnxsim_quantize_weight_only_int8_block);
  function("onnxsim_quantize_weight_only_mxfp4",
           &onnxsim_quantize_weight_only_mxfp4);
  function("onnxsim_quantize_weight_only_matmul_nbits",
           &onnxsim_quantize_weight_only_matmul_nbits);
  function("onnxsim_apply_double_quantization",
           &onnxsim_apply_double_quantization);
  function("onnxsim_apply_any_precision_llm", &onnxsim_apply_any_precision_llm);
  function("onnxsim_apply_quarot", &onnxsim_apply_quarot);
  function("onnxsim_apply_iq4_nl", &onnxsim_apply_iq4_nl);
  function("onnxsim_apply_gguf_q4_0", &onnxsim_apply_gguf_q4_0);
  function("onnxsim_apply_gguf_q4_1", &onnxsim_apply_gguf_q4_1);
  function("onnxsim_apply_gguf_q5_0", &onnxsim_apply_gguf_q5_0);
  function("onnxsim_apply_gguf_q5_1", &onnxsim_apply_gguf_q5_1);
  function("onnxsim_apply_gguf_q8_0", &onnxsim_apply_gguf_q8_0);
  function("onnxsim_apply_gguf_q2_k", &onnxsim_apply_gguf_q2_k);
  function("onnxsim_apply_gguf_q3_k", &onnxsim_apply_gguf_q3_k);
  function("onnxsim_apply_gguf_q4_k", &onnxsim_apply_gguf_q4_k);
  function("onnxsim_apply_gguf_q5_k", &onnxsim_apply_gguf_q5_k);
  function("onnxsim_apply_gguf_ternary", &onnxsim_apply_gguf_ternary);
  function("onnxsim_apply_fp6_llm", &onnxsim_apply_fp6_llm);
  function("onnxsim_apply_gguf_q6_k", &onnxsim_apply_gguf_q6_k);
  function("onnxsim_apply_leptoquant", &onnxsim_apply_leptoquant);
  function("onnxsim_quantize_weight_only_nf4",
           &onnxsim_quantize_weight_only_nf4);
  function("onnxsim_quantize_weight_only_if4",
           &onnxsim_quantize_weight_only_if4);
  function("onnxsim_quantize_weight_only_nvfp4",
           &onnxsim_quantize_weight_only_nvfp4);
  function("onnxsim_apply_deepseek_fp8", &onnxsim_apply_deepseek_fp8);
  function("onnxsim_apply_kmeans_quantization",
           &onnxsim_apply_kmeans_quantization);
  function("onnxsim_apply_hqq", &onnxsim_apply_hqq);
  function("onnxsim_apply_ibert_gelu", &onnxsim_apply_ibert_gelu);
  function("onnxsim_apply_ibert_softmax", &onnxsim_apply_ibert_softmax);
  function("onnxsim_apply_adpq", &onnxsim_apply_adpq);
  function("onnxsim_apply_icquant", &onnxsim_apply_icquant);
  function("onnxsim_apply_olive", &onnxsim_apply_olive);
  function("onnxsim_apply_aqlm", &onnxsim_apply_aqlm);
  function("onnxsim_apply_drop_by_drop", &onnxsim_apply_drop_by_drop);
  function("onnxsim_apply_lo_bcq", &onnxsim_apply_lo_bcq);
  function("onnxsim_apply_quip_sharp", &onnxsim_apply_quip_sharp);
  function("onnxsim_apply_attention_quantization",
           &onnxsim_apply_attention_quantization);
  function("onnxsim_apply_zeroquant", &onnxsim_apply_zeroquant);
  function("onnxsim_apply_intactkv", &onnxsim_apply_intactkv);
  function("onnxsim_apply_kbvq_moe", &onnxsim_apply_kbvq_moe);
  function("onnxsim_quantize_weight_only_llm_fp4",
           &onnxsim_quantize_weight_only_llm_fp4);
  function("onnxsim_apply_qoq", &onnxsim_apply_qoq);
  function("onnxsim_apply_dsq", &onnxsim_apply_dsq);
  function("onnxsim_apply_daq", &onnxsim_apply_daq);
  function("onnxsim_apply_low_rank_compensation",
           &onnxsim_apply_low_rank_compensation);
  function("onnxsim_quantize_embedding_binary",
           &onnxsim_quantize_embedding_binary);
  function("onnxsim_prune_magnitude", &onnxsim_prune_magnitude);
  function("onnxsim_apply_structured_pruning",
           &onnxsim_apply_structured_pruning);
  function("onnxsim_apply_attention_head_pruning",
           &onnxsim_apply_attention_head_pruning);
  function("onnxsim_apply_moe_expert_channel_pruning",
           &onnxsim_apply_moe_expert_channel_pruning);
  function("onnxsim_apply_qmoe_expert_channel_pruning",
           &onnxsim_apply_qmoe_expert_channel_pruning);
  function("onnxsim_apply_embedding_vocab_pruning",
           &onnxsim_apply_embedding_vocab_pruning);
  function("onnxsim_apply_embedding_vocab_magnitude_pruning",
           &onnxsim_apply_embedding_vocab_magnitude_pruning);

  // Calibration-driven passes (see their own doc comments above): each takes
  // calibration batches -- [{ name: { dtype, dims, data } }, ...] with raw
  // little-endian bytes -- runs probe sub-models through the module's
  // executor, and applies the rewrite. Like onnxsim_fold_constant, these may
  // return a Promise in the ORT-web build and the caller must await it when
  // it is one.
  function("onnxsim_apply_structured_wanda_pruning",
           &onnxsim_apply_structured_wanda_pruning);
  function("onnxsim_apply_attention_head_wanda_pruning",
           &onnxsim_apply_attention_head_wanda_pruning);
  function("onnxsim_apply_sparsegpt_pruning", &onnxsim_apply_sparsegpt_pruning);
  function("onnxsim_apply_wanda_pruning", &onnxsim_apply_wanda_pruning);
  function("onnxsim_apply_moe_whole_expert_pruning",
           &onnxsim_apply_moe_whole_expert_pruning);
  function("onnxsim_apply_qmoe_whole_expert_pruning",
           &onnxsim_apply_qmoe_whole_expert_pruning);
  function("onnxsim_apply_transformer_block_pruning",
           &onnxsim_apply_transformer_block_pruning);
  function("onnxsim_apply_imatrix_quantization",
           &onnxsim_apply_imatrix_quantization);
  function("onnxsim_apply_outlier_suppression",
           &onnxsim_apply_outlier_suppression);
  function("onnxsim_apply_outlier_suppression_plus",
           &onnxsim_apply_outlier_suppression_plus);
  function("onnxsim_apply_llm_int8", &onnxsim_apply_llm_int8);
  function("onnxsim_apply_spqr", &onnxsim_apply_spqr);
  function("onnxsim_quantize_weight_only_pb_llm",
           &onnxsim_quantize_weight_only_pb_llm);
  function("onnxsim_quantize_weight_only_squeezellm",
           &onnxsim_quantize_weight_only_squeezellm);
  function("onnxsim_apply_billm", &onnxsim_apply_billm);
  function("onnxsim_quantize_kv_cache", &onnxsim_quantize_kv_cache);
  function("onnxsim_apply_owq", &onnxsim_apply_owq);
  function("onnxsim_apply_gear", &onnxsim_apply_gear);
  function("onnxsim_apply_rotatekv", &onnxsim_apply_rotatekv);
  function("onnxsim_apply_gptq", &onnxsim_apply_gptq);
  function("onnxsim_apply_adaround", &onnxsim_apply_adaround);
  function("onnxsim_apply_qronos", &onnxsim_apply_qronos);
  function("onnxsim_apply_tesseraq", &onnxsim_apply_tesseraq);
  function("onnxsim_apply_awq", &onnxsim_apply_awq);
  function("onnxsim_apply_quarot_gptq", &onnxsim_apply_quarot_gptq);
  function("onnxsim_apply_gptvq", &onnxsim_apply_gptvq);
  function("onnxsim_apply_smoothquant", &onnxsim_apply_smoothquant);

  // Block-wise QAT: build one block's step graph, run the loop in JS on
  // onnxruntime-web, write the trained state back (see the doc comments
  // above, and onnxsim/qat_entry.h for the flow they implement).
  function("onnxsim_qat_build_step_graph", &onnxsim_qat_build_step_graph);
  function("onnxsim_qat_write_back", &onnxsim_qat_write_back);
  // Qualified, unlike its two neighbours, and not by preference: the bare
  // `function` the registrations above use is found by argument-dependent
  // lookup, which needs an argument type from emscripten's own namespace to
  // associate it. Every other binding here takes or returns em::val and so
  // drags it in; this one is bool(int), which associates nothing.
  em::function("onnxsim_qat_release_plan", &onnxsim_qat_release_plan);

  // LoRA/QLoRA fine-tuning: inject a trainable low-rank branch, build one
  // block's step graph, run the loop in JS on onnxruntime-web, write the
  // trained state back, release the plan (see the doc comments above, and
  // onnxsim/lora_entry.h for the flow they implement). Unlike QAT, LoRA also
  // needs its own injection binding -- see onnxsim_lora_inject's own comment
  // for why.
  function("onnxsim_lora_inject", &onnxsim_lora_inject);
  function("onnxsim_lora_build_step_graph", &onnxsim_lora_build_step_graph);
  function("onnxsim_lora_write_back", &onnxsim_lora_write_back);
  // Qualified for the same reason onnxsim_qat_release_plan is -- see that
  // registration's own comment.
  em::function("onnxsim_lora_release_plan", &onnxsim_lora_release_plan);

  em::register_vector<std::string>("string_list");
}
