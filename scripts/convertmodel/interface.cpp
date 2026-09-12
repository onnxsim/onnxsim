#include "bias_correction_entry.h"
#include "lora_entry.h"
#include "model_info.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
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

// In the ORT-web build (ONNXSIM_WASM_ORT_WEB) onnxsim is compiled without
// ONNXSIM_HAS_ORT -- no ONNX Runtime is linked in -- and constant folding is
// delegated to the page's onnxruntime-web via this executor instead.
#ifdef ONNXSIM_WASM_ORT_WEB
#include "js_model_executor.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

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
#ifdef ONNXSIM_WASM_ORT_WEB
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
#ifdef ONNXSIM_WASM_ORT_WEB
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
    entry.shape = em::vecFromJSArray<int64_t>(item["shape"]);
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
