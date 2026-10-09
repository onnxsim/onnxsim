// AXCL-backed worker for constrained-device operator tests.
//
// The request operation is the path of an .axmodel.  This intentionally keeps
// the first adapter simple: a host-side ORT EP can compile/cache a claimed
// subgraph to an axmodel and use this worker as its Execute() transport.  The
// worker accepts and returns float32 tensors, which matches the existing AX
// operator-test corpus and avoids pulling a serializer onto the device.
//
// Tensors carry their own ONNX dtype: float32 in `data`, any other dtype
// (fp16/bf16/int8/...) as raw little-endian bytes in `raw_data`. That is what
// lets a compiled LLM layer (bf16 hidden state, KV cache and mask) run here.
//
// Loaded models are kept between requests (LRU, --max-loaded) unless
// --no-cache is given; `io_info` describes a model's tensors, `unload` drops
// it, and `run` is the path form for paths longer than the op field allows.
// See the "AXCL worker" section of README.md.
#include "remote_transport.h"

#include <axcl.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace onnx_remote;
namespace fs = std::filesystem;

static fs::path g_cache_dir;

static bool valid_artifact_id(const std::string& id) {
  if (id.empty() || id.size() > kMaxArtifactIdBytes) return false;
  for (char c : id) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
      return false;
  }
  return true;
}

static bool materialize_artifact(const Request& request, fs::path& path,
                                 std::string& error) {
  if (!valid_artifact_id(request.artifact_id)) {
    error = "invalid AXCL artifact id";
    return false;
  }
  if (g_cache_dir.empty()) {
    error = "AXCL artifact cache is not configured";
    return false;
  }
  std::error_code ec;
  fs::create_directories(g_cache_dir, ec);
  if (ec) {
    error = "cannot create AXCL artifact cache: " + ec.message();
    return false;
  }
  path = g_cache_dir / (request.artifact_id + ".axmodel");
  if (!request.artifact.empty()) {
    if (request.artifact.size() > kMaxArtifactBytes) {
      error = "AXCL artifact exceeds transport limit";
      return false;
    }
    const fs::path temporary = path.string() + ".tmp." +
                               std::to_string(static_cast<long long>(::getpid()));
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      error = "cannot write AXCL artifact cache";
      return false;
    }
    output.write(reinterpret_cast<const char*>(request.artifact.data()),
                 static_cast<std::streamsize>(request.artifact.size()));
    output.close();
    if (!output) {
      fs::remove(temporary);
      error = "cannot write AXCL artifact cache";
      return false;
    }
    fs::rename(temporary, path, ec);
    if (ec) {
      fs::remove(temporary);
      error = "cannot publish AXCL artifact: " + ec.message();
      return false;
    }
  }
  if (!fs::is_regular_file(path, ec) || ec) {
    error = "AXCL artifact is not cached: " + request.artifact_id;
    return false;
  }
  return true;
}

static uint64_t elapsed_us(const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

// AXCL engine dtype (axclrtEngineDataType) -> ONNX TensorProto.DataType, so a
// tensor carries its real element type over the wire instead of float32 only.
// Engine types with no ONNX counterpart (int4, fp8, fp4) are rejected.
static bool onnx_dtype_of_axcl(uint32_t axcl, uint8_t& onnx) {
  switch (axcl) {
    // llm_build layers report their fp16 hidden state / KV cache / output as
    // NONE rather than FLOAT16 (seen on llama_p64_l0_together.axmodel), so NONE
    // is read as FLOAT16. Byte sizes are still checked against the engine.
    case 0:  onnx = 10; return true;   // NONE -> FP16 (llm_build convention)
    case 3:  onnx = 3;  return true;   // INT8
    case 4:  onnx = 2;  return true;   // UINT8
    case 5:  onnx = 5;  return true;   // INT16
    case 6:  onnx = 4;  return true;   // UINT16
    case 7:  onnx = 6;  return true;   // INT32
    case 8:  onnx = 12; return true;   // UINT32
    case 9:  onnx = 7;  return true;   // INT64
    case 10: onnx = 13; return true;   // UINT64
    case 13: onnx = 10; return true;   // FP16
    case 14: onnx = 16; return true;   // BF16
    case 15: onnx = 1;  return true;   // FP32
    case 16: onnx = 11; return true;   // FP64
    default: return false;
  }
}

// ---- loaded models -------------------------------------------------------
//
// One LoadedModel owns everything a run needs on the card: the engine model,
// its context, the IO description, the IO object and one device buffer per
// input and output (bound to the IO object once, at load). With the cache on
// (the default) it is kept between requests, so a host-side LLM decode loop
// that calls 31 models per token pays for 31 loads once, not per token.

struct IoSpec {
  std::string name;
  uint32_t axcl_dtype = 0;
  uint8_t onnx_dtype = 0;       // 0 when the engine type has no ONNX counterpart
  std::vector<int64_t> shape;
  uint64_t bytes = 0;
};

struct LoadedModel {
  std::string key;              // the path the request named (or the artifact's cache path)
  uint64_t model = 0, context = 0;
  axclrtEngineIOInfo info = nullptr;
  axclrtEngineIO io = nullptr;
  std::vector<void*> inputs, outputs;       // device buffers
  std::vector<IoSpec> in, out;
  // Identity of the file the model was loaded from; a changed file is reloaded.
  int64_t mtime_ns = 0;
  uint64_t file_size = 0;
};

static bool g_use_cache = true;
static size_t g_max_loaded = 64;
// Most recently used first.
static std::list<LoadedModel> g_models;
static volatile std::sig_atomic_t g_stop = 0;

static void on_stop_signal(int) { g_stop = 1; }

static bool file_identity(const std::string& path, int64_t& mtime_ns, uint64_t& size) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
  mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000ll +
             static_cast<int64_t>(st.st_mtim.tv_nsec);
  size = static_cast<uint64_t>(st.st_size);
  return true;
}

static void free_model(LoadedModel& m) {
  for (void* p : m.inputs) if (p) axclrtFree(p);
  for (void* p : m.outputs) if (p) axclrtFree(p);
  m.inputs.clear(); m.outputs.clear();
  if (m.io) axclrtEngineDestroyIO(m.io);
  if (m.info) axclrtEngineDestroyIOInfo(m.info);
  if (m.model) axclrtEngineUnload(m.model);
  m.io = nullptr; m.info = nullptr; m.model = 0; m.context = 0;
}

static void fill_shape(const axclrtEngineIODims& dims, std::vector<int64_t>& shape) {
  shape.clear();
  for (int k = 0; k < dims.dimCount; ++k) shape.push_back(dims.dims[k]);
}

// Loads `path` and allocates/binds its device buffers. On failure everything
// acquired so far is released and `error` holds the response message.
static bool load_model(const std::string& path, LoadedModel& m, std::string& error) {
  m.key = path;
  file_identity(path, m.mtime_ns, m.file_size);
  if (path.empty() || axclrtEngineLoadFromFile(path.c_str(), &m.model) ||
      axclrtEngineCreateContext(m.model, &m.context) ||
      axclrtEngineGetIOInfo(m.model, &m.info) ||
      axclrtEngineCreateIO(m.info, &m.io)) {
    free_model(m);
    error = "AXCL model setup failed: " + path;
    return false;
  }
  const uint32_t n_in = axclrtEngineGetNumInputs(m.info);
  const uint32_t n_out = axclrtEngineGetNumOutputs(m.info);
  m.inputs.assign(n_in, nullptr); m.in.resize(n_in);
  m.outputs.assign(n_out, nullptr); m.out.resize(n_out);
  for (uint32_t i = 0; i < n_in; ++i) {
    IoSpec& s = m.in[i];
    axclrtEngineDataType type{}; axclrtEngineIODims dims{};
    axclrtEngineGetInputDataType(m.info, i, &type);
    axclrtEngineGetInputDims(m.info, 0, i, &dims);
    const char* name = axclrtEngineGetInputNameByIndex(m.info, i);
    s.name = name ? name : "";
    s.axcl_dtype = static_cast<uint32_t>(type);
    if (!onnx_dtype_of_axcl(s.axcl_dtype, s.onnx_dtype)) s.onnx_dtype = 0;
    fill_shape(dims, s.shape);
    s.bytes = axclrtEngineGetInputSizeByIndex(m.info, 0, i);
    if (axclrtMalloc(&m.inputs[i], s.bytes, AXCL_MEM_MALLOC_NORMAL_ONLY) ||
        axclrtEngineSetInputBufferByIndex(m.io, i, m.inputs[i], s.bytes)) {
      free_model(m);
      error = "AXCL input buffer setup failed";
      return false;
    }
  }
  for (uint32_t i = 0; i < n_out; ++i) {
    IoSpec& s = m.out[i];
    axclrtEngineDataType type{}; axclrtEngineIODims dims{};
    axclrtEngineGetOutputDataType(m.info, i, &type);
    axclrtEngineGetOutputDims(m.info, 0, i, &dims);
    const char* name = axclrtEngineGetOutputNameByIndex(m.info, i);
    s.name = name ? name : "";
    s.axcl_dtype = static_cast<uint32_t>(type);
    if (!onnx_dtype_of_axcl(s.axcl_dtype, s.onnx_dtype)) s.onnx_dtype = 0;
    fill_shape(dims, s.shape);
    s.bytes = axclrtEngineGetOutputSizeByIndex(m.info, 0, i);
    if (axclrtMalloc(&m.outputs[i], s.bytes, AXCL_MEM_MALLOC_NORMAL_ONLY) ||
        axclrtEngineSetOutputBufferByIndex(m.io, i, m.outputs[i], s.bytes)) {
      free_model(m);
      error = "AXCL output buffer setup failed";
      return false;
    }
  }
  return true;
}

static std::list<LoadedModel>::iterator find_loaded(const std::string& key) {
  for (auto it = g_models.begin(); it != g_models.end(); ++it)
    if (it->key == key) return it;
  return g_models.end();
}

static void evict(std::list<LoadedModel>::iterator it) {
  free_model(*it);
  g_models.erase(it);
}

static size_t unload_all() {
  const size_t n = g_models.size();
  while (!g_models.empty()) evict(g_models.begin());
  return n;
}

// Returns the cached model for `path`, loading it if it is absent or if the
// file changed since it was loaded. `loaded` reports whether this call loaded.
static LoadedModel* acquire_cached(const std::string& path, bool& loaded,
                                   std::string& error) {
  loaded = false;
  auto it = find_loaded(path);
  if (it != g_models.end()) {
    int64_t mtime_ns = 0; uint64_t size = 0;
    if (file_identity(path, mtime_ns, size) && mtime_ns == it->mtime_ns &&
        size == it->file_size) {
      g_models.splice(g_models.begin(), g_models, it);
      return &g_models.front();
    }
    evict(it);  // replaced or removed on disk: never run the stale model
  }
  while (g_models.size() >= g_max_loaded) evict(std::prev(g_models.end()));
  LoadedModel fresh;
  bool ok = load_model(path, fresh, error);
  // A load can fail because the card is out of memory for one more model:
  // retry after giving back the least recently used ones. A path that is not
  // a file is not worth emptying the cache for.
  int64_t mtime_ns = 0; uint64_t size = 0;
  while (!ok && !g_models.empty() && file_identity(path, mtime_ns, size)) {
    evict(std::prev(g_models.end()));
    fresh = LoadedModel();
    ok = load_model(path, fresh, error);
  }
  if (!ok) return nullptr;
  g_models.push_front(std::move(fresh));
  loaded = true;
  return &g_models.front();
}

// Runs one request against a loaded model. Returns false (response.error set)
// on failure; `device_failed` says the failure came from the card rather than
// from a malformed request, so a cached model should not be kept.
static bool run_loaded(LoadedModel& m, const Request& request, Response& response,
                       const std::chrono::steady_clock::time_point& started,
                       bool& device_failed) {
  device_failed = false;
  auto add_profile = [&](const char* name, uint64_t begin, uint64_t duration,
                         const char* detail) {
    if (request.profiling != ProfilingLevel::Off) {
      response.profile.push_back(ProfileEvent{name, "axcl", begin, duration, detail});
    }
  };
  auto fail = [&](const std::string& message, bool device) {
    response.ok = false; response.error = message; response.outputs.clear();
    device_failed = device;
    return false;
  };
  const uint32_t n_in = static_cast<uint32_t>(m.in.size());
  const uint32_t n_out = static_cast<uint32_t>(m.out.size());
  if (n_in != request.inputs.size()) return fail("AXCL input count mismatch", false);

  const uint64_t upload_begin = elapsed_us(started);
  for (uint32_t i = 0; i < n_in; ++i) {
    // Float32 keeps its historical home in `data`; every other dtype travels
    // as its little-endian payload in `raw_data`.
    const Tensor& in = request.inputs[i];
    const uint8_t expected = m.in[i].onnx_dtype;
    if (expected == 0 || in.dtype != expected)
      return fail("AXCL input " + std::to_string(i) + " dtype mismatch: model wants ONNX dtype " +
                  std::to_string(expected) + ", request sent " + std::to_string(in.dtype), false);
    const void* src = in.dtype == 1 ? static_cast<const void*>(in.data.data())
                                    : static_cast<const void*>(in.raw_data.data());
    const uint64_t src_bytes = in.dtype == 1 ? in.data.size() * sizeof(float) : in.raw_data.size();
    if (m.in[i].bytes != src_bytes)
      return fail("AXCL input " + std::to_string(i) + " must have the model's exact size", false);
    if (axclrtMemcpy(m.inputs[i], src, m.in[i].bytes, AXCL_MEMCPY_HOST_TO_DEVICE))
      return fail("AXCL input upload failed", true);
  }
  if (request.profiling == ProfilingLevel::Detailed) {
    add_profile("axcl_upload", upload_begin, elapsed_us(started) - upload_begin,
                "host-to-device inputs");
  }
  response.outputs.resize(n_out);
  for (uint32_t i = 0; i < n_out; ++i) {
    Tensor& out = response.outputs[i];
    out.dtype = m.out[i].onnx_dtype;
    if (out.dtype == 0 || dtype_bytes(out.dtype) == 0)
      return fail("AXCL output " + std::to_string(i) + " has an unsupported dtype", false);
    out.shape = m.out[i].shape;
    uint64_t elements = 1;
    for (int64_t d : out.shape) elements *= static_cast<uint64_t>(d);
    if (elements * dtype_bytes(out.dtype) != m.out[i].bytes)
      return fail("AXCL output shape/size mismatch", false);
    if (out.dtype == 1) out.data.resize(static_cast<size_t>(elements));
    else out.raw_data.resize(static_cast<size_t>(m.out[i].bytes));
  }
  const uint64_t execute_begin = elapsed_us(started);
  if (axclrtEngineExecute(m.model, m.context, 0, m.io)) return fail("AXCL execute failed", true);
  add_profile("axcl_execute", execute_begin,
              elapsed_us(started) - execute_begin, "AXCL engine execution");
  const uint64_t download_begin = elapsed_us(started);
  for (uint32_t i = 0; i < n_out; ++i) {
    Tensor& out = response.outputs[i];
    void* dst = out.dtype == 1 ? static_cast<void*>(out.data.data())
                               : static_cast<void*>(out.raw_data.data());
    if (axclrtMemcpy(dst, m.outputs[i], m.out[i].bytes, AXCL_MEMCPY_DEVICE_TO_HOST))
      return fail("AXCL output download failed", true);
  }
  if (request.profiling == ProfilingLevel::Detailed) {
    add_profile("axcl_download", download_begin,
                elapsed_us(started) - download_begin, "device-to-host outputs");
  }
  response.ok = true;
  return true;
}

// Calls `use(model)` on the model at `path`: the cached one (loading it on a
// miss), or with --no-cache a model loaded for this request only.
template <typename Use>
static Response with_model(const Request& request, const std::string& path, Use&& use) {
  Response response;
  response.request_id = request.request_id;
  const auto started = std::chrono::steady_clock::now();
  std::string error;
  bool loaded = false;
  LoadedModel transient;
  LoadedModel* m = nullptr;
  if (g_use_cache) {
    m = acquire_cached(path, loaded, error);
  } else if (load_model(path, transient, error)) {
    m = &transient;
    loaded = true;
  }
  if (!m) { response.ok = false; response.error = error; return response; }
  if (loaded && request.profiling != ProfilingLevel::Off) {
    response.profile.push_back(ProfileEvent{
        "axcl_load", "axcl", 0, elapsed_us(started),
        g_use_cache ? "model loaded into the cache" : "model loaded for this request"});
  }
  bool device_failed = false;
  use(*m, response, started, device_failed);
  if (!g_use_cache) free_model(transient);
  else if (device_failed) evict(g_models.begin());  // `m` is the front entry
  return response;
}

static Response execute_axmodel(const Request& request) {
  return with_model(request, request.op,
                    [&](LoadedModel& m, Response& response,
                        const std::chrono::steady_clock::time_point& started,
                        bool& device_failed) {
                      run_loaded(m, request, response, started, device_failed);
                    });
}

// ---- io_info --------------------------------------------------------------

static const char* onnx_dtype_name(uint8_t dtype) {
  switch (dtype) {
    case 1: return "FLOAT";
    case 2: return "UINT8";
    case 3: return "INT8";
    case 4: return "UINT16";
    case 5: return "INT16";
    case 6: return "INT32";
    case 7: return "INT64";
    case 10: return "FLOAT16";
    case 11: return "DOUBLE";
    case 12: return "UINT32";
    case 13: return "UINT64";
    case 16: return "BFLOAT16";
    default: return "UNSUPPORTED";
  }
}

static std::string json_string(const std::string& value) {
  std::string out = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c < 0x20) {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
      out += buffer;
    } else out += static_cast<char>(c);
  }
  return out + "\"";
}

static std::string io_specs_json(const std::vector<IoSpec>& specs) {
  std::string out = "[";
  for (size_t i = 0; i < specs.size(); ++i) {
    const IoSpec& s = specs[i];
    if (i) out += ",";
    out += "{\"name\":" + json_string(s.name) +
           ",\"dtype\":" + std::to_string(s.onnx_dtype) +
           ",\"dtype_name\":\"" + onnx_dtype_name(s.onnx_dtype) + "\"" +
           ",\"axcl_dtype\":" + std::to_string(s.axcl_dtype) + ",\"shape\":[";
    for (size_t k = 0; k < s.shape.size(); ++k) {
      if (k) out += ",";
      out += std::to_string(s.shape[k]);
    }
    out += "],\"bytes\":" + std::to_string(s.bytes) + "}";
  }
  return out + "]";
}

static Response io_info(const Request& request, const std::string& path) {
  return with_model(request, path,
                    [&](LoadedModel& m, Response& response,
                        const std::chrono::steady_clock::time_point&, bool&) {
                      response.manifest =
                          "{\"schema_version\":1,\"model\":" + json_string(path) +
                          ",\"cached\":" + (g_use_cache ? "true" : "false") +
                          ",\"inputs\":" + io_specs_json(m.in) +
                          ",\"outputs\":" + io_specs_json(m.out) + "}";
                      if (response.manifest.size() > kMaxManifestBytes) {
                        response.manifest.clear();
                        response.error = "AXCL io_info exceeds the manifest limit";
                        return;
                      }
                      response.artifact_id = request.artifact_id;
                      response.ok = true;
                    });
}

// `io_info`, `unload` and `run` name their model either by artifact id (the
// cached artifact) or, with an empty artifact id, by a path carried as UTF-8
// text in the request's `model` bytes (the op field is taken by the op name).
static bool resolve_target(const Request& request, bool allow_none, std::string& path,
                           std::string& error) {
  path.clear();
  if (!request.artifact_id.empty()) {
    if (!valid_artifact_id(request.artifact_id)) { error = "invalid AXCL artifact id"; return false; }
    if (g_cache_dir.empty()) { error = "AXCL artifact cache is not configured"; return false; }
    path = (g_cache_dir / (request.artifact_id + ".axmodel")).string();
    return true;
  }
  path.assign(request.model.begin(), request.model.end());
  if (path.empty() && !allow_none) {
    error = "AXCL " + request.op + " needs an artifact id or a model path in the model field";
    return false;
  }
  return true;
}

static std::string capabilities_manifest() {
  return std::string(
             "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
             "\"runner_id\":\"axcl-worker\",\"ready\":true,"
             "\"graph_execution\":false,"
             "\"supported_ops\":[\"load_compiled\",\"run_compiled\",\"run\",\"io_info\",\"unload\"],"
             "\"supported_dtypes\":[\"FLOAT\",\"FLOAT16\",\"BFLOAT16\",\"DOUBLE\","
             "\"INT8\",\"UINT8\",\"INT16\",\"UINT16\",\"INT32\",\"UINT32\",\"INT64\",\"UINT64\"],"
             "\"model_cache\":") +
         (g_use_cache ? "true" : "false") +
         ",\"max_loaded\":" + std::to_string(g_use_cache ? g_max_loaded : 0) +
         ",\"loaded\":" + std::to_string(g_models.size()) +
         ",\"profiling\":true}";
}

static Response execute_request(const Request& request) {
  if (request.op == "capabilities") {
    Response response;
    response.request_id = request.request_id;
    response.ok = true;
    response.artifact_id = "axcl-worker";
    response.manifest = capabilities_manifest();
    return response;
  }
  if (request.op == "io_info" || request.op == "unload" || request.op == "run") {
    Response response;
    response.request_id = request.request_id;
    std::string path;
    if (!resolve_target(request, request.op == "unload", path, response.error))
      return response;
    if (request.op == "io_info") return io_info(request, path);
    if (request.op == "run") {
      // Same as sending the path as the op, without the op field's 128-byte limit.
      Request by_path = request;
      by_path.op = path;
      return execute_axmodel(by_path);
    }
    // unload: one model, or every loaded model when no target is named.
    size_t unloaded = 0;
    if (path.empty()) unloaded = unload_all();
    else {
      auto it = find_loaded(path);
      if (it != g_models.end()) { evict(it); unloaded = 1; }
    }
    response.ok = true;
    response.artifact_id = request.artifact_id;
    response.manifest = "{\"schema_version\":1,\"unloaded\":" + std::to_string(unloaded) +
                        ",\"loaded\":" + std::to_string(g_models.size()) + "}";
    return response;
  }
  if (request.op != "run_compiled" && request.op != "load_compiled")
    return execute_axmodel(request);
  Request cached = request;
  cached.request_id = request.request_id;
  std::string error;
  fs::path artifact_path;
  if (!materialize_artifact(request, artifact_path, error)) {
    Response response;
    response.request_id = request.request_id;
    response.error = error;
    return response;
  }
  if (request.op == "load_compiled") {
    Response response;
    response.request_id = request.request_id;
    response.ok = true;
    response.artifact_id = request.artifact_id;
    return response;
  }
  cached.op = artifact_path.string();
  return execute_axmodel(cached);
}

int main(int argc, char** argv) {
  uint16_t port = 39501;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoul(argv[++i]));
    else if (std::string(argv[i]) == "--cache-dir" && i + 1 < argc) g_cache_dir = argv[++i];
    else if (std::string(argv[i]) == "--max-loaded" && i + 1 < argc) {
      g_max_loaded = static_cast<size_t>(std::stoul(argv[++i]));
      if (g_max_loaded == 0) { std::cerr << "--max-loaded must be at least 1 (use --no-cache)\n"; return 2; }
    }
    else if (std::string(argv[i]) == "--no-cache") g_use_cache = false;
    else if (std::string(argv[i]) == "--help") {
      std::cout << "usage: onnx-remote-axcl-worker [--port PORT]"
                   " [--cache-dir DIR] [--max-loaded N] [--no-cache]\n"
                   "  --max-loaded N  models kept loaded on the card, least recently used"
                   " evicted first (default 64)\n"
                   "  --no-cache      load and unload the model on every request\n";
      return 0;
    }
    else { std::cerr << "unknown argument: " << argv[i] << '\n'; return 2; }
  }
  std::signal(SIGPIPE, SIG_IGN);
  // No SA_RESTART: a blocked accept() returns EINTR so the loop can leave and
  // the loaded models are released before the process ends.
  struct sigaction stop {};
  stop.sa_handler = on_stop_signal;
  sigemptyset(&stop.sa_mask);
  sigaction(SIGINT, &stop, nullptr);
  sigaction(SIGTERM, &stop, nullptr);
  axclError e = axclInit("/usr/bin/axcl/axcl.json");
  if (e) { std::cerr << "axclInit failed: 0x" << std::hex << e << '\n'; return 1; }
  axclrtDeviceList devices;
  if ((e = axclrtGetDeviceList(&devices)) || devices.num == 0 ||
      (e = axclrtSetDevice(devices.devices[0])) || (e = axclrtEngineInit(AXCL_VNPU_DISABLE))) {
    std::cerr << "AXCL device init failed: 0x" << std::hex << e << '\n'; axclFinalize(); return 1;
  }
  int listener = listen_tcp(port);
  if (listener < 0) {
    std::cerr << "listen failed: " << std::strerror(errno) << '\n';
    axclrtEngineFinalize(); axclFinalize();
    return 1;
  }
  std::cerr << "onnx-remote-axcl-worker listening on " << port
            << (g_use_cache ? " (model cache on)" : " (model cache off)") << '\n';
  while (!g_stop) {
    int fd = accept_tcp(listener); if (fd < 0) continue;
    Request request; Response response; std::string error;
    if (!receive_request(fd, request, error)) { response.error = error; }
    else response = execute_request(request);
    if (!send_response(fd, response, error)) std::cerr << "response failed: " << error << '\n';
    close_socket(fd);
  }
  close_socket(listener);
  unload_all();
  axclrtEngineFinalize();
  axclFinalize();
  return 0;
}
