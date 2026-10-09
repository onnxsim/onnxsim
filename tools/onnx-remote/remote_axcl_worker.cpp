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
#include "remote_transport.h"

#include <axcl.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

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

static Response execute_axmodel(const Request& request) {
  Response response;
  response.request_id = request.request_id;
  const auto started = std::chrono::steady_clock::now();
  auto add_profile = [&](const char* name, uint64_t begin, uint64_t duration,
                         const char* detail) {
    if (request.profiling != ProfilingLevel::Off) {
      response.profile.push_back(ProfileEvent{name, "axcl", begin, duration, detail});
    }
  };
  uint64_t model = 0, context = 0;
  axclrtEngineIOInfo info = nullptr;
  axclrtEngineIO io = nullptr;
  std::vector<void*> inputs, outputs;
  std::vector<uint64_t> input_bytes, output_bytes;
  auto fail = [&](const std::string& message) {
    response.ok = false; response.error = message;
    for (void* p : inputs) if (p) axclrtFree(p);
    for (void* p : outputs) if (p) axclrtFree(p);
    if (io) axclrtEngineDestroyIO(io);
    if (info) axclrtEngineDestroyIOInfo(info);
    if (model) axclrtEngineUnload(model);
    return response;
  };

  if (request.op.empty() || axclrtEngineLoadFromFile(request.op.c_str(), &model) ||
      axclrtEngineCreateContext(model, &context) ||
      axclrtEngineGetIOInfo(model, &info) ||
      axclrtEngineCreateIO(info, &io))
    return fail("AXCL model setup failed: " + request.op);

  const uint32_t n_in = axclrtEngineGetNumInputs(info);
  const uint32_t n_out = axclrtEngineGetNumOutputs(info);
  if (n_in != request.inputs.size()) return fail("AXCL input count mismatch");
  inputs.resize(n_in); input_bytes.resize(n_in);
  outputs.resize(n_out); output_bytes.resize(n_out);

  for (uint32_t i = 0; i < n_in; ++i) {
    axclrtEngineDataType type{};
    axclrtEngineGetInputDataType(info, i, &type);
    input_bytes[i] = axclrtEngineGetInputSizeByIndex(info, 0, i);
    // Float32 keeps its historical home in `data`; every other dtype travels
    // as its little-endian payload in `raw_data`.
    const Tensor& in = request.inputs[i];
    uint8_t expected = 0;
    if (!onnx_dtype_of_axcl(type, expected) || in.dtype != expected)
      return fail("AXCL input " + std::to_string(i) + " dtype mismatch: model wants ONNX dtype " +
                  std::to_string(expected) + ", request sent " + std::to_string(in.dtype));
    const void* src = in.dtype == 1 ? static_cast<const void*>(in.data.data())
                                    : static_cast<const void*>(in.raw_data.data());
    const uint64_t src_bytes = in.dtype == 1 ? in.data.size() * sizeof(float) : in.raw_data.size();
    if (input_bytes[i] != src_bytes ||
        axclrtMalloc(&inputs[i], input_bytes[i], AXCL_MEM_MALLOC_NORMAL_ONLY) ||
        axclrtEngineSetInputBufferByIndex(io, i, inputs[i], input_bytes[i]))
      return fail("AXCL input " + std::to_string(i) + " must have the model's exact size");
    if (axclrtMemcpy(inputs[i], src, input_bytes[i], AXCL_MEMCPY_HOST_TO_DEVICE))
      return fail("AXCL input upload failed");
  }
  response.outputs.resize(n_out);
  for (uint32_t i = 0; i < n_out; ++i) {
    axclrtEngineDataType type{}; axclrtEngineIODims dims{};
    axclrtEngineGetOutputDataType(info, i, &type);
    axclrtEngineGetOutputDims(info, 0, i, &dims);
    output_bytes[i] = axclrtEngineGetOutputSizeByIndex(info, 0, i);
    Tensor& out = response.outputs[i];
    if (!onnx_dtype_of_axcl(type, out.dtype) || dtype_bytes(out.dtype) == 0)
      return fail("AXCL output " + std::to_string(i) + " has an unsupported dtype");
    if (axclrtMalloc(&outputs[i], output_bytes[i], AXCL_MEM_MALLOC_NORMAL_ONLY) ||
        axclrtEngineSetOutputBufferByIndex(io, i, outputs[i], output_bytes[i]))
      return fail("AXCL output buffer setup failed");
    out.shape.reserve(dims.dimCount);
    uint64_t elements = 1;
    for (int k = 0; k < dims.dimCount; ++k) {
      out.shape.push_back(dims.dims[k]);
      elements *= static_cast<uint64_t>(dims.dims[k]);
    }
    if (elements * dtype_bytes(out.dtype) != output_bytes[i])
      return fail("AXCL output shape/size mismatch");
    if (out.dtype == 1) out.data.resize(static_cast<size_t>(elements));
    else out.raw_data.resize(static_cast<size_t>(output_bytes[i]));
  }
  const uint64_t execute_begin = elapsed_us(started);
  if (axclrtEngineExecute(model, context, 0, io)) return fail("AXCL execute failed");
  add_profile("axcl_execute", execute_begin,
              elapsed_us(started) - execute_begin, "AXCL engine execution");
  const uint64_t download_begin = elapsed_us(started);
  for (uint32_t i = 0; i < n_out; ++i) {
    Tensor& out = response.outputs[i];
    void* dst = out.dtype == 1 ? static_cast<void*>(out.data.data())
                               : static_cast<void*>(out.raw_data.data());
    if (axclrtMemcpy(dst, outputs[i], output_bytes[i], AXCL_MEMCPY_DEVICE_TO_HOST))
      return fail("AXCL output download failed");
  }
  if (request.profiling == ProfilingLevel::Detailed) {
    add_profile("axcl_download", download_begin,
                elapsed_us(started) - download_begin, "device-to-host outputs");
  }
  response.ok = true;
  for (void* p : inputs) axclrtFree(p);
  for (void* p : outputs) axclrtFree(p);
  axclrtEngineDestroyIO(io); axclrtEngineDestroyIOInfo(info); axclrtEngineUnload(model);
  return response;
}

static Response execute_request(const Request& request) {
  if (request.op == "capabilities") {
    Response response;
    response.request_id = request.request_id;
    response.ok = true;
    response.artifact_id = "axcl-worker";
    response.manifest =
        "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
        "\"runner_id\":\"axcl-worker\",\"ready\":true,"
        "\"graph_execution\":false,"
        "\"supported_ops\":[\"load_compiled\",\"run_compiled\"],"
        "\"supported_dtypes\":[\"FLOAT\",\"FLOAT16\",\"BFLOAT16\",\"DOUBLE\","
        "\"INT8\",\"UINT8\",\"INT16\",\"UINT16\",\"INT32\",\"UINT32\",\"INT64\",\"UINT64\"],"
        "\"profiling\":true}";
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
    else if (std::string(argv[i]) == "--help") {
      std::cout << "usage: onnx-remote-axcl-worker [--port PORT]"
                   " [--cache-dir DIR]\n";
      return 0;
    }
    else { std::cerr << "unknown argument: " << argv[i] << '\n'; return 2; }
  }
  std::signal(SIGPIPE, SIG_IGN);
  axclError e = axclInit("/usr/bin/axcl/axcl.json");
  if (e) { std::cerr << "axclInit failed: 0x" << std::hex << e << '\n'; return 1; }
  axclrtDeviceList devices;
  if ((e = axclrtGetDeviceList(&devices)) || devices.num == 0 ||
      (e = axclrtSetDevice(devices.devices[0])) || (e = axclrtEngineInit(AXCL_VNPU_DISABLE))) {
    std::cerr << "AXCL device init failed: 0x" << std::hex << e << '\n'; axclFinalize(); return 1;
  }
  int listener = listen_tcp(port);
  if (listener < 0) { std::cerr << "listen failed: " << std::strerror(errno) << '\n'; return 1; }
  std::cerr << "onnx-remote-axcl-worker listening on " << port << '\n';
  for (;;) {
    int fd = accept_tcp(listener); if (fd < 0) continue;
    Request request; Response response; std::string error;
    if (!receive_request(fd, request, error)) { response.error = error; }
    else response = execute_request(request);
    if (!send_response(fd, response, error)) std::cerr << "response failed: " << error << '\n';
    close_socket(fd);
  }
}
