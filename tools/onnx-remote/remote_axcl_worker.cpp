// AXCL-backed worker for constrained-device operator tests.
//
// The request operation is the path of an .axmodel.  This intentionally keeps
// the first adapter simple: a host-side ORT EP can compile/cache a claimed
// subgraph to an axmodel and use this worker as its Execute() transport.  The
// worker accepts and returns float32 tensors, which matches the existing AX
// operator-test corpus and avoids pulling a serializer onto the device.
#include "remote_transport.h"

#include <axcl.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace onnx_remote;

static uint64_t elapsed_us(const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

static Response execute_axmodel(const Request& request) {
  Response response;
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
    for (void* p : inputs) axclrtFree(p);
    for (void* p : outputs) axclrtFree(p);
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
    axclrtEngineDataType type = 0;
    axclrtEngineGetInputDataType(info, i, &type);
    input_bytes[i] = axclrtEngineGetInputSizeByIndex(info, 0, i);
    if (type != 15 || input_bytes[i] != request.inputs[i].data.size() * sizeof(float) ||
        axclrtMalloc(&inputs[i], input_bytes[i], AXCL_MEM_MALLOC_NORMAL_ONLY) ||
        axclrtEngineSetInputBufferByIndex(io, i, inputs[i], input_bytes[i]))
      return fail("AXCL input must be float32 with the model's exact size");
    if (axclrtMemcpy(inputs[i], request.inputs[i].data.data(), input_bytes[i], AXCL_MEMCPY_HOST_TO_DEVICE))
      return fail("AXCL input upload failed");
  }
  response.outputs.resize(n_out);
  for (uint32_t i = 0; i < n_out; ++i) {
    axclrtEngineDataType type = 0; axclrtEngineIODims dims{};
    axclrtEngineGetOutputDataType(info, i, &type);
    axclrtEngineGetOutputDims(info, 0, i, &dims);
    output_bytes[i] = axclrtEngineGetOutputSizeByIndex(info, 0, i);
    if (type != 15 || output_bytes[i] % sizeof(float) ||
        axclrtMalloc(&outputs[i], output_bytes[i], AXCL_MEM_MALLOC_NORMAL_ONLY) ||
        axclrtEngineSetOutputBufferByIndex(io, i, outputs[i], output_bytes[i]))
      return fail("AXCL output must be float32");
    response.outputs[i].shape.reserve(dims.dimCount);
    uint64_t elements = 1;
    for (int k = 0; k < dims.dimCount; ++k) {
      response.outputs[i].shape.push_back(dims.dims[k]);
      elements *= static_cast<uint64_t>(dims.dims[k]);
    }
    if (elements * sizeof(float) != output_bytes[i]) return fail("AXCL output shape/size mismatch");
    response.outputs[i].data.resize(static_cast<size_t>(elements));
  }
  const uint64_t execute_begin = elapsed_us(started);
  if (axclrtEngineExecute(model, context, 0, io)) return fail("AXCL execute failed");
  add_profile("axcl_execute", execute_begin,
              elapsed_us(started) - execute_begin, "AXCL engine execution");
  const uint64_t download_begin = elapsed_us(started);
  for (uint32_t i = 0; i < n_out; ++i)
    if (axclrtMemcpy(response.outputs[i].data.data(), outputs[i], output_bytes[i], AXCL_MEMCPY_DEVICE_TO_HOST))
      return fail("AXCL output download failed");
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

int main(int argc, char** argv) {
  uint16_t port = 39501;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoul(argv[++i]));
    else if (std::string(argv[i]) == "--help") { std::cout << "usage: onnx-remote-axcl-worker [--port PORT]\n"; return 0; }
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
    else response = execute_axmodel(request);
    if (!send_response(fd, response, error)) std::cerr << "response failed: " << error << '\n';
    close_socket(fd);
  }
}
