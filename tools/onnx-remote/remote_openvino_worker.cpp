// OpenVINO-backed onnx-remote worker.
//
// Speaks the same v5 transport as the other workers and adds compile and
// profiling on top of graph execution:
//
//   capabilities                  runner manifest (devices, ops, dtypes)
//   subgraph | onnx               compile (cached) and run a serialized ONNX
//                                 graph in one request, like the ORT worker
//   compile[:DEVICE]              compile a serialized ONNX graph for DEVICE
//                                 (default: --device) and return a portable
//                                 compiled artifact (OpenVINO blob) plus a
//                                 JSON manifest describing its inputs/outputs
//   load_compiled                 import an artifact into the runtime cache
//   run_compiled                  run an artifact by artifact_id; the artifact
//                                 bytes are only needed on first use or after
//                                 the worker restarts
//
// Profiling: Summary reports the total inference time. Detailed also emits one
// event per OpenVINO node from ov::InferRequest::get_profiling_info(). Node
// events carry real_time and exec_type, but OpenVINO does not expose per-node
// start times, so their start_us values are the cumulative sum of the
// preceding node times (a layout, not a measured timeline).
//
// Artifact layout (self-describing, so run_compiled needs no side channel):
//   "OVX1" | u32 big-endian device-name length | device name | OpenVINO blob

#include "remote_transport.h"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace onnx_remote;

namespace {

constexpr char kArtifactMagic[] = "OVX1";
constexpr size_t kArtifactHeaderBytes = 4 + 4;
// ONNX TensorProto.DataType value for float32; that tensor uses Tensor::data.
constexpr uint8_t kOnnxFloat = 1;

uint64_t elapsed_us(const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

std::string truncate(const std::string& text, size_t limit) {
  return text.size() <= limit ? text : text.substr(0, limit);
}

struct DtypeMap {
  uint8_t onnx;
  ov::element::Type_t ov;
};

// Only dtypes with a direct ONNX <-> OpenVINO mapping are carried. Anything
// else is rejected explicitly rather than reinterpreted.
const DtypeMap kDtypes[] = {
    {1, ov::element::Type_t::f32},   {10, ov::element::Type_t::f16},
    {16, ov::element::Type_t::bf16}, {11, ov::element::Type_t::f64},
    {2, ov::element::Type_t::u8},    {3, ov::element::Type_t::i8},
    {4, ov::element::Type_t::u16},   {5, ov::element::Type_t::i16},
    {6, ov::element::Type_t::i32},   {12, ov::element::Type_t::u32},
    {7, ov::element::Type_t::i64},   {13, ov::element::Type_t::u64},
    {9, ov::element::Type_t::boolean},
};

bool to_ov_type(uint8_t onnx, ov::element::Type& type) {
  for (const auto& entry : kDtypes) {
    if (entry.onnx == onnx) {
      type = ov::element::Type(entry.ov);
      return true;
    }
  }
  return false;
}

bool to_onnx_type(const ov::element::Type& type, uint8_t& onnx) {
  for (const auto& entry : kDtypes) {
    if (ov::element::Type(entry.ov) == type) {
      onnx = entry.onnx;
      return true;
    }
  }
  return false;
}

// Wrap a transport tensor in an ov::Tensor that aliases the request payload
// (no copy). The request must outlive inference.
bool wrap_input(const Tensor& input, const ov::element::Type& expected,
                ov::Tensor& out, std::string& error) {
  ov::element::Type type;
  if (!to_ov_type(input.dtype, type)) {
    error = "unsupported input dtype " + std::to_string(input.dtype);
    return false;
  }
  if (type != expected) {
    error = "dtype " + type.get_type_name() + " does not match model input " +
            expected.get_type_name();
    return false;
  }
  ov::Shape shape;
  size_t count = 1;
  for (const int64_t dim : input.shape) {
    if (dim < 0) {
      error = "negative input dimension";
      return false;
    }
    shape.push_back(static_cast<size_t>(dim));
    count *= static_cast<size_t>(dim);
  }
  void* data = nullptr;
  if (input.dtype == kOnnxFloat) {
    if (input.data.size() != count) {
      error = "float payload does not match its shape";
      return false;
    }
    data = const_cast<float*>(input.data.data());
  } else {
    if (input.raw_data.size() != count * type.size()) {
      error = "payload does not match its shape and dtype";
      return false;
    }
    data = const_cast<uint8_t*>(input.raw_data.data());
  }
  out = ov::Tensor(type, shape, data);
  return true;
}

bool from_ov_tensor(const ov::Tensor& tensor, Tensor& result, std::string& error) {
  uint8_t onnx = 0;
  if (!to_onnx_type(tensor.get_element_type(), onnx)) {
    error = "OpenVINO returned unsupported output dtype " +
            tensor.get_element_type().get_type_name();
    return false;
  }
  result = Tensor{};
  for (const size_t dim : tensor.get_shape())
    result.shape.push_back(static_cast<int64_t>(dim));
  result.dtype = onnx;
  if (onnx == kOnnxFloat) {
    const float* data = tensor.data<float>();
    result.data.assign(data, data + tensor.get_size());
  } else {
    const auto* data = static_cast<const uint8_t*>(tensor.data());
    result.raw_data.assign(data, data + tensor.get_byte_size());
  }
  return true;
}

std::string device_from_op(const std::string& op, const std::string& fallback) {
  const auto colon = op.find(':');
  if (colon == std::string::npos || colon + 1 == op.size()) return fallback;
  return op.substr(colon + 1);
}

std::string hash_bytes(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 1469598103934665603ull;
  for (const uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
  return std::to_string(hash) + "-" + std::to_string(bytes.size());
}

std::vector<uint8_t> build_artifact(const std::string& device, const std::string& blob) {
  std::vector<uint8_t> artifact(kArtifactHeaderBytes);
  std::memcpy(artifact.data(), kArtifactMagic, 4);
  const uint32_t length = static_cast<uint32_t>(device.size());
  artifact[4] = static_cast<uint8_t>(length >> 24);
  artifact[5] = static_cast<uint8_t>(length >> 16);
  artifact[6] = static_cast<uint8_t>(length >> 8);
  artifact[7] = static_cast<uint8_t>(length);
  artifact.insert(artifact.end(), device.begin(), device.end());
  artifact.insert(artifact.end(), blob.begin(), blob.end());
  return artifact;
}

bool parse_artifact(const std::vector<uint8_t>& artifact, std::string& device,
                    std::string& blob, std::string& error) {
  if (artifact.size() < kArtifactHeaderBytes ||
      std::memcmp(artifact.data(), kArtifactMagic, 4) != 0) {
    error = "artifact is not an OpenVINO compiled artifact";
    return false;
  }
  const uint32_t length = (static_cast<uint32_t>(artifact[4]) << 24) |
                          (static_cast<uint32_t>(artifact[5]) << 16) |
                          (static_cast<uint32_t>(artifact[6]) << 8) |
                          static_cast<uint32_t>(artifact[7]);
  if (length > artifact.size() - kArtifactHeaderBytes) {
    error = "artifact device header is truncated";
    return false;
  }
  const char* base = reinterpret_cast<const char*>(artifact.data());
  device.assign(base + kArtifactHeaderBytes, length);
  blob.assign(base + kArtifactHeaderBytes + length,
              artifact.size() - kArtifactHeaderBytes - length);
  return true;
}

// A compiled model plus the device and profiling mode it was built for.
struct Compiled {
  ov::CompiledModel model;
  std::string device;
  bool profiling = false;
};

// Owns the OpenVINO core and a bounded LRU cache of compiled models. Graph
// entries are keyed by content hash; artifact entries by artifact_id. Both
// include the device and profiling mode so profiling never leaks into a
// non-profiling request.
class Runner {
 public:
  Runner(std::string default_device, size_t max_models)
      : default_device_(std::move(default_device)), max_models_(max_models) {}

  const std::string& default_device() const { return default_device_; }

  std::vector<std::string> devices() {
    std::vector<std::string> names;
    for (const auto& device : core_.get_available_devices()) names.push_back(device);
    return names;
  }

  std::shared_ptr<Compiled> compile_graph(const std::vector<uint8_t>& graph,
                                          const std::string& device, bool profiling,
                                          bool& cache_hit) {
    const std::string key = "graph|" + device + (profiling ? "|p|" : "|") + hash_bytes(graph);
    if (auto cached = lookup(key)) {
      cache_hit = true;
      return cached;
    }
    cache_hit = false;
    const std::string bytes(graph.begin(), graph.end());
    auto model = core_.read_model(bytes, ov::Tensor());
    ov::AnyMap config;
    if (profiling) config.insert(ov::enable_profiling(true));
    auto compiled = std::make_shared<Compiled>();
    compiled->model = core_.compile_model(model, device, config);
    compiled->device = device;
    compiled->profiling = profiling;
    store(key, compiled);
    return compiled;
  }

  // Compile for `device` and build a self-describing artifact + manifest.
  void compile_artifact(const std::vector<uint8_t>& graph, const std::string& device,
                        Response& response) {
    const std::string bytes(graph.begin(), graph.end());
    auto model = core_.read_model(bytes, ov::Tensor());
    // Artifacts are always compiled with profiling enabled. Some plugins (the
    // NPU, for one) cannot report per-node profiles from a blob compiled
    // without it, and a later run_compiled may request Detailed profiling.
    // Requests still choose their own profile level at run time.
    ov::AnyMap config{ov::enable_profiling(true)};
    const auto started = std::chrono::steady_clock::now();
    auto compiled = core_.compile_model(model, device, config);
    const uint64_t compile_us = elapsed_us(started);
    std::ostringstream blob(std::ios::binary);
    compiled.export_model(blob);
    const std::string data = blob.str();
    response.artifact = build_artifact(device, data);
    response.artifact_id = "ov-" + device + "-" + hash_bytes(graph);
    response.manifest = manifest_json(response.artifact_id, device, compiled,
                                      compile_us, data.size());
    response.ok = true;
  }

  std::shared_ptr<Compiled> load_artifact(const std::string& artifact_id,
                                          const std::vector<uint8_t>& artifact,
                                          bool profiling, bool& cache_hit,
                                          std::string& error) {
    const std::string key = "artifact|" + artifact_id + (profiling ? "|p" : "|");
    if (auto cached = lookup(key)) {
      cache_hit = true;
      return cached;
    }
    cache_hit = false;
    if (artifact.empty()) {
      error = "compiled artifact is not loaded; attach it on first use";
      return nullptr;
    }
    std::string device, blob;
    if (!parse_artifact(artifact, device, blob, error)) return nullptr;
    try {
      std::istringstream stream(blob, std::ios::binary);
      ov::AnyMap config;
      if (profiling) config.insert(ov::enable_profiling(true));
      auto compiled = std::make_shared<Compiled>();
      compiled->model = core_.import_model(stream, device, config);
      compiled->device = device;
      compiled->profiling = profiling;
      store(key, compiled);
      return compiled;
    } catch (const std::exception& exception) {
      error = "OpenVINO import failed for " + device + ": " + exception.what();
      return nullptr;
    }
  }

 private:
  static std::string manifest_json(const std::string& artifact_id,
                                   const std::string& device,
                                   ov::CompiledModel& compiled,
                                   uint64_t compile_us, size_t blob_bytes) {
    std::ostringstream out;
    out << "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
        << "\"compiler\":{\"name\":\"openvino\",\"version\":\""
        << ov::get_openvino_version().buildNumber << "\"},"
        << "\"artifact_id\":\"" << artifact_id << "\","
        << "\"device\":\"" << device << "\","
        << "\"compile_us\":" << compile_us << ","
        << "\"blob_bytes\":" << blob_bytes << ",\"inputs\":[";
    for (size_t i = 0; i < compiled.inputs().size(); ++i) {
      const auto& port = compiled.inputs()[i];
      out << (i ? "," : "") << "{\"index\":" << i << ",\"name\":\""
          << port.get_any_name() << "\",\"type\":\""
          << port.get_element_type().get_type_name() << "\",\"shape\":\""
          << port.get_partial_shape() << "\"}";
    }
    out << "],\"outputs\":[";
    for (size_t i = 0; i < compiled.outputs().size(); ++i) {
      const auto& port = compiled.outputs()[i];
      out << (i ? "," : "") << "{\"index\":" << i << ",\"name\":\""
          << port.get_any_name() << "\",\"type\":\""
          << port.get_element_type().get_type_name() << "\",\"shape\":\""
          << port.get_partial_shape() << "\"}";
    }
    out << "]}";
    return out.str();
  }

  std::shared_ptr<Compiled> lookup(const std::string& key) {
    const auto found = cache_.find(key);
    if (found == cache_.end()) return nullptr;
    const auto order_it = std::find(order_.begin(), order_.end(), key);
    if (order_it != order_.end()) order_.erase(order_it);
    order_.push_back(key);
    return found->second;
  }

  void store(const std::string& key, const std::shared_ptr<Compiled>& compiled) {
    if (max_models_ == 0) return;
    while (cache_.size() >= max_models_ && !order_.empty()) {
      cache_.erase(order_.front());
      order_.erase(order_.begin());
    }
    cache_[key] = compiled;
    order_.push_back(key);
  }

  ov::Core core_;
  std::string default_device_;
  size_t max_models_;
  std::unordered_map<std::string, std::shared_ptr<Compiled>> cache_;
  std::vector<std::string> order_;
};

// Run `compiled` on `inputs` and build the response with outputs and profile.
Response run_model(const Request& request, Compiled& compiled, bool cache_hit,
                   const std::vector<Tensor>& inputs) {
  Response response;
  response.request_id = request.request_id;
  const auto& input_ports = compiled.model.inputs();
  if (input_ports.size() != inputs.size()) {
    response.error = "OpenVINO input count mismatch: model has " +
                     std::to_string(input_ports.size()) + ", request has " +
                     std::to_string(inputs.size());
    return response;
  }
  auto infer = compiled.model.create_infer_request();
  std::vector<ov::Tensor> wrapped(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string error;
    if (!wrap_input(inputs[i], input_ports[i].get_element_type(), wrapped[i], error)) {
      response.error = "input " + std::to_string(i) + ": " + error;
      return response;
    }
    if (wrapped[i].get_shape() != input_ports[i].get_shape()) {
      response.error = "input " + std::to_string(i) +
                       " shape does not match the compiled model";
      return response;
    }
    infer.set_input_tensor(i, wrapped[i]);
  }
  const auto started = std::chrono::steady_clock::now();
  try {
    infer.infer();
  } catch (const std::exception& exception) {
    response.error = std::string("OpenVINO inference failed: ") + exception.what();
    return response;
  }
  const uint64_t run_us = elapsed_us(started);
  for (size_t i = 0; i < compiled.model.outputs().size(); ++i) {
    Tensor tensor;
    std::string error;
    if (!from_ov_tensor(infer.get_output_tensor(i), tensor, error)) {
      response.error = error;
      response.outputs.clear();
      return response;
    }
    response.outputs.push_back(std::move(tensor));
  }

  if (request.profiling != ProfilingLevel::Off) {
    if (cache_hit)
      response.profile.push_back(ProfileEvent{"ov_cache_hit", "openvino", 0, 0,
                                              "reused compiled model on " + compiled.device});
    response.profile.push_back(ProfileEvent{"ov_infer", "openvino", 0, run_us,
                                            "device " + compiled.device});
    if (request.profiling == ProfilingLevel::Detailed && compiled.profiling) {
      uint64_t cursor = 0;
      for (const auto& info : infer.get_profiling_info()) {
        if (response.profile.size() >= kMaxProfileEvents) break;
        const uint64_t duration = static_cast<uint64_t>(info.real_time.count());
        response.profile.push_back(ProfileEvent{
            truncate(info.node_name, kMaxProfileNameBytes), "openvino_node", cursor,
            duration,
            truncate("exec_type=" + info.exec_type + " start=cumulative",
                     kMaxProfileDetailBytes)});
        cursor += duration;
      }
    }
  }
  response.ok = true;
  return response;
}

Response execute(const Request& request, Runner& runner) {
  Response response;
  response.request_id = request.request_id;
  const bool profiling = request.profiling != ProfilingLevel::Off;
  try {
    if (request.op == "capabilities") {
      std::string devices;
      for (const auto& device : runner.devices()) {
        if (!devices.empty()) devices += ",";
        devices += "\"" + device + "\"";
      }
      response.ok = true;
      response.artifact_id = "openvino-worker";
      response.manifest =
          "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
          "\"runner_id\":\"openvino-" + runner.default_device() + "\","
          "\"ready\":true,\"graph_execution\":true,"
          "\"supported_ops\":[\"subgraph\",\"onnx\",\"compile\","
          "\"load_compiled\",\"run_compiled\"],"
          "\"supported_dtypes\":[\"FLOAT\",\"FLOAT16\",\"BFLOAT16\",\"DOUBLE\","
          "\"INT8\",\"UINT8\",\"INT16\",\"UINT16\",\"INT32\",\"UINT32\","
          "\"INT64\",\"UINT64\",\"BOOL\"],"
          "\"devices\":[" + devices + "],\"profiling\":true}";
      return response;
    }
    if (request.op == "subgraph" || request.op == "onnx") {
      if (request.model.empty()) {
        response.error = "OpenVINO worker requires serialized ONNX graph bytes";
        return response;
      }
      bool cache_hit = false;
      const auto compiled = runner.compile_graph(request.model, runner.default_device(),
                                                 profiling, cache_hit);
      return run_model(request, *compiled, cache_hit, request.inputs);
    }
    if (request.op.rfind("compile", 0) == 0) {
      if (request.model.empty()) {
        response.error = "compile requires a serialized ONNX graph";
        return response;
      }
      runner.compile_artifact(request.model,
                              device_from_op(request.op, runner.default_device()),
                              response);
      response.request_id = request.request_id;
      return response;
    }
    if (request.op == "load_compiled") {
      if (request.artifact_id.empty() || request.artifact.empty()) {
        response.error = "load_compiled requires artifact_id and artifact";
        return response;
      }
      bool cache_hit = false;
      std::string error;
      if (!runner.load_artifact(request.artifact_id, request.artifact, profiling,
                                cache_hit, error)) {
        response.error = error;
        return response;
      }
      response.ok = true;
      response.artifact_id = request.artifact_id;
      return response;
    }
    if (request.op == "run_compiled") {
      bool cache_hit = false;
      std::string error;
      const auto compiled = runner.load_artifact(request.artifact_id, request.artifact,
                                                 profiling, cache_hit, error);
      if (!compiled) {
        response.error = error;
        return response;
      }
      return run_model(request, *compiled, cache_hit, request.inputs);
    }
    response.error =
        "OpenVINO worker supports capabilities, subgraph, onnx, compile[:DEVICE], "
        "load_compiled and run_compiled";
    return response;
  } catch (const std::exception& exception) {
    response.error = std::string("OpenVINO worker: ") + exception.what();
    return response;
  }
}

}  // namespace

int main(int argc, char** argv) {
  uint16_t port = 39504;
  std::string device = "CPU";
  size_t max_models = 4;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc) {
      port = static_cast<uint16_t>(std::stoul(argv[++i]));
    } else if (std::string(argv[i]) == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else if (std::string(argv[i]) == "--max-models" && i + 1 < argc) {
      max_models = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (std::string(argv[i]) == "--help") {
      std::cout << "usage: onnx-remote-openvino-worker [--port PORT] [--device DEV]"
                   " [--max-models N]\n"
                   "  DEV is an OpenVINO device name such as CPU, GPU or NPU.\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << argv[i] << '\n';
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  Runner runner(device, max_models);
  const int listener = listen_tcp(port);
  if (listener < 0) {
    std::cerr << "listen failed\n";
    return 1;
  }
  std::cerr << "onnx-remote-openvino-worker listening on " << port << " (device "
            << device << ")\n";
  for (;;) {
    const int fd = accept_tcp(listener);
    if (fd < 0) continue;
    Request request;
    Response response;
    std::string error;
    if (!receive_request(fd, request, error)) {
      response.error = error;
    } else {
      response = execute(request, runner);
    }
    if (!send_response(fd, response, error))
      std::cerr << "response failed: " << error << '\n';
    close_socket(fd);
  }
}
