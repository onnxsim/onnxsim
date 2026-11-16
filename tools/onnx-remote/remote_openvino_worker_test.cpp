// Smoke test for onnx-remote-openvino-worker.
//
// Usage: onnx-remote-openvino-worker-test MODEL.onnx PORT [D0,D1,...]
//
// Sends a float32 model through the worker's operations and checks that:
//   * capabilities advertises compile, load_compiled and run_compiled;
//   * compile returns an artifact, an artifact_id and a manifest;
//   * run_compiled with the artifact produces the same outputs as subgraph;
//   * a second run_compiled without the artifact bytes reuses the cache;
//   * Detailed profiling returns ov_infer and per-node openvino_node events.
// The model must have float32 inputs and outputs. The optional shape gives the
// input shape (default: 1,512) and inputs are deterministic pseudo-random values.

#include "remote_transport.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace onnx_remote;

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>());
}

std::vector<int64_t> parse_shape(const std::string& text) {
  std::vector<int64_t> shape;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) shape.push_back(std::stoll(item));
  return shape;
}

// Send one request on a fresh connection and return the response.
bool roundtrip(uint16_t port, const Request& request, Response& response) {
  const int fd = connect_tcp("127.0.0.1", port);
  if (fd < 0) {
    std::cerr << "cannot connect to 127.0.0.1:" << port << '\n';
    return false;
  }
  std::string error;
  bool ok = send_request(fd, request, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok) std::cerr << "transport error: " << error << '\n';
  return ok;
}

bool fail(const std::string& message) {
  std::cerr << "FAILED: " << message << '\n';
  return false;
}

bool same_outputs(const std::vector<Tensor>& a, const std::vector<Tensor>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].shape != b[i].shape || a[i].dtype != b[i].dtype ||
        a[i].data != b[i].data || a[i].raw_data != b[i].raw_data)
      return false;
  }
  return true;
}

bool has_event(const Response& response, const std::string& name) {
  for (const auto& event : response.profile)
    if (event.name == name) return true;
  return false;
}

// Per-node events are named after the OpenVINO node and tagged by category.
bool has_node_event(const Response& response) {
  for (const auto& event : response.profile)
    if (event.category == "openvino_node") return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: onnx-remote-openvino-worker-test MODEL.onnx PORT [D0,D1,...]\n";
    return 2;
  }
  const std::vector<uint8_t> model = read_file(argv[1]);
  const uint16_t port = static_cast<uint16_t>(std::strtoul(argv[2], nullptr, 10));
  const std::vector<int64_t> shape = parse_shape(argc > 3 ? argv[3] : "1,512");
  if (model.empty()) return fail("cannot read model") ? 0 : 1;

  Tensor input;
  input.shape = shape;
  size_t count = 1;
  for (const int64_t dim : shape) count *= static_cast<size_t>(dim);
  input.data.resize(count);
  uint32_t seed = 12345;
  for (auto& value : input.data) {
    seed = seed * 1664525u + 1013904223u;
    value = static_cast<float>(static_cast<int32_t>(seed >> 8) % 2000) / 1000.0f - 1.0f;
  }

  // 1. capabilities
  Request capabilities;
  capabilities.op = "capabilities";
  Response response;
  if (!roundtrip(port, capabilities, response) || !response.ok) return fail("capabilities") ? 0 : 1;
  for (const char* op : {"compile", "load_compiled", "run_compiled"})
    if (response.manifest.find(op) == std::string::npos)
      return fail(std::string("capabilities missing ") + op) ? 0 : 1;
  std::cout << "capabilities: " << response.manifest << '\n';

  // 2. compile
  Request compile;
  compile.op = "compile";
  compile.model = model;
  if (!roundtrip(port, compile, response) || !response.ok || response.artifact.empty() ||
      response.artifact_id.empty() || response.manifest.empty())
    return fail("compile did not return an artifact: " + response.error) ? 0 : 1;
  const std::string artifact_id = response.artifact_id;
  const std::vector<uint8_t> artifact = response.artifact;
  std::cout << "compiled " << artifact_id << " (" << artifact.size() << " bytes)\n";

  // 3. subgraph reference on the same device
  Request subgraph = MakeSubgraphRequest(1, model, {input}, ProfilingLevel::Detailed);
  Response reference;
  if (!roundtrip(port, subgraph, reference) || !reference.ok)
    return fail("subgraph: " + reference.error) ? 0 : 1;

  // 4. run_compiled with artifact bytes (first use, imports the blob)
  Request run;
  run.request_id = 2;
  run.op = "run_compiled";
  run.artifact_id = artifact_id;
  run.artifact = artifact;
  run.inputs = {input};
  run.profiling = ProfilingLevel::Detailed;
  if (!roundtrip(port, run, response) || !response.ok)
    return fail("run_compiled: " + response.error) ? 0 : 1;
  if (!same_outputs(response.outputs, reference.outputs))
    return fail("run_compiled outputs differ from subgraph outputs") ? 0 : 1;
  if (!has_event(response, "ov_infer") || !has_node_event(response)) {
    std::cerr << "profile events:";
    for (const auto& event : response.profile) std::cerr << ' ' << event.name;
    std::cerr << '\n';
    return fail("Detailed profile is missing ov_infer or openvino_node events") ? 0 : 1;
  }
  std::cout << "run_compiled: " << response.profile.size() << " profile events\n";

  // 5. run_compiled without artifact bytes must hit the cache
  Request cached = run;
  cached.request_id = 3;
  cached.artifact.clear();
  if (!roundtrip(port, cached, response) || !response.ok)
    return fail("cached run_compiled: " + response.error) ? 0 : 1;
  if (!same_outputs(response.outputs, reference.outputs))
    return fail("cached run outputs differ") ? 0 : 1;
  if (!has_event(response, "ov_cache_hit"))
    return fail("cached run did not report ov_cache_hit") ? 0 : 1;
  std::cout << "cached run_compiled: ov_cache_hit reported\n";

  std::cout << "openvino worker self-test passed\n";
  return 0;
}
