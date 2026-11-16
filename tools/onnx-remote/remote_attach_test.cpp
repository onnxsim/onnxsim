#include "remote_transport.h"

#include <cstdlib>
#include <iostream>

using namespace onnx_remote;

static int fail(const std::string& message) {
  std::cerr << "onnx-remote-attach-test: " << message << '\n';
  return 1;
}

int main(int argc, char** argv) {
  const bool run_only = argc == 4 && std::string(argv[3]) == "--run-only";
  if (argc != 1 && argc != 3 && !run_only) {
    std::cerr << "usage: onnx-remote-attach-test [HOST PORT [--run-only]]\n";
    return 2;
  }
  const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
  const uint16_t port = argc > 2
                            ? static_cast<uint16_t>(std::strtoul(argv[2], nullptr, 10))
                            : 39510;
  const std::string id = "mock-artifact-1";
  std::string error;
  Response response;
  int fd = -1;
  {
    // Capability-gated dispatch: verify the runner is reachable and parse
    // its manifest before uploading an artifact.
    Request capabilities;
    capabilities.op = "capabilities";
    capabilities.request_id = 1;
    fd = connect_tcp_timeout(host, port, 2000);
    if (fd < 0) return fail("capabilities: cannot connect to " + host + ":" + std::to_string(port));
    if (!send_request(fd, capabilities, error)) return fail("capabilities: send failed: " + error);
    if (!receive_response(fd, response, error)) {
      close_socket(fd);
      return fail("capabilities: receive failed: " + error);
    }
    if (!response.ok) {
      close_socket(fd);
      return fail("capabilities: runner error: " + response.error);
    }
    close_socket(fd);
    if (response.manifest.find("mock-runner") == std::string::npos ||
        response.manifest.find("run_compiled") == std::string::npos) {
      return fail("unexpected runner capability manifest: " + response.manifest);
    }
  }
  if (!run_only) {
    Request load;
    load.op = "load_compiled";
    load.artifact_id = id;
    load.artifact = {1, 2, 3, 4};
    fd = connect_tcp_timeout(host, port, 2000);
    if (fd < 0) return fail("load_compiled: cannot connect");
    if (!send_request(fd, load, error)) return fail("load_compiled: send failed: " + error);
    if (!receive_response(fd, response, error)) return fail("load_compiled: receive failed: " + error);
    if (!response.ok) return fail("load_compiled: runner error: " + response.error);
    close_socket(fd);
  }

  Request run;
  run.op = "run_compiled";
  run.artifact_id = id;
  run.profiling = ProfilingLevel::Detailed;
  run.inputs.push_back(Tensor{{3}, {1.0f, 2.0f, 3.0f}});
  fd = connect_tcp_timeout(host, port, 2000);
  if (fd < 0) return fail("run_compiled: cannot connect");
  if (!send_request(fd, run, error)) return fail("run_compiled: send failed: " + error);
  if (!receive_response(fd, response, error)) return fail("run_compiled: receive failed: " + error);
  close_socket(fd);
  if (!response.ok) return fail("run_compiled: runner error: " + response.error);
  if (response.outputs.size() != 1 || response.outputs[0].data != run.inputs[0].data)
    return fail("run_compiled: outputs do not echo the input");
  if (response.profile.empty()) return fail("run_compiled: Detailed profiling returned no events");
  std::cout << "compiled artifact attach/run self-test passed\n";
  return 0;
}
