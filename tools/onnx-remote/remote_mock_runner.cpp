#include "remote_transport.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace onnx_remote;

static std::unordered_map<std::string, std::vector<uint8_t>> artifacts;
static std::mutex artifacts_mutex;

static Response execute(const Request& request) {
  Response response;
  if (request.op == "load_compiled") {
    if (request.artifact_id.empty() || request.artifact.empty()) {
      response.error = "load_compiled requires artifact_id and artifact";
      return response;
    }
    std::lock_guard<std::mutex> lock(artifacts_mutex);
    artifacts[request.artifact_id] = request.artifact;
    response.ok = true;
    response.artifact_id = request.artifact_id;
    return response;
  }
  if (request.op != "run_compiled") {
    response.error = "mock runner only accepts load_compiled/run_compiled";
    return response;
  }
  {
    std::lock_guard<std::mutex> lock(artifacts_mutex);
    if (!request.artifact.empty()) artifacts[request.artifact_id] = request.artifact;
    if (request.artifact_id.empty() || artifacts.count(request.artifact_id) == 0) {
      response.error = "compiled artifact is not attached";
      return response;
    }
  }
  // The mock does not interpret the artifact. Identity output makes it useful
  // for testing the transport/cache handshake without a vendor runtime.
  response.outputs = request.inputs;
  response.ok = true;
  if (request.profiling != ProfilingLevel::Off)
    response.profile.push_back(ProfileEvent{"mock_run", "mock", 0, 1, "identity"});
  return response;
}

int main(int argc, char** argv) {
  uint16_t port = 39510;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc)
      port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::string(argv[i]) == "--help") {
      std::cout << "usage: onnx-remote-mock-runner [--port PORT]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << argv[i] << '\n';
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  const int listener = listen_tcp(port);
  if (listener < 0) {
    std::cerr << "listen failed: " << std::strerror(errno) << '\n';
    return 1;
  }
  for (;;) {
    const int fd = accept_tcp(listener);
    if (fd < 0) continue;
    Request request;
    Response response;
    std::string error;
    if (!receive_request(fd, request, error)) response.error = error;
    else response = execute(request);
    if (!response.ok && response.error.empty()) response.error = "mock runner failed";
    send_response(fd, response, error);
    close_socket(fd);
  }
}
