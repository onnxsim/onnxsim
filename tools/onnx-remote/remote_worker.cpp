#include "remote_transport.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string>

using namespace onnx_remote;

static uint64_t micros_since(
    const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

static bool same_shape(const Tensor& a, const Tensor& b) { return a.shape == b.shape; }

static Response execute(const Request& r) {
  Response out; out.ok = false;
  const auto started = std::chrono::steady_clock::now();
  auto profile = [&](const char* name, uint64_t begin, uint64_t duration) {
    if (r.profiling != ProfilingLevel::Off) {
      out.profile.push_back(ProfileEvent{name, "remote", begin, duration, "reference-worker"});
    }
  };
  auto execute_op = [&](auto&& fn) {
    const uint64_t begin = micros_since(started);
    fn();
    profile(r.op.c_str(), begin, micros_since(started) - begin);
  };
  if (r.op == "identity") {
    if (r.inputs.size() != 1) { out.error = "identity expects one input"; return out; }
    execute_op([&] { out.outputs = r.inputs; }); out.ok = true; return out;
  }
  if (r.op == "relu") {
    if (r.inputs.size() != 1) { out.error = "relu expects one input"; return out; }
    Tensor y;
    execute_op([&] { y = r.inputs[0]; for (float& x : y.data) if (x < 0.0f) x = 0.0f; });
    out.outputs.push_back(std::move(y)); out.ok = true; return out;
  }
  if (r.op == "add" || r.op == "mul") {
    if (r.inputs.size() != 2 || !same_shape(r.inputs[0], r.inputs[1])) {
      out.error = r.op + " expects two tensors with the same shape"; return out;
    }
    Tensor y = r.inputs[0];
    execute_op([&] {
      for (size_t i = 0; i < y.data.size(); ++i)
        y.data[i] = r.op == "add" ? r.inputs[0].data[i] + r.inputs[1].data[i]
                                   : r.inputs[0].data[i] * r.inputs[1].data[i];
    });
    out.outputs.push_back(std::move(y)); out.ok = true; return out;
  }
  out.error = "unsupported operation: " + r.op; return out;
}

int main(int argc, char** argv) {
  uint16_t port = 39501;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::string(argv[i]) == "--help") { std::cout << "usage: onnx-remote-worker [--port PORT]\n"; return 0; }
    else { std::cerr << "unknown argument: " << argv[i] << '\n'; return 2; }
  }
  std::signal(SIGPIPE, SIG_IGN);
  int listener = listen_tcp(port);
  if (listener < 0) { std::cerr << "listen failed: " << std::strerror(errno) << '\n'; return 1; }
  std::cerr << "onnx-remote-worker listening on " << port << '\n';
  for (;;) {
    int fd = accept_tcp(listener); if (fd < 0) continue;
    Request request; Response response; std::string error;
    if (!receive_request(fd, request, error)) { response.error = error; }
    else response = execute(request);
    if (!response.ok && response.error.empty()) response.error = error.empty() ? "request failed" : error;
    if (!send_response(fd, response, error)) std::cerr << "response failed: " << error << '\n';
    close_socket(fd);
  }
}
