// Checks the onnx-remote reference worker on Knights Corner: RUN requests to the card, results compared locally.
#include "remote_transport.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace onnx_remote;
static bool run(const char* host, uint16_t port, const char* op, std::vector<Tensor> in, std::vector<float>& out) {
  int fd = connect_tcp(host, port);
  if (fd < 0) { std::cerr << "connect failed\n"; return false; }
  Request req; req.request_id = 1; req.op = op; req.inputs = std::move(in);
  std::string err; Response resp;
  bool ok = send_request(fd, req, err) && receive_response(fd, resp, err);
  close_socket(fd);
  if (!ok || !resp.ok) { std::cerr << op << ": " << (err.empty() ? resp.error : err) << "\n"; return false; }
  out = resp.outputs.empty() ? std::vector<float>() : resp.outputs[0].data;
  return true;
}
int main(int argc, char** argv) {
  const char* host = argc > 1 ? argv[1] : "172.31.1.1";
  uint16_t port = argc > 2 ? (uint16_t)std::atoi(argv[2]) : 39501;
  Tensor x; x.shape = {5}; x.data = {-2, -1, 0, 1, 2};
  Tensor y; y.shape = {5}; y.data = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  int fails = 0;
  std::vector<float> out;
  if (run(host, port, "relu", {x}, out)) {
    bool ok = out.size() == 5 && out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 1 && out[4] == 2;
    std::printf("relu: %s\n", ok ? "PASS" : "FAIL"); fails += !ok;
  } else fails++;
  if (run(host, port, "add", {x, y}, out)) {
    bool ok = out.size() == 5 && std::fabs(out[0] + 1.5f) < 1e-6f && std::fabs(out[4] - 2.5f) < 1e-6f;
    std::printf("add: %s\n", ok ? "PASS" : "FAIL"); fails += !ok;
  } else fails++;
  if (run(host, port, "tanh", {x}, out)) {
    float err = 0; for (int i = 0; i < 5; i++) err = std::fmax(err, std::fabs(out[i] - std::tanh(x.data[i])));
    bool ok = err < 1e-5f;
    std::printf("tanh: %s (max err %g)\n", ok ? "PASS" : "FAIL", (double)err); fails += !ok;
  } else fails++;
  return fails ? 1 : 0;
}
