#include "remote_transport.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace onnx_remote;

static int self_test() {
  int fd = connect_tcp("127.0.0.1", 39501);
  if (fd < 0) { std::cerr << "connect failed (start onnx-remote-worker --port 39501)\n"; return 1; }
  Request r; r.op = "relu"; r.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::string error;
  if (!send_request(fd, r, error)) { std::cerr << error << '\n'; close_socket(fd); return 1; }
  Response response;
  if (!receive_response(fd, response, error) || !response.ok || response.outputs.size() != 1) {
    std::cerr << (error.empty() ? response.error : error) << '\n'; close_socket(fd); return 1;
  }
  const auto& got = response.outputs[0].data; const float want[] = {0, 0, 0, 1, 2};
  if (got.size() != 5) return 1;
  for (size_t i = 0; i < got.size(); ++i) if (std::fabs(got[i] - want[i]) > 1e-6f) return 1;
  std::cout << "remote transport self-test passed\n"; close_socket(fd); return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
  if (argc != 4) { std::cerr << "usage: onnx-remote-client HOST PORT OP\n"; return 2; }
  int fd = connect_tcp(argv[1], static_cast<uint16_t>(std::strtoul(argv[2], nullptr, 10)));
  if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
  Request r; r.op = argv[3]; r.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::string error; Response response;
  bool ok = send_request(fd, r, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok || !response.ok) { std::cerr << (error.empty() ? response.error : error) << '\n'; return 1; }
  for (float x : response.outputs[0].data) std::cout << x << '\n';
  return 0;
}
