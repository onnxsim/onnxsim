#include "remote_transport.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

using namespace onnx_remote;

static int self_test() {
  Request payload_request;
  payload_request.request_id = 7;
  payload_request.op = "relu";
  payload_request.profiling = ProfilingLevel::Detailed;
  payload_request.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::vector<uint8_t> payload;
  std::string error;
  if (!encode_request_payload(payload_request, payload, error)) {
    std::cerr << "request payload encode failed: " << error << '\n';
    return 1;
  }
  Request decoded_request;
  if (!decode_request_payload(payload.data(), payload.size(), decoded_request, error) ||
      decoded_request.op != payload_request.op ||
      decoded_request.profiling != payload_request.profiling) {
    std::cerr << "request payload round-trip failed\n";
    return 1;
  }
  int fd = connect_tcp("127.0.0.1", 39501);
  if (fd < 0) { std::cerr << "connect failed (start onnx-remote-worker --port 39501)\n"; return 1; }
  Request r = payload_request;
  if (!send_request(fd, r, error)) { std::cerr << error << '\n'; close_socket(fd); return 1; }
  Response response;
  if (!receive_response(fd, response, error) || !response.ok || response.outputs.size() != 1) {
    std::cerr << (error.empty() ? response.error : error) << '\n'; close_socket(fd); return 1;
  }
  if (response.request_id != payload_request.request_id) {
    std::cerr << "response request id mismatch\n";
    close_socket(fd);
    return 1;
  }
  const auto& got = response.outputs[0].data; const float want[] = {0, 0, 0, 1, 2};
  if (got.size() != 5) return 1;
  for (size_t i = 0; i < got.size(); ++i) if (std::fabs(got[i] - want[i]) > 1e-6f) return 1;
  if (response.profile.empty() || response.profile[0].name != "relu") return 1;
  std::cout << "remote transport self-test passed (profile events: "
            << response.profile.size() << ")\n"; close_socket(fd); return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
  if (argc == 5 && std::string(argv[1]) == "--compile") {
    std::ifstream input(argv[4], std::ios::binary);
    if (!input) { std::cerr << "cannot open model\n"; return 1; }
    Request request;
    request.op = "compile";
    request.model.assign(std::istreambuf_iterator<char>(input), {});
    int fd = connect_tcp(argv[2], static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10)));
    if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
    std::string error;
    Response response;
    bool ok = send_request(fd, request, error) && receive_response(fd, response, error);
    close_socket(fd);
    if (!ok || !response.ok) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      return 1;
    }
    std::cout << "artifact_id=" << response.artifact_id
              << " bytes=" << response.artifact.size() << '\n'
              << response.manifest << '\n';
    return 0;
  }
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
