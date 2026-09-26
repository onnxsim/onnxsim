#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_remote {

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

enum class ProfilingLevel : uint8_t {
  Off = 0,
  Summary = 1,
  Detailed = 2,
};

struct ProfileEvent {
  std::string name;
  std::string category;
  // Timestamps are microseconds relative to the beginning of the request on
  // the worker. The client anchors them to its local trace clock.
  uint64_t start_us = 0;
  uint64_t duration_us = 0;
  // Short human-readable metadata. It is deliberately not a JSON blob so the
  // constrained worker never needs a JSON library.
  std::string detail;
};

struct Request {
  std::string op;
  // Optional serialized ONNX ModelProto. A model handle can be used instead
  // by leaving this empty and putting the handle in `op`.
  std::vector<uint8_t> model;
  std::vector<Tensor> inputs;
  ProfilingLevel profiling = ProfilingLevel::Off;
};

struct Response {
  bool ok = false;
  std::string error;
  std::vector<Tensor> outputs;
  std::vector<ProfileEvent> profile;
};

// The implementation uses one request per connected socket. These limits are
// deliberately conservative for an embedded endpoint and prevent a malformed
// peer from turning a length field into an unbounded allocation.
constexpr uint32_t kMaxOpBytes = 128;
constexpr uint32_t kMaxProfileNameBytes = 128;
constexpr uint32_t kMaxProfileDetailBytes = 512;
constexpr uint32_t kMaxProfileEvents = 256;
constexpr uint32_t kMaxTensors = 32;
constexpr uint32_t kMaxRank = 8;
constexpr uint64_t kMaxTensorBytes = 256ull * 1024ull * 1024ull;
constexpr uint64_t kMaxMessageBytes = 512ull * 1024ull * 1024ull;

int listen_tcp(uint16_t port, int backlog = 16);
int accept_tcp(int listener);
int connect_tcp(const std::string& host, uint16_t port);
void close_socket(int fd);

bool receive_request(int fd, Request& request, std::string& error);
bool send_request(int fd, const Request& request, std::string& error);
bool receive_response(int fd, Response& response, std::string& error);
bool send_response(int fd, const Response& response, std::string& error);

}  // namespace onnx_remote
