// DORA node adapter for the dependency-free ONNX remote transport.
//
// The node accepts a UInt8 message named `run` containing the payload from
// encode_request_payload() and emits a UInt8 `result` message containing the
// status-prefixed payload from encode_response_payload(). DORA provides the
// dataflow/discovery lifecycle; the existing TCP worker remains responsible
// for model execution and accelerator-specific profiling.

#include "remote_transport.h"

#include <node_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace onnx_remote;

namespace {

std::string env_string(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || *value == '\0' ? fallback : value;
}

uint16_t env_port() {
  const char* value = std::getenv("ONNXSIM_DORA_REMOTE_PORT");
  if (value == nullptr || *value == '\0') return 39501;
  const unsigned long port = std::strtoul(value, nullptr, 10);
  return port > 0 && port <= 65535 ? static_cast<uint16_t>(port) : 39501;
}

bool forward(const uint8_t* data, size_t size, std::vector<uint8_t>& result,
             std::string& error) {
  Request request;
  if (!decode_request_payload(data, size, request, error)) return false;
  const int fd = connect_tcp(env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1"),
                             env_port());
  if (fd < 0) {
    error = "DORA adapter: remote worker connection failed";
    return false;
  }
  Response response;
  const bool sent = send_request(fd, request, error);
  const bool received = sent && receive_response(fd, response, error);
  close_socket(fd);
  if (!received) return false;
  return encode_response_payload(response, result, error);
}

void report_error(void* context, const std::string& error) {
  dora_log(context, "error", 5, error.data(), error.size());
  Response response;
  response.ok = false;
  response.error = error;
  std::vector<uint8_t> payload;
  std::string encode_error;
  if (encode_response_payload(response, payload, encode_error)) {
    dora_send_output(context, "result", 6, reinterpret_cast<const char*>(payload.data()),
                     payload.size());
  }
}

}  // namespace

int main() {
  void* context = init_dora_context_from_env();
  if (context == nullptr) {
    std::cerr << "failed to initialize DORA context\n";
    return 1;
  }

  for (;;) {
    void* event = dora_next_event(context);
    if (event == nullptr) break;
    const DoraEventType type = read_dora_event_type(event);
    if (type == DoraEventType_Stop) {
      free_dora_event(event);
      break;
    }
    if (type != DoraEventType_Input) {
      free_dora_event(event);
      continue;
    }

    char* id = nullptr;
    size_t id_len = 0;
    char* data = nullptr;
    size_t data_len = 0;
    read_dora_input_id(event, &id, &id_len);
    read_dora_input_data(event, &data, &data_len);
    const bool is_run = id != nullptr && id_len == 3 && std::memcmp(id, "run", 3) == 0;
    if (is_run && data != nullptr) {
      std::vector<uint8_t> result;
      std::string error;
      if (forward(reinterpret_cast<const uint8_t*>(data), data_len, result, error)) {
        if (dora_send_output(context, "result", 6,
                             reinterpret_cast<const char*>(result.data()),
                             result.size()) != 0) {
          report_error(context, "DORA adapter: result output failed");
        }
      } else {
        report_error(context, error.empty() ? "DORA adapter: request failed" : error);
      }
    }
    free_dora_event(event);
  }
  free_dora_context(context);
  return 0;
}
