#include "remote_transport.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

using namespace onnx_remote;

int main() {
  Request request;
  request.op = "run_compiled";
  request.artifact_id = "qairt-test-1";
  request.model = {0x01, 0x02, 0x03};
  request.artifact = {0xaa, 0xbb};
  request.profiling = ProfilingLevel::Detailed;
  request.inputs.push_back(Tensor{{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}});

  std::vector<uint8_t> payload;
  std::string error;
  assert(encode_request_payload(request, payload, error));
  Request decoded_request;
  assert(decode_request_payload(payload.data(), payload.size(), decoded_request,
                                error));
  assert(decoded_request.op == request.op);
  assert(decoded_request.artifact_id == request.artifact_id);
  assert(decoded_request.model == request.model);
  assert(decoded_request.artifact == request.artifact);
  assert(decoded_request.profiling == request.profiling);
  assert(decoded_request.inputs.size() == 1);
  assert(decoded_request.inputs[0].shape == request.inputs[0].shape);
  assert(decoded_request.inputs[0].data == request.inputs[0].data);

  Response response;
  response.ok = true;
  response.outputs = request.inputs;
  response.profile.push_back(
      ProfileEvent{"qnn_execute", "qnn", 12, 34, "test"});
  response.artifact_id = request.artifact_id;
  response.artifact = {0x10, 0x20};
  response.manifest = "{\"schema_version\":1}";
  assert(encode_response_payload(response, payload, error));
  Response decoded_response;
  assert(decode_response_payload(payload.data(), payload.size(),
                                decoded_response, error));
  assert(decoded_response.ok);
  assert(decoded_response.outputs[0].data == response.outputs[0].data);
  assert(decoded_response.profile.size() == 1);
  assert(decoded_response.profile[0].duration_us == 34);
  assert(decoded_response.artifact_id == response.artifact_id);
  assert(decoded_response.artifact == response.artifact);
  assert(decoded_response.manifest == response.manifest);

  Request invalid = request;
  invalid.op.assign(kMaxOpBytes + 1, 'x');
  assert(!encode_request_payload(invalid, payload, error));
  Response oversized = response;
  oversized.manifest.assign(kMaxManifestBytes + 1, 'x');
  assert(!encode_response_payload(oversized, payload, error));
  return 0;
}
