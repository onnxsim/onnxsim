// Integration test for onnx-remote-tensorrt-worker. The model is embedded so
// the test needs no ONNX Runtime or Python: y = Add(Relu(x), Relu(x)) over a
// float[N,3] input with a dynamic batch dimension.
#include "remote_capabilities.h"
#include "remote_transport.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace onnx_remote;

namespace {

const std::vector<uint8_t> kModel = {
    0x08, 0x09, 0x3a, 0x51, 0x0a, 0x0e, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x72,
    0x22, 0x04, 0x52, 0x65, 0x6c, 0x75, 0x3a, 0x00, 0x0a, 0x10, 0x0a, 0x01,
    0x72, 0x0a, 0x01, 0x72, 0x12, 0x01, 0x79, 0x22, 0x03, 0x41, 0x64, 0x64,
    0x3a, 0x00, 0x12, 0x01, 0x67, 0x5a, 0x14, 0x0a, 0x01, 0x78, 0x12, 0x0f,
    0x0a, 0x0d, 0x08, 0x01, 0x12, 0x09, 0x0a, 0x03, 0x12, 0x01, 0x4e, 0x0a,
    0x02, 0x08, 0x03, 0x62, 0x14, 0x0a, 0x01, 0x79, 0x12, 0x0f, 0x0a, 0x0d,
    0x08, 0x01, 0x12, 0x09, 0x0a, 0x03, 0x12, 0x01, 0x4e, 0x0a, 0x02, 0x08,
    0x03, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x11};

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) std::cerr << "[TensorRT] " << message << '\n';
  }
};

Response call(uint16_t port, const Request& request) {
  const int fd = connect_tcp_timeout("127.0.0.1", port, 2000);
  if (fd < 0) throw std::runtime_error("cannot connect to TensorRT worker");
  std::string error;
  if (!send_request(fd, request, error)) throw std::runtime_error(error);
  Response response;
  if (!receive_response(fd, response, error)) throw std::runtime_error(error);
  close_socket(fd);
  return response;
}

Tensor make_input(int64_t batch) {
  Tensor tensor;
  tensor.shape = {batch, 3};
  for (int64_t i = 0; i < batch * 3; ++i)
    tensor.data.push_back(static_cast<float>(i % 5) - 2.f);
  return tensor;
}

void expect_relu_add(const Tensor& input, const Response& response) {
  if (!response.ok) throw std::runtime_error("worker error: " + response.error);
  if (response.outputs.size() != 1 || response.outputs[0].shape != input.shape)
    throw std::runtime_error("unexpected output shape");
  for (size_t i = 0; i < input.data.size(); ++i)
    if (std::fabs(response.outputs[0].data[i] - 2.f * std::max(input.data[i], 0.f)) > 1e-5f)
      throw std::runtime_error("output value mismatch");
}

bool has_event(const Response& response, const std::string& name) {
  return std::any_of(response.profile.begin(), response.profile.end(),
                     [&](const ProfileEvent& e) { return e.name == name; });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: onnx-remote-tensorrt-worker-test PORT\n";
    return 2;
  }
  try {
    const uint16_t port = static_cast<uint16_t>(std::stoul(argv[1]));

    Request caps;
    caps.request_id = 8000;
    caps.op = "capabilities";
    const Response caps_response = call(port, caps);
    CapabilitySummary summary;
    std::string error;
    if (!caps_response.ok || !parse_capability_manifest(caps_response.manifest, summary, error) ||
        !summary.graph_execution || !summary.profiling)
      throw std::runtime_error("capability contract mismatch");

    Request request;
    request.op = "subgraph";
    request.model = kModel;
    request.profiling = ProfilingLevel::Detailed;

    // First call builds an engine for N=4, the second reuses it, and N=2 is a
    // different input signature so it builds again.
    request.request_id = 8001;
    request.inputs = {make_input(4)};
    Response first = call(port, request);
    expect_relu_add(request.inputs[0], first);
    if (!has_event(first, "trt_engine_build") || !has_event(first, "trt_enqueue"))
      throw std::runtime_error("missing build/enqueue profile events");
    request.request_id = 8002;
    Response second = call(port, request);
    expect_relu_add(request.inputs[0], second);
    if (!has_event(second, "trt_engine_cache_hit"))
      throw std::runtime_error("engine cache was not reused");
    request.request_id = 8003;
    request.inputs = {make_input(2)};
    expect_relu_add(request.inputs[0], call(port, request));

    // Prebuilt plan through the "engine" op, then by artifact_id alone.
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser->parse(kModel.data(), kModel.size())) throw std::runtime_error("local parse failed");
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    const char* input_name = network->getInput(0)->getName();
    for (auto selector : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::OptProfileSelector::kMAX})
      profile->setDimensions(input_name, selector, nvinfer1::Dims2{3, 3});
    config->addOptimizationProfile(profile);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("local engine build failed");

    Request engine_request;
    engine_request.request_id = 8004;
    engine_request.op = "engine";
    engine_request.artifact_id = "relu-add-n3";
    engine_request.artifact.assign(static_cast<const uint8_t*>(plan->data()),
                                   static_cast<const uint8_t*>(plan->data()) + plan->size());
    engine_request.inputs = {make_input(3)};
    engine_request.profiling = ProfilingLevel::Summary;
    expect_relu_add(engine_request.inputs[0], call(port, engine_request));
    engine_request.request_id = 8005;
    engine_request.artifact.clear();
    const Response by_id = call(port, engine_request);
    expect_relu_add(engine_request.inputs[0], by_id);
    if (!has_event(by_id, "trt_engine_cache_hit"))
      throw std::runtime_error("engine artifact_id was not reused");

    // Errors must come back as a response, not kill the worker.
    Request bad = request;
    bad.request_id = 8006;
    bad.inputs = {make_input(2), make_input(2)};
    if (call(port, bad).ok) throw std::runtime_error("input count mismatch was accepted");
    Request unknown = engine_request;
    unknown.request_id = 8007;
    unknown.artifact_id = "never-uploaded";
    if (call(port, unknown).ok) throw std::runtime_error("unknown artifact_id was accepted");
    request.request_id = 8008;
    request.inputs = {make_input(2)};
    expect_relu_add(request.inputs[0], call(port, request));

    std::cout << "TensorRT worker integration passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
