#include "remote_transport.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace onnx_remote;

namespace {

uint64_t elapsed_us(const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

// ONNX TensorProto.DataType <-> TensorRT DataType. Anything not listed is
// rejected rather than silently cast.
bool to_trt_type(uint8_t onnx, nvinfer1::DataType& out) {
  switch (onnx) {
    case 1: out = nvinfer1::DataType::kFLOAT; return true;
    case 10: out = nvinfer1::DataType::kHALF; return true;
    case 16: out = nvinfer1::DataType::kBF16; return true;
    case 3: out = nvinfer1::DataType::kINT8; return true;
    case 2: out = nvinfer1::DataType::kUINT8; return true;
    case 6: out = nvinfer1::DataType::kINT32; return true;
    case 7: out = nvinfer1::DataType::kINT64; return true;
    case 9: out = nvinfer1::DataType::kBOOL; return true;
    default: return false;
  }
}

bool to_onnx_type(nvinfer1::DataType type, uint8_t& out, size_t& bytes) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: out = 1; bytes = 4; return true;
    case nvinfer1::DataType::kHALF: out = 10; bytes = 2; return true;
    case nvinfer1::DataType::kBF16: out = 16; bytes = 2; return true;
    case nvinfer1::DataType::kINT8: out = 3; bytes = 1; return true;
    case nvinfer1::DataType::kUINT8: out = 2; bytes = 1; return true;
    case nvinfer1::DataType::kINT32: out = 6; bytes = 4; return true;
    case nvinfer1::DataType::kINT64: out = 7; bytes = 8; return true;
    case nvinfer1::DataType::kBOOL: out = 9; bytes = 1; return true;
    default: return false;
  }
}

std::string hash_bytes(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 1469598103934665603ull;
  for (const uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
  return std::to_string(hash) + ":" + std::to_string(bytes.size());
}

std::string shapes_key(const std::vector<Tensor>& inputs) {
  std::string key;
  for (const auto& tensor : inputs) {
    key += '|';
    key += std::to_string(tensor.dtype);
    for (const int64_t dim : tensor.shape) key += "," + std::to_string(dim);
  }
  return key;
}

nvinfer1::Dims to_dims(const std::vector<int64_t>& shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) dims.d[i] = shape[i];
  return dims;
}

struct Compiled {
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
};

struct Options {
  bool fp16 = false;
  size_t workspace_mb = 1024;
  size_t max_engines = 4;
};

class EngineCache {
 public:
  EngineCache(Logger& logger, const Options& options)
      : logger_(logger),
        options_(options),
        runtime_(nvinfer1::createInferRuntime(logger)) {
    if (!runtime_) throw std::runtime_error("cannot create TensorRT runtime");
  }

  std::shared_ptr<Compiled> Find(const std::string& key) {
    const auto it = engines_.find(key);
    if (it == engines_.end()) return nullptr;
    order_.erase(std::find(order_.begin(), order_.end(), key));
    order_.push_back(key);
    return it->second;
  }

  // Build a plan from ONNX bytes. Dynamic dims are pinned to the request's
  // shapes (min == opt == max), so one engine serves one input signature.
  std::shared_ptr<Compiled> BuildOnnx(const std::string& key,
                                      const std::vector<uint8_t>& model,
                                      const std::vector<Tensor>& inputs) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
    if (!builder) throw std::runtime_error("cannot create TensorRT builder");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger_));
    if (!parser->parse(model.data(), model.size())) {
      std::string message = "ONNX parse failed";
      for (int i = 0; i < parser->getNbErrors(); ++i)
        message += std::string("; ") + parser->getError(i)->desc();
      throw std::runtime_error(message);
    }
    if (static_cast<size_t>(network->getNbInputs()) != inputs.size())
      throw std::runtime_error("model has " + std::to_string(network->getNbInputs()) +
                               " inputs, request has " + std::to_string(inputs.size()));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               options_.workspace_mb << 20);
    if (options_.fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    bool dynamic = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
      nvinfer1::ITensor* input = network->getInput(i);
      const nvinfer1::Dims declared = input->getDimensions();
      for (int d = 0; d < declared.nbDims; ++d) dynamic |= declared.d[d] < 0;
      const nvinfer1::Dims dims = to_dims(inputs[i].shape);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    }
    if (dynamic) config->addOptimizationProfile(profile);
    std::unique_ptr<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("TensorRT engine build failed");
    return Insert(key, plan->data(), plan->size());
  }

  std::shared_ptr<Compiled> LoadPlan(const std::string& key,
                                     const std::vector<uint8_t>& plan) {
    return Insert(key, plan.data(), plan.size());
  }

 private:
  std::shared_ptr<Compiled> Insert(const std::string& key, const void* plan,
                                   size_t size) {
    auto compiled = std::make_shared<Compiled>();
    compiled->engine.reset(runtime_->deserializeCudaEngine(plan, size));
    if (!compiled->engine)
      throw std::runtime_error(
          "cannot deserialize engine (built with a different TensorRT version or GPU?)");
    compiled->context.reset(compiled->engine->createExecutionContext());
    if (!compiled->context) throw std::runtime_error("cannot create execution context");
    while (engines_.size() >= std::max<size_t>(options_.max_engines, 1) &&
           !order_.empty()) {
      engines_.erase(order_.front());
      order_.pop_front();
    }
    engines_[key] = compiled;
    order_.push_back(key);
    return compiled;
  }

  Logger& logger_;
  Options options_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unordered_map<std::string, std::shared_ptr<Compiled>> engines_;
  std::deque<std::string> order_;
};

struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) {
    check_cuda(cudaMalloc(&ptr, std::max<size_t>(bytes, 1)), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// The raw payload of a request tensor in its wire dtype.
const void* tensor_payload(const Tensor& tensor, size_t& bytes) {
  if (tensor.dtype == 1) {
    bytes = tensor.data.size() * sizeof(float);
    return tensor.data.data();
  }
  bytes = tensor.raw_data.size();
  return tensor.raw_data.data();
}

void run(Compiled& compiled, const Request& request, Response& response) {
  nvinfer1::ICudaEngine& engine = *compiled.engine;
  nvinfer1::IExecutionContext& context = *compiled.context;
  std::vector<const char*> input_names, output_names;
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char* name = engine.getIOTensorName(i);
    (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? input_names
                                                                     : output_names)
        .push_back(name);
  }
  if (input_names.size() != request.inputs.size())
    throw std::runtime_error("engine has " + std::to_string(input_names.size()) +
                             " inputs, request has " +
                             std::to_string(request.inputs.size()));

  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<std::pair<const void*, size_t>> uploads;
  for (size_t i = 0; i < input_names.size(); ++i) {
    nvinfer1::DataType want = engine.getTensorDataType(input_names[i]);
    nvinfer1::DataType got;
    if (!to_trt_type(request.inputs[i].dtype, got))
      throw std::runtime_error("unsupported input dtype " +
                               std::to_string(request.inputs[i].dtype));
    if (got != want)
      throw std::runtime_error(std::string("input '") + input_names[i] +
                               "' dtype does not match the engine");
    size_t bytes = 0;
    const void* payload = tensor_payload(request.inputs[i], bytes);
    size_t elements = 1;
    for (const int64_t dim : request.inputs[i].shape) elements *= static_cast<size_t>(dim);
    uint8_t unused;
    size_t element_bytes = 0;
    to_onnx_type(want, unused, element_bytes);
    if (bytes != elements * element_bytes)
      throw std::runtime_error(std::string("input '") + input_names[i] +
                               "' payload size does not match its shape");
    if (!context.setInputShape(input_names[i], to_dims(request.inputs[i].shape)))
      throw std::runtime_error(std::string("shape rejected for input '") +
                               input_names[i] + "'");
    buffers.push_back(std::make_unique<DeviceBuffer>(bytes));
    context.setTensorAddress(input_names[i], buffers.back()->ptr);
    uploads.emplace_back(payload, bytes);
  }

  struct Output {
    const char* name;
    nvinfer1::Dims dims;
    uint8_t dtype;
    size_t bytes;
    size_t buffer;
  };
  std::vector<Output> outputs;
  for (const char* name : output_names) {
    Output output{name, context.getTensorShape(name), 0, 0, 0};
    size_t element_bytes = 0;
    if (!to_onnx_type(engine.getTensorDataType(name), output.dtype, element_bytes))
      throw std::runtime_error(std::string("unsupported output dtype on '") + name + "'");
    size_t elements = 1;
    for (int d = 0; d < output.dims.nbDims; ++d) {
      if (output.dims.d[d] < 0)
        throw std::runtime_error(std::string("output '") + name +
                                 "' has a data-dependent shape");
      elements *= static_cast<size_t>(output.dims.d[d]);
    }
    output.bytes = elements * element_bytes;
    buffers.push_back(std::make_unique<DeviceBuffer>(output.bytes));
    output.buffer = buffers.size() - 1;
    context.setTensorAddress(name, buffers.back()->ptr);
    outputs.push_back(output);
  }

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  try {
    for (size_t i = 0; i < uploads.size(); ++i)
      check_cuda(cudaMemcpyAsync(buffers[i]->ptr, uploads[i].first, uploads[i].second,
                                 cudaMemcpyHostToDevice, stream),
                 "upload");
    cudaEventRecord(begin, stream);
    if (!context.enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
    cudaEventRecord(end, stream);
    std::vector<std::vector<uint8_t>> host(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      host[i].resize(outputs[i].bytes);
      check_cuda(cudaMemcpyAsync(host[i].data(), buffers[outputs[i].buffer]->ptr,
                                 outputs[i].bytes, cudaMemcpyDeviceToHost, stream),
                 "download");
    }
    check_cuda(cudaStreamSynchronize(stream), "inference");
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, begin, end);
    if (request.profiling != ProfilingLevel::Off)
      response.profile.push_back(ProfileEvent{
          "trt_enqueue", "tensorrt", 0, static_cast<uint64_t>(gpu_ms * 1000.f),
          "GPU time of enqueueV3"});
    for (size_t i = 0; i < outputs.size(); ++i) {
      Tensor tensor;
      for (int d = 0; d < outputs[i].dims.nbDims; ++d)
        tensor.shape.push_back(outputs[i].dims.d[d]);
      if (outputs[i].dtype == 1) {
        tensor.data.resize(outputs[i].bytes / sizeof(float));
        std::memcpy(tensor.data.data(), host[i].data(), outputs[i].bytes);
      } else {
        tensor.dtype = outputs[i].dtype;
        tensor.raw_data = std::move(host[i]);
      }
      response.outputs.push_back(std::move(tensor));
    }
  } catch (...) {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    throw;
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);
}

Response execute(const Request& request, EngineCache& cache, const Options& options) {
  Response response;
  response.request_id = request.request_id;
  if (request.op == "capabilities") {
    response.ok = true;
    response.artifact_id = "tensorrt-worker";
    response.manifest =
        "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
        "\"runner_id\":\"tensorrt-worker\",\"ready\":true,"
        "\"graph_execution\":true,"
        "\"supported_ops\":[\"subgraph\",\"onnx\",\"engine\"],"
        "\"supported_dtypes\":[\"FLOAT\",\"FLOAT16\",\"BFLOAT16\",\"INT8\","
        "\"UINT8\",\"INT32\",\"INT64\",\"BOOL\"],"
        "\"profiling\":true,\"tensorrt_version\":\"" +
        std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) +
        "." + std::to_string(NV_TENSORRT_PATCH) + "\",\"fp16\":" +
        (options.fp16 ? "true" : "false") + "}";
    return response;
  }
  const bool is_engine = request.op == "engine";
  if (!is_engine && request.op != "subgraph" && request.op != "onnx") {
    response.error = "TensorRT worker expects subgraph, onnx or engine operation";
    return response;
  }
  try {
    std::string key;
    std::shared_ptr<Compiled> compiled;
    bool cache_hit = false;
    const auto started = std::chrono::steady_clock::now();
    if (is_engine) {
      // A prebuilt, serialized plan. Either upload it once under an
      // artifact_id, or send only the artifact_id to reuse the cached one.
      key = "engine:" + (request.artifact_id.empty() ? hash_bytes(request.artifact)
                                                      : request.artifact_id);
      compiled = cache.Find(key);
      cache_hit = compiled != nullptr;
      if (!compiled) {
        if (request.artifact.empty())
          throw std::runtime_error("unknown artifact_id and no engine bytes supplied");
        compiled = cache.LoadPlan(key, request.artifact);
      }
    } else {
      if (request.model.empty())
        throw std::runtime_error("TensorRT worker requires serialized ONNX graph bytes");
      key = "onnx:" + hash_bytes(request.model) + shapes_key(request.inputs);
      compiled = cache.Find(key);
      cache_hit = compiled != nullptr;
      if (!compiled) compiled = cache.BuildOnnx(key, request.model, request.inputs);
    }
    if (request.profiling != ProfilingLevel::Off)
      response.profile.push_back(ProfileEvent{
          cache_hit ? "trt_engine_cache_hit" : "trt_engine_build", "tensorrt", 0,
          elapsed_us(started),
          cache_hit ? "reused cached engine" : "built or deserialized engine"});
    run(*compiled, request, response);
    response.ok = true;
    return response;
  } catch (const std::exception& error) {
    response.error = std::string("TensorRT worker: ") + error.what();
    response.outputs.clear();
    response.profile.clear();
    return response;
  }
}

}  // namespace

int main(int argc, char** argv) {
  uint16_t port = 39505;
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--port" && i + 1 < argc)
      port = static_cast<uint16_t>(std::stoul(argv[++i]));
    else if (arg == "--fp16")
      options.fp16 = true;
    else if (arg == "--workspace-mb" && i + 1 < argc)
      options.workspace_mb = static_cast<size_t>(std::stoul(argv[++i]));
    else if (arg == "--max-engines" && i + 1 < argc)
      options.max_engines = static_cast<size_t>(std::stoul(argv[++i]));
    else if (arg == "--help") {
      std::cout << "usage: onnx-remote-tensorrt-worker [--port PORT] [--fp16]"
                   " [--workspace-mb N] [--max-engines N]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  try {
    Logger logger;
    EngineCache cache(logger, options);
    const int listener = listen_tcp(port);
    if (listener < 0) {
      std::cerr << "listen failed\n";
      return 1;
    }
    std::cerr << "onnx-remote-tensorrt-worker listening on " << port << '\n';
    for (;;) {
      const int fd = accept_tcp(listener);
      if (fd < 0) continue;
      Request request;
      Response response;
      std::string error;
      if (!receive_request(fd, request, error))
        response.error = error;
      else
        response = execute(request, cache, options);
      if (!send_response(fd, response, error))
        std::cerr << "response failed: " << error << '\n';
      close_socket(fd);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
