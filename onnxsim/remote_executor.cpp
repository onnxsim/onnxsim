#include "remote_executor.h"

#ifdef ONNXSIM_BUILTIN_REMOTE_EXECUTOR

#include <cstring>
#include <stdexcept>
#include <utility>

#include "profiler.h"
#include "remote_transport.h"

namespace {

struct OutputOwner {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

DLManagedTensor* WrapOutput(OutputOwner* owner) {
  auto* result = new DLManagedTensor{};
  result->dl_tensor.data = owner->data.data();
  result->dl_tensor.device = DLDevice{kDLCPU, 0};
  result->dl_tensor.ndim = static_cast<int32_t>(owner->shape.size());
  result->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  result->dl_tensor.shape = owner->shape.data();
  result->dl_tensor.strides = nullptr;
  result->dl_tensor.byte_offset = 0;
  result->manager_ctx = owner;
  result->deleter = [](DLManagedTensor* tensor) {
    delete static_cast<OutputOwner*>(tensor->manager_ctx);
    delete tensor;
  };
  return result;
}

class RemoteModelExecutor final : public ModelExecutor {
 public:
  explicit RemoteModelExecutor(RemoteExecutorOptions options)
      : options_(std::move(options)) {}

  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
    onnx_remote::Request request;
    request.op = options_.operation;
    request.profiling = options_.profiling;
    auto& profiler = onnxsim::Profiler::Instance();
    const bool collect_profile =
        profiler.enabled() &&
        options_.profiling != onnx_remote::ProfilingLevel::Off;
    const uint64_t profile_anchor =
        collect_profile ? profiler.ElapsedMicros() : 0;
    const std::string serialized = model.SerializeAsString();
    request.model.assign(serialized.begin(), serialized.end());
    request.inputs.reserve(inputs.size());
    for (const DLManagedTensor* input : inputs) {
      const DLTensor& tensor = input->dl_tensor;
      if (tensor.device.device_type != kDLCPU ||
          tensor.dtype.code != kDLFloat || tensor.dtype.bits != 32 ||
          tensor.dtype.lanes != 1 || tensor.strides != nullptr) {
        throw std::runtime_error(
            "remote executor currently supports contiguous CPU float32 inputs "
            "only");
      }
      onnx_remote::Tensor wire;
      wire.shape.assign(tensor.shape, tensor.shape + tensor.ndim);
      const auto* begin = static_cast<const float*>(tensor.data) +
                          tensor.byte_offset / sizeof(float);
      size_t elements = 1;
      for (int32_t i = 0; i < tensor.ndim; ++i)
        elements *= static_cast<size_t>(tensor.shape[i]);
      wire.data.assign(begin, begin + elements);
      request.inputs.emplace_back(std::move(wire));
    }

    const int fd = onnx_remote::connect_tcp(options_.host, options_.port);
    if (fd < 0) throw std::runtime_error("remote executor: connection failed");
    std::string error;
    onnx_remote::Response response;
    const uint64_t rpc_start = collect_profile ? profiler.ElapsedMicros() : 0;
    const bool sent = onnx_remote::send_request(fd, request, error);
    const bool received =
        sent && onnx_remote::receive_response(fd, response, error);
    const uint64_t rpc_end = collect_profile ? profiler.ElapsedMicros() : 0;
    onnx_remote::close_socket(fd);
    if (collect_profile) {
      profiler.RecordExternalEvent(
          "RemoteRPC", "remote_transport", rpc_start,
          rpc_end >= rpc_start ? rpc_end - rpc_start : 0,
          "{\"host\":\"" + JsonEscape(options_.host) + "\"}");
      for (const auto& event : response.profile) {
        profiler.RecordExternalEvent(
            event.name, event.category, profile_anchor + event.start_us,
            event.duration_us,
            "{\"detail\":\"" + JsonEscape(event.detail) +
                "\",\"remote_start_us\":" + std::to_string(event.start_us) +
                "}");
      }
    }
    if (!received || !response.ok) {
      throw std::runtime_error("remote executor: " +
                               (error.empty() ? response.error : error));
    }

    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(response.outputs.size());
    for (auto& output : response.outputs) {
      auto owner = std::make_unique<OutputOwner>();
      owner->data = std::move(output.data);
      owner->shape = std::move(output.shape);
      outputs.emplace_back(WrapOutput(owner.release()));
    }
    return outputs;
  }

 private:
  RemoteExecutorOptions options_;
};

}  // namespace

std::shared_ptr<const ModelExecutor> GetRemoteModelExecutor(
    const RemoteExecutorOptions& options) {
  return std::make_shared<RemoteModelExecutor>(options);
}

#endif  // ONNXSIM_BUILTIN_REMOTE_EXECUTOR
