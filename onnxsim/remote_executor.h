#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "onnxsim.h"
#include "remote_transport.h"

// Native remote executor for constant folding. The remote endpoint receives
// each fold-group ModelProto and its CPU tensors through the small transport in
// tools/onnx-remote. It is opt-in: the built-in ORT executor remains the
// default, and this declaration is available only with
// ONNXSIM_BUILTIN_REMOTE_EXECUTOR.
#ifdef ONNXSIM_BUILTIN_REMOTE_EXECUTOR

struct RemoteExecutorOptions {
  std::string host = "127.0.0.1";
  uint16_t port = 39501;
  // The compiler may run on a separate host. Empty/zero preserves the
  // execution endpoint above, so existing configurations remain unchanged.
  std::string compile_host;
  uint16_t compile_port = 0;
  int connect_timeout_ms = 5000;
  // The operation is a backend selector. "onnx" conventionally means that
  // the worker executes the serialized ModelProto; an AXCL adapter may use a
  // model handle instead and leave model empty.
  std::string operation = "onnx";
  // Compile each distinct fold-group submodel once on the external compiler,
  // then execute the returned artifact. The default preserves the simple
  // model-per-run protocol.
  bool compile_model = false;
  bool cache_compiled_models = true;
  std::string compile_operation = "compile";
  std::string compiled_operation = "run_compiled";
  // Send artifact bytes with every compiled run. This is reliable for a
  // stateless worker; a future load/cache handshake can disable it.
  bool send_compiled_artifact = true;
  // Off keeps profiling work at the minimum. Summary is suitable for
  // constrained cards; Detailed may include one event per device operation.
  onnx_remote::ProfilingLevel profiling = onnx_remote::ProfilingLevel::Off;
};

std::shared_ptr<const ModelExecutor> GetRemoteModelExecutor(
    const RemoteExecutorOptions& options = {});

#endif  // ONNXSIM_BUILTIN_REMOTE_EXECUTOR
