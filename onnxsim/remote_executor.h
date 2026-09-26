#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "onnxsim.h"
#include "remote_transport.h"

// Native remote executor for constant folding. The remote endpoint receives
// each fold-group ModelProto and its CPU tensors through the small transport in
// tools/onnx-remote. It is opt-in: the built-in ORT executor remains the
// default, and this declaration is available only with
// ONNXSIM_BUILTIN_REMOTE_EXECUTOR.
#ifdef ONNXSIM_BUILTIN_REMOTE_EXECUTOR

// A legalizer may rewrite a fold-group model before it is serialized for the
// compiler/runner. It returns false with a short reason when the target cannot
// accept the graph. This keeps vendor-specific rewrites out of the transport.
using RemoteLegalizer = std::function<bool(
    onnx::ModelProto&, const std::string& target, std::string& error)>;

// Called after an external compiler returns its manifest and before the
// artifact enters the process cache. A caller can reject a stale or
// incompatible compiler result without teaching the transport JSON.
using RemoteManifestValidator =
    std::function<bool(const std::string& manifest, std::string& error)>;

struct RemoteExecutorOptions {
  std::string host = "127.0.0.1";
  uint16_t port = 39501;
  // The compiler may run on a separate host. Empty/zero preserves the
  // execution endpoint above, so existing configurations remain unchanged.
  std::string compile_host;
  uint16_t compile_port = 0;
  int connect_timeout_ms = 5000;
  int io_timeout_ms = 0;
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
  std::string load_compiled_operation = "load_compiled";
  // Ask the runner to persist/attach an artifact once, then omit artifact
  // bytes from subsequent compiled runs. This is opt-in because stateless
  // runners only understand the inline artifact form.
  bool attach_compiled_artifact = false;
  // Send artifact bytes with every compiled run. This is reliable for a
  // stateless worker; a future load/cache handshake can disable it.
  bool send_compiled_artifact = true;
  // Off keeps profiling work at the minimum. Summary is suitable for
  // constrained cards; Detailed may include one event per device operation.
  onnx_remote::ProfilingLevel profiling = onnx_remote::ProfilingLevel::Off;
  // Stable compiler/runner target name passed to the legalization hook. The
  // compiler service's --target value should normally match this string.
  std::string target = "generic";
  // Optional target-specific graph legalization and post-compile manifest
  // validation. Both hooks are deliberately dependency-free; applications can
  // parse the manifest with their own JSON library or use a small allow-list.
  RemoteLegalizer legalizer;
  RemoteManifestValidator manifest_validator;
  // If non-empty, every remaining node after legalization must use one of
  // these operator types. This is a cheap preflight for constrained runners.
  std::vector<std::string> supported_ops;
};

std::shared_ptr<const ModelExecutor> GetRemoteModelExecutor(
    const RemoteExecutorOptions& options = {});

#endif  // ONNXSIM_BUILTIN_REMOTE_EXECUTOR
