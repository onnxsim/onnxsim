# ONNX remote execution transport

This directory contains the small, dependency-free transport used by the
remote execution experiments.  It is intentionally separate from Python,
protobuf, and ONNX Runtime so that the client can be built for a constrained
target (for example an AX8850/Snapdragon device) with only a C++17 compiler
and the system socket library.

The protocol is a length-prefixed binary request/response protocol:

```text
client                         worker
  RUN(op, tensors)  --------->
                    <---------  OK(tensors) or ERR(message)
```

The first worker is a reference implementation for transport tests.  It
implements `identity`, `relu`, `add`, and `mul` over float32 tensors.  It is
not intended to be the production accelerator backend.  An AXCL worker can
replace the operation callback while keeping the wire format unchanged.

## Build

```sh
cmake -S tools/onnx-remote -B build/onnx-remote
cmake --build build/onnx-remote -j2
```

## Run the reference worker

```sh
./build/onnx-remote/onnx-remote-worker --port 39501
```

The protocol library is also usable directly from a C++ remote execution
provider.  `onnx-remote-client` is a small smoke-test client; it sends one
input tensor containing `-2,-1,0,1,2` to `relu` and checks the response.
The self-test also requests detailed profiling and checks that the worker
returns at least one binary profile event.

Run that smoke test while the worker is running:

```sh
./build/onnx-remote/onnx-remote-client --self-test
```

## Design constraints

* bounded message size and tensor count;
* complete reads/writes (short socket writes are expected);
* network byte order for all integer fields;
* no dynamic dependency on Python or protobuf;
* one request per connection for simple failure isolation;
* explicit operation and tensor metadata, so the remote side never guesses
  dtype or shape. Protocol v4 preserves raw little-endian payloads for
  FLOAT16, BFLOAT16, integer, DOUBLE, and BOOL tensors; the reference worker
  still executes float32 only.
* optional `Off`, `Summary`, or `Detailed` profiling in the request;
  profile timestamps are worker-relative and require no clock synchronization.
* bounded connection setup when the native executor's `connect_timeout_ms` is
  configured;

Profile events are returned with the response rather than streamed. This keeps
the constrained worker simple and is sufficient for a completed subgraph
profile. A future ROS/HTTP gateway can stream progress separately while using
the same event fields for the final trace.

## DORA adapter

The optional `onnx-remote-dora-node` is a standalone DORA C node. It accepts a
UInt8 message named `run`, containing the payload produced by
`encode_request_payload()`, forwards it to the normal TCP worker, and emits a
UInt8 `result` message containing `encode_response_payload()` output. This
keeps DORA's dataflow/discovery layer separate from the accelerator transport.

Build it against a DORA C node API static library:

```sh
cmake -S tools/onnx-remote -B build/onnx-remote \
  -DONNXSIM_REMOTE_DORA=ON \
  -DDORA_NODE_API_INCLUDE_DIR=/path/to/dora/apis/c/node \
  -DDORA_NODE_API_LIBRARY=/path/to/dora/target/release/libdora_node_api_c.a
cmake --build build/onnx-remote --target onnx-remote-dora-node
```

Set `ONNXSIM_DORA_REMOTE_HOST` and `ONNXSIM_DORA_REMOTE_PORT` in the node's
environment to select the native worker (defaults are `127.0.0.1:39501`). A
minimal dataflow declares `run` as the node input and `result` as its output.
The DORA node API carries raw UInt8 messages; tensor and profile serialization
remain the same as the dependency-free transport.

The next integration layer can make an ONNX Runtime plugin EP claim a maximal
supported subgraph and send it as an operation/model handle over this
transport.  The reference worker deliberately does not pretend to be that EP
yet.

Set `ONNXSIM_DORA_ANNOUNCE=1` and declare optional `status` and `capabilities`
outputs to publish readiness and the binary protocol capability document. The
adapter also accepts `ONNXSIM_DORA_CONNECT_TIMEOUT_MS` and
`ONNXSIM_DORA_IO_TIMEOUT_MS` for unreliable links.

## ROS2 bridge

The optional `onnx-remote-ros2-bridge` uses ROS2 only for discovery and control:
`run` and `result` are `std_msgs/UInt8MultiArray`, while `health` and
`capabilities` are `std_srvs/Trigger` services. The UInt8 payload is exactly
the DORA/native payload format, so a ROS2 graph, DORA graph, or direct TCP
client can share the same worker.

Build inside a sourced ROS2 workspace:

```sh
cmake -S tools/onnx-remote -B build/onnx-remote \
  -DONNXSIM_REMOTE_ROS2=ON
cmake --build build/onnx-remote --target onnx-remote-ros2-bridge
build/onnx-remote/onnx-remote-ros2-bridge \
  --ros-args -p remote_host:=runner.local -p remote_port:=39501
```

For a reproducible development environment using `nix-ros-overlay`:

```sh
cd tools/onnx-remote
nix develop
cmake -S . -B build -DONNXSIM_REMOTE_ROS2=ON
cmake --build build --target onnx-remote-ros2-bridge
cmake --install build --prefix /tmp/onnx-remote-install
```

The flake currently pins the ROS2 Humble package set from nixpkgs. It does not
replace the project toolchain; it only supplies CMake, `ament_cmake`, `rclcpp`,
`std_msgs`, `std_srvs`, and the ROS2 CLI.

### ROS2 runner discovery

Set `auto_discover:=true` on a bridge that should select another bridge's
remote worker automatically. Bridges publish transient-local JSON announcements
on `onnx_remote/runners`, so late-joining nodes receive the latest endpoint:

```sh
build/onnx-remote-ros2-bridge --ros-args \
  -p auto_discover:=true -p discovery_target:=tensorrt-cuda \
  -p discovery_topic:=onnx_remote/runners
```

The announcing bridge can advertise a Tailscale address or DNS name with
`advertise_host:=100.x.y.z`. Discovery is only the ROS2 control plane; tensor
payloads still use the binary `run`/`result` topics and the selected bridge's
TCP connection. Direct `remote_host`/`remote_port` remains the fallback when
discovery is disabled. Announcement selection is target-filtered but does not
provide authentication; use DDS security or a trusted ROS2 domain on shared
networks.

The bridge exposes `health` and `capabilities` for ROS2 discovery/selection;
tensor and profile data remain binary rather than being converted to ROS
messages.

`onnx-remote-mock-runner` and `onnx-remote-attach-test` provide a vendor-free
test of the compiled-artifact handshake. The mock stores opaque artifact bytes
on `load_compiled`, accepts ID-only `run_compiled`, and returns identity output
with a profile event; it is for CI protocol coverage, not model execution.

The transport also carries an optional serialized `ModelProto`. This is the
native `onnxsim::ModelExecutor` integration point: a host-side executor can
send each constant-folding submodel to a worker without Python. The worker may
run that model with native ONNX Runtime, an accelerator compiler, or translate
it to a device-specific model handle.

## External compiler contract

The simple path remains the default:

```text
RUN(op=model-runner, model, tensors) -> outputs
```

An external compiler can be selected by setting `compile_model=true` in
`RemoteExecutorOptions`. The executor then performs:

```text
COMPILE(model) -> artifact_id, artifact_bytes, manifest
RUN_COMPILED(artifact_id, optional artifact_bytes, tensors) -> outputs
```

The manifest is opaque to the transport and should describe the compiler,
target device, runtime/driver requirements, I/O ABI, shape constraints, and
legalization profile. The existing transport only bounds and carries it.

Native callers can add a legalization preflight through
`RemoteExecutorOptions::legalizer`. It receives a mutable fold-group
`ModelProto` and the configured `target`, and can rewrite it or return a short
reason for rejection. `supported_ops` then performs a dependency-free
allow-list check on the rewritten graph. For compiled execution,
`manifest_validator` runs before the artifact enters the process cache, so a
caller can reject a compiler result whose backend, ABI, chip, or legalization
profile is incompatible. These hooks intentionally leave JSON and vendor
rewrites outside the transport library.

The native executor caches compiled artifacts in memory, keyed by the exact
serialized fold-group model. This prevents repeated compilation within one
onnxsim process. The compiler/runner should own the persistent device cache:
it can validate the artifact ID against compiler version, SDK, driver, chip,
and ABI, then reuse or reject it. `send_compiled_artifact=true` is the safe
stateless default; a later load/attach handshake can send only `artifact_id`
once the worker confirms its cache.

### Standalone compiler service

`onnx-remote-compiler` is a dependency-free compiler-side service for this
contract. It is intended to run on a host with QAIRT/QNN installed while the
runner stays on a Snapdragon or AX8850 device. The service owns a persistent
on-disk cache; its key includes the serialized model, target, configured
compiler command, and explicit compiler identity. Cache files are published
atomically with a completion marker, so a second service instance cannot
observe a partial or mixed artifact/manifest pair.

```sh
cmake -S tools/onnx-remote -B build/onnx-remote
cmake --build build/onnx-remote --target onnx-remote-compiler
build/onnx-remote/onnx-remote-compiler \
  --port 39502 --target qnn-htp --compiler-id qairt-2.31.0 \
  --cache-dir /var/cache/onnxsim-qnn --max-cache-bytes 1073741824 \
  --command 'python scripts/qualcomm/qnn_compile.py --input {input} \
             --output {output} \
             --manifest {manifest} --target {target}'
```

The command is trusted local configuration, not request data. It must write the
compiled artifact to `{output}` and a bounded UTF-8 manifest to `{manifest}`;
`{input}` is the received ONNX ModelProto. A no-command service copies the
model into an artifact and is useful for validating networking and cache
plumbing before installing QAIRT. A QNN wrapper can run the converter,
backend-specific graph preparation, and context-binary generation as one
command, while keeping those SDK-version-specific details out of onnxsim.
The repository's `scripts/qualcomm/qnn_compile.py` adapter uses the
`onnxruntime-qnn` plugin to generate an embedded QNN EP-context ONNX artifact;
set `QNN_BACKEND_PATH` or let the package select its bundled HTP backend.

TensorRT can use the same compiler service contract on an NVIDIA/Jetson compile
host. The adapter records TensorRT's detailed engine-inspector layer and tactic
metadata in the manifest, while the serialized engine is the artifact sent to a
TensorRT-capable runner:

```sh
build/onnx-remote/onnx-remote-compiler \
  --port 39503 --target tensorrt-cuda --compiler-id tensorrt-10 \
  --cache-dir /var/cache/onnxsim-tensorrt \
  --command 'python3 scripts/nvidia/trt_compile.py --input {input} \
             --output {output} --manifest {manifest} --target {target} \
             --fp16'
```

The TensorRT Python environment must provide `tensorrt`, `onnx`, NumPy, and the
CUDA runtime. `scripts/nvidia/trt_harness.py` remains the reference runner and
reports mean execution time; its engine inspector data is also retained in the
compiled manifest for remote profiling. TensorRT engine blobs are generally
specific to the TensorRT/CUDA/GPU combination, so the runner should validate the
compiler ID, target, and driver/runtime ABI before loading a cache hit.

When onnxsim profiling is enabled, remote traces emit separate
`RemoteRPC/compile`, `RemoteRPC/load`, and `RemoteRPC/execute` events. Worker
events retain their accelerator category and gain a `phase` argument; compiled
artifacts add `RemoteArtifactReady` and `RemoteArtifactAttached` metadata
events. They use the same Chrome Trace/Perfetto output as local spans.

## AXCL worker

On a machine with the AXCL SDK, configure with `-DONNX_REMOTE_AXCL=ON`.
The resulting `onnx-remote-axcl-worker` accepts the `.axmodel` path as the
request operation and currently requires float32 model inputs and outputs:

```sh
cmake -S tools/onnx-remote -B build/onnx-remote \
  -DONNX_REMOTE_AXCL=ON -DAXCL_INCLUDE_DIR=/usr/include/axcl \
  -DAXCL_RT_LIBRARY=/usr/lib/axcl/libaxcl_rt.so \
  -DAXCL_SYS_LIBRARY=/usr/lib/axcl/libaxcl_sys.so
```

For the external compiler path, send `op=load_compiled`, an `artifact_id`, and
the `.axmodel` bytes once. The worker stores the artifact under `--cache-dir`;
later `op=run_compiled` requests may send only the same artifact ID. IDs are
restricted to filename-safe characters, and artifact publication is atomic.
This is the runner-side cache/load handshake used by the AX8850 path. A
stateless runner can skip the load operation and continue sending bytes with
each `run_compiled` request.

The model is loaded for each request in this first correctness-oriented
adapter.  That is deliberately simple and isolates model-load failures; a
persistent model cache should be added once the wire protocol is exercised on
the card.
