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
implements `identity` for every protocol dtype and `relu`, `add`, `mul`,
`sub`, `div`, `max`, `min`, `abs`, `neg`, `sqrt`, `exp`, `log`, and `tanh` over float32 tensors.
Binary arithmetic follows ONNX/NumPy trailing-dimension broadcasting. It is
not intended to be the production accelerator backend. An AXCL worker can
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

The reference worker also serves a dependency-free capability document over
the same transport. This is useful before sending a model or selecting a
runner discovered by ROS2/DORA:

```sh
./build/onnx-remote/onnx-remote-client --capabilities runner.local 39501
```

## ONNX Runtime Execution Provider adapter

There is an optional source-level ONNX Runtime EP in
`onnx_remote_execution_provider.{h,cpp}`. It is deliberately pinned to the ORT
source ABI: ORT's
`IExecutionProvider` and `NodeComputeInfo` are internal headers and are not in
the public prebuilt SDK. Enable it only in a build that has the matching ORT
source checkout:

```bash
cmake -S . -B build-ort-ep \
  -DONNXSIM_BUILTIN_ORT=ON -DONNXSIM_REMOTE_TRANSPORT=ON \
  -DONNXSIM_REMOTE_ORT_EP=ON \
  -DONNXSIM_ORT_SOURCE_DIR=/path/to/compatible-legacy-onnxruntime
```

This is intentionally a configure-time compatibility check. Current public ORT
source releases such as 1.29 do not expose the legacy internal header used by
this adapter, so they fail early with a precise diagnostic instead of producing
an apparently usable but ABI-incompatible build. The standalone public `OrtEp`
plugin is built separately with `ONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN=ON` and a
public ORT SDK include/library:

```bash
cmake -S tools/onnx-remote -B build-public-ep \
  -DONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN=ON \
  -DONNXSIM_ORT_PUBLIC_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXSIM_ORT_PUBLIC_LIBRARY=/path/to/onnxruntime/lib/libonnxruntime.so
cmake --build build-public-ep --target onnxsim_remote_ep
```

The plugin exports `CreateEpFactories`/`ReleaseEpFactory`, advertises the
reference remote ops, and uses the same environment variables for endpoint and
profiling configuration. The public SDK path is covered by the optional load
and remote-execution tests in the remote CMake project. When ORT profiling is
enabled, the plugin also reports the returned remote kernel events through
`OrtEpProfilerImpl`; `ONNXSIM_REMOTE_EP_PROFILE_FILE` remains available for a
transport-native JSON copy.

The provider partitions `Identity`, `Relu`, `Add`, `Mul`, `Sub`, `Div`, `Max`,
`Min`, `Abs`, `Neg`, and `Sqrt` by default and
executes float32 tensors through the dependency-free worker. Unsupported ops
remain on the normal CPU EP. `Options::profiling` forwards worker events into
onnxsim's profiler. Applications using ORT's internal provider registration
API can construct it with `onnxsim::ort_remote::CreateRemoteExecutionProvider`.
Registration maps can use `host`, `port`, `connect_timeout_ms`,
`io_timeout_ms`, `profiling=off|summary|detailed`, and comma-separated
`supported_ops`; use `OptionsFromProviderOptions` to parse them.
Only operations with a reference transport mapping are claimable; adding an
unknown name to `supported_ops` therefore leaves that node on the CPU EP.
With ORT's internal C++ session API, registration is the normal EP flow:

```cpp
onnxruntime::InferenceSession session(session_options, env);
session.RegisterExecutionProvider(
    onnxsim::ort_remote::CreateRemoteExecutionProviderFromOptions(options));
```

The public `Ort::SessionOptions` API does not expose arbitrary internal EP
objects; applications using only the public ABI should package this adapter as
an ORT plugin EP instead.
This is a software/reference EP and does not require Snapdragon, AX8850, QNN,
or TensorRT hardware.

### Optional ORT-backed graph worker

For software-only graph transport tests, build the optional worker against a
matching public ONNX Runtime SDK:

```sh
cmake -S tools/onnx-remote -B build-ort-worker \
  -DONNXSIM_REMOTE_ORT_WORKER=ON \
  -DONNXSIM_ORT_WORKER_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DONNXSIM_ORT_WORKER_LIBRARY=/path/to/onnxruntime/lib/libonnxruntime.so
cmake --build build-ort-worker --target onnx-remote-ort-worker
```

Run it with `--threads 1` (the default) for bounded CPU use, or raise the
value on a compile host with spare cores. `--max-models 2` bounds the LRU
serialized-graph session cache; set it to `0` to disable caching. Detailed
profiling reports `ort_session_cache_hit` when a graph session is reused.

The worker accepts `subgraph` or `onnx` requests containing serialized ONNX
bytes in `Request::model`, executes float32 inputs on ORT CPU, and returns
float32 outputs plus an `ort_session_run` profile event. It is optional and
separate from the dependency-free worker, so embedded targets do not inherit
an ORT or C++ runtime dependency.

It also answers the native `capabilities` request with an ORT CPU runner
manifest, so ROS2/DORA discovery can verify and select it like the reference
worker.
Graph-capable manifests set `graph_execution:true`; host dispatchers should
prefer this capability for serialized `subgraph` requests instead of building
another operator allow-list.

`onnx-remote-ort-worker-test MODEL.onnx PORT` runs the same float32 model
locally through ORT and compares every output with the remote subgraph result,
including the required profile event. For a quick manual check without the
test harness:

```sh
onnx-remote-ort-worker --port 39503 &
onnx-remote-client --subgraph 127.0.0.1 39503 MODEL.onnx 1,2,3 --shape 3 \
  --input 4,5,6@3
```

The positional values are the first input; repeat `--input VALUES[@D0,D1]`
for additional model inputs. Output shapes, values, and profile events are
printed.

## TensorRT worker (Jetson / NVIDIA GPU)

`onnx-remote-tensorrt-worker` runs graphs on TensorRT. It speaks the same wire format as
the ORT worker and handles three operations:

- `subgraph` / `onnx`: `Request::model` holds a serialized ONNX `ModelProto`. The worker
  parses it with `nvonnxparser`, builds an engine pinned to the request's input shapes
  (dynamic dimensions become min = opt = max), runs it, and returns the outputs. Engines
  are cached per (model hash, input dtypes/shapes) in an LRU of `--max-engines` (default 4).
- `engine`: `Request::artifact` is a prebuilt serialized plan, for example the output of
  `scripts/nvidia/trt_compile.py` through `onnx-remote-compiler`. Send it once under an
  `artifact_id`; later requests can send only the `artifact_id`. A plan only loads on the
  same TensorRT version and GPU it was built for.
- `capabilities`: manifest with the TensorRT version and `fp16` setting.

Inputs are matched to engine inputs by order; dtypes must match the engine exactly
(FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT32, INT64, BOOL are carried). Outputs with
data-dependent shapes are rejected. Profiling returns `trt_engine_build` or
`trt_engine_cache_hit`, plus the GPU time of `enqueueV3` as `trt_enqueue`. Errors come
back as an `ERR` response and do not stop the worker.

```sh
cmake -S tools/onnx-remote -B build/onnx-remote -DONNXSIM_REMOTE_TENSORRT_WORKER=ON
cmake --build build/onnx-remote --target onnx-remote-tensorrt-worker \
      onnx-remote-tensorrt-worker-test
./build/onnx-remote/onnx-remote-tensorrt-worker --port 39505 [--fp16] [--workspace-mb 1024]
./build/onnx-remote/onnx-remote-tensorrt-worker-test 39505   # with the worker running
```

The build needs CUDA and TensorRT headers/libraries (`TENSORRT_ROOT` can point at a
non-default install). It was built and tested on a Jetson Orin Nano with JetPack 7.2.1
(CUDA 13.2, TensorRT 10.16). The worker is single-threaded and serves one request per
connection, like the other workers. Building an engine takes seconds to minutes and blocks
other clients, so prefer prebuilt `engine` plans for anything larger than a test graph.

## Design constraints

* bounded message size and tensor count;
* complete reads/writes (short socket writes are expected);
* network byte order for all integer fields;
* no dynamic dependency on Python or protobuf;
* one request per connection for simple failure isolation;
* explicit operation and tensor metadata, so the remote side never guesses
  dtype or shape. Protocol v5 preserves raw little-endian payloads for
  FLOAT16, BFLOAT16, integer, DOUBLE, and BOOL tensors. The reference worker
  executes typed identity and keeps arithmetic intentionally float32-only.
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

CI and dependency-minimal environments can compile-check the adapter against
the checked-in API stub (this does not emulate a DORA runtime):

```sh
cmake -S tools/onnx-remote -B build/onnx-remote-dora \
  -DONNXSIM_REMOTE_DORA=ON -DONNXSIM_REMOTE_DORA_STUB=ON
cmake --build build/onnx-remote-dora --target onnx-remote-dora-node
```

The same stub can exercise the forwarding and incremental profile path against the reference worker,
without installing DORA. Set `ONNXSIM_DORA_STUB_RUNTIME=1`; it injects a
synthetic `relu` input into the node and validates the binary response,
including request correlation and output values. For example:

```sh
ONNXSIM_DORA_STUB_RUNTIME=1 \
  build/onnx-remote-dora/onnx-remote-worker --port 39501 &
ONNXSIM_DORA_STUB_RUNTIME=1 ONNXSIM_DORA_PUBLISH_PROFILE_EVENTS=1 \
  build/onnx-remote-dora/onnx-remote-dora-node
```

This is a transport/adapter integration test only; it does not emulate DORA's
actual graph scheduler or discovery daemon.

Set `ONNXSIM_DORA_REMOTE_HOST` and `ONNXSIM_DORA_REMOTE_PORT` in the node's
environment to select the native worker (defaults are `127.0.0.1:39501`). A
minimal dataflow declares `run` as the node input and `result` as its output.
The DORA node API carries raw UInt8 messages; tensor and profile serialization
remain the same as the dependency-free transport.

Set `ONNXSIM_DORA_PUBLISH_PROFILE=1` and declare an optional `profile` output
to publish the same compact JSON profile summary used by the ROS2 bridge. The
`result` output continues to carry the complete binary profile events.

The next integration layer can make an ONNX Runtime plugin EP claim a maximal
supported subgraph and send it as an operation/model handle over this
transport.  The reference worker deliberately does not pretend to be that EP
yet.

Set `ONNXSIM_DORA_ANNOUNCE=1` and declare optional `status` and `capabilities`
outputs to query and publish actual worker readiness and the worker's capability
document. If the worker is unavailable, `status` reports `unavailable` and the
capabilities output contains a short error object. The adapter also accepts
`ONNXSIM_DORA_CONNECT_TIMEOUT_MS` and
`ONNXSIM_DORA_IO_TIMEOUT_MS` for unreliable links.
Set `ONNXSIM_DORA_REQUIRE_GRAPH_EXECUTION=1` to make startup fail unless the
queried worker advertises `graph_execution:true`; this performs the capability
query even when `ONNXSIM_DORA_ANNOUNCE` is disabled.

The status output uses `schema_version: 1` and includes the selected host and
port. When the capability probe fails, its escaped error is included in both
the status object and the `ready: false` capabilities object.
With `ONNXSIM_DORA_ANNOUNCE=1`, later forwarding failures also publish an
`unavailable` status event while preserving the lossless binary error response
on `result`.

In a DORA dataflow, declare the adapter's `status` and `capabilities` outputs
and connect consumers to those outputs. DORA resolves those graph edges when
the dataflow starts; the C node API does not provide ROS-style runtime
advertisement or dynamic target selection. `status` reports worker readiness,
and `capabilities` publishes the worker's manifest for downstream selection
logic. The dependency-free DORA integration check starts the reference worker,
checks both announcement payloads, and forwards a real binary request through
the adapter.

```yaml
nodes:
  - id: remote
    path: ./onnx-remote-dora-node
    env:
      ONNXSIM_DORA_ANNOUNCE: "1"
    inputs:
      run: client/run
    outputs: [result, status, capabilities]

  - id: monitor
    path: ./runner-monitor
    inputs:
      readiness: remote/status
      capabilities: remote/capabilities
    outputs: []
```

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

The same build produces `onnx-remote-ros2-service-smoke`, a dependency-light
C++ client for validating the bridge's `health` and `capabilities` services
when a ROS installation does not include the optional `ros2 service` CLI
extension.

It also produces `onnx-remote-ros2-discovery-smoke`. Run it with one bridge
announcing `runner-a` and a second bridge configured with
`auto_discover:=true` and `discovery_target:=target` to verify that the second
bridge selects the announced runner through the transient-local discovery
topic.

`onnx-remote-ros2-failover-smoke` observes the same bridge's lease-expiry
transition. Start it while the discovery smoke topology is running, then stop
the announcing `runner-a` bridge; the observer requires both `selected` and
`expired` status events. The auto-discovering bridge then restores its
configured fallback endpoint.

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

Discovery verifies each candidate by querying its native `capabilities`
operation before selecting it (`verify_discovery:=true` by default). Set it to
false only when the announced endpoint is intentionally unavailable during
startup; the first tensor request will then perform the connectivity check.
Announcements with an unsupported `schema_version` are ignored before any
endpoint connection is attempted.
Set `require_graph_execution:=true` when the bridge must select a runner that
advertises serialized subgraph execution, such as the optional ORT graph
worker; candidates without `graph_execution:true` are rejected with a status
diagnostic. Announcements also carry an advisory `graph_execution` hint, so a
runner that explicitly reports `false` is rejected without a TCP probe
(the hint is still confirmed by verification when present).
Verified `selected` and `expired` status events include the optional
`graph_execution` boolean, allowing a UI to display the selected runner's
capability without issuing another worker request.

The announcing bridge can advertise a Tailscale address or DNS name with
`advertise_host:=100.x.y.z`. Discovery is only the ROS2 control plane; tensor
payloads still use the binary `run`/`result` topics and the selected bridge's
TCP connection. Direct `remote_host`/`remote_port` remains the fallback when
discovery is disabled. Announcement selection is target-filtered but does not
provide authentication; use DDS security or a trusted ROS2 domain on shared
networks.
When multiple announcements match, a bridge keeps the first runner while its
advertised lease is alive instead of switching endpoints on DDS delivery order.
After the lease expires, the next matching announcement is selected and the
bridge returns to its configured `remote_host`/`remote_port` if no replacement
appears.

Discovery transitions are published as JSON diagnostics on
`onnx_remote/discovery_status` by default. Override the topic with
`discovery_status_topic:=...`, or disable it with
`publish_discovery_status:=false`. Messages have `schema_version: 1` and a
`state` of `selected`, `rejected`, or `expired`, together with the runner ID,
host, and port; rejected messages also include an `error` string. This topic is
transient-local and reliable, so late-joining monitors receive the most recent
transition. Health/capability probes additionally publish `ready` or
`unavailable`. It is only a control/status surface: tensor and profile
payloads remain binary.

The bridge exposes `health` and `capabilities` for ROS2 discovery/selection;
both services query the selected worker, while tensor and profile data remain
binary rather than being converted to ROS messages. The `capabilities` service
returns the worker's manifest JSON in its `TriggerResponse.message` field.

Set `publish_profile:=true` (the default) to additionally publish a compact
JSON profile summary on the `profile_topic` (default: `profile`). The `result`
topic remains the lossless binary response, including the same profile events;
the JSON topic is intended for ROS tools and lightweight profiling UIs.
Set `publish_profile_events:=true` to additionally publish one JSON message per
event on `profile_event_topic` (default: `profile_events`). DORA provides the
equivalent `profile_event` output when `ONNXSIM_DORA_PUBLISH_PROFILE_EVENTS=1`.
These incremental surfaces are optional; the binary result remains lossless.
Native consumers can call `profile_chrome_trace_json()` on a response to emit
Chrome Trace Event Format for Perfetto or `chrome://tracing`; timestamps remain
worker-relative microseconds and therefore do not require clock synchronization.
For ROS2/DORA response payloads captured as raw bytes, the dependency-free
receiver converts them without Python or protobuf:

```sh
build/onnx-remote/onnx-remote-profile-dump \
  --input response.bin --format chrome > trace.json
```

`onnx-remote-mock-runner` and `onnx-remote-attach-test` provide a vendor-free
test of the compiled-artifact handshake. The mock stores opaque artifact bytes
on `load_compiled`, accepts ID-only `run_compiled`, and returns identity output
with a profile event; it is for CI protocol coverage, not model execution. The
mock answers `capabilities` as a non-graph runner, and the attach test verifies
that manifest before uploading. Use
`--cache-dir DIR` on the mock runner to persist artifacts across runner
restarts; `onnx-remote-attach-test HOST PORT --run-only` verifies that
restart/reload path without uploading the artifact again.

The transport also carries an optional serialized `ModelProto`. This is the
native `onnxsim::ModelExecutor` integration point: a host-side executor can
send each constant-folding submodel to a worker without Python. The worker may
run that model with native ONNX Runtime, an accelerator compiler, or translate
it to a device-specific model handle.

Graph-aware callers can use `MakeSubgraphRequest()` to set the canonical
`subgraph` operation and place serialized graph bytes in the existing
`Request::model` field. This preserves v5 compatibility while allowing a
compiler or runner to interpret the payload as ONNX `ModelProto` or a
backend-specific graph envelope.

Native consumers can use `receive_profile(response, callback)` to consume the
lossless profile events directly. The callback is independent of JSON and can
forward each event to an ORT profiler, ROS2/DORA stream, browser transport, or
an embedded aggregate without allocating a complete trace string.

## Optional gRPC schema

For hosts that need generated RPC clients, `proto/onnxsim_remote.proto` defines
a small `OnnxSimExecutor` service for compile, artifact load, execute, and
capability queries. It intentionally does not make protobuf/gRPC a dependency
of the native worker: generated bindings belong in an optional gateway or
compiler/runner service. The messages use the same raw little-endian tensor
layout and bounded profiling fields as the v5 binary transport.

This is a lighter contract than KServe V2 and is the preferred API for
onnxsim-specific compile/load/run workflows. A later KServe adapter can map
`ModelInfer` to `Execute` at the gateway, while ROS2 or DORA remains the
discovery/control plane. Keep model and artifact bytes in their dedicated RPC
fields; `parameters` is reserved for small options. The gateway forwards
`Compile` to a separate `--compiler-host`/`--compiler-port` endpoint when
configured, mirroring the native executor's compile/runner split, and reports
the selected worker's own `supported_ops` from its capability manifest.

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

For a dynamic ONNX model whose backend requires static artifacts, enable
`compile_per_static_shape`. The native executor then keys its in-process cache
by the serialized model plus each input's ONNX dtype and concrete dimensions,
and sends those input descriptors to the compiler. External compiler commands
can consume the descriptors through the `{shapes}` JSON placeholder. The
runner still receives the concrete tensors at execution time; omit this option
when the compiler produces one genuinely dynamic artifact.

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
onnxsim process. Set `max_cached_models` to a positive value when dynamic
shape specialization could produce many variants; eviction is LRU and is
local to that executor instance, so reusing a specialization promotes it.
The compiler/runner should own the persistent device cache:
it can validate the artifact ID against compiler version, SDK, driver, chip,
and ABI, then reuse or reject it. `send_compiled_artifact=true` is the safe
stateless default; a later load/attach handshake can send only `artifact_id`
once the worker confirms its cache.
When `send_compiled_artifact=false`, `attach_compiled_artifact=true` is
required; otherwise the executor rejects the configuration before issuing a
compiled run instead of sending an artifact-less request to a stateless
runner.

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

The dependency-free bundle installs the compiler service, reference worker,
mock runner, client, and attach-test utilities under `bin/`; it does not
require Python, protobuf, or gRPC on the execution host.

The command is trusted local configuration, not request data. It must write the
compiled artifact to `{output}` and a bounded UTF-8 manifest to `{manifest}`;
`{input}` is the received ONNX ModelProto. `{target}` and `{compiler_id}` are
the configured backend identity values, shell-quoted like the file paths.
`{shapes}` is a JSON array of input dtype/shape descriptors for static-shape
specialization. A no-command service copies the
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
request operation. Each tensor must carry the model's exact ONNX dtype and byte
size: float32 travels in `data`; fp16, bf16, int8/int16/int32/int64 and the
other integer and double types travel as raw little-endian bytes in
`raw_data`. Outputs come back with the dtype the engine reports, so an LLM
layer's bf16 hidden state and KV cache round-trip unchanged:

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

The worker answers `capabilities` as a non-graph compiled-artifact runner
(`load_compiled`, `run_compiled`, `run`, `io_info`, `unload`, `run_resident`,
`reset_state`), so ROS2/DORA
discovery can verify it and capability-gated dispatch can reject it when graph
execution is required. The manifest also carries `model_cache`, `max_loaded`,
the current `loaded` count, `resident_state` (whether `run_resident` can be
used: true unless `--no-cache`) and `row_copy` (`device` or `host`).

### Model cache

A model is loaded once and kept: the engine model, its context, the IO
description, the IO object and one device buffer per input and output stay
allocated between requests, so a request for a loaded model is an upload, an
execute and a download. That is what makes a host-side LLM decode loop usable
(31 models per token for a 30-layer model; see
`docs/axera-llm-rpc-decode.md`).

- The cache key is the model path exactly as the request spelled it (for
  `run_compiled`, the artifact's file under `--cache-dir`). Two spellings of
  one file are two entries.
- Each use checks the file's modification time and size; a model whose file
  changed or disappeared is unloaded and loaded again (or fails), never run
  stale. Re-sending bytes with `load_compiled` therefore replaces the model.
- `--max-loaded N` (default 64) bounds the number of loaded models; the least
  recently used one is unloaded to make room. If a load fails while other
  models are loaded (the card can run out of memory before `N`), the worker
  unloads least recently used models one at a time and retries. A path that
  is not a file fails at once and leaves the cache alone.
- A model whose upload, execute or download fails is dropped, so the next
  request loads it afresh. Request errors (input count, dtype, size) keep it.
- `--no-cache` restores the original behaviour: load, run and unload on every
  request. No resident state exists then (see below).
- SIGINT/SIGTERM unload everything and finalize the runtime before exiting.

With `Summary` or `Detailed` profiling a request that had to load the model
carries an `axcl_load` event; `Detailed` adds `axcl_upload` and
`axcl_download` around `axcl_execute`.

### `io_info`, `unload` and `run`

These three ops name their model by `artifact_id` (the cached artifact) or,
with an empty artifact ID, by a path sent as UTF-8 text in the request's
`model` bytes, since the op field holds the op name.

- `io_info` loads the model (into the cache) and returns a JSON manifest, so a
  client can pack tensors without hard-coding names, order, dtypes or sizes:

  ```json
  {"schema_version": 1, "model": "/models/llama_p128_l0_together.axmodel", "cached": true,
   "inputs": [{"name": "K_cache", "dtype": 10, "dtype_name": "FLOAT16", "axcl_dtype": 0,
               "shape": [1, 255, 192], "bytes": 97920}],
   "outputs": [{"name": "output", "dtype": 10, "dtype_name": "FLOAT16", "axcl_dtype": 0,
                "shape": [1, 1, 576], "bytes": 1152}]}
  ```

  Entries are in engine order. `dtype` is the ONNX `TensorProto.DataType` a
  request must use for that input and the one an output comes back with,
  from the same engine-to-ONNX mapping the run path checks against;
  `axcl_dtype` is the engine's own code. An engine type with no ONNX
  counterpart is reported as `"dtype": 0, "dtype_name": "UNSUPPORTED"` and
  such a model cannot be run. `llm_build` layers report their bf16 K/V/hidden
  tensors with engine type 0, which the worker reads as FLOAT16, and `mask` as
  BFLOAT16; the payload is the same 16-bit pattern either way.
- `unload` drops one model from the cache and answers
  `{"schema_version": 1, "unloaded": 1, "loaded": 3}`; with neither an
  artifact ID nor a path it drops every model. Unloading a model that is not
  loaded is not an error (`"unloaded": 0`).
- `run` runs the model at the path in `model`. It is the path-as-op request
  form for paths longer than the op field's 128 bytes.

### Resident inputs: `run_resident` and `reset_state`

A cached model's device buffers stay allocated between requests, so an input
can simply stay where it is. That is what an LLM decode loop wants for a
layer's KV cache: each call changes one row of a `[1, cache_len, kv_dim]`
tensor, and the original request form uploads the whole of it every time.
`run_resident` expresses three things per request, without changing the v5
wire format:

| | how it is said |
| --- | --- |
| input `i` stays as it is on the card, nothing is sent for it | the tensor for input `i` has no elements (shape `[0]`), under the input's dtype |
| after execution, output `j` is copied into input `i` at byte offset `slot` x (byte size of output `j`) | control line `write j i slot` |
| output `j` is not returned | control line `omit j`; it comes back as a tensor with no elements |

The model is named as for `run` (artifact ID, or a path in the `model` bytes).
The control is UTF-8 text in the request's `artifact` bytes, one directive per
line, at most 4096 bytes:

```
state 1791538852693001
write 0 0 17
write 1 1 17
omit 0
omit 1
```

`state <id>` names the resident state the request continues. The response's
manifest is `{"schema_version": 1, "state_id": <id>}`.

- `reset_state` (same naming of the model) loads the model if needed,
  zero-fills every input buffer on the card and returns a new state id: the
  start of a conversation, with nothing uploaded.
- `run_resident` without a `state` line establishes a state from the request:
  every input has to be sent in full, and the response carries the new id.
- `run_resident` with `state <id>` uploads the inputs that are sent, keeps the
  empty ones, and answers with the same id.
- A `write` needs an output and an input of the same engine dtype, the
  input's size a multiple of the output's, and `slot` inside it. The copy
  happens after execution and before the outputs are downloaded, so a row can
  be both written and returned.
- The whole request is checked before anything is uploaded: a refused request
  (count, dtype, size, a bad directive) leaves the state as it was.

**Lost state.** The state belongs to the cache entry. It ends when the entry
does (eviction by `--max-loaded` or by a failed load's retry, a changed or
missing file, `unload`, a device failure, a worker restart), when the model
is run through any of the original request forms (which overwrite every
input), and when another `reset_state` or stateless `run_resident` replaces
it. A request whose `state` is not the entry's current one, or that sends an
empty input without a `state`, is refused with an error that starts with
`resident state lost` and never runs on whatever the buffers hold; the client
then sends every input in full. State ids start from the clock at worker
start, so an id from before a restart cannot match. With `--no-cache` no
model stays loaded: both ops are refused with the same error, and the
capabilities say `"resident_state": false`.

**Two clients.** There is one state per model entry and the worker serves one
request at a time. Two clients that use different models do not interact. Two
clients that use the same model each invalidate the other's state when they
reset or re-establish it: each then gets `resident state lost` and uploads
again. They never read each other's cache, but they make no progress faster
than the full-upload form, so give each conversation its own copy of the
model files (the cache key is the path).

**Where the row copy runs.** By default (`--row-copy device`) it is one
`axclrtMemcpy(input + offset, output, row, AXCL_MEMCPY_DEVICE_TO_DEVICE)` per
`write`: the row never comes to the worker's memory. The whole-buffer form of
that call is what `scripts/axera/vm/axcl_batch_runner.c` and the resident
training runners use on the card; a destination that points *into* a device
buffer has not been run on a card yet (see "what was checked" below).
`--row-copy host` avoids it: the worker keeps its own copy of each input,
downloads the row, patches the copy and uploads the whole input again. That
uses whole-buffer copies only and still saves the network transfer, but not
the upload from the worker to the card.

**Why this form.** The Hexagon worker already keeps recurrent state on the
device, and its idiom is reused where it fits: an input sent empty means "the
value the device holds", and the output that is left on the device comes back
empty. Its state pairs are fixed by the compiled program (`program.txt`), with
the whole output looped into the whole input, so nothing else needs saying.
Here the request has to name an output, an input and a slot, and has to say
which state it believes the worker holds, so those travel as a small explicit
control. It is text rather than JSON because the worker has no JSON parser,
and it rides in `artifact` because that field is otherwise unused by a run
that names a model already on the worker. Old clients never send
`run_resident` and old request forms are answered exactly as before (the only
change they see is internal: inputs are now all checked before the first is
uploaded).

With `Detailed` profiling a `run_resident` request adds an `axcl_row_copy`
event between `axcl_execute` and `axcl_download`.

`tools/onnx-remote/python/onnx_remote_client.py` is a numpy-only Python client
for the wire protocol (`Client.run`, `run_path`, `io_info`, `unload`,
`load_compiled`, `run_compiled`, `capabilities`, `reset_state`,
`run_resident` with `keep(dtype)` for an input that is not sent); it works
against every worker here.

`remote_axcl_worker.cpp` can be syntax-checked without the SDK against the
declarations in `test/axcl_stub/axcl.h` (the command is in that header). The
stub has no definitions and is not on the real build's include path.

What was checked for `run_resident` / `reset_state`, and what was not: the
worker compiles with `-Wall -Wextra` against the stub and against the AXCL
SDK's own headers (syntax only). Linked against a scratch fake AXCL (host
memory as device memory, outputs a hash of all input bytes), it was driven by
the Python client: 36 resident calls on three models with SmolLM2-135M's layer
shapes matched the full-upload form bit for bit in both `--row-copy` modes,
and every lost-state case above was produced (eviction, changed file,
`unload`, old-form run, another client's reset, `--no-cache`). It has **not**
been built against the real runtime or run on a card.

## Allwinner VIPLite worker (Vivante VIP9000 NPU: A733, T527, ...)

`remote_viplite_worker.cpp` is the runner for NBG (`.nb`) artifacts through the VIPLite runtime on an Allwinner SoC; build it with
`-DONNX_REMOTE_VIPLITE=ON -DVIPLITE_INCLUDE_DIR=... -DVIPLITE_LIBRARY=...` (NDK for Android). It keeps networks resident after
`load_compiled`, converts FLOAT tensors to and from the NBG's own quantization, and listens on loopback by default. `--bench FILE.nb`
is a no-network smoke test. The compiler side is `scripts/allwinner/compile_nbg.py` (Acuity `pegasus` behind `onnx-remote-compiler`).
Build, deploy, measured numbers and what is and is not verified: `scripts/allwinner/README.md`.

## Hexagon cDSP worker (tinygrad-generated v65 programs)

`remote_hexagon_worker.cpp` is the runner for `tghx-v65` artifacts: whole ONNX models compiled by the tinygrad fork into one
standalone Hexagon v65 program (the Snapdragon 845's cDSP class; newer cDSPs run it too). The compiler side is
`scripts/android/tinygrad_hexagon_bridge/openpilot_v65/compile_v65.sh`, a `--command` for `onnx-remote-compiler`. It runs the
tinygrad capture, checks the emitted program bit for bit under qemu, builds the FastRPC skel, and packs the skel, the weights
and a plain-text I/O contract. The artifact is a small little-endian container, so the worker needs no JSON or archive library.

```sh
onnx-remote-compiler --port 39502 --cache-dir ~/.cache/onnxsim-v65 --target hexagon-v65 \
  --compiler-id "tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)" --command '.../openpilot_v65/compile_v65.sh {input} {output} {manifest}'
.../openpilot_v65/build_worker.sh out/          # Android aarch64: NDK + the Hexagon SDK's qaic and libcdsprpc
CLIENT=build/onnx-remote/onnx-remote-client .../openpilot_v65/e2e.sh out/onnx-remote-hexagon-worker model.onnx \
  --input-raw 2:1,1382400:img.bin --input-raw 1:1,3:calib.bin --dump out.bin --iters 5
```

The compiler's cache does not see the tinygrad checkout behind the command, so include its commit in `--compiler-id`.
`load_compiled` writes the skel as `tg_graph_<id>.so` into `--cache-dir` (put on `ADSP_LIBRARY_PATH` at startup), opens it in
an unsigned PD, and uploads the weights once in 8 MB chunks. Several programs can be resident at once. `run_compiled` takes
the ONNX inputs in graph order (FLOAT or UINT8, exact sizes), leaves out inputs the program never reads, and returns every
ONNX output as float32 in graph order. `--threads N` overrides the DSP thread count baked into the program. `Summary`
profiling reports the DSP-side run time; `Detailed` adds one event per kernel call. `onnx-remote-client --compile-run` drives
compile, `load_compiled` and `run_compiled` against separate compiler and runner endpoints and can compare or dump the outputs.

### v68+ cDSPs (`tghx-v69`: qfloat HVX, HMX)

The same worker runs programs built for newer cDSPs. The artifact is the same container with the `tg_graph` ABI, and its
magic names the arch (`TGHXV69\0`). `scripts/android/tinygrad_hexagon_bridge/onnx_remote/compile_v69.sh` is the compiler
command. It drives the tinygrad fork's `examples/dsp_graph_onnx.py` (branch `bevformer-onnx-remote`, commit `73f3ef25e` at the time of writing), which captures any
ONNX model's kernels without running them, since qemu can't run qfloat or HMX code. It then emits them with the v65 path's
emitter and skel, extended for HMX: the skel votes HMX power, takes VTCM + the HMX context for each run and holds the HMX
lock on its graph thread. Kernels and skel are built with the Hexagon SDK's `hexagon-clang`.

```sh
HMX=1 QDQ_HMX=1 HMX_VTCM_KB=4096 TINYGRAD_ROOT=... HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... \
onnx-remote-compiler --port 39503 --cache-dir ~/.cache/onnxsim-v69 --target hexagon-v69 \
  --compiler-id "tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)-hmx4096" --command '.../onnx_remote/compile_v69.sh {input} {output} {manifest}'
CLIENT=build/onnx-remote/onnx-remote-client COMPILER_PORT=39503 .../openpilot_v65/e2e.sh out/onnx-remote-hexagon-worker backbone1.q8.onnx \
  --input-raw 2:1,480,800,3:img.bin --dump out.bin --iters 5 --profile
python3 .../onnx_remote/check_outputs.py backbone1.q8.onnx out.bin --input-raw 2:1,480,800,3:img.bin
```

The compiler service's environment is part of the program, so put it in `--compiler-id`: `HMX=1 QDQ_HMX=1` lowers int8 QDQ
graphs onto HMX, `HMX_VTCM_KB` sets the tile pool, and `DSP_THREADS` (default 4) sets the HVX thread pool. HMX kernels always
run on the graph thread. There is no compile-time check, so `check_outputs.py` compares the dumped outputs with
onnxruntime. Measured on a Xiaomi 12S (V69, 4 threads):

| model | load | run, DSP / RPC-inclusive | vs onnxruntime |
|---|---:|---:|---|
| BEVFormer-tiny int8 backbone, one camera (`backbone1.q8.onnx`, HMX) | 0.81 s | 308 / 347 ms | 0 of 96000 differ |
| BEVFormer-tiny encoder, 1 layer (`enc1.sim.onnx`, fp32, GridSample) | 0.29 s | 319 / 564 ms | max rel 8.0e-7 |
| BEVFormer-tiny encoder, 3 layers (`enc3.sim.onnx`) | 0.47 s | 959 / 1198 ms | max rel 3.5e-6 |
| BEVFormer-tiny decoder (`decoder.sim.onnx`, fp32, 6 GridSamples) | 0.71 s | 1151 / 1234 ms | max rel 3.1e-6 |
| one 3x3 float conv, 64 -> 64 channels, 30 x 50 | 56 ms | 351 / 376 ms | max rel 8.6e-7 |

The fork's ONNX importer has `GridSample` (2-D bilinear, zero padding) for these. The driver sets layout choices that keep the
generated kernels' vector lanes on unit-stride axes; each is a tinygrad env flag, off in tinygrad itself, on in
`dsp_graph_onnx.py` (`FLAG=0` turns one off). Each row shows the cumulative DSP time (decoder / encoder layer / 3-layer
encoder) after adding that change, in the order they were added:

| change | decoder | encoder layer | 3-layer encoder |
|---|---:|---:|---:|
| none (first working build) | 25.4 s | 11.3 s | |
| `GEMM_PRETRANSPOSE`: constant `transB=1` Gemm weights stored transposed (kept (N, K), the output lanes read them one scalar at a time) | 3.39 s | 0.91 s | 3.55 s |
| `SOFTMAX_CONTIGUOUS` + `HVX_UPCAST_ODD`: softmax not fused into P @ V; an axis like 900 takes a 100-lane upcast (largest divisor up to the lane budget) instead of 4 lanes, on unit-stride axes only | 1.84 s | 0.96 s | 3.71 s |
| `MATMUL_CONTIGUOUS` + `TRANSPOSE_CONTIGUOUS`: Gemm products and Transposes realized, so consumers don't read them through strided views | 1.19 s | 0.39 s | 1.23 s |
| `REDUCE_TWO_STAGE`: last-axis reductions (LayerNorm, softmax) as 32-lane partial sums, then the 32 | 1.12 s | 0.32 s | 0.97 s |

Tried and not kept: `MATMUL_CONTIGUOUS` alone slowed the decoder (1.84 -> 3.18 s: Q @ K^T read K through the transpose view)
until `TRANSPOSE_CONTIGUOUS` came with it; register blocking over a matmul's rows (5 rows x 128 lanes per step) made the Gemms
and attention much slower (decoder 7.8 s). The float paths are still far from the hand-written MSDA pipeline
(`scripts/android/vision_models/bevformer_tiny/msda_hvx`, about 90 ms per frame): the largest kernels now are the
fp32 Gemms at about 12 GMAC/s.

**fp16 Gemms on the HMX (`GEMM_FP16=1`, opt-in).** In the compiler service's environment, every fp32 Gemm runs on the HMX in
fp16: the operands are cast to fp16 and padded to whole 32 x 32 tiles, HMX accumulates internally, and the product is rounded
to fp16 once before the cast back and the bias add in fp32. It changes the numerics, so compare with a looser threshold
(`check_outputs.py ... --rtol 2e-3`):

| model | fp32 (HVX) | `GEMM_FP16=1` (HMX) | vs onnxruntime with `GEMM_FP16=1` |
|---|---:|---:|---|
| decoder | 1151 ms | 581 ms | max rel 6.0e-4, cos 1.000000 |
| encoder, 1 layer | 319 ms | 143 ms | max rel 3.3e-4, cos 0.9999998 |
| encoder, 3 layers | 959 ms | 450 ms | max rel 1.2e-3, cos 0.99999994 |

A 2500x256x256 Gemm takes about 1.1 ms this way (0.7 ms of it on the HMX), against about 14 ms on the HVX in fp32. Getting
there took several fixes in the fork; each of these was slow and/or wrong on the phone:
- the casts fused into the HMX kernel's tile packing;
- the cast back fused into its output epilogue;
- tinygrad's own padding of a tensor-core axis;
- the pad done in the same kernel as a fused ReLU;
- the v65 qemu mock's fp16 workarounds leaking into v69 captures.

One bug remains in the fork's HMX lowering: a single 32-row M tile with several K tiles gives wrong results (cos 0.71, in
hexagon-sim too). `GEMM_FP16` avoids it by padding M to at least 64 rows.

**The hand MSDA kernel as an ONNX op (`onnxsim::MSDA`).** `scripts/android/vision_models/bevformer_tiny/msda_hvx/export_msda_op.py`
exports the encoder and decoder with each deformable-attention span (sampling grid -> GridSample -> attention weights -> sum
over points -> average over visible cameras) as one `onnxsim::MSDA` node. It checks that export against `model.py`'s own
forward in torch. The fork's ONNX importer lowers that node on the DSP to one call of the hand-written kernel
(`msda_kernel.h`, its HVX qf32 body), with the queries split across `DSP_THREADS`; on other devices it uses GridSample. The rest
of the graph is the usual generated code. onnxruntime has no MSDA, so `check_outputs.py --ref-model` takes the reference from the
plain export (`<piece>.sim.onnx`) on the same inputs. With `GEMM_FP16=1` too:

| model | GridSample (above) | `onnxsim::MSDA` | vs onnxruntime on the plain export |
|---|---:|---:|---|
| decoder (`decoder.msda.onnx`) | 581 ms | 528 ms | max rel 6.0e-4, cos 1.000000 |
| encoder, 1 layer (`enc1.msda.onnx`) | 143 ms | 77 ms | max rel 3.3e-4, cos 0.9999998 |
| encoder, 3 layers (`enc3.msda.onnx`) | 450 ms | 266 ms | max rel 8.4e-4, cos 0.99999994 |

```sh
python3 .../bevformer_tiny/msda_hvx/export_msda_op.py enc3 --ckpt <checkpoint.pth> --work work
python3 .../onnx_remote/check_outputs.py work/enc3.msda.onnx out.bin --input-raw ... --ref-model work/enc3.sim.onnx --rtol 2e-3
```

**Attention on the HMX, with hand-written glue.** With `GEMM_FP16=1`, MatMul (batched too) also takes the fp16 HMX path. That
alone made the decoder slower (528 -> 1194 ms). The two HMX products of each self-attention layer took 11 ms, but the glue
around them was generated and slow. Zero-padding the 8 x 900 x 900 probabilities to whole tiles took 129 ms, and the softmax
read the padded product through a slice view. Two hand-written HVX kernels in the fork replace that glue on the DSP:
- a row copy for the zero pad;
- an fp16 softmax: when a last-axis Softmax reads a `GEMM_FP16` product, it runs on the padded fp16 product and writes the
  padded fp16 probabilities that the next product (P @ V) takes directly.

Each decoder layer's attention glue went from about 45 ms to 3.6 ms. Two more changes help the encoder:
- `HVX_UPCAST_ELEMWISE`, now a default in `dsp_graph_onnx.py`: the fp16 cast of temporal attention's concat(prev_bev, query)
  ran fully scalar, 17 ms per layer;
- a third hand kernel for softmax over 2, 4 or 8 floats (the sampling points' weights). It deals 32 groups into R vectors,
  computes lane-wise and shuffles back. The generated max and sum read each group at stride R, which HVX does as scalar gathers.

DSP time with `onnxsim::MSDA` and `GEMM_FP16=1`, each column adding to the one before:

| model | MSDA | + attention glue | + `HVX_UPCAST_ELEMWISE` | + small softmax | vs onnxruntime on the plain export |
|---|---:|---:|---:|---:|---|
| decoder | 528 ms | 165 ms | 164 ms | 156 ms | max rel 9.8e-4 |
| encoder, 1 layer | 77 ms | 77 ms | 57 ms | 40 ms | max rel 3.2e-4 |
| encoder, 3 layers | 266 ms | 235 ms | 162 ms | 118 ms | max rel 1.2e-3 |

The attention softmax computes exp in fp16 and the row sum in qf32. Probabilities below 2^-13 of the row's largest become 0. On
hexagon-sim its maximum absolute error is 1e-3 against fp32 on the same fp16 inputs.

**Fused attention in VTCM.** After the glue above, the decoder's attention was still about 14 ms per layer. Most of it was P @ V
(8 ms), which packed each of P's 6728 activation tiles from DRAM. A hand kernel in the fork
(`test/external/dsp/hand/attn/`) keeps everything between the two products in VTCM:
- per head, K^T and V are packed into VTCM tiles once;
- per 32-query block, the HMX writes S = Q K^T into VTCM tiles;
- the softmax runs on those tiles in place, in the HMX tile layout (one vector is an interleaved row pair);
- the tiles are then P @ V's activation tiles as they are, in one `:deep` span.

An HMX call runs on the graph thread, so the kernel is split per head into three calls: an HMX call for S, the softmax on the
4-thread HVX pool, and an HMX call for P @ V. The pooled call gets the VTCM address through a small token buffer, because a
kernel that mentions the HMX runs under its lock, off the pool. Each query block's tiles sit in their own 64 KB VTCM slot,
because an HMX operand span that crosses a 256 KB window page-faults on the phone. On hexagon-sim, at 8 heads x 900 x 900, the
fused single-call form takes 12.3 M cycles: 3.0 M for the HMX products and packing, the rest for the single-thread softmax.
`HMX_ATTN=1` keeps that form; `HMX_ATTN=0` turns both off.

| model | before | fused, one call (`HMX_ATTN=1`) | split, softmax on 4 threads (default) | vs onnxruntime on the plain export |
|---|---:|---:|---:|---|
| decoder | 156 ms | 115 ms | 88 ms | max rel 1.15e-3 |

The encoders have no Q K^T attention and are unchanged (41 / 111 ms).

The capture runs no kernels, so any value the importer reads on the host must exist before the capture. `dsp_graph_onnx.py`
reads the small constants' values first (Gather indices, Split sizes). Before this, the 3-layer encoder's scalar Gather indices
on its merged `value_proj` output all read as 0, so every layer sampled layer 0's camera maps (cos 0.59).

**Hand kernels replaced by generated code.** Three of the hand kernels above are now tinygrad-generated by default; the
`HAND_*` variables bring the hand ones back.
- *Small softmax* (`HAND_SOFTMAX_GRP=1` for the hand kernel). The group max and sum over 2, 4 or 8 floats are spelled as
  elementwise ops over the R lanes instead of reductions. That gives two elementwise kernels, `e = exp(x - max)` and
  `e * 1 / sum`, with the groups read as register shuffles. The reduce form computed the exp twice. On hexagon-sim, 160K
  floats: R=2 0.98 M cycles (hand 1.14 M), R=4 1.06 M (0.82 M), R=8 1.22 M (0.64 M).
- *Row pads before an HMX product* (`HAND_ROWCOPY=1`). They are a plain pad, as fast as the hand copy. A column pad that ends
  inside a 64-lane vector (900 -> 928 columns) was 60x slower generated and still takes the hand copy; BEVFormer has none
  left, because the attention takes K^T unpadded.
- *The attention's VTCM state*. The split attention keeps S in VTCM across three calls, which the scheduler can't see. An
  unrelated HMX Gemm placed between them overwrote the tiles (decoder cls off by 13%). `dsp_graph_onnx.py` now keeps each
  head's calls adjacent.

An HMX A operand read once (P @ V without the fused attention) now gets its whole K panel prefetched; before, the prefetch
fetched the tile 32 rows down. That took P @ V from 7.3 to 6.2 ms per call.

| model | hand kernels | generated (default) | vs onnxruntime on the plain export |
|---|---:|---:|---|
| decoder | 88 ms | 84.0 ms | max rel 1.15e-3 |
| encoder, 1 layer | 41 ms | 40.8 ms | max rel 3.2e-4 |
| encoder, 3 layers | 111 ms | 111.5 ms | max rel 1.2e-3 |

Still hand-written:
- *The attention.* Generated (`HMX_ATTN=0`), the decoder takes 136 ms, because S and P make a 13.8 MB round trip through
  DRAM per layer.
- *MSDA.* Generated (`HAND_MSDA=0`, the GridSample form), the decoder's sampling is 1.8 ms per call against 1.6 by hand
  (decoder 87.6 ms). The encoder's camera sampling (6 maps, 8 points, visibility mask) is 36.9 ms against 6.3 ms, and its
  temporal sampling 12.4 against 7.1 ms. A form split like the hand kernel was also tried:
  - one elementwise kernel computes each tap's row and weight;
  - a second gathers one 128-byte row per tap.

  Its gather alone took 22.8 ms. The hand kernel skips the cameras a query doesn't see and the taps that fall outside the map.
  A BEVFormer query sees 1-2 of the 6 cameras, and a static generated kernel samples all 6 with weight 0. Also, the vector
  floor / float -> int helpers (`HVX_REVEC_CMP=1`) gave wrong results on the phone: there HVX float math is qf32, which
  hexagon-sim doesn't model.

**Intermediates in VTCM, placed by the graph.** A kernel can't keep data in VTCM for a later kernel, because it doesn't know
what runs in between. The graph emitter does know:
- each buffer's live range over the call order;
- which calls use the HMX tile pool;
- how each kernel reads and writes each buffer (from its source).

It now places intermediates in a VTCM arena past the tile pool (`DSP_GRAPH_VTCM_KB`), which no kernel touches. Placement is
first fit over live ranges, with the most calls per byte first. Only buffers that some single call writes whole qualify, and
only if every kernel touches them as whole HVX vectors or through the HMX pack/store helpers. VTCM has no L1, so a buffer
read with 16-byte `float4` loads ran 40% slower there and stays in DDR. Hand kernels' buffers stay in DDR too.

The skel acquires the tile pool plus the arena on every run. A cDSP that grants less runs those buffers from one DDR block:
the results are the same, only slower. `dsp_graph_onnx.py` sizes the arena to what the tile pool leaves of
`DSP_VTCM_TOTAL_KB` (default 8192, the V69's). A 3 MB tile pool (`HMX_VTCM_KB=3072`) with a 5 MB arena measured best; the
split attention needs 2.1 MB of the pool.

The same round fixed two issues found along the way:
- A threaded kernel whose buffer numbering skips one declares its core id before its last buffer. The emitter used to pass
  the core id last, which faulted on the phone.
- A reduction over 960 fp16 values with a per-column bias took the rows as its vector lanes, which made every load a
  2-byte gather.

| model | before | VTCM arena (`HMX_VTCM_KB=3072`) | vs onnxruntime on the plain export |
|---|---:|---:|---|
| decoder | 84.0 ms | 61.7 ms | max rel 1.15e-3 |
| encoder, 1 layer | 40.8 ms | 34.0 ms | max rel 3.2e-4 |
| encoder, 3 layers | 111.5 ms | 101.3 ms | max rel 1.2e-3 |

`HMX_ATTN=3` runs the decoder attention as generated kernels, one head at a time, so a head's S^T and P^T fit the arena:
- S^T = K Q^T on the HMX;
- the softmax over S^T's rows, as vector reductions with an fp16 exp computed once;
- O^T = V^T P^T on the HMX.

It passes but takes 199 ms against 62 ms for the hand attention. About 13 ms per layer is the heads themselves; another
~15 ms per layer goes to the q / k / v transposes and assembling the heads, which codegen rendered as scalar copies.

A transposing copy now upcasts both of its axes by 32 (`HVX_TRANSPOSE`). Loads and stores are whole rows, and the transpose
happens in registers. On hexagon-sim a per-head 900 x 32 float -> 32 x 960 fp16 copy went from 11.0 M to 1.9 M cycles; its
values are within one fp16 ulp, because the vector float -> half conversion rounds a few values differently. The kernel
spills a 97 KB frame, which overran the 96 KB pool-thread stacks and faulted the PD, so the stacks are now 192 KB. With this,
`HMX_ATTN=3` takes the decoder from 199 to 174 ms. The default path has no such copy and is unchanged.

The same phone ran ~4% slower on the second day for identical programs, so compare runs from one session only.

**Threading the kernels the pool skipped.** A small phone-side test loop found two kinds of kernel running on one thread: a
one-op ONNX model run through onnx-remote, like the full models, and then compiled locally to read the kernel.
- *No axis divides by the pool size.* BEVFormer's LayerNorm second stage reduces 2500 rows, and after a 4-row upcast 625 = 5^4
  are left, which no thread count from 2 to 4 divides. The pool already strides a kernel's slices over its threads, so such a
  kernel now takes more slices than threads when they spread evenly (25 slices over 4 threads, 89%): 0.7 -> 0.2 ms per call.
- *Few elements, each expensive.* The threading threshold counted elements as if each were cheap. It is now weighted by the
  kernel's arithmetic, with transcendentals and reciprocals counted 20 times. The decoder's inverse sigmoid of its reference
  points (2700 values, an exp, two reciprocals and a log polynomial each, scalar) took 1.2 ms on one thread.

Both apply to v68+ programs only. Also in this round: a pure splat of a loaded lane no longer goes through the lane gather (a
regression from the lane-gather commit that the unit tests caught). A butterfly transpose network for the columns of a 32 x 32
block of vectors is in place too: 1.0 M -> 67 K cycles for a 2560 x 32 row sum on hexagon-sim. BEVFormer's row sums have 2500
or 900 rows, though, so it isn't used in these models yet.

| model | before | threading (`HMX_VTCM_KB=3072`) | vs onnxruntime on the plain export |
|---|---:|---:|---|
| decoder | 63.5 ms | 59.3 ms | max rel 1.15e-3 |
| encoder, 1 layer | 35.8 ms | 32.5 ms | max rel 3.2e-4 |
| encoder, 3 layers | 105.8 ms | 95.4 ms | max rel 1.2e-3 |

The "before" column is the same day's run of the previous fork commit.

**Transposes, call order and VTCM placement.** Another round, again measured with one-op micro models on the phone:
- *Odd-sized transposes.* A transposing copy whose swapped axis isn't a multiple of 32 (the camera maps, 6 x 256 x 375 ->
  6 x 375 x 256) now runs in three parts: the multiple-of-32 part and the last 32 (overlapping), each through the 32 x 32
  transpose network, then a concatenation. 6.2 -> 0.61 ms. The concatenation needed one more renderer fix: a select whose
  condition is the same scalar in every lane is now one vector select, not 128 lane selects (2.1 -> 0.2 ms).
- *VTCM ranking.* Candidates are ranked by uses per byte-call of VTCM held, so small buffers read twice 200 calls apart no
  longer block the arena.
- *Call order.* The capture runs every kernel as early as possible, so a producer near the start could feed a consumer 70 calls
  later. A pass now sinks each call to just before its first dependent, comparing byte ranges within buffers.
- *Buffer reuse.* tinygrad's own memory planner aliased unrelated tensors, which created false dependencies between calls. It is
  now off (`NO_MEMORY_PLANNER=1`), and the emitter plans DDR reuse by liveness after the reordering (`DSP_GRAPH_REUSE=1`). The
  3-layer encoder's scratch went from 108 to 37 MB.
- *Hand kernels in VTCM.* A hand kernel can declare which of its buffers it reads only as whole HVX vectors
  (`/* vtcm-params: 1 */`). MSDA declares its value maps, and those are placed first. Per MSDA call: decoder 1.6 -> 0.83 ms,
  two-frame temporal 6.5 -> 3.2 ms, cameras 6.0 -> 4.3 ms.

| model | before | now (`HMX_VTCM_KB=3072`) | vs onnxruntime on the plain export |
|---|---:|---:|---|
| decoder | 59.3 ms | 44.7 ms | max rel 1.15e-3 |
| encoder, 1 layer | 32.5 ms | 20.7 ms | max rel 3.2e-4 |
| encoder, 3 layers | 95.4 ms | 60.6 ms | max rel 1.2e-3 |

At the start of this series (fused hand attention, before the VTCM arena) the same three took 84 / 41 / 111 ms.

**Generated attention.** The decoder's self-attention can also run as generated kernels (`HMX_ATTN=3`, opt-in), head by head
and transposed: S^T = K Q^T on the HMX, softmax over S^T's rows, O^T = V^T P^T on the HMX. This round took it from 128 to
52 ms. Most of the gain came from these changes:
- *Operands from the projections.* The ONNX graph's realized (8, 900, 32) and (8, 32, 900) copies are read through back to
  the (900, 256) projection outputs. Each operand is then one fp32 row copy or 32 x 32 block transpose, cast to fp16 last. Each
  had been a second transposing copy over rows of 900 values (2.9 ms each, 13 ms per layer).
- *Softmax.* The max shift runs in fp16 and the exp is computed in place over S^T, so each head needs one 1.8 MB VTCM region. A
  row of ones in V^T gives each query's sum of exp from the P V product itself, so there is no normalization pass. The fp16 exp2
  helper now keeps its polynomial in qf16, as the hand softmax does: 0.39 -> 0.18 ms per head.
- *Output.* Each head's O^T and its sums are stored into rows of one buffer, then one block transpose divides and writes
  (900, 256).
- *HMX output pairing.* When every B tile is cached by the inner tile loop, two adjacent output tiles are now computed together
  and written as full 128-byte rows. S^T went from 0.385 to 0.18 ms per head, and the decoder's other generated Gemms from 15
  to 11 ms. Those Gemms are what sped up the default build too.
- *VTCM.* A buffer left out may now take the place of overlapping ones that carry less traffic (uses x bytes). The graph also
  hoists calls that don't depend on an HMX-to-HMX chain out of it: call sinking had put the V projection between a layer's
  first S^T and its P V.

| model | before | now, hand attention (default) | `HMX_ATTN=3` | vs onnxruntime on the plain export |
|---|---:|---:|---:|---|
| decoder | 44.7 ms | 40.5 ms | 51.9 ms | max rel 1.15e-3 |
| encoder, 1 layer | 20.7 ms | 18.6 ms | | max rel 3.2e-4 |
| encoder, 3 layers | 60.6 ms | 53.8 ms | | max rel 1.2e-3 |

The hand attention keeps S in the HMX tile layout from its S call through the softmax to P V, so it never converts S to
row-major or packs P (its S and P V calls take under 0.02 ms each per head). That is most of the remaining 0.3 ms per head.

**Raw-tile attention.** The generated attention now keeps S^T and P in the HMX tile layout too, which took `HMX_ATTN=3` from
51.9 to 42.7 ms. Converting S^T's tiles to rows had been ~0.125 of the 0.18 ms S^T call per head. Tinygrad's tensor cores
don't apply when the tile layout is visible as a view (the key axis splits into row pairs and their parity), so the layout
lives below tinygrad:
- *Raw-tile buffers.* An fp16 buffer of 1024 k + 1000 elements is marked as raw tiles by its size. Tinygrad indexes it as a
  row-major matrix in both matmuls, so the tensor cores apply as usual. The HMX renderer writes the S^T tiles as they are,
  straight into VTCM (`__hmx_store_raw`), and P V reads P's K panel in place (`__hmx_mac_span_raw`). The softmax kernels read
  the same memory as `(query tiles, key tiles, 16 row pairs, 64 lanes)`: the max over the keys is a vector max over the row
  pairs, then a max over lane pairs. The sums come from a reduce over P, so V^T needs no ones row and P V only one M tile.
- *VTCM.* Raw-tile regions are placed first, 2 KB aligned. Calls that read them are pulled up to just after their last
  dependency: call sinking had moved a layer's last P V behind the other heads, its P missed the VTCM arena, and that P V
  copied its tiles in from DDR (0.3 ms instead of 0.05).
- *Deferred HMX output* (`HMX_DEFER`, now default). An output tile pair goes to a free VTCM slot pair and is converted to rows
  after the next tile's load pair is issued, instead of right after its own store. This helps every generated Gemm.
- *No prefetch on VTCM.* The emitter now writes kernels after placing regions. A kernel's `dcfetch` prefetches of parameters
  placed in VTCM become no-ops: VTCM isn't behind L2, and the exp over the tiles went from 0.165 to 0.137 ms per head.
- *Heuristic.* An elementwise kernel whose unit-stride axis fills whole HVX vectors no longer upcasts a broadcast axis for
  reuse when that makes its loads gathers. That upcast had run the exp and the max's broadcast copy scalar. A stack of
  per-head outputs keeps its upcast: a comparison reads its axis.

| model | before | now, hand attention (default) | `HMX_ATTN=3` | vs onnxruntime on the plain export |
|---|---:|---:|---:|---|
| decoder | 40.5 ms | 38.2 ms | 51.9 -> 42.7 ms | max rel 1.15e-3 (`HMX_ATTN=3`: 9.9e-4) |
| encoder, 1 layer | 18.6 ms | 18.0 ms | | max rel 3.2e-4 |
| encoder, 3 layers | 53.8 ms | 51.9 ms | | max rel 1.2e-3 |

Per head, the generated attention now spends about 0.14 ms on the exp, 0.05 ms each on the max and sum reduces, and 0.05 ms on
P V. Its S^T is below the profile's cutoff. The hand softmax does the max, exp, sum and normalization in one 0.215 ms pass over
the tiles, and that fusion is the remaining gap (about 0.1 ms per head).

**Generated attention by default.** The generated attention went from 0.34 to about 0.2 ms per head, under the hand path's
0.26 ms, so it is now the default: the decoder takes 36.5 ms with it and 37.9 ms with the hand kernels (`HMX_ATTN=2`, or 1
for the single fused call). The softmax still runs as separate kernels. Instead of fusing them, this round removed the work
and the stalls between the HMX operations:

| per head, V69 | before | now | what changed |
|---|---:|---:|---|
| S^T (HMX) | 50 us | 39 us | each raw tile's place computed from the loop counters, the VTCM check done once per call (it was a divide, a modulo and a range check, ~25 scalar packets between each store and the next load) |
| max | 45 us | 14 us | no key-padding bias: the max and the sums cover only the valid row pairs (N even), and the max is clamped at -15 so the padding keys' zero scores can't overflow the exp |
| exp | 137 us | 74 us | no bias load; fp16 exp2 kernels four vectors wide (`HVX_EW_EXP_LANES=256`): at two vectors the qf16 polynomial ran one HVX op per packet |
| P V (HMX) | 47 us | 31 us | each single output tile is converted to rows after the next tile's loads are issued (`HMX_DEFER_ONE`), and V^T's tiles are L2-prefetched |
| sums | 49 us | 30 us | fp16 over runs of up to 16 row pairs, then fp32. Summed in fp32 throughout, each row pair cost 7 packets: LLVM widens to a qf32 pair, re-interleaves the lanes and converts after every add |

The max and the sums now spread 30 rows over 4 threads as 15 slices instead of 3 threads (`HVX_THREAD_BALANCE`).

| model | before | now (generated attention) | hand attention (`HMX_ATTN=2`) | vs onnxruntime on the plain export |
|---|---:|---:|---:|---|
| decoder | 38.2 ms | 36.5 ms | 37.9 ms | max rel 1.22e-3 |
| encoder, 1 layer | 18.0 ms | 17.5 ms | | max rel 3.2e-4 |
| encoder, 3 layers | 51.9 ms | 50.8 ms | | max rel 1.2e-3 |

The encoders have no such attention; they gain from the deferred single-tile outputs and the thread balance in their own
Gemms and reduces. A one-layer attention micro model (Q/K/V projections of a 900 x 256 input, 8 heads of 32) shows every
per-head call in the profile, which the full decoder's 254-call cap hides. Building it exposed a separate bug, now fixed. LLVM
keeps a value widened from fp16 in qf32, and a multiply of it by a constant was never renormalized. A MatMul -> scalar Mul ->
head Transpose chain had the scale fused into the fp16 product's cast back, and came out with cos 0.007 on the phone (it is
exact in hexagon-sim). Such a multiply now goes through the sf barrier helper, as products of two computed values already did.

**HMX stalls and scalar kernels.** The decoder went from 36.5 to 29.6 ms. Most of the gain is in the generated HMX Gemms, where
the single thread waited on memory or did scalar work between HMX operations. Timing-only builds found each stall in one run:
the same build with one stage left out (no tile packing, no output conversion) gives wrong results, but its time is that
stage's cost.

| change | effect |
|---|---|
| One l2fetch at a time. An l2fetch issued while another is in flight replaces it, so the per-row prefetch boxes fetched little more than the last row. A row tile's whole A panel is now one box (the 64-bit form takes any panel), and the next row tile's panel is fetched right after it | decoder 36.5 -> 33.6 ms; the 2528-row value projection 0.5 -> 0.18 ms per call |
| A packed two K tiles per shuffle from aligned 128-byte rows, its slot sets alternating by row tile | -0.8 ms |
| Tile-pair output conversion as one halfword deal (`vdealvdd`) instead of two deals, two rotates and two predicated muxes; the flush loads four row pairs before storing | the 900-row projections 84 -> 66 us per call |
| The attention's sums from the HMX: V^T's row of ones gives them as one more output row, at the cost of a second M tile in P V | -14 us per head |
| P V's output tiles (raw-tile B) computed and converted in pairs | -6 us per head |
| `__hmx_begin` once per call instead of before every output tile, and the raw output's VTCM flag a local instead of a static reloaded after every asm store | S^T 35 -> 29 us per head |
| A scalar kernel weighs a vector's work per element when choosing threads. The decoder's reference-point updates (sigmoid / inverse sigmoid over 900 rows of a 10-wide tensor) ran on one or two threads | decoder 31.0 -> 29.6 ms |

| model | before | now | vs onnxruntime on the plain export |
|---|---:|---:|---|
| decoder | 36.5 ms | 29.6 ms | max rel 1.42e-3 |
| encoder, 1 layer | 17.5 ms | 15.5 ms | max rel 3.2e-4 |
| encoder, 3 layers | 50.8 ms | 46.0 ms | max rel 1.2e-3 |

Tried and dropped:
- *Raw-tile A operands.* A threaded producer wrote the HMX tile order, so the Gemm copied tiles instead of packing them. The
  producers cost what the packing saved (decoder 32.9 vs 32.6 ms, enc3 47.8 vs 47.2).
- *A degree-3 exp2 polynomial.* Its error is 1.3e-4 with fp16 coefficients, under fp16's half ulp, and it saved 8 us per head.
  But the decoder's class scores went past the 2e-3 tolerance (2.6e-3).
- *Vectorizing the reference-point kernels.* With stride-10 lane gathers and 30-lane transcendental helpers they were slower
  than scalar (0.5 vs 0.35 ms per call).

What remains in the decoder:
- the hand MSDA kernel, 5.1 ms;
- the attention's exp, 3.0 ms;
- fp32 -> fp16 casts in front of each projection, ~1.3 ms, and the attention's block transposes, ~1.7 ms. Both are
  DDR-bound copies that only multi-output kernels could fuse away.

**Data-dependent skipping in generated kernels.** The generated MSDA had no way to skip work. The hand kernel builds, per
query, the list of cameras that see it; BEVFormer's cameras see 19% of the (camera, query) pairs (2129 of the 2500 queries
are seen by one camera, 367 by two). The generated spatial cross-attention computed all six and multiplied by the mask.
The code generator now supports data-dependent skipping (`GATED_REDUCE`, on by default, generic):
- A sum over a loop of `where(g, t, 0)`, with `g` the same in every lane of a step, accumulates `t` under `g`: a gated
  accumulator store, an `IF` once linearized. This also applies when the term sits under a sum of unrolled terms.
- After linearizing, `sink_into_ifs` merges adjacent `IF` blocks under one gate (a vector store split into per-lane stores).
  It also moves into each `IF` the pure ops and loads of its own scope that only the block uses, so `t` is not computed when
  `g` is false.
- Memory coalescing groups gated stores by their gate, and a store's gate gets its load.

The generated MSDA (`HAND_MSDA=0`) now sums `where(visible[camera, query], term, 0)` over cameras and points, all cameras in
one GridSample. The cross-attention kernel's whole body runs inside `if (visible)`, and its accumulate is again one vector add
(the masked select had split the term into 32 scalar lanes).

| | before | now | hand MSDA (default) |
|---|---:|---:|---:|
| spatial cross-attention, per encoder layer | 61.4 ms | 21.1 ms | ~4.3 ms |
| encoder, 1 layer, generated MSDA | 83.8 ms | 43.2 ms | 15.5 ms |
| encoder, 3 layers, generated MSDA | 248.9 ms | 121.2 ms | 46.1 ms |

All pass against onnxruntime; the default build (hand MSDA) is unchanged. The hand kernel stays the default. The generated one
still computes each tap's location, floor and bilinear weights in scalar code, while the hand kernel does 32 sampling points
per vector. That is the next step toward parity, and the temporal self-attention (no mask, 14 ms per layer generated vs ~3.3
hand) needs it too.

**Generated MSDA: gathered buffers in VTCM, direct sampling.** Two smaller steps on the generated MSDA (`HAND_MSDA=0`):
- *Gathered parameters get VTCM first.* The renderer marks the parameters a kernel loads at addresses computed from loaded
  values (`/* vtcm-gathered: j */`), and the graph emitter weighs them like a hand kernel's declared `vtcm-params`. The value
  rows had lost the arena to plainer buffers: from DDR, every gathered 128-byte row is a cache miss.
- *Taps straight from the value rows.* Pixel `x = ref.x * W + off.x - 0.5`, with no per-point division (`off / (W, H)` had been
  a reciprocal for each point) and no GridSample layout round trip.

| generated MSDA | before | gathered in VTCM | + direct sampling |
|---|---:|---:|---:|
| spatial cross-attention, per layer | 21.1 ms | 18.9 ms | 18.2 ms |
| temporal self-attention, per layer | 14.1 ms | 11.3 ms | 11.3 ms |
| encoder, 1 layer | 43.1 ms | 38.1 ms | 37.4 ms |
| encoder, 3 layers | 121.2 ms | 113.4 ms | 111.6 ms |

All pass; the default build (hand MSDA) is unchanged. The cross-attention's body is now ~1370 instruction packets per visible
(camera, query, head): 8 scalar chains of address and weight math, one per sampling point, each feeding its own gathers. The
hand kernel computes 32 points' addresses per vector instruction. Generated code needs a kernel with two vector axes (points for
the address math, channels for the taps), which the scheduler can't express yet. Splitting it into two kernels would write
~15-20 MB of per-point addresses and weights to DDR per layer.

**Generated MSDA: arithmetic sampling, and what vectorizing it would take.** The generated MSDA's sampling is now arithmetic only:
- floor by `+- 1.5 * 2^23` and a `max`/`min` step;
- corner validity as `max`/`min` products;
- the row index through the mantissa of `index + 2^23`.

Cross-attention went from 18.2 to 16.5 ms per layer. Generated-MSDA encoder: 37.4 -> 35.7 ms (1 layer), 111.6 -> 106.0 ms (3
layers), all pass. This needed a fix in tinygrad's simplifier: with float reassociation on, it folded `(t + c) - c` back to `t`.
Fused, it reordered `r - t` into `(R - t) - c` and lost `t`'s fraction at magnitude `2^23`. It no longer reassociates around a
float constant of `2^20` or more, which leaves the default models unchanged.

Tried and dropped: vectorizing the per-point address math. An output-axis upcast made the 8 heads x 8 points 64 lanes, and a
forward (SLP) vectorizer turned the per-lane scalar chains into vector ops (kernel 2998 -> 767 lines). But taking ~512
addresses and weights back out lane by lane per (query, camera) cost what the vector math saved (cross-attention 17.5 vs 16.5
ms). The upcast alone made the temporal attention 4x slower, and the result failed the output check on the phone (qf32
rounding). The hand kernel stages the values through a small table instead: vector stores, then scalar loads in the per-head
loop. Generated code would need two loop nests joined by a local buffer in one kernel, which tinygrad's scheduler can't express
yet. That is the missing piece for a generated MSDA at the hand kernel's speed.

**Hand MSDA kernel: tap tables.** The default build's MSDA (`HAND_MSDA=1`) got faster. The kernel builds each work item's tap
offsets and weights with HVX, then reads them back as scalars to address the value rows. In hexagon-sim timing mode, that
table-to-address chain was 43 of 89 cycles per point. Changes:
- **Two points per 64-bit table read.** The decoder's call went from 848 to 684 us.
- **One 2 x 2 window per point, kept inside the map.** The corners' weights shift with it at a border, so the table holds one
  row offset per point instead of four: 684 -> 587 us.
- **The next queries' offsets and weights L2-prefetched** inside the kernel: 587 -> 542 us. Putting them in VTCM instead was
  slightly faster for the call but displaced other buffers.

| model | before | now |
|---|---:|---:|
| decoder | 29.5 ms | 27.7 ms |
| encoder, 1 layer | 15.6 ms | 13.5 ms |
| encoder, 3 layers | 46.2 ms | 40.0 ms (SCA 12.9 -> 9.9, TSA 9.8 -> 6.8) |

All pass against onnxruntime. The kernel's own hexagon-sim checks pass for all eight case kinds (fp32 / uint8, multi-level).
Details and the dropped variants are in `scripts/android/msda_hvx/README.md`.

**Closing tinygrad's modeling gaps.** The hand kernels had relied on seven things the fork's tinygrad couldn't express. Each
was worked on in parallel on its own branch, then merged (fork commit `73f3ef25e`):
- **A staging buffer inside one kernel** (`Tensor.stage`). Two loop nests share a register table: vector code builds per-query
  tap rows and weights, and a scalar-indexed loop reads them back. This is the hand MSDA's structure. With it, the generated
  MSDA beats the hand kernel on the encoder. Cross-attention per layer: generated 16.5 -> 2.4 ms, hand 3.3 ms. Temporal
  attention: 11.3 -> 2.0 ms, hand 2.3 ms. MSDA now picks by query count: generated from 2048 queries (`MSDA_GEN_MIN_Q`), so
  the encoder's 2500 use it, and hand below that. The decoder's 900 stay hand: generated, the decoder was 25.3 vs 23.5 ms.
- **Multi-output kernels** (`MULTIOUT`). An elementwise consumer merges into its producer as an extra store. A LayerNorm or
  bias add now writes the fp16, row-padded HMX operand itself. That drops the separate cast and pad kernels: 1.3 ms in the
  decoder, about 3 ms in the 3-layer encoder.
- **Layout and residency as buffer attributes.** The HMX raw-tile layout is now `ParamArg.layout`; before, it was marked by
  an odd buffer size. Model code can request VTCM residency. The VTCM placer gets a refill pass that places regions it had
  passed over, and MSDA reads its inputs in place instead of copying them.
- **qf32 rules.** A qf32 subtract of nearly equal sf operands is not exact, so add-a-big-constant rounding tricks are unsafe in
  vectors. Vector int <-> float conversion, by contrast, is already exact integer code. On top of these rules:
  - a conversion-free LOG2;
  - an inline half -> float;
  - vectorization of strided transcendental kernels.

  The decoder's reference-point updates went from 1.15 to 0.27 ms.
- **A prefetch cost model.** Strided DDR streams get one l2fetch box per chunk of loop iterations, from a strip-mined loop.
  Contiguous vector streams keep their dcfetches, because boxes were slower for them. LayerNorm statistics went from 150 to
  49 us. The attention transposes went from 251 to 70 us: their cost was mostly a vector reciprocal per lane, now one
  reciprocal of the row.
- **fp16 weights.** GEMM weights are stored in fp16 instead of being cast on every run, and tiny-row pads use the hand row
  copy.
  - Overlapping HMX and HVX work is implemented, but behind `DSP_GRAPH_HMX_OVERLAP`, off by default. It is correct but saves
    nothing: moved calls held VTCM longer, and an in-kernel helper-thread queue cost more in fences than it parallelized.
- **Data-dependent work lists.** Measured, then dropped: the camera gate costs at most about 1.3 ms per layer, and a per-query
  work list gained nothing on real data. The generated MSDA's real cost was per-tap weights computed as qf32 vector chains
  over splats. `HVX_SPLAT_SCALAR` (off by default) keeps them scalar: 16.5 -> 8.7 ms per layer for the old generated
  cross-attention path.

| model | before | now |
|---|---:|---:|
| decoder | 27.7 ms | 23.5 ms |
| encoder, 1 layer | 13.5 ms | 9.7 ms |
| encoder, 3 layers | 40.0 ms | 29.6 ms |

All pass against onnxruntime.
