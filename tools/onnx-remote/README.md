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
(`load_compiled`, `run_compiled`, `run`, `io_info`, `unload`), so ROS2/DORA
discovery can verify it and capability-gated dispatch can reject it when graph
execution is required. The manifest also carries `model_cache`, `max_loaded`
and the current `loaded` count.

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
  request.
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

`tools/onnx-remote/python/onnx_remote_client.py` is a numpy-only Python client
for the wire protocol (`Client.run`, `run_path`, `io_info`, `unload`,
`load_compiled`, `run_compiled`, `capabilities`); it works against every
worker here.

`remote_axcl_worker.cpp` can be syntax-checked without the SDK against the
declarations in `test/axcl_stub/axcl.h` (the command is in that header). The
stub has no definitions and is not on the real build's include path.

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
