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
  dtype or shape.
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
on-disk cache; its key includes the serialized model, target, and configured
compiler command and explicit compiler identity.

```sh
cmake -S tools/onnx-remote -B build/onnx-remote
cmake --build build/onnx-remote --target onnx-remote-compiler
build/onnx-remote/onnx-remote-compiler \
  --port 39502 --target qnn-htp --compiler-id qairt-2.31.0 \
  --cache-dir /var/cache/onnxsim-qnn \
  --command 'qnn_compile_wrapper --input {input} --output {output} \
             --manifest {manifest} --target {target}'
```

The command is trusted local configuration, not request data. It must write the
compiled artifact to `{output}` and a bounded UTF-8 manifest to `{manifest}`;
`{input}` is the received ONNX ModelProto. A no-command service copies the
model into an artifact and is useful for validating networking and cache
plumbing before installing QAIRT. A QNN wrapper can run the converter,
backend-specific graph preparation, and context-binary generation as one
command, while keeping those SDK-version-specific details out of onnxsim.

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

The model is loaded for each request in this first correctness-oriented
adapter.  That is deliberately simple and isolates model-load failures; a
persistent model cache should be added once the wire protocol is exercised on
the card.
