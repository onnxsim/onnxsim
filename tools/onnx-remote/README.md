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

Profile events are returned with the response rather than streamed. This keeps
the constrained worker simple and is sufficient for a completed subgraph
profile. A future ROS/HTTP gateway can stream progress separately while using
the same event fields for the final trace.

The next integration layer can make an ONNX Runtime plugin EP claim a maximal
supported subgraph and send it as an operation/model handle over this
transport.  The reference worker deliberately does not pretend to be that EP
yet.

The transport also carries an optional serialized `ModelProto`. This is the
native `onnxsim::ModelExecutor` integration point: a host-side executor can
send each constant-folding submodel to a worker without Python. The worker may
run that model with native ONNX Runtime, an accelerator compiler, or translate
it to a device-specific model handle.

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
