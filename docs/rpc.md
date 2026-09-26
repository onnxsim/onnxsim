# onnxsim RPC: run, time and fold on a remote device

A TVM-style remote-execution workflow for ONNX models: start a small server on the machine (or
device) you care about, then `connect`, `upload`, `load_model` and `time_evaluator` from your
host, or let `onnxsim.simplify` evaluate constant folding on that machine. It borrows the
*shape* of TVM's RPC (device keys, tracker, session, `upload`/`load`/`time_evaluator`) on a much
smaller protocol of its own. It is **not** wire-compatible with TVM's RPC; see
[Relationship to TVM RPC](#relationship-to-tvm-rpc).

The dependency-free native transport also supports optional worker-side
profiling. `Off` adds no profile payload, `Summary` returns aggregate runner
timings, and `Detailed` returns bounded events with request-relative timestamps.
The host anchors those events into the onnxsim Chrome/Perfetto trace alongside
the host-side `RemoteRPC` duration. This is suitable for constrained
Snapdragon/AX8850 workers because the device needs neither a JSON library nor
clock synchronization.

The transport is independent of the control-plane protocol. A ROS2/rosbridge
or DORA gateway can expose discovery, compile, run, and profile actions while
forwarding the same binary tensor and profile payloads. Large tensors and
traces should stay binary; use the control plane for metadata, request IDs,
health, and progress.

For DORA specifically, `tools/onnx-remote/onnx-remote-dora-node` is an optional
C node adapter. Its `run` input and `result` output carry the transport's
payload-only format as raw UInt8 messages, so DORA does not need to understand
the ONNX tensor schema. The adapter forwards to the existing TCP worker and
can therefore be used with the reference or AXCL worker.

## External compiler and artifact caching

The native executor keeps the original model-per-run path as the default. With
`RemoteExecutorOptions.compile_model=true`, each distinct serialized fold-group
is compiled once and subsequent runs use the returned artifact ID. The
compiler response may include an opaque manifest and inline artifact bytes.

Compilation and execution may use different endpoints: `compile_host` and
`compile_port` select the compiler, while the existing `host` and `port` select
the runner. An empty compiler host or zero compiler port falls back to the
runner endpoint.

Caching is split deliberately: onnxsim owns a short-lived in-process cache to
avoid compiling the same subgraph repeatedly during one simplification; the
compiler/runner owns persistent artifact caching and compatibility validation.
The latter is the only component that knows whether an artifact remains valid
for a particular compiler, SDK, driver, device, and I/O ABI.

```python
import numpy as np
import onnxsim
import onnxsim.rpc as rpc

remote = rpc.connect("127.0.0.1", 9090, key="pixel")     # or rpc.connect_tracker(h, p).request("pixel")
print(remote.info)                                       # platform, onnx / onnxruntime versions, providers

remote.upload("model.onnx")                              # copy into the server's workspace
model = remote.load_model("model.onnx")                  # keeps an onnxruntime session alive
out = model.run({"x": np.zeros((1, 3, 224, 224), "float32")})
t = model.time_evaluator({"x": x}, number=5, repeat=3)   # device-side timing only
print(f"{t.median * 1e3:.2f} ms", t.results)

with rpc.remote_executor(remote):                        # constant folding runs on the device
    simplified, ok = onnxsim.simplify(model_proto)
```

## Server, tracker, devices

```bash
python -m onnxsim.rpc server --host 127.0.0.1 --port 9090 --key pixel        # on the target
python -m onnxsim.rpc tracker --port 9190                                    # optional, anywhere
python -m onnxsim.rpc server --port 9090 --key pixel --tracker host:9190     # register a device
```

- **Key.** A server started with `--key` only accepts clients presenting the same key, like a TVM
  device key; the tracker hands out servers by key, round-robin.
- **Reaching a device.** Forward the port instead of exposing it, e.g.
  `adb forward tcp:9090 tcp:9090`, or an SSH tunnel. The server binds to loopback by default.
- **Runtime.** Models run with onnxruntime when it is installed on the server (one cached session
  per loaded model, so `time_evaluator` excludes session creation) and with onnxsim's pure-Python
  reference evaluator otherwise. `providers=[...]` selects onnxruntime execution providers, and is
  checked against what the server actually has.
- **Where it runs.** The server is Python, so the target needs Python, `onnx` and ideally
  `onnxruntime`: Linux boards and servers, containers, Termux. Stock Android has no Python, so
  the phone in this repo's Hexagon experiments would need either Termux or a native server that
  speaks the same protocol (it is deliberately small: see below) -- not provided yet.

## API

| Call | Meaning |
|---|---|
| `rpc.connect(host, port, key="")` | open a `Session`; the handshake checks the key |
| `rpc.connect_tracker(host, port).request(key)` / `.summary()` | get a session by device key / list registered servers |
| `Session.info` | server platform, `onnx`, `onnxruntime`, providers, `onnxsim` versions |
| `Session.upload(path_or_bytes, name=None)` | store a file in the server workspace (name sanitised) |
| `Session.load_model(name/path/bytes/ModelProto, providers=None)` | returns a `RemoteModel` |
| `RemoteModel.run(inputs)` | outputs by name as NumPy arrays |
| `RemoteModel.time_evaluator(inputs, number, repeat)` | `ProfileResult` (`results`, `mean`, `median`, `min`, `max`, `std`), seconds per call |
| `Session.run(model, inputs)` | one-shot run without keeping a handle |
| `rpc.remote_executor(session)` | context manager: `onnxsim.simplify` folds constants remotely |

## Benchmarking tinygrad's code generation

`load_model(..., runtime="tinygrad", device="NV", options={"BEAM": 2})` runs the model through
tinygrad's ONNX frontend on a tinygrad device of the *server's* hardware, so tinygrad's codegen can
be benchmarked (and compared with onnxruntime through the same API) on any machine that can run a
server. `time_evaluator` then returns tinygrad-specific `stats` next to the timings:

| Key | Meaning |
|---|---|
| `results` | wall time per call of a steady-state `TinyJit` replay (no Python ONNX interpreter cost) |
| `kernels`, `gflops`, `gbytes` | kernel count, FLOPs and bytes moved by one replay (tinygrad's counters) |
| `kernel_time_s` | summed device kernel time of one replay (measured under `DEBUG=2`) |
| `eager_call_s`, `first_call_s` | one interpreted call, and the first call including kernel compilation |
| `device`, `options` | what actually ran |

Codegen `options` are allow-listed (`BEAM`, `NOOPT`). Each loaded tinygrad model runs in its own
worker process, so every `(device, options)` pair starts from clean kernel/schedule caches -- a
`BEAM` setting really applies instead of reusing kernels compiled earlier under another setting --
and tinygrad's thread-bound state and GPU faults stay out of the server. tinygrad's ONNX frontend
caches Python constants between calls, so this suits static-shape models. `scripts/tinygrad/`
has a benchmark built on this; a server started with `MOCKDSP=1` also exposes tinygrad's Hexagon
renderer (kernels executed under `qemu-hexagon-static`, where "time" is an instruction count).

## Remote constant folding

onnxsim's folder builds a throwaway sub-model per fold group and hands it to a `ModelExecutor`
(see [dlpack-executor.md](dlpack-executor.md)). `remote_executor` swaps that executor for one that
ships each sub-model to the server, so folding uses the *target's* numerics and kernels, and an
operator the host's onnxruntime lacks can still be folded if the device has it. Each fold group
is one round trip, so this suits high-latency links poorly on models with thousands of groups.

## Protocol

Each message is `<u32 header_len><u32 blob_count><JSON header>` followed by `blob_count` blobs of
`<u64 length><bytes>`. Tensors are `{"name","dtype","shape"}` records in the header plus one
little-endian, C-contiguous blob each; supported dtypes are float16/32/64, (u)int8/16/32/64 and
bool. Operations: `hello` (key handshake), `info`, `upload`, `load_model`, `unload`, `run`, `time`,
`run_once`, `close`; the tracker speaks `register`, `request`, `summary`. There is no pickle
anywhere: a server only ever parses JSON and raw buffers. A native (C++) server only needs to
implement this framing and call its own runtime.

## Relationship to TVM RPC

| | TVM RPC | onnxsim RPC |
|---|---|---|
| Purpose | run TVM-compiled modules and PackedFuncs on a device | run ONNX models, and onnxsim's constant folding, on a device |
| Session shape | `connect`, `upload`, `load_module`, `time_evaluator`, tracker + device keys | the same names and flow |
| Wire protocol | TVM's (`0xff271` handshake, PackedFunc packets, DLTensor copies) | own, JSON + blobs |
| Interoperable with TVM clients/servers | yes, with TVM | **no** |
| Server | C++ `tvm_rpc` (Android, Hexagon, iOS ...) | Python (a native server can follow the protocol) |

Being wire-compatible would mean reimplementing TVM's session layer (`rpc_endpoint.cc`, PackedFunc
argument marshalling, module and device APIs) and tracking TVM's protocol across versions; that
was judged out of proportion for what onnxsim needs (running ONNX models and folding remotely).
If interop with an existing TVM RPC server is needed, TVM's own client can be used alongside: the
two are independent.

## Security

A server executes whatever ONNX model a client sends. It binds to loopback by default; the key is
an identifier, not authentication; onnxruntime custom-op libraries are never loaded; uploads are
flattened to a sanitised file name inside the workspace; blobs are size-capped (4 GiB default).
Expose a server only on a trusted link (`adb forward`, an SSH tunnel, a private network).

## Limits

The tracker has no exclusive locking or priorities (it round-robins live registrations), sessions
are one connection each, and dtypes such as bfloat16, strings and float8 are not transferred yet.
