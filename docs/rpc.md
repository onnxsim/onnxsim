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

Graph-aware native clients should use the `subgraph` operation with serialized
graph bytes in the request model field. Profile events can be consumed through
the C++ `receive_profile` callback, while the response retains the complete
bounded event list for lossless forwarding.

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

`tools/onnx-remote/onnx-remote-compiler` provides a small dependency-free
compiler endpoint for this split. It serves each request on its own thread:
cache hits are answered while other models compile, `--jobs N` (default 2)
bounds concurrent cold compiles, and a second request for a model already being
compiled waits for that compile and reuses its artifact. It accepts `COMPILE` requests, invokes a
trusted command template with `{input}`, `{output}`, `{manifest}`, and
`{target}` paths, and persists the resulting artifact and manifest. This is a
convenient SNPE replacement boundary: a QAIRT/QNN wrapper can perform ONNX
conversion, legalization, and context-binary generation on the compile host,
while the execution host only receives the final artifact. The service has a
passthrough mode for transport tests; it is not itself a QNN compiler.

Set `RemoteExecutorOptions.require_graph_execution=true` to probe the
runner's capability manifest before the first model run and fail fast unless
it advertises `graph_execution:true`. The probe result is cached per
executor; legacy manifests without the field are rejected under this flag,
so leave it off for pre-graph unary workers.

Compiled execution can optionally use a load/attach handshake: the host sends
`load_compiled(artifact_id, artifact)` once to the runner, then sends
`run_compiled(artifact_id, tensors)` without repeating the artifact bytes. The
native executor keeps this disabled by default for stateless compatibility; set
`attach_compiled_artifact=true` for a runner with persistent artifact storage.

A runner may also keep a model's recurrent state resident (the Hexagon v65 runner does, for artifacts whose manifest pairs an
output with the input it feeds, e.g. openpilot's `next_state_img_q` -> `state_img_q`). After one call that sends the state,
a client sends that input as an empty tensor (a zero dimension) to mean "the previous call's output", and the matching output
comes back empty. `onnx-remote-client --compile-run ... --resident IN:OUT,...` does this and checks the result bit for bit
against sending the state back explicitly. For openpilot's driving model this is 4.4 MB less per call: 470 -> 356 ms
RPC-inclusive for 340 ms of runner time. On the Hexagon runner the state also stays on the DSP: the program loops each state
output back into its input region after every run and the runner leaves the state out of the FastRPC transfer in both
directions (`flags` of `tg_graph_run`), which took the runner-side overhead of a driving call from 6.2 to 3.7 ms.

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
# Or send only shape/dtype metadata; the server generates and reuses random inputs.
t = model.time_evaluator({"x": x}, number=5, repeat=3, random_inputs=True, seed=7)
# Shape-only inputs avoid allocating template arrays on the host. Bounds are [low, high).
t = model.time_evaluator(
    {"tokens": rpc.RandomInput((1, 128), "int32", low=0, high=32000)},
    number=5,
    repeat=3,
    random_inputs=True,
)
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
- **Where it runs.** This high-level Python RPC server needs Python, `onnx` and ideally
  `onnxruntime`: Linux boards and servers, containers, Termux. Stock Android has no Python;
  use the dependency-free native worker under `tools/onnx-remote` for the binary v5 transport
  and remote EP path instead.

### TPU-MLIR compilation and SG2002 execution

The Python RPC server can also compile a static-shape float32 ONNX model with TPU-MLIR, then
send the CVI model and input tensors to an SG2002 over SSH. The server host needs the TPU-MLIR
tools and Paramiko; the board needs its matching CVI runtime and `model_runner`.
The SG2002 target is compiled as `cv181x` with BF16 compute and float32 I/O by default. Set the
board password in the environment variable named by `--tpu-password-env` (default
`TPU_MLIR_SSH_PASSWORD`):

```sh
export TPU_MLIR_SSH_PASSWORD="$BOARD_PASSWORD"
python -m onnxsim.rpc server --host 127.0.0.1 --port 9090 --key sg2002 \
  --tpu-ssh-host "$BOARD_ADDRESS" --tpu-ssh-user root
```

Connect from a client and select `runtime="tpu_mlir"` when loading the model:

```python
remote = rpc.connect("127.0.0.1", 9090, key="sg2002")
model = remote.load_model("model.onnx", runtime="tpu_mlir")
outputs = model.run({"input": input_array})
timing = model.time_evaluator({"input": input_array}, number=1000, repeat=3)
print(timing.median * 1e3, "ms on the board")
```

For SG2002 engine counters, use `pmu=True` in place of a normal timing run:

```python
profile = model.time_evaluator({"input": input_array}, number=10, repeat=3, pmu=True)
print(profile.stats["tiu_ms"], profile.stats["tdma_ms"], profile.stats["inference_ms"])
```

This enables `TPU_ENABLE_PMU` on the board and returns median TIU, TDMA, and inference
intervals, plus clock, bandwidth, and engine activity fields in `profile.stats`. One initial run
per repeat is discarded as warm-up. `profile.stats["samples"]` contains every remaining
per-inference sample, including raw TIU, TDMA, and inference tick counts. PMU runs have profiling
overhead; their `results` report the PMU inference interval and should not be compared with normal
wall/device timing.

For INT8, create a calibration table with TPU-MLIR from representative input samples and
provide its path on the server:

```python
model = remote.load_model(
    "model.onnx",
    runtime="tpu_mlir",
    options={
        "quantize": "INT8",
        "calibration_table": "/server/path/model-cali.table",
        "opt": 1,  # TPU-MLIR layer-group scheduling level (1–3)
        "do_winograd": True,
        "matmul_perchannel": True,
    },
)
```

Compilation happens once at `load_model`; model files are removed from the board when the
session closes. `time_evaluator` uses `model_runner`'s device timer, so SSH transfer and process
startup are excluded from the reported inference time. The current backend requires fixed,
positive input dimensions and float32 model inputs. `--tpu-python`, `--tpu-model-transform`,
and `--tpu-model-deploy` select the compiler Python environment and tool scripts, including a
source checkout's scripts. Board artifacts use `/data/onnxsim-tpu-rpc` by default; override it
with `--tpu-remote-dir` if the board stores model files elsewhere.

## API

| Call | Meaning |
|---|---|
| `rpc.connect(host, port, key="")` | open a `Session`; the handshake checks the key |
| `rpc.connect_tracker(host, port).request(key)` / `.summary()` | get a session by device key / list registered servers |
| `Session.info` | server platform, `onnx`, `onnxruntime`, providers, `onnxsim` versions |
| `Session.upload(path_or_bytes, name=None)` | store a file in the server workspace (name sanitised) |
| `Session.load_model(name/path/bytes/ModelProto, providers=None)` | returns a `RemoteModel` |
| `RemoteModel.run(inputs)` | outputs by name as NumPy arrays |
| `RemoteModel.time_evaluator(inputs, number, repeat, random_inputs=False, seed=None, pmu=False)` | `ProfileResult` (`results`, `mean`, `median`, `min`, `max`, `std`), seconds per call; random mode accepts arrays as templates or `rpc.RandomInput(shape, dtype, low, high)`; `pmu=True` requests SG2002 TPU counters |
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

## XDNA ResNet compilation and execution

Run the RPC server on the machine with IRON, XRT, and the XDNA device. Start it
with that machine's Python environment so compiler subprocesses inherit the
same toolchain:

```bash
export XILINX_XRT=/path/to/xrt
PYTHONPATH=/path/to/onnx-simplifier:$PYTHONPATH \
  python -m onnxsim.rpc server --host 127.0.0.1 --port 9192 --key xdna \
  --xdna-python /path/to/iron-python \
  --vitis-python /path/to/vitis-ai-python
```

The RPC process uses this source checkout for the XDNA scripts.
`--xdna-python` selects the subprocess environment with IRON/XRT, and
`--vitis-python` selects the one with `VitisAIExecutionProvider`. This supports
installations where the two providers are available in separate environments.
For a Ryzen AI venv, the server also sets `RYZEN_AI_INSTALLATION_PATH` and adds
its VOE runtime libraries to the Vitis child's library path, keeping system XRT
libraries first to avoid mixing incompatible XRT versions.

The client can compile a fused bottleneck and quantized MaxPool remotely, then
pass their server-side artifact paths to a graph run:

```python
import onnxsim.rpc as rpc

remote = rpc.connect("127.0.0.1", 9192, key="xdna")
model = "/client/path/resnet.onnx"
build = remote.xdna_compile_resnet(model, "resnet", {
    "example": "/server/path/whole_array.py", "device": "npu2",
    "optimize_small_m": True,
})
block = remote.xdna_compile_resnet(
    model, "fused_bottleneck", {"block": "/layer1/layer1.0", "device": "npu2"}
)
pool = remote.xdna_compile_resnet(model, "maxpool_u8", {
    "channels": 64, "input_height": 18, "input_width": 20,
    "output_height": 8, "output_width": 8,
    "kernel_height": 3, "kernel_width": 3,
    "stride_height": 2, "stride_width": 2,
    "tile_output_rows": 8, "tile_channels": 4,
})
report = remote.xdna_run_resnet(model, build["manifest"], {
    "cpu_small_m": 64, "cpu_backend": "torch", "cpu_threads": 2,
    "warmup": 2, "iters": 10,
    "fused_blocks": [{"prefix": "/layer1/layer1.0",
                      "xclbin": block["xclbin"], "insts": block["insts"]}],
    "maxpool_uint8": {"xclbin": pool["xclbin"], "insts": pool["insts"]},
})
print(report["avg_ms"], report["cpu_reference"])
remote.close()
```

For the best measured schedule, compile the whole bottleneck body as **one** xclbin
(`kind="resnet_body"`; one core column per block kind, so ResNet-50's 16 blocks use 8
columns) and run it with the stem Conv and MaxPool on the host, which avoids every
xclbin switch:

```python
groups = [["/layer1/layer1.0"], ["/layer1/layer1.1", "/layer1/layer1.2"],
          ["/layer2/layer2.0"], ["/layer2/layer2.1", "/layer2/layer2.2", "/layer2/layer2.3"],
          ["/layer3/layer3.0"], [f"/layer3/layer3.{i}" for i in range(1, 6)],
          # layer4 groups: smaller weight chunks + a double-buffered weight FIFO (~-9% on the body)
          {"blocks": ["/layer4/layer4.0"], "chunk_cap": 17000, "depth": 2},
          {"blocks": ["/layer4/layer4.1", "/layer4/layer4.2"], "chunk_cap": 17000, "depth": 2}]
body = remote.xdna_compile_resnet(model, "resnet_body", {"groups": groups})
report = remote.xdna_run_resnet(model, build["manifest"], {
    "cpu_small_m": 256, "cpu_backend": "numpy", "host_maxpool": True,
    "fused_body": {"xclbin": body["xclbin"], "insts": body["insts"], "groups": groups},
    "warmup": 5, "iters": 30,
})
```

The whole network, including the stem Conv and MaxPool, can run on the device as one xclbin with one
core column per bottleneck stage (`kind="resnet_network"`; runtime-shaped kernels; the host only
quantizes the image and runs the classifier tail):

```python
stages = [[f"/layer1/layer1.{i}" for i in range(3)], [f"/layer2/layer2.{i}" for i in range(4)],
          [f"/layer3/layer3.{i}" for i in range(6)], [f"/layer4/layer4.{i}" for i in range(3)]]
net = remote.xdna_compile_resnet(model, "resnet_network", {
    "stages": stages,
    "cols": 8,                       # reach every shim DMA channel
    "split_weights": [0, 0, 1, 1],   # per-core weight streams for the two heavy stages
    "weight_depths": [1, 1, 1, 2],   # double-buffer layer4's weight FIFOs
})
report = remote.xdna_run_resnet(model, build["manifest"], {
    "device_network": {"xclbin": net["xclbin"], "insts": net["insts"], "stages": stages},
    "warmup": 5, "iters": 30,
})   # ~3.6-3.8 ms end to end on the quicktest ResNet, logits identical to ONNX Runtime CPU
```

The layer-sequential engine (`kind="resnet_engine"`) is faster still: every conv layer, plus the stem
Conv and MaxPool, runs as a job spread over all 32 cores (Vitis-AI style), with weights streamed through
the memtiles. The artifact depends only on the ResNet-50 structure; `options["stem"] = False` compiles the
variant that leaves stem/MaxPool to the host:

```python
eng = remote.xdna_compile_resnet(model, "resnet_engine", {})
report = remote.xdna_run_resnet(model, build["manifest"], {
    "layer_engine": {"xclbin": eng["xclbin"], "insts": eng["insts"], "stages": stages},
    "warmup": 5, "iters": 30,
})   # ~1.7 ms end to end on the quicktest ResNet, logits identical to ONNX Runtime CPU
```

All blocks of a group must share shapes and weight chunking (they differ only in
weights and requantization scales). `fused_stage` compiles also accept
`options["blocked"]` (1-8 blocks, vectorized kernels; pass `"blocked": True` on the
matching `fused_stages` run entry).

`xdna_compile_resnet` also supports `kind="resnet"`; provide the server-side
IRON `whole_array.py` path as `options["example"]`. Compiled paths remain on the
server, so compile and run calls must use the same server. Compiler failures
are returned with the subprocess log tail for diagnosis.

### XDNA compile cache

Compiles take 40 s to a few minutes and are usually repeated with identical inputs, so
`xdna_compile_resnet` caches artifacts on the server, content-addressed. The reply has the same
shape as a fresh compile plus `"cache": "hit" | "miss" | "bypass"` and `"cache_key"`; artifact
paths point into the cache entry (`<cache>/<key>/artifacts/...`) and stay valid until the entry is
evicted.

The key is a SHA-256 over: the compile kind; the model bytes; the exact compiler command
(device, columns, `compile_all`, `optimize_small_m`, block(s), groups/chunk caps/depths, `blocked`,
pool geometry, ... -- only options that reach the compiler count, so e.g. `no_cache` does not);
the content hash of the `resnet` `example` file; every `.py`/`.cc`/`.h` file under
`scripts/xdna` (excluding `__pycache__`) plus `onnxsim/rpc/xdna.py`; the IRON python identity
(path, Python version, mlir_aie/aie version, Peano `clang` stat, XRT `version.info`, probed once
per process); and the compile-relevant environment (`XDNA_*` such as `XDNA_BLOCKED_MAX_CHUNK`,
`AIE_*`, `IRON_*`, `PEANO*`, `MLIR_AIE*`, `XILINX_XRT`, `PYTHONPATH`, `PATH`,
`LD_LIBRARY_PATH`). Any change to these misses; nothing else needs manual invalidation.

Entries are built in a temp directory and published by atomic rename under a per-key `flock`, so
concurrent identical requests compile once and a crash never leaves a partial entry. On a hit
every recorded file must still exist with its recorded non-zero size, otherwise the entry is
discarded and rebuilt. The cache keeps the newest entries by last use (LRU by mtime).

| Setting | Meaning |
| --- | --- |
| `options["no_cache"]=True` | skip the cache; compile into a fresh `<work_dir>/xdna-rpc/<uuid>` (`"cache": "bypass"`) |
| `ONNXSIM_XDNA_CACHE=0` | disable the cache server-wide |
| `ONNXSIM_XDNA_CACHE_DIR` / `options["cache_dir"]` | cache root (default `<work_dir>/xdna-cache`) |
| `ONNXSIM_XDNA_CACHE_MAX_ENTRIES` / `options["cache_max_entries"]` | entries kept (default 20) |

For an apples-to-apples full-graph comparison, use
`Session.xdna_compare_vitis_resnet(model, manifest, options)`. It runs the XDNA
graph and then Vitis AI sequentially on the same server, with identical input
seed, warmup count, and measured iteration count. The response includes both
full reports, XDNA per-op timings, summarized Vitis provider events, CPU
reference errors, and a latency ratio only when both backends report real NPU
execution. Both reports must have matching input shape and seed, and the Vitis
trace must contain NPU-assigned nodes for the comparison to be marked valid.
Vitis profiling is performed in a separate untimed pass by default; set
`profile_vitis=False` to skip the trace, in which case NPU node placement and
the latency ratio cannot be verified.

```python
comparison = remote.xdna_compare_vitis_resnet(model, build["manifest"], {
    "warmup": 20, "iters": 200, "seed": 0,
    "cpu_small_m": 64, "cpu_backend": "torch", "cpu_threads": 2,
})
print(comparison["comparison_valid"], comparison.get("xdna_vs_vitis_latency_ratio"))
```

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

For applications that already compile a TVM module, `onnxsim.rpc.tvm_compat`
provides a small version-tolerant wrapper around the installed TVM Python
client. It also works with TVM FFI-enabled builds: the session still comes
from `tvm.rpc`, and uploaded FFI-compatible modules and their exported
functions are loaded and called through TVM's remote module interface. The
adapter returns the TVM module and function objects unchanged, so FFI tensor
and object arguments are marshalled by the installed TVM client rather than
converted by onnxsim.

```python
from onnxsim.rpc import tvm_compat

with tvm_compat.connect_tracker("tracker", 9190, "hexagon") as session:
    session.upload("model.so")
    module = session.load_module("model.so")
    run = module.get_function("run")
    run(...)
```

For an FFI-enabled TVM application, import and use `tvm_ffi` as that module's
API requires; `tvm_compat` does not need to import or wrap FFI values. The
native TVM RPC server must have a compatible TVM runtime and any target-side
runtime libraries available. This does not make a TVM RPC server an ONNX
executor. The binary onnxsim-v5 transport remains the portable protocol for
the remote EP and compiler/runner services.

## Security

A server executes whatever ONNX model a client sends. It binds to loopback by default; the key is
an identifier, not authentication; onnxruntime custom-op libraries are never loaded; uploads are
flattened to a sanitised file name inside the workspace; blobs are size-capped (4 GiB default).
Expose a server only on a trusted link (`adb forward`, an SSH tunnel, a private network).

## Limits

The tracker has no exclusive locking or priorities (it round-robins live registrations), sessions
are one connection each, and dtypes such as bfloat16, strings and float8 are not transferred yet.
