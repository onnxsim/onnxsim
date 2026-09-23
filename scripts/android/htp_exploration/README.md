# Getting a real throughput number for Hexagon's HTP/NPU, not just HVX

Every kernel in `../tinygrad_hexagon_bridge/` targets Hexagon's general-purpose HVX
vector-DSP path (`vrmpy`, accessible through a normal C compiler). The same SoC also
ships a separate, dedicated matrix-multiply block -- the Hexagon Tensor Processor
(HTP), Qualcomm's NPU -- architecturally distinct from HVX and capable of far higher
int8 throughput, but programmed through a completely different, closed toolchain
(QNN, Qualcomm's AI Engine Direct SDK) rather than a C/LLVM compiler target. Nothing
in this project has ever touched it. This directory is a first attempt to get a real
number from it, running the *same* int8 Mask R-CNN backbone this whole project has
been hand-optimizing for HVX, for a direct comparison against the project's ~44.5
GMAC/s hand-tuned HVX peak and TVM's 43 GMAC/s real-backbone HVX baseline (see
`scripts/android/maskrcnn_e2e/README.md` and `../tinygrad_hexagon_bridge/README.md`).

No Hexagon SDK on this machine (`~/.cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2`,
the one every other Hexagon script in this repo uses) ships any HTP/QNN component --
confirmed by searching it for `*htp*`/`*qnn*`, nothing found. So this is explicitly
**not** about hand-writing HTP kernels (no QNN Op-Package SDK is available to do
that with). It's about running the already-int8-quantized backbone through
Qualcomm's own QNN runtime via ONNX Runtime's QNN Execution Provider -- the same
"use the vendor's closed compiler first" approach this whole project originally took
with TVM before hand-writing kernels replaced it for HVX.

## What's available on this machine

- `~/bev-tmp/qnn-env`: a Python venv with `onnxruntime` 1.30.0 + `onnxruntime_qnn`
  2.6.0 (bundling QNN SDK 2.50.40's `libQnnCpu.so`/`libQnnHtp.so` as x86 backend
  libraries) already installed, from an earlier, unrelated project on this machine.
- The real test phone (device `239dbd8f`, Hexagon V69) already has QNN/SNPE HTP
  runtime libraries on its vendor partition (`adb shell find /vendor/lib64 -iname
  '*htp*' -o -iname '*Qnn*'`): `libQnnHtp.so`, `libQnnHtpV69Stub.so`,
  `libSnpeHtpV68Stub.so`, `libSnpeHtpV69Stub.so` -- standard on Qualcomm devices,
  not something this project installed.
- `~/bev-tmp/onnxruntime-android-qnn-2.6.0.aar`: a prebuilt Android ONNX Runtime
  build with the QNN EP linked in, for **stage 2** (real on-device execution,
  not reached in this pass -- see below).

## Stage 1: does QNN even accept this graph, on this host?

Built `../../maskrcnn_e2e/backbone.onnx` (via `prepare.py`, not checked into the
repo) and tried to run it through `QNNExecutionProvider` targeting the bundled
x86 HTP backend library, using the exact API an earlier, working probe on this
machine used (`~/bev-tmp/check_qnn.py`, `providers=["QNNExecutionProvider"]` +
`provider_options=[{"backend_path": ...}]`). Result: the session created, but
**every one of the 204 (post-fusion) nodes silently fell back to
`CPUExecutionProvider`** -- confirmed via `SessionOptions.log_severity_level=0`
verbose graph-partition tracing, which showed zero mentions of "qnn" anywhere in
the entire initialization log, only `Adding default CPU execution provider.` and
`All nodes placed on [CPUExecutionProvider]`.

**Isolated whether this was graph-specific or environmental** with
`make_tiny_qdq_conv.py`, the smallest possible QDQ int8 conv (one
`DequantizeLinear(x) -> DequantizeLinear(w) -> Conv -> QuantizeLinear(y)`,
matching the real backbone's per-layer QDQ pattern exactly). Same result: total,
silent fallback to CPU, even for this trivial one-conv graph. **Not a
graph-support rejection** (unlike TVM's own real "cannot compile the whole
graph" finding for this same model, documented in
`../../maskrcnn_e2e/README.md`'s finding #1) -- this is environmental.

## Root cause, found via `check_qnn_htp.py`

`ort.get_ep_devices()` (ONNX Runtime 1.30's newer plugin-EP device-enumeration
API) shows exactly why:

```
ep='CPUExecutionProvider' vendor='Microsoft' device_type=OrtHardwareDeviceType.CPU device_vendor=AMD
ep='QNNExecutionProvider' vendor='Qualcomm' device_type=OrtHardwareDeviceType.CPU device_vendor=AMD
```

The QNN EP registers correctly, but the **only hardware device ORT can see it
attached to, on this x86 development host, is the CPU** -- there is no Hexagon
HTP device descriptor to route work to, emulated or otherwise, through this
newer device-based selection model. That's why the legacy
`providers=["QNNExecutionProvider"]` call silently fell back (the device
selection policy picked the only device that actually exists here -- CPU) and
why the newer `SessionOptions.add_provider(ep_name, {"backend_path": ...})` call
fails outright with `INVALID_ARGUMENT: Provider configuration is not supported`
(tried with the HTP backend path, an empty config, and a `backend_type` key --
all four combinations rejected identically, ruling out a simple wrong-key-name
bug). This is a version/API-surface mismatch between whatever
`onnxruntime`/`onnxruntime_qnn` version pairing `~/bev-tmp/check_qnn_bev.log`'s
earlier successful run used (that log shows real `qnn-htp` results, `0.0e+00`
error) and the versions actually installed in `~/bev-tmp/qnn-env` right now
(1.30.0 / 2.6.0) -- not re-investigated further in this pass; pinning to
whatever combination the working log came from is the natural next step for
someone picking this up.

## What this does and doesn't establish

**Not established**: any real HTP throughput number, on host or on-device --
the whole point of this exploration. Also not established: whether QNN's HTP
graph compiler would even *accept* this specific real int8 QDQ backbone graph
(the question Stage 1 originally set out to answer) -- the environment-level
block above happened before that question could be tested at all.

**Established**: a precise, reproducible diagnosis of why the host-side probe
doesn't work in the current environment, distinct from (and easier to fix than)
"QNN rejects this graph." `check_qnn_htp.py` reproduces this in about a second
against either the tiny probe model or the real backbone.

## Next steps if picked back up

1. Fix the host-side version mismatch (pin `onnxruntime`/`onnxruntime_qnn` to
   whatever pairing `check_qnn_bev.log`'s original successful run used, or find
   the current API's correct device-selection call -- `add_provider_for_devices`
   takes explicit `OrtEpDevice` objects rather than a bare options dict, and
   wasn't tried here) to get real host-emulated HTP correctness/timing numbers
   for the real backbone graph, the way `check_qnn_bev.py`'s BEVFormer probe
   already did for a different model.
2. Real on-device execution needs a QNN-EP-enabled ONNX Runtime running natively
   on the phone (not the host emulator) -- `~/bev-tmp/onnxruntime-android-qnn-2.6.0.aar`
   is a prebuilt Android library; investigate whether ONNX Runtime + QNN EP can
   be driven from a plain `adb shell`-launched process (the same access pattern
   `native_transport/`'s client and `tvm_rpc_android` already use successfully
   on this exact phone) rather than requiring a full Android app/APK -- not
   attempted in this pass.
3. If/when a real HTP number is obtained, compare against `../tinygrad_hexagon_bridge/README.md`'s
   ~44.5 GMAC/s hand-tuned HVX peak and `../maskrcnn_e2e/README.md`'s 43 GMAC/s
   real-backbone TVM/HVX baseline -- the comparison this whole exploration exists
   to make.

## Result: the whole backbone runs on the HTP, 58 ms (`qnn_shell_findings.md`)

Items 2 and 3 above are done. PR #1810's `libQnnHtp.so` "not found" wall was the *vendor's*
copy being hidden from apps; bundling Qualcomm's own QNN runtime (`com.qualcomm.qti:qnn-runtime`
2.50.0 from Maven Central, matching the QNN 2.50 the ORT QNN EP plugin 2.6.0 was built against)
and driving it from a plain `adb shell` native harness (`qnn_shell/`) needs no root and no
system-app status. All 578 backbone nodes run on the HTP (strict mode, no CPU fallback), in an
unsigned PD with our own `libQnnHtpV69Skel.so`:

| Backbone on the phone | steady state |
|---|---:|
| QNN HTP, burst | **53 ms** (~3.0 TMAC/s) |
| QNN HTP, default | 57 ms |
| ORT CPU, 4 threads | 635 ms |
| TVM int8 on HVX | 3.7 s (43 GMAC/s) |

Accuracy is in the same band as the TVM int8 pipeline (FPN max abs err 1.03 vs host ORT); on
`cats.jpg` all 3 detections match the all-ORT pipeline (box IoU 0.986, mask IoU 0.982). Graph
compile costs ~6 s per process; an EP-context model cuts session creation to ~440 ms.

## Files

- `make_tiny_qdq_conv.py` -- builds the minimal QDQ int8 conv model used to
  isolate the environment-level blocker from a graph-specific one.
- `check_qnn_htp.py` -- probes whether `QNNExecutionProvider` actually routes a
  given model to HTP on this host (device enumeration, both the legacy
  `providers=[...]` API and the newer `SessionOptions.add_provider` API), and
  reports exactly where it fails.
- `qnn_shell/` -- adb-shell native ORT + QNN EP harness: `fetch_libs.sh` (Maven downloads,
  not committed), `run.sh` (build, push, run), `qnn_run.cpp`, `compare.py` (tensor-level vs host
  ORT), `detect_compare.py` (detection-level through `rest.onnx`),
  `make_tiny_qdq_conv_f32io.py` (smoke-test model). See `qnn_shell_findings.md`.
- `qnn_shell_findings.md` -- the working HTP path and its measurements.
