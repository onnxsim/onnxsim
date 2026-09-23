# The whole int8 backbone runs on the HTP: 58 ms, all 578 nodes, from a plain adb shell

Follow-up to `jnifix_findings.md` (PR #1810), which stopped at `dlopen failed: library
"libQnnHtp.so" not found` and concluded no HTP number was reachable on this phone. That
conclusion was about the wrong library: the app was trying to load the **vendor's** copy
(`/vendor/lib64/libQnnHtp.so`), which the OEM's `/vendor/etc/public.libraries.txt` hides from
apps. The `onnxruntime-android-qnn-2.6.0.aar` the app used ships only the ORT QNN EP plugin
(`libonnxruntime_providers_qnn.so`), not Qualcomm's QNN runtime. Qualcomm publishes that runtime
separately on Maven Central (`com.qualcomm.qti:qnn-runtime`), including the DSP-side skel. With
our own copies there is no vendor library to be hidden, and no root or system-app status needed.

## What was run

`qnn_shell/` is a small native harness (`qnn_run.cpp`, ORT C++ API, NDK-built, static libc++),
run from `adb shell` the same way `../tinygrad_hexagon_bridge/native_transport/` drives the CDSP.
Everything it loads comes from Maven Central (`qnn_shell/fetch_libs.sh`, not committed):

| Library | Source |
|---|---|
| `libonnxruntime.so` | `com.microsoft.onnxruntime:onnxruntime-android:1.26.0` |
| `libonnxruntime_providers_qnn.so` | `com.qualcomm.qti:onnxruntime-android-qnn:2.6.0` (same SHA as `~/bev-tmp`'s AAR) |
| `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`, `libQnnHtpV69Stub.so`, `libQnnHtpV69Skel.so` | `com.qualcomm.qti:qnn-runtime:2.50.0` |

Versions matter: the plugin was built against QNN 2.50 (the `onnxruntime-qnn` 2.6.0 wheel's
`build_and_package_info.py` says `qnn_version = '2.50.40'`); `qnn-runtime` 2.50.0 is the matching
Maven release. `ADSP_LIBRARY_PATH` lists our directory first so the DSP loads our skel.

logcat confirms genuine HTP execution, in an unsigned protection domain, with our skel:

```
remote_session_control Unsigned PD enable 1 request for domain 3
cdsprpcd: Successfully opened file /vendor/dsp/cdsp/fastrpc_shell_unsigned_3
Created user PD on domain 3 (attrs 0x8, debug_trace 0x0)
Successfully opened file libQnnHtpV69Skel.so
remote_handle64_open: Successfully opened handle ... for file:///libQnnHtpV69Skel.so?qnn_2_50_0_skel_handle_invoke&_modver=1.0&_dom=cdsp on domain 3
```

## Partitioning: everything on HTP

The real `backbone.onnx` (578 nodes: 76 Conv, 280 DequantizeLinear, 141 QuantizeLinear, Add,
Relu, Resize, MaxPool, Sigmoid, Reshape, Transpose, Unsqueeze) runs with
`session.disable_cpu_ep_fallback=1`, which makes session creation fail if any node is assigned
to the CPU EP. It succeeded, so all 578 nodes run inside the QNN/HTP partition. No op needed a
workaround.

## Correctness (one real image, `cats.jpg` at 800x1088)

Against host ONNX Runtime CPU on the same input (`qnn_shell/compare.py`): FPN feature maps are
within 1.03 max / ~0.10 mean absolute error on a roughly ±14 range; RPN objectness within 0.13 max;
box deltas within 0.21 max. That is the same band `../maskrcnn_e2e/README.md` reports for the TVM
int8 pipeline (max abs err 1.025 vs host ORT): int8 kernels rounding differently from ORT's
`QLinearConv`, not a defect.

Detections, HTP backbone feeding `rest.onnx` on host ORT vs. the all-ORT pipeline
(`qnn_shell/detect_compare.py`, same matching as the main README):

| Ref dets | HTP dets | Matched | Mean box IoU | Mean score \|Δ\| | Mean mask IoU |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 3 | 0.986 | 0.022 | 0.982 |

Only one image: the README's six COCO val images aren't on this machine.

## Latency on the phone (backbone only, 159.2 GMAC)

| | first call | steady state | throughput |
|---|---:|---:|---:|
| **QNN HTP**, default perf mode | 83 ms | 57–58 ms | ~2.8 TMAC/s |
| **QNN HTP**, `htp_performance_mode=burst` | 74 ms | 53 ms | ~3.0 TMAC/s |
| TVM int8 on HVX (`../maskrcnn_e2e/README.md`) | | 3.7 s | 43 GMAC/s |
| ORT CPU on the phone, 4 threads | | 635 ms | |
| ORT CPU on the phone, 1 thread | | 2.5 s | |

Setup cost: compiling the graph for HTP takes ~6.0–6.2 s at session creation. With an EP-context
model (`Ort::CompileModel`, embed mode, 30 MB `.onnx`) a fresh process creates the session in
~440 ms and runs at the same speed; the first two runs are still ~80 ms (warm-up), then ~57 ms.

The HTP runs the whole backbone about **65x faster than the TVM/HVX path** and ~67x faster than
this project's best hand-written HVX kernel throughput (~44.5 GMAC/s on its best shapes). It is
also 11x faster than ORT on all four big CPU cores. The HVX work in `../tinygrad_hexagon_bridge/`
is general-purpose vector code on the DSP's scalar core; the HTP is the dedicated matrix engine,
and this is the size of that gap.

## The vendor's own libQnnHtp.so

An adb-shell process is not subject to the app linker namespace, so it *can* `dlopen`
`/vendor/lib64/libQnnHtp.so` (no "not found"). It then segfaults on first use with the 2.50
plugin: the vendor build is an older QNN release. Bundling the matching runtime is the way.

## Not done

- The Android app path (`scripts/android/app`): the same fix applies (add
  `com.qualcomm.qti:qnn-runtime:2.50.0` so the libs land in the app's own `lib/` dir); not built
  or run here.
- Multiple images / COCO evaluation; the rest of the model (`rest.onnx`) on the HTP.
