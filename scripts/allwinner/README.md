# Allwinner NPU (Vivante VIP9000): RPC runner and ONNX -> NBG converter

Targets the NPU in Allwinner A733 / T736 (VIP9000NANODI_PLUS, hardware ID `0x1000003b`) and the same VIPLite runtime on T527,
V853 and friends. Two pieces, both plugged into the existing `onnx-remote` compiler/runner split (`tools/onnx-remote`,
`docs/rpc.md`):

| piece | where | what |
|---|---|---|
| **runner** | `tools/onnx-remote/remote_viplite_worker.cpp` | on-device server: `load_compiled` / `run_compiled` for NBG artifacts through VIPLite (`libNBGlinker.so`) |
| **converter** | `scripts/allwinner/compile_nbg.py` | `onnx-remote-compiler --command`: ONNX -> `.nb` by driving Acuity's `pegasus` |

## Status: what is verified and what is not

**Verified on hardware** (iPlay 70 S tablet, A733, Android 15, adb shell, no root): the runner builds with the NDK, opens
`/dev/vipcore` as the unprivileged shell user, loads and runs Allwinner's shipped `.nb` files, and serves
`capabilities` / `load_compiled` / `run_compiled` over TCP to the stock `onnx-remote-client` and `onnx-remote-compiler`
(passthrough mode). Bad artifacts and wrong input sizes come back as errors without killing the worker. The worker's conversion math
(`tools/onnx-remote/viplite_convert.h`: half-float, affine/fixed-point quantize and dequantize, saturation, ties-to-even) is unit-tested
on the host without any SDK (`onnx-remote-viplite-convert` under ctest), and the half-float encoder was also cross-checked against numpy's
`float16` on 13.5 M values; that test found and fixed a round-half-up bug (ties must go to even).

**Not verified: the converter against real Acuity.** Acuity is not publicly downloadable (Allwinner distributes it as a Docker image
through its customer portal), so `compile_nbg.py` has only been exercised against a fake `pegasus`
(`tests/test_allwinner_compile_nbg.py`: step order, flags, per-SoC `--optimize` target, calibration hand-off, manifest, errors). The
command lines mirror Allwinner's `awnpu_model_zoo` scripts, but real-toolkit behaviour -- in particular the `acuitylib` calls in
`acuity_inputmeta.py` and `.npy` calibration files in a `TEXT` dataset -- is untested. Expect to adjust on first contact.

There is no from-scratch ONNX -> NBG compiler here: an NBG is machine code for the NPU and only Acuity emits it.

## Benchmarks

Measured on the A733 tablet (8 CPU cores, schedutil governor, adb shell, screen state and thermals not controlled -- the thermal
zones are not readable from the shell). `scripts/allwinner/bench.py` runs it: 100 timed iterations after a warm-up, repeated 3
times, median of the repeated medians; the range column shows run-to-run spread. "NPU" is `vip_run_network`; "in"/"out" are the
worker's float32 quantize/dequantize; "total" is one `run_compiled` body without the network transfer. Inputs are random.

Current build (output buffers reused, see "Hiding the conversion cost" below), one caller, worker pinned to a big core with
`taskset 80` (`bench.py --taskset 80`); 100 iterations x 3 repeats:

| network | input dims (as the NBG reports them) | NPU ms | in ms | out ms | total ms | calls/s | NPU range | load ms |
|---|---|---|---|---|---|---|---|---|
| deepHeadPose uint8 | 1x1x112x336 | 0.42 | 0.05 | 0.03 | 0.51 | 1980 | 0.35-0.43 | 6.4 |
| LPRNet uint8 (built for MR536) | 1x24x94x3 | 20.20 | 0.04 | 0.20 | 20.41 | 49.0 | 20.13-20.21 | 3.3 |
| RetinaFace pcq | 1x640x640x3 | 8.06 | 5.42 | 0.66 | 14.03 | 71.3 | 8.06-8.12 | 24.0 |
| YOLOv5n uint8 | 1x1x640x1920 | 10.26 | 5.39 | 1.34 | 16.25 | 61.6 | 10.21-10.33 | 23.8 |
| YOLOv5s uint8 (zoo, A733) | 1x640x640x3 | 22.96 | 2.94 | 3.61 | 29.61 | 33.8 | 22.92-22.99 | 68.4 |
| MobileNetV2 pcq (zoo, built for T527) | | rejected: not a valid NBG for this NPU generation | | | | | | |

Before the output-buffer change and without pinning (first measurement), YOLOv5s took 39.2 ms total (out 11.3 ms) and YOLOv5n 21.9 ms
(out 5.4 ms). With output conversion now small, the float32 *input* quantize (4.9 MB read per call for a 640x640x3 network) is the
largest CPU-side cost for RetinaFace and YOLOv5n (5.4 ms each, not further optimized); native uint8 input skips it.

CPU baseline, same tablet and same network: ONNX Runtime 1.26 CPU provider, fp32 `yolov5s_rt.onnx` (the model the zoo's YOLOv5s NBG
is built from), `scripts/allwinner/bench_ort_cpu.cpp`, 10 iterations x 3 repeats:

| CPU threads | median ms | range | NPU speedup (NPU time / incl. conversion) |
|---|---|---|---|
| 1 | 929.3 | 925.3-931.0 | 40x / 24x |
| 4 | 514.6 | 514.5-515.3 | 22x / 13x |
| 8 | 462.6 | 462.5-468.2 | 20x / 12x |

Read these carefully: the CPU runs fp32 and the NPU runs uint8, so this is a speed comparison, not an accuracy-matched one (no
accuracy was measured). The LPRNet figure (20 ms for a tiny 24x94 network) is surprisingly slow and is likely a consequence of
running an MR536-targeted NBG on the A733 rather than a native build; it was not investigated. NBGs built for T527 are rejected,
so a network must be compiled for the right NPU generation.

Feeding a uint8 network its native `UINT8` tensor skips input quantization. Over `adb forward` the float32 tensors dominate
wall time for large models (YOLOv5s: ~440 ms RPC-inclusive); run the client on the device or on the same LAN for real numbers.

## Profiling: what exists

With `--profile` (Summary) or `Detailed`, `run_compiled` returns these events; `--bench` prints their medians.

| event | meaning |
|---|---|
| `viplite_run` | wall time around `vip_run_network` (includes driver submit/wait) |
| `viplite_hw` | the driver's own counters for that run (`VIP_NETWORK_PROP_PROFILING`): hardware inference time, and in `detail` the NPU `cycles`, `layers` and the implied `clock_mhz` |
| `viplite_input` / `viplite_output` (Detailed) | float32 quantize+upload and dequantize in the worker |

On the A733 the effective NPU clock is about **846 MHz** on the big networks (YOLOv5s: 19.1 M cycles, 73 layers, 22.8 ms driver time vs
23.4 ms wall; RetinaFace 6.4 M cycles / 77 layers; YOLOv5n 8.2-9.7 M / 76; deepHeadPose 0.19 M / 18). For tiny networks `clock_mhz`
reads lower (680-800) because the driver's time includes a fixed per-run cost.

### System-level profile from outside the runtime

`scripts/allwinner/trace_npu.py` (needs `pip install perfetto` on the host) records a Perfetto trace on the device while a network
runs and summarizes it. The adb shell can't enable ftrace directly (tracefs is read-only) but the `perfetto` service can, and it
shows things the sysfs nodes hide. Measured on the A733 with YOLOv5s (the trace is also viewable in ui.perfetto.dev):

| observation | value |
|---|---|
| NPU interrupt (`vipcore_0`, also in `/proc/interrupts`) | exactly one per inference (81 runs -> 81 IRQs), handler ~11-17 us |
| NPU temperature (`npu_thermal_zone`; not readable via sysfs from the shell) | ~40 C idle -> 43-44 C after 80 back-to-back runs |
| NPU clock events (`devfreq`, `clk_set_rate`) | none during runs: the clock is not changed through the Linux clock framework (consistent with a steady ~846 MHz) |
| worker thread while the NPU runs | asleep (about 22 ms of each call), no spin-wait; on a CPU only for the float32 conversion and submit |
| completion IRQ -> worker running again | median ~250-350 us (p90 ~350-440): most of the gap between `viplite_run` (23.2 ms) and `viplite_hw` (22.7 ms) |
| which core runs the worker | unpinned: 80% on little cores (capacity 399; the A733 here has 6 little + 2 big cores), call = 44-47 ms; pinned to a big core (`taskset 80`): 39 ms |

The takeaways from that: the NPU time itself (22.7 ms) does not change with CPU placement, but the CPU-side float32 conversion does
(output dequantize 14-17 ms -> 11.8 ms on a big core), so **pin the worker to a big core** (`taskset 80 ./onnx-remote-viplite-worker ...`)
or feed native uint8 where the network allows it. Tracing itself slows the CPU-side phases (not the NPU time), so use `bench.py` for
timings and this for the system view.

### Hiding the conversion cost (two changes the profile led to)

1. **Reuse output buffers.** Of the ~7 ms the worker spent turning YOLOv5s' 1.6M-element output into float32, 5.4 ms was `std::vector::resize`
   zero-filling and page-faulting a fresh 6.5 MB allocation each call; the dequantize loop itself was 1.3 ms. The worker now takes
   output vectors back after a response is sent and reuses them. Worker pinned to a big core (`taskset 80`): output phase 11.8 ms -> 4.0 ms
   median, whole call 39.3 ms -> 30.6-32.2 ms (3 runs); NPU time unchanged.
2. **Overlap conversion with the NPU.** Each network has two independent I/O buffer sets ("slots"); only `vip_run_network` is serialized,
   and the server handles each connection on its own thread (at most 8 in flight, then it answers "busy"). While request A's NPU run
   executes, request B converts its input into the other slot and A's output is converted after A's run. Rebinding the network to the other
   slot's buffers costs ~7 us, and the outputs were bit-identical across slots and across two concurrent TCP clients.
   `--bench FILE.nb N --threads K` measures it (YOLOv5s, worker pinned with `taskset c0`, 200 calls x 3 repeats):

   | concurrent callers | ms per call (throughput) | NPU ms under load |
   |---|---|---|
   | 1 | 31.4 (31.8 calls/s) | 22.4-22.7 |
   | 2 | 25.2 (39.6 calls/s, +25%) | 23.8-24.3 |
   | 3 | 25.5 (39.3 calls/s) | 24.1-24.3 |

   Two pipelined callers reach ~95% of the ceiling set by the NPU's own time under load (24 ms -> 41.7 calls/s); the NPU itself runs a
   little slower while the CPU converts, which looks like DRAM contention (not verified). This is a **throughput** feature: per-call latency
   rises (median ~50 ms with two callers, since each waits for the NPU), and a single serial client sees no gain from it. It only pays off
   when clients send requests concurrently; over `adb forward` the ~13 MB of float32 per call dominates (~570 ms) and hides it.
3. **Pre-warm the CPU cluster (opt-in, costs power).** What was left in the profile is that the CPU work right after an NPU wait runs
   slowly. A standalone loop quantizing 1.2M floats takes 0.8-1.0 ms when the core is hot or after a 1-5 ms sleep, but 3.3-3.7 ms in the
   first few ms after a ~22 ms sleep (a second run immediately after is 1.0 ms again). That is the big cluster's shared clock falling
   while the worker sleeps through the NPU run: a spinner on the *other* big core during the last 4 ms of the sleep brings the
   measured core back to 1.0 ms, while splitting the work over two threads does not help. Setting a scheduler utilization clamp, the
   clean fix, is refused for the shell user (`sched_setattr` -> EPERM). So `--prewarm PCT [--prewarm-cpu N]` runs a helper thread that
   busy-waits for the last PCT% of the network's expected NPU time (a running average of its earlier runs). Worker on core 7
   (`taskset 80`), helper on core 6, 150 calls each:

   | network | no pre-warm | 50% | 70% | 85% |
   |---|---|---|---|---|
   | YOLOv5s (NPU 23 ms) | 31.5 ms | 32.4 | 28.2 | **26.7** (-15%) |
   | RetinaFace (NPU 8 ms) | 14.0 ms | 13.9 | 11.1 | **10.7** (-24%) |
   | YOLOv5n (NPU 10 ms) | 17.0 ms | 17.1 | **12.7** (-25%) | 13.0 |

   It only works when the spinner covers most of the wait (50% does nothing, 70-85% does), so the price is roughly that fraction of one
   big core's power for as long as the worker is serving requests: use it when latency matters more than energy. The first CPU phase after
   the NPU shrinks (input quantize 5.4 -> 1.5-2.9 ms); NPU time and outputs are unchanged (bit-identical in serial, 2-caller and TCP runs).
   With two pipelined callers the CPU is already busy, so it adds little (39.6 -> 41.0 calls/s).

### How close to ideal

**The NPU itself (YOLOv5s): about a quarter of rated peak.** Counting from `yolov5s_rt.onnx`, the model is 8.22 G MACs (16.4 GOP,
60 convolutions). The NPU runs it in 22.96 ms and 19.1 M cycles: 0.72 TOPS, or 430 MAC/cycle. The vendor rating is "up to 3 TOPS"
(the A733 datasheet gives no clock or data type for it), so that is about **24% of peak**; if the array is 2048 MAC/cycle, about 21%. Why
it is not higher cannot be told from here, because there are no per-layer or bandwidth counters. Candidates: early layers with few
channels (3 -> 32) filling the MAC array poorly, 57 SiLU (Sigmoid + Mul) activations and concat/resize layers, and memory traffic:
only 26 M of the model's 87.5 M activation elements are convolution outputs, so up to 175 MB of int8 activations per call if nothing stays
on chip (the real figure is unknown). 20-50% is typical for a small YOLO network on an edge NPU; raising it is a compiler/model matter.

**Everything around the NPU, ideal = the 22.96 ms NPU time:**

| configuration | ms per call | share of ideal |
|---|---|---|
| first measurement (unpinned, fresh output vectors) | 39.2 | 59% |
| pinned to a big core, output vectors reused | 29.6 | 78% |
| plus `--prewarm 85` (costs power) | 26.7 | 86% |
| two pipelined callers | 24.4 per call (41.0 calls/s) | 94% of the 43.6 calls/s ceiling; ~99% of the 41.5 ceiling at the NPU's loaded 24.1 ms |

The 3.7 ms left in the pre-warmed serial case is within about 1.2 ms of a rough floor for converting float32 in and out (~2.2 ms) plus
~0.3 ms interrupt wake-up; native uint8 input removes the input conversion. **Remote clients are far from ideal**: over `adb forward` a
YOLOv5s call is ~440 ms, about 15x the NPU time, all of it moving 13 MB of float32 (4.9 MB in, 8.4 MB out). `run_compiled_native` fixes
the output side only for NBGs that have quantized outputs (not the zoo's YOLOv5s/RetinaFace), and sending uint8 input fixes the input side.

What is **not** available, checked on this device:
- **Per-layer timing or counters** (neither from VIPLite nor from the trace). VIPLite reports whole-network time and cycles plus a layer count, nothing per layer. The library
  has an internal profiling hook but no documented switch; the Vivante-style debug environment variables
  (`VIV_VX_PROFILE`, `VIV_VX_DEBUG_LEVEL`, ...) change nothing here. Per-layer data would need the vendor's offline tools
  (Acuity simulation / profiling in the Docker image) rather than the runtime.
- **Memory bandwidth, utilization, DRAM traffic.** No counters exposed.
- **NPU clock control.** `vip_power_management(SET_FREQUENCY)` returns `VIP_ERROR_NOT_SUPPORTED` (-4) on this driver; the clock stays
  under the kernel's devfreq (not readable from the adb shell). `--fscale N` therefore reports `failed -4` and changes nothing, so a
  compute-bound vs memory-bound frequency sweep is not possible from userspace.

## Build and deploy the runner

The VIPLite headers and `libNBGlinker.so` come from Allwinner's public model zoo
(`https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz`, `common/npuruntime/`); they are not vendored here.

```bash
ZOO=awnpu_model_zoo-*/common/npuruntime
cmake -S tools/onnx-remote -B build-aw -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
  -DBUILD_TESTING=OFF -DONNX_REMOTE_VIPLITE=ON \
  -DVIPLITE_INCLUDE_DIR=$ZOO/include -DVIPLITE_LIBRARY=$ZOO/lib_android/A733/arm64-v8a/libNBGlinker.so
cmake --build build-aw --target onnx-remote-viplite-worker      # Release matters: -O0 is ~10x slower in the conversion loops

adb push build-aw/onnx-remote-viplite-worker model.nb /data/local/tmp/viplite/
adb shell /data/local/tmp/viplite/onnx-remote-viplite-worker --bench /data/local/tmp/viplite/model.nb 20   # smoke test, no network
adb shell "cd /data/local/tmp/viplite && nohup ./onnx-remote-viplite-worker --port 39503 --cache-dir ./cache >worker.log 2>&1 &"
adb forward tcp:39503 tcp:39503
```

At link time `libNBGlinker.so` is the zoo's copy; at run time the device's own `/system/lib64/libNBGlinker.so` is used. For older
SoCs (T527, ...) link their `libVIPlite.so` instead.

The worker listens on **127.0.0.1** by default (`--host 0.0.0.0` to change): it submits whatever NBG a client sends to the NPU.

## Run a model

```bash
onnx-remote-client --capabilities 127.0.0.1 39503          # hardware cid, driver version
# compile (Acuity host) -> load_compiled -> run_compiled, against separate endpoints; ONNX -> .nb needs Acuity:
onnx-remote-compiler --port 39510 --cache-dir ~/.cache/onnxsim-aw --target allwinner-a733 --compiler-id "$ACUITY_VERSION" \
  --command 'python3 scripts/allwinner/compile_nbg.py {input} {output} {manifest} --platform a733 --quant pcq --calib-dir calib/'
onnx-remote-client --compile-run 127.0.0.1 39510 127.0.0.1 39503 model.onnx --input-raw 1:1,3,224,224:input.f32 --iters 10 --profile
# Already have an .nb? Compiler passthrough mode (no --command) accepts it as the "model":
onnx-remote-compiler --port 39510 --cache-dir cc --target allwinner-a733 &
onnx-remote-client --compile-run 127.0.0.1 39510 127.0.0.1 39503 model.nb --input-raw 1:1,3,224,224:input.f32
```

I/O contract: `run_compiled` takes the model's inputs in graph order and returns every output as FLOAT in ONNX (NCHW) order. The
worker quantizes/dequantizes using the scale/zero-point (or fixed-point position) stored in the NBG. An input already in the
network's native integer format is copied through. `--profile` reports `viplite_run` (NPU time), plus input/output conversion
with `Detailed`.

**Native outputs.** `run_compiled_native` (client: `--native-out`) returns outputs as raw bytes in the network's own dtype instead of
float32, with no dequantize on the device: 4x less data for a uint8/int8 output. The response manifest says how to turn them back:
`{"outputs":[{"name":"...","native":true,"quant":"affine","scale":0.481335759,"zero_point":196}, ...]}` with real = (q - zero_point) *
scale (`"dfp"` outputs carry `fixed_point_pos`: real = q * 2^-pos). Outputs whose format has no raw dtype come back as float32 with
`"native":false`. Checked on the A733: deepHeadPose (3 x 66 outputs) and YOLOv5n (630 000 outputs) return 4x fewer bytes
(YOLOv5n: 131 ms -> 80 ms RPC-inclusive over adb), and dequantizing them on the host from the manifest is **bit-identical** to the
device's float32 (the manifest prints the scale with 9 digits; at the default 6 it was off by up to 7e-5, which this check caught).
The zoo's YOLOv5s and RetinaFace NBGs gain nothing: they were compiled with Acuity's post-process node, so their outputs are already
float32 (`"native":false`). A model compiled with `compile_nbg.py` keeps quantized outputs by default (`acuity_inputmeta.py` only adds
that node with `--postproc`), so it benefits.

## Converter

```
compile_nbg.py INPUT.onnx OUTPUT.nb MANIFEST.json [--platform a733] [--quant pcq|uint8|int16|bf16|float]
               [--calib-dir DIR] [--calib-count N] [--input-shape NAME:1,3,224,224] [--no-simplify]
               [--docker-image IMAGE] [--inputmeta-args ' --preproc IMAGE_RGB --mean 0,0,0 --scale 0.0039216']
```

Steps (the zoo's `pegasus_import.sh`, `pegasus_quantize.sh`, `pegasus_export_ovx_nbg.sh`): onnxsim + static-shape check ->
`pegasus import onnx` -> `generate inputmeta` / `postprocess-file` -> `acuity_inputmeta.py` -> `quantize` ->
`export ovxlib --pack-nbg-unify --optimize VIP9000NANODI_PLUS_PID0X1000003B` -> `network_binary.nb`. Environment: either a native
Acuity install (`ACUITY_PATH`, `VIV_SDK`) or `--docker-image` (`AW_NPU_DOCKER_IMAGE`; the image presets those itself).

Limits: single-input models only when quantizing (calibration is one dataset; use `--quant float` for multi-input); Acuity needs
static shapes; the default keeps ONNX semantics (no baked-in image preprocessing, `TENSOR` input) -- pass `--preproc IMAGE_RGB`
to get a uint8 camera-frame input instead. The SoC -> `--optimize` table is copied from the zoo's script; `a733` and `t736` share a
target, and the A733's reported hardware ID (`0x1000003b`) matches it.
