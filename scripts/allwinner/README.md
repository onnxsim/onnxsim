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

Robustness, tested on the A733: 140 back-to-back network replacements (`load_compiled` with a new artifact, alternating a small network and
YOLOv5s) gave 0 failures, and the worker's memory stayed flat after the first 70 (3.8 MB idle -> 33.9 MB, then unchanged to 140; file
descriptors and threads unchanged). Garbage bytes, a valid header declaring an absurd length, a truncated header and an abrupt close
mid-request all leave the worker answering. At most 8 requests are in flight; beyond that it replies "busy". Silent connections used to
hold those 8 slots forever, so any peer that could reach the port could lock everyone else out; each socket call now has an idle limit
(`--io-timeout-ms`, default 30 s, per recv/send so slow transfers that keep progressing are unaffected), and the silent connections are
dropped by the server and the worker recovers by itself. There is no authentication: keep it on loopback or a trusted network.

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

## Transformer models

What is established here comes from Allwinner's NPU operator-support list (v1.5, A733 chapter) and from running exported transformers
through `onnxsim` and `npu_rewrite.py`. **None of it has been through Acuity**, so the operator list is a necessary condition, not a
guarantee, and no transformer has run on the NPU. (The list is marked confidential by Allwinner although it ships in their public
model zoo; only operator names and limits are used here, nothing is copied.)

**Operator coverage.** The ONNX importer is documented as ONNX 1.14.0. Listed: `MatMul`, `Gemm`, `Softmax`, `Erf`, `Tanh`, `Sigmoid`,
`Silu`, `Exp`, `Sqrt`, `Pow`, `Reciprocal`, `ReduceMean`, `Where`, `Cast`, `Gather`, `Transpose`, `Reshape`, `Slice`, `Split`, `Concat`,
`Expand`, `Tile`, `Cumsum`, `TopK`, the comparisons and `Neg`/`Add`/`Sub`/`Mul`. Not listed: **`LayerNormalization`** (opset 17+),
**`Gelu`** (opset 20), **`Div`** (the hardware table does have a divide kernel, so this may be a documentation gap), plus `Not`, `Trilu`,
`Einsum`, `RMSNormalization`, and `Identity`/`Dropout`/`Constant`, which simplification removes.

**Experiment.** Three small transformers written the way Hugging Face writes them (BERT-style post-LayerNorm + GELU; LLaMA-style RMSNorm
+ rotary embeddings + causal mask + SiLU-gated MLP; ViT with a convolutional patch embedding), each exported with torch at opsets 17, 18
and 20 (d=256, 4 heads, 2 layers, 64 tokens):

| model | nodes as exported | after onnxsim | still not listed | after `npu_rewrite.py` |
|---|---|---|---|---|
| BERT-style | 93 (79 at opset 20) | 64 (56) | `LayerNormalization` x5, `Div` x4, `Gelu` x2 (opset 20) | none |
| LLaMA-style | 189-194 | 118 | `Div` x7 | none |
| ViT | 111 (97 at opset 20) | 70 (62) | `LayerNormalization` x4, `Div` x4, `Gelu` x2 (opset 20) | none |

`onnxsim` removes every `Identity` and `Constant` (and the causal mask folds to a constant). `scripts/allwinner/npu_rewrite.py` then
replaces `LayerNormalization` (-> `ReduceMean`, `Sub`, `Mul`, `Sqrt`, `Reciprocal`, with the deviation pre-scaled by 1/16 before it is
squared so that fp16 cannot overflow; see the accuracy study below), `Gelu` (-> `Erf`, or `Tanh` for the approximate form)
and `Div` (-> `Mul` by a folded reciprocal, or `Reciprocal` + `Mul`) and checks the result against the original on random inputs: the
largest difference was 1e-6 absolute (about 1e-6 of the output range), float32 rounding. `compile_nbg.py` runs it by default
(`--no-rewrite` to skip) and records what it changed under `rewrites` in the manifest. It does not guess at `Einsum`, `Trilu` or
`RMSNormalization`: they are reported. Unit tests (`tests/test_allwinner_npu_rewrite.py`) cover both `ReduceMean` conventions (axes as
an attribute before opset 18, as an input after), both `Gelu` forms, constant and tensor divisors, zero divisors and integer division.

**Precision is the real question.** Every listed operator runs on one of two modules: **NN**, the integer MAC engine (i8/u8/i16), or
**PPU**, a programmable unit for fp32/fp16/bf16. Fully connected layers (`fcl2`) are NN-only; `matrixmul`, `softmax`, `layer_norm`, `gelu`
and the elementwise operators exist on both. So an int8-quantized transformer runs its matrix multiplies on the fast engine, while a
float one runs on the PPU, whose speed relative to the NN engine has not been measured here. Post-training int8 is also where transformers
lose accuracy (softmax inputs, LayerNorm statistics, GELU outliers), so the options are `--quant int16` (dynamic fixed point), bf16/float
for the whole graph, or **hybrid quantization**: `compile_nbg.py --hybrid-layers FILE` keeps named layers (softmax, normalization) at 16
bits while the rest is int8, following the zoo's `yolov8_hybrid` flow (a normal quantize, the layer list pasted under
`customized_quantize_layers:` in the `.quantize` file, a second `quantize --hybrid`, and an export from the hybrid files). The file is
`layer_name: dtype` lines such as `ln.0/Add_1_output_0_7: dynamic_fixed_point-i16`; the layer names come from the imported
`model.json` and differ between Acuity versions, so they cannot be generated here. The flow is unit-tested against a fake `pegasus` only.

**Limits to check.** The list gives size limits per operator (softmax input/output dimensions up to 8191, matrix-multiply and
fully-connected operands up to 16383-1048575 depending on the axis, convolution kernels up to 15). Which ONNX axis maps to which
documented axis is not stated, so `npu_rewrite.py` prints a warning for any `Softmax`/`MatMul`/`Gemm` tensor with a dimension above 8191,
for example a 32 000-entry vocabulary projection or a long-sequence attention matrix, as something to check or split. Acuity also needs
static shapes, so a decoder with a KV cache would need fixed maximum lengths with the cache as explicit inputs and outputs; not explored.

**What is plausible.** Encoder-sized models (BERT-small class, ViT-tiny/small, a speech encoder) have the operator coverage and sizes
that fit, subject to the accuracy question above. Autoregressive LLM decoding is limited by memory bandwidth rather than MACs: each token
streams the whole weight set, so tokens per second is at most the DRAM bandwidth divided by the weight bytes (the tablet's bandwidth was
not measured), which makes anything beyond small models impractical, independent of operator support.

**Integer inputs.** A transformer's input is a token-id tensor. The worker now passes `INT32`/`INT64`/`UINT32` tensors through to an NBG
input of the same type (needed for embedding lookups); this is **untested on hardware**, because no network available here has an integer
input.

### Accuracy of a real transformer under the NPU's quantization schemes (DistilBERT, SST-2)

`scripts/allwinner/transformer_accuracy/` measures what each precision scheme the Acuity toolchain offers costs a fine-tuned transformer,
without needing Acuity or the NPU. `study_sst2.py` downloads `distilbert-base-uncased-finetuned-sst-2-english` (safetensors only) and the
SST-2 data, exports it with a fixed shape (batch 16 for speed, 64 tokens; the longest validation sentence is 55), runs it through
`onnxsim` + `npu_rewrite.py`, and evaluates each scheme as emulated by `quantsim.py`, which inserts fake-quantize nodes after every float
tensor and fake-quantizes the weights (`uint8` = asymmetric affine, `pcq` = per-channel symmetric int8 weights with 8-bit activations,
`int16` = dynamic fixed point with power-of-two scales, `fp16`, `bf16`). Hybrid variants leave named tensors in float. Its primitives are
unit-tested against numpy (`tests/test_allwinner_quantsim.py`). **This simulates the schemes; it is not Acuity and not the NPU**: Acuity's
calibration algorithm, operator fusion and internal precisions may differ, and speed is not measured at all.

The pipeline itself is sound: PyTorch fp32 gives **91.06%** on the 872 SST-2 validation sentences (the model's published figure), and the
converted model (simplified, LayerNorm/Div rewritten, only documented operators) gives 91.06% with 100% prediction agreement.

| scheme (emulated) | accuracy | vs fp32 | predictions agreeing with fp32 | logit error (RMSE / range) |
|---|---|---|---|---|
| fp32 | 91.06% | | | |
| fp16, every tensor and weight | 90.94% | -0.12 | 99.89% | 0.0005 |
| bf16, every tensor and weight | 91.06% | 0.00 | 100% | 0.0034 |
| int16 fixed point; softmax input fp16; LayerNorm/GELU interiors float | 91.06% | 0.00 | 100% | 0.0010 |
| int16 fixed point; softmax input fp16; LayerNorm/GELU per-op | 91.17% | +0.11 | 98.05% | 0.161 |
| int16 fixed point, everything | 75.92% | -15.14 | 75.46% | 0.306 |
| weights only, int8 per-channel (activations float) | 90.60% | -0.46 | 99.31% | 0.0082 |
| weights only, uint8 per-tensor | 90.25% | -0.81 | 98.51% | 0.0255 |
| **int8 on matmul inputs only, rest fp16**, pcq, min/max calibration | 90.48% | -0.58 | 98.51% | 0.0265 |
| same, moving-average calibration | 90.37% | -0.69 | 98.17% | 0.0310 |
| same, 99.9th-percentile calibration | 86.93% | -4.13 | 88.07% | 0.183 |
| int8 on every tensor (softmax input fp16, LayerNorm/GELU interiors float), min/max | 85.32% | -5.74 | 88.30% | 0.350 |
| same, moving-average calibration | 88.88% | -2.18 | 93.46% | 0.246 |
| same, 99.9th-percentile calibration | 84.86% | -6.20 | 87.84% | 0.198 |

Calibration was 128 training sentences. Repeating the two best 8-bit configurations over three different draws of 16, 64 and 256
sentences (nine calibration sets each) gave: int8 on matmul inputs only, mean 90.33% (89.68-90.94%, 97.7-98.6% agreement) with min/max
and 90.29% (90.02-90.60%) with moving average; int8 on every tensor, moving average, mean 88.95% (87.61-89.91%, 90.4-95.0% agreement).
More calibration data did not help. With 872 sentences the standard error of one accuracy is about 1 point, so accuracy differences
under a point are not significant on their own; the agreement and logit-error columns are paired and tighter.

What it found:

1. **A decomposed LayerNorm overflows fp16.** The first fp16 run lost 0.7 points (6% logit error). Bisecting by tensor put all of it on
   one: the squared deviation `d*d`, whose range reaches 3.4e5, above fp16's 65504, because DistilBERT's residual stream has outlier
   channels with deviations up to +-582. The sum of squares becomes infinity and the normalized output zero. `npu_rewrite.py` now
   pre-scales the deviation by 1/16 before squaring (mathematically identical, one extra multiply), which brought fp16 from 90.37% to 90.94%
   and the logit error from 0.059 to 0.0005. Any fp16 execution of the decomposed form would hit this unless the compiler fuses
   LayerNorm with a wider internal accumulator; whether Acuity does is unknown.
2. **The attention padding mask breaks any integer scheme.** The mask is added to the attention scores with fill value -3.4e38, so that
   tensor's calibrated range is 3.4e38 and a fixed-point or 8-bit scale rounds every real score to zero: int16 everywhere falls to 75.9%,
   and 8-bit schemes with nothing kept float sat at 52-53% (chance; run before the LayerNorm change, which does not affect this).
   Keeping just the six scores-plus-mask tensors in fp16 brings int16 back to 91.17%. In a toolchain whose hybrid layers only offer
   16-bit fixed point that tensor cannot be kept this way, so feed fixed-length inputs without padding, or test a modest fill value
   (not tried here), or use float for the graph.
3. **Outlier channels make 8-bit on every tensor expensive.** Quantizing all tensors to 8 bits costs 2-6 points, and the cost depends
   on calibration. Clipping at the 99.9th percentile is worse than min/max (-4.1 versus -0.6 on matmul inputs) because the outliers
   carry signal. Keeping everything except matmul inputs in 16-bit float (the emulated matrix-engine split) costs about 0.7 points.
4. **16-bit schemes are near-lossless.** bf16 and int16 fixed point (once the mask tensor and the fused LayerNorm/GELU interiors are
   float) match fp32; per-op int16 LayerNorm interiors alone cost a 16% logit error even though accuracy happens to hold.
5. Weight quantization alone is cheap (-0.5 int8 per-channel, -0.8 uint8 per-tensor); the loss is in the activations.

Not done: another architecture or task, perplexity for a decoder, the real Acuity or NPU. The conclusions are about the schemes, on one
model and one dataset.
