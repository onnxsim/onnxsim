# WebGPU (Vulkan) on the Snapdragon 8+ Gen 1 phone: implementation survey

Target: SM8475 (Adreno 730), `vulkan.adreno.so`, Vulkan 1.3 (conformance 1.2.2).
Question: can we run ONNX models through a WebGPU implementation on its Vulkan backend?

Provenance: a web-only survey. Only a few pages were read in full. Items marked
*unverified* come from memory or search snippets. No Adreno 730 numbers were found for any
WebGPU runtime, so performance has to be measured on the device.

## Comparison

| Stack | State (2026) | Android/Vulkan | ONNX maturity | Adreno notes |
|---|---|---|---|---|
| ORT WebGPU EP (Dawn + Tint) | Active (Microsoft, Google) | Listed as supported; `tools/ci_build/github/android/default_full_aar_build_settings.json` passes `--use_webgpu` | Best of the group: broad ops, MatMulNBits and LLM kernels | Real bugs, see below |
| Dawn alone | Active | Vulkan backend | Not an ML runtime; it is what ORT uses | Same bugs (Tint: WGSL to SPIR-V) |
| wgpu / wgpu-native (Rust, naga) | Very active | Vulkan | No ONNX runtime | Adreno 730 storage-buffer corruption reported downstream (bevy #24926); f16 and subgroups are opt-in |
| wonnx | Archived 2025-05 | Vulkan | ~100 ops, no int64, static shapes only | Dead; not viable |
| burn (CubeCL wgpu/Vulkan) | Active | Vulkan, WebGPU | Own format; ONNX only via import converter | No Adreno data |
| tract, candle | CPU-centric | No GPU path found | tract ONNX coverage good on CPU | n/a |
| emdawnwebgpu | Browser port of Dawn's webgpu.h | Not for native Android | n/a | n/a |
| ncnn / MNN (non-WebGPU reference) | Active | Mature on Adreno (Vulkan/OpenCL) | Converters | Hand-tuned; Adreno OpenCL usually faster than Vulkan (*unverified*) |
| tinygrad, TVM (non-WebGPU reference) | Active | tinygrad: QCOM/OpenCL; TVM: Vulkan/OpenCL | TVM ONNX frontend solid | Romou (MobiCom'22): 6.6-9.1x over stock TVM on Adreno |

Not researched: Diligent, sokol, webgpu-native headers (graphics/header layers, no ML runtime).

## Known Adreno / Dawn problems

- **Subgroups crash on Adreno 750.** ORT 1.30's MatMulNBits subgroup-shuffle path segfaults
  Qualcomm's shader compiler (`libllvm-qgl.so`). Adreno 660 is unaffected. Workaround: create the
  device without the `subgroups` feature. https://github.com/musetric/musetric/issues/901
- **Adreno 640 subgroups.** `subgroupBroadcast` makes `CreateComputePipelines` fail.
  https://issues.chromium.org/issues/351745820
- **Adreno 7xx pipeline failures** (Chrome 153): 8 of 23 compute pipelines fail with
  `VK_ERROR_UNKNOWN`. The shared trigger is a WGSL struct that is both the element type of a
  `var<workgroup>` array and stored whole into a `var<storage, read_write>` array. Workaround:
  split the struct into scalars. https://github.com/linebender/vello/issues/1957,
  https://github.com/francisdb/cuelight/issues/237
- **Adreno 6xx with Vulkan 1.1**: Dawn adapter creation fails (missing extensions). Our 730 has
  a 1.3 driver, so this likely does not apply.
- **Adreno 750 f16 miscompile**: f16vec4 prefetch arrays carried across loops next to fp32
  accumulators (SimdPaddleOCR PR #22). Unverified on the 730.
- llama.cpp Vulkan has Adreno compiler bugs and large-batch failures (#5186, #8743).
- Not found: f16, `packed_4x8_integer_dot_product` and max-workgroup-size limits for the 730.
  Query them on the device (`vulkaninfo`).

## Recommendation / plan

1. Use the ORT WebGPU EP. It is the only ONNX-capable WebGPU path worth trying.
2. Build with `--android --android_abi arm64-v8a --use_webgpu` (out of tree, one heavy job at a time).
3. Start with `subgroups` and `shader-f16` disabled, then enable each separately and compare
   outputs to the CPU EP to catch miscompiles.
4. Keep ncnn/MNN or the QNN path as the performance baseline; WebGPU on Adreno will probably not beat them.

## Sources

- https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
- https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/support.md
- https://github.com/webonnx/wonnx
- https://github.com/tracel-ai/burn
- https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md
- https://github.com/bevyengine/bevy/issues/24926
- https://microsoft.com/en-us/research/uploads/prod/2022/02/mobigpu_mobicom22_camera.pdf

## Measured on the phone (2026-09-29)

Build: ORT `125ea21` (main, Aug 2026), `--android --android_abi arm64-v8a --android_api 29
--use_webgpu` (static Dawn/Vulkan inside `libonnxruntime.so`, NDK r27.2, ~32 MB). Test: a
native C API program on the Adreno 730 via `adb`, WebGPU EP defaults, device features not
restricted (the EP has no option to switch off f16/subgroups).

- The EP loads and runs on Vulkan. Small graphs: ~0.9-2 ms per run for a Conv-Relu-GAP-MatMul-Softmax net.
- Correct on device: Relu, GlobalAveragePool, MatMul (3072x10), Softmax, Transpose `[0,1,3,2]`.
- **Wrong on device: `Transpose perm=[0,2,3,1]` (NCHW->NHWC) and therefore every Conv.**
  A lone NCHW->NHWC transpose of a 1x3x32x32 tensor is 50% wrong: output columns 8-15 and 24-31
  are bad in every row and channel, columns 0-7 and 16-23 are right (an 8-wide alternating
  pattern, consistent with a tiled/workgroup kernel problem). Conv variants (3x3, 1x1, padded,
  unpadded, 1/4/16 output channels) are 59-96% wrong, max abs error up to ~1.2.
- The phone CPU EP matches the host CPU exactly on the same model, so the model and input are fine.

### Core-op sweep (235 single-op models, `webgpu_ops/`)

`webgpu_ops/gen_ops.py` builds the models (fp32, input 1x3x32x32, opset 20): unary and binary
elementwise, broadcasts, comparisons, reductions, ArgMax/Min, Softmax, CumSum, Reshape/Concat/
Split/Slice/Pad/Tile/Expand/Gather/Resize, all 24 rank-4 Transposes, pooling, norms, MatMul/Gemm,
Conv variants and ConvTranspose. Each runs on the phone and is compared to host CPU ORT.

- Default settings: **195 pass, 40 fail.**
- **Root cause: ORT's shared-memory *tiled* Transpose kernel** (`use_shared` in
  `core/providers/webgpu/tensor/transpose.cc`: 16x16 workgroup, `var<workgroup>` tile,
  `workgroupBarrier()`). It runs for any transpose that coalesces to 2-D, and for the
  NCHW<->NHWC transposes ORT inserts around layout-sensitive ops.
  - Failing explicit perms: 0231 0312 2031 2301 2310 3012 3102 3120, and a plain 2-D `[96,32]`
    transpose. The other 16 rank-4 perms and both rank-3 cases pass (they take the plain kernel).
  - Everything else that failed goes through those inserted transposes: all Conv variants,
    ConvTranspose, MaxPool, AveragePool, GlobalMaxPool, BatchNormalization, SpaceToDepth.
- **Confirmed two ways.** (1) With `preferredLayout=NCHW` (provider option key `preferredLayout`,
  not the `ep.webgpuexecutionprovider.` config name) the layout-inserted failures go away, except
  `Conv` with 1 output channel. (2) With the tiled path disabled
  (an experiment; that env-var patch has since been replaced, see the fix below) **all 40
  Transpose/Conv/Pool/BN/S2D failures pass in the default NHWC layout, including that 1-channel Conv.**
- The remaining 5 mismatches (Equal, Greater, LessOrEqual, And, Cast-to-int) were a test artifact: the probe built
  its input in float32 on the phone while the host used float64-then-round, so exact-tie thresholds differed by 1 ulp
  (the phone CPU EP showed the same). With the input written to `x.bin` and read by both sides, all 235 match.
- **Host reference (same ORT revision, unpatched, x86-64 build):** the RTX 5050 (NVIDIA
  driver), the Radeon 8060S (RADV) and lavapipe (software Vulkan) each pass 230/235; the 5
  misses are the same tie artifacts. So the tiled Transpose, and everything that goes through
  it, is correct on three other Vulkan implementations, including Tint's output on them. The
  fault is specific to the Adreno 730 Vulkan stack (its driver/shader compiler) rather than ORT's
  kernel logic. Driver selected per run with `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/<x>_icd.json`.
- **Construct isolated** (`webgpu_ops/dawn_repro/`, a standalone Dawn harness plus WGSL kernels
  using the same Dawn/Tint as ORT): the tile's odd row stride. ORT declares
  `tile: array<array<f32, tile_size + 1>, tile_size>` (stride 17) and the guarded store/barrier
  pattern around it miscompiles on the Adreno 730 (Qualcomm Vulkan 512.615.0, compiler
  EV031.36.08.11). ORT's kernel is wrong even at 1x16 (50%) and 256x256 (26%); the same kernel
  with stride 16 is correct on 8 shapes; a flat stride-17 array is still wrong; the same WGSL is
  correct on the RTX 5050, RADV and lavapipe. Dropping the bounds guards also "fixes" it but is
  not general; adding `storageBarrier()` leaves 2.4% wrong (a race-like symptom).
- Tint/Dawn have Qualcomm-gated workarounds (matrix pass-by-pointer, std140 column vectors,
  `NClamp` scalarization, uniform vector component loads, command-buffer splits) but none for
  workgroup memory or barriers, so this kernel gets Tint's normal output.
- **Fix, verified through ORT:** `webgpu_ops/ort_transpose_qualcomm_unpadded_tile.patch` uses an
  unpadded tile when `adapter_info.vendor == "qualcomm"` (Dawn reports `qualcomm` / `adreno-7xx`
  here). With it the full sweep passes **230/235 on the phone at default settings**; the other 5
  are the tie artifacts. The padded tile is kept for other vendors (avoids bank conflicts).
- Not yet known: whether other Adreno generations (6xx, 8xx) share it, and the kernel-speed cost
  of the unpadded tile on Adreno.

### Ops with no WebGPU kernel (CPU fallback), now implemented

A pass in the sweep does not mean the op ran on the GPU: the WebGPU EP silently assigns unsupported nodes to
the CPU EP. `probe.cc` with `PROFILE=<prefix>` writes ORT's profile, and `placement.py` reports every node that
ran on the CPU EP. In the 235-op sweep that found eight ops without a WebGPU kernel: **Sign, Round, Softsign,
Selu, IsNaN, LogSoftmax, SpaceToDepth, LRN** (ArgMax/ArgMin/DepthToSpace are registered; the only other CPU
nodes are the test's own int64 `Cast`s, which need the EP option `enableInt64`).

`webgpu_ops/ort_webgpu_missing_ops.patch` (applies on ORT `125ea21` after the Transpose patch; new files under
`onnxruntime/core/` need `git add -f` because of a global `core` gitignore pattern) adds them:

- Sign, Round (WGSL `round` is ties-to-even like ONNX), Softsign, Selu (alpha/gamma baked into the shader and the
  cache hint), opsets as in ONNX incl. 22.
- IsNaN: float32 only, bool output; WGSL has no `isnan` and the compiler may assume no NaNs, so it tests the bit
  pattern (`bitcast<u32>(x) & 0x7fffffff > 0x7f800000`).
- LogSoftmax: the Softmax kernel with a flag: `(x - max) - log(sum)`, no clamp; opsets 1/11/13 like Softmax.
- SpaceToDepth: reuses the generic permutation program of DepthToSpace (NCHW perm `[0,3,5,1,2,4]`, NHWC
  `[0,1,3,2,4,5]`); LRN: a new per-element kernel, sum of squares over the channel window in f32, NCHW and NHWC via a
  channel stride. Both need `kMSInternalNHWCDomain` registrations too, because ORT's layout transformer rewrites them
  to NHWC on WebGPU (the first build failed at session creation with "Kernel not found: com.ms.internal.nhwc.LRN").

Verification on the phone (`gen_new_op_tests.py`, 48 more models: NaN/Inf/denormal inputs spliced in with `Where`,
exact halves and large values for Round, custom Selu alpha/gamma, block sizes 2/4/8 and rectangular/batched
SpaceToDepth, a S2D->D2S round trip, LogSoftmax on all axes / 3072-long rows / very large logits / opset 11, LRN
with sizes 1-7 at strong alpha): **283 models, 275 match host-CPU ORT; the other 8 are even-size LRN, which ORT's
CPU LRN rejects, and all of LRN (incl. those) matches a float64 numpy reference to ~2e-7.** `placement.py` then
shows 499 nodes on WebGPU and only the 6 int64 `Cast`s on CPU.

Not covered: float16 (the new kernels accept it via `WebGpuSupportedFloatTypes`, but only fp32 was run), other
Adreno generations, kernel speed (correctness only so far), and ORT's own unit tests (not built here).


## Speed on the Adreno 730 (2026-09-30)

Setup: ORT `125ea21` + the Transpose and missing-ops patches, WebGPU EP at defaults (NHWC layout, fp32),
against the same build's CPU EP with 4 threads. Median of 10-30 timed runs after warm-up, under the phone
lock (`bench.cc`, `bench_all.sh`, raw output in `webgpu_ops/bench_results.txt`). GFLOP counts Conv/MatMul/
Gemm/ConvTranspose only (`model_flops.py`; 8.18 for ResNet-50 and 6.54 for YOLO11n match the published
figures). The demo app's models are QDQ-quantized for the Hexagon, so they were converted to float twins
(`qdq_to_float.py`: same architecture and dequantized weights, no activation quantization) to run on
WebGPU. Outputs of every model match the CPU EP on the phone (worst relative error 5.6e-5, most ~1e-6).

The GPU ceiling was measured, not taken from a datasheet: a Dawn FMA loop (`dawn_repro/peak.cc`) reaches
**986 GFLOPS fp32** with 64 independent scalar chains (it was still rising slowly; 540 with vec2 and 8
chains). `shader-f16` exists on the adapter, but f16 FMA measured only ~310 GFLOPS, slower than f32.

| model | GFLOP | CPU EP x4 | WebGPU EP | first run | WebGPU GFLOPS | % of measured peak | ideal at peak |
|---|---|---|---|---|---|---|---|
| ResNet-50 (224) | 8.18 | 66.8 ms | 66.9 ms | 260 ms | 122 | 12.4% | 8.3 ms |
| YOLO11n | 6.54 | 65.9 ms | 73.7 ms | 546 ms | 89 | 9.0% | 6.6 ms |
| YOLO26n | 5.48 | 56.3 ms | 66.1 ms | 404 ms | 83 | 8.4% | 5.6 ms |
| RT-DETR `pre` | 59.43 | 464.4 ms | 396.3 ms | 645 ms | 150 | 15.2% | 60.3 ms |
| RT-DETR `mid0` | 0.81 | 6.2 ms | 19.6 ms | 100 ms | 41 | 4.2% | 0.8 ms |
| RT-DETR `mid1` | 0.81 | 6.2 ms | 19.2 ms | 106 ms | 42 | 4.3% | 0.8 ms |
| RT-DETR `post` | 0.45 | 3.1 ms | 10.8 ms | 60 ms | 41 | 4.2% | 0.5 ms |
| EfficientViT-SAM-L0 encoder (512) | 69.58 | 579.1 ms | 440.5 ms | 611 ms | 158 | 16.0% | 70.6 ms |
| EfficientViT-SAM-L0 decoder | 3.62 | 41.0 ms | 64.7 ms | 497 ms | 56 | 5.7% | 3.7 ms |

Against the repo's Hexagon and tinygrad numbers for the same demo models (those are quantized QNN HTP runs
or fp16 tinygrad, so not iso-precision; sources in `tinygrad_aot/README.md`, `vision_models/*/README.md`):

| model | Hexagon HTP | tinygrad, Adreno OpenCL fp16 | ORT WebGPU fp32 | WebGPU / HTP |
|---|---|---|---|---|
| YOLO11n | 2.58 ms (int8) | 44.3 ms | 73.7 ms | 29x |
| YOLO26n | 2.5 ms (int8) | 44.8 ms | 66.1 ms | 26x |
| RT-DETR `pre` (default `bb8enc16`, uint8 value maps) | 11.8 ms | | 396 ms | 34x |
| RT-DETR `mid0+mid1+post` | 3.06 ms | | 49.6 ms | 16x |
| SAM-L0 encoder | 41.8 ms | | 440.5 ms | 10.5x |
| SAM-L0 decoder | 11.0 ms | | 64.7 ms | 5.9x |
| ResNet-50 | no number in the repo | | 66.9 ms | |

(The RT-DETR MSDA kernel, 3.03 ms on the HVX, has no WebGPU counterpart here.) My CPU EP x4 numbers agree
with the repo's: SAM decoder 41.0 ms here against 44.5 ms recorded.

What the numbers say:

- **WebGPU is on par with the 4-thread CPU on Conv-heavy models** (0.76-1.17x the CPU's time: faster on RT-DETR
  `pre` and the SAM encoder, slower on YOLO) and 1.6-3.5x slower on small, many-node ones (SAM decoder 1.6x,
  RT-DETR `mid` 3.1x, `post` 3.5x). It is 1.5-1.7x slower than tinygrad's fp16 OpenCL on
  this same GPU and 26-34x slower than the HTP on the vision models.
- **It reaches 4-16% of the measured FMA ceiling**, so the Conv kernels leave most of the GPU idle; ResNet-50
  would take ~8 ms at the ceiling and takes 67 ms.
- **Per-dispatch cost is ~0.2-0.3 ms**: tiny kernels take 200-300 us in the profiler (an Add over 200k
  elements, a small Transpose), and RT-DETR `post` (30 nodes) costs 10.8 ms. That, not arithmetic, dominates
  the small models.
- **The first run costs 0.26-0.65 s** (shader compilation); an app must warm up at start.
- **YOLO's profile** is dominated by layout Transposes (14.6 ms of encode time) and SiLU, which ORT fuses into
  `QuickGelu` (19.8 ms), on top of the Convs (11.9 ms).
- Tuning options do not help: `validationMode=disabled`, `maxNumPendingDispatches` 64/256 and
  `storageBufferCacheMode=bucket` are within noise or slightly worse; `preferredLayout=NCHW` is 2x slower;
  fp16 gives nothing (f16 arithmetic is slower than f32 on this driver).
- **Graph capture** returns no output tensors through plain `Run()`, so the 0.4 ms / 30 ms figures it first printed were
  not latencies. Through `RunWithBinding` it works and is measured below (-28..-31% on small models, -3% on ResNet-50).
- The adapter supports `timestamp-query`, and the profile's `Api` events carry per-dispatch durations (I believe
  they are GPU timestamps; not verified). The phone's kgsl clock and governor files need root, so GPU clocks and
  thermal state were not recorded, and the timings include whatever DVFS state the runs happened to be in.

## Graph capture and convolution investigation (2026-09-30)

### Graph capture

`enableGraphCapture=1` works, but not through plain `Run()` (replays return no output tensors). It needs
`RunWithBinding` with the run option `gpu_graph_id=0`, inputs bound with `BindInput` and outputs with
`BindOutputToDevice` (CPU memory info); `bench.cc iobind=1` does this. Timing needs a real GPU sync after every run:
`sync=tiny.onnx` runs and reads back a second tiny model on the same device (its own latency, ~0.3 ms, is subtracted;
baseline numbers with and without it agree). Outputs after replay are identical to the baseline's (same relative
error vs the CPU: 3.6e-7 / 7.7e-7 / 5.6e-7).

| model | baseline | graph capture | change |
|---|---|---|---|
| RT-DETR `post` (30 nodes) | 10.7 ms | 7.4 ms | -31% |
| RT-DETR `mid0` (71 nodes) | 20.4 ms | 14.8 ms | -28% |
| ResNet-50 224 (91 nodes) | 67.3 ms | 65.4 ms | -3% |
| ResNet-50 64x64 input | 30.5 ms | 21.0 ms | -31% |
| ResNet-50 32x32 input | 27.7 ms | 16.9 ms | -39% |

So capture removes roughly 0.1 ms of CPU cost per node, which matters for small, many-node models and not for
GPU-bound ones. Even with capture, ResNet-50 on a 32x32 input (0.17 GFLOP) still takes 17 ms, about 0.19 ms per
dispatch, so the remaining floor is GPU-side.

### Where the time goes (Adreno 730 through WebGPU)

Measured ceilings (`dawn_repro/peak.cc`, `bw.cc`, `chain.cc`), all in Dawn directly:

| resource | measured |
|---|---|
| fp32 FMA, scalar, 64 independent chains | 986 GFLOPS (vec4-typed FMA loops: about half) |
| fp16 FMA (`shader-f16` present) | ~310 GFLOPS |
| vec4 loads, storage buffer | 163 GB/s (10 G loads/s), even for a 4 KB working set |
| vec4 loads, texture | 240 GB/s (15 G loads/s) |
| vec4 loads, workgroup memory | 200-277 GB/s (13-17 G loads/s) |
| one dependent tiny dispatch | ~6 us GPU, ~2 us CPU (a pass per dispatch or a submit per dispatch: 6 / 19 us) |

1. **ORT's Conv is close to the limit of its own design.** Conv2dMM/MatMul stage tiles in workgroup memory and give each
   thread 4x4 outputs, i.e. 8 FMA per workgroup-memory vec4 load. At 17 G loads/s that caps at ~272 GFLOPS; ResNet-50's
   convolutions average 217 GFLOPS (37.7 ms of GPU time for 8.17 GFLOP), about 80% of that bound and 22% of the FMA peak.
2. **Tile shape sweep inside ORT** (env overrides, `ort_tile_env_override_experiment.patch`; outputs verified): the
   default 4 rows per thread on an 8x8 workgroup is best. ResNet-50 / YOLO11n: 66.5 / 73.7 ms default, 71.9 / 79.4 with
   2 rows, 85.2 / 90.3 with 8 rows, 99.9 / 122.5 with 8 rows on an 8x4 workgroup. Other workgroup shapes are rejected by the
   shader generator's constraints. Bigger register tiles lose to register pressure.
3. **Alternatives I wrote did not beat it.** In `dawn_repro/gemm.cc`: register-only vec4 and scalar kernels, textures
   for A and/or B, an NC4HW4-style coalesced layout, and shared-memory kernels reach 109-276 GFLOPS depending on shape
   (about 165 on short-K layers, 230-276 on long-K ones). Summed over ResNet-50's 20 distinct conv shapes the best
   variant per shape projects to 54 ms against ORT's 37.7 ms. Dawn's robustness / Vulkan-memory-model toggles (as ORT
   sets them) change nothing beyond noise (126-168 GFLOPS).
4. **Small grids are *not* the inefficiency (corrected).** My first reading was that batch-1 layers with a 14x14 or 7x7
   output leave the GPU under-occupied: a single dispatch of a toy shared-memory GEMM ran at 50-63 GFLOPS against 137-183 in
   throughput mode, and layers with <=196 output pixels are 46% of ResNet-50's conv time in the profile. The split-K
   experiment below disproves it: multiplying the workgroup count by 4-16 barely changes the kernel time, so ORT's real
   kernel is throughput-bound (workgroup-memory loads), not occupancy-bound. The toy GEMM's single-dispatch figure is a
   property of that toy kernel, not of ORT's.
5. **A fixed per-node cost remains** (~0.1 ms CPU removed by graph capture, ~0.19 ms GPU-side that is not): ResNet-50
   never goes below ~17-26 ms however small the input, so many-node models pay it in full (fusing nodes would help).
6. **Per-kernel timings need care**: the profiler's `Api` events are GPU timestamps, but in profile mode every dispatch
   is its own pass and they sum to 42.5 of ResNet's 69 ms, and some dispatches report an impossible ~5 us (a 231-MFLOP
   convolution cannot take that), so per-layer profile numbers are only indicative; wall-clock medians are the reliable ones.

### Split-K for Conv2dMM (tried, no gain)

`ort_conv2d_splitk_experiment.patch` (on top of the Transpose and missing-ops patches; off unless
`ORT_WGPU_CONV_SPLITK=auto|N`) adds a deterministic two-stage split-K to the channels-last vec4 Conv2dMM:

1. a partial stage: the existing shader with the shared generator's split-K mode (`dispatch_z = S`, each z-slice covers
   `ceil(K/S)` rounded up to the tile size) writing raw partial sums to slice `z` of a temporary
   `[S, out_h, out_w, C/4]` buffer, with no bias or activation;
2. a reduce kernel that sums the S slices, adds the bias and applies the fused activation (ReLU/Clip/...).

Unlike ORT's own MatMul split-K (Intel-only, atomic compare-exchange adds, needs no fused activation) it needs no atomics and
keeps the fused activation. One bug worth knowing: `mm_readA` derives the output width from `uniforms.result_shape[2]`, so the
temporary buffer must be declared `[S, H, W, C/4]`, not `[S, M, 1, C/4]` (that version got only the first image row right).

Results (fp32, 224x224 ResNet-50; outputs verified against the CPU, relative error <= 8.1e-7 for all settings; single-conv
models with/without bias, ReLU and stride 2 match to <= 7.8e-7):

| ORT_WGPU_CONV_SPLITK | median | change |
|---|---|---|
| off | 66.3 ms | |
| auto (only layers with < 256 workgroups and K >= 512, up to 16 slices) | 66.7 ms | +0.6% |
| 2 | 67.8 ms | +2% |
| 4 | 68.6 ms | +3% |
| 8 | 69.1 ms | +4% |
| 16 | 71.1 ms | +7% |

Individual layers do get 10-20% faster (e.g. the 14x14 256-channel 3x3 conv 1.86 -> 1.49 ms with 4 slices, the 28x28 128-channel
one 1.71 -> 1.44 ms), but the extra partial-sum traffic and reduce dispatch cancel it across the network, and forcing splits
on large-grid layers makes it worse. Since 4-16x more workgroups barely speeds a layer up, ORT's Conv is bound by its
workgroup-memory load throughput, as computed above, and only a design with a higher FMA-per-load ratio (or fewer loads per FMA,
e.g. fp16 storage with fp32 accumulation) can beat it. Not done: fusing Conv+Add+Relu / SiLU to cut the node count.

## Operator fusion (2026-09-30)

Bounds first (nodes deleted from the graph, numerics ignored, timing only), fp32 defaults, wall-clock median:

| candidate | removed | saved |
|---|---|---|
| ResNet-50 ReLU after the residual Add | 16 dispatches | 0.28 ms (0.4%) |
| ResNet-50 residual Add + ReLU | 32 dispatches | 0.36 ms (0.5%) |
| YOLO11n SiLU (`QuickGelu`) after convolutions | 77 dispatches | 3.8 ms (5%) |

So the small elementwise dispatches are cheap (10-25 us in ResNet, ~50 us in YOLO) and the ~0.19 ms per node floor seen earlier comes
from the convolutions, not from them. Residual-Add fusion is not worth doing; SiLU is.

**Conv + SiLU epilogue** (`ort_conv_silu_fusion.patch`, on top of the two base patches). ORT already turns `x * sigmoid(a * x)` into a
`com.microsoft::QuickGelu` node after layout handling. Two small changes make the convolution absorb it:

- `ConvActivationFusion` accepts a `QuickGelu` after a Conv on the WebGPU EP and stores `alpha` (default 1.702) as the fused activation
  parameter;
- the WebGPU EP gets `ActivationKind::QuickGelu`, parsed from the `activation` attribute, and one shader snippet
  `value = value / (1 + exp(-alpha * value))` that every convolution path already shares (Conv2dMM, the MatMul path used for 1x1,
  GroupedConv, Conv3D).

| model | before | Conv+SiLU fused | change | dispatches |
|---|---|---|---|---|
| YOLO11n | 73.5 ms | 70.0 ms | -4.7% | 251 -> 174 |
| YOLO26n | 66.1 ms | 63.4 ms | -4.1% | |
| ResNet-50 | 66.3 ms | 67.0 ms | unchanged (no SiLU) | |

Verified: YOLO11n / YOLO26n / ResNet-50 outputs match the CPU (2.3e-6 / 1.1e-6 / 5.6e-7 relative); the 283-model regression sweep is
unchanged (275 match, 8 even-size LRN checked by numpy); dedicated tests in `gen_new_op_tests.py` cover 1x1, 3x3, with bias, stride 2,
depthwise, `alpha` = 0.5 / 1.702 / 2.0 (all fuse, error <= 5.7e-7) and a SiLU whose pre-activation is also used elsewhere (correctly not
fused). Limit: an explicit `com.microsoft::QuickGelu` node in the source model is not fused, because the transpose optimizer cannot move
the layout transposes through a contrib op, so the Conv is followed by a Transpose; SiLU written as Sigmoid + Mul (what PyTorch exports)
is the case that fuses.

**Graph capture on top** (`iobind=1 enableGraphCapture=1`, with the fusion): YOLO11n 71.3 -> 64.8 ms (-9%), YOLO26n 64.4 -> 57.7 (-10%),
SAM-L0 decoder 65.1 -> 58.1 (-11%), RT-DETR `pre` 396.9 -> 373.1 (-6%). YOLO11n end to end: 73.5 ms -> 64.8 ms (-12%).

**What is left in YOLO11n** (after the fusion, GPU-timestamp sum per kind): Conv2dMM 24.1 ms and MatMul (1x1 convs) 14.0 ms, then Concat
3.5 (23 dispatches), Transpose 3.0 (18), Softmax 2.1, GroupedConv 1.6, Pool 1.5, Split 1.5. ORT's optimized graph has 16 Transposes: about
7 are the detection heads' `Reshape` (needs NCHW order), about 4 sit around attention MatMul/Softmax, and 4 surround the two `Resize` (nearest
2x upsample) nodes, which the EP explicitly keeps out of NHWC (`ShouldConvertDataLayoutForOp`, kernels commented out). An NHWC Resize is
feasible (the nearest path is already axis-generic; the bilinear/trilinear/cubic paths hard-code the last two axes as spatial, so they would
need a transposing fallback) and would remove those 4 Transposes (~1.2 ms of GPU time, an estimated 1-2%); not done.

## Winograd and depthwise microbenchmarks (2026-10-01)

`dawn_repro/conv_alt.cc` (`conv_alt conv C H`, `conv_alt dw C H`; Dawn toggles like ORT: robustness off, Vulkan memory model) compares, per
ResNet-50 3x3 shape (batch 1, NHWC, fp32, all 0.231 GFLOP), a **direct** conv (implicit GEMM with the `sc` scalar-accumulator register-tile
design, 3x3 gather in the A load) against **Winograd F(2,3)** (input transform -> 16 batched register-tile GEMMs -> output transform;
weights are transformed offline, 16/9 of the weight memory). Each is swept over 7 (TM, NV, workgroup) configs; the best is shown. Whole
sequences run back to back (10 per submission, best of 10); outputs match a double-precision CPU reference (max relative error <= 2e-6 for both).
Two full runs, ms (GFLOPS-equivalent = direct-conv FLOPs / time):

| layer | direct, run 1 / run 2 | Winograd, run 1 / run 2 | Winograd / direct time |
|---|---|---|---|
| 64ch @ 56x56 | 1.34 / 1.26 ms (172 / 183) | 1.01 / 0.87 (229 / 265) | 0.75 / 0.69 |
| 128ch @ 28x28 | 1.36 / 1.42 (170 / 162) | 1.17 / 0.72 (198 / 319) | 0.86 / 0.51 |
| 256ch @ 14x14 | 1.49 / 1.40 (155 / 165) | 1.06 / 0.75 (218 / 309) | 0.71 / 0.53 |
| 512ch @ 7x7 | 1.91 / 1.86 (121 / 125) | 0.91 / 0.82 (254 / 283) | 0.48 / 0.44 |

- **Winograd wins on every ResNet 3x3 shape**, by 1.15-2.3x depending on the shape (run-to-run noise on the phone is large, ~20-40% on the
  28x28 and 14x14 rows, so quote ranges, not points); it wins most on the small, channel-heavy late layers (7x7: ~2.2x) where the direct
  kernel has few rows to tile. The GEMM dominates (0.58-0.84 ms); the transforms cost 0.03-0.16 ms each (largest at 56x56).
- Its batched GEMMs run at only 137-177 GFLOPS on the reduced work (K = Cin is short, 16 small dispatches per layer), well below the 255-282
  of the large standalone GEMMs, so tuning the GEMM stage further (larger tiles, fewer z-slices per workgroup) has headroom.
- The direct implicit-GEMM conv reaches 121-183 GFLOPS, below the 255 of the same tile as a plain GEMM: the gather (per-row validity, base
  index per tap) costs ~30%.
- Not measured: fusing the transforms with neighbouring ops (input transform into the previous layer's epilogue, output transform +
  bias/activation/residual), which is where a real Winograd Conv would recover more; F(4,3) (fewer multiplies but larger transforms and
  worse fp32 conditioning); f16.

**Depthwise 3x3** (YOLO-like: 64@160, 128@80, 256@40, 512@20, NHWC), naive ORT-style (one scalar output per thread, 9 bounds-checked loads)
vs vec4-channel kernels producing 1/2/4/8 adjacent x-pixels per thread from one shared input window:

| shape | traffic floor at 163 GB/s | naive | vec4, 1 px | vec4, 2 px | vec4, 4 px | vec4, 8 px |
|---|---|---|---|---|---|---|
| 64ch @ 160 | 0.08 ms | 2.18 ms | 1.97 | 1.15 | **0.93** | 1.15 |
| 128ch @ 80 | 0.04 | 1.23 | 0.93 | 0.59 | **0.45** | 0.56 |
| 256ch @ 40 | 0.02 | 0.61 | 0.54 | 0.35 | **0.25** | 0.26 |
| 512ch @ 20 | 0.01 | 0.27 | 0.26 | 0.17 | **0.13** | 0.14 |

The 4-pixel vec4 kernel is 2.0-2.4x faster than the naive one on all four shapes (correct to 1e-7). It still moves only 12-15 GB/s, ~10x under
the buffer-bandwidth ceiling and ~10x over the traffic floor, so depthwise is latency/issue-bound (27 dependent-ish loads per thread) rather than
bandwidth-bound; the obvious next steps (wider workgroups, staging the window through workgroup memory, f16 storage) were not tried. Depthwise
convs are ~10% of YOLO11n's GPU time, so even a 2x kernel is worth ~5% there. These are standalone kernels, not integrated into ORT.

## Register-tile GEMM vs ORT's shared-memory design, f16 storage, subgroups (Adreno 730, 2026-10-01)

Standalone GEMM microbenchmark (`dawn_repro/gemm.cc`, Dawn toggles `RB=off VMM=1` like ORT, best of 3 runs, warm GPU, ResNet-50 GEMM shapes).
`sh` = ORT's shared-memory tile design (TM=4, 8x8 workgroup); `sc` = register tile, scalar accumulators (TM=8, NV=2, 8x8 outputs per thread);
`p16` = same register tile with A/B stored as packed f16 (one vec4<u32> load = 8 values, unpacked to f32, f32 accumulate; TM=4, NV=2).

First session (GPU cold-clean, phone otherwise idle), GFLOPS:

| M x N x K | `sh` (ORT design) | `sc` TM=8 NV=2 (best wg) |
|---|---|---|
| 784x512x128 | 174 | 282 |
| 3136x256x64 | 161 | 273 |
| 784x128x1152 | 177 | 255 |
| 196x256x2304 | 155 | 224 |
| 196x1024x256 | 162 | 249 |
| 49x512x4608 | 132 | 211 |

`sc` beats the ORT design by 1.4-1.7x on every shape (fp32 storage, so it is a like-for-like comparison). The best workgroup shape varies
per shape (784x128x1152: 16x4 = 255, 32x4 = 141), so a per-shape choice matters.

Second session, run back to back (the phone was shared with tuning jobs and ran ~30-40% slower in absolute terms, so compare only within this table):

| M x N x K | `sh` | `sc` TM=8 NV=2 | `p16` TM=4 NV=2 (best wg) | `sg` (subgroup shuffle) |
|---|---|---|---|---|
| 784x512x128 | 111 | 135 | **180** | 2 |
| 3136x256x64 | 109 | 134 | **179** | n/a (K%128) |
| 784x128x1152 | 92 | 123 | **178** | 15 (wrong) |
| 196x256x2304 | 114 | 119 | **182** | 20 (wrong) |
| 196x1024x256 | 88 | 136 | **195** | 4 (wrong) |
| 49x512x4608 | 86 | 131 | **150** | 21 (wrong) |

- **f16 storage helps**: `p16` is 1.15-1.5x faster than the fp32 register tile and 1.6-2.0x faster than the ORT-style tile, and the register footprint
  drops so TM=4 NV=2 (not TM=8) is the best tile. Errors are at f16-rounded-input level (reference uses the rounded inputs; max abs error 2e-7..1e-5).
  It halves weight/activation traffic, which is the bottleneck, but it needs f16 activations and weights in memory (conversion passes, or an f16 graph),
  and an earlier run of a peak-f16-arithmetic kernel was slower than f32 -- only the *storage* is what wins.
- **Subgroups do not help**: the adapter exposes `Subgroups`, but the `sg` variant (lane l loads one A vec4 and the row group shuffles it with
  `subgroupShuffle`) runs 8-90x slower than `p16` and fails validation on 5 of 6 shapes (wrong when the workgroup x-extent does not line up
  with the hardware subgroup / control flow is not uniform enough). The Adreno shuffle path is not a substitute for the A broadcast the cache already does.

## Register-tile Conv2dMM inside ORT (negative result)

`webgpu_ops/ort_conv2d_regtile_experiment.patch` (on ORT `125ea21` + the base patch stack) adds an opt-in Conv2dMM main loop:
8 output pixels x 8 output channels per thread (64 scalar accumulators, workgroup 16x4, no workgroup memory), with the
im2col address math hoisted out of the channel loop. It is correct (ResNet-50 logits match the CPU to 5.6e-7) but slower
than ORT's shared-memory tile, although the same design is 1.4-1.7x faster as a standalone GEMM (see the GEMM section above).

| `ORT_WEBGPU_CONV_REGTILE` | ResNet-50 median (ms) |
|---|---|
| 0 (default, ORT tile) | 67.0 |
| 2 (1x1 convs only) | 71.8 |
| 3 (3x3 convs only) | 116.7 |
| 1 (all convs) | 120 |

The standalone GEMM gain does not transfer: even the 1x1 layers (pure GEMMs) lose ~5 ms. Untested explanations: register
pressure on the 3x3 kernel, and lower GPU clocks between short dependent dispatches than in the back-to-back batches the
microbenchmark uses. The profiler's per-dispatch timestamps were not usable for a per-layer split.

Rows per thread (`ORT_WEBGPU_CONV_REGTILE_ROWS`, default 8; fewer rows = fewer accumulators): 3x3 convs only: 8 rows 116.7 ms, 4 rows 88.4, 2 rows 82.7;
1x1 convs only: 8 rows 71.8, 4 rows 71.0, 2 rows 71.9. So register pressure explains part of the 3x3 loss, but no setting beats ORT's
shared-memory tile (67.0 ms), and the 1x1 layers do not react to the tile at all -- in the network they are not limited by the GEMM inner loop
the way the back-to-back microbenchmark is (which also re-reads cache-resident operands).

## Winograd F(2,3) Conv in ORT's WebGPU EP (works: ResNet-50 -9%, SAM-L0 encoder -13%, RT-DETR pre -13%)

`webgpu_ops/ort_conv_winograd.patch` (on ORT `125ea21` + transpose -> missing_ops -> silu_fusion; it also contains the register-tile
experiment above, so apply it *instead of* `ort_conv2d_regtile_experiment.patch`) adds a Winograd path to `Conv` for NHWC fp32,
group 1, 3x3, stride 1, dilation 1, Cin and Cout multiples of 4 and min(Cin, Cout) >= 64. Four programs run per conv: weight
transform (cached in the kernel when the weights are a prepacked initializer), input transform (4x4 tiles of 2x2 outputs),
16 batched GEMMs (`[16][tiles][Cin] x [16][Cin][Cout]`, TM=4 tiles x 2 vec4 channels per thread, workgroup 16x4, scalar accumulators),
and the output transform with bias and the fused activation. Environment knobs: `ORT_WEBGPU_CONV_WINOGRAD=0` disables it,
`ORT_WEBGPU_WINO_{TM,NV,WX,WY,MINC,MINHW,MAXHW}` tune it.

Correct on the phone: logits/outputs match the CPU EP to <= 5e-5 relative on ResNet-50, YOLO11n/26n, SAM-L0 encoder and the
single-conv models (with bias, ReLU and SiLU epilogues); the 235-op sweep still passes.

| model | ORT default (ms) | Winograd (ms) |
|---|---|---|
| ResNet-50 | 66.7 / 67.5 | **61.4 / 60.9** |
| SAM-L0 encoder | 441 / 442 | **383 / 382** |
| RT-DETR pre | 396 / 399 | **344 / 347** |
| YOLO11n | 71.8 / 71.3 | 73.0 / 72.0 (neutral) |
| YOLO26n | 64.8 / 64.4 | 64.3 / 65.3 (neutral) |

Two runs each, phone median. The min-channel rule matters: with no threshold YOLO11n/26n get 10-15% *slower* (their 3x3 convs have
16-64 channels at high resolution, where the 4x-larger transformed tensors cost more than the multiplications saved). Restricting to
the small feature maps (<= 28) still keeps most of the ResNet gain; the 56x56 layers add little. GEMM tile sweep: TM=4, NV=1 or 2
are equal (61.0-61.3 ms), TM=8 is 70+ ms. This is the first change in this investigation that makes the WebGPU EP faster on the
end-to-end ResNet, and it agrees with the standalone `conv_alt.cc` result.

## Winograd F(4,3) vs F(2,3) (standalone, `dawn_repro/conv_alt43.cc`)

Same harness as `conv_alt.cc` (NHWC fp32, batch 1, pad 1, Cin=Cout=C, best of a GEMM tile sweep per algorithm), with one generic
transform generator so F(2,3) and F(4,3) (6x6 tiles -> 4x4 outputs, 36 batched GEMMs, points 0, +-1, +-2, inf) are timed the same
way. Two runs each on the Adreno 730 (ms; "x direct" = time / best direct register-tile conv):

| layer | direct | F(2,3) | F(4,3) | F(4,3)/F(2,3) | relerr F(2,3) / F(4,3) |
|---|---|---|---|---|---|
| 64ch @ 56x56 | 1.32 / 1.40 | 0.94 / 0.79 | **0.81 / 0.67** | 0.86 / 0.85 | 6e-7 / 1e-5 |
| 128ch @ 28x28 | 1.38 / 1.34 | 1.11 / 1.00 | 1.06 / 1.03 | 0.95 / 1.03 | 4e-7 / 5e-6 |
| 256ch @ 14x14 | 1.50 / 1.47 | **0.86 / 0.97** | 1.01 / 1.12 | 1.17 / 1.15 | 1e-6 / 1e-5 |
| 512ch @ 7x7 | 1.97 / 1.76 | **0.80 / 0.78** | 1.63 / 1.72 | 2.05 / 2.22 | 8e-7 / 6e-6 |

- Multiplies per output pixel and channel pair: direct 9, F(2,3) 4, F(4,3) 2.25, but F(4,3) needs 36 GEMMs with 4x fewer rows each. It only wins where there are many
  tiles: 56x56 (196 tiles) by ~15%; 28x28 (49 tiles) is a tie; at 14x14 (16 tiles) and 7x7 (4 tiles) the GEMMs are too small to fill the GPU and
  F(4,3) is 15% / 2x slower than F(2,3). The transforms are minor (0.05-0.15 ms per stage; the F(4,3) input transform is a little dearer, the output transform cheaper).
- Accuracy: F(4,3) has ~10x the error of F(2,3) (5e-6..1e-5 vs 4e-7..1e-6 relative to the max output, fp32); fine in fp32, but it would not be safe in fp16.
- Recommended choice: F(4,3) for tile counts >= ~150 (64ch@56 and larger feature maps), F(2,3) below that, direct for < 64 channels. At 56x56 the
  end-to-end gain is bounded: ResNet-50 has only a few such layers, and in the ORT experiment the 56x56 layers added little over the smaller ones, so the expected
  network gain from F(4,3) is small (well under 1 ms of 61).

## Winograd with f16 intermediates (standalone microbenchmark, `dawn_repro/conv_alt_f16.cc`)

Same 3x3 Winograd F(2,3) pipeline as above, but the transformed tensors V (input) and U (weights) are stored packed f16 (8 halves per
`vec4<u32>`), the 16 batched GEMMs unpack to f32 and accumulate in f32, and M is either f32 or packed f16. Graph input and output stay
f32, so it would drop into ORT's Conv without an f16 graph (the input transform rounds, the output transform reads f16/f32). U is
computed on the host and not timed (ORT caches it). Best of a few tile configs per pipeline, one submission of 10 repetitions,
Cin = Cout = C, 0.231 GFLOP each. relerr = max|err| / max|ref| over 300 sampled outputs vs a double-precision direct convolution.
The phone was busy, so absolute times are ~1.5-2x higher than in the section above; compare only within a row.

| layer | f32 pipeline (ms) | f16 V,U + f32 M (ms, relerr) | f16 V,U,M (ms, relerr) |
|---|---|---|---|
| 64ch @ 56x56 | 1.88 (6e-7) | **0.93 (1.4e-3)** = 2.0x | **0.76 (1.8e-3)** = 2.5x |
| 128ch @ 28x28 | 1.49 (4e-7) | **0.85 (8.0e-4)** = 1.8x | **0.65 (1.3e-3)** = 2.3x |
| 256ch @ 14x14 | 1.63 (1.1e-6) | **0.94 (1.0e-3)** = 1.7x | 0.99 (1.5e-3) = 1.7x |
| 512ch @ 7x7 | 1.49 (7e-7) | 1.26 (1.0e-3) = 1.2x | 1.33 (1.2e-3) = 1.1x |

With post-ReLU-like (nonnegative) input the speedups are 1.9x/1.7x/1.6x/1.15x (M f32) and the errors 1.7e-3, 9e-4, 2e-3, 1.2e-3.

- The GEMM stage is what shrinks (half the bytes per load and the same f32 fma count): 1.5 -> 0.46 ms at 64ch@56, 1.3 -> 0.75 at 128ch@28.
  The transforms cost the same (input 0.05-0.21 ms, output 0.03-0.16 ms). Packing M as f16 helps only where the output transform and the
  GEMM store dominate (56x56 and 28x28); at 14x14 and 7x7 keeping M in f32 is as fast or faster.
- Accuracy is ~1e-3 relative to the largest output (8e-4 .. 2.2e-3), i.e. f16 rounding of V and U (2^-11 each) accumulated over K; the f32
  pipeline is at 1e-6. That is the usual f16-inference error level, but it is NOT "f32 conv results": an f16 path would need to be
  opt-in (or limited to models already run in f16), and needs an accuracy check on a real network (logits / detection boxes), not just
  random data.
- Speed potential in ORT: the 3x3 layers of ResNet-50 go from ~1.5 ms to ~0.9 ms per 0.231-GFLOP layer in the standalone pipeline; the
  earlier Winograd integration shows the standalone ratios carry over only in part (the register-tile GEMM did not), so expect a fraction of
  this. Worth trying because it needs only two changed transforms and a p16-style GEMM stage inside the existing four programs, with the
  weight transform (cached) writing packed f16; the memory footprint of V, U, M also halves.

## Stride-2 3x3 convs: polyphase Winograd microbenchmark (small gain)

`webgpu_ops/dawn_repro/conv_s2.cc` (standalone Dawn kernels, not in ORT) compares, for the 3x3 stride-2 pad-1 NHWC fp32 convs of
ResNet-50 v1.5 (Cin = Cout = C, 0.231 GFLOP each), the direct register-tile implicit GEMM with a *polyphase hybrid Winograd*.
Splitting the input into even/odd phases gives `o[y] = w1 xe[y] + w0 xo[y-1] + w2 xo[y]` per dimension: the two-tap part uses F(2,2)
(3 mults per 2 outputs), the one-tap part is direct (2 mults), 5 mults per 2 outputs, so 25 per 2x2 output tile instead of 36 (0.69x).
The input transform makes only 1.56x more data than the image (vs 4x for stride-1 F(2,3)); 25 batched GEMMs follow, then the output
transform. Outputs match a double-precision CPU reference (relative error <= 2.3e-6). Best of 7 tile configs per algorithm, two runs:

| layer | direct (ms) | polyphase (ms) | polyphase/direct | of which GEMM (ms) |
|---|---|---|---|---|
| 128ch 56->28 | 1.37 / 1.41 | 1.18 / 1.33 | 0.87 / 0.94 | 0.91 / 0.96 |
| 256ch 28->14 | 1.47 / 1.48 | 1.38 / 1.36 | 0.94 / 0.92 | 1.36 / 1.30 |
| 512ch 14->7 | 1.89 / 1.78 | 1.69 / 1.55 | 0.90 / 0.87 | 1.34 / 1.38 |
| 64ch 112->56 | 1.80 / 1.74 | 1.39 / 1.45 | 0.77 / 0.84 | 0.88 / 0.95 |

It is 6-23% faster than direct (>10% on 512ch and 64ch, borderline on the others), a gain of 0.1-0.2 ms per layer, much less than the
stride-1 Winograd gain because the multiplication saving is only 1.44x. ResNet-50 has only three such convs (128@56, 256@28,
512@14), so this is worth about 0.4-0.5 ms of ~61 ms (<1%); YOLO-style networks have more stride-2 convs but with fewer channels,
where the transforms would eat the gain. Not worth integrating into ORT unless a network has many wide stride-2 convs.

## Winograd with f16 intermediates in ORT (opt-in, `ORT_WEBGPU_WINO_F16=1`)

`ort_conv_winograd.patch` now also has an f16 mode (Cin, Cout multiples of 8): the weight and input transforms write V and U as packed
f16 (`uint32` tensors, 8 halves per vec4), the 16 batched GEMMs read them and accumulate in f32 (M stays f32), so graph tensors stay
f32. Phone medians, f32 Winograd -> f16 intermediates: ResNet-50 61.1 -> **57.9 ms**, SAM-L0 encoder 382 -> **354**, RT-DETR pre 345 -> **318**,
YOLO26n 64.3 -> 64.4 (unaffected). Cost is accuracy: ResNet-50 logits differ from the CPU by 1.9e-3 relative (top-1 unchanged), against
8e-7 for the f32 Winograd path, hence opt-in. Together with Winograd: ResNet-50 67 -> 58 ms (-14%), SAM-L0 441 -> 354 (-20%), RT-DETR pre 397 -> 318 (-20%).

Not worth integrating (standalone results above): F(4,3) (wins only at 56x56, <1 ms of ResNet-50) and polyphase stride-2 Winograd (~0.4 ms of ResNet-50).

## Whole-conv packed-f16 path (f16 activations + weights, f32 accumulate) -- microbenchmark, `dawn_repro/conv_f16io.cc`

Question: if the *graph* ran f16 (activations and weights packed 8 halves per `vec4<u32>`, f32 accumulators, bias+ReLU epilogue and packed-f16
output), how much faster would the ResNet-50 conv classes get, and is the accuracy plausible? Same design as the earlier `p16` GEMM, applied
to whole convs: 1x1 as NHWC GEMMs and a direct 3x3 (implicit GEMM, 9 taps), each swept over TM 2/4/8 x 6 workgroup shapes; the f32 baseline
is the best of the seven f32 register-tile configs of `conv_alt.cc` (no epilogue, so it is slightly favoured). Random non-negative
activations, He-uniform weights, batch 1, Dawn with ORT's toggles (`RB=off VMM=1`), best of 10 batches of 10 back-to-back dispatches.

**The phone was in a slow/noisy state for this session** (the f32 register-tile kernels ran at 70-100 GFLOPS on the 1x1 shapes, against 200+ in
the cleanest earlier runs), and the 64->256@56 case swung 0.29 -> 0.68 ms between two runs, so read the ranges of two full runs and the ratios,
not the absolute GFLOPS.

| conv | GFLOP | f32 reg-tile (ms) | f16io packed (ms) | f16io / f32 time | ORT-style `sh` GEMM (ms, gemm.cc, same session, 1x1 only) |
|---|---|---|---|---|---|
| 1x1 64->256 @56 | 0.103 | 1.06-1.08 | 0.29-0.68 | 0.27-0.63 | 1.20 |
| 1x1 256->64 @56 | 0.103 | 1.19-1.20 | 0.47 | 0.39 | 1.00 |
| 1x1 512->128 @28 | 0.103 | 1.09-1.29 | 0.68-0.80 | 0.62 | 0.88 |
| 1x1 1024->256 @14 | 0.103 | 1.45-1.50 | 0.96-1.00 | 0.67 | 1.00 |
| 1x1 2048->512 @7 | 0.103 | 1.02-1.18 | 0.63-0.86 | 0.62-0.73 | 1.20 |
| 3x3 64 @56 | 0.231 | 1.32-1.39 | 1.12-1.32 | 0.85-0.95 | -- |
| 3x3 128 @28 | 0.231 | 1.31-1.41 | 0.91-0.95 | 0.65-0.69 | -- |
| 3x3 256 @14 | 0.231 | 1.44-1.50 | 1.07-1.10 | 0.72-0.76 | -- |
| 3x3 512 @7 | 0.231 | 1.70-1.88 | 1.25-1.29 | 0.66-0.74 | -- |

(`sh` = classic shared-memory tile, TM=4 x 4 outputs/thread, workgroup 8x8, REPS-batched throughput timing, so only roughly comparable.)

- The packed-f16 kernels are **1.4-1.6x faster than the f32 register tile on the deep/narrow layers (1x1 at 28/14/7, all 3x3 at <= 28x28)**, up to
  2-3x on the wide-M 1x1 layers (56x56), and only ~1.1x on the 3x3 at 56x56 (compute-bound there, the gather dominates). Best f16io throughput
  in this state: 150-350 GFLOPS on 1x1 (best 352), 175-254 on 3x3.
- **Accuracy, single layer** (300 sampled outputs vs an f64 reference computed from the *original f32* data, so input, weight and output
  rounding all count): max relative error 4.9e-4 - 9.6e-4 (rms 4-6e-4), i.e. one f16 rounding of the output (2^-11 = 4.9e-4); the f32 kernels sit at 1e-6.
- **Accuracy, 10 stacked 3x3 layers** (He-init random weights, biases, ReLU; f16 activations *between* layers; error vs an f64 chain of the same
  weights): the error grows roughly linearly with depth, ~4e-4 per layer:

  | layer | 1 | 2 | 3 | 5 | 7 | 10 |
  |---|---|---|---|---|---|---|
  | C=64 @28, max / rms | 6.2e-4 / 5.0e-4 | 1.1e-3 / 8.3e-4 | 1.6e-3 / 1.3e-3 | 2.0e-3 / 2.0e-3 | 3.1e-3 / 2.7e-3 | **4.5e-3 / 3.8e-3** |
  | C=128 @28 | 7.7e-4 / 5.1e-4 | 1.1e-3 / 8.3e-4 | 1.4e-3 / 1.1e-3 | 2.2e-3 / 1.8e-3 | 2.9e-3 / 2.5e-3 | 4.1e-3 / 3.6e-3 |
  | C=256 @14 | 8.1e-4 / 5.2e-4 | 1.2e-3 / 8.4e-4 | 1.5e-3 / 1.2e-3 | 2.2e-3 / 1.9e-3 | 2.6e-3 / 2.6e-3 | 3.9e-3 / 3.7e-3 |

  Extrapolated linearly to ResNet-50's ~50 convs that is ~2e-2 relative rms on the deepest activations (residual adds and BatchNorm folded into
  the weights do not add much, but were not modelled). That is the usual size of an f16 inference error: top-1 is normally unchanged but the output is not
  close to bit-exact and it needs a per-model accuracy check; keep the f32 path as the default.
- **Estimate for ResNet-50 end to end** (from the per-class ratios and ~55% of the FLOPs in 1x1 convs, ~40% in 3x3): conv time x ~0.65 on the 1x1
  layers and x ~0.75-0.8 on the 3x3 layers if the standalone ratios carried over, i.e. **roughly 20-25% of the conv time, ~10-15 ms of ORT's 61 ms**
  (58 ms with the f16 Winograd intermediates). But the register-tile GEMM lesson applies: its standalone gain (1.4-1.7x) did not transfer into
  ORT's Conv2dMM at all (1x1 layers got slightly slower), and here the ORT Winograd path already takes over most 3x3 layers, so a realistic in-network
  gain is a fraction of this -- I would expect **~5-8%**, and it needs an f16 graph (or f16 activation tensors between fused convs) plus an
  accuracy gate, which is a larger change than the Winograd-internal f16 already integrated. Worth a prototype only for the 1x1 layers (56x56 -> 14x14),
  where the ratio is best.

## Dispatch counts, per-dispatch floor and fusion/elimination targets (wall-clock measured)

Scripts: `webgpu_ops/dispatch_floor.py` (chains of N trivial `Add`s), `profile_breakdown.py` + `fusion_estimate.py` (from `PROFILE=` json of
`bench.cc`). Current phone build (Winograd f32 on). The profiler's per-node durations only tell node order/type here (they sum to 10-30 ms on a
60-350 ms model); every time below is either a wall-clock median or an estimate built from a wall-clock-calibrated cost model, and is labelled so.

**Per-dispatch floor (wall clock, `chain_N.onnx`, 4096 floats, so no bandwidth cost).**

| N chained Adds | plain Run (ms) | IO-binding + sync | graph capture (ms) |
|---|---|---|---|
| 1 | 0.32 | 0.27 | 0.03 |
| 50 | 1.08 | 1.09 | 0.27 |
| 100 | 1.69 | 1.71 | 0.52 |
| 200 | 3.07 | 3.07 | 1.03 |
| 400 | 5.49 | 5.59 | 2.02 |

Slope: **~13 us per dispatch** without graph capture (ORT node overhead + Dawn command recording, the CPU side), **~5 us** with capture (the
GPU-side floor). So the "0.19 ms per node" floor seen in ResNet-50 is not per-dispatch overhead: it is the convs' own time. A network with ~130-280
dispatches pays only 1.7-3.7 ms (3-5%) of pure dispatch floor. With 3.2 MB tensors (`chb`, 800k floats) an `Add` costs ~0.25-0.28 ms each
(9.6 MB moved -> ~35-45 GB/s effective, DRAM-resident constants), 0.22 ms with graph capture, i.e. elementwise ops on big activations cost
memory traffic, not dispatch overhead. Calibration on ResNet-50 with ablated graphs (wall clock, 3 runs each): 61.1 ms full, 60.5 without the
post-Add ReLU (`r50_norelu`), **58.9 without both Add and ReLU (`r50_noaddrelu`)**: 32 nodes / 44 MB of outputs cost 2.2-2.4 ms
(~50 GB/s effective + 13 us per node). Cost model used below: `13 us + moved_bytes / 50 GB/s` per removed node (2x the output size for a copy or
unary op, 3x for a binary op).

(This corrects the earlier "residual Add+ReLU is worth <=0.5%" note: removing both is worth ~3.5-4% of ResNet-50, though fusing them into the
convolution epilogue can only recover part of it -- the residual operand is still read -- an upper bound of ~2 ms / 3.3%.)

**Dispatches and node types per run** (nodes from the profile, no-dispatch = Reshape/Unsqueeze/Squeeze/Flatten; Winograd adds 3 dispatches per
eligible conv, counted from the ONNX graph: 3x3, stride 1, group 1, min(Cin,Cout) >= 64).

| model | graph nodes | no-dispatch | Winograd convs | est. dispatches | 13 us floor (ms) |
|---|---|---|---|---|---|
| resnet50 | 91 | 1 | 13 | 129 | 1.7 |
| yolo11n | 180 | 8 | 14 | 214 | 2.8 |
| yolo26n | 210 | 12 | 6 | 216 | 2.8 |
| rtdetr_pre | 247 | 37 | 25 | 282 | 3.7 |
| rtdetr_mid0 / mid1 | 64 | 19 | 0 | 45 | 0.6 |
| rtdetr_post | 30 | 8 | 0 | 22 | 0.3 |
| sam_l0_enc | 204 | 8 | 6 | 214 | 2.8 |

Op mix (node counts): ResNet-50 Conv 53, Add 16, Relu 16 (the residual Add and post-Add ReLU are separate dispatches; Conv+ReLU inside the blocks is
fused), Transpose 2. YOLO11n Conv 88, Concat 23, Add 16, Transpose 16, Split 11, MaxPool 3, Resize 2. YOLO26n Conv 102, Concat 23, Transpose 22, Add 21,
Split 11, MatMul 4. RT-DETR pre Conv 69, Reshape 36, Add 30, Transpose 24 (57 MB of outputs), Gemm 22, QuickGelu 12, Concat 5, plus **3 CPU-EP nodes**
(`Unsqueeze`, 2x `Tile`) and 2 `MemcpyFromHost` + 1 `MemcpyToHost` after `TopK`: a mid-graph sync (the GPU queue is drained, the CPU runs three
tiny nodes, the result is uploaded again), roughly 2-4 ms of bubble/copies on a 345 ms model by node durations. SAM-L0 encoder Conv 73, **Gelu 30 (all
directly after a Conv, not fused)**, Add 25, Slice 20, Transpose 13, Pad 6, MatMul 8. The RT-DETR mid/post stages are Gemm/Reshape/Relu graphs of 22-45 dispatches:
nothing worth fusing (<0.4 ms).

**Upper-bound savings from removing dispatches/traffic** (cost model above; "n" = nodes that would disappear):

| model (wall-clock) | target | n | est. saving |
|---|---|---|---|
| ResNet-50 (61 ms) | 1. Conv+residual Add epilogue | 16 | 1.1 ms |
| | 2. Post-Add ReLU into the same epilogue | 16 | 1.1 ms |
| | 3. the two layout Transposes | 2 | 0.05 ms |
| YOLO11n (72 ms) | 1. Concat elimination (producers write channel slices) | 23 | 1.9 ms |
| | 2. Transpose elimination (Conv>Transpose>Reshape attention layouts, NCHW/NHWC) | 16 | 1.2 ms |
| | 3. Split as strided views | 11 | 0.8 ms |
| YOLO26n (64 ms) | 1. Concat elimination | 23 | 1.7 ms |
| | 2. Transpose elimination | 22 | 1.2 ms |
| | 3. Split as strided views | 11 | 0.6 ms |
| RT-DETR pre (345 ms) | 1. Transpose elimination (24, 57 MB) | 24 | 2.6 ms |
| | 2. Conv+residual Add epilogue | 20 | 1.9 ms |
| | 3. Concat elimination (5, 29 MB) | 5 | 1.2 ms |
| SAM-L0 encoder (354 ms) | 1. **Gelu into the 1x1 Conv epilogue** (Conv->Gelu x30, large MLP tensors) | 30 | **10.0 ms** |
| | 2. Conv+residual Add epilogue | 20 | 2.0 ms |
| | 3. Transpose elimination / Slice | 13 / 20 | 1.2 / 0.9 ms |

These are upper bounds (a fused epilogue still reads its second operand, and channel-slice writes are strided) -- expect 50-70% of them in
practice. Ranked by wall-clock value for one implementation effort: **Conv+Gelu epilogue** (SAM, ~2.8% of the model, ~1.5-2% after the epilogue
cost), **Conv+Add(+ReLU) residual epilogue** (all conv models: 1.1-2.2 ms, 1.5-3.5% on ResNet-50/RT-DETR), then **Concat/Split/Transpose
elimination** in the YOLO family (~4 ms of 64-72 ms, 5-6%, but a much larger change: it needs a layout pass, not a shader epilogue). Dispatch-count
reduction alone (graph capture, persistent kernels) tops out at the 13 us floor: 1.7-3.7 ms per run, and graph capture already recovers 8 of the
13 us. RT-DETR's TopK/Tile CPU round trip is a further ~2-4 ms worth removing with a WebGPU `Tile`/`Unsqueeze` (small, self-contained).

## GPU clock after idle: real, large, and not fixable from user space (`dawn_repro/clock.cc`, `bench.cc` `SLEEP_MS`)

Question: does the Adreno 730 run at a lower effective clock in network-like workloads than in the back-to-back microbenchmarks? Yes -- not
because of dispatch structure, but because of idle time. Workload: the register-tile fp32 GEMM 784x512x128 (`sc` TM=8 NV=2, 32x4), 0.103 GFLOP.

| pattern (Dawn/Vulkan, warm) | GFLOPS |
|---|---|
| one batched dispatch (z=32) | 126-183 (noisy: depends on the clock state it starts in) |
| 100 dependent dispatches, one submission | 193-197 |
| 100 GEMMs interleaved with a cheap elementwise dispatch each | 192-197 (elementwise alone: 1.7 ms of 53) |
| submit N GEMMs, wait, repeat (no idle): N=1 / 2 / 4 / 8 / 32 / 100 | 104-118 / 160 / 157 / 165-181 / 183-192 / 198-202 |
| same, but submit everything ahead and wait once | 184-192 (N=1), 203-214 (N>=2) |

So dispatch granularity costs nothing (interleaving and dependent chains are free) and the only structural loss is synchronising on every
submission (~0.35-0.45 ms CPU round trip per wait; ORT syncs once per inference so it does not pay this). The clock effect is separate:

- After >= 100 ms with the GPU idle, 100 GEMMs run at **98 GFLOPS instead of 193** (105 ms instead of 53 ms). After 20 ms idle: 166-169; after 5 ms: no loss.
- The slow state does not clear quickly under load: 20-GEMM submissions after 1 s idle run at 1.3 ms/GEMM (vs 0.50 at full clock) for ~3 s of
  continuous work and are still improving at 8 s (0.64 ms/GEMM at t = 8 s). With an inference-like duty cycle (20 GEMMs = ~10 ms of work, then a 20 ms sleep) it
  **never ramps**: 1.5 ms/GEMM, i.e. 3x slower than the microbenchmark, for the whole 6 s (2 runs).
- Same in ORT (`bench.cc` now honours `SLEEP_MS`, an idle gap after each timed inference), ResNet-50 median with the Winograd path (ORT default in brackets):

  | idle between inferences | 0 ms | 5 ms | 20 ms | 50 ms | 200 ms |
  |---|---|---|---|---|---|
  | Winograd | 61.4 | 61.4 | 73.8 | 74.4 | **125.1** |
  | ORT default conv | 67.5 | 67.5 | 75.3 | 73.5 | **148.2** |

  A model called once per 20-200 ms therefore runs 20-100% slower than every benchmark in this document (all of which run inferences back to back);
  Winograd keeps its advantage but the absolute gain shrinks at low duty cycles (e.g. 148 -> 125 ms at 200 ms idle).
- Keep-alive tricks do not help. A second device on another thread issuing a continuous tiny-dispatch stream, or one tiny dispatch every 5 ms or 1 ms, leaves the
  inference-like pattern at 1.5 ms/GEMM (8 runs, all within noise of the no-keep-alive case). The governor evidently needs real shader load, not submissions;
  a genuinely saturating background kernel would ramp the clock only by time-slicing away the throughput it is meant to protect. Reading or pinning the clock needs
  root (`/sys/class/kgsl`), which is out of scope here.

Consequences: (1) any latency claim for a once-per-frame GPU model should be measured with the real frame gap, (2) batching or pipelining consecutive
inferences so the GPU never idles > ~10 ms is the only lever found (throughput-style use), (3) the microbenchmark-to-network gap seen earlier (e.g. the register-tile
conv) is not explained by this effect, since back-to-back ORT runs are at full clock (61 ms Winograd run has no idle).

## Conv + Gelu epilogue fusion (SAM-L0 encoder -3%)

`webgpu_ops/ort_conv_gelu_fusion.patch` (applies after `ort_conv_silu_fusion.patch`) lets ConvActivationFusion fuse an opset-20 `Gelu` (exact erf
or `approximate="tanh"`, passed as an activation parameter) into the WebGPU Conv epilogue, in the MatMul path, Conv2dMM, GroupedConv and the Winograd
output stage. The SAM-L0 encoder has 30 Conv -> Gelu(tanh) pairs (1x1, 3x3 and depthwise): all 30 Gelu nodes disappear, the outputs still match the
CPU EP to 5e-5, and the encoder goes 382 -> 372 ms (Winograd f32 on). `gen_new_op_tests.py` gained Conv+Gelu (both variants, 5 conv kinds) and Winograd
(plain, bias+ReLU, tanh-Gelu, no pad, asymmetric pad) cases; the sweep is 299 OK, 0 wrong. `ort_conv_winograd.patch` now contains only the four
`nn/conv*` files (the earlier version wrongly repeated the `fuse_utils` hunks from the SiLU patch).

## Why Adreno OpenCL beats WebGPU/Vulkan: the same work in both (`dawn_repro/cl_vs_wg.cc`)

Adreno 730, OpenCL 3.0 driver (compiler E031.38.11.11), 4 compute units, extensions incl. `cl_khr_fp16`, `cl_khr_subgroups`, `cl_qcom_perf_hint`,
`cl_qcom_ml_ops`, `cl_qcom_dot_product8`, `cl_qcom_subgroup_shuffle`, `cl_qcom_reqd_sub_group_size`, `cl_qcom_recordable_queues`,
`cl_qcom_accelerated_image_ops`. Same kernels (register-tile GEMM, scalar accumulators, TM=8 NV=2), same inputs, same session, OpenCL runs
after a 3 s warm-up, WebGPU runs via `gemm_w` (gemm.cc with a 4 s in-process warm-up). Phone noise is +-20%, so read ratios; best of several sweeps.

**Traps first.** (1) OpenCL `fma()` is emulated: ~1000x slower than `mad()`/`a*b+c` (a 2^18-thread fma loop never finished in 20 min). (2) 64 float
accumulator chains in one thread spill (141 vs 866 GFLOPS at 32 chains). (3) `-cl-fast-relaxed-math` / `-cl-mad-enable` change nothing.

| measurement | OpenCL | WebGPU (Dawn/Vulkan) |
|---|---|---|
| FMA peak f32 (32 chains, mad) | 866 GFLOPS | 988 (64 chains, fma) |
| FMA peak f16 scalar / half2 / half4 | **1001 / 1624 / 1412** | 307-313 (scalar, vec2) |
| load bandwidth, buffer float4 | 97 GB/s (6 G loads/s) | 164 GB/s (10 G/s) |
| load, image RGBA32F | 145 GB/s (9 G texels/s) | 240 GB/s (15 G/s) |
| load, image RGBA16F (`read_imagef` or `read_imageh`) | 138-204 GB/s, **17-25 G texels/s** | not measured |
| buffer half4 (convert) | 76-80 GB/s (9-10 G loads/s); `vload_half4` 19 GB/s | -- |

| GEMM (GFLOPS) | 784x512x128 | 3136x256x64 | 196x256x2304 |
|---|---|---|---|
| OpenCL f32, buffers | 197-201 | 196-199 | 160-164 |
| OpenCL f32, B as image | **474-482** | **463-487** | **382-399** |
| OpenCL f16 storage, buffers, f32 acc | 232 | 214-220 | 219-221 |
| OpenCL f16 images, f32 acc | 413-446 | 381-385 | 342-360 |
| OpenCL f16 images, f16 acc (error 5e-3..3.5e-2) | 583-621 | 537-544 | 512-572 |
| WebGPU `sh` (ORT design) | 169 | 158 | 153 |
| WebGPU `sc` f32 buffers | 219-264 | 230-268 | 161-247 |
| WebGPU `p16` packed f16 | 192-221 | 164-207 | 159-417 (noisy) |
| WebGPU `tt` (A and B in textures) | 327-383 | 201-325 | 163-259 |

| direct 3x3 conv (GFLOPS) | 64ch @56 | 256ch @14 |
|---|---|---|
| OpenCL f32 buffers | 109 | 80 |
| OpenCL f32 images (X and weights) | 228 | 157-163 |
| OpenCL f16 images, f32 acc | 306 | 176 |
| OpenCL f16 images, f16 acc (err 6e-3..8e-3) | 353 | 217 |
| WebGPU `conv_alt` direct (buffers) | 169-179 | 156-158 |

**Ranked explanation of the OpenCL advantage**
1. **The texture/image path (biggest).** OpenCL's own *buffer* kernels are not better than WebGPU's (97 vs 164 GB/s, conv 109 vs 175 GFLOPS): reading the B
   operand (and the conv input) through `image2d` roughly doubles OpenCL's GEMM (200 -> 480 GFLOPS) and its conv. A WebGPU kernel that uses textures
   (`tt`) recovers most of this (327-383 vs 480), so this part is obtainable in WebGPU, but ORT's Conv/MatMul use storage buffers only.
2. **Half storage in images.** RGBA16F texels double the texel rate (17-25 G/s vs 9 G/s for RGBA32F) and halve the bytes: 380-450 (f32 acc) vs 480 f32 images; with
   f16 accumulation 510-620. The f32-accumulate path keeps the accuracy of the f32 kernels (error ~1e-6 from the half-rounded inputs); f16 accumulation does not (5e-3 to 3.5e-2).
3. **Half ALU rate.** OpenCL `half` mad reaches 1000 (scalar) to 1624 GFLOPS (half2), 1.2-1.9x the f32 rate, while Vulkan/WGSL f16 arithmetic runs at 310
   (0.3x of f32, reproducible with f16 scalar and vec2). This is a driver/compiler difference in how each API lowers 16-bit math, and it is the one piece WebGPU cannot reach today.
4. **Precision of the comparison.** The earlier "OpenCL 1.5x faster" network comparison was fp16 OpenCL vs fp32 WebGPU; items 1-3 show that most of the gap is
   explained by images and half, not by a slower WebGPU runtime (dispatch overhead is ~13 us, ~3-5% of a run).
5. **Compiler flags: nothing.** `-cl-fast-relaxed-math`, `-cl-mad-enable` give identical times; only avoiding `fma()` matters.

**GPU clock: OpenCL can pin it without root.** `cl_qcom_perf_hint` (`clSetPerfHintQCOM(context, CL_PERF_HINT_HIGH_QCOM)`, exported by libOpenCL.so) removes the
post-idle slowdown in an OpenCL process: after >= 200 ms idle the default (NORMAL) hint runs a 32-chain mad kernel at 240-260 GFLOPS for the whole next 600 ms (and
for 3 s), HIGH runs 440-480 in the first 100 ms window and 515-530 afterwards, for every idle time from 0 to 3 s (back-to-back it is 510-560 either way). The hint
from a separate idle OpenCL process does **not** speed up a concurrently started WebGPU process (cold `gemm` 120-145 GFLOPS with and without it), so it helps only
OpenCL work in the same context. Cold starts matter: the WebGPU `gemm` tool without an in-process warm-up measures 120-170 GFLOPS where the warmed-up numbers above are 220-270,
which is why all WebGPU figures in this section come from a build with a 4 s warm-up.

**Practical consequences.** For an OpenCL-free stack: (a) use textures for the B operand / conv input in ORT's WebGPU EP (the `tt` design is 1.3-1.5x the
buffer design here); (b) keep f16 storage with f32 accumulation (packed loads) rather than f16 arithmetic; (c) the remaining gap (OpenCL image kernels
480 vs WebGPU 330-380 at f32, and all of half arithmetic) needs either OpenCL itself (tinygrad's OpenCL backend, TVM OpenCL) or a driver that lowers WGSL `f16` well.

## Memory layouts for conv / GEMM on the Adreno 730 (`dawn_repro/layouts.cc`)

One register-tile implicit-GEMM generator (scalar accumulators, TMxNV vec4 outputs per thread, no workgroup memory) run with different
layouts: activations NHWC (ORT's), NC4HW4 `[C/4][HW]`, 4x4-pixel tiles (`z4`, thread order follows the tiles), RGBA32F/RGBA16F textures
(`texa`: x = channel block, y = pixel; `texb`: x = ix*C4+c4, y = iy), packed f16 buffers (`nhwc16`, `nc416`); weights HWIO `[K][N4]` (ORT's),
output-blocked `[N4][K]` (`ok4`), vec4-over-k with `dot()` (`nk4`), RGBA32F/RGBA16F textures x = n4, y = k (`texw`, `texw16`), packed f16 over n
(`w16`); thread maps `colx` (x = channel group) and `rowx` (x = pixels, coalesced). Math is always f32 (f16 layouts unpack to f32).
Batch 1, stride 1, warm GPU (>= 5 s burn), every candidate checked against a double CPU reference (rel. error <= 3e-6, computed from the
f16-rounded operands for f16 layouts). The table is the **interleaved head-to-head** (the 12 best layouts of a layer timed in alternation,
6 rounds, min per kernel): a sequential sweep is misleading here because the GPU clock drifts during a long run and other jobs share the phone (the same
nhwc/hwio kernel read 185 GFLOPS early in one sweep and 108 late in another). Raw output: `dawn_repro/layouts_results.txt`.

Min time over 6 interleaved rounds (ms); "ORT-like" = NHWC f32 buffer activations + HWIO f32 buffer weights, best tile config:

| layer | ORT-like | nhwc + weights in f32 texture | nhwc + weights in f16 texture | best layout found |
|---|---|---|---|---|
| 1x1 64->256 @56 | 0.954 | 0.718 (1.33x) | 0.621 (1.54x) | **0.439** (2.17x): f16 texture act + f16 texture weights |
| 1x1 512->128 @28 | 0.604 | 0.384 (1.57x, nc4 act) | 0.397 (1.52x) | **0.342** (1.77x): f16 texture act + f16 texture weights |
| 1x1 1024->256 @14 | 1.348 | — | 0.849 (1.59x) | **0.648** (2.08x): f16 texture act + f16 texture weights |
| 3x3 64 @56 | 1.240 | 0.784 (1.58x) | 0.741 (1.67x) | **0.685** (1.81x): packed-f16 nc4 act + f16 texture weights, rowx |
| 3x3 128 @28 | 2.498 | 1.445 (1.73x) | 1.273 (1.96x) | **1.165** (2.14x): packed-f16 nhwc act + f16 texture weights |
| 3x3 256 @14 | 1.558 | 1.200 (1.30x) | 0.785 (1.98x) | **0.706** (2.21x): f16 texture act + f16 texture weights |

Findings:

1. **Weights through a texture are the one layout change that matters**: 1.3-1.7x with f32 RGBA32F, 1.5-2.0x with RGBA16F (half the bytes; the texture
   unit converts to f32 on load, so no f16 arithmetic and no accuracy change beyond rounding the weights to f16). The weight stream is the hot load in a
   register-tile conv (every thread re-reads the full K x N slab) and the texture path has the higher read throughput (240 vs 163 GB/s buffer).
2. **Activation layout is second order**: NHWC vs NC4HW4 vs 4x4-tiled are within about 10% of each other with buffer weights (z4 is +8% on 3x3 @56 and
   -10..-50% on smaller maps, NC4HW4 is not consistently better, `texb` is worse than `texa`). Activations in a texture (`texa`, x = channel block, y = pixel)
   gain 5-25% over the buffer, and RGBA16F activations stack with f16 weights for the best rows above.
3. **Packed-f16 activation buffers** help only together with texture weights (nhwc16/texw16 vs nhwc/texw16: -10%..+14%); with buffer weights they are noise.
4. **Weight buffer variants lose**: output-blocked `[N4][K]` (`ok4`) is 0.95-1.2x slower than HWIO, vec4-over-k with `dot()` (`nk4`) is 1.5-2.6x slower
   (and 10x+ on 3x3 because of the taps addressing). Packed f16 weights in a buffer (`w16`) are between the f32 buffer and the f16 texture.
5. **Thread map**: `colx` (x = channel group) wins except for 3x3 @56 where `rowx` (pixels along x) with TM4 NV4 is best; rowx with NHWC buffers reads strided
   and is otherwise 10-40% worse.
6. **Channel padding does not help** (1x1 @28, tiny problem so ~40-70 GFLOPS): 100 -> 104/112/128 channels changes padded GFLOPS by +0..+22% but loses 5-25% of
   *useful* GFLOPS; padding beyond the vec4 multiple (x4) is pure cost. 60 -> 64 (x8..x32) is a wash.
7. **Layout conversion is not free**: NHWC -> NC4HW4 runs at 16-34 GB/s (0.02-0.4 ms for ResNet activations), NC4HW4 -> NHWC is a scatter at 3-17 GB/s
   (0.04-2.4 ms, e.g. 2.4 ms for 256 ch @56). A per-layer layout switch costs as much as the conv it would speed up, and the activation layouts are within ~10% of each
   other, so the activation layout should stay NHWC end to end (as ORT has it).

Recommendation, ranked: (1) keep prepacked conv/GEMM **weights in an RGBA16F (or RGBA32F) 2D texture** `[K][N/4]` and read them with `textureLoad`; (2) keep NHWC
activations (optionally f16 RGBA textures between layers if the graph is f16 anyway); (3) `colx` thread map, `rowx` only for wide 3x3 maps; (4) do not pad channels past a
multiple of 4; (5) do not switch activation layouts between layers. Estimated ORT end-to-end effect: **unverified**. The microbenchmark baseline is the same
register-tile kernel with buffer weights, so the 1.3-2.0x per layer is a pure weight-load effect, but ORT's Conv2dMM stages weights through workgroup memory, ORT's Program
API has no texture inputs, and earlier register-tile microbenchmark gains did not carry over into the network. If even half of the per-layer gain on the weight-heavy layers
carried over, ResNet-50 (about 80% conv time) would drop roughly 10-20%; treat 5-10% as the realistic expectation until an ORT prototype (texture weights in the Winograd GEMM
stage, which is a plain register-tile GEMM, is the cheapest place to try it) confirms it.

## Winograd weights in a texture (`ORT_WEBGPU_WINO_TEX=16`; ResNet-50 -4%, SAM-L0 -8%, RT-DETR pre -10%)

Follows the layout study: the weight stream is the hot load, and texture reads are faster than storage-buffer reads. ORT's Program API had
no texture bindings, so `webgpu_ops/ort_webgpu_extra_texture.patch` adds one optional extra 2D texture per program (`ProgramBase::SetExtraTexture`:
write-only storage texture or sampled unfilterable-float texture, bound after the buffers and the uniform; `ComputeContextBase::Device()`).
`ort_conv_winograd.patch` uses it: the weight-transform program `textureStore`s U (16*Cin rows x Cout/4 texels, RGBA16F or RGBA32F) and the
GEMM stage `textureLoad`s it (cached across runs for prepacked weights; requires Cin <= 512). Phone medians (Gelu fusion on), buffer weights -> RGBA16F texture:
ResNet-50 60.4 -> **58.2 ms**, SAM-L0 encoder 371 -> **343**, RT-DETR pre 345 -> **309**; TM=4 NV=1 WG 16x8 is marginally better for ResNet (57.3).
RGBA32F texture is not faster (63.7 ms). Cost: weights are rounded to f16 (ResNet-50 logits 1.3e-3 relative vs 8e-7; top-1 unchanged), so it is opt-in,
like the f16-intermediates mode (the two are not combined yet). Far below the 1.8-2.2x per layer of the standalone layout study, consistent with
earlier standalone gains only partly transferring. Applying the patches: transpose -> missing_ops -> silu_fusion -> gelu_fusion -> extra_texture -> conv_winograd.

### Texture weights + f16 intermediates together: no extra gain

`ORT_WEBGPU_WINO_F16=1 ORT_WEBGPU_WINO_TEX=16` (packed-f16 V, RGBA16F texture U) works (ResNet-50 logits 1.8e-3 from the CPU, top-1 unchanged)
but is not faster than either alone. Phone medians (ms): ResNet-50 60.5 (neither) / 57.8 (texture) / 57.7 (f16 V) / 58.2 (both);
SAM-L0 encoder 343 (texture) / 345 (both); RT-DETR pre 312 (texture) / 316 (both). Both modes remove the same GEMM-stage load traffic, and the four
Winograd stages are now limited by the transforms and dispatch overheads rather than by the GEMM loads, so stacking them does not help.

## int8 on the Adreno 730 through WebGPU (`dawn_repro/int8.cc`)

**The driver has no int8 dot product.** `vk_int8_query.cc`: Vulkan 1.1.128, `VK_KHR_8bit_storage` and `VK_KHR_shader_float16_int8` (shaderInt8 = 1) are present,
but not `VK_KHR_shader_integer_dot_product` (shaderIntegerDotProduct = 0). Dawn exposes the WGSL feature `packed_4x8_integer_dot_product` anyway and Tint
polyfills `dot4I8Packed`: `dot4I8Packed`, `dot(unpack4xI8, unpack4xI8)` and a hand-written extractBits multiply-add all run at the same 30.1 G dot4/s = 241 GOPS int8
(peak kernel, 16 independent chains), i.e. ~120 G int8 MAC/s against 493 G fp32 FMA/s (750-775 GFLOPS measured in that session; 986 earlier). So int8 arithmetic in
WGSL is ~4x *slower* than fp32 per MAC on this GPU; an int8 GEMM with `dot4I8Packed` runs at 130-160 GOPS, no better than the f32 register-tile kernel.

**int8 storage with f32 arithmetic is what works.** Variant `f8`: A and B packed 4 x int8 per u32 (16 k per vec4 load for A, 4 columns x 4 k per vec4 load for B), `unpack4xI8` +
convert once per operand, `dot()` in f32 (exact for int8 data while sums stay < 2^24), per-channel `clamp(round(acc * scale[n]), -128, 127)` epilogue packed 4 outputs per u32 (a QLinearConv-style
requantize). Loads are 4x smaller than f32 and 2x smaller than packed f16. Correct against an exact CPU integer reference (0 wrong of 600 samples per case, all shapes). Same
session, best tile per variant (the phone was shared and slow: f32 `sc` read 92-146 GFLOPS instead of the usual 200-280, so read ratios, not absolute values; `RB=off VMM=1`, 5 s warm-up for the int8 binary):

| GEMM (M N K) | f32 `sc` | p16 (f16 storage) | int8 `dot` (polyfill) | int8 `f8` (TM=4 NV=1) | f8 / f32 | f8 / p16 |
|---|---|---|---|---|---|---|
| 784 512 128 | 146 | 172-195 | 141-159 | **358** (wg 16x8) | 2.5x | 2.0x |
| 3136 256 64 | 131 | 166 | 136 | **329-346** | 2.6x | 2.0x |
| 784 128 1152 | 116 | 180 | 142 | **308** | 2.7x | 1.7x |
| 196 256 2304 | 95 | 154-160 | 136 | **324** (wg 32x4) | 3.4x | 2.0x |

Caveats: `f8` is very sensitive to the tile: the 64-accumulator tile (TM=8 NV=2) spills and runs at 5-6 GOPS, TM=4 NV=2 gives 120-138, TM=4 NV=1 with a 16x8 or 32x4 workgroup gives 300-360 (and
some workgroup shapes swing by 2x between shapes), so a real kernel needs a per-shape tile table. All GOPS are 2*M*N*K/time and count int8 MACs like f32 FMAs.

**ResNet-50 estimate (not measured in ORT).** Conv MACs: 1x1 2.12 G (36 layers, 51%), 3x3 1.85 G (16 layers, 45%), 7x7 stem 0.12 G. 1x1 convs are plain GEMMs and would take the `f8` kernel
(2.0-2.7x over f32, 17.5 M activation elements = 17.5 MB int8 instead of 70 MB f32). The 3x3 layers already use f32 Winograd, which is ~1.6x faster than a direct kernel, while a direct int8 3x3 conv
pays the gather penalty (~30% below a plain GEMM), so it would be roughly on par with Winograd f32: leave them in f32 and fuse the (de)quantize into the neighbouring epilogues (the 1x1 requantize
epilogue and the Winograd output transform) so no standalone Q/DQ dispatches appear; residual Adds would need an int8 or dequantize-add kernel. With the ~25 ms that the 1x1 layers take of the current ~58 ms:
full microbenchmark gain -> ~11-12 ms (58 -> ~44 ms), half of it transferring (as with earlier GEMM results) -> ~17 ms (58 -> ~50 ms). That is -12% to -24% on ResNet-50, for
mixed-precision PTQ accuracy (int8 activations on the 1x1 layers) that must be checked per model. **Integration cost is high:** the ORT WebGPU EP has no QLinearConv / ConvInteger kernel, so it needs
a new kernel, QDQ-fusion registration for the EP, per-shape tile tuning and a calibration flow; it is worth doing only if int8 accuracy is acceptable for the target models. The cheaper
intermediate step is weights-only int8 (dequantize in the kernel), which cuts weight traffic 4x but leaves activations f32.

## fp16 graphs on ORT's WebGPU EP (stock fp16 kernels, phone, 2026-10-01)

Scripts: `webgpu_ops/fp16/` (`convert_fp16.py`: `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`; `conv_only_fp16.py`,
`conv_kind_fp16.py`: fp16 only around selected Convs; `run_fp16.sh`). Same ORT build and GPU as the fp32 numbers (Winograd default on, which only
applies to fp32 tensors, so fp16 graphs take ORT's stock fp16 paths), >= 5 s warm-up, medians. Graph I/O stays fp32.

| model | fp32 (Winograd etc.) | fp16 graph | error vs fp32 (same build) |
|---|---|---|---|
| ResNet-50 | 61.2 ms | **55.3 ms** (-10%) | logits rms 3.6e-3, max 8e-3; top-1 911 = fp32 = CPU |
| YOLO11n | 72.0 ms | **59.7 ms** (-17%) | rms 1.9e-3, max 2.3e-2 of the output range |
| SAM-L0 encoder | 370 ms | 340 ms | **all NaN** |
| RT-DETR pre | 343 ms | 318 ms | **all NaN** (outputs 0-4), outputs 5/6 wrong |

- **No CPU fallbacks** in the fp16 graphs of ResNet-50 (279 nodes), YOLO11n (558) and SAM (525) (profiler); RT-DETR pre keeps the same 9 CPU nodes as fp32
  (`Unsqueeze` x3, `Tile` x6). So the stock fp16 Conv/MatMul path is complete; the fp16 win is 10-17% on the two models that survive, compared with
  Winograd + texture weights at 58 ms on ResNet-50 (the paths do not combine: Winograd is fp32-only).
- **SAM-L0 and RT-DETR are not usable in fp16 with the stock kernels, and the timings above are void.** Bisecting SAM (33 intermediate outputs): the first
  NaNs appear in the third stage's MLP Gelu after the linear-attention `MatMul`/`Add` (values up to ~500); `LayerNormalization`/`Softmax`/`Div`/`Erf`/`Gelu`/`MatMul`/`Gemm`
  kept in fp32 (op block lists) did not remove them. Converting **only Convs** to fp16 (fp32 elsewhere, Casts around each Conv) isolates it to the **group=1 1x1 convs**:
  fp16 on just those 40 convs gives rms error **0.62** (and 351 ms, i.e. fast because wrong), fp16 on the 11 group-1 3x3 convs 1.2e-2, on the 18 grouped convs 1.7e-2.
  ORT's 1x1 fp16 path is the MatMul-style kernel with f16 accumulators, which loses the sum over K = 256-3072 once activations reach tens to hundreds.
  Wrapping fp16 around Conv only (no other fp16 ops) is also no faster: SAM 382 ms and RT-DETR 342 ms with Casts, vs 370 / 343 fp32.
- **Implication for this investigation:** the lower-precision route that works here is f16 *storage* with f32 *accumulation* (the `p16` GEMM, the f16 Winograd
  intermediates and the RGBA16F texture weights above), not ORT's f16-accumulating kernels. A fp16 activation graph would need ORT's Conv/MatMul shaders changed
  to accumulate in f32 before it is safe on attention-heavy models.
- **Weight-only fp16 (fp16 initializers + runtime `Cast` to fp32) could not be measured:** keeping the initializers as graph inputs (so ORT cannot fold the Cast back
  to fp32) makes the phone `bench` crash on load (the same model runs in host onnxruntime), and with the Casts folded it is just the fp32 model. A runtime Cast would
  also add a write+read of the fp32 weights, the opposite of a bandwidth saving; the useful version of the idea is the texture-weights result (RGBA16F weights read
  directly by the GEMM).
- Recommendation: keep fp32 as the default. fp16 graphs are a 10-17% win for ResNet/YOLO-style convnets but need a per-model accuracy check and are broken on
  SAM/RT-DETR until the f16 kernels accumulate in f32.

## f32 accumulation in ORT's fp16 MatMul/Conv2dMM: does not fix the SAM / RT-DETR NaNs (opt-in, `ORT_WEBGPU_F16_ACC32=1`)

The fp16-graph run above suspected that ORT's fp16 1x1 conv accumulates in f16. `webgpu_ops/ort_f16_f32_accumulate.patch` (apply before
`ort_conv_winograd.patch`, which calls the new `f32_accumulate` argument) makes the vec4 packed MatMul/Conv2dMM accumulators f32 for fp16 inputs
(non-transposed, alpha 1, no split-K). Result on the fp16 graphs (phone medians; error vs the CPU EP):

| model | f16 accumulate | f32 accumulate |
|---|---|---|
| ResNet-50 | 58.0 ms, 9.8e-3 | 60.3 ms, 8.3e-3 |
| YOLO11n | 60.7 ms, 1.4e-2 | 65.0 ms, 1.1e-2 |
| SAM-L0 encoder | 355 ms, NaN | 405 ms, NaN |
| RT-DETR pre | 322 ms, NaN | 368 ms, NaN |

So accumulation precision is not the cause of the NaNs; f32 accumulation costs 4-14% and buys little accuracy, hence opt-in. The NaN source in SAM/RT-DETR is still
open (first NaN is at the third-stage MLP Gelu, where values reach ~500; the tanh-Gelu's x^3 overflows f16 above |x| ~ 40, and other f16 intermediates in the
attention blocks may overflow too -- not isolated).

## Conv + residual Add (+ReLU) fusion, and the Winograd weight cache that was not being used

`webgpu_ops/ort_conv_add_fusion.patch` (last in the stack: transpose -> missing_ops -> silu_fusion -> gelu_fusion -> extra_texture -> f16_f32_accumulate
-> conv_winograd -> conv_add_fusion):
- **Graph:** `ConvActivationFusion` gets a WebGPU-only rule that fuses `Conv(NHWC internal, with bias) -> Add(same-shape residual)` into
  `com.microsoft::NhwcFusedConv(X, W, B, Z)` (the existing contrib schema, which already has the optional residual input `Z`); the existing Conv+activation
  rule then also fuses a following ReLU into the NhwcFusedConv (`activation` attribute; order is act(conv + bias + Z)). `ORT_WEBGPU_CONV_ADD_FUSION=0` disables it.
- **Kernel:** the WebGPU `NhwcFusedConv` kernel reuses `Conv<true, true>`. The residual is applied natively in the vec4 channels-last MatMul path (1x1 convs)
  and in the Winograd output stage; every other conv kind (Conv2dMM 3x3 with < 64 channels, stride-2, grouped/depthwise, Im2col, odd channel counts) runs the
  convolution without its activation into a temporary and then one `ConvResidualAdd` program does act(a + z), so the result is always correct.
- ResNet-50 loses all 48 Add and 48 Relu dispatches. `gen_new_op_tests.py` has 12 Conv+Add(+ReLU) cases covering the native and fallback paths; sweep: 311 OK, 0 wrong
  (the first run caught a real bug: the Winograd call site was not passing the residual).
- **Also fixed:** the call site of the Winograd path never passed the weight cache (`winograd_u_`; a silently non-matching `replace` earlier), so the weight transform and the
  texture creation ran on every inference. With the cache in place the transform runs once.

Phone medians (fp32 logits within 8e-7 of the CPU, top-1 unchanged), add fusion off -> on, weights cached:

| model | before this change | add fusion + weight cache | + `ORT_WEBGPU_WINO_TEX=16` |
|---|---|---|---|
| ResNet-50 | 61.7 ms | **55.2-56.7 ms** | 53.8 ms |
| YOLO11n | 75.8 | 73.3 (fusion alone -3%) | 71.5 |
| YOLO26n | 66.2 | 65.1 | -- |
| SAM-L0 encoder | 370 | 369 | 337 |
| RT-DETR pre | 357 | 346 | 312 |

(The phone was a few percent slower this session than in earlier tables, so compare within a row.) ORT's default conv was 67 ms on ResNet-50, i.e. fp32-exact ResNet-50 is now
~17% faster than stock and ~20% faster with texture weights.
