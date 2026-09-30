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
