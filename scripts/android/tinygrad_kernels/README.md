# tinygrad-generated matmul / conv kernels for WebGPU (WGSL) and Vulkan (SPIR-V)

Generates compute kernels for matmul and 2-D convolution with [tinygrad](https://github.com/tinygrad/tinygrad) (0.14.x),
renders each one for **both** targets, runs them on the phone and checks them against a numpy reference:

| target | tinygrad renderer | consumed by |
|---|---|---|
| WebGPU | `WGSLRenderer` (tinygrad's own) | Dawn's Vulkan backend (`tgrun_dawn`) |
| Vulkan | `GLSLRenderer` (`glsl_renderer.py`, new: tinygrad has no SPIR-V path) -> `glslangValidator` -> SPIR-V | raw Vulkan (`tgrun_vk`) |

Generation needs no GPU: tinygrad's lowering runs offline (the `"WEBGPU"` device string is only a tag, see
`onnxsim/webgpu_tinygrad_codegen.py`). It needs tinygrad 0.14.x on `PYTHONPATH`, `numpy` and `glslangValidator`.

## Files

- `glsl_renderer.py` -- Vulkan GLSL 4.50 compute renderer on tinygrad's `CStyleLanguage` (float/int32/uint32/bool; std430 buffer per
  kernel parameter at binding = slot; `shared` arrays hoisted; logical `&&`/`||`/`!=` for bool ops).
- `tgk.py` -- library: problem definitions (`mm:MxNxK`, `conv:N,Cin,H,W,Cout,kh,kw[,stride[,pad]]`), lowering to the single scheduled
  kernel, rendering to WGSL + GLSL + SPIR-V, tinygrad's Opt candidate enumeration, reference data.
- `gen.py` -- CLI: the default kernel plus BEAM's one-step Opt candidates (from the plain gridded kernel `c<i>` and from the default
  heuristic state `d<i>`), written to a problem directory with a `manifest.txt`, inputs and the numpy reference.
- `tune.py` -- device-timed BEAM search: expands the top states by every further Opt, times all candidates on the phone, repeats.
  tinygrad's own BEAM cannot open the phone's GPU from the host, so this reuses only its device-free half (Scheduler,
  `get_kernel_actions`) and does the timing through the runners over adb, under the shared phone lock.
- `tgrun_dawn.cc`, `tgrun_vk.cc`, `tgrun_common.h` -- the runners. Each prints one `RESULT` line per variant: correctness
  (`err` = max abs error / max |ref|, OK if < 1e-3), pipeline compile time, single-dispatch latency, per-dispatch time in a back-to-back
  batch, GFLOPS (`tgrun_vk` also the GPU-timestamp time).

## Use

```
export PYTHONPATH=<tinygrad 0.14 checkout>:scripts/android/tinygrad_kernels
python scripts/android/tinygrad_kernels/gen.py OUT mm:784x512x128 conv:1,256,14,14,256,3,3,1,1 --candidates 40
python scripts/android/tinygrad_kernels/tune.py OUT mm:784x512x128 --rounds 5 --beam 4 --backend vulkan
# with register-tile seeds (16-64 accumulators/thread), which one-step BEAM rarely reaches: add --tiles 100 --max-new 60
```

Build the runners with the NDK (`tgrun_vk`: `-lvulkan -static-libstdc++`; `tgrun_dawn`: link the static Dawn/Tint/abseil/spirv
libraries of an ORT `--use_webgpu` build in a linker group, as `../webgpu_ops/dawn_repro/README.md` describes) and push them with
the problem directory to `/data/local/tmp/tg-kernels`.

## Things that bit

- **Candidates must be gridded first.** tinygrad's own `apply_opts` calls `Scheduler.convert_loop_to_global()` before applying any Opt;
  enumerating `get_kernel_actions` on a fresh `Scheduler` (as `onnxsim/webgpu_kernel_tuning.py` does) yields serial one-thread kernels.
- **WebGPU limits are lower than Vulkan's.** 256 invocations per workgroup (Vulkan on Adreno: 1024): a few Opt combinations run on
  Vulkan and are invalid on WebGPU. `tgrun_dawn` keeps Dawn validation on, so these are reported as `FAIL` instead of aborting.
- **Do not batch slow kernels.** A submission of several 200 ms dispatches trips Android's GPU hang watchdog (`VK_ERROR_DEVICE_LOST`);
  the runners use one dispatch per submission when a single one takes > 20 ms.
- tinygrad's WGSL always declares an `INFINITY` uniform at binding 0 (storage buffers start at 1); the GLSL renderer has none
  (buffers start at 0). `tgrun_dawn` builds an explicit bind-group layout because the uniform is usually unused.

## Results on the Adreno 730 (Snapdragon 8+ Gen 1, 2026-10-01)

`tune.py --rounds 5 --beam 4 --backend vulkan`, then the best variant of each problem and tinygrad's default kernel re-run through both
`tgrun_vk` and `tgrun_dawn` in one warm session. All 32 runs match the numpy reference (max relative error 1.4e-6). GFLOPS are per-dispatch
times in a back-to-back batch of dependent dispatches (single-dispatch latency is higher). The best kernels are in `best/<problem>/`
(`tuned.wgsl`, `tuned.comp`, the Opts in `best.txt`, launch dims in `launch.txt`), next to the default ones.

| problem | GFLOP | tinygrad default (Vulkan) | tuned, Vulkan | tuned, WebGPU (Dawn) |
|---|---|---|---|---|
| mm 784x512x128 | 0.103 | 103 GFLOPS (1.00 ms) | **205** (0.50 ms) | **202** (0.51 ms) |
| mm 3136x256x64 (1x1 conv as GEMM) | 0.103 | 105 (0.98 ms) | **201** (0.51 ms) | **207** (0.50 ms) |
| mm 784x128x1152 | 0.231 | 130 (1.78 ms) | 142 (1.63 ms) | 142 (1.63 ms) |
| mm 196x256x2304 | 0.231 | 55 (4.17 ms) | 134 (1.72 ms) | 136 (1.70 ms) |
| conv 1x64x56x56 -> 256, 1x1 | 0.103 | 105 (0.98 ms) | **194** (0.53 ms) | **194** (0.53 ms) |
| conv 1x64x56x56, 3x3 | 0.231 | 134 (1.73 ms) | 131 (1.77 ms) | 165 (1.40 ms) |
| conv 1x128x28x28, 3x3 | 0.231 | 90 (2.57 ms) | 166 (1.39 ms) | 158 (1.47 ms) |
| conv 1x256x14x14, 3x3 | 0.231 | **6** (38.6 ms) | 92 (2.51 ms) | 84 (2.75 ms) |

- tinygrad's untuned kernels are often poor (its 3x3 conv on a 14x14 grid runs at 6 GFLOPS). The device-timed BEAM search gains
  0.98x-15x over the warm default, typically ~2x (1.9-2.4x on the GEMM-shaped problems, 1.8x on the 28x28 conv, none on the 56x56 3x3 conv
  and 1.09x on mm 784x128x1152): 3-5 rounds of ~20-70 candidates, a few minutes per problem including the phone round trips.
- **Measure warm.** GPU clock state matters: `tune.py`'s round-0 default reading (taken first, on an idle GPU) was 74 GFLOPS for the 56x56 conv
  against 134 GFLOPS in the warm re-run above, so the "default" printed by `tune.py` understates the baseline; the table above is the fair comparison.
- WebGPU (WGSL through Dawn/Tint) and raw Vulkan (GLSL through glslang) agree within ~5% on 6 of 8 tuned kernels; the outliers are the 56x56
  3x3 conv (Dawn 165 vs Vulkan 131 GFLOPS) and the 14x14 one (84 vs 92). The compilers emit different SPIR-V for the same tinygrad kernel, so
  the same Opts are not always equally fast on the two targets. The untuned kernels agree on the GEMMs and 1x1 conv, and differ on the 3x3 convs
  (Dawn 168 vs 134 at 56x56, 110 vs 90 at 28x28).
- For context, ORT's WebGPU Conv/MatMul kernels run ResNet-50's layers at roughly 120-175 GFLOPS (`../WEBGPU_SURVEY.md`; profiler GPU
  timestamps per dispatch, a different timing method, so only roughly comparable). The tuned tinygrad kernels are in the same range: ahead on the
  GEMM/1x1 shapes (~200 GFLOPS) and behind on the 14x14 3x3 conv (92 vs ~124).

## Register-tile seeds (2026-10-01)

The hand-written GEMM in `../webgpu_ops/dawn_repro/gemm.cc` reaches 255-282 GFLOPS with scalar accumulators and an 8x8 output tile per
thread (64 accumulators), against ~175 for an ORT-style shared-memory GEMM in the same session. `tune.py` above stopped at ~200 because
BEAM adds one Opt per round from a one-thread-per-output kernel and rarely gets to that tile. `tgk.tile_candidates` (`tune.py --tiles N`)
now seeds round 1 with random `UPCAST a x b` (a*b = 16/32/64 accumulators over two global axes) + `UNROLL` + `LOCAL` (32-256 invocations)
combinations that tinygrad accepts, then the usual beam continues from the best ones. The tuner needs no other change; the TM=8,NV=2 tile is
`UPCAST axis0 8, UPCAST axis1 8` and is found on mm 784x512x128 and 3136x256x64.

Re-tuned with `--tiles 100 --max-new 60 --rounds 3 --beam 4`. Warm re-run of the tinygrad default, the new best and the previous best in one
session (GFLOPS from per-dispatch times; the phone was slower/noisier than in the first table, so compare within a row, not against the table above):

| problem | default (Vulkan) | new best, Vulkan / Dawn | previous best, Vulkan / Dawn | tuner-time reading (Vulkan) |
|---|---|---|---|---|
| mm 784x512x128 | 60 | 196 / 142 | 179 / 151 | 253 |
| mm 3136x256x64 | 61 | **212** / **183** | 184 / 159 | 265 |
| mm 784x128x1152 | 76 | **138 / 120** | 92 / 80 | 160 |
| mm 196x256x2304 | 38 | 122 / 120 | **133 / 136** (kept) | 134 |
| mm 196x1024x256 (new) | 42 | 161 / 137 | - | 177 |
| mm 49x512x4608 (new) | 25 | 120 / 118 | - | 125 |
| conv 64->256 1x1 @56 | 61 | 197 / 193 | 193 / 175 | 253 |
| conv 64 3x3 @56 | 77 | **162 / 164** | 135 / 155 | 201 |
| conv 128 3x3 @28 | 46 | 57 / 101 | **110 / 146** (kept) | 109 |
| conv 256 3x3 @14 | 6 | 96 / 85 | 96 / 86 (kept) | 94 |

- `best/<problem>/` holds the winner of each row (new for 7 problems, the previous kernel for the last three).
- The tuner's own readings (253-265 GFLOPS on the 1x1/GEMM shapes) approach the hand kernel's 282 but do not survive a warm re-run (196-212):
  round-0/round-N readings on this phone drift by 20-25% with GPU clock state, so only same-session comparisons mean anything.
- What still separates tinygrad from the hand kernel: tinygrad requires every LOCAL/UPCAST factor to divide the axis (M=784 or 196 gives
  only small workgroup factors, the hand kernel guards the tail instead) and it cannot vectorize the B loads across the 8 columns in the
  way the hand kernel does. On 784x128x1152 (hand 255) and 196x256x2304 (hand 224) that leaves it at 120-140.
- Not done: a multi-kernel (whole-network) runner; every problem here is still a single kernel.
