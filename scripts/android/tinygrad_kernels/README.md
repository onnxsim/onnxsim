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
