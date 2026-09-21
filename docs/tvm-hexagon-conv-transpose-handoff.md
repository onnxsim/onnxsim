# TVM Hexagon ConvTranspose handoff

Updated: 2026-09-21

## Goal and current state

Continue vectorization and data-layout optimization for TVM ConvTranspose on the connected Xiaomi 12S (ADB serial `239dbd8f`, product `thor`). The existing benchmark is `scripts/android/bench_tvm_hexagon_conv_transpose.py`; it supports a static Mask R-CNN mask-head workload when `--model` is omitted.

**RPC blocker resolved (2026-09-21).** Sessions now work on the reconstructed cache build. Root cause: the SDK 6.4 dynamic `libc++abi.so.1` needs `aligned_alloc`/`__cxa_thread_atexit_impl`, which the DSP's built-in libc lacks (pushing `libc.so` does not help; FastRPC uses its own). Two fixes, both required:

1. Skel: relink statically against `libc++.a`/`libc++abi.a` with two shims (`aligned_alloc` via `posix_memalign`, `__cxa_thread_atexit_impl` no-op) -- `scripts/android/relink_hexagon_skel_static_libcxx.sh`. The stock skel is kept as `libhexagon_rpc_skel.so.dyn-libcxx.bak` in `hexagon_api_output`; rerun the script after any TVM Hexagon rebuild.
2. Kernels: the bench's link wrapper adds `-nostdlib++` so generated `.so` files don't NEED `libc++.so.1` (that dependency crashed `libc++abi` with a TLBMISS once the skel loaded).

The `dsp_libs/`/`DSP_LIBRARY_PATH` launcher edit in the cached `build.py` is no longer needed. A non-fatal `hexagon_rpc_send failed: 39` is printed from `release_resources()` at session exit; timings and correctness are unaffected (not yet root-caused).

## Results on the reconstructed setup (2026-09-21, N=8, IC=OC=256, 14x14, 2x2 s2)

Channel-vectorized ConvTranspose, median of 3, max abs err 4.9e-7, weights pre-packed, layout copies excluded:

| channel tile | NCHW input | NHWC input |
|---|---|---|
| 8 | 635 ms | 461 ms |
| 16 | 153 ms | 67.2 ms |
| 32 | 92.4 ms | 40.2 ms |
| 64 | 68.4 ms | 35.3 ms |
| 128 | -- | 34.6 ms |
| 256 | -- | 40.1 ms |

**Hand-written HVX qf32 kernel (`--qf32`, `_qf32_module`).** LLVM's Hexagon backend converts qf32<->sf around *every* fmul/fadd (374 converts for 374 ops in the generated asm; `-fast-math`/`llvm-options`/`llvm`-kind targets change nothing), i.e. ~4 HVX ops per MAC. A `te.extern` that calls `llvm.hexagon.V6.vmpy.qf32.sf.128B` + `vadd.qf32.128B` directly (one `vconv.sf.qf32` + bias add at the end) halves that. Buffers are bound with `data_alignment=128` (removes `vmem`+`valign` pairs). NHWC input, median of 3, max abs err 5.07e-7:

| oc tile (vectors) | pixel block | unroll | time |
|---|---|---|---|
| 32 (1) | 2 | 4 | 26.0 ms |
| 64 (2) | 2 | 4 | 18.8 ms |
| 128 (4) | 1 | 4 | 14.1 ms |
| 128 (4) | 2 | 2 | **12.8 ms** |

**fp16 (`bench_tvm_hexagon_conv_transpose_fp16.py`):** chunked qf16 (widened to qf32 every 8 channels) 4.15 ms at 2e-3 error, widening `vmpy.qf32.hf` 5.56 ms at 8.8e-4, plain qf16 4.04 ms at 1.2e-2 (too coarse); LLVM-generated fp16 is 12.4 ms (1.2e-2). Configs with many live accumulators silently return wrong results (see the README fp16 section).

End to end with the copies: 4.5 + 12.8 + 6.2 = ~23.5 ms (generic TOPI 5.89 s; LLVM-vectorized 45 ms). Findings: serial is 43 ms so 4 threads scale ~2.9x; pixel blocks of 7 get *slower* because LLVM auto-unrolls the reduction and stores every accumulator back to memory each iteration (56 `vmem` stores per 56 `vmpy`; `-unroll-count` etc. via `-llvm-options` did not reach it) -- keeping accumulators in registers for larger blocks is the next lever. Making the parallel task a (parity, oc-block) weight slice (`--weight-major`) did not help (16.5 ms), so weight re-streaming is not the limit. The same LLVM `pixel_block` schedule without intrinsics (`--pixel-blocks`) is slower than unblocked (75 ms at 7).

Layout copies: NCHW->NHWC 4.5 ms; NHWC->NCHW 32.5 ms untiled, and with channel tiling (fixed a fuse-order bug in the tiled schedule): c16 20.8, c32 17.5, c64 11.1, c128 7.7, **c256 6.2 ms**. Reproduces the pre-cache-loss numbers (tile 16: 149.7/66.8 ms). Best end-to-end so far: NCHW->NHWC 4.5 + kernel 34.6 (tile 128; optimum is 64-128) + NHWC->NCHW 6.2 ≈ 45 ms vs 5.89 s generic. Next: keep larger accumulator blocks in registers in the qf32 kernel, fp16, or fuse the layout copies into neighbours; the Hexagon arch is built for `v68` but the SoC reports V69 (SD 8+ Gen 1), so the "V73" in older notes is likely wrong.

## Persistent build locations

- TVM v0.17 source: `/home/takecheeze/.cache/tvm-hexagon/tvm-v0.17.0`
- Host build: `/home/takecheeze/.cache/tvm-hexagon/tvm-v0.17.0/build-host-hexagon`
- Hexagon API output: `/home/takecheeze/.cache/tvm-hexagon/build-hexagon-api-unix/hexagon_api_output`
- Hexagon SDK 6.4.0.2: `/home/takecheeze/.cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2`
- Python dependencies: `/home/takecheeze/.cache/tvm-hexagon/python-deps`
- NDK: `/home/takecheeze/android-sdk/ndk/27.2.12479018`

TVM host build uses LLVM 19, including a local cached-source compatibility change for LLVM 19's `getHostCPUFeatures()` API. The Hexagon API was configured for `v68` because the TVM 0.17 configuration expected SDK toolchain paths not present for the requested architecture. Confirm architecture/runtime compatibility before treating DSP timings as representative of V73.

## Environment and benchmark command

Use these environment settings from the repository root:

```sh
export PYTHONPATH=/home/takecheeze/.cache/tvm-hexagon/python-deps:/home/takecheeze/.cache/tvm-hexagon/tvm-v0.17.0/python
export TVM_LIBRARY_PATH=/home/takecheeze/.cache/tvm-hexagon/tvm-v0.17.0/build-host-hexagon
export LD_LIBRARY_PATH=/usr/lib/llvm-19/lib
export HEXAGON_RPC_LIB_DIR=/home/takecheeze/.cache/tvm-hexagon/build-hexagon-api-unix/hexagon_api_output
export HEXAGON_TOOLCHAIN=/home/takecheeze/.cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools
export HEXAGON_SDK_ROOT=/home/takecheeze/.cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2

python scripts/android/bench_tvm_hexagon_conv_transpose.py \
  --tiles 8,16,32,64 --repeat 3 --channel-only --channel-inputs nhwc
```

The script's default static workload is `N=roi_batch` (default 8), `IC=OC=256`, input spatial size `14x14`, and a `2x2` kernel with stride 2. To use the former Mask R-CNN model path, pass `--model PATH`.

## RPC debugging notes

See above for the resolution. To debug FastRPC load failures: `adb -s 239dbd8f logcat -c`, run once, then `logcat -d | grep "adsprpc\|tvm_rpc\|cdsprpcd"` (crashes appear as "Process ... CRASHED" with the faulting shared object; `dlopen_ex failed` lines name unresolved symbols). The per-run phone workspace is deleted on teardown.

Do not move the TVM source/build/SDK back to `/tmp`; it currently has 54 GB available but is volatile. The cache tree is persistent and has about 14 GB of TVM/SDK files.

## Prior performance context

Before the cache loss, measured V73 results for the same mask-head ConvTranspose were approximately:

- Generic TOPI implementation: `5.89 s`
- Direct parity-tile implementation (tile 16): `3.14 s`
- Channel-vectorized implementation, packed `[KH, KW, IC, OC]`, tile 16: about `149.7 ms` with NCHW input and `66.8 ms` with NHWC input
- DSP NCHW-to-NHWC copy: `4.34 ms`; NHWC-to-NCHW copy: `32.6 ms`
- Maximum absolute error observed: under `5e-7`

These are prior results, not measurements from the reconstructed setup. The latest uncommitted repository change extends the benchmark script with the static workload mode, input-layout selection, and a `setuptools` import needed by TVM 0.17 under Python 3.12. The branch is `codex/tinygrad-dsp-survey-tvm-int8`; progress was previously pushed to PR #1720. Check `git status` before staging: there are unrelated untracked files in the workspace.
