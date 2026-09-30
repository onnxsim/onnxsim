# Adreno 730: padded shared-memory tile transpose miscompiles (Dawn/Vulkan)

`harness.cc` runs a WGSL transpose kernel through Dawn (the same Dawn/Tint that ORT's WebGPU EP
builds) and checks `o[c*R+r] == a[r*C+c]`. Build it against the static libs of an ORT
`--use_webgpu` build (`_deps/dawn-build`, abseil, spirv), e.g. for Android:

    clang++ -std=c++20 -static-libstdc++ -I <build>/_deps/dawn-src/include \
      -I <build>/_deps/dawn-build/gen/include harness.cc -o harness \
      -Wl,--start-group $(find <build>/_deps -name '*.a') -Wl,--end-group -landroid -llog -ldl
    ./harness k0_ort.wgsl R C ceil(R/16) ceil(C/16)

`k0_ort.wgsl` is ORT's tiled Transpose (`tile: array<array<f32, 17>, 16>`, workgroup 16x16, guarded
store, `workgroupBarrier()`). Results on Adreno 730 (Qualcomm Vulkan 512.615.0, compiler
EV031.36.08.11, vendor `qualcomm`, arch `adreno-7xx`); NVIDIA RTX 5050, RADV and lavapipe are
correct for every kernel and shape here:

| kernel | change from k0 | Adreno 730 (96x32) |
|---|---|---|
| k0_ort | (ORT's kernel) | 27% wrong (also 1x16: 50%, 3x1024: 50%, 256x256: 26%) |
| k1_nopad | tile row stride 16 instead of 17 | correct on 8 shapes incl. 1x16, 33x17, 3x1024 |
| k2_flat | flat `array<f32,272>`, stride 17 | 25% wrong (not the nested array) |
| k4_noguard | drop the two bounds `if`s | correct (needs exact-multiple shapes; not a fix) |
| k5_storebar | + `storageBarrier()` | 2.4% wrong (looks like a race) |

So the odd (17) padded stride combined with the guarded store trips the Adreno compiler; the
unpadded tile is correct. Tint has Qualcomm workarounds (matrix pass-by-pointer, std140, NClamp,
uniform vector component loads) but none for workgroup memory or barriers.
`../ort_transpose_qualcomm_unpadded_tile.patch` makes ORT use the unpadded tile when the adapter
vendor is `qualcomm`; with it the 235-op sweep passes 230/235 on the phone (the other 5 are 1-ulp
input ties that the phone CPU EP shows too).

## Microbenchmarks used for the convolution investigation (see `../../WEBGPU_SURVEY.md`)

Build each like `harness.cc`; run on the phone under `phone-run`.

- `peak.cc` -- FMA throughput: `peak f32|f16 WORKGROUPS ITERS [VEC CHAINS WG]` (986 GFLOPS fp32 at 64 scalar chains).
- `bw.cc` -- vec4 load bandwidth: `bw buf|tex|lds WORKING_SET_KB ITERS` (buffer ~163 GB/s, texture ~240, workgroup memory 200-277).
- `gemm.cc` -- GEMM design variants: `gemm sh|reg|sc|p16|sg|tx|tt|nc4 M N K REPS [TM NV WX WY]`; `RB=off VMM=1` mirror ORT's Dawn toggles.
  `sh` shared-memory tiles (ORT's design), `reg`/`sc` register tiles with vec4/scalar accumulators, `tx`/`tt` A (and B) through
  textures, `nc4` NC4HW4-style coalesced layout, `p16` A/B stored as packed f16 (unpacked to f32, f32 accumulate), `sg` `p16` with A shared across a row group through `subgroupShuffle`. REPS>1 keeps the whole GPU busy (throughput); REPS=1 is one real layer's latency.
- `chain.cc` -- dispatch overhead: `chain dep|ind N ELEMS [PASSES SUBMIT_EVERY]` (a dependent 4096-element dispatch costs ~6 us
  of GPU time and ~2 us of CPU in Dawn directly).
- `conv_alt.cc` -- Winograd F(2,3) vs direct register-tile 3x3 conv, and depthwise 3x3 variants: `conv_alt conv|dw C H` (see `../../WEBGPU_SURVEY.md`).
- `conv_s2.cc` -- 3x3 stride-2 pad-1 NHWC conv (ResNet-50 v1.5 downsampling): direct implicit GEMM vs a polyphase hybrid Winograd (25 mults per 2x2 output tile instead of 36): `conv_s2 C H`.
- `conv_f16io.cc` -- whole-conv packed-f16 path (f16 activations + weights, f32 accumulate, bias+ReLU, packed f16 out) vs the f32 register tile on ResNet 1x1/3x3 shapes, plus a 10-layer error-accumulation chain: `conv_f16io pw Cin Cout H | c3 C H | chain C H L`. Build like `conv_alt.cc`.
- `cl_vs_wg.cc` -- the same microbenchmarks in **OpenCL** (libOpenCL.so loaded with dlopen, Khronos headers only): `cl_vs_wg info|warm S|peak|bw|gemm|conv|hint|hold S`.
  FMA peak (float/half/half2/half4), load bandwidth (buffer float4/half4/half8, image2d RGBA32F/RGBA16F), register-tile GEMM and direct 3x3 conv in
  buffer/image and f32/f16 variants, the `cl_qcom_perf_hint` experiment. Compare with `peak`, `bw`, `gemm`, `conv_alt` in the same session (see `../../WEBGPU_SURVEY.md`).
