# Benchmarking tinygrad's code generation through an onnxsim RPC server

`bench_codegen_rpc.py` is an RPC *client*. The server (see `docs/rpc.md`) runs on the machine whose
hardware you want to measure, with tinygrad installed, and runs each ONNX workload with onnxruntime
and with tinygrad on a chosen device and codegen setting. The client reports correctness against
onnxruntime, wall time per call, and tinygrad's kernel statistics.

```bash
# on the target (tinygrad + onnxruntime installed; clang on PATH for tinygrad's CPU device)
python -m onnxsim.rpc server --port 9191 --key bench
# anywhere that can reach it
python scripts/tinygrad/bench_codegen_rpc.py --port 9191 --key bench --device CUDA --beam 0,2
# tinygrad's Hexagon renderer under QEMU: a second server started with MOCKDSP=1
MOCKDSP=1 python -m onnxsim.rpc server --port 9192 --key dspmock
python scripts/tinygrad/bench_codegen_rpc.py ... --dsp-server 127.0.0.1:9192 --dsp-key dspmock
```

How the numbers are taken: the server times a `TinyJit` replay, so the Python ONNX interpreter's
per-call cost (1-5.6 ms eager on these shapes, against 0.1-2 ms replayed) is out of the
measurement, and reads device kernel time from tinygrad's counters (one replay under `DEBUG=2`).
Each loaded tinygrad model runs in its own worker process, so every `(device, BEAM)` pair starts
from clean caches.

## Results

Measured on one desktop: RTX 5050, tinygrad 0.14.0, onnxruntime 1.30 on a 32-core CPU, fp32,
batch 1, ResNet-50 / Mask R-CNN backbone layer shapes. Every tinygrad output matches onnxruntime
(max abs error 1e-6 to 2e-5). Each workload compiles to a single tinygrad kernel. Device kernel
time in ms (GFLOP/s in parentheses); onnxruntime is wall time on the CPU:

| Workload | ORT (CPU) | tinygrad CUDA | tinygrad CUDA `BEAM=2` | tinygrad NV | tinygrad NV `BEAM=2` |
|---|---:|---:|---:|---:|---:|
| 7x7 s2 stem 3->64 @224 | 0.23 | 0.099 (2380) | **0.055 (4290)** | 0.090 | 8.29 |
| 3x3 64->64 @56 | 0.24 | 0.151 (1530) | **0.070 (3300)** | 0.143 | 8.13 |
| 1x1 256->64 @56 | 0.29 | 0.075 (1370) | **0.029 (3550)** | 0.066 | 3.57 |
| 3x3 256->256 @14 | 0.14 | 2.102 (110) | **0.088 (2620)** | 2.196 | 8.13 |
| matmul 1024^3 + bias + relu | 1.06 | 1.393 (1540) | **0.265 (8100)** | 1.329 | 74.3 |
| add + relu 256x56x56 | 0.18 | 0.018 | **0.013** | 0.010 | 1.18 |

Findings:

1. **Default heuristics leave one big outlier.** The 3x3 256->256 convolution at 14x14 takes
   2.1 ms (110 GFLOP/s), 25x slower than the similar 3x3 64->64 at 56x56. `BEAM=2` fixes it
   (24x, 2.6 TFLOP/s).
2. **`BEAM=2` helps everywhere heavy on the CUDA backend**: 1.8-2.6x on the other convolutions and
   5.3x on the matmul, which reaches ~8 TFLOP/s (near the card's fp32 peak). With `BEAM=2` the GPU
   beats the 32-core CPU onnxruntime on every workload here.
3. **`BEAM` is broken on tinygrad's NV backend in this setup**: the "search" finishes in about a
   second and every kernel comes out at a uniform ~28 GFLOP/s (up to ~120x slower than the
   default), whatever the workload. The same setting on the CUDA backend works, so this looks
   like a tinygrad NV-backend problem (worth reporting upstream), not a property of BEAM. It
   surfaced only because the server isolates each configuration in a fresh process.
4. onnxruntime's CPU numbers move 2-7x between runs on this shared machine (for example the 1024
   matmul measured 1.0 ms and 7.6 ms in different runs), so read that column as indicative; the
   GPU kernel times were stable to a few percent.

### tinygrad's Hexagon renderer (Snapdragon 845 / mock DSP, QEMU)

`TINYGRAD_HEXAGON_TARGET=snapdragon845` selects the Snapdragon 845's Hexagon
685 / V65 HVX profile. Kernels are compiled with `clang --target=hexagon
-mcpu=hexagonv65 -mhvx=v65 -mhvx-length=128b` and run under
`qemu-hexagon-static`; the "time" is an instruction count, reported per multiply-add:

| Workload | instructions | instructions per MAC |
|---|---:|---:|
| 1x1 64->64 @28 | 10.4 M | 3.25 |
| 3x3 64->64 @14 | 34.6 M | 4.79 |
| matmul 128^3 | 12.4 M | 5.93 |

That is 3-6 instructions per MAC from a single-thread scalar loop, roughly 50-100x what a
vectorized HVX kernel needs, so tinygrad's default Hexagon codegen leaves the vector unit unused.
The phone's DSP itself is out of reach for tinygrad's runtime (see `scripts/android/maskrcnn_e2e`).
