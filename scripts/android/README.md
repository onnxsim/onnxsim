# ONNX Runtime Android integration smoke test

This smoke test runs an ONNX model and its `onnxsim` result with ONNX Runtime's
CPU execution provider on a connected Android arm64 device. When given the QNN
provider AAR, it also tries QNN HTP, QNN GPU, and Android NNAPI with NNAPI's CPU
device disabled. It also builds a tiny NNAPI RELU model and compiles it directly
for `qti-dsp`, checking that the DSP driver accepts and executes the program.
The QNN sessions disable ORT CPU fallback; a pass means QNN accepted the
complete graph. The automatic NNAPI run records available device names/types
and disables NNAPI's CPU device.

The CPU and NNAPI probes keep a float32 `Relu` node and remove redundant
`Identity` nodes around it. The QNN HTP probe uses a separate quantize/dequantize
`Relu` graph because HTP expects quantized models. Each run checks both graph
outputs against the expected values with CPU fallback disabled for hardware
providers.

## Requirements

- A connected arm64 Android phone on API 29+ with USB debugging enabled and `adb` authorized.
- Android SDK platform tools, API 26 or newer, and NDK 27.2 (or a compatible NDK).
- CMake and Python packages `onnx`, `numpy`, and this repository's `onnxsim`.
- An ONNX Runtime Android AAR, such as `onnxruntime-android-1.30.0.aar`.
- For DSP/GPU runs, a QNN provider AAR built for the same ONNX Runtime version.
- For QNN HTP/GPU app runs, the matching Qualcomm QNN runtime AAR as well. The
  provider AAR contains the ORT plugin; the runtime AAR supplies QNN backend
  libraries and HTP skeletons needed by Android apps.

## Run

```bash
python scripts/android/run_onnxruntime_android.py \
  --runtime-aar /path/to/onnxruntime-android.aar \
  --qnn-aar /path/to/onnxruntime-android-qnn.aar \
  --qnn-runtime-aar /path/to/qnn-runtime.aar \
  --android-sdk "$ANDROID_HOME" \
  --ndk-version 27.2.12479018
```

Use `--adb /path/to/adb` or `--serial SERIAL` to select a specific ADB client or
device. QNN HTP and GPU runs are attempted when `--qnn-aar` is provided; a
backend that cannot load or execute reports `SKIP`. Use `--require-htp` and/or
`--require-gpu` to make the matching QNN backend mandatory. Use
`--require-nnapi-hw` to require NNAPI execution with its CPU device disabled.
Use `--require-nnapi-dsp` to require the explicit `qti-dsp` NNAPI compile and run.
Use `--only-target qnn-htp` or `--only-target nnapi-no-cpu` (repeatable) to run
selected accelerator targets. For a real model, pass `--model` with its sample
`--input-tensor-pb`; `--reference-output-pb` additionally checks the CPU output
against a published sample output. Repeat that option in graph output order for
multiple outputs. The Android runners currently accept float32 or int64 outputs.
Android chooses among its available NNAPI hardware devices, so this check does
not claim a specific GPU driver was selected. CPU is always required. Host build
artifacts use a temporary directory unless `--work-dir` is provided; phone
staging files are removed after the run. The debug test APK remains installed.

The runtime and QNN AARs are not vendored. The script extracts only the arm64
runtime library, headers, and QNN provider needed to build the smoke-test
executable.

## Real MobileNetV2 model

The same runner accepts the ONNX Model Zoo MobileNetV2 QDQ model and its
published sample tensors. The model expects an RGB tensor `[1,3,224,224]`
normalized with ImageNet mean and standard deviation. Its archive includes the
preprocessed input and expected 1000-class output. After extracting it, run:

```bash
python scripts/android/run_onnxruntime_android.py \
  --runtime-aar /path/to/onnxruntime-android.aar \
  --qnn-aar /path/to/onnxruntime-android-qnn.aar \
  --android-sdk "$ANDROID_HOME" \
  --model /path/to/mobilenetv2-12-qdq.onnx \
  --input-tensor-pb /path/to/test_data_set_0/input_0.pb \
  --reference-output-pb /path/to/test_data_set_0/output_0.pb
```

The fp32 MobileNetV2 archive can be tested the same way. QNN HTP is tested with
the QDQ variant; NNAPI can be tested with either variant.

## Xiaomi 12S probe result

The CPU and NNAPI float32 probes pass, including direct compilation on
`qti-dsp`. Android reports `qti-gpu` and `qti-dsp` hardware devices; the NNAPI
execution provider chooses the device automatically, while the direct DSP probe
explicitly selects `qti-dsp`. The QNN EP exposes an NPU device, but the
quantized QDQ `Relu` graph still leaves nodes assigned to the default CPU EP
with fallback disabled. It also does not expose a separate QNN GPU device on
this phone, so QNN HTP/GPU remain unconfirmed.

The real ONNX Model Zoo MobileNetV2 fp32 and QDQ models both pass Android CPU
comparison against their published sample outputs. The QDQ model still leaves
nodes on CPU in QNN HTP, and NNAPI leaves nodes on CPU for both fp32 and QDQ
MobileNet when CPU fallback is disabled. The independent direct `qti-dsp` RELU
probe passes on the image tensor's first four values.

## TVM Hexagon Mask R-CNN kernel probe

`test_tvm_hexagon_maskrcnn.py` is an opt-in test for TVM's Hexagon code
generator and RPC execution path. It reads ResNet/FPN convolution and pooling,
resize, RoIAlign, mask-head ConvTranspose, and a static uint8 QDQ pair from a
Mask R-CNN ONNX model. It compiles representative convolution + bias + ReLU,
max-pooling, nearest-neighbor FPN resize, four-level 7x7 RoIAlign,
mask-head 2x transpose convolution, and quantize/dequantize kernels for V73,
runs them on the connected Hexagon DSP, and compares their results with
TVM/LLVM CPU kernels or NumPy. The convolution schedule vectorizes eight
contiguous output-width elements and parallelizes the remaining output tiles;
pooling uses 32-wide output tiles, while resize, RoIAlign, and QDQ use
Hexagon's vectorized/parallel injective schedule. On the tested V73 device,
these schedules reduced representative kernel times versus the initial
schedule by 5.7x for ResNet 3x3 convolution, 4.7x for mask-head transpose
convolution, 2.5x for 56x56 RoIAlign, 3.5x for QDQ, 1.6x for resize, and 1.5x
for max-pooling. These speedups compare the old and updated schedules on the
same device, with buffers preallocated and input/output transfers excluded.
The sweeps used three single-invocation timings for convolution, pooling,
resize, and RoIAlign; five for QDQ; and one for transpose convolution. They are
per-kernel timings, not end-to-end model latency. The uint8 quantized values
are checked exactly; dequantized floats allow a 1e-5 absolute tolerance
for DSP floating-point rounding. The ROI workloads use a configurable
synthetic proposal batch (default 8) because the model's ROI count is dynamic.
Tensor values, weights, and regions are randomized. This checks individual
kernels, not the full Mask R-CNN graph, model weights, QNN integration, or
detector accuracy.

The probe needs an Apache TVM build with Hexagon enabled, the matching Hexagon
SDK/toolchain, and Python packages `onnx`, `numpy`, and TVM's dependencies. Set
`HEXAGON_SDK_ROOT` and `HEXAGON_TOOLCHAIN` to the corresponding SDK roots and
make the TVM Python package and host library available with `PYTHONPATH` and
`TVM_LIBRARY_PATH`. For example:

```bash
python scripts/android/test_tvm_hexagon_maskrcnn.py \
  --model /path/to/MaskRCNN-12-qdq.onnx \
  --serial ANDROID_SERIAL
```

Generated shared objects are stored in `/tmp/tvm-maskrcnn` by default. Pass
`--artifact-dir` to choose another location or `--roi-batch` to change the
synthetic ROI workload size. The device needs to be reachable by ADB, and the
host must allow local TVM RPC connections.

## TVM Hexagon NCHWc int8 convolution probe

`bench_tvm_hexagon_nchwc_int8.py` compares a scalar NCHW uint8-by-int8
convolution with TVM's Hexagon NCHWc schedule, which tensorizes the inner
reduction with HVX `vrmpy`. The probe checks each int32 accumulator exactly
against NumPy and reports five-run median kernel times on the connected phone.
It uses 64 input and output channels at 56x56, with 1x1 and 3x3 kernels. It
times prepacked device inputs and weights; host-side layout packing, RPC
transfers, bias, and output requantization are excluded. These are kernel
results, not end-to-end ONNX model latency.

```bash
python scripts/android/bench_tvm_hexagon_nchwc_int8.py --kernels 1,3
```

On the tested Xiaomi 12S, the 1x1 probe measured 6.768 ms for scalar NCHW and
0.330 ms for tensorized NCHWc (20.5x). The 3x3 probe measured 176.402 ms and
1.212 ms respectively (145.6x). Disassembly of the generated 3x3 module
contains HVX `vrmpy` instructions. Inputs and weights use uint8 and int8 dtypes
with small random ranges; the measurements do not include quantization
parameter handling or real model weights.

## Tinygrad Hexagon survey

See [TINYGRAD_HEXAGON_SURVEY.md](TINYGRAD_HEXAGON_SURVEY.md) for the survey of
Tinygrad's local Hexagon backend. It targets V65 and expects Linux FastRPC
device nodes, so its DSP runtime does not directly match this Android/TVM RPC
setup. The source offers a codegen experiment, but does not establish a
phone-specific memory-bandwidth model.

## Other Mask R-CNN Hexagon operator timings

`bench_tvm_hexagon_maskrcnn_ops.py` extends the phone measurements to
max-pooling, FPN resize, RoIAlign, mask-head ConvTranspose, and the model's
uint8 quantize/dequantize pair. It uses the model-derived shapes, randomized
values, preallocated DSP buffers, five single-invocation timing samples, and
operator-specific CPU/NumPy correctness checks. Input generation, transfers,
and output reads are outside the timed region. Run it with:

```bash
python scripts/android/bench_tvm_hexagon_maskrcnn_ops.py \
  --model /path/to/MaskRCNN-12-qdq.onnx
```

On the tested Xiaomi 12S (Hexagon V73), median kernel times were:

| Operator | Workload | Median |
|---|---|---:|
| MaxPool | `[1,64,112,112]`, 3x3, stride 2 | 6.894 ms |
| FPN resize | `[1,256,14,14]` to `[1,256,28,28]` | 7.654 ms |
| RoIAlign | `[8,256,56,56]` to `[8,256,7,7]` | 27.712 ms |
| Mask-head ConvTranspose | `[8,256,14,14]` to `[8,256,28,28]` | 5802.006 ms |
| Quantize + dequantize | model input `[1,3,224,224]` | 3.890 ms |

The generic TOPI ConvTranspose timing was 5.822 s in a follow-up run. The
experimental `bench_tvm_hexagon_conv_transpose.py` compares it with a direct
stride-2 parity-plane schedule for this exact 2x2, zero-padding workload. The
direct schedule avoids the three zero positions introduced by input dilation,
reducing arithmetic from about 1.64B to 411M MACs. On the Xiaomi 12S, width
tiles 4, 8, and 16 measured 5.666 s, 3.680 s, and 3.140 s respectively; tile 16
was 1.85x faster than the generic baseline. All direct variants matched the
NumPy reference with maximum absolute error below 5e-7. This is useful but
still far from practical latency, so the specialization remains an exploratory
benchmark pending better Hexagon vectorization and scheduling. These numbers
are per-kernel synthetic workloads, not end-to-end Mask R-CNN inference.

The next schedule vectorizes output channels and packs the weights to
`[kernel_h, kernel_w, input_channel, output_channel]`, with NHWC output. At
channel tile 16 it measured 149.7 ms with NCHW input and 68.1 ms with NHWC
input. The corresponding DSP layout copies took 4.3 ms for NCHW-to-NHWC input
and 32.6 ms for NHWC-to-NCHW output. Including both copies, the NHWC-input
path is about 105 ms, roughly 56x faster than the 5.89 s generic baseline;
weight packing is excluded because model weights can be packed once. Numerical
error remained below 5e-7. The output conversion still costs about half of the
optimized path, so fusing it into a following operator or retaining NHWC across
operators is the next opportunity. The results are from an exploratory
benchmark and do not yet change the general ConvTranspose implementation.

## Remaining Mask R-CNN operator coverage

`bench_tvm_hexagon_maskrcnn_more_ops.py` covers the rest of the model's operator set
(`MaskRCNN-12-qdq.onnx` has 42 op types; the six above were covered before). Static shapes
(residual feature maps, box-head weights, the RPN NMS IoU threshold) are read from the model;
data-dependent counts use `--roi-batch` (8) and `--proposals` (1000). Each kernel is verified
against a NumPy reference on the phone, then timed (median of 5 single invocations, warm
buffers, so the sub-10 us rows are at the timer's resolution).

```bash
python scripts/android/bench_tvm_hexagon_maskrcnn_more_ops.py \
  --model /path/to/MaskRCNN-12-qdq.onnx
```

| Group | ONNX ops | Workload | Median |
|---|---|---|---:|
| Residual | Add, Relu (+fused) | `[1,256,56,56]` add / relu / add+relu | 0.51 / 0.29 / 0.46 ms |
| | | `[1,2048,7,7]` add / relu / add+relu | 0.07 / 0.01 / 0.07 ms |
| Box head | MatMul, Add, Relu | `[8,12544]x[12544,1024]` generic TE schedule | 78.5 ms |
| | | same, hand-written HVX qf32 (weights pre-packed) | 12.6 ms |
| | | `[8,1024]x[1024,1024]` generic / HVX | 6.2 / 1.19 ms |
| | | `[8,1024]x[1024,324]` (bbox) generic / HVX | 8.4 / 0.61 ms |
| | | `[8,1024]x[1024,81]` (cls) generic / HVX | 2.5 / 0.11 ms |
| | Softmax, Clip, Div | `[8,81]` | 0.05 ms |
| Mask head | Sigmoid | `[8,81,28,28]` TOPI (libm `expf`) / polynomial exp | 10.0 / 1.0 ms |
| Box decode | Exp, Mul, Add, Sub, Div, Clip | 1000 boxes | 0.11 ms |
| Level mapper | Sqrt, Log, Floor, Clip | 1000 boxes | 0.05 ms |
| Filtering | Greater, Less, Not, And, Cast | 9408 anchors | 0.05 ms |
| Proposals | TopK | 9408 candidates to 1000 | 0.93 ms |
| | NonMaxSuppression | 1000 boxes, IoU 0.7 (greedy) | 8.1 ms |
| | NonZero | 9408-element mask | 0.05 ms |
| | Gather | 1000 rows of `[9408,4]` | 0.02 ms |
| RoI merge | ScatterElements | `[8,256,7,7]` | 0.44 ms |
| Layout | Concat, Slice/Split, Transpose, Flatten, ReduceMin | RPN levels `12543x4`, `[1000,4]`, `[1,1000]`, `[8,256,7,7]` | 0.12 / 0.01 / 0.003 / 0.20 / 0.002 ms |

Metadata-only ops (`Shape`, `Unsqueeze`, `Squeeze`, `Reshape`, `ConstantOfShape`, `Expand`,
integer `Cast`) touch tiny tensors and are not benchmarked. Notes from getting these to run
well on Hexagon:

- The generic box-head MatMul streams weights at under 1 GB/s. Column-tile parallelism with
  all activation rows held in HVX qf32 accumulators (the ConvTranspose kernel's approach) is
  6-23x faster; fc6 is then weight-bandwidth bound (51 MB fp32), which the model's int8
  weights would cut 4x.
- `te.floor` and `te.abs` lower to one scalar libm call per lane, and `tvm.tir.if_then_else`
  keeps a loop scalar; `Select` plus an int-cast floor keeps polynomial `exp`/Sigmoid
  vectorized (10x faster than TOPI's Sigmoid).
- NMS and NonZero are single-threaded `te.extern` loops; NMS at 8 ms for 1000 boxes is the
  clear next candidate (bit-mask/parallel-row formulation).
- Kernels must link with `-nostdlib++` and the skeleton must be relinked with
  `relink_hexagon_skel_static_libcxx.sh` (see `docs/tvm-hexagon-conv-transpose-handoff.md`).

## Hexagon DSP vs. Adreno GPU code generation

`bench_tvm_adreno_maskrcnn_ops.py` runs the same operator shapes through TVM's OpenCL code
generator (`opencl -device=adreno`) on the phone's Adreno GPU. Operators are written as plain
TE and scheduled automatically by `tvm.dlight` (Matmul/GEMV/Reduction/Fallback rules); there
are no hand-written GPU schedules. Kernels run over RPC and are checked against a host-CPU
TVM build of the same compute. This needs a TVM Android runtime with OpenCL, which the
Hexagon build does not include:

```bash
cmake $TVM -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DUSE_LLVM=OFF -DUSE_OPENCL=ON \
  -DUSE_RPC=ON -DUSE_CPP_RPC=ON -DCMAKE_CXX_FLAGS="-isystem <dir containing CL/opencl.h>"
ninja tvm_runtime tvm_rpc   # push both to the phone, then:
./tvm_rpc server --host=0.0.0.0 --port=9190 --port-end=9199 --key=adreno
adb forward tcp:9190 tcp:9190
TVM_NDK_CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++ \
  python scripts/android/bench_tvm_adreno_maskrcnn_ops.py
```

(`3rdparty/OpenCL-Headers` is an empty submodule in the TVM release tarball, hence the header
directory; the runtime `dlopen`s the phone's `libOpenCL.so`.) Medians of single invocations,
fp32 on both sides, device buffers preallocated, transfers excluded, ROI batch 8:

| Operator | Hexagon DSP | Adreno GPU (dlight) |
|---|---:|---:|
| Conv 3x3 + bias + ReLU `[1,64,56,56]` | 178.5 ms | 20.3 ms |
| Conv 1x1 `[1,64,56,56]` | 12.2 ms | 3.4 ms |
| Conv 3x3 `[1,256,14,14]` | 158.9 ms | 29.0 ms |
| Conv 3x3 `[8,256,14,14]` (mask head) | 1337.7 ms | 169.9 ms |
| MaxPool `[1,64,112,112]` | 6.5 ms | 0.20 ms |
| FPN resize `[1,256,14,14]`->28 | 7.2 ms | 0.05 ms |
| RoIAlign 56x56 -> 7x7 x8 | 27.5 ms | 0.99 ms |
| ConvTranspose `[8,256,14,14]`, generic TOPI | 5802 ms | 192.8 ms |
| ConvTranspose, stride-2 parity formulation | 12.8 ms (hand-written HVX; ~23 ms with layout copies) | 40.8 ms |
| Quantize + dequantize `[1,3,224,224]` | 3.64 ms | 0.08 ms |
| Add / Relu / Add+Relu `[1,256,56,56]` | 0.51 / 0.29 / 0.46 ms | 0.31 / 0.19 / 0.32 ms |
| Add / Relu / Add+Relu `[1,2048,7,7]` | 0.07 / 0.01 / 0.07 ms | 0.04 / 0.03 / 0.04 ms |
| MatMul `[8,12544]x[12544,1024]` + ReLU | 78.5 ms generic, 12.6 ms HVX | 34.3 ms |
| MatMul `[8,1024]x[1024,1024]` + ReLU | 6.2 ms generic, 1.19 ms HVX | 2.27 ms |
| MatMul `[8,1024]x[1024,324]` | 8.4 ms generic, 0.61 ms HVX | 5.54 ms |
| MatMul `[8,1024]x[1024,81]` | 2.5 ms generic, 0.11 ms HVX | 5.33 ms |
| Softmax `[8,81]` | 0.05 ms | 0.01 ms |
| Sigmoid `[8,81,28,28]` | 10.0 ms TOPI, 1.0 ms polynomial exp | 0.11 ms |
| Box decode / level mapper / score filter | 0.11 / 0.05 / 0.05 ms | 0.01 / 0.007 / 0.01 ms |
| Gather / Concat / Split / Transpose / Flatten | 0.02 / 0.12 / 0.01 / 0.003 / 0.20 ms | 0.007 / 0.013 / 0.02 / 0.005 / 0.03 ms |
| ReduceMin `[1000]` | 0.002 ms | 0.20 ms |
| TopK, NMS, NonZero, ScatterElements | 0.93, 8.1, 0.05, 0.44 ms | not attempted (data-dependent) |

How to read this:

- **Like-for-like schedules favour the GPU by 5-20x** for convolution, pooling, resize and
  RoIAlign in fp32; the DSP has no native HVX fp32 multiply-add on this generation (fp32 goes
  through qf32 conversions), while the Adreno has fp32 ALUs and far higher memory bandwidth.
  The DSP's strength is int8 `vrmpy`: the NCHWc int8 3x3 probe above runs the same
  `[1,64,56,56]` convolution in 1.2 ms, a different dtype but the relevant comparison for a
  quantized model.
- **Hand-tuned DSP kernels beat auto-scheduled GPU ones where effort was spent**: the HVX
  ConvTranspose (12.8 vs 40.8 ms) and the small-batch box-head MatMuls (up to 9x). A tuned
  GPU kernel for these was not written, so this is an effort comparison as much as a
  hardware one.
- Tiny operators (<0.1 ms) are within launch/timer noise on both sides; the GPU's 0.2 ms
  ReduceMin is a single-workgroup reduction, and NMS/TopK/NonZero-style sequential ops
  belong on the CPU or DSP scalar side either way.
- Timings are per-kernel with warm buffers, not end-to-end; power and the cost of moving
  tensors between DSP and GPU are not measured. Hexagon kernels were compiled for `v73`
  while the SoC reports V69.

## fp16 ConvTranspose on Hexagon

`bench_tvm_hexagon_conv_transpose_fp16.py` repeats the mask-head ConvTranspose (`[8,256,14,14]`,
2x2 kernel, stride 2) with fp16 activations, weights and output (NHWC, packed weights) on the
connected phone. Errors are the max absolute error divided by the output's max magnitude,
against an fp32 reference computed from the fp16-rounded operands; fp16 output rounding alone
gives about 5e-4. Medians of three single invocations, layout copies and weight packing excluded.

| Kernel | Time | Error | Notes |
|---|---:|---:|---|
| fp32, LLVM-generated (best) | 34.6 ms | 1e-6 | baseline from the fp32 section |
| fp32, hand-written HVX qf32 | 12.8 ms | 1e-6 | `vmpy.qf32.sf` + `vadd.qf32` |
| fp16, LLVM-generated, fp16 accumulate | 12.4 ms | 1.2e-2 | 64 lanes but qf16 conversions per op |
| fp16, LLVM-generated, fp32 accumulate | 24.7 ms | 7e-4 | |
| fp16, HVX `vmpy.qf16.hf` + `vadd.qf16` | 4.04 ms | 1.2e-2 | 64 MACs per vector op; fp16 accumulation over K=256 is too coarse |
| fp16, HVX qf16 chunks of 8, widened into qf32 | **4.15 ms** | 2.0e-3 | `vmpy.qf32.qf16` by qf16 1.0 every 8 input channels |
| fp16, HVX `vmpy.qf32.hf` widening, fp32 accumulate | 5.56 ms | 8.8e-4 | at the fp16-output rounding floor |

So fp16 storage roughly triples the hand-written kernel's speed (12.8 to 4.2-5.6 ms) and is
about 8x faster than the LLVM-generated fp32 path, with `hf` operands halving weight and
activation traffic. Notes:

- Accumulating 256 products in qf16 (the same precision class as LLVM's fp16 path) costs about
  1.2% error; chunked qf16 with widening recovers most of it at almost no speed cost, and
  the widening `vmpy.qf32.hf` form reaches the fp16 rounding floor at ~30% more time.
- LLVM 19 has no `llvm.hexagon.V6.vmpy.rt.hf` intrinsic (the SDK headers do); the kernels use
  `lvsplath` + `vmpy.qf16.hf` instead.
- **Silent wrong results under register pressure.** Configurations with many live
  accumulators (for example `--vectors 4 --pixel-blocks 2 --unrolls 4` for `qf32w`, or any
  7-pixel block) return garbage (error >= 1) while the same tile shape with `--unrolls 1` or
  `2` is correct, so the failure follows spilling/hoisted loads and was not root-caused. The
  script marks any error above 5% as `** WRONG RESULT **`; only verified configurations are
  quoted above (`--vectors 2 --pixel-blocks 2`).
- The V69 phone has HVX `qf16`/`hf` instructions but no IEEE fp16 vector arithmetic, so fp16
  kernels still convert through qf formats; fp16 activations would need matching fp16
  layout-copy kernels to keep the whole path fp16 (not done).

