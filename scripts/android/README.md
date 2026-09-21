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
same device, using the median of five single-invocation timings with buffers
preallocated; they are per-kernel timings, not end-to-end model latency. The uint8 quantized
values are checked exactly; dequantized floats allow a 1e-5 absolute tolerance
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
