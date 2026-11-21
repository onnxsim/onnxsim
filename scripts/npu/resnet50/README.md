# ResNet50 through our NPU path (lowering executor)

`mixed_resnet50.py` walks the ONNX graph of torchvision's ResNet50 (opset 17). Every node runs on the Intel NPU
through our own kernels:

- Conv: tiled MLIR -> SHAVE kernels, see `tools/npu-isa/conv/` in nfcxx (the module `conv_npu`).
- Relu, Add, MaxPool, GlobalAveragePool, Gemm: SHAVE elementwise and reduction kernels, tiled to the 200704-element
  launch size. MaxPool is a max over the shifted windows; GlobalAveragePool is a sum tree plus a host divide by H*W.
- Flatten: a view (a reshape of the same data).

`export_resnet50.py` exports torchvision's ResNet50 (ImageNet weights) to `~/models/resnet50_opset17.onnx`
and checks it against torch.

## Running

```sh
CONV_NPU_DIR=/path/to/nfcxx/tools/npu-isa/conv \
LD_LIBRARY_PATH=$HOME/ze-loader/lib \
python mixed_resnet50.py Flatten,Conv,Relu,Add,MaxPool,GlobalAveragePool,Gemm
```

The script prints per-op node counts and the maximum absolute difference against onnxruntime. On the test machine it
gives top-1 904 (matching onnxruntime) with a maximum difference of 0.018 against a reference maximum of 7.5, in about
a minute per image.

## Conv and Gemm on the DPU (`mixed_resnet50_dpu.py`)

`mixed_resnet50_dpu.py` runs the same walk with Conv and Gemm on the NPU's DPU (the fixed-function convolution engine)
instead of SHAVE. `dpu_layers.py` compiles each layer with its real weights through OpenVINO's `compile_tool` (one blob
per layer, about 0.1 s to compile), imports the blob in-process, and runs one launch per layer. Bias is added on the host.
Compiled models are cached in memory by weight hash, so each layer compiles once per process.

```sh
OPENVINO_COMPILE_TOOL=/path/to/openvino/bin/intel64/Release/compile_tool \
LD_LIBRARY_PATH=$HOME/ze-loader/lib \
python mixed_resnet50_dpu.py
```

On the test machine (Intel NPU 5020, OpenVINO 2026.4) all 23 ResNet50 conv shapes and the classifier Gemm match a float32
numpy reference with a relative error of at most 5e-4, and the full network runs in 54 launches (one per conv or Gemm).

An earlier version split convs into chunks and patched a compiler-made template's constant buffer with the real weights.
That worked only for some shapes, and the buffer layout turned out to depend on the template's shape and contents, so it
was dropped in favour of compiling each layer.

## Dependencies on that machine

- `conv_npu` (from nfcxx) and its kernel build tools (MoviTools via wine, Intel `vpux-opt`).
- The Relu, Add, MaxPool, GlobalAveragePool and Gemm blobs under `~/npu-tinygrad/blobs` and `~/add-build-*`,
  `~/max-build-*`, `~/mul-build-*`: built with the same pipeline (carrier compile, MLIR template, `build.sh`).
  Those paths are hardcoded; a clean rebuild script is not part of this PR yet.
- Level-zero loader on `LD_LIBRARY_PATH`; OpenVINO's Python package (in-process launches).

## Known limits

- Speed: about 115 s per image without the in-process change, about 57 s with it (SHAVE path). Nearly all of the
  remaining time is inside our scalar SHAVE conv loops. Vectorizing them is the next step.
- DPU path: 8.3 s for one image, including compiling the 54 layers (not measured separately for later images).
  Top-1 904 matches onnxruntime. The precompiled OpenVINO blob runs the same network in about 4 ms.
- Numerics: fp16 activations; f32 accumulation inside the conv kernel. Subnormal inputs flush to zero on the device.
- Coverage is 122 of 122 nodes on the NPU for ResNet50; other graphs will need more op kernels.
