# Which ONNX ops OpenVINO places on the Intel NPU DPU

Census of standalone op compiles on Intel NPU 5020 (OpenVINO 2026.4). Each op was compiled alone at one typical shape with
`compile_tool -d NPU -ip FP32 -op FP32`, then its ELF sections were checked:

- **DPU-only**: `task.dpu.*` present, no `shave.*` or `task.shave.*` sections.
- **mixed**: both DPU and SHAVE sections. The split of work is not measured, so a mixed op is not a DPU op.
- **SHAVE-only**: no DPU sections.
- **no-compute**: layout-only (Transpose here, Reshape as a view).

| Op | Class | Shape used |
|---|---|---|
| Add, Sub, MaxPool, AveragePool, GlobalAveragePool, ReduceMean, Resize (nearest), ConvTranspose | DPU-only | 1x64x28x28 (typical) |
| Relu, LeakyRelu, Clip, Softmax, MatMul | mixed | 1x64x28x28, 1x1000, 1x512 |
| Sigmoid, Tanh, Mul, Concat, Pad | SHAVE-only | 1x64x28x28 |
| Transpose, Reshape | no-compute | 1x64x28x28 |

Caveats:
- One shape per op. A SHAVE-only op may run on the DPU at other shapes or with fusion; this was not tested.
- Mixed ops are not counted as DPU coverage.
- Numerics: the five DPU-only ops in `test_dpu_single_op.py` match numpy references to a relative error of 2.5e-4 to
  4.1e-4 at the shapes listed in that file.

Reproduce with `dpu_single_op.run(op, inputs, out_shape, attrs, consts)`; see `test_dpu_single_op.py` for examples.
Requires `OPENVINO_COMPILE_TOOL` (or `compile_tool` on PATH) and the level-zero loader on `LD_LIBRARY_PATH`.
