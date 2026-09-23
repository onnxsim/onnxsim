# tinygrad AX650 backend skeleton (template + patch)

First code for `docs/axera-tinygrad-emitter-plan.md` (#1834): architecture B,
wired over the emitters this repository has already validated, with the
tinygrad-facing classes built on the user's tinygrad fork. No Pulsar2 builds,
no device runs, no model exports. Module:
`scripts/axera/tinygrad_ax_backend.py`; tests:
`tests/test_axera_tinygrad_ax_backend.py`.

## tinygrad fork

- Fork: `onnxsim/tinygrad`, master at
  `ba00cbb3c6b8a6bcbee04c22cb4a0b0b9cee216e` (2026-09-22, includes the merged HVX
  codegen, onnxsim/tinygrad#2). Recorded as `TINYGRAD_FORK_SHA`.
- Install: `uv run --no-project --with
  "tinygrad@https://github.com/onnxsim/tinygrad/archive/ba00cbb3c6b8a6bcbee04c22cb4a0b0b9cee216e.tar.gz"`.
- No change to the fork was needed. The skeleton subclasses the fork's
  `tinygrad.device.Compiler`, `Allocator` and `Program` from outside. Registering
  an `AX` device in tinygrad's `Device` registry (a `tinygrad/runtime/ops_ax.py`)
  is the hook that will need a fork branch, once the AXCL transport exists
  (roadmap milestone 1, which needs the device).
- tinygrad is imported lazily (`tinygrad_classes()`), so everything else in the
  module, and 23 of the 24 tests, run without it.

## What the skeleton implements

| Plan interface (section 6) | Here | Backed by |
| --- | --- | --- |
| `TemplateKey` | frozen dataclass: op, data-input shapes, template-selecting attrs (`perm`, `indices`, `w`/`strides`/`pads`), dtypes, calibration class, toolchain; JSON round-trip | -- |
| `TemplateCache.get_or_build` | `lookup` resolves to committed fixtures; a miss raises `NotImplementedError` (a miss needs a Pulsar2 build, out of scope) | fixture registries of the modules below |
| `Edit.validate` / `apply` | `TemplateOnly` | Transpose, `transpose_real_shapes.py` (#1758) |
| | `GatherIndexEdit` | last-axis Gather, `memory_emit.py` (19/19 step shapes incl. the chunked stem) |
| | `ElementwiseScaleEdit` | Relu/Sqrt scale retarget, `elementwise_scale_emit.py` (#1840); same-shape Add/Sub/Mul/Div, `binary_op_scale_emit.py` (`docs/axera-binary-op-scale-emit.md`) |
| | `ConvWeightEdit` | frozen-weight refresh, `conv_weight_learn.py` (#1769) and `conv_bias_requant.py` (#1771) |
| `EditSet` | validates every edit against the resolved template before applying any | -- |
| tile-table cross-check | `predicted_npu_params` | `dma_tile_predict.py`, `add_tile_predict.py`, `elementwise_two_input_tile_predict.py` |
| graph-level hook | `extract_step_ops` / `plan_node` / `coverage_report` on a real ONNX graph; `build_request` is the JSON request a JIT-boundary hook would emit for one fused subgraph | -- |
| `AXCompiler` | tinygrad `Compiler`; `compile(src)` takes a template request and returns patched `.axmodel` bytes; `compile_cached` works through tinygrad's own path | the above |
| `AXAllocator` / `AXProgram` | stubs that raise `NotImplementedError` | AXCL transport (milestone 1) |

Every op, shape, calibration class, dtype, toolchain or edit that no committed
fixture validates raises `ValueError`. `ConvWeightEdit` covers only the two
Conv shapes whose full `npu_params` pipeline was checked against a native
held-out build (stage-1 3x3 64/64; 1x1 64->128 stride 2). It edits `npu_params`
only; the MCode stays the template's, so the activation calibration must be the
template's own. `ElementwiseScaleEdit` keeps the template's zero points, which
the key's calibration class selects.

## Coverage of the ResNet18 training step

`coverage_report` on the committed op records of the real step
(`scripts/axera/fixtures/tinygrad_ax_backend/resnet18_step_ops.json.gz`,
extracted from `/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, 1,104
nodes):

| Status | Nodes | Which |
| --- | --- | --- |
| covered | 82 | Gather 41 (`GatherIndexEdit`), Transpose 41 (`TemplateOnly`) |
| conditional | 324 | Relu 17 and Sqrt 39 (`ElementwiseScaleEdit`, only if the node's zero points are one of the template classes `x0,y0` / `x128,y128`); same-shape Add 101, Mul 63, Div 44, Sub 42 (`ElementwiseScaleEdit`, classes `x0,y0,z0` / `x128,y128,z128`, Div `x128,y128,z0`); 18 bias-flatten Reshapes (fuse into a neighbour; `reshape_emit.py` covers Relu-neighbour pairs only) |
| refused | 698 | Mul 334, Reshape 152, ReduceSum 44, Add 43, MatMul 41, Conv 20, Cast 19, Greater 18, Div 8, Sub 4, Sqrt 3, 12 others |

Refusal reasons are reported per node. The notable ones:

- **Conv (20).** 5 Convs (the four stage-1 3x3 64/64 and the 1x1 64->128
  downsample) match a template shape but are refused:
  their weights are graph inputs, i.e. training state, so a frozen-weight edit
  is not the training path. The other 15 have no validated template.
- **Add/Sub/Mul/Div (389 of 639 still refused).** 254 have a constant operand
  and 132 a broadcast operand, which compile to different programs with no
  templates yet; 3 sit at shapes not built (`[1,1]`, `[1024,9,3136]`,
  `[16,64,112,112]`). The 250 same-shape nodes are served by
  `binary_op_scale_emit.py`, which re-encodes the decompressed short units
  (#1850); see `docs/axera-binary-op-scale-emit.md`.
- **Sqrt (3).** `[512,512,3,3]` failed its held-out check in #1840.

This is standalone coverage. Composition rewrites MCode (#1732, #1763, #1783,
#1802), so none of these edits reproduces the compiled whole step; per the plan,
that needs graph-level templates plus a recalibration edit, which the
short-unit encoding also blocks (#1836).

## Tests

`tests/test_axera_tinygrad_ax_backend.py`, 24 tests, no device:

- cache refusals for an unknown op, index count, shape, zero-point class and
  dtype; a cache miss raises `NotImplementedError` instead of building;
- `GatherIndexEdit` writes the indices and leaves MCode and the table tail
  unchanged, and refuses bad index vectors;
- `TemplateOnly` reproduces the Transpose template byte for byte and is refused
  for templates that need an edit;
- `ElementwiseScaleEdit` reproduces held-out native Relu and Sqrt builds (MCode
  outside the 301-325 noise window, and `npu_params`) byte for byte;
- the Relu tile predictor equals the template's `npu_params`;
- `ConvWeightEdit` reproduces the native held-out weight-code region for both
  Conv shapes, with the requant block within the tolerance #1769/#1771 document;
- `coverage_report` totals on the real step, and the trainable-Conv refusal;
- `AXCompiler.compile_cached` through the pinned fork matches
  `compile_request` (skipped when tinygrad is not installed).
