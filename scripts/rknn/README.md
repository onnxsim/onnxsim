# Rockchip RKNN integration check

Verifies that `onnxsim`'s output still converts and runs through
[`rknn-toolkit2`](https://pypi.org/project/rknn-toolkit2/), Rockchip's real
ONNX -> RKNN converter for the RK35xx/RV1106 NPU line -- the same toolchain
[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) (already
listed as a downstream onnxsim user in the top-level README) runs onnxsim
ahead of, before feeding its export scripts' ONNX output into `rknn-toolkit2`.

## Real vs. static: where this sits among the sibling checks

Unlike [`scripts/axera`](../axera) (Pulsar2/AXCL has no pip package, so that
check is a static op-support heuristic), `rknn-toolkit2` **is** a real,
pip-installable x86-64 Linux wheel (`pip install rknn-toolkit2`, cp310/cp311/
cp312 on PyPI) -- this harness runs the actual Rockchip converter and its
built-in **PC simulator**, the same way
[`scripts/qualcomm`](../qualcomm) (QNN) and [`scripts/intel`](../intel)
(OpenVINO) run the real backend via a pip EP wheel. The difference from those
two: RKNN is not an ONNX Runtime execution provider at all -- there is no
`RknnExecutionProvider` to register. `rknn-toolkit2` is Rockchip's own
standalone package built around `rknn.api.RKNN`: `load_onnx()` parses the
graph, `build()` compiles it into Rockchip's internal IR, and
`init_runtime(target=None)` runs that IR on the host CPU via Rockchip's PC
simulator -- no RK35xx/RV1106 device attached, no ADB, no NPU-transfer.
Real on-device inference is a separate, later step (`rknn-toolkit-lite2` on
the board itself, or `init_runtime(target="rk3588", ...)` over a connected
device) that this harness does not exercise -- see "Fidelity tiers" below.

## Two real, verified findings

Both reproduced directly against `rknn-toolkit2==2.3.2` (the latest release
on PyPI at the time of writing) on a plain x86-64 Linux host with no RK
device -- no Docker, no hardware:

1. **`onnx.mapping` no longer exists.** `rknn-toolkit2` 2.3.2 declares
   `onnx>=1.16.1` but its C-extension code (`rknn/api/base_utils.py`) still
   does `import onnx.mapping` / reads
   `onnx.mapping.TENSOR_TYPE_TO_NP_TYPE`/`NP_TYPE_TO_TENSOR_TYPE`, a
   submodule the `onnx` pip package no longer ships (verified against onnx
   1.22.0, installed alongside a stock `pip install rknn-toolkit2`). A plain
   `rknn.load_onnx()` call fails with `AttributeError: module 'onnx' has no
   attribute 'mapping'` before ever reaching onnxsim's own code -- not an
   onnxsim bug, but it does mean nothing in this ecosystem can hand
   `rknn-toolkit2` an ONNX model at all on a modern `onnx` install without a
   workaround. `rknn_backend.py`'s `_ensure_onnx_mapping_shim()` patches in a
   tiny replacement (built from the still-current
   `onnx.helper.tensor_dtype_to_np_dtype` table) before `rknn.api` is
   imported, so this harness needs no old, pinned `onnx` version.
2. **`inference()` defaults to NHWC regardless of the ONNX graph's own
   layout.** Every onnx-native CV model onnxsim ever sees is NCHW; feeding a
   plain NCHW ndarray to `rknn.inference()` without `data_format="nchw"`
   raises (`ValueError: The input(ndarray) shape (1, 3, 16, 16) is wrong,
   expect 'nhwc' like (1, 16, 16, 3)!`) rather than silently misinterpreting
   it -- but only because the shapes happened to be distinguishable in this
   probe; a model with equal spatial and channel extents would not raise and
   would just be run transposed. `rknn_backend.run()` always passes
   `data_format="nchw"` explicitly for rank-4 inputs.

## What it checks

Framed the same way as the QNN check -- original vs. simplified through the
**same** RKNN PC-simulator build, so a fixed simulator limitation (an
unsupported op, or the simulator's own float-vs-CPU-reference numeric slack,
see "Fidelity tiers" below) cancels out and only an onnxsim-introduced change
fails the check:

1. `simplify` the model with onnxsim.
2. Convert (`load_onnx` + `build(do_quantization=False)`) and run
   (`init_runtime(target=None)` + `inference()`) the **original** graph.
   If that already fails, `rknn-toolkit2` just doesn't support the graph ->
   reported as `unsupported`, **not** a failure.
3. Convert and run the **simplified** graph the same way.
   If the original converted/ran but the simplified doesn't ->
   `rknn_regression` (a failure): simplification broke RKNN compatibility.
4. Compare the two RKNN outputs. Divergence beyond tolerance ->
   `rknn_regression`: simplification changed the simulator result.
5. Record the ONNX Runtime CPU-reference diff as information only (see
   "Fidelity tiers" -- the PC simulator is not expected to match it tightly).

`do_quantization=False` throughout: this check is about graph-compile and
float-numerics fidelity, not INT8 quantization accuracy, which is a separate,
calibration-dataset-dependent concern already covered elsewhere in this repo
(`onnxsim`'s own calibration/quantization tests).

## Files

| file | purpose |
| --- | --- |
| `rknn_backend.py` | wraps `rknn.api.RKNN`: the `onnx.mapping` compat shim, converts + runs a model through the PC simulator, and the ORT CPU reference. Input synthesis/comparison come from `scripts/common/ep_numerics.py`. Degrades gracefully (`RKNN_AVAILABLE`) when `rknn-toolkit2` is absent. |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the same small, network-free suite of synthetic graphs shared with the Apple/Intel/Qualcomm/Axera harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess (the converter can abort at the C-extension level), printing one `__RESULT__<json>` line. |
| `run_rknn_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Requires an x86-64 Linux host (the `rknn-toolkit2` wheel's supported
platform).

```bash
pip install rknn-toolkit2       # brings its own onnxruntime/numpy/torch
pip install .                   # or install an onnxsim wheel

python scripts/rknn/run_rknn_compat.py --output rknn-compat.csv
```

The in-tree smoke test `tests/test_rknn_compat.py` reuses this harness and is
skipped automatically when `rknn-toolkit2` isn't installed.

## Fidelity tiers (what this does and doesn't cover)

This check runs `rknn-toolkit2`'s real converter and its **PC simulator**.
That validates *graph convertibility* and gives a *functional* numeric
result with no device. Two things it deliberately does not do:

- **Bit-exact NPU numerics.** Rockchip documents the PC simulator as
  functional, not bit-exact hardware emulation. Verified directly: even with
  `do_quantization=False`, a small Conv+Bias+Relu graph's PC-simulator output
  differs from the ONNX Runtime CPU reference by a small but nonzero amount
  (~4e-3 max-abs on values of order 1-8). That is why this harness compares
  original-vs-simplified through the *same* simulator build rather than
  asserting tight agreement with the CPU reference.
- **Real RK35xx/RV1106 device numerics, and INT8 quantization accuracy.**
  For on-device validation, `init_runtime(target=..., device_id=...)` needs a
  connected board (over ADB/NPU-transfer) or `rknn-toolkit-lite2` running
  directly on one -- neither is provisioned by this repository. Quantized
  (`do_quantization=True`) accuracy also needs a representative calibration
  dataset, which is out of scope for this graph-compatibility check.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models (e.g. the Hugging Face
[`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo) set used by the
large-model regression) can be layered on by passing an on-disk path as
`worker.py`'s second argument; a scheduled job can iterate those the way
`scripts/regression` does.
