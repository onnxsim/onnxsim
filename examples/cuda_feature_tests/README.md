# CUDA feature tests notebook

A notebook exercising onnxsim's CUDA-related features against a real NVIDIA
GPU -- meant to be run **by hand**, on demand, not wired into an unattended
nightly CI trigger. (Google Colab's terms of service treat scheduled/
background automation as out of scope for what Colab is for, so it is not a
suitable nightly-CI backend; see the discussion that led to this notebook.
For unattended GPU CI, a self-hosted GitHub Actions runner on your own or a
cloud GPU box is the usual fit.)

## What it covers

- `onnxsim.simplify(..., providers=[...])` / `onnxsim.backend.run_model` /
  the CLI's `--providers` and `--cuda` flags -- constant folding and
  correctness checking on the GPU, compared against CPU for parity.
- The `(name, options)` provider tuple form, e.g. pinning `device_id`.
- The DLPack zero-copy path for CUDA `torch.Tensor` inputs
  (`onnxsim.backend.as_ort_value` / `Runner.run_with_ort_values`).
- Provider validation: requesting a provider the installed onnxruntime does
  not offer raises `ValueError` instead of silently falling back to CPU.
- `providers` threaded through `onnxsim.accuracy.measure_accuracy_drop`.

## Running it

Open [`cuda_feature_tests.ipynb`](cuda_feature_tests.ipynb) in Colab (use the
badge at the top of the notebook, or upload it manually) and select a GPU
runtime: `Runtime > Change runtime type > T4 GPU` (or any NVIDIA GPU). Then
run the cells top to bottom. By default the install cell installs a
pre-built wheel of the latest development build from
[TestPyPI](https://test.pypi.org/project/onnxsim/) (published from `master`
daily by CI -- no C++ toolchain needed) plus the GPU build of onnxruntime
(`onnxruntime-gpu`). Set `BUILD_FROM_SOURCE = True` in that cell to instead
build a specific `BRANCH`/tag from source, e.g. to exercise unreleased
changes.

It works the same way outside Colab: any Jupyter environment with an NVIDIA
GPU and driver, `pip install onnxsim[onnxruntime] onnxruntime-gpu`, and the
CUDA-specific cells run as-is (skip the install cell and just install
onnxsim however you normally would).

Each test prints `[PASS]`/`[FAIL]` as it runs and raises immediately on the
first failure, so a red cell points straight at what broke; the final cell
re-asserts that every test passed.
