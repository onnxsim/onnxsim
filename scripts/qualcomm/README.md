# Qualcomm QNN integration check

Verifies that `onnxsim`'s output still works with the **Qualcomm AI Runtime
(QNN)** software stack — the runtime behind SNPE / the Qualcomm Neural Processing
SDK and QAIRT. The goal is to catch the failure mode the unit tests and the
large-model regression don't: a simplification that produces a graph the
Qualcomm toolchain can no longer **compile**, or that **changes the result** on a
Qualcomm target.

It uses the pip-installable [`onnxruntime-qnn`](https://pypi.org/project/onnxruntime-qnn/)
wheel (the QNN execution provider for ONNX Runtime). On an ordinary **x86-64
Linux CPU runner with no Snapdragon device**, the bundled **HTP** backend still
performs the full *offline graph compile* (`libHtpPrepare.so` — the same
preparation a Qualcomm converter runs) and then executes the compiled graph in
the **x86 host emulator**. So the whole check runs on a free GitHub-hosted runner
with nothing but `pip install onnxruntime-qnn`.

Sibling checks for other vendor execution providers follow the same
pattern: [`scripts/apple`](../apple) (Core ML), [`scripts/intel`](../intel)
(OpenVINO), and [`scripts/amd`](../amd) (MIGraphX -- needs a self-hosted ROCm
runner, since unlike the other three it has no CPU fallback or emulator, so
its workflow is dormant until one is provisioned). NVIDIA (CUDA/TensorRT)
needs real GPU hardware too and isn't covered by a harness yet.
[`scripts/axera`](../axera) (Pulsar2/AXCL NPU) is the odd one out: Pulsar2 has
no pip package or execution provider at all, so that check is a static
op-support heuristic rather than a real compiler invocation -- see its README
for why.

## What it checks

For each model the harness runs **original vs. simplified through the same QNN
backend**, so backend quirks cancel and only an onnxsim-introduced change can
fail the run:

1. `simplify` the model with onnxsim.
2. Compile + run the **original** graph on the QNN EP.
   If that already fails, the backend just doesn't support the graph → reported
   as `unsupported`, **not** a failure.
3. Compile + run the **simplified** graph on the QNN EP.
   If the original compiled but the simplified doesn't → `qnn_regression`
   (a failure): simplification broke QNN compatibility.
4. Compare the two QNN outputs. Divergence beyond tolerance → `qnn_regression`:
   simplification changed the on-device result.
5. Record the ONNX Runtime CPU-reference diff and the QNN **coverage** (does the
   whole graph map onto QNN, or do some nodes fall back to ORT's CPU provider)
   as information.

Partial coverage and `unsupported` are reported, never failed — plenty of valid
graphs are not 100% HTP-mappable, and that is a backend property, not an onnxsim
bug.

## Files

| file | purpose |
| --- | --- |
| `qnn_backend.py` | wraps the QNN EP: registers the plugin, builds/runs a model on QNN (offline HTP compile + x86 emulation) and on the ORT CPU reference, measures coverage. Input synthesis/comparison come from `scripts/common/ep_numerics.py`. Degrades gracefully (`QNN_AVAILABLE`) when the EP is absent. |
| `models.py` | alias for `scripts/common/synthetic_models.py`, a small, network-free suite of synthetic graphs covering common layer patterns (conv/BN/relu, foldable shape→reshape, MLP, cancelling transposes, swish), each carrying the redundancy onnxsim is meant to remove. Shared with the Apple Core ML (`scripts/apple`) and Intel OpenVINO (`scripts/intel`) harnesses so the suite isn't duplicated per vendor. |
| `worker.py` | runs the check for one model in an isolated subprocess (the HTP compiler can abort at the C++ level), printing one `__RESULT__<json>` line. |
| `run_qnn_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Requires an x86-64 Linux host (the `onnxruntime-qnn` HTP backend platform).

```bash
pip install onnxruntime-qnn      # brings its own onnxruntime
pip install .                    # or install an onnxsim wheel

python scripts/qualcomm/run_qnn_compat.py --output qnn-compat.csv
python scripts/qualcomm/run_qnn_compat.py \
  --model /models/mobilenetv2-qdq.onnx \
  --model /models/resnet50-qdq.onnx \
  --output qnn-models.csv
```

`--model` is repeatable and runs on-disk models through the same
original-versus-simplified comparison as the built-in suite. For the HTP path,
prefer fixed-shape, quantized/QDQ models: the QNN EP documents that HTP models
must be quantized and that dynamic shapes are unsupported. A model that QNN
cannot compile before simplification is reported as `unsupported`, while a
model whose simplified graph regresses is a failure.

The in-tree smoke test `tests/test_qnn_compat.py` reuses this harness and is
skipped automatically when `onnxruntime-qnn` isn't installed.

## Fidelity tiers (what this does and doesn't cover)

This check runs the QNN **HTP** backend via **offline compile + x86 emulation**.
That validates *graph compatibility* and *functional numerics* with no device.
Two things it deliberately does not do:

- **True CPU reference backend.** The pip wheel does not ship `libQnnCpu.so`
  (the QNN CPU reference backend); it comes with the full QAIRT SDK. To use it
  instead of HTP emulation, install the SDK and set
  `QNN_BACKEND_PATH=/path/to/libQnnCpu.so` — the harness picks it up.
- **Real NPU/HTP hardware numerics.** Emulation is functional, not
  bit-identical to silicon, and does not quantize. For on-device validation use
  a self-hosted Snapdragon runner or [Qualcomm AI Hub](https://aihub.qualcomm.com/)
  (`qai-hub`), which compiles/profiles/runs on cloud Snapdragon devices via an
  API token.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models (e.g. the Hugging Face
[`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo) set used by the large-model
regression) can be layered on by passing an on-disk path as `worker.py`'s second
argument; a scheduled job can iterate those the way `scripts/regression` does.
