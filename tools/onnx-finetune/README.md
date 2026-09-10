# onnx-finetune

A small C++ CLI that fine-tunes an ONNX model in place, using ONNX Runtime's
on-device training API, and exports a normal inference-ready `.onnx` file.
No PyTorch, no re-export from a training framework -- it trains directly
against the ONNX graph.

## How it fits together

Fine-tuning a bare `.onnx` file with ONNX Runtime is a two-step workflow,
split across two different tools because of where each capability lives:

1. **`scripts/generate_artifacts.py`** (Python, one-time, offline) -- builds
   the gradient graph for your model and writes four files: `training_model.onnx`,
   `eval_model.onnx`, `optimizer_model.onnx`, `checkpoint`. This step needs
   Python because gradient-graph construction (`onnxruntime.training.artifacts`)
   is only exposed there -- there is no C/C++ API for it.
2. **`onnx-finetune`** (this C++ binary, runs training) -- loads those four
   files and runs the actual train loop (forward, backward, optimizer step)
   purely through ONNX Runtime's native C++ API, with zero Python dependency.
   It exports the trained result back to a plain `.onnx` file at the end.

This split is deliberate, not a limitation of this tool specifically: it's
how ONNX Runtime's on-device training is designed to be deployed (mobile/edge
apps generate artifacts once during a build step, then ship only the C++/Java/
Swift runtime, never Python).

## Prerequisite: a training-enabled ONNX Runtime build

For `scripts/generate_artifacts.py` (and, if you'd rather train from Python
than build the C++ CLI below, `onnxruntime.training.api` too): just
`pip install onnxruntime-training` -- a separate PyPI distribution from
plain `onnxruntime`, published by the ONNX Runtime team with
`onnxruntime.training.artifacts`/`.api` included, no source build needed.
(Pinned at 1.19.2 as of this writing -- there have been no newer releases of
this particular distribution; match that version if you build `onnx-
finetune` itself from source per the following, to keep the artifact
format and IR version expectations aligned between the two.)

`onnx-finetune` itself (the C++ CLI) and its WASM binding are a different
story: neither the `pip install onnxruntime` wheels nor the official
prebuilt release tarballs (the ones under GitHub Releases) include the
training C++ headers/library at all, in any distribution -- for those two,
you need to build ONNX Runtime from source with training enabled, once:

```sh
git clone --branch v1.19.2 https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git submodule update --init --recursive

python3 tools/ci_build/build.py \
  --build_dir build --config Release --parallel \
  --skip_tests --allow_running_as_root \
  --build_shared_lib --enable_training_apis
```

This is a real ~20-40 minute build (protobuf, abseil, onnx, and ONNX Runtime
itself, all from source). `--enable_training_apis` is enough for the C++ CLI
and WASM binding (no Python bindings needed, so no `--enable_pybind
--build_wheel`); if you separately need a from-source *Python* build of
`onnxruntime.training` for some reason `pip install onnxruntime-training`
doesn't cover (a different commit, GPU support, ...), add `--enable_training
--enable_pybind --build_wheel` instead of `--enable_training_apis` -- and if
`pip install .` on that wheel build fails with a `setuptools`/`distutils`
`install_layout` error, that's an unrelated packaging-step bug in older
`setup.py` against newer `setuptools`: the compiled Python module already
exists by that point, at `<build_dir>/Release/build/lib/`, and can be used
directly by pointing `PYTHONPATH` there without needing an installed wheel:

```sh
PYTHONPATH=/path/to/onnxruntime/build/Release/build/lib \
  python3 scripts/generate_artifacts.py ...
```

## Building onnx-finetune

```sh
cmake -B build \
  -DORT_SOURCE_DIR=/path/to/onnxruntime \
  -DORT_BUILD_DIR=/path/to/onnxruntime/build/Release
cmake --build build
```

`onnx-finetune-distill-step-graph` (see "Knowledge distillation (graph_grad)" above) is a
separate target with a much lighter requirement -- `ORT_HOME` pointing at *any* plain
onnxruntime distribution that ships `onnxruntime_cxx_api.h` + `libonnxruntime`, no
`--enable_training_apis` build needed at all:

```sh
# An official prebuilt release works -- no build, no ORT_SOURCE_DIR/ORT_BUILD_DIR.
curl -sSL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar xzf ort.tgz
cmake -B build -DORT_HOME=$PWD/onnxruntime-linux-x64-1.19.2
cmake --build build --target onnx-finetune-distill-step-graph
```

Both `-DORT_SOURCE_DIR`/`-DORT_BUILD_DIR` and `-DORT_HOME` can be given to the same `cmake -B
build` invocation to build both tools at once; each is independently optional (whichever
you omit, that tool's target is simply skipped).

## Usage

```sh
# 1. Generate training artifacts for your model (offline, once).
python3 scripts/generate_artifacts.py model.onnx -o artifacts --loss mse --optimizer adamw

# Freeze everything except a subset (e.g. only fine-tune a head/adapter --
# this is what keeps memory and compute down on larger models):
python3 scripts/generate_artifacts.py model.onnx -o artifacts \
  --freeze-prefix backbone. --loss cross-entropy --optimizer adamw

# 2. Run the actual training loop.
./build/onnx-finetune \
  --artifacts-dir artifacts \
  --train-input train_input.bin --train-target train_target.bin \
  --input-dim 4 --target-dim 1 --num-samples 2048 \
  --batch-size 32 --epochs 20 --lr 0.01 \
  --output-model finetuned.onnx --output-names output
```

`--train-input`/`--train-target` are raw contiguous `float32` binary files
(`num_samples * dim` values each, row-major) -- deliberately the simplest
possible format so the tool has no dataset-library dependency. Convert
whatever you actually have (images, tokenized text, ...) to that layout
before calling this; see `scripts/make_synthetic_data.py` for a minimal
example of writing the format from numpy.

`finetuned.onnx` is a normal inference-only ONNX model afterward -- load it
with any ONNX Runtime build (including a plain `pip install onnxruntime`,
verified in testing), `onnxsim` it, ship it, whatever you'd normally do.

## Try it end-to-end with the included toy example

```sh
python3 scripts/make_toy_model.py -o toy_model.onnx
python3 scripts/make_synthetic_data.py --num-samples 2048
python3 scripts/generate_artifacts.py toy_model.onnx -o artifacts
./build/onnx-finetune \
  --artifacts-dir artifacts \
  --train-input train_input.bin --train-target train_target.bin \
  --input-dim 4 --target-dim 1 --num-samples 2048 \
  --batch-size 32 --epochs 20 --lr 0.01 \
  --output-model finetuned.onnx --output-names output
```

The toy task is learning `y = sum(x)` with a 2-layer MLP; loss should drop
from roughly 5 to under 0.001 over the 20 epochs.

## LoRA (low-rank adaptation)

`--freeze-prefix` above still trains full-size parameter tensors, just fewer
of them. For real LoRA -- freezing every original weight and training only a
pair of small rank-decomposition matrices alongside each targeted layer --
use `scripts/inject_lora.py` to do the graph surgery first, then point
`generate_artifacts.py` at the resulting adapter manifest:

```sh
# 1. Graft a rank-4 adapter onto every eligible MatMul/Gemm weight (or use
#    --target-contains to pick specific layers, e.g. --target-contains q_proj).
python3 scripts/inject_lora.py model.onnx -o model_lora.onnx \
  --rank 4 --params-out lora.json

# 2. Generate training artifacts that train *only* the injected lora_A/lora_B
#    tensors -- everything else (including the original weights) is frozen.
python3 scripts/generate_artifacts.py model_lora.onnx -o artifacts \
  --lora-params-file lora.json --loss cross-entropy --optimizer adamw

# 3. Train as usual.
./build/onnx-finetune --artifacts-dir artifacts ... --output-model finetuned.onnx --output-names output

# 4. Pull just the trained adapter back out (a few KB, not the whole model).
python3 scripts/extract_lora_adapter.py finetuned.onnx --params-file lora.json -o adapter.onnx

# 5. Re-apply that adapter to any fresh copy of the base model, no retraining.
python3 scripts/apply_lora_adapter.py model.onnx --adapter adapter.onnx \
  --params-file lora.json -o finetuned2.onnx
```

`inject_lora.py` adds, per targeted weight `W`, a
`(alpha/rank) * (X @ lora_A @ lora_B)` branch alongside that layer's existing
output (`lora_A` Kaiming-normal, `lora_B` zero, so injecting is a no-op until
trained). Eligible layers are MatMul, Gemm (any `transA`/`transB`), and
1x1/`group=1` Conv (a pointwise conv is just a per-pixel linear layer over
channels) -- see `scripts/lora_surgery.py`'s own docstring for the exact
rule. `adapter.onnx` from step 4 is not a runnable model by itself -- it's a
plain tensor container holding just the `lora_A`/`lora_B` initializers --
meant to be re-merged with `apply_lora_adapter.py`, not loaded directly into
a session. This is a portable, ONNX-native adapter format built on these
scripts, distinct from ONNX Runtime's own `.onnx_adapter` file format
described next.

### Loading an adapter natively at inference time (no Olive needed)

`apply_lora_adapter.py`/`extract_lora_adapter.py` above bake one specific
trained adapter into its own merged copy of the model. ONNX Runtime has a
native alternative for swapping adapters in and out of a *single* loaded
session instead: `onnxruntime.LoraAdapter` / `RunOptions.add_active_adapter`
(ORT >= 1.20), fed by a `.onnx_adapter` file. People usually reach that
format through Microsoft's separate Olive tool (`olive convert-adapters`),
but Olive's own command is a thin wrapper around one public onnxruntime
class (`onnxruntime.AdapterFormat`) -- see `ConvertAdaptersCommand.run` in
Olive's source. `scripts/export_onnx_adapter.py` calls that class directly,
needing only `onnxruntime` itself (no `olive-ai`/`torch`/`peft`):

```sh
# 1. Inject with --adapter-inputs: lora_A/lora_B become optional graph
#    inputs (defaulting to their baked initializer) instead of pure
#    constants, so ONNX Runtime can override them at Run() time.
python3 scripts/inject_lora.py model.onnx -o model_lora.onnx \
  --rank 4 --adapter-inputs --params-out lora.json

# 2-3. Generate artifacts and train exactly as in the plain-LoRA workflow.

# 4. Extract the trained adapter as usual.
python3 scripts/extract_lora_adapter.py finetuned.onnx --params-file lora.json -o adapter.onnx

# 5. Export it to ONNX Runtime's native format.
python3 scripts/export_onnx_adapter.py adapter.onnx --params-file lora.json -o adapter.onnx_adapter
```

```python
import onnxruntime as ort

session = ort.InferenceSession("model_lora.onnx")  # loaded once
adapter = ort.LoraAdapter()
adapter.Load("adapter.onnx_adapter")
run_opts = ort.RunOptions()
run_opts.add_active_adapter(adapter)  # swap adapters per call, no reload
outputs = session.run(None, {"input": x}, run_options=run_opts)
```

Calling `session.run` without an active adapter falls back to the baked
(zero-init, or whatever `apply_lora_adapter.py --adapter-inputs` last
applied) default -- verified end-to-end in
`tests/test_lora.py::test_export_onnx_adapter_matches_merged_model_via_native_lora_adapter`,
which checks this against the merged-model output bit for bit. `--adapter-
inputs` is also available on `apply_lora_adapter.py`, for turning a
specific trained adapter into a natively swappable base model without
retraining. `export_onnx_adapter.py`'s tensor names only need to be
internally consistent with the model's own graph input names (both come
from this same toolchain); unlike Olive's own HuggingFace-oriented naming
convention, there is no fixed scheme to match here.

### QLoRA (NF4-quantized base + LoRA)

`scripts/prepare_qlora.py` wires the LoRA graph surgery above together with
`onnxsim.nf4.quantize_weight_only_nf4` (bitsandbytes' NF4 4-bit format, see
`onnxsim/nf4.py`) to reproduce Dettmers et al. 2023's actual QLoRA recipe:
freeze a 4-bit-quantized base model and train small full-precision adapters
on top of its on-the-fly dequantized weights, instead of plain LoRA on an
unquantized base. It replaces step 1 above; steps 2-5 are unchanged:

```sh
python3 scripts/prepare_qlora.py model.onnx -o model_qlora.onnx \
  --rank 4 --block-size 64 --params-out lora.json

python3 scripts/generate_artifacts.py model_qlora.onnx -o artifacts \
  --lora-params-file lora.json --loss cross-entropy --optimizer adamw
# ...then train, extract, and re-apply exactly as in the plain-LoRA workflow.
```

This needs the `onnxsim` package itself importable (build/install this
repo's own wheel) for the NF4 quantization step -- unlike every other
onnx-finetune script, which needs only `onnx` + `numpy`. Injection must run
before quantization (quantizing first leaves nothing recognizable for
`inject_lora.py`'s graph surgery to graft onto), and `prepare_qlora.py`
always excludes the newly-injected `lora_A`/`lora_B` weights from
quantization itself -- structurally they're just another small MatMul
against a 2-D initializer, indistinguishable from any other layer's weight
by shape alone, so leaving them eligible would NF4-quantize the adapter too
and defeat the point of keeping it full precision. NF4 quantization only
covers 2-D MatMul/vanilla-Gemm weights (see `onnxsim/nf4.py`), so a 1x1-Conv
LoRA target stays full precision under `prepare_qlora.py` too.

## Knowledge distillation

Train a small "student" classifier to mimic a larger "teacher" via knowledge distillation
(Hinton et al. -- temperature-scaled soft-target cross-entropy, combined with the usual
hard-label cross-entropy): `generate_artifacts.py --loss distillation` builds a custom loss
graph (`scripts/distillation_loss.py`) that takes a second model's logits as an extra input,
rather than one of the three plain `--loss` enum choices above.

```sh
# 1. A teacher (bigger) and a student (smaller) classifier -- any two ONNX
#    models producing (batch, num_classes) logits work; make_toy_classifier.py
#    is provided for trying this out without one of your own.
python3 scripts/make_toy_classifier.py -o teacher.onnx --hidden-dim 32 --num-classes 4
python3 scripts/make_toy_classifier.py -o student.onnx --hidden-dim 8  --num-classes 4

# 2. Training artifacts for the student only -- the teacher never enters
#    generate_artifacts.py at all; its logits are supplied at *train* time
#    (step 4), not baked into the artifacts. --distill-temperature/--distill-alpha
#    default to 2.0/0.5 (1.0 = pure distillation, 0.0 = plain supervised training).
python3 scripts/generate_artifacts.py student.onnx -o artifacts \
  --loss distillation --distill-temperature 2.0 --distill-alpha 0.5

# 3. Labels are int64 class indices, not float32 -- make_synthetic_classification_data.py
#    writes them in that layout (see make_synthetic_data.py for the plain-regression
#    equivalent this tool's other --loss modes use).
python3 scripts/make_synthetic_classification_data.py --num-classes 4 --num-samples 2048

# 4. Train: --teacher-model runs a plain frozen inference session against
#    --teacher-model on each batch's input to supply teacher_logits, so
#    --train-input/--train-target stay exactly the same two files any other
#    --loss mode uses. --label-dtype int64 is required here (see below).
./build/onnx-finetune \
  --artifacts-dir artifacts --teacher-model teacher.onnx \
  --train-input train_input.bin --train-target train_labels.bin --label-dtype int64 \
  --input-dim 8 --target-dim 1 --num-samples 2048 \
  --batch-size 32 --epochs 20 --lr 0.01 \
  --output-model distilled.onnx --output-names logits
```

The training graph `--loss distillation` produces declares **three** external inputs at
runtime, in this fixed order: the model's own input, `teacher_logits`, then `labels` -- not
the usual two -- which is exactly what `--teacher-model` mode feeds (the ordinary two data
files, plus a teacher forward pass computed internally each step). Training also exposes the
soft/hard sub-losses as two extra outputs (`kd_soft_loss`/`kd_hard_loss`); `--log-every`'s
printed line includes both automatically whenever `--teacher-model` is given.

The WASM binding mirrors this one-for-one: pass a 5th argument (`teacherModelBytes`) to
`new Module.FinetuneSession(...)` to enable distillation mode -- `trainStep()`'s own JS
signature is unchanged, and `session.lastSubLosses()` returns `{soft, hard}` after each step
(`undefined` outside distillation mode). See `wasm/README.md`.

This is a general-purpose classification loss, not tied to any particular architecture --
unlike `examples/llm_distillation/`'s standalone PyTorch/ONNX Runtime Web demos elsewhere in
this repo (a ~1B/~162M-parameter causal-LM pair, and a browser-trained toy Llama respectively),
which distill actual language models. Nothing here requires `torch`/`transformers`: only
`onnx` + a training-enabled `onnxruntime`, same as every other `generate_artifacts.py` mode.

### Knowledge distillation (graph_grad) -- no training-enabled ONNX Runtime at all

A second, independent implementation of the same idea, existing alongside the one above rather
than replacing it: `scripts/generate_distillation_step_graph.py` differentiates the whole
student forward pass and the KD loss itself with onnxsim's own reverse-mode autodiff
(`onnxsim.graph_grad`/`onnxsim.qat_graph` -- the same machinery `onnxsim.lora`/`onnxsim.qat`
already use to avoid `onnxruntime.training` for LoRA and block-wise QAT), instead of
`onnxruntime.training.artifacts`/`onnxblock`. The result is **one** ordinary ONNX graph --
forward, loss, backward, and an Adam step all baked in -- runnable by repeatedly calling
`session.run()`/`Run()` on a **plain, non-training** onnxruntime. No `pip install
onnxruntime-training`, no from-source `--enable_training_apis` build, anywhere in this path --
see that script's own module docstring for the full design writeup, including the one real
trade-off it makes (a step graph's batch size is fixed at build time; `onnxruntime.training`'s
graph tolerates a symbolic one).

```sh
# 1. Same teacher/student pair as above.
python3 scripts/make_toy_classifier.py -o teacher.onnx --hidden-dim 32 --num-classes 4
python3 scripts/make_toy_classifier.py -o student.onnx --hidden-dim 8  --num-classes 4

# 2. One step graph, batch size fixed at build time. Also writes step.onnx.manifest.txt
#    and step.onnx.initial_state.bin alongside it -- a non-Python caller (the native CLI
#    below, or the wasm runner) needs both to actually run it.
python3 scripts/generate_distillation_step_graph.py student.onnx -o step.onnx --batch-size 32

# 3. Same synthetic data as above (int64 labels; this path builds its own one-hot matrix
#    from them on the host, see the script's docstring on why that's not done in-graph).
python3 scripts/make_synthetic_classification_data.py --num-classes 4 --num-samples 2048

# 4. Train with the plain-onnxruntime native tool (see "Building" below for ORT_HOME) --
#    a completely separate binary from ./build/onnx-finetune, built against a *plain*
#    onnxruntime (an official prebuilt release tarball works, no build at all).
./build/onnx-finetune-distill-step-graph \
  --step-graph step.onnx --teacher-model teacher.onnx \
  --train-input train_input.bin --train-target train_labels.bin --num-samples 2048 \
  --epochs 20 --lr 0.01 --output-weights final_weights.bin

# 5. Reassemble the trained weights into an ordinary, inference-ready .onnx -- the plain
#    Ort::Session this tool uses has no ExportModelForInferencing equivalent, so this one
#    step stays in Python, the same division of labor the rest of this repo already uses.
python3 scripts/apply_trained_weights.py student.onnx step.onnx.manifest.txt final_weights.bin \
  -o distilled.onnx
```

Building `onnx-finetune-distill-step-graph` needs only `-DORT_HOME=/path/to/a/plain/onnxruntime`
(see "Building onnx-finetune" below) -- unlike `onnx-finetune` itself, it never needs
`ORT_SOURCE_DIR`/`ORT_BUILD_DIR`. The browser has its own runner too, built on the *official*
`onnxruntime-web` npm package rather than a custom Emscripten build: see
`wasm/distill_step_graph/`.

## CLI reference

| flag | required | description |
|---|---|---|
| `--artifacts-dir` | yes | directory containing `checkpoint`, `training_model.onnx`, `eval_model.onnx`, `optimizer_model.onnx` |
| `--train-input` / `--train-target` | yes | raw binary files -- `--train-input` always float32, `--train-target` float32 unless `--label-dtype int64` |
| `--input-dim` / `--target-dim` | yes | per-sample element counts |
| `--num-samples` | yes | total samples in the input/target files |
| `--output-model` | yes | where to write the fine-tuned inference `.onnx` |
| `--output-names` | yes | comma-separated graph output name(s) of the *original* model |
| `--batch-size` | no (default 8) | |
| `--epochs` | no (default 10) | |
| `--lr` | no (default 1e-3) | |
| `--save-checkpoint` | no | also save the post-training checkpoint (for resuming training later) |
| `--log-every` | no (default 50) | print loss every N steps; 0 disables |
| `--label-dtype` | no (default `float32`) | `int64` for class-index labels (`--loss cross-entropy`/`distillation`) |
| `--teacher-model` | no | enables distillation mode: path to a frozen teacher `.onnx`, run each step to produce `teacher_logits`. Requires `--label-dtype int64` and artifacts from `--loss distillation` |
| `--teacher-output-name` | no | which of the teacher model's outputs to use; defaults to its sole output |
