# ONNX Simplifier

[![PyPI version](https://img.shields.io/pypi/v/onnxsim.svg)](https://pypi.python.org/pypi/onnxsim/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/onnxsim.svg)](https://pypi.python.org/pypi/onnxsim/)
[![PyPI license](https://img.shields.io/pypi/l/onnxsim.svg)](https://pypi.python.org/pypi/onnxsim/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/onnxsim/onnxsim/pulls)
[![Discord](https://img.shields.io/discord/1475920534847099121?logo=discord)](https://discord.gg/W3ht33v4)

_ONNX is great, but sometimes too complicated._

## Background

One day I wanted to export the following simple reshape operation to ONNX:

```python
import torch


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
```

The input shape in this model is static, so what I expected is

![simple_reshape](imgs/simple_reshape.png)

However, I got the following complicated model instead:

![complicated_reshape](imgs/complicated_reshape.png)

## Our solution

ONNX Simplifier is presented to simplify the ONNX model. It infers the whole computation graph
and then replaces the redundant operators with their constant outputs (a.k.a. constant folding).

## Features

At its core onnxsim runs a fixed point of shape inference, graph optimization and
constant folding until the model stops changing. Around that it offers:

- **Constant folding.** Evaluates the constant parts of the graph and replaces
  redundant operators with their computed outputs. By default initializers count
  as constants; pass `--initializers-as-non-constants` (Python:
  `initializers_as_constants=False`) to keep weights as tunable tensors so nodes
  rooted only at initializers — and value-baking fusions such as fuse BatchNorm
  into Conv — are left untouched.
- **Graph optimization passes.** Runs onnx-optimizer's fusions and eliminations
  (e.g. fuse BatchNorm into Conv). List them with
  `onnxsim --list-default-optimizers`; skip all or some with
  `--skip-optimization [pass ...]`. A pass not in the default set (typically a
  graph-shape rewrite rather than a pure node reduction, e.g. a defusion) can
  be requested explicitly with `--enable-optimization pass [pass ...]`
  (Python: `extra_optimizers=`); list those with `--list-other-optimizers`.
- **Shape inference.** Propagates tensor shapes through the graph — including
  partial shape evaluation via ONNX data propagation — to unlock more folding.
- **Correctness checking.** Optionally validates the simplified model against the
  original on `N` random inputs (the positional `check_n` argument, with
  configurable `--check-rtol`/`--check-atol`). Choose how the generated inputs are
  filled with `--input-fill` (Python: `input_fill=`): `random` (uniform `[0, 1)`,
  the default), `ones`, `zeros` or `arange`.
- **Fixed and dynamic input shapes.** Pin a dynamic model's shapes for
  simplification/checking with `--overwrite-input-shape` and `--test-input-shape`.
- **[Custom operators](#custom-operators).** Keeps custom ops (TensorRT plugins,
  vendor domains, or custom ops in the default ONNX domain) unchanged and picks
  up schemas registered via `onnx.defs.register_schema` automatically.
- **[Opset conversion](#changing-the-opset-version).** Upgrade or downgrade the
  model's opset while simplifying with `--target-opset`.
- **Function inlining.** Flatten the model's local (model-defined) functions into
  the main graph before simplifying with `--inline-functions` (Python:
  `inline_functions=True`), so the optimizer, shape inference and constant folding
  can see through function calls. Schema-defined (built-in) functions are left
  alone.
- **[Custom rewriters](#custom-rewriters).** Plug your own rewriting logic into
  the fixed point with `custom_rewriter`, or express data-only `FunctionProto`
  rules that also run from the C and Rust bindings.
- **[Safetensors / GGUF archives](#safetensors--gguf-archives).** Export a model
  to a standalone `.safetensors` or `.gguf` file (graph + weights in one
  ecosystem-standard archive) and import it back, from every binding.
- **[Transformers export](#transformers-export).** Export a Hugging Face
  `transformers` model straight to a simplified ONNX deployment directory with
  `onnxsim.export_transformers_model()`.
- **[Diffusion model export](#diffusion-model-export).** Export a Hugging
  Face `diffusers` pipeline (Stable Diffusion, SDXL, ...) straight to a
  simplified ONNX deployment directory with
  `onnxsim.export_diffusion_model()`.
- **[Detectron2 export](#detectron2-export).** Trace a
  [Detectron2](https://github.com/facebookresearch/detectron2) model
  (Faster/Mask/Keypoint R-CNN, RetinaNet, ...) to ONNX and simplify it with
  `onnxsim.export_detectron_model()`.
- **[SAM 2 export](#sam-2-export).** Trace a Meta
  [SAM 2](https://github.com/facebookresearch/sam2) image encoder and
  prompt/mask decoder to ONNX and simplify both with
  `onnxsim.export_sam2_model()`.
- **[Quantization-aware fine-tuning](#quantization-aware-fine-tuning).**
  Recover accuracy a quantization lost with `onnxsim.apply_qat()`:
  label-free, block-wise fine-tuning of the fp32 weights themselves against
  the float model's own activations, over any block topology --- and, with
  `learn_activation_scales=True`, of `quantize_static`'s activation
  quantizers jointly with them. The training step is emitted as an ONNX
  graph, so it runs on a GPU, an NPU execution provider or WebGPU via
  `step_providers=`.
- **Subgraph simplification.** Simplify `If`/`Loop`/`Scan` subgraph bodies too
  with `--include-subgraph`.
- **[MLIR export](#exporting-to-mlir-torch-mlir--onnx-mlir).** Hand the simplified
  model to [torch-mlir](https://github.com/llvm/torch-mlir) (Torch dialect) or
  [onnx-mlir](https://github.com/onnx/onnx-mlir) (ONNX dialect) with `--emit-mlir`
  (Python: `onnxsim.export_mlir`) — a bridge into MLIR-based compiler stacks
  (torch-mlir, IREE, onnx-mlir). Both backends are optional.
- **[Core ML export](#exporting-to-core-ml).** Convert the simplified model to a
  Core ML `.mlpackage`/`.mlmodel` with `--emit-coreml` (Python:
  `onnxsim.export_coreml`), via a built-in ONNX-to-MIL translator and
  [coremltools](https://github.com/apple/coremltools)' MIL-to-Core-ML backend.
  coremltools is optional.
- **[TensorFlow Lite export](#exporting-to-tensorflow-lite).** Convert the
  simplified model to a `.tflite` flatbuffer with `--emit-tflite` (Python:
  `onnxsim.export_tflite`), via a built-in ONNX-to-TensorFlow translator and
  `tf.lite.TFLiteConverter`. TensorFlow is optional; pass `--tflite-backend
  onnx2tf` to route through
  [onnx2tf](https://github.com/PINTO0309/onnx2tf) instead for far broader op
  coverage.
- **Large-model handling.** Guard against blow-up from ops like `Tile`/
  `ConstantOfShape` (`--no-large-tensor`), read and write external-data models,
  and eliminate unused outputs (`--unused-output`).
- **Many ways to run it.** A zero-install [web version](#web-version), a Python
  package and `onnxsim` CLI, a C API, and a Rust wrapper — all sharing the same
  C++ core. onnxruntime is optional; onnxsim falls back to the onnx reference
  evaluator when it isn't installed.

## Getting started

### Web version

We have published ONNX Simplifier on [GitHub pages](https://onnxsim.github.io/onnxsim/). It works out of the box and **doesn't need any installation**. Note that it runs in the browser locally and your model is completely safe.

### Python version


```
pip3 install -U pip && pip3 install onnxsim
```

Then

```
onnxsim input_onnx_model output_onnx_model
```

For more advanced features, try the following command for help message

```
onnxsim -h
```

`onnx` is the only required dependency. Everything else is an extra, installed
only if you want it -- among them `rich`, which is used purely to colour and box
the terminal reports (the original-vs-simplified table, the memory plan, the
graph diff, CLI warnings). Without it those print as plain-text ASCII tables and
the simplified models are byte-for-byte the same:

```
pip3 install "onnxsim[rich]"
```

### Node.js version

The same WebAssembly build backing the web version above is also published as
an npm package, for JavaScript tooling that wants ONNX simplification without
a native build step or a Python runtime. See
[`npm/onnxsim/README.md`](npm/onnxsim/README.md) for usage.

```
npm install onnxsim
```

## Demonstration

An overall comparison between
[a complicated model](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)
and its simplified version:

![Comparison between old model and new model](imgs/comparison.png)

## In-script workflow

If you would like to embed ONNX simplifier python package in another script, it is just that simple.

```python
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load(filename)

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object
```

You can see more details of the API in [onnxsim/onnx_simplifier.py](onnxsim/onnx_simplifier.py)

## Custom operators

Models that contain custom operators, such as TensorRT plugins
(`BatchedNMS_TRT`, `EfficientNMS_TRT`, ...), are supported. onnxsim keeps these
ops unchanged and simplifies the rest of the graph around them. This works
whether the custom op lives in a vendor-specific domain (e.g. `TRT`) or in the
default ONNX domain, so you no longer need to manually move it into a custom
domain to get past validation (issues
[#107](https://github.com/onnxsim/onnxsim/issues/107) and
[#220](https://github.com/onnxsim/onnxsim/issues/220)).

If you describe your custom operator to ONNX with
[`onnx.defs.register_schema`](https://onnx.ai/onnx/api/defs.html), onnxsim
picks that schema up automatically: onnxsim links its own copy of ONNX, so its
operator registry is separate from the `onnx` Python module's, and every
`simplify` call imports the schemas you registered into onnxsim's registry
before validating the model (issue
[#326](https://github.com/onnxsim/onnxsim/issues/326)). You can also trigger the
import explicitly with `onnxsim.import_onnx_schemas()`, or turn the automatic
import off with `onnxsim.simplify(model, import_custom_schemas=False)` (CLI:
`--skip-schema-import`).

```python
import onnx
import onnxsim

# Teach ONNX about your custom operator.
onnx.defs.register_schema(my_op_schema)

# simplify() imports the schema into onnxsim automatically.
model_simp, check_ok = onnxsim.simplify(model)
```

If a registered schema also has a type/shape-inference function (set via
`onnx.defs.OpSchema.set_type_and_shape_inference_function`), onnxsim registers a
trampoline that calls it back through `onnx.shape_inference.infer_node_outputs`
during simplification, so the custom operator's output shapes are inferred too.
Custom operators without an inference function are still imported; shape
inference simply flows past them.

## Changing the opset version

You can upgrade (or downgrade) the model's opset version while simplifying. Pass
`target_opset_version` to `simplify` (CLI: `--target-opset`) and onnxsim converts
the default ONNX domain to that opset — using onnx's own version converter —
before running the simplification, so any redundant nodes the conversion
introduces get cleaned up too.

```python
import onnx
import onnxsim

model = onnx.load(filename)

# Convert the model to opset 18 and simplify it.
model_simp, check = onnxsim.simplify(model, target_opset_version=18)
```

On the command line:

```
onnxsim input_onnx_model output_onnx_model --target-opset 18
```

When `target_opset_version` is left unset (the default), the model's opset
version is preserved.

The conversion runs inside onnxsim's C++ core, so every binding shares it —
the Python package, the C API and its Rust wrapper (`Options::target_opset_version`),
the standalone `onnxsim` binary (`--target-opset`), and the
[web version](https://onnxsim.github.io/onnxsim/) (the "target opset version"
field).

## Exporting to MLIR (torch-mlir / onnx-mlir)

Downstream compiler stacks built on [MLIR](https://mlir.llvm.org/) —
[torch-mlir](https://github.com/llvm/torch-mlir),
[IREE](https://iree.dev/) on top of it, and
[onnx-mlir](https://github.com/onnx/onnx-mlir) — consume models as MLIR rather
than as an ONNX `ModelProto`. onnxsim can bridge the gap: after simplifying, it
emits the model as MLIR in one of two dialects, chosen with `--mlir-target`
(Python: the `target` argument):

- **`torch`** (default) — **Torch-dialect** MLIR via torch-mlir's pure-Python
  ONNX importer.
- **`onnx`** — **ONNX-dialect** MLIR via the onnx-mlir compiler binary.

Simplifying first is the point — constant folding and the optimizer passes
collapse the shape-manipulation subgraphs the importer would otherwise translate
op by op, so the emitted MLIR is smaller and closer to what the compiler needs.

Both backends are **optional** (just like onnxruntime for constant folding):
neither is imported/located unless you actually emit MLIR.

### torch-mlir (Torch dialect)

Install torch-mlir:

```
pip install torch-mlir
```

Prebuilt wheels are listed at <https://github.com/llvm/torch-mlir>.

From the CLI, add `--emit-mlir`. Passed without a path it writes the MLIR next to
the output model with a `.mlir` extension; pass a path to choose the location:

```
# writes simplified.onnx and simplified.mlir
onnxsim input.onnx simplified.onnx --emit-mlir

# choose the MLIR path explicitly
onnxsim input.onnx simplified.onnx --emit-mlir model.mlir
```

From Python, `onnxsim.export_mlir` converts a model (typically the output of
`simplify`) and returns the MLIR text, optionally writing it to a file:

```python
import onnx
import onnxsim

model = onnx.load("input.onnx")
model_simp, ok = onnxsim.simplify(model)
assert ok

# Return the MLIR as a string...
mlir_text = onnxsim.export_mlir(model_simp)
# ...and/or write it to a file.
onnxsim.export_mlir(model_simp, "model.mlir")
```

### onnx-mlir (ONNX dialect)

onnx-mlir has no pip-installable importer, so this backend shells out to the
`onnx-mlir` compiler binary (`--EmitONNXIR`). Build or install it from
<https://github.com/onnx/onnx-mlir>, then make it discoverable — put `onnx-mlir`
on your `PATH`, set `ONNX_MLIR_HOME` to its install prefix (the binary is
expected at `$ONNX_MLIR_HOME/bin/onnx-mlir`), set `ONNX_MLIR` to the binary
path, or pass the path explicitly.

```
# locate onnx-mlir via PATH / ONNX_MLIR_HOME / ONNX_MLIR
onnxsim input.onnx simplified.onnx --emit-mlir --mlir-target onnx

# or point at the binary directly
onnxsim input.onnx simplified.onnx --emit-mlir model.mlir \
  --mlir-target onnx --onnx-mlir /path/to/onnx-mlir
```

```python
mlir_text = onnxsim.export_mlir(model_simp, target="onnx")
# with an explicit binary path:
onnxsim.export_mlir(model_simp, "model.mlir", target="onnx",
                    onnx_mlir="/path/to/onnx-mlir")
```

`export_mlir` accepts a few keyword arguments, forwarded to the selected backend
— e.g. `opset_version` to run ONNX's version converter first (both targets
prefer recent opsets), `verify=False` (torch) to skip MLIR verification, and
`emit` / `extra_args` (onnx) to change the onnx-mlir emit flag or pass extra
compiler options. See `onnxsim/mlir_export.py` for the full signatures.

## Exporting to Core ML

Apple platforms want the graph as a Core ML model instead of ONNX or MLIR.
coremltools dropped its own ONNX frontend in version 7 (it only converts
TensorFlow/PyTorch models, or an in-memory MIL program) — there's no
off-the-shelf "convert this ONNX model" call left to lean on, so onnxsim ships
its own ONNX-to-MIL translator and hands the result to coremltools'
MIL-to-Core-ML backend to produce the actual model. It covers a practical
subset of ONNX ops (conv/pooling/normalization, matmul/gemm, elementwise math,
reshapes, reductions, and more — see `coreml_export.SUPPORTED_ONNX_OPS`); a
node whose op isn't supported raises a clear error naming the op, rather than
silently producing a wrong model. Feeding in a *simplified* model is the point,
same as with MLIR export: onnxsim's constant folding turns more of the graph
into plain initializers, so more of it lands on the translator's supported-op
list.

coremltools is **optional**, just like onnxruntime for constant folding: it
isn't imported unless you actually export to Core ML.

```
pip install coremltools
```

Converting an ONNX model to MIL / Core ML needs no macOS-specific
functionality (MIL construction and `.mlpackage` serialization are pure
Python/protobuf), so it runs the same on Linux, macOS, or Windows. Only
*loading the produced model back for a prediction* needs Core ML's runtime,
i.e. an Apple OS — pass `skip_model_load=False` (Python) once you're on macOS
to get a model that's ready to call `.predict()` on; the default
(`skip_model_load=True`) lets conversion succeed everywhere else too.

Graph inputs must have fully static shapes (dynamic axes aren't supported).

From the CLI, add `--emit-coreml`. Passed without a path it writes the model
next to the output model with a `.mlpackage`/`.mlmodel` extension; pass a path
to choose the location:

```
# writes simplified.onnx and simplified.mlpackage
onnxsim input.onnx simplified.onnx --emit-coreml

# choose the path and the legacy .mlmodel format explicitly
onnxsim input.onnx simplified.onnx --emit-coreml model.mlmodel --coreml-format neuralnetwork
```

From Python, `onnxsim.export_coreml` converts a model (typically the output of
`simplify`) and returns the `coremltools.models.MLModel`, optionally saving it:

```python
import onnx
import onnxsim

model = onnx.load("input.onnx")
model_simp, ok = onnxsim.simplify(model)
assert ok

# Return the MLModel...
mlmodel = onnxsim.export_coreml(model_simp)
# ...and/or save it to a .mlpackage (or .mlmodel with convert_to="neuralnetwork").
onnxsim.export_coreml(model_simp, "model.mlpackage")
```

`export_coreml` accepts a few keyword arguments: `convert_to` (`"mlprogram"`,
the default, or the legacy `"neuralnetwork"`), `compute_units` (which devices
the model may run on, e.g. `"CPU_ONLY"`), `compute_precision`,
`minimum_deployment_target` (e.g. `"iOS16"`), `io_dtype` (see below), and
`skip_model_load` (see above). See `onnxsim/coreml_export.py` for the full
signature.

`io_dtype="fp16"` (CLI: `--coreml-io-dtype fp16`) declares the model's float
inputs and outputs float16 instead of float32. An ML Program already computes
in float16, so the float32 default only buys a conversion in each direction on
every call, over twice the bytes — with no accuracy difference, since a float32
output is just an upcast of the float16 value Core ML computed either way. It's
worth most where the same large float tensors cross the boundary repeatedly, as
a transformer decoder's KV cache does on every generated token. Requires
`convert_to="mlprogram"` and raises the deployment target to iOS16/macOS13 when
one isn't given. See
[`scripts/apple/README.md`](scripts/apple/README.md)'s "fp16 model interface"
section.

## Exporting to TensorFlow Lite

Mobile/embedded runtimes built on TensorFlow want the graph as a `.tflite`
flatbuffer instead of ONNX or Core ML. `onnx-tensorflow`/`onnx-tf` (the
project that used to fill this gap) has been unmaintained for years and only
tracks very old opsets, so -- same situation as Core ML after coremltools
dropped its own ONNX frontend -- onnxsim ships its own ONNX-to-TensorFlow
translator: it builds the equivalent computation with plain TensorFlow ops
inside a `tf.function`, traces it into a concrete function, and hands that to
`tf.lite.TFLiteConverter` to produce the actual `.tflite` model. It covers a
practical subset of ops (conv/pooling/normalization, matmul/gemm, elementwise
math, reshapes, reductions, and more -- see
`tflite_export.SUPPORTED_ONNX_OPS`); a node whose op isn't supported raises a
clear error naming the op, rather than silently producing a wrong model.
Feeding in a *simplified* model is the point, same as with the other export
backends: onnxsim's constant folding turns more of the graph's
shape-manipulation subgraphs into plain initializers, which this translator
needs at conversion time for things like a `Reshape`'s target shape or a
`Slice`'s bounds.

TensorFlow is **optional**, just like onnxruntime for constant folding and
coremltools for Core ML export: it isn't imported unless you actually export
to TFLite.

```
pip install tensorflow
```

TensorFlow Lite's own op kernels are NHWC-only, while ONNX's conv/pool ops
are NCHW; this translator keeps the graph's public tensors in ONNX's NCHW
layout and transposes to/from NHWC only around the ops that need it, so no
manual layout conversion is required on your part. Graph inputs must have
fully static shapes (dynamic axes aren't supported) -- pin them first with
`--overwrite-input-shape`/`--test-input-shape` if needed.

From the CLI, add `--emit-tflite`. Passed without a path it writes the model
next to the output model with a `.tflite` extension; pass a path to choose
the location:

```
# writes simplified.onnx and simplified.tflite
onnxsim input.onnx simplified.onnx --emit-tflite

# choose the path explicitly, and enable TFLite's default post-training
# (dynamic-range) quantization
onnxsim input.onnx simplified.onnx --emit-tflite model.tflite --tflite-optimize
```

From Python, `onnxsim.export_tflite` converts a model (typically the output
of `simplify`) and returns the serialized `.tflite` flatbuffer (`bytes`),
optionally writing it to a file:

```python
import onnx
import onnxsim

model = onnx.load("input.onnx")
model_simp, ok = onnxsim.simplify(model)
assert ok

# Return the flatbuffer bytes...
tflite_model = onnxsim.export_tflite(model_simp)
# ...and/or write it to a file.
onnxsim.export_tflite(model_simp, "model.tflite")
```

`export_tflite` accepts an `optimizations` keyword argument, forwarded to
`tf.lite.TFLiteConverter.optimizations` (e.g. `["DEFAULT"]`, what
`--tflite-optimize` sets, to enable post-training dynamic-range
quantization). See `onnxsim/tflite_export.py` for the full signature.

### A broader-coverage backend: onnx2tf

The built-in translator above covers a practical op subset. For a model that
hits an unsupported op, pass `backend="onnx2tf"` (CLI: `--tflite-backend
onnx2tf`) to route the conversion through
[onnx2tf](https://github.com/PINTO0309/onnx2tf) instead -- a separate,
actively maintained project with far broader op coverage (~200 ops) and years
of production hardening across real-world model zoos.

```
pip install onnx2tf
```

onnx2tf is a much heavier dependency than the builtin backend needs (it pulls
its own TensorFlow, onnxruntime, onnx-graphsurgeon, and a couple dozen small
`*4onnx` helper packages), and it changes the model's public input/output
tensor layout to channel-last by default -- it converts *every* tensor of
rank >= 3 to that convention, not just 4-D image tensors, unlike the builtin
backend which always keeps ONNX's own declared shapes. Pass onnx2tf's own
`keep_ncw_or_nchw_or_ncdhw_input_names` (a list of input names to keep in
their original ONNX layout) as an extra keyword argument if you need specific
inputs to keep their original layout.

```
onnxsim input.onnx simplified.onnx --emit-tflite --tflite-backend onnx2tf
```

```python
tflite_model = onnxsim.export_tflite(model_simp, backend="onnx2tf")
```

`--tflite-optimize`/`optimizations` only applies to the builtin backend; use
onnx2tf's own quantization options (forwarded as extra keyword arguments,
e.g. `output_integer_quantized_tflite=True`) instead. See
`onnxsim/onnx2tf_export.py` for the full signature and onnx2tf's own
documentation for its option list.

## Constant folding on the GPU (CUDA execution provider)

onnxsim constant-folds by running the foldable sub-graphs through ONNX Runtime.
By default it uses the CPU execution provider, which is always available and
gives deterministic results. For large models it can be much faster to fold on
an NVIDIA GPU. Pass `providers` to `simplify` to choose the ONNX Runtime
[execution providers](https://onnxruntime.ai/docs/execution-providers/), in
priority order:

```python
import onnx
import onnxsim

model = onnx.load(filename)

# Fold on the GPU, falling back to CPU for ops CUDA cannot run.
model_simp, check = onnxsim.simplify(
    model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

On the command line:

```
# Explicit provider list (priority order):
onnxsim input_onnx_model output_onnx_model \
    --providers CUDAExecutionProvider CPUExecutionProvider

# Or the shortcut, equivalent to the line above:
onnxsim input_onnx_model output_onnx_model --cuda
```

Keeping `CPUExecutionProvider` last is recommended: ONNX Runtime falls back to
it for any operator the GPU provider cannot run. Each provider entry may also be
a `(name, options)` tuple, exactly as
[`onnxruntime.InferenceSession`](https://onnxruntime.ai/docs/api/python/api_summary.html)
accepts it, for example to pin a specific `device_id`:

```python
model_simp, check = onnxsim.simplify(
    model,
    providers=[("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
)
```

The CUDA execution provider requires the GPU build of ONNX Runtime
(`pip install onnxruntime-gpu`). If you request a provider the installed ONNX
Runtime does not offer, onnxsim raises a `ValueError` listing the available
providers instead of silently folding on the CPU. When `providers` is left
unset (the default), folding runs on the CPU.

## Profiling the optimization

Simplification alternates a handful of transforms -- shape inference, the
onnx-optimizer passes, constant folding and any custom rewriter -- to a joint
fixed point. To see where the time and memory go, pass `profile` to `simplify`
(or `--profile` on the command line). onnxsim then measures each fixed-point
function's wall-clock and CPU duration and the peak resident memory reached while
it runs, prints a per-function summary, and writes a
[Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
JSON. Open that file in `chrome://tracing` or at
[ui.perfetto.dev](https://ui.perfetto.dev) to view it as a **flame graph**: the
nested fixed points appear as parent spans and the individual transforms as their
children, one box per invocation, annotated with peak RSS and CPU time.

Constant folding's actual work is running the model through ONNX Runtime, so
those session runs are profiled too. Each fold group appears under `FoldConstant`
as an `OrtSession` span, which times running that group's sub-model through the
inference executor. This works for every binding, since it wraps the one call
site common to both the built-in ONNX Runtime executor and the Python executor
that `simplify()` uses. When the built-in executor runs, the `OrtSession` span is
split further into `OrtSessionInit` (building the session, where ONNX Runtime
loads the graph and usually the dominant cost) and `OrtSessionRun` (the
inference). This makes it easy to see how much of simplification time is spent
inside ONNX Runtime versus in shape inference and the optimizer passes.

```python
import onnx
import onnxsim

model = onnx.load(filename)

# Write the trace to profile.json (open it in chrome://tracing or ui.perfetto.dev).
model_simp, check = onnxsim.simplify(model, profile="profile.json")
```

On the command line:

```
# Give a path, or omit it to use onnxsim_profile.json in the current directory.
onnxsim input_onnx_model output_onnx_model --profile profile.json
```

The printed summary looks like:

```
onnxsim profiling summary (per fixed-point function)
-------------------------------------------------------------------------------------
function                calls     wall(ms)      cpu(ms) max wall(ms)    peak(MiB)
-------------------------------------------------------------------------------------
Simplify                    1       260.59       270.93       260.59       112.95
  Pipeline                  3       259.75       269.67       100.76       112.94
    OptAndShape             3       158.63       165.13        53.20       101.43
    FoldConstant            3       100.36       103.78        47.69       112.93
      Optimize              3       112.56       116.99        37.77       101.42
      InferShapes           3        45.46        47.10        15.22        78.68
      OrtSession           12        71.44        74.02        18.31       112.93
        OrtSessionInit     12        58.02        60.11        15.90       112.93
        OrtSessionRun      12         9.85        10.42         2.71       109.10
-------------------------------------------------------------------------------------
```

(`OrtSessionInit`/`OrtSessionRun` show only when the built-in ONNX Runtime
executor runs the fold; the Python `simplify()` path shows just `OrtSession`.)

`calls` is how many times a function ran across all fixed-point rounds, `cpu(ms)`
is process CPU time (it can exceed wall time when constant folding runs multiple
ONNX Runtime threads), and `peak(MiB)` is the highest process RSS observed while
that function was on the stack (sampled by a lightweight background thread; tune
the interval with `ONNXSIM_PROFILE_INTERVAL_MS`, default 5ms).

Profiling is implemented in onnxsim's C++ core and is driven by the
`ONNXSIM_PROFILE` environment variable (the Python `profile` argument and the
`--profile` flag just set it), so it also works from the C ABI and the Rust
wrapper without any code change:

```
ONNXSIM_PROFILE=profile.json onnxsim input_onnx_model output_onnx_model
```

### ONNX Runtime's own session profiler

The `OrtSession` span above times each folding session as a whole. For a
finer, per-operator breakdown *inside* those sessions, turn on ONNX Runtime's
own [session profiler](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
with `ort_profile` (or `--ort-profile`). This flips on
`SessionOptions.enable_profiling` for the ONNX Runtime sessions onnxsim runs
while simplifying (the constant-folding sessions, plus the correctness-check
runs when `check_n > 0`), so each one writes ONNX Runtime's detailed per-kernel
Chrome trace:

```python
# Write onnxruntime session traces with the given file prefix.
model_simp, check = onnxsim.simplify(model, ort_profile="ort_profile")
```

```
onnxsim input_onnx_model output_onnx_model --ort-profile ort_profile
```

The value is a file **prefix**: ONNX Runtime writes one
`<prefix>_<timestamp>.json` per session, so a run that folds in several batches
produces several files (open each in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev)). It is independent of `profile` --
use either, or both together (`profile` for onnxsim's pipeline, `ort_profile`
for what ONNX Runtime does inside each fold). Like `profile`, it is driven by an
environment variable (`ONNXSIM_ORT_PROFILE`), so it works from every binding:

```
ONNXSIM_ORT_PROFILE=ort_profile onnxsim input_onnx_model output_onnx_model
```

#### Merging it into onnxsim's trace

Rather than juggling separate files, `merge_ort_profile` (or `--merge-ort-profile`)
splices ONNX Runtime's per-operator events straight into onnxsim's `profile`
trace, so each `OrtSession` span gets ONNX Runtime's operator-level detail lined
up beneath it on its own **onnxruntime** track -- one unified flame graph. It
implies `profile` (defaulting to `onnxsim_profile.json`), and ONNX Runtime's
intermediate traces are captured to a temporary directory and removed after
merging, so nothing is left behind. This works for every executor, including the
Python one `simplify()` uses:

```python
model_simp, check = onnxsim.simplify(model, profile="profile.json", merge_ort_profile=True)
```

```
onnxsim input_onnx_model output_onnx_model --profile profile.json --merge-ort-profile
```

The merge is also available from the **C ABI, Rust and WASM bindings** (which
fold through the built-in ONNX Runtime executor): set the `ONNXSIM_MERGE_ORT_PROFILE`
environment variable and it is done entirely in onnxsim's C++ core -- no Python
needed. It implies `ONNXSIM_PROFILE` (defaulting to `onnxsim_profile.json`):

```
ONNXSIM_MERGE_ORT_PROFILE=1 onnxsim input_onnx_model output_onnx_model
```

### Node-reduction plot

A `profile` trace also records how many nodes the graph holds right after
every round of each fixed-point loop (`Optimize`, `FoldConstant`, and
`Rewrite` when a `custom_rewriter` is given), as `NodeCount` counter events.
`onnxsim.profile_plot.plot_node_reduction` (or `--node-reduction-plot` on the
command line) turns those into a PNG with one subplot per loop -- node count
against round index -- so you can see at a glance how many rounds each loop
took and whether it converged (a flat tail) or hit the round cap
(`ONNXSIM_FIXED_POINT_ITERS`, default 50) still descending. It needs
matplotlib (`pip install onnxsim[plot]`):

```python
model_simp, check = onnxsim.simplify(model, profile="profile.json")

from onnxsim.profile_plot import plot_node_reduction
plot_node_reduction("profile.json")  # -> profile.json_node_reduction.png
```

```
# Implies --profile if not given explicitly.
onnxsim input_onnx_model output_onnx_model --node-reduction-plot
```

## Custom rewriters

Beyond the built-in optimizer passes, you can plug your own graph rewriting
logic into simplification with the `custom_rewriter` parameter of `simplify()`.
It accepts a callable

```python
Callable[[onnx.ModelProto], Optional[onnx.ModelProto]]
```

that either returns a rewritten model or mutates the model in place and returns
`None`. The callable runs **inside** onnxsim's simplification fixed point,
interleaved with shape inference, the built-in optimizer and constant folding —
so a rewrite can expose new optimization/folding opportunities and vice versa,
and the whole pipeline iterates until it converges. onnxsim itself takes no
dependency on any particular rewriting library; you bring your own.

### Using onnx-rewriter (`onnxscript.rewriter`)

[onnx-rewriter](https://github.com/microsoft/onnxscript) lets you express a
subgraph pattern and its replacement as plain Python and have it matched and
rewritten anywhere in the model. Install it alongside onnxsim:

```
pip3 install onnxscript
```

Then define a rule set and hand it to `simplify` via `custom_rewriter`. This
example fuses `MatMul` + `Add` into a single `Gemm`:

```python
import onnx
import onnxsim
from onnxscript.rewriter import pattern, rewrite

# The subgraph to match: y = MatMul(x, w) + b
def matmul_add_pattern(op, x, w, b):
    return op.Add(op.MatMul(x, w), b)

# What to replace it with: y = Gemm(x, w, b)
def gemm_replacement(op, x, w, b):
    return op.Gemm(x, w, b)

rules = pattern.RewriteRuleSet(
    [pattern.RewriteRule(matmul_add_pattern, gemm_replacement)]
)

model = onnx.load("model.onnx")
model_simp, check = onnxsim.simplify(
    model,
    custom_rewriter=lambda m: rewrite(m, pattern_rewrite_rules=rules),
)
assert check, "Simplified ONNX model could not be validated"
```

Because the rewriter runs every round of the fixed point, the fused `Gemm`
above (and anything it unlocks) is folded and re-optimized together with the
rest of the graph.

### Skipping the copy when nothing is rewritten

The rewriter runs on every fixed-point round, including the final one where it
has nothing left to do — and the fixed point always ends with at least one
such no-op round to detect convergence. onnxsim hands the model to your
callable as protobuf bytes and parses whatever comes back into a fresh
`ModelProto`, so a rewriter that reports a rewritten model each round pays for
that copy even when it changed nothing.

Return `False` to tell onnxsim that this round rewrote nothing; onnxsim then
keeps the model it already has and skips the round-trip. Run the rules through
onnx-ir's `PassManager` — `onnxscript.rewriter.RewritePass` wraps a rule set as
an IR pass — and read the `modified` flag of the `PassResult` it returns. That
flag is the reliable signal: an IR round-trip can reorder the serialized bytes
even when no rule fires, so a byte comparison would falsely report a change.

```python
from onnxscript import ir
from onnxscript.rewriter import RewritePass, pattern

rules = pattern.RewriteRuleSet(
    [pattern.RewriteRule(matmul_add_pattern, gemm_replacement)]
)
rewrite_pass = ir.passes.PassManager([RewritePass(rules)])

def apply_rules(model: onnx.ModelProto):
    model_ir = ir.serde.deserialize_model(model)
    result = rewrite_pass(model_ir)  # ir.passes.PassResult
    if not result.modified:
        return False  # no rule fired this round: skip the copy
    return ir.serde.serialize_model(result.model)

model_simp, check = onnxsim.simplify(model, custom_rewriter=apply_rules)
```

The plain `lambda m: rewrite(m, pattern_rewrite_rules=rules)` form still works —
it just always returns a model, so onnxsim copies it back every round.

A few things to keep in mind:

- **Keep the model schema-valid.** After each rewrite onnxsim validates the
  model, so any op you introduce must be registered at the model's opset (for
  example `Gelu` only exists from opset 20). Custom-domain ops are fine — see
  [Custom operators](#custom-operators) for registering their schemas.
- **Match the opset your rules target.** Convert the model to the opset your
  patterns expect (e.g. with `onnx.version_converter`) before simplifying if
  needed.
- **You are not limited to onnx-rewriter.** Any callable works — a hand-written
  pass over `model.graph`, an [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
  edit, etc. — as long as it takes and returns a `ModelProto`.

### From the C API and Rust

The custom rewriter lives in onnxsim's C++ core, so the C API and its Rust
wrapper expose it too — the model is exchanged as serialized `ModelProto` bytes
across the boundary instead of as an `onnx.ModelProto` object. In Rust, use
[`simplify_with_rewriter`](rust/onnxsim) (or `simplify_path_with_rewriter`) and
pass a closure `FnMut(&[u8]) -> Result<Option<Vec<u8>>, E>`: return `Ok(None)`
when a round rewrote nothing (onnxsim skips the copy, matching the Python
`False` sentinel), `Ok(Some(bytes))` for the rewritten model, or `Err(..)` to
abort.

```rust
let simplified = onnxsim::simplify_with_rewriter(
    &model_bytes,
    &onnxsim::Options::new(),
    |bytes: &[u8]| {
        // Decode `bytes`, rewrite, and return the new bytes — or Ok(None).
        let _ = bytes;
        Ok::<_, onnxsim::Error>(None)
    },
)?;
```

In C, pass an `OnnxsimRewriteFn` callback (and an optional matching free
callback) to `onnxsim_simplify` / `onnxsim_simplify_path`; see
[`onnxsim/capi/onnxsim_c_api.h`](onnxsim/capi/onnxsim_c_api.h) for the contract.
The only binding without it is the standalone CLI, which has no way to carry a
user callback.

### FunctionProto rules (works in every binding)

`custom_rewriter` takes a Python **callable**, so it only works from the Python
binding. If instead you express a rule as **pure data** — a `(pattern,
replacement)` pair of `onnx.FunctionProto` — onnxsim matches and applies it in
its C++ core, so the *same* rule set also works from the C and Rust bindings
with no dependency on onnxscript. The pattern's inputs are wildcards that bind
to graph values, its body is the subgraph to match, and its outputs are rewired
to the replacement's outputs. Build the FunctionProtos with
`onnx.parser.parse_function`:

```python
import onnx
import onnxsim
from onnx import parser

pattern = parser.parse_function("""
<domain: "com.example", opset_import: ["" : 18]>
matmul_add_pattern (x, w, b) => (y)
{
    t = MatMul(x, w)
    y = Add(t, b)
}
""")
replacement = parser.parse_function("""
<domain: "com.example", opset_import: ["" : 18]>
gemm_replacement (x, w, b) => (y)
{
    y = Gemm(x, w, b)
}
""")

model = onnx.load("model.onnx")
model_simp, check = onnxsim.simplify(
    model, function_rewrite_rules=[(pattern, replacement)]
)
```

This is enough to stand in for a hand-written onnxoptimizer pass: the rule above
reproduces the built-in `fuse_matmul_add_bias_into_gemm` fusion. Skip the
built-in pass and let the rule do it:

```python
model_simp, check = onnxsim.simplify(
    model,
    skipped_optimizers=["fuse_matmul_add_bias_into_gemm"],
    function_rewrite_rules=[(pattern, replacement)],
)
```

A node attribute written `@name` (an ONNX-text *ref attribute*) is an attribute
wildcard: it binds the matched node's attribute and is substituted into the
replacement. `function_rewrite_rules` is mutually exclusive with
`custom_rewriter`.

Many of onnxscript's ready-made
[common rewrite rules](https://github.com/microsoft/onnxscript/tree/main/onnxscript/rewriter/rules/common)
(e.g. `matmul_add_to_gemm_rule`, `reshape_reshape_rule`) are simple structural
pattern→replacement rules that translate directly into a FunctionProto pair like
the one above; see `tests/test_function_rewriter_common_rules.py` for worked
examples that check parity against the onnxscript rule itself.

Instead of writing the ONNX text by hand you can author each side as an
`onnxscript.script` function and call `.to_function_proto()` — a Python-typed
attribute parameter (`alpha: float`) even compiles to the `@name` wildcard form:

```python
from onnxscript import script
from onnxscript import opset18 as op

@script()
def matmul_add(a, b, c):
    return op.Add(op.MatMul(a, b), c)

@script()
def gemm(a, b, c):
    return op.Gemm(a, b, c)

model_simp, check = onnxsim.simplify(
    model,
    function_rewrite_rules=[(matmul_add.to_function_proto(), gemm.to_function_proto())],
)
```

See `tests/test_function_rewriter_onnxscript_script.py` for the `@script`
approach, including the attribute-wildcard case. To reuse an *existing*
`onnxscript` rule, its structural `pattern` method can be compiled to a
FunctionProto through the same `@script` converter —
`tests/test_function_rewriter_compile_rule.py` shows a small helper that does
this (paired with a simple replacement), noting the boundary where a rewrite
that derives attributes from the match can't be compiled that way.

From C, call `onnxsim_simplify_with_rules` with the serialized FunctionProto
pairs (see `onnxsim/capi/onnxsim_c_api.h`); from Rust, use
`Options::function_rewrite_rule(pattern_bytes, replacement_bytes)`.

**Capabilities and limits of the built-in matcher.** It matches arbitrary
connected DAG patterns with one or more outputs, tries both operand orders for
the commutative binary ops (`Add`, `Mul`, …), matches attributes exactly or as
`@name` wildcards, matches a pattern `Constant` against a byte-equal
initializer, and refuses a rewrite that would break a value consumed outside the
match. It does **not** (in this version) traverse `If`/`Loop`/`Scan` subgraph
bodies, handle variadic/optional-input arity mismatches, match >2-operand
commutative permutations, or evaluate attribute *predicates* — for those, the
Python-only `onnxscript.rewriter` via `custom_rewriter` remains the richer
option.

## Safetensors / GGUF archives

**Reading [onnx-safetensors](https://github.com/justinchuby/onnx-safetensors)-styled
models needs no special handling.** `onnx_safetensors.save_file`/`save_model`/
`load_file_as_external_data` write an ordinary `.onnx` graph whose initializers
use *standard* ONNX external data (`location`/`offset`/`length`) pointing into a
real `.safetensors` file — the safetensors JSON header is simply skipped over by
`offset`. That's exactly the external-data mechanism `simplify()` already reads
(from a path or from `onnx.load()`), so a model produced by onnx-safetensors —
or any other tool following the same convention — loads and simplifies with no
onnxsim-specific code involved:

```python
import onnx
import onnxsim

# model.onnx + model.safetensors, written by onnx_safetensors.save_model(...)
model_opt, check_ok = onnxsim.simplify("model.onnx")
# ...or with the weights already resolved into the ModelProto by onnx itself:
model_opt, check_ok = onnxsim.simplify(onnx.load("model.onnx"))
```

This is unrelated to the standalone archive format described below (which
embeds the *graph* in the safetensors/GGUF file too, in an onnxsim-specific
layout) — onnx-safetensors' two-file layout is just a plain `.onnx` model as
far as onnxsim (or any other ONNX consumer) is concerned.

Besides plain `.onnx`, a model can be exported to (and imported back from) a
**standalone safetensors or GGUF archive**: every initializer's bytes move into
the archive with real, byte-accurate offsets — openable by the `safetensors`
Python package / HF tooling, or any GGUF reader, with no onnxsim involved — and
the graph itself is embedded alongside them, so the one archive file is both
the model's weights and its graph. This is useful for interop with the
safetensors/GGUF ecosystems, or simply as a one-file way to move a model
around; it does not itself simplify anything, so pair it with `simplify()` if
you want both.

From Python:

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
onnxsim.export_safetensors(model, "model.onnx.safetensors")
onnxsim.export_gguf(model, "model.onnx.gguf")

# ... later, or in another process:
model = onnxsim.import_safetensors("model.onnx.safetensors")
model = onnxsim.import_gguf("model.onnx.gguf")
```

From Rust:

```rust
let model_bytes = std::fs::read("model.onnx")?;
onnxsim::export_safetensors(&model_bytes, "model.onnx.safetensors")?;
onnxsim::export_gguf(&model_bytes, "model.onnx.gguf")?;

let model_bytes = onnxsim::import_safetensors("model.onnx.safetensors")?;
let model_bytes = onnxsim::import_gguf("model.onnx.gguf")?;
```

From C, `onnxsim_export_safetensors`/`onnxsim_import_safetensors` and their
`_gguf` counterparts follow the same `out_data`/`out_size`/`out_error`
convention as `onnxsim_simplify` (see
[`onnxsim/capi/onnxsim_c_api.h`](onnxsim/capi/onnxsim_c_api.h)); the export
side takes the model as bytes and writes straight to a path, the import side
reads a path and hands back a freshly allocated buffer of model bytes.

The [web version](#web-version) exposes the same thing as UI: a format
dropdown (`.onnx` / `.onnx.safetensors` / `.onnx.gguf`) next to the converter's
**Download** button, and the file picker accepts either archive format as an
upload, decoding it back to ONNX before simplifying.

Importing an archive with no embedded model — e.g. a plain, weights-only
safetensors/GGUF file from somewhere else, with no onnxsim-authored graph
alongside the tensors — fails with a clear error rather than silently
returning nothing: there is no graph in it to import.

## Transformers export

A Hugging Face `transformers` model distribution (a `config.json` plus
`.safetensors` weights, e.g. anything under a Hub repo like
`meta-llama/...`/`Qwen/...`) has no ONNX graph in it at all — the model's
structure lives in the `transformers` Python modeling code, driven by
`config.json`, not in the weights file. onnxsim has no PyTorch tracing code of
its own to turn that into a graph, and does not need any: Hugging Face's own
[`optimum`](https://github.com/huggingface/optimum) package already exports
hundreds of architectures (via `optimum.exporters.onnx`) to plain ONNX —
including the split multi-file encoder/decoder-with-past shape autoregressive
generation needs. That export deliberately does no runtime-specific op fusion,
so there is real simplification left for onnxsim to find.

This is a different tool for a different job than [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)'s
own model builder (`onnxruntime_genai.models.builder`): that one only covers a
fixed, curated list of decoder-only causal-LM architectures, and its output is
already fused/quantized into ORT-specific ops (`com.microsoft::MatMulNBits`,
`GroupQueryAttention`, ...) meant to be consumed directly by ORT GenAI's own
`generate()` loop — there's little left for a generic simplifier to do to it,
and it doesn't cover encoder-only, seq2seq, vision, or audio architectures at
all. `optimum`'s export is the right shape for onnxsim to build on instead: a
plain graph, for any architecture with an `OnnxConfig`.

`onnxsim.export_transformers_model()` wraps the export-with-optimum,
simplify-every-graph-in-place recipe as one call:

```python
import onnxsim

results = onnxsim.export_transformers_model(
    "hf-internal-testing/tiny-random-t5",
    "exported_and_simplified",
    task="text2text-generation-with-past",
)
# {"encoder_model.onnx": True, "decoder_model.onnx": True, "decoder_with_past_model.onnx": True}
```

`output_dir` ends up holding the same files a plain `optimum` export would
(tokenizer/config files copied through untouched), except every `.onnx` file
has been simplified in place — so the directory is still deployable exactly
as-is (e.g. via `optimum.onnxruntime.ORTModelForSeq2SeqLM.from_pretrained`).
Needs the optional `torch`/`transformers`/`optimum` (with the `optimum-onnx`
distribution) packages: `pip install onnxsim[transformers]`.

Every simplified graph is saved with its weights in a companion `.data` file
by default (`save_as_external_data=True`) rather than inline — unlike the
`onnxsim` CLI's own `--save-as-external-data`/plain `onnx.save`, which default
off and only use external data as a fallback once a graph is too large to
serialize inline at all (>2GB). A real with-past export is several
*independent* graphs (encoder/decoder/decoder-with-past), each with its own
inline copy of whatever weights it uses — e.g. the decoder's weights end up
duplicated inline across both `decoder_model.onnx` and
`decoder_with_past_model.onnx` — and every pass in onnxsim's own optimization
pipeline that touches a graph copies those inline bytes along with it. Keeping
every graph's large tensors on disk from the start means both the in-memory
copying during simplification and the inline duplication across split files
shrink to metadata (name/offset/length) instead of the tensors themselves —
worth it even for the tiny example above, which now writes e.g.
`encoder_model.onnx` + `encoder_model.onnx.data` side by side. Pass
`save_as_external_data=False` to keep a small/toy checkpoint as a single
self-contained `.onnx` file instead:

```python
onnxsim.export_transformers_model(
    "hf-internal-testing/tiny-random-t5",
    "exported_and_simplified",
    task="text2text-generation-with-past",
    save_as_external_data=False,
)
```

## Diffusion model export

A Hugging Face `diffusers` pipeline (Stable Diffusion, SDXL, ...) isn't one
model but several — typically a text encoder, a UNet (or transformer)
denoiser, and a VAE encoder/decoder, each its own graph — driven by a
`model_index.json` plus per-component subdirectories rather than a single
`config.json`. As with a plain `transformers` model, onnxsim has no PyTorch
tracing code of its own to turn that into ONNX graphs, and does not need any:
[`optimum`](https://github.com/huggingface/optimum)'s own
`optimum.exporters.onnx.main_export` (the same call `optimum-cli export onnx`
makes) already detects a diffusers pipeline and exports every sub-model into
its own `<component>/model.onnx` — `text_encoder/model.onnx`,
`unet/model.onnx`, `vae_encoder/model.onnx`, `vae_decoder/model.onnx`, plus
e.g. `text_encoder_2/model.onnx` for SDXL. That export is plain, un-fused
ONNX, so there is real simplification left for onnxsim to find, exactly as
for a transformers export.

`onnxsim.export_diffusion_model()` wraps the export-with-optimum,
simplify-every-graph-in-place recipe as one call — the diffusion counterpart
of `onnxsim.export_transformers_model()`:

```python
import onnxsim

results = onnxsim.export_diffusion_model(
    "hf-internal-testing/tiny-stable-diffusion-torch",
    "exported_and_simplified",
)
# {"text_encoder/model.onnx": True, "unet/model.onnx": True,
#  "vae_encoder/model.onnx": True, "vae_decoder/model.onnx": True}
```

`output_dir` ends up holding the same layout a plain `optimum` export would
(`model_index.json`, `scheduler/`, `tokenizer/`, each component's
`config.json`, ...), except every `<component>/model.onnx` has been
simplified in place — so the directory is still deployable exactly as-is,
e.g. via `optimum.onnxruntime.ORTStableDiffusionPipeline.from_pretrained`.
Needs the optional `torch`/`diffusers`/`optimum` (with the `optimum-onnx`
distribution) packages: `pip install onnxsim[diffusion]`.

Like `export_transformers_model`, every simplified graph is saved with its
weights in a companion `.data` file by default (`save_as_external_data=True`)
rather than inline — a real (non-tiny) diffusion export's UNet routinely
exceeds `onnx.save`'s 2GB inline limit on its own, and every pass in
onnxsim's own optimization pipeline that touches a graph copies its inline
bytes along with it. Pass `save_as_external_data=False` to keep a small/toy
pipeline as self-contained `.onnx` files instead:

```python
onnxsim.export_diffusion_model(
    "hf-internal-testing/tiny-stable-diffusion-torch",
    "exported_and_simplified",
    save_as_external_data=False,
)
```

## Detectron2 export

A [Detectron2](https://github.com/facebookresearch/detectron2) model (Faster/
Mask/Keypoint R-CNN, RetinaNet, ...) has no `optimum`-style single-call ONNX
exporter to build on: Detectron2's own supported recipe is
`detectron2.export.TracingAdapter` wrapped around `torch.onnx.export` -- the
same steps Detectron2's own `tools/deploy/export_model.py` runs by hand,
because the model's `forward()` takes/returns Python dicts and `Instances`
objects that neither `torch.jit.trace` nor `torch.onnx.export` understand
directly. That export is plain, un-fused ONNX, so there is real
simplification left for onnxsim to find, exactly as for a transformers or
diffusion export.

`onnxsim.export_detectron_model()` wraps the build-model, trace, export
recipe and feeds the result straight into `onnxsim.simplify()`:

```python
import onnxsim

onnxsim.export_detectron_model(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "faster_rcnn_simplified.onnx",
)
```

`config_file` is either a path to a Detectron2 YAML config or, as above, the
name of one of Detectron2's built-in model zoo configs (resolved via
`detectron2.model_zoo.get_config_file`). Pass `weights=` a checkpoint path
or URL to trace with pretrained weights (e.g.
`detectron2.model_zoo.get_checkpoint_url(config_file)` for a model zoo
config's official COCO weights) -- if omitted (the default), no checkpoint
is loaded at all, regardless of what `cfg.MODEL.WEIGHTS` the config itself
specifies (most model zoo configs default it to an ImageNet-pretrained
backbone URL, which would otherwise mean a surprise network fetch on every
call), and the model traces with its random initialization instead. Pass
`image=` a sample image path (or an already-loaded array) to trace with
instead of the default synthetic random image.

Needs the optional `torch` package (`pip install onnxsim[detectron2]`).
`detectron2` itself is not published to PyPI, so it must be installed
separately from source, e.g.
`pip install 'git+https://github.com/facebookresearch/detectron2.git'` (see
[Detectron2's install docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)).

## SAM 2 export

Meta's [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything
2) has the same problem as Detectron2: no single-call ONNX exporter, and a
`forward()` that isn't directly traceable end-to-end -- image embedding and
prompt-driven mask decoding are two separate stages meant to be called many
times per embedding (one embedding, many point/box prompts). The
established recipe (Meta's own original SAM `scripts/export_onnx_model.py`,
carried over to SAM 2 by the community) is two separate traced graphs
instead: an image encoder, and a decoder wrapping the prompt encoder plus
mask decoder.

[`samexporter`](https://github.com/vietanhdev/samexporter) already
implements and maintains that split for SAM/SAM2/SAM3 -- and its own CLI
already calls `onnxsim.simplify()` under its `--simplify` flag, so onnxsim
already sits downstream of it for real users. `onnxsim.export_sam2_model()`
wraps that same build-model/trace/export sequence as a reusable entry point
that always simplifies, the SAM 2 counterpart of
`onnxsim.export_detectron_model()`:

```python
import onnxsim

onnxsim.export_sam2_model(
    "sam2.1_hiera_tiny",
    "sam2_exported",
)
# {"encoder.onnx": True, "decoder.onnx": True}
```

`model_type` selects one of `samexporter`'s bundled Hydra model configs
(`"sam2.1_hiera_tiny"`, `"sam2.1_hiera_small"`, `"sam2.1_hiera_base_plus"`,
`"sam2.1_hiera_large"`, or the plain `"sam2_hiera_*"` names for the original
SAM 2 release). Pass `checkpoint=` a path to a `.pt` checkpoint to trace
with real weights -- if omitted, the model traces with its random
initialization instead, with no checkpoint or network needed at all, e.g.
for testing.

Needs the optional `torch`, `samexporter`, and `hydra-core` packages
(`pip install onnxsim[sam2]`). The real `sam2` package itself is **not**
published to PyPI under its official name -- install it from source:
`pip install 'git+https://github.com/facebookresearch/sam2.git'` (see
[SAM 2's install docs](https://github.com/facebookresearch/sam2#installation)).
A same-named `sam2` package that *is* on PyPI is an unrelated, unofficial
third-party upload as of this writing, not Meta's code -- don't substitute
it for this.

## Quantization-aware fine-tuning

onnxsim's post-training quantization passes stop at *rounding*: AdaRound,
BRECQ, FlexRound, AutoRound and the rest each pick, for every weight, which of
the two neighbouring integers it rounds to, and nothing more.
`onnxsim.apply_qat()` goes one step further -- the fp32 weight itself is the
trained parameter, fake-quantized in the forward against
`quantize_weight_only_int4`'s block-wise INT4 grid with a straight-through
estimator, so an element can migrate several codes away from where
round-to-nearest put it.

It is **label-free**. The float model is the teacher, and the loss is the
reconstruction error of one block's output against the float model's own
activations for that block. There are no labels, no dataset API, no metric and
no training-loop lifecycle -- the lifecycle stays the one onnxsim already has,
a model in and a model out. Task-loss QAT remains out of scope; the design
note is [`docs/qat.md`](docs/qat.md), which records both what was built and
where the boundary is drawn.

Two things here are genuinely new:

- **Any block topology, not just a linear chain.** `apply_brecq` is capped at
  a strict chain of MatMul/Gemm layers, because every op shape between two
  quantized layers used to mean another hand-derived backward pass.
  `onnxsim/graph_grad.py`'s `build_backward` removed that cost: it
  differentiates a slice of an ONNX graph by walking it in reverse and
  emitting ordinary ONNX nodes, so a normalization, an activation, a GELU's
  `Erf`, a Softmax or a residual between two Linears is now simply more nodes
  in the slice. A block that `apply_brecq` returns byte-identical, because its
  discovery cannot see that topology at all, can be reconstructed here.
- **The training step runs on an accelerator.** The fake-quant forward, that
  backward, and one Adam update per trained tensor are emitted as a single
  ONNX *step graph* -- a pure `(constants, state, per-step scalars) -> (next
  state, loss)` function (`onnxsim/qat_graph.py`) -- so the loop runs on
  whatever execution providers `step_providers=` names: a GPU, an NPU
  execution provider, or WebGPU in the WASM build, rather than in host numpy.
  The parts onnxsim emits -- the fake-quant, the backward, the optimizer --
  stay inside `qat_graph.EP_FRIENDLY_OPS`, a deliberately small operator set;
  the block's own forward nodes are copied in as they are, so whether a
  particular block's step graph runs on a particular accelerator also depends
  on that backend's coverage of the operators the block itself contains.
  `onnxsim.backend.Runner` keeps the parameters and the optimizer state
  resident on the provider's device between steps, so only the per-step
  scalars go up and the loss comes down.

A block is named by its input and output tensor, exactly the way
`apply_brecq` names one:

```python
import onnx
import onnxsim

float_model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_int4(float_model)

# Representative input batches: {input_name: np.ndarray} dicts matching the
# float model's graph inputs. Random data is the default when omitted; real
# data (onnxsim.load_huggingface_calibration_data) is a far better target.
calibration_data = [{"input": batch} for batch in batches]

losses = []  # the reconstruction loss, appended once per step
tuned = onnxsim.apply_qat(
    float_model,
    quantized,
    block_input_name="input",
    block_output_name="block_out",
    calibration_data=calibration_data,
    losses=losses,
    step_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

onnx.save(tuned, "model_int4_qat.onnx")
```

Only that block's INT4 weight initializers are rewritten; every other byte of
the quantized model is untouched. One block is trained per call, so a sliding
window over a model's blocks is the caller's own loop to write. Naming a block
that cannot be trained is a loud error rather than a silently unchanged model:
an op `graph_grad` has no gradient rule for, a block containing no
`quantize_weight_only_int4`-quantized MatMul/Gemm, or shapes that cannot be
inferred statically are all refused before any calibration runs.
`learn_scales=True` trains each weight's per-block quantization scale
alongside, LSQ-style (off by default, since it makes the problem non-convex in
two coupled parameter sets at once). The whole calibration set is one
full-batch objective by default -- `batch_size=` opts into minibatching, but
the budget stays a calibration-scale one either way.

By default this targets `quantize_weight_only_int4`'s weight-only scheme.
`learn_activation_scales=True` instead targets `quantize_static`'s QDQ scheme
-- uint8 asymmetric activations, per-output-channel INT8 weights -- and trains
each layer's activation `(scale, zero_point)` jointly with its weights. The
flag necessarily *selects a scheme* rather than adding a feature to the other
one: a weight-only model has no activation quantizer anywhere to train, and
inserting one would invent a W4A8 model no `quantize_*` function here emits.
Asking for the wrong pairing raises with the mismatch named, in both
directions. `activation_learning_rate=` (default `1e-2`) tunes it.

### When to reach for it, and when not to

It is **not** uniformly better than `apply_adaround`, and the boundary is worth
knowing before you spend a thousand Adam steps. On a two-Linear-plus-`Relu`
block -- a topology `apply_brecq` cannot reconstruct at all -- the
reconstruction error falls from 16.0 (round-to-nearest) to 6.5, and a GELU
block's loss falls ~12x. But on a *single* layer, where the objective is
identical to AdaRound's and only the parametrization differs, freeing the
weight wins when the calibration activations are low-rank (round-to-nearest
5.96, AdaRound 3.07, `apply_qat` 1.78) and **loses** when they are full-rank
(28.5, 14.6, 22.8). The reason is not subtle: a well-determined reconstruction
problem has its optimum within one quantization step of round-to-nearest, so
floor/ceil is all the freedom worth having there, and AdaRound's continuous
relaxation optimizes that restricted problem better than a hard
straight-through estimator does. Real activations are widely observed to be
close to low-rank -- the premise the whole reconstruction-based PTQ literature
leans on -- which is the case for having this at all, though onnxsim has not
measured that on your model, and "QAT beats AdaRound" is not a claim it
makes. Both directions are measured in `tests/test_qat.py`, not
assumed.

`learn_activation_scales` has a boundary of its own, and it is the same shape.
On a block whose activation range was calibrated from one unrepresentative
outlier -- ~30x too wide, so activation quantization is the binding constraint
-- training the quantizers alongside the weights takes the whole-model output
error from 4.08 (weights alone) to 2.54, 38% better, on all three seeds tried.
On the *same* block calibrated on representative data it is a small regression
at every learning rate tried: min/max on representative data is already close
to MSE-optimal, so there is little left for a learned clip range to find, and a
second coupled parameter group makes a solved problem harder rather than a hard
one easier. It is a fix for a quantizer whose range is *wrong*, not a free
improvement on one whose range is right --- and when re-calibrating is
available at all, that costs one forward pass rather than a training budget.

So: reach for `apply_qat` for a block no rounding pass here can reconstruct (a
normalization, an activation, a GELU, a residual in the middle of it), and for
low-rank calibration activations at 4 bits. Reach for
`learn_activation_scales=True` when a `quantize_static` model's activation
ranges are the binding constraint and re-calibrating them is not an option.
Stay with `apply_adaround` for a single well-conditioned layer.

### Running AdaRound, AdaQuant and AutoRound on an accelerator

The step-graph machinery is not QAT-only. `apply_adaround`, `apply_adaquant`
and `apply_autoround` take the same `step_providers=` argument, which runs
their optimization loop as an ONNX step graph instead of in host numpy:

```python
tuned = onnxsim.apply_adaround(
    float_model,
    quantized_model,
    calibration_data=batches,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    step_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

`providers=` (unchanged) still selects where the float model's calibration
activations are captured; `step_providers=` selects where the optimization
itself runs. Omitting it keeps the existing float64 numpy loop, which is what
CI runs: it is exact and reproducible, and a non-CPU provider is neither, since
GPU/NPU reductions reassociate. `apply_qat` has no numpy alternative -- the
step graph *is* the implementation there -- so `step_providers=None` simply
means CPU.

## Projects Using ONNX Simplifier

ONNX Simplifier is most often used as a post-export cleanup step, run on a
freshly exported ONNX graph before it is handed to a mobile / edge / accelerator
runtime converter. Projects that actively use it in their current export or
conversion tooling include:

* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Megvii) — the ONNX export runs onnxsim by default (`--no-onnxsim` to disable)
* [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (PaddlePaddle) — simplifies exported detectors (PP-YOLOE, PicoDet, RT-DETR, …) with onnxsim before deployment
* [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) (PaddlePaddle) — runs `onnxsim.simplify` in its ONNX → Paddle conversion optimizer
* [ncnn](https://github.com/Tencent/ncnn) (Tencent) — recommends simplifying with onnxsim before `onnx2ncnn`
* [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) (Rockchip) — runs onnxsim in its ONNX export scripts before RKNN conversion

## Chat

We created a Chinese QQ group for ONNX!

ONNX QQ Group (Chinese): 1021964010, verification code: nndab. Welcome to join!

For English users, I'm active on the [ONNX Slack](https://github.com/onnx/onnx#discuss). You can find and chat with me (daquexian) there.
