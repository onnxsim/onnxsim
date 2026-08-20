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
  `--skip-optimization [pass ...]`.
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
- **Subgraph simplification.** Simplify `If`/`Loop`/`Scan` subgraph bodies too
  with `--include-subgraph`.
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

## Loading large models

Passing `simplify()` a **file path** (rather than an already-loaded
`ModelProto`) is the recommended way to simplify a large model: it lets
onnxsim defer loading a model's (potentially huge) weights for as long as
possible, and is the input this project's own tooling and tests default to.
When the model's weights live in a separate classic external-data file (the
`<name>.onnx` + `<name>.data` pair `onnx.save(..., save_as_external_data=True)`
produces), that file is memory-mapped straight into onnxsim's C++
`TensorPool` rather than being read into a heap buffer first, avoiding an
extra buffered copy of the weights on the way in.

The same loading path is available directly as `onnxsim.load_model`, for
when you want the loaded `ModelProto` itself rather than passing a path
straight to `simplify()`:

```python
import onnxsim

model = onnxsim.load_model("model.onnx")  # instead of onnx.load("model.onnx")
sim_model, check_ok = onnxsim.simplify(model)
```

`onnxsim.load_model(path)` is functionally identical to `onnx.load(path)` —
a fully self-contained `ModelProto`, with no `TensorPool` or lazy state for
the caller to manage — just with fewer copies of the weights along the way
for a model that uses external data.

`simplify(path)` itself goes further: any single external-data tensor over
1.5GB is left un-hydrated (a lazy, memory-mapped reference, not resident in
memory) for as long as the simplification fixed point never actually needs
its value — every onnxsim/onnx-optimizer pass that reads a constant tensor's
value already treats such a tensor as unavailable rather than misreading its
absent bytes as zero, so this only ever costs that one tensor's
value-dependent optimizations (constant folding, BatchNorm fusion, duplicate-
initializer dedup, ...), never correctness. A tensor dropped by dead-code
elimination before ever being read pays no hydration cost at all. Whatever
is still un-hydrated when simplification finishes is fully materialized
before the result is saved, so the output file is always an ordinary,
self-contained model either way.

## Safetensors / GGUF archives

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
