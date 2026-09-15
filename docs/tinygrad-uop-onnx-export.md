# Representing a tinygrad UOp graph in ONNX, round-trippable (experimental)

**Status: experimental.** `onnxsim.tinygrad_uop_export` exports a real
tinygrad `UOp` graph -- the same per-kernel `Ops.SINK`-rooted AST
`onnxsim.webgpu_tinygrad_codegen` renders to WGSL (see
`docs/webgpu-kernel-dispatch.md`) -- as ONNX, under a private, non-standard
domain (`"tinygrad.uop"`), **and imports it back** into a real `UOp` graph.
Neither export shape is runnable by any ONNX runtime: no runtime implements
`ALU`/`RANGE`/`BUFFER`/`REDUCE`/... as tensor ops. This exists so a `UOp`
graph can be inspected through ordinary ONNX tooling instead of tinygrad's
own `VIZ=1` debug server, and durably stored/exchanged as plain protobuf
instead of a pickle -- and, unlike a pickle, safely and faithfully
reconstructed back into a real graph tinygrad can render again.

## Two export shapes

- **`uop_to_onnx_model(root)`** -- a standalone `ModelProto`: one flat
  graph, every `UOp` a node, synthetic `u<i>` names throughout (including
  the graph's own single output). Inverse: `onnx_model_to_uop(model)`.
- **`uop_to_onnx_function(root, name, inputs)`** -- the same nodes, but
  wrapped as an ONNX **local function** (`FunctionProto`, referenced from
  `model.functions`) with a real I/O boundary instead of synthetic names
  throughout: `inputs` maps specific leaf `UOp`s (typically a kernel's own
  `PARAM` nodes) to real names, which become the function's own formal
  parameters -- exactly like a Python function's parameters, they get *no*
  defining node in the function body (a name can't be both "supplied by the
  caller" and "produced internally"), so the body holds only the non-leaf
  `UOp`s. A model can then reference the function from an ordinary
  top-level graph via one `NodeProto` (`domain="tinygrad.uop",
  op_type=name`). Netron has a dedicated "expand function" feature for
  exactly this shape: a reader sees the ordinary-looking op first and can
  drill into its UOp-level decomposition on demand, instead of only ever
  seeing one giant flat graph of raw UOps. Inverse:
  `onnx_function_to_uop(function, inputs)`, where `inputs` (required) binds
  real `UOp`s to the function's own formal parameter names.

## Why bother, given it can't run

tinygrad already has its own UOp graph export/inspection story: `VIZ=1`
records every graph-rewrite step as a `RewriteTrace` dataclass and pickles
it to a temp file (`tinygrad/viz/serve.py`), then serves a bundled
d3.js/dagre web UI that converts a `UOp` to a JSON node/edge structure
(`uop_to_json`) on the fly for rendering. This module trades that for:

- **A stable, safe container that round-trips.** A pickle is a live Python
  object graph frozen to disk -- unpickling runs arbitrary code, and the
  format silently breaks across tinygrad versions whenever a pickled
  class's shape changes (`UOp`, `ShapeTracker`, `KernelInfo`, ...). A
  `.onnx` file is plain protobuf: safe to open from anywhere, and --
  unlike a pickle -- actually **importable back** into a real `UOp` graph
  by this same module, with no arbitrary code execution involved at any
  point. Decoding is a small whitelisted dispatch over JSON (an explicit
  type tag this module itself writes), never `eval`/`pickle.loads`, so an
  untrusted file can only make it raise, never execute anything.
- **Free generic tooling.** Netron (and any other protobuf/ONNX-aware
  viewer) renders a custom, unrecognized domain's nodes and edges
  generically, with the function-expansion behavior described above.

## Encoding `UOp.arg`, faithfully, for a whitelisted set of shapes

`UOp.arg`'s type varies wildly by op: a plain `int`/`float`/`bool` for
`CONST`, a `ConstFloat` (a tagged `float` subclass) also for `CONST`, an
`(int, AxisType)` pair for `RANGE`, an `(Ops, int)` pair for `REDUCE`, a
`ParamArg` dataclass (itself holding a `DType` and an `AddrSpace` enum
member) for `PARAM`, a `KernelInfo` dataclass for `SINK`, or plain `None`
for most ALU/`INDEX`/`STORE`/`END` nodes. `_encode_value`/`_decode_value`
handle exactly this whitelisted set -- recursively, as a single JSON-valued
`arg_json` string attribute (tagged by Python type, e.g. `{"t": "AxisType",
"name": "WEAK"}`) -- and raise a clear `TypeError`/`ValueError` for
anything outside it, rather than silently guessing or falling back to an
unparseable `repr()`.

Two known, deliberate gaps, both raising clearly rather than mis-encoding
or silently losing information:

- A pre-schedule `UOp` graph (movement ops with a `ShapeTracker`/`View`
  arg) is out of scope -- this module targets the *post-schedule*
  per-kernel AST `_lower_tensor_program` itself works with, where movement
  ops have already been lowered into index arithmetic.
- `KernelInfo.applied_opts`/`opts_to_apply`/`estimates` are only supported
  in their default (empty/`None`) state -- a BEAM-search-tuned kernel's
  exact tuning is not reconstructed. `name`, `axis_types`,
  `dont_use_locals`, and `beam` always round-trip.

## Verified round trip

`tests/test_tinygrad_uop_export.py` doesn't just check structural equality
after import -- for *both* export shapes, it re-renders the *reconstructed*
graph through the exact same `to_program`/`WGSLRenderer` pipeline
`onnxsim.webgpu_tinygrad_codegen` uses, for a real `Conv2D`+`Relu` kernel
AST (fused by tinygrad's own scheduler into one kernel), and asserts the
WGSL text is **byte-identical** to rendering the original. That is the
actual bar for "faithful": everything the real codegen path reads off a
`UOp` survives the round trip, not just what a human eyeballing the graph
would notice missing.

## Scope

Exports/imports exactly the `UOp` DAG reachable from one root (typically an
`Ops.SINK`-rooted per-kernel AST, the same one
`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program` passes to
`to_program`) -- not a whole multi-kernel schedule (`schedule_linear()`'s
own result, which threads several such ASTs together via `Ops.CALL` nodes).
Handling one of those is just calling this once per kernel AST; stitching
multiple exported graphs/functions into one file isn't done here.

A kernel's own `PARAM` leaves carry no real-world tensor identity by
themselves (just an argument-slot number) -- knowing "this `PARAM` is
really the model's `x` input" requires the *surrounding* schedule context
(the same buffer-identity matching `onnxsim.webgpu_tinygrad_codegen`'s own
`_lower_tensor_program` already does against a node's real input/output
names). `uop_to_onnx_function`'s `inputs` parameter takes real `UOp` object
references rather than trying to guess this itself, so a caller who has
that context (e.g. code building on `_lower_tensor_program`) supplies the
real names directly; on its own, this module only knows a leaf's slot
number.

## Testing

`tests/test_tinygrad_uop_export.py` builds a real per-kernel AST from a
small `Conv2D`+`Relu` (tagged for the `WEBGPU` device, never actually
opened -- same reasoning as `onnxsim.webgpu_tinygrad_codegen`'s own tests)
and checks:

- `onnx.checker.check_model` accepts both the flat-model export and a model
  embedding the function export with a calling node.
- Node count/`op_type`/inputs/`dtype` attribute match the `UOp` toposort
  exactly for the flat export (not just "some graph got produced").
- For the function export: every named leaf becomes a formal parameter
  with no defining node in the body (the name-collision rule above), and
  `onnx_function_to_uop` raises `ValueError` if not given exactly the
  function's own formal parameters.
- **Both** export shapes round-trip to byte-identical WGSL after
  reconstruction.
- Encoding an out-of-whitelist `arg` type raises `TypeError` rather than
  silently mis-encoding.

Skipped when `tinygrad` isn't installed (`pytest.importorskip("tinygrad")`),
matching `tests/test_webgpu_tinygrad_codegen.py`'s own convention.
