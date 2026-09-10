# QONNX/Brevitas interop

## What this is

Three independent, additive pieces of support for models exported from
[Brevitas](https://github.com/Xilinx/brevitas) (the PyTorch
quantization-aware-training library FINN/Xilinx deployments are built on),
whose *default* ONNX export path (`brevitas.export.export_qonnx`) does not
use standard `QuantizeLinear`/`DequantizeLinear` at all. It emits
[QONNX](https://github.com/fastmachinelearning/qonnx)'s own generic
fake-quantization custom ops instead: `Quant`, `BipolarQuant`, `Trunc`, and
(for minifloat grids) `FloatQuant`, in the `qonnx.custom_op.general` domain
(or its predecessor, `finn.custom_op.general`, from older Brevitas/FINN
versions). Without support for these four ops, onnxsim treats every one of
them as an opaque, unknown custom op: no shape inference propagates past a
`Quant` node, and nothing downstream of one can be constant-folded, fused, or
otherwise simplified.

Like `docs/dynamic-quantization.md` says of its own scope: this is not a
from-scratch reimplementation of Brevitas or QONNX -- those are large,
independent projects. What's here is (1) teaching onnxsim's own shape
inference about the four ops so simplification isn't blocked by them, (2)
teaching `onnxsim.qat_interop` (deliverable A of `docs/qat.md`, "QAT interop")
to recognize a `Quant` node as a fourth shape of already-learned quantizer,
alongside the three QDQ shapes it already understood -- so a Brevitas-trained
model's learned scales/zero-points survive `quantize_static_keeping_qdq_scales`
exactly like a QDQ-exported QAT model's do -- and (3) the reverse direction,
emitting `Quant` nodes for an external (Brevitas-side) trainer to fine-tune.

## 1. Shape inference (`onnxsim/qonnx_schemas.cpp`)

`RegisterQonnxCustomOpSchemas()` registers a schema for each of `Quant`,
`BipolarQuant`, `Trunc`, `FloatQuant`, in both `qonnx.custom_op.general` and
`finn.custom_op.general`, run automatically by every simplification entry
point (no opt-in needed) -- the same mechanism `contrib_schemas.cpp` uses for
ONNX Runtime's `com.microsoft` ops and `bev_custom_op_schemas.cpp` uses for
mmdeploy/mmcv/BEVDet ops.

All four ops share one contract: "fake-quantize the first input, then
immediately dequantize the result back to that input's own dtype" -- so the
output is always shaped, and typed, exactly like the first input
(`Quant`/`BipolarQuant`/`Trunc`/`FloatQuant`'s own `X`). Each schema's
`TypeAndShapeInferenceFunction` is `propagateShapeAndTypeFromFirstInput`,
the same helper `com.microsoft`'s `QLinearSigmoid`/`QLinearLeakyRelu`/
`QLinearSoftmax` schemas already use for an analogous "output looks like the
input" contract.

This intentionally goes no further than shape inference. `QuantizeLinear`/
`DequantizeLinear` are explicitly excluded from onnxsim's constant folder
even when their inputs are constant (`IsQDQ` in `constant_folding.cpp`), so
the quantization boundary survives simplification instead of collapsing into
a plain float constant. `Quant`/`BipolarQuant`/`Trunc`/`FloatQuant` get the
same outcome for a different, simpler reason: `IsOfficialOp` only recognizes
the default ONNX domain, so a node in a custom domain is never a
constant-folding candidate at all, regardless of `IsQDQ`.

Covers: any model containing these four ops, regardless of what produced it
-- shape inference doesn't care who authored the node. In practice this is
overwhelmingly Brevitas/FINN exports, since QONNX itself is that ecosystem's
interchange format.

## 2. QAT ingest (`onnxsim/qat_interop.py`)

`find_existing_qdq` now recognizes a `Quant` node as a fourth pattern shape,
which `onnxsim.qat_interop`'s own module docstring documents in full under
"QONNX/Brevitas ingest". The short version: a `Quant` node's first input
already *is* the float tensor (there's no separate integer tensor or
producing `QuantizeLinear` to look for, unlike a QDQ pair), and its output is
already the fake-quantized-and-dequantized result, so recognizing one is
simpler than the QDQ case -- at the cost of `Quant`'s own extra constraints:

- **Covered:** an 8- or 16-bit `Quant` node with constant
  `scale`/`zeropoint`/`bitwidth` -- `signed` and `bitwidth` together select
  which of onnxsim's own schemes it maps to (8-bit signed: its symmetric
  int8 weight scheme; 8-bit unsigned: `quantize_static`'s uint8 activation
  scheme; 16-bit unsigned: `quantize_static_int16`'s uint16 one -- a
  *weight* is always carried as 8-bit signed regardless of the activation
  scheme in play, the one scheme onnxsim's weight quantization has). The
  scale may be per-tensor for either role, or -- for a **weight** only --
  single-axis per-channel: `_infer_weight_axis` reads the broadcast axis
  straight off the scale's own shape against the weight's real shape, since
  a valid `Quant` node's scale must already broadcast correctly against the
  tensor it multiplies at runtime -- no `axis` attribute needed, and no
  guessing. From there on, a QONNX-sourced learned quantizer and a
  QDQ-sourced one are indistinguishable to the rest of the pipeline
  (`quantize_static_keeping_qdq_scales`'s writeback, `strip_existing_qdq`'s
  canonicalization) -- both are just "this float tensor's learned `(scale,
  zero_point)`" by the time detection is done.
- **Not covered, deliberately:** a `Quant` scale broadcasting over more than
  one axis (a per-block scale, no onnxsim counterpart) or belonging to an
  activation (whose shape isn't known statically here); bit-widths other
  than 8/16 (the arbitrary-precision case QONNX exists for in the first
  place, and the one onnxsim's own integer schemes have no counterpart for);
  and `BipolarQuant`/`Trunc`/`FloatQuant` (recognized so a model using them
  isn't silently mis-scanned as an ordinary float graph, but not yet
  canonicalized -- none of onnxsim's own schemes are 1-bit, truncating, or
  float-grid). Each has its own `SKIP_REASONS` entry
  (`qonnx_per_channel_scale_unsupported`, `qonnx_bitwidth_unsupported`,
  `qonnx_op_unsupported`) and is left in the graph exactly as exported,
  never guessed at -- the same "refuse rather than guess" policy the rest of
  `qat_interop.py` already applies to QDQ.

## 3. QAT egress (`onnxsim/qat_interop.py`)

`export_fake_quant_qonnx` is the QONNX-emitting counterpart of
`export_fake_quant`: it runs the exact same calibration/rewrite/naming
`export_fake_quant` already does, then `_rewrite_qdq_to_quant` replaces each
emitted `QuantizeLinear`/`DequantizeLinear` pair (or lone weight
`DequantizeLinear`) with an equivalent `Quant` node in place -- same
position in the node list, since a `Quant` node's own `X` input may be an
intermediate activation an *earlier* node in the graph produces, so simply
appending the new nodes at the end would leave the graph topologically
invalid. `learnable_scales` names are unaffected (`Quant`'s `scale` input is
the same tensor, same convention, as a QDQ pair's own); `learnable_zero_points`
names are remapped, since QONNX's zero-point is always a float tensor rather
than an integer one of the target dtype.

A model this produces, trained externally, and then handed to
`quantize_static_keeping_qdq_scales` recovers exactly the trained parameters
-- the two halves are tested together
(`test_round_trip_through_an_external_trainer_qonnx`), not just each in
isolation against plain QDQ.

## Tests

- `tests/test_qonnx_custom_op_schemas.py` -- proves shape inference actually
  runs the registered schemas (a `Shape`/`Gather` chain past each op folds to
  a literal), for both domains.
- `tests/test_qat_interop.py`'s "QONNX/Brevitas ingest" section -- detection
  (including per-channel weight scale and 8-/16-bit), activation/weight
  preservation through `quantize_static_keeping_qdq_scales`, canonicalization
  through `strip_existing_qdq`, each refusal reason, `export_fake_quant_qonnx`,
  and the egress/ingest round trip.

## Not covered

- **A `Quant` scale broadcasting over more than one axis** (a per-block
  scale) or an **activation's per-channel scale** -- QONNX carries no `axis`
  attribute, so a weight's own broadcast shape is the only thing this module
  can check a per-channel claim against; an activation has no static shape
  to check it against at all.
- **Bit-widths other than 8/16** -- the arbitrary/learned precision QONNX
  exists to support in the first place has no counterpart in onnxsim's own
  integer schemes. A model actually using this (rather than 8- or 16-bit)
  falls back to calibration for that tensor.
- **`BipolarQuant`/`Trunc`/`FloatQuant` canonicalization.** Recognized at the
  schema level (shape inference) and at the ingest-detection level (a
  reported, not silent, refusal), but there is no onnxsim quantization
  scheme yet for any of the three to round-trip into -- `quantize_fp8` (a
  whole-graph `Cast`-based scheme with no scale/zero-point at all) and
  `quantize_ternary` (a *structural*-ternary-weight detector) are both
  architecturally too different from the QDQ-preservation pattern this
  module is built around to be a natural target for `FloatQuant`/
  `BipolarQuant` respectively; a real port would need a new ingest pathway
  of its own, not just a new scheme mapping.
- **`apply_qat` does not understand `Quant` nodes directly.** The QAT
  *training* loop (`docs/qat.md`'s deliverable B, block-wise fine-tuning) has
  its own fake-quantizer/backward machinery
  (`onnxsim.graph_grad`/`onnxsim.qat_graph`) with no notion of a QONNX node.
  A Brevitas-exported model reaches it by going through ingest first
  (`Quant` nodes -> `quantize_static_keeping_qdq_scales` -> ordinary onnxsim
  QDQ -> `apply_qat` works on it like any other QDQ model), not by
  differentiating a raw `Quant`/`Trunc` node directly.
- **Brevitas's other export paths** (`export_onnx_qcdq`'s
  `QuantizeLinear`-`Clip`-`DequantizeLinear`, `export_onnx_qop`'s
  `com.microsoft` QOperator ops) already worked before this change: the
  first is ordinary QDQ plus a `Clip` node quantize_static already leaves
  alone; the second was already covered by `contrib_schemas.cpp`'s
  `com.microsoft` schemas.
