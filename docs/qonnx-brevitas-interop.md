# QONNX/Brevitas interop

## What this is

Two independent, additive pieces of support for models exported from
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
inference about the four ops so simplification isn't blocked by them, and (2)
teaching `onnxsim.qat_interop` (deliverable A of `docs/qat.md`, "QAT interop")
to recognize a `Quant` node as a fourth shape of already-learned quantizer,
alongside the three QDQ shapes it already understood -- so a Brevitas-trained
model's learned scales/zero-points survive `quantize_static_keeping_qdq_scales`
exactly like a QDQ-exported QAT model's do.

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

- **Covered:** an 8-bit, per-tensor `Quant` node with constant
  `scale`/`zeropoint`/`bitwidth` -- `signed=1` maps onto onnxsim's own
  symmetric int8 weight scheme, `signed=0` onto its uint8 activation scheme,
  exactly like an ordinary QDQ pair with that zero-point dtype. From there
  on, a QONNX-sourced learned quantizer and a QDQ-sourced one are
  indistinguishable to the rest of the pipeline
  (`quantize_static_keeping_qdq_scales`'s writeback, `strip_existing_qdq`'s
  canonicalization) -- both are just "this float tensor's learned `(scale,
  zero_point)`" by the time detection is done.
- **Not covered, deliberately:** per-channel `Quant` scale (QONNX carries no
  `axis` attribute to check a broadcast shape's claim against, unlike
  `DequantizeLinear`), bit-widths other than 8 (the case QONNX exists for,
  and the one onnxsim's own 8-bit-only schemes have no counterpart for), and
  `BipolarQuant`/`Trunc`/`FloatQuant` (recognized so a model using them isn't
  silently mis-scanned as an ordinary float graph, but not yet
  canonicalized -- none of onnxsim's own schemes are 1-bit, truncating, or
  float-grid). Each has its own `SKIP_REASONS` entry
  (`qonnx_per_channel_scale_unsupported`, `qonnx_bitwidth_unsupported`,
  `qonnx_op_unsupported`) and is left in the graph exactly as exported,
  never guessed at -- the same "refuse rather than guess" policy the rest of
  `qat_interop.py` already applies to QDQ.

## Tests

- `tests/test_qonnx_custom_op_schemas.py` -- proves shape inference actually
  runs the registered schemas (a `Shape`/`Gather` chain past each op folds to
  a literal), for both domains.
- `tests/test_qat_interop.py`'s "QONNX/Brevitas ingest" section -- detection,
  activation/weight preservation through `quantize_static_keeping_qdq_scales`,
  canonicalization through `strip_existing_qdq`, and each refusal reason.

## Not covered

- **Per-channel weight quantization.** This is the common case for a real
  Brevitas `QuantConv2d`/`QuantLinear` export, so it is the biggest practical
  gap left: such a weight is recognized (detection succeeds) but reported as
  `qonnx_per_channel_scale_unsupported` rather than preserved, falling back
  to onnxsim's own round-to-nearest per-channel scale -- accurate, but not
  the trained one.
- **Sub-8-bit and super-8-bit precision** -- the entire reason QONNX's
  `bitwidth` is a runtime input rather than an implied dtype. A model
  actually using this (rather than defaulting to 8-bit) falls back to
  calibration for that tensor.
- **`BipolarQuant`/`Trunc`/`FloatQuant` canonicalization.** Recognized at the
  schema level (shape inference) and at the ingest-detection level (a
  reported, not silent, refusal), but there is no onnxsim quantization
  scheme yet for any of the three to round-trip into.
- **Egress** (`export_fake_quant`) still only emits standard QDQ; there is no
  QONNX-emitting counterpart. An external Brevitas-side trainer round-trips
  through ordinary QDQ, same as any other QDQ-based QAT trainer.
- **Brevitas's other export paths** (`export_onnx_qcdq`'s
  `QuantizeLinear`-`Clip`-`DequantizeLinear`, `export_onnx_qop`'s
  `com.microsoft` QOperator ops) already worked before this change: the
  first is ordinary QDQ plus a `Clip` node quantize_static already leaves
  alone; the second was already covered by `contrib_schemas.cpp`'s
  `com.microsoft` schemas.
