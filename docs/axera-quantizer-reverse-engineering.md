# Reverse-engineering Pulsar2's quantizer: what drives a tensor's range, and how to influence it

Scope: not the mcode instruction format (`docs/axera-mcode-quantize-elimination-probe.md`,
dead end for eliminating redundant quantize instructions) and not the weight-table
bit-permutation encoding (`scripts/axera/emitter.py`, already reverse-engineered) --
this is about the **PTQ calibration algorithm itself**: how Pulsar2 decides a
tensor's `(scale, zero_point)`, and whether that decision can be influenced from
inside its own sanctioned pipeline rather than only by patching a compiled binary
after the fact.

## Confirmed, from this session's own archival corpus: the range is driven by calibration data, not graph structure

This session's prior work (the loss=0 calibration-bug investigation, PR #1346) left
five real compiled builds of closely related graphs sitting in scratch --
`work_b1`, `work_b4`, `work_fixed_b4`, `work_matmul_b4`, `work_split_b4` -- each
with a real `quant/quant_axmodel.json` (Pulsar2's own per-tensor quantization
record: a `tensor_configs` map from node name to output-tensor `{bit_width,
policy, hash}`, and a `values` map from that `hash` to the actual `{scale,
zero_point}` Pulsar2 calibrated). Reading these directly, across the `loss_diff`
tensor (the squared-error term feeding the loss):

| build | batch | calibration data | `loss_diff` scale | zero_point |
| --- | --- | --- | --- | --- |
| `work_b1` | 1 | default | 0.137735 | 123 |
| `work_b4` | 4 | default (buggy one-hot placement, PR #1346) | 0.146161 | 145 |
| `work_fixed_b4` | 4 | **fixed** one-hot placement | 0.163908 | 132 |
| `work_matmul_b4` | 4 | same as `work_b4` | 0.146161 | 145 |
| `work_split_b4` | 4 | same as `work_b4` | 0.146161 | 145 |

`work_b4` and `work_fixed_b4` are the **same architecture and batch size**,
differing only in calibration data (PR #1346's bug: a one-hot label scattered
into the flattened tensor instead of placed per-row) -- and get **measurably
different** scale/zero-point. `work_matmul_b4` and `work_split_b4` are the
**same calibration data as `work_b4`** but a structurally different graph (a
different reduction spelling for the same arithmetic, from the redundant-quantize
investigation) -- and get **byte-identical** scale/zero-point to `work_b4`. Together
these isolate the variable: calibration data changes the range, graph structure
alone does not.

**Quantitative confirmation of the exact formula**: reran `work_b4`'s float
`step.onnx` over its own 4-sample calibration set (`onnxruntime`, all 7 real
inputs) and computed textbook asymmetric-MinMax `scale = (max - min) / 255`,
`zero_point = round(-min / scale)` directly from the resulting `loss_sq` output
values (a graph output, not just an internal node, so directly observable):
predicted **1.775640**, Pulsar2 recorded **1.769047** -- 0.37% apart, well
inside float32/aggregation-order noise. This is not merely "some data-driven
scheme" -- it is the textbook asymmetric MinMax formula, over the caller-supplied
calibration set, exactly as `pulsar2_quantizer.py`'s docstring already labeled it
(`calibration_method: MinMax`) but never numerically confirmed before now.

**The actionable conclusion**: the gradient-underflow ceiling
(`docs/axera-on-device-training-handoff.md`, "The ceiling: the gradient dies")
could plausibly be addressed by **recalibrating with late-training-scale
(small-magnitude) synthetic gradient values**, rather than whatever calibration
data has been used throughout this thread so far (which reflects early-training,
large-magnitude values) -- entirely within Pulsar2's own sanctioned pipeline, no
mcode patching, no `layer_configs` FP32 fighting. **Not yet tried against the
real ceiling** -- this document establishes the mechanism is real and
quantitatively textbook; a real build+train-past-the-old-ceiling test with
deliberately small-gradient calibration data is the natural, concrete next step,
and needs the AX650N/Docker toolchain (contended by a concurrent fork at the time
of writing).

## Already known, found in `scripts/axera/README.md` from a prior (LLM/RMSNorm) investigation -- cited, not re-derived

Three findings directly relevant here that predate and inform PR #1353's
FP32-seed plateau, found by searching this repo's own prior work rather than
re-deriving from scratch:

- **The exact `data_type: "FP32"` allowlist, from Pulsar2's own official docs**
  (`user_guides_advanced/advanced_build_guides.html`, fetched during that
  session): `LeakyRelu, Sigmoid, Relu, Add, Mul, Div, Sub, Concat, Softmax`.
  Anything outside this list (that investigation's example: `ReduceMean`/`Sqrt`)
  is a **silent no-op** -- Pulsar2 does not validate or warn, it just ignores
  the override. PR #1353's finding that `MatMul`/`Conv` don't accept it is
  consistent with this list, now fully enumerated rather than known only at
  the one or two data points prior forks happened to test.
- **`Conv` has a separate, narrower override**: `output_data_type: "FP32"` --
  distinct from `data_type`, and (per that investigation) the only lever that
  reaches a `Conv`'s precision at all. **Whether `MatMul` has an analogous
  `output_data_type` override was never checked** -- this repo's only mention
  of `output_data_type` is the one `Conv` sentence. This is the single most
  promising untried lever for PR #1353's plateau specifically: the weight
  gradient is produced by a `MatMul`, and if `output_data_type: "FP32"` is
  valid there too (not confirmed either way), the seed's FP32-ness might
  survive past exactly the boundary PR #1353 found it doesn't. **Concrete next
  step, needs real hardware**: try `layer_configs` with `{"op_type": "MatMul",
  "output_data_type": "FP32"}` (or whatever the exact schema key turns out to
  be) on the gradient-producing `MatMul` and check whether it's accepted or
  silently ignored, the same diagnostic PR #1353 already used (sweep the seed,
  check whether the returned gradient's zero-fraction actually changes).
- **`model_type: "QuantONNX"` is a real, distinct ingestion path** that lets a
  caller feed Pulsar2 an already-QDQ-annotated ONNX graph and **skips its own
  PTQ calibration** for tensors that already carry `QuantizeLinear`/
  `DequantizeLinear` (confirmed via `pulsar2 build`'s own log line, `"... is a
  QuantONNX model, disable concat align config"`) -- this is the direct answer
  to "can we hand Pulsar2 our own pre-quantized graph and have it treated as
  authoritative," which had been an open, untested question in this thread
  until this search surfaced that it already has an answer. **But it hits a
  real, reproducible Pulsar2-internal bug**: any `MatMul` whose weight comes
  through a `DequantizeLinear` (the standard ONNX QDQ per-channel weight
  pattern, exactly what `onnxsim.quantize_static`/`quantize_weight_only` both
  emit) crashes Pulsar2's own PPQ-based `ax_quant_graph_optimize` pass with
  `ValueError: Can not feed value to operation <node>, expects exact 2 inputs,
  however 1 was given`. Since the training step's own weight-gradient and
  forward-matmul nodes are exactly this shape (`MatMul` with a
  `DequantizeLinear`-derived weight, once legalized), `QuantONNX` as a way to
  hand-author the gradient's own quantization is **currently blocked by this
  same bug**, not a fresh problem -- worth retesting if that bug is ever fixed
  upstream, not worth reattempting as-is.

## What this doesn't cover

- No hardware was touched by this investigation -- a concurrent fork was using
  the AX650N/`axcl-vm`/Docker toolchain the whole time; this stayed entirely to
  archival analysis of files already on disk plus a search of this repo's own
  prior documentation.
- The `MatMul`-`output_data_type` question and the calibration-data-recalibration
  idea are both real, concrete, and **untested against real hardware** -- both
  need the toolchain free, and both are more promising leads than anything
  chased so far for the gradient-underflow ceiling specifically.
