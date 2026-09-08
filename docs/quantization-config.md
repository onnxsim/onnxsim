# Unified quantization config (`QuantizationConfig` / `quantize`)

## What this is

onnxsim ships more than a dozen `quantize_*` functions
(`quantize_dynamic`, `quantize_static`, `quantize_weight_only_int4`, ...),
each documenting and exposing its own scheme's specific parameters — see the
other docs in this directory for each one directly. `onnxsim.QuantizationConfig`
and `onnxsim.quantize` (`onnxsim/accuracy.py`) add a second, unified way to
reach all of them: describe *what* quantization you want as one typed
config object, and let `quantize` dispatch to the right underlying function.

This doesn't replace any existing `quantize_*` function — call those
directly when you already know which scheme you want. `quantize` is for code
that picks a scheme *programmatically*: a sweep over configs to compare
accuracy/size tradeoffs, a scheme read from a config file or CLI flag, or
any caller that wants one call site regardless of which scheme ends up
selected.

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
config = onnxsim.QuantizationConfig(scheme="weight_only", dtype="int4")
quantized = onnxsim.quantize(model, config)
onnx.save(quantized, "model.int4.onnx")
```

## The scheme / dtype / granularity matrix

| `scheme` | `dtype` | `granularity` | Calls |
| --- | --- | --- | --- |
| `"dynamic"` | `"int8"` | `"per_channel"` | `quantize_dynamic` |
| `"dynamic_fused"` | `"int8"` | `"per_channel"` | `quantize_dynamic_matmul_integer_to_float` |
| `"ternary"` | `"int8"` | `"per_channel"` | `quantize_ternary` |
| `"weight_only"` | `"int8"` | `"per_channel"` | `quantize_weight_only` |
| `"weight_only"` | `"int8"` | `"per_block"` | `quantize_weight_only_int8_block` |
| `"weight_only"` | `"int16"` | `"per_channel"` | `quantize_weight_only_int16` |
| `"weight_only"` | `"int4"` | `"per_block"` (only option) | `quantize_weight_only_int4` |
| `"static"` | `"int8"` | `"per_channel"` | `quantize_static` |
| `"static_int16"` | `"int16"` | `"per_channel"` | `quantize_static_int16` |
| `"qoperator"` | `"int8"` | `"per_channel"` | `quantize_qoperator` |
| `"float"` | `"float16"` | n/a | `quantize_fp16` |
| `"float"` | `"bfloat16"` | n/a | `quantize_bf16` |
| `"float"` | `"float8_e4m3"` | n/a | `quantize_fp8(format="e4m3")` |
| `"float"` | `"float8_e5m2"` | n/a | `quantize_fp8(format="e5m2")` |

`granularity` is only consulted for `scheme="weight_only", dtype="int8"` —
every other row has exactly one granularity onnxsim implements today, and
the field is ignored for those. Follow the linked docs above (or each
function's own docstring) for what each scheme actually does numerically;
this page only covers the dispatch layer.

An unknown `scheme`, a `dtype` not valid for that `scheme`, or an invalid
`granularity` for `weight_only`/`int8` all raise `ValueError` with a message
naming the valid options — `quantize` never silently guesses at a
misconfigured request.

## Chained passes (`awq`, `gptq`, `gptaq`, `qat`, `double_quant`)

For `scheme="weight_only", dtype="int4"`, five flags chain onnxsim's
correction passes onto `quantize_weight_only_int4`'s plain round-to-nearest
output, in one fixed order: `awq` → `gptq` → `gptaq` → `qat` → `double_quant`.
The last two positions are forced rather than preferred. QAT warm-starts from
whatever codes the model already carries, so it wants the corrected model as
its starting point; and `apply_double_quantization` moves each scale out of an
initializer and into a nested `DequantizeLinear`, which is exactly the shape
QAT's layer finder needs to recognize a quantized layer — run the other way
round, `discover_qat_blocks` returns nothing and QAT is a silent no-op.

`qat` is the one of the five that is not int4-only: with `scheme="static"` it
chains `apply_qat_all_blocks` onto `quantize_static`'s output instead, training
that scheme's activation quantizers alongside its INT8 weights. Which of
`apply_qat`'s two modes runs is decided by the scheme rather than by a separate
field — `learn_activation_scales` is `True` for `static` and `False` for
`int4`, the only settings `apply_qat` accepts for those models — so this API
cannot express the pairing it refuses.

`qat_num_iterations` (optimizer steps *per block*) and `qat_learning_rate` are
the only two knobs exposed; everything else about the walk is
`apply_qat_all_blocks`' own defaults, and [qat.md](qat.md) covers what it is
worth and where it loses. It is by far the most expensive flag here, and it is
not a uniform win: it loses to AdaRound on a single layer with full-rank
calibration activations, and the activation training `scheme="static"` implies
is a small regression on a model whose ranges were calibrated on representative
data — it fixes a clip range that is wrong rather than improving one that is
right.

If `qat` trains no block — none discovered, or all skipped — `quantize` warns
and returns the untuned quantized model rather than raising, matching
`apply_qat_all_blocks`' own per-block leniency.

## Other `QuantizationConfig` fields

- `calibration_data`, `num_calibration_samples`, `seed`, `providers`,
  `calibration_method` — forwarded as-is to `quantize_static`/
  `quantize_static_int16`/`quantize_qoperator` for the three calibration-based
  schemes; ignored otherwise. See
  [dynamic-quantization.md](dynamic-quantization.md) and
  [qoperator-quantization.md](qoperator-quantization.md) for what calibration
  data is and how to supply real (not random) data via
  `onnxsim.load_huggingface_calibration_data`.
- `keep_io_types` — forwarded to `quantize_fp16`/`quantize_bf16`/`quantize_fp8`
  for `scheme="float"`; ignored otherwise. See
  [fp16-quantization.md](fp16-quantization.md).

## Measuring what a config actually costs

Picking a `QuantizationConfig` is only half the question — how much accuracy
it costs is the other half. See
[precision-estimation.md](precision-estimation.md#whole-model-rollup-estimate_model_quantization_drop)
for a fast, data-free estimate, and [accuracy-drop.md](accuracy-drop.md) for
an actual, data-driven measurement of a specific quantized model.
