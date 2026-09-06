# LLM-FP4: searched FP4 format + real-valued per-block scale (`quantize_weight_only_llm_fp4`)

## What this is

`onnxsim.quantize_weight_only_llm_fp4` quantizes a layer's weight block-wise
onto a **searched** standard sign/exponent/mantissa FP4 format: for each
weight tensor, it tries every way to split FP4's 3 non-sign bits between
exponent and mantissa (E1M2, E2M1, E3M0), and for each `(output channel,
block)` group, a grid of real-valued scale candidates -- keeping whichever
`(format, scale)` combination minimizes reconstruction MSE.

```
Before:
  Y = MatMul(X, W) [+ bias]          -- W constant, [K, N], float32

After:
  Codebook: initializer, float32 [16]     -- winning format's 16 values
  Wq: initializer, uint8 [K, N]           -- codebook index per element
  Ws: initializer, float32 [K/block_size, N]  -- real-valued scale per block
  What_hat = Reshape(Mul(Reshape(Gather(Codebook, Wq)), Ws), ...)
  Y = MatMul(X, What_hat) [+ bias]
```

## Where this comes from

[LLM-FP4](https://arxiv.org/abs/2310.16836) (Liu, Yuan, Yang, Cheng, Yang,
Liu, Zhu and Xu, 2023, EMNLP) frames its quantizer's real-valued scale and
its FP4 format's own exponent bias as the *same* degree of freedom
("pre-shifted exponent bias"): scaling a block by `2^d` before quantizing to
a fixed-bias FP4 codebook is identical to quantizing the unscaled block
against a codebook whose bias has been shifted by `d`. The paper searches
this freedom two ways -- a per-tensor search over the exponent/mantissa bit
split, and a (for activations, per-channel; here, per weight-block)
real-valued scale search -- rather than committing to one fixed format and a
power-of-two scale the way `onnxsim.mx_quantization`'s MXFP4 does.

This module realizes both searches directly:

- **Format search**: `onnxsim.llm_fp4.FP4_FORMATS` enumerates E1M2, E2M1
  (MXFP4's own element format), and E3M0 -- every way to split FP4's 3
  non-sign bits. One format is chosen per weight tensor, by total
  reconstruction MSE across every block.
- **Scale search**: for each `(output channel, block)` group, a grid of
  clip-ratio candidates (`min_clip_ratio` to `1.0`) is tried; the scale is
  `max_abs_block * ratio / format_max_magnitude`, real-valued rather than
  restricted to a power of two (`onnxsim.mx_quantization`'s own
  restriction).

Both searches minimize direct reconstruction MSE against the weight's own
values -- the same "grid search over a small set of candidates against a
direct error metric" shape `onnxsim.calibration`'s own `_entropy_threshold`
already uses for INT8 range calibration (that function searches clip
cutoffs against KL divergence; this module searches `(format, clip ratio)`
pairs against MSE).

## Scope

**Weight-only quantization (`quantize_weight_only_llm_fp4`).** This
function quantizes weights only. The paper's activation side is provided by
two *separate* passes in the same module, described below --
`apply_llm_fp4_activation_quantization_per_tensor` (the paper's own
calibrated per-tensor quantizer) and
`apply_llm_fp4_activation_quantization` (a simpler per-token, data-free
alternative). Neither performs the paper's **per-channel outlier
migration**; that half stays where it already lives in this repo, as
`onnxsim.apply_smoothquant`/`onnxsim.apply_outlier_suppression`, run first
by the caller. See `onnxsim/llm_fp4.py`'s own module docstring for the full
rationale.

**Activation quantization, option A
(`apply_llm_fp4_activation_quantization_per_tensor`) -- the paper's own
quantizer.** One real-valued **per-tensor** scale per activation, fit
offline from calibration data and emitted as a **constant initializer**.
The scale is found by the *same* clip-ratio-vs-reconstruction-MSE grid
search the weight side uses (`_search_fp4_clip_ratio`, shared by both):
`scale = max_abs * ratio / max(abs(Codebook))`, keeping whichever `ratio`
minimizes the codebook round-trip MSE of the captured activation. The FP4
*format* is **not** re-searched -- it was already fixed for that layer by
`quantize_weight_only_llm_fp4`'s own per-weight-tensor search, and the
recovered codebook is exactly what encodes that choice. Activations are
captured with the `_add_probe_outputs` + `onnxsim.backend.run_model`
pattern every calibration-driven pass in this repo uses (see
`onnxsim.apply_gptq`); `onnxsim.generate_random_calibration_data` supplies
the default batches when `calibration_data` is omitted. For every layer
`quantize_weight_only_llm_fp4` already weight-quantized, it inserts:

```
scale   -- constant float32 initializer, fit offline
x_normalized = X / scale
nearest = Gather(Codebook, ArgMin(Abs(Unsqueeze(x_normalized, -1) - Codebook), axis=-1))
x_dequant = nearest * scale
```

Because `scale` is a compile-time constant, this emits **no**
`Abs`/`ReduceMax`/`Max` range reduction at all: 7 nodes per layer (`Div`,
`Unsqueeze`, `Sub`, `Abs`, `ArgMin`, `Gather`, `Mul`) against the per-token
pass's 11, which is the practical point of the per-tensor design. It needs
only **opset 13** (`Unsqueeze`'s `axes`-as-input form; `ArgMin`, `Gather`,
`Sub`, `Abs`, `Div`, `Mul` are older) rather than the per-token pass's 18.

**This is the quantizer half only.** The paper pairs it with per-channel
outlier migration, which this function does **not** perform -- the caller
composes it, migration first:

```python
import onnxsim

migrated = onnxsim.apply_smoothquant(model)          # or apply_outlier_suppression
weight_q = onnxsim.quantize_weight_only_llm_fp4(migrated)
w4a4 = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(weight_q)
```

Running it without the migration step is supported and produces a valid
model, but then quantizes an unmigrated activation with a single shared
scale -- precisely the case the paper's migration exists to avoid.

**Activation quantization, option B
(`apply_llm_fp4_activation_quantization`) -- a
different granularity than the paper's own design.** A second,
self-contained activation-side pass, **not** a port of the paper's
per-tensor scheme above. It computes a **per-token**
(per-row), data-free scale fresh at graph-run time from each token's own
values -- no calibration data, no cross-layer migration pass -- exactly the
convention `onnxsim.zeroquant`/`onnxsim.quarot`/`onnxsim.duquant`/
`onnxsim.attention_quantization` already use for their own per-token
runtime quantizers, just against FP4's non-uniform 16-value codebook
instead of a uniform integer grid. For every layer
`quantize_weight_only_llm_fp4` already weight-quantized (found by walking
its weight input backward through that function's own exact
`Gather`/`Reshape`/`Mul`/`Cast` dequantization pattern -- the winning
format isn't recorded anywhere else, so this is the only robust way to
recover it), it inserts:

```
scale = max(ReduceMax(Abs(X), axis=-1, keepdims=1), epsilon) / max(abs(Codebook))
x_normalized = X / scale
nearest = Gather(Codebook, ArgMin(Abs(Unsqueeze(x_normalized, -1) - Codebook), axis=-1))
x_dequant = nearest * scale
```

using that same layer's own recovered codebook, and rewires the MatMul/Gemm
node's activation input to `x_dequant`. A layer not already matched by
`quantize_weight_only_llm_fp4`'s exact pattern is left completely
untouched -- this pass never runs the weight-side quantization itself.
Composing with `apply_smoothquant`/`apply_outlier_suppression` beforehand
remains possible (they act on the weight/LayerNorm side, which this pass
never touches) but is not required by this simpler design. See
`apply_llm_fp4_activation_quantization`'s own docstring in
`onnxsim/llm_fp4.py` for the full honesty note.

**Choosing between the two activation passes:**

| | `..._per_tensor` (option A) | per-token (option B) |
|---|---|---|
| Scale fit from | calibration data, offline | the tensor itself, at graph-run time |
| Granularity | one scale per tensor | one scale per token (row) |
| Scale lives in | a constant initializer | runtime `Abs`/`ReduceMax`/`Max` nodes |
| Nodes added per layer | 7 | 11 |
| Minimum opset | 13 | 18 (`ReduceMax` axes-as-input) |
| Paper fidelity | the paper's own quantizer (migration half is the caller's) | a repo-convention alternative, explicitly not the paper's design |

Neither pass emits a native FP4 tensor (ONNX has no FP4 type): both are
simulated ("fake") quantization -- values are snapped onto the FP4 codebook
grid and immediately dequantized back to float32, so the `MatMul` still
runs in float, exactly as `onnxsim.mx_quantization`/`onnxsim.nf4`/
`onnxsim.zeroquant` already do.

Handled (weight-only quantization):

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W)` with
  `transA=0`, `alpha=1`, `beta=1`. `transB` may be 0 or 1.
- Any opset -- the codebook lookup is built entirely from ordinary
  `Gather`/`Reshape`/`Mul`/`Cast` (opset 11+).
- `formats` to restrict (and speed up) the per-tensor search; `skip_names`
  to leave specific matched weights unquantized.

Left untouched (safe no-op, node passes through as-is):

- Non-constant weights, non-2-D weights, or a reduction dimension not
  divisible by `block_size`.
- (Both activation passes) any layer whose weight input isn't fed by
  `quantize_weight_only_llm_fp4`'s own exact dequantization pattern.
- (Per-token activation pass) any model whose opset is older than 18
  (`ReduceMax`'s `axes`-as-input form).
- (Per-tensor activation pass) any model whose opset is older than 13
  (`Unsqueeze`'s `axes`-as-input form), and any layer whose activation no
  calibration batch reached, or for which every captured value was zero or
  non-finite -- that layer is left untouched rather than silently falling
  back to a different scale scheme.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_llm_fp4(model)  # block_size=32
onnx.save(quantized, "model.llm_fp4.onnx")

# Optional: complete W4A4 with a per-token, data-free activation quantizer
# (a different granularity than the paper's own design -- see "Scope").
w4a4 = onnxsim.apply_llm_fp4_activation_quantization(quantized)
onnx.save(w4a4, "model.llm_fp4.w4a4.onnx")
```

Or, for the paper's own activation scheme -- per-channel outlier migration
first, then the calibrated per-tensor FP4 activation quantizer:

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
migrated = onnxsim.apply_smoothquant(model)  # or apply_outlier_suppression
weight_q = onnxsim.quantize_weight_only_llm_fp4(migrated)  # block_size=32
w4a4 = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(weight_q)
onnx.save(w4a4, "model.llm_fp4.w4a4.onnx")
```

The **weight** side needs no calibration data: both the per-tensor format
choice and the per-block scale are fit directly to each weight's own
values, by exhaustive grid search minimizing reconstruction MSE. The same
is true of the **per-token** activation pass. Only
`apply_llm_fp4_activation_quantization_per_tensor` (and the
`apply_smoothquant`/`apply_outlier_suppression` migration it composes with)
uses calibration data -- random batches from
`onnxsim.generate_random_calibration_data` by default, or real batches
passed as `calibration_data` (see
`onnxsim.load_huggingface_calibration_data`), which give a much more
representative activation range.

## Relationship to onnxsim's other 4-bit floating-point modules

| | Format | Scale | Chosen how |
|---|---|---|---|
| `quantize_weight_only_mxfp4` | fixed E2M1 | power-of-two only | from the block's own max-abs |
| `quantize_weight_only_nf4` | N/A (normal-distribution codebook, not sign/exp/mantissa) | absmax | from the block's own max-abs |
| `quantize_weight_only_llm_fp4` | searched per tensor (E1M2/E2M1/E3M0) | real-valued, searched per block | grid search minimizing reconstruction MSE |
