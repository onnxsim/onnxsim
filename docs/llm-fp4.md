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

**Weight-only quantization (`quantize_weight_only_llm_fp4`).** The paper's
other headline contribution -- migrating a per-channel real-valued
*activation* scale into the preceding weight or LayerNormalization (via
exactly the algebra `onnxsim.apply_smoothquant`/
`onnxsim.apply_outlier_suppression` already implement), so that a single
shared exponent bias suffices for a per-tensor FP4 *activation* quantizer,
enabling full W4A4 -- is **not implemented here**. That migration machinery
already exists in this repo and composes with this module unchanged (run
either migration pass first). See `onnxsim/llm_fp4.py`'s own module
docstring for the full rationale.

**Activation quantization (`apply_llm_fp4_activation_quantization`) -- a
different granularity than the paper's own design.** This module now also
provides an activation-side pass, but it is **not** a port of the paper's
per-tensor-via-migration scheme above. It computes a **per-token**
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
- (Activation quantization) any layer whose weight input isn't fed by
  `quantize_weight_only_llm_fp4`'s own exact dequantization pattern, and
  any model whose opset is older than 18 (`ReduceMax`'s `axes`-as-input
  form).

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_llm_fp4(model)  # block_size=32
onnx.save(quantized, "model.llm_fp4.onnx")

# Optional: complete W4A4 with a per-token, data-free activation quantizer
# (a different granularity than the paper's own per-tensor-via-migration
# design -- see "Scope" above).
w4a4 = onnxsim.apply_llm_fp4_activation_quantization(quantized)
onnx.save(w4a4, "model.llm_fp4.w4a4.onnx")
```

Needs no calibration data: both the per-tensor format choice and the
per-block scale are fit directly to each weight's own values, by exhaustive
grid search minimizing reconstruction MSE.

## Relationship to onnxsim's other 4-bit floating-point modules

| | Format | Scale | Chosen how |
|---|---|---|---|
| `quantize_weight_only_mxfp4` | fixed E2M1 | power-of-two only | from the block's own max-abs |
| `quantize_weight_only_nf4` | N/A (normal-distribution codebook, not sign/exp/mantissa) | absmax | from the block's own max-abs |
| `quantize_weight_only_llm_fp4` | searched per tensor (E1M2/E2M1/E3M0) | real-valued, searched per block | grid search minimizing reconstruction MSE |
