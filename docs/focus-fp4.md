# FOCUS for MXFP4: a decoupled quantization scale (`quantize_weight_only_mxfp4_focus`)

## What this is

`onnxsim.quantize_weight_only_mxfp4_focus` quantizes exactly what
`onnxsim.quantize_weight_only_mxfp4` quantizes, and emits exactly the format
it emits -- same initializer names, shapes and dtypes, same nodes in the same
order, and a bit-identical power-of-two `*_mxfp4_scale`. The *only* thing
that differs is which E2M1 code each element gets.

```
Before:
  Y = MatMul(X, W) [+ bias]          -- W constant, [K, N], float32

After (identical to docs/mxfp4.md's own output):
  Codebook: initializer, float32 [16]     -- E2M1's own 16 fixed values
  Wq: initializer, uint8 [K, N]           -- codebook index per element  <-- only this differs
  Ws: initializer, float32 [K/32, N]      -- power-of-two scale per block
  W_hat = Reshape(Mul(Reshape(Gather(Codebook, Wq)), Ws), ...)
  Y = MatMul(X, W_hat) [+ bias]
```

Read `docs/mxfp4.md` first -- this page only covers how the codes are
chosen.

## Where the idea comes from

**FOCUS** is Tencent AngelSlim's FP4 scale-optimization method; the
description this module is written against is AngelSlim's own documentation
for it (`docs/source/features/quantization/focus_fp4.md`, "FOCUS: end-to-end
scale optimization for FP4").

Its observation is about an asymmetry in how a block-scaled FP4 format is
actually deployed. Ordinarily one scale does both jobs:

```
W_bar_i = Q_E2M1(W_i / S_i)     (quantize -- offline, on the host)
W_hat_i = W_bar_i * S_i         (dequantize -- on the accelerator)
```

but the **quantization** scale exists only offline, while the FP4 codes are
being generated. It is never stored in the deployed model, so nothing about
it has to satisfy the hardware's scale format. Only the **dequantization**
scale is constrained -- E8M0 (a pure power of two, no mantissa bits at all)
for MXFP4, E4M3 for NVFP4. FOCUS widens the optimization space along two axes
that exploit that asymmetry:

- **CRS (Coupled-Relaxation Scaling)** gives each block a full-precision
  coefficient `c_i` that decouples the two scales:

  ```
  S_i^q = S_i^dq * c_i
  W_bar_i = Q_E2M1(W_i / S_i^q)
  W_hat_i = W_bar_i * S_i^dq
  ```

  `S_i^dq` still satisfies the hardware constraint; `c_i` participates only
  in the offline fit and **is discarded at export**.

- **DGS (Dual-Granularity Scaling)** splits each hardware block into smaller
  sub-blocks (AngelSlim's own choice: 8 elements inside the 32-element MX
  block) and gives each sub-block its own `c_i^k`. The dequantization scale
  stays at the original hardware granularity, so the deployed format is again
  unchanged.

## What is ported, and what is not

FOCUS optimizes `c_i` with an **end-to-end trained** procedure (AngelSlim's
documentation ships training configs and a multi-GPU setup for it). onnxsim
is stateless PTQ graph rewriting with no training loop, so **that part is
deliberately not ported**.

What is ported is the *mechanism* -- the decoupled quantization scale and its
two granularities -- fitted instead by a **data-free per-block grid search**
over `c_i` against that block's own local reconstruction error. That is the
same shape `onnxsim.llm_fp4`'s `_search_llm_fp4_blockwise` already uses for
its `(FP4 format, clip ratio)` search, and the same PTQ scoping
`onnxsim.adaround`/`onnxsim.omniquant` use for their own per-layer auxiliary
quantities. (`llm_fp4`'s searcher itself could not be reused: its scale
candidates move the *dequantization* grid too, which is precisely the
coupling FOCUS removes, so the two searches optimize different things.)

**FOCUS's reported accuracy is therefore neither claimed nor reproduced
here.** A grid search against a local objective is a strictly weaker fitting
procedure than the paper's own trained one, and this port has not been
evaluated on any of the paper's benchmarks.

## What *is* guaranteed

The structural property CRS/DGS rest on, and by construction rather than by
measurement:

- The stored dequantization scale is not merely "also a power of two" -- it
  is the *identical array* `onnxsim.mx_quantization._quantize_mxfp4_blockwise`
  itself computes, taken from that function directly rather than
  recalculated.
- The graph is emitted by the very same driver
  (`onnxsim.mx_quantization._quantize_weight_only_mxfp4_impl`) that
  `quantize_weight_only_mxfp4` uses.

So a FOCUS-quantized model differs from a plain-MXFP4-quantized one *only* in
the `*_mxfp4_q` code bytes: swap those back and the two protobufs are
byte-identical. `tests/test_focus_fp4.py` asserts exactly that. Nothing
downstream -- a runtime, an exporter, a checker -- can tell the two formats
apart.

## Why the search objective is not plain element-wise MSE

This is the part worth reading carefully, because the natural assumption is
wrong.

With the dequantization scale `S^dq` held fixed, the reachable reconstruction
values for an element are exactly `{codebook[j] * S^dq}`. The choice
minimizing that element's *squared error* is, by definition, the nearest one
-- which is precisely what plain `quantize_weight_only_mxfp4` already does
(`argmin` over the codebook of `|W / S^dq - codebook|`). So **a search over
`c_i` judged by plain, unweighted, element-wise reconstruction MSE provably
cannot beat plain MXFP4**: `c_i = 1` is already its optimum, and every other
coefficient can only tie or lose. At a fixed dequantization scale, CRS/DGS
buy nothing under an i.i.d.-input assumption. That is a theorem about the
objective, not a shortcoming of the search.

This module does not paper over that. `aggregate_error_weight=0.0` reproduces
plain MXFP4's codes *exactly* -- byte-identical model -- and the test file
asserts it.

Where the decoupled scale does buy something is where FOCUS's own end-to-end
loss lives: the error that reaches the **layer's output**, not the weight.
For `Y = X W`, a weight error `D` costs `D^T E[X^T X] D`, and the data-free
`E[X^T X] = I` simplification is exactly what collapses that back to
element-wise MSE. Real transformer activations are not that: post-GELU /
post-ReLU activations are non-negative and share a mean component, so
`E[X^T X]` carries a rank-one term. Taking
`E[X^T X] = sigma^2 (I + lambda * 11^T)` -- the standard "mean-shifted,
positively correlated inputs" model, the same phenomenon
`onnxsim.bias_correction` exists to correct -- gives, per block:

```
objective = sum_j D_j^2  +  lambda * (sum_j D_j)^2
```

element-wise MSE plus a penalty on the block's **aggregate signed** error.
That second term is *not* minimized by per-element nearest rounding, so it is
a genuine, non-degenerate use of the extra freedom CRS/DGS provide: a
slightly different `c_i` flips a handful of near-boundary elements the other
way, paying a small MSE cost to cancel a much larger systematic bias.
`lambda` is the `aggregate_error_weight` parameter.

**This objective is onnxsim's own choice, not something AngelSlim's
documentation specifies.** FOCUS trains against the network loss itself,
which needs data; this is the data-free surrogate that keeps the mechanism
non-trivial.

The search is monotone by construction -- the `c_i = 1` candidate is always
evaluated first (so ties keep plain MXFP4's own codes), and DGS refinement is
coordinate descent started from CRS's own solution that accepts only strict
improvements -- so the objective it reports is never worse than plain
MXFP4's, at any setting, and DGS is never worse than CRS.

## Measured behaviour

On random Gaussian `[K, N]` weights, block size 32, DGS sub-block size 8,
`aggregate_error_weight=1.0`, aggregated over 8 seeds, comparing against
plain `quantize_weight_only_mxfp4` on the same weights (this is
`tests/test_focus_fp4.py`'s own measurement, at looser asserted thresholds):

| Metric | CRS | DGS |
|---|---|---|
| Block objective `sum(D^2) + (sum D)^2` | **-39%** | **-43%** |
| Real layer-output MSE, mean-shifted inputs (`x = 1 + N(0,1)`) | **-37%** | **-42%** |
| Real layer-output MSE, i.i.d. inputs (`x = N(0,1)`) | +9% | +8% |
| Element-wise weight MSE | +8.5% | +8.3% |

Both halves of that trade are asserted in the tests, not just the favourable
one. The i.i.d. row is the theorem above showing up empirically: with
uncorrelated inputs there is nothing for the extra freedom to buy, and FOCUS
pays a modest penalty for optimizing something else. Whether that is the
right trade for a given model is a property of that model's activations,
which is what `aggregate_error_weight` is for -- set it to `0.0` and you get
plain MXFP4 back, exactly.

## Scope

Handled -- identical to `quantize_weight_only_mxfp4`'s own scope, since the
matching and emission are literally the same code:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W)` with
  `transA=0`, `alpha=1`, `beta=1`. `transB` may be 0 or 1.
- Any opset (the codebook lookup is ordinary `Gather`/`Reshape`/`Mul`/`Cast`,
  opset 11+).
- `skip_names` to leave specific matched weights unquantized.

Left untouched (safe no-op): non-constant weights, non-float32 weights,
non-2-D weights, or a reduction dimension not divisible by `block_size`.

**MXFP4 only.** The searcher (`onnxsim.focus_fp4._search_focus_codes`) takes
an arbitrary codebook and an arbitrary already-fixed dequantization scale, so
it is not specific to E8M0 and would apply unchanged to
`onnxsim.nvfp4_quantization`'s E4M3-scaled NVFP4. Only the MXFP4 entry point
is shipped: MXFP4 is the case where the coupling is most constrained (E8M0
has no mantissa bits at all), and adding an NVFP4 entry point would mean
duplicating that module's graph emission -- which, unlike
`onnxsim.mx_quantization`'s, has not been factored into a reusable driver --
for a mechanism whose benefit here comes from the aggregate-error term rather
than from the scale format. That is a mechanical follow-up, not a design
question.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")

# Default: DGS with AngelSlim's own 8-element sub-blocks inside the 32-element
# MX hardware block.
quantized = onnxsim.quantize_weight_only_mxfp4_focus(model)

# CRS only -- one coefficient per hardware block.
quantized = onnxsim.quantize_weight_only_mxfp4_focus(model, sub_block_size=None)

# Degenerate: byte-identical to onnxsim.quantize_weight_only_mxfp4(model).
quantized = onnxsim.quantize_weight_only_mxfp4_focus(model, aggregate_error_weight=0.0)

onnx.save(quantized, "model.mxfp4.onnx")
```

Needs no calibration data: the codebook, the per-block power-of-two scale and
the searched coefficients all come from the weight's own values.

## Relationship to onnxsim's other MXFP4 / FP4 modules

| | Dequantization scale | Quantization scale | Chosen how |
|---|---|---|---|
| `quantize_weight_only_mxfp4` | power-of-two (E8M0) | same as dequant | block max-abs |
| `quantize_weight_only_mxfp4_focus` | power-of-two (E8M0), **identical to the above** | `S^dq * c_i`, `c_i` per block (CRS) or per 8-element sub-block (DGS), discarded at export | grid search minimizing `sum(D^2) + lambda*(sum D)^2` |
| `quantize_weight_only_nvfp4` | E4M3 + per-tensor FP32 | same as dequant | block max-abs |
| `quantize_weight_only_llm_fp4` | real-valued float32 | same as dequant | grid search over `(format, clip ratio)` minimizing MSE |
