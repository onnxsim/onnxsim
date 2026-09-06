# QuaRot: rotation-based W4A4 quantization (`apply_quarot`)

## What this is

`onnxsim.apply_quarot` rotates a layer's weight *and* activation by the
same random orthogonal matrix, then quantizes **both** to INT4 with plain
round-to-nearest -- no calibration data needed at all.

```
Before:
  Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

After:
  U: initializer, float32 [K, K]            -- the random rotation
  Xrot = MatMul(X, U)                       -- runtime activation rotation
  Xq   = round_to_nearest_int4_per_token(Xrot)   -- data-free, no calibration
  Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
                                              -- INT4 codes, [K, N]
  Y = MatMul(Xq, Wtilde_hat) [+ bias]
```

Every other INT4 scheme in onnxsim (`quantize_weight_only_int4` and
everything built on it -- `apply_spinquant`, `quantize_weight_only_spqr`,
`apply_quip_sharp`) only quantizes the **weight**; the activation stays
float32, so the MatMul itself still runs at float precision. QuaRot's own
contribution is that the same random-rotation trick which makes a weight
easy to quantize (already used by `apply_quip_sharp`) works just as well
on the activation, letting *both* operands drop to INT4.

## Where this comes from

[QuaRot](https://arxiv.org/abs/2404.00456) (Ashkboos et al., 2024)
observes that `apply_quip_sharp`'s own incoherence-processing idea --
conjugating a weight by a random orthogonal rotation removes its outlier
directions with high probability, regardless of the weight's own
structure (a concentration-of-measure argument) -- applies equally well
to *activations*: rotating a token's activation vector by the same kind
of random rotation spreads out whatever outlier channels it had, exactly
the problem `apply_smoothquant` addresses by a different route (migrating
outlier magnitude into the weight via a fitted per-channel scale, rather
than removing it via rotation). Once both operands are outlier-free,
plain round-to-nearest INT4 works on each independently, with no
calibration data required for either.

This module reuses `onnxsim.quip_sharp`'s own `_random_orthogonal_matrix`
(a Haar-random matrix via QR-decomposing a random Gaussian, not the real
QuaRot's Hadamard-structured construction -- see `quip_sharp`'s own
docstring for why a plain orthogonal matrix is an equally valid, simpler
substitute) and `onnxsim.omniquant`'s own
`_quantize_blockwise_int4_with_clip` for the weight side, exactly as
`apply_spinquant` does. The activation is quantized per token at
graph-run time using the same data-free pattern
`onnxsim.quantize_kv_cache`'s Value-style rewrite already uses:
`scale = max(|x|) / 7`, computed fresh from that token's own rotated
values, with no stored calibration statistics.

**What's not reproduced.** The real QuaRot rotates the *entire residual
stream* by one shared rotation, fused into every adjacent weight matrix
across a whole decoder stack so the rotation is "free" at inference (plus
a second, smaller rotation for the attention head dimension specifically).
Fusing one global rotation across an entire model needs a model-level
graph walk this module does not attempt; instead, matching
`apply_spinquant`'s own scope, this module fits one independent rotation
**per matched layer**, at the cost of an explicit `MatMul(X, U)` per layer
rather than a fused rotation. `apply_quarot_fused` (below) recovers the
"free at inference" half of that gap on the subset of edges where the
algebra allows it, without the model-level walk. The real QuaRot also offers an optional
GPTQ-based weight quantizer for a tighter error bound; `apply_quarot`
itself always uses plain round-to-nearest for the weight, but see
`apply_quarot_gptq` below for the GPTQ-based variant.

## `apply_quarot_gptq`: GPTQ-based weight quantization

`onnxsim.apply_quarot_gptq` is identical to `apply_quarot` in every
respect -- same candidate matching, same per-layer rotation `U` (for a
given `seed`, byte-identical to `apply_quarot`'s own), same data-free
per-token INT4 activation quantization -- except the *weight* is
quantized via `onnxsim.gptq`'s Hessian-based column algorithm instead of
independent round-to-nearest, using real calibration activations:

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_quarot_gptq(model, block_size=32, seed=0)
onnx.save(quantized, "model.quarot_gptq.onnx")
```

The composition: each matched layer's own (pre-rotation) activation `X`
is captured from `model` via calibration data (the same
`_add_probe_outputs` + `backend.run_model` pattern `apply_gptq` uses),
then rotated by that layer's own `U` (`X_rotated = X @ U`) so the
Hessian `H = X_rotated.T @ X_rotated` GPTQ's column algorithm needs is
computed in the same (rotated) space as the weight it is quantizing,
`Wtilde = W @ U` -- not the original, unrotated activation space, since
that is not the space GPTQ's reconstruction objective is being evaluated
in here. The per-block scale is still the one
`_quantize_blockwise_int4_with_clip` computes (unchanged from
`apply_quarot`); GPTQ only changes which integer each element rounds to.

A matched layer with no calibration data reaching it, or whose captured
activation isn't a plain 2-D array, or whose feature dimension doesn't
match the weight's own `K`, is left completely untouched by
`apply_quarot_gptq` -- no rotation, no quantization -- rather than
silently falling back to plain round-to-nearest under GPTQ's own name.
`calibration_data=None` generates it via
`onnxsim.generate_random_calibration_data`, the same convention as
`apply_gptq`/`apply_awq`/every other calibration-driven onnxsim pass.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W[, B])` with
  `transA=0`, `alpha=1`, `beta=1` (when `B` is present) under the same
  weight constraint. `transB` may be 0 or 1.
- Opsets >= 21 (INT4's tensor type and `DequantizeLinear`'s `block_size`
  attribute both need opset 21, matching `quantize_weight_only_int4`).

Left untouched (safe no-op, node passes through as-is):

- Non-constant weights, non-2-D weights, or a reduction dimension not
  divisible by `block_size`.
- A model with no matching layer, or an opset older than 21.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_quarot(model, block_size=32, seed=0)
onnx.save(quantized, "model.quarot.onnx")
```

No `calibration_data` argument exists -- unlike `apply_spinquant`,
`apply_smoothquant`, or `apply_awq`, the rotation is random rather than
fit to data, and both quantization scales come from the rotated values
themselves (offline for the weight, at graph-run time for the activation),
so there is nothing to calibrate.

## Fusing the rotation into the producer (`apply_quarot_fused`)

`apply_quarot` pays an explicit `MatMul(X, U)` at run time for every
layer it quantizes. `onnxsim.apply_quarot_fused` does the same matching,
draws the same rotations from the same seed, and quantizes the same way --
but wherever the graph allows it, it folds that `MatMul(X, U)` into the
*producing* layer's weight offline, so the rotation costs nothing at
inference. That is the "free at inference" property the real QuaRot gets
from its residual-stream-wide rotation, here obtained one edge at a time.

### The algebra

If layer `P` produces the tensor `T` that layer `L` consumes, and `P`'s
weight is constant:

```
P's output:  T = X_prev @ W_P            (+ b_P)
Fold:        W_P' = W_P @ U    and (if P has a bias)  b_P' = b_P @ U
Then:        X_prev @ W_P' (+ b_P')  ==  (X_prev @ W_P + b_P) @ U  ==  T @ U
```

so `P` directly emits the rotated activation `L` wants. `L`'s own weight
is rotated exactly as `apply_quarot` rotates it, so
`(T @ U) @ (U.T @ W_L) == T @ W_L` still holds. The fold is exact
algebra: nothing is lost by it, and the only lossy steps remain the same
two INT4 roundings. The result is `apply_quarot`'s accuracy with one
`MatMul` node and one `[K, K]` float initializer fewer per fused layer.

Note the two rotations act on **different axes** of a weight: a layer's
own input-side rotation acts on its reduction dim `K` (right-multiply,
`W @ U`), while an output-side fold acts on its output dim `N`. A layer
that is both a fused producer and a quantized consumer -- the middle of a
chain `A -> B -> C` -- gets **both**, and both are applied while its
weight is still float, before it is quantized. Chains of any length are
handled: the pass plans every rotation first, then rotates every weight,
and only then quantizes.

### Exactly which edges are fusable

`L`'s activation edge is fused only when **all** of these hold:

- the activation tensor is produced by a node in the main graph that the
  same matcher accepts (a `MatMul`, or a `Gemm` with `transA=0`,
  `alpha=1`, `beta=1`) whose weight is a constant 2-D float32
  initializer. The producer does *not* itself need to be quantizable --
  a producer whose own `K` is not divisible by `block_size` stays in
  float and simply gets its rotated float weight written back;
- that tensor is referenced by exactly one node input in the whole model,
  subgraph bodies included. A tensor consumed twice cannot be rotated for
  one consumer without corrupting the other;
- that tensor is not a graph output -- a graph output must keep its
  unrotated value;
- the producer's output dimension equals `L`'s reduction dimension `K`;
- if the producer has a bias, that bias is a constant float32 initializer
  shaped `[N]` or `[1, N]` -- the only shapes whose last axis is
  unambiguously the output-channel axis, and so the only ones `b @ U` is
  defined for. A scalar/broadcast `Gemm` bias is not folded, and its edge
  is not fused.

An edge that fails any of these is **not** skipped and **not** silently
mis-rotated: its layer falls back to `apply_quarot`'s explicit runtime
`MatMul(X, U)`, exactly as before. When no edge in a model is fusable,
`apply_quarot_fused` returns byte-for-byte what `apply_quarot` returns.
Original initializers are never edited in place -- a rotated producer
weight or bias is written under a fresh `_quarot_folded` name -- so a
weight shared by several nodes stays exact for the nodes that were not
folded into.

### What this still is not

This is still a **per-layer** rotation, not the real QuaRot's single
residual-stream-wide one: each matched layer keeps its own independent
`U`. What `apply_quarot_fused` removes is only the *runtime cost* of that
rotation, on the subset of edges where the algebra above applies. Layers
fed by anything other than another constant-weight `MatMul`/`Gemm` -- a
model input, a normalization, an activation function, a residual `Add`,
an attention softmax -- are not fusable and keep their explicit
`MatMul(X, U)`. In a real transformer that means the projections that
immediately follow a LayerNorm or a residual add (i.e. most of them) still
pay for their rotation; fusing those needs the model-level residual-stream
walk this module still does not attempt.

### Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_quarot_fused(model, block_size=32, seed=0)
onnx.save(quantized, "model.quarot.onnx")
```

Same signature as `apply_quarot`, and still no `calibration_data`.
