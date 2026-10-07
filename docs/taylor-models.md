# Taylor models and the mean-value form

`onnxsim.taylor` bounds the values of every tensor of an ONNX model over a box of inputs, like
`onnxsim.interval`, `onnxsim.zonotope` and `onnxsim.crown`, but with *curvature*: a smooth op is
enclosed by a low-degree polynomial plus a certified remainder instead of a linear relaxation.

It exists for the ops the linear methods have no good rule for: **Softmax, GELU, LayerNorm /
RMSNorm, Exp / Log / Sqrt / Reciprocal, attention scores**. On those, over a narrow box, it stays
within roughly 4% to 10% of the true width for Softmax, GELU and LayerNorm (1.6x for a one-head
attention block), where the other methods return the trivial interval or have no rule.

## When to use which

| Situation | Use |
|---|---|
| Softmax, GELU, LayerNorm, attention, exp/log/sqrt/reciprocal on a *narrow* box | `taylor` |
| ReLU networks, conv nets with saturating activations, unbounded or wide boxes | `crown` (alpha-CROWN) or `zonotope` |
| Original-versus-simplified equivalence | `zonotope.bound_difference` |
| Low-dimensional input, a cheap second opinion | `taylor.mean_value_bounds` |
| Not sure | `taylor.compare_methods` runs all of them side by side |

Taylor models are **not** uniformly the tightest. On a two-layer Conv-Sigmoid net CROWN (1.66x of
the true width) and zonotopes (1.95x) beat them (2.29x); see the measurements below.

## Quick start

<!-- doctest -->
```python
import numpy as np
from onnx import parser

from onnxsim import taylor

model = parser.parse_model(
    '<ir_version: 9, opset_import: ["" : 17]> '
    "m (float[1,4] x) => (float[1,4] y) { y = Softmax<axis=-1>(x) }"
)

res = taylor.propagate(model, {"x": (-0.1, 0.1)})
lo, hi = res.bounds("y")
print("lo", lo.round(4), "hi", hi.round(4))
print("symbols", res.n_symbols, "notes", res.notes)
```

<!-- output -->
```text
lo [[0.2096 0.2096 0.2096 0.2096]] hi [[0.2904 0.2904 0.2904 0.2904]]
symbols 4 notes []
```

The true range of each output is `[0.2144, 0.2894]`, so the bounds are sound and about 8% wider
than the truth. Plain intervals can only say `[0, 1]`.

Input ranges can also come from the model's own `onnxsim.range.*` annotations
(see [ranges.md](ranges.md)). Every graph input needs a finite range and a static shape.

### Side by side

<!-- doctest -->
```python
cmp = taylor.compare_methods(model, {"x": (-0.1, 0.1)}, samples=500)
print(cmp.table())
```

<!-- output -->
```text
method                         y
interval              1 ( 13.34x)
zonotope              1 ( 13.34x)
crown                 1 ( 13.34x)
taylor          0.08085 (  1.08x)
mean_value      0.09963 (  1.33x)
sampled                  0.07494
```

Each cell is the mean output width, and the ratio is to `sampled`: random, random-vertex and
corner inputs through onnxruntime. `sampled` is an *inner* approximation of the true range, so a
sound method sits at or above it. A method that cannot handle the model (an unbounded input, say)
is listed under `errors` instead of aborting the comparison. `unbounded` in a table means the
method has no rule for an op in the model and says so, not that the true range is unbounded.

### The mean-value form

<!-- doctest -->
```python
mv = taylor.mean_value_bounds(model, {"x": (-0.1, 0.1)})
print("lo", mv.bounds["y"][0].round(4), "hi", mv.bounds["y"][1].round(4))
```

<!-- output -->
```text
lo [[0.2047 0.2047 0.2047 0.2047]] hi [[0.3043 0.3043 0.3043 0.3043]]
```

`f(x) in f(c) + J(box) (x - c)`, where `J(box)` is an interval enclosure of the Jacobian over the
whole box, from forward-mode interval automatic differentiation. It needs one Jacobian row per
input *element*, so it is for low-dimensional inputs (`max_inputs`, default 256).

### Degree and the truncation knobs

<!-- doctest -->
```python
for deg in (1, 2):
    r = taylor.propagate(model, {"x": (-0.1, 0.1)}, degree=deg)
    l, h = r.bounds("y")
    print("degree", deg, "mean width", round(float(np.mean(h - l)), 5))
capped = taylor.propagate(model, {"x": (-0.1, 0.1)}, max_vars=2, quad_vars=2)
l, h = capped.bounds("y")
print("max_vars=2: symbols", capped.n_symbols, "mean width", round(float(np.mean(h - l)), 5))
```

<!-- output -->
```text
degree 1 mean width 0.08562
degree 2 mean width 0.08085
max_vars=2: symbols 2 mean width 0.09301
```

## What it computes

Every tensor element is

```
x = c + sum_j L[j] eps_j + sum_{j<=k<Q} Qd[j,k] eps_j eps_k + [rlo, rhi],     eps_j in [-1, 1]
```

with a constant `c`, linear terms over the noise symbols (one per input element), quadratic terms
among the `quad_vars` largest-radius symbols, and an interval remainder. A smooth function `f`
is applied by expanding around `c` and bounding the Lagrange remainder over the argument's range.
The module docstring in `onnxsim/taylor.py` has the full argument; in short:

* **The remainder is exact where it can be.** For Exp, Log, Sqrt, Reciprocal and Rsqrt, and for
  Sigmoid / Tanh / Erf on boxes that do not straddle an inflection of the relevant derivative,
  `f^(k+1)` has a definite sign, so the remainder is monotone (or unimodal) in the deviation and
  its range is read off the two endpoints. Otherwise a one-sided Lagrange bound is used, with each
  side's own derivative range. For `1/x` expanded to degree 2 around 2.735 on `[1.47, 4.0]` the
  largest remainder is about 0.07, where a symmetric Lagrange bound gives 0.43.
* **Derivative ranges are exact**: every derivative used is monotone in magnitude or has explicit
  critical points, and the tests compare each against a dense grid.
* **Never looser than intervals.** Every tensor also carries an independent interval enclosure;
  the reported bounds are the intersection. This also keeps facts a polynomial cannot see
  (`exp(x) > 0`), which Softmax's `1 / sum` needs.
* **ReLU** uses the DeepZ parallelogram (exact when stable).
* **Unsupported ops** become an interval box (`precision lost at <op>` is added to `notes`), and
  if even that is unbounded the tensor is unbounded. Never a wrong bound and never a NaN.

Supported ops: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Pow` (constant exponent in {-1, -0.5, 0.5, 1, 2, 3}),
`Exp`, `Log`, `Sqrt`, `Reciprocal`, `Sigmoid`, `Tanh`, `Erf`, `Relu`, `Softmax`, `Gelu`,
`LayerNormalization`, `RMSNormalization`, `MatMul` (constant on either side, or both abstract),
`Gemm`, `Conv` (2-D, strides, pads, dilations, groups), `AveragePool`, `GlobalAveragePool`,
`BatchNormalization`, `ReduceMean`, `ReduceSum`, `Transpose`, `Reshape`, `Flatten`, `Squeeze`,
`Unsqueeze`, `Concat`, `Identity`, `Cast` (to float).

## Measured tightness

Mean output width, and its ratio to the sampled width (random, random-vertex and corner inputs,
1500 samples, through onnxruntime). Seeded; reproduce with `python scripts/taylor_tightness.py`.
`unbounded` = the method has no rule for the op. Random weights are scaled so the networks are not
saturated.

| case | box w | sampled | interval | zonotope | crown | taylor | mean_value |
|---|---|---|---|---|---|---|---|
| Softmax [2,4] | 0.1 | 0.07494 | 1 (13.34x) | 1 (13.34x) | 1 (13.34x) | 0.08085 (1.08x) | 0.09963 (1.33x) |
| Softmax [2,4] | 0.25 | 0.1865 | 1 (5.36x) | 1 (5.36x) | 1 (5.36x) | 0.2222 (1.19x) | 0.2605 (1.40x) |
| GELU fused [1,6], x in 0.3 ± w | 0.25 | 0.3639 | unbounded | unbounded | unbounded | 0.3639 (1.00x) | 0.3639 (1.00x) |
| GELU fused [1,6], x in 0.3 ± w | 1.0 | 1.344 | unbounded | unbounded | unbounded | 1.489 (1.11x) | 1.806 (1.34x) |
| GELU decomposed (Div/Erf/Add/Mul), x in ± w | 0.25 | 0.25 | 0.2994 (1.20x) | 0.2994 (1.20x) | 0.2994 (1.20x) | 0.25 (1.00x) | 0.2994 (1.20x) |
| GELU decomposed (Div/Erf/Add/Mul), x in ± w | 0.5 | 0.5 | 0.6915 (1.38x) | 0.6915 (1.38x) | 0.6915 (1.38x) | 0.5 (1.00x) | 0.6915 (1.38x) |
| LayerNorm fused [1,6], spread centres ± w | 0.05 | 0.1485 | unbounded | unbounded | unbounded | 0.1546 (1.04x) | 0.2085 (1.40x) |
| LayerNorm fused [1,6], spread centres ± w | 0.1 | 0.2961 | unbounded | unbounded | unbounded | 0.3272 (1.10x) | 0.5668 (1.91x) |
| LayerNorm decomposed, spread centres ± w | 0.05 | 0.1485 | 0.3476 (2.34x) | 0.2892 (1.95x) | 0.3476 (2.34x) | 0.1544 (1.04x) | 0.2085 (1.40x) |
| LayerNorm decomposed, spread centres ± w | 0.1 | 0.2961 | 0.7061 (2.38x) | 0.5847 (1.97x) | 0.7061 (2.38x) | 0.3258 (1.10x) | 0.5668 (1.91x) |
| 1-head attention [4,8], 32 inputs | 0.05 | 0.1699 | 0.9898 (5.82x) | 0.9898 (5.82x) | 0.9898 (5.82x) | 0.2675 (1.57x) | 0.2696 (1.59x) |
| 1-head attention [4,8], 32 inputs | 0.15 | 0.5098 | 2.969 (5.82x) | 2.969 (5.82x) | 2.969 (5.82x) | 1.504 (2.95x) | 1.605 (3.15x) |
| Conv-Sigmoid-Conv-Sigmoid [1,2,6,6] | 0.1 | 0.2732 | 0.8666 (3.17x) | 0.5323 (1.95x) | 0.454 (1.66x) | 0.6251 (2.29x) | 0.7216 (2.64x) |
| Conv-Sigmoid-Conv-Sigmoid [1,2,6,6] | 0.3 | 0.6429 | 0.9845 (1.53x) | 0.9676 (1.50x) | 0.9125 (1.42x) | 0.9845 (1.53x) | 0.9845 (1.53x) |

How to read it:

* **Where it wins.** Softmax, GELU, LayerNorm and attention: within 1.0x to 1.6x of the truth at
  narrow boxes, where the linear methods either have no rule or return the trivial interval
  (about 6x to 13x wider for Softmax and attention). LayerNorm needs a box where the row cannot
  collapse to a constant (distinct centres): if every element can sit at the row mean the variance
  approaches 0 and `rsqrt` is unbounded for *every* method.
* **Where it does not.** Two Conv-Sigmoid layers: CROWN and zonotopes are tighter. In a separate
  run (its own random weights) raising `max_vars` to 128 and `quad_vars` to 48 only moved this
  case from 2.30x to 2.07x, so the truncation knobs are not the main cause; the low-order
  expansion of a composition of saturating layers is the likely one (not isolated further).
* **It converges faster than the linear methods as the box shrinks.** On
  `Sigmoid(x) * Sigmoid(-x)` (x around 0.8, box half-widths 0.4, 0.2, 0.1, 0.05) intervals
  overestimate the true width by +163% to +172% at every box width, a degree-1 model's excess
  halves each time the box halves (+41%, +18%, +9%, +4%), and degree 2's quarters (+10%, +2.5%,
  +0.6%, +0.2%). The tests assert that pattern.
* **On a wide box it degrades.** The remainder grows like `|D|^(degree+1)`. At w = 0.15 the
  attention bound is 2.95x of the truth (still half of the interval's 5.82x), and at w = 0.3 the
  Conv-Sigmoid case is no better than the interval.

## Limits

* **Curse of dimensionality.** Storage is `n_symbols x tensor size` for the linear part and
  `quad_vars (quad_vars + 1) / 2 x tensor size` for the quadratic part, per tensor. Only the
  `max_vars` (default 64) largest-radius input elements get a symbol; the rest enter as interval
  remainders, which is sound but loses their correlation. Only the `quad_vars` (default 12)
  largest-radius symbols get quadratic terms. This is for small and medium windows, not full-size
  networks.
* **Degree is 1 or 2.** Degree 1 is a zonotope with a Lagrange remainder. Higher degrees are not
  implemented.
* **float64, not directed rounding.** After every operation the remainder is padded by
  `8 eps_64` times the magnitude of the terms, and the final bounds are widened by a relative
  `1e-9`. That is an allowance, not a proof. The bounds describe the real-number function; float32
  execution can exceed them by float32 rounding (about `1e-6` relative per op), which the
  soundness tests budget as `1e-4`.
* **"Never looser than intervals" holds up to that `1e-9` widening.**
* **Domains.** `Log`, `Sqrt`, `Rsqrt` need a strictly positive argument range and `Reciprocal` a
  range that excludes 0; otherwise the result is unbounded (not a wrong bound).
* **Mean-value form.** Rigorous for the real-number function; `f(c)` is evaluated in float64. An op
  without a differentiation rule (anything outside the list above) makes everything computed from
  it unbounded.

## Tests

`tests/test_taylor.py` checks, bottom up: every derivative formula against finite differences;
every derivative range and critical point against a dense grid (both an upper *and* a lower
tightness bound); the Lagrange remainder of each function against the true remainder, in both the
exact-endpoint and the one-sided branch; the polynomial range and the product, pointwise, including
remainders; each function applied to a Taylor model, at every noise value; and every supported op
and composition against onnxruntime with every intermediate tensor exposed, at degree 1 and 2.
It also runs this page's examples and compares their output with what is quoted here.
