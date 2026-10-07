# Numerical-safety lint (`onnxsim.numeric_lint`)

`numeric_lint.lint` answers one question: **for which operations can this model produce `inf`,
`nan` or an overflow for *some* input inside a box you declare?** It propagates the box through the
graph with `onnxsim.interval` (optionally refined with `onnxsim.crown`) and checks each operation's
domain and range. It is opt-in and nothing in `simplify()` calls it.

```python
report = numeric_lint.lint(model, {"image": (0.0, 1.0)}, dtype="fp16", witness=100)
print(report)               # findings, worst first
report.ok                   # no can-fail finding under the declared ranges
report.confirmed            # findings a real input was found for
```

or from the shell: `python -m onnxsim.numeric_lint model.onnx --range image=0,1 --dtype fp16 --witness 100`
(exit status 1 when there is a `can-fail` finding; `--json` for machine output).

## A first example

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import numeric_lint

m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 17]> '
    "g (float[4] a, float[4] b) => (float[4] y) { y = Div(a, b) }"
)
# b may be 0 for some input in its box; a cannot hurt
report = numeric_lint.lint(m, {"a": (1.0, 2.0), "b": (-1.0, 1.0)}, witness=20)
print(report)
f = report.findings[0]
print(f.rule, f.severity, f.certainty, f.observed_what, "=", f.observed)
```
```text
numeric lint (fp32): 1 can-fail, 0 can-lose-precision, 0 info; 1 confirmed
[can-fail/CONFIRMED] div-by-zero at Div_0 (Div): denominator 'b' can be 0 for some input (4 of 4 elements have an interval containing 0, hull [-1, 1]): result is inf/nan [witness search saw min |denominator| = 0]
    fix: add a small epsilon to the denominator (x / (d + eps)), clamp it away from 0, or narrow the declared input range if the real data cannot reach 0
div-by-zero can-fail confirmed min |denominator| = 0.0
```

## The rules

| rule | severity | what it flags |
|---|---|---|
| `div-by-zero` | can-fail | `Div` / `Reciprocal` whose denominator interval contains 0 |
| `log-domain` | can-fail | `Log` on a domain that includes <= 0 |
| `sqrt-domain` | can-fail | `Sqrt` on a domain that includes < 0 |
| `pow-domain` | can-fail | `Pow` with a constant exponent: negative base with a fractional exponent, or a base that can be 0 with a negative exponent |
| `exp-overflow` | can-fail | `Exp` input above ln(dtype max) (88.7 for fp32/bf16, 11.09 for fp16). A decomposed softmax (`Exp` feeding a `ReduceSum`) without the max subtraction gets a tailored fix. `x - ReduceMax(x)` is recognised as `<= 0` |
| `saturation` | can-lose-precision / info | `Sigmoid` / `Tanh` / `Softplus` driven past +-ln(dtype max): a kernel built on a naive `exp` overflows there. Native `Softmax` in fp16 whose logits span more than ln(65504) is info. Both are **implementation dependent** |
| `norm-variance` | can-fail / info | `LayerNormalization` / `RMSNormalization` / `SimplifiedLayerNormalization`: variance (or mean of squares) above the dtype max; the sum of squares above it (implementation dependent: a kernel that accumulates in fp32 is fine); a row that can be constant while `epsilon` underflows to 0 in the dtype. A merely *subnormal* epsilon is info |
| `range-overflow` | can-fail | any float tensor whose interval exceeds the dtype's largest finite value (65504 for fp16), reported **where it arises** (all of its inputs are still in range); later tensors that overflow because an input already does are counted in `report.consequences`, not listed |
| `const-overflow` | can-fail, confirmed | a constant that is not representable in the dtype. The classic case is a `-3.4028235e38` attention mask, which is `-inf` in fp16 and turns a fully masked softmax row into `nan`. It involves no input, so it is `confirmed` by construction |
| `range-underflow` | can-lose-precision / info | fp16: constants that flush to 0 (a warning once at least 0.1% of the nonzero values do, else info) or go subnormal (info); activations that stay below the smallest normal number |
| `int32-wrap` | can-fail | `MatMulInteger` / `ConvInteger` accumulator that can leave the int32 range, bounded exactly from the constant weights when there are any |
| `cast-range` | can-fail | `Cast` to a narrower integer type or fp16 whose range the input interval exceeds |
| `dead-op` | info | `Clip` that never changes its input or always saturates, `Relu` that is the identity or always 0, `Where` whose condition is decided |

Each finding carries the node, the tensor the claim is about, the **proving interval**, the **exact
input ranges the analysis assumed** (`finding.assumption`), a suggested fix, and flags
`implementation_dependent`, `certainty`, `unbounded`, `refined`.

## What a finding means

Read this before acting on one.

* A finding is a **sound over-approximation**: *if the real inputs stay inside the declared box, this
  can happen for some input in the box*. It is **not** a confirmed bug (`certainty="possible"`).
* Plain intervals lose correlations, so on deep networks the hulls grow much faster than the real
  activations. Most `range-overflow` and `norm-variance` findings on a deep model are looseness
  (see the real-model results below).
* With `witness=N` the linter searches (the box midpoint and corners, random points, then
  hill-climbing; at most `N` runs of the model in onnxruntime, plus any `witness_data` you pass first)
  for an input that actually breaks the condition in the **fp32 reference execution**, and upgrades the
  finding to `confirmed` only then. For fp16 and bf16 the run is still fp32: *confirmed* means the exact
  value is outside the dtype's range, not that an fp16 kernel was executed. `report.replay(i)` re-runs
  the stored witness input so you can check it yourself.
* `finding.observed` is the extreme the search saw (maximum |value|, minimum |denominator|, ...), so a
  certified bound next to a much smaller observed value shows how loose the bound is. A search that
  finds nothing is **not** a proof of safety: it only means this search did not reach the failure.
* Everything rests on the ranges you declare (`onnxsim.ranges` annotations or the `input_ranges`
  argument). An input with no range is unbounded; findings that exist only because of that are
  suppressed and counted in `report.suppressed_unbounded` unless `include_unbounded=True`.

A certified bound that is *not* loose, next to what the search reaches:

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import numeric_lint

# y = relu(x @ ones) * 2 with 64 inputs in [0, 2000]: the interval bound is the exact worst case here,
# so the witness search reaches it; on a deep network the same search would stay far below the bound
m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 17]> '
    "g (float[1,64] x) => (float[1,1] y) { a = MatMul(x, W) b = Relu(a) y = Mul(b, two) }"
)
m.graph.initializer.extend(
    [
        numpy_helper.from_array(np.ones((64, 1), np.float32), "W"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
    ]
)
for dtype in ("fp32", "fp16"):
    report = numeric_lint.lint(m, {"x": (0.0, 2000.0)}, dtype=dtype, witness=20)
    over = [f for f in report.findings if f.rule == "range-overflow"]
    print(dtype, "range-overflow:", [(f.tensor, f.certainty) for f in over], "folded:", report.consequences)
f = [f for f in report.findings if f.rule == "range-overflow"][0]
print("certified up to", round(f.interval[1]), "; search saw", round(f.observed))
print("replayed witness gives", float(report.replay(report.findings.index(f))["a"].max()))
```
```text
fp32 range-overflow: [] folded: 0
fp16 range-overflow: [('a', 'confirmed')] folded: 2
certified up to 128000 ; search saw 128000
replayed witness gives 128000.0
```

The overflow is reported at the `MatMul` (`a`); the `Relu` and `Mul` after it overflow too, but only
because their input does, so they are folded into it (`folded: 2`).

And one that *is* loose, where CROWN (`refine=True`) settles it:

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import numeric_lint

# z2 = -z1, so y = 100 * (relu(z) + relu(-z)) = 100 * |z1| <= 6e4: it fits fp16. Plain intervals treat
# the two relus as independent (each up to 600) and report 1.2e5.
m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 17]> '
    "g (float[1,2] x) => (float[1,1] y) { z = MatMul(x, W1) h = Relu(z) y = MatMul(h, W2) }"
)
m.graph.initializer.extend(
    [
        numpy_helper.from_array(np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32), "W1"),
        numpy_helper.from_array(np.array([[100.0], [100.0]], np.float32), "W2"),
    ]
)
box = {"x": (-300.0, 300.0)}
plain = numeric_lint.lint(m, box, dtype="fp16", witness=40)
f = plain.by("can-fail")[0]
print(f.rule, f.certainty, "certified", round(f.interval[1]), "observed", round(f.observed))
refined = numeric_lint.lint(m, box, dtype="fp16", refine=True)
print("after CROWN refinement:", [x.rule for x in refined.by("can-fail")], "dropped", refined.refined_away)
```
```text
range-overflow possible certified 120000 observed 60000
after CROWN refinement: [] dropped 1
```

Here the true maximum is 60000, which fits fp16, but the interval bound is 120000: the finding stays
`possible` however long the search runs, and CROWN proves it away. `refine=True` is dense and only
meant for small models.

## dtype

`dtype` is the precision the model will be **run** in. The thresholds:

| dtype | largest finite | smallest normal | smallest subnormal | ln(largest) |
|---|---|---|---|---|
| `fp32` | 3.40e38 | 1.18e-38 | 1.40e-45 | 88.72 |
| `fp16` | 65504 | 6.10e-05 | 5.96e-08 | 11.09 |
| `bf16` | 3.39e38 | 1.18e-38 | 9.18e-41 | 88.72 |

The same model can be clean in fp32 and not in fp16, which is the point: the second example above, and a
mask constant:

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import numeric_lint

# a masked attention score: the usual -3.4e38 "minus infinity" is not an fp16 number
m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 17]> '
    "g (float[1,4] x, bool[1,4] keep) => (float[1,4] y) "
    "{ neg = Constant<value = float {-3.4028235e38}>() y = Where(keep, x, neg) }"
)
rng = {"x": (-1.0, 1.0), "keep": (0, 1)}
for dtype in ("fp32", "fp16"):
    report = numeric_lint.lint(m, rng, dtype=dtype)
    print(dtype, [(f.rule, f.severity) for f in report.findings])
fp16 = numeric_lint.lint(m, rng, dtype="fp16")
print(fp16.findings[0].fix)
```
```text
fp32 []
fp16 [('const-overflow', 'can-fail')]
use a finite mask value that fits the dtype (e.g. -1e4 / -65504 in fp16)
```

## Continuing past operations the interval analysis does not model

`onnxsim.interval` cannot bound every op, and an op it does not model leaves everything downstream
unknown. For ops whose output bound follows from their mathematics the linter **cuts** the graph there:
the op's output becomes a fresh input with that range, and propagation continues. Each cut is sound:

| op | output range used |
|---|---|
| `Gather` | min/max of the gathered rows of a constant table (or of the data's hull) |
| `LayerNormalization` | `scale * z + bias` with `|z| <= sqrt(N)` per normalised row (N = features), since the sum of `z^2` is below N |
| `SimplifiedLayerNormalization`, `RMSNormalization` | `|y| <= sqrt(N) * |scale|` |
| `Cast` | the input hull, clamped to the target type's range |
| `Where` | the union of the two branches' hulls (whichever one the condition picks) |
| comparisons and logical ops | `[0, 1]` |
| `Gelu` | `[-0.1701, max(0, hi)]` |
| `ConstantOfShape`, `Abs`, `Sin`/`Cos`/`Sign`, `Reciprocal` (denominator excludes 0) | from the constant, the hull, or `[-1, 1]` |
| a decomposed softmax `Div(e, ReduceSum(e))` with `e >= 0` | `[0, 1]` |

`report.cuts` says how many were applied and `report.unanalysed` lists tensors that still have no
bound. Without cuts DistilBERT is analysed up to its first `Gather` (99 of 297 tensors bounded); with
them it is analysed completely.

## Evaluation on real models

Three real exports, each linted in fp32 and fp16, with three kinds of evidence for every finding:

* the **certified** interval (what the linter claims),
* a **150-run witness search** on synthetic inputs in the box, and
* **real data** replayed in fp32 with every flagged tensor exposed: 200 ImageNet validation images
  (evenly spaced over a 5,000-image subset) for the two CNNs, and all 872 SST-2 validation sentences for
  DistilBERT.

Models: torchvision ResNet18 and MobileNetV2 with ImageNet weights (`batch`x3x224x224, opset 13) and
DistilBERT fine-tuned on SST-2 (64 tokens, opset 17, native `LayerNormalization`/`Softmax`). Ranges:
ImageNet-normalised per-channel boxes (`(0 - mean)/std .. (1 - mean)/std`) for the CNNs;
`input_ids` in `[0, 30521]` and `attention_mask` in `[0, 1]` for DistilBERT. Python 3.12.13, onnx 1.23.1,
onnxruntime 1.30.0, numpy 2.5.3, one CPU (AMD Ryzen AI Max+ 395, 32 threads, shared with other jobs, so
the times are wall-clock upper bounds and one run each).

| model | dtype | analysis | can-fail | by construction | confirmed by the 150-run search | triggered by real data | unconfirmed |
|---|---|---|---|---|---|---|---|
| ResNet18 | fp32 | 2.0 s | 0 | 0 | 0 | 0 | 0 |
| ResNet18 | fp16 | 2.2 s | 2 | 0 | 0 | 0 | 2 |
| MobileNetV2 | fp32 | 1.0 s | 0 | 0 | 0 | 0 | 0 |
| MobileNetV2 | fp16 | 1.1 s | 0 | 0 | 0 | 0 | 0 |
| DistilBERT | fp32 | 14.9 s | 0 | 0 | 0 | 0 | 0 |
| DistilBERT | fp16 | 13.7 s | 44 | 1 | 0 | 0 | 43 |

(*By construction* is the `const-overflow` finding, which involves no input. The 150-run search adds
under 1.5 s on every model. Replaying real data took 1.1 s for 200 images and 24 s for 872 sentences.)

**What it found**

* **fp32: nothing** on any of the three (no finding of any severity), which matches these models
  running fine in fp32.
* **MobileNetV2 fp16: no `can-fail`.** Its `ReLU6` (`Clip(0, 6)`) caps every activation at 6, so the
  intervals stay tight; the only output is 50 info notes about subnormal weights. Architectures with a
  bounded activation are easy for this analysis.
* **DistilBERT fp16: one real hazard.** The attention mask is built from the constant
  `-3.4028235e38` (`float32` minimum), which is `-inf` in fp16: a fully masked softmax row becomes `nan`.
  It is reported by `const-overflow` and is `confirmed` by construction.
* **Everything else flagged as `can-fail` was not confirmed by anything**, and the bounds it rests on are
  very loose:

| model | finding | n | certified | seen on real data | seen by the witness search |
|---|---|---|---|---|---|
| ResNet18 fp16 | `range-overflow` (layer2.0 `conv1` and its downsample `Conv`) | 2 | 7.1e4 .. 1.3e5 | 3.6 .. 3.9 | 5.4 .. 6.3 |
| DistilBERT fp16 | `range-overflow` (attention, `out_lin`, `lin2` `MatMul`s) | 18 | 6.8e4 .. 6.5e7 | 6.7 .. 917 | 2.1 .. 398 |
| DistilBERT fp16 | `norm-variance` (variance overflow) | 12 | 4.6e9 .. 1.0e13 | 1.7 .. 2130 | - |
| DistilBERT fp16 | `norm-variance` (`epsilon=1e-12` underflows to 0, a row can be constant) | 13 | - | min row variance 0.0013 .. 0.87 | - |

  For the 30 DistilBERT range and variance findings the certified bound is **162x to 2.6e12x** the
  real-data maximum (median 2.7e5x); for the 2 ResNet18 findings it is 1.8e4x to 3.5e4x. On these models
  the unconfirmed `can-fail` findings look like **interval looseness, not bugs**: 45 of the 46
  `can-fail` findings (98%) are unconfirmed, and the one that is not is the deterministic mask constant.
  "Unconfirmed" is evidence, not proof. The largest real values were 3.9 (ResNet18), 917 (DistilBERT
  activations) and 2130 (DistilBERT row variances) against an fp16 limit of 65504, so DistilBERT's real
  activations use about 1.4% of the range and its variances about 3%, which is not "nowhere near" but is
  not close either.
* **The witness search is not redundant with real data.** On ResNet18 the synthetic corners reached
  larger activations (5.4 and 6.3) than 200 real images did (3.6 and 3.9), and on DistilBERT's attention
  `MatMul`s the search saw up to 398 against 917 for real sentences. Neither dominates; `witness_data=`
  lets you feed real samples to the same search.
* **The `epsilon=1e-12` findings are real facts about the model but theoretical here.** The model's
  LayerNorm epsilon (the HF default `1e-12`) is `0` as an fp16 number, so a row with zero variance would
  give `nan`; the real data never gets near that (minimum row variance 0.0013). They are `possible` and
  stay that way.
* **Underflow is mostly noise.** DistilBERT has 61 `range-underflow` notes (60 info and one
  `can-lose-precision`: `layer.5.attention.k_lin.bias`, where at least 0.1% of the nonzero values flush to
  0); the first version of this rule flagged 31 as warnings, mostly 15 flushed weights out of 23 million,
  which is why the fraction threshold exists.

**Cost.** Analysis is 1 to 2 s for the two CNNs and about 14 s for DistilBERT. The DistilBERT figure is
the repeated interval propagation across the 9 cut-and-continue passes (17 cuts), each of which
re-runs numpy matmuls and shape inference over the whole graph; a quiet-machine profile showed 12.8 s of
13.5 s in `_analyse`. It is a one-off check, not something to run inside `simplify()`.

## Limits

* **Intervals are loose on deep networks.** This is the dominant source of noise, not a bug; see the
  numbers above. Use `witness=` / `witness_data=` to separate real hazards from looseness, and
  `refine=True` on small models.
* **Findings depend on the declared ranges.** A wrong range gives wrong findings in both directions.
* **fp16 and bf16 are judged on the fp32 reference execution.** No fp16 kernel is run.
* **Not modelled:** underflow of intermediate sums, accumulation order, fused kernels (several rules are
  marked implementation dependent), and `ArgMax`/`TopK` ties.
* **Ops the interval analysis cannot bound and that have no cut rule** stop the analysis downstream;
  `report.unanalysed` shows what was lost.
* **`witness` is a search, not a proof.** A finding left `possible` after it is still only possible.
* `refine=True` uses dense CROWN: small models only. It silently keeps the interval result if CROWN
  cannot handle the graph.
