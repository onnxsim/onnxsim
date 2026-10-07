# Certified floating-point roundoff bounds (`onnxsim.fp_error`)

`onnxsim.interval`, `crown`, `zonotope` and `certify` reason about the **real-number**
semantics of a model. A model *runs* in float32 (or float16 / bfloat16). This module bounds
the gap:

```
|execution(x) - real_semantics(x)|  <=  bound        for every x in the input box
```

and, from it, gives a **certified `atol`** for comparing the executions of a model and its
simplified version, so `simplify(check_atol=...)` / `certify(atol=...)` need not be an arbitrary
constant.

Every snippet below marked as a doctest is executed by `tests/test_docs_fp_error.py`, so the
numbers on this page cannot drift from the code.

## Quick start

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import fp_error

rng = np.random.default_rng(0)
model = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[1,64] x) => (float[1,16] y) { y = Gemm(x, W, B) }"
)
model.graph.initializer.extend(
    [
        numpy_helper.from_array((rng.standard_normal((64, 16)) / 8).astype(np.float32), "W"),
        numpy_helper.from_array((rng.standard_normal(16) / 8).astype(np.float32), "B"),
    ]
)

for precision in ("fp32", "fp16", "bf16"):
    r = fp_error.roundoff_bound(model, {"x": (-1.0, 1.0)}, precision=precision)
    print(f"{precision}: certified |error| <= {r.worst:.1e}")
    assert r.bounded

# Without an input box the magnitudes are unbounded, and so is the error. Never a wrong number.
print("no box:", fp_error.roundoff_bound(model).worst)
```

Output:

```
fp32: certified |error| <= 3.1e-05
fp16: certified |error| <= 2.7e-01
bf16: certified |error| <= 2.8e+00
no box: inf
```

`roundoff_bound` returns a `RoundoffResult`: `outputs` (elementwise bound per graph output),
`errors` (every analysed tensor), `notes` (why something is `inf`, which pass ran) and
`unsupported` (ops that fell back to an infinite error). `worst` is the largest output bound.

## What is proved, and what is only assumed

Read this before trusting a number.

**Proved** (pure arithmetic, no hypothesis about any runtime), under the textbook model
`fl(a op b) = (a op b)(1 + d)`, `|d| <= u` (`u = 2^-24` fp32, `2^-11` fp16, `2^-8` bf16), plus an
absolute term `eta` for underflow:

* Add/Sub/Mul/Div/Neg/Pow(2), and the exact ops (Relu, Clip, Max/Min, Abs, Reshape, Transpose,
  Slice, Gather, Concat, MaxPool, ...), which add no rounding of their own.
* A dot product / convolution of `n` terms is within `gamma_n * sum|w||x|`, `gamma_n = n u / (1 - n u)`
  (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 3.5). This holds for **any
  summation order** (sequential, pairwise, blocked, SIMD lanes, tree reductions) and with fused
  multiply-add. That is what makes it valid for kernels that reduce in different orders.
* Softmax with max-subtraction: input sensitivity `expm1(2 ||e||_inf)` plus a relative error built from
  the argument rounding, the exp error, the `n`-term sum and the final division.

**Assumed** (a hypothesis about the runtime; each is a parameter or a stated limit):

* **Library functions** (Sigmoid, Tanh, Exp, Erf, Sqrt, Log, Softplus) are not correctly rounded in
  fast kernels. Their error is the `LibmModel`: `rel * u * |f| + abs * u` per call. The defaults are
  about 3x the worst error measured for onnxruntime 1.x CPU fp32 on 2 million points per range:

  | op | worst observed (units of `u`) | default `rel` / `abs` |
  |---|---|---|
  | Exp | 1.3 rel | 4 / 0 |
  | Erf | 1.4 rel | 4 / 0 |
  | Sqrt | 1.0 rel | 2 / 0 |
  | Tanh | 5.3 rel | 16 / 0 |
  | Softplus | 5.5 rel, 8 abs | 16 / 4 |
  | Log | 4 rel near 1 (17 abs over `[1e-6, 1e6]`) | 12 / 4 |
  | Sigmoid | 4 rel away from the tails, **3.0 abs** (relative error is ~1e8 `u` in the tails) | 12 / 8 |

  These are measurements of **one runtime**. A different kernel, or fp16/bf16 (where the same
  multipliers are reused in units of that precision), can be worse. Pass your own `lib=` to certify
  against a runtime you have characterised.
* **The graph executes as written.** Runtime rewrites change roundoff. The BatchNorm bound is robust to
  the most common one, folding BatchNorm into the preceding Conv/Gemm weights (its constant-rounding
  term is scaled by `sum|w||x|` of the producer instead of `|conv output|`); onnxruntime does this by
  default and the validation below runs with it both on and off. Other fusions (fused GELU, fused
  attention, LayerNorm kernels) are not modelled. **Winograd/FFT convolution has larger error than
  any direct summation and is not covered.** A kernel that accumulates in a *wider* format only makes
  the bound conservative.
* **No overflow or NaN.** If a computed magnitude may exceed the format's largest finite number
  (fp16: 65504) the error is reported `inf`, and so is the case of a dot product whose partial sums could.
* **Inputs are the numbers actually fed:** the box describes values already representable in the
  analysed precision. Constants stored in a *wider* format than the analysed precision are rounded
  (relative error `u`), so analysing an fp32 model at `precision="fp16"` is a "what if I ran it in
  fp16" bound.
* The real-value ranges come from `onnxsim.interval` (float64 with a `1e-14` relative widening) and,
  by default, `onnxsim.zonotope`; their own float64 rounding is ~1e-16 relative, far below every bound
  here, and is not added.

Anything without a rule yields an infinite error for that tensor and a note, never a silently small
bound.

## A certified `atol` for comparing two models

`tolerance_for` bounds `|run(orig) - run(simplified)|` by the real-arithmetic difference
(`zonotope.bound_difference`) plus both models' roundoff bounds:

`|o^ - s^| <= |o^ - o| + |o - s| + |s - s^|`.

<!-- doctest -->
```python
import numpy as np
import onnxruntime as ort
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import fp_error

k = 4
rng = np.random.default_rng(2)
f = lambda *s: rng.standard_normal(s).astype(np.float32)
orig = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[1,3,12,12] x) => (float[1,4,10,10] y) { "
    "c = Conv(x, W, B)  b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)  y = Relu(b) }"
)
orig.graph.initializer.extend(
    numpy_helper.from_array(v, n)
    for n, v in dict(
        W=f(k, 3, 3, 3) * 0.3, B=f(k), g=rng.uniform(0.5, 1.5, k).astype(np.float32),
        be=f(k), mu=f(k), var=rng.uniform(0.5, 2, k).astype(np.float32),
    ).items()
)
simplified, _ = onnxsim.simplify(orig, certify=False)  # BatchNorm folded into the Conv

tol = fp_error.tolerance_for(orig, simplified, {"x": (-1.0, 1.0)})
print(f"certified atol = {tol.atol['y']:.1e}")
print(f"  real-arithmetic difference : {tol.real_difference['y']:.1e}")
print(f"  roundoff of the original   : {tol.roundoff_orig['y']:.1e}")
print(f"  roundoff of the simplified : {tol.roundoff_simplified['y']:.1e}")

# It really covers the two executions (corner inputs maximise accumulation):
def run(m, x):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), opts).run(None, {"x": x})[0]

worst = max(
    float(np.max(np.abs(run(orig, x) - run(simplified, x))))
    for x in (rng.choice([-1.0, 1.0], size=(1, 3, 12, 12)).astype(np.float32) for _ in range(20))
)
assert worst <= tol.atol["y"]
```

Output:

```
certified atol = 3.9e-05
  real-arithmetic difference : 4.2e-07
  roundoff of the original   : 2.3e-05
  roundoff of the simplified : 1.5e-05
```

The real-arithmetic difference is the one figure that is not reproducible to the last digit: it
comes from float32 constants that `simplify` folds, and the folding arithmetic differs slightly
between platforms (an aarch64 build prints `3.9e-07` where x86-64 prints `4.2e-07`). It is orders of
magnitude below the roundoff terms, so the certified `atol` is unaffected at the digits shown.

For comparison, the default `check_atol=1e-5` is *below* that certified absolute value. The default
check also has `check_rtol=1e-4`, which covers it wherever the output is not close to zero, so the
exposure is outputs near zero; this module gives the absolute figure that is actually certified
instead of a constant. `certify(..., atol="certified")` uses it, see the next section; wiring it into
`simplify(check_atol=...)` is still a follow-up.

### From `certify`: `atol="certified"`

`certify()` proves closeness in *real* arithmetic against `atol=1e-5, rtol=1e-4`. With
`atol="certified"` that proof runs as before and each proved output also gets the certified fp32
tolerance `X = t + roundoff(orig) + roundoff(simplified)`, where `t` is the real-arithmetic part
(the smaller of the zonotope bound and what the proof itself established: `0` for a structural proof,
`atol + rtol * max|simplified|` for an SMT or shrink proof; a congruence proof is *not* used, because
it propagates an input gap through an op whose gain may exceed 1). The verdict is `proved-fp32`; the
real-arithmetic verdicts stay in `report.real_outputs`. This example was run (the printed digits can
differ slightly on other platforms):

```python
import numpy as np
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import certify

rng = np.random.default_rng(0)
k = 4
model = parser.parse_model(
    """<ir_version: 8, opset_import: ["" : 13]>
    m (float[1,3,6,6] x) => (float[1,4,4,4] y) {
      c = Conv(x, W, B)
      b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
      y = Relu(b)
    }"""
)
weights = dict(
    W=rng.standard_normal((k, 3, 3, 3)),
    B=rng.standard_normal(k),
    g=rng.uniform(0.5, 1.5, k),
    be=rng.standard_normal(k),
    mu=rng.standard_normal(k),
    var=rng.uniform(0.5, 2, k),
)
model.graph.initializer.extend(
    numpy_helper.from_array(v.astype(np.float32), n) for n, v in weights.items()
)
simplified, _ = onnxsim.simplify(model, certify=False)

box = {"x": (-1.0, 1.0)}
default = certify.certify(model, simplified, input_ranges=box)
print("default   :", default.outputs, "(real arithmetic, atol=1e-5, rtol=1e-4)")

report = certify.certify(model, simplified, input_ranges=box, atol="certified")
t = report.tolerance
print("certified :", report.outputs, "| real arithmetic was:", report.real_outputs)
print(f"  fp32 tolerance X   = {t.atol['y']:.2e}")
print(f"    real part        = {t.real_bound['y']:.2e}")
print(f"    roundoff (orig)  = {t.roundoff_orig['y']:.2e}")
print(f"    roundoff (simpl) = {t.roundoff_simplified['y']:.2e}")
print("  within 1e-5?", report.within(1e-5), "| within 1e-3?", report.within(1e-3))

print("no range  :", certify.certify(model, model, atol="certified").outputs)
```

```
default   : {'y': 'proved-congruence'} (real arithmetic, atol=1e-5, rtol=1e-4)
certified : {'y': 'proved-fp32'} | real arithmetic was: {'y': 'proved-congruence'}
  fp32 tolerance X   = 1.68e-04
    real part        = 1.34e-06
    roundoff (orig)  = 9.71e-05
    roundoff (simpl) = 7.00e-05
  within 1e-5? False | within 1e-3? True
no range  : {'y': 'skipped'}
```

What to read from it:

* "Proved at 1e-5" and "proved equal within 1.7e-4 on fp32" are different statements; the second is the
  one about what actually runs. Here X is about 17x the default threshold, and nearly all of it is the
  roundoff terms (the folded constants differ by ~1e-6).
* An output with **no finite tolerance is `skipped`, never proved**: no input range (last line:
  the models are identical, which *is* proved in real arithmetic, but the roundoff of an unbounded
  input is unbounded), an op `fp_error` has no rule for (`Sin`, `Floor`, ...), a model too large for
  the zonotope bound, or an exhausted time budget. The reason is in the report's windows.
* A **skipped** output (the real-arithmetic proof at the default thresholds is missing) still gets its
  certified tolerance in `report.tolerance`, with a `for information` window. That is a bound, not
  an equality: a wrong rewrite also has a finite X, which is why the verdict stays `skipped` and
  `report.within(...)` is `False` unless every output is proved.
* It is opt-in and costs a zonotope bound plus two roundoff bounds, so the default-on check inside
  `simplify()` is unchanged. `fp_error` is loose on deep nets (a 128-wide MLP is thousands of times the
  observed error; the table below has the numbers), so read X as a safe bound, not an estimate.

Measured with `onnxruntime` optimisations **disabled** (otherwise it fuses Conv+BN in the original too
and both sessions run the same kernel, an observed difference of exactly 0), input box `[-1, 1]`:

| case | default (real, 1e-5) | certified | X | observed executed difference |
|---|---|---|---|---|
| Conv+BN+Relu 6x6 | proved-congruence | proved-fp32 | 1.7e-4 | 3.8e-6 |
| Conv+BN+Relu 8x8 | proved-congruence | proved-fp32 | 1.3e-4 | 3.8e-6 |
| MatMul/Add/Relu/MatMul/Add 16-8-4 | proved-smt | proved-fp32 | 3.5e-4 | 0 |
| MatMul/Add 128-64-4 | proved-smt | proved-fp32 | 1.2e-1 | 0 |
| 2 x Conv+BN+Relu 8x8 | skipped | skipped (X 3.5e-3 .. 9.6e-3 over 5 weight draws) | n/a | 1.5e-5 |
| 2 x Conv+BN+Relu 16x16 | skipped | skipped | n/a | 1.3e-5 |

The two-layer conv stacks are the instructive rows: the observed fp32 difference (1.3e-5 to 1.5e-5)
is already *above* the default 1e-5 threshold, so the real-arithmetic proof at that threshold is
rightly unavailable (zonotope bound ~3e-5), and certified mode does not hide that.

## Why two passes (`tight=True`)

The first pass propagates each layer's error through `|W|`: it assumes the errors of all units can
align against the sign of every weight, and it compounds that per layer (about `sum|w|` per layer).
The true propagation is `W @ e`, where signs cancel. So the forward bound is sound but, for deep nets,
very pessimistic.

The tight pass makes every rounding event a bounded noise symbol (radius = the local rounding the
first pass certified for that node) and bounds `|perturbed model - clean model|` with
`zonotope.bound_difference` on shared symbols, so weights act as `W`. Both bounds are sound; the
elementwise minimum is returned. It needs a bounded box and a modest graph (`<= 8192` noise
symbols; generators are dense), and is skipped with a note otherwise or when a value may overflow.

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import fp_error

rng = np.random.default_rng(5)
widths = [48, 64, 64, 64, 8]
body, init, prev = [], {}, "x"
for i in range(len(widths) - 1):
    init[f"W{i}"] = (rng.standard_normal((widths[i], widths[i + 1])) / np.sqrt(widths[i])).astype(np.float32)
    init[f"B{i}"] = (rng.standard_normal(widths[i + 1]) * 0.1).astype(np.float32)
    body.append(f"a{i} = MatMul({prev}, W{i})\n b{i} = Add(a{i}, B{i})")
    prev = f"b{i}"
    if i < len(widths) - 2:
        body.append(f"r{i} = Relu(b{i})")
        prev = f"r{i}"
body.append(f"y = Identity({prev})")
mlp = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> m (float[1,48] x) => (float[1,8] y) { '
    + "\n".join(body) + " }"
)
mlp.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in init.items())

box = {"x": (-1.0, 1.0)}
a = fp_error.roundoff_bound(mlp, box, tight=False, ranges="interval")
b = fp_error.roundoff_bound(mlp, box, tight=False)
c = fp_error.roundoff_bound(mlp, box)
print(f"interval ranges, forward only : {a.worst:.2e}")
print(f"zonotope ranges, forward only : {b.worst:.2e}")
print(f"+ tight pass                  : {c.worst:.2e}")
assert c.worst <= b.worst <= a.worst
```

Output:

```
interval ranges, forward only : 1.58e-02
zonotope ranges, forward only : 1.37e-02
+ tight pass                  : 7.84e-03
```

## Validation: how loose is it, and has it ever been exceeded?

Empirical check against real executions: onnxruntime 1.x CPU float32 against a float64 reference of
the same graph, with graph optimisations **off** (executed as written) and **on** (default fusions,
including Conv+BatchNorm folding), 300 samples per net (40 of them +/-1 corners, which maximise
accumulation). Observed error vs certified bound (`ranges="auto"`, `tight=True`):

| net | max observed | certified bound | bound / observed |
|---|---|---|---|
| Conv+BN+Relu 16x16, x in [-1,1] | 1.9e-6 | 3.1e-5 | 16x |
| Conv+BN+Relu 16x16, x in [0,1] | 5.1e-7 | 2.5e-5 | 49x |
| Residual conv net | 2.8e-7 | 1.1e-3 | ~4000x |
| MLP 64-128-128-10 | 4.3e-7 | 6.8e-3 | ~16000x |
| Tanh / Sigmoid / Softmax head | 1.7e-7 | 3.5e-4 | ~2000x |
| MLP 6 x 256 (deep) | 5.7e-7 | 30 | ~5e7x |

**The bound was never exceeded** (the largest observed/bound ratio over all of the above was 0.10,
with graph optimisations both on and off). An adversarial search (hill-climbing the +/-1 corners to maximise
observed/bound over 6 random weight draws per net, 120 steps each, optimisations on and off) found
at most **0.13** of the bound for Conv+BN+Relu and at most 0.005 for the others.

The looseness is real and has two sources, neither of which is a bug:

* **Worst-case rounding.** The any-order bound `gamma_n sum|w||x|` is about `sqrt(n)` times what
  random-sign accumulation does, and the `sum|w||x|` / `|sum wx|` ratio adds more. Exact fp16
  emulation shows it: a sequential 256-term fp16 dot product reaches ~0.4% of its bound.
* **Worst-case inputs.** The bound covers every input in the box, including adversarial ones random
  sampling never reaches; error scales with the (worst-case) activation magnitudes. Hill climbing
  closes only part of that gap.

A single layer is within ~16-50x; the gap grows with depth because errors compound at their worst
case. A tighter, still certified, bound for a deep network needs a model of the actual summation
order, which is exactly what this module declines to assume.

## Library functions and unusual ops

Elementary functions use the error model above. `Softmax` needs opset >= 13 (earlier opsets flatten to
2-D and are not modelled). `LayerNormalization`, `Winograd` convolution, attention fusions and any
op without a rule are reported `inf` with a note. A different runtime, kernel or fusion invalidates the
*assumed* part of the guarantee, not the arithmetic part.

## Limits

* Dense memory: the tight pass and zonotope ranges are for small and medium graphs (`ranges="auto"`
  skips zonotopes above 4096 input elements and the tight pass above 8192 noise symbols; both say so in
  `notes`).
* The deep-network bound is pessimistic by orders of magnitude (above); use it to *prove* a tolerance
  is safe, not to estimate the error you will see.
* `tolerance_for` is absolute only; there is no `rtol`, which keeps it sound where an output crosses
  zero.
* Not wired into `simplify()` / `certify()` yet.
