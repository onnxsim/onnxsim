# The box part of `onnxsim.zonotope`: carrying one-sided noise as a radius

`onnxsim.zonotope` keeps every value as an affine form over shared noise symbols, stored **densely**:
a tensor with `k` symbols holds `k x tensor size` floats, and a Conv applied to it is run `k` times.
For a conv net with image side `H` both factors grow with the image area, so cost grows with `H^4`
(measured: 1 s at 8x8, 10 s at 16x16, 110 s and 7.2 GB at 32x32 for `quant_verify` on a 3-layer
8-channel net). Most of those symbols describe noise that **exists in one model only**: the rounding
noise of a quantizer, and the relaxation error of the second model's nonlinearities. Nothing ever
cancels against them, so tracking each as its own dense symbol buys nothing.

The *box part* carries such noise as one non-negative radius per element instead:

    x_i = c_i + sum_j G[j, i] * eps_j + e_i,        |e_i| <= r_i,   every e_i independent.

This is a **superset** of the zonotope the same noise would generate as one symbol per element (a box
contains it), so every bound stays sound. It costs `O(tensor size)` instead of `O(k x tensor size)`.

## API

```python
zonotope.propagate(model, input_ranges, box_inputs=None, box_fresh=None)
zonotope.bound_difference(orig, simplified, input_ranges, box_inputs=None, box_fresh=None)
zonotope.proves_equal(orig, simplified, input_ranges, atol, rtol, box_inputs=None, box_fresh=None)
```

* `box_inputs` -- graph-input names, or `fnmatch` patterns such as `"__qnoise_*"` (the names
  `onnxsim.quant_verify` gives its rounding-noise inputs). Those inputs are carried as radii, not
  dense symbols.
* `box_fresh` -- put the error of every `Relu` / `Sigmoid` / `Tanh` relaxation and interval fallback
  into the box part. Default `None` means `bool(box_inputs)`. In `bound_difference` it applies to the
  **second** model only (see below).
* `result.stats` (`propagate`) and `bound.stats` (`bound_difference`, per model) report the largest
  symbol count, the largest dense generator array (elements) and the largest box part, so the saving
  is measurable.

With neither option, behaviour is **bit-identical** to before: no tensor has a box part and every
existing test passes unchanged (checked bit for bit against `origin/master` on `propagate` and
`bound_difference` for an MLP, Conv+BN+Relu, a residual net, a Sigmoid/Tanh chain and a
Conv-GlobalAveragePool-Flatten-Gemm net, including real `onnxsim.simplify` output).

## What each op does to the box

| op | box radius of the result |
|---|---|
| Conv / MatMul / Gemm / BatchNormalization | the radius pushed through `abs(W)` (no bias): `abs(sum w_i e_i) <= sum abs(w_i) r_i` |
| Add / Sub of two zonotopes | `r_a + r_b`, for either sign (a box added to itself doubles: sound, looser) |
| Mul by a constant, Neg | `r * abs(k)` |
| Relu / Sigmoid / Tanh | `lam * r` plus the relaxation error: the relaxation slope `lam` is in `[0, 1]`, so it scales the box; the error goes into the box (`box_fresh`) or into a fresh dense symbol |
| Reshape, Flatten, Transpose, Squeeze, Unsqueeze, Concat | the same shape op on the radius |
| GlobalAveragePool, ReduceMean, ReduceSum | the same reduction of the radii (non-negative coefficients) |
| anything else | interval fallback from the bounds, which include the box: sound, with a `precision lost at <op>` note |

## What is boxed, and what must not be

A box gives up **cancellation**. Two things follow, and they decide what to list in `box_inputs`.

1. **Only noise that exists once.** A box that *both* models contain does not cancel in their
   difference: `Conv(x) - Conv(x)` is exactly zero with dense symbols and not with a boxed `x`
   (below). So list only inputs one model reads -- quantizer rounding noise -- never a real input.
2. **The first model's own relaxation symbols stay dense.** In `bound_difference` the second model's
   nonlinearities are *paired* with the first's (`out_b = out_a - (f(x_a) - f(x_b))`), so `out_b`
   contains `out_a`'s relaxation symbols, and the difference cancels them only if they are tracked.
   Boxing them as well (measured, not offered as an API) made the 8x8 bound **66x** looser
   (187.7 against 2.83).

A box also loses cancellation wherever the same noise reaches one element along more than one path:

* a residual `Add`;
* **two consecutive summing layers**: `sum_m abs(W2[i,m]) * abs(W1[m,j]) >= abs(sum_m W2[i,m] W1[m,j])`.
  Signed paths through different middle elements cancel in a zonotope but not in a box;
* pooling or reductions that merge elements.

The box equals the dense bound when the noise passes **at most one summing layer**
(Conv / MatMul / Gemm) with only elementwise and shape ops around it. Paths through a `Relu` are
already approximated by the relaxation in the dense engine, which is why the measured loss on real
quantized models is modest (see below).

A single symbol shared across the elements of a channel or tensor would be **unsound**: independent
per-element errors are not a subset of perfectly correlated ones. It is deliberately not offered.

## Examples

One-sided noise in a difference problem. The float model ignores two noise inputs; the "quantized"
one adds them at two sites, which is how `quant_verify` lays out a quantizer's rounding noise:

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import zonotope

rng = np.random.default_rng(0)


def model(body, **weights):
    m = parser.parse_model('<ir_version: 8, opset_import: ["" : 15]> ' + body)
    m.graph.initializer.extend(numpy_helper.from_array(v.astype(np.float32), k) for k, v in weights.items())
    return m


w1, w2 = rng.standard_normal((4, 3, 3, 3)) * 0.4, rng.standard_normal((4, 4, 3, 3)) * 0.3
sig = "float[1,3,8,8] x, float[1,4,8,8] n0, float[1,4,4,4] n1"
ref = model(f"m ({sig}) => (float[1,4,4,4] y) {{ c = Conv<pads=[1,1,1,1]>(x, w1)  r = Relu(c)  y = Conv<strides=[2,2], pads=[1,1,1,1]>(r, w2) }}", w1=w1, w2=w2)
noisy = model(f"m ({sig}) => (float[1,4,4,4] y) {{ c = Conv<pads=[1,1,1,1]>(x, w1)  cn = Add(c, n0)  r = Relu(cn)  d = Conv<strides=[2,2], pads=[1,1,1,1]>(r, w2)  y = Add(d, n1) }}", w1=w1, w2=w2)
ranges = {"x": (-1.0, 1.0), "n0": (-0.08, 0.08), "n1": (-0.05, 0.05)}

dense = zonotope.bound_difference(ref, noisy, ranges)
boxed = zonotope.bound_difference(ref, noisy, ranges, box_inputs=["n*"])
for name, r in (("dense", dense), ("box_inputs=['n*']", boxed)):
    s = r.stats["simplified"]
    print(f"{name:18s} bound {r.worst:.3f}  symbols {s['max_symbols']:5d}  dense generator elements {s['max_generator_elements']:8d}")
```
```text
dense              bound 0.925  symbols  1024  dense generator elements   245760
box_inputs=['n*']  bound 0.925  symbols   448  dense generator elements   114688
```

The same bound with 56% fewer symbols and 53% fewer dense generator elements. What it costs when the
same noise does reach an element twice (`c - c`, the worst case for a box):

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import zonotope

rng = np.random.default_rng(1)
w = rng.standard_normal((3, 3, 3, 3)).astype(np.float32)
m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> m (float[1,3,5,5] x) => (float[1,3,5,5] y) '
    "{ c = Conv<pads=[1,1,1,1]>(x, w)  y = Sub(c, c) }"
)
m.graph.initializer.append(numpy_helper.from_array(w, "w"))
box = {"x": (-1.0, 1.0)}
for label, kw in (("dense", {}), ("box_inputs=['x']", {"box_inputs": ["x"]})):
    lo, hi = zonotope.propagate(m, box, **kw).bounds("y")
    print(f"{label:18s} y = c - c lies in [{lo.min():.3f}, {hi.max():.3f}]")
```
```text
dense              y = c - c lies in [0.000, 0.000]
box_inputs=['x']   y = c - c lies in [-35.644, 35.644]
```

Both are sound; the second is the reason `box_inputs` is for one-sided noise only.

## Measured

Everything below was measured on one 32-core machine, with the TestPyPI wheel's compiled extension
under this branch's Python, on **synthetic random-weight** networks (not real models). The scaling
cells come from `scripts/zonotope_box_noise_bench.py`; the unit tests pin the properties, not these
numbers.

### Soundness

54 cases -- three nets (Conv-Relu-Conv, a 3-layer MLP, a residual conv net) x six quantization
schemes (static int8 / int16, qoperator, weight-only int8 / int4, dynamic) x input boxes of +-0.25,
+-1, +-4 -- run through `quant_verify.verify` with the engine switched to each mode (by wrapping
`zonotope.bound_difference`; `quant_verify.py` is untouched) and compared with the largest error
onnxruntime produced for 100 random inputs plus a 30-step corner search:

* **0 violations** (observed error above the certified bound) in any mode.
* The certified bound is a median 12-17x above the observed error in every mode (median
  observed/bound 0.08 dense, 0.06 boxed): it is a safety certificate, not an accuracy estimate.

### Tightness, where the dense engine finishes

Bound ratio against the all-dense engine over the same 54 cases (1.0 = identical):

| mode | median | p10 | p90 | max |
|---|---|---|---|---|
| `box_inputs` only (`box_fresh=False`) | 1.05 | 1.00 | 1.93 | 2.24 |
| `box_inputs` + `box_fresh` (the default when `box_inputs` is set) | 1.47 | 1.00 | 2.57 | 2.91 |

By net, `box_inputs` + `box_fresh`: the conv nets stay at or below 1.5x (group medians 1.0-1.5),
the fully connected **MLP** reaches group medians of 1.7-2.8x and a maximum of 2.9x. That is the composition loss described above: in a dense layer every
element reaches every later element through many paths, and the box treats them as independent. A
box is therefore a better fit for conv-shaped (local) structure than for fully connected stacks.

### Scaling on conv nets

A 3-layer conv net (stride 2 in the middle layer) with the rounding noise of all three activations
and fake-quantized weights, input box +-1, one certified bound (`quant_verify` makes several such
calls; this is the first, the total bound). `symbols` = largest symbol count on any tensor; `gen` =
largest dense generator array; `cap` = the default 16384-symbol cap was reached, so generators were
merged into boxes (sound, but cancellation with the other model is lost). **Timings were taken while
other jobs were running on the same machine; single run per cell.**

| channels | px | engine | bound | time | peak RSS | symbols | gen | cap |
|---|---|---|---|---|---|---|---|---|
| 8 | 8 | dense | 2.83 | 0.24 s | 351 MB | 2,240 | 0.007 GB | |
| 8 | 8 | `box_inputs` + `box_fresh` | 3.33 | 0.14 s | 329 MB | 832 | 0.003 GB | |
| 8 | 16 | dense | 2.95 | 2.4 s | 887 MB | 8,960 | 0.11 GB | |
| 8 | 16 | `box_inputs` | 3.27 | 1.9 s | 787 MB | 5,888 | 0.08 GB | |
| 8 | 16 | `box_inputs` + `box_fresh` | 3.45 | 3.1 s | 633 MB | 3,328 | 0.05 GB | |
| 8 | 32 | dense | 34.9 | 23.6 s | 7.2 GB | 16,384 | 1.07 GB | **cap** |
| 8 | 32 | `box_inputs` | 32.8 | 22.3 s | 6.2 GB | 16,384 | 1.07 GB | **cap** |
| 8 | 32 | `box_inputs` + `box_fresh` | **3.45** | 18.7 s | 5.1 GB | 13,312 | 0.74 GB | |
| 8 | 64 | `box_inputs`, `box_inputs` + `box_fresh` | -- | killed by the 16 GB memory cap (about 20 s) | | | | |
| 16 | 16 | dense | 6.53 | 16.9 s | 2.4 GB | 16,384 | 0.43 GB | **cap** |
| 16 | 16 | `box_inputs` | 7.38 | 13.1 s | 1.9 GB | 11,008 | 0.29 GB | |
| 16 | 16 | `box_inputs` + `box_fresh` | 7.80 | 9.2 s | 1.3 GB | 5,888 | 0.16 GB | |
| 16 | 32 | `box_inputs` + `box_fresh` | 521 | 110 s | 14.5 GB | 16,384 | 2.1 GB | **cap** |
| 16 | 32 | `box_inputs` | -- | killed by the 16 GB memory cap (50 s) | | | | |
| 16 | 64 | `box_inputs`, `box_inputs` + `box_fresh` | -- | killed by the 16 GB memory cap (34 s) | | | | |

Dense at 16 channels and 32 px was not attempted: it timed out at 180 s in an earlier measurement.

What this shows, plainly:

* **The box buys little at a given size, not an order of magnitude.** Memory falls to 0.55-0.9x of
  dense in every row. Time is mixed: 1.3x (`box_inputs`) and 1.8x (`+ box_fresh`) faster at
  16 channels / 16 px, but at 8 channels / 16 px `box_inputs` is 1.25x faster and
  `+ box_fresh` is 0.8x, i.e. *slower* (3.1 s against 2.4 s) -- single runs under concurrent
  load, so differences of this size are not reliable. (The dense run at 16 channels / 16 px also hit
  the symbol cap, so its work is not directly comparable.)
* **Its larger effect is on the symbol cap.** At 8 channels / 32 px the dense default (and
  `box_inputs` alone) hit the 16384-symbol cap and the bound degraded **10x** (34.9, 32.8 against
  3.45); with `box_fresh` the symbol count stays under the cap and the bound is the same 3.45 as at
  16 px. So some of the earlier "slow" results were also *wrong-ish* (very loose), not only slow.
* **It does not reach 64 px.** At 64 px both box modes are killed by the 16 GB memory cap within
  about 20 s at 8 channels (and 34 s at 16 channels); at 32 px / 16 channels `box_inputs` alone is
  killed after 50 s, and `+ box_fresh` finishes in 110 s using 14.5 GB but at the symbol cap, with a
  bound (521) that is meaningless.
* **The bound does not depend on the image size once the image exceeds the receptive field.** The
  8-channel `box_inputs` + `box_fresh` bound is 3.452 at both 16 px and 32 px: interior outputs see
  identical neighbourhoods and the worst case is attained there. So the cheaper way to scale is to
  analyse a *window* (the way `certify`'s `proved-reduced` shrinks spatial size), and the box part
  makes a larger window affordable; it is not a substitute for choosing one.

### The next bottleneck

After boxing, the symbols left are exactly the **shared** ones: one per input element (`3 H W`)
plus one per unstable `Relu` neuron of the *first* model, whose relaxation symbols must stay dense
because the second model is built from them. At 8 channels: 832 = 192 + 640 at 8 px, and
13,312 = 3,072 + 10,240 at 32 px, i.e. every `Relu` neuron of the float model (8x32x32 +
8x16x16) contributes a symbol (with these random weights and a +-1 box every neuron is unstable --
the counts match exactly; trained networks have stable neurons, which create no symbol). At 64 px that is about 53,000 symbols on tensors of 32,768 elements: one dense array of
`53,248 x 32,768 x 8` bytes = 14.0 GB, which is what the memory cap stops.

To check that the first model's symbols really have to stay dense, boxing them too (an experiment,
not an API: the first evaluator's `box_fresh` forced on) leaves only the `3 H W` input symbols
(192 at 8 px) and makes the bound **66x** looser at 8 px (187.7 against 2.83) and 69x at 16 px
(204.5 against 2.95): the difference stops cancelling.

So the remaining cost is structural to the paired-evaluation design (carry both models' values and
subtract at the end). The principled fix is to carry the **difference** between the two models'
values as the primary object -- it involves only the symbols of what differs (weight error, noise) --
and keep the float model's values as plain intervals for the relaxation slopes. That is a different
engine, not an option on this one, and is not done here.

## Limits

* Only noise that exists once belongs in `box_inputs`; a box shared by both models does not cancel.
* The box is looser than dense wherever the same noise reaches an element along several paths
  (consecutive summing layers, pooling, residual adds); measured 1.0-3x on the quantized nets above,
  worst on fully connected stacks.
* It does not make 64-pixel windows feasible: see the next bottleneck.
* All numbers are on synthetic random-weight nets, single runs, on a machine that was running other
  jobs; read the ratios, not the seconds.
* Float64 with the usual `1e-9` relative widening; the bounds describe the real-number function, and
  float32 execution can exceed them by float32 rounding.
* `quant_verify` is not switched over here: it calls `zonotope.bound_difference` without the options.
  Passing `box_inputs=["__qnoise_*"]` there is a separate, small change.
