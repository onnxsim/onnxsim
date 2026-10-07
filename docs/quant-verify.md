# Certified bounds for quantized models (`onnxsim.quant_verify`)

`onnxsim.accuracy.measure_accuracy_drop` tells you how far a quantized model is from its float model
**on the data you gave it**. `onnxsim.precision_estimator` gives a fast, data-free *heuristic*. Neither is
a guarantee. `quant_verify.verify` is: for **every** input in a box you choose, it proves an upper bound on
`|float(x) - quantized(x)|` per output.

```python
report = quant_verify.verify(float_model, quantized_model, {"x": (-1.0, 1.0)})
report.worst          # certified max |float - quantized| over all outputs
report.within(atol)   # sound check against a tolerance
report.sites          # which quantizer contributes how much
report.hazards        # what to look at before trusting the number
```

## How it works

The quantized model is rewritten into an equivalent **float graph with bounded noise**, and the float
model and that graph are compared with the shared-symbol machinery of `onnxsim.zonotope`
(`bound_difference`), so everything the two models share cancels exactly.

| In the quantized model | In the analysed graph |
|---|---|
| `QuantizeLinear` → `DequantizeLinear` on an activation | the value plus a noise input with `|e| <= scale/2` (per-tensor or per-channel) |
| `DequantizeLinear` on a **constant** (per-tensor, per-axis, blockwise `block_size`, int4 through int16) | the exact dequantized constant, so the *weight* error is structure, not noise |
| `QuantizeLinear` on a constant | evaluated exactly (round half to even, saturate) |
| `QLinearConv`, `QLinearMatMul` | the float op on the dequantized operands, then the output quantizer's noise |
| `DynamicQuantizeLinear` followed by `MatMulInteger → Cast → Mul`, or `MatMulIntegerToFloat` | `MatMul(x + e, W_dequantized)` with `|e| <= s_max/2`, `s_max = (max(hi,0) - min(lo,0)) / 255` over the propagated range of `x` |

The per-site bound is the Z3-proved lemma in `tests/test_formal_verify_quantize_round_trip.py`
(`|x - DQ(Q(x))| <= scale/2` inside the representable range, for any tie rule); the composition through
MatMul/Conv/Gemm is the lemma in `tests/test_formal_verify_quantized_mac_bound.py`, applied automatically by
the zonotope arithmetic rather than re-derived.

### Saturation is charged, never assumed away

A value outside the representable range `[(qmin - zp) * scale, (qmax - zp) * scale]` saturates, and
`scale/2` does not bound that error. For a quantizer whose input ranges over `[lo, hi]` the noise interval is

    [ -max(scale/2, hi - Rmax),  +max(scale/2, Rmin - lo) ]

(one-sided: above the range the error is `Rmax - x <= 0`, below it `Rmin - x >= 0`). `[lo, hi]` is the
*propagated* range of the tensor entering the quantizer **including upstream noise**, which depends on the
noise of earlier sites, so the intervals are grown to a fixed point (they only ever grow). If no fixed point
is reached, the affected noise is `inf` -- never a wrong finite number. A site that can saturate for some
input in the box is listed in `report.hazards` and flagged `CLIPPED`.

`verify(..., clipping="clamp")` instead models saturation exactly as `clamp(x, a, b) = a + Relu(x-a) -
Relu(x-b)` plus `scale/2` noise. It is sound but **looser** (measured below), so it is not the default.

## Example

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import accuracy, precision_estimator, quant_verify

rng = np.random.default_rng(1)


def weights(*shape):
    return (rng.standard_normal(shape) * 0.5).astype(np.float32)


model = parser.parse_model(
    '<ir_version: 9, opset_import: ["" : 21]> '
    "m (float[2,32] x) => (float[2,8] y) "
    "{ a = MatMul(x, W1) b = Add(a, B1) r = Relu(b) c = MatMul(r, W2) d = Relu(c) y = MatMul(d, W3) }"
)
for name, shape in dict(W1=(32, 64), B1=(64,), W2=(64, 64), W3=(64, 8)).items():
    model.graph.initializer.append(numpy_helper.from_array(weights(*shape), name))

cfg = accuracy.QuantizationConfig(scheme="static", dtype="int8", num_calibration_samples=16)
quantized = accuracy.quantize(model, cfg)

print("--- inside the calibrated range (box +-0.25)")
box = {"x": (-0.25, 0.25)}
report = quant_verify.verify(model, quantized, box)
print(report)
print("within 100:", report.within(100.0), " within 10:", report.within(10.0))
print("observed (corner search):", round(quant_verify.observed_error(model, quantized, box, n=100, adversarial=300), 4))

print("--- box wider than calibrated (+-4)")
wide = quant_verify.verify(model, quantized, {"x": (-4.0, 4.0)})
print(wide)
print("observed (corner search):", round(quant_verify.observed_error(model, quantized, {"x": (-4.0, 4.0)}, n=100, adversarial=300), 4))
```

Output (a 32-64-64-8 ReLU MLP, `static` int8 quantization, calibration on 16 random batches):

<!-- doctest-output -->
```text
--- inside the calibrated range (box +-0.25)
quantization verify: worst |float - quantized| <= 45.3354
  weights/constants only: 4.24741
  site _v_17 (QuantizeLinear): scale/2 0.01366, radius 0.01366, alone 45.3354
within 100: True  within 10: False
observed (corner search): 0.807
--- box wider than calibrated (+-4)
quantization verify: worst |float - quantized| <= 1815.78
  weights/constants only: 67.9585
  site _v_17 (QuantizeLinear): scale/2 0.01366, radius 0.9407 CLIPPED, alone 1815.78
  hazard: site _v_17: propagated range exceeds the representable range; clipping error is included in its radius (0.940661 vs scale/2 0.0136578)
observed (corner search): 63.093
```

Reading it: the certified bound dominates the error actually seen (0.807 and 63.1), and it grows by a factor
of 40 when the box exceeds what calibration saw, because the quantizer saturates there. The "weights/constants
only" line is the error that remains if every activation quantizer were exact; the per-site `alone` figure is
the bound with only that site's noise active, which is how you find the quantizer worth leaving in float.

## Validation

`observed_error(..., adversarial=N)` runs onnxruntime on random inputs, then on random box *vertices*, then
hill-climbs by flipping subsets of input elements to the opposite bound. It is an empirical lower bound on the
true worst case, so a certified bound below it is a bug. On models produced by onnxsim's own quantizers
(`accuracy.quantize`: `static`, `static_int16`, `qoperator`, `weight_only` int8 and int4, `dynamic`,
`dynamic_fused`) over three small float models (conv-relu-conv, a 3-layer MLP, a residual conv block) and
three boxes (±0.25, ±1, ±4) the bound was never below the observed error: 39 scheme/model/box combinations
in the sweep script (the schemes that change each model), and the 42 non-skipped combinations of
`tests/test_quant_verify.py` (3 models x 6 schemes x 3 boxes; the quantizer leaves the conv models unchanged
under `dynamic` and `int4`). Observed against certified, a few rows of that sweep:

| model | scheme | box | observed (adversarial) | certified | ratio |
|---|---|---|---|---|---|
| conv-relu-conv | static int8 | ±0.25 | 0.240 | 1.94 | 8x |
| conv-relu-conv | static int16 | ±0.25 | 0.075 | 0.212 | 2.8x |
| conv-relu-conv | weight_only int8 | ±1 | 0.278 | 0.822 | 3.0x |
| residual | static int8 | ±1 | 0.479 | 2.98 | 6.2x |
| 3-layer MLP | static int8 | ±1 | 1.20 | 64.8 | 54x |
| 3-layer MLP | weight_only int4 | ±1 | 13.0 | 339 | 26x |
| 3-layer MLP | dynamic int8 | ±1 | 1.06 | 31.9 | 30x |
| 3-layer MLP | static int8 | ±4 (saturating) | 73.1 | 2073 | 28x |
| conv-relu-conv | qoperator int8 | ±1 (saturating) | 3.65 | 46.0 | 13x |
| 3-layer MLP | qoperator int8 | ±1 (saturating) | 1.33 | 725 | 544x |

The ratios are honest looseness, not just a weak baseline: the corner search raises the observed error by
1.4-10x over uniform sampling on these models, and the ratios above are against the corner-search number. Three things drive the large ones: the bound is a
worst case over the whole box while real inputs are far from the corners; ReLU relaxations keep every
unstable neuron; and saturation noise is treated as independent per element although it is a deterministic
function of the input. The 544x row is that last effect: the quantized output range was calibrated on random
data and the box corners exceed it.

**Choosing how saturation is modelled** (measured, `qoperator` int8, box ±1): independent one-sided noise
gave certified 46.0 / 725 / 86 for the conv net / MLP / residual net; the exact `clamp` model gave 123 / 2213
/ 6778 (all sound). The clamp's unstable ReLUs are paired against the float model's ReLUs in the zonotope
difference and the pairing loosens the bound a lot, so `clipping="noise"` is the default.

**How strongly the tests constrain the implementation.** Deliberately unsound mutants of the verifier were
run against the test suite: halving the rounding noise, ignoring saturation, flipping the saturation sign and
halving the dynamic scale were all caught. But halving the rounding noise is caught only by the hand-computed
single-site tests, *not* by the real-quantizer domination tests, because their bounds are loose enough to
survive it. The domination tests catch gross unsoundness; the exact-value tests carry the fine-grained claim.

## Output ranges

`quant_verify.verify_against_annotation(quantized_model, input_ranges)` checks every annotated output range
(`onnxsim.ranges`) on the quantized graph -- inputs over the box and every rounding/saturation outcome -- with
`onnxsim.crown.verify_output_ranges`. `proved=False` means *not proved*, not violated.

## Compared with `precision_estimator`

Only one quantity is directly comparable: the int32 accumulator check. `verify` flags `MatMulInteger` /
`QLinearConv` / `QLinearMatMul` nodes whose worst case (`sum|w - zp_w| * max|q_x - zp_x|`, per output
channel, with the actual quantized weights but the **full** activation integer range) exceeds `2**31 - 1`.
That is the same kind of data-independent worst case as the estimator's `K * 127 * 255` (it can be lower when
the weights are small), and it ignores the activation range the model can actually produce, which
`onnxsim.interval.quantization_bounds` does use -- so it is not the tighter tool there. The estimator's other output, the
relative RMS figure (`0.0234` for the model above), is by its own documentation a heuristic for ranking
models, not a bound, so there is nothing to call tighter or looser: it estimates a typical error, this
certifies a worst case (45.3 absolute for the same model against an observed 0.81).

## Limits -- read before relying on a number

* **Real arithmetic.** The bound describes the real-number functions of both models. float32 evaluation adds
  roundoff (about `1e-6` relative per op); budget for it (the tests allow `1e-4`).
* **The integer pipeline is not verified.** That the int32 accumulation, the requantisation multiply and
  `Cast` reproduce the QDQ semantics bit for bit is *assumed*, not proved here. Accumulator overflow is only
  *checked* (full-range worst case) and reported as a hazard.
* **Unsupported quantized-domain ops give an infinite bound with a reason**: `ConvInteger`, `QLinearAdd` and
  the other `QLinear*` ops, `MatMulNBits`, FP8, blockwise `QuantizeLinear` on activations, activations
  quantized in an earlier model (an `int8` graph input). An integer-domain tensor that the rewrite cannot
  account for (an accumulator consumed outside the recognised pattern) is refused, not guessed.
* **Dynamic quantization** bounds the data-dependent scale by the propagated range of the whole tensor, so it
  is worst-case in the scale as well.
* **Tightness degrades with depth and width** (see the ratios above); `onnxsim.zonotope` stores generators
  densely, so this suits small and medium models, not full-size networks.
* Only static shapes are analysed; every graph input needs a finite range.
