# Verifying integer-quantized models (`onnxsim.quant_int_verify`)

`onnxsim.quant_int_verify` answers two questions about a model that has been quantized into
integer form, with certificates instead of sampled inputs:

1. **Can an int32 accumulator wrap?** (`no-wrap`, a soundness finding)
2. **Does the integer pipeline compute what its fake-quant (QDQ / float) reference computes?**
   (`requant-1lsb`, `dynamic-rel-error`, `weight-fidelity`, soundness findings; and the informational
   `requant-bit-exact`, `float-cast-exact`)

It understands `MatMulInteger`, `ConvInteger`, `QLinearMatMul`, `QLinearConv`, and onnxsim's own
`DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul` chain (what `onnxsim.quantize_dynamic` emits).
It complements, and does not replace, `onnxsim.precision_estimator` (which assumes the activation uses
all 8 bits) and `onnxsim.interval.quantization_bounds` (which bounds the float-vs-quantized *error*):
this module is about the integer arithmetic itself.

## Usage

`verify_model(integer_model, reference_model=None, input_ranges=None)` returns a `ModelReport`; each layer has
findings with a `verdict` (`proved` / `refuted` / `skipped`) and a `severity` (`soundness` decides
`report.ok`; `informational` never does). `verify_layer(node, consts, ...)` does one node.

### The int32 boundary, decided exactly

For `uint8` activations and `int8` weights each product is at most `255 * 127`, so a one-channel
reduction of depth `K` can wrap exactly when `K * 255 * 127 > 2**31 - 1`:

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser
from onnxsim import quant_int_verify as Q


def matmul_integer(k):
    m = parser.parse_model(
        f'<ir_version: 8, opset_import: ["" : 13]> m (uint8[1,{k}] a) => (int32[1,1] y) {{ y = MatMulInteger(a, W) }}'
    )
    m.graph.initializer.append(numpy_helper.from_array(np.full((k, 1), 127, np.int8), "W"))
    return m


for k in (66311, 66312):
    f = Q.verify_model(matmul_integer(k)).layers[0].of("no-wrap")[0]
    print(k, f.verdict)
```
```text
66311 proved
66312 refuted
```

A `refuted` finding carries a concrete input that reproduces the wrap, and it can be replayed on
onnxruntime with an integer "twin" of the layer (so it works for `QLinear*` layers too, whose output is
requantized and hides the accumulator):

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser
from onnxsim import quant_int_verify as Q


def matmul_integer(k):
    m = parser.parse_model(
        f'<ir_version: 8, opset_import: ["" : 13]> m (uint8[1,{k}] a) => (int32[1,1] y) {{ y = MatMulInteger(a, W) }}'
    )
    m.graph.initializer.append(numpy_helper.from_array(np.full((k, 1), 127, np.int8), "W"))
    return m


m = matmul_integer(66312)
rep = Q.verify_model(m)
f = rep.layers[0].of("no-wrap")[0]
observed, exact = Q.replay_no_wrap(m, rep.layers[0].name, f)
print("exact   ", int(exact[0, 0]))
print("observed", int(observed[0, 0]))
```
```text
exact    2147514120
observed -2147453176
```

`observed` is exactly `exact - 2**32`: onnxruntime wraps as two's complement.

### onnxsim's quantizers, against their float reference

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser
import onnxsim
from onnxsim import quant_int_verify as Q

rng = np.random.default_rng(0)
ref = parser.parse_model('<ir_version: 8, opset_import: ["" : 13]> m (float[2,64] x) => (float[2,8] y) { y = MatMul(x, W) }')
ref.graph.initializer.append(numpy_helper.from_array(rng.standard_normal((64, 8)).astype(np.float32), "W"))
q = onnxsim.quantize_dynamic(ref)
report = Q.verify_model(q, reference_model=ref)
print(report.ok)
for f in report.layers[0].findings:
    print(f.kind, f.severity, f.verdict)
```
```text
True
no-wrap soundness proved
float-cast-exact informational proved
dynamic-rel-error soundness proved
weight-fidelity soundness proved
```

## What each finding means

| kind | severity | claim |
|---|---|---|
| `no-wrap` | soundness | for every input the layer can receive, the int32 accumulator stays inside `[-2**31, 2**31-1]` |
| `requant-1lsb` | soundness | (`QLinear*`) wherever the integer output differs from the ideal real-arithmetic requantization, it differs by exactly one step, never more |
| `dynamic-rel-error` | soundness | (dynamic chain) `Cast<float>(acc) * (Xs * Ws)` is within a stated relative error of the exact `acc * Xs * Ws`, for every operand value |
| `weight-fidelity` | soundness | (needs `reference_model`) every integer weight is the round-to-nearest quantization of a reference weight |
| `requant-bit-exact` | informational | the fp32 requantization equals the ideal one for every reachable accumulator, or else a reachable *tie hazard* is listed |
| `float-cast-exact` | informational | `Cast<float>(acc)` is exact (`|acc| <= 2**24`); beyond that the cast rounds |
| `internal-inconsistency` | soundness | the closed form and the Z3 encoding disagreed; never expected, and never swallowed |

`skipped` always carries a reason (a non-constant scale, a per-row zero point, a 3-D convolution, ...).
A skipped soundness finding is not `proved`, so `report.ok` is false: an unverified layer is never
reported as verified.

## How each claim is decided

**No-wrap is exact, not a bound.** One output channel's accumulator is `bias + sum_k w_k * d_k` with
`d_k = a_k - a_zero_point` ranging independently over `[d_lo, d_hi]`. Its extremes are attained at box
vertices: `max = P*d_hi - N*d_lo` and `min = P*d_lo - N*d_hi`, where `P` and `N` are the sum of the
positive weights and of the absolute values of the negative ones. So a wrap that the bound allows is
*reachable*, which is why `refuted` comes with a witness. `tests/test_quant_int_verify.py` checks the
closed form against brute-force enumeration. For convolutions, padded taps contribute exactly 0 (the input
is padded with its zero point), so with padding the bound is only attained at a position whose receptive
field lies fully inside the input; without the input shape the layer is `skipped`, not guessed.

**Activation ranges are real, not the dtype's.** The default is the dtype's full range (sound, pessimistic).
`input_ranges` narrows it: a float input feeding a `QuantizeLinear` is mapped through interval analysis to
the codes it can produce; an integer graph input takes its range as codes. A narrower range proves layers the
full range cannot.

**Dynamic quantization couples the codes to the zero point.** `DynamicQuantizeLinear` picks one zero point
for the whole tensor, so the codes and `Xzp` are not independent. For a fixed zero point the maximum is
`P*(255 - zp) + N*zp`, linear in `zp`, so over the possible zero points it is attained at an endpoint. A
non-negative input range forces `zp = 0`, a non-positive one `zp = 255`, a sign-mixed one allows `0..255`.
That is tighter than the naive `255 * sum|w|`.

**Requantization is checked for all accumulators at once.** The runtime computes
`saturate(rne(fl32(acc) * m32) + zy)`; the ideal specification computes `saturate(rne(acc * m) + zy)` in exact
arithmetic. The two real numbers differ by at most a relative `eps` (the cast, the multiply, and the multiplier's
own rounding), so their roundings can differ only if a half-way tie `t = k + 1/2` lies between them, i.e. only if
`|acc*m - t| <= eps*|t| / (1 - eps)`. There are at most `qmax - qmin` ties, so the candidate accumulators are
a finite, small set; `requant_differences` enumerates each window and evaluates both pipelines exactly. The test
suite checks the enumerated set against exhaustive evaluation of every accumulator in a 120,000-wide range.
This is the decomposition that scales: a per-output dot product plus one function checked once over its
whole input range, instead of a proof per layer size.

<!-- doctest -->
```python
import numpy as np
from onnxsim import quant_int_verify as Q

m_real, m32 = Q.requant_multiplier(np.float32(0.02), np.float32(0.013), np.float32(0.021))
diffs = Q.requant_differences(m_real, m32, 7, 0, 255, -60000, 60000)
print(len(diffs), diffs)
a = diffs[0]
print(Q.fp32_requant(a, m32, 7, 0, 255), Q.real_requant(a, m_real, 7, 0, 255))
```
```text
3 [2625, 11025, 13125]
39 40
```

At accumulator 2625 the fp32 pipeline produces 39 and the ideal one 40: a tie that a runtime rounding
half-away, or using a double multiplier, would resolve the other way. That is how two runtimes silently disagree
by one LSB. Whether such an accumulator is *reachable* is decided separately (constructively, then with Z3 for
small layers); `requant-bit-exact` is `refuted` only when a reachable one is found, and `skipped` when
reachability could not be decided.

**The dynamic chain's float arithmetic is bounded exactly.** `fl(fl(acc) * fl(Xs*Ws))` equals
`acc*Xs*Ws*(1+d1)(1+d2)(1+d3)` with `|d_i| <= 2**-24` (`d1 = 0` when `|acc| <= 2**24`). The product is
multilinear in the `d_i`, so its extremes are at the 8 corners, which are enumerated exactly.

**Weight fidelity** checks `|w_ref - scale * (wq - zp)| <= scale/2 + |w_ref| * 2**-23` per element. The second
term is the fp32 rounding of the quantizer's own division and grows with `|w_ref|`. A *fixed* slack was a bug
found while measuring this module: on a 4096 x 4096 layer it flagged 9 correct weights (of 16.7M) sitting within
about 1e-6 of a tie at `|w/s| ~ 127`.

<!-- doctest -->
```python
import numpy as np
from onnxsim import quant_int_verify as Q

lo, hi = Q.accumulator_range(np.array([3, -2, 5]), 0, 255)
print(lo, hi)
rng = np.random.default_rng(0)
w = rng.standard_normal((6, 3)).astype(np.float32)
s = (np.abs(w).max(axis=0) / 127).astype(np.float32)
wq = np.rint(w / s).astype(np.int8)
print(Q.weight_fidelity(w, wq, s, 0, axis=1)[0])
wq[0, 0] += 3
verdict, detail, idx = Q.weight_fidelity(w, wq, s, 0, axis=1)
print(verdict, idx)
```
```text
-510 2040
proved
refuted (0, 0)
```

## Z3: small, bounded, and isolated

Z3 is used, but deliberately lightly. The closed forms above decide the claims; Z3 re-derives the no-wrap
claim from two's-complement semantics for layers with at most `z3_max_k` (default 24) taps per channel, as a
bit-vector problem (32-bit modular accumulation next to a 64-bit exact one), and a disagreement is reported
as `internal-inconsistency`. It also models the fp32 requantization with Z3's floating-point theory
(`z3_requant_expr`, checked against onnxruntime/numpy in the tests), and finds a witness input for a target
accumulator (`z3_reach`).

Z3 proofs are sensitive to the state earlier work leaves in the process (the CI hang fixed in PR #2082 was
exactly that), so:

* every solver call has a timeout and returns `skipped` rather than hanging;
* every proof gets a fresh `z3.Context()`, and every constructor that takes an optional `ctx` is passed it
  explicitly (several default to the *global* context and silently mix with the fresh one);
* the encodings are pure bit-vectors / small linear integer problems, with no `ForAll`.

## Limits and what is not claimed

* The verdicts are exact for the **modelled** semantics: two's-complement int32 accumulation, round-half-to-even,
  saturation, fp32 multiply. A runtime that deviates is not described. `probe_u8s8_saturation()` runs a tiny
  extreme-value `MatMulInteger` because some AVX2 `u8 x s8` kernels sum adjacent products in saturating int16,
  which is not wraparound. On the machine used for these checks onnxruntime matched exact int32 arithmetic,
  which says nothing about other CPUs.
* The requantization model (`sat(rne(fl32(acc) * fl32(fl32(sa*sb)/sy)) + zy)`) was checked bit-exactly against
  onnxruntime's `QLinearMatMul` on random layers with per-channel weight scales. Other runtimes may order the
  scale product differently; the 1-LSB bound does not depend on that, the exact tie set does.
* Convolution: 2-D only, `auto_pad = NOTSET`, constant weights/scales/zero points, a scalar activation zero
  point. Whether a *specific* tie accumulator is reachable through a convolution's tap structure is not decided
  (reported as undecided). With padding and no input shape the layer is `skipped`.
* No whole-model error propagation: this module is per-layer. For the float-vs-quantized *error* use
  `onnxsim.interval.quantization_bounds` and `onnxsim.zonotope`.
* `weight-fidelity` needs a float reference tensor of matching shape; `MatMulInteger` / `ConvInteger` carry
  no scale, so there is nothing to compare for them.
* Attention, softmax, and other quantized ops are not covered.

## Measured

On the machine used for development (timings include everything `verify_model` does):

| layer | time |
|---|---|
| dynamic `MatMul` 256 x 256, with weight fidelity | 0.009 s |
| dynamic `MatMul` 4096 x 4096, with weight fidelity | 1.25 s |
| `QLinearMatMul` 4096 x 1024, per-channel scale | 2.1 s |
| `QLinearConv` 64 -> 64, 3x3, 56x56, padded, int32 bias | 0.18 s |
| `MatMulInteger` at the int32 boundary (K = 66311 / 66312) | < 1 ms |
