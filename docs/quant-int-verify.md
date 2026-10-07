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

## Kernels that saturate int16 pair sums (`semantics="u8s8_pair_saturating"`)

Some AVX2 `u8 x s8` kernels (`vpmaddubsw`, on CPUs without VNNI) add *adjacent products* in saturating int16
before widening to int32. That is not two's-complement wraparound, and it gives different accumulators, so
exact-arithmetic verdicts do not describe such a machine. This was found the hard way: a CI runner returned
`1086422270` for an extreme `MatMulInteger` where exact arithmetic gives `2147481735`. `verify_layer`,
`verify_model` and `replay_no_wrap` therefore take `semantics=`, and `host_semantics()` says which one
onnxruntime follows on the machine you are on.

**The model**, for a uint8 activation row `a` (raw codes, the zero point is *not* subtracted first) and an
int8 weight column `w` of K taps:

1. taps are paired along K as (0,1), (2,3), ...; an odd K pairs the last tap with an implicit 0;
2. each pair contributes `sat16(a[2j]*w[2j] + a[2j+1]*w[2j+1])`, `sat16` clamping to [-32768, 32767];
3. pair values are summed in int32 (wrapping);
4. the zero point is an exact correction: `acc = sum - za * sum(w)`.

`QLinearConv` pairs taps in `(kh, kw, channel)` order and pads with the raw zero point.

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import quant_int_verify as Q

for k in (66311, 66312):
    a = np.full((1, k), 255, np.uint8)
    w = np.full((k, 1), 127, np.int8)
    print(k, "exact", k * 255 * 127, "pair-saturating", int(Q.u8s8_pair_saturating_matmul(a, w)[0, 0]))

m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 13]> m (uint8[1,66312] a) => (int32[1,1] y) { y = MatMulInteger(a, W) }'
)
m.graph.initializer.append(numpy_helper.from_array(np.full((66312, 1), 127, np.int8), "W"))
node, consts = m.graph.node[0], Q._consts(m)
for sem in (Q.EXACT, Q.U8S8_PAIR_SATURATING):
    rep = Q.verify_layer(node, consts, semantics=sem)
    print(f"{sem:21s}", rep.of("no-wrap")[0].verdict, [f.verdict for f in rep.of("pair-saturation")])
```
```text
66311 exact 2147481735 pair-saturating 1086422270
66312 exact 2147514120 pair-saturating 1086422652
exact                 refuted []
u8s8_pair_saturating  proved ['refuted']
```

The same layer wraps under exact arithmetic and does not under the saturating kernel (33156 saturated pairs add
up to far less than int32), and the new informational `pair-saturation` finding says the layer's *result* is not
the exact one on such a CPU. Under this semantics `no-wrap` is decided from the saturating accumulator's exact
range (`saturating_accumulator_range`: pairs read disjoint activations and `sat16` is monotone, so each pair's
extreme is attained independently), `requant-1lsb` is unchanged (it bounds the fp32 multiply, whatever the
accumulator), and `requant-bit-exact` reachability is left undecided rather than searched with exact dot
products.

### What is and is not validated

This machine's kernel is exact, so the model could not be compared with a saturating onnxruntime *here*. It is
pinned instead by values recorded from the saturating CI runner (the `tests/test_quant_int_verify.py` tests run
on every host and reproduce them):

| Recorded on the saturating runner | Model |
|---|---|
| `MatMulInteger`, K=66311 / 66312, a=255, w=127: 1086422270 / 1086422652 | identical |
| `QLinearMatMul`, K=96, za=110, per-channel scales: 1222 / 1221 / 880 of 1600 outputs differ from exact (seeds 0 / 1 / 2) | identical counts |
| `QLinearConv`, no padding, cin=2: one output 161 (exact pipeline: 164) | 161; tap orders (c,kh,kw) and (c,kw,kh) give 164 and 160, so they are excluded |
| `ConvInteger`: two exact-arithmetic tests *passed* on that runner | not modelled (`skipped`) |

and by an independent lane-by-lane emulation, and an exhaustive search for the accumulator range. On a
saturating host the oracle tests compare the model with onnxruntime bit for bit
(`validate_saturating_model_on_host()`); if they disagree, those tests are skipped with the disagreement as the
reason.

**Not validated, so not claimed:** padded or grouped convolutions, a non-zero weight zero point, strides and
dilations other than 1 (the recorded case uses 1), any CPU other than that one runner, and the order of the two
operands. On that last point: `sat16(a0*w0 + a1*w1)` is symmetric in the two operands' raw codes, so swapping
which tensor a runtime calls "first" cannot change a pair sum. What *does* matter is the dtype pair: `u8 x s8`
saturates; `s8 x s8` may reach the same kernel through an offset of the activation (changing the raw codes) and
`u8 x u8` does not use this instruction -- the model makes no claim about either and reports them `skipped`.
This was reasoned, not measured: it needs a saturating host to check.

## Opt-in checks after the quantizers

`onnxsim.quantize_dynamic(model, verify=True)` and `onnxsim.quantize_qoperator(model, verify=True)` run
`quant_int_verify.attach_verification` on the model they emit. It is **off by default** (the check can call Z3
for small layers). The return value is unchanged; a one-line summary goes to
`metadata_props["onnxsim.quant_int_verify"]`, a `RuntimeWarning` is emitted if a soundness finding is refuted, and
the check never raises (a failure is recorded in the summary). For the full report call `verify_model`.

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

import onnxsim

f = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 13]> m (float[2,16] x) => (float[2,4] y) { y = MatMul(x, W) }'
)
f.graph.initializer.append(
    numpy_helper.from_array(np.random.default_rng(0).standard_normal((16, 4)).astype(np.float32), "W")
)
q = onnxsim.quantize_dynamic(f, verify=True)
print(next(p.value for p in q.metadata_props if p.key == "onnxsim.quant_int_verify"))
print([p.key for p in onnxsim.quantize_dynamic(f).metadata_props])
```
```text
proved; layers=1; refuted=none; skipped_findings=0; u8s8_pair_saturation_hazard_layers=0
[]
```

`u8s8_pair_saturation_hazard_layers` counts layers whose result would differ on a saturating CPU. For
`quantize_dynamic`'s chain the pair-saturating model is not applied (its activation zero point is chosen at run
time), so that count is 0 there, which means "not assessed", not "none".

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
  saturation, fp32 multiply. A runtime that deviates is not described by the default `semantics="exact"`.
  `probe_u8s8_saturation()` / `host_semantics()` detect the main known deviation (AVX2 `u8 x s8` kernels that sum
  adjacent products in saturating int16), which `semantics="u8s8_pair_saturating"` models -- within the limits
  stated in the section above. On the machine these checks were developed on onnxruntime matched exact int32
  arithmetic; the saturating model is pinned by values recorded on a CI runner that does saturate.
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
