# Requantization: the biggest single term, and what is actually known

## The target

Per the HMX README's own bisection of the 3x3 QDQ conv family (682k pcycles on hexagon-sim):

| part | pcycles | share |
|---|---:|---:|
| **requantization** | **~330k** | **48%** |
| activation packing (after the four-block pack) | ~150k | 22% |
| byte-plane -> accumulator adds | ~55k | 8% |
| the HMX MACs themselves | ~20k | 3% |

So the arithmetic is the *small* term. This is where the time is, and it is why the
stem's 3.59x of wasted MACs turned out not to be the lever I first took it for.

`__hmx_rqwf` (`tinygrad/runtime/ops_dsp.py`) is ~31 vector ops per 32-lane group, and
`__hmx_rq4f` runs it over four rows, so ~131 vector ops per 128 outputs = **1.02 ops/output**,
against the README's measured 149 packets per 128 outputs = **1.16 cycles/output**. The hand
kernel's exact mode does the same work in **~0.84**.

Most of `rqwf`'s cost is *per-lane normalization*, not arithmetic:

```
L = min(vcl0w(u) - 1, 30)      // per-lane shift, from the accumulator's own magnitude
a = u << L
p = vmpyewuh_64(a, mm7) / vmpyowh_64_acc   // the 24x24 product
F = L + emf ; Fc = clamp(F, 1, 31)         // per-lane exponent
y = (r + (1 << Fc-1)) >> Fc                // per-lane round half up
d = |(r & mask) - half| ; win = ...        // the tie window, per lane
*flag |= (d <= win) & (F > 0)              // per lane
```

Every `|acc|` in a conv tile spans a wide range, so `L` and `F` genuinely vary per lane and
cannot be hoisted. That is the cost.

## What the hand kernel does instead, and the trade

`QC_FAST` (`test/external/dsp/hand/hmx/hmx_qconv.h`) moves all of it to the **host**:
`qc_pack_params` precomputes a per-column table, and the device does a single
`hmx_blk_store_u8cm` - one `:after:cm:sat.ub` store, with the scale, the bias, `round(zy/M)` and
the rounding-centring bias all folded into the table. No per-lane shift, no per-lane exponent,
no tie window: the hardware's own convert does it in one pass.

**Cost: accuracy.** The HMX keeps only ~4 fractional output bits and an 11-bit scale, so
roughly 5-8% of outputs land 1 off. The README puts QNN's own HTP output at **5.6-7.7% off vs
ORT on the same layers** - the same error class. So this is not a correctness bug, it is a
precision choice of the same kind QNN makes.

**That is the whole trade, and it is a product decision, not an engineering one.** It is the
single largest win available (48% of the 3x3 family's cost) and it takes exactness against
ORT away. I am not making that call unilaterally.

## Not yet established

- **I have not measured QC_FAST's real error rate.** I tried to emulate the `:cm.ub` convert in
  numpy and got 8.6% off with a max error of 186, which is far outside the expected class -
  so the emulation is wrong (the scale and the `+0.5` live in the hardware's table word, and
  `hmx_blk_set_table2`/`hmx_blk_store_u8cm` are not a formula I can reproduce in numpy). The
  5-8% figure is the hand kernel's own documentation, not something I verified.
- **I have not measured the speed.** It needs a skel rebuild, which needs the Hexagon SDK's
  `qaic` and FastRPC headers.
- The faster-than-expected `:after:cm:sat.ub` path is what makes QC_EXACT work at all (four
  `:retain` stores at 1, 2^-8, 2^-16, 2^-24 give the four bytes of the exact int32), so the
  device can already do single-store conversion; the missing piece for tinygrad is building the
  per-column table at load time, which the handoff already identified ("its table
  (`fp16(512 m)`, `b + lrint(zy / m)`) is per-layer constant data that `qc_pack_params` builds
  on the host ... It belongs with the ONNX loader").

## If this is taken forward

1. Get the SDK, build, and **measure QC_FAST's error on the real ResNet-18 output** - the same
   25088-output comparison the whole graph is checked with. Do not emulate it in numpy.
2. Report it as "N% of outputs 1 off vs ORT" alongside QNN's 35.6%-off-by-one-or-more figure
   for the same graph, so the two are comparable.
3. Keep `QDQ_STEM_KPACK`-style flags: the exact path is the current behaviour and the default
   until the accuracy trade is accepted explicitly.
