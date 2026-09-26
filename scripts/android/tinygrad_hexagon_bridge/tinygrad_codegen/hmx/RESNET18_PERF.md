# ResNet-18 tinygrad DSP perf: measured breakdown and the next lever

Baseline reproduced 2026-09-26 from master `a72ed55d`, tinygrad fork `hvx-hmx-qdq` @ `3605e14ca`
(hexagon-sim `-mv69 --mhmx 1`, `HMX_VTCM_KB=4096`):

```
150 kernel calls {other: 122, hmx: 20, qadd: 8}, 155 buffers: constants 11.23 MB, scratch 6.10 MB
hexagon-sim: 0/25088 mismatches vs ORT's semantics; 23945103 pcycles/inference (PASS)
  vs ORT CPU's output: 0/25088 mismatches
```

23.9M pcycles matches the handoff's number, so this is the same baseline.

## Where the cycles go

| kind | pcycles | share | calls |
|---|---:|---:|---:|
| HMX conv (`:cm`) | 18,107,679 | 75.6% | 20 |
| other (copies/pads/moves) | 4,990,421 | 20.8% | 122 |
| QLinearAdd | 832,169 | 3.5% | 8 |

QLinearAdd is *not* the problem any more (the handoff's 56x56x64 number is per-call, not per-graph).
The convs are, and one conv dominates:

| pid | kernel | pcycles | share | name |
|---|---|---:|---:|---|
| 57 | k25 | 6,320,833 | **26.4%** | `r_2_202_..._4_4` (the stem) |
| 138 | k81 | 1,897,231 | 7.9% | `r_16_..._3_3_16` (layer4) |
| 143 | k82 | 1,863,513 | 7.8% | `r_16_..._3_3_16` (layer4) |
| 132 | k75 | 1,858,018 | 7.8% | `r_16_..._3_3_16` (layer4) |
| 87 | k46 | 656,220 | 2.7% | `r_4_26_..._3_3_2` |

The three layer4 convs together are another 23.4%, so stem + layer4 = ~50%.

## The stem: 3.6x wasted MACs, all of it on the K axis

`k25` is 202 panels x 2 phases x a 4x4 reduce nest = 6,464 HMX `:cm` MAC ops
(each 32x64x32 = 65,536 MACs) = **423.6M MACs** for a conv whose real math is
112*112*64*3*7*7 = **118.0M MACs**.

It runs at 67 MAC/cycle. So the stem is not slow per MAC - it does 3.6x too many of them.

**The output grid is not the problem.** `P64 = r64(112*112) = 12544`, exactly the valid
112x112, ratio 1.000. (The README's "output grid 112 x 230, half the pixels garbage"
described the *pre-phase-split* stem; the phase split removed that.) The downsample convs
still carry it - 28x28 with `Wp=56` is `P64=1600` over 784 valid = 2.04x - but that is
2 of 20 convs.

All 3.6x is on K, and it decomposes cleanly:

- The 2x2 phase split turns the 7x7 conv into a **4x4** window over the 112x112 phase
  grid (`sk = 7//2 + 1 = 4`). Useful K = 7*7*3 = **147** values; the kernel uses 16 blocks
  x 32 lanes = **512**. 512/147 = 3.48x, which is the 3.59x measured (the small remainder
  is the 202-panel round-up to 64).
- **The 3 real channels are widened to 8** (line 220: `.pad(..., (0, 8 - xin.c))`), so each
  32-lane block is 4 phases x 8 channels and only lanes 0..2 of each 8-group carry data:
  12 of 32 lanes useful = 37.5%. That is 8/3 = **2.67x** on its own.
- The tap count 16 vs 49/4 = 12.25 effective is the other **1.31x**.

2.67 x 1.35 = 3.59x.

## The stem lever, worked out

The 2x2 phase split is **forced by the geometry**, not a free parameter: a phase split `f`
divides the *output* grid, and only `f=2` yields the required 112x112 (f=1 gives 218, f=4
gives 28). The 7x7/s2 conv becomes a 4x4 window (`sk=4`) over the 112x112 phase grid, with
the ring padding absorbing the 109 -> 112 difference.

So a 32-lane K block always holds 4 phase-pixels, and with 3 real channels that is 12
useful lanes. The three ways out, and why two fail:

1. **Fewer phases per block (f=1)**: 7x7 = 49 blocks, 3/32 lane use = 10.7x waste. Worse.
2. **A K block narrower than 32**: the `:cm` tile op is fixed at 32x64x32. Not available.
3. **Pack other taps into the 20 spare lanes** - the only real option.

Option 3 in detail. A K block would hold 10 phase-pixels x 3 channels = 30 lanes + 2 pad:

- The stem needs 16 taps x 4 phases = **64** phase-pixels.
- At 10 per 32-lane block that is `ceil(64/10) = 7` K blocks, against 16 today:
  **2.29x fewer MACs** (288 lanes of MAC work per 147 useful, 1.96x, versus 3.48x).
- Stem MACs 423.6M -> ~185M; at the 67 MAC/cycle it already achieves, 6.32M pcycles ->
  **~2.8M**, saving ~3.5M of 23.9M (15% of the whole graph).

The cost: an A tile row is currently 32 lanes = 4 phase-pixels of **one** grid row, copied
flat by `__hmx_i8_copy_a` (the contiguous 2 KB copy the README credits for 24.2 -> 22.7 ms).
Under option 3 the 10 phase-pixels of a lane group span ~3 grid rows, so the A pack becomes
a strided gather. The stem alone packs 202 panels x 2 phases x 16 blocks x 2 KB = 12.9 MB
of A; the README puts whole-graph activation packing at ~150k pcycles, so even a 3x gather
cost is well covered by the 3.5M saved. **Net: clearly positive.**

> **This estimate turned out to be wrong - see "Tried" below.** Packing 64 phase-pixels x 3
> channels needs exactly 192 lanes = 6 blocks of 32 (100% lane use, no padding, 2.67x fewer
> MACs), but the gather costs 46.4M pcycles, not the ~150k this section assumed, and the
> result is 2.9x slower. The reasoning error is identified in the "Tried" section; the
> analysis of where the 3.59x goes is unaffected.

## Tried: packing other taps into the spare K lanes (does not pay)

The plan above was implemented on a `stem-kpack` branch of the fork
(`tinygrad/nn/onnx_qdq.py`, `QDQ_STEM_KPACK=1`, the 2x2 phase split's 4 phases x 8 padded
channels replaced by 64 phase-pixels x 3 real channels = `kp = 4*sk*sk` phase-pixels,
`kb = kp*xin.c/32 = 6` K blocks of 32 instead of 16 blocks):

- **Numerically exact.** 0/25088 off ORT, and identical to the old layout in a standalone
  numpy check over a 16x16x3 stem (0 mismatches of 1024).
- **2.67x fewer MACs, as predicted** - but **2.9x SLOWER overall**: 68,918,887 pcycles
  against the baseline's 23,945,103.

| kernel | pcycles | share | note |
|---|---:|---:|---|
| `E_12928_32_6` (new) | 46,377,606 | **67.3%** | the gather, materialized as its own kernel |
| `r_2_202_..._6` (stem) | 6,707,880 | 9.7% | was 6,320,833 with 16 K blocks |
| everything else | ~15.8M | | unchanged |

**Why.** The K lanes of an A tile row were 4 phase-pixels of *one* grid row, so the whole
2 KB tile was one contiguous `__hmx_i8_copy_a`. With the lanes spanning the window, one row
gathers from ~3 grid rows, which tinygrad materialises as a separate copy
(`E_12928_32_6`, 12928 x 32 x 6 bytes = 2.5 MB) - and the stem conv itself only went from
6.32M to 6.71M, i.e. the 2.67x MAC saving was entirely eaten by losing the contiguous copy.

The estimate in "The stem lever, worked out" was wrong: it priced the strided gather against
a whole-graph ~150k pcycles for activation packing, but the stem's A operand alone is
202 panels x 2 phases x 16 blocks x 2 KB = 12.9 MB, and gathering it costs ~3.6x what the
flat copy did. Saving 4.3M pcycles of MACs and paying 46.4M for the gather is a large net
loss.

**What this says about the lever.** Removing the stem's wasted MACs needs the *pack* to stay
flat, not just the MAC count to drop. That means the wasted lanes have to be filled from
data already contiguous in memory - the hand kernel's crouton layout, where a 3x3 conv is
`:single` windows over shifted copies of an already-packed 2 KB activation. That is a
graph-level layout change (the README's own next step), not a re-lane-packing of the same
grid. The branch was dropped rather than kept behind a flag.

## The three layer4 convs (23.4%)

`r_16_2_..._3_3_16` - 16 panels, 3x3, 16 K blocks. At ~1.86M pcycles each these are the
next target after the stem, and unlike the stem they are not doing wasted MACs; they are
paying the same fixed overheads (requant ~1.2 cyc/output vs the hand kernel's 0.84,
activation packing, byte-plane reassembly).

## Reproduction

```
cd <onnxsim>/scripts/android/tinygrad_hexagon_bridge/tinygrad_codegen/hmx
HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC=clang-19 \
  HEXAGON_TOOLS=~/.cache/hexagon-oa-19/Tools HMX_VTCM_KB=4096 \
  PYTHONPATH=<onnxsim/tinygrad checkout of hvx-hmx-qdq> \
  python qdq_net.py <backbone_qdq.onnx> <outdir> --sim input.bin ref.bin
```

`<outdir>/sim_profile.txt` is `graph <pcycles>` then one `call <pid> pcycles <n>` per call;
`graph.h` maps each `kNN(...); G_PROF(pid);` line back to its kernel file and shape comment.
