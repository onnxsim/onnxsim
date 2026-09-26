# The layer4 3x3 convs: 34.7% of the phone's time, and why

Xiaomi 12S (SM8475, V69), whole graph as one FastRPC call, 10 iters, `HMX_VTCM_KB=4096`,
0/25088 off ORT, baseline **22,208 us**.

| shape | us | share | n |
|---|---:|---:|---:|
| `r_16_..._3_3_16` (layer4 3x3) | 7708 | **34.7%** | 3 |
| `r_2_202_..._4_4` (the stem) | 4499 | 20.2% | 1 |
| `E_13276_8_2_2` | 1636 | 7.4% | 1 |
| `r_2_51_..._3_3_2` | 1573 | 7.1% | 4 |
| `r_2_6464_32_3_3` | 1502 | 6.8% | 1 |

## First: there are no wasted MACs here

Unlike the stem (3.59x), these are exact. Issued HMX `:cm` ops = 16 panels x 9 taps x 16 K = 2304,
each 32x64x32 = 65,536 MACs, so 150,994,944 against a real conv of 8x8x512x512x9 = the same
150,994,944. **1.00x.** And per MAC they are *better* than the stem (823 vs 978 pcy/MAC on the
simulator), so the obvious "it is slow per operation" reading is wrong too.

The 1.6x higher cost per HMX op (1.115 vs 0.696 us/op on the phone) is all per-op overhead.

## What the tile planner actually chose

`HMX_DEBUG=1` on the current fork:

```
3 x  i8 plan: rank 4 (quad True, quad A False) A 288 B 144 slots of 1920, kt 36
3 x  i8 plan: rank 4 (quad True, quad A False) A 288 B 144 slots of 1920, kt 72
```

Rank 4 is the **quad weight path** (four adjacent N tiles share a 128-byte weight row line, packed
four at a time with `__hmx_i8_pack_b4x4`, one `:deep` load pair for tiles n and n+1). That is on.

**`quad A` is False**, and that is the lever. The activation pack is the expensive half:

```
per HMX op in k81:  1 x __hmx_i8_copy_a   (a flat 2 KB copy)
                     8 x __hmx_i8_pack_b4  (each one 128 B of weight, 2 vshuff + stores)
```

`__hmx_i8_pack_b4x4` collapses the 8 weight packs into one 4-at-a-time form, so the *weight*
side is already optimised. The *activation* side is not: each op still does a 2 KB `copy_a`.

## Why quad-A is off, and what fixing it is worth

`_hmx_quad_k_rows` (ops_dsp.py:1864) enables the four-block activation pack when, for every A row
window, the window **advances exactly 32 bytes per K block** and is **128-byte aligned at
`k % 4 == 0`**. That holds when the A rows are contiguous NHWC channel blocks - the 1x1 convs and
the stem, where one K block is one 32-channel group of the same pixels.

It does not hold for a 3x3: the A rows are a *window* over the grid, at
`dy*Wp + dx + (K block)*C_stride`, which is a strided gather rather than 32-byte steps, so the
check returns False and the quad-A path is skipped.

The opportunity is the same one the README already identified for the 3x3 family: the hand
kernel's crouton layout makes the activation tile *already* be the 32x64 A crouton, so a 3x3 is
`:single` windows over shifted copies and the four consecutive K blocks read the same 128-byte
lines. In tinygrad the activation is a padded NHWC grid, so those lines are not contiguous and the
check fails.

Measured size of the prize, on the phone:

| build | us | note |
|---|---:|---|
| baseline | 22,208 | |
| `HMX_I8_QUAD_A=0` | 22,707 | the quad A path is worth only 0.5 ms **overall** |

So quad-A on the 3x3 family would be worth at most ~0.5 ms across the whole graph, not the 7.7 ms
the tile cache delivers elsewhere. The honest conclusion is that **the layer4 convs are not
obviously broken** - they have no wasted MACs, they are better per MAC than the stem, and their
weight path is already on the quad optimisation. What is left is the per-op activation copy, and
the 0.5 ms measurement says that is a small prize.

## The remaining 3.4 ms needs explaining

If it is not wasted MACs, not per-MAC inefficiency, and not the activation pack, it is
per-op fixed cost spread over 2304 ops per conv: the `:cm` issue, the `addq` of the four byte
planes back into the accumulator, and the requant of the finished tile. The README's own bisection
of this family says the same: requant ~330k of 682k pcycles, activation packing ~150k, MACs ~20k.

Which means the requant - the thing the stem work correctly identified as 48% of a 3x3 family -
is very likely the biggest single term here too, and it is the one with a real accuracy cost
(`QC_FAST`). See REQUANT.md and STATUS.md.

## Reproducing

```bash
HMX_DEBUG=1 HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 HMX_VTCM_KB=4096 \
  PYTHONPATH=<tinygrad> python qdq_net.py backbone_qdq.onnx <out> --sim input.bin ref.bin
```

prints one `i8 plan:` line per HMX kernel - the rank, whether the quad weight and quad A paths are
on, and the slot counts. That is the fastest way to see which optimisation a given conv got.
