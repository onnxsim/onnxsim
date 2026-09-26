# Where the ResNet-18 DSP time goes, measured on the phone

Companion to PHONE_PROFILE.md (the breakdown) and REQUANT.md (one candidate). All numbers from
the Xiaomi 12S (SM8475, taro, V69), whole graph as one FastRPC call, 10 iters, `HMX_VTCM_KB=4096`.

## Baseline

```
22207.7 us/inference (10 iters) PASS     0/25088 mismatches
```

## What each mechanism is worth, by rebuilding and re-measuring

Each row is a full rebuild from source and a fresh phone run, not a model:

| build | us | vs baseline |
|---|---:|---:|
| **baseline** | **22,208** | - |
| `HMX_I8_CACHE=0` (no VTCM tile cache) | 29,878 | **+7,670 (+34.5%)** |
| `HMX_I8_QUAD=0` (no 4-block weight tile / quad path) | 23,739 | +1,532 (+6.9%) |

So the VTCM tile cache is the single largest mechanism already in the build - it is carrying
**6.9 ms**, more than any individual kernel. The quad path is worth 1.5 ms.

Both are on by default, so neither is available as a win; the numbers are here to show that
the remaining cost is *not* dominated by something already being paid for, and to size what a
new mechanism would have to beat.

## Where the target is

From PHONE_PROFILE.md, the three layer4 3x3 convs (`r_16_..._3_3_16`) are **34.7%** of the
phone's time and the stem is 20.2%. `k81`'s shape:

```
16 panels (Lidx4) x 3x3 taps (Ridx0,Ridx1) x 16 K-blocks (Ridx2)
5 HMX MAC sites, 3 __hmx_rq4f groups, buf0[2048]
A: __hmx_i8_copy_a (the flat 2 KB) + __hmx_i8_pack_a4 + __hmx_i8_pack_a4x4
B: 8 x __hmx_i8_pack_b4 per MAC site
```

The weight offset `alu18` is `((Ridx2<<14) + alu0 + (Ridx1<<18) + (Ridx0*786432))` - it depends
on the tap and the K block but **not on the panel index**. So all 16 panels want the same
weight tiles, and they are kept in VTCM by the tile cache (the 6.9 ms above is exactly that).
The A operand is the flat 2 KB copy, which is the invariant the stem work established must not
be broken.

## The measurement trap, recorded so it is not repeated

`HMX_VTCM_KB` defaults to 256 KB. The phone grants **4 MB**. A build made with the default
measures **25,558 us** - 14.6% slower - and the client's first line says
`vtcm 262144` instead of `vtcm 4194304`. That is not a regression, it is the pool being a
sixteenth of the size the tiles were planned for, and it is visible in one field of the
client's output.

Always read that line: `codes power 0 ... hvx 0 hmx 0` and the `vtcm` value, before believing
any timing.
