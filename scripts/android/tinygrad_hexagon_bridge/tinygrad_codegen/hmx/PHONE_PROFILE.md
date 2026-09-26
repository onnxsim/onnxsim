# ResNet-18 on the phone: the per-kernel breakdown, measured

All numbers from the **Xiaomi 12S (SM8475, taro, V69)**, 10 iterations, the whole graph as one
FastRPC call. 0/25088 off ORT. Total **22,238-22,282 us** depending on run.

The phone can now be asked for this directly: `client <uri> <case> 10 prof` prints
per-call us in `graph.h`'s G_PROF order (`dsp_graph.py` emits `G_PROF(i)` after every call; the
skel defines it with `HAP_perf_get_time_us`, the simulator with `hexagon_sim_read_pcycles`).

## Where the time goes

| shape | us | share | n |
|---|---:|---:|---:|
| `r_16_2_..._3_3_16` (layer4 3x3 convs) | 7708 | **34.7%** | 3 |
| `r_2_202_..._4_4` (the stem) | 4499 | **20.2%** | 1 |
| `E_13276_8_2_2` | 1636 | 7.4% | 1 |
| `r_2_51_..._3_3_2` | 1573 | 7.1% | 4 |
| `r_2_6464_32_3_3` | 1502 | 6.8% | 1 |
| `r_8_4_..._3_3_8` | 829 | 3.7% | 3 |
| `r_4_14_..._3_3_4` | 685 | 3.1% | 3 |
| everything else | ~1800 | 8% | |

The clock reads in 1 us, so calls under 4 us are indistinguishable from zero; the client omits
those and they are ~1% of the total.

## The simulator was wrong by up to 1.5x, and pointed the wrong way

| call | kernel | hexagon-sim | phone | ratio |
|---|---|---:|---:|---:|
| 57 | `r_2_202_..._4_4` (stem) | 26.4% | 20.2% | 0.77x |
| 138 | `r_16_..._3_3_16` | 7.9% | 11.7% | **1.47x** |
| 143 | `r_16_..._3_3_16` | 7.8% | 11.7% | **1.50x** |
| 132 | `r_16_..._3_3_16` | 7.8% | 11.3% | **1.46x** |
| 56 | `E_13276_8_2_2` | 8.5% | 7.4% | 0.86x |
| 62 | `r_2_6464_32_3_3` | 5.2% | 6.8% | 1.29x |

**The three layer4 3x3 convs are 34.7% on the phone and 23.4% on the simulator.** Optimising
the stem - which the simulator made look like the one thing worth doing, at 26.4% - addresses
20.2% of the real cost. That is why the two stem attempts produced less than the numbers
suggested, and it is the single most useful thing this measurement settled.

The simulator is still the right tool for *where a candidate change landed*, because it gives
the same breakdown in ~3 minutes without a device. It is not the right tool for deciding what
to work on next.

## What the layer4 convs are

`k81` (and its two siblings): 16 panels x 3x3 taps x 16 K-blocks, 5 HMX MAC sites, 3 `__hmx_rq4f`
groups, `buf0[2048]`. They take the **quad path** (`__hmx_i8_pack_a4x4` present, alongside
`__hmx_i8_copy_a` and `__hmx_i8_pack_a4`), so the activation packing is already the four-block
form the 24.2 -> 22.7 ms change introduced.

## Reproducing

```bash
export HEXAGON_SDK_ROOT=/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2
export HEXAGON_TOOLCHAIN=$HOME/.cache/hexagon-oa-19/Tools
export TINYGRAD=<onnxsim/tinygrad checkout>
cd scripts/android/tinygrad_hexagon_bridge/tinygrad_codegen/hmx
HMX=1 DEV=DSP MOCKDSP=1 TC=1 TC_OPT=1 HVX_ARCH=v69 CC=clang-19 HMX_VTCM_KB=4096 PYTHONPATH=$TINYGRAD \
  python qdq_net.py backbone_qdq.onnx <outdir> --skel
# push <outdir>/{tg_hmx_rpc.so,client,blob.bin} + input.bin as a.bin + ref.bin, then
# client <uri> <case dir> 10 [prof]
```

`HMX_VTCM_KB=4096` matters: at the 256 KB default the same build measured **25,557 us**, 14.6%
slower, and the client reports `vtcm 262144` instead of `vtcm 4194304`. The phone grants 4 MB.

Always check `codes power 0 ... hvx 0 hmx 0` in the client's first line. `hmx 0` is the
`HAP_power_set_HMX` vote succeeding; without it the first HMX tile op hangs the cDSP until the
phone reboots.
