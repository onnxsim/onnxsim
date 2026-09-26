# MCC's query decoder, hand-written for the Hexagon DSP (HMX + HVX), Xiaomi 12S (V69)

The demo app's MCC 3D mode (`../maskrcnn_demo_app`, `../vision_models/mcc`) spends most of a reconstruction
in decoder chunks: 1024 query points against the image's cached seen K/V. Through QNN on the HTP a chunk
takes 63 ms (fp16) / 47 ms (w8a16, `dec_opt.py a16c`), and QNN's per-op profile shows why more GEMM
speed would not help: the GEMMs are a minority of the time; softmax (38%), LayerNorm, the element-wise
Muls / Concat / Transpose around the attention are most of it. This directory fuses the whole decoder
by hand instead: every GEMM on HMX, everything else on HVX, activations kept in VTCM in the HMX tile
layout from one step to the next.

**Result: a 1024-query chunk in 21.6 ms on the DSP (22.2 ms per FastRPC call), 2.1x QNN's w8a16 and
2.9x its fp16, at fp16-level accuracy** -- in the app, a reconstruction's decoder goes from 2.24 s to
1.06 s and "3D" from 2.5 s to 1.3 s, and the result is *closer* to the float64 model than the w8a16
graph it replaces (table at the end).

## Files

| file | what |
|---|---|
| `mcc_block.h` | one decoder block, SPMD over 1..4 threads (HMX on thread 0); scalar-C fallbacks per step |
| `mb_hvx.h` | the HVX steps on the tile layout: LayerNorm, base-2 softmax + self term, GELU, exp2 / reciprocal helpers |
| `mcc_decoder.h` | the whole decoder: positional embedding, 8 blocks, final LayerNorm, prediction layer, color head; `mb_pack_kv` |
| `mcc_hmx_rpc.idl`, `mcc_hmx_impl.c` | FastRPC skel: resident weights (`load_block`, `load_head`), `set_kv` / `set_kv_tiles` per image, `decode` per chunk, `run` (blocks only, benchmark) |
| `mcc_hmx_client.c`, `run.sh`, `build.sh` | phone benchmark / check client, push + run + msda health check, build (also `mcc_hmx_stub.o` for the app) |
| `ref.py` | packs the weights into HMX tiles (`weights`), exports test data + float64 references (`export`) |
| `sim/` | hexagon-sim programs: `block_sim.c` / `decoder_sim.c` (vs `ref.py`'s float64), `softmax_test.c`, `hvx_unit.c`, and the qfloat probes `qf_probe.c`, `cvt_probe.c`, `qf16_ops.c` |

## Design

- **head_dim 32 = one HMX tile.** q / k / v of head h are qkv output column blocks h / 16+h / 32+h; S = q K^T
  is one K tile against K^T (224 columns: 197 seen tokens padded, the padding biased to -65504 through
  the column table); P V is 7 K tiles; o_h is column block h of o, which is directly the proj GEMM's K
  tile h (HMX outputs and activations share the tile layout, so nothing is repacked between steps).
- **Residual adds on HMX:** one more K tile per output tile, the x tile times an identity weight tile,
  into the same exact fp16 accumulator -- x + y is rounded once and HVX never touches the residual stream.
- **Base-2 softmax:** K is packed pre-scaled by scale * log2(e) (and the self score likewise), so scores
  come out of HMX in log2 units and t = s - (max - 7) is one subtract; e = 2^t in hf (exp2 by the
  1536.0 magic-number round + a degree-4 polynomial + exponent add), sums and 1 / sum in qf32.
- **Threads:** thread 0 holds the HMX lock and issues every HMX op; thread 1 streams each weight column
  block ("job") into one of two VTCM buffers ahead of it (sequence counters + spin waits, no barrier per
  job); row-wise HVX work is handed out through an atomic counter per phase so thread 0 joins when its
  HMX work is done. Attention is pipelined by head (phase k: thread 0 finishes head k-1 and prepares
  head k+1 into the other half of a double buffer while everyone runs head k's softmax); the MLP runs in
  two row halves (fc1 of one half beside GELU of the other).
- **VTCM (7 MB acquired):** residual X, LayerNorm output H, a 4 MB scratch (attention: o, 2 x q/k/v, 2 x S;
  MLP: the fc1 / GELU output), two 128 KB weight buffers, tables, identity tile, 2 x p_self. Row blocks
  start at multiples of their span, so no HMX operand span crosses a 256 KB window (`../hmx_gemm` rule).

## V69 qfloat, measured (hexagon-sim)

- The widening hf x hf -> qf32 multiply splits a tile vector exactly into its two rows (lo = even lanes =
  row 2p, hi = odd = row 2p+1), and qf32 pair -> hf re-interleaves (`qf_probe.c`): per-row statistics
  are 32-lane reductions, a per-row scalar is one narrowing.
- The IEEE `vcvt` conversions are V73-only (the compiler refuses them for v69); qf32 -> hf rounds to
  nearest with ties away from zero (`cvt_probe.c`).
- **qf16 arithmetic is loose**: an exact product 0.0078125 x 128 comes back 1.00098, `2 - d r` two ulps
  low, biased the same way (`qf16_ops.c`), so chained qf16 compounds -- a Newton reciprocal stalled
  near 1e-3 and a 5-step polynomial reached ~1%. Every qf16 op goes back to hf; sums / reciprocals /
  the self score run in qf32. (The first "optimized" softmax chained qf16 and lost the row sums by 3%.)
- The hf reciprocal bit trick (0x7800 - bits) needs d < 2^15 and a normal result: 1 / (row sum) with a
  2^7 offset is out of range, so 128 / sum is computed as 1 / (sum / 128).

## Phone (Xiaomi 12S, turbo, under the phone lock, msda health check after every run)

One block at Q = 1024, per step of the work (`run.sh run 1024 1 3 <hvx> <threads>`):

| version | ms per block |
|---|---:|
| HMX GEMMs, element-wise in scalar C | 1185 |
| + HVX element-wise, 1 thread | 8.07 |
| + residual on HMX (identity tile), 4 threads | 3.11 |
| + weight-copy thread (double buffer) | 3.10 |
| + base-2 softmax (K pre-scaled), 4-wide exp, e parked in S | 2.92 |
| + attention pipelined by head, MLP in two row halves | **2.60** |

The HMX work adds up to 1.14 ms per block, the single-context MAC rate (3.07 TMAC/s) on 3.5 GMAC; the
rest is HVX (softmax ~0.66 ms, GELU ~0.25, LayerNorm ~0.21 on 4 threads) and synchronization.

Whole decoder (`run.sh run decode 1024 4`): **21.6 ms DSP, 22.2 ms per FastRPC call** (occupancy logit
max error 0.045 vs float64, no p > 0.3 decision flipped, color mean error 0.09/255).

In the app (quest2 headset, "3D" repeated on one mask, `app_check.py` on the `dump=1` tensors):

| decoder | chunks | decoder | "3D" total | recall / precision / chamfer / color L1 vs dense host fp32 |
|---|---|---|---|---|
| QNN fp16 (`dec_q1024.onnx`, refine 0.05) | 59 x 1024 | 3.7 s | 4.3 s | 0.993 / 0.992 / 0.0008 / 0.46 |
| QNN w8a16 (`dec_q1024.a16c.onnx`, refine 0.1) | 48 x 1024 | 2.24 s | 2.5 s | 0.990 / 0.988 / 0.0011 / 0.76 |
| **this (DSP, HMX + HVX)** | 48 (last of a level: multiple of 32) | **1.06 s** | **1.30 s** | **0.993 / 0.992 / 0.0008 / 0.43** |

## Reproduce

```bash
H=<Hexagon SDK 6.x>; export HEXAGON_SDK_ROOT=$H HEXAGON_TOOLCHAIN=$H/tools/HEXAGON_Tools/19.0.04/Tools
python ref.py export --out $W/q128 --q 128                  # + --q 1024 for the phone
../hmx_gemm/sim/run.sh sim/block_sim.c $W/q128 128 8 30     # 8 blocks, all HVX steps -> PASS
../hmx_gemm/sim/run.sh sim/decoder_sim.c $W/q128 128        # whole decoder -> PASS
OUT=$W/build NDK_CLANG=<ndk>/aarch64-linux-android29-clang ./build.sh
PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run sh -c "OUT=$W/build ./run.sh setup $W/q1024 && \
  ./run.sh run 1024 1 3 30 4 && ./run.sh run decode 1024 4 && ./run.sh health"
python ref.py weights --out $W/weights                      # the app: MCC_HMX=$W/weights ../maskrcnn_demo_app/deploy.sh
```

## Next

- The HVX work is now the larger part (softmax first); a second HMX context is refused in one unsigned PD
  (`../hmx_gemm`), so the MACs stay at one context's rate.
- `set_kv` packs K / V on the DSP in 66 ms (scalar); the app packs them on the CPU instead
  (`set_kv_tiles`).
- **Generalizing with tinygrad: `tg/`** -- the same block as plain tinygrad Tensor code through the onnxsim/tinygrad
  fork's DSP backend, captured and replayed on hexagon-sim (real HMX, per-kernel check against the CPU backend) and the
  phone. Running it fixed three fork bugs (single-K-tile HMX ops, the A-panel prefetch that faulted the phone, scalar
  transcendentals on v69, scalar float max, gather-heavy upcasts, a tile cache too small for the MLP, slow HMX epilogues:
  onnxsim/tinygrad#7);
  correct at 1024 queries, 36.2 ms per block vs this kernel's 2.6. `tg/README.md` has the remaining gap (every step its
  own kernel through DDR, no fusion, one thread).
