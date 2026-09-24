# MCC's decoder block through tinygrad's DSP backend (generalizing the hand kernel)

`../` hand-writes MCC's query decoder for the V69 DSP (HMX + 4 HVX threads, 2.6 ms per block at 1024 queries). This
directory runs the same block as **plain tinygrad Tensor code** through the onnxsim/tinygrad fork's DSP backend
(`hvx-hmx`: HVX codegen, qfloat, the HMX TensorCore) on the phone, checks it kernel by kernel, and fixes what that
turned up in the fork (onnxsim/tinygrad#7, branch `hvx-hmx-mcc`). The hand kernel is the target to match.

**Status: correct on the phone (cos 0.9999996 vs float64 at 1024 queries, the hand kernel's accuracy), 36.2 ms per
block** -- about 14x the hand kernel. It started at "doesn't compile / wrong / faults on the phone"; the table at the end has the steps and the
remaining gap.

## Pipeline

tinygrad's own DSP runtime can't reach this phone (raw FastRPC ioctls; see `../../tinygrad_hexagon_bridge`), and qemu
(MOCKDSP) can't run these kernels either: it can't execute HMX, and its Hexagon decoder aborts on the qfloat code the fork
now emits (`decode_packet: assertion failed`). So the kernels are captured and replayed:

| step | tool | what |
|---|---|---|
| capture | `mcc_tg.py check --no-run --save <bundle>` (`capture.py`) | runs tinygrad under MOCKDSP without executing (and without the mock compile: `CAPTURE_NO_COMPILE=1`, `PARALLEL=0`) and records every kernel call: source (from the PROGRAM uop -- lowering happens in worker processes), argument buffers, the pre-lowering AST |
| memory | `capture.save` | tinygrad's memory planner reuses one allocation for several logical buffers of different sizes (and buffers are views): the replay reproduces the layout as regions of merged address intervals, each region's image built from first-seen contents |
| hexagon-sim | `emit.py sim <bundle> [--ref ...] [--dump]` | every kernel rebuilt for real HMX (no `HMX_REF`), a driver replays the calls on `hexagon-sim -mv69 --mhmx 1` |
| per kernel | `verify.py <bundle>` | re-runs each call's AST on tinygrad's CPU backend from the replay's own inputs (`--dump`) and compares every argument buffer: the first call that differs is the bug |
| phone | `build.sh <bundle>` + `run.sh <bundle> [iters] [ref]` (`replay_rpc.idl`, `replay_impl.c`, `replay_client.c`) | a FastRPC skel with the kernels linked in: regions loaded once, VTCM + HMX acquired (VTCM aligned to 256 KB: the kernels' tile cache assumes one 256 KB window), calls timed one by one |

`tc_test.py` / `ew_test.py` do the same for one matmul / one element-wise op (the unit cases below).

## What running it found (fixed in onnxsim/tinygrad#7 unless noted)

1. **Fusion into matmul operands** (model code, `mcc_tg.py`): a LayerNorm / cast / GELU fused into a matmul's loads hides
   the plain row loads the HMX rewrite needs (it fell back to the generic tile op); GELU fused into fc1's epilogue was
   unrolled per output-tile element (1.5 MB of C, 10+ minutes of clang); S = q K^T fused with the self-score concatenate
   became an 86%-of-the-block scalar kernel. Each matmul operand is materialized (`.contiguous()`).
2. **HMX, single K tile** (fork): with head_dim 32, S = q K^T has K = one tile and no reduce loop; the rewrite took the
   enclosing output-tile loop for the reduction (accumulating across output tiles, a compile error), and its generic
   output path wrote 128-lane vectors over 32-element rows. Now: REDUCE-axis loops only; no reduce loop = begin before the
   op, store after it, row-pair output.
3. **HMX, A-panel prefetch** (fork): A's whole-panel L2 prefetch used B's (32-column) shape, 32*kt rows at A's row
   stride -- 8 MB past A for fc2 (K = 2048), a PD fault (rc 0x4e) on the phone once it reaches an unmapped page; hexagon-sim
   has no MMU and ran it fine. Isolated with `tc_test.py 1 1024 2048 512` (faults) vs `... 512 512` (fine).
4. **qfloat transcendentals** (fork): tinygrad's EXP2 decomposition needs vector int<->float conversion (v73+), so every
   exp / sigmoid / tanh kernel ran scalar on v69 (~190 cycles per element). EXP2 / RECIPROCAL now render as
   conversion-free HVX helpers (magic-number rounding, polynomial, exponent add; bit estimate + Newton), float division as
   a * reciprocal(b), and a lane column may be several vectors' consecutive lanes -- GELU 64x512 3.56M -> 0.19M Pcycles.
5. **Softmax row width** (model code): rows of 225 (224 seen + the self score) don't vectorize; the self term is kept
   apart as in the hand kernel.
6. **Replay pitfalls** (this directory): buffer aliasing (region replay), VTCM window alignment, qemu-free validation.
7. **Reductions across rows** (model code + fork heuristic): with queries as rows, every reduction (LayerNorm over
   features, softmax over tokens) upcast 128 *rows* and gathered one element per row. `mcc_tg.py --layout t` (default)
   runs the block feature-major, x^T (512 x Q), the queries on the vector lanes, so reductions are contiguous vector
   accumulations -- but tinygrad's upcast heuristic then still picked axes some buffer strides across (gathers) and, for
   a reduction nothing broadcasts into (the per-query self score q . k), unrolled the reduce and left it scalar. On the
   DSP the heuristic now sorts gathers first and upcasts the contiguous output axis before unrolling.
8. **Float max stayed scalar** (fork): a float MAX renders as a statement expression (NaN semantics), so the softmax row
   max never vectorized (100 ms). On qfloat targets a vector max is HVX's native vmax (`__builtin_elementwise_max`).
9. **HMX tile cache too small** (fork): 116 slots of 2 KB in one 256 KB window, so fc2 (K = 64 tiles) repacked both
   operands per output tile, and after the loop interchange qkv / fc1 repacked the activation panel (a 32-way reuse
   through an 8-way tag cache). `HMX_VTCM_KB=4096`: one pool in 4 MB of VTCM, K panels at a stride that keeps every
   load pair inside a 256 KB window; weights and activation panels are packed once per call. The capture records it
   (`vtcm_kb.txt`) and the replay skel acquires that much VTCM.
10. **P V on HVX** (model code): `o.reshape(512, Q).contiguous()` made the kernel's output 512 rows, tinygrad merged head
    and head_dim into that axis, P's index depended on it (row // 32) and the tensor core didn't apply. Materialized as
    (16, 32, Q) first, P V is an HMX matmul (13 -> 5.8 ms, and closer to float64).
11. **HMX epilogues** (fork): a bias / residual after an HMX matmul cost 2.5-5x the matmul itself -- a column-major
    accumulator for single-M-tile outputs (expanded axes now ordered by store stride), a deal clang scalarized through the
    stack (now one vdealh per register), clang forwarding those registers into the row loads and rebuilding every row with
    vinserts (asm stores), 64-byte row loads with no prefetch. fc1 + bias 10 -> 3.8 ms, fc2 + residual 5.8 -> 4.0.
12. **Softmax exp twice** (model code): e = exp(s - m) fused into both the row sum and the P = e / sum kernel. Now e is
    stored once (half), the sum reads it, and the 1 / sum moves into P V's epilogue (o = (V^T e) / sum): -1 kernel.
13. **fp32 transcendentals** (fork + model code): a half exp now renders as the hand kernel's hf exp2 (64 lanes per register
    instead of 32 through float), and a vector half division as a * reciprocal (clang scalarized it: GELU in half first
    ran 190 ms). The softmax exponent and GELU run in half, as in the hand kernel: exp 9.0 -> 4.8 ms, GELU 8.4 -> 5.8.
14. **Strided reductions starved for data** (fork): reductions down the rows (LayerNorm statistics, softmax max / sum in
    the feature-major layout) move 2 KB per step and prefetched one step ahead (~1.2 GB/s); they now prefetch 4 steps
    ahead (mean 0.83 -> 0.42 ms, softmax max / sum 2.8 / 2.9 -> 1.4 / 1.4). SQRT was decomposed into scalar code; it is a
    qfloat rsqrt-Newton helper now (LayerNorm variance 1.30 -> 0.40 ms).

## Phone (Xiaomi 12S, one decoder block, 1024 queries, `run.sh`)

| version | ms per block | vs float64 |
|---|---:|---|
| first valid run (after 1-3): S on HMX, element-wise scalar | 914 | cos 0.9999997 |
| + qfloat exp2 / reciprocal, softmax 224 + self | (GELU 299 -> 8.4 ms) | |
| + multi-vector lane columns (softmax normalize 368 -> 28 ms) | 572 | cos 0.9999997 |
| + feature-major block (`--layout t`) | 649 | |
| + DSP upcast heuristic: no gathers (7) | 258.5 | cos 0.9999986 |
| + vector float max (8) | 160.8 | |
| + contiguous upcast before the reduce unroll (7: self score 37.8 -> 0.7 ms) | 123.4 | |
| + 4 MB HMX tile pool (9) | 77.2 | cos 0.9999986 |
| + P V on HMX (10) | 70.4 | cos 0.9999997 |
| + exp once, normalize in P V's epilogue (12) | 64.0 | |
| + HMX epilogue fixes (11) | 48.8 | cos 0.9999997 |
| + hf exp2, half GELU (13) | 41.9 | cos 0.9999996 |
| + stride-aware prefetch, vector sqrt (14) | **36.2** | cos 0.9999996 |
| hand kernel (`../mcc_block.h`) | 2.6 | cos 0.9999996 |

Per kernel now (ms): HVX -- GELU 5.8, softmax exp 4.8, row max 1.4 + row sum 1.4, LayerNorm 2 x (0.4 + 0.4 statistics,
1.9 apply), self score 0.5; HMX (with epilogues) -- P V 4.5, fc2 4.0, fc1 3.8, qkv 2.9, proj 2.7, S 2.0 (the hand
kernel's HMX work is 1.14 ms in total, at 3.07 TMAC/s).

## The remaining gap, largest first

1. **Element-wise / reduction kernels through DDR** (exp + GELU 10.6, softmax statistics 2.8, LayerNorm 5.5 ms vs the hand
   kernel's ~1.1 ms on 4 threads): each is its own kernel streaming DDR on one thread; the hand kernel fuses them into
   the block on VTCM tiles.
2. **HMX kernels from DDR**: every matmul still packs its operands from row-major DDR into VTCM tiles once per call and
   unpacks the output through DDR (19.9 ms for 3.6 GMAC, bare matmuls ~2 ms each); the hand kernel keeps activations in
   VTCM between steps in the tile layout and streams prepacked weights.
3. **No fusion across ops**: 15 kernels, every intermediate through DDR (the hand kernel: one fused block).
4. **One thread**: the backend runs one HVX context; the hand kernel splits row blocks over 4 and overlaps HMX with HVX.

## Reproduce

```bash
TG=<onnxsim/tinygrad hvx-hmx-mcc checkout>
ENV="PARALLEL=0 CAPTURE_NO_COMPILE=1 HMX=1 HMX_VTCM_KB=4096 DEV=DSP MOCKDSP=1 TC=1 HVX_ARCH=v69 CC=clang-19 HEXAGON_TOOLCHAIN=<Hexagon tools>"
env $ENV PYTHONPATH=$TG:.:.. python mcc_tg.py check --data <../ref.py export --q 1024 dir> --q 1024 --blocks 1 --no-run --save $B
python -c "import numpy as np; np.ascontiguousarray(np.fromfile('<dir>/ref_out0.bin', np.float32).reshape(-1, 512)[:1024].T).tofile('$B/ref.bin')"
python emit.py sim $B --ref $B/ref.bin --q 512                    # hexagon-sim, real HMX (output is x^T: 512 rows)
python emit.py sim $B --dump && THREADS=0 PYTHONPATH=$TG python verify.py $B   # per-kernel check vs the CPU backend
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... NDK_CLANG=... ./build.sh $B
PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run ./run.sh $B 3 $B/ref.bin
```
