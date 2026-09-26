# HMX GEMM on Hexagon V69 (Xiaomi 12S / SM8475), from an unsigned FastRPC skel

Follow-up to `../hmx_probe/` (PRs #1890, #1905), which found the working HMX recipe for an unsigned PD:
HVX power + **`HAP_power_set_HMX` power_up** (without it the first tile op wedged the cDSP until a
reboot) -> `HAP_compute_res` (VTCM + hmx param) -> `HAP_compute_res_hmx_lock` on the issuing thread ->
tiles in VTCM -> `bias = mxmem` column table -> `{activation = mxmem(a,Rt):deep; weight = mxmem(w,Rt)}`
-> `mxmem(o,0):after... = acc`. `hexagon-sim -mv69 --mhmx 1` models HMX (standalone code sets SSR bit
26, which QuRT's hmx_lock does on the phone) and matched the phone on every sequence, so every new
sequence here was developed in the simulator first.

## Tile layouts (all derived on hexagon-sim with one-hot operands, 0 mismatches over every byte)

32x32 tiles. `IDX(i, j) = 64*(i/2) + 2*j + i%2` ("2x1 interleave": row pairs, one 128-byte vector per
row pair).

| | fp16 (`.hf`) | int8 (`activation.ub` x `weight.b`) |
|---|---|---|
| activation A(r, k) | halfword `IDX(r, k)` | byte `2*IDX(r, k) + 1` = `128*(r/2) + 4*k + 2*(r%2) + 1`: **odd bytes only**, even bytes are never read |
| weight W(k, c) | halfword `IDX(k, c)` | byte `128*(k/4) + 4*c + k%4` |
| output C(r, c) | halfword `IDX(r, c)` (`:after.hf`) | `:sat.uh acc:2x1`: u16 `IDX(r, c)`; `:sat.ub`: byte `2*IDX(r, c) + 1` (= the activation layout, so a u8 output tile is directly the next layer's activation tile) |
| bytes per K=32 block | A 2048, W 2048 | A 2048, W 1024 |
| K > 32 | consecutive blocks for both operands, one `:deep` load pair with `Rt = (K/32)*2048 - 1`, **at most 32 blocks (K <= 1024) per load pair** (K=1056 is wrong); longer K = consecutive load pairs, which keep accumulating until the store (exact to K=4096) | same `Rt` for both; the weight reads `Rt/2` bytes (exact to K=256; wrong at K=1024 -- max depth not bisected) |

Output column table (`bias = mxmem(T)`, 256 B, words 0..31 = columns 0..31, words 32..63 unused):

- fp16 out: word c high 16 bits = fp16 **bias** of column c (added exactly); low 16 bits: leave 0.
- int8 out: word c low 16 bits = fp16 **scale** `s`; `:sat.uh` = `min(65535, floor(acc * s / 2))`
  (0x4000 = x1), negative accumulators clamp to 0; `:sat.ub` = `min(255, floor(acc * s / 2 / 256))`.
  High bits unused. There is no signed int8 output and no int-accumulator -> fp16 output on v69
  (`:after.hf` after int8 loads writes only the table's bias), so signed int8 GEMMs need an offset
  K-block (activation 255 x per-column positive weights) folded into the output zero point.
- fp16 accumulation is exact (e.g. `1000 - 1000 + 30*2^-10` comes out exact) and rounded once on store.

Simulator programs (`sim/`, run with `./sim/run.sh sim/<prog>.c [args]`): `map_int8.c`, `map_fp16.c`
(one-hot layout maps), `scale_probe.c`, `scale_probe2.c`, `round_probe.c` (column table / conversion
modes), `gemm_sim.c` (the GEMM below vs a double reference).

Two rules that only showed up on the phone (hexagon-sim does not model the first):

- **An operand span must not cross a 256 KB VTCM boundary.** A load pair whose A or W span crosses
  one takes a user-PD page fault at the boundary (`Bad VA` = VTCM base + 0x40000 / 0x80000; the PD
  dies with rc 0x4e, the cDSP stays healthy). `hmx_valloc` places every span inside one 256 KB window.
- At most 32 K-tiles per load pair (above), also true in the simulator.

## GEMM (`hmx_gemm.h`, header-only)

`hmx_gemm_f16(A, Wp, bias, C, M, K, N, vtcm, vtcm_bytes)`: `C[M,N] = A[M,K] . W[K,N] + bias`, fp16
row-major A/C, W prepacked once with `hmx_pack_w_f16` into per-32-column blocks of K/32 tiles; K, N
multiples of 32, any M. A is packed into VTCM once, then per 32-column block the weight tiles are copied
into VTCM and every 32-row block is one `:deep` MAC over all of K plus one tile store.

Packing/unpacking is HVX: one `vshuff(row1, row0, -2)` of a row pair gives the row-pair vector of two
K blocks; output tiles unpack with `vdeal h` + `vmux`/`vror` into 64-column row segments. Weight tiles
stream DDR -> VTCM with HVX copies, l2fetch-prefetched two 16 KB chunks ahead.

hexagon-sim, vs a double reference (tolerance = fp16 rounding of the result): 32x32x32, 45x576x128 +
bias, 128x576x192 + bias, 32x1056x64, 45x1536x128 + bias, 128x4096x64 + bias -- 0 elements beyond
fp16 rounding.

### Phone (Xiaomi 12S, one HMX thread, turbo, `hmx_gemm_client`; every run PASS vs a double reference)

Pure MAC + tile store with A and W resident in VTCM (`mode 1`): **3.07 TMAC/s fp16** at 128x576x1536,
512^3 and 1024^3 (36.9 / 43.8 / 349 us) -- the single-thread HMX issue rate (QNN's HTP reaches ~16
TMAC/s on this phone with both HMX units and its own scheduling).

End to end from DDR (`mode 0`: pack A, stream every weight tile into VTCM, unpack C to DDR), per call,
with the phase split in core cycles (~1.5 GHz):

| M x K x N | us | TMAC/s | pack A | W copy | MAC + store | unpack C |
|---|---|---|---|---|---|---|
| 128 x 576 x 576 | 53.5 | 0.79 | 11.6 k | 29.1 k | 7.3 k | 32.0 k |
| 128 x 576 x 1536 | 161.6 | 0.70 | 24.3 k | 113.4 k | 19.3 k | 84.5 k |
| 128 x 1536 x 576 | 205.2 | 0.55 | 114.6 k | 118.9 k | 8.6 k | 65.0 k |
| 512 x 576 x 1536 | 376.3 | 1.20 | 58.8 k | 121.5 k | 48.0 k | 334.6 k |
| 1024 x 1024 x 1024 | 1407 | 0.76 | | | | |

The MACs are 5-12% of the time: fp16 weights stream at ~22 GB/s (1.77 MB in ~80 us) and C goes back
to DDR at ~7 GB/s. HMX pays off where operands stay in VTCM across calls (fused pipelines), not as a
standalone GEMM over DDR buffers. Run with `./run.sh setup; ./run.sh gemm <mode> <M> <K> <N> [iters]
[bias]; ./run.sh health` under the phone lock (`build.sh` needs `HEXAGON_SDK_ROOT` for qaic/headers and
`HEXAGON_TOOLCHAIN` for -mhmx).

**Two HMX threads:** a second `HAP_compute_res_acquire` with the HMX parameter, while the first is held
by the same unsigned PD, is refused (ctx 0; `mode 2` of `hmx_gemm_client`), so this setup drives one
HMX context (2.91 TMAC/s MAC loop at 256x512x512, `mode 3`). Whether QNN's ~16 TMAC/s comes from a
second unit or from its own scheduling is not visible from here.

## First use: SmolLM2-135M prompt processing (prefill) projections on HMX

`llm_prefill.py prep` runs SmolLM2-135M (fp32 torch) on a 128-token prompt, captures every layer's real
projection inputs with forward hooks (fp16), fuses q|k|v (576 -> 960) and gate|up (576 -> 3072), and
prepacks the fp16 weights; `hmx_gemm_llm_client` runs all 120 GEMMs (4 per layer x 30) on the phone in
one FastRPC session; `llm_prefill.py compare` checks them against torch (fp16 inputs, float64
reference). Phone, turbo, under the phone lock, health check clean:

| GEMM per layer | M x K x N | DSP time per call | x 30 layers |
|---|---|---|---|
| q\|k\|v | 128 x 576 x 960 | 114 us | 3.42 ms |
| o | 128 x 576 x 576 | 81 us | 2.42 ms |
| gate\|up | 128 x 576 x 3072 | 302 us | 9.07 ms |
| down | 128 x 1536 x 576 | 200 us | 5.99 ms |
| **all 120** | | | **20.9 ms** on the DSP (0.65 TMAC/s), 64-118 ms wall |

Every output matches torch to fp16 rounding (worst cos 1.0000000, worst max-relative error 4.6e-4).

**Versus the HTP:** QNN runs the *whole* fp16 prefill (these projections plus attention, RMSNorm,
RoPE, SwiGLU) in 24.5 ms (`../llm_tinygrad/`, #1886). The HMX projections alone already take 20.9 ms,
so a DSP-side prefill built on this GEMM would at best tie the HTP: the fp16 weights (212 MB per
prefill) stream into VTCM at ~22 GB/s and the outputs go back to DDR at ~7 GB/s, while the HMX MACs
are only ~12% of the time. The wall time is 3-6x the DSP time because FastRPC copies each call's
weights in and out; a real prefill would keep the weights resident in the DSP heap like the decode
loop (`../llm_tinygrad/hvx/decode_impl.c`, 101 tok/s). What would make it win: int8 weights (half the
bytes; needs the offset-K-block trick for signed int8 output on v69), keeping activations and outputs in
VTCM across a layer instead of DDR, and a second HMX context if the PD can get one.

Reproduce (host: `~/.cache/llm-venv` with torch + transformers; phone steps under the phone lock):

```
python llm_prefill.py prep --work $W                              # 120 GEMMs: $W/<layer>.<proj>.{a,wp,ref}
./build.sh && ./run.sh setup
adb push hmx_gemm_llm_client $D/; (cd $W && tar cf - manifest.txt *.a *.wp) | adb shell "mkdir -p $D/llm/out && cd $D/llm && tar xf -"
adb shell "cd $D && ADSP_LIBRARY_PATH=$D ./hmx_gemm_llm_client 'file:///hmx_gemm_rpc.so?hmx_gemm_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' llm 1"
adb pull $D/llm/out/. $W/phone/ && python llm_prefill.py compare --work $W --phone phone
```

`tests/test_hmx_gemm.py` runs `sim/gemm_sim.c` on hexagon-sim (32x64x64, 45x576x128 + bias,
45x1056x128 + bias) when `HEXAGON_TOOLS` points at a Hexagon toolchain.

## Handoff for code generation (tinygrad DSP backend: HMX as a TensorCore)

Everything a generated kernel needs, split so the generator emits only the kernel:

| file | what | emitted by a code generator? |
|---|---|---|
| `hmx_block.h` | the block primitive: `hmx_blk_set_table`, `hmx_blk_mac_f16` / `hmx_blk_mac_u8s8` (load K tiles, accumulating; splits K into legal load pairs), `hmx_blk_store_f16` / `_u16` / `_u8` (store one 32x32 tile with conversion, clears the accumulator); full layout + table + precondition notes in its header comment | yes -- this is the TensorCore "instruction" |
| `hmx_runtime.h` | `hmx_rt_power` (HVX + **HMX** power vote + turbo), `hmx_rt_acquire` (VTCM + HMX context), `hmx_rt_lock/unlock` (HVX + HMX lock on the issuing thread), `hmx_rt_release` | no -- the runtime (skel) does this once |
| `hmx_gemm.h` | reference GEMM built on the block: HVX pack/unpack, VTCM windows, weight streaming | as a reference to match |
| `sim/block_ref.c` | **bit-exact** reference test of the three store modes (random operands, any K) | the diff target |

Rules a generated kernel must follow (all measured; the first two only on the phone):

1. Vote `HAP_power_set_HMX` power_up before any HMX op, and issue HMX ops only on a thread holding the
   HVX lock and `HAP_compute_res_hmx_lock` (a QuRT worker thread with >= 64 KB stack, not the FastRPC
   thread). Without the power vote the first tile op wedged the cDSP until a phone reboot.
2. Operands in the HMX context's VTCM, tiles 2048-byte aligned, and **no operand span may cross a 256 KB
   VTCM boundary** (user-PD page fault; see `hmx_valloc`).
3. At most 32 K-tiles per fp16 load pair and 8 per int8 load pair; longer K = more load pairs before one
   store (they accumulate).
4. Layouts: `IDX(i, j) = 64*(i/2) + 2*j + i%2`. fp16 A(r,k) / W(k,c) / C(r,c) at halfword `IDX`;
   int8 A(r,k) at byte `2*IDX(r,k)+1` (odd bytes, 2048 B / K-block), W(k,c) at byte
   `128*(k/4)+4*c+k%4` (1024 B / K-block), C at u16 `IDX` (`_u16`) or byte `2*IDX+1` (`_u8`, which is the
   int8 activation layout -- chainable).
5. Column table (256 B, word c = column c): fp16 out = `rne_fp16(exact acc + bias_c)` with bias in the
   high half; int8 out = `floor(acc * s_c / 2)` (`_u16`) or `floor(acc * s_c / 512)` (`_u8`), `s_c` fp16
   in the low half, saturating, negative acc -> 0 (use the offset-K-block trick for signed products).

Harness: `./sim/run.sh sim/block_ref.c <ktiles>` builds with the login-free Hexagon_open_access 19 tools
(`HEXAGON_TOOLCHAIN`, default `~/.cache/hexagon-oa-19/Tools`; `NCSHIM` = a dir with libncurses.so.5)
and runs `hexagon-sim -mv69 --mhmx 1`; it prints per-mode mismatch counts and PASS (K = 32, 256, 384
all 0 mismatches). A generated kernel should replace the `hmx_blk_*` calls in it and keep PASS. Standalone
sim code must set SSR bit 26 (HMX enable) itself; `hmx_lock` does it on the phone. `tests/test_hmx_gemm.py`
runs `gemm_sim.c` and `block_ref.c` in CI when `HEXAGON_TOOLS` is set.

Targets measured on the phone (one HMX context, turbo):

| what | best |
|---|---|
| MAC + tile store, operands resident in VTCM (fp16, 128x576x1536 .. 1024^3) | **3.07 TMAC/s** |
| same, int8 (`hmx_probe` issue-rate loop) | ~3.7 TMAC/s |
| full GEMM from DDR, fp16 (512x576x1536) | 1.20 TMAC/s (weights ~22 GB/s in, C ~7 GB/s out) |
| SmolLM2-135M prefill projections, 120 GEMMs | 20.9 ms (HTP whole prefill: 24.5 ms) |
| QNN / HTP practical int8 ceiling on this phone | ~16 TMAC/s (14.2 per 1x1 conv layer, re-measured below) |
| int8 `:cm` chained 1x1 layers, weights streamed (`hmx_gemm_u8.h`, next section) | **17.0 TMAC/s** |

What the remaining gap to QNN looked like: see the next section -- the int8 gap was the instruction mode,
not the number of HMX units, and it is closed for chained 1x1 layers.

## Matching QNN's int8 throughput: the "cm" instruction mode (`hmx_gemm_u8.h`)

QNN's HTP reached ~16 TMAC/s int8 on this phone (`../htp_exploration/ceiling_findings.md`) against ~3-4
TMAC/s from the kernels above, and fp16 was never compared. This section measures both sides on the same
shapes, finds where the difference comes from, and closes it.

### 1. The QNN side, reproduced (`qnn_parity/gen_models.py`)

Chains of L identical layers (int8: uint8 activations / int8 weights QDQ, as the ceiling study; fp16: a plain
fp32 graph, which the QNN EP runs in fp16), run with `../htp_exploration/ceiling/run_ceiling.sh` (ORT 1.26 +
QNN EP 2.6.0 + QNN 2.50, strict HTP, EP-context cache, burst) and differenced L=6 minus L=2 by
`summarize_ceiling.py`, so the per-layer time excludes graph launch and I/O. A 1x1 conv over an HxW map is a
GEMM with M = H*W rows.

| layer | QNN per layer | QNN TMAC/s |
|---|---:|---:|
| int8 1x1 conv 1024->1024, 64x64 (M = 4096) | 302 us | 14.20 |
| int8 1x1 conv 1024->1024, 32x64 (M = 2048) | 155 us | 13.85 |
| int8 1x1 conv 512->512, 64x64 | 80 us | 13.42 |
| int8 1x1 conv 512->512, 32x64 | 65 us | 8.26 |
| int8 MatMul 4096 x 1024 x 1024 | 3081 us | 1.39 |
| fp16 1x1 conv 256 / 512 / 1024, 64x64 | 95 / 415 / 1533 us | 2.83 / 2.59 / 2.80 |
| fp16 MatMul 4096 x 1024 x 1024 | 1446 us | 2.97 |

**fp16 is already at parity**: QNN's whole fp16 layer runs at 2.6-3.0 TMAC/s, the same as this directory's
fp16 MAC loop (3.07 TMAC/s). Only int8 was behind. (int8 MatMul through QNN is 10x slower than the same GEMM
as a 1x1 conv, so convs are the fair reference.)

### 2. Where QNN's int8 speed comes from

QNN's V69 skel (`libQnnHtpV69Skel.so` from the `qnn-runtime` 2.50.0 AAR) keeps its symbols, so
`hexagon-llvm-objdump -d --mattr=+hmxv69` shows which HMX instructions it issues. Its int8 1x1 conv
(`hmx_convbbb1x1_stride1`) is a loop of `{ activation.ub = mxmem(A, Rt):cm; weight.b = mxmem(W, 0x3ff) }`,
one 2 KB activation crouton per instruction, then one `mxmem(C, Rt):after:cm:sat.ub = acc` per output tile.
The `:cm` form was never tried here. Mapped on hexagon-sim with one-hot operands (`sim/gemm_u8_sim.c` is the
bit-exact check):

- `:cm` reads the 2 KB crouton as **64 rows x 32 channels, one byte each** (`A(s, k)` at byte `32*s + k`),
  where the non-cm int8 form uses only the odd bytes of 2 KB for 32 rows. One instruction is 64 x 32 x 32 =
  65536 MACs instead of 32768, for the same bytes read.
- `:after:cm:sat.ub` writes 64 rows x 32 columns (`C(s, c)` at byte `32*s + c`) -- exactly the next layer's
  activation crouton, so layers chain with no repacking.
- `weight.b = mxmem(W, 0x7ff):deep` (2 KB = two 32-column weight blocks) fills both accumulators: 64 x 32 x 64
  per instruction, one activation read for twice the MACs. QNN's 1x1 kernel does not use it.
- Output conversion is `min(255, floor(max(acc, 0) * s_c / 512))` with `s_c` the fp16 in the low half of
  table word `c`. It is exact for power-of-two scales; with arbitrary fp16 scales about 5% of outputs come out
  1 LSB low (the multiply is not done at full precision).

The hypotheses from the handoff above, one at a time:

| hypothesis | result |
|---|---|
| QNN drives two HMX units | **No.** Its skel imports only `compute_resource_hmx_lock`/`_unlock` (no `lock2`/`lock3`, one context), and one context here already beats it (below). |
| a faster HMX clock | V69 has no separate HMX clock: `HAP_power_set_HMX_v2` with a turbo corner is refused (rc -3). The core clock matters: without the DCVS turbo vote the MAC loop runs at 9.66 TMAC/s instead of 17.2 (the DSP runs at ~0.79 vs 1.40 GHz). `HAP_power_get` is refused in the unsigned PD; clocks come from pcycles / us. |
| better HMX scheduling | Not the main cause. Per-store column-table reloads, the K loop and the stores cost nothing measurable (below: chained layers vs the bare MAC loop). |
| DMA instead of HVX copies | Not needed for this: an HVX copy on a second thread streams the next layer's weights (1 MB) while the HMX runs the current layer (below). QNN does ship a DMA manager. |
| **the instruction mode** | **Yes.** int8 `:cm` does 2x the MACs per instruction of the non-cm int8 form and needs a quarter of the VTCM bytes per MAC of fp16. |

### 3. Our side (`hmx_gemm_u8.h`)

`hmx_blk_mac_u8cm_deep` / `hmx_blk_store_u8cm` (in `hmx_block.h`), `hmx_pack_w_u8cm` (host weight packing),
`hmx_layer_u8cm` (one layer, crouton activations on both sides), `hmx_gemm_u8_prof` (row-major DDR GEMM with
HVX 4x4 32-byte transposes for packing/unpacking). Every configuration below is **bit-exact** against an
exact host reference (power-of-two scales), on hexagon-sim and on the phone (turbo, under the phone lock,
health check clean after each run). Clocks from pcycles / us: 1.40 GHz.

| what (phone, one HMX context) | per layer | TMAC/s |
|---|---:|---:|
| int8 `:cm` MAC loop, A and W resident, weight `:deep` (`gemm_u8` mode 2), 1024^3 .. 7168 x 1024 x 512 | | **17.2** (12250 MAC/pcycle) |
| same without weight `:deep` (QNN's 1x1 choice; mode 3) | | 14.15 |
| **chained layers, 1024->1024, M = 2048, next layer's weights streamed DDR -> VTCM on a second HVX thread** (`layers` flag 1) | **126 us** (QNN: 155 us) | **17.03** (QNN: 13.85) |
| same, 512->512, M = 2048 / 4096 | 31.8 / 63.0 us (QNN: 65 / 80 us) | 16.90 / 17.03 (QNN: 8.26 / 13.42) |
| same, 256->256, M = 2048 / 4096 | 8.3 / 16.1 us | 16.11 / 16.62 |
| same, weights copied just before each layer on the HMX thread (flag 0) | 172 us | 12.46 |
| same, no weight copies at all (compute-only reference, flag 2) | 124.7 us | 17.22 |
| row-major GEMM from DDR (pack A, stream W, unpack C on one thread; `gemm_u8` mode 0), 2048 x 1024 x 1024 | 450 us | 4.78 (MAC phase 25% of the time, pack/unpack 62%; fp16 equivalent: 1.20) |
| earlier int8 (non-cm) MAC loop, `../hmx_probe` | | ~3.7 |
| fp16 MAC loop (above) | | 3.07 (QNN fp16 layer: 2.6-3.0) |

So a chain of int8 1x1 layers now runs **~1.2x faster than QNN's on the same shapes** (2x for 512->512 at
M = 2048, where QNN's per-layer overhead shows), at 99% of the bare MAC-loop rate, from one HMX context in
an unsigned PD. M = 4096 at 1024 channels does not fit two activation buffers + two weight buffers in the
8 MB VTCM (7.75 MB is the most one acquire got), so that row is only measured as the bare MAC loop (17.2).

`hexagon-sim --timing` (`sim/rate_sim.c`) predicted the ranking before any phone run: ~12000 MAC/pcycle for
`:cm` + weight `:deep` (phone: 12250), ~8000 without `:deep` (phone: 10074), ~2500 for fp16 (phone: ~2200).

### What this is not yet

- **Not a drop-in QNN conv.** The output is `sat_u8(floor(max(acc, 0) * s / 512))`: no per-column bias, no
  activation zero point (QNN's QDQ models use uint8 zero point 128, which it folds into a bias), and
  arbitrary fp16 scales round 1 LSB low about 5% of the time. The table's other fields (V81's HMX manual
  documents input/output bias fields in the 64-bit bias registers, loaded by `bias = mxmem2`) are the next
  thing to map for int8.
- Only 1x1 convs / GEMMs. QNN's 3x3 kernels use `:dilate` weights and `:above` activations for spatial
  taps; those forms are unexplored here.
- The row-major DDR GEMM is bound by its single-thread HVX pack/unpack (62% of its time); a pipeline that keeps
  activations in crouton form between ops (as `layers` does) is the fast path.

Reproduce (host builds; phone steps under `PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run`):

```
./build.sh && D=/data/local/tmp/<dir> ./run.sh setup
./run.sh gemm_u8 2 2048 1024 1024 5              # bare MAC loop (mode 3: without weight :deep)
./run.sh layers 1 2048 1024 6 5                  # chained layers, weights streamed on a second thread
./run.sh health
python qnn_parity/gen_models.py $M && REMOTE_DIR=/data/local/tmp/<dir>/qnn \
  ../htp_exploration/ceiling/run_ceiling.sh $M 25 burst && ../htp_exploration/ceiling/summarize_ceiling.py $M burst
```

## QDQ-exact 1x1 convolution (`hmx_qconv.h`)

The `:cm` GEMM above as a drop-in for a QDQ `Conv` in the form onnxsim's `full_qdq` / QNN use: uint8
activations with a zero point (128), per-channel symmetric int8 weights, int32 bias, uint8 output with its own
scale and zero point, optional fused Relu. The reference is ORT CPU itself (it fuses `DQ -> Conv -> Q` into
QLinearConv): `y = clamp(rne(fp32(fp32(acc) * M[c])) + zy)`, `M[c] = fp32(fp32(sx * sw[c]) / sy)`, where
`acc = sum (xq - zx) * wq + bq`. `qnn_parity/qdq_ref.py` is that formula in numpy and matches ORT's output
exactly (0 mismatches over 1.4 M outputs of four layers).

### What the HMX conversion can and cannot do (hexagon-sim, `bias = mxmem2`)

The 64-bit column table (`bias = mxmem2`, 256 B, **256-byte aligned** -- a 128-aligned table loads garbage)
in int8 mode:

- **high word: an int32 added exactly to the accumulator.** The activation zero point therefore folds into
  it (`bq - zx * sum_k wq`), and the HMX consumes the raw uint8 activations.
- low word bits 15:0: the fp16 scale `s`; **bit 22: +0.5 before the floor** (round half up). No other low-word
  bits had an effect.
- conversion: `floor(trunc(acc + B) * s / 512 [+ 0.5])`, where **the accumulator is truncated to a multiple of
  `2^(5 - E)`** (`E` = the exponent of `s`): only ~4 fractional output bits survive, and the scale has 11 bits.
  So a QDQ conv cannot be bit-exact through this path.
- **`mxmem(C, 0):after:cm.ub = acc` (no `:sat`) wraps**, and `:retain` keeps the accumulator: four stores at
  scales 1, 2^-8, 2^-16, 2^-24 (`0x6000 0x4000 0x2000 0x0800`) give the four bytes of the exact int32 accumulator,
  two's complement (0 mismatches, |acc| up to 1.3 M, both halves of a weight `:deep` op). The 16-bit stores
  (`:uh acc:2x1 / 2x2`) are no use after `:cm`: they expose only the odd rows.
- An HMX tile store drops the low 11 address bits (2 KB alignment) -- a misaligned scratch tile silently
  overwrote its neighbour.

### Two requantization modes

| mode | how | vs ORT |
|---|---|---|
| `QC_FAST` | one `:cm:sat.ub` store; table = fp16(512 M), round bit, int32 `B = bq - zx sum w + round(zy / M) + 2^(4-E)` (the last term centres the truncation) | off by one on ~1-3% of outputs |
| `QC_EXACT` | 4 byte-plane stores -> exact acc; HVX integer requant `r = round(acc 2^L bm / 2^31) ~ v 2^F` (per-column `L`, `F`, `bm` so nothing overflows and `F` <= 21), `y = round-half-up(r / 2^F) + zy`; outputs whose `r` is within a per-column window of a .5 boundary (our error + fp32's own rounding in ORT's formula) are recomputed on the scalar core with ORT's fp32 formula (`convert_w2sf`, `sfmpy`, `convert_sf2w`) | **bit-exact** |

Scalar loads from VTCM are slow (a tile needing the fix took 38 k cycles until HVX copied the group into stack
memory first). The fix path runs for 0.1-0.5% of 4-row groups.

### Phone (Xiaomi 12S, turbo, one HMX context; `qnn_parity/qdq_layer.py` layers, M = 2048 = 32x64 pixels)

| layer | QNN HTP vs ORT | ours `QC_FAST` vs ORT | ours `QC_EXACT` vs ORT | `QC_FAST` time | `QC_EXACT` time | QNN per layer |
|---|---:|---:|---:|---:|---:|---:|
| 256->256 | 7.46% off by one | 3.14% | **0** | 8.1 us (16.6 TMAC/s) | 304 us | -- |
| 256->256 + Relu (zy = 0) | 5.57% | 1.34% | **0** | 8.1 us | 347 us | -- |
| 512->512 | 7.07% | 3.02% | **0** | 31.4 us (17.1) | 721 us | 65 us (per-tensor chain above) |
| 1024->1024 | 7.39% | 3.23% | **0** | 125 us (17.2) | 1478 us | 155 us |
| 3x3 128->128, stride 1 / 2 (QNN only, for chunk 2) | 7.55% / 7.74% | | | | | |

- **QNN's HTP is not bit-exact against ORT either**: ~7% of outputs are off by one on every layer here
  (`qdq_layer.py` model through `qnn_run`, strict HTP; mostly -1). "QNN-class accuracy" is therefore a 7%
  off-by-one rate, and `QC_FAST` halves it at full HMX speed.
- `QC_EXACT` is bit-exact on all 4.2 M outputs, but its HVX requant (~0.6 ns per output on one thread) is
  10-40x slower than the HMX. Spreading it over the other HVX threads is the obvious next step.

Reproduce: `qnn_parity/qdq_layer.py <dir> 1024 1024 32 64` (model + ORT reference), `qnn_parity/export_case.py
<dir> <case>`, push the case, `./run.sh qconv <case> 20`; `sim/qconv_sim.c <case>` on hexagon-sim.
`tests/test_hmx_gemm.py::test_hmx_qconv_qdq_exact_on_hexagon_sim` builds a small layer and checks both modes.

## QDQ 3x3 convolution, stride 1 and 2 (`hmx_qconv3.h`)

Same requantization as above (`QC_FAST` / `QC_EXACT`), for `Conv` 3x3 with pad 1.

### The instruction form: `:single` row-offset windows (hexagon-sim one-hot map)

`activation.ub = mxmem(Rs, Rt):single:cm` reads a 64-row window that starts `4 * Rs[10:7]` rows into the crouton at
`Rs` (2 KB aligned) and continues into the crouton at `Rs + Rt[31:11]` (`Rt[10:0] = 0x7ff` as for `:cm`); `Rs[1]`
had no effect, so window starts are multiples of 4 rows. It runs at the full `:cm` + weight `:deep` rate (11 200
MAC/cycle in `hexagon-sim --timing`). `:above:cm` behaved the same in the plain probe, but hung the simulator in a
loop with a guessed `Rt`: QNN's 3x3 kernels (`hmx_convbbb_stride1`) pair it with per-instruction offset fields
from a table, which this directory has not mapped. `weight.b ... :dilate` ran at the same rate as the plain
weight load; its layout is not mapped either.

### Layout and taps

Activations are stored *flat*: pixel (y, x) is flat pixel `M0 + (y+1)*Wp + x`, with row stride `Wp = roundup(W+1, 4)`
(padding columns W..Wp-1), a padding row above and below, and a margin `M0` chosen so that the output region starts
on a 64-pixel block (an output tile is directly a block of the next layer's input). Padding pixels hold the
activation zero point, so padding is exact. Pixels go 64 per crouton (the `:cm` row dimension).

- Tap (dy, dx) of an output block reads the 64 input pixels at flat offset `dy*Wp + dx`. `dy*Wp` is a multiple of
  4, which is exactly a `:single` window (second crouton = next pixel block, `Rt[31:11] = kt * 2 KB`). `dx = -1/+1`
  read two one-pixel-shifted copies of the input (HVX `vlalign`/`valign` by 32 bytes).
- A 3x3 conv is then `9 * kt` `:single` instructions (with weight `:deep`, 64 output channels) per output tile
  pair, accumulating like a longer K. No im2col.
- Stride 2: HVX splits the input into its four row/column phases (`vdeal` by 32 bytes: even/odd pixels) in the
  output's flat geometry. Tap (dy, dx) reads phase (dy != 0, dx != 0) with a row offset of -1 for dy = -1 and the
  one-pixel-shifted odd-column phase for dx = -1. The stride-1 machinery then runs on quarter-size maps.
- **The 256 KB VTCM rule bites `:single`**: its two croutons are kt * 2 KB apart, and a pair straddling a 256 KB
  boundary faulted the PD (`Bad VA` just below the boundary; the simulator does not model it). About 1-3% of
  windows straddle. `qc_conv3x3_plan` finds them once per layer and routes them through side croutons that
  `qc_conv3x3_stitch` fills each run (16 vector moves each). `sim/qconv3_stitch_sim.c` forces every offset window
  through that path.
- The per-instruction addresses are a table built once per layer (building it per call cost more than the conv).

### Phone (Xiaomi 12S, turbo; `qdq_layer.py` layers at 32x64, both modes vs ORT CPU)

| layer | QNN vs ORT | `QC_FAST` vs ORT | `QC_EXACT` vs ORT | `QC_FAST` per layer (prep) | `QC_EXACT` per layer | QNN per layer |
|---|---:|---:|---:|---:|---:|---:|
| 3x3 128->128, s1 | 7.55% | 3.37% | **0** | 33.8 us (13.4) | 227 us | 14 us (L22 - L2 chain) |
| 3x3 256->256, s1 | -- | 3.12% | **0** | 103.9 us (28.9) | 1070 us | 82 us |
| 3x3 128->128, s2 | 7.74% | 3.40% | **0** | 29.1 us (23.9) | 81.5 us | not separable (whole 1-layer model 0.25 ms incl. ~0.2 ms launch/IO) |
| 3x3 64->64, s1 8x12 / s2 9x11 (odd sizes) | | 3.2% / 3.7% | **0** / **0** | | | |

- **Correctness: bit-exact against ORT CPU in `QC_EXACT` for stride 1 and 2, odd sizes, fused Relu**, on the
  phone and on hexagon-sim (stitched and direct windows).
- **Speed: QNN is still ahead on 3x3** (14 vs 34 us at 128 channels, 82 vs 104 us at 256). About 13-29 us of ours
  is the HVX shifted copies (and the phase split at stride 2), which QNN avoids. Its 21.6 TMAC/s at 128 channels
  is also above our `:cm` + `:deep` MAC-loop ceiling (17.2), so QNN's `:above`/`:dilate` pairing (2D spatial
  croutons with X offsets, presumably) does something this form does not. Mapping those is the lever. Running
  the copies on another HVX thread in parallel with the previous layer's HMX work would hide most of our prep.

Reproduce: `qdq_layer.py <dir> 128 128 32 64 3 1` (or `... 3 2` for stride 2), `export_case.py`, `./run.sh qconv
<case>`; `sim/qconv3_sim.c <case>`. QNN chains: `qnn_parity/gen_models.py <dir> 2 22 conv3x3` + `run_ceiling.sh`.

## A QDQ graph on the DSP in one FastRPC call: ResNet-18, bit-exact against ORT (`runner/`)

The conversion + runtime part of an SNPE/QNN replacement, on top of the kernels above.

- **Converter** (`runner/qdq_graph.py`, host Python): takes a static QDQ ONNX in onnxsim's `full_qdq` +
  `quantized_io(nhwc_inputs=...)` form and lowers it to a program. Supported ops: Conv (k x k, stride 1/2, pad k/2,
  per-channel int8, int32 bias; a Relu folded into the output Q is implicit), Add (ORT's QLinearAdd), MaxPool
  (3x3 s2 p1) and the NHWC input Transpose. Any other op stops with an error naming the node (no fallback
  partitioning yet). It also contains an **exact integer emulator** of the program, checked against ORT CPU.
- **Loader / planner** (`runner/rn_load.h`, portable C: the phone client and hexagon-sim):
  - packs the weights and the requantization params (`qc_pack_wk`, `qc_pack_params`); the stem's 3 input channels
    are padded to 32;
  - gives every tensor a flat geometry (`qc_geom2`: padding rows/columns per class of tensors that must share one);
  - builds each window op's sources and taps: column-shifted copies for stride 1; row/column phases plus their
    shifted copies for stride 2, including the 7x7 stem;
  - plans VTCM with liveness-based first-fit reuse (7.4 MB peak for ResNet-18 at 224x224).
- **DSP executor** (`runner/rn_exec.h`, skel RPCs `rn_load` / `rn_run` / `rn_unload`):
  - The weights (11.6 MB) stay in the DSP heap. `rn_load` acquires VTCM and plans every conv's instruction-address
    table.
  - `rn_run` executes the whole op list in **one FastRPC call**: HMX convs (`qc_convk`), the HVX Add and MaxPool,
    and padding maintenance after each op.
  - A second HVX thread prefetches the next conv's weights while the current ops run.

### Semantics that had to match ORT exactly

- Conv: `rne(fp32(acc) * M)` as above. Every conv and the MaxPool matched ORT in isolation from the start.
- **Add (ORT's MLAS QLinearAdd) is `rne(rb*b + (ra*a + fixed))`, `fixed = zy - (ra*za + rb*zb)`, with separate
  (unfused) fp32 operations in exactly that order.** Other orders, or a fused multiply-add, were 1 LSB off on 2 of
  200 k outputs of the first Add. Through the ReLU network those 2 LSBs grew into **20% of the final output**, so
  "close enough" is not: one LSB anywhere compounds.
- The DSP Add computes `a * ra` exactly: `a` times ra's 24-bit mantissa, split into 12-bit halves (`vmpy` + shifts).
  Lanes whose fraction is within the window of fp32's own rounding in ORT's formula are recomputed on the scalar
  core with that formula (`sfmpy`/`sfadd`/`convert_sf2w`). Build the loader with `-ffp-contract=off`.

### Result (Xiaomi 12S, V69, turbo; `runner/resnet18_qdq.py`: torchvision weights, onnxsim `full_qdq` calibrated on 32 coco128 images, 224x224, stem .. layer4 = 20 Conv + 8 Add + MaxPool)

| | vs ORT CPU (25 088 outputs of layer4) | time per inference |
|---|---:|---:|
| **ours, `QC_EXACT`** | **0 mismatches (bit-exact)** | **3.43 ms** on the DSP, 5.3-5.7 ms wall per call |
| ours, `QC_FAST` | 35.0% off (errors compound through the net) | 1.83 ms on the DSP, 3.5-3.9 ms wall per call |
| QNN HTP (ORT QNN EP 2.6.0 / QNN 2.50, strict, EP-context, burst) | **35.6% off** (4427 by more than 1, max 9; cosine 0.9963) | **0.43 ms** wall |

- **Accuracy: QNN's own result is 35.6% off ORT on this model.** Every conv and Add's 1-LSB differences compound.
  Our exact mode is the only one of the three that reproduces ORT, and our fast mode lands in the same class as QNN.
- **Speed: QNN is 4.3x faster on the DSP time (8x on wall).** Where our fast-mode 1.83 ms goes (`RN_OPS=1` per op
  and phase):
  - HMX + requant (all convs): ~0.52 ms. The MACs alone would be ~0.1 ms at 17 TMAC/s. The 7x7 stem computes
    32 padded channels for 3 real ones (10x), and small layers pay per-tile overheads.
  - Source building (shifted copies, phase splits): ~0.35 ms, 0.16 ms of it the stem. QNN's `:above`/`:dilate`
    instruction forms (unmapped here) avoid these copies.
  - Adds: ~0.64 ms, of which 0.23 ms is scalar near-tie fixes in the first three; the HVX loop is 0.55 cycles per
    element in hexagon-sim.
  - Layer4 weight streaming: ~0.25 ms that the one-op-ahead prefetch does not hide yet, since the Adds in between
    are short. DMA would help, as would prefetching two ops ahead.
  - MaxPool: 0.17 ms.
- Exact mode adds ~1.6 ms: the HVX integer requant on one thread (0.47 ms more on the stem alone). Spreading it over
  the idle HVX threads is the next step there.
- Wall per call adds ~1.7-2 ms on top of the DSP time: FastRPC copies the 1.65 MB flat input (zero-copy rpcmem
  buffers and packing on the DSP from the 150 KB NHWC input would remove most of it), plus thread and lock setup.

Optimizations measured on the way (phone, fast mode):

| step | per inference |
|---|---:|
| first working version | 3.75 ms |
| shift copies without a per-vector select (2.2x faster in sim) | |
| Add with exact mantissa products (window ~100 instead of ~350) and branch-free inner loop (4.6x in sim) | |
| Add near-tie rescan by 64-bit words instead of bytes (0.6 -> 0.12 ms on the first Add) | 2.18 ms |
| page-aligned input (aligned HVX loads) + `l2fetch` of the next input rows in the stem's phase split (stem sources 510 -> 165 us) | 1.82 ms |
| one-conv-ahead weight prefetch on a second HVX thread (vs synchronous copies) | 1.83 vs 2.00 ms |

Exact mode went from 23.9 ms to 3.43 ms. Two things fixed on the way:
- Near-dead output channels (tiny M) had made the stem's near-tie window enormous, flagging every output. Their
  output is exactly zy, which is now short-circuited.
- The fp32(acc) rounding term is dropped where those outputs saturate anyway.

Reproduce:

```
python runner/resnet18_qdq.py resnet18-f37072fd.pth <coco images> $R     # model, input, ORT reference
python runner/qdq_graph.py $R/backbone_qdq.onnx $R/prog $R/input.bin $R/ref.bin   # program + emulator check
./build.sh; D=/data/local/tmp/<dir> ./run.sh setup; adb push $OUT/hmx_runner_client $R/prog $R/input.bin $R/ref.bin $D/...
adb shell "cd $D && ADSP_LIBRARY_PATH=$D RN_OPS=1 ./hmx_runner_client '<uri>' prog input.bin ref.bin 20"
```

`tests/test_hmx_gemm.py::test_hmx_graph_runner_bit_exact_on_hexagon_sim` runs the whole runner on hexagon-sim for a
tiny ResNet-shaped graph (`runner/make_tiny.py`) and requires 0 mismatches against ORT.
