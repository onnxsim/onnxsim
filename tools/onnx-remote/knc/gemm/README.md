# SGEMM for Intel Knights Corner (k1om)

`gemm_knc.c` computes `C = A*B` in single precision (row-major; any shape through `sgemm_knc_any`, fastest with
`M % 16 == 0`, `K % 128 == 0`, `N % 64 == 0`) on a Xeon Phi
coprocessor 3120 series (Knights Corner) with the k1om GCC 5.1.1 toolchain (`NFCXX_K1OM_ROOT`, default `/opt/k1om`).

## Build and run

```sh
tools/onnx-remote/knc/gemm/build_gemm_knc.sh          # -> build/knc-gemm/gemm_knc, bench_kernel
# on the card (copy the binary over SSH as in tools/onnx-remote/knc/README.md):
./gemm_knc M N K [THREADS] [REPS]                     # e.g. ./gemm_knc 2048 2048 2048 228 8 (default 228 threads)
./bench_kernel THREADS REPS [PIN]                      # kernel only, L1-resident; threads pinned unless PIN=0
```

Output: max error on 64 sampled entries against a double-precision reference, and the best time per call. The check
runs on the output of the last timed call (C is filled with NaN before each call), so it covers the path that was timed.
`ANY=1` sends aligned shapes through `sgemm_knc_any` (both operands packed per call).

## Design

- **Microkernel** (`kernel_8x32`): 8 rows x 32 columns, 16 zmm accumulators. The KNC instructions are written in inline
  assembly: the k1om GCC turns generic vector code into x87 scalar code, and its `-mavx512f` mode emits AVX-512
  encodings the card does not implement. A reads are FMA memory broadcasts (`{1to16}`, escaped as `%{1to16%}`), B rows
  are aligned 64-byte loads, and the k loop is unrolled by four with software prefetch.
- **Packing**: B is packed into 32-column panels once (`sgemm_knc_pack_B`, for weights). A is packed as plain 8-row
  segments in the same pool pass as the compute (`sgemm_knc_run_overlap`, the default): packing items come first in the
  shared counter, and each compute item waits only for the packed blocks of its row chunk. `PHASED=1` selects the
  older two-phase path (`sgemm_knc_run`).
- **Blocking**: K blocks of 128 (a B panel is 16 KB), work items of 16 rows by 2 panels (64 columns, `-DRC`, `-DPG`), so
  a B sub-panel is reused across the rows of an item while it is in L1. Measured at 228 threads (2048^3 / 1024^3 / 512^3):
  RC=16 PG=2 802 / 629 / 352, RC=32 PG=4 748 / 489 / 213, RC=16 PG=1 732 / 669 / 420, RC=64 PG=4 610 / 425 / -,
  RC=8 599-620 / 477-491 / -. The panel count is chosen per call: PG=1 when there are fewer than `PG_MIN_ITEMS` (8)
  work items per thread at PG=2, which picked the faster of the two on every shape measured (512^3 to 2048^3, and skinny
  shapes such as 256 x 4096 x 1024); `PGSEL=n` overrides it. `-DINLINE_A=1` (packing A inside each item, no phase) was measured
  slower (325 vs 382 GFLOPS) and is off by default.
- **Short K** (K at most 512, `FULLK_NKC`): work items use the full-K microkernel (`kernel_full`), which keeps the
  accumulators in registers over all of K and writes C once instead of reading and writing it per K block. Measured
  (blocked vs full-K): 2048 x 2048 x 256 641 vs 786, 2048 x 2048 x 512 728 vs 807, 512^3 394 vs 446; for longer K the
  blocked kernel wins (2048 x 2048 x 1024 776 vs 700, 2048^3 781 vs 601) because it reuses each B sub-panel from L1.
  `NOFULLK=1` disables it.
- **Shared-panel schedule** (opt-in, `SHARED=1`, 228 threads, `M % 32 == 0`): the four hardware threads of a core work
  on the same panel group together, so they share one B panel in L1, with a full-K microkernel (`kernel_full`) that
  writes C once and A packing overlapped. It is measured slower than the default (681 vs 737 GFLOPS at 2048^3).
- **Threads**: a persistent pool; each thread is pinned to one hardware thread, and the calling thread runs task 0 (pinned
  the same way). The card numbers CPUs in groups of four per core, and spreading the first hardware thread of each core
  before the second is measured at 1.4x at 228 threads. Workers spin on a generation counter between runs (with the KNC
  `delay` instruction, which leaves the core's issue slots to the other hardware threads) for up to `POOL_SPIN_MS`
  (default 100 ms), then sleep on a condition variable. An empty run costs about 0.03 ms (4-10 ms with condition
  variables alone), which also removed most of the run-to-run noise.
- **Any shape** (`sgemm_knc_any`): A and B are packed straight from the caller's arrays in one parallel pass, with the
  padding (M to RC, N to 32 x PG, K to 128) written as zeros; C is written in place when its layout allows, else into a kept
  scratch buffer that is copied back in parallel.
- **K split** (`-DKS=n`): each K group writes a partial C, reduced at the end. With RC=32 PG=4 and the spinning pool,
  `KS=2` measured 755 vs 744 GFLOPS at 2048^3 and 551 vs 485 at 1024^3, but 755 vs 796 at 4096 x 2048 x 2048; the
  default is `KS=1`.
- **Prefetch distance** (`-DPF_DIST`, default 4): 2 and 8 measured 715 and 737 vs 744 GFLOPS at 2048^3.

## Measurements (one card, 228 threads unless noted)

Current default (overlapped A packing, spinning pool, RC=16 PG=2), medians of three interleaved rounds:

| shape | GFLOPS | % of 2006 GFLOPS nominal peak |
|---|---|---|
| 2048 x 2048 x 2048 | 777-802 | 39-40% |
| 4096 x 2048 x 2048 | 780 | 39% |
| 4096 x 4096 x 4096 | 775-788 | 39% |
| 2048 x 2048 x 512 | 799-806 | 40% |
| 2048 x 2048 x 256 | 751-786 | 37-39% |
| 768 x 3072 x 768 | 731-743 | 37% |
| 1024 x 1024 x 1024 | 657-667 | 33% |
| 1000 x 1000 x 1000 (padded, both operands packed per call) | 490-500 | 25% |
| 512 x 512 x 512 | 428-430 | 21% |
| 2048^3 with RC=32 PG=4: 228 / 171 / 114 / 57 threads | 737 / 683 / 547 / 370 | |
| 2048^3 with RC=32 PG=4: `PHASED=1` / `SHARED=1` | 733 / 681 | |
| kernel microbenchmark, 228 pinned threads, 100000 repetitions | 1024 | 51% |

History at 2048^3 (best of several runs, condition-variable pool):

| configuration | GFLOPS |
|---|---|
| generic vector code (x87 scalar) | 5.5 |
| KNC assembly microkernel, no blocking | 67 |
| blocked, packed panels, thread pool | 170 |
| pinned threads, 114 threads | 290 |
| B packed once, 114 threads | 390 |
| A packing overlapped with compute, 114-228 threads | 275-385 (interleaved medians) |
| spinning pool, 228 threads | 737-746 |
| work items of 16 rows x 2 panels (current) | 777-802 |

The nominal peak is 57 cores x 1.1 GHz x 32 flops per cycle. With the spinning pool the run-to-run variation is about +-2%.
At 2048^3 the GEMM runs at about 77% of the microkernel's own rate.

**Correction.** Earlier revisions of this file reported 540-690 GFLOPS at 2048^3 and 1037-1214 GFLOPS at
4096 x 2048 x 2048 for the shared-panel schedule. Those numbers were wrong: `kernel_full`'s inner k loop had lost its
back-edge, so it computed 4 of every 128 k values, and the benchmark checked the error on an earlier call's output (the
phased path) instead of on the timed call's. With the loop fixed, the shared-panel schedule measures 252-275 GFLOPS with
the old pool and 681 with the new one. The B-reuse, core-count, alignment, huge-page and prefetch comparisons below were
made against that broken default and under the old pool's noise; they are kept for the record but are not conclusive.

## Measured and not kept

- **B reuse through item order** (re-measured with the spinning pool): work items ordered panel-major in groups of G row
  chunks, so the items running at the same time share B panels. 2048^3: 782 (default, row-chunk-major) vs 755-768 for
  G = 4, 8, 16 and all; 4096^3: 779 vs 629-756. B re-reads from L2/memory are not the limit.
- **What limits the kernel: L1 capacity.** In the kernel benchmark each thread has its own 16 KB B panel and 4 KB A block,
  so four threads per core need 80 KB against a 32 KB L1. With K blocks that fit (KC=64 at 2 threads per core, KC=32 at 4)
  the kernel reaches 1055 and 1207 GFLOPS (vs 658 and 1003 at KC=128). In the GEMM, KC=64 and KC=32 were slower (2048^3:
  687 and 596 vs 786 at 228 threads), because C is read and written per K block and the calls are shorter, so KC stays 128.
- **Sharing B within a core (effective sharing), re-measured.** Kernel only (`bench_kernel` variant with one B panel per
  core, threads `id % 57` sharing): 1183-1190 GFLOPS vs 1017-1022 with private panels, so the sharing itself is worth
  about +16%. In the GEMM the shared-panel schedule (`SHARED=1`) does not gain from it: at 2048^3 it measured 706-734
  against 784 for the default; at 1024^3 517-520 against 664-677. Two causes, measured by removing each (`NOPACK`
  experiment, results invalid, packing skipped):
  - packing A inside the lockstep units costs about 35 GFLOPS at 2048^3 and about 200 at 1024^3 (where units are short);
  - without packing, the lockstep schedule (762-765) is level with the default (770-775) at 2048^3, so the barriers
    absorb the gain.
  Panel-group size in the lockstep schedule: 4 panels give 798-807 at 2048^3 without packing (2% over the default, within
  noise) but 447-457 at 1024^3, where there are too few units for 228 threads. Not adopted.
- **Microkernel encodings:** one base register per A row (all A displacements 8-bit, 6-7 byte instructions instead of
  10-byte ones): no change (1012-1031 vs 1015-1020 GFLOPS kernel-only). A loaded once per 4 k with `vbroadcastf32x4` and
  broadcast by register swizzles (`{aaaa}`..`{dddd}`, no memory operand in the FMAs): 884-890 with the FMAs of one row
  back to back, 1009-1025 interleaved, so no gain.
- **Hardware counters:** the card's kernel accepts `perf_event_open`, but the call blocks in the kernel (uninterruptible
  sleep), and the card needed a reset afterwards. Traffic figures are therefore model-based, not counted.
- **Streaming bandwidth** (vector loads, 1 GB): 36 GB/s at 57 threads, 19 GB/s at 171, 14.5 GB/s at 228 (reads). It falls with
  thread count, which points at contention; a true ceiling was not established.

- **Leaving the OS core alone** (`-DNCU=56`, 224 threads), **2 MB alignment** (`-DBIGALIGN=2097152`), **explicit huge pages**
  (`HUGETLB=1` with `/proc/sys/vm/nr_hugepages` reserved) and **prefetch distance** 2/8/16: none was better in interleaved
  runs. Transparent huge pages are set to `always` on the card, but `AnonHugePages` stays at 0 during a run. The knobs stay
  as options.

- **Operand streaming (next bottleneck candidate), measured at 228 threads.** Kernel alone reaches 1014 GFLOPS. The GEMM
  has a steady-state rate of about 812 GFLOPS per unit of K (slope of time vs K at 2048 x 2048: 0.0103 ms per k, with a
  fixed cost of about 0.6 ms). At fixed flops, the speed follows how many panels share each A block concurrently:
  8192 x 64 x 2048 395, 8192 x 256 x 2048 630, 8192 x 512 x 2048 728, 2048^3 800. Other item shapes do not help:
  RC=16 PG=4 785 (1024^3 681), RC=32 PG=2 801 (1024^3 649), RC=8 PG=4 602. The current 16 x 2 stays. NOFULLK at K=128
  gives 687 (one K block), so the per-call work, not C traffic, sets the steady-state rate. The kernel-only rate at 57,
  114 and 228 threads is 677, 708 and 1014.
- **Barrier-free core-shared schedule** (`STATIC=1`: A packed in a separate pass, the four threads of a core take row
  chunks of one panel group with no barriers). 2048^3 743 GFLOPS total (compute alone 22.9 ms, vs 22.0 ms for the
  default including packing), 1024^3 683 vs 667, 8192 x 512 x 2048 706 vs 706. So the B sharing that gains 16% in the
  kernel does not carry over to the GEMM, and the limit is the A and B operand fetch per call, not B's L1 capacity.

## Measurement noise

With the condition-variable pool, the same binary varied by about +-15% between runs (thread wake-ups of several ms per
pool run). With the spinning pool it is about +-2%. Compare variants in interleaved rounds and use the median.

## Shapes

`sgemm_knc_any` accepts any M, N, K (see Design). Odd shapes are checked against the reference (for example 1x1x1,
37x129x65, 33x257x130, 500x700x300, 1000^3, 2000^3). Measured per call: 37x129x65 0.43 ms (was 123 ms), 1000^3 5.3 ms
(was 71 ms; serial padding copies and fresh allocations), 2000^3 24.9 ms (was 235 ms). With the per-call panel count
and the short-K kernel: 37x129x65 0.23 ms, 1000^3 4.0 ms, 500x700x300 1.0 ms.

## Limits

- Requires the k1om toolchain and a card; the kernel is written for KNC (KC == 128 is enforced).
- Tested on one card with MPSS 3.8.6.
