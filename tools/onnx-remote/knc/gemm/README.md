# SGEMM for Intel Knights Corner (k1om)

`gemm_knc.c` computes `C = A*B` in single precision (row-major, `M % 64 == 0`, `K % 128 == 0`, `N % 128 == 0`) on a Xeon Phi
coprocessor 3120 series (Knights Corner) with the k1om GCC 5.1.1 toolchain (`NFCXX_K1OM_ROOT`, default `/opt/k1om`).

## Build and run

```sh
tools/onnx-remote/knc/gemm/build_gemm_knc.sh          # -> build/knc-gemm/gemm_knc, bench_kernel
# on the card (copy the binary over SSH as in tools/onnx-remote/knc/README.md):
./gemm_knc M N K THREADS REPS                         # e.g. ./gemm_knc 2048 2048 2048 171 8
./bench_kernel THREADS REPS [PIN]                      # kernel only, L1-resident; threads pinned unless PIN=0
```

Output: max error on 64 sampled entries against a double-precision reference, and the best time per call.

## Design

- **Microkernel** (`kernel_8x32`): 8 rows x 32 columns, 16 zmm accumulators. The KNC instructions are written in inline
  assembly: the k1om GCC turns generic vector code into x87 scalar code, and its `-mavx512f` mode emits AVX-512
  encodings the card does not implement. A reads are FMA memory broadcasts (`{1to16}`, escaped as `%{1to16%}`), B rows
  are aligned 64-byte loads, and the k loop is unrolled by four with software prefetch.
- **Packing**: B is packed into 32-column panels once (`sgemm_knc_pack_B`, for weights). A is packed as plain 8-row
  segments in the same pool pass as the compute (`sgemm_knc_run_overlap`, the default): packing items come first in the
  shared counter, and each compute item waits only for the packed blocks of its row chunk. `PHASED=1` selects the
  older two-phase path (`sgemm_knc_run`).
- **Blocking**: K blocks of 128 (a B panel is 16 KB), work items of 32 rows by 4 panels (128 columns), so a B sub-panel is
  reused across the rows of an item while it is in L1. `-DINLINE_A=1` (packing A inside each item, no phase) was measured
  slower (325 vs 382 GFLOPS) and is off by default.
- **Shared-panel schedule** (default at 228 threads, `M % 32 == 0`): the four hardware threads of a core work on the same
  panel group together, so they share one B panel in L1 (kernel-only: 1112 vs 903 GFLOPS). The microkernel keeps its 16
  accumulators across all K blocks, so C is written once. Cores claim units from a global counter; two per-core barriers
  keep the four threads in step. A packing is overlapped with it: each core's leader claims the quad four ahead, and its four
  threads pack one block each while computing (the first four quads are packed before the run). Other thread counts use
  the overlapped path.
- **Threads**: a persistent pool; each thread is pinned to one hardware thread. The card numbers CPUs in groups of four per
  core, and spreading the first hardware thread of each core before the second is measured at 1.4x at 228 threads.
- **K split** (`-DKS=n`): each K group writes a partial C, reduced at the end. It is measured slower on this card (the
  partial buffers and the reduction cost more than the extra parallelism); the default is `KS=1`.

## Measurements (2048 x 2048 x 2048, one card, best of several runs)

| configuration | GFLOPS | % of 2006 GFLOPS nominal peak |
|---|---|---|
| generic vector code (x87 scalar) | 5.5 | 0.3% |
| KNC assembly microkernel, no blocking | 67 | 3% |
| blocked, packed panels, thread pool | 170 | 8% |
| pinned threads, 114 threads | 290 | 14% |
| B packed once, 114 threads | 390 | 19.5% |
| small work items (PG=1, RC=32), 228 threads | 360 | 18% |
| work items of 32 rows x 4 panels, 171 threads (default) | 419 | 21% |
| same, 114 threads | 408 | 20% |
| same, 228 threads | 376 | 19% |
| A packing overlapped with compute, 171 threads | 468-509 | 23-25% |
| shared-panel schedule (default at 228 threads), full-K kernel | 540-577 | 27-29% |
| shared-panel schedule, 4096 x 2048 x 2048 | 890-1018 | 44-51% |
| shared-panel schedule with A packing overlapped (default), 2048^3 | 620-690 | 31-34% |
| same, 4096 x 2048 x 2048 | 1037 | 52% |
| same, 228 threads | 489 | 24% |
| kernel microbenchmark, 228 pinned threads (no GEMM overhead) | 765 | 38% |

The nominal peak is 57 cores x 1.1 GHz x 32 flops per cycle. Run-to-run variation is about +-10%. The GEMM is not at its
ideal: the remaining cost is packing A (about 7 ms of about 45 ms per call, serial in phase), and the compute phase
reaches about 75% of the microkernel's own rate.

## Shapes

`sgemm_knc_any` accepts any M, N, K: it pads M to 32, N to 128 and K to 128 with zeros, runs the fast path, and copies
the result back. Odd shapes are checked (for example 37x129x65, 33x257x130, 1000x1000x1000) against the reference. Small
shapes are slow because the padding copies and the thread-pool wakeups cost more than the arithmetic.

## Limits

- Requires the k1om toolchain and a card; the kernel is written for KNC (KC == 128 is enforced).
- Tested on one card with MPSS 3.8.6.
