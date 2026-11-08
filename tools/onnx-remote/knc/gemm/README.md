# SGEMM for Intel Knights Corner (k1om)

`gemm_knc.c` computes `C = A*B` in single precision (row-major, `M % 64 == 0`, `K % 128 == 0`, `N % 128 == 0`) on a Xeon Phi
coprocessor 3120 series (Knights Corner) with the k1om GCC 5.1.1 toolchain (`NFCXX_K1OM_ROOT`, default `/opt/k1om`).

## Build and run

```sh
tools/onnx-remote/knc/gemm/build_gemm_knc.sh          # -> build/knc-gemm/gemm_knc, bench_kernel
# on the card (copy the binary over SSH as in tools/onnx-remote/knc/README.md):
./gemm_knc M N K THREADS REPS                         # e.g. ./gemm_knc 2048 2048 2048 114 8
./bench_kernel THREADS REPS PIN                       # kernel only, L1-resident, PIN=1 pins threads
```

Output: max error on 64 sampled entries against a double-precision reference, and the best time per call.

## Design

- **Microkernel** (`kernel_8x32`): 8 rows x 32 columns, 16 zmm accumulators. The KNC instructions are written in inline
  assembly: the k1om GCC turns generic vector code into x87 scalar code, and its `-mavx512f` mode emits AVX-512
  encodings the card does not implement. A reads are FMA memory broadcasts (`{1to16}`, escaped as `%{1to16%}`), B rows
  are aligned 64-byte loads, and the k loop is unrolled by four with software prefetch.
- **Packing**: B is packed into 32-column panels once (`sgemm_knc_pack_B`, for weights); A is packed as plain 8-row
  segments on each call (`sgemm_knc_run`).
- **Blocking**: K blocks of 128 (a B panel is 16 KB), work items of 32 rows by 2 panels, so a B sub-panel is reused
  across the rows of an item while it is in L1.
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
| kernel microbenchmark, 228 pinned threads (no GEMM overhead) | 765 | 38% |

The nominal peak is 57 cores x 1.1 GHz x 32 flops per cycle. Run-to-run variation is about +-10%. The GEMM is not at its
ideal: the remaining cost is packing A (about 7 ms of about 45 ms per call, serial in phase), and the compute phase
reaches about 75% of the microkernel's own rate.

## Limits

- Requires the k1om toolchain and a card; the kernel is written for KNC (KC == 128 is enforced).
- Dimension restrictions as above; no edge handling.
- Tested on one card with MPSS 3.8.6.
