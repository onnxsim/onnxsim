# Opt-in GPU backend for `onnxsim.crown` / `backward_diff` / `quant_verify`

`device=` and `precision=` are accepted by `crown.bounds`, `crown.bab_bounds`,
`crown.verify_output_ranges`, `backward_diff.bound_difference` and
`quant_verify.verify(engine="backward")`. The default (`device=None`/`"cpu"`) is the
unchanged numpy float64 path and never imports torch.

| `device` | meaning |
|---|---|
| `"cpu"` (default) | numpy float64, as before |
| `"cuda"`, `"cuda:N"` | torch on an accelerator (CUDA, or ROCm through torch's `cuda` API) |
| `"auto"` | `"cuda"` if available, else `"cpu"` |
| `"torch-cpu"` | the torch code path on the CPU (used by CI to cover the backend) |

`precision` is `"float64"` (default) or `"float32"` (needs a torch device). A requested
accelerator that is missing raises; onnxsim never falls back to the CPU silently.
Results come back as numpy on the host. Backward rows are chunked to fit free device
memory; on an out-of-memory error the chunk is halved and retried, and a single row that
still does not fit raises `DeviceMemoryError`.

## Sound float32

float32 is 16-33x faster than float64 on consumer GPUs, but plain float32 bounds are not
sound. `precision="float32"` therefore evaluates every backward step with a running
rounding-error bound (Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 3):

* every value is a pair (computed, error bound); sums use block-wise tree reduction with
  a `gamma_n` bound, products/divisions add `u` relative error, all inflated by a margin;
* the relaxation lines and constants are rounded outward (`up32`/`down32`) and lines are
  re-validated in float32; PRIMA cuts are rounded so they stay valid at the hull vertices;
* the final lower/upper bound is `value -/+ error` (`-inf`/`+inf` on overflow or NaN);
* TF32 and cuDNN Winograd/FFT convolution are disabled (`strict_float32()`); backward
  convolution is expressed as an einsum/GEMM;
* alpha/beta: the optimiser runs in plain float32 (any alpha is valid), then the chosen
  slopes/multipliers are **re-evaluated rigorously**, so the reported bound is sound.

Assumptions (the scheme is only as sound as these): IEEE-754 round-to-nearest float32
add/mul/div/fma on the device, no fast-math, and that a GEMM/reduction of length n obeys the
`gamma_n` bound for any summation order (true for standard blocked kernels).

Honest note on the existing float64 path: it only widens results by a small relative
amount (~1e-14, plus a 1e-9 slack in places). That is not rigorous directed rounding
either; the float32 mode is in this respect *stricter* than float64.

Tests: `tests/test_gpu_backend.py`. Mutation tests break the error constants and the
cut rounding and check the primitive/soundness checks then fail; sampled outputs of
onnxruntime must lie inside the float32 bounds.

## Measured speedups (honest)

`scripts/gpu_bench.py` (not run in CI); raw JSON lines were written to
`/mnt/data/cache/claude-work/gpu-bench/results/`. Median of 2 repetitions, wall seconds,
including host-device upload. Hardware: RTX 5050 (8 GB), Ryzen AI 8060S iGPU (ROCm),
host CPU numpy. The conv classifier is 3x(Conv,Relu)->GAP->head, 10 outputs.
**Timings are noisy (single machine, 2 reps, first call pays import/JIT costs); read
differences under ~30% as noise.** `torch-cpu` varies widely for the same reason.

| case | numpy f64 | 5050 f32 sound | 5050 f64 | 8060S f32 sound | 8060S f64 |
|---|---|---|---|---|---|
| crown, 8ch, 224px | 1.80 | 1.27 | 0.63 | 0.63 | 0.73 |
| crown, 16ch, 224px | 2.61 | 1.81 | 0.97 | 1.03 | 0.78 |
| backward-diff verify, 8ch, 224px | 7.79 | 5.73 | 3.76 | 3.31 | 2.89 |
| backward-diff verify, 16ch, 224px | 13.19 | 6.79 | 5.40 | 5.98 | 5.26 |
| alpha, 8ch, 32px | 3.61 | 2.10 | 0.92 | 0.97 | 0.83 |
| BaB conv 16px (budget 16) | 13.45 | 1.55 | 2.03 | 1.40 | 1.12 |
| BaB MLP 256 evals | 1.18 | 4.99 | 1.82 | 3.60 | 1.80 |
| BaB MLP relu/beta | 4.58 | 4.54 | 3.14 | 4.05 | 3.07 |
| ResNet18 crown 224 | 4.69 | 2.30 | 2.72 | 2.16 | 2.01 |
| ResNet18 weight-only int8 verify 224 | 38.35 | 24.83 | 27.05 | 22.73 | 23.06 |
| crown/verify at <=64px | 0.2-1.1 | 0.4-1.3 | 0.3-0.6 | 0.3-0.6 | 0.3-0.6 |

Takeaways:

* **Speedup is modest: about 1.5-3x at best on end-to-end runs, ~10x on one conv BaB
  case, and none (or a slowdown) for small problems.** The 224px, 16-channel
  `backward_diff` run is 2.0-2.5x faster on either GPU. Below ~64 px, or for MLP BaB, the
  GPU is as slow or slower (launch/transfer overhead dominates; the sound float32 mode is
  *slower than float64* on the 5050 in several small cases because of error tracking).
* The earlier raw conv-kernel measurement (33x float32 on the 5050) does not carry over to
  these end-to-end runs: the classifier ends in global average pooling with a 10-row
  backward pass, so there are few rows to batch, and much time is spent outside the GEMMs
  (relaxation construction, host numpy bookkeeping).
* float64 is *not* slower than sound float32 here, so for these workloads `float64` on the
  GPU is the better default; float32 only pays if the model is large enough to be compute
  bound.
* Upload cost: moving all ResNet18 weights to the device took 0.01-0.02 s (47 MB as
  float32, 93 MB as float64), negligible against the runs above.
* Peak device memory was small (< 600 MB in every case; the only run above 500 MB was
  `head_verify_224_16`), well inside the 5050's 8 GB.
* Looseness cost of sound float32: bound widths differ from float64 in the 4th-5th
  significant digit (e.g. 58.33 vs 58.34, 175.0 vs 175.1); about 0.002% for crown and
  0.01% for alpha on the test nets, up to ~0.25% on a PRIMA MLP probe.
* The ResNet18 weight-only int8 bound is still vacuous (6e19 at 64px, 1.6e21 at 224px).
  Faster arithmetic does not change that; it is a relaxation-tightness problem.

## Not done / limits

* BaB sub-boxes are *not* batched on the device: each region still builds its own analyser
  and relaxations, so BaB gains come only from faster per-region passes. Batching needs a
  restructure of `_Bab`.
* `zonotope` engine has no device support (`quant_verify` raises if you ask for one).
* ORT WebGPU was measured and was worse; not used.
* float32 soundness depends on the hardware assumptions above and was checked by sampling
  and mutation tests, not formally verified.
* Not measured: multi-GPU, `refine` on tensors > 4096 elements (never refined), runs
  longer than the 900 s per-case timeout (none occurred).
