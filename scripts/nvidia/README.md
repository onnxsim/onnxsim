# NVIDIA (TensorRT / CUDA) validation

Real-hardware checks of onnxsim output against the TensorRT builder, complementing the
hand-built-graph tests in `tests/test_tensorrt_*.py` (which never invoke TensorRT).

| file | interpreter | role |
|---|---|---|
| `qdq_pairs.py` | onnxsim (Python >= 3.11) | writes `<name>.orig.onnx` / `<name>.sim.onnx` pairs |
| `trt_harness.py` | system Python with `tensorrt` | builds engines, dumps per-layer tactic/precision, times inference, compares orig vs sim outputs |
| `trt_compile.py` | system Python with `tensorrt` | external compiler-service adapter; writes a serialized engine and profiling manifest |
| `modelopt_pipeline.py` | onnxsim + `nvidia-modelopt[onnx]` | fixes batch, runs `simplify()` and ModelOpt INT8 quantization on a real model, with an un-simplified control |
| `bench_trtexec.py` | any (stdlib) | builds/times every variant with `trtexec` and prints a table |
| `imagenette_data.py` | any with Pillow | preprocesses Imagenette (real ImageNet images, 10 classes) into val + calibration `.npy` sets |
| `ort_cpu_check.py` | any with onnxruntime | top-1 of ONNX models on a val subset via ORT CPU: separates "quantized model is inaccurate" from "TensorRT mishandles it" |
| `eval_accuracy.py` | system Python with `tensorrt` | streams the val set through each variant's TensorRT engine; top-1 and agreement with fp32 |
| `sparsity_check.py` | onnxsim (`gen`) then system Python with `tensorrt` (`trt`) | checks whether TensorRT's builder gives 2:4-pruned `MatMul`/`Gemm` weights sparse tactics |
| `trt_nms_check.py` | onnxsim (`gen`), system Python with `tensorrt` (`trt`), any (`compare`) | differentially checks `rewrite_trt_batched_nms`'s output against the real `BatchedNMSDynamic_TRT` plugin |
| `run_cuda_feature_notebook.py` | onnxsim + a GPU `onnxruntime` | runs `examples/cuda_feature_tests/cuda_feature_tests.ipynb`'s tests as a plain script |
| `llm_pipeline.py` | onnxsim (Python >= 3.11) | pins shapes on a decoder-with-KV-cache LLM export and runs `simplify()` on it |
| `llm_block_split.py` | onnxsim (`split`), system Python with `tensorrt` (`build`) | splits a decoder LLM into N-layer TensorRT-buildable blocks, chains them, measures real end-to-end latency per block size |
| `edgellm_simplify.py` | onnxsim (Python >= 3.11) | runs onnxsim on a TensorRT Edge-LLM export dir for `llm_build` and checks plugin nodes / graph I/O are untouched; `rms-stack` writes a synthetic RMSNorm+MLP stack for `trtexec` |

They are split because JetPack 6's TensorRT Python bindings are cp310-only while onnxsim
needs Python >= 3.11; models are exchanged as `.onnx` files.

```sh
python3.12 scripts/nvidia/qdq_pairs.py /tmp/pairs                      # onnxsim venv
python3.10 scripts/nvidia/trt_harness.py compare /tmp/pairs --int8 --fp16   # tensorrt venv
python3.10 scripts/nvidia/trt_harness.py build model.onnx --int8            # per-layer detail
python3.10 scripts/nvidia/trt_harness.py profile model.engine --json       # runtime latency record
```

The TensorRT venv needs `onnx` and `numpy<2` (`uv venv --system-site-packages` picks up
the apt-installed `tensorrt`). CUDA is reached via ctypes on `libcudart.so.12`.
`/usr/src/tensorrt/bin/trtexec` (package `libnvinfer-bin`) is useful for cross-checking:
on the 32-channel INT8 Conv it reported 0.049 ms mean GPU compute vs 0.062 ms wall-clock
from the harness (the harness includes launch overhead).

## Findings (Jetson Orin Nano 8GB, JetPack 6 / L4T R36.4.7, TensorRT 10.3.0, CUDA 12.6, sm_87)

`simplify()` on TensorRT's documented explicit-quantization Q/DQ conventions:

- Per-channel weight Q/DQ (axis 0) + per-tensor activation Q/DQ on a Conv: the real engine
  fuses both into one INT8 `sm80_xmma_..._i8f32_i8i32` convolution (2 layers), before and
  after `simplify()`.
- Symmetric zero-point MatMul: builds and runs in INT8; identical engine before/after.
- Residual `Add` with an independent Q/DQ per branch: engine contains
  `PWN(Add)` with two INT8 inputs, i.e. the whole `Add` runs in INT8, before and after.
- For all four models (three Q/DQ + a Conv/BN/ReLU control) in both INT8 and FP16:
  identical layer counts and **bit-identical outputs** (`max|diff| = 0`) between original
  and simplified. `simplify()` leaves these Q/DQ graphs node-for-node unchanged, so the
  engines are the same.
- Control (Conv+BN+ReLU, no Q/DQ): `simplify()` folds BN (3 -> 2 nodes) but TensorRT
  already folds BN itself (1 fused layer either way), so no engine-level gain here.

Caveats: tiny models, latencies are dominated by launch overhead and are not benchmark
numbers. The harness only enables `--int8` for graphs that contain Q/DQ nodes (TensorRT
fails with "no scaling factors" otherwise, since no calibrator is supplied).

Memory note: the Orin's 7.4 GB is shared CPU/GPU. `trtexec` failed with CUDA OOM while a
parallel C++ build was running; retry on an idle machine.

## ModelOpt + TensorRT: latency and accuracy on 3 ImageNet classifiers

ONNX model zoo `resnet18-v1-7`, `resnet50-v1-7`, `mobilenetv2-12`, all pretrained
ImageNet-1k. The pipeline lifts each to opset 17, pins the batch, then runs
`onnxsim.simplify()` (`sim`; ResNet-18 69 -> 49 nodes, ResNet-50 175 -> 122, MobileNetV2
106 -> 100) and ModelOpt INT8 (explicit Q/DQ, entropy or max calibration, mixed FP16/INT8
output, built with `--int8 --fp16`). `raw` is the same model without onnxsim.

```sh
python3.12 scripts/nvidia/imagenette_data.py imagenette2-320 /tmp/data      # needs Pillow
python3.12 scripts/nvidia/modelopt_pipeline.py resnet18.onnx /tmp/pl_r18 --batch 1 8 \
    --calib-npy /tmp/data/calib_x.npy --methods entropy max
python3 scripts/nvidia/bench_trtexec.py /tmp/pl_r18 --glob '*.sim*.onnx'       # latency
python3.10 scripts/nvidia/eval_accuracy.py /tmp/pl_r18 /tmp/data              # accuracy
```

**Accuracy** is 1000-way top-1 on the **Imagenette val split**: 3,925 real ImageNet images
of 10 ImageNet-1k classes (the gated ImageNet val set was not available). These are easy
classes, so absolute numbers run high; compare variants, not against published top-1.
Calibration uses 128 *train* images of the same 10 classes, disjoint from val, which is
in-distribution and likely flatters INT8 compared with a proper 1000-class calibration
set. Standard error on 3,925 images is ~0.65 points; "agree" is top-1 agreement with the
fp32 engine (two fp32 builds of one model agree only 99.9-100%: TensorRT tactic noise).

| model (sim, batch 8 eval) | fp32 | fp16 | int8 entropy | int8 max | agree w/ fp32 (entropy) |
|---|---|---|---|---|---|
| ResNet-18 | 75.18 | 75.16 | 75.46 | 75.44 | 95.95% |
| ResNet-50 | 80.99 | 80.97 | 80.82 | 80.74 | 97.38% |
| MobileNetV2 | 79.03 | 79.03 | **77.76** | **76.99** | 90.96% |

**Latency** (mean GPU ms, `trtexec --noDataTransfers`, MAXN_SUPER, simplified model,
INT8 = entropy calibration; max calibration is within 3% of it everywhere):

| model | batch | fp32 | fp16 | int8+fp16 | int8 vs fp16 |
|---|---|---|---|---|---|
| ResNet-18 | 1 | 1.578 | 0.758 | 0.485 | 1.56x |
| ResNet-18 | 8 | 8.357 | 3.537 | 1.927 | 1.84x |
| ResNet-50 | 1 | 3.596 | 1.814 | 1.190 | 1.52x |
| ResNet-50 | 8 | 19.969 | 8.816 | 4.930 | 1.79x |
| MobileNetV2 | 1 | 1.386 | 0.850 | 0.781 | **1.09x** |
| MobileNetV2 | 8 | 8.147 | 3.925 | 2.446 | 1.60x |

Takeaways:
- ResNets: INT8 is 1.5-1.8x faster than FP16 (3.0-4.3x vs fp32) at no measurable accuracy
  cost (within the ~0.65-point standard error).
- MobileNetV2 is the counter-example: INT8 loses 1.3 (entropy) to 2.0 (max) points and
  only 91% of predictions match fp32, while batch-1 INT8 is just 1.09x faster than FP16
  (depthwise convs get little from INT8 and add many Q/DQ reformats). On this board
  MobileNetV2 is better left at FP16 unless batch >= 8. Entropy beats max calibration.
- **onnxsim did not change accuracy or latency** for any of the three: raw and simplified
  models agree within noise (MobileNetV2's quantized raw and sim models score identically),
  since TensorRT and ModelOpt already fold BN/constants themselves. The earlier ResNet-18
  latency check (raw vs sim, batch 1/8) likewise showed <1% differences.

## Transformer: ViT-B/16 (`Xenova/vit-base-patch16-224`, HF `google/vit-base-patch16-224`)

```sh
python3.12 scripts/nvidia/imagenette_data.py imagenette2-320 /tmp/data_vit vit    # resize 224, mean=std=0.5
python3.12 scripts/nvidia/modelopt_pipeline.py vit_base.onnx /tmp/pl_vit --batch 1 --chw 3 224 224 \
    --calib-npy /tmp/data_vit/calib_x.npy --calib-n 32 --methods entropy \
    --variants default hp32 noattn hp32+noattn only-linear      # see variant_options()
python3 scripts/nvidia/bench_trtexec.py /tmp/pl_vit
python3.10 scripts/nvidia/eval_accuracy.py /tmp/pl_vit /tmp/data_vit --glob 'b1.*.onnx' \
    --ref b1.raw.onnx --mean .5 .5 .5 --std .5 .5 .5
```

The HF export is opset 11 with symbolic H/W and 1,375 nodes once lifted to opset 17
(385 `Constant`, 111 `Shape`, hand-decomposed LayerNorm/GELU); `simplify()` takes it to 420
nodes (standard `LayerNormalization`, `Gemm`, `Split`). Batch 1 only: a batch-8 ModelOpt run
was OOM-killed on the 8 GB board even with process isolation and 32 calibration images.
Calibration is 32 train images, same-10-class caveat as above. Standard error is ~0.6 points.

| batch 1 | latency ms | top-1 % |
|---|---|---|
| fp32, raw | 13.06 | 85.22 |
| fp32, onnxsim | 13.65 | 85.22 |
| fp16, raw | 5.59 | **18.96** |
| fp16, onnxsim | 5.68 | 85.27 |
| ModelOpt int8, default (all ops), raw | 4.30 | **0.03** |
| ModelOpt int8, default (all ops), onnxsim | 4.58 | **4.99** |
| ModelOpt int8, onnxsim, `only-linear` (fp16 remainder) | **4.92** | **84.51** |
| ModelOpt int8, onnxsim, `only-linear` (fp32 remainder) | 5.07 | 84.51 |

Findings, and here onnxsim does matter:

1. **Raw FP16 is broken; onnxsim fixes it.** The raw graph's hand-decomposed LayerNorm does
   `Pow(x, 2)`. On a real image, 13 of its 25 LayerNorms receive values up to |x| = 1418
   (the residual-stream outliers), so x^2 ~ 2e6 exceeds FP16's 65,504 and overflows: top-1
   collapses to 19%. onnxsim collapses the pattern into `LayerNormalization`, which TensorRT
   accumulates in fp32 (extra `Cast`s in the engine), and FP16 stays at 85.27%. The
   correct engine is ~2-7% *slower* than the broken one (92 vs 89 layers: per-encoder-layer
   extra small kernels and unfused `Gemm`s), a price worth paying. This was read from engine
   layer names, not profiled.
2. **ModelOpt's default INT8 collapses ViT-B** (0.03% raw, 5.0% simplified; max calibration
   and excluding attention MatMuls / head / FP16 remainder change nothing, all 0.4-5%).
   ORT CPU shows the same collapse (4.7% vs 78.7% fp32 on 150 images), so it is the
   quantized model, not TensorRT.
3. **Cause: ModelOpt quantizes the residual-stream `Add`s** (the ~46 Q nodes beyond the
   Linears in the `noattn` model), where those |x| ~ 1000 outliers live. Quantizing any one
   Linear group alone (QKV, attn-out, FC1, FC2) is harmless (77-79% vs 78.7% on ORT-150),
   and all 48 Linears together (`only-linear`, 96 Q nodes, `Add`/attention left in
   FP16/FP32) give **84.51% (-0.7 vs fp32, ~ within noise)**.
4. That correct INT8 model is **1.15x faster than FP16 (4.92 vs 5.68 ms) and 2.8x faster than
   FP32**, far less than the 1.5-1.8x the ResNets get, because attention, LayerNorm, GELU
   and the residual stream stay in higher precision. The fully-quantized 4.58 ms model is
   only ~7% faster and produces garbage.
5. The raw-graph counterpart of `only-linear` could not be built: the group matcher found no
   weight Linears in the raw graph (0 Q nodes, so those rows are plain FP16 at 19%). The
   only raw INT8 result is the default one (0.03%).

## DLA (NVDLA): not available on this board

The Orin Nano has no DLA. `tensorrt.Builder().num_DLA_cores == 0`, there is no
`/dev/nvdla*`, and `trtexec --useDLACore=0` (with or without `--allowGPUFallback`, or
`--buildDLAStandalone`) fails with `Cannot create DLA engine, 0 not available` even
though `nvidia-l4t-dla-compiler` is installed. DLA compilation and latency need an
Orin NX or AGX Orin; none of the numbers above involve a DLA.

## 2:4 sparsity: does `convert_matmul_to_gemm` matter on TensorRT 10.3?

`scripts/nvidia/sparsity_check.py` checks `onnxsim/tensorrt_sparsity.py`'s claim (citing
[NVIDIA/TensorRT#2271](https://github.com/NVIDIA/TensorRT/issues/2271), filed against an
older TensorRT) that N:M sparse math only ever applies to `Gemm`, not `MatMul`, so
2:4-pruned transformer FFN/attention weights (which are wired through `MatMul`, since
their activation is 3-D) need `convert_matmul_to_gemm` first. **Scope note**: that claim
is specifically about *ONNX Runtime's* TensorRT execution provider
(`ORT_TENSORRT_SPARSITY_ENABLE=1`); this checks the underlying TensorRT builder directly
(same as `trt_harness.py`), which ORT's EP delegates to -- informative, not a direct test
of the literal claim (see `run_cuda_feature_notebook.py`'s findings for why ORT-TRT-EP
itself could not be run on this board).

```sh
python3.12 scripts/nvidia/sparsity_check.py gen /tmp/sp --layers 6      # onnxsim venv
python3.10 scripts/nvidia/sparsity_check.py trt /tmp/sp --runs 3        # tensorrt venv
```

Built a 6-layer ViT-B-shaped MLP stack (768&rarr;3072 ReLU 3072&rarr;768) at three
activation shapes -- `2d197` `[197,768]`, `3d197` `[1,197,768]` (the realistic
batched/transformer case), `2d2048` `[2048,768]` (larger, 2-D) -- each as dense and
`apply_magnitude_pruning(n=2, m=4)`-pruned weights (correctly 50% zero, valid 2:4-along-K
pattern, confirmed programmatically), both as plain `MatMul` and after
`convert_matmul_to_gemm` (value-preserving: `max_abs_diff = 0.0` against the un-converted
model). Built each of the 12 resulting models with `trtexec --fp16
--sparsity={disable,enable,force}` (36 engines) and timed the successful ones (median of
2 rounds; TensorRT's internal timing-based tactic autotuner was still re-run per engine).

**Eligibility** (`enable` mode; `force` ignores actual weight content and is a sanity
check, not a real signal):

| shape | dense (either op) | pruned `MatMul` | pruned `Gemm` |
|---|---|---|---|
| `2d197` (2-D) | 0/12 eligible | **12/12** | 12/12 |
| `3d197` (3-D, batched) | 0/12 eligible | **12/12** | 12/12 |
| `2d2048` (2-D, large) | 0/12 eligible | 11/12 | 12/12 |

**`MatMul` does get sparse-tactic eligibility on TensorRT 10.3** -- at small/medium
shapes, identically to `Gemm`; at the largest shape tested, nearly so (11 vs 12). This
updates the premise behind issue #2271 for current TensorRT: it is no longer categorically
true that `MatMul` never gets N:M sparse math. The one clean exception found: `force`
mode on the 3-D-activation *dense* `MatMul` got 0/12 eligible (vs `Gemm`'s 12/12 via its
reshape scaffold) -- but `force` on dense weights isn't the real-world case either way.

**Latency** (median GPU ms, `enable` mode -- the real-world setting):

| shape | dense | pruned `MatMul` | pruned `Gemm` | pruning speedup |
|---|---|---|---|---|
| `2d197` | 1.33 ms | 1.061 ms | 1.052 ms | ~1.26x, both ops tied (&lt;1% apart) |
| `3d197` | dense `MatMul` 1.34 ms / `Gemm` 1.48 ms | **1.059 ms** | 1.212 ms | `MatMul` 1.27x; converting to `Gemm` is **13% slower**, not faster |
| `2d2048` | ~12.2 ms | **10.17 ms** | 10.84 ms | `MatMul` 1.20x; converting to `Gemm` is **7% slower** |

At every shape tested, `convert_matmul_to_gemm` gave **no latency benefit** -- and at the
two shapes where it isn't a zero-overhead rewrite (`3d197`'s reshape/unflatten scaffold
around the batched activation; `2d2048`, plain 2-D, where the extra overhead is less
obvious), the *converted* engine was measurably **slower** than leaving it as `MatMul`.
Pruning itself is worth it either way (~1.2-1.3x over dense at `fp16`); the conversion
pass is not, at least at these shapes on TRT 10.3.

Numeric correctness: `convert_matmul_to_gemm` is exact (checked in `gen`, fp32); the fp16
engines' relative error vs the fp32 reference stayed in the same range regardless of form
(`~0.1%-1%`), and the pruned+`enable` engines were consistently *more* accurate than their
dense fp16 counterparts (e.g. `2d197`: `1.25e-3` vs `1.05e-2`) -- plausibly because a sparse
kernel accumulates over fewer (only nonzero) terms, though this isn't a guaranteed property
and is incidental to the claim under test.

**Caveat -- tactic-selection nondeterminism**: two independent full `trt`-stage runs gave
the *same* eligibility pattern above both times, but a different `chosen` count for one
config (`2d2048 pruned_matmul enable`: 6/11 chosen in run 1, 11/11 in run 2 -- TensorRT's
timing-based autotuner re-profiles tactics per build and can land on a different one).
The eligibility table is corroborated across both runs; the latency table is from one
full run only (each number's own 2-sample spread was tight, 0.0-1.1%, but a re-build
could plausibly pick different tactics and shift the exact numbers, especially at
`2d2048` where `MatMul`'s eligibility was already partial). Given this, treat the
qualitative result -- pruning helps, conversion doesn't, on TRT 10.3 -- as the reliable
takeaway rather than the exact percentages.

## CUDA feature notebook (`examples/cuda_feature_tests/`): blocked on this board

`scripts/nvidia/run_cuda_feature_notebook.py` runs `cuda_feature_tests.ipynb`'s 7 tests
(Tests A-G: `backend.run_model` CPU/CUDA parity, `simplify(providers=CUDA)` GPU constant
folding, the `(name, options)` device-pinning tuple form, CLI `--cuda`, the
unavailable-provider `ValueError`, DLPack zero-copy with a CUDA `torch.Tensor`, and
`measure_accuracy_drop(providers=CUDA)`) as a plain script, so they can run outside
Jupyter/Colab -- the notebook is explicitly hand-run-only and has never executed on real
hardware.

```sh
python3.12 scripts/nvidia/run_cuda_feature_notebook.py
```

**Could not actually run any of the 7 tests on this board.** onnxsim requires Python >=
3.11 (its wheel is `cp312-abi3`), but the only real GPU-capable `onnxruntime` for JetPack
6/CUDA 12.6 -- the Jetson AI Lab index
(`--index-url https://pypi.jetson-ai-lab.io/jp6/cu126`) -- ships `onnxruntime-gpu` (1.24.0)
for `cp310` only (confirmed with `uv pip install --dry-run`, not by guessing: it resolves
cleanly against `/usr/bin/python3.10` and fails ABI resolution against 3.12). No single
interpreter on this board can import both `onnxsim` and a working GPU `onnxruntime`.

PyPI's plain `onnxruntime-gpu==1.30.0` does have a `cp312`/aarch64 wheel and installs
without error, so it is tempting to reach for as a workaround -- but it requires CUDA
13.x/cuDNN 9.x, and this board runs CUDA 12.6. Concretely reproduced (not just inferred
from the version requirement): `rt.get_available_providers()` lists
`CUDAExecutionProvider` regardless -- that check is static metadata, not a real capability
probe -- but creating an `InferenceSession` with it requested fails to `dlopen
libcublasLt.so.13` and **silently falls back to `CPUExecutionProvider`**, with no
exception, only a stderr warning:

```
Failed to load library .../libonnxruntime_providers_cuda.so with error:
  libcublasLt.so.13: cannot open shared object file: No such file or directory
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*.
```

**This is worth a maintainer's attention beyond this board's mismatch**: onnxsim's own
provider validation (`onnxsim/backend.py:178`, `available = set(rt.get_available_providers())`)
checks exactly the same static list that just lied above. So `onnxsim.simplify(providers=
["CUDAExecutionProvider"])` or `backend.run_model(..., providers=CUDA)` on a
version-mismatched `onnxruntime-gpu` install raises nothing and silently returns a
CPU-computed result -- indistinguishable from a real GPU run to the caller, including
Test A's "CPU vs CUDA parity" check, which would trivially pass either way (both sides
would be CPU). Not fixed here (a behavior change to a core runtime path deserves its own
review, not a bundled verification-script PR); the fix would compare each requested
provider against the *session's actual* `sess.get_providers()` after construction (which
does reflect real fallback) rather than trusting `get_available_providers()` alone, and
warn or raise on mismatch.

Not attempted: building `onnxruntime-gpu` from source for `cp312`/CUDA 12.6/sm_87 (a
multi-hour build disproportionate to this check). The script itself is unaffected by any
of this and should work as-is on a board where a GPU `onnxruntime` matching onnxsim's
Python floor actually exists (e.g. an x86 box with `onnxruntime-gpu`, or a future JetPack
release on CUDA 13).

## Decoder LLM: Qwen2.5-0.5B-Instruct (KV cache, RoPE, GQA, RMSNorm)

`scripts/nvidia/llm_pipeline.py` checks onnxsim on a real decoder-with-KV-cache export --
a different shape of graph than any CNN/ViT above: RoPE and RMSNorm hand-decomposed into
primitive ops, grouped-query attention (14 query heads sharing 2 KV heads), and 48
`past_key_values.N.{key,value}` inputs / 48 `present.N.{key,value}` outputs alongside
`input_ids`/`attention_mask`/`position_ids`. Model:
[`onnx-community/Qwen2.5-0.5B-Instruct`](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)
`onnx/model_fp16.onnx` (opset 14, IR 10, 2759 nodes, 24 layers, head_dim 64; no `If` node --
`past_key_values` are required inputs, a length-0 cache standing in for prefill).

```sh
python3.12 scripts/nvidia/llm_pipeline.py model_fp16.onnx /tmp/llm --seq 1 --past 31   # decode step
```

Pinning `batch_size`/`sequence_length`/`past_sequence_length` to concrete values (batch 1,
1 new token, 31 cached) and running `simplify()`:

**2759 -> 1703 nodes (-38%)**, reproduced across two independent runs (bit-identical node
counts and, per ORT CPU, bit-identical logits: `max_abs_diff = 0.0` between the raw and
simplified fixed-shape models, argmax token matches). `RMSNorm`'s decomposition (`Pow`/
`ReduceMean`/`Sqrt`/`Div`, 49 each) and RoPE's (`Neg`, 48) are **untouched** -- no fusion
pass recognizes either pattern here -- so the whole reduction comes from constant-folding
now-static shape/dtype bookkeeping (`Concat` 267 -> 96, `Expand` 54 -> 48, `Cast` 99 -> 98):
once every KV-cache dimension is a fixed number instead of a symbolic
`past_sequence_length`, the `Shape`/`Gather`/`Range`/`Where`/`Concat` chains that computed
those dimensions at graph-run time become foldable constants outright. ORT CPU session
load+run was also faster on the simplified model (1.9s vs 3.1s, one sample, not a
controlled benchmark).

**Could not complete a TensorRT engine build for either variant on this board.**
TensorRT compiles this attention pattern as a single large fused ("Myelin") subgraph
rather than decomposable layers, and its constant-weight staging buffer needs
**physically contiguous** GPU memory: the raw model requested a 988 MB contiguous
allocation, the *simplified* model (fewer nodes, same weights) needed only 272 MB at
first -- a real ~3.6x reduction from `simplify()`:

```
NvMapMemAllocInternalTagged: ... error 12
Error Code 1: Cuda Runtime (out of memory)
Requested amount of GPU memory (272573440 bytes) could not be allocated.
```

The board's CMA (contiguous memory allocator) pool -- `CmaTotal` in `/proc/meminfo` --
was originally capped at **256 MB**, well under 272 MB. Raising it (`cma=` in
`/boot/extlinux/extlinux.conf`, needs `sudo` + reboot) turned out to be more involved
than a size fix, and **did not unblock the build**:

- `cma=1024M` **failed to reserve at boot** (`dmesg`: `cma: Failed to reserve 1024 MiB`)
  -- this board's physical memory layout has other fixed carveouts (framebuffer, VPR,
  camera debug, PVA, ...) that a 1 GB contiguous request can't fit around -- leaving CMA
  at **0 MB**, worse than the default.
- `cma=512M` **did** reserve cleanly (`dmesg`: `cma: Reserved 512 MiB`, confirmed in
  `/proc/meminfo`) -- but with CMA actually available, TensorRT's tactic autotuner
  stopped picking the 272 MB strategy at all and consistently requested **988 MB**
  instead (matching the raw model's original request almost exactly), which still
  exceeds the 512 MB pool. `--noBuilderCache --noCompilationCache`,
  `--builderOptimizationLevel=0/1/3`, and dropping the page cache before the build made
  no difference -- this looks like a genuine TensorRT tactic-selection interaction with
  CMA availability (a bigger contiguous pool makes a more memory-hungry fusion strategy
  look viable to the cost model, which then doesn't fit either), not a simple
  size-threshold problem. Not investigated further (would need TensorRT internals access
  this write-up doesn't have); a CMA size between 256 MB and 512 MB, never tried, might
  keep the cheaper tactic while still fitting it, but that needs another reboot per
  attempt and was not pursued past this point.

CMA was left at 512 MB (a reasonable general-purpose increase for future GPU work on
this board) rather than reverted. A dynamic-INT8 variant of the same model
(`model_int8.onnx`, `MatMulInteger`/`DynamicQuantizeLinear`) was fetched as a
smaller-weights workaround attempt but not pursued: it is a substantially different graph
shape (dynamic per-token activation quantization) that risks confounding TensorRT parser
compatibility with the actual question, for uncertain payoff given the same ceiling.

**Takeaway**: on hardware with enough contiguous GPU memory, this would be worth finishing
(TensorRT build/latency/correctness comparison, matching the CNN/ViT sections above).
Here, the graph-level finding stands on its own: `simplify()` is exact and shrinks a real
decoder LLM export by more than a third, entirely by folding now-static KV-cache shape
arithmetic, without touching (or needing to touch) RoPE/RMSNorm/GQA at all.

## Splitting the decoder LLM into TensorRT-buildable blocks

Following on from the CMA finding above, `scripts/nvidia/llm_block_split.py` tests the
actual fix: split the ONNX graph into `N`-layer blocks *before* it reaches the builder, so
no single TensorRT network ever needs to build the whole 24-layer fused subgraph at once.

```sh
python3.12 scripts/nvidia/llm_block_split.py split model.sim.onnx /tmp/blocks \
    --block-sizes 1 2 3 4 6 8 12 24
python3.10 scripts/nvidia/llm_block_split.py build /tmp/blocks --seq 1 --past 31
```

`split` (onnxsim interpreter) extracts each block as a standalone ONNX model via
`onnx.utils.extract_model`, run in its own subprocess per block (~37 extractions across
all block sizes; each one reloads and shape-infers the full ~1 GB source model, which
leaks enough memory across repeated in-process calls to OOM otherwise). `build` (TensorRT
interpreter) builds every block with `trtexec --fp16`, then *chains* each buildable size's
engines together -- also one subprocess per size, for the same reason: loading every
block's engine keeps all of them resident simultaneously, and that alone exceeded
available memory for larger blocks even though each one built fine on its own (confirmed
directly: k=6's chain crashed the whole sweep before this was isolated).

**Every block count from K=2 to K=12 builds and chains reliably.** K=1 (24 single-layer
engines) is reliable too, just slowest. **K=24 (unsplit) is fastest when it works, but is
not reliable**: its single Myelin subgraph sits right at this board's memory ceiling, so it
succeeds or fails depending on transient memory state and which tactic TensorRT's
autotuner happens to pick -- observed directly, same config: OOM on one attempt (right
after the earlier whole-sweep crash left memory fragmented), three clean successes
immediately after (on a cleaner boot). This matches the tactic nondeterminism already
documented in the 2:4-sparsity section above.

Real end-to-end decode-step latency (chained engines, 1 new token / 31 cached, mean of 20
iterations after 5 warmup; I/O buffers set up once per engine and reused, not
malloc'd/queried fresh every call -- an earlier version of this measurement did that and
its numbers were dominated by setup overhead, not real compute, visible as per-block
timings that didn't remotely sum to the measured total):

| K (layers/block) | engines | latency (ms) | vs K=1 | max\|logit diff\| vs K=1 | argmax match |
|---|---|---|---|---|---|
| 1 | 24 | 64.3 | 1.00x | ref | ref |
| 2 | 12 | 56.2 | 1.14x | 2.60 | yes |
| 3 | 8 | 53.2 | 1.21x | 2.08 | yes |
| 4 | 6 | 51.0 | 1.26x | 2.88 | yes |
| 6 | 4 | 46.9 | 1.37x | 2.15 | yes |
| 8 | 3 | 38.6 | 1.67x | 2.34 | yes |
| 12 | 2 | 37.1 | 1.73x | 2.12 | yes |
| 24 | 1 | ~35.4 | 1.82x | 2.03 | yes |

Latency drops smoothly and monotonically as K grows -- more fused compute per engine,
less inter-engine H2D/D2H round-tripping -- with clearly diminishing returns past K=8
(38.6 -> 37.1 -> 35.4 ms for K=8 -> 12 -> 24, versus 64.3 -> 56.2 -> 53.2 ms for K=1 -> 2 ->
3). **Every K's output has the identical argmax (top predicted token) to K=1's**, so this
is fully correct for greedy decoding. Raw logit magnitudes differ by ~2-3 absolute
(~13-19% of the largest logit, ~15.4) between K values -- a real, exactly-reproducible
(confirmed bit-for-bit identical across independent runs and across a mid-project timing
methodology fix) fp16 accumulation difference from different fusion-boundary rounding
order, not a bug or run-to-run noise. Whether that magnitude matters depends on the
downstream use: irrelevant for greedy/argmax decoding, potentially relevant for anything
reading exact logit values (temperature sampling, distillation, calibration).

**Practical takeaway for this board**: K=6 to K=12 (4-6 blocks) is the sweet spot --
within ~1.3-1.7x of the theoretical (unreliable) single-engine latency, while building and
loading reliably every time, unlike K=24. This is a genuine, real-hardware answer to "can
a small decoder LLM be deployed via plain ONNX-import TensorRT on an 8 GB Jetson Orin Nano
at all" -- yes, with this splitting, even though the unsplit graph cannot.

## TensorRT-LLM and TensorRT Edge-LLM: ONNX status and onnxsim fusion

Checked 2026-09-23 against TensorRT-LLM `main` (1.3.0rc28) / 1.2.1 and TensorRT Edge-LLM
0.10.1 (`e8b2952`), on an x86 RTX 5050 (sm_120, driver 615.71) host.

**TensorRT-LLM has no ONNX path for onnxsim to plug into.** It builds models from PyTorch
module definitions, not ONNX. The legacy TensorRT-engine backend was removed from `main` in
July 2026 -- Python modules in
[NVIDIA/TensorRT-LLM#15918](https://github.com/NVIDIA/TensorRT-LLM/pull/15918), C++ modules
and every plugin (`GPTAttention` included) in
[#16369](https://github.com/NVIDIA/TensorRT-LLM/pull/16369) -- so 1.3 no longer links
TensorRT at all and runs on the PyTorch backend only. 1.2.1, the last stable release with
the TensorRT path, touches ONNX in exactly two places: `network.py`/`tools/onnx_utils.py`'s
`to_onnx` (a weightless "ONNX-like" *visualization dump* of a TensorRT network, not a
runnable model) and `tools/multimodal_builder.py`, which exports ~20 VLMs' vision encoders
with TorchScript `torch.onnx.export(opset_version=17)` and builds them with TensorRT's ONNX
parser -- the only real ONNX -> TensorRT route, and deleted from `main` by #15918.

TensorRT-LLM 1.2.1 (pip wheel: torch 2.9.1+cu128, TensorRT 10.14.1) **runs on the RTX 5050
(sm_120)** via the PyTorch backend, after two workarounds:

- The `LLM` API spawns its worker with `MPI_Comm_spawn`, which fails under the pip
  `openmpi` wheel (`OPAL ERROR ... dpm.c`, then a hang). `TLLM_WORKER_USE_SINGLE_PROCESS=1`
  runs a TP=1 worker in-process instead.
- It then segfaults in `nvmlSystemGetConfComputeSettings`: `tensorrt_llm/_utils.py` passes
  the ctypes struct *by value* where NVML takes a pointer (undefined behavior that crashes
  on driver 615.71). Fixed on `main` (`byref(cc_settings)`); a 1.2.1 backport is requested
  in issue [#18816](https://github.com/NVIDIA/TensorRT-LLM/issues/18816). Patching the one call in the
  installed `_utils.py` the same way fixes it.

`Qwen/Qwen3-0.6B` fp16, same 3 chat prompts x 128 greedy tokens as the Edge-LLM run below:
176-195 tok/s end to end through the Python `LLM` API (Edge-LLM's C++ runtime: 212 tok/s
wall-clock), 2.5 GB peak host RAM. Outputs match Edge-LLM's for the first 117-237
characters (one of the three identically in full) before fp16 kernel differences diverge
them.

**TensorRT Edge-LLM is ONNX-first**: HF checkpoint -> `tensorrt-edgellm-export`
(`torch.onnx.export(dynamo=True, optimize=True)`) -> ONNX with custom-domain plugin nodes ->
C++ `llm_build` (TensorRT ONNX parser + its plugin library) -> C++ runtime. x86 sm_120 is a
"Developer" platform in its support matrix; no prebuilt sm_120 CuTe DSL kernels ship, but
`kernelSrcs/build_cutedsl.py --kernels fmha --gpu_arch sm_120` generates them in seconds.

`Qwen/Qwen3-0.6B`, exported on CPU in fp16 (opset 24, 856 nodes): attention, RoPE, QK-norm
and the paged-KV-cache update are all inside 28 `trt_edgellm::AttentionPlugin` nodes (with
empty-string optional inputs); what is left is 57 decomposed RMSNorms and 28 SwiGLU MLPs.

- onnxsim handles the custom-domain plugin nodes fine (kept verbatim, empty optional
  inputs preserved, constants around them folded), but **before this change it was a
  no-op: 856 -> 856 nodes**. Every norm is HF's fp32-upcast spelling
  (`Cast(X, FLOAT) -> Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul -> Cast(fp16) -> Mul(weight)`),
  which `fuse_rms_norm` explicitly declined.
- `fuse_rms_norm` now matches that spelling too, as
  `RMSNormalization<stash_type=FLOAT>(X, weight)` -- exactly the op's own reference body
  (Cast X to the stash type, normalize, Cast back to T, multiply by scale). **856 -> 400
  nodes**: all 57 norms fused, 114 of 115 `Cast`s gone, 2.5 s, 5.7 GB peak RSS.
- Fixed along the way: `fuse_rms_norm` accepted a `ReduceMean` over *any* single axis, but
  `RMSNormalization` normalizes over every axis from `axis` to the last, so e.g.
  `ReduceMean(axes=[1])` on a rank-3 tensor fused into a reduction over axes 1 *and* 2
  (onnxruntime: max |diff| 0.94 vs the decomposed graph on unit-scale input). It now only
  fuses a last-axis reduction.

**Real TensorRT, RTX 5050 (sm_120), TensorRT 11.3.0, CUDA 13.4: the fusion is exact and
speed-neutral.** TensorRT's Myelin compiler already fuses the *decomposed* fp32-upcast norm
into one kernel (`__myl_CastMulMeanAddSqrtDivMulCastMul`), so both forms produce the same
engine. Edge-LLM was built from source for sm_120 only (`CMAKE_CUDA_ARCHITECTURES=120`,
`-DCUTE_DSL_ARTIFACT_TAG=sm_120 -DENABLE_CUTE_DSL=fmha`); its `llm_build` accepts the
onnxsim output as-is (standard `RMSNormalization`, plugin nodes byte-identical).

```sh
python3.12 scripts/nvidia/edgellm_simplify.py EXPORT/llm SIM/llm        # onnxsim venv
./build/examples/llm/llm_build --onnxDir SIM/llm --engineDir ENG --maxBatchSize 1 \
    --maxInputLen 512 --maxKVCacheCapacity 1024
python3.12 scripts/nvidia/edgellm_simplify.py rms-stack /tmp/rms       # synthetic, trtexec
```

| Qwen3-0.6B fp16, Edge-LLM `llm_build` + `llm_inference` | as exported | onnxsim |
|---|---|---|
| ONNX nodes | 856 | 400 |
| engine build | 26.5 s, 4.7 GB peak | 25.8 s, 5.2 GB peak |
| prefill | 2821 tok/s (9.22 ms) | 2854 tok/s (9.11 ms) |
| decode | 218.0 tok/s | 218.9 tok/s |
| greedy output, 3 prompts x 128 tokens | -- | identical |

One run each (~1% differences are noise). Both builds print the same 31 TensorRT warnings
(28 of them `Attribute xqa_jit_kernels not found in plugin node`, from the export itself,
not onnxsim). The plugin-free `rms-stack` model (28 Qwen3-0.6B-shaped RMSNorm + SwiGLU
blocks, `trtexec` with CUDA graphs) isolates the norm: 113 engine layers either way, decode
1.979 vs 1.979 ms, 512-token prefill 13.71 vs 13.78 ms (decomposed vs fused, median of 500).

So for Edge-LLM on TensorRT the fusion is a graph-size/readability win, not a speed win.
Any speed benefit would have to come from a backend without a Myelin-style fuser of its
own that does dispatch `RMSNormalization` to a fused kernel -- not measured here.

### Edge-LLM quantized exports: fp8, int4_awq, nvfp4, mxfp8, int8_sq

Every backbone quantization Edge-LLM 0.10.1 offers, on `Qwen/Qwen3-0.6B`, RTX 5050 (sm_120):
`tensorrt-edgellm-quantize llm --quantization <q> --text_dataset wikitext --num_samples 128`
(ModelOpt calibration on the GPU, 19 s-2 min, <= 6.1 GB host RAM) -> `tensorrt-edgellm-export`
(CPU) -> `edgellm_simplify.py` -> `llm_build` / `llm_inference`, same 3 prompts x 128 greedy
tokens as the fp16 run above.

**onnxsim keeps every quantization scheme intact.** The graphs carry very different
quantization machinery -- standard `QuantizeLinear`/`DequantizeLinear` with FP8 / INT8
initializers (fp8, int8_sq), `trt::TRT_FP4DynamicQuantize` + `trt::DequantizeLinear` with
FLOAT4E2M1 weights (nvfp4), `trt::TRT_MXFP8DynamicQuantize` / `TRT_MXFP8DequantizeLinear`
(mxfp8), `trt_edgellm::Int4GroupwiseGemmPluginV2` + `QkvConcatPlugin` (int4_awq) -- and
in all five no Q/DQ or plugin node is touched, the plugin attributes and graph I/O stay
byte-identical (`edgellm_simplify.py`'s check), and the count of FP8/FP4/INT8/UINT8
initializers is unchanged (no weight `DequantizeLinear` constant-folded into fp16). What
changes is the same as for fp16: the 57 fp32-upcast RMSNorms fuse into `RMSNormalization`
and redundant `Cast`s go (e.g. int4_awq 1136 -> 680 nodes, nvfp4 1752 -> 1296).

| dtype | decode tok/s (orig / onnxsim) | prefill tok/s | peak GPU MB | onnxsim greedy output identical |
|---|---|---|---|---|
| fp16 | 218 / 219 | 2821 / 2854 | 1698 | 3/3 |
| fp8 | 327 / 327 | 4574 / 4537 | 1266 | 3/3 |
| int4_awq | **413 / 412** | 3847 / 3858 | **1076** | 2/3 |
| nvfp4 | 337 / 340 | 4326 / 4322 | 1074 | 3/3 |
| mxfp8 | 253 / 253 | 3320 / 3403 | 1272 | 0/3 |
| int8_sq | 315 / 315 | 2660 / 2836 | 1314 | 3/3 |

Speed and memory are unchanged by onnxsim (single runs; ~1-3% differences are noise).
Where the greedy text differs it diverges late (after 172-298 characters) into equally
fluent text; per-step log-probabilities (`llm_inference --numLogprobs 5`, mxfp8) differ by
0.002-0.06 on average before the split, and two of the three splits are at near-exact ties
in the original (top-1/top-2 margin 0.000 and 0.031) -- rounding-level differences in how
TensorRT fuses the fused vs decomposed RMSNorm feeding the quantizer, amplified by MXFP8's
dynamic block quantization (the largest single-step top-1 difference seen was 0.44). Not
verified against an independent reference, so "equivalent quality" is inferred, not measured.

**int4_awq on x86 needs the `int4_fp16_gemm` CuTe group.** With only `fmha` built (the
quick-start default), `Int4GroupwiseGemmPluginV2` compiles with no kernels
(`#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED`), `llm_build` still succeeds, and inference fails
at the first enqueue (`Custom layer callback ... failed`, `warmup enqueueV3 failed`).
`build_cutedsl.py --kernels fmha,int4_fp16_gemm --gpu_arch sm_120` (the group targets the
Ampere instruction set, so it runs on sm_120) plus `-DENABLE_CUTE_DSL="fmha;int4_fp16_gemm"`
fixes it (tracked in onnxsim/onnxsim#1916).

## ONNX -> TensorRT-LLM via AutoDeploy (`onnxsim.to_torch`)

TensorRT-LLM has no ONNX importer, but AutoDeploy -- its path for arbitrary models -- only
needs a PyTorch module with `forward(input_ids, position_ids) -> logits` from a *model
factory*: it `torch.export`s it, pattern-matches attention / GQA repeat / RoPE / RMSNorm
onto its canonical ops, and inserts TensorRT-LLM's paged-KV-cache attention kernels.
`onnxsim/to_torch.py` supplies that module for an ONNX decoder LLM, and
`scripts/nvidia/trtllm_autodeploy_onnx.py` registers it as an `OnnxModelForCausalLM`
factory:

```sh
python3.12 -c "import onnx, onnxsim; m = onnx.load('model_fp16.onnx'); \
  s, _ = onnxsim.simplify(m, target_opset_version=23, check_n=0); \
  onnx.save(s, 'model_fp16_sim23.onnx', save_as_external_data=True)"
python3.12 scripts/nvidia/trtllm_autodeploy_onnx.py model_fp16_sim23.onnx --tokenizer EXPORT_DIR \
    --compile-backend torch-cudagraph                       # TensorRT-LLM venv, onnxsim --no-deps
```

`onnx_to_torch` interprets the ONNX graph op by op inside `forward`, so `torch.export`
flattens it into plain aten ops. Shape arithmetic (`Shape` -> `Gather`/`Concat`/... ->
`Reshape`/`Expand`) is carried as Python (Sym)ints -- note `torch.SymInt` is *not* an `int`
subclass -- which is what keeps batch and sequence dynamic. Decoder attention
(`MatMul(q, kT) [-> scale] [-> +mask] -> Softmax -> MatMul(., v)`, or opset-23 `Attention`)
becomes causal `F.scaled_dot_product_attention`, `past_key_values.*` inputs are stripped
(`Concat(past, new)` -> `new`), and everything only the mask / `present.*` outputs needed
is dead and never traced. One non-obvious rewrite: old HF `rotary_emb` slices its cos/sin
table to `[:past_len + seq_len]` before indexing it with `position_ids`; with the cache
stripped that slice must become the whole table, or every decode step (seq_len 1, large
positions) reads past the traced end.

Checked on [`onnx-community/Qwen2.5-0.5B-Instruct`](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)
`onnx/model_fp16.onnx` (optimum export, opset 14, 2759 nodes, 24 layers, GQA 14/2, past-KV
inputs), TensorRT-LLM 1.2.1, RTX 5050:

- **Conversion is exact.** On an fp32 copy of the model, prefill logits vs onnxruntime:
  relative L2 4.5e-6, argmax 100%, in both `attention="exact"` (every op, KV cache kept)
  and the default `"sdpa"` mode (cache stripped, all 24 attention blocks recognized).
- **The fp16 export is broken in real fp16, and onnxsim fixes it.** Its RMSNorm is
  decomposed and computed entirely in fp16; the residual stream reaches ~1,700 by layer 3,
  so `Pow(x, 2)` overflows fp16 (max 65,504) and greedy decoding degenerates ("four, four,
  four, ..."), both through AutoDeploy and with the converter alone (so not a cache bug).
  (onnxruntime's CPU provider upcasts internally and does not overflow, but its fp16 path
  is inaccurate here in the other direction: `mean(x^2)` ~ 1e-4 at the embedding hits
  fp16 subnormals, 2.8e-2 relative error in the first RMSNorm vs 3.8e-4 for torch --
  so it is not a usable reference for this model in fp16.) `fuse_rms_norm` fuses the chain
  into `RMSNormalization` with an explicit `stash_type=FLOAT` -- after being taught to read
  fp16 scalar constants, which it previously silently declined on -- and `to_torch` emits
  that as HF's upcast RMSNorm, which AutoDeploy's `match_rmsnorm_pattern` recognizes (it
  matched 0 of the fp16 decompositions). `simplify(target_opset_version=23)`: 2759 -> 2461
  nodes, 49 `RMSNormalization`.
- **AutoDeploy then recognizes everything**: 24 attention (-> cached attention), 48 GQA
  repeats, 24 RoPE (-> its optimized RoPE), 49 RMSNorm (-> fused RMSNorm). Output is
  coherent and matches the converter's own no-cache greedy decode.

| Qwen2.5-0.5B-Instruct fp16 ONNX, 3 chat prompts, greedy, batch 1 | build | end-to-end tok/s |
|---|---|---|
| AutoDeploy `torch-simple` | 37 s | 130-133 |
| AutoDeploy `torch-cudagraph` | 25 s | 181-183 |

(For scale: TensorRT-LLM's own PyTorch backend on the HF `Qwen3-0.6B` checkpoint gave
176-195 tok/s in the section above -- a different, slightly larger model.)
`torch-cudagraph` needs `cuda_graph_batch_sizes` capped at `max_batch_size` (the script
does this): AutoDeploy 1.2.1's default capture list includes larger batch sizes and fails
with `Data too large for buffer 'cu_seqlen'`. `max_batch_size` must be >= 2: AutoDeploy
traces with that batch size and `torch.export` specializes a size-1 example dimension.

### More models and export formats: ONNX Runtime contrib ops and int4

Two of the three most common ways LLMs are shipped as ONNX turned out to be the *ONNX
Runtime GenAI builder* spelling, not optimum's: `com.microsoft` `GroupQueryAttention`
(q/k/v in `[B, S, H*D]`, past KV fed straight in, `seqlens_k`/`total_sequence_length`
derived from `attention_mask`), contrib `RotaryEmbedding` (cos/sin cache looked up by
`position_ids`), `SimplifiedLayerNormalization` / `SkipSimplifiedLayerNormalization`
(RMSNorm, and residual-add + RMSNorm), and for quantized files `MatMulNBits`. `to_torch`
now converts all of them: GQA -> causal SDPA with HF's `repeat_kv` spelling (past and
seqlens inputs are never read in `sdpa` mode), RMSNorms in HF's upcast form, and
`GroupQueryAttention` with `do_rotary=1` (Llama-3.2's export: RoPE inside the op,
positions derived from `seqlens_k`, and no `position_ids` graph input) takes the positions
from `forward`'s `position_ids` -- which AutoDeploy supplies -- as a synthetic input, and
`MatMulNBits` *dequantized once at conversion* to a dense fp16 `[K, N]` weight -- a
working path onto TensorRT-LLM, not its int4 kernels (no int4 memory saving; the packed
weights are not kept). `GroupQueryAttention` with sliding window or softcap,
interleaved / scaled contrib RoPE and `MatMulNBits` with `g_idx` raise instead of guessing.

All three checked against onnxruntime on an fp32 copy (prefill logits, `exact` and `sdpa`
modes), then run through AutoDeploy (`torch-cudagraph`, 3 chat prompts, greedy, RTX 5050):

| model (export) | vs onnxruntime | AutoDeploy matches | end-to-end tok/s |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-360M-Instruct` `model_fp16.onnx` (contrib ops) | rel 5.6e-6, argmax 100% | 32 attn, 64 GQA repeat, 65 RMSNorm | 216-220 |
| same, `model_q4f16.onnx` (int4 `MatMulNBits` x224) | rel 8.2e-6, argmax 100% | same | 215-220 |
| `onnx-community/Qwen3-0.6B-ONNX` `model_fp16.onnx` (contrib ops, per-head q/k-norm) | -- | 28 attn, 56 GQA repeat, 113 RMSNorm | 132 |
| `onnx-community/Llama-3.2-1B-Instruct-ONNX` `model_fp16.onnx` (RoPE *inside* GQA, no `position_ids` input) | rel 2.5e-6, argmax 100% | 16 attn, 32 GQA repeat, 33 RMSNorm | 100-101 |

All outputs are coherent (the int4 SmolLM2's answers differ from fp16's, as expected).
onnxruntime's own `GroupQueryAttention` kernel has restrictions the conversion does not
(head size a multiple of 8, of 16 with `do_rotary`; batch 1 when a multi-token input has a
past), which only matters for the tests' reference runs. The speeds above are from before
the two fixes below; updated numbers follow them.

**Why AutoDeploy did not fuse the converted RoPE (fixed).** Its `match_rope_pattern` is
one pattern over a q *and* k rotation that share one `cos.unsqueeze(1)` /
`sin.unsqueeze(1)` node, registered only for the `[B, N, S, D]` / `unsqueeze_dim=1`
layout. Three things had to hold, and the conversion violated each in turn: (1) q and k
each have their own ONNX `RotaryEmbedding` node, so they must be handed the *same*
unsqueezed cos/sin; (2) that unsqueeze is *inside* the pattern, so it must be used by
exactly one q/k pair -- memoizing it across the whole forward call shared it with every
layer, and the matcher refuses a replacement whose internal node has outside users;
(3) the replacement `torch_rope(q, k, cos, sin)` is inserted where q's rotation starts, so
k's `reshape`/`transpose` must already exist there -- i.e. q and k are prepared together
(the converter now rotates each q/k `RotaryEmbedding` pair, found through the
`GroupQueryAttention` consuming both, at whichever node comes first). With all three,
every model's RoPE matches (SmolLM2 32/32, Qwen3 28/28, Llama-3.2 16/16 -- the in-op GQA
rotary only needed (2)).

**Why Qwen3 was 22% slower than the same model from its HF checkpoint (fixed).** Running
AutoDeploy on the HF checkpoint itself (same prompts, same settings) gave 176 tok/s vs
137 for the ONNX -- so the gap was the conversion, not AutoDeploy. An `nsys` kernel
summary of one 128-token generation (`torch-simple`, so kernels are visible): 898 ms of
GPU kernels vs 693 ms, and the whole difference was one extra GEMV per layer (3,556
launches = 28 layers x 127 steps, 180 ms, ~50 us each, plus a `cublasLt::splitKreduce`
per launch). Every matmul was plain fp16 x fp16 at the aten level; the cause was weight
**layout**: ONNX `MatMul` stores `[K, N]` weights (`x @ W`), PyTorch linears `[N, K]`
(`F.linear(x, W)`), and for batch-1 decode cuBLAS sent one of the seven per-layer
`[K, N]` matmuls to a slow split-K GEMV. (A first guess -- the tied LM head computed as
`MatMul(x, Transpose(embed))`, 311 MB transposed every step -- was tested by
pre-transposing it in the ONNX and made no difference: the transpose is a view.) The
converter now registers a constant 2-D `MatMul` weight transposed and emits `F.linear`
(also AutoDeploy's canonical linear); `MatMulNBits` dequantizes straight to `[N, K]`.
Numerics are unchanged (all four models still match onnxruntime at 2.5e-6 to 8.2e-6).

| AutoDeploy `torch-cudagraph`, RTX 5050, 3 chat prompts x 128 greedy tokens | before | after both fixes |
|---|---|---|
| `onnx-community/Qwen3-0.6B-ONNX` fp16 | 132 | **173-175** (HF checkpoint through AutoDeploy: 176) |
| `onnx-community/Qwen2.5-0.5B-Instruct` fp16 (optimum, via `simplify(opset 23)`) | 181-183 | **196-207** |
| `onnx-community/Llama-3.2-1B-Instruct-ONNX` fp16 | 100-101 | **106-109** |
| `HuggingFaceTB/SmolLM2-360M-Instruct` fp16 | 216-220 | 208-231 (3 runs; noise ~+-5%) |
| same, `model_q4f16.onnx` (int4, dequantized) | 215-220 | 202-227 |

### int4 that stays int4: `matmul_nbits="packed"` and a Triton W4A16 kernel

Dequantizing `MatMulNBits` once to fp16 works but throws away what int4 is for. Real int4
execution on this stack turned out to need a kernel of our own:

- **TensorRT-LLM's weight-only int4 GEMMs refuse sm_120.** `finegrained_mixed_dtype_gemm`
  (what its PyTorch backend's W4A16 AWQ linear calls) and `weight_only_quant_gemm` both fail
  with `Not Implemented: SM120 GEMM only supports nvfp4` (`cutlass_heuristic.cpp`) -- the
  consumer-Blackwell build ships NVFP4 GEMMs only on that path.
- **AutoDeploy has no int4 kernel at all**, in 1.2.1 or on `main`: its AWQ and GPTQ int4 ops
  (`torch_fake_quant_int4_linear`, `..._gptq_linear`) are *fake-quant* -- they dequantize
  the whole weight in PyTorch on every call; only FP8 / NVFP4 get `fuse_*_linear` onto real
  kernels.
- PyTorch's `aten._weight_int4pack_mm` (tinygemm) runs on sm_120 but takes **bf16**
  activations only; for these fp16 models that would round every int4 linear's input to bf16.

So `onnx_to_torch(..., matmul_nbits="packed")` keeps 4-bit `MatMulNBits` weights in their own
layout (`[N, K/2]` uint8, `[N, groups]` scales, unpacked zeros) and emits
`torch.ops.onnxsim.matmul_nbits` (`onnxsim/_matmul_nbits_op.py`): a `torch.library` custom op
with a fake implementation (AutoDeploy exports it as one opaque node, CUDA graphs capture it),
a PyTorch dequantize-and-`F.linear` fallback, and on CUDA the Triton kernels in
`onnxsim/_triton_w4a16.py` -- a split-K W4A16 kernel that dequantizes in registers for
M <= 16 (decode), and a GPU dequantize + cuBLAS for larger M (prefill). Same arithmetic as
the dequantize path (`(q - zp) * s` rounded to fp16, fp32 accumulation), different
summation order: Llama-3.2-1B q4f16 logits agree at rel 2.4e-3 (max |diff| 0.047 of 23.7,
argmax 100% on an 8-token decode-path input) -- enough to flip greedy near-ties (top-1/top-2
margins of 0.016 occur), so the two modes' generations can diverge after a few tokens.

**Verified end to end against ONNX Runtime.** Prefill logits of the real int4 files on the
GPU (fp16) vs ONNX Runtime on an fp32 copy of the same file (int4 weights exact):
SmolLM2-360M q4f16 3.5e-3 relative (decode-sized input, Triton split-K kernel) / 2.7e-3
(54-token prefill), Llama-3.2-1B q4f16 2.4e-3 / 2.7e-3, argmax 100% -- the same as the
`dequant` path and as a plain fp16 model (2.4e-3), i.e. fp16 rounding. One trap: Llama's
`MatMulNBits` carry `accuracy_level=4`, which lets ONNX Runtime's CPU kernel quantize the
*activations* to int8; against that reference both paths look 10x worse (3.2e-2, argmax 97%).
The conversion ignores `accuracy_level` and always uses full-precision activations.

**Kernel speed: measure with weights that don't fit in L2.** Timing one layer in a loop
keeps its dense fp16 weight resident in the RTX 5050's 24 MB L2 (a 2048x2048 fp16 weight is
8 MB), which made dense look faster than int4 for every layer up to ~3072 wide at M = 2-16.
In a model the weights stream from DRAM every token (Llama-3.2-1B: 2.5 GB), so the table
below rotates each layer over enough copies to exceed 4x L2 (also: 2 s clock warm-up,
7 interleaved rounds, medians, CUDA-graph replay; group 32):

| layer K x N | M=1 | M=8 | M=32 | M=64 | M=128 | M=512 |
|---|---|---|---|---|---|---|
| 960 x 960 | 2.52x | 1.11x | 0.82x | 0.60x | 0.76x | 0.91x |
| 2048 x 2048 | 1.64x | 1.53x | 1.31x | 0.82x | 0.71x | 0.93x |
| 2048 x 8192 | 1.90x | 1.86x | 1.50x | 1.11x | 0.82x | 0.82x |
| 8192 x 2048 | 1.69x | 1.64x | 1.33x | 0.85x | 0.61x | 0.82x |

(int4 speedup over dense fp16 `F.linear`.) M <= 16 is the split-K decode kernel; 16 < M <= 256
the same kernel as a plain tile GEMM (tuned from a sweep: it beat the first version's
transient-dequant + cuBLAS path at every M <= 128, e.g. 73 vs 274 us at M = 32 on
2048x8192); M > 256 dequantizes to a transient fp16 weight and calls cuBLAS. Decode --
nearly all of a chat generation -- is 1.1-2.5x faster; mid-size prefills are the weak spot.

| AutoDeploy `torch-cudagraph`, RTX 5050, 3 prompts x 128 greedy tokens | weights on GPU | tok/s |
|---|---|---|
| Llama-3.2-1B `model_fp16.onnx` | ~2.5 GB fp16 | 106-109 |
| Llama-3.2-1B `model_q4f16.onnx`, `matmul_nbits="dequant"` | ~2.5 GB fp16 | 108-109 |
| Llama-3.2-1B `model_q4f16.onnx`, **`matmul_nbits="packed"`** | int4 + fp16 embedding | **159-160** |
| SmolLM2-360M `model_q4f16.onnx`, `"dequant"` | 0.725 GB | 218-229 |
| SmolLM2-360M `model_q4f16.onnx`, **`"packed"`** | **0.272 GB** | **347-365** (290-349 before the tile path) |

(`trtllm_autodeploy_onnx.py --matmul-nbits packed`. AutoDeploy's "Estimated parameters
memory" log line counts `parameters()` only and misses the uint8 buffers -- the byte counts
above are from the state dict.) Only 4-bit without `g_idx`, group sizes that divide or are
multiples of 128, and zero points that are absent, packed uint8, or integral floats in
[0, 15] take the kernel; anything else (e.g. fractional float zero points, which
MatMulNBits allows) falls back to dequantizing.
