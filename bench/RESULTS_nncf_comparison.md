# onnxsim vs. NNCF: INT4 weight-only compression

Measured with `bench/nncf_comparison.py`. This answers candidate future-work
item 3 in `docs/nncf-comparison-future-work.md`, which noted that the
onnxsim-vs-NNCF comparison recorded there was architectural and that "no
benchmark in this repo actually runs the same model through both onnxsim and
NNCF."

**Environment**: NNCF 3.3.0, onnxruntime 1.29.0 (CPUExecutionProvider), onnx
1.20.0, numpy 2.3.5, x86_64, Python 3.11. onnxsim built from this checkout.

**Model**: `bench/nncf_comparison.py`'s synthetic 4-layer MatMul stack,
`in_dim = hidden = 256` (1024 KiB of float32 weights). See "Caveats" — the
synthetic model is the main limitation of these numbers.

## Results

`rel L2` is the quantized model's output error against the float model's on
shared random evaluation data; `KiB` counts only initializers a node actually
references; `dtypes` is the referenced-initializer element-type histogram.

| configuration | rel L2 | KiB | x smaller | quant s | weight dtypes |
|---|---|---|---|---|---|
| onnxsim int4 (RTN) | 0.1953 | 160.1 | 6.40 | 0.03 | FLOATx4, INT4x4 |
| **onnxsim int4 + GPTQ** | **0.1571** | 160.1 | 6.40 | 0.12 | FLOATx4, INT4x4 |
| onnxsim int4 + AWQ | 0.1926 | 164.0 | 6.24 | 0.20 | FLOATx8, INT4x4 |
| onnxsim int4 + AWQ + GPTQ | 0.1650 | 164.0 | 6.24 | 0.36 | FLOATx8, INT4x4 |
| nncf int4_sym (NNCF defaults) | 0.1765 | 167.2 | 6.12 | 0.18 | FLOATx4, INT4x3, UINT8x2 |
| nncf int4_sym all_layers g32 | 0.1698 | 160.0 | 6.40 | 0.07 | FLOATx4, INT4x4 |
| nncf int4_sym g32 + AWQ | 0.1698 | 160.0 | 6.40 | 0.24 | FLOATx4, INT4x4 |
| nncf int4_sym g32 + ScaleEstimation | 0.1765 | 160.0 | 6.40 | 0.16 | FLOATx4, INT4x4 |

At matched settings (`all_layers=True`, `group_size=32`) both tools reach the
same 6.40x compression and emit the same graph shape, so the accuracy column
is the comparison. **onnxsim + GPTQ is the most accurate configuration
measured** (0.1571), ahead of NNCF's best (0.1698). onnxsim's plain
round-to-nearest is the *least* accurate (0.1953) — see finding 1.

## Findings

### 1. onnxsim's INT4 gives up the `-8` code; NNCF does not

Decoding the INT4 initializers each tool emits for the same weight:

```
onnxsim INT4 code range: (-7, 7)
nncf    INT4 code range: (-8, 7)
```

`passes/quantize_matmul_common.h`'s `TryQuantizeWeightBlockwiseInt4InPlace`
uses `scale = max(|w|) / 7` and clamps to `[-7, 7]`, forgoing the
representable `-8`. NNCF uses the full `[-8, 7]`. For identical storage that
makes onnxsim's quantization step about 12.5% coarser, which is enough to
account for the plain-RTN gap in the table (0.1953 vs 0.1698) at byte-identical
size. Whether to change this is a real decision, not an oversight to fix
blindly: a symmetric `[-7, 7]` grid keeps `-x` representable whenever `x` is,
which several onnxsim passes' own reasoning (and `_gptq_quantize_columns`'s
clip bounds) already assume. But the cost is now measured rather than
theoretical.

### 2. GPTQ needs more calibration rows than the reduction dimension

An earlier run of this benchmark reported onnxsim + GPTQ at **0.2384** —
*worse* than plain RTN. That was an artifact of the benchmark, not of GPTQ:
the calibration set had 64 rows for a `K = 256` reduction dimension, leaving
GPTQ's `[K, K]` Hessian rank-deficient. Raising the calibration set above `K`
moved the same configuration to 0.1571, the best result in the table. The
script now sizes calibration data at `2 * in_dim` rows for this reason.

Anyone comparing GPTQ-style methods should check this first; it inverts the
conclusion.

### 3. NNCF quantizes the last layer to INT8 unless told otherwise

NNCF classifies a model's final MatMul as a non-"ratio-defining" layer (the
`lm_head` convention) and leaves it at `int8_asym` per-channel. That is
visible in the `dtypes` column: NNCF's default row is `INT4x3, UINT8x2` on a
4-layer model, versus `INT4x4` with `all_layers=True`. onnxsim's INT4 pass
quantizes every eligible layer unconditionally. Comparing the two defaults
head-to-head would compare different bit widths and misattribute the
difference to quantization quality — the benchmark passes `all_layers=True`
for the matched rows and reports NNCF's default separately.

### 4. onnxsim's INT4 pass silently skips layers with no `value_info`

`weight_only_quantize_int4_matmul.h` requires the activation's element type
(`info.x->elemType() != FLOAT` → no match), and a hand-built ONNX graph has no
`value_info` for intermediate tensors. On the first version of this benchmark
onnxsim quantized **1 of 4** layers (1.27x) against NNCF's 4 of 4 (6.40x) —
which looks like a capability gap and is actually a missing-type-annotation
gap. `onnx.shape_inference.infer_shapes` fixes it, and the script now runs it
on both the synthetic and the user-supplied model. (The same issue produced a
misleading exact tie in `tests/test_gptaq.py`; see PR #1192.) Real exported
models normally carry this `value_info`, but any tooling that hand-builds
graphs will hit it.

## Caveats

- **The model is synthetic.** A plain Gaussian MatMul stack has none of the
  per-channel activation outliers that AWQ and Scale Estimation exist to
  exploit, so both tools' data-aware modes are understated here: NNCF's AWQ
  changed nothing at all (0.1698 → 0.1698) and its Scale Estimation was
  slightly worse (0.1765); onnxsim's AWQ moved 0.1953 → 0.1926. These numbers
  say nothing about how those methods behave on a real transformer, and should
  not be read as ranking AWQ against GPTQ. `bench/nncf_comparison.py` accepts a
  path to a real `.onnx` model for exactly this reason.
- **Accuracy is output relative L2, not a task metric.** No perplexity or
  downstream-accuracy evaluation is involved.
- **Latency is not measured.** At matched settings both tools emit the same
  `DequantizeLinear`-based graph, so a latency number would measure
  onnxruntime's kernel rather than either quantizer.
- **One model, one seed.** These are single-configuration measurements, not a
  sweep; treat the ordering as indicative and re-run on your own model.

## Reproducing

```bash
pip install nncf
python bench/nncf_comparison.py --layers 4 --hidden 256 --in-dim 256
python bench/nncf_comparison.py path/to/real_model.onnx   # far more meaningful
```
