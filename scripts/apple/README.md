# Apple Core ML integration check

Verifies that `onnxsim`'s output still works with **Core ML** — the runtime
behind `coremltools` model deployment on macOS/iOS. The goal is to catch the
failure mode the unit tests and the large-model regression don't: a
simplification that produces a graph Core ML can no longer **compile**, or
that **changes the result** on Apple's stack.

It uses the [`CoreMLExecutionProvider`](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
built into the standard `onnxruntime` PyPI wheel — no extra package, but it
only exists on the **macOS** build (`get_available_providers()` omits it on
Linux/Windows). So the whole check runs on a stock macOS GitHub-hosted
runner with nothing but `pip install onnxruntime`.

## What it checks

For each model the harness runs **original vs. simplified through the same
Core ML backend**, so backend quirks cancel and only an onnxsim-introduced
change can fail the run:

1. `simplify` the model with onnxsim.
2. Compile + run the **original** graph on the Core ML EP.
   If that already fails, the backend just doesn't support the graph →
   reported as `unsupported`, **not** a failure.
3. Compile + run the **simplified** graph on the Core ML EP.
   If the original compiled but the simplified doesn't → `coreml_regression`
   (a failure): simplification broke Core ML compatibility.
4. Compare the two Core ML outputs. Divergence beyond tolerance →
   `coreml_regression`: simplification changed the on-device result.
5. Record the ONNX Runtime CPU-reference diff and the Core ML **coverage**
   (does the whole graph map onto Core ML, or do some nodes fall back to
   ORT's CPU provider) as information.

Partial coverage and `unsupported` are reported, never failed — plenty of
valid graphs are not 100% Core ML-mappable, and that is a backend property,
not an onnxsim bug.

## Files

| file | purpose |
| --- | --- |
| `coreml_backend.py` | wraps the Core ML EP: builds/runs a model on Core ML and on the ORT CPU reference, measures coverage. Degrades gracefully (`COREML_AVAILABLE`) when the EP is absent (non-macOS). |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the small synthetic-graph suite shared with the other EP harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_coreml_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Requires macOS (the platform Core ML itself runs on).

```bash
pip install onnxruntime      # the macOS wheel bundles the Core ML EP
pip install .                # or install an onnxsim wheel

python scripts/apple/run_coreml_compat.py --output coreml-compat.csv
```

The in-tree smoke test `tests/test_coreml_compat.py` reuses this harness and
is skipped automatically when the Core ML EP isn't available (e.g. running on
Linux/Windows).

## Fidelity tiers (what this does and doesn't cover)

This check runs the **real** Core ML compiler and runtime — there is no
emulation step the way QNN's HTP backend needs one, since the check already
runs on real Apple hardware (the macOS CI runner itself). What it leaves
uncovered:

- **Compute unit selection.** `MLComputeUnits=ALL` (the default here) lets
  Core ML place ops on CPU, GPU, or the Neural Engine as it judges best. Set
  `COREML_COMPUTE_UNITS=CPUOnly` (or `CPUAndGPU`, `CPUAndNeuralEngine`) to
  pin a specific target if you need to isolate one.
- **iOS-specific behavior.** This runs the macOS Core ML stack; iOS devices
  share the same compiler but can differ in available ops per OS version.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models can be layered on by passing an on-disk path as
`worker.py`'s second argument, the same way `scripts/qualcomm` and
`scripts/regression` do.

## General staged ONNX deployment benchmark

`benchmark_onnx_pipeline.py` benchmarks staged ONNX deployments without
assuming a particular task or model family. A JSON manifest names each graph,
optional NPZ input feeds, and tensor connections between stages. This covers
pipelines such as backbone/head, encoder/decoder, and detector/preprocessor
graphs while leaving image transforms and task-specific input construction in
the calling application. Unspecified feeds use deterministic values for
static-shape smoke benchmarks; provide NPZ files for representative inputs.

For example, save `image.npz` and `head_inputs.npz` with NumPy and create
`pipeline.json`:

```json
{
  "name": "detector",
  "stages": [
    {"name": "backbone", "model": "backbone.onnx", "feeds": "image.npz"},
    {"name": "head", "model": "head.onnx", "feeds": "head_inputs.npz"}
  ],
  "connections": [
    {"from": ["backbone", "features"], "to": ["head", "features"]}
  ],
  "backends": {"backbone": "coreml", "head": "tinygrad_metal_jit"}
}
```

Run it on macOS with:

```bash
python scripts/apple/benchmark_onnx_pipeline.py pipeline.json \
  --output detector-m4.json --compute-units ALL \
  --compute-precision FLOAT32 --repeats 8
```

Each stage is reported against ONNX Runtime CPU for Core ML, eager tinygrad
Metal, and tinygrad Metal JIT. The staged benchmark defaults Core ML to
`ALL`, allowing its CPU, GPU, or Neural Engine placement logic to choose per
operation; pass `--compute-units CPU_AND_NE` for an ANE-only comparison. The
selected per-stage backends are then timed end to end, including tensor
handoffs. Set
`input_connections` for inputs that are shared by multiple stages but are not
produced by an earlier stage (for example, projection LUTs). Set `fuse` to an object such as
`{"stages": ["backbone", "head"], "backend": "coreml"}` to additionally measure
one merged ONNX graph; the original per-stage and unfused end-to-end results
remain in the report. The fused graph is written beside the JSON output.

Pass `--fallback-to-tinygrad` to keep measuring when a stage's chosen backend
has no working runner -- a case Core ML genuinely cannot lower (RoiAlign's
`average` mode, `NonZero`'s data-dependent output length), where the end-to-end
number would otherwise just be an "unavailable" error. Such a stage is timed on
a tinygrad Metal backend instead, Metal JIT first and eager second, and every
substitution is listed under `backend_fallbacks` in the report:

```json
"backend_fallbacks": [
  {"stage": "head", "requested_backend": "coreml",
   "used_backend": "tinygrad_metal_jit"}
]
```

It is opt-in and per stage: a stage whose selected backend did build is never
swapped, and without the flag the manifest's choice stands unchanged. Note the
fallback only helps where the *other* backend can run the op -- Mask R-CNN's
remainder, for instance, still fails because tinygrad does not implement
`NonMaxSuppression` or `RoiAlign` either.

M4 results for the Hexagon-deployed YOLO11n, YOLO26n, and YOLO26s models are
in [`bench/RESULTS_m4_hexagon_yolo_coreml_metal.md`](../../bench/RESULTS_m4_hexagon_yolo_coreml_metal.md).
The Core ML translator lowers static nearest-neighbor Resize to Core ML's own
`resize_nearest_neighbor` when the resize is a whole-number replication of its
input, and otherwise to constant index gathers, which preserve ONNX's sampling
rule. The gate is deliberate: measured on Core ML, a downscale or a fractional
upscale samples *different* rows than ONNX does, so only the replication case
uses the native kernel, and it requires iOS15/macOS12 or newer. On M4 this took
Fast-BEV M0's encoder from 95.6 ms to 65.9 ms and its end-to-end Core ML chain
from 174.3 ms to 143.9 ms, with ORT parity unchanged (cosine 1.0).

The same gate also speeds up the YOLO detectors below, whose head upsample is an
integer 2x nearest resize (FLOAT16, `ALL`, 8 runs after 3 warm-ups, same input
and machine). A/B against the gather path, with score p99 error vs ONNX Runtime
unchanged in both arms:

| model | gather | native | speedup | score p99 error |
|---|---:|---:|---:|---:|
| YOLO11n | 4.66 ms | **2.74 ms** | 1.70x | 5.0e-5 |
| YOLO26n | 4.80 ms | **2.37 ms** | 2.03x | 8.6e-6 |
| YOLO26s | 9.45 ms | 9.54 ms | 0.99x (no change) | 1.7e-5 |

YOLO26s is unchanged because its Resize input is not a whole-number
replication, so it keeps the gathers; its head is large enough that the two
`Resize` nodes are not on the critical path. Note these FLOAT16 latencies are
lower than the `CPU_AND_NE` FLOAT16 numbers recorded in
[`bench/RESULTS_m4_hexagon_yolo_coreml_metal.md`](../../bench/RESULTS_m4_hexagon_yolo_coreml_metal.md),
which predate the native-Resize change; that file's accuracy notes still apply.

## rustnn WebNN vs. tinygrad benchmark (`benchmark_webnn_tinygrad.py`)

Times small simplified models, per node and as a whole, on rustnn's native WebNN
implementation (`cpu`, which is ONNX Runtime's CPU EP, and `npu`, which is Core ML)
and on tinygrad (`CPU` and `METAL`, at BEAM widths 0 and 2). Each result is checked
against onnx's reference evaluator. The script prints a Markdown table.

    python3 scripts/apple/benchmark_webnn_tinygrad.py --require npu,METAL

It needs `pip install pywebnn onnxruntime tinygrad==0.14.0` on Python 3.12. The
`rustnn-webnn` job in `apple-integration.yml` runs it. See `docs/rustnn.md` for the
device mapping and its caveats. In particular, `npu` means the Core ML path, not a
confirmed placement on the Neural Engine.

## SAM Core ML + tinygrad Metal hybrid benchmark

`benchmark_sam_hybrid.py` measures Core ML and tinygrad Metal on the two
existing SAM pieces (image encoder and prompt decoder). tinygrad's eager
`OnnxRunner` launches a Metal kernel for each op; the benchmark also measures
`TinyJit` replay, which captures the graph and avoids that per-op launch cost.
Each stage and the end-to-end Core ML-encoder/Metal-decoder path are compared
against the ONNX Runtime CPU result. The Core ML input adapter exposes the SAM
raw-pixel uint8 tensor as float32 because Core ML has no uint8 model input.

First export the models as described in
[`scripts/android/vision_models/sam/README.md`](../android/vision_models/sam/README.md),
then run on macOS:

```bash
python scripts/apple/benchmark_sam_hybrid.py "$SAM_WORK/edgesam" \
  --image /path/to/image.jpg --compute-units ALL --compute-precision FLOAT32 \
  --output edgesam-m4.json --repeats 8
```

On an M4 Mac mini, with 8 measured runs after 3 warm-ups, the following
medians include both model predictions and the encoder-to-decoder handoff.
The reference is a single sample image and center-point prompt; see
[`bench/RESULTS_m4_sam_hybrid.md`](../../bench/RESULTS_m4_sam_hybrid.md)
for the full results and accuracy limits.

| model | input | Core ML → Core ML | Core ML → Metal JIT | Hexagon HTP reference |
| --- | ---: | ---: | ---: | ---: |
| EdgeSAM | 1024² | 32.0 ms | 39.5 ms | 92.2 ms |
| MobileSAM | 1024² | 66.6 ms | 67.5 ms | 330.7 ms |
| EfficientViT-SAM-L0 | 512² | 30.8 ms | 36.7 ms | 52.8 ms |

`Core ML → Metal JIT` keeps the heavier image encoder on Core ML and runs the
prompt decoder through tinygrad's captured Metal graph. All results above use
full-precision Core ML compute; default Core ML precision was faster for
EdgeSAM (11.7 ms for both stages) but gave 0.86 thresholded mask IoU on the
sample, so it is not included in the deployment comparison.

## LLM decode benchmark (`export_llm_to_coreml.py` / `run_llm_decode_benchmark.py`)

A separate pair of tools for a different question than the compatibility
check above: not "does Core ML accept this graph", but "how fast does a
causal LM actually decode through `onnxsim.export_coreml`, on-device" --
the same two axes (decode tok/s, peak memory) as
[DeviceMark](https://devicemark.github.io/)'s on-device LLM leaderboard
methodology. DeviceMark's own on-device runtime is a different, private
"Core AI" engine (`aimodel` format), not `CoreML.framework`, and its board
tests a different model roster (Qwen3.5, LFM2.5, Granite-4.0-H, and others --
see `bench/TODO_quality_retention_eval.md`'s "What DeviceMark measures"
section) -- this benchmark answers the same *kind* of question for onnxsim's
own Core ML exporter, not a literal comparison against DeviceMark's own
numbers.

- `export_llm_to_coreml.py` exports a Hugging Face causal LM (via
  [`optimum-onnx`](https://github.com/huggingface/optimum-onnx)) to an ONNX
  decoder-with-past, runs it through `onnxsim.simplify`, and converts it with
  `onnxsim.export_coreml` using its `dynamic_shapes` argument to keep
  `sequence_length` and `past_sequence_length` genuinely dynamic (bounded by
  `--max-context-length`) instead of baking them to fixed values. The result
  is one Core ML model that supports a real, O(1)-per-token growing KV
  cache: a single forward pass over the whole prompt builds the initial
  cache (prefill), and each new token is generated with a single-token
  forward pass that reuses it, instead of reprocessing the whole context
  every step. This exercises onnxsim's Core ML exporter's dynamic-shape
  support (`onnxsim/coreml_export.py`'s `dynamic_shapes` argument) against
  the largest, most control-flow-heavy transformer graph it's been run on.
- `run_llm_decode_benchmark.py` loads the resulting `.mlpackage`, greedily
  decodes a prompt by prefilling once and then decoding one token at a time
  against the growing cache, and reports prefill latency, decode tok/s
  (decode steps only, matching DeviceMark's methodology), and peak RSS. Like
  the rest of this directory, it only *runs* a model on macOS (that's where
  Core ML's runtime lives); the export step itself needs no Apple hardware.

`coreml-integration.yml`'s `benchmark-decode-macos` job runs this exact pair
end-to-end on a macOS GitHub-hosted runner, over a matrix spanning the
smoke-test tier (`HuggingFaceTB/SmolLM2-135M-Instruct`, plus its
`--quantize-weights`/`--matmul-to-conv` variants -- see below) and the
few-billion-parameter tier -- the same rough weight class as DeviceMark's own
leaderboard (roughly 0.8-5B; its own current roster is a different,
non-overlapping model list, see `bench/TODO_quality_retention_eval.md`)
(`HuggingFaceTB/SmolLM2-1.7B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`,
`meta-llama/Llama-3.2-3B-Instruct`, `microsoft/Phi-3.5-mini-instruct`) --
multiple architecture families (Llama-style, Qwen2, Phi-3's fused
`qkv_proj`/`gate_up_proj` projections) so a translator regression specific
to one doesn't hide behind another passing. The `Llama-3.2-*` entries are
gated (need `HF_TOKEN`, same read-only CI secret
`prepare_benchmark_models.py` already uses); `Qwen2.5-3B-Instruct` and
`Llama-3.2-3B-Instruct` previously OOM'd during ONNX export/trace in a
15GB-RAM dev sandbox (see `prepare_benchmark_models.py`'s `BENCHMARK_MODELS`
notes) -- not a translator issue, but untested on the CI runner's own
memory until this matrix actually runs them. Posts each model's numbers to
that run's job summary -- `workflow_dispatch`/schedule-only, like the other
real-model jobs in that workflow, not on every PR.

### Theoretical ceiling

`run_llm_decode_benchmark.py` reports a decode tok/s number, but not what to
expect from it. This section gives a back-of-envelope ceiling to compare a
measured number against, sourced from Manjeet Singh's reverse-engineering of
the M4 Neural Engine ("Inside the M4 ANE, Part 4: The Complete Machine",
[maderix.github.io](https://maderix.github.io/articles/inside-the-m4-ane-part-4/),
Aug 2026) -- the only public source we're aware of with hardware-level,
measured (not marketing) numbers for this chip.

**M4 ANE (H16G, 16 cores), measured:**

| | |
| --- | --- |
| fp16 peak | 19 TFLOPS (18.77 TFLOPS / 98.8% of ceiling measured on a 64-layer conv1x1 chain -- a single isolated matmul only reaches ~30%) |
| W8A8 (packed int8) peak | 38 TOPS (36.01 TOPS measured) |
| Power at fp16 peak | 4.57 W (4.1 TFLOPS/W); 0 mW idle |
| Unified DRAM bandwidth | 120 GB/s, shared with the CPU and GPU |
| Dispatch floor | ~90 µs of host-side (XPC) overhead per ANE program submission, independent of the work submitted |

**Why decode is bandwidth-bound, not compute-bound.** A single greedy decode
step (this benchmark's shape: batch 1, one new token, reusing the KV cache)
does roughly `2 x parameter_count` FLOPs of *compute* -- for
`Qwen/Qwen2.5-1.5B-Instruct`, about 3 GFLOP, ~160 µs at the ANE's 19 TFLOPS
ceiling. But producing that token requires reading essentially the entire
weight set once (nothing amortizes a weight read across tokens at batch
size 1, unlike prefill's one-pass-over-the-whole-prompt or a
multi-sequence-batched server), and moving those bytes is the actual
constraint:

```
decode tok/s ceiling ≈ DRAM bandwidth / weight bytes read per token
                      ≈ 120 GB/s / (model parameter count x bytes per weight)
```

| Model | Params | fp16 weights | fp16 ceiling | int8-weight ceiling |
| --- | --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 270 MB | ~444 tok/s | ~889 tok/s |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 3.0 GB | ~40 tok/s | ~80 tok/s |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | 3.4 GB | ~35 tok/s | ~71 tok/s |

For every model in this table, the bandwidth-bound time per token (µs to
low-ms) is well past both the ~90 µs ANE dispatch floor and the sub-200 µs
compute time -- so at these sizes, dispatch overhead and raw FLOPs are noise
next to weight-streaming time, and the lever that actually moves decode tok/s
is **bytes per weight**, not compute throughput. (The article's own explicit
finding backs the general shape of this: *"For LLM inference, prefill
provides the large matrix operations that suit the ANE. Token-by-token
decode contains smaller operations for which the 90 µs submission cost can
dominate, making CPU/SME execution more suitable"* -- true for a
small-enough model or a system where the ANE isn't reading gigabytes of
weights per step, but bandwidth dominates first at the model sizes in this
suite.)

This is also why W8A8 (packed int8 *compute*, up to 1.95x the fp16 rate per
the article's own measurements) isn't the right lever here: it speeds up the
compute time we've just shown is already negligible, and per the same
article, weight-only int8 with fp16 activations "stays on the fp16 compute
path" -- no compute speedup at all. What halving weight bytes *does* help,
regardless of compute path, is the number in the denominator above:
`--quantize-weights` (below) is aimed squarely at that, not at compute
throughput.

**Caveats:** this ceiling assumes the full 120 GB/s is available to weight
streaming alone (no contention from the KV-cache read/write, other
processes, or `MLComputeUnits=ALL` routing some ops to the GPU/CPU instead,
each with their own bandwidth share); it is an optimistic upper bound, not a
number `run_llm_decode_benchmark.py` should be expected to hit. It is also
specific to the M4 -- later chips (see "The M6: Dual ANE" in the source
article) change these constants.

### Decode parity (`check_decode_parity.py`)

The decode benchmark above measures speed, not correctness -- a model that
runs fast but computes the wrong thing would still produce a number.
`check_decode_parity.py` closes that gap: it greedily generates the same
prompt through **both** the exported `.mlpackage` (via `CoreMLDecoder`, real
Core ML runtime) and the original Hugging Face model (`transformers`,
CPU-only, no macOS needed for that half), then reports the token-level
agreement rate and the index of the first divergence between the two
sequences.

This is deliberately not a bit-exact check: Core ML runs at fp16 internally
regardless of the ONNX graph's own dtype, while the `transformers` reference
here runs at fp32 on CPU, so a close-logit token can legitimately flip the
greedy argmax on one side and not the other -- and once one token diverges,
every later token's context differs too, so agreement is expected to trail
off after that point rather than resume. What the check actually watches for
is *how much* the two disagree (`--min-agreement`, default 80%) and *how
early* the first mismatch happens -- either one being far off is the signal
that something in the export/conversion pipeline is wrong, not just fp16
rounding.

```bash
python check_decode_parity.py HuggingFaceTB/SmolLM2-135M-Instruct \
    smollm2.mlpackage --prompt "The capital of France is" --max-new-tokens 20
```

`coreml-integration.yml`'s `benchmark-decode-macos` job runs this after the
decode benchmark, once per matrix entry, also posting to the job summary.
The pure comparison logic (`compare_token_sequences`) has no coremltools/
torch/transformers dependency and is unit-tested directly in
`tests/test_check_decode_parity.py`.

When a parity run fails, the next question is whether the first divergence
is fp16 noise or a systematic mistranslation. The discriminating measurement
is the margin at the divergence point: run both decoders teacher-forced on
the agreed prefix and compare full logit vectors (top-6 sets, EOS rank and
logit on each side, max abs drift over the vocab). On
`SmolLM2-135M-Instruct` the first divergence is a **0.06-logit coin flip**
between the top two tokens (both sides agree on the top-6 set; cross-side
drift ~0.03 on ~10.5-scale logits, 0.4 max over the vocab), and the EOS
logit error there is ordinary (rank ~42k on both sides) -- noise, with no
EOS-specific mistreatment. A second prompt agrees 20/20. A systematic bug
would instead show a large margin or an EOS logit far off its reference;
that is the signal to look for before blaming the translator.

### Weight-only quantization (`--quantize-weights`)

The "Theoretical ceiling" section above works out that a decode step is
bandwidth-bound, not compute-bound, at every model size this suite has
tested -- the whole weight set has to be read from DRAM once per token
regardless of how little arithmetic that token needs, since nothing
amortizes a weight read across tokens at batch size 1. `--quantize-weights
{int8,int4}` pulls the lever that actually follows from that: it applies
`coremltools.optimize.coreml.linear_quantize_weights` to the *converted*
Core ML model (`constexpr_affine_dequantize`, per-channel, symmetric),
replacing full-precision weight constants with int8/int4 ones that get
dequantized back to float on the fly at compute time.

```bash
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --quantize-weights int8 --output smollm2-int8.mlpackage
```

Deliberately **not** the same thing as Core ML's packed W8A8 mode (int8
weights *and* activations, which the M4 ANE can run at up to ~2x the fp16
*compute* rate): this flag only quantizes weights, leaving activations and
the actual compute in float, so it doesn't touch the ANE's compute
throughput at all -- appropriately, since compute isn't the bottleneck here
per the ceiling analysis. What it does do is roughly halve (int8) or
quarter (int4) the bytes moved from DRAM per decode step, which is the side
of the ceiling that's actually binding. `run_llm_decode_benchmark.py` and
`check_decode_parity.py` both work unchanged against a quantized
`.mlpackage` -- shapes and dtypes at the model's I/O boundary don't change,
only the weight constants' on-disk/in-graph representation does.

`coreml-integration.yml`'s `benchmark-decode-macos` job includes a
`quantize_weights: int8` matrix entry (same model as the unquantized
baseline) so both the decode-tok/s effect and decode parity are measured on
real hardware, not just argued for from the theoretical ceiling.

### matmul-to-conv1x1 (`--matmul-to-conv`)

Where `--quantize-weights` targets the bandwidth side of the ceiling,
`--matmul-to-conv` targets the *compute* side -- the M4 ANE's compute array
parallelizes over convolution output channels, and its native `matmul` path
measures well below conv1x1's throughput on the same hardware (see
"Theoretical ceiling" above: a single matmul reaches ~30% of the fp16
ceiling; a conv1x1 chain reaches ~99%). `onnxsim/coreml_export.py`'s
translator normally lowers every ONNX `MatMul` straight to MIL's `matmul`;
this flag makes it lower a **linear-projection** `MatMul` -- `x [batch,
sequence, C_in] @ w [C_in, C_out]` with a compile-time-constant 2-D `w`,
exactly the shape every attention/MLP projection in a transformer decoder
takes -- to a 1x1/pointwise `conv` instead (transpose to conv1d's `[n, C_in,
L]` layout, `conv` with a reshaped/transposed weight, transpose back; see
`convert_to_coreml`'s docstring for exactly which shapes qualify). Any
`MatMul` that doesn't match that shape (a non-constant or non-2-D weight, or
`x` of any rank other than 3 -- the real, non-linear-projection attention
score/context matmuls in every decoder layer, which multiply two activations
together) is left on the native `matmul` path.

```bash
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --matmul-to-conv --output smollm2-conv.mlpackage
```

Validated so far only at the level this dev sandbox can reach: the rewrite
is unit-tested for numeric correctness via MIL constant-folding
(`tests/test_coreml_export.py`, since `conv` itself has no
`value_inference` to fold through directly, the tests instead pull its
already-foldable `transpose`/`const` inputs and apply conv1x1's documented
semantics in plain numpy against the same `MatMul` reference), and a full
real-model export (`HuggingFaceTB/SmolLM2-135M-Instruct`) converts cleanly
with an unchanged I/O signature and file size, replacing all 168
linear-projection matmuls with `conv` while correctly leaving the ~60
genuine attention-score/context matmuls alone. What that validation
*cannot* show, lacking macOS/real Core ML in this environment, is whether
it actually helps decode tok/s at this pipeline's shapes -- the measurements
this flag is based on come from deep, wide conv1x1 chains (32-64 layers,
512-1024 channels), not the single-token decode steps this suite mostly
runs, and per-token sequence length here is far smaller than what those
measurements used. `coreml-integration.yml`'s `benchmark-decode-macos` job
includes a `matmul_to_conv: "true"` matrix entry (same model as the
unquantized baseline, decode parity included) specifically to get that
answer on real hardware; default it on only once that comparison actually
shows an improvement.

**Measured on real hardware** (`HuggingFaceTB/SmolLM2-135M-Instruct`,
`benchmark-decode-macos`, same runner/run as the unquantized baseline): this
flag made decode *slower*, not faster -- 2.82 tok/s vs. the matmul
baseline's 3.62 tok/s (-22%), with prefill roughly 50% slower too (1538ms
vs. 1025ms for 5 tokens). The opposite of what the raw ANE conv1x1-vs-matmul
throughput numbers suggested: those come from long, wide conv1x1 chains, not
this pipeline's short, mostly-single-token sequences, where the extra
transpose/conv/transpose bookkeeping this rewrite adds around every
projection apparently costs more than the ANE's per-op throughput gains
recover. (For reference, `--quantize-weights int8` on the same model/run
*did* help, as the bandwidth-bound theory predicted: 4.29 tok/s, +18.5% over
the same baseline.) Stays opt-in and off by default; not revisiting unless a
different model size/shape or a cheaper way to express the rewrite changes
this result.

`coreml-integration.yml`'s `benchmark-decode-macos` job also includes a
`quantize_weights: int8` + `matmul_to_conv: "true"` matrix entry
(`smollm2-135m-int8-conv`), checking the two flags together directly rather
than assuming they compose from the two isolated results above -- int8 only
changes how the weight constants are stored/dequantized before an op runs,
matmul-to-conv only changes which op the dequantized weight feeds into, so
they shouldn't interact, but that's an assumption worth checking rather than
trusting.

**Measured, and they don't compose neutrally.** Same run, same runner, same
model as the table above: baseline 3.61 tok/s, `--quantize-weights int8`
alone 4.64 (+28.5%), `--matmul-to-conv` alone 3.04 (-16%), **both together
2.26 (-37% vs. baseline, worse than either flag alone and worse than
picking just int8)**. Mean decode step latency follows the same pattern
(441.6ms combined vs. 215.5ms int8-only). Whatever's behind
`--matmul-to-conv`'s standalone slowdown -- the added transpose/conv/
transpose bookkeeping around every projection, most likely -- evidently
gets worse, not better, once the weights it operates on are also
quantized, rather than the two costs just adding. Reinforces the same
conclusion from the solo measurement above: `--matmul-to-conv` isn't worth
enabling for this pipeline's shapes, combined with int8 or otherwise.

### Static-context export (`--static-context N`)

All the flags above keep the dynamic KV cache (RangeDim `sequence_length` /
`past_sequence_length`); only the *weights* or *ops* change. This one changes
the dynamism itself: `--static-context N` pins one decode step at a fixed
context of N tokens (1 new token against N-1 cached) with fully static
shapes, instead of the dynamic cache. Pinning happens before `simplify`, so
constant folding also collapses the shape subgraphs dynamic dims would keep
alive (7364 -> 1981 nodes vs. 7364 -> 2663 dynamic, same model).

The reason is ANE plan construction, not speed directly. The dynamic model's
ANE-including plan fails to build on current macOS -- first on broadcast
`select`s (fixed in `onnxsim/coreml_export.py`'s `Where` lowering, which now
broadcasts explicitly), then on the inherently dynamic outputs themselves
(`Invalid blob shape: Data-dependent shapes were disabled`, for the growing
`present_*` cache). The static model builds cleanly, and the placement split
is exactly the "dynamic glue on CPU, compute on NPU" shape: `MLComputePlan`
reports **2055 ops (77.5% of estimated cost) on ANE, 11 on CPU** (shape casts,
a gather pair, one matmul) for `SmolLM2-135M` at N=512.

```bash
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --static-context 512 --output smollm2-static.mlpackage
```

Caveats, both measured on real hardware: at 135M/batch-1 the ANE path is
*slower* than CPU for a decode step (11-14ms vs. ~8ms steady-state) --
per-op dispatch and CPU/ANE transfers dominate where the ceiling analysis
says bandwidth should; NPU wins need more compute per step than this shape
offers. And a baked-in cache size means this is a placement probe and a
single-step benchmark, not a multi-step decoder: each new context length
needs its own export.

#### Static context × int8 weights: unlocking ANE at 1B+ scale

`--static-context` and `--quantize-weights int8` compose (the flags touch
independent stages: shape pinning before `simplify`, weight quantization
after conversion), and together they move the placement boundary. fp16
models at 1.5B+ land 100% on GPU -- E5RT spills the whole graph off ANE
somewhere between ~1.2GB and ~3.3GB of weights, and the 0.5B model sits
exactly in the transition (ANE 61% / GPU 37% / CPU 2%). Halving the weight
bytes with int8 pulls 1B+ models back under ANE capacity:

```bash
python export_llm_to_coreml.py Qwen/Qwen2.5-1.5B-Instruct \
    --max-context-length 512 --dtype fp16 --static-context 512 \
    --quantize-weights int8 --output qwen15-static-int8.mlpackage
```

Single decode step at context 512, static shapes, M4 Mac mini (placement =
`MLComputePlan` cost share, latency = mean `predict()` step):

| Model | fp16 placement | fp16 CPU / best | int8 ANE share | int8 CPU / int8+NE |
|---|---|---|---|---|
| Qwen2.5-0.5B | ANE 61 / GPU 37 / CPU 2 | 19.2 / 17.4 (GPU) | majority | 20.9 / **13.3 (1.57x)** |
| Qwen2.5-1.5B | GPU 100 | 55.9 / 42.9 (GPU) | 1918 ops (36%) | 59.1 / **36.2 (fastest overall)** |
| SmolLM2-1.7B | GPU 100 | 76.4 / 59.5 (GPU) | 1479 ops (43%) | 78.2 / 92.6 (CPU wins) |

So int8 is what engages the NPU at this scale -- the 1.5B int8+NE step is
the fastest measured configuration anywhere in this section -- but ANE
placement is not an ANE win by itself: SmolLM2-1.7B keeps 43% on ANE yet
runs slower than CPU (~1480 dispatches x ~90us floor each eats the compute
gains), while Qwen's matmul mix converts the same placement into a real
speedup. Numerics spot-check (1.5B, zeros feed): max abs drift 1.97 against
a 16.9 reference peak, no blowup; text-level parity needs the dynamic model
and is out of reach for statics by construction.

One hard warning: int8-quantized models **break the GPU backend**
(`MPSGraph` MLIR assertion inside `predict`), so a quantized model is an
NE-or-CPU choice -- `CPU_AND_GPU` is off the table once weights are
quantized.

#### Static-coverage summary (one decode step, context 512, M4 Mac mini)

Every model below is a `--static-context 512` export (`--dtype fp16` except
135M-fp32); placement is the `MLComputePlan` cost share, latency the mean
single-step `predict()`:

| Model | Weights | Placement | CPU | GPU | NE |
|---|---|---|---|---|---|
| SmolLM2-135M fp32 | 0.3GB | ANE 77 / CPU 23 | 8.1ms | -- | 11-14ms |
| SmolLM2-135M fp16 | 0.2GB | ANE 73 / CPU 27 | 7.9ms | -- | 11.4ms |
| SmolLM2-135M int8 | 0.2GB | ANE + GPU dequant lane | 8.5ms | -- | 8.0ms |
| SmolLM2-135M int4 | 79MB | ANE 47 / GPU 13 | 30.3ms | -- | 11.1ms |
| Qwen2.5-0.5B fp16 | 1.2GB | ANE 61 / GPU 37 / CPU 2 | 19.2ms | 17.4ms | 19.4ms |
| Qwen2.5-0.5B int8 | 0.6GB | ANE majority | 20.9ms | broken | **13.3ms** |
| Qwen2.5-1.5B fp16 | 3.3GB | GPU 100 | 55.9ms | 42.9ms | 59.7ms (CPU fallback) |
| Qwen2.5-1.5B int8 | 1.7GB | ANE 36% | 59.1ms | broken | **36.2ms (fastest)** |
| SmolLM2-1.7B fp16 | 3.4GB | GPU 100 | 76.4ms | 59.5ms | 79.4ms (CPU fallback) |
| SmolLM2-1.7B int8 | 1.7GB | ANE 43% | 78.2ms | broken | 92.6ms (CPU wins) |
| Phi-3.5-mini fp16 | 7.1GB | GPU 100 | 272.2ms | 183.2ms | 237.2ms (CPU fallback) |
| Phi-3.5-mini int8 | 3.6GB | GPU 100 | 6017ms (dequant hell) | 996.2ms | 474.1ms |

Patterns: per-op misses are boundary glue (embedding `gather`s, output
reshape/cast cluster, one bandwidth-bound vocab-head `matmul` -- 271/272
matmuls sit on ANE); `--io-dtype fp16` does not move the ANE share
(75.3% vs. 77.5%, noise); whole-graph GPU spill tracks weight size
(~1.2GB split zone, 3.3GB+ fully GPU), and int8 pulls graphs back under
ANE capacity -- except Phi-3.5, where int8 is a regression almost
everywhere (GPU 996ms vs. fp16 183ms; int8-on-CPU pathologically slow at
6s/step from per-step weight dequantization). int4 (135M only, needs an
iOS18-built model -- post-hoc quantization of a lower-target model is
refused outright) shrinks weights to 79MB with ANE still engaged but no
latency win at this size, as dispatch-bound theory predicts. Phi-3.5 is a new architecture family for this pipeline
(fused `qkv_proj` + partial rotary): its first real-device run failed plan
build on an empty pass-through slice concatenated back (`[96:96]` of dim
96, ORT-verified empty) that E5RT/MPS mis-shapes to 97 -- fixed by dropping
provably-empty inputs from `Concat` in the translator, after which Phi
traces and runs as tabulated.

#### Why dynamic models stay off ANE (the `Range` boundary)

Bisecting the dynamic decoder axis by axis (static S/P/T combinations of
the same graph) shows the ANE plan gate trips on any tensor whose shape
derives from a runtime *value*, not just a `RangeDim`: a 5-op repro of
`Range` with a runtime limit fails plan build with the exact production
error (`Invalid blob shape: Data-dependent shapes were disabled`), and so
does `Slice` with a runtime bound -- even a raw graph-input bound. Pure
`RangeDim` propagation (matmul/reshape/transpose/select/concat of dynamic
tensors) is fine. In the decoder the poison is the causal-mask machinery:
two model-level `Range`s with computed limits feed ~60 KV-cache slices and
a gather with Range-derived bounds, plus the present-cache concat chains.

No MIL reformulation fixes this class -- the information "this runtime
value equals dim S" cannot be expressed, so the translator leaves these
ops alone (conversion itself succeeds; pinned by test). What would fix it,
in increasing order of invasiveness: (1) static export (this section);
(2) computing positions/masks host-side and passing them as inputs instead
of building them from `Range` in-graph (model-interface change, fragile
across families -- not attempted); (3) stateful Core ML (cache as internal
state, static outputs -- the direction community artifacts like TokForge
explore); (4) Apple relaxing the gate. The ORT CoreML execution provider
does not help either: it partitions the decoder into 241 CoreML subgraphs
fine, but the poisoned partitions fail the same E5RT build at run time.

### fp16 model interface (`--io-dtype fp16`)

Where `--quantize-weights` cuts the bytes read from DRAM *inside* the model and
`--matmul-to-conv` changes which op runs, this flag touches neither: it changes
the dtype of the model's **interface** -- the float tensors a caller hands in
and gets back -- from float32 to float16.

The reason it's worth a flag: an ML Program already computes in float16
(coremltools' `compute_precision` default), but coremltools declares the
model's inputs and outputs float32 unless told otherwise. That mismatch is not
free. It shows up directly in the emitted program -- exporting a two-op graph
and reading back the ops Core ML actually stores (`const` instances, which just
carry other ops' arguments, elided):

| `io_dtype` | ops in the ML Program |
| --- | --- |
| `fp32` (default) | `cast`, `relu`, `sqrt`, `cast` |
| `fp16` | `relu`, `sqrt` |

Those two `cast`s are the boundary conversion, and they run on every single
`predict()` call: float32 down to float16 on the way in, float16 back up to
float32 on the way out, with twice the bytes crossing the boundary in each
direction. This pipeline is close to the worst case for that. The KV cache
crosses the boundary **twice per decode step** -- in as `past_key_values_*`,
out as `present_*` -- it is by far the largest thing moving (far larger than
the single token's activations that step actually computes), and it grows with
every token generated. `--io-dtype fp16` deletes both conversions.

The finding this follows is from the same M4 ANE reverse-engineering work as
the "Theoretical ceiling" section above -- its companion repository
[`maderix/ANE`](https://github.com/maderix/ANE), which drives the ANE directly
through the private `_ANEClient`/`_ANECompiler` APIs rather than through Core
ML. Passing tensors as fp16 in its IOSurface I/O path measures **~37% faster
than fp32** over the same buffers ("How It Works", step 3). That number is from
a different stack (raw IOSurface I/O, no Core ML framework in between), so
treat it as the reason to expect the effect, not as a prediction of this
pipeline's speedup -- `coreml-integration.yml`'s `benchmark-decode-macos`
matrix has an `io_dtype: fp16` entry (same model as the unquantized baseline,
decode parity included) to measure what it's actually worth here.

```bash
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --io-dtype fp16 --output smollm2-io16.mlpackage
```

Unlike the other two flags, this one is **not** a precision or structural
trade-off. `--quantize-weights` changes stored values; `--matmul-to-conv`
changes which ops run. This changes neither: the computation was float16 on
both sides of the flag, so an fp32 output was only ever an upcast copy of the
fp16 value Core ML had already computed. What changes is whether the caller is
handed that value directly.

Notes:

- Only **float** inputs and outputs move. Integer inputs (`input_ids`,
  `attention_mask`, `position_ids`) are untouched -- Core ML has no float16
  form for them.
- Needs `--format mlprogram` (the default; the legacy `neuralnetwork` format
  has no float16 interface) and iOS16/macOS13 or newer. `onnxsim.export_coreml`
  raises the deployment target to iOS16 itself when one isn't given, and
  rejects an explicitly *lower* one rather than emitting a model Core ML would
  refuse to load.
- The graph body is lowered exactly as it is without the flag: each fp16 input
  is cast straight back to the dtype the ONNX graph declares before any node is
  translated, so no op sees a different dtype than it otherwise would, and
  coremltools' own fp16 compute-precision pass folds the round trip away (the
  table above is that folding, observed).
- `run_llm_decode_benchmark.py`, `check_decode_parity.py` and
  `lm_eval_coreml_adapter.py` all read their feed dtypes off the model's own
  spec (`_DATATYPE_TO_NUMPY`), so they run against either interface with no
  changes.

#### What else came from `maderix/ANE`, and why not (yet)

That repository documents several other ANE findings. Two are worth recording
as deliberately *not* taken here, so the next person doesn't have to re-derive
the reasoning:

- **INT8 W8A8 (int8 weights *and* activations)** -- measured there at
  **1.85-1.88x** fp16 throughput (35.1 vs 18.6 TOPS on 128x conv 512ch), via
  MIL `quantize`/`dequantize` between tiles so activations stay int8 in L2
  SRAM. Core ML's own equivalent is
  `coremltools.optimize.coreml.linear_quantize_activations`, which needs a
  calibration pass over real data. It is a **compute** lever, and the
  "Theoretical ceiling" section above works out that single-token decode at
  these model sizes is bandwidth-bound with compute already down in the noise
  -- so it would be aimed at the side of the ceiling that isn't binding.
  Interesting for *prefill* (whole-prompt, genuinely compute-bound), which this
  suite doesn't currently optimize for; not for the decode tok/s number
  `run_llm_decode_benchmark.py` reports.
- **Channel-first `[1,C,1,S]` layout throughout** -- that project eliminates
  transposes by keeping *its whole CPU-side pipeline* in the ANE's native
  IOSurface layout, not by transposing around each op. Notably, that is the
  opposite of what `--matmul-to-conv` does here: it wraps every projection in a
  transpose/conv/transpose, and measured *slower* (see above). Doing this
  properly means a layout-propagation pass over the whole graph that inserts
  transposes only where the layout genuinely has to change -- a real
  translator-level project, not a flag, and one that should only be started
  once there's a measurement showing the transposes (rather than something
  else) are what cost `--matmul-to-conv` its 16-22%.

Its remaining findings are specific to driving the ANE through the private
APIs and have no Core ML analogue to port: the ~119-compiles-per-process limit
worked around by `exec()`, the single-input packing constraint, and the
SDPA-`attn_mask` gap (that project decomposes causal attention itself because
ANE hardware ignores `attn_mask`; `onnxsim/coreml_export.py` never emits a
fused SDPA op in the first place -- it lowers attention as explicit
`matmul`/`softmax`/`add`, so there is no mask to be dropped).

### Compute-unit device placement (`coreml_compute_plan_trace.m`)

`--matmul-to-conv`'s standalone measurement above raised an obvious
follow-up: was the slowdown because `matmul` and `conv` land on different
Core ML compute units (CPU/GPU/ANE), or is something else going on?
`run_llm_decode_benchmark.py`/`coreml_backend.py` only ever call
`.predict()` with `MLComputeUnits=ALL` and report aggregate tok/s + RSS --
no per-op visibility into which unit actually ran anything.

`coreml_compute_plan_trace.m` closes that gap using `MLComputePlan`
(macOS 14+), the Core ML framework's own static analysis API: given a
*compiled* model (`xcrun coremlcompiler compile model.mlpackage <dir>`
first -- `MLComputePlan` doesn't load `.mlpackage` directly), it walks
every operation in the ML Program (`onnxsim/coreml_export.py`'s
`convert_to="mlprogram"` default -- the only format this tool supports)
and asks the framework for each op's preferred compute device and
estimated relative cost, **without running the model**. Output is Chrome
Trace Event Format JSON (openable at `chrome://tracing` or
https://ui.perfetto.dev) -- one timeline lane per device, so which ops
landed where is visible at a glance instead of read off a text dump.

**This is a static estimate, not a measurement**: `dur` in the emitted
trace is `MLComputePlanCost`'s `weight` (a relative-cost fraction) scaled
by 1e6 purely so a trace viewer renders something legible, not
microseconds from an actual `.predict()` call. Real per-op wall-clock
timing would need Instruments' Core ML template
(`xcrun xctrace record --template "Core ML"`) attached to a live
prediction run instead -- a far less tractable trace format to parse than
`MLComputePlan`'s structured API, so out of scope here. What this tool
answers is narrower and cheaper: which compute unit does Core ML's own
placement logic *prefer* for each op, before spending any time actually
running it.

```bash
clang -O2 -o coreml_compute_plan_trace coreml_compute_plan_trace.m \
    -framework CoreML -framework Foundation
xcrun coremlcompiler compile model.mlpackage compiled
./coreml_compute_plan_trace compiled/model.mlmodelc out.json
```

Plain C/Objective-C compiled with plain `clang` (matching the pattern in
[freedomtan/coreml_modelc_profling](https://github.com/freedomtan/coreml_modelc_profling),
whose `MLComputePlan` API calls this file's traversal is adapted from) --
no Xcode project, no Swift toolchain. Unlike everything else in
`scripts/apple`, this file couldn't be validated at all before landing in
CI (no macOS/Core ML in any environment developing this repo, and no way
to even syntax-check Objective-C against the real `CoreML.framework`
headers) -- `coreml-integration.yml`'s `benchmark-decode-macos` job is the
first place it actually compiles and runs, specifically against the
`smollm2-135m` / `smollm2-135m-conv` matrix entries (the plain-matmul vs.
matmul-to-conv comparison this tool exists to explain), uploading each
trace as a workflow artifact.

The first real run built and executed cleanly but produced an **empty**
trace -- `computeDeviceUsageForMLProgramOperation:`/
`estimatedCostOfMLProgramOperation:` returned `nil` for every operation in
the model. Root cause: those two `MLComputePlan` methods need the *traced
model's own* `minimum_deployment_target` to be at or above roughly the
iOS17.4/macOS15.4 SDK generation -- unrelated to which OS/Xcode the machine
running this tool has. `smollm2-135m`/`smollm2-135m-conv` don't quantize
weights, so they got whatever (lower) target `onnxsim.export_coreml` picks
by default. Fixed by giving `export_llm_to_coreml.py` a
`--minimum-deployment-target` flag and having `coreml-integration.yml` pass
`iOS18` (the highest target coremltools exposes) for just those two matrix
entries. The tool itself also now prints `operations.count` and per-op
nil/non-nil `deviceUsage`/`estimatedCost` status for the first few
unanalyzable ops, so a still-empty trace after this fix would point at the
real cause immediately instead of requiring another blind guess.

That first fix in turn surfaced a second, real bug: with
`minimum_deployment_target=iOS18`, `xcrun coremlcompiler compile` started
rejecting the exported model outright (`Failed to parse the model
specification. Error: Unable to parse ML Program: ... Required param
'validate_indices' is missing`, on a `Gather` op) -- `onnxsim/coreml_export.py`
always built its MIL program at coremltools' lowest default opset regardless
of the requested target, relying on `ct.convert`'s own op-version-upgrade
pass to bridge the gap when a higher target was requested. That pass doesn't
backfill newly-applicable optional inputs (`validate_indices` was added to
`gather` at the iOS17 op version); building at coremltools' lowest opset and
then upgrading in place left it unset in the serialized spec, and
`coremlcompiler` treats it as required to load. Fixed by threading the
resolved `minimum_deployment_target` into `_build_mil_program` as the MIL
program's own `opset_version`, so MIL's builder synthesizes each op's
version-appropriate default inputs itself instead of upgrading after the
fact -- see `test_gather_at_ios18_target_serializes_validate_indices` in
`tests/test_coreml_export.py`.

With both of those fixed, the first real trace against real CI (a macOS-15
GitHub-hosted runner) surfaced a third finding, not a bug this time: real
per-op data, but only half the picture. `main function has 6865
operation(s)` / `recorded 0/6865 op(s) (3668 missing deviceUsage, 6865
missing estimatedCost)` -- `computeDeviceUsageForMLProgramOperation:` now
returns real placement data for ~45% of ops, but
`estimatedCostOfMLProgramOperation:` returns `nil` for *every* op, on both
`smollm2-135m` and `smollm2-135m-conv`. The tool previously required both to
be non-nil before recording an event, so it kept producing an empty trace
even with real device-placement data sitting right there. Fixed by
decoupling the two: an event is now recorded whenever `deviceUsage` alone is
non-nil, with `cost_available: false` and a 0 weight in `args` when
`estimatedCost` isn't -- the per-lane summary switches from a "% of total
estimated cost" line to a plain op count whenever no op in the whole run got
real cost data, so the output doesn't paper over a real gap with a
misleading 0.00%. Whether `estimatedCost`'s unavailability is a further
SDK-version gap or a standing limitation of on-device compute-plan cost
analysis (as opposed to Xcode's own Model Performance Report) is
unconfirmed -- device *placement* (the actual question this tool exists to
answer: does `--matmul-to-conv` move ops to a different compute unit) is
unaffected by it.

### Quality and retention eval (`run_quality_eval.py` / `compute_retention.py`)

The decode benchmark and parity check above measure speed and short-generation
correctness; they say nothing about actual model *quality* -- DeviceMark's
third axis (alongside decode speed and memory), scored via IFEval, MMLU-Pro,
and MATH-500, plus **retention**: how much of the float model's benchmark
score survives quantization (`quantized_score / float_score`). See
`bench/TODO_quality_retention_eval.md` for the full plan (including exactly
how DeviceMark itself defines this -- subset sizes, 0-shot, and a
completed-only accuracy `compute_retention.py` now replicates too, see below);
this is the first implemented slice of it.

Rather than reimplementing any benchmark's prompt formatting or answer
scoring, these two scripts lean on
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
(`pip install "lm-eval[ifeval]"`), which already has all three as task
definitions:

- `lm_eval_coreml_adapter.py` registers a `coreml` model backend wrapping
  `CoreMLDecoder` so the harness can score an exported `.mlpackage` the same
  way it scores any other model. It only implements `generate_until` --
  IFEval, MMLU-Pro, and `hendrycks_math500` (MATH-500) are all
  `generate_until` tasks in this harness version (free-form generation,
  scored by a verifier or answer extraction), never `loglikelihood`-based
  multiple choice, so `CoreMLDecoder`'s existing greedy `generate()` is
  already the right primitive -- no teacher-forced-logprob code path needed.
- `run_quality_eval.py` is a thin CLI over `lm_eval.simple_evaluate`,
  supporting both `--model hf` (the float side -- CPU-only, no macOS needed)
  and `--model coreml` (the quantized side -- macOS/real Core ML only) against
  the same task/subset, writing a small JSON summary. Each scored metric
  reports plain `acc` (no-answer-within-budget counts as wrong) and, whenever
  `--max-gen-toks` is passed explicitly, `acc_completed` too -- accuracy over
  only the samples whose response finished before exhausting that budget
  (`is_completed()`; validated against real generations -- see that
  function's docstring), DeviceMark's own retention definition.
- `compute_retention.py` takes one JSON from each side and reports the
  per-task, per-metric retention ratio, preferring each side's `acc_completed`
  when it's present and defined and falling back to plain `acc` otherwise
  (`_resolve_score()`). `--output` also writes a flat `"records"` list (one
  object per task/metric: `model_id`, `benchmark`, `metric`, `subset_n`,
  `float_acc`, `quantized_acc`, `retention`, `float_basis`, `quantized_basis`)
  alongside the nested summary -- `coreml-integration.yml` uploads these as
  the `quality-retention-results` CI artifact, so a run's numbers don't have
  to be re-parsed out of `$GITHUB_STEP_SUMMARY` text.
- `aggregate_quality_trend.py` reads several `compute_retention.py --output`
  files (e.g. downloaded from a handful of past `quality-eval-macos` runs)
  and groups their `"records"` by `(model_id, benchmark, metric)`, so a
  model's retention/accuracy can be read as a trend across runs instead of
  one isolated data point per run:
  ```bash
  python aggregate_quality_trend.py retention_ifeval_run1.json \
      retention_ifeval_run2.json --output trend.json
  ```
  Not wired into CI -- `quality-eval-macos` runs a single fixed model with no
  run-history persistence today, so this is meant to be run by hand against
  artifacts fetched from past runs (`gh run download` or the Actions UI), not
  something the workflow calls itself yet.

```bash
pip install "optimum-onnx" transformers coremltools onnxruntime "lm-eval[ifeval]"
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --output model.mlpackage
python run_quality_eval.py --model hf \
    --model-args pretrained=HuggingFaceTB/SmolLM2-135M-Instruct,dtype=float32 \
    --tasks ifeval --limit 10 --max-gen-toks 128 --apply-chat-template \
    --output float_ifeval.json
python run_quality_eval.py --model coreml --model-args pretrained=model.mlpackage \
    --tasks ifeval --limit 10 --max-gen-toks 128 --apply-chat-template \
    --output coreml_ifeval.json
python compute_retention.py float_ifeval.json coreml_ifeval.json
```

A `--limit`-restricted run like the one above is explicitly **not**
benchmark-grade (`lm_eval` itself warns about this) -- treat it as "did this
get meaningfully worse", not as a score comparable to a published
leaderboard entry. `--max-gen-toks` matters more here than on a typical
batched GPU eval: each generated token is its own single-token forward pass
on an unbatched decoder (HF on CPU, or a real Core ML `.mlpackage`), so a
benchmark's default generation budget (MMLU-Pro's is 2048 tokens) directly
sets wall-clock cost. A prototype run of even a single MMLU-Pro example
against `HuggingFaceTB/SmolLM2-135M-Instruct` on CPU took well over a
minute at that default -- `coreml-integration.yml`'s `quality-eval-macos`
job (workflow_dispatch/schedule-only) therefore runs IFEval and MATH-500
(`hendrycks_math500`, cheap at this suite's scale -- no large default
generation budget in its task YAML, ~7s/example prototyped on CPU) with both
`--limit` and `--max-gen-toks` capped. `--max-gen-toks 256` does fix MMLU-Pro's
cost too (~14s/example against that same model, verified not to be cutting
answers off mid-generation), but at `--limit 10` that model's float-side score
on `mmlu_pro_biology` is 0/10 regardless of the cap -- a zero float score makes
`compute_retention.py`'s ratio undefined (`None`) by design, so MMLU-Pro stays
future work until that's addressed rather than the cost itself (see the plan
doc's "Next steps").

The retention-ratio logic (`compute_retention`) has no lm-evaluation-harness/
torch/coremltools dependency and is unit-tested directly in
`tests/test_compute_retention.py`.

### Scaling to few-billion-parameter models

DeviceMark's own leaderboard tests models mostly in the 0.8-5B range (its
current rows: Qwen3.5-0.8B/2B/4B, LFM2.5-1.2B, Granite-4.0-H-1B,
Youtu-LLM-2B, Nemotron-3-Nano-4B, Nanbeige4.1-3B, Gemma 4 E2B), well past
`HuggingFaceTB/SmolLM2-135M-Instruct`'s 135M. The export pipeline above
has also been validated end-to-end against `HuggingFaceTB/SmolLM2-1.7B-Instruct`
(24 layers, ~3.4GB of fp16 weights) -- converting a model at that scale
needs a couple of extra considerations `export_llm_to_coreml.py` handles for
you, and one flag worth knowing about:

- `--dtype fp16` traces and exports the ONNX graph in half precision
  instead of the default float32. Tracing a multi-billion-parameter model in
  float32 can transiently hold more than one full-size copy of its weights
  in memory (PyTorch's own model plus the in-progress ONNX graph); `fp16`
  roughly halves that peak. This is independent of Core ML's own output
  precision, which defaults to float16 regardless of the ONNX input's dtype.
- The batch-size fix-up and `onnxsim.simplify` step both operate on the
  model **by file path**, not as an in-memory `ModelProto` -- passing a
  `ModelProto` to either serializes the whole model to one protobuf message
  first, which protobuf itself caps at 2GiB (comfortably cleared by a
  multi-billion-parameter model's weights). Passing a path instead uses
  onnx's/onnxsim's own file-based C++ entry points, which need only about
  1x the model's size in peak memory rather than 2x+ (see
  `bench/RESULTS_synthetic_decoder_oom.md` in the repo root for the
  investigation that fixed the `onnxsim.simplify` side of this).
- `main_export(..., do_validation=False)` skips `optimum`'s own
  PyTorch-vs-ONNX-Runtime comparison pass, which otherwise keeps a second
  full copy of the model resident purely to check optimum's own export --
  a check this pipeline doesn't rely on (onnxsim's `onnx.checker.check_model`
  and the Core ML conversion actually succeeding are the checks that matter
  here).

```bash
pip install "optimum-onnx" transformers coremltools onnxruntime
python export_llm_to_coreml.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --max-context-length 512 --output smollm2.mlpackage
python run_llm_decode_benchmark.py smollm2.mlpackage \
    --prompt "The capital of France is" --max-new-tokens 20
```

**Measured on real Core ML** (`benchmark-decode-macos`'s widened matrix,
5-token prompt, macOS GitHub-hosted runner -- see
`prepare_benchmark_models.py`'s `BENCHMARK_MODELS` for the full notes per
model):

| Model | Params | decode tok/s | prefill | peak RSS | parity |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 0.1B | 3.6 | 1.0s | 0.8GB | 20% (fp16-vs-fp32, see "Decode parity" above) |
| Llama-3.2-1B-Instruct | 1B | 1.76 | 11.3s | 3.1GB | 100% OK |
| Qwen2.5-1.5B-Instruct | 1.5B | 1.76 | 9.5s | 3.9GB | 30% FAIL |
| SmolLM2-1.7B-Instruct | 1.7B | 2.11 | 7.5s | 4.0GB | 66.7% FAIL |
| Qwen2.5-3B-Instruct | 3B | 0.04 | 30.2s | 5.6GB | not measured |
| Llama-3.2-3B-Instruct | 3B | export fails (disk space, see `BENCHMARK_MODELS`) | | | |
| Phi-3.5-mini-instruct | 3.8B | export fixed, not yet re-benchmarked | | | |

Decode tok/s does **not** scale smoothly with parameter count on this
runner class: Qwen2.5-3B-Instruct's 0.04 tok/s is a ~50x cliff from the
1.7B model's 2.11, not the ~2x the weight-size ratio alone would predict
(bandwidth-bound reasoning per the "Theoretical ceiling" section above
would suggest roughly linear scaling with weight bytes). Re-running the
same model/benchmark on a 16GB M4 Mac mini (CPU_ONLY, 7.3GB peak RSS)
gives **2.44 tok/s** with a 15.8s prefill -- faster than even the
weight-ratio extrapolation (~0.9 tok/s from the 1.5B row) -- so the cliff
does not reproduce where memory is plentiful, and is best read as the old
runner's memory ceiling (5.6GB RSS there), not an architectural property
of the 3B tier. `SmolLM2-135M-Instruct`'s parity failure is
the fp16-Core-ML-vs-fp32-HF-reference divergence the "Decode parity"
section above already explains (different generated content after the
first mismatch, not a stopping-point difference). `SmolLM2-1.7B-Instruct`'s
is a different failure mode: the tokens it generated *agree* with the HF
reference everywhere the reference has tokens to compare -- Core ML just
kept generating past where the 3-token HF reference stopped
(`'Paris.\nThe capital of France is Paris...'` looping). That shape --
agreement, then the HF side emitting EOS where Core ML does not -- is what
a sub-0.1-logit fp16 coin flip landing exactly on the EOS decision looks
like, and the margin probe below confirms flips of exactly that size
decide top-1 elsewhere; no EOS-specific mistreatment was found (the EOS
logit error is ordinary, rank ~42k on both sides at a divergence point).

### Benchmarking a real model suite (`prepare_benchmark_models.py`)

A batch wrapper around `export_llm_to_coreml.py`: exports every model in its
`BENCHMARK_MODELS` list (or a `--only` subset) into its own
`<output-dir>/<slug>/model.mlpackage`, so the result is a ready-made set of
models to run `run_llm_decode_benchmark.py` against on macOS -- the actual
"reproduce a DeviceMark-style benchmark" step (in spirit: same rough weight
class and the same decode-tok/s-and-memory axes, not literally DeviceMark's
own models or its private on-device runtime -- see
`bench/TODO_quality_retention_eval.md`). The default list spans a few
architecture families in DeviceMark's own ~0.8-5B weight class (Llama-style,
Qwen2, Phi-3), not just one, since `onnxsim/coreml_export.py`'s translator is
a hand-written ONNX-to-MIL mapping where different architectures can exercise
different op combinations -- see the script's module docstring for which
entries have actually been run through this pipeline versus which are
expected to work but not yet exercised (larger models need more RAM/disk than
a constrained dev sandbox has). `meta-llama/Llama-3.2-*-Instruct` is gated on
Hugging Face (needs an accepted license + `HF_TOKEN`) but is in the default
list anyway -- this repo's CI has a read-only `HF_TOKEN` secret, wired into
the `coreml-integration` workflow's `export-benchmark-models` job (runs on
`workflow_dispatch`/schedule only, not on every PR, since it's a multi-GB
download). `google/gemma-2-*-it` -- also gated, and with more architectural
unknowns (sliding-window attention, logit soft-capping) not yet checked
against this translator at all -- is left out of the default list; pass it as
`--only` once you have access, to try it anyway.

```bash
python prepare_benchmark_models.py --output-dir benchmark_models
# then, per model, on macOS:
python run_llm_decode_benchmark.py benchmark_models/Qwen_Qwen2_5_1_5B_Instruct/model.mlpackage \
    --prompt "The capital of France is" --max-new-tokens 20
```

### Training on ANE (`train_mlp_step_coreml.py`)

Everything above is inference. `train_mlp_step_coreml.py` closes the loop:
it builds a 2-layer MLP Adam training step as an ONNX graph with onnxsim's
own training-graph tooling (`graph_grad.build_backward` for the backward
pass, `qat_graph.adam_update` + `make_step_graph` for the update and the
state-in/state-out plumbing), converts it with `onnxsim.export_coreml`, and
runs the loop through `predict()`, feeding updated parameters and Adam
moments back each step -- the same state-feedback shape as the KV-cache
decode loop, but for weights. The maderix/ANE project reaches this hardware
for training through reverse-engineered private APIs instead; this is the
same question answered through the public Core ML stack.

```bash
python train_mlp_step_coreml.py --output mlp_step.mlpackage \
    --steps 15 --compute-units CPU_AND_NE
```

### Transformer-block training (`train_block_step_coreml.py`)

The same loop around a pre-norm-less transformer block instead of an MLP:
separate Q/K/V projections, head split, scaled dot-product attention,
output projection, residual, SiLU FFN, residual, MSE loss, Adam on all 12
projection params -- the attention backward (`sdpaBwd` in maderix/ANE
terms) runs on the Neural Engine here along with everything else, where
maderix splits it (forward+dx on ANE, dW/Adam on CPU).

```bash
python train_block_step_coreml.py --output block_step.mlpackage \
    --steps 8 --compute-units CPU_AND_NE
```

Measured on real hardware (M4 Mac mini, 32x64x512 block, 8 heads, FFN
2048, ~3.2M params): **352/352 ops on ANE**, loss 2.41 -> 2.22 over 8
steps tracking CPU, at 34.8ms/step vs. 40.3ms CPU (1.16x). Per-param
throughput trails maderix's Stories110M loop (~0.09 vs. ~1.2M params/ms)
-- expected: every step shuttles ~100MB of weights/moments through
`predict()`, where their IOSurface pipeline keeps weights resident.

Two adaptations the Core ML target forces: rank-0 scalar inputs are widened
to shape-[1] (MIL has no rank-0 Placeholder; broadcast-identical where used),
and the step uses loss scaling with eps=1e-3, since default fp16 compute
zeroes Adam's 1e-8 epsilon and underflows small gradients (the same fp16
backward-underflow class maderix documents).

Measured on real hardware (M4 Mac mini, 1024-2048-1024 MLP, batch 256,
~4.2M params): the whole step (forward + backward + Adam) places **100% on
ANE**, and loss falls 0.80 -> 0.37 over 15 steps, tracking the CPU and
ONNX-Runtime references. Per-step time is 10.3ms on ANE vs. 9.7ms CPU --
~50MB of weights/moments cross the `predict()` boundary per step, so
transfers dominate at this size; a resident-weight loop (maderix's IOSurface
approach) would be needed to chase their throughput rather than just their
placement.

### int8 training (`--qat-int8`, QDQ lowering + straight-through gradients)

Post-conversion weight quantization cannot touch a training loop (its
weights are loop-carried inputs, not constants -- the pass quantizes one
tiny scalar and nothing else). Real int8 training is quantization-*aware*:
fake-quantize the MatMul weights in the forward pass
(`DequantizeLinear(QuantizeLinear(w))`, symmetric int8, scales calibrated
once from the initial weights) while Adam trains the fp32 masters through
straight-through estimation. Supporting that took two translator-adjacent
additions, both covered by unit tests:

- `QuantizeLinear`/`DequantizeLinear` lowering in `onnxsim/coreml_export.py`
  (MIL-native `quantize`/`dequantize`, iOS17+ -- conversion raises the
  deployment floor automatically, like fp16 I/O's iOS16 bump), including the
  previously-rejected int8/uint8 constant initializers they need for scales
  and zero points.
- Straight-through gradient rules in `onnxsim/graph_grad.py`'s
  python-only table (the pair differentiates as the identity and emits no
  nodes; no C++ mirror needed, same as `Where`).

```bash
python train_mlp_step_coreml.py --qat-int8 --output mlp_step_qat.mlpackage \
    --steps 15 --compute-units CPU_AND_NE
```

Measured on real hardware (same S1 MLP): the QAT step places **114/114 ops
on ANE, quantize/dequantize pairs included**, and loss falls 0.82 -> 0.37 --
identical to the fp32-master run -- at 10.4ms/step. The int8 forward costs
nothing measurable here because the weights still cross the boundary fp32
every step; the win this unlocks is the one the previous section measured
for inference (halved weight bytes fitting ANE capacity), now available to
a training forward pass too.

### Resident weights (`--resident`, Core ML states)

The loop above still shuttles every parameter and moment through
`predict()` twice per step (~50MB at S1). `export_coreml`'s `state`
argument (also behind `--resident` here) holds them resident on-device
instead, in Core ML states: each named input becomes an fp16 state read
back at the boundary, each named output is written into its state via
`coreml_update_state` and dropped from the interface, so only the batch,
the target, the scalars -- and the loss -- cross per step (~2MB here).

```bash
python train_mlp_step_coreml.py --resident --compute-units CPU_ONLY \
    --output mlp_step_resident.mlpackage
```

Measured on real hardware (same S1 MLP): loss 0.82 -> 0.37, tracking the
shuttling runs, at **5.1ms/step vs. 9.7-10.3ms** -- the traffic win is
real. Three platform facts came out of validating it: states are fp16-only
(the weights live one rounding away from the fp32 masters -- convergence
is unaffected); stateful conversion needs iOS18+ (raised automatically,
like the other deployment floors); and ANE compilation rejects stateful
programs on current macOS (`ANECCompile() FAILED`), so resident loops run
CPU/GPU-side -- the GPU accepts them but slowly (105ms/step here), making
CPU the best resident backend today. Conversion also drops one coremltools
pass (`common::canonicalize_inplace_pattern`, which crashes reordering
clustered trailing updates) -- canonicalization only, functionally
neutral, pinned by test.
