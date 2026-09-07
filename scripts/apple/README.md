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
would suggest roughly linear scaling with weight bytes) -- peak RSS
approaching the runner's likely memory ceiling at that tier is the leading
suspect, but this wasn't isolated; treat it as a real, measured number and
not yet a fully explained one. `SmolLM2-135M-Instruct`'s parity failure is
the fp16-Core-ML-vs-fp32-HF-reference divergence the "Decode parity"
section above already explains (different generated content after the
first mismatch, not a stopping-point difference). `SmolLM2-1.7B-Instruct`'s
is a different failure mode: the tokens it generated *agree* with the HF
reference everywhere the reference has tokens to compare -- Core ML just
kept generating past where the 3-token HF reference stopped
(`'Paris.\nThe capital of France is Paris...'` looping), a
greedy-decoding/EOS-handling difference at this size, not yet root-caused,
rather than a translator correctness bug.

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
