# Re-fusion of compute-dominant operators (Attention & friends)

A transformer's runtime is almost entirely the attention block and the two FFN
GEMMs.  Every serious runtime therefore *re-fuses* the exported node soup back
into single kernels -- ONNX Runtime rewrites the block into
`com.microsoft.Attention` / `MultiHeadAttention`, `SkipLayerNormalization`,
`BiasGelu`; TensorRT does the same inside its own builder.  Those rewrites are
**pattern matches on the exact node sequence the exporter emitted**.

That makes "simplification" and "fusability" pull against each other: a graph
rewrite that removes one node but perturbs the pattern can cost the model an
entire fused attention kernel.  This survey measures which way onnxsim pulls.

Everything here is reproducible with the two scripts in this directory:

```bash
python bench/refusion_survey.py --preset tiny --bisect      # pattern-level, CPU
bash   bench/refusion_gpu_run.sh                            # latency, CUDA EP
```

Environment for the numbers below: onnx 1.22.0, onnxruntime 1.28.0,
CPUExecutionProvider, `--preset tiny` (2 layers, hidden 256, 4 heads, seq 128),
onnxsim built from this branch (`claude/operator-refusion-survey-ve7ee6`, at
`5db7370`) and, where noted, onnxsim 0.7.0 from PyPI for comparison.

## How it is measured

Each fixture is a hand-built ONNX graph with the topology real exporters emit
(`bench/refusion_fixtures.py`), and is run through this matrix:

| variant | meaning |
| --- | --- |
| `raw` | the exported graph |
| `sim` | `onnxsim.simplify` with default settings |
| `sim_nogemm` | `simplify` with `fuse_matmul_add_bias_into_gemm_batched` skipped |
| `ort(raw)` | ONNX Runtime's transformer fusion on the export — the ceiling |
| `ort(sim)` | ORT fusion *after* onnxsim — **the re-fusion question** |
| `sim(ort(raw))` | onnxsim on an already-fused model — **the preservation question** |

Two independent fusion measures are reported: `onnxruntime.transformers`
(`opt_level=0`, pure pattern matching, so every fused op came from a matcher
walking that exact graph) and ORT's own session-level optimizer
(`ORT_ENABLE_ALL`, which is what actually runs, and is execution-provider
dependent).

---

## Finding 1 — onnxsim currently produces a **broken model** for a standard attention block

On this branch, the default pipeline turns the BERT fixture into a model that
ONNX Runtime refuses to run:

```
Node (layer0.attn/PV) Op (MatMul) [ShapeInferenceError] Incompatible dimensions
Reshape 'layer0.attn.k/Reshape': input shape {128,256}, requested shape {128,256,4,64}
```

Minimal reproduction — three nodes, no attention required:

```python
# x[1,8,16] -> MatMul(W[16,16]) -> Add(b[16]) -> Reshape([0,0,4,4]) -> y[1,8,4,4]
sim, _ = onnxsim.simplify(model)      # default settings
# -> Reshape(x,[-1,16]) -> Gemm -> Reshape(·,[0,0,4,4])   ... which cannot run
```

Two passes combine to produce it:

1. `fuse_matmul_add_bias_into_gemm_batched` (opted into by default in
   `onnxsim.cpp`) rewrites the rank-3 `MatMul+Add` of every Q/K/V projection
   into `Reshape → Gemm → Reshape(restore rank)`.
2. `eliminate_consecutive_idempotent_ops` (upstream onnx-optimizer) then treats
   `Reshape(Reshape(x, s1), s2)` as `Reshape(x, s2)`.  That identity is **false
   whenever `s2` contains a `0`**: with the default `allowzero=0`, a `0` means
   "copy the corresponding dimension *of this node's input*", and the rewrite
   changes what that input is.  The pass does not check for zeros:
   `third_party/onnx-optimizer/onnxoptimizer/passes/eliminate_consecutive_idempotent_ops.h`
   only checks `uses().size() == 1`.

Exporters emit `Reshape` shapes with `0` constantly (`x.view(B, S, H, D)` traces
to `[0, 0, 4, 64]`), and the head-splitting reshape directly follows the Q/K/V
projection — so *every* attention block hits this.

Severity notes:

* **It is silent.** Both `simplify(model)` and the `onnxsim in.onnx out.onnx`
  CLI default to `check_n=0`, so nothing verifies the result.  With `check_n=3`
  onnxsim does catch it (it raises while running the check).
* The zero-dim `Reshape`-fusion bug is **not new** — onnxsim 0.7.0 from PyPI
  mis-simplifies the two-consecutive-`Reshape` repro identically.  What is new
  on this branch is that the default pipeline now *creates* that node pair on
  every transformer linear layer, so ordinary models now trip over it.
* `simplify(..., skipped_optimizers=["fuse_matmul_add_bias_into_gemm_batched"])`
  or `["eliminate_consecutive_idempotent_ops"]` both avoid it.

## Finding 2 — a one-node rewrite costs the model its Attention fusion

Comparing against `sim_nogemm` (the variant that still runs):

| variant | nodes | fused ops ORT can recover |
| --- | --- | --- |
| `raw` | 74 | `Attention=2, BiasGelu=2, SkipLayerNormalization=4` |
| `sim` (default) | 68 | model does not load |
| `sim_nogemm` | 72 | `BiasGelu=2, SkipLayerNormalization=4` |
| `sim_safe` (both culprit passes skipped) | 73 | `Attention=2, BiasGelu=2, SkipLayerNormalization=4` |

`--bisect` names both culprits on its own:

```
optimizer passes that cost the model a downstream fusion:
  fuse_matmul_add_bias_into_gemm_batched  recovers BiasGelu=2, SkipLayerNormalization=4
  fuse_consecutive_unsqueezes             recovers Attention=2
```

(the first "recovers" everything because skipping it is what makes the model
load at all — Finding 1).  With both skipped, `ort(sim_safe)` reaches ORT's
ceiling exactly: 16 nodes, `Attention=2, BiasGelu=2, SkipLayerNormalization=4`,
one node *fewer* than `ort(raw)`'s 17 — simplification and fusability are not
actually in conflict here, the default pass set just gives up the fusion for
nothing.  The same holds for the decomposed-LayerNorm export
(`bert_layer_decomposed_ln`, 106 nodes): `ort(sim_safe)` again lands on 16 nodes
with the full fused census, so onnxsim's rewrites do not hurt ORT's
`LayerNormFusion`.

The entire difference is `fuse_consecutive_unsqueezes` (an onnxsim-specific
pass, `onnxsim/passes/fuse_consecutive_unsqueezes.h`).  HuggingFace's attention
mask is exported as

```
mask[B,S] → Unsqueeze → Unsqueeze → Cast → Sub(1-m) → Mul(-10000) → Add(scores)
```

and ORT's `FusionAttention` walks exactly
`["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"]` back to the 2-D mask input to
build the `mask_index` input of `com.microsoft.Attention`
(`onnxruntime/transformers/fusion_attention.py`).  Merging the two `Unsqueeze`
nodes into one saves **one node out of 74** and costs **two fused attention
kernels**.

The `--bisect` mode of the survey script finds this automatically: it re-runs
`simplify` with one more pass skipped each round (greedily, because some fusions
are gated on others — attention fusion anchors on a `LayerNormalization` node)
and reports the passes whose removal restores a fusion.

## Finding 3 — already-fused operators *are* preserved

Good news, and it is worth keeping it that way with a regression test.  onnxsim
leaves fused compute-dominant ops alone in both domains:

| fixture | `raw` | after `simplify` |
| --- | --- | --- |
| `fused_onnx_attention` (ai.onnx opset 23) | `Attention=1, RMSNormalization=1, RotaryEmbedding=2` (17 nodes) | unchanged, 15 nodes |
| `fused_contrib_mha` (`com.microsoft`) | `MultiHeadAttention=1, SkipLayerNormalization=1, FastGelu=1` (16 nodes) | unchanged, 18 nodes |
| `sim(ort(raw))` on the BERT fixture | — | `Attention=2, BiasGelu=2, SkipLayerNormalization=4` kept, 17 → 16 nodes |

Shape inference flows through them (onnxsim registers the `com.microsoft`
contrib schemas), constant folding does not try to evaluate them, and the opset
23 `Attention` is not inlined into its schema-defined function body.

Two caveats:

* On `fused_contrib_mha` the node count goes **up**, 16 → 18: the batched-Gemm
  rewrite converts three `MatMul+Add` pairs into `Reshape+Gemm` and ORT then
  fuses fewer `Gemm`s of its own (`Gemm=4` after `ort(sim)` vs `Gemm=6` from the
  raw model's own session-level optimization).
* onnxsim's shape inference emits `The model contains custom operator(s)
  ['SimplifiedLayerNormalization'] in the default ONNX domain` for ORT-fused
  decoder models — ORT writes some contrib ops without a domain.  They are
  passed through unchanged, which is the right behaviour.

## Finding 4 — onnxsim rejects ONNX Runtime's own optimized models

Feeding an ORT-optimized model straight back into onnxsim fails outright:

```
RuntimeError: No opset import for domain 'com.microsoft'
==> Context: Bad node spec for node. Name: Attention_0 OpType: Attention
```

ORT's transformer optimizer emits `com.microsoft` nodes **without** adding the
matching entry to `opset_import` (verified: the fused model's `opset_import` is
`[("", 17)]` while its nodes use both `""` and `"com.microsoft"`).  That model is
invalid ONNX, so onnx's checker — and therefore onnxsim — rejects it, even
though ORT itself loads it happily.

`optimize → simplify` is a natural pipeline (fuse what ORT knows, then clean up
around it), and it is currently blocked by one missing line of metadata.  onnxsim
already registers `com.microsoft` schemas internally, so tolerating (or
auto-adding) an opset import for a domain whose schemas it knows would unblock
it; the survey script works around it in `ensure_opset_imports()`.

## Finding 5 — decoder models: there is no attention re-fusion to lose (yet)

For the LLaMA-style fixture (RMSNorm + rotary embeddings + SwiGLU), ORT recovers
only the norm fusions, both before and after onnxsim:

| variant | fused ops |
| --- | --- |
| `ort(raw)` | `SimplifiedLayerNormalization=1, SkipSimplifiedLayerNormalization=4` |
| `ort(sim)` | `SimplifiedLayerNormalization=1, SkipSimplifiedLayerNormalization=4` |
| `sim(ort(raw))` | same, preserved |

No `MultiHeadAttention`/`GroupQueryAttention` appears in either — ORT's
`FusionRotaryAttention` needs the rotary embedding to already be a
`com.microsoft.RotaryEmbedding` node, and `FusionRotaryEmbeddings` only builds
one from (a) a torch-exported local *function* named `RotaryEmbedding`, or
(b) a decomposed RoPE that still carries its **dynamic shape scaffolding**
(`Slice` bounds fed by `Unsqueeze ← Div ← Gather ← Shape`, cos/sin sliced by a
`position_ids` gather).  A plain export with constant slice bounds matches
neither.

Two consequences worth flagging, both untested here and worth a follow-up:

* Case (a) is directly at risk from `inline_functions=True` — inlining the
  `RotaryEmbedding` function is exactly what removes ORT's handle on it.
* Case (b) is directly at risk from onnxsim's constant folding, whose whole job
  is to collapse `Shape → Gather → Div → Unsqueeze` chains into constants.

So for decoder models the current answer is "onnxsim costs nothing because there
is nothing to lose", and that stops being true as soon as the export carries
either form.

## Finding 6 — onnxsim cannot re-fuse anything itself

onnxsim's pass set only ever *removes* or *flattens*: there is no pass that
recognises a decomposed block and emits a fused operator.  The nine
onnxsim-specific passes in `onnxsim/passes/` are conv/bias/reshape rewrites;
onnx-optimizer has no attention, layer-norm or GELU fusion.  Every fused op seen
in this survey came either from the exporter or from ONNX Runtime.

That is a gap worth closing *in standard ONNX*, not in a vendor domain: opset 17
has `LayerNormalization`, opset 20 has `Gelu`, opset 23 has `Attention`,
`RMSNormalization` and `RotaryEmbedding`.  Emitting those helps every runtime,
not just ORT, and it makes the pattern-fragility problem moot — a model that
already says `Attention` cannot have its `Attention` pattern broken.

onnxsim already has the machinery: `function_rewrite_rules` takes data-only
`(pattern, replacement)` `FunctionProto` pairs and runs them inside the fixed
point (`onnxsim/function_rewriter.cpp`), and `custom_rewriter` takes arbitrary
Python.  The simple shapes (erf-GELU → `Gelu`, decomposed RMSNorm →
`RMSNormalization`) fit the current matcher; full attention needs attribute
derivation (`num_heads` from a reshape constant) and so needs either the
Python rewriter or a matcher extension.

---

## Recommendations, in priority order

1. **Fix the correctness bug (Finding 1).**  Guard
   `eliminate_consecutive_idempotent_ops` for `Reshape`: only compose when the
   second shape tensor is a constant with no `0` entries (or when
   `allowzero=1`).  onnxsim already shadows nine onnx-optimizer passes via
   `RegisterOrReplace` in `onnxsim/custom_optimizer_passes.cpp`, so a guarded
   copy fits the existing pattern; the fix belongs upstream too.
2. **Add a regression test that the fixtures still run** (`check_n=3` over the
   transformer fixtures).  The existing suite never exercised a `Reshape` with
   `0` dims after a batched-Gemm rewrite.
3. **Treat downstream fusability as a first-class metric.**  `--bisect` in
   `bench/refusion_survey.py` already computes it; wiring "ORT recovers the same
   fused-op census before and after `simplify`" into CI would have caught
   Finding 2 the day the pass landed.
4. **Reconsider `fuse_consecutive_unsqueezes` (and the batched-Gemm pass) in the
   default set.**  Both trade a node or two for a rewrite the runtime would
   rather do itself — ORT does its own `MatMul+Add → Gemm` at session level
   anyway (`Gemm=12` appears in the session-level census of the *raw* model).
   Either drop them from the defaults, or gate them behind a
   `--target-runtime`-style switch for users who really want a Gemm-centric
   graph.
5. **Accept ORT-optimized models** (Finding 4): auto-add an opset import for a
   custom domain whose schemas onnxsim has registered, instead of failing.
6. **Ship re-fusion rules** (Finding 6), starting with the two that fit the
   existing `FunctionProto` rewriter (erf-GELU → `Gelu`, decomposed RMSNorm →
   `RMSNormalization`) and measuring the payoff with this harness before
   attempting `Attention`.

## What the GPU run adds

Everything above is a *graph* measurement; on a CPU the fused kernels are worth
little, and the latency column in the CPU survey is dominated by noise (a shared
container, ±40 % between repeats).  On an NVIDIA GPU the same census turns into
real time: the CUDA EP's fused attention kernels (and, in fp16 on Turing, the
TensorRT fused-attention path) are the reason `com.microsoft.Attention` exists.

`bench/refusion_gpu_bench.py` times every variant on the CUDA EP in fp32 and
fp16, and dumps the session-level fused-op census per variant, so the cost of
Finding 2 can be read directly as milliseconds.  See "Running it on a GPU" in
`bench/refusion_gpu_run.sh` — on an RTX 2060 (Turing, sm75, 6 GB) the
`bert-base` preset fits comfortably.

Results from that run are not included here — this survey was produced on a
CPU-only machine.
