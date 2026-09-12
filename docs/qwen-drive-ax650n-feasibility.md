# Does Qwen/Qwen-Drive-1.0-4B fit the AX650N?

A real-export, real-op-coverage check of
[`Qwen/Qwen-Drive-1.0-4B`](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B),
following up on a config-only answer that flagged two suspected blockers
without verifying them. This checks those two, and a third the config alone
could not answer at all. Random-init only (`AutoModel.from_config`, never
`from_pretrained` with real weights) -- the question here is exportability
and op coverage, not numerical correctness, so no checkpoint download is
needed. No AX650N/VM/Docker touched; this is a host-only export +
`AX650_SUPPORTED_OPS` diff, the same method
`docs/axera-audio-speech-op-coverage.md` used for Whisper/wav2vec2.

The model has three components. They are answered separately -- they are not
close enough in readiness for one verdict to be honest.

## The planning head (diffusion expert): cannot be instantiated at all

`config.json`'s outer `model_type` is `qwen_drive`
(`architectures: ["QwenDriveForPlanning"]`), with the planning head itself a
nested `qwen_drive_planning_expert` config. Neither is present in
`transformers==5.17.0` (the latest release on PyPI as of this check --
confirmed via `pip index versions transformers`):

```
ValueError: The checkpoint you are trying to load has model type
`qwen_drive` but Transformers does not recognize this architecture.
```

The HF repo ships no custom modeling code either -- its file listing has no
`modeling_qwen_drive.py`/`configuration_qwen_drive.py`, and `config.json` has
no `auto_map` pointing anywhere. So the class that actually runs this model
does not exist in any publicly released `transformers` version and is not
shipped alongside the checkpoint. There is currently no way to instantiate,
let alone export or compile, this component outside whatever internal/dev
environment produced it. This settles nothing about the diffusion-loop
static-graph question raised in the earlier config-only answer -- the
question is moot until the code exists somewhere reachable.

## The text/LLM backbone: exports with `dynamo=True`, five real op gaps

The backbone (`vlm_config.text_config`, `model_type: qwen3_5_text`) *is*
real and released -- `Qwen3_5TextModel`, confirmed instantiable and runs a
real forward pass on CPU. Its "hybrid attention" is concrete, not a vague
label: `layer_types` alternates three `linear_attention` layers then one
`full_attention` layer (`full_attention_interval: 4`), and the linear layers
are **Gated DeltaNet** (`chunk_gated_delta_rule`), a specific gated
linear-attention/SSM formulation -- confirmed from the model's own runtime
log:

```
`causal_conv1d_fn` is falling back to its reference PyTorch implementation
because `causal_conv1d` is not installed.
`chunk_gated_delta_rule` is falling back to its reference PyTorch
implementation because `flash-linear-attention` is not installed.
```

Both fall back to a pure-PyTorch reference path automatically (no optimized
kernel installed here), which is what makes it traceable at all. It is not,
however, ONNX-exportable through the standard legacy exporter:

```
UnsupportedOperatorError: Exporting the operator 'aten::diff' to ONNX
opset version 18 is not supported
```

`aten::diff` (discrete difference) shows up inside the Gated DeltaNet
recurrence's reference implementation.

**Update: `dynamo=True` clears `aten::diff` entirely -- it was a
legacy-exporter gap, not a real one.** `pip install onnxscript` and
`torch.onnx.export(..., dynamo=True)` on the identical model/inputs never
even mentions `aten::diff` -- the newer `torch.export`-based exporter
decomposes it into ONNX-expressible ops before op translation, where the
legacy TorchScript tracer just gave up. It does hit one *different*, earlier
blocker first: `torch.export` requires every input/output to be a
pytree-registered type, and `Qwen3_5TextModel`'s default `use_cache=True`
puts a `transformers.cache_utils.DynamicCache` in the output, which isn't
registered -- `RuntimeError: Found <class
'transformers.cache_utils.DynamicCache'> in output, which is not a known
type`. Trivial workaround, not a real architectural constraint: wrap the
model and call it with `use_cache=False` (a training step doesn't want an
incremental-decode cache anyway -- this is exactly the same "training
doesn't need KV-cache decode" point already made about the `pulsar2 build`
vs. `pulsar2 llm_build` path choice elsewhere in this project's docs). With
that wrapper, `dynamo=True` **exports cleanly end to end** -- graph capture,
decompositions, ONNX translation, and the exporter's own graph optimizer all
report success.

The real op-coverage question is now answerable, and it looks a lot like the
vision encoder's: real, but small. The raw export is 3124 nodes over 31
unique op types; `onnxsim.simplify()` (with its own numeric check passing)
takes it to 2917 nodes -- a much smaller reduction than the vision encoder's
529-&gt;210, because most of this graph is Gated DeltaNet's actual chunked-scan
arithmetic (`Transpose`/`Slice`/`Gather`/`ScatterElements`/`ScatterND`/`ReduceSum`
in the hundreds each), not shape scaffolding, so there's little scaffolding
for `simplify()` to remove:

| | before `simplify()` | after `simplify()` |
| --- | --- | --- |
| nodes | 3124 | 2917 (check_ok=True) |
| ops outside `AX650_SUPPORTED_OPS` | `CumSum`, `IsNaN`, `Neg`, `Reciprocal`, `Trilu` | same five, all survive |

All five are load-bearing (they survive constant-folding/DCE, same standard
the vision-encoder check used to separate real gaps from scaffolding):

- `Neg` -- already solved, `legalize.py`'s existing `neg_to_mul` rule (same
  one the vision encoder needs).
- `CumSum` -- the same op the vision encoder's windowed-attention indexing
  also needed; now confirmed to show up in a second, unrelated architecture
  (Gated DeltaNet's chunked recurrence uses cumulative sums over the chunk
  dimension), which raises its priority as a legalization rule worth writing
  once, not a one-off.
- `IsNaN` -- likely a numerical-stability guard somewhere in the reference
  fallback kernel (not traced to the exact line); this is the same op family
  the audio-speech survey (`docs/axera-audio-speech-op-coverage.md`) flagged
  as having no `graph_grad` backward rule for wav2vec2's masking, though here
  the question is forward op coverage on the NPU, not backward differentiability.
- `Reciprocal` -- a new gap, not seen in any prior model checked in this
  project. `Div` is covered by `AX650_SUPPORTED_OPS`, so a `Reciprocal(x)` ->
  `Div(1, x)` rewrite is a plausible cheap legalization if this turns out to
  matter, but that is a guess, not something tried here.
- `Trilu` -- almost certainly the causal-attention mask's triangular
  constant construction (`full_attention` layers need a causal mask; `Trilu`
  is the standard ONNX op for building one). If it is applied to a
  compile-time-constant shape, this may fold to a plain initializer under a
  different simplify configuration or opset; not checked here.

So: **the text/LLM backbone exports (with a one-line, semantically-irrelevant
`use_cache=False` wrapper), and needs the same order of new legalization work
as the vision encoder -- roughly four to five new/shared rules, not a
fundamental blocker.** The original "linear attention ops are probably
outside `AX650_SUPPORTED_OPS`" guess turns out to be not quite right either:
none of the five gap ops are exotic linear-attention-specific primitives --
`chunk_gated_delta_rule`'s actual arithmetic (`MatMul`, `Sigmoid`, `Softplus`,
`Mul`/`Add`/`Sub`) is entirely within `AX650_SUPPORTED_OPS`. The gaps are in
its indexing/bookkeeping (`CumSum`, `Reciprocal`, `Trilu`) and a stability
guard (`IsNaN`), the same *category* of gap the vision encoder has, not a
deeper architectural mismatch.

## The vision encoder: exports cleanly, four real op gaps survive `simplify()`

`vlm_config.vision_config` (`model_type: qwen3_5_vision`) is a Qwen2-VL-style
ViT: patchified `hidden_states` + a `grid_thw` grid-shape input, 24 blocks,
`gelu_pytorch_tanh` activation. This one **exports successfully** through the
legacy tracer (529 nodes, 35 unique op types) and, unlike the config-only
guess assumed, is worth simplifying before judging -- `onnxsim.simplify()`
was run on the raw export, the same step every other model in this project's
training work goes through before a coverage verdict is drawn:

| | before `simplify()` | after `simplify()` |
| --- | --- | --- |
| nodes | 529 | 210 |
| ops outside `AX650_SUPPORTED_OPS` | `CumSum`, `Mod`, `Neg`, `OneHot`, `Range`, `Shape` | `CumSum`, `Mod`, `Neg`, `OneHot`, `Range` |

`Shape` was pure scaffolding and folded away, as expected once `grid_thw` is
a compile-time constant (the export's own `TracerWarning`s already say the
grid shape gets baked in as a constant -- true for one fixed image
resolution, and the reason a real deployment would need to re-export per
input shape, same caveat every static-shape NPU compile in this project's
history carries). The other four did **not** fold away -- they survive
`simplify()`'s constant-folding and dead-code-elimination, meaning they are
load-bearing computation, not shape scaffolding. `CumSum`/`Range`/`Mod` most
likely come from Qwen2-VL-style windowed-attention index computation (window
boundary/rotary-position bookkeeping); this was not traced further op-by-op
back to source lines, so treat that attribution as a reasonable guess, not a
confirmed one.

`Neg` is not a new problem -- `scripts/axera/legalize.py` already has a
`neg_to_mul` rule for exactly this (see `docs/axera-on-device-training-handoff.md`'s
rules table: "`Neg` is the one backward-pass op off the AX650 list"). The
other three (`CumSum`, `Mod`, `OneHot`) are genuinely new findings, not
covered by anything in this project's existing legalization rules.

## Verdict, componentized (not blended)

| component | exports? | AX650 op coverage | verdict |
| --- | --- | --- | --- |
| planning head (diffusion expert) | **no** -- code doesn't exist in any released `transformers` | N/A | blocked entirely, upstream of any AX650 question |
| text/LLM backbone | **yes**, with `dynamo=True` + a `use_cache=False` wrapper | 5 real gaps (`CumSum`, `IsNaN`, `Neg`, `Reciprocal`, `Trilu`), all fixable-shaped | viable in principle, needs ~4-5 new legalization rules |
| vision encoder | **yes** | 4 real gaps (`CumSum`, `Mod`, `OneHot`, `Range`) + 1 known-solved (`Neg`) | closest to viable, still needs 4 new legalization rules |

This sharpens, and this time strengthens rather than reverses, the picture:
**two of the three real (i.e. code-exists) components now export and land on
the same kind of gap** -- a handful of indexing/bookkeeping ops
(`CumSum`/`Mod`/`OneHot`/`Range`/`Reciprocal`/`Trilu`) plus one
already-solved op (`Neg`), not a deep architectural mismatch and not the
exotic-linear-attention-op blocker the original config-only answer guessed
at. `CumSum` recurring in *both* independently-checked components (windowed
vision attention and the DeltaNet chunked recurrence) is the strongest signal
in this whole investigation that it's worth writing one shared legalization
rule rather than two one-offs. The planning head remains fully blocked, for
a reason no amount of export-tooling cleverness fixes: the code to even
instantiate it does not exist anywhere reachable.

The raw weight-memory arithmetic from the original config-only answer (INT8
~4.23 GiB plausible against the AX650N's ~6.875 GiB CMM, INT4 ~2.11 GiB
comfortable) still hasn't been the binding constraint at any point in this
investigation -- every blocker found so far has been at the export/op-coverage
layer, not memory.

## What would actually move this forward, cheapest first

1. **Done, this update**: `dynamo=True` + `onnxscript` clears `aten::diff`
   for the text backbone; a `use_cache=False` wrapper clears the
   `DynamicCache` pytree issue underneath it. Both were cheap (five minutes
   each) and both worked.
2. Write the `CumSum` legalization rule first, since it's now confirmed
   load-bearing in two unrelated components -- highest leverage of anything
   found across this whole investigation. Then `Mod`/`OneHot`/`Range`
   (vision) and `Reciprocal`/`Trilu`/`IsNaN` (text backbone), each following
   this project's established discipline of tracing back to the exact
   compiler complaint before generalizing a rule (see `legalize.py`'s
   `avgpool_ceil_to_floor`/`rank0_to_rank1` for the pattern) -- none of the
   five new ops here were traced to exact source lines in this pass, so that
   tracing is real remaining work, not a rubber stamp.
3. The planning head has no cheap next step -- it needs the actual
   `qwen_drive` modeling code, which is not publicly available anywhere this
   check could reach.
