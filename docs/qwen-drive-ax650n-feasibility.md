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

## The text/LLM backbone: real export failure, not a config guess

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
recurrence's reference implementation. The newer `torch.export`-based
(`dynamo=True`) exporter was not tried to completion -- it needs `onnxscript`,
not installed in this environment -- so whether ONNX's decomposition-based
exporter handles `aten::diff` where the legacy tracer doesn't is a real,
concrete, cheap next step (`pip install onnxscript` and retry), not answered
here. As it stands with the tooling on hand: **the text backbone does not
export**, full stop, and the earlier "linear attention ops are probably
outside `AX650_SUPPORTED_OPS`" guess doesn't even get the chance to be
checked -- the block is one step earlier, at PyTorch-to-ONNX, not at
ONNX-to-Pulsar2.

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
| text/LLM backbone | **no** -- `aten::diff` unsupported by the legacy exporter | not reached | blocked at PyTorch-to-ONNX; untried: `dynamo=True` + `onnxscript` |
| vision encoder | **yes** | 4 real gaps (`CumSum`, `Mod`, `OneHot`, `Range`) + 1 known-solved (`Neg`) | closest to viable, still needs 4 new legalization rules |

This sharpens, and partly reverses, the earlier config-only answer. The raw
weight-memory arithmetic there (INT8 ~4.23 GiB plausible against the AX650N's
~6.875 GiB CMM, INT4 ~2.11 GiB comfortable) was never the real question --
two of the three components can't even be exported to ONNX today, by
completely different mechanisms (missing model code vs. an unsupported
PyTorch op), and the one that does export needs four new op-coverage rules
that don't exist yet. "Does it fit" doesn't have a meaningful answer until
those are addressed; weight size was never close to being the binding
constraint.

## What would actually move this forward, cheapest first

1. `pip install onnxscript` and retry the text backbone with
   `torch.onnx.export(..., dynamo=True)` -- five minutes, answers whether
   `aten::diff` is a legacy-exporter gap or a real one.
2. Trace `CumSum`/`Mod`/`OneHot` back to the exact vision-tower source lines
   that emit them, to know whether they're avoidable by a different
   windowing choice (same spirit as this project's own `avgpool_ceil_to_floor`
   and `rank0_to_rank1` legalization rules -- answer a specific compiler
   complaint, don't design a general rule speculatively) or need a genuine
   new `graph_grad`/`legalize.py` rule each.
3. The planning head has no cheap next step -- it needs the actual
   `qwen_drive` modeling code, which is not publicly available anywhere this
   check could reach.
