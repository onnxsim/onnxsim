# Op coverage for training audio/speech models on the AX650N: a survey

Companion to `docs/axera-on-device-training-handoff.md`, which got a resnet18
fine-tuning step running and fast on real hardware. This asks a narrower
question before anyone spends a session on it: for a *speech* model, which
architecture families does today's pipeline (`onnxsim.graph_grad.build_backward`
+ `scripts/axera/legalize.py`'s `TRAINING_RULES`) actually reach, and which
have a real, specific gap? Nothing here touched the AX650N or Pulsar2 Docker
-- it is a static code-coverage check plus real ONNX exports of three
architecture families, cross-referenced against the two things that decide
trainability on this hardware: `onnxsim.graph_grad._RULES` (what
`build_backward` can differentiate) and `scripts/axera/pulsar2_ops.py`'s
`AX650_SUPPORTED_OPS` (what the NPU can run at all).

## Corrections to a first-pass guess

Before this survey, the following was assumed from memory rather than the
code. All four held up, with one real refinement:

- **Conv1D is supported** by `act_weight_conv_to_matmul` -- confirmed by
  reading the rule and by `tests/test_axera_training_legalize.py`'s existing
  `test_a_live_weight_convolution_becomes_matmuls_in_1d_and_2d`, which already
  exercises `nd=1` (kernel 3, stride 1, pad 1) and checks it against
  onnxruntime at >100 dB SNR. **Refinement**: both this rule and the newer,
  faster `_linearize_trainable_convs` (`scripts/axera/build_resident_train_step.py`,
  landed in PR #1336) require `group=1, dilation=1` -- "1-D and 2-D, any
  stride" was accurate but incomplete; grouped/dilated convolutions of any
  rank are declined outright by both. See "The depthwise-conv gap" below --
  this is the one place the first pass undersold a real limitation.
- **Softmax and LayerNormalization are both differentiable** (`_grad_softmax`,
  `_grad_layer_normalization` in `onnxsim/graph_grad.py`'s `_RULES` dispatch
  table) **and NPU-executable** (both names are in `AX650_SUPPORTED_OPS`) --
  confirmed by reading both tables directly.
- **The raw ONNX `Gelu` op has no backward rule** -- confirmed: absent from
  `_RULES`, and `legalize.py` has no decompose-to-`Erf` rule for it either
  (`grep -rn "Gelu" scripts/axera/legalize.py onnxsim/graph_grad.py` -- no
  hits). `Gelu` itself *is* in `AX650_SUPPORTED_OPS`, so it runs fine at
  inference; it just cannot appear on a path back to a trainable weight
  today. See "The Gelu gap" below for why this turned out not to matter for
  the models actually inspected.
- **LSTM has no backward rule**, confirmed two ways: `LSTM` is absent from
  `_RULES`, and a real `torch.onnx.export` of a plain `torch.nn.LSTM` (opset
  17) emits it as a single opaque `LSTM` node, not a decomposed
  gate-by-gate graph -- so there is no way to route around the missing rule
  by legalizing it into already-covered ops, the way `act_weight_conv_to_matmul`
  routes a `Conv` into `MatMul`s. `LSTM` **is** in `AX650_SUPPORTED_OPS` (it
  runs fine at inference); `GRU` is in neither table. Classic RNN-based
  ASR/TTS models are not trainable through this pipeline today, full stop.

## Three real architectures, real exports

Random-initialized, opset-17 ONNX exports via `torch.onnx.export` (no
pretrained weights needed or fetched -- only architecture shape matters
here), built with tiny configs (`hidden_size`/`d_model` 32, 1-2 layers) purely
to get a real op graph rather than reason from documentation. Every op type
below is cross-checked against `onnxsim.graph_grad._RULES`'s 28 keys.

### Whisper encoder -- clean, would build today

`transformers.WhisperModel(...).get_encoder()`, 128 nodes:

| op types present | in `_RULES`? |
| --- | --- |
| `Add`, `Identity`, `MatMul`, `Mul`, `Transpose`, `Reshape`, `LayerNormalization`, `Div`, `Erf`, `Conv`, `Softmax` | yes, all of them |
| `Constant` | needs no gradient (a literal, not a function of any input) |

**Every node type in a real Whisper-encoder export already has a backward
rule.** No `Where`, no mask-construction ops at all -- Whisper's encoder has
no variable-length attention masking (it always processes a fixed 30 s /
3000-frame window, padded upstream), which is exactly what keeps this graph
simple. `Erf` is GELU's decomposed form -- `transformers`' export already
emits the Erf-based GELU here, not the fused `Gelu` op, so the Gelu gap above
does not block this model in practice.

### wav2vec2 -- mostly clean, one open question

`transformers.Wav2Vec2Model(...)`, 228 nodes:

| op types present | in `_RULES`? |
| --- | --- |
| `Add`, `Mul`, `Identity`, `MatMul`, `Transpose`, `Reshape`, `Div`, `Erf`, `Conv`, `LayerNormalization`, `Softmax`, `InstanceNormalization` | yes |
| `Where`, `IsNaN`, `GreaterOrEqual`, `Equal`, `Expand`, `Slice`, `Cast`, `ConstantOfShape`, `Shape`, `Unsqueeze` | **no** |
| `Constant` | needs no gradient |

The second row is real but, on inspection, looks like attention-mask/padding
construction from the `attention_mask` input (`Where`/`IsNaN` guard against
`-inf` propagating through softmax on fully-masked rows, `Equal`/`GreaterOrEqual`
build the boolean mask itself) -- a subgraph that depends only on the
(non-trainable) input mask, not on any trainable weight, and so should never
need differentiating if the trainable region is a tail of encoder layers
downstream of where the mask is already folded in as an additive bias.
**This is not proven here** -- it is a static op-list argument, not a traced
backward slice. The concrete follow-up: pick a trainable tail (e.g. the last
1-2 transformer layers, same recipe as resnet18's last-four-layers), run
`build_backward` over just that slice, and confirm it never needs to touch
the `Where`/mask subgraph. If the trainable slice is chosen the same way the
resnet18 work chose its trainable tail (downstream of everything
mask-related), this is very likely fine, but "likely" is doing real work in
that sentence until someone runs it.

### Wav2Vec2-Conformer -- two real, specific gaps

`transformers.Wav2Vec2ConformerModel(...)`, 231 nodes. Same `Where`-family
caveat as wav2vec2 above, plus two new, concrete blockers:

- **A real depthwise `Conv`.** The conformer block's `conv_module` has
  `/encoder/layers.0/conv_module/depthwise_conv/Conv` with **`group=32`**
  (fully depthwise: `groups == channels`) sitting right next to two ordinary
  `group=1` pointwise convs. If this convolution's weight were made
  trainable, **neither `act_weight_conv_to_matmul` nor
  `_linearize_trainable_convs` has any path for it** -- both decline
  `group != 1` outright and fall through unlegalized, which would fail at
  Pulsar2 build time exactly the way an un-legalized live-weight `Conv` did
  for resnet18 before that rule existed. See "The depthwise-conv gap" below.
- **`Split`, with no backward rule.** The conv module's gating
  (`Split` + `Sigmoid` + `Mul`, a GLU) uses an op type absent from `_RULES`.
  `Sigmoid` and `Mul` are both covered; `Split` alone is the gap. Unlike the
  depthwise conv, this would be a small addition -- `Split`'s gradient is
  just the reverse operation, `Concat` of the incoming per-output gradients
  back along the same axis, which is itself already an op Pulsar2's own
  compiler emits from `act_weight_conv_to_matmul`'s tap-fusion (`fuse=True`),
  so it is at least a known-compilable op type on this hardware, just not
  yet a registered gradient rule.

## What actually blocks each family, ranked by cost to fix

| gap | blocks | fix, roughly |
| --- | --- | --- |
| Raw `Gelu` has no backward rule | only an exporter that emits fused `Gelu` instead of decomposed Erf-GELU (neither Whisper's nor wav2vec2's `transformers` export does this) | cheapest: a `legalize.py` rule decomposing `Gelu` into `Mul`/`Add`/`Erf`/`Div`-by-constant, all already covered |
| `Split` has no backward rule | Conformer's GLU gating only (not wav2vec2, not Whisper) | small: gradient of `Split` is `Concat` of the per-output gradients along the same axis |
| Grouped/depthwise `Conv` has no forward-legalization path | Conformer's `conv_module` only (not wav2vec2, not Whisper -- neither uses a grouped conv anywhere) | real but bounded: `onnxsim.graph_grad._grad_conv`'s own docstring states groups "cost nothing extra" for the *backward* im2col-as-gather identity it already uses (the group axis splits out of the channel axis via reshape, and `MatMul` batches over it) -- the same generalization applied to `_linearize_trainable_convs`'s forward-side use of that identity (PR #1336) would very likely close this, since the underlying primitive already handles groups; it is un-extended, not fundamentally blocked |
| `Where`/mask-construction ops have no backward rule | *maybe* wav2vec2 and Wav2Vec2-Conformer, if a trainable tail is chosen upstream of where the mask is folded in -- unconfirmed | not a fix, a design constraint: choose the trainable region downstream of all mask handling, the same way resnet18's trainable tail was chosen downstream of the frozen stem |
| LSTM/GRU have no backward rule at all, and the real ONNX op is opaque (no legalization route around it) | any classic RNN-based ASR/TTS model, full stop | the largest lift here -- would need either a dedicated LSTM-cell backward rule (the four gates are themselves ordinary `MatMul`/`Sigmoid`/`Tanh` arithmetic once unrolled, so the primitives exist) or accepting only models that unroll their own recurrence in ONNX rather than emitting the fused `LSTM` op |

## Do the two silent vendor bugs generalize?

The resnet18 work found two silent Pulsar2 bugs and fixed both generically
enough that a new architecture should not need to rediscover them:

- **Bare `ReduceMean` only reduces the last axis on this hardware.**
  `graph_grad`'s own `_grad_layer_normalization` and
  `_grad_instance_normalization` always emit `ReduceMean` with an explicit
  `axes=` attribute (confirmed: `grep -n "ReduceMean" onnxsim/graph_grad.py`
  shows every occurrence passing `axes=...`) -- so anything `graph_grad`
  itself emits for LayerNorm/InstanceNorm's backward is safe by
  construction. This is a *coding discipline* the module follows, not an
  automatic guard: a source model's own *forward* graph could still contain
  a bare `ReduceMean` somewhere outside a fused norm op (an ad hoc pooling
  head, say), which would need catching on a case-by-case basis the way the
  resnet18 handoff caught it -- nothing here makes that check unnecessary
  for a new model, it just confirms the tooling itself doesn't reintroduce
  the bug.
- **A constant bias lets Pulsar2 reconstruct a `Gemm` we removed.** The fix
  (`legalize._unfusable_bias`, reshaping a bias to `[1, N]`) is already a
  general helper used inside both `gemm_to_matmul` and
  `act_weight_conv_to_matmul`, which any model runs through via
  `TRAINING_RULES` regardless of architecture -- confirmed by reading both
  call sites. Attention's `MatMul`+bias projections should already be
  protected the same way resnet18's classifier head was, with no new work.

## Recommendation: smallest real next trial

**Whisper's encoder**, trainable tail = last 1-2 transformer layers (same
shape as resnet18's "last four layers," frozen stem + trainable tail). It is
the cleanest of the three surveyed: full op coverage confirmed with zero open
questions (no `Where`-family uncertainty to resolve, no depthwise conv, no
`Split`), and a fixed-length input (no attention-mask handling at all) that
avoids the one thing this survey could not settle statically. wav2vec2 is the
second choice, gated on confirming the `Where`-family ops stay off the
trainable slice's backward path -- worth trying once the Whisper trial is
working, not before. Wav2Vec2-Conformer needs the depthwise-conv and `Split`
gaps closed first and should wait.

## Reproducing the exports

```python
from transformers import WhisperConfig, WhisperModel
cfg = WhisperConfig(vocab_size=51865, num_mel_bins=80, encoder_layers=2,
                     encoder_attention_heads=2, decoder_layers=2,
                     decoder_attention_heads=2, d_model=32, decoder_ffn_dim=64,
                     encoder_ffn_dim=64, max_source_positions=32,
                     max_target_positions=32)
enc = WhisperModel(cfg).eval().get_encoder()
torch.onnx.export(enc, (torch.randn(1, 80, 64),), "whisper_enc.onnx",
                   opset_version=17, dynamo=False)
```

wav2vec2 and Wav2Vec2-Conformer follow the same shape with
`Wav2Vec2Config`/`Wav2Vec2Model` and `Wav2Vec2ConformerConfig`/
`Wav2Vec2ConformerModel`; `conv_depthwise_kernel_size=31` on the Conformer
config is what produces the `group=32` node above. None of these need
pretrained weights -- random initialization is enough to get a real op graph,
which is all this survey needed.
