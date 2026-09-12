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

### wav2vec2 -- one gap, now confirmed real (not "maybe")

`transformers.Wav2Vec2Model(...)`, 228 nodes:

| op types present | in `_RULES`? |
| --- | --- |
| `Add`, `Mul`, `Identity`, `MatMul`, `Transpose`, `Reshape`, `Div`, `Erf`, `Conv`, `LayerNormalization`, `Softmax`, `InstanceNormalization` | yes |
| `Where`, `IsNaN`, `GreaterOrEqual`, `Equal`, `Expand`, `Slice`, `Cast`, `ConstantOfShape`, `Shape`, `Unsqueeze` | **no** |
| `Constant` | needs no gradient |

The first pass of this survey guessed the `Where`/`IsNaN` row was probably
confined to attention-mask *construction* -- upstream of, and excludable
from, any trainable tail chosen downstream of it -- and flagged that as
unproven. **It is disproven.** Tracing the real export (`onnx.load` +
following each `Where`/`IsNaN` node's producer/consumer edges, not just its
op-type membership) shows each of the two encoder layers has its own,
per-layer `attn_weights = Where(IsNaN(attn_weights), 0, attn_weights)`
sitting **directly between that layer's own `Softmax` and its own
`MatMul` with `V`** --
`/encoder/layers.{i}/attention/IsNaN` -> `/encoder/layers.{i}/attention/Where_1`
-> `MatMul`. This is numerical-stability cleanup (a fully-masked row's
softmax is uniform, not `NaN`, until floating-point cancellation makes it one
in practice -- HF's own attention implementations guard against exactly
this), and it is baked into *every* layer's own forward computation, not
shared or hoisted out. So there is no way to choose a trainable tail that
excludes it: even "the last layer only" (`V`'s projection, the output
projection) needs a gradient through its own layer's `Where`, because that
`Where`'s input and output are both fresh intermediates computed inside that
layer -- unlike an external bias tensor merely *added* to the scores, there
is nothing outside the layer to treat as an opaque boundary input instead.
**wav2vec2 (the plain, non-Conformer model) cannot train through any tail
that includes attention output until `Where` has a backward rule.** That
rule is not hard -- `Where`'s gradient is `dX = mask * g`, `dY = (1-mask) *
g` with `mask = Cast(condition, FLOAT)`, i.e. the same "boolean condition
becomes a float 0/1 multiplied in, never re-emitted as `Where` itself"
convention `graph_grad`'s own module docstring already states as a rule for
every hand-written gradient here -- but it is real, unimplemented work, not
a design workaround. Left for the next person who wants wav2vec2 itself
(Whisper, below, remains gap-free and is still the better first trial).

### Wav2Vec2-Conformer -- both gaps closed; one different, unresolved question

`transformers.Wav2Vec2ConformerModel(...)`, 231 nodes. Two concrete blockers
found by the first pass, **both fixed and verified on host** (no hardware
needed for either -- both are onnxruntime-checked numerically, the same bar
every other rule in this codebase holds itself to):

- **A real depthwise `Conv`, group-legalization now handled.** The conformer
  block's `conv_module` has `/encoder/layers.0/conv_module/depthwise_conv/Conv`
  with **`group=32`** (fully depthwise: `groups == channels`) beside two
  ordinary `group=1` pointwise convs. `scripts/axera/legalize.py`'s
  `act_weight_conv_to_matmul` now legalizes `group > 1` too: each group's
  own channel slice of the (already-transposed-once) activation and weight
  runs through the exact same per-tap-matmul path as `group=1`, and the
  per-group outputs concatenate back onto the `Cout` axis before the shared
  bias add -- no new op types (`Slice`/`Concat` are already used for
  tap-fusion and were already in this rule's own output vocabulary), and
  `group=1` (resnet18, and every model this rule ran on before) takes the
  exact pre-existing node sequence, unchanged, so nothing already shipped
  moves. Verified against onnxruntime at >100 dB SNR for a fully depthwise
  1-D case (`group == cin == cout`, the literal Conformer shape, both plain
  and strided) and a grouped-not-depthwise 2-D case
  (`tests/test_axera_training_legalize.py::test_a_grouped_live_weight_convolution_becomes_per_group_matmuls`,
  plus a biased-grouped-conv case checking the bias broadcasts correctly
  over the concatenated output). `_linearize_trainable_convs`
  (`scripts/axera/build_resident_train_step.py`, PR #1336 -- open at the
  time of this fix, not yet merged) still declines `group != 1` on purpose
  and falls through to `act_weight_conv_to_matmul` for those convs, so a
  Conformer trial gets full grouped-conv coverage today even before that
  optimization is extended to match; extending it the same way would be a
  pure speed win on top, not a correctness gap.
- **`Split`, backward rule added.** The conv module's gating (`Split` +
  `Sigmoid` + `Mul`, a GLU) used an op type absent from `graph_grad._RULES`.
  `Sigmoid` and `Mul` were already covered; `Split` alone was the gap. Its
  gradient is now `onnxsim.graph_grad._grad_split`, registered in a new
  `_MULTI_OUTPUT_RULES` table (`build_backward`'s per-node dispatch now
  checks this table first) rather than `_RULES`, because a rule for an op
  with more than one *output* genuinely needs more than one incoming
  gradient -- a different argument shape from every other rule in this
  module, not a variant of the existing single-output `Rule` contract.
  Deliberately **not** `Concat` of the incoming gradients (the textbook
  VJP): `Concat` is not in `graph_grad.BACKWARD_OPS` (this module's own
  WebGPU/WebNN/NPU-portable allowlist), and `tests/test_graph_grad.py`'s own
  harness asserts every emitted backward op stays inside it. Instead, each
  output's gradient is right-multiplied by a constant 0/1 selection matrix
  (`E_i = eye(N)[offset_i : offset_i + size_i, :]`) after moving the split
  axis to the last position (`Transpose`, skipped when already there) --
  `MatMul`/`Add`/`Transpose` only, all three already in `BACKWARD_OPS`, so
  no allowlist change was needed at all. `Split` is consequently visible
  through `graph_grad.supported_ops()` (what QAT/LoRA block discovery
  already keys off) but deliberately **not** in the parity-pinned
  `SUPPORTED_OPS` constant, since it has no C++/WASM mirror yet -- a real,
  stated limitation (Python-only today), not an oversight. Verified against
  finite differences for: both outputs consumed (equal sizes), one output
  consumed with the other zero-contributing (uneven sizes, negative axis),
  and a non-last split axis (`tests/test_graph_grad.py::test_rule_matches_finite_differences[split_*]`,
  6 cases).

**What's still open for Conformer, and it is a different situation from
plain wav2vec2's confirmed block above, not the same one:** Conformer's
encoder layers also each have a `Where` node
(`/encoder/layers.{i}/self_attn/Where`), but tracing it shows a different
shape than wav2vec2's -- no `IsNaN` anywhere in the whole 231-node Conformer
export (confirmed by listing every `Where`/`IsNaN`/`Equal`/`GreaterOrEqual`
node in the real export), and this `Where`'s *condition* operand comes from
`/encoder/Expand_1`, computed once outside the per-layer loop and shared,
not recomputed per layer. That is the shape of an additive attention-mask
bias built once and added into each layer's own scores (`Where(shared_cond,
per-layer-scaled-value, 0)` -> `Add` to that layer's scores) -- the pattern
the original first-pass guess was about, and which a trainable tail chosen
downstream of the shared mask-bias tensor (treating it as an opaque external
input to the slice, the same way a `discover_qat_blocks`-style boundary
would) can plausibly route around. **This was not traced end-to-end the way
wav2vec2's was** (time-boxed: the wav2vec2 case already gave a definitive,
generalizable answer -- "does a `Where` ever sit between two internally-
computed tensors with no external tensor to treat as a boundary" -- and
Conformer's shares HF's attention code enough that resolving it precisely
needs its own trace, not an inference from wav2vec2's). Flagged, not closed.

## What actually blocks each family, ranked by cost to fix

| gap | status | blocks | fix |
| --- | --- | --- | --- |
| Grouped/depthwise `Conv` has no forward-legalization path | **fixed** (`scripts/axera/legalize.py`'s `act_weight_conv_to_matmul`) | was: Conformer's `conv_module` | per-group tap-matmul + `Concat` back onto `Cout`, `group=1` path untouched |
| `Split` has no backward rule | **fixed** (`onnxsim/graph_grad.py`'s `_grad_split`, via the new `_MULTI_OUTPUT_RULES` table) | was: Conformer's GLU gating | `MatMul` against a constant 0/1 selection matrix per output, no `Concat`, stays inside `BACKWARD_OPS` |
| `Where`/`IsNaN` numerical-stability cleanup has no backward rule | **confirmed blocking**, not a design workaround | plain wav2vec2, any tail touching attention output -- Conformer's own (differently-shaped) `Where` usage is still unresolved | `dX = Cast(cond, FLOAT) * g`, `dY = (1 - that) * g` -- same "float mask, not `Where` itself" convention this module already uses everywhere else; not yet implemented |
| Raw `Gelu` has no backward rule | open, low priority | only an exporter emitting fused `Gelu` instead of decomposed Erf-GELU (neither Whisper's nor wav2vec2's `transformers` export does this) | a `legalize.py` rule decomposing `Gelu` into `Mul`/`Add`/`Erf`/`Div`-by-constant, all already covered |
| LSTM/GRU have no backward rule at all, and the real ONNX op is opaque (no legalization route around it) | open, largest lift | any classic RNN-based ASR/TTS model, full stop | a dedicated LSTM-cell backward rule (the four gates are themselves ordinary `MatMul`/`Sigmoid`/`Tanh` arithmetic once unrolled) or accepting only models that unroll their own recurrence in ONNX |

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
shape as resnet18's "last four layers," frozen stem + trainable tail), still
stands as the best first trial and is now *more* clearly so than the first
pass thought: full op coverage confirmed with zero open questions, and a
fixed-length input (no attention-mask handling at all) that sidesteps every
`Where`-shaped question this survey has now spent real effort on for the
other two families.

The ranking below it changed. Wav2Vec2-Conformer's conv-module gaps
(depthwise `Conv`, `Split`) are now **closed** -- it is Conformer's
still-open, differently-shaped `Where` question (see above) that gates it
next, not the conv-module work this pass finished. Plain wav2vec2 dropped
from "second choice, likely fine" to **blocked**: its `Where`/`IsNaN`
numerical-stability cleanup is confirmed to sit inside every encoder layer's
own attention computation with no external tensor to route a trainable
tail's boundary around, so it needs `Where`'s backward rule implemented
before any tail including attention output can train, not just a careful
choice of trainable region. Concretely, after Whisper: either implement
`Where`'s backward rule (unblocks wav2vec2 outright, and is the same
"boolean condition, not the op, becomes a float mask" work either way) or
trace Conformer's shared-condition `Where` end-to-end to confirm it is
excludable by block boundary (cheaper if it pans out, but unproven, and
resolves only Conformer -- `Where`'s backward rule resolves both).

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

## A smaller target than Whisper: wav2vec2's CNN feature extractor alone

`docs/axera-on-device-training-handoff.md`'s Whisper section (PRs #1351/
#1357/#1359) found that `whisper-base`'s encoder -- 6 real transformer
layers, LayerNorm + attention + GELU-MLP each -- has a gradient that
underflows an 8-bit quantizer structurally: the true per-step weight
movement is 4-5 orders of magnitude smaller than the weight's own scale,
confirmed by two independently-built calibrations landing on the identical
result. That is not a calibration problem the multi-phase swap technique
(PRs #1355/#1356) can fix. This asks whether a *shallower*, non-transformer
real speech architecture avoids it.

**`Wav2Vec2Model(Wav2Vec2Config()).feature_extractor`** -- the raw-waveform
CNN front-end every wav2vec2/HuBERT/Conformer variant shares, 7 Conv1D
layers (`conv_dim=(512,)*7`, strides `(5,2,2,2,2,2,2)`), 4.2M of the full
model's 94.4M params, no attention, no LayerNorm chain, one
`InstanceNormalization` per layer instead. Real op-coverage check (random
init, `torch.onnx.export`, opset 17): raw export is `{Add, Constant, Conv,
Div, Erf, InstanceNormalization, Mul, Reshape, Shape, Unsqueeze}`, all of
which are in `AX650_SUPPORTED_OPS`; `onnxsim.simplify()` folds `Reshape`/
`Shape` away as pure scaffolding (parallel to the vision-encoder finding
above). Every remaining op except one has a `graph_grad` gradient rule
already (`Add`, `Conv`, `Div`, `Erf`, `InstanceNormalization`, `Mul`).

**The one gap**: `Unsqueeze` (adding the channel axis to the raw waveform
input, `(1,16000) -> (1,1,16000)`) has no entry in `graph_grad.SUPPORTED_OPS`
-- confirmed by a real `UnsupportedOpError` from `build_backward`. It sits
only on the non-trainable input's own path, never between a target weight
and the loss, but `build_backward` walks every node reachable from the loss
regardless (per `onnxsim.qat_graph`'s own docstring: "a gradient for every
input of every node it visits, including one that heads nowhere"), so it
still needs a rule to build at all. **Not implemented as a `graph_grad`
rule here** -- worked around the way `legalize.py`'s `flatten_to_reshape`
already treats an equivalent case: `Unsqueeze` with a static input shape is
exactly a `Reshape` to a known target shape (which does have a rule), so
substituting the node before `build_backward` runs closes the gap with no
new gradient machinery. A real `unsqueeze_to_reshape` legalize rule
following that exact pattern is the concrete next step if this becomes a
committed pipeline rather than a survey probe.

**The gradient-magnitude evidence, the actual point of this check**: built
the real training-step graph (forward -> flatten -> MSE loss ->
`build_backward` -> in-graph SGD update, `conv_layers.0.conv.weight`
trainable, 172 nodes) and ran 5 real float32 SGD steps on host, comparing
`|grad|`'s mean against the weight's own mean magnitude at every step
(`weight_scale=0.05`, this project's own convention):

| step | loss | mean \|grad\| | mean \|weight\| | ratio |
| --- | --- | --- | --- | --- |
| 0 | 0.090749 | 0.0012737 | 0.0400694 | **0.0318** |
| 4 | 0.090743 | 0.0012736 | 0.0400694 | **0.0318** |

A finite-difference check on a real weight element confirmed the backward
pass is correct (`0.0017314` analytic vs. `0.0017323` finite-difference,
0.05% apart) before trusting the ratio above.

**0.032 is roughly three orders of magnitude better than Whisper's ~1e-4 to
1e-5** -- a gradient that's ~3% of the weight's own scale is squarely inside
what an 8-bit quantizer resolves (this project's own working resnet18 case
lives in a comparable regime), not buried under its noise floor the way
Whisper's is. This is real, if indirect, support for the depth hypothesis:
a 7-layer pure-CNN backward pass doesn't attenuate a gradient anywhere near
as much as a 6-layer transformer's LayerNorm+attention chain does.

**Not done here** (host-only survey task; real hardware was not confirmed
free and this doesn't need it to answer the question above): compiling this
step graph on Pulsar2/AX650N and confirming the gradient survives multiple
*quantized* steps, not just the float reference. That's the natural next
step for whoever picks this up -- the host-side evidence says it should
behave like resnet18, not like Whisper, but only a real compile+run settles
it the way this project settles everything else.

### Real hardware follow-up: compiles, `highest_mix_precision` fails a third way, and quantized gradient dies for a different reason than Whisper's

Rebuilt the training-step graph at the real `Wav2Vec2Config()` default scale
(4.2M-param feature extractor, matching the section above exactly --
`fe.conv_layers.0.conv.weight` trainable rather than layer 6, since
`build_resident_step`'s own `onnxsim.simplify()` pass renames later layers'
weight initializers via CSE, e.g. `fe.conv_layers.6.conv.weight` ->
`_v_100`, while layer 0's name survives -- picking layer 0 sidesteps the
naming churn rather than fighting it). 172 nodes, matching the host-only
survey's own count exactly. Host-verified again on this exact build:
perturbing along the gradient's own direction (the fix this project's own
Whisper work already established for finite-difference noise at this scale)
gives 0.2-0.3% agreement against a reference at properly-tuned step sizes.

**Compiles cleanly under standard INT8** (`pulsar2:7.0-lite`, 31.9s) -- the
first real compile of any wav2vec2-family training graph in this project's
history.

**`highest_mix_precision` fails here too, a third distinct way.** Not
Whisper's `LayerNorm` tiling limit (PR #1359's architecture has none) or
resnet18's `AvgPool` scheduler `TypeError` (this graph has none either) --
a real `TileFailException` on `AxErf`: `'dont support lut_float opr in
AXOPS/ONNXOPS/CUSTOM_OPS'`, on the GELU activation's `Erf` node, forced to
FP32 by the flag. Tried the same escape hatch PR #1359 tried for Whisper's
LayerNorm -- a `layer_configs` entry forcing just `Erf` back to `U8`
alongside `highest_mix_precision` -- and got the identical error, confirming
(a third time, on a third architecture) that `highest_mix_precision` does
not compose with `layer_configs` overrides at all; it is genuinely
whole-graph-only. Three architectures, three different real ops
(`LayerNorm`, `AvgPool`, `Erf`), three different real NPU-backend failure
signatures (a `TileFailException` on a tiling-workspace limit, a Python
`TypeError` inside the closed-source scheduler, and a `TileFailException`
on an unsupported float lookup-table operator) -- this is now a consistent
pattern, not a one-off: Pulsar2's FP32 tiling path does not reliably support
ordinary ops that appear in almost any real model, and `highest_mix_precision`
is not currently usable end-to-end on anything this project has actually
built.

**Standard INT8 compiles and runs, but the quantized gradient still dies
after step 0 -- for a different, more mundane reason than Whisper's SNR
floor.** Real hardware run (resident runner adapted to this model's real
I/O order, confirmed via `probe_io`: inputs `[x, y,
fe.conv_layers.0.conv.weight, lr, grad_seed]`, outputs `[updated weight,
loss]`):

| step | `w[0]` | loss |
| --- | --- | --- |
| initial | 0.453123 | -- |
| 0 | 0.4580865502 | 0 |
| 1-14 | 0.4580865502 (unchanged) | 0 |

A real, nonzero step-0 update happens (delta +0.00496), then the weight
freezes bit-identical from step 1 on, with loss reading exactly 0
throughout -- the same *symptom* as Whisper's "dies at step 1," but not the
same *cause*. The host-side evidence above already established this
model's true gradient-to-weight ratio (~0.032) is easily resolvable by an
8-bit quantizer -- there is no SNR floor here the way there is for Whisper.
The step-0 delta itself is the tell: `(0.453123 - 0.458087) / 1e-4 ≈ -49.6`
effective gradient magnitude, roughly five orders of magnitude larger than
the host-measured true gradient (~0.0013 mean absolute) -- this is a
calibration-range mismatch, the same class of bug PR #1346 found and fixed
for resnet18 (arbitrary, unmeasured `weight_scale`/`x_scale` guesses fed to
`make_training_calib.py` rather than values matched to this model's real
activation statistics), not a new fundamental limit. **Not chased further
here** -- fixing it needs proper calibration data (real or realistically-
scaled `x`/weight statistics, following PR #1354's confirmed textbook-MinMax
calibration behavior, or PR #1346's own fix pattern) rather than a config
flag, and is the concrete next step for whoever wants this model actually
training multiple real steps on hardware.

### Fixed: real calibration data, and a second, distinct degenerate-range bug in `lr` -- the first real multi-step audio training result on this hardware

Measured this model's real trajectory on host first, rather than guessing:
8 real float32 SGD steps (`x_scale=1.0`, `y` ~N(0, 0.01), the real exported
`fe.conv_layers.0.conv.weight` initializer, not a `weight_scale=0.05` draw)
gave mean\|grad\| ~1.4-1.8e-4, the weight itself essentially unmoving at
`lr=1e-4` -- confirming the ~0.032 ratio finding above and giving real
numbers to calibrate against, instead of the arbitrary `x_scale=0.3`/
`weight_scale=0.05` defaults the previous section's build used.

Rebuilt calibration with `make_training_calib.py`'s `real_data=` override:
`x_scale=1.0`, `weight_scale=0.36` (the real initializer's own mean\|w\|),
and `real_data={"fe.conv_layers.0.conv.weight": <8-step w trajectory>, "y":
<8-step y trajectory>, "grad_seed": [1.0]*6}`. Compiled cleanly. On real
hardware, with real (not `memset`-pattern) `x`/`y` fed via host files: loss
now reads a real, sensible `0.0754611` (matching the host trajectory's
0.073-0.082 range) instead of the previous exact `0`, confirming the input
calibration fix alone was real and correct.

**But the weight still froze bit-identical from step 1 on.** Diagnosed
rather than assumed: swept `lr` from 0.01 to 10000 at runtime and got
**bit-identical loss and weight at every value** -- the same "calibrated
narrowly, pins to a constant regardless of runtime input" signature PR
#1353 found for `grad_seed` on a different model, this time on `lr`.
Confirmed the cause directly: `make_training_calib.py`'s hardcoded `elif
inp.name == "lr": arr = np.array([1e-4], ...)` branch feeds the *identical*
value for all `n` calibration samples, so MinMax computes a zero-width
range and the compiled model can only represent that one value -- runtime
`lr` is silently clipped to it regardless of what's actually fed. This is a
second, distinct bug from the input-calibration mismatch above, not a
restatement of it, and it generalizes: any Axera training build with a
scalar runtime input calibrated from constant samples (this project's own
default for both `grad_seed`, historically, and `lr`, still) has this
failure mode latent, whether or not it happens to matter for a given
model's own trainable magnitude.

Fixed by giving `lr` a real *spread* in its calibration data instead of a
constant (`real_data={"lr": [0.01, 0.1, 1.0, 10.0, 100.0, 1.0]}`, spanning
the range this test actually swept). Recompiled, reran on real hardware:

| lr | step 0 loss | step 7 loss | step 0 `w[0]` | step 7 `w[0]` |
| --- | --- | --- | --- | --- |
| 0.01, 1.0 | 0.0754611 (unchanged) | 0.0754611 (unchanged) | frozen | frozen |
| 10 | 0.0735344 | 0.0693600 | frozen | frozen |
| **100** | **0.04335** | **0.0240833** | **-0.2461140752** | **-0.1757957637** |

At `lr=100`, both loss and the tracked weight element move smoothly and
**monotonically across all 8 real steps, with no freezing at any point** --
genuine, resolvable gradient descent on real AX650N hardware. At `lr=10`,
loss also moves monotonically (0.0735 -> 0.0694) while `w[0]` specifically
stays frozen -- plausibly a different weight channel's own gradient
dominates the visible loss movement at that scale while `w[0]`'s own share
stays sub-quantization-step; not chased further, since the `lr=100` result
already answers the question this section exists to settle. `lr=0.01`/`1.0`
still freeze completely -- consistent with the true per-step update
(~lr x 1.4e-4) staying below one quantization level of the weight-state
output's own calibrated range at those scales, not a remaining bug.

**This is the first real, fully-working, multi-step training result on any
audio/speech model in this project's history** -- not just "compiles" or
"one real step then dies," but a real loss curve moving in the right
direction across a real hardware run. Both fixes were calibration-only (real
input statistics instead of arbitrary defaults; a non-degenerate `lr`
range instead of a single repeated value) -- no graph, `legalize.py`, or
compiler-flag change was needed, unlike Whisper's and resnet18/50's own
paths past their respective ceilings.

**Flagged, not fixed here**: the same degenerate-constant-calibration bug
almost certainly affects every prior Axera build's `lr` input (all of
which used the same hardcoded `1e-4` constant), and `grad_seed` had the
*opposite* problem in this build specifically -- `make_training_calib.py`
has no `grad_seed`-aware branch at all, so it silently fell into the
generic `weight_scale`-noise path before this fix's `real_data` override
caught it. Whether this explains any part of PR #1353's own "seed sweep
returns bit-identical gradients at every value" finding on a *different*
model is an open, real question this task did not have scope to chase --
noted here as a concrete follow-on, not asserted.

### Batching and vNPU concurrency: vNPU compounds cleanly at batch=1, batching itself regresses correctness

Applied this project's two already-proven speed levers (batching, PR #1342;
vNPU concurrency, PR #1345/#1346) to the now-working feature-extractor case
-- the first real audio model where this is worth trying.

**Made the training-step graph batch-parametric.** Unlike resnet18's
`set_batch()` (a post-hoc shape edit, since nothing downstream of `x` bakes
a batch-specific constant), this model's `build()` computes a `flatten_shape`
initializer from the *exported* batch dim -- Conv1D's per-sample output
length threads through several strided layers before the manual flatten step,
unlike resnet18's pooling-then-Gemm tail. So batch is a real **export**
parameter here (`_export_feature_extractor(..., batch=N)`,
`torch.randn(N, 4000)`), not a graph edit after the fact --
`build_w2v2fe_batch_calib.py` is the new driver. Confirmed the raw export's
own op sequence is identical at batch 1 and 4 (74 nodes, same op list) before
trusting anything downstream. Host-verified the batched graph's gradient
correctness the way `test_set_batch_gradient_is_the_mean_of_per_sample_
gradients` already does for resnet18: batch=4's returned gradient matches
the mean of four separately-run batch=1 gradients to 3.2e-5 relative error
-- but only once every non-trainable layer's random initialization was
pinned identical across the two builds (`torch.manual_seed(42)` before each
export) -- the first attempt compared two *differently randomly initialized*
models and failed at ~116% relative error, a test-methodology bug, not a
graph bug, caught before it was mistaken for one.

**Real hardware, batch=1 (correctness already confirmed above): both levers
work.**

| config | throughput | notes |
| --- | --- | --- |
| solo (`AXCL_VNPU_DISABLE`) | 164.3 steps/s | min 5.590ms/step |
| 4x concurrent (`-v`) | 519.2 steps/s aggregate | 3.16x solo |
| 8x concurrent (`-v`) | 607.9 steps/s aggregate | 3.70x solo |

Confirmed non-corrupting first, the same check PR #1345 used: `-v` and
non-`-v` runs of the same model produce bit-identical loss/weight
trajectories, only step timing differs. The sub-linear scaling shape (3.16x
at N=4, 3.70x at N=8, saturating) matches resnet18's own qualitative finding
(2.53x at N=8) -- vNPU concurrency generalizes to this architecture.

**Real hardware, batch=4/8: compiles and times, but the gradient is
completely dead -- a new, unfixed calibration-range regression, not a
throughput result.** Both compiled cleanly (68.4s and 107.7s, well under the
batch-16+ compile-time wall) and ran without error, but **loss is
bit-identical across all 8 real steps at both batch sizes** (`1.11174`
constant at batch=4, `1.10973` at batch=8) -- unlike batch=1, where loss
moved smoothly every step. The calibration for both reused this model's own
*batch-shaped* real trajectory (not stale batch=1 data), so this is not a
repeat of the already-fixed generic-defaults or degenerate-constant-`lr`
bugs -- something about batch>1's real quantization range specifically
kills the gradient, not yet root-caused. Real step times measured anyway,
since timing doesn't depend on numerical correctness, but they should **not**
be read as "batching gives N.Nx throughput for training" the way resnet18's
own batching section can be -- there is no confirmed-correct training
happening at either batch size to be fast *at*:

| batch | step time (min/avg) | cmm |
| --- | --- | --- |
| 1 | 5.590ms / 5.865ms | 11.039 MiB |
| 4 | 18.820ms / 18.891ms | 25.534 MiB |
| 8 | 35.113ms / 35.270ms | 45.957 MiB |

**Net**: vNPU concurrency is a real, confirmed, generalizable win for this
architecture at batch=1. Batching needs its own dedicated diagnosis (likely
the same class of investigation PR #1370 already did twice -- direct
`quant_axmodel.json` inspection, an `lr`-style sweep -- applied to whichever
tensor's calibrated range is wrong at batch>1) before it can be trusted here;
flagged as the concrete next step, not chased further in this pass.
