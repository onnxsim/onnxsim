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

### wav2vec2 -- one gap, confirmed real and now fixed

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
**wav2vec2 (the plain, non-Conformer model) could not train through any
tail that includes attention output until `Where` had a backward rule --
and now it can.** `Where`'s gradient is `dX = mask * g`, `dY = (1-mask) *
g` with `mask = Cast(condition, FLOAT)`, the same "boolean condition
becomes a float 0/1 multiplied in, never re-emitted as `Where` itself"
convention `graph_grad`'s own module docstring already states as a rule for
every hand-written gradient here, and it is now `onnxsim.graph_grad`'s
`_grad_where`. `IsNaN` itself got a companion rule (`_grad_is_nan`,
returning no gradient) for a narrower but load-bearing reason: this
per-layer guard sits *inline* in the forward slice between `Softmax` and
`Where`, and `build_backward` requires a registered rule for every node
type it walks -- including one no gradient reaches -- not just the ones a
seed actually flows through, so differentiating a slice containing
`IsNaN(attn_weights)` raised `UnsupportedOpError` even though nothing ever
needed its own gradient.

Both rules live in a new `_PYTHON_ONLY_RULES` table (not `_RULES`): neither
has a C++/WASM mirror yet, so folding them into the parity-pinned
`SUPPORTED_OPS` `tests/test_qat_parity.py` checks against `qat_entry.cpp`
would be a claim about the browser path that is not yet true. They *are*
picked up automatically by every ordinary `build_backward(..., rules=None)`
call and are visible through `graph_grad.supported_ops()`, so QAT/LoRA
block discovery already treats a block containing them as differentiable --
see `_PYTHON_ONLY_RULES`'s own comment in `onnxsim/graph_grad.py` for the
full rationale, which mirrors `_grad_split`'s pre-existing
`_MULTI_OUTPUT_RULES` precedent for the same "real rule, no C++ port yet"
situation.

**Not yet done:** an actual trainable tail spanning wav2vec2's attention
output, built and verified on real AX650N hardware -- every wav2vec2 build
script in `scripts/axera/` today (`build_w2v2_feature_extractor_step.py`,
`build_w2v2fe_batch_calib.py`) still only trains the convolutional feature
extractor, which never touches this `Where`. The backward rule itself is
tested against finite differences on host
(`tests/test_graph_grad.py`'s `where_branch_select`, `where_broadcasts_x`,
and `where_isnan_guard` cases -- the last one mirroring this exact
`Where(IsNaN(x), 0, x)` shape).

### Real hardware follow-up: attention-output tail compiles and trains, once Pulsar2's *frontend* `IsNaN` gap is also routed around

`build_w2v2_encoder_attn_step.py` builds the graph this section's own "not
yet done" used to name: a real `Wav2Vec2Model` (small custom config --
`hidden_size=128`, 2 real encoder layers, `num_attention_heads=4` -- kept
small only for export/compile time; the convolutional feature extractor
stays at the real `Wav2Vec2Config()` default 512-channel/7-layer scale),
with **`layers.0`'s own `q_proj` weight** as the trainable tensor
specifically so a real gradient must differentiate back through *both*
encoder layers' `Where`/`IsNaN` guards to reach it, not just one.

**A second, separate real wall, found immediately after the backward-rule
gap closed:** `pulsar2 build` on the un-stripped step graph fails frontend
parsing outright -- `KeyError('dont support IsNaN opr in AXOPS/ONNXOPS/
CUSTOM_OPS')` -- independent of training or `onnxsim.graph_grad` entirely.
Pulsar2's ONNX frontend has never supported the `IsNaN` op at all, so no
wav2vec2-with-attention model, trained or not, has ever compiled on this
hardware before now; `onnxsim.graph_grad._grad_where`/`_grad_is_nan` were
necessary but not sufficient on their own.

**Routed around, exactly, not approximately:** `build_w2v2_encoder_attn_step
._strip_isnan_guard` removes every `Where(IsNaN(x), 0, x)` pair and rewires
consumers straight onto `x`. This is provably lossless *for this specific
deployment shape* -- the build never passes an `attention_mask`, so every
row is a real, unmasked position and HF's own Softmax output can never
actually contain a `NaN` (the guard exists only for the floating-point
cancellation a *fully masked* row's softmax can produce); `IsNaN(x)` is
therefore always `False`, and `Where(False, ., Y)` always selects `Y`. Not
a general graph-structure fact `legalize.py` could verify on its own (hence
local to this build script, not promoted there), and the graph's *other*
per-layer `Where` (the mask-bias one, condition from `Expand`/
`GreaterOrEqual`) is left untouched -- still real, still a genuine exercise
of `graph_grad._grad_where` on real hardware.

**Compiles cleanly** (`pulsar2:7.0-lite`, ~25s) after the strip.
**Host-verified correct** against finite differences at a properly-tuned
step size (0.3-2.3% agreement across several tries, matching this project's
own established bar) both before and after stripping the guard -- confirming
the strip changes nothing the graph computes, only what compiles.

**On real AX650N hardware, the weight state updates correctly and
consistently across 15 real steps -- but only once `lr` is calibrated and
run large enough to clear this weight's own INT8 quantization step.** This
one weight's real host gradient is ~1e-6/element (two real encoder layers
deep, one of many weights in the model), roughly 100-1000x smaller than the
feature extractor's own conv-weight gradient, so `lr*grad` needs
proportionally more scale to survive quantization at all -- the same
gradient-quantization-ceiling signature this project has documented
repeatedly (the Whisper SNR floor, the multi-phase calibration-swap
technique), now confirmed to extend to an attention-output tail too, not a
new failure mode:

| `lr` (calibrated + run at) | weight state across steps | verdict |
| --- | --- | --- |
| 1.0 | moves once (step 0), then bit-identical forever | dead -- below this weight's own quantization resolution |
| 2000.0 | `w[0]`: 0.1145 -> 0.1264 -> 0.1343 -> ... -> 0.2804 (all 15 steps distinct, monotonic) | real, consistent per-step movement |

The scalar `loss` output stayed bit-identical across all 15 steps at both
`lr` values, even while the weight state visibly moved at `lr=2000` -- a
separate, unchased detail (plausibly the loss output's own INT8 resolution
not resolving one 128x128 weight's worth of movement inside a much larger,
otherwise-fixed network; not the same mechanism as resnet50's own
loss-reads-exactly-0 finding, which was a calibration bug, not a resolution
one). The *weight update* -- what actually matters for whether training
works -- is the confirmed result here.

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
| `Where`/`IsNaN` numerical-stability cleanup has no backward rule | **fixed** (`onnxsim/graph_grad.py`'s `_grad_where`/`_grad_is_nan`, via the new `_PYTHON_ONLY_RULES` table) | was: plain wav2vec2, any tail touching attention output -- Conformer's own (differently-shaped) `Where` usage is a separate, still-open question (see below) | `dX = Cast(cond, FLOAT) * g`, `dY = (1 - that) * g` -- same "float mask, not `Where` itself" convention this module already uses everywhere else; `IsNaN` itself gets a no-gradient rule so `build_backward` can walk over it inline |
| Raw `Gelu` has no backward rule | open, low priority | only an exporter emitting fused `Gelu` instead of decomposed Erf-GELU (neither Whisper's nor wav2vec2's `transformers` export does this) | a `legalize.py` rule decomposing `Gelu` into `Mul`/`Add`/`Erf`/`Div`-by-constant, all already covered |
| `LSTM` has no backward rule; NPU-executable otherwise | **fixed** (`scripts/axera/legalize.py`'s `unroll_lstm`) | was: any classic RNN-based ASR model | unrolled into its own per-timestep `MatMul`/`Sigmoid`/`Tanh`/`Mul`/`Add`/`Concat` gate arithmetic, every op already covered on both axes |
| `GRU` has no backward rule *and* is not NPU-executable at all | **fixed** (`scripts/axera/legalize.py`'s `unroll_gru`) | was: any classic RNN-based TTS/vocoder model | same unroll as `LSTM`, and a strictly bigger win here: this is the only way a `GRU`-containing model runs on this hardware **at all**, training or not |

### A real LSTM test bed: NVIDIA Parakeet's own RNN-T prediction network

Every earlier LSTM check in this project used a synthetic `torch.nn.LSTM`
wrapper -- real enough to confirm *the op itself* exports opaque, but not a
published architecture. `transformers` (already this project's dependency
for every other real-model export here) ships one:
`transformers.models.parakeet.modeling_parakeet.ParakeetRNNTDecoder` is
NVIDIA Parakeet's real RNN-Transducer "prediction network" -- a plain
`nn.Embedding` -> 2-layer `nn.LSTM` -> `nn.Linear` decoder, no attention, no
Conformer (that lives in the model's separate FastConformer encoder, not
exported here). `scripts/axera/build_parakeet_lstm_probe.py` exports it at a
tiny config (`vocab_size=64`, `decoder_hidden_size=32`,
`num_decoder_layers=2`) and cross-references every resulting op type against
`onnxsim.graph_grad`'s rule tables and `scripts/axera/pulsar2_ops.py`'s
`AX650_SUPPORTED_OPS`, the same two tables this doc's whole survey uses.

**Confirms the synthetic finding on a real architecture**: the 2-layer LSTM
exports as **two separate opaque `LSTM` nodes** (one per layer, each with
its own `hidden_size` attribute) -- both `LSTM`-typed nodes are in
`AX650_SUPPORTED_OPS` (the NPU runs them fine at inference) and absent from
every one of `graph_grad`'s rule tables (no backward rule), so
`build_backward` cannot differentiate through either. Every other op in the
export (`Gather`, `MatMul`, `Add`, `Transpose` have backward rules;
`Concat`/`Constant`/`Expand`/`Squeeze`/`Unsqueeze`/`Shape` are pure
shape/indexing plumbing around `h0`/`c0` initialization, not real
weight-adjacent computation) is exactly the kind of scaffolding this
project's other coverage checks have already found harmless. This is the
concrete real target the "open, largest lift" row above needed -- an actual
LSTM-cell backward rule (or an unroll-based legalization route) can now be
tested against a real published model, not just a hand-built probe.

### A real GRU test bed too: DeepMind's WaveRNN vocoder -- a stricter gap than LSTM

`transformers` has zero `nn.GRU` usage anywhere in its model zoo, and
`silero-vad` (initially assumed to be GRU-based from its public reputation)
turned out to use `nn.LSTMCell` in its current release, not `nn.GRU` --
checked directly by loading it (`pip install torchaudio silero-vad`,
`silero_vad.load_silero_vad(onnx=False)`, then inspecting the loaded
`RecursiveScriptModule`'s printed module tree). `torchaudio.models.WaveRNN`
is the real find: DeepMind's own WaveRNN vocoder ("Efficient Neural Audio
Synthesis") has two real `nn.GRU` layers (`self.rnn1`, `self.rnn2` in
`torchaudio/models/wavernn.py`) driving its autoregressive sample
generation, with an ordinary config-driven constructor (`upsample_scales`,
`n_rnn`, `n_freq`, ... all plain ints, no pretrained weights needed).
`scripts/axera/build_wavernn_gru_probe.py` exports it at a tiny config
(`n_rnn=16`, `n_freq=8`, `upsample_scales=[2, 2]`) the same way the LSTM
probe above does.

**A real, previously-only-assumed finding, now confirmed on a published
architecture**: unlike `LSTM` (NPU-executable, only missing a backward
rule), **`GRU` is not in `AX650_SUPPORTED_OPS` at all** -- both real
`GRU` nodes in this export are absent from *both* tables. This is a
strictly bigger gap than LSTM's: a GRU-containing model cannot even *run*
on this hardware at inference, before training enters the picture at all.
Every other op in the export is either already covered (`Conv`, `MatMul`,
`Add`, `Gather`, `Identity`, `Relu`, `Reshape`, `Transpose`) or, again,
harmless shape/indexing scaffolding (`Concat`/`Constant`/`Expand`/`Slice`/
`Squeeze`/`Tile`/`Unsqueeze`/`Shape`) around the upsampling network feeding
the GRU stack -- confirming the gap is scoped to `GRU` itself, not
something this export incidentally also needs. (The WaveRNN export itself
turned out to hit a separate, unrelated `torch.onnx` exporter bug at
session -- an `Unsqueeze` axis miscomputation in `UpsampleNetwork`'s own
export path, reproducible across every config size tried, on this
torch/torchaudio version pairing -- so this op-coverage finding rests on
static graph inspection (`onnx.checker` plus op-type enumeration), not a
real onnxruntime execution of the *whole* WaveRNN graph; `unroll_gru`
itself, below, is verified independently against a clean, executing
`nn.GRU`-only export instead.)

### Both closed: `unroll_lstm`/`unroll_gru` legalize both ops into what the pipeline already covers

The natural follow-up question once both gaps had real target models: is
either op's math actually *reachable* with ops this pipeline already
covers, the same way `act_weight_conv_to_matmul` routes a live-weight `Conv`
into `MatMul`s? Both are. `LSTM`'s four gates and `GRU`'s three are each
ordinary `Sigmoid`/`Tanh` activations over a sum of two `MatMul`s and a
bias -- textbook, and every one of those op types was already confirmed
covered on both axes (a `graph_grad` backward rule and NPU support) earlier
in this doc. `scripts/axera/legalize.py`'s `unroll_lstm`/`unroll_gru`
replace a forward, single-direction `LSTM`/`GRU` node with exactly that
per-timestep arithmetic, verified numerically exact (float32 precision, not
an approximation) three ways: against a hand-built native ONNX node,
against a real `torch.onnx.export`-produced node with the module's own
learned weights extracted directly from the graph, and end-to-end against
the real Parakeet decoder probe above (`2.98e-8` max output difference
after unrolling both of its LSTM layers).

**Gate order and exact semantics were pinned down empirically, not assumed
from the spec text** -- both operators' bias/gate layouts have real,
easy-to-get-backwards subtleties (`LSTM`'s gate order is `i, o, f, c`, not
PyTorch's own `i, f, g, o`; `GRU`'s `linear_before_reset=1` -- what
`torch.onnx.export` always emits for `nn.GRU` -- applies the reset gate to
`h @ Rh + Rbh` as a whole, not `(reset * h) @ Rh`, and an initial attempt
assuming the opposite convention was off by up to `0.48` before the correct
formula was found by testing every gate-order/reset-variant/update-variant
combination against a real onnxruntime-executed native node). This is the
same "checked directly, not derived from memory of the spec" discipline
`docs/axera-on-device-training-handoff.md`'s ONNX `LSTM`/`GRU` gate-order
notes elsewhere in this project already follow.

**One real, separate gap this surfaced along the way**: differentiating
through the unrolled `Y` (full sequence) output -- what both Parakeet's
decoder and WaveRNN actually consume downstream, not just the final
`Y_h`/`Y_c` state -- needs a gradient through the `Concat` that stacks each
timestep's hidden state. `Concat` had no backward rule at all before this.
Fixed as `onnxsim.graph_grad._grad_concat` (a new `_PYTHON_ONLY_RULES`
entry, no C++/WASM mirror yet): one `Gather` per input, each pulling that
input's own contiguous slice back out of the incoming gradient along the
concat axis -- `Split`'s adjoint, reached a different way than
`_grad_split`'s own selection-matrix `MatMul` (that rule's docstring
explains why it couldn't reuse `Concat` for its own, opposite direction;
`Gather` with a compile-time-constant, single-axis index range is squarely
the shape of use this module's `BACKWARD_OPS` note already permits for
`_grad_conv`'s own indexing, so nothing needed to be *added* to that
allowlist). Verified against finite differences directly, and exercised
end-to-end in `tests/test_axera_legalize.py`'s own differentiability test
(`build_backward` reaching every one of an unrolled 2-layer LSTM's 8
per-gate weight tensors through its stacked `Concat` output).

**What actually gets trained changes.** `unroll_lstm`/`unroll_gru` split
each op's packed `W`/`R` weight into one initializer per gate (four for
LSTM, three for GRU) rather than leaving one packed tensor -- the same
kind of topology change `act_weight_conv_to_matmul` already makes for a
live-weight `Conv`. A resident training step built from an unrolled
LSTM/GRU trains those per-gate tensors, not a single packed one --
correction to an earlier draft of this section: `build_resident_step()`
needs nothing taught here at all, since `params` is already just a plain
list of initializer names supplied by the caller, not an automatic
finder; the real, if narrower, question settled below is whether
`graph_grad.build_backward` can actually reach every one of those
per-gate names once a real model's *other* forward-graph plumbing sits
in the way.

**Confirmed real, on real AX650N hardware.**
`scripts/axera/build_parakeet_lstm_train_step.py` builds a resident step
from the real Parakeet decoder (cut at its embedding-lookup output, fed
straight to the unrolled LSTM stack -- `input_ids`' real `[batch, seq]`
shape is a genuinely *batched* `Gather`, a case `graph_grad._grad_gather`'s
own docstring declines on purpose to avoid a wrong-but-silent reshape, and
nothing here needs the embedding table's own gradient anyway), training
the first LSTM layer's 8 per-gate weights against `decoder_output` (the
real `Concat`-built sequence output, not `Y_h`). Getting there surfaced one
more real, load-bearing gap this doc's earlier sections did not need:
`Squeeze` (a real node in Parakeet's own export, sitting between its two
LSTM layers -- unrelated to anything `unroll_lstm` itself emits) had no
`graph_grad` backward rule at all. Fixed as `onnxsim.graph_grad
._grad_squeeze_or_unsqueeze` (a `Reshape` back to the input's own shape,
identical math to `_grad_reshape`, registered for both `Squeeze` and
`Unsqueeze`) -- the same "found real, fixed generically" pattern as every
other gap this project's coverage checks have caught.

With both fixes in place: real `pulsar2 build` (`pulsar2:7.0-lite`)
compiled the 764-node unrolled step graph in **29 seconds** -- no compile-
time blowup from the many small per-timestep nodes, the open question this
section's own earlier draft flagged as worth watching. Real AX650N
hardware, `lr=200` (a real host trajectory measured this layer's gradient
at ~5.3e-6/element, two gates and a full second LSTM layer plus the output
projector away from the loss -- comparable to wav2vec2's own "two real
encoder layers deep" gradient scale elsewhere in this doc, and needing a
similarly large `lr` to clear its own INT8 quantization step): **real,
non-frozen loss** (`0.101409 -> 0.100904 -> ... -> 0.0998951`, settling into
the same quantization-step plateau-with-occasional-noise signature this
project's other real trainings runs show, not a smooth curve), 403 steps/s
(2.4-2.5 ms/step). Confirmed genuine and `lr`-driven, not calibration noise
on a frozen weight, the same control this project always runs: `lr=0`
freezes the loss completely bit-identical across 10 real steps
(`0.10494` every time), while `lr=200` visibly moves it.

**`GRU` confirmed real, on real AX650N hardware too** -- the natural
follow-on the previous paragraph named. `WaveRNN`'s own export still can't
be used directly (its unrelated `torch.onnx` exporter bug, confirmed not a
quick fix: reproduced identically across every opset 13-18 tried), so this
used `scripts/axera/build_gru_decoder_probe.py` -- the same
`nn.Embedding -> nn.GRU -> nn.Linear` decoder shape `ParakeetRNNTDecoder`
uses, `nn.LSTM` swapped for `nn.GRU`, at the identical scale
(`hidden_size=32`, 2 layers) for a direct comparison. `scripts/axera/
build_gru_decoder_train_step.py`/`build_gru_decoder_calib.py` mirror the
LSTM scripts exactly: cut at the embedding output, `unroll_gru`-legalized,
6 per-gate weights (`z, r, h`) of the first GRU layer trained against the
real `decoder_output`.

Getting there surfaced two more real, generically-fixed gaps -- one on
each side of the pipeline:

- **A real host-side bug, found by `unroll_gru`'s output specifically**:
  `build_resident_step`'s own pre-`build_backward` constant-folding pass
  (`_fold_constants`) calls `onnxsim.simplify()` with that function's
  default `initializers_as_constants=True` -- fine when `params` survive
  untouched, but for this graph the optimizer silently *dropped* the
  per-gate `W`/`R` initializers `unroll_gru` had just created (no error;
  `build_resident_step` only noticed several steps later, unable to find
  `params` by name any more). `unroll_lstm`'s own output happened not to
  trigger the same optimizer behavior -- exactly the kind of silent,
  model-specific failure worth closing off generally rather than routing
  around once. Fixed by passing `initializers_as_constants=False`
  explicitly at that call site.
- **A real Pulsar2 PPQ quantizer limitation, found only once compilation
  was reached**: the reset-gate term's own `Add(nhr, nrb)` (both operands
  lacking an unambiguous "real, input-derived" tensor to anchor the
  quantizer's platform-assignment tracer to, unlike `_lstm_step`'s
  equivalent `Add(xw, hr)`, whose `xw` operand is obviously real) failed
  to compile with `RuntimeError: Op Execution Error ... TargetPlatform.
  UNSPECIFIED`, reproduced in a minimal, isolated unrolled-`GRU`-only step
  graph (not specific to this decoder). Fixed losslessly by distributing
  the multiplication algebraically -- `rt*(nhr+nrb)` computed as
  `rt*nhr + rt*nrb` instead (identical by the distributive law), which
  gives every intermediate an unambiguous real anchor (`rt`) the way
  `_lstm_step`'s gates already have. Confirmed both changes preserve
  `unroll_gru`'s own numerical exactness (`tests/test_axera_legalize.py`
  unchanged, still passing) before compiling for real.

With both fixed: real `pulsar2 build` (`pulsar2:7.0-lite`) compiled the
646-node unrolled step graph in **26.6 seconds** -- comparable to `LSTM`'s
own 29s, no compile-time blowup here either. Real AX650N hardware,
`lr=200`: **real, non-frozen loss** (`0.0928868 -> 0.0910008 -> ... ->
0.0867572` over the first dozen steps, then the same quantization-step
plateau-with-noise signature every other real training in this project
shows), ~480 steps/s (~2.0-2.1 ms/step). Confirmed genuinely gradient-driven
with the same control every real result here uses: `lr=0` freezes the loss
completely bit-identical across 10 real steps (`0.110333` every time),
while `lr=200` visibly and immediately moves it.

**Net for LSTM/GRU as a pair**: both gaps are closed at the legalization
level, and now confirmed end to end on real hardware, not just
host-verified. `LSTM` needed only a backward rule's worth of arithmetic,
already NPU-executable as the opaque op; `GRU`'s fix was the bigger win,
since unrolling is the *only* way a `GRU`-containing model runs on this
hardware at all, sidestepping the "Pulsar2 itself would need to support the
op" ceiling a native-op fix would have hit -- and, unlike `LSTM`, needed one
additional real fix (the distributive reset-gate reformulation above)
before Pulsar2's own quantizer would accept it at all. Both are registered
in `legalize.TRAINING_RULES`, running before `graph_grad.build_backward`
sees the graph, the same slot `act_weight_conv_to_matmul` already occupies
for its own live-weight rewrite.

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

The ranking below it changed twice now. Wav2Vec2-Conformer's conv-module
gaps (depthwise `Conv`, `Split`) are **closed**, and plain wav2vec2's own
`Where`/`IsNaN` block is now **closed too** -- `onnxsim.graph_grad._grad_where`/
`_grad_is_nan` (see the wav2vec2 section above) give both `Where`'s
numerical-stability guard and Conformer's shared-condition `Where` a real
backward rule, generically, regardless of which shape either usage takes.
What is left is not implementing the rule (done) but *using* it: no
`scripts/axera/` build script has yet built a wav2vec2 (or Conformer) step
graph whose trainable tail spans attention output, so the concrete next
step is that build-and-verify-on-hardware work, not further `graph_grad`
coverage. Conformer's own shared-condition `Where` (see above) still has a
narrower, separate open question -- whether its condition is in practice
excludable by a careful block boundary, cheaper than differentiating
through it at all -- but that is now an optimization question, not a
blocker, since the rule handles either shape correctly if the boundary
question is left unanswered.

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

**Real hardware, batch=4/8: fixed -- a third calibration bug, same class as
the first two, on a tensor nobody had calibrated at all.** Direct
`quant_axmodel.json` inspection (the same method PR #1346/#1370 used)
traced the frozen gradient to the raw pre-`lr`-scaling gradient tensor
(`resident_step__reshape_406`, feeding `Mul_124: lr * grad`): its calibrated
range was `[-5.4e-5, 6.8e-5]` (scale `4.79e-7`), while a real batch=4
forward+backward at the intended `grad_seed=1.0` produces gradients up to
`1.87e-3` -- **~27x too narrow**, saturating almost the entire tensor.
Root cause: `grad_seed` (the backward-pass seed multiplying the *entire*
gradient before it ever reaches this tensor) had **never been given
`real_data` calibration anywhere in this pipeline, not even at batch=1** --
it fell through to `make_training_calib.py`'s generic
`weight_scale=0.05`-scaled random-draw branch, producing calibration values
near 0 rather than the real runtime value of 1.0. Same bug *class* as PR
#1370's `lr` fix (a scalar multiplier whose calibration was never matched to
its real runtime usage) on a *different* scalar -- batch=1 happened not to
manifest it (its naturally larger unbatched gradient apparently still fit
inside the resulting undersized range), batch=4/8's more-averaged gradient
did not.

Fixed by adding `real_data` for `grad_seed` (jittered around 1.0, matching
the `lr` fix's own pattern) to `build_w2v2fe_batch_calib.py`. Confirmed on
the compiled model: the raw-gradient scale widened `4.79e-7 -> 1.41e-5`
(~29x, matching the ~27x under-calibration found). **Real hardware, batch=4,
8 steps, `lr=100`: loss moves smoothly and monotonically -- `1.02927 ->
0.98068`** -- where it was bit-identical before. (The tracked weight
readback itself steps in coarse ~0.0102 increments across a few repeated
values -- expected, not a bug: that tensor's own state-output quantization
scale is `0.01018`, so consecutive real updates smaller than one
quantization step legitimately round to the same U8 code; loss, quantized
far more finely relative to its own dynamic range, is the reliable signal
here, the same distinction PR #1346's per-sample-vs-final-scalar debug tap
already established.)

Real step times, now trustworthy since correctness is confirmed at batch=4:

| batch | step time (min/avg) | cmm |
| --- | --- | --- |
| 1 | 5.590ms / 5.865ms | 11.039 MiB |
| 4 | 18.838ms / 19.353ms | 25.534 MiB |

**Batch=8: re-verified with the `grad_seed` fix, confirmed working.** Same
`build_w2v2fe_batch_calib.py` driver, `batch=8`. Real hardware, `lr=100`,
8 steps: loss moves smoothly and monotonically, `1.03951 -> 1.00037`
(plateauing over the last two steps at the same tracked-weight-quantization
granularity already documented for batch=4 -- not a bug). Step time
35.577ms min / 36.186ms avg, cmm 45.957 MiB.

**vNPU + batching composition: real, but weaker than resnet18's, and
saturates earlier.** `-v` at batch=8, N=1: bit-identical loss/weight
trajectory vs. disabled, confirming non-corrupting at this batch size too
(same check PRs #1345/#1346/#1372 already established at batch=1). Real
concurrent throughput, batch=8, separate OS processes per context, all
converging to the identical loss (`0.965574`) confirming correctness held
under concurrency:

| config | aggregate steps/s | aggregate samples/s | vs. batch=1,N=1 baseline (164.3 samples/s) |
| --- | --- | --- | --- |
| batch=8, N=1 | 27.3 | 218.4 | 1.33x |
| batch=8, N=4 | 82.5 | 660.0 | 4.02x |
| batch=8, N=8 | 85.6 | 684.8 | 4.17x |
| batch=1, N=8 (PR #1372) | -- | 607.9 | 3.70x |

Unlike resnet18 (PR #1346: 6.5x batching alone x 2.5x vNPU alone -> 18.1x
combined, cleanly multiplicative), this model's batching gain alone is
much smaller (1.33x, not 6.5x -- Conv1D quantize/glue overhead dominates
differently here) and the combination **saturates by N=4** (660 -> 684.8
samples/s from N=4 to N=8, a 3.8% gain for double the contexts) rather than
continuing to scale to N=8 the way resnet18 did. Real, honest result: the
two levers still both help, but this architecture's per-step compute at
batch=8 already occupies enough of the NPU that fewer concurrent contexts
saturate it, so the combination is additive-ish rather than cleanly
multiplicative. Best practical point here is N=4 (nearly all of N=8's gain
for half the contexts), not N=8.

**Net**: three real calibration bugs found and fixed for this model across
PRs #1367/#1370/#1373 -- `x_scale`/`weight_scale` mismatch, degenerate
constant-`lr` range, and uncalibrated `grad_seed` -- all in the same
family (a scalar or tensor whose calibration data was never matched to its
real runtime distribution). Batching is now confirmed working at batch=4
and batch=8; vNPU+batch composition is confirmed real but weaker and
earlier-saturating than resnet18's own result -- both flagged follow-ons
from PR #1373 are now closed.

### 2,000 real steps: neither resnet18's clean death nor Whisper's, a third signature

Every wav2vec2 run before this one was 8 real steps -- too short to know
whether this model has the same eventual gradient-underflow ceiling
resnet18 does (U8 dies ~step 1,000, U16 ~step 5,000, `docs/axera-on-device-
training-handoff.md`'s ceiling section) before it matters in practice. Real
AX650N hardware, batch=4, `lr=100`, the same working config as the section
above, **2,000 real steps, ~39s wall-clock (51.3 steps/s)**:

| step | loss |
| --- | --- |
| 0 | 1.0116 |
| 100 | 0.896746 |
| 300-700 | 0.879076 (flat) |
| 800-1600 | 0.874659 (flat) |
| 1700-1999 | 0.870241 (flat, including 203 consecutive bit-identical steps at the end) |

Real, substantial progress through roughly the first 1,500 steps (a ~14%
relative loss reduction), then the loss settles into a sequence of flat
plateaus, each held for hundreds of steps before dropping to the next
level -- not a smooth curve, and not resnet18's signature either (which
goes to *exactly* zero gradient and stays there, output frozen for good).
**The tracked weight (`w[0]`) keeps moving throughout, including during the
final 200+-step loss plateau** -- bouncing between several distinct
quantized values (`0.1425719261`, `0.1527556330`, `0.1629393399`,
`0.1731230468`), never settling to one constant the way a genuinely dead
gradient would leave it. So this is not resnet18's "gradient rounds to
exactly zero, both loss and weight freeze for good" ceiling, and it
survived vastly longer than Whisper's one-step death.

**What it actually is, stated precisely rather than guessed**: with only
final-loss and weight-readback instrumentation available (this runner
doesn't expose the raw gradient's own nonzero-fraction the way `resident_
runner.c`'s ceiling characterization did for resnet18), the two candidate
explanations -- real learning continuing below the *loss output's own*
quantization resolution at this stage, versus a genuinely stalled/oscillating
gradient with only quantization noise moving the weight -- are not
distinguishable from this data alone. Both are consistent with what's
observed; picking between them needs the same debug-tap technique (`Reduce
Max`/per-sample-loss, PR #1346, #1358) this project has already used
elsewhere, not done here.

**Does the multi-phase calibration-swap technique (PRs #1355/#1356) apply?**
Unclear, and for a specific reason worth stating rather than assuming either
way: that technique recalibrates the *gradient* tensor's own quantization
range for a smaller expected magnitude. What's plateauing here is the
*loss* output's readable resolution, not (as far as this data shows) the
gradient's. If the real mechanism turns out to be the loss-output-resolution
explanation, the applicable fix would be recalibrating the **loss** output's
range for its late-training magnitude, not the gradient's -- a different
tensor than the one PR #1355/#1356's technique targets, though the same
general principle (recompile calibrated for the value's own late-training
scale). If it turns out to be a genuinely stalled gradient instead, PR
#1355/#1356's existing technique would be the direct fit. Settling which
needs the gradient-nonzero-fraction instrumentation noted above -- a
follow-on, not done here.

### The 2,000-step plateau (PR #1376) is a resolution ceiling, not convergence -- settled with a real lr-drop experiment

PR #1376 ran this model for 2,000 real steps (batch=4, `lr=100`) and found
real loss progress through ~step 1,500 (`1.0116 -> 0.870241`), then a flat
plateau -- but with the tracked weight still visibly bouncing between a
handful of quantized values through the plateau, unlike resnet18's clean
frozen-solid gradient death. It left open which of two things this was:
ordinary SGD convergence at an oversized `lr` (weight oscillates near a
minimum, loss just can't show sub-quantization-step progress), or a
genuinely degraded gradient signal.

**Settled with a direct experiment**, reusing the exact compiled artifact
and calibration from that run (`/tmp/w2v2_longrun_work/step.onnx.axmodel`,
still on disk): a new runner variant
(`scripts/axera/tools/w2v2fe_runner_lrdrop.c`) keeps the weight state
device-resident continuously across a *single* run while switching `lr` by
a host->device scalar write partway through -- no restart, no state
round-trip. Ran 2,200 steps at `lr=100`, then dropped to `lr=1` (100x
lower) at step 1,600, well inside the plateau:

* **Steps 775-1,599 (`lr=100`)**: loss frozen bit-identical the entire
  stretch (`0.87465894`, one single transition to `0.87024146` around step
  1,575-1,599); the weight visibly hops between ~10 distinct U8-quantized
  levels (`0.14257` ... `0.35643` ... back down), never settling.
* **Steps 1,600-2,199 (`lr=1`)**: both loss *and* weight go completely
  bit-identical for the full remaining 600 steps -- `loss=0.87024146`,
  `w[0]=0.1833067536`, no movement at all.

**Neither original hypothesis is quite right.** If this were ordinary
convergence at an oversized `lr` (hypothesis 1), dropping `lr` should let
the model resolve finer progress -- it should keep decreasing, more slowly.
It didn't move at all. If it were simple SGD oscillation near a minimum,
the *loss* landscape sampled across ~800 different weight values (steps
775-1,600) should show *some* variation -- it never did, not once. The
weight's visible movement at `lr=100` was not directed learning; it was a
genuinely small underlying gradient, amplified by an oversized `lr` into
steps just large enough to hop the weight's own U8 quantization boundary
without ever producing a large enough *loss* change to clear the loss
tensor's own (much coarser, at this point in training) quantization step.
Drop `lr` back to a sane value and that same small gradient no longer moves
the weight at all.

This is its own, milder variant of the same family of problem as Whisper's
SNR floor (PR #1359) -- the real signal is small relative to what this
hardware's fixed-point representation can resolve -- but arrived at far
later and far more mildly: this model got a genuine ~14% loss reduction
over 1,500 real steps before hitting it, where Whisper hit an equivalent
wall after exactly one step. Call it a **resolution ceiling**, not a dead
gradient (resnet18's pattern: instant, total, every-tensor freeze) and not
an SNR floor (Whisper's pattern: no real signal ever gets through at all).
**No learning-rate change fixes this** (confirmed directly, not assumed) --
whatever comes next would need the same class of fix explored for the
other ceilings (PRs #1355/#1356's multi-phase calibration-swap, if the
remaining gradient at this point is large enough to benefit from a
narrower recalibrated range -- untested here, a real follow-on) rather than
a hyperparameter change.

### Multi-phase calibration swap breaks the plateau, then hits a new one -- real, but a one-shot gain, not a general fix

The follow-on above, run for real. Direct evidence the plateau was a
calibration-range problem, not a dead gradient: the frozen plateau loss
value (`0.87024146`) was checked against the compiled model's own
`quant_axmodel.json` -- the `loss` tensor's calibrated range
(`scale=0.0044175`, representable span `255*scale=1.131`) versus the real
observed loss trajectory across the whole 2,000-step run (`[0.8702,
1.0293]`, span `0.159`) -- ~7x narrower than what was calibrated, meaning
the quantizer was spending most of its 256 codes on loss values the model
never actually produced once training got this far.

**Built phase 2**: a new runner variant, `w2v2fe_runner_capture.c`, adds
the missing ingredient no earlier run persisted -- it dumps the full
trainable-weight tensor (not just its `w[0]` scalar readback) to disk at a
window of late-training steps, plus the final state, so a real trajectory
exists on disk to calibrate against. Ran it against the exact PR #1376
compiled artifact for 1,700 steps (same `lr=100`, same seed), capturing the
weight at steps 700/800/.../1600 -- squarely inside the plateau region --
and the final state at step 1,699.

`build_w2v2fe_mp_swap_phase2.py` rebuilds the same step graph and
recalibrates the trainable weight against those 10 real captures, paired
with the runner's own fixed, repeating `x0`/`y0` batch (not a diverse
trajectory -- that fixed pair genuinely is the real runtime distribution
here, since the runner reads `.x0`/`.y0` once and reuses them every step).
A host check confirms this combination reproduces the real plateau loss
almost exactly (`0.8717-0.8761` across the 10 captures, against the real
hardware's `0.8702-0.8791` for the same step window) -- an earlier attempt
that paired the same late-stage weight captures with an unrelated
early-training `x`/`y` trajectory left the `loss` tensor's calibrated range
essentially unchanged (`scale` moved by <1%), because the computed
calibration loss never actually landed in the real plateau band either;
matching the fixed real inputs, not just the weight, was what mattered.

Compiled phase 2 for real (`pulsar2_docker.build`, same config shape as
phase 1). The `loss` tensor's calibrated span narrowed from `1.131`
(phase 1) to `0.876` (phase 2, `scale=0.0034355`) -- real, but a modest
~1.3x, not the ~7x the raw trajectory span suggested, since MinMax still
calibrates to the captures' own min/max rather than the tighter band a
single fixed step would occupy.

**Real hardware, seeded from phase 1's actual plateaued state
(`w[0]=0.1833067536`, byte-identical, no restart from scratch)**: step 0
on the phase-2 compile immediately reads `loss=0.862318` -- lower than
phase 1's frozen `0.87024146`, a real, additional step of progress the
first compile's calibration could not resolve. Running 2,000 more steps
confirms this is not noise: loss alternates between exactly two values,
`0.862318` and `0.865753`, both consistently below phase 1's plateau,
never regressing back to `0.87024146` and never moving further.

**Net**: the technique works -- swapping to a phase compiled with
calibration matched to the real late-training distribution genuinely
recovers real loss progress a stuck compile could not deliver, confirming
PRs #1355/#1356's resnet18 result generalizes to a real trained model, not
just their small Conv+Gemm demo. But it is a **one-shot gain, not a fix**:
phase 2 hits its own new, finer resolution ceiling within single-digit
steps (the same mechanism as phase 1's, just at smaller scale), and
would need a phase 3 -- calibrated against phase 2's own new plateau
trajectory -- to gain further. Each phase buys a fixed, shrinking amount of
additional resolution, not unbounded continued training; whether repeated
phases converge to the real minimum or hit diminishing returns quickly is
unmeasured, a real follow-on.

### Phase 3: same mechanism, but diminishing returns -- the technique hits a floor after one real gain

The follow-on above, answered for real. First, confirmed phase 2's plateau
is the *same class* of problem as phase 1's, not a new one: a fresh
`w2v2fe_runner_capture.c` run against the real `w2v2_phase2.axmodel`
(steps 5-24, squarely inside its plateau) shows real loss alternating
between exactly two adjacent values, `0.862318` and `0.865753` -- a span
of `0.0034356`, matching phase 2's own `quant_axmodel.json` `loss` scale
(`0.0034355`) almost to the last digit. So phase 2's calibrated
representable span (`255*scale=0.876`) is not ~7x too wide like phase 1's
was -- it is ~255x too wide, the identical "MinMax calibrates to the
captures' own min/max, not the far tighter band a continuing run settles
into" mechanism as phase 1's, just compounded by another round. Not an
SNR-floor problem like Whisper's; still a resolution-mismatch problem.

**Built phase 3** (`build_w2v2fe_mp_swap_phase3.py`, same shape as phase
2's generator): recalibrated the trainable weight against 20 real captures
of phase 2's own trajectory (steps 5-24), paired with the same fixed
`x0`/`y0` batch every phase has used. A host check confirms this
combination reproduces the real plateau band closely (`0.8661-0.8702`
across the 20 captures, against real hardware's `0.8623-0.8658` for the
same steps). Compiled for real (`pulsar2_docker.build`, same config shape).

**The recalibration barely moved anything this time.** Phase 3's `loss`
scale came back `0.0034118` -- a change of under 1% from phase 2's
`0.0034355`, despite calibrating against a genuinely different, directly
relevant real trajectory (not the "unrelated x/y" bug this project has hit
before). Unlike phase 1 -> 2's real ~23% narrowing (`1.131` -> `0.876`),
phase 2 -> 3's calibration essentially reproduced the same representable
span (`0.876` -> `0.870`).

**Real hardware, seeded from phase 2's actual plateaued state
(`w[0]=0.2563081682`, byte-identical continuation)**: 200 steps read loss
alternating between exactly two values, `0.863183` and `0.866594` -- a
span of `0.003411`, again matching the new build's own calibrated scale
almost exactly, the same one-quantization-step signature as phase 2's own
plateau. But this time there is **no net progress**: phase 3's low value
(`0.863183`) is *higher* than phase 2's low value (`0.862318`) -- a real,
if tiny, regression, not a gain. Plateaus within single-digit steps, same
as phase 2 did.

**Net across all three phases**: phase 1 -> 2 bought a real gain
(`0.87024146` -> `0.862318`, `Δ=-0.00792`, ~14% of the whole PR #1376 run's
total reduction, in one recalibration). Phase 2 -> 3 bought nothing
(`0.862318` -> `0.863183`, `Δ=+0.00086`). **This settles the repeatability
question the phase-2 writeup left open: the technique is a diminishing-
returns, not-repeatable-indefinitely fix, not a "few large steps" one.**
Each recalibration can only narrow the calibrated range by as much as the
real variation still present in the captured trajectory allows -- and by
phase 2, that variation had already collapsed to a single quantization
step's worth of oscillation, leaving nothing left for a phase-3
recalibration to exploit. The first swap worked because phase 1's real
captures still spanned a meaningfully wide band (weight actively
transitioning into its plateau); by the second swap, the captures
themselves were already as narrow as the model's own resolution ceiling,
so recalibrating against them just reproduces the same ceiling. Phases
1/2/3's full artifacts and captures are on the AX650N VM
(`/root/auto_phase1.axmodel`, `/root/w2v2_phase{2,3}.axmodel*`,
`/root/w2v2_capture_p{2,3}/`) for any follow-on that wants to inspect them
directly.
