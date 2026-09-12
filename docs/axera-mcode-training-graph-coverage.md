# Does the mcode tooling cover training-graph mcode? Confirmed yes, with one real, permanent boundary

`scripts/axera/mcode.py`'s structural validator and `scripts/axera/emitter.py`'s
weight-table machinery were reverse-engineered and tested (`tests/test_axera_mcode_validator.py`,
`tests/test_axera_emitter.py`) entirely against **inference-only** compiles --
frozen-weight, forward-only graphs like the `piper_vocoder` fixture. This
project has since compiled dozens of **training**-step graphs (resnet18,
resnet50, Whisper, wav2vec2's feature extractor, the multi-phase
calibration-swap probes) whose mcode had never been checked against this
tooling at all. This records that check, done directly against four real
compiles rather than assumed either way.

## What was checked

Pulled four real training-step `.axmodel`s still on disk from this session's
prior hardware work, covering three structurally different architecture
families and a 30x size range:

| model | architecture | mcode size (`*_neu` raw bytes) | `npu_params` size |
| --- | --- | --- | --- |
| `r18_b1` | resnet18, CNN | 190,800 B | 6,412,676 B |
| `r50step` | resnet50, CNN | 198,952 B | 19,928,418 B |
| `whisper_step` | Whisper encoder, transformer | 5,840,112 B | 13,164,200 B |
| `w2v2fe` | wav2vec2 feature extractor, Conv1D | 202,976 B | 8,662,691 B |

## Finding 1: `mcode.check()`'s structural rules generalise cleanly -- confirmed on all four, not assumed

`mcode.check()` (the tag-set/verb-set/segment-table structural validator) was
run against each of the four real training-graph mcode blobs, extracted the
same way `emitter.py`'s own tools already do (`emitter.mcode_name()` to find
the `*_neu` initializer, then the raw bytes). **All four pass with zero
violations** -- the same result as the three pre-existing inference-only
fixtures (`conv64_k5_d2`, `conv128_k7_d12`, `piper_vocoder`).

This isn't just "no exception was raised" -- the tag-byte-frequency profile
(the same statistical measure `mcode.py`'s own `WIDE_TAGS`/`ALL_TAGS` sets
were originally derived from, per its docstring's "Fourth/Fifth correction"
references) was compared directly:

| blob | `WIDE_TAGS` share | `ALL_TAGS` share |
| --- | --- | --- |
| `piper_vocoder` (inference, reference) | 15.0% | 17.7% |
| `r18_b1` (training) | 18.6% | 20.4% |
| `r50step` (training) | 16.4% | 18.5% |
| `whisper_step` (training) | 16.1% | 19.6% |
| `w2v2fe` (training) | 14.9% | 17.9% |

All five fall within a tight band (14.9%-18.6% / 17.7%-20.4%), with no
training-graph outlier. A `Gather`/`MatMul`-heavy backward pass and an
in-graph SGD update (`Mul`+`Sub` on the trainable-weight state) don't produce
a statistically distinct instruction mix at this level -- Pulsar2 apparently
lowers a training step's extra arithmetic through the same instruction
vocabulary an inference graph uses, not something structurally novel.
**Conclusion: no change needed to `mcode.check()` or the tag/verb sets it's
built from.** Coverage is confirmed, not a gap.

Added `w2v2fe_training_step` as a fourth committed fixture
(`scripts/axera/fixtures/w2v2fe_training_step.mcode.gz`, from the real
`w2v2fe.axmodel` this thread's PR #1370 compiled) to
`tests/test_axera_mcode_validator.py`'s parametrized `_BLOBS` table --
picked over the other three candidates because it's the smallest (202,976 B
raw, 88,330 B gzipped) and the newest architecture family (Conv1D audio) this
project has working training for, giving future work on `mcode.py` a
training-graph regression check on every push, not just three inference-only
graphs.

## Finding 2: `emitter.py`'s weight-table machinery cannot reach a trainable tensor at all -- a real, permanent scope boundary, not a bug

`emitter.py`'s entire weight-patching machinery (`codes_of`, `learn`,
`emit_table`, `requant_block`) is built around a compiled model's static
`npu_params` initializer -- the byte-exact quantised encoding of a **frozen**
Conv/MatMul weight, baked in at compile time. A training step's trainable
weights are graph *state* (`onnxsim.qat_graph.StepGraph`'s sense -- both an
input and an output of the compiled graph, per `build_resident_train_step.py`
throughout this thread's work), not initializers at all.

Checked directly on `w2v2fe.axmodel`, the simplest case (exactly one
trainable tensor): `fe.conv_layers.0.conv.weight` is declared as a
**graph input** (confirmed in `model.graph.input`, not
`model.graph.initializer`), shape `[512, 1, 10]`, 5,120 elements. The model's
only other non-`npu_params`/non-mcode initializer is `npu_dyn_params`, shape
`[0]` -- empty, matching this project's own earlier finding (`pulsar2_ops.py`'s
docstring) that `npu_dyn_params` is empty in every compiled model checked so
far. **There is no initializer anywhere in the compiled model that could hold
an encoding of the trainable weight** -- `npu_params`' 8,662,691 bytes are
necessarily entirely the six *other*, frozen feature-extractor conv layers.

This is confirmed by simple arithmetic too, not just by the absence of an
alternative initializer: `r18_b1`'s `npu_params` is 6,412,676 bytes for a
model whose frozen backbone is resnet18's non-trainable ~6.3M-parameter
prefix (11.7M total minus the 5,361,664 trainable last-four-layers weights
this project's resnet18 work has used throughout) -- consistent with
`npu_params` holding roughly one byte's worth of code per frozen weight
(INT8/S8 codes plus per-channel scale/zero overhead), and *not* consistent
with it also containing the 5.36M trainable weights on top.

**Conclusion**: `emitter.py`'s weight-learning/emission tools apply only to a
training graph's *frozen* (non-trainable) backbone -- exactly the same as for
any inference-only model, since a frozen weight is architecturally
indistinguishable from one in a pure inference graph. They **cannot reach a
trainable tensor at all**, not because of a bug or an unimplemented case, but
because a live/state tensor has no static byte encoding in the compiled
artifact for `emit_table()` to write into -- it's carried entirely through
ordinary I/O buffers (`resident_runner.c`'s device-resident state pattern),
resolved at runtime, not compile time. This is a real, permanent scope
boundary worth stating plainly, the same way `mcode.py`'s own docstring is
explicit about what "no evaluator" means -- not a gap to close, a fact about
what a *compile-time-baked* format can and cannot represent about a
*runtime*-varying tensor. (This is the same underlying fact PRs #1353-#1356's
gradient-quantization work already established from the other direction --
a trainable tensor's precision is fixed by calibration at compile time, with
no runtime-adjustable handle into it; this doc's finding is the mcode-format
-level restatement of the same constraint.)

## What this means for future mcode/emitter work

- `mcode.check()` needs no training-graph-specific rule -- it already covers
  this domain, confirmed on real, structurally diverse samples.
- Any future attempt to patch a *trainable* tensor's on-device representation
  post-hoc (mirroring what `emit_mcode()`/`nudge_output_quantisation()` do for
  a frozen output's scale/zero-point) would need an entirely different
  mechanism than everything in `emitter.py` today, since there is no static
  byte range corresponding to a live weight to locate or patch in the first
  place -- not a missing feature of the current tools, a different problem
  shape.
- `emitter.py` remains fully applicable to a training graph's frozen
  backbone specifically (e.g. swapping in a different pretrained backbone's
  weights without recompiling), which was never tested here but follows
  directly from Finding 2's own evidence -- a frozen weight in a training
  graph is not different from one in an inference graph.
