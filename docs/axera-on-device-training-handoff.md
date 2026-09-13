# On-device training on the Axera AX650N: handoff note

**Status: working, with measured limits.** A resnet18 fine-tuning step compiles
and runs on real AX650N hardware with its gradients computed on the NPU. This
records what works, what the ceiling is, and which walls are the vendor's
rather than ours, so none of it has to be rediscovered.

Companion to `docs/ozaki-scheme-axera-handoff.md`, whose conclusion this work
overturned: that note said the toolchain could not host a split-and-correct
matmul because every documented way to pin a per-matmul scale is broken. It
can, by not asking for one.

## What runs

resnet18d, last four layers trainable, 64x64 input, on the card:

| output | cosine vs onnxruntime | SNR |
| --- | --- | --- |
| `layer4.0.downsample` weight gradient | 0.97676 | 13.35 dB |
| `layer4.1.conv2` weight gradient | 0.97272 | 12.60 dB |
| `layer4.1.conv1` weight gradient | 0.98140 | 14.23 dB |
| classifier weight gradient | 0.99710 | 22.14 dB |
| loss | 1.00000 | 37.23 dB |

5,361,664 trainable parameters, 731 nodes reduced to 210 by `onnxsim.simplify`,
compiled in 97 s to a 7.0 MB `.axmodel`, **200.6 ms per step** moving 21.5 MB in
and 21.4 MB out. The gradients are directionally right but not precise -- 12-14
dB on the convolutions -- which is the INT8 backward pass, not a bug.

Smaller graphs are exact: eight independent weight tensors through 702 nodes
return gradients at cosine 0.9971-0.9996, and a 16-channel convolution trains
from -3.38 dB to 34.02 dB against a teacher.

## How it is put together

1. `onnxsim.graph_grad.build_backward()` emits the backward pass as ordinary
   ONNX nodes. **26 of its 28 differentiable op types are on
   `AX650_SUPPORTED_OPS`**, none on the confirmed-broken list.
2. Every trainable weight is promoted from initializer to **graph input**, so
   the compiler never bakes it in and the host can change it per step.
3. `legalize.TRAINING_RULES` makes the result compilable (below).
4. `onnxsim.simplify(skipped_optimizers=[...])` -- worth 69-71% of the nodes.
5. `pulsar2 build` with `layer_configs` promoting the backward ops to `U16`.
6. A resident runner keeps the model loaded and streams frames.

## The ceiling: the gradient dies

The gradient is an **output tensor**, quantised at a range fixed when the model
was built. As training converges the true gradient shrinks below half a
quantisation step and rounds to zero -- every entry, eventually.

**`docs/axera-quantizer-reverse-engineering.md`** confirms, quantitatively, that
Pulsar2's calibration is textbook asymmetric MinMax over the caller-supplied
calibration data specifically (not graph structure) -- meaning recalibrating
with late-training-scale (small) synthetic gradient values, entirely within
Pulsar2's own sanctioned pipeline, is a real, untested candidate fix for this
ceiling. That document also found `Conv` has a separate `output_data_type:
"FP32"` override distinct from `layer_configs`' `data_type` override -- whether
`MatMul` has the same is the most promising untried lever for a plateau found
while pursuing loss scaling via an FP32 gradient seed (see the still-open PR
that introduced that finding for the full context once merged).

| gradient tensor | dies at | best SNR reached |
| --- | --- | --- |
| U8 | step ~1,000 | 30.88 dB |
| U16 | step ~5,000 | 34.02 dB |

Widening buys a factor of five in steps and 3 dB. It does not remove the
mechanism. 34.02 dB is within 1.3 dB of that shape's INT8 *forward* floor, so
for fine-tuning there is little left to win; for anything longer, this is the
wall.

**Loss scaling cannot fix it here.** The seed must reach the graph as a tensor,
and a tensor is quantised to a fixed range with linear levels: over
`[1, 2**21]` a U8 seed has 255 evenly spaced levels, so a seed of 1.0 rounds to
**zero**; calibrated narrowly it pins to a constant. A probe sweeping the seed
from 1 to 2**24 returned **bit-identical gradients at every value**. A
multiplicative scale meant to span decades cannot live in a linear fixed-point
tensor. `finetune.LossScaler` therefore detects that it is having no effect and
stands down.

### The FP32 gradient seed: tested, real, but it plateaus rather than opens the door

The untried way out was: `layer_configs` accepts `data_type: "FP32"` for
elementwise ops, and the seed feeds a `Mul`. If the seed and its consumer stay
float, scaling should work. It does something real -- but not the open-ended
fix the theory suggested, and the reason is itself a hard, documented Pulsar2
limitation, not a bug in this repo.

**First, a real infrastructure gap had to close before this was even
testable.** `build_resident_step()`'s gradient seed was `b.const(1.0)` -- baked
into the graph at *build* time, not a per-step input. No adaptive controller
(`finetune.LossScaler`'s whole reason to exist) can vary a value that isn't a
runtime input. Fixed: `grad_seed` is now a scalar graph input declared the
same way `lr` is (`qat_graph.make_step_graph`'s `scalars=`), with a new host
test (`test_grad_seed_is_a_runtime_input_that_linearly_scales_the_gradient`)
confirming `build_backward`'s own linearity in its seed holds exactly on host
before ever trusting an on-device number.

**On real hardware, on a small from-scratch Conv/Conv/Gemm step graph** (not
the full resnet18 pipeline -- built fresh to isolate the seed mechanism from
every other speed/legalization change in this doc), two builds of the same
graph via `pulsar2_docker.build(config_path=...)`, differing only in
`quant.layer_configs: [{"op_types": ["Mul","Add","Sub","Div"], "data_type":
"FP32"}]`:

| build | seed=1 | seed=100 | seed=1,000 | seed=100,000 |
| --- | --- | --- | --- | --- |
| baseline | 100% nonzero | 100% nonzero | 100% nonzero | 100% nonzero |
| FP32 elementwise | **0%** nonzero | 0.05% nonzero | **20.1%** nonzero | 20.1% nonzero |

Two things confirmed, one real limit found:

* **The FP32 override has a genuine, monotonic effect that tracks the seed.**
  In the FP32 build, growing the seed from 1 to 1,000 rescues more and more
  gradient elements from rounding to exactly zero (0% -> 0.05% -> 20.1%) --
  precisely the signature loss scaling predicts, and precisely what "the seed
  reaches the graph as a tensor pinned to a constant" (the original diagnosis)
  would make impossible. This is real, on-device, hardware-measured evidence
  the mechanism works, not a host simulation.
* **The baseline is not simply "constant regardless of seed" on this smaller
  graph** -- unlike the original resnet18 probe's bit-identical finding, here
  the per-element ratio between seed=1,000 and seed=1 is chaotic (mean -14.8,
  std 3701, over a supposed 1000x expected ratio) rather than either constant
  or proportional. The seed reaches *something* without the FP32 override,
  just not coherently -- a smaller, differently-shaped graph than the original
  characterization, so treat the *qualitative* finding (baseline doesn't scale
  correctly, FP32 does something real) as the transferable result, not the
  exact percentages.

  **Retroactive check, prompted by two later, separate findings on a
  different model** (`docs/axera-audio-speech-op-coverage.md`'s wav2vec2
  work): a scalar input never given real, varied calibration data --
  producing a degenerate MinMax range that silently clips/saturates the real
  runtime value -- broke `lr` and then `grad_seed` there (a zero-width range
  from identical calibration samples in one case, a generic small-random
  default uncorrelated with the real runtime magnitude in the other). Worth
  asking whether the *same* mechanism explains this "chaotic ratio" finding.
  Rebuilt this exact probe fresh and inspected its `grad_seed` calibration
  directly: with `make_training_calib.py`'s current generic fallback (no
  `real_data`), `grad_seed`'s calibrated range comes out `[-0.039, 0.081]`
  (`scale=0.00047`, `zero_point=83`, U8) -- the same class of narrow,
  uncorrelated range as the two wav2vec2 bugs, on a fourth model/probe now.
  But swept on real hardware (seed = 1, 100, 1000, 100,000 against this
  fresh build), the result was **0% nonzero at every seed** -- bit-identical
  zero, not the originally-reported 100%-nonzero-but-chaotic-ratio pattern.
  **Inconclusive, not confirmed or refuted**: this rebuild's baseline
  behavior does not match the one being explained closely enough to serve as
  a stand-in for it, most likely because `make_training_calib.py`'s default
  `x_scale`/`weight_scale` (and therefore the *output* `dW` tensor's own
  calibrated range, separate from `grad_seed`'s) have been retuned by the
  fixes this doc's own later sections describe, since PR #1353's original run
  -- a moving target a fresh rebuild can't reproduce. The narrow-`grad_seed`-
  calibration finding stands on its own as a fourth real instance of the
  pattern (worth the same `real_data`/jitter treatment if this probe is
  revisited), but it does not settle what specifically made the *original*
  baseline's ratio chaotic rather than constant.
* **It plateaus.** 20.1% nonzero at seed=1,000 and at seed=100,000 -- unchanged
  over two more orders of magnitude of seed. This is not the seed failing to
  reach the graph again; it is the *next* quantised boundary downstream taking
  over as the limit. The weight gradient here is produced by a `MatMul`
  (`dW = dOut^T @ X`, unavoidable for any linear/conv layer's gradient), and
  `scripts/axera/README.md`'s own prior investigation into this exact
  mechanism (a different session, a different model, same Pulsar2 version)
  already confirmed `data_type: "FP32"` **is not a valid override for `MatMul`
  or `Conv` at all** -- only a fixed, documented list of elementwise ops
  (`LeakyRelu, Sigmoid, Relu, Add, Mul, Div, Sub, Concat, Softmax`) accepts it.
  So the seed's own value survives as real FP32 through the `Reshape`/`Mul`
  chain that first consumes it, but the moment it reaches the `MatMul` that
  actually computes the gradient, that node's own INT8/16 quantisation --
  which `layer_configs` cannot touch -- reimposes a ceiling. Growing the seed
  further cannot rescue more elements past whatever that `MatMul`'s
  activation/output quantisation can resolve.

**Net effect on the original ceiling**: this should genuinely push U8's
~1,000-step and U16's ~5,000-step death points further out (more of the
gradient survives at a given true magnitude than before), but it does **not**
make training open-ended the way a true floating-point gradient path would --
the `MatMul` boundary means there is still *some* magnitude below which the
gradient dies again, just a smaller one than today. **Not measured**: a real
multi-thousand-step training run with `LossScaler` driving the seed adaptively
against this FP32 build, to quantify exactly how much further the ceiling
moves in steps/SNR -- a substantial follow-on this pass didn't reach, since
establishing that the mechanism works at all (the infrastructure gap, the
sweep, and the `MatMul` limit) filled the available time. That run, plus
re-enabling `LossScaler` against a real `grad_seed` input instead of the
`ineffective` stand-down path, is the concrete next step.

A second obstacle waits behind it either way: fixed-point clipping is **silent
and local**. It happens to intermediates that never reach an output, so a
controller reading the returned gradient sees a healthy tensor while the
computation upstream is destroyed -- unlike fp16, where overflow makes an inf
that propagates. Reliable back-off needs the graph to export `ReduceMax(|t|)`
on those tensors.

### Two more angles on controlling gradient quantization directly, both closed with real evidence

Following the `MatMul`-boundary finding above, two further angles were tried
against a small, purpose-built `MatMul`-only probe graph (`y = x @ w`,
`loss = sum(y^2)`, `dW = grad(loss, w)` seeded by a real `grad_seed` runtime
input -- isolates the exact node the ceiling lives at, smaller and faster to
iterate than the Conv/Conv/Gemm probe above). Both close cleanly negative,
with compiler-level evidence rather than speculation.

**Angle 1: force the gradient-producing `MatMul`'s own output to `FP32` via
`layer_configs`' `output_data_type` field** -- a real, separate proto field
from `data_type` (which `MatMul` cannot use at all), documented for `Conv`
("quantize weight type for Conv" / "quantize data type for Conv" in
`build_config.proto`'s own comments) but not textually restricted to it.
Tried both ways `layer_configs` can select a target: `op_types: ["MatMul"]`
(matches every `MatMul` in the graph) and `layer_names: ["matmul_14"]` (the
exact, confirmed post-fusion name of the gradient-producing node, read out of
a first build's `quant_axmodel.json` per this doc's own established
"build once, target the surviving name" method). **Both silently downgrade to
U8.** `quant_axmodel.json`'s own `quant_config.mix_precision_configs` records
the request (`{"MatMul": {"dtype": "U8"}}` -- not absent, so the selector
matched something) but the *value* it settled on is `U8`, and `matmul_14`'s
own `tensor_configs` entry confirms it: `bit_width: 8, quant_min: 0,
quant_max: 255`. Pulsar2 acknowledges the request and refuses it, the same
"asks may be silently downgraded" shape `scripts/axera/README.md`'s own
LayerNorm/`TileFailException` finding already established for a different op,
just via a quieter failure mode here (no build error, no exception -- only a
config file that says "no").

Decomposing the `MatMul` into `Mul` + `ReduceSum` (both being, in principle,
individually-configurable ops) was considered but not built: `ReduceSum` is
not on the confirmed `data_type: FP32`-eligible list (`LeakyRelu, Sigmoid,
Relu, Add, Mul, Div, Sub, Concat, Softmax`, from this doc's own earlier
finding), so the *reduction* -- the actual op whose output becomes the
gradient tensor -- would still hit exactly the same quantized-output wall,
just moved one node later. Combined with the `output_data_type` result
above (Pulsar2 refuses a *specific, correctly-named* gradient-producing node
an FP32 output), the wall looks structural rather than `MatMul`-specific:
**whichever op produces the final gradient tensor, Pulsar2 quantizes its
output, and `layer_configs` has no lever over that specific tensor's own
output precision.**

**Angle 2: skip fighting for FP32 upstream, and instead directly re-narrow
the gradient output's already-quantized scale/zero-point post-hoc**, using
`scripts/axera/emitter.py`'s existing `learn_mcode`/`nudge_output_quantisation`/
`emit_mcode` machinery (built for a different original purpose -- writing new
weights into a compiled `.axmodel` without recompiling). The idea: track the
shrinking true gradient by periodically patching the compiled mcode's output
quantization range, the same principle as loss scaling but applied to the
*output* tensor's own quantization parameters rather than an input that has
to survive an entire graph unmolested.

**Closed by the same finding this doc's mcode-quantize-elimination section
already made, now reproduced on a different, much smaller model**:
`learn_mcode`'s whole method depends on Pulsar2 compiling the *same shape*
close enough to byte-stably that a target value's own bytes (its scale/zero
literal) can be told apart from everything else that also varies build to
build. Tested directly: compiled the exact same probe graph, same
calibration data, same config -- twice. **1,249 of the compiled mcode's 4,704
bytes (26.6%) differ between the two identical-input builds.** For
comparison, deliberately changing the calibration data's weight scale by 10x
(to shift the real gradient range, which is what a re-narrowing patch would
need to reliably target) moved a *similar* number of bytes (1,163 of 4,704,
24.7%) -- meaning the noise floor from pure recompile non-determinism is as
large as, or larger than, the signal from an actual, deliberate range change.
There is no byte-level signal to separate "this moved because the target
range changed" from "this moved because Pulsar2 recompiled the same inputs
differently," on this model, with this method. This is not a smaller-model
fluke: it is the identical mechanism PR #1344's mcode-quantize-elimination
probe already found on the (larger, different) resnet18 training step,
confirmed to generalize rather than being an artifact of that specific
model's size or shape.

**Net verdict on both angles: closed, not open questions.** Neither
`layer_configs` nor post-hoc mcode patching gives real control over the
gradient tensor's own output precision on this compiler. The FP32 seed
(above) remains the only angle with a measured, real, if partial, effect --
and it does nothing for the boundary these two angles were trying to reach
past. Moving the ceiling further than the FP32 seed already does would need
either a Pulsar2 capability that doesn't exist in the config surface explored
so far, or solving the compiler's own byte-level non-determinism first (a
precondition for Angle 2 that this project has no access to, being
closed-source) -- not a small follow-on to either angle tried here.

### The FP32 seed, the quantizer's own internals, and two more levers -- one real, one dead

Follow-on work (branches `axera-fp32-gradient-seed` / PR #1353, `axera-quantizer-
reverse-engineering` / PR #1354, not yet merged as of this section) took the
"untried way out" above further. Confirmed the FP32 seed is real but plateaus:
forcing the seed's elementwise `Mul` to `data_type: "FP32"` rescues 0% -> 20% of
gradient elements as scale grows, then stops, because the gradient itself comes
out of a `MatMul` and `layer_configs`' `data_type` override does not apply to
`MatMul`/`Conv` at all (confirmed directly in the compiler's own
`quant_axmodel.json`: the request is recorded but silently downgraded to U8).
Reverse-engineering Pulsar2's calibration algorithm from archival build data
then confirmed the range is a function of calibration *data*, not graph
structure -- and found two more untried config-surface levers, `output_data_type`
(a `LayerConfig` field distinct from `data_type`) and calibrating deliberately
for the gradient's expected late-training scale. Tested both directly, on a
minimal isolated `y=x@w`/`dW=grad(loss,w)` probe (`scripts/axera/
build_matmul_grad_probe.py`) rather than the full pipeline, for fast iteration:

**`output_data_type: "FP32"` on the gradient `MatMul`: dead, more silently than
`data_type`.** Pulsar2's own protobuf source
(`/opt/pulsar2/axnn/yamain/config/build_config.proto`) documents this field's
own comment as *"quantize data type for **Conv**"* -- a different, Conv-scoped
control, not a generic per-layer output-precision override. Tried anyway,
targeting the gradient MatMul both by `op_types: ["MatMul"]` and by its exact
post-fusion `layer_name` (`matmul_14`): both compile successfully, and both
leave `matmul_14`'s own tensor config **byte-for-byte identical** to the
baseline's (`bit_width: 8, quant_min: 0, quant_max: 255`, identical hash) --
and unlike the `data_type` case, `mix_precision_configs` doesn't even record
the request (empty `{}` in every variant). The override isn't downgraded; it's
not recognized for this op type at all.

**Calibrating for the expected gradient scale: real, exact, and confirmed on
real hardware -- but a one-shot, build-time trade, not an adaptive fix.**
Compiled the identical probe twice, differing only in the calibration data
supplied for `grad_seed` -- `1.0` (this thread's default so far) vs. `1e-4`
(a late-training-scale stand-in). The compiler's own recorded output scale
moved by **exactly 10,000.00x** (`0.0966 -> 9.659e-6`), matching the calibration
ratio to five significant figures -- textbook linear MinMax, not an
approximation. On real AX650N hardware, feeding both compiled models a real
`grad_seed = 1e-4`:

| model | seed | nonzero elements | dW[0] |
| --- | --- | --- | --- |
| baseline (calibrated for seed~1.0) | 1e-4 | **0/128** | 0 |
| recalibrated (calibrated for seed~1e-4) | 1e-4 | **115/128** | -2.8977e-05 |
| baseline | 1.0 | 115/128 | -0.28977 |
| recalibrated | 1.0 | 115/128 | **-2.8977e-05** (identical to its own 1e-4 result) |

The recalibrated model recovers the exact late-training gradient (`-2.8977e-05
= -0.28977 x 1e-4`, matching the seed's own linearity precisely) at the scale
it was built for. The fourth row is the real limit, not a footnote: fed the
*old*, large seed, the recalibrated model returns the **identical, saturated**
value it gives for the tiny seed -- it has lost the ability to represent large
gradients, clipped at the top of its now-much-smaller range. **This settles
the adaptive-vs-one-shot question directly**: a single compiled model cannot
serve both an early-training and a late-training gradient scale at once. Real,
practical shape of the win: calibrate for the training regime a given deployed
model will actually run (a short fine-tuning run, or a specific stage of a
longer one), or swap to a differently-calibrated recompiled model at defined
checkpoints as training progresses -- not a continuously-adaptive per-step
scheme, since nothing about Pulsar2's build-time calibration is revisitable at
runtime.

Combining both levers (recalibration + `output_data_type` override) produced a
compiled model byte-identical in every quantization-relevant field to
recalibration alone -- confirming, independently, that `output_data_type` truly
contributes nothing on `MatMul`.

### Multi-phase calibration swap: the mechanism, hardware-verified past the isolated probe

Turned the recalibration lever above into a real, working demonstration
rather than leaving it as a design note. Not on the full resnet18 pipeline --
`resnet18d_folded.onnx` (the forward model earlier sections' compiles used) is
no longer present in this session's scratch, and regenerating it (BN-folding
included) was out of scope for the time this task had -- but on
`tests/test_build_resident_train_step.py`'s own small real forward model
(`x -> Conv -> Relu -> Flatten -> Gemm -> logits`, trainable `cw`/`gw`)
through the **real, unmodified pipeline**
(`add_mse_loss`/`build_resident_step`/`pulsar2_docker.build()`) -- a genuine
multi-node training-step graph (35 nodes after simplify), not the single-`MatMul`
isolated probe the levers above were tested on. `gb` (the Gemm bias) was left
frozen: trained, it hits a real, separate compiler crash
(`TileFailException("AxQuantizedSub, tuple index out of range")` on the SGD
subtract for that specific tiny 10-element tensor) unrelated to anything this
task investigated -- worth a bug report if this model shape is revisited, not
chased further here.

Built two compiles, identical graph, differing only in `make_training_calib`'s
`weight_scale` (0.05 -- "early training" -- vs. 0.0005 -- "late training", the
same 100x ratio methodology as the isolated-probe result above), each in
**~15s** (this graph's own size, not resnet18's -- the batching section's
70-450s figures don't apply here). Confirmed the compiler-recorded scales
move by **exactly 100.00x** between the two builds, for both trainable
tensors and their SGD-updated state outputs alike (`quant_axmodel.json`'s
`tensor_configs`/`values`, same evidentiary method as the isolated-probe
result):

| tensor | phase 1 scale (weight_scale=0.05) | phase 2 scale (weight_scale=0.0005) | ratio |
| --- | --- | --- | --- |
| `cw` | 0.0012102693 | 1.2102693e-05 | 100.00x |
| `cw`'s updated state | 0.0012102675 | 1.2102679e-05 | 100.00x |
| `gw` | 0.0013282200 | 1.3282201e-05 | 100.00x |
| `gw`'s updated state | 0.0013282195 | 1.3282196e-05 | 100.00x |

This confirms the isolated probe's finding generalizes past one `MatMul` to a
real multi-node graph with a real Conv, a real SGD update, and simplify()
in the loop.

**Real hardware demonstration**: fed the identical late-training-scale weight
state (`cw`/`gw` values ~100x smaller than phase 1's own calibration) and the
identical batch/lr into both compiled models, 5 steps each:

| model | fed | `cw[0]` before | `cw[0]` after | relative update |
| --- | --- | --- | --- | --- |
| phase 1 (calibrated for scale 0.05) | late-scale weights | 0.000152358538 | **0** | dead |
| phase 2 (calibrated for scale 0.0005) | the same late-scale weights | 0.000152358538 | **0.000157334827** | **1.03266x** |
| phase 1 (reference) | its own matching early-scale weights | 0.0152358543 | 0.0157334786 | 1.03266x |

Phase 2 recovers the **exact same relative update** (1.03266x) that phase 1
gets on weights at its own calibrated scale -- a real, correctly-proportioned
SGD step -- while phase 1 fed the same late-scale state loses it completely,
landing on exactly zero. This is the "swap to a recalibrated model when the
current one's gradient dies" mechanism working end to end: the same weight
*values* handed from one compiled model to the next (via ordinary host-side
files -- `resident_runner.c`'s existing `.state<k>` convention, no
graph-level coupling between the two compiles needed), only the calibration
differs.

**What this does and doesn't prove.** This demonstrates the mechanism with
one real transition (2 phases, chosen and swapped by hand) -- not a 3-phase
sweep, and not an automatic controller. Real, open costs and questions for
whoever builds on this:

- **A phase swap costs a full recompile** -- ~15s on this small graph, and
  per the batching section's own numbers, 70-450s+ on the real resnet18
  pipeline. Not free, and not something to do every few steps.
- **How many phases a genuinely long run needs, and where to place the
  boundaries, is open.** This demo used one 100x jump because that's the
  ratio already validated on the isolated probe; a real schedule would likely
  want smaller, more numerous steps, chosen from the actual gradient-decay
  curve of a real training run rather than picked by hand.
- **Triggering is manual here.** `finetune.LossScaler`'s existing
  `zero_fraction`/`DEFAULT_UNDERFLOW` detector is the natural fit for
  deciding *when* to swap (its role changes from "grow/backoff a scale" to
  "signal a model swap"), but wiring that up, and actually swapping the
  running `resident_runner` process out from under a live loop, is
  unbuilt.
- **Hiding the recompile cost** (building phase N+1 speculatively while phase
  N is still training, so the swap is instant when it's needed) is a real,
  unexplored option given a spare CPU core and Docker daemon are cheap
  relative to the AX650N itself.

### Reducing the per-phase recompile cost: what actually drives it, and one real lever

Real recompile cost matters at scale: at resnet18 batch 8's ~28.5 steps/s and
a ~5,000-step-per-phase ceiling, one phase's own training time (~175 s) is
*shorter* than its own recompile (70-450 s+ depending on batch) -- a long
multi-phase run would spend more wall-clock recompiling than training.
Investigated four ways to cut that cost, on a real resnet18 step graph
(`resnet18d_folded.onnx`, last-4-layers trainable, rebuilt fresh from
`scripts/axera/resnet18-dedupe-fanout` scratch since the earlier note that it
was "no longer present" turned out to be about a different scratch dir):

**1. Staged timing, real breakdown.** A real batch-1 build's own timestamped
log splits cleanly: extract/frontend-optimize (~0.8 s), quant/calibration
passes (~2 s), then **`compile npu subgraph` -- the NPU backend's own
tiling/dependency/execution-unit-assignment scheduler -- for ~32 s** (`build
op serially` 7 s, `calc output dependencies` 8 s, `assign eu heuristic` 9 s,
`build jobs` 4 s), then `assemble model`/`fuse subgraph` for another ~12 s.
**The NPU backend scheduling stage is where the time actually goes** -- not
quantization/calibration, confirming (from the compiler's own log, not
inference) that recalibration was never going to be free regardless of how
it's triggered, because the expensive stage runs on the *quantized* graph
regardless of what its calibration values are.

**2. No documented mode skips it.** `pulsar2 build --help`'s full option
list (`--quant.*`, `--compiler.*`, `--model_type {ONNX,QuantAxModel,
QuantONNX}`) has no incremental-build, cache, or scheduler-reuse flag.
`--model_type QuantAxModel` (feed an already-quantized graph in) would only
skip the ~2 s quant stage, not the ~32 s scheduling stage that dominates --
not worth pursuing for this specifically.

**3. Determinism-pinning re-checked, still dead, now with a real number at
this scale too.** Rebuilding the *identical* batch-1 graph+calibration twice
produced `compiled.axmodel`s of different length (8,955,471 vs 8,955,431
bytes) with 1.9% of the shared prefix differing -- confirms PR
#1344/#1353's non-determinism finding generalizes to this model, and no
`--seed`/determinism flag exists in the CLI to pin it. mcode-level
patching remains dead for the reason already established: recompile noise
and a genuine calibration change are not reliably distinguishable in size.

**4. The frozen/trainable split: real, but not for the reason expected.**
Hypothesis: since only the trainable tail's calibration changes between
phases, compiling *only* the trainable tail (feeding the frozen backbone's
boundary activation in as a graph input, frozen backbone compiled once and
reused) should make the per-phase recompile track the tail's size, not the
whole model's. Tested by cutting the graph at the real block boundary
(`/layer3/layer3.1/act2/Relu_output_0`, the tensor feeding both layer4's
main and shortcut branches) via manual forward-BFS extraction (`onnx.utils.
extract_model` hit an internal topological-sort bug on this graph -- worked
around, not investigated further, not this project's bug to fix) --
**56-node frozen-included forward down to a 15-node tail-only forward, 136
total step-graph nodes down to 95. Compile time: unchanged.** The NPU
backend stage was still ~32 s (31.97 s vs. the full model's 31.95 s) --
removing 41 frozen nodes bought **nothing**. Went further: cut all the way
to *only* `fc.weight` trainable (a single `Gemm` forward, one node, 16-node
step graph) -- **that** compiled in 16.7 s total, NPU backend stage collapsed
to **0.18 s**. **The real driver isn't total node count or which part is
frozen -- it's the trainable tail's own convolution-backward complexity**
(the im2col-as-gather tap expansion `_linearize_trainable_convs` emits for
each trained `Conv`, not anything in the frozen backbone). Splitting frozen
from trainable is architecturally sound (AXCL can chain two resident models
via a host-mediated device-to-device copy between `Execute()` calls, the
same pattern this project's vNPU-concurrency work already uses for multiple
resident models) but **doesn't save anything on its own** when the trainable
tail still contains the expensive part -- confirmed by measurement, not
assumed, so not prototyped further given it wouldn't pay off as hypothesized.

**The one real, actionable lever this surfaced**: per-phase recompile cost
scales with how many convolutions are in the *trainable* tail, not with
model depth or total node count. A multi-phase schedule that trains fewer
(or zero) convolutions per phase -- e.g. `fc.weight`-only phases, falling
back to the full last-4-layers scope only for an initial or final phase --
would make most phase transitions cost ~17 s instead of ~70-450 s+, at the
price of a narrower trainable scope for those phases. Untested: whether
`fc.weight`-only phases still rescue the gradient the way the 2-phase
Conv+Gemm demo above did -- a real next step, not assumed to follow from
this compile-time finding alone.

## Speed

`axcl_run_model` costs ~580 ms per invocation (process start, device open,
model load) plus ~3 ms per inference, and the LXD plumbing adds ~800 ms more in
`lxc exec` calls and file copies. AXCL exposes load-once/run-many
(`axclrtEngineLoadFromFile`/`CreateContext`/`CreateIO`/`Execute`), so a
~120-line resident runner does the fixed work once:

| | per step |
| --- | --- |
| `axcl_run_model` | ~1400 ms |
| resident runner, 16-channel step | **1.70 ms** |
| resident runner, resnet18 step | 200.6 ms (42.9 MB moved) |

**820x** on the small step: 20,000 training steps in 30 seconds instead of
eight hours. Past that the cost is transfer, not overhead, and residency is the
lever:

| 1024x1024 step | ms | moved |
| --- | --- | --- |
| send every input, read every output | 63.13 | 12.583 MB |
| weight resident on the card | 41.03 | 8.389 MB |
| weight resident, gradient not read back | **29.79** | 4.194 MB |

**Double buffering is not worth it.** Feeding an updated weight back by
device-to-device copy (27.57 ms) and by swapping the two device pointers
(28.97 ms) are indistinguishable from no feedback edge (27.64 ms). A 4 MB copy
inside the card's own DRAM is free; the two `Set*BufferByIndex` calls a swap
needs cost more than the copy it avoids. The card has 7040 MiB of CMM with 12
in use.

Batch size is nearly free until it is not: batch 1 to 16 costs 0.38 ms
(1.40 -> 1.78) for **12.6x** the throughput; batch 64 is worse *per sample*
than 16.

### Graph-computed calibration/saturation signals: the mechanism works, and it's redundant on the tensor everyone tried it on

The original ceiling note above says the fix plainly: *"fixed-point clipping
is silent and local... a graph that wants reliable back-off needs to export
`ReduceMax(|t|)` on those tensors as extra outputs."* Nobody had built this
until now. `scripts/axera/build_calib_signal_probe.py` adds it: an
`Abs`+`ReduceMax` tap on the gradient tensor, exposed as a real extra graph
output alongside the existing `state`/`loss` outputs (`qat_graph.make_step_graph`
has no generic "extra output" parameter, so this reimplements
`build_resident_step`'s body directly, the same choice
`build_multiphase_calib_swap_probe.py`/`build_matmul_grad_probe.py` made for
their own one-off experiments -- and adds the output *before* running
`simplify()`, so common-subexpression/dead-code elimination has no chance to
drop a branch that would otherwise head nowhere).

**The mechanism is real and verified.** Both ops are on `AX650_SUPPORTED_OPS`
(confirmed, not assumed). The graph survives `legalize`/`simplify` with the
signal outputs intact -- `onnxsim.simplify()`'s own contract (preserve
declared inputs/outputs) held exactly as documented. Cross-checked against an
independent finite-difference gradient on host: the graph's own
`ReduceMax(Abs(grad))` output matched `np.abs(finite_diff_grad).max()` to
**0.02%** (ratio 0.9998), for both trainable tensors in a real (if small)
Conv/Relu/Flatten/Gemm training step -- the tap is computing exactly what it
says it computes.

**But tapping the *final* gradient tensor -- the one everyone's first
instinct reaches for, including this investigation's own first attempt --
turns out to be redundant, and it's worth stating plainly why.** The host
already reads the gradient tensor back directly, every step, to apply the
SGD update. `np.abs(returned_grad).max()` is computable from data the host
loop already has, with no graph change at all. Adding a graph-computed
`ReduceMax` on that *same* tensor duplicates information already available,
for the cost of an extra output (however cheap) -- it does not, and cannot,
give a *leading* indicator of anything, because it is a direct function of a
tensor the host was never blind to in the first place.

**Where the technique actually pays for itself** is exactly where the
original note said: on tensors that do *not* otherwise reach any declared
output -- an intermediate accumulator inside the backward pass (e.g. the
gradient-seed's own `Mul` output, or an intermediate reduction stage before
the final gradient is assembled) that can silently saturate or underflow
*without the final gradient revealing why*, since fixed-point clipping is
local to wherever it happens, not visible downstream. This investigation's
own test model -- a small, shallow Conv/Relu/Flatten/Gemm step, chosen
deliberately for fast iteration the same way PRs #1353/#1355/#1356 did --
does not have a rich enough backward chain to exhibit an intermediate that
saturates independently of the final gradient it feeds; demonstrating the
technique's *real* value needs a deeper graph (more layers between the
instrumented intermediate and any existing output) than this probe has.
**Verdict: the mechanism is proven and cheap to add; its value is unproven on
this graph because this graph's backward pass has no hidden intermediate
worth instrumenting** -- not because the idea is wrong, but because the toy
model that makes iteration fast is also too shallow to need it. The natural
next probe is a graph with real depth between an internal tensor and its
nearest existing output (the real resnet18/Whisper backward passes both
qualify) with the tap placed on something genuinely internal, not the final
gradient.

Real hardware was not needed to reach this verdict -- the redundancy argument
holds from the tensor's own definition (it's the thing the host already
reads), independent of anything Pulsar2's compiler does to it, so this stayed
entirely host-side.

**A related, second question -- deriving calibration data from something
computed rather than hand-picked -- was not built, but the groundwork this
thread already has makes the answer fairly confident without a new
experiment.** PRs #1354/#1355 established, to textbook-MinMax precision and
an exact 10,000x/100x calibration-ratio match on real hardware, that
Pulsar2's calibration range is a direct, precise function of whatever
calibration data is supplied -- the two prior probes' 0.05/0.0005 and
100x-ratio choices were hand-picked round numbers, not derived from anything.
Replacing that guess with the real magnitude trajectory a host-side float
reference run of the same training step actually produces (record N float
SGD steps' true gradient magnitudes, use the value the run has reached at
each intended phase-transition point as that phase's calibration data,
instead of guessing a round number) is a straightforward workflow change to
`scripts/axera/make_training_calib.py`, not a new mechanism -- the underlying
calibration-precision result is already proven. Left as a follow-on rather
than built here: this needs its own real multi-phase build-and-swap run to
show the *difference* a computed magnitude makes over a hand-picked one
(which requires the AX650N/Docker toolchain), and this task's time went to
verifying the graph-signal mechanism above instead.

### Weights resident with in-graph updates: 7.0x, measured

The 42.9 MB the resnet18 step moves is every trainable weight crossing the
host boundary twice -- once in as a graph input, once back out as the
gradient the host then applies. `scripts/axera/build_resident_train_step.py`
puts the update itself in the graph instead: each trained weight is a
`qat_graph.StepGraph`-style *state* tensor (both input and output, `w_next =
w - lr * grad` computed by ordinary `Mul`/`Sub` nodes), so
`scripts/axera/tools/resident_runner.c` can copy each step's output buffer
straight back into its own input buffer device-to-device and never send the
weight across the host boundary at all -- only the batch (`x`, `y`) goes in
and the scalar loss comes out.

Same four tensors, same `resnet18d` shape, real AX650N, 30 timed steps after
5 warmup:

| | avg | min |
| --- | --- | --- |
| baseline (host applies `w -= lr*grad`, 42.9 MB/step) | 200.6 ms | -- |
| in-graph update, **non-resident** (`-n`: state round-tripped through host) | 114.8 ms | 110.6 ms |
| in-graph update, **resident** (state copied device-to-device) | 38.3 ms | 36.0 ms |
| resident + `_linearize_trainable_convs` (no weight transpose) | **28.6 ms** | 26.3 ms |

**5.2x** from residency alone, not quite the "roughly 10x" estimated -- real,
and short of the estimate for a real reason (below), not a measurement
artifact: the non-resident row isolates that some of the win is just the
in-graph update itself (fewer distinct tensors cross the wire even before
residency helps), and the resident row is the full effect. Removing the
weight-transpose tax (last row, see below) pushes the total to **7.0x**
(200.6 ms -> 28.6 ms).

A `--compiler.npu_perf` profile of the resident graph (`pulsar2_docker.
build(profile=True)`, see `scripts/axera/README.md`'s "Real NPU profiling"
section) explains the gap from 10x: op-type cycle share for this step is

| op type | share of cycles |
| --- | --- |
| `AxQuantizeLinear` + `AxDequantizeLinear` | 48.8% |
| `AxTranspose` + `AxSlice` | 26.3% |
| `assign` (state write-back) | 6.7% |
| `AxQuantizedMul`/`Sub`/`MatMul`/`ReduceSum` (the actual backward arithmetic + SGD update) | 16.2% |
| `AxQuantizedConv` (the untouched forward convs) | 0.9% |

**Three quarters of the NPU's own cycles are quantize/dequantize and
transpose/slice glue, not arithmetic** -- the tax `act_weight_conv_to_matmul`
pays for turning a live-weight `Conv` into per-tap `MatMul`s (each tap needs
its own `Slice` and `Transpose`, and apparently its own quantization
boundary). Residency removed the *transfer* bottleneck; this is what is left,
and it is now the bigger one.

### The quantize redundancy is real, and not fixable from the ONNX side

Each trainable weight is read directly by **three** nodes -- the forward
conv-as-matmul's weight-transpose, its own gradient's reshape, and the
in-graph SGD `Sub` -- and Pulsar2 inserts a **separate `AxQuantizeLinear` per
edge** rather than sharing one quantized copy: confirmed on the compiled
graph, `fc.weight` and all three `layer4` weights are each quantized 2-3
times, while **no ordinary multi-consumer activation in the same graph is
ever requantized more than once**. Checking the quantize nodes' own
parameters found the redundancy is partly real: two of the three edges for
`layer4.1.conv1`'s weight (the forward-matmul path and the gradient-reshape
path) quantize to the *identical* domain (`S8`, `scale=0.00209808`,
`zeropoint=0`) -- genuinely the same computation, done twice. The third (the
SGD-update `Sub`) quantizes to a different domain entirely (`U8`,
`scale=0.00209251`, `zeropoint=128`), so that one is not redundant: the
update path legitimately needs its own quantization range.

**Tried and failed: routing all three edges through one shared node.** Two
spellings, both mathematically identity and both checked bit-exact against
the un-rewritten graph on host (`onnxruntime`, max abs diff `0.0`):

| shared-node spelling | result |
| --- | --- |
| `Reshape(w, same_shape)` | no change |
| `Mul(w, ones_like(w))` | no change |

Both compiled to the **exact same 213-node optimized/quantized graph**
(`frontend/optimized_quant_axmodel.onnx`) and the **exact same
`max_cycle`** (31,369,216) as the unmodified graph -- Pulsar2's own frontend
optimizer canonicalizes a same-shape `Reshape` and a multiply-by-a-literal-
all-ones-constant as identities and removes them **before** its
quantization-boundary insertion pass runs, independent of anything onnxsim
did upstream (both variants only needed `onnxsim.simplify()`'s
`eliminate_nop_reshape` to *not* run, which it didn't here since these were
inserted after the last `simplify()` call -- and it made no difference,
because Pulsar2 does the same collapse internally regardless). This is a
different failure shape than the `Gemm`-reconstruction bug elsewhere in this
document: that fix worked by changing *which pattern matches* (a 1-D bias
vs. a `[1, N]` one); there is no equivalent lever here, because the thing
being matched is generic identity-elimination, not a specific fusion
pattern with a shape precondition to dodge.

**So this one op-type share is confirmed structural, not an oversight
onnxsim's graph shape controls.** Whatever decides Pulsar2 requantizes a
live-weight input per direct consumer instead of per distinct
(source, quantization-domain) pair is internal to Pulsar2's own frontend
compiler; there is no ONNX-graph-level lever this project has access to that
moves it. Not investigated: whether Pulsar2's `layer_configs` can pin one
named intermediate's quantization domain such that two edges are *forced*
into the same domain by construction rather than merely happening to match
-- worth trying if this is revisited, but it wasn't tried here since the two
redundant edges already match by calibration coincidence, not by any config
this project controls, so forcing it would need to survive the same
collapse just demonstrated.

Correctness was checked the same way as the rest of this document -- a
directional-derivative check against `onnxruntime` on host (not the on-device
number itself, which was not re-measured to gradient precision this time;
see `tests/test_build_resident_train_step.py` for the from-scratch,
no-hardware version of that check) -- and a coarse on-device sanity check: a
near-zero dummy batch's loss landed at 0.1416 on the card against 0.0972 from
the fp32 host reference, the right order of magnitude for INT8 quantization
noise on an untuned calibration set, not a wiring bug.

**One level lower, also closed: `docs/axera-mcode-quantize-elimination-probe.md`**
investigated whether the same redundancy could be removed by patching the
*compiled mcode* directly (this project's reverse-engineered NPU
command-queue codec, `scripts/axera/mcode.py`/`emitter.py`) rather than the
ONNX graph. Confirmed the exact redundant pair at the instruction level
(`fc.weight`'s two S8, same-scale/zeropoint `AxQuantizeLinear`s) but found
the win available even in the best case is small (~1.5% of step cycles, not
the 48.8-56.7% the op-type total suggests -- most of that total is the two
trainable convolutions' *legitimately* different raw-layout vs.
im2col-tap-layout quantizations, not redundant copies), and found a harder
blocker than expected: recompiling the *identical* graph and calibration
data a second time produced a **63%-different mcode blob** (and a different
length), despite an identical compiler cost estimate (`max_cycle`) both
times -- so any address-level patch would need to be rederived per build,
not learned once. Concluded not feasible to pursue further with current
understanding; see that doc for the full evidence and what would change the
conclusion.

### The transpose/slice half was fixable, from the ONNX side -- 28% more

Unlike the quantize half, the `AxTranspose`/`AxSlice` 26.3% *was* squarely
onnxsim's own graph shape, and a real fix landed. The suspect going in was
per-tap transpose duplication (`act_weight_conv_to_matmul` redoing the
activation transpose once per tap instead of once per convolution) -- reading
the rule directly showed that hypothesis was **wrong**: the transpose is
already hoisted once per convolution, both for the activation and the weight.
The actual cost was almost entirely (89.6% of the whole step's `AxTranspose`
cycles) the **weight** transpose alone, `[Cout, Cin, k...] -> [k..., Cin,
Cout]`, on exactly the two 512-channel trainable convs -- large enough
(2.36 MB) that Pulsar2 shards it into 16 hardware sub-instructions per
occurrence, each costing the same ~184K cycles, and it is recomputed from
scratch on every `Execute()` even though the weight it operates on is now
resident state that barely changes step to step.

**Why not just pre-transpose the state once and keep it in that layout.**
That was the first attempt, and it numerically works (a finite-difference
check confirmed it) but is not what shipped: `act_weight_conv_to_matmul`'s
own construction needs `Pad`/`Slice`/`Concat`, and `onnxsim.graph_grad` has
no gradient rule for any of the three -- built-in or registerable without
hand-writing three new ones (`graph_grad.register_gradient`/`custom_gradient`
exist for exactly this, but three new rules is a materially bigger, riskier
change than what actually fixed this).

**What shipped instead: avoid needing a weight transpose at all.**
`onnxsim.graph_grad._grad_conv` already differentiates a live-weight `Conv`
without emitting a convolution, by the same im2col identity its own
docstring spells out:

```
col[c, t, o] = X[c, position(o, t)]      (im2col: one Gather)
Y[m, o]      = sum_{c, t} W[m, c, t] * col[c, t, o]
```

-- where `W` reshaped to `[M, C*K]` is `w.reshape(Cout, -1)`, a **free**
C-order reshape of the weight's original `[Cout, Cin, k...]` layout (`Cin`
is already the second axis, `k...` already trailing), unlike
`act_weight_conv_to_matmul`'s `[k..., Cin, Cout]`, which moves `Cout` from
first to last and is what actually costs. `build_resident_train_step.py`'s
`_linearize_trainable_convs` now builds the **forward** pass this same way
-- reusing `graph_grad`'s own `_conv_geometry`/`_im2col_indices` so the
index/mask tables are exactly the ones `_grad_conv` would derive for the
same node -- for every trainable `Conv`, *before* `build_backward` ever
runs. Since `Gather`/`Mul`/`MatMul`/`Reshape` are all builtin-differentiable,
`build_backward` needs no custom gradient registration at all, and the
gradient comes out in `w`'s original, unchanged shape -- state stays exactly
the shape it always was, `params`/`state`/`shapes[p]` unaffected.

Verified on host first (`tests/test_build_resident_train_step.py`'s new
`test_linearize_trainable_convs_matches_conv_and_drops_the_weight_transpose`:
the two resnet18 geometries this actually has to handle -- a strided, biased
1x1 downsample and a padded, stride-1, biased 3x3, with and without bias --
matched plain `Conv` on `onnxruntime` to `1e-4`, and no `Transpose` reads the
weight), then on real hardware, same calibration/build methodology as the
quantize-half check above:

| | max_cycle | step time (avg / min) |
| --- | --- | --- |
| before (weight-transpose per step) | 31,369,216 | 37.5 ms / 35.4 ms |
| after (`_linearize_trainable_convs`) | 22,625,104 | **28.6 ms / 26.3 ms** |

**-27.9% max_cycle, -23.6% step time** (28.07M vs the naive 45.56M cycle-sum
this section's profile table was built from -- **-38.4%** by that metric).
Fresh profile of the "after" graph: `AxTranspose` fell from 6,762,810 cycles
(14.8% of the old total) to 228,012 (0.8% of the new, smaller total) --
essentially gone, and better than the design aimed for: Pulsar2's own
lowering turned the `Gather`s this rule emits into native `AxSlice`
instructions wherever the index pattern was regular enough to allow it (no
`AxGather` appears in the new profile at all), cheaper than asking for a
`Gather` outright. `AxSlice` itself dropped too, 5,239,500 -> 3,550,346
cycles. `AxQuantizeLinear`/`AxDequantizeLinear` are now the dominant cost by
a wide margin (56.7% combined of the new, smaller total) -- `AxDequantizeLinear`
is unchanged in absolute cycles (6,553,923, exactly the old number), which is
the same structural quantize-per-consumer tax the previous section already
found and closed; not reinvestigated here.

Committed as `scripts/axera/build_resident_train_step.py`'s
`_linearize_trainable_convs`, exercised by the new test above plus the
existing `test_state_output_is_sgd_update_of_the_input`/
`test_in_graph_gradient_matches_finite_differences` (unchanged, still
passing -- this function changes nothing any test outside it observes,
by design).

### Batching: real, and confirms the "not enough arithmetic" diagnosis

At batch 1, the resident step's own real op shapes (from a real
`pulsar2 build --compiler.npu_perf` profile, the same one "The transpose/slice
half" section's numbers came from) sum to 207,564,800 MACs of useful
arithmetic -- confirmed against Pulsar2's own `group 0 QuantAxModel macs:`
build-log line at batch 16/32/64, which reports exactly 16x/32x/64x that
figure, so per-sample compute is exact and batch-invariant, as it should be
for a network with no cross-sample interaction. Against the AX650N's rated
**18 TOPS INT8**, batch-1's 415.13M FLOPs/step over a 26.9 ms (min) step is
**15.4 GOPS achieved -- 0.086% of rated throughput.** That is not
inefficiency at the achieved rate; it is that a 64x64, ~5.4M-trainable-param
training step simply has too little arithmetic per call to occupy an 18 TOPS
chip, and 44-83% of the cycles it does spend are quantize/transpose/slice
glue rather than MACs (see the two sections above). Batching should raise
achieved throughput roughly with batch size while adding much less than
proportional latency -- confirmed here, not just assumed from an unrelated
step elsewhere in this doc's own history:

| batch | step time (min / avg) | samples/s | achieved GOPS (min) | rated-TOPS utilization |
| --- | --- | --- | --- | --- |
| 1  | 26.9 ms / 29.2 ms | 34.2  | 15.4  | 0.086% |
| 4  | 27.2 ms / 30.4 ms | 131.6 (3.85x) | 61.0 (3.95x) | 0.339% |
| 8  | 33.0 ms / 35.1 ms | 228.0 (6.67x) | 100.7 (6.52x) | 0.559% |

Batch 1->4 is close to free (+0.3 ms min), matching this doc's earlier
"nearly free" batching finding on an unrelated step shape. Batch 8 starts
costing real latency (+6.1 ms min over batch 1) but throughput and achieved
GOPS still scale faster than latency grows -- worth it if the training loop
can actually use larger minibatches.

**Batch 16 and above do not currently compile in practical time.** Offline
`pulsar2 build` time (not step time -- this is the one-time compile cost, run
once per model) grows sharply worse than the runtime cost does: 70 s (batch
1) -> 130 s (batch 4) -> 386 s (batch 8) -> **did not finish within 900 s**
at batch 16, confirmed still compiling and using a full CPU core at 25+
minutes wall-clock when checked directly inside its (by-then-orphaned, see
below) Docker container -- killed rather than let run indefinitely. Batch 32
showed the identical pattern in isolation (no other build running
concurrently) and was killed at the same ~25-minute mark, still short of any
progress-bar stage past "calc input dependencies." Batch 64 was not
meaningfully tested: its build was killed within its first two minutes to
free the host for the batch-32 measurement above, so its short recorded time
is an artifact of that intervention, not a real data point -- don't read
"batch 64: 129.6 s" out of `compile_results.json` as a real number, it isn't
one. Pulsar2's own per-batch reported MACs (`group 0 QuantAxModel macs:`
being exactly `batch x 207,564,800` at 16/32/64, logged before compilation
stalls) confirms these larger graphs and their bigger calibration sets were
correctly built and handed to the compiler; the growth is somewhere in
Pulsar2's own tiling/dependency-graph machinery (`build op serially`, `add
ddr swap`, `calc input dependencies` stage counts grew from 2295/129802/... at
batch 8 to noticeably larger at batch 32), not in anything onnxsim controls.

**One operational note for whoever runs this again**: `pulsar2_docker.build()`'s
`subprocess.run(..., timeout=...)` does not stop the underlying `docker run`
container when it times out -- only the Python-side wait gives up. A timed-out
build keeps consuming a full CPU core and several GB of RAM indefinitely
unless the container is killed separately (`docker ps` / `docker kill`), and
will silently contaminate the *next* build's timing if left running
concurrently with it (this happened once while gathering the numbers above;
the batch-32 build's early timing includes a period of contention with an
orphaned batch-16 container, though its final ~25-minute figure was measured
alone after that container was killed). Worth fixing in `pulsar2_docker.py`
itself -- kill the container on `TimeoutExpired` -- if this sweep is
revisited.

### Execution overlap: async dispatch is unsupported here; concurrent vNPU contexts are real

AXCL's headers (`/usr/include/axcl/axcl_rt_engine.h`) declare
`axclrtEngineExecuteAsync(modelId, contextId, group, io, stream)` alongside
the synchronous `axclrtEngineExecute` `resident_runner.c` uses, plus
`axclrtCreateStream`/`axclrtSynchronizeStream`. **It does not work on this
device/SDK build.** `axclrtCreateStream` succeeds, but every call to
`axclrtEngineExecuteAsync` returns `AXCL_ERR_UNSUPPORT` (`0x4`), confirmed
with a double-buffered variant built specifically to exercise it (overlapping
the next step's `x`/`y` host-to-device copy with the current step's
in-flight execute -- the only per-step host-side work in this benchmark with
no dependency on the current step's output, and so the only thing that could
legally overlap `Execute` without a data race). This is `axclhost` 2.25.0 on
the PCIe-host path (the `axcl-vm` LXD VM this project's hardware work runs
through); async execute may be implemented on a native/on-SoC build this
project has not had access to, but on what's here, it is a documented,
declared, non-functional API, not a missing feature to add around.

**vNPU partitioning is real, and correctness holds.** `axclrtEngineInit`
accepts `AXCL_VNPU_ENABLE` (and `_BIG_LITTLE`/`_LITTLE_BIG`) alongside the
`AXCL_VNPU_DISABLE` `resident_runner.c` uses, splitting the NPU into
concurrently-schedulable partitions. Confirmed non-corrupting first: 20 steps
of the resnet50 step under `AXCL_VNPU_DISABLE` and under `AXCL_VNPU_ENABLE`
produced bit-identical output (loss and the first four floats of
`fc.weight'`, both `0, -0.0212390665, 0.0157326423, 0.0110128485`). Then
measured concurrently -- N separate OS processes, each its own model load,
context and device buffers, `AXCL_VNPU_ENABLE`, same resnet50 step as the
batching table above:

| concurrent contexts | aggregate throughput | vs. N=1 `VNPU_DISABLE` baseline | per-context throughput |
| --- | --- | --- | --- |
| 1 (`VNPU_DISABLE`) | 30.3 steps/s | 1.00x (baseline) | 30.3 |
| 1 (`VNPU_ENABLE`) | 28.2 steps/s | 0.93x | 28.2 |
| 2 | 51.6 steps/s | **1.70x** | ~25.8 each |
| 4 | 79.0 steps/s | **2.61x** | ~19.8 each |
| 8 | 86.5 steps/s | **2.86x** | ~10.8 each |

This is genuine hardware concurrency, not queueing: at N=2 and N=4 each
process's *own* reported per-step latency stays close to the solo
`VNPU_ENABLE` figure (measured from inside that process, independent of what
the other processes are doing) rather than roughly doubling/quadrupling the
way it would if the partitions were only time-slicing one physical resource
end to end. Scaling is real but not free and not unbounded: per-context
throughput falls as concurrency rises (93% of solo at N=2, 70% at N=4, 36% at
N=8), and the aggregate curve is clearly saturating between N=4 and N=8 (+9%
aggregate for double the contexts, against +52% going from N=2 to N=4) --
N=4 is the better efficiency point of what was measured, not N=8. Enabling
`AXCL_VNPU_ENABLE` also costs ~7% off solo throughput versus `VNPU_DISABLE`
even with nothing else running, which is the price of leaving partitioning on
by default rather than only under real concurrent load.

Net: **running several independent training-step contexts concurrently under
vNPU partitioning is a real, orthogonal lever to batching** -- unlike batch
16+ (this doc, above), it does not hit Pulsar2's compile-time wall, since
each context compiles its own small, already-proven graph rather than one
larger one. It trades per-context latency for aggregate throughput similarly
to batching, tops out around 2.6-2.9x in what was measured here, and would
suit a scenario with several independent models/replicas to train rather
than one already-batched step. `scripts/axera/tools/resident_runner.c`
now takes a `-v` flag for `AXCL_VNPU_ENABLE` to reproduce this; there is no
committed orchestration script for launching N of them, a shell loop over N
separate copies of the compiled model (see this section's own measurement
method) is enough.

### Batching and vNPU concurrency compound -- multiplicatively, cleanly

The open question above (do they compose?) is answered: **yes.** Measured N
concurrent `AXCL_VNPU_ENABLE` contexts, each running the resnet18 resident
step at batch B, for every (N, B) combination in {1, 2, 4, 8} x {1, 4, 8} that
doesn't repeat a number already in this doc -- reusing the batch-1/4/8
`.axmodel`s from the batching section above, N separate OS processes each its
own model load/context/buffers, same method as the vNPU section above. Loss
and gradients were not re-verified per point (the batch-dimension math was
already checked host-side in the batching section, and vNPU non-corruption
was already checked bit-for-bit in the section above); what *was* checked
here is that enabling `-v` doesn't change the (batch>1) result at all -- one
batch-4, single-context run under `VNPU_DISABLE` and under `VNPU_ENABLE`
produced identical timing and identical (zero, see caveat below) loss, the
same non-corruption signature the vNPU section's bit-identical check used,
just not repeated for the full 20-step rigor of that section for every point
in this sweep -- a real, if lighter-weight, gap against this doc's usual
standard, noted here rather than glossed over.

Per-sample compute is exact and batch-invariant (207,564,800 MACs = 415.13M
FLOPs/sample, this doc's batching section above), so every point below
converts to aggregate achieved GOPS the same way that section's table does,
from each run's own reported aggregate throughput (`sum` of each concurrent
process's own `throughput=... steps/s` line):

| contexts x batch | aggregate steps/s | aggregate samples/s | aggregate achieved GOPS | vs. 1x1 |
| --- | --- | --- | --- | --- |
| 1 x 1 | 34.0 | 34.0 | 14.1 | 1.00x |
| 8 x 1 (pure vNPU) | 85.9 | 85.9 | 35.7 | 2.53x |
| 1 x 8 (pure batch) | 27.8 | 222.4 | 92.3 | 6.54x |
| 2 x 4 | 53.2 | 212.8 | 88.4 | 6.27x |
| 4 x 4 | 77.0 | 308.0 | 127.9 | 9.06x |
| 8 x 4 | 81.8 | 327.2 | 135.9 | 9.63x |
| 2 x 8 | 44.2 | 353.6 | 146.8 | 10.40x |
| 4 x 8 | 70.9 | 567.2 | 235.6 | 16.70x |
| 8 x 8 | 76.8 | 614.4 | 255.2 | **18.09x** |

The best point measured (8x8, 255.2 GOPS) beats the best pure-batching point
(1x8, 92.3 GOPS) by **2.8x** and the best pure-vNPU point (8x1, 35.7 GOPS) by
**7.1x** -- a real compounding effect, not a wash and not redundant with
either lever alone. 4x8 gets 92% of 8x8's throughput for half the contexts,
the better efficiency point of what was measured (mirroring the vNPU
section's own N=4-vs-N=8 finding at batch 1).

**Why it compounds cleanly**, checked rather than assumed: define
`efficiency(N, B) = aggregate_steps/s(N, B) / (N x solo_steps/s(B))` -- how
much of the naive N-times-linear throughput each combined point actually
gets. This should depend only on N (contention among N concurrent NPU
partitions) and not on B (how much work each partition does per step) if the
two levers are genuinely independent effects rather than interacting:

| N | efficiency at B=1 | efficiency at B=4 | efficiency at B=8 |
| --- | --- | --- | --- |
| 2 | 0.85 | 0.86 | 0.80 |
| 4 | 0.65 | 0.63 | 0.64 |
| 8 | 0.36 | 0.33 | 0.35 |

Each row agrees to within about 4% across batch sizes -- the same
concurrency-efficiency curve the vNPU section measured at batch 1 (0.85 /
0.65 / 0.36 at N=2/4/8) holds regardless of B. So aggregate throughput
factors cleanly as `solo(B) x N x efficiency(N)`: batching improves
per-context efficiency (more useful arithmetic per quantize/transpose-glue
dollar, this doc's batching section), vNPU concurrency multiplies that by
however many partitions minus contention, and neither lever changes how the
other one behaves. The practical upshot: pick the batch size that's
efficient for the *model* (per the batching section's own tradeoffs, not
revisited here) and the context count that's efficient for the *hardware*
(N=4, per both this table and the vNPU section) roughly independently, rather
than needing to jointly search the combination.

**The batch>1 loss=0 caveat above is resolved: it was a calibration-data
bug, not a graph or hardware bug, and not the same mechanism as resnet50's
loss=0 (below).**

Every training-step compile so far (this section's, the batching section's,
resnet50's) grew its own ad-hoc, uncommitted calibration-work-dir generator
in a session scratchpad rather than a committed one. That generator's `y`
one-hot label was placed by scattering a single `1.0` into the **flattened**
`[batch, classes]` tensor (`arr.reshape(-1)[rng.integers(0, arr.size)] =
1.0`) -- indistinguishable from a correct per-row one-hot at batch 1, but at
batch>1 it leaves `(batch-1)/batch` of the rows an all-zero "label" in every
one of the (few) calibration samples. That degenerate calibration data
miscalibrated the `loss` output's quantization range enough to clip a real,
non-degenerate runtime loss down to exactly 0.

Confirmed directly, not just inferred: rebuilding a batch-4 step graph with
an extra debug output tapping `loss_sq` (the per-sample-per-class squared
error, *before* the batch-mean reduction) and running it on the real AX650N
showed every row's per-sample value correctly non-zero and identical across
all 4 rows (mean 0.0460, matching the memset input being bit-identical per
row) -- proving the forward pass and per-sample loss are computed correctly
on-device at batch>1. Only the final scalar reduction read 0. Two workaround
spellings were tried and both still failed the same way (a `ReduceMean` split
into two single-axis passes; a `MatMul` against a constant `1/N` averaging
vector instead of any `ReduceMean` over the batch axis at all) -- ruling out
"a specific `ReduceMean` spelling is broken" and pointing at the calibration
range itself. Inspecting the actual calibration `y.tar` confirmed it: every
sample had exactly one nonzero element total across all 4 rows, not one per
row. Fixing the generator (one one-hot placed per row) and rebuilding the
identical batch-4 graph with nothing else changed made the on-device loss
read a real, consistent, non-zero `0.113021` across every step -- matching
the same order of magnitude as batch 1's `0.117937`.

The fixed generator is now committed as `scripts/axera/make_training_calib.py`
(previously every compile re-derived its own copy from scratch, which is
exactly how this bug shipped unnoticed across three PRs) with a regression
test (`tests/test_make_training_calib.py`) pinning the per-row placement.

**This does not explain resnet50's loss=0** (next section) -- that build's
own calibration generator used a dense `N(0,1)` draw for `y`, not a one-hot,
and is not degenerate the way this one was; its batch is 1 throughout, where
this exact bug is invisible by construction. That caveat remains open; see
its own note below for the current best guess and the concrete next step.

### Device memory: nowhere close to the constraint

Measured with `axcl-smi` and cross-checked against a real API (below), not
estimated from `.axmodel` file sizes:

| state | CMM usage | vs. 7040 MiB total | NPU util |
| --- | --- | --- | --- |
| idle | 12 MiB | 0.2% | 0% |
| resnet18, batch=1, one context | 69 MiB | 1.0% | 61% |
| resnet18, batch=8, one context | 72 MiB | 1.0% | 65% |
| resnet50, one context | 88 MiB | 1.3% | 61% |
| 8 concurrent vNPU contexts, batch=8 each | 432 MiB | 6.1% | **100%** |

Per-process CMM tracks the compiled `.axmodel` size directly (`axcl-smi`'s
own per-PID column: ~6.9 MiB for resnet18's 6.6 MB file, ~19.6 MiB for
resnet50's 20.1 MB file), plus a fixed per-context overhead of roughly
45-57 MiB (I/O buffers, per-context firmware/task state) that does **not**
get shared across concurrent vNPU contexts -- it compounds per context, which
is most of why 8 contexts cost 432 MiB rather than 8 x 7 MiB = 56 MiB.

The headline: even at 8-way vNPU concurrency saturating NPU compute (100%
utilization, confirming the previous section's "saturating" finding), CMM
usage is 6.1% of the card's total. Compute saturates *long* before memory
would become a constraint at any concurrency level measured so far.

**A real, working API for this**, found and verified on hardware rather than
assumed from the header (`axclrtEngineExecuteAsync` was declared but
`AXCL_ERR_UNSUPPORT` on this SDK, so header presence alone proves nothing):
`axclrtEngineGetUsage(modelPath, &sysSize, &cmmSize)` reports the engine's
own required-memory accounting **from the file path alone, before loading**;
`axclrtEngineGetUsageFromMem`/`axclrtEngineGetUsageFromModelId` are the same
query from an in-memory model buffer or an already-loaded `modelId`. Now
wired into `resident_runner.c` (its stderr diagnostics and the parseable
`cmm=...MiB` field on the final summary line), queried once per run right
after load. One discrepancy worth flagging rather than resolving: this API
reports a larger number than `axcl-smi`'s live per-process column for the
same model (15.3 MiB vs. ~6.9 MiB for resnet18) -- read it as the engine's
planned working-set budget, not a live-usage snapshot, and don't expect the
two to match. A lower-level, system-wide family
(`AXCL_SYS_MemQueryStatus`/`AXCL_SYS_MemGetPartitionInfo` in `axcl_sys.h`,
likely what `axcl-smi`'s own aggregate row queries) exists but was **not**
verified here -- flagged as an unconfirmed lead, not a fact, following this
doc's own standard of not claiming a header's presence as working capability.

### Trading free memory for throughput: `Gather` off a resident dataset -- a real Pulsar2 backend gap, not a shape or scale issue

Given how much CMM sits idle even at 8-way vNPU concurrency (above), the
obvious next question: could a training step stop re-uploading a fresh `x`/`y`
batch every step (`resident_runner.c`'s main loop does this even though the
weight *state* is already resident) by keeping a whole dataset resident
on-device instead, and `Gather`-ing each step's minibatch rows from it --
`onnxsim.qat_graph`'s own module docstring documents exactly this trade
("the whole set stays resident and the graph selects rows"), never applied to
this pipeline.

`build_resident_train_step.add_resident_dataset()` does this: bakes a
`[N, ...]` array in as a plain graph initializer and replaces `x`/`y` with
`Gather(dataset, batch_index)`, so per-step host traffic drops to a handful of
`int64` row indices. Host-verified exactly: gathered-minibatch training
produces gradients and loss identical (rtol 1e-5) to feeding the same rows
directly -- correctness is not in question.

**It does not currently compile for real hardware, at any dataset size
tried, and the reason is a genuine Pulsar2 NPU-backend gap, not a shape
choice or a size threshold.** Built the real resnet18 (`onnx::Conv_268/271/
274`, `fc.weight`, 5,361,664 trainable params, the same scope every resnet18
result in this doc uses) both ways: a 136-node baseline (plain `x`/`y`) and a
138-node `Gather` variant, at two dataset sizes -- 4096 rows (207.6 MiB, the
size originally sized against the free-memory headroom) and 256 rows. The
baseline compiled and ran cleanly (confirms nothing else regressed: 26.3 ms
min / 28.7 ms avg per step, matching this doc's established batch-1 numbers).
**Both `Gather` variants failed identically** in Pulsar2's NPU backend
compiler:

```
op: AxGather, attrs = {'dim': 0, 'keepdim': 1, ...}
input = {'x': Tensor(FP32, name=x_dataset, shape=(4096, 1, 2, 64), ...),
         'indices': Tensor(S32, name=batch_index, shape=(1,), ...)}
Exception: (unspecified, NPUBackendError)
```

Pulsar2's own frontend has already reshaped/tiled the `[N, 3, 64, 64]`
dataset into an odd `[N, 1, k, 64]` shape (`k=2` at N=4096, `k=6` at N=256)
before handing the `Gather` to its NPU backend -- the same "the compiler
does its own opaque restructuring before this project's control resumes"
pattern the mcode-quantize probe (above) already found for a different op.
The identical failure at two very different `N` (4096 and 256, a 16x range)
rules out a size/tiling threshold: this is `Gather` over a 4D,
conv-activation-shaped resident tensor specifically, not a "too much data"
problem -- `Gather` is otherwise a normal, supported op elsewhere in this
pipeline (index-based weight/embedding lookups, the conv-as-matmul tap
legalization), so the gap is scoped to this exact usage pattern, not the op
in general.

**Update: the pre-flattened-view workaround this section named as untried
works on real hardware, up to a real, exactly-characterized row-count
ceiling.** `add_resident_dataset(..., flatten=True)` (the new default) now
stores and gathers any rank>=3 array as a flat `[N, prod(shape[1:])]`
initializer, `Reshape`-ing each gathered row back to its real shape
immediately afterward -- an ordinary, separately-supported op, entirely
inside the graph, no change to what data is stored or how the runner binds
buffers. Rebuilt the exact same real resnet18d probe this section's original
investigation used (`Conv_268/271/274` + `fc.weight`, 5,361,664 trainable
params, 64x64 input) with the flattened dataset, at the same two sizes that
failed identically before (4096, 256): **both still fail, but with a
completely different, far more concrete error** -- no longer the opaque
`NPUBackendError` this section originally reported, but a specific,
quantified on-chip-memory (OCM) capacity assertion:

```
job io size > ocm size, AxGather
    67108864 > 3141632          (N=4096)
    op: gather_x
    op_input: {'x': Tensor(FP32, name=x_dataset, shape=(4096, 12288), ...),
               'indices': Tensor(S32, name=batch_index, shape=(1,), ...)}
    job_io_ocm_tensor: [(Tensor(FP32, ..., shape=(4096, 4096), ...), 67108864)]
```

Pulsar2's `AxGather` backend materializes an `[N, 4096]`-shaped on-chip
selection tensor regardless of the gathered row's real width (12288 here) --
a fixed 4096-wide lane, apparently -- and that whole tensor must fit in one
~3.14 MiB OCM budget alongside a few smaller fixed-size companions. Binary-
searching `N` against this real hardware (not just the error message) found
the exact boundary: **N=190 compiles and runs correctly on the card; N=191
fails the same OCM assertion.** N=128 was also confirmed compiling and
running correctly (real, non-zero, stable loss over 15 real steps, 24.1 MiB
CMM, ~33 ms/step) before the boundary search narrowed it further to 190.

This means the flattened-view fix **does** clear the original blocking gap
(any conv-shaped resident dataset, at any size, failed identically before)
and replaces it with a real, size-scoped ceiling: the flattened-dataset
`Gather` pattern works for up to 190 resident rows of this shape (12288
floats/row) -- comfortably enough for many real minibatch-index use cases,
just not the original 4096-row "keep the free 207.6 MiB memory headroom
resident" ambition that motivated this section in the first place. A
dataset wanting more rows than the OCM ceiling allows would need either a
narrower row width (the ceiling trades directly against row width, since
the OCM tensor's size is `N * lane_width`) or splitting the resident dataset
across multiple smaller `Gather`s, both untried here.

**A second, separate latent bug found once compilation got past the OCM
wall**: the ONNX graph declares `batch_index` as `int64` per
`add_resident_dataset`'s own docstring, but Pulsar2's compiled `.axmodel`
silently **downcasts it to `int32`** on-device (confirmed independently by
the compiler's own calibration-time error message reporting `Tensor(S32,
name=batch_index, ...)`, and by `probe_model_io` reporting the compiled
input's size as 4 bytes for a declared-int64, shape-`[1]` tensor -- int64
would be 8). `gather_runner.c` originally wrote `int64_t` indices into that
now-4-byte buffer, half-initializing it with garbage that the NPU then read
as an out-of-range row and faulted `axclrtEngineExecute` with `0x8030070c`
on every attempt -- a clean compile followed by a hard runtime fault, not
the "device stalled from an earlier bad run" pattern this doc's other
sections have seen with that same error code (confirmed via `axcl-smi`
showing a clean, idle device immediately before the fault, and the fault
recurring identically on a fresh process/fresh model load). Fixed by writing
`int32_t` indices instead; `gather_runner.c` also gained a `-rN` flag so its
index generator can be told the real dataset row count (previously
hardcoded to the original 4096, which would have produced out-of-range
indices once the row count moved below that).

**Net**: the resident-dataset mechanism is real, correct, and committed
(`add_resident_dataset()`, host-verified), and now genuinely **usable on
real hardware** for a conv-shaped dataset up to a real, measured 190-row
ceiling for this shape -- the pre-flattened-view fix this section originally
named as untried is confirmed to work, not merely plausible. `scripts/axera/
tools/gather_runner.c` (this section's own runner) also flagged a real,
separate latent bug while building this: **neither `resident_runner.c`
nor `whisper_resident_runner.c` actually feeds `grad_seed`** (added as a
real graph input by the FP32-gradient-seed work above) -- both allocate its
buffer but never write to it, leaving it as whatever device memory happened
to contain. **Audited below ("Audited: the 'unfed grad_seed' scare"
section): a real gap, but zero actual impact** -- every model either
runner has ever actually been run against predates `grad_seed`'s promotion
to a graph input, so nothing this project has reported was affected. Both
runners were fixed regardless, for any future rebuild.

## Two vendor bugs, both silent

**`ReduceMean` with no `axes` reduces only the last axis.** ONNX reduces all of
them. Confirmed on the card: a `(1,16,32)` input returned 16 values where
onnxruntime returned 1, with no warning and ignoring the declared output shape.
Any model with a bare `ReduceMean` -- mean pooling, layer-norm statistics, most
loss reductions -- gets a wrong answer on this hardware. Every rule in
`legalize.py` names its axes explicitly.

**A constant bias lets the backend rebuild a `Gemm` we removed.**
`MatMul(live) + Add(constant 1-D)` is what `fuse_matmul_add_bias_into_gemm`
matches. Skipping that pass in onnxsim (as onnxsim#1332 does upstream) is
necessary but **not sufficient**: Pulsar2 runs its own optimizer and fuses it
back, into a `Gemm` whose weight is live -- the node it cannot lower. It does
not say so. The build dies inside PPQ's calibrator with
`ValueError('The truth value of an array with more than one element is
ambiguous')`, naming no node and nothing about `Gemm`.

Bisecting 58 nodes to 4 by recompiling ranges found it; then four spellings of
the same arithmetic separated cause from coincidence:

| bias spelling | result |
| --- | --- |
| 1-D initializer | FAILED |
| `Constant` node instead of initializer | FAILED |
| operand order swapped | FAILED |
| **shaped `[1, N]`** | **BUILT** |
| `Identity` between `MatMul` and `Add` | BUILT |

`_unfusable_bias` takes the reshape. The `Identity` also works and is the wrong
fix -- it is exactly what a later dead-code pass would delete, putting the bug
back.

**The general lesson:** a graph that needs an op *gone* must be shaped so
nothing can put it back. Suppressing your own optimizer is half the job.

## The rules

`legalize.TRAINING_RULES`, in an order that matters --
`inline_local_functions` first, because every later rule inspects op types and
would look straight past a function call.

| rule | the failure it answers |
| --- | --- |
| `inline_local_functions` | `KeyError('dont support GradAdd opr')` -- a local `FunctionProto`, not an op type; one per residual connection |
| `act_weight_conv_to_matmul` | `AxQuantizedActWeightConv, shapefn failed` -- the weight stays FP32 while the activation is U8 |
| `gemm_to_matmul` | `NotImplementedError('Should fuse Gemm (two non-parameter inputs) to MatMul.')` |
| `rank0_to_rank1` | `RuntimeError: zero-dimensional tensor ... cannot be concatenated` |
| `neg_to_mul` | `Neg` is the one backward-pass op off the AX650 list |
| `avgpool_ceil_to_floor` | `graph_grad` declines `ceil_mode=1`; cleared only where provably a no-op |
| `flatten_to_reshape`, `global_pool_to_reduce` | no gradient rule, and none needed |

`act_weight_conv_to_matmul` is the substantial one. With the activation
transposed to `[N, spatial..., Cin]`, each tap is one `MatMul` against
`w[..., k]` reshaped to `[Cin, Cout]`; stride becomes the tap slice's `step`.
1-D and 2-D, any stride, with or without bias; dilation and groups are declined
rather than approximated. `fuse=True` concatenates the taps into **one** matmul
`K**2` times deeper -- exact, since the taps share an output and differ only
along the reduction axis -- which is what the matrix unit rewards.

## Simplify the gradient graph

Nobody was, and it is worth a great deal:

| graph | before | after |
| --- | --- | --- |
| resnet18 training step | 731 | **210** (-71%) |
| earlier variant | 759 | 236 (-69%) |

`graph_grad` spells the chain rule out literally and the tap rewrite multiplies
it, so consecutive taps slicing the same tensors collapse under
common-subexpression elimination -- simplification is worth *more* after
legalization than before. onnxsim#1332 now does this inside
`qat_graph.make_step_graph()`; the Axera path assembles its `ModelProto` by
hand and so calls `simplify()` itself, with the same skip list.

## A different architecture: resnet50, first compile

Everything above was resnet18d. The pipeline (`build_resident_train_step.py`,
`legalize.TRAINING_RULES`, `_linearize_trainable_convs`) was written against
that one shape; nothing in it names resnet18 specifically, but nothing had
tried a second architecture either. resnet50d's `layer4` is bottleneck blocks
(1x1 -> 3x3 -> 1x1, 2048-channel output) rather than resnet18's basic blocks
(3x3 -> 3x3, 512-channel output) -- a real structural difference, not just
"more of the same shape."

**Scope chosen:** resnet50d, 64x64 input (same as resnet18d, deliberately --
first compile of a new architecture is not the moment to also fight a bigger
input), only `layer4.2` (the last bottleneck block: `conv1` 1x1
`[512,2048,1,1]`, `conv2` 3x3 `[512,512,3,3]`, `conv3` 1x1 `[2048,512,1,1]`)
plus `fc.weight` `[1000,2048]` trainable -- **not** the whole final stage
(all three bottleneck blocks, ~3x the matmul-tap count). Given PR #1342 found
Pulsar2's own compile time blowing up non-linearly well short of any runtime
limit (70s -> 130s -> 386s -> >25 min with no result, batch 8 -> 16 -> 32 on
the much smaller resnet18 graph), starting with a trainable tail sized to
match resnet18's own (3 convs + `fc`, 6,504,448 trainable params against
resnet18's 5,361,664 -- comparable scale, genuinely different block shape)
was the deliberate choice over reaching for all of `layer4` on the first try.

One export wrinkle worth recording: `timm.create_model('resnet50d', ...)`
defaults to `zero_init_last=True` (the last BN in each residual block starts
at gamma=0, a standard residual-net init trick). After BN-folding this makes
every block's `conv3` weight fold to an **all-zero** tensor -- correct, not a
bug, but onnxsim's own CSE then correctly notices all those all-zero tensors
are identical and merges `layer4.0.conv3`, `layer4.1.conv3` and
`layer4.2.conv3` into **one shared initializer**, which would have made
`layer4.2`'s trainable weight secretly alias two other blocks' forward convs.
`zero_init_last=False` avoids the degenerate collision; worth checking for on
any future model export, since it recurs by construction wherever
zero-init-residual is the default.

**Built and verified on host:** 201 nodes (resnet18's comparable graph was
210 -- genuinely similar scale despite resnet50 being a much deeper network
overall, because the frozen backbone stays native `Conv` regardless of depth
and only the trainable tail's node count depends on this pipeline). Central-
difference check against the in-graph analytic gradient across all 4
trainable tensors (24 sampled elements spanning both 1x1 convs, the 3x3, and
`fc.weight`): **cosine similarity 0.99994**, confirming
`_linearize_trainable_convs`'s im2col-as-gather identity generalizes to the
bottleneck-block shapes with no resnet18-specific assumption breaking.

**Compiled cleanly:** `pulsar2 build`, 57.8s (faster than resnet18's original
97s, despite the deeper backbone -- number of *trainable* matmul taps
dominates compile time more than total graph depth), one fused NPU subgraph,
20.1 MB `.axmodel` (larger than resnet18's 6.6 MB, from the bigger frozen
backbone's weights).

**Ran on real hardware:** `resident_runner` (unmodified -- the I/O count (7
in, 5 out) and positional state-pairing convention happened to match
resnet18's exactly, since both trainable tails have 4 weights) gave
**29.8 ms min / 32.2 ms avg per step**, essentially the same as resnet18's
28.6 ms despite the much deeper frozen forward pass -- consistent with the
quantize/dequantize tax being the dominant cost regardless of model depth
(see "The quantize redundancy is real" above), not something resnet50's
extra layers meaningfully add to.

**Still not verified, and now a narrower, still-open question:** the
reported loss read exactly `0` every step. `resident_runner`'s built-in test
batch is a fixed `memset(hx, 0x11, ...)`/`memset(hy, 0x22, ...)` byte
pattern (not real image data), ~1e-28 as float32 -- effectively zero at both
ends of the loss computation.

This is at batch 1, so it is **not** the batch-axis calibration bug the
batching section above found and fixed (that bug is invisible by
construction at batch 1 -- a single scattered one-hot in a one-row tensor is
already a correct per-row one-hot). It also is not obviously the same root
cause by inspection: resnet50's own calibration generator (a separate,
equally ad-hoc scratchpad copy) draws `y` from `N(0, 1)` rather than a
one-hot, and its actual calibration data checked out non-degenerate (dense,
reasonable min/max/mean across every input tensor, no all-zero rows). So
this remains the original, weaker hypothesis -- "the quantizer is correctly
rounding a near-zero range to zero," plausible precisely *because* a
`N(0,1)`-calibrated loss range is wide relative to the near-zero value a
`memset` input actually produces, unlike the one-hot case above where the
calibrated range was itself the bug -- but **unconfirmed**, the same
"checked coarsely, not the full gradient table" caveat PR #1335's original
resnet18 run carried.

**Settled by PR #1382 (resnet50 batch scaling): benign, as originally
hypothesized.** Real, non-`memset` image data (the `<model>.x0`/`.y0` host
files `resnet50_realdata_runner.c` reads) trains with a real, monotonically
decreasing loss at every batch size tested -- the `loss=0` reading was
specific to `resident_runner`'s own fixed `memset` test pattern rounding to
zero under real calibration, exactly the same conclusion resnet18's own
memset runs already supported. No debug-tap rebuild was needed once real
data settled it directly.

**Recommendation for next time:** the pipeline needs no resnet50-specific
changes -- the natural next step is either the remaining two bottleneck
blocks of `layer4` (watch compile time; extrapolate from this block's 57.8s
before jumping straight to all three) or the debug-tap check above to settle
resnet50's own loss=0 question, not further architecture generalization
work.

~~Both done.~~ The loss=0 question is settled below ("resnet50 batch
scaling" section); the remaining `layer4` blocks are done further below
("All three `layer4` bottleneck blocks" section) -- jumping straight to all
three worked on the first attempt, no incremental step needed after all.

### resnet50 batch scaling: real, and it settles the loss=0 caveat too

Every batch-scaling measurement so far in this doc (the "Batching" and
"Batching and vNPU concurrency compound" sections above) was resnet18 only;
resnet50 had never had `batch>1` tested at all. Closed that gap directly,
reusing `build_resident_train_step.set_batch` on the same `layer4.2` +
`fc.weight` scope this section built, real host-trajectory calibration (this
doc's own established fix for the lr/grad_seed degenerate-calibration bug
class, PRs #1370/#1373 -- applied from the start here, not bolted on after a
third repeat of that bug), and a `resident_runner.c` variant that reads real
`x0`/`y0`/state files from disk (rather than resident_runner's own fixed
`memset` test pattern) so the loss curve is unambiguous.

**Three real export/pipeline gaps hit and fixed getting there, none specific
to batch scaling itself:**

1. The legacy TorchScript exporter (`dynamo=False`) that this doc's original
   resnet50 compile presumably used loses real per-layer names for
   `resnet50d`'s non-stem tensors the same way `build_whisper_train_step`'s
   docstring already documents for Whisper's encoder layers (`layer4.2.
   conv1.weight` etc. all come out as generic `onnx::Conv_NNN`) -- confirmed
   directly here, not assumed from the Whisper case. `dynamo=True`
   (torch.export-based) preserves real names throughout, at the cost of
   landing on **opset 18** regardless of the requested `opset_version` (the
   onnxscript version-converter's fallback to the ONNX C API fails on this
   graph and silently leaves it at 18).
2. `ReduceMean`'s `axes` moved from attribute to input at opset 18.
   `legalize._set_axes` already exists for exactly this (its own docstring:
   "resnet18d is opset 18, the training graphs built by hand are opset 17"),
   but `onnxsim.qat_graph.make_step_graph` always declares its *own* output
   model at a fixed opset 17 while copying node protos through verbatim --
   so an axes-as-input `ReduceMean` surviving from an opset-18 forward
   export into the step graph fails the opset-17 schema (`Node with schema
   ReduceMean:13 has input size 2 not in range [min=1, max=1]`). Fixed by
   downgrading every such node back to the attribute form (and the model's
   declared opset to 17) right after export, before anything else touches
   the graph -- a real, if narrow, pre-existing gap in the opset-18-forward
   -> opset-17-step-graph pipeline that this project's resnet18/Whisper work
   never happened to trigger.
3. `timm`'s `SelectAdaptivePool2d.flatten` traces to `Reshape(mean, [1,
   2048])` -- the *batch-1* value baked in as a literal constant by the
   exporter's trace, since `set_batch` (by its own docstring) only rewrites
   the declared *input* shape and clears stale `value_info`, not every
   batch-shaped constant an exporter baked in downstream. At batch>1 this
   reshapes a `[batch, 2048, 1, 1]` tensor into `[1, 2048]`, which
   onnxruntime correctly refuses on element-count mismatch. Fixed by
   rewriting that one Reshape's target to `[-1, 2048]` (the standard ONNX
   "infer this dim" sentinel) before `set_batch` runs -- the same *class* of
   fix `build_w2v2_feature_extractor_step.py` needed for wav2vec2's own
   batch-dependent flatten shape (PR #1372), on a different architecture.

None of these three are specific to batch scaling -- (1) and (2) would have
hit the very first resnet50 compile too, had that session's export happened
to use `dynamo=True`; (3) only bites at `batch>1`, and is the one genuinely
new finding of that kind.

**Host-verified before touching hardware**: central-difference check against
the in-graph gradient at batch 1 (same methodology as this section's
original 0.99994 result) -- **0.99987 directional cosine similarity** across
the 4 trainable tensors, confirming the pipeline change didn't alter the
gradient math.

**Real batch sweep, real AX650N** (per-sample compute exactly batch-
invariant -- Pulsar2's own `group 0 QuantAxModel macs:` line reads
390,889,472 / 1,563,557,888 / 3,127,115,776 at batch 1/4/8, exactly 4x/8x
batch 1's figure, matching resnet18's own batch-invariance finding):

| batch | compile time | step time (min / avg) | samples/s | achieved GOPS (min) | rated-TOPS utilization |
| --- | --- | --- | --- | --- | --- |
| 1 | 57.4s | 29.258 ms / 30.252 ms | 34.2 | 26.7 | 0.148% |
| 4 | 117.4s | 31.081 ms / 31.748 ms | 128.7 (3.77x) | 100.6 (3.77x) | 0.559% |
| 8 | 339.6s | 36.655 ms / 38.788 ms | 218.3 (6.39x) | 170.6 (6.39x) | 0.948% |

Batch 1 is close to resnet18's own 29.8ms first-compile number (this
section, above) as expected -- the quantize/dequantize tax dominating step
time regardless of depth, this doc's earlier finding, applies here again.
The batch-scaling shape itself closely mirrors resnet18's own table (3.85x/
6.67x samples/s at batch 4/8): **resnet50's bottleneck-block architecture
scales with batch size the same way resnet18's basic-block architecture
does** -- not a foregone conclusion (a real structural difference was the
whole reason this doc's earlier section called out resnet50 as a distinct
architecture to test in the first place), but confirmed rather than assumed.
Compile time again grows faster than linear (57s -> 117s -> 340s, a 5.9x
cost for 8x the batch, close to resnet18's own 70s -> 130s -> 386s at the
same batch sizes) -- consistent with the established, generic Pulsar2
compile-time-wall finding, not something specific to this architecture.
Batch 16+ was not attempted, per the generic wall already established on
resnet18.

**This also settles the "still not verified" loss=0 caveat** this section's
first-compile subsection left open. Every run above used real host-
trajectory data (real `x`/`y`/weight values crossing the host boundary, not
resident_runner.c's `memset(0x11)`/`memset(0x22)` pattern), and every batch
size read a real, monotonically decreasing loss across 30 real steps (e.g.
batch 1: `501.66 -> 492.4 -> ... -> 441.55`, no zeros, no NaNs, no frozen
steps beyond ordinary INT8 quantization-step plateaus). This confirms the
original *weaker* hypothesis was the right one: the reported `loss=0` was
the quantizer correctly rounding a `memset`-near-zero runtime input down to
zero, not a calibration or graph bug -- the same conclusion reached for
resnet18's own memset-input runs, just not previously confirmed for
resnet50 specifically.

**Bonus check: vNPU concurrency still composes with batch>1 on this
architecture too.** The "Execution overlap" section above already measured
resnet50's own vNPU concurrency scaling at batch=1 (2.86x aggregate at N=8);
the open question was whether that composes with `batch>1` the clean way
the "Batching and vNPU concurrency compound" section found for resnet18.
One data point, not resnet18's full sweep: 4 concurrent
`AXCL_VNPU_ENABLE` contexts, each at batch=8 (same method as that section --
N separate OS processes, each its own model-file copy), against a
batch=8 `AXCL_VNPU_ENABLE` solo baseline of 24.9 steps/s (vs. 25.8
`VNPU_DISABLE`, the same ~3-4% partitioning tax found before):

| point | aggregate steps/s | aggregate GOPS | vs. 1x1 baseline (26.7 GOPS) |
| --- | --- | --- | --- |
| 1x1 (batch=1, disable) | 34.2 | 26.7 | 1.00x |
| 4x1 (pure vNPU, batch=1, from "Execution overlap" above) | 79.0 | -- | -- |
| 1x8 (pure batch) | 218.3 | 170.6 | 6.39x |
| 4x8 | 71.0 (17.8+17.5+17.8+17.9) | 444.1 | **16.6x** |

`efficiency(N=4) = 71.0 / (4 x 24.9) = 0.71` -- close to, if a bit higher
than, resnet18's own N=4 efficiency figures (0.65 at batch=1, 0.63-0.64 at
batch 4/8, from the compound-scaling section's table), and the 16.6x
combined figure lands almost exactly on resnet18's own 4x8 point (16.70x)
despite the different block architecture. **Composition is real here too**,
not just on resnet18 -- consistent with the compound-scaling section's own
finding that the two levers are independent effects (vNPU contention depends
on N, not on the model or its batch size), now confirmed on a second
architecture rather than assumed to generalize.

### All three `layer4` bottleneck blocks: the "remaining two blocks" recommendation, done directly

The "resnet50, first compile" section above left an explicit next step
unfinished: "the remaining two bottleneck blocks of `layer4` (watch compile
time; extrapolate from this block's 57.8s before jumping straight to all
three)". Rather than the cautious incremental route that sentence
recommends, went straight to all three blocks (`layer4.0` + `layer4.1` +
`layer4.2`'s main-path convs, 9 convs total, plus `fc.weight` -- 10
trainable tensors, ~2.5x the original single-block scope's tap count; each
block's shortcut/`downsample` conv stays frozen, matching `layer4.2`'s own
original main-path-only scope choice) and it worked on the first attempt --
the "watch compile time" caution turned out to be conservative here, not a
real risk at this particular scope.

**Built and verified on host:** 327 nodes (up from 201 for `layer4.2` alone,
265 for a `layer4.1`+`layer4.2` two-block scope also built along the way --
node count grows roughly linearly with trainable tap count across all three
points, not the non-linear blowup the batch-size axis shows). Central-
difference check on two of the ten trainable tensors (`layer4.0.conv1` and
`layer4.0.conv2`, chosen as the two furthest from the loss and therefore the
most exposed to any gradient-accumulation mistake across the wider scope):
0.99992 and 0.99895 cosine similarity, confirming `_linearize_trainable_convs`
still generalizes correctly at this scope.

**Compiled cleanly:** `pulsar2 build`, batch 1, **184.6s** -- 3.2x the
single-block scope's 57.8s for 2.5x the trainable tap count, comfortably
short of the compile-time wall the batching sections above establish (which
only bites well past this, e.g. resnet18 batch 32 at >25 minutes). One fused
NPU subgraph, 11.7 MB `.axmodel`.

**A new, real calibration-stability finding, not previously hit by any
single-tensor scope:** a naive `lr` sweep across several orders of magnitude
(1e-6 to 1e-2, matching the dynamic range `build_w2v2fe_batch_calib.py`'s own
`lr=100`-based recipe used successfully for a *single* trainable tensor) blew
up to `loss=9e8` at `lr=1e-2` and produced `NaN` weights by `lr>=1` when
applied to a **compounding** real host trajectory across all ten tensors at
once -- ten tensors' worth of gradient magnitude flowing through the same
`lr` multiplier destabilizes far sooner than any single-tensor case in this
project has needed to consider. Fixed by holding `lr` constant at `1e-4`
(confirmed stable, real loss decrease step-over-step) for the whole real
host trajectory used to seed calibration, and leaving `make_training_calib`'s
own built-in default (a non-degenerate +/-0.1% jitter around `1e-4`) to
provide the calibration range's actual variety rather than a real_data
override -- the lesson generalizes: **a real_data host trajectory's own
stability (not just its calibration non-degeneracy) becomes a real
constraint once enough trainable tensors compound in a single SGD step**,
something no earlier single- or four-tensor scope in this doc had reason to
find.

**Ran on real hardware:** a new `n_state`-parametric runner
(`scripts/axera/tools/resnet_layer4_runner.c`, generalizing
`resnet50_realdata_runner.c`'s hardcoded `N_STATE=4` to any trainable-tensor
count via a runtime argument, same I/O layout convention otherwise) gave a
real, monotonically non-increasing loss curve at batch 1, real host-trajectory
calibration, `lr=1e-4`: `335.2 -> 167.6 -> 125.7 -> 125.7 -> 125.7 -> 125.7 ->
83.8 -> 83.8 -> ...` (20 real steps, no zeros, no NaNs) -- the flat runs
between drops are the same ordinary INT8 quantization-step plateau this doc's
other resnet50/resnet18 sections already document, not a new finding. **76.5
ms avg / 72.0 ms min per step**, roughly 2.5x `layer4.2`-alone's 29.8 ms,
tracking the ~2.5x increase in trainable weight moved per step rather than
the quantize/dequantize-tax-dominates-regardless-of-depth finding those
earlier sections make (that finding was about *frozen* backbone depth, not
trainable-tensor count, and does not contradict this). **32.5 MiB CMM** --
still a small fraction of the card's 7040 MiB budget, consistent with every
earlier case in this doc.

**Net finding: scaling the trainable scope of a real architecture from one
bottleneck block to all three "just worked" on the first real attempt**, with
the only genuine new gap being the calibration-stability finding above (now
documented for the next model that trains many tensors from one compounding
host trajectory) -- not a pipeline change, a compile-time wall, or a backend
bug. The two-block `layer4_1_2` intermediate scope
(`scripts/axera/build_resnet50_layer4_step.py`'s other `SCOPES` entry) was
built and host-verified but not compiled/run on hardware, since going
straight to all three succeeded and made the intermediate case moot for this
pass.

## A memory-heavy case: Whisper-base encoder training

Every case above -- resnet18d and resnet50d, both at 64x64 input with a
handful of trainable convs -- was chosen small enough to stay well clear of
Pulsar2's own compile-time wall (batching section above), and consequently
never put real pressure on device memory either (the "Device memory"
section's worst case, 8-way vNPU concurrency, was still only 6.1% of the
card's 7040 MiB CMM). This section is the opposite choice, deliberately: a
real-size Whisper-base encoder, trained deep and at its real sequence length,
specifically to have a genuine "memory matters here" case on hand -- the
motivation being a parallel investigation into recomputation/checkpointing
and graph-scheduling techniques for cutting training memory, which needs an
example where memory is actually the constraint to have anything to bite on.

**Model:** `transformers.WhisperModel` with `openai/whisper-base`'s real
config (`d_model=512`, `encoder_layers=6`, `encoder_attention_heads=8`,
`encoder_ffn_dim=2048`), encoder only, at its real, unmodified input size --
3000 mel frames, `max_source_positions=1500` -- not a shrunk `max_source_
positions` the way the audio-speech op-coverage survey used to keep its
export small (`docs/axera-audio-speech-op-coverage.md`'s reproduction
snippet uses `max_source_positions=32`). 20,590,592 encoder parameters
total, exported via `torch.onnx.export(..., opset_version=17, dynamo=False)`
-- 340 nodes, confirming the survey's op-coverage finding at real scale, not
just its toy one: `Add`, `Constant`, `Conv` (the frozen stem only), `Div`,
`Erf`, `Identity`, `LayerNormalization`, `MatMul`, `Mul`, `Reshape`,
`Softmax`, `Transpose` -- Erf-GELU, not the fused `Gelu` op, exactly as the
survey found, so the raw-`Gelu`-has-no-backward-rule gap it flagged does not
apply here either.

**Two scopes trained, both the whole width of every chosen layer** (unlike
resnet18/50's single trainable conv/block) -- picked by taking a suffix of
the model's own float32 initializers in first-use (= layer) order, not a
name-based layer selector (the exporter only keeps meaningful names for
`layers.0`'s own tensors; every other layer's weights get generic
`onnx::MatMul_NNN` names, so "last K layers" was carved out positionally):

| scope | tensors | trainable params | step-graph nodes |
| --- | --- | --- | --- |
| `last_half` (roughly the last 3 of 6 layers) | 14 | 11,534,336 | 521 |
| `full_encoder` (everything but the frozen conv stem) | 28 | 20,431,360 | 905 |

Building either needed one real fix to `build_resident_step`'s own
pipeline-order assumption, not specific to Whisper: `graph_grad.
build_backward` demands a gradient rule for **every** node type it walks,
`Constant` included, even though a zero-input op has nothing to backprop
through. Whisper's Erf-GELU decomposition leaves several per-layer `Constant`
nodes (the `0.5`/`sqrt(2)` literals) that a plain forward export never folds
away, and resnet18/50 never exercised this path because their forward graphs
happen not to have any. Fix: run `onnxsim.simplify()` (same skip list as
the pipeline's own final pass, `fuse_matmul_add_bias_into_gemm`/`fuse_
transpose_into_gemm` skipped, so as not to reshuffle which initializer name
carries which weight before `params` is chosen) *before* `build_backward`,
then fold whatever `Constant` nodes CSE didn't fully eliminate by hand (a
`Constant` node is an initializer wearing a node's clothes -- same
`TensorProto`, no inputs). One more real trap the same fold step walked
into: the fused-QKV projection's `Split` node takes its per-output sizes as
a second, `int64` **input**, and after CSE merges the six layers' identical
size-list constants into one shared initializer, a naive "any float-typed
node input" trainable-candidate scan would be fine (`int64` is excluded by
construction) -- but a differently-written scan that also picks up
newly-hand-folded scalars (the GELU/attention-scale literals, which *are*
float32) needs an explicit rank check (`len(dims) >= 1`) to exclude them: a
weight always has rank >= 1, a decomposition constant never does.

**Correctness, verified on host, both scopes.** Per-element finite
differences turned out to be the wrong tool at this scale: a 900-node graph
reducing over 1500 x 512-element tensors accumulates enough fp32 rounding
noise that a single perturbed weight element's loss delta is swamped by
noise at any reasonable epsilon (confirmed directly -- the "finite-difference
gradient" for 5 of 6 first-tried elements was itself just fp32 ULP noise on
the loss, an exact multiple of ~1.19e-7). The standard fix for graphs this
size is a **directional** check instead of a per-element one: take the full
gradient tensor the graph itself computed for one trainable weight
(`g = w - w_next`, since `w_next = w - lr*grad`), step `t` units along `-g`,
and confirm the loss falls by the amount local curvature predicts. Checked
at `t` in `{1e4, 1e6, 1e8}` against 6 (`last_half`) and 8 (`full_encoder`)
sampled tensors: **monotonic loss decrease at every step size below the
point where a nonlinear network's local linear approximation should be
expected to break down**, with actual-vs-predicted first-order drop ratios
of 0.3-0.9 -- the right sign, the right order of magnitude, and the right
qualitative shape (undershooting the linear prediction as curvature bends
the descent, exactly what a locally-convex loss surface does), which is the
standard evidence bar a directional gradient check is held to.

**Real device memory, `last_half`:** compiled cleanly via `pulsar2_docker.
build()` in 454.7s (7.6 min -- well inside the batching section's demonstrated
wall, this is a bigger graph than any batch-scaling point that blew up past
25 minutes there, but node *count* and matmul *shape* drive Pulsar2's compile
time more than raw depth, the same finding the resnet50 section made), one
fused NPU subgraph, 101,154,816,000 MACs per Pulsar2's own reported count,
17.8 MB `.axmodel`. Queried with the real, verified `axclrtEngineGetUsage()`
API (`docs/`'s own "Device memory" section above) straight from the compiled
file, no device execution needed: **142.4 MiB CMM** -- 5.5-9.3x every case
measured before it (resnet18 15.3 MiB, resnet50 25.7 MiB), and this is only
the *half*-encoder scope.

**`full_encoder` does not compile, and neither does the obvious fix.**
Training every non-stem tensor promotes the LayerNorm affine (scale) that
every one of the 13 `LayerNormalization` nodes shares -- PyTorch's default
init (`weight=1`, `bias=0`) makes all 13 layers' affine params bit-identical,
so CSE had already merged them into one shared initializer before any of
this pipeline's own code ran. Promoting that one shared tensor to state
therefore makes *every* `LayerNormalization` in the graph live at once, and
Pulsar2's frontend hard-errors on it: `KeyError('layers.0.self_attn_layer_
norm.weight')`, thrown from its own native-parser's reference-attribute
lookup, not a graceful "unsupported" diagnostic -- a live-weight
`LayerNormalization` is a real, unfixed gap in this pipeline's legalization
coverage (`legalize.TRAINING_RULES` has no rule that does for `LayerNorm`
what `_linearize_trainable_convs`/`act_weight_conv_to_matmul` do for `Conv`
with a live weight). Freezing just that one shared tensor
(`full_encoder_no_ln`, 27 of 28 tensors, 20,430,848 of 20,431,360
params -- 99.998% of the same trainable weight, host-verified the same way,
directional checks passing at the same ratios) gets past the frontend, but
hits a **second, different** wall at the NPU backend's tiling stage:
`TileFailException("AxQuantizedAdd, tuple index out of range")` on a
`(2048,)`-shaped `Add` deep in the FFN's own SGD-update arithmetic --
internal to Pulsar2's closed-source scheduler, not diagnosable from the ONNX
side the way the frontend `KeyError` was, and not chased further here (this
is where the resnet18 quantize/dequant work stopped too, at a comparable
"the compiler's own internals, not ours" wall).

**Where this leaves the recomputation/scheduling investigation:** one fully
working, host-verified, **real-hardware-measured** case (`last_half`,
142.4 MiB CMM, 5.5-9.3x every prior case) plus two well-diagnosed compile
walls mapping out where the *bigger* scopes actually stop, rather than a
vague "probably doesn't scale." Both walls are legalization/compiler gaps a
future pass could plausibly close (a live-weight-`LayerNormalization` rule
for the first; the second needs a Pulsar2-side bug report or a workaround
that avoids whatever shape/dtype combination trips its tiler), not
fundamental limits -- so `last_half`'s 142.4 MiB is a floor for how memory-
heavy a real Whisper training case can get on this pipeline today, not a
ceiling. The pipeline needed no Whisper-specific change to get this far
beyond the `Constant`-folding fix above (now upstreamed into
`build_resident_step` itself, not left as a one-off script) -- a real
confirmation that `build_resident_train_step.py` generalizes past CNNs to
attention architectures with no new legalization rules for anything that
*did* compile, matching resnet50's own "no code changes needed" finding for
a second architecture in a row.

### `last_half` actually trains -- a real 30-step run, and the gradient dies immediately

Rebuilt `last_half` clean on current master (11,534,336 params, 521 nodes --
identical to the numbers above) and ran it for real: 30 resident steps on the
AX650N via a Whisper-shaped variant of `resident_runner.c`
(`whisper_resident_runner.c`, new -- 14 trainable-weight state tensors
instead of resnet18's 4, same positional convention, same device-to-device
residency). `axclrtEngineGetUsage()` on the fresh build: **142.719 MiB CMM**,
matching the earlier compile-only measurement (142.4 MiB) to within noise.
**244.2 ms min / 246.8 ms avg per step, 4.1 steps/s** -- roughly 8.5x
resnet18's 28.6ms and 8x resnet50's ~30ms, in line with a trainable slice an
order of magnitude bigger processing a real 1500-token sequence rather than
a 64x64 image.

One calibration decision worth recording: `make_training_calib.py`'s default
treats every name in `label_inputs` (`("y",)`) as a **2-D one-hot**
classification target (`arr[row, rng.integers(0, classes)] = 1.0`). Whisper's
`y` is a dense `[1, 1500, 512]` MSE regression target, not a classification
label -- applying the one-hot logic to it would misinterpret `dims[1]` (1500)
as a class count and stamp one axis-1 position's entire 512-vector to 1.0,
reproducing the exact *class* of degenerate-calibration bug this tool exists
to prevent, just in a shape it wasn't written for. Built with
`label_inputs=()` instead, so `y` gets the same plausible-scale random draw
every other non-`x`/`lr` input gets.

**The loss reads bit-identical (`1.00265`) at every one of the 30 steps --
investigated rather than assumed benign, given this exact symptom has been a
real bug twice before in this thread (the batch>1 calibration bug, PR #1346;
resnet50's still-open loss=0 case).** Instrumented the runner to read a
state tensor's raw bytes immediately before and after `Execute()` on the
first 3 steps. Result: **step 0 genuinely updates the weight**
(`[0.0172792, 0.0410809, 0.0165219, -0.0651579]` ->
`[0.0178826, 0.0417261, 0.0158957, -0.0655696]`, confirmed via
device-to-host memcpy of the raw buffer, not the runner's own reporting
path) -- **steps 1 through 29 are exactly byte-identical, before and
after.** The residency plumbing is correct; the gradient itself rounds to
exactly zero after one step. The loss's own apparent constancy is then
consistent, not a separate bug: it reflects each step's *input* weights, a
~1% shift in early-layer weights doesn't move a quantized MSE loss across a
level boundary, and once the gradient dies at step 1 there is nothing left
to move it at all.

**This is the same mechanism `docs/axera-quantizer-reverse-engineering.md`
and PR #1355 characterized in general, now confirmed on a real
architecture** -- and it collapses far faster here than on any CNN case in
this document (resnet18: ~1,000-5,000 steps; Whisper `last_half`: 1 step).
The most likely reason, not confirmed by a second run: `y` and the 14
trainable tensors were calibrated with a generic `weight_scale=0.05` random
draw with no attempt to match Whisper's actual gradient magnitude through a
much deeper backward pass and a real 1500-token attention mechanism, exactly
the mismatch PR #1355 showed causes immediate death (its own probe read
*zero* nonzero gradient elements when a model calibrated for one magnitude
regime was fed a gradient from a different one). The calibration-scale-
matched multi-phase technique from PR #1356 is a plausible, untried fix for
this exact case -- not attempted here, out of this task's scope, but a
direct, concrete next step rather than a vague "investigate quantization
further."

### Recalibrating Whisper for its real gradient scale: two attempts, one real finding, one working (if under-demonstrated) mechanism

Following up on the previous section's own suggested next step ("the
calibration-scale-matched multi-phase technique from PR #1356 is a
plausible, untried fix for this exact case"). Two things were built and run
for real; neither gave the clean "survives many steps" result hoped for, and
the reason turned out to be more fundamental than a calibration-choice
problem.

**Measured Whisper's real gradient trajectory first, on host, in float --
no guessing.** Ran `last_half`'s own step graph through `onnxruntime` (no
quantization) for 40 real SGD steps at `lr=1e-2` against a **fixed** batch
(real convergent training, not fresh-random-batch noise each step): the loss
moves from 1.08926 to 1.08893 over all 40 steps, and the per-tensor gradient
absmax stays essentially flat around 7e-5 to 1.3e-4 throughout -- **no
significant decay**, unlike the framing "the gradient shrinks as training
converges" implicitly assumes. A second run tracking the trainable tensors'
own values step-by-step found the real per-step movement is tiny in
absolute terms too: one tensor's absmax moved from 0.10320068 to 0.10320099
over 7 real steps, a change of about 3e-7 against a baseline of ~0.1 --
**roughly 4-5 orders of magnitude smaller than the weight's own scale.**

**Two independent calibration attempts, both real-hardware-verified, died
identically.** Built `last_half` twice via `pulsar2_docker.build()` (real
compiles, ~460-480s each, matching this section's own earlier figure):

- **Attempt 1**: calibrated the 14 trainable state tensors with their real
  captured initializer values plus a small (0.1%) per-sample jitter, and a
  real-scale `x`/`y` sample (extended `make_training_calib.py`'s
  `make_work_dir` with a new `real_data=` parameter for this -- calibrating
  a tensor against its own real values instead of an unrelated
  `weight_scale`-scaled random draw, since a real trained/initialized
  network's weights are structured, not i.i.d. random, and a random draw at
  the same *scale* does not calibrate the same *gradient magnitude* a real
  forward+backward pass through real weights produces).
- **Attempt 2**: calibrated with the actual 7-step float trajectory measured
  above (one real sample per real step, not a jittered single point) --
  built specifically to rule out "the jitter was too narrow to cover real
  per-step movement" as the cause.

On real AX650N hardware (`whisper_state_probe.c`, new -- reads a state
tensor's raw device buffer directly before/after each step, the same
methodology the original "gradient dies immediately" finding used, rather
than trusting the loss): **both attempts show the identical pattern** -- a
real, substantial step-0 update (`max|delta|` ~3.67e-4, attempt 1; ~3.67e-4,
attempt 2 -- indistinguishable between the two calibrations), then **exactly
zero movement from step 1 onward**, both attempts, to 10 steps checked.

**The real finding: this isn't a calibration-choice problem, it's an SNR
floor.** The real step-0 hardware delta (~3.7e-4, implying an effective
gradient around 0.037 at `lr=1e-2`) is 2-3 orders of magnitude *larger* than
the true float-precision gradient measured on host (~1e-4, and the real
per-step weight movement corresponds to an even smaller ~4e-6) -- meaning
the "signal" INT8 quantization returns at step 0 is dominated by
quantization error, not the true (much smaller) gradient, and by step 1
whatever that noise settles to reads as exactly zero. Calibrating more
precisely for the *real* weight/gradient scale cannot fix this: an 8-bit,
256-level quantizer whose range must also cover the weight's own ~0.1 scale
fundamentally cannot resolve a true signal 4-5 orders of magnitude smaller,
regardless of how well the calibration data matches reality -- confirmed
empirically by two differently-constructed, equally-well-matched
calibrations landing on the identical result. This is a different, more
specific mechanism than the general "the gradient shrinks and eventually
underflows" ceiling documented earlier in this doc: here the true gradient
is *already* below the noise floor at step 0, for a task (MSE regression
against an unrelated random target, near initialization) whose true
learning signal is inherently tiny -- not a symptom of training having
progressed.

**Built the automatic zero-fraction-triggered swap loop this thread's own
findings have been asking for, and it correctly detects-and-swaps -- just
not yet demonstrated over a *survives-then-dies* transition.**
`mp_calib_swap_auto_runner.c` (new) runs a live resident loop that checks,
after every step, whether the update collapsed to no signal (`>=99%` of a
combined state vector either unchanged from its input or crushed to exactly
zero -- covering *both* death signatures this project's history has found:
`w_next == w`, PR #1357's Whisper case; `w_next` reading hard zero regardless
of a nonzero `w`, PR #1356's own manual demo) and, on death, writes the
last-known-good state to host files and exits with a distinct code, rather
than running a fixed step count and having a human eyeball the printed
numbers afterward (what `mp_calib_swap_runner.c`/PR #1356's own demonstration
did). A Python orchestrator (`_work/auto_orchestrator.py`, not committed --
one-off harness, not a reusable tool) runs phase 1, checks for the death
exit code, and if seen, automatically launches phase 2 seeded from phase 1's
own dumped handoff state -- no human-picked transition step anywhere in the
loop.

**Run for real** against fresh compiles of the same small Conv+Gemm probe
`build_multiphase_calib_swap_probe.py` builds (~15s each, this graph's own
size): the controller correctly detected death at step 1 of phase 1's run
and automatically swapped to phase 2, seeded from the handoff state -- the
detection-and-handoff mechanism itself worked exactly as designed, with no
manual intervention. **But phase 2 also died at its own step 1**, so this
run does not demonstrate a full "trains fine, then automatically recovers
past a real death" cycle. Root cause, diagnosed rather than left a mystery:
this run's seed weight values (`seed_cw_normal.bin`, freshly drawn for this
task) were generated independently of `make_training_calib.py`'s own
internal calibration draw for these two builds, rather than reusing PR
#1356's own carefully-derived, confirmed-matching weight values -- so both
phases' *own* calibration likely didn't match the fed-in weights closely
enough to survive even their intended regime, independent of the
zero-fraction controller's correctness. **The controller is proven; a clean
survives-then-recovers demonstration needs the next attempt to reuse
matched calibration/seed values end to end (the way PR #1356's original
manual demo did), not freshly-drawn ones** -- a concrete, scoped next step,
not a re-open of the mechanism's own correctness.

### Audited: the "unfed grad_seed" scare -- real gap, zero actual impact

PR #1360 (investigating resident-dataset `Gather` minibatching) noticed while
writing its own `gather_runner.c` that neither `resident_runner.c` nor
`whisper_resident_runner.c` allocates a `grad_seed` buffer and then feeds it
-- raising a real worry: had every gradient-magnitude claim through those two
runners, since `grad_seed` became a real graph input (#1353), been computed
against an unintended, unwritten device buffer instead of the intended
`1.0`? Checked directly on real hardware rather than reasoned about --

**Every compiled model those two runners have ever actually been run
against has exactly the input count their hardcoded indices expect, and
none of them include `grad_seed` at all**, confirmed by loading each one and
reading `axclrtEngineGetNumInputs` back: `r18_b1.axmodel` (resnet18 speed/
batching/vNPU/memory work) reports **7** inputs; `whisper_step.axmodel` and
`whisper_p1.axmodel` (the Whisper section above) report **17**; both match
`resident_runner.c`'s and `whisper_resident_runner.c`'s own hardcoded
layouts exactly, with no 8th/18th slot to leave unfed. These models all
predate `grad_seed`'s promotion to a graph input in the code that built them
(confirmed from `master`'s own merge order: PR #1357 merged *before* #1353,
so the Whisper build it produced still had `grad_seed` baked in as
`build_backward`'s old default constant, not a runtime input) -- so **the
central claims in this project's history that route through these two
runners, Whisper's step-0/step-1 death included, were never exposed to this
bug and need no correction.** The `mp_calib_swap_auto_runner.c`
multi-phase-controller demo above checks out the same way: its own compiled
probe reports 5 inputs (`x y cw gw lr`), no `grad_seed` slot either.

The worry wasn't baseless, though -- it correctly spotted a real, *latent*
landmine rather than an active one. `build_resident_step()` on current
`master` unconditionally adds `grad_seed` (`scalars=["lr", "grad_seed"]`),
so a *fresh* rebuild of any of these graphs today would produce a model with
one more input than these two runners' fixed-size `in_bufs[]` arrays have
room for -- silently overflowing the array (not just leaving a buffer
unfed), since neither runner previously checked `ni` against what it
expected. Fixed proactively: both now bound-check `ni` against their known-
good count (refusing to guess if it's neither that nor one more), size
`in_bufs[]` for the one-more-input case, and feed `grad_seed=1.0` if it's
present -- exactly `gather_runner.c`'s own already-correct pattern, which is
what caught this in the first place. Re-ran both against the same real
models above after the fix: identical numbers (resnet18 29ms/step,
loss=0.117937; Whisper 245-246ms/step, loss=1.00265 constant across steps)
-- confirms the fix is a no-op for every model these runners have actually
been pointed at, and now safe to point at a newer one. (Also fixed, found
along the way: `whisper_resident_runner.c`'s header comment describing its
own I/O order was still resnet18's, left over from copying
`resident_runner.c` -- the actual index constants in the body were always
right, only the prose above them was stale.)

### The multi-thousand-step LossScaler run, attempted on Whisper's real graph -- blocked by a new NPU-backend crash, not a quantization-math wall

Following up on this doc's own "not yet done" item: a real multi-thousand-step
run with `finetune.LossScaler` driving `grad_seed` adaptively against the FP32
override, on `last_half` itself rather than the small isolated probes the
mechanism was previously validated on. `finetune.py`'s `LossScaler`/`train()`
already implement exactly this adaptive loop (grow the scale on sustained
underflow, back off and discard on saturation) -- it was never exercised
against a real `grad_seed` graph input before, only designed against the
pre-#1353 `ineffective`-stand-down path.

**Before spending real compile time on thousands of steps, ran the cheaper,
directly discriminating check first**: does a compiled `last_half`, with the
FP32 elementwise override, survive past its established step-1 death at all,
at any seed? A dead build makes a multi-thousand-step run pointless before it
starts. Built `last_half` fresh (523 nodes, 14 trainable tensors, matching
this doc's own established numbers), calibrated with the same real-scale
method "Recalibrating Whisper for its real gradient scale"'s Attempt 1 used
(real captured initializer values +0.1% jitter for the 14 state tensors,
`label_inputs=()` for `y`), plus a wide `real_data` list for `grad_seed`
itself spanning `1 -> 1e10` (`make_training_calib.py`'s existing "a real
multi-step trajectory" list form, applied to widen a sweep range rather than
calibrate an actual trajectory) -- avoiding the narrow-default-calibration bug
class this doc's own retroactive check (above) flagged as a real risk for
this exact tensor.

**The baseline control reproduced the established result exactly**: compiled
with no `layer_configs` override, ran on real AX650N hardware --
`loss=1.00264454` at every step (matching the earlier "last_half actually
trains" section's `1.00265` to five significant figures), a real step-0
update, then bit-identical-zero from step 1 on, confirmed unmoved by sweeping
`grad_seed` from 1 to 1e6 (identical output at both) -- the calibration setup
reproduces prior work correctly before trusting what comes next.

**The FP32-override build does not compile at all, on three independent
`layer_configs` targets, each with the identical failure signature**:

| `op_types` targeted | result |
| --- | --- |
| `Mul, Add, Sub, Div` | `NPUBackendError`: `KeyError('resident_step__div_72')` in `ddr_allocate` |
| `Mul, Add, Sub` (no `Div`) | identical `KeyError('resident_step__div_72')` |
| `Mul` alone | `KeyError('resident_step__mul_67')` -- different node, same crash site |

All three fail inside Pulsar2's own closed-source scheduler
(`axnn.yasched.test_onepass.ddr_allocate`), not in quantization or graph
validation -- the compiler gets as far as building models per-subgraph
(`graph2models`/`results2model`) before crashing on a `KeyError` for
whichever node its own FP32 promotion touched, name confirming this is a real
allocator bug reacting to the *marked* node, not a coincidence. **This is a
new, distinct NPU-backend implementation gap from anything documented
before** -- not the `MatMul`/`Conv` `data_type` rejection (that fails
cleanly, downgrading to U8 with no crash), not `highest_mix_precision`'s
`TileFailException` on `LayerNorm` tiling or resnet18's `AvgPool` scheduler
`TypeError` (both also real crashes, but in different subsystems, on the
*forward* graph's own ops, not this backward-pass DDR-allocation step) --
a fourth, independently-discovered instance of the same overall pattern this
project keeps finding: **a mechanism proven correct and safe on a small,
isolated probe graph does not survive contact with a real, ~500-node
architecture**, this time failing at compile time rather than silently
producing wrong numbers.

**Net**: the multi-thousand-step run itself was never reached, and correctly
so -- there is no compiled FP32-seed build of Whisper's real graph to run it
against, on this Pulsar2 version, with any `layer_configs` override tried.
This also sharpens, rather than merely repeats, the standing "SNR floor"
finding: even if this crash did not exist, the small-probe FP32-seed
mechanism only ever rescues gradient elements up to the boundary where the
protected elementwise chain feeds into an unprotected `MatMul` -- and
Whisper's own true signal was already established (host, float, no
quantization at all) to be 4-5 orders of magnitude below what INT8 can
resolve, a much larger gap than the 0%->20.1% rescue effect the small probe
demonstrated. So even a working compile would very likely have reproduced
the same step-1 death, for the reason "Recalibrating Whisper for its real
gradient scale" already established (accumulated INT8 precision loss through
many intermediate backward-pass ops a seed applied only near the loss cannot
reach) -- this compile crash forecloses confirming that directly, but does
not on its own reopen the SNR-floor conclusion. Not chased further: routing
around a closed-source scheduler's own `ddr_allocate` `KeyError` is not
something this project's side of the stack can fix, the same verdict
`highest_mix_precision`'s own real-architecture crashes already reached.

### Surveying NVIDIA TransformerEngine: one accidental discovery beats everything else tried

Full writeup: `docs/transformerengine-low-precision-survey.md`. Most of
TransformerEngine's real techniques don't port as designed -- its core
mechanism (a runtime-adjustable FP8 scale, no recompile) is confirmed
incompatible with Pulsar2's compile-time-baked quantisation, the same wall
every recalibration-without-recompiling idea in this thread has hit. But
checking *why*, against Pulsar2's own `build_config.proto`, surfaced a real,
previously-untried field: **`quant.highest_mix_precision: true`**. Confirmed
empirically (not from its name) to force **every op type in the graph,
`MatMul` included**, to FP32 -- a blunt whole-graph override, not
TE-style targeted mixed precision. Built and ran it on real AX650N hardware
against the small multi-phase probe: it reproduces the exact `onnxruntime`
float32 answer (`loss` and both trainable weights' updates match to
displayed precision) where the standard INT8 build gets the **wrong sign**
on both updates on the same feed -- and at this small scale, costs no more
step time than the standard build (0.250ms vs. 0.266ms min). This is a
stronger, more complete result than the FP32-seed plateau (PR #1353) or the
build-time-locked calibration swap (PR #1355/#1356): no plateau, nothing to
recompile per regime, exact float agreement, through the exact `MatMul`
boundary nothing else reached. **Tested at real scale, and it does not
survive contact with either real architecture this project has.** Whisper
`last_half` (523 nodes, 460.3s INT8 baseline reconfirmed) fails
`highest_mix_precision` after 46.7s with a real `TileFailException` on the
first `AxLayerNorm` -- an FP32 `(1,1500,512)` LayerNorm exceeds the NPU
backend's own tiling workspace limit; a `layer_configs` attempt to force
just that op back to `U8` does not compose (`highest_mix_precision`
confirmed non-overridable per-op). resnet18 (136 nodes, 65.9s INT8 baseline
reconfirmed) fails after 17.2s with a *different* real error -- an actual
Python `TypeError` inside Pulsar2's own scheduler (`'>' not supported
between instances of 'list' and 'int'`) when its `AvgPool` (resnet18d's own
avgpool-downsample shortcut, in the frozen backbone, unavoidable) is forced
to FP32. Both are real, named NPU-backend implementation gaps in Pulsar2's
own FP32 tiling support -- not a quantization-math problem, and not
something recalibration or graph restructuring on this project's side can
route around. Full writeup and exact tracebacks:
`docs/transformerengine-low-precision-survey.md`'s follow-up section.

Also built, as a genuinely portable idea separated from TE's own
CUDA-specific mechanism: `scripts/axera/amax_calibration.py`, a rolling
amax-history algorithm (TE's own statistic, applied to choosing multi-phase
calibration *targets* instead of a hand-picked ratio) that answers this
doc's own standing "how many phases, where do the boundaries go" question
from a real trajectory -- and correctly reports zero boundaries for a
trajectory shaped like Whisper's own real (flat, non-decaying) one, rather
than recommending a schedule that would not have fixed Whisper's actual
failure mode.

## What to do next

1. ~~The FP32 gradient seed.~~ **Tested: real effect, not a full fix.** See
   "The FP32 gradient seed: tested, real, but it plateaus rather than opens
   the door" above -- `grad_seed` is now a runtime input (was baked in at
   1.0), and forcing its consuming `Mul`/`Add`/`Sub`/`Div` chain to FP32
   measurably rescues gradient elements from underflow as the seed grows
   (0% -> 20.1% nonzero over seed 1 -> 1,000 on a small test graph), but
   plateaus once the seed's value reaches the weight-gradient `MatMul` --
   `layer_configs`' `FP32` override is confirmed invalid for `MatMul`/`Conv`.
   **Attempted on Whisper's real graph: blocked, not measured.** See "The
   multi-thousand-step LossScaler run, attempted on Whisper's real graph"
   above -- `layer_configs`' FP32 override, proven safe on the small probe,
   crashes Pulsar2's own NPU-backend `ddr_allocate` step on `last_half`'s
   real ~500-node graph, on every `op_types` combination tried. No compiled
   build exists to run the adaptive `LossScaler` loop against. Given
   Whisper's independently-established SNR floor (the true gradient is 4-5
   orders of magnitude below INT8's resolution, a much larger gap than the
   FP32-seed mechanism's own demonstrated rescue range), a working compile
   would likely have reproduced the same step-1 death anyway -- this remains
   the honest, unclosed state of the question, not a confirmed negative.
2. ~~Weights resident with in-graph updates.~~ **Done: 7.0x** (200.6 ms ->
   28.6 ms/step) -- see "Weights resident with in-graph updates" above.
   Residency alone was 5.2x; `_linearize_trainable_convs` (avoiding the
   per-step weight transpose `act_weight_conv_to_matmul` was paying,
   confirmed 89.6% of the whole step's `AxTranspose` cost) pushed it the
   rest of the way. Both sub-bottlenecks the profile originally flagged are
   now resolved one way or another: quantize/dequantize (48.8%) is
   **confirmed structural** (Pulsar2's own optimizer, not onnxsim's graph
   shape -- see "The quantize redundancy is real, and not fixable from the
   ONNX side"); transpose/slice (26.3%) is **fixed** (see "The
   transpose/slice half was fixable, from the ONNX side -- 28% more").
   What's left in this direction: `AxQuantizeLinear`/`AxDequantizeLinear` are
   now 56.7% of the (smaller) remaining total on a fresh profile of the fixed
   graph -- `AxDequantizeLinear`'s absolute cycle count is unchanged from
   before this fix (6,553,923, exactly), so it is very likely the same
   structural tax already investigated and not a new lead; not
   reinvestigated. **Batching more than one training step's worth of work
   per `Execute()` call -- done and confirmed real**, see "Batching: real,
   and confirms the 'not enough arithmetic' diagnosis" above: batch 8 reaches
   6.5x the achieved GOPS of batch 1 for +23% latency. The remaining lever in
   this direction is now the *offline compile time* at batch >= 16, which
   blows up (>25 min, no result) rather than the runtime step cost -- worth
   a look at Pulsar2's own tiling/dependency stages if larger batches matter,
   but this is a build-tooling problem now, not a graph-shape one.
3. **All 23 tensors, and 224x224.** Only the last four layers and a 64x64 input
   have been built for resnet18. (A different-architecture trial -- resnet50d
   `layer4.2`, still 64x64 -- compiled and ran fine, see "A different
   architecture: resnet50, first compile" above; the resolution/full-model
   question above is still open for either architecture.)
4. **An optimiser beyond SGD.** `qat_graph.adam_update` exists and has never
   been put through this path -- and now has a real in-graph-update precedent
   to extend (`build_resident_step` currently hand-rolls plain SGD rather
   than calling `qat_graph.sgd_momentum_update`/`adam_update`, specifically
   to match `finetune.py`'s zero-momentum host loop; either builtin optimizer
   would carry its own extra state tensor(s), which residency handles the
   same way).
5. **Does QAT through the card beat training in float and quantising?** On a
   single linear layer it bought +0.22 dB, which is a question the experiment
   could not answer -- a single layer has nothing to route around. The depth
   case is untested and is the one that matters.

## Where the time went

Of the blockers that cost real time, **most were self-inflicted**: a stale
artifact that made a working fix look broken, an over-strict assertion, a probe
file named `bisect.py` that shadowed the stdlib module `random` imports, a
calibration range that pinned a seed to a constant, and a `GraphBuilder` whose
functions were never attached to the model. The two genuine vendor bugs both
hid behind errors that named nothing -- which is the argument for the bisector
in `scripts/axera`: recompiling node ranges took 58 nodes to 4 in eight builds
and found in twenty minutes what four hypotheses had failed to guess.
