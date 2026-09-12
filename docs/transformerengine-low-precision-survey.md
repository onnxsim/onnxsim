# Surveying NVIDIA TransformerEngine for portable ideas -- and one accidental discovery that outdoes all of them

Scope: assess which of TransformerEngine's (github.com/NVIDIA/TransformerEngine, NVIDIA's
real, open-source FP8 low-precision training library for CUDA tensor cores)
techniques are separable from its CUDA-specific mechanism and portable to
Pulsar2/AX650N, given what this project has already confirmed about this
specific hardware's constraints (`docs/axera-on-device-training-handoff.md`'s
ceiling/multi-phase/Whisper sections, PRs #1353-#1362;
`docs/axera-quantizer-reverse-engineering.md`, PR #1354). None of TE's own
techniques turned out to be directly portable as designed. **Reading
Pulsar2's own build config schema while checking why, one real, working,
previously-untried config field was found that solves the underlying problem
TE's own FP8 recipes exist to solve -- reproducing the exact float32 answer
on real AX650N hardware, through the exact `MatMul` boundary that killed
every other lever tried in this whole thread.**

## The headline finding: `highest_mix_precision: true`, and it is not a downgrade-and-hope option

Pulsar2's `build_config.proto` (`/opt/pulsar2/axnn/yamain/config/
build_config.proto` inside the `pulsar2:7.0-lite` Docker image) documents a
`quant.highest_mix_precision` boolean, comment: *"enable highest mix
precision quantization"*. Nothing in this project's history had tried it --
every prior lever (`data_type`, `output_data_type`, mcode patching) targeted
one op or one tensor at a time and hit a wall at `MatMul` specifically.

**What it actually does, confirmed empirically, not from the name alone**:
compiled the real small Conv+Gemm training-step probe
(`build_multiphase_calib_swap_probe.py`, the same one PR #1356's multi-phase
demonstration used) twice -- once with the project's existing default config,
once adding `"highest_mix_precision": true` to the `quant` block -- and read
the real compiled `quant_axmodel.json`:

```json
"mix_precision_configs": {
  "Reshape": {"dtype": "FP32"}, "Gather": {"dtype": "FP32"},
  "Mul": {"dtype": "FP32"}, "MatMul": {"dtype": "FP32"},
  "Transpose": {"dtype": "FP32"}, "Relu": {"dtype": "FP32"},
  "Add": {"dtype": "FP32"}, "Sub": {"dtype": "FP32"},
  "Greater": {"dtype": "FP32"}, "Cast": {"dtype": "FP32"},
  "ReduceSum": {"dtype": "FP32"}, "ReduceMean": {"dtype": "FP32"}
}
```

Every op type in the graph, `MatMul` and `Sub` (the SGD-update subtract)
included -- and `tensor_configs` in the resulting file is **empty**: there is
nothing left to quantise. This is not a smart, sensitivity-driven "keep the
precision-critical layers wider, quantise the rest" scheme the name might
suggest -- it is a blunt, whole-graph FP32 override. Worth stating plainly:
this is the opposite of what "mixed precision" means in the TE/AMP sense (a
*mix* of precisions, chosen per-tensor for accuracy vs. throughput) -- here
it is all-or-nothing, at least on this graph.

**Confirmed real and correct on real AX650N hardware**, not just a compiler
curiosity: both builds were pushed to the `axcl-vm` VM and executed via
`axclrtEngineExecute`, fed identical `x`/`y`/`cw`/`gw`/`lr`/`grad_seed`
inputs, compared against an independent `onnxruntime` float32 reference over
the same `step.onnx` and the same feeds:

| build | loss | `cw[0]` delta | `cw[5]` delta |
| --- | --- | --- | --- |
| host float32 reference (`onnxruntime`) | 0.01378791 | -0.00024406239 | -0.00009133667 |
| `highest_mix_precision: true`, real hardware | 0.0137879 | -0.000244062 | -9.13367e-05 |
| standard default build, real hardware | 0.0111384 | **+2.9806e-05** | **+3.27863e-05** |

The `highest_mix_precision` build reproduces the true float32 answer to
displayed precision, on real hardware, through the real `MatMul` this whole
thread's other levers could never reach. The standard INT8 build is not
merely less precise here -- it gets the **wrong sign** on both weights'
updates, on this feed. This is a materially stronger, more complete result
than the FP32-gradient-seed work's 20%-nonzero plateau (PR #1353) or the
calibration-swap technique's build-time-locked single regime (PR #1355/#1356):
no plateau, no regime lock-in, exact agreement with float.

**The real cost, honestly measured rather than assumed**: on this specific
tiny probe (37 nodes, kilobyte-scale tensors), `highest_mix_precision`'s
compiled model ran at **0.250ms min / 0.260ms avg** per `Execute()` call
against the standard build's **0.266ms min / 0.286ms avg** -- indistinguishable,
if anything marginally faster. **This does not settle the real question.**
At this tiny scale, both numbers are dominated by fixed per-call dispatch
overhead, not arithmetic -- exactly the same "not enough arithmetic to see
the real cost" trap `docs/axera-on-device-training-handoff.md`'s own batching
section already documents for a different measurement. Whether an all-FP32
build costs meaningfully more step time on a real, arithmetic-heavy graph
(resnet18's real trainable tail, or Whisper's real transformer block) is
**untested** and is the obvious, concrete next step -- and specifically
worth checking whether FP32 ops here still dispatch to the NPU's own compute
engine at all, or fall back to a CPU/DSP path that would not scale to a
bigger graph the way this tiny probe's timing suggests.

**Direct implication for Whisper's SNR-floor problem** (PR #1359): that
investigation found the fundamental issue is that the in-graph SGD update's
output (`w_next`) must represent a real weight's full ~0.1 dynamic range
*and* a ~4e-6 update within it, in the same 8-bit quantised tensor --
`highest_mix_precision` sidesteps this by not quantising that tensor (or
anything else) at all. If the real-graph step-time cost turns out to be
tolerable, this is a complete, exact fix for Whisper's specific failure --
worth testing there directly before building anything more elaborate.
**Not done in this task** -- scope was the survey plus this one probe-scale
confirmation; a real resnet18/Whisper-scale timing and Whisper-specific
correctness test is the natural next task, not this one.

## TransformerEngine ideas assessed, and why most don't port as designed

### FP8 E4M3 (forward) vs. E5M2 (backward/gradients) -- different formats per tensor role

The underlying insight -- gradients need dynamic range more than mantissa
precision, weights/activations need the reverse -- is a real, general
numerical-methods point, independent of FP8. This project already has an
analogous, coarser lever: U16 for backward ops (the *original* handoff's own
finding, predating this survey). Checked whether anything finer-grained
exists in the build config for asymmetric range/precision allocation beyond
U8/S8/U16/FP32(-on-a-narrow-op-list): no. `common.DataType` (referenced
throughout `build_config.proto`'s `data_type`/`weight_data_type`/
`output_data_type`/`conv_bias_data_type`/`ln_scale_data_type` fields) is the
only precision-selection axis found, and it is a fixed enum of whole dtypes,
not a configurable range/mantissa split the way FP8's E4M3/E5M2 choice is.
**Verdict: the insight is already informally applied here (U16 for
backward), the specific FP8-format mechanism has no analogue.**

### Delayed/current scaling via an amax history

TE tracks recent per-tensor amax values and derives the *next iteration's*
scale from them, passed as a runtime argument to the FP8 GEMM call -- no
recompile. **Confirmed not portable, precisely**: Pulsar2 bakes scale/
zero-point as literals into the compiled binary (`docs/axera-quantizer-
reverse-engineering.md`), and there is no `AxQuantizedConv`/
`AxQuantizedMatMul` runtime-scale operand in the compiled program the way a
CUDA FP8 GEMM call takes one -- every scale change measured in this project's
history has required a full recompile, full stop, and nothing in
`build_config.proto` exposes a runtime-configurable scale independent of a
build.

**The separable idea -- the amax-history *algorithm*, not the mechanism --
is portable, and is built and committed as part of this task**:
`scripts/axera/amax_calibration.py`. Rather than TE's per-iteration runtime
scale update, this project's own forced constraint (recalibration needs a
recompile) means the algorithm's job is choosing *which calibration
magnitude the next phase should be built for*, and *where a longer
trajectory's boundaries should fall* -- directly answering the multi-phase
section's own stated open question ("how many phases a genuinely long run
needs, and where to place the boundaries, is open... chosen from the actual
gradient-decay curve of a real training run rather than picked by hand").
`AmaxHistory` mirrors TE's own rolling-window-max statistic (robust to one
noisy step, unlike using the single latest value); `phase_boundaries` walks
a real host-side float trajectory and reports where the rolling amax has
drifted far enough from the currently-active phase's target to warrant a
swap. Host-verified (`tests/test_amax_calibration.py`, 5 tests): tracks a
rolling max correctly, ages out a stale outlier once it leaves the window,
finds a real 100x decay transition in a synthetic trajectory shaped like
this project's own multi-phase demonstrations, and -- checked deliberately,
not just the positive case -- reports **zero boundaries** for a trajectory
shaped like Whisper's own real measured one (flat ~7e-5..1.3e-4, no
significant decay): confirming this tool would have correctly told that
investigation "this is not a decaying-scale problem, a multi-phase schedule
would not have helped" rather than recommending a schedule that (as PR
#1359 found) genuinely would not fix Whisper's actual (SNR-floor, not
decay) failure mode.

### Per-block/microscaling formats (finer-than-per-tensor granularity)

Checked directly, not assumed: `quant_axmodel.json`'s real per-tensor
`policy` structure has explicit `PER_TENSOR`, `PER_CHANNEL`, and **`PER_BLOCK`**
boolean fields (plus `PER_CHANNEL_RE_GROUP`) -- the *concept* of block-level
quantisation granularity genuinely exists somewhere in Pulsar2's internal
quantiser. On both probe builds in this task, every observed tensor's policy
had `PER_BLOCK: false` and `PER_CHANNEL: false` (activations/gradients are
per-tensor, matching PR #1354's confirmed finding; only weights get
per-channel, also already confirmed). **No field was found anywhere in
`build_config.proto` to *request* `PER_BLOCK` or per-channel granularity for
a non-weight tensor** -- the schema exposes dtype selection
(`data_type`/`output_data_type`/etc.) and a handful of named per-role
overrides, not a quantisation-granularity knob. **Verdict: the concept is
real and present in Pulsar2's own internal quantiser representation, but
not exposed as something this project can request for the tensors that
matter here (gradients/updates) -- a genuinely open question for whoever
next has Pulsar2 support-channel access, not something checkable further
from the Docker image alone.**

### Stochastic rounding

Not found as a configurable option anywhere in `build_config.proto`
(`calibration_method` is `MinMax | Percentile | MSE | KL` -- four real,
distinct calibration *methods*, none of them a rounding-mode control). Also,
honestly: stochastic rounding addresses accumulated rounding *bias* over many
steps, not the dynamic-range/resolution problem Whisper's SNR floor is --
the wrong tool for that specific failure even where it exists. Not pursued
further.

### First/last-layer-stays-higher-precision heuristics

Common in low-precision training recipes generally (TE's own guidance
includes it): keep the first and last layers of a network at higher
precision since they are disproportionately sensitive. This project's own
"train only the last N layers" scoping (resnet18's last-4-layers, Whisper's
`last_half`) is a related but different choice -- selecting *which weights
are trainable*, not their quantisation precision specifically. Given
`highest_mix_precision`'s existence, the more targeted version of this idea
(FP32 on *just* the trainable tail's ops, INT8 elsewhere, rather than the
whole graph) is now a concrete, checkable follow-on: does `layer_configs`'
existing `op_types`/`layer_names` targeting (already used throughout PRs
#1353/#1355 for `data_type`/`output_data_type`) compose with
`highest_mix_precision`, or is it genuinely whole-graph-only? **Not checked
in this task** -- a real, scoped next step if the whole-graph FP32 cost
turns out too high on a real arithmetic-heavy graph.

## Other real config-surface findings from this survey, not TE-derived but found along the way

- `calibration_method`: `MinMax` (confirmed used throughout this project),
  `Percentile`, `MSE`, `KL` are all real, distinct options. None address
  Whisper's specific failure (a single per-tensor range fundamentally cannot
  span both a real weight's scale and its tiny update, regardless of which
  statistic chooses that range) but are real, untried levers for ordinary
  outlier-driven calibration problems elsewhere in this project's work.
- `enable_adaround`: a real AdaRound (learned per-weight rounding, a
  published PTQ technique) toggle, `finetune_block_size` alongside it.
  Weight-rounding-specific, not applicable to the activation/gradient
  problem this survey was scoped to, but worth knowing exists.

## Follow-up: real-scale test, and `highest_mix_precision` does not survive contact with a real architecture

Tested on the two real models this project actually cares about, not just
the tiny probe above. **Both failed to compile -- with two different, real
NPU-backend implementation limits, not a correctness or calibration
problem.**

**Whisper `last_half`** (the real 523-node, 11,534,336-trainable-param step
graph from PR #1357/#1359): the standard INT8 build reproduced its
established 460.3s compile time exactly (sanity-checking that nothing else
had drifted). `highest_mix_precision: true` fails after 46.7s with a real
`TileFailException` on the very first `AxLayerNorm`:

```
TileFailException("AxLayerNorm, ... output_dtype: 'FP32' ...
    inputs: {'x': Tensor(FP32, ..., shape=(1, 1500, 512), ...)}
    mem_limit: MemLimit(workspace=524288, max_mem_size=3141632)")
```

An FP32 LayerNorm over a real `(1, 1500, 512)` activation exceeds the NPU
backend's own tiling workspace limit -- the same failure *class* PR #1351
already found for `full_encoder`'s shared-LayerNorm case, now confirmed for
a *different* reason (FP32 memory footprint, not a live-weight/CSE conflict)
on the smaller `last_half` scope that otherwise compiles fine at INT8.
Tried the obvious partial-FP32 workaround this doc's own "what this doesn't
cover" section proposed -- a `layer_configs` entry forcing
`LayerNormalization` back to `U8` alongside `highest_mix_precision: true` --
and it does **not** compose: the identical error recurs, `output_dtype`
still reads `'FP32'`. `highest_mix_precision` is confirmed non-overridable
per-op, exactly as its blunt whole-graph behavior on the probe already
suggested; there is no cheaper partial-FP32 build available this way.

**resnet18** (the established 136-node, last-4-layers-trainable step graph):
standard INT8 reproduced its established ~62-70s compile time (65.9s here).
`highest_mix_precision: true` fails after 17.2s with a **different** real
error -- not a tiling *limit* this time but an actual exception inside
Pulsar2's own closed-source scheduler:

```
TileFailException("AxAvgPool, '>' not supported between instances of
'list' and 'int' ...
    attrs: {..., 'output_dtype': 'FP32'} ...")
```

A Python-level `TypeError` inside Pulsar2's own tiling code when it hits an
`AvgPool` forced to FP32 -- resnet18d's own avgpool-downsample shortcut
(this architecture family's own signature "d" variant trick, present in the
frozen backbone every forward pass runs through, not something the trainable
tail's own scope can avoid). This is a real bug in Pulsar2's FP32 handling
for this op, not a resource limit or a configuration mistake.

**Conclusion: `highest_mix_precision`'s exact-float-agreement result is real
and reproducible on the tiny Conv+Gemm probe, but the mechanism does not
currently survive on either real architecture this project has built a
training step for.** Both failures are specific, named, real NPU-backend
implementation gaps (an FP32 `LayerNorm` tiling-workspace limit; an FP32
`AvgPool` internal `TypeError`) -- not evidence the *quantization math* is
wrong, and not something a different calibration or graph restructuring on
this project's side can route around, since the failure is inside Pulsar2's
own closed-source scheduler reacting to `output_dtype: 'FP32'` on ops this
project's real graphs already contain. The honest read: `highest_mix_precision`
is a genuine, confirmed capability of the compiler, gated behind real bugs
in its own FP32 tiling support for at least two ordinary op types (LayerNorm,
AvgPool) that appear in almost any real model. Worth revisiting if a newer
Pulsar2 release fixes either, or if a future task finds which *other* op
types' FP32 paths are actually solid (the probe's `MatMul`/`Sub`/`Reshape`/
etc. all worked) -- narrowly targeting `highest_mix_precision` at a subgraph
that avoids `LayerNorm`/`AvgPool` entirely might still be viable, but that is
untested and would need its own real check, not assumed from this result.

## What this doesn't cover

- Whether some other real model/subgraph shape -- one that avoids `LayerNorm`
  and `AvgPool` specifically -- would let `highest_mix_precision` actually
  compile and run at real scale. Not tested; the two architectures this
  project actually has both hit one of these two ops.
- Whether Pulsar2 7.0-lite is the only affected version, or whether a
  different Pulsar2 release has more complete FP32 tiling support for these
  ops -- not checked.
- The `PER_BLOCK` policy question (whether it's requestable for a non-weight
  tensor via some mechanism not visible in `build_config.proto` alone) is
  left open, not resolved.
