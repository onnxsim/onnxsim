# Can the redundant weight-quantize tax be removed by patching mcode? No.

**Status: investigated, concluded not feasible with current understanding.**
This is a bounded feasibility probe, not an attempt to build the capability --
see `docs/axera-on-device-training-handoff.md` for the resnet18 training-step
speed work this follows on from, and `scripts/axera/mcode.py`/`emitter.py`
for what the existing mcode reverse-engineering already provides.

## The question

The resnet18 training step's quantize/dequantize tax
(`AxQuantizeLinear`+`AxDequantizeLinear`, 56.7% of NPU cycles on the current
graph) was already shown structural at the ONNX level: Pulsar2's own frontend
optimizer canonicalizes away any ONNX-graph-level attempt to share a
redundant quantize between two consumers of the same tensor, before its
quantization pass runs. Since the redundancy survives ONNX-level and
compiler-optimizer-level, the question was whether it could be removed one
level lower -- by directly patching the **compiled mcode** (the NPU command
queue embedded in the `.axmodel`'s `*_neu` initializer) to alias a redundant
quantize instruction's consumer onto an already-quantized buffer instead.

## What the existing tooling can and cannot do

`scripts/axera/mcode.py` is a lossless byte-level codec and structural
validator. Its own docstring is exact about the limit: there is no
evaluator, and "no verb's datapath semantics have been established."
`scripts/axera/emitter.py` builds on it to do one thing well: given **one
fixed reference-compiled `.axmodel`**, patch in new weight values (via a
bit-permutation map learned from several same-shape reference builds with
different weights) and new output-quantization scale/zero-point literals (via
byte offsets learned the same way). It never adds, removes, or redirects
instructions -- every edit writes a new *value* into a byte position whose
existence and format Pulsar2 already fixed.

## What was found

**The redundant pair is real and precisely confirmed, not just plausible.**
Built the current resident training-step graph (`build_resident_train_step.py`
on `master`, same one `docs/axera-on-device-training-handoff.md`'s numbers
come from) with `pulsar2_docker.build(profile=True)` and inspected the
frontend-dumped `optimized_quant_axmodel.onnx` plus `op_profile.csv` (the
work directory reused is `resnet18-batch-sweep/work_b1`, the batch=1 build
from the batch-scaling PR). `fc.weight` -- the cleanest case in the graph,
since it needs no im2col/tap reshaping the way the trainable convolutions'
weights do -- has three separate `AxQuantizeLinear` instructions,
`fc.weight_QuantizeLinear_{0,1,2}`, one per consumer (the forward matmul, the
gradient matmul, and the in-graph SGD-update `Sub`). Their quantization
params, read straight from `op_profile.csv`'s `const_inputs`:

| instruction | consumer | dtype | zero point | scale |
| --- | --- | --- | --- | --- |
| `_QuantizeLinear_0` | forward matmul | S8 | 0 | 0.00202852 |
| `_QuantizeLinear_1` | gradient matmul | S8 | 0 | 0.00202852 |
| `_QuantizeLinear_2` | SGD-update `Sub` | U8 | 124 | 0.00197445 |

`_0` and `_1` are bit-for-bit the same quantization domain -- a genuinely,
exactly redundant pair, not an approximation. `_2` is legitimately different
(different dtype and scale) and was never a candidate. This confirms the
ONNX-level finding at one level lower, with exact numbers, for the first
time.

**But the win available even in the best case is small.** This clean pair
costs 8 x 51,256.25 = 410,010 cycles (the op is split into four sub-tiles
each), against a step total of 28,069,688 cycles in this profile -- **1.46%**
of the step, eliminable only in the fully-successful case. The much larger
share of quantize cost (`onnx::Conv_277`/`onnx::Conv_274`, the two trainable
3x3 convolutions' weights, 30.8% of the step's cycles combined) is **not**
this kind of redundancy: each has exactly one whole-tensor quantize (feeding
whichever consumer needs the raw weight layout) and one separately-tiled
16-way-split quantize of a *differently-shaped* view (feeding the
im2col/tap-reshaped matmul consumers) -- two genuinely different tensor
layouts from the same logical weight, not two copies of the same layout. A
buffer-aliasing edit cannot help there at all; the two representations are
not interchangeable bytes.

**The blocking finding: mcode is not byte-stable across builds of the
identical graph and calibration data.** Recompiled the exact same
`step_b1.onnx` with the exact same calibration inputs (`seed=1, n=4`,
matching the batch-scaling PR's build exactly) a second time. The compiler's
own cost estimate was identical both times (`max_cycle` = 22,625,104 in
both) -- its scheduling/tiling decisions are deterministic. But the compiled
mcode blob was **not**: different length (190,800 vs 190,808 bytes) and
120,622 of ~190,800 bytes differ -- **63% of the stream**. The `npu_params`
weight table, despite carrying the exact same float weights both times, also
differs byte-for-byte throughout. Whatever assigns buffer addresses and lays
out the weight table is not seeded/deterministic the way the graph-level
cost model is.

This does not contradict `emitter.py`'s existing results -- `emit`/
`emit_table`/`emit_mcode` only ever patch bytes within **one specific
reference build**, and `learn`/`learn_mcode` only ever compare builds that
are *members of one intentionally-correlated batch* used together, never
claiming a map learned from one build (or one batch) transfers to some other,
independently-compiled build of the same shape. It does mean a
redundant-quantize-elimination patch, if one could be found, would need to be
**rederived for every new build**, not learned once and reused -- a
materially higher bar than what `emitter.py` already pays for, and one that
was not anticipated going in.

## Why this closes the question, not just pauses it

Putting the two findings together: the one clean, precisely-confirmed
redundant case is worth at most ~1.5% of step time, and even reaching that
would require (a) locating an address/buffer-selector operand whose meaning
is currently completely unestablished (unlike the scale/zero-point literals
`emitter.py` already learned, no anchor or hypothesis for this exists yet),
(b) almost certainly also locating and correctly patching whatever encodes
the consuming instruction's *wait/dependency* condition -- redirecting a
buffer address without redirecting the corresponding readiness signal risks
a read-before-write race that this format's lack of an evaluator would not
catch before it reached real hardware -- and (c) redoing both (a) and (b) on
every future build, since the byte positions in question are not stable
across builds of even the identical graph.

Given the reward is small and each of (a)-(c) is independently a real,
unsolved reverse-engineering problem (the kind that took dozens of paired
builds and weeks of work for the narrower weight-value/scale-literal case
`emitter.py` already covers), this is **not feasible to pursue further with
current understanding**. It was not attempted as a live hardware edit --
that would be the point to test any of the above, and none of the above is
established well enough yet to make that a meaningful test rather than a
blind gamble on real hardware.

## What would change this

- Establishing how per-instruction execution dependencies are encoded (what
  gates a consumer instruction on its producer's completion) would remove
  blocker (b) and is probably the single highest-leverage next step if
  anyone wants to revisit this -- it is also likely needed for other,
  unrelated future mcode work, not just this one.
- If a much larger, more clearly duplicated case turns up in a different
  model or shape (unlike this graph, where the biggest quantize cost is
  legitimate dual-layout need, not redundancy), the reward side of the
  trade-off could look different even with (a)-(c) unsolved.
- None of this blocks the ONNX-graph-level work this project already does
  well -- it is a statement about this one specific, lower-level avenue.

## Reproducing this

The reference build reused here: `scripts/axera/build_resident_train_step.py`
on current `master`, compiled via `pulsar2_docker.build(profile=True)`, same
inputs as the batch-scaling PR's `batch=1` case (`seed=1, n=4` calibration
samples). `scripts/axera/mcode.mcode_name()`/`mcode.segments()`/
`mcode.decode()` read the `*_neu` initializer; `op_profile.csv`'s
`const_inputs` column carries each `AxQuantizeLinear`'s scale/zero-point
directly, which is the fastest way to check a hypothesised redundant pair
without cross-referencing the ONNX graph by hand.
