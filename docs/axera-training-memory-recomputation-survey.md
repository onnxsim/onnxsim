# Recomputation and scheduling for training memory: a survey, not a build

Every device-memory number this project has measured so far
(`docs/axera-on-device-training-handoff.md`'s "Device memory" section) sits
at **0.2-6.1% of the AX650N's 7040 MiB CMM**, including 8-way vNPU
concurrency saturating NPU compute at 100%. Compute is the bottleneck on
every case built so far, not memory, by a wide margin. Before spending any
engineering effort on activation recomputation (gradient checkpointing) or
memory-aware scheduling, this asks: would either actually move a real number
on this hardware, and if not yet, at what size would it start to?

**Answer, up front.** Recomputation is a small, additive change if it's ever
needed -- but nothing built so far needs it, and a rough estimate below
suggests it takes a genuinely deep, long-sequence model (several trainable
transformer layers at full-length attention, or heavy batching on top) before
it would. Scheduling/execution-order changes, by contrast, are very unlikely
to have *any* effect on the Axera path specifically, for a reason this
project already found while looking at something else entirely (see below) --
this is not a "not yet," it is closer to "probably never, for this
compiler."

## 1. Activation recomputation / gradient checkpointing

**What `graph_grad.build_backward()` does today: keeps everything, by
explicit design.** Its own docstring says so directly:

> Nothing is added to the forward graph, and no forward node is modified: the
> rules read the forward tensors by name, including node *outputs* where
> reusing them is cheaper than recomputing (`Sigmoid`, `Tanh`, `Exp`, `Sqrt`,
> `Softmax`). So the caller must place these nodes after the forward ones in
> the same graph, and keep the forward intermediates available -- which for a
> step graph they always are, since it is one graph evaluated once.

Reading `_Backward` (the per-build state class) confirms this isn't an
oversight: it holds exactly two things, the `GraphBuilder` and a shape
dictionary (`onnxsim/graph_grad.py:218-222`). There is no cache, no
reference-counting, no notion of "this tensor's forward node has already run
and its value may be discarded." Every rule reads a forward tensor by its
original name, and the module leans on that being safe *because* a step
graph is "one graph evaluated once" -- the same property that makes the
in-graph SGD update (this project's PR #1335) work at all. The five
op-output-reuse cases named above (`Sigmoid`/`Tanh`/`Exp`/`Sqrt`/`Softmax`)
are the opposite of checkpointing: a deliberate decision to keep a forward
*output* alive an extra moment specifically because recomputing it would cost
more than storing it. So this module's stance isn't "recomputation hasn't
come up" -- it's "cache, don't recompute," stated as a design principle, at
least for those five ops.

**Adding a recompute option is additive, not a rewrite -- because
`build_backward()` never needs to know.** It takes an ordered `nodes:
Sequence[onnx.NodeProto]` and reads tensor names referentially; it has no
opinion about whether a given node in that sequence is "the real forward
node" or a duplicate. So checkpointing a span of forward nodes doesn't touch
`graph_grad.py` at all -- it's a pass that runs *before* `build_backward()`,
at the level that assembles the full step graph (`build_resident_train_step.py`,
or `qat_graph.make_step_graph()`'s caller):

1. Pick a contiguous topological span of forward nodes to checkpoint (a
   transformer block is the natural unit, matching `onnxsim.qat._liveness_cuts`'s
   own boundary-finding logic below).
2. Clone that span with fresh output names, and insert the clone immediately
   before the backward nodes that need its outputs (which is already where
   `build_backward()` expects forward tensors to still be live).
3. Point the affected gradient rules at the clone's outputs instead of the
   originals -- a name substitution, not a `graph_grad.py` change.
4. The *original* span's outputs (if nothing downstream still needs them
   past the checkpoint) become dead and get removed by `onnxsim.simplify()`'s
   ordinary dead-code elimination, no new pass required for that part.

This is why the verdict is "small and additive": the mechanism composes with
existing pieces (a node-cloning pass, name substitution, `simplify()`'s
existing DCE) rather than needing new capability inside the differentiation
engine itself. What it costs is the obvious thing checkpointing always costs
-- the checkpointed span's forward compute runs twice per step (once in the
original placement, once in the recompute clone) -- and, on *this*
architecture specifically, whatever extra quantize/dequantize glue a cloned
span picks up (this project's biggest single measured cost, per the
handoff's "quantize redundancy" section, so doubling a span's forward compute
here is not free the way it might be on a compute-rich accelerator).

**Existing precedent for liveness-style reasoning in this codebase** (found,
not assumed): `onnxsim.model_info.ModelInfo._peak_memory_footprint`
(`onnxsim/model_info.py:733-799`) already does a liveness pass over a
topologically-ordered graph to compute peak resident bytes --
weights stay resident, an activation lives from its producing node to its
last consumer. `onnxsim.memory_planning.plan_activation_memory` builds on
exactly that liveness convention to compute a byte-offset arena plan, packing
two tensors into overlapping address ranges only when their liveness
intervals never overlap (`onnxsim/memory_planning.py:32-37`) --
a genuine memory-reduction mechanism, but a *buffer-reuse* one (register
allocation, given a fixed node order), not a *recomputation* one, and it is
not wired into the Axera path at all (`grep`-confirmed: only `backend.py` and
its own tests reference it). `onnxsim.qat._liveness_cuts`
(`onnxsim/qat.py:2526`) uses the same liveness-interval idea for a different
purpose -- finding block-cut boundaries for QAT's own block-wise training --
and is worth reusing as the boundary-finder for a checkpointing pass rather
than re-deriving one, since "narrows to exactly one live tensor" is precisely
where a checkpoint boundary should sit (its own docstring notes this is
exactly where a residual block naturally ends). None of these three do
recomputation; all three are evidence that this codebase already reasons
about tensor liveness comfortably, so a checkpointing pass would fit an
existing pattern rather than introducing a foreign one.

## 2. Execution/scheduling improvements

**The question:** does the order nodes execute in change peak memory (by
changing how long a tensor must stay live before its last consumer runs and
its buffer can be freed), and if so, is that a lever this project can
actually pull for the Axera deployment path.

**`onnxsim.simplify()`/onnx-optimizer do not reorder for memory today.**
Confirmed by inspection: the vendored onnx-optimizer's pass classes are
solely algebraic (fusion, constant-folding, dead-code-elimination,
attribute-normalization); grepping every pass source file for anything
naming ordering/scheduling/liveness/topology turns up nothing. `ModelInfo`'s
liveness pass and `memory_planning.py`'s arena planner (above) both **assume
the graph's existing node order** and compute a plan or a peak figure from
it -- neither searches over alternative valid topological orders to find one
with a lower peak. So today, nothing in this codebase would change an ONNX
graph's node order to reduce memory even if it wanted to; that would be new
work on top of the existing liveness machinery, not a flag to flip.

**For the Axera path specifically, this project already has direct evidence
the answer is "it wouldn't matter anyway" -- from an unrelated investigation.**
The mcode quantize-elimination probe (PR #1344,
`docs/axera-mcode-quantize-elimination-probe.md`) recompiled the *identical*
graph with *identical* calibration data twice and got mcode blobs 63%
different by byte count and a different total length, despite an identical
compiler cost estimate (`max_cycle`) both times. That is direct proof
Pulsar2's own compiler does its own internal instruction scheduling and
buffer/address assignment, independently each time, not by respecting
whatever order the input ONNX graph declared. A second, independent
confirmation from this project's own device-memory work (PR #1347): calling
`axclrtEngineGetUsage(modelPath, ...)` returns a fixed CMM size **from the
compiled file's bytes alone, before the model is even loaded onto a context,
let alone executed** -- meaning Pulsar2 has already decided, at compile
time, exactly how much device memory the compiled program needs, as a fixed
number baked into the artifact. That is what an ahead-of-time, statically
sized memory arena looks like from the outside (the same *shape* of
optimization `memory_planning.py` does explicitly for its own targets, just
implemented inside Pulsar2's closed compiler instead of onnxsim). Put
together: Pulsar2 already schedules and already statically allocates,
entirely on its own, and (per PR #1344) does not even do so
deterministically across identical inputs -- so there is no reason to expect
that changing the *order* of nodes in the ONNX graph handed to `pulsar2
build` would have any reliable, predictable effect on the compiled result's
memory or performance. This is the same finding as PR #1344's quantize-dedup
attempts, generalized: Pulsar2's own optimizer already stands between
onnxsim's output and what actually runs, for scheduling exactly as it does
for op fusion. **Verdict: not a lever for this project's actual hardware
target**, and re-litigating it the way PR #1344 did for quantize-dedup (two
spellings, both silently canonicalized away) would very likely reproduce the
same "no effect, compiler already decided" result -- flagged here as a
reasoned prediction from precedent, not confirmed by a fresh hardware test
in this task (this task deliberately did not touch hardware; see Boundaries
in the coordinating task).

**For other execution backends this project cares about, the answer is
different and more conventional.** `onnxsim.qat_graph`'s own module
docstring frames its step graphs as reaching onnxruntime CPU/CUDA execution
providers and browser targets (WebGPU via onnxruntime-web, WebNN) via the
same graph. A runtime that executes nodes in declared order and frees a
tensor once its last consumer has run (which onnxruntime's arena allocator
does) *is* sensitive to node order in the ordinary way this survey's
question 2 describes -- and `memory_planning.py`'s arena-packing plan exists,
plausibly, for exactly this kind of target (nothing in this codebase wires
it up yet; that's a separate, real gap from the Axera-specific finding
above). If a future case needs this, `memory_planning.py`'s existing
liveness convention is the piece to build a reordering search on top of --
but that is speculative future work, not something this survey found a
concrete need for today.

## 3. How big would a case need to get?

**Cross-checking `onnxsim.model_info.ModelInfo` against real hardware first,
since this survey needs an estimate and that module is the estimator this
codebase already has.** Run directly against two step graphs still on disk
from this thread's own work:

| model | `ModelInfo.memory_footprint` (fp32, static) | real measured CMM (from the handoff doc) |
| --- | --- | --- |
| resnet18 step, batch=1 | 82.7 MiB | 69 MiB |
| resnet50 step | 136.2 MiB | 88 MiB |

The static fp32 estimate and the real (mostly INT8/U8-quantized) on-device
figure land within about 1.2-1.5x of each other, estimate high -- expected,
since the estimator assumes fp32 tensors throughout while the real graph
runs mostly quantized, and the two effects (fp32-vs-int8 activations
pushing the estimate high; the compiled mcode program and quantization
parameter tables adding real bytes the estimator doesn't model, pushing it
low) partially cancel. That's close enough to use `ModelInfo` for an
order-of-magnitude projection, which is what the question in this section
needs -- not close enough to treat as a calibrated predictor of exact CMM
bytes.

**Rough projection for a memory-heavy audio case** (a several-encoder-layer
Whisper-scale model at full 30s/3000-mel-frame input, several transformer
blocks trainable -- the case a parallel task in this session is attempting
to build for real; its actual numbers, once available, should supersede this
estimate rather than be read alongside it). Self-attention's dominant memory
cost is its score matrix, `seq_len^2 x heads x 4 bytes` per layer -- at
Whisper-base's post-downsampling sequence length of ~1500 frames and 8 heads,
that's `1500^2 x 8 x 4 = ~68.7 MiB` per layer for the attention-score tensor
alone (and, per the `graph_grad` caching behavior confirmed in section 1,
`Softmax`'s output is a second full-size tensor kept alive by design, not
recomputed). Six trainable encoder layers at this scale puts attention-score
memory alone in the **~400-500 MiB** range, before Q/K/V/FFN activations (a
few tens of MiB more per layer) or any batching multiplier are added -- and
batching multiplies this linearly the same way it multiplied compute in the
batching section of the handoff doc, so batch=8 on a case like this would
land in the **several-GiB** range.

**Threshold, stated plainly:** a single-digit-percent-of-7040-MiB case (everything
built in this project so far) gives recomputation nothing to buy -- there is
no pressure to relieve. The rough projection above suggests a full-depth,
full-sequence-length, multiple-trainable-transformer-layer audio case,
especially combined with the batching this project has already shown is
close to free on the compute side, is a plausible way to actually reach the
few-hundred-MiB-to-low-GiB range where checkpointing's compute-for-memory
trade would have real memory to trade against -- call it **10-20% of the
7040 MiB CMM (roughly 700 MiB-1.4 GiB peak) as the point worth reopening this
question**, not the couple hundred MiB single-context numbers seen so far.
Below that, the extra forward-compute cost of recomputation (real, and
expensive specifically on this compiler given the quantize-glue tax) is pure
loss with no corresponding memory benefit realized. Whether the audio case
being built in parallel actually reaches that range is an open, empirical
question this survey did not settle -- it should be checked against that
case's real numbers once available, not against this section's estimate.

## Recommendation

- **Recomputation/checkpointing: not worth building now.** Nothing measured
  in this project needs it. It is cheap to add later (a node-cloning pass
  ahead of `build_backward()`, reusing `qat._liveness_cuts` for boundaries,
  no `graph_grad.py` changes) if a case crosses the rough ~700 MiB-1.4 GiB
  single-context threshold above. Revisit once (if) the audio/speech work
  produces a case that size, not before.
- **Scheduling/execution-order changes for the Axera path: not a productive
  direction at all**, on the strength of two independent pieces of evidence
  already in this project's own history (PR #1344's non-byte-stable
  recompiles; PR #1347's fixed pre-load CMM size) rather than a fresh
  negative result from this task. Don't re-run the "try two spellings, see
  if Pulsar2's optimizer eats it" experiment PR #1344 already ran for
  quantize-dedup -- the reasoning above predicts the same outcome, and a
  hardware test to confirm that prediction would be low-value effort spent
  reproducing an already-strong inference rather than a genuinely open
  question.
- **Scheduling/execution-order changes for other backends** (onnxruntime
  CPU/CUDA, WebGPU/WebNN): a real, different question this survey did not
  resolve -- `memory_planning.py`'s existing liveness/arena machinery is
  unused today and would be the right foundation if a case needing it shows
  up, but nothing in this project's current work targets those backends for
  training memory reduction specifically.

**What would need hardware to actually confirm, versus what's settled from
code alone:** everything in section 1 (recomputation's mechanism and cost)
is a code-level design argument, not yet tested even on host -- if this is
ever built, host-side correctness verification (matching this project's
usual standard) comes before any hardware measurement. Section 2's Axera
verdict rests on *inference* from two already-confirmed findings (PR #1344,
PR #1347) rather than a fresh test; a skeptical reader could still ask for
one more direct check (build two ONNX graphs differing only in declared node
order, compile both, compare `axclrtEngineGetUsage`'s reported `cmmSize` --
if they match, that's a third, more direct confirmation; if they differ,
the whole "scheduling doesn't matter here" conclusion needs revisiting) --
flagged here as the one experiment that would actually close this out,
rather than leaving it as inference from precedent alone.
