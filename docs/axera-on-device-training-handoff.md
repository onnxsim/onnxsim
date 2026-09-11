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

The untried way out: `layer_configs` accepts `data_type: "FP32"` for
elementwise ops, and the seed feeds a `Mul`. If the seed and its consumer stay
float, scaling should work. **Not tested.**

A second obstacle waits behind it: fixed-point clipping is **silent and local**.
It happens to intermediates that never reach an output, so a controller reading
the returned gradient sees a healthy tensor while the computation upstream is
destroyed -- unlike fp16, where overflow makes an inf that propagates. Reliable
back-off needs the graph to export `ReduceMax(|t|)` on those tensors.

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
than one already-batched step (batching and vNPU concurrency were not tried
together; whether they compose is open). `scripts/axera/tools/resident_runner.c`
now takes a `-v` flag for `AXCL_VNPU_ENABLE` to reproduce this; there is no
committed orchestration script for launching N of them, a shell loop over N
separate copies of the compiled model (see this section's own measurement
method) is enough.

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

**Not verified:** the reported loss read exactly `0` every step. `resident_
runner`'s built-in test batch is a fixed `memset(hx, 0x11, ...)`/`memset(hy,
0x22, ...)` byte pattern (not real image data), which as float32 is ~1e-28 --
effectively zero at both ends of the loss computation, so a `loss=0` reading
is consistent with the quantizer correctly rounding a near-zero range to zero
rather than a hardware/graph bug. This is the same caveat PR #1335's original
resnet18 hardware run carried ("only checked coarsely... not the full
per-tensor gradient cosine/SNR table") -- worth real image/label data through
this same runner before trusting *on-device* numeric correctness, though the
host-side finite-difference check above is materially more rigorous than
what shipped with the original resnet18 work.

**Recommendation for next time:** the pipeline needs no resnet50-specific
changes -- the natural next step is either the remaining two bottleneck
blocks of `layer4` (watch compile time; extrapolate from this block's 57.8s
before jumping straight to all three) or real calibration data through
`resident_runner` to settle the loss=0 question, not further architecture
generalization work.

## What to do next

1. **The FP32 gradient seed.** The one untried route past the dying gradient,
   and the difference between a 5,000-step horizon and an open-ended one.
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
