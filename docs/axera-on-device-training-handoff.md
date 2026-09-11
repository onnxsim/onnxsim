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

### Weights resident with in-graph updates: 5.2x, measured

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
| in-graph update, **resident** (state copied device-to-device) | **38.3 ms** | 36.0 ms |

**5.2x**, not quite the "roughly 10x" estimated -- real, and short of the
estimate for a real reason (below), not a measurement artifact: the
non-resident row isolates that some of the win is just the in-graph update
itself (fewer distinct tensors cross the wire even before residency helps),
and the resident row is the full effect.

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
and it is now the bigger one. Untried: whether `fuse=True`'s single wide
matmul-per-conv (already default) can be pushed further, or whether the
per-tap quantize/dequantize pairs can be coalesced.

Correctness was checked the same way as the rest of this document -- a
directional-derivative check against `onnxruntime` on host (not the on-device
number itself, which was not re-measured to gradient precision this time;
see `tests/test_build_resident_train_step.py` for the from-scratch,
no-hardware version of that check) -- and a coarse on-device sanity check: a
near-zero dummy batch's loss landed at 0.1416 on the card against 0.0972 from
the fp32 host reference, the right order of magnitude for INT8 quantization
noise on an untuned calibration set, not a wiring bug.

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

## What to do next

1. **The FP32 gradient seed.** The one untried route past the dying gradient,
   and the difference between a 5,000-step horizon and an open-ended one.
2. ~~Weights resident with in-graph updates.~~ **Done: 5.2x** (200.6 ms ->
   38.3 ms/step) -- see "Weights resident with in-graph updates" above.
   What's left in this direction: the quantize/dequantize + transpose/slice
   glue around `act_weight_conv_to_matmul`'s per-tap decomposition is now
   *the* cost (75% of NPU cycles, per the profile above) -- worth another
   pass on its own before reaching for anything else here.
3. **All 23 tensors, and 224x224.** Only the last four layers and a 64x64 input
   have been built.
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
