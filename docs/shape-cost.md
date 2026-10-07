# Certified memory and compute bounds for dynamic shapes (`onnxsim.shape_cost`)

`onnxsim.memory_planning` and `onnxsim.model_info` need static shapes (or dim-symbol polynomials).
A model with a bounded dynamic batch (`batch <= 8`), a bounded sequence (`seq <= 2048`) or a
data-dependent op (`NonZero`, `NonMaxSuppression`, ...) gets no certified answer from them, although
the question a deployment target asks is exactly *"does it provably fit?"*. `shape_cost` answers it
with intervals: for every tensor a range of element counts and bytes, and for the whole graph a
range of peak memory, MACs and bytes moved, valid for **every** shape the ranges allow.

## Quick start

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import shape_cost

rng = np.random.default_rng(0)
model = parser.parse_model(
    """<ir_version: 8, opset_import: ["" : 17]>
    g (float[N,3,H,W] x) => (float[N,10] y) {
      c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
      r1 = Relu(c1)
      p1 = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r1)
      c2 = Conv<pads=[1,1,1,1], strides=[2,2]>(p1, W2, B2)
      r2 = Relu(c2)
      gp = GlobalAveragePool(r2)
      fl = Flatten(gp)
      y = Gemm<transB=1>(fl, W3, B3)
    }"""
)
for name, shape in dict(W1=(8, 3, 3, 3), B1=(8,), W2=(16, 8, 3, 3), B2=(16,), W3=(10, 16), B3=(10,)).items():
    model.graph.initializer.append(
        numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), name)
    )

cb = shape_cost.bounds(model, {"N": (1, 4), "H": (8, 32), "W": (8, 32)})
print("weights            :", cb.weight_bytes, "bytes")
print("peak live bytes    :", cb.peak_live_bytes)
print("MACs               :", cb.macs)
print("static arena       :", cb.arena.arena_bytes, "bytes, verified:", cb.arena.verified)
print("conv1 output       :", cb.tensors["c1"].bytes, "bytes")
```
```text
weights            : 6248 bytes
peak live bytes    : [10344, 268392]
MACs               : [18592, 1180288]
static arena       : 180224 bytes, verified: True
conv1 output       : [2048, 131072] bytes
```

`dim_ranges` maps each `dim_param` of the graph inputs to `(lo, hi)` (or an `int`, or `(lo, None)` for
an open upper end). A dimension with neither a static size nor a range is **unbounded**, and so is
everything that depends on it (below). `input_shapes=` overrides an input's dims directly and
`input_ranges=` gives *value* ranges, which matter when a shape depends on a value (a `TopK` `K`
input, a `NonZero` over data known to be positive).

### Does it provably fit?

`check_budget` compares the bounds with a target's limits:

<!-- doctest -->
```python
dims = {"N": (1, 4), "H": (8, 32), "W": (8, 32)}
print(shape_cost.check_budget(model, dims, memory_bytes=300_000, macs=2_000_000))
print(shape_cost.check_budget(model, dims, memory_bytes=200_000, arena_bytes=100_000))
print(shape_cost.check_budget(model, dims, memory_bytes=10_000))
```
```text
budget: FITS
  peak_live_bytes: proved  (bound [10344, 268392], limit 300000)
  macs: proved  (bound [18592, 1180288], limit 2000000)
budget: UNKNOWN
  peak_live_bytes: unknown  (bound [10344, 268392], limit 200000)
  arena_bytes: unknown  (bound [2048, 180224], limit 100000)
budget: DOES NOT FIT
  peak_live_bytes: exceeds  (bound [10344, 268392], limit 10000)
```

* `proved` -- the upper bound is within the limit: it fits for every shape in range.
* `exceeds` -- even the *lower* bound is over the limit: it cannot fit at any shape in range.
* `unknown` -- the limit lies between: it fits at some shapes in range and not at others (or a
  bound is unbounded). This is the common answer for a limit between the smallest and the
  largest shape, and it is not a failure of the analysis.

`fits` is `True` only if every given limit is proved, `False` if any is provably exceeded, `None`
otherwise.

### Command line

```
$ python -m onnxsim.shape_cost conv.onnx --dim N=1:4 --dim H=8:32 --dim W=8:32 --budget-memory 300000
weights            : 6248 B
peak live bytes    : [10344, 268392]  (weights + live activations)
peak activations   : [4096, 262144]
static arena       : 180224 B (verified, naive 377504 B)
MACs               : [18592, 1180288]   FLOPs: [37184, 2360576]
memory access      : [17552, 711944]
complete analysis  : True;  unbounded tensors: 0
budget: FITS
  peak_live_bytes: proved  (bound [10344, 268392], limit 300000)
```

`--json` prints the same as JSON (add `--tensors` for every tensor), `--input-range NAME=LO:HI` gives
value ranges, `--budget-arena` and `--budget-macs` add limits. The exit status is 1 only when a limit
is provably exceeded.

## What is bounded, and exactly what is claimed

| quantity | meaning | why it is a certified bound |
|---|---|---|
| per tensor `numel`, `bytes` | every valid execution inside the ranges produces a tensor whose size is in `[lo, hi]` | shape rules are interval-sound (below) |
| `peak_live_bytes` | weights resident + activations live at once, [`model_info`](../onnxsim/model_info.py)'s convention (live from production to last use, graph outputs to the end, an unconsumed tensor stays live), ideal allocator | monotone in sizes, see below |
| `peak_activation_bytes` | the same without the weights | same |
| `macs`, `flops` | multiply-accumulates of `Conv`, `ConvTranspose`, `Gemm`, `MatMul`, `Attention` and the quantized twins (`flops = 2 * macs`) | each formula is a product of dims, non-decreasing in each |
| `mem_access_bytes` | every node's inputs (weights included) plus outputs | a sum of tensor sizes |
| `arena` | a static byte-offset allocation valid for **every** shape in range | verified independently, see below |

### Why `peak_live_bytes` is sound

`max over steps t of sum over tensors live at t of size_i` is non-decreasing in every size, and which
tensors are live at step `t` depends only on the graph and its (fixed) node order -- not on sizes. So
evaluating it at the upper (lower) sizes bounds it for every shape in range. It is the peak of an
*ideal* allocator with zero fragmentation and no in-place reuse; a real allocator can need more, which
is what `arena` is for.

### Why `arena` is claimed more carefully

The obvious idea -- run `memory_planning.plan_activation_memory` at the upper-bound shapes and call its
`arena_bytes` an upper bound -- is **not sound**, for two separate reasons found while building this:

1. **The planner's arena is not monotone in sizes.** Its greedy placement can need *more* memory for
   smaller tensors. A random search over graphs of `Relu` branches merged by `Concat` found 3 cases in
   3,000 where the arena at smaller shapes exceeded the arena at larger ones; the worst: input lengths
   `[96, 96, 592, 592]` give 11,392 bytes but `[76, 20, 554, 559]` give 11,888 bytes (same graph,
   merges `[[7,1],[0,7],[5,8]]`). Measured with the compiled planner shipped in `onnxsim 0.7.3.dev4719`;
   a different planner version may behave differently, but nothing guarantees monotonicity.
2. **Its in-place aliasing is decided on the shapes it is given.** For `Add`, `Mul`, ... it aliases the
   output onto an operand when that operand's *byte size equals the output's*. At the upper bounds that
   can hold while at a smaller shape the operand is broadcast and smaller than the output --
   `Add(a[N,8], b[M,8])` with `N=1, M=4` would overwrite `a` while it is still being read.

So `arena` claims something smaller and checkable. A plan computed at the upper-bound shapes gives every
tensor a slot of its upper-bound size; that is a valid static allocation for **every** shape in range if
no two tensors that are live together overlap. `shape_cost`

* disables the planner's binary-op aliasing unless the aliased operand provably has the output's shape
  (a static, identical shape), by renaming the op in a throwaway copy;
* then **verifies** the resulting plan itself, without trusting the planner: every slot is at least the
  tensor's upper-bound size and inside the arena; any two tensors live together occupy disjoint
  addresses, except an in-place pair that is valid at every shape (a unary elementwise op or a view op,
  where the shape or the element count is preserved);
* returns the plan only if it passes (`ArenaPlan.verified`), otherwise drops it with a note.

A rank-0 tensor is presented to the planner as shape `[1]` (the same size): the repository's metrics
core ignores rank-0 tensors, so `model_info` counts a scalar as 0 bytes and the planner would leave it
unplanned. `shape_cost` counts scalars, so its static-shape numbers equal `model_info`'s on graphs
without scalar activations and are slightly larger (and correct) on graphs with them.

### What makes a quantity unbounded

Never guessed; always stated in `CostBounds.notes`, and a total that depends on one is unbounded too:

* a dimension with no range (`notes`: `dimension(s) without a range`);
* an op with no shape rule, or in a custom domain: its outputs are unbounded (`shape unknown (output
  of Mystery)`);
* a tensor whose element type is unknown or has no fixed width;
* a control-flow subgraph (`If`, `Loop`, `Scan`): its body is not analysed, so `macs`,
  `mem_access_bytes` and `peak_live_bytes` are unbounded and `complete` is `False` (summing a body once
  would under-count a loop; `model_info` does that, `shape_cost` does not).

### Where the shapes come from

`onnxsim.interval.propagate` already carries ranged shapes and the integer shape arithmetic of
`Shape -> Gather -> Mul -> Concat -> Reshape` chains, plus `NonZero`, `TopK`, `Compress`, `Unique` and
`NonMaxSuppression` (rules in `onnxsim.shape_ranges`). `shape_cost` adds interval shape rules, through an
optional `shape_fallback` argument of `propagate` (default `None`: no change for any other caller; it is
called only for a node `propagate` has no rule for whose output shapes are still unknown), for `Conv`/`ConvInteger`/`QLinearConv`/`ConvTranspose`, pooling,
`Gemm`, `MatMulInteger`/`QLinearMatMul`, normalisation layers (including `LayerNormalization`'s
statistics outputs), `Resize`/`Upsample` (constant scales or sizes), `Split`, `Einsum`, the remaining
`Reduce*`, `ArgMax`/`ArgMin`, `ScatterND`/`ScatterElements`/`GatherElements`, `DepthToSpace`/
`SpaceToDepth`, `OneHot`, `GridSample`, `RoiAlign`, the quantize/dequantize ops and the unary and
broadcasting elementwise ops. The result is intersected with the dims ONNX shape inference states exactly.

## Validation

`tests/test_shape_cost.py` checks, against onnxruntime with every intermediate tensor exposed:

* **30 shape-rule cases**, each on its own graph, at both corners of its dim ranges and at random
  interior points: every output's shape, element count and bytes lie in the certified bounds (`Conv`
  with stride/dilation/padding/groups/`SAME_UPPER`/1-D, `ConvInteger`, `MaxPool` with `ceil_mode`,
  `AveragePool`, `GlobalMaxPool`, `ConvTranspose`, `Resize` by integer scale, fractional scale and sizes,
  `Split`, `Gemm` with both transposes, `BatchNormalization`, `LayerNormalization` with its statistics
  outputs, `ReduceL2`, `ArgMax`, `Einsum`, `DepthToSpace`/`SpaceToDepth`, `OneHot`,
  `ScatterElements`/`GatherElements`, `Pow` and three-way `Sum` broadcasting, `MatMulInteger`,
  `QuantizeLinear`/`DequantizeLinear`/`DynamicQuantizeLinear`, `GridSample`, `RoiAlign`). For 24 of
  them (every case except the fractional `Resize`, `Resize` by sizes, the equal `Split` with a dynamic
  axis, `Gemm`, `Pow` and `Sum` broadcasting) the test also asserts that the upper bound is *attained*
  at the upper corner.
* **Whole models** (a conv net with dynamic batch and image size, a transformer block with dynamic batch
  and sequence, a `Shape -> Gather -> Mul -> Reshape` chain, `NonZero`, `Compress`, `TopK` with a runtime
  `K`, `NonMaxSuppression`): no tensor of any run escapes its bound, the liveness peak of the real run
  (an independent re-implementation) lies in `peak_live_bytes`, and every tensor of the run fits the slot
  the verified arena gave it.
* **Conventions:** with every dim fixed, the bounds collapse to points equal to `ModelInfo`'s `macs`,
  `mem_access` and `memory_footprint` (three shapes).
* **The verifier and the aliasing rule**: a hand-made plan that overlaps live tensors, a slot that is too
  small, a slot outside the arena, and the broadcasting-`Add` aliasing above are all rejected.
* **Mutation check:** seven deliberately unsound changes (an upper bound one too small in the shape rules,
  in the byte count, MACs of a convolution halved, scalars counted as free, binary-op aliasing always
  allowed, the plan verifier switched off, the stale declared shapes of a static export trusted after an
  override) each make the suite fail.

`scripts/shape_cost_validation.py` repeats the whole-model checks over more samples and prints how loose
the bounds are -- **certified upper bound / actual**, at the upper corner of the ranges (where a tight
bound is 1.00) and over random interior shapes (12 runs per model; one CPU):

| model (dynamic dims) | bound time | corner: peak / MACs / tensor (median) | interior: peak / MACs | arena / peak bound |
|---|---|---|---|---|
| conv net (N<=4, H,W in [8,32]) | 37 ms | 1.00 / 1.00 / 1.00 | 3.09 / 3.22 | 0.67 |
| transformer block (B<=4, S<=64) | 9 ms | 1.00 / 1.00 / 1.00 | 15.63 / 12.94 | 0.98 |
| `Shape -> Gather -> Mul -> Reshape` chain | 6 ms | 1.01 / 1.02 / 1.00 | 3.49 / 4.10 | 0.95 |
| `NonZero` (N<=16) | 2 ms | 1.98 / - / 1.50 | 5.53 / - | 0.99 |
| `Compress` (N<=32) | 2 ms | 1.44 / 2.91 / 2.91 | 3.34 / 5.49 | 0.96 |
| `TopK`, runtime `K` in [1,4] | 1 ms | 1.16 / - / 1.33 | 2.62 / - | 1.00 |
| `NonMaxSuppression` (B<=100, 5 per class) | 1 ms | 1.00 / - / 1.00 | 1.78 / - | 1.00 |

The same script validates any ONNX file (`--model PATH --dim NAME=LO:HI`, random inputs). On real exports
(torchvision ResNet18 and MobileNetV2 with a dynamic batch `<= 8`; a DistilBERT SST-2 classifier, whose
export has static `[1, 64]` inputs) every run again stayed inside every bound:

| model | tensors | bound time | unbounded | runs | corner: peak / MACs / tensor (median) | interior: peak / MACs |
|---|---|---|---|---|---|---|
| ResNet18 (batch<=8) | 92 | 423 ms | 0 | 6 | 1.00 / 1.00 / 1.00 | 1.20 / 1.47 |
| MobileNetV2 (batch<=8) | 277 | 52 ms | 0 | 6 | 1.00 / 1.00 / 1.00 | 2.99 / 5.33 |
| DistilBERT SST-2 (static `[1,64]`) | 403 | 2.2 s | 0 | 2 | 1.00 / 1.00 / 1.00 | (one shape: the bound is a point) |

On DistilBERT the certified peak (270,392,340 bytes, 268 MB of it weights) equals the real run's peak to the
byte and `macs` is the point 2,756,249,088. Treating its batch and sequence as dynamic (`B<=4, S<=128`) gives a
peak of `[270,392,340, 278,060,052]` and MACs of `[2.756e9, 1.102e10]`. The lower end does not fall below the
export's size because the export hard-codes sizes: 25 of its 26 `Reshape` targets are constants and the
attention-mask path is built from length-64 constants (35 of the first 40 tensors stay fully static even with
dynamic inputs). Only a run at that shape is a valid execution of this graph, so the bound is sound but not
tight for it; a differently exported graph would give a tighter range.

Validating on real models found two defects that the synthetic cases had missed, both fixed here:

1. **One missing rule made 195 tensors unbounded.** The exported attention mask goes through a `Where` whose
   output shape neither interval analysis (static inputs, unbounded values) nor ONNX inference could state; the
   unknown shape cascaded through the whole encoder. `Where` and the other ops interval analysis only handles
   with finite values (`MatMul`, `Reduce*`, `Gather`, `Transpose`, `Concat`, ...) now have rules, and the test
   suite checks a `Where` case against onnxruntime.
2. **An `input_shapes` override was silently clamped to the export's static shapes.** ONNX infers shapes from the
   *declared* inputs, so the declared (and exporter-declared intermediate/output) shapes of a static export
   stayed in force after the caller made an input dynamic. `onnxsim.interval.declare_input_dims` now re-declares
   the inputs and drops the stale shapes before inference, and a regression test pins it.

Read the ratios for what they are: **at the corner the bound is exact for the shape-exact graphs**; in
the interior it grows with how wide the range is (a transformer's attention scores are quadratic in the
sequence length, so a random point sits far below the `S<=64` corner). For data-dependent ops the upper
bound is the worst case -- `NonZero` assumes every element is nonzero, `Compress` that every row is kept
-- so the ratio there measures the data (random data keeps about half), not slack in the analysis.

The same analysis at larger ranges takes milliseconds: the transformer block at `B<=8, S<=2048` has a
certified peak of 1,075,851,464 bytes (about 1.0 GiB, almost all of it the attention scores
`B * heads * S^2 * 4 B`), a static arena of 1,075,838,976 bytes and at most 1,124,073,472 MACs.

## Limits

* **Independent dims.** Two inputs that both carry a `dim_param` `batch` are one number at run time, but
  each is bounded separately, and the bounds assume every dim sits at its upper bound at once. The result
  is sound, and looser than necessary when dims are tied.
* **Shape arithmetic is widened by one.** The integer shape arithmetic comes from
  `onnxsim.interval.propagate`, whose float widening can add one to a product: the `B*S` token count in the
  reshape chain above is bounded `[0, 41]` where the truth is `[1, 40]`.
* **Ideal-allocator peak.** `peak_live_bytes` follows `model_info`'s convention (no in-place reuse, no
  fragmentation); use `arena` for a real single-buffer allocation. `check_budget`'s `exceeds` for a
  *memory* limit refers to this non-aliasing metric; for the arena it uses only the largest single tensor
  as the lower bound.
* **Op coverage.** An op with no shape rule leaves its outputs unbounded (see `notes`). Not covered:
  recurrent layers (`LSTM`/`GRU`/`RNN`), `Attention`'s output shape (its MACs are counted), most
  `com.microsoft` ops, ops with a non-constant shape-valued operand that `interval` cannot evaluate.
* **Soundness is conditional** on the model being valid for the dims sampled (a model that errors at run
  time has no execution to bound), on the ranges being honest, and on the shape rules, which are checked
  against onnxruntime as above but not proved.
* **Models over 2 GiB** must be loaded by path with external data; only `ModelProto` or a path to a
  model that fits in memory is analysed here, and element types without a fixed byte width (strings)
  make a tensor unbounded.
