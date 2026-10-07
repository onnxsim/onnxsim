# Range-driven simplification (`onnxsim.range_opt`)

`onnxsim.simplify` rewrites a graph so that it computes the same function **for every input**. A
*declared input range* ("this is an image in `[0, 1]`", "these are token ids below 30522") justifies
more: a `Relu` whose operand is provably non-negative is the identity, a `Clip` whose bounds the
operand already satisfies is the identity, a `Where` with a decided condition is one of its branches,
an `int64` index tensor whose values fit in `int32` can be an `int32` tensor.
`onnxsim.range_opt` finds those rewrites with interval analysis ([`onnxsim.interval`](ranges.md),
optionally tightened by `onnxsim.crown`) and applies them.

## The safety contract

**Every range-dependent rewrite is valid only inside the declared box.** The module is built around
that:

* **Opt-in, explicitly.** Nothing in `simplify()` or the CLI calls it. `apply` raises without ranges.
* **The box travels with the model.** The rewritten model records the box it relied on under
  `onnxsim.precondition.range.<input>` and a log under `onnxsim.range_opt.log` (model-level
  `metadata_props`, the same place and JSON as [range annotations](ranges.md)).
  `check_precondition` rejects inputs outside it, so a deployment can guard.
* **Each rewrite carries its proof**: the interval that justifies it. A rewrite that is also provable
  with every input unbounded is marked `unconditional` and needs no precondition.
* **Interface changes need a second opt-in.** An `int64` graph input becoming `int32` changes what
  callers must feed, so it is only applied with `allow_interface_change=True`.

<!-- doctest -->
```python
import numpy as np
from onnx import numpy_helper, parser

from onnxsim import range_opt

m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[4] x) => (float[4] y) { a = Add(x, C)  b = Relu(a)  y = Clip(b, LO, HI) }"
)
m.graph.initializer.extend(
    [
        numpy_helper.from_array(np.full(4, 0.5, np.float32), "C"),
        numpy_helper.from_array(np.float32(0.0), "LO"),
        numpy_helper.from_array(np.float32(6.0), "HI"),
    ]
)

box = {"x": (1.0, 2.0)}
for rw in range_opt.analyze(m, box):
    if rw.applies:
        print(f"{rw.rule:<12} {rw.node:<8} {'needs the box' if not rw.unconditional else 'unconditional'}")
new, log = range_opt.apply(m, box)
print([n.op_type for n in new.graph.node])
print(log[0]["description"])
print(sorted(range_opt.preconditions(new)))
```
```text
dead_relu    Relu#1   needs the box
dead_clip    Clip#2   needs the box
['Add', 'Identity']
Relu(a) is the identity: a in [1.5, 2.5] is non-negative
['x']
```

With `x` in `[1, 2]`, `a = x + 0.5` lies in `[1.5, 2.5]`: the `Relu` and the `Clip(0, 6)` are both
identities, and what is left is `Add` plus an `Identity` that keeps the graph output's name. Both
rewrites need the box, so the model now records it, and the guard enforces it:

<!-- doctest -->
```python
inside, outside = {"x": np.full(4, 1.5, np.float32)}, {"x": np.full(4, -1.0, np.float32)}
print(range_opt.check_precondition(new, inside))
try:
    range_opt.check_precondition(new, outside)
except range_opt.PreconditionViolation as e:
    print("rejected:", e)
```
```text
[]
rejected: input 'x': observed [-1, -1] leaves the declared box [1, 2] the rewrites rely on (4 of 4 elements)
```

Outside the box the rewritten model differs from the original (here `Relu(-1 + 0.5)` is `0`, the
rewritten model returns `-0.5`): that is exactly what the recorded precondition is for. Without any
range the function refuses to run:

<!-- doctest -->
```python
try:
    range_opt.apply(m)
except ValueError as e:
    print(str(e)[:70])
```
```text
range_opt.apply needs at least one input range (input_ranges=... or an
```

## Rules

`analyze(model, input_ranges)` returns the proposals with their proofs; `apply` performs them
(`rules=[...]` restricts).

| rule | rewrite | what must be proven |
|---|---|---|
| `dead_relu` | `Relu(x)` → `x` | `lo(x) >= 0` |
| `dead_abs` | `Abs(x)` → `x`, or `Neg(x)` | `lo(x) >= 0`, or `hi(x) <= 0` |
| `dead_clip` | `Clip(x, a, b)` → `x`, or `Clip` without the redundant bound | `lo(x) >= a` and/or `hi(x) <= b` (constant bounds, input or attribute form) |
| `dead_minmax` | `Max(x, y)` → `x`, `Min(x, y)` → `x` | `lo(x) >= hi(y)` (`Max`), `hi(x) <= lo(y)` (`Min`); output shape must equal `x`'s |
| `decided_where` | `Where(c, a, b)` → `a` or `b` | `c` is a comparison (`Greater`, `Less`, `*OrEqual`, `Equal`, `Not`/`And`/`Or` of those) the intervals decide; output shape must equal the branch's |
| `decided_if` | `If(c)` → its taken branch, inlined | same; branches containing nested subgraphs are left alone |
| `cast_noop` | `Cast` to the tensor's own type → identity | none (unconditional) |
| `cast_roundtrip` | `Cast(to=B)(Cast(to=A)(x))` → `x` for a `B` tensor `x` | `A` holds every value of `x` exactly: widening is unconditional; `int64`→`int32`→`int64` needs the range to fit; `float32`→`float16` is refused |
| `narrow_const_index` | constant `int64` index tensors of `Gather`/`GatherElements`/`Slice` → `int32` | values fit; `Slice` bounds beyond `int32` (the usual "to the end" `2^63-1`) are clamped, which `Slice` semantics make equivalent; refused if anything else (`Reshape` shapes, ...) also reads the tensor |
| `narrow_input` | `int64` graph input → `int32` | range fits `int32` **and** every consumer is a `Gather`/`GatherElements` index or a `Cast`; interface change, so opt-in |
| `softmax_no_max` | decomposed softmax `Exp(x - ReduceMax(x)) / ReduceSum(...)` loses the max subtraction | the exact normalisation pattern, same reduction axes, and `hi + ln(n) <= 88` and `lo >= -80` for float32 so `exp` cannot overflow and the denominator stays normal |

A native `Softmax` op is never touched: it stabilises internally. `softmax_no_max` only applies to
graphs exported in decomposed form.

Two **report-only** findings are never applied:

* `fp16_risk`: float32 tensors, `Exp` operands and `MatMul`/`Gemm`/`Conv` accumulators
  (`sum|w| * max|x|`) whose *proven bound* exceeds float16's 65504. Intervals over-approximate, so a
  bound beyond float16 means **overflow is not excluded**, not that it happens; tensors with no finite
  bound are counted separately as unknown, not as risky.
* `int64_fits_int32`: computed `int64` tensors whose range fits `int32`. Report only, because ops such
  as `Reshape`, `Expand`, `Tile` and `ConstantOfShape` require `int64` shape operands.

### Narrowing an input is an interface change

<!-- doctest -->
```python
import onnx

emb = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (int64[2,5] ids) => (float[2,5,4] y) { y = Gather(W, ids) }"
)
emb.graph.initializer.append(numpy_helper.from_array(np.arange(400, dtype=np.float32).reshape(100, 4), "W"))

box = {"ids": (0, 99)}
(rw,) = [r for r in range_opt.analyze(emb, box) if r.rule == "narrow_input"]
print("reported:", rw.description.split(".")[0])
print("applied by default:", rw.applies)

new, _ = range_opt.apply(emb, box)
print("default apply, input type:", onnx.TensorProto.DataType.Name(new.graph.input[0].type.tensor_type.elem_type))
new, log = range_opt.apply(emb, box, allow_interface_change=True)
print("opt-in apply, input type: ", onnx.TensorProto.DataType.Name(new.graph.input[0].type.tensor_type.elem_type))
print([(r["rule"], r["interface_change"]) for r in log if r["applied"]])
print(sorted(range_opt.preconditions(new)))
```
```text
reported: graph input ids int64 -> int32 (used only as Gather[1]; hull [0, 99] fits int32)
applied by default: False
default apply, input type: INT64
opt-in apply, input type:  INT32
[('narrow_input', True)]
['ids']
```

The narrowed model records `ids` in `[0, 99]`, and `check_precondition` rejects an id outside it.

## Tighter bounds: `engine="crown"`

Plain intervals lose correlations: in `r - r` they treat the two `r` as independent. `engine="crown"`
re-tries the rules the intervals could not decide, using CROWN bounds
([`onnxsim.crown`](ranges.md)) for the operands:

<!-- doctest -->
```python
m = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[4] x) => (float[4] y) { r = Relu(x)  d = Sub(r, r)  a = Add(d, C)  y = Relu(a) }"
)
m.graph.initializer.append(numpy_helper.from_array(np.full(4, 0.1, np.float32), "C"))

box = {"x": (-1.0, 1.0)}
for engine in ("interval", "crown"):
    fired = [r for r in range_opt.analyze(m, box, engine=engine) if r.applies]
    print(engine, len(fired), [r.proof["engine"] for r in fired])
new, _ = range_opt.apply(m, box, engine="crown")
print([n.op_type for n in new.graph.node])
```
```text
interval 0 []
crown 1 ['crown']
['Relu', 'Sub', 'Add', 'Identity']
```

CROWN keeps `d = r - r` exactly `0`, so `a` is `0.1` and the last `Relu` is dead; the first `Relu`
(`x` in `[-1, 1]`) stays. CROWN's coefficient matrices are dense, so the engine only asks about
operands of at most `max_crown_elements` (4096) elements and models of at most `max_crown_nodes`
(80) nodes; larger tensors keep their interval bounds.

## Command line

```
python -m onnxsim.range_opt model.onnx --range x=0,1                      # analyze (default)
python -m onnxsim.range_opt model.onnx --range x=0,1 --apply -o out.onnx  # apply, print the log
python -m onnxsim.range_opt model.onnx --range ids=0,30521 --apply -o out.onnx --allow-interface-change
```

`--rules a,b`, `--engine crown` and `--margin` mirror the Python arguments. `--apply` without any
`--range` (and no `onnxsim.range.*` annotation) is an error.

## What it found on real exports

Measured with `scripts/range_opt_bench.py` on the ONNX exports of torchvision ResNet18 and
MobileNetV2 (opset 13, batch pinned to 1, ImageNet-normalised pixels per channel) and a
DistilBERT-SST-2 export (opset 17, `input_ids` in `[0, 30521]`, 0/1 `attention_mask`); onnxruntime
1.30.0, onnx 1.23.1, numpy 2.5.3, one CPU thread.

| model | nodes | analysis | rewrites applied | output difference inside the box (8 samples) |
|---|---|---|---|---|
| DistilBERT-SST-2 | 297 → 295 | 1.4 s | **7**: 2 `cast_noop` (`Cast` of a bool to bool), 3 `narrow_const_index`, 2 `narrow_input` (`input_ids`, `attention_mask`) | 0.0 |
| ResNet18 | 49 | 12.7 s | **0** | 0.0 (model unchanged) |
| MobileNetV2 | 170 | 8.5 s | **0** | 0.0 (model unchanged) |

The analysis time for DistilBERT includes the second, unbounded analysis that decides which rewrites
are `unconditional`; the two CNN times were measured without it (`--no-unconditional`). The full
analysis runs the interval propagation twice, so it takes longer on those models (not measured).

**Speed.** Interleaved on one thread, the rewritten DistilBERT ran at 1.005× the original's median
latency while the original measured against *itself* gave 1.006×: the difference is noise. These
rewrites remove two no-op casts and change index dtypes; they do not make a CPU run faster. The value
of narrowing is on hardware without `int64` support, which this measurement does not cover.

**Why nothing fires on the CNNs.** The rules are sound, so they need a *proof*, and plain intervals
cannot prove an activation's sign or range after a few layers (the wrapping effect):

| model | op | operands provably in range | first operand's proven hull | widest hull |
|---|---|---|---|---|
| ResNet18 | `Relu` (17) | 0 non-negative | `[-17.6, 18.0]` | `±1.2e20` |
| MobileNetV2 | `Clip(0, 6)` (35, the ReLU6s) | 0 inside `[0, 6]` | `[-5.51, 6.68]` | `[-16545, 16535]` |

So the ReLU6 `Clip`s of MobileNetV2 are *not* removable by this method even with a realistic input
range, and the first operand misses `[0, 6]` on both sides. `fp16_risk` says the same thing about
float16: all 170 MobileNetV2 tensors have a proven bound below 65504 (the bounds are loose but
finite), while on ResNet18 only 13 of 49 are proven inside float16 and 53 findings have a bound
beyond it. That is overflow *not excluded*, not overflow.

What this means in practice: the rewrites that fire are the ones that do not depend on activation
ranges (dtype and index work, no-op casts) and the ones behind bounded activations (`Sigmoid`,
`Tanh`, `Softmax` outputs feeding `Relu`/`Clip`/`Abs`/`Min`/`Max`), plus small or shallow graphs.
For deep CNNs a tighter analysis is needed; replacing the proof with *calibrated* ranges would be
unsound and is deliberately not offered.

## How it is verified

`tests/test_range_opt.py` gives every rule a positive case, a negative case where the interval does
**not** justify it (it must not fire), and an equivalence check on sampled inputs inside the box with
onnxruntime. Outside the box the difference is demonstrated, the guard is shown to catch it, and
`onnxsim.certify` confirms small rewrites inside the box and refutes one outside. A seeded randomized
test builds 60 random op chains with random boxes and requires every applied rewrite to stay
equivalent. Ten deliberate soundness mutants of the module (wrong interval endpoint, a dropped
underflow bound, a dropped `ln n`, an unsafe clamp, an int-to-int cast treated as lossless, ...) were
each caught by the suite; the first run missed the `ln n` mutant, which exposed a missing boundary test
that was then added. The mutation driver is not part of CI.

## Limits

* Intervals enclose the **real-number** function; float32 execution can differ by rounding, so a
  rewrite justified at the edge of a range (`lo` exactly `0` for a `Relu`) can change a value by
  float-rounding magnitude. `margin=` demands a gap.
* The box is a promise from the caller. A wrong declared range makes the rewritten model wrong; the
  precondition metadata and `check_precondition` exist to catch that at deployment.
* Rewrites are found one node at a time on the graph as written; `apply` repeats up to four passes.
  Patterns are matched exactly (the decomposed softmax, `Clip` with constant bounds, ...).
* `decided_if` inlines only branches without nested subgraphs.
* Narrowing covers constant index tensors and graph inputs; narrowing *computed* `int64`
  shape subgraphs is reported (`int64_fits_int32`), not applied.
* Analysis cost is that of `onnxsim.interval.propagate` (per-element float64 arrays); the unconditional
  check runs it twice (`check_unconditional=False` skips the second run).
* Not covered: any rewrite that depends on data statistics rather than a proven range.
