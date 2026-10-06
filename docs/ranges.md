# Expected value ranges

A range is a promise about *real use*: "this input is an image in `[0, 1]`", "this
output is a probability". onnxsim can store that promise on the model itself, and
then uses it in four places:

- the random inputs `simplify(check_n=...)` generates (instead of a blanket uniform
  `[0, 1)`), and a warning when an output leaves its range;
- the input box `onnxsim.certify` proves a rewrite over (see
  [Equivalence certification](#equivalence-certification-on-by-default));
- the interval analysis in `onnxsim.interval`, including its quantization bounds.

Ranges are optional. A model without them behaves exactly as before, apart from the
default-on certification described below, which can only prove so much without them.

Every Python block in this page marked as a doctest is executed by
`tests/test_docs_ranges.py`, so the numbers quoted here are checked, not remembered.

## Annotating a model

<!-- doctest -->
```python
import numpy as np
import onnx
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import ranges

rng = np.random.default_rng(0)
k = 4
model = parser.parse_model(
    """
    <ir_version: 8, opset_import: ["" : 15]>
    m (float[1,3,6,6] x) => (float[1,4,4,4] y) {
      c = Conv(x, W, B)
      b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
      y = Relu(b)
    }"""
)
f32 = lambda *s: rng.standard_normal(s).astype(np.float32)
for name, arr in dict(
    W=f32(k, 3, 3, 3), B=f32(k), g=rng.uniform(0.5, 1.5, k).astype(np.float32),
    be=f32(k), mu=f32(k), var=rng.uniform(0.5, 2, k).astype(np.float32),
).items():
    model.graph.initializer.append(numpy_helper.from_array(arr, name))

ranges.set_range(model, "x", -1.0, 1.0)  # x is an image-like input in [-1, 1]
ranges.set_range(model, "y", 0.0, None)  # y is non-negative, unbounded above
assert set(ranges.get_ranges(model)) == {"x", "y"}
```

`set_range(model, name, lo, hi)` annotates in place. `lo`/`hi` may be scalars or arrays
that broadcast to the tensor's shape, and `None` means unbounded on that side. It
replaces any earlier annotation of the same tensor and raises `ValueError` for an
empty range (`lo > hi`). `get_ranges(model)` returns `{name: (lo, hi)}` as float64
arrays, with `-inf`/`inf` for unbounded sides; `clear_range(model, name)` removes one
annotation and `clear_range(model)` removes all of them.

The same thing from the command line, on a saved model:

```console
$ python -m onnxsim.ranges model.onnx --set image=0,1 --set prob=0,1
$ python -m onnxsim.ranges model.onnx --show
image: [0, 1]
prob: [0, 1]
$ python -m onnxsim.ranges model.onnx --set prob=0,none --show
image: [0, 1]
prob: [0, inf]
$ python -m onnxsim.ranges model.onnx --clear --show
```

`--set NAME=LO,HI` takes `none` for an unbounded side, `--clear` drops every range
annotation first, `-o out.onnx` writes elsewhere (the default is in place), and with no
other flag the ranges are printed. Python's `runpy` also prints a `RuntimeWarning` to
stderr (`'onnxsim.ranges' found in sys.modules after import of package 'onnxsim'`)
because `onnxsim` already imports the module; it is harmless.

## Storage format

Each annotation is one entry in the **model-level** `metadata_props`:

```
onnxsim.range.<tensor name> = {"min": <number | nested list | null>, "max": <number | nested list | null>}
```

`null` is an unbounded side. A list must broadcast to the tensor's shape, so a
per-channel range for an NCHW image is a `[1, C, 1, 1]`-shaped nested list, which
`set_range` builds from any numpy array:

<!-- doctest -->
```python
import json

stored = {p.key: p.value for p in model.metadata_props}
assert stored["onnxsim.range.x"] == '{"min": -1.0, "max": 1.0}'

per_channel = onnx.ModelProto()
per_channel.CopyFrom(model)
ranges.set_range(per_channel, "x", np.array([0.0, -1.0, 0.5]).reshape(1, 3, 1, 1), 1.0)
value = {p.key: p.value for p in per_channel.metadata_props}["onnxsim.range.x"]
assert json.loads(value) == {"min": [[[[0.0]], [[-1.0]], [[0.5]]]], "max": 1.0}
print(value)
```

```
{"min": [[[[0.0]], [[-1.0]], [[0.5]]]], "max": 1.0}
```

### Why model-level, and not on the input or output

ONNX also lets you attach `metadata_props` to an individual input or output
(`ValueInfoProto.metadata_props`), which would be the more natural place. It does not
survive `simplify`. This was checked rather than assumed, with the block below: a note on
the graph input is gone after simplification, a note on the model is kept.

<!-- doctest -->
```python
probe = parser.parse_model(
    '<ir_version: 10, opset_import: ["" : 21]> '
    "m (float[1,3,4,4] x) => (float[1,3,4,4] y) { y = Relu(x) }"
)
entry = probe.graph.input[0].metadata_props.add()
entry.key, entry.value = "range.min", "0"
entry = probe.metadata_props.add()
entry.key, entry.value = "my.note", "kept"

simplified, _ = onnxsim.simplify(probe, certify=False)
assert list(simplified.graph.input[0].metadata_props) == []
assert [p.value for p in simplified.metadata_props if p.key == "my.note"] == ["kept"]
```

So the annotation lives on the model, keyed by tensor name. In practice that means
**graph inputs and outputs**: their names survive simplification. Annotating an
intermediate tensor works for the model as written, but `simplify` may rename or remove
that tensor, and then the annotation no longer points at anything.

## Sampling and checking from Python

<!-- doctest -->
```python
x = ranges.sample(ranges.get_ranges(model)["x"], (1, 3, 6, 6), rng=np.random.default_rng(1))
assert x.dtype == np.float32 and x.min() >= -1.0 and x.max() <= 1.0

problems = ranges.check_outputs(
    {"p": (np.asarray(0.0), np.asarray(1.0))}, {"p": np.array([0.5, 1.5])}
)
print(problems)
```

```
['p: observed [0.5, 1.5] leaves the annotated range (1 of 2 elements)']
```

`sample((lo, hi), shape, dtype=np.float32, rng=None)` draws uniformly inside the range. A
side that is infinite is replaced by one unit past the finite side (both infinite gives
`[-1, 1]`), so the sample is always finite. `check_outputs(ranges, outputs, rtol=1e-5,
atol=1e-6)` returns one message per floating-point tensor that leaves its range, with the
tolerance absorbing float32 rounding at the edges.

## What uses a range

### Random inputs for `check_n`

With `check_n > 0`, `simplify` compares the original and simplified models on random
inputs. An input that has an annotation is sampled inside its range; an integer input
(token ids, say) is sampled from the inclusive integer range. Two things leave the old
behaviour untouched:

- inputs without an annotation keep the old uniform `[0, 1)` fill;
- only the default `input_fill="random"` uses ranges. `ones`, `zeros` and `arange` are
  deliberate and are left alone, and so is data you pass as `input_data`.

### Output-range warnings

The same comparison also looks at the outputs. If an annotated output leaves its range
in the original or the simplified model, a warning is printed once per tensor and side.
It never fails the check, because a wrong annotation is not an onnxsim bug:

<!-- doctest -->
```python
probe = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[1,8] x) => (float[1,8] y) { y = Relu(x) }"
)
ranges.set_range(probe, "x", 10.0, 11.0)
ranges.set_range(probe, "y", 0.0, 1.0)  # wrong on purpose: y is in [10, 11]
_, ok = onnxsim.simplify(probe, check_n=2, certify=False)
assert ok
```

```
WARNING: original model output y: observed [10.1565, 10.8829] leaves the annotated range (8 of 8 elements)
WARNING: simplified model output y: observed [10.1565, 10.8829] leaves the annotated range (8 of 8 elements)
```

(The observed numbers depend on the random draw; the form of the message does not.)

### Equivalence certification, on by default

`simplify` also tries to *prove* that its result equals the original, with Z3
(`onnxsim.certify`; the design and its limits are in
[`verified-computation-survey.md`](verified-computation-survey.md)). It is best effort and
quiet: it runs when `z3-solver` is installed (the `verify` extra) and the model is
small, and it only speaks up when it finds a counterexample or fails internally.

| `certify=` | Behaviour |
|---|---|
| `None` (default) | Run if possible; print only a `refuted` or `error` verdict. |
| `True` | Run, and always print the outcome, including why it did not run. |
| `False` | Skip entirely. |

The CLI has `--certify` and `--no-certify` for the same three states.

The verdict is recorded in the returned model's `metadata_props` as `onnxsim.certify`
and `onnxsim.certify.detail` (the detail is cut to 300 characters). It is recorded only
when certification ran; nothing is written when it was skipped. The values are:

| Verdict | Meaning |
|---|---|
| `proved` | Every output was proved equal within `check_atol`/`check_rtol`. The detail names what proved each one: `proved-structural`, `proved-congruence`, `proved-smt`, `proved-reduced` or `proved-affine` (see below). |
| `refuted` | Z3 found an input in the range where the two models differ. Always printed. |
| `unproven-no-ranges` | Z3 found a difference, but only for huge inputs and no range was annotated; see below. |
| `skipped` | Some window was too large for the encoder, used an unsupported op, or timed out, and the zonotope fallback (below) did not prove it either. |
| `not-run` | The original and simplified models do not have the same inputs/outputs (for example `unused_output`), so they cannot be compared. |
| `error` | Certification itself failed. Printed, and never fails `simplify`. |

Certification is skipped, with the reason printed when `certify=True`, for a model whose
serialized size is over 64 MiB (the file size, when you pass a path), a model given as a
path that uses external data, a call with `output_path` (the result is streamed to disk),
or an environment without `z3-solver`. Each window is
limited to about 100,000 multiply-adds and 5 seconds, and the whole check to 10 seconds.
The input box it proves over comes from the annotations on the **original** model.

#### Why a folded BatchNorm needs a range

<!-- doctest -->
```python
sim, _ = onnxsim.simplify(model, certify=True)  # model has x in [-1, 1] annotated
assert {p.key: p.value for p in sim.metadata_props}["onnxsim.certify"] == "proved"

bare = onnx.ModelProto()
bare.CopyFrom(model)
ranges.clear_range(bare)
sim, _ = onnxsim.simplify(bare, certify=True)
verdict = {p.key: p.value for p in sim.metadata_props}["onnxsim.certify"]
assert verdict == "unproven-no-ranges"
```

```
Certify: proved (proved-congruence)
Certify: unproven-no-ranges (differs only for very large inputs (rounding of re-computed constants?); annotate input ranges with onnxsim.ranges.set_range to decide)
```

Folding a BatchNorm into the preceding Conv recomputes the weights as
`scale / sqrt(var + eps)` times the old ones, which rounds. The folded model therefore
differs from the original by a tiny amount that grows with the input. Over *unbounded*
real inputs no fixed tolerance holds, so Z3 correctly finds a counterexample (for the
model above, at an input of magnitude about `1.2e8`). That is not a bug in the fold, but
without a range it cannot be told apart from one. When the model carries no range
annotation at all and every counterexample needs that kind of huge input, the verdict is
downgraded from `refuted` to `unproven-no-ranges` and nothing is printed. Annotate the
input and the same fold is `proved`.

The downgrade looks only at annotations on graph **inputs**, because only those bound
anything certification can use. A model whose only annotation is on an output still has an
unbounded input, so the same fold is `unproven-no-ranges` (and nothing is printed), not
`refuted`. An annotation on an output never counts as a range for the input box.

#### When Z3 gives up: the zonotope fallback

Z3 handles a window by case-splitting every `Relu` in it, which stops being practical once
a `Relu` sits between two layers that were both rewritten (for example two
`Conv -> BatchNorm -> Relu` blocks whose BatchNorms were folded). If an output is still
`skipped` after the Z3 steps and **every graph input has a finite range**,
`onnxsim.certify` tries `onnxsim.zonotope`: both models are evaluated on the same input box
as affine forms over shared noise symbols, so what they share cancels exactly and only a
genuinely different rewrite costs precision. When the certified bound satisfies
`max|orig - simplified| <= atol + rtol * min|simplified|` the output's verdict is
`proved-affine`, and it counts as `proved`.

Measured with real `onnxsim.simplify` output and `x` in `[-1, 1]`: two
`Conv -> BatchNorm -> Relu` blocks on a `1x3x8x8` input went from `skipped` (Z3 gave up) to
`proved-affine` with a bound of `2.2e-06`, and the `1x3x16x16`, 4-channel version from
`skipped` to `proved-affine` with `2.8e-06`, about half a second later.

What the fallback does not do:

- **It never reports `refuted`.** A bound larger than the tolerance is an over-approximation;
  it does not show a real difference. The output stays `skipped`, with the bound in the
  detail.
- **It needs finite ranges.** With an unbounded input it is not attempted.
- **It is bounded by memory, not time.** Generators are dense, so it is tried only when a
  cost estimate made from shapes alone stays under about 256 MB (the evaluation itself
  cannot be interrupted); larger windows stay `skipped` with the reason.
- **It is real arithmetic, like the Z3 steps.** It does not model float32 rounding of the
  evaluation.

### Interval analysis and quantization bounds

`onnxsim.interval.propagate(model, input_ranges=None)` computes, for every tensor, an
elementwise `[lo, hi]` that encloses every value the tensor can take for inputs in the
box. `input_ranges` (`{input: (lo, hi)}`) is merged over the model's own annotations, and
an input with neither is unbounded. The result has `hull(name)`, `contains(name, value)`
and `unsupported`, the list of op types that fell back to "unbounded".

`quantization_bounds(model, input_ranges=None, weight_bits=8, act_bits=8)` then answers,
for each MatMul/Gemm/Conv with a constant weight, from the *reachable* activation range
instead of the scheme's full range (which is what `onnxsim.precision_estimator` assumes):
the worst-case int32 accumulator, whether it can overflow int32 or lose exactness in a
float32 cast (2**24), the activation scale and zero point the range implies, and a
certified worst-case rounding error as a fraction of the layer's output range. Weights
are quantized symmetrically per output channel and activations asymmetrically, with the
range extended to include 0. A layer whose input is unbounded, or whose weight is not a
constant, is omitted.

A worked example: a 100-term dot product with every weight `0.5`, on an input in `[-1, 1]`.

<!-- doctest -->
```python
from onnxsim import interval

layer = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[1,100] x) => (float[1,2] y) { y = MatMul(x, W) }"
)
layer.graph.initializer.append(
    numpy_helper.from_array(np.full((100, 2), 0.5, np.float32), "W")
)

result = interval.propagate(layer, {"x": (-1.0, 1.0)})
lo, hi = result.hull("y")
assert result.unsupported == [] and round(lo) == -50 and round(hi) == 50

(b,) = interval.quantization_bounds(layer, {"x": (-1.0, 1.0)})
assert b.act_zero_point == 128 and b.acc_bound == 100 * 127 * 128   # 1,625,600
assert b.acc_bound_full_range == 100 * 127 * 255                    # 3,238,500
assert b.int32_safe and b.fp32_cast_exact
assert round(b.tightening, 3) == 1.992
assert round(b.max_abs_error, 4) == 0.3937 and round(b.relative_error, 4) == 0.0039
print(interval.format_quantization_report([b]))
```

```
node                              K            act range    acc bound x tighter int32  fp32  rel err
MatMul_0                        100              [-1, 1]      1625600      1.99    ok exact    0.004
```

Reading it: every weight quantizes to `+-127`, and the activation range `[-1, 1]` gives a
zero point of 128, so a quantized activation is at most 128 away from zero point and the
accumulator is at most `100 * 127 * 128 = 1,625,600`. The full-range bound would be
`100 * 127 * 255 = 3,238,500`, so the real range is 1.99 times tighter. That is well under
int32 and under 2**24, so the integer accumulation is safe and the float32 cast is exact.
The worst-case rounding error of any output is 0.3937 against an output range of 100
(`[-50, 50]`), 0.39%.

## Limits

- **Real arithmetic, not float32 evaluation.** Both certification and interval analysis
  reason about the real-number function of the stored constants. A float32 run can exceed
  an interval by float32 rounding (about `1e-6` relative per op), which `contains` has a
  `slack` for. A `proved` verdict is a statement about the rewrite, not about every
  kernel's rounding.
- **Plain intervals lose correlations.** `x - x` is not `0`, so bounds loosen with depth
  and with residual structure. They are sound, not tight. Bound propagation over a
  whole original-vs-simplified graph is useless as soon as a `Relu` sits in both
  branches, which is why certification works on the rewritten window instead.
- **Unsupported ops become unbounded.** An op with no transfer rule gives an unbounded
  interval for its outputs, never a wrong one, and is listed in `unsupported`; layers
  downstream of it then drop out of `quantization_bounds`.
- **The quantization numbers are worst-case over the box**, not an estimate of accuracy on
  real data. A calibrated scale is usually far narrower than the interval range. Use the
  bounds to prove safety and to spot risk, and calibration to choose scales.
- **Only graph input and output names reliably keep an annotation** through `simplify`
  (see above).
- **Certification covers small windows.** A window the Z3 encoding cannot take is tried with
  the zonotope fallback when all inputs have finite ranges and it fits the memory estimate;
  otherwise it is reported `skipped` rather than proved. Only 2-D `Conv` is encoded.
