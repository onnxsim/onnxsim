# Ranged shapes and data-dependent ops

ONNX shape inference gives up on ops whose output size depends on the *values* of their
input. `NonZero` of a `float[3, 4]` comes out as `[2, unk__0]`, and `TopK` with a runtime `K`
as all-unknown dims. Before this, `onnxsim.interval.propagate` dropped every such tensor, and
with it everything downstream.

But the dynamic dimension is *bounded*: `NonZero` returns between 0 and `numel(x)` columns,
`TopK` returns `K` of them, `NonMaxSuppression` at most
`batches * classes * max_output_boxes_per_class` rows. `onnxsim.shape_ranges` represents that
bound, and `interval.propagate` keeps going with it.

## Representation

- A dimension is a `shape_ranges.Dim(lo, hi, sym)`: an integer in `[lo, hi]`, `hi=None`
  meaning unbounded, `sym` an optional name (an ONNX `dim_param` such as `batch`).
- A tensor with at least one non-exact dimension is an `interval.RangedTensor(shape, hull)`.
  Its `hull` is **one** `(lo, hi)` enclosing every element: with a dynamic number of elements
  there is no fixed per-element array to hold.
- Tensors with a fully known shape stay in `IntervalResult.intervals` as per-element arrays,
  exactly as before. Ranged ones live in `IntervalResult.ranged` (`ranged_names` lists them).
  `IntervalResult.shape(name)`, `.hull(name)` and `.contains(name, value)` work for both; for
  a ranged tensor `contains` also checks the value's shape against the ranged shape.

## What it gives you

```python
from onnx import parser
from onnxsim import interval, shape_ranges as sr

model = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[3,4] x) => (float s) {"
    "  shp = Constant<value=int64[1] {-1}>()"
    "  f = Reshape(x, shp)"
    "  nz = NonZero(f)"
    "  ax = Constant<value=int64[1] {0}>()"
    "  idx = Squeeze(nz, ax)"
    "  g = Gather(f, idx)"
    "  s = ReduceSum<keepdims=0>(g)"
    "}"
)
res = interval.propagate(model, {"x": (-1.0, 1.0)})
print(sr.shape_str(res.shape("nz")), res.ranged["nz"].hull)
print(sr.shape_str(res.shape("g")), res.ranged["g"].hull)
print(res.hull("s"), res.ranged_names)
```

```text
[1, [0,12]] (0.0, 11.0)
[[0,12]] (-1.0, 1.0)
(-12.0, 12.0) ['g', 'idx', 'nz']
```

`nz` has between 0 and 12 columns and its values are indices in `[0, 11]`. Gathering by them
yields between 0 and 12 values, each in the data's hull `[-1, 1]`. Summing at most 12 such
values gives `[-12, 12]` for `s`, which has a fully known (scalar) shape, so it is an
ordinary entry in `intervals`.

### Value information sharpens the count

If the input range excludes zero, every element is non-zero and `N` is exactly `numel`:

```python
res = interval.propagate(model, {"x": (0.5, 1.0)})
print(sr.shape_str(res.shape("nz")), "nz" in res.ranged)
lo, hi = res.hull("s")
print(round(lo, 6), round(hi, 6))
```

```text
[1, 12] False
6.0 12.0
```

The shape is static again (so `nz` moves back into `intervals`), and the sum of 12 values in
`[0.5, 1]` is `[6, 12]`. The same sharpening applies per element: with per-element input
ranges, `N` lies between the number of elements *certainly* non-zero and the number that
*might* be.

### Dynamic inputs

An input with a `dim_param` or an unknown dimension is ranged too, instead of being dropped.
`input_shapes` pins it down:

```python
dyn = parser.parse_model(
    '<ir_version: 8, opset_import: ["" : 15]> '
    "m (float[N,4] x) => (float[4] s) {"
    "  r = Relu(x)"
    "  ax = Constant<value=int64[1] {0}>()"
    "  s = ReduceSum<keepdims=0>(r, ax)"
    "}"
)
print(interval.propagate(dyn, {"x": (0.0, 1.0)}).hull("s"))
print(interval.propagate(dyn, {"x": (0.0, 1.0)}, input_shapes={"x": [(1, 8), 4]}).hull("s"))
```

```text
(0.0, inf)
(0.0, 8.0)
```

An unbounded batch gives an unbounded sum, honestly; a batch in `[1, 8]` gives at most 8.

## Shape rules

`shape_ranges` holds the rules, usable on their own:

```python
print(sr.shape_str(sr.nonzero((sr.exact(3), sr.exact(4)))))
print(sr.shape_str(sr.topk((sr.exact(2), sr.exact(6)), -1, sr.rng(1, 4))))
print(sr.shape_str(sr.nms((sr.exact(1), sr.exact(5), sr.exact(4)),
                          (sr.exact(1), sr.exact(2), sr.exact(5)), sr.exact(3))))
```

```text
[2, [0,12]]
[2, [1,4]]
[[0,6], 3]
```

| Op | Output shape | Values (scalar hull) |
|---|---|---|
| `NonZero` | `[rank, N]`, `N` in `[0, numel]` (sharpened by value info) | row `i` in `[0, dim_i - 1]` |
| `TopK` | `K` along `axis`, `K` may be a range, capped by the axis length | values in the input hull; indices in `[0, axis_len - 1]`. A *constant* `K` on a static input uses order statistics per position instead |
| `Compress` | kept length in `[0, min(dim, len(cond))]`, sharpened by a known condition | input hull |
| `Unique` | `[1, numel]` (or `0` when empty) | input hull; `counts` in `[1, numel]` |
| `NonMaxSuppression` | `[S, 3]`, `S <= batches * classes * min(max_per_class, boxes)` | indices below the largest of the three dims |
| `Range` | `max(0, ceil((limit - start) / delta))`, ranged when inputs are | from `start`/`limit` |
| `ConstantOfShape` | the shape vector, each entry a range | the constant |
| `Reshape`, `Expand`, `Tile`, `Slice`, `Concat`, `Gather`, `GatherND`, `Squeeze`, `Unsqueeze`, `Transpose`, `Flatten`, `Pad`, `MatMul`, `Reduce*` | propagated on ranged dims | unchanged (reductions and `MatMul` fold the dynamic extent in, below) |

`Shape` of a ranged tensor is an ordinary integer tensor whose entries are intervals, so
shape arithmetic (`Gather`, `Mul`, `Div`, `Concat`, ...) flows into `ConstantOfShape`,
`Reshape`, `Expand` and `Tile` and produces ranged outputs. `Div` on tensors derived from a
ranged one widens to `[floor(lo), ceil(hi)]`, because integer `Div` truncates.

## Soundness, and what is deliberately not claimed

Every rule returns dimensions containing the dimension of every *valid* execution, and every
hull encloses every element. The tests sample inputs, run onnxruntime with every
intermediate tensor exposed, and require each observed shape **and** value to lie inside what
the analysis claims.

- **Dependence on the dynamic extent is handled or dropped, never guessed.** `ReduceSum` and
  `MatMul` over a dynamic dimension multiply its range into the hull (`n` items each in
  `[lo, hi]` sum to `[min(n*lo), max(n*hi)]` over `n` in the range). `ReduceMean`,
  `ReduceMax` and `ReduceMin` over a dimension that might be empty return the full hull
  `(-inf, inf)`. An op without a rule is listed in `IntervalResult.unsupported`, and its outputs
  are *unknown*: with a static shape they get the full hull `(-inf, inf)` in `intervals`; with a
  ranged (dynamic) shape they are absent from both `intervals` and `ranged`.
- **A scalar hull is coarse.** `NonZero` indices are one hull over all rows, not one per row;
  `Gather` by them gives the data hull and loses any per-element structure of the data.
- **Reshape targets must be exact where it matters.** A target entry that *might* be `0` or
  `-1` leaves the output unknown rather than being guessed.
- **Same real-number caveat as the rest of `interval`:** float64 with a small widening, not
  rigorous float32 rounding.
- **Not used by** `crown`, `zonotope` or `certify`, which still need static shapes. `certify`
  does not use `interval` at all; `crown` and `zonotope` call `interval.propagate` for a
  warm start and read only `IntervalResult.intervals`, so a ranged tensor is simply absent
  for them -- exactly as such tensors were before ranged shapes existed.
