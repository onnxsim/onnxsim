# Backward bound on the difference of two models (`onnxsim.backward_diff`)

`onnxsim.zonotope.bound_difference` certifies `|orig(x) - converted(x)|` tightly, but it carries a dense
`(symbols x tensor size)` array through every layer. The symbol count grows with `C*H*W`, so memory and conv
work grow with the *square* of the image area: a 3-layer 8-channel conv net takes 1 s at 8x8, 10 s at 16x16,
and 113 s / 7 GB at 32x32. `backward_diff.bound_difference` computes the same kind of bound with a **backward**
(CROWN-style) pass, whose cost is about *(number of output elements) x (graph size)*. For a classifier head that
is 10-1000 rows, not thousands of symbols.

```python
from onnxsim import backward_diff, quant_verify

bound = backward_diff.bound_difference(float_model, converted_model, input_ranges)  # same type as zonotope's
bound.max_abs["y"]            # certified max |float - converted| per output element

report = quant_verify.verify(float_model, quantized_model, {"x": (-1.0, 1.0)}, engine="backward")
```

`quant_verify.verify(..., engine="zonotope")` (the default) is unchanged; `engine="backward"` swaps only the
engine that bounds the difference.

## How it works

1. **The difference network.** For the two graphs (which share their real inputs) the module tracks, for every
   pair of aligned tensors `(a, b)`, the *difference tensor* `D = a - b` explicitly, by exact algebra:

   | in the pair | `D_out` |
   |---|---|
   | linear layer with constants `W_A, W_B` | `W_B D_in + (W_A - W_B) a_in + (b_A - b_B)` |
   | rounding-noise site `b_out = b_in + e` | `D_in - e` |
   | add / sub / shape ops / average pooling | the same op on `D` |
   | paired nonlinearity (Relu, Clip, Sigmoid, Tanh) | `m_mid * D_pre + e'` with `|e'| <= (m_hi - m_lo)/2 * max|D_pre|` |
   | MaxPool | box of radius = window-max of `max|D_in|` |
   | any other aligned op | box `[lo_A - hi_B, hi_A - lo_B]` from the two independent intervals (noted) |

   The nonlinearity row is the mean value theorem: `f(a) - f(b) = m (a - b)` with `m` inside the range of `f'`
   over the union of the two pre-activation ranges. It is what stops a plain product graph `orig - converted`
   from degenerating: relaxing the two Relus *independently* gives interval-style numbers (23.6 and 51.4 on exact
   rewrites, measured earlier in `onnxsim.zonotope`'s write-up).
2. **One CROWN pass.** The float model's own graph (it provides the activations the `(W_A - W_B)` terms read)
   plus the difference chain form one ONNX graph whose outputs are the `D` of the model outputs, and
   `crown.bounds` bounds them backward. Coefficients on shared tensors accumulate, so cancellation is automatic,
   and the float graph's nonlinearities only ever see the tiny `(W_A - W_B)` coefficients.

**Alignment** (which tensor of `converted` corresponds to which of `orig`) is by node structure: same op, same
attributes, aligned inputs, with `Identity` transparent. It also understands what `simplify` does: a Conv in the
converted graph can stand for the original's *Conv + BatchNorm* (`W*s`, `(b-mu)*s+beta`, in float64), and a Gemm
for *MatMul + Add*. Alignment affects tightness only (see below).

## What is and is not guaranteed

* **Sound, whatever the alignment.** Every identity above holds for any two tensors that really are the inputs of
  nodes with the stated op and constants, and the intervals behind the slope ranges and `max|D_pre|` enclose the
  actual tensors. A wrong pairing makes `D` large, not the bound wrong. A node that cannot be aligned falls back to
  a box from independent intervals, with a note; an output that cannot be aligned gets the independent-interval
  difference, with a note.
* Float64 arithmetic with a `1e-9` relative widening of every box radius. That is not directed rounding. The
  bounds describe the real-number functions; float32 execution can exceed them by float32 rounding (the tests
  budget `1e-4` relative).
* It is *looser* than the zonotope where correlation across several layers matters, and it costs one backward row
  per output *element*, so a model whose output is a large feature map costs about as much as the zonotope.

## Measured

All numbers are from one CPU, one run each, on the models described. "Certified" is the bound; "observed" is the
largest difference seen in onnxruntime over 200 random points plus a 40-step corner search.

**Tightness against the zonotope on real `simplify` output**, input box [-1, 1] (independent intervals = bounding
the two models separately and subtracting):

| case | independent intervals | zonotope | backward |
|---|---|---|---|
| Conv -> BN -> Relu, 8x8 | 20.7 | 6.7e-7 | 6.7e-7 |
| MatMul/Add/Relu/MatMul/Add -> Gemm | 93.6 | 0 | 0 |
| 2 x (Conv -> BN -> Relu), 16x16 | 63.1 | 3.3e-6 | 4.3e-6 |

**On onnxsim's own quantizers** (`accuracy.quantize`: static, static_int16, qoperator, weight_only int8/int4,
dynamic; models cnn2, mlp3, residual; boxes 0.25 and 1.0; 36 cases): the backward bound is sound in all 36 and
never infinite. Ratio backward / zonotope: **median 1.25, min 1.00, max 1.64** (the ratio is exactly 1.00 on the
conv model; the MLP is the loosest at about 1.64).

**Scaling on a conv classifier** (3 x Conv+Relu, global average pool, 10-way head; int8 weights, three int8
activation sites; box [-1, 1]; `quant_verify.verify`, per-site breakdown off):

| input | channels | zonotope | backward |
|---|---|---|---|
| 8x8 | 8 | 1.0 s, 358 MB | 0.24 s, 309 MB |
| 16x16 | 8 | 11.2 s, 1.0 GB | 0.24 s, 312 MB |
| 32x32 | 8 | 121 s, **7.6 GB** (bound 49.9, loosened) | 0.32 s, 319 MB |
| 64x64 | 8 | not run | 0.69 s, 356 MB |
| 32x32 | 16 | did not finish in 180 s | 0.46 s, 331 MB |
| 64x64 | 16 | not run | 1.26 s, 400 MB |
| 112x112 | 16 | not run | 4.9 s, 589 MB |
| 224x224 | 16 | not run | **20.1 s, 1.1 GB** |

The backward time grows about 4x per doubling of the image side (linear in pixels, with 10 rows), where the zonotope
grows about 11x. The certified bound barely moves with the image size (6.1 at 32x32 to 6.6 at 64x64 with 8
channels; 19.2 to 21.2 from 32x32 to 224x224 with 16), as expected behind a global average pool.

On a model whose *output is a feature map* the cost is rows x graph: the same 3-layer 8-channel net with a
feature-map output (per-site breakdown on, i.e. five bound computations) took 0.17 s at 8x8, 1.2 s at 16x16 and 32 s
at 32x32 for the backward engine, against 0.96 s, 10.3 s and 113 s for the zonotope. With 16 channels at 32x32 it
took 98.6 s and, on a second run on a busier machine, 192.9 s (the zonotope did not finish in 180 s); peak memory
stayed near 600 MB. Timings on this machine vary by up to 2x with load. Use `outputs=` or analyse the logits.

**Depth is the limit, for both engines.** Same classifier, 16x16 input, 8 channels, certified bound against the
observed error:

| conv layers | backward certified / observed | zonotope certified / observed |
|---|---|---|
| 2 | 92x | 77x |
| 3 | 85x | n/a |
| 5 | 1,170x | 537x |
| 8 (8x8 input) | 87,700x | 11,660x |

Beyond about 5 layers neither bound says much about the *actual* error; backward is 1.2x to 7.5x looser than the
zonotope at depth. The cause is the forward interval bounds that decide `max|D_pre|` and the float graph's own
pre-activation boxes, which grow multiplicatively per layer.

**Refinement (`refine=True`) is not worth it here.** CROWN-refining the intermediate boxes changed the certified
bound by under 1.1% on the 3-, 5- and 8-layer nets (5.703 vs 5.761, 239.3 vs 240.6, 3.643e4 vs 3.645e4) for
50-80x the time (4.4 s vs 0.08 s, 28.6 s vs 0.37 s), so `refine=False` is the default.

## ResNet18 (the real export), honestly

Float ResNet18 (torchvision ImageNet weights, batch pinned to 1) against its onnxsim-quantized counterpart,
analysing the first K logits, input box = the ImageNet-normalised image range per channel:

| quantization | input | K | backward | certified bound |
|---|---|---|---|---|
| weight_only int8 | 64x64 | 1 | 5.7 s, 3.0 GB | 5.7e19 |
| weight_only int8 | 64x64 | 10 | 5.0 s, 3.0 GB | 6.1e19 |
| weight_only int8 | 224x224 | 1 | 24 s, 3.5 GB | 1.5e21 |
| static int8 (21 sites) | 64x64 | 1 | 3.7 s, 1.5 GB | **inf** (22 hazards) |

The zonotope engine on the first row was killed by the 24 GB memory cap after 106 s. So the backward engine
*runs* on the real model in seconds, and the bound is *useless*. Per-layer instrumentation shows why: the interval
width of the **float network's own activations** grows from 36 after the first conv to about 6e19 at the logits.
The difference interval starts at about 2% of the activation width in the first block (2.25 against 118), and
from the second stage on it is as wide as the independent-interval difference (it is capped there, e.g. 5.27e6
against 5.27e6), i.e. the chain carries no information at that depth. That is plain interval blow-up of the
intermediate bounds in a 20-layer network, not an artefact of the difference construction. The
static case is infinite because `quant_verify` first needs finite clipping radii at every site, and those come
from the same blown-up ranges (the verifier reports it as a hazard, before any engine runs).

## Limits

* **Deep networks.** On the real ResNet18 export (20 convs, `MaxPool`, residual adds) the bound is vacuous (see
  below). A useful certified bound for a network of that depth needs tighter intermediate boxes than interval
  propagation gives, and refining them costs the same (image area)^2 that the backward pass avoids.
* **Many outputs.** Feature-map outputs cost a row each; pass `outputs=` to restrict, or analyse the logits.
* **Ops.** Rules exist for Conv (2-D), Gemm, MatMul, Add, Sub, Mul by a constant, Relu, Clip, Sigmoid, Tanh,
  MaxPool, AveragePool, GlobalAveragePool, Flatten, Reshape, Transpose, Squeeze, Unsqueeze, Slice and Identity.
  Anything else that aligns becomes a box; `quant_verify`'s clamp mode (`clipping="clamp"`) puts a B-only
  nonlinear sub-graph in the converted model and is not aligned, so use the default `"noise"` mode.
* **Dense memory.** Coefficient arrays are dense (rows x tensor size) inside `crown`; very large intermediate
  tensors with many rows are out of reach.
