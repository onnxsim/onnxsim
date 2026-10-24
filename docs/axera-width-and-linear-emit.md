# Other widths, ReduceMean at any width, and linear layers for new weights (AX650)

`docs/axera-graph-stitch.md` stitches a fused graph from standalone per-op programs, one
program per op and shape. This page removes two of its inputs for the LLM-shaped ops:

- a compiled `[1,64]` program is moved to another width without a Pulsar2 build at that
  width (`width_retarget.py`);
- a linear layer `y = x @ w` is emitted for a new weight matrix from one native build of
  the same shape (`linear_emit.py`).

Everything here was checked offline against native Pulsar2 7.0-lite builds (AX650,
MinMax). "Equal" means the comparison of `graph_stitch.compare_models`: every
decompressed record outside segment 0's slot table, `npu_params`, the blob's FlatBuffer
fields and the model proto. A rewritten segment is recompressed with
`short_unit_codec.encode`; its compressed bytes can differ from Pulsar2's for the same
records (the compressed length was equal in every comparison). Segments that do not
change keep the template's stream.

- Code: `scripts/axera/width_retarget.py`, `scripts/axera/linear_emit.py`; two rules
  added to `scripts/axera/graph_stitch.py`.
- Tests (no device): `tests/test_axera_width_retarget.py`,
  `tests/test_axera_linear_emit.py`, `tests/test_axera_graph_stitch.py`.
- Fixtures: `scripts/axera/fixtures/width_retarget/` (40 native builds, 106 KB) and
  `scripts/axera/fixtures/linear_emit/` (8 native builds, 724 KB: two 576x576 builds of
  339 KB each, gzip barely shrinks random weight codes; six 64x64 builds of 7 KB each).
  The 576x576 float weights are not committed (1.3 MB each): the tests regenerate them
  from the build's seed and check a committed SHA-256.

## Width retargeting

```python
import width_retarget as wr

moved = wr.retarget_width(template_64, 576, "Sigmoid")     # template's calibration
model = wr.calibrate_program(moved, "Sigmoid", scales, zero_points,
                             slot_order=["x"], output_param="late")
```

`retarget_width` rewrites the width-dependent records of a `[1,64]` program;
`calibrate_program` then moves it to a calibration (a one-op `graph_stitch`; Neg goes
through `misc_op_record_emit.retarget`). A native build also chooses two things freely,
so they are arguments: the input slot order, and where the output's PARAM job runs.

### Formulas

| what | rule | status |
| --- | --- | --- |
| scratch buffers | bump-allocated from `0x2f7000` in address order, `k·n` bytes rounded up to 32 (`k` = 1, 2, 4); PARAM buffers stay 32 bytes | derived |
| `a7 0x0100` | `0x17c00 \| (PARAM buffer − 0x2f7000) / 0x20` | derived |
| IO sizes, shapes, `outputs_info` | `n` in place of 64 | derived |
| `0x02a0 0x03c0 0x04e0 0x0710 0x1b60` | `k·n − 1` | fitted |
| `0x09b0 0x0ad0 0x0b60` | `ceil16(n) − 1` | fitted |
| `0x0bf0` | `ceil8(n) − 1` | fitted |
| `0x0e80` | `ceil(n/8) − 1` | fitted |
| block counts (`0x1fe0 0x1ff0 0x1bf0 0x1c00 0x1ca0 0x1cb0 0x1d50..0x1d80`) | `ceil(n/16) − 1`; `ceil(n/32) − 1` with flag `0x01000000` or `0x04000000` | fitted |
| a buffer above 1024 bytes | leaves the scratch window, bump-allocated from address 0 in the same order | fitted (Softmax 576; Sigmoid, Mul, Softmax 2048) |
| `0x1a70` / `0x1a80` (stride `W − 1`, `W` 8 or 32) | `n − 1` when `n` is not a multiple of `W` | fitted on n = 100 only |

### Native builds compared

Each row is predicted from the `[1,64]` template and equals the native build:

| op | widths |
| --- | --- |
| Sigmoid | 100, 384, 512, 576, 2048 |
| Mul | 100, 384, 512, 576, 2048 |
| Softmax | 384, 512, 576, 2048 |
| Add, Div, Sqrt | 100, 576 |
| Neg | 576 at two calibrations (the small and the large Neg program) |

Fused SiLU at `[1,576]` stitched from the retargeted Sigmoid and Mul templates equals the
native fused build at both calibrations.

### Refused

`ValueError` for anything outside the measured conditions:

- a template that is not `[1,64]`;
- a width below 64 or above the op's largest measured width (2048 for Sigmoid, Mul,
  Softmax; 576 for Add, Div, Sqrt, Neg);
- a width that is not a multiple of 32, except 100 for Sigmoid, Mul, Add, Div and Sqrt
  (the only such width built);
- **Softmax at a width that is not a multiple of 32.** The native `[1,100]` Softmax is a
  different, padded program: 496 main-engine records instead of 476;
- a relocated buffer whose size is not a multiple of 128 when another relocated buffer
  follows (only 1152, 2048 and 4096 bytes were seen, so the alignment is unknown). This
  refuses the multiples of 32 above 1024 that are not multiples of 128;
- a template that already holds a relocated buffer.

Multiples of 32 inside the range that were not built (96, 128, ...) are accepted: they
exercise only formulas pinned by the builds above, but no native build confirms each one.

### The output's PARAM job

Most builds run the graph output's PARAM job after the core job. Sigmoid at 512 and
ReduceMean at 512 and 576 run it before, and the fused `[1,576]` RMSNorm runs it before
ReduceMean's core. No rule predicting it was found (it is not a function of the width:
Sigmoid at 576 and 2048 are "late"). It is therefore an input: `output_param` of
`calibrate_program` and `emit_reducemean` (`"early"` / `"late"`), and
`wiring["output_param"]` in `graph_stitch`. Both orders are valid programs with the same
jobs.

## ReduceMean

```python
model = wr.emit_reducemean(576, scales, zero_points, output_param="early",
                           small=reducemean_64, large=reducemean_576)
```

ReduceMean over the last axis of `[1,n]` has three core jobs, so one template does not
serve every width:

| width | core | from |
| --- | --- | --- |
| `n ≤ 256` | direct: `0x0310 = n − 1`, `0x1ed0 = 0x40000 \| (n − 1)` | small template |
| `n > 256`, not a multiple of 256 | 256-wide pooling windows, the last one padded: `0x0310 = n − 1`, `0x0cd0 = ceil256(n) − n`, `0x0cf0 = n`, `0x0e80 = ceil(n/256) − 1` | large template |
| `n > 256`, a multiple of 256 | the large core without its pad group (`0x0c20`, `0x0cd0..0x0da0`), with `0x0280` and `0x0c10` of the small core | both |

The emitted program equals the native build at 64, 100, 128, 256, 288, 384, 512, 576 and
2048, and at the second calibration at 64 and 576 (11 builds). Four template pairs give
the same 11 programs: (64, 576), (128, 288), (256, 384), (256, 576). `output_param` is
required (previous section). Templates outside the measured kinds are refused: the small
one must be a multiple of 32 up to 256, the large one a multiple of 32 in 257..576 that
is not a multiple of 256.

The calibration of the wide core differs from the `[1,64]` one in three ways, now in
`graph_stitch`:

- the lane is single precision, `float32(float32(s_x / s_y) / n)`. The float64 form
  `s_x / (s_y · n)` is one ulp off on the native `[1,384]` build;
- the accumulator zero point is `zp_x · min(n, 256)`: one pooling window;
- a padded core holds `zp_x` in all four bytes of the eight lanes `0x0d30..0x0da0`.

## RMSNorm at 576

`graph_stitch` needed two additions for the `[1,576]` RMSNorm, both learned from the
native fused build (`docs/axera-graph-stitch.md`): the output PARAM job before
ReduceMean's core (`wiring["output_param"] = 1`), and the wide ReduceMean calibration
above. With them the graph stitched from native components (`Mul(x, x)`, ReduceMean,
`Div` and `Mul(x, gain)` at 576; `Add` and `Sqrt` at `[1,1]`; all built at `pm1`) equals
the native fused build at both calibrations. The ReduceMean component can also be the one
emitted from the 256 and 384 templates.

## Linear layers

```python
import linear_emit as le

model = le.emit_linear(template, w, {"x": s_x, "y": s_y}, {"x": zp_x, "y": zp_y})
model, scales, zero_points = le.emit_linear_from_samples(template, w, samples)
```

`w` is the float matrix `[in, out]` of `MatMul(x, w)`. The template is a native build of
the same shape with any weights. Two shapes are served: `[1,64] @ [64,64]` and
`[1,576] @ [576,576]`.

A weight set changes only `npu_params` and nine records (the DEQUANT job's eight lanes
`float32(s_y)` and its `0x1a90 = zp_y`); another input calibration also changes the QUANT
job's eight lanes `float32(1 / s_x)`.

| part | rule | status |
| --- | --- | --- |
| weight codes | per output channel `s_w = max\|w[:, c]\| / 127.5`, `code = clip(rint(w / s_w), −128, 127) + 128` in float32 | derived (`emitter.codes_of`) |
| tiles | 64 output channels per tile: a weight block (4608 bytes at 64, 36864 at 576), then a 512-byte block | derived |
| bias and multiplier | per tile, 64 float32 `zp_y − zp_x · Σq · M` then 64 float32 `M = s_x · s_w / s_y` | derived (`emitter.requant_block`) |
| 64x64 code address | plain byte at `288·((o>>1)&15) + 36·(o&1) + 72·((o>>5)&1) + 144·(i//36) + i%36` | fitted, six builds |
| 576x576 code address | two nibble planes: `nib_p(c[o,2j]) \| nib_p(c[o,2j+1]) << 4` at `1152·(o&15) + 72·((o>>4)&1) + 18432·(o>>5) + 144·(j//36) + j%36 + 36·p`, `o` inside its tile | fitted, six builds |

The scratch analysis learned the address map per shape from five builds and checked the
sixth (leave-one-out, 6 of 6 at both sizes). The module uses the closed form, so one
template is enough. The tests emit each of the six 64x64 builds from each of the other
five (30 pairs) and the two committed 576x576 builds from each other; `npu_params` is
byte-equal and the records are equal outside segment 0's slot table. The two 576 builds
differ in weights and in calibration (inputs of about ±1 and ±4), which also pins the
QUANT lanes.

With `emit_linear_from_samples` the calibration comes from
`pulsar_free_calibration`. On three of the six 64x64 builds every scale is bit-exact and
the emitted model equals the native one. On the other three `s_y` is one ulp off: the
eight DEQUANT lanes and the bias/multiplier floats move by an ulp or two, nothing else.

Refused (`ValueError`): any other shape; weights with a non-finite value or an all-zero
output channel; a zero point of 0 for `x` or `y` (the native program would leave a
zero-point write out, which no build shows); a template whose `npu_params` size or
main-engine jobs are not a linear layer's.

Limits: a native build of the shape is still needed as the template. Only weights drawn
from U(−0.1, 0.1) were built. `zp_x` was never varied between two linear builds of one
shape (128 at 64, 127 at 576); the QUANT job's `0x1b10 = zp_x` write follows the rule of
every other program.

## Device results

AX8850, AXCL V3.6.5, 2026-10-10. Emitted model versus the native build of the same
weights: the outputs were bit-identical in every case.

| model | cases | values compared per case |
| --- | --- | --- |
| linear 64x64, held-out weights | three weight sets | 3,840 |
| linear 576x576, held-out weights | three weight sets | 34,560 |

The `[1,576]` graph results (stitched SiLU, stitched RMSNorm from native components) are
in `docs/axera-graph-stitch.md`.

**Unseen weights (no native build exists).** Models for `default_rng(4242)` weights at
64x64 and 576x576, with the calibration derived by `pulsar_free_calibration` from 16
samples, were built with these modules and run on the device (80 random inputs each).
There is no native build to compare with, so the references are the float product and an
exact integer model (`q_x @ q_w`, one requantize per output channel):

| model | vs float `x @ w` | integer model vs device codes |
| --- | --- | --- |
| 64x64 | mean 0.30 output steps, max 19.2 (3 of 5,120 codes saturate) | 100% equal |
| 576x576 | mean 0.28 output steps, max 8.6 (4 of 46,080 codes saturate) | 100% equal |

The card's health check read 0.0 after each model.

## 16-bit activations (U16 linear layers)

```python
model = le.emit_linear_u16(template_u16, w, {"x": s_x, "y": s_y}, {"x": zp_x, "y": zp_y})
```

`template_u16` is a native build with `layer_configs: [{"op_types": ["FullyConnected"],
"data_type": "U16"}]`. The weights stay s8 per output channel (same codes, same
bias/multiplier formula); `x` and `y` are 16-bit, zero points `1..65535`. The differences
from the U8 layer:

| part | U16 rule | status |
| --- | --- | --- |
| table | per 32-output group, per input segment: 16 rows of `72·ceil(pairs/18)` bytes, then a 256-byte block (32 biases, 32 multipliers) | fitted |
| code address | pair `j` at `72·(j//18) + j%18` in its row, high nibble plane at +18, outputs 16..31 at +36, unused slots code 128 | fitted |
| segments | `k = 576`: one of 288 pairs; `k = 1536`: `[224, 272, 272]` | fitted, no other `k` built |
| core `0x03d0` | low half = `zp_x` (upper half: mode bits), in segments other than the main engine | fitted on three real-calibration builds |
| wide output | 576x49152: CORE / DEQUANT pairs after QUANT; only the first DEQUANT writes lanes and `zp_y`; 100-byte IO tail | template self-check |

Checked offline (`tests/test_axera_linear_emit_u16.py`, native builds as local scratch,
skipped when absent): from a template, held-out weights at 576x192, 576x576, 576x1536 and
1536x576 give byte-equal `npu_params` and equal records; the three SmolLM2-135M layer-0
MLP builds calibrated on real activations (another `zp_x`) are equal too; the 576x49152
layer reproduces its own `npu_params`.

On the device (AX8850, 2026-10-24) the eight SmolLM2-135M layer-0 + post linears were
emitted at U16 with scales from real decode steps (range x2) and run on 64 held-out decode
steps. Run on the float reference input, each is within 0.4%..1.8% relative RMS of float
(the U8 emits of q, k and v, whose inputs carry almost no earlier error, were at
3.3%..10.6% in the chained run). Chained with the other single-op programs of the layer and the
final norm + lm_head (16-bit RoPE, attention and SiLU programs from Pulsar2), the logits
are within 0.98% relative RMS of float and the top-1 token agrees on 63 of 64 steps (all
U8: 14.8%, 48 of 64). The card's health check read 0.0 throughout.

## 16-bit single-op programs at a new calibration (`u16_retarget.py`)

```python
import u16_retarget as ur

q = ur.calibrate(build_dir, "Mul")            # U16 MinMax from the program's samples
model = ur.retarget(template, "Mul", ur.quant_of(template_dir), q, ("a", "b"))
```

`retarget` moves a native single-op U16 build (Mul, Add, Div, ReduceMean, Sqrt, Sigmoid,
Softmax, dynamic MatMul) to another calibration. It finds each calibration record by value:
an A1 record or `npu_params` word equal to one candidate formula on the template's calibration
is rewritten with that formula on the target's. QUANT jobs may hold only input fields and
DEQUANT jobs only output fields. The formulas are the U8 `graph_stitch` rules at 16 bits plus:
Add's `0x1ea0 = 15 - k` shift with a `Q(15 - k)` offset, the dynamic MatMul lane blocks
`zp_y, M, zp_y, 256 M`, and a 256-entry u16 Sigmoid table (one entry per 256 input codes).
Pulsar2 leaves out a write of the value a register already holds; `retarget` drops writes that
become redundant and re-pads the segment. It refuses a template whose zero points are below 256
or shared between tensors, because such a template has already left out writes.

`calibrate` reproduces Pulsar2's U16 MinMax on 27 of the 28 checked builds (zero point
`rint(-lo / s)`, but 32767 for a symmetric range `lo = -hi`; dynamic MatMul inputs signed,
`max|x| / 32767.5`). On the 28th, one ReduceMean output scale is two ulps off.

Offline (`tests/test_axera_u16_retarget.py`, native builds as local scratch): across 16 op/shape
groups (the SmolLM2-135M layer-0 programs plus calibration variants), every template-to-target
pair either equals the native build of the target calibration or is refused for a stated reason.

On the device (AX8850, 2026-10-24), the whole SmolLM2-135M layer 0 + final norm + lm_head ran as
single-op programs with no Pulsar2 build per calibration. That covers 32 retarget programs, ten
`emit_linear_u16` linears (RoPE's rotate-half folded into the q and k weights, an exact
permutation) and two uncalibrated Concat programs. On 64 held-out decode steps, the logits are
within 0.96% relative RMS of float and the top-1 token agrees on 63 of 64. The card's health
check read 0.0 throughout.
