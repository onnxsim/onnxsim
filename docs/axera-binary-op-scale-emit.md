# Binary-op scale-retarget emitter (Add, Sub, Mul, Div)

`scripts/axera/binary_op_scale_emit.py` rewrites a compiled standalone AX650
`y = op(x, z)` (two live inputs of the same shape) for new per-tensor
quantization scales without Pulsar2. `scripts/axera/binary_op_scale_validate.py`
compares an emitted model with a native Pulsar2 build of the same calibration.

#1840 refused binary ops because a scale-ratio change rewrote the MCode's
variable-length short units. #1850 showed those units are LZ77 tokens over
plain 8-byte register records, so this emitter works on the *decompressed*
records and re-encodes with `short_unit_codec.encode`.

## What a calibration change touches

At fixed zero points, across every held-out pair below, the only records that
change are these (all in the TENG segment, `segments[2]`):

| value | registers | ops |
| --- | --- | --- |
| `f32(1/s_x)`, `f32(1/s_z)`, `f32(s_y)` | lane group `0x0f50..0x0fc0` | all |
| `f32(s_y/(s_x*s_z))` | `0x0fd0..0x1040` | Mul |
| `f32(s_x/(s_y*s_z))` | `0x0f50..0x0fc0` | Div |
| `15 - k` (Q15 shift) | every nonzero `0x1ea0` | Add, Sub |
| zero-point offset (int32) | `0x1ef0..0x1f20` | Add, Sub |

Products are taken in float64 over Pulsar2's float32 scales (sorted names
`x, y, z`), then rounded to float32. Each value is written once per lane (8
lanes) per tile block. Large shapes rewrite a value whenever the lane registers
were reused in between, so a value can appear `n * 8` times; `n` depends on the
shape, not on the calibration (for example `1/s_x` 16 x 8 and `1/s_z` 1 x 8 at
`[16,64,56,56]`). The emitter replaces every record holding the old value on
its registers and requires the same count on every lane register, in one
segment.

Add and Sub also carry, outside the MCode:

- **Q15 requant header** (first words of `npu_params`): `round(r * 2**(15-k))`
  for `r = s_x/s_y` and `s_z/s_y`, uint16 LE, one word when both round equal.
  `k` is the smallest shift with `max(r) * 2**-k < 1`, picked on the ratio,
  not on the rounded word: a ratio of `0.99999` keeps `k = 0` and stores
  `32768` (native census `sub_asym`, `sub_mix`, `sub_r3`). #1756's
  `round(r * 32768)` is the `k = 0` case.
- **zero-point offset**: `trunc((z_y - z_x*f32(s_x/s_y) -/+ z_z*f32(s_z/s_y)) * 2**15)`
  (`-` for Add, `+` for Sub), independent of `k`. That formula matched all 38
  Add/Sub builds on disk; plain `round`, float32 accumulation, or rounding
  per term each miss some of them by 1-2 LSB. When every zero point is 0, the
  offset is 0 and nothing is written. For Add it is also exactly 0 when
  calibration gives `s_y = s_x + s_z` with all zero points at 128; with the
  offset 0, its lanes can't be located, so a nonzero target is refused.

Mul and Div have no scale-dependent zero-point term: native pairs with nonzero
zero points (`div_s4 -> div_mix`, `mul_c2 <-> mul_c3`) match with the scale
values alone.

## Not calibration: input order and segment 0

Pulsar2 compiles the node as `op(x, z)` or `op(z, x)` per build. That swaps the
slot numbers `2 <-> 4` on `0x03d0` and `1 <-> 3` on `0x02b0` and permutes
segment 0, which is rebuild noise. An emitted model keeps its template's
order, which is self-consistent. Validation drops segment 0 and normalizes the
order (`normalized_records`).

## Relayout when the stream crosses a 32-byte boundary

Streams are zero-padded to 32 bytes (table key 2). If the re-encoded stream
pads differently, the blob is relaid out. Native Mul `[64,64,3,3]` builds show
Pulsar2 doing exactly this: segment 2 pads to 1312 in one build and 1344 in
another. The emitter updates:

- the segment's key 2 and key 5, and key 3 (start word) of every later segment;
- the FlatBuffers byte-vector length (`header - 12`) and the total segment word
  count (`header - 8`);
- every header uoffset, and the root table's negative soffset, whose target
  lies past the segment;
- the MCode initializer's `dims`.

Tail tables are relative and move as one block. Emitting the 1344-byte
calibration from the 1312-byte template, and the reverse, reproduced the native
blob byte for byte outside segment 0.

## Validated

**Every same-shape (op, shape) in the ResNet18 training step.** Three single-node
outliers are left out: Add `[1,1]` and Mul `[1024,9,3136]`/`[16,64,112,112]`.
Each (op, shape) has two zero-point classes, `x0,y0,z0` and `x128,y128,z128`;
Div uses `x128,y128,z0` because the step's denominators (`sqrt(v)+eps`) are
positive. Each class got a template and a held-out Pulsar2 build at another
calibration, with ranges `[1.3, 0.7]` against `[2.1, 0.9]` (`[0.37, 0.29]` or
`[3.3, 1.1]` where MinMax landed a zero point on 127).

The calibration data is mirrored (`x = [u, -u]`), so every zero point lands
exactly on the class. For the Add/Sub 128 class, the x and z extremes sit at
different indices, so `s_y != s_x + s_z` and the template's zero-point offset is
nonzero. Emission was checked in both directions (template -> held-out and
held-out -> template):

| op | (shape, class) templates | directions | records + `npu_params` exact | byte-exact outside segment 0 | via relayout |
| --- | --- | --- | --- | --- | --- |
| Add | 44 | 88 | 88 | 18 | 6 |
| Sub | 34 | 68 | 68 | 18 | 6 |
| Mul | 44 | 88 | 88 | 30 | 12 |
| Div | 36 | 72 | 72 | 20 | 3 |

"Records exact" means `npu_params` byte-identical, and every decompressed record
of segments 1.. identical after input-order normalization. Where the blob is not
byte-exact outside segment 0, one of two things applies:

- the held-out build chose the other input order, so the raw records differ
  and the LZ77 parse follows;
- Pulsar2 picked a different, equal-length back-reference (one offset byte),
  the codec's documented case.

The emitted layout is always self-consistent: `relayout_segment` re-decodes its
own output.

The earlier builds agree:

- `[64,64,3,3]` at zero points 0: 12 byte-exact, 36 record-exact;
- the register census and the `[1,16,8,8]` confirm builds: 4 byte-exact, 18
  record-exact;
- every other build on disk (big shapes, asymmetric zero points), retargeted to
  its own scales: 61 byte-identical.

The only refusals were the coinciding-values class.

`tests/test_axera_binary_op_scale_emit.py` re-checks all 158 committed oracles
with no Docker or device, and pins the Q15/shift/offset arithmetic to five
native builds (including `k = 3`, the stored `32768`, and negative offsets).

## Device

AX8850 in `axcl-vm`, device lock per run, card below 85C. Each case ran a native
template control first and a native health run after the emitted model. The
emitted models used calibrations nobody built natively (template scales
x 1.37 / 0.83 / 1.21 for x / z / y), and inputs spanned the representable range.
Expected output is numpy on the quantized inputs, requantized to y:

| op | shape | class | control max error | emitted max error | emitted mean | health |
| --- | --- | --- | --- | --- | --- | --- |
| Add | `[16,64,56,56]` | 128 | 1 LSB | 1 LSB | 0.000 LSB | ok |
| Add | `[16,128,28,28]` | 0 | 1 LSB | 1 LSB | 0.001 LSB | ok |
| Mul | `[16,128,28,28]` | 128 | 1 LSB | 1 LSB | 0.000 LSB | ok |
| Mul | `[16,1000]` | 0 | 0 | 0 | 0 | ok |
| Div | `[512,256,3,3]` | 128 (z0) | 1 LSB | 1 LSB | 0.000 LSB | ok |
| Sub | `[1,512]` | 128 | 1 LSB | 1 LSB | 0.002 LSB | ok |

The emitter only rewrites values Pulsar2 itself writes to the same registers. No
lane-0 or `W`-companion slot is touched.

## Coverage

`tinygrad_ax_backend.ElementwiseScaleEdit` now also serves these templates
(`TemplateEntry` kind `binary`). `extract_step_ops` records each binary node's
operand form (`same_shape`, `broadcast`, `const`). A rank-1 tensor (the
`[C]` parameter vectors) is served by the `[1, C]` template, because Pulsar2
cannot tile a standalone rank-1 Add (`TileFailException`); both have the same
contiguous layout. On the 1,104-node step:

| op | nodes | conditional (was 0) | refused: const operand | refused: broadcast | refused: no template |
| --- | --- | --- | --- | --- | --- |
| Add | 144 | 101 | 42 | 0 | 1 (`[1,1]`) |
| Sub | 46 | 42 | 2 | 2 | 0 |
| Mul | 397 | 63 | 205 | 127 | 2 |
| Div | 52 | 44 | 5 | 3 | 0 |
| total | 639 | 250 | 254 | 132 | 3 |

Step totals go from 82 covered / 74 conditional / 948 refused to **82 covered /
324 conditional / 698 refused**. "Conditional" means the node's calibrated zero
points must be one of the template classes.

## Refused (`ValueError`)

- a template or target whose formula values coincide (for example `s_x == s_z`,
  or Mul with `1/s_z == s_y/(s_x*s_z)`): Pulsar2 compiles a different, shorter
  program for that case;
- a Q15 header width change (the ratios round equal on one side only);
- a shift of `k = 15`;
- a zero-offset Add/Sub template for a target with a nonzero offset;
- a zero-point change: zero points are part of the template key. Add with
  asymmetric zero points is a structurally different program (64 more
  decompressed bytes in the census); Div's `z_y` sits in `0x1a90`/`0x1b10`.
  Retargeting zero points is future work;
- a constant or broadcast operand (254 + 132 step nodes): a different compiled
  program; no templates.
