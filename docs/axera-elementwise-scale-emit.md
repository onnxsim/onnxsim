# Elementwise scale-retarget emitter

`scripts/axera/elementwise_scale_emit.py` rewrites a compiled standalone AX650
elementwise model for a new quantization scale without Pulsar2.
`scripts/axera/elementwise_scale_sweep.py` builds the validation references
with *controlled* calibration: every sample is `linspace(lo, hi)`, so MinMax
sees exactly `lo` and `hi`.

## What the scale depends on

At a fixed zero point, a standalone Relu's whole compiled model depends on the
calibration scale through float32 copies in the TENG segment (`segments()[2]`)
and nothing else:

- quant multiplier `f32(1 / s)`: one copy per output lane in the
  `0x0f50..0x0f80` register-write group (more copies in tiled programs);
- dequant multiplier `f32(s)`: one copy per lane in the second group.

`s` is Pulsar2's own float32 scale. Both formulas matched all 35 batch-1
builds bit for bit, including scales that differ in mantissa bytes, not only
exponent. `npu_params`, the node attributes and every other MCode byte stay
the same.

`minmax_params(lo, hi)` reproduces Pulsar2's MinMax `(scale, zero_point)` for
all 35 builds (the zero point uses the unrounded range, round-half-to-even).

## Coverage

Against the ResNet18 training step (1,104 nodes), for a new calibration at a
template's zero points:

| op | step nodes | covered nodes | shapes covered | zero-point classes | held-out byte-exact |
| --- | --- | --- | --- | --- | --- |
| Relu | 17 | 17 | 5/5 | x=y=0, x=y=128 | 25/25 |
| Sqrt | 42 | 39 | 16/17 (not `[512,512,3,3]`) | x=y=0 | 48/48 on committed shapes |
| Add, Sub, Mul, Div | 144, 46, 397, 52 | 0 | 0 | -- | refused (below) |

That is 56 of the step's 1,104 nodes. Everything else raises `ValueError`:
other ops, shapes and zero points, and Sqrt with equal input and output scales.

## Validated

### Relu: every shape in the ResNet18 training step

Templates at zero point 0 and 128 are committed for all five Relu shapes in the
step (`[16,64,56,56]`, `[16,128,28,28]`, `[16,256,14,14]`, `[16,512,7,7]`, and
the stem's `[16,64,112,112]`; 17 of the step's 1,104 nodes). Retargeting each
template to held-out, non-power-of-two scales reproduced the native Pulsar2
build byte for byte, outside the known 301-325 noise window:

| shapes | zero point | held-out calibrations | byte-exact |
| --- | --- | --- | --- |
| 5 real step shapes | 0 | 3 each | 15/15 |
| 5 real step shapes | 128 | 2 each | 10/10 |
| `[1,64,56,56]` | 128 | 6 | 6/6 |
| `[1,64,56,56]` | 0 | 1 | 1/1 |

Five more held-out builds used a symmetric range (`±0.685`) whose MinMax zero
point came out 127, not 128, through float rounding; the emitter correctly
refused them (zero point is the template key, not range symmetry).
`tests/test_axera_elementwise_scale_emit.py` re-checks one held-out native
build per template with no Docker or device.

On the AX8850 (`axcl-vm`, device lock per run, a native control run first and a
native health run after each emitted model), four emitted models matched
`numpy.maximum(x, 0)` within half an output LSB -- the expected rounding bound --
and every health run passed. Two of the four used calibrations Pulsar2 never
built (`[0, 2.9]` and `[-0.4, 0.4]`):

| shape | zero point | scale | max error | 1 LSB |
| --- | --- | --- | --- | --- |
| `[16,64,56,56]` | 0 | 0.0121569 | 0.0061 | 0.0122 |
| `[16,512,7,7]` | 128 | 0.0121569 | 0.0061 | 0.0122 |
| `[16,128,28,28]` | 0 | 0.0113725 | 0.0057 | 0.0114 |
| `[16,256,14,14]` | 128 | 0.0031373 | 0.0016 | 0.0031 |

An emitted model differs from a native build of the same calibration only
inside the noise window, whose bytes come from the template (itself a native
build), so these runs never put an untested byte pattern on the card. No lane-0
or `W`-companion value is patched away from what Pulsar2 itself writes there.

### Sqrt: 16 of 17 step shapes

Sqrt carries three float groups in the TENG segment, four lane copies each:
`f32(1/sx)`, `f32(sx)`, `f32(sy)` (`npu_params` doesn't move). When `sx == sy`
the third group merges away and the stream is 64 bytes shorter, so a template
with distinct scales serves the general case -- for a non-negative input
range `[0, hi]`, `sx = hi/255` and `sy = sqrt(hi)/255` differ unless `hi = 1`,
and the emitter refuses a target whose floats coincide. Sqrt's input and output
are non-negative, so zero point 0 is the only class needed.

Across all 17 distinct Sqrt shapes in the step (42 nodes: the Adam update's
`sqrt(v)` for every parameter), 49 of 51 held-out calibrations were
byte-exact. The two misses are both `[512,512,3,3]`, and they show the limit of
this approach (next section). Templates for the 16 shapes that were 3/3 clean
are committed (39 of the 42 Sqrt nodes); `[512,512,3,3]` is left out.

### Where the patched floats live

Each copy is found by value and must sit in whole groups of four lane copies
one record apart. Per template, copies in `V`/`W` register-write records (`R`)
vs compressed short units (`s`):

| templates | quant group(s) | dequant group |
| --- | --- | --- |
| Relu `[16,128,28,28]`, `[16,256,14,14]`, `[16,512,7,7]` | `RRRR` | `sRRR` |
| Relu `[16,64,56,56]`, `[16,64,112,112]` (tiled) | `RRRR` + `ssss` | `ssss` |

In a short unit a float is a fixed-width 4-byte literal run
(`[03][value32][tag][dist]`), so patching it in place never changes a length.

## Limit: in-place patching assumes the encoder's choices don't move

The emitter does not decompress and re-encode; it overwrites literal float bytes
in place. That is byte-exact only when Pulsar2's own encoder would make the same
literal/back-reference choices for the new value. In the two `[512,512,3,3]`
Sqrt misses it did not: for the new `sx` Pulsar2 wrote each copy as three
literal bytes plus a longer back-reference that absorbs the float's last byte
(`3b`), because that byte and what follows already occur earlier in the
stream. The segment content shrank by 4 bytes. Whether a given value triggers
this depends on the shape's whole stream, so no template is guaranteed for
every scale; the 74/76 held-out rate is the evidence, not a proof.

So, to answer #1836 directly: **this emitter does not handle the
variable-length compressed units.** It patches fixed-width float literals only,
and it refuses zero-point changes.

## Binary ops: refused

(Superseded: `binary_op_scale_emit.py` now retargets same-shape Add/Sub/Mul/Div
on the decompressed records; see `docs/axera-binary-op-scale-emit.md`. This
section records why in-place patching could not.)

Controlled builds of Add, Sub, Mul and Div were made at `[64,64,3,3]`, a real
step shape: a template and three held-out calibrations each, all zero points 0.
The `x`/`z` ranges had different ratios (e.g. `[0,1.3]/[0,0.7]` vs
`[0,2.1]/[0,0.9]`). They show that changing the scale *ratios* changes far more
than float literals:

| op | MCode regions differing from the template (outside 301-325) | of which plain 4-byte float swaps | `npu_params` |
| --- | --- | --- | --- |
| Add | 17, 47, 49 | 4, 4, 4 | differs |
| Sub | 44, 49, 53 | 4, 4, 8 | differs |
| Mul | 25, 34, 49 (one build 32 bytes longer) | 4, 4, 8 | same |
| Div | 45, 46, 48 | 0, 8, 9 | same |

The rest are changes of one to a few bytes:

- in the compressed register writes: literal runs becoming back-references
  and vice versa, and shifted distances;
- in segment-table header bytes, e.g. `7a` <-> `78` at offsets 204, 232, 252
  and 280.

The floats #1837 lists do occur: for Add, `f32(1/x)` and `f32(y)` each appear
as four lane copies. But, as #1839 found, `1/z` does not appear as a float
when its ratio to `x` differs. The ratio-dependent fields are exactly the
variable-length units this emitter cannot re-encode, so binary ops are refused
at every shape. A template would only serve targets with its exact scale
ratios. A training step calibrates each tensor independently, so its targets
won't share those ratios, and no such narrow class is shipped.

## Which fields are patched, and how

- **Patched in place:** only the scale floats of Relu and Sqrt, always as whole
  4-byte values. Some copies sit in `V`/`W` register-write records and some in
  the 4-byte literal runs of compressed short units (see the table above). No
  length ever changes.
- **Not patched (part of the template key instead):** zero points, and every
  field that depends on scale ratios (binary ops). These live in re-encoded
  variable-length units.

## Zero point: not handled

A zero-point change is refused. The zero point is written through the stream's
compressed register writes (#1836's blocker). A controlled zero-point sweep
(22 values at a fixed scale) shows these are LZ-style: a record's bytes are
either literal or copied from an earlier position, at a distance in 4-byte
units. For zero points 129, 191, 200 and 254 the stream is byte-identical
except for the single literal zero-point byte, but a value that already
occurs earlier (e.g. 100) is back-referenced instead, which changes lengths
and shifts later distances. Retargeting a zero point would need a full decoder,
plus a re-encoder that makes the same literal/back-reference choices as
Pulsar2. This emitter has neither.
