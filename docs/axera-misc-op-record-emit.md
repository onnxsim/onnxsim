# Record-level emission for ReduceSum, Greater→Cast, Sqrt [512,512,3,3] and the tail ops

This page covers the ResNet18 training step's (`t6-r18fold/step.onnx`) non-memory ops
that no emitter handled yet. All of the work was offline: existing compiler builds were
decompressed with `short_unit_codec` (#1850), their records were compared, and the
comparison was checked against held-out builds. No new Pulsar2 builds and no device runs
were used.

- Code: `scripts/axera/misc_op_record_emit.py`.
- Tests (no device): `tests/test_axera_misc_op_record_emit.py`.
- Fixtures: `scripts/axera/fixtures/misc_op_record_emit/` (the index, the held-out
  calibrations and the Sqrt builds), plus the existing `reducesum_decode/` and
  `teng_register_census/` builds.

## Method

The method is the one from `docs/axera-reshape-decompressed.md` (#1862):

1. Decompress every segment.
2. Group builds by record structure, meaning the `(verb, register)` sequence.
3. Within a group, sort each varying record into either a shape record or a
   calibration record.

For these ops, every build with a different shape turned out to have its own record
structure. The 1×N Greater→Cast sweep does share structures, but its SRAM addresses
switch regime inside a group, so no shape polynomial fits. That leaves the emitters with
exact-shape templates plus calibration edits. Calibration edits touch whole records only.

## ReduceSum

Each tile of a compiled ReduceSum runs three stages. Each stage writes two zero-point
registers and then one run of 8 scale lanes:

| stage | `0x1a90` | `0x1b10` | lanes (`0x0f50..0x0fc0`) |
| --- | --- | --- | --- |
| dequantize | `0` | `zp_x` | `1/s_x` |
| requantize | `zp_x * 2**k` | `zp_y` | `s_x/s_y` |
| output | `zp_y` | `0` | `s_y` |

- **Lane values.** Each lane value is float32 scales combined in float64, then rounded
  to float32.
- **The `2**k` factor.** It depends only on shape. Measured values:
  - `2**4` for 16-row reductions;
  - `2**6` for the census `[1,1,64,3136]` axis 2;
  - `2**8` for `[16,1000]` axis 1 and for `[16,1,64,3136]` axes (0,3).

  The emitter reads it from the template, as the template's own value divided by the
  template's `zp_x`.
- **Omitted writes.** Pulsar2 omits a `0x1b10` write when the register already holds the
  value.
  - `reducesum_s1` has `zp_x = zp_y = 128`. It has no requantize `0x1b10` write, and it
    writes `0x1b10 = 0` after the output stage's `0x1a90`.
  - `reducesum_asym` has `zp_x = 64` and `zp_y = 0`, and is the other way round.
  - An omitted write goes directly after its stage's `0x1a90` record.
- **Validation.** The retarget turns each of `reducesum_s1`, `_s4` and `_asym` into the
  other two record-for-record, in all 6 directions. Each of the 5 committed templates
  also reproduces itself exactly from its own calibration.
- **Tiled programs.** Tiled builds (`[16,1,64,3136]` axes (0,3): 4 tiles) repeat the
  three stages per tile, and the same rule holds.

Refused (`ValueError`):

- `zp_x = 0`, because an omitted dequantize `0x1b10` write was never observed;
- a `0x1a90` write that would equal the register's current value;
- a segment that doesn't end in the `0xa2` terminator plus 0 to 3 zero pad records.

**Record-count changes (added after the step-shape builds).** With `zp_y = 0` the elision
rule removes records overall: the output stage's `0x1b10 = 0` write goes (the requantize
stage already left 0 there), and on a tiled program the next tile's requantize
`0x1b10 = zp_y` write goes too. `[16,1000]` axes (0,1) loses 1 record and
`[16,1,128,784]` axes (0,3) (2 tiles) loses 4. Pulsar2 then re-pads the segment: every
decompressed segment is a whole number of 4-record (32-byte) groups, with all-zero
records after the `0xa2` terminator (0 to 3 of them in every one of 20 ReduceSum builds).
The emitter strips the pad, applies the same state machine, and re-pads. Both pairs (step
calibration ↔ `zp_y = 0`) reproduce the other build record for record in both
directions.

**Packed `zp_x` lanes.** Multi-axis programs (`[16,1000]` axes (0,1), `[16,1,256,196]`
and `[16,1,512,49]` axes (0,3)) also write `zp_x * 0x01010101` (the zero point in all
four bytes) on eight lanes `0x0d30..0x0da0`. The emitter rewrites them with the zero
points, and refuses if those lanes don't hold the template's own packed `zp_x`.

## Sqrt [512,512,3,3]

#1840 left this shape out because only 1 of 3 held-out builds matched its patch of the
compressed bytes. Decompressed, the four builds (`_t`, `_h1`, `_h2`, `_h3`) share one
record structure of 2000 segment-2 records, and only the scale lanes differ:

- `1/s_x` and `s_x` on `0x0f50..`;
- `s_y` on `0x0fd0..` and `0x0f50..`.

The rewrite of those lanes matches all 12 ordered pairs record-for-record. The old miss
was the LZ77 re-encoding, not the program. Sqrt is now fully emitted for the step:
42/42 nodes, 39 from `elementwise_scale_emit` and 3 from this module.

## Greater→Cast

Greater→Cast is not quantized. `gtcast_s1` and `gtcast_asym` are record-identical except
segment 0's slot table (records 2 to 5, a per-build permutation that is also present
across rebuilds of the same model). So a build at the exact shape is the complete
program, and `emit_model("GreaterCast:<shape>")` returns it as is.

No step shape has a build yet. The committed template is the census shape
`[1,64,56,56]`, and the `pgtcast_1x*` sweep only goes up to 131072 elements.

## Coverage on the real step

Run `python scripts/axera/misc_op_record_emit.py step.onnx` for the report.

| op | nodes | covered now | still needs |
| --- | --- | --- | --- |
| Sqrt | 42 | 42 (39 via `elementwise_scale_emit`, 3 here) | -- |
| ReduceSum | 44 | 44 (one through a same-bytes equivalent, below) | -- |
| Greater | 18 | 18 | -- |
| Cast | 19 | 19 (each one follows a Greater or Less) | -- |
| Less | 1 | 1 | -- |

Tail ops (batch F). Each was built at its step shape at two calibrations
(`fixtures/misc_op_step_templates/`). Each template reproduces the other build record for
record in both directions, and reproduces itself from its own calibration.

| op (nodes) | template key | calibration records | status |
| --- | --- | --- | --- |
| Softmax (3) | `Softmax:16x1000:axis1` | lanes `1/s_x`, `s_x`, `1/s_y`, `s_y`; one `0x1b10 = zp_x` before the first run | conditional: `zp_x != 0`, `zp_y = 0` (a Softmax output always calibrates to 0). The held-out pair also moves `zp_x` 127 → 126 |
| Softmax (not in the step) | `Softmax:1x64:axis1` | the same lanes and `0x1b10 = zp_x` write | same conditions. Five held-out builds cover `zp_x` 64, 128 and 232, device-checked on an AX8850 (see "Softmax at an LLM shape") |
| Log (2) | `Log:16x1000` | lane `1/s_x`; a 258-entry u8 lookup table, two u16 entries per record at `0x1050..0x1850`: `clip(rint(log((q − zp_x)·s_x)/s_y) + zp_y, 0, 255)` for `q` = 0..255 (`log 0` → 0), then entry 255 again and 0. The table is exact on both builds. `0x1850` is written after an unrelated `0x1860..0x1a50` block, so the emitter finds table records by register | conditional: zero points fixed at the template's (0, 255). Log's input is a Softmax output, so `zp_x = 0` |
| MaxPool (1) | `MaxPool:16x64x112x112:k3x3:s2x2:p1,1,1,1` | lanes `1/s_x`, `s_x` (`s_y = s_x`) | conditional: zero points fixed (0; input is a Relu) |
| ReduceMean (1) | `ReduceMean:16x512x7x7:axes2,3:k1` | lanes `1/s_x`, `s_x/(s_y·N)` (`N` = 49 reduced elements), `s_y` | conditional: zero points fixed (0; input is a Relu). A template with nonzero `zp_x` can move both (see "Sigmoid, Mul, Add, Div and ReduceMean at `[1, 64]`"); this one cannot, as for ReduceSum |
| Neg (2) | `Neg:1x1` (large program) and `Neg:1x1:small` | lanes `1/s_x`, `s_x` (`s_y = s_x`); zero points on `0x1a90`/`0x1ad0`/`0x1b10` with write omission (see below) | conditional: any `zp_x` (`zp_y = 255 - zp_x`) and scale (`s_y = s_x`); the scale picks the program |
| Squeeze (1) | `[16,512,1,1]→[16,512]` | a standalone Squeeze hits the scheduler's ZeroDivisionError (`AX650_CONFIRMED_BROKEN_OPS`), but Squeeze → Relu compiles record for record to the Reshape → Relu program, so the node takes the Reshape step template `16x512x1x1->16x512` (`reshape_step_templates/`) | conditional: nonzero zero point, as every Reshape step template |

## Builds still needed (run one batch at a time)

Each build is a single-op model at the step shape, using the step's calibration class
(random normal inputs give `zp_x ≈ 127..128` and `zp_y ∉ {zp_x, 0}`).

Adding a second calibration for the first build of each new group gives a held-out check
of the lane and zero-point rule at that shape. Batches marked (+held-out) should include
one.

- **A: Greater→Cast `x > 0`** (5 builds; covers 17 Greater and 17 Cast). One build per
  shape is complete.
  - `[16,64,56,56]`
  - `[16,128,28,28]`
  - `[16,256,14,14]`
  - `[16,512,7,7]`
  - `[16,64,112,112]`
- **B: MaxPool backward masks** (2 builds; covers 1 Greater, 1 Less and 2 Cast). Both
  compare `[1024,9,3136]` against a live broadcast `[1024,1,3136]` input:
  - Greater→Cast
  - Less→Cast
- **C1: ReduceSum axis 0 on `[16,1,C,K]`, keepdims 0** (4 builds, 6 nodes, +held-out):
  - `C=64`: `K=147`
  - `C=128`: `K=64`, `576`, `1152`
- **C2: the same, larger `C`** (6 builds, 10 nodes):
  - `C=256`: `K=128`, `1152`, `2304`
  - `C=512`: `K=256`, `2304`, `4608`
- **D: ReduceSum axes (0,3), keepdims 0** (4 builds, 13 nodes, +held-out):
  - `[16,1,128,784]`
  - `[16,1,256,196]`
  - `[16,1,512,49]`
  - `[16,1,64,12544]`
- **E: ReduceSum, the rest** (3 builds, 4 nodes):
  - `[16,1000]` axes (0,1) k1
  - `[1024,9,3136]` axis 1 k1
  - `[1024,9,12544]` axis 1 k0
- **F: tail ops** (5 builds, two calibrations each to locate lanes):
  - Softmax `[16,1000]`
  - Log `[16,1000]`
  - Neg `[1,1]`
  - MaxPool `[16,64,112,112]`
  - ReduceMean `[16,512,7,7]`

After A to E, ReduceSum, Greater, Less and Cast reach 44/44, 18/18, 1/1 and 19/19,
provided each new template passes the self-identity check (`test_own_calibration_is_identity`
pattern).

## Step-shape builds (batches A to E)

Built from `step.onnx` one at a time (`fixtures/misc_op_step_templates/`), registered in
`index.json` only after the checks pass:

- **Greater→Cast and Less→Cast** (batches A and B, 7 builds): registered as built, since
  they are not quantized.
- **ReduceSum** (batches C to E, 16 templates): each passes the self-identity check
  (retarget away and back). Four have a second build at another calibration and
  reproduce it record for record in both directions:
  - `[16,1,512,4608]` axis 0 and `[16,1,64,147]` axis 0 (both `zp_y != 0`);
  - `[16,1000]` axes (0,1) k1 and `[16,1,128,784]` axes (0,3), each paired with a
    `zp_y = 0` build that changes the record count. Before the re-pad rule above,
    these two were refused.
- **Not buildable as written:** `[16,1,64,12544]` axes (0,3) k0 (1 node,
  `ReduceSum_474`, the stem's bias gradient). Pulsar2 fails with
  `TileFailException: AxQuantizedReduceSum, Can not tile` (a 12.8 MB U8 input
  against a 3 MB memory limit), standalone and also cut from the step with its
  real neighbours (`Reshape_465 -> ReduceSum_474 -> Reshape_475`), so the step
  itself can't compile it either. The same reduction over the same contiguous
  bytes, `[16,64,112,112]` axes (0,2,3) k0 -> `[64]` (output bytes equal the
  `[1,64]` result), does compile.
  `REDUCESUM_EQUIVALENTS` maps the node's key to that template
  (`ReduceSum:16x64x112x112:axes0,2,3:k0`), which reproduces a second native
  build record for record in both directions. A split into two compilable
  reductions (axis 3, then axis 0) also builds, but adds an intermediate u8
  rounding, so it isn't used. A held-out build calibrated to `zp_y = 0` was
  refused by the existing rule (a later tile's `0x1a90 = 0` would repeat the
  register value, never observed), so this template is conditional on
  `zp_y != 0` like the other multi-tile ReduceSums.

## Neg: two programs picked by the scale

The four batch-F builds that looked like four layouts are two programs, each
with zero-point write omission on top. A sweep of 20+ builds of `Neg [1,1]`
(`fixtures/misc_op_neg/`, `sweep.json`) at pinned calibrations shows:

- Neg always calibrates to `s_y = s_x` and `zp_y = 255 - zp_x`, so in u8 it is
  `q_y = 255 - q_x`; the emitter refuses any other calibration.
- **The scale alone picks the program.** Every build with float32 `s < 1/64` compiles
  to the small program (268 records in segment 2) and every build with
  `s >= 1/64` to the large one (292/296 records, an extra compute block).
  Zero points 0 and 255 occur on both sides, and `zp_x` 42..246 at fixed scale
  never switch programs. At the boundary, `s = 0.01562` is small and
  `s = 0.015625` (exactly `1/64`) and `0.01563` are large
  (`misc.neg_program`).
- **Within a program**, a calibration change touches the lanes `1/s_x` and
  `s_x` (the small program writes `s_x` on 16 lanes, `0x0f50..0x1040`) and the
  zero-point records on `0x1a90`, `0x1ad0` and `0x1b10`, each `zp_x`, `zp_y`
  or 0. A write outside a register-block dump is omitted when the register
  already holds the value (registers start at 0); block dumps
  (`0x1b00..0x1b60`, consecutive registers) are always written in full. So
  `zp_x = 0` or `zp_y = 0` targets drop records, and the segment is re-padded
  to whole 4-record groups. A template must be a build with no omitted write
  (both zero points nonzero); `Neg:1x1` is `neg_L128` and `Neg:1x1:small` is
  `neg_z128`.
- Every template → every other build of its program reproduces record for
  record, including the step's own class: the step's Neg inputs are
  cross-entropy sums (`<= 0`), which calibrate to `zp_x = 255, zp_y = 0`
  (`step_neg__v4`, large program).

## Softmax at an LLM shape: `Softmax:1x64:axis1`

A last-axis Softmax on `x[1, 64]` (float32 I/O, Pulsar2 7.0-lite, MinMax over 16
samples) is one more template for the existing Softmax rules. Nothing in the emitter
changed. Six builds at different calibrations have the same 3360-byte MCode layout and
474 records in segment 2. They differ in 32 lane records (`1/s_x`, `s_x`, `1/s_y`, `s_y`
on `0x0f50..0x0fc0`) and the `0x1b10 = zp_x` write. Rebuilding one calibration twice
changes only segment 0's slot table, as elsewhere.

| build | input range | `s_x` | `zp_x` | `s_y` |
| --- | --- | --- | --- | --- |
| template | U[−4, 4] | 0.031351 | 128 | 0.00070684 |
| `softmax_1x64_pm1` | U[−1, 1] | 0.0078377 | 128 | 0.00015480 |
| `softmax_1x64_pm8` | U[−8, 8] | 0.062702 | 128 | 0.0016829 |
| `softmax_1x64_pm0p5` | U[−0.5, 0.5] | 0.0039189 | 128 | 0.00010082 |
| `softmax_1x64_asym` | U[−1, 3] | 0.015675 | 64 | 0.00030302 |
| `softmax_1x64_asym2` | U[−5, 0.5] | 0.021554 | 232 | 0.00044158 |

Every held-out build retargets to every other one record for record (20 ordered pairs,
plus `emit_spec` from the template with `axis=-1`).

**On the device** (AX8850, AXCL V3.6.5, 2026-10-09): the template retargeted to each of
the five held-out calibrations gives output bit-identical to the native build, 7,680
values per build, 0 mismatches. The card's health check read 0.0 before and after.

### Calibration rules

* Input: the range is widened to include 0, `s_x = (hi − lo)/255`, `zp_x = rint(−lo/s_x)`.
  Bit-exact in float32 on the five held-out builds (`minmax_input_calibration`).
* Output: `s_y` is the largest probability seen in calibration divided by 255, with
  `zp_y = 0` (`softmax_output_scale`). It is not `1/255`. It matches the builds to 1e-7
  relative. A row whose largest probability exceeds the calibrated one saturates at code
  255: 4 of 300 random rows at U[−4, 4], 13 of 120 at U[−0.5, 0.5].

### Softmax semantics

`softmax_codes` models the compiled op:

    q    = clip(rint(x / s_x) + zp_x, 0, 255)
    d    = max(q) − q                          # per row
    p    = exp(−d · s_x) / Σ exp(−d · s_x)
    code = clip(rint(p / s_y), 0, 255)         # the device returns code · s_y

Against 46,080 device codes over the six calibrations, every code is within 1 of the
model and 98.8% are equal (97.3% to 99.8% per build). The misses are all within about
0.12 of a rounding tie, in both directions, so they are arithmetic precision and not
structure. Evidence for the distance form: 40 rows, each shifted by ±16 and ±32 input
codes, gave bit-identical device outputs. The quantized row is not renormalized (row
sums 0.88 to 1.005).

Not determined: the arithmetic behind the last 1.2% (a per-row scale factor explains
465 of 500 rows, so it is not only the reciprocal of the sum; a table estimated from
the capture matches `exp(−d · s_x)` to 0.15%, the noise floor of that estimate). Tie
rounding is therefore unverified. Only `[1, 64]` was measured.

A subset of the capture is committed (`fixtures/misc_op_record_emit/
softmax_1x64_device.npz`: 20 rows per calibration and 6 shift probes) and checked by
`tests/test_axera_misc_op_record_emit.py`.

## Sigmoid, Mul, Add, Div and ReduceMean at `[1, 64]`

Five more standalone templates at the same LLM-like shape (float32 I/O, Pulsar2
7.0-lite, MinMax over 16 samples per input, `default_rng(0)`): `Sigmoid:1x64`,
`Mul:1x64`, `Add:1x64`, `Div:1x64` and `ReduceMean:1x64:axes1:k1` (`axes=-1`,
`keepdims=1`). Each has three builds. `pm4` (U[−4, 4]) is the template; `pm1`
(U[−1, 1]) and `asym` (U[−1, 3]) are held out. Div's divisor is drawn from U[0.5, 4],
U[0.5, 1.5] and U[1, 3]. Two-input ops use the keys `a`, `b`, `y`.

| op | build | scales (`x` or `a`, `b`; `y`) | zero points |
| --- | --- | --- | --- |
| Sigmoid | template / pm1 / asym | 0.031351; 0.0038508 / 0.0078377; 0.0028661 / 0.015675; 0.0037352 | 128, 0 / 128, 0 / 64, 0 |
| Mul | template / pm1 / asym | 0.031351, 0.031252; 0.12116 / 0.0078377, 0.0078129; 0.0075727 / 0.015675, 0.015626; 0.044280 | 128, 128, 125 / 128, 128, 125 / 64, 64, 65 |
| Add | template / pm1 / asym | 0.031351, 0.031252; 0.060862 / 0.0078377, 0.0078129; 0.015216 / 0.015675, 0.015626; 0.030431 | 128, 128, 130 / 128, 128, 130 / 64, 64, 65 |
| Div | template / pm1 / asym | 0.031351, 0.015660; 0.056715 / 0.0078377, 0.0058748; 0.014594 / 0.015675, 0.011750; 0.014637 | 128, 0, 133 / 128, 0, 135 / 64, 0, 67 |
| ReduceMean | template / pm1 / asym | 0.031351; 0.0037486 / 0.0078377; 0.00093715 / 0.015675; 0.0051703 | 128, 85 / 128, 85 / 64, 0 |

Within one op the three builds have the same MCode size and, except ReduceMean's
`asym`, the same record layout. All calibration-dependent records are in segment 2.
Each lane run is eight records of one float32 (float32 scales combined in float64 and
rounded). The zero-point writes listed with a run are the `0x1a90`/`0x1ad0`/`0x1b10`
records between the previous run and that one. Every op starts with `0x1a90 = 0` and
`0x1ad0 = 0` (part of a register-block dump) before the first `0x1b10` write.

| op (segment 2 records) | lane runs in program order, with their zero-point writes | other calibration records |
| --- | --- | --- |
| Sigmoid (396) | `1/s_x` (`0x1b10 = zp_x`); `s_y` (`0x1b10 = 0`) | the 258-entry u8 table in Log's layout (`0x1050..0x1850`, two u16 per record): `clip(rint(σ((q − zp_x)·s_x)/s_y) + zp_y, 0, 255)` for `q` = 0..255, then entry 255 again and 0 (`sigmoid_table`) |
| Mul (312) | `1/s_a` (`0x1b10 = zp_a`); `1/s_b`; `s_y/(s_a·s_b)` on `0x0fd0..0x1040` (`0x1a90 = zp`, `0x1ad0 = zp`, `0x1b10 = zp_y`, with `zp = zp_a = zp_b`); `s_y` (`0x1a90 = zp_y`, `0x1ad0 = 0`, `0x1b10 = 0`) | none |
| Add (360) | `1/s_a` (`0x1b10 = zp`, `zp = zp_a = zp_b`); `1/s_b`; `s_y` (`0x1a90 = zp_y`, `0x1b10 = 0`) | an int32 offset on four lanes `0x1ef0..0x1f20`, right after `0x1ea0 = 15`: `trunc((zp_y − zp_a·r_a − zp_b·r_b)·2^15)` with `r = float32(s/s_y)` (`add_offset`). `npu_params` starts with the Q15 weights `round(s_a/s_y·2^15)`, `round(s_b/s_y·2^15)` (`add_q15`) |
| Div (308) | `1/s_a` (`0x1b10 = zp_a`); `1/s_b` (`0x1b10 = 0`, taken as `zp_b`); `s_a/(s_b·s_y)` (`0x1a90 = zp_a`, `0x1b10 = zp_y`); `s_y` (`0x1a90 = zp_y`, `0x1b10 = 0`) | none |
| ReduceMean (260, `asym` 259 + 1 pad) | `1/s_x` (`0x1b10 = zp_x`); `s_x/(s_y·64)` (`0x1a90 = zp_x·64`, `0x1b10 = zp_y`); `s_y` (`0x1a90 = zp_y`, `0x1b10 = 0`) | none |

The emitter finds every one of these by register and by lane-run order, not by record
index, and checks each record's old value against the template's calibration before
writing. For the four new ops the lane runs are matched in program order, so two
formulas may coincide (`s_a = s_b`).

**ReduceMean** has ReduceSum's three stages, with `0x1a90 = zp_x·N` in the requantize
stage. `retarget` now runs ReduceSum's zero-point model on it (same elision of a
`0x1b10` write whose value the register already holds, same re-padding). That
reproduces all six ordered pairs at `[1, 64]`, including `asym`, where `zp_y = 0`
elides the output stage's `0x1b10 = 0` write: the `s_y` run starts at record 238
instead of 239 and the segment ends in three pad records instead of two. A template
whose `zp_x` is 0 (the step's `16x512x7x7`) still refuses a zero-point change.

**Slot selects.** Add's and Div's `asym` builds differ from the template in four more
records: verb `0xa8` on `0x03d0`/`0x02b0`, holding slot ids 1..4. The two input stages
use slots (2, 1) then (4, 3) in one build and (4, 3) then (2, 1) in the other. These
ids pair with segment 0's slot table, which already differs between rebuilds of one
calibration. The emitter leaves them as the template has them, and the tests compare
Add and Div with the ids renumbered by first appearance (`slot_normalized_records`).
For Sigmoid, Mul and ReduceMean all six ordered pairs match under plain
`normalized_records`. Pairs between Add's or Div's template and `pm1` do too.

**On the device** (AX8850, AXCL V3.6.5, 2026-10-09; template `pm4` retargeted, 100
random inputs each):

| op | retargeted template vs native build | arithmetic model vs device |
| --- | --- | --- |
| Sigmoid | `pm1` and `asym` bit-identical (6,400 values each) | table lookup (`sigmoid_table`): 100% of 19,200 codes |
| Mul | `pm1` and `asym` bit-identical | product model: 100% of 19,200 codes |
| Add | `pm1` and `asym` bit-identical | float model: within 1 code, 99.9% equal |
| Div | `pm1` and `asym` bit-identical | float model: 100% |
| ReduceMean | `pm1` and `asym` bit-identical (200 values each) | not modelled. The native build is far from a float mean on random inputs (up to 189 output steps at `pm1`), most likely saturation: the output range comes from 16 calibration means. Not confirmed |

Add's and Div's retargeted templates keep the template's slot selects and still match
the native `asym` builds on the device.

**Refused (`ValueError`), because no build measured it:**

* Mul or Add with `zp_a != zp_b` (the builds cannot tell which record is which);
* Div with `zp_b != 0` (every divisor calibration here is positive);
* Sigmoid with `zp_y != 0`;
* for these four ops, a zero point that is 0 on one side of the retarget and nonzero
  on the other (a zero write may be elided), and any target where a zero-point write
  would repeat its register's value while the template's does not, e.g. Mul with
  `zp_y = zp_a`;
* an Add target whose Q15 weights differ from the template's (16879, 16826).
  `npu_params` was identical in the three builds, so its rewrite is unverified and the
  emitter does not attempt it. In practice the Add template serves targets with the
  template's `s_a/s_y` and `s_b/s_y`.

**Not tested:**

* ReduceMean with `zp_x = zp_y` follows ReduceSum's measured elision rule; no
  ReduceMean build has it.
* Which of Mul's `0x1a90`/`0x1ad0` and Add's `0x1b10` holds `zp_a` and which `zp_b`.
* Any shape other than `[1, 64]`, any ReduceMean axes other than the last, and inputs
  that are constants or shared (`a = b`).
* Add's rounding: the float model misses 0.1% of codes by 1.
