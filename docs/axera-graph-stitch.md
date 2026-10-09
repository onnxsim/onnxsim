# Stitching a fused op chain from standalone programs (AX650)

`docs/axera-compose.md` found that Pulsar2 does not extend a chain's MCode op by op: it
rewrites the whole program. This page shows that the rewrite is regular enough to
reproduce. A fused graph's compiled model can be built from one compiled *standalone*
program per op (a one-op graph at the op's shape) plus the fused graph's scales and zero
points, without running Pulsar2's compiler on the fused graph.

- Code: `scripts/axera/graph_stitch.py` (`stitch`, `stitch_model`).
- Tests (no device): `tests/test_axera_graph_stitch.py`.
- Fixtures: `scripts/axera/fixtures/graph_stitch/`. `index.json` records, for each graph,
  its op chain, wiring, component programs, native fused build, fused scales and zero
  points, and constant values. `python scripts/axera/graph_stitch.py` stitches every
  committed graph and prints its record counts.

## Graphs

All builds are Pulsar2 7.0-lite for AX650 with MinMax calibration, at two calibrations
per graph (`pm1`, `pm4`: input ranges of about ±1 and ±4; positive only where a Sqrt
reads the input directly).

| graph | ops | standalone programs |
| --- | --- | --- |
| `silu` | `x * Sigmoid(x)`, `[1,64]` | Sigmoid, Mul |
| `sqrt_mul` | `x * Sqrt(x)`, `[1,64]` | Sqrt, Mul |
| `sig_add` | `x + Sigmoid(x)`, `[1,64]` | Sigmoid, Add |
| `mul_sig` | `Sigmoid(a * b)`, `[1,64]` | Mul, Sigmoid |
| `rmsnorm` | `x / Sqrt(ReduceMean(x * x) + eps) * gain`, `[1,64]` | Mul, ReduceMean, Add, Sqrt, Div, Mul |
| `attention` | `Softmax(q @ kt) @ v`, `q [8,64]`, `kt [64,8]`, `v [8,64]` | MatMul, Softmax, MatMul |

Each stitched model equals the native fused build in:

- every decompressed record of all five segments, except that segment 0's slot table
  (its four waits for the other engines' final signals) holds the same records in
  another order. That order differs between two native builds of one graph, so it is
  compared as a set;
- `npu_params`;
- the blob's FlatBuffer fields, header and tail;
- the model proto outside the MCode bytes (node, IO, `value_info`, metadata).

## Method

A segment is an LZ77 stream over 8-byte register records (`short_unit_codec`). Segment 2
drives the main engine, segments 0 and 1 the two matrix engines, and segments 3 and 4
the two copy engines.

1. **Split jobs at `a9` launches.** Each job of a standalone program gets a role from
   its register state at launch: PARAM (loads a quantization word from `npu_params`),
   QUANT (reads a graph input), DEQUANT (writes the graph output), or CORE (the op).
   The fused program keeps every CORE job, the PARAM+QUANT pair of each graph input at
   its first consumer, and the PARAM+DEQUANT pair of the graph output. The jobs that
   convert an intermediate tensor are dropped.
2. **Replay register state.** Pulsar2 writes only the registers whose value differs
   from the running state, in one fixed register order. The stitcher replays each op's
   own state job by job with the fused calibration and addresses substituted, and emits
   each kept job as the difference to the fused running state. The register order is
   the merge of every source job's own write order.
3. **Restore live registers.** A kept job also rewrites a register that an earlier op
   left at another value, if the register is live in that job. A register's gates are
   the conditions that hold in every standalone job that writes it: bits 8 to 11 of the
   job's `0x0150` select word, and "mode register g is nonzero". The register is
   restored when all its gates are on.
4. **Allocate scratch.** Buffers are bump-allocated from `0x2f7000` in op order, with
   each op's own address order and buffer sizes. A tensor already in a buffer is not
   quantized again.
5. **Load constants.** Initializer operands and the constants Add and MatMul
   synthesize live in `npu_params`. A copy-engine loader copies each into scratch. The
   loaders are re-emitted from the standalone loader of that engine with length, offset
   and destination replaced. Initializer bytes are copied from the standalone program;
   synthesized constants are recomputed from the fused calibration.
6. **Number the sync records.** `a2` records signal and wait between engines. Each
   job's dependencies come from the standalone program's own waits and from the tensors
   it reads; the fused waits and signal numbers follow from them.
7. **Rebuild the blob.** The MCode blob is an ordinary FlatBuffer. `parse_blob` and
   `build_blob` reproduce all 42 committed blobs byte for byte, so the fused blob is
   built from the fused IO list and segments.

## Derived versus learned

"Derived" rules hold in the standalone programs and are checked there: `Program`
validates them on load, and stitching a one-op graph reproduces each of the 30 committed
standalone programs (one exception, below). "Learned" rules cannot be read off a
standalone program. Each was taken from one native fused graph, and both calibrations of
that graph agree.

| rule | status | source |
| --- | --- | --- |
| Job roles, IO slot numbers, PARAM offsets, `a7` head | derived | every standalone program |
| Calibration formulas (Sigmoid table; Sqrt, Mul, ReduceMean, Div lanes and zero points; Softmax lanes and output data type; QUANT/DEQUANT) | derived | every standalone program |
| Add and MatMul `npu_params` constants | derived | standalone Add and MatMul builds |
| Register order, delta encoding | derived | every standalone program |
| Liveness gates outside the two groups below | derived | every standalone program |
| Scratch sizes, loader fields, matrix-engine operand offset | derived | standalone programs with those engines |
| What a job waits for | derived | every standalone program |
| Segment compression, FlatBuffer layout, model wrapper | derived | all 42 committed blobs |
| Mode-register group is always live | learned | `silu` |
| One gate set per input-port block | learned | `rmsnorm` |
| `npu_params` order: initializers, then synthesized constants | learned | `rmsnorm` |
| Loaders alternate copy engines 4, 3, 4, ... across ops | learned | `rmsnorm` |
| A job's waits precede its register writes | learned | `rmsnorm` |
| The main engine does not repeat a wait | learned | `attention` |
| Graph-input jobs run before the first core job | learned | `attention` |
| Engine placement of each MatMul (`place`) | learned | `attention` |
| The main-engine transpose job (`MAIN_TRANSPOSE_JOB`) | learned | `attention` |
| Where the output's PARAM job runs (`wiring["output_param"]`) | learned, not predicted | `rmsnorm` at `[1,576]` |
| Wide ReduceMean: single-precision lane, `zp_x * min(n, 256)`, packed pad lanes | learned | ReduceMean at 64 to 2048, `rmsnorm` at `[1,576]` |
| The pad group `0x0cd0..0x0da0` is live only while `0x0c20` is nonzero | learned | ReduceMean at 288 and 384 |

The transpose job is the largest learned item. In the fused attention block, the second
MatMul's operand transpose runs as a 30-record main-engine job. No standalone program
has that job: the standalone MatMul runs its transpose on copy engine 3. The job's
register list, record order and four constants (`0x0280`, `0x0290`, `0x0c10`, `0x0c20`)
are copied from the native build. Only the operand addresses and the permutation words
come from the standalone copy-engine job.

Three structural choices are assumptions that no build contradicts and none isolates:
scratch buffers, PARAM words and each engine's sub-programs are numbered in op order.

## Width 576

Two graphs were also built at `[1,576]` (`docs/axera-width-and-linear-emit.md`, fixtures
in `scripts/axera/fixtures/width_retarget/`). SiLU stitches from the `[1,576]` Sigmoid
and Mul programs with no new rule. RMSNorm needed the last three rows of the table:

- **Output PARAM placement.** The native `[1,576]` RMSNorm runs the graph output's PARAM
  job just before ReduceMean's core job; the `[1,64]` one runs it just before the DEQUANT
  job. Standalone builds differ the same way (Sigmoid at 512 and ReduceMean at 512 and
  576 run it before the core, every other width after), and nothing found predicts it.
  It is a wiring input: `"output_param": k` (before the first core job of op `k`) or
  `"late"`. A job that moves is emitted from the register state its op has at that job
  in the standalone order. Without the placement the stitched RMSNorm has 544 main-engine
  records instead of 532.
- **Wide ReduceMean calibration.** The lane is `float32(float32(s_x / s_y) / n)` (the
  float64 form is one ulp off on the native `[1,384]` build and equal on every `[1,64]`
  one); the accumulator zero point is `zp_x * min(n, 256)`; a padded core holds `zp_x` in
  all four bytes of `0x0d30..0x0da0`.
- **Pad group liveness.** An exception to "the mode group is always live": the pad group
  is restored only while `0x0c20` is nonzero.

With them RMSNorm at `[1,576]` stitched from native components (built at `pm1`) equals
the native fused build at both calibrations: records 8 / 4 / 532 / 52 / 56.

## Record counts

Decompressed records per segment, identical for stitched and native builds and for both
calibrations (except the `mul_sig` blob size):

| graph | matrix 0 | matrix 1 | main | copy 3 | copy 4 | main jobs | restored registers | `npu_params` bytes | blob bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `silu` | 8 | 4 | 452 | 4 | 4 | 6 | 7 | 40 | 3328 |
| `sqrt_mul` | 8 | 4 | 292 | 4 | 4 | 6 | 0 | 40 | 2208 |
| `sig_add` | 8 | 4 | 492 | 40 | 44 | 6 | 0 | 44 | 4104 |
| `mul_sig` | 8 | 4 | 492 | 4 | 4 | 8 | 10 | 60 | 3544 (pm1), 3320 (pm4) |
| `rmsnorm` | 8 | 4 | 524 | 52 | 56 | 10 | 37 | 109 | 3816 |
| `attention` | 100 | 100 | 608 | 68 | 44 | 11 | 18 | 720 | 5536 |

"Restored registers" counts the writes added by step 3 at the `pm1` calibration.

## Offline checks

`tests/test_axera_graph_stitch.py` runs without a device or Docker:

- 12 stitched models (6 graphs, 2 calibrations) equal their native builds as described
  above.
- Standalone sources built at another calibration give the same models: 16 two-op
  cases (sources `pm1`, `pm4` or `asym` for each fused calibration) and both swaps for
  RMSNorm and attention. The standalone programs contribute structure only.
- A one-op graph stitched from each of the 30 standalone programs reproduces that
  program's segments, `npu_params` and model. The exception is the `asym` Add build: its
  two `npu_params` scale words are in the other order.
- SiLU and RMSNorm at `[1,576]` (two calibrations each) equal their native builds, and
  the ReduceMean lane is pinned on the `[1,384]` build.
- The refused cases below raise.

## Device results

AX8850, AXCL V3.6.5, 2026-10-09. Stitched model versus native fused build: the outputs
were bit-identical in every case.

| graph | values compared per calibration |
| --- | --- |
| `silu` | 9,600 |
| `sqrt_mul` | 6,400 |
| `sig_add` | 6,400 |
| `mul_sig` | 6,400 |
| `rmsnorm` | 9,600 |
| `attention` | 40,960 |

The large-program Neg retarget (`docs/axera-misc-op-record-emit.md`) was run the same
way at `[1,64]`: `pm4` to `asym` and `asym` to `pm4`, 6,400 values in each direction,
bit-identical to the native build of the target calibration.

AX8850, AXCL V3.6.5, 2026-10-10, at `[1,576]`. Stitched model versus native fused
build: the outputs were bit-identical.

| graph | values compared |
| --- | --- |
| `silu` at `[1,576]` | 34,560 per calibration |
| `rmsnorm` at `[1,576]`, from native components | 34,560 |

Calibrate + stitch without Pulsar2's quant json was run the same day
(`docs/axera-pulsar-free-calibration.md`).

## Limits

- **Calibration is an input.** The stitcher replaces the compiler, not the quantizer.
  The fused scales and zero points come from the fused build's `quant_axmodel.json`, or
  from `pulsar_free_calibration` (float graph plus samples, no Pulsar2; about 8% of its
  scales are one float32 ulp from Pulsar2's).
- **One shape per program.** A standalone program serves the exact shape it was built
  at. `width_retarget` moves a `[1,64]` program of seven ops to other widths and emits
  ReduceMean at the measured widths (`docs/axera-width-and-linear-emit.md`).
- **The output PARAM placement is not predicted.** At `[1,576]` the caller has to give
  it (`output_param`); the value used here was read off the native build.
- **Sqrt and Div zero points.** A Sqrt with a nonzero input or output zero point and a
  Div with a nonzero divisor zero point raise `NotImplementedError`. No standalone build
  has one, so their registers are unknown.
- **Unsupported wirings** raise `NotImplementedError`: an op outside `SUPPORTED_OPS`
  (Sigmoid, Sqrt, Mul, Add, ReduceMean, Div, Softmax, MatMul); two ops on one matrix
  engine; a copy-engine sub-program that would need a register restore; a sub-program
  placed on an engine of another kind; several MatMuls without an explicit placement.
- **Rules resting on a single native build.** Every learned rule in the table comes from
  one fused graph. The placement and the transpose job were seen only in the attention
  block at these shapes; another graph with two MatMuls may need other values. A
  placement that no rule refuses is accepted, but only the native build's was verified.
- **Add operand order.** Add's two scale words are written in ONNX input order. That
  holds in `sig_add`, `rmsnorm` and four of five standalone Add builds. The `asym`
  standalone Add build has them swapped, and nothing here predicts when.
- **Initializer bytes are copied.** The quantized bytes of an initializer operand come
  from the standalone program, so that program must have been built with the same
  constant. `consts[...]["values"]` checks it.
- **Input slot order.** The compiled input slot order is a per-build permutation (the
  two attention builds differ). The caller chooses it in `wiring["inputs"]`.
