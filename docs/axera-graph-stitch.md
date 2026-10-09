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
| `attn16` | the same block at `q [16,32]`, `kt [32,16]`, `v [16,32]` | MatMul, Softmax, MatMul |
| `mm_chain` | `(a @ b) @ c`, `a [8,64]`, `b [64,8]`, `c [8,64]` | MatMul, MatMul |
| `mm_chain3` | `((a @ b) @ c) @ d`, `d [64,8]` | MatMul, MatMul, MatMul |
| `attn_proj` | `(Softmax(q @ kt) @ v) @ w`, `w` a constant `[64,64]` | MatMul, Softmax, MatMul, constant-weight MatMul |

The last four are described in [Several MatMuls](#several-matmuls).

Each stitched model equals the native fused build in:

- every decompressed record of all five segments, except that segment 0's slot table
  (its four waits for the other engines' final signals) holds the same records in
  another order. That order differs between two native builds of one graph, so it is
  compared as a set;
- `npu_params`;
- the blob's FlatBuffer fields, header and tail. In `mm_chain` at `pm1` and `attn_proj`
  at both calibrations the stream layout and the tail differ, because the other order
  of those four slot-table records compresses segment 0 to another length. With the
  native order of the four records (`with_slot_order`) they are equal too;
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
   `build_blob` reproduce all 64 committed blobs byte for byte, so the fused blob is
   built from the fused IO list and segments.

## Derived versus learned

"Derived" rules hold in the standalone programs and are checked there: `Program`
validates them on load, and stitching a one-op graph reproduces each of the 44 committed
standalone programs (one exception, below). "Learned" rules cannot be read off a
standalone program. Each was taken from one native fused graph, and both calibrations of
that graph agree. "Fitted" rules were adjusted until five native graphs with several
MatMuls all matched ([Several MatMuls](#several-matmuls)).

| rule | status | source |
| --- | --- | --- |
| Job roles, IO slot numbers, PARAM offsets, `a7` head | derived | every standalone program |
| Calibration formulas (Sigmoid table; Sqrt, Mul, ReduceMean, Div lanes and zero points; Softmax lanes and output data type; QUANT/DEQUANT) | derived | every standalone program |
| Add and MatMul `npu_params` constants | derived | standalone Add and MatMul builds |
| Register order, delta encoding | derived | every standalone program |
| Liveness gates outside the two groups below | derived | every standalone program |
| Scratch sizes, loader fields, matrix-engine operand offset | derived | standalone programs with those engines |
| What a job waits for | derived | every standalone program |
| Segment compression, FlatBuffer layout, model wrapper | derived | all 64 committed blobs |
| Constant-weight MatMul: weight-region loader, two-engine split, per-channel lanes | derived | standalone `c_mm_proj` |
| A matrix engine does not repeat a wait (its first sub-program's waits do not count) | derived | every standalone MatMul |
| Two groups on one matrix engine: staged compute, combined launch words | derived | standalone `[1,576]` linear (`fixtures/linear_emit`) |
| Mode-register group is always live | learned | `silu` |
| One gate set per input-port block | learned | `rmsnorm` |
| `npu_params` order: initializers, then synthesized constants | learned | `rmsnorm` |
| Loaders alternate copy engines 4, 3, 4, ... across ops | learned | `rmsnorm` |
| A job's waits precede its register writes | learned | `rmsnorm` |
| The main engine does not repeat a wait | learned | `attention` |
| Graph-input jobs run before the first core job | learned | `attention` |
| Engine placement of each MatMul, loader start, PARAM hoists, signed-output bit, destination gate block (`_FITTED_RULES`) | fitted | five graphs, see below |
| The main-engine transpose job: record order and three constants (`TRANSPOSE_UNEXPLAINED`) | learned, unexplained | ten native builds of five graphs |
| The rest of the main-engine transpose job (`MAIN_TRANSPOSE_JOB`) | derived | standalone copy-engine transpose and DEQUANT jobs |
| Where the output's PARAM job runs (`wiring["output_param"]`) | learned, not predicted | `rmsnorm` at `[1,576]` |
| Wide ReduceMean: single-precision lane, `zp_x * min(n, 256)`, packed pad lanes | learned | ReduceMean at 64 to 2048, `rmsnorm` at `[1,576]` |
| The pad group `0x0cd0..0x0da0` is live only while `0x0c20` is nonzero | learned | ReduceMean at 288 and 384 |

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

## Several MatMuls

Four more graphs were built to test what the attention block alone could not: `attn16`,
`mm_chain`, `mm_chain3` and `attn_proj` (table above), with standalone programs for each
of their ops. Together with `attention` that is five graphs and ten native builds. All
ten stitch exactly, without a `place` in the wiring.

### Where each MatMul runs

Every build agrees between its two calibrations. "Group k/n" means that n ops share the
matrix engine and are pipelined on it. The transpose is the B operand's; a constant B
needs none.

| build | MatMul | A | B | matrix engine | loader | B transpose |
| --- | --- | --- | --- | --- | --- | --- |
| `attention` | 1 | `q [8,64]` input | `kt [64,8]` input | 0 | copy 3 | copy 3 |
| `attention` | 2 | `p [8,8]` Softmax | `v [8,64]` input | 1 | copy 4 | main |
| `attn16` | 1 | `q [16,32]` input | `kt [32,16]` input | 0 | copy 3 | copy 3 |
| `attn16` | 2 | `p [16,16]` Softmax | `v [16,32]` input | 1 | copy 4 | main |
| `mm_chain` | 1 | `a [8,64]` input | `b [64,8]` input | 0 (group 1/2) | copy 3 | copy 3 |
| `mm_chain` | 2 | `m [8,8]` MatMul 1 | `c [8,64]` input | 0 (group 2/2) | copy 4 | main |
| `mm_chain3` | 1 | `a [8,64]` input | `b [64,8]` input | 0 (group 1/3) | copy 3 | copy 3 |
| `mm_chain3` | 2 | `m [8,8]` MatMul 1 | `c [8,64]` input | 0 (group 2/3) | copy 4 | main |
| `mm_chain3` | 3 | `n [8,64]` MatMul 2 | `d [64,8]` input | 0 (group 3/3) | copy 3 | main |
| `attn_proj` | 1 | `q [8,64]` input | `kt [64,8]` input | 1 (group 1/3) | copy 4 | copy 3 |
| `attn_proj` | 2 | `p [8,8]` Softmax | `v [8,64]` input | 1 (group 2/3) | copy 3 | copy 3 |
| `attn_proj` | 3 | `o [8,64]` MatMul 2 | `w [64,64]` constant | 0 and 1 (group 3/3 on 1) | copy 4 | none |

The same MatMuls in their standalone programs:

| standalone program | shape | matrix engine | loader | B transpose |
| --- | --- | --- | --- | --- |
| `matmul`, `c_matmul_qk`, `c_mm_nd` | `[8,64] @ [64,8]` | 1 | copy 4 | copy 3 |
| `c_mm_qk16` | `[16,32] @ [32,16]` | 1 | copy 4 | copy 3 |
| `c_mm_pv16` | `[16,16] @ [16,32]` | 1 | copy 4 | copy 3 |
| `c_matmul_pv`, `c_mm_mc` | `[8,8] @ [8,64]` | 0 | copy 3 | copy 3 |
| `c_mm_proj` | `[8,64] @` constant `[64,64]` | 0 and 1 | copy 4 | none |

No single placement rule fits all builds. Candidates and their counterexamples:

| candidate rule | holds for | fails for |
| --- | --- | --- |
| a fused MatMul keeps its standalone matrix engine | 5 of 12 fused MatMuls | the first MatMul of `attention`, `attn16`, `mm_chain`, `mm_chain3`; `attention` 2, `mm_chain3` 3, `attn_proj` 2 |
| a fused MatMul keeps its standalone loader engine | 4 of 12 | every MatMul of `attention` (2), `mm_chain` and `mm_chain3`; `attn16` 1 |
| a transpose stays on copy engine 3 | 6 of 11 fused transposes | `attention` 2, `attn16` 2, `mm_chain` 2, `mm_chain3` 2 and 3 |
| loader engine = 3 + matrix engine | 17 of 20 (fused and standalone) | `mm_chain` 2, `mm_chain3` 2, `attn_proj` 2 |
| loaders alternate 4, 3, ... in op order (the RMSNorm rule) | 9 of 20 | every MatMul of `attention`, `attn16`, `mm_chain`, `mm_chain3`; standalone `c_matmul_pv`, `c_mm_mc` |
| the first MatMul of a graph runs on engine 1 | 7 of 13 graphs | `attention`, `attn16`, `mm_chain`, `mm_chain3`, standalone `c_matmul_pv`, `c_mm_mc` |
| the first MatMul of a graph runs on engine 0 | 6 of 13 | `attn_proj` and the other six standalone programs |

### The fitted rule set

The stitcher therefore uses a rule set that was **fitted** to these five graphs: rules
were added and adjusted until all ten builds matched. It is a description of ten builds,
not a model of Pulsar2's scheduler. Several rules rest on one graph, and another graph
may need other rules. Each rule is a named entry of `_FITTED_RULES` in
`scripts/axera/graph_stitch.py`. The last column lists the graphs that stop matching
their native build when the rule is replaced by the alternative in parentheses (both
calibrations agree in every cell; `test_fitted_rule_is_needed_by`).

| rule | what it says | breaks without it |
| --- | --- | --- |
| `first_engine` | With several matrix ops the first MatMul runs on matrix engine 0, or on 1 when the graph has a constant-weight MatMul. A graph with one matrix op keeps its standalone engines. | (every MatMul on its standalone engine) all five |
| `second_engine` | A MatMul whose A operand is the previous MatMul's output stays on that MatMul's engine. Otherwise it takes the other engine, unless an op already uses that one. | (always the same engine) `attention`, `attn16`; (always the other) `mm_chain`, `mm_chain3`, `attn_proj` |
| `loader_start` | Loaders alternate over the graph. They start on copy engine 3 when the first matrix op runs on matrix engine 0, else on 4. | (always start on 4) `attention`, `attn16`, `mm_chain`, `mm_chain3` |
| `later_transpose` | The first transpose runs on copy engine 3. Later ones run as main-engine jobs, or on copy engine 3 when the graph has a constant-weight MatMul. | (all on copy 3) `attention`, `attn16`, `mm_chain`, `mm_chain3`; (later ones all on main) `attn_proj` |
| `hoist_param` | The MatMul pipelined right behind the first one on its matrix engine issues its first PARAM job one job early, before the last QUANT job already planned. | `mm_chain`, `mm_chain3`, `attn_proj` (main segment) |
| `hoist_output_param` | The output's PARAM job runs before the first core job that waits for an engine sub-program depending on the last graph-input job. `wiring["output_param"]` overrides it. | `attn_proj` (main segment) |
| `signed_output` | A MatMul whose output tensor is quantized signed (`m` and `n` of the chains, read by a MatMul with two live operands; `o` of `attn_proj`, read by the constant-weight MatMul, is unsigned) sets bit 21 of its compute's `0x03d0` and adds 128 to its offset lane in `npu_params`. | `mm_chain`, `mm_chain3` (matrix segment 0, `npu_params`) |
| `destination_block` | The destination descriptor registers (`0x0710..`) share one gate set, like an input-port block. | `attn_proj` (main segment) |

The `attn_proj`-only parts are the weakest: engine 1 for the first MatMul, "unless an op
already uses that one", copy engine 3 for later transposes, `hoist_output_param` and
`destination_block` are each one observation, all on the only graph with a
constant-weight MatMul. `hoist_output_param` does not predict RMSNorm at `[1,576]`,
which still needs `output_param`.

An explicit `place` on an op (`matrix`, `loader`, `transpose`, any subset) overrides the
placement rules for that op. Whether the device accepts a placement other than the
native build's was not tested: every model run on the device is one whose bytes equal a
native build outside the slot table. A different but self-consistent placement
stitches (for example the attention block with both MatMuls on matrix engine 0), and
nothing here says whether it would run correctly.

### What several MatMuls needed besides placement

- **Pipelining on one matrix engine.** An op's sub-programs on a matrix engine are some
  loads followed by one compute. When two ops share the engine, the first op's compute
  is written up to `0x0220 = 0x100000` but not launched; the last load of the next op
  launches both with `0x0230 = 0x11ff03` and a final `0x0220 = 0x03000000`. The same
  words and the same staging occur inside one standalone program, the `[1,576]` linear
  layer (segment 1, four times), so this part is derived.
- **Later sub-programs as state differences.** A later group on a matrix engine, and a
  copy-engine transpose that is not first on its engine, are written as the difference
  between their launch state and the engine's running state.
- **Constant-weight MatMul** (`"linear": True`). Its weights are loaded into the weight
  region the matrix engines address from 0, not into scratch, and its work is split
  over both matrix engines. `npu_params` holds the quantized weights (copied), then per
  output channel an offset lane and a scale lane: `scale[c] = s_a * s_w[c] / s_y` and
  `offset[c] = zp_y - scale[c] * zp_a * S[c]`, with `S[c]` the sum of channel c's
  quantized weights. The per-channel weight scales are a list in `scales`.
- **Operand offset words.** A load's four offset words (`0x0330`, `0x0320`, `0x0310`,
  `0x0300`) split `0xffff0 - address / 0x20` into a 10-bit part and a page. Every
  standalone program has page `0x3a1`; the third MatMul of `mm_chain3` reads an operand
  past `0x2f8000` and has page `0x3a0`.
- **A loader template from another program.** Both `attn16` MatMuls load on copy engine
  4 standalone, and `attn16` needs a loader on copy engine 3. The template comes from
  `c_matmul_pv` through `gate_corpus` (`fixture_corpus`).

### The main-engine transpose job

A later MatMul's B operand is transposed by a main-engine job that no standalone program
contains (30 records in `attention` and `mm_chain`, 26 in `attn16`, 30 and 24 in
`mm_chain3`; the counts differ because unchanged registers are not rewritten). Earlier
this page copied the whole job from the attention build. It is now derived from the
standalone copy-engine transpose of the same MatMul, and reproduces all ten builds:

| main-engine registers | value |
| --- | --- |
| `0x02a0..0x0350` (source port block) | copy engine 3's `0x0160..0x0210`, shifted by `0x140` = main `0x02c0` - copy `0x0180`, the distance between the two source address registers |
| `0x0710..0x07c0` (destination port block) | copy engine 3's `0x0390..0x0440`, shifted by `0x380` = main `0x0730` - copy `0x03b0` |
| `0x0c30..0x0cc0` (permutation) | copy engine 3's `0x04e0..0x0570`: the block after the mode word on both engines (`0x04d0` on the copy engine, `0x0c20` on the main engine) |
| `0x0160` (job type) | copy engine 3's `0x04c0` |
| `0x0290` | `0x3fd00`, the main engine's descriptor of a scratch source, read from the DEQUANT job of the standalone programs (the same in all of them) |
| scratch addresses | moved to the fused layout |
| `0x0280 = 1`, `0x0c10 = 0x100040`, `0x0c20 = 0x104` | **unexplained constants** (`TRANSPOSE_UNEXPLAINED`) |

The three constants are the same in every main-engine transpose job of the ten builds,
across operand shapes `[8,64]`, `[16,32]` and `[64,8]`, and no committed standalone job
holds any of them in that register. They are copied, as is the job's record order.

## Record counts

Decompressed records per segment, identical for stitched and native builds and for both
calibrations (except the `mul_sig` and `attn16` blob sizes):

| graph | matrix 0 | matrix 1 | main | copy 3 | copy 4 | main jobs | restored registers | `npu_params` bytes | blob bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `silu` | 8 | 4 | 452 | 4 | 4 | 6 | 7 | 40 | 3328 |
| `sqrt_mul` | 8 | 4 | 292 | 4 | 4 | 6 | 0 | 40 | 2208 |
| `sig_add` | 8 | 4 | 492 | 40 | 44 | 6 | 0 | 44 | 4104 |
| `mul_sig` | 8 | 4 | 492 | 4 | 4 | 8 | 10 | 60 | 3544 (pm1), 3320 (pm4) |
| `rmsnorm` | 8 | 4 | 524 | 52 | 56 | 10 | 37 | 109 | 3816 |
| `attention` | 100 | 100 | 608 | 68 | 44 | 11 | 18 | 720 | 5536 |
| `attn16` | 100 | 96 | 604 | 64 | 44 | 11 | 16 | 464 | 5536 (pm1), 5544 (pm4) |
| `mm_chain` | 156 | 4 | 328 | 68 | 44 | 9 | 13 | 720 | 3640 |
| `mm_chain3` | 212 | 4 | 408 | 80 | 44 | 12 | 13 | 868 | 4080 |
| `attn_proj` | 96 | 204 | 572 | 92 | 60 | 10 | 22 | 5840 | 5856 |

"Restored registers" counts the writes added by step 3 at the `pm1` calibration.

## Offline checks

`tests/test_axera_graph_stitch.py` runs without a device or Docker:

- 20 stitched models (10 graphs, 2 calibrations) equal their native builds as described
  above.
- Standalone sources built at another calibration give the same models: 16 two-op
  cases (sources `pm1`, `pm4` or `asym` for each fused calibration) and both swaps for
  RMSNorm, attention and the four other MatMul graphs. The standalone programs
  contribute structure only.
- A one-op graph stitched from each of the 44 standalone programs reproduces that
  program's segments, `npu_params` and model. The exception is the `asym` Add build: its
  two `npu_params` scale words are in the other order.
- The attention block stitches without `place` to the same bytes as with it; the
  placement of every MatMul in the five graphs and in the standalone programs is the
  one in the tables above.
- Each fitted rule, replaced by its alternative, breaks exactly the graphs listed for it.
- The main-engine transpose jobs hold the derived descriptor and the three constants,
  and no standalone job holds those constants.
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

AX8850, AXCL V3.6.5, 2026-10-10, the graphs with several MatMuls. Stitched model (no
`place`, fitted rules) versus native fused build: the outputs were bit-identical.

| graph | values compared per calibration |
| --- | --- |
| `mm_chain` | 30,720 |
| `mm_chain3` | 3,840 |
| `attn16` | 30,720 |
| `attn_proj` | 30,720 |

These stitched models equal the native builds outside segment 0's slot table, so the
runs confirm the stitched files, not the rules: no placement other than the native
build's was run.

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
  (Sigmoid, Sqrt, Mul, Add, ReduceMean, Div, Softmax, MatMul); a matrix-engine group
  that is not loads followed by one compute; a sub-program placed on an engine of
  another kind; a copy-engine loader with no template in any program given; a
  constant-weight MatMul built with input zero point 0.
- **Rules resting on a single native build.** Every learned rule in the table comes from
  one fused graph.
- **The MatMul rules are fitted.** The placement rules, the two PARAM hoists, the
  signed-output bit and the destination gate block were fitted to five graphs (ten
  builds), several of them to `attn_proj` alone, and three transpose-job constants are
  unexplained. A graph outside those five gets a placement from the same rules with no
  evidence that Pulsar2 would choose it. Whether the device accepts a different,
  self-consistent placement was not tested.
- **Add operand order.** Add's two scale words are written in ONNX input order. That
  holds in `sig_add`, `rmsnorm` and four of five standalone Add builds. The `asym`
  standalone Add build has them swapped, and nothing here predicts when.
- **Initializer bytes are copied.** The quantized bytes of an initializer operand come
  from the standalone program, so that program must have been built with the same
  constant. `consts[...]["values"]` checks it.
- **Input slot order.** The compiled input slot order is a per-build permutation (the
  two attention builds differ; so do the two builds of `mm_chain`, `mm_chain3` and
  `attn16`). The caller chooses it in `wiring["inputs"]`.
