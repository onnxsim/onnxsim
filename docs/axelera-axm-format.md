# Metis `.axm` package format (SDK 1.9.0, measured)

Notes from inspecting packages compiled by the Voyager SDK on build host (H8) (Metis M.2 113m).
Everything here was checked on the card unless marked "inferred".

## Container
- An `.axm` is a zip archive (stored, not compressed in the packages we built).
- Typical entries for a single-kernel model: `ai_core_kernel_0.elf`, `ai_core_kernel_0.c`, `model.json`.
  Larger models add `compile_config.json`, `model_info.json`, `manifest.json`, constant pool
  binaries (`pool_*.bin`, `l2_constant.bin`), and `kernel_function.*` names.

## `model.json`
- `memory_pools`: name, size, alignment, memory (`ddr`/`l1`/`l2`), depth, optional `blob_file`.
- `inputs` / `outputs`: name, memory pool, offset, shape, dtype.
- `kernels`: name, argument pool list, pipeline stage, source file.
- For the Metis LLM packages (type `llm`) the runtime exposes no inputs or outputs; I/O goes through pools.

## The kernel ELF
- A 64-bit RISC-V ELF (`Machine: RISC-V`, `Flags: RVC, double-float ABI`). It is the control program that runs
  on the card's RISC-V core, not the matrix program itself.
- The same firmware functions appear in every kernel we compared (52 functions, including `main`,
  `module_init`, `handle_packed_tasks`, DMA and token helpers).
- Per-model data: `ai_core_kernel_0_packed_tasks` (the task stream) in `.rodata`, pool names in `.dea_memory`,
  and the scale array in `.l1_memory` (see below).

## Packed task stream (`ai_core_kernel_0_packed_tasks`)
- A sequence of chunks. Each chunk: `count` (u16), two reserved bytes, `size` (u32, total bytes including the
  8-byte chunk header and any trailing padding), then `count` tasks, then padding.
- Each task starts with an 8-byte header: `type` (u8), reserved (u8), `size` (u16, bytes), reserved (u32).
- Task types (from the SDK's `task_handler.h`): COMMAND 0, INSTRUCTION 1, TOKEN 2, POLLING 3,
  START_DMA 4, POLL_DMA 5, C2C_SYNC 6, TIMESTAMP 7, FUNCTION 8.
- Payload layouts used by the parser (`scripts/axelera/axm/pstream.py`):
  - COMMAND: target module id (u8, byte 8), then 64-bit words from byte 16. Module ids come from `module.h`
    (0 D_IFD_0, 1 D_IFD_1, 2 D_ODR, 3 D_DPU, 4 D_IAU, 5 D_DWPU, 6 M_IFD_0, 7 M_IFD_1, 8 M_IFD_W, 9 M_ODR,
    10 M_IAU, 11 M_DPU, 12 M_MVMEXE, 13 M_MVMPRG).
  - START_DMA: channel (byte 8), count (u32 at 12), then `count` transfers of three u64 (src, dst, size) from byte 16.
  - INSTRUCTION, TOKEN, POLLING, POLL_DMA, C2C_SYNC, TIMESTAMP, FUNCTION: see `pstream.py`.
- Round trip: `pstream.py` parses and re-encodes five compiled streams (832 to 6336 bytes) byte for byte.
  Padding after the last task is kept verbatim.
- The payload *meanings* are not documented in the SDK headers. The command headers' token counts are
  consistent (matrix execute consumes two tokens, program produces two), but I haven't verified the rest.

## Requantization scale (the one field we can edit and verify)
- The `mul` and `dpu_cast` in the GEMM scale the int32 accumulator by a float32 constant and saturate to int8.
- The constant is stored as 512 float32 values, one per lane across the eight row tiles, in the ELF section
  `.l1_memory` (2048 bytes, preloaded into L1 on load).
- The stream's per-tile COMMAND (module D_IFD_1, byte 25) holds an index into that array. Its value changes with
  the scale because the compiler assigns slots differently; it is not the scale itself.
- Verified on the card (clean device state, see below):

  | Array value | Device output matches | Exact |
  |---|---|---|
  | 1/4096 (compiled) | 1/4096 reference | 99.16% |
  | 1/2048 (edited) | 1/2048 reference | 98.50% |

- Setting every value to 0.0 makes the output all zeros, which confirms the device reads this array.
- `scripts/axelera/axm/axm_scale.py set` edits the array (refuses unless every value equals the expected old value);
  `check` runs the package and compares against a numpy reference.

## Device state rule
- The device keeps state between model loads. A result can reflect the previously loaded package.
- Several results I first read as "no effect" were stale-state artifacts (for example, the 1/2048 rewrite).
- Before any comparison, run a different, small model in a **separate process** (`axm_scale.py check` does this).
  Running the reset in the same process did not clear the state.

## Open questions
- Which payload fields of COMMAND, INSTRUCTION, and TOKEN are semantic (addresses, lane counts, sync modes)?
- How the per-tile index in D_IFD_1 is chosen, and whether it can be redirected to a new constant table.
- Whether the runtime accepts a stream we author (not a compiler output) without a matching constant layout.

## Field map (differential, measured)

Method: compile the unrolled GEMM at shapes that differ in one dimension (base 1024x512x512, N doubled,
K doubled), align the two packed streams with a sequence matcher (task count changes from 217 to 221), and
record which bytes change inside matched tasks.

Confirmed:
- **START_DMA transfers** (task type 4): each transfer is three u64 words at task byte 16 + 24k:
  - `src`: pool index in bits 48-55 (0 = ddr input here), offset in bits 0-39, enable flag in bits 56-63 (1 for a source).
  - `dst`: pool index and offset in the same layout. Offsets are L1 addresses (for example 0x18380000) and move with the shape.
  - `size`: bytes moved. Doubling N doubles the size (32768 to 65536 for the weight tile; 131072 to 262144 for the wider case).
  - The bytes that change across shapes are the offset and size fields (task bytes 17-18, 25-26, 33-34).

Partly mapped:
- **COMMAND** payload words (task type 0) change with every shape, in bytes 25-31 of most tasks. That is where
  the matrix commands store their tile dimensions and addresses. I haven't assigned individual fields yet.
- **INSTRUCTION** (type 1) and **TOKEN** (type 2) payloads change in a few tasks, which matches the extra
  tasks that appear with the wider shapes.

Not yet mapped: which COMMAND bits select the matrix unit's tile size, the token mode bits, and the IMC load
parameters.

## Matrix-execute field (M_MVMEXE, measured)

- Word 1 of each M_MVMEXE command (8 per GEMM, one per tile) has a 2-bit field at bits 16-17.
- Measured values, unrolled GEMM, 8 tiles:

  | Shape (M x K x N) | bits 16-17 | Numerics vs reference |
  |---|---|---|
  | 1024 x 512 x 512 | 1 | 99.16% exact |
  | 1024 x 512 x 1024 | 2 | 99.13% exact |
  | 1024 x 1024 x 512 | 2 | 98.53% exact |
  | 1024 x 512 x 2048 | 0 | 99.15% exact |
  | 1024 x 2048 x 512 | 0 | 98.53% exact |

- The field is linear in the 512-wide chunk count along N or K (1, 2, then 4), truncated to two bits.
- Results stay correct at the wrapped value, so the hardware either ignores the high bits or reads the field
  modulo 4. I haven't confirmed which. Treat the meaning as a hypothesis until a shape where the wrap matters
  is checked.

## Correction: the matrix-execute field is three bits (measured)

- The chunk field is bits 16-18 of word 1 of M_MVMEXE, not bits 16-17. Word 1 of the 2048-wide kernel is
  `0x40000` against `0x10000` for 512 wide: bit 18 is set and bits 16-17 are zero.
- Values by shape: 512 gives 1, 1024 gives 2, 2048 gives 4. That is a count of 512-wide chunks.
- The earlier "wrap to 0" reading came from looking at two bits only, so it is withdrawn.
- Setting the count to 0 on the 512-wide kernel (bits 16-17 cleared, count 0) crashed the card. A zero count is
  an invalid command, which fits the crash.
- Not yet tested: setting the count to 2 or 4 on the 512-wide kernel, and whether the output then matches a
  reference for that shape.

## Count test on the 512-wide kernel (measured)

- Setting bits 16-18 of M_MVMEXE word 1 to 0 crashed the card.
- Setting them to 2 also crashed the card (`ZE_RESULT_ERROR_DEVICE_LOST`), with the card healthy before the run.
- Setting them to the compiled value 1 runs and matches the reference (99.16%).
- So the count must match the shape, and a mismatch is fatal. The count is a shape-derived parameter the device
  checks, not a free knob. The value 4 is untested on the card and expected to fail the same way.

## Device recovery (measured)

- After a device-lost crash, a metis module reload did not recover the card, and a host reboot did.
- A PCIe recovery was tried on build host (H8) (card at 03:00.0, driver `axl`):
  1. unbind the driver: ok
  2. bus reset through the `reset` sysfs file: I/O error
  3. rebind: I/O error. The kernel log shows `Link down` after the link came up at Gen1, then
     `Failed to enable MSI`, then the driver released its DMA memory.
- The card stayed enumerated on the bus with no driver. A host reboot restored it.
- So a PCIe bus reset does not recover the card in this state. Recovery needs a reboot.

## COMMAND bits that change with shape (read-only decode, measured)

Base 1024x512x512 compared with 1024x512x2048 (N) and 1024x2048x512 (K), matched tasks only.

| Module | Words / bits that change | Changes with |
|---|---|---|
| M_MVMEXE (12) | word 1, bits 16 and 18 | N and K (chunk count, see above) |
| M_DPU (11) | word 1 and word 2, bits 42 and 44 | N (not K) |
| M_IFD_0 (6) | word 0 bit 48; word 1 bits 58, 60, 62 | N and K |
| M_ODR (9) | word 1, bits 17, 19, 20 (K) and 19, 21, 60, 62 (N) | N and K |
| M_IAU (10) | word 0 bit 48 (K only) | K |
| D_IFD_0 (0), D_ODR (2) | word 1, several bits in 17 to 21 and 58 to 62 | N and K |
| D_IFD_1 (1) | word 1, bits 8 to 10 | N and K (the per-tile slot index, see above) |
| D_DPU (3), D_IAU (4), D_DWPU (5) | D_DPU word 1 and 2, bits 42 and 44 (N only); D_IAU and D_DWPU no change | N |

Reading: the bits that move with the N width look like tile-width or address fields in the output and DPU
commands. The bits that move with K look like input-stream lengths. These are groupings from the data, not
confirmed meanings. Next: confirm one field per module by changing a value on a shape that does not crash the
card, which needs a reset plan first.

## Stage 1 result: the D_IFD_1 index selects a scale vector (measured)

- Setup: the scale array's vector 1 (lanes 64 to 127) set to 1/2048, all others 1/4096. Tile 0's D_IFD_1
  index changed from 3 to 1. Before the run, the indices were [3, 5, 0, 1, 4, 2, 7, 6].
- Prediction: tiles 0 and 3 use vector 1 (1/2048); tiles 1, 2, 4 to 7 keep 1/4096.
- Result: all eight tiles matched their prediction.

  | Tile | Exact vs 1/2048 | Exact vs 1/4096 | Best |
  |---|---|---|---|
  | 0 | 98.49% | 1.34% | 1/2048 |
  | 1 | 1.30% | 99.18% | 1/4096 |
  | 2 | 1.31% | 99.17% | 1/4096 |
  | 3 | 98.52% | 1.26% | 1/2048 |
  | 4 to 7 | about 1.3% | about 99.1% | 1/4096 |

- Conclusion: the per-tile index in D_IFD_1 selects which 64-lane scale vector the tile uses. Changing it is
  safe on this shape and produces the predicted per-tile change, with no crash.

## Stage 2 result: bit 42 of M_DPU crashes the card (measured)

- Setup: bit 42 of word 1 of the first M_DPU command (target 11) flipped, 0x40000010000 to 0x10000.
  Health check passed before the run.
- Result: the device was lost on the first run (`ZE_RESULT_ERROR_DEVICE_LOST`). No per-tile output was obtained.
- The prediction (output change or rejection) was wrong on the second count: the device was lost.
- Conclusion: bit 42 is not a free field. Changing it on one command crashes the card. Stage 3 fields should be
  treated as crash-prone and tested only with explicit approval and a reboot plan.

## Stage 3, M_ODR bit 17 (measured)

- Setup: bit 17 of word 1 of the first M_ODR command (target 9) flipped, 0x1000000018360000 to
  0x1000000018340000. Health check passed before the run.
- Result: no crash. Tile 0 is almost entirely wrong (64,982 of 65,536 values mismatched, in every 64-column
  block). Tiles 1 to 7 are at the normal rounding level (about 0.85% mismatched, 538 to 569 of 65,536).
- Reading: bit 17 of the M_ODR command changes the output of its own tile across all columns. It is a
  write-side field (output layout or format), not a crash field, at least on this shape.
- Prediction check: the output-change branch of the prediction held; no crash. Confirming which bit does what
  needs the next bits in the list.

## Stage 3, M_ODR bit 19 (measured)

- Setup: bit 19 of word 1 of the first M_ODR command flipped, 0x1000000018360000 to 0x10000000183e0000.
  Health check passed before the run.
- Result: no crash. Tile 0 almost entirely wrong (65,081 of 65,536 mismatched, in every column block). Tiles 1
  to 7 unchanged at the rounding level.
- Same pattern as bit 17. Bits 17 and 19 may belong to one multi-bit field in the M_ODR write. Test by
  flipping them together or by checking whether bit 18 is part of the same field.

## Stage 3, M_ODR bit 20 (measured)

- Setup: bit 20 of word 1 of the first M_ODR command flipped, 0x1000000018360000 to 0x1000000018260000.
  Health check passed before the run.
- Result: no crash. Tile 0 corrupted with the same per-block counts as bit 19 (65,081 mismatched). Tiles 1 to 7
  unchanged at the rounding level.
- Bits 17, 19, and 20 all corrupt tile 0. Bits 19 and 20 give identical output even though the two modified
  words differ. So the effect is probably from a shared field or an address that both bits select. Next: test
  bits 19 and 20 together, and bit 21.

## Stage 3, M_ODR bit 21 (measured)

- Setup: bit 21 of word 1 of the first M_ODR command flipped, 0x1000000018360000 to 0x1000000018160000.
  Health check passed before the run.
- Result: no crash. Tile 0 corrupted with the same per-block counts as bits 19 and 20 (65,081 mismatched).
  Tiles 1 to 7 unchanged at the rounding level.
- Bits 19, 20, and 21 give identical tile-0 output. Bit 17 gives a slightly different count (64,982).
- Hypothesis: any change in bits 19 to 21 of this command makes tile 0's output a fixed wrong result, for
  example written from another address or zeroed. To test: compare tile 0's device output with the other tiles'
  outputs and with zero.

## Tile 0 after the M_ODR bit 21 flip (measured)

- Tile 0's output has 247 distinct values, not zero. It matches its own reference in 0.7% of values and tile 1's
  reference in 0.8%.
- So the corruption is not a zero fill and not a copy of another tile.
- Remaining hypotheses: a different scale or rounding on this tile's write, or a permutation of lanes or column
  blocks. Next test: compare the histogram of tile 0's values with its reference, and check for a column
  permutation.

## Tile 0 permutation test after the M_ODR bit 21 flip (measured)

- No row of tile 0 is a permutation of its reference row (0 of 128 rows).
- Tile 0's value histogram differs from the reference histogram.
- So the corruption is not a lane or column reordering. The bit change alters the arithmetic for that tile.
- Next: sweep a scale for tile 0 (powers of two) and check whether it matches one of them. That would show
  whether the flip changes the requantization for that tile.

## Tile 0 scale sweep after the M_ODR bit 21 flip (measured)

- Power-of-two scales 2^-4 to 2^-16 do not match tile 0's output. The best match is 21% at 2^-4, which is what
  clipping alone gives.
- So the flip does not change the requantization scale for tile 0 either.
- Remaining hypothesis: tile 0's matrix computation uses different operands (input or weight source) because
  bits 19 to 21 of its M_ODR write select them. Next test would compare tile 0 against products of other tiles'
  inputs and of permuted weights.

### M_MVMPRG index (device test, measured)

- Setup: 1024×512×1024 kernel, two matrix programs (M_MVMPRG). Bits 8–15 of word 1 are their index: 0 and 1.
  Baseline run (no change) matches the reference at 99.1%.
- Swap: the two indices exchanged (0 to 1, 1 to 0). Health check passed before and after.
- Result: both 512-column chunks mismatch about 99%, overall exact 0.9%.
- Conclusion: the index selects which weight chunk a matrix program loads. Swapping makes each chunk's columns
  use the other chunk's weights. Confirmed prediction; no crash.

### M_IFD_W address (device check, measured)

- 1024×512×1024 kernel, two weight loads (M_IFD_W). Both word 1 values are identical: `0x40000000182e0000`.
- Baseline matches the reference at 99.1%. The one-sided address change I planned was a no-op, because the
  low 32 bits were already equal. No device change was made.
- So the low address does not select the chunk. The chunk is selected by the matrix-program index (M_MVMPRG).
- The address varies with shape (for example `0x181e0000` at N=2048), so it is a staging location for the
  weight block, not a per-chunk pointer. Exact meaning still open; a test needs a second, different valid address.
- Card healthy before and after.
