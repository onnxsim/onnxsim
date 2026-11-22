# Metis `.axm` package format (SDK 1.9.0, measured)

Notes from inspecting packages compiled by the Voyager SDK on takecheeze-H8 (Metis M.2 113m).
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
