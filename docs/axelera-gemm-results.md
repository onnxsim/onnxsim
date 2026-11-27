# Metis M.2 113m: GEMM throughput and kernel-level visibility

Draft for review. Not sent.

## Setup
- Device: Metis M.2 113m, 4 AIPU cores, firmware 1.9.0, clock 800 MHz reported by `axdevice`.
- Host: build host (H8), x86_64, Ubuntu 24.04, metis-dkms 1.7.16.
- SDK: Voyager 1.9.0, Python 3.12, `AXE_PROFILING_CONFIG=HOST_TEMPLATE`, `host_profiling=True`.

## Measured results (int8 GEMM, K=512, N=512 unless noted)
| Shape | Kernel time (krn) | TOPS (kernel) |
|---|---|---|
| M=1024, single tile | 1514 µs | 0.36 |
| M=4096, single tile | 3896 µs | 0.55 |
| M=8192, single tile | 8612 µs | 0.50 |
| M=4096, 1x1 conv form, 1 instance | 3007 µs | 0.71 |
| 1x1 conv form, 3 instances, double buffer | — | 1.25 (best) |

Per-call host fields at M=4096: prep 5 µs, inp 266 µs, krn 3939 µs, out 235 µs.
So kernel execution is about 88% of each call.

## Observations
1. Kernel time does not scale with work. Halving N or K saves about 20%; halving M saves about 40%.
2. Splitting one GEMM into 1, 2, 4, 8 row tiles raises kernel time from 1364 to 2459 µs at M=1024, roughly 156 µs per added tile. Added tiles add 2 DMA start/poll pairs and 2 token pairs each.
3. The compiler's own stream for the same 1024x512x512 GEMM has 27 tasks (8 DMA pairs, 1 token pair, 7 commands). A hand-written tiled stream of the same GEMM has 217 tasks (34 DMA pairs, 16 tokens, 130 commands).
4. Removing the per-tile scale (mul) saves about 5%.
5. Compiler async scheduling (`inter_operator_async`) makes the 1x1-conv GEMM about 16% slower. Grouping (`group_ifdw_tasks`) recovers it to baseline, with no gain.

## Blocked items
- A software-pipelined tile loop (SDK `_n_outer_pipelined` pattern, `A.tiles(..., drain=2)`) fails at compile time with `Check failed: lanes > 1 (-1 vs. 1)`, from a hand-written AxelerScript kernel. The same pattern works inside the SDK's own operator code.
- Forcing the SDK's pipelined schedule for M>1 via `axol.core.matmul(pipeline=True)` compiles but runs at 0.007 to 0.056 TOPS, far below the default schedule.
- Reloading the weights into IMC once per row tile (`axol.atomic.load_imc` per tile) crashes at runtime with `ZE_RESULT_ERROR_DEVICE_LOST`, and later with `AXR_ERROR_RUNTIME_ERROR` from `zeCommandQueueExecuteCommandLists`. Reboot clears the card.

## Visibility gaps
- No per-command or matrix-unit busy-time counters are readable. AI-core trace returned no data.
- Level Zero timestamps are unsupported (`ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`).
- The packed task stream is readable, and the command target IDs decode (`module.h`), but the command payload semantics aren't documented.

## Questions
1. Is there a supported way to get per-command or per-module busy time on Metis, or device-side timestamps for a kernel run?
2. Is `lanes > 1 (-1 vs. 1)` a known limitation for hand-written `A.tiles(..., drain=N)` loops outside operator code? What is the supported form?
3. Is reloading IMC weights per tile intended to work? The runtime error and device loss suggest not.
4. Recommended way to overlap input/output DMA with matrix execution in a hand-written kernel?
5. Is the 214 TOPS INT8 figure per device or per core, and under what clock and shape?



## Double buffering (added)
- 1x1-conv GEMM, M=4096, K=512, N=512, aggregate TOPS with `double_buffer=True` in `load_model_instance`:
  1 instance 1.06; 2 instances 1.17; 3 instances 1.09 (run-to-run variation around 1.1 to 1.25).
- Without double buffering, one instance measured 0.71 TOPS, so double buffering gives about 50% at one instance.
- Still about 0.5% of the 214 TOPS device peak.


## Wider N with double buffering (added)
- Conv-form GEMM, `double_buffer=True`, aggregate TOPS:
  | Shape (M x K x N) | Instances | TOPS |
  |---|---|---|
  | 4096 x 512 x 512 | 2 | 1.17 |
  | 8192 x 512 x 512 | 1 | 1.08 |
  | 8192 x 512 x 512 | 2 | 1.17 |
  | 4096 x 512 x 1024 | 1 | 1.25 |
  | 4096 x 512 x 1024 | 2 | 1.73 |
  | 4096 x 512 x 1024 | 3 | 1.73 |
  | 4096 x 512 x 2048 | 2 | 1.70 |
- Wider N helps more than more instances or more rows. Best so far: 1.73 TOPS, about 0.8% of the 214 TOPS peak.

## Multi-core (added)
- Passing a core list to the SDK's registered core matmul (`cores=(0,1)`, `tiled=True`) makes the compiler emit the multi-core schedule (`_multicore_n`). Single-core output of the same schedule runs at 0.18 TOPS (1024x512x512, DDR input).
- With 2 or 4 cores, compile succeeds but the first run fails with `AXR_ERROR_RUNTIME_ERROR` at `zeCommandQueueExecuteCommandLists`, then `ZE_RESULT_ERROR_DEVICE_LOST` (0x70000001). The device stays lost after the run: a metis module reload does not recover it, and a host reboot does. Reproduced twice, including after a reboot.
- Placing the input in one core's L1 instead fails at compile time: `Buffer a_l1 used in conflicting cores: 0 vs 1`.
- Working multi-core path today: separate model instances, three instances, 1.25 TOPS. Four instances fail to connect (`AXR_ERROR_CONNECTION_ERROR`).
- Question 6: Is the multi-core N-shard schedule supported on Metis 1.9.0 for a kernel invoked through `axol.core.matmul` with `cores=(0,1)`? If so, what input placement does it require? The device-loss on first run suggests a runtime or firmware issue.
- Repro: `python multicore_mm.py 1024 512 512 --cores 2` (scripts/axelera/multicore_mm.py, branch claude/axelera-gemm-experiments is not yet updated with this file).


## lanes > 1 reproducer (added)
- Loop form: `for s in A.tiles(NT, drain=2):` with a body that indexes a buffer by the loop variable (`b_l1[s % 2, ...]`, or a tile offset computed from `s`) fails with `Check failed: lanes > 1 (-1 vs. 1)`.
- Same loop with no loop-variable indexing compiles, and so does a fully unrolled program.
- Minimal repros: `scripts/axelera/repro/bisect_slot.py` (fails), `bisect_off.py` (fails), `bisect_loopnovar.py` (compiles).
- Unrolled double-buffered pipeline (`pipe_unroll.py`) compiles without explicit waits, but shows no overlap: 0.21 to 0.23 TOPS, same as the baseline. Output verified (1024x512x512, 8 tiles): matches a numpy reference on 99.2% of values (remaining values differ by one rounding step) and is identical to the baseline kernel's output.

## Repro
- Scripts: `scripts/axelera/tiled_mm8.py` (args: M K N, `--tiles`, env `MUL`, `NOCOPY`, `USE_IDS`, `INLINE_WAIT`), `scripts/axelera/core_pipelined_mm.py`, `scripts/axelera/pipelined_mm.py`, `scripts/axelera/async_test.py` (conv form).
- Example: `python tiled_mm8.py 4096 512 512 --tiles 1 --frames 100` with `USE_IDS=0 INLINE_WAIT=0 MUL=1`.

## Package format and command fields (added)
- The requantization scale is a float32 array in the kernel ELF's `.l1_memory` section (512 values, one per
  lane across 8 tiles, preloaded into L1). Setting it to 2^-11 changes the device output to match a 1/2048
  reference (98.5%). Setting it to 0 gives all-zero output, so the device reads it.
- The packed task stream's per-tile COMMAND byte is an index into that array.
- M_MVMEXE word 1 bits 16-18 carry a count of 512-wide chunks (1 at 512, 2 at 1024, 4 at 2048, measured by
  compiling those shapes). Setting the count to 0 or 2 on the 512-wide kernel crashes the device
  (`ZE_RESULT_ERROR_DEVICE_LOST`). The count is checked against the shape, not a free parameter.
- Question 7: Is the M_MVMEXE chunk count documented? Is there a validation error we can read before the
  device crash? What do the other COMMAND payload bits select?
- Question 8: Recovery. A metis module reload does not recover a lost device. A PCIe bus reset through the
  `reset` sysfs file fails with an I/O error, the link retrains to Gen1 and drops, and the driver probe fails.
  Only a host reboot restored the card, in every case so far.
- Scripts: `scripts/axelera/axm/axm_scale.py` (scale show, set, check with reset), `pstream.py` (stream codec).


## CNN throughput and instance scaling (measured on one Metis M.2 113m)

Batch 1, streaming with one Python thread per instance, double buffering on. Effective TOPS = 2 × MACs × images/s.
MACs are standard published values at 224×224.

| Model | MACs | Config | Images/s | Effective TOPS |
|---|---|---|---|---|
| ResNet-18 | 1.82 G | 1 instance, 1 core | 685 | 2.5 |
| ResNet-50 | 4.11 G | 1 instance, 1 core | 380 | 3.1 |
| ResNet-50 | 4.11 G | 1 instance, 4 cores | 451 | 3.7 |
| ResNet-50 | 4.11 G | 2 instances, 1 core each | 685 | 5.6 |
| ResNet-50 | 4.11 G | 3 instances, 1 core each | 964 | 7.9 |
| ResNet-50 | 4.11 G | 4 instances, 1 core each | 1085 | 8.9 |
| MobileNetV2 | 0.30 G | 1 instance, 1 core | 766 | 0.46 |
| EfficientNet-B0 | 0.39 G | 1 instance, 1 core | 356 | 0.28 |

- Published vendor ResNet-50 (M.2): 1756 FPS, about 14.4 effective TOPS. Our four-instance result is about 62% of it.
- Scaling levels off: three to four instances adds about 12%.
- Multi-core within one instance gives about 19% at four cores. The gain from instances is larger.
- DINOv2-S/14 is about 6.1 GMACs by our estimate. The vendor's 131 FPS is about 1.6 effective TOPS. Not measured on our card: the zoo has no prebuilt ViT package.


## Single-core ceiling analysis

- The SDK documents 214 INT8 TOPS as the theoretical peak across all four cores at 800 MHz. The per-core
  peak is therefore about 53 TOPS.
- Measured, as share of per-core peak:

  | Result | Effective TOPS | Share of per-core peak |
  |---|---|---|
  | Best GEMM (3 instances) | 1.7 | about 3% |
  | ResNet-50, 1 instance, 1 core | 3.1 | about 6% |
  | ResNet-50, 4 instances | 8.9 total | about 4% per core |

- 50 TOPS on one core would be about 93% of the per-core theoretical peak. We don't consider that reachable.
- What would be needed:
  - Matrix unit busy nearly all the time: DMA fully overlapped with compute. Blocked by the `lanes > 1` compiler
    error for the loop form; the unrolled form compiles but does not overlap.
  - Little synchronization inside a kernel. Each extra tile costs about 156 µs; each call has about 1 ms fixed
    overhead.
  - Large shapes with weights resident. The reload variant crashes the runtime.
- Realistic next targets: a few more TOPS per core through better compiler output and the instance and width
  settings already found. Roughly 10 to 20 TOPS per core needs working overlap, which depends on the `lanes`
  fix or Axelera support.

## Blocker attempt: tile loop as a registered operator (compiler, measured)

- Goal: the `lanes > 1` error is raised for loop-form tile loops in prim_func. The SDK's pipelined templates
  use `with A.tiles(...)` inside an operator's implementation. Test: a custom ScriptOp (`tiled_op.py`) whose
  implementation runs the tile loop.
- Results:
  1. Inside `with A.script():` the `with A.tiles` form is rejected: "Invalid context expression".
  2. Without `A.script()` (the SDK's pattern, where the implementation runs as ordinary Python), the with-form is
     accepted and the operator compiles past the context check.
  3. The next error: the input tile's shape is a symbolic `min(...)` expression, and the output tile is a
     fixed 128 rows. The compiler's shape check rejects the mismatch.
- Conclusion: the operator route removes the context error but not the symbolic-tile-shape error, which is the
  same family as `lanes > 1`. The SDK avoids it with clamped ramps and `where=` predicates in its pipelined
  templates; the loop form of those predicates is what fails for us. Closing this needs either a compiler fix
  or matching the SDK's template exactly, which we haven't achieved.

## Shared-memory multi-core (SDK configuration, measured)

The hardware reference describes a shared-memory mode with `aipu_cores_used: 4`. The SDK's deployment code
compiles with `MulticoreMode.BATCH`, `aipu_cores_used=N`, and `resources_used=0.25 N`. Measured with ResNet-18
(ONNX, random calibration, batch 1):

| Build | Runtime config | Images/s |
|---|---|---|
| 1 core | 1 sub-device, 1 core | 671 |
| 4 cores (4 kernels) | 1 sub-device, 4 cores | fails to create executor |
| 4 cores (4 kernels) | 4 sub-devices, 4 cores | 540 |

- The batch-4 build is rejected by the compiler: "Only batch sizes of 1 supported".
- The four-core build produces four kernels, one per core, so the split is real.
- With this model the four-core split is slower than one core. Per-core kernels add cross-core synchronization.
- The card stayed healthy after every run.


## Corrections: TOPS convention and hardware checks (2026-11-24)

### TOPS convention
- The SDK hardware reference defines 1 TOPS = 10¹² multiply-accumulate (MAC) operations per second.
- Earlier figures in this document used 2 ops per MAC (2 × MACs × FPS). Under the SDK convention they are about
  half the values shown. Restated:

  | Result | Earlier (2 ops/MAC) | SDK convention (MAC/s) |
  |---|---|---|
  | Best GEMM, 2 to 3 instances | 1.73 | 0.87 |
  | ResNet-50, 1 instance, 1 core | 3.1 | 1.6 |
  | ResNet-50, 4 instances | 8.9 | 4.5 |
  | ResNet-50, vendor published (M.2, 1756 FPS) | 14.4 | 7.2 |
  | Peak, 4 cores at 800 MHz | 214 (marketing) | 214 |

- The percentages of peak in earlier sections were computed against the 2-op figures. Against the SDK-convention
  peak they are about half as large. The conclusions are unchanged: utilization is far below peak.

### Per-clock peak (derived)
- 214 TOPS ÷ 4 cores ÷ 800 MHz ≈ 66,900 MACs per clock per core, close to 2¹⁶.
- Derived from the marketing peak; not measured. If the peak assumed 2 ops per MAC, the per-clock figure would be
  about 33,400 MACs.

### Hardware checks on the 113m card

| Hardware reference claim | Card | Status |
|---|---|---|
| 4 AI cores | 4 (`axcmd --get-aicore-count`) | matches |
| 0 PVE cores | 0 (`axcmd --get-pvecore-count`) | matches |
| 800 MHz clock | 800 MHz (`axdevice`) | matches |
| 4 GB DDR (PCIe / M.2) | 8 GiB (`axdevice`) | mismatch for the 113m M.2 |
| PCIe Gen 3 ×4 | slot max 8 GT/s ×4; link trained at 8 GT/s ×2 | link width is ×2 |
| ~214 INT8 TOPS | not directly measurable | unverified |

- On-chip memory sizes are not in the reference. Measured limits: about 4 MiB L1 (compiler constraint error) and
  an 8 MiB L2 workspace (runtime allocation error).

## Compiler pipelining options on the 1×1-conv GEMM (measured)

- Options set: `pipeline_spatial_tiles`, `pipeline_channel_tiles`, `imc_double_buffer_pipeline`,
  `use_list_scheduler`. The compiler accepted all four.
- Shape M=4096, K=512, N=512, one instance, double buffering on:

  | Build | Aggregate TOPS (ops convention) |
  |---|---|
  | Default config | 1.084 |
  | Pipelining options | 1.128 |

- The task streams are identical: same task counts by type (START_DMA 19, POLL_DMA 19, INSTRUCTION 3,
  COMMAND 26, TOKEN 5) and the same size (2240 bytes).
- So these options did not change the schedule for this shape. The 4% difference is within run-to-run noise.
- Likely reason: the shape gives the pipeliner no tiles to overlap. Not tested at shapes that tile.

## Compiler pipelining on a shape that tiles (measured)

- Shape M=4096, K=2048, N=2048, 1×1 conv, one instance, double buffering on.
- The same four pipelining options as before (`pipeline_spatial_tiles`, `pipeline_channel_tiles`,
  `imc_double_buffer_pipeline`, `use_list_scheduler`).

  | Build | Task stream | Aggregate TOPS (ops convention) |
  |---|---|---|
  | Default | 93,056 bytes; COMMAND 12,599, TOKEN 21, START_DMA 71 | 3.11 |
  | Pipelined | 99,328 bytes; COMMAND 289, TOKEN 83, START_DMA 55 | 3.71 |

- The task streams differ in structure, unlike the smaller 4096×512×512 shape, where the options changed
  nothing. The pipelined build has about 40 times fewer commands and more tokens, consistent with a loop
  that overlaps tiles and synchronizes between them.
- Result: about 19% faster. The first measured gain from the compiler's pipelining.
- Card healthy before and after.

### Pipelining with more instances (same shape, measured)

| Instances | Default (TOPS, ops convention) | Pipelined (TOPS) |
|---|---|---|
| 1 | 3.11 | 3.71 |
| 2 | 4.19 | 4.35 |
| 4 | 4.62 | 4.61 |

- The pipelining gain shrinks as instances increase: about 19% at one instance, about 4% at two, none at four.
- Running several instances already overlaps the transfers and matmuls, so the compiler's pipelining adds little on top.
- Card healthy before and after.

### Pipelining options on ResNet-18 (one core, measured)

| Build | Task stream | Images/s (1 instance, 1 core) |
|---|---|---|
| Default | 34,752 bytes; COMMAND 6,401 | 666 |
| Pipelined | 57,408 bytes; COMMAND 24,835 | 579 |

- The pipelined build is about 13% slower here, the reverse of the large GEMM, where it was about 19% faster.
- Its stream is larger (more commands and tokens), so the extra synchronization and command overhead outweighs
  any overlap for this model.
- The pipelining options help a large tiled matmul but hurt a convolutional network. Enable them per model, not globally.
- Card healthy before and after.

### Power limiter (measured)

`axdevice --set-power-limit` enables a limiter that reduces AI-core performance when power exceeds the limit.
The card was at 8.5 W. Measured with the limiter on, then off, then back on:

| Workload | 8.5 W limit | Limiter off | Restored 8.5 W |
|---|---|---|---|
| 4096×2048×2048 GEMM, 1 instance (TOPS, ops convention) | 3.11 | 3.27 | not repeated |
| ResNet-18, 1 core (images/s) | 673 | 827 | 645 |

- With the limiter off, ResNet-18 is about 20% faster (827 against 645 to 673 with it on).
- The GEMM gain is about 5%, within the range where the earlier runs varied.
- Not measured: temperature and software throttling with the limiter off. Check before running sustained workloads.
- The limiter was restored to 8.5 W after the test. Card healthy.

### Scheduler options on the large GEMM (measured, one instance)

Shape 4096×2048×2048, 1×1 conv, double buffering on.

| Build | Options | TOPS (ops convention) |
|---|---|---|
| Default | none | 3.11 |
| List scheduler only | `use_list_scheduler` | 3.06 |
| IFDW grouping only | `group_ifdw_tasks` | 3.10 |
| Pipelining options (earlier) | four pipelining options | 3.71 |
| Combined | pipelining + list scheduler + IFDW grouping | 3.79 |

- Neither option helps alone: both are within run-to-run noise of the default.
- Adding them to the pipelining options gives about 2% more, which is within noise.
- The list-scheduler build's packed stream uses a different chunk layout from the default build. My stream
  decoder doesn't parse it, so its task counts were not compared.
- Card healthy before and after.

### Taller pipelined GEMM (8192×2048×2048, measured)

- Build: pipelining options plus scheduler and IFDW grouping. Compiled without error.
- One instance: 4.23 TOPS (ops convention), up from 3.71 for the 4096-row pipelined build.
- Four instances: 4.93 TOPS, the best measured so far. Only 20 calls per run, so treat as approximate.
- Per core this is about 0.6 MAC-TOPS against an ideal of about 53 MAC-TOPS per core: roughly 1% of ideal.
- Health check passed before and after.

### Overlap with waits placed after both launches (device test, measured)

- Form (AxelerScript guide section 12): for each tile, launch the copy for the next tile and the matmul for this
  tile, each with an id, then wait for both after both launches. Two L1 slots; the first copy waited before the first matmul.
- Compiled with working waits: no `DMA POLL` or `Cannot determine core` errors. The earlier wait failures came from
  the wait placement, not from missing support.
- Correct: 99.16% of values match the reference (same as the non-overlapped build).
- Speed: 2316 µs kernel time at 1024×512×512, 8 tiles. The non-overlapped build measured 2355 µs. No gain.
- Reading: the waits are correct, but the copy and the matmul still don't run at the same time in a way that shortens
  the kernel. The per-tile cost is not mainly the DMA wait.
- Card healthy before and after.

### Repeated resident-operand matmul, scale and cast once (device test, measured)

- Setup: 256×512×512. Activations and weights are copied to L1 once. Then R matmuls run with no DMA between them,
  and only the last repetition applies scale and cast. 200 timed calls per point, one compile per point.
- Matrix-execute commands in each stream: 1, 2, 4, 8 for R = 1, 2, 4, 8, so the compiler kept every matmul.
- Kernel median: R=1 806 µs, R=2 813 µs, R=4 918 µs, R=8 982 µs.
- Per extra matmul: about 25 µs. The full-array estimate is about 1.25 µs for this matmul, so the matmul alone runs
  near 5% of the derived per-core peak.
- Compared with scale and cast on every repetition (shared output, 808 µs to 1189 µs at R=8): the scale and cast
  add about 29 µs per repetition, leaving about 25 µs for the matmul itself.
- Reading: with operands already resident and no DMA in the loop, the matmul is still far below full-array rate.
  DMA starvation does not explain this part. Either the input feed from L1 to the matrix array or the matrix
  program uses only part of the array. Not yet separated.

### Width versus chunk count (device test, measured)

- 256×512×1024 (two 512-column chunks): R=1 1089 µs, R=4 1242 µs. About 51 µs per extra matmul, about 25 µs per chunk.
- 256×512×512 (one chunk): about 25 µs per extra matmul, as above.
- 128×512×512 (half the rows): R=1 756 µs, R=4 696 µs. The slope is inside the noise, so the per-row cost is not measured.
- Reading: cost scales with 512-column chunks, matching the chunk-count model. Single-call time is dominated by a fixed
  cost of roughly 700 µs.

### Row scaling of the resident-operand matmul (device test, measured)

- K=512, N=512, R=1, 4, 8 per row count, scale and cast once, 200 timed calls per point, one compile per point.
- Per extra matmul (slope R=1 to R=8): 128 rows 16.4 µs, 256 rows 25.2 µs, 512 rows 33.2 µs.
- Four times the rows costs about two times the time, so the cost is not linear in rows. A linear fit gives roughly 11 µs
  fixed per matmul plus about 0.044 µs per row.
- Full-array estimates for these shapes: 0.63 µs (128 rows), 1.25 µs (256), 2.5 µs (512). The measured times are 20 to 26 times above
  those estimates.
- Reading: the fixed part per matmul plus the sublinear row term point to per-matmul overhead in the matrix program, not a
  full array at 5% duty. This is not yet a proof of partial array use; it is consistent with it.

### K scaling of the resident-operand matmul (device test, measured)

- M=256, N=512, R=1 and 8, scale and cast once, 200 timed calls per point, two compiles per K.
- Per extra matmul (slope from R=1 to R=8), run a and run b: K=256 3.3 and 10.7 µs; K=512 17.1 and 17.8 µs;
  K=1024 23.5 and 21.5 µs.
- Slope rises about 3× for a 4× increase in K, so it is not proportional to K.
- Noise: the K=512 point was 25.2 µs in an earlier single run and 17.5 µs here, and the R=1 baseline moved by about 50 µs
  between sessions. Treat each slope as ±5 µs. The fixed-cost fit from the row-scaling section is not robust to this drift.
- Full-array estimate for K=256 is 0.63 µs; the measured slopes are above it for every K.

### Row scaling repeated with three compiles per point (device test, measured)

- K=512, N=512, R=1 and 8, scale and cast once, 200 timed calls per point, three compiles per row count.
- Per extra matmul (slope from R=1 to R=8), compiles a, b, c: 128 rows 9.2, 8.7, 8.1 µs (mean 8.7, tight);
  256 rows 25.1, 1.6, 25.5 µs (run b is an outlier at R=8); 512 rows 26.6, 26.2, 40.2 µs (run c is an outlier at R=8).
- Using the median of three: 128 rows 8.7 µs, 256 rows 25.1 µs, 512 rows 26.6 µs. The 256 and 512 points are not
  clearly separable with three compiles, so the earlier sublinear fit (33 µs at 512 rows) should not be relied on.
- Reading: the 128-row point is reproducible. Above 256 rows the cost plateaus within this data, which fits a fixed
  per-call cost at larger shapes rather than a linear row term. Still above the full-array estimate at every size.

### Other row counts, three compiles per point (device test, measured)

- K=512, N=512, R=1 and 8, scale and cast once, 200 timed calls per point, three compiles per row count.
- Per extra matmul (slope from R=1 to R=8), compiles a, b, c, with medians:
  64 rows 6.5, -1.7, 0.2 µs (median 0.2); 128 rows 8.7 (median, from the previous section);
  192 rows -0.2, -0.8, 8.5 µs (median -0.2); 256 rows median 25.1; 384 rows 17.2, 26.5, 9.0 µs (median 17.2);
  512 rows median 26.6; 768 rows 24.9, 29.0, 44.4 µs (median 29.0).
- The R=1 baseline also moves by 50 to 100 µs between compiles at 64 and 192 rows, which swamps the slope there.
- Reading: no clean row law. The median slope is near zero at 64 and 192 rows, and rises to about 25 to 29 µs at 256 to
  768 rows, with 384 rows in between. Per-compile noise is about ±15 µs, so these medians are not a fitted curve.
- Full-array estimate at 768 rows is 3.75 µs per matmul; the measured median is about 29 µs, roughly 8 times above it.

### Single wide op versus repeated narrow ops (device test, measured)

- K=512, R=1, scale and cast once, 200 timed calls per point, three compiles per point. Medians:
  - 256 rows, 512 columns (one chunk): about 806 µs (earlier section).
  - 256 rows, 1024 columns: 1146 µs. 256 rows, 2048 columns: 1717 µs. 256 rows, 4096 columns: 3159 µs.
  - 64 rows, 2048 columns: 1276 µs.
- Each extra 512-column chunk inside one op adds about 300 to 360 µs. Repeating a one-chunk op adds only about 25 µs
  per repetition (the earlier repeat test). So per-op fixed cost is not what dominates a single wide call.
- At 2048 columns, 192 extra rows add about 440 µs, about 2.3 µs per row.
- Reading: the repeat test excludes the weight load and most of the data movement, so it measured matmul compute only.
  In a single call, data movement (weights scaling with K×N, activations and outputs scaling with M×N) takes most of the
  time. The matmul compute for 256×512×2048 would be about 5 µs at full-array rate and about 100 µs at the repeat-test
  rate, against 1717 µs measured in a single call.

### Compile-only stream counts for single matmuls (read-only, measured)

- 256×512×N, one op, compiled without the device. Command counts from the packed stream:

| N (columns) | M_MVMPRG (matrix program) | M_IFD_W (weight feed) | M_MVMEXE (execute) | M_DPU | Total tasks |
|---|---|---|---|---|---|
| 512 | 1 | 1 | 1 | 1 | 35 |
| 2048 | 8 | 8 | 2 | 2 | 81 |
| 4096 | 32 | 32 | 8 | 8 | 206 |

- Matrix-program and weight-feed commands grow faster than the column count: 8 at 2048 and 32 at 4096. The execute count
  grows about linearly (2 and 8). So the compiler issues about 4 weight-program loads per execute at these sizes.
- Hypothesis, not verified: most single-call time is weight-program and weight-feed work rather than the execute itself.
  This fits the earlier observation that each extra 512-column chunk adds about 300 µs. Testing it needs a device run
  that varies the program count with the total work fixed.

### Equal-work shapes versus program count (device test, measured)

- Three shapes with the same total work (537 MMAC each), K=512, R=1, scale and cast once, 200 timed calls per point,
  three compiles per shape. Matrix programs from the compile-only stream count:

| Shape (M×K×N) | Matrix programs | Executes | Kernel median (three compiles) |
|---|---|---|---|
| 256×512×4096 | 32 | 8 | 3123 µs (3079, 3314, 3123) |
| 512×512×2048 | 16 | 4 | 2335 µs (2335, 2329, 2341) |
| 1024×512×1024 | 8 | 4 | 2169 µs (2166, 2214, 2169) |

- Time falls as the program count falls: about 40 µs per program between 32 and 8, about 21 µs per program between 16 and 8.
- Confounded: M changes from 256 to 1024 across the rows, so activation and output movement also change. This run cannot
  separate program count from row count.
- Reading: at equal work, the time is still about 200 times the full-array estimate (about 10 µs for 537 MMAC).

### Matrix program count versus shape (compile-only, measured)

- Program count depends on every dimension, so shape alone cannot hold it fixed while movement changes:
  - Rows at N=2048, K=512: 64 rows 4 programs, 256 rows 8, 1024 rows 32.
  - Columns at M=256, K=512: 512 cols 1, 1024 cols 2, 2048 cols 8, 4096 cols 32.
  - K at M=256, N=2048: K=256 8 programs, K=512 8, K=1024 16. K=2048 did not compile in this run (likely the 4 MiB L1 limit on
    the 2048×2048 weight block).
- A clean program-count control needs a compiler setting that changes the tiling directly, not a shape change.

### Compiler-setting control: executes doubled, matrix programs fixed (device test, measured)

- Compile-only sweep at 256×512×2048 (K=512, one op), matrix-program and execute counts for each override:
  - default, mvm_utilization_limit 0.5 or 1.0, max_tiling_depth 1, enable_software_pipeline true or false,
    dpu_unroll true: 8 programs, 2 executes.
  - min_tiling_factor 2: 8 programs, 4 executes.
  - enable_auto_tiling false: fails to compile (pre-allocation L1 overflow by 256 bytes).
- Device run, R=1, three compiles each, same session:
  - default: 2061, 2035, 2033 µs (median 2035 µs).
  - min_tiling_factor 2: 2031, 2010, 2056 µs (median 2031 µs).
- Reading: doubling the executes with the program count fixed changed the time by about 4 µs, inside the noise. So
  execute count is not the cost at this size. This supports the program-count hypothesis, but it does not isolate it: no
  setting found so far changes the program count with the executes fixed.
- Drift: the same default shape measured 1717 µs earlier this session and 2035 µs here. Compare only within one session.

### Programs versus executes, interleaved (device test, measured)

- Compile-only: `min_tiling_factor=4` at 256×512×2048 gives 16 programs and 16 executes (the `=2` setting gives 8 and 4,
  the default 8 and 2). Other settings tried (`max_tiling_attempts` 8, `enable_datapath_fusion` false, `enable_icr`,
  `enable_2d_dma` false, `enable_dual_channel_dma` false, `enable_async_scheduling` false, `enable_buffer_promotion`
  false, `enable_memory_paging`) left the count at 8 and 2. `max_tiling_attempts` 1 fails to compile.
- Device, one block, rounds interleaved, three compiles per setting (kernel median):
  - default (8 programs, 2 executes): 1660, 1659, 1664 and 1674, 1705, 1713 µs. Median about 1680 µs.
  - min_tiling_factor 2 (8 programs, 4 executes): 1675, 1688, 1703 µs. Median 1688 µs.
  - min_tiling_factor 4 (16 programs, 16 executes): 1869, 1838, 2094 µs. Median 1869 µs.
- Fit T = a + b·programs + c·executes through the three medians (exact, three points): b about 17 µs per program,
  c about 4 µs per execute, a about 1540 µs. The execute term is within noise; the program term is the one the data
  support. This is still three points with no residual check.
- Drift: the default measured 2035 µs in an earlier block and 1680 µs here. Only compare settings inside one block.

## Open issues for ideal GEMM performance

1. Weight load cost. The fit gives about 17 µs per matrix program, but a weight load has not been timed on its own.
2. Per-row cost breakdown. Input DMA, scale and cast, and output DMA are mixed in the measurements.
3. Matrix unit utilization inside an execute. Unverified, because it needs matrix-program fields on the card.
4. Larger matrix programs. Only settings that increase the program count were found; no control for bigger blocks yet.
5. Compiler limits. The lowering iteration limit blocks 4096³, and the L1 limit caps resident data.
6. Measurement stability. Drift between blocks is about 18%; per-point noise is about ±15 µs.
7. Command fields. INSTRUCTION and TOKEN semantics, the M_IFD_W address meaning, and the M_DPU width rule are unknown; some flag bits crash the device.

### Scale step, combined effect (device test, measured)

- 256×512×2048, one op, interleaved, three compiles each (kernel median):
  - full (multiply, then cast): 1724, 1790, 1789 µs. Median 1789 µs. Matrix programs 8, executes 2.
  - scale skipped (cast of the accumulator only): 1641, 1666, 1595 µs. Median 1641 µs. Matrix programs 4, executes 1.
- Difference about 148 µs (about 8%). Skipping the scale also changes the compiler's tiling (programs 8 to 4, executes 2
  to 1), so this is a combined effect, not a pure scale cost. The program-count fit predicts about 70 µs, so the
  scale step may account for the rest, but the two cannot be separated with these builds.
