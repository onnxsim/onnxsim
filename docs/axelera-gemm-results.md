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
