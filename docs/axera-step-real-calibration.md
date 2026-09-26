# ResNet18 step coverage at a real calibration

The coverage report marked 562 of the step's 1,104 nodes as "conditional":
a template exists, but it only works if the node's calibrated zero points are
in the template's class (for example `x128,y128`). This work predicts
Pulsar2's actual quantization of the step and settles every one of those
nodes.

## Predicting Pulsar2's quantization offline

Pulsar2 never builds the whole step (live-weight Conv, `build_result_*.txt` in
`t6-r18fold`), so `quant_axmodel.json` cannot come from the step itself.
`scripts/axera/step_calibration.py` predicts it instead. It runs the float
graph over the step's own Pulsar2 calibration set (`t6-r18fold/wd`: the first
four steps of a real training trajectory, `make_wd.py`) and applies the rules
below. The rules were read off our template builds and then checked against
**every tensor of all 435 builds with a `quant_axmodel.json`: 2136 of 2136
match** (scale to 1e-4 relative, zero point to ±1, signedness exactly).

- **Asymmetric uint8, MinMax.** Widen the range to include 0, then
  `s = f32((hi - lo) / 255)` and `zp = round(-lo / s)`.
- **Symmetric int8 for MatMul inputs.** A MatMul, Gemm or live-weight Conv
  requantizes each input to `max|x| / 127.5` with zero point 0. The
  producer's own output is int8 only when *every* consumer is such an input.
  A Mul that feeds both a MatMul and a ReduceSum stays uint8 at its own
  range, and the MatMul gets a separate int8 config for it. Probe:
  `mixed_mm_rs`.
- **A passive op whose output feeds only MatMuls, while its input has other
  uses, requantizes.** Its output gets its own symmetric int8 parameters over
  its own range and does not overlap its uint8 input (the `mixed_reshape_mm`
  probe: `Mul -> {ReduceSum, Reshape -> MatMul}`, and the side-output Conv
  builds). This is what makes the 17 MatMul-only Reshapes int8, and the
  rearranged dX kernels too. `assign` splits the passive group there.
- **Passive ops share parameters.** Reshape, Transpose, Squeeze, Slice, Pad,
  Relu and MaxPool outputs are `OVERLAPPED` with their input: one scale and
  zero point over the union of the ranges.
- **A Relu fuses into its producer only when it is the producer's only
  consumer.** Then both take the Relu output's range, so the zero point is 0
  (probes `add_relu`, `mm_relu`). In the step, every Relu input also feeds
  the backward `Greater` mask. So no Relu fuses, and each shares its
  pre-activation zero point (probe `add_relu_gt`).
- **A constant-weight Conv keeps a uint8 activation input.**
  `t_step_attr/ladder` has the conv, conv+relu and conv-relu-conv builds.

`python scripts/axera/step_calibration.py calibrate step.onnx wd/config/*.json
-o calib.json` writes the prediction for the legalized step: live Convs as
per-tap MatMuls and the Gemm as a MatMul, which are the tensor names the
MatMul templates use. It evaluates one node at a time, so memory stays near
the step's real live set of about 1 GiB. A whole-graph onnxruntime session
passed the 16 GiB cap. Committed:
`fixtures/step_calibration/resnet18_step_calibration.json.gz`.

`tinygrad_ax_backend.coverage_report(records, calibration=...)` and the CLI
`coverage --calibration` settle each conditional node with
`plan_at_calibration`:

- zero-point class checks for elementwise, binary, misc and Reshape nodes;
- a live-operand MatMul is recalibrated for real
  (`matmul_record_emit.recalibrate` onto the predicted scales);
- a node computed inside a covered MatMul chain template (its Gather, Mul,
  Reshape, Transpose, Squeeze) counts as covered by that template.

## New zero-point handling (step 2)

| gap at the real calibration | fix | checked |
| --- | --- | --- |
| 17 Relu: shared zero points 86..173, but templates only `x0,y0`/`x128,y128` | `ew.retarget_relu_records`: a Relu writes its zero point as whole words to `0x1b10/0x1eb0/0x1a90` (the Reshape->Relu layout), so the `x128` template moves to any nonzero zero point | 10 native builds (5 step shapes x zero points 169/86): record for record, both directions |
| MaxPool zero point 108, but its template is `x0,y0` | new nonzero template, and `misc.retarget` now moves MaxPool's one shared zero point (`0x1b10`, `0x1a90`) | zp 108 <-> 71 builds, both directions |
| ReduceSum `[1024,9,3136]` with `zp_x = zp_y = 0` | zero-point-0 template; `retarget` keeps unchanged zero points (including 0) instead of refusing | two native builds, both directions |
| Reshape `[1024,28224]->[1024,9,3136]` with zero point 0 | zero-point-0 template (`rre.step_template_zp0`), scale-only retarget | two native builds, both directions |

A zero point of 0 stays its own program everywhere: its zero-point writes
are elided (`axera-zp-register-write-elision`).

## Coverage at the predicted calibration

The current Pulsar-free UOp-to-mcode planner covers **570 of 1,104 nodes**
at this calibration and refuses 534. This includes 42 live broadcast binary
nodes (all `Mul` nodes in this step), which are emitted with a validated
full-shape binary template and expanded at the segment boundary. The
regression test in `tests/test_axera_step_calibration.py` pins this report.

Master reported 120 covered / 562 conditional / 422 refused. Settling the
conditionals against the predicted calibration:

| | covered | conditional | refused |
| --- | ---: | ---: | ---: |
| master's templates, conditionals settled | 395 | 18 | 691 |
| + nodes computed inside covered MatMul chains | 426 | 18 | 660 |
| + the zero-point work above | 446 | 18 | 640 |
| + #1892's Neg and same-bytes ReduceSum templates, settled here | 449 | 18 | 637 |
| + bias-flatten Reshapes as fused ReduceSum chains (below) | **467** | 0 | 637 |

Per op, covered at the predicted calibration, before (master's templates) ->
after:

- Relu 0 -> 17
- MaxPool 0 -> 1
- ReduceSum 42 -> 44 (one via #1892's same-bytes `[16,64,112,112]` equivalent)
- Neg 0 -> 2 (#1892's large-program template, zero points x255,y0)
- Reshape 118 -> 153 (18 via the fused ReduceSum chains below)
- Mul 0 -> 14 (inside MatMul chains)
- Squeeze 0 -> 1 (inside a MatMul chain)

Already at their final counts: Sqrt 42/42, Softmax 3/3, Log 2/2, Cast,
Greater, Less, Gather and Transpose all, and MatMul 24/41.

### The 18 bias-flatten Reshapes

All 18 are Adam bias-gradient flattens, `ReduceSum -> [1,C] -> Reshape ->
[C]`, feeding two Muls (`(1-beta1)*g` and `g*g`). They share the ReduceSum's
quantization (a Reshape is passive), so the question is only what Pulsar2
compiles for the pair. Measured per step `C` (`t_step_rsfused/`), each at two
calibrations with nonzero zero points, `P = ReduceSum -> Reshape` against
`Q = ReduceSum` alone:

| C | ReduceSum input | P vs Q |
| ---: | --- | --- |
| 512, 256 | `[16,1,C,HW]` axes 0,3 | identical records, tables and tail (only the 301-325 segment-0 noise window differs): the Reshape compiles to nothing |
| 1000 | `[16,1000]` axis 0, keepdims | P is 96 decompressed bytes longer |
| 128, 64 | `[16,1,C,HW]` axes 0,3 | a different ReduceSum tiling (`npu_params` 340 vs 480 and 660 vs 720 bytes) |

A rebuild of each differing P and Q reproduces itself record for record, so
the differences are real, not rebuild noise. Either way the fused chain `P` is
the template: `misc_op_record_emit.retarget(op="ReduceSum")` carries every P
(and Q) build to its other calibration record for record in both directions.
They are registered as `ReduceSum:<shape>:axes..:k..:reshape<C>` in the
misc-op index, and `extract_step_ops` gives each flatten a `fused_key`.
`Reshape_475` needs no chain build: `ReduceSum_474`'s same-bytes equivalent
`[16,64,112,112]` over axes (0,2,3) already writes the flattened `[64]`. All 18
settle as covered at the predicted calibration (every zero point nonzero) and
emit.

Two caveats. In a real step compile the ReduceSum and its flatten are one
program, so a runner should emit the chain template for the pair rather than
the ReduceSum's standalone template (which differs for C = 1000, 128, 64).
And the flatten's rank-1 consumers can't compile as written: `Reshape -> Mul`
at `[C]` fails Pulsar2's Mul tiler (`TileFailException ... tuple index out of
range`), as #1869 found for rank-1 binary ops; those Muls run at `[1,C]`.

## What is still refused, and why

- **Binary ops with two live same-shape inputs:** 63 Mul, 42 Sub, 43 Div
  and 101 Add. Their real zero points are spread (for example
  `x142,y155,z130`), but the templates only exist at `{0, 128}` classes.
  #1869 found that asymmetric zero points compile Add to a different
  program, so this needs the binary-op emitter to move zero points, the way
  Relu's now can.
- **Constant or broadcast operands:** 330 Mul, and some Add/Sub/Div.
- **Conv:** 8 were conditional and now fail
  `recalibrate`: `zero point goes between zero and nonzero`. The templates
  were built with a nonnegative input (zero point 0), but in the step the
  input is a Relu output that shares its pre-activation's nonzero zero
  point. They need templates built with a signed input range. The other 12
  Conv have no template yet.
- **Gemm:** two zero-point roles tie at the predicted scales, so
  `recalibrate` refuses rather than guess.
- **17 Reshapes whose output feeds only a MatMul:** they are int8 in the
  real compile, while every Reshape template is uint8. These are the inputs
  of the MatMul chains, so the cleaner fix is to extend those chains by one
  Reshape.
- **ReduceMean:** its output (`pool1_fwd`) feeds only MatMuls, so it is
  int8.

Reproduce: `tests/test_axera_step_calibration.py` (no device, no Pulsar2; the
rule checks run on committed probe graphs, their ranges and their
`quant_axmodel.json`).
