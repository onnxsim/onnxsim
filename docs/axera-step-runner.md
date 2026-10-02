# Running the ResNet18 training step with its covered nodes on the AX650

`coverage_report` counts the nodes of the training step
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, 1,104 nodes, batch 16,
distillation loss plus Adam) that our emitters can produce at the step's
predicted calibration (`docs/axera-step-real-calibration.md`). This note is
about actually running the step that way, on the card, and measuring what
comes out.

## The pieces

- `scripts/axera/vm/axcl_batch_runner.c` is a persistent runner built inside
  the LXD guest. One `axclInit`, one device context, and line commands on
  stdin: `LOAD` a model, `RUN` it on input files, `UNLOAD` it. Tensors travel
  as raw files on the guest's virtiofs share (`/mnt/share`), not through the
  pipe. Loading a model takes about 13 ms, and running the Relu health model
  about 25 ms end to end. `axcl_run_model` through the harness instead costs
  a process, a runtime init and an `lxc file push` per call.
- `scripts/axera/axcl_session.py` is the host side (`AXSession`). It holds
  `/tmp/axcl-device.lock` for the session, so other device work queues behind
  it. It also provides `health_check`: the native `x128,y128` Relu template
  on [-1, 1] data, which must come back within 1 LSB.
- `scripts/axera/step_runner.py` turns the step into segments and runs them.
  - Every node `plan_at_calibration` covers becomes an NPU segment: its
    template, retargeted to the predicted calibration.
  - A covered live-operand MatMul runs its whole chain template.
  - Greater/Less -> Cast runs as its pair template.
  - A bias-flatten Reshape runs fused with its ReduceSum (#1908).
  - Same-shape float32 Add at exactly `[16, 1000]` uses its checked-in
    Pulsar2 FP32 template. The indexed FP32 corpus also covers the exact
    Mul/Div input and output shape signatures of the training graph's former
    binary fallbacks, including broadcasted initializer operands. These
    templates have no quantize/dequantize boundary; each signature is
    selected by op and exact input/output shapes and was compared with NumPy
    on AXCL. Uncaptured signatures retain their existing path.
  - The ResNet18 loss-tail one-hot `Mul_20` and broadcast `Sub_24` use
    checked-in native templates at their exact measured calibration. The
    Sub template expands `[16, 1]` to `[16, 1000]`; both routes are refused
    if their tensor scales or zero points change.
  - `Div_0`, `Div_1`, and `Div_34` divide `[16, 1000]` tensors by the
    constant `2`. Pulsar2 folds each into a native multiply program; the
    runner selects one of three checked-in templates by the exact measured
    input/output zero point and scale. Other Div constants or calibrations
    still require a general native template before they leave the host.
  - The crop-mask normalization `Div_453` uses a checked-in FP32
    `Expand -> Max(count, 1) -> Div` model. Float execution has counts in
    `[1,9]`; quantized mask inputs can produce empty positions, where both
    numerator and count are zero. The explicit clamp makes those positions
    zero and avoids both implicit broadcast in AxDiv and `0/0`. The exact
    `[1024,9,3136]` template passed an AX8850 VM test including zero counts;
    the step runner also verifies this segment against its matching safe-div
    simulation.
  - The checked-in exact S16 overrides are enabled by default; a different
    JSON file can be selected with `--precision-overrides overrides.json`.
    They select only templates whose full scale and
    zero-point tuple matches the fixture index; a mismatch raises instead of
    silently reverting to ORT. The current range-matched fixtures cover
    `Mul_5`, `Mul_11`, `Mul_25`, `Mul_33`, `Mul_46`, `Mul_68`, `Mul_91`, and
    `Mul_113`, `Mul_148`, `Mul_170`, `Mul_193`, `Mul_215`, `Mul_250`,
    `Mul_272`, `Mul_295`, `Mul_317`, `Mul_352`, `Mul_374`, `Mul_397`, and
    `Mul_419` at the committed ResNet18 calibration. Twelve shared-scale S16
    templates cover 21 matrix-shaped `lr * tensor` optimizer multiplies. Five
    more templates cover all 21 rank-1 optimizer multiplies by packing vectors as
    `[1,N]` for Pulsar and restoring the logical `[N]` output shape in the
    runner. A range-matched S16 Sub template also covers `Sub_32`, expanding
    its `[16,1]` input to `[16,1000]` on-device. With supported overrides,
    planning goes from 193 to 126 host nodes: 62 Mul and one Sub override are
    selected (Mul fallbacks fall from 146 to 84; Sub from 3 to 0), and two
    singleton Div nodes are guarded/folded. The two large `1-Cast(Greater)` /
    `1-Cast(Less)` masks now use swapped inclusive comparisons on-device; this
    preserves equality behavior for finite inputs.
    Two scalar constant-first Div nodes also fold when `batch_size` is exactly
    the calibrated singleton; runtime guards preserve the ORT path if it differs
    (`Div` fallbacks 44 to 42). All 17 shared optimizer templates and the
    `Sub_32` and both comparison-complement templates passed AXCL VM checks.
    Calibration metadata is in
    `fixtures/binary_op_precision/index.json`.
  - Everything else runs on the host, one node at a time in onnxruntime.
  - Segments exchange float32 tensors. Quantized templates quantize and
    dequantize internally; the FP32 Add template consumes and returns float32
    directly.
  - Every NPU segment is also simulated on the host on the same inputs:
    inputs fake-quantized at the predicted parameters, the float ops, and
    the output fake-quantized for quantized templates; unquantized templates
    run directly in float. The runner then compares device against simulation
    in output LSBs, and both against float.
  - With `--health-every 1`, the default, the health model runs after every
    device segment, so a bad model is caught at the segment that broke the
    card.
- `tinygrad_ax_backend.register_ax_device()` makes `"AX"` a tinygrad device.
  - `AXAllocator` gives host-staged buffers. AXCL binds device memory to one
    loaded model's IO, so buffers are staged per call.
  - `AXProgram` loads `AXCompiler` output, or any emitted `.axmodel`, into
    the session and runs it on the card.
  - `tests/test_axera_step_runner.py` runs a Relu (an `AXCompiler` request)
    and the step's fc dX MatMul chain through tinygrad `Buffer("AX", ...)`
    objects on the device.
  - tinygrad's own scheduler cannot target the device: no renderer maps
    UOps to templates.

## What running it found

All runs are on 2026-09-24, on the AX8850 through `axcl-vm`, with the step's
first batch (`step1_ref.pkl`) and the committed calibration
(`fixtures/step_calibration/resnet18_step_calibration.json.gz`).

1. **Emitters wrote MCode without updating its size, and that took the
   card down.**
   - `misc_op_record_emit.emit_model`, `reshape_record_emit.emit_step_reshape`,
     `elementwise_scale_emit._write` and `ElementwiseScaleEdit` replaced the
     MCode initializer's bytes but not its `dims`.
   - A zero-point move re-encodes the stream to a different length:
     ReduceSum_62, `[16,1,512,4608]`, went from 120,024 to 120,088 bytes.
   - The runtime reads the size from `dims`. A standalone load fails with
     `0x80300709`. In the first whole-step run it wedged the card instead:
     firmware dead, full stack reload needed.
   - All four emitters now go through `step_recalibrate.with_mcode`.
     ReduceSum_62 then matches its simulation exactly.
   - The native template of the same shape ran fine throughout, which is
     what isolated the fault to the emitted model.
2. **The step Reshape templates compute a Relu (#1891).**
   - The step Reshape templates are `Reshape -> Relu` builds, and the
     emitted model applies the Relu.
   - Reshape_42, on a signed input: 40% of the elements are off by up to
     164 LSB, and the error relative to float is 79%.
   - 100 of the 119 Reshape segments take an input whose calibration range
     is negative. The runner marks them `unsafe` and keeps them on the host.
     The other 19 run on the NPU and match.
   - So `coverage_report`'s Reshape count overstates what is correct by
     those 100 nodes.
   - Fixed since: signed Reshapes now take `Reshape -> Identity` templates,
     which are exact on the device (`docs/axera-reshape-signed-templates.md`).
3. **Gather templates kept their own calibration.**
   - `GatherIndexEdit` moves the indices but leaves the template's
     `(1/s, s, zp)`. On the device, Gather_57 differed from float at 26% of
     its elements.
   - A Gather is passive, so the runner now retargets it like a Reshape.
     `reshape_record_emit.retarget_scale` gained `zp_regs=GATHER_ZP_REGS`,
     since a Gather writes only 0x1b10/0x1a90. It also now accepts an `s`
     lane that is not exactly `f32(1 / (1/s))`, as a Gather template stores
     141.6667 next to 0.00705882.
   - All 27 Gather segments are now within 1.83 LSB of their simulation.
4. **Five segments disagree with their simulation. Validation pass:
   `--mode npu`, safe segments only, 275 segments, 366 nodes.**
   - 270 of 275 are within 2 LSB.
   - The failures:

     | Segment | What the device did |
     |---|---|
     | `Add_976` (x128,y128,z128) | up to 115 LSB off |
     | `Reshape_371` (fused chain `ReduceSum:16x1x64x3136:axes0,3:k0:reshape64` at s_x = 1.0e-5, s_y = 4.6e-3) | up to 144 LSB off |
     | `Reshape_416` (same chain, s_x = 1.4e-5, s_y = 5.9e-3) | device fault `0x8030070C`; the health check passed right after |
     | `Log_3`, `Log_10` | 17-89 LSB off: the emitter did not move Log's `s_y` lanes. #1910 fixed this; on current master both are exact (0 LSB) |

   - Rechecked on current master: Add_976 is still 115 LSB off, and
     Reshape_371 is still 111 LSB off. Both are emitter defects to chase:
     the Add `x128,y128,z128` retarget at these scales, and the
     `...:reshape64` fused chain at an `s_y/s_x` of about 460.
     `--validated <report>` keeps failing segments on the host in later
     runs.
   - The `...:reshape64` failures were `relayout_segment` shifting a scalar
     header word (0x1000 at byte 72) when the stream re-encoded 32 bytes
     shorter; fixed, and exact on the device
     (`docs/axera-reshape-signed-templates.md`). Add_976 is still open.
5. **The loss head cannot be uint8.** The device matches the simulation
   there; the damage is from quantization itself.
   - With Softmax on the NPU, the student's probabilities, which are about
     1e-3 for 1,000 classes, quantize at `s = 3.5e-3` to mostly 0. The
     `softmax - target` gradient is then gone.
   - With every safe segment on the NPU, the median gradient cosine to
     float is -0.62. In a run with only the loss-head/misc segments on the
     NPU, keeping just Softmax on the host brings it back to 0.998.
6. **The optimizer update cannot be uint8 either.**
   - With Adam's `m`, `v`, `sqrt(v)` and `m / (sqrt(v) + eps)` on the NPU,
     the weight update is off by a factor of about 10^6.
   - A per-tensor MinMax range cannot hold `v` of about 1e-8 next to its
     maximum.
   - `--host-optimizer` keeps every node after the gradients in float.

## The numbers

"Grad" is each weight's gradient, the tensor the Adam update consumes,
against a float run of the same batch. "Update" is `w' - w` against the float
step. Medians are over the 42 trainable tensors. Device time is the sum of
`axclrtEngineExecute` wall times inside the guest.

| Run | NPU segments / nodes | Loss (float 17.0582) | Grad cos median / min | Grad rel. err median | Update cos median |
|---|---|---|---|---|---|
| every safe segment (validation pass) | 275 / 366 | 15.864 | -0.62 / -0.78 | 4.4 | -0.11 |
| minus failing segments and the loss head | 265 / 354 | 17.091 | 0.9935 / 0.19 | 0.128 | 0.46 |
| ... and the optimizer on the host | **180 / 269** | **17.091** | **0.9935 / 0.19** | **0.128** | **0.80** |

- In the last run, every one of the 180 device segments was within 2 LSB
  of its simulation, and the health check passed before the first and
  after each one (362 device runs in all).
- The two lowest gradient cosines are the 7x7 stem's weight and bias, at
  0.19. Their backward path is the longest chain of quantized ops; every
  other gradient has a cosine of 0.986 or more.
- The update cosine stays below the gradient cosine because Adam's first
  step is close to `lr * sign(g)`: a small gradient error near zero flips
  the sign.

Time for one step, the last configuration:

| | Seconds |
|---|---|
| NPU execute, all 180 segments | 1.13 |
| Wall, including host ops, per-segment simulation and float checks, file IO and 182 health checks | 63 |
| Host-only float run of the same step | 5.1 |

This is a correctness harness, not a speedup. The NPU nodes are 24% of the
graph, and the Convs, most binary ops and all of the optimizer are still on
the host.

## Reproduce

```sh
PY=/mnt/data/cache/claude-work/tg-venv/bin/python   # numpy, onnx, onnxruntime, tinygrad fork
cd scripts/axera
systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  $PY step_runner.py --mode npu --out /path/validate.json
systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  $PY step_runner.py --mode npu --validated /path/validate.json \
  --exclude '^(Softmax|Log|Neg)_' --host-optimizer --out /path/final.json
```

- `--mode float` checks the runner itself: the loss matches the reference
  to 1e-7 relative, and every gradient has cosine 1.0.
- `--mode sim` runs the whole step on the simulation. It produces NaN,
  because the fake-quantized `Div`/`Log` see zeros that the device
  saturates.
- The peak host memory is 10 GB, from the node-by-node evaluation and the
  per-segment float checks.

## FP32 binary device profile

`profile_fp32_binary.py` profiles the checked-in unquantized Add model through
the persistent AXCL guest runner. The guest reports `axclrtEngineExecute`
time; host round-trip time also includes input staging, file exchange, and the
runner protocol.

On 2026-09-28, the AX8850 in `axcl-vm` ran the `[16,1000]` FP32 Add template
for 500 measured executions after 20 warmups. The output matched NumPy Add
exactly. Device execution averaged **296.6 us** (median 291 us, p95 319 us);
host round-trip averaged 2.28 ms (median 2.22 ms, p95 2.55 ms). This is the
template's device latency, not a full training-step speedup measurement.

Reproduce from the repository root:

```sh
AXCL_LXD_VM=axcl-vm python scripts/axera/profile_fp32_binary.py \
  --warmup 20 --runs 500 --output fp32-add-profile.json
```

Before the shape expansion below, the plan left 126 nodes on ONNX Runtime,
all binary operations: 84 Mul and 42 Div.

### Captures for the remaining shapes

`capture_fp32_binaries.py` captured the 126 former host Mul/Div nodes as 55
unique `(op, input shapes, output shape)` templates (37 Mul, 18 Div). Each
build sets that binary op's `layer_configs.data_type` to `FP32`, uses the
compiled model's FP32 IO, and is rejected unless a real AXCL execution exactly
matches NumPy. The profile record for each capture is in
`fixtures/fp32_binary/index.json`; larger tensors use fewer repetitions to
limit input staging. Their mean device times range from 253 us to 48.1 ms for
Mul and 267 us to 12.3 ms for Div, reflecting tensor size and broadcast work.

Paired AX8850 measurements also show FP32 Mul beating the matching S16
templates for `[128,128,3,3]` (590 vs 673 us median) and `[1000,512]`
(661 vs 720 us). These signatures are already selected as `fp32_binary` by
the current plan. The 100-run measurements, including p95 and fixture paths,
are recorded in `fixtures/fp32_binary/device_comparisons.json`. This is a
kernel-time comparison, not a whole-step speedup claim; apparent wins with
different broadcast input shapes were excluded.

For the native scalar-broadcast Mul segments, shape-correct profiling found
nine signatures where FP32 was at least 10% faster than the S16 template.
The planner now chooses their FP32 templates with operands reordered to match
the captured model IO, replacing 18 S16 nodes. Measured median speedups range
from 1.14x to 2.16x over 40 paired runs; marginal results below the 10%
threshold remain on S16. Details are in
`fixtures/fp32_binary/native_mul_speed_profiles.json`.

The updated plan now assigns all 1,104 graph nodes to NPU segments with zero
ONNX Runtime fallback. The full training graph ran on AX8850: all 127 FP32
binary segments had zero errors, zero LSB difference, and zero difference
from the float op; health checks passed before and after the run. The full
graph took 2.74 s of AXCL engine time and 171 s wall time including host
simulation, checking, tensor staging, and health checks. Its loss was 17.555
versus 17.058 for the float reference, and the median gradient cosine was
0.0. That remaining training accuracy loss comes from the other quantized
segments; making the binary fallbacks FP32 does not fix it.

Capture and profile on the VM-backed device with:

```sh
AXCL_LXD_VM=axcl-vm python scripts/axera/capture_fp32_binaries.py \
  --refresh --runs 30
```

## Retargeting inaccurate covered binaries

The default FP32 capture list is built from nodes that the current planner
would leave on the host. `--nodes NAME,...` additionally captures named
binary nodes even when another native emitter claims them; these explicit
entries are selected by the planner and retain the step's calibrated input
and output quantization boundaries. This lets a validated FP32 arithmetic
kernel replace a failing quantized route without changing its surrounding
calibration contract.

On 2026-09-29, an AX8850 replay identified 27 distinct signatures among the
failing binary segments. Pulsar2 built and device-validated 22 directly. Its
calibrator rejected the five scalar-first Mul signatures because rank-0 input
shapes fail in calibration, so the capture path now represents scalar inputs
as `[1]` (broadcast-equivalent) in the compiled template and reshapes the
single runtime value accordingly. All five then built and matched FP32
exactly. The 27 targeted templates cover 30 named step nodes, including
`Add_976` and the five scalar Mul nodes; a focused run of those five Mul nodes
on real step data passed at 0 LSB with no NaN updates.

The next full replay, with the five scalar templates included, ran 914 NPU
segments (1,105 graph-node executions) with no device errors and no planner
host nodes. It measured 175 exact FP32 binary segments; the strict 2-LSB gate
still rejected 33 MatMul chains and 21 quantized elementwise segments. The
training loss was 16.729 vs 17.058 float, and median gradient cosine was
-0.577, so this is not yet a validated full-training result. Report:
`/tmp/axera-fp32-qio-full-next.json`.

Two high-impact elementwise failures (`Mul_705`, `Add_730`) were separately
captured as FP32 templates and matched the calibrated simulation at 0 LSB on
real step inputs; selecting just these two restored loss to within 2e-6 of
float, with median update relative error 2.3e-6. They are deliberately not
marked `prefer_fp32_nodes`: the measured end-to-end device path took 96/133 ms
per segment, versus 14/32 ms for the local host simulation, and the capture
round trips were 28/82 ms. Thus they are useful accuracy probes, but do not
meet the faster-than-host criterion for replacing fallback. MatMul-chain
calibration/emission and faster native arithmetic remain the priority.

## Stable softmax-gradient rewrite (`--stable-softmax-grad`)

`step_runner.py --stable-softmax-grad CALIB.json` applies
`rewrite_softmax_ratio_gradients` to the step, patches the per-node records,
and regenerates the calibration for the rewritten graph over the same 4-step
calibration set (cached in `CALIB.json`; about a minute the first time). The
rewrite replaces `p * (a/p - sum(a/p*p))` with `p * (a - p*sum(a))`, removing
the quantized `0/0` behind the non-finite `Log_10`/`Div_21` segments.

On 2026-09-29, on the AX8850 with `--host-optimizer`: 328 NPU segments, zero
device errors, health 0 LSB before and after, no runtime fallbacks. Median
gradient cosine against the float reference rose from -0.577 to **0.839** with
no NaN (simulation: 0.918); median update cosine 0.53 (was -0.0001 with 7
NaN updates). The loss is unchanged (16.729 vs 17.058, forward path). The
remaining gradient error is in 34 MatMul-chain segments that fail the 2-LSB
gate. The default run without the flag is unchanged.

```sh
AXCL_LXD_VM=axcl-vm $PY step_runner.py --mode npu --host-optimizer \
  --stable-softmax-grad /path/stable-calib.json --out /path/stable.json
```

## Adam update: `--fp32-optimizer`

In simulation the NaN updates come from the uint8 `Sqrt` and `Add(eps)`
segments in front of each optimizer `Div`: `sqrt(v)` is about 1e-5 against a
tensor max of 0.43, so the 1e-8 eps is lost and 509,119 of 512,000 elements
of the 1000x512 weight quantize to 0 (0/0 -> NaN). `--fp32-optimizer` never
gives optimizer nodes a quantized template: each runs as an FP32 binary
template when one is captured for its exact shapes, else on the host in float.
Simulation gives 0 NaN updates and the same update cosine as
`--host-optimizer` (0.64), with 263 optimizer nodes (Sqrt, Add-eps, scalar
Mul) still on the host for lack of FP32 templates. Not yet run on the device.

## 16-bit MatMul pilot

`pilot_matmul_u16.py` builds a bare live-operand `MatMul(x, w)` with Pulsar2
7.0-lite at U8, U16 and S16 (`quant.layer_configs` with
`op_types: ["MatMul"]`). Pulsar2 accepts U16 and S16 for MatMul. On the AX8850,
[1,64,128]x[1,128,64] has relative error against float of 1.76e-2 at U8 and
6.8e-5 at U16/S16 (about 260x lower), at comparable latency
(0.38 ms U8, 0.28 ms U16). The 16-bit axmodel is larger (5,919 vs 4,343 bytes)
but uses the same scale-lane roles as 8-bit, plus two fixed lane constants
(256.0 and 1.0, `matmul_record_emit.FIXED_LANES`) and an `npu_params`
multiplier lane of `256 * s_x * s_w / s_y` (the `mult256` role). With those,
`matmul_record_emit.recalibrate` moves one U16 build onto another exactly in
both directions (records and params), and the emitted model matches the
native held-out build bit for bit on the AX8850 (max diff 0.0, 6.5e-5 relative
error against float). This is a bare MatMul: the step's Gather/Reshape chains
and int8-symmetric input rules at 16 bits are still to be checked.

## 16-bit MatMul chains (`--u16-matmul`)

`step_runner.py --u16-matmul REGEX` rebuilds the `matmul_chain` segments whose
name matches with Pulsar2 `layer_configs` U16 (`u16_chain.py`), calibrated on
the reference batch's real tensors (one axmodel per segment, cached in
`--u16-cache-dir`). Segments still pass and return float32; a segment passes
when it is within `U16_MAX_REL` (5e-3) of the float chain. Two Pulsar2
details, both measured:

- Every op type of the chain is set to U16, and the bias `Add` is also named
  in a `layer_names` entry.
- A `Transpose`/`Reshape` that ends the chain makes Pulsar2 quantize the whole
  output path to 8 bits (the forward Conv chains stayed at 2.2e-2 error).
  `chain_model` cuts those ops off and the runner applies them on the host as
  the segment's `output_transform`; the forward `stage2_conv2` chain went from
  2.2e-2 to 3.5e-4.

On real step data the bare backward MatMuls are about 240x closer to float at
U16 (median 2-7e-2 -> 1-6e-4) for about 2.5x the device time (100 ms -> 251 ms
summed over 17 templates; `dX_MatMul_54` 14 -> 75 ms).

AX8850 replay with `--stable-softmax-grad --host-optimizer` and the 20 forward
Convs plus eight small backward MatMuls at 16-bit (28 segments; median
float error 7e-4; health 0 LSB, no runtime fallback): median gradient cosine
**0.974** (8-bit: 0.839; MatMul chains on the host in float: 0.994), median
update cosine 0.72, loss 16.660 vs 17.058 float. Not yet at 16 bits:
`conv0_fwd` (its Pulsar2 build exceeds the 30 minute timeout), `dense0_fwd`
(1.4e-2 from float, so it ran as float), and the large backward chains
(`TMPDIR` must point at disk: `/tmp` is tmpfs and the calibration tars of the
biggest chains overflow it).

### Where the remaining error is (16-bit MatMuls, AX8850 replay)

- `--exact-fp32-io` lets the FP32 binary segments pass float instead of
  re-quantizing to the step's 8-bit boundaries (11 segments, max error 0). It
  leaves the gradients where they were (median cosine 0.971).
- `--u16-kinds` extends `--u16-matmul` to other segment kinds. The forward
  loss error comes only from the `misc` segments (Softmax, Log, Neg,
  ReduceSum): simulation with `misc` on the host in float gives loss 17.073
  against 17.058, and no other kind moves it. At U16 on the device Softmax
  is 4e-4 from float, Log 1.4e-4 and ReduceSum exact.
- The remaining gradient error is the still-8-bit backward MatMul chains
  (34 segments, 4-11% each).

### `misc` at 16 bits closes the loss gap

AX8850 replay with `--stable-softmax-grad --host-optimizer --exact-fp32-io
--u16-kinds matmul_chain,misc` (61 segments at U16: the 20 forward Convs
except `conv0_fwd`, eight small backward MatMuls, and the Softmax, Log, Neg and
26 ReduceSum segments): loss **17.004 against 17.058** float (16.660 before),
median gradient cosine **0.977**, median update cosine 0.75; zero device
errors, health 0 LSB, no runtime fallback. Five 16-bit segments failed their
gate or build (`ReduceSum_460` exceeded the 30 minute build timeout) and ran as
float. The rest of the gradient error is the 34 backward MatMul chains still at
8 bits (12 of them beyond 2 LSB of their simulation).

### All MatMul and Conv chains at 16 bits, `--u16-splits`

`--u16-splits 4,16` retries a 16-bit chain that does not compile at the step's
batch (Pulsar2 build cap `U16_BUILD_TIMEOUT`, default 1800 s) at a smaller
batch: the chain is rebuilt with every batch-leading input shrunk by the
factor, calibrated on all the chunks together, and the segment runs the small
model once per chunk (`batch_split`). `conv0_fwd` does not build at batch 16
(it exceeded 3 hours) but builds at batch 4 and runs four times. A failed build
leaves a `.failed` marker in `--u16-cache-dir`, so it is not retried unless
`U16_RETRY_FAILED` is set.

AX8850 replay, `--stable-softmax-grad --host-optimizer --exact-fp32-io
--u16-kinds matmul_chain,misc --u16-splits 4` (95 segments at U16, none of the
step's MatMul or Conv chains left at 8 bits; zero device errors, health 0
LSB): median gradient cosine **0.9974** (was 0.839 at 8 bits), minimum
0.021 (was 0), median update cosine 0.82, loss 17.004 against 17.058 float.
`MatMul_121` (bare MatMul, 1.8e-4) and `conv0_fwd` (9e-4) were the last two
8-bit chains and accounted for most of the remaining gradient error. Eight
16-bit segments miss their gate and run as float: `dense0_fwd` (1.3e-2),
`Softmax_9`, `Log_10` (zero probabilities), `MatMul_471` (9.8e-3), three
`ReduceSum` (0.6-1.2e-2), and `MatMul_325` (input shape mismatch). `ReduceSum_460`
does not build within 40 minutes even at batch 1.

### The Adam update on the NPU (`--fp32-optimizer-chains`)

Pulsar2 applies `layer_configs` FP32 to `Sqrt` as well (it is not in the
documented FP32 list): a real optimizer `Sqrt` is 3.0e-2 from float at U8, 5.5e-4
at U16 and 2.1e-5 at FP32. An FP32 layer does not quantize, so one build
serves every node with the same operator, attributes and input shapes.
`--fp32-optimizer-chains` (with `--fp32-optimizer`) builds each optimizer node
that has no captured FP32 template as a one-node FP32 model with every input,
constants included, as a float graph input; 263 nodes (Sqrt, +eps, scalar Mul,
Sub, ...) took 65 builds.

AX8850 replay, `--stable-softmax-grad --fp32-optimizer --fp32-optimizer-chains
--exact-fp32-io --u16-kinds matmul_chain,misc --u16-splits 4` (no
`--host-optimizer`): **1,098 of 1,102 graph nodes on the NPU** (907 segments:
95 U16, 327 FP32 binary, 263 FP32 chains), zero device errors, health 0 LSB
before and after, no runtime fallback, no NaN. Gradient cosine 0.9974, update
cosine 0.82 (the same as with the update on the host, so the optimizer numerics
equal float; Adam's normalization amplifies small gradient errors), loss 17.004
against 17.058. The three nodes still planned on the host are two Muls whose
zero points match no template class and `Sub_32` (no template at its shape).
Eight 16-bit segments still miss their gate and run as float on the host.

### The last 16-bit misses

`--u16-margin REGEX` calibrates the matching segments on their data and a copy
scaled by 1.3 (`--u16-margin-factor`): the replay's inputs come from upstream
16-bit segments and can leave the float run's range. It fixed `dense0_fwd`,
`Softmax_9` and two `ReduceSum` segments, and made `MatMul_471` and
`ReduceSum_472` (the stem convolution's weight and bias gradients) worse, so
those are not listed. A 16-bit model built at the full batch also has to reset
the plan's `batch_split`: `MatMul_325` had an 8-bit template built at batch 4 and
failed with a chunk-sized input until it did. `--u16-fp32 REGEX` builds the
matching segments with FP32 layers instead of U16 (isolated on the float
inputs: `ReduceSum_368` 8e-8, `MatMul_471` 7e-7).

With those, AX8850 replay (1,098 of 1,102 nodes in 907 segments, health 0 LSB,
no NaN, gradient cosine 0.9973): only `Log_10` (a zero probability), `MatMul_471`
and `ReduceSum_472` still miss their gate and run as float.

### Everything on the NPU

`--fp32-refused` builds the nodes the plan refuses (two Muls whose zero
points match no template class, and `Sub_32`, which has no template at its
shape) as one-node Pulsar2 FP32 models, like `--fp32-optimizer-chains`. With
`--u16-fp32 '^(MatMul_471|ReduceSum_472)$'` the stem convolution's weight and
bias gradients build with FP32 layers (`ReduceSum_472` exact, `MatMul_471`
4.9e-3 from float).

AX8850 replay (`--stable-softmax-grad --fp32-optimizer --fp32-optimizer-chains
--fp32-refused --exact-fp32-io --u16-kinds matmul_chain,misc --u16-splits 4
--u16-margin ... --u16-fp32 ...`): **1,101 of 1,102 graph nodes in 910 NPU
segments, none left on the host by the plan**; health 0 LSB before and after,
no runtime fallback, no NaN, 5.1 s of engine time. Median gradient cosine
0.9973 (minimum 0.015), median update cosine 0.82, loss 17.004 against 17.058
float. Segments by kind: 95 U16, 327 FP32 binary, 266 FP32 single-node, 35
elementwise, 81 reshape, 22 gather, 19 binary-precision, 18 reducesum, 17
relu, 17 compare/cast, plus small ones. One segment fails its gate at runtime:
`Log_10` (the 16-bit `Softmax_9` outputs exact zeros where float has tiny
positive probabilities), and runs as float on the host.

Costs to know: the 16-bit MatMul chains take about 2.5x the device time of
their 8-bit templates, and calibration uses the reference batch the replay
runs on, so a training loop needs per-step recalibration (the record emitter
covers a bare MatMul, not these chains yet).

### Latency: the NPU is fast, the staging is not

`--no-check --health-every 0` times the replay with no per-segment simulation,
float check or health check. AX8850, whole step (1,101 nodes in 910 segments),
AXCL VM: **58.4 s wall against 6.6 s for the host-only float run of the same
step**, about 9x slower. `axclrtEngineExecute` totals only **4.1 s over 1,152
runs** (3.6 ms per run), below the host total; the other ~54 s is per-segment
overhead of 63 ms on average. `profile_session_overhead.py` times one 16-bit
segment (`stage2_conv2_fwd`, 12.9 MB in and 6.4 MB out): engine 2.4 ms, model
load 5.3 ms, input write 4.1 ms, and 58.5 ms for the whole `run` call. Every
segment stages its tensors through the file share to the VM (about 350 MB/s)
and loads and unloads its model per call, so the step is a correctness harness
and not a speedup. Keeping intermediates on the device (the session's
`resident_pairs` mechanism) or compiling the step as one graph would leave about
the engine time, which is 1.6x under the host run now and roughly doubles for
the 16-bit chains.

### Per-step recalibration of the 16-bit chains

The 16-bit chains are built on the reference batch, but a training step's
tensor ranges move every step. `probe_u16_recalibrate.py` builds a real step
node at U16 at two calibrations (the second with every input scaled by a
different factor, so every scale and zero point moves) and checks
`matmul_record_emit.check` both ways. A bare MatMul (`dX_MatMul_240`) and the
forward Conv chain (Transpose, Slice, Reshape, MatMul, bias Add; stage2_conv2)
recalibrate exactly, with zero record and `npu_params` differences in both
directions. No Pulsar2 build is needed to move a chain to a new calibration.

`u16_chain.predict_scales16` gives the scales Pulsar2 would assign, from the
chain's tensor ranges on the new step's data and a template's
`quant_axmodel.json`: a tensor that is (or shares a quantization with) a live
MatMul operand is symmetric int16 (`max|x| / 32767.5`); every other tensor is
unsigned 16-bit over its range widened to include 0 (`f32((hi - lo) / 65535)`,
zero point `round(-lo / scale)`); tensors Pulsar2 marks OVERLAPPED (Transpose,
Slice, Reshape) share their dominator's quantization over the group's union
range. Prediction matches every tensor of both builds at both calibrations.
Recalibrating template A onto the predicted B scales reproduces the native B
build with zero differences, and on the AX8850 the emitted model's output is
bit-identical to the native B build's (and differs from template A's).
`tests/test_axera_matmul_record_emit.py` covers both chains in both directions.

Scope: this covers the MatMul and Conv chains (62 of the 95 16-bit segments).
The Softmax, Log, Neg and ReduceSum 16-bit segments need their own record
roles. The ranges of a chain's intermediates (its MatMul output) depend on
the step's data, so a loop applies them with delayed scaling (the previous
step's ranges with headroom), as `--u16-margin` does for one step.

### Keeping tensors on the device (`--resident`)

The guest runner (`vm/axcl_batch_runner.c`) now has a device-side tensor store:
`TPUT`/`TGET`/`TDEL`/`TCLEAR` and `RUNT`, which takes a model's inputs from named
device tensors and stores its outputs under names by device-to-device copy, so
tensors pass between models without crossing the VM boundary (`AXSession.tput`,
`tget`, `tdel`, `run_t`; `AXSession` rebuilds the runner when its source
changes). `--mode npu --no-check --resident` runs every segment that needs no
host work this way: a `ResidentEnv` holds host arrays and `DeviceTensor`s, a
device segment reads the raw entry, and a tensor is downloaded only when a host
op or a staged segment reads it. The device holds at least 6 GiB of tensors.

The 16-bit chains' trailing Transpose (cut off to keep the chain's output at 16
bits) now runs as its own Transpose-only model: a Transpose model is
bit-exact on the NPU (max error 0.0 at U8, U16 and FP32; 5.7 ms for
`[16,112,112,64]`), so it adds no error. Small loaded models are kept loaded and
shared by every segment with the same blob.

AX8850, whole step, `--no-check --health-every 0`: **38.8 s resident against
58.4 s staged** (host-only float: 6.6 s), and **all 42 gradients are bit-identical**
to the staged run. Per-command time left: staged `RUN` 12.0 s (49 segments in
296 calls: scalar-broadcast inputs and tiled mask products that need host
broadcasting), `RUNT` 8.6 s (4.1 s of it engine), model load 4.7 s (516 loads of
the large chains), tensor upload 3.9 s and download 3.1 s (1.5 GB each; the
weights and optimizer state would stay on the device across steps). The next
steps are device-side broadcast for the staged segments, binding the tensor
store's buffers to the model I/O instead of copying, and keeping the state on the
device across steps.

`RUNT` binds the named tensors' device buffers directly as the model's input and
output buffers instead of copying them (`axclrtEngineSetInputBufferByIndex` /
`SetOutputBufferByIndex`; a plain `RUN` rebinds the model's own buffers first).
That takes the `RUNT` time from 8.6 s to 4.5 s over the step, at the 4.1 s of
engine time, and the gradients stay bit-identical to the staged run. (The
wall-clock figure above was measured before this change; the next measurement is
taken with the machine otherwise idle.)

### FP32 nodes instead of the staged 8-bit templates (`--fp32-elementwise`)

The 49 segments that still staged in resident mode were the planner's 8-bit
`elementwise`, `binary_precision` and `mul_mask_exact` templates for plain
Mul/Add/Sub/Div nodes: they tile, pack or broadcast their inputs on the host.
`--fp32-elementwise elementwise,binary_precision,mul_mask_exact` swaps them for
Pulsar2 FP32 one-node models (exact, no layout transforms, device-resident; 55
segments, 24 builds, one per (operator, shapes) signature).

AX8850, whole step, `--no-check --health-every 0 --resident --fp32-elementwise ...`
(with the Pulsar2 rebuild still running on the host CPU): **20.4 s wall**, down
from 58.4 s staged and 38.8 s resident (host-only float: 6.6 s). Segments still
on the staged path: 7. Gradient cosine 0.9969, update cosine 0.845 (was 0.823),
loss 17.039 against 17.058 float (was 17.004); the 8-bit mask products were a
small error source. Time left by command: model load 4.9 s (509 loads of the
large chains), `RUNT` 4.6 s (the 4.1 s engine floor), staged `RUN` 3.2 s (11 calls),
tensor upload 1.9 s and download 1.3 s, unload 1.0 s.

### Lazy model I/O and models kept loaded (`LOADT`, `--repeat`)

`RUNT` binds tensor-store buffers, so a model's own I/O buffers (tens of MB for
the large chains) were allocated at load and never used. `LOADT` loads a model
without them; a plain `RUN` allocates them on first use. A lazily loaded
model costs only its weights and code (all 62 MB of 16-bit builds), so the
runner now keeps loaded models across runs (`--repeat N` runs the step N times
and reports each wall time).

AX8850, whole step, `--no-check --health-every 0 --resident --fp32-elementwise ...
--repeat 3` (gradients bit-identical to the previous resident run):

| | wall |
|---|---|
| first run (loads the models) | 15.2 s |
| steady state, runs 2 and 3 | **11.5 s, 11.7 s** |
| staged (before this work) | 58.4 s |
| host-only float | 6.6 s |

The steady state is 1.75x the host-only run; 4.1 s of it is `axclrtEngineExecute`.
What is left per step (3 runs): `RUNT` 4.8 s, staged `RUN` 2.4 s (about 10
segments), tensor upload 1.6 s and download 1.0 s (the weights and optimizer state
would stay on the device across steps), `TDEL` 0.5 s, and the host's own Python.

### Four real training steps: recalibration policies and the loss drift

The calibration dataset holds the first four steps of a real training
trajectory (every graph input per step; step 0 is the reference batch).
`--u16-recal static|delayed|exact --steps 0,1,2,3` runs them on the AX8850 with the
16-bit MatMul/Conv chains recalibrated per step from `quant_axmodel.json`
sidecars (`u16_chain.predict_scales16` and `matmul_record_emit.recalibrate`; no
Pulsar2 build): static = the step-0 templates, delayed = the previous step's
ranges times `--u16-margin-factor`, exact = the step's own ranges.
`--probe-nodes` prints a node's relative error against the float run.

| policy | gradient cosine (median), steps 1 / 2 / 3 |
|---|---|
| static | 0.9953 / 0.9962 / 0.9975 |
| delayed | 0.9950 / 0.9964 / 0.9987 |
| exact | 0.9970 / 0.9964 / 0.9985 |

39 of the 62 chains moved with no device error or NaN; delayed scaling is as good
as exact, and the step-0 templates already hold for these four steps. The 22
refused chains are the forward 3x3 and strided Convs: `value ... matches no scale
formula` at register `0x1ef0`. A 3x3 chain concatenates its nine taps and
requantizes the asymmetric weight (zero point 32043) to a symmetric one; the
first `0x1ef0` record is `int32(float32(zw * sw / swcat * 2^15))` (five builds at
different calibrations; the low bits are float32 granularity, all multiples
of 64). The second record is not yet decoded.

The loss was the real problem: within 0.02 of float at step 0 but 0.7 to 0.8
low at steps 1 to 3, under every policy. Bisecting on step 1 (kinds on the
NPU, the rest in float): data movement alone is exact; adding `relu` or the
binary/elementwise kinds stays within 0.01; the MatMul/Conv chains alone are
fine (+0.010); the 16-bit `misc` segments alone give -0.82. Within them, the
loss `Neg` and `ReduceSum` segments (`Neg_8`, `Neg_14`, `ReduceSum_6`,
`ReduceSum_12`) are the cause: a 16-bit build is calibrated on step 0's value
range, and the next step's loss term lies outside it and saturates. Building
those four with FP32 layers (`--u16-fp32 '^(Neg_8|Neg_14|ReduceSum_6|ReduceSum_12)$'`)
removes the dependence on a calibrated range:

| step | loss | float | error |
|---|---|---|---|
| 0 | 17.041 | 17.058 | -0.017 |
| 1 | 17.344 | 17.339 | +0.006 |
| 2 | 17.526 | 17.496 | +0.029 |
| 3 | 17.587 | 17.555 | +0.032 |

(gradient cosines unchanged at 0.995 to 0.997.) Any tensor that is a loss
term or a sum should be FP32 or recalibrated per step; a range calibrated on one
batch is not a bound on the next.

### Training on the device with the state resident (`--train-steps`)

`--train-steps 0,1,2,3` trains across dataset steps with the weights and Adam
`m` and `v` (126 tensors, 140 MB) staying on the device: a step's updated state
tensors are the next step's state inputs (`run(keep_device=...)` returns them as
`DeviceTensor`s; the output name prefix alternates per step so a state tensor and
its update never share a buffer, and the previous step's state is freed after
use). Only the 7 per-step inputs (data, teacher logits, labels, learning rate,
...: 9.8 MB) go in and the loss comes out. `--train-validate` also runs a float
chain (its own state carried the same way) and a float step on the device's own
state.

AX8850, final configuration (16-bit chains, FP32 optimizer, FP32 elementwise
and loss reductions, resident), no validation downloads: **10.4 s per step
steady state** (14.8 s for the first, which loads the models), from 11.5 s with
the state staged; host-only float is 6.6 s. Validated over the four real steps:

| step | loss (device) | float chain | float at the device's state | gradient cosine at the device's state | gradient cosine vs float chain |
|---|---|---|---|---|---|
| 0 | 17.041 | 17.058 | 17.058 | 0.99687 | 0.99687 |
| 1 | 17.377 | 17.339 | 17.367 | 0.99605 | 0.94498 |
| 2 | 17.540 | 17.496 | 17.502 | 0.99622 | 0.84259 |
| 3 | 17.454 | 17.553 | 17.429 | 0.99795 | 0.68508 |

At the device's own state the gradients stay at 0.996 to 0.998 and the loss
follows the float loss at that state, so the device computation stays accurate
across steps and the state is carried correctly. The drop against the float chain
is trajectory divergence: Adam's first updates are close to `lr * sign(g)`, so
gradient errors flip the sign of small-gradient elements (update cosine 0.85 at
step 0) and the two chains' weights part (largest weight relative difference
1.6% after four steps), after which they see different gradients. A float
implementation with different rounding diverges the same way.

### The second `0x1ef0` record of the 3x3 Conv chains (partial)

For the refused forward 3x3 Convs the second `0x1ef0` record, over five builds at
different calibrations, is `-round(zb * sb / sy * 2^14)` when the MatMul and output
zero points are 0 (two builds, exact: `zb`, `sb` the bias's zero point and scale,
`sy` the output scale). With nonzero zero points an extra term appears that is not
linear in the zero points (its coefficient is 7.4 in one pair of builds and 1.0 in
another, which suggests saturating arithmetic), so these chains are not yet
recalibrated and keep their template scales.

### Closing the gap to the host: device Expand, NaN guards, profile

Steady-state profile of the 10.4 s step: `RUNT` 4.9 s (4.1 s of it engine), staged
`RUN` 2.4 s, `TPUT` 1.1 s, `TDEL` 0.5 s (877 calls), `TGET` 0.5 s, the host's
Python 1.0 s. The staged `RUN` was the max-pool backward's crop mask: `Less_447`
and `Greater_444` (compare a `[1024,9,3136]` window tensor with its per-window
maximum, `[1024,1,3136]`) and `Div_453` (`safe_masked_div`), each moving about
115 MB through the VM per step.

- `safe_masked_div` has no host transform and was only excluded from the resident
  path out of caution; `nan_guard` (a host check of a segment's inputs for NaN,
  written for the old quantized segments) is skipped in resident mode, since it
  would download every input and a NaN still shows in the loss and gradients.
  Steady state 10.4 s to 9.2 s, losses identical.
- A broadcast input (the `[1024,1,3136]` maximum for a model that wants
  `[1024,9,3136]`) is expanded on the device by a one-op Expand model: exact on the
  NPU (max error 0.0, 7.7 ms for 115 MB). Built once per shape pair
  (`u16_chain.expand_model`, `StepRunner._expand_blob`).

AX8850, final configuration, `--train-steps 0,1,2,3,0,1,2,3`: **6.6 to 6.7 s per
step steady state**, the same as the host-only float step (6.6 s), with the
losses identical to the 10.4 s runs (max difference 0.0). Only `conv0_fwd`
(four batch chunks) is still on the staged path, 0.2 s per step.

### Pipelined commands

The guest runner executes commands in order, so a `RUNT` or `TDEL` need not wait for
its reply before the next command is sent. `AXSession` sends them without waiting
(`_cmd_async`) and reads the replies at the next synchronous command, at `sync()`,
or when 256 are outstanding; an `ERR` reply is raised then. Of the steady-state step
(6.9 s: `RUNT` 5.1 s over 928 calls for 4.1 s of engine, `TDEL` 0.6 s over 883
calls, host Python 0.5 s), what is left is the engine plus about 0.2 s each of
upload, download and staged `conv0_fwd`.

AX8850, final configuration, `--train-steps 0,1,2,3,0,1,2,3`: **5.8 s per step
steady state** against 6.6 s for the host-only float step (the 4.1 s of engine time
is the floor; 4.6 s of the step is the host waiting for the device), with the losses
identical to the unpipelined runs (max difference 0.0).

### The forward 3x3 Conv chains recalibrate (all 61 chains)

The 22 chains refused in the first four-step run (the forward 3x3 and strided Convs)
fail at register `0x1ef0`. Building stage2_conv0's forward chain at 11 calibrations
(`variants_probe.py`-style sweeps of the weight scale and shift and of the input
offset) showed the records are the existing 8-bit roles, with 16-bit details:

- the first `0x1ef0` record is `rqoff`, the requantize of the asymmetric weight
  into the symmetric `wcat`: `-int(f32(f32(zw) * (f32(sw) / f32(swcat))) * 32768)`,
  shift word `0x8f`;
- the second is `zpoff` of the fused bias Add, `trunc((zy - zmm * f32(smm/sy) - zb *
  f32(sb/sy)) * 2^q)`, with shift word `0x0f` or `0x0e`. The only change was the tie
  rule: a ratio of **exactly 1** keeps `q = 15` (the MatMul output and the biased
  sum share a scale in the step's real templates), so `_add` now takes the smallest
  `k` with a ratio `> 1` instead of `>= 1`.

Tensors that share a scale tie their roles in a template, so `recalibrate` resolves
a tie by the located Add: `s` and `ratio` lanes in terms of the Add's output (its
output scale, x over it) and the MatMul's multiplier lane (`mult`, `mult256`) in terms
of its input. In a 16-bit build the four lanes at `0x0fd0..0x1000` are the constant
1.0, so a tied `ratio` that also evaluates to 1.0 must not move them. A new `zp16`
role covers the 16-bit zero points an asymmetric activation leaves in `npu_params`
(a table of `z | z << 16` words ending in a half-word, `zp16h`). 34 of 34 accepted
ordered pairs across the 11 builds are exact in records and `npu_params`; the others
cross the zero/nonzero zero-point boundary and are refused by design (Pulsar2 emits a
different record count there). `tests/test_axera_matmul_record_emit.py` covers ten
pairs and the refusal.

In the multi-step driver, a constant tensor in a template (a gather mask) has a scale
but no range to predict from and keeps the template's scale; this `KeyError` was
refusing 19 more chains. AX8850, four real steps, **all 61 MatMul/Conv chains
recalibrated at every step, none refused**:

| policy | step | gradient cosine (median) | loss | float | error |
|---|---|---|---|---|---|
| delayed | 1 / 2 / 3 | 0.9946 / 0.9954 / 0.9977 | 17.337 / 17.478 / 17.559 | 17.339 / 17.496 / 17.555 | -0.001 / -0.019 / +0.004 |
| exact | 1 / 2 / 3 | 0.9955 / 0.9965 / 0.9980 | 17.338 / 17.504 / 17.551 | 17.339 / 17.496 / 17.555 | -0.001 / +0.008 / -0.004 |

Delayed (the previous step's ranges times 1.3) matches exact, and both match the
static step-0 templates (0.9953 / 0.9962 / 0.9975): on these four steps recalibration
costs nothing and the machinery is in place for ranges that move more.

### Longer runs: the static templates drift, recalibration holds

`--train-steps` over twelve steps (the four real batches three times, the weights
and Adam state evolving on the device), with `--train-validate` comparing the
gradients at the device's own state against a float step on that state:

| n | batch | static: gradient cosine | static: loss error |
|---|---|---|---|
| 0 / 1 / 2 / 3 | 0 / 1 / 2 / 3 | 0.9969 / 0.9961 / 0.9962 / 0.9980 | -0.017 / +0.009 / +0.038 / +0.025 |
| 4 / 5 / 6 / 7 | 0 / 1 / 2 / 3 | 0.977 / 0.945 / 0.913 / 0.573 | +0.024 / -0.005 / -0.018 / +0.005 |
| 8 / 9 / 10 / 11 | 0 / 1 / 2 / 3 | 0.150 / 0.449 / 0.184 / 0.677 | -0.029 / -0.099 / -0.116 / -0.058 |

The loss follows float, but the gradients stop being right: the 16-bit chains are
calibrated on step 0's ranges and, as the state evolves, backward tensors leave
those ranges and clip. The four-step recalibration test could not show this: it
used the dataset's own trajectory, where the ranges barely move.
`--train-recal static|delayed|exact` (with `--train-steps`) moves the 61 chains per
step onto scales predicted from the state the step starts from (`exact`) or from the
previous step's ranges times `--u16-margin-factor` (`delayed`). The ranges come from
a float step on the downloaded state, so this is a measurement, not a fast path.
First eight steps, gradient cosine at the device's state (loss error within 0.01
throughout for the recalibrated runs):

| step n | static | exact | delayed (x1.3) |
|---|---|---|---|
| 4 | 0.977 | 0.994 | 0.989 |
| 5 | 0.945 | 0.996 | 0.996 |
| 6 | 0.913 | 0.996 | 0.994 |
| 7 | 0.573 | 0.989 | 0.977 |

Exact recalibration removes the degradation (the chains all moved, none refused), and
delayed scaling recovers most of it with only the previous step's ranges. A long
run on the device therefore needs per-step ranges for each chain's tensors (the
chain input, the weight, the bias, the MatMul output and the output) without
downloading the state: device-side min/max reductions are the missing piece.
`session.exec_by_tag` attributes engine time to segments: of the 3.85 s per step,
the largest are `MatMul_471` (the stem convolution's weight gradient, FP32: 0.37 s),
the three stage-4 forward Convs (about 0.2 s each), `Gather_457` (0.19 s) and
`MatMul_121` (0.17 s).

### Ranges measured on the device (`--train-recal device`)

A long run needs per-step ranges for the 16-bit chains without downloading the state.
What exists and is verified:

- **A device min/max model** (`u16_chain.amax_model`): the NPU reduces only the last
  axis, so a tensor of any shape is read as `[numel]`, reshaped to `[R, C]` and
  reduced twice (`[R,1]` -> `[1,R]` -> `[1,1]`), FP32, output `[max, min]`. Exact on
  the AX8850 for sizes from 64 to 37.7 M elements (26 ms for the largest); 38 sizes
  cover the step's chains (`build_amax_models.py`).
- **Measurement in the resident path** (`StepRunner._measure`, `collect_amax`): after
  each chain runs, its inputs and output are reduced on the device and the `[1,2]`
  results read back at the end of the step (about 250 tiny reads).
- **`u16_chain.derive_ranges`**: the chain's other tensors from what was measured.
  Tensors Pulsar2 groups into one quantization share the measured member's range;
  a mean (`ReduceMean`) is bounded by its input's range; an Add's unmeasured input
  (the MatMul output) follows the output's factor; the rest scale by the factor of the
  tensors they depend on. **`BAKED` tensors (a gather mask) always keep their template
  scale**: deriving one from the margin changed its scale without re-quantizing its
  baked data and broke the dX chains (0.815 at step 1).
- Offline `check_derive_ranges.py` and `check_delayed_derive.py` compare derived with
  exact scales on later dataset steps.

**What is not solved.** On the dataset's own states, recalibrating the chains with
*derived* ranges (`--u16-recal derived`) is below recalibrating them with *exact*
ranges: at margin 2.0, gradient cosine 0.980 / 0.959 / 0.922 at steps 1 / 2 / 3
against 0.994 / 0.997 / 0.997 (exact ranges at the same margin; at margin 4.0 the
exact-range policy still gives 0.991 / 0.995 / 0.997, so headroom is not the cause).
In the device-driven loop over eight steps the result is 0.936 .. 0.962 at steps
1 .. 5 (static degrades to 0.573 by step 7, exact holds 0.989 to 0.996). What was
ruled out: the record emitter (exact against native builds at 2x, 3x and 0.5x scale
ratios); the margin; the MatMul-output estimate (an interval bound, a factor
estimate, and the *exact* `mm`/`m0` range each gave the same result); the derived
code path (running the `delayed` policy on the derived policy's own blobs reproduces
0.98355 / 0.96100 exactly, so the emitted models carry it). Swapping in the derived
blobs for only the forward Convs reproduces the whole drop at step 1, for stages 2
and 3 (0.992 and 0.988) and not stage 1 or 4, but no single chain explains it, and
the blobs differ from the exact policy's by 1e-5 in scale in a few lanes (zero-point
offsets and multiplier lanes). Perturbing every blob by 1e-4 through the margin moves
the gradient cosine by at most 0.0003. At step 2, nine forward chains' derived
`mm`/`m0` scale is 5 to 18% below the exact one and the dX chains' gather outputs are up
to 2x above it.

So: the on-device measurement is exact and recalibration with exact ranges holds a long
run, but the derived ranges lose 0.01 to 0.07 of gradient cosine against exact ones for a
reason not yet found. A fast long-run path would need either that residual understood or
more tensors measured directly. Debug hooks left in `step_runner.py`: `RECAL_DUMP`
(write each chain's emitted blob and applied scales), `RECAL_LOAD`/`RECAL_LOAD_ONLY`
(run another policy's blobs for the matching chains), `DERIVED_EXACT_MM`,
`--recal-chains`, `--train-recal-diag`.
