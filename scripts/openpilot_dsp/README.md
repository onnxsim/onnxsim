# openpilot's models on the Snapdragon 845 Hexagon V65 cDSP (HVX int8): simulator study

openpilot no longer uses SNPE. At master `027770d` (2026-09-24), `modeld` and `dmonitoringmodeld` compile
their ONNX models with tinygrad (`DEV=QCOM IMAGE=1 FLOAT16=1`, `openpilot/selfdrive/modeld/SConscript`) and
run them on the comma device's Adreno 630 GPU. SNPE's DSP runtime used to run them on the Hexagon DSP.
This directory asks whether that can be done again without SNPE, with our own int8 HVX kernels on the
845's V65 cDSP. The GPU would then be free for other work.

**No 845 device was used.** Every number below is one of:

- **measured on the host**: ONNX Runtime accuracy on real route frames, or hexagon-sim cycles;
- **measured on the device in the route's own logs**: `modelExecutionTime` from the comma device that
  recorded the route;
- **estimated**: marked as such, with the assumption stated.

## Summary

| | driving (`driving_supercombo.onnx`) | driver monitoring (`dmonitoring_model.onnx`) |
|---|---|---|
| Compute | 0.85 GMAC: backbone 0.79 (FastViT-style convs), heads 0.053 (23.5 M params, GEMV-shaped) | 0.41 GMAC: backbone 0.41, heads 0.001 |
| GPU today, from the route's rlog (openpilot 0.10.4, older model) | 29.2-29.7 ms mean, p95 ≤ 30.7 ms | 14.3-14.8 ms mean |
| openpilot CI budget (`test_onroad.py`) | mean < 40 ms, max < 60 ms | mean < 50 ms, max < 300 ms |
| fp16 (deployed precision) vs fp32, held-out segments 8 / 5 | plan lateral 0.9 / 0.4 mm, lead prob p95 5e-4 / 5e-6 | worst probability head 2.3e-4 / 1.9e-4 |
| Best integer config (§6) | **W16A16**: plan 5.7 / 4.5 mm, lead p95 0.009 / 7e-5 (6-12x fp16's error) | **W16A16**: worst head 5.9e-4 / 4.2e-4 (~2.5x fp16), blink agreement ≥ 99.8% |
| Cheaper config (§6) | W8 AdaRound + 10 convs W16, A16: plan 1.4 / 1.7 cm, lead p95 0.017 | W8 AdaRound, A16: worst head 0.006, blink ≥ 99.7% |
| Plain int8 PTQ (§3) | W8A8: **broken** (plan 4.3 m) | W8A8: blink agreement 86-95% |
| Projected V65 backbone latency, 2 HVX threads at 1.0 GHz (assumed), after §7 | W16A16 31.3 ms, cheaper config 19.2 ms, W8A16 17.7 ms, W8A8 10.8 ms | W16A16 18.5 ms, W8A16 10.2 ms, W8A8 5.8 ms |
| Heads, simulated int8/int16 GEMVs streamed from DDR (§8), 1 thread | 2.4 ms (int8 weights, u8) to 8.7 ms (int16, u16); mixed-precision heads 4.9 ms | 0.2-0.3 ms |
| Per frame, both models (§8), 20 Hz budget 50 ms | best-accuracy config 55 ms (over budget); cheaper config 34.6 ms | |
| Upstream tinygrad `DEV=DSP` on these models | fp16 graph: fails to link (`__extendhfsf2`/`__truncsfhf2` undefined). fp32 graph: not tried (see the DM column) | fp32 graph: compiles (730 kernels) and matches ORT to 1.4e-5 under qemu. It is scalar float code: V65 HVX has no float |

**Accuracy sets the cost.** No integer config reaches fp16's own error against fp32: not even with every
weight and activation in 16 bits, and not with the stem in float on top of that (§6). The best integer
config (W16A16) is within 6-12x of fp16 on the driving model and ~2.5x on DM. At the assumed
2 threads x 1 GHz it projects to 31 ms (driving) + 19 ms (DM), after §7's depthwise work, of DSP time, above the GPU's 29.5 + 14.5 ms
in the route's logs. The cheaper configs fit, at 15-40x fp16's error.

## 1. Models and profile (`profile_models.py`)

The models were fetched through the LFS media URL. Their sha256 matches the pointers at `027770d`.

- **driving_supercombo.onnx**: 377 nodes, 30.0 M fp16 params, opset 20.
  - Inputs: `new_img` u8 `[2,6,128,256]` (narrow and wide road camera, YUV420 as 6 channels) plus the
    recurrent queues `state_img_q` u8 `[2,5,6,128,256]`, `state_desire_q` `[100,1,8]` and `state_feat_q`
    `[96,1,512]`, and `desire`/`traffic_convention`/`action_t`.
  - Output: `[1,2576]` float, split by the `output_slices` metadata.
  - Backbone MACs: 0.794 G, of which stem dense 3x3/s2 24→64 is 113 M, 30 pointwise convs 638 M, and
    29 depthwise 3x3/7x7 43 M (FastViT-style reparameterized blocks, tanh-GELU, layer scale).
  - Heads: 82 Gemm + 8 MatMul (one attention block), 53 M MACs over 23.5 M params.
- **dmonitoring_model.onnx**: 201 nodes, 3.6 M fp16 params.
  - Inputs: `input_img` u8 `[1,1440*960]` (warped luma) and `calib` `[1,3]`. Output: fp16 `[1,553]`.
  - Backbone MACs: 0.411 G, of which 3 dense 3x3/s2 convs are 187 M, 28 pointwise 212 M, and
    15 depthwise 12 M.
  - Heads: 53 Gemm, 1.1 M MACs.
- **fp16 sensitivity**: fp16 vs fp32 on real frames is negligible.
  - Driving: plan lateral error 0.9 mm mean, lead probability error 1e-4.
  - DM: probability errors ≤ 2e-4.

## 2. Real inputs (`prepare_inputs.py`, `read_rlog.py`)

- **Frames**: from the public demo route `5beb9b58bd12b691/0000010a--a51155e496` (the route
  `openpilot/tools` uses), fetched through `api.commadotai.com/v1/route/.../files`.
  - Segments 3 and 12 are for calibration; segment 8 is the held-out evaluation set.
  - Each segment is 600 frames at 20 Hz from `fcamera`/`ecamera`/`dcamera.hevc` (1344x760, device `mici`,
    sensor os04c10).
- **Warps**: modeld's own, reimplemented in numpy.
  - Driving: `get_warp_matrix(rpyCalib, intrinsics, bigmodel)` with nearest-neighbour sampling, then
    tinygrad `compile_warp.py`'s 6-channel YUV420 layout.
  - DM: `cam.intrinsics @ inv(dmonitoringmodel_intrinsics)`, border fill 16.
- **Calibration**: `rpyCalib` comes from each segment's `extrinsicsCalibration` in the rlog (pitch 0.164
  rad). With the nominal zero calibration, the narrow camera shows the hood.
- **Driving run**: sequential, one call per frame, with the model's `next_state_*` outputs fed back as
  in modeld. So a quantized model free-runs on its own recurrent state.
  - Fixed inputs: `desire` 0, LHD, `action_t` = `[0.25, 0.55]`. This is a nominal value; the route's
    delays are not read.
- **Timings**: the rlog also carries the device's own `modelExecutionTime`, which is where the GPU
  numbers above come from. That is openpilot 0.10.4, whose driving model predates master's.

## 3. Int8 PTQ accuracy (`quantize.py`, `run_models.py compare`, `sweep_driving_groups.py`)

Setup:
- Quantizer: onnxsim `quantize_full_qdq` (whole-graph QDQ, int8 per-channel weights, int32 bias),
  calibrated on segments 3+12 and evaluated on segment 8 (600 frames).
- ORT runs at the *basic* optimization level, so QDQ is simulated rather than fused.
- References: the fp32 conversion of each model (`run_models.py to-fp32`), compared through openpilot's
  own `Parser`.

**Driver monitoring**, probability heads (the reference blink rate is 11-13% of frames):

| policy | raw cos | L/R blink decision agreement | eyes | sleep |
|---|---:|---:|---:|---:|
| W8A8 minmax | 0.984 | 0.917 / 0.938 | 0.997 | 0.972 |
| W8A8 percentile | 0.993 | 0.805 / 0.840 | 1.0 | 0.972 |
| W8A8 mse | 0.993 | 0.727 / 0.770 | 1.0 | 0.972 |
| **W8A16 minmax** | **0.9996** | **0.988 / 0.993** | 1.0 | 0.972 |

- **Blink is the int8-sensitive output.** A prefix/suffix sweep of which convs get 16-bit activations
  found no small subset that fixes it. onnxsim's greedy `search_activation_precision_for_budget` ended
  with 191 of 201 nodes in float.
- **`sleep_prob` is a calibration-coverage bug, not a precision one.** Its logit is never positive in
  either calibration segment, so its output range is clamped at 0 and every quantized model outputs
  sleep = 0.5 on the 17 frames where the reference says > 0.5. Fix: keep the DM heads float (1.1 M MACs,
  free on the CPU), or calibrate on drowsy-driver data.

**Driving**. Plan is the lateral position error over the 33 plan points. Lead is `lead_prob[0]`; the
reference has a lead in 96% of frames.

| policy | plan lat mean | plan lat @10 s p95 | lead prob mean / p95 error | lead decision agreement | lead x error |
|---|---:|---:|---:|---:|---:|
| fp16 (deployed precision) | 0.001 m | 0.016 m | 0.0001 | 1.000 | 0.02 m |
| W8A8, heads float | 4.30 m | 54.8 m | 0.125 | 0.957 | 55 m |
| **W8A16 backbone, heads float** | **0.069 m** | 0.91 m | 0.034 / 0.19 | 0.938 | 1.8 m |
| mixed A8/A16 backbone (`policy_driving_mixed.json`), heads float | 0.137 m | 2.08 m | 0.051 / 0.24 | 0.910 | 3.0 m |
| int8 weights only, float activations (backbone convs) | 0.060 m | 0.87 m | 0.032 | 0.945 | 1.9 m |
| int8 weights only, float activations (heads) | 0.043 m | 0.62 m | 0.005 | 0.993 | 0.66 m |
| whole graph QDQ incl. heads (u8 or u16) | ≥ 0.25 m | ≥ 5 m | | | 21-64 m |

Findings:
- **uint8 activations break the backbone.** Activations are heavy-tailed: GELU/residual max/std is
  20-60 (e.g. `gelu_17` reaches 193 with std 3.3).
- **Where it breaks.** A window-by-window sweep (only one window uint8, the rest uint16) puts the damage
  in the stem/stage 0 (windows 0-3) and the last stage (windows 12-13). Windows 4-11 tolerate uint8;
  that is the mixed policy.
- **W8A16's remaining error is mostly the int8 weights, not the activations.** Weight-only int8 already
  gives it. Per group, the first 3 convs (the stem's dense 3x3/s2, then dw 3x3/s2, then 1x1) matter
  most: keeping just those three in float roughly halves the plan and lead error.
- **Heads.** Quantizing the heads' activations crushes the concatenated `outputs` tensor (one scale for
  plan metres and probabilities), so the heads' outputs must stay float. The heads' int8 *weights* are
  mostly fine (lead agreement 0.993).

Not tried yet, and the obvious next step: QAT (the weights and data exist only on comma's side), or
AdaRound/GPTQ-style weight rounding (onnxsim has `adaround.py`/`gptq`) for the backbone convs. Also
16-bit weights for the 3 stem convs, which the kernels can run as a 2-pass `vrmpy` like `pw16`.

## 4. V65 HVX kernels on hexagon-sim (`hvx65/`)

- **Toolchain.** The Qualcomm open-access toolchain 19.0.02 no longer accepts `-mv65`, and hexagon-sim
  only models v68+. So `kernels.c` is compiled by upstream **clang-19 `-mcpu=hexagonv65 -mhvx=v65`**
  (the assembler rejects any instruction V65 lacks), linked with the toolchain's v68 standalone
  runtime, and run on **`hexagon-sim -mv68 --timing`**.
- **Timing.** Cycles come from UPCYCLE (SSR.CE enabled) around one kernel call. They are a
  **V68-pipeline proxy** for V65: same ISA subset, and not validated against 845 silicon.
- **Correctness.** Every kernel is bit-exact against a scalar C reference of the same fixed-point spec
  (`./run_sim.sh <op> ... check`).

| kernel | what | measured (single thread) |
|---|---|---|
| `pw` | 1x1 conv, u8 act x s8 weight, `vrmpy(Vu.ub=32 px x 4 ch, Rt.b)`, 8 output channels per activation load, fused Q31 requant, output in the next layer's layout | 40-64 MAC/cycle on real shapes (K ≥ 128), 25-30 at K = 64 (requant-bound) |
| `pw16` | the same with u16 activations (W8A16): two `vrmpy` passes (lo/hi bytes), combined at requant | ~2x `pw` cycles |
| `dwc` | depthwise kxk (3, 5, 7), stride 1/2, channels-last, `vzxt` + `vmpyacc(Ww, Vh, Vh)`, 2 pixels per iteration | 4-15 MAC/cycle (C padded to 128: C = 64/96/192 layers waste lanes) |
| `gelu` | u8→u8 table, 8x `vlut32` | 4.0 bytes/cycle |

The 4-channel `pw` variant (`pw4`) stalls on the `vrmpy` accumulate latency at 29 MAC/cycle.
Depthwise is the weakest kernel: 7.9-9.2 M of driving's 32-39 M cycles for 5% of its MACs. It is the
first thing to optimize, e.g. a 2-pixel packing for C = 64.

**Projection** (`project_latency.py`). Every backbone conv of both models runs through the simulator on
its real shape; the cycles are cached in `sim_cycles_cache.json` and the per-layer rows are in
`results/`. Some parts are estimated:
- the im2col copy for dense kxk convs, at 64 B/cycle;
- u16 depthwise and u16 elementwise ops, at 2x the u8 kernels;
- 1x1-spatial GEMVs, from weight bytes at 8 GB/s.

| model / policy | single-thread Mcycles (pw / dense / dw / eltwise) | 2 threads @ 1.0 GHz |
|---|---:|---:|
| DM W8A8 | 12.2 (5.1 / 4.5 / 1.5 / 0.8) | 6.1 ms |
| DM W8A16 | 21.7 (8.9 / 7.8 / 3.0 / 1.7) | 10.9 ms |
| driving W8A8 | 23.6 (15.4 / 2.1 / 4.6 / 1.4) | 11.8 ms |
| driving mixed A8/A16 | 32.5 (18.2 / 3.8 / 8.0 / 2.4) | 16.2 ms |
| driving W8A16 | 39.5 (23.6 / 3.8 / 9.2 / 2.8) | 19.7 ms |

Assumptions:
- **HVX contexts.** The 845's Hexagon 685 is listed as having "dual HVX"
  ([Notebookcheck](https://www.notebookcheck.net/Qualcomm-Snapdragon-845-SoC-Benchmarks-and-Specs.299446.0.html),
  [GSMArena](https://m.gsmarena.com/qualcomm_full_specs_snapdragon_845-news-28620.php)). So 2 threads,
  with perfect scaling assumed.
- **Clock.** The cDSP clock is **not known here**; 1.0 GHz is a placeholder, and latency scales
  inversely with it.
- **Heads.** Not included. The driving heads are GEMV-shaped over 23.5 M params, so they are bound by
  weight bandwidth: about 3 ms in int8 at 8 GB/s (estimated), or they stay on the CPU/GPU in fp16.

## 5. Upstream tinygrad `DEV=DSP` (`tinygrad_dsp_check.py`)

tinygrad at openpilot's pinned commit `9d0446a` has a DSP backend (`tinygrad/runtime/ops_dsp.py`):
- it compiles `--target=hexagon -mcpu=hexagonv65 -mhvx=v65 -nostdlib`;
- on the device it talks to `/dev/adsprpc-smd` + ION directly and boots `/dsp/cdsp/fastrpc_shell_3`;
- `MOCKDSP=1` runs the same code under `qemu-hexagon`.

Its CI benchmark (`benchmark.yml`, job `testqualcommdsp`) runs an int8 MobileNetV2 with `DEV=DSP NOOPT=1`
on a self-hosted **comma4** runner, with a `testsig-*.so` symlinked in. So the signed-PD route through
a test signature is what tinygrad uses on comma hardware.

Results here, with `MOCKDSP=1` and openpilot's `compile_onnx.py`:
- **DM fp16 model**: link error, `undefined symbol: __extendhfsf2` / `__truncsfhf2` (`-nostdlib`
  without compiler-rt).
- **DM fp32 conversion**: compiles into 730 kernels and matches ORT on 2 real frames, max |diff| 1.4e-5.
  But V65 HVX has no float, so this is scalar float code. It gives no speed signal (qemu), and it is not
  the int8 HVX path this study needs.

## 6. Accuracy, round 2: better rounding, mixed precision, calibration (`evaluate.py`, `adaround_conv.py`, `mixed_bits.py`)

Everything is scored against the fp32 model on **two held-out segments (8 and 5)** with openpilot's
parser (`evaluate.py`), free-running with recurrent state as in modeld. Calibration and every ranking
sweep use segments 3 and 12 only. Results: `results/acc_driving.json`, `results/acc_dm.json`.

**The target is fp16's own error vs fp32**, measured first (row "fp16"). Integer configs are compared
to that floor.

Three fixes to round 1's setup, which apply to every row below:
- **Exact input ranges** (`quantize.exact_input_ranges`). The pixel normalization's range is propagated
  from the uint8 camera input ([-2, 2] for driving, [0, 1] for DM) instead of calibrated. Calibration had
  seen no pixel below 22, so darker pixels on other segments were clipped at the first tensor. W16A16
  plan error: 1.4 → 0.57 cm.
- **Recurrent state and heads stay float** (`quantize.backbone_nodes`). "Float after the last Conv" is
  replaced by "everything the Convs don't depend on". The `next_state_*` Concats sit early in node
  order; they turned out not to be quantized in round 1 either, so the numbers didn't move.
- **Quiet eval.** ORT runs at the basic level, as before.

### Driving

| config (heads + state float) | plan lat mean s8 / s5 | lead prob p95 err s8 / s5 | lead agree s8 | projected ms (2 thr @ 1 GHz) |
|---|---:|---:|---:|---:|
| fp16 (floor) | 0.0009 / 0.0004 m | 0.0005 / 5e-6 | 1.000 | GPU |
| W8A8 | 4.31 / 1.05 m | 0.46 / 0.068 | 0.953 | 11.8 |
| W8A16, round-to-nearest | 0.061 / 0.048 m | 0.178 / 5e-4 | 0.940 | 19.7 |
| W8A16, AdaRound | 0.046 / 0.041 m | 0.199 / 0.0012 | 0.938 | 19.7 |
| + `conv2d_2` W16 (top 1) | 0.026 / 0.024 m | 0.049 / 4e-4 | 0.992 | 19.9 |
| + top 4 W16 | 0.017 / 0.022 m | 0.030 / 2e-4 | 0.993 | 20.7 |
| + top 10 W16 | 0.014 / 0.017 m | 0.017 / 3e-4 | 0.998 | 21.3 |
| **W16A16** | **0.0057 / 0.0045 m** | **0.0088 / 7e-5** | 0.997 | 33.4 |
| W16A16, stem stage (12 nodes) float | 0.0038 / 0.0021 m | 0.0043 / 4e-5 | 0.998 | not on the DSP |

**Weight rounding.**
- **onnxsim's own passes don't cover these models.**
  - `apply_adaround` / `apply_gptq` / `apply_tesseraq` target weight-only INT4 MatMul/Gemm, and
    `apply_adaquant` targets QDQ MatMul/Gemm. None handles Conv, and both backbones are all Conv.
  - `cross_layer_equalize` matched **0 pairs**: GELU is not positive-homogeneous, and most convs are
    depthwise.
  - `correct_bias` made things much worse: plan 6 cm → 1.6-1.9 m. It measures each layer's *cumulative*
    output error (upstream error included, not chained) and adds it after *every* layer, so through 60
    convs and 88 Gemms the corrections stack up.
- **`adaround_conv.py`** is a small torch port of AdaRound's objective for Conv, layer-wise, on real
  frames. Mean reconstruction gain is 2.1 dB (driving) and 4.6 dB (DM). Driving plan error drops
  25%, and the lead error is unchanged. On DM it is the difference between W8A16 (worst head 0.017)
  and 0.006.

**Mixed precision.**
- **Weights.** `mixed_bits.py sweep` ranks each conv by the damage of making only it int8 (everything
  else W16), on segment 3. One conv dominates: `conv2d_2`, the stem's 1x1 64→64, which accounts for
  half of all the int8-weight error. Making just it W16 costs 1% more cycles and cuts lead error 4x.
- **Activations.** With every weight W16, a windowed "quantize only this window" sweep on segment 12
  finds the activation error concentrated in the **stem stage**, the first 12 backbone nodes. Keeping
  that stage float still leaves 4-6x fp16's error, spread over the rest of the network.
- **Why A16 falls short of fp16.** fp16's rounding error is relative to each value. A16's is relative to
  the tensor's range. These activations are heavy-tailed (max/std 20-60), so typical values get 20-60x
  less relative precision than the range suggests.

**Other options tried:**
- **Per-channel activation scales** (`--per-channel-acts`, foldable into the neighbouring convs on the
  DSP) are **worse**: plan 1.5-18 cm. Per-channel min/max from 64 frames clips channels that are rarely
  active in calibration.
- **A range margin** (`--range-margin`, headroom against clipping) is also worse: ×1.25 is neutral,
  ×2 doubles the error. So the error is resolution, not clipping.
- **Calibration method** (`results/acc_driving_crossfit.json`): W16A16 cross-fit, calibrating on
  segment 3 or on segment 12, each scored on 8 and 5. **minmax wins in both folds**; percentile and mse
  are 2-6x worse. At 16 bits, clipping costs more than resolution. The fold spread itself is large (s8
  plan 0.66 vs 2.4 cm), so calibration data matters as much as the method.

### Driver monitoring

| config (heads float) | worst head err s8 / s5 | blink agree L/R s8 | projected ms |
|---|---:|---:|---:|
| fp16 (floor) | 2.3e-4 / 1.9e-4 | 1.000 / 1.000 | GPU |
| W8A8 | 0.102 / 0.107 | 0.913 / 0.930 | 6.1 |
| W8A8, AdaRound | 0.121 / 0.117 | 0.857 / 0.873 | 6.1 |
| W8A16 | 0.017 / 0.011 | 0.988 / 0.993 | 10.9 |
| W8A16, AdaRound | 0.0059 / 0.0050 | 0.998 / 0.998 | 10.9 |
| **W16A16** | **5.9e-4 / 4.2e-4** | 1.000 / 1.000 | 19.2 |

The heads now include `sleep_prob`: the round 1 clamp is gone because the heads are float. AdaRound
hurts at A8. Its rounding is optimized against float inputs, so the activation error dominates there.

**Cost model for W16.** `project_latency.py --w16` estimates a W16 pointwise/dense conv at 2x its int8
cycles: two `vrmpy` passes over the weight bytes, with the low byte unsigned (`vrmpy(Vu.ub, Rt.ub)`).
This is an estimate, not simulated. The depthwise kernel already multiplies 16-bit weights, so it costs
no more.

**Where this leaves it.** Beating the GPU's route-log timing needs the cheaper configs (W8 AdaRound +
top-10 W16 at 21 ms, DM W8A16 AdaRound at 11 ms), at 15-40x fp16's own error. Matching fp16's accuracy
isn't possible with 16-bit integer tensors on these nets. The next levers are QAT with the real training
data (comma-side only), or faster kernels, so that W16A16's 33 ms fits (§4, depthwise first).

## 7. Depthwise conv speed (`hvx65/kernels.c`: `dw3`, `dwc3`)

Round 1's `dwc` reached only 4-15 MAC/cycle. That was ~21% of driving's W8A8 cycles for 5% of its
MACs.

**The simulator is load-bound.** A microbenchmark measures hexagon-sim sustaining **one HVX vector load
per ~2 cycles** from L2 (2.0-2.2 cycles/vector for 16 KB-1 MB working sets; stores are the same).
Depthwise is load-heavy: every tap needs a new input vector, and per-channel weights are vectors too.
It was never limited by the multiply instruction. Versions tried, all bit-exact against the scalar
reference (`./run_sim.sh <op> ... check`):

| kernel | idea | result |
|---|---|---|
| `dwc2` | `dwc` + per-layer shift, requant by narrowing `vasr(Vw,Vw):rnd:sat` (≈12 instead of ≈45 ops per 128 outputs), 4 pixels per iteration | 1.2x |
| `dwc3` | taps as `vrmpy`'s 4-way reduction: 4 tap vectors byte+halfword-shuffled (`vshuff`) so a word holds one channel's 4 taps; `vrmpy(Vub, Vb)` is single-resource (2 per packet), no 16-bit widening | 1.3-1.4x (5x5, 7x7) |
| `dwc4` | `dwc3` with each input row shuffled once into a ring buffer, amortized over k output rows | **0.5x**: the extra store + reload costs more than the shuffles it saves |
| **`dw3`** | 3x3 only: all 12 weight vectors in registers for a whole row, 3x3 window sliding along x (3 new loads per output pixel at stride 1) | **1.6-1.7x** |
| **C = 64 packing** (`dw3`, `dwc3`) | channels-last C=64 tensors put 2 pixels per vector instead of padding to 128 lanes; the odd tap is `valign(next, cur, 64)` (`dw3`) or an unaligned load (`dwc3`) | **2.7-3.3x** on the C = 64 layers (driving's largest-spatial stage) |

Not done: **fusing into the neighbouring 1x1.** The two kernels also don't share a layout yet.
- `pw` wants `[P/32][C/4][32 px][4 ch]`, and the depthwise kernels are channels-last.
- A real pipeline needs a transpose between them: about 5 permute ops + a load/store per vector, very
  roughly 5% of total cycles (estimated, not in the projection).
- A fused dw→pw kernel would remove that transpose and the depthwise output's store/reload. It is the
  natural next step, but a larger one than this chunk.

Per-layer cycles (single thread; `sim_cycles_cache.json`). Every depthwise shape in both models is
listed; "new" is `dw3` for 3x3 and `dwc3` (2 pixels per iteration) otherwise:

| C | H x W | k | s | round 1 `dwc` | new | speedup |
|---:|---|---:|---:|---:|---:|---:|
| 64 | 32x64 | 3 | 1 | 276,390 | 83,131 | 3.32x |
| 64 | 64x128 | 3 | 2 | 587,284 | 467,790 | 1.26x |
| 96 | 15x23 | 3 | 1 | 50,146 | 29,460 | 1.70x |
| 128 | 16x32 | 3 | 1 | 70,982 | 42,834 | 1.66x |
| 192 | 15x23 | 3 | 1 | 98,042 | 57,356 | 1.71x |
| 256 | 8x16 | 3 | 1 | 36,840 | 22,384 | 1.65x |
| 512 | 4x8 | 3 | 1 | 19,816 | 12,260 | 1.62x |
| 512 | 8x12 | 3 | 1 | 54,344 | 33,030 | 1.65x |
| 576 | 15x23 | 3 | 2 | 67,369 | 41,958 | 1.61x |
| 64 | 30x45 | 5 | 1 | 351,188 | 128,765 | 2.73x |
| 128 | 8x12 | 5 | 1 | 26,902 | 19,904 | 1.35x |
| 192 | 30x45 | 5 | 2 | 184,240 | 129,802 | 1.42x |
| 384 | 8x12 | 5 | 1 | 75,895 | 54,835 | 1.38x |
| 512 | 8x12 | 5 | 1 | 100,392 | 72,283 | 1.39x |
| 64 | 32x64 | 7 | 1 | 814,158 | 299,435 | 2.72x |
| 64 | 32x64 | 7 | 2 | 205,552 | 151,756 | 1.35x |
| 128 | 16x32 | 7 | 1 | 205,550 | 151,764 | 1.35x |
| 128 | 16x32 | 7 | 2 | 53,312 | 39,868 | 1.34x |
| 256 | 8x16 | 7 | 1 | 104,207 | 77,307 | 1.35x |
| 256 | 8x16 | 7 | 2 | 27,999 | 21,293 | 1.31x |
| 512 | 4x8 | 7 | 1 | 53,585 | 40,121 | 1.34x |

Updated projection (`project_latency.py --dw-kernel best`, the default now; 2 threads @ 1 GHz assumed):

| model / config | dw Mcycles, round 1 → now | total Mcycles | projected ms |
|---|---:|---:|---:|
| driving W8A8 | 4.62 → 2.55 | 23.6 → 21.5 | 11.8 → 10.8 |
| driving W8A16 | 9.24 → 5.10 | 39.5 → 35.3 | 19.7 → 17.7 |
| driving W8 AdaRound + top-10 W16, A16 | 9.24 → 5.10 | 42.5 → 38.4 | 21.3 → 19.2 |
| driving W16A16 | 9.24 → 5.10 | 66.8 → 62.7 | 33.4 → 31.3 |
| DM W8A8 | 1.51 → 0.87 | 12.2 → 11.6 | 6.1 → 5.8 |
| DM W8A16 | 3.02 → 1.75 | 21.7 → 20.5 | 10.9 → 10.2 |
| DM W16A16 | 3.02 → 1.75 | 38.3 → 37.1 | 19.2 → 18.5 |

Depthwise is now 12-14% of driving's cycles, and **pointwise is 64-75%**. `pw` sits at 40-64 MAC/cycle
against a 2-`vrmpy`-per-packet ceiling of 256. So pointwise is where the next speedup is, especially
for W16A16's 4-pass (estimated) pointwise.

## 8. Heads: simulated GEMVs and their weight precision (`project_heads.py`, `head_bits.py`, `hvx65` `gemv`)

Round 1 estimated the driving heads at ~3 ms from weight bytes and an assumed 8 GB/s. They are now
simulated. The weights (23.5 MB for driving, 1.1 MB for DM) are **88 + 53 Gemm/MatMul with constant
weights**: batch-1 GEMVs, plus the attention block's 9-token GEMMs.

**Kernel** (`gemv M K N [a16]`, bit-exact):
- int8 weights prepacked `[N/32][K/4][32][4]`; each 4-byte activation group is splatted, and
  `vrmpy(Vub = splat(x), Vb = W)` produces 32 outputs x 4 k per instruction.
- The next 32-output block is prefetched with `l2fetch` while the current one computes.
- The harness evicts L2 before the call, so every weight comes from DDR, as it would every frame.
- The 9-token GEMMs use all 9 rows per weight vector, and 2 output blocks per splat
  (`gemv9x2_u8`, u8 activations).

**Memory in the sim.**
- Beyond L2 (~1 MB here), an unprefetched HVX load stream sustains 12.9 cycles/vector (10 B/cycle).
- With the prefetch, a K=1024, N=2048 GEMV reaches **19.4 B/cycle** (5.7x the unprefetched 3.4 B/cycle).
- These are hexagon-sim's DDR model numbers, not the 845's LPDDR4X.

| head | activations | weights | Mcycles (1 thread) | ms @ 1 GHz |
|---|---|---|---:|---:|
| driving | u8 | int8 (23.5 MB) | 2.42 | 2.4 (sim) |
| driving | u16 | int8 | 4.36 | 4.4 (sim) |
| driving | u8 / u16 | mixed int8/int16 (29.3 MB, below) | 2.84 / 4.89 | 2.8 / 4.9 (int16 part: 2x est.) |
| driving | u8 / u16 | int16 (46.9 MB) | 4.84 / 8.72 | 4.8 / 8.7 (2x est.) |
| DM | u8 / u16 | int8 | 0.24 / 0.31 | 0.2 / 0.3 (sim) |

- **The 9-token GEMMs dominate** (the attention block's 512→2048→512 MLP and 512→1536 QKV): 1.2 of
  2.4 Mcycles at u8. They are splat-bound, not bandwidth-bound; the batch-1 GEMVs run at 16-19 B/cycle.
- **Not simulated:** the two activation-by-activation attention MatMuls (8 heads x 9x64x9, ~0.1 MMAC),
  and the heads' float epilogues (LayerNorm, Softmax, Sigmoid, GELU, output scaling).

**Weight precision (`head_bits.py`, results in `results/headsweep_driving*.json`).** The heads so far
were float in §6. Fake-quantized per output channel on top of the W16A16 backbone, with the heads'
activations still float:

| heads' weights | plan lat s8 / s5 | lead prob p95 s8 / s5 |
|---|---:|---:|
| float (§6's W16A16) | 0.0057 / 0.0045 m | 0.0088 / 7e-5 |
| int16 | 0.0057 / 0.0045 m | 0.0088 / 7e-5 |
| int8 | 0.042 / 0.036 m | 0.026 / 2e-4 |
| **mixed: int16 for 5 groups (5.8 M params), int8 for the rest (17.7 M)** | **0.0080 / 0.0071 m** | **0.018 / 7e-5** |

- **Int8 head weights alone cost 7x W16A16's whole-backbone plan error.**
- A per-group sweep (only that group int8, rest int16) on segment 12 finds the plan error almost
  entirely in `temporal_hydra.final_layer` (0.25 MB). A lead-probability sweep points at
  `vision_model.policy.hydra`.
- Keeping those two, plus `temporal_hydra.in_layer`, `.resblock` and `summarizer.resblock.block_a`, in
  int16 recovers the plan (0.80 vs 0.57 cm). Lead p95 is halfway (0.018 vs 0.009).
- **Open:** the heads' *activations* on the DSP. There is no float on V65 HVX, and the heads contain
  LayerNorm/Softmax/Sigmoid/GELU. Quantizing the heads' output (`outputs`, one tensor mixing metres
  and logits) was already ruled out in §3. They could run in 16-bit fixed point on the DSP, or on the
  CPU. This was not tested.

**Per-frame DSP time.** Projection: backbone on 2 HVX threads plus heads on 1 thread, at an assumed
1 GHz. The 20 Hz budget for both models together is 50 ms.

| config | driving | DM | total | accuracy vs fp32 (s8) |
|---|---:|---:|---:|---|
| best: W16A16 backbones, mixed heads (u16) | 31.3 + 4.9 = 36.2 ms | 18.5 + 0.3 = 18.8 ms | **55 ms: over budget** | plan 0.8 cm, lead p95 0.018; DM worst head 6e-4 (heads float in eval) |
| cheaper: W8 AdaRound + top-10 W16 A16 backbone, mixed heads; DM W8 AdaRound A16 | 19.2 + 4.9 = 24.1 ms | 10.2 + 0.3 = 10.5 ms | **34.6 ms** | plan 1.4 cm, lead p95 0.021; DM worst head 0.006 |
| GPU today (route logs, older model, fp16) | 29.5 ms | 14.5 ms | | |

Only the cheaper config fits the 20 Hz budget on one cDSP at the assumed clock. That is before the
unmodeled dw↔pw layout conversions (§7, ~5%) and head epilogues. Its error is 15x fp16's on plan and
40x on lead.

## Going on device (plan)

1. **Access.** comma devices run AGNOS (Linux, root).
   - tinygrad already reaches the cDSP there without a Qualcomm SDK: `/dev/adsprpc-smd` +
     `fastrpc_shell_3` + ION.
   - The dynamically loaded skel needs a **test signature** for the device's serial (`testsig-0x<serial>.so`,
     generated with the Hexagon SDK's `elfsigner`/`signer` tooling). Install it under the DSP search path,
     which root allows. This is the same route as tinygrad's comma4 CI.
   - Alternative: link against AGNOS's `libcdsprpc.so` and use `remote_handle64_open` like
     `../android/tinygrad_hexagon_bridge/native_transport`.
2. **Skel.** `hvx65/kernels.c`'s kernels go into one skel with a fused, static-schedule driver (the
   `native_transport` pattern). Weights stay resident; each 20 Hz frame is one FastRPC call on
   ION/dmabuf buffers.
3. **Measure first.** Before optimizing, get the real cDSP clock (DCVS/turbo vote) and the HVX thread
   count, then check the sim's cycles against the device.
4. **Accuracy.** See §6: W16A16 for fp16-like accuracy (still 2.5-12x fp16's error), or the cheaper
   mixed configs at 15-40x.

## Files

| file | purpose |
|---|---|
| `profile_models.py` | op histogram, per-op MACs, I/O |
| `prepare_inputs.py` | route hevc → modeld-exact model inputs (`.npz`) |
| `read_rlog.py` | calibration + on-device model timings from rlogs |
| `run_models.py` | fp16→fp32 conversion, sequential model runs with state feedback, openpilot-parser comparison |
| `quantize.py` | onnxsim `quantize_full_qdq` with real-frame calibration, head-float and policy options |
| `sweep_driving_groups.py`, `policy_driving_mixed.json` | per-window uint8 damage sweep and the resulting mixed policy |
| `hvx65/kernels.c`, `build_sim.sh`, `run_sim.sh` | V65 HVX kernels + hexagon-sim harness (`dw3`/`dwc3`: §7's depthwise kernels; `dwc2`/`dwc4`: the variants that didn't pay off; `warm` runs a kernel once untimed first) |
| `project_latency.py`, `sim_cycles_cache.json`, `results/` | per-layer sim runs → model latency projection |
| `tinygrad_dsp_check.py` | upstream tinygrad `DEV=DSP MOCKDSP=1` vs ORT on real DM frames |
| `evaluate.py` | held-out-segment scoring vs fp32 through openpilot's parser (driving + DM) |
| `adaround_conv.py` | layer-wise AdaRound for Conv int8 per-channel weights (torch) |
| `mixed_bits.py` | per-conv int16 weights on top of `quantize_full_qdq`, per-conv / activation-window sensitivity sweeps, per-channel activation experiment |
| `project_heads.py`, `head_bits.py`, `results/heads_*.json`, `results/headsweep_driving*.json` | §8: simulated head GEMVs, head weight-precision sweeps |
| `results/acc_*.json`, `results/w16_rank_driving.txt`, `results/sweep_w16.json`, `results/actsweep_only.json`, `results/adaround_*.log` | §6 results |

Reproduce (paths are examples):

```bash
export PYTHONPATH=<openpilot checkout>:<this repo>:scripts/openpilot_dsp
python scripts/openpilot_dsp/prepare_inputs.py --road seg8_fcamera.hevc --wide seg8_ecamera.hevc \
  --driver seg8_dcamera.hevc --rpy-calib=-0.00028,0.16415,0.00528 --frames 600 --out inputs_seg8
python scripts/openpilot_dsp/run_models.py to-fp32 driving_supercombo.onnx driving_fp32.onnx
python scripts/openpilot_dsp/quantize.py driving driving_fp32.onnx inputs_seg3.npz,inputs_seg12.npz q.onnx \
  --samples 64 --stride 18 --activation-dtype uint16 --float-after-last-conv
python scripts/openpilot_dsp/run_models.py run driving q.onnx inputs_seg8.npz q.npy
python scripts/openpilot_dsp/run_models.py compare driving driving_supercombo.onnx ref_fp32.npy q.npy
scripts/openpilot_dsp/hvx65/build_sim.sh && scripts/openpilot_dsp/hvx65/run_sim.sh pw 2048 64 192 check
python scripts/openpilot_dsp/project_latency.py driving_fp32.onnx --act uint16 \
  --policy scripts/openpilot_dsp/policy_driving_mixed.json
```
