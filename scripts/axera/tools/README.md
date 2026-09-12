# AXCL runtime tools

Fourteen small C programs against the AXCL engine API (`/usr/include/axcl`), for
things `axcl_run_model` cannot do.

`axcl_run_model` only ever runs a model's **first** shape group. An
`llm_build` layer file has two -- group 0 is decode, group 1 is prefill -- so
prefill cannot be timed with the shipped CLI. These talk to
`axclrtEngineExecute()` directly, which takes the group index.

- `probe_model_io.c` -- print a compiled model's shape groups, and each
  group's input and output names and sizes. This is what shows an
  `llm_build` layer to have two groups rather than the zero the CLI reports.
- `bench_shape_group.c` -- allocate device buffers for one group, execute it
  `repeat` times and report min/avg latency.
- `resident_runner.c` -- run a compiled `onnxsim.qat_graph.StepGraph`-shaped
  training step (see `../build_resident_train_step.py`) in a loop with its
  trainable-weight state kept **device-resident** between `Execute()` calls
  (copied output-buffer -> input-buffer device-to-device, never through the
  host), and only the batch crossing the host boundary each step. `-n` runs
  the same loop but round-trips state through a host buffer instead, for a
  direct before/after comparison against the same compiled model -- see
  `docs/axera-on-device-training-handoff.md`'s "Weights resident with
  in-graph updates" section for the numbers this produced on a real AX650N.
  `-v` runs with `AXCL_VNPU_ENABLE` instead of `AXCL_VNPU_DISABLE` --
  confirmed non-corrupting, and the way to get real concurrent throughput:
  run several copies of this binary at once (each against its own model-file
  copy) and the NPU schedules them concurrently instead of serializing. See
  the handoff doc's "Execution overlap" section for the scaling numbers and
  for why `axclrtEngineExecuteAsync` -- AXCL's other overlap primitive --
  is not an option (`AXCL_ERR_UNSUPPORT` on this device/SDK build). Also
  reports device memory: `axclrtEngineGetUsageFromModelId()` is queried once
  after load and printed both as a stderr diagnostic and as the `cmm=...MiB`
  field on the final summary line -- confirmed real and working (unlike
  `axclrtEngineExecuteAsync`), see the handoff doc's "Device memory" section
  for what it reports versus `axcl-smi`'s own numbers, which don't match and
  aren't supposed to (one is a planned budget, the other a live snapshot).
- `whisper_resident_runner.c` -- `resident_runner.c` with the I/O layout
  changed for a Whisper `last_half` training step (14 trainable-weight state
  tensors instead of resnet18's 4, same positional convention: inputs =
  `[input.1, y, 14 state tensors, lr]`, outputs = `[14 updated state
  tensors, loss]`) -- see `../build_whisper_train_step.py` and the handoff
  doc's "`last_half` actually trains" section for what a real 30-step run on
  this found (a real update at step 0, then the gradient rounds to zero from
  step 1 on -- confirmed via a direct pre/post read of the raw state buffer,
  not the runner's own reporting path).
- `mp_calib_swap_runner.c` -- runner for the multi-phase calibration-swap
  demonstration (`docs/axera-on-device-training-handoff.md`'s "Multi-phase
  calibration swap" section) against the small Conv+Gemm training step
  `build_multiphase_calib_swap_probe.py` builds -- fixed I/O order (`x y cw
  gw lr`), CLI `lr` and a `y` host file so the exact "does the update
  survive" experiment can be run without rebuilding.
- `mp_calib_swap_auto_runner.c` -- automatic-trigger variant of the above:
  after every step, checks whether the update carries no signal (either
  unchanged from its input, or crushed to hard zero -- the two distinct
  death signatures this project's history has found) and, on death, dumps
  the last-known-good state to host files and exits with a distinct code
  instead of running a fixed step count for a human to inspect afterward.
  An `inject_step` argument can swap in a different-scale state mid-run (to
  exercise the detector without needing a real multi-thousand-step
  convergence run first) -- see the handoff doc's "Recalibrating Whisper for
  its real gradient scale" section for what a real run of this found:
  the detection-and-handoff mechanism itself works correctly, though a
  clean survives-then-recovers demonstration still needs matched
  calibration/seed values end to end.
- `whisper_state_probe.c` -- like `whisper_resident_runner.c`, but prints
  one state tensor's raw device-buffer contents directly before/after each
  step instead of trusting the loss, which this project's history has
  repeatedly found can look constant while hiding either a real update or a
  dead one underneath (see the handoff doc's Whisper sections). Reads
  `x`/`y` from fixed host files (`/root/whisper_x.bin`/`whisper_y.bin`) so
  the same batch can be reused across differently-calibrated compiles of
  the same graph shape.
- `gather_runner.c` -- resident runner for the two I/O layouts
  `build_resident_train_step.py`'s `add_resident_dataset()` work produces:
  the plain baseline (`x y state[4] lr grad_seed`, `-g` omitted) and the
  resident-dataset variant (`batch_index state[4] lr grad_seed`, `-g`) --
  a single binary switches layout via the flag rather than needing a
  separate copy per shape the way the Whisper runner did. Explicitly feeds
  `grad_seed`, which neither `resident_runner.c` nor
  `whisper_resident_runner.c` actually does (both allocate its buffer but
  never write it -- a real, separate latent gap found while building this,
  still outstanding in those two). See the handoff doc's "Trading free
  memory for throughput" section: the baseline mode confirmed this
  project's established resnet18 numbers; `-g` mode is confirmed **working
  on real hardware** for the pre-flattened-dataset workaround
  (`../build_resident_dataset_gather_probe.py`), up to a real, measured
  190-row OCM-capacity ceiling for that scope -- writes `int32_t` indices
  (`-rN` sets the row count they're drawn mod), not the `int64_t` an
  earlier version of this file wrote: Pulsar2 silently downcasts the ONNX
  graph's declared `int64` `batch_index` input to `int32` on-device, so the
  old code only half-initialized that buffer and reliably faulted
  `axclrtEngineExecute` with `0x8030070c`.
- `w2v2fe_runner.c` -- resident runner for `../build_w2v2_feature_extractor_step.py`'s
  training step (one trainable state tensor, its own I/O layout, real
  `probe_io`-confirmed). Compiles under standard INT8; `highest_mix_precision`
  fails a third distinct way here (`AxErf`'s `lut_float` path, not
  Whisper's `LayerNorm` tiling limit or resnet18's `AvgPool` scheduler
  crash) -- see the audio-speech coverage doc's real-hardware follow-up
  section. Prints `w[0]` alongside loss every step, the same
  read-the-raw-state-buffer diagnostic this project has needed twice
  before to catch a gradient dying silently behind a healthy-looking loss
  output.
- `w2v2fe_runner_realdata.c` -- `w2v2fe_runner.c` with `x`/`y` read from
  `<model>.x0`/`<model>.y0` host files instead of a fixed `memset` pattern,
  used to confirm real (not degenerate) inputs actually train once
  calibration is fixed -- see the audio-speech coverage doc's "Fixed: real
  calibration data" section for the real multi-step result this produced
  and the two calibration bugs (input-scale mismatch, a degenerate constant
  `lr` calibration range) it found along the way. `argv[4]` overrides `lr`
  at runtime for sweeping it against the compiled model's own calibrated
  range. Also gained a `-v` flag (`AXCL_VNPU_ENABLE`, same lever/methodology
  as `resident_runner.c`'s own -- PRs #1345/#1346) and real device-memory
  reporting (`axclrtEngineGetUsageFromModelId`, PR #1347) -- confirmed
  non-corrupting and 3.16x/3.70x aggregate throughput at N=4/8 concurrent
  contexts on the batch=1 model, the audio-speech coverage doc's "Batching
  and vNPU concurrency" section has the full numbers. The loop itself is
  batch-size-agnostic (buffer sizes come from the compiled model's own
  IOInfo), but batch>1 builds currently train incorrectly on real hardware
- `w2v2fe_runner_lrdrop.c` -- `w2v2fe_runner_realdata.c` variant built to
  settle whether a long-run training plateau (PR #1376, 2,000 steps) was
  ordinary SGD convergence at an oversized `lr` or a genuinely stalled
  gradient. Keeps weight state device-resident continuously across a
  *single* run while switching `lr` via a host->device scalar write at a
  given step (`argv[3]`, between `argv[4]`'s and `argv[5]`'s two `lr`
  values) -- no restart, no state round-trip, so the before/after comparison
  is on the exact same resident state rather than two separate runs.
  Usage: `w2v2fe_runner_lrdrop model.axmodel steps switch_step lr1 lr2
  [warmup]`. Found a third distinct gradient-ceiling pattern this way --
  see the audio-speech coverage doc's "The 2,000-step plateau (PR #1376) is
  a resolution ceiling, not convergence" section for the real result.
  (a calibration-range regression, not a runner bug) -- see that same
  section before trusting a batch>1 run's loss/weight output.
- `w2v2fe_runner_capture.c` -- `w2v2fe_runner_realdata.c` variant built to
  supply the one thing no earlier wav2vec2 run persisted: the full
  trainable-weight tensor (not just its `w[0]` scalar readback) at a window
  of late-training steps, plus the final state -- the real trajectory data
  `../build_w2v2fe_mp_swap_phase2.py` needs to recalibrate against, since a
  multi-phase calibration swap (PRs #1355/#1356's technique) needs real
  late-stage values on disk, and nothing before this wrote any.
  Usage: `w2v2fe_runner_capture model.axmodel steps warmup lr capture_start
  capture_stride capture_count out_dir` -- writes `out_dir/w_capture_<i>.bin`
  (raw float32, full weight tensor) for `capture_count` steps starting at
  `capture_start` every `capture_stride` steps, `out_dir/loss_capture.txt`,
  and `out_dir/final.state0` (the run's last weight state, in
  `resident_runner.c`'s own `.state0` convention -- feed it straight to the
  next phase's compile as its seed; also reused as-is to capture phase 2's
  own trajectory for a phase-3 build). See the audio-speech coverage doc's
  "Multi-phase calibration swap breaks the plateau, then hits a new one" and
  "Phase 3" sections for the real result this produced: a genuine, confirmed
  loss decrease past PR #1376's plateau on the first swap, but a
  diminishing-returns, not-repeatable-indefinitely technique -- the second
  swap (phase 2 -> 3) bought no further gain, since phase 2's own real
  trajectory had already narrowed to a single quantization step with
  nothing left for another recalibration to exploit.
- `resnet50_realdata_runner.c` -- `resident_runner.c` (N_STATE=4, same I/O
  layout: inputs `x y layer4.2.conv1.weight layer4.2.conv2.weight
  layer4.2.conv3.weight fc.weight lr[grad_seed]`, outputs the four updated
  states then `loss`) with `x`/`y` read from `<model>.x0`/`<model>.y0` host
  files instead of `resident_runner.c`'s fixed `memset` pattern, and loss
  printed every step rather than just the first 5 -- used to close the
  "resnet50 never had batch>1 tested" gap
  (`../build_resnet50_batch_step.py`, `docs/axera-on-device-training-
  handoff.md`'s "resnet50 batch scaling" section) with an unambiguous,
  monotonically-checkable real loss curve at each batch size, the same
  reason `w2v2fe_runner_realdata.c` exists for its own model. Confirmed
  real, non-degenerate training at batch 1/4/8 this way -- also settles
  that section's own earlier "reported loss read exactly 0" caveat (a
  `memset`-near-zero test input rounding to 0 under real calibration, not a
  bug, the same conclusion resnet18's own memset runs already supported).
- `w2v2_encoder_attn_runner.c` -- `w2v2fe_runner_realdata.c` with only the
  header comment and I/O names changed for
  `../build_w2v2_encoder_attn_step.py`'s own model: the first wav2vec2
  build whose trainable tail spans attention output, exercising
  `onnxsim.graph_grad._grad_where`/`_grad_is_nan` on real hardware (see the
  audio-speech coverage doc's own real-hardware follow-up section). Same
  I/O shape as `w2v2fe_runner_realdata.c` (one trainable state tensor), so
  no new runner logic. `argv[4]` (`lr`) matters more here than it did
  there: this model's real gradient is ~100-1000x smaller (two real
  encoder layers deep), so it needs a correspondingly larger calibrated
  `lr` to clear its own INT8 quantization step -- `lr=1` freezes the
  weight after step 0, `lr=2000` (this model's own calibrated real
  trajectory) moves it consistently every step.

Build and run them where the card is visible (inside the VM, if the device is
passed through -- see `../vm/README.md`):

```sh
gcc -O2 -I/usr/include/axcl -o bench_shape_group bench_shape_group.c \
    -L/usr/lib/axcl -laxcl_rt -Wl,-rpath,/usr/lib/axcl
./probe_model_io  layer.axmodel
./bench_shape_group layer.axmodel 1 15      # group 1 = prefill

gcc -O2 -std=c11 -I/usr/include/axcl -o resident_runner resident_runner.c \
    -L/usr/lib/axcl -laxcl_rt -Wl,-rpath,/usr/lib/axcl
./resident_runner train_step.axmodel 30       # resident (device-to-device)
./resident_runner train_step.axmodel 30 -n    # non-resident (host round trip)
./resident_runner train_step.axmodel 30 -v    # AXCL_VNPU_ENABLE (run N copies concurrently for real throughput)
```

Group 0's latency from `bench_shape_group` matches `axcl_run_model`'s to
within a few percent (9.117 ms against 9.14 ms on the model measured in the
README), which is the check that the buffers and the timing loop are right.
