# AXCL runtime tools

Eight small C programs against the AXCL engine API (`/usr/include/axcl`), for
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
  never write it -- a real, separate latent gap found while building this).
  See the handoff doc's "Trading free memory for throughput" section: the
  baseline mode confirmed this project's established resnet18 numbers;
  `-g` mode is written and correct but has nothing to run yet, since the
  `Gather`-off-a-resident-dataset variant doesn't currently compile on real
  hardware (a genuine Pulsar2 NPU-backend gap, not a bug in this runner).

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
