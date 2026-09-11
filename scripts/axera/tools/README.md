# AXCL runtime tools

Three small C programs against the AXCL engine API (`/usr/include/axcl`), for
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
  is not an option (`AXCL_ERR_UNSUPPORT` on this device/SDK build).

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
