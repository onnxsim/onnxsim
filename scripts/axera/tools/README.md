# AXCL runtime tools

Two small C programs against the AXCL engine API (`/usr/include/axcl`), for
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

Build and run them where the card is visible (inside the VM, if the device is
passed through -- see `../vm/README.md`):

```sh
gcc -O2 -I/usr/include/axcl -o bench_shape_group bench_shape_group.c \
    -L/usr/lib/axcl -laxcl_rt -Wl,-rpath,/usr/lib/axcl
./probe_model_io  layer.axmodel
./bench_shape_group layer.axmodel 1 15      # group 1 = prefill
```

Group 0's latency from `bench_shape_group` matches `axcl_run_model`'s to
within a few percent (9.117 ms against 9.14 ms on the model measured in the
README), which is the check that the buffers and the timing loop are right.
