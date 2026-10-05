# SG2002 TPU-MLIR profiling

`profile_model.py` profiles an already compiled CVI model on an SG2002. It uses the board's
`model_runner` for repeated device timings and the CVI Runtime PMU counters. Pass
`--driver-usage` to additionally sample the Linux driver's one-second
`/proc/tpu/usage_profiling` counter during a sustained run.

The host needs Python and Paramiko. The board needs its matching `model_runner`, runtime
libraries, and a `.cvimodel` generated for `cv181x`. The input NPZ must contain the tensors and
types expected by that model. Use an SSH key or provide the board password through the named
environment variable.

```sh
export TPU_MLIR_SSH_PASSWORD="$BOARD_PASSWORD"
python bench/sg2002/profile_model.py \
  resnet50.cvimodel resnet50-input.npz \
  --host kvm-9c35.local \
  --count 100 --repeat 3 --batch 1 \
  --driver-usage \
  --json-out resnet50-profile.json
```

Normal timing runs use `model_runner --enable-timer`. PMU runs set `TPU_ENABLE_PMU=1`, collect
per-inference TIU, TDMA, and inference measurements, and discard the first sample of each
process as warm-up. `--driver-usage` temporarily enables driver profiling, samples while the
model runs, and disables it afterward. The script refuses to change the counter if profiling is
already active. A counter sample outside 0–100% (which can occur during startup) is retained in
the report and excluded from the median.

The model and NPZ are uploaded to a unique directory under `--remote-dir` (default
`/tmp/onnxsim-tpu-profile`) and removed after measurement. Compilation is not included in timing.

`onnxsim.rpc` can compile an ONNX model with TPU-MLIR, execute it on the SG2002, and return the
same runtime PMU counters; see [the RPC guide](../../docs/rpc.md#tpu-mlir-compilation-and-sg2002-execution).
The CLI here is useful when a CVI model has already been compiled or when driver usage sampling
is needed.
