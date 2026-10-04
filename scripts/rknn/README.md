# Rockchip RKNN integration check

## Luckfox RV1106 target

The connected Luckfox image is a Rockchip RV1106G3 Buildroot system with the
RKNPU driver and `librknnmrt.so` installed. Compile its models on the host with
the RV1106 target:

```python
from rknn.api import RKNN

rknn = RKNN(verbose=False)
rknn.config(target_platform="rv1106")
rknn.load_onnx(model="simplified.onnx")
rknn.build(do_quantization=False)
rknn.export_rknn("model.rknn")
rknn.release()
```

Then benchmark real NPU execution on the board:

```bash
python scripts/rknn/benchmark_luckfox.py model.rknn \\
  --host 192.168.0.238 --input-shape 1,3,224,224
```

`benchmark_luckfox.py` uploads the compiled model and the dependency-free
`luckfox_rknn_runner.py`; the latter calls the board's `librknnmrt.so` through
ctypes and reports min/mean/p50/p95/max latency. SSH authentication is left to
the normal `ssh`/`scp` configuration. The stock Buildroot image documents
`root`/`luckfox` as its default account; use an SSH key or agent for unattended
benchmark runs.

For a repeatable local smoke suite, build two small calibrated INT8 models:

```bash
python scripts/rknn/build_luckfox_models.py \\
  --output-dir /tmp/luckfox-rv1106-models
```

RV1106 rejects floating-point-only RKNN builds, so the builder supplies a
small calibration set and uses `do_quantization=True`. The measured board
results are recorded in [`bench/RESULTS_luckfox_rv1106.md`](../../bench/RESULTS_luckfox_rv1106.md).

To quantize with onnxsim first, use:

```bash
python scripts/rknn/build_luckfox_models.py \
  --output-dir /tmp/luckfox-rv1106-onnxsim \
  --onnxsim-quantize
```

This emits full-graph INT8 QDQ ONNX, leaving `Relu` and
`GlobalAveragePool` outside the QDQ regions. RKNN recognizes it as a
pre-quantized QAT model and the builder uses `do_quantization=False`; both
models compiled and ran on the connected RV1106. RKNN optimization level 3 is
used for this path. Model-specific accuracy and latency checks are still
needed because level 3 may alter numerical behavior.

The builder also runs [`check_rv1106_compat.py`](check_rv1106_compat.py) after
simplification. It reports unsupported or unknown operators before invoking
RKNN, while retaining warnings for partial operators and grouped convolutions
that require a real compiler probe:

```bash
python scripts/rknn/check_rv1106_compat.py model.onnx \
  --output model.rv1106.legalized.onnx
```

The checker is based on Rockchip's published
[`RKNNToolKit2_OP_Support-2.3.2.md`](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/RKNNToolKit2_OP_Support-2.3.2.md)
and the RV1103/RV1106 compiler constraint table. It is a preflight gate, not
a replacement for RKNN's target compiler.

For dense NPU throughput experiments, build a larger Conv workload:

```bash
python scripts/rknn/build_rv1106_peak.py \
  --output-dir /tmp/luckfox-rv1106-peak \
  --channels 64 --layers 8 --size 224
```

The measured stress results are included in the benchmark report. They are
intended to expose sustained throughput and memory/clock ceilings, not to
represent an application model.

Standard ImageNet models can be compiled with:

```bash
python scripts/rknn/build_rv1106_imagenet.py resnet18.onnx \
  --output-dir /tmp/luckfox-rv1106-imagenet
```

The runner supports `--output-format native` for models whose output uses
RV1106's packed native layout. ResNet-18 and MobileNetV2 measurements are
recorded in the benchmark report.

Toolkit2 2.3.2 rejects `quantized_dtype="w4a16"` for the RV1106 target, so
INT4 is not currently an available RV1106 deployment path. The connected
image also does not expose NPU clock controls; see the benchmark report for
the negative checks.

The other onnxsim INT4 route was also probed. `quantize_weight_only_int4`
produces standard ONNX block-wise `DequantizeLinear` + `MatMul` QDQ, which is
valid for ONNX Runtime and other weight-only backends. RKNN-Toolkit2 2.3.2
loads that graph as QAT, but RV1106 rejects the required
`do_quantization=False` build; enabling calibration instead returns to the
ordinary INT8 path. `com.microsoft::MatMulNBits` is likewise not an RV1106
operator. The RV1106 compatibility checker now reports both forms before
conversion.

The host SDK is already the latest public `rknn-toolkit2` 2.3.2. Its public
changelog documents W4A16 for RK3576, not RV1106, so updating the host SDK
does not unlock INT4 on this board. The board's runtime library/driver is a
separate vendor image component; no compatible public RV1106 runtime upgrade
was available to install remotely during this check.

Verifies that `onnxsim`'s output still converts and runs through
[`rknn-toolkit2`](https://pypi.org/project/rknn-toolkit2/), Rockchip's real
ONNX -> RKNN converter for the RK35xx/RV1106 NPU line -- the same toolchain
[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) (already
listed as a downstream onnxsim user in the top-level README) runs onnxsim
ahead of, before feeding its export scripts' ONNX output into `rknn-toolkit2`.

## Real vs. static: where this sits among the sibling checks

Unlike [`scripts/axera`](../axera) (Pulsar2/AXCL has no pip package, so that
check is a static op-support heuristic), `rknn-toolkit2` **is** a real,
pip-installable x86-64 Linux wheel (`pip install rknn-toolkit2`, cp310/cp311/
cp312 on PyPI) -- this harness runs the actual Rockchip converter and its
built-in **PC simulator**, the same way
[`scripts/qualcomm`](../qualcomm) (QNN) and [`scripts/intel`](../intel)
(OpenVINO) run the real backend via a pip EP wheel. The difference from those
two: RKNN is not an ONNX Runtime execution provider at all -- there is no
`RknnExecutionProvider` to register. `rknn-toolkit2` is Rockchip's own
standalone package built around `rknn.api.RKNN`: `load_onnx()` parses the
graph, `build()` compiles it into Rockchip's internal IR, and
`init_runtime(target=None)` runs that IR on the host CPU via Rockchip's PC
simulator -- no RK35xx/RV1106 device attached, no ADB, no NPU-transfer.
Real on-device inference is a separate, later step (`rknn-toolkit-lite2` on
the board itself, or `init_runtime(target="rk3588", ...)` over a connected
device) that this harness does not exercise -- see "Fidelity tiers" below.

## Two real, verified findings

Both reproduced directly against `rknn-toolkit2==2.3.2` (the latest release
on PyPI at the time of writing) on a plain x86-64 Linux host with no RK
device -- no Docker, no hardware:

1. **`onnx.mapping` no longer exists.** `rknn-toolkit2` 2.3.2 declares
   `onnx>=1.16.1` but its C-extension code (`rknn/api/base_utils.py`) still
   does `import onnx.mapping` / reads
   `onnx.mapping.TENSOR_TYPE_TO_NP_TYPE`/`NP_TYPE_TO_TENSOR_TYPE`, a
   submodule the `onnx` pip package no longer ships (verified against onnx
   1.22.0, installed alongside a stock `pip install rknn-toolkit2`). A plain
   `rknn.load_onnx()` call fails with `AttributeError: module 'onnx' has no
   attribute 'mapping'` before ever reaching onnxsim's own code -- not an
   onnxsim bug, but it does mean nothing in this ecosystem can hand
   `rknn-toolkit2` an ONNX model at all on a modern `onnx` install without a
   workaround. `rknn_backend.py`'s `_ensure_onnx_mapping_shim()` patches in a
   tiny replacement (built from the still-current
   `onnx.helper.tensor_dtype_to_np_dtype` table) before `rknn.api` is
   imported, so this harness needs no old, pinned `onnx` version.
2. **`inference()` defaults to NHWC regardless of the ONNX graph's own
   layout.** Every onnx-native CV model onnxsim ever sees is NCHW; feeding a
   plain NCHW ndarray to `rknn.inference()` without `data_format="nchw"`
   raises (`ValueError: The input(ndarray) shape (1, 3, 16, 16) is wrong,
   expect 'nhwc' like (1, 16, 16, 3)!`) rather than silently misinterpreting
   it -- but only because the shapes happened to be distinguishable in this
   probe; a model with equal spatial and channel extents would not raise and
   would just be run transposed. `rknn_backend.run()` always passes
   `data_format="nchw"` explicitly for rank-4 inputs.

## What it checks

Framed the same way as the QNN check -- original vs. simplified through the
**same** RKNN PC-simulator build, so a fixed simulator limitation (an
unsupported op, or the simulator's own float-vs-CPU-reference numeric slack,
see "Fidelity tiers" below) cancels out and only an onnxsim-introduced change
fails the check:

1. `simplify` the model with onnxsim.
2. Convert (`load_onnx` + `build(do_quantization=False)`) and run
   (`init_runtime(target=None)` + `inference()`) the **original** graph.
   If that already fails, `rknn-toolkit2` just doesn't support the graph ->
   reported as `unsupported`, **not** a failure.
3. Convert and run the **simplified** graph the same way.
   If the original converted/ran but the simplified doesn't ->
   `rknn_regression` (a failure): simplification broke RKNN compatibility.
4. Compare the two RKNN outputs. Divergence beyond tolerance ->
   `rknn_regression`: simplification changed the simulator result.
5. Record the ONNX Runtime CPU-reference diff as information only (see
   "Fidelity tiers" -- the PC simulator is not expected to match it tightly).

`do_quantization=False` throughout: this check is about graph-compile and
float-numerics fidelity, not INT8 quantization accuracy, which is a separate,
calibration-dataset-dependent concern already covered elsewhere in this repo
(`onnxsim`'s own calibration/quantization tests).

## Files

| file | purpose |
| --- | --- |
| `rknn_backend.py` | wraps `rknn.api.RKNN`: the `onnx.mapping` compat shim, converts + runs a model through the PC simulator, and the ORT CPU reference. Input synthesis/comparison come from `scripts/common/ep_numerics.py`. Degrades gracefully (`RKNN_AVAILABLE`) when `rknn-toolkit2` is absent. |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the same small, network-free suite of synthetic graphs shared with the Apple/Intel/Qualcomm/Axera harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess (the converter can abort at the C-extension level), printing one `__RESULT__<json>` line. |
| `run_rknn_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Requires an x86-64 Linux host (the `rknn-toolkit2` wheel's supported
platform).

```bash
pip install rknn-toolkit2       # brings its own onnxruntime/numpy/torch
pip install .                   # or install an onnxsim wheel

python scripts/rknn/run_rknn_compat.py --output rknn-compat.csv
# Select the Luckfox/RV1106 compiler target explicitly:
python scripts/rknn/run_rknn_compat.py --target-platform rv1106
```

The in-tree smoke test `tests/test_rknn_compat.py` reuses this harness and is
skipped automatically when `rknn-toolkit2` isn't installed.

## NanoPC-T6 / RK3588 real-NPU target

The PC simulator above validates *convertibility*, not what a real RK3588 NPU
computes. A second, higher-fidelity path runs on a connected NanoPC-T6
(FriendlyElec Ubuntu 24.04.5, kernel 6.1.141, RKNPU driver v0.9.8,
`/usr/lib/librknnrt.so` 2.3.0), again host-compiles / board-runs:

```python
from rknn.api import RKNN

rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3588", mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])
rknn.load_onnx(model="simplified.onnx")
rknn.build(do_quantization=True, dataset="calib.txt")
rknn.export_rknn("model.rknn")
rknn.release()
```

```bash
# Benchmark compiled models on the board.
RKNN_SSH_PASSWORD=... python scripts/rknn/benchmark_nanopc.py model.rknn \
  --warmup 20 --iterations 200

# Build the two synthetic benchmark models for rk3588 first:
python scripts/rknn/build_nanopc_models.py --output-dir /tmp/nanopc-rk3588
```

The uploaded [`nanopc_rknn_runner.py`](nanopc_rknn_runner.py) is
dependency-free apart from numpy and drives the board's `librknnrt.so` through
ctypes, like the Luckfox runner. It handles multiple inputs/outputs and can
write every output buffer back to disk, which is what makes the correctness
check below possible.

**The device node needs no configuration on this board.** Its RKNPU driver
registers as a DRM device (`dmesg`: `[drm] Initialized rknpu 0.9.8 ... on minor
1`), so `/dev/rknpu` does *not* exist -- the nodes are `/dev/dri/card1` and
`/dev/dri/renderD129`, owned by group `render` (which the stock `pi` user is
already in). `librknnrt.so` 2.3.0 discovers this itself, so no path is passed
in. Code hardcoding `/dev/rknpu` will fail here. Note also that host-side
`list_devices()` does not work over SSH: it needs ADB/NTB against a
USB-attached board and raises otherwise.

### Real-NPU correctness check

[`run_rknn3588_compat.py`](run_rknn3588_compat.py) runs the same
original-vs-simplified comparison the simulator harness does, but on the actual
NPU: both graphs are compiled for `rk3588` from an identical calibration set,
uploaded, executed, and their float32 outputs compared.

```bash
RKNN_SSH_PASSWORD=... python scripts/rknn/run_rknn3588_compat.py \
  --output rknn3588-compat.csv
```

Because both builds go through the same compiler, the same NPU and the same
calibration, a fixed backend difference cancels out and only an
onnxsim-introduced change fails. Measured on the connected NanoPC-T6: **all five
suite models agree bit-for-bit** (`max_abs_diff` exactly 0.0), where the PC
simulator can only be compared loosely. The ORT-vs-NPU column in the CSV is
informational INT8 quantization error, not a pass/fail criterion.

**No latency win is claimed.** This harness's `*_latency_*` columns are single
un-replicated samples; two identical runs put `conv_bn_relu` at -43.9% then
+116.7%, because these graphs execute in 0.03-0.09 ms and are dominated by
launch overhead. For a defensible timing use
[`ab_latency_nanopc.py`](ab_latency_nanopc.py), which compiles each variant once
and interleaves A/B/B/A over many rounds.

**Latency here is host-bound, not NPU-bound.** Measured with
[`nanopc_npu_utilization.py`](nanopc_npu_utilization.py): a single inference
stream keeps one core at a ~26% duty cycle and leaves the other two at 0%, i.e.
only ~9% of total NPU capacity. Running 3-4 concurrent streams reaches ~56-60%,
so the hardware is not the constraint -- per-inference driver/IOCTL overhead on
the host is.

**Batching does not help** ([`batch_probe_nanopc.py`](batch_probe_nanopc.py)).
`rknn-toolkit2` 2.3.2's `config()` has no `batch_size` parameter at all (it
existed in the v1 toolkit), so batch must be in the ONNX graph. Batches of 1/2/4/8
all compile and the batch survives into the `.rknn`, but utilisation stays flat
at ~11-12% and per-sample throughput does not move (~240-277 samples/s at every
batch size) while latency scales linearly (3.8 -> 33.4 ms) -- the signature of
RKNN executing batch items sequentially rather than batching them.

**Where the fixed cost is, and what reduces it**
([`overhead_probe_nanopc.py`](overhead_probe_nanopc.py)). Regressing latency
against real NPU work gives `latency = 0.096 ms fixed + 0.051 ms per conv-layer`
-- a ~0.1 ms host-side floor per inference, which is essentially the *entire*
cost of these small graphs. Two measured levers, and only two:

- **int8 `pass_through`** (`attr.pass_through = 1` with the model's native
  dtype, skipping the per-call float<->int8 conversion): **-4.1%**.
- **Run on a big core.** Pinning to CPU 0 costs **+35.8%**; pinning to CPU 6/7
  (the A76s) instead is **~27% faster** than a little A55 core. This is not IRQ
  avoidance -- all three NPU IRQs are on CPU0 -- it is the big/little cluster and
  the `ondemand` governor, since the ~0.1 ms of submission work is
  frequency-sensitive.

`SCHED_FIFO` and realtime scheduling both landed inside the noise. None of this
changes the harness's conclusions: the correctness result (bit-identical
outputs) is unaffected, and the fixed cost still dwarfs what onnxsim's node
reduction removes, which is why this harness claims correctness on real silicon
and not throughput. Details in
[`bench/RESULTS_nanopc_rk3588.md`](../../bench/RESULTS_nanopc_rk3588.md).

The in-tree smoke test is [`tests/test_rknn3588_compat.py`](../../tests/test_rknn3588_compat.py)
(it skips unless `RKNN_3588_HOST` is set), and
`.github/workflows/rknn3588-integration.yml` runs it on a schedule against a
board reached through the repository's Tailscale credentials.

### Profiling is not available on this board

`rknn-toolkit2` has real per-layer profiling (`init_runtime(perf_debug=True)`,
`eval_perf()`, `accuracy_analysis()`, `eval_mem=True`), but none of it is
reachable here, all verified directly:

- `eval_perf()` raises `Not support in simulator environment` -- the PC
  simulator has no NPU timing to report.
- Reaching a real device requires **ADB or NTB**, not SSH. The host toolkit's
  transport is ADB-based (`rknn_platform`: `adb_devices`, `add_adb_forward`,
  `forward localabstract:transfer_proxy`), so `init_runtime(target="rk3588",
  device_id=...)` cannot work over a network shell.
- The board's `librknnrt.so` 2.3.0 does **not implement** the
  `RKNN_QUERY_PERF_DETAIL` / `RKNN_QUERY_PERF_RUN` query codes: both are inside
  the valid command range (`[0, 17]`) yet return `-5` (unsupported) even when
  given correctly-sized structs. There is also no `rknn_set_perf_debug` export
  to switch it on.

`probe_nanopc_perf.py` reproduces this directly against the board. What the
board *does* offer is coarse utilization only, via
`/sys/kernel/debug/rknpu/{load,freq,power,volt}` -- no per-layer breakdown.

Upgrading the runtime does not change this. The board runs `librknnrt.so`
**2.3.0**; the newest runtime upstream publishes (from
`rknn-toolkit2`'s `master` branch, since PyPI and the latest GitHub release are
both still 2.3.2) is **2.3.2**. Side-loaded via
`nanopc_rknn_runner.py --library` -- which leaves the system library untouched --
it reports the same query range and the same `-5` (unsupported) for both perf
codes, with the same 34 exported `rknn_*` symbols. It *does* run the board's
models and produce bit-for-bit identical output, so it is a safe side-load, but
per-layer profiling is absent from both builds. Note also that
`rknn-toolkit-lite2` ships **no** `librknnrt.so` (it dlopens the host's), and
its `rknn_perf` module is memory profiling only -- `collect_memory_detail`, no
layer timing.

The **driver** side has nothing newer either: the board already runs the newest
vendor `rknpu` (v0.9.8). The newer mainline `accel/rocket` (Linux 6.18+) is a
different stack that cannot load `.rknn` models at all, and
[`rkopnu`](https://github.com/marfrit/rkopnu) -- which *does* present the vendor
ioctl ABI so `librknnrt.so` works on mainline -- needs a mainline kernel with
`CONFIG_DRM_ACCEL=y`, which this board's Rockchip BSP 6.1.141 kernel does not
have (no `drivers/accel/` at all, and no kernel build tree). Either way, no
driver change adds per-layer profiling, because that capability is absent from
the runtime itself.

## Fidelity tiers (what this does and doesn't cover)

This check runs `rknn-toolkit2`'s real converter and its **PC simulator**.
That validates *graph convertibility* and gives a *functional* numeric
result with no device. Two things it deliberately does not do:

- **Bit-exact NPU numerics.** Rockchip documents the PC simulator as
  functional, not bit-exact hardware emulation. Verified directly: even with
  `do_quantization=False`, a small Conv+Bias+Relu graph's PC-simulator output
  differs from the ONNX Runtime CPU reference by a small but nonzero amount
  (~4e-3 max-abs on values of order 1-8). That is why this harness compares
  original-vs-simplified through the *same* simulator build rather than
  asserting tight agreement with the CPU reference.
- **Real RK35xx/RV1106 device numerics, and INT8 quantization accuracy.**
  `init_runtime(target=..., device_id=...)` needs a connected board (over
  ADB/NPU-transfer) or `rknn-toolkit-lite2` running directly on one -- neither
  is available in a generic CI job. Quantized (`do_quantization=True`) accuracy
  also needs a representative calibration dataset, which is out of scope for
  this graph-compatibility check.

Real device numerics *are* covered by the separate NanoPC-T6 path above
(`run_rknn3588_compat.py`), which runs both graphs on actual RK3588 silicon
rather than the simulator -- but only against a board this repository can reach,
so it stays out of the default matrix.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models (e.g. the Hugging Face
[`onnxmodelzoo`](https://huggingface.co/onnxmodelzoo) set used by the
large-model regression) can be layered on by passing an on-disk path as
`worker.py`'s second argument; a scheduled job can iterate those the way
`scripts/regression` does.
