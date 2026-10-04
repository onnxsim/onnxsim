# NanoPC-T6 / RK3588 NPU benchmark

Date: 2026-10-04

Board: FriendlyElec **NanoPC-T6**, Rockchip **RK3588** (8x Cortex-A55), FriendlyElec
Ubuntu 24.04.5 LTS, kernel 6.1.141, RKNPU driver v0.9.8, reached over SSH
(pi@nanopc-t6.tailf0b7b1.ts.net). The board ships `/usr/lib/librknnrt.so`
**2.3.0** (`c949ad889d@2024-11-07T11:35:33`) but no RKNN-Lite Python package, so
inference is driven through ctypes by `scripts/rknn/nanopc_rknn_runner.py`.

Models were generated as static NCHW ONNX graphs, simplified with onnxsim,
compiled with RKNN-Toolkit2 2.3.2 for `target_platform="rk3588"`, and calibrated
to INT8. Each timing is 10-20 warmups followed by 20-200 timed `rknn_run`
calls, measured on-device; SSH and model upload are excluded.

## How much of the NPU are we actually using? (measured)

Short answer for the harness's own workload: **~9% of total NPU capacity** (26%
of a single core, cores 1-2 entirely idle). But that is *not* a hardware limit
-- it is a single-stream host round-trip limit, and it is the same host-bound
ceiling that made the original-vs-simplified latency comparison come out as
noise.

Measured with `scripts/rknn/nanopc_npu_utilization.py`, which drives the model in
a tight loop and samples `/sys/kernel/debug/rknpu/load` (per-core duty-cycle
counter, confirmed to read a clean `0%` on all three cores when idle) across the
whole window rather than taking a single sample. 28 samples over 25 s on
`conv_bn_relu_224`:

| core | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| core0 | 26.1% | 26.0% | 25% | 27% |
| core1 | 0% | 0% | 0% | 0% |
| core2 | 0% | 0% | 0% | 0% |

**= 26.1% of the busiest core, 8.7% of the 3-core NPU.** Very stable
(min 25%, max 27% over 25 s), so this is a real steady-state duty cycle, not a
sampling artifact.

### The limit is host round-trips, not the NPU

Running **N concurrent inference streams** scales utilisation almost linearly
until the NPU saturates -- the runtime *does* spread work across all three cores
on its own, once there is enough of it to spread:

| concurrent streams | core0 | core1 | core2 | total NPU capacity used |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 39.1% | 13.2% | 0% | **17.4%** |
| 2 | 55.5% | 33.8% | 13.0% | **34.1%** |
| 3 | 78.8% | 48.0% | 40.9% | **55.9%** |
| 4 | 77.8% | 60.9% | 42.1% | **60.3%** |

So the hardware is not the constraint -- three concurrent streams already reach
~56% of total capacity, and the plateau at 3-4 streams is the NPU filling up
rather than the driver failing to scale. A single RKNN model is **not** split
across cores (the runtime schedules whole inferences), which is exactly why one
stream leaves two cores idle: there simply isn't enough concurrent work.

The practical implication for the timing work above: **latency here is bounded
by per-inference CPU-side overhead, not NPU throughput.** Each inference is
~2.9 ms wall-clock while the NPU is busy only ~0.75 ms of that (26% duty cycle);
the rest is driver submission, IOCTL and memory-map overhead on the host side.
That is why onnxsim's node-count reduction cannot show up in wall-clock latency
on this setup -- the NPU is not the thing being measured. It also means the
harness is a *correctness* check on real hardware, which is exactly the claim it
makes, and not a throughput benchmark.

If a throughput number is what is wanted, the right shape is N concurrent
clients against one model (or a real application model large enough to occupy a
core on its own), not a single serialized inference loop.

### Would a larger batch help? No (measured)

Batch size is the obvious lever for a 26%-of-one-core duty cycle, so it was
tested rather than assumed. `scripts/rknn/batch_probe_nanopc.py` builds the same
Conv-BN-Relu block with a real batch dimension N, compiles each for `rk3588`, and
measures both NPU utilisation and end-to-end throughput.

**First, a hard limit on the obvious approach:** `rknn-toolkit2` 2.3.2's
`config()` has **no `batch_size` parameter** (verified by signature -- it
existed in `rknn-toolkit` v1). Batch is a property of the ONNX graph, not a
converter knob. Building the batch into the graph does work: all of N = 1, 2, 4,
8 compile cleanly, and the batch survives into the `.rknn` (input
`[8,224,224,3]`, output `[8,32]`).

But it buys nothing:

| batch | RKNN size | cores (mean %) | NPU capacity used | ms / call | calls/s | **samples/s** |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 76 KiB | 34.7 / 0 / 0 | 11.6% | 3.80 | 263.2 | **263.2** |
| 2 | 120 KiB | 34.9 / 0 / 0 | 11.6% | 7.45 | 134.2 | **268.5** |
| 4 | 203 KiB | 35.1 / 0 / 0 | 11.7% | 14.46 | 69.2 | **276.6** |
| 8 | 370 KiB | 29.9 / 0 / 0 | 10.0% | 33.43 | 29.9 | **239.3** |

**Utilisation is flat (~11-12% of capacity) and per-sample throughput does not
improve** (~240-277 samples/s at every batch size), while latency scales almost
exactly linearly with N (3.8 -> 33.4 ms). That is the signature of RKNN **not
batching**: the NPU processes the N batch items sequentially inside one call, so
amortizing the submission buys nothing and the artifact just grows with N. The
slight drop at N=8 is where the larger working set starts costing more than the
saved submissions.

So the conclusion from the previous section stands and gets sharper: the
bottleneck is **per-inference host-side overhead**, and neither batching (which
only reduces submissions if the device honours it) nor more concurrency is
something this model can exploit on its own. The way to actually fill this NPU is
N independent clients or threads, as the concurrency table above shows -- not a
batched graph.

### Where the per-call host overhead actually goes (measured)

`scripts/rknn/overhead_probe_nanopc.py` regresses per-call latency against real
NPU work (a ladder of Conv+Relu stacks at identical op mix, growing depth), then
tests concrete candidate reductions. A linear fit over the ladder gives:

```
latency = 0.096 ms (fixed) + 0.051 ms per conv-layer
```

So a **fixed ~0.1 ms per inference** is host-side, independent of model work.
Against a larger model that fixed cost is ~21% of total; for the small
benchmark graphs that earlier came in at 0.03-0.09 ms *total*, the fixed cost
is essentially all of it. That is the quantitative version of "these graphs are
launch-bound."

What reduces it, measured (4 interleaved trials, median of `min`, 300 iters):

| change | latency | vs baseline |
| --- | ---: | ---: |
| baseline | 0.469 ms | — |
| **int8 `pass_through`** (skip per-call quantize/dequantize) | 0.450 ms | **-4.1%** |
| pin to **CPU 0** (a little A55) | 0.637 ms | **+35.8%** |

**The one real but modest win is `pass_through`.** Setting
`attr.pass_through = 1` with the model's native int8 dtype lets the runtime skip
the float<->int8 conversion it otherwise does on every `rknn_run`. It is a
genuine saving, but only ~4% -- and an initial single trial suggested -17%,
which did not survive repetition. Treat -4% as the number.

**CPU pinning backfires, and CPU choice matters more than any of this.** Pinning
to a *specific* CPU is what hurt: it removes the scheduler's ability to move the
thread, and CPU 0 is a little core. Choosing a *big* core instead is a real
~27% win. Three alternating rounds of 400 iterations:

| pinned to | latency (min of run) |
| --- | ---: |
| CPU 0 (little) | 0.609 / 0.634 / 0.680 ms |
| CPU 6 (big) | 0.480 / 0.481 / 0.483 ms |
| CPU 7 (big) | 0.482 / 0.493 / 0.466 ms |

Consistently ~27% faster on the big cores. Note this is *not* IRQ-affinity
avoidance: all three NPU IRQs (37-39, `fdab9000.iommu, fdab0000.npu`) are
serviced on **CPU0** per `/proc/interrupts`, so avoiding CPU0 does not dodge
them -- it is purely the RK3588's big/little A55 cluster, where the little cores
run at a lower frequency (`ondemand` governor, confirmed) and the ~0.1 ms of
host-side submission work is sensitive to that. RK3588 has 4x A76 at the top
and 4x A55 at the bottom; CPUs 4-7 are the big ones.

**What this does *not* help, tested:** `SCHED_FIFO` (+8.6% on the earlier single
trial) and combining pinning with realtime scheduling (-5.6%) both landed
inside the noise, and once the thread is on a big core there is little left for
realtime priority to buy.

Practical takeaway for the ~0.1 ms fixed cost: the two levers that actually
move it are **`pass_through` int8 IO (~4%)** and **running on a big core (~27%
for the pinning decision itself)**. Everything else -- realtime scheduling,
NPU governor changes, batching (previous section) -- is noise or worse. None of
these change the harness's conclusions: the correctness result (bit-identical
outputs) is unaffected, and the fixed cost is still large enough relative to
these small graphs that onnxsim's node reduction cannot surface in wall-clock
latency.

## Latency (absolute, not a simplification win)

| model | ONNX nodes | simplified nodes | RKNN size | input | mean | p50 | p95 | max |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| conv_bn_relu_224 | 5 | 4 | 67 KiB | 224x224x3 | 2.888 ms | 2.805 ms | 3.068 ms | 3.946 ms |
| depthwise_pointwise_112 | 5 | 5 | 58 KiB | 112x112x3 | 0.701 ms | 0.705 ms | 0.858 ms | 0.969 ms |

(200 timed iterations, zero-filled inputs -- a pure latency measurement. These
are the *simplified* models; see "no measurable speedup" below for why no
original-vs-simplified latency claim is made from this harness.)

The NPU really is doing the work: during a sustained run
`/sys/kernel/debug/rknpu/load` reported `Core0: 27%`, and
`rknn_get_device_properties` independently identified the device as
`RKNN_SOC_RK3588` with `cores: 3`. These are not CPU-fallback numbers.

## No measurable speedup from simplification (measured)

**This board shows no reproducible latency win from onnxsim, and it would be
wrong to claim one.** Two identical back-to-back runs of
`run_rknn3588_compat.py` produced contradictory deltas:

| model | run 1 delta | run 2 delta |
| --- | ---: | ---: |
| conv_bn_relu | -43.9% | +116.7% |
| redundant_transpose | -39.5% | +6.8% |
| sigmoid_mul_swish | +5.3% | -10.7% |

The original-vs-simplified columns in that harness's CSV are **single,
un-replicated samples and must not be read as a speedup** -- three of five
change sign between runs.

Interleaved A/B sampling (`ab_latency_nanopc.py`: compile once per variant,
then alternate A/B/B/A ordering over 20 rounds so drift and background load hit
both equally, reporting min and a paired per-round difference) also found no
effect. **Every model came back `no-change`** -- the paired differences straddle
zero, and the interquartile range of those differences (0.027-0.165 ms) is as
large as or larger than the entire measured latency of most of these graphs
(~0.029-0.059 ms):

| model | nodes | orig min | simp min | delta | paired IQR | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| conv_bn_relu | 5->4 | 0.0376 ms | 0.0385 ms | +2.4% | 0.039 ms | no-change |
| foldable_shape_reshape | 6->3 | 0.0321 ms | 0.0315 ms | -1.9% | 0.028 ms | no-change |
| matmul_bias_tanh | 4->2 | 0.0589 ms | 0.0516 ms | -12.4% | 0.165 ms | no-change |
| redundant_transpose | 5->2 | 0.0300 ms | 0.0289 ms | -3.7% | 0.027 ms | no-change |
| sigmoid_mul_swish | 3->3 | 0.0359 ms | 0.0367 ms | +2.2% | 0.045 ms | no-change |

The nominal deltas (-12.4% to +2.4%) are *inside the noise*: for every model the
spread of round-to-round differences exceeds the difference being claimed, and
an independent 6-round run put `sigmoid_mul_swish` at +138.7% on one pass and
+2.2% on another for the same two artifacts.

### Why: the synthetic suite is launch-bound, and RKNN re-fuses anyway

Two independent reasons, both verified:

1. **These graphs are launch-overhead bound, so node count is irrelevant.** At
   0.03-0.06 ms the whole inference is dominated by fixed `rknn_run` dispatch
   cost. Removing one BN/Reshape node out of five changes nothing measurable.
   The heavier models above (`conv_bn_relu_224` at 2.888 ms) are the ones that
   could in principle show a difference, but they were not compiled from an
   un-simplified counterpart in this measurement.
2. **RKNN's own optimizer already does the fusion onnxsim does.** Several
   variants compile to **byte-identical-size** artifacts after RKNN's
   `OpFusing` pass (`foldable_shape_reshape`, `redundant_transpose`,
   `sigmoid_mul_swish`: same size to the byte), so onnxsim removed work that
   the NPU compiler would have removed anyway. Node counts before the RKNN
   backend are simply not the quantity that determines NPU latency.

Artifact sizes barely move: 150.4 KiB total original vs 145.6 KiB total
simplified, the whole delta coming from one model (`matmul_bias_tanh`,
38.2 -> 33.2 KiB, where onnxsim's `MatMul`+`Add` fold does survive into a
smaller binary).

To actually demonstrate a simplification win on this board you would need a
model whose simplified form changes RKNN's *scheduling* -- e.g. one where folding
removes a layout change, lets a large op become a single NPU kernel, or unlocks
a fusion RKNN's optimizer level does not reach. Small synthetic CNNs cannot.

## Correctness: original vs onnxsim-simplified, both on the real NPU

`scripts/rknn/run_rknn3588_compat.py` compiles both graphs for `rk3588` from an
identical calibration set, runs both on the NPU, and compares.

| model | nodes | simplified | diff(orig, simp) on NPU | diff(ORT CPU, NPU) |
| --- | ---: | ---: | ---: | ---: |
| conv_bn_relu | 5 | 4 | **0.0** | 0.0085 |
| foldable_shape_reshape | 6 | 3 | **0.0** | 8.3485 |
| matmul_bias_tanh | 4 | 2 | **0.0** | 0.0466 |
| redundant_transpose | 5 | 2 | **0.0** | 1.7381 |
| sigmoid_mul_swish | 3 | 3 | **0.0** | 5.5160 |

**Every model agrees exactly** (`max_abs_diff` bit-for-bit 0.0) between the
original and the simplified graph on real RK3588 silicon. That is a stronger
result than the PC simulator can give, and it is what
`scripts/rknn/run_rknn_compat.py` could not check: simplification did not
change what the NPU computes, at any of these topologies.

The `diff(ORT CPU, NPU)` column is **informational, not a failure** -- it is
ordinary INT8 quantization error against a float32 reference. It scales with
output magnitude, e.g. `foldable_shape_reshape`'s output peaks at 8.349 in ORT
and 7.956 on the NPU (~4.8% relative, one INT8 step at that activation scale),
while `conv_bn_relu`'s output peaks at 2.712 with an absolute error of 0.0085.
A per-tensor INT8 activation range covering 0..8.3 simply has ~0.03 resolution.

## Profiling: what RKNN offers, and what this board actually provides

RKNN has detailed profiling, but **almost none of it is reachable here.** All
of the following was verified directly.

### The host toolkit (`rknn-toolkit2` 2.3.2) exposes

| API | what it does | reachable on this board? |
| --- | --- | --- |
| `init_runtime(..., perf_debug=True)` | turns on per-layer profiling | flag is accepted, but see below |
| `eval_perf(is_print=True, fix_freq=True)` | prints an NPU/CPU time split, writes `eval_perf.csv` | **no** |
| `accuracy_analysis(...)` | per-layer numerical deviation vs a reference | **no** |
| `eval_mem=True` | memory footprint breakdown | **no** |
| internal `get_run_perf_detail` / `_get_run_perf_on_hardware` | raw `rknn_query` perf codes | **no** |

Two hard blocks, both reproduced:

- **`eval_perf` is simulator-only-hostile**: `init_runtime(target=None,
  perf_debug=True)` returns 0 and is accepted, but `eval_perf()` then raises
  `ValueError: Not support in simulator environment, please set target in
  init_runtime!`. Rockchip's own strings confirm the message
  *"You can set perf_debug to True in init_runtime to get more detail
  performance information!"*.
- **Reaching a real device needs ADB or NTB, not SSH.** The host toolkit's
  device transport is ADB/NTB-based (`rknn_platform`'s strings:
  `adb_devices`, `add_adb_connect`, `add_adb_forward`,
  `forward localabstract:transfer_proxy local:transfer_proxy`,
  `npu_transfer_proxy.exe`). Our board is reached over Tailscale/SSH, and as
  already noted `list_devices()` raises from `get_ntb_devices()` with no
  USB-attached board. `init_runtime(target="rk3588", device_id=...)` cannot
  work over SSH.
- **`load_rknn` is not runnable on the simulator**, so a pre-built `.rknn`
  cannot be profiled that way either: *"RKNN model that loaded by 'load_rknn'
  not support inference on the simulator, please set 'target' first!"*.

### The board runtime (`librknnrt.so` 2.3.0) does NOT implement the perf queries

The header-level query codes are accepted as *commands* but are not supported
by this build:

```
E RKNN: rknn_query, cmd = 100 is not in [0, 17]!     # codes 4 and 5 are in range
PERF_RUN (cmd 5):   rc=-5                            # -5 = unsupported
PERF_DETAIL (cmd 4): rc=-5
```

Probed with correctly-sized `rknn_perf_run` (8 bytes) and `rknn_perf_detail`
(272 bytes) structs, so the rejection is genuinely "unsupported", not "buffer
too small" -- the runtime's own size guards
(`info_len < sizeof(rknn_perf_run)`) would have reported differently. The
binary does contain `rknn_perf_detail`/`rknn_perf_run` handling strings, so the
code path is compiled in but disabled in this release.

There is also **no `rknn_set_perf_debug` export** in this `librknnrt.so`
(verified with `nm -D`), so there is no way to switch the profiler on from the
board side.

Reproduce with `scripts/rknn/probe_nanopc_perf.py` (uploads a model, runs it,
then queries every perf code and prints the return codes).

### A newer runtime does not help (verified)

The board runs runtime **2.3.0**, so the obvious next step was a newer one.
Checked every available source:

- **PyPI**: `rknn-toolkit2` latest is still **2.3.2** (what the host already
  has); `rknn-toolkit-lite2` is also **2.3.2**. Upstream's newest GitHub
  release is **v2.3.2** (April 2025) -- there is no 2.4/2.5 anywhere public.
- **`rknn-toolkit-lite2` 2.3.2's aarch64 wheel ships no `librknnrt.so`** -- it
  dlopens the host's (`strings`: *"Please put it into /usr/lib/ directory"*,
  `LIBRKNNRT_PATH`). Its `rknn_perf` extension module is **memory profiling
  only** (`collect_memory_detail`, `format_memory_detail`) -- no layer timing.
- **The repo's master branch does carry a newer runtime**: downloaded
  `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so` from `master` --
  **2.3.2** (`429f97ae6b@2025-04-09T09:09:27`), newer than the board's 2.3.0.
  Side-loaded onto the board via `nanopc_rknn_runner.py --library` (the
  system library is untouched) and re-probed.

Result: **2.3.2 behaves identically.** Same 34 exported `rknn_*` symbols (still
no `rknn_set_perf_debug`), same query range `[0, 17]`, and both perf codes still
return `-5`:

| runtime | PERF_RUN (5) | PERF_DETAIL (4) | query range |
| --- | ---: | ---: | --- |
| 2.3.0 (board system) | -5 | -5 | [0, 17] |
| 2.3.2 (upstream master) | -5 | -5 | [0, 17] |

2.3.2 does load the board's 2.3.0-built models and execute them, producing
**bit-for-bit identical output** (`np.array_equal` True, max diff 0.0) and the
same `RKNN_SOC_RK3588`/`cores: 3` device report. So upgrading the runtime is
safe and mildly faster on this micro-benchmark, but it **does not unlock
per-layer profiling** -- the feature is simply absent from both builds.

### A newer *driver* is not available either -- and would not help

The other half of the stack is the kernel driver (`rknpu.ko`). The board already
runs the **newest vendor driver that exists**: **v0.9.8** (confirmed via
`/sys/kernel/debug/rknpu/version`, and the vendor's own version numbering tops
out there). `rknn-toolkit2` 2.3.2 was already the newest compiler.

There *is* a newer driver, but it is a different stack, not an upgrade:

- **Mainline `accel/rocket`** (Tomeu Vizoso, merged Linux **6.18**, Nov 2025) is
  the in-tree RK3588 NPU driver. It is deliberately minimal -- "just powers the
  hardware on and off, allocates and maps buffers, and submits jobs to the
  frontend unit. Everything else is done in userspace" -- with four ioctls, no
  compiler and no model format. All compute lives in Mesa's `rocket` Gallium
  driver (via the Teflon TFLite delegate), which accelerates **only convolutions**
  (per-tensor quantization, no dilation) plus additions fused into the preceding
  convolution. It **cannot load `.rknn` models at all**, so it is useless for
  this repo's ONNX -> RKNN -> onnxsim comparison. It is also clock-pinned at
  200 MHz with no devfreq (~2.6x throughput left on the table: 91 vs 242.8
  inferences/s on MobileNet V1), does not gang cores for one model, and supports
  RK3588 only (RK3576 posted upstream, unmerged).
- **`rkopnu`** ([marfrit/rkopnu](https://github.com/marfrit/rkopnu)) is the
  interesting middle path: a fork of `accel/rocket` with its uAPI swapped for the
  **vendor `rknpu` ioctl ABI**, so the closed vendor `librknnrt.so` (and thus
  real `.rknn` models) runs on a mainline kernel. It reports vendor parity for a
  llama.cpp prefill workload (~51.5 vs ~51.6 tok/s) and adds a 1 GHz OPP table and
  `core_mask` honouring. **It cannot be tried on this board**, verified directly:
  - It requires `CONFIG_DRM_ACCEL=y` and the RK3588 NPU/IOMMU device-tree nodes.
    This board's kernel is the Rockchip **BSP** 6.1.141, whose
    `/lib/modules/6.1.141/kernel/drivers/` has **no `accel/` directory** at all
    (the accel subsystem is mainline-only). The repo is a fork of rocket's
    *sources*, so it needs a mainline kernel to build against.
  - There is no kernel build tree to build against:
    `/lib/modules/6.1.141/build` does not exist and no matching
    `linux-headers` package is available.
  - Even with a mainline kernel, its NPU nodes ship `status = "disabled"` in
    mainline `rk3588-base.dtsi` and need a board-specific overlay.

  Installing it would therefore mean replacing the board's working BSP kernel --
  a reboot-riskier, out-of-scope change for this investigation.

**Neither driver route adds per-layer profiling.** That capability is absent from
`librknnrt.so` itself (verified on both 2.3.0 and 2.3.2 above), and rkopnu is
explicitly just the kernel ABI for that same closed runtime. So the profiling
conclusion is unchanged by any driver work.

### What *is* available on the board

Only coarse utilization, via debugfs (no per-layer detail):

```
/sys/kernel/debug/rknpu/{load,freq,power,volt,delayms,version,reset}
# e.g. NPU load: Core0: 27%, Core1: 0%, Core2: 0%
```

### Practical upshot for the onnxsim question

The "does simplification change NPU scheduling?" question from the section above
**cannot be settled with per-layer profiling on this board**, because no
per-layer profiling is exposed. Settling it would need either a board reachable
over ADB (so the host toolkit's `eval_perf`/`perf_debug` path works), or
Rockchip's `rknn-toolkit-lite2` + a newer `librknnrt` whose perf queries are
enabled. The proxy signals used above -- compiled artifact size, and
`/sys/kernel/debug/rknpu/load` during a sustained run -- remain valid but
coarse.

## Device access findings (all verified on the board)

1. **The NPU is a DRM render node, not `/dev/rknpu`.** This image's RKNPU driver
   registers as a DRM device -- `dmesg` reports
   `[drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1`, and
   `/dev/rknpu` does not exist. The nodes are `/dev/dri/card1` and
   `/dev/dri/renderD129`, owned by group `render` (the stock `pi` user is
   already a member). `librknnrt.so` 2.3.0 finds the node itself -- its strings
   contain both the modern `/dev/dri/%s` + `%s/renderD%d` and the legacy
   `/dev/rknpu` fallback -- so **no device path needs to be passed in**. Code or
   docs that hardcode `/dev/rknpu` will fail on this board.

2. **`librknnrt.so` 2.3.0 exports no `rknn_get_sdk_version`.** Confirmed via
   `nm -D`. It does export `rknn_get_device_properties`, which is the better
   check anyway since it returns the actual SoC. The runtime version string is
   only available as text embedded in the binary
   (`strings ... | grep "librknnrt version"`).

3. **`rknn_get_device_properties`' first field is an enum, not a string.**
   `rknn_device_properties.soc_id` is `rknn_soc_id`; reading it as `char[32]`
   returns `'\x02'`, which is in fact `RKNN_SOC_RK3588 == 2`.

4. **A `want_float` buffer must be sized from `size_with_stride`, not
   `n_elems`.** `rknn_build()` rewrites the default I/O dtype to int8 (it warns
   about this), and the float32 view the runtime converts into needs the same
   *stride-padded* extent at 4 bytes/element. Sizing from `n_elems` silently
   under-allocates for padded tensors:

   ```
   E RKNN: rknn_set_io_mem, input memory size(1728) < model input size(2304)
   ```

   (1728 = 432x4, 2304 = 576x4, for `sigmoid_mul_swish`'s `[1,12,12,3]` input
   with `size=432`, `size_with_stride=576`). Models whose layout happens to be
   unpadded (`conv_bn_relu`, where `size_with_stride == size`) work either way,
   so this only shows up on some models.

5. **`mean_values`/`std_values` must match a rank-4 input's channel count.**
   RKNN rejects a mismatch outright:
   `The len of mean_values ([0.0]) for input 0 is wrong, expect 32!`. For
   non-rank-4 inputs they must be omitted entirely so RKNN applies identity
   scaling itself.

6. **Host-side `list_devices()` does not work over SSH.** `rknn-toolkit2`'s
   ADB/NTB-based device discovery raises `RuntimeError` from `get_ntb_devices()`
   with no USB-attached device. Over a network board, the host compiles and the
   board runs -- the same split `scripts/rknn/benchmark_luckfox.py` uses.

## Reproducing

```bash
# Host: build the two benchmark models for rk3588.
PYTHONPATH=. python scripts/rknn/build_nanopc_models.py --output-dir /tmp/nanopc-rk3588

# Benchmark them on the board.
RKNN_SSH_PASSWORD=... python scripts/rknn/benchmark_nanopc.py \
  /tmp/nanopc-rk3588/*.rknn --warmup 20 --iterations 200

# The original-vs-simplified correctness check on real silicon.
RKNN_SSH_PASSWORD=... python scripts/rknn/run_rknn3588_compat.py \
  --output rknn3588-compat.csv

# The interleaved A/B latency comparison (the defensible way to time this).
RKNN_SSH_PASSWORD=... python scripts/rknn/ab_latency_nanopc.py \
  --rounds 20 --output ab.json
```

See `scripts/rknn/README.md` for what this tier does and does not cover.