# Profiling Metis (Axelera M.2 113m): what works, what doesn't

Notes from bringing up profiling on a Metis M.2 113m card with Voyager SDK 1.9.0
and firmware 1.9.0. Two hosts were used: an aarch64 board (Ubuntu 26.04, kernel
7.0 qcom) and an x86_64 host (Ubuntu 26.04, kernel 7.0 generic, Secure Boot
enabled, later disabled). Use this as a map of the options, with the evidence
for each.

## Summary

| Route | Granularity | Works on 1.9.0? | Notes |
| --- | --- | --- | --- |
| Pipeline `OperatorProfiler`, `axzoo benchmark` | per operator | yes | The compiled model is one `AxRuntimeModel` row, flagged DEVICE WAIT |
| `inference.py --show-stats` | per GStreamer element | yes | Aggregate AIPU time as `inference-task0:inference` |
| Runtime host trace (`AXE_LZ_PERF_LOG_PATH`, `host_profiling=True`) | per frame, kernel and host stages | yes | One line per frame in nanoseconds (see below) |
| Per-layer split (`scripts/axelera/layer_profile.py`) | per layer, each run alone | yes, for layers the SDK can compile alone | Includes per-run overhead; does not sum to the fused model |
| `device_profiling=True` instance property | per kernel | no visible effect | No device-side output; no Level Zero timestamp calls |
| Level Zero tools API (`zet*`, metric groups, tracer) | per kernel | no | Proc-address tables return success, every entry is NULL |
| Level Zero events with kernel timestamps | per kernel | no | `timestampValidBits=0`; query, barrier, and immediate command lists all return `0x78000003` (unsupported feature) |
| AI-core trace (`axtrace --atrace`, `axlogdevice --atrace`) | per core, datapath or megakernel | no data | Command accepted once Secure Boot was off; the trace ring stays empty |
| fwtrace debugfs (`/sys/kernel/debug/metis/...`) | firmware trace sources | partly | Needs lockdown off (Secure Boot off); `enable` works; `aicore0_trace` stays empty |
| MSI interrupt counters (`/proc/interrupts`) | dispatch count | yes | One interrupt per completed inference on vectors p0 and p8 |

## The host trace (usable today)

Set these before creating the runtime context:

```
AXE_LZ_PERF_LOG_PATH=<file>
AXE_PROFILING_CONFIG=HOST_TEMPLATE
```

Load the instance with `host_profiling=True`. Each frame appends one line:

```
[T] idx:0:00000001 beg:<hex> prep:<hex> inp:<hex> krn:<hex> out:<hex> tot:<hex>
```

- All fields are nanoseconds. `beg` is on the monotonic clock (it matches
  `time.monotonic_ns()` to within tens of microseconds).
- `prep + inp + krn + out` equals `tot` within about 5 microseconds.
- The line is flushed before `run()` returns, so a reader can take the newest
  line after each call.
- Truncating the file is safe: the runtime writes from offset 0 afterwards.

`krn` is the kernel time as the runtime sees it. It is the closest thing to
device time that the stack exposes.

## Overhead of an isolated dispatch

Measured on one 3x3 convolution from ResNet-18 (`03_Conv_3`, 1x64x56x56):

| Method | Per call | Calls/s |
| --- | --- | --- |
| Sequential | 574 us | 1712 |
| Streamed, no double buffer | 580 us | 1725 |
| Streamed, double buffer | 355 us | 2816 |
| Double buffer, 2 instances on 2 sub-devices | 183 us | 5481 |
| Double buffer, 4 instances on 4 sub-devices | 162 us | 6156 |

- Batching is not available for this layer: a batch-8 compile fails with
  `Only batch sizes of 1 supported`.
- Direct floor: 1x1 convolutions of increasing size, each run alone (300 frames):

  | Input | MACs | Kernel us | Wall us |
  | --- | --- | --- | --- |
  | 1x1x1x1 | 1 | 773 | 781 |
  | 1x64x1x1 | 4096 | 684 | 693 |
  | 1x64x56x56 | 12.8 M | 1096 | 1129 |

  A dispatch costs about 0.7 ms regardless of arithmetic. Run-to-run noise is
  about 0.1 ms.
- An earlier linear fit over the ResNet-18 convolutions gave an intercept of
  about 1155 us, but its R^2 was 0.07, and the intercept also absorbed the
  activation I/O of the larger layers. Use the direct floor instead.
- The whole ResNet-18 compiles to one kernel ELF and runs in 1722 us (kernel),
  against 27.2 ms for the sum of its isolated convolutions. Per-layer sums
  overstate a fused model by about 16x.

## Fusion

The DPU fusion pass absorbs a convolution's elementwise tail into one datapath.
A Conv+ReLU model compiles to one kernel (641 us kernel time), and a standalone
Conv measures about 505 us. The compiler config does not expose a switch to turn
fusion off (`CompilerConfig(dpu_fuse=False)` is rejected), so the fused and
unfused costs could not be separated directly.

Consequence for profiling: per-layer splits don't see fusion. A ReLU that is
fused into a convolution has no separate timing, and a lone ReLU can't be
compiled by the SDK at all (its graph cleaner treats it as preprocessing).

## Why device-side profiling is out of reach on this stack

1. **`device_profiling=True`** issues no Level Zero timestamp or kernel-profile
   calls, checked by interposing the exported `zeCommandList*`, `zeCommandQueue*`,
   and `zeEvent*` entry points.
2. **Level Zero tools API.** `zetGetMetricGroupProcAddrTable`,
   `zetGetMetricGroupExpProcAddrTable`, `zetGetTracerExpProcAddrTable`,
   `zetGetKernelProcAddrTable`, and `zetGetCommandListProcAddrTable` return
   success, but every function pointer they fill in is NULL.
3. **Level Zero events.** `timestampValidBits=0` on the device. Kernel-timestamp
   queries, event barriers, and immediate command lists are all rejected with
   `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`.
4. **AI-core trace.** `CMD_TRACE_AICORE` is accepted for `datapath` and
   `megakernel` targets once Secure Boot is off, but the trace rings stay empty
   for 200-frame runs. `aicore0` fills with `[B] Machine External interrupt`
   lines instead. The SDK ships `aikernel_test_trace_kernel_simple`, which has the
   datapath and megakernel tracepoints, but no host-side runner executes it.

## Firmware trace (fwtrace) details

- The driver exposes `/sys/kernel/debug/metis/<device>/fwtrace/{stats,enable}`.
- Kernel lockdown in `integrity` mode (set by Secure Boot) blocks debugfs writes.
  `lockdown=none` on the kernel command line is ignored while Secure Boot is on.
- The datastream detaches on every firmware reload, which the runtime triggers on
  connect. Enable a source after the driver is loaded and before the runtime
  connects.
- Stream sources (firmware indices, from `axl-aipu.h`): `aicore0_log` = 5,
  `aicore0_trace` = 6, `aicore1_trace` = 8, and so on; `sysctrl_monitor` = 83.
- Reading a source needs a session (`AXL_IOCTL_FWTRACE_OPEN_SESSION` on the
  device node). None was opened in these tests.

## Interrupts as a dispatch signal

`/proc/interrupts` lists one `msi-metis-<dev>-p<N>` vector per source. For 200
inferences, vectors p0 and p8 each rose by exactly 200. This gives an exact
dispatch count per run, without timestamps and without debugfs.

## Reproduce

- Host trace and per-layer split: `scripts/axelera/layer_profile.py`
  (`python layer_profile.py model.onnx --frames 200`).
- Level Zero event probe: `scripts/axelera/ze_event_probe.cpp`, built against
  the SDK's `level_zero` headers and `libaxruntime.so`.

## Open questions for Axelera support

- Does firmware 1.9.0 support `CMD_TRACE_AICORE` for the 113m, and do the
  datapath and megakernel tracepoints emit on that build?
- Is a trace-enabled firmware or a newer driver planned that reports
  `timestampValidBits` and supports Level Zero command lists?
- Is there a supported way to run `aikernel_test_trace_kernel_simple` on a core?
