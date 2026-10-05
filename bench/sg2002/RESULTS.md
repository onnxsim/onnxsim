# SG2002 ResNet-50 profiling results

Captured on 2026-10-05 using CVI Runtime 1.4.0, a 700 MHz CV181x TPU, and TPU-MLIR-generated
INT8-input ResNet-50 models at 224×224. Models were run directly with the board's `model_runner`;
normal timing excludes compile and file transfer. Throughput is batch size divided by the
reported per-batch device time.

## Device timing

| Build | Batch | Runs | Time per batch | Images/s |
| --- | ---: | ---: | ---: | ---: |
| Default | 1 | 100 | 32.474 ms | 30.79 |
| Default | 2 | 60 | 63.831 ms | 31.33 |
| Default | 4 | 60 | 126.596 ms | 31.60 |
| Opt 2 + Winograd | 1 | 100 | 32.380 ms | 30.88 |

Increasing batch size to four improved throughput by less than 3%; per-image latency remained
near 32 ms. Opt 2 + Winograd changed single-image latency by about 0.3% in this run.

## PMU counters

PMU values below are medians of 60–100 per-inference samples after discarding one initial sample
per process. Engine active times overlap, so TIU and TDMA times must not be added as sequential
stages.

| Batch | Inference | TIU active | TIU % | TDMA active | TDMA % | Reported traffic rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.31 ms | 26.98 ms | 83.50% | 19.45 ms | 60.18% | 1.840 GB/s |
| 2 | 63.64 ms | 53.89 ms | 84.67% | 36.52 ms | 57.38% | 1.763 GB/s |
| 4 | 126.33 ms | 107.77 ms | 85.31% | 71.48 ms | 56.58% | 1.734 GB/s |

The compiler's Opt 2 + Winograd build measured 32.18 ms inference, 26.90 ms TIU-active, and
19.22 ms TDMA-active at batch 1, so the difference from the default build was small.

## Driver usage counter

With `/proc/tpu/usage_profiling` enabled during a 200-inference batch-1 run (6.47 s), valid
one-second driver samples were 97%, 100%, 97%, 100%, and 97%. The initial sample was
135438975% during counter startup and was discarded as out of range. Profiling was disabled
again after the run.

The driver counter measures busy time, not MAC datapath occupancy. These measurements show
little idle time at the driver level, but they do not yet explain why useful compute throughput
is well below the nominal peak. The available TPU-MLIR profile/command decoders do not support
CV181x per-layer attribution, so that remains an open profiling gap.
