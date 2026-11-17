# Parallel CPU decode follow-up

Measured 2026-11-18 on the same Q8B, Qwen3-1.7B Q4_K_M file and tinygrad
v0.14.0 source commit recorded in [the original results](../README.md).
Both baseline and threaded runs unpack the weights once into FP16 storage.
This experiment changes CPU execution, not model weights or quantization.

## Matched measurements

Each run uses context capacity 256, three generated warmup tokens and
30 measured tokens. Baseline and threaded runs use the same CPU partition
limit (`NUM_CPU_THREADS=4`) and affinity (CPUs 4–7). They ran sequentially
in alternating order, in separate processes:

| Run | Workers | Tokens/s | Median seconds/token |
|---|---:|---:|---:|
| Baseline A | 0 | 1.950 | 0.512 |
| Threaded A | 4 | 3.241 | 0.307 |
| Baseline B | 0 | 1.955 | 0.511 |
| Threaded B | 4 | 3.232 | 0.308 |

The two threaded runs improve throughput by about **66%** over the matched
serial runs. All 33 generated token IDs agree across these four runs.
The prompt is `The capital of France is`. Raw samples are in the `*-30.json`
files. tinygrad's allocation counter is unchanged at **3,472,381,992 bytes**;
the pool adds host threads rather than another copy of model weights.

For `1 + 1 =`, the corresponding 30-sample runs produced 1.96 tokens/s
with the serial runtime and 3.18 tokens/s with the pool. All 33 tokens
matched for that prompt too: **66 matching token IDs across two prompts**.
`comparison.json` records the aggregate rates and comparisons.

The threaded samples have more variance than the serial baseline, including
occasional slow tokens. Rates above include those samples. CPU frequency
was not pinned, and startup/compilation costs are excluded from steady
decode timings. These remain short-context CPU measurements, not HMX or
long-context prefill results.

## Why this helps

The pinned synchronous CPU runtime implements `CPUProgram.__call__` by
looping over `core_id` partitions and invoking each native function serially.
`NUM_CPU_THREADS` controls generated partitions but does not by itself make
this loop concurrent. See the
[pinned CPU runtime](https://github.com/tinygrad/tinygrad/blob/6f87158d77f66a36d5f8bbe915170b24e2acabe8/tinygrad/runtime/ops_cpu.py).

`cpu_thread_pool.py` provides an opt-in context that dispatches those
partitions to persistent threads. Each task has its own argument array and
the caller waits for every partition to finish. ctypes releases the GIL
during native execution. LVP, kernels without `core_id`, and single-partition
calls retain the original execution path.

The context temporarily replaces `CPUProgram.__call__` process-wide and
restores it on exit, including exceptions. It targets this pinned internal
API, rejects HCQ2, and should not be installed concurrently by unrelated
sessions in one process. The default benchmark behavior remains unchanged:
`--cpu-workers=0`.

## Tuning and rejected alternatives

A `DEBUG=2` profile identified 487 kernels in one replay. The feed-forward
gate/up projections accounted for roughly 52% of kernel time;
the vocabulary projection accounted for another 16%. `profile-summary.json`
records the grouped timings. Profiling overhead is excluded from the matched
measurements above.

Generation-only BEAM search was also tried, leaving weight loading at BEAM=0:

| Exploratory configuration (12 measured tokens) | Tokens/s |
|---|---:|
| Serial, 8 partitions, generation BEAM=1 | 2.530 |
| Serial, 8 partitions, generation BEAM=2 | 2.502 |
| Thread pool, 4 partitions, generation BEAM=1 | 2.808 |

All produced the saved 15-token sequence. Neither wider search nor combining
search with the pool beat the untuned pool. The combined experiment uses a
separate cache database: tuning choices measured with the serial dispatcher
should not silently stand in for choices measured with a parallel dispatcher.
These exploratory results are not interleaved repetitions.

## Reproduce

Use the source checkout and Python environment from the parent report. On
the Q8B the model, code, logs and caches are under
`/home/radxa/tinygrad-opt/llm`; follow-up artifacts are in `tune2/`.

```sh
export PYTHONPATH=/home/radxa/tinygrad-opt/tinygrad-source
export DEV=CPU NUM_CPU_THREADS=4 BEAM=0 HALF=1

taskset -c 4-7 python scripts/tinygrad/bench_llm.py Qwen3-1.7B-Q4_K_M.gguf \
  --realize --tokens 30 --json baseline.json
taskset -c 4-7 python scripts/tinygrad/bench_llm.py Qwen3-1.7B-Q4_K_M.gguf \
  --realize --tokens 30 --cpu-workers 4 --json threaded.json
```

To tune generation independently of loading, use `--decode-beam 1`. Use a
separate `CACHEDB` when changing dispatcher/partition configurations.

Validation includes concurrency and argument-isolation checks, exception
propagation and restoration, fallback paths, HCQ2 rejection, and a real
partitioned matvec compared bit-exactly with serial execution on three
distinct inputs. Run the five checks with:

```sh
python tests/test_tinygrad_cpu_thread_pool.py -v
```
