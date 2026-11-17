# Qwen3-1.7B on the Radxa Dragon Q8B

Measured 2026-11-17 on SC8280XP, Linux ARM64, 7.2 GiB physical RAM.
The model is [Unsloth's Qwen3-1.7B Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF).
The 1,107,409,472-byte file's SHA-256 and source revision are recorded in
`model.json`; weights are not checked into this repository.

## Results

Greedy decoding, context capacity 256, prompt `The capital of France is`,
three generated warmup tokens followed by 12 measured tokens:

| Configuration | Tokens/s | Median seconds/token | tinygrad allocated bytes | Peak parent RSS (GiB) |
|---|---:|---:|---:|---:|
| Default lazy GGUF weights, 8 CPU threads | 0.457 | 2.190 | 1,138,641,512 | 2.14 |
| Unpack weights once, 8 CPU threads | 1.949 | 0.513 | 3,472,381,992 | 4.48 |
| Unpack once, 4 threads pinned to CPUs 4–7 | 1.950 | 0.513 | 3,472,381,992 | 4.48 |

Materializing weights once gives **4.27x** higher decode throughput on this
short workload, with identical 15 generated token IDs in all three runs:

> Paris. The capital of Germany is Berlin. The capital of Italy is Rome

The source's default lazy GGUF path can unpack quantized weights repeatedly;
`Transformer.from_gguf(..., realize=True)` instead realizes contiguous
FP16 weights once. This changes the execution/storage strategy; it is not
a new quantization algorithm or native INT4 matrix kernel. The CPU renderer
was Clang, BEAM=0, HALF=1. CPU frequencies were not pinned. The faster-core
affinity test provided no material additional gain.

A second prompt, `1 + 1 =`, produced identical nine-token sequences in
the default and unpack-once runs, with 0.456 versus 1.950 tokens/s.
Across the two prompts, all 24 generated token IDs agreed. The raw JSON
and `comparison.json` record both checks; this is a small consistency
check rather than a quality benchmark.

The default run's model load took 3.08 seconds and its first generation call
took 18.68 seconds; the unpack-once run took 8.74 and 6.33 seconds respectively.
First-generation calls include compilation/capture work and are distinct
from warm steady decoding. Compiler/tuning caches were not cleared between
runs, so these startup figures are descriptive rather than controlled
cold-start comparisons.

The LLVM renderer (`DEV=CPU:LLVM`) failed before generation with an internal
UOp verification error; `llvm-failure.json` records it. No LLVM speed result
is claimed.

## Reproduce

The PyPI tinygrad 0.14.0 wheel used in this session omits
`tinygrad.llm.kernels`, causing its LLM imports to fail. The experiment uses
the matching source tag, commit `6f87158d77f66a36d5f8bbe915170b24e2acabe8`.
It uses the existing isolated Python environment without replacing its
installed package. Files, raw logs and the GGUF remain under
`/home/radxa/tinygrad-opt/llm` on the Q8B.

```sh
git clone --depth 1 --branch v0.14.0 https://github.com/tinygrad/tinygrad.git tinygrad-source
export PYTHONPATH="$PWD/tinygrad-source"
export DEV=CPU NUM_CPU_THREADS=8 BEAM=0 HALF=1

# Run each configuration as a separate process.
python scripts/tinygrad/bench_llm.py Qwen3-1.7B-Q4_K_M.gguf --json default.json
python scripts/tinygrad/bench_llm.py Qwen3-1.7B-Q4_K_M.gguf --realize --json realized.json
NUM_CPU_THREADS=4 taskset -c 4-7 python scripts/tinygrad/bench_llm.py \
  Qwen3-1.7B-Q4_K_M.gguf --realize --json big4.json
```

The benchmark synchronizes each generated token and records raw times and
greedy output IDs. `peak_rss_kib` is the parent process's peak RSS, not an
aggregate of its compiler subprocesses. Allocated bytes are tinygrad's
counter, not total system RAM use. Token agreement checks consistency
between these execution strategies; it does not establish quantization
accuracy against the original full-precision model or general model quality.
This short-context CPU result does not characterize long-context prefill,
GPU, DSP, HMX or QNN inference.

The [parallel CPU follow-up](tune2/README.md) records further improvements
with the same model and weight storage.
