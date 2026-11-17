#!/usr/bin/env python3
"""Measure greedy tinygrad GGUF decoding; run each configuration in a fresh process.

Example: DEV=CPU NUM_CPU_THREADS=8 python bench_llm.py model.gguf --json run.json
Use --realize to unpack weights once, trading resident memory for decode work.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--realize", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.warmup < 0:
        parser.error("--tokens must be positive and --warmup must be nonnegative")
    if args.context < 1:
        parser.error("--context must be positive")

    from tinygrad import Device, Tensor
    from tinygrad.helpers import NUM_CPU_THREADS, GlobalCounters
    from tinygrad.llm.cli import SimpleTokenizer
    from tinygrad.llm.model import Transformer

    Tensor.manual_seed(42)
    start = time.perf_counter()
    model, metadata = Transformer.from_gguf(args.model, args.context, realize=args.realize)
    Device[Device.DEFAULT].synchronize()
    load_s = time.perf_counter() - start
    tokenizer = SimpleTokenizer.from_gguf_kv(metadata)
    prompt = tokenizer.encode(args.prompt)
    if not prompt:
        parser.error("--prompt must encode to at least one token")
    if len(prompt) + args.warmup + args.tokens >= model.max_context:
        parser.error("prompt and generated tokens must fit the model's effective context")
    generator = model.generate(prompt.copy(), temperature=0.0)
    generated, times = [], []
    for index in range(args.warmup + args.tokens):
        start = time.perf_counter()
        token = next(generator)
        Device[Device.DEFAULT].synchronize()
        elapsed = time.perf_counter() - start
        generated.append(token)
        times.append(elapsed)
        print(json.dumps(dict(step=index, token=token, seconds=elapsed,
                              text=tokenizer.decode([token]))), flush=True)
    measured = times[args.warmup:]
    result = dict(model=str(args.model.resolve()), model_name=metadata.get("general.name"),
                  model_bytes=args.model.stat().st_size, tinygrad=importlib.metadata.version("tinygrad"),
                  device=Device.DEFAULT, target=str(Device[Device.DEFAULT].renderer.target),
                  threads=NUM_CPU_THREADS.value,
                  affinity=sorted(os.sched_getaffinity(0)), realize=args.realize,
                  beam=os.environ.get("BEAM", "0"), half=os.environ.get("HALF", "1"),
                  context=model.max_context, prompt=args.prompt, prompt_tokens=prompt,
                  generated_tokens=generated, text=tokenizer.decode(generated),
                  load_s=load_s, first_token_s=times[0], warmup=args.warmup,
                  decode_samples_s=measured, median_decode_s=statistics.median(measured),
                  tokens_per_second=len(measured) / sum(measured),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  tinygrad_allocated_bytes=GlobalCounters.mem_used,
                  timing="greedy generation with synchronization, excluding first warmup tokens")
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
