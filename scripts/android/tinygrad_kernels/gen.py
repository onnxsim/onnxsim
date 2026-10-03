#!/usr/bin/env python3
"""Generate matmul/conv kernels with tinygrad for WebGPU (WGSL) and Vulkan (SPIR-V).

  PYTHONPATH=<tinygrad 0.14 checkout> python gen.py OUTDIR PROBLEM [PROBLEM ...] [--candidates N]

PROBLEM: mm:MxNxK  or  conv:N,Cin,H,W,Cout,kh,kw[,stride[,pad]]  (tinygrad Tensor.conv2d, NCHW).
Per problem it writes OUTDIR/<problem>/: in<slot>.bin, ref.bin, manifest.txt and, per variant, <v>.wgsl, <v>.comp, <v>.spv.
Variant "default" is what tinygrad emits with its own heuristics. c<i> are the one-step Opt candidates of tinygrad's
autotuner (BEAM's action set, without running it) from the plain gridded kernel, d<i> the same from the default state
(hand-coded heuristics + one more Opt); up to --candidates of each.
"""

import argparse
import sys

import tgk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--candidates", type=int, default=24)
    args = ap.parse_args()
    for spec in args.problems:
        p = tgk.parse_problem(spec)
        low = tgk.Lowered(p)
        d = f"{args.outdir}/{p.name}"
        lines = tgk.write_problem(p, low, d)
        made, skipped = 0, 0
        line, why = tgk.write_variant(
            d, "default", low.ast, "tinygrad default heuristics"
        )
        if line:
            lines.append(line)
            made += 1
        else:
            print(f"  default failed: {why}", file=sys.stderr)
        families = [
            ("c", tgk.base_scheduler(low.ast)),
            ("d", tgk.default_scheduler(low.ast)),
        ]
        for prefix, state in families:
            for i, (opts, sched) in enumerate(
                tgk.one_step_candidates(state)[: args.candidates]
            ):
                line, why = tgk.write_variant(
                    d, f"{prefix}{i}", sched.copy().get_optimized_ast(), opts
                )
                if line:
                    lines.append(line)
                    made += 1
                else:
                    skipped += 1
                    print(f"  {prefix}{i} skipped: {why}", file=sys.stderr)
        open(f"{d}/manifest.txt", "w").write("\n".join(lines) + "\n")
        print(
            f"{p.name}: {made} variants ({skipped} skipped), slots={[(s, n, e) for s, n, e in low.slots]}, {p.flops / 1e9:.3f} GFLOP -> {d}"
        )


if __name__ == "__main__":
    main()
