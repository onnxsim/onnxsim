#!/usr/bin/env python3
"""BEAM search over tinygrad Opts for a matmul/conv problem, timed on the phone's GPU.

  PYTHONPATH=<tinygrad 0.14> python tune.py OUTDIR PROBLEM [--rounds 3] [--beam 4] [--backend vulkan|dawn]
                                            [--runner-dir DIR_WITH_tgrun_vk_and_tgrun_dawn]

tinygrad's own BEAM needs a Device it can open on the machine that generates the kernels; here the GPU is on a phone, so this
reuses only tinygrad's device-free half (Scheduler + get_kernel_actions) and does the timing step through tgrun_vk / tgrun_dawn
over adb (under ~/.cache/android-phone/phone-run, the shared phone lock). Each round expands the current top `beam` states by every
single further Opt, renders/compiles all candidates, runs them on the phone, and keeps the fastest `beam` correct ones.
Writes OUTDIR/<problem>/best.txt (best variant, its Opts, GFLOPS per round) and leaves every tried variant in place.
"""

import argparse
import re
import subprocess

import tgk

PHONE_DIR = "/data/local/tmp/tg-kernels"
RESULT = re.compile(
    r"RESULT (\w+) (\S+) (\w+) err=(\S+) compile_ms=(\d+) single_ms=([\d.]+) disp_ms=([\d.]+) gflops=([\d.]+)"
)


def phone(cmd):
    return subprocess.run(
        [
            "bash",
            "-c",
            f"~/.cache/android-phone/phone-run bash -c {subprocess.list2cmdline([cmd])}",
        ],
        capture_output=True,
        text=True,
        timeout=3000,
    ).stdout


def run_on_phone(d, name, names, backend, runner_dir):
    runner = "tgrun_vk" if backend == "vulkan" else "tgrun_dawn"
    cmd = (
        f"adb shell 'mkdir -p {PHONE_DIR}/{name}' >/dev/null; adb push {runner_dir}/{runner} {PHONE_DIR}/ >/dev/null; "
        f"adb push {d}/. {PHONE_DIR}/{name}/ >/dev/null 2>&1; "
        f"adb shell 'cd {PHONE_DIR} && chmod +x {runner} && timeout 1500 ./{runner} {name} {' '.join(names)} 2>&1 | grep -E \"RESULT\"'"
    )
    out = {}
    for line in phone(cmd).splitlines():
        m = RESULT.match(line)
        if m and m[3] == "OK":
            out[m[2]] = float(m[8])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("problem")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--backend", default="vulkan", choices=["vulkan", "dawn"])
    ap.add_argument("--runner-dir", default="/mnt/data/cache/claude-work/wgbench/tg")
    ap.add_argument(
        "--max-new", type=int, default=200, help="cap on candidates rendered per round"
    )
    ap.add_argument(
        "--tiles",
        type=int,
        default=0,
        help="also time this many register-tile seeds (tgk.tile_candidates: 16-64 scalar accumulators per thread) in round 1",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for --tiles")
    args = ap.parse_args()
    p = tgk.parse_problem(args.problem)
    low = tgk.Lowered(p)
    d = f"{args.outdir}/{p.name}"
    header = tgk.write_problem(p, low, d)
    manifest = list(header)
    seen, states = {}, {}
    # seed: tinygrad's default state and the plain gridded kernel
    frontier = [
        ("default", tgk.default_scheduler(low.ast)),
        ("base", tgk.base_scheduler(low.ast)),
    ]
    line, why = tgk.write_variant(d, "default", low.ast, "tinygrad default heuristics")
    assert line, why
    manifest.append(line)
    gflops = run_on_phone_batch = None
    open(f"{d}/manifest.txt", "w").write("\n".join(manifest) + "\n")
    res = run_on_phone(d, p.name, ["default"], args.backend, args.runner_dir)
    best = [("default", res.get("default", 0.0), "tinygrad default heuristics")]
    print(f"round 0: default = {best[0][1]:.1f} GFLOPS")
    count = 0
    for rnd in range(1, args.rounds + 1):
        new = []
        if rnd == 1 and args.tiles:
            for opts, cand in tgk.tile_candidates(low.ast, args.tiles, args.seed):
                seen[opts] = True
                new.append((opts, cand))
        for _, st in frontier:
            for opts, cand in tgk.one_step_candidates(st):
                if opts in seen:
                    continue
                seen[opts] = True
                new.append((opts, cand))
        new = new[: args.max_new + (args.tiles if rnd == 1 else 0)]
        names = []
        for opts, cand in new:
            vname = f"t{count}"
            count += 1
            line, why = tgk.write_variant(
                d, vname, cand.copy().get_optimized_ast(), opts
            )
            if not line:
                continue
            manifest.append(line)
            names.append(vname)
            states[vname] = (opts, cand)
        open(f"{d}/manifest.txt", "w").write("\n".join(manifest) + "\n")
        res = run_on_phone(d, p.name, names, args.backend, args.runner_dir)
        ranked = sorted(res.items(), key=lambda kv: -kv[1])
        print(
            f"round {rnd}: {len(names)} rendered, {len(res)} correct; top: "
            + ", ".join(f"{k}={v:.0f}" for k, v in ranked[:5])
        )
        if not ranked:
            break
        best.append((ranked[0][0], ranked[0][1], states[ranked[0][0]][0]))
        frontier = [(k, states[k][1]) for k, _ in ranked[: args.beam]]
    top = max(best, key=lambda t: t[1])
    open(f"{d}/best.txt", "w").write(
        f"backend {args.backend}\nbest {top[0]} {top[1]:.1f} GFLOPS\nopts {top[2]}\nrounds "
        + " ".join(f"{n}={g:.1f}" for n, g, _ in best)
        + "\n"
    )
    print(
        f"BEST {p.name} [{args.backend}]: {top[0]} = {top[1]:.1f} GFLOPS (default {best[0][1]:.1f})  opts: {top[2][:200]}"
    )


if __name__ == "__main__":
    main()
