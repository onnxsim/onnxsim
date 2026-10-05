#!/usr/bin/env python3
"""Benchmark the Allwinner NPU through onnx-remote-viplite-worker, with an ONNX Runtime CPU baseline on the same device.

    bench.py --serial SERIAL [--dir /data/local/tmp/viplite] --nb a.nb --nb b.nb \\
             [--cpu-onnx yolov5s_rt.onnx --cpu-nb yolov5s_rt_uint8_a733.nb] [--repeats 3] [--out results.md]

Everything runs on the device through `adb shell`, one job at a time (the NPU and CPU runs must not overlap). The device directory
must already hold the worker (`onnx-remote-viplite-worker`), the networks, and for the CPU baseline `bench_ort_cpu` +
`libonnxruntime.so` (see README). Each configuration is measured `--repeats` times; the table reports the median of the repeated
medians and the spread between repeats, so a noisy device shows up as a wide range instead of a confident number.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys


def adb(serial, command, timeout=900):
    out = subprocess.run(
        ["adb", "-s", serial, "shell", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return out.stdout + out.stderr


def thermal(serial):
    """Highest thermal-zone temperature in deg C, or None when the zones are not readable from the shell."""
    out = adb(serial, "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null")
    temps = [int(t) for t in out.split() if t.lstrip("-").isdigit()]
    return max(temps) / (1000 if max(temps, default=0) > 1000 else 1) if temps else None


def parse_npu(text):
    """Pull the per-phase medians out of `--bench` output."""
    total = re.search(
        r"median ([\d.]+) ms, min ([\d.]+) ms, max ([\d.]+) ms over (\d+)", text
    )
    if not total:
        raise RuntimeError("no timing in worker output:\n" + text)
    phases = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"(viplite_\w+): median ([\d.]+) ms", text)
    }
    load = re.search(r"load\+prepare: ([\d.]+) ms", text)
    tput = re.search(r"throughput with \d+ concurrent callers: ([\d.]+) calls/s", text)
    inp = re.search(r"input\s+\S+ fmt=(\d+) .* dims=([\d ]+)", text)
    return {
        "total_ms": float(total.group(1)),
        "npu_ms": phases.get("viplite_run"),
        "in_ms": phases.get("viplite_input"),
        "out_ms": phases.get("viplite_output"),
        "load_ms": float(load.group(1)) if load else None,
        "calls_per_s": float(tput.group(1)) if tput else None,
        "input": inp.group(2).split() if inp else None,
    }


def bench_npu(a, nb):
    runs = []
    for _ in range(a.repeats):
        pin = f"taskset {a.taskset} " if a.taskset else ""
        text = adb(
            a.serial,
            f"cd {a.dir} && {pin}./onnx-remote-viplite-worker --bench {nb} {a.iters} --threads {a.threads}",
        )
        if "load:" in text and "vip_create_network failed" in text:
            return {"error": "not a valid NBG for this NPU generation"}
        runs.append(parse_npu(text))
    return {
        k: statistics.median(r[k] for r in runs) if runs[0][k] is not None else None
        for k in ("total_ms", "npu_ms", "in_ms", "out_ms", "load_ms", "calls_per_s")
    } | {
        "npu_spread_ms": (
            min(r["npu_ms"] for r in runs),
            max(r["npu_ms"] for r in runs),
        ),
        "input": runs[0]["input"],
    }


def bench_cpu(a, onnx_path, threads):
    meds = []
    for _ in range(a.repeats):
        text = adb(
            a.serial,
            f"cd {a.dir} && LD_LIBRARY_PATH=. ./bench_ort_cpu {onnx_path} {threads} {a.cpu_iters}",
        )
        m = re.search(r"median ([\d.]+) ms", text)
        if not m:
            raise RuntimeError("no timing from bench_ort_cpu:\n" + text)
        meds.append(float(m.group(1)))
    return {"median_ms": statistics.median(meds), "spread_ms": (min(meds), max(meds))}


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--serial", required=True)
    p.add_argument("--dir", default="/data/local/tmp/viplite")
    p.add_argument(
        "--nb",
        action="append",
        default=[],
        help="network (file name inside --dir); repeatable",
    )
    p.add_argument(
        "--cpu-onnx", help="ONNX model (inside --dir) for the ONNX Runtime CPU baseline"
    )
    p.add_argument("--cpu-nb", help="the --nb that --cpu-onnx is the same network as")
    p.add_argument(
        "--cpu-threads",
        default="1,4,8",
        help="CPU thread counts for the ONNX Runtime baseline (default 1,4,8)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="concurrent callers against the worker (1 = serial; 2 overlaps conversion with the NPU)",
    )
    p.add_argument(
        "--taskset",
        help="hex CPU mask for the worker (e.g. 80 = one big core on the A733); the default placement is noisy",
    )
    p.add_argument("--iters", type=int, default=100, help="NPU iterations per repeat")
    p.add_argument(
        "--cpu-iters", type=int, default=10, help="CPU iterations per repeat"
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", help="write the markdown table here")
    p.add_argument("--json", help="write raw results here")
    a = p.parse_args(argv)

    result = {"temp_start_c": thermal(a.serial), "npu": {}, "cpu": {}}
    for nb in a.nb:
        print(f"NPU {nb} ...", file=sys.stderr)
        result["npu"][nb] = bench_npu(a, nb)
    if a.cpu_onnx:
        for t in a.cpu_threads.split(","):
            print(f"CPU {a.cpu_onnx} threads={t} ...", file=sys.stderr)
            result["cpu"][t] = bench_cpu(a, a.cpu_onnx, int(t))
    result["temp_end_c"] = thermal(a.serial)

    setup = f"{a.threads} concurrent caller(s)" + (
        f", worker pinned with taskset {a.taskset}" if a.taskset else ""
    )
    lines = [
        f"Worker: {setup}.",
        "",
        f"| network | input dims (as the NBG reports them) | NPU ms | in ms | out ms | total ms | calls/s | NPU range ({a.repeats} repeats) | load ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nb, r in result["npu"].items():
        if "error" in r:
            lines.append(f"| {nb} | - | {r['error']} | | | | | | |")
            continue
        lo, hi = r["npu_spread_ms"]
        # calls/s is only reported for concurrent callers; serially it is 1000 / total_ms.
        cps = r["calls_per_s"] or 1000.0 / r["total_ms"]
        lines.append(
            f"| {nb} | {'x'.join(r['input'])} | {fmt(r['npu_ms'])} | {fmt(r['in_ms'])} | {fmt(r['out_ms'])} | "
            f"{fmt(r['total_ms'])} | {cps:.1f} | {lo:.2f}-{hi:.2f} | {fmt(r['load_ms'], 1)} |"
        )
    if result["cpu"]:
        ref = result["npu"].get(a.cpu_nb, {})
        lines += [
            "",
            f"ONNX Runtime CPU, {a.cpu_onnx} (fp32), same device:",
            "",
            "| threads | median ms | range | NPU speedup (NPU time / incl. conversion) |",
            "|---|---|---|---|",
        ]
        for t, r in result["cpu"].items():
            lo, hi = r["spread_ms"]
            sp = (
                f"{r['median_ms'] / ref['npu_ms']:.0f}x / {r['median_ms'] / ref['total_ms']:.0f}x"
                if ref.get("npu_ms")
                else "-"
            )
            lines.append(f"| {t} | {r['median_ms']:.1f} | {lo:.1f}-{hi:.1f} | {sp} |")
    lines += [
        "",
        f"Device temperature (max thermal zone): start {result['temp_start_c']}, end {result['temp_end_c']} C",
    ]
    table = "\n".join(lines)
    print(table)
    if a.out:
        open(a.out, "w").write(table + "\n")
    if a.json:
        open(a.json, "w").write(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
