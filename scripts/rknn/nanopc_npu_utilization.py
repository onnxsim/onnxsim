#!/usr/bin/env python3
"""Sample /sys/kernel/debug/rknpu/load on the board while inference runs.

Reports how much of the RK3588's NPU the harness actually keeps busy, per core,
as a time-averaged distribution rather than a single sample.

Why time-average: `/sys/kernel/debug/rknpu/load` is an instantaneous duty-cycle
counter per core, and sampling it once tells you almost nothing -- an early
measurement in this repo took a single reading and saw `Core0: 27%`, which is
not representative of either the peak or the sustained value. This runs the
inference in a tight loop for the whole measurement window and samples the
counter from a second process, so the number reflects the duty cycle the model
actually sustains.

Note on core_mask: `rknn_set_core_mask` is not called by the harness, so
`librknnrt.so` picks its own policy. On this board `rknn_get_device_properties`
reports `cores: 3`, and a single-model RKNN model is *not* split across cores --
so a one-core reading is the expected shape for a single small model, and
cross-core utilisation only appears for workloads the runtime chooses to
schedule that way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

_LINE = re.compile(r"Core(\d+):\s*(\d+)%")


def _ssh_prefix(args: argparse.Namespace) -> list[str]:
    options = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]
    base = ["sshpass", "-p", args.ssh_password] if args.ssh_password else []
    return [*base, "ssh", *options, f"{args.user}@{args.host}"]


def _read_load(args: argparse.Namespace) -> dict[int, int]:
    proc = subprocess.run(
        [*_ssh_prefix(args), "echo pi | sudo -S cat /sys/kernel/debug/rknpu/load"],
        capture_output=True, text=True, timeout=30,
    )
    found = {int(m.group(1)): int(m.group(2)) for m in _LINE.finditer(proc.stdout)}
    if not found:
        raise RuntimeError(f"could not parse NPU load from: {proc.stdout!r}")
    return found


def _board_loop(args: argparse.Namespace, runner: str, model: str,
                feeds: str, work: str, stop: threading.Event) -> None:
    """Keep the NPU busy for the whole measurement window.

    Termination note: killing the local ``ssh`` client does **not** stop the
    remote ``python3`` -- it survives the session and keeps hammering the NPU
    (observed: six runners left behind, NPU still reading 79/64/45% after the
    host-side processes were gone). So the loop wraps the command in a unique
    marker and kills it by that marker on the board before returning.
    """
    marker = f"rknn-util-{os.getpid()}"
    inner = (
        f"cd {work} && exec -a {marker} python3 {runner} {model} "
        f"--input-feeds {feeds} --warmup 5 --iterations 100000 "
        f"--output-dir {work}/util_out"
    )
    proc = subprocess.Popen(
        [*_ssh_prefix(args), inner],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        stop.wait()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Reap anything the local ssh teardown left running on the board.
        subprocess.run(
            [*_ssh_prefix(args),
             f"pkill -f {marker} || true; sleep 1; "
             f"pkill -9 -f {marker} || true"],
            capture_output=True, text=True, timeout=30,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="nanopc-t6.tailf0b7b1.ts.net")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--ssh-password", default=None)
    ap.add_argument("--model", required=True,
                    help="path on the board of the .rknn model to drive")
    ap.add_argument("--runner", default="/tmp/rknn-util/nanopc_rknn_runner.py")
    ap.add_argument("--feeds", default="/tmp/rknn-util/feeds.npz")
    ap.add_argument("--work", default="/tmp/rknn-util")
    ap.add_argument("--duration", type=float, default=25.0,
                    help="measurement window in seconds")
    ap.add_argument("--interval", type=float, default=0.4)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.ssh_password is None:
        args.ssh_password = "pi"

    idle = [_read_load(args) for _ in range(3)]
    print("idle baseline:", idle, flush=True)

    stop = threading.Event()
    driver = threading.Thread(
        target=_board_loop, args=(args, args.runner, args.model, args.feeds,
                                  args.work, stop), daemon=True,
    )
    driver.start()
    time.sleep(2.0)  # let the first inference land before sampling

    samples: list[dict[int, int]] = []
    deadline = time.time() + args.duration
    while time.time() < deadline:
        try:
            samples.append(_read_load(args))
        except Exception:
            pass
        time.sleep(args.interval)

    stop.set()
    driver.join(timeout=20)

    if not samples:
        print("no samples collected", file=sys.stderr)
        return 1

    cores = sorted({core for sample in samples for core in sample})
    per_core = {
        core: [sample[core] for sample in samples if core in sample]
        for core in cores
    }
    summary = {
        "model": args.model,
        "samples": len(samples),
        "window_s": args.duration,
        "per_core": {
            f"core{core}": {
                "mean_pct": round(statistics.mean(values), 1),
                "median_pct": round(statistics.median(values), 1),
                "max_pct": max(values),
                "min_pct": min(values),
            }
            for core, values in per_core.items()
        },
    }
    # Overall busiest-core utilisation, the headline number.
    summary["busiest_core_mean_pct"] = round(
        max(statistics.mean(values) for values in per_core.values()), 1
    )
    summary["npu_3core_capacity_used_pct"] = round(
        sum(statistics.mean(values) for values in per_core.values()) / 3.0, 1
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())