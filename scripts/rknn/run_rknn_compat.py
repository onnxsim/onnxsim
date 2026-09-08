#!/usr/bin/env python3
"""Run the RKNN compatibility check over the model suite and report.

Drives ``worker.py`` once per model (isolated subprocess, hard timeout),
collects the JSON results, writes a CSV, prints a summary, and exits non-zero
if any model failed. This is the entry point the ``RKNN Integration``
workflow calls.

A model **fails** the check when simplification breaks RKNN convert/run
compatibility or changes the PC-simulator result (``rknn_regression``), when
onnxsim raises (``simplify_error``), or when the worker crashes/times out. A
graph `rknn-toolkit2` cannot convert or run even *before* simplification is
``unsupported`` and is **reported, not failed** -- that is a converter
limitation, not an onnxsim bug.

If `rknn-toolkit2` is unavailable on the host, every model reports
``skipped`` and the run passes (nothing to test) unless ``--require-rknn`` is
given.

Usage:
    run_rknn_compat.py --output rknn-compat.csv
    run_rknn_compat.py --require-rknn         # fail if rknn-toolkit2 is missing
    run_rknn_compat.py --models conv_bn_relu matmul_bias_tanh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import models  # noqa: E402

FAIL_STATUSES = {"rknn_regression", "simplify_error", "crash", "timeout", "error"}


def run_one(model_name: str, timeout: int) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py"), model_name],
            capture_output=True,
            text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "status": "timeout",
            "error": f">{timeout}s",
            "seconds": timeout,
        }
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__") :])
    if result is None:
        result = {
            "model": model_name,
            "status": "crash",
            "error": (proc.stderr or proc.stdout or "no result line")[-400:],
            "seconds": round(time.time() - t0, 1),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="subset of model names to run (default: the whole suite)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-model wall-clock cap in seconds; <=0 disables",
    )
    ap.add_argument(
        "--require-rknn",
        action="store_true",
        help="fail if rknn-toolkit2 is unavailable instead of skipping",
    )
    ap.add_argument("--output", default="rknn-compat.csv")
    args = ap.parse_args()

    selected = args.models or models.names()
    print(f"RKNN compatibility check | {len(selected)} models", flush=True)

    rows = []
    failures = []
    skipped = 0
    for i, name in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name} ...", end=" ", flush=True)
        r = run_one(name, args.timeout)
        rows.append(r)
        status = r.get("status")
        if status == "skipped":
            skipped += 1
            print(f"skipped ({r.get('error')})", flush=True)
            continue
        detail = ""
        if r.get("orig_nodes") is not None:
            detail = f"{r.get('orig_nodes')}->{r.get('simp_nodes')} nodes"
        if r.get("diff_vs_rknn_orig") is not None:
            detail += f", d(orig)={r.get('diff_vs_rknn_orig'):.2g}"
        print(f"{status} ({detail}) {r.get('seconds')}s", flush=True)
        if status in FAIL_STATUSES:
            failures.append((name, status, str(r.get("error"))[:200]))

    fields = [
        "model",
        "status",
        "orig_nodes",
        "simp_nodes",
        "diff_vs_rknn_orig",
        "diff_vs_cpu_ref",
        "seconds",
        "error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.output} ({len(rows)} rows)", flush=True)

    if skipped == len(selected):
        msg = "rknn-toolkit2 unavailable on this host; all models skipped."
        if args.require_rknn:
            print(f"\n{msg} (--require-rknn set -> failing)", flush=True)
            return 1
        print(f"\n{msg} Nothing to test; passing.", flush=True)
        return 0

    if failures:
        print(f"\n{len(failures)} FAILED:", flush=True)
        for name, status, err in failures:
            print(f"  - {name}: {status} {err}", flush=True)
        return 1
    passed = sum(1 for r in rows if r.get("status") == "ok")
    unsupported = sum(1 for r in rows if r.get("status") == "unsupported")
    print(
        f"\nall passed ({passed} ok, {unsupported} unsupported, {skipped} skipped)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
