#!/usr/bin/env python3
"""Run the RKNN3 `load_llm()` compatibility check and report.

Drives ``worker.py`` once (isolated subprocess, hard timeout -- the export +
convert + PC-simulator build takes tens of seconds for the tiny checkpoint
this harness uses, see `rknn3_backend.py`), collects the JSON result, writes
a one-row CSV, and exits non-zero on failure. Entry point for CI.

A ``rknn3_regression`` (simplification broke `load_llm()` compat or changed
the simulator result) or ``simplify_error`` (onnxsim raised) fails the run.
``unsupported`` (RKNN3-Toolkit rejected even the unsimplified graph) is
reported, not failed -- a converter limitation, not an onnxsim bug. If
`rknn-toolkit` (the RKNN3 one) is unavailable, the run reports ``skipped``
and passes unless ``--require-rknn3`` is given.

Usage:
    run_rknn3_compat.py --output rknn3-compat.csv
    run_rknn3_compat.py --require-rknn3   # fail if RKNN3-Toolkit is missing
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

FAIL_STATUSES = {"rknn3_regression", "simplify_error", "crash", "timeout", "error"}


def run_one(timeout: int) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "worker.py")],
            capture_output=True,
            text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f">{timeout}s", "seconds": timeout}
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result = json.loads(line[len("__RESULT__") :])
    if result is None:
        result = {
            "status": "crash",
            "error": (proc.stderr or proc.stdout or "no result line")[-400:],
            "seconds": round(time.time() - t0, 1),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="wall-clock cap in seconds for the HF download + export + "
        "convert + build + simulate run; <=0 disables",
    )
    ap.add_argument(
        "--require-rknn3",
        action="store_true",
        help="fail if RKNN3-Toolkit is unavailable instead of skipping",
    )
    ap.add_argument("--output", default="rknn3-compat.csv")
    args = ap.parse_args()

    print("RKNN3 load_llm() compatibility check", flush=True)
    r = run_one(args.timeout)
    status = r.get("status")
    detail = ""
    if r.get("diff_vs_rknn3_orig") is not None:
        detail = f"diff(orig)={r.get('diff_vs_rknn3_orig'):.2g}"
    print(f"{r.get('model')}: {status} ({detail}) {r.get('seconds')}s", flush=True)
    if r.get("error"):
        print(f"  {r.get('error')}", flush=True)

    fields = ["model", "status", "diff_vs_rknn3_orig", "seconds", "error"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(r)
    print(f"wrote {args.output}", flush=True)

    if status == "skipped":
        msg = f"RKNN3-Toolkit unavailable; skipped ({r.get('error')})."
        if args.require_rknn3:
            print(f"{msg} (--require-rknn3 set -> failing)", flush=True)
            return 1
        print(f"{msg} Nothing to test; passing.", flush=True)
        return 0

    if status in FAIL_STATUSES:
        print("FAILED", flush=True)
        return 1
    print("passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
