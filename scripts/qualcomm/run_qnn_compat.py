#!/usr/bin/env python3
"""Run the QNN compatibility check over the model suite and report.

Drives ``worker.py`` once per model (isolated subprocess, hard timeout), collects
the JSON results, writes a CSV, prints a summary, and exits non-zero if any model
failed. This is the entry point the ``Qualcomm Integration`` workflow calls.

A model **fails** the check when simplification breaks QNN compatibility or
changes the on-device result (``qnn_regression``), when onnxsim raises
(``simplify_error``), or when the worker crashes/times out. A graph the QNN
backend cannot handle even *before* simplification is ``unsupported`` and is
**reported, not failed** — that is a backend limitation, not an onnxsim bug.
Partial QNN coverage (some nodes falling back to ORT's CPU provider) is likewise
reported, not failed — plenty of valid graphs are not 100% HTP-mappable.

If the QNN EP is unavailable on the host, every model reports ``skipped`` and the
run passes (nothing to test) unless ``--require-qnn`` is given.

Usage:
    run_qnn_compat.py --output qnn.csv
    run_qnn_compat.py --require-qnn          # fail if the QNN EP is missing
    run_qnn_compat.py --models conv_bn_relu matmul_add_gelu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAIL_STATUSES = {"qnn_regression", "simplify_error", "crash", "timeout", "error"}


def run_one(model_name: str, timeout: int, onnx_path: str | None = None) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "worker.py"),
                model_name,
                *([onnx_path] if onnx_path else []),
            ],
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
        "--model",
        action="append",
        default=[],
        metavar="MODEL.onnx",
        help="also run an on-disk ONNX model (repeatable)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-model wall-clock cap in seconds; <=0 disables",
    )
    ap.add_argument(
        "--require-qnn",
        action="store_true",
        help="fail if the QNN EP is unavailable instead of skipping",
    )
    ap.add_argument("--output", default="qnn-compat.csv")
    args = ap.parse_args()

    selected = args.models or []
    if not args.model and not selected:
        import models

        selected = models.names()
    external = [(Path(path).stem, path) for path in args.model]
    missing = [path for _, path in external if not os.path.isfile(path)]
    if missing:
        ap.error("ONNX model does not exist: " + ", ".join(missing))
    total = len(selected) + len(external)
    print(f"QNN compatibility check | {total} models", flush=True)

    rows = []
    failures = []
    skipped = 0
    cases = [(name, None) for name in selected] + external
    for i, (name, onnx_path) in enumerate(cases, 1):
        print(f"[{i}/{total}] {name} ...", end=" ", flush=True)
        r = run_one(name, args.timeout, onnx_path)
        rows.append(r)
        status = r.get("status")
        if status == "skipped":
            skipped += 1
            print(f"skipped ({r.get('error')})", flush=True)
            continue
        detail = ""
        if r.get("orig_nodes") is not None:
            detail = f"{r.get('orig_nodes')}->{r.get('simp_nodes')} nodes"
        if r.get("coverage_orig") or r.get("coverage_simp"):
            detail += f", qnn={r.get('coverage_orig')}->{r.get('coverage_simp')}"
        if r.get("diff_vs_qnn_orig") is not None:
            detail += f", d(orig)={r.get('diff_vs_qnn_orig'):.2g}"
        print(f"{status} ({detail}) {r.get('seconds')}s", flush=True)
        if status in FAIL_STATUSES:
            failures.append((name, status, str(r.get("error"))[:200]))

    fields = [
        "model",
        "status",
        "orig_nodes",
        "simp_nodes",
        "coverage_orig",
        "coverage_simp",
        "diff_vs_qnn_orig",
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

    if skipped == total:
        msg = "QNN EP unavailable on this host; all models skipped."
        if args.require_qnn:
            print(f"\n{msg} (--require-qnn set -> failing)", flush=True)
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
