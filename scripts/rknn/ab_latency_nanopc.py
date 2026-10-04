#!/usr/bin/env python3
"""Interleaved A/B latency comparison: original ONNX vs onnxsim-simplified.

Why this exists
---------------
Measuring "original then simplified" once each does not work on this board.
Re-running the same pair moves results by >100% and flips the sign of the
difference -- the ~0.08 ms measurements these graphs produce are dominated by
RKNN launch overhead and buried in the noise of whatever else the 8-core
Cortex-A55 box is doing. Confirmed by two identical back-to-back runs of
``run_rknn3588_compat.py``: ``conv_bn_relu`` measured -43.9% then +116.7%,
``redundant_transpose`` -39.5% then +6.8%, ``sigmoid_mul_swish`` +5.3% then
-10.7%.

So this driver:

* compiles each variant **once** and reuses the same ``.rknn`` for every
  sample, removing build-to-build variance from the comparison entirely,
* **interleaves** the variants (A B A B ...) so thermal drift, frequency
  changes and background load hit both equally instead of biasing whichever
  ran second,
* reports **min** as the headline, with the full distribution, because min is
  the statistic least contaminated by preemption on a non-realtime Linux
  kernel,
* reports a **paired** statistic (per-round difference, median and IQR) so a
  claimed win can be judged against its own spread rather than against a
  single number.

Usage:
    python scripts/rknn/ab_latency_nanopc.py --rounds 15
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_HERE)
for path in (_SCRIPTS_DIR, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.ep_numerics import random_feeds  # noqa: E402
from common.synthetic_models import build, names  # noqa: E402

from onnxsim import simplify  # noqa: E402


def _board_latency(host: str, user: str, password: str | None, rknn_path: str,
                   feeds_path: str, remote_dir: str, warmup: int,
                   iterations: int) -> dict:
    """One board run; returns the runner's latency summary block (ms)."""
    import run_rknn3588_compat as harness

    board = harness.run_on_board(host, user, password, rknn_path, feeds_path,
                                 remote_dir, warmup, iterations)
    return board["latency_ms"]


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "min": round(ordered[0], 4),
        "p50": round(statistics.median(ordered), 4),
        "mean": round(statistics.mean(ordered), 4),
        "p95": round(ordered[min(len(ordered) - 1,
                                 int(round((len(ordered) - 1) * 0.95)))], 4),
    }


def compare_model(name: str, args: argparse.Namespace, work_dir: str) -> dict:
    import run_rknn3588_compat as harness

    model = build(name)
    raw = os.path.join(work_dir, f"{name}.onnx")
    simp_path = os.path.join(work_dir, f"{name}.simplified.onnx")
    import onnx

    onnx.save(model, raw)
    simplified, ok = simplify(model, check_n=0)
    if not ok:
        return {"model": name, "status": "simplify_error"}
    onnx.save(simplified, simp_path)

    feeds = random_feeds(model, seed=0)
    feeds_path = os.path.join(work_dir, f"{name}.feeds.npz")
    np.savez(feeds_path, **feeds)
    calibration = [feeds[i.name] for i in model.graph.input if i.name in feeds]

    variants = {}
    for label, path in (("orig", raw), ("simp", simp_path)):
        rknn_path = os.path.join(work_dir, f"{name}.{label}.rknn")
        # Compile once; every sample below reuses this exact artifact.
        harness.compile_rknn(path, rknn_path, calibration, model.graph.input[0].name)
        variants[label] = rknn_path

    # Interleaved A/B sampling.
    rounds = {"orig": [], "simp": []}
    for round_index in range(args.rounds):
        order = ("orig", "simp") if round_index % 2 == 0 else ("simp", "orig")
        for label in order:
            rounds[label].append(
                _board_latency(args.host, args.user, args.ssh_password,
                               variants[label], feeds_path,
                               f"{args.remote_dir}/ab-{name}", args.warmup,
                               args.iterations)["min"]
            )

    orig_min = rounds["orig"]
    simp_min = rounds["simp"]
    paired = [o - s for o, s in zip(orig_min, simp_min)]

    orig_stats = _stats(orig_min)
    simp_stats = _stats(simp_min)
    delta_pct = (simp_stats["min"] - orig_stats["min"]) / orig_stats["min"] * 100

    # A claim is only credible if the paired differences do not straddle zero.
    paired_iqr = (statistics.quantiles(paired, n=4)[2]
                  - statistics.quantiles(paired, n=4)[0]) if len(paired) > 3 else 0.0
    verdict = ("no-change" if min(paired) <= 0 <= max(paired)
               else ("simplified-faster" if statistics.median(paired) > 0
                     else "simplified-slower"))

    return {
        "model": name,
        "status": "ok",
        "orig_nodes": len(model.graph.node),
        "simp_nodes": len(simplified.graph.node),
        "rounds": args.rounds,
        "orig": orig_stats,
        "simp": simp_stats,
        "delta_min_pct": round(delta_pct, 2),
        "paired_median_ms": round(statistics.median(paired), 5),
        "paired_iqr_ms": round(paired_iqr, 5),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--host", default="nanopc-t6.tailf0b7b1.ts.net")
    ap.add_argument("--user", default="pi")
    ap.add_argument("--ssh-password", default=os.environ.get("RKNN_SSH_PASSWORD"))
    ap.add_argument("--remote-dir", default="/tmp/onnxsim-rknn-ab")
    ap.add_argument("--work-dir", default="/tmp/onnxsim-rknn-ab-work")
    ap.add_argument("--rounds", type=int, default=15,
                    help="interleaved A/B rounds (>=4 recommended)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    rows = []
    for i, name in enumerate(args.models or names(), 1):
        print(f"[{i}] {name} ...", flush=True)
        row = compare_model(name, args, args.work_dir)
        rows.append(row)
        if row["status"] == "ok":
            print(f"    {row['orig_nodes']}->{row['simp_nodes']} nodes  "
                  f"orig min={row['orig']['min']:.4f} simp min={row['simp']['min']:.4f} "
                  f"({row['delta_min_pct']:+.1f}%)  paired median="
                  f"{row['paired_median_ms']:+.5f} IQR={row['paired_iqr_ms']:.5f} "
                  f"-> {row['verdict']}", flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())