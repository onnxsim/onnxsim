#!/usr/bin/env python3
"""Run onnxsim's bound engines on public VNN-COMP 2021 instances and compare with references.

NOT part of CI: it needs the benchmark files (downloaded separately, see docs/vnnlib-bench.md),
runs for a long time, and writes one JSON line per instance so an interrupted run resumes.

  run            verify every instance of a benchmark with one engine
  summary        soundness first, then coverage against the published results
  export-specs   write instances (network path, input box, spec rows) for the reference script
  tightness      our spec lower bounds vs auto_LiRPA's (scripts/vnncomp_ref_bounds.py), with a hard
                 soundness check of every bound against the smallest value actually observed

Published ground truth comes from the competition's own results repository
(VNN-COMP/vnncomp2021_results, results_csv/*.csv), consolidated by --ground-truth (a JSON file
{"<bench>|<onnx>|<vnnlib>": {"verdict": safe|unsafe|conflict|open, "holds_by": [...], "violated_by": [...]}}).
'safe' = some tool reported holds and none violated; 'unsafe' = the reverse; 'conflict' = both:
a published disagreement, never used as truth.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import onnx

from onnxsim import vnnlib

DEFAULT_ROOT = "/mnt/data/cache/claude-work/vnncomp/VNN-COMP__vnncomp2021/benchmarks"
DEFAULT_GT = "/mnt/data/cache/claude-work/vnncomp/ground_truth.json"


def instances(root: str, bench: str) -> List[Tuple[str, str, float]]:
    with open(os.path.join(root, bench, f"{bench}_instances.csv")) as f:
        return [(r[0], r[1], float(r[2])) for r in csv.reader(f) if len(r) >= 3]


def key_of(bench: str, onnx_rel: str, vnn_rel: str) -> str:
    return f"{bench}|{onnx_rel}|{vnn_rel}"


def done_keys(path: str, engine: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("engine") == engine:
                out.add(r["key"])
    return out


def cmd_run(a: argparse.Namespace) -> int:
    gt: Dict[str, Any] = (
        json.load(open(a.ground_truth)) if os.path.exists(a.ground_truth) else {}
    )
    rows = instances(a.root, a.bench)
    if a.stride > 1:
        rows = rows[:: a.stride]
    if a.limit:
        rows = rows[: a.limit]
    done = done_keys(a.out, a.engine)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_new = 0
    for onnx_rel, vnn_rel, _ in rows:
        key = key_of(a.bench, onnx_rel, vnn_rel)
        if key in done:
            continue
        t0 = time.time()
        onnx_path = os.path.join(a.root, a.bench, onnx_rel)
        vnn_path = os.path.join(a.root, a.bench, vnn_rel)
        if (
            a.engine == "attack"
        ):  # no bounds at all: the counterexample search on its own
            try:
                cex = vnnlib.find_counterexample(
                    onnx_path, vnn_path, seconds=a.attack_seconds, seed=a.seed
                )
                v = vnnlib.Verdict(
                    vnnlib.SAT if cex is not None else vnnlib.UNKNOWN,
                    "attack",
                    time.time() - t0,
                    "counterexample replayed on onnxruntime"
                    if cex is not None
                    else "none found",
                )
            except (vnnlib.VnnlibError, ValueError, NotImplementedError) as e:
                v = vnnlib.Verdict(
                    vnnlib.UNSUPPORTED,
                    "attack",
                    time.time() - t0,
                    f"{type(e).__name__}: {e}",
                )
        else:
            v = vnnlib.verify(
                onnx_path,
                vnn_path,
                engine=a.engine,
                timeout=a.timeout,
                budget=a.budget,
                attack=not a.no_attack,
                attack_seconds=a.attack_seconds,
            )
        rec = {
            "key": key,
            "bench": a.bench,
            "engine": a.engine,
            "status": v.status,
            "seconds": round(v.seconds, 3),
            "wall": round(time.time() - t0, 3),
            "detail": v.detail,
            "clauses": v.clauses,
            "regions": v.regions,
            "inconsistent": v.inconsistent,
            "gt": gt.get(key, {}).get("verdict", "none"),
            "timeout": a.timeout,
            "budget": a.budget,
        }
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_new += 1
        print(
            f"{a.bench} {a.engine:8s} {v.status:11s} {v.seconds:7.2f}s gt={rec['gt']:8s} {onnx_rel} {vnn_rel}",
            flush=True,
        )
    print(f"done: {n_new} new records in {a.out}")
    return 0


def load(paths: List[str]) -> List[Dict[str, Any]]:
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    return recs


def cmd_summary(a: argparse.Namespace) -> int:
    recs = load(a.results)
    if not recs:
        print("no records")
        return 1
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in recs:
        by[(r["bench"], r["engine"])].append(r)
    # instances with a counterexample that replays on onnxruntime, found by the separate
    # 'attack' pass: independent of every bound engine, so any engine 'unsat' there is a bug
    attacked = {
        r["key"] for r in recs if r["engine"] == "attack" and r["status"] == "sat"
    }
    print("== SOUNDNESS (first, because it is the point)")
    print(
        f"  replayed counterexamples from the independent attack pass: {len(attacked)} instances"
    )
    bad = 0
    for (bench, eng), rs in sorted(by.items()):
        if eng == "attack":
            continue
        alarms = [r for r in rs if r["inconsistent"]]
        hard = [r for r in rs if r["status"] == "unsat" and r["key"] in attacked]
        soft = [r for r in rs if r["status"] == "unsat" and r["gt"] == "unsafe"]
        cex_vs_safe = [r for r in rs if r["status"] == "sat" and r["gt"] == "safe"]
        bad += len(alarms) + len(hard)
        print(
            f"  {bench:8s} {eng:9s} n={len(rs):4d}  unsat-despite-replayed-counterexample={len(hard)}  "
            f"internal alarms={len(alarms)}  unsat-but-published-unsafe={len(soft)}  "
            f"replayed-sat-but-published-safe={len(cex_vs_safe)}"
        )
        for r in alarms + hard + soft + cex_vs_safe:
            print(
                f"      !! {r['key']}  status={r['status']} gt={r['gt']} inconsistent={r['inconsistent']}"
            )
    print(
        f"  total hard soundness violations (unsat despite a replayed counterexample, or an internal alarm): {bad}"
    )
    print("\n== VERDICTS vs the published verdict")
    hdr = [
        "benchmark",
        "engine",
        "n",
        "unsat",
        "sat",
        "unknown",
        "unsupp",
        "pub:safe",
        "proved",
        "pub:unsafe",
        "found",
        "avg s",
        "max s",
    ]
    print("  " + " ".join(f"{h:>10s}" for h in hdr))
    for (bench, eng), rs in sorted(by.items()):
        c = collections.Counter(r["status"] for r in rs)
        safe = [r for r in rs if r["gt"] == "safe"]
        unsafe = [r for r in rs if r["gt"] == "unsafe"]
        proved = sum(1 for r in safe if r["status"] == "unsat")
        found = sum(1 for r in unsafe if r["status"] == "sat")
        secs = [r["seconds"] for r in rs]
        row = [
            bench,
            eng,
            len(rs),
            c["unsat"],
            c["sat"],
            c["unknown"],
            c["unsupported"],
            len(safe),
            proved,
            len(unsafe),
            found,
            f"{sum(secs) / len(secs):.2f}",
            f"{max(secs):.1f}",
        ]
        print("  " + " ".join(f"{str(x):>10s}" for x in row))
    return 0


# ---- export for the reference + tightness ------------------------------------------------------


def _box_key(lo: np.ndarray, hi: np.ndarray) -> str:
    return hashlib.sha1(lo.tobytes() + hi.tobytes()).hexdigest()[:12]


def cmd_export(a: argparse.Namespace) -> int:
    rows = instances(a.root, a.bench)
    if a.only:  # explicit instances: "<onnx>:<vnnlib>" pairs, comma separated
        want = {tuple(p.split(":")) for p in a.only.split(",")}
        rows = [r for r in rows if (r[0], r[1]) in want]
    if a.stride > 1:
        rows = rows[:: a.stride]
    if a.limit:
        rows = rows[: a.limit]
    out: List[Dict[str, Any]] = []
    for onnx_rel, vnn_rel, _ in rows:
        prop = vnnlib.parse_file(os.path.join(a.root, a.bench, vnn_rel))
        groups: Dict[str, Dict[str, Any]] = {}
        for ci, c in enumerate(prop.clauses):
            if not c.atoms or not (
                np.all(np.isfinite(c.lo)) and np.all(np.isfinite(c.hi))
            ):
                continue
            ctr, half = (
                (c.lo + c.hi) / 2.0,
                (c.hi - c.lo) / 2.0 * a.shrink,
            )  # --shrink < 1: near-tight regime
            lo_s, hi_s = ctr - half, ctr + half
            g = groups.setdefault(
                _box_key(lo_s, hi_s),
                {
                    "lo": lo_s.tolist(),
                    "hi": hi_s.tolist(),
                    "A": [],
                    "b": [],
                    "clause": [],
                },
            )
            for at in c.atoms:
                g["A"].append(at.a.tolist())
                g["b"].append(at.b)
                g["clause"].append(ci)
        for gi, g in enumerate(groups.values()):
            out.append(
                {
                    "key": key_of(a.bench, onnx_rel, vnn_rel),
                    "group": gi,
                    "onnx": os.path.join(a.root, a.bench, onnx_rel),
                    **g,
                }
            )
    json.dump(out, open(a.out, "w"))
    print("exported", len(out), "groups from", len(rows), "instances ->", a.out)
    return 0


def observed_spec_min(
    net: onnx.ModelProto, lo: np.ndarray, hi: np.ndarray, A: np.ndarray, seed: int = 0
) -> np.ndarray:
    """The smallest value of each spec row ``a_j . net(x)`` actually reached on onnxruntime.

    Samples, corners and a per-row PGD (torch) -- every reported value is an onnxruntime output of
    a point inside the box, so it is an UPPER bound on the true minimum and a sound bound must
    never exceed it.
    """
    name, shape = vnnlib._single_io(net)
    rng = np.random.default_rng(seed)
    sess = vnnlib._ort_session(net)
    best = np.full(A.shape[0], np.inf)

    def ev(x: np.ndarray) -> np.ndarray:
        xin = vnnlib._inside_f32(x.reshape(-1), lo, hi)
        y = sess.run(None, {name: xin.reshape(shape)})[0].reshape(-1).astype(np.float64)
        return A @ y

    pts = [(lo + hi) / 2, lo.copy(), hi.copy()]
    pts += [lo + (hi - lo) * rng.integers(0, 2, lo.shape) for _ in range(100)]
    pts += [lo + (hi - lo) * rng.random(lo.shape) for _ in range(1000)]
    for x in pts:
        best = np.minimum(best, ev(x))
    try:
        import torch

        tn = vnnlib._TorchNet(net)
        xs = tuple(shape[1:])
        lo_t = torch.tensor(lo.reshape(xs), dtype=torch.float32)
        hi_t = torch.tensor(hi.reshape(xs), dtype=torch.float32)
        width = hi_t - lo_t
        at = torch.tensor(A, dtype=torch.float32)
        for j in range(A.shape[0]):
            n = 24
            x = lo_t + width * torch.rand((n,) + xs)
            bx, bf = x.clone(), torch.full((n,), float("inf"))
            for step in range(120):
                x = x.detach().requires_grad_(True)
                f = tn(x).reshape(n, -1) @ at[j]
                f.sum().backward()
                with torch.no_grad():
                    better = f.detach() < bf
                    bf = torch.where(better, f.detach(), bf)
                    bx[better] = x.detach()[better]
                    lr = width * (0.05 * 0.95**step + 5e-5)
                    x = torch.minimum(
                        torch.maximum(x.detach() - lr * x.grad.sign(), lo_t), hi_t
                    )
            for i in torch.argsort(bf)[:3].tolist():
                best = np.minimum(best, ev(bx[i].numpy().astype(np.float64)))
    except Exception:  # noqa: BLE001 - torch missing / op unsupported: sampling only
        pass
    return best


def cmd_tightness(a: argparse.Namespace) -> int:
    specs = json.load(open(a.specs))
    ref = {(r["key"], r["group"]): r for r in json.load(open(a.ref))}
    engines = a.engines.split(",")
    rows: List[Dict[str, Any]] = []
    unsound = 0
    margins: Dict[str, List[float]] = {}
    for rec in specs:
        r = ref.get((rec["key"], rec["group"]))
        if r is None:
            continue
        net = onnx.load(rec["onnx"])
        name, shape = vnnlib._single_io(net)
        lo, hi, A = np.array(rec["lo"]), np.array(rec["hi"]), np.array(rec["A"])
        spec_model, spec = vnnlib._spec_model(net, A)
        rng_in = vnnlib._ranges(name, shape, lo, hi)
        obs = observed_spec_min(net, lo, hi, A)
        ours: Dict[str, Any] = {}
        for eng in engines:
            t0 = time.time()
            try:
                lb = vnnlib._bounds(spec_model, rng_in, spec, eng)[0].reshape(-1)
                ours[eng] = {"lb": lb, "seconds": time.time() - t0}
            except Exception as e:  # noqa: BLE001
                ours[eng] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        entry = {
            "key": rec["key"],
            "group": rec["group"],
            "rows": A.shape[0],
            "obs": obs,
            "ours": ours,
            "ref": r,
        }
        rows.append(entry)
        # float32 evaluation vs float64 bounds: onnxruntime outputs carry ~1e-7 relative rounding, so
        # a bound may exceed the smallest observed value by that much without being unsound
        tol = a.tol
        for who, lbv in [
            (f"ours:{e}", o["lb"]) for e, o in ours.items() if "lb" in o
        ] + [
            (f"ref:{m}", np.array(r[m]["lb"]))
            for m in ("CROWN", "alpha-CROWN")
            if m in r and "lb" in r[m]
        ]:
            viol = lbv - obs
            margins.setdefault(who, []).extend((obs - lbv).tolist())
            if np.any(viol > tol * (1 + np.abs(obs))):
                unsound += 1
                print(
                    f"!! UNSOUND? {rec['key']} g{rec['group']} {who}: max(lb - observed_min) = {viol.max():.3e}"
                )
    print(
        f"\n== soundness: bounds above an observed value by more than {a.tol:g} relative (must be 0): {unsound}"
    )
    print(
        "   margin = observed_min - lb per spec row (>= 0 for a sound bound; tiny = a check with teeth;"
    )
    print("   slightly negative = float32 rounding of onnxruntime's outputs):")
    for who, m in sorted(margins.items()):
        mm = np.array(m)
        print(
            f"     {who:18s} rows={len(mm):4d}  min {mm.min():+.2e}  median {np.median(mm):.2e}  max {mm.max():.2e}"
        )
    print(f"== tightness on {len(rows)} groups; per group the median over spec rows of")
    print(
        "   share of the IBP->reference gap recovered:  (lb_ours - lb_ibp) / (lb_ref - lb_ibp)  (1.0 = as tight as the reference)"
    )
    for ref_name in ("CROWN", "alpha-CROWN"):
        for eng in engines:
            if eng == "ibp":
                continue
            shares, tighter, looser, same = [], 0, 0, 0
            for e in rows:
                if (
                    "lb" not in e["ours"].get(eng, {})
                    or "lb" not in e["ours"].get("ibp", {})
                    or ref_name not in e["ref"]
                    or "lb" not in e["ref"][ref_name]
                ):
                    continue
                lb, ibp, rf = (
                    e["ours"][eng]["lb"],
                    e["ours"]["ibp"]["lb"],
                    np.array(e["ref"][ref_name]["lb"]),
                )
                gap = rf - ibp
                ok = gap > 1e-9
                if ok.any():
                    shares.append(float(np.median((lb[ok] - ibp[ok]) / gap[ok])))
                d = lb - rf
                tighter += int(np.sum(d > 1e-6 * (1 + np.abs(rf))))
                looser += int(np.sum(d < -1e-6 * (1 + np.abs(rf))))
                same += int(np.sum(np.abs(d) <= 1e-6 * (1 + np.abs(rf))))
            if shares:
                print(
                    f"   vs {ref_name:11s} ours={eng:9s} median share {np.median(shares):6.3f}  (rows: tighter {tighter}, same {same}, looser {looser})"
                )
    json.dump(
        [
            {
                "key": e["key"],
                "group": e["group"],
                "obs": e["obs"].tolist(),
                "ours": {
                    k: (
                        {"lb": v["lb"].tolist(), "seconds": v["seconds"]}
                        if "lb" in v
                        else v
                    )
                    for k, v in e["ours"].items()
                },
                "ref": {m: e["ref"].get(m) for m in ("CROWN", "alpha-CROWN")},
            }
            for e in rows
        ],
        open(a.out, "w"),
    )
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    benches = ["acasxu", "mnistfc", "eran", "oval21", "cifar2020"]
    r = sub.add_parser("run")
    r.add_argument("--bench", required=True, choices=benches)
    r.add_argument("--engine", required=True, choices=list(vnnlib.ENGINES) + ["attack"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--root", default=DEFAULT_ROOT)
    r.add_argument("--ground-truth", default=DEFAULT_GT)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=30.0)
    r.add_argument("--budget", type=int, default=100)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--no-attack", action="store_true")
    r.add_argument(
        "--attack-seconds",
        type=float,
        default=5.0,
        help="total counterexample-search budget per instance",
    )
    s = sub.add_parser("summary")
    s.add_argument("results", nargs="+")
    e = sub.add_parser("export-specs")
    e.add_argument("--bench", required=True, choices=benches)
    e.add_argument("--root", default=DEFAULT_ROOT)
    e.add_argument("--out", required=True)
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument(
        "--only",
        default="",
        help="comma-separated '<onnx>:<vnnlib>' instances to export",
    )
    e.add_argument(
        "--shrink",
        type=float,
        default=1.0,
        help="scale each input box around its centre (a sharper soundness check: bounds on a small box are close to the truth)",
    )
    t = sub.add_parser("tightness")
    t.add_argument("specs")
    t.add_argument("ref")
    t.add_argument("--engines", default="ibp,zonotope,crown,alpha")
    t.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="relative slack for float32 evaluation (default 1e-6)",
    )
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {
        "run": cmd_run,
        "summary": cmd_summary,
        "export-specs": cmd_export,
        "tightness": cmd_tightness,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
