#!/usr/bin/env python3
"""Our crown variants against auto_LiRPA CROWN run with its own options toggled (float64).

auto_LiRPA defaults: ``compare_crown_with_ibp=False`` (no intersection of CROWN intermediate bounds with
interval bounds) and ``sparse_intermediate_bounds=True`` (interval bounds are kept for neurons the interval
pass finds stable). The audit's hypothesis: onnxsim.crown differs by intersecting EVERY refined box with the
interval box, so auto_LiRPA with ``compare_crown_with_ibp=True`` should match onnxsim's default.

  python vnncomp_tightness_refvariants.py
"""

import json
import os
from typing import Dict, List

import numpy as np
from vnncomp_tightness_audit import BENCHES, classify, ours

OUT = "/mnt/data/cache/claude-work/crown-audit"
REFS = ("CROWN", "CROWN+ibpcmp", "CROWN+dense", "CROWN+dense+ibpcmp")
OURS = ("default", "no_intersect", "cascade")


def main() -> int:
    table: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    print(
        "rows: tighter/same/looser (tol 1e-9) of OUR variant against each auto_LiRPA float64 variant"
    )
    print(f"{'bench':8s} {'ours':13s} " + " ".join(f"{r:>21s}" for r in REFS))
    for b in BENCHES:
        specs = json.load(
            open(f"/mnt/data/cache/claude-work/vnncomp/results/specs_{b}_s1.0.json")
        )
        refs = {
            (r["key"], r["group"]): r
            for r in json.load(open(os.path.join(OUT, f"ref64v_{b}_s1.0.json")))
        }
        for v in OURS:
            diffs: Dict[str, List[np.ndarray]] = {r: [] for r in REFS}
            vals: Dict[str, List[np.ndarray]] = {r: [] for r in REFS}
            for rec in specs:
                k = (rec["key"], rec["group"])
                lo, _, _ = ours(rec, v)
                for r in REFS:
                    if "lb" in refs[k].get(r, {}):
                        rl = np.array(refs[k][r]["lb"])
                        diffs[r].append(lo - rl)
                        vals[r].append(rl)
            cells = []
            for r in REFS:
                c = classify(np.concatenate(diffs[r]), np.concatenate(vals[r]), 1e-9)
                table.setdefault(b, {}).setdefault(v, {})[r] = list(c)
                cells.append(f"{c[0]:5d}/{c[1]:5d}/{c[2]:5d}")
            print(f"{b:8s} {v:13s} " + " ".join(f"{x:>21s}" for x in cells))
    json.dump(table, open(os.path.join(OUT, "refvariants.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
