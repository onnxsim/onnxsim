#!/usr/bin/env python3
"""Per-network ablation: tighter / same / looser than auto_LiRPA CROWN (float64) for chosen networks.

python vnncomp_tightness_pernet.py acasxu:ACASXU_run2a_5_4 mnistfc:mnist-net_256x4 ...
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from vnncomp_tightness_audit import classify, load, ours

VARIANTS = tuple(
    os.environ.get(
        "AUDIT_VARIANTS", "default,no_intersect,no_refine,alpha0_one,alpha0_zero"
    ).split(",")
)


def main(argv: List[str]) -> int:
    out: Dict[str, Dict[str, Tuple[int, int, int]]] = {}
    print(
        f"{'network':34s} "
        + " ".join(f"{v:>16s}" for v in VARIANTS)
        + "   (tighter/same/looser vs auto_LiRPA CROWN f64, tol 1e-9)"
    )
    for spec in argv:
        bench, net = spec.split(":")
        specs, _, r64 = load(bench, "1.0")
        line = []
        out[spec] = {}
        for v in VARIANTS:
            ds, rs = [], []
            for rec in specs:
                if net not in rec["onnx"]:
                    continue
                k = (rec["key"], rec["group"])
                lo, _, _ = ours(rec, v)
                rl = np.array(r64[k]["CROWN"]["lb"])
                ds.append(lo - rl)
                rs.append(rl)
            c = classify(np.concatenate(ds), np.concatenate(rs), 1e-9)
            out[spec][v] = c
            line.append(f"{c[0]:4d}/{c[1]:4d}/{c[2]:4d}")
        print(f"{spec[:34]:34s} " + " ".join(f"{x:>16s}" for x in line))
    json.dump(
        out, open("/mnt/data/cache/claude-work/crown-audit/pernet.json", "w"), indent=1
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
