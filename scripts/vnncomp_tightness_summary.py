#!/usr/bin/env python3
"""Summarise the exact-reference JSONs written by vnncomp_tightness_exact.py with one rigorous definition.

* violation  = a value ATTAINED by a real input (PGD point through the actual network) lies outside the bound
               (min side: ``ours_lb > pgd_min``; max side: ``ours_ub < pgd_max``), beyond 1e-12 relative.
* certified  = the MILP dual bound (a bound on the true extreme from the outside) lies on the safe side of the bound,
               i.e. ``ours_lb <= dual_min`` / ``ours_ub >= dual_max`` (HiGHS tolerances apply: 1e-7 relative here).
* exact      = the MILP closed its gap (primal == dual to 1e-6 relative): the true range of the row is known.
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List

OUT = "/mnt/data/cache/claude-work/crown-audit"


def rel(x: float) -> float:
    return 1e-12 * (1 + abs(x))


def main(argv: List[str]) -> int:
    pats = argv or ["exact_*.json"]
    print(
        f"{'file':44s} rows  violations  min margin(lb side)  min margin(ub side)  MILP-exact rows  dual-certified (lb/ub)"
    )
    total_bad = 0
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(OUT, pat))):
            rows: List[Dict[str, Any]] = json.load(open(f))
            bad, m_lb, m_ub, exact, cl, cu, nl, nu = 0, [], [], 0, 0, 0, 0, 0
            for r in rows:
                if r.get("pgd_min") is None:
                    continue
                lb, ub = r["ours"]
                d_lb, d_ub = r["pgd_min"] - lb, ub - r["pgd_max"]
                m_lb.append(d_lb)
                m_ub.append(d_ub)
                bad += int(d_lb < -rel(lb)) + int(d_ub < -rel(ub))
                mm, mx = r.get("milp_min"), r.get("milp_max")
                if mm and mx and None not in (mm[0], mm[1], mx[0], mx[1]):
                    exact += int(
                        abs(mm[0] - mm[1]) <= 1e-6 * (1 + abs(mm[0]))
                        and abs(mx[0] - mx[1]) <= 1e-6 * (1 + abs(mx[0]))
                    )
                if mm and mm[1] is not None:
                    nl += 1
                    cl += int(lb <= mm[1] + 1e-7 * (1 + abs(mm[1])))
                if mx and mx[1] is not None:
                    nu += 1
                    cu += int(ub >= mx[1] - 1e-7 * (1 + abs(mx[1])))
            total_bad += bad
            print(
                f"{os.path.basename(f)[:44]:44s} {len(rows):4d}  {bad:10d}  {min(m_lb, default=float('nan')):19.2e}  {min(m_ub, default=float('nan')):19.2e}  {exact:15d}  {cl}/{nl} , {cu}/{nu}"
            )
    print(f"\nviolations by attained values over all files: {total_bad}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
