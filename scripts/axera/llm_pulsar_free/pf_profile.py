"""Per-program card time of the ``rest`` set (``--set first``: of the
``first`` set): after a few real steps (so every binding is the one a step
uses), each program runs ``--n`` times back to back in one ``PLAN``; the
runner's exec time / n is that program's time.

usage: MODEL=qwen3_0.6b pf_profile.py [--n 10] [--set rest|first]
       -> <out>/prof_<set>.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import axcl_session as A  # noqa: E402
import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
import pf_plan  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--set", default="rest", choices=("rest", "first"))
    a = ap.parse_args()
    cfg = pf_model.load()
    rows = []
    with A.AXSession() as sess:
        print("health before", A.health_check(sess), flush=True)
        pl = pf_plan.Plan(sess, cfg)
        ids = cfg.passage_ids(lay.EVAL_DOCS[0])[:8]
        for p in range(cfg.first_pos, cfg.first_pos + 5):
            _, tm = pl.step(ids[p], p)
        print("step exec / plan us", tm, flush=True)
        for (st, wh, k), m in pl.M.items():
            if st != a.set:
                continue
            pl.lines = [f"X {m.id}"] * a.n
            ex = pl.run("prof")[0]
            nbytes = sum(s.nbytes for s in m.inputs) + sum(s.nbytes for s in m.outputs)
            rows.append(dict(wh=wh, k=k, us=ex / a.n, io_bytes=nbytes))
        print("health after", A.health_check(sess), flush=True)
    json.dump(rows, open(os.path.join(cfg.out, f"prof_{a.set}.json"), "w"), indent=1)
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["k"]].append(r["us"])
    tot = sum(r["us"] for r in rows)
    print("sum of program us", tot)
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(
            f"{k:12s} n={len(v):3d} mean {np.mean(v):8.1f} us  "
            f"total {sum(v) / 1e3:7.2f} ms  {100 * sum(v) / tot:5.1f}%"
        )


if __name__ == "__main__":
    main()
