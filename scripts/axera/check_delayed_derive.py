#!/usr/bin/env python3
"""Offline: for each recalibratable chain, the scale the ``derived`` policy would
apply at step K (ranges derived from step K-1's measured inputs and output, widened
by MARGIN) against the exact scale step K needs. A ratio below 1 clips.

Usage: check_delayed_derive.py K MARGIN [SEGMENT_REGEX]"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matmul_record_emit as mre  # noqa: E402
import step_runner as sr  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402
import u16_chain  # noqa: E402

CACHE = os.environ.get("U16_CACHE", "/mnt/data/cache/claude-work/u16-seg-cache")


def main() -> None:
    k, margin = int(sys.argv[1]), float(sys.argv[2])
    pat = sys.argv[3] if len(sys.argv) > 3 else "."
    model = sr.load_step()
    segs, _ = sr.build_plan(model, sr.load_records(), axb.load_calibration(sr.STEP_CALIB), None, False, precision_overrides=None)
    segs = [s for s in segs if s.kind == "matmul_chain" and re.search(pat, s.name)]
    info: dict = {}
    sr.build_u16_segments(model, segs, sr.load_reference()["feeds"], pat, CACHE, need_quant=True, info_out=info)
    need = {t for i in info.values() for t in i["inputs"]}
    prev, _ = sr.StepRunner(model, []).run(sr.load_step_feeds(k - 1), "float", keep=sorted(need))
    cur, _ = sr.StepRunner(model, []).run(sr.load_step_feeds(k), "float", keep=sorted(need))
    bad = []
    for name, i in info.items():
        quant = i["path"] + ".quant.json"
        ts = mre.load_scales(quant)
        out_t = [o.name for o in i["sub"].graph.output][0]
        ex_prev = u16_chain.chain_ranges(i["sub"], sr._chain_samples(i, prev))
        derived = u16_chain.derive_ranges(i["sub"], quant, ts, {t: ex_prev[t] for t in (*i["inputs"], out_t)})
        applied = u16_chain.predict_scales16(quant, derived, margin)
        need_s = u16_chain.predict_scales16(quant, u16_chain.chain_ranges(i["sub"], sr._chain_samples(i, cur)))
        worst = min(((applied[t][0] / need_s[t][0], t) for t in need_s if t in applied and need_s[t][0] > 0), default=(9, ""))
        if worst[0] < 1.0:
            bad.append((worst[0], name, worst[1][-34:]))
    bad.sort()
    print(f"step {k}, margin {margin}: {len(bad)} of {len(info)} chains have a tensor whose applied scale is below the needed one")
    for r, n, t in bad[:14]:
        print(f"  {r:6.3f}  {n:28} {t}")


if __name__ == "__main__":
    main()
