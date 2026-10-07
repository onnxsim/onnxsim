#!/usr/bin/env python3
"""Mechanism check: where is the interval (IBP) box tighter than the CROWN box of a pre-activation?

onnxsim.crown.refine() tightens every Relu input box with CROWN and then intersects it with the interval
box (both are sound). auto_LiRPA's CROWN keeps the CROWN box only. This counts, layer by layer and for one
spec group, the neurons on which the interval box beats the CROWN-only box, and by how much. If the
'tighter than auto_LiRPA' rows come from this intersection, these counts are non-zero exactly there.

  python vnncomp_tightness_mech.py --bench acasxu --net ACASXU_run2a_3_1 --scale 1.0
"""

import argparse
import json
import os
from typing import List

import numpy as np
import onnx
from vnncomp_tightness_audit import _refine_variant, load

from onnxsim import crown, vnnlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="acasxu")
    ap.add_argument("--net", default="ACASXU_run2a_3_1")
    ap.add_argument("--scale", default="1.0")
    ap.add_argument("--group", type=int, default=0)
    a = ap.parse_args()
    specs, _, _ = load(a.bench, a.scale)
    rec = next(r for r in specs if a.net in r["onnx"] and r["group"] == a.group)
    net = onnx.load(rec["onnx"])
    name, shape = vnnlib._single_io(net)
    lo, hi, A = np.array(rec["lo"]), np.array(rec["hi"]), np.array(rec["A"])
    spec_model, _ = vnnlib._spec_model(net, A)
    rng = vnnlib._ranges(name, shape, lo, hi)
    # CROWN-only boxes (no intersection) vs interval boxes, taken from two analyzers
    an = crown._Analyzer(spec_model, rng)
    ibp = {k: (np.array(v[0]), np.array(v[1])) for k, v in an.ib.items()}
    saved = crown._Analyzer.refine
    crown._Analyzer.refine = _refine_variant(1 << 40, False)  # type: ignore[method-assign]
    try:
        an.refine()
    finally:
        crown._Analyzer.refine = saved  # type: ignore[method-assign]
    rows: List[dict] = []
    for idx, node in enumerate(an.nodes):
        if node.op_type != "Relu":
            continue
        x = node.input[0]
        il, iu = ibp[x]
        cl, cu = an.ib[x]
        beat_lo, beat_hi = (
            il > cl + 1e-9 * (1 + np.abs(cl)),
            iu < cu - 1e-9 * (1 + np.abs(cu)),
        )
        w_ibp, w_crown = iu - il, cu - cl
        rows.append(
            {
                "layer": len(rows) + 1,
                "neurons": int(il.size),
                "ibp_tighter_lower": int(beat_lo.sum()),
                "ibp_tighter_upper": int(beat_hi.sum()),
                "median_width_ibp": float(np.median(w_ibp)),
                "median_width_crown_only": float(np.median(w_crown)),
                "unstable_ibp": int(((il < 0) & (iu > 0)).sum()),
                "unstable_crown_only": int(((cl < 0) & (cu > 0)).sum()),
            }
        )
    print(
        f"{a.net} group {a.group} scale {a.scale}: interval box vs CROWN-only box of each Relu input"
    )
    print(
        f"{'layer':>5s} {'neurons':>7s} {'ibp<crown lo':>12s} {'ibp<crown hi':>12s} {'med width ibp':>13s} {'med width crown':>15s} {'unstable ibp':>12s} {'unstable crown':>14s}"
    )
    for r in rows:
        print(
            f"{r['layer']:5d} {r['neurons']:7d} {r['ibp_tighter_lower']:12d} {r['ibp_tighter_upper']:12d} {r['median_width_ibp']:13.4g} {r['median_width_crown_only']:15.4g} {r['unstable_ibp']:12d} {r['unstable_crown_only']:14d}"
        )
    # the cascade: the DEFAULT refinement (CROWN boxes intersected with interval boxes) vs CROWN-only, per layer
    an2 = crown._Analyzer(spec_model, rng)
    an2.refine()
    print(
        "\ncascade: default refinement (with interval intersection) vs CROWN-only refinement"
    )
    print(
        f"{'layer':>5s} {'unstable default':>16s} {'unstable crown-only':>19s} {'med width default':>17s} {'med width crown-only':>20s} {'mean width ratio':>16s}"
    )
    k = 0
    for idx, node in enumerate(an.nodes):
        if node.op_type != "Relu":
            continue
        x = node.input[0]
        dl, du = an2.ib[x]
        cl, cu = an.ib[x]
        k += 1
        ratio = float(np.mean((du - dl) / np.maximum(cu - cl, 1e-300)))
        print(
            f"{k:5d} {int(((dl < 0) & (du > 0)).sum()):16d} {int(((cl < 0) & (cu > 0)).sum()):19d} {float(np.median(du - dl)):17.4g} {float(np.median(cu - cl)):20.4g} {ratio:16.4f}"
        )
        rows[k - 1].update(
            unstable_default=int(((dl < 0) & (du > 0)).sum()),
            median_width_default=float(np.median(du - dl)),
            mean_width_ratio=ratio,
        )
    json.dump(
        rows,
        open(
            os.path.join(
                "/mnt/data/cache/claude-work/crown-audit",
                f"mech_{a.bench}_{a.net}_g{a.group}_s{a.scale}.json",
            ),
            "w",
        ),
        indent=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
