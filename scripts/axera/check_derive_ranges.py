#!/usr/bin/env python3
"""Offline check of ``u16_chain.derive_ranges``: with only a chain's inputs and
output measured (what the device can measure), how close are the scales derived for
the other tensors to the exact ones from a float run of a later step?

Usage: check_derive_ranges.py STEP SEGMENT... (templates built on the reference
batch, from the 16-bit build cache of a ``--u16-recal`` run)."""

from __future__ import annotations

import os
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
    step, names = int(sys.argv[1]), sys.argv[2:]
    model = sr.load_step()
    segs, _ = sr.build_plan(model, sr.load_records(), axb.load_calibration(sr.STEP_CALIB), None, False, precision_overrides=None)
    segs = [s for s in segs if s.name in names]
    info: dict = {}
    sr.build_u16_segments(
        model, segs, sr.load_reference()["feeds"], "|".join(f"^{n}$" for n in names), CACHE,
        need_quant=True, info_out=info,
    )
    need = {t for i in info.values() for t in i["inputs"]}
    outs, _ = sr.StepRunner(model, []).run(sr.load_step_feeds(step), "float", keep=sorted(need))
    print(f"{'chain':32} {'tensors':>7} {'measured':>8}   derived/exact scale: median min max (<1 would clip)")
    for name, i in info.items():
        quant = i["path"] + ".quant.json"
        tscales = mre.load_scales(quant)
        exact = u16_chain.chain_ranges(i["sub"], sr._chain_samples(i, outs))
        out_t = [o.name for o in i["sub"].graph.output][0]
        measured = {t: exact[t] for t in (*i["inputs"], out_t)}
        derived = u16_chain.derive_ranges(i["sub"], quant, tscales, measured)
        se = u16_chain.predict_scales16(quant, exact)
        sd = u16_chain.predict_scales16(quant, derived)
        ratios = [sd[t][0] / se[t][0] for t in se if t in sd and t not in measured and se[t][0] > 0]
        st = [tscales[t][0] / se[t][0] for t in se if t in tscales and t not in measured and se[t][0] > 0]
        worst = min(
            (sd[t][0] / se[t][0], t.split("_", 2)[-1][:28]) for t in se if t in sd and t not in measured and se[t][0] > 0
        )
        print(f"{name:32} {len(se):7d} {len(measured):8d}   {np.median(ratios):.3f} {min(ratios):.3f} {max(ratios):.3f}   | static: {np.min(st):.3f} {np.max(st):.3f} | smallest derived/exact {worst[0]:.3f} at {worst[1]}")


if __name__ == "__main__":
    main()
