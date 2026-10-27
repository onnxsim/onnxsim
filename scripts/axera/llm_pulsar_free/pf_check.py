"""Check every program of a set on the card against its host model: after a
few real decode steps (``pf_plan.Plan``), each program's input and output
buffers (as the last step bound them) are read back, the program's host
evaluation (``pf_fake_quant``: the program's float graph at its calibration,
or the s8 linear) runs on the card's own inputs, and the card's outputs are
compared with it. A program is retargeted or emitted from templates, not
compiled for its calibration, so this is the check that it computes what its
float graph and calibration say, on the card, for every program of the set.

The relative error is the RMS of (card - host) over the RMS of host. The
host model evaluates each U16 Add as the card does (``pf_fake_quant.card_add``:
fixed-point ratios and Pulsar2's offset word, which leave a bias of up to ~2
output codes on every element; ``exact_offset`` for a set patched by
``pf_add_offset.py``). U16 rounding of the host's other inner tensors differs
from the card's by about one step, so a correct program reads ~1e-5..1.1e-2
on both sets (the largest: the last layer's ``dn`` and the attention); a
wrong record (scale, zero point, Add shift, weight table) reads 1e-1 or more.
Without the card's Add (a float sum), llama-160m's layer-1 ``on2`` read 1.3e-2
and Qwen3's first set up to 3.3e-2: that was the Add's bias, not the program.

--set nl (llama-160m): the steps of the first passage with its line breaks,
up to its first newline (which runs on the nl set).

usage: MODEL=qwen3_0.6b pf_check.py [--steps 3] [--tol T] [--set rest]
       -> <out>/check_<set>.json; exit status 1 if any program is above --tol
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
import pf_plan  # noqa: E402
from pf_fake_quant import AddEval, FusedEval, LinearEval  # noqa: E402

F32 = np.float32


def host_fn(cfg, root, wh, k, pk, e):
    """The host model of loaded program ``pk`` (``k``'s part for a split linear)."""
    li = None if wh == "post" else int(wh[1:])
    if e["kind"] == "fused":
        ev = FusedEval(
            os.path.join(root, e["model"]),
            e["calibration"],
            e["signed"],
            card_add=True,
            exact_offset=e.get("add_offset") == "exact",
        )
        return lambda f: ev(**f)
    if e["kind"] == "add":
        ad = AddEval(
            e["calibration"], True, exact_offset=e.get("add_offset") == "exact"
        )
        return lambda f: ad(f["a"], f["b"])
    if e["kind"] == "split":
        sp = e["split"][pk.rsplit("_", 1)[1]]
        if pk.endswith("_add"):
            ad = AddEval(
                sp["calibration"], True, exact_offset=e.get("add_offset") == "exact"
            )
            return lambda f: ad(f["a"], f["b"])
        w = cfg.merged_weight(k, li, gptq=False)
        q, r = cfg.split_weights(w)
        lin = LinearEval(
            w if pk.endswith("_hi") else r, *map(sp["calibration"].get, "xy")
        )
        return lambda f: lin(f["x"])
    c = e["calibration"]
    lin = LinearEval(cfg.merged_weight(k, li), c["x"], c["y"])
    return lambda f: lin(f["x"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-2, help="default 2e-2 (both sets)")
    ap.add_argument("--set", default="rest", choices=("first", "rest", "nl"))
    a = ap.parse_args()
    import axcl_session as A

    cfg = pf_model.load()
    ids = cfg.passage_ids(lay.EVAL_DOCS[0], lines=a.set == "nl")
    p0 = cfg.first_pos if a.set == "first" else cfg.first_pos + a.steps
    if a.set == "nl":  # up to the passage's first newline at or after p0
        p0 = next(p for p in range(p0, len(ids)) if ids[p] in cfg.nl_tokens)
    rows, bad = [], 0
    with A.AXSession() as sess:
        print("health before", A.health_check(sess), flush=True)
        pl = pf_plan.Plan(sess, cfg)
        for p in range(cfg.first_pos, p0 + 1):
            pl.step(ids[p], p)
        reads = []

        def rd(b, off, n):
            f = f"r{len(reads)}.bin"
            reads.append(f)
            pl.lines.append(f"R {b} {off} {n} {pl.g}/t/{f}")
            return f

        todo = []
        for (st, wh, pk), m in pl.M.items():
            if st != a.set:
                continue
            k = next(
                kk
                for kk in cfg.programs(st, wh)
                if pk in pf_plan._parts(pl.man[st][wh][kk], kk)
            )
            e = pl.man[st][wh][k]
            src = e.get("shared", st)
            e = pl.man[src][wh][k]
            bnd = pl.binds[m.id]
            ins = {s.name: rd(*bnd[("I", i)], s.nbytes) for i, s in enumerate(m.inputs)}
            outs = {
                s.name: rd(*bnd[("O", i)], s.nbytes) for i, s in enumerate(m.outputs)
            }
            todo.append((wh, k, pk, e, m, ins, outs))
        pl.run("check_reads")

        def load(f, spec):
            return np.fromfile(os.path.join(pl.h, "t", f), F32).reshape(spec.shape)

        for wh, k, pk, e, m, ins, outs in todo:
            fn = host_fn(cfg, pl.root, wh, k, pk, e)
            feeds = {s.name: load(ins[s.name], s) for s in m.inputs}
            want = fn(feeds)
            want = want if isinstance(want, tuple) else (want,)
            names = [s.name for s in m.outputs]
            if e["kind"] == "fused":
                import onnx

                order = [
                    o.name
                    for o in onnx.load(
                        os.path.join(pl.root, e["model"]), load_external_data=False
                    ).graph.output
                ]
                want = dict(zip(order, want))
            else:
                want = dict(zip(names, want))
            for s in m.outputs:
                rel = lay.rel_err(load(outs[s.name], s), want[s.name])
                rows.append(dict(wh=wh, program=pk, output=s.name, rel=rel))
                bad += rel > a.tol
        print("health after", A.health_check(sess), flush=True)
    json.dump(rows, open(os.path.join(cfg.out, f"check_{a.set}.json"), "w"), indent=1)
    worst = {}
    for r in rows:
        k = (r["program"], r["output"])
        worst[k] = max(worst.get(k, 0.0), r["rel"])
    for (pk, o), v in sorted(worst.items()):
        print(f"{pk:12s} {o:4s} worst rel {v:.2e}")
    print(f"{len(rows)} outputs checked, {bad} above {a.tol}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
