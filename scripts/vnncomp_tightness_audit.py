#!/usr/bin/env python3
"""Audit: why onnxsim.crown is tighter (or looser) than auto_LiRPA's CROWN on the VNN-LIB specs.

Data: the spec groups exported by ``vnncomp_bench.py export-specs`` (input box + spec matrix ``A``),
the float32 reference from ``vnncomp_ref_bounds.py`` and the float64 reference from
``vnncomp_tightness_ref64.py``. Sub-commands:

  compare   per-row ours vs auto_LiRPA CROWN (float32 and float64 references) at several tolerances
  ablate    one-switch-at-a-time variants of our crown against the float64 reference
  exact     near-exact reference for 'tighter' rows (MILP for small MLPs, strong PGD otherwise)

Each sub-command writes JSON under ``--out-dir`` and prints a plain-text table. Not run in CI.
"""

import argparse
import contextlib
import json
import os
import time
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import onnx

from onnxsim import crown, interval, vnnlib

RES = "/mnt/data/cache/claude-work/vnncomp/results"
BENCHES = ("acasxu", "eran", "mnistfc", "oval21")
TOLS = (1e-6, 1e-9, 1e-12)


# ----------------------------------------------------------------------------- variants of our crown


def _finish_no_intersect(lo, hi, ilo, ihi):  # type: ignore[no-untyped-def]
    """crown._finish without the final min/max against the interval bounds."""
    big = crown._BIG
    lo = np.where(np.isnan(lo) | (lo <= -big * 0.1), -np.inf, lo)
    hi = np.where(np.isnan(hi) | (hi >= big * 0.1), np.inf, hi)
    return interval._widen(lo, hi)


def _refine_variant(max_elems: int, intersect: bool):  # type: ignore[no-untyped-def]
    def refine(self, max_elems_arg: int = 4096, force: bool = False) -> None:  # type: ignore[no-untyped-def]
        if self._refined and not force:
            return
        self._refined = True
        for idx, node in enumerate(self.nodes):
            if node.op_type not in crown._NONLINEAR or not self._supported(node):
                continue
            x = node.input[0]
            if x in self.leaf or self.ib[x][0].size > max_elems:
                continue
            lo, hi = self.tensor_bounds(x)
            ilo, ihi = self.ib[x]
            if intersect:
                nlo, nhi = np.maximum(ilo, lo), np.minimum(ihi, hi)
            else:  # trust CROWN alone; fall back to the interval box where CROWN gave nothing finite
                nlo = np.where(np.isfinite(lo), lo, ilo)
                nhi = np.where(np.isfinite(hi), hi, ihi)
            self.ib[x] = (nlo, nhi)
            self._relax_cache.pop(idx, None)

    return refine


CASCADE_STATS: Dict[str, Any] = {
    "ok": 0,
    "errors": [],
    "shape_mismatch": 0,
    "tightened": 0,
}


def _refine_cascade():  # type: ignore[no-untyped-def]
    """Refinement that also re-propagates INTERVAL bounds from each refined box through the layers below.

    onnxsim's refine() intersects a refined box with the interval box computed once from the raw input box.
    Here, after every refined Relu input, the interval pass is re-run from that refined box (the graph
    suffix, with the refined tensor as its input) and the result is intersected into every downstream
    tensor, including the interval bounds used for the final bound. This is a prototype of what auto_LiRPA's
    ``compare_crown_with_ibp`` does (its IBP of a layer is computed from the already refined bounds).
    """

    def refine(self, max_elems: int = 4096, force: bool = False) -> None:  # type: ignore[no-untyped-def]
        if self._refined and not force:
            return
        self._refined = True
        out_name = self.model.graph.output[0].name
        inferred = onnx.shape_inference.infer_shapes(
            self.model
        )  # Extractor needs value_info for tensor shapes
        for idx, node in enumerate(self.nodes):
            if node.op_type not in crown._NONLINEAR or not self._supported(node):
                continue
            x = node.input[0]
            if x in self.leaf or self.ib[x][0].size > max_elems:
                continue
            lo, hi = self.tensor_bounds(x)
            ilo, ihi = self.ib[x]
            nlo, nhi = np.maximum(ilo, lo), np.minimum(ihi, hi)
            self.ib[x] = (nlo, nhi)
            self._relax_cache.pop(idx, None)
            try:
                sub = onnx.utils.Extractor(inferred).extract_model([x], [out_name])
                res = interval.propagate(sub, {x: (nlo, nhi)})
            except Exception as e:  # noqa: BLE001 - unsupported shape/op: skip the cascade for this layer
                CASCADE_STATS["errors"].append(f"{type(e).__name__}: {str(e)[:80]}")
                continue
            CASCADE_STATS["ok"] += 1
            for name, (vlo, vhi) in res.intervals.items():
                if name == x or name not in self.ib or name in self.leaf:
                    continue
                if self.ib[name][0].shape != np.asarray(vlo).shape:
                    CASCADE_STATS["shape_mismatch"] += 1
                    continue
                new = (
                    np.maximum(self.ib[name][0], vlo),
                    np.minimum(self.ib[name][1], vhi),
                )
                if np.any(new[0] > self.ib[name][0]) or np.any(
                    new[1] < self.ib[name][1]
                ):
                    CASCADE_STATS["tightened"] += 1
                self.ib[name] = new
                if name in self.ibp.intervals:
                    self.ibp.intervals[name] = self.ib[name]
                self._relax_cache.pop(idx, None)
            for j, other in enumerate(
                self.nodes
            ):  # relaxations of later nonlinear nodes read the new boxes
                if j > idx:
                    self._relax_cache.pop(j, None)

    return refine


def _relu_relax_with_alpha0(mode: str):  # type: ignore[no-untyped-def]
    orig = crown._relu_relax

    def relax(lo, hi):  # type: ignore[no-untyped-def]
        r = orig(lo, hi)
        if mode == "one":
            r["alpha0"] = r["unstable"].copy()
        elif mode == "zero":
            r["alpha0"] = np.zeros_like(r["unstable"])
        return r

    return relax


VARIANTS = (
    "default",
    "no_refine",
    "no_intersect",
    "no_intersect_no_final",
    "refine_all",
    "cascade",
    "alpha0_one",
    "alpha0_zero",
    "ibp",
)


@contextlib.contextmanager
def variant(name: str) -> Iterator[Dict[str, Any]]:
    """Patch onnxsim.crown for one named variant; yields extra kwargs for ``crown.bounds``."""
    saved = (crown._Analyzer.refine, crown._finish, crown._relu_relax)
    saved_scurve: List[Any] = [crown._scurve_relax]
    kw: Dict[str, Any] = {"method": "crown", "refine": True}
    try:
        if name == "no_refine":
            kw["refine"] = False
        elif name == "no_intersect":  # refinement and final bound trust CROWN alone
            crown._Analyzer.refine = _refine_variant(4096, False)  # type: ignore[method-assign]
            crown._finish = _finish_no_intersect  # type: ignore[assignment]
        elif name == "no_intersect_no_final":  # only the final intersection removed
            crown._finish = _finish_no_intersect  # type: ignore[assignment]
        elif name.startswith(
            "sigmoid_grid"
        ):  # finer validity grid for Sigmoid/Tanh lines => smaller slack
            g = int(name[len("sigmoid_grid") :])
            orig_scurve = crown._scurve_relax
            saved_scurve[0] = orig_scurve
            crown._scurve_relax = lambda kind, lo, hi, grid=33: orig_scurve(
                kind, lo, hi, grid=g
            )  # type: ignore[assignment]
        elif (
            name == "cascade"
        ):  # refined boxes also re-propagated through the interval pass downstream
            crown._Analyzer.refine = _refine_cascade()  # type: ignore[method-assign]
        elif name == "refine_all":  # lift the 4096-element refinement budget
            crown._Analyzer.refine = _refine_variant(1 << 40, True)  # type: ignore[method-assign]
        elif name == "alpha0_one":
            crown._relu_relax = _relu_relax_with_alpha0("one")  # type: ignore[assignment]
        elif name == "alpha0_zero":
            crown._relu_relax = _relu_relax_with_alpha0("zero")  # type: ignore[assignment]
        elif name == "ibp":
            kw["method"] = "ibp"
        elif name != "default":
            raise ValueError(name)
        yield kw
    finally:
        crown._Analyzer.refine, crown._finish, crown._relu_relax = saved  # type: ignore[method-assign]
        crown._scurve_relax = saved_scurve[0]  # type: ignore[assignment]


# ----------------------------------------------------------------------------- data


def load(
    bench: str, scale: str
) -> Tuple[
    List[Dict[str, Any]], Dict[Tuple[str, int], Any], Dict[Tuple[str, int], Any]
]:
    specs = json.load(open(f"{RES}/specs_{bench}_s{scale}.json"))
    r32 = {
        (r["key"], r["group"]): r
        for r in json.load(open(f"{RES}/ref_{bench}_s{scale}.json"))
    }
    p64 = os.path.join(
        os.environ.get("AUDIT_DIR", "/mnt/data/cache/claude-work/crown-audit"),
        f"ref64_{bench}_s{scale}.json",
    )
    r64 = {(r["key"], r["group"]): r for r in json.load(open(p64))}
    return specs, r32, r64


def ours(rec: Dict[str, Any], name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    net = onnx.load(rec["onnx"])
    inp, shape = vnnlib._single_io(net)
    lo, hi, A = np.array(rec["lo"]), np.array(rec["hi"]), np.array(rec["A"])
    spec_model, spec = vnnlib._spec_model(net, A)
    rng = vnnlib._ranges(inp, shape, lo, hi)
    with variant(name) as kw:
        t0 = time.time()
        tb = crown.bounds(spec_model, rng, output=spec, **kw)[spec]
        dt = time.time() - t0
    return (
        np.asarray(tb.lo, np.float64).reshape(-1),
        np.asarray(tb.hi, np.float64).reshape(-1),
        dt,
    )


def classify(d: np.ndarray, ref: np.ndarray, tol: float) -> Tuple[int, int, int]:
    s = tol * (1 + np.abs(ref))
    return int(np.sum(d > s)), int(np.sum(np.abs(d) <= s)), int(np.sum(d < -s))


# ----------------------------------------------------------------------------- compare


def cmd_compare(a: argparse.Namespace) -> int:
    summary: Dict[str, Any] = {}
    cases: List[Dict[str, Any]] = []
    print(
        "ours = default crown; ref64 / ref32 = auto_LiRPA CROWN in float64 / float32; counts are spec rows"
    )
    print(
        f"{'bench':8s} {'rows':>5s} | "
        + " | ".join(f"tol {t:g}: tight/same/loose" for t in TOLS)
        + " | ref32 vs ref64 (tol 1e-6): tight/same/loose"
    )
    for b in a.benches.split(","):
        specs, r32, r64 = load(b, a.scale)
        rows: List[Dict[str, Any]] = []
        for rec in specs:
            k = (rec["key"], rec["group"])
            if k not in r64 or "CROWN" not in r64[k] or "lb" not in r64[k]["CROWN"]:
                continue
            lo, hi, dt = ours(rec, "default")
            rl, ru = np.array(r64[k]["CROWN"]["lb"]), np.array(r64[k]["CROWN"]["ub"])
            l32 = (
                np.array(r32[k]["CROWN"]["lb"])
                if k in r32 and "lb" in r32[k].get("CROWN", {})
                else None
            )
            il, iu = np.array(r64[k]["IBP"]["lb"]), np.array(r64[k]["IBP"]["ub"])
            for j in range(len(lo)):
                rows.append(
                    {
                        "key": rec["key"].split("|")[-1],
                        "full_key": rec["key"],
                        "net": os.path.basename(rec["onnx"]),
                        "group": rec["group"],
                        "row": j,
                        "ours_lb": lo[j],
                        "ours_ub": hi[j],
                        "ref64_lb": rl[j],
                        "ref64_ub": ru[j],
                        "ref32_lb": None if l32 is None else l32[j],
                        "ibp64_lb": il[j],
                        "ibp64_ub": iu[j],
                        "seconds": dt,
                    }
                )
        ol = np.array([r["ours_lb"] for r in rows])
        rl = np.array([r["ref64_lb"] for r in rows])
        d = ol - rl
        counts = [classify(d, rl, t) for t in TOLS]
        has32 = np.array([r["ref32_lb"] is not None for r in rows])
        l32 = np.array(
            [r["ref32_lb"] if r["ref32_lb"] is not None else np.nan for r in rows]
        )
        c32 = classify((l32 - rl)[has32], rl[has32], 1e-6) if has32.any() else (0, 0, 0)
        print(
            f"{b:8s} {len(rows):5d} | "
            + " | ".join(f"{c[0]:4d}/{c[1]:4d}/{c[2]:4d}" for c in counts)
            + f" | {c32[0]}/{c32[1]}/{c32[2]}"
        )
        summary[b] = {
            "rows": len(rows),
            "counts": {str(t): c for t, c in zip(TOLS, counts)},
            "ref32_vs_ref64_1e-6": c32,
        }
        for r in rows:
            r["d_rel"] = (r["ours_lb"] - r["ref64_lb"]) / (1 + abs(r["ref64_lb"]))
            wo, wr = r["ours_ub"] - r["ours_lb"], r["ref64_ub"] - r["ref64_lb"]
            r["width_ratio"] = wo / wr if wr > 0 and np.isfinite(wo) else None
        order = sorted(rows, key=lambda r: r["d_rel"])
        cases += [
            dict(r, bench=b, kind="looser") for r in order[:3] if r["d_rel"] < -1e-9
        ]
        cases += [
            dict(r, bench=b, kind="tighter")
            for r in order[::-1][:3]
            if r["d_rel"] > 1e-9
        ]
        wr_ = np.array([r["width_ratio"] for r in rows if r["width_ratio"] is not None])
        summary[b]["width_ratio_median"] = float(np.median(wr_)) if len(wr_) else None
        summary[b]["max_tighter_rel"] = float(max(r["d_rel"] for r in rows))
        summary[b]["max_looser_rel"] = float(min(r["d_rel"] for r in rows))
        json.dump(
            rows, open(os.path.join(a.out_dir, f"compare_{b}_s{a.scale}.json"), "w")
        )
    json.dump(
        {"summary": summary, "cases": cases},
        open(os.path.join(a.out_dir, f"compare_summary_s{a.scale}.json"), "w"),
        indent=1,
    )
    print(
        "\nlargest differences per benchmark (rel = (ours_lb - ref64_lb) / (1 + |ref64_lb|)); widths are ub - lb of the spec row"
    )
    print(
        f"{'kind':7s} {'bench':8s} {'net':22s} g{'':2s}{'row':>3s} {'ours lb':>12s} {'ours ub':>12s} {'ref64 lb':>12s} {'ref64 ub':>12s} {'rel diff':>10s} {'width ratio':>11s}"
    )
    for c in cases:
        wr = "n/a" if c["width_ratio"] is None else f"{c['width_ratio']:.4f}"
        print(
            f"{c['kind']:7s} {c['bench']:8s} {c['net'][:22]:22s} g{c['group']:<2d}{c['row']:3d} {c['ours_lb']:12.5g} {c['ours_ub']:12.5g} {c['ref64_lb']:12.5g} {c['ref64_ub']:12.5g} {c['d_rel']:10.2e} {wr:>11s}"
        )
    return 0


# ----------------------------------------------------------------------------- ablate


def cmd_ablate(a: argparse.Namespace) -> int:
    out: Dict[str, Any] = {}
    benches = a.benches.split(",")
    variants = a.variants.split(",") if a.variants else list(VARIANTS)
    print(
        "each variant vs auto_LiRPA CROWN float64 at tol 1e-9: tighter/same/looser (spec rows); seconds = total"
    )
    print(f"{'bench':8s} " + " ".join(f"{v:>24s}" for v in variants))
    for b in benches:
        specs, _, r64 = load(b, a.scale)
        line = []
        for v in variants:
            ds, refs, secs = [], [], 0.0
            for rec in specs:
                k = (rec["key"], rec["group"])
                if k not in r64 or "lb" not in r64[k].get("CROWN", {}):
                    continue
                try:
                    lo, _, dt = ours(rec, v)
                except Exception as e:  # noqa: BLE001 - recorded
                    out.setdefault(b, {}).setdefault(v, {})["error"] = (
                        f"{type(e).__name__}: {str(e)[:120]}"
                    )
                    continue
                rl = np.array(r64[k]["CROWN"]["lb"])
                ds.append(lo - rl)
                refs.append(rl)
                secs += dt
            d, rl = np.concatenate(ds), np.concatenate(refs)
            c = classify(d, rl, 1e-9)
            out.setdefault(b, {})[v] = {
                "tighter": c[0],
                "same": c[1],
                "looser": c[2],
                "seconds": round(secs, 2),
                "max_tighter_rel": float(np.max(d / (1 + np.abs(rl)))),
                "max_looser_rel": float(np.min(d / (1 + np.abs(rl)))),
            }
            line.append(f"{c[0]:4d}/{c[1]:4d}/{c[2]:4d} {secs:6.1f}s")
        print(f"{b:8s} " + " ".join(f"{x:>24s}" for x in line))
    json.dump(
        out,
        open(
            os.path.join(a.out_dir, f"ablate_{'_'.join(benches)}_s{a.scale}.json"), "w"
        ),
        indent=1,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("compare", cmd_compare), ("ablate", cmd_ablate)):
        p = sub.add_parser(name)
        p.add_argument("--scale", default="1.0")
        p.add_argument("--out-dir", default="/mnt/data/cache/claude-work/crown-audit")
        p.add_argument("--benches", default=",".join(BENCHES))
        p.add_argument("--variants", default="")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
