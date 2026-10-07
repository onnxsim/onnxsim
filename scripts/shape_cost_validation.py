#!/usr/bin/env python3
"""Validate onnxsim.shape_cost against real onnxruntime runs and report how tight the bounds are.

For each model: sample concrete dynamic dims inside the ranges (both extremes plus random
points), run onnxruntime exposing every tensor, and check that

* every tensor's shape / element count / bytes is inside its certified bound,
* the liveness peak of the real run is inside ``peak_live_bytes``,
* the verified arena gives every tensor of the run a slot that fits it,
* the MACs of the real run (the same per-op formulas on the real tensor shapes) are inside ``macs``.

Then it prints tightness = certified upper bound / actual, per metric: at the upper corner of the
ranges (where a tight bound has ratio 1.0) and over the random interior samples. Not run in CI.

    python scripts/shape_cost_validation.py [--samples 12] [--json out.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnx

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"),
)
import test_shape_cost as T  # noqa: E402  (the models and the run harness live with the tests)

from onnxsim import shape_cost as SC  # noqa: E402
from onnxsim import shape_ranges as sr  # noqa: E402


def _boxes(rng, n):
    xy = rng.random((1, n, 2)) * 0.5
    wh = rng.random((1, n, 2)) * 0.5 + 0.05
    return np.concatenate([xy, xy + wh], axis=2).astype(np.float32)


def cases():
    return [
        ("conv net (N<=4, H,W in [8,32])", T.conv_net(), {"N": (1, 4), "H": (8, 32), "W": (8, 32)}, None,
         lambda rng, p: {"x": T._f(rng, p["N"], 3, p["H"], p["W"])}),
        ("transformer block (B<=4, S<=64)", T.transformer_block(), {"B": (1, 4), "S": (1, 64)}, None,
         lambda rng, p: {"x": T._f(rng, p["B"], p["S"], 16)}),
        ("transformer block (B<=8, S<=2048)", T.transformer_block(), {"B": (1, 8), "S": (1, 2048)}, None,
         None),  # bound only: too large to run everywhere; reported as a bound-only row
        ("Shape->Gather->Mul->Reshape chain", T.shape_chain(), {"B": (1, 4), "S": (1, 10)}, None,
         lambda rng, p: {"x": T._f(rng, p["B"], p["S"], 8)}),
        ("NonZero (N<=16)", T.nonzero_net(), {"N": (1, 16)}, None,
         lambda rng, p: {"x": T._f(rng, p["N"], 4) * (rng.random((p["N"], 4)) > 0.5)}),
        ("Compress (N<=32)", T.compress_net(), {"N": (1, 32)}, None,
         lambda rng, p: {"x": T._f(rng, p["N"], 6), "c": rng.random(p["N"]) > 0.5}),
        ("TopK, runtime K in [1,4]", T.topk_net(), {"N": (1, 8)}, {"k": (1, 4)},
         lambda rng, p: {"x": T._f(rng, p["N"], 10), "k": np.array([int(rng.integers(1, 5))], np.int64)}),
        ("NonMaxSuppression (B<=100, 5/class)", T.nms_net(), {"B": (1, 100)}, None,
         lambda rng, p: {"boxes": _boxes(rng, p["B"]), "scores": rng.random((1, 1, p["B"])).astype(np.float32)}),
    ]  # fmt: skip


def _random_feeds(model, int_max):
    """Random inputs for an arbitrary model: floats ~ N(0,1), ints in [0, int_max), bools coin flips."""
    inits = {t.name for t in model.graph.initializer}
    ins = [i for i in model.graph.input if i.name not in inits]

    def make(rng, p):
        feeds = {}
        for vi in ins:
            dims = [
                d.dim_value
                if d.HasField("dim_value") and d.dim_value > 0
                else p[d.dim_param]
                for d in vi.type.tensor_type.shape.dim
            ]
            et = vi.type.tensor_type.elem_type
            if et in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                feeds[vi.name] = rng.integers(0, int_max, size=dims).astype(
                    np.int64 if et == onnx.TensorProto.INT64 else np.int32
                )
            elif et == onnx.TensorProto.BOOL:
                feeds[vi.name] = rng.random(dims) > 0.5
            else:
                feeds[vi.name] = rng.standard_normal(dims).astype(np.float32)
        return feeds

    return make


def actual_macs(model, vals):
    total = 0
    for node in model.graph.node:
        m = SC._node_macs(
            node, lambda n: sr.from_ints(vals[n].shape) if n in vals else None
        )
        if m is not None:
            assert m.hi is not None
            total += m.hi
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--json")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="also validate this ONNX file (random inputs)",
    )
    ap.add_argument(
        "--dim",
        action="append",
        default=[],
        metavar="NAME=LO:HI",
        help="dynamic dim range for --model",
    )
    ap.add_argument(
        "--int-max",
        type=int,
        default=1000,
        help="exclusive upper limit of random integer inputs",
    )
    args = ap.parse_args()
    rows = []
    rng = np.random.default_rng(0)
    todo = cases() if not args.model else []
    user_dims = {
        n: (int(a), int(b))
        for n, r in (d.split("=") for d in args.dim)
        for a, b in [r.split(":")]
    }
    for path in args.model:
        mdl = onnx.load(path)
        todo.append(
            (
                os.path.basename(path),
                mdl,
                user_dims,
                None,
                _random_feeds(mdl, args.int_max),
            )
        )
    for title, model, dims, vranges, make in todo:
        t0 = time.perf_counter()
        cb = SC.bounds(model, dims, input_ranges=vranges)
        t_bound = time.perf_counter() - t0
        row = {
            "model": title, "bound_ms": round(t_bound * 1000), "tensors": len(cb.tensors),
            "unbounded": len(cb.unbounded_tensors), "complete": cb.complete,
            "peak_live": [cb.peak_live_bytes.lo, cb.peak_live_bytes.hi],
            "macs": [cb.macs.lo, cb.macs.hi],
            "arena": None if cb.arena is None else cb.arena.arena_bytes,
        }  # fmt: skip
        if make is None:
            rows.append(row)
            continue
        corner_hi = {k: v[1] for k, v in dims.items()}
        picks = [{k: v[0] for k, v in dims.items()}, corner_hi] + [
            {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in dims.items()}
            for _ in range(max(0, args.samples - 2))
        ]
        ratio_peak, ratio_macs, ratio_tensor, corner = [], [], [], {}
        for i, p in enumerate(picks):
            feeds = make(rng, p)
            vals = T.run_all_tensors(model, feeds)
            assert T.violations(cb, vals) == [], (title, p)
            pk = T.actual_peak(model, vals)
            assert cb.peak_live_bytes.lo <= pk <= cb.peak_live_bytes.hi
            am = actual_macs(model, vals)
            assert cb.macs.lo <= am <= cb.macs.hi, (title, am, cb.macs)
            if cb.arena is not None:
                for name, (_o, size) in cb.arena.tensor_offsets.items():
                    if name in vals:
                        assert vals[name].nbytes <= size
            rp = cb.peak_live_bytes.hi / pk
            rm = (cb.macs.hi / am) if am else None
            rt = [
                cb.tensors[n].bytes.hi / vals[n].nbytes
                for n in cb.tensors
                if n in vals
                and cb.tensors[n].kind in ("activation", "output")
                and vals[n].nbytes
            ]
            if p == corner_hi:
                corner = {
                    "peak": rp,
                    "macs": rm,
                    "tensor_median": float(np.median(rt)) if rt else None,
                }
            elif i >= 2:
                ratio_peak.append(rp)
                if rm:
                    ratio_macs.append(rm)
                ratio_tensor.append(float(np.median(rt)) if rt else None)
        row["runs"] = len(picks)
        row["upper_corner_ratio"] = corner
        row["interior_ratio"] = {
            "peak_median": float(np.median(ratio_peak)) if ratio_peak else None,
            "macs_median": float(np.median(ratio_macs)) if ratio_macs else None,
        }
        rows.append(row)
    print(
        f"{'model':40s} {'bound ms':>8s} {'tensors':>7s} {'unb':>3s}  runs  corner peak/macs/tensor   interior peak/macs  arena/peak_hi"
    )
    for r in rows:
        c = r.get("upper_corner_ratio") or {}
        i = r.get("interior_ratio") or {}

        def g(x, nd=2):
            return "-" if x is None else f"{x:.{nd}f}"

        ar = (
            "-"
            if r["arena"] is None
            else f"{r['arena'] / r['peak_live'][1]:.2f}"
            if r["peak_live"][1]
            else "-"
        )
        print(
            f"{r['model']:40s} {r['bound_ms']:8d} {r['tensors']:7d} {r['unbounded']:3d}  "
            f"{r.get('runs', 0):4d}  {g(c.get('peak')):>6s}/{g(c.get('macs')):>6s}/{g(c.get('tensor_median')):>6s}"
            f"          {g(i.get('peak_median')):>6s}/{g(i.get('macs_median')):>6s}   {ar:>6s}"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
