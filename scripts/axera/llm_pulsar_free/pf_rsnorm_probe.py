"""Probe: can Pulsar2 compile a per-row row-scaled RMS norm in U16, and does the card match float?

The norm after a residual Add in Qwen3-4B layout 8 uses one per-tensor U16 scale for ``x * x``;
a row with a massive channel (|x| ~ 6300 at 2560 wide) then zeroes every ordinary row
(``pf_templates`` NORM2_C). The row-scaled form divides ``x`` by its per-row max |x| first:

    s = ReduceMax(Abs(x));  xs = x / s;  u = xs / sqrt(mean(xs^2) + eps)

``u`` equals ``x / sqrt(mean(x^2) + eps)`` up to the eps term (the scale cancels), and every
U16 tensor after the division lies in [-1, 1]. This script builds that single program with
Pulsar2 (one ``pulsar2_docker`` build, ~20 s), runs it on the card on fresh rows, and compares
the card's output with float and with the host fake-quant of the input alone (the floor a
per-tensor U16 input sets, whatever the norm does).

usage:
  pf_rsnorm_probe.py build OUT_DIR [--seed 7]        (heavy lock; OUT_DIR/out/compiled.axmodel)
  pf_rsnorm_probe.py card COMPILED OUT_JSON           (card lock; health check before and after)
  pf_rsnorm_probe.py ident A_COMPILED B_COMPILED      (job records: compare_jobs)
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

F = np.float32
H = 2560
EPS = 1e-6
#: the single program: input ``x`` [1, H], output ``u``
BODY = """
  ax = Abs(x)
  mx = ReduceMax<axes=[-1], keepdims=1>(ax)
  xs = Div(x, mx)
  sq = Mul(xs, xs)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  u = Div(xs, r)
"""
OPS = ["Abs", "ReduceMax", "Div", "Mul", "ReduceMean", "Add", "Sqrt"]


def calibration(seed: int) -> list[np.ndarray]:
    """18 rows: ordinary rows, and two massive-channel rows of opposite sign (the input's
    U16 range must cover both -4200 and +6309, as the real residual's does)."""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(18):
        x = (r.standard_normal((1, H)) * r.uniform(1.0, 3.0)).astype(F)
        if i == 3:
            x[0, 4], x[0, 0], x[0, 100] = 6309.0, 300.0, -180.0
        if i == 11:
            x[0, 4], x[0, 100] = -4200.0, 250.0
        rows.append(x)
    return rows


def build(out_dir: str, seed: int) -> bool:
    import onnx
    import pf_templates as T

    os.makedirs(out_dir, exist_ok=True)
    m = T.model_of(BODY, {"x": (1, H)}, {"u": (1, H)}, {"eps": np.array([[EPS]], F)})
    onnx.save(m, os.path.join(out_dir, "model.onnx"))
    ok, err = T._pulsar2_build(out_dir, ("x",), {"x": calibration(seed)}, OPS)
    print("build", "ok" if ok else "REFUSED", "" if ok else err, flush=True)
    return ok


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b**2).mean()))


def ref(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return x64 / np.sqrt((x64**2).mean(1, keepdims=True) + EPS)


def test_rows() -> list[tuple[str, np.ndarray]]:
    r = np.random.default_rng(12345)  # not the calibration seeds
    out = []
    for name in (
        "ordinary_a",
        "ordinary_b",
        "outlier_pos",
        "outlier_neg",
        "small_scale",
    ):
        x = (r.standard_normal((1, H)) * r.uniform(1.0, 3.0)).astype(F)
        if name == "small_scale":  # below the input's U16 step: informational only
            x = x * F(1e-2)
        if name.startswith("outlier"):
            x[0, 4] = 6309.0 if name == "outlier_pos" else -4200.0
            x[0, 0], x[0, 100] = 300.0, -180.0
        out.append((name, x))
    return out


def card(compiled: str, out_json: str) -> dict:
    import axcl_session as A

    rows = test_rows()
    lo, hi = (
        float(min(x.min() for x in calibration(7))),
        float(max(x.max() for x in calibration(7))),
    )
    step = (hi - lo) / 65535.0
    res = {"model": compiled, "input_range": [lo, hi], "input_step": step, "rows": []}
    with A.AXSession() as s:
        res["health_before"] = A.health_check(s)
        m = s.load(compiled)
        res["in"] = [(i.name, i.shape) for i in m.inputs]
        res["out"] = [(o.name, o.shape) for o in m.outputs]
        for name, x in rows:
            (y,) = s.run(m, [x])
            y = y.reshape(1, H).astype(np.float64)
            want = ref(x)
            xq = (np.rint((x - lo) / step) * step + lo).astype(F)
            floor = ref(xq)  # the input's own U16 rounding, nothing else
            ordi = np.abs(x[0]) < 100
            res["rows"].append(
                dict(
                    row=name,
                    card_rel_rms=rel(y, want),
                    host_input_floor_rel_rms=rel(floor, want),
                    max_abs_err=float(np.abs(y - want).max()),
                    nan=int(np.isnan(y).sum()),
                    ordinary_elems_rel_rms=rel(y[0][ordi], want[0][ordi]),
                )
            )
        s.unload(m)
        res["health_after"] = A.health_check(s)
    json.dump(res, open(out_json, "w"), indent=1)
    for r in res["rows"]:
        print(
            f"{r['row']:12s} card {r['card_rel_rms']:.4g}  input-floor {r['host_input_floor_rel_rms']:.4g}"
            f"  max|err| {r['max_abs_err']:.4g}  nan {r['nan']}",
            flush=True,
        )
    return res


def ident(a: str, b: str) -> list[str]:
    import onnx
    import u16_retarget as ur

    return ur.compare_jobs(onnx.load(a), onnx.load(b))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("card")
    c.add_argument("compiled")
    c.add_argument("out_json")
    i = sub.add_parser("ident")
    i.add_argument("a")
    i.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return 0 if build(args.out_dir, args.seed) else 1
    if args.cmd == "card":
        card(args.compiled, args.out_json)
        return 0
    diffs = ident(args.a, args.b)
    print("compare_jobs diffs:", diffs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
