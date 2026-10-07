#!/usr/bin/env python3
"""Near-exact reference for the spec rows where onnxsim.crown is tighter than auto_LiRPA's CROWN.

For chain MLPs (ACAS Xu, MNIST-FC, ERAN ReLU nets) the exact range of ``a . f(x)`` over the input box is
computed with an independent method: a MILP (big-M, interval bounds computed here, independent of
onnxsim) solved by HiGHS, plus a batched signed-gradient PGD search. 'primal' values are attained by a
real input (so they bound the true extreme from the inside); a MILP 'dual' bound bounds it from the
outside. A sound bound must enclose the best-known extreme: ``lb <= primal_min`` and ``ub >= primal_max``.

  python vnncomp_tightness_exact.py --benches acasxu,mnistfc --milp
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from vnncomp_tightness_audit import load, ours

Layers = List[Tuple[np.ndarray, np.ndarray]]


def extract_mlp(model: onnx.ModelProto) -> Tuple[Layers, int]:
    """Chain MLP -> ``[(W, b), ...]`` with ``y_i = relu(W_i y_{i-1} + b_i)`` (no relu after the last)."""
    consts = {
        t.name: numpy_helper.to_array(t).astype(np.float64)
        for t in model.graph.initializer
    }
    for n in model.graph.node:
        if n.op_type == "Constant":
            v = next(x for x in n.attribute if x.name == "value")
            consts[n.output[0]] = numpy_helper.to_array(v.t).astype(np.float64)
    inp = [i for i in model.graph.input if i.name not in consts][0]
    dims = [
        d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim
    ]
    n_in = int(np.prod(dims))
    W, b, cur = np.eye(n_in), np.zeros(n_in), inp.name
    layers: Layers = []

    def vec(c: np.ndarray, d: int) -> np.ndarray:
        c = np.asarray(c, np.float64).reshape(-1)
        return np.full(d, c[0]) if c.size == 1 else c

    for n in model.graph.node:
        if n.op_type == "Constant":
            continue
        ins = [i for i in n.input if i]
        if n.op_type in ("Flatten", "Reshape", "Identity"):
            cur = n.output[0]
            continue
        assert ins[0] == cur, f"not a chain at {n.op_type}"
        d = W.shape[0]
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type in ("Sub", "Add"):
            b = b + (1 if n.op_type == "Add" else -1) * vec(consts[ins[1]], d)
        elif n.op_type in ("Mul", "Div"):
            s = vec(consts[ins[1]], d)
            s = s if n.op_type == "Mul" else 1.0 / s
            W, b = W * s[:, None], b * s
        elif n.op_type == "MatMul":
            B = consts[ins[1]]
            W, b = B.T @ W, B.T @ b
        elif n.op_type == "Gemm":
            B = consts[ins[1]]
            B = B.T if at.get("transB", 0) else B
            al, be = at.get("alpha", 1.0), at.get("beta", 1.0)
            C = vec(consts[ins[2]], B.shape[1]) if len(ins) > 2 else 0.0
            W, b = al * (B.T @ W), al * (B.T @ b) + be * C
        elif n.op_type == "Relu":
            layers.append((W, b))
            W, b = np.eye(W.shape[0]), np.zeros(W.shape[0])
        else:
            raise NotImplementedError(n.op_type)
        cur = n.output[0]
    layers.append((W, b))
    return layers, n_in


def pgd_extreme(
    layers: Layers,
    a: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    sign: float,
    restarts: int = 512,
    iters: int = 400,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Best-known value of ``a . f(x)`` that maximises ``sign * a . f``: batched signed-gradient ascent."""
    rng = np.random.default_rng(seed)
    x = lo + (hi - lo) * rng.random((restarts, lo.size))
    k = min(8, restarts)
    x[:k] = np.where(rng.random((k, lo.size)) < 0.5, lo, hi)  # a few corners
    best, bx = -np.inf, x[0].copy()
    for t in range(iters):
        acts, y = [], x
        for i, (W, b) in enumerate(layers):
            z = y @ W.T + b
            if i < len(layers) - 1:
                acts.append(z > 0)
                y = np.maximum(z, 0.0)
            else:
                y = z
        val = sign * (y @ a)
        j = int(np.argmax(val))
        if val[j] > best:
            best, bx = float(val[j]), x[j].copy()
        g = np.broadcast_to(sign * a, (restarts, a.size))
        for i in range(len(layers) - 1, -1, -1):
            g = g @ layers[i][0]
            if i > 0:
                g = g * acts[i - 1]
        step = (hi - lo) * 0.2 * (1 - t / iters) ** 2 + 1e-9
        x = np.clip(x + step * np.sign(g), lo, hi)
    return sign * best, bx


def ibp_layers(
    layers: Layers, lo: np.ndarray, hi: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Plain interval bounds of every pre-activation (independent of onnxsim)."""
    out, lo_, hi_ = [], lo, hi
    for i, (W, b) in enumerate(layers):
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        zl, zu = Wp @ lo_ + Wn @ hi_ + b, Wp @ hi_ + Wn @ lo_ + b
        out.append((zl, zu))
        if i < len(layers) - 1:
            lo_, hi_ = np.maximum(zl, 0), np.maximum(zu, 0)
    return out


def milp_extreme(
    layers: Layers,
    a: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    sign: float,
    time_limit: float,
) -> Tuple[Any, Any, int, int]:
    """MILP for the extreme of ``a . f`` (sign=+1: minimum, sign=-1: maximum); returns (primal, dual, status, #binaries)
    in the original sign of ``a . f`` (a primal is attained by a feasible input; the dual bounds the true extreme)."""
    import scipy.sparse as sp
    from scipy.optimize import Bounds, LinearConstraint, milp

    pre = ibp_layers(layers, lo, hi)
    n0 = lo.size
    nvar = n0
    lbs, ubs, integ = list(lo), list(hi), [0] * n0
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    rlo: List[float] = []
    rhi: List[float] = []

    def add(entries: List[Tuple[int, float]], l_: float, u_: float) -> None:
        r = len(rlo)
        for c_, v_ in entries:
            rows.append(r)
            cols.append(c_)
            vals.append(v_)
        rlo.append(l_)
        rhi.append(u_)

    prev = list(range(n0))
    for i in range(len(layers) - 1):
        W, b = layers[i]
        zl, zu = pre[i]
        d = W.shape[0]
        zs, ys = list(range(nvar, nvar + d)), list(range(nvar + d, nvar + 2 * d))
        nvar += 2 * d
        lbs += [float(v) for v in zl] + [max(float(v), 0.0) for v in zl]
        ubs += [float(v) for v in zu] + [max(float(v), 0.0) for v in zu]
        integ += [0] * (2 * d)
        for j in range(d):
            add(
                [(zs[j], 1.0)]
                + [(prev[k], -float(W[j, k])) for k in np.nonzero(W[j])[0]],
                float(b[j]),
                float(b[j]),
            )
            if zu[j] <= 0:
                add([(ys[j], 1.0)], 0.0, 0.0)
            elif zl[j] >= 0:
                add([(ys[j], 1.0), (zs[j], -1.0)], 0.0, 0.0)
            else:
                av = nvar
                nvar += 1
                lbs.append(0.0)
                ubs.append(1.0)
                integ.append(1)
                add([(ys[j], 1.0), (zs[j], -1.0)], 0.0, np.inf)
                add(
                    [(ys[j], 1.0), (zs[j], -1.0), (av, -float(zl[j]))],
                    -np.inf,
                    -float(zl[j]),
                )
                add([(ys[j], 1.0), (av, -float(zu[j]))], -np.inf, 0.0)
        prev = ys
    W, b = layers[-1]
    c = np.zeros(nvar)
    const = float(a @ b)
    aw = a @ W
    for k in np.nonzero(aw)[0]:
        c[prev[k]] = sign * aw[k]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(rlo), nvar))
    res = milp(
        c,
        constraints=LinearConstraint(A, rlo, rhi),
        integrality=np.array(integ),
        bounds=Bounds(lbs, ubs),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-7, "disp": False},
    )
    primal = sign * (res.fun + sign * const) if res.x is not None else None
    dual_b = getattr(res, "mip_dual_bound", None)
    dual = None if dual_b is None else sign * (dual_b + sign * const)
    return primal, dual, int(res.status), int(sum(integ))


def fmt(v: Any) -> str:
    return "      n/a" if v is None else f"{v:9.5g}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--benches", default="acasxu,eran,mnistfc")
    ap.add_argument("--scale", default="1.0")
    ap.add_argument("--out-dir", default="/mnt/data/cache/claude-work/crown-audit")
    ap.add_argument(
        "--n", type=int, default=6, help="rows per benchmark (largest 'tighter' first)"
    )
    ap.add_argument("--kind", default="tighter", choices=("tighter", "looser", "any"))
    ap.add_argument(
        "--variant",
        default="default",
        help="which crown variant's bounds to check (see vnncomp_tightness_audit.VARIANTS)",
    )
    ap.add_argument("--restarts", type=int, default=512)
    ap.add_argument("--milp", action="store_true")
    ap.add_argument("--max-relu", type=int, default=400)
    ap.add_argument("--time-limit", type=float, default=90.0)
    a = ap.parse_args()
    out: List[Dict[str, Any]] = []
    print(
        "min side: 'lb' must be <= the true minimum; max side: 'ub' must be >= the true maximum."
    )
    print(
        "pgd/milp primal = value attained by a real input; dual = MILP bound on the true extreme from the outside."
    )
    for bench in a.benches.split(","):
        data = json.load(
            open(os.path.join(a.out_dir, f"compare_{bench}_s{a.scale}.json"))
        )
        specs, _, _ = load(bench, a.scale)
        # the FULL key: the short form ('prop_1.vnnlib') is shared by many networks and silently selected the wrong one
        by = {(r["key"], r["group"]): r for r in specs}
        if a.kind == "tighter":
            pick = sorted(
                (r for r in data if r["d_rel"] > 1e-6), key=lambda r: -r["d_rel"]
            )[: a.n]
        elif a.kind == "looser":
            pick = sorted(
                (r for r in data if r["d_rel"] < -1e-6), key=lambda r: r["d_rel"]
            )[: a.n]
        else:
            pick = data[:: max(1, len(data) // a.n)][: a.n]
        print(f"\n== {bench}: {len(pick)} rows ({a.kind})")
        for r in pick:
            rec = by[(r["full_key"], r["group"])]
            assert os.path.basename(rec["onnx"]) == r["net"], "spec/network mismatch"
            if (
                a.variant != "default"
            ):  # check THIS variant's bounds instead of the default crown's
                v_lo, v_hi, _ = ours(rec, a.variant)
                r = dict(
                    r, ours_lb=float(v_lo[r["row"]]), ours_ub=float(v_hi[r["row"]])
                )
            model = onnx.load(rec["onnx"])
            lo, hi, Arow = (
                np.array(rec["lo"]),
                np.array(rec["hi"]),
                np.array(rec["A"])[r["row"]],
            )
            row: Dict[str, Any] = {
                "bench": bench,
                "net": r["net"],
                "group": r["group"],
                "row": r["row"],
                "ours": [r["ours_lb"], r["ours_ub"]],
                "ref64": [r["ref64_lb"], r["ref64_ub"]],
                "ref32_lb": r["ref32_lb"],
            }
            try:
                layers, _ = extract_mlp(model)
                pmin, xmin = pgd_extreme(layers, Arow, lo, hi, -1.0, a.restarts)
                pmax, _ = pgd_extreme(layers, Arow, lo, hi, +1.0, a.restarts)
                import onnxruntime as ort

                sess = ort.InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                shape = [
                    d if isinstance(d, int) and d > 0 else 1
                    for d in sess.get_inputs()[0].shape
                ]
                got = float(
                    sess.run(
                        None,
                        {
                            sess.get_inputs()[0].name: xmin.reshape(shape).astype(
                                np.float32
                            )
                        },
                    )[0].reshape(-1)
                    @ Arow
                )
                row.update(
                    pgd_min=pmin, pgd_max=pmax, ort_replay_abs_diff=abs(got - pmin)
                )
                n_relu = sum(w.shape[0] for w, _ in layers[:-1])
                row["relus"] = n_relu
                if a.milp and n_relu <= a.max_relu:
                    t0 = time.time()
                    row["milp_min"] = milp_extreme(
                        layers, Arow, lo, hi, +1.0, a.time_limit
                    )
                    row["milp_max"] = milp_extreme(
                        layers, Arow, lo, hi, -1.0, a.time_limit
                    )
                    row["milp_seconds"] = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001 - recorded, never hidden
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            # Rigorous direction only: a value ATTAINED by a real input (PGD point evaluated through the network)
            # that lies outside the bound disproves it. MILP objectives are not used for that (HiGHS accepts
            # constraint violations up to 1e-6, so its primal can overshoot the true extreme slightly).
            tmin, tmax = row.get("pgd_min"), row.get("pgd_max")
            tol = 1e-12
            row["best_min"], row["best_max"] = tmin, tmax
            row["ours_lb_encloses"] = (
                None
                if tmin is None
                else bool(r["ours_lb"] <= tmin + tol * (1 + abs(tmin)))
            )
            row["ours_ub_encloses"] = (
                None
                if tmax is None
                else bool(r["ours_ub"] >= tmax - tol * (1 + abs(tmax)))
            )
            row["ref64_lb_encloses"] = (
                None
                if tmin is None
                else bool(r["ref64_lb"] <= tmin + tol * (1 + abs(tmin)))
            )
            # Certificate from the outside: the MILP dual bounds the true extreme (min: dual <= true min <= primal)
            mm_, mx_ = row.get("milp_min"), row.get("milp_max")
            row["lb_certified_by_milp_dual"] = (
                None
                if not mm_ or mm_[1] is None
                else bool(r["ours_lb"] <= mm_[1] + 1e-7 * (1 + abs(mm_[1])))
            )
            row["ub_certified_by_milp_dual"] = (
                None
                if not mx_ or mx_[1] is None
                else bool(r["ours_ub"] >= mx_[1] - 1e-7 * (1 + abs(mx_[1])))
            )
            out.append(row)
            mm, mx = (
                row.get("milp_min") or [None, None],
                row.get("milp_max") or [None, None],
            )
            print(
                f"{r['net'][:22]:22s} g{r['group']} r{r['row']} | MIN ours {fmt(r['ours_lb'])} ref64 {fmt(r['ref64_lb'])} pgd {fmt(row.get('pgd_min'))} "
                f"milp {fmt(mm[0])}/{fmt(mm[1])} | MAX ours {fmt(r['ours_ub'])} ref64 {fmt(r['ref64_ub'])} pgd {fmt(row.get('pgd_max'))} "
                f"milp {fmt(mx[0])}/{fmt(mx[1])} | encloses lb {row['ours_lb_encloses']} ub {row['ours_ub_encloses']} {row.get('error', '')}"
            )
            json.dump(
                out,
                open(
                    os.path.join(
                        a.out_dir,
                        f"exact_{a.variant}_{a.kind}_{a.benches.replace(',', '_')}_s{a.scale}.json",
                    ),
                    "w",
                ),
                indent=1,
                default=float,
            )
    bad = [
        r
        for r in out
        if r.get("ours_lb_encloses") is False or r.get("ours_ub_encloses") is False
    ]
    print(
        f"\n== rows where OUR bound fails to enclose the best-known extreme (soundness violations): {len(bad)} of {len(out)}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
