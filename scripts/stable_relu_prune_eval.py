"""Evaluate onnxsim.stable_relu_prune on REAL trained models (not run in CI).

Two tasks, results written as JSON next to a printed summary:

* ``digits``  -- ``--nets`` independent deep MLPs trained here on the real sklearn digits
  (the same recipe as scripts/quant_sensitivity_bench.py). Boxes: the valid-pixel box
  ``[0, 1]^64`` (CERTIFIED for every image) and the per-pixel min/max of the training data
  (NOT certified for other inputs).
* ``resnet18`` -- torchvision ResNet18 (ImageNet weights, ONNX export, BatchNorm folded) on
  real ImageNet validation images (uint8 NHWC arrays prepared by scripts/quant_sensitivity_prep.py).
  Boxes: the valid-image box (per-channel ``((0-mean)/std, (1-mean)/std)``, CERTIFIED), the
  per-pixel min/max of the data (NOT certified), and, as the ceiling, how many channels are
  *empirically* never positive / never negative on real images (a statement about those
  images only). It also prunes the empirically dead channels with ``prune_units`` (UNCHECKED)
  to show what a data-driven version would cost in accuracy.

Usage::

    python scripts/stable_relu_prune_eval.py digits --nets 10 --out RESULTS_DIR
    python scripts/stable_relu_prune_eval.py resnet18 --data DIR_WITH_NPY --model resnet18.onnx --out RESULTS_DIR
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnx

from onnxsim import stable_relu_prune as S


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _session(model: onnx.ModelProto, extra_outputs=()):
    import onnxruntime as ort

    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for n in extra_outputs:
        if n not in have:
            m.graph.output.append(
                onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None)
            )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(
        m.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )


def empirical_stability(model, rep, batches, input_name):
    """Per ReLU layer: units that were never positive / never negative on the given data."""
    shapes = S._static_shapes(S._pin_dynamic_dims(model))
    names = [lay.pre_activation for lay in rep.layers]
    sess = _session(model, names)
    lo = {}
    hi = {}
    out_names = [o.name for o in sess.get_outputs()]
    for xb in batches:
        outs = dict(zip(out_names, sess.run(None, {input_name: xb})))
        for n in names:
            a = outs[n]
            ax = 1 if a.ndim == 4 else a.ndim - 1
            red = tuple(i for i in range(a.ndim) if i != ax)
            mn, mx = a.min(axis=red), a.max(axis=red)
            lo[n] = mn if n not in lo else np.minimum(lo[n], mn)
            hi[n] = mx if n not in hi else np.maximum(hi[n], mx)
    res = []
    for lay in rep.layers:
        n = lay.pre_activation
        res.append(
            {
                "relu": lay.relu,
                "units": int(len(lo[n])),
                "emp_dead": int((hi[n] <= 0).sum()),
                "emp_on": int((lo[n] >= 0).sum()),
                "emp_dead_units": np.flatnonzero(hi[n] <= 0).tolist(),
            }
        )
    del shapes
    return res


def macs_total(model) -> int:
    pinned = S._pin_dynamic_dims(model)
    gr = S._Graph(pinned, S._static_shapes(pinned))
    tot = 0
    for n in pinned.graph.node:
        if n.op_type in ("Conv", "MatMul", "Gemm") and n.input[1] in gr.init:
            tot += S._macs_of(gr, n, gr.init[n.input[1]].shape)
    return tot


def params_total(model) -> int:
    return int(sum(np.prod(t.dims) for t in model.graph.initializer))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fidelity(sess_a, sess_b, batches, input_name, labels=None):
    """top-1 agreement, mean KL(a || b), max |logit diff|, and accuracy of both if labels given."""
    agree = n = 0
    kl = 0.0
    maxd = 0.0
    corr_a = corr_b = 0
    k = 0
    for xb in batches:
        a = sess_a.run(None, {input_name: xb})[0].astype(np.float64)
        b = sess_b.run(None, {input_name: xb})[0].astype(np.float64)
        pa, pb = softmax(a), softmax(b)
        kl += float((pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))).sum())
        agree += int((a.argmax(1) == b.argmax(1)).sum())
        maxd = max(maxd, float(np.abs(a - b).max()))
        if labels is not None:
            yb = labels[k : k + len(a)]
            corr_a += int((a.argmax(1) == yb).sum())
            corr_b += int((b.argmax(1) == yb).sum())
        k += len(a)
        n += len(a)
    out = {"agree": agree / n, "kl": kl / n, "max_logit_diff": maxd, "n": n}
    if labels is not None:
        out["acc_orig"] = corr_a / n
        out["acc_pruned"] = corr_b / n
    return out


def summarize_report(rep):
    return {
        "methods": rep.methods,
        "skipped_methods": rep.skipped_methods,
        "units": rep.total_units,
        "dead": rep.total("dead"),
        "always_on": rep.total("always_on"),
        "unstable": rep.total("unstable"),
        "by_method": {m: list(v) for m, v in rep.total_by_method().items()},
        "prunable_layers": sum(1 for lay in rep.layers if lay.prunable),
        "layers": len(rep.layers),
        "params_removed": rep.params_removed,
        "macs_removed": rep.macs_removed,
        "per_layer": [
            {
                "relu": lay.relu,
                "units": lay.units,
                "dead": lay.dead,
                "on": lay.always_on,
                "prunable": lay.prunable,
                "reason": lay.reason,
                "by_method": {k: list(v) for k, v in lay.by_method.items()},
            }
            for lay in rep.layers
        ],
    }


# ---------------------------------------------------------------------------------------
# digits
# ---------------------------------------------------------------------------------------


def train_digits_mlps(n_nets, depth, width, seed0):
    import torch
    from onnx import numpy_helper, parser
    from sklearn.datasets import load_digits

    d = load_digits()
    x = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    tr, ca, te = perm[:900], perm[900:1200], perm[1200:]
    models = []
    for k in range(n_nets):
        torch.manual_seed(seed0 + k)
        dims = [64] + [width] * (depth - 1) + [10]
        net = torch.nn.Sequential(
            *[
                m
                for i in range(depth)
                for m in (
                    torch.nn.Linear(dims[i], dims[i + 1]),
                    torch.nn.ReLU() if i < depth - 1 else torch.nn.Identity(),
                )
            ]
        )
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        xt, yt = torch.from_numpy(x[tr]), torch.from_numpy(y[tr])
        for _ in range(400):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt), yt).backward()
            opt.step()
        lines, init, prev = [], {}, "x"
        lin = [m for m in net if isinstance(m, torch.nn.Linear)]
        for i, m in enumerate(lin):
            init[f"W{i}"] = m.weight.detach().numpy().T.astype(np.float32).copy()
            init[f"B{i}"] = m.bias.detach().numpy().astype(np.float32).copy()
            out = "logits" if i == depth - 1 else f"r{i}"
            lines += [f"h{i} = MatMul({prev}, W{i})", f"a{i} = Add(h{i}, B{i})"]
            lines.append(
                f"{out} = Identity(a{i})" if i == depth - 1 else f"{out} = Relu(a{i})"
            )
            prev = out
        body = "m (float[N,64] x) => (float[N,10] logits) { " + " ".join(lines) + " }"
        mdl = parser.parse_model('<ir_version: 9, opset_import: ["" : 17]> ' + body)
        mdl.graph.initializer.extend(
            numpy_helper.from_array(v, n) for n, v in init.items()
        )
        models.append(mdl)
    return models, x[np.concatenate([tr, ca])], x[te], y[te]


def main_digits(args):
    models, fit_x, test_x, test_y = train_digits_mlps(
        args.nets, args.depth, args.width, args.seed
    )
    lo_d, hi_d = fit_x.min(axis=0), fit_x.max(axis=0)
    boxes = {
        "valid-pixel box [0,1]^64 (certified)": (np.zeros(64), np.ones(64)),
        "per-pixel data box (NOT certified)": (lo_d, hi_d),
    }
    in_box = np.all((test_x >= lo_d) & (test_x <= hi_d), axis=1).mean()
    log(
        f"digits: {args.nets} nets, depth {args.depth} width {args.width}; "
        f"{(lo_d == hi_d).sum()} of 64 pixels are constant in the fitting data; "
        f"{in_box:.1%} of held-out digits lie inside the data box"
    )
    results = []
    for k, m in enumerate(models):
        sess0 = _session(m)
        acc = float((sess0.run(None, {"x": test_x})[0].argmax(1) == test_y).mean())
        row = {"net": k, "float_accuracy": acc, "boxes": {}}
        rep0 = S.analyze(
            m,
            {"x": boxes["valid-pixel box [0,1]^64 (certified)"]},
            methods=("interval",),
        )
        emp = empirical_stability(m, rep0, [fit_x], "x")
        row["empirical"] = {
            "dead": sum(e["emp_dead"] for e in emp),
            "on": sum(e["emp_on"] for e in emp),
            "units": sum(e["units"] for e in emp),
        }
        # UNCHECKED data-driven pruning: remove the units that were never positive on the fitting
        # data (a statement about that data only) and measure what it costs on held-out digits.
        units = {e["relu"]: e["emp_dead_units"] for e in emp if e["emp_dead_units"]}
        unc, n_removed, _ = S.prune_units(m, units)
        row["empirical_prune"] = {
            "units_removed": n_removed,
            "params_before": params_total(m),
            "params_after": params_total(unc),
            "fidelity_heldout": fidelity(sess0, _session(unc), [test_x], "x", test_y),
        }
        for bname, (lo, hi) in boxes.items():
            t0 = time.time()
            rep = S.analyze(
                m,
                {"x": (lo, hi)},
                methods=("interval", "crown", "zonotope"),
                max_elements={"zonotope": 100000, "crown": 100000},
            )
            ana_s = time.time() - t0
            pruned, prep = S.apply(
                m,
                {"x": (lo, hi)},
                merge_active=True,
                methods=("interval", "crown"),
                max_elements={"crown": 100000},
            )
            fid = fidelity(sess0, _session(pruned), [test_x], "x", test_y)
            inside = np.all((test_x >= lo) & (test_x <= hi), axis=1)
            row["boxes"][bname] = {
                **summarize_report(rep),
                "analyze_seconds": ana_s,
                "removed_units": prep.removed_units,
                "merged_layers": prep.merged_layers,
                "params_before": params_total(m),
                "params_after": params_total(pruned),
                "fidelity_heldout": fid,
                "heldout_inside_box": float(inside.mean()),
                "self_check_max_diff": prep.verified_max_abs_diff,
            }
        # How fast does provable stability disappear as the box grows? Scale the data box toward
        # the mean image: s = 0 is a single point (everything is stable), s = 1 is the data box.
        mean = fit_x.mean(axis=0)
        curve = []
        for sc in args.shrink:
            lo_s, hi_s = mean + sc * (lo_d - mean), mean + sc * (hi_d - mean)
            r = S.analyze(
                m,
                {"x": (lo_s, hi_s)},
                methods=("interval", "crown", "zonotope"),
                max_elements={"zonotope": 100000, "crown": 100000},
            )
            curve.append(
                {
                    "scale": sc,
                    "units": r.total_units,
                    "dead_on_union": r.total("dead") + r.total("always_on"),
                    "by_method": {mm: list(v) for mm, v in r.total_by_method().items()},
                }
            )
        row["shrink_curve"] = curve
        results.append(row)
        b0 = row["boxes"]["valid-pixel box [0,1]^64 (certified)"]
        b1 = row["boxes"]["per-pixel data box (NOT certified)"]
        log(
            f"net {k}: acc {acc:.3f} | certified box: dead {b0['dead']} on {b0['always_on']} "
            f"unstable {b0['unstable']} of {b0['units']} | data box: dead {b1['dead']} on {b1['always_on']} "
            f"| empirical dead {row['empirical']['dead']} on {row['empirical']['on']}"
        )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "digits.json"), "w") as f:
        json.dump(
            {
                "task": "digits",
                "nets": args.nets,
                "depth": args.depth,
                "width": args.width,
                "constant_pixels": int((lo_d == hi_d).sum()),
                "heldout_inside_data_box": float(in_box),
                "results": results,
            },
            f,
            indent=1,
        )
    return 0


# ---------------------------------------------------------------------------------------
# resnet18
# ---------------------------------------------------------------------------------------

_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def prep_images(u8):
    x = u8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return ((x - _MEAN) / _STD).astype(np.float32)


def main_resnet18(args):
    xs = np.load(os.path.join(args.data, "imagenet_val5k_x.npy"), mmap_mode="r")
    ys = np.load(os.path.join(args.data, "imagenet_val5k_y.npy"))
    model = onnx.load(args.model)
    cal_n = args.calib
    bs = 50
    # The arrays are class-ordered (every 10th ImageNet image, sorted by class), so a prefix split
    # would calibrate on some classes and evaluate on others. Use a seeded random split instead.
    perm = np.random.default_rng(0).permutation(len(ys))
    cal_idx = np.sort(perm[:cal_n])
    eval_idx = np.sort(perm[cal_n:])
    y_eval = ys[eval_idx]

    def batches(idx):
        for i in range(0, len(idx), bs):
            yield prep_images(np.asarray(xs[idx[i : i + bs]]))

    all_idx = np.arange(len(ys))
    log(
        f"resnet18: {len(ys)} images; random split (seed 0): {len(cal_idx)} calibration, {len(eval_idx)} evaluation"
    )
    valid_lo = ((0 - _MEAN) / _STD).astype(np.float64)
    valid_hi = ((1 - _MEAN) / _STD).astype(np.float64)
    valid = (
        np.broadcast_to(valid_lo, (1, 3, 224, 224)),
        np.broadcast_to(valid_hi, (1, 3, 224, 224)),
    )
    # per-pixel data box over ALL images (streaming)
    t0 = time.time()
    dlo = np.full((1, 3, 224, 224), np.inf, np.float32)
    dhi = np.full((1, 3, 224, 224), -np.inf, np.float32)
    for xb in batches(all_idx):
        dlo = np.minimum(dlo, xb.min(axis=0, keepdims=True))
        dhi = np.maximum(dhi, xb.max(axis=0, keepdims=True))
    log(
        f"per-pixel data box built in {time.time() - t0:.0f}s; mean width as a fraction of the valid box: "
        f"{float(((dhi - dlo) / (valid[1] - valid[0])).mean()):.4f}"
    )
    out = {"task": "resnet18", "images": int(len(ys)), "calibration": cal_n}
    for bname, box in (
        ("valid-image box (certified)", valid),
        (
            "per-pixel data box (NOT certified)",
            (dlo.astype(np.float64), dhi.astype(np.float64)),
        ),
    ):
        t0 = time.time()
        rep = S.analyze(model, {"x": box}, methods=("interval", "crown"))
        s = summarize_report(rep)
        s["seconds"] = time.time() - t0
        out[bname] = s
        log(
            f"[{bname}] interval: {rep.total_by_method().get('interval')} (dead, on) of {rep.total_units} units "
            f"in {s['seconds']:.0f}s; skipped: {rep.skipped_methods}"
        )
    # empirical ceiling on the calibration images
    rep0 = S.analyze(model, {"x": valid}, methods=("interval",))
    t0 = time.time()
    emp = empirical_stability(model, rep0, batches(cal_idx), "x")
    log(
        f"empirical stability over {cal_n} real images ({time.time() - t0:.0f}s): "
        f"dead {sum(e['emp_dead'] for e in emp)}, on {sum(e['emp_on'] for e in emp)} of {sum(e['units'] for e in emp)} channels"
    )
    out["empirical"] = [
        {k: v for k, v in e.items() if k != "emp_dead_units"} for e in emp
    ]
    # certified apply (whatever it finds; likely nothing) and unchecked data-driven pruning
    sess_f = _session(model)
    pruned, prep = S.apply(model, {"x": valid}, methods=("interval",), verify_samples=2)
    out["certified_apply"] = {
        "removed_units": prep.removed_units,
        "params_removed": prep.params_removed,
    }
    log(f"certified apply (valid box): removed {prep.removed_units} units")
    if prep.removed_units:
        out["certified_apply"]["fidelity_eval"] = fidelity(
            sess_f, _session(pruned), batches(eval_idx), "x", y_eval
        )
    units = {e["relu"]: e["emp_dead_units"] for e in emp if e["emp_dead_units"]}
    p_before, m_before = params_total(model), macs_total(model)
    unc, removed, skipped = S.prune_units(model, units)
    out["empirical_prune"] = {
        "units_requested": sum(len(v) for v in units.values()),
        "units_removed": removed,
        "skipped": skipped,
        "params_before": p_before,
        "params_after": params_total(unc),
        "macs_before": m_before,
        "macs_after": macs_total(unc),
    }
    if removed:
        out["empirical_prune"]["fidelity_eval"] = fidelity(
            sess_f, _session(unc), batches(eval_idx), "x", y_eval
        )
    else:
        out["empirical_prune"]["fidelity_eval"] = fidelity(
            sess_f, sess_f, batches(eval_idx), "x", y_eval
        )
    log(
        "empirical (UNCHECKED) prune:",
        json.dumps(out["empirical_prune"], default=float)[:600],
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "resnet18.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="task", required=True)
    d = sub.add_parser("digits")
    d.add_argument("--nets", type=int, default=10)
    d.add_argument("--depth", type=int, default=8)
    d.add_argument("--width", type=int, default=32)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument(
        "--shrink",
        type=lambda v: [float(x) for x in v.split(",")],
        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        help="scales of the data box toward the mean image for the stability-vs-box-size curve",
    )
    d.add_argument("--out", required=True)
    r = sub.add_parser("resnet18")
    r.add_argument("--data", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--calib", type=int, default=2000)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    return main_digits(args) if args.task == "digits" else main_resnet18(args)


if __name__ == "__main__":
    sys.exit(main())
