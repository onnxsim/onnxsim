#!/usr/bin/env python3
"""Do quantization-sensitivity estimators predict real sensitivity? (not run in CI)

For a real pretrained model on real labelled data this

1. splits the data into a CALIBRATION split (estimators + activation quantizers are fit here) and a
   disjoint EVALUATION split (ground truth is measured here);
2. measures ground truth by brute force: quantize ONE group (site or block) at a time and record
   the logit KL to the float model, top-1 agreement and accuracy on the evaluation split;
3. scores every group with each estimator in ``onnxsim.quant_sensitivity`` using only the
   calibration split;
4. reports Spearman / Kendall of each estimator against the ground truth, with a bootstrap over
   evaluation samples for the ground truth's own noise;
5. runs the decision-relevant test: keep the k groups an estimator calls most sensitive in float,
   quantize everything else, and measure the result (against random and an ORACLE built from the
   ground truth).

Usage (see docs/quant-sensitivity.md for the exact commands and the measured results)::

    python scripts/quant_sensitivity_bench.py cnn --model resnet18 --kind weights --bits 4
    python scripts/quant_sensitivity_bench.py bert --kind weights --bits 4
    python scripts/quant_sensitivity_bench.py digits --kind weights --bits 3 --nets 10
    python scripts/quant_sensitivity_bench.py stability --model resnet18 --kind weights --bits 4   # needs the cnn run's JSON
    python scripts/quant_sensitivity_bench.py brute --model resnet18 --kind weights --bits 4       # like-for-like cost baseline

Data: the CNN benchmark reads ``imagenet_val5k_{x,y}.npy`` (every 10th image of the ImageNet-1k
validation set, class stratified, resized 256 / center-crop 224; produced by the preparation
script described in the docs). Everything is cached under ``--workdir``.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import onnx

from onnxsim import quant_sensitivity as qs

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_stratified(y: np.ndarray, n_calib_per_class: int):
    """Per class, the first ``n_calib_per_class`` samples are calibration, the rest evaluation."""
    calib, evals = [], []
    seen: dict = {}
    for i, c in enumerate(y.tolist()):
        k = seen.get(c, 0)
        seen[c] = k + 1
        (calib if k < n_calib_per_class else evals).append(i)
    return np.array(calib), np.array(evals)


def load_cnn_data(workdir: str, calib_per_class: int):
    x = np.load(os.path.join(workdir, "imagenet_val5k_x.npy"))
    y = np.load(os.path.join(workdir, "imagenet_val5k_y.npy"))
    ci, ei = split_stratified(y, calib_per_class)

    def prep(idx):
        a = x[idx].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return ((a - MEAN) / STD).astype(np.float32)

    return {"x": prep(ci)}, y[ci], {"x": prep(ei)}, y[ei]


def load_bert_data(workdir: str, calib_n: int):
    """SST-2 validation (872 real labelled sentences), tokenized for DistilBERT, fixed length."""
    z = np.load(os.path.join(workdir, "sst2_val_tok.npz"))
    ids, mask, y = z["input_ids"], z["attention_mask"], z["label"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    ci, ei = perm[:calib_n], perm[calib_n:]
    return (
        {"input_ids": ids[ci], "attention_mask": mask[ci]},
        y[ci],
        {"input_ids": ids[ei], "attention_mask": mask[ei]},
        y[ei],
    )


def bert_groups(sites):
    """One group per transformer block (all its linear layers), plus the classification head."""
    groups: dict = {}
    for s in sites:
        m = re.search(r"/layer\.(\d+)/", s.name)
        groups.setdefault(f"block{int(m.group(1))}" if m else "head", []).append(s.name)
    return groups


def bootstrap_corr(
    est: np.ndarray, kl_samples: np.ndarray, groups, corr, B: int, seed: int
):
    """Percentile CI of corr(estimator, ground truth) when the evaluation samples are resampled."""
    rng = np.random.default_rng(seed)
    n = kl_samples.shape[1]
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        gt = kl_samples[:, idx].mean(axis=1)
        c = corr(est, gt)
        if not np.isnan(c):
            vals.append(c)
    return (
        (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        if vals
        else (float("nan"),) * 2
    )


def train_digits_mlps(n_nets: int, depth: int, width: int, seed0: int):
    """Train ``n_nets`` independent deep MLPs on the REAL sklearn digits (8x8 handwritten images).

    These weights are trained here (a few seconds each), not downloaded: the point is a small,
    real-data, trained model on which the *certified* estimator is feasible (64 input pixels).
    Returns ``(models, calib, calib_y, test, test_y)``; models are ONNX ModelProtos.
    """
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
        with torch.no_grad():
            acc = float(
                (net(torch.from_numpy(x[te])).argmax(1).numpy() == y[te]).mean()
            )
        log(
            f"digits MLP seed {seed0 + k}: {depth} layers x {width}, held-out accuracy {acc:.3f}"
        )
        models.append(mdl)
    return models, {"x": x[ca]}, y[ca], {"x": x[te]}, y[te]


def main_digits(args):
    """Several trained real-data MLPs: per-net rank agreement, aggregated (the certified method fits here)."""
    models, calib, calib_y, ev, ev_y = train_digits_mlps(
        args.nets, args.depth, args.width, args.seed
    )
    box = {"x": (0.0, 1.0)}
    per_net = []
    methods = (
        [
            "weight_err",
            "weight_err_rel",
            "fisher",
            "taylor",
            "hessian_trace",
            "certified",
        ]
        if args.kind == "weights"
        else ["act_scale", "fisher", "taylor", "certified"]
    )
    for k, model in enumerate(models):
        sites = qs.find_sites(model)
        groups = qs.as_groups(sites)
        names = list(groups)
        ref = qs.run_logits(model, ev)
        quants = None
        if args.kind == "activations":
            quants = (
                qs.calibrate_activations_interval(model, sites, box, args.bits)
                if args.act_calib == "interval"
                else qs.calibrate_activations(model, sites, calib, args.bits)
            )
        gt = qs.measure(model, groups, args.kind, args.bits, ev, ev_y, quants, ref=ref)
        kl = np.array([gt[g]["kl"] for g in names])
        rep = qs.rank(model, sites, None, calib, args.kind, args.bits, methods, labels=calib_y, act_bits=args.bits,
                      input_ranges=box, n_samples=args.est_samples, quants=quants, seed=args.seed)  # fmt: skip
        row = {
            "net": k,
            "kl": kl.tolist(),
            "skipped": rep.skipped,
            "float_accuracy": float((ref.argmax(1) == ev_y).mean()),
        }
        for m, sc in rep.scores.items():
            row[m] = {"scores": sc.tolist(), "spearman": qs.spearman(sc, kl), "kendall": qs.kendall(sc, kl),
                      "top1_hit": bool(np.argmax(sc) == np.argmax(kl))}  # fmt: skip
        per_net.append(row)
        log(
            f"net {k}: "
            + "  ".join(f"{m}: rho {row[m]['spearman']:+.2f}" for m in rep.scores)
        )
    agg = {}
    for m in per_net[0].keys() - {"net", "kl", "skipped", "float_accuracy"}:
        rhos = np.array([r[m]["spearman"] for r in per_net])
        agg[m] = {
            "mean_spearman": float(np.nanmean(rhos)), "median_spearman": float(np.nanmedian(rhos)),
            "sem": float(np.nanstd(rhos) / np.sqrt(np.sum(~np.isnan(rhos)))),
            "top1_hit_rate": float(np.mean([r[m]["top1_hit"] for r in per_net])),
        }  # fmt: skip
    chance = 1.0 / (args.depth)
    log(
        f"== digits, {args.nets} trained nets x {args.depth} layers, {args.kind} {args.bits}-bit; chance top-1 = {chance:.0%}"
    )
    for m, a in agg.items():
        log(
            f"  {m:16s} mean rho {a['mean_spearman']:+.3f} +- {a['sem']:.3f} (SEM)  median {a['median_spearman']:+.3f}  top-1 hit {a['top1_hit_rate']:.0%}"
        )
    out = args.out or os.path.join(
        args.workdir, f"result_digits_{args.kind}{args.bits}.json"
    )
    with open(out, "w") as f:
        json.dump({"task": "digits", "kind": args.kind, "bits": args.bits, "nets": args.nets, "depth": args.depth,
                   "width": args.width, "aggregate": agg, "per_net": per_net, "argv": sys.argv}, f, indent=1)  # fmt: skip
    log("wrote", out)


def main_stability(args):
    """How stable are fisher/taylor under the choice (and number) of calibration samples?

    Ground truth is read from the matching full run's JSON; only the estimators are recomputed,
    on ``--stab-subsets`` random calibration subsets of each size.
    """
    res_path = os.path.join(
        args.workdir, f"result_{args.model}_{args.kind}{args.bits}.json"
    )
    gt = json.load(open(res_path))
    kl = np.array(gt["ground_truth"]["kl"])
    model = onnx.load(os.path.join(args.workdir, f"{args.model}.onnx"))
    calib, calib_y, _, _ = load_cnn_data(args.workdir, args.calib_per_class)
    sites = qs.find_sites(model)
    quants = None
    if args.kind == "activations":
        cal_n = min(512, len(calib_y))
        quants = qs.calibrate_activations(
            model, sites, {k: v[:cal_n] for k, v in calib.items()}, args.bits
        )
    rng = np.random.default_rng(args.seed)
    log(
        f"stability: {args.model} {args.kind} {args.bits}-bit, {len(sites)} sites, pool of {len(calib_y)} calibration images"
    )
    rows = []
    for n in [int(x) for x in args.stab_sizes.split(",")]:
        for mode in ("sample", "argmax"):
            vals: dict = {"fisher": [], "taylor": []}
            for r in range(args.stab_subsets):
                idx = rng.choice(len(calib_y), n, replace=False)
                sub = {k: v[idx] for k, v in calib.items()}
                rep = qs.rank(model, sites, None, sub, args.kind, args.bits, ["fisher", "taylor"], labels=calib_y[idx],
                              act_bits=args.bits, n_samples=n, label_mode=mode, quants=quants, seed=args.seed + r)  # fmt: skip
                for m in vals:
                    vals[m].append(qs.spearman(rep.scores[m], kl))
            for m, v in vals.items():
                rows.append(
                    {
                        "n": n,
                        "label_mode": mode,
                        "method": m,
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v)),
                        "min": float(np.min(v)),
                    }
                )
                log(
                    f"n={n:4d} y~{mode:6s} {m:7s} Spearman vs KL: mean {np.mean(v):+.3f} std {np.std(v):.3f} min {np.min(v):+.3f}"
                )
    out = args.out or os.path.join(
        args.workdir, f"stability_{args.model}_{args.kind}{args.bits}.json"
    )
    json.dump(
        {"model": args.model, "kind": args.kind, "bits": args.bits, "rows": rows},
        open(out, "w"),
        indent=1,
    )
    log("wrote", out)


def main_brute(args):
    """The like-for-like baseline: brute force on the SAME few calibration samples the estimators use.

    The full benchmark measures ground truth on the whole evaluation split, which is far more data
    than the gradient estimators see; comparing their cost to that would flatter them. Here every
    group is quantized one at a time and measured on only ``--brute-sizes`` calibration samples, and
    the resulting ranking is compared with the full-split ground truth read from the matching run.
    """
    is_bert = args.model == "distilbert_sst2"
    res_path = os.path.join(
        args.workdir, f"result_{args.model}_{args.kind}{args.bits}.json"
    )
    gt = json.load(open(res_path))
    kl = np.array(gt["ground_truth"]["kl"])
    model = onnx.load(os.path.join(args.workdir, f"{args.model}.onnx"))
    if is_bert:
        calib, calib_y, ev, ev_y = load_bert_data(args.workdir, calib_n=256)
    else:
        calib, calib_y, ev, ev_y = load_cnn_data(args.workdir, args.calib_per_class)
    sites = qs.find_sites(model)
    groups = qs.as_groups(sites, bert_groups(sites) if is_bert else None)
    quants = None
    if args.kind == "activations":
        cal_n = min(512, len(calib_y))
        quants = qs.calibrate_activations(
            model, sites, {k: v[:cal_n] for k, v in calib.items()}, args.bits
        )
    rng = np.random.default_rng(args.seed)
    rows = []
    first_kl: dict = {}
    for n in [int(x) for x in args.brute_sizes.split(",")]:
        vals, secs = [], []
        for _ in range(args.stab_subsets):
            idx = rng.choice(len(calib_y), n, replace=False)
            sub = {k: v[idx] for k, v in calib.items()}
            t0 = time.time()
            ref = qs.run_logits(model, sub)
            m = qs.measure(
                model, groups, args.kind, args.bits, sub, None, quants, ref=ref
            )
            secs.append(time.time() - t0)
            if n not in first_kl:
                first_kl[n] = [m[g]["kl"] for g in groups]
            vals.append(qs.spearman([m[g]["kl"] for g in groups], kl))
        rows.append(
            {
                "n": n,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "seconds": float(np.mean(secs)),
            }
        )
        log(
            f"brute force on {n:4d} calibration samples: Spearman vs full-split ground truth mean {np.mean(vals):+.3f} std {np.std(vals):.3f} min {np.min(vals):+.3f}; {np.mean(secs):.1f}s per pass over all {len(groups)} groups"
        )
    # decision test: keep the k groups this cheap brute-force ranking calls most sensitive in
    # float, quantize the rest, measure on the evaluation split (same protocol as the full run)
    selection: dict = {}
    sn = args.select_n
    if sn in first_kl:
        names = list(groups)
        order = [names[i] for i in np.argsort(-np.asarray(first_kl[sn]))]
        ref_ev = qs.run_logits(model, ev)
        for k in sorted(gt["selection"]["k"], key=int):
            r = qs.evaluate_selection(
                model,
                groups,
                order[: int(k)],
                args.kind,
                args.bits,
                ev,
                ev_y,
                quants,
                ref=ref_ev,
            )
            selection[k] = {"kl": r["kl"], "accuracy": r.get("accuracy")}
            log(f"brute force (n={sn}) ranking, keep k={k} in float: KL {r['kl']:.4f}")
    out = args.out or os.path.join(
        args.workdir, f"brute_{args.model}_{args.kind}{args.bits}.json"
    )
    json.dump(
        {
            "model": args.model, "kind": args.kind, "bits": args.bits, "rows": rows,
            "select_n": sn, "selection": selection,
        },
        open(out, "w"),
        indent=1,
    )  # fmt: skip
    log("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", choices=["cnn", "bert", "digits", "stability", "brute"])
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--workdir", default="/mnt/data/cache/claude-work/qs-work")
    ap.add_argument("--kind", choices=["weights", "activations"], default="weights")
    ap.add_argument(
        "--bits",
        type=int,
        default=4,
        help="weight bits, or activation bits for --kind activations",
    )
    ap.add_argument("--calib-per-class", type=int, default=2)
    ap.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="use only the first N evaluation samples (pilot runs)",
    )
    ap.add_argument(
        "--est-samples",
        type=int,
        default=128,
        help="calibration samples the gradient estimators use",
    )
    ap.add_argument("--methods", default="")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--random-draws", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--select-n",
        type=int,
        default=64,
        help="brute: calibration samples whose ranking drives the decision test",
    )
    ap.add_argument(
        "--brute-sizes",
        default="16,64,256",
        help="brute: calibration samples to measure on",
    )
    ap.add_argument(
        "--stab-sizes",
        default="8,16,32,64,128,256",
        help="stability: calibration subset sizes",
    )
    ap.add_argument(
        "--stab-subsets", type=int, default=5, help="stability: random subsets per size"
    )
    ap.add_argument(
        "--act-calib",
        choices=["data", "interval"],
        default="data",
        help="digits: activation quantizer range from calibration data (percentile) or from "
        "interval analysis (guaranteed never to clip inside the input box)",
    )
    ap.add_argument(
        "--nets",
        type=int,
        default=10,
        help="digits: number of independently trained MLPs",
    )
    ap.add_argument("--depth", type=int, default=8, help="digits: layers per MLP")
    ap.add_argument("--width", type=int, default=32, help="digits: hidden width")
    args = ap.parse_args()

    if args.task == "digits":
        return main_digits(args)
    if args.task == "stability":
        return main_stability(args)
    if args.task == "brute":
        return main_brute(args)
    t_start = time.time()
    if args.task == "bert":
        args.model = (
            "distilbert_sst2"  # results are named after the model, not the CLI default
        )
    path = args.onnx or os.path.join(args.workdir, f"{args.model}.onnx")
    model = onnx.load(path)
    if args.task == "cnn":
        calib, calib_y, ev, ev_y = load_cnn_data(args.workdir, args.calib_per_class)
        groups_spec = None
    else:
        calib, calib_y, ev, ev_y = load_bert_data(args.workdir, calib_n=256)
        groups_spec = None
    if args.eval_limit:
        ev = {k: v[: args.eval_limit] for k, v in ev.items()}
        ev_y = ev_y[: args.eval_limit]
    sites = qs.find_sites(model)
    if args.task == "bert":
        groups_spec = bert_groups(sites)
    groups = qs.as_groups(sites, groups_spec)
    names = list(groups)
    log(
        f"{args.task}/{args.model}: {len(sites)} sites, {len(groups)} groups; calib {len(calib_y)} eval {len(ev_y)}"
    )

    # ---- float reference
    ref = qs.run_logits(model, ev)
    acc_float = float((ref.argmax(1) == ev_y).mean())
    log(f"float top-1 accuracy on the evaluation split: {acc_float:.4f}")

    quants = None
    if args.kind == "activations":
        cal_n = min(512, len(calib_y))
        quants = qs.calibrate_activations(
            model, sites, {k: v[:cal_n] for k, v in calib.items()}, args.bits
        )

    # ---- ground truth
    t0 = time.time()
    gt = qs.measure(model, groups, args.kind, args.bits, ev, ev_y, quants, ref=ref)
    t_gt = time.time() - t0
    kl = np.array([gt[g]["kl"] for g in names])
    acc = np.array([gt[g]["accuracy"] for g in names])
    agree = np.array([gt[g]["agreement"] for g in names])
    kl_samples = np.stack([gt[g]["kl_samples"] for g in names])
    log(
        f"ground truth: {len(names)} groups in {t_gt:.0f}s; KL range {kl.min():.3g}..{kl.max():.3g}; "
        f"accuracy range {acc.min():.4f}..{acc.max():.4f}"
    )

    # ---- estimators
    if args.methods:
        methods = args.methods.split(",")
    elif args.kind == "weights":
        methods = ["weight_err", "weight_err_rel", "fisher", "taylor", "hessian_trace"]
    else:
        methods = ["act_scale", "fisher", "taylor"]
    calib_sub = {k: v for k, v in calib.items()}
    rep = qs.rank(
        model, sites, groups_spec, calib_sub, args.kind, args.bits, methods, labels=calib_y,
        act_bits=args.bits, n_samples=args.est_samples, quants=quants, seed=args.seed,
    )  # fmt: skip
    log("estimator wall-clock (s):", {k: round(v, 1) for k, v in rep.seconds.items()})
    for k, v in rep.skipped.items():
        log("skipped", k, v)

    # ---- rank correlation
    rows = []
    for m, s in rep.scores.items():
        row = {"method": m}
        for tag, truth in (("kl", kl), ("acc_drop", acc_float - acc)):
            row[f"spearman_{tag}"] = qs.spearman(s, truth)
            row[f"kendall_{tag}"] = qs.kendall(s, truth)
        lo, hi = bootstrap_corr(
            s, kl_samples, names, qs.spearman, args.bootstrap, args.seed
        )
        row["spearman_kl_ci95"] = [lo, hi]
        row["top1_hit"] = bool(int(np.argmax(s)) == int(np.argmax(kl)))
        top3 = set(np.argsort(s)[-3:].tolist())
        row["top3_overlap"] = len(top3 & set(np.argsort(kl)[-3:].tolist()))
        rows.append(row)
    # the noise floor: how well does the ground truth agree with itself when re-measured on a
    # resampled half? (split-half rank agreement, averaged)
    rng = np.random.default_rng(args.seed)
    half = []
    for _ in range(args.bootstrap):
        p = rng.permutation(kl_samples.shape[1])
        a, b = p[: len(p) // 2], p[len(p) // 2 :]
        half.append(qs.spearman(kl_samples[:, a].mean(1), kl_samples[:, b].mean(1)))
    noise_floor = float(np.nanmean(half))
    log(
        f"ground-truth split-half Spearman (the ceiling any estimator can reach): {noise_floor:.3f}"
    )
    log(
        f"{'method':16s} {'rho(KL)':>8s} {'95% CI':>16s} {'tau(KL)':>8s} {'rho(accdrop)':>12s} {'top1':>5s} {'top3':>5s}"
    )
    for r in rows:
        ci = r["spearman_kl_ci95"]
        log(
            f"{r['method']:16s} {r['spearman_kl']:+8.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] {r['kendall_kl']:+8.3f} "
            f"{r['spearman_acc_drop']:+12.3f} {str(r['top1_hit']):>5s} {r['top3_overlap']:>3d}/3"
        )

    # ---- budgeted selection
    ks = [int(k) for k in args.ks.split(",") if int(k) <= len(names)]
    rng = np.random.default_rng(args.seed)
    all_q = qs.evaluate_selection(
        model, groups, [], args.kind, args.bits, ev, ev_y, quants, ref=ref
    )
    sel = {"all_quantized": all_q, "k": {}}
    log(
        f"all groups quantized: KL {all_q['kl']:.4f} acc {all_q.get('accuracy', float('nan')):.4f} (float {acc_float:.4f})"
    )
    strategies = {m: qs.select_float_sites(rep, max(ks), m) for m in rep.scores}
    oracle_order = [names[i] for i in np.argsort(-kl)]
    for k in ks:
        sel["k"][k] = {}
        picks = {m: qs.select_float_sites(rep, k, m) for m in rep.scores}
        picks["oracle(GT)"] = oracle_order[:k]
        for m, keep in picks.items():
            r = qs.evaluate_selection(
                model, groups, keep, args.kind, args.bits, ev, ev_y, quants, ref=ref
            )
            sel["k"][k][m] = {
                "kl": r["kl"],
                "accuracy": r.get("accuracy"),
                "keep": keep,
            }
        rk, ra = [], []
        for _ in range(args.random_draws):
            keep = [names[i] for i in rng.choice(len(names), k, replace=False)]
            r = qs.evaluate_selection(
                model, groups, keep, args.kind, args.bits, ev, ev_y, quants, ref=ref
            )
            rk.append(r["kl"])
            ra.append(r.get("accuracy", float("nan")))
        sel["k"][k]["random(mean)"] = {
            "kl": float(np.mean(rk)),
            "accuracy": float(np.mean(ra)),
        }
        log(
            f"k={k}: "
            + "  ".join(f"{m}: KL {v['kl']:.4f}" for m, v in sel["k"][k].items())
        )
    del strategies

    result = {
        "task": args.task, "model": args.model, "kind": args.kind, "bits": args.bits,
        "n_calib": int(len(calib_y)), "n_eval": int(len(ev_y)), "n_groups": len(names),
        "float_accuracy": acc_float, "groups": names,
        "ground_truth": {"kl": kl.tolist(), "accuracy": acc.tolist(), "agreement": agree.tolist()},
        "scores": {m: s.tolist() for m, s in rep.scores.items()}, "skipped": rep.skipped,
        "seconds": rep.seconds, "ground_truth_seconds": t_gt, "correlations": rows,
        "ground_truth_split_half_spearman": noise_floor, "selection": sel,
        "total_seconds": time.time() - t_start, "argv": sys.argv,
    }  # fmt: skip
    out = args.out or os.path.join(
        args.workdir, f"result_{args.model}_{args.kind}{args.bits}.json"
    )
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    log("wrote", out, f"(total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
