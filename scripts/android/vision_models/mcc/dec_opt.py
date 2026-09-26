"""MCC decoder chunk variants for the phone: build, quantize, score end to end on the HTP.

  dec_opt.py prep  [--app-dump <dir>]           K/V caches, dense fp32 references, query sets
  dec_opt.py quant --policy P [--calib quest2]  dec_q1024.<P>.onnx (onnxsim.full_qdq)
  dec_opt.py eval  <model.onnx> [--sets spyro quest2m]   phone coarse-to-fine recon per set vs its ref
  dec_opt.py quant-enc --policy P               enc.<P>.onnx, calibrated on quest2 + spyro
  dec_opt.py eval-enc <enc.onnx> [--dec <dec.onnx>]      phone encoder on quest2m -> K/V -> phone recon

Sets (each a K/V cache + the dense granularity-0.1 fp32 grid of the host split):
  quest2   upstream demo, iPhone points (the calibration set)
  spyro    upstream demo, iPhone points (a different object: held out)
  quest2m  the demo app's own run on quest2.jpg: SAM mask + MoGe-2 points (dump=1; held out)
The query set of a set is exactly what mcc.py's coarse-to-fine issues on it (the host fp32 grid
decides it; calibration samples chunks of it, so calibration sees the surface-heavy distribution
the phone queries, not the mostly-empty dense grid).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mcc as C  # noqa: E402
import model as M  # noqa: E402
import queries as Qs  # noqa: E402

WORK = C.WORK
CKPT = str(Path.home() / ".cache/onnxsim-mcc/co3dv2_all_categories.pth")
Q = 1024


def kv_path(s):
    return WORK / f"kv_{s}.npz"


def ref_path(s):
    return WORK / f"ref_{s}_0.1.npz"


def query_levels(p_dense, levels=2, lo=0.05):
    """The coarse-to-fine query points (xyz, per level) mcc.py recon issues when every queried
    point gets the dense grid's p (exact: a query depends only on its own xyz)."""
    nt = p_dense.shape[0]
    out, q_prev = [], None
    for li, n in enumerate([nt >> s for s in range(levels, -1, -1)]):
        p = p_dense[:: nt // n, :: nt // n, :: nt // n]
        if li == 0:
            want, have = np.ones((n,) * 3, bool), np.zeros((n,) * 3, bool)
        else:
            have = Qs.embed(q_prev)
            want = have | Qs.refine(np.where(q_prev, p[::2, ::2, ::2], 0) > lo, n)
        out.append(C.grid_xyz(np.argwhere(want & ~have), n))
        q_prev = want
    return out


def cmd_prep(a):
    torch.set_grad_enabled(False)
    m = M.load_mcc(CKPT)
    enc, dec = M.Encoder(m).eval(), M.QueryDecoder(m).eval()
    repo = os.environ.get("MCC_REPO", os.path.expanduser("~/.cache/onnxsim-mcc/MCC"))
    kvs = {"quest2": np.load(WORK / "kv_quest2.npz")}
    img, xyz = M.load_demo(repo, "spyro")
    k, v = enc(img, *M.xyz_windows(xyz))
    kvs["spyro"] = {"k": k.numpy(), "v": v.numpy()}
    if a.app_dump:  # the app's own encoder inputs, host fp32 encoder (the phone's K/V differ by fp16)
        d = Path(a.app_dump)

        def rd(n, shape):
            return torch.from_numpy(np.fromfile(d / n, np.float32).reshape(shape))

        k, v = enc(rd("img.f32", (1, 3, 224, 224)), rd("xyz_win.f32", (196, 64, 3)), rd("valid.f32", (196, 64)))
        kvs["quest2m"] = {"k": k.numpy(), "v": v.numpy()}
        np.savez(
            WORK / "inputs_quest2m.npz",
            **{n: rd(f"{n}.f32", s).numpy() for n, s in (("img", (1, 3, 224, 224)), ("xyz_win", (196, 64, 3)), ("valid", (196, 64)))},
        )
    win, val = M.xyz_windows(xyz)
    np.savez(WORK / "inputs_spyro.npz", img=img.numpy(), xyz_win=win.numpy(), valid=val.numpy())
    g = M.grid(0.1)
    for s, kv in kvs.items():
        if s != "quest2":
            np.savez(kv_path(s), k=kv["k"], v=kv["v"])
        if not ref_path(s).exists():
            t = time.time()
            k, v = torch.from_numpy(kv["k"]), torch.from_numpy(kv["v"])
            occ, rgb = zip(*(dec(g[:, i : i + 4096], k, v) for i in range(0, g.shape[1], 4096)))
            np.savez(ref_path(s), xyz=g[0].numpy(), occ=torch.cat(occ, 1)[0].numpy(), rgb=torch.cat(rgb, 1)[0].numpy())
            print(f"{s}: dense ref in {time.time() - t:.0f} s")
        ref = np.load(ref_path(s))
        p = (1 / (1 + np.exp(-ref["occ"].astype(np.float64)))).reshape((60,) * 3)
        lv = query_levels(p)
        np.savez(WORK / f"queries_{s}.npz", *lv)
        print(f"{s}: {[len(x) for x in lv]} queries per level, {(p > 0.3).sum()} occupied")


POLICIES = {  # tag -> quantize_full_qdq kwargs
    "sm16": {"op_types": ["Softmax"], "activation_dtype": "uint16"},
    "smgelu16": {"op_types": ["Softmax", "Gelu"], "activation_dtype": "uint16"},
    "a16": {"activation_dtype": "uint16"},
    "a16c": {"activation_dtype": "uint16", "float_color": True},
    "a8": {},
    "a8c": {"float_color": True},
    # uint8 with uint16 islands (tensor groups, see groups16): x = query xyz embedding, a = attention
    # scores / softmax / K,V, r = residual stream (every LayerNorm input), o = the output head (final
    # LayerNorm -> prediction linear -> occ), m = MLP hidden (fc1 output, Gelu output)
    "a8c_x16": {"float_color": True, "groups16": "x"},
    "a8c_xa16": {"float_color": True, "groups16": "xa"},
    "a8c_xr16": {"float_color": True, "groups16": "xr"},
    "a8c_xar16": {"float_color": True, "groups16": "xar"},
    "a8c_o16": {"float_color": True, "groups16": "o"},
    "a8c_om16": {"float_color": True, "groups16": "om"},
    "a16c_m8": {"activation_dtype": "uint16", "float_color": True, "groups8": "m"},  # uint16, MLP hidden uint8
}


def groups16(m, which):
    """Activation tensors to keep at uint16 in a uint8 graph, by group letter."""
    ns = m.graph.node
    out = set()
    if "x" in which:  # the query xyz through shrink + the positional-embedding linear
        out.add("xyz")
        for n in ns:
            if n.op_type == "LayerNormalization":
                out.add(n.input[0])
                break
            out.update(n.output)
    if "a" in which:
        out.update(["k", "v"])
        color = set(color_tail(m))
        for n in ns:
            if n.op_type == "Softmax" and n.name not in color:
                out.update([n.input[0], n.output[0]])
            if n.op_type == "Concat":
                out.update(n.input)
    if "r" in which:
        out.update(n.input[0] for n in ns if n.op_type == "LayerNormalization")
    if "o" in which:  # the last LayerNorm's output to the occ output (the occupancy logit shares the
        # prediction linear's range with 768 color logits)
        ln = [n for n in ns if n.op_type == "LayerNormalization"][-1]
        prod = {o: n for n in ns for o in n.output}
        t = "occ"
        while t != ln.output[0]:
            out.add(t)
            t = prod[t].input[0]
        out.add(t)
    if "m" in which:
        for n in ns:
            if n.op_type == "Gelu":
                out.update([n.input[0], n.output[0]])
    return out


def color_tail(m):
    """The color head after the shared prediction layer: rgb <- ReduceSum <- Mul(levels) <- Softmax
    <- Div(temperature 0.1) <- Reshape <- Slice. uint16 there wrecks the colors (L1 ~90/255): the
    temperature-0.1 softmax over 256 levels is nearly one-hot and needs float."""
    prod = {o: n for n in m.graph.node for o in n.output}
    names, t = [], "rgb"
    while True:
        n = prod[t]
        names.append(n.name)
        if n.op_type == "Slice":
            return names
        t = n.input[0]


def calib_data(sets, per_set, seed=0, q=Q):
    rng = np.random.default_rng(seed)
    data = []
    for s in sets:
        kv = np.load(kv_path(s))
        qs = np.concatenate(list(np.load(WORK / f"queries_{s}.npz").values()))
        for _ in range(per_set):
            x = qs[rng.choice(len(qs), q, replace=len(qs) < q)][None].astype(np.float32)
            data.append({"xyz": x, "k": kv["k"], "v": kv["v"]})
    return data


def cmd_quant(a):
    import onnx

    from onnxsim.full_qdq import quantize_full_qdq

    t = time.time()
    kw = dict(POLICIES[a.policy], method=a.method)
    m = onnx.load(WORK / f"dec_q{a.chunk}.onnx")
    if kw.pop("float_color", False):
        kw["exclude_nodes"] = color_tail(m)
    if "groups16" in kw:
        kw["tensor_dtypes"] = {t: "uint16" for t in groups16(m, kw.pop("groups16"))}
        print(f"{len(kw['tensor_dtypes'])} tensors at uint16")
    if "groups8" in kw:  # the same groups, overridden down to uint8 in a uint16 graph
        kw["tensor_dtypes"] = {t: "uint8" for t in groups16(m, kw.pop("groups8"))}
        print(f"{len(kw['tensor_dtypes'])} tensors at uint8")
    if a.pin_kv:  # K/V ranges pinned with margin: |K| <= 12.2, |V| <= 5.9 on every set so far
        kw["ranges"] = {"k": (-16.0, 16.0), "v": (-8.0, 8.0)}
    q = quantize_full_qdq(m, calib_data(a.calib, a.per_set, q=a.chunk), **kw)
    tag = a.policy + ("" if a.method == "minmax" else "_" + a.method) + ("_pin" if a.pin_kv else "") + ("_qio" if a.qio else "")
    out = WORK / f"dec_q{a.chunk}.{tag}.onnx"
    if a.qio:  # K/V become the quantized graph inputs: the app quantizes them once per image
        from onnxsim.full_qdq import quantized_io

        q, io = quantized_io(q, inputs=["k", "v"], outputs=[])
        out.with_suffix(".json").write_text(json.dumps(io, indent=1, default=str))
        print(io)
    onnx.save(q, out)
    print(f"{out.name}: {sum(n.op_type == 'QuantizeLinear' for n in q.graph.node)} Q nodes, {time.time() - t:.0f} s")


def quant(x, io):
    q = np.round(x / float(io["scale"])) + int(io["zero_point"])
    return np.clip(q, 0, 65535).astype(np.uint16)


def cmd_eval(a):
    model = Path(a.model)
    io = json.loads(model.with_suffix(".json").read_text()) if model.with_suffix(".json").exists() else {}
    for s in a.sets:
        kv = np.load(kv_path(s))
        k, v, dt = kv["k"], kv["v"], "f32"
        if io:
            k, v, dt = quant(k, io["k"]), quant(v, io["v"]), "u16"
        t = time.time()
        p, rgb, n_q, ms = C.phone_recon(
            k, v, model, model.stem, WORK / "phone" / "opt" / s, iters=a.iters, lo=a.lo, kv_dtype=dt, chunk=a.chunk
        )
        rec, prec, ch, col, n, nr = C.score(p, rgb, np.load(ref_path(s)))
        n_chunks = int(np.ceil(n_q / a.chunk))
        print(
            f"{model.name} {s}: {ms} ms/chunk x {n_chunks} = {ms * n_chunks / 1e3:.2f} s decoder; "
            f"recall {rec:.4f} precision {prec:.4f} chamfer {ch:.4f} color L1 {col:.2f}/255 "
            f"({n} vs {nr} points) [{time.time() - t:.0f} s]"
        )


ENC_IN = ("img", "xyz_win", "valid")


def cmd_quant_enc(a):
    import onnx

    from onnxsim.full_qdq import quantize_full_qdq

    t = time.time()
    data = [{n: np.load(WORK / f"inputs_{s}.npz")[n] for n in ENC_IN} for s in a.calib]
    q = quantize_full_qdq(onnx.load(WORK / "enc.onnx"), data, **dict(POLICIES[a.policy]))
    out = WORK / f"enc.{a.policy}.onnx"
    onnx.save(q, out)  # int8 weights: ~200 MB, one file (phone.sh pushes one file)
    print(f"{out.name}: {sum(n.op_type == 'QuantizeLinear' for n in q.graph.node)} Q nodes, {time.time() - t:.0f} s")


def cmd_eval_enc(a):
    model, s = Path(a.model), a.set
    inp = np.load(WORK / f"inputs_{s}.npz")
    ms, (k, v) = C.phone(model, model.stem, [(n, inp[n], "f32") for n in ENC_IN], a.iters, WORK / "phone" / "enc_opt" / model.stem)
    k, v = k.reshape(8, 16, 197, 32), v.reshape(8, 16, 197, 32)
    ref = np.load(kv_path(s))
    print(f"{model.name} {s}: {ms} ms; K cos {C.cos(k, ref['k']):.6f} V cos {C.cos(v, ref['v']):.6f}")
    dec = Path(a.dec)
    p, rgb, n_q, dms = C.phone_recon(k, v, dec, dec.stem, WORK / "phone" / "enc_opt" / "recon", iters=2, lo=a.lo)
    rec, prec, ch, col, n, nr = C.score(p, rgb, np.load(ref_path(s)))
    print(f"  + {dec.name}: recall {rec:.4f} precision {prec:.4f} chamfer {ch:.4f} color L1 {col:.2f}/255 ({n} vs {nr} points)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--app-dump")
    q = sub.add_parser("quant")
    q.add_argument("--policy", required=True, choices=list(POLICIES))
    q.add_argument("--method", default="minmax")
    q.add_argument("--calib", nargs="+", default=["quest2"])
    q.add_argument("--per-set", type=int, default=12)
    q.add_argument("--pin-kv", action="store_true")
    q.add_argument("--chunk", type=int, default=Q)
    q.add_argument("--qio", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("--sets", nargs="+", default=["spyro", "quest2m"])
    e.add_argument("--iters", type=int, default=4)
    e.add_argument("--lo", type=float, default=0.05)
    e.add_argument("--chunk", type=int, default=Q)
    qe = sub.add_parser("quant-enc")
    qe.add_argument("--policy", default="a16", choices=["a16", "a8"])
    qe.add_argument("--calib", nargs="+", default=["quest2", "spyro"])
    ee = sub.add_parser("eval-enc")
    ee.add_argument("model")
    ee.add_argument("--set", default="quest2m")
    ee.add_argument("--dec", default=str(WORK / f"dec_q{Q}.onnx"))
    ee.add_argument("--iters", type=int, default=6)
    ee.add_argument("--lo", type=float, default=0.05)
    a = ap.parse_args()
    {"prep": cmd_prep, "quant": cmd_quant, "eval": cmd_eval, "quant-enc": cmd_quant_enc, "eval-enc": cmd_eval_enc}[
        a.cmd.replace("_", "-")
    ](a)


if __name__ == "__main__":
    main()
