"""MCC on the phone: export, phone runs, scoring.

  mcc.py export  --ckpt <pth>             enc.onnx + dec_q<Q>.onnx (fp32) -> onnxsim; ORT CPU vs torch
  mcc.py phone   <piece> [--iters N]      run one piece strict all-HTP (fp16) under the phone lock
  mcc.py recon   --gran G [--strategy S]  whole reconstruction on the phone -> score vs ref_<g>.npz

Work dir: $MCC_WORK (default ~/.cache/onnxsim-mcc/work). Upstream code: $MCC_REPO.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

WORK = Path(os.environ.get("MCC_WORK", Path.home() / ".cache/onnxsim-mcc/work"))
LOCK = [str(Path.home() / ".cache/android-phone/phone-run")]


def simplify(src, dst):
    import onnx

    import onnxsim

    m, ok = onnxsim.simplify(onnx.load(src), skipped_optimizers=["fuse_attention"])
    assert ok, src
    odd = {
        (x.domain, x.op_type) for x in m.graph.node if x.domain not in ("", "ai.onnx")
    }
    assert not odd, f"{src}: non-standard ops after onnxsim: {odd}"
    onnx.save(m, dst)
    Path(src).unlink()


def ort_run(path, feeds):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return s.run(None, {k: np.ascontiguousarray(v) for k, v in feeds.items()})


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def export(a):
    torch.set_grad_enabled(False)
    WORK.mkdir(parents=True, exist_ok=True)
    m = M.load_mcc(a.ckpt)
    inp = np.load(WORK / "inputs_quest2.npz")
    img, win, val = (torch.from_numpy(inp[k]) for k in ("img", "xyz_win", "valid"))
    enc = M.Encoder(m).eval()
    k, v = enc(img, win, val)
    torch.onnx.export(
        enc,
        (img, win, val),
        WORK / "enc_g.onnx",
        input_names=["img", "xyz_win", "valid"],
        output_names=["k", "v"],
        opset_version=20,
        dynamo=False,
    )
    simplify(WORK / "enc_g.onnx", WORK / "enc.onnx")
    ok, ov = ort_run(
        WORK / "enc.onnx",
        {"img": img.numpy(), "xyz_win": win.numpy(), "valid": val.numpy()},
    )
    print(
        f"enc ORT vs torch: k max {np.abs(ok - k.numpy()).max():.3g}, v max {np.abs(ov - v.numpy()).max():.3g}"
    )
    np.savez(WORK / "kv_quest2.npz", k=k.numpy(), v=v.numpy())
    dec = M.QueryDecoder(m).eval()
    pts = M.grid(0.1)
    for q in a.chunks:
        x = pts[
            :, pts.shape[1] // 2 : pts.shape[1] // 2 + q
        ]  # grid middle: some occupied points
        occ, rgb = dec(x, k, v)
        name = f"dec_q{q}"
        torch.onnx.export(
            dec,
            (x, k, v),
            WORK / f"{name}_g.onnx",
            input_names=["xyz", "k", "v"],
            output_names=["occ", "rgb"],
            opset_version=20,
            dynamo=False,
        )
        simplify(WORK / f"{name}_g.onnx", WORK / f"{name}.onnx")
        oo, orr = ort_run(
            WORK / f"{name}.onnx", {"xyz": x.numpy(), "k": k.numpy(), "v": v.numpy()}
        )
        print(
            f"{name} ORT vs torch: occ max {np.abs(oo - occ.numpy()).max():.3g}, "
            f"rgb max {np.abs(orr - rgb.numpy()).max():.3g}"
        )


def write_raw(d, name, arr, dtype="f32"):
    arr = np.ascontiguousarray(
        arr.astype({"f32": np.float32, "u8": np.uint8, "u16": np.uint16}[dtype])
    )
    arr.tofile(d / f"{name}.bin")
    return f"{name} {dtype} {name}.bin {','.join(map(str, arr.shape))}"


def phone(model_path, name, feeds, iters, loc, mode="htp"):
    """feeds: list of (input name, array, dtype). Returns (median ms, [output arrays])."""
    ms, outs = phone_sets(model_path, name, [feeds], iters, loc, mode)
    return ms, outs[0] if outs else []


def phone_sets(model_path, name, sets, iters, loc, mode="htp"):
    """Run the model once per feed set (the first `iters` times) in one locked phone.sh call.
    Inputs with identical names+contents are written once and md5-skipped on push.
    Returns (median ms of set 0, [[output arrays] per set])."""
    loc.mkdir(parents=True, exist_ok=True)
    mans = []
    for k, feeds in enumerate(sets):
        lines = []
        for n, arr, dt in feeds:
            fname = (
                n if n in ("k", "v") else f"{n}_{k}"
            )  # the K/V cache is shared by every set
            lines.append(
                write_raw(loc, fname, arr, dt).replace(f"{fname} ", f"{n} ", 1)
            )
        (loc / f"m{k}.txt").write_text("\n".join(lines) + "\n")
        mans.append(f"m{k}.txt")
    env = dict(os.environ, PHONE_LOCK_OWNER="codex/android-mcc")
    out = subprocess.run(
        LOCK
        + [str(HERE / "phone.sh"), str(model_path), mode, str(iters), str(loc), name]
        + mans,
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    ms, res, k = None, [], -1
    for line in out.splitlines():
        if line.startswith("=== set"):
            k = int(line.split()[2])
            res.append([])
        if line.startswith("median_ms") and ms is None:
            ms = float(line.split()[1])
        if line.startswith("out ") and k >= 0:
            _, i, _, dt, shape = line.split()[:5]
            dims = [
                int(t) for t in shape.strip("[]()").replace("x", ",").split(",") if t
            ]
            arr = np.fromfile(
                loc / f"out{k}_o{i}.bin",
                dtype={"f32": np.float32, "u8": np.uint8, "u16": np.uint16}[dt],
            )
            res[k].append(
                arr.reshape(dims) if dims and np.prod(dims) == arr.size else arr
            )
    if "PASS" not in out or len(res) != len(sets):
        print(out[-3000:])
    return ms, res


def cmd_phone(a):
    inp = np.load(WORK / "inputs_quest2.npz")
    kv = np.load(WORK / "kv_quest2.npz")
    if a.piece == "enc":
        feeds = [
            ("img", inp["img"], "f32"),
            ("xyz_win", inp["xyz_win"], "f32"),
            ("valid", inp["valid"], "f32"),
        ]
        ref = [kv["k"], kv["v"]]
    else:
        q = int(a.piece.split("q")[1])
        pts = M.grid(0.1)
        x = pts[:, pts.shape[1] // 2 : pts.shape[1] // 2 + q].numpy()
        feeds = [("xyz", x, "f32"), ("k", kv["k"], "f32"), ("v", kv["v"], "f32")]
        m = M.load_mcc(a.ckpt)
        with torch.no_grad():
            o, r = M.QueryDecoder(m)(
                torch.from_numpy(x),
                torch.from_numpy(kv["k"]),
                torch.from_numpy(kv["v"]),
            )
        ref = [o.numpy(), r.numpy()]
    ms, outs = phone(
        WORK / f"{a.piece}.onnx", a.piece, feeds, a.iters, WORK / "phone" / a.piece
    )
    print(
        f"{a.piece}: {ms} ms on the HTP; "
        + ", ".join(
            f"out{i} cos {cos(o, r):.6f} max {np.abs(o.reshape(r.shape) - r).max():.3g}"
            for i, (o, r) in enumerate(zip(outs, ref))
        )
    )


def grid_xyz(idx, n, world=3.0):
    return ((idx - n / 2.0) / ((n / 2.0) / world)).astype(np.float32)


def phone_recon(k, v, dec, name, loc, gran=0.1, levels=2, lo=0.05, chunk=1024, iters=4, kv_dtype="f32"):
    """Adaptive coarse-to-fine decoder chunks on the phone for one K/V cache (already quantized when
    kv_dtype is "u16"). Returns (p, rgb) on the target grid, the number of queries and the per-chunk
    median ms."""
    import queries as Qs

    n_target = int(round(6 / gran))
    levels_n = [n_target >> s for s in range(levels, -1, -1)]  # coarse -> target
    probs, dec_ms, n_q = {}, None, 0
    p_prev = q_prev = None
    for li, n in enumerate(levels_n):
        if li == 0:
            want, have = np.ones((n, n, n), bool), np.zeros((n, n, n), bool)
        else:
            have = Qs.embed(q_prev)
            want = have | Qs.refine(p_prev > lo, n)
        todo = np.argwhere(want & ~have)
        p = np.zeros((n, n, n), np.float32)
        if li:
            p[::2, ::2, ::2] = p_prev
        rgb = np.zeros((n, n, n, 3), np.float32)
        if li:
            rgb[::2, ::2, ::2] = probs[levels_n[li - 1]][1]
        sets, idxs = [], []
        for s0 in range(0, len(todo), chunk):
            idx = todo[s0 : s0 + chunk]
            x = np.zeros((1, chunk, 3), np.float32)
            x[0, : len(idx)] = grid_xyz(idx, n)
            sets.append([("xyz", x, "f32"), ("k", k, kv_dtype), ("v", v, kv_dtype)])
            idxs.append(idx)
        ms, outs = phone_sets(dec, name, sets, iters, loc / f"l{li}")
        dec_ms = dec_ms or ms
        for idx, (o, r) in zip(idxs, outs):
            o, r = o.reshape(-1)[: len(idx)], r.reshape(-1, 3)[: len(idx)]
            p[tuple(idx.T)] = 1 / (1 + np.exp(-o))
            rgb[tuple(idx.T)] = r
        n_q += len(todo)
        probs[n] = (p, rgb)
        p_prev, q_prev = p, want
        print(f"level n={n}: {len(todo)} queries in {len(sets)} chunks")
    return probs[n_target][0], probs[n_target][1], n_q, dec_ms


def score(p, rgb, ref, thr=0.3):
    """Reconstruction (p, rgb on the target grid) vs a dense fp32 reference npz (occ logits, rgb):
    (recall, precision, chamfer, color L1 x255, points, reference points)."""
    import queries as Qs

    n = p.shape[0]
    rp = 1 / (1 + np.exp(-ref["occ"].astype(np.float64))).reshape((n,) * 3)
    rrgb = ref["rgb"].reshape((n,) * 3 + (3,))
    occ_ref, occ = rp > thr, p > thr
    both = occ_ref & occ
    rec, prec = both.sum() / occ_ref.sum(), both.sum() / max(occ.sum(), 1)
    ch = Qs.chamfer(Qs.coords(occ, n), Qs.coords(occ_ref, n))
    col = np.abs(rgb[both] - rrgb[both]).mean() * 255
    return rec, prec, ch, col, int(occ.sum()), int(occ_ref.sum())


def cmd_recon(a):
    """Encoder + adaptive coarse-to-fine decoder chunks on the phone, scored vs the dense ref."""
    inp = np.load(WORK / "inputs_quest2.npz")
    loc = WORK / "phone" / "recon"
    enc_ms, (k, v) = phone(
        WORK / "enc.onnx",
        "enc",
        [
            ("img", inp["img"], "f32"),
            ("xyz_win", inp["xyz_win"], "f32"),
            ("valid", inp["valid"], "f32"),
        ],
        a.iters,
        loc / "enc",
    )
    k, v = k.reshape(8, 16, 197, 32), v.reshape(8, 16, 197, 32)
    p, rgb, n_q, dec_ms = phone_recon(
        k, v, WORK / f"dec_q{a.chunk}.onnx", f"dec_q{a.chunk}", loc, a.gran, a.levels, a.lo, a.chunk, a.iters
    )
    rec, prec, ch, col, n_occ, n_ref = score(p, rgb, np.load(WORK / f"ref_quest2_{a.gran}.npz"), a.thr)
    t = enc_ms / 1e3 + n_q / a.chunk * dec_ms / 1e3
    print(
        f"recon g={a.gran} levels={a.levels} lo={a.lo}: {n_q} queries; enc {enc_ms} ms + "
        f"{int(np.ceil(n_q / a.chunk))} x {dec_ms} ms chunks -> {t:.2f} s (steady-state); "
        f"vs dense host fp32: recall {rec:.4f} precision {prec:.4f} chamfer {ch:.4f} "
        f"color L1 {col:.2f}/255 ({n_occ} vs {n_ref} points)"
    )
    np.savez(WORK / f"recon_{a.gran}_{a.levels}_{a.lo}.npz", p=p, rgb=rgb)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--chunks", type=int, nargs="+", default=[2048, 4096, 8192])
    p = sub.add_parser("phone")
    p.add_argument("piece")
    p.add_argument(
        "--ckpt",
        default=str(Path.home() / ".cache/onnxsim-mcc/co3dv2_all_categories.pth"),
    )
    p.add_argument("--iters", type=int, default=6)
    r = sub.add_parser("recon")
    r.add_argument("--gran", type=float, default=0.1)
    r.add_argument("--levels", type=int, default=2)
    r.add_argument("--lo", type=float, default=0.05)
    r.add_argument("--thr", type=float, default=0.3)
    r.add_argument("--chunk", type=int, default=1024)
    r.add_argument("--iters", type=int, default=4)
    a = ap.parse_args()
    t = time.time()
    {"export": export, "phone": cmd_phone, "recon": cmd_recon}[a.cmd](a)
    print(f"[{time.time() - t:.0f} s]")


if __name__ == "__main__":
    main()
