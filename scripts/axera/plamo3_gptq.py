"""GPTQ for PLaMo 3's int4 layer matrices, group-wise (one fp32 scale per group of G input columns and output channel).

The output is what plamo3_onnx.build_block(qsource=...) and export_chunks.py --qdir read: per layer <i>.npz with
<key>_q (int8 codes [out, in], -8..7) and <key>_s (fp32 scales [in / G, out]) for key in qkv, o, gu, d.

  plamo3_gptq.py CHECKPOINT OUT_DIR [--ncal 8192] [--group 128] [--text FILE ...]

Calibration inputs are the fp32 reference's activations on the given text files (tokenized with the model's tokenizer:
tokenizer.jsonl + tokenization_plamo.py from the checkpoint directory), in independent 256-token chunks. Each layer is
quantized on its own fp32 inputs (not sequentially on quantized activations).
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plamo3_reference as R  # noqa: E402

MATS = {
    "qkv": "mixer.qkv_proj.weight",
    "o": "mixer.o_proj.weight",
    "gu": "mlp.gate_up_proj.weight",
    "d": "mlp.down_proj.weight",
}


def layer_capture(m, i, h):
    """Plamo3Reference.layer with the inputs of the four matmuls returned. Returns (out, {key: input [T, in]})."""
    cap = {}
    p = f"model.layers.layers.{i}."
    full = (i + 1) % m.pattern == 0
    T = h.shape[0]
    res = h
    x = m.rms(h, m.w(p + "pre_mixer_norm.weight"), 1.0)
    cap["qkv"] = x
    qkv = x @ m.w(p + "mixer.qkv_proj.weight").T
    q_dim, k_dim = m.nh * m.hd, m.nkv * m.hd
    q = qkv[:, :q_dim].reshape(T, m.nh, m.hd).transpose(1, 0, 2)
    k = qkv[:, q_dim : q_dim + k_dim].reshape(T, m.nkv, m.hd).transpose(1, 0, 2)
    v = qkv[:, q_dim + k_dim :].reshape(T, m.nkv, m.hd).transpose(1, 0, 2)
    q = m.rms(q, m.w(p + "mixer.q_norm.weight"), 1.0)
    k = m.rms(k, m.w(p + "mixer.k_norm.weight"), 1.0)
    pos = np.arange(T)
    q = m.rope(q, pos, full)
    k = m.rope(k, pos, full)
    g = m.nh // m.nkv
    k = np.repeat(k, g, axis=0)
    v = np.repeat(v, g, axis=0)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(m.hd)
    s = s + np.triu(np.full((T, T), -np.inf, np.float32), 1)[None]
    s = s - s.max(-1, keepdims=True)
    pr = np.exp(s)
    pr = pr / pr.sum(-1, keepdims=True)
    att = (pr @ v).transpose(1, 0, 2).reshape(T, q_dim)
    cap["o"] = att
    o = att @ m.w(p + "mixer.o_proj.weight").T
    o = m.rms(o, m.w(p + "post_mixer_norm.weight"), 0.2)
    h = res + o
    res = h
    x = m.rms(h, m.w(p + "pre_mlp_norm.weight"), 1.0)
    cap["gu"] = x
    gu = x @ m.w(p + "mlp.gate_up_proj.weight").T
    a = gu[:, : m.I] / (1 + np.exp(-gu[:, : m.I])) * gu[:, m.I :]
    cap["d"] = a
    mo = a @ m.w(p + "mlp.down_proj.weight").T
    mo = m.rms(mo, m.w(p + "post_mlp_norm.weight"), 0.2**1.5)
    return res + mo, cap


def gptq_int4_group(W, H, G=128, damp=0.01):
    """W [out, in] (reference layout), H [in, in] = sum of X^T X over the inputs. Returns (codes [out, in] int8,
    scales [in / G, out] fp32). A group's scale is set at the group's start from the weights the earlier groups' error
    has already been propagated into (GPTQ's group-wise variant)."""
    W = W.astype(np.float64).copy()
    out, cols = W.shape
    assert cols % G == 0
    H = H.astype(np.float64).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H[np.diag_indices(cols)] += damp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T
    Q = np.zeros_like(W)
    scales = np.zeros((cols // G, out), np.float64)
    for g0 in range(0, cols, G):
        g1 = g0 + G
        s = np.abs(W[:, g0:g1]).max(axis=1) / 7.0
        s[s == 0] = 1.0
        scales[g0 // G] = s
        W1 = W[:, g0:g1]
        Err = np.zeros((out, G))
        Hi = Hinv[g0:g1, g0:g1]
        for i in range(G):
            w = W1[:, i]
            q = np.clip(np.round(w / s), -8, 7) * s
            Q[:, g0 + i] = q
            err = (w - q) / Hi[i, i]
            Err[:, i] = err
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
        W[:, g1:] -= Err @ Hinv[g0:g1, g1:]
    codes = np.zeros((out, cols), np.int8)
    for g0 in range(0, cols, G):
        codes[:, g0 : g0 + G] = np.clip(
            np.round(Q[:, g0 : g0 + G] / scales[g0 // G][:, None]), -8, 7
        )
    return codes, scales.astype(np.float32)


def rtn_int4_group(W, G=128):
    out, cols = W.shape
    Wg = W.reshape(out, cols // G, G)
    s = np.abs(Wg).max(axis=2) / 7.0
    s[s == 0] = 1.0
    codes = (
        np.clip(np.round(Wg / s[:, :, None]), -8, 7).astype(np.int8).reshape(out, cols)
    )
    return codes, s.T.astype(np.float32)


def dequant_group(codes, scales, G=128):
    out, cols = codes.shape
    s = scales.T
    return (
        codes.astype(np.float32).reshape(out, cols // G, G) * s[:, :, None]
    ).reshape(out, cols)


def load_tokenizer(checkpoint):
    spec = importlib.util.spec_from_file_location(
        "tp", os.path.join(checkpoint, "tokenization_plamo.py")
    )
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    return tp.Plamo3Tokenizer(vocab_file=os.path.join(checkpoint, "tokenizer.jsonl"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--ncal", type=int, default=8192)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument(
        "--text", action="append", default=[], help="calibration text file (repeatable)"
    )
    ap.add_argument("--layers", default="0-23")
    a = ap.parse_args()
    if not a.text:
        ap.error("give at least one --text calibration file")
    tok = load_tokenizer(a.checkpoint)
    ids = []
    for f in a.text:
        with open(f, encoding="utf-8") as fh:
            ids += tok.encode(fh.read(), add_special_tokens=False)
        if len(ids) >= a.ncal:
            break
    ids = np.array(ids[: a.ncal], np.int64)
    chunks = [ids[i : i + a.chunk] for i in range(0, len(ids) - a.chunk + 1, a.chunk)]
    m = R.Plamo3Reference.from_checkpoint(a.checkpoint)
    emb = m.w("model.embed_tokens.weight")
    hs = [emb[c].astype(np.float32) for c in chunks]
    lo, hi = (int(x) for x in a.layers.split("-"))
    os.makedirs(a.out, exist_ok=True)
    for li in range(m.L):
        if li > hi:
            break
        t = time.time()
        H, outs = {}, []
        for h in hs:
            out, cap = layer_capture(m, li, h)
            outs.append(out)
            if li < lo:
                continue
            for k, x in cap.items():
                x64 = x.astype(np.float64)
                H[k] = H.get(k, 0) + x64.T @ x64
        hs = outs
        if li < lo:
            continue
        res = {}
        for k, wname in MATS.items():
            codes, s = gptq_int4_group(
                m.w(f"model.layers.layers.{li}.{wname}"), H[k], G=a.group
            )
            res[k + "_q"], res[k + "_s"] = codes, s
        np.savez(os.path.join(a.out, f"{li:02d}.npz"), **res)
        print(f"layer {li} in {time.time() - t:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
