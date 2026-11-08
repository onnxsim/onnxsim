#!/usr/bin/env python3
"""Greedy PLaMo 3 translation on the phone with row-block layer programs (HMX).

Every pass runs a block of ROWS token positions through the 24 one-layer programs (export_chunks.py --rows ROWS, compiled
by compile_v69.sh with GEMM_FP16=1 HMX=1). The prompt goes in as one block (or several, if longer); each generated token is
a block with one real row. The host keeps the embedding, the final norm and LM head, and each layer's KV cache.

  run_phone_block.py CHECKPOINT TGHX_DIR --ids 1,16,... [--rows 32] [--cache-len 128] [--max-new 64] [--check]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..", "..")
sys.path.insert(0, os.path.join(REPO, "tools", "onnx-remote", "python"))
sys.path.insert(0, os.path.join(REPO, "scripts", "axera"))

import onnx_remote_client as C  # noqa: E402
import plamo3_reference as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("tghx_dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39520)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cache-len", type=int, default=128)
    ap.add_argument("--ids", required=True, help="prompt token ids, comma separated")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--eos", default="1,16")
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare every step's logits with the NumPy reference",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="compare each layer's hidden state of the first block with the reference",
    )
    a = ap.parse_args()

    ref = R.Plamo3Reference.from_checkpoint(a.checkpoint)
    emb = ref.w("model.embed_tokens.weight")
    gain = ref.w("model.norm.weight")
    lm_T = ref.w("lm_head.weight").T.copy()
    nkv, hd, H, L = ref.nkv, ref.hd, ref.H, ref.L
    C_, RW = a.cache_len, a.rows
    cli = C.Client(a.host, a.port, io_timeout=3600.0)
    names = [
        os.path.basename(p)
        for p in sorted(glob.glob(os.path.join(a.tghx_dir, "chunk_*.tghx")))
    ]
    assert len(names) == L, f"expected {L} layer programs, found {len(names)}"
    t0 = time.time()
    for n in names:
        cli.load_compiled(n, open(os.path.join(a.tghx_dir, n), "rb").read())
    print(f"loaded {len(names)} programs in {time.time() - t0:.1f}s", flush=True)

    kc = [np.zeros((nkv, C_, hd), np.float32) for _ in range(L)]
    vc = [np.zeros((nkv, C_, hd), np.float32) for _ in range(L)]
    eos = {int(x) for x in a.eos.split(",")}

    def block(tokens, p0):
        """run tokens (<= RW) at cache offset p0; returns the logits of the last real row"""
        n = len(tokens)
        hidden = np.zeros((RW, H), np.float32)
        hidden[:n] = emb[np.asarray(tokens)]
        pos = np.array([p0 + min(r, n - 1) for r in range(RW)], np.int64)
        mask = np.full((RW, C_ + RW), -1e9, np.float32)
        for r in range(RW):
            if r < n:
                mask[r, :p0] = 0.0
                mask[r, C_ : C_ + r + 1] = 0.0
            else:
                mask[r, C_] = (
                    0.0  # padding rows attend one slot (no NaN); their outputs are dropped
                )
        href = (
            emb[np.asarray(tokens)].astype(np.float32)
            if (a.trace and p0 == 0)
            else None
        )
        for i in range(L):
            outs = cli.run_compiled(names[i], [hidden, pos, mask, kc[i], vc[i]])
            hidden = outs[0].reshape(RW, H).astype(np.float32)
            if href is not None:
                href = ref.layer(i, href)
                d = float(np.max(np.abs(hidden[:n] - href)) / np.max(np.abs(href)))
                print(f"  trace layer {i}: rel-max-diff {d:.3e}", flush=True)
            kc[i][:, p0 : p0 + n] = outs[1].reshape(nkv, RW, hd)[:, :n]
            vc[i][:, p0 : p0 + n] = outs[2].reshape(nkv, RW, hd)[:, :n]
        last = R.Plamo3Reference.rms(hidden[n - 1 : n], gain, 1.0)
        return (last @ lm_T).reshape(-1)

    ids = [int(x) for x in a.ids.split(",")]
    assert len(ids) + a.max_new <= C_, "prompt + output must fit the cache"
    seq = list(ids)
    p0 = 0
    t_prefill = time.time()
    logits = None
    while p0 < len(ids):
        chunk = ids[p0 : p0 + RW]
        logits = block(chunk, p0)
        p0 += len(chunk)
    print(f"prefill {len(ids)} tokens in {time.time() - t_prefill:.2f}s", flush=True)
    generated = []
    worst = 0.0
    for step in range(a.max_new):
        nxt = int(np.argmax(logits))
        if a.check:
            ref_logits = ref.logits(seq)[-1]
            d = float(
                np.max(np.abs(logits - ref_logits))
                / (np.max(np.abs(ref_logits)) + 1e-9)
            )
            worst = max(worst, d)
            print(
                f"  step {step}: argmax {nxt} (ref {int(np.argmax(ref_logits))}) rel-max-diff {d:.3e}",
                flush=True,
            )
        generated.append(nxt)
        seq.append(nxt)
        if nxt in eos:
            break
        t = time.time()
        logits = block([nxt], len(seq) - 1)
        print(f"  step {step}: token {nxt} in {time.time() - t:.2f}s", flush=True)
    print("generated", generated, flush=True)
    if a.check:
        print(f"worst rel logit diff {worst:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
