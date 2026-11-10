#!/usr/bin/env python3
"""Greedy PLaMo 3 translation on the phone with prompt-lookup speculative decoding (HMX row-block programs).

Each pass runs one block of up to ROWS token positions through the layer-group programs. The device computes all ROWS rows
whatever the number of real ones, so a pass costs the same. The block holds the next token and up to ROWS-1 draft tokens,
copied from the prompt and output so far (prompt lookup). Row i's logits give the greedy choice after draft i. The longest
prefix of drafts that matches those choices is accepted, plus one bonus token from the first mismatching row, so the
output is identical to plain greedy decoding. KV rows of rejected drafts stay in the device cache, but they are masked
and later overwritten by the next block, so nothing is rolled back.

When every program has the same layer count, a pass is one chained run_compiled request ("a+b+...": the worker feeds each
program's hidden output to the next one on the phone), instead of one request per program.

  run_phone_spec.py CHECKPOINT TGHX_DIR --ids 1,16,... [--rows 32] [--cache-len 256] [--max-new 40] [--no-spec] [--no-chain] [--profile]
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


def draft_tokens(seq, k, nmax=4):
    """Prompt lookup: the tokens that followed the most recent earlier occurrence of the longest suffix of seq
    (length nmax down to 1). Returns up to k tokens, or [] when nothing matches."""
    for n in range(min(nmax, len(seq) - 1), 0, -1):
        suffix = seq[-n:]
        for s in range(len(seq) - n - 1, -1, -1):
            if seq[s : s + n] == suffix:
                return seq[s + n : s + n + k]
    return []


def block_inputs(emb_rows, p0, rows, cache_len):
    """The row-block graph's inputs for len(emb_rows) real tokens at cache offset p0 (export_chunks.py --state): hidden,
    position_ids, attn_mask (the cache before p0 plus causal within the block; padding rows attend one slot so softmax has
    no NaN), and wsel / wmask, which write the real rows' K/V to cache slots p0..p0+n-1."""
    n, H = emb_rows.shape
    assert 1 <= n <= rows and p0 + n <= cache_len
    hidden = np.zeros((rows, H), np.float32)
    hidden[:n] = emb_rows
    pos = np.array([p0 + min(r, n - 1) for r in range(rows)], np.int64)
    mask = np.full((rows, cache_len + rows), -1e9, np.float32)
    for r in range(rows):
        if r < n:
            mask[r, :p0] = 0.0
            mask[r, cache_len : cache_len + r + 1] = 0.0
        else:
            mask[r, cache_len] = 0.0
    wsel = np.zeros((cache_len, rows), np.float32)
    wmask = np.zeros((cache_len, 1), np.float32)
    for r in range(n):
        wsel[p0 + r, r] = 1.0
        wmask[p0 + r, 0] = 1.0
    return hidden, pos, mask, wsel, wmask


def generate(
    block,
    ids,
    rows,
    cache_len,
    max_new,
    eos=(),
    spec=True,
    on_pass=None,
    on_prefill=None,
):
    """Greedy decoding over block(tokens, p0) -> logits [len(tokens), V] (the block writes the tokens' KV rows from
    cache offset p0). With spec, each pass also feeds prompt-lookup drafts and keeps the longest prefix that matches
    the greedy choices, plus the bonus token. Returns (tokens, passes, accepted, drafted)."""
    P = 0
    logits = None
    while P < len(ids):
        chunk = ids[P : P + rows]
        logits = block(chunk, P)
        P += len(chunk)
    if on_prefill:
        on_prefill()
    # invariant: seq[:P] has its KV in the cache; seq[P] is the next token to feed
    seq = list(ids) + [int(np.argmax(logits[-1]))]
    out, passes, accepted, drafted = (
        [seq[-1]],
        0,
        0,
        0,
    )  # the prefill's own prediction is the first output token
    while len(out) < max_new and out[-1] not in eos:
        drafts = draft_tokens(seq, min(rows - 1, cache_len - P - 2)) if spec else []
        tokens = [seq[P]] + drafts
        t = time.time()
        logits = block(tokens, P)
        passes += 1
        g = [
            int(np.argmax(logits[i])) for i in range(len(tokens))
        ]  # the greedy choice after each row
        k = 0
        while k < len(drafts) and g[k] == drafts[k]:
            k += 1
        accepted += k
        drafted += len(drafts)
        new = drafts[:k] + [g[k]]  # the accepted drafts + the bonus token
        seq = seq[: P + 1] + new
        P += 1 + k
        for tok in new:
            out.append(tok)
            if tok in eos or len(out) >= max_new:
                break
        if on_pass:
            on_pass(passes, len(tokens), len(drafts), k, new, time.time() - t)
    return out, passes, accepted, drafted


def main() -> int:
    import onnx_remote_client as C
    import plamo3_reference as R

    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("tghx_dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39520)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cache-len", type=int, default=256)
    ap.add_argument("--ids", required=True, help="prompt token ids, comma separated")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--eos", default="1,16")
    ap.add_argument(
        "--no-spec",
        action="store_true",
        help="plain greedy (one token per pass), for comparison",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="split each decode pass into host setup / transfer / device / LM head",
    )
    ap.add_argument(
        "--no-chain",
        action="store_true",
        help="one request per program, instead of one chained request per pass (the worker feeds each program's "
        "hidden output to the next: 0.03 s of transfer per pass instead of 0.14 s)",
    )
    ap.add_argument(
        "--profile-calls",
        help="write the per-call device times of the last decode pass to this JSON file",
    )
    a = ap.parse_args()

    ref = R.Plamo3Reference.from_checkpoint(a.checkpoint)
    emb = ref.w("model.embed_tokens.weight")
    gain = ref.w("model.norm.weight")
    lm_T = ref.w("lm_head.weight").T.copy()
    nkv, hd, H, L = ref.nkv, ref.hd, ref.H, ref.L
    C_, RW = a.cache_len, a.rows
    cli = C.Client(a.host, a.port, io_timeout=3600.0)
    progs = []
    for path in sorted(glob.glob(os.path.join(a.tghx_dir, "chunk_*.tghx"))):
        lo, hi = (
            int(x)
            for x in os.path.basename(path)[len("chunk_") : -len(".tghx")].split("_")
        )
        progs.append((os.path.basename(path), lo, hi))
    covered = [i for _, lo, hi in progs for i in range(lo, hi + 1)]
    assert covered == list(range(L)), (
        f"programs cover layers {covered}, expected 0..{L - 1}"
    )
    t0 = time.time()
    for n, _, _ in progs:
        cli.load_compiled(n, open(os.path.join(a.tghx_dir, n), "rb").read())
    print(f"loaded {len(progs)} programs in {time.time() - t0:.1f}s", flush=True)

    ran = set()  # programs whose device-resident caches are valid
    tm = {"setup": 0.0, "rpc": 0.0, "device": 0.0, "head": 0.0}
    per_prog, calls = {}, {}
    level = (
        C.PROFILING_DETAILED
        if a.profile_calls
        else C.PROFILING_SUMMARY
        if a.profile
        else C.PROFILING_OFF
    )

    # a chain needs every program to take the same inputs (the same layer count)
    a.chain = not a.no_chain and len({hi - lo for _, lo, hi in progs}) == 1
    if a.chain:
        groups = [("+".join(n for n, _, _ in progs), progs[0][1], progs[0][2])]
    else:
        groups = progs

    def block(tokens, p0):
        calls.clear()  # the per-call times of the latest pass only
        ts = time.time()
        hidden, pos, mask, wsel, wmask = block_inputs(
            emb[np.asarray(tokens)], p0, RW, C_
        )
        tm["setup"] += time.time() - ts
        for name, lo, hi in groups:
            inputs = [hidden, pos, mask, wsel, wmask]
            for _ in range(lo, hi + 1):
                # the caches live on the phone: the first call of a program sends zeros, later calls send nothing
                if name in ran:
                    inputs += [np.zeros((0,), np.float32)] * 2
                else:
                    inputs += [np.zeros((nkv, C_, hd), np.float32)] * 2
            ran.add(name)
            tr = time.time()
            resp = cli.request(
                "run_compiled", inputs, artifact_id=name, profiling=level
            )
            tm["rpc"] += time.time() - tr
            for e in resp.profile:
                # a chain's events name their program first in the detail
                prog = e.detail.split()[0] if a.chain and e.detail else name
                if e.name == "hexagon_run":
                    tm["device"] += e.duration_us / 1e6
                    per_prog[prog] = per_prog.get(prog, 0.0) + e.duration_us / 1e6
                elif a.profile_calls:
                    calls.setdefault(prog, []).append((e.name, e.detail, e.duration_us))
            hidden = resp.outputs[0].reshape(RW, H).astype(np.float32)
        th = time.time()
        lg = R.Plamo3Reference.rms(hidden[: len(tokens)], gain, 1.0) @ lm_T
        tm["head"] += time.time() - th
        return lg

    ids = [int(x) for x in a.ids.split(",")]
    assert len(ids) + a.max_new <= C_ - RW, (
        "prompt + output (+ one block of drafts) must fit the cache"
    )
    t_start = time.time()
    t_dec = [0.0]

    def prefilled():
        print(f"prefill {len(ids)} tokens in {time.time() - t_start:.2f}s", flush=True)
        for k in tm:
            tm[k] = 0.0  # count decode passes only
        per_prog.clear()
        t_dec[0] = time.time()

    def pass_done(i, fed, nd, k, new, dt):
        print(
            f"  pass {i}: fed {fed} (drafts {nd}, accepted {k}) -> {new} in {dt:.2f}s",
            flush=True,
        )

    out, passes, accepted, drafted = generate(
        block,
        ids,
        RW,
        C_,
        a.max_new,
        {int(x) for x in a.eos.split(",")},
        not a.no_spec,
        pass_done,
        prefilled,
    )
    dt = time.time() - t_dec[0]
    print(f"generated {out}", flush=True)
    print(
        f"{len(out)} tokens in {passes} passes, {dt:.2f}s decode: {dt / max(1, len(out)):.3f} s/token, "
        f"{len(out) / max(1, passes):.2f} tokens/pass, drafts accepted {accepted}/{drafted}",
        flush=True,
    )
    if a.profile or a.profile_calls:
        pp = max(1, passes)
        print(
            f"per pass: total {dt / pp:.3f}s = setup {tm['setup'] / pp:.3f} + rpc {tm['rpc'] / pp:.3f} "
            f"(device {tm['device'] / pp:.3f}, transfer+worker {(tm['rpc'] - tm['device']) / pp:.3f}) + head {tm['head'] / pp:.3f}",
            flush=True,
        )
        print(
            "device per program per pass: "
            + ", ".join(f"{k} {v / pp * 1e3:.0f} ms" for k, v in per_prog.items()),
            flush=True,
        )
    if a.profile_calls:
        import json

        with open(a.profile_calls, "w") as f:
            json.dump(calls, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
