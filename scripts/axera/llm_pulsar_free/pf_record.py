"""Record the inputs of every program, all layers + post, over real-text float
decode steps (the calibration data of ``pf_generate.py`` and ``pf_gptq.py``).

  rest  : positions first_pos+1 .. S-1 of each ``CAL_DOCS`` passage
  first : position first_pos with ``N_FIRST`` different tokens (the passages'
          tokens), the positions before it as in the first passage
  nl    : with ``Config.nl_tokens`` (llama-160m), the rest set's positions of
          each passage both as prose and with its line breaks kept
          (``pf_layout.lines``): the first newline of a sequence is massive

Per layer: x (layer input), a (n1 out), qp, kp, vp, att, op, x1, m (n2 out),
gate, up, act, down, and the position p (for the RoPE tables); the attention
inputs (qr = RoPE'd q, K / V in the plan layout: the new row in slot p, row S
zero) for about ``N_ATTN`` sampled steps. Post: the final hidden, xn, logits.
Writes ``<out>/acts/<set>/L<l>.npz`` and ``post.npz``.

usage: MODEL=llama160m pf_record.py [--n-first 512] [--n-attn 64] [--[no-]nl]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402

F32 = np.float32
KEYS = ("a", "qp", "kp", "vp", "att", "op", "x1", "m", "gate", "up", "act", "down")


class Recorder:
    def __init__(self, cfg: pf_model.Config, n_attn: int, seed: int = 11):
        self.cfg, self.n_attn = cfg, n_attn
        sets = ("first", "rest", "nl")
        self.acc = {st: [dict() for _ in range(cfg.L)] for st in sets}
        self.post = {st: {} for st in sets}
        self.cur: dict = {}
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _put(d, k, v):
        a = np.asarray(v, F32)
        d.setdefault(k, []).append(
            a.reshape(1, -1) if a.ndim <= 2 and k not in ("K", "V") else a
        )

    def record(self, li, t):
        S, p = self.cfg.S, self.cur["p"]
        a = self.acc[self.cur["set"]][li]
        self._put(a, "x", t["input"])
        for k in KEYS:
            self._put(a, k, t[k])
        a.setdefault("p", []).append(p)
        if self.cur["attn"]:
            K, V = t["K_cache_out"].copy(), t["V_cache_out"].copy()
            K[p], V[p] = K[S], V[S]  # plan layout: the new row in slot p, row S zero
            K[S], V[S] = 0, 0
            self._put(a, "K", K)
            self._put(a, "V", V)
            self._put(a, "qr", np.asarray(t["qr"], F32).reshape(1, -1))
            a.setdefault("p_attn", []).append(p)

    def steps(self, ids, positions, st, attn_rate):
        cfg = self.cfg
        Ks, Vs = cfg.new_cache()
        for p in range(max(positions) + 1):
            rec = p in positions
            self.cur.update(p=p, set=st, attn=rec and self.rng.random() < attn_rate)
            if (
                p == 0 and cfg.zero_pos0
            ):  # zero end to end: nothing to compute, caches stay zero
                continue
            logits, pn = cfg.float_step(Ks, Vs, ids[p], p, self.record if rec else None)
            if rec:
                pp = self.post[st]
                self._put(pp, "x", pn["input"])
                self._put(pp, "xn", pn["xn"])
                self._put(pp, "logits", logits)

    def save(self, st):
        d = os.path.join(self.cfg.out, "acts", st)
        os.makedirs(d, exist_ok=True)
        for li in range(self.cfg.L):
            arr = {
                k: np.stack(v) if k not in ("p", "p_attn") else np.asarray(v)
                for k, v in self.acc[st][li].items()
            }
            np.savez(os.path.join(d, f"L{li}.npz"), **arr)
            self.acc[st][li] = None
        np.savez(
            os.path.join(d, "post.npz"),
            **{k: np.stack(v) for k, v in self.post[st].items()},
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-first", type=int, default=512)
    ap.add_argument("--n-attn", type=int, default=64)
    ap.add_argument(
        "--nl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also the nl set (default: when the model has nl_tokens)",
    )
    a = ap.parse_args()
    cfg = pf_model.load()
    rec = Recorder(cfg, a.n_attn)
    S, fp = cfg.S, cfg.first_pos
    t0 = time.time()
    n_rest = len(lay.CAL_DOCS) * (S - 1 - fp)
    firsts = []
    for doc in lay.CAL_DOCS:
        ids = cfg.passage_ids(doc)
        firsts += ids[fp:]
        rec.steps(ids, set(range(fp + 1, S)), "rest", a.n_attn / n_rest)
        print(doc, f"{time.time() - t0:.0f}s", flush=True)
    rec.save("rest")
    toks = list(dict.fromkeys(firsts))[: a.n_first]
    base = cfg.passage_ids(lay.CAL_DOCS[0])[:fp]
    for i, t in enumerate(toks):
        rec.steps(base + [t], {fp}, "first", a.n_attn / len(toks))
        if (i + 1) % 128 == 0:
            print("first", i + 1, f"{time.time() - t0:.0f}s", flush=True)
    rec.save("first")
    if cfg.nl_tokens if a.nl is None else a.nl:
        # after first: the rest / first sets' attention samples stay as without
        passages = [(d, lines) for lines in (False, True) for d in lay.CAL_DOCS]
        n_nl = len(passages) * (S - 1 - fp)
        for doc, lines in passages:
            ids = cfg.passage_ids(doc, lines=lines)
            rec.steps(ids, set(range(fp + 1, S)), "nl", a.n_attn / n_nl)
            print("nl", doc, "lines" if lines else "", f"{time.time() - t0:.0f}s")
        rec.save("nl")
    print("done", f"{time.time() - t0:.0f}s", "first tokens", len(toks))


if __name__ == "__main__":
    main()
