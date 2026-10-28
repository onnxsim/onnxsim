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

``--device cuda`` runs the float steps on torch (``pf_float_torch``: the same
graphs, about 5x faster); the activations then differ from the CPU's
(``ReferenceEvaluator``) in the last bits.

usage: MODEL=llama160m pf_record.py [--n-first 512] [--n-attn 64] [--[no-]nl] [--device cuda]
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


def rss_mb() -> int:
    """This process's resident set (VmRSS, MB), for the group log."""
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) // 1024
    return -1


class Recorder:
    def __init__(self, cfg: pf_model.Config, n_attn: int, seed: int = 11, fwd=None):
        self.cfg, self.n_attn = cfg, n_attn
        self.fwd = fwd or cfg  # new_cache / float_step: the CPU's or pf_float_torch's
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

    def run_jobs(self, jobs, group: int, st: str):
        """Run the sequences ``jobs`` = [(ids, positions, set, attn_rate)] of one set,
        layer-major: each group of ``group`` layers (0: all) runs every job's positions
        before the next group, so only one group's float graphs are resident (the
        per-token float step needs all layers at once: all ``L`` graphs, ~16 GB for
        Qwen3-4B). The numbers are the per-token step's: the same layer ops at each
        position, the caches written in position order. The attention draws are made
        in job order, as ``steps`` makes them. Torch's float step keeps ``steps``."""
        cfg = self.cfg
        plan = []
        for ids, positions, st, attn_rate in jobs:  # the draws, in the old order
            n = max(positions) + 1
            flags = [p in positions and self.rng.random() < attn_rate for p in range(n)]
            plan.append(dict(ids=ids, pos=positions, st=st, flags=flags, n=n))
        if not hasattr(self.fwd, "float_layer"):  # torch: the per-token step
            for j in plan:
                self._steps_planned(j)
            self.save(st)
            return

        def live(j):
            return [p for p in range(j["n"]) if not (p == 0 and cfg.zero_pos0)]

        for j in plan:
            j["x"] = {p: self.fwd.embed_row(j["ids"][p]) for p in live(j)}
        L = cfg.L
        step = group if group else L
        for a in range(0, L, step):
            layers = list(range(a, min(a + step, L)))
            for j in plan:
                Ks = {li: np.zeros((cfg.S, cfg.kvd), F32) for li in layers}
                Vs = {li: np.zeros((cfg.S, cfg.kvd), F32) for li in layers}
                for p in live(j):
                    rec = p in j["pos"]
                    self.cur.update(p=p, set=j["st"], attn=j["flags"][p])
                    x = j["x"][p]
                    for li in layers:
                        t = self.fwd.float_layer(li, x, Ks[li], Vs[li], p)
                        if rec:
                            self.record(li, t)
                        Ks[li][p] = t["K_cache_out"][cfg.S]
                        Vs[li][p] = t["V_cache_out"][cfg.S]
                        x = t["output"].reshape(1, cfg.H)
                    j["x"][p] = x
            self.fwd.drop_layers(layers)
            self.save_layers(st, layers)  # the group's layers are complete: to disk
            print(
                f"  layers {layers[0]}-{layers[-1]} done rss {rss_mb()} MB", flush=True
            )
        for j in plan:
            for p in live(j):
                if p in j["pos"]:
                    logits, pn = self.fwd.post_of(j["x"][p])
                    pp = self.post[j["st"]]
                    self._put(pp, "x", pn["input"])
                    self._put(pp, "xn", pn["xn"])
                    self._put(pp, "logits", logits)
        self.save_post(st)

    def _steps_planned(self, j):
        cfg = self.cfg
        Ks, Vs = self.fwd.new_cache()
        for p in range(j["n"]):
            rec = p in j["pos"]
            self.cur.update(p=p, set=j["st"], attn=j["flags"][p])
            if p == 0 and cfg.zero_pos0:
                continue
            logits, pn = self.fwd.float_step(
                Ks, Vs, j["ids"][p], p, self.record if rec else None
            )
            if rec:
                pp = self.post[j["st"]]
                self._put(pp, "x", pn["input"])
                self._put(pp, "xn", pn["xn"])
                self._put(pp, "logits", logits)

    def steps(self, ids, positions, st, attn_rate):
        cfg = self.cfg
        Ks, Vs = self.fwd.new_cache()
        for p in range(max(positions) + 1):
            rec = p in positions
            self.cur.update(p=p, set=st, attn=rec and self.rng.random() < attn_rate)
            if (
                p == 0 and cfg.zero_pos0
            ):  # zero end to end: nothing to compute, caches stay zero
                continue
            logits, pn = self.fwd.float_step(
                Ks, Vs, ids[p], p, self.record if rec else None
            )
            if rec:
                pp = self.post[st]
                self._put(pp, "x", pn["input"])
                self._put(pp, "xn", pn["xn"])
                self._put(pp, "logits", logits)

    def save_layers(self, st, layers):
        """Write ``L<li>.npz`` of each layer in ``layers`` and free its records (the
        records of a layer are complete once its group has run for every sequence)."""
        d = os.path.join(self.cfg.out, "acts", st)
        os.makedirs(d, exist_ok=True)
        for li in layers:
            if self.acc[st][li] is None:
                continue
            arr = {
                k: np.stack(v) if k not in ("p", "p_attn") else np.asarray(v)
                for k, v in self.acc[st][li].items()
            }
            np.savez(os.path.join(d, f"L{li}.npz"), **arr)
            self.acc[st][li] = None

    def save_post(self, st):
        d = os.path.join(self.cfg.out, "acts", st)
        os.makedirs(d, exist_ok=True)
        np.savez(
            os.path.join(d, "post.npz"),
            **{k: np.stack(v) for k, v in self.post[st].items()},
        )

    def save(self, st):
        self.save_layers(st, range(self.cfg.L))
        self.save_post(st)


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
    ap.add_argument("--device", default="cpu", help="cpu (numpy) or a torch device")
    ap.add_argument(
        "--layers-per-group",
        type=int,
        default=4,
        help="layers whose float graphs are resident at once (0: all of them)",
    )
    a = ap.parse_args()
    cfg = pf_model.load()
    fwd = None
    if a.device != "cpu":
        import pf_float_torch

        fwd = pf_float_torch.TorchFloat(cfg, a.device)
    rec = Recorder(cfg, a.n_attn, fwd=fwd)
    S, fp = cfg.S, cfg.first_pos
    t0 = time.time()
    group = a.layers_per_group
    n_rest = len(lay.CAL_DOCS) * (S - 1 - fp)
    # each set's sequences run layer-major (``Recorder.run_jobs``); the draws stay in
    # the order rest, first, nl
    rest = []
    firsts = []
    for doc in lay.CAL_DOCS:
        ids = cfg.passage_ids(doc)
        firsts += ids[fp:]
        rest.append((ids, set(range(fp + 1, S)), "rest", a.n_attn / n_rest))
    rec.run_jobs(rest, group, "rest")
    print("rest", f"{time.time() - t0:.0f}s", flush=True)
    toks = list(dict.fromkeys(firsts))[: a.n_first]
    base = cfg.passage_ids(lay.CAL_DOCS[0])[:fp]
    first = [(base + [t], {fp}, "first", a.n_attn / len(toks)) for t in toks]
    rec.run_jobs(first, group, "first")
    print("first", len(toks), f"{time.time() - t0:.0f}s", flush=True)
    if cfg.nl_tokens if a.nl is None else a.nl:
        # after first: the rest / first sets' attention samples stay as without
        passages = [(d, lines) for lines in (False, True) for d in lay.CAL_DOCS]
        n_nl = len(passages) * (S - 1 - fp)
        nl = [
            (
                cfg.passage_ids(doc, lines=lines),
                set(range(fp + 1, S)),
                "nl",
                a.n_attn / n_nl,
            )
            for doc, lines in passages
        ]
        rec.run_jobs(nl, group, "nl")
        print("nl", f"{time.time() - t0:.0f}s", flush=True)
    print("done", f"{time.time() - t0:.0f}s", "first tokens", len(toks))


if __name__ == "__main__":
    main()
