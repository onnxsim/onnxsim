"""Host estimate of a model's Pulsar-free program set: the plan's chain, every
program evaluated as the card runs it (``pf_fake_quant``: inputs / inner
tensors / outputs rounded at their U16 calibrations, linears with s8
weights), K / V caches in the plan layout, position 0 skipped when it is zero
(``zero_pos0``), position ``first_pos`` on the ``first`` set, later positions
on ``rest``; compared with the float model, teacher-forced.

--float: no rounding anywhere and float weights (a wiring check against
``Config.float_step``). Ablations: --exact KEYS leaves those programs exact,
--float-weights KEYS keeps those linears' float weights (activations still
rounded). --card-add: every U16 Add as the card's fixed-point Add
(``pf_fake_quant.card_add``), with its zero-point bias.

usage: MODEL=llama160m pf_estimate.py [--doc 0] [--prompt 48] [--float]
       -> <out>/est_doc<d>[_float].json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
from pf_fake_quant import AddEval, FusedEval, LinearEval  # noqa: E402

F32 = np.float32


def _exact_linear(w):
    W = np.asarray(w, np.float64)
    return lambda x: (np.asarray(x, np.float64) @ W).astype(F32)


def _exact_add(a, b):
    return (a + b).astype(F32)


class Chain:
    """One calibration set's programs as host functions (``root``: where the
    manifests and the programs' float graphs are, default ``cfg.out``)."""

    def __init__(
        self,
        cfg,
        st,
        exact=False,
        exact_keys=(),
        float_weights=(),
        root=None,
        card_add=False,
    ):
        self.cfg = cfg
        root = root or cfg.out
        man = json.load(open(os.path.join(root, f"manifest_{st}.json")))
        self.f = {}
        for wh, ents in man.items():
            li = None if wh == "post" else int(wh[1:])
            for k, e in ents.items():
                ex = exact or k in exact_keys
                if e["kind"] == "fused":
                    self.f[(wh, k)] = FusedEval(
                        os.path.join(root, e["model"]),
                        {} if ex else e["calibration"],
                        e["signed"],
                        card_add=card_add,
                        exact_offset=e.get("add_offset") == "exact",
                    )
                elif e["kind"] == "split":
                    w = cfg.merged_weight(k, li, gptq=False)
                    if ex:
                        self.f[(wh, k)] = _exact_linear(w)
                        continue
                    _, r = cfg.split_weights(w)
                    sp = e["split"]
                    hi = LinearEval(
                        w, sp["hi"]["calibration"]["x"], sp["hi"]["calibration"]["y"]
                    )
                    lo = LinearEval(
                        r, sp["lo"]["calibration"]["x"], sp["lo"]["calibration"]["y"]
                    )
                    ad = AddEval(
                        sp["add"]["calibration"],
                        card_add,
                        e.get("add_offset") == "exact",
                    )
                    self.f[(wh, k)] = (lambda hi, lo, ad: lambda x: ad(hi(x), lo(x)))(
                        hi, lo, ad
                    )
                elif e["kind"] == "linear":
                    w = cfg.merged_weight(k, li)
                    if ex:
                        self.f[(wh, k)] = _exact_linear(w)
                    else:
                        c = e["calibration"]
                        self.f[(wh, k)] = LinearEval(
                            w, c["x"], c["y"], w8=k not in float_weights
                        )
                else:
                    self.f[(wh, k)] = (
                        _exact_add
                        if ex
                        else AddEval(
                            e["calibration"], card_add, e.get("add_offset") == "exact"
                        )
                    )

    def layer_fused(self, li, x, K, V, p):
        """A layer of the merged layout. ``x``: the embedding row (layer 0) or
        the previous layer's ``(x1, down)``, whose Add the layer's ``n1`` holds;
        returns this layer's ``(x1, down)``."""
        cfg = self.cfg
        kvd = cfg.kvd

        def f(k):
            return self.f[(f"L{li}", k)]

        t = cfg.rope_tiles(p, li)
        g1, g2 = cfg.gains(li)
        if isinstance(x, tuple):
            x, a = f("n1")(a=x[0], b=x[1], g=g1)
        else:
            a = f("n1")(x=x, g=g1)
        y5 = f("qkv5")(a)
        o = lay.offsets(cfg.QKV)

        def sl(k, n):
            return y5[:, o[k] : o[k] + n]

        qr, kr = self.rope(f, sl, t)
        K[p], V[p] = kr.reshape(-1), sl("v_proj", kvd).reshape(-1)
        att = f("attn")(q=qr, k=K, v=V, mask=cfg.mask_plan(p))
        x1, m = f("n2")(a=x, b=f("o_proj")(att), g=g2)
        gu = f("gu")(m)
        og = lay.offsets(cfg.GU)
        gate = gu[:, og["gate_proj"] : og["gate_proj"] + cfg.I]
        up = gu[:, og["up_proj"] : og["up_proj"] + cfg.I]
        return x1, f("down_proj")(f("silu_f")(gate=gate, up=up))

    def layer_fused2(self, li, x, K, V, p):
        """A layer of the ``fuse=2`` layout. ``x``: the embedding row (layer 0)
        or the previous layer's ``dn`` outputs ``(x, normed x)``."""
        cfg = self.cfg
        kvd = cfg.kvd

        def f(k):
            return self.f[(f"L{li}", k)]

        t = cfg.rope_tiles(p, li)
        g1, g2 = cfg.gains(li)
        x, a = x if isinstance(x, tuple) else (x, f("n1")(x=x, g=g1))
        y5 = f("qkv5")(a)
        o = lay.offsets(cfg.QKV)

        def sl(k, n):
            return y5[:, o[k] : o[k] + n]

        qr, kr = self.rope(f, sl, t)
        K[p], V[p] = kr.reshape(-1), sl("v_proj", kvd).reshape(-1)
        att_in = dict(q=qr, k=K, v=V, mask=cfg.mask_plan(p))
        if (f"L{li}", "aon2") in self.f:  # fuse=7: the attention is in aon2
            x1, m = f("aon2")(**att_in, a=x, g=g2)
        else:
            x1, m = f("on2")(z=f("attn")(**att_in), a=x, g=g2)
        gu = f("gu")(m)
        og = lay.offsets(cfg.GU)
        gate = gu[:, og["gate_proj"] : og["gate_proj"] + cfg.I]
        up = gu[:, og["up_proj"] : og["up_proj"] + cfg.I]
        gn = cfg.gains(li + 1)[0] if li + 1 < cfg.L else cfg.ln_f
        if (f"L{li}", "silu_f") not in self.f:  # fuse=3: the SiLU is in dn
            return f("dn")(gate=gate, up=up, a=x1, g=gn)
        act = f("silu_f")(gate=gate, up=up)
        if (f"L{li}", "down_proj") in self.f:
            return f("dn")(a=x1, b=f("down_proj")(act), g=gn)
        return f("dn")(z=act, a=x1, g=gn)

    def rope(self, f, sl, t):
        """q and the K row from the layer's RoPE program(s) (``qk``, or ``rq``
        and ``rk``) on the q / k slices ``sl(part, width)``."""
        cfg = self.cfg
        nq, kvd = cfg.nh * cfg.hd, cfg.kvd
        if cfg.qk_norm:
            return f("qk")(
                q=sl("q_proj", nq),
                qr=sl("q_rot", nq),
                k=sl("k_proj", kvd),
                kr=sl("k_rot", kvd),
                cq=t["cos_q"],
                sq=t["sin_q"],
                ck=t["cos_k"],
                sk=t["sin_k"],
            )
        qr = f("rq")(q=sl("q_proj", nq), r=sl("q_rot", nq), c=t["cos_q"], sn=t["sin_q"])
        kr = f("rk")(
            q=sl("k_proj", kvd), r=sl("k_rot", kvd), c=t["cos_k"], sn=t["sin_k"]
        )
        return qr, kr

    def layer_fused4(self, li, x, K, V, p):
        """A layer of the ``fuse=4`` layout. ``x``: the embedding row (layer 0)
        or the previous layer's ``{"x": residual, "a": normed | "y5": q / k / v}``;
        returns this layer's."""
        cfg = self.cfg
        kvd = cfg.kvd
        wh = f"L{li}"

        def f(k):
            return self.f[(wh, k)]

        t = cfg.rope_tiles(p, li)
        g1, g2 = cfg.gains(li)
        if not isinstance(x, dict):
            x = {"x": x, "y5": f("n1q")(x=x, g=g1)}
        res = x["x"]
        y5 = x["y5"] if "y5" in x else f("qkv5")(x["a"])
        o = lay.offsets(cfg.QKV)

        def sl(k, n):
            return y5[:, o[k] : o[k] + n]

        qr, kr = self.rope(f, sl, t)
        K[p], V[p] = kr.reshape(-1), sl("v_proj", kvd).reshape(-1)
        att = f("attn")(q=qr, k=K, v=V, mask=cfg.mask_plan(p))
        if (wh, "on2g") in self.f:
            x1, gu = f("on2g")(z=att, a=res, g=g2)
        else:
            x1, m = f("on2")(z=att, a=res, g=g2)
            gu = f("gu")(m)
        og = lay.offsets(cfg.GU)
        gate = gu[:, og["gate_proj"] : og["gate_proj"] + cfg.I]
        up = gu[:, og["up_proj"] : og["up_proj"] + cfg.I]
        gn = cfg.gains(li + 1)[0] if li + 1 < cfg.L else cfg.ln_f
        if (wh, "down_proj") in self.f:
            dn = f("down_proj")(f("silu_f")(gate=gate, up=up))
            xn, a = f("dn")(a=x1, b=dn, g=gn)
            return {"x": xn, "a": a}
        if (wh, "dnq") in self.f:
            xn, y5n = f("dnq")(gate=gate, up=up, a=x1, g=gn)
            return {"x": xn, "y5": y5n}
        xn, a = f("dn")(gate=gate, up=up, a=x1, g=gn)
        return {"x": xn, "a": a}

    def layer_fused5(self, li, x, K, V, p):
        """A layer of the ``fuse=5`` layout. ``x``: the embedding row (layer 0)
        or the previous layer's ``{"x": residual, "a": normed | "q": q}`` (its
        K / V rows already written)."""
        cfg = self.cfg
        nh, hd, kvd = cfg.nh, cfg.hd, cfg.kvd
        wh = f"L{li}"

        def f(k):
            return self.f[(wh, k)]

        def tabs(lj):
            t = cfg.rope_tiles(p, lj)
            return dict(cq=t["cos_q"], sq=t["sin_q"], ck=t["cos_k"], sk=t["sin_k"])

        g1, g2 = cfg.gains(li)
        if not isinstance(x, dict):
            qr, kr, v = f("n1qk")(x=x, g=g1, **tabs(li))
            K[p], V[p] = kr.reshape(-1), v.reshape(-1)
            x = {"x": x, "q": qr}
        if "q" not in x:  # after a split first-set layer
            y5 = f("qkv5")(x["a"])
            o = lay.offsets(cfg.QKV)

            def sl(k, n):
                return y5[:, o[k] : o[k] + n]

            qr, kr = f("qk")(
                q=sl("q_proj", nh * hd),
                qr=sl("q_rot", nh * hd),
                k=sl("k_proj", kvd),
                kr=sl("k_rot", kvd),
                **tabs(li),
            )
            K[p], V[p] = kr.reshape(-1), sl("v_proj", kvd).reshape(-1)
            x = {"x": x["x"], "q": qr}
        res = x["x"]
        att_in = dict(q=x["q"], k=K, v=V, mask=cfg.mask_plan(p))
        if (wh, "alyr") in self.f:  # fuse=8: the whole layer
            gn = cfg.gains(li + 1)[0]
            ins = {f"{k}_a": v for k, v in att_in.items()}
            xn, qn, kn, vn = f("alyr")(**ins, a=res, g_1=g2, g=gn, **tabs(li + 1))
            return {"x": xn, "q": qn, "kv": (kn, vn)}
        if (wh, "aon2g") in self.f:  # fuse=7: the attention is in aon2g
            x1, gu = f("aon2g")(**att_in, a=res, g=g2)
        elif (wh, "on2g") in self.f:
            x1, gu = f("on2g")(z=f("attn")(**att_in), a=res, g=g2)
        else:
            x1, m = f("on2")(z=f("attn")(**att_in), a=res, g=g2)
            gu = f("gu")(m)
        og = lay.offsets(cfg.GU)
        gate = gu[:, og["gate_proj"] : og["gate_proj"] + cfg.I]
        up = gu[:, og["up_proj"] : og["up_proj"] + cfg.I]
        gn = cfg.gains(li + 1)[0] if li + 1 < cfg.L else cfg.ln_f
        if (wh, "down_proj") in self.f:
            dn = f("down_proj")(f("silu_f")(gate=gate, up=up))
            xn, a = f("dn")(a=x1, b=dn, g=gn)
            return {"x": xn, "a": a}
        if (wh, "dnqk") in self.f:
            xn, qn, kn, vn = f("dnqk")(gate=gate, up=up, a=x1, g=gn, **tabs(li + 1))
            return {"x": xn, "q": qn, "kv": (kn, vn)}
        xn, a = f("dn")(gate=gate, up=up, a=x1, g=gn)
        return {"x": xn, "a": a}

    def layer(self, li, x, K, V, p):
        cfg = self.cfg
        # fuse=7 without a per-head q / k norm is layout 3 with aon2
        big = cfg.fuse if cfg.qk_norm else min(cfg.fuse, 3)
        if big >= 5:
            if isinstance(x, dict) and "kv" in x:  # the previous layer computed them
                K[p], V[p] = x["kv"][0].reshape(-1), x["kv"][1].reshape(-1)
            return self.layer_fused5(li, x, K, V, p)
        if big >= 4:
            return self.layer_fused4(li, x, K, V, p)
        if cfg.fuse >= 2:
            return self.layer_fused2(li, x, K, V, p)
        if cfg.fuse:
            return self.layer_fused(li, x, K, V, p)
        nh, hd, kvd = cfg.nh, cfg.hd, cfg.kvd

        def f(k):
            return self.f[(f"L{li}", k)]

        t = cfg.rope_tiles(p, li)
        g1, g2 = cfg.gains(li)
        a = f("n1_rms")(x=f("n1_c")(x=x), g=g1)
        y5 = f("qkv5")(a)
        o = lay.offsets(cfg.QKV)

        def sl(k, n):
            return y5[:, o[k] : o[k] + n]

        qr = f("rq")(
            q=sl("q_proj", nh * hd), r=sl("q_rot", nh * hd), c=t["cos_q"], sn=t["sin_q"]
        )
        if cfg.qk_norm:  # two-pass k norm: kc's u, then qkrope normalizes it exactly
            uk = f("kc")(
                x=np.concatenate([sl("k_proj", kvd), sl("k_rot", kvd)], axis=-1)
            )
            kr = f("rk")(q=uk[:, :kvd], r=uk[:, kvd:], c=t["cos_k"], sn=t["sin_k"])
        else:
            kr = f("rk")(
                q=sl("k_proj", kvd), r=sl("k_rot", kvd), c=t["cos_k"], sn=t["sin_k"]
            )
        K[p], V[p] = kr.reshape(-1), sl("v_proj", kvd).reshape(-1)
        att = f("attn")(q=qr, k=K, v=V, mask=cfg.mask_plan(p))
        x1 = f("res1")(x, f("o_proj")(att))
        m = f("n2_rms")(x=f("n2_c")(x=x1), g=g2)
        gu = f("gu")(m)
        og = lay.offsets(cfg.GU)
        gate = gu[:, og["gate_proj"] : og["gate_proj"] + cfg.I]
        up = gu[:, og["up_proj"] : og["up_proj"] + cfg.I]
        act = f("silu_f")(gate=gate, up=up)
        return f("res2")(x1, f("down_proj")(act))

    def post(self, x):
        if self.cfg.fuse >= 4 and self.cfg.qk_norm:
            return self.f[("post", "lm_head")](x["a"]).reshape(-1)
        if self.cfg.fuse >= 2:
            return self.f[("post", "lm_head")](x[1]).reshape(-1)
        if self.cfg.fuse:
            _, xn = self.f[("post", "nf")](a=x[0], b=x[1], g=self.cfg.ln_f)
            return self.f[("post", "lm_head")](xn).reshape(-1)
        u = self.f[("post", "nf_c")](x=x)
        xn = self.f[("post", "nf_rms")](x=u, g=self.cfg.ln_f)
        return self.f[("post", "lm_head")](xn).reshape(-1)


def run(cfg, ids, chains) -> list[dict]:
    """Teacher-forced steps over ``ids``; one row per step vs ``float_step``."""
    K = [np.zeros((cfg.S + 1, cfg.kvd), F32) for _ in range(cfg.L)]
    V = [np.zeros((cfg.S + 1, cfg.kvd), F32) for _ in range(cfg.L)]
    fK, fV = cfg.new_cache()
    rows = []
    for p, tok in enumerate(ids):
        if p == 0 and cfg.zero_pos0:
            continue  # zero end to end (the float caches stay zero there too)
        flg, _ = cfg.float_step(fK, fV, tok, p)
        c = chains["first" if p == cfg.first_pos else "rest"]
        x = cfg.model.embed([int(tok)]).reshape(1, cfg.H).astype(F32)
        for li in range(cfg.L):
            x = c.layer(li, x, K[li], V[li], p)
        lg = c.post(x)
        rows.append(
            dict(
                p=p,
                top1=int(lg.argmax()),
                float_top1=int(np.argmax(flg)),
                rel=lay.rel_err(lg, flg),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--prompt", type=int, default=48)
    ap.add_argument("--float", action="store_true")
    ap.add_argument(
        "--exact", default="", help="comma-separated program keys left exact"
    )
    ap.add_argument("--float-weights", default="", help="comma-separated linear keys")
    ap.add_argument(
        "--card-add", action="store_true", help="the card's fixed-point U16 Adds"
    )
    a = ap.parse_args()
    cfg = pf_model.load()
    ids = cfg.passage_ids(lay.EVAL_DOCS[a.doc])[: a.prompt]
    keys = set(filter(None, a.exact.split(",")))
    fw = set(filter(None, a.float_weights.split(",")))
    chains = {
        st: Chain(cfg, st, a.float, keys, fw, card_add=a.card_add)
        for st in ("first", "rest")
    }
    rows = run(cfg, ids, chains)
    res = dict(
        model=cfg.name,
        doc=lay.EVAL_DOCS[a.doc],
        float=a.float,
        **lay.summarize(rows),
        steps=rows,
    )
    json.dump(
        res,
        open(
            os.path.join(
                cfg.out,
                f"est_doc{a.doc}{'_float' if a.float else ''}"
                f"{'_card_add' if a.card_add else ''}.json",
            ),
            "w",
        ),
        indent=1,
    )
    print({k: v for k, v in res.items() if k != "steps"})


if __name__ == "__main__":
    main()
