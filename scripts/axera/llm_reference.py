"""Numpy float32 reference forward of an HF `LlamaForCausalLM` checkpoint (SmolLM2, ...).

No torch needed: the safetensors file is parsed by hand (8-byte little-endian
header length, JSON header, raw little-endian tensor bytes), BF16 tensors are
widened to float32 exactly.

Layer math (HF Llama, pre-norm):
    a   = rmsnorm(x, input_layernorm)
    q,k,v = a @ Wq.T, a @ Wk.T, a @ Wv.T          (no biases)
    q,k rotated by RoPE at the token's absolute position
    attn = softmax(q.k / sqrt(head_dim) + mask) . v   (grouped-query attention)
    x   = x + attn @ Wo.T
    m   = rmsnorm(x, post_attention_layernorm)
    x   = x + (silu(m @ Wgate.T) * (m @ Wup.T)) @ Wdown.T
Final: logits = rmsnorm(x, model.norm) @ lm_head.T   (lm_head tied to embed_tokens)

Public surface:
    Model.load(checkpoint_dir)
    model.embed(ids)                               -> [T, H] float32
    model.layer_step(li, x, pos, K, V, mask)       -> (y, k_row, v_row)   one token, explicit cache
    model.forward_full(ids)                        -> (logits [T, V], hiddens [L+1, T, H])  uncached
    model.forward_cached(ids)                      -> (logits [T, V], hiddens [L+1, T, H])  one token at a time
    model.greedy(prompt_ids, max_new_tokens)       -> list[int]
    model.post(x, apply_norm=True)                 -> logits

`hiddens[0]` is the embedding, `hiddens[l + 1]` is the output of layer `l`
(before the final norm), i.e. exactly what the compiled layer `l` should
return in its `output` tensor.

`python llm_reference.py CHECKPOINT_DIR [--torch]` checks the cached forward
against the uncached one, and with --torch against transformers on CPU.
See docs/axera-llm-rpc-decode.md.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Callable, Optional

import numpy as np

_ST_DTYPES = {
    "F32": "<f4",
    "F16": "<f2",
    "F64": "<f8",
    "I64": "<i8",
    "I32": "<i4",
    "BF16": "<u2",  # widened below
}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Read every tensor of a .safetensors file as float32 (BF16/F16 widened exactly)."""
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            beg, end = info["data_offsets"]
            f.seek(base + beg)
            raw = f.read(end - beg)
            arr = np.frombuffer(raw, dtype=_ST_DTYPES[info["dtype"]]).reshape(
                info["shape"]
            )
            if info["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[name] = np.ascontiguousarray(arr, dtype=np.float32)
    return out


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(var + np.float32(eps))) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (np.float32(1.0) + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class Model:
    def __init__(self, cfg: dict, w: dict[str, np.ndarray]):
        self.cfg = cfg
        self.w = w
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = self.H // self.nh
        self.kvdim = self.nkv * self.hd
        self.eps = cfg["rms_norm_eps"]
        self.vocab = cfg["vocab_size"]
        self.eos = cfg.get("eos_token_id")
        self.E = w["model.embed_tokens.weight"]
        self.lm_head = w["lm_head.weight"] if "lm_head.weight" in w else self.E
        if "lm_head.weight" not in w and not cfg.get("tie_word_embeddings", False):
            raise ValueError("no lm_head.weight and tie_word_embeddings is false")
        # HF: inv_freq = 1 / theta ** (arange(0, d, 2) / d); computed in float64, used as float32.
        self.inv_freq = 1.0 / (
            float(cfg["rope_theta"])
            ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        )
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            self.layers.append(
                {
                    "n1": w[p + "input_layernorm.weight"],
                    "q": w[p + "self_attn.q_proj.weight"],
                    "k": w[p + "self_attn.k_proj.weight"],
                    "v": w[p + "self_attn.v_proj.weight"],
                    "o": w[p + "self_attn.o_proj.weight"],
                    "n2": w[p + "post_attention_layernorm.weight"],
                    "gate": w[p + "mlp.gate_proj.weight"],
                    "up": w[p + "mlp.up_proj.weight"],
                    "down": w[p + "mlp.down_proj.weight"],
                }
            )

    @classmethod
    def load(cls, checkpoint_dir: str) -> "Model":
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            cfg = json.load(f)
        return cls(
            cfg, read_safetensors(os.path.join(checkpoint_dir, "model.safetensors"))
        )

    # ---- pieces ---------------------------------------------------------
    def embed(self, ids) -> np.ndarray:
        return self.E[np.asarray(ids, dtype=np.int64)]

    def rope_cos_sin(self, pos) -> tuple[np.ndarray, np.ndarray]:
        """cos/sin of shape [..., head_dim] (HF layout: cat(freqs, freqs))."""
        ang = np.asarray(pos, dtype=np.float64)[..., None] * self.inv_freq
        ang = np.concatenate([ang, ang], axis=-1)
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    @staticmethod
    def _rot(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        h = x.shape[-1] // 2
        rh = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
        return x * cos + rh * sin

    def qkv_row(self, li: int, x: np.ndarray, pos: int):
        """One token. Returns (q [nh, hd] rotated, k_row [nkv*hd] rotated, v_row [nkv*hd])."""
        lw = self.layers[li]
        a = rmsnorm(x.reshape(self.H), lw["n1"], self.eps)
        cos, sin = self.rope_cos_sin(pos)
        q = self._rot((lw["q"] @ a).reshape(self.nh, self.hd), cos, sin)
        k = self._rot((lw["k"] @ a).reshape(self.nkv, self.hd), cos, sin)
        v = lw["v"] @ a
        return (
            q.astype(np.float32),
            k.reshape(-1).astype(np.float32),
            v.astype(np.float32),
        )

    def layer_step(
        self,
        li: int,
        x: np.ndarray,
        pos: int,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray,
    ):
        """One decode step of layer `li` with an explicit cache.

        x    [H]            hidden state of the current token
        pos                 absolute position of the current token (RoPE)
        K, V [S, nkv*hd]    cache rows (K already rotated at its own position)
        mask [S + 1]        additive; entry j < S applies to cache row j, entry S to the current token
        Returns (y [H], k_row [nkv*hd], v_row [nkv*hd]); k_row/v_row are what a
        host stores in the cache for this token.
        """
        lw = self.layers[li]
        x = x.reshape(self.H).astype(np.float32, copy=False)
        q, k_row, v_row = self.qkv_row(li, x, pos)
        S = K.shape[0]
        Kall = np.concatenate([K.reshape(S, self.kvdim), k_row[None]], axis=0).reshape(
            S + 1, self.nkv, self.hd
        )
        Vall = np.concatenate([V.reshape(S, self.kvdim), v_row[None]], axis=0).reshape(
            S + 1, self.nkv, self.hd
        )
        g = self.nh // self.nkv
        qg = q.reshape(self.nkv, g, self.hd)
        scores = np.einsum("kgd,skd->kgs", qg, Kall).astype(np.float32) / np.float32(
            np.sqrt(self.hd)
        )
        p = softmax(scores + mask.reshape(1, 1, S + 1).astype(np.float32))
        att = np.einsum("kgs,skd->kgd", p, Vall).reshape(self.H).astype(np.float32)
        x = x + lw["o"] @ att
        m = rmsnorm(x, lw["n2"], self.eps)
        x = x + lw["down"] @ (silu(lw["gate"] @ m) * (lw["up"] @ m))
        return x.astype(np.float32), k_row, v_row

    def post(self, x: np.ndarray, apply_norm: bool = True) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if apply_norm:
            x = rmsnorm(x, self.w["model.norm.weight"], self.eps)
        return (x @ self.lm_head.T).astype(np.float32)

    # ---- uncached (whole sequence, causal mask) -------------------------
    def forward_full(self, ids):
        T = len(ids)
        x = self.embed(ids)
        hid = [x]
        cos, sin = self.rope_cos_sin(np.arange(T))  # [T, hd]
        causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
        g = self.nh // self.nkv
        for lw in self.layers:
            a = rmsnorm(x, lw["n1"], self.eps)
            q = self._rot(
                (a @ lw["q"].T).reshape(T, self.nh, self.hd), cos[:, None], sin[:, None]
            )
            k = self._rot(
                (a @ lw["k"].T).reshape(T, self.nkv, self.hd),
                cos[:, None],
                sin[:, None],
            )
            v = (a @ lw["v"].T).reshape(T, self.nkv, self.hd)
            qg = q.reshape(T, self.nkv, g, self.hd)
            sc = np.einsum("tkgd,skd->kgts", qg, k) / np.float32(np.sqrt(self.hd))
            p = softmax(sc + causal)
            att = np.einsum("kgts,skd->tkgd", p, v).reshape(T, self.H)
            x = x + att @ lw["o"].T
            m = rmsnorm(x, lw["n2"], self.eps)
            x = x + (silu(m @ lw["gate"].T) * (m @ lw["up"].T)) @ lw["down"].T
            x = x.astype(np.float32)
            hid.append(x)
        return self.post(x), np.stack(hid)

    # ---- cached, one token at a time ------------------------------------
    def new_state(self):
        return {
            "K": [np.zeros((0, self.kvdim), np.float32) for _ in range(self.L)],
            "V": [np.zeros((0, self.kvdim), np.float32) for _ in range(self.L)],
            "pos": 0,
        }

    def step(
        self,
        state,
        token: int,
        rnd: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """Feed one token. Returns (logits [V], hiddens [L+1, H]).

        `rnd`, if given, is applied to every tensor that crosses a layer
        boundary in the compiled pipeline (embedding, layer output, K/V rows,
        logits) -- pass llm_layer_loop.bf16_round to model bf16 I/O exactly.
        """
        r = rnd if rnd is not None else (lambda a: a)
        x = r(self.E[int(token)])
        hid = [x]
        pos = state["pos"]
        for li in range(self.L):
            K, V = state["K"][li], state["V"][li]
            mask = np.zeros(K.shape[0] + 1, np.float32)
            x, k_row, v_row = self.layer_step(li, x, pos, K, V, mask)
            x, k_row, v_row = r(x), r(k_row), r(v_row)
            state["K"][li] = np.concatenate([K, k_row[None]], axis=0)
            state["V"][li] = np.concatenate([V, v_row[None]], axis=0)
            hid.append(x)
        state["pos"] = pos + 1
        return r(self.post(x)), np.stack(hid)

    def forward_cached(self, ids, rnd=None):
        st = self.new_state()
        logits, hids = [], []
        for t in ids:
            lg, h = self.step(st, t, rnd)
            logits.append(lg)
            hids.append(h)
        return np.stack(logits), np.stack(hids, axis=1)  # [T, V], [L+1, T, H]

    def greedy(
        self,
        prompt_ids,
        max_new_tokens: int,
        rnd=None,
        stop_at_eos: bool = False,
        trace: Optional[dict] = None,
    ):
        """Greedy generation, one token at a time with the KV cache.

        trace (optional dict) receives 'hiddens' [L+1, T_fed, H] and 'logits' [T_fed, V].
        """
        st = self.new_state()
        hs, lgs = [], []
        lg = None
        for t in prompt_ids:
            lg, h = self.step(st, t, rnd)
            hs.append(h)
            lgs.append(lg)
        out: list[int] = []
        for i in range(max_new_tokens):
            nxt = int(np.argmax(lg))
            out.append(nxt)
            if (stop_at_eos and nxt == self.eos) or i == max_new_tokens - 1:
                break
            lg, h = self.step(st, nxt, rnd)
            hs.append(h)
            lgs.append(lg)
        if trace is not None:
            trace["hiddens"] = np.stack(hs, axis=1)
            trace["logits"] = np.stack(lgs)
        return out


def _selftest(checkpoint_dir: str, use_torch: bool) -> None:
    m = Model.load(checkpoint_dir)
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
        ids = tok.encode("The capital of France is").ids
    except Exception:  # the tokenizer is a convenience only
        ids = [1, 2, 3, 4, 5]
    print("prompt ids", ids)
    lf, hf_ = m.forward_full(ids)
    lc, hc = m.forward_cached(ids)
    print(
        "cached vs uncached: max|dlogits| %.3e  max|dhidden| %.3e  (hidden scale %.2f)"
        % (np.abs(lf - lc).max(), np.abs(hf_ - hc).max(), np.abs(hf_).max())
    )
    assert (lf.argmax(-1) == lc.argmax(-1)).all()
    gen = m.greedy(ids, 16)
    print("greedy", gen)
    if not use_torch:
        return
    import torch
    from transformers import AutoModelForCausalLM

    tm = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        o = tm(torch.tensor([ids]), output_hidden_states=True)
        tl = o.logits[0].numpy()
        tg = tm.generate(torch.tensor([ids]), max_new_tokens=16, do_sample=False)
    tg = tg[0, len(ids) :].tolist()
    print(
        "vs transformers: max|dlogits| %.3e (logit scale %.1f)"
        % (np.abs(tl - lf).max(), np.abs(tl).max())
    )
    print("transformers greedy", tg, "identical:", tg == gen)
    assert tg == gen


if __name__ == "__main__":
    import sys

    _selftest(sys.argv[1], use_torch="--torch" in sys.argv)
