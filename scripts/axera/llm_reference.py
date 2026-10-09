"""Numpy float32 reference forward of an HF llama-family checkpoint.

Config-driven: every size comes from `config.json`, every optional piece from
what the safetensors file contains. Checked against transformers for
`model_type` "llama" (SmolLM2-135M, JackFram/llama-160m) and "qwen3"
(Qwen3-0.6B); see docs/axera-llm-rpc-decode.md.

No torch needed: the safetensors file is parsed by hand (8-byte little-endian
header length, JSON header, raw little-endian tensor bytes) and memory-mapped.
A tensor becomes float32 only when it is used: F32 tensors are used straight
from the mapping (no copy), BF16/F16 ones are widened exactly and kept in a
least-recently-used cache bounded by `cache_bytes` (default 3 GiB, enough to
hold all of a 0.6B-parameter model, i.e. about 2.4 GB once every layer has
run; `cache_bytes=0` converts on every use and holds nothing).

Layer math (HF Llama / Qwen3, pre-norm), nh heads, nkv key/value heads,
hd = `head_dim` when the config has one, else hidden / nh:
    a   = rmsnorm(x, input_layernorm)
    q,k,v = a @ Wq.T, a @ Wk.T, a @ Wv.T          (+ bias when the file has one)
            q is [nh, hd], k and v are [nkv, hd]; nh * hd need not equal hidden
    qwen3 only: q = rmsnorm(q, q_norm), k = rmsnorm(k, k_norm), per head over hd
    q,k rotated by RoPE at the token's absolute position (theta = `rope_theta`)
    attn = softmax(q.k / sqrt(hd) + mask) . v     (grouped-query: head h uses kv head h // (nh / nkv))
    x   = x + attn @ Wo.T
    m   = rmsnorm(x, post_attention_layernorm)
    x   = x + (silu(m @ Wgate.T) * (m @ Wup.T)) @ Wdown.T
Final: logits = rmsnorm(x, model.norm) @ lm_head.T   (`lm_head.weight` if the file
has it, else the embedding when `tie_word_embeddings`)

Public surface:
    Model.load(checkpoint_dir, cache_bytes=...)
    model.embed(ids)                               -> [T, H] float32
    model.layer_step(li, x, pos, K, V, mask)       -> (y, k_row, v_row)   one token, explicit cache
    model.forward_full(ids)                        -> (logits [T, V], hiddens [L+1, T, H])  uncached
    model.forward_cached(ids)                      -> (logits [T, V], hiddens [L+1, T, H])  one token at a time
    model.greedy(prompt_ids, max_new_tokens)       -> list[int]
    model.post(x, apply_norm=True)                 -> logits

`hiddens[0]` is the embedding, `hiddens[l + 1]` is the output of layer `l`
(before the final norm), i.e. exactly what the compiled layer `l` should
return in its `output` tensor.

`python llm_reference.py CHECKPOINT_DIR [--torch] [--prompt-ids 1,2,3] [--new N]`
checks the cached forward against the uncached one, and with --torch against
transformers on CPU (which loads the whole model as float32).
"""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np

_ST_DTYPES = {
    "F32": "<f4",
    "F16": "<f2",
    "F64": "<f8",
    "I64": "<i8",
    "I32": "<i4",
    "BF16": "<u2",  # bit patterns; widened by `widen`
}

DEFAULT_CACHE_BYTES = 3 << 30


def widen(arr: np.ndarray, kind: str) -> np.ndarray:
    """A stored tensor (or a slice of one) as float32. BF16/F16 widen exactly;
    F32 is returned as it is, without a copy."""
    if kind == "BF16":
        out = np.empty(arr.shape, "<u4")
        out[...] = arr
        out <<= 16
        return out.view("<f4")
    if kind == "F32":
        return arr
    return arr.astype(np.float32)


class SafeTensors:
    """A .safetensors file, memory-mapped. Nothing is read until a tensor is used."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        self._base = 8 + n
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        # a plain read-only ndarray over the mapping, so results are not np.memmap instances
        self._map = np.asarray(np.memmap(path, dtype=np.uint8, mode="r"))

    def __contains__(self, name: str) -> bool:
        return name in self.meta

    def keys(self):
        return self.meta.keys()

    def kind(self, name: str) -> str:
        return self.meta[name]["dtype"]

    def raw(self, name: str) -> np.ndarray:
        """The stored tensor, a read-only view of the mapping (BF16 as uint16 bit patterns)."""
        info = self.meta[name]
        beg, end = info["data_offsets"]
        flat = self._map[self._base + beg : self._base + end]
        return flat.view(_ST_DTYPES[info["dtype"]]).reshape(info["shape"])

    def f32(self, name: str) -> np.ndarray:
        return widen(self.raw(name), self.kind(name))


class _DictWeights:
    """The same interface over a plain {name: float32 array} dict."""

    def __init__(self, tensors: dict):
        self._t = tensors

    def __contains__(self, name):
        return name in self._t

    def keys(self):
        return self._t.keys()

    def kind(self, name):
        return "F32"

    def raw(self, name):
        return self._t[name]

    def f32(self, name):
        return np.asarray(self._t[name], dtype=np.float32)


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """Every tensor of a .safetensors file as float32, in memory (BF16/F16 widened
    exactly). `SafeTensors` is the memory-mapped, tensor-at-a-time form."""
    st = SafeTensors(path)
    return {
        name: np.ascontiguousarray(st.f32(name), dtype=np.float32) for name in st.keys()
    }


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(var + np.float32(eps))) * w


def silu(x: np.ndarray) -> np.ndarray:
    with np.errstate(
        over="ignore"
    ):  # exp overflows to inf for very negative x: x / inf = -0.0
        return x / (np.float32(1.0) + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# Per-layer tensors this reference implements; anything else in a layer is refused.
_LAYER_TENSORS = {
    "input_layernorm.weight": "n1",
    "self_attn.q_proj.weight": "q",
    "self_attn.k_proj.weight": "k",
    "self_attn.v_proj.weight": "v",
    "self_attn.o_proj.weight": "o",
    "post_attention_layernorm.weight": "n2",
    "mlp.gate_proj.weight": "gate",
    "mlp.up_proj.weight": "up",
    "mlp.down_proj.weight": "down",
}
_LAYER_OPTIONAL = {
    "self_attn.q_norm.weight": "qn",  # qwen3: RMSNorm over head_dim, before RoPE
    "self_attn.k_norm.weight": "kn",
    "self_attn.q_proj.bias": "qb",
    "self_attn.k_proj.bias": "kb",
    "self_attn.v_proj.bias": "vb",
    "self_attn.o_proj.bias": "ob",
    "mlp.gate_proj.bias": "gateb",
    "mlp.up_proj.bias": "upb",
    "mlp.down_proj.bias": "downb",
}
# Buffers some old checkpoints store per layer; not weights (inv_freq is checked against the config's).
_LAYER_BUFFERS = {"self_attn.rotary_emb.inv_freq"}
_EMBED = "model.embed_tokens.weight"


def _lin(w: np.ndarray, b: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
    """x @ w.T (+ b) for x of shape [..., in]."""
    y = x @ w.T
    return y if b is None else y + b


class Model:
    def __init__(self, cfg: dict, w, cache_bytes: int = DEFAULT_CACHE_BYTES):
        self.cfg = cfg
        self._w = _DictWeights(w) if isinstance(w, dict) else w
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict = OrderedDict()
        self._cached = 0
        self.model_type = cfg.get("model_type", "llama")
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg.get("num_key_value_heads") or self.nh
        self.hd = cfg.get("head_dim") or self.H // self.nh
        self.qdim = self.nh * self.hd  # width of q_proj's output / o_proj's input
        self.kvdim = self.nkv * self.hd
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.vocab = cfg["vocab_size"]
        self.eos = cfg.get("eos_token_id")
        if isinstance(self.eos, (list, tuple)):
            self.eos_ids = tuple(int(e) for e in self.eos)
            self.eos = self.eos_ids[0] if self.eos_ids else None
        else:
            self.eos_ids = () if self.eos is None else (int(self.eos),)
        self.bos = cfg.get("bos_token_id")
        if self.nh % self.nkv:
            raise ValueError(
                f"{self.nh} heads are not a multiple of {self.nkv} kv heads"
            )
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError(
                f"hidden_act {cfg['hidden_act']!r}: only silu is implemented"
            )
        if cfg.get("rope_scaling"):
            raise ValueError("rope_scaling is not implemented")
        self._head = "lm_head.weight" if "lm_head.weight" in self._w else _EMBED
        if self._head == _EMBED and not cfg.get("tie_word_embeddings", False):
            raise ValueError("no lm_head.weight and tie_word_embeddings is false")
        # HF: inv_freq = 1 / theta ** (arange(0, d, 2) / d); computed in float64, used as float32.
        # LlamaConfig's default theta is 10000 (llama-160m's config has no rope_theta).
        self.rope_theta = float(cfg.get("rope_theta", 10000.0))
        self.inv_freq = 1.0 / (
            self.rope_theta ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        )
        p0 = "model.layers.0."
        have = {n[len(p0) :] for n in self._w.keys() if n.startswith(p0)}
        missing = set(_LAYER_TENSORS) - have
        unknown = have - set(_LAYER_TENSORS) - set(_LAYER_OPTIONAL) - _LAYER_BUFFERS
        if missing or unknown:
            raise ValueError(
                f"layer tensors: missing {sorted(missing)}, not implemented {sorted(unknown)}"
            )
        self._layer_names = {
            **_LAYER_TENSORS,
            **{n: k for n, k in _LAYER_OPTIONAL.items() if n in have},
        }
        self.qk_norm = "self_attn.q_norm.weight" in have
        if "self_attn.rotary_emb.inv_freq" in have:
            stored = self._w.f32(p0 + "self_attn.rotary_emb.inv_freq")
            if stored.shape != self.inv_freq.shape or not np.allclose(
                stored, self.inv_freq, rtol=1e-5
            ):
                raise ValueError(
                    "the checkpoint's stored RoPE inv_freq differs from rope_theta's"
                )
        if self.model_type == "qwen3" and not self.qk_norm:
            raise ValueError("model_type qwen3 without q_norm/k_norm weights")
        q_shape = tuple(self._w.raw(p0 + "self_attn.q_proj.weight").shape)
        k_shape = tuple(self._w.raw(p0 + "self_attn.k_proj.weight").shape)
        if q_shape != (self.qdim, self.H) or k_shape != (self.kvdim, self.H):
            raise ValueError(
                f"q_proj {q_shape} / k_proj {k_shape} do not match the config "
                f"({self.nh} heads, {self.nkv} kv heads, head_dim {self.hd}, hidden {self.H})"
            )

    @classmethod
    def load(
        cls, checkpoint_dir: str, cache_bytes: int = DEFAULT_CACHE_BYTES
    ) -> "Model":
        with open(os.path.join(checkpoint_dir, "config.json")) as f:
            cfg = json.load(f)
        m = cls(
            cfg,
            SafeTensors(os.path.join(checkpoint_dir, "model.safetensors")),
            cache_bytes,
        )
        # generation_config.json may list more stop tokens than config.json (qwen3: <|im_end|>, <|endoftext|>).
        try:
            with open(os.path.join(checkpoint_dir, "generation_config.json")) as f:
                eos = json.load(f).get("eos_token_id")
        except (OSError, ValueError):
            eos = None
        if eos is not None:
            extra = eos if isinstance(eos, list) else [eos]
            m.eos_ids = tuple(dict.fromkeys([*m.eos_ids, *(int(e) for e in extra)]))
            if m.eos is None:
                m.eos = m.eos_ids[0]
        return m

    # ---- weights --------------------------------------------------------
    def weight(self, name: str) -> np.ndarray:
        """A tensor as float32: a view of the file for F32, else widened and kept in the LRU."""
        c = self._cache
        if name in c:
            c.move_to_end(name)
            return c[name][0]
        a = self._w.f32(name)
        size = 0 if self._w.kind(name) == "F32" else a.nbytes
        if size <= self.cache_bytes:
            while self._cached + size > self.cache_bytes and c:
                self._cached -= c.popitem(last=False)[1][1]
            c[name] = (a, size)
            self._cached += size
        return a

    def layer_weights(self, li: int) -> dict:
        p = f"model.layers.{li}."
        lw = {key: self.weight(p + name) for name, key in self._layer_names.items()}
        for key in _LAYER_OPTIONAL.values():
            lw.setdefault(key, None)
        return lw

    @property
    def E(self) -> np.ndarray:
        """The whole embedding table as float32 [vocab, hidden]."""
        return self.weight(_EMBED)

    def embedding_bits(self) -> Optional[np.ndarray]:
        """The embedding table as stored bf16 bit patterns (uint16 [vocab, hidden], a view
        of the file), or None when the checkpoint does not store it as BF16."""
        return self._w.raw(_EMBED) if self._w.kind(_EMBED) == "BF16" else None

    # ---- pieces ---------------------------------------------------------
    def embed(self, ids) -> np.ndarray:
        """Embedding rows as float32; only the rows asked for are read and widened."""
        rows = self._w.raw(_EMBED)[np.asarray(ids, dtype=np.int64)]
        return np.asarray(widen(rows, self._w.kind(_EMBED)), dtype=np.float32)

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

    def _qkv(self, lw: dict, a: np.ndarray, cos: np.ndarray, sin: np.ndarray):
        """a [..., H] (already normed) -> q [..., nh, hd] and k [..., nkv, hd], both
        (q/k-normed and) rotated, and v [..., nkv, hd]. cos/sin broadcast against [..., heads, hd]."""
        lead = a.shape[:-1]
        q = _lin(lw["q"], lw["qb"], a).reshape(*lead, self.nh, self.hd)
        k = _lin(lw["k"], lw["kb"], a).reshape(*lead, self.nkv, self.hd)
        v = _lin(lw["v"], lw["vb"], a).reshape(*lead, self.nkv, self.hd)
        if lw["qn"] is not None:
            q = rmsnorm(q, lw["qn"], self.eps)
            k = rmsnorm(k, lw["kn"], self.eps)
        return (
            self._rot(q, cos, sin).astype(np.float32),
            self._rot(k, cos, sin).astype(np.float32),
            v.astype(np.float32),
        )

    def _mlp(self, lw: dict, x: np.ndarray) -> np.ndarray:
        m = rmsnorm(x, lw["n2"], self.eps)
        act = silu(_lin(lw["gate"], lw["gateb"], m)) * _lin(lw["up"], lw["upb"], m)
        return _lin(lw["down"], lw["downb"], act)

    def qkv_row(self, li: int, x: np.ndarray, pos: int):
        """One token. Returns (q [nh, hd] rotated, k_row [nkv*hd] rotated, v_row [nkv*hd])."""
        lw = self.layer_weights(li)
        a = rmsnorm(x.reshape(self.H), lw["n1"], self.eps)
        cos, sin = self.rope_cos_sin(pos)
        q, k, v = self._qkv(lw, a, cos, sin)
        return q, k.reshape(-1), v.reshape(-1)

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
        lw = self.layer_weights(li)
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
        att = np.einsum("kgs,skd->kgd", p, Vall).reshape(self.qdim).astype(np.float32)
        x = x + _lin(lw["o"], lw["ob"], att)
        x = x + self._mlp(lw, x)
        return x.astype(np.float32), k_row, v_row

    def post(self, x: np.ndarray, apply_norm: bool = True) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if apply_norm:
            x = rmsnorm(x, self.weight("model.norm.weight"), self.eps)
        return (x @ self.weight(self._head).T).astype(np.float32)

    # ---- uncached (whole sequence, causal mask) -------------------------
    def forward_full(self, ids):
        T = len(ids)
        x = self.embed(ids)
        hid = [x]
        cos, sin = self.rope_cos_sin(np.arange(T))  # [T, hd]
        causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
        g = self.nh // self.nkv
        for li in range(self.L):
            lw = self.layer_weights(li)
            a = rmsnorm(x, lw["n1"], self.eps)
            q, k, v = self._qkv(lw, a, cos[:, None], sin[:, None])
            qg = q.reshape(T, self.nkv, g, self.hd)
            sc = np.einsum("tkgd,skd->kgts", qg, k) / np.float32(np.sqrt(self.hd))
            p = softmax(sc + causal)
            att = np.einsum("kgts,skd->tkgd", p, v).reshape(T, self.qdim)
            x = x + _lin(lw["o"], lw["ob"], att)
            x = (x + self._mlp(lw, x)).astype(np.float32)
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
        x = r(self.embed(int(token)))
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
            if (stop_at_eos and nxt in self.eos_ids) or i == max_new_tokens - 1:
                break
            lg, h = self.step(st, nxt, rnd)
            hs.append(h)
            lgs.append(lg)
        if trace is not None:
            trace["hiddens"] = np.stack(hs, axis=1)
            trace["logits"] = np.stack(lgs)
        return out


def _selftest(
    checkpoint_dir: str, use_torch: bool, ids: Optional[list] = None, new: int = 16
) -> None:
    m = Model.load(checkpoint_dir)
    print(
        "%s: hidden %d, %d layers, %d heads / %d kv heads x head_dim %d, vocab %d, rope_theta %g, "
        "q/k norm %s, lm_head %s"
        % (
            m.model_type,
            m.H,
            m.L,
            m.nh,
            m.nkv,
            m.hd,
            m.vocab,
            m.rope_theta,
            m.qk_norm,
            "tied (the embedding)" if m._head == _EMBED else "its own tensor",
        )
    )
    if ids is None:
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
    gen = m.greedy(ids, new)
    print("greedy", gen)
    if not use_torch:
        return
    import torch
    from transformers import AutoModelForCausalLM

    tm = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, dtype=torch.float32
    ).eval()
    with torch.no_grad():
        o = tm(torch.tensor([ids]), output_hidden_states=True)
        tl = o.logits[0].numpy()
        th = torch.stack(o.hidden_states)[
            :, 0
        ].numpy()  # [L+1, T, H]; the last one is post-norm
        tg = tm.generate(torch.tensor([ids]), max_new_tokens=new, do_sample=False)
    tg = tg[0, len(ids) :].tolist()
    print(
        "vs transformers: max|dlogits| %.3e (logit scale %.1f)  max|dhidden| %.3e (layers 0..L-1 outputs)"
        % (np.abs(tl - lf).max(), np.abs(tl).max(), np.abs(th[:-1] - hf_[:-1]).max())
    )
    print("transformers greedy", tg, "identical:", tg == gen)
    assert tg == gen


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="self-check of the numpy reference")
    _ap.add_argument("checkpoint")
    _ap.add_argument(
        "--torch", action="store_true", help="also compare with transformers on CPU"
    )
    _ap.add_argument(
        "--prompt-ids", help="comma-separated token ids (no tokenizer needed)"
    )
    _ap.add_argument("--new", type=int, default=16, help="greedy tokens to generate")
    _a = _ap.parse_args()
    _selftest(
        _a.checkpoint,
        _a.torch,
        [int(x) for x in _a.prompt_ids.split(",")] if _a.prompt_ids else None,
        _a.new,
    )
