"""The model side of the Pulsar-free U16 decode: which checkpoint, its sizes,
the float reference of one decode step, and the weights / tables each program
is emitted from or fed with.

``load()`` picks the model from the environment:

* ``MODEL``: a key of ``MODELS`` (``llama160m``, default, or ``qwen3_0.6b``);
* ``LLM_PF_SNAPSHOT``: a local checkpoint directory (default: the pinned
  Hugging Face revision, fetched with ``huggingface_hub.snapshot_download``);
* ``LLM_PF_OUT``: the output directory of the pipeline (default
  ``llm_pulsar_free_out/<MODEL>`` under the current directory);
* ``GPTQ=1``: linear programs are emitted from ``<out>/gptq/*.npz``
  (``pf_gptq.py``) instead of round-to-nearest;
* ``LLM_PF_FUSE``: the program layout (``pf_layout``), default the model's
  ``fuse``: ``0`` one program per step (``LAYER``, 15 executes per Qwen3
  layer), ``1`` residual Add + two-pass norm and q / k RoPE + k norm merged
  (``LAYER_FUSED``, 9), ``2`` also ``o_proj`` / ``down_proj`` into the Add +
  norm after them (``LAYER_FUSED2``, 7), ``3`` also the SiLU before
  ``down_proj`` (``LAYER_FUSED3``, 6), ``4`` also the linear after each norm
  (``LAYER_FUSED4``, 4), ``5`` also the next layer's q / k / v linears and
  RoPE into ``dn`` (``LAYER_FUSED5``, 3), ``6`` the same programs with
  rotate_half made in ``dnqk`` / ``n1qk`` (three linears instead of five),
  ``7`` also the attention into the program after it (``LAYER_FUSED7``, 2;
  without a per-head q / k norm layout 3's ``attn`` + ``on2``, 6), ``8`` the
  whole layer as one program (``LAYER_FUSED8``, 1; Qwen3 only).
  Qwen3-0.6B defaults to 5, llama-160m (no per-head q / k norm: ``4`` to
  ``6`` and ``8`` do not apply, ``7`` is layout 3 with ``aon2``) to 3.

Calibration sets: ``first`` is position ``first_pos`` with many different
tokens (the position whose activations are massive), ``rest`` the positions
after it. llama-160m: the BOS embedding is all zeros, so position 0 is zero
end to end (zero K / V rows, zero logits) and is skipped; position 1 carries
the massive activations (residual ~1.7e3-2e3 at layers 3..10 for common first
tokens vs <= ~300 later). Qwen3-0.6B: no BOS; position 0 (any first token)
carries them (residual ~6.4e3-7e3 from layer ~4 on, <= ~400 later).
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(AXERA))
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_layer_decomposition as ld  # noqa: E402
import llm_reference as ref  # noqa: E402
import pf_fake_quant as fqm  # noqa: E402
import pf_layout as lay  # noqa: E402
import rope_rotate_half as rr  # noqa: E402

F32 = np.float32

MODELS = {
    "llama160m": dict(
        repo="JackFram/llama-160m",
        revision="aca9b687d1425f863dcf5de9a4c96e3fe36266dd",
        first_pos=1,
        zero_pos0=True,
        fuse=3,
    ),
    # Per-head q / k RMSNorm before RoPE (qk_norm). The linears are shared by
    # both calibration sets (one copy of the weights on the card; calibration =
    # the union) when position 0's ranges are within SHARE_MAX of the rest's.
    # Position 0 creates the massive activation through the first layers' MLP
    # linears: s8 weights x a huge activation put K / V row 0 (the attention
    # sink) off, so their first-set programs get ~16-bit weights as two s8
    # linears (W ~ Q8 + R8) and an Add.
    "qwen3_0.6b": dict(
        repo="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        first_pos=0,
        zero_pos0=False,
        share_linears=True,
        split_first=dict(layers=[0, 1, 2], keys=["gu", "down_proj"]),
        fuse=5,
    ),
}
#: share a linear between the sets when the first set's ranges are within this factor
SHARE_MAX = 2.0


def snapshot(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )


class Config:
    """One model of the pipeline: ``model`` (an ``llm_reference.Model``), the
    calibration-set positions, and where the pipeline's files go (``out``)."""

    def __init__(
        self,
        model: ref.Model,
        name: str = "custom",
        out: str = ".",
        first_pos: int = 0,
        zero_pos0: bool = False,
        share_linears: bool = False,
        split_first: dict | None = None,
        gptq: bool = False,
        tokenizer_dir: str | None = None,
        seq: int = lay.S,
        fuse: int = 0,
    ):
        self.model, self.name, self.out = model, name, out
        self.L, self.H, self.hd = model.L, model.H, model.hd
        self.nh, self.nkv, self.kvd = model.nh, model.nkv, model.kvdim
        self.I = model.layer_weights(0)["gate"].shape[0]  # noqa: E741
        self.G = self.nh // self.nkv
        self.S = seq
        self.first_pos, self.zero_pos0 = first_pos, zero_pos0
        self.share_linears = share_linears
        self.split_first = split_first or dict(layers=[], keys=[])
        self.gptq = gptq
        self.tokenizer_dir = tokenizer_dir
        self.qk_norm = bool(model.qk_norm)
        self.eps = float(model.eps)
        self.ln_f = model.weight("model.norm.weight").astype(F32).reshape(1, self.H)
        self.QKV = lay.qkv_parts(self.nh, self.hd, self.kvd, self.qk_norm)
        self.GU = lay.gu_parts(self.I)
        self.fuse = int(fuse)
        self.layer_programs = lay.layer_programs(self.qk_norm, self.fuse)
        self.post_programs = lay.post_programs(self.fuse)
        self.k_program = lay.k_program(self.qk_norm, self.fuse)
        self._lg: dict = {}
        self._post = ReferenceEvaluator(
            ld.make_graph(
                "post_norm",
                {"input": (1, self.H)},
                {"eps": np.array([self.eps], F32), "ln_f": self.ln_f.reshape(-1)},
                ld.rmsnorm_lines("nf", "input", "ln_f", "xn"),
                ["xn"],
            )
        )
        self._tok = None

    @functools.cached_property
    def head_t(self) -> np.ndarray:
        """The head weights ``[H, vocab]`` (computed on first use: a strided copy
        of 0.6 GB for Qwen3, which ``pf_generate.py`` with GPTQ never needs)."""
        m = self.model
        return np.ascontiguousarray(m.weight(m._head).T.astype(F32))

    @functools.cached_property
    def vocab(self) -> int:
        raw = getattr(getattr(self.model, "_w", None), "raw", None)
        if raw is not None:
            return int(raw(self.model._head).shape[0])
        return int(self.head_t.shape[1])

    # ---- float reference ------------------------------------------------------------
    def mask_ref(self, p):
        return lay.mask_ref(p, self.S)

    def mask_plan(self, p):
        return lay.mask_plan(p, self.S)

    def new_cache(self):
        z = [np.zeros((self.S, self.kvd), F32) for _ in range(self.L)]
        return z, [a.copy() for a in z]

    def float_layer(self, li, x, K, V, p) -> dict:
        if li not in self._lg:
            self._lg[li] = ReferenceEvaluator(ld.layer_graph(self.model, li, self.S))
        feeds = {
            "input": np.asarray(x, F32).reshape(1, self.H),
            "K_cache": K,
            "V_cache": V,
            "mask": self.mask_ref(p),
            **ld.rope_inputs(self.model, p),
        }
        t = self._lg[li].run(None, feeds, intermediate=True)
        return {k: np.asarray(v, F32) for k, v in t.items()}

    def float_step(self, Ks, Vs, token, p, record=None):
        """Float decode step (all layers + post); writes slot ``p`` of every cache;
        ``record(li, t)`` sees each layer's intermediates. Returns (logits [V],
        post intermediates incl. ``input``)."""
        x = self.model.embed([int(token)]).reshape(1, self.H).astype(F32)
        for li in range(self.L):
            t = self.float_layer(li, x, Ks[li], Vs[li], p)
            if record:
                record(li, t)
            Ks[li][p] = t["K_cache_out"][self.S]
            Vs[li][p] = t["V_cache_out"][self.S]
            x = t["output"].reshape(1, self.H)
        pn = self._post.run(None, {"input": x}, intermediate=True)
        pn = {k: np.asarray(v, F32) for k, v in pn.items()}
        pn["input"] = x
        return (pn["xn"].reshape(1, self.H) @ self.head_t).reshape(-1), pn

    def kv_producer(self, st: str, li: int) -> tuple[str, str]:
        """``(where, program)`` that computes layer ``li``'s K row (and, with
        ``fuse>=5``, its V row: ``yk`` and ``v``)."""
        if self.fuse < 5 or not self.qk_norm:
            return f"L{li}", self.k_program[0]
        if li == 0:
            return "L0", "n1qk"
        if self.is_split(st, li - 1, "down_proj"):
            return f"L{li}", "qk"
        return f"L{li - 1}", "alyr" if self.fuse >= 8 else "dnqk"

    def programs(self, st: str, wh: str) -> list[str]:
        """The programs of set ``st`` at ``wh`` (``L<i>`` or ``post``), in order."""
        if wh == "post":
            return list(self.post_programs)
        progs = list(self.layer_programs)
        if self.fuse >= 4 and self.qk_norm:
            li = int(wh[1:])
            five = self.fuse >= 5
            split = self.is_split(st, li, "down_proj")
            head = ["n1qk" if five else "n1q"] if li == 0 else []
            if li > 0 and self.is_split(st, li - 1, "down_proj"):
                head = ["qkv5", "qk"] if five else ["qkv5"]
            dnq = "dnqk" if five else "dnq"
            tail = ["on2g", dnq if li + 1 < self.L else "dn"]
            attn = ["attn"]
            if self.fuse >= 7:  # the attention joins on2g
                attn, tail[0] = [], "aon2g"
            if self.fuse >= 8 and li + 1 < self.L:  # the whole layer
                tail = ["alyr"]
            if split:
                attn, tail = ["attn"], ["on2", "gu", "silu_f", "down_proj", "dn"]
            return head + ([] if five else ["qk"]) + attn + tail
        if self.fuse >= 2:
            li = int(wh[1:])
            if li == 0:
                progs.insert(0, "n1")
            if self.is_split(st, li, "down_proj"):
                if "aon2" in progs:  # fuse=7: the split layer keeps attn and on2
                    i = progs.index("aon2")
                    progs[i : i + 1] = ["attn", "on2"]
                split = ["silu_f", "down_proj"] if self.fuse >= 3 else ["down_proj"]
                i = progs.index("dn")
                progs[i:i] = split
        return progs

    # ---- program inputs and weights ---------------------------------------------------
    def gains(self, li):
        lw = self.model.layer_weights(li)
        return (
            lw["n1"].astype(F32).reshape(1, self.H),
            lw["n2"].astype(F32).reshape(1, self.H),
        )

    def rot(self, x, heads):
        """rotate_half's permutation applied per head of ``hd``."""
        perm = np.kron(np.eye(heads, dtype=F32), rr.perm_matrix(self.hd))
        return (x.reshape(-1, heads * self.hd) @ perm).reshape(x.shape).astype(F32)

    def rope_tiles(self, p, li=None) -> dict:
        """The position's RoPE tables tiled over the heads. With qk_norm (Qwen3)
        the layer's q / k norm gain is folded in (c = g * cos, sn = rot(g) * sin):
        the fused ``qkrope`` program then computes
        (q*c + rot(q)*sn) / rms_head(q) = RoPE(q / rms_head(q) * g) exactly (rot
        permutes within a head)."""
        r = ld.rope_inputs(self.model, p)

        def tile(v, n):
            return np.tile(np.asarray(v, F32).reshape(1, self.hd), (1, n))

        nh, nkv = self.nh, self.nkv
        out = dict(
            cos_q=tile(r["cos_q"], nh),
            sin_q=tile(r["sinq"], nh),
            cos_k=tile(r["cos_k"], nkv),
            sin_k=tile(r["sink"], nkv),
        )
        if self.qk_norm:
            lw = self.model.layer_weights(li)
            P = rr.perm_matrix(self.hd)
            for key, g, n in (("q", lw["qn"], nh), ("k", lw["kn"], nkv)):
                g = np.asarray(g, F32).reshape(1, self.hd)
                out[f"cos_{key}"] = (out[f"cos_{key}"] * tile(g, n)).astype(F32)
                out[f"sin_{key}"] = (out[f"sin_{key}"] * tile(g @ P, n)).astype(F32)
        return out

    def linear_weight(self, name, li=None):
        if name == "lm_head":
            return self.head_t
        lw = self.model.layer_weights(li)
        key = {
            "q_proj": "q",
            "k_proj": "k",
            "v_proj": "v",
            "q_rot": "q",
            "k_rot": "k",
            "o_proj": "o",
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }[name]
        w = np.ascontiguousarray(lw[key].T.astype(F32))
        if name in (
            "q_rot",
            "k_rot",
        ):  # rotate_half folded in: a column permutation per head
            heads = w.shape[1] // self.hd
            perm = np.kron(np.eye(heads, dtype=F32), rr.perm_matrix(self.hd))
            w = (w @ perm).astype(F32)
        return w

    def merged_weight(self, key, li=None, gptq=True):
        """The float weights a linear program is emitted from: the ``QKV`` parts
        for ``qkv5``, gate | up for ``gu``, the plain matrix otherwise. With
        ``self.gptq`` and ``gptq``: ``pf_gptq.py``'s rounding on the card's s8 grid."""
        if gptq and self.gptq:
            wh = "post" if li is None else f"L{li}"
            return np.load(os.path.join(self.out, "gptq", f"{wh}.npz"))[key]
        if key == "qkv5":
            return np.concatenate(
                [self.linear_weight(p, li) for p, _ in self.QKV], axis=1
            )
        if key == "gu":
            return np.concatenate(
                [self.linear_weight(p, li) for p, _ in self.GU], axis=1
            )
        return self.linear_weight(key, li)

    def is_split(self, st, li, key) -> bool:
        return (
            st == "first"
            and li in self.split_first["layers"]
            and key in self.split_first["keys"]
        )

    @staticmethod
    def split_weights(w):
        """(Q, R): Q the card's s8 per-output-channel weights of w, dequantized;
        R = w - Q (emitted as a second s8 linear: together ~16-bit weights)."""
        q = fqm.s8_weights(w).astype(F32)
        return q, (np.asarray(w, F32) - q).astype(F32)

    # ---- text ---------------------------------------------------------------------------
    def tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(
                os.path.join(self.tokenizer_dir, "tokenizer.json")
            )
        return self._tok

    def passage_ids(self, doc, n=None):
        """The first ``n`` (default ``S``) tokens of a repo doc's running text."""
        n = n or self.S
        ids = self.tokenizer().encode(lay.prose(os.path.join(REPO, doc))).ids
        if len(ids) < n:
            raise ValueError(f"{doc}: {len(ids)} tokens < {n}")
        return ids[:n]


def load(name: str | None = None, out: str | None = None) -> Config:
    """The ``Config`` the environment selects (see the module docstring)."""
    name = name or os.environ.get("MODEL", "llama160m")
    if name not in MODELS:
        raise ValueError(f"MODEL={name!r}: one of {', '.join(MODELS)}")
    spec = dict(MODELS[name])
    repo, revision = spec.pop("repo"), spec.pop("revision")
    snap = os.environ.get("LLM_PF_SNAPSHOT") or snapshot(repo, revision)
    out = (
        out or os.environ.get("LLM_PF_OUT") or os.path.join("llm_pulsar_free_out", name)
    )
    os.makedirs(out, exist_ok=True)
    if os.environ.get("LLM_PF_FUSE"):
        spec["fuse"] = int(os.environ["LLM_PF_FUSE"])
    return Config(
        ref.Model.load(snap),
        name=name,
        out=out,
        gptq=os.environ.get("GPTQ") == "1",
        tokenizer_dir=snap,
        **spec,
    )
