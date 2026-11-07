"""Float numpy reference of PLaMo 3 (``Plamo3ForCausalLM``), written from pfnet's modeling code.

Differences from a Qwen3/Llama decoder:

- four RMSNorms per layer; the two post-sublayer norms act on the attention and MLP outputs
  before each residual add. The RMSNorm gain is ``offset + weight`` with offsets 1, 1/5, 1
  and 1/5**1.5 (the post-MLP norm);
- q and k get a per-head RMSNorm (offset 1) before RoPE;
- full-attention layers (every ``sliding_window_pattern``-th) use RoPE base ``rope_theta``
  with YaRN scaling and cos/sin multiplied by the YaRN concentration; the other layers use
  ``rope_local_theta`` and no scaling;
- sliding-window attention (``window_size``) on the other layers. It is a no-op while the
  context is at most ``window_size`` tokens, so this reference does not implement the window.

No remote code is needed: the config and the weights are read from the checkpoint directory
(``config.json`` and the safetensors shards), and the forward is written out below.
"""

import json
import math
import os

import numpy as np

EPS = 1e-6


class DictWeights:
    """Weights from a name -> array mapping (tests and in-memory models)."""

    def __init__(self, tensors):
        self._tensors = {k: np.asarray(v, np.float32) for k, v in tensors.items()}

    def keys(self):
        return list(self._tensors)

    def f32(self, name):
        return self._tensors[name]


def load_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_weights(checkpoint_dir):
    """The checkpoint's safetensors weights as float32 (bf16 is widened exactly)."""
    import llm_reference as ref

    return ref.open_checkpoint_weights(checkpoint_dir)


class Plamo3Reference:
    """``cfg`` is the checkpoint's ``config.json`` as a dict; ``weights`` has ``f32(name)``."""

    def __init__(self, cfg, weights):
        self.c = cfg
        self.src = weights
        self.L = cfg["num_hidden_layers"]
        self.H = cfg["hidden_size"]
        self.I = cfg["intermediate_size"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.pattern = cfg["sliding_window_pattern"]
        self._cache = {}
        rp = cfg.get("rope_parameters", {})
        yarn = rp.get("full_attention", {})
        self.inv_local = self._inv_freq(cfg["rope_local_theta"], None, None)
        self.inv_full, self.conc = self._inv_freq(
            cfg["rope_theta"],
            cfg["rope_scaling_factor"],
            yarn if yarn.get("rope_type") == "yarn" else None,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir):
        return cls(load_config(checkpoint_dir), load_weights(checkpoint_dir))

    def w(self, name):
        if name not in self._cache:
            self._cache[name] = self.src.f32(name)
        return self._cache[name]

    def _inv_freq(self, base, factor, yarn):
        d = self.hd
        freq = base ** (np.arange(0, d, 2, dtype=np.float64) / d)
        if factor is None:
            return 1.0 / freq
        # the YaRN parameters come from the checkpoint's rope_parameters
        beta_fast = float(yarn["beta_fast"])
        beta_slow = float(yarn["beta_slow"])
        orig = float(yarn["original_max_position_embeddings"])
        dh = d // 2
        low = dh * math.log(orig / (beta_fast * 2 * math.pi)) / math.log(base)
        high = dh * math.log(orig / (beta_slow * 2 * math.pi)) / math.log(base)
        interp = 1.0 / (factor * freq)
        extrap = 1.0 / freq
        ramp = (np.arange(dh, dtype=np.float64) - low) / (high - low)
        mask = 1 - np.clip(ramp, 0, 1)
        inv = interp * (1 - mask) + extrap * mask
        conc = 0.1 * math.log(factor) + 1.0
        return inv, conc

    @staticmethod
    def rms(x, gain=None, offset=1.0):
        v = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        y = x * (1.0 / np.sqrt(v + EPS))
        if gain is not None:
            y = y * (offset + gain)
        return y.astype(np.float32)

    @staticmethod
    def rotate_half(x):
        h = x.shape[-1] // 2
        return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

    def rope(self, x, pos, full):
        inv = self.inv_full if full else self.inv_local
        conc = self.conc if full else 1.0
        freqs = np.outer(pos.astype(np.float64), inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos = (np.cos(emb) * conc).astype(np.float32)
        sin = (np.sin(emb) * conc).astype(np.float32)
        return x * cos[None, :, :] + self.rotate_half(x) * sin[None, :, :]

    def layer(self, i, h, capture=None):
        """One decoder layer over a whole sequence ``h`` [T, H]. ``capture`` (a dict) receives
        the layer's post-RoPE keys and values ``k``, ``v`` [nkv, T, hd]."""
        p = f"model.layers.layers.{i}."
        full = (i + 1) % self.pattern == 0
        T = h.shape[0]
        res = h
        x = self.rms(h, self.w(p + "pre_mixer_norm.weight"), 1.0)
        qkv = x @ self.w(p + "mixer.qkv_proj.weight").T
        q_dim, k_dim = self.nh * self.hd, self.nkv * self.hd
        q = qkv[:, :q_dim].reshape(T, self.nh, self.hd).transpose(1, 0, 2)
        k = (
            qkv[:, q_dim : q_dim + k_dim]
            .reshape(T, self.nkv, self.hd)
            .transpose(1, 0, 2)
        )
        v = qkv[:, q_dim + k_dim :].reshape(T, self.nkv, self.hd).transpose(1, 0, 2)
        q = self.rms(q, self.w(p + "mixer.q_norm.weight"), 1.0)
        k = self.rms(k, self.w(p + "mixer.k_norm.weight"), 1.0)
        pos = np.arange(T)
        q = self.rope(q, pos, full)
        k = self.rope(k, pos, full)
        if capture is not None:
            capture["k"] = k.copy()
            capture["v"] = v.copy()
        g = self.nh // self.nkv
        k = np.repeat(k, g, axis=0)
        v = np.repeat(v, g, axis=0)
        s = q @ k.transpose(0, 2, 1) / math.sqrt(self.hd)
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
        s = s + mask[None]
        s = s - s.max(-1, keepdims=True)
        pr = np.exp(s)
        pr = pr / pr.sum(-1, keepdims=True)
        att = (pr @ v).transpose(1, 0, 2).reshape(T, q_dim)
        o = att @ self.w(p + "mixer.o_proj.weight").T
        o = self.rms(o, self.w(p + "post_mixer_norm.weight"), 1.0 / 5)
        h = res + o
        res = h
        x = self.rms(h, self.w(p + "pre_mlp_norm.weight"), 1.0)
        gu = x @ self.w(p + "mlp.gate_up_proj.weight").T
        a = gu[:, : self.I] / (1 + np.exp(-gu[:, : self.I])) * gu[:, self.I :]
        m = a @ self.w(p + "mlp.down_proj.weight").T
        m = self.rms(m, self.w(p + "post_mlp_norm.weight"), 1.0 / 5**1.5)
        return res + m

    def logits(self, ids):
        h = self.w("model.embed_tokens.weight")[np.asarray(ids)].astype(np.float32)
        for i in range(self.L):
            h = self.layer(i, h)
        h = self.rms(h, self.w("model.norm.weight"), 1.0)
        return h @ self.w("lm_head.weight").T

    def greedy(self, ids, n, eos=(1,)):
        ids = list(ids)
        for _ in range(n):
            nxt = int(np.argmax(self.logits(ids)[-1]))
            ids.append(nxt)
            if nxt in eos:
                break
        return ids
