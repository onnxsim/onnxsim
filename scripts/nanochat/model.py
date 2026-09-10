"""Standalone, ONNX-exportable reimplementation of nanochat's core GPT.

nanochat (https://github.com/karpathy/nanochat) is Andrej Karpathy's
from-scratch "best ChatGPT you can train for ~$100" pipeline: tokenizer,
pretraining, midtraining/SFT, RL, and a chat web UI, all built around a
from-scratch GPT defined in ``nanochat/gpt.py``. Its own docstring lists the
notable features: rotary embeddings (no positional embeddings), QK norm,
untied token-embedding/lm_head weights, relu^2 activation in the MLP, a norm
after the token embedding, no learnable params in RMSNorm, no bias in linear
layers, and Group-Query Attention (GQA).

nanochat has no ONNX export path of its own, and its attention
(``nanochat/flash_attention.py``) is Flash Attention 3 -- a custom CUDA
kernel that does not trace through ``torch.onnx.export`` -- combined with a
sliding-window pattern across layers that plain SDPA does not support either
(nanochat's own ``scripts/base_train.py`` warns about exactly this: "SDPA has
no support for sliding window attention... Recommend using
--window-pattern L").

This module reimplements the architecture above -- same rotary/QK-norm/
untied-embedding/relu^2/GQA/no-bias/unlearned-RMSNorm structure as upstream
-- with two changes needed to make it ONNX-exportable and to keep the demo
self-contained:

  * full (non-windowed) causal attention, computed with plain matmul +
    softmax instead of Flash Attention 3 -- i.e. what upstream's own
    ``--window-pattern L`` recommends for a non-FA3 backend;
  * no KV cache, value embeddings, "smear"/"backout" residual tricks,
    per-layer learnable resid/x0 scalars, or the Muon optimizer -- all
    training-loop or incremental-decoding machinery that doesn't change the
    shape of the exported single-shot forward-pass graph.

The result is close enough structurally that it is still useful as a
compatibility target for onnxsim: rotary embeddings, RMSNorm, GQA and the
relu^2 MLP all show up as real ops in the exported graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 2
    n_kv_head: int = 2
    n_embd: int = 256


def config_for_depth(
    depth: int, aspect_ratio: int = 64, head_dim: int = 128, **overrides
) -> GPTConfig:
    """Reproduce nanochat's own depth -> (n_layer, n_head, n_embd) sizing.

    Mirrors ``scripts/base_train.py``'s ``build_model_meta()``: model_dim is
    ``depth * aspect_ratio`` nudged up to the next multiple of ``head_dim``
    (so head_dim divides evenly), and num_heads = model_dim // head_dim.
    ``base_train.py`` always sets ``n_kv_head == n_head`` (full MHA, no GQA);
    pass ``n_kv_head=`` in ``overrides`` to try the GQA path instead.
    """
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    overrides.setdefault("n_kv_head", num_heads)
    return GPTConfig(n_layer=depth, n_head=num_heads, n_embd=model_dim, **overrides)


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm with no learnable scale, matching upstream's ``norm()``."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: float = 100000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    return cos[None, :, None, :], sin[None, :, None, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_kv_head <= config.n_head and config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = Linear(config.n_embd, self.n_head * self.head_dim)
        self.c_k = Linear(config.n_embd, self.n_kv_head * self.head_dim)
        self.c_v = Linear(config.n_embd, self.n_kv_head * self.head_dim)
        self.c_proj = Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)  # QK norm

        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:  # GQA: broadcast kv heads up to the query head count
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale + causal_mask
        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2, not GELU
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), cos, sin, causal_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = Linear(config.n_embd, padded_vocab_size)

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary_embeddings(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        causal_mask = torch.triu(
            torch.full((config.sequence_len, config.sequence_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask[None, None], persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Not upstream's exact init scheme (uniform, zero-initialised
        # projections) -- a plain small-std normal is enough for this demo.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        mask = self.causal_mask[:, :, :T, :T]

        x = self.wte(idx)
        x = rms_norm(x)  # norm after token embedding
        for block in self.h:
            x = block(x, cos, sin, mask)
        x = rms_norm(x)

        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]  # crop the vocab padding
        logits = 15.0 * torch.tanh(logits / 15.0)  # logit softcap to [-15, 15]
        return logits
