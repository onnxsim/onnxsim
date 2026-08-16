#!/usr/bin/env python3
"""A BitNet b1.58 decoder in PyTorch, in deployment form, plus its ONNX export.

This is the *source* side of the converter: it produces the kind of ONNX graph
``bitnet_ort.convert`` consumes, offline and with random weights, so the whole
pipeline can be exercised without downloading a multi-GB checkpoint.

The architecture follows microsoft/bitnet-b1.58-2B-4T
(https://huggingface.co/microsoft/bitnet-b1.58-2B-4T):

* ``BitLinear`` on all seven projections (q/k/v/o, gate/up/down); the token
  embedding and the LM head stay full precision, as in the released model;
* RMSNorm pre-norms plus BitNet's ``subln`` -- an extra RMSNorm on the input of
  the attention output projection and of the FFN down projection;
* grouped-query attention with rotary position embeddings;
* a squared-ReLU gated FFN (BitNet b1.58 2B-4T uses ``relu(gate)**2 * up``
  rather than LLaMA's SiLU SwiGLU).

``BitLinear`` here is in **inference form**: its weight buffer already holds the
absmean-quantized ternary values ``round(w/s).clamp(-1,1) * s``, which is what a
deployed BitNet checkpoint stores (packed) and what an ONNX export therefore
contains. Training-time straight-through estimation is not reproduced -- it does
not affect the exported graph.

Only ``torch`` is required; ``transformers`` is not used.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class BitNetConfig:
    """Shape of the decoder. Defaults are a small offline-testable stand-in."""

    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def bitnet_b158_2b_4t(cls) -> "BitNetConfig":
        """The released microsoft/bitnet-b1.58-2B-4T shape."""
        return cls(
            vocab_size=128256,
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=30,
            num_attention_heads=20,
            num_key_value_heads=5,
            max_position_embeddings=4096,
        )


def weight_quant(weight: torch.Tensor) -> torch.Tensor:
    """BitNet's absmean ternary quantizer: ``round(w/s).clamp(-1,1) * s``."""
    scale = weight.abs().mean().clamp_(min=1e-5)
    return (weight / scale).round().clamp_(-1, 1) * scale


class BitLinear(nn.Linear):
    """``nn.Linear`` whose weight is ternary (one shared scale), no bias.

    ``quantize_()`` folds the absmean quantizer into the stored weight, which is
    what makes the ONNX export contain ternary initializers.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)

    @torch.no_grad()
    def quantize_(self) -> None:
        self.weight.copy_(weight_quant(self.weight))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config
        heads, kv_heads, dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_proj = BitLinear(config.hidden_size, heads * dim)
        self.k_proj = BitLinear(config.hidden_size, kv_heads * dim)
        self.v_proj = BitLinear(config.hidden_size, kv_heads * dim)
        self.o_proj = BitLinear(heads * dim, config.hidden_size)
        # BitNet's subln: normalize the attention output before o_proj.
        self.attn_sub_norm = RMSNorm(heads * dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        cfg = self.config
        q = self.q_proj(x).view(batch, seq, cfg.num_attention_heads, cfg.head_dim)
        k = self.k_proj(x).view(batch, seq, cfg.num_key_value_heads, cfg.head_dim)
        v = self.v_proj(x).view(batch, seq, cfg.num_key_value_heads, cfg.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, k, cos, sin)

        repeat = cfg.num_attention_heads // cfg.num_key_value_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        weights = torch.softmax(scores + mask, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(self.attn_sub_norm(out))


class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size)
        self.ffn_sub_norm = RMSNorm(config.intermediate_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squared ReLU gate, as in BitNet b1.58 2B-4T (not SwiGLU).
        gated = F.relu(self.gate_proj(x)).square() * self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(gated))


class BitNetLayer(nn.Module):
    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = BitNetAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = BitNetMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class BitNetForCausalLM(nn.Module):
    """Prefill-only causal LM (no KV cache) -- the graph an exporter traces."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            BitNetLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @torch.no_grad()
    def quantize_(self) -> "BitNetForCausalLM":
        """Fold the ternary quantizer into every BitLinear weight."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.quantize_()
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        cos = self.cos_cached[:seq].unsqueeze(0)
        sin = self.sin_cached[:seq].unsqueeze(0)
        mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.lm_head(self.norm(x))


def build(config: BitNetConfig, seed: int = 0) -> BitNetForCausalLM:
    """A randomly initialised, already-ternarized BitNet decoder."""
    torch.manual_seed(seed)
    return BitNetForCausalLM(config).eval().quantize_()


def export_onnx(
    model: BitNetForCausalLM,
    path: str,
    batch: int = 1,
    seq: int = 32,
    opset: int = 18,
    constant_folding: bool = True,
) -> str:
    """Trace ``model`` to ONNX with a dynamic batch/sequence signature.

    ``F.linear`` is ``x @ weight.T`` because torch stores weights as
    ``[out, in]`` while ONNX ``MatMul`` wants ``[in, out]``, so every BitLinear
    starts life as ``Transpose(weight) -> MatMul``. torch's own constant folding
    normally collapses that into a plain initializer; ``constant_folding=False``
    leaves it in place, which is how an export whose ternary weights are hidden
    behind a Transpose -- and therefore invisible to ``bitnet_ort.convert``
    until onnxsim folds them -- is produced.
    """
    if seq > model.config.max_position_embeddings:
        raise ValueError(
            f"seq={seq} exceeds max_position_embeddings="
            f"{model.config.max_position_embeddings}; the rotary tables are baked "
            "into the export, so the graph cannot run past that length"
        )
    example = torch.randint(0, model.config.vocab_size, (batch, seq))
    torch.onnx.export(
        model,
        (example,),
        path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        dynamo=False,
        do_constant_folding=constant_folding,
    )
    return path
