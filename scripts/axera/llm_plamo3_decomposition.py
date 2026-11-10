"""PLaMo 3 as the Pulsar-free decomposition: one decoder layer and the post model in the op
sequence the emitters compile (the same ops as ``llm_layer_decomposition``), for the
checkpoint's float weights.

What differs from the Llama / Qwen3 decomposition (``plamo3_reference`` is the reference):

* four RMSNorms per layer with the gains folded: the gain is ``offset + weight`` with offsets
  1 (pre-mixer, q / k, pre-MLP, final), 1/5 (post-mixer) and 1/5**1.5 (post-MLP);
* the post-MLP norm is scale invariant, so ``silu(gate)`` and ``up`` are each normalized per
  position before the product (``unit_lines``): the product's range no longer depends on a few
  large positions (see the PLaMo notes in ``scripts/axera/README.md``);
* RoPE base 1e4 on the sliding layers, YaRN (base 1e6, factor 256, cos / sin times the YaRN
  concentration) on every ``sliding_window_pattern``-th layer; the sliding window is a no-op at
  the cache lengths used here, so it is not in the graph.

Inputs and outputs of the layer graph match ``llm_layer_decomposition.layer_graph``: ``input``
[1, H], ``K_cache`` / ``V_cache`` [S, nkv*hd] (rotated keys of the earlier tokens), ``mask``
[1, S+1], the four rope rows (per position, see ``rope_inputs``); outputs ``output``,
``K_cache_out``, ``V_cache_out``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_layer_decomposition as ld  # noqa: E402
import plamo3_reference as R  # noqa: E402
import rope_rotate_half as rr  # noqa: E402

#: epsilon of the normalizations of silu(gate) and up (not the model's eps: it only keeps a
#: zero row finite, and the post-MLP norm then removes the scale)
UNIT_TINY = 1e-12


def unit_lines(p: str, src: str, out: str, tiny: str = "unit_tiny") -> list:
    """Per-position unit RMS: x / sqrt(mean(x^2) + tiny) over the last axis (no gain)."""
    return [
        f"{p}_sq = Mul({src}, {src})",
        f"{p}_m = ReduceMean<axes=[-1], keepdims=1>({p}_sq)",
        f"{p}_e = Add({p}_m, {tiny})",
        f"{p}_r = Sqrt({p}_e)",
        f"{out} = Div({src}, {p}_r)",
    ]


def rope_tables(ref: R.Plamo3Reference, li: int, pos: int):
    """cos / sin rows [hd] of layer ``li`` at ``pos`` (concentration applied on full layers)."""
    full = (li + 1) % ref.pattern == 0
    inv = ref.inv_full if full else ref.inv_local
    conc = ref.conc if full else 1.0
    freqs = (pos * inv).astype(np.float64)
    emb = np.concatenate([freqs, freqs])
    return (np.cos(emb) * conc).astype(np.float32), (np.sin(emb) * conc).astype(
        np.float32
    )


def rope_inputs(ref: R.Plamo3Reference, li: int, pos: int) -> dict:
    """The four rope rows of one token: q's carry the 1/sqrt(hd) scale, sines are sign-folded."""
    cos, sin = rope_tables(ref, li, pos)
    scale = np.float32(1.0) / np.float32(np.sqrt(ref.hd))
    sin_s = rr.sin_signed(sin)
    return {
        "cos_q": (cos * scale).reshape(1, -1).astype(np.float32),
        "sinq": (sin_s * scale).reshape(1, -1).astype(np.float32),
        "cos_k": cos.reshape(1, -1).astype(np.float32),
        "sink": sin_s.reshape(1, -1).astype(np.float32),
    }


def layer_graph(ref: R.Plamo3Reference, li: int, S: int) -> onnx.ModelProto:
    """The decode-step graph of PLaMo layer ``li`` with a cache of S earlier tokens."""
    H, hd, nh, nkv, inter = ref.H, ref.hd, ref.nh, ref.nkv, ref.I
    g = nh // nkv
    qdim, kvd = nh * hd, nkv * hd
    p = f"model.layers.layers.{li}."
    w_qkv = ref.w(p + "mixer.qkv_proj.weight")  # [(nh + 2 nkv) hd, H]
    w_gu = ref.w(p + "mlp.gate_up_proj.weight")  # [2 I, H]
    consts = {
        "eps": np.array([R.EPS], np.float32),
        "unit_tiny": np.array([UNIT_TINY], np.float32),
        "ln1": (1.0 + ref.w(p + "pre_mixer_norm.weight")).astype(np.float32),
        "wq_t": np.ascontiguousarray(w_qkv[:qdim].T),
        "wk_t": np.ascontiguousarray(w_qkv[qdim : qdim + kvd].T),
        "wv_t": np.ascontiguousarray(w_qkv[qdim + kvd :].T),
        "qn": (1.0 + ref.w(p + "mixer.q_norm.weight")).astype(np.float32),
        "kn": (1.0 + ref.w(p + "mixer.k_norm.weight")).astype(np.float32),
        "rot_perm": rr.perm_matrix(hd),
        "wo_t": np.ascontiguousarray(ref.w(p + "mixer.o_proj.weight").T),
        "gpm": (0.2 + ref.w(p + "post_mixer_norm.weight")).astype(np.float32),
        "ln2": (1.0 + ref.w(p + "pre_mlp_norm.weight")).astype(np.float32),
        "wg_t": np.ascontiguousarray(w_gu[:inter].T),
        "wu_t": np.ascontiguousarray(w_gu[inter:].T),
        "wd_t": np.ascontiguousarray(ref.w(p + "mlp.down_proj.weight").T),
        "gpmlp": (0.2**1.5 + ref.w(p + "post_mlp_norm.weight")).astype(np.float32),
        "shape_q": np.array([nh, hd], np.int64),
        "shape_k": np.array([nkv, hd], np.int64),
        "shape_kv_rows": np.array([1, kvd], np.int64),
        "shape_att": np.array([1, qdim], np.int64),
        "shape_kv_t": np.array([S + 1, nkv, hd], np.int64),
        "shape_grp": np.array([nkv, g, hd], np.int64),
    }
    body = []
    body += ld.rmsnorm_lines("n1", "input", "ln1", "a")
    body += [
        "qp = MatMul(a, wq_t)",
        "q0 = Reshape(qp, shape_q)",
        "kp = MatMul(a, wk_t)",
        "k0 = Reshape(kp, shape_k)",
        "vp = MatMul(a, wv_t)",
        "v_row = Reshape(vp, shape_kv_rows)",
    ]
    body += ld.rmsnorm_lines("qnorm", "q0", "qn", "qg_")
    body += ld.rmsnorm_lines("knorm", "k0", "kn", "kg_")
    body += ld.rope_lines("rq", "qg_", "cos_q", "sinq", "qr", "rot_perm")
    body += ld.rope_lines("rk", "kg_", "cos_k", "sink", "kr", "rot_perm")
    body += [
        "k_row = Reshape(kr, shape_kv_rows)",
        "K_cache_out = Concat<axis=0>(K_cache, k_row)",
        "V_cache_out = Concat<axis=0>(V_cache, v_row)",
        "Kh = Reshape(K_cache_out, shape_kv_t)",
        "Kt = Transpose<perm=[1,2,0]>(Kh)",
        "Vh = Reshape(V_cache_out, shape_kv_t)",
        "Vt = Transpose<perm=[1,0,2]>(Vh)",
        "qgrp = Reshape(qr, shape_grp)",
        "sc = MatMul(qgrp, Kt)",
        "sm = Add(sc, mask)",
        "pr = Softmax<axis=-1>(sm)",
        "pv = MatMul(pr, Vt)",
        "att = Reshape(pv, shape_att)",
        "op = MatMul(att, wo_t)",
    ]
    body += ld.rmsnorm_lines("pmix", "op", "gpm", "on")
    body += ["x1 = Add(input, on)"]
    body += ld.rmsnorm_lines("n2", "x1", "ln2", "m")
    body += [
        "gate = MatMul(m, wg_t)",
        "up = MatMul(m, wu_t)",
        "sig = Sigmoid(gate)",
        "silu = Mul(gate, sig)",
    ]
    body += unit_lines("ug", "silu", "usg")
    body += unit_lines("uu", "up", "uup")
    body += [
        "act = Mul(usg, uup)",
        "down = MatMul(act, wd_t)",
    ]
    body += ld.rmsnorm_lines("pmlp", "down", "gpmlp", "mo")
    body += ["output = Add(x1, mo)"]
    runtime = {
        "input": (1, H),
        "K_cache": (S, kvd),
        "V_cache": (S, kvd),
        "mask": (1, S + 1),
        "cos_q": (1, hd),
        "sinq": (1, hd),
        "cos_k": (1, hd),
        "sink": (1, hd),
    }
    return ld.make_graph(
        f"plamo3_layer_{li}",
        runtime,
        consts,
        body,
        ["output", "K_cache_out", "V_cache_out"],
    )


def post_graph(ref: R.Plamo3Reference) -> onnx.ModelProto:
    """Final RMSNorm (offset 1) and the LM head: logits [1, V] = norm(x) @ lm_head.T."""
    V = ref.c["vocab_size"]
    consts = {
        "eps": np.array([R.EPS], np.float32),
        "ln_f": (1.0 + ref.w("model.norm.weight")).astype(np.float32),
        "head_t": np.ascontiguousarray(ref.w("lm_head.weight").T),
    }
    body = ld.rmsnorm_lines("nf", "input", "ln_f", "xn")
    body += ["logits = MatMul(xn, head_t)"]
    return ld.make_graph(
        "plamo3_post",
        {"input": (1, ref.H), "head_t": (ref.H, V)},
        consts,
        body,
        ["logits"],
    )
