"""Tests for ``onnxsim.qwen3_5_reconstruct`` -- building the Qwen3.5-VL
vision encoder and hybrid linear/full-attention text decoder ONNX graphs
directly from a HuggingFace-shaped checkpoint (see that module's own
docstring for the architecture and scope).

Same rigor as ``test_hf_reconstruct.py``/``test_gguf_reconstruct.py``: the
core claim under test is that each graph computes the same function the
real architecture (confirmed against ``transformers``' real
``modeling_qwen3_5.py``/``vision_utils.py`` source, see
``onnxsim/qwen3_5_reconstruct.py``'s docstring) does, checked against an
*independent* from-scratch numpy implementation -- not a reuse of anything
in ``qwen3_5_reconstruct.py`` itself -- run against the identical
hand-written safetensors checkpoint and compared to the ONNX graph's own
output via ``onnx.reference.ReferenceEvaluator``.
"""

import json
import math
import struct

import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

import onnxsim
from onnxsim.gguf_reconstruct import UnsupportedArchitectureError
from onnxsim.qwen3_5_reconstruct import (
    _extract_qwen3_5_configs,
    reconstruct_qwen3_5_language_model,
    reconstruct_qwen3_5_vision_encoder,
    reconstruct_qwen3_5_vlm,
)


def _write_safetensors(path, tensors):
    """Same hand-rolled, dependency-free ``.safetensors`` writer as
    ``test_hf_reconstruct.py`` -- see that file for the format."""
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        blobs.append(arr.astype("<f4").tobytes())
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(x))


def _rmsnorm_zero_centered(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * (1.0 + w)


def _rmsnorm_gated(x, gate, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    normed = (x / np.sqrt(var + eps)) * w
    return normed * _silu(gate)


def _l2norm(x, eps=1e-6):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_partial_rope(x, cos, sin, rotary_dim, head_dim):
    if rotary_dim == head_dim:
        return x * cos + _rotate_half(x) * sin
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = x_rot * cos + _rotate_half(x_rot) * sin
    return np.concatenate([x_embed, x_pass], axis=-1)


def _causal_depthwise_conv1d(x_bcs, weight_c1k):
    """`x_bcs`: (B, C, S). `weight_c1k`: (C, 1, K). Causal (left-padded by
    K-1 zeros) depthwise conv -- ``causal_conv1d_fn``'s own semantics."""
    batch, channels, seq = x_bcs.shape
    kernel = weight_c1k.shape[-1]
    xp = np.concatenate(
        [np.zeros((batch, channels, kernel - 1), dtype=x_bcs.dtype), x_bcs], axis=-1
    )
    out = np.zeros_like(x_bcs)
    w = weight_c1k[:, 0, :]  # (C, K)
    for t in range(seq):
        window = xp[:, :, t : t + kernel]  # (B, C, K)
        out[:, :, t] = np.sum(window * w[None, :, :], axis=-1)
    return out


def _recompose_mrope_freqs(freq_t, freq_h, freq_w, mrope_section):
    """A literal, independent port of ``Qwen3_5TextRotaryEmbedding.
    recomposition_frequencies`` (start from T everywhere, then overwrite
    the H/W strided slices) -- deliberately *not* the selection-mask
    formulation ``qwen3_5_reconstruct._mrope_selection_masks`` uses, so this
    is a genuine cross-check of that helper rather than a restatement of it."""
    out = freq_t.copy()
    h_len = mrope_section[1] * 3
    out[..., 1:h_len:3] = freq_h[..., 1:h_len:3]
    w_len = mrope_section[2] * 3
    out[..., 2:w_len:3] = freq_w[..., 2:w_len:3]
    return out


def _mrope_cos_sin_np(position_ids, rotary_dim, rope_theta, mrope_section):
    n_freq = rotary_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    )
    freqs_axes = []
    for axis in range(3):
        pos = position_ids[axis].astype(np.float64)[..., None]  # (batch, seq, 1)
        freqs_axes.append(pos * inv_freq[None, None, :])
    freqs = _recompose_mrope_freqs(
        freqs_axes[0], freqs_axes[1], freqs_axes[2], mrope_section
    )
    assert freqs.shape[-1] == n_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _gated_delta_net_np(
    h, weights, p, batch, seq, num_k, num_v, head_k_dim, head_v_dim, kernel, eps
):
    key_dim = head_k_dim * num_k
    value_dim = head_v_dim * num_v

    mixed_qkv = h @ weights[f"{p}.in_proj_qkv.weight"].T
    z = h @ weights[f"{p}.in_proj_z.weight"].T
    beta_pre = h @ weights[f"{p}.in_proj_b.weight"].T
    a_pre = h @ weights[f"{p}.in_proj_a.weight"].T

    conv_out = _causal_depthwise_conv1d(
        mixed_qkv.transpose(0, 2, 1), weights[f"{p}.conv1d.weight"]
    )
    mixed_qkv = _silu(conv_out).transpose(0, 2, 1)

    query = mixed_qkv[..., :key_dim].reshape(batch, seq, num_k, head_k_dim)
    key = mixed_qkv[..., key_dim : 2 * key_dim].reshape(batch, seq, num_k, head_k_dim)
    value = mixed_qkv[..., 2 * key_dim : 2 * key_dim + value_dim].reshape(
        batch, seq, num_v, head_v_dim
    )
    z = z.reshape(batch, seq, num_v, head_v_dim)

    beta = _sigmoid(beta_pre)
    g = -np.exp(weights[f"{p}.A_log"]) * _softplus(a_pre + weights[f"{p}.dt_bias"])

    n_rep = num_v // num_k
    if n_rep > 1:
        query = np.repeat(query, n_rep, axis=2)
        key = np.repeat(key, n_rep, axis=2)

    query = _l2norm(query) / math.sqrt(head_k_dim)
    key = _l2norm(key)

    state = np.zeros((batch, num_v, head_k_dim, head_v_dim), dtype=np.float32)
    outs = []
    for t in range(seq):
        q_t, k_t, v_t = query[:, t], key[:, t], value[:, t]
        g_t, beta_t = g[:, t], beta[:, t]
        decay_t = np.exp(g_t)[:, :, None, None]
        state = state * decay_t
        k_col = k_t[:, :, :, None]
        kv_mem = np.sum(state * k_col, axis=-2)
        delta = (v_t - kv_mem) * beta_t[:, :, None]
        state = state + k_col * delta[:, :, None, :]
        q_col = q_t[:, :, :, None]
        outs.append(np.sum(state * q_col, axis=-2))
    core = np.stack(outs, axis=1)  # (batch, seq, num_v, head_v_dim)

    normed = _rmsnorm_gated(
        core.reshape(-1, head_v_dim),
        z.reshape(-1, head_v_dim),
        weights[f"{p}.norm.weight"],
        eps,
    ).reshape(batch, seq, value_dim)
    return normed @ weights[f"{p}.out_proj.weight"].T


def _full_attention_np(
    h,
    weights,
    p,
    batch,
    seq,
    n_head,
    n_head_kv,
    head_dim,
    rotary_dim,
    eps,
    cos,
    sin,
    mask,
):
    q_gate = (h @ weights[f"{p}.q_proj.weight"].T).reshape(
        batch, seq, n_head, 2 * head_dim
    )
    q, gate = (
        q_gate[..., :head_dim],
        q_gate[..., head_dim:].reshape(batch, seq, n_head * head_dim),
    )
    k = (h @ weights[f"{p}.k_proj.weight"].T).reshape(batch, seq, n_head_kv, head_dim)
    v = (h @ weights[f"{p}.v_proj.weight"].T).reshape(batch, seq, n_head_kv, head_dim)

    q = _rmsnorm_zero_centered(q, weights[f"{p}.q_norm.weight"], eps).transpose(
        0, 2, 1, 3
    )
    k = _rmsnorm_zero_centered(k, weights[f"{p}.k_norm.weight"], eps).transpose(
        0, 2, 1, 3
    )
    v = v.transpose(0, 2, 1, 3)

    q = _apply_partial_rope(q, cos, sin, rotary_dim, head_dim)
    k = _apply_partial_rope(k, cos, sin, rotary_dim, head_dim)

    n_rep = n_head // n_head_kv
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim) + mask
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch, seq, n_head * head_dim)
    out = out * _sigmoid(gate)
    return out @ weights[f"{p}.o_proj.weight"].T


def _build_tiny_qwen3_5_text_checkpoint(tmp_path, seed=0):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    hidden_size = 8
    n_head, n_head_kv, head_dim = 2, 1, 4
    num_k, num_v, head_k_dim, head_v_dim = 2, 4, 4, 4
    conv_kernel_dim = 3
    intermediate_size = 12
    vocab_size = 10
    eps = 1e-5
    rope_theta = 10000.0
    partial_rotary_factor = 1.0
    mrope_section = [1, 1, 0]
    layer_types = ["linear_attention", "full_attention"]

    key_dim = head_k_dim * num_k
    value_dim = head_v_dim * num_v
    conv_dim = key_dim * 2 + value_dim

    weights = {}
    p0 = "model.language_model.layers.0"
    weights[f"{p0}.input_layernorm.weight"] = rand(hidden_size)
    weights[f"{p0}.linear_attn.in_proj_qkv.weight"] = rand(conv_dim, hidden_size)
    weights[f"{p0}.linear_attn.in_proj_z.weight"] = rand(value_dim, hidden_size)
    weights[f"{p0}.linear_attn.in_proj_b.weight"] = rand(num_v, hidden_size)
    weights[f"{p0}.linear_attn.in_proj_a.weight"] = rand(num_v, hidden_size)
    weights[f"{p0}.linear_attn.conv1d.weight"] = rand(conv_dim, 1, conv_kernel_dim)
    weights[f"{p0}.linear_attn.A_log"] = rng.uniform(0.01, 2.0, size=num_v).astype(
        np.float32
    )
    weights[f"{p0}.linear_attn.dt_bias"] = rand(num_v)
    weights[f"{p0}.linear_attn.norm.weight"] = rand(head_v_dim) + 1.0
    weights[f"{p0}.linear_attn.out_proj.weight"] = rand(hidden_size, value_dim)
    weights[f"{p0}.post_attention_layernorm.weight"] = rand(hidden_size)
    weights[f"{p0}.mlp.gate_proj.weight"] = rand(intermediate_size, hidden_size)
    weights[f"{p0}.mlp.up_proj.weight"] = rand(intermediate_size, hidden_size)
    weights[f"{p0}.mlp.down_proj.weight"] = rand(hidden_size, intermediate_size)

    p1 = "model.language_model.layers.1"
    weights[f"{p1}.input_layernorm.weight"] = rand(hidden_size)
    weights[f"{p1}.self_attn.q_proj.weight"] = rand(n_head * head_dim * 2, hidden_size)
    weights[f"{p1}.self_attn.k_proj.weight"] = rand(n_head_kv * head_dim, hidden_size)
    weights[f"{p1}.self_attn.v_proj.weight"] = rand(n_head_kv * head_dim, hidden_size)
    weights[f"{p1}.self_attn.o_proj.weight"] = rand(hidden_size, n_head * head_dim)
    weights[f"{p1}.self_attn.q_norm.weight"] = rand(head_dim)
    weights[f"{p1}.self_attn.k_norm.weight"] = rand(head_dim)
    weights[f"{p1}.post_attention_layernorm.weight"] = rand(hidden_size)
    weights[f"{p1}.mlp.gate_proj.weight"] = rand(intermediate_size, hidden_size)
    weights[f"{p1}.mlp.up_proj.weight"] = rand(intermediate_size, hidden_size)
    weights[f"{p1}.mlp.down_proj.weight"] = rand(hidden_size, intermediate_size)

    weights["model.language_model.norm.weight"] = rand(hidden_size)
    weights["lm_head.weight"] = rand(vocab_size, hidden_size)

    hf_dir = tmp_path / "tiny_qwen3_5_text"
    hf_dir.mkdir()
    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": hidden_size,
        "num_hidden_layers": 2,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head_kv,
        "head_dim": head_dim,
        "rms_norm_eps": eps,
        "vocab_size": vocab_size,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "layer_types": layer_types,
        "intermediate_size": intermediate_size,
        "linear_num_key_heads": num_k,
        "linear_num_value_heads": num_v,
        "linear_key_head_dim": head_k_dim,
        "linear_value_head_dim": head_v_dim,
        "linear_conv_kernel_dim": conv_kernel_dim,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
            "mrope_section": mrope_section,
        },
    }
    with open(hf_dir / "config.json", "w") as f:
        json.dump(
            {
                "text_config": text_config,
                "vision_config": {"model_type": "qwen3_5_vision"},
            },
            f,
        )
    _write_safetensors(hf_dir / "model.safetensors", weights)

    meta = dict(
        hidden_size=hidden_size,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        num_k=num_k,
        num_v=num_v,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_dim=conv_kernel_dim,
        eps=eps,
        rope_theta=rope_theta,
        rotary_dim=int(head_dim * partial_rotary_factor),
        mrope_section=mrope_section,
        layer_types=layer_types,
    )
    return hf_dir, text_config, weights, meta


def _text_reference_forward(inputs_embeds, position_ids, weights, meta, batch, seq):
    cos, sin = _mrope_cos_sin_np(
        position_ids, meta["rotary_dim"], meta["rope_theta"], meta["mrope_section"]
    )
    cos_b, sin_b = cos[:, None, :, :], sin[:, None, :, :]
    causal_mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)

    x = inputs_embeds
    for i, layer_type in enumerate(meta["layer_types"]):
        p = f"model.language_model.layers.{i}"
        resid = x
        h = _rmsnorm_zero_centered(
            x, weights[f"{p}.input_layernorm.weight"], meta["eps"]
        )
        if layer_type == "linear_attention":
            attn_out = _gated_delta_net_np(
                h,
                weights,
                f"{p}.linear_attn",
                batch,
                seq,
                meta["num_k"],
                meta["num_v"],
                meta["head_k_dim"],
                meta["head_v_dim"],
                meta["conv_kernel_dim"],
                meta["eps"],
            )
        else:
            attn_out = _full_attention_np(
                h,
                weights,
                f"{p}.self_attn",
                batch,
                seq,
                meta["n_head"],
                meta["n_head_kv"],
                meta["head_dim"],
                meta["rotary_dim"],
                meta["eps"],
                cos_b,
                sin_b,
                causal_mask,
            )
        x = resid + attn_out
        resid = x
        h = _rmsnorm_zero_centered(
            x, weights[f"{p}.post_attention_layernorm.weight"], meta["eps"]
        )
        gate = h @ weights[f"{p}.mlp.gate_proj.weight"].T
        up = h @ weights[f"{p}.mlp.up_proj.weight"].T
        act = _silu(gate) * up
        x = resid + act @ weights[f"{p}.mlp.down_proj.weight"].T

    x = _rmsnorm_zero_centered(
        x, weights["model.language_model.norm.weight"], meta["eps"]
    )
    return x @ weights["lm_head.weight"].T


def test_qwen3_5_language_model_matches_numpy_reference(tmp_path):
    hf_dir, text_config, weights, meta = _build_tiny_qwen3_5_text_checkpoint(tmp_path)
    batch, seq = 1, 3

    from onnxsim.hf_reconstruct import _index_safetensors_checkpoint

    entries = _index_safetensors_checkpoint(str(hf_dir))
    model = reconstruct_qwen3_5_language_model(
        text_config, entries, batch_size=batch, seq_len=seq
    )

    import onnx

    onnx.checker.check_model(model)

    rng = np.random.default_rng(42)
    inputs_embeds = (
        rng.standard_normal((batch, seq, meta["hidden_size"])).astype(np.float32) * 0.1
    )
    position_ids = np.tile(np.arange(seq, dtype=np.int64)[None, None, :], (3, batch, 1))

    expected = _text_reference_forward(
        inputs_embeds, position_ids, weights, meta, batch, seq
    )

    sess = ReferenceEvaluator(model)
    (actual,) = sess.run(
        None, {"inputs_embeds": inputs_embeds, "position_ids": position_ids}
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_qwen3_5_language_model_unknown_layer_type_raises(tmp_path):
    hf_dir, text_config, weights, meta = _build_tiny_qwen3_5_text_checkpoint(tmp_path)
    text_config["layer_types"] = ["linear_attention", "something_else"]

    from onnxsim.hf_reconstruct import _index_safetensors_checkpoint

    entries = _index_safetensors_checkpoint(str(hf_dir))
    with pytest.raises(UnsupportedArchitectureError):
        reconstruct_qwen3_5_language_model(
            text_config, entries, batch_size=1, seq_len=3
        )


def _ref_vision_position_ids(h, w, merge):
    positions = []
    for br in range(h // merge):
        for bc in range(w // merge):
            for ir in range(merge):
                for ic in range(merge):
                    positions.append((br * merge + ir, bc * merge + ic))
    return positions


def _ref_axis_taps_weights(index, size, side):
    src = index * (side - 1) / max(size - 1, 1)
    floor = math.floor(src)
    taps = [max(0, min(side - 1, floor)), max(0, min(side - 1, floor + 1))]
    weights = [max(0.0, 1.0 - abs(src - floor - off)) for off in (0, 1)]
    return taps, weights


def _ref_vision_rope_cos_sin(positions, head_dim, theta):
    spatial_dim = head_dim // 2
    freq_idx = np.arange(0, spatial_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (freq_idx / spatial_dim))
    cos_list, sin_list = [], []
    for row, col in positions:
        freq_h = row * inv_freq
        freq_w = col * inv_freq
        freq_hw = np.concatenate([freq_h, freq_w])
        freq_full = np.concatenate([freq_hw, freq_hw])
        cos_list.append(np.cos(freq_full))
        sin_list.append(np.sin(freq_full))
    return np.array(cos_list, dtype=np.float32), np.array(sin_list, dtype=np.float32)


def _build_tiny_qwen3_5_vision_checkpoint(tmp_path, seed=1):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    hidden_size = 8
    num_heads = 2
    head_dim = hidden_size // num_heads
    intermediate_size = 10
    patch_size, temporal_patch_size, in_channels = 2, 1, 1
    spatial_merge_size = 2
    out_hidden_size = 6
    num_position_embeddings = 16
    grid_thw = (1, 4, 4)
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

    weights = {}
    weights["model.visual.patch_embed.proj.weight"] = rand(
        hidden_size, in_channels, temporal_patch_size, patch_size, patch_size
    )
    weights["model.visual.patch_embed.proj.bias"] = rand(hidden_size)
    weights["model.visual.pos_embed.weight"] = rand(
        num_position_embeddings, hidden_size
    )
    p = "model.visual.blocks.0"
    weights[f"{p}.norm1.weight"] = rand(hidden_size) + 1.0
    weights[f"{p}.norm1.bias"] = rand(hidden_size)
    weights[f"{p}.attn.qkv.weight"] = rand(3 * hidden_size, hidden_size)
    weights[f"{p}.attn.qkv.bias"] = rand(3 * hidden_size)
    weights[f"{p}.attn.proj.weight"] = rand(hidden_size, hidden_size)
    weights[f"{p}.attn.proj.bias"] = rand(hidden_size)
    weights[f"{p}.norm2.weight"] = rand(hidden_size) + 1.0
    weights[f"{p}.norm2.bias"] = rand(hidden_size)
    weights[f"{p}.mlp.linear_fc1.weight"] = rand(intermediate_size, hidden_size)
    weights[f"{p}.mlp.linear_fc1.bias"] = rand(intermediate_size)
    weights[f"{p}.mlp.linear_fc2.weight"] = rand(hidden_size, intermediate_size)
    weights[f"{p}.mlp.linear_fc2.bias"] = rand(hidden_size)

    merged_hidden = hidden_size * spatial_merge_size**2
    weights["model.visual.merger.norm.weight"] = rand(hidden_size) + 1.0
    weights["model.visual.merger.norm.bias"] = rand(hidden_size)
    weights["model.visual.merger.linear_fc1.weight"] = rand(
        merged_hidden, merged_hidden
    )
    weights["model.visual.merger.linear_fc1.bias"] = rand(merged_hidden)
    weights["model.visual.merger.linear_fc2.weight"] = rand(
        out_hidden_size, merged_hidden
    )
    weights["model.visual.merger.linear_fc2.bias"] = rand(out_hidden_size)

    hf_dir = tmp_path / "tiny_qwen3_5_vision"
    hf_dir.mkdir()
    vision_config = {
        "model_type": "qwen3_5_vision",
        "hidden_size": hidden_size,
        "depth": 1,
        "num_heads": num_heads,
        "intermediate_size": intermediate_size,
        "hidden_act": "gelu_pytorch_tanh",
        "patch_size": patch_size,
        "temporal_patch_size": temporal_patch_size,
        "in_channels": in_channels,
        "spatial_merge_size": spatial_merge_size,
        "out_hidden_size": out_hidden_size,
        "num_position_embeddings": num_position_embeddings,
        "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0},
    }
    with open(hf_dir / "config.json", "w") as f:
        json.dump(
            {
                "text_config": {"model_type": "qwen3_5_text"},
                "vision_config": vision_config,
            },
            f,
        )
    _write_safetensors(hf_dir / "model.safetensors", weights)

    meta = dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        patch_dim=patch_dim,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=num_position_embeddings,
        grid_thw=grid_thw,
    )
    return hf_dir, vision_config, weights, meta


def _gelu_tanh_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _gelu_erf_np(x):
    from math import erf

    vec_erf = np.vectorize(erf)
    return 0.5 * x * (1.0 + vec_erf(x / np.sqrt(2.0)))


def _layernorm_np(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _vision_reference_forward(pixel_values, weights, meta):
    hidden_size, num_heads, head_dim = (
        meta["hidden_size"],
        meta["num_heads"],
        meta["head_dim"],
    )
    t, h, w = meta["grid_thw"]
    merge = meta["spatial_merge_size"]
    num_patches = t * h * w
    side = int(round(meta["num_position_embeddings"] ** 0.5))

    conv_w = weights["model.visual.patch_embed.proj.weight"].reshape(
        hidden_size, meta["patch_dim"]
    )
    x = pixel_values @ conv_w.T + weights["model.visual.patch_embed.proj.bias"]

    positions = _ref_vision_position_ids(h, w, merge)
    pos_embed_w = weights["model.visual.pos_embed.weight"]
    pos_embeds = np.zeros((num_patches, hidden_size), dtype=np.float32)
    for i, (row, col) in enumerate(positions):
        idxs_row, weights_row = _ref_axis_taps_weights(row, h, side)
        idxs_col, weights_col = _ref_axis_taps_weights(col, w, side)
        acc = np.zeros(hidden_size, dtype=np.float32)
        for ridx, rw in zip(idxs_row, weights_row):
            for cidx, cw in zip(idxs_col, weights_col):
                acc += pos_embed_w[ridx * side + cidx] * (rw * cw)
        pos_embeds[i] = acc
    x = x + pos_embeds

    cos, sin = _ref_vision_rope_cos_sin(
        positions, head_dim, 10000.0
    )  # (num_patches, head_dim)

    p = "model.visual.blocks.0"
    resid = x
    hn = _layernorm_np(x, weights[f"{p}.norm1.weight"], weights[f"{p}.norm1.bias"])
    qkv = hn @ weights[f"{p}.attn.qkv.weight"].T + weights[f"{p}.attn.qkv.bias"]
    qkv = qkv.reshape(num_patches, 3, num_heads, head_dim)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q = q * cos[:, None, :] + _rotate_half(q) * sin[:, None, :]
    k = k * cos[:, None, :] + _rotate_half(k) * sin[:, None, :]
    q, k, v = (a.transpose(1, 0, 2) for a in (q, k, v))  # (heads, N, head_dim)
    scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = (attn @ v).transpose(1, 0, 2).reshape(num_patches, hidden_size)
    out = out @ weights[f"{p}.attn.proj.weight"].T + weights[f"{p}.attn.proj.bias"]
    x = resid + out

    resid = x
    hn = _layernorm_np(x, weights[f"{p}.norm2.weight"], weights[f"{p}.norm2.bias"])
    fc1 = (
        hn @ weights[f"{p}.mlp.linear_fc1.weight"].T
        + weights[f"{p}.mlp.linear_fc1.bias"]
    )
    act = _gelu_tanh_np(fc1)
    fc2 = (
        act @ weights[f"{p}.mlp.linear_fc2.weight"].T
        + weights[f"{p}.mlp.linear_fc2.bias"]
    )
    x = resid + fc2

    x = _layernorm_np(
        x,
        weights["model.visual.merger.norm.weight"],
        weights["model.visual.merger.norm.bias"],
    )
    merge_unit = merge * merge
    x = x.reshape(num_patches // merge_unit, hidden_size * merge_unit)
    fc1 = (
        x @ weights["model.visual.merger.linear_fc1.weight"].T
        + weights["model.visual.merger.linear_fc1.bias"]
    )
    act = _gelu_erf_np(fc1)
    return (
        act @ weights["model.visual.merger.linear_fc2.weight"].T
        + weights["model.visual.merger.linear_fc2.bias"]
    )


def test_qwen3_5_vision_encoder_matches_numpy_reference(tmp_path):
    hf_dir, vision_config, weights, meta = _build_tiny_qwen3_5_vision_checkpoint(
        tmp_path
    )

    from onnxsim.hf_reconstruct import _index_safetensors_checkpoint

    entries = _index_safetensors_checkpoint(str(hf_dir))
    model = reconstruct_qwen3_5_vision_encoder(
        vision_config, entries, grid_thw=meta["grid_thw"]
    )

    import onnx

    onnx.checker.check_model(model)

    rng = np.random.default_rng(7)
    t, h, w = meta["grid_thw"]
    pixel_values = (
        rng.standard_normal((t * h * w, meta["patch_dim"])).astype(np.float32) * 0.1
    )

    expected = _vision_reference_forward(pixel_values, weights, meta)

    sess = ReferenceEvaluator(model)
    (actual,) = sess.run(None, {"pixel_values": pixel_values})

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_qwen3_5_vision_encoder_rejects_video(tmp_path):
    hf_dir, vision_config, weights, meta = _build_tiny_qwen3_5_vision_checkpoint(
        tmp_path
    )

    from onnxsim.hf_reconstruct import _index_safetensors_checkpoint

    entries = _index_safetensors_checkpoint(str(hf_dir))
    with pytest.raises(UnsupportedArchitectureError):
        reconstruct_qwen3_5_vision_encoder(vision_config, entries, grid_thw=(2, 4, 4))


def test_extract_qwen3_5_configs_flat_and_wrapped():
    flat = {
        "text_config": {"model_type": "qwen3_5_text"},
        "vision_config": {"model_type": "qwen3_5_vision"},
    }
    vlm, text, vision = _extract_qwen3_5_configs(flat)
    assert text["model_type"] == "qwen3_5_text"
    assert vision["model_type"] == "qwen3_5_vision"

    wrapped = {
        "model_type": "qwen_drive",
        "vlm_config": {
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_5_vision"},
        },
        "expert_config": {"model_type": "qwen_drive_planning_expert"},
    }
    vlm, text, vision = _extract_qwen3_5_configs(wrapped)
    assert text["model_type"] == "qwen3_5_text"
    assert vision["model_type"] == "qwen3_5_vision"


def test_extract_qwen3_5_configs_rejects_unrelated_config():
    with pytest.raises(UnsupportedArchitectureError):
        _extract_qwen3_5_configs({"model_type": "llama"})


def test_reconstruct_qwen3_5_vlm_end_to_end(tmp_path):
    """Both graphs, from one checkpoint directory carrying both the text
    and vision weights side by side (as a real Qwen3.5-VL checkpoint
    would)."""
    text_hf_dir, text_config, text_weights, _ = _build_tiny_qwen3_5_text_checkpoint(
        tmp_path
    )
    vision_hf_dir, vision_config, vision_weights, vision_meta = (
        _build_tiny_qwen3_5_vision_checkpoint(tmp_path)
    )

    combined_dir = tmp_path / "combined"
    combined_dir.mkdir()
    combined_weights = {**text_weights, **vision_weights}
    _write_safetensors(combined_dir / "model.safetensors", combined_weights)
    with open(combined_dir / "config.json", "w") as f:
        json.dump(
            {
                "vlm_config": {
                    "text_config": text_config,
                    "vision_config": vision_config,
                }
            },
            f,
        )

    models = reconstruct_qwen3_5_vlm(
        str(combined_dir), batch_size=1, seq_len=3, grid_thw=vision_meta["grid_thw"]
    )

    import onnx

    assert set(models) == {"vision_encoder", "language_model"}
    for m in models.values():
        onnx.checker.check_model(m)


def test_reconstruct_qwen3_5_vlm_end_to_end_simplifies(tmp_path):
    """Integration test: the full public entry point (config extraction
    from a wrapped ``vlm_config`` + both graph builders + checkpoint
    loading), then a real ``onnxsim.simplify()`` pass over each of its two
    returned graphs -- the same "does it actually simplify" claim the
    per-function tests below check in isolation, exercised here through
    the whole pipeline a real caller would use rather than by calling
    ``reconstruct_qwen3_5_language_model``/``reconstruct_qwen3_5_vision_encoder``
    directly."""
    text_hf_dir, text_config, text_weights, _ = _build_tiny_qwen3_5_text_checkpoint(
        tmp_path
    )
    vision_hf_dir, vision_config, vision_weights, vision_meta = (
        _build_tiny_qwen3_5_vision_checkpoint(tmp_path)
    )

    combined_dir = tmp_path / "combined_simplify"
    combined_dir.mkdir()
    combined_weights = {**text_weights, **vision_weights}
    _write_safetensors(combined_dir / "model.safetensors", combined_weights)
    with open(combined_dir / "config.json", "w") as f:
        json.dump(
            {
                "vlm_config": {
                    "text_config": text_config,
                    "vision_config": vision_config,
                }
            },
            f,
        )

    models = reconstruct_qwen3_5_vlm(
        str(combined_dir), batch_size=1, seq_len=3, grid_thw=vision_meta["grid_thw"]
    )

    for name, model in models.items():
        before = sum(1 for n in model.graph.node if n.op_type == "Transpose")
        simplified, check_ok = onnxsim.simplify(model, check_n=1)
        assert check_ok, f"{name} failed its post-simplify numerical check"
        after = sum(1 for n in simplified.graph.node if n.op_type == "Transpose")
        assert after < before, f"{name}'s weight Transposes should fold away"


def test_qwen3_5_language_model_simplifies_and_folds_weight_transposes(tmp_path):
    """Same claim as gguf_reconstruct.py's own
    ``test_simplify_folds_weight_transposes``: ``_linear`` (reused from
    that module) parameterizes every weight with a runtime ``Transpose``
    node rather than pre-transposing it at build time -- once the
    checkpoint's own tensors hydrate it, that Transpose is over a plain
    constant, so ``onnxsim.simplify()`` should constant-fold it away
    (leaving only the genuine per-call Q/K/V transposes), and the
    simplified graph should still check out numerically equivalent to the
    unsimplified one."""
    hf_dir, text_config, _weights, _meta = _build_tiny_qwen3_5_text_checkpoint(tmp_path)

    from onnxsim.hf_reconstruct import _index_safetensors_checkpoint

    entries = _index_safetensors_checkpoint(str(hf_dir))
    model = reconstruct_qwen3_5_language_model(
        text_config, entries, batch_size=1, seq_len=3
    )
    before = sum(1 for n in model.graph.node if n.op_type == "Transpose")

    simplified, check_ok = onnxsim.simplify(model, check_n=1)

    assert check_ok
    after = sum(1 for n in simplified.graph.node if n.op_type == "Transpose")
    assert after < before


def test_qwen3_5_vision_encoder_simplifies_and_folds_weight_transposes(tmp_path):
    hf_dir, vision_config, _weights, meta = _build_tiny_qwen3_5_vision_checkpoint(
        tmp_path
    )

    from onnxsim.hf_reconstruct import _index_safetensors_checkpoint

    entries = _index_safetensors_checkpoint(str(hf_dir))
    model = reconstruct_qwen3_5_vision_encoder(
        vision_config, entries, grid_thw=meta["grid_thw"]
    )
    before = sum(1 for n in model.graph.node if n.op_type == "Transpose")

    simplified, check_ok = onnxsim.simplify(model, check_n=1)

    assert check_ok
    after = sum(1 for n in simplified.graph.node if n.op_type == "Transpose")
    assert after < before
