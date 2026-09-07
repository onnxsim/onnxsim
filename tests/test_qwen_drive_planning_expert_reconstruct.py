"""Tests for ``onnxsim.qwen_drive_planning_expert_reconstruct`` -- building
Qwen-Drive-1.0's planning expert (the flow-matching diffusion transformer
that turns the VLM's attention cache into a driving trajectory; see that
module's own docstring for why this, not the separately-released BEV
perception stack, is the component ``QwenDriveForPlanning`` is actually
named for).

Unlike ``qwen_drive_perception_reconstruct.py`` (whose sheer size ruled out
a full independent reimplementation), this architecture -- a handful of
transformer layers plus a short Euler flow-matching loop, no custom CUDA
kernels or data-dependent shapes -- is small enough for a genuine
from-scratch numpy port of the *entire* reference ``PlanningExpert.sample()``
(:mod:`np_reference_planner`-style helpers, written independently below, not
imported from the module under test). :func:`test_matches_independent_numpy_reference`
therefore cross-checks the built ONNX graph's actual output values, not just
its shape, against that reference -- the same rigor
``test_hf_reconstruct.py``/``test_qwen3_5_reconstruct.py`` apply.
"""

import json
import math
import struct

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

import onnxsim
from onnxsim.gguf_reconstruct import UnsupportedArchitectureError
from onnxsim.qwen_drive_planning_expert_reconstruct import (
    reconstruct_qwen_drive_planning_expert,
)

_TINY_CONFIG = dict(
    hidden_size=16,
    intermediate_size=20,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    layers_per_kv=2,
    rms_norm_eps=1e-5,
    time_embed_dim=8,
    time_embed_scale=1000.0,
    fourier_num_features=4,
    fourier_max_frequency=16.0,
    nav_command_classes=3,
    ego_status_dim=5,
    history_dynamics_dim=2,
    rope_theta=10000.0,
    partial_rotary_factor=0.5,
    mrope_section=[1, 1, 0],
    num_future_points=5,
    num_history_points=4,
    trajectory_point_dim=3,
    trajectory_scale=[10.0, 5.0, 1.57],
    num_inference_steps=3,
    noise_init_std=1.0,
    min_one_minus_t=0.1,
)


def _write_safetensors(path, tensors):
    """Same hand-rolled ``.safetensors`` writer as the rest of this test
    family -- see ``test_hf_reconstruct.py`` for the format."""
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        blobs.append(arr.tobytes())
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def _lin(rng, out_f, in_f, bias=True):
    w = {".weight": (rng.standard_normal((out_f, in_f)) * 0.02).astype(np.float32)}
    if bias:
        w[".bias"] = (rng.standard_normal((out_f,)) * 0.02).astype(np.float32)
    return w


def _mlp_weights(rng, prefix, in_f, hidden):
    t = {}
    for k, v in _lin(rng, hidden, in_f).items():
        t[f"{prefix}.0{k}"] = v
    for k, v in _lin(rng, hidden, hidden).items():
        t[f"{prefix}.2{k}"] = v
    return t


def _build_tiny_checkpoint(tmp_path, cfg=None, seed=0):
    cfg = dict(_TINY_CONFIG if cfg is None else cfg)
    rng = np.random.default_rng(seed)
    tensors = {}

    hidden = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]
    num_layers = cfg["num_hidden_layers"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    n_rep = num_heads // num_kv_heads
    qkv_out = num_kv_heads * (n_rep * 2 + 2) * head_dim
    time_dim = cfg["time_embed_dim"]
    fourier_features = cfg["fourier_num_features"]
    point_dim = cfg["trajectory_point_dim"]
    nav_classes = cfg["nav_command_classes"]
    ego_dim = cfg["ego_status_dim"]
    dyn_dim = cfg["history_dynamics_dim"]
    num_history_points = cfg["num_history_points"]
    num_future_points = cfg["num_future_points"]

    for k, v in _lin(rng, hidden, point_dim).items():
        tensors[f"trajectory_proj{k}"] = v
    tensors.update(
        _mlp_weights(
            rng, "fourier_encoder.net", point_dim * fourier_features * 2, hidden
        )
    )
    tensors["waypoint_embed.weight"] = (
        rng.standard_normal((num_future_points, hidden)) * 0.02
    ).astype(np.float32)
    tensors.update(_mlp_weights(rng, "time_mlp", time_dim, hidden))
    tensors.update(_mlp_weights(rng, "nav_mlp", nav_classes, hidden))
    tensors.update(_mlp_weights(rng, "ego_mlp", ego_dim, hidden))
    history_dim = (num_history_points - 1) * point_dim + nav_classes
    tensors.update(_mlp_weights(rng, "history_encoder", history_dim, hidden))
    dynamics_dim = num_history_points * dyn_dim
    tensors.update(_mlp_weights(rng, "history_velocity_encoder", dynamics_dim, hidden))
    tensors.update(
        _mlp_weights(rng, "history_acceleration_encoder", dynamics_dim, hidden)
    )
    tensors.update(_mlp_weights(rng, "query_fusion", hidden * 7, hidden))

    for i in range(num_layers):
        p = f"layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = (
            rng.standard_normal((hidden,)) * 0.02 + 1.0
        ).astype(np.float32)
        tensors[f"{p}.qkv_proj.weight"] = (
            rng.standard_normal((qkv_out, hidden)) * 0.02
        ).astype(np.float32)
        tensors[f"{p}.q_norm.weight"] = (
            rng.standard_normal((head_dim,)) * 0.02 + 1.0
        ).astype(np.float32)
        tensors[f"{p}.k_norm.weight"] = (
            rng.standard_normal((head_dim,)) * 0.02 + 1.0
        ).astype(np.float32)
        tensors[f"{p}.o_proj.weight"] = (
            rng.standard_normal((hidden, num_heads * head_dim)) * 0.02
        ).astype(np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = (
            rng.standard_normal((hidden,)) * 0.02 + 1.0
        ).astype(np.float32)
        tensors[f"{p}.gate_up_proj.weight"] = (
            rng.standard_normal((2 * intermediate, hidden)) * 0.02
        ).astype(np.float32)
        tensors[f"{p}.down_proj.weight"] = (
            rng.standard_normal((hidden, intermediate)) * 0.02
        ).astype(np.float32)
        for k, v in _lin(rng, 6 * hidden, hidden).items():
            tensors[f"{p}.adaln_modulation.1{k}"] = v

    tensors["final_layernorm.weight"] = (
        rng.standard_normal((hidden,)) * 0.02 + 1.0
    ).astype(np.float32)
    for k, v in _lin(rng, point_dim, hidden).items():
        tensors[f"out_proj{k}"] = v

    hf_dir = tmp_path / "planner_ckpt"
    hf_dir.mkdir()
    _write_safetensors(hf_dir / "model.safetensors", tensors)
    with open(hf_dir / "config.json", "w") as f:
        json.dump({**cfg, "model_type": "qwen_drive_planning_expert"}, f)
    return str(hf_dir), cfg, tensors


# ---------------------------------------------------------------------------
# Independent, from-scratch numpy port of the real ``PlanningExpert.sample()``
# (``planning_expert.py``) -- ground truth for the numeric cross-check below.
# Deliberately does not import anything from
# ``onnxsim.qwen_drive_planning_expert_reconstruct``.


def _silu_np(x):
    return x / (1.0 + np.exp(-x))


def _rmsnorm_np(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def _linear_np(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def _mlp_np(x, t, prefix):
    h = _silu_np(_linear_np(x, t[f"{prefix}.0.weight"], t[f"{prefix}.0.bias"]))
    return _linear_np(h, t[f"{prefix}.2.weight"], t[f"{prefix}.2.bias"])


def _wrap_heading_np(x):
    wrapped = np.mod(x[..., 2:3] + math.pi, 2 * math.pi) - math.pi
    return np.concatenate([x[..., :2], wrapped], axis=-1)


def _normalize_trajectory_np(x, scale):
    return _wrap_heading_np(x) / scale.reshape(1, 1, -1)


def _denormalize_trajectory_np(x, scale):
    return _wrap_heading_np(x * scale.reshape(1, 1, -1))


def _normalize_history_np(history, scale):
    history = _wrap_heading_np(history - history[:, 0:1, :])
    return _normalize_trajectory_np(history[:, 1:, :], scale)


def _one_hot_masked_np(index, num_classes):
    index = np.asarray(index)
    valid = (index >= 0) & (index < num_classes)
    clamped = np.clip(index, 0, num_classes - 1)
    onehot = np.eye(num_classes, dtype=np.float32)[clamped]
    return onehot * valid.astype(np.float32)[..., None]


def _fourier_encode_np(waypoints, t, prefix, num_features, max_frequency):
    freqs = np.logspace(0, math.log10(max_frequency), num=num_features).astype(
        np.float32
    )
    angles = waypoints[..., None] * freqs * (2 * math.pi)
    feats = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    batch, length, point_dim, _ = feats.shape
    feats = feats.reshape(batch, length, point_dim * num_features * 2)
    return _mlp_np(feats, t, f"{prefix}.net")


def _sinusoidal_time_embedding_np(time_val, dim, scale):
    half = dim // 2
    decay = math.log(10000.0) / (half - 1)
    freqs = np.exp(np.arange(half) * -decay)
    angles = scale * time_val * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)]).astype(np.float32)


def _mrope_cos_sin_np(position_ids, rotary_dim, rope_theta, mrope_section):
    n_freq = rotary_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    )
    sel_h = np.zeros(n_freq, dtype=np.float64)
    sel_w = np.zeros(n_freq, dtype=np.float64)
    h_len = min(mrope_section[1] * 3, n_freq)
    for i in range(1, h_len, 3):
        sel_h[i] = 1.0
    w_len = min(mrope_section[2] * 3, n_freq)
    for i in range(2, w_len, 3):
        sel_w[i] = 1.0
    sel_t = 1.0 - sel_h - sel_w
    axis_freqs = [
        position_ids[axis].astype(np.float64)[..., None] * inv_freq for axis in range(3)
    ]
    freqs = axis_freqs[0] * sel_t + axis_freqs[1] * sel_h + axis_freqs[2] * sel_w
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _rotate_half_np(x, dim):
    return np.concatenate([-x[..., dim // 2 : dim], x[..., : dim // 2]], axis=-1)


def _apply_partial_rope_np(x, cos, sin, rotary_dim, head_dim):
    if rotary_dim == head_dim:
        return x * cos + _rotate_half_np(x, head_dim) * sin
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:head_dim]
    x_embed = x_rot * cos + _rotate_half_np(x_rot, rotary_dim) * sin
    return np.concatenate([x_embed, x_pass], axis=-1)


def _split_qkv_np(fused, batch, length, num_kv_heads, n_rep, head_dim):
    width = (n_rep * 2 + 2) * head_dim
    fused = fused.reshape(batch, length, num_kv_heads, width)
    gated_query = fused[..., : n_rep * 2 * head_dim]
    key = fused[..., n_rep * 2 * head_dim : n_rep * 2 * head_dim + head_dim]
    value = fused[..., n_rep * 2 * head_dim + head_dim :]
    query = gated_query[..., : n_rep * head_dim].reshape(
        batch, length, num_kv_heads * n_rep, head_dim
    )
    gate = gated_query[..., n_rep * head_dim :].reshape(
        batch, length, num_kv_heads * n_rep, head_dim
    )
    return query, gate, key, value


def _softmax_np(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _planning_expert_layer_np(
    hidden_states,
    scene_key_t,
    scene_value_t,
    cos,
    sin,
    rotary_dim,
    condition,
    t,
    prefix,
    cfg,
):
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    n_rep = num_heads // num_kv_heads
    eps = cfg["rms_norm_eps"]
    intermediate = cfg["intermediate_size"]
    batch, length, _ = hidden_states.shape

    mod = _linear_np(
        _silu_np(condition),
        t[f"{prefix}.adaln_modulation.1.weight"],
        t[f"{prefix}.adaln_modulation.1.bias"],
    )
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
        p[:, None, :] for p in np.split(mod, 6, axis=-1)
    )

    residual = hidden_states
    normed = _rmsnorm_np(hidden_states, t[f"{prefix}.input_layernorm.weight"], eps)
    x = normed * (1 + scale_attn) + shift_attn

    fused = _linear_np(x, t[f"{prefix}.qkv_proj.weight"])
    query, gate, key, value = _split_qkv_np(
        fused, batch, length, num_kv_heads, n_rep, head_dim
    )
    query = _rmsnorm_np(query, t[f"{prefix}.q_norm.weight"], eps)
    key = _rmsnorm_np(key, t[f"{prefix}.k_norm.weight"], eps)
    query = _apply_partial_rope_np(query, cos, sin, rotary_dim, head_dim)
    key = _apply_partial_rope_np(key, cos, sin, rotary_dim, head_dim)

    query_t = query.transpose(0, 2, 1, 3)
    key_t = key.transpose(0, 2, 1, 3)
    value_t = value.transpose(0, 2, 1, 3)
    key_cat = np.concatenate([scene_key_t, key_t], axis=2)
    value_cat = np.concatenate([scene_value_t, value_t], axis=2)

    q5 = query_t.reshape(batch, num_kv_heads, n_rep, length, head_dim)
    k5 = key_cat[:, :, None, :, :]
    v5 = value_cat[:, :, None, :, :]
    scores = (q5 @ np.swapaxes(k5, -1, -2)) / math.sqrt(head_dim)
    attn = _softmax_np(scores, axis=-1)
    out = (attn @ v5).reshape(batch, num_heads, length, head_dim).transpose(0, 2, 1, 3)
    out_flat = out.reshape(batch, length, num_heads * head_dim)
    gate_flat = gate.reshape(batch, length, num_heads * head_dim)
    gated = out_flat * (1.0 / (1.0 + np.exp(-gate_flat)))
    o = _linear_np(gated, t[f"{prefix}.o_proj.weight"])
    hidden_states = residual + (1 + gate_attn) * o

    residual2 = hidden_states
    normed2 = _rmsnorm_np(
        hidden_states, t[f"{prefix}.post_attention_layernorm.weight"], eps
    )
    x2 = normed2 * (1 + scale_ffn) + shift_ffn
    gate_up = _linear_np(x2, t[f"{prefix}.gate_up_proj.weight"])
    ffn = _silu_np(gate_up[..., :intermediate]) * gate_up[..., intermediate:]
    down = _linear_np(ffn, t[f"{prefix}.down_proj.weight"])
    return residual2 + (1 + gate_ffn) * down


def _sample_np(
    t,
    cfg,
    scene_cache,
    position_anchor,
    history,
    history_velocity,
    history_acceleration,
    nav_command,
    ego_status,
    noise,
    num_steps,
):
    hidden = cfg["hidden_size"]
    num_hidden_layers = cfg["num_hidden_layers"]
    layers_per_kv = cfg["layers_per_kv"]
    nav_classes = cfg["nav_command_classes"]
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    scale = np.array(cfg["trajectory_scale"], dtype=np.float32)
    batch, length, _ = noise.shape

    normalized_history = _normalize_history_np(history, scale)
    nav_onehot = _one_hot_masked_np(nav_command, nav_classes)
    pose_q = _mlp_np(
        np.concatenate([normalized_history.reshape(1, -1), nav_onehot], axis=-1),
        t,
        "history_encoder",
    )
    vel_q = _mlp_np(history_velocity.reshape(1, -1), t, "history_velocity_encoder")
    acc_q = _mlp_np(
        history_acceleration.reshape(1, -1), t, "history_acceleration_encoder"
    )
    fixed_condition = _mlp_np(nav_onehot, t, "nav_mlp") + _mlp_np(
        ego_status, t, "ego_mlp"
    )

    waypoint_embed_exp = np.broadcast_to(
        t["waypoint_embed.weight"][None, :length, :], (batch, length, hidden)
    )

    steps_arr = np.arange(1, length + 1).reshape(1, 1, length)
    positions = position_anchor.reshape(3, 1, 1) + steps_arr
    cos, sin = _mrope_cos_sin_np(
        positions, rotary_dim, cfg["rope_theta"], cfg["mrope_section"]
    )
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]

    scene_kv_t = []
    for j in range(num_hidden_layers // layers_per_kv):
        sk = np.broadcast_to(scene_cache[j][0], (batch,) + scene_cache[j][0].shape[1:])
        sv = np.broadcast_to(scene_cache[j][1], (batch,) + scene_cache[j][1].shape[1:])
        scene_kv_t.append((sk.transpose(0, 2, 1, 3), sv.transpose(0, 2, 1, 3)))

    waypoints = noise.astype(np.float32) * cfg["noise_init_std"]
    step_size = 1.0 / num_steps
    for step in range(num_steps):
        time_val = step * step_size
        time_condition = _mlp_np(
            _sinusoidal_time_embedding_np(
                time_val, cfg["time_embed_dim"], cfg["time_embed_scale"]
            ).reshape(1, -1),
            t,
            "time_mlp",
        )
        condition = time_condition + fixed_condition

        traj_proj = _linear_np(
            waypoints, t["trajectory_proj.weight"], t["trajectory_proj.bias"]
        )
        fourier = _fourier_encode_np(
            waypoints,
            t,
            "fourier_encoder",
            cfg["fourier_num_features"],
            cfg["fourier_max_frequency"],
        )
        broadcast = lambda v: np.broadcast_to(v[:, None, :], (batch, length, hidden))  # noqa: E731
        fused = np.concatenate(
            [
                traj_proj,
                fourier,
                broadcast(time_condition),
                broadcast(pose_q),
                waypoint_embed_exp,
                broadcast(vel_q),
                broadcast(acc_q),
            ],
            axis=-1,
        )
        hidden_states = _mlp_np(fused, t, "query_fusion")

        for layer_idx in range(num_hidden_layers):
            kv_idx = layer_idx // layers_per_kv
            scene_key_t, scene_value_t = scene_kv_t[kv_idx]
            hidden_states = _planning_expert_layer_np(
                hidden_states,
                scene_key_t,
                scene_value_t,
                cos,
                sin,
                rotary_dim,
                condition,
                t,
                f"layers.{layer_idx}",
                cfg,
            )

        normed = _rmsnorm_np(
            hidden_states, t["final_layernorm.weight"], cfg["rms_norm_eps"]
        )
        endpoint = _linear_np(normed, t["out_proj.weight"], t["out_proj.bias"])

        remaining = max(1.0 - time_val, cfg["min_one_minus_t"])
        waypoints = waypoints + (endpoint - waypoints) / remaining * step_size

    return _denormalize_trajectory_np(waypoints, scale)


def _feed_and_scene_cache(rng, cfg, num_samples, prefix_len):
    num_kv_sources = cfg["num_hidden_layers"] // cfg["layers_per_kv"]
    feed = {}
    scene_cache = []
    for j in range(num_kv_sources):
        k = (
            rng.standard_normal(
                (1, prefix_len, cfg["num_key_value_heads"], cfg["head_dim"])
            )
            * 0.1
        ).astype(np.float32)
        v = (
            rng.standard_normal(
                (1, prefix_len, cfg["num_key_value_heads"], cfg["head_dim"])
            )
            * 0.1
        ).astype(np.float32)
        feed[f"scene_key_{j}"] = k
        feed[f"scene_value_{j}"] = v
        scene_cache.append((k, v))
    position_anchor = np.array([[prefix_len - 1]] * 3, dtype=np.int64)
    feed["position_anchor"] = position_anchor
    history = np.cumsum(
        (rng.standard_normal((1, cfg["num_history_points"], 3)) * 0.1).astype(
            np.float32
        ),
        axis=1,
    )
    feed["history"] = history
    history_velocity = rng.standard_normal(
        (1, cfg["num_history_points"], cfg["history_dynamics_dim"])
    ).astype(np.float32)
    feed["history_velocity"] = history_velocity
    history_acceleration = rng.standard_normal(
        (1, cfg["num_history_points"], cfg["history_dynamics_dim"])
    ).astype(np.float32)
    feed["history_acceleration"] = history_acceleration
    nav_command = np.array([1], dtype=np.int64)
    feed["nav_command"] = nav_command
    ego_status = rng.standard_normal((1, cfg["ego_status_dim"])).astype(np.float32)
    feed["ego_status"] = ego_status
    noise = rng.standard_normal((num_samples, cfg["num_future_points"], 3)).astype(
        np.float32
    )
    feed["noise"] = noise
    return (
        feed,
        scene_cache,
        position_anchor,
        history,
        history_velocity,
        history_acceleration,
        nav_command,
        ego_status,
        noise,
    )


@pytest.mark.parametrize(
    "num_samples,num_hidden_layers,num_attention_heads,num_key_value_heads,layers_per_kv,prefix_len",
    [
        (2, 4, 4, 2, 2, 6),
        (3, 6, 6, 3, 3, 9),
    ],
)
def test_matches_independent_numpy_reference(
    tmp_path,
    num_samples,
    num_hidden_layers,
    num_attention_heads,
    num_key_value_heads,
    layers_per_kv,
    prefix_len,
):
    cfg = dict(_TINY_CONFIG)
    cfg.update(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        layers_per_kv=layers_per_kv,
    )
    hf_dir, cfg, tensors = _build_tiny_checkpoint(tmp_path, cfg, seed=num_hidden_layers)

    model = reconstruct_qwen_drive_planning_expert(hf_dir, num_samples=num_samples)
    onnx.checker.check_model(model)

    rng = np.random.default_rng(num_samples * 97 + prefix_len)
    (
        feed,
        scene_cache,
        position_anchor,
        history,
        history_velocity,
        history_acceleration,
        nav_command,
        ego_status,
        noise,
    ) = _feed_and_scene_cache(rng, cfg, num_samples, prefix_len)

    sess = ReferenceEvaluator(model)
    (onnx_out,) = sess.run(None, feed)
    assert onnx_out.shape == (
        num_samples,
        cfg["num_future_points"],
        cfg["trajectory_point_dim"],
    )
    assert np.isfinite(onnx_out).all()

    expected = _sample_np(
        tensors,
        cfg,
        scene_cache,
        position_anchor,
        history,
        history_velocity,
        history_acceleration,
        nav_command,
        ego_status,
        noise,
        cfg["num_inference_steps"],
    )
    np.testing.assert_allclose(onnx_out, expected, atol=1e-3, rtol=1e-3)


def test_reconstruct_qwen_drive_planning_expert_simplifies(tmp_path):
    """Real ``onnxsim.simplify()`` over the built graph. Neither
    ``reconstruct_qwen_drive_planning_expert`` nor this module calls
    ``simplify()`` itself -- it's an opt-in step for the caller, same as
    every other module in this reconstruction family. The
    ``scene_key_i``/``scene_value_i`` inputs carry a dynamic ``prefix_len``
    axis, so ``test_input_shapes`` fixes it for ``simplify()``'s own
    numerical ``check_n`` pass (the established pattern for this family's
    dynamic-shape graphs -- see e.g. ``test_gan.py``/``test_python_api.py``)."""
    hf_dir, cfg, _tensors = _build_tiny_checkpoint(tmp_path)
    num_samples = 2
    prefix_len = 6
    model = reconstruct_qwen_drive_planning_expert(hf_dir, num_samples=num_samples)
    before = len(model.graph.node)

    num_kv_sources = cfg["num_hidden_layers"] // cfg["layers_per_kv"]
    test_input_shapes = {}
    for j in range(num_kv_sources):
        test_input_shapes[f"scene_key_{j}"] = [
            1,
            prefix_len,
            cfg["num_key_value_heads"],
            cfg["head_dim"],
        ]
        test_input_shapes[f"scene_value_{j}"] = [
            1,
            prefix_len,
            cfg["num_key_value_heads"],
            cfg["head_dim"],
        ]

    simplified, check_ok = onnxsim.simplify(
        model, check_n=1, test_input_shapes=test_input_shapes
    )

    assert check_ok
    after = len(simplified.graph.node)
    assert after < before
    onnx.checker.check_model(simplified)


def test_unsupported_model_type_raises(tmp_path):
    hf_dir = tmp_path / "not_planning_expert"
    hf_dir.mkdir()
    with open(hf_dir / "config.json", "w") as f:
        json.dump({"model_type": "llama"}, f)
    _write_safetensors(hf_dir / "model.safetensors", {})
    with pytest.raises(UnsupportedArchitectureError):
        reconstruct_qwen_drive_planning_expert(str(hf_dir), num_samples=1)


def test_wrap_heading_matches_torch_remainder_convention():
    """``wrap_heading`` must wrap into ``[-pi, pi)`` with
    ``torch.remainder``'s always-non-negative-for-a-positive-divisor
    semantics -- this is exactly why the module builds it from
    ``Floor``/``Div``/``Mul``/``Sub`` instead of ONNX's ``Mod`` op (which,
    for floats, follows C's ``fmod`` and would give the wrong sign for a
    negative dividend). Cross-checked here against the reference formula
    directly, including the sign-flip-prone case of a negative heading."""
    headings = np.array(
        [0.0, math.pi - 1e-3, -math.pi + 1e-3, 3.5, -3.5, 100.0, -100.0],
        dtype=np.float32,
    )
    x = np.zeros((headings.shape[0], 3), dtype=np.float32)
    x[:, 2] = headings
    wrapped = _wrap_heading_np(x)[:, 2]
    assert np.all(wrapped >= -math.pi) and np.all(wrapped < math.pi)
    # Reference: torch.remainder(a, b) for b > 0 is always in [0, b).
    expected = np.mod(headings.astype(np.float64) + math.pi, 2 * math.pi) - math.pi
    np.testing.assert_allclose(wrapped, expected, atol=1e-5)
