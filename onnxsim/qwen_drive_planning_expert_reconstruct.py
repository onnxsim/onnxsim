"""Builds a runnable ONNX graph *and* its weights directly from
Qwen-Drive-1.0's planning expert (``qwen_drive_planning_expert``) -- the
flow-matching diffusion transformer that turns the VLM's own attention
cache into a driving trajectory. Per ``src/qwen_drive/modeling_qwen_drive.py``'s
own docstring, this -- not the separately-released BEV perception stack in
:mod:`onnxsim.qwen_drive_perception_reconstruct` -- is the component
``QwenDriveForPlanning`` (Qwen-Drive-1.0's actual top-level model class) is
named for: ``"a Qwen3.5 VLM driving a flow-matching planning expert"``. The
perception stack is never imported by ``modeling_qwen_drive.py`` at all; it
is a structurally separate package wired up only from standalone scripts.

Same "known architecture template, hydrate with the checkpoint's own
tensors" approach as the rest of this module family, reusing
:mod:`onnxsim.gguf_reconstruct`'s ``_Builder``/``_linear``/``_unsqueeze``/
``_slice_last_dim``/``_rmsnorm`` and :mod:`onnxsim.qwen3_5_reconstruct`'s
``_slice_axis``/``_silu``/``_apply_partial_rope``/``_text_mrope_cos_sin``
unchanged -- the waypoint
rotary embedding (``WaypointRotaryEmbedding`` in ``planning_expert.py``) is
architecturally the *same* interleaved multi-section M-RoPE recomposition
as the VLM's own text rotary embedding (confirmed against real source: both
start every frequency-pair index on the "T" axis, then let H's
``[offset : mrope_section[1]*3 : 3]``/W's ``[offset : mrope_section[2]*3 :
3]`` slices override their own disjoint indices), just applied to a
``[batch, length, heads, head_dim]``-layout tensor instead of the VLM's
``[batch, heads, length, head_dim]`` -- an extra ``unsqueeze(2)`` on the
returned ``cos``/``sin`` (broadcasting over the head axis at position 2
instead of 1) is the only difference, so both helpers are reused verbatim.

Architecture, confirmed against ``src/qwen_drive/planning_expert.py`` and
``configuration_qwen_drive.py`` in ``QwenLM/Qwen-Drive-1.0``:

* ``num_future_points`` waypoint tokens, each fused from seven signals
  (noisy waypoint, its per-channel Fourier features, the flow-matching time
  embedding, an encoding of the re-referenced history pose, a learned
  per-position embedding, and encodings of the raw history
  velocity/acceleration), then run through ``num_hidden_layers`` diffusion-
  transformer layers with AdaLN-Zero conditioning (on time + navigation
  command + ego status) and *joint* attention: every layer's keys/values are
  the concatenation of the (shared, cached) VLM prefix and the waypoint
  tokens' own, so waypoints read the driving scene and each other in one
  attention op, with no causal mask (bidirectional, unlike the VLM itself).
* Flow matching with a clean-endpoint parameterization: starting from
  Gaussian noise, ``num_steps`` Euler integration steps each predict the
  clean trajectory and blend a fraction of the way there.

Two build-time-only optimizations, both provably exact (pure functions of
already-fixed inputs, called identically every time in the reference code):

1. **The VLM cache's per-call ``.expand(batch, -1, -1, -1)`` and the
   rotary ``cos``/``sin`` are each computed once, not once per layer per
   step.** The reference code recomputes both on every
   ``predict_endpoint`` call (once per Euler step) and, for the cache
   expand, again inside every layer -- but neither depends on the step
   index, the layer index, or the evolving waypoints, so recomputing them
   is pure duplicated work.
2. **The time embedding is precomputed in ``numpy``, not built as graph
   ops.** ``SinusoidalTimeEmbedding`` has no learned parameters and its
   input, the Euler step's flow-matching time, is a python float already
   fixed at graph-*build* time once ``num_steps`` is chosen (unlike the
   navigation-command/ego-status conditioning, which are genuine per-call
   graph inputs and so stay as graph ops all the way through their own
   ``nav_mlp``/``ego_mlp``). Only the subsequent ``time_mlp`` -- which does
   have learned weights -- is built as graph ops.

Scope, narrower than ``QwenDriveForPlanning.generate_trajectory``:

* **The VLM prefill is not part of this graph.** ``scene_key_i``/
  ``scene_value_i`` (the post-rotary keys/values of the VLM's
  ``full_attention`` layers, one pair per ``num_hidden_layers //
  layers_per_kv`` cache the expert reads) and ``position_anchor`` (the
  M-RoPE position of the prefix's last token) are graph *inputs* --
  genuinely data-dependent on the actual prompt/scene of a specific call,
  unlike this module family's usual build-time-constant-if-possible
  quantities. Producing them is :mod:`onnxsim.qwen3_5_reconstruct`'s job
  (a separate graph); composing the two is a caller-side concern, exactly
  like the encoder/vision-tower split that module's own docstring
  describes.
* **Single scene per call** (batch dimension 1 for every input except the
  caller-chosen, static ``num_samples`` trajectory samples drawn from
  independent noise) -- matches how ``_plan_from_cache`` itself is always
  called (one ``DrivingScene`` at a time).
* **``num_steps`` is a caller-chosen, build-time-static unroll count**,
  like every other static dimension in this module family -- defaults to
  the checkpoint's own ``num_inference_steps``.
* **Output is the fully denormalized trajectory** (metres/radians, ego
  frame), i.e. this graph already applies ``denormalize_trajectory`` --
  unlike :mod:`onnxsim.qwen_drive_perception_reconstruct`'s heads, there is
  no genuinely variable-length post-processing step left to strip out here
  (``NMSFreeCoder``'s dynamic filtering has no analogue in this module).
* **The caller supplies raw standard-normal noise directly.** Multiplying
  by ``noise_init_std`` happens inside the graph; drawing the noise itself
  (seeded per-sample RNG, see ``QwenDriveForPlanning._initial_noise``) does
  not, matching this whole module family's convention that RNG stays the
  caller's concern.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper

from onnxsim.gguf_reconstruct import (
    _IR_VERSION,
    _OPSET,
    UnsupportedArchitectureError,
    _Builder,
    _linear,
    _rmsnorm,
    _slice_last_dim,
    _unsqueeze,
)
from onnxsim.hf_reconstruct import (
    _index_safetensors_checkpoint,
    _read_tensor,
    read_hf_config,
)
from onnxsim.qwen3_5_reconstruct import (
    _apply_partial_rope,
    _silu,
    _slice_axis,
    _text_mrope_cos_sin,
)

_SUPPORTED_MODEL_TYPE = "qwen_drive_planning_expert"

_PLANNING_EXPERT_DEFAULTS = {
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "layers_per_kv": 4,
    "rms_norm_eps": 1e-5,
    "time_embed_dim": 128,
    "time_embed_scale": 1000.0,
    "fourier_num_features": 16,
    "fourier_max_frequency": 16.0,
    "nav_command_classes": 3,
    "ego_status_dim": 8,
    "history_dynamics_dim": 2,
    "rope_theta": 1.0e7,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "num_future_points": 50,
    "num_history_points": 16,
    "trajectory_point_dim": 3,
    "trajectory_scale": [165.0, 25.0, 1.5703125],
    "num_inference_steps": 10,
    "noise_init_std": 1.0,
    "min_one_minus_t": 0.1,
}


def _cfg(config: dict, key: str):
    if key in config:
        return config[key]
    return _PLANNING_EXPERT_DEFAULTS[key]


# ---------------------------------------------------------------------------
# Small numpy-only precomputes (no learned weights involved).


def _fourier_freqs_np(num_features: int, max_frequency: float) -> np.ndarray:
    """``torch.logspace(0, log10(max_frequency), steps=num_features)`` --
    ``FourierFeatureEncoder``'s per-call frequency table. No learned state,
    so (like the rest of this family's pure-geometry precomputes) built once
    in ``numpy`` rather than as graph ops."""
    return np.logspace(
        0, math.log10(max_frequency), num=num_features, dtype=np.float64
    ).astype(np.float32)


def _sinusoidal_time_embedding_np(t: float, dim: int, scale: float) -> np.ndarray:
    """``SinusoidalTimeEmbedding`` for one flow-matching time ``t``. Stateless
    (no learned parameters) and, once ``num_steps`` fixes every Euler step's
    ``t`` at graph-build time, a pure function of already-known numbers --
    see this module's docstring, optimization 2."""
    half = dim // 2
    decay = math.log(10000.0) / (half - 1)
    freqs = np.exp(np.arange(half, dtype=np.float64) * -decay)
    angles = scale * t * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Generic op helpers specific to this module.


def _mlp_block(
    b: _Builder, x: str, declare, prefix: str, in_dim: int, hidden: int
) -> str:
    """``_mlp(in_features, hidden) = nn.Sequential(Linear, SiLU, Linear)`` --
    every conditioning encoder (``time_mlp``/``nav_mlp``/``ego_mlp``/
    ``history_encoder``/``history_velocity_encoder``/
    ``history_acceleration_encoder``/``query_fusion``) shares this shape,
    Sequential indices 0 and 2 (1 is the parameter-free ``SiLU``)."""
    h = _linear(
        b, x, declare(f"{prefix}.0.weight"), declare(f"{prefix}.0.bias"), f"{prefix}.l0"
    )
    h = _silu(b, h, f"{prefix}.act")
    return _linear(
        b, h, declare(f"{prefix}.2.weight"), declare(f"{prefix}.2.bias"), f"{prefix}.l2"
    )


def _one_hot_masked(b: _Builder, index: str, num_classes: int, prefix: str) -> str:
    """``_one_hot`` (``planning_expert.py``): one-hot, mapping any
    out-of-range index to an all-zero row instead of raising -- native
    ``OneHot`` needs an in-range index, so the index is clamped first and
    the out-of-range rows are zeroed afterwards by a validity mask,
    reproducing the reference's ``clamp`` + ``* valid`` exactly."""
    depth_c = b.const(np.array(num_classes, dtype=np.int64), prefix=f"{prefix}.depth")
    values_c = b.const(
        np.array([0.0, 1.0], dtype=np.float32), prefix=f"{prefix}.values"
    )
    zero_c = b.const(np.array(0, dtype=np.int64), prefix=f"{prefix}.zero")
    max_c = b.const(np.array(num_classes - 1, dtype=np.int64), prefix=f"{prefix}.max")
    num_classes_c = b.const(
        np.array(num_classes, dtype=np.int64), prefix=f"{prefix}.numc"
    )
    idx_clamped = b.op("Clip", [index, zero_c, max_c], f"{prefix}.clip")
    onehot = b.op("OneHot", [idx_clamped, depth_c, values_c], prefix, axis=-1)
    ge = b.op("GreaterOrEqual", [index, zero_c], f"{prefix}.ge")
    lt = b.op("Less", [index, num_classes_c], f"{prefix}.lt")
    valid = b.op("And", [ge, lt], f"{prefix}.valid")
    valid_f = b.op("Cast", [valid], f"{prefix}.validf", to=onnx.TensorProto.FLOAT)
    valid_f = _unsqueeze(b, valid_f, [-1], f"{prefix}.validf.unsq")
    return b.op("Mul", [onehot, valid_f], f"{prefix}.masked")


def _wrap_heading3(b: _Builder, x: str, prefix: str) -> str:
    """``wrap_heading``: wraps channel 2 (heading) of a ``(..., 3)`` tensor
    into ``[-pi, pi)`` via floor-division modulo -- ``torch.remainder``'s
    always-non-negative-for-a-positive-divisor semantics, which ONNX's own
    ``Mod`` op does *not* reproduce for floats (``Mod`` with ``fmod=1``
    follows C's ``fmod``, matching the *dividend*'s sign instead), so it is
    built by hand from ``Floor``/``Div``/``Mul``/``Sub`` instead."""
    pi_c = b.const(np.array(math.pi, dtype=np.float32), prefix="pi")
    twopi_c = b.const(np.array(2.0 * math.pi, dtype=np.float32), prefix="twopi")
    xy = _slice_last_dim(b, x, 0, 2, f"{prefix}.xy")
    h = _slice_last_dim(b, x, 2, 3, f"{prefix}.h")
    h_shift = b.op("Add", [h, pi_c], f"{prefix}.hshift")
    q = b.op(
        "Floor", [b.op("Div", [h_shift, twopi_c], f"{prefix}.qdiv")], f"{prefix}.qfloor"
    )
    h_mod = b.op(
        "Sub", [h_shift, b.op("Mul", [q, twopi_c], f"{prefix}.qmul")], f"{prefix}.hmod"
    )
    h_wrapped = b.op("Sub", [h_mod, pi_c], f"{prefix}.hwrapped")
    return b.op("Concat", [xy, h_wrapped], prefix, axis=-1)


def _normalize_trajectory(b: _Builder, x: str, scale_c: str, prefix: str) -> str:
    wrapped = _wrap_heading3(b, x, f"{prefix}.wrap")
    return b.op("Div", [wrapped, scale_c], prefix)


def _denormalize_trajectory(b: _Builder, x: str, scale_c: str, prefix: str) -> str:
    scaled = b.op("Mul", [x, scale_c], f"{prefix}.scaled")
    return _wrap_heading3(b, scaled, prefix)


def _normalize_history(
    b: _Builder, history: str, scale_c: str, num_history_points: int, prefix: str
) -> str:
    """``normalize_history``: re-references history to its oldest pose (which
    becomes the origin and is dropped), then normalizes. ``wrap_heading`` is
    applied twice in a row here -- once directly, once again inside
    ``normalize_trajectory`` -- exactly as the reference code does (harmless:
    wrapping an already-wrapped angle is a no-op), kept rather than
    "optimized" away to mirror the reference structurally."""
    origin = _slice_axis(b, history, 1, 0, 1, f"{prefix}.origin")
    shifted = b.op("Sub", [history, origin], f"{prefix}.shifted")
    wrapped = _wrap_heading3(b, shifted, f"{prefix}.wrap1")
    dropped = _slice_axis(b, wrapped, 1, 1, num_history_points, f"{prefix}.dropped")
    return _normalize_trajectory(b, dropped, scale_c, prefix)


def _expand_batch(b: _Builder, x: str, batch: int, prefix: str) -> str:
    """Broadcasts a rank-4 ``[1, *rest]`` tensor to ``[batch, *rest]`` where
    ``rest`` may contain a dynamic (symbolic) dimension -- the VLM cache's
    sequence length is not known at graph-build time, so the target shape
    is assembled from a runtime ``Shape`` rather than a python constant."""
    shape_full = b.op("Shape", [x], f"{prefix}.shape")
    rest = _slice_axis(b, shape_full, 0, 1, 4, f"{prefix}.rest")
    batch_c = b.const(np.array([batch], dtype=np.int64), prefix=f"{prefix}.batch_c")
    target = b.op("Concat", [batch_c, rest], f"{prefix}.target", axis=0)
    return b.op("Expand", [x, target], prefix)


def _fourier_encoder(
    b: _Builder,
    waypoints: str,
    freqs_c: str,
    declare,
    prefix: str,
    point_dim: int,
    num_features: int,
    hidden: int,
    batch: int,
    length: int,
) -> str:
    """``FourierFeatureEncoder``: per-channel Fourier features of a waypoint,
    then a two-layer MLP."""
    two_pi_c = b.const(
        np.array(2.0 * math.pi, dtype=np.float32), prefix=f"{prefix}.two_pi"
    )
    wp_unsq = _unsqueeze(b, waypoints, [-1], f"{prefix}.wp_unsq")
    angles = b.op("Mul", [wp_unsq, freqs_c], f"{prefix}.angles_raw")
    angles = b.op("Mul", [angles, two_pi_c], f"{prefix}.angles")
    sin_a = b.op("Sin", [angles], f"{prefix}.sin")
    cos_a = b.op("Cos", [angles], f"{prefix}.cos")
    feats = b.op("Concat", [sin_a, cos_a], f"{prefix}.feats", axis=-1)
    feats_flat = b.op(
        "Reshape",
        [feats, b.shape_const([batch, length, point_dim * num_features * 2])],
        f"{prefix}.flat",
    )
    return _mlp_block(
        b, feats_flat, declare, f"{prefix}.net", point_dim * num_features * 2, hidden
    )


def _split_qkv(
    b: _Builder,
    fused: str,
    batch: int,
    length: int,
    num_kv_heads: int,
    n_rep: int,
    head_dim: int,
    prefix: str,
) -> Tuple[str, str, str, str]:
    """``PlanningExpertLayer._split_qkv``: the fused projection's layout is,
    per key/value group, that group's query heads and their output gates
    first, then one key and one value head. Slicing happens on the still-
    grouped ``[batch, length, groups, ...]`` tensor exactly as the reference
    ``torch.split``/``chunk`` do; only the *final* reshape collapses
    ``(groups, heads_per_group)`` into ``num_heads`` (row-major, i.e. each
    group's ``heads_per_group`` query heads end up consecutive -- matching
    the ``q5`` grouped-attention reshape in :func:`_planning_expert_layer`)."""
    per_group_width = (n_rep * 2 + 2) * head_dim
    fused_r = b.op(
        "Reshape",
        [fused, b.shape_const([batch, length, num_kv_heads, per_group_width])],
        f"{prefix}.r",
    )
    gated_query = _slice_last_dim(b, fused_r, 0, n_rep * 2 * head_dim, f"{prefix}.gq")
    key = _slice_last_dim(
        b, fused_r, n_rep * 2 * head_dim, n_rep * 2 * head_dim + head_dim, f"{prefix}.k"
    )
    value = _slice_last_dim(
        b, fused_r, n_rep * 2 * head_dim + head_dim, per_group_width, f"{prefix}.v"
    )
    query = _slice_last_dim(b, gated_query, 0, n_rep * head_dim, f"{prefix}.q")
    gate = _slice_last_dim(
        b, gated_query, n_rep * head_dim, n_rep * 2 * head_dim, f"{prefix}.gate"
    )
    num_heads = num_kv_heads * n_rep
    query_r = b.op(
        "Reshape",
        [query, b.shape_const([batch, length, num_heads, head_dim])],
        f"{prefix}.q_r",
    )
    gate_r = b.op(
        "Reshape",
        [gate, b.shape_const([batch, length, num_heads, head_dim])],
        f"{prefix}.gate_r",
    )
    return query_r, gate_r, key, value


def _planning_expert_layer(
    b: _Builder,
    hidden_states: str,
    scene_key_t: str,
    scene_value_t: str,
    cos: str,
    sin: str,
    rotary_dim: int,
    condition: str,
    declare,
    prefix: str,
    hidden: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate: int,
    eps: float,
    batch: int,
    length: int,
) -> str:
    """One ``PlanningExpertLayer``: AdaLN-Zero-conditioned joint attention
    (own waypoint keys/values concatenated with the VLM's cached
    keys/values) + a gated SwiGLU feed-forward."""
    n_rep = num_heads // num_kv_heads
    one_c = b.const(np.array(1.0, dtype=np.float32), prefix="one")

    mod = _linear(
        b,
        _silu(b, condition, f"{prefix}.adaln.act"),
        declare(f"{prefix}.adaln_modulation.1.weight"),
        declare(f"{prefix}.adaln_modulation.1.bias"),
        f"{prefix}.adaln_modulation",
    )
    mod_parts = []
    for i in range(6):
        part = _slice_last_dim(b, mod, i * hidden, (i + 1) * hidden, f"{prefix}.mod{i}")
        mod_parts.append(_unsqueeze(b, part, [1], f"{prefix}.mod{i}.unsq"))
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod_parts

    residual = hidden_states
    normed = _rmsnorm(
        b,
        hidden_states,
        declare(f"{prefix}.input_layernorm.weight"),
        eps,
        f"{prefix}.ln1",
    )
    x = b.op(
        "Add",
        [
            b.op(
                "Mul",
                [normed, b.op("Add", [one_c, scale_attn], f"{prefix}.scale_attn1")],
                f"{prefix}.x_scaled",
            ),
            shift_attn,
        ],
        f"{prefix}.x",
    )

    fused = _linear(b, x, declare(f"{prefix}.qkv_proj.weight"), None, f"{prefix}.qkv")
    query, gate, key, value = _split_qkv(
        b, fused, batch, length, num_kv_heads, n_rep, head_dim, f"{prefix}.split"
    )

    query = _rmsnorm(
        b, query, declare(f"{prefix}.q_norm.weight"), eps, f"{prefix}.qnorm"
    )
    key = _rmsnorm(b, key, declare(f"{prefix}.k_norm.weight"), eps, f"{prefix}.knorm")
    query = _apply_partial_rope(
        b, query, cos, sin, rotary_dim, head_dim, f"{prefix}.q_rope"
    )
    key = _apply_partial_rope(
        b, key, cos, sin, rotary_dim, head_dim, f"{prefix}.k_rope"
    )

    query_t = b.op("Transpose", [query], f"{prefix}.q_t", perm=[0, 2, 1, 3])
    key_t = b.op("Transpose", [key], f"{prefix}.k_t", perm=[0, 2, 1, 3])
    value_t = b.op("Transpose", [value], f"{prefix}.v_t", perm=[0, 2, 1, 3])

    key_cat = b.op("Concat", [scene_key_t, key_t], f"{prefix}.k_cat", axis=2)
    value_cat = b.op("Concat", [scene_value_t, value_t], f"{prefix}.v_cat", axis=2)

    q5 = b.op(
        "Reshape",
        [query_t, b.shape_const([batch, num_kv_heads, n_rep, length, head_dim])],
        f"{prefix}.q5",
    )
    k5 = _unsqueeze(b, key_cat, [2], f"{prefix}.k5")
    v5 = _unsqueeze(b, value_cat, [2], f"{prefix}.v5")
    k5t = b.op("Transpose", [k5], f"{prefix}.k5t", perm=[0, 1, 2, 4, 3])
    inv_sqrt_d = b.const(
        np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), prefix="inv_sqrt_d"
    )
    scores = b.op(
        "Mul",
        [b.op("MatMul", [q5, k5t], f"{prefix}.scores"), inv_sqrt_d],
        f"{prefix}.scores_scaled",
    )
    attn = b.op("Softmax", [scores], f"{prefix}.softmax", axis=-1)
    out5 = b.op("MatMul", [attn, v5], f"{prefix}.out5")
    out = b.op(
        "Reshape",
        [out5, b.shape_const([batch, num_heads, length, head_dim])],
        f"{prefix}.out_r",
    )
    out = b.op("Transpose", [out], f"{prefix}.out_t", perm=[0, 2, 1, 3])
    out_flat = b.op(
        "Reshape",
        [out, b.shape_const([batch, length, num_heads * head_dim])],
        f"{prefix}.out_flat",
    )

    gate_flat = b.op(
        "Reshape",
        [gate, b.shape_const([batch, length, num_heads * head_dim])],
        f"{prefix}.gate_flat",
    )
    gated = b.op(
        "Mul",
        [out_flat, b.op("Sigmoid", [gate_flat], f"{prefix}.gate_sig")],
        f"{prefix}.gated",
    )
    o = _linear(b, gated, declare(f"{prefix}.o_proj.weight"), None, f"{prefix}.o_proj")
    hidden_states = b.op(
        "Add",
        [
            residual,
            b.op(
                "Mul",
                [b.op("Add", [one_c, gate_attn], f"{prefix}.gate_attn1"), o],
                f"{prefix}.o_gated",
            ),
        ],
        f"{prefix}.h1",
    )

    residual2 = hidden_states
    normed2 = _rmsnorm(
        b,
        hidden_states,
        declare(f"{prefix}.post_attention_layernorm.weight"),
        eps,
        f"{prefix}.ln2",
    )
    x2 = b.op(
        "Add",
        [
            b.op(
                "Mul",
                [normed2, b.op("Add", [one_c, scale_ffn], f"{prefix}.scale_ffn1")],
                f"{prefix}.x2_scaled",
            ),
            shift_ffn,
        ],
        f"{prefix}.x2",
    )
    gate_up = _linear(
        b, x2, declare(f"{prefix}.gate_up_proj.weight"), None, f"{prefix}.gate_up"
    )
    swiglu_gate = _slice_last_dim(b, gate_up, 0, intermediate, f"{prefix}.sw_gate")
    swiglu_up = _slice_last_dim(
        b, gate_up, intermediate, 2 * intermediate, f"{prefix}.sw_up"
    )
    ffn = b.op(
        "Mul",
        [_silu(b, swiglu_gate, f"{prefix}.sw_act"), swiglu_up],
        f"{prefix}.ffn_mul",
    )
    down = _linear(
        b, ffn, declare(f"{prefix}.down_proj.weight"), None, f"{prefix}.down"
    )
    return b.op(
        "Add",
        [
            residual2,
            b.op(
                "Mul",
                [b.op("Add", [one_c, gate_ffn], f"{prefix}.gate_ffn1"), down],
                f"{prefix}.down_gated",
            ),
        ],
        f"{prefix}.h2",
    )


def reconstruct_qwen_drive_planning_expert(
    hf_dir: str,
    num_samples: int,
    num_steps: Optional[int] = None,
) -> onnx.ModelProto:
    """Builds the planning expert's flow-matching sampler as one ONNX graph,
    from a ``qwen_drive_planning_expert`` checkpoint directory.

    :param hf_dir: checkpoint directory (``config.json`` + safetensors). The
            weight keys are the ``PlanningExpert`` module's own
            (``planning_expert.``-unprefixed) state-dict keys -- the
            ``planning_expert.`` prefix is also tried per key, matching
            ``QwenDriveForPlanning.load_planner``'s own prefix-stripping.
    :param num_samples: static number of independently-noised trajectory
            samples to draw per call (``generate_trajectory``'s own
            ``num_samples``).
    :param num_steps: static Euler-integration step count; defaults to the
            checkpoint's own ``num_inference_steps``.
    :returns: the constructed, hydrated model.

            Inputs: ``scene_key_i``/``scene_value_i`` for
            ``i in range(num_hidden_layers // layers_per_kv)``
            (``float32[1, prefix_len, num_key_value_heads, head_dim]``,
            ``prefix_len`` a dynamic axis -- the VLM's post-rotary cached
            keys/values for its ``full_attention`` layers),
            ``position_anchor`` (``int64[3, 1]``, the M-RoPE position of the
            prefix's last token), ``history``
            (``float32[1, num_history_points, trajectory_point_dim]``),
            ``history_velocity``/``history_acceleration``
            (``float32[1, num_history_points, history_dynamics_dim]``),
            ``nav_command`` (``int64[1]``), ``ego_status``
            (``float32[1, ego_status_dim]``), ``noise``
            (``float32[num_samples, num_future_points,
            trajectory_point_dim]``, raw standard-normal draws).

            Output: ``trajectories``
            (``float32[num_samples, num_future_points,
            trajectory_point_dim]``, metres/radians, ego frame).
    """
    config = read_hf_config(hf_dir)
    if config.get("model_type") != _SUPPORTED_MODEL_TYPE:
        raise UnsupportedArchitectureError(
            f"model_type={config.get('model_type')!r}, expected {_SUPPORTED_MODEL_TYPE!r}"
        )
    entries = _index_safetensors_checkpoint(hf_dir)
    b = _Builder()
    declared: Dict[str, str] = {}

    def declare(name: str) -> str:
        # Every conditioning encoder (`time_mlp`/`nav_mlp`/`ego_mlp`/...) and
        # every transformer layer is invoked once per Euler step (see the
        # sampling loop below), so the *same* weight is requested many times
        # -- memoized here so each checkpoint tensor becomes exactly one
        # initializer, not one per call site.
        cached = declared.get(name)
        if cached is not None:
            return cached
        resolved = name
        entry = entries.get(resolved)
        if entry is None:
            resolved = f"planning_expert.{name}"
            entry = entries.get(resolved)
        if entry is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor {name!r} "
                f"(also tried {resolved!r})"
            )
        b.initializers.append(_read_tensor(entry, resolved))
        if entry.dtype in ("F32", "BF16"):
            result = resolved
        else:
            result = b.op(
                "Cast", [resolved], f"{resolved}.f32", to=onnx.TensorProto.FLOAT
            )
        declared[name] = result
        return result

    hidden = int(_cfg(config, "hidden_size"))
    intermediate = int(_cfg(config, "intermediate_size"))
    num_hidden_layers = int(_cfg(config, "num_hidden_layers"))
    num_heads = int(_cfg(config, "num_attention_heads"))
    num_kv_heads = int(_cfg(config, "num_key_value_heads"))
    head_dim = int(_cfg(config, "head_dim"))
    layers_per_kv = int(_cfg(config, "layers_per_kv"))
    eps = float(_cfg(config, "rms_norm_eps"))
    time_embed_dim = int(_cfg(config, "time_embed_dim"))
    time_embed_scale = float(_cfg(config, "time_embed_scale"))
    fourier_num_features = int(_cfg(config, "fourier_num_features"))
    fourier_max_frequency = float(_cfg(config, "fourier_max_frequency"))
    nav_classes = int(_cfg(config, "nav_command_classes"))
    ego_dim = int(_cfg(config, "ego_status_dim"))
    dyn_dim = int(_cfg(config, "history_dynamics_dim"))
    rope_theta = float(_cfg(config, "rope_theta"))
    partial_rotary_factor = float(_cfg(config, "partial_rotary_factor"))
    mrope_section = list(_cfg(config, "mrope_section"))
    num_future_points = int(_cfg(config, "num_future_points"))
    num_history_points = int(_cfg(config, "num_history_points"))
    point_dim = int(_cfg(config, "trajectory_point_dim"))
    trajectory_scale = list(_cfg(config, "trajectory_scale"))
    num_inference_steps = int(
        num_steps if num_steps is not None else _cfg(config, "num_inference_steps")
    )
    noise_init_std = float(_cfg(config, "noise_init_std"))
    min_one_minus_t = float(_cfg(config, "min_one_minus_t"))

    rotary_dim = int(head_dim * partial_rotary_factor)
    num_kv_sources = num_hidden_layers // layers_per_kv
    batch = num_samples
    length = num_future_points

    scene_key_names = [f"scene_key_{j}" for j in range(num_kv_sources)]
    scene_value_names = [f"scene_value_{j}" for j in range(num_kv_sources)]
    position_anchor_in = "position_anchor"
    history_in = "history"
    history_velocity_in = "history_velocity"
    history_acceleration_in = "history_acceleration"
    nav_command_in = "nav_command"
    ego_status_in = "ego_status"
    noise_in = "noise"

    graph_inputs: List[onnx.ValueInfoProto] = []
    for kname, vname in zip(scene_key_names, scene_value_names):
        for name in (kname, vname):
            graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    name,
                    onnx.TensorProto.FLOAT,
                    [1, "prefix_len", num_kv_heads, head_dim],
                )
            )
    graph_inputs += [
        onnx.helper.make_tensor_value_info(
            position_anchor_in, onnx.TensorProto.INT64, [3, 1]
        ),
        onnx.helper.make_tensor_value_info(
            history_in, onnx.TensorProto.FLOAT, [1, num_history_points, point_dim]
        ),
        onnx.helper.make_tensor_value_info(
            history_velocity_in,
            onnx.TensorProto.FLOAT,
            [1, num_history_points, dyn_dim],
        ),
        onnx.helper.make_tensor_value_info(
            history_acceleration_in,
            onnx.TensorProto.FLOAT,
            [1, num_history_points, dyn_dim],
        ),
        onnx.helper.make_tensor_value_info(nav_command_in, onnx.TensorProto.INT64, [1]),
        onnx.helper.make_tensor_value_info(
            ego_status_in, onnx.TensorProto.FLOAT, [1, ego_dim]
        ),
        onnx.helper.make_tensor_value_info(
            noise_in,
            onnx.TensorProto.FLOAT,
            [num_samples, num_future_points, point_dim],
        ),
    ]

    # --- VLM cache: expand to `batch` and transpose to [B, G, P, D], once --
    scene_kv_t: List[Tuple[str, str]] = []
    for j, (kname, vname) in enumerate(zip(scene_key_names, scene_value_names)):
        k_exp = _expand_batch(b, kname, batch, f"scene_kv{j}.k_exp")
        v_exp = _expand_batch(b, vname, batch, f"scene_kv{j}.v_exp")
        k_t = b.op("Transpose", [k_exp], f"scene_kv{j}.k_t", perm=[0, 2, 1, 3])
        v_t = b.op("Transpose", [v_exp], f"scene_kv{j}.v_t", perm=[0, 2, 1, 3])
        scene_kv_t.append((k_t, v_t))

    # --- trajectory (de)normalization --------------------------------
    scale_c = b.const(
        np.array(trajectory_scale, dtype=np.float32).reshape(1, 1, point_dim),
        prefix="traj_scale",
    )
    normalized_history = _normalize_history(
        b, history_in, scale_c, num_history_points, "norm_hist"
    )

    # --- conditioning fixed for the whole sample() call ----------------
    nav_onehot = _one_hot_masked(b, nav_command_in, nav_classes, "nav_onehot")
    hist_flat = b.op(
        "Reshape",
        [normalized_history, b.shape_const([1, (num_history_points - 1) * point_dim])],
        "hist_flat",
    )
    hist_nav_cat = b.op("Concat", [hist_flat, nav_onehot], "hist_nav_cat", axis=-1)
    pose_q = _mlp_block(
        b,
        hist_nav_cat,
        declare,
        "history_encoder",
        (num_history_points - 1) * point_dim + nav_classes,
        hidden,
    )
    vel_flat = b.op(
        "Reshape",
        [history_velocity_in, b.shape_const([1, num_history_points * dyn_dim])],
        "vel_flat",
    )
    vel_q = _mlp_block(
        b,
        vel_flat,
        declare,
        "history_velocity_encoder",
        num_history_points * dyn_dim,
        hidden,
    )
    acc_flat = b.op(
        "Reshape",
        [history_acceleration_in, b.shape_const([1, num_history_points * dyn_dim])],
        "acc_flat",
    )
    acc_q = _mlp_block(
        b,
        acc_flat,
        declare,
        "history_acceleration_encoder",
        num_history_points * dyn_dim,
        hidden,
    )
    nav_out = _mlp_block(b, nav_onehot, declare, "nav_mlp", nav_classes, hidden)
    ego_out = _mlp_block(b, ego_status_in, declare, "ego_mlp", ego_dim, hidden)
    fixed_condition = b.op("Add", [nav_out, ego_out], "fixed_condition")

    waypoint_embed_w = declare("waypoint_embed.weight")
    waypoint_embed_unsq = _unsqueeze(b, waypoint_embed_w, [0], "waypoint_embed_unsq")
    waypoint_embed_exp = b.op(
        "Expand",
        [waypoint_embed_unsq, b.shape_const([batch, length, hidden])],
        "waypoint_embed_exp",
    )

    # --- waypoint rotary embedding, computed once (see docstring, opt. 1) --
    steps_c = b.const(np.arange(1, length + 1, dtype=np.int64), prefix="wp_steps")
    anchor_unsq = _unsqueeze(b, position_anchor_in, [-1], "anchor_unsq")
    positions = b.op("Add", [anchor_unsq, steps_c], "positions")
    cos, sin = _text_mrope_cos_sin(
        b, positions, rotary_dim, rope_theta, mrope_section, "wp_rope"
    )
    cos = _unsqueeze(b, cos, [2], "wp_rope.cos.unsq")
    sin = _unsqueeze(b, sin, [2], "wp_rope.sin.unsq")

    freqs_c = b.const(
        _fourier_freqs_np(fourier_num_features, fourier_max_frequency),
        prefix="fourier_freqs",
    )

    noise_scale_c = b.const(
        np.array(noise_init_std, dtype=np.float32), prefix="noise_scale"
    )
    waypoints = b.op("Mul", [noise_in, noise_scale_c], "waypoints_init")

    step_size = 1.0 / num_inference_steps
    for step in range(num_inference_steps):
        t = step * step_size
        time_np = _sinusoidal_time_embedding_np(
            t, time_embed_dim, time_embed_scale
        ).reshape(1, time_embed_dim)
        time_const = b.const(time_np, prefix=f"time_embed_{step}")
        time_condition = _mlp_block(
            b, time_const, declare, "time_mlp", time_embed_dim, hidden
        )
        condition = b.op("Add", [time_condition, fixed_condition], f"condition_{step}")

        traj_proj = _linear(
            b,
            waypoints,
            declare("trajectory_proj.weight"),
            declare("trajectory_proj.bias"),
            f"traj_proj_{step}",
        )
        fourier = _fourier_encoder(
            b,
            waypoints,
            freqs_c,
            declare,
            "fourier_encoder",
            point_dim,
            fourier_num_features,
            hidden,
            batch,
            length,
        )
        time_exp = b.op(
            "Expand",
            [time_condition, b.shape_const([batch, length, hidden])],
            f"time_exp_{step}",
        )
        pose_exp = b.op(
            "Expand",
            [pose_q, b.shape_const([batch, length, hidden])],
            f"pose_exp_{step}",
        )
        vel_exp = b.op(
            "Expand", [vel_q, b.shape_const([batch, length, hidden])], f"vel_exp_{step}"
        )
        acc_exp = b.op(
            "Expand", [acc_q, b.shape_const([batch, length, hidden])], f"acc_exp_{step}"
        )
        fused = b.op(
            "Concat",
            [
                traj_proj,
                fourier,
                time_exp,
                pose_exp,
                waypoint_embed_exp,
                vel_exp,
                acc_exp,
            ],
            f"query_fusion_in_{step}",
            axis=-1,
        )
        hidden_states = _mlp_block(
            b, fused, declare, "query_fusion", hidden * 7, hidden
        )

        for layer_idx in range(num_hidden_layers):
            kv_idx = layer_idx // layers_per_kv
            scene_key_t, scene_value_t = scene_kv_t[kv_idx]
            hidden_states = _planning_expert_layer(
                b,
                hidden_states,
                scene_key_t,
                scene_value_t,
                cos,
                sin,
                rotary_dim,
                condition,
                declare,
                f"layers.{layer_idx}",
                hidden,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate,
                eps,
                batch,
                length,
            )

        normed = _rmsnorm(
            b, hidden_states, declare("final_layernorm.weight"), eps, f"final_ln_{step}"
        )
        endpoint = _linear(
            b,
            normed,
            declare("out_proj.weight"),
            declare("out_proj.bias"),
            f"out_proj_{step}",
        )

        remaining = max(1.0 - t, min_one_minus_t)
        alpha_c = b.const(
            np.array(step_size / remaining, dtype=np.float32), prefix=f"alpha_{step}"
        )
        diff = b.op("Sub", [endpoint, waypoints], f"diff_{step}")
        waypoints = b.op(
            "Add",
            [waypoints, b.op("Mul", [diff, alpha_c], f"scaled_{step}")],
            f"waypoints_{step}",
        )

    trajectories = _denormalize_trajectory(b, waypoints, scale_c, "trajectories")

    graph = onnx.helper.make_graph(
        b.nodes,
        "qwen_drive_planning_expert",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                trajectories,
                onnx.TensorProto.FLOAT,
                [num_samples, num_future_points, point_dim],
            )
        ],
        b.initializers,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )
    return model
