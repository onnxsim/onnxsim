"""Builds runnable ONNX graphs *and* their weights directly from a
Qwen3.5-VL-family HuggingFace checkpoint directory (``config.json`` +
``*.safetensors``) -- the vision-language backbone behind
``Qwen/Qwen-Drive-1.0-4B``'s ``vlm_config`` (``model_type: "qwen3_5"``,
``architectures: ["Qwen3_5ForConditionalGeneration"]``).

Same "known architecture template, build from declared hyperparameters,
hydrate with the checkpoint's own tensors" approach as
:mod:`onnxsim.hf_reconstruct`/:mod:`onnxsim.gguf_reconstruct`, and reuses
those modules' safetensors reader and op-building helpers (``_Builder``,
``_linear``, ``_unsqueeze``, ``_slice_last_dim``, ``_rotate_half``)
unchanged. Every architectural detail below was confirmed against the real
``transformers`` source on the ``huggingface/transformers`` ``main`` branch
at the time this was written (not yet in a tagged release):
``src/transformers/models/qwen3_5/modeling_qwen3_5.py``,
``configuration_qwen3_5.py``, and ``src/transformers/vision_utils.py``
(``get_vision_interpolation_indices_and_weights``/``get_vision_position_ids``/
``_interpolation_axis_taps_weights``) -- not guessed from the architecture's
paper/README description.

**Two separate graphs, not one combined graph** -- :func:`reconstruct_qwen3_5_vlm`
returns ``{"vision_encoder": ..., "language_model": ...}``:

- ``vision_encoder``: ``pixel_values`` (pre-flattened patches, see
  :class:`Qwen3_5VisionPatchEmbed`'s own ``.view()`` call) -> ``image_embeds``
  (one row per *merged* spatial-merge-block of ``spatial_merge_size**2``
  patches).
- ``language_model``: ``inputs_embeds`` (already-merged text+image
  embeddings) + ``position_ids`` (the 3-axis M-RoPE ``[3, batch, seq]``
  layout ``Qwen3_5TextModel`` itself takes) -> ``logits``.

Splitting the graph this way -- rather than reproducing the real model's
``masked_scatter``-based merge of image embeddings into the text embedding
sequence as ONNX ops -- sidesteps the one piece of this architecture that is
genuinely data-dependent (*where* the image tokens sit in ``input_ids`` is
not known until the actual prompt is fixed): every other graph in this
module's family already commits to a caller-chosen static ``batch_size``/
``seq_len`` (see :mod:`onnxsim.hf_reconstruct`'s scope note), and the token
embedding lookup (``embed_tokens.weight[input_ids]``) plus the splice that
drops ``image_embeds`` into the text embedding sequence at the (also
caller-known, once a prompt is fixed) image-token span is a trivial
``numpy``/``onnxruntime`` operation outside either graph -- not something
that needs its own ONNX subgraph. This is the same "onnx.utils.extract_model
one graph into several, or build the several graphs directly" split
real-world multi-file VLM ONNX exports already use (encoder/decoder splits
for seq2seq models, a separate vision tower for LLaVA-style models) --
:mod:`onnxsim.transformers_export`'s own multi-file encoder/decoder(-with-past)
export is this module's own family's version of the same idea.

Scope, deliberately narrower than a full multimodal generation pipeline:

- **Single image, no video.** ``grid_thw`` is a single caller-chosen
  ``(t, h, w)`` (``t`` is baked in as ``1``), not a batch of images/video
  clips -- multi-image packing (the vision tower's ``cu_seqlens``-driven
  attention splitting across several images) is real, confirmed-real-code
  behavior this does not build; a single image is exactly one packed
  attention segment, i.e. plain full attention over every patch, so that
  path is what gets built.
- **No KV cache / incremental decode.** Same "single-shape prefill-style
  forward pass" scope as :mod:`onnxsim.hf_reconstruct`. The linear-attention
  (Gated DeltaNet) layers therefore always use the checkpoint's own
  *sequential*, per-token recurrence (``torch_recurrent_gated_delta_rule``
  in the real source) rather than its *chunked*, UT-transform-parallelized
  form (``torch_chunk_gated_delta_rule``) -- confirmed mathematically
  identical (the real module's own docstring: "Same args and return value
  as torch_chunk_gated_delta_rule"; chunking is a training/prefill
  throughput optimization over the same recurrence, not a different
  function), and far simpler to lay out as a static graph: an explicit,
  Python-unrolled per-timestep update chain (no ONNX ``Loop`` subgraph)
  over the recurrent state, one graph node group per sequence position.
  This scales linearly in nodes with ``seq_len`` -- fine at the small
  scale this template targets (tests, short prompts), not a substitute for
  a real chunked/parallel-scan kernel at production sequence lengths.
- **No ``attention_mask`` / padding.** Every position is assumed valid,
  same omission :func:`onnxsim.hf_reconstruct._reconstruct_llama_family_hf`
  already makes.
- Weight key layout assumes the *standalone* ``Qwen3_5ForConditionalGeneration``
  naming confirmed from source (``Qwen3_5Model.visual``/``.language_model``,
  i.e. ``model.visual.*``/``model.language_model.*``/``lm_head.weight``).
  A checkpoint that wraps this VLM inside another module (e.g.
  ``Qwen-Drive-1.0-4B``'s own ``QwenDriveForPlanning.vlm``, whose exact
  wrapping prefix was not confirmed against a real downloaded checkpoint --
  only its ``config.json`` was) needs its own thin prefix adapter; see
  ``key_prefix``.

Precision/BF16 handling, safetensors parsing, and the whole "known
template, fail clearly otherwise" philosophy are unchanged from
:mod:`onnxsim.hf_reconstruct` -- see that module's docstring.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnx.helper

from onnxsim.gguf_reconstruct import (
    _IR_VERSION,
    _OPSET,
    UnsupportedArchitectureError,
    _Builder,
    _linear,
    _rotate_half,
    _slice_last_dim,
    _unsqueeze,
)
from onnxsim.hf_reconstruct import (
    _index_safetensors_checkpoint,
    _read_tensor,
    read_hf_config,
)

_SUPPORTED_TEXT_MODEL_TYPE = "qwen3_5_text"
_SUPPORTED_VISION_MODEL_TYPE = "qwen3_5_vision"


def _extract_qwen3_5_configs(config: dict) -> Tuple[dict, dict, dict]:
    """Pulls ``(vlm_config, text_config, vision_config)`` out of a checkpoint's
    top-level ``config.json``, accepting either a plain standalone
    Qwen3.5-VL config (``text_config``/``vision_config`` at the top level)
    or one more level of nesting under ``vlm_config`` -- the shape
    ``Qwen-Drive-1.0-4B``'s own ``config.json`` uses (a ``QwenDriveForPlanning``
    wrapper around ``vlm_config`` + a separate, unrelated ``expert_config``
    this module does not build)."""
    vlm_config: dict = config
    if "text_config" not in config or "vision_config" not in config:
        nested = config.get("vlm_config")
        if not isinstance(nested, dict):
            raise UnsupportedArchitectureError(
                "config has no text_config/vision_config, and no vlm_config "
                "wrapping them either -- not a Qwen3.5-VL-shaped checkpoint"
            )
        vlm_config = nested
    text_config = vlm_config.get("text_config")
    vision_config = vlm_config.get("vision_config")
    if text_config is None or vision_config is None:
        raise UnsupportedArchitectureError(
            "vlm_config is missing text_config or vision_config"
        )
    if text_config.get("model_type") != _SUPPORTED_TEXT_MODEL_TYPE:
        raise UnsupportedArchitectureError(
            f"text_config.model_type={text_config.get('model_type')!r}, "
            f"expected {_SUPPORTED_TEXT_MODEL_TYPE!r}"
        )
    if vision_config.get("model_type") != _SUPPORTED_VISION_MODEL_TYPE:
        raise UnsupportedArchitectureError(
            f"vision_config.model_type={vision_config.get('model_type')!r}, "
            f"expected {_SUPPORTED_VISION_MODEL_TYPE!r}"
        )
    return vlm_config, text_config, vision_config


def _rmsnorm_zero_centered(
    b: _Builder, x: str, weight_name: str, eps: float, prefix: str
) -> str:
    """``Qwen3_5RMSNorm``: unlike the Llama-family RMSNorm (weight
    initialized to ones, multiplies the normalized value directly -- see
    :func:`onnxsim.gguf_reconstruct._rmsnorm`), confirmed from real source
    (``modeling_qwen3_5.py``'s ``Qwen3_5RMSNorm.forward``/``_init_weights``)
    this norm's own weight is initialized to *zeros* and used as
    ``normed * (1 + weight)`` -- "Llama does x.to(float16) * w whilst
    Qwen3_5 is (x * w).to(float16)" per that file's own comment. Used for
    every plain norm in this architecture (input/post-attention layernorms,
    q_norm/k_norm, the final norm) -- *not* ``Qwen3_5RMSNormGated`` (see
    :func:`_rmsnorm_gated`), which keeps the ordinary ones-initialized
    convention."""
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="eps")
    one_c = b.const(np.array(1.0, dtype=np.float32), prefix="one")
    x2 = b.op("Mul", [x, x], f"{prefix}.sq")
    mean = b.op("ReduceMean", [x2], f"{prefix}.mean", axes=[-1], keepdims=1)
    var_eps = b.op("Add", [mean, eps_c], f"{prefix}.var_eps")
    rms = b.op("Sqrt", [var_eps], f"{prefix}.rms")
    normed = b.op("Div", [x, rms], f"{prefix}.normed")
    weight_p1 = b.op("Add", [weight_name, one_c], f"{prefix}.weight_p1")
    return b.op("Mul", [normed, weight_p1], f"{prefix}.scaled")


def _rmsnorm_gated(
    b: _Builder, x: str, gate: str, weight_name: str, eps: float, prefix: str
) -> str:
    """``Qwen3_5RMSNormGated``: ordinary (ones-initialized) RMSNorm, then
    multiplied by ``silu(gate)`` -- confirmed norm-then-gate ordering from
    real source ("Norm before gate")."""
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="eps")
    x2 = b.op("Mul", [x, x], f"{prefix}.sq")
    mean = b.op("ReduceMean", [x2], f"{prefix}.mean", axes=[-1], keepdims=1)
    var_eps = b.op("Add", [mean, eps_c], f"{prefix}.var_eps")
    rms = b.op("Sqrt", [var_eps], f"{prefix}.rms")
    normed = b.op("Div", [x, rms], f"{prefix}.normed")
    normed = b.op("Mul", [normed, weight_name], f"{prefix}.scaled")
    gate_sig = b.op("Sigmoid", [gate], f"{prefix}.gate_sig")
    silu_gate = b.op("Mul", [gate, gate_sig], f"{prefix}.silu_gate")
    return b.op("Mul", [normed, silu_gate], prefix)


def _reduce_sum(
    b: _Builder, x: str, axes: List[int], keepdims: int, prefix: str
) -> str:
    """``ReduceSum``'s reduction axes moved from an attribute to an
    (optional, second) *input* at opset 13 -- unlike ``ReduceMean``, whose
    axes stayed an attribute until opset 18 (see
    :mod:`onnxsim.gguf_reconstruct`'s own opset comment). This module
    targets opset 17 (> 13), so every ``ReduceSum`` here needs its axes as
    a const input, not a kwarg."""
    axes_c = b.const(np.array(axes, dtype=np.int64), prefix="reduce_axes")
    return b.op("ReduceSum", [x, axes_c], prefix, keepdims=keepdims)


def _l2norm_last_dim(b: _Builder, x: str, eps: float, prefix: str) -> str:
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="l2eps")
    x2 = b.op("Mul", [x, x], f"{prefix}.sq")
    ss = _reduce_sum(b, x2, [-1], 1, f"{prefix}.ss")
    ss_eps = b.op("Add", [ss, eps_c], f"{prefix}.ss_eps")
    inv_norm = b.op("Sqrt", [ss_eps], f"{prefix}.norm")
    return b.op("Div", [x, inv_norm], prefix)


def _silu(b: _Builder, x: str, prefix: str) -> str:
    sig = b.op("Sigmoid", [x], f"{prefix}.sig")
    return b.op("Mul", [x, sig], prefix)


def _gelu_tanh(b: _Builder, x: str, prefix: str) -> str:
    """``gelu_pytorch_tanh`` -- the tanh approximation, used by the vision
    tower's own MLP (``config.hidden_act`` for ``qwen3_5_vision``). Built
    from ``Tanh`` rather than the native ``Gelu`` op, which only exists from
    opset 20 (this module targets opset 17, same as the rest of this
    family)."""
    c0 = b.const(np.array(0.5, dtype=np.float32), prefix="gelu_half")
    c1 = b.const(np.array(1.0, dtype=np.float32), prefix="gelu_one")
    csqrt = b.const(
        np.array(np.sqrt(2.0 / np.pi), dtype=np.float32), prefix="gelu_sqrt2opi"
    )
    ccube = b.const(np.array(0.044715, dtype=np.float32), prefix="gelu_cube_c")
    x3 = b.op("Mul", [x, b.op("Mul", [x, x], f"{prefix}.x2")], f"{prefix}.x3")
    inner = b.op(
        "Add", [x, b.op("Mul", [x3, ccube], f"{prefix}.x3c")], f"{prefix}.inner"
    )
    inner = b.op("Mul", [inner, csqrt], f"{prefix}.inner_scaled")
    t = b.op("Tanh", [inner], f"{prefix}.tanh")
    t1 = b.op("Add", [t, c1], f"{prefix}.tanh_p1")
    return b.op("Mul", [b.op("Mul", [x, c0], f"{prefix}.half_x"), t1], prefix)


def _gelu_erf(b: _Builder, x: str, prefix: str) -> str:
    """Exact (erf-based) GELU -- ``nn.GELU()``'s default, used by the vision
    patch merger (a plain ``nn.GELU()`` in real source, distinct from the
    vision blocks' own ``gelu_pytorch_tanh``)."""
    c0 = b.const(np.array(0.5, dtype=np.float32), prefix="gelu_half")
    c1 = b.const(np.array(1.0, dtype=np.float32), prefix="gelu_one")
    inv_sqrt2 = b.const(
        np.array(1.0 / np.sqrt(2.0), dtype=np.float32), prefix="inv_sqrt2"
    )
    erf = b.op(
        "Erf", [b.op("Mul", [x, inv_sqrt2], f"{prefix}.scaled")], f"{prefix}.erf"
    )
    return b.op(
        "Mul",
        [
            b.op("Mul", [x, c0], f"{prefix}.half_x"),
            b.op("Add", [erf, c1], f"{prefix}.erf_p1"),
        ],
        prefix,
    )


def _slice_axis(
    b: _Builder, x: str, axis: int, start: int, end: int, prefix: str
) -> str:
    starts = b.const(np.array([start], dtype=np.int64), prefix="slice_start")
    ends = b.const(np.array([end], dtype=np.int64), prefix="slice_end")
    axes = b.const(np.array([axis], dtype=np.int64), prefix="slice_axis")
    return b.op("Slice", [x, starts, ends, axes], prefix)


def _squeeze(b: _Builder, x: str, axes: List[int], prefix: str) -> str:
    axes_c = b.const(np.array(axes, dtype=np.int64), prefix="squeeze_axes")
    return b.op("Squeeze", [x, axes_c], prefix)


def _repeat_interleave_heads(
    b: _Builder, x: str, dims: List[int], n_rep: int, prefix: str
) -> str:
    """``x.repeat_interleave(n_rep, dim=head_axis)`` for a ``[..., n_heads,
    head_dim]``-shaped tensor with static `dims` (the shape *before*
    repeating): duplicates each head `n_rep` times consecutively (head order
    ``[0, 0, 1, 1, ...]`` for ``n_rep=2``), matching
    ``Qwen3_5GatedDeltaNet.forward``'s own ``query.repeat_interleave(...,
    dim=2)`` exactly (as opposed to grouped-query attention's own
    broadcast-based repeat, which does not need this consecutive-duplicate
    order)."""
    *lead, n_heads, head_dim = dims
    x = _unsqueeze(b, x, [len(lead) + 1], f"{prefix}.unsq")
    tile_reps = np.ones(len(lead) + 3, dtype=np.int64)
    tile_reps[len(lead) + 1] = n_rep
    reps_c = b.const(tile_reps, prefix="tile_reps")
    x = b.op("Tile", [x, reps_c], f"{prefix}.tile")
    out_shape = b.shape_const([*lead, n_heads * n_rep, head_dim])
    return b.op("Reshape", [x, out_shape], prefix)


def _apply_partial_rope(
    b: _Builder, x: str, cos: str, sin: str, rotary_dim: int, head_dim: int, prefix: str
) -> str:
    """``apply_rotary_pos_emb`` (text side): only the first `rotary_dim` of
    `head_dim` dims rotate (``partial_rotary_factor``); the rest pass
    through unchanged. `cos`/`sin` already have `rotary_dim` as their last
    dim."""
    if rotary_dim == head_dim:
        rotated = _rotate_half(b, x, head_dim, f"{prefix}.rot")
        a = b.op("Mul", [x, cos], f"{prefix}.a")
        c = b.op("Mul", [rotated, sin], f"{prefix}.c")
        return b.op("Add", [a, c], prefix)
    x_rot = _slice_last_dim(b, x, 0, rotary_dim, f"{prefix}.xrot")
    x_pass = _slice_last_dim(b, x, rotary_dim, head_dim, f"{prefix}.xpass")
    rotated = _rotate_half(b, x_rot, rotary_dim, f"{prefix}.rot")
    a = b.op("Mul", [x_rot, cos], f"{prefix}.a")
    c = b.op("Mul", [rotated, sin], f"{prefix}.c")
    x_embed = b.op("Add", [a, c], f"{prefix}.embed")
    return b.op("Concat", [x_embed, x_pass], prefix, axis=-1)


def _mrope_selection_masks(
    mrope_section: Sequence[int], n_freq: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduces ``Qwen3_5TextRotaryEmbedding.recomposition_frequencies``:
    every rotary frequency-pair index starts assigned to the T axis, then H's
    ``freq[..., 1 : mrope_section[1]*3 : 3]`` and W's ``freq[..., 2 :
    mrope_section[2]*3 : 3]`` slices overwrite their own (disjoint, since
    they differ mod 3) indices -- confirmed against real source, general
    ``mrope_section`` (not hardcoded to Qwen-Drive-1.0's own ``[11, 11,
    10]``). Returned as three ``{0.,1.}`` selection masks over the
    ``n_freq`` frequency-pair axis, one-hot per index, so the actual
    per-axis frequencies (computed at graph-build time from a runtime
    ``position_ids`` input, so *not* something these masks can be folded
    into) can be combined as ``t*sel_t + h*sel_h + w*sel_w``."""
    sel_h = np.zeros(n_freq, dtype=np.float32)
    sel_w = np.zeros(n_freq, dtype=np.float32)
    h_len = min(mrope_section[1] * 3, n_freq)
    for i in range(1, h_len, 3):
        sel_h[i] = 1.0
    w_len = min(mrope_section[2] * 3, n_freq)
    for i in range(2, w_len, 3):
        sel_w[i] = 1.0
    sel_t = 1.0 - sel_h - sel_w
    return sel_t, sel_h, sel_w


def _text_mrope_cos_sin(
    b: _Builder,
    position_ids: str,
    rotary_dim: int,
    rope_theta: float,
    mrope_section: Sequence[int],
    prefix: str,
) -> Tuple[str, str]:
    """`position_ids` is `[3, batch, seq]` int64 (T/H/W rows, exactly
    ``Qwen3_5TextModel.forward``'s own convention once its leading "text"
    row -- present only when a caller passes the 4-row legacy layout -- has
    been dropped). Returns `(cos, sin)` each `[batch, seq, rotary_dim]`."""
    n_freq = rotary_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    )
    inv_freq_c = b.const(
        inv_freq.reshape(1, 1, -1).astype(np.float32), prefix=f"{prefix}.inv_freq"
    )
    sel_t, sel_h, sel_w = _mrope_selection_masks(mrope_section, n_freq)
    sel_t_c = b.const(sel_t.reshape(1, 1, -1), prefix=f"{prefix}.sel_t")
    sel_h_c = b.const(sel_h.reshape(1, 1, -1), prefix=f"{prefix}.sel_h")
    sel_w_c = b.const(sel_w.reshape(1, 1, -1), prefix=f"{prefix}.sel_w")

    axis_freqs = []
    for axis in range(3):
        pos = _slice_axis(b, position_ids, 0, axis, axis + 1, f"{prefix}.pos{axis}")
        pos = _squeeze(b, pos, [0], f"{prefix}.pos{axis}.sq")
        pos_f = b.op("Cast", [pos], f"{prefix}.pos{axis}.f", to=onnx.TensorProto.FLOAT)
        pos_f = _unsqueeze(b, pos_f, [-1], f"{prefix}.pos{axis}.unsq")
        axis_freqs.append(b.op("Mul", [pos_f, inv_freq_c], f"{prefix}.freq{axis}"))
    freqs = b.op("Mul", [axis_freqs[0], sel_t_c], f"{prefix}.ft")
    freqs = b.op(
        "Add",
        [freqs, b.op("Mul", [axis_freqs[1], sel_h_c], f"{prefix}.fh")],
        f"{prefix}.fth",
    )
    freqs = b.op(
        "Add",
        [freqs, b.op("Mul", [axis_freqs[2], sel_w_c], f"{prefix}.fw")],
        f"{prefix}.fthw",
    )
    emb = b.op("Concat", [freqs, freqs], f"{prefix}.emb", axis=-1)
    cos = b.op("Cos", [emb], f"{prefix}.cos")
    sin = b.op("Sin", [emb], f"{prefix}.sin")
    return cos, sin


def _gated_delta_net_layer(
    b: _Builder,
    h: str,
    declare,
    p: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_dim: int,
    eps: float,
) -> str:
    """``Qwen3_5GatedDeltaNet.forward``, confirmed against real source --
    see this module's own docstring for the "sequential, not chunked"
    scope note."""
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim

    mixed_qkv = _linear(
        b, h, declare(f"{p}.in_proj_qkv.weight"), None, f"{p}.in_proj_qkv"
    )
    z = _linear(b, h, declare(f"{p}.in_proj_z.weight"), None, f"{p}.in_proj_z")
    beta_pre = _linear(b, h, declare(f"{p}.in_proj_b.weight"), None, f"{p}.in_proj_b")
    a_pre = _linear(b, h, declare(f"{p}.in_proj_a.weight"), None, f"{p}.in_proj_a")

    # Causal depthwise Conv1d (kernel=conv_kernel_dim, left-padded by
    # kernel-1, groups=conv_dim) + SiLU -- ``causal_conv1d_fn``. ONNX's own
    # `Conv` op takes the checkpoint's [conv_dim, 1, kernel] weight
    # unchanged (PyTorch's own depthwise-conv1d weight layout).
    mixed_qkv_t = b.op("Transpose", [mixed_qkv], f"{p}.qkv_t", perm=[0, 2, 1])
    conv_w = declare(f"{p}.conv1d.weight")
    conv_out = b.op(
        "Conv",
        [mixed_qkv_t, conv_w],
        f"{p}.conv",
        kernel_shape=[conv_kernel_dim],
        group=conv_dim,
        pads=[conv_kernel_dim - 1, 0],
    )
    conv_out = _silu(b, conv_out, f"{p}.conv_silu")
    mixed_qkv = b.op("Transpose", [conv_out], f"{p}.qkv_back", perm=[0, 2, 1])

    query = _slice_last_dim(b, mixed_qkv, 0, key_dim, f"{p}.q_split")
    key = _slice_last_dim(b, mixed_qkv, key_dim, 2 * key_dim, f"{p}.k_split")
    value = _slice_last_dim(
        b, mixed_qkv, 2 * key_dim, 2 * key_dim + value_dim, f"{p}.v_split"
    )

    query = b.op(
        "Reshape",
        [query, b.shape_const([batch_size, seq_len, num_k_heads, head_k_dim])],
        f"{p}.q_r",
    )
    key = b.op(
        "Reshape",
        [key, b.shape_const([batch_size, seq_len, num_k_heads, head_k_dim])],
        f"{p}.k_r",
    )
    value = b.op(
        "Reshape",
        [value, b.shape_const([batch_size, seq_len, num_v_heads, head_v_dim])],
        f"{p}.v_r",
    )
    z = b.op(
        "Reshape",
        [z, b.shape_const([batch_size, seq_len, num_v_heads, head_v_dim])],
        f"{p}.z_r",
    )

    beta = b.op("Sigmoid", [beta_pre], f"{p}.beta")
    a_log = declare(f"{p}.A_log")
    dt_bias = declare(f"{p}.dt_bias")
    neg_exp_a_log = b.op(
        "Neg", [b.op("Exp", [a_log], f"{p}.a_log_exp")], f"{p}.neg_exp_a_log"
    )
    softplus_in = b.op("Add", [a_pre, dt_bias], f"{p}.softplus_in")
    g = b.op(
        "Mul",
        [neg_exp_a_log, b.op("Softplus", [softplus_in], f"{p}.softplus")],
        f"{p}.g",
    )

    n_rep = num_v_heads // num_k_heads
    if n_rep > 1:
        query = _repeat_interleave_heads(
            b,
            query,
            [batch_size, seq_len, num_k_heads, head_k_dim],
            n_rep,
            f"{p}.q_rep",
        )
        key = _repeat_interleave_heads(
            b, key, [batch_size, seq_len, num_k_heads, head_k_dim], n_rep, f"{p}.k_rep"
        )

    query = _l2norm_last_dim(b, query, 1e-6, f"{p}.q_l2")
    key = _l2norm_last_dim(b, key, 1e-6, f"{p}.k_l2")
    inv_sqrt_hk = b.const(
        np.array(1.0 / np.sqrt(head_k_dim), dtype=np.float32), prefix="inv_sqrt_hk"
    )
    query = b.op("Mul", [query, inv_sqrt_hk], f"{p}.q_scaled")

    state_shape = [batch_size, num_v_heads, head_k_dim, head_v_dim]
    state = b.const(np.zeros(state_shape, dtype=np.float32), prefix=f"{p}.state0")
    outputs = []
    for t in range(seq_len):
        q_t = _squeeze(
            b, _slice_axis(b, query, 1, t, t + 1, f"{p}.q{t}"), [1], f"{p}.q{t}.sq"
        )
        k_t = _squeeze(
            b, _slice_axis(b, key, 1, t, t + 1, f"{p}.k{t}"), [1], f"{p}.k{t}.sq"
        )
        v_t = _squeeze(
            b, _slice_axis(b, value, 1, t, t + 1, f"{p}.v{t}"), [1], f"{p}.v{t}.sq"
        )
        g_t = _squeeze(
            b, _slice_axis(b, g, 1, t, t + 1, f"{p}.g{t}"), [1], f"{p}.g{t}.sq"
        )
        beta_t = _squeeze(
            b, _slice_axis(b, beta, 1, t, t + 1, f"{p}.beta{t}"), [1], f"{p}.beta{t}.sq"
        )

        decay_t = _unsqueeze(
            b, b.op("Exp", [g_t], f"{p}.decay{t}"), [2, 3], f"{p}.decay{t}.unsq"
        )
        state = b.op("Mul", [state, decay_t], f"{p}.state{t}.decay")

        k_col = _unsqueeze(b, k_t, [-1], f"{p}.k{t}.col")
        kv_mem = _reduce_sum(
            b,
            b.op("Mul", [state, k_col], f"{p}.kvmem{t}.mul"),
            [-2],
            0,
            f"{p}.kvmem{t}",
        )
        delta = b.op(
            "Mul",
            [
                b.op("Sub", [v_t, kv_mem], f"{p}.delta{t}.diff"),
                _unsqueeze(b, beta_t, [-1], f"{p}.beta{t}.unsq"),
            ],
            f"{p}.delta{t}",
        )
        update = b.op(
            "Mul",
            [k_col, _unsqueeze(b, delta, [-2], f"{p}.delta{t}.rowunsq")],
            f"{p}.update{t}",
        )
        state = b.op("Add", [state, update], f"{p}.state{t}.upd")

        q_col = _unsqueeze(b, q_t, [-1], f"{p}.q{t}.col")
        out_t = _reduce_sum(
            b, b.op("Mul", [state, q_col], f"{p}.out{t}.mul"), [-2], 0, f"{p}.out{t}"
        )
        outputs.append(_unsqueeze(b, out_t, [1], f"{p}.out{t}.seq"))

    core_attn_out = (
        b.op("Concat", outputs, f"{p}.core_attn_out", axis=1)
        if seq_len > 1
        else outputs[0]
    )
    core_attn_out = b.op(
        "Reshape", [core_attn_out, b.shape_const([-1, head_v_dim])], f"{p}.core_flat"
    )
    z_flat = b.op("Reshape", [z, b.shape_const([-1, head_v_dim])], f"{p}.z_flat")
    normed = _rmsnorm_gated(
        b, core_attn_out, z_flat, declare(f"{p}.norm.weight"), eps, f"{p}.gated_norm"
    )
    normed = b.op(
        "Reshape",
        [normed, b.shape_const([batch_size, seq_len, value_dim])],
        f"{p}.gated_norm_r",
    )
    return _linear(b, normed, declare(f"{p}.out_proj.weight"), None, f"{p}.out_proj")


def _full_attention_layer(
    b: _Builder,
    h: str,
    declare,
    p: str,
    batch_size: int,
    seq_len: int,
    n_head: int,
    n_head_kv: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    cos: str,
    sin: str,
    causal_mask: str,
) -> str:
    """``Qwen3_5Attention.forward``: GQA + per-head zero-centered QK-norm +
    partial M-RoPE + a sigmoid output gate folded into `q_proj`'s own
    doubled output width -- all confirmed against real source."""
    n_rep = n_head // n_head_kv

    q_gate = _linear(b, h, declare(f"{p}.q_proj.weight"), None, f"{p}.q_proj")
    q_gate = b.op(
        "Reshape",
        [q_gate, b.shape_const([batch_size, seq_len, n_head, 2 * head_dim])],
        f"{p}.q_gate_r",
    )
    q = _slice_last_dim(b, q_gate, 0, head_dim, f"{p}.q_only")
    gate = _slice_last_dim(b, q_gate, head_dim, 2 * head_dim, f"{p}.gate_only")
    gate = b.op(
        "Reshape",
        [gate, b.shape_const([batch_size, seq_len, n_head * head_dim])],
        f"{p}.gate_flat",
    )

    k = _linear(b, h, declare(f"{p}.k_proj.weight"), None, f"{p}.k_proj")
    v = _linear(b, h, declare(f"{p}.v_proj.weight"), None, f"{p}.v_proj")
    k = b.op(
        "Reshape",
        [k, b.shape_const([batch_size, seq_len, n_head_kv, head_dim])],
        f"{p}.k_r",
    )
    v = b.op(
        "Reshape",
        [v, b.shape_const([batch_size, seq_len, n_head_kv, head_dim])],
        f"{p}.v_r",
    )

    q = _rmsnorm_zero_centered(b, q, declare(f"{p}.q_norm.weight"), eps, f"{p}.q_norm")
    k = _rmsnorm_zero_centered(b, k, declare(f"{p}.k_norm.weight"), eps, f"{p}.k_norm")

    q = b.op("Transpose", [q], f"{p}.q_t", perm=[0, 2, 1, 3])
    k = b.op("Transpose", [k], f"{p}.k_t", perm=[0, 2, 1, 3])
    v = b.op("Transpose", [v], f"{p}.v_t", perm=[0, 2, 1, 3])

    q = _apply_partial_rope(b, q, cos, sin, rotary_dim, head_dim, f"{p}.q_rope")
    k = _apply_partial_rope(b, k, cos, sin, rotary_dim, head_dim, f"{p}.k_rope")

    q5 = b.op(
        "Reshape",
        [q, b.shape_const([batch_size, n_head_kv, n_rep, seq_len, head_dim])],
        f"{p}.q5",
    )
    k5 = _unsqueeze(b, k, [2], f"{p}.k5")
    v5 = _unsqueeze(b, v, [2], f"{p}.v5")
    k5t = b.op("Transpose", [k5], f"{p}.k5t", perm=[0, 1, 2, 4, 3])
    inv_sqrt_d = b.const(
        np.array(1.0 / np.sqrt(head_dim), dtype=np.float32), prefix="inv_sqrt_d"
    )
    scores = b.op(
        "Mul",
        [b.op("MatMul", [q5, k5t], f"{p}.scores"), inv_sqrt_d],
        f"{p}.scores_scaled",
    )
    scores = b.op("Add", [scores, causal_mask], f"{p}.scores_masked")
    attn = b.op("Softmax", [scores], f"{p}.softmax", axis=-1)
    out5 = b.op("MatMul", [attn, v5], f"{p}.attn_out5")

    out = b.op(
        "Reshape",
        [out5, b.shape_const([batch_size, n_head, seq_len, head_dim])],
        f"{p}.out_r",
    )
    out = b.op("Transpose", [out], f"{p}.out_t", perm=[0, 2, 1, 3])
    out = b.op(
        "Reshape",
        [out, b.shape_const([batch_size, seq_len, n_head * head_dim])],
        f"{p}.out_flat",
    )
    out = b.op("Mul", [out, b.op("Sigmoid", [gate], f"{p}.gate_sig")], f"{p}.out_gated")
    return _linear(b, out, declare(f"{p}.o_proj.weight"), None, f"{p}.o_proj")


def reconstruct_qwen3_5_language_model(
    text_config: dict,
    entries: dict,
    batch_size: int = 1,
    seq_len: int = 8,
    key_prefix: str = "model.language_model.",
    lm_head_prefix: str = "lm_head.weight",
) -> onnx.ModelProto:
    """The Qwen3.5 hybrid linear/full-attention text decoder alone --
    inputs ``inputs_embeds`` (``float32[batch_size, seq_len, hidden_size]``,
    already merged with any image embeddings by the caller -- see this
    module's own docstring) and ``position_ids`` (``int64[3, batch_size,
    seq_len]``, the T/H/W M-RoPE rows), output ``logits``."""
    hidden_size = int(text_config["hidden_size"])
    n_layer = int(text_config["num_hidden_layers"])
    n_head = int(text_config["num_attention_heads"])
    n_head_kv = int(text_config["num_key_value_heads"])
    head_dim = int(text_config.get("head_dim") or (hidden_size // n_head))
    eps = float(text_config.get("rms_norm_eps", 1e-6))
    vocab_size = int(text_config["vocab_size"])
    tie_word_embeddings = bool(text_config.get("tie_word_embeddings", False))
    hidden_act = text_config.get("hidden_act", "silu")
    if hidden_act != "silu":
        raise UnsupportedArchitectureError(
            f"text_config.hidden_act={hidden_act!r} -- this builder's MLP/"
            "Gated-DeltaNet-conv activation is hardcoded to SiLU (the only "
            "value confirmed against a real Qwen3.5 checkpoint)"
        )
    layer_types = text_config["layer_types"]
    rope_params = text_config["rope_parameters"]
    rope_theta = float(rope_params["rope_theta"])
    partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
    mrope_section = rope_params.get("mrope_section", [11, 11, 10])
    rotary_dim = int(head_dim * partial_rotary_factor)

    num_k_heads = int(text_config["linear_num_key_heads"])
    num_v_heads = int(text_config["linear_num_value_heads"])
    head_k_dim = int(text_config["linear_key_head_dim"])
    head_v_dim = int(text_config["linear_value_head_dim"])
    conv_kernel_dim = int(text_config["linear_conv_kernel_dim"])

    b = _Builder()

    def declare(name: str) -> str:
        full_name = key_prefix + name if not name.startswith(lm_head_prefix) else name
        entry = entries.get(full_name)
        if entry is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor {full_name!r}"
            )
        b.initializers.append(_read_tensor(entry, full_name))
        if entry.dtype in ("F32", "BF16"):
            return full_name
        return b.op("Cast", [full_name], f"{full_name}.f32", to=onnx.TensorProto.FLOAT)

    inputs_embeds = "inputs_embeds"
    position_ids = "position_ids"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            inputs_embeds, onnx.TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        ),
        onnx.helper.make_tensor_value_info(
            position_ids, onnx.TensorProto.INT64, [3, batch_size, seq_len]
        ),
    ]

    cos, sin = _text_mrope_cos_sin(
        b, position_ids, rotary_dim, rope_theta, mrope_section, "mrope"
    )
    cos_b = _unsqueeze(b, cos, [1], "mrope.cos_b")
    sin_b = _unsqueeze(b, sin, [1], "mrope.sin_b")
    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    mask_c = b.const(causal_mask, prefix="causal_mask")

    x = inputs_embeds
    for i in range(n_layer):
        p = f"layers.{i}"
        resid = x
        h = _rmsnorm_zero_centered(
            b, x, declare(f"{p}.input_layernorm.weight"), eps, f"{p}.attn_norm"
        )
        if layer_types[i] == "linear_attention":
            attn_out = _gated_delta_net_layer(
                b,
                h,
                declare,
                f"{p}.linear_attn",
                batch_size,
                seq_len,
                hidden_size,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_kernel_dim,
                eps,
            )
        elif layer_types[i] == "full_attention":
            attn_out = _full_attention_layer(
                b,
                h,
                declare,
                f"{p}.self_attn",
                batch_size,
                seq_len,
                n_head,
                n_head_kv,
                head_dim,
                rotary_dim,
                eps,
                cos_b,
                sin_b,
                mask_c,
            )
        else:
            raise UnsupportedArchitectureError(
                f"unknown layer_types[{i}]={layer_types[i]!r}"
            )
        x = b.op("Add", [resid, attn_out], f"{p}.attn_resid")

        resid = x
        h = _rmsnorm_zero_centered(
            b, x, declare(f"{p}.post_attention_layernorm.weight"), eps, f"{p}.ffn_norm"
        )
        gate = _linear(
            b, h, declare(f"{p}.mlp.gate_proj.weight"), None, f"{p}.gate_proj"
        )
        up = _linear(b, h, declare(f"{p}.mlp.up_proj.weight"), None, f"{p}.up_proj")
        act = b.op("Mul", [_silu(b, gate, f"{p}.silu"), up], f"{p}.act")
        ffn_out = _linear(
            b, act, declare(f"{p}.mlp.down_proj.weight"), None, f"{p}.down_proj"
        )
        x = b.op("Add", [resid, ffn_out], f"{p}.ffn_resid")

    x = _rmsnorm_zero_centered(b, x, declare("norm.weight"), eps, "output_norm")

    if not tie_word_embeddings and lm_head_prefix in entries:
        lm_head = declare(lm_head_prefix)
    else:
        lm_head = declare("embed_tokens.weight")
    logits = _linear(b, x, lm_head, None, "lm_head")

    graph = onnx.helper.make_graph(
        b.nodes,
        "qwen3_5_language_model",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                logits, onnx.TensorProto.FLOAT, [batch_size, seq_len, vocab_size]
            )
        ],
        initializer=b.initializers,
    )
    return onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )


def _vision_position_ids(t: int, h: int, w: int, merge: int) -> np.ndarray:
    """Reproduces ``transformers.vision_utils.get_vision_position_ids``
    (``include_temporal=False`` -- Qwen3.5-VL's own vision rotary embedding
    only rotates over H/W, see ``Qwen3_5VisionRotaryEmbedding``'s own
    docstring) for a single image: `(h*w*t, 2)` int array of `(h, w)`
    coordinates, laid out in spatial-merge-block order."""
    hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    block_shape = (h // merge, merge, w // merge, merge)
    hpos_flat = hpos.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
    wpos_flat = wpos.reshape(block_shape).transpose(0, 2, 1, 3).reshape(-1)
    pos = np.stack([hpos_flat, wpos_flat], axis=-1)
    return np.tile(pos, (t, 1))


def _vision_rope_cos_sin(
    t: int, h: int, w: int, merge: int, head_dim: int, rope_theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``Qwen3_5VisionRotaryEmbedding``: axial 2D RoPE, same frequencies for
    H and W, computed here directly as build-time numpy constants (grid_thw
    is a static, caller-chosen build parameter for this whole graph, unlike
    the text side's runtime `position_ids` input) -- confirmed against real
    source (``compute_axial_rope_parameters``/``forward``/
    ``recomposition_frequencies``)."""
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, spatial_dim, 2, dtype=np.float64) / spatial_dim)
    )
    pos = _vision_position_ids(t, h, w, merge).astype(np.float64)
    freq_h = pos[:, 0:1] * inv_freq[None, :]
    freq_w = pos[:, 1:2] * inv_freq[None, :]
    freq_hw = np.concatenate([freq_h, freq_w], axis=-1)
    freq_full = np.concatenate([freq_hw, freq_hw], axis=-1)
    return np.cos(freq_full).astype(np.float32), np.sin(freq_full).astype(np.float32)


def _vision_pos_embed_interp_indices_weights(
    t: int, h: int, w: int, merge: int, num_grid_per_side: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduces ``transformers.vision_utils.get_vision_interpolation_indices_and_weights``
    (``mode="bilinear"``, ``align_corners=True``, ``padding="border"``) for
    a single image: `(h*w*t, 4)` gather indices into the flattened
    `num_grid_per_side x num_grid_per_side` learned position-embedding
    table, and their matching interpolation weights -- confirmed against
    real source (``_interpolation_axis_taps_weights``/
    ``get_vision_interpolation_indices_and_weights``), computed here at
    graph-build time since `grid_thw` is a static parameter (only the
    learned ``pos_embed`` *weights* -- gathered/summed by the graph itself
    -- depend on the checkpoint)."""
    side = num_grid_per_side

    def axis_taps_weights(
        index: np.ndarray, size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        index = index.astype(np.float64)
        src = index * (side - 1) / max(size - 1, 1)
        floor = np.floor(src)
        offsets = np.array([0, 1], dtype=np.float64)
        raw_taps = floor[:, None] + offsets[None, :]
        taps = np.clip(raw_taps, 0, side - 1).astype(np.int64)
        distance = np.abs(src[:, None] - floor[:, None] - offsets[None, :])
        weights = np.clip(1 - distance, a_min=0, a_max=None)
        return taps, weights

    pos = _vision_position_ids(t, h, w, merge)
    row, col = pos[:, 0], pos[:, 1]
    h_taps, h_weights = axis_taps_weights(row, h)
    w_taps, w_weights = axis_taps_weights(col, w)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices.astype(np.int64), weights.astype(np.float32)


def reconstruct_qwen3_5_vision_encoder(
    vision_config: dict,
    entries: dict,
    grid_thw: Tuple[int, int, int] = (1, 4, 4),
    key_prefix: str = "model.visual.",
) -> onnx.ModelProto:
    """The Qwen3.5-VL vision tower alone, for a single image (`grid_thw`'s
    `t` must be `1` -- see this module's own docstring's scope note):
    input ``pixel_values`` (``float32[t*h*w, in_channels *
    temporal_patch_size * patch_size * patch_size]``, already
    patch-flattened and spatial-merge-block-ordered exactly as
    ``Qwen3_5VisionPatchEmbed.forward``'s own ``.view()`` expects -- the
    same convention the real HF image processor's own output uses), output
    ``image_embeds`` (``float32[(t*h*w) / spatial_merge_size**2,
    out_hidden_size]``, one row per merged spatial block)."""
    t, h, w = grid_thw
    if t != 1:
        raise UnsupportedArchitectureError(
            "reconstruct_qwen3_5_vision_encoder only builds a single-image "
            f"(grid_thw[0] == 1) graph, got t={t!r} -- see this module's scope note"
        )
    vision_hidden_act = vision_config.get("hidden_act", "gelu_pytorch_tanh")
    if vision_hidden_act != "gelu_pytorch_tanh":
        raise UnsupportedArchitectureError(
            f"vision_config.hidden_act={vision_hidden_act!r} -- this "
            "builder's vision-block MLP activation is hardcoded to the tanh "
            "GELU approximation (the only value confirmed against a real "
            "Qwen3.5-VL checkpoint)"
        )

    hidden_size = int(vision_config["hidden_size"])
    depth = int(vision_config["depth"])
    num_heads = int(vision_config["num_heads"])
    head_dim = hidden_size // num_heads
    patch_size = int(vision_config["patch_size"])
    temporal_patch_size = int(vision_config["temporal_patch_size"])
    in_channels = int(vision_config["in_channels"])
    spatial_merge_size = int(vision_config["spatial_merge_size"])
    out_hidden_size = int(vision_config["out_hidden_size"])
    num_position_embeddings = int(vision_config["num_position_embeddings"])
    num_grid_per_side = int(round(num_position_embeddings**0.5))
    rope_theta = float(
        vision_config.get("rope_parameters", {}).get("rope_theta", 10000.0)
    )

    if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
        raise UnsupportedArchitectureError(
            f"grid_thw h={h}, w={w} must both be divisible by "
            f"spatial_merge_size={spatial_merge_size}"
        )

    num_patches = t * h * w
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

    b = _Builder()

    def declare(name: str) -> str:
        full_name = key_prefix + name
        entry = entries.get(full_name)
        if entry is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor {full_name!r}"
            )
        b.initializers.append(_read_tensor(entry, full_name))
        if entry.dtype in ("F32", "BF16"):
            return full_name
        return b.op("Cast", [full_name], f"{full_name}.f32", to=onnx.TensorProto.FLOAT)

    pixel_values = "pixel_values"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            pixel_values, onnx.TensorProto.FLOAT, [num_patches, patch_dim]
        )
    ]

    # Patch embed: Conv3d with kernel_shape == stride == input spatial dims
    # is exactly one dot product per patch -- a Linear layer over the
    # already-flattened patch, once the checkpoint's own [out, in, kt, kh,
    # kw] conv weight is flattened to [out, in*kt*kh*kw] (row-major, which
    # is exactly how PyTorch's own Conv3d indexes it -- confirmed from
    # Qwen3_5VisionPatchEmbed.forward's own `.view()` call, which assumes
    # pixel_values already comes in flattened in that same order).
    conv_w = declare("patch_embed.proj.weight")
    conv_w_flat = b.op(
        "Reshape",
        [conv_w, b.shape_const([hidden_size, patch_dim])],
        "patch_embed.w_flat",
    )
    conv_b = declare("patch_embed.proj.bias")
    x = _linear(b, pixel_values, conv_w_flat, conv_b, "patch_embed")

    interp_idx, interp_w = _vision_pos_embed_interp_indices_weights(
        t, h, w, spatial_merge_size, num_grid_per_side
    )
    idx_c = b.const(interp_idx, prefix="pos_interp_idx")
    weight_c = b.const(interp_w.reshape(num_patches, 4, 1), prefix="pos_interp_weight")
    pos_embed_w = declare("pos_embed.weight")
    idx_flat = b.op("Reshape", [idx_c, b.shape_const([-1])], "pos_interp_idx_flat")
    gathered = b.op("Gather", [pos_embed_w, idx_flat], "pos_embed_gathered", axis=0)
    gathered = b.op(
        "Reshape",
        [gathered, b.shape_const([num_patches, 4, hidden_size])],
        "pos_embed_gathered_r",
    )
    pos_embeds = _reduce_sum(
        b, b.op("Mul", [gathered, weight_c], "pos_embed_weighted"), [1], 0, "pos_embeds"
    )
    x = b.op("Add", [x, pos_embeds], "patch_plus_pos")

    cos_np, sin_np = _vision_rope_cos_sin(
        t, h, w, spatial_merge_size, head_dim, rope_theta
    )
    cos_c = b.const(cos_np.reshape(num_patches, 1, head_dim), prefix="vis_rope_cos")
    sin_c = b.const(sin_np.reshape(num_patches, 1, head_dim), prefix="vis_rope_sin")

    for i in range(depth):
        p = f"blocks.{i}"
        resid = x
        hn = b.op(
            "LayerNormalization",
            [x, declare(f"{p}.norm1.weight"), declare(f"{p}.norm1.bias")],
            f"{p}.norm1",
            axis=-1,
            epsilon=1e-6,
        )
        qkv = _linear(
            b,
            hn,
            declare(f"{p}.attn.qkv.weight"),
            declare(f"{p}.attn.qkv.bias"),
            f"{p}.attn.qkv",
        )
        qkv = b.op(
            "Reshape",
            [qkv, b.shape_const([num_patches, 3, num_heads, head_dim])],
            f"{p}.attn.qkv_r",
        )
        q = _squeeze(
            b, _slice_axis(b, qkv, 1, 0, 1, f"{p}.attn.q"), [1], f"{p}.attn.q.sq"
        )
        k = _squeeze(
            b, _slice_axis(b, qkv, 1, 1, 2, f"{p}.attn.k"), [1], f"{p}.attn.k.sq"
        )
        v = _squeeze(
            b, _slice_axis(b, qkv, 1, 2, 3, f"{p}.attn.v"), [1], f"{p}.attn.v.sq"
        )

        q_rot = _rotate_half(b, q, head_dim, f"{p}.attn.q_rot")
        q = b.op(
            "Add",
            [
                b.op("Mul", [q, cos_c], f"{p}.attn.q_cos"),
                b.op("Mul", [q_rot, sin_c], f"{p}.attn.q_sin"),
            ],
            f"{p}.attn.q_rope",
        )
        k_rot = _rotate_half(b, k, head_dim, f"{p}.attn.k_rot")
        k = b.op(
            "Add",
            [
                b.op("Mul", [k, cos_c], f"{p}.attn.k_cos"),
                b.op("Mul", [k_rot, sin_c], f"{p}.attn.k_sin"),
            ],
            f"{p}.attn.k_rope",
        )

        # (num_patches, heads, head_dim) -> (1, heads, num_patches, head_dim); full
        # bidirectional attention over every patch (a single image is one packed
        # attention segment -- see this module's scope note on multi-image packing).
        q = _unsqueeze(
            b,
            b.op("Transpose", [q], f"{p}.attn.q_t", perm=[1, 0, 2]),
            [0],
            f"{p}.attn.q_b",
        )
        k = _unsqueeze(
            b,
            b.op("Transpose", [k], f"{p}.attn.k_t", perm=[1, 0, 2]),
            [0],
            f"{p}.attn.k_b",
        )
        v = _unsqueeze(
            b,
            b.op("Transpose", [v], f"{p}.attn.v_t", perm=[1, 0, 2]),
            [0],
            f"{p}.attn.v_b",
        )

        inv_sqrt_d = b.const(
            np.array(1.0 / np.sqrt(head_dim), dtype=np.float32), prefix="vis_inv_sqrt_d"
        )
        kt = b.op("Transpose", [k], f"{p}.attn.kt", perm=[0, 1, 3, 2])
        scores = b.op(
            "Mul",
            [b.op("MatMul", [q, kt], f"{p}.attn.scores"), inv_sqrt_d],
            f"{p}.attn.scores_scaled",
        )
        attn = b.op("Softmax", [scores], f"{p}.attn.softmax", axis=-1)
        out = b.op("MatMul", [attn, v], f"{p}.attn.out")
        out = b.op(
            "Transpose",
            [_squeeze(b, out, [0], f"{p}.attn.out_sq")],
            f"{p}.attn.out_t",
            perm=[1, 0, 2],
        )
        out = b.op(
            "Reshape",
            [out, b.shape_const([num_patches, hidden_size])],
            f"{p}.attn.out_flat",
        )
        out = _linear(
            b,
            out,
            declare(f"{p}.attn.proj.weight"),
            declare(f"{p}.attn.proj.bias"),
            f"{p}.attn.proj",
        )
        x = b.op("Add", [resid, out], f"{p}.attn_resid")

        resid = x
        hn = b.op(
            "LayerNormalization",
            [x, declare(f"{p}.norm2.weight"), declare(f"{p}.norm2.bias")],
            f"{p}.norm2",
            axis=-1,
            epsilon=1e-6,
        )
        fc1 = _linear(
            b,
            hn,
            declare(f"{p}.mlp.linear_fc1.weight"),
            declare(f"{p}.mlp.linear_fc1.bias"),
            f"{p}.mlp.fc1",
        )
        act = _gelu_tanh(b, fc1, f"{p}.mlp.act")
        fc2 = _linear(
            b,
            act,
            declare(f"{p}.mlp.linear_fc2.weight"),
            declare(f"{p}.mlp.linear_fc2.bias"),
            f"{p}.mlp.fc2",
        )
        x = b.op("Add", [resid, fc2], f"{p}.mlp_resid")

    # Merger: LayerNorm at the *un-merged* hidden_size (use_postshuffle_norm
    # is False for this checkpoint -- confirmed from real source), then
    # group every spatial_merge_size**2 consecutive (already
    # block-ordered) tokens into one wider row.
    x = b.op(
        "LayerNormalization",
        [x, declare("merger.norm.weight"), declare("merger.norm.bias")],
        "merger.norm",
        axis=-1,
        epsilon=1e-6,
    )
    merge_unit = spatial_merge_size * spatial_merge_size
    merged_hidden = hidden_size * merge_unit
    x = b.op(
        "Reshape",
        [x, b.shape_const([num_patches // merge_unit, merged_hidden])],
        "merger.grouped",
    )
    fc1 = _linear(
        b,
        x,
        declare("merger.linear_fc1.weight"),
        declare("merger.linear_fc1.bias"),
        "merger.fc1",
    )
    act = _gelu_erf(b, fc1, "merger.act")
    image_embeds = _linear(
        b,
        act,
        declare("merger.linear_fc2.weight"),
        declare("merger.linear_fc2.bias"),
        "merger.fc2",
    )

    graph = onnx.helper.make_graph(
        b.nodes,
        "qwen3_5_vision_encoder",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                image_embeds,
                onnx.TensorProto.FLOAT,
                [num_patches // merge_unit, out_hidden_size],
            )
        ],
        initializer=b.initializers,
    )
    return onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )


def reconstruct_qwen3_5_vlm(
    hf_dir: str,
    batch_size: int = 1,
    seq_len: int = 8,
    grid_thw: Tuple[int, int, int] = (1, 4, 4),
) -> Dict[str, onnx.ModelProto]:
    """Builds both graphs (see this module's docstring) from a Qwen3.5-VL
    HuggingFace checkpoint directory, returning ``{"vision_encoder": ...,
    "language_model": ...}``.

    :param hf_dir: checkpoint directory (``config.json`` + safetensors)
    :param batch_size: static batch dimension for the language-model graph
    :param seq_len: static sequence length for the language-model graph
    :param grid_thw: static ``(t, h, w)`` patch grid for the vision-encoder
            graph -- ``t`` must be ``1`` (single image; see this module's
            scope note)
    :raises UnsupportedArchitectureError: not a recognized Qwen3.5-VL
            checkpoint, or a required tensor is missing
    """
    config = read_hf_config(hf_dir)
    _vlm_config, text_config, vision_config = _extract_qwen3_5_configs(config)
    entries = _index_safetensors_checkpoint(hf_dir)
    return {
        "vision_encoder": reconstruct_qwen3_5_vision_encoder(
            vision_config, entries, grid_thw
        ),
        "language_model": reconstruct_qwen3_5_language_model(
            text_config, entries, batch_size, seq_len
        ),
    }
