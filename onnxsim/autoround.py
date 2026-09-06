"""Intel's AutoRound -- Cheng et al., 2023, "Optimize Weight Rounding via
Signed Gradient Descent for the Quantization of LLMs"
(https://arxiv.org/abs/2309.05516). Closes the one specific gap between
:mod:`onnxsim.adaround` (AIMET's AdaRound) and AutoRound proper that this
codebase actually needs: AdaRound optimizes only each weight element's
rounding decision (floor vs. ceil) at a *fixed* scale
(``onnxsim.apply_adaround`` is guaranteed to leave scale unchanged -- see
its own docstring and ``tests/test_adaround.py``'s
``test_adaround_preserves_scale_and_shape``). AutoRound additionally lets
the per-(output-channel, block) clipping range -- and therefore the
scale -- move during the same optimization, jointly with rounding: a
single outlier in a block can otherwise force a scale so large that every
other element in that block quantizes to near-zero information, something
no amount of per-element rounding choice can fix.

This is not a port of AutoRound as a whole framework (its own per-block,
per-transformer-layer calibration pipeline, multiple bit widths, etc. --
see "AutoRound" in ``docs/dynamic-quantization.md``'s list of large,
independent projects onnxsim does not try to reimplement). It targets the
exact same ``quantize_weight_only_int4``-produced MatMul/Gemm layers as
:mod:`onnxsim.adaround`, reuses that module's candidate search and
rounding relaxation verbatim, and adds one thing on top: a second,
per-(output-channel, block) trainable clip-ratio parameter that rescales
each block's scale within a bounded range, optimized by the same
closed-form-gradient Adam loop AdaRound already uses.

The rounding gradient is unchanged from AdaRound. The new clip-ratio
gradient follows LSQ's (Esser et al., 2020, "Learned Step Size
Quantization") derivation for a round-clip quantizer's scale gradient: in
the non-saturated region, ``d(w_hat)/d(scale) = code - w/scale`` (the gap
between the rounded code and the un-rounded ratio); in the saturated
region it is just the saturating code itself. Both this and AdaRound's own
rounding gradient use a straight-through estimator for ``floor`` (its true
derivative is zero almost everywhere) -- standard QAT practice, and why
this module is validated by reconstruction-error improvement (see
``tests/test_autoround.py``) rather than a finite-difference gradient
check, which would not agree with a straight-through gradient by
construction.

Jointly optimizing two coupled parameter sets is also a harder, non-convex
problem than AdaRound's fixed-scale search over rounding alone -- both `v`
and the clip-ratio parameter `c` influence `floor_base` every iteration, so
with a matched iteration budget this can occasionally converge to a worse
local optimum than AdaRound's own decoupled search would reach. Rather
than risk that regression, :func:`_optimize_rounding_and_clip` always runs
AdaRound's own fixed-scale search as well and keeps whichever of the two
has the lower measured reconstruction error -- so :func:`apply_autoround`
is guaranteed to never do worse than :func:`onnxsim.apply_adaround` would
on the same layer and calibration data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import (
    _GAMMA,
    _ZETA,
    _find_int4_matmul_candidates,
    _h_and_dhdv,
    _optimize_rounding,
    _pack_int4,
)
from onnxsim.bias_correction import _activation_rows, _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _clip_ratio_and_dratio_dc(
    c: np.ndarray, cmin: float, cmax: float
) -> "tuple[np.ndarray, np.ndarray]":
    """``clip_ratio(c) = sigmoid(c) * (cmax - cmin) + cmin`` -- a bounded,
    smooth reparameterization of "how much to shrink/grow this block's
    scale", the same rectified-range trick as AdaRound's own ``h(v)``
    (:func:`onnxsim.adaround._h_and_dhdv`) minus the clip (sigmoid alone
    already stays strictly inside ``(cmin, cmax)``, so no hard clip -- and
    no dead zero-gradient region -- is needed here). ``c = 0`` starts at
    ``clip_ratio == 1.0`` (unchanged scale) whenever ``cmin + cmax == 2.0``,
    true of this module's default range.
    """
    s = 1.0 / (1.0 + np.exp(-c))
    ratio = s * (cmax - cmin) + cmin
    dratio_dc = s * (1.0 - s) * (cmax - cmin)
    return ratio, dratio_dc


def _optimize_rounding_and_clip(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    block_size: int,
    x: np.ndarray,
    n_min: float,
    n_max: float,
    num_iterations: int,
    learning_rate: float,
    clip_learning_rate: float,
    reg_param: float,
    warm_start: float,
    beta_range: "tuple[float, float]",
    clip_ratio_range: "tuple[float, float]",
) -> "tuple[np.ndarray, np.ndarray]":
    """Like :func:`onnxsim.adaround._optimize_rounding`, but jointly
    optimizes a per-(N, block) clip-ratio parameter alongside the
    per-element rounding relaxation. ``w_nk``/``scale_blocks`` are laid out
    ``[N, K]``/``[N, K / block_size]`` (output channel first), matching
    :func:`onnxsim.apply_autoround`'s own normalization. Returns
    ``(codes_nk, scale_blocks_optimized)``.

    Jointly optimizing two coupled parameter sets (v and c both influence
    ``floor_base`` every iteration, unlike AdaRound's fixed-scale search
    over v alone) is a harder, non-convex problem than AdaRound's own --
    with a matched iteration budget it can occasionally converge to a
    reconstruction error *worse* than AdaRound's decoupled optimum would
    reach. Rather than accept that risk, this always finishes by running
    AdaRound's own fixed-scale optimization too (``clip_ratio == 1``
    throughout, i.e. exactly :func:`onnxsim.adaround._optimize_rounding`'s
    own search) and returns whichever of the two actually has the lower
    measured reconstruction error on ``x`` -- so :func:`apply_autoround`
    is guaranteed never to do worse than :func:`onnxsim.apply_adaround`
    would on the same layer and calibration data.
    """
    n_out, k = w_nk.shape
    num_blocks = scale_blocks.shape[1]
    y_float = x @ w_nk.T  # [num_samples, N]

    scale_nk0 = np.repeat(scale_blocks, block_size, axis=1)[:, :k]
    ratio0 = w_nk / scale_nk0
    floor0 = np.floor(ratio0)
    frac = np.clip(ratio0 - floor0, 1e-4, 1.0 - 1e-4)
    sig0 = (frac - _GAMMA) / (_ZETA - _GAMMA)
    sig0 = np.clip(sig0, 1e-4, 1.0 - 1e-4)
    v = np.log(sig0 / (1.0 - sig0))
    c = np.zeros_like(scale_blocks)

    m_v, v2_v = np.zeros_like(v), np.zeros_like(v)
    m_c, v2_c = np.zeros_like(c), np.zeros_like(c)
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    warm_start_iters = int(num_iterations * warm_start)
    beta_start, beta_end = beta_range
    cmin, cmax = clip_ratio_range
    n = x.shape[0] * n_out

    for t in range(num_iterations):
        clip_ratio, dratio_dc = _clip_ratio_and_dratio_dc(c, cmin, cmax)
        scale_eff_blocks = scale_blocks * clip_ratio
        scale_eff = np.repeat(scale_eff_blocks, block_size, axis=1)[:, :k]

        h, dh_dv = _h_and_dhdv(v)
        ratio_wk = w_nk / scale_eff
        floor_base = np.floor(ratio_wk)
        raw = floor_base + h
        code = np.clip(raw, n_min, n_max)
        active = (raw > n_min) & (raw < n_max)
        w_hat = code * scale_eff

        y_hat = x @ w_hat.T
        dl_dy = 2.0 * (y_hat - y_float) / n
        dl_dw_hat = dl_dy.T @ x  # [N, K]

        # Rounding gradient: identical derivation to AdaRound's own,
        # scale_eff standing in for the (there, fixed) scale.
        dl_dh = dl_dw_hat * np.where(active, scale_eff, 0.0)
        grad_v = dl_dh * dh_dv
        if t >= warm_start_iters:
            progress = (t - warm_start_iters) / max(
                1, num_iterations - warm_start_iters - 1
            )
            beta = beta_start + (beta_end - beta_start) * progress
            u = 2.0 * h - 1.0
            abs_u = np.abs(u)
            dreg_dh = -2.0 * reg_param * beta * np.sign(u) * np.power(abs_u, beta - 1.0)
            grad_v = grad_v + dreg_dh * dh_dv

        # Clip-ratio gradient: LSQ-style d(w_hat)/d(scale), block-summed
        # (one clip-ratio parameter is shared by every element in a block)
        # then chained through scale_eff_blocks = scale_blocks * clip_ratio(c).
        dw_hat_ds_eff = np.where(active, code - ratio_wk, code)
        dl_ds_eff = dl_dw_hat * dw_hat_ds_eff
        dl_ds_eff_blocks = dl_ds_eff.reshape(n_out, num_blocks, block_size).sum(axis=2)
        grad_c = dl_ds_eff_blocks * scale_blocks * dratio_dc

        m_v = adam_beta1 * m_v + (1.0 - adam_beta1) * grad_v
        v2_v = adam_beta2 * v2_v + (1.0 - adam_beta2) * (grad_v * grad_v)
        m_hat_v = m_v / (1.0 - adam_beta1 ** (t + 1))
        v_hat_v = v2_v / (1.0 - adam_beta2 ** (t + 1))
        v = v - learning_rate * m_hat_v / (np.sqrt(v_hat_v) + adam_eps)

        m_c = adam_beta1 * m_c + (1.0 - adam_beta1) * grad_c
        v2_c = adam_beta2 * v2_c + (1.0 - adam_beta2) * (grad_c * grad_c)
        m_hat_c = m_c / (1.0 - adam_beta1 ** (t + 1))
        v_hat_c = v2_c / (1.0 - adam_beta2 ** (t + 1))
        c = c - clip_learning_rate * m_hat_c / (np.sqrt(v_hat_c) + adam_eps)

    clip_ratio_final, _ = _clip_ratio_and_dratio_dc(c, cmin, cmax)
    scale_blocks_joint = scale_blocks * clip_ratio_final
    scale_eff_joint = np.repeat(scale_blocks_joint, block_size, axis=1)[:, :k]
    h_final, _ = _h_and_dhdv(v)
    floor_final = np.floor(w_nk / scale_eff_joint)
    codes_joint = np.clip(floor_final + np.round(h_final), n_min, n_max)

    # Safety net (see docstring): never return worse than AdaRound's own
    # fixed-scale optimum would.
    codes_ada_only = _optimize_rounding(
        w_nk,
        scale_nk0,
        x,
        n_min,
        n_max,
        num_iterations,
        learning_rate,
        reg_param,
        warm_start,
        beta_range,
    )
    loss_joint = np.mean((x @ (codes_joint * scale_eff_joint).T - y_float) ** 2)
    loss_ada_only = np.mean((x @ (codes_ada_only * scale_nk0).T - y_float) ** 2)
    if loss_ada_only <= loss_joint:
        return codes_ada_only, scale_blocks
    return codes_joint, scale_blocks_joint


def apply_autoround(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    clip_learning_rate: float = 0.03,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: "tuple[float, float]" = (20.0, 2.0),
    clip_ratio_range: "tuple[float, float]" = (0.5, 1.5),
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """AutoRound: jointly optimizes both adaptive rounding and the
    per-block clipping range for every ``quantize_weight_only_int4``-
    quantized MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``, using real activations
    captured from ``float_model``. See this module's own docstring for the
    technique and how it differs from :func:`onnxsim.apply_adaround`.

    Unlike :func:`onnxsim.apply_adaround`, which never changes a layer's
    scale, this may rewrite both the rounding codes *and* the scale
    initializer -- a block whose fixed scale was dominated by an outlier
    can end up with a smaller scale (and that outlier more clipped) if
    doing so reduces the block's overall reconstruction error. A layer
    whose joint optimization does not actually beat AdaRound's own
    fixed-scale search keeps its original scale (see this module's own
    docstring for the guarantee behind that).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
    :param calibration_data: representative input batches to optimize
            against -- see :func:`onnxsim.apply_adaround`'s own parameter
            of the same name.
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per layer
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation
    :param clip_learning_rate: Adam learning rate for the per-block
            clip-ratio parameter -- kept separate from, and by default
            smaller than, ``learning_rate`` since one clip-ratio value is
            shared by an entire block's worth of rounding decisions
    :param reg_param: weight of the regularization term that pulls each
            rounding element's relaxation toward a hard 0/1 (floor/ceil)
            decision -- see :func:`onnxsim.apply_adaround`
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the rounding regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the rounding
            regularization term's exponent, linearly annealed across the
            iterations after ``warm_start``
    :param clip_ratio_range: ``(min, max)`` bounds the optimized scale can
            move to, expressed as a multiple of the original (RTN) scale.
            Must satisfy ``min + max == 2.0`` for the optimization to start
            at the unmodified scale.
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its AutoRound-optimized codes, and its
            scale initializer rewritten to the optimized per-block scale
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    if not candidates:
        return quantized_model

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    optimized_codes: Dict[str, np.ndarray] = {}
    optimized_scale: Dict[str, np.ndarray] = {}
    for cand in candidates:
        acts = _activation_rows(activations[cand.float_node.input[0]])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(cand.w_float_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(cand.ws_init).astype(np.float64)
        dim0, dim1 = w.shape

        if cand.weight_transposed:
            w_nk = w  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        if x.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip

        codes_nk, scale_blocks_new = _optimize_rounding_and_clip(
            w_nk,
            scale_blocks,
            cand.block_size,
            x,
            n_min=-7.0,
            n_max=7.0,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            clip_learning_rate=clip_learning_rate,
            reg_param=reg_param,
            warm_start=warm_start,
            beta_range=beta_range,
            clip_ratio_range=clip_ratio_range,
        )
        codes_orig = codes_nk if cand.weight_transposed else codes_nk.T
        scale_orig = scale_blocks_new if cand.weight_transposed else scale_blocks_new.T
        assert codes_orig.shape == (dim0, dim1)
        optimized_codes[cand.wq_name] = codes_orig.astype(np.int8)
        optimized_scale[cand.ws_init.name] = scale_orig.astype(np.float32)

    if not optimized_codes:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    for t in corrected.graph.initializer:
        codes = optimized_codes.get(t.name)
        if codes is not None:
            t.raw_data = _pack_int4(codes)
            continue
        new_scale = optimized_scale.get(t.name)
        if new_scale is not None:
            t.CopyFrom(onnx.numpy_helper.from_array(new_scale, name=t.name))

    return corrected
