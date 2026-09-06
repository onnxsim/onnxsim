"""TesseraQ (Dong, Yang, Wan, Wu, Cao, Cheng, Wang, Cheng, Yale University,
2024, "TesseraQ: Ultra Low-Bit LLM Post-Training Quantization with Block
Reconstruction", https://arxiv.org/abs/2410.19103).

:mod:`onnxsim.adaround` already ports this paper's closest relative -- Nagel
et al.'s AdaRound -- which optimizes a per-weight-element soft rounding
variable via the "rectified sigmoid" relaxation, pulled from soft (fully
continuous, any value in ``[0, 1]``) to hard (0 or 1) by a *single*,
smoothly-annealed regularization term applied uniformly to every element at
once: all elements are equally "soft" until late in the schedule, when the
regularizer's exponent sweep (``beta_range``) simultaneously sharpens every
element's own decision together. TesseraQ's own distinguishing
contribution, "Progressive Adaptive Rounding" (PAR), replaces that single
monolithic anneal with a **coarse-to-fine, element-by-element hardening
schedule**: reconstruction runs in a handful of rounds, and at the end of
each round the elements whose current soft value already sits closest to a
hard 0/1 decision (this module's confidence score: ``|h - 0.5]``, the
distance from the relaxation's own undecided midpoint) are permanently
frozen at that decision for the rest of optimization, while the *remaining*
still-soft elements keep being optimized against a reconstruction loss that
now sees the frozen elements as fixed constants rather than as more
optimization variables. Each round therefore reconstructs against a
smaller, better-conditioned free variable set than the last -- confident
elements stop absorbing gradient noise from indecisive neighbours, and by
construction 100% of the weight is hardened by the final round, unlike
AdaRound's `beta`-anneal which only pushes every element toward (never
strictly to) 0/1 without an explicit committal step. The paper reports this
stabilizes reconstruction specifically in the 2-3 bit regime, where
AdaRound's simultaneous-anneal tends to leave a residue of contested,
still-fractional elements that a single hard round-off at the end handles
poorly.

TesseraQ's second contribution ported here: unlike AdaRound (and
:mod:`onnxsim.adaquant`, which jointly optimizes weight rounding with the
*activation's* clip range but still leaves the *weight's* own dequantization
scale exactly as calibration left it), this module also treats each
weight block's own dequantization scale as a reconstruction-loss variable,
optimized in log-space by the same Adam loop that optimizes the rounding --
"jointly", in the sense that both live in the same computation graph and
the same per-iteration gradient step, not alternated. This matters
specifically because this module supports narrowing the *effective* bit
width below what :func:`onnxsim.quantize_weight_only_int4` originally
calibrated its scale for (INT4's own ``[-7, 7]`` symmetric range): asking
PAR to round every element into a narrower ``[-3, 3]`` (3-bit) or
``[-1, 1]`` (2-bit) range while leaving the *scale* fixed at whatever
calibration picked for the *wider* INT4 range would waste most of the
narrower range's resolution outside the weight's actual distribution --
jointly shrinking the scale during reconstruction (initialized as the
original scale, not re-derived from scratch) is what makes low bit widths
usable at all rather than merely "more clipping".

This targets exactly the same graph shape and the same
:func:`onnxsim.quantize_weight_only_int4`-produced starting point
:mod:`onnxsim.adaround` does (``DequantizeLinear(Wq, Ws, axis=<reduction
axis>, block_size=...)`` feeding a MatMul/Gemm, ``Wq`` INT4 storage, one
scale per ``(block, output channel)``) -- see
``weight_only_quantize_int4_matmul.h`` -- and reuses that module's own
candidate matcher, rectified-sigmoid relaxation helpers, and INT4 nibble
packing outright rather than re-deriving them. The *storage* container
stays the ONNX INT4 tensor type regardless of ``num_bits``: a narrower
``num_bits`` just constrains PAR's own optimization range (and the
hardened codes it ultimately writes) to a subset of the INT4 nibble's
``[-8, 7]`` range, the same way :mod:`onnxsim.spqr`/:mod:`onnxsim.billm`
reuse INT4/INT8 containers for effective bit widths below their nominal
size -- no new ONNX tensor type or contrib op is introduced.

Everything here is plain numpy with hand-derived gradients and a small
hand-rolled Adam loop, exactly :mod:`onnxsim.adaround`/
:mod:`onnxsim.adaquant`'s own style -- no autodiff framework, no
calibration data beyond what :mod:`onnxsim.calibration` already provides.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import (
    _GAMMA,
    _ZETA,
    _Candidate,
    _find_int4_matmul_candidates,
    _h_and_dhdv,
    _pack_int4,
)
from onnxsim.bias_correction import _activation_rows, _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _optimize_tesseraq(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    block_size: int,
    x: np.ndarray,
    n_min: float,
    n_max: float,
    num_iterations: int,
    par_rounds: int,
    learning_rate: float,
    scale_learning_rate: float,
    reg_param: float,
    warm_start: float,
    beta_range: Tuple[float, float],
) -> "tuple[np.ndarray, np.ndarray]":
    """Runs TesseraQ's Progressive Adaptive Rounding for one weight matrix,
    jointly optimizing its per-block dequantization scale. ``w_nk`` is the
    float weight and ``scale_blocks`` its per-``(output channel, block)``
    scale (i.e. not yet broadcast across each block's own elements), both
    laid out ``[N, K]``/``[N, K / block_size]`` (output channel first)
    regardless of the op's own storage layout; ``x`` is real calibration
    activations captured from the float model, shape ``[num_samples, K]``.

    Returns ``(codes_nk, scale_blocks_optimized)``: the fully-hardened
    integer codes (shape like ``w_nk``, values in ``[n_min, n_max]``) and
    the optimized per-block scale (shape like ``scale_blocks``).
    """
    n, k = w_nk.shape
    num_blocks = scale_blocks.shape[1]
    y_float = x @ w_nk.T  # [S, N]

    scale_nk0 = np.repeat(scale_blocks, block_size, axis=1)[:, :k]
    ratio = w_nk / scale_nk0
    floor_base = np.floor(ratio)
    frac = np.clip(ratio - floor_base, 1e-4, 1.0 - 1e-4)
    sig0 = np.clip((frac - _GAMMA) / (_ZETA - _GAMMA), 1e-4, 1.0 - 1e-4)
    v = np.log(sig0 / (1.0 - sig0))

    log_delta = np.zeros_like(scale_blocks)  # per-block multiplicative scale correction

    hard_mask = np.zeros((n, k), dtype=bool)
    hard_code = np.zeros((n, k), dtype=np.float64)

    m_v, v2_v = np.zeros_like(v), np.zeros_like(v)
    m_s, v2_s = np.zeros_like(log_delta), np.zeros_like(log_delta)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    par_rounds = max(1, par_rounds)
    iters_per_round = max(1, num_iterations // par_rounds)
    total_iters = iters_per_round * par_rounds
    warm_start_iters = int(total_iters * warm_start)
    beta_start, beta_end = beta_range
    n_elems = x.shape[0] * n

    global_t = 0
    for round_idx in range(par_rounds):
        for _ in range(iters_per_round):
            scale_hat_blocks = scale_blocks * np.exp(log_delta)
            scale_hat_nk = np.repeat(scale_hat_blocks, block_size, axis=1)[:, :k]

            h, dh_dv = _h_and_dhdv(v)
            raw = np.where(hard_mask, hard_code, floor_base + h)
            clipped = np.clip(raw, n_min, n_max)
            active = (raw > n_min) & (raw < n_max) & ~hard_mask
            w_hat = clipped * scale_hat_nk

            y_hat = x @ w_hat.T  # [S, N]
            dl_dy = 2.0 * (y_hat - y_float) / n_elems
            dl_dw_hat = dl_dy.T @ x  # [N, K]

            dl_dh = dl_dw_hat * np.where(active, scale_hat_nk, 0.0)
            grad_v = dl_dh * dh_dv

            if global_t >= warm_start_iters:
                progress = (global_t - warm_start_iters) / max(
                    1, total_iters - warm_start_iters - 1
                )
                beta = beta_start + (beta_end - beta_start) * progress
                u = 2.0 * h - 1.0
                abs_u = np.abs(u)
                dreg_dh = (
                    -2.0 * reg_param * beta * np.sign(u) * np.power(abs_u, beta - 1.0)
                )
                grad_v = grad_v + dreg_dh * dh_dv
            grad_v = np.where(hard_mask, 0.0, grad_v)

            dl_dscale_hat_nk = dl_dw_hat * clipped  # [N, K]
            dl_dscale_hat_blocks = dl_dscale_hat_nk.reshape(
                n, num_blocks, block_size
            ).sum(axis=2)
            grad_log_delta = dl_dscale_hat_blocks * scale_hat_blocks

            global_t += 1
            m_v = beta1 * m_v + (1.0 - beta1) * grad_v
            v2_v = beta2 * v2_v + (1.0 - beta2) * (grad_v * grad_v)
            bias_c1 = 1.0 - beta1**global_t
            bias_c2 = 1.0 - beta2**global_t
            v = v - learning_rate * (m_v / bias_c1) / (
                np.sqrt(v2_v / bias_c2) + adam_eps
            )

            m_s = beta1 * m_s + (1.0 - beta1) * grad_log_delta
            v2_s = beta2 * v2_s + (1.0 - beta2) * (grad_log_delta * grad_log_delta)
            log_delta = log_delta - scale_learning_rate * (m_s / bias_c1) / (
                np.sqrt(v2_s / bias_c2) + adam_eps
            )

        # End of round: permanently harden the most-confident still-soft
        # elements (PAR's own coarse-to-fine schedule) -- confidence is each
        # element's distance from the relaxation's undecided midpoint, i.e.
        # how far its own soft value already sits from 0.5.
        h_now, _ = _h_and_dhdv(v)
        if round_idx == par_rounds - 1:
            newly = ~hard_mask
        else:
            target_fraction = (round_idx + 1) / par_rounds
            target_count = int(round(target_fraction * n * k))
            to_harden = max(0, target_count - int(hard_mask.sum()))
            newly = np.zeros((n, k), dtype=bool)
            if to_harden > 0:
                soft_idx = np.flatnonzero(~hard_mask.ravel())
                if to_harden >= soft_idx.size:
                    newly_flat = soft_idx
                else:
                    confidence = np.abs(h_now.ravel()[soft_idx] - 0.5)
                    top = np.argpartition(confidence, -to_harden)[-to_harden:]
                    newly_flat = soft_idx[top]
                newly.ravel()[newly_flat] = True
        hard_code = np.where(newly, floor_base + np.round(h_now), hard_code)
        hard_mask = hard_mask | newly

    codes_nk = np.clip(hard_code, n_min, n_max)
    scale_blocks_optimized = scale_blocks * np.exp(log_delta)
    return codes_nk, scale_blocks_optimized


def apply_tesseraq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_bits: int = 4,
    num_iterations: int = 400,
    par_rounds: int = 4,
    learning_rate: float = 0.1,
    scale_learning_rate: float = 0.01,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes TesseraQ-style Progressive Adaptive Rounding, jointly with
    each weight block's own dequantization scale, for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``, using
    real activations captured from ``float_model``. See this module's own
    docstring for the technique and its relationship to
    :func:`onnxsim.apply_adaround`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to optimize the
            reconstruction against. Each batch is a ``{input_name:
            np.ndarray}`` dict matching ``float_model``'s graph inputs --
            see :func:`onnxsim.generate_random_calibration_data` (the
            default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative optimization target than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_bits: effective signed bit width PAR rounds each element
            into, ``n_max = 2 ** (num_bits - 1) - 1`` (symmetric, same
            convention as :func:`onnxsim.quantize_weight_only_int4`'s own
            ``[-7, 7]``). Must be between 2 and 4 -- the codes are always
            packed into the underlying INT4 nibble storage
            :func:`onnxsim.quantize_weight_only_int4` already allocated,
            regardless of ``num_bits``; a value below 4 only narrows PAR's
            own optimization/rounding range (and relies on the jointly
            -optimized scale -- see this module's own docstring -- to make
            that narrower range usable)
    :param num_iterations: total Adam steps to run per layer, split evenly
            across ``par_rounds``
    :param par_rounds: number of Progressive Adaptive Rounding rounds. Each
            round (except the last, which hardens every remaining element)
            permanently hardens an additional ``1 / par_rounds`` of the
            *original* element count -- the most-confident still-soft
            elements first -- then continues optimizing only what remains
            soft. ``par_rounds=1`` degenerates to a single reconstruction
            round followed by one hardening step, with no progressive
            coarse-to-fine schedule
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation (same role as :func:`onnxsim.apply_adaround`'s
            ``learning_rate``)
    :param scale_learning_rate: Adam learning rate for each weight block's
            dequantization scale (optimized in log-space, as a
            multiplicative correction on top of ``quantized_model``'s own
            calibrated scale)
    :param reg_param: weight of the regularization term that pulls each
            still-soft element's relaxation toward a hard 0/1 (floor/ceil)
            decision ahead of its round's own hardening step
    :param warm_start: fraction of the total iteration budget (from the
            start, across every round) run with the regularization term
            disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed across the iterations after
            ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            codes and per-block scale initializers rewritten to their
            PAR-optimized values (same shape and dtype -- only the codes
            and the scale's own values change)
    """
    if not 2 <= num_bits <= 4:
        raise ValueError(f"num_bits must be between 2 and 4, got {num_bits}")

    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    candidates: List[_Candidate] = _find_int4_matmul_candidates(
        float_model, quantized_model
    )
    if not candidates:
        return quantized_model

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    n_max = float(2 ** (num_bits - 1) - 1)
    n_min = -n_max

    optimized_codes: Dict[str, np.ndarray] = {}
    optimized_scale: Dict[str, np.ndarray] = {}
    for c in candidates:
        acts = _activation_rows(activations[c.float_node.input[0]])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        dim0, dim1 = w.shape

        if c.weight_transposed:
            w_nk = w  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        if x.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip
        if w_nk.shape[1] % c.block_size != 0:
            continue  # ragged block; quantize_weight_only_int4 never produces this

        codes_nk, scale_blocks_opt = _optimize_tesseraq(
            w_nk,
            scale_blocks,
            c.block_size,
            x,
            n_min=n_min,
            n_max=n_max,
            num_iterations=num_iterations,
            par_rounds=par_rounds,
            learning_rate=learning_rate,
            scale_learning_rate=scale_learning_rate,
            reg_param=reg_param,
            warm_start=warm_start,
            beta_range=beta_range,
        )
        codes_orig = codes_nk if c.weight_transposed else codes_nk.T
        scale_orig = scale_blocks_opt if c.weight_transposed else scale_blocks_opt.T
        assert codes_orig.shape == (dim0, dim1)
        optimized_codes[c.wq_name] = codes_orig.astype(np.int8)
        optimized_scale[c.ws_init.name] = scale_orig.astype(np.float32)

    if not optimized_codes:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    for t in corrected.graph.initializer:
        codes = optimized_codes.get(t.name)
        if codes is not None:
            t.raw_data = _pack_int4(codes)
            continue
        scale = optimized_scale.get(t.name)
        if scale is not None:
            t.CopyFrom(onnx.numpy_helper.from_array(scale, name=t.name))

    return corrected
