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

Both halves of that -- the joint search and the AdaRound comparison run --
also exist as an ONNX *step graph* (:mod:`onnxsim.qat_graph`), reached by
``apply_autoround(..., step_providers=[...])``, so the optimization can run
on a GPU, an NPU execution provider or WebGPU instead of in host numpy. The
one thing that makes AutoRound a harder port than AdaRound's or AdaQuant's
is the same thing that makes it AutoRound: because the scale is being
optimized, each element's quantization bin ``floor(w / scale_eff)`` cannot
be precomputed once in float64 and fed in, so it is recomputed inside the
graph every step, in float32, with a composed ``floor`` (:func:`_floor`).
See :func:`_joint_loop_on_graph` for how closely the two paths then track
each other, and ``docs/qat.md`` for the wider picture.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend, qat_graph
from onnxsim.adaround import (
    _GAMMA,
    _ZETA,
    _find_int4_matmul_candidates,
    _h_and_dhdv,
    _init_relaxation,
    _optimize_rounding,
    _optimize_rounding_on_graph,
    _pack_int4,
    _rounding_codes,
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


def _init_autoround(
    w_nk: np.ndarray, scale_blocks: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """This layer's joint optimization problem at its starting point:
    ``(scale_nk0, v, c)`` -- the original (RTN, ``clip_ratio == 1``)
    per-element scale, the rounding relaxation, and the clip-ratio
    parameter.

    Factored out for the same reason
    :func:`onnxsim.adaround._init_relaxation` and
    :func:`onnxsim.adaquant._init_adaquant` are: the numpy loop and the
    step-graph loop (:func:`_optimize_rounding_and_clip_on_graph`) must
    optimize *the same problem from the same warm start*, and the only way
    to keep that true as either one changes is for there to be a single
    copy of it. The rounding half is literally AdaRound's own function
    rather than a second copy of the same six lines.

    ``_init_relaxation``'s second return value -- each element's
    quantization bin ``floor(w / scale)`` -- is deliberately dropped here,
    and it is the one thing that genuinely cannot be shared with AdaRound.
    There, the scale is fixed, so the bin is a loop constant computed once;
    here the scale moves with ``c`` every iteration, so the bin has to be
    recomputed inside the loop (and, in the step graph, inside the graph)
    from the *current* effective scale.

    ``c = 0`` is the unmodified scale: ``sigmoid(0) == 0.5`` puts
    ``clip_ratio`` at ``(cmin + cmax) / 2``, which is 1.0 for any range
    satisfying this module's own ``cmin + cmax == 2`` precondition.
    """
    scale_nk0 = np.repeat(scale_blocks, block_size, axis=1)[:, : w_nk.shape[1]]
    v, _ = _init_relaxation(w_nk, scale_nk0)
    return scale_nk0, v, np.zeros_like(scale_blocks)


def _autoround_results(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    block_size: int,
    v: np.ndarray,
    c: np.ndarray,
    n_min: float,
    n_max: float,
    clip_ratio_range: "tuple[float, float]",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The deployable values the two optimized parameter groups have settled
    on: ``(codes, scale_blocks_optimized, scale_eff)`` -- the last lines of
    both optimization paths, shared so they cannot drift.

    Both collapses happen here. The clip ratio becomes a concrete per-block
    scale (the thing the model actually stores, and the reason
    :func:`apply_autoround` may rewrite a scale initializer where
    :func:`onnxsim.apply_adaround` never does), and each element's rounding
    relaxation collapses to its nearest hard floor/ceil choice against the
    bin that *this* scale implies -- not the bin it started in, which is
    exactly what moving the scale is allowed to change.
    """
    cmin, cmax = clip_ratio_range
    clip_ratio, _ = _clip_ratio_and_dratio_dc(c, cmin, cmax)
    scale_blocks_opt = scale_blocks * clip_ratio
    scale_eff = np.repeat(scale_blocks_opt, block_size, axis=1)[:, : w_nk.shape[1]]
    floor_final = np.floor(w_nk / scale_eff)
    codes = _rounding_codes(v, floor_final, n_min, n_max)
    return codes, scale_blocks_opt, scale_eff


def _keep_better_of(
    x: np.ndarray,
    y_float: np.ndarray,
    scale_blocks: np.ndarray,
    scale_nk0: np.ndarray,
    joint: "tuple[np.ndarray, np.ndarray, np.ndarray]",
    codes_ada_only: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """AutoRound's safety net: the joint result, or AdaRound's fixed-scale
    one, whichever actually has the lower reconstruction error on ``x``.

    See this module's docstring for why it exists at all (the joint problem
    is non-convex and can, at a matched iteration budget, land on a worse
    local optimum than AdaRound's decoupled search reaches). It lives in one
    place because the guarantee it provides -- never worse than
    :func:`onnxsim.apply_adaround` on the same layer and data -- has to hold
    on the step-graph path exactly as it does on the numpy one.

    The comparison itself stays in host float64 on both paths even when the
    optimization ran on an execution provider: it is two matmuls once per
    layer, not once per step, so there is nothing to gain by moving it, and
    a *decision* between two candidates is the last place to want float32's
    tie-breaking.
    """
    codes_joint, scale_blocks_joint, scale_eff_joint = joint
    loss_joint = np.mean((x @ (codes_joint * scale_eff_joint).T - y_float) ** 2)
    loss_ada_only = np.mean((x @ (codes_ada_only * scale_nk0).T - y_float) ** 2)
    if loss_ada_only <= loss_joint:
        return codes_ada_only, scale_blocks
    return codes_joint, scale_blocks_joint


def _joint_loop(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    block_size: int,
    x: np.ndarray,
    y_float: np.ndarray,
    v: np.ndarray,
    c: np.ndarray,
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
    """``num_iterations`` Adam steps on the rounding relaxation ``v`` and the
    clip parameter ``c`` jointly, in host numpy (float64). Returns the two
    optimized parameters, not the codes they collapse to.

    Split out of :func:`_optimize_rounding_and_clip` so that it and
    :func:`_joint_loop_on_graph` are two implementations of exactly one
    thing -- the same warm start in, the same parameters out, with the
    collapse (:func:`_autoround_results`) and the AdaRound safety net
    (:func:`_keep_better_of`) around them shared rather than duplicated.
    That is also what makes the two comparable in a test at all: the safety
    net can pick different branches on the two paths, so an assertion about
    *the optimizers* has to be able to look at ``(v, c)`` before that choice
    is made.
    """
    n_out, k = w_nk.shape
    num_blocks = scale_blocks.shape[1]

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

    return v, c


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
    y_float = x @ w_nk.T  # [num_samples, N]
    scale_nk0, v, c = _init_autoround(w_nk, scale_blocks, block_size)
    v, c = _joint_loop(
        w_nk,
        scale_blocks,
        block_size,
        x,
        y_float,
        v,
        c,
        n_min,
        n_max,
        num_iterations,
        learning_rate,
        clip_learning_rate,
        reg_param,
        warm_start,
        beta_range,
        clip_ratio_range,
    )
    joint = _autoround_results(
        w_nk, scale_blocks, block_size, v, c, n_min, n_max, clip_ratio_range
    )

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
    return _keep_better_of(x, y_float, scale_blocks, scale_nk0, joint, codes_ada_only)


def _int64_const(
    b: qat_graph.GraphBuilder,
    value: Union[Sequence[int], np.ndarray],
    hint: str = "i",
) -> str:
    """An int64 initializer holding ``value``.

    :meth:`onnxsim.qat_graph.GraphBuilder.const` is float32-only, which is
    right for a builder whose whole job is arithmetic on float tensors; the
    three int64 tensors this step graph needs are not arithmetic at all but
    *structure* -- a ``Gather`` index, a ``Reshape`` shape, and a
    ``ReduceSum`` axis list, all of which ONNX requires as int64 inputs
    rather than attributes at opset 17. They are constant for the life of
    the graph, so they are initializers like any other.
    """
    name = b.name(hint)
    b.initializer.append(
        onnx.numpy_helper.from_array(np.asarray(value, dtype=np.int64), name)
    )
    return name


def _floor(b: qat_graph.GraphBuilder, a: str) -> str:
    """``floor(a)``, composed rather than emitted as ``Floor``.

    ``Floor`` is not in :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS` and this is
    not the place to argue for adding it: that set is deliberately small and
    an addition has to be justified by coverage on the WebGPU and WebNN
    backends, not by one caller's convenience. Composing it costs five
    nodes.

    A float-to-int32 ``Cast`` truncates toward zero (the same ONNX-specified
    behaviour :meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest` is
    built on), which is ``floor`` for non-negative values and ``floor + 1``
    for negative non-integers. So subtracting the 0/1 mask ``trunc(a) > a``
    -- true exactly in that second case -- turns truncation into flooring,
    with no special case for negative integers, where truncation is already
    exact and the mask is 0.

    AdaRound never needs this: its scale is fixed, so each element's bin
    ``floor(w / scale)`` is a loop constant computed once in host numpy and
    fed in. AutoRound's scale moves every step, so the bin moves with it and
    the flooring has to happen inside the graph.
    """
    truncated = b.op(
        "Cast",
        [b.op("Cast", [a], to=onnx.TensorProto.INT32)],
        to=onnx.TensorProto.FLOAT,
    )
    overshoot = b.op(
        "Cast", [b.op("Greater", [truncated, a])], to=onnx.TensorProto.FLOAT
    )
    return b.sub(truncated, overshoot)


def _build_autoround_step_graph(
    num_rows: int,
    n: int,
    k: int,
    num_blocks: int,
    block_size: int,
    n_min: float,
    n_max: float,
    clip_ratio_range: "tuple[float, float]",
) -> qat_graph.StepGraph:
    """One Adam step of :func:`_optimize_rounding_and_clip`, as an ONNX
    graph.

    Node for node the same computation the numpy loop performs -- the same
    rectified-sigmoid rounding relaxation, the same bounded clip-ratio
    reparameterization, the same straight-through masks, the same LSQ-style
    scale gradient, the same annealed regularizer, the same Adam -- expressed
    so it can run on an execution provider instead of on the host. See
    :mod:`onnxsim.qat_graph` for why a hand-derived gradient can be written as
    an inference graph at all, and ``docs/qat.md`` for what it is for.

    Two structural differences from
    :func:`onnxsim.adaround._build_rounding_step_graph`, both consequences of
    the one thing AutoRound optimizes that AdaRound does not:

    - **The quantization bin is computed in the graph, not fed in.** AdaRound
      passes ``floor_base = floor(w / scale)`` in as a constant because its
      scale never moves. Here the scale is a function of the parameter being
      optimized, so the float weight ``w`` is the constant and the division
      and flooring happen every step (see :func:`_floor`, since ``Floor`` is
      not in the pinned operator set).
    - **Two parameter groups, so two** :func:`onnxsim.qat_graph.adam_update`
      **calls and six state tensors** -- the per-element relaxation ``v`` and
      the per-(output channel, block) clip parameter ``c``, each with its own
      pair of Adam moments, exactly as :mod:`onnxsim.adaquant` does for its
      three groups. They take separate learning rates (``lr`` and
      ``clip_lr``, matching the numpy loop's two) and share one pair of bias
      corrections, since both take their first step on the same iteration.

    The per-block scale reaches per-element shape by a ``Gather`` along the
    block axis rather than by a repeat-and-reshape: one node, and the index
    vector states the block each column belongs to directly. The gradient
    goes back the other way -- ``Reshape`` to ``[n, num_blocks, block_size]``
    and ``ReduceSum`` over the last axis -- which is the numpy loop's own
    ``reshape(...).sum(axis=2)``, and which (like that line) requires
    ``k == num_blocks * block_size``, true of every
    :func:`onnxsim.quantize_weight_only_int4` weight.

    Shapes are baked in at build time: the accelerator backends this exists
    for -- WebNN, and the NPU execution providers -- compile a graph once and
    want static shapes, and a step graph is rebuilt per layer anyway.
    """
    b = qat_graph.GraphBuilder()
    cmin, cmax = clip_ratio_range
    span = cmax - cmin

    x, y_float, w, scale_blocks = "x", "y_float", "w", "scale_blocks"
    v, m_v, vv_v = "v", "m_v", "vv_v"
    c, m_c, vv_c = "c", "m_c", "vv_c"

    # The bounded clip-ratio reparameterization and its derivative --
    # _clip_ratio_and_dratio_dc's own two lines. No clip, and so no dead
    # zero-gradient region: sigmoid already stays strictly inside the range.
    s_c = b.sigmoid(c)
    clip_ratio = b.add(b.mul(s_c, b.const(span)), b.const(cmin))
    dratio_dc = b.mul(b.mul(s_c, b.sub(b.const(1.0), s_c)), b.const(span))

    # This step's effective scale, per block and then per element. The
    # index vector is `column // block_size`, i.e. which block each of the k
    # columns draws its scale from.
    scale_eff_blocks = b.mul(scale_blocks, clip_ratio)  # [n, num_blocks]
    block_of_column = _int64_const(b, np.arange(k) // block_size, "block")
    scale_eff = b.op("Gather", [scale_eff_blocks, block_of_column], "scale_eff", axis=1)

    # h(v), the rectified sigmoid, and its derivative -- _h_and_dhdv's own
    # two lines, with the "is this element still inside the clip" test as a
    # float 0/1 mask rather than a Where.
    s = b.sigmoid(v)
    raw_h = b.add(b.mul(s, b.const(_ZETA - _GAMMA)), b.const(_GAMMA))
    h = b.clip(raw_h, 0.0, 1.0)
    active_h = b.mul(b.greater_mask(raw_h, 0.0), b.less_mask(raw_h, 1.0))
    ds = b.mul(s, b.sub(b.const(1.0), s))
    dh_dv = b.mul(active_h, b.mul(ds, b.const(_ZETA - _GAMMA)))

    # The soft weight the relaxation and the current clip ratio jointly
    # imply, and the layer's reconstruction error against the float model's
    # own output.
    ratio_wk = b.div(w, scale_eff)
    raw = b.add(_floor(b, ratio_wk), h)
    code = b.clip(raw, n_min, n_max)
    active = b.mul(b.greater_mask(raw, n_min), b.less_mask(raw, n_max))
    w_hat = b.mul(code, scale_eff)

    y_hat = b.matmul(x, b.transpose(w_hat))  # [num_rows, n]
    diff = b.sub(y_hat, y_float)
    dl_dy = b.mul(diff, b.const(2.0 / (num_rows * n)))
    dl_dw_hat = b.matmul(b.transpose(dl_dy), x)  # [n, k]

    # Rounding gradient: identical derivation to AdaRound's own, scale_eff
    # standing in for the (there, fixed) scale.
    dl_dh = b.mul(dl_dw_hat, b.mul(active, scale_eff))
    grad_v = b.mul(dl_dh, dh_dv)

    # The rounding regularizer, pulling each relaxation toward a hard 0/1.
    # `reg_scale` is the caller's reg_param, or 0 during the warm start -- a
    # scalar fed per step, so the warm start needs no second graph.
    u = b.sub(b.mul(b.const(2.0), h), b.const(1.0))
    pow_u = b.op("Pow", [b.op("Abs", [u]), b.sub("beta", b.const(1.0))])
    dreg_dh = b.mul(
        b.mul(b.mul(b.const(-2.0), "reg_scale"), "beta"),
        b.mul(b.op("Sign", [u]), pow_u),
    )
    grad_v = b.add(grad_v, b.mul(dreg_dh, dh_dv))

    # Clip-ratio gradient: LSQ's d(w_hat)/d(scale), which is `code - w/scale`
    # where the code is not saturating and the code itself where it is. The
    # numpy loop writes that as a Where; `code - active * ratio` is the same
    # function without one, since `active` is already the 0/1 mask.
    dw_hat_ds_eff = b.sub(code, b.mul(active, ratio_wk))
    dl_ds_eff = b.mul(dl_dw_hat, dw_hat_ds_eff)  # [n, k]
    # One clip-ratio parameter is shared by a whole block, so its gradient is
    # the block's sum, then chained through scale_eff = scale * clip_ratio(c).
    blocked = b.op(
        "Reshape", [dl_ds_eff, _int64_const(b, [n, num_blocks, block_size], "shape")]
    )
    dl_ds_eff_blocks = b.op(
        "ReduceSum", [blocked, _int64_const(b, [2], "axis")], keepdims=0
    )
    grad_c = b.mul(b.mul(dl_ds_eff_blocks, scale_blocks), dratio_dc)

    v_next, m_v_next, vv_v_next = qat_graph.adam_update(
        b, v, grad_v, m_v, vv_v, "lr", "m_correction", "v_correction"
    )
    c_next, m_c_next, vv_c_next = qat_graph.adam_update(
        b, c, grad_c, m_c, vv_c, "clip_lr", "m_correction", "v_correction"
    )

    return qat_graph.make_step_graph(
        b,
        constants={
            x: [num_rows, k],
            y_float: [num_rows, n],
            w: [n, k],
            scale_blocks: [n, num_blocks],
        },
        state={
            v: ([n, k], v_next),
            m_v: ([n, k], m_v_next),
            vv_v: ([n, k], vv_v_next),
            c: ([n, num_blocks], c_next),
            m_c: ([n, num_blocks], m_c_next),
            vv_c: ([n, num_blocks], vv_c_next),
        },
        scalars=["lr", "clip_lr", "reg_scale", "beta", "m_correction", "v_correction"],
        loss=b.mean_square(diff),
        name="onnxsim_autoround_step",
    )


def _joint_loop_on_graph(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    block_size: int,
    x: np.ndarray,
    y_float: np.ndarray,
    v0: np.ndarray,
    c0: np.ndarray,
    n_min: float,
    n_max: float,
    num_iterations: int,
    learning_rate: float,
    clip_learning_rate: float,
    reg_param: float,
    warm_start: float,
    beta_range: "tuple[float, float]",
    clip_ratio_range: "tuple[float, float]",
    providers: Optional[Sequence[str]],
) -> "tuple[np.ndarray, np.ndarray]":
    """:func:`_joint_loop`, run through :mod:`onnxsim.qat_graph` instead of
    in host numpy, on ``providers``. Same signature, same warm start in, the
    same two optimized parameters out.

    Same optimization up to the float32 the step graph computes in -- the
    numpy loop uses float64, which is not something a GPU/NPU execution
    provider offers. AutoRound has one divergence source AdaRound's port does
    not, and it is worth being precise about: because the scale moves, each
    element's quantization bin ``floor(w / scale_eff)`` is recomputed every
    step, so an element whose ratio sits within a float32 ulp of an integer
    can take a bin one lower in one path than the other -- a *discontinuous*
    change of that element's contribution to both gradients, which then
    steers the rest of the run. That is on top of the boundary case AdaRound
    already has (a relaxation landing near ``h = 0.5``).

    The effect is real and measured rather than hypothetical: running this
    module's own numpy loop in float32 instead of float64 moves the final
    clip ratios by as much as ~9% and flips ~1-4% of the codes, i.e. about
    as much as swapping the numpy loop for this graph does (see
    ``tests/test_autoround_step_graph.py``, which measures both). The two
    paths therefore agree on the *objective* -- the reconstruction error they
    reach -- far more tightly than they agree parameter by parameter, which
    is the honest way to state the guarantee.
    """
    step = _build_autoround_step_graph(
        x.shape[0],
        w_nk.shape[0],
        w_nk.shape[1],
        scale_blocks.shape[1],
        block_size,
        n_min,
        n_max,
        clip_ratio_range,
    )

    warm_start_iters = int(num_iterations * warm_start)
    beta_start, beta_end = beta_range

    def scalars(t: int) -> Dict[str, float]:
        values = {"lr": learning_rate, "clip_lr": clip_learning_rate}
        if t >= warm_start_iters:
            progress = (t - warm_start_iters) / max(
                1, num_iterations - warm_start_iters - 1
            )
            values["reg_scale"] = reg_param
            values["beta"] = beta_start + (beta_end - beta_start) * progress
        else:
            # The regularizer is switched off by its own weight rather than by
            # a second graph. `beta` still needs a value Pow can evaluate --
            # 1.0 makes the (zero-weighted) term |u|^0, finite everywhere.
            values["reg_scale"] = 0.0
            values["beta"] = 1.0
        values.update(qat_graph.adam_bias_corrections(t))
        return values

    final = qat_graph.run_step_graph(
        step,
        constants={
            "x": x,
            "y_float": y_float,
            "w": w_nk,
            "scale_blocks": scale_blocks,
        },
        state={
            "v": v0,
            "m_v": np.zeros_like(v0),
            "vv_v": np.zeros_like(v0),
            "c": c0,
            "m_c": np.zeros_like(c0),
            "vv_c": np.zeros_like(c0),
        },
        num_steps=num_iterations,
        scalars=scalars,
        providers=providers,
    )
    return final["v"].astype(np.float64), final["c"].astype(np.float64)


def _optimize_rounding_and_clip_on_graph(
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
    providers: Optional[Sequence[str]],
) -> "tuple[np.ndarray, np.ndarray]":
    """:func:`_optimize_rounding_and_clip`, run on ``providers`` as an ONNX
    step graph instead of in host numpy.

    Structurally the same function: the same warm start
    (:func:`_init_autoround`), the same collapse
    (:func:`_autoround_results`), the same safety net
    (:func:`_keep_better_of`), with only the loop in the middle swapped for
    :func:`_joint_loop_on_graph` -- see there for how closely the two
    actually track each other.

    The AdaRound comparison run the safety net needs goes through
    :func:`onnxsim.adaround._optimize_rounding_on_graph` on the same
    providers, not through the numpy loop: the point of ``step_providers`` is
    that a layer's optimization happens on the accelerator, and quietly
    running half of it on the host would give that back. It also keeps the
    comparison fair -- both candidates are then optimized in the same
    precision, so the branch is decided by the two searches rather than by
    one of them having had float64.
    """
    y_float = x @ w_nk.T
    scale_nk0, v0, c0 = _init_autoround(w_nk, scale_blocks, block_size)
    v, c = _joint_loop_on_graph(
        w_nk,
        scale_blocks,
        block_size,
        x,
        y_float,
        v0,
        c0,
        n_min,
        n_max,
        num_iterations,
        learning_rate,
        clip_learning_rate,
        reg_param,
        warm_start,
        beta_range,
        clip_ratio_range,
        providers,
    )
    joint = _autoround_results(
        w_nk, scale_blocks, block_size, v, c, n_min, n_max, clip_ratio_range
    )
    codes_ada_only = _optimize_rounding_on_graph(
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
        providers=providers,
    )
    return _keep_better_of(x, y_float, scale_blocks, scale_nk0, joint, codes_ada_only)


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
    step_providers: Optional[Sequence[str]] = None,
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
    :param step_providers: onnxruntime execution providers to run the *Adam
            optimization itself* on, as an ONNX step graph
            (:mod:`onnxsim.qat_graph`) rather than in host numpy -- the way
            to reach a GPU, an NPU execution provider, or (in the WASM
            build) WebGPU with this loop. Both halves of the layer's work go
            there: the joint rounding-and-clip search and the AdaRound
            comparison run its safety net needs. ``None``, the default,
            keeps the in-process float64 numpy loop, which is exact and
            deterministic; a step graph computes in float32, so its result
            agrees closely rather than bit-exactly. See ``docs/qat.md``.
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

        optimize = (
            _optimize_rounding_and_clip
            if step_providers is None
            else functools.partial(
                _optimize_rounding_and_clip_on_graph, providers=step_providers
            )
        )
        codes_nk, scale_blocks_new = optimize(
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
