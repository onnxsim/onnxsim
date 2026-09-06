"""AffineQuant (Ma et al., 2024, ICLR, "AffineQuant: Affine Transformation
Quantization for Large Language Models", https://arxiv.org/abs/2403.12544).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.omniquant` (AffineQuant's own reference
implementation, like OmniQuant's, optimizes live PyTorch modules with
backpropagation, with no ONNX export path).

**Relationship to :mod:`onnxsim.omniquant`.** AffineQuant is explicitly
framed by its own paper as a generalization of OmniQuant's Learnable
Equivalent Transformation (LET): OmniQuant (and, before it,
:mod:`onnxsim.smoothquant`) migrates activation quantization difficulty
into the weight via a *diagonal* per-channel transform -- a single scale
(and, for OmniQuant, a shift) applied independently to each activation
channel, with no mixing across channels. AffineQuant instead optimizes a
full *invertible affine transformation matrix* jointly with the
quantization: because a general matrix can also rotate/mix channels
against each other (not just rescale each one on its own), it has
strictly more representational freedom to reduce quantization error than
any diagonal transform can reach, at higher compensation cost (a matrix
multiply on the activation side and a matrix-weight product on the weight
side, instead of an elementwise scale). A diagonal matrix is a special
case of an affine matrix, so this module's own search is set up to never
do worse than :mod:`onnxsim.omniquant`'s own diagonal LET on the same
layer -- see "Search strategy" below.

**Scope: block-diagonal, not dense.** The paper's own experiments use a
*fully dense* per-layer transformation matrix, optimized end-to-end by
gradient descent. A dense ``[K, K]`` matrix does not fit this repo's
closed-form/grid-search style (see :mod:`onnxsim.omniquant`'s own
docstring for why this series prefers that over a hand-rolled autodiff
loop): choosing a good dense ``K x K`` matrix without gradients means
solving a ``K``-dimensional optimization problem, and *compensating* it
means inverting a dense ``K x K`` matrix, which for a real transformer's
hidden size (thousands) is both numerically fragile (a poorly-conditioned
dense inverse can blow up the compensated weight's dynamic range, making
quantization *worse*, not better) and expensive to insert as a runtime
``MatMul``. This module instead restricts the transform to
**block-diagonal**: the ``K`` activation channels are partitioned into
fixed-size blocks (``affine_block_size``, default 8; configurable), and
each block gets its own small, independently-invertible square matrix,
with zero coupling across blocks. This keeps the search per-block (a
handful of channels at a time, tractable to search or solve in closed
form) and the compensation an efficient block-diagonal matrix multiply
(a ``[K, K]`` matrix that is mostly zero, structurally similar to grouped
convolution) rather than a dense ``K x K`` solve. It is a strictly less
expressive family than the paper's own dense matrix, but strictly more
expressive than OmniQuant's diagonal one -- a deliberate middle point
documented here as this module's own tractability tradeoff.

**Search strategy.** Per compensated layer, three transform families are
tried and the empirically-best one (lowest weight reconstruction error
against real calibration activations, exactly as :mod:`onnxsim.omniquant`
already measures) is kept:

1. **No transform** -- plain Learnable Weight Clipping (LWC) only, reusing
   :mod:`onnxsim.omniquant`'s own grid-searched per-block clip ratio
   unchanged.
2. **Diagonal LET** -- :mod:`onnxsim.omniquant`'s own closed-form shift
   plus alpha-grid-searched per-channel scale, unchanged.
3. **Block-affine LET** -- this module's own contribution: within each
   ``affine_block_size``-channel block, the block's own calibration
   activation covariance (after the same closed-form mean-shift as (2))
   is diagonalized via ``numpy.linalg.eigh``, giving an orthonormal basis
   (a rotation) for that block. Stacking these per-block rotations along
   the diagonal gives a block-diagonal orthogonal matrix ``R`` for the
   whole layer -- trivially and exactly invertible (``R^-1 = R^T``, no
   numerical solve needed, unlike an arbitrary invertible matrix) by
   construction, which is what keeps this restriction numerically stable.
   OmniQuant's own alpha-grid-searched per-channel scale is then
   recomputed *in the rotated basis* (on the rotated activation and
   rotated weight column statistics) and searched exactly as in (2).
   Because a rotation is orthogonal, family (2) is exactly family (3)
   restricted to ``R = I`` (or, when ``affine_block_size == 1``, every
   block's "rotation" is a trivial ``1x1`` matrix that cannot mix
   anything) -- so trying (3) can only match or beat (2), and (2) can
   only match or beat (1), on the same reconstruction-error objective
   this module measures. This mirrors :mod:`onnxsim.omniquant`'s own
   "each stage's first candidate reproduces the previous stage" guarantee
   that keeps it never worse than plain RTN.

Like OmniQuant, and unlike a pure migration module such as
:mod:`onnxsim.outlier_suppression`/:mod:`onnxsim.smoothquant`, this
module does not return a float model for a later, separate quantizer to
consume: the paper's own design (following OmniQuant's LET framework
directly) jointly optimizes the equivalent transformation *against* the
weight-only INT4 quantization error itself, so the transform choice and
the quantization it enables are inseparable here -- exactly the same
scope and non-float-migration contract :mod:`onnxsim.omniquant` already
documents and this module deliberately keeps, for the same reason.

This targets the same shape :mod:`onnxsim.omniquant` targets: every
``quantize_weight_only_int4``-quantized MatMul/Gemm layer (by node output
name) present in both a float model and its quantized counterpart, whose
activation input is a plain 2-D tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs, _pack_int4
from onnxsim.bias_correction import (
    _activation_rows,
    _add_probe_outputs,
    _all_names,
    _unique_name,
)
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip


@dataclass
class _AffineQuantRewrite:
    output_name: str
    activation_name: str
    wq_name: str
    ws_name: str
    codes: np.ndarray  # int8, original (Wq's) layout/shape
    scale: np.ndarray  # float32, original (Ws's) layout/shape
    channel_scale: Optional[np.ndarray]  # None means no LET transform needed
    shift: Optional[np.ndarray]  # [K]; paired with channel_scale
    rotation: Optional[np.ndarray]  # [K, K] block-diagonal; None means diagonal-only
    bias_correction: Optional[np.ndarray]  # [N]; paired with channel_scale


def _block_diagonal_rotation(
    x_centered: np.ndarray, affine_block_size: int
) -> np.ndarray:
    """Builds the ``[K, K]`` block-diagonal orthogonal rotation matrix used
    by this module's block-affine LET candidate: each ``affine_block_size``
    -channel block gets the eigenvectors of that block's own calibration
    covariance (over ``x_centered``, already mean-shifted) as its local
    orthonormal basis. Orthogonal by construction (``numpy.linalg.eigh``
    always returns orthonormal eigenvectors of a symmetric matrix), so the
    result is exactly invertible via its own transpose -- no matrix solve.
    """
    num_samples, k = x_centered.shape
    rotation = np.zeros((k, k), dtype=np.float64)
    for start in range(0, k, affine_block_size):
        stop = min(start + affine_block_size, k)
        block = x_centered[:, start:stop]
        cov = (block.T @ block) / max(num_samples, 1)
        # eigh: cov is symmetric PSD by construction (a Gram matrix), so
        # its eigenvectors are real and orthonormal.
        _, eigvecs = np.linalg.eigh(cov)
        rotation[start:stop, start:stop] = eigvecs
    return rotation


def _diagonal_scale_grid(
    weight_col_absmax: np.ndarray, act_col_absmax: np.ndarray, alphas: np.ndarray
):
    """Yields, for each ``alpha``, the same SmoothQuant/OmniQuant-style
    closed-form per-channel scale ``s_j = act_j**alpha / weight_j**(1 -
    alpha)`` (renormalized to unit geometric mean) that
    :func:`onnxsim.apply_omniquant`'s own LET stage already computes,
    reused verbatim here in whatever basis the caller passes in.
    """
    for alpha in alphas:
        raw = act_col_absmax**alpha / weight_col_absmax ** (1.0 - alpha)
        yield raw / np.exp(np.mean(np.log(raw)))


def _reconstruction_error(
    x: np.ndarray,
    y_float: np.ndarray,
    w_hat_nk: np.ndarray,
    shift,
    rotation,
    channel_scale,
    bias_correction,
) -> float:
    if channel_scale is None:
        y_hat = x @ w_hat_nk.T
    else:
        x_centered = x - shift[np.newaxis, :]
        x_rot = x_centered @ rotation if rotation is not None else x_centered
        x_transformed = x_rot / channel_scale[np.newaxis, :]
        y_hat = x_transformed @ w_hat_nk.T + bias_correction[np.newaxis, :]
    return float(np.mean((y_float - y_hat) ** 2))


def apply_affinequant(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clip_steps: int = 20,
    num_alpha_steps: int = 20,
    min_clip_ratio: float = 0.5,
    affine_block_size: int = 8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies AffineQuant-style learnable weight clipping and block-affine
    learnable equivalent transformation to every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``,
    using real activations captured from ``float_model``. See this
    module's own docstring for the technique and how it generalizes
    :func:`onnxsim.apply_omniquant`'s diagonal transform.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact.
    :param calibration_data: representative input batches to search and
            measure the transform on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_clip_steps: grid points for the LWC clip ratio, evenly
            spaced over ``[min_clip_ratio, 1.0]`` inclusive (``1.0``,
            i.e. plain min/max scaling, is always the grid's last point,
            so a block LWC can't improve keeps its original scale)
    :param num_alpha_steps: grid points for the LET migration-strength
            exponent, evenly spaced over ``[0, 1]`` inclusive (``0``,
            i.e. no scale/shift transform at all, is always the grid's
            first point), used identically for both the diagonal and the
            block-affine candidate
    :param min_clip_ratio: the LWC grid's lower bound
    :param affine_block_size: the size of each independently-rotated block
            of activation channels in the block-affine candidate (see this
            module's own "Scope" docstring section). Larger blocks can
            capture more cross-channel structure but cost a bigger
            per-block eigendecomposition and a bigger inserted block of
            the compensating ``MatMul``; must be a positive divisor
            consideration only -- layers whose K isn't evenly divisible by
            it simply skip the block-affine candidate (falling back to (2)
            or (1)) rather than erroring.
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer AffineQuant measurably
            improved rewritten: its INT4 weight/scale replaced by the
            LWC-reclipped, LET-transformed-and-requantized versions, and
            (only when a diagonal or block-affine transform was found to
            help) a new ``Sub``/(optionally ``MatMul``)/``Mul`` inserted
            before it transforming its activation input plus a new
            ``Add`` folding in the constant bias correction after it. A
            layer AffineQuant found no LET improvement for still gets its
            LWC-only reclipping, with no inserted activation-side nodes.
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

    clip_ratios = np.linspace(min_clip_ratio, 1.0, num_clip_steps)[::-1]  # 1.0 first
    alphas = np.linspace(0.0, 1.0, num_alpha_steps)

    rewrites: List[_AffineQuantRewrite] = []
    for c in candidates:
        acts = _activation_rows(activations[c.float_node.input[0]])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if c.weight_transposed else w.T  # [N, K]
        if x.shape[1] != w_nk.shape[1]:
            continue
        k = w_nk.shape[1]

        y_float = x @ w_nk.T

        # Candidate 1: LWC only (no LET) -- clip_ratio=1.0 is always tried
        # first and is exactly quantize_weight_only_int4's own scale, so
        # this candidate can only match or improve on plain RTN.
        best_codes_nk, best_scale_blocks = _quantize_blockwise_int4_with_clip(
            w_nk, c.block_size, 1.0
        )
        best_err = _reconstruction_error(
            x,
            y_float,
            best_codes_nk * np.repeat(best_scale_blocks, c.block_size, axis=1),
            None,
            None,
            None,
            None,
        )
        best_clip_ratio = 1.0
        for clip_ratio in clip_ratios[1:]:
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_nk, c.block_size, clip_ratio
            )
            w_hat_nk = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            err = _reconstruction_error(x, y_float, w_hat_nk, None, None, None, None)
            if err < best_err:
                best_err = err
                best_clip_ratio = clip_ratio
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks

        shift = np.mean(x, axis=0)  # [K]
        x_centered = x - shift[np.newaxis, :]

        best_channel_scale = None
        best_shift = None
        best_rotation = None
        best_bias_correction = None

        # Candidate 2: diagonal LET (OmniQuant's own transform) on top of
        # the best LWC ratio found -- alpha == 0 (no transform) is
        # already covered by candidate 1, so only alpha > 0 is tried.
        weight_col_absmax = np.maximum(np.abs(w_nk).max(axis=0), 1e-12)  # [K]
        act_col_absmax = np.maximum(np.abs(x_centered).mean(axis=0), 1e-12)  # [K]
        bias_correction = w_nk @ shift  # [N]
        for channel_scale in _diagonal_scale_grid(
            weight_col_absmax, act_col_absmax, alphas[1:]
        ):
            w_scaled_nk = w_nk * channel_scale[np.newaxis, :]
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_scaled_nk, c.block_size, best_clip_ratio
            )
            w_hat_nk = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            err = _reconstruction_error(
                x, y_float, w_hat_nk, shift, None, channel_scale, bias_correction
            )
            if err < best_err:
                best_err = err
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks
                best_channel_scale = channel_scale
                best_shift = shift
                best_rotation = None
                best_bias_correction = bias_correction

        # Candidate 3: block-affine LET -- this module's own contribution.
        # Only attempted when K divides evenly into affine_block_size
        # blocks (see the "Scope" docstring section); otherwise this
        # candidate is skipped and the search falls back to whichever of
        # (1)/(2) already won above.
        if affine_block_size >= 1 and k % affine_block_size == 0:
            rotation = _block_diagonal_rotation(x_centered, affine_block_size)
            x_rot = x_centered @ rotation
            w_rot_nk = w_nk @ rotation
            weight_col_absmax_rot = np.maximum(np.abs(w_rot_nk).max(axis=0), 1e-12)
            act_col_absmax_rot = np.maximum(np.abs(x_rot).mean(axis=0), 1e-12)
            for channel_scale in _diagonal_scale_grid(
                weight_col_absmax_rot, act_col_absmax_rot, alphas[1:]
            ):
                w_hat_rot_nk = w_rot_nk * channel_scale[np.newaxis, :]
                codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                    w_hat_rot_nk, c.block_size, best_clip_ratio
                )
                w_hat_nk = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
                err = _reconstruction_error(
                    x,
                    y_float,
                    w_hat_nk,
                    shift,
                    rotation,
                    channel_scale,
                    bias_correction,
                )
                if err < best_err:
                    best_err = err
                    best_codes_nk, best_scale_blocks = codes_nk, scale_blocks
                    best_channel_scale = channel_scale
                    best_shift = shift
                    best_rotation = rotation
                    best_bias_correction = bias_correction

        codes_orig = best_codes_nk if c.weight_transposed else best_codes_nk.T
        scale_orig = best_scale_blocks if c.weight_transposed else best_scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)
        rewrites.append(
            _AffineQuantRewrite(
                output_name=c.output_name,
                activation_name=c.float_node.input[0],
                wq_name=c.wq_name,
                ws_name=c.ws_init.name,
                codes=codes_orig.astype(np.int8),
                scale=scale_orig.astype(np.float32),
                channel_scale=best_channel_scale,
                shift=best_shift,
                rotation=best_rotation,
                bias_correction=best_bias_correction,
            )
        )

    if not rewrites:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)

    codes_by_name = {r.wq_name: r.codes for r in rewrites}
    scale_by_name = {r.ws_name: r.scale for r in rewrites}
    for t in corrected.graph.initializer:
        codes = codes_by_name.get(t.name)
        if codes is not None:
            t.raw_data = _pack_int4(codes)
        scale = scale_by_name.get(t.name)
        if scale is not None:
            t.CopyFrom(onnx.numpy_helper.from_array(scale, name=t.name))

    taken_names: Set[str] = _all_names(corrected.graph)
    q_by_output = _node_outputs(corrected.graph)
    for r in rewrites:
        if r.channel_scale is None or r.shift is None or r.bias_correction is None:
            continue
        qn = q_by_output[r.output_name]
        act_input = r.activation_name
        inv_scale = (1.0 / r.channel_scale).astype(np.float32)

        shift_name = _unique_name(f"{act_input}_affinequant_shift", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(r.shift.astype(np.float32), name=shift_name)
        )
        centered_name = _unique_name(f"{act_input}_affinequant_centered", taken_names)
        sub_node = onnx.helper.make_node(
            "Sub",
            [act_input, shift_name],
            [centered_name],
            name=_unique_name(f"{act_input}_affinequant_sub", taken_names),
        )
        node_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(node_idx, sub_node)
        node_idx += 1

        rotated_name = centered_name
        if r.rotation is not None:
            rotation_name = _unique_name(
                f"{act_input}_affinequant_rotation", taken_names
            )
            corrected.graph.initializer.append(
                onnx.numpy_helper.from_array(
                    r.rotation.astype(np.float32), name=rotation_name
                )
            )
            rotated_name = _unique_name(f"{act_input}_affinequant_rotated", taken_names)
            matmul_node = onnx.helper.make_node(
                "MatMul",
                [centered_name, rotation_name],
                [rotated_name],
                name=_unique_name(f"{act_input}_affinequant_matmul", taken_names),
            )
            corrected.graph.node.insert(node_idx, matmul_node)
            node_idx += 1

        inv_scale_name = _unique_name(f"{act_input}_affinequant_inv_scale", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(inv_scale, name=inv_scale_name)
        )
        scaled_name = _unique_name(f"{act_input}_affinequant_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [rotated_name, inv_scale_name],
            [scaled_name],
            name=_unique_name(f"{act_input}_affinequant_mul", taken_names),
        )
        corrected.graph.node.insert(node_idx, mul_node)
        qn.input[0] = scaled_name

        old_output = qn.output[0]
        base_name = _unique_name(f"{r.output_name}_affinequant_base", taken_names)
        qn.output[0] = base_name
        bias_name = _unique_name(f"{r.output_name}_affinequant_bias", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(
                r.bias_correction.astype(np.float32), name=bias_name
            )
        )
        add_node = onnx.helper.make_node(
            "Add",
            [base_name, bias_name],
            [old_output],
            name=_unique_name(f"{r.output_name}_affinequant_bias_add", taken_names),
        )
        qn_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(qn_idx + 1, add_node)

    return corrected
