"""OmniQuant (Shao et al., 2023, "OmniQuant: Omnidirectionally Calibrated
Quantization for Large Language Models", https://arxiv.org/abs/2308.13137).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (OmniQuant's own
reference implementation optimizes live PyTorch modules block by block
with backpropagation, with no ONNX export path).

OmniQuant combines two calibrated (data-driven, but *not* requiring a
full backprop training loop the way the paper's own block-wise gradient
descent does -- see below for what replaces it here) adjustments on top
of plain round-to-nearest block quantization:

- **Learnable Weight Clipping (LWC).** :func:`onnxsim.quantize_weight_only_int4`
  and every other onnxsim INT4 scheme derive each block's scale directly
  from that block's own min/max -- exactly the point a couple of outlier
  elements can distort (the same problem :mod:`onnxsim.hqq` addresses with
  a robust Lp loss instead). OmniQuant's LWC scales the block's min/max by
  a *learned* per-block clipping ratio (`` in (0, 1]``) instead, letting a
  block deliberately clip its most extreme elements when doing so reduces
  the block's own overall reconstruction error more than it costs. The
  paper learns this ratio by gradient descent through a straight-through
  rounding relaxation (the same relaxation :mod:`onnxsim.adaround` uses
  for a different parameter -- which bin each element rounds to, not the
  scale itself). This module instead **grid-searches** the ratio per
  block, the same search strategy :mod:`onnxsim.awq` already uses for its
  own per-channel scale: the objective is one-dimensional and well-behaved
  enough per block that a grid reliably finds as good an optimum as a few
  steps of noisy straight-through gradient descent would, without the risk
  of a hand-rolled autodiff-adjacent implementation silently getting a
  gradient wrong.
- **Learnable Equivalent Transformation (LET).** Like :mod:`onnxsim.smoothquant`,
  LET migrates activation quantization difficulty into the weight via a
  per-channel scale -- but it also *shifts* the activation by its own
  per-channel mean first, letting the transform also absorb a channel-wise
  DC offset (useful when the preceding op, e.g. a LayerNorm, leaves
  activations asymmetric around zero) before the scale is even applied.
  The paper learns both the scale and the shift jointly with LWC via the
  same block-wise gradient descent. This module instead sets the shift in
  **closed form** (each channel's own mean over the calibration set --
  the natural choice for "center the activation before scaling it") and
  reuses :mod:`onnxsim.smoothquant`'s own closed-form per-channel scale
  formula on the now-centered activation, then greedily re-searches LWC's
  clip ratio once more against the transformed weight. Because the shift
  is constant (fixed once calibration finishes), its effect on the
  layer's output is a constant too -- ``shift @ W`` -- so it costs nothing
  at inference beyond one extra additive bias, not a runtime shift
  operation.

Both simplifications trade the paper's own joint gradient-descent
optimization for cheaper, closed-form-or-grid-searched alternatives that
target the same two objectives (a learned clip ratio, a learned
scale-and-shift activation transform) -- consistent with how every other
refinement pass in this series (:mod:`onnxsim.hqq`'s IRLS in place of
half-quadratic splitting, :mod:`onnxsim.squeezellm`'s sensitivity-weighted
k-means in place of the paper's own solver) substitutes an independently
verifiable classical technique for a paper's own bespoke or
gradient-based one, rather than risk an unverifiable line-for-line
reproduction.

Like :mod:`onnxsim.apply_awq`/:mod:`onnxsim.apply_gptq`, this only
rewrites :func:`onnxsim.quantize_weight_only_int4`'s own INT4 codes/scale
in place -- it never touches that ``DequantizeLinear``'s ``axis``
attribute -- so a plain (non-``transB=1``) MatMul/Gemm layer this module
touches is affected by the same ONNX Runtime ``MatMulNBitsFusion``
optimizer bug :mod:`onnxsim.ort_matmul_nbits_workaround` works around
(verified transparently compatible with that workaround, the same as
every other technique built on ``quantize_weight_only_int4``'s output).
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
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data


@dataclass
class _OmniQuantRewrite:
    output_name: str
    activation_name: str
    wq_name: str
    ws_name: str
    codes: np.ndarray  # int8, original (Wq's) layout/shape
    scale: np.ndarray  # float32, original (Ws's) layout/shape
    channel_scale: Optional[np.ndarray]  # None means no LET transform needed
    shift: Optional[np.ndarray]  # [K]; paired with channel_scale
    bias_correction: Optional[np.ndarray]  # [N]; paired with channel_scale


def _quantize_blockwise_int4_with_clip(
    w_nk: np.ndarray, block_size: int, clip_ratio: float
) -> "tuple[np.ndarray, np.ndarray]":
    """Round-to-nearest block-wise INT4 quantization of ``w_nk`` ([N, K],
    output channel first), with each block's scale computed from
    ``clip_ratio * max(|w| in block) / 7`` instead of the plain (
    ``clip_ratio = 1``) min/max scale -- OmniQuant's Learnable Weight
    Clipping, here with the ratio grid-searched rather than learned. See
    this module's own docstring.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale_blocks = np.maximum(np.abs(blocks).max(axis=2) * clip_ratio, 1e-12) / 7.0
    scale_full = np.repeat(scale_blocks, block_size, axis=1)
    codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)
    return codes_nk, scale_blocks


def _reconstruction_error(
    x: np.ndarray,
    y_float: np.ndarray,
    w_hat_nk: np.ndarray,
    shift,
    channel_scale,
    bias_correction,
) -> float:
    if channel_scale is None:
        y_hat = x @ w_hat_nk.T
    else:
        x_transformed = (x - shift[np.newaxis, :]) / channel_scale[np.newaxis, :]
        y_hat = x_transformed @ w_hat_nk.T + bias_correction[np.newaxis, :]
    return float(np.mean((y_float - y_hat) ** 2))


def apply_omniquant(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clip_steps: int = 20,
    num_alpha_steps: int = 20,
    min_clip_ratio: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies OmniQuant-style learnable weight clipping and learnable
    equivalent transformation to every ``quantize_weight_only_int4``-quantized
    MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``, using real activations
    captured from ``float_model``. See this module's own docstring for
    the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input isn't a plain 2-D tensor, are left untouched.
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
            i.e. no LET transform at all, is always the grid's first
            point, so a layer LET can't improve gets no inserted nodes)
    :param min_clip_ratio: the LWC grid's lower bound
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer OmniQuant measurably
            improved rewritten: its INT4 weight/scale replaced by the
            LWC-reclipped, LET-transformed-and-requantized versions, and
            (only when LET was found to help) a new ``Sub``/``Mul``
            inserted before it transforming its activation input plus a
            new ``Add`` folding in the constant bias correction after it.
            A layer OmniQuant found no LET improvement for (``alpha = 0``
            best) still gets its LWC-only reclipping, with no inserted
            activation-side nodes.
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

    rewrites: List[_OmniQuantRewrite] = []
    for c in candidates:
        acts = [a for a in activations[c.float_node.input[0]] if a.ndim == 2]
        if not acts:
            continue
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if c.weight_transposed else w.T  # [N, K]
        if x.shape[1] != w_nk.shape[1]:
            continue

        y_float = x @ w_nk.T

        # Stage 1: LWC only (no LET) -- clip_ratio=1.0 is always tried
        # first and is exactly quantize_weight_only_int4's own scale, so
        # this stage can only match or improve on plain RTN.
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
        )
        best_clip_ratio = 1.0
        for clip_ratio in clip_ratios[1:]:
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_nk, c.block_size, clip_ratio
            )
            w_hat_nk = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            err = _reconstruction_error(x, y_float, w_hat_nk, None, None, None)
            if err < best_err:
                best_err = err
                best_clip_ratio = clip_ratio
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks

        # Stage 2: LET (shift + scale) on top of the best LWC ratio found.
        shift = np.mean(x, axis=0)  # [K]
        x_centered = x - shift[np.newaxis, :]
        weight_col_absmax = np.maximum(np.abs(w_nk).max(axis=0), 1e-12)  # [K]
        act_col_absmax = np.maximum(np.abs(x_centered).mean(axis=0), 1e-12)  # [K]

        best_channel_scale = None
        best_shift = None
        best_bias_correction = None
        for alpha in alphas[1:]:  # alpha == 0 (no LET) already covered by stage 1
            raw = act_col_absmax**alpha / weight_col_absmax ** (1.0 - alpha)
            channel_scale = raw / np.exp(np.mean(np.log(raw)))
            w_scaled_nk = w_nk * channel_scale[np.newaxis, :]
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_scaled_nk, c.block_size, best_clip_ratio
            )
            w_hat_nk = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            bias_correction = w_nk @ shift  # [N]
            err = _reconstruction_error(
                x, y_float, w_hat_nk, shift, channel_scale, bias_correction
            )
            if err < best_err:
                best_err = err
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks
                best_channel_scale = channel_scale
                best_shift = shift
                best_bias_correction = bias_correction

        codes_orig = best_codes_nk if c.weight_transposed else best_codes_nk.T
        scale_orig = best_scale_blocks if c.weight_transposed else best_scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)
        rewrites.append(
            _OmniQuantRewrite(
                output_name=c.output_name,
                activation_name=c.float_node.input[0],
                wq_name=c.wq_name,
                ws_name=c.ws_init.name,
                codes=codes_orig.astype(np.int8),
                scale=scale_orig.astype(np.float32),
                channel_scale=best_channel_scale,
                shift=best_shift,
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

        shift_name = _unique_name(f"{act_input}_omniquant_shift", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(r.shift.astype(np.float32), name=shift_name)
        )
        centered_name = _unique_name(f"{act_input}_omniquant_centered", taken_names)
        sub_node = onnx.helper.make_node(
            "Sub",
            [act_input, shift_name],
            [centered_name],
            name=_unique_name(f"{act_input}_omniquant_sub", taken_names),
        )
        inv_scale_name = _unique_name(f"{act_input}_omniquant_inv_scale", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(inv_scale, name=inv_scale_name)
        )
        scaled_name = _unique_name(f"{act_input}_omniquant_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [centered_name, inv_scale_name],
            [scaled_name],
            name=_unique_name(f"{act_input}_omniquant_mul", taken_names),
        )
        node_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(node_idx, sub_node)
        corrected.graph.node.insert(node_idx + 1, mul_node)
        qn.input[0] = scaled_name

        old_output = qn.output[0]
        base_name = _unique_name(f"{r.output_name}_omniquant_base", taken_names)
        qn.output[0] = base_name
        bias_name = _unique_name(f"{r.output_name}_omniquant_bias", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(
                r.bias_correction.astype(np.float32), name=bias_name
            )
        )
        add_node = onnx.helper.make_node(
            "Add",
            [base_name, bias_name],
            [old_output],
            name=_unique_name(f"{r.output_name}_omniquant_bias_add", taken_names),
        )
        qn_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(qn_idx + 1, add_node)

    return corrected
