"""AWQ -- Activation-aware Weight Quantization (Lin et al., 2023, "AWQ:
Activation-aware Weight Quantization for LLM Compression and Acceleration",
https://arxiv.org/abs/2306.00978). Originally implemented for PyTorch/CUDA
LLM deployment (most visibly as part of Meta's ``torchao`` -- see this
repository's own research into whether ``torchao``/``mslk`` have any real
ONNX interop point, which they do not: they quantize live PyTorch
``nn.Module``s with runtime-only tensor subclasses, with no ONNX export
path). AWQ's *algorithm*, unlike its PyTorch implementation, is entirely
portable: it needs nothing but a weight tensor, a round-to-nearest
block-wise integer quantizer, and real calibration activations -- exactly
:func:`onnxsim.quantize_weight_only_int4`'s own scheme and
:mod:`onnxsim.adaround`'s own "post-hoc adjustment from real activations"
style. This module ports the algorithm, not any framework's code.

AWQ's key empirical observation: not every weight element matters equally to
a layer's output -- a weight column feeding a large-magnitude *activation*
channel dominates that channel's contribution to the output, so quantization
error there costs more than an equal-sized error on a column whose
activation is small. Round-to-nearest (what :func:`quantize_weight_only_int4`
does) treats every element identically regardless of this, and per-element
optimization of *which way* to round (:func:`onnxsim.apply_adaround`) never
touches the weight's own magnitude, only which quantization bin it lands in.
AWQ instead rescales entire input-channel *columns* of the weight upward in
proportion to that channel's own average activation magnitude before
quantizing -- inflating a salient column's share of its block's dynamic
range so round-to-nearest's fixed per-block step size costs it
proportionally less -- and applies the exact inverse scale to the
activation, via a new ``Mul`` node inserted right before the layer, so the
transformation is a no-op on the *unquantized* function and only changes how
much quantization error each column ends up absorbing.

For every :func:`onnxsim.quantize_weight_only_int4`-quantized MatMul/Gemm
present (by node output name) in both ``float_model`` and ``quantized_model``
(the same matching :mod:`onnxsim.adaround` uses -- see
``weight_only_quantize_int4_matmul.h``'s scheme this targets): measures each
input channel's average activation magnitude from real calibration data,
then grid-searches a single scalar exponent ``alpha`` (the per-channel scale
is ``activation_magnitude ** alpha``, geometric-mean-normalized to keep the
compensating activation scale well-conditioned) to minimize the layer's own
reconstruction error, re-quantizing the rescaled weight from scratch at each
candidate ``alpha`` and measuring it against real activations -- the same
grid search the AWQ paper itself uses, not a closed-form guarantee. ``alpha
= 0`` (uniform scale 1, i.e. plain round-to-nearest) is always one of the
candidates, so a layer AWQ can't improve keeps its original quantization and
gets no inserted node at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs, _pack_int4
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data


@dataclass
class _AwqRewrite:
    output_name: str  # the MatMul/Gemm node's own output name
    activation_name: str  # that node's activation input name
    wq_name: str
    ws_name: str
    codes: np.ndarray  # int8, original (Wq's) layout/shape
    scale: np.ndarray  # float32, original (Ws's) layout/shape
    channel_scale: Optional[np.ndarray]  # None means "no Mul needed" (alpha == 0)


def _quantize_blockwise_int4(
    w_nk: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Fresh round-to-nearest block-wise INT4 quantization of ``w_nk``
    ([N, K], output channel first), matching
    ``TryQuantizeWeightBlockwiseInt4InPlace``'s own scheme (one scale per
    ``(output channel, block-of-K)`` group, ``scale = max(|w| in block) /
    7``, codes clamped to ``[-7, 7]``). Returns ``(codes_nk, scale_blocks)``
    with ``scale_blocks`` shape ``[N, K // block_size]``. Assumes ``K %
    block_size == 0`` (true of every candidate this module matches, since
    ``block_size`` is read from the existing DequantizeLinear node that
    already quantized this same weight).
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale_blocks = np.maximum(np.abs(blocks).max(axis=2), 1e-12) / 7.0
    scale_full = np.repeat(scale_blocks, block_size, axis=1)
    codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)
    return codes_nk, scale_blocks


def apply_awq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_alpha_steps: int = 20,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies AWQ-style activation-aware per-channel weight rescaling to
    every ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present
    (by node output name) in both ``float_model`` and ``quantized_model``,
    using real activations captured from ``float_model``. See this module's
    own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input isn't a plain 2-D tensor, are left untouched. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to search and
            measure the rescaling on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative search than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_alpha_steps: grid points for the per-channel scale exponent
            ``alpha``, evenly spaced over ``[0, 1]`` inclusive (matching the
            AWQ paper's own grid search); higher values search more finely
            at proportionally more cost (one full re-quantization and
            reconstruction-error measurement per candidate, per layer)
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer AWQ measurably improved
            rewritten: its INT4 weight and scale initializers replaced with
            the rescaled-and-requantized versions, and a new ``Mul`` node
            inserted before it applying the compensating inverse channel
            scale to its activation input. A layer AWQ found no improvement
            for (``alpha = 0`` best) is left completely untouched.
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

    alphas = np.linspace(0.0, 1.0, num_alpha_steps)

    rewrites: List[_AwqRewrite] = []
    for c in candidates:
        acts = [a for a in activations[c.float_node.input[0]] if a.ndim == 2]
        if not acts:
            continue  # not a plain 2-D activation; skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if c.weight_transposed else w.T  # [N, K], output channel first
        if x.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip

        y_float = x @ w_nk.T
        # AWQ's own saliency signal: each input channel's average activation
        # magnitude across the calibration set.
        act_magnitude = np.maximum(np.abs(x).mean(axis=0), 1e-12)  # [K]

        # alpha == 0 (uniform scale 1, i.e. plain round-to-nearest) is
        # always a candidate and always the grid's first point -- evaluate
        # it first so best_* below never needs an Optional/None sentinel.
        best_channel_scale = np.ones_like(act_magnitude)
        best_codes_nk, best_scale_blocks = _quantize_blockwise_int4(w_nk, c.block_size)
        best_scale_full = np.repeat(best_scale_blocks, c.block_size, axis=1)
        best_err = float(
            np.mean((y_float - x @ (best_codes_nk * best_scale_full).T) ** 2)
        )
        best_alpha = 0.0

        for alpha in alphas[1:]:
            raw = act_magnitude**alpha
            # Geometric-mean normalization keeps the scale (and its inverse,
            # applied to the activation) centered around 1 -- standard AWQ
            # practice, and here purely for numerical hygiene since
            # weight-only quantization has no activation dynamic-range
            # constraint to respect.
            channel_scale = raw / np.exp(np.mean(np.log(raw)))
            w_scaled_nk = w_nk * channel_scale[np.newaxis, :]
            codes_nk, scale_blocks = _quantize_blockwise_int4(w_scaled_nk, c.block_size)
            scale_full = np.repeat(scale_blocks, c.block_size, axis=1)
            w_hat_nk = codes_nk * scale_full
            y_hat = (x / channel_scale[np.newaxis, :]) @ w_hat_nk.T
            err = float(np.mean((y_float - y_hat) ** 2))
            if err < best_err:
                best_err = err
                best_alpha = alpha
                best_codes_nk = codes_nk
                best_scale_blocks = scale_blocks
                best_channel_scale = channel_scale

        codes_orig = best_codes_nk if c.weight_transposed else best_codes_nk.T
        scale_orig = best_scale_blocks if c.weight_transposed else best_scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)
        rewrites.append(
            _AwqRewrite(
                output_name=c.output_name,
                # quantize_weight_only_int4 only ever replaces a node's
                # weight input (index 1), never its activation input (index
                # 0) or the node's own identity -- so the quantized graph's
                # matching node still has this exact activation input name.
                activation_name=c.float_node.input[0],
                wq_name=c.wq_name,
                ws_name=c.ws_init.name,
                codes=codes_orig.astype(np.int8),
                scale=scale_orig.astype(np.float32),
                channel_scale=None if best_alpha == 0.0 else best_channel_scale,
            )
        )

    if not rewrites:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)

    codes_by_name = {r.wq_name: r.codes for r in rewrites}
    scale_by_name = {
        r.ws_name: r.scale for r in rewrites if r.channel_scale is not None
    }
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
        if r.channel_scale is None:
            continue
        qn = q_by_output[r.output_name]
        act_input = r.activation_name
        inv_scale = (1.0 / r.channel_scale).astype(np.float32)

        scale_name = _unique_name(f"{act_input}_awq_inv_scale", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(inv_scale, name=scale_name)
        )
        scaled_name = _unique_name(f"{act_input}_awq_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [act_input, scale_name],
            [scaled_name],
            name=_unique_name(f"{act_input}_awq_mul", taken_names),
        )
        node_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(node_idx, mul_node)
        qn.input[0] = scaled_name

    return corrected
