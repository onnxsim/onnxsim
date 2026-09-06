"""SliM-LLM (Huang, Shao, Dong, Luo, Qiao et al., 2024, "SliM-LLM:
Salience-Driven Mixed-Precision Quantization for Large Language Models",
https://arxiv.org/abs/2405.14917).

:mod:`onnxsim.mixed_precision` already assigns different bit-widths within
one model, but only at **layer** granularity: it picks one bit-width
(INT4 or INT8) for an entire matched MatMul/Gemm weight, based on that
layer's own sensitivity score. SliM-LLM's distinguishing contribution is a
finer granularity -- it picks a bit-width per **group** *within* a single
weight matrix: the same reduction-axis blocks
:func:`onnxsim.quantize_weight_only_int4`/:mod:`onnxsim.mixed_precision`
already partition a layer's weight into (``group_size`` contiguous
elements along ``K``) each get their own, independently chosen bit-width,
so two groups in the *same layer* can end up quantized at different
precisions. The key phrase: :mod:`onnxsim.mixed_precision` picks one
bit-width per *layer*; this module picks one bit-width per *group within a
layer*.

**The salience score.** Mirrors :mod:`onnxsim.owq`'s own Optimal Brain
Surgeon-style column saliency (itself built on :mod:`onnxsim.gptq`'s
Hessian machinery) rather than reinventing one: for a layer's weight ``W``
([N, K], output channel first) and calibration activations ``X``
([samples, K]), ``H = X^T X`` is the same per-layer Hessian
:mod:`onnxsim.gptq`/:mod:`onnxsim.owq` already compute, and

    sensitivity_j = mean_n[(W[n, j] - RTN_low(W[n, j]))^2] / [H^-1]_jj

is :mod:`onnxsim.owq`'s own per-*column* saliency -- ``RTN_low`` here is
round-to-nearest at ``low_bits`` (this module's least-precise tier), so the
numerator captures how much reconstruction error each column would suffer
at the aggressive end of the bit budget, and ``[H^-1]_jj`` discounts
columns whose error other columns could compensate for. This module
aggregates that per-column score to a per-*group* salience (the mean of
``sensitivity_j`` over each group's ``group_size`` columns), then ranks
groups by it -- the same score OWQ uses to rank columns, applied one level
coarser to rank groups instead.

**Bit assignment.** Each layer gets a two-tier budget, ``low_bits`` (the
default candidate set is ``{2, 4}``, matching the paper's own ultra-low-bit
regime) and ``high_bits``, mixed so the layer's own average bits/weight
hits ``target_bits``: solving
``fraction_high * high_bits + (1 - fraction_high) * low_bits = target_bits``
for ``fraction_high`` (clipped to ``[0, 1]``) gives the count of groups
(rounded, most-salient first) that get ``high_bits``; every other group
gets ``low_bits``. This mirrors :mod:`onnxsim.mixed_precision`'s own
``high_bits_fraction`` parameter, except computed from a target *average*
bit budget instead of being supplied directly, and applied to groups
within one layer instead of to whole layers across the model.

**Storage.** Both tiers' codes are symmetric round-to-nearest integers in
``[-(2^(bits-1) - 1), 2^(bits-1) - 1]``, stored as a single ``INT8``
initializer per layer (wide enough for every bit-width this module
supports, so no bit-packing is needed -- the same simplification
:mod:`onnxsim.mixed_precision`'s own INT8 tier already makes) with a
per-(group, output-channel) float32 scale, reconstructed via one
``DequantizeLinear(..., axis=0, block_size=group_size)`` -- identical graph
shape to :func:`onnxsim.quantize_weight_only_int8_block`, since a group's
chosen bit-width only changes what range its own codes and scale were
computed over, not how they are dequantized. Each layer's per-group
bit-width is additionally recorded in a parallel ``INT64`` initializer
(mirroring how :mod:`onnxsim.owq`/:mod:`onnxsim.spqr` already attach
per-group/per-column metadata alongside their packed codes) purely for
inspection -- the dequantization subgraph does not read it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import (
    _activation_rows,
    _add_probe_outputs,
    _all_names,
    _unique_name,
)
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _inverse_hessian_cholesky
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _quantize_blockwise_nbit(
    w_nk: np.ndarray, block_size: int, bits: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Round-to-nearest block-wise symmetric quantization of ``w_nk``
    ([N, K], output channel first) to ``bits``-bit codes in
    ``[-(2^(bits-1) - 1), 2^(bits-1) - 1]`` -- the same per-(output
    channel, block) granularity :mod:`onnxsim.mixed_precision`'s own
    ``_quantize_blockwise_int8`` uses, generalized to an arbitrary bit
    count instead of a fixed 8.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    qmax = float(2 ** (bits - 1) - 1)
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale_blocks = np.maximum(np.abs(blocks).max(axis=2), 1e-12) / qmax
    scale_full = np.repeat(scale_blocks, block_size, axis=1)
    codes_nk = np.clip(np.round(w_nk / scale_full), -qmax, qmax)
    return codes_nk, scale_blocks


def apply_slim_llm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    target_bits: float = 3.0,
    low_bits: int = 2,
    high_bits: int = 4,
    group_size: int = 32,
    percdamp: float = 0.01,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight whose reduction dimension ``K`` is divisible by
    ``group_size`` to a per-group mix of ``low_bits``/``high_bits``
    integer codes, chosen from a calibration-driven per-group salience
    score so each layer's own average bits/weight lands at ``target_bits``
    -- see this module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to build each layer's Hessian and per-column
            error -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a more representative salience ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param target_bits: target average bits/weight for each matched
            layer, met by mixing ``low_bits``- and ``high_bits``-coded
            groups (most salient groups get ``high_bits``); must lie in
            ``[low_bits, high_bits]`` -- values outside that range are
            clipped to it
    :param low_bits: bit-width for the least salient groups in each layer
            (the paper's own regime is ultra-low-bit, so the default is 2)
    :param high_bits: bit-width for the most salient groups in each layer
    :param group_size: elements per quantization group along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :mod:`onnxsim.gptq`/:mod:`onnxsim.owq`'s own default
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            per-group mixed ``low_bits``/``high_bits`` INT8-stored codes
            plus a per-group float32 scale
            (``DequantizeLinear(..., axis=0, block_size=group_size)``) and
            a parallel per-group bit-width INT64 initializer; output
            tensor name unchanged. Layers with a non-constant, non-2-D
            weight, a reduction dimension not divisible by ``group_size``,
            or no calibration activation available, are left untouched; a
            model with no matching layer, or an opset older than 21
            (``DequantizeLinear``'s ``block_size`` attribute needs opset
            21), is returned unchanged
    """
    if low_bits < 2 or high_bits <= low_bits:
        raise ValueError("require 2 <= low_bits < high_bits")

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    nodes = list(graph.node)
    candidates = []
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, bias_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        dims = list(w_init.dims)
        w_nk_shape = dims if weight_transposed else dims[::-1]
        if w_nk_shape[1] % group_size != 0:
            continue
        candidates.append((node, x_name, w_name, bias_name, weight_transposed))

    if not candidates:
        return out

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(model, probe_names)
    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            activations[name].extend(_activation_rows([x]))

    target_bits_clamped = min(max(target_bits, float(low_bits)), float(high_bits))
    fraction_high = (target_bits_clamped - low_bits) / (high_bits - low_bits)

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        acts = activations.get(x_name)
        if not acts:
            continue  # no usable calibration activation observed; skip

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape

        x = np.concatenate(acts, axis=0)
        if x.shape[1] != k:
            continue  # activation's feature dim doesn't match K; skip

        h = x.T @ x
        u = _inverse_hessian_cholesky(h, percdamp)  # inv(h) == u.T @ u
        h_inv_diag = np.maximum((u**2).sum(axis=0), 1e-12)  # [K]

        codes_low_nk, scale_low_nk = _quantize_blockwise_nbit(
            w_nk, group_size, low_bits
        )
        scale_full = np.repeat(scale_low_nk, group_size, axis=1)
        w_rtn_low = codes_low_nk * scale_full
        col_error = np.mean((w_nk - w_rtn_low) ** 2, axis=0)  # [K]
        sensitivity = col_error / h_inv_diag  # [K], OWQ's own per-column score

        num_groups = k // group_size
        group_salience = sensitivity.reshape(num_groups, group_size).mean(axis=1)

        num_high = int(round(fraction_high * num_groups))
        high_groups = set(np.argsort(-group_salience)[:num_high].tolist())

        codes_high_nk, scale_high_nk = _quantize_blockwise_nbit(
            w_nk, group_size, high_bits
        )

        codes_nk = codes_low_nk.copy()
        scale_blocks_nk = scale_low_nk.copy()
        group_bits = np.full(num_groups, low_bits, dtype=np.int64)
        for g in high_groups:
            lo, hi = g * group_size, (g + 1) * group_size
            codes_nk[:, lo:hi] = codes_high_nk[:, lo:hi]
            scale_blocks_nk[:, g] = scale_high_nk[:, g]
            group_bits[g] = high_bits

        codes_kn = codes_nk.T.astype(np.int8)  # [K, N], ready for DequantizeLinear
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/group_size, N]

        prefix = f"{w_name}_slimllm"
        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        codes_tensor = onnx.TensorProto()
        codes_tensor.name = codes_name
        codes_tensor.data_type = onnx.TensorProto.INT8
        codes_tensor.dims.extend([k, n])
        codes_tensor.raw_data = codes_kn.tobytes()
        graph.initializer.append(codes_tensor)

        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_kn, name=scale_name)
        )

        bits_name = _unique_name(f"{prefix}_group_bits", taken_names)
        graph.initializer.append(
            onnx.helper.make_tensor(
                bits_name,
                onnx.TensorProto.INT64,
                [num_groups],
                group_bits.tobytes(),
                raw=True,
            )
        )

        new_nodes: List[onnx.NodeProto] = []

        def _new(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"{prefix}_{out_suffix}", taken_names)
            n_ = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{out_suffix}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n_)
            return out_name

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=group_size,
        )
        core = _new("MatMul", [x_name, w_dequant], "core")

        old_output = node.output[0]
        if bias_name is not None:
            final = onnx.helper.make_node(
                "Add",
                [core, bias_name],
                [old_output],
                name=_unique_name(f"{prefix}_bias_add_node", taken_names),
            )
        else:
            final = onnx.helper.make_node(
                "Identity",
                [core],
                [old_output],
                name=_unique_name(f"{prefix}_identity_node", taken_names),
            )
        new_nodes.append(final)

        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
