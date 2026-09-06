"""Empirical Bias Correction -- the data-driven half of "Data-Free
Quantization Through Weight Equalization and Bias Correction" (Nagel et al.,
2019), also shipped as part of Qualcomm's AIMET toolkit. Its sibling
technique, Cross-Layer Equalization, is :func:`onnxsim.cross_layer_equalize`.

Quantizing a layer's weight is not a zero-mean operation in general: the
rounding error correlates with the weight distribution (e.g. clipped/
asymmetric distributions round more one direction than the other), so a
quantized Conv/Gemm/MatMul's output picks up a systematic *mean* shift per
output channel on top of the expected per-element quantization noise.
:func:`onnxsim.cross_layer_equalize` and every ``quantize_*`` scheme leave
this shift uncorrected -- it is a real, measurable bias, not something
either of them targets. :func:`correct_bias` measures it directly (the
"empirical" variant of the paper's Bias Correction -- the "analytic"
variant, which estimates the same shift from BatchNorm statistics instead
of running real data through the model, is not implemented here) and
cancels it.

For every Conv/Gemm/MatMul node present (by output tensor name) in both
``float_model`` and ``quantized_model``, this runs both models on the same
calibration data, measures each such layer's own per-output-channel mean
error (``float_output - quantized_output``, independent of any other
layer's correction -- not chained through progressively-corrected
activations), and adds that as a constant per-channel offset right after
the layer in ``quantized_model``. Adding a constant to an affine layer's
output is exactly equivalent to folding it into that layer's bias, whatever
internal shape the ``quantize_*`` scheme that produced it happens to use
(a straight Conv/Gemm/MatMul weight-only rewrite, a multi-node dynamic-
quantization chain, ...) -- so this needs no scheme-specific knowledge of
where a bias tensor lives internally, only that the layer's own output
tensor kept its original name, which every onnxsim ``quantize_*`` pass
guarantees (downstream consumers are never rewired by name).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx

from onnxsim import backend
from onnxsim.calibration import Tensors, generate_random_calibration_data

# op_type -> the axis its own output tensor's "channel" dimension sits on.
# Conv is always NCHW-style (batch, channel, spatial...), so channel is
# axis 1 regardless of spatial rank. Gemm/MatMul put the output feature
# dimension last regardless of transpose attributes (Gemm's output shape is
# always [M, N] whatever transA/transB are; MatMul has no transpose
# attributes at all), so channel is axis -1.
_CORRECTABLE_OPS: Dict[str, int] = {"Conv": 1, "Gemm": -1, "MatMul": -1}


def _all_names(graph: onnx.GraphProto) -> Set[str]:
    names: Set[str] = set()
    for t in graph.initializer:
        names.add(t.name)
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(vi.name)
    for n in graph.node:
        if n.name:
            names.add(n.name)
        names.update(n.input)
        names.update(n.output)
    return names


def _unique_name(base: str, taken: Set[str]) -> str:
    name = base
    i = 0
    while name in taken:
        i += 1
        name = f"{base}_{i}"
    taken.add(name)
    return name


def _add_probe_outputs(model: onnx.ModelProto, names: Sequence[str]) -> onnx.ModelProto:
    # Same technique as calibration.py's calibrate(): expose intermediate
    # tensors as extra graph outputs so the backend computes (and returns)
    # them without the graph's own computation changing at all.
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    for name in names:
        if name not in existing:
            probe.graph.output.append(onnx.ValueInfoProto(name=name))
            existing.add(name)
    return probe


def _activation_rows(arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Flattens each captured activation to 2-D ``[rows, K]``, dropping any
    that has no feature axis at all.

    A MatMul/Gemm activation is ``[..., K]``: 2-D ``[tokens, K]`` for a plain
    MLP, but ``[batch, seq, K]`` for essentially every real transformer, since
    ONNX's MatMul broadcasts over leading dimensions. Every calibration-driven
    pass in this repo wants the same thing from it -- the set of rows that
    multiply ``W`` -- and a layer's reconstruction objective
    ``||W X^T - Ŵ X^T||²`` sums over all of those rows however the leading
    dimensions happen to group them. So collapsing the leading dimensions is
    exact, not an approximation: it is the same set of rows in the same order.

    This exists because filtering to ``ndim == 2`` instead (the previous
    convention here) silently skipped every layer of a ``[batch, seq, hidden]``
    model, i.e. made GPTQ/AWQ and friends no-ops on exactly the models they
    are for, with no diagnostic.
    """
    rows = []
    for a in arrays:
        if a.ndim < 2:
            continue  # no feature axis to multiply W with
        rows.append(a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a)
    return rows


def correct_bias(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    correction_threshold: float = 1e-12,
) -> onnx.ModelProto:
    """Empirically corrects the per-channel output bias
    ``quantized_model``'s Conv/Gemm/MatMul layers picked up from their own
    weight quantization, using real calibration data run through both
    models. See this module's own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), e.g. from :func:`onnxsim.quantize` or
            any ``quantize_*`` function. Assumes ``quantized_model`` was
            produced from ``float_model`` without renaming any
            Conv/Gemm/MatMul node's own output tensor -- true of every
            onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to measure the
            correction on. Each batch is a ``{input_name: np.ndarray}``
            dict matching ``float_model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a much more representative correction than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :param correction_threshold: skip correcting a layer whose measured
            per-channel error never exceeds this (in absolute value) --
            avoids inserting a numerically-pointless zero-offset node for a
            layer ``quantized_model`` left untouched (e.g. a scheme that
            declined to quantize it). Not an accuracy knob: even the
            default, near-zero threshold only filters out true no-ops.
    :returns: ``quantized_model`` with a per-channel correction applied
            after every measurably-biased Conv/Gemm/MatMul layer
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    quantized_outputs: Set[str] = set()
    for n in quantized_model.graph.node:
        quantized_outputs.update(n.output)

    candidates = []  # (output_name, channel_axis)
    for n in float_model.graph.node:
        axis = _CORRECTABLE_OPS.get(n.op_type)
        if axis is None or not n.output or n.output[0] not in quantized_outputs:
            continue
        candidates.append((n.output[0], axis))
    if not candidates:
        return quantized_model

    candidate_names = [name for name, _ in candidates]
    float_probe = _add_probe_outputs(float_model, candidate_names)
    quantized_probe = _add_probe_outputs(quantized_model, candidate_names)

    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    ranks: Dict[str, int] = {}
    for batch in calibration_data:
        float_out = backend.run_model(float_probe, batch, providers=providers)
        quantized_out = backend.run_model(quantized_probe, batch, providers=providers)
        for name, axis in candidates:
            f = np.asarray(float_out[name], dtype=np.float64)
            q = np.asarray(quantized_out[name], dtype=np.float64)
            if f.shape != q.shape or f.ndim == 0:
                continue
            ch_axis = axis if axis >= 0 else f.ndim + axis
            diff = f - q
            reduce_axes = tuple(i for i in range(diff.ndim) if i != ch_axis)
            channel_sum = diff.sum(axis=reduce_axes) if reduce_axes else diff
            channel_count = diff.size // diff.shape[ch_axis]
            if name in sums:
                sums[name] = sums[name] + channel_sum
                counts[name] += channel_count
            else:
                sums[name] = channel_sum
                counts[name] = channel_count
                ranks[name] = f.ndim

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    taken_names = _all_names(corrected.graph)
    axis_by_name = dict(candidates)

    for name, total in sums.items():
        correction = (total / counts[name]).astype(np.float32)
        if np.max(np.abs(correction)) <= correction_threshold:
            continue
        _apply_correction(
            corrected, name, axis_by_name[name], ranks[name], correction, taken_names
        )

    return corrected


def _apply_correction(
    model: onnx.ModelProto,
    output_name: str,
    axis: int,
    rank: int,
    correction: np.ndarray,
    taken_names: Set[str],
) -> None:
    producer_idx = None
    output_index = None
    for idx, n in enumerate(model.graph.node):
        for oi, out in enumerate(n.output):
            if out == output_name:
                producer_idx, output_index = idx, oi
                break
        if producer_idx is not None:
            break
    if producer_idx is None:
        return  # shouldn't happen -- output_name came from this same graph

    pre_correction_name = _unique_name(
        f"{output_name}_bias_correction_pre", taken_names
    )
    model.graph.node[producer_idx].output[output_index] = pre_correction_name

    ch_axis = axis if axis >= 0 else rank + axis
    broadcast_shape = [1] * rank
    broadcast_shape[ch_axis] = correction.shape[0]
    scale_name = _unique_name(f"{output_name}_bias_correction", taken_names)
    scale_tensor = onnx.numpy_helper.from_array(
        correction.reshape(broadcast_shape), name=scale_name
    )
    model.graph.initializer.append(scale_tensor)

    add_node = onnx.helper.make_node(
        "Add",
        [pre_correction_name, scale_name],
        [output_name],
        name=_unique_name(f"{output_name}_bias_correction_add", taken_names),
    )
    model.graph.node.insert(producer_idx + 1, add_node)
