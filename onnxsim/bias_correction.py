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

The same measurement applies to any output-preserving *algorithm* swap, not
only quantization: e.g. a deployment target whose Resize kernel doesn't
implement every interpolation mode, so onnxsim (or a downstream converter)
rewrites a model's Resize node to a mode the accelerator does support. That
rewrite is exact for some inputs and systematically off for others in a way
that, like quantization rounding, tends to have a nonzero mean per channel
-- ``correct_bias`` treats it identically to a quantized Conv/Gemm/MatMul's
bias shift, measuring and cancelling it the same way. It does not (and
cannot) recover a mode swap's non-systematic, input-dependent error --
that would need real fine-tuning of downstream weights, not a constant
offset.

For every Conv/Gemm/MatMul/Resize node present (by output tensor name) in
both ``float_model`` and ``quantized_model``, this runs both models on the
same calibration data, measures each such layer's own per-output-channel
mean error (``float_output - quantized_output``, independent of any other
layer's correction -- not chained through progressively-corrected
activations), and adds that as a constant per-channel offset right after
the layer in ``quantized_model``. Adding a constant to an affine layer's
output is exactly equivalent to folding it into that layer's bias, whatever
internal shape the ``quantize_*`` scheme that produced it happens to use
(a straight Conv/Gemm/MatMul weight-only rewrite, a multi-node dynamic-
quantization chain, a Resize mode substitution, ...) -- so this needs no
scheme-specific knowledge of where a bias tensor lives internally, only
that the layer's own output tensor kept its original name, which every
onnxsim ``quantize_*`` pass (and any Resize-attribute rewrite that edits
the node in place) guarantees (downstream consumers are never rewired by
name).

A geometric change to Resize (a different ``coordinate_transformation_mode``,
or ``mode`` itself) mostly does *not* fit this model, though: the resampling
error at a given output pixel depends on the local gradient of whatever the
input happens to be at that spatial location, which is roughly as likely to
push the value up as down. Averaged per channel over many differently-
content calibration images, that error washes out to ~0 -- there is no
"channel bias" for :func:`correct_bias` to find, however much calibration
data it is given. What *does* survive averaging is the part of the error
that is consistent by *position* rather than by content: deployments where
the calibration images share spatial structure (a fixed-mount camera, a
consistent framing/crop, a scene layout that recurs across samples --
common in e.g. ADAS/robotics perception pipelines) see the same
misalignment at the same output pixel across samples, which a per-channel
constant still cannot represent but a per-*position* map can.
:func:`correct_spatial_bias` measures exactly that: a coarse per-channel
grid of position-wise mean error, smoothed back up to full resolution. Since
that only helps when the calibration data actually has this shared spatial
structure -- and can otherwise make things measurably worse by fitting
position-wise noise -- it holds out part of the calibration data and only
applies the correction where it verifiably reduces held-out error.

Computational cost: both functions only ever run *forward* inference --
``num_samples`` (default 8 for :func:`correct_bias`, 16 for
:func:`correct_spatial_bias`) forward passes through each of the two
models, no backward pass and no weight updates, so the total cost is
~2x``num_samples`` ordinary inferences plus O(output size) numpy reductions
(negligible next to that). That is orders of magnitude cheaper than actual
fine-tuning (e.g. the LoRA/distillation path in ``tools/onnx-finetune``),
which needs a backward pass and an optimizer step per batch, repeated over
multiple epochs, and a training-capable ONNX Runtime build -- these
functions need neither gradients nor a training build, only whatever
inference backend :mod:`onnxsim.backend` already uses. The tradeoff is
exactly the one documented above: this cheap path only recovers a
systematic (mean, or spatially-consistent) shift, not the full error a
real algorithm change can introduce.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx

from onnxsim import backend
from onnxsim.calibration import Tensors, generate_random_calibration_data

# op_type -> the axis its own output tensor's "channel" dimension sits on.
# Conv and Resize are always NCHW-style (batch, channel, spatial...), so
# channel is axis 1 regardless of spatial rank (this matches the same
# assumption structured_pruning's MatchResizeChannelPassThrough makes about
# Resize's layout). Gemm/MatMul put the output feature dimension last
# regardless of transpose attributes (Gemm's output shape is always [M, N]
# whatever transA/transB are; MatMul has no transpose attributes at all),
# so channel is axis -1.
_CORRECTABLE_OPS: Dict[str, int] = {
    "Conv": 1,
    "Gemm": -1,
    "MatMul": -1,
    "Resize": 1,
}


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
    ``quantized_model``'s Conv/Gemm/MatMul/Resize layers picked up from
    their own weight quantization (or, for Resize, from an algorithm/mode
    change), using real calibration data run through both models. See this
    module's own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a modified version of ``float_model`` (onnx
            ModelProto or file path) whose Conv/Gemm/MatMul/Resize layers
            keep their original output tensor names -- e.g. a quantized
            model from :func:`onnxsim.quantize` or any ``quantize_*``
            function, or a model with a Resize node's ``mode``/
            ``coordinate_transformation_mode`` swapped for one an
            accelerator supports. Assumes ``quantized_model`` was produced
            from ``float_model`` without renaming any candidate node's own
            output tensor -- true of every onnxsim ``quantize_*`` function
            and of any in-place Resize-attribute rewrite.
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


def _splice_add_correction(
    model: onnx.ModelProto,
    output_name: str,
    correction: np.ndarray,
    name_prefix: str,
    taken_names: Set[str],
) -> None:
    """Renames ``output_name``'s producer to a fresh internal name, then
    reinstates ``output_name`` as an ``Add`` of that renamed value with
    ``correction`` (already broadcastable to the output's shape) -- so
    ``output_name`` keeps resolving to the *corrected* value for every
    downstream consumer and graph output alike.
    """
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

    pre_correction_name = _unique_name(f"{output_name}_{name_prefix}_pre", taken_names)
    model.graph.node[producer_idx].output[output_index] = pre_correction_name

    scale_name = _unique_name(f"{output_name}_{name_prefix}", taken_names)
    scale_tensor = onnx.numpy_helper.from_array(correction, name=scale_name)
    model.graph.initializer.append(scale_tensor)

    add_node = onnx.helper.make_node(
        "Add",
        [pre_correction_name, scale_name],
        [output_name],
        name=_unique_name(f"{output_name}_{name_prefix}_add", taken_names),
    )
    model.graph.node.insert(producer_idx + 1, add_node)


def _apply_correction(
    model: onnx.ModelProto,
    output_name: str,
    axis: int,
    rank: int,
    correction: np.ndarray,
    taken_names: Set[str],
) -> None:
    ch_axis = axis if axis >= 0 else rank + axis
    broadcast_shape = [1] * rank
    broadcast_shape[ch_axis] = correction.shape[0]
    _splice_add_correction(
        model,
        output_name,
        correction.reshape(broadcast_shape),
        "bias_correction",
        taken_names,
    )


def _block_average(array: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Downsamples a ``[C, H, W]`` array to ``[C, grid_h, grid_w]`` by
    averaging each of its (possibly unevenly sized, when ``H``/``W`` doesn't
    divide evenly) rectangular blocks -- the coarse grid a spatial
    correction is fit on, before being smoothed back up to full resolution.
    """
    row_blocks = np.array_split(np.arange(array.shape[1]), grid_h)
    col_blocks = np.array_split(np.arange(array.shape[2]), grid_w)
    coarse = np.empty((array.shape[0], grid_h, grid_w), dtype=array.dtype)
    for i, rows in enumerate(row_blocks):
        for j, cols in enumerate(col_blocks):
            coarse[:, i, j] = array[:, rows][:, :, cols].mean(axis=(1, 2))
    return coarse


def _bilinear_upsample(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsamples a ``[C, gh, gw]`` grid to ``[C, out_h, out_w]`` by bilinear
    interpolation between grid-cell centers (edge cells extend flat past
    their center, i.e. clamp-to-edge) -- turns a coarse, low-frequency
    correction grid into a smooth full-resolution correction map with no
    blockiness at the grid boundaries.
    """
    _, grid_h, grid_w = grid.shape
    if grid_h == out_h and grid_w == out_w:
        return grid

    def _interp_coords(out_n: int, grid_n: int) -> tuple:
        centers = (np.arange(out_n) + 0.5) * grid_n / out_n - 0.5
        centers = np.clip(centers, 0, grid_n - 1)
        lo = np.floor(centers).astype(np.int64)
        hi = np.clip(lo + 1, 0, grid_n - 1)
        weight = centers - lo
        return lo, hi, weight

    y0, y1, wy = _interp_coords(out_h, grid_h)
    x0, x1, wx = _interp_coords(out_w, grid_w)
    wy = wy[np.newaxis, :, np.newaxis]
    wx = wx[np.newaxis, np.newaxis, :]

    top = grid[:, y0][:, :, x0] * (1 - wx) + grid[:, y0][:, :, x1] * wx
    bottom = grid[:, y1][:, :, x0] * (1 - wx) + grid[:, y1][:, :, x1] * wx
    return top * (1 - wy) + bottom * wy


def correct_spatial_bias(
    float_model: Union[str, onnx.ModelProto],
    modified_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 16,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    grid_size: int = 8,
    validation_fraction: float = 0.3,
) -> onnx.ModelProto:
    """Empirically corrects the per-*position* output bias a Conv/Resize
    layer in ``modified_model`` picked up from a spatially-structured
    algorithm change (e.g. swapping a Resize node's ``mode`` or
    ``coordinate_transformation_mode`` for one a deployment target
    supports), when :func:`correct_bias`'s per-channel constant can't
    represent it. See this module's own docstring for when this actually
    helps versus when it's a no-op by construction.

    Unlike :func:`correct_bias`, this only ever applies a correction it has
    verified helps: ``calibration_data`` is split into a fit portion (used
    to measure the correction) and a held-out validation portion, and a
    layer's correction is applied only if it measurably reduces error on
    the held-out portion. On calibration data with no shared spatial
    structure across samples, every correction is expected to fail that
    check and this is a no-op -- not "an accuracy knob left at a
    conservative default", but the actual expected behavior, since a
    correction that only fit noise would otherwise make held-out predictions
    worse, not better.

    :param float_model: the original onnx ModelProto or file path
    :param modified_model: a modified version of ``float_model`` (onnx
            ModelProto or file path) whose Conv/Resize layers keep their
            original output tensor names -- see :func:`correct_bias`'s
            ``quantized_model`` parameter, which this mirrors
    :param calibration_data: representative input batches, ideally sharing
            the deployment's actual spatial structure (e.g. real frames
            from the same camera/mount, not independent random images) --
            see :func:`correct_bias`
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted (note: random per-sample noise
            has no shared spatial structure, so this default is only useful
            for exercising the code path, not for a real correction)
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :param grid_size: side length of the coarse per-channel grid fit before
            smoothing back up to full resolution -- larger captures finer
            spatial patterns but needs more calibration data to validate
            reliably; clamped to the layer's actual output height/width
    :param validation_fraction: fraction of ``calibration_data`` held out
            to validate each layer's correction; the rest is used to fit it
    :returns: ``modified_model`` with a per-position correction applied
            after every Conv/Resize layer where one measurably reduces
            held-out error
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(modified_model, str):
        modified_model = onnx.load(modified_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    calibration_data = list(calibration_data)
    if len(calibration_data) < 2:
        return modified_model  # can't hold out a validation split

    n_val = min(
        max(1, round(len(calibration_data) * validation_fraction)),
        len(calibration_data) - 1,
    )
    fit_data, val_data = calibration_data[:-n_val], calibration_data[-n_val:]

    modified_outputs: Set[str] = set()
    for n in modified_model.graph.node:
        modified_outputs.update(n.output)

    candidates = [
        n.output[0]
        for n in float_model.graph.node
        if _CORRECTABLE_OPS.get(n.op_type) == 1
        and n.output
        and n.output[0] in modified_outputs
    ]
    if not candidates:
        return modified_model

    float_probe = _add_probe_outputs(float_model, candidates)
    modified_probe = _add_probe_outputs(modified_model, candidates)

    def _per_position_errors(data: Sequence[Tensors]) -> Dict[str, np.ndarray]:
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for batch in data:
            float_out = backend.run_model(float_probe, batch, providers=providers)
            modified_out = backend.run_model(modified_probe, batch, providers=providers)
            for name in candidates:
                f = np.asarray(float_out[name], dtype=np.float64)
                q = np.asarray(modified_out[name], dtype=np.float64)
                if f.shape != q.shape or f.ndim != 4:
                    continue  # only NCHW-style, image-shaped outputs have a grid to fit
                batch_sum = (f - q).sum(axis=0)  # [C, H, W]
                if name in sums:
                    sums[name] += batch_sum
                    counts[name] += f.shape[0]
                else:
                    sums[name] = batch_sum
                    counts[name] = f.shape[0]
        return {name: total / counts[name] for name, total in sums.items()}

    fit_mean_error = _per_position_errors(fit_data)
    if not fit_mean_error:
        return modified_model

    corrections: Dict[str, np.ndarray] = {}
    for name, mean_error in fit_mean_error.items():
        _, h, w = mean_error.shape
        gh, gw = min(grid_size, h), min(grid_size, w)
        coarse = _block_average(mean_error, gh, gw)
        corrections[name] = _bilinear_upsample(coarse, h, w).astype(np.float32)

    sq_error_before: Dict[str, float] = {name: 0.0 for name in corrections}
    sq_error_after: Dict[str, float] = {name: 0.0 for name in corrections}
    for batch in val_data:
        float_out = backend.run_model(float_probe, batch, providers=providers)
        modified_out = backend.run_model(modified_probe, batch, providers=providers)
        for name, correction in corrections.items():
            f = np.asarray(float_out[name], dtype=np.float64)
            q = np.asarray(modified_out[name], dtype=np.float64)
            if f.shape != q.shape or f.ndim != 4:
                continue
            sq_error_before[name] += float(np.sum((f - q) ** 2))
            sq_error_after[name] += float(np.sum((f - (q + correction)) ** 2))

    accepted = {
        name: correction
        for name, correction in corrections.items()
        if sq_error_after[name] < sq_error_before[name]
    }
    if not accepted:
        return modified_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(modified_model)
    taken_names = _all_names(corrected.graph)
    for name, correction in accepted.items():
        _splice_add_correction(
            corrected, name, correction[np.newaxis], "spatial_correction", taken_names
        )
    return corrected
