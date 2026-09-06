"""Sensitivity-based mixed-precision weight quantization.

Every quantizer already in onnxsim applies **one uniform scheme to the
whole model**: :func:`onnxsim.quantize_weight_only_int4` block-INT4s every
matched layer, :func:`onnxsim.accuracy.recommend_quantization` searches
across *global* schemes (try INT4-everywhere, then INT8-everywhere, ...)
and returns whichever single one meets the accuracy budget -- but nothing
in onnxsim assigns *different* bit-widths to *different layers* within one
model. That leaves real compression on the table: in any real network,
some layers are far more sensitive to quantization error than others (the
premise behind the mixed-precision/bit-width-search literature -- e.g.
HAQ, Dettmers & Zettlemoyer's LLM.int8() outlier analysis, GPTQ's own
per-layer error reporting), so spending the same number of bits on every
layer either wastes precision on layers that tolerate INT4 fine, or loses
too much on the few layers that don't.

This module is deliberately not a new *algorithm* the way
:mod:`onnxsim.spinquant`/:mod:`onnxsim.duquant` are -- it is a dispatcher
over two schemes onnxsim already has (block-wise INT4 and, for the most
sensitive layers, block-wise INT8), choosing which one each layer gets
from a data-driven **sensitivity score**, then reusing existing
graph-construction machinery for both.

**The sensitivity score.** For a layer with weight ``W`` and calibration
activation ``X``, this asks "how much would this layer's *output* change
if ``W`` were quantized to INT4?" -- not just "how big is the raw
quantization error", which ignores that a layer whose input is typically
tiny barely matters even with a large per-weight error. Concretely:

    mse = mean((W - INT4_dequant(W))^2)     -- INT4's own reconstruction error
    sensitivity = mse * mean(X^2)           -- scaled by typical input magnitude

``mean(X^2)`` is the same per-layer activation-energy signal
:mod:`onnxsim.duquant`'s own sensitivity ranking is built on (there,
per-channel; here, a single per-layer scalar, since the decision being
made -- INT4 vs INT8 -- is per-layer, not per-channel). Layers are ranked
by this score; the top ``high_bits_fraction`` (by count, most sensitive
first) get block-wise INT8 (:func:`onnxsim.quantize_weight_only_int8_block`'s
own granularity, reimplemented locally here since that function quantizes
a whole model uniformly and can't be dispatched per-layer); every other
layer gets ordinary block-wise INT4
(:mod:`onnxsim.omniquant`'s own ``_quantize_blockwise_int4_with_clip``,
the same backend :mod:`onnxsim.spinquant`/:mod:`onnxsim.spqr` already use).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend

# onnxsim.accuracy does not import anything from this module (checked by
# grep), so this is not a cycle -- it is in fact the reverse of
# onnxsim/__init__.py's own import order (`accuracy` is imported well before
# `mixed_precision`), so `onnxsim.accuracy` is already fully initialized in
# `sys.modules` by the time this module is loaded as part of `import onnxsim`.
from onnxsim.accuracy import AccuracyDropReport, measure_accuracy_drop
from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _quantize_blockwise_int8(
    w_nk: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Round-to-nearest block-wise INT8 (``[-127, 127]``) quantization of
    ``w_nk`` ([N, K], output channel first) -- the same granularity
    :func:`onnxsim.quantize_weight_only_int8_block` uses, reimplemented
    here (rather than reused) since that function quantizes a whole model
    uniformly and this module needs to apply it to only some layers.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale_blocks = np.maximum(np.abs(blocks).max(axis=2), 1e-12) / 127.0
    scale_full = np.repeat(scale_blocks, block_size, axis=1)
    codes_nk = np.clip(np.round(w_nk / scale_full), -127.0, 127.0)
    return codes_nk, scale_blocks


def apply_mixed_precision_quantization(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    high_bits_fraction: float = 0.2,
    block_size: int = 32,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight whose reduction dimension ``K`` is divisible by
    ``block_size`` to either block-wise INT8 or block-wise INT4, chosen
    per layer from a calibration-driven sensitivity score -- see this
    module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to measure each layer's own typical activation
            magnitude -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a more representative ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param high_bits_fraction: fraction of matched layers (by count, most
            sensitive first) that get block-wise INT8 instead of INT4;
            ``0.0`` quantizes every layer to INT4 (matching
            :func:`onnxsim.quantize_weight_only_int4`'s own behavior),
            ``1.0`` quantizes every layer to INT8
    :param block_size: elements per quantization block along ``K``, for
            both the INT4 and INT8 tiers -- matching
            :func:`onnxsim.quantize_weight_only_int4`'s own default
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            block-wise INT4 or INT8 codes plus a per-block float32 scale
            (``DequantizeLinear(..., axis=0, block_size=block_size)``);
            output tensor name unchanged. Layers with a non-constant,
            non-2-D weight, a reduction dimension not divisible by
            ``block_size``, or no calibration activation available, are
            left untouched; a model with no matching layer, or an opset
            older than 21 (INT4's tensor type and ``DequantizeLinear``'s
            ``block_size`` attribute both need opset 21), is returned
            unchanged
    """
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
        if w_nk_shape[1] % block_size != 0:
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
    mean_x_sq: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            total = float(np.sum(x**2))
            mean_x_sq[name] = mean_x_sq.get(name, 0.0) + total
            counts[name] = counts.get(name, 0) + x.size

    for name in list(mean_x_sq):
        if counts[name] > 0:
            mean_x_sq[name] /= counts[name]

    # Sensitivity per candidate: how much INT4 quantization error this
    # layer's weight would have, scaled by that layer's own typical
    # (calibration) input magnitude -- see module docstring.
    sensitivities: List[Optional[float]] = []
    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        if x_name not in mean_x_sq:
            sensitivities.append(None)
            continue
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K]
        codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
            w_nk, block_size, 1.0
        )
        scale_full = np.repeat(scale_blocks_nk, block_size, axis=1)
        dequant_nk = codes_nk * scale_full
        mse = float(np.mean((w_nk - dequant_nk) ** 2))
        sensitivities.append(mse * mean_x_sq[x_name])

    eligible_idx = [i for i, s in enumerate(sensitivities) if s is not None]
    eligible_idx.sort(key=lambda i: sensitivities[i] or 0.0, reverse=True)
    num_high_bits = int(round(high_bits_fraction * len(eligible_idx)))
    high_bits_set = set(eligible_idx[:num_high_bits])

    for idx, (node, x_name, w_name, bias_name, weight_transposed) in enumerate(
        candidates
    ):
        if idx not in eligible_idx:
            continue

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape

        use_int8 = idx in high_bits_set
        if use_int8:
            codes_nk, scale_blocks_nk = _quantize_blockwise_int8(w_nk, block_size)
            codes_dtype = onnx.TensorProto.INT8
            prefix = f"{w_name}_mixedprec_int8"
        else:
            codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
                w_nk, block_size, 1.0
            )
            codes_dtype = onnx.TensorProto.INT4
            prefix = f"{w_name}_mixedprec_int4"

        codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        codes_tensor = onnx.TensorProto()
        codes_tensor.name = codes_name
        codes_tensor.data_type = codes_dtype
        codes_tensor.dims.extend([k, n])
        if codes_dtype == onnx.TensorProto.INT4:
            codes_tensor.raw_data = _pack_int4(codes_kn)
        else:
            codes_tensor.raw_data = codes_kn.astype(np.int8).tobytes()
        graph.initializer.append(codes_tensor)

        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_kn, name=scale_name)
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
            block_size=block_size,
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


DEFAULT_SEARCH_FRACTIONS: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
"""Default ``fractions`` sweep for :func:`search_mixed_precision_for_budget`,
front-loaded towards small ``high_bits_fraction`` values (dense near 0,
sparse near 1). Most of the compression benefit of mixed precision comes
from keeping ``high_bits_fraction`` small -- only the few outlier-sensitive
layers actually need INT8 -- so most models that can meet a reasonable
accuracy budget at all will meet it at one of these early, small fractions.
Trying small fractions first lets the search stop (see
:func:`search_mixed_precision_for_budget`'s early-stopping) after as few
:func:`onnxsim.accuracy.measure_accuracy_drop` calls as possible, since each
one re-runs both the float and quantized model on every calibration sample.
"""


@dataclass
class MixedPrecisionSearchResult:
    """One :func:`search_mixed_precision_for_budget` result: the winning (or,
    if none met the budget, least-lossy -- i.e. highest ``high_bits_fraction``
    -- one tried) fraction, its measured accuracy drop, and the model already
    quantized with it. Mirrors :class:`onnxsim.accuracy.QuantizationRecommendation`'s
    shape, adapted to a search over ``high_bits_fraction`` within the mixed-
    precision scheme rather than over whole quantization schemes.
    """

    high_bits_fraction: float
    report: AccuracyDropReport
    quantized_model: onnx.ModelProto
    meets_budget: bool
    fractions_tried: List[float]


def search_mixed_precision_for_budget(
    model: Union[str, onnx.ModelProto],
    accuracy_budget: float = 0.02,
    fractions: Optional[Sequence[float]] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    providers: Optional[Sequence[str]] = None,
) -> MixedPrecisionSearchResult:
    """Accuracy-aware search over :func:`apply_mixed_precision_quantization`'s
    own ``high_bits_fraction`` parameter: tries ``fractions`` in order
    (default :data:`DEFAULT_SEARCH_FRACTIONS`, smallest -- most compressed --
    first), stopping as soon as one fraction's measured accuracy drop
    (:func:`onnxsim.accuracy.measure_accuracy_drop`) meets ``accuracy_budget``.

    This is the "iterate until target met" control loop described as the
    missing piece over :func:`apply_mixed_precision_quantization`'s own
    single-shot per-layer sensitivity dispatcher -- promoting more of the
    most-sensitive layers to INT8 (by trying larger and larger
    ``high_bits_fraction`` values) until the model's *actual measured*
    accuracy drop, not just its estimated per-layer sensitivity score, is
    under budget. It is not a replacement for
    :func:`onnxsim.accuracy.recommend_quantization`, which searches across
    whole quantization *schemes* rather than per-layer bit-width assignment
    within this one scheme.

    ``calibration_data`` is generated once (if not supplied) and reused,
    unchanged, for every fraction tried -- both to quantize
    (:func:`apply_mixed_precision_quantization`'s own sensitivity ranking)
    and to measure (:func:`onnxsim.accuracy.measure_accuracy_drop`). Measuring
    every fraction against the same data is what makes their accuracy drops
    comparable at all; regenerating fresh random data per fraction would make
    each measurement a comparison against different noise.

    If ``model`` has no eligible mixed-precision candidate,
    :func:`apply_mixed_precision_quantization` returns it unchanged
    regardless of ``high_bits_fraction`` -- every fraction would measure the
    same (no-op) accuracy drop, so this doesn't special-case that: it simply
    tries ``fractions`` in order as usual and stops at ``fractions[0]``
    (typically meeting even a tight budget, since nothing was quantized) or
    proceeds through the full list if ``fractions[0]`` itself doesn't meet
    budget for some other reason (e.g. an already-embedded quantization
    error in ``model``, or an unreasonably tight ``accuracy_budget``).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param accuracy_budget: maximum acceptable worst-case relative L2 error
            (see :attr:`onnxsim.accuracy.AccuracyDropReport.worst_relative_l2`)
            for a fraction to be accepted
    :param fractions: ``high_bits_fraction`` values to try, in the order
            given -- defaults to :data:`DEFAULT_SEARCH_FRACTIONS`. Must be
            non-empty. Not required to be sorted, but since the search stops
            at the first fraction that meets budget, an order other than
            increasing-fraction defeats the "stop as early as possible on the
            most-compressed option" intent
    :param calibration_data: representative input batches, used both to rank
            per-layer sensitivity (:func:`apply_mixed_precision_quantization`)
            and to measure accuracy drop
            (:func:`onnxsim.accuracy.measure_accuracy_drop`) at every
            fraction tried. See :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            more representative search than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block, forwarded to
            :func:`apply_mixed_precision_quantization` for every fraction
            tried
    :param providers: onnxruntime execution providers to calibrate/measure on
    :returns: the winning (or, if none met budget, last-tried) fraction's
            result. ``result.fractions_tried`` lists every fraction actually
            measured, in the order tried, so a caller can tell e.g. that only
            ``fractions[0]`` was needed
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if fractions is None:
        fractions = DEFAULT_SEARCH_FRACTIONS
    if not fractions:
        raise ValueError("`fractions` must be non-empty")
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    fractions_tried: List[float] = []
    result: Optional[MixedPrecisionSearchResult] = None
    for frac in fractions:
        fractions_tried.append(frac)
        quantized = apply_mixed_precision_quantization(
            model,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            high_bits_fraction=frac,
            block_size=block_size,
            providers=providers,
        )
        report = measure_accuracy_drop(
            model,
            quantized,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            providers=providers,
        )
        meets_budget = report.all_finite and report.worst_relative_l2 < accuracy_budget
        result = MixedPrecisionSearchResult(
            high_bits_fraction=frac,
            report=report,
            quantized_model=quantized,
            meets_budget=meets_budget,
            fractions_tried=list(fractions_tried),
        )
        if meets_budget:
            return result

    assert result is not None  # `fractions` was checked non-empty above
    return result
