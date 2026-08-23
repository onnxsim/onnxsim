"""Calibration data generation and static (calibration-based) quantization.

``onnxsim.quantize_dynamic`` needs no calibration data: it defers the
activation's quantization range to a ``DynamicQuantizeLinear`` computed fresh
on every inference. Static quantization instead *fixes* that range ahead of
time, from representative data -- usually a better trade (no per-inference
range computation) but only as good as the calibration data it is given.

This module provides two calibration data sources, meant to be used in order
as a model moves from a quick smoke test towards real deployment:

- :func:`generate_random_calibration_data` -- synthetic random data. Works
  out of the box with no external dependency or dataset to find, so
  :func:`quantize_static` falls back to it automatically. Good for checking
  the quantization pipeline itself works; a poor proxy for a real model's
  actual activation statistics.
- :func:`load_huggingface_calibration_data` -- real examples pulled from a
  Hugging Face Hub dataset (needs the optional ``datasets`` package). Gives
  calibration ranges that actually reflect deployment-time data, at the cost
  of needing a dataset whose columns can be matched to the model's inputs.

:func:`calibrate` runs the float model over either data source through
ONNX Runtime to produce the ``{tensor_name: (min, max)}`` ranges
:func:`onnxsim.quantize_static` (this module's main entry point) needs.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

import onnxsim.onnxsim_cpp2py_export as C

Tensors = Dict[str, np.ndarray]

_ELEM_TYPE_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.BOOL: np.bool_,
}


def _input_specs(model: onnx.ModelProto) -> List[Tuple[str, List[int], type]]:
    """(name, shape, np_dtype) for every graph input that is not an initializer.

    Dynamic dimensions (including an unset/symbolic batch dimension) are
    fixed to 1, so the returned shapes are always fully concrete.
    """
    initializer_names = {i.name for i in model.graph.initializer}
    specs = []
    for ipt in model.graph.input:
        if ipt.name in initializer_names:
            continue
        shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in ipt.type.tensor_type.shape.dim
        ]
        np_dtype = _ELEM_TYPE_TO_NP.get(ipt.type.tensor_type.elem_type, np.float32)
        specs.append((ipt.name, shape, np_dtype))
    return specs


def generate_random_calibration_data(
    model: Union[str, onnx.ModelProto],
    num_samples: int = 8,
    seed: int = 0,
) -> List[Tensors]:
    """
    Generate ``num_samples`` batches of random input data matching ``model``'s
    input shapes/dtypes: a calibration data source that works with no
    external dependency, for a first pass through the static quantization
    pipeline before wiring up real representative data (e.g.
    :func:`load_huggingface_calibration_data`).

    Floating-point inputs are drawn from a standard normal distribution (a
    closer proxy for typical activation/feature statistics than a uniform
    [0, 1) draw); integer and boolean inputs are filled with zeros, a safe
    default when random values are unlikely to be valid indices (e.g. token
    ids) -- pass your own calibration data for a model whose behavior
    actually depends on integer input values.

    :param model: onnx ModelProto object or file path
    :param num_samples: number of calibration batches to generate
    :param seed: seed for reproducibility
    :returns: a list of ``{input_name: np.ndarray}`` dicts, one per batch
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    specs = _input_specs(model)
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        batch = {}
        for name, shape, np_dtype in specs:
            if np.issubdtype(np_dtype, np.floating):
                batch[name] = rng.standard_normal(shape).astype(np_dtype)
            else:
                batch[name] = np.zeros(shape, dtype=np_dtype)
        batches.append(batch)
    return batches


# A short list of common Hugging Face dataset column names for each of a few
# common ONNX input names, tried (after `field_map` and an exact name match)
# by load_huggingface_calibration_data before giving up on an input.
_COMMON_COLUMN_ALIASES = {
    "pixel_values": ["pixel_values", "image", "img"],
    "input_ids": ["input_ids", "input_id", "tokens", "text"],
    "attention_mask": ["attention_mask", "mask"],
    "input_values": ["input_values", "audio", "speech"],
}


def load_huggingface_calibration_data(
    dataset: str,
    model: Union[str, onnx.ModelProto],
    num_samples: int = 8,
    split: str = "train",
    field_map: Optional[Dict[str, str]] = None,
    seed: int = 0,
) -> List[Tensors]:
    """
    Best-effort loader that pulls ``num_samples`` real examples from a
    Hugging Face ``datasets`` dataset and adapts them into calibration
    batches for ``model``. Requires the optional ``datasets`` package
    (``pip install datasets``).

    Matching a dataset's columns to a model's ONNX input names is inherently
    ambiguous -- there is no schema linking the two -- so this only handles
    the common case. For each input, in order: an explicit ``field_map``
    entry, a same-named column, then a short list of common aliases (e.g.
    ONNX input "pixel_values" matches a dataset column named "image"). If an
    input still cannot be matched, this raises ``ValueError`` naming it
    rather than silently feeding it random or zero data -- pass ``field_map``
    to resolve the mismatch, or fall back to
    :func:`generate_random_calibration_data` (or your own preprocessing
    pipeline, feeding its output directly to :func:`quantize_static` /
    :func:`calibrate` as ``calibration_data``) instead of this loader.

    A matched column's values are converted to a numpy array and cast to the
    target input's dtype; this only handles a column that is already a plain
    (optionally ragged, single-example) tensor, so a variable-length sequence
    column (e.g. un-padded tokenized text) needs its own tokenizer/padding
    step before it can be used here.

    :param dataset: a Hugging Face Hub dataset id, e.g. "mnist" or "cifar10"
    :param model: onnx ModelProto object or file path
    :param num_samples: number of examples to pull from the dataset
    :param split: dataset split to sample from
    :param field_map: optional explicit ``{onnx_input_name: dataset_column_name}``
            overrides, for a dataset whose columns don't already match by
            name or common alias
    :param seed: seed used to shuffle the dataset before sampling
    :returns: a list of ``{input_name: np.ndarray}`` dicts, one per example
    """
    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ImportError(
            "load_huggingface_calibration_data needs the optional 'datasets' "
            "package: pip install datasets"
        ) from e

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    specs = _input_specs(model)

    ds = hf_datasets.load_dataset(dataset, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=max(num_samples * 4, 64))
    examples = list(itertools.islice(ds, num_samples))
    if not examples:
        raise ValueError(f'Dataset "{dataset}" split "{split}" yielded no examples.')
    columns = set(examples[0].keys())

    resolved: Dict[str, str] = {}
    unmatched: List[str] = []
    for name, _shape, _np_dtype in specs:
        if field_map and name in field_map:
            resolved[name] = field_map[name]
            continue
        candidates = [name] + _COMMON_COLUMN_ALIASES.get(name, [])
        match = next((c for c in candidates if c in columns), None)
        if match is None:
            match = next((c for c in columns if c.lower() == name.lower()), None)
        if match is None:
            unmatched.append(name)
        else:
            resolved[name] = match

    if unmatched:
        raise ValueError(
            'Could not match onnx input(s) {} to any column of dataset "{}" '
            "(columns: {}). Pass field_map={{onnx_input_name: "
            "dataset_column_name}} to resolve manually.".format(
                unmatched, dataset, sorted(columns)
            )
        )

    batches = []
    for example in examples:
        batch = {}
        for name, shape, np_dtype in specs:
            arr = np.asarray(example[resolved[name]], dtype=np_dtype)
            if arr.shape != tuple(shape):
                if arr.size != int(np.prod(shape)):
                    raise ValueError(
                        f'Dataset "{dataset}" column "{resolved[name]}" has '
                        f"{arr.size} elements per example, which does not "
                        f'match onnx input "{name}"\'s shape {shape} '
                        f"({int(np.prod(shape))} elements). Pass your own "
                        "preprocessing pipeline's output as calibration_data "
                        "instead."
                    )
                arr = arr.reshape(shape)
            batch[name] = arr
        batches.append(batch)
    return batches


def _smooth_distribution(p: np.ndarray, eps: float = 1e-4) -> Optional[np.ndarray]:
    """
    Replace zero entries in histogram ``p`` with a small ``eps``, taking the
    total added back out of the non-zero entries proportionally, so a KL
    divergence computed against the result stays finite (``p * log(p / 0)``
    would otherwise be ``-inf``). Standard technique for entropy calibration
    (e.g. MXNet's and TensorRT's own KL calibrators use the same smoothing).

    Returns ``None`` -- rather than a negative-count histogram -- when a
    non-zero bin is smaller than the epsilon it would need to give up (only
    possible when zero bins vastly outnumber non-zero ones); the caller skips
    that candidate threshold.
    """
    p = p.astype(np.float64)
    is_zeros = p == 0
    n_zeros = int(is_zeros.sum())
    if n_zeros == 0:
        return p
    n_nonzeros = p.size - n_zeros
    if n_nonzeros == 0:
        return None
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    hist = p.copy()
    hist[~is_zeros] -= eps1
    hist[is_zeros] = eps
    if (hist[~is_zeros] < 0).any():
        return None
    return hist


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) for two same-length, already-normalized-or-not histograms
    (only their relative proportions matter -- see how each is built below).
    Bins where P is zero contribute nothing (the usual ``0 * log(0/q) = 0``
    convention)."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _entropy_threshold(
    values: np.ndarray,
    num_bins: int = 2048,
    num_quantized_bins: int = 128,
    min_coverage: float = 0.999,
) -> float:
    """
    Find the symmetric clip threshold ``T`` minimizing the KL divergence
    between ``values``' distribution and its simulated INT8-quantized one
    after clipping to ``[-T, T]`` -- entropy ("KL-divergence") calibration,
    as TensorRT popularized it (Migacz, "8-bit Inference with TensorRT",
    2017). Works on ``|values|`` throughout, since a single shared threshold
    only makes sense as a magnitude clip; :func:`calibrate` intersects
    ``[-T, T]`` with the tensor's observed ``(min, max)`` afterwards to get
    the final (possibly asymmetric) range.

    The search: build a fine-grained (``num_bins``) histogram of
    ``|values|``, then for every candidate cutoff ``i`` from the
    ``min_coverage`` percentile's bin up to ``num_bins``, compare the
    reference distribution (the histogram clipped at ``i``, with everything
    beyond folded into the last bin) against a simulated quantization of that
    same clipped range down to ``num_quantized_bins`` levels and back. The
    cutoff with the lowest KL divergence is the threshold that loses the
    least distributional information by quantizing -- often tighter than the
    raw max for a heavy-tailed distribution, where a handful of outliers
    would otherwise stretch the whole range and waste quantization levels on
    rarely-hit values.

    ``min_coverage`` bounds the search from below at that percentile of
    ``|values|`` (default: the top 0.1% may be clipped, no more). Without it,
    the search can pick a pathologically small threshold on data that is
    already smooth (no long tail to reward clipping against, e.g. a raw
    Gaussian): quantizing *any* narrow slice of a locally-flat distribution
    reproduces its shape almost exactly, so KL divergence stays near zero for
    every threshold and the search has nothing to distinguish them by,
    including catastrophically aggressive ones. Real, mostly-heavy-tailed
    activation distributions have plenty of room below this floor for the
    search to still find a genuinely tighter-than-max threshold in.

    Falls back to the observed max (i.e. no clipping) when there is too
    little data, or too little dynamic range, to build a meaningful
    histogram.
    """
    abs_values = np.abs(values.astype(np.float64)).ravel()
    abs_values = abs_values[np.isfinite(abs_values)]
    if abs_values.size == 0:
        return 0.0
    abs_max = float(abs_values.max())
    if abs_max <= 0.0 or abs_values.size < num_quantized_bins:
        return abs_max

    hist, bin_edges = np.histogram(abs_values, bins=num_bins, range=(0.0, abs_max))
    hist = hist.astype(np.float64)

    coverage_floor = float(np.percentile(abs_values, min_coverage * 100.0))
    # bin_edges[i] is the upper edge of the i-th bin (0-indexed), so the first
    # cutoff whose upper edge reaches the floor is searchsorted's insertion
    # point; clamped into [num_quantized_bins, num_bins] either end.
    i_start = int(
        np.clip(
            np.searchsorted(bin_edges, coverage_floor), num_quantized_bins, num_bins
        )
    )

    best_threshold = abs_max
    best_divergence = float("inf")
    for i in range(i_start, num_bins + 1):
        ref_dist = hist[:i].copy()
        # Clipped, not dropped: fold the tail's count into the last
        # reference bin so `ref_dist` still sums to the full sample count.
        ref_dist[-1] += hist[i:].sum()
        if ref_dist.sum() == 0:
            continue

        # Simulate quantizing the clipped range to num_quantized_bins levels:
        # merge ref_dist's `i` fine bins into num_quantized_bins groups, then
        # spread each group's total back out evenly over its own non-empty
        # fine bins, so the simulated distribution has the same length (`i`)
        # as ref_dist and the two are directly comparable bin-for-bin.
        groups = np.array_split(np.arange(i), num_quantized_bins)
        candidate = np.zeros(i, dtype=np.float64)
        for idxs in groups:
            nonzero = ref_dist[idxs] > 0
            count = int(nonzero.sum())
            if count == 0:
                continue
            candidate[idxs[nonzero]] = ref_dist[idxs].sum() / count

        p = _smooth_distribution(ref_dist)
        q = _smooth_distribution(candidate)
        if p is None or q is None:
            continue
        divergence = _kl_divergence(p, q)
        if divergence < best_divergence:
            best_divergence = divergence
            best_threshold = float(bin_edges[i])

    return best_threshold


def calibrate(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
    num_bins: int = 2048,
    num_quantized_bins: int = 128,
    extra_tensor_names: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Run the float ``model`` over every batch in ``calibration_data`` through
    ONNX Runtime, recording each quantizable activation's calibration range
    across all batches -- the calibration ranges :func:`onnxsim.quantize_static`
    needs.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches, e.g. from
            :func:`generate_random_calibration_data` or
            :func:`load_huggingface_calibration_data`
    :param providers: onnxruntime execution providers to run calibration on
            (defaults to onnxruntime's own default provider selection)
    :param extra_tensor_names: additional tensor names to calibrate on top of
            ``onnxsim_cpp2py_export.list_quantizable_activations``' own list --
            :func:`quantize_qoperator` passes its *output* tensor names here,
            since QOperator format needs a calibrated range for a quantized
            node's output too, not just its activation (see
            ``onnxsim_cpp2py_export.list_qoperator_quantizable_outputs``).
    :param method: ``"minmax"`` (default) uses each tensor's observed
            ``(min, max)`` directly -- simple, and enough calibration data to
            cover the real range is all it needs. ``"entropy"`` instead finds,
            per tensor, the symmetric clip threshold minimizing the KL
            divergence between the observed distribution and its simulated
            INT8-quantized one (see :func:`_entropy_threshold`), then
            intersects that clip with the observed ``(min, max)``. This can
            give a tighter, better-behaved range than min/max alone when a
            handful of outlier activations would otherwise stretch the whole
            range and starve the common values of quantization levels -- the
            same trade TensorRT's entropy calibrator makes. It needs
            noticeably more calibration data than ``"minmax"`` to build a
            meaningful per-tensor histogram (a couple of batches suffices for
            min/max; entropy search wants at least ``num_quantized_bins``
            observed values per tensor, and is more reliable with hundreds).
    :param num_bins: (``"entropy"`` only) histogram resolution the threshold
            search scans over -- see :func:`_entropy_threshold`.
    :param num_quantized_bins: (``"entropy"`` only) number of levels the
            search simulates quantizing down to -- see
            :func:`_entropy_threshold`. Left at INT8's 128 (one sign's worth)
            regardless of onnxsim's own uint8 range, matching the standard
            entropy-calibration convention this implements.
    :returns: ``{tensor_name: (min, max)}`` for every tensor
            ``onnxsim_cpp2py_export.list_quantizable_activations`` reports,
            plus ``extra_tensor_names`` if given
    """
    import onnxruntime as ort

    if method not in ("minmax", "entropy"):
        raise ValueError(f"unknown calibration method: {method!r}")

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    model_bytes = model.SerializeToString()
    tensor_names = set(C.list_quantizable_activations(model_bytes))
    if extra_tensor_names:
        tensor_names |= set(extra_tensor_names)
    if not tensor_names:
        return {}

    # Expose every candidate tensor as an extra graph output, so onnxruntime
    # computes (and returns) it without the graph itself needing to change.
    calib_model = onnx.ModelProto()
    calib_model.CopyFrom(model)
    existing_outputs = {o.name for o in calib_model.graph.output}
    for name in tensor_names:
        if name not in existing_outputs:
            calib_model.graph.output.append(onnx.ValueInfoProto(name=name))

    sess = ort.InferenceSession(
        calib_model.SerializeToString(),
        providers=list(providers) if providers else None,
    )
    output_names = [o.name for o in sess.get_outputs()]

    ranges: Dict[str, Tuple[float, float]] = {}
    # Only "entropy" needs every observed value kept around (to build a
    # histogram from); "minmax" only ever needs a running (min, max).
    collected: Dict[str, List[np.ndarray]] = {}
    for batch in calibration_data:
        outputs = sess.run(output_names, batch)
        for name, value in zip(output_names, outputs):
            if name not in tensor_names:
                continue
            arr = np.asarray(value)
            if arr.size == 0:
                continue
            batch_min = float(arr.min())
            batch_max = float(arr.max())
            if name in ranges:
                prev_min, prev_max = ranges[name]
                ranges[name] = (min(prev_min, batch_min), max(prev_max, batch_max))
            else:
                ranges[name] = (batch_min, batch_max)
            if method == "entropy":
                collected.setdefault(name, []).append(arr.ravel())

    if method == "entropy":
        for name, chunks in collected.items():
            threshold = _entropy_threshold(
                np.concatenate(chunks),
                num_bins=num_bins,
                num_quantized_bins=num_quantized_bins,
            )
            obs_min, obs_max = ranges[name]
            ranges[name] = (max(obs_min, -threshold), min(obs_max, threshold))

    return ranges


def quantize_static(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every MatMul, every "vanilla"
    Gemm (transA=0, alpha=1, beta=1), and every Conv, whose weight is a
    constant float32 tensor (2-D for MatMul/Gemm, rank >= 3 -- [Cout,
    Cin/groups, k...] -- for Conv).

    Unlike :func:`onnxsim.quantize_dynamic`, the activation's quantization
    range is *calibrated*: fixed ahead of time from ``calibration_data``
    (falling back to :func:`generate_random_calibration_data` when omitted)
    rather than recomputed on every inference. A
    QuantizeLinear/DequantizeLinear pair is inserted around each quantized
    tensor (the "QDQ" format): the graph still computes in float32, ready for
    a QDQ-aware runtime to fuse the pattern into a true integer kernel at
    load time.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            activation ranges from. Each batch is a ``{input_name: np.ndarray}``
            dict matching the model's graph inputs -- see
            :func:`generate_random_calibration_data` (the default, a quick
            smoke test) and :func:`load_huggingface_calibration_data` (real
            data, a much better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    ranges = calibrate(model, calibration_data, providers=providers, method=method)
    return onnx.load_from_string(C.quantize_static(model.SerializeToString(), ranges))


def quantize_static_int16(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Same as :func:`quantize_static`, but a "W8A16" scheme: the weight stays
    INT8 (identical per-output-channel symmetric scheme), while the
    activation is quantized to uint16 instead of uint8 -- an 8x finer
    calibrated affine step (1/65535 relative vs uint8's 1/255).

    Useful for activations a QDQ round trip is unusually sensitive to (e.g.
    post-softmax attention scores, or a tensor whose calibrated range is wide
    relative to its typical value), without giving up INT8's weight
    compression the way widening the weight too would. Needs opset >= 21
    (uint16 QuantizeLinear/DequantizeLinear support), unlike
    :func:`quantize_static`'s uint8 scheme, which only needs opset 13.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            activation ranges from. Each batch is a ``{input_name: np.ndarray}``
            dict matching the model's graph inputs -- see
            :func:`generate_random_calibration_data` (the default, a quick
            smoke test) and :func:`load_huggingface_calibration_data` (real
            data, a much better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    ranges = calibrate(model, calibration_data, providers=providers, method=method)
    return onnx.load_from_string(
        C.quantize_static_int16(model.SerializeToString(), ranges)
    )


def quantize_qoperator(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every MatMul and every "vanilla"
    Gemm (transA=0, alpha=1, beta=1), whose weight is a constant 2-D float32
    tensor, into the "QOperator" format -- ``QLinearMatMul``, ONNX's
    directly-quantized matmul op -- rather than :func:`quantize_static`'s QDQ
    (QuantizeLinear/DequantizeLinear wrapping a float MatMul) format. Both are
    standard ONNX; QOperator format is the older, still-standard alternative
    some runtimes' int8 kernels key off of specifically, while QDQ is the
    now-preferred, more composable format when several quantized ops chain
    together.

    Unlike QDQ format, ``QLinearMatMul`` computes directly in int8 -- there is
    no float MatMul left in the graph at all -- so this needs a calibrated
    range for each quantized node's *output* too, not just its activation
    (:func:`calibrate` is called with ``extra_tensor_names`` set to
    ``onnxsim_cpp2py_export.list_qoperator_quantizable_outputs``' result for
    this reason).

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            activation/output ranges from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`generate_random_calibration_data` (the
            default, a quick smoke test) and
            :func:`load_huggingface_calibration_data` (real data, a much
            better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_quantizable_outputs(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator(model_bytes, ranges))


def quantize_qoperator_elementwise(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every elementwise Add/Mul node
    whose two inputs are both non-constant float32 tensors (e.g. a residual
    connection, or an elementwise gate between two activations) into ONNX
    Runtime's "com.microsoft" contrib ops ``QLinearAdd``/``QLinearMul`` -- the
    elementwise, "QOperator"-format analogue of :func:`quantize_qoperator`'s
    ``QLinearMatMul`` rewrite.

    Unlike every other ``quantize_*`` function in this module, the result is
    **not** portable standard ONNX: ``QLinearAdd``/``QLinearMul`` are ONNX
    Runtime contrib ops (standard ONNX has no quantized elementwise-binary
    op), so the quantized model needs a "com.microsoft"-aware runtime --
    ONNX Runtime itself, or another runtime importing the same contrib
    schemas -- to execute. A node with a constant operand (e.g. a per-channel
    bias or embedding added elementwise) is left alone -- that operand is
    better quantized from its own static values than force-fed through
    calibration as if it varied at inference time.

    Like :func:`quantize_qoperator`, this needs a calibrated range for the
    node's *output* on top of its inputs, since QLinearAdd/QLinearMul compute
    directly in int8 with no float intermediate -- but unlike
    :func:`quantize_qoperator` (one calibrated activation, one weight
    quantized from its own static values), QLinearAdd/QLinearMul have no
    "weight" role at all, so *both* operands need a calibrated range too
    (:func:`calibrate` is called with ``extra_tensor_names`` set to
    ``onnxsim_cpp2py_export.list_qoperator_elementwise_quantizable_tensors``'
    result for this reason).

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            operand/output ranges from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`generate_random_calibration_data` (the
            default, a quick smoke test) and
            :func:`load_huggingface_calibration_data` (real data, a much
            better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_elementwise_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_elementwise(model_bytes, ranges))


def quantize_qoperator_activation(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every standalone ``Sigmoid`` or
    ``LeakyRelu`` node whose input is a float32 tensor into ONNX Runtime's
    "com.microsoft" contrib ops ``QLinearSigmoid``/``QLinearLeakyRelu`` -- the
    unary-activation analogue of :func:`quantize_qoperator_elementwise`'s
    ``QLinearAdd``/``QLinearMul`` rewrite. ``LeakyRelu``'s ``alpha`` attribute
    is carried over unchanged.

    Like :func:`quantize_qoperator_elementwise`, the result is **not**
    portable standard ONNX -- ``QLinearSigmoid``/``QLinearLeakyRelu`` are ONNX
    Runtime contrib ops, so the quantized model needs a
    "com.microsoft"-aware runtime to execute -- and needs a calibrated range
    for the node's *output* on top of its input, since these compute
    directly in int8 with no float intermediate
    (:func:`calibrate` is called with ``extra_tensor_names`` set to
    ``onnxsim_cpp2py_export.list_qoperator_activation_quantizable_tensors``'
    result for this reason).

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            input/output ranges from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`generate_random_calibration_data` (the
            default, a quick smoke test) and
            :func:`load_huggingface_calibration_data` (real data, a much
            better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_activation_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_activation(model_bytes, ranges))


def quantize_qoperator_concat(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every ``Concat`` node whose
    inputs are all non-constant float32 tensors into ONNX Runtime's
    "com.microsoft" contrib op ``QLinearConcat`` -- the variadic analogue of
    :func:`quantize_qoperator_elementwise`'s ``QLinearAdd``/``QLinearMul``
    rewrite.

    Like :func:`quantize_qoperator_elementwise`, the result is **not**
    portable standard ONNX -- ``QLinearConcat`` is an ONNX Runtime contrib
    op, so the quantized model needs a "com.microsoft"-aware runtime to
    execute -- and every input needs a calibrated range on top of the node's
    *output*, since ``QLinearConcat`` computes directly in int8 with no float
    intermediate (:func:`calibrate` is called with ``extra_tensor_names`` set
    to ``onnxsim_cpp2py_export.list_qoperator_concat_quantizable_tensors``'
    result for this reason). A node with a constant operand is left alone --
    that operand is better quantized from its own static values than
    force-fed through calibration as if it varied at inference time.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            input/output ranges from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`generate_random_calibration_data` (the
            default, a quick smoke test) and
            :func:`load_huggingface_calibration_data` (real data, a much
            better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_concat_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_concat(model_bytes, ranges))


def quantize_qoperator_softmax(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every standalone ``Softmax`` node
    whose input is a float32 tensor into ONNX Runtime's "com.microsoft"
    contrib op ``QLinearSoftmax`` -- the reduction-axis analogue of
    :func:`quantize_qoperator_activation`'s ``QLinearSigmoid``/
    ``QLinearLeakyRelu`` rewrite. ``Softmax``'s ``axis`` attribute is carried
    over unchanged (defaulting to -1 when absent).

    Like :func:`quantize_qoperator_activation`, the result is **not**
    portable standard ONNX -- ``QLinearSoftmax`` is an ONNX Runtime contrib
    op, so the quantized model needs a "com.microsoft"-aware runtime to
    execute -- and needs a calibrated range for the node's *output* on top of
    its input, since it computes directly in int8 with no float intermediate
    (:func:`calibrate` is called with ``extra_tensor_names`` set to
    ``onnxsim_cpp2py_export.list_qoperator_softmax_quantizable_tensors``'s
    result for this reason).

    ``QLinearSoftmax`` additionally needs to know which of standard ONNX's
    two incompatible ``Softmax`` axis semantics to replicate (pre-opset-13
    flattens the tensor at ``axis`` and reduces the trailing dimension;
    opset-13+ reduces over ``axis`` in place). This is resolved from
    ``model``'s own default-domain opset import, not guessed -- a model with
    no resolvable default-domain opset import is left untouched (no
    ``Softmax`` node is quantizable in it, so :func:`calibrate` has nothing
    extra to calibrate for it either).

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            input/output ranges from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`generate_random_calibration_data` (the
            default, a quick smoke test) and
            :func:`load_huggingface_calibration_data` (real data, a much
            better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param method: calibration range method, passed through to
            :func:`calibrate` -- ``"minmax"`` (default) or ``"entropy"``
            (KL-divergence calibration; see that function for the tradeoff
            and its extra data requirement).
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_softmax_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_softmax(model_bytes, ranges))
