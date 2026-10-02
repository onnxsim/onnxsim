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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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

    Dynamic dimensions (a symbolic ``dim_param``, or no dimension value set
    at all -- typically an unset/symbolic batch dimension) are fixed to 1,
    so the returned shapes are always fully concrete. A dimension whose
    ``dim_value`` is genuinely, statically 0 (e.g. an empty KV-cache
    sentinel state some exporters emit) is kept as 0 rather than promoted
    to 1: ``dim.dim_value`` reads back as 0 both when it was explicitly set
    to 0 and when the field is unset entirely, so only ``HasField`` tells
    the two apart (matches ``model_info.py``'s own ``HasField`` check for
    the same ambiguity).
    """
    initializer_names = {i.name for i in model.graph.initializer}
    specs = []
    for ipt in model.graph.input:
        if ipt.name in initializer_names:
            continue
        shape = [
            dim.dim_value if dim.HasField("dim_value") else 1
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
    coverage_floor = float(np.percentile(abs_values, min_coverage * 100.0))
    return _entropy_threshold_from_hist(
        hist, bin_edges, coverage_floor, num_quantized_bins=num_quantized_bins
    )


def _entropy_threshold_from_hist(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    coverage_floor: float,
    num_quantized_bins: int = 128,
) -> float:
    """The search half of :func:`_entropy_threshold`, on an already-built
    ``|values|`` histogram over ``[0, abs_max]`` (``bin_edges[-1]``) -- what
    :func:`calibrate`'s streaming collection accumulates batch by batch
    instead of keeping the values themselves. ``coverage_floor`` is the
    ``min_coverage`` percentile of ``|values|`` the search starts from."""
    hist = np.asarray(hist, dtype=np.float64)
    num_bins = hist.size
    abs_max = float(bin_edges[-1])
    if abs_max <= 0.0 or hist.sum() < num_quantized_bins:
        return abs_max
    # bin_edges[i] is the upper edge of the i-th bin (0-indexed), so the first
    # cutoff whose upper edge reaches the floor is searchsorted's insertion
    # point; clamped into [num_quantized_bins, num_bins] either end.
    i_start = int(
        np.clip(
            np.searchsorted(bin_edges, coverage_floor), num_quantized_bins, num_bins
        )
    )
    tail = np.concatenate([np.cumsum(hist[::-1])[::-1], [0.0]])

    best_threshold = abs_max
    best_divergence = float("inf")
    for i in range(i_start, num_bins + 1):
        ref_dist = hist[:i].copy()
        # Clipped, not dropped: fold the tail's count into the last
        # reference bin so `ref_dist` still sums to the full sample count.
        ref_dist[-1] += tail[i]
        if ref_dist.sum() == 0:
            continue

        # Simulate quantizing the clipped range to num_quantized_bins levels:
        # merge ref_dist's `i` fine bins into num_quantized_bins groups (the
        # same split np.array_split makes: the first i % k groups one bin
        # longer), then spread each group's total back out evenly over its
        # own non-empty fine bins, so the simulated distribution has the same
        # length (`i`) as ref_dist and the two are comparable bin-for-bin.
        base, extra = divmod(i, num_quantized_bins)
        sizes = np.full(num_quantized_bins, base, dtype=np.int64)
        sizes[:extra] += 1
        sizes = sizes[sizes > 0]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        nonzero = ref_dist > 0
        group_sum = np.add.reduceat(ref_dist, starts)
        group_count = np.add.reduceat(nonzero.astype(np.int64), starts)
        per_bin = np.divide(
            group_sum,
            group_count,
            out=np.zeros_like(group_sum),
            where=group_count > 0,
        )
        candidate = np.where(nonzero, np.repeat(per_bin, sizes), 0.0)

        p = _smooth_distribution(ref_dist)
        q = _smooth_distribution(candidate)
        if p is None or q is None:
            continue
        divergence = _kl_divergence(p, q)
        if divergence < best_divergence:
            best_divergence = divergence
            best_threshold = float(bin_edges[i])

    return best_threshold


def _mse_threshold(
    values: np.ndarray,
    num_candidates: int = 100,
    min_coverage: float = 0.5,
) -> float:
    """
    Find the symmetric clip threshold ``T`` minimizing the *direct* mean
    squared quantization error against ``values`` themselves (not a
    histogram-binned KL-divergence proxy the way :func:`_entropy_threshold`
    does) -- the "MSE calibration" family of clip-range search, e.g. the
    percentile/MSE calibrators several PTQ toolkits ship, and specifically
    the paper this repo's own :mod:`onnxsim.outlier_suppression` explicitly
    declined to port: Outlier Suppression's own "Token-Wise Clipping" (Wei
    et al., 2022, https://arxiv.org/abs/2209.13325) searches a clip range
    the same way -- minimizing quantized reconstruction error directly --
    rather than accepting a fixed min/max or an entropy-matched range.

    The search: for ``num_candidates`` candidate thresholds ``t`` evenly
    spaced between the ``min_coverage`` percentile of ``|values|`` and the
    observed max, simulate symmetric INT8 quantization (``scale = t / 127``,
    round-to-nearest, clip to ``[-127, 127]``, dequantize) of the *actual*
    ``values`` (not a coarsened histogram) and measure the mean squared
    error against the unquantized values. The threshold with the lowest MSE
    wins. Unlike :func:`_entropy_threshold` (which only ever sees a
    ``num_bins``-histogram approximation of the data), this measures the
    real reconstruction error every candidate would actually produce, at
    the cost of one full quantize-and-compare pass per candidate rather
    than one histogram pass total.

    ``min_coverage`` bounds the search from below (default: the search
    never considers clipping away more than the top 50% by magnitude),
    the same purpose :func:`_entropy_threshold`'s own ``min_coverage``
    serves -- without it, a smooth, tail-free distribution (e.g. a raw
    Gaussian) can still show a spuriously low MSE at a pathologically
    small threshold purely from candidates degenerating together.

    Falls back to the observed max (i.e. no clipping) when there is too
    little data, or too little dynamic range, to search meaningfully.
    """
    v = values.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    abs_v = np.abs(v)
    abs_max = float(abs_v.max())
    if abs_max <= 0.0:
        return abs_max

    floor = max(float(np.percentile(abs_v, min_coverage * 100.0)), abs_max * 1e-6)
    if floor >= abs_max:
        return abs_max

    best_threshold = abs_max
    best_mse = float("inf")
    for t in np.linspace(floor, abs_max, num_candidates):
        scale = t / 127.0
        q = np.clip(np.round(v / scale), -127, 127) * scale
        mse = float(np.mean((v - q) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_threshold = float(t)

    return best_threshold


def _hist_quantile(hist: np.ndarray, bin_edges: np.ndarray, q: float) -> float:
    """The ``q`` (in ``[0, 1]``) quantile of the values ``hist`` counts,
    linearly interpolated inside the bin it falls in."""
    cdf = np.cumsum(np.asarray(hist, dtype=np.float64))
    total = cdf[-1]
    if total <= 0:
        return float(bin_edges[0])
    target = q * total
    i = int(np.clip(np.searchsorted(cdf, target, side="left"), 0, hist.size - 1))
    below = cdf[i - 1] if i > 0 else 0.0
    count = cdf[i] - below
    frac = 0.0 if count <= 0 else float(np.clip((target - below) / count, 0.0, 1.0))
    return float(bin_edges[i] + frac * (bin_edges[i + 1] - bin_edges[i]))


def _fold_abs(hist: np.ndarray) -> np.ndarray:
    """``|values|`` histogram over ``[0, A]`` from a signed one over
    ``[-A, A]`` with an even bin count (the two halves' edges line up)."""
    half = hist.size // 2
    return hist[half:] + hist[:half][::-1]


def _mse_threshold_from_hist(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    num_candidates: int = 100,
    min_coverage: float = 0.5,
) -> float:
    """:func:`_mse_threshold` on a signed histogram over ``[-A, A]`` instead
    of the values themselves: every value is represented by its bin's center,
    weighted by the bin's count. With :func:`calibrate`'s default 2 x 2048
    bins the centers sit ~16x finer than an INT8 step at ``A``, so the
    reconstruction error each candidate is scored by is the same to well
    within the step (see the streaming-vs-exact test)."""
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    abs_max = float(bin_edges[-1])
    if total <= 0 or abs_max <= 0.0:
        return 0.0 if total <= 0 else abs_max
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # The observed |max| is at most one bin past the last non-empty bin's
    # center; use that bin's upper edge magnitude as the search ceiling.
    nz = np.nonzero(hist)[0]
    ceiling = float(max(abs(bin_edges[nz[0]]), abs(bin_edges[nz[-1] + 1])))
    ceiling = min(ceiling, abs_max)
    abs_edges = bin_edges[hist.size // 2 :]
    floor = max(
        _hist_quantile(_fold_abs(hist), abs_edges, min_coverage), ceiling * 1e-6
    )
    if floor >= ceiling:
        return ceiling
    mask = hist > 0
    c = centers[mask]
    w = hist[mask]
    best_threshold = ceiling
    best_mse = float("inf")
    for t in np.linspace(floor, ceiling, num_candidates):
        scale = t / 127.0
        q = np.clip(np.round(c / scale), -127, 127) * scale
        mse = float(np.sum(w * (c - q) ** 2) / total)
        if mse < best_mse:
            best_mse = mse
            best_threshold = float(t)
    return best_threshold


# -- Quark-style power-of-two MinMSE ("minmse_pof2") ----------------------------
#
# AMD Quark's ``PowerOfTwoMethod.MinMSE`` (what its XINT8 preset calibrates
# with): a per-tensor symmetric power-of-two scale, picked among five
# candidates around the min/max scale by the quantization error measured on a
# 2048-bin histogram (bin centres weighted by counts). Reimplemented from its
# observable behaviour; the numeric dtypes (float32 centres, int32 codes,
# float64 sums) follow it so the same data picks the same scale.

# Quark's symmetric integer ranges (``get_qmin_qmax_for_qType(symmetric=True)``)
_POF2_QRANGE = {
    "int8": (-127, 127),
    "uint8": (0, 255),
    "int16": (-32767, 32767),
    "uint16": (0, 65535),
}
_POF2_CANDIDATES = 5  # scales 2^-(p-1) ... 2^-(p+3) around the min/max position p


def _pof2_position(scale: float) -> int:
    """The fixed-point position nearest to ``scale``: ``round(-log2(scale))``."""
    scale = min(max(scale, float(2.0**-127)), float(2.0**127))
    return int(np.rint(-np.log2(scale)))


def _pof2_base(rmin: Any, rmax: Any, qmin: int, qmax: int) -> Tuple[int, int]:
    """``(position, zero_point)`` of the symmetric min/max scale rounded to a
    power of two (float32 arithmetic, as the reference does)."""
    lo, hi = np.float32(min(rmin, 0.0)), np.float32(max(rmax, 0.0))
    absmax = np.maximum(np.abs(lo), np.abs(hi))
    scale = np.float64(absmax + absmax) / np.float64(qmax - qmin)
    if scale < np.finfo(np.float32).tiny:
        return _pof2_position(1.0), 0
    zp = int(np.round(qmin + np.float64(absmax) / scale))
    pos = _pof2_position(float(np.float32(scale)))
    s = np.float32(2.0**-pos)
    new_rmin = min((np.float32(qmin) - np.float32(zp)) * s, np.float32(0))
    zp = int(np.round(qmin - new_rmin / s))
    if qmin == 0 and qmax == 255 and zp == 127:  # keep the zero point centred
        zp = 128
    return pos, zp


class _Pof2Histogram:
    """Per-tensor histogram laid out like Quark's: ``num_bins`` uniform bins
    over the first batch's ``[min, max]``, extended with bins of the same
    width when a later batch exceeds it. Accumulated one batch at a time."""

    def __init__(self, num_bins: int = 2048):
        self.num_bins = num_bins
        self.counts: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.rmin = np.float32(0)
        self.rmax = np.float32(0)

    def add(self, arr: np.ndarray) -> None:
        data = np.nan_to_num(np.asarray(arr).ravel().astype(np.float32), nan=0.0)
        if not data.size:
            return
        lo, hi = np.float32(data.min()), np.float32(data.max())
        d64 = data.astype(np.float64)
        if self.counts is None or self.edges is None:
            self.counts, self.edges = np.histogram(d64, bins=self.num_bins)
            self.rmin, self.rmax = lo, hi
            return
        e0, e1 = self.edges[0], self.edges[-1]
        width = float(self.edges[1] - self.edges[0])
        if float(lo) >= e0 and float(hi) <= e1:
            self.counts = self.counts + np.histogram(d64, bins=self.edges)[0]
        else:
            left = max(0, int(np.ceil((e0 - float(lo)) / width)))
            right = max(0, int(np.ceil((float(hi) - e1) / width)))
            n = self.counts.size + left + right
            edges = np.linspace(e0 - left * width, e1 + right * width, n + 1)
            grown = np.zeros(n, dtype=self.counts.dtype)
            grown[left : left + self.counts.size] = self.counts
            self.counts = grown + np.histogram(d64, bins=edges)[0]
            self.edges = edges
        self.rmin, self.rmax = min(self.rmin, lo), max(self.rmax, hi)

    def scale(self, dtype: str = "uint8") -> float:
        """The MinMSE power-of-two scale for quantizing to ``dtype``."""
        if self.counts is None or self.edges is None or not self.counts.sum():
            return 1.0
        qmin, qmax = _POF2_QRANGE[dtype]
        pos, zp = _pof2_base(self.rmin, self.rmax, qmin, qmax)
        c32 = ((self.edges[:-1] + self.edges[1:]) / 2).astype(np.float32)
        c64, h64 = c32.astype(np.float64), self.counts.astype(np.float64)
        best, best_s = float("inf"), np.float32(2.0**-pos)
        for i in range(_POF2_CANDIDATES):
            s = np.float32(2.0 ** -(pos + i - 1))
            q = np.clip(np.round(c32 / s).astype(np.int32) + zp, qmin, qmax)
            dq = (q.astype(np.float64) - zp) * float(s)
            diff = float(np.sum(h64 * (c64 - dq) ** 2))
            if diff < best:
                best, best_s = diff, s
        if qmin == 0 and zp == 0:
            # an all-zero tensor: Quark's zero point is 0 (not the centre), so
            # its threshold range is [0, qmax * s], which the symmetric
            # quantizer then reads as twice the scale
            return float(2 * best_s)
        return float(best_s)


def pof2_minmse_weight_scale(w: np.ndarray, qmin: int = -127, qmax: int = 127) -> float:
    """MinMSE power-of-two scale of one weight tensor (symmetric, per tensor):
    the candidate among ``2^-(p-1) ... 2^-(p+3)`` -- ``p`` the position of the
    min/max scale -- with the least squared error over the weight values
    themselves (what Quark's XINT8 does to weights and biases)."""
    data = np.asarray(w, dtype=np.float32).ravel()
    if not data.size:
        return 1.0
    pos, zp = _pof2_base(float(data.min()), float(data.max()), qmin, qmax)
    best, best_s = np.float32("inf"), np.float32(2.0**-pos)
    for i in range(_POF2_CANDIDATES):
        s = np.float32(2.0 ** -(pos + i - 1))
        q = np.clip(np.round(data / s) + np.float32(zp), qmin, qmax)
        diff = np.sum(((q - np.float32(zp)) * s - data) ** 2, dtype=np.float32)
        if diff < best:
            best, best_s = diff, s
    return float(best_s)


class _StreamingHistogram:
    """Per-tensor signed histogram over a fixed ``[-A, A]`` (``A`` = the
    tensor's observed ``max(|min|, |max|)`` from :func:`calibrate`'s first,
    min/max-only pass), accumulated one batch at a time: memory is
    ``2 * num_bins`` int64 counts per tensor however much calibration data
    streams through, instead of every observed value."""

    def __init__(self, abs_max: float, num_bins: int):
        self.edges = np.linspace(-abs_max, abs_max, 2 * num_bins + 1)
        self.abs_max = abs_max
        self.counts = np.zeros(2 * num_bins, dtype=np.int64)

    def add(self, arr: np.ndarray) -> None:
        a = arr.ravel()
        if not np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float32)
        # np.histogram with an explicit range drops NaN/inf, like the exact
        # search functions' own isfinite filter.
        h, _ = np.histogram(
            a, bins=self.counts.size, range=(-self.abs_max, self.abs_max)
        )
        self.counts += h

    def entropy_threshold(self, num_quantized_bins: int, min_coverage: float) -> float:
        abs_hist = _fold_abs(self.counts)
        abs_edges = self.edges[self.counts.size // 2 :]
        coverage_floor = _hist_quantile(abs_hist, abs_edges, min_coverage)
        return _entropy_threshold_from_hist(
            abs_hist, abs_edges, coverage_floor, num_quantized_bins=num_quantized_bins
        )

    def mse_threshold(self, num_candidates: int, min_coverage: float) -> float:
        return _mse_threshold_from_hist(
            self.counts, self.edges, num_candidates, min_coverage
        )

    def percentile_range(self, percentile: float) -> Tuple[float, float]:
        q = percentile / 100.0
        return (
            _hist_quantile(self.counts, self.edges, 1.0 - q),
            _hist_quantile(self.counts, self.edges, q),
        )


_HIST_METHODS = ("entropy", "mse", "percentile")
# What method="auto" chooses between, per tensor: every fixed method, with the
# percentile at three clip levels (``"percentile:<p>"``).
AUTO_CANDIDATES = (
    "minmax",
    "percentile:99.9",
    "percentile:99.99",
    "percentile:99.999",
    "entropy",
    "mse",
)
# Ops whose *input* tail carries the information: clipping a Sigmoid/Softmax
# logit at T caps its output at f(T) -- a detector's rare confident scores
# (YOLO11n's class logits, almost all background) are exactly that tail.
_TAIL_SENSITIVE_INPUT_OPS = {"Sigmoid", "HardSigmoid", "Softmax", "LogSoftmax", "Tanh"}
_BOUNDED_OUTPUT_OPS = {"Sigmoid", "HardSigmoid", "Softmax", "Tanh"}


def _parse_method(method: str) -> Tuple[str, Optional[float]]:
    """``"percentile:99.99"`` -> ``("percentile", 99.99)``; else ``(method, None)``."""
    if ":" in method:
        base, arg = method.split(":", 1)
        if base != "percentile":
            raise ValueError(f"unknown calibration method: {method!r}")
        return base, float(arg)
    return method, None


def _asym_qparams(lo: float, hi: float, qmax: int) -> Tuple[float, int]:
    """uint8/uint16 asymmetric (scale, zero_point) for ``[lo, hi]`` widened to
    include 0 -- the same grid quantize_static / quantize_full_graph emit."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo <= 0.0:
        return 1.0, 0
    scale = (hi - lo) / qmax
    return scale, int(np.clip(np.round(-lo / scale), 0, qmax))


def _hist_asym_mse(
    counts: np.ndarray, edges: np.ndarray, lo: float, hi: float, qmax: int = 255
) -> float:
    """Expected squared error of quantizing the values ``counts`` holds (at
    their bin centers) onto the asymmetric ``[lo, hi]`` grid: rounding error
    inside the range plus clipping error outside it."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    mask = counts > 0
    c = (0.5 * (edges[:-1] + edges[1:]))[mask]
    w = counts[mask].astype(np.float64)
    scale, zp = _asym_qparams(lo, hi, qmax)
    q = (np.clip(np.round(c / scale) + zp, 0, qmax) - zp) * scale
    return float(np.sum(w * (c - q) ** 2) / total)


class CalibrationStats:
    """What one calibration run over the data observed: every tensor's exact
    ``(min, max)`` and (unless collected with ``histograms=False``) its
    fixed-size streaming histogram. Every calibration method's ranges are a
    pure function of these -- :meth:`ranges` derives any of them, and
    :meth:`auto_ranges` picks one per tensor, without running the model
    again. Built by :func:`collect_calibration_stats`."""

    def __init__(
        self,
        observed: Dict[str, Tuple[float, float]],
        histograms: Dict[str, "_StreamingHistogram"],
        graph_outputs: Set[str],
        bounded_outputs: Set[str],
        tail_sensitive: Set[str],
        upstream: Dict[str, List[str]],
    ):
        self.observed = observed
        self.histograms = histograms
        self.graph_outputs = graph_outputs
        self.bounded_outputs = bounded_outputs
        self.tail_sensitive = tail_sensitive
        self._upstream = upstream
        #: Quark-layout histograms of ``method="minmse_pof2"`` (see
        #: :func:`collect_calibration_stats`), empty otherwise
        self.pof2_histograms: Dict[str, _Pof2Histogram] = {}
        #: Quark-layout histograms of the ``"quark_*"`` methods (see
        #: :mod:`onnxsim.quark_calibration`), empty otherwise
        self.quark_histograms: Dict[str, Any] = {}
        #: per-tensor mean over batches of each batch's (min, max) -- the
        #: ``"minmax_mean"`` method (filled by :func:`collect_calibration_stats`)
        self.mean_observed: Dict[str, Tuple[float, float]] = {}
        # (tensor, method, *params) -> range: "auto" and a model-level pick
        # ask for the same thresholds (entropy's search is the slow one)
        self._range_cache: Dict[Tuple, Tuple[float, float]] = {}

    def _method_range(
        self,
        name: str,
        method: str,
        percentile: float,
        num_quantized_bins: int,
        num_mse_candidates: int,
        mse_min_coverage: float,
        entropy_min_coverage: float,
    ) -> Tuple[float, float]:
        obs_min, obs_max = self.observed[name]
        h = self.histograms.get(name)
        base, arg = _parse_method(method)
        if base == "minmax_mean" and name in self.mean_observed:
            return self.mean_observed[name]
        if base == "minmax" or h is None or base == "minmax_mean":
            return obs_min, obs_max
        if base == "percentile":
            p = percentile if arg is None else arg
            key: Tuple = (name, base, p)
        elif base == "entropy":
            key = (name, base, num_quantized_bins, entropy_min_coverage)
        elif base == "mse":
            key = (name, base, num_mse_candidates, mse_min_coverage)
        else:
            raise ValueError(f"unknown calibration method: {method!r}")
        if key in self._range_cache:
            return self._range_cache[key]
        if base == "percentile":
            lo, hi = h.percentile_range(p)
            r = (max(obs_min, lo), min(obs_max, hi))
        else:
            if base == "entropy":
                t = h.entropy_threshold(num_quantized_bins, entropy_min_coverage)
            else:
                t = h.mse_threshold(num_mse_candidates, mse_min_coverage)
            r = (max(obs_min, -t), min(obs_max, t))
        self._range_cache[key] = r
        return r

    def head_tensors(self, depth: int) -> Set[str]:
        """Tensors at most ``depth`` producing nodes upstream of a graph
        output (``depth=1``: the inputs of the nodes that write the outputs)."""
        out: Set[str] = set()
        frontier = set(self.graph_outputs)
        for _ in range(depth):
            frontier = {i for t in frontier for i in self._upstream.get(t, ())} - out
            out |= frontier
        return out

    def ranges(
        self,
        method: str = "minmax",
        percentile: float = 99.999,
        num_quantized_bins: int = 128,
        num_mse_candidates: int = 100,
        mse_min_coverage: float = 0.5,
        entropy_min_coverage: float = 0.999,
        minmax_tensor_names: Optional[Sequence[str]] = None,
        **auto_kwargs,
    ) -> Dict[str, Tuple[float, float]]:
        """``{tensor: (min, max)}`` for ``method`` -- any :func:`calibrate`
        method, ``"percentile:<p>"``, or ``"auto"`` (see :meth:`auto_ranges`,
        which ``auto_kwargs`` go to; ``"minmse_pof2"`` takes ``dtype`` and
        ``tensor_dtypes``, see :meth:`pof2_ranges`)."""
        if method == "minmse_pof2":
            return self.pof2_ranges(
                minmax_tensor_names=minmax_tensor_names, **auto_kwargs
            )
        if method == "auto":
            return self.auto_ranges(
                minmax_tensor_names=minmax_tensor_names,
                num_quantized_bins=num_quantized_bins,
                num_mse_candidates=num_mse_candidates,
                mse_min_coverage=mse_min_coverage,
                entropy_min_coverage=entropy_min_coverage,
                **auto_kwargs,
            )[0]
        if _parse_method(method)[0] not in ("minmax", "minmax_mean") + _HIST_METHODS:
            raise ValueError(f"unknown calibration method: {method!r}")
        if (
            method not in ("minmax", "minmax_mean")
            and not self.histograms
            and self.observed
        ):
            raise ValueError(
                f"method {method!r} needs histograms: collect with histograms=True"
            )
        keep = set(minmax_tensor_names or ())
        return {
            name: (
                self.observed[name]
                if name in keep
                else self._method_range(
                    name,
                    method,
                    percentile,
                    num_quantized_bins,
                    num_mse_candidates,
                    mse_min_coverage,
                    entropy_min_coverage,
                )
            )
            for name in self.observed
        }

    def pof2_ranges(
        self,
        dtype: str = "uint8",
        tensor_dtypes: Optional[Dict[str, str]] = None,
        minmax_tensor_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Quark's ``PowerOfTwoMethod.MinMSE`` (needs
        ``collect_calibration_stats(..., pof2_histograms=True)``): each
        tensor's symmetric power-of-two scale ``s`` for ``dtype`` (or its
        ``tensor_dtypes`` override), expressed as the range ``(-s * half,
        s * half)`` -- ``half`` the dtype's positive half range -- so that a
        symmetric power-of-two quantizer recovers exactly ``s``."""
        if not self.pof2_histograms and self.observed:
            raise ValueError(
                "method 'minmse_pof2' needs pof2_histograms=True when collecting"
            )
        tensor_dtypes = tensor_dtypes or {}
        keep = set(minmax_tensor_names or ())
        out: Dict[str, Tuple[float, float]] = {}
        for name, h in self.pof2_histograms.items():
            if name in keep:
                out[name] = self.observed[name]
                continue
            dt = tensor_dtypes.get(name, dtype)
            if dt not in _POF2_QRANGE:
                raise ValueError(f"minmse_pof2 does not support dtype {dt!r}")
            lo, hi = _POF2_QRANGE[dt]
            half = hi if lo < 0 else (hi - lo) // 2
            absmax = h.scale(dt) * half
            out[name] = (-absmax, absmax)
        return out

    def auto_ranges(
        self,
        candidates: Sequence[str] = AUTO_CANDIDATES,
        minmax_tensor_names: Optional[Sequence[str]] = None,
        protect_outputs: bool = True,
        protect_bounded: bool = True,
        protect_head_depth: int = 0,
        activation_type: str = "uint8",
        num_quantized_bins: int = 128,
        num_mse_candidates: int = 100,
        mse_min_coverage: float = 0.5,
        entropy_min_coverage: float = 0.999,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str]]:
        """Per-tensor automatic calibration: for every tensor, derive each
        ``candidates`` method's range from its histogram and keep the one with
        the lowest expected quantization error on the tensor's *own*
        distribution -- on the asymmetric ``activation_type`` grid the
        quantizer actually emits, counting both rounding error and the error
        of clipped values (:func:`_hist_asym_mse`). Ties keep the earlier
        candidate (minmax first: no clipping unless it measurably helps).

        Error on a tensor is not error on the task: a clipped tail can be the
        only part that matters. So some tensors always keep their exact range
        (``"minmax"``):

        - ``minmax_tensor_names``;
        - ``protect_outputs``: graph outputs;
        - ``protect_bounded``: outputs of Sigmoid/HardSigmoid/Softmax/Tanh
          and the *inputs* of Sigmoid/HardSigmoid/Softmax/LogSoftmax/Tanh --
          clipping a logit at ``T`` caps the score at ``f(T)``, which is how
          percentile/entropy calibration zeroes a detector's rare confident
          class scores. A Sigmoid that only gates its own input (SiLU's
          ``x * sigmoid(x)``, HardSwish) is an activation, not a score: its
          input is not protected;
        - ``protect_head_depth=N``: every tensor within ``N`` nodes upstream
          of a graph output (a detection/regression head).

        :returns: ``(ranges, choices)``; ``choices[tensor]`` is the method
                picked (``"minmax (protected)"`` for the guarded ones)
        """
        qmax = {"uint8": 255, "uint16": 65535}[activation_type]
        protected = set(minmax_tensor_names or ())
        if protect_outputs:
            protected |= self.graph_outputs
        if protect_bounded:
            protected |= self.bounded_outputs | self.tail_sensitive
        if protect_head_depth > 0:
            protected |= self.head_tensors(protect_head_depth)
        for c in candidates:
            if _parse_method(c)[0] not in ("minmax",) + _HIST_METHODS:
                raise ValueError(f"unknown auto candidate: {c!r}")
        kw: Dict[str, Any] = dict(
            num_quantized_bins=num_quantized_bins,
            num_mse_candidates=num_mse_candidates,
            mse_min_coverage=mse_min_coverage,
            entropy_min_coverage=entropy_min_coverage,
        )
        ranges: Dict[str, Tuple[float, float]] = {}
        choices: Dict[str, str] = {}
        for name, obs in self.observed.items():
            h = self.histograms.get(name)
            if name in protected or h is None:
                ranges[name] = obs
                choices[name] = "minmax (protected)" if name in protected else "minmax"
                continue
            best = None
            for c in candidates:
                r = self._method_range(name, c, 99.999, **kw)
                err = _hist_asym_mse(h.counts, h.edges, r[0], r[1], qmax)
                if best is None or err < best[0]:
                    best = (err, c, r)
            assert best is not None
            ranges[name], choices[name] = best[2], best[1]
        return ranges, choices


def _is_gate(n: onnx.NodeProto, consumers: Dict[str, List[onnx.NodeProto]]) -> bool:
    """A Sigmoid/HardSigmoid only gating its own input -- ``x * sigmoid(x)``
    (SiLU/Swish), ``x * hardsigmoid(x)`` (HardSwish): its input is an
    ordinary activation, not a score logit whose tail must survive."""
    if n.op_type not in ("Sigmoid", "HardSigmoid") or not n.output[0]:
        return False
    uses = consumers.get(n.output[0], [])
    return bool(uses) and all(
        u.op_type == "Mul" and n.input[0] in u.input for u in uses
    )


def collect_calibration_stats(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]] = None,
    tensor_names: Optional[Sequence[str]] = None,
    extra_tensor_names: Optional[Sequence[str]] = None,
    histograms: bool = True,
    num_bins: int = 2048,
    pof2_histograms: bool = False,
    quark_hist: Optional[Tuple[bool, int]] = None,
    exact_session: bool = False,
) -> CalibrationStats:
    """Run the float ``model`` over ``calibration_data`` once and record what
    every calibration method needs (see :class:`CalibrationStats`): pass 1
    the exact ``(min, max)`` per tensor, pass 2 (``histograms=True``) a
    ``2 * num_bins``-bin signed histogram over ``[-max|x|, max|x|]`` per
    tensor, streamed one batch at a time -- memory is #tensors x bins however
    much data is used. Tensors are ``tensor_names`` if given, else
    ``list_quantizable_activations``' list, plus ``extra_tensor_names``.
    ``pof2_histograms=True`` also records, in pass 1, the Quark-layout
    histograms ``method="minmse_pof2"`` needs. ``quark_hist=(absolute,
    num_bins)`` records :class:`onnxsim.quark_calibration.QuarkHistogram`
    histograms of that layout (the ``"quark_*"`` methods).
    ``exact_session=True`` runs the model with ONNX Runtime's graph
    optimizations off (as Quark's calibrators do): fused kernels differ from
    the unfused ones in the last float bit, which can move a histogram count
    across a bin edge."""
    import onnxruntime as ort

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    names: Set[str]
    if tensor_names is not None:
        names = set(tensor_names)
    else:
        names = set(C.list_quantizable_activations(model.SerializeToString()))
    if extra_tensor_names:
        names |= set(extra_tensor_names)

    g = model.graph
    upstream: Dict[str, List[str]] = {}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in g.node:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)
    bounded: Set[str] = set()
    tail: Set[str] = set()
    for n in g.node:
        ins = [i for i in n.input if i]
        for o in n.output:
            if o:
                upstream[o] = ins
        if n.op_type in _BOUNDED_OUTPUT_OPS:
            bounded |= {o for o in n.output if o}
        if (
            n.op_type in _TAIL_SENSITIVE_INPUT_OPS
            and ins
            and not _is_gate(n, consumers)
        ):
            tail.add(ins[0])
    stats = CalibrationStats(
        {}, {}, {o.name for o in g.output}, bounded, tail, upstream
    )
    if not names:
        return stats

    # Expose every candidate tensor as an extra graph output, so onnxruntime
    # computes (and returns) it without the graph itself needing to change.
    calib_model = onnx.ModelProto()
    calib_model.CopyFrom(model)
    existing_outputs = {o.name for o in calib_model.graph.output}
    for name in names:
        if name not in existing_outputs:
            calib_model.graph.output.append(onnx.ValueInfoProto(name=name))

    so = ort.SessionOptions()
    if exact_session:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        calib_model.SerializeToString(),
        so,
        providers=list(providers) if providers else None,
    )
    output_names = [o.name for o in sess.get_outputs()]

    def outputs_of(batch: Tensors):
        for name, value in zip(output_names, sess.run(output_names, batch)):
            if name in names:
                arr = np.asarray(value)
                if arr.size:
                    yield name, arr

    if histograms and not isinstance(calibration_data, Sequence):
        calibration_data = list(calibration_data)

    # Pass 1: exact running (min, max) -- all "minmax" needs, and the fixed
    # histogram range for the others.
    ranges = stats.observed
    per_batch: Dict[str, Tuple[List, List]] = {}  # name -> (batch mins, batch maxes)
    for batch in calibration_data:
        for name, arr in outputs_of(batch):
            batch_min = float(arr.min())
            batch_max = float(arr.max())
            lists = per_batch.setdefault(name, ([], []))
            lists[0].append(arr.min())
            lists[1].append(arr.max())
            if pof2_histograms:
                stats.pof2_histograms.setdefault(name, _Pof2Histogram()).add(arr)
            if quark_hist is not None:
                from onnxsim.quark_calibration import QuarkHistogram

                stats.quark_histograms.setdefault(
                    name, QuarkHistogram(quark_hist[1], quark_hist[0])
                ).add(arr)
            if name in ranges:
                prev_min, prev_max = ranges[name]
                ranges[name] = (min(prev_min, batch_min), max(prev_max, batch_max))
            else:
                ranges[name] = (batch_min, batch_max)
    # (the mean of the per-batch extremes, in the tensor's own precision like
    # ONNX Runtime's ``np.nanmean`` over its batches)
    stats.mean_observed = {
        n: (float(np.nanmean(np.array(lo))), float(np.nanmean(np.array(hi))))
        for n, (lo, hi) in per_batch.items()
    }
    if not histograms:
        return stats

    # Pass 2: stream every batch into a fixed-size histogram per tensor.
    for name, (lo, hi) in ranges.items():
        abs_max = max(abs(lo), abs(hi))
        if 0.0 < abs_max < float("inf"):
            stats.histograms[name] = _StreamingHistogram(abs_max, num_bins)
    for batch in calibration_data:
        for name, arr in outputs_of(batch):
            if name in stats.histograms:
                stats.histograms[name].add(arr)
    return stats


def calibrate(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
    num_bins: int = 2048,
    num_quantized_bins: int = 128,
    num_mse_candidates: int = 100,
    mse_min_coverage: float = 0.5,
    extra_tensor_names: Optional[Sequence[str]] = None,
    percentile: float = 99.999,
    tensor_names: Optional[Sequence[str]] = None,
    minmax_tensor_names: Optional[Sequence[str]] = None,
    entropy_min_coverage: float = 0.999,
    auto_options: Optional[Dict] = None,
    activation_type: str = "uint8",
    tensor_dtypes: Optional[Dict[str, str]] = None,
    range_symmetric: Optional[bool] = None,
    quark_num_bins: Optional[int] = None,
    percentile_candidates: Sequence[float] = (99.99, 99.999, 99.99999),
    lwp_metric: str = "mae",
    exact_session: Optional[bool] = None,
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
    :param method: ``"minmax_mean"`` is ``"minmax"`` with each tensor's range
            the *mean over batches* of every batch's (min, max) instead of the
            global extremes (what Quark's ``CalibMovingAverage`` computes);
            ``"minmax"`` (default) uses each tensor's observed
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
            ``"mse"`` instead searches, per tensor, the symmetric clip
            threshold minimizing quantized reconstruction error measured
            directly against the observed values (see
            :func:`_mse_threshold`) -- Outlier Suppression's own
            "Token-Wise Clipping" (the technique :mod:`onnxsim.
            outlier_suppression`'s own docstring explicitly declines to
            port, since this function is where it belongs instead). Unlike
            ``"entropy"``'s histogram-KL proxy, this measures real
            reconstruction error at the cost of one full pass per candidate
            threshold rather than one histogram pass total; needs the same
            "at least a few hundred observed values per tensor" amount of
            calibration data ``"entropy"`` does.
            ``"percentile"`` clips each tensor to its ``(100 - percentile)``
            and ``percentile`` quantiles (two-tailed, so an asymmetric
            uint8 range keeps its own shape -- a post-ReLU tensor's lower
            quantile stays 0), then intersects with the observed range.
            Cheaper and more predictable than entropy/mse: a fixed fraction
            of outliers is clipped, whatever the distribution.
            ``"percentile:<p>"`` is the same with ``percentile=p``.
            ``"minmse_pof2"`` is AMD Quark's ``PowerOfTwoMethod.MinMSE``: a
            symmetric power-of-two scale per tensor (``activation_type``),
            the least-error one of five candidates around the min/max scale on
            a 2048-bin histogram. The returned range is ``(-s * half, s *
            half)`` for the chosen scale ``s``, so a symmetric power-of-two
            quantizer recovers ``s`` exactly.
            ``"auto"`` picks one of those per tensor, by the expected
            quantization error on that tensor's own histogram, and never
            clips graph outputs, Sigmoid/Softmax outputs and inputs, or
            ``minmax_tensor_names`` -- see
            :meth:`CalibrationStats.auto_ranges` (``auto_options`` are
            passed to it) and, to choose one method for the whole model by
            a task metric instead, :func:`onnxsim.pick_calibration`.

            All three histogram methods stream: a first pass records every
            tensor's exact ``(min, max)``; a second pass accumulates a
            fixed ``2 * num_bins``-bin signed histogram over
            ``[-max|x|, max|x|]`` per tensor, one batch at a time. Memory is
            #tensors x bins, independent of how much calibration data is
            used (the model runs twice over ``calibration_data``, which is
            materialized as a list if it is a one-shot iterator). ``"mse"``
            and the entropy coverage floor are evaluated on bin centers --
            see :func:`_mse_threshold_from_hist`.
    :param num_mse_candidates: (``"mse"`` only) number of candidate clip
            thresholds the search evaluates -- see :func:`_mse_threshold`.
    :param mse_min_coverage: (``"mse"`` only) floor on the search, as a
            percentile of ``|values|`` below which no threshold is
            considered -- see :func:`_mse_threshold`.
    :param percentile: (``"percentile"`` only) e.g. ``99.99`` or ``99.999``
    :param tensor_names: calibrate exactly these tensors instead of
            ``list_quantizable_activations``' list (``extra_tensor_names`` is
            still added) -- what a quantizer that places its own QDQ pairs
            (:func:`onnxsim.qdq_full_graph.quantize_full_graph`) passes
    :param minmax_tensor_names: tensors kept at their exact observed range
            whatever ``method`` is, e.g. graph inputs whose range is known
            (normalized pixels are exactly ``[0, 1]``)
    :param entropy_min_coverage: (``"entropy"`` only) the search's floor --
            see :func:`_entropy_threshold`'s ``min_coverage``
    :param activation_type: (``"minmse_pof2"`` only) the integer dtype the
            power-of-two scale is searched for: ``"uint8"`` (default), ``"int8"``,
            ``"uint16"`` or ``"int16"``
    :param tensor_dtypes: (``"minmse_pof2"`` only) per-tensor overrides of
            ``activation_type``
    :param auto_options: (``"auto"`` only) keyword arguments for
            :meth:`CalibrationStats.auto_ranges`, e.g.
            ``{"protect_head_depth": 2}``
    :param range_symmetric: Quark's ``CalibTensorRangeSymmetric`` (``"minmax"``
            and the ``"quark_*"`` methods): the range of every tensor is
            ``(-r, r)``. ``None`` is each method's own default (Quark's:
            ``"minmax"`` False, ``"quark_percentile"`` / ``"quark_layerwise_
            percentile"`` True, ``"quark_entropy"`` / ``"quark_distribution"``
            ignore it)
    :param quark_num_bins: histogram bins of a ``"quark_*"`` method (default:
            128 for ``"quark_entropy"``, else 2048)
    :param percentile_candidates: (``"quark_layerwise_percentile"``) the
            percentiles to choose between, per tensor
    :param lwp_metric: (``"quark_layerwise_percentile"``) ``"mae"`` or ``"mse"``

            ``"quark_percentile[:p]"``, ``"quark_entropy"``,
            ``"quark_distribution"`` and ``"quark_layerwise_percentile"`` are
            AMD Quark's calibrators of those names, scale-for-scale (see
            :mod:`onnxsim.quark_calibration`); ``"entropy"`` / ``"percentile"``
            above are onnxsim's own variants and differ from them.
    :param exact_session: run the calibration session with ONNX Runtime's graph
            optimizations off, as Quark's calibrators do (default: on for
            ``"minmse_pof2"`` and the Quark methods, off otherwise)
    :returns: ``{tensor_name: (min, max)}`` for every tensor
            ``onnxsim_cpp2py_export.list_quantizable_activations`` reports
            (or ``tensor_names``), plus ``extra_tensor_names`` if given
    """
    if method.split(":", 1)[0] in _QUARK_METHODS:
        return _calibrate_quark(
            model,
            calibration_data,
            method,
            providers=providers,
            tensor_names=tensor_names,
            extra_tensor_names=extra_tensor_names,
            minmax_tensor_names=minmax_tensor_names,
            percentile=percentile,
            activation_type=activation_type,
            range_symmetric=range_symmetric,
            num_bins=quark_num_bins,
            num_quantized_bins=num_quantized_bins,
            percentile_candidates=percentile_candidates,
            lwp_metric=lwp_metric,
        )
    base, _ = _parse_method(method)
    if (
        method not in ("auto", "minmse_pof2")
        and base not in ("minmax", "minmax_mean") + _HIST_METHODS
    ):
        raise ValueError(f"unknown calibration method: {method!r}")
    if method == "percentile" and not 50.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (50, 100], got {percentile}")

    stats = collect_calibration_stats(
        model,
        calibration_data,
        providers=providers,
        tensor_names=tensor_names,
        extra_tensor_names=extra_tensor_names,
        histograms=method not in ("minmax", "minmax_mean", "minmse_pof2"),
        num_bins=num_bins,
        pof2_histograms=method == "minmse_pof2",
        # (Quark calibrates on an unoptimized graph: ONNX Runtime's fusions and
        # NCHWc layout change the order of float operations, which is enough to
        # move a value across a histogram bin edge and flip a near-tied MinMSE
        # candidate)
        exact_session=(
            method == "minmse_pof2" if exact_session is None else exact_session
        ),
    )
    if method in ("minmax", "minmax_mean") and range_symmetric:
        out = dict(stats.mean_observed if method == "minmax_mean" else stats.observed)
        for name in stats.observed:
            if name in set(minmax_tensor_names or ()):
                out[name] = stats.observed[name]
            elif range_symmetric:
                r = max(abs(out[name][0]), abs(out[name][1]))
                out[name] = (-r, r)
        return out
    if method == "minmse_pof2":
        return stats.pof2_ranges(
            activation_type,
            tensor_dtypes,
            minmax_tensor_names=minmax_tensor_names,
        )
    kw: Dict[str, Any] = dict(
        num_quantized_bins=num_quantized_bins,
        num_mse_candidates=num_mse_candidates,
        mse_min_coverage=mse_min_coverage,
        entropy_min_coverage=entropy_min_coverage,
        minmax_tensor_names=minmax_tensor_names,
    )
    if method == "auto":
        return stats.auto_ranges(**kw, **(auto_options or {}))[0]
    return stats.ranges(method, percentile=percentile, **kw)


_QUARK_METHODS = (
    "quark_percentile",
    "quark_entropy",
    "quark_distribution",
    "quark_layerwise_percentile",
)


def _calibrate_quark(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    method: str,
    providers: Optional[Sequence[str]],
    tensor_names: Optional[Sequence[str]],
    extra_tensor_names: Optional[Sequence[str]],
    minmax_tensor_names: Optional[Sequence[str]],
    percentile: float,
    activation_type: str,
    range_symmetric: Optional[bool],
    num_bins: Optional[int],
    num_quantized_bins: int,
    percentile_candidates: Sequence[float],
    lwp_metric: str,
) -> Dict[str, Tuple[float, float]]:
    """The ``"quark_*"`` methods of :func:`calibrate`."""
    from onnxsim.quark_calibration import lwp_select

    base, _, arg = method.partition(":")
    if arg and base != "quark_percentile":
        raise ValueError(f"unknown calibration method: {method!r}")
    if base == "quark_percentile" and arg:
        percentile = float(arg)
    if base in ("quark_percentile", "quark_layerwise_percentile"):
        symmetric = True if range_symmetric is None else bool(range_symmetric)
        absolute, bins = symmetric, num_bins or 2048
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    else:
        symmetric, absolute = True, False
        bins = num_bins or (128 if base == "quark_entropy" else 2048)
    if base == "quark_distribution" and bins < 512:
        raise ValueError("quark_distribution needs num_bins >= 512")
    if base == "quark_layerwise_percentile" and activation_type not in (
        "int8",
        "uint8",
        "int16",
        "uint16",
    ):
        raise ValueError(f"unsupported activation_type: {activation_type!r}")
    stats = collect_calibration_stats(
        model,
        calibration_data,
        providers=providers,
        tensor_names=tensor_names,
        extra_tensor_names=extra_tensor_names,
        histograms=False,
        quark_hist=(absolute, bins),
        exact_session=True,
    )
    keep = set(minmax_tensor_names or ())
    out: Dict[str, Tuple[float, float]] = {}
    for name, observed in stats.observed.items():
        h = stats.quark_histograms.get(name)
        if name in keep or h is None or h.hist is None:
            out[name] = observed
        elif base == "quark_percentile":
            out[name] = h.percentile_range(percentile, symmetric)
        elif base == "quark_entropy":
            out[name] = h.entropy_range(num_quantized_bins)
        elif base == "quark_distribution":
            out[name] = h.distribution_range()
        else:
            out[name] = lwp_select(
                h, percentile_candidates, activation_type, lwp_metric, symmetric
            )
    return out


def quantize_static(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
    full_graph: bool = False,
    per_channel: bool = True,
    nodes_to_exclude: Optional[Sequence[str]] = None,
    op_types_to_exclude: Optional[Sequence[str]] = None,
    activation_type: str = "uint8",
    percentile: float = 99.999,
    minmax_tensor_names: Optional[Sequence[str]] = None,
    auto_options: Optional[Dict] = None,
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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement; or ``"percentile"``, or ``"auto"``
            (a method per tensor -- :meth:`CalibrationStats.auto_ranges`;
            to pick one method for the whole model by a task metric, see
            :func:`pick_calibration`).
    :param full_graph: QDQ *every* float activation (uint8/uint16
            asymmetric), with INT8 weights and INT32 biases, instead of only
            the MatMul/Gemm/Conv inputs -- what a whole-graph integer NPU
            (e.g. the Qualcomm HTP through ORT's QNN EP) needs; see
            :mod:`onnxsim.qdq_full_graph`. Graph inputs and the outputs of
            bounded ops (Sigmoid, Softmax...) always keep their exact
            observed range whatever ``method`` is -- see
            :func:`onnxsim.qdq_full_graph.bounded_output_tensors`.
    :param per_channel: (``full_graph`` only) per-output-channel weight
            scales; the default scheme is always per channel
    :param nodes_to_exclude: (``full_graph`` only) node names left in float
    :param op_types_to_exclude: (``full_graph`` only) op types left in float
    :param activation_type: (``full_graph`` only) ``"uint8"`` or ``"uint16"``
    :param percentile: (``method="percentile"`` only) passed to
            :func:`calibrate`
    :param auto_options: (``method="auto"`` only) passed to
            :meth:`CalibrationStats.auto_ranges`
    :param minmax_tensor_names: (``full_graph`` only) tensors kept at their
            exact observed range whatever ``method`` is. A detector's score
            path is the typical case: its logits are almost all background,
            so the rare large logits that *are* the detections sit above any
            percentile/entropy/mse clip (YOLO11n at the 99.99th percentile
            clips its class logits to ~0, capping every score at ~0.5)
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    plan = _StaticPlan(
        model,
        full_graph=full_graph,
        per_channel=per_channel,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_exclude=op_types_to_exclude,
        activation_type=activation_type,
        minmax_tensor_names=minmax_tensor_names,
    )
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        percentile=percentile,
        tensor_names=plan.tensor_names,
        minmax_tensor_names=plan.minmax_tensor_names,
        auto_options=dict(auto_options or {}, activation_type=activation_type)
        if method == "auto"
        else None,
    )
    return plan.apply(ranges)


class _StaticPlan:
    """What :func:`quantize_static` calibrates and how it applies the ranges,
    shared with :func:`pick_calibration` (which applies many range sets from
    one calibration run)."""

    def __init__(
        self,
        model: onnx.ModelProto,
        full_graph: bool = False,
        per_channel: bool = True,
        nodes_to_exclude: Optional[Sequence[str]] = None,
        op_types_to_exclude: Optional[Sequence[str]] = None,
        activation_type: str = "uint8",
        minmax_tensor_names: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.full_graph = full_graph
        self.per_channel = per_channel
        self.activation_type = activation_type
        self.exclude = dict(
            nodes_to_exclude=nodes_to_exclude or (),
            op_types_to_exclude=op_types_to_exclude or (),
        )
        self.tensor_names: Optional[List[str]] = None
        self.minmax_tensor_names: Optional[List[str]] = None
        if full_graph:
            from onnxsim import qdq_full_graph

            inits = {t.name for t in model.graph.initializer}
            self.tensor_names = qdq_full_graph.list_full_graph_activations(
                model, **self.exclude
            )
            self.minmax_tensor_names = (
                [i.name for i in model.graph.input if i.name not in inits]
                + qdq_full_graph.bounded_output_tensors(model)
                + list(minmax_tensor_names or ())
            )
        elif (
            not per_channel
            or minmax_tensor_names
            or nodes_to_exclude
            or op_types_to_exclude
            or activation_type != "uint8"
        ):
            raise ValueError(
                "per_channel=False, nodes_to_exclude, op_types_to_exclude, "
                "minmax_tensor_names and activation_type only apply with full_graph=True (use "
                "quantize_static_int16 for W8A16 on the default scheme)"
            )

    def apply(self, ranges: Dict[str, Tuple[float, float]]) -> onnx.ModelProto:
        if self.full_graph:
            from onnxsim import qdq_full_graph

            return qdq_full_graph.quantize_full_graph(
                self.model,
                ranges,
                per_channel=self.per_channel,
                activation_type=self.activation_type,
                **self.exclude,
            )
        return onnx.load_from_string(
            C.quantize_static(self.model.SerializeToString(), ranges)
        )


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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
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


def quantize_qoperator_pool(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every standalone ``AveragePool``
    or ``GlobalAveragePool`` node whose input is a float32 tensor into ONNX
    Runtime's "com.microsoft" contrib ops ``QLinearAveragePool``/
    ``QLinearGlobalAveragePool`` -- the pooling analogue of
    :func:`quantize_qoperator_activation`'s ``QLinearSigmoid``/
    ``QLinearLeakyRelu`` rewrite. Every attribute the original
    ``AveragePool`` node has (``kernel_shape``, ``pads``, ``strides``,
    ``ceil_mode``, ``count_include_pad``, ``auto_pad``) is carried over
    unchanged; both ops additionally get a ``channels_last`` attribute set
    to 0, since onnxsim only ever produces NCHW-layout graphs.

    Like :func:`quantize_qoperator_activation`, the result is **not**
    portable standard ONNX -- these are ONNX Runtime contrib ops, so the
    quantized model needs a "com.microsoft"-aware runtime to execute -- and
    need a calibrated range for the node's *output* on top of its input,
    since they compute directly in int8 with no float intermediate
    (:func:`calibrate` is called with ``extra_tensor_names`` set to
    ``onnxsim_cpp2py_export.list_qoperator_pool_quantizable_tensors``'s
    result for this reason).

    An ``AveragePool`` node with a ``dilations`` attribute (standard ONNX
    opset 19+) is left untouched: ONNX Runtime's ``QLinearAveragePool``
    kernel does not accept that attribute.

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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_pool_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_pool(model_bytes, ranges))


def quantize_qoperator_where(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every ``Where`` node whose two
    data operands (the second and third inputs -- the condition is always
    boolean and never quantized) are both non-constant float32 tensors into
    ONNX Runtime's "com.microsoft" contrib op ``QLinearWhere`` -- the
    ternary-select analogue of :func:`quantize_qoperator_elementwise`'s
    ``QLinearAdd``/``QLinearMul`` rewrite.

    Like :func:`quantize_qoperator_elementwise`, the result is **not**
    portable standard ONNX -- ``QLinearWhere`` is an ONNX Runtime contrib
    op, so the quantized model needs a "com.microsoft"-aware runtime to
    execute -- and every operand needs a calibrated range on top of the
    node's *output*, since ``QLinearWhere`` computes directly in int8 with
    no float intermediate (:func:`calibrate` is called with
    ``extra_tensor_names`` set to
    ``onnxsim_cpp2py_export.list_qoperator_where_quantizable_tensors``'s
    result for this reason). A node with a constant operand is left alone --
    that operand is better quantized from its own static values than
    force-fed through calibration as if it varied at inference time.

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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_where_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_where(model_bytes, ranges))


def quantize_qoperator_gemm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every ``Gemm`` node whose weight
    ``B`` is a constant 2-D float32 tensor into ONNX Runtime's
    "com.microsoft" contrib op ``QGemm`` -- the fully-general analogue of
    :func:`quantize_qoperator`'s ``QLinearMatMul`` rewrite, which only
    handles "vanilla" Gemm (``transA=0``, ``alpha=1``) because
    ``QLinearMatMul`` has no transpose/scale attributes of its own.
    ``QGemm`` keeps ``transA``/``transB``/``alpha`` as attributes, so this
    function handles any ``transA``, ``transB``, or ``alpha`` value
    :func:`quantize_qoperator` cannot.

    ``B`` is quantized per output channel (INT8, symmetric) in its own
    storage layout -- no forced transpose, since ``QGemm`` keeps ``transB``
    as its own attribute. A bias ``C``, when present, is quantized ahead of
    time into INT32 with zero point 0 and a per-column scale of
    ``alpha * a_scale * b_scale[n]`` (``QGemm``'s own documented bias
    convention) and accumulated directly in the quantized compute -- unlike
    :func:`quantize_qoperator`'s vanilla-Gemm handling, which adds the bias
    back in float *after* dequantizing. Only a 1-D ``C`` of exactly ``N``
    elements (the common per-column-bias case) with ``beta == 1`` is
    handled; any other ``C`` shape or a non-default ``beta`` is left alone,
    since ``QGemm`` has no ``beta`` attribute of its own to carry a
    different value through.

    Like :func:`quantize_qoperator_elementwise`, the result is **not**
    portable standard ONNX -- ``QGemm`` is an ONNX Runtime contrib op, so
    the quantized model needs a "com.microsoft"-aware runtime to execute --
    and needs a calibrated range for the node's *output* on top of the
    activation's, since it computes directly in int8 with no float
    intermediate (:func:`calibrate` is called with ``extra_tensor_names``
    set to ``onnxsim_cpp2py_export.list_qoperator_gemm_quantizable_tensors``'s
    result for this reason).

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
            :func:`calibrate` -- ``"minmax"`` (default), ``"entropy"``
            (KL-divergence calibration), or ``"mse"`` (direct reconstruction-
            error calibration); see that function for the tradeoffs and
            their extra data requirement.
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    model_bytes = model.SerializeToString()
    extra_names = C.list_qoperator_gemm_quantizable_tensors(model_bytes)
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=method,
        extra_tensor_names=extra_names,
    )
    return onnx.load_from_string(C.quantize_qoperator_gemm(model_bytes, ranges))
