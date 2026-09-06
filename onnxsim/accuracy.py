"""A unified, typed entry point over onnxsim's quantization schemes
(:class:`QuantizationConfig` / :func:`quantize`), and a data-driven
measurement of a specific quantized model's actual accuracy drop
(:func:`measure_accuracy_drop`).

onnxsim ships more than a dozen ``quantize_*`` functions
(:func:`onnxsim.quantize_dynamic`, :func:`onnxsim.quantize_static`,
:func:`onnxsim.quantize_weight_only_int4`, ...), each its own scheme with its
own parameter surface, documented and callable directly as always. This
module adds a second, unified way to reach all of them: describe *what*
quantization you want (scheme, dtype, granularity, calibration settings) as
one :class:`QuantizationConfig`, and let :func:`quantize` dispatch to the
right underlying function -- useful for code that picks a scheme
programmatically (a sweep over configs, a config file, a CLI flag) rather
than calling a specific ``quantize_*`` function by name.

For how much accuracy a given quantization actually costs, two tools, at two
different price points:

- :func:`onnxsim.estimate_model_quantization_drop` (in
  ``precision_estimator.py``) -- static, no execution or data needed, an
  *estimate* from the model's weights and shapes alone. Fast pre-check.
- :func:`measure_accuracy_drop`, here -- runs the float and quantized models
  through ONNX Runtime (or the reference evaluator, see ``backend.py``) on
  the same input data and reports actual output differences. Slower (needs
  data and two full model runs per sample), but it's a measurement, not an
  estimate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim import backend

# apply_awq/apply_gptq/apply_gptaq/apply_double_quantization are imported at
# module scope (not deferred inside quantize()) -- none of awq.py, gptq.py,
# gptaq.py, or double_quantization.py import anything from onnxsim.accuracy,
# so there is no import cycle to avoid here.
from onnxsim.awq import apply_awq
from onnxsim.calibration import (
    _ELEM_TYPE_TO_NP,
    Tensors,
    generate_random_calibration_data,
    quantize_qoperator,
    quantize_static,
    quantize_static_int16,
)
from onnxsim.double_quantization import apply_double_quantization
from onnxsim.gptaq import apply_gptaq
from onnxsim.gptq import apply_gptq
from onnxsim.onnx_simplifier import (
    quantize_bf16,
    quantize_dynamic,
    quantize_dynamic_matmul_integer_to_float,
    quantize_fp8,
    quantize_fp16,
    quantize_ternary,
    quantize_weight_only,
    quantize_weight_only_int4,
    quantize_weight_only_int8_block,
    quantize_weight_only_int16,
)

# scheme -> the dtypes it accepts for QuantizationConfig.dtype.
_SCHEME_DTYPES = {
    "dynamic": {"int8"},
    "dynamic_fused": {"int8"},
    "ternary": {"int8"},
    "weight_only": {"int8", "int16", "int4"},
    "static": {"int8"},
    "static_int16": {"int16"},
    "qoperator": {"int8"},
    "float": {"float16", "bfloat16", "float8_e4m3", "float8_e5m2"},
}
_CALIBRATION_SCHEMES = {"static", "static_int16", "qoperator"}


@dataclass
class QuantizationConfig:
    """Describes one onnxsim quantization scheme and its parameters, for
    :func:`quantize` to dispatch on. Every field not relevant to ``scheme``
    is simply ignored (e.g. ``calibration_data`` for ``scheme="dynamic"``,
    which needs none) -- see :func:`quantize`'s docstring for the full
    scheme/dtype/granularity matrix and which onnxsim function each maps to.

    :param scheme: one of ``"dynamic"``, ``"dynamic_fused"``, ``"ternary"``,
            ``"weight_only"``, ``"static"``, ``"static_int16"``,
            ``"qoperator"``, ``"float"``.
    :param dtype: the quantized representation. Meaning depends on
            ``scheme`` -- see :func:`quantize`.
    :param granularity: ``"per_channel"`` (default) or ``"per_block"``.
            Only ``scheme="weight_only"`` currently offers a choice (its
            ``dtype="int8"`` case); every other scheme/dtype combination has
            exactly one granularity onnxsim implements today, and this field
            is ignored for them.
    :param calibration_data: representative input batches for the
            calibration-based schemes (``"static"``, ``"static_int16"``,
            ``"qoperator"``) -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data`.
    :param num_calibration_samples: random calibration batches to generate
            when ``calibration_data`` is omitted (calibration-based schemes
            only).
    :param seed: seed for the random calibration data (calibration-based
            schemes only; ignored if ``calibration_data`` is supplied).
    :param providers: onnxruntime execution providers to calibrate on
            (calibration-based schemes only).
    :param calibration_method: ``"minmax"`` (default) or ``"entropy"``,
            passed through to :func:`onnxsim.calibrate` (calibration-based
            schemes only).
    :param keep_io_types: for ``scheme="float"`` only -- keep the model's
            external input/output types at float32 (inserting boundary
            ``Cast`` nodes) instead of redeclaring them in the target
            format. Default ``True``.
    :param awq: only meaningful for ``scheme="weight_only"``,
            ``dtype="int4"`` -- if ``True``, chain
            :func:`onnxsim.apply_awq` (activation-aware per-channel weight
            rescaling, using ``calibration_data``) onto
            :func:`onnxsim.quantize_weight_only_int4`'s output. Ignored
            (silently) for every other scheme/dtype -- see :func:`quantize`'s
            docstring for the fixed pipeline order when combined with
            ``gptq``/``gptaq``/``double_quant``.
    :param gptq: only meaningful for ``scheme="weight_only"``,
            ``dtype="int4"`` -- if ``True``, chain
            :func:`onnxsim.apply_gptq` (Hessian-compensated sequential
            rounding, using ``calibration_data``) after ``awq`` (if also
            set). Ignored (silently) for every other scheme/dtype.
    :param gptaq: only meaningful for ``scheme="weight_only"``,
            ``dtype="int4"`` -- if ``True``, chain
            :func:`onnxsim.apply_gptaq` (asymmetric-calibration variant of
            GPTQ, using ``calibration_data``) after ``gptq`` (if also set --
            GPTQ's own correction is itself a valid starting point for
            GPTAQ's further correction, so setting both is not rejected).
            Ignored (silently) for every other scheme/dtype.
    :param double_quant: only meaningful for ``scheme="weight_only"``,
            ``dtype="int4"`` -- if ``True``, chain
            :func:`onnxsim.apply_double_quantization` (re-quantizes the
            model's own per-block float32 scales into a smaller
            representation; needs no calibration data) last, after
            ``awq``/``gptq``/``gptaq``. Ignored (silently) for every other
            scheme/dtype.
    :param double_quant_min_elements: forwarded to
            :func:`onnxsim.apply_double_quantization`'s own
            ``min_elements`` parameter when ``double_quant`` is set. Default
            matches that function's own default (``64``).
    """

    scheme: str
    dtype: str = "int8"
    granularity: str = "per_channel"
    calibration_data: Optional[Sequence[Tensors]] = None
    num_calibration_samples: int = 8
    seed: int = 0
    providers: Optional[Sequence[str]] = None
    calibration_method: str = "minmax"
    keep_io_types: bool = True
    awq: bool = False
    gptq: bool = False
    gptaq: bool = False
    double_quant: bool = False
    double_quant_min_elements: int = 64


def quantize(
    model: Union[str, onnx.ModelProto], config: QuantizationConfig
) -> onnx.ModelProto:
    """Quantizes ``model`` according to ``config``, dispatching to the
    matching onnxsim ``quantize_*`` function. The scheme/dtype/granularity
    matrix:

    ================  ===============  ==============  ===========================================
    scheme            dtype            granularity     onnxsim function
    ================  ===============  ==============  ===========================================
    dynamic           int8             per_channel      :func:`onnxsim.quantize_dynamic`
    dynamic_fused     int8             per_channel      :func:`onnxsim.quantize_dynamic_matmul_integer_to_float`
    ternary           int8             per_channel      :func:`onnxsim.quantize_ternary`
    weight_only       int8             per_channel      :func:`onnxsim.quantize_weight_only`
    weight_only       int8             per_block        :func:`onnxsim.quantize_weight_only_int8_block`
    weight_only       int16            per_channel      :func:`onnxsim.quantize_weight_only_int16`
    weight_only       int4             per_block        :func:`onnxsim.quantize_weight_only_int4`
    static            int8             per_channel      :func:`onnxsim.quantize_static`
    static_int16      int16            per_channel      :func:`onnxsim.quantize_static_int16`
    qoperator         int8             per_channel      :func:`onnxsim.quantize_qoperator`
    float              float16          n/a             :func:`onnxsim.quantize_fp16`
    float              bfloat16         n/a             :func:`onnxsim.quantize_bf16`
    float              float8_e4m3      n/a             :func:`onnxsim.quantize_fp8` (format="e4m3")
    float              float8_e5m2      n/a             :func:`onnxsim.quantize_fp8` (format="e5m2")
    ================  ===============  ==============  ===========================================

    ``weight_only``/``int4`` is always block-wise (block_size=32, the only
    granularity :func:`onnxsim.quantize_weight_only_int4` implements) --
    ``granularity`` is not consulted for it, and likewise for every other
    row with exactly one implemented granularity.

    For ``scheme="weight_only"``, ``dtype="int4"`` specifically,
    :attr:`QuantizationConfig.awq`, ``gptq``, ``gptaq``, and
    ``double_quant`` chain onnxsim's calibration-driven correction passes
    onto :func:`onnxsim.quantize_weight_only_int4`'s plain round-to-nearest
    output, in this fixed order: AWQ's activation-aware rescale
    (:func:`onnxsim.apply_awq`) first, then GPTQ's Hessian-compensated
    rounding (:func:`onnxsim.apply_gptq`), then GPTAQ's
    asymmetric-calibration variant of the same (:func:`onnxsim.apply_gptaq`),
    then double-quantization of the resulting scales
    (:func:`onnxsim.apply_double_quantization`) last -- matching AWQ's own
    rescale feeding GPTQ's Hessian pass feeding double-quantization. ``gptq``
    and ``gptaq`` may both be set: GPTQ's own correction is itself a valid
    starting point for GPTAQ's further correction, so this simply applies
    both in sequence rather than being rejected as redundant. Every flag
    defaults to ``False`` (behaving exactly like calling
    :func:`onnxsim.quantize_weight_only_int4` directly), and all four are
    silently ignored for every other scheme/dtype pair -- see
    :class:`QuantizationConfig`'s own docstring.

    Raises :class:`ValueError` for an unknown ``scheme``, a ``dtype`` not
    valid for that ``scheme``, or (``scheme="weight_only"``, ``dtype="int8"``)
    with a ``granularity`` other than ``"per_channel"``/``"per_block"``.

    :param model: onnx ModelProto object or file path
    :param config: the quantization scheme and its parameters
    :returns: the quantized onnx ModelProto
    """
    scheme = config.scheme
    valid_dtypes = _SCHEME_DTYPES.get(scheme)
    if valid_dtypes is None:
        raise ValueError(
            f"unknown QuantizationConfig.scheme {scheme!r}; expected one of "
            f"{sorted(_SCHEME_DTYPES)}"
        )
    if config.dtype not in valid_dtypes:
        raise ValueError(
            f"scheme={scheme!r} does not support dtype={config.dtype!r}; "
            f"expected one of {sorted(valid_dtypes)}"
        )

    if scheme == "dynamic":
        return quantize_dynamic(model)
    if scheme == "dynamic_fused":
        return quantize_dynamic_matmul_integer_to_float(model)
    if scheme == "ternary":
        return quantize_ternary(model)

    if scheme == "weight_only":
        if config.dtype == "int16":
            return quantize_weight_only_int16(model)
        if config.dtype == "int4":
            # apply_awq/apply_gptq/apply_gptaq each take the *original*
            # float model as their own first argument -- load `model` once
            # here (it may be a bare path) so every call below (including
            # quantize_weight_only_int4 itself) sees the exact same loaded
            # ModelProto, rather than each re-loading its own separate copy
            # from disk.
            float_model = (
                onnx.load(model, load_external_data=False)
                if isinstance(model, str)
                else model
            )
            quantized = quantize_weight_only_int4(float_model)
            if config.awq:
                quantized = apply_awq(
                    float_model,
                    quantized,
                    calibration_data=config.calibration_data,
                    num_samples=config.num_calibration_samples,
                    seed=config.seed,
                    providers=config.providers,
                )
            if config.gptq:
                quantized = apply_gptq(
                    float_model,
                    quantized,
                    calibration_data=config.calibration_data,
                    num_samples=config.num_calibration_samples,
                    seed=config.seed,
                    providers=config.providers,
                )
            if config.gptaq:
                quantized = apply_gptaq(
                    float_model,
                    quantized,
                    calibration_data=config.calibration_data,
                    num_samples=config.num_calibration_samples,
                    seed=config.seed,
                    providers=config.providers,
                )
            if config.double_quant:
                quantized = apply_double_quantization(
                    quantized, min_elements=config.double_quant_min_elements
                )
            return quantized
        # dtype == "int8": the one scheme/dtype pair with a granularity choice.
        if config.granularity == "per_channel":
            return quantize_weight_only(model)
        if config.granularity == "per_block":
            return quantize_weight_only_int8_block(model)
        raise ValueError(
            "scheme='weight_only', dtype='int8' supports granularity "
            f"'per_channel' or 'per_block', got {config.granularity!r}"
        )

    if scheme in _CALIBRATION_SCHEMES:
        fn = {
            "static": quantize_static,
            "static_int16": quantize_static_int16,
            "qoperator": quantize_qoperator,
        }[scheme]
        return fn(
            model,
            calibration_data=config.calibration_data,
            num_calibration_samples=config.num_calibration_samples,
            seed=config.seed,
            providers=config.providers,
            method=config.calibration_method,
        )

    # scheme == "float"
    if config.dtype == "float16":
        return quantize_fp16(model, keep_io_types=config.keep_io_types)
    if config.dtype == "bfloat16":
        return quantize_bf16(model, keep_io_types=config.keep_io_types)
    fp8_format = "e4m3" if config.dtype == "float8_e4m3" else "e5m2"
    return quantize_fp8(model, format=fp8_format, keep_io_types=config.keep_io_types)


def _cast_batch_to_model_inputs(model: onnx.ModelProto, batch: Tensors) -> Tensors:
    """Casts each array in `batch` to the graph input's own declared dtype,
    for a floating-point mismatch only -- e.g. a ``keep_io_types=False``
    float16 quantization (see :func:`onnxsim.quantize_fp16`) redeclares
    graph inputs in the narrow format directly, so the same float32
    calibration data used against the original model needs casting before
    it can feed the quantized one. Integer/bool inputs, and bfloat16/float8
    targets (no native numpy dtype in ``_ELEM_TYPE_TO_NP``; pre-cast
    ``calibration_data`` yourself with ``ml_dtypes`` for those), are left
    untouched.
    """
    elem_type_by_name = {
        i.name: i.type.tensor_type.elem_type for i in model.graph.input
    }
    out = {}
    for name, arr in batch.items():
        elem_type = elem_type_by_name.get(name)
        np_dtype = _ELEM_TYPE_TO_NP.get(elem_type) if elem_type is not None else None
        if (
            np_dtype is not None
            and np.issubdtype(np_dtype, np.floating)
            and arr.dtype != np_dtype
        ):
            arr = arr.astype(np_dtype)
        out[name] = arr
    return out


@dataclass
class OutputAccuracyStats:
    """Worst-case-over-samples accuracy stats for one model output -- see
    :func:`measure_accuracy_drop`."""

    output_name: str
    relative_l2: float  # ||float - quantized|| / ||float||, worst over samples
    max_abs_error: float  # max(|float - quantized|), worst over samples
    cosine_similarity: (
        float  # dot(float, quantized) / (||float|| * ||quantized||), worst over samples
    )


@dataclass
class AccuracyDropReport:
    """Measured (not estimated) accuracy drop between a float model and a
    quantized version of it -- see :func:`measure_accuracy_drop`."""

    num_samples: int
    per_output: Dict[str, OutputAccuracyStats] = field(default_factory=dict)
    worst_relative_l2: float = float("nan")  # max over every output/sample
    worst_cosine_distance: float = float("nan")  # 1 - min cosine_similarity
    all_finite: bool = True  # False if the quantized model ever produced NaN/Inf


def measure_accuracy_drop(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> AccuracyDropReport:
    """Runs ``float_model`` and ``quantized_model`` on the same input data
    (through :func:`onnxsim.backend.run_model` -- onnxruntime when
    installed, the pure-Python reference evaluator otherwise) and reports
    how far each of the quantized model's outputs actually drifts from the
    float model's -- an empirical measurement, not
    :func:`onnxsim.estimate_model_quantization_drop`'s static estimate.

    Runs single-threaded with onnxruntime's ``use_deterministic_compute`` set
    (see :func:`onnxsim.backend.run_model`'s ``single_threaded``/
    ``deterministic`` parameters), at some performance cost: a measurement
    that silently varies with the host's core count or SIMD capabilities
    (e.g. AVX-512 vs. AVX2) isn't a trustworthy quality gate for CI or
    anywhere else the same model+data might be measured on different
    machines. This does not change *what* is measured -- only removes those
    two axes of run-to-run variance from how it is measured.

    Assumes ``float_model`` and ``quantized_model`` declare the same output
    *names* and count -- true of every onnxsim ``quantize_*``/:func:`quantize`
    call, which never renames or adds/removes graph outputs. Per-output
    stats are the **worst case across samples**, not an average: the point
    of measuring accuracy drop is to know the worst a deployment might see,
    not to average it away.

    :param float_model: the original (unquantized) onnx ModelProto or file path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), e.g. from :func:`quantize` or any
            ``quantize_*`` function
    :param calibration_data: representative input batches to measure on.
            Each batch is a ``{input_name: np.ndarray}`` dict matching
            ``float_model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a much more representative measurement than random
            input for a real deployment).
    :param num_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :returns: the measured accuracy-drop report
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    output_names = [o.name for o in float_model.graph.output]
    per_output_l2: Dict[str, List[float]] = {name: [] for name in output_names}
    per_output_abs: Dict[str, List[float]] = {name: [] for name in output_names}
    per_output_cos: Dict[str, List[float]] = {name: [] for name in output_names}
    all_finite = True

    for batch in calibration_data:
        float_out = backend.run_model(
            float_model,
            batch,
            providers=providers,
            single_threaded=True,
            deterministic=True,
        )
        quantized_out = backend.run_model(
            quantized_model,
            _cast_batch_to_model_inputs(quantized_model, batch),
            providers=providers,
            single_threaded=True,
            deterministic=True,
        )
        for name in output_names:
            f = np.asarray(float_out[name], dtype=np.float64).ravel()
            q = np.asarray(quantized_out[name], dtype=np.float64).ravel()
            if not np.all(np.isfinite(q)):
                all_finite = False
            f_norm = float(np.linalg.norm(f))
            q_norm = float(np.linalg.norm(q))
            rel_l2 = float(np.linalg.norm(f - q)) / max(f_norm, 1e-12)
            max_abs = float(np.max(np.abs(f - q))) if f.size else 0.0
            denom = f_norm * q_norm
            cos_sim = float(np.dot(f, q) / denom) if denom > 0 else float("nan")
            per_output_l2[name].append(rel_l2)
            per_output_abs[name].append(max_abs)
            per_output_cos[name].append(cos_sim)

    per_output: Dict[str, OutputAccuracyStats] = {}
    for name in output_names:
        per_output[name] = OutputAccuracyStats(
            output_name=name,
            relative_l2=max(per_output_l2[name]),
            max_abs_error=max(per_output_abs[name]),
            cosine_similarity=min(per_output_cos[name]),
        )

    worst_relative_l2 = max(
        (s.relative_l2 for s in per_output.values()), default=float("nan")
    )
    worst_cosine_similarity = min(
        (s.cosine_similarity for s in per_output.values()), default=1.0
    )
    return AccuracyDropReport(
        num_samples=len(calibration_data),
        per_output=per_output,
        worst_relative_l2=worst_relative_l2,
        worst_cosine_distance=1.0 - worst_cosine_similarity,
        all_finite=all_finite,
    )


DEFAULT_QUANTIZATION_CANDIDATES: List[QuantizationConfig] = [
    QuantizationConfig(scheme="ternary"),
    QuantizationConfig(scheme="weight_only", dtype="int4"),
    QuantizationConfig(scheme="dynamic"),
    QuantizationConfig(scheme="weight_only", dtype="int8", granularity="per_block"),
    QuantizationConfig(scheme="weight_only", dtype="int8", granularity="per_channel"),
    QuantizationConfig(scheme="qoperator"),
    QuantizationConfig(scheme="static"),
    QuantizationConfig(scheme="weight_only", dtype="int16"),
    QuantizationConfig(scheme="static_int16", dtype="int16"),
    QuantizationConfig(scheme="float", dtype="float8_e4m3"),
    QuantizationConfig(scheme="float", dtype="bfloat16"),
    QuantizationConfig(scheme="float", dtype="float16"),
]
"""Default search order for :func:`recommend_quantization`, most compressed
first: ~1-2 bit ternary, then 4/8/16-bit integer schemes (``dynamic`` before
the per-channel/per-block ``weight_only`` int8 variants -- it needs no
calibration data and is usually onnxsim's best-supported path), then 8-bit
float, then 16-bit float as the final, almost-always-safe fallback. Not a
benchmarked ranking -- actual size/latency/accuracy trade-offs are model- and
backend-dependent; pass your own ``candidates`` to :func:`recommend_quantization`
for a different order or a narrower/wider set."""


@dataclass
class QuantizationRecommendation:
    """One :func:`recommend_quantization` result: the winning (or, if none
    met the budget, least-lossy) scheme, its measured accuracy drop, and the
    model already quantized with it."""

    config: QuantizationConfig
    report: AccuracyDropReport
    quantized_model: onnx.ModelProto
    meets_budget: bool


def _initializer_nbytes(model: onnx.ModelProto) -> int:
    """Total size of ``model``'s initializers actually referenced by a node
    input. onnxsim's ``quantize_*`` passes rewrite a node to point at new
    (smaller) quantized initializers but don't themselves prune the
    now-unused original -- that's ``simplify()``'s job, run separately --
    so counting every initializer regardless of reachability would make a
    successful quantization look like it grew the model.
    """
    referenced = {name for n in model.graph.node for name in n.input}
    total = 0
    for t in model.graph.initializer:
        if t.name not in referenced:
            continue
        total += (
            len(t.raw_data) if t.HasField("raw_data") else len(t.SerializeToString())
        )
    return total


def recommend_quantization(
    model: Union[str, onnx.ModelProto],
    accuracy_budget: float = 0.02,
    candidates: Optional[Sequence[QuantizationConfig]] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> QuantizationRecommendation:
    """Automatically picks a quantization scheme for ``model`` by measuring
    (not estimating -- see :func:`measure_accuracy_drop`) each candidate in
    ``candidates`` (default :data:`DEFAULT_QUANTIZATION_CANDIDATES`, most
    compressed/lossy first) on the same data, and returning the first one
    that both (a) actually shrinks the model's initializers and (b) keeps
    the worst-case relative L2 error under ``accuracy_budget`` with only
    finite outputs.

    onnxsim has no notion of a deployment target -- no hardware, runtime, or
    latency budget enters this search. It only ever optimizes for *accuracy*
    under an ever-more-aggressive-first sweep of the schemes
    :func:`quantize` already knows how to dispatch to. A candidate a given
    runtime can't execute efficiently (or at all) is still a candidate here;
    filter ``candidates`` yourself for that constraint.

    Condition (a) exists because a scheme that finds nothing to quantize in
    ``model`` (e.g. ``"ternary"`` on a model with no ternary-quantizable
    pattern) still returns a valid -- just unchanged -- model, which would
    otherwise look like a perfect (zero-error) "win" despite quantizing
    nothing. Candidates that don't shrink the model are skipped outright,
    including as the final fallback.

    If no candidate meets both conditions, the least-lossy one that did
    shrink the model is returned anyway (``meets_budget=False``) so the
    caller can still inspect or use it, or retry with a raised
    ``accuracy_budget``. Raises :class:`ValueError` only if every candidate
    either failed to apply (:func:`quantize` itself raised) or failed to
    shrink the model.

    :param model: onnx ModelProto object or file path
    :param accuracy_budget: maximum acceptable worst-case relative L2 error
            (see :attr:`AccuracyDropReport.worst_relative_l2`) for a
            candidate to be accepted
    :param candidates: schemes to try, in order; defaults to
            :data:`DEFAULT_QUANTIZATION_CANDIDATES`. Each candidate's
            ``calibration_data``/``num_calibration_samples``/``seed``/
            ``providers`` fields are overwritten with this function's own
            same-named parameters -- set ``scheme``/``dtype``/``granularity``
            only.
    :param calibration_data: representative input batches, used both to
            calibrate the calibration-based schemes (``"static"``,
            ``"static_int16"``, ``"qoperator"``) and to measure every
            candidate's accuracy drop. See
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a much more representative search than random input).
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to calibrate/run on
    :returns: the winning (or, if none met budget, least-lossy) candidate
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    float_nbytes = _initializer_nbytes(model)

    best: Optional[QuantizationRecommendation] = None
    for base_config in (
        DEFAULT_QUANTIZATION_CANDIDATES if candidates is None else candidates
    ):
        config = replace(
            base_config,
            calibration_data=calibration_data,
            num_calibration_samples=num_samples,
            seed=seed,
            providers=providers,
        )
        try:
            quantized = quantize(model, config)
            if _initializer_nbytes(quantized) >= float_nbytes:
                continue  # no-op quantization -- not a real candidate
            report = measure_accuracy_drop(
                model,
                quantized,
                calibration_data=calibration_data,
                num_samples=num_samples,
                seed=seed,
                providers=providers,
            )
        except Exception:
            # Either quantize() declined this scheme, or the quantized
            # model quantized fine but the execution backend can't run it
            # (e.g. float8e4m3 MatMul on the CPU EP) -- either way, not a
            # usable candidate on this model/backend; try the next.
            continue
        meets_budget = report.all_finite and report.worst_relative_l2 < accuracy_budget
        recommendation = QuantizationRecommendation(
            config=config,
            report=report,
            quantized_model=quantized,
            meets_budget=meets_budget,
        )
        if meets_budget:
            return recommendation
        if best is None or (
            report.all_finite
            and (
                not best.report.all_finite
                or report.worst_relative_l2 < best.report.worst_relative_l2
            )
        ):
            best = recommendation

    if best is None:
        raise ValueError(
            "no candidate quantization scheme both applied to this model and "
            "shrank it -- pass a different `candidates` list"
        )
    return best


def quantize_auto(
    model: Union[str, onnx.ModelProto],
    accuracy_budget: float = 0.02,
    **kwargs,
) -> onnx.ModelProto:
    """Convenience wrapper around :func:`recommend_quantization` that
    returns just the quantized model -- see its docstring for every
    parameter, forwarded here via ``**kwargs``.

    Warns (does not raise) if no candidate met ``accuracy_budget``: the
    returned model is still the least-lossy one actually measured, already
    quantized and usable, just not guaranteed to be within budget. Call
    :func:`recommend_quantization` directly instead if you need the measured
    report or want to react to a missed budget programmatically rather than
    via a warning.

    :param model: onnx ModelProto object or file path
    :param accuracy_budget: see :func:`recommend_quantization`
    :param kwargs: forwarded to :func:`recommend_quantization`
    :returns: the recommended scheme's quantized model
    """
    recommendation = recommend_quantization(
        model, accuracy_budget=accuracy_budget, **kwargs
    )
    if not recommendation.meets_budget:
        warnings.warn(
            f"no quantization candidate met accuracy_budget={accuracy_budget}; "
            f"returning the least-lossy one tried ({recommendation.config.scheme}/"
            f"{recommendation.config.dtype}, worst_relative_l2="
            f"{recommendation.report.worst_relative_l2:.4f})",
            stacklevel=2,
        )
    return recommendation.quantized_model
