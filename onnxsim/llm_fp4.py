"""LLM-FP4 (Liu, Yuan, Yang, Cheng, Yang, Liu, Zhu and Xu, 2023, EMNLP,
"LLM-FP4: 4-Bit Floating-Point Quantized Transformers",
https://arxiv.org/abs/2310.16836). onnxsim ports the algorithm, not any
framework's code, per the same rationale as :mod:`onnxsim.smoothquant`/
:mod:`onnxsim.zeroquant`.

**Relationship to onnxsim's other 4-bit floating-point modules.** onnxsim
already has two other 4-bit *floating-point* codebook formats:
:mod:`onnxsim.mx_quantization` (MXFP4: fixed E2M1 elements, per-block scale
restricted to a **power of two**, per the OCP MX spec) and :mod:`onnxsim.nf4`
(NormalFloat4: a fixed, data-*independent* 16-value codebook fit to a
standard normal distribution's own quantile points, not a sign/exponent/
mantissa format at all). LLM-FP4 is neither: it is a **standard**
sign/exponent/mantissa FP4 format (like MXFP4's own E2M1), but

1. its per-block scale is an ordinary **real-valued** float, not restricted
   to a power of two (MXFP4's own restriction), and
2. it does not fix the exponent/mantissa bit split at E2M1 -- it **searches**
   a small set of splits (this module: E1M2, E2M1, E3M0 -- every way to
   divide FP4's 3 non-sign bits between exponent and mantissa) per tensor,
   picking whichever minimizes reconstruction error, rather than using one
   format for every tensor unconditionally.

Both properties come from the paper's own "pre-shifted exponent bias"
framing: for a *floating-point* quantizer (unlike an affine INT4 quantizer),
the per-block scale and the format's own exponent bias are two names for the
same real-valued degree of freedom -- multiplying a block by ``2^d`` before
quantizing to a fixed-bias FP4 codebook is identical to quantizing the
unscaled block against a codebook whose bias has been shifted by ``d``. This
module realizes that freedom directly as a per-block real-valued scale (the
same representation :mod:`onnxsim.nf4` already uses for its own non-power-
of-two scale), searched, together with the bit-split choice, by grid search
against direct reconstruction MSE -- the same "hold everything else fixed,
scan a small set of candidates, keep whichever minimizes a direct error
metric" shape :func:`onnxsim.calibration._mse_threshold`/:func:`onnxsim.
calibration._entropy_threshold` already use for INT8 range calibration
(``_mse_threshold`` scans candidate clip thresholds against *direct*
reconstruction MSE -- the closer analogue to this module's own objective;
``_entropy_threshold`` scans the same kind of candidates against a
histogram-based KL divergence instead. This module scans candidate
*(format, clip ratio)* pairs against MSE -- a different, two-dimensional
search space, same "grid search over candidates" shape).

    Before:
      Y = MatMul(X, W) [+ bias]      -- W constant, [K, N], float32

    After:
      Codebook: initializer, float32 [16]  -- winning format's 16 fixed
                                               values (shared across every
                                               layer that picks this format)
      Wq: initializer, uint8 [K, N]        -- codebook index per element
      Ws: initializer, float32 [K/block_size, N]  -- real-valued (not
                                               power-of-two) scale per block
      What_hat = Reshape(Mul(Reshape(Gather(Codebook, Wq)), Ws), ...)
      Y = MatMul(X, What_hat) [+ bias]

**Deliberately not ported: activation quantization (W4A4) via cross-layer
exponent-bias migration.** The paper's other headline contribution is
"pre-shifted exponent bias" applied to *activations*: transformer
activations carry per-channel outliers (the same phenomenon
:mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression` address for
INT8), so the paper computes a per-channel real-valued scale from
calibration data and migrates it algebraically into the preceding weight or
LayerNormalization (exactly :mod:`onnxsim.smoothquant`'s and
:mod:`onnxsim.outlier_suppression`'s own migration, with FP4's real-valued
scale-as-bias standing in for those modules' INT8 quantization range) so
that, after migration, a *single* shared exponent bias suffices for a
straightforward per-tensor FP4 activation quantizer at graph-run time. That
migration machinery already exists in this repo (:func:`onnxsim.
apply_smoothquant`, :func:`onnxsim.apply_outlier_suppression`) and composes
with this module unchanged: run either migration pass first, then quantize
the migrated model's activations with a per-tensor FP4 QDQ-style insertion.
Building that activation-side QDQ insertion itself -- the graph-run-time
"quantize X to FP4, matmul against Wq" pipeline, analogous to
:mod:`onnxsim.zeroquant`'s own runtime activation quantization but emitting
FP4 codes instead of INT8 -- is real, non-trivial additional scope (a new
runtime dequantization/quantization subgraph, not a data-flow migration),
and is not implemented here. This module covers weight-only W4 quantization
with the paper's own format-and-bias search as its differentiator from
:mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`; W4A4 is a legitimate
follow-up, not attempted in this module.

ONNX has no native FP4 tensor type (as of this writing), so -- following the
exact same approach :mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4` already
use for their own codebook formats -- this module builds the dequantization
out of ordinary ONNX ops any opset-11+ runtime already supports: ``Gather``
the per-element code out of a 16-entry constant codebook, then ``Mul`` by
the per-block real-valued scale.

**Update: two activation-side passes now exist.** The "deliberately not
ported" section above is no longer wholly accurate: this paragraph
supersedes its closing "is not implemented here"/"W4A4 is a legitimate
follow-up, not attempted in this module" claims (its description of what
the paper's own activation scheme *is* remains accurate, and is what
:func:`apply_llm_fp4_activation_quantization_per_tensor` below implements
the quantizer half of). Both passes act only on layers
:func:`quantize_weight_only_llm_fp4` has already weight-quantized (found by
walking the weight input backward through that function's own exact
dequantization pattern -- see :func:`_find_llm_fp4_weight_codebook`), and
both reuse that layer's own recovered codebook rather than re-deriving one:

* :func:`apply_llm_fp4_activation_quantization` -- **per-token,
  data-free**, and *not* the paper's own design. It computes the scale
  fresh from each token's own values at graph-run time, exactly the
  convention every other per-token runtime quantizer in this repo already
  uses (:mod:`onnxsim.zeroquant`, :mod:`onnxsim.quarot`,
  :mod:`onnxsim.duquant`, :mod:`onnxsim.attention_quantization`,
  :mod:`onnxsim.kv_cache_quantization`'s Value-style rewrite), just against
  FP4's non-uniform 16-value codebook instead of a uniform integer grid.
  No calibration data and no migration pass needed.
* :func:`apply_llm_fp4_activation_quantization_per_tensor` -- the
  **calibrated per-tensor** half of the paper's own design. It fits one
  real-valued scale per activation tensor offline, from calibration data,
  by the *same* ``(clip ratio -> reconstruction MSE)`` grid search the
  weight side already uses (:func:`_search_fp4_clip_ratio`, shared by both),
  and bakes it into the graph as a constant initializer -- so the emitted
  quantize/dequantize subgraph contains no runtime range reduction at all.
  The *other* half of the paper's design -- migrating the per-channel
  outliers into the preceding weight/LayerNormalization, which is what
  makes a single shared exponent bias sufficient in the first place -- is
  still **not** performed by that function; the caller composes it by
  running :func:`onnxsim.apply_smoothquant` or
  :func:`onnxsim.apply_outlier_suppression` first, as the "deliberately not
  ported" section above already anticipated. See that function's own
  docstring for the three-call sequence and the full honesty note.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data

# Every way to split FP4's 3 non-sign bits between exponent and mantissa --
# the paper's own candidate set for its per-tensor format search. Named
# "eXmY" for X exponent bits, Y mantissa bits (X + Y == 3 always, since FP4
# is 1 sign bit + 3 remaining bits).
FP4_FORMATS: Dict[str, Tuple[int, int]] = {
    "e1m2": (1, 2),
    "e2m1": (2, 1),  # MXFP4's own element format (onnxsim.mx_quantization)
    "e3m0": (3, 0),
}

# Cap on how many activation elements
# :func:`apply_llm_fp4_activation_quantization_per_tensor` feeds into its
# per-tensor scale search. That search allocates a ``size x 16`` float64
# distance tensor per candidate clip ratio, and its input is every
# calibration sample concatenated -- unbounded, unlike the weight side,
# whose input is bounded by the weight. 2**18 elements caps the search at a
# few tens of MB while still estimating a *single* scalar's MSE objective
# from a quarter-million samples. See that function for how it is applied
# (``max_abs`` stays exact over the full data; only the MSE estimate is
# subsampled).
_PER_TENSOR_FIT_MAX_ELEMENTS = 1 << 18


def _fp4_magnitudes(e_bits: int, m_bits: int) -> List[float]:
    """The 8 non-negative magnitudes an ``e_bits``-exponent/``m_bits``-
    mantissa 4-bit float evaluates to, per the standard IEEE-754-style
    definition (bias ``2^(e_bits-1) - 1``, exponent field 0 == subnormal).
    Ascending, starting at ``0.0``. ``e_bits + m_bits`` must be 3 (FP4's 3
    non-sign bits).
    """
    assert e_bits + m_bits == 3, "FP4 has 3 non-sign bits to split"
    bias = (1 << (e_bits - 1)) - 1 if e_bits > 0 else 0
    magnitudes = set()
    for exp_field in range(1 << e_bits):
        for mant_field in range(1 << m_bits):
            frac = mant_field / float(1 << m_bits)
            if exp_field == 0:
                value = frac * (2.0 ** (1 - bias))  # subnormal
            else:
                value = (1.0 + frac) * (2.0 ** (exp_field - bias))  # normal
            magnitudes.add(value)
    return sorted(magnitudes)


def _fp4_codebook(e_bits: int, m_bits: int) -> List[float]:
    """The full 16 signed codes for an ``e_bits``/``m_bits`` FP4 format:
    ``[-max, ..., -0.0, 0.0, ..., max]`` -- the same negatives-then-
    positives-with-a-duplicate-zero layout :mod:`onnxsim.mx_quantization`'s
    own ``MXFP4_CODEBOOK`` and :mod:`onnxsim.nf4`'s own ``NF4_CODEBOOK``
    use, so ``_nearest_codebook_index``'s indexing convention matches
    theirs.
    """
    magnitudes = _fp4_magnitudes(e_bits, m_bits)  # 8 ascending, [0] == 0.0
    negatives = [-m for m in reversed(magnitudes)]  # -max ... -0.0
    return negatives + magnitudes  # 16: -max...-0.0, 0.0...max


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(w_name, weight_transposed)`` or ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[1], False
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[1], weight_transposed
    return None


def _search_fp4_clip_ratio(
    values: np.ndarray,
    max_abs: np.ndarray,
    codebook: np.ndarray,
    clip_ratios: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The clip-ratio half of this module's own ``(format, clip ratio)`` grid
    search, for one *fixed* ``codebook``: for each group along ``values``'
    leading axes, scans ``clip_ratios`` and keeps whichever
    ``scale = max_abs * ratio / max(codebook)`` minimizes that group's own
    codebook round-trip MSE (in the original, unnormalized units).

    ``values`` has shape ``group_shape + (group_size,)`` and ``max_abs``
    shape ``group_shape`` -- so the same routine serves both the weight
    side (``group_shape == (N, K // block_size)``: one scale per
    ``(output channel, block)`` group, see :func:`_search_llm_fp4_blockwise`)
    and the activation side (``group_shape == ()``: a single per-tensor
    scale over one flat calibration sample, see
    :func:`apply_llm_fp4_activation_quantization_per_tensor`). Returns
    ``(best_error, best_scale, best_codes)`` with shapes ``group_shape``,
    ``group_shape`` and ``values.shape`` respectively; ``codebook`` must be
    ascending so its last entry is the largest magnitude.
    """
    max_mag = codebook[-1]
    group_shape = values.shape[:-1]
    best_error = np.full(group_shape, np.inf)
    best_scale = np.zeros(group_shape)
    best_codes = np.zeros(values.shape, dtype=np.int64)

    for r in clip_ratios:
        # The "pre-shifted exponent bias" search, realized as a real-valued
        # scale: r < 1 clips outliers harder but sharpens resolution for the
        # bulk of the group, exactly the clip-vs-resolution trade
        # _mse_threshold's own cutoff search makes for INT8 ranges.
        scale = np.maximum(max_abs * r / max_mag, 1e-30)  # group_shape
        normalized = values / scale[..., np.newaxis]
        diffs = np.abs(normalized[..., np.newaxis] - codebook)
        codes = np.argmin(diffs, axis=-1)  # values.shape
        dequant_normalized = codebook[codes]
        error = (
            np.sum((dequant_normalized - normalized) ** 2, axis=-1) * scale**2
        )  # group_shape, in the original (unnormalized) units

        improved = error < best_error
        best_error = np.where(improved, error, best_error)
        best_scale = np.where(improved, scale, best_scale)
        best_codes = np.where(improved[..., np.newaxis], codes, best_codes)

    return best_error, best_scale, best_codes


def _search_llm_fp4_blockwise(
    w_nk: np.ndarray,
    block_size: int,
    formats: Sequence[str],
    clip_ratios: np.ndarray,
) -> "tuple[str, np.ndarray, np.ndarray]":
    """Searches, for ``w_nk`` ([N, K], output channel first), the ``formats``
    x ``clip_ratios`` grid that minimizes total reconstruction MSE, per this
    module's own docstring. Returns ``(best_format, codes_nk, scale_blocks)``:
    codebook indices in ``[0, 15]`` (shape ``[N, K]``) and one real-valued
    scale per ``(output channel, block-of-K)`` group (shape
    ``[N, K // block_size]``) for the winning format. Assumes
    ``K % block_size == 0`` and ``formats``/``clip_ratios`` non-empty.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    max_abs = np.maximum(np.abs(blocks).max(axis=2), 1e-30)  # [N, num_blocks]

    best_format = formats[0]
    best_total_error = np.inf
    best_codes = np.zeros((n, num_blocks, block_size), dtype=np.uint8)
    best_scale = np.zeros((n, num_blocks))

    for fmt in formats:
        e_bits, m_bits = FP4_FORMATS[fmt]
        codebook = np.asarray(_fp4_codebook(e_bits, m_bits), dtype=np.float64)

        fmt_best_error, fmt_best_scale, fmt_best_codes = _search_fp4_clip_ratio(
            blocks, max_abs, codebook, clip_ratios
        )

        total_error = float(fmt_best_error.sum())
        if total_error < best_total_error:
            best_total_error = total_error
            best_format = fmt
            best_codes = fmt_best_codes
            best_scale = fmt_best_scale

    return best_format, best_codes.astype(np.uint8).reshape(n, k), best_scale


def quantize_weight_only_llm_fp4(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    formats: Sequence[str] = ("e1m2", "e2m1", "e3m0"),
    num_scale_candidates: int = 17,
    min_clip_ratio: float = 0.5,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D float32
    weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into LLM-FP4's weight format -- see this module's own
    docstring for the technique and its scope. This function itself is
    **weight-only**; to also quantize the activation (W4A4), follow it with
    either :func:`apply_llm_fp4_activation_quantization_per_tensor` (the
    paper's own calibrated per-tensor quantizer, which composes with
    :func:`onnxsim.apply_smoothquant`/
    :func:`onnxsim.apply_outlier_suppression` run first) or
    :func:`apply_llm_fp4_activation_quantization` (a data-free per-token
    alternative). Needs no calibration data: both the per-tensor
    format choice and the per-block scale are fit directly to each weight's
    own values, by exhaustive grid search minimizing reconstruction MSE.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale group
            along the reduction dimension
    :param formats: candidate FP4 exponent/mantissa bit splits to search per
            tensor -- keys into :data:`FP4_FORMATS`. The paper's own
            ablation set (every way to split FP4's 3 non-sign bits) is the
            default; a subset restricts (and speeds up) the search
    :param num_scale_candidates: number of per-block clip-ratio candidates
            to grid-search (evenly spaced over ``[min_clip_ratio, 1.0]``)
            for each format; more candidates costs more search time for a
            finer-grained scale
    :param min_clip_ratio: lower end of the per-block clip-ratio search
            range -- ``1.0`` keeps the block's own max-abs element exactly
            at the format's largest representable magnitude (no clipping);
            values below ``1.0`` let the search trade a harder clip on
            outliers for sharper resolution on the rest of the block
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Reshape(Gather(codebook, Cast(Wq, INT64)), ...), Ws) ->
            Reshape(..., original shape)`` feeding the original MatMul/Gemm
            node -- ordinary ONNX ops only, no contrib op and no minimum
            opset beyond what ``Gather``/``Cast``/``Reshape``/``Mul``
            themselves need (opset 11+). Layers with a non-constant,
            non-2-D, or non-block-divisible weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()
    formats = list(formats)
    if not formats or any(fmt not in FP4_FORMATS for fmt in formats):
        raise ValueError(f"formats must be a non-empty subset of {sorted(FP4_FORMATS)}")
    clip_ratios = np.linspace(min_clip_ratio, 1.0, max(num_scale_candidates, 1))

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    codebook_names: Dict[str, str] = {}  # format -> initializer name, created lazily

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        w_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        fmt, codes_nk, scale_blocks = _search_llm_fp4_blockwise(
            w_nk, block_size, formats, clip_ratios
        )
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        if fmt not in codebook_names:
            e_bits, m_bits = FP4_FORMATS[fmt]
            codebook_names[fmt] = _unique_name(f"llm_fp4_codebook_{fmt}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(_fp4_codebook(e_bits, m_bits), dtype=np.float32),
                    name=codebook_names[fmt],
                )
            )
        codebook_name = codebook_names[fmt]
        num_blocks = k // block_size

        wq = onnx.numpy_helper.from_array(
            codes_orig.astype(np.uint8),
            name=_unique_name(f"{w_name}_llmfp4_q", taken_names),
        )
        graph.initializer.append(wq)
        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_llmfp4_scale", taken_names),
        )
        graph.initializer.append(ws)

        if weight_transposed:
            blocked_shape = [n, num_blocks, block_size]
            scale_shape = [n, num_blocks, 1]
        else:
            blocked_shape = [num_blocks, block_size, n]
            scale_shape = [num_blocks, 1, n]

        cast_out = _unique_name(f"{w_name}_llmfp4_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [wq.name], [cast_out], to=onnx.TensorProto.INT64
        )

        gather_out = _unique_name(f"{w_name}_llmfp4_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather", [codebook_name, cast_out], [gather_out], axis=0
        )

        blocked_shape_name = _unique_name(f"{w_name}_llmfp4_blocked_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(blocked_shape, dtype=np.int64), name=blocked_shape_name
            )
        )
        reshaped_out = _unique_name(f"{w_name}_llmfp4_reshaped", taken_names)
        reshape1_node = onnx.helper.make_node(
            "Reshape", [gather_out, blocked_shape_name], [reshaped_out]
        )

        scale_shape_name = _unique_name(f"{w_name}_llmfp4_scale_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(scale_shape, dtype=np.int64), name=scale_shape_name
            )
        )
        scale_reshaped_out = _unique_name(
            f"{w_name}_llmfp4_scale_reshaped", taken_names
        )
        reshape2_node = onnx.helper.make_node(
            "Reshape", [ws.name, scale_shape_name], [scale_reshaped_out]
        )

        scaled_out = _unique_name(f"{w_name}_llmfp4_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul", [reshaped_out, scale_reshaped_out], [scaled_out]
        )

        orig_shape_name = _unique_name(f"{w_name}_llmfp4_orig_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([dim0, dim1], dtype=np.int64), name=orig_shape_name
            )
        )
        dq_out = _unique_name(f"{w_name}_llmfp4_dq", taken_names)
        reshape3_node = onnx.helper.make_node(
            "Reshape",
            [scaled_out, orig_shape_name],
            [dq_out],
            name=_unique_name(f"{w_name}_llmfp4_dequant", taken_names),
        )

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in (
            cast_node,
            gather_node,
            reshape1_node,
            reshape2_node,
            mul_node,
            reshape3_node,
        ):
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _find_llm_fp4_weight_codebook(
    w_name: str,
    producer_map: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
) -> Optional[str]:
    """Walks backward from a MatMul/Gemm's own weight input through the
    *exact* dequantization pattern :func:`quantize_weight_only_llm_fp4`
    builds above -- ``Reshape(Mul(Reshape(Gather(Codebook,
    Cast(Wq))), Reshape(Ws)), orig_shape)`` -- and returns that layer's own
    ``Codebook`` initializer name, or ``None`` if ``w_name`` isn't fed by
    exactly this pattern (an unquantized layer, or one quantized by a
    different onnxsim scheme entirely).

    Matched structurally, op-by-op, against the real node-construction
    order in :func:`quantize_weight_only_llm_fp4` above -- not by
    initializer/node naming convention -- because the winning format isn't
    recorded anywhere else in the graph: the ``Gather`` node's own first
    input is the only source of truth for which codebook a given layer
    actually used.
    """
    reshape3 = producer_map.get(w_name)
    if reshape3 is None or reshape3.op_type != "Reshape" or len(reshape3.input) != 2:
        return None
    mul = producer_map.get(reshape3.input[0])
    if mul is None or mul.op_type != "Mul" or len(mul.input) != 2:
        return None
    reshape1 = producer_map.get(mul.input[0])
    reshape2 = producer_map.get(mul.input[1])
    if reshape1 is None or reshape1.op_type != "Reshape" or len(reshape1.input) != 2:
        return None
    if reshape2 is None or reshape2.op_type != "Reshape" or len(reshape2.input) != 2:
        return None
    if reshape2.input[0] not in initializer_map:  # Ws: constant per-block scale
        return None
    gather = producer_map.get(reshape1.input[0])
    if gather is None or gather.op_type != "Gather" or len(gather.input) != 2:
        return None
    codebook_name = gather.input[0]
    if codebook_name not in initializer_map:
        return None
    cast = producer_map.get(gather.input[1])
    if cast is None or cast.op_type != "Cast" or len(cast.input) != 1:
        return None
    if cast.input[0] not in initializer_map:  # Wq: constant codebook indices
        return None
    return codebook_name


def apply_llm_fp4_activation_quantization(
    model: Union[str, onnx.ModelProto],
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """Completes W4A4 for every layer :func:`quantize_weight_only_llm_fp4`
    has already weight-quantized, by inserting a **per-token, data-free**
    FP4 quantize/dequantize round-trip on that layer's own activation
    input, using that same layer's own (byte-identical, recovered -- not
    re-derived) codebook.

    **Honesty note -- this is NOT the paper's own activation-quantization
    design; read before using.** The real LLM-FP4 paper's headline
    activation contribution is a *per-channel* real-valued scale, fit from
    calibration data and migrated into the preceding weight or
    LayerNormalization (via :func:`onnxsim.apply_smoothquant`/
    :func:`onnxsim.apply_outlier_suppression`), so that a single shared
    exponent bias suffices for a *per-tensor* FP4 activation quantizer --
    see this module's own docstring and ``docs/llm-fp4.md``'s "Scope"
    section. The *quantizer* half of that design --  one calibrated,
    real-valued **per-tensor** scale, baked into the graph as a constant --
    is implemented by this module's own
    :func:`apply_llm_fp4_activation_quantization_per_tensor`, which is the
    function to use if you want the paper's own scheme (compose it after
    :func:`onnxsim.apply_smoothquant`/
    :func:`onnxsim.apply_outlier_suppression` for the migration half). This
    function does **not** reproduce that design: it computes
    a **per-token** (per-row) scale fresh, at graph-run time, from each
    token's own values -- no calibration data, no migration pass, no
    stored statistics -- exactly the convention every other per-token
    runtime quantizer in this repo already uses (:mod:`onnxsim.zeroquant`,
    :mod:`onnxsim.quarot`, :mod:`onnxsim.duquant`, :mod:`onnxsim.
    attention_quantization`, :mod:`onnxsim.kv_cache_quantization`'s
    Value-style rewrite). It is a simpler, self-contained alternative that
    completes W4A4 without requiring either migration pass first --
    composing with :func:`onnxsim.apply_smoothquant`/:func:`onnxsim.
    apply_outlier_suppression` beforehand remains possible (they act on the
    weight/LayerNorm side and are unaffected by this pass, since it never
    touches those), but this simpler design does not require it.

    For each layer already matched by :func:`quantize_weight_only_llm_fp4`
    (found by walking its weight input backward through that function's
    exact dequantization pattern -- see :func:`_find_llm_fp4_weight_codebook`
    -- so a layer this module didn't itself weight-quantize is left
    completely untouched), the activation ``X`` is quantized as::

        scale = max(ReduceMax(Abs(X), axis=-1, keepdims=1), epsilon)
                / max(abs(Codebook))                    -- one scale per token
        x_normalized = X / scale
        nearest = Gather(Codebook, ArgMin(Abs(Unsqueeze(x_normalized, -1)
                                              - Codebook), axis=-1))
        x_dequant = nearest * scale

    "Nearest codebook value" is found by broadcasting every element against
    all 16 codebook entries and taking ``ArgMin`` over the distances, then
    gathering the codebook by that index -- the same "distance to every
    codebook entry, then argmin" idea this module's own (offline, numpy)
    weight-side search and :mod:`onnxsim.iq4_nl`'s own codebook search
    already use, just expressed as runtime ONNX ops since ``X`` is not a
    compile-time constant here. This construction was checked directly
    against a numpy reference via ``onnxruntime`` before being committed to
    this module (see ``tests/test_llm_fp4.py``).

    **Runtime cost, beyond node count.** That broadcast is the price of a
    *non-uniform* codebook: unlike a uniform integer grid (which quantizes
    with plain arithmetic, as :mod:`onnxsim.quarot`/:mod:`onnxsim.zeroquant`
    do), finding the nearest of 16 arbitrary values needs each element
    compared against all 16, so the ``Sub``/``Abs`` intermediates are **16x
    the activation's own size** while the lookup executes. On a large
    activation that transient dominates this pass's memory, and no count of
    inserted nodes reflects it. ONNX has no non-uniform-codebook
    quantization op that would avoid the materialization.

    :param model: the original or weight-quantized onnx ModelProto or file
            path -- must already have been passed through
            :func:`quantize_weight_only_llm_fp4` for this function to do
            anything, since it only recognizes that function's own exact
            weight-dequantization pattern
    :param epsilon: floor applied to a token's own max-abs value before
            using it as a scale, avoiding a divide-by-zero on an all-zero
            token
    :returns: ``model`` with every already-FP4-weight-quantized layer's
            activation input replaced by its own per-token FP4
            quantize/dequantize round-trip; every other input (weight,
            bias) and the node's own output name are left exactly as
            :func:`quantize_weight_only_llm_fp4` left them. A layer whose
            weight input isn't fed by exactly that function's own
            dequantization pattern is left completely untouched. A model
            with no such layer, or with an opset older than 18
            (``ReduceMax``'s ``axes``-as-input form needs opset 18, the
            same gate :func:`onnxsim.apply_zeroquant` uses), is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 18):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    producer_map = {output: node for node in graph.node for output in node.output}
    taken_names = _all_names(graph)

    candidates = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        w_name, _weight_transposed = match
        x_name = node.input[0]  # guaranteed by _match_matmul_like's transA=0 check
        codebook_name = _find_llm_fp4_weight_codebook(
            w_name, producer_map, initializer_map
        )
        if codebook_name is None:
            continue
        candidates.append((node, x_name, w_name, codebook_name))

    if not candidates:
        return out

    axes_last_name = _unique_name("llmfp4act_axes_last", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([-1], dtype=np.int64), name=axes_last_name
        )
    )
    eps_name = _unique_name("llmfp4act_eps", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
    )

    codebook_max_names: Dict[str, str] = {}  # codebook initializer name -> maxabs const

    for node, x_name, w_name, codebook_name in candidates:
        if codebook_name not in codebook_max_names:
            codebook_values = onnx.numpy_helper.to_array(initializer_map[codebook_name])
            codebook_max = float(np.max(np.abs(codebook_values)))
            max_name = _unique_name(f"{codebook_name}_maxabs", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.array(codebook_max, dtype=np.float32), name=max_name
                )
            )
            codebook_max_names[codebook_name] = max_name
        codebook_max_name = codebook_max_names[codebook_name]

        prefix = f"{w_name}_llmfp4act"
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

        # Per-token scale: max(|x|) over the token's own last axis, floored
        # by epsilon, normalized by the codebook's own max magnitude (the
        # FP4 analogue of onnxsim.zeroquant's/onnxsim.quarot's own
        # "scale = max(|x|) / <format max>" per-token quantizer).
        x_abs = _new("Abs", [x_name], "x_abs")
        x_max = _new("ReduceMax", [x_abs, axes_last_name], "x_max", keepdims=1)
        x_safe_max = _new("Max", [x_max, eps_name], "x_safe_max")
        x_scale = _new("Div", [x_safe_max, codebook_max_name], "x_scale")
        x_norm = _new("Div", [x_name, x_scale], "x_norm")

        # Nearest-codebook-value lookup: broadcast every normalized element
        # against all 16 codebook entries, ArgMin the distances, Gather the
        # codebook back -- the runtime-ops analogue of this module's own
        # (offline, numpy) nearest-codebook search in
        # _search_llm_fp4_blockwise above.
        x_norm_unsq = _new("Unsqueeze", [x_norm, axes_last_name], "x_norm_unsq")
        diff = _new("Sub", [x_norm_unsq, codebook_name], "diff")
        diff_abs = _new("Abs", [diff], "diff_abs")
        nearest_idx = _new("ArgMin", [diff_abs], "nearest_idx", axis=-1, keepdims=0)
        nearest_val = _new(
            "Gather", [codebook_name, nearest_idx], "nearest_val", axis=0
        )
        x_dequant = _new("Mul", [nearest_val, x_scale], "x_dequant")

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(insertion_point + offset, new_node)

        node.input[0] = x_dequant

    return out


def apply_llm_fp4_activation_quantization_per_tensor(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    clip_ratios: Optional[Sequence[float]] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Completes W4A4 for every layer :func:`quantize_weight_only_llm_fp4`
    has already weight-quantized, by fitting **one real-valued per-tensor
    scale** to that layer's own activation from calibration data and baking
    it into the graph as a **constant initializer**, then inserting a
    *static* FP4 quantize/dequantize round-trip against that same layer's
    own (byte-identical, recovered -- not re-derived) codebook.

    This is the **calibrated per-tensor** half of the LLM-FP4 paper's own
    activation-quantization design -- the quantizer "pre-shifted exponent
    bias" is built for. The paper's per-tensor FP4 activation quantizer
    works because a *single* shared exponent bias (here: a single
    real-valued per-tensor scale -- the same degree of freedom, see this
    module's own docstring) suffices once the activation's per-channel
    outliers have been migrated away, and it fits that bias offline from
    calibration data rather than recomputing a range at graph-run time.
    This function does exactly that fitting and that insertion, reusing the
    weight side's own search: :func:`_search_fp4_clip_ratio` scans the same
    ``scale = max_abs * ratio / max(abs(codebook))`` clip-ratio grid against
    the same codebook-round-trip MSE objective
    :func:`_search_llm_fp4_blockwise` already uses per weight block, just
    with the whole captured activation as a single group.

    **Honesty note -- this implements the quantizer, NOT the migration;
    read before using.** The paper's activation scheme is *two* pieces, and
    this function is only the second one:

    1. **Per-channel outlier migration** -- compute a per-channel
       real-valued scale from calibration data and push it algebraically
       into the preceding weight or LayerNormalization, so that what
       reaches this layer no longer carries per-channel outliers and a
       single shared scale/exponent bias actually suffices. This function
       does **not** do this, and does not check whether it has been done.
    2. **The per-tensor FP4 quantizer itself** -- this function.

    Piece 1 already exists in this repo, as
    :func:`onnxsim.apply_smoothquant` and
    :func:`onnxsim.apply_outlier_suppression` (their INT8 migration algebra
    is exactly the migration the paper describes, with FP4's real-valued
    scale-as-bias standing in for those modules' INT8 quantization range),
    so the paper's actual pipeline is a **three-call sequence** the caller
    composes, migration first::

        import onnxsim

        # 1. migrate the per-channel activation outliers into the preceding
        #    weight/LayerNormalization (apply_outlier_suppression is the
        #    alternative)
        migrated = onnxsim.apply_smoothquant(model)
        # 2. weight-side FP4, with the paper's own (format, scale) search
        weight_q = onnxsim.quantize_weight_only_llm_fp4(migrated)
        # 3. this function: one calibrated, constant per-tensor FP4 scale
        #    per activation
        w4a4 = onnxsim.apply_llm_fp4_activation_quantization_per_tensor(
            weight_q
        )

    Running this function *without* step 1 is supported and produces a
    valid model, but it is then quantizing an unmigrated activation with a
    single shared scale -- precisely the case the paper's migration exists
    to avoid, and on outlier-heavy transformer activations it will be
    noticeably worse than the paper's own reported numbers.

    **How this differs from
    :func:`apply_llm_fp4_activation_quantization`** (the per-token pass in
    this same module):

    * *Scale source*: calibration data, fit offline -- versus data-free,
      recomputed from the tensor's own values at graph-run time.
    * *Granularity*: one scale for the whole activation tensor -- versus
      one scale per token (per row of the last axis).
    * *Where the scale lives*: a constant initializer -- versus runtime
      ``Abs``/``ReduceMax``/``Max`` nodes.
    * *Runtime cost*: 7 nodes per layer (``Div``, ``Unsqueeze``, ``Sub``,
      ``Abs``, ``ArgMin``, ``Gather``, ``Mul``) -- versus 11 for the
      per-token pass (those same 7, plus ``Abs``, ``ReduceMax``, ``Max``
      and a second ``Div`` to derive the scale). Strictly fewer runtime
      ops is the practical point of the per-tensor design. Note node count
      is not the whole cost: **both** passes share the
      ``Unsqueeze``/``Sub``/``Abs``/``ArgMin`` nearest-codebook broadcast,
      whose intermediates are 16x the activation's own size while it
      executes -- inherent to a non-uniform codebook, and unchanged by
      moving the scale to a constant. See
      :func:`apply_llm_fp4_activation_quantization`'s own "Runtime cost"
      note.
    * *Paper fidelity*: this is the paper's own quantizer (its migration
      half being the caller's job, above) -- the per-token pass is a
      simpler repo-convention alternative
      (:mod:`onnxsim.zeroquant`/:mod:`onnxsim.quarot`/
      :mod:`onnxsim.duquant` all quantize activations per-token) that is
      explicitly *not* the paper's design.

    Neither pass emits a native FP4 tensor (ONNX has no FP4 type): both are
    simulated ("fake") quantization -- values are snapped onto the FP4
    codebook grid and immediately dequantized back to float32, so the
    ``MatMul`` still runs in float. That is the same simulation
    :mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`/
    :mod:`onnxsim.zeroquant` already use, and it measures the accuracy
    effect of FP4 activations without needing runtime FP4 kernels.

    For each layer already matched by :func:`quantize_weight_only_llm_fp4`
    (found by walking its weight input backward through that function's
    exact dequantization pattern -- see
    :func:`_find_llm_fp4_weight_codebook` -- so a layer this module didn't
    itself weight-quantize is left completely untouched), the activation
    ``X`` is quantized as::

        scale   -- constant float32 initializer, fit offline (see below)
        x_normalized = X / scale
        nearest = Gather(Codebook, ArgMin(Abs(Unsqueeze(x_normalized, -1)
                                              - Codebook), axis=-1))
        x_dequant = nearest * scale

    -- the same ``Unsqueeze``/``Sub``/``Abs``/``ArgMin``/``Gather``
    nearest-codebook construction the per-token pass already uses (and
    which was verified against a hand-computed numpy reference through
    ``onnxruntime``; see ``tests/test_llm_fp4.py``), but with a
    compile-time-constant ``scale``, so **no** ``Abs``/``ReduceMax``/
    ``Max`` range-reduction node is emitted at all.

    ``scale`` itself is fit by capturing ``X`` over ``calibration_data``
    (the ``_add_probe_outputs`` + :func:`onnxsim.backend.run_model` capture
    every calibration-driven pass in this repo uses -- see
    :func:`onnxsim.apply_gptq`), concatenating every captured batch into
    one flat sample, and grid-searching ``clip_ratios`` against that
    sample's own codebook round-trip MSE. The FP4 *format* is **not**
    re-searched: it was already fixed for this layer by
    :func:`quantize_weight_only_llm_fp4`'s own per-weight-tensor search,
    and the recovered codebook is exactly what encodes that choice.

    :param model: a weight-quantized onnx ModelProto or file path -- must
            already have been passed through
            :func:`quantize_weight_only_llm_fp4` for this function to do
            anything, since it only recognizes that function's own exact
            weight-dequantization pattern
    :param calibration_data: representative input batches to fit each
            activation's scale on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative activation range than random input)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param clip_ratios: per-tensor clip-ratio candidates to grid-search --
            ``1.0`` puts the activation's own observed max-abs value
            exactly at the codebook's largest magnitude (no clipping),
            values below ``1.0`` trade a harder clip on outliers for
            sharper resolution on the bulk. Defaults to the weight side's
            own default grid: 17 points evenly spaced over ``[0.5, 1.0]``
            (:func:`quantize_weight_only_llm_fp4`'s own ``min_clip_ratio``/
            ``num_scale_candidates`` defaults)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every already-FP4-weight-quantized layer's
            activation input replaced by a static, per-tensor FP4
            quantize/dequantize round-trip driven by a constant scale
            initializer; every other input (weight, bias) and the node's
            own output name are left exactly as
            :func:`quantize_weight_only_llm_fp4` left them. A layer whose
            weight input isn't fed by exactly that function's own
            dequantization pattern, or whose activation no calibration
            batch reached (or for which every captured value was zero or
            non-finite), is left completely untouched -- this function
            never silently falls back to a different scale scheme under its
            own name. A model with no such layer, or with an opset older
            than 13, is returned unchanged. (Opset 13, not the per-token
            pass's 18: the ops emitted here are ``Unsqueeze`` -- whose
            ``axes``-as-input form is what needs 13 -- plus ``ArgMin``/
            ``Gather``/``Sub``/``Abs``/``Div``/``Mul``, all older. The
            per-token pass needs 18 only for ``ReduceMax``'s own
            ``axes``-as-input form, and this pass emits no ``ReduceMax``.)
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 13):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    producer_map = {output: node for node in graph.node for output in node.output}
    taken_names = _all_names(graph)

    candidates = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        w_name, _weight_transposed = match
        x_name = node.input[0]  # guaranteed by _match_matmul_like's transA=0 check
        codebook_name = _find_llm_fp4_weight_codebook(
            w_name, producer_map, initializer_map
        )
        if codebook_name is None:
            continue
        candidates.append((node, x_name, w_name, codebook_name))

    if not candidates:
        return out

    if clip_ratios is None:
        ratios = np.linspace(0.5, 1.0, 17)
    else:
        ratios = np.asarray(list(clip_ratios), dtype=np.float64)
    if ratios.size == 0:
        raise ValueError("clip_ratios must be non-empty")

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    probe_names = sorted({x_name for _node, x_name, _w, _cb in candidates})
    probe_model = _add_probe_outputs(model, probe_names)
    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        captured = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            value = captured.get(name)
            if value is None:
                continue
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.floating):
                activations[name].append(array.astype(np.float64).ravel())

    # Fit every scale before touching the graph, so a run in which no
    # layer's activation turns out usable leaves `out` byte-identical to
    # `model` rather than half-rewritten.
    fitted: List[Tuple[onnx.NodeProto, str, str, str, float]] = []
    for node, x_name, w_name, codebook_name in candidates:
        samples = [a for a in activations[x_name] if a.size]
        if not samples:
            continue  # no calibration batch reached this activation
        values = np.concatenate(samples)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        max_abs = float(np.abs(values).max())
        if max_abs <= 0.0:
            continue  # an all-zero activation has no meaningful scale

        codebook = onnx.numpy_helper.to_array(initializer_map[codebook_name]).astype(
            np.float64
        )
        # The weight side's own clip-ratio search, with the whole flattened
        # activation as a single group instead of one group per weight
        # block -- that single group's winning scale *is* the per-tensor
        # scale being fit here.
        #
        # _search_fp4_clip_ratio materializes a ``values.size x 16`` float64
        # distance tensor per candidate ratio. On the weight side that is
        # bounded by the weight itself, but here ``values`` is *every*
        # calibration sample concatenated, which is unbounded (more
        # calibration data would mean more memory, up to tens of GB on a
        # realistic model). Fitting a single scalar does not need every
        # sample: ``max_abs`` above is computed exactly over all of them,
        # and the MSE objective is estimated on a large, deterministic
        # random subsample, which bounds the search's peak memory
        # regardless of how much calibration data was supplied.
        fit_values = values
        if fit_values.size > _PER_TENSOR_FIT_MAX_ELEMENTS:
            picks = np.random.default_rng(0).choice(
                fit_values.size, size=_PER_TENSOR_FIT_MAX_ELEMENTS, replace=False
            )
            fit_values = fit_values[picks]
        _error, scale, _codes = _search_fp4_clip_ratio(
            fit_values, np.asarray(max_abs), codebook, ratios
        )
        scale_value = float(scale)
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            continue
        fitted.append((node, x_name, w_name, codebook_name, scale_value))

    if not fitted:
        return out

    axes_last_name = _unique_name("llmfp4act_pt_axes_last", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([-1], dtype=np.int64), name=axes_last_name
        )
    )

    for node, x_name, w_name, codebook_name, scale_value in fitted:
        # w_name (this layer's own dequantized-weight tensor) is unique per
        # layer, unlike x_name -- two layers can share one activation.
        prefix = f"{w_name}_llmfp4act_pt"
        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(scale_value, dtype=np.float32), name=scale_name
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

        # No Abs/ReduceMax/Max anywhere in the scale path: the scale is a
        # compile-time constant, which is the whole practical point of the
        # per-tensor design over the per-token one. The single Abs below is
        # the codebook-distance one, shared with the per-token pass.
        x_norm = _new("Div", [x_name, scale_name], "x_norm")
        x_norm_unsq = _new("Unsqueeze", [x_norm, axes_last_name], "x_norm_unsq")
        diff = _new("Sub", [x_norm_unsq, codebook_name], "diff")
        diff_abs = _new("Abs", [diff], "diff_abs")
        nearest_idx = _new("ArgMin", [diff_abs], "nearest_idx", axis=-1, keepdims=0)
        nearest_val = _new(
            "Gather", [codebook_name, nearest_idx], "nearest_val", axis=0
        )
        x_dequant = _new("Mul", [nearest_val, scale_name], "x_dequant")

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(insertion_point + offset, new_node)

        node.input[0] = x_dequant

    return out
