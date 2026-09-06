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

**Update: a self-contained activation-side pass now exists, but it is NOT
the paper's own per-tensor-via-migration design.** The "deliberately not
ported" section above still accurately describes the real LLM-FP4 paper's
own activation-quantization scheme: a per-*channel* real-valued scale fit
from calibration data and migrated into the preceding weight/
LayerNormalization (via :func:`onnxsim.apply_smoothquant`/:func:`onnxsim.
apply_outlier_suppression`) so that a single shared exponent bias suffices
for a per-*tensor* FP4 activation quantizer -- that migration-based design
is still not implemented here. :func:`apply_llm_fp4_activation_quantization`
below instead completes W4A4 with a *different*, simpler granularity: a
per-*token* (per-row), data-free scale computed fresh from each token's own
values at graph-run time -- exactly the convention every other per-token
runtime quantizer in this repo already uses (:mod:`onnxsim.zeroquant`,
:mod:`onnxsim.quarot`, :mod:`onnxsim.duquant`, :mod:`onnxsim.
attention_quantization`, :mod:`onnxsim.kv_cache_quantization`'s Value-style
rewrite), just against FP4's non-uniform 16-value codebook instead of a
uniform integer grid, and needing no cross-layer migration pass first. See
that function's own docstring for the full honesty note on this
difference.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name

# Every way to split FP4's 3 non-sign bits between exponent and mantissa --
# the paper's own candidate set for its per-tensor format search. Named
# "eXmY" for X exponent bits, Y mantissa bits (X + Y == 3 always, since FP4
# is 1 sign bit + 3 remaining bits).
FP4_FORMATS: Dict[str, Tuple[int, int]] = {
    "e1m2": (1, 2),
    "e2m1": (2, 1),  # MXFP4's own element format (onnxsim.mx_quantization)
    "e3m0": (3, 0),
}


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
        max_mag = codebook[-1]

        fmt_best_error = np.full((n, num_blocks), np.inf)
        fmt_best_scale = np.zeros((n, num_blocks))
        fmt_best_codes = np.zeros((n, num_blocks, block_size), dtype=np.int64)

        for r in clip_ratios:
            # The "pre-shifted exponent bias" search, realized as a
            # real-valued per-block scale: r < 1 clips outliers harder but
            # sharpens resolution for the bulk of the block, exactly the
            # clip-vs-resolution trade _mse_threshold's own cutoff search
            # makes for INT8 ranges.
            scale = np.maximum(max_abs * r / max_mag, 1e-30)  # [N, num_blocks]
            normalized = blocks / scale[:, :, np.newaxis]
            diffs = np.abs(normalized[..., np.newaxis] - codebook)
            codes = np.argmin(diffs, axis=-1)  # [N, num_blocks, block_size]
            dequant_normalized = codebook[codes]
            error = (
                np.sum((dequant_normalized - normalized) ** 2, axis=-1) * scale**2
            )  # [N, num_blocks], in the original (unnormalized) units

            improved = error < fmt_best_error
            fmt_best_error = np.where(improved, error, fmt_best_error)
            fmt_best_scale = np.where(improved, scale, fmt_best_scale)
            fmt_best_codes = np.where(improved[:, :, np.newaxis], codes, fmt_best_codes)

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
    docstring for the technique and its scope (weight-only; W4A4 activation
    quantization is out of scope, but composes with this repo's existing
    :func:`onnxsim.apply_smoothquant`/:func:`onnxsim.apply_outlier_suppression`
    migrations, run first). Needs no calibration data: both the per-tensor
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
    section. This function does **not** reproduce that design: it computes
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
