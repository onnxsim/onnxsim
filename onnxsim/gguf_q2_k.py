"""llama.cpp's GGUF "Q2_K" K-quant format -- the *lowest*-precision member
of the K-quant family, the block layout real ``.gguf`` presets (``Q2_K``,
``Q2_K_S``, ...) use for their least output-sensitive tensors, where two
bits per element is all the budget a weight gets. Joins
:mod:`onnxsim.gguf_kquant` (Q4_K) and :mod:`onnxsim.gguf_q6_k` (Q6_K) as
this repo's third member of the K-quant family proper -- distinct from the
plainer :mod:`onnxsim.gguf_legacy_quant` (Q4_0/Q4_1, one flat scale per
block, no super-block) and :mod:`onnxsim.iq4_nl`/
:mod:`onnxsim.gguf_ternary_quant` (a fixed codebook or ternary code, not a
re-quantized affine scale).

onnxsim already *reads* Q2_K when importing a GGUF checkpoint -- see
``onnxsim/ggml_kquant.h``'s ``DequantizeQ2_KBlock``, which this module's
own dequantization math is deliberately kept consistent with (the same
"verified against this repo's own existing decoder" discipline
:mod:`onnxsim.gguf_kquant`'s own docstring already documents). What has
been missing is the encoder direction: nothing in onnxsim could take a
float32 ONNX weight and quantize it *into* that format.

**Q2_K's block structure**: a 256-element super-block is split into **16
sub-blocks of 16** elements (not Q4_K/Q5_K's 8 sub-blocks of 32). Each
sub-block gets its own asymmetric affine quantizer, ``dequant = q *
sub_scale + sub_min``, whose element code ``q`` is only **2 bits**
(``[0, 3]``) -- four reconstruction levels per sub-block, which is why
Q2_K compensates with four times as many sub-blocks as Q4_K. As in Q4_K,
the per-sub-block ``(scale, min)`` pair is itself re-quantized rather
than stored as two float32s: each sub-block's scale and negative-offset
magnitude become **4-bit** codes (``[0, 15]``, packed two-to-a-byte in a
real ``.gguf`` file) relative to one shared pair of super-block-level
float16 reference values (``d``/``dmin``). Per sub-block ``j``:

    value = d * sc_j * q - dmin * m_j,  q in [0, 3],  sc_j, m_j in [0, 15]

-- transcribed directly from, and cross-checked element-for-element
against, ``onnxsim/ggml_kquant.h``'s own ``DequantizeQ2_KBlock`` (whose
``dl = d * (sc & 0xF)`` / ``ml = dmin * (sc >> 4)`` is exactly the
4-bit scale code and 4-bit min code sharing one byte). In the
``q * sub_scale + sub_min`` form above that is ``sub_scale_j = d*sc_j``
and ``sub_min_j = -(dmin*m_j)``.

**Honesty note on exactly how this was verified**: as with
:mod:`onnxsim.gguf_kquant`'s own Q4_K encoder and
:mod:`onnxsim.gguf_q6_k`'s Q6_K one, what is verified here is the
*dequantization* formula above -- it matches this repository's own
already-verified decoder, ``ggml_kquant.h``'s ``DequantizeQ2_KBlock``,
which is itself transcribed from GGML's ``ggml-quants.c`` reference and
is what onnxsim reads real ``.gguf`` files with today. llama.cpp's own
*encoder* (``quantize_row_q2_K``'s exact procedure for choosing each
sub-block's ideal (scale, min) and requantizing them to 4 bits -- e.g.
whether it uses a plain min/max affine fit or a weighted least-squares
search) was **not** available to read alongside that decoder. This
module's encoder therefore uses the same straightforward,
honestly-scoped mechanism :mod:`onnxsim.gguf_kquant`'s Q4_K encoder
already established, adapted to Q2_K's own bit-widths: an ordinary
per-sub-block min/max affine fit, then requantizing each sub-block's
scale/min to a 4-bit code relative to the super-block's own largest
scale/min (so the largest sub-block scale/min maps to the top of the
4-bit range, as a shared-reference-pair scheme requires). It is **not**
claimed to be a byte-exact reproduction of llama.cpp's own encoder --
only of its documented format's own reconstruction semantics, verified
in ``tests/test_gguf_q2_k.py`` with real numpy arithmetic (not
recalled/assumed constants), including that this genuinely coarser
format really does reconstruct measurably worse than Q4_K's own 4-bit
codes on the same weight.

**Scope**: matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`, shared with several other
onnxsim modules) and, optionally, ``Conv`` -- any constant float32
weight, of any rank, is flattened, zero-padded up to a whole number of
256-element super-blocks, quantized, and reshaped back; padding elements
are quantized along with the rest but discarded before reshaping, so they
cannot affect the tensor's real values (only, very slightly, a boundary
sub-block's own chosen scale/min, for tensors whose element count isn't
itself a multiple of 16). No calibration data is needed -- like NF4,
every quantization decision comes from the weight tensor's own values.
As onnxsim has no ONNX tensor type below INT4 -- let alone one for Q2_K's
own packed, fractional (2.625-bit-per-element average) storage -- the
result is represented as an ordinary float32 quantize-dequantize round
trip folded directly into a new initializer, exactly the pattern every
weight-only ``quantize_*``/``apply_*`` function in this repo already uses
(no new graph nodes needed).
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_SUB_BLOCK_SIZE = 16
_SUB_BLOCKS_PER_SUPER_BLOCK = 16
_SUPER_BLOCK_SIZE = _SUB_BLOCK_SIZE * _SUB_BLOCKS_PER_SUPER_BLOCK  # 256
_MAX_CODE = 3  # 2-bit quant code range [0, 3]
_MAX_SUB_SCALE_CODE = 15  # 4-bit sub_scale/sub_min code range [0, 15]


def _quantize_dequantize_superblocks(blocks: np.ndarray) -> np.ndarray:
    """``blocks``: float64 ``[num_super_blocks, 16, 16]`` (a whole number of
    256-element super-blocks, each already split into its 16 sub-blocks of
    16). Returns the same shape, each element replaced by its Q2_K
    quantize-dequantize round trip.

    Reconstruction formula (per super-block, per sub-block ``j``):
    ``value = d * sc_j * q - dmin * m_j`` -- transcribed from
    ``onnxsim/ggml_kquant.h``'s ``DequantizeQ2_KBlock`` (see this module's
    own docstring for exactly what was and wasn't independently verified).
    ``d``/``dmin`` are one shared pair per super-block (rounded to float16
    here, since a real ``.gguf`` file stores them as ``ggml_half``);
    ``sc_j``/``m_j`` are 4-bit codes (``[0, 15]``) approximating sub-block
    ``j``'s own ideal affine scale and negative-offset magnitude relative
    to the largest such value in the super-block; ``q`` is a 2-bit element
    code (``[0, 3]``).
    """
    mins = blocks.min(axis=-1)  # [S, 16]
    maxs = blocks.max(axis=-1)  # [S, 16]
    ideal_scale = np.maximum(maxs - mins, 1e-12) / _MAX_CODE  # [S, 16]
    # Q2_K's affine form only represents a non-positive sub-block offset
    # (sub_min = -(dmin * m), m >= 0) -- see this module's docstring. A
    # sub-block whose true minimum happens to be positive (rare for
    # zero-centered weight tensors) has its offset floored to 0 here rather
    # than represented exactly; this is the one place this mechanism cannot
    # be fully general, called out explicitly rather than silently.
    ideal_neg_offset = np.maximum(-mins, 0.0)  # [S, 16]

    d = np.maximum(ideal_scale.max(axis=-1), 1e-12) / _MAX_SUB_SCALE_CODE  # [S]
    dmin = np.maximum(ideal_neg_offset.max(axis=-1), 1e-12) / _MAX_SUB_SCALE_CODE  # [S]
    # Real Q2_K stores d/dmin as ggml_half (float16) -- round-tripping
    # through float16 here reproduces that precision loss too, not just the
    # 4-bit sub-scale/sub-min codes.
    d = d.astype(np.float16).astype(np.float64)
    dmin = dmin.astype(np.float16).astype(np.float64)

    sc = np.clip(np.round(ideal_scale / d[:, np.newaxis]), 0, _MAX_SUB_SCALE_CODE)
    m = np.clip(
        np.round(ideal_neg_offset / dmin[:, np.newaxis]), 0, _MAX_SUB_SCALE_CODE
    )

    sub_scale = d[:, np.newaxis] * sc  # [S, 16]
    sub_min = -(dmin[:, np.newaxis] * m)  # [S, 16], <= 0

    safe_sub_scale = np.where(sub_scale > 0, sub_scale, 1.0)
    q = np.clip(
        np.round((blocks - sub_min[..., np.newaxis]) / safe_sub_scale[..., np.newaxis]),
        0,
        _MAX_CODE,
    )
    return q * sub_scale[..., np.newaxis] + sub_min[..., np.newaxis]


def quantize_dequantize_q2_k(values: np.ndarray) -> np.ndarray:
    """Q2_K quantize-dequantize round trip over a flattened float array of
    any length -- padded with zeros up to a whole number of 256-element
    super-blocks before quantizing (the padding is quantized along with
    the real data but discarded before returning, so it cannot change any
    real value, only -- very slightly -- a boundary sub-block's own chosen
    scale/min when ``values.size`` isn't itself a multiple of 16).

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, each element replaced by its Q2_K
            round trip, as float64
    """
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    padded_n = -(-n // _SUPER_BLOCK_SIZE) * _SUPER_BLOCK_SIZE
    if padded_n != n:
        flat = np.concatenate([flat, np.zeros(padded_n - n, dtype=np.float64)])
    blocks = flat.reshape(-1, _SUB_BLOCKS_PER_SUPER_BLOCK, _SUB_BLOCK_SIZE)
    dequantized = _quantize_dequantize_superblocks(blocks).reshape(-1)
    return dequantized[:n].reshape(original_shape)


def apply_gguf_q2_k_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched ``MatMul``/"vanilla" ``Gemm``
    (and, optionally, ``Conv``) float32 weight into llama.cpp's own Q2_K
    K-quant format -- see this module's own docstring for the technique
    and its honestly-scoped encoder. Needs no calibration data: every
    quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input (any rank),
            not just ``MatMul``/``Gemm``'s 2-D one
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by its
            Q2_K quantize-dequantize round-tripped float32 version, same
            name and shape as the node's original weight input. Layers
            with a non-constant or non-float32 weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    # MatMul/vanilla-Gemm's weight (via _match_matmul_like) and Conv's
    # weight are both always node.input[1] -- no separate index bookkeeping
    # needed.
    candidates: "list[tuple[onnx.NodeProto, str]]" = []
    for node in graph.node:
        w_name: Optional[str] = None
        match = _match_matmul_like(node)
        if match is not None:
            _x_name, matched_w_name, _bias_name, _weight_transposed = match
            w_name = matched_w_name
        elif include_conv and node.op_type == "Conv" and len(node.input) >= 2:
            w_name = node.input[1]
        if w_name is None or w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if w_init is None or w_init.data_type != onnx.TensorProto.FLOAT:
            continue
        candidates.append((node, w_name))

    quantized_names: "dict[str, str]" = {}
    for node, w_name in candidates:
        new_name = quantized_names.get(w_name)
        if new_name is None:
            w_init = initializer_map[w_name]
            w = onnx.numpy_helper.to_array(w_init)
            w_quant = quantize_dequantize_q2_k(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_q2_k", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
