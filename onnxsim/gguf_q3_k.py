"""llama.cpp's GGUF "Q3_K" K-quant format -- the block layout behind the
``Q3_K_S``/``Q3_K_M``/``Q3_K_L`` quantization presets, the family's
smallest-per-element member (roughly 3.4 bits per weight once its scale
table is amortized). Joins :mod:`onnxsim.gguf_kquant` (Q4_K) and
:mod:`onnxsim.gguf_q6_k` (Q6_K) as this repo's third member of the K-quant
family proper -- distinct from the plainer
:mod:`onnxsim.gguf_legacy_quant` (Q4_0/Q4_1, one flat scale per block, no
super-block) and :mod:`onnxsim.iq4_nl`/:mod:`onnxsim.gguf_ternary_quant`
(a fixed codebook or ternary code, not a re-quantized affine scale).

onnxsim already *reads* Q3_K when importing a GGUF checkpoint -- see
``onnxsim/ggml_kquant.h``'s ``DequantizeQ3_KBlock`` (and its
``UnpackQ3KScales`` helper), which this module's own dequantization math
is deliberately kept consistent with (the same "verified against this
repo's own existing decoder" discipline :mod:`onnxsim.gguf_kquant`'s own
docstring already documents). What has been missing is the encoder
direction.

**Q3_K's block structure**: a 256-element super-block is split into 16
sub-blocks of 16 elements. Each sub-block gets its own 6-bit unsigned
scale code ``sc_j`` (``[0, 63]``), made signed by a fixed ``-32`` offset;
one shared float16 super-block scale ``d_all`` multiplies that offset code
to give the sub-block's own effective scale. Structurally this is Q6_K's
shape -- a *single* shared super-block reference (no separate
``d``/``dmin`` pair as in Q4_K/Q5_K/Q2_K) times a signed per-sub-block
scale code times a signed element code, with no zero-point subtraction
anywhere -- just with a much coarser element code:

    dequant = d_all * (sc_j - 32) * q,   q in [-4, 3],   sc_j in [0, 63]

The element code ``q`` is where Q3_K differs from every sibling: a real
Q3_K block stores 2 low bits per element plus one bit per element in a
separate 32-byte high-bit mask, and ``DequantizeQ3_KBlock`` combines them
as ``low - (hmask_bit_set ? 0 : 4)``, so the effective code range is the
**asymmetric** 8-value set ``{-4, -3, -2, -1, 0, 1, 2, 3}``. That
asymmetry (one more level below zero than above, rather than Q6_K's
symmetric ``[-32, 31]``-style split around it) is the real format's own
documented behaviour, not a transcription error here -- it is exactly what
this repo's own already-verified decoder computes.

**Honesty note**: as with :mod:`onnxsim.gguf_kquant`'s and
:mod:`onnxsim.gguf_q6_k`'s own encoders, this module does **not** reproduce
the real format's on-disk bit-packing -- neither the 12-byte packed
6-bit-scale array (``UnpackQ3KScales``'s own bit-twiddling) nor the
2-bit-plus-separate-high-bit-mask element layout. Following this repo's
established convention for every K-quant encoder, only the *reconstructed
float32 values* are produced, folded directly into a new initializer (ONNX
has no tensor type below INT4, let alone Q3_K's own fractional packed
storage, so there is no packed binary layout to target either way).

What is verified here is the *dequantization* formula above (matching this
repo's own already-verified decoder). llama.cpp's own *encoder*,
``quantize_row_q3_K`` -- its exact procedure for choosing each sub-block's
ideal scale and requantizing it to 6 bits -- was not available to read
alongside the decoder, so the specific per-sub-block-scale/per-element
quantization procedure below is this module's own straightforward,
honestly-scoped design, not a transcription of it. It is "verified" only
in the sense that it reproduces the format's documented reconstruction
*shape*: one shared float16 super-block scale, 16 per-sub-block scale
codes (restricted to non-negative offset codes, see next paragraph), and
the asymmetric 8-value element range -- checked with real numpy arithmetic
in this module's own test file, ``tests/test_gguf_q3_k.py``, not against
recalled or assumed constants.

The encoder, per sub-block ``j`` of 16 elements:
``ideal_scale_j = max(|sub_block_j|) / 4`` (4 being the larger-magnitude
side of the asymmetric ``[-4, 3]`` code range), then per super-block
``d_all = max_j(ideal_scale_j) / 31`` (rounded to float16, as a real
``.gguf`` file stores it) and ``sc_j = 32 + clip(round(ideal_scale_j /
d_all), 0, 31)``. The ``31`` rather than ``63`` is deliberate: like
:mod:`onnxsim.gguf_q6_k`'s own encoder -- which only ever produces
non-negative codes for its own signed scale field -- this encoder restricts
itself to ``sc_j in [32, 63]``, i.e. ``sc_j - 32 in [0, 31]``, so a
sub-block's effective scale is always non-negative. It uses half of the
raw type's range, and is called out here rather than left implicit.

**Scope**: matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) and, optionally, ``Conv`` --
any constant float32 weight, of any rank, is flattened, zero-padded up to
a whole number of 256-element super-blocks, quantized, and reshaped back.
No calibration data is needed.
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
# The real format's own asymmetric 3-bit element code range (2 low bits
# plus one high-mask bit): {-4, ..., 3}, one more level below zero than
# above -- see this module's docstring.
_CODE_MIN = -4
_CODE_MAX = 3
_SCALE_MAGNITUDE = -_CODE_MIN  # 4, the larger-magnitude side of that range
_MAX_SCALE_OFFSET_CODE = 31  # this encoder's own non-negative sc_j - 32 range


def _quantize_dequantize_superblocks(blocks: np.ndarray) -> np.ndarray:
    """``blocks``: float64 ``[num_super_blocks, 16, 16]`` (a whole number
    of 256-element super-blocks, each already split into its 16 sub-blocks
    of 16). Returns the same shape, each element replaced by its Q3_K
    quantize-dequantize round trip.

    Reconstruction formula (per super-block, per sub-block ``j``):
    ``value = d_all * (sc_j - 32) * q`` -- transcribed from
    ``onnxsim/ggml_kquant.h``'s ``DequantizeQ3_KBlock`` (see this module's
    own docstring for exactly what was and wasn't independently verified).
    ``d_all`` is one shared float16 super-block scale; ``sc_j`` is a 6-bit
    scale code, restricted by this encoder to ``[32, 63]`` so the
    sub-block scale ``d_all * (sc_j - 32)`` is never negative; ``q`` is
    the format's own asymmetric element code (``[-4, 3]``).
    """
    ideal_scale = (
        np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12) / _SCALE_MAGNITUDE
    )  # [S, 16]

    d_all = np.maximum(ideal_scale.max(axis=-1), 1e-12) / _MAX_SCALE_OFFSET_CODE  # [S]
    # Real Q3_K stores d_all as ggml_half (float16) -- round-tripping
    # through float16 here reproduces that precision loss too, not just
    # the 6-bit scale code.
    d_all = d_all.astype(np.float16).astype(np.float64)

    scale_offset = np.clip(
        np.round(ideal_scale / d_all[:, np.newaxis]), 0, _MAX_SCALE_OFFSET_CODE
    )  # sc_j - 32, in [0, 31]
    sub_scale = d_all[:, np.newaxis] * scale_offset  # [S, 16], >= 0

    safe_sub_scale = np.where(sub_scale > 0, sub_scale, 1.0)
    q = np.clip(
        np.round(blocks / safe_sub_scale[..., np.newaxis]), _CODE_MIN, _CODE_MAX
    )
    return q * sub_scale[..., np.newaxis]


def quantize_dequantize_q3_k(values: np.ndarray) -> np.ndarray:
    """Q3_K quantize-dequantize round trip over a flattened float array of
    any length -- padded with zeros up to a whole number of 256-element
    super-blocks before quantizing (the padding is quantized along with
    the real data but discarded before returning, so it cannot change any
    real value, only -- very slightly -- a boundary sub-block's own chosen
    scale when ``values.size`` isn't itself a multiple of 16).

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, each element replaced by its Q3_K
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


def apply_gguf_q3_k_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched ``MatMul``/"vanilla" ``Gemm``
    (and, optionally, ``Conv``) float32 weight into llama.cpp's own Q3_K
    K-quant format -- see this module's own docstring for the technique
    and its honestly-scoped encoder. Needs no calibration data: every
    quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input (any rank),
            not just ``MatMul``/``Gemm``'s 2-D one
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by its
            Q3_K quantize-dequantize round-tripped float32 version, same
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
            w_quant = quantize_dequantize_q3_k(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_q3_k", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
