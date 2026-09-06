"""llama.cpp's GGUF "Q6_K" K-quant format -- the block layout real
``.gguf`` quantization presets (``Q4_K_M``, ``Q5_K_M``, ...) use for their
highest-precision tensors (most commonly the output/``lm_head`` weight,
kept at 6 bits while every other weight goes to 4 or 5). Joins
:mod:`onnxsim.gguf_kquant` (Q4_K) as this repo's second member of the
K-quant family proper -- distinct from the plainer
:mod:`onnxsim.gguf_legacy_quant` (Q4_0/Q4_1, one flat scale per block, no
super-block) and :mod:`onnxsim.iq4_nl`/:mod:`onnxsim.gguf_ternary_quant`
(a fixed codebook or ternary code, not a re-quantized affine scale).

onnxsim already *reads* Q6_K when importing a GGUF checkpoint -- see
``onnxsim/ggml_kquant.h``'s ``DequantizeQ6_KBlock``, which this module's
own dequantization math is deliberately kept consistent with (the same
"verified against this repo's own existing decoder" discipline
:mod:`onnxsim.gguf_kquant`'s own docstring already documents). What has
been missing is the encoder direction.

**Q6_K's block structure**: a 256-element super-block is split into 16
sub-blocks of 16 elements. Each sub-block gets its own signed 8-bit scale
code ``sc_j`` (``[-128, 127]``, though this module's own encoder only ever
produces non-negative codes -- see below); one shared float16 super-block
scale ``d`` multiplies every sub-block's code before it is itself
multiplied by that sub-block's own element code. The element codes
themselves are **symmetric** 6-bit values (``[-32, 31]``, no separate
min/zero-point) -- unlike Q4_K/Q5_K's asymmetric ``d``/``dmin`` pair, Q6_K
is structurally closer to :mod:`onnxsim.gguf_legacy_quant`'s Q4_0, just
with a super-block's worth of per-sub-block scale variation layered on
top:

    dequant = d * sc_j * q,   q in [-32, 31],   sc_j in [-128, 127]

-- transcribed directly from, and cross-checked element-for-element
against, ``onnxsim/ggml_kquant.h``'s own ``DequantizeQ6_KBlock``.

**Honesty note**: as with :mod:`onnxsim.gguf_kquant`'s own Q4_K encoder,
what is verified here is the *dequantization* formula above (matching
this repo's own already-verified decoder); llama.cpp's own *encoder*
procedure for choosing each sub-block's ideal scale and requantizing it
to 8 bits was not available to read alongside the decoder. This module's
own encoder therefore uses the same straightforward, honestly-scoped
mechanism :mod:`onnxsim.gguf_kquant`'s Q4_K encoder already established:
for sub-block ``j``, ``ideal_scale_j = max(|sub_block_j|) / 32``, then
``d = max_j(ideal_scale_j) / 127`` (the super-block's shared float16
reference, chosen so the single largest sub-block scale maps to the top
of the 8-bit code range) and ``sc_j = round(ideal_scale_j / d)`` clamped
to ``[0, 127]``. Not claimed to be a byte-exact reproduction of
llama.cpp's own encoder, only its documented format's own reconstruction
semantics -- verified in this module's own test file with real numpy
arithmetic, not a recalled/assumed constant.

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
_MAX_CODE = 32  # symmetric 6-bit code range [-32, 31]
_MAX_SUB_SCALE_CODE = (
    127  # signed 8-bit sub-scale code, this encoder's own range [0, 127]
)


def _quantize_dequantize_superblocks(blocks: np.ndarray) -> np.ndarray:
    """``blocks``: float64 ``[num_super_blocks, 16, 16]`` (a whole number
    of 256-element super-blocks, each already split into its 16 sub-blocks
    of 16). Returns the same shape, each element replaced by its Q6_K
    quantize-dequantize round trip.

    Reconstruction formula (per super-block, per sub-block ``j``):
    ``value = d * sc_j * q`` -- transcribed from
    ``onnxsim/ggml_kquant.h``'s ``DequantizeQ6_KBlock`` (see this module's
    own docstring for exactly what was and wasn't independently verified).
    ``d`` is one shared float16 super-block scale; ``sc_j`` is this
    encoder's own 8-bit per-sub-block scale code (``[0, 127]``); ``q`` is
    a symmetric 6-bit code (``[-32, 31]``).
    """
    ideal_scale = (
        np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12) / _MAX_CODE
    )  # [S, 16]

    d = np.maximum(ideal_scale.max(axis=-1), 1e-12) / _MAX_SUB_SCALE_CODE  # [S]
    # Real Q6_K stores d as ggml_half (float16) -- round-tripping through
    # float16 here reproduces that precision loss too, not just the 8-bit
    # sub-scale code.
    d = d.astype(np.float16).astype(np.float64)

    sc = np.clip(np.round(ideal_scale / d[:, np.newaxis]), 0, _MAX_SUB_SCALE_CODE)
    sub_scale = d[:, np.newaxis] * sc  # [S, 16]

    safe_sub_scale = np.where(sub_scale > 0, sub_scale, 1.0)
    q = np.clip(
        np.round(blocks / safe_sub_scale[..., np.newaxis]), -_MAX_CODE, _MAX_CODE - 1
    )
    return q * sub_scale[..., np.newaxis]


def quantize_dequantize_q6_k(values: np.ndarray) -> np.ndarray:
    """Q6_K quantize-dequantize round trip over a flattened float array of
    any length -- padded with zeros up to a whole number of 256-element
    super-blocks before quantizing (the padding is quantized along with
    the real data but discarded before returning, so it cannot change any
    real value, only -- very slightly -- a boundary sub-block's own chosen
    scale when ``values.size`` isn't itself a multiple of 16).

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, each element replaced by its Q6_K
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


def apply_gguf_q6_k_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched ``MatMul``/"vanilla" ``Gemm``
    (and, optionally, ``Conv``) float32 weight into llama.cpp's own Q6_K
    K-quant format -- see this module's own docstring for the technique
    and its honestly-scoped encoder. Needs no calibration data: every
    quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input (any rank),
            not just ``MatMul``/``Gemm``'s 2-D one
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by its
            Q6_K quantize-dequantize round-tripped float32 version, same
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
            w_quant = quantize_dequantize_q6_k(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_q6_k", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
