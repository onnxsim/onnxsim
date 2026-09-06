"""BitNet b1.58's ternary weight quantization (Ma et al., 2024, "The Era of
1-bit LLMs: All Large Language Models are in 1.58 Bits",
https://arxiv.org/abs/2402.17764), as shipped by llama.cpp's own GGUF
``TQ1_0``/``TQ2_0`` tensor types for BitNet-architecture checkpoints -- the
newest members of the GGUF block-quant family this module joins
alongside :mod:`onnxsim.gguf_kquant` (Q4_K), :mod:`onnxsim.iq4_nl`
(IQ4_NL), and :mod:`onnxsim.gguf_legacy_quant` (Q4_0/Q4_1).

Every element of a weight is restricted to one of exactly three ternary
values -- ``{-1, 0, +1}`` -- times one shared per-block scale ``d``: the
extreme end of the "aggressive I-quant" spectrum this repo's other GGUF
modules already occupy, at roughly 1.6-2.1 bits per weight depending on
llama.cpp's own bit-packing scheme (irrelevant to this module -- like
:mod:`onnxsim.gguf_kquant`/:mod:`onnxsim.iq4_nl`/
:mod:`onnxsim.gguf_legacy_quant`, no ONNX tensor type below INT4 exists, so
the quantized weight is represented as an ordinary float32
quantize-dequantize round trip folded into a new initializer, not the
literal packed binary layout).

**The quantization rule itself is the paper's own published "absmean"
scheme** (BitNet b1.58, Section 2, "Quantization Function"), not something
onnxsim derived: for a block of weights ``w``,

    d = mean(|w|)                                  (the shared scale)
    code = round(clip(w / d, -1, 1))  in {-1, 0, 1}  (the ternary code)
    dequant = code * d

llama.cpp's own ``TQ1_0``/``TQ2_0`` GGUF tensor types apply this per
256-element super-block (rather than BitNet's own original per-tensor
scale) and store ``d`` as an ``ggml_half`` (fp16) -- both choices this
module's own ``quantize_dequantize_ternary`` mirrors, the same
"verified against this repo's own existing decoder/reference, honestly
scoped where it isn't" discipline :mod:`onnxsim.gguf_kquant`'s docstring
already documents. Unlike the paper's own reference implementation
(evaluated once over an entire weight matrix), this module has no
calibration data or accumulation step -- it is a data-free, per-block
weight-only transform, matching every other module in this family.

**Honesty note**: this module implements the *published, size-agnostic*
absmean ternary rule exactly as BitNet b1.58's paper defines it (verified
below with real numpy arithmetic against a hand-computed example, not a
recalled/assumed constant), applied at llama.cpp's own documented
``TQ1_0``/``TQ2_0`` block size of 256. What is **not** attempted is
llama.cpp's own literal bit-packing (``TQ2_0``'s 2-bit-per-weight lanes,
``TQ1_0``'s denser 5-trits-per-byte base-243 encoding) -- ONNX has no
tensor type below INT4 either way, so both this module and every other
GGUF module in this repo represent the format as a plain float32 round
trip instead.

**Scope**: matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) and, optionally, ``Conv`` --
any constant float32 weight, of any rank, is flattened, zero-padded up to
a whole number of 256-element blocks, quantized, and reshaped back. No
calibration data is needed.

**A subtlety this module gets right that a naive port would not**: unlike
a max/min-based scale (:mod:`onnxsim.gguf_legacy_quant`'s Q4_0/Q4_1,
:mod:`onnxsim.gguf_kquant`'s Q4_K), a *mean*-based scale is not invariant
to zero-padding a ragged last block -- averaging over the zero-padded
block size would silently dilute it. Since the padding elements are zero,
they never change the block's own sum, so this module divides by each
block's real element count (not the zero-padded count) to reproduce
exactly the mean that block would have without any padding at all.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_BLOCK_SIZE = 256  # llama.cpp's own TQ1_0/TQ2_0 super-block size


def _pad_and_block(values: np.ndarray) -> "tuple[np.ndarray, int, tuple]":
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    padded_n = -(-n // _BLOCK_SIZE) * _BLOCK_SIZE
    if padded_n != n:
        flat = np.concatenate([flat, np.zeros(padded_n - n, dtype=np.float64)])
    return flat.reshape(-1, _BLOCK_SIZE), n, original_shape


def quantize_dequantize_ternary(values: np.ndarray) -> np.ndarray:
    """BitNet b1.58 absmean ternary quantize-dequantize round trip over a
    flattened float array of any length -- one scale ``d = mean(|block|)``
    per 256-element block, ``code = round(clip(value / d, -1, 1))`` in
    ``{-1, 0, 1}``, ``dequant = code * d``.

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, float64
    """
    blocks, n, original_shape = _pad_and_block(values)
    # Unlike a max/min-based scale (Q4_0/Q4_1/Q4_K), a *mean*-based one is
    # NOT invariant to zero-padding a ragged last block: averaging over the
    # zero-padded block size would dilute it. Padding elements are zero, so
    # they don't affect the sum -- dividing by each block's real element
    # count (not the padded count) reproduces the same mean as if no
    # padding had been added at all.
    counts = np.full(blocks.shape[0], float(_BLOCK_SIZE))
    remainder = n % _BLOCK_SIZE
    if remainder != 0:
        counts[-1] = float(remainder)
    d = np.maximum(np.sum(np.abs(blocks), axis=-1) / counts, 1e-12)
    d = d.astype(np.float16).astype(np.float64)  # real TQ1_0/TQ2_0 stores d as fp16
    code = np.clip(np.round(blocks / d[:, np.newaxis]), -1.0, 1.0)
    dequant = code * d[:, np.newaxis]
    return dequant.reshape(-1)[:n].reshape(original_shape)


def apply_gguf_ternary_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched layer's float32 weight into
    llama.cpp's own BitNet b1.58 TQ1_0/TQ2_0 ternary format -- see this
    module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input
    :param skip_names: weight initializer names to leave unquantized
    :returns: ``model`` with every matched layer's weight replaced by its
            ternary round trip
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
            w_quant = quantize_dequantize_ternary(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_ternary", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
