"""llama.cpp's GGUF "Q8_0" block quant format -- the simplest, and the
highest-precision, of every GGUF quant format this repo covers.

A Q8_0 block is a single flat 32-element block sharing one fp16 scale
``d``, with each element's code a *full signed 8-bit integer*
(``[-128, 127]``, real two's complement): ``dequant = code * d``. There
is no bias (unlike :mod:`onnxsim.gguf_legacy_quant`'s own Q4_0, whose
unsigned 4-bit code carries a fixed ``-8`` offset), no explicit min
(unlike that module's Q4_1), no sub-block requantization, no codebook,
and -- since the code is already a whole byte -- no bit-packing at all.
On disk a block is literally 2 bytes of float16 scale followed by 32 raw
signed bytes. Structurally this is closer to a trivial per-32-element
INT8 round trip than to the bit-packed 4/5-bit legacy family.

onnxsim already *reads* the format -- see ``onnxsim/ggml_kquant.h``'s
``DequantizeQ8_0Block``, the verified decoder this module's own
dequantization math is deliberately kept consistent with. Note that the
decoder lives in the *K-quant* header rather than
``onnxsim/ggml_legacy_quant.h``: onnxsim's own dtype mapping classifies
Q8_0 alongside the K-quant family (see ``onnxsim/gguf_dtype.h``'s own
comment on the types "this mapping covers", and its ``IsKQuant``), even
though Q8_0 is structurally a flat 32-element block rather than a
256-element super-block. Readers looking for it among the legacy
Q4_0/Q4_1/Q5_0/Q5_1 encoders (:mod:`onnxsim.gguf_legacy_quant`,
:mod:`onnxsim.gguf_legacy_quant_5bit`) will find only the format family
it resembles, not Q8_0 itself -- hence this separate module.

The result is represented as an ordinary float32 quantize-dequantize
round trip folded directly into a new initializer -- the same
simplification every sibling GGUF encoder here already makes.

**Honesty note**: the *dequantization* formula this encoder targets
(``code * d``) is transcribed directly from, and matches exactly, this
repository's own verified ``onnxsim/ggml_kquant.h`` (itself transcribed
from GGML's own ``ggml-quants.c`` reference -- see that header's own
top-of-file comment). What is **not** independently verified is
llama.cpp's own *encoder* procedure for choosing ``d``
(``quantize_row_q8_0``'s own exact max handling) -- this module's own
encoder uses the straightforward, honestly-scoped choice:
``d = max(abs(block)) / 127``, so the largest-magnitude element of a
block maps to code ``+127`` or ``-127``. A consequence worth stating:
code ``-128`` is technically unreachable by this encoder, since real
two's-complement int8 has the asymmetric range ``[-128, 127]`` while a
magnitude-based scale only ever reaches the symmetric ``[-127, 127]``
subset. That costs at most one code out of 256 and keeps zero mapping
exactly to zero. Not claimed to be a byte-exact reproduction of
llama.cpp's own encoder, only its documented format's own reconstruction
semantics -- verified in this module's own test file with real numpy
arithmetic, not a recalled/assumed constant.

**Scope**: matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) and, optionally, ``Conv`` --
any constant float32 weight, of any rank, is flattened, zero-padded up to
a whole number of 32-element blocks, quantized, and reshaped back. No
calibration data is needed.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_BLOCK_SIZE = 32
_MAX_CODE = 127  # signed 8-bit code range [-128, 127]
_MIN_CODE = -128


def _pad_and_block(values: np.ndarray) -> "tuple[np.ndarray, int, tuple]":
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    padded_n = -(-n // _BLOCK_SIZE) * _BLOCK_SIZE
    if padded_n != n:
        flat = np.concatenate([flat, np.zeros(padded_n - n, dtype=np.float64)])
    return flat.reshape(-1, _BLOCK_SIZE), n, original_shape


def quantize_dequantize_q8_0(values: np.ndarray) -> np.ndarray:
    """Q8_0 quantize-dequantize round trip over a flattened float array of
    any length -- one scale ``d`` per 32-element block,
    ``dequant = code * d`` with the signed ``code`` clamped to
    ``[-128, 127]``.

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, float64
    """
    blocks, n, original_shape = _pad_and_block(values)
    d = np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12) / _MAX_CODE
    d = d.astype(np.float16).astype(np.float64)  # real Q8_0 stores d as fp16
    code = np.clip(np.round(blocks / d[:, np.newaxis]), _MIN_CODE, _MAX_CODE)
    dequant = code * d[:, np.newaxis]
    return dequant.reshape(-1)[:n].reshape(original_shape)


def apply_gguf_q8_0_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched layer's float32 weight into
    llama.cpp's own Q8_0 format -- see this module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input
    :param skip_names: weight initializer names to leave unquantized
    :returns: ``model`` with every matched layer's weight replaced by its
            Q8_0 round trip
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
            w_quant = quantize_dequantize_q8_0(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_q8_0", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
