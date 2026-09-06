"""llama.cpp's GGUF legacy "Q5_0"/"Q5_1" block quant formats -- the
direct 5-bit-code counterparts of :mod:`onnxsim.gguf_legacy_quant`'s own
Q4_0/Q4_1 (same legacy format family, same single 32-element block, same
fp16-stored scale; only the code width differs). onnxsim already *reads*
both formats when importing a GGUF checkpoint -- see
``onnxsim/ggml_legacy_quant.h``'s
``DequantizeQ5_0Block``/``DequantizeQ5_1Block``, which this module's own
dequantization math is deliberately kept consistent with (the same
"verified against this repo's own existing decoder" discipline
:mod:`onnxsim.gguf_legacy_quant`'s own docstring already documents). What
has been missing is the encoder direction.

**Q5_0** (symmetric, no separate min): a single 32-element block shares
one scale ``d``; each element's 5-bit code is unsigned ``[0, 31]`` but
represents a *signed* value via a fixed ``-16`` bias baked into the
format itself (``code - 16`` ranges ``[-16, 15]``), so
``dequant = (code - 16) * d`` -- no explicit zero-point/min stored at
all. Exactly Q4_0's own scheme with one more bit of code resolution.

**Q5_1** (asymmetric, explicit min): the same single 32-element block and
5-bit code, but used unsigned (``[0, 31]``, no bias) with an explicit
per-block additive min ``m`` stored alongside the scale:
``dequant = code * d + m``. Exactly Q4_1's own scheme, one bit wider --
it exists for the same reason, representing non-zero-centered blocks more
accurately than Q5_0's fixed symmetric range can.

Both are represented as an ordinary float32 quantize-dequantize round
trip folded directly into a new initializer -- the same simplification
every sibling legacy-quant module here already makes. In particular the
real on-disk layout's 5th (high) bit, which llama.cpp physically packs
into a separate 4-byte ``qh`` bitfield rather than alongside the other
four bits (see the ``qh`` shifting in the two decoders cited above), is
**not** reproduced: only the reconstructed float32 *values* matter here.
No ONNX tensor type below INT4 exists, so the literal packed
5-bit-plus-fp16-scale binary layout has no native ONNX representation
either way.

**Honesty note**: the *dequantization* formula each encoder here targets
(``(code - 16) * d`` for Q5_0, ``code * d + m`` for Q5_1) is transcribed
directly from, and matches exactly, this repository's own verified
``onnxsim/ggml_legacy_quant.h`` (itself transcribed from GGML's own
``ggml-quants.c`` reference -- see that header's own top-of-file
comment). What is **not** independently verified is llama.cpp's own
*encoder* procedure for choosing ``d``/``m``
(``quantize_row_q5_0``/``quantize_row_q5_1``'s own exact min/max
handling) -- this module's own encoder uses the straightforward,
honestly-scoped choice: for Q5_0, ``d = max(abs(block)) / 16`` (so the
largest-magnitude element maps to code 0 or 31); for Q5_1,
``d = (max(block) - min(block)) / 31`` and ``m = min(block)`` (an
ordinary min/max affine fit). Not claimed to be a byte-exact
reproduction of llama.cpp's own encoder, only its documented format's own
reconstruction semantics -- verified in this module's own test file with
real numpy arithmetic, not a recalled/assumed constant.

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
_MAX_CODE = 31  # 5-bit code range [0, 31]
_Q5_0_BIAS = 16  # code - 16 -> signed [-16, 15]


def _pad_and_block(values: np.ndarray) -> "tuple[np.ndarray, int, tuple]":
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    padded_n = -(-n // _BLOCK_SIZE) * _BLOCK_SIZE
    if padded_n != n:
        flat = np.concatenate([flat, np.zeros(padded_n - n, dtype=np.float64)])
    return flat.reshape(-1, _BLOCK_SIZE), n, original_shape


def quantize_dequantize_q5_0(values: np.ndarray) -> np.ndarray:
    """Q5_0 quantize-dequantize round trip over a flattened float array of
    any length -- one symmetric scale ``d`` per 32-element block,
    ``dequant = (code - 16) * d`` with ``code`` clamped to ``[0, 31]``.

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, float64
    """
    blocks, n, original_shape = _pad_and_block(values)
    d = np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12) / _Q5_0_BIAS
    d = d.astype(np.float16).astype(np.float64)  # real Q5_0 stores d as fp16
    code = np.clip(np.round(blocks / d[:, np.newaxis]) + _Q5_0_BIAS, 0, _MAX_CODE)
    dequant = (code - _Q5_0_BIAS) * d[:, np.newaxis]
    return dequant.reshape(-1)[:n].reshape(original_shape)


def quantize_dequantize_q5_1(values: np.ndarray) -> np.ndarray:
    """Q5_1 quantize-dequantize round trip over a flattened float array of
    any length -- one scale ``d`` and one min ``m`` per 32-element block,
    ``dequant = code * d + m`` with ``code`` clamped to ``[0, 31]``.

    :param values: any-shape float array; flattened internally
    :returns: same shape as ``values``, float64
    """
    blocks, n, original_shape = _pad_and_block(values)
    m = blocks.min(axis=-1)
    d = np.maximum(blocks.max(axis=-1) - m, 1e-12) / _MAX_CODE
    d = d.astype(np.float16).astype(np.float64)
    m = m.astype(np.float16).astype(np.float64)  # real Q5_1 stores both as fp16
    code = np.clip(
        np.round((blocks - m[:, np.newaxis]) / d[:, np.newaxis]), 0, _MAX_CODE
    )
    dequant = code * d[:, np.newaxis] + m[:, np.newaxis]
    return dequant.reshape(-1)[:n].reshape(original_shape)


def _apply_legacy_quant(
    model: Union[str, onnx.ModelProto],
    quant_fn,
    tag: str,
    include_conv: bool,
    skip_names: Optional[Iterable[str]],
) -> onnx.ModelProto:
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
            w_quant = quant_fn(w).astype(np.float32)

            new_name = _unique_name(f"{w_name}_{tag}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out


def apply_gguf_q5_0_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched layer's float32 weight into
    llama.cpp's own Q5_0 legacy format -- see this module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input
    :param skip_names: weight initializer names to leave unquantized
    :returns: ``model`` with every matched layer's weight replaced by its
            Q5_0 round trip
    """
    return _apply_legacy_quant(
        model, quantize_dequantize_q5_0, "q5_0", include_conv, skip_names
    )


def apply_gguf_q5_1_quantization(
    model: Union[str, onnx.ModelProto],
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched layer's float32 weight into
    llama.cpp's own Q5_1 legacy format -- see this module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param include_conv: also quantize ``Conv``'s weight input
    :param skip_names: weight initializer names to leave unquantized
    :returns: ``model`` with every matched layer's weight replaced by its
            Q5_1 round trip
    """
    return _apply_legacy_quant(
        model, quantize_dequantize_q5_1, "q5_1", include_conv, skip_names
    )
