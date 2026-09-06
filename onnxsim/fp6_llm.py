"""FP6-LLM (Xia et al., 2024, "FP6-LLM: Efficiently Serving Large Language
Models Through FP6-Centric Algorithm-System Co-Design",
https://arxiv.org/abs/2401.14112) -- weight-only quantization of every
matched layer's float32 weight into a 6-bit floating-point format, one
scale per block. The paper's own central finding is that FP6 sits at a
sweet spot INT4-style formats miss: narrow enough (6 bits) to roughly
halve memory versus FP8/INT8 while, being floating-point rather than a
fixed integer grid, representing the long-tailed, non-uniform distribution
of transformer weights markedly better than a 4-bit integer grid at a
comparable bit budget -- no calibration data needed, unlike GPTQ/AWQ.

The paper's own system contribution -- a custom "unified" GPU kernel
(``TC-FP6``) that unpacks FP6 into FP16 on the fly inside the GEMM's
inner loop, letting FP6-packed weights and FP16 activations use the same
tensor-core FP16xFP16 matmul path without an separate dequantize-then-copy
pass -- is a Tensor Core scheduling technique with no ONNX equivalent
(there is no ONNX tensor type below INT4 either way), so, like every other
sub-8-bit format in this repo (:mod:`onnxsim.gguf_kquant`,
:mod:`onnxsim.iq4_nl`, :mod:`onnxsim.deepseek_fp8`, ...), this module
represents only the *numerical* side: an ordinary float32
quantize-dequantize round trip folded into a new initializer.

**The FP6 format itself is not something onnxsim derived**: it is cast
through ``ml_dtypes.float6_e3m2fn``/``float6_e2m3fn`` -- the same
``ml_dtypes`` library ``onnx.helper.tensor_dtype_to_np_dtype`` already
uses internally for ONNX's own ``FLOAT8E4M3FN`` (see
:mod:`onnxsim.deepseek_fp8`'s own docstring for that precedent), not a
new/unverified dependency. The paper's own primary format is **E3M2**
(3 exponent bits, 2 mantissa bits, matching this module's default) for
weights; **E2M3** (2 exponent bits, 3 mantissa bits) is its secondary
format, offered here as an option. Both dtypes' representable range was
verified empirically with real numpy arithmetic (a dense sweep, not a
recalled constant) rather than assumed -- see ``tests/test_fp6_llm.py``'s
own reproduction of that sweep, which cross-checks the module's own
hardcoded ``_FP6_MAX`` constants: E3M2's largest finite magnitude is
``28.0``, E2M3's is ``7.5``.

**Why a per-block scale is required at all** (unlike
:mod:`onnxsim.quantize_fp8`'s -- if this repo had one -- unscaled FP8
cast): FP6's exponent range is far narrower than FP8's, so casting a raw
transformer weight straight to FP6 would silently clip most of its
dynamic range to ``_FP6_MAX`` or flush it to zero. Rescaling each block by
its own ``max(|block|) / _FP6_MAX`` first (the same "scale to fill the
target format's own representable range" idea
:mod:`onnxsim.deepseek_fp8`'s block scaling and
:mod:`onnxsim.gguf_legacy_quant`'s Q4_0 both already use) keeps every
block's own largest-magnitude element right at the edge of FP6's
representable range instead.

**Scope**: matches ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) and, optionally, ``Conv`` --
any constant float32 weight, of any rank, is flattened, zero-padded up to
a whole number of blocks, quantized, and reshaped back. No calibration
data is needed.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import ml_dtypes
import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_BLOCK_SIZE = 64

# Largest finite magnitude each format can represent -- verified
# empirically (see this module's own docstring and
# tests/test_fp6_llm.py's reproduction of the sweep that found these),
# not recalled/assumed constants.
_FP6_MAX = {
    "e3m2": 28.0,
    "e2m3": 7.5,
}
_FP6_DTYPE = {
    "e3m2": ml_dtypes.float6_e3m2fn,
    "e2m3": ml_dtypes.float6_e2m3fn,
}


def _pad_and_block(values: np.ndarray) -> "tuple[np.ndarray, int, tuple]":
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    padded_n = -(-n // _BLOCK_SIZE) * _BLOCK_SIZE
    if padded_n != n:
        flat = np.concatenate([flat, np.zeros(padded_n - n, dtype=np.float64)])
    return flat.reshape(-1, _BLOCK_SIZE), n, original_shape


def quantize_dequantize_fp6(values: np.ndarray, fmt: str = "e3m2") -> np.ndarray:
    """FP6-LLM quantize-dequantize round trip over a flattened float array
    of any length -- one scale ``s = max(|block|) / _FP6_MAX[fmt]`` per
    64-element block, ``dequant = round_to_fp6(value / s) * s``.

    :param values: any-shape float array; flattened internally
    :param fmt: ``"e3m2"`` (the paper's primary weight format) or
            ``"e2m3"`` (its secondary format)
    :returns: same shape as ``values``, float64
    """
    if fmt not in _FP6_MAX:
        raise ValueError(
            f"Unknown FP6 format {fmt!r}; expected one of {sorted(_FP6_MAX)}"
        )
    fp6_max = _FP6_MAX[fmt]
    dtype = _FP6_DTYPE[fmt]

    blocks, n, original_shape = _pad_and_block(values)
    scale = np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12) / fp6_max
    scaled = blocks / scale[:, np.newaxis]
    quantized = scaled.astype(np.float32).astype(dtype).astype(np.float64)
    dequant = quantized * scale[:, np.newaxis]
    return dequant.reshape(-1)[:n].reshape(original_shape)


def apply_fp6_llm_quantization(
    model: Union[str, onnx.ModelProto],
    fmt: str = "e3m2",
    include_conv: bool = True,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched layer's float32 weight into
    FP6-LLM's 6-bit floating-point format -- see this module's own
    docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param fmt: ``"e3m2"`` (default, the paper's primary weight format) or
            ``"e2m3"``
    :param include_conv: also quantize ``Conv``'s weight input
    :param skip_names: weight initializer names to leave unquantized
    :returns: ``model`` with every matched layer's weight replaced by its
            FP6 round trip
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
            w_quant = quantize_dequantize_fp6(w, fmt).astype(np.float32)

            new_name = _unique_name(f"{w_name}_fp6_{fmt}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant, name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
