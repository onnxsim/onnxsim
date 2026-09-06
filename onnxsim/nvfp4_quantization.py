"""NVFP4 (NVIDIA, "Pretraining Large Language Models with NVFP4",
https://arxiv.org/abs/2509.25149, 2025; the same two-level scaling scheme
NVIDIA's own Model-Optimizer implements at
``modelopt/torch/quantization/qtensor/nvfp4_tensor.py`` and Transformer
Engine documents as its ``NVFP4`` recipe). onnxsim ports the *format's*
own definition, not any framework's fitting code -- the same rationale
:mod:`onnxsim.mx_quantization` and :mod:`onnxsim.if4_quantization` already
give for MXFP4/IF4 (a data representation, not an algorithm someone else's
reference implementation could diverge from).

Read :mod:`onnxsim.mx_quantization` first. NVFP4 shares MXFP4's exact 4-bit
element format -- E2M1, the same 16-value codebook (:data:`onnxsim.
mx_quantization.MXFP4_CODEBOOK`) -- but makes two different choices for the
**scale**:

- **Block size 16, not 32.** A finer grain than OCP MX's canonical choice
  (the same tradeoff :mod:`onnxsim.if4_quantization` already notes for its
  own block size).
- **The per-block scale is not a pure power of two.** MXFP4's E8M0 scale
  can only shrink or grow a block by an exact factor of 2, which wastes up
  to 41% of a block's own dynamic range (half a binade, on average) between
  E2M1's coarse codebook and the block's actual magnitude. NVFP4 instead
  stores each block's own scale as an **E4M3** FP8 value (1 sign, 4
  exponent, 3 mantissa bits; max representable magnitude ``448.0``) --
  three more bits than E8M0's mantissa-less exponent-only field buys a much
  closer fit, at the cost of the scale itself needing a real multiply
  instead of an exponent add.

E4M3 (like any float format) still has a bounded dynamic range of its own,
so NVFP4 adds a **second, per-tensor FP32 scale** (``global_scale``) that
the *whole tensor* shares, chosen so the largest block's own scale still
fits inside what E4M3 can represent. Given ``FLOAT8_E4M3_MAX = 448.0`` and
``FLOAT4_E2M1_MAX = 6.0`` (E2M1's own largest representable magnitude, same
constant as :mod:`onnxsim.mx_quantization`'s ``_MXFP4_MAX_MAGNITUDE``):

```
global_scale       = amax(tensor) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
raw_block_scale_i  = amax(block_i) / (global_scale * FLOAT4_E2M1_MAX)
block_scale_i      = round_to_nearest_e4m3(raw_block_scale_i)
dequant(code, i)   = E2M1_CODEBOOK[code] * block_scale_i * global_scale
```

(this is the formula NVIDIA's own Model-Optimizer and Transformer Engine
implementations use; verified against both independently of this repo).
Since ``amax(block_i) <= amax(tensor)``, ``raw_block_scale_i`` is always
``<= FLOAT8_E4M3_MAX``, so it never needs clamping before rounding.

Exactly as :mod:`onnxsim.mx_quantization` stores E8M0's *value* rather than
its raw 8-bit exponent field (ONNX has no E8M0 tensor type to store it in),
this module stores ``block_scale_i * global_scale`` -- already E4M3-rounded,
then combined with the per-tensor scale -- as one plain float32 value per
``(output channel, block)`` group, rather than the on-disk E4M3 byte plus a
separate FP32 scalar. Reconstruction is numerically identical to what a
real two-level NVFP4 dequantize produces (the E4M3 rounding step, which is
where this format's accuracy actually comes from relative to a naive
float32-scaled block, still happens during fitting); only the two-field
on-disk *bit* layout isn't reproduced, same simplification as
:mod:`onnxsim.mx_quantization`/:mod:`onnxsim.nf4`/:mod:`onnxsim.
if4_quantization` already make for their own formats.

Needs no calibration data: the block scale, the global scale, and the
codebook indices all come from the weight's own values.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.mx_quantization import (
    _MXFP4_MAX_MAGNITUDE,
    MXFP4_CODEBOOK,
    _match_matmul_like,
    _nearest_codebook_index,
)

# NVFP4's own reference block size: groups of 16, half OCP MX's canonical
# 32 -- see module docstring.
NVFP4_BLOCK_SIZE = 16

# E2M1's own largest representable magnitude (same value as
# onnxsim.mx_quantization's _MXFP4_MAX_MAGNITUDE, re-exported under NVFP4's
# own naming for readers coming from the NVFP4 literature).
FLOAT4_E2M1_MAX = _MXFP4_MAX_MAGNITUDE

# E4M3 (1 sign, 4 exponent, 3 mantissa bits, the "e4m3fn" variant with no
# infinities and a single NaN pattern) largest representable magnitude.
FLOAT8_E4M3_MAX = 448.0


def _e4m3_positive_grid() -> np.ndarray:
    """Every nonnegative magnitude the OCP E4M3 ("e4m3fn") format can
    represent: 1 sign bit (dropped -- magnitudes only), 4 exponent bits
    (bias 7), 3 mantissa bits. Exponent field ``0`` is subnormal; fields
    ``1..15`` are normal, except the single reserved NaN pattern
    (exponent ``15``, mantissa ``0b111``). 127 distinct magnitudes,
    0.0 to 448.0.
    """
    magnitudes = set()
    for exponent_field in range(16):
        for mantissa in range(8):
            if exponent_field == 15 and mantissa == 7:
                continue  # reserved NaN pattern (S.1111.111)
            if exponent_field == 0:
                magnitude = (mantissa / 8.0) * 2.0 ** (1 - 7)
            else:
                magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent_field - 7)
            magnitudes.add(magnitude)
    return np.asarray(sorted(magnitudes), dtype=np.float64)


_E4M3_POSITIVE_GRID = _e4m3_positive_grid()


def _round_to_e4m3(magnitudes: np.ndarray) -> np.ndarray:
    """Rounds nonnegative magnitudes to the nearest value E4M3 can
    represent, clamping at ``FLOAT8_E4M3_MAX``.
    """
    clamped = np.clip(magnitudes, 0.0, FLOAT8_E4M3_MAX)
    hi_idx = np.clip(
        np.searchsorted(_E4M3_POSITIVE_GRID, clamped), 0, len(_E4M3_POSITIVE_GRID) - 1
    )
    lo_idx = np.clip(hi_idx - 1, 0, len(_E4M3_POSITIVE_GRID) - 1)
    lo, hi = _E4M3_POSITIVE_GRID[lo_idx], _E4M3_POSITIVE_GRID[hi_idx]
    return np.where((hi - clamped) < (clamped - lo), hi, lo)


def _quantize_nvfp4_blockwise(
    w_nk: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Returns ``(codes_nk, effective_scale_blocks, global_scale)`` for
    ``w_nk`` ([N, K], output channel first): E2M1 codebook indices in
    ``[0, 15]``, one *effective* scale (the E4M3-rounded block scale
    already multiplied by the per-tensor global scale -- see module
    docstring) per ``(output channel, block-of-K)`` group, shape
    ``[N, K // block_size]``, and the scalar ``global_scale`` itself.
    Assumes ``K % block_size == 0``.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)

    tensor_amax = max(float(np.abs(w_nk).max()), 1e-30)
    global_scale = tensor_amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)

    block_amax = np.maximum(np.abs(blocks).max(axis=2), 1e-30)  # [N, num_blocks]
    raw_block_scale = block_amax / (global_scale * FLOAT4_E2M1_MAX)
    block_scale = _round_to_e4m3(raw_block_scale)

    effective_scale = block_scale * global_scale  # [N, num_blocks]
    normalized = blocks / effective_scale[:, :, np.newaxis]
    codes = _nearest_codebook_index(normalized)
    return codes.reshape(n, k), effective_scale, global_scale


def quantize_weight_only_nvfp4(
    model: Union[str, onnx.ModelProto],
    block_size: int = NVFP4_BLOCK_SIZE,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into NVFP4 -- see this module's own docstring for the
    format. Needs no calibration data: the codebook, the per-block E4M3
    scale, and the per-tensor global scale all come from the weight's own
    values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) scale group
            along the reduction dimension; NVFP4's own canonical choice is
            16
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Reshape(Gather(codebook, Cast(Wq, INT64)), ...), Ws) ->
            Reshape(..., original shape)`` feeding the original MatMul/Gemm
            node -- ordinary ONNX ops only, no contrib op and no minimum
            opset beyond what ``Gather``/``Cast``/``Reshape``/``Mul``
            themselves need (opset 11+), identical graph shape to
            :func:`onnxsim.mx_quantization.quantize_weight_only_mxfp4`.
            Layers with a non-constant, non-2-D, or non-block-divisible
            weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    codebook_name = None  # created lazily on first match

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

        if codebook_name is None:
            codebook_name = _unique_name("nvfp4_codebook", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.asarray(MXFP4_CODEBOOK, dtype=np.float32), name=codebook_name
                )
            )
        num_blocks = k // block_size

        codes_nk, scale_blocks, _global_scale = _quantize_nvfp4_blockwise(
            w_nk, block_size
        )
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.numpy_helper.from_array(
            codes_orig.astype(np.uint8),
            name=_unique_name(f"{w_name}_nvfp4_q", taken_names),
        )
        graph.initializer.append(wq)
        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_nvfp4_scale", taken_names),
        )
        graph.initializer.append(ws)

        if weight_transposed:
            blocked_shape = [n, num_blocks, block_size]
            scale_shape = [n, num_blocks, 1]
        else:
            blocked_shape = [num_blocks, block_size, n]
            scale_shape = [num_blocks, 1, n]

        cast_out = _unique_name(f"{w_name}_nvfp4_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [wq.name], [cast_out], to=onnx.TensorProto.INT64
        )

        gather_out = _unique_name(f"{w_name}_nvfp4_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather", [codebook_name, cast_out], [gather_out], axis=0
        )

        blocked_shape_name = _unique_name(f"{w_name}_nvfp4_blocked_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(blocked_shape, dtype=np.int64), name=blocked_shape_name
            )
        )
        reshaped_out = _unique_name(f"{w_name}_nvfp4_reshaped", taken_names)
        reshape1_node = onnx.helper.make_node(
            "Reshape", [gather_out, blocked_shape_name], [reshaped_out]
        )

        scale_shape_name = _unique_name(f"{w_name}_nvfp4_scale_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(scale_shape, dtype=np.int64), name=scale_shape_name
            )
        )
        scale_reshaped_out = _unique_name(f"{w_name}_nvfp4_scale_reshaped", taken_names)
        reshape2_node = onnx.helper.make_node(
            "Reshape", [ws.name, scale_shape_name], [scale_reshaped_out]
        )

        scaled_out = _unique_name(f"{w_name}_nvfp4_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul", [reshaped_out, scale_reshaped_out], [scaled_out]
        )

        orig_shape_name = _unique_name(f"{w_name}_nvfp4_orig_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([dim0, dim1], dtype=np.int64), name=orig_shape_name
            )
        )
        dq_out = _unique_name(f"{w_name}_nvfp4_dq", taken_names)
        reshape3_node = onnx.helper.make_node(
            "Reshape",
            [scaled_out, orig_shape_name],
            [dq_out],
            name=_unique_name(f"{w_name}_nvfp4_dequant", taken_names),
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
