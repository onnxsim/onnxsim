"""DeepSeek-V3-style fine-grained block FP8 quantization (DeepSeek-AI,
2024, "DeepSeek-V3 Technical Report", https://arxiv.org/abs/2412.19437,
Section 3.3's low-precision scheme). onnxsim ports the *quantization
scheme* as a post-training pass -- not the paper's own FP8 *training*
recipe, which this module makes no claim to reproduce -- the same
rationale :mod:`onnxsim.awq`/:mod:`onnxsim.gptq` give for porting an
algorithm rather than a framework's training code. This exact scheme
(FP8 E4M3, weight scaled per 128x128 block, activation scaled per-token
per-128-channel-group) is also the serving format both SGLang and vLLM
ship for DeepSeek-V3/R1 checkpoints (commonly labelled ``fp8_w8a8`` /
"DeepSeek block-quantized FP8" in their own docs).

Every FP8 pass already in onnxsim (:func:`onnxsim.quantize_fp8`) is a
whole-tensor, calibration-free, **unscaled** cast -- every value maps
directly onto FP8's own native dynamic range, with no separate scale
factor at all. That is fine for a format with FP8's own wide dynamic
range when the *whole* tensor's values are already reasonably close to
FP8's own representable magnitudes, but a large weight matrix or
activation tensor routinely has very different typical magnitudes in
different regions (different output channels, different token positions)
-- an unscaled cast wastes FP8's scarce 3-bit E4M3 mantissa on whichever
region happens to dominate. DeepSeek-V3's own fix is the opposite of
onnxsim's existing per-tensor `quantize_fp8`: quantize in small, scaled
**blocks**, each with its own scale chosen so that block's own values
actually use FP8's full range, at much finer granularity than a typical
per-tensor or per-channel INT8/FP8 scheme:

- **Weight**: one scale per **128x128 block** (tiling both the reduction
  and output-channel axes) -- finer than :func:`onnxsim.quantize_weight_only_int4`'s
  own 32-element-per-*row* blocks are wide, but tiled in *two* dimensions
  instead of one, matching the paper's own choice for a large GEMM weight.
- **Activation**: one scale per **token, per 128-element channel group**
  (i.e. grouping the feature dimension into contiguous chunks of 128, a
  separate scale for each chunk of each token) -- finer than this
  repo's own existing per-token dynamic INT8 schemes
  (:mod:`onnxsim.quarot`/:mod:`onnxsim.zeroquant`/:mod:`onnxsim.duquant`),
  which use one scale for a whole token's row.

Both operands are quantized via a **real** ONNX ``Cast`` to
``FLOAT8E4M3FN`` and back (opset 19+, when the ONNX ``Cast`` operator
gained float8 support) -- not a manually-simulated round-to-nearest the
way this repo's other float-round-trip passes approximate a bit-width
ONNX has no native tensor type for (:mod:`onnxsim.easyquant`,
:mod:`onnxsim.any_precision_llm`, ...). FP8 E4M3 *is* a native ONNX
tensor type, so the actual hardware rounding behavior is reproduced
exactly by delegating to the same ``Cast`` semantics a real runtime
would use, rather than approximating it.

Per-block/per-group **scale** itself is the ordinary choice for any
floating-point quantization format: ``scale = max(abs(block)) / 448``
(``448`` is E4M3's own largest finite magnitude), floored to avoid a
divide-by-zero on an all-zero block/group. Weight scales are folded
directly into a new float32 initializer (no calibration data needed,
same as every other onnxsim weight-only pass); activation scales are
computed at graph-run time via ``Reshape``/``Abs``/``ReduceMax``, since
the activation is a runtime tensor, not a constant.

**Scope**: only ``MatMul`` and "vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) with a constant 2-D float32
weight are matched; ``Conv`` is left untouched (a scope decision several
other onnxsim modules already make). Needs opset 19+ (``Cast``'s float8
support); an older-opset model is left unchanged.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Union

import ml_dtypes
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.llm_int8 import _match_matmul_like

_FP8_MAX = 448.0  # FLOAT8E4M3FN's own largest finite magnitude


def _fp8_round_trip(values: np.ndarray) -> np.ndarray:
    # The real ONNX FLOAT8E4M3FN cast semantics -- ml_dtypes is the same
    # library onnx.helper.tensor_dtype_to_np_dtype itself maps
    # FLOAT8E4M3FN to internally, not a new/unverified dependency.
    clipped = np.clip(values, -_FP8_MAX, _FP8_MAX)
    return clipped.astype(ml_dtypes.float8_e4m3fn).astype(np.float64)


def quantize_dequantize_block_fp8(
    values: np.ndarray, block_size: int = 128
) -> np.ndarray:
    """FP8 E4M3 quantize-dequantize round trip over a 2-D array, one scale
    per ``block_size`` x ``block_size`` tile (DeepSeek-V3's own weight
    scheme). Padded with zeros up to whole tiles before quantizing (the
    padding is quantized along with the real data but discarded before
    returning, so it cannot change any real value, only -- very slightly
    -- a boundary tile's own chosen scale, when a dimension isn't itself a
    multiple of ``block_size``).

    :param values: 2-D float array
    :param block_size: tile size along both axes
    :returns: same shape as ``values``, float64
    """
    values = np.asarray(values, dtype=np.float64)
    rows, cols = values.shape
    padded_rows = -(-rows // block_size) * block_size
    padded_cols = -(-cols // block_size) * block_size
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    padded[:rows, :cols] = values

    out = np.empty_like(padded)
    for r in range(0, padded_rows, block_size):
        for c in range(0, padded_cols, block_size):
            block = padded[r : r + block_size, c : c + block_size]
            scale = max(float(np.max(np.abs(block))), 1e-12) / _FP8_MAX
            out[r : r + block_size, c : c + block_size] = (
                _fp8_round_trip(block / scale) * scale
            )
    return out[:rows, :cols]


def apply_deepseek_fp8(
    float_model: Union[str, onnx.ModelProto],
    weight_block_size: int = 128,
    act_group_size: int = 128,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """W8A8-quantizes every matched MatMul/"vanilla" Gemm layer to
    DeepSeek-V3's own block-wise FP8 E4M3 scheme -- see this module's own
    docstring for the technique. Needs no calibration data: every weight
    quantization decision comes from the weight tensor's own values, and
    every activation scale is computed fresh at graph-run time.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param weight_block_size: weight tile size along both axes (K and N)
    :param act_group_size: activation channel-group size (one scale per
            token per this many contiguous feature-dimension elements)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``float_model`` with every matched layer's weight replaced
            by its block-FP8 round trip and a Cast-based FP8 round trip
            inserted before its activation input. Layers with a
            non-constant, non-2-D weight are left untouched; a model with
            an opset below 19 is returned unchanged.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(float_model)
    graph = out.graph

    opset_ge_19 = any(
        o.domain in ("", "ai.onnx") and o.version >= 19 for o in out.opset_import
    )
    if not opset_ge_19:
        return out  # Cast's FLOAT8E4M3FN support needs opset >= 19

    initializer_map = {t.name: t for t in graph.initializer}
    candidates = []  # (node, x_name, w_name, weight_transposed)
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, _bias_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_init, weight_transposed))
    if not candidates:
        return out

    taken_names = _all_names(graph)

    fp8_dtype = onnx.TensorProto.FLOAT8E4M3FN

    for node, x_name, w_init, weight_transposed in candidates:
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output-channel first
        w_quant_nk = quantize_dequantize_block_fp8(w_nk, weight_block_size)
        w_quant = w_quant_nk if weight_transposed else w_quant_nk.T

        new_w_name = _unique_name(f"{w_init.name}_deepseek_fp8", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(w_quant.astype(np.float32), name=new_w_name)
        )
        node.input[1] = new_w_name

        prefix = _unique_name(f"{node.output[0]}_deepseek_fp8", taken_names)
        k = w_nk.shape[1]
        num_groups = -(-k // act_group_size)

        new_nodes: List[onnx.NodeProto] = []

        def _op(op_type, inputs, tag, **attrs):
            out_name = _unique_name(f"{prefix}_{tag}", taken_names)
            n = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{tag}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n)
            return out_name

        def _const(value: np.ndarray, tag: str) -> str:
            name = _unique_name(f"{prefix}_{tag}", taken_names)
            graph.initializer.append(onnx.numpy_helper.from_array(value, name=name))
            return name

        # Group the activation's last dimension into `num_groups` chunks of
        # `act_group_size` (zero-padding the last, ragged group is not
        # attempted here -- act_group_size is expected to divide K evenly,
        # documented as a scope decision; a non-dividing K is left
        # unquantized on the activation side, matching this repo's own
        # "decline rather than guess" convention -- checked below).
        if k % act_group_size != 0:
            continue

        shape_grouped = _const(
            np.asarray([-1, num_groups, act_group_size], dtype=np.int64), "shape_grp"
        )
        shape_flat = _const(np.asarray([-1, k], dtype=np.int64), "shape_flat")
        eps_name = _const(np.asarray(1e-12, dtype=np.float32), "eps")
        fp8_max_name = _const(np.asarray(_FP8_MAX, dtype=np.float32), "fp8_max")
        # ReduceMax's `axes` is an input (not an attribute) from opset 18
        # onwards -- this pass requires opset >= 19 already (Cast's float8
        # support), so always use the input form.
        axes_name = _const(np.asarray([-1], dtype=np.int64), "axes")

        grouped = _op("Reshape", [x_name, shape_grouped], "grouped")
        abs_grouped = _op("Abs", [grouped], "abs")
        max_grouped = _op("ReduceMax", [abs_grouped, axes_name], "max", keepdims=1)
        safe_max = _op("Clip", [max_grouped, eps_name], "safe_max")
        act_scale = _op("Div", [safe_max, fp8_max_name], "scale")
        scaled = _op("Div", [grouped, act_scale], "scaled")
        fp8 = _op("Cast", [scaled], "fp8", to=fp8_dtype)
        dequant_fp8 = _op("Cast", [fp8], "dequant_fp8", to=onnx.TensorProto.FLOAT)
        dequant_grouped = _op("Mul", [dequant_fp8, act_scale], "dequant_grouped")
        dequant_flat = _op("Reshape", [dequant_grouped, shape_flat], "dequant_flat")
        node.input[0] = dequant_flat

        insertion_point = next(i for i, n in enumerate(graph.node) if n is node)
        for new_node in new_nodes:
            graph.node.insert(insertion_point, new_node)
            insertion_point += 1

    return out
