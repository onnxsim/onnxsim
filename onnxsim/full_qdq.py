"""Whole-graph static QDQ quantization for NPU backends (QNN HTP, ...).

:func:`onnxsim.quantize_static` wraps only the *inputs* of MatMul/Gemm/Conv in
QuantizeLinear/DequantizeLinear. That is the right shape for a CPU runtime
that fuses ``DQ -> MatMul`` into an integer kernel, but an NPU compiler such
as Qualcomm's QNN HTP backend only runs a node in integer arithmetic when it
forms a complete *QDQ node unit*: every float input comes from a
``DequantizeLinear`` and every output goes straight into a
``QuantizeLinear``. A Conv whose output stays float (and every Relu/Add/
MaxPool in between) runs in fp16 instead, with conversions on both sides.

:func:`quantize_full_qdq` produces that whole-graph form:

- every float activation touching a quantized node gets a calibrated
  ``Q -> DQ`` pair (uint8 by default, uint16 for ``activation_dtype="uint16"``);
- Conv/ConvTranspose/Gemm/MatMul constant weights become int8, symmetric,
  per output channel; Conv/Gemm biases become int32 with scale
  ``input_scale * weight_scale`` (what integer accumulators need);
- any other constant input of a quantized node is quantized per tensor with
  the activation dtype;
- data-movement ops (Reshape, Transpose, MaxPool, Resize, GridSample, ...)
  reuse their input's quantization parameters for their output, so they are
  exact and need no requantization;
- a Relu right after a quantized producer is folded into that producer's
  output quantization (a uint8 Q with zero point 0 already clamps at 0);
- ``remove_qdq_after`` / ``fold_activation`` / ``adjust_activation_ranges`` /
  ``align_ops`` / ``unshared_ops`` / ``shared_ops`` / ``weight_symmetric`` /
  ``quantize_bias`` reproduce AMD Quark's graph-placement and quantizer
  options (:mod:`onnxsim.quark_compat` sets them from ``extra_options``);
- nodes can be left in float (``op_types`` include list, ``exclude_op_types``,
  ``exclude_nodes``): mixed precision, e.g. keeping LayerNorm/Softmax and
  sampling coordinates in fp16 while the Linear layers run in int8.

:func:`quantized_io` then optionally drops the float boundary: a graph input
becomes the uint8 tensor its ``QuantizeLinear`` produced (the caller
quantizes on the host, typically for free in preprocessing), a graph output
becomes the quantized tensor before its ``DequantizeLinear``.

Calibration reuses :func:`onnxsim.calibration.calibrate` (every calibration
method it supports is available through ``method``).
"""

import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxsim.calibration import Tensors, calibrate, pof2_minmse_weight_scale

__all__ = ["quantize_full_qdq", "quantized_io", "sampling_coordinate_tensors"]

# Output values are a subset / convex combination of the data input's values
# (with 0 for padding, and every range here contains 0), so the output can
# share the input's quantization parameters exactly.
_SHARED_QPARAM_OPS = {
    "Reshape",
    "Transpose",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "MaxPool",
    "Slice",
    "Split",
    "Expand",
    "Tile",
    "Gather",
    "DepthToSpace",
    "SpaceToDepth",
    "Identity",
    "GridSample",
    "Resize",
}

# Inputs that carry tensor data; the rest (shapes, indices, scales, axes, ...)
# are never quantized. Ops not listed: every input is data.
_DATA_INPUTS = {
    "Reshape": (0,),
    "Expand": (0,),
    "Tile": (0,),
    "Slice": (0,),
    "Split": (0,),
    "Squeeze": (0,),
    "Unsqueeze": (0,),
    "Gather": (0,),
    "Resize": (0,),
    "Pad": (0, 2),
    "Clip": (0,),
    "TopK": (0,),
    "ReduceMean": (0,),
    "ReduceSum": (0,),
    "ReduceMax": (0,),
}

# Never quantized: shape arithmetic, existing Q/DQ, control flow.
_NEVER_QUANTIZED = {
    "QuantizeLinear",
    "DequantizeLinear",
    "Shape",
    "Size",
    "Constant",
    "ConstantOfShape",
    "Range",
    "NonZero",
    "Cast",
    "If",
    "Loop",
    "Scan",
    "ArgMax",
    "ArgMin",
}

_WEIGHT_AXIS_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
#: producers whose output Quark leaves un-Q/DQ'd before a Relu-like consumer
QUARK_QDQ_PRODUCERS = (
    "Conv",
    "Add",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "MatMul",
    "Gemm",
    "ConvTranspose",
)
_ELTWISE_OPS = {"Add", "Sub", "Mul", "Div", "Min", "Max"}

_DTYPES = {
    "uint8": (TensorProto.UINT8, np.uint8, 0, 255),
    "uint16": (TensorProto.UINT16, np.uint16, 0, 65535),
    "int8": (TensorProto.INT8, np.int8, -128, 127),
    "int16": (TensorProto.INT16, np.int16, -32768, 32767),
}
# the code ranges ``reduce_range`` leaves weights (Quark's reduced-range table)
_REDUCED_RANGES = {
    "int8": (-64, 64),
    "uint8": (0, 127),
    "int16": (-16384, 16384),
    "uint16": (0, 32767),
}
_WIDE_DTYPES = ("uint16", "int16")  # need com.microsoft Q/DQ below opset 21


def _pof2(scale: float) -> float:
    """The smallest power of two >= ``scale`` (never clips more than ``scale``).
    A scale a rounding error above a power of two (``absmax / 127 * 127``)
    counts as that power of two."""
    return float(2.0 ** np.ceil(np.log2(scale) - 1e-9)) if scale > 0 else scale


# Quark's symmetric integer ranges (``get_qmin_qmax_for_qType(symmetric=True)``)
_QUARK_SYMMETRIC_RANGE = {
    (-128, 127): (-127, 127),
    (-32768, 32767): (-32767, 32767),
}


def _quark_pof2_params(
    lo: float, hi: float, qmin: int, qmax: int, symmetric: bool
) -> Tuple[float, int]:
    """``(scale, zero_point)`` of Quark's ``compute_scale_zp`` with a power-of-two
    method: the min / max scale (float32 range, float64 division) taken to the
    *nearest* fixed-point position -- not rounded up -- with the zero point
    recomputed for it. Ranges the MinMSE search produced come out exact; a range
    set by hand (a Softmax output's ``(0, 1)``) is rounded, not covered."""
    if symmetric:
        qmin, qmax = _QUARK_SYMMETRIC_RANGE.get((qmin, qmax), (qmin, qmax))
    rmin, rmax = np.float32(min(lo, 0.0)), np.float32(max(hi, 0.0))
    if symmetric:
        absmax = np.maximum(np.abs(rmin), np.abs(rmax))
        rmin, rmax = -absmax, absmax
    scale = np.float64(rmax - rmin) / np.float64(qmax - qmin)
    if scale < np.finfo(np.float32).tiny:
        scale32, zp = np.float32(1.0), 0
    else:
        zp = int(np.round(qmin - np.float64(rmin) / scale))
        scale32 = np.float32(scale)
    pos = int(np.rint(-np.log2(min(max(float(scale32), 2.0**-127), 2.0**127))))
    pof2 = np.float32(2.0**-pos)
    new_rmin = np.minimum((np.float32(qmin) - np.float32(zp)) * pof2, np.float32(0))
    new_zp = int(np.round(np.float32(qmin) - new_rmin / pof2))
    if symmetric and qmin == 0 and qmax == 255 and new_zp == 127:
        new_zp = 128  # (the hardware wants the zero point centred)
    return float(pof2), new_zp


def _pof2_minmse_asymmetric(w: np.ndarray) -> Tuple[float, int]:
    """``(scale, zero_point)`` of an int8 weight (or bias) tensor on Quark's
    asymmetric power-of-two MinMSE grid: the zero point of the min / max scale
    taken to the nearest fixed-point position, and, around that position, the
    candidate scale ``2^-(p-1) ... 2^-(p+3)`` with the least squared error *at
    that zero point*; codes are clipped to ``[-127, 127]``."""
    data = np.asarray(w, dtype=np.float32).ravel()
    if not data.size:
        return 1.0, 0
    base, zp = _quark_pof2_params(
        float(data.min()), float(data.max()), -128, 127, False
    )
    pos = int(np.rint(-np.log2(min(max(base, 2.0**-127), 2.0**127))))
    best, best_s = np.float32("inf"), np.float32(base)
    for i in range(5):
        cand = np.float32(2.0 ** -(pos + i - 1))
        q = np.clip(np.round(data / cand) + np.float32(zp), -127, 127)
        diff = np.sum(((q - np.float32(zp)) * cand - data) ** 2, dtype=np.float32)
        if diff < best:
            best, best_s = diff, cand
    return float(best_s), zp


_TINY = float(np.finfo(np.float32).tiny)


def _qparams(
    lo: float,
    hi: float,
    qmin: int,
    qmax: int,
    symmetric: bool = False,
    power_of_two: bool = False,
    quark_rounding: bool = False,
) -> Tuple[float, int]:
    """``(scale, zero_point)`` for the range ``[lo, hi]`` (always including 0).

    ``symmetric``: ``scale = absmax / half-range`` with the zero point in the
    middle of the integer range -- 0 for signed types, 128 / 32768 for unsigned
    ones. ``power_of_two`` rounds the scale up to a power of two;
    ``quark_rounding`` (with it) takes the nearest position instead, as Quark's
    power-of-two ``compute_scale_zp`` does."""
    if power_of_two and quark_rounding:
        return _quark_pof2_params(float(lo), float(hi), qmin, qmax, symmetric)
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if symmetric:
        absmax = max(-lo, hi)
        half = qmax if qmin < 0 else (qmax - qmin) // 2
        if qmin >= 0 and not power_of_two and absmax > 0:
            # Quark's unsigned symmetric grid spans the whole code range
            # (scale 2 * absmax / qmax) with the zero point rounded (half to
            # even, in float32 / float64 like its compute_scale_zp) to the
            # middle; uint8's 127 is bumped to 128
            a32 = np.float32(absmax)
            scale64 = np.float64(a32 + a32) / np.float64(qmax - qmin)
            zp = int(np.round(np.float64(qmin) + np.float64(a32) / scale64))
            if qmax - qmin == 255 and zp == 127:
                zp = 128
            return float(np.float32(scale64)), zp
        scale = absmax / half
        zp = 0 if qmin < 0 else half + 1
        if scale < _TINY:
            return 1.0, 0  # (Quark's compute_scale_zp for an all-zero range)
        return (_pof2(scale) if power_of_two else scale), zp
    # Quark's compute_scale_zp: the range is float32 and its width is taken in
    # float32 before the float64 division
    lo32, hi32 = np.float32(lo), np.float32(hi)
    scale = float(np.float64(hi32 - lo32) / np.float64(qmax - qmin))
    if scale < _TINY:
        return 1.0, 0  # (Quark's compute_scale_zp for an all-zero range)
    if power_of_two:
        scale = _pof2(scale)
    zp = int(np.clip(round(qmin - lo / scale), qmin, qmax))
    return scale, zp


_CLIP_BOUNDS = {(0.0, 6.0), (0.0, 1.0)}


def _clip_bounds(
    n: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[float, float]]:
    """``(min, max)`` of a ``Clip`` with constant, scalar bounds (Quark's
    ``is_clip_with_min_max`` tolerance: 1e-6 relative to 1), else ``None``."""
    if len(n.input) != 3 or n.input[1] not in inits or n.input[2] not in inits:
        return None
    try:
        lo = float(numpy_helper.to_array(inits[n.input[1]]).item())
        hi = float(numpy_helper.to_array(inits[n.input[2]]).item())
    except ValueError:
        return None
    for want in _CLIP_BOUNDS:
        if abs(lo - want[0]) < 1e-6 and abs(hi - want[1]) < 1e-6:
            return want
    return (lo, hi)


def _weight_qparams(
    lo: object, hi: object, qmin: int, qmax: int, symmetric: bool
) -> Tuple[np.float32, int]:
    """``(scale, zero_point)`` of a weight tensor (or channel) on the
    ``[qmin, qmax]`` grid, in the dtypes Quark's ``compute_scale_zp`` uses:
    float32 range, float64 scale and zero point, float32 scale stored."""
    lo32 = np.minimum(np.float32(lo), np.float32(0))
    hi32 = np.maximum(np.float32(hi), np.float32(0))
    if symmetric:
        a = np.maximum(np.abs(lo32), np.abs(hi32))
        lo32, hi32 = -a, a
    scale = np.float64(hi32 - lo32) / np.float64(qmax - qmin)
    if scale < np.finfo(np.float32).tiny:
        return np.float32(1.0), 0
    zp = int(np.round(np.float64(qmin) - np.float64(lo32) / scale))
    if symmetric and qmin == 0 and qmax == 255 and zp == 127:
        zp = 128
    return np.float32(scale), zp


def _weight_axis(node: onnx.NodeProto, rank: int) -> Optional[int]:
    if node.op_type == "Conv":
        return 0
    if node.op_type == "ConvTranspose":
        return 1
    if rank != 2:
        return None
    if node.op_type == "Gemm":
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        return 0 if trans_b else 1
    return 1  # MatMul [K, N]


def _float_tensor_names(model: onnx.ModelProto) -> set:
    g = model.graph
    floats = set()
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.tensor_type.elem_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
            floats.add(vi.name)
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT:
            floats.add(init.name)
    return floats


def _constants_to_initializers(model: onnx.ModelProto) -> None:
    g = model.graph
    keep = []
    for n in g.node:
        if (
            n.op_type == "Constant"
            and len(n.attribute) == 1
            and n.attribute[0].name == "value"
        ):
            t = onnx.TensorProto()
            t.CopyFrom(n.attribute[0].t)
            t.name = n.output[0]
            g.initializer.append(t)
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)


def _is_quantized_node(
    n: onnx.NodeProto,
    op_types: Optional[set],
    exclude_op_types: set,
    exclude_nodes: set,
    skip_nodes: Optional[set] = None,
) -> bool:
    if n.domain not in ("", "ai.onnx") or n.op_type in _NEVER_QUANTIZED:
        return False
    if op_types is not None and n.op_type not in op_types:
        return False
    if skip_nodes and (n.name in skip_nodes or n.output[0] in skip_nodes):
        return False
    return (
        n.op_type not in exclude_op_types
        and n.name not in exclude_nodes
        and n.output[0] not in exclude_nodes
    )


def _data_inputs(
    n: onnx.NodeProto, inits: Optional[Dict[str, TensorProto]] = None
) -> List[str]:
    """The inputs of ``n`` that carry data. With ``inits``, the bias of a Conv /
    ConvTranspose / Gemm that is not a constant is left out as well: Quark's
    quantizers only quantize a bias that is a weight (a Gemm whose ONNX Runtime
    fused ``C`` is an activation keeps it float)."""
    idx = _DATA_INPUTS.get(n.op_type)
    out = [x for i, x in enumerate(n.input) if x and (idx is None or i in idx)]
    if (
        inits is not None
        and n.op_type in ("Conv", "ConvTranspose", "Gemm")
        and len(n.input) > 2
        and n.input[2]
        and n.input[2] not in inits
    ):
        out = [x for x in out if x != n.input[2] or x in n.input[:2]]
    return out


def _is_clamp(n: onnx.NodeProto, inits: Dict[str, TensorProto]) -> bool:
    """Relu-like: Relu / LeakyRelu / PRelu, or Clip to [0, 6] / [0, 1]."""
    if n.op_type in ("Relu", "LeakyRelu", "PRelu"):
        return True
    if n.op_type != "Clip":
        return False
    if len(n.input) >= 3:
        vals = [
            float(numpy_helper.to_array(inits[x]))
            if x in inits and inits[x].dims == []
            else None
            for x in n.input[1:3]
        ]
    else:
        at = {a.name: a.f for a in n.attribute}
        vals = [at.get("min"), at.get("max")]
    return vals[0] == 0.0 and vals[1] in (6.0, 1.0)


def quantize_full_qdq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    activation_dtype: str = "uint8",
    per_channel: bool = True,
    op_types: Optional[Iterable[str]] = None,
    exclude_op_types: Iterable[str] = (),
    exclude_nodes: Iterable[str] = (),
    skip_nodes: Iterable[str] = (),
    fold_relu: bool = True,
    method: str = "minmax",
    providers: Optional[Sequence[str]] = None,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    tensor_dtypes: Optional[Dict[str, str]] = None,
    symmetric_activations: Optional[bool] = None,
    power_of_two: bool = False,
    weight_dtype: str = "int8",
    convert_inputs: bool = True,
    pof2_mode: str = "ceil",
    int8_bias: bool = False,
    int8_constants: bool = False,
    reduce_range: bool = False,
    softmax_unit_range: bool = False,
    align_eltwise_dtype: bool = False,
    tensor_symmetric: Optional[Dict[str, bool]] = None,
    calibrate_options: Optional[Dict[str, object]] = None,
    remove_qdq_after: Optional[Iterable[str]] = None,
    remove_qdq_producers: Iterable[str] = QUARK_QDQ_PRODUCERS,
    fold_activation: Optional[bool] = None,
    adjust_activation_ranges: bool = False,
    quantize_prelu_slope: bool = False,
    align_ops: Optional[Iterable[str]] = None,
    unshared_ops: Iterable[str] = (),
    quantize_bias: bool = True,
    weight_symmetric: bool = True,
    shared_ops: Iterable[str] = (),
    float_clamp_input: bool = False,
) -> onnx.ModelProto:
    """
    Quantize the whole graph to QDQ form for an NPU backend (see the module
    docstring for the exact rules).

    :param model: onnx ModelProto object or file path (float32)
    :param calibration_data: representative input batches (``{name: array}``),
            run through ONNX Runtime by :func:`onnxsim.calibration.calibrate`.
            Not needed when ``ranges`` is given.
    :param activation_dtype: ``"uint8"`` (default), ``"uint16"`` (W8A16;
            opset < 21 models get ``com.microsoft`` Q/DQ, which ONNX Runtime
            and its QNN execution provider accept), or the signed ``"int8"`` /
            ``"int16"`` (zero point 0)
    :param per_channel: int8 weights per output channel (default) or per tensor
    :param op_types: only quantize nodes of these op types (default: all)
    :param exclude_op_types: never quantize nodes of these op types
    :param exclude_nodes: node names (or first-output names) to keep in float
    :param skip_nodes: node names (or first-output names) a quantizer would
            leave alone like a node outside ``op_types`` (it marks none of its
            tensors; its neighbours' quantizers still do), e.g. Quark's
            ``Relu`` / ``Clip`` fed by an unquantized tensor
    :param fold_relu: fold a Relu into its quantized producer's output Q (not
            done with symmetric activations: their zero point is centred)
    :param method: calibration method, passed to
            :func:`onnxsim.calibration.calibrate`
    :param providers: onnxruntime providers for calibration
    :param ranges: precomputed ``{tensor: (min, max)}`` (e.g. calibrated on a
            batch-1 twin of the model); skips calibration for those tensors
    :param tensor_dtypes: per-activation overrides of ``activation_dtype``
            (``{tensor: "uint16"}``), e.g. 16-bit sampling coordinates in an
            otherwise 8-bit graph. Data-movement ops pass their input's dtype
            on unless their output is overridden too.
    :param symmetric_activations: ``scale = absmax / half-range`` with a
            centred zero point (0 for signed types, 128 / 32768 for unsigned).
            Default: on for the signed dtypes, off for the unsigned ones.
    :param tensor_symmetric: per-activation overrides of
            ``symmetric_activations`` (``{tensor: True}``), alongside
            ``tensor_dtypes``
    :param power_of_two: round every activation and weight scale up to a
            power of two (fixed-point friendly, as Quark's ``XINT8``)
    :param weight_dtype: ``"int8"`` (default), ``"int16"`` or ``"uint8"`` for the
            Conv / ConvTranspose / Gemm / MatMul weights (int16 weights use
            ``com.microsoft`` Q/DQ below opset 21, like the 16-bit activations)
    :param convert_inputs: re-quantize an input whose dtype differs from its
            node's output dtype (the 8/16-bit "convert" described above);
            False leaves such nodes consuming one dtype and producing the
            other, as Quark's mixed-precision presets do
    :param pof2_mode: how ``power_of_two`` rounds the *weight* (and int8 bias)
            scales: ``"ceil"`` (default) takes the smallest power of two that
            does not clip; ``"minmse"`` is Quark's ``MinMSE`` search -- the
            power of two among five around the min/max scale with the least
            squared error over the weight values
            (:func:`onnxsim.calibration.pof2_minmse_weight_scale`), which may
            clip a few outliers for finer resolution. Activations are rounded
            by ``method`` instead: ``"minmse_pof2"`` searches them the same
            way (Quark's activation calibrator), any other method rounds up
    :param int8_bias: quantize Conv/ConvTranspose/Gemm biases to int8 with a
            per-tensor scale (a power of two under ``power_of_two``) instead
            of int32 with ``input_scale * weight_scale`` -- what Quark's
            ``XINT8`` emits
    :param reduce_range: Quark's legacy ``reduce_range``: weights (and the other
            constants quantized like weights) use the reduced code range --
            ``[-64, 64]`` for int8, ``[0, 127]`` for uint8, ``[-16384, 16384]``
            for int16 -- instead of the full one; activations and the biases'
            ``input * weight`` scales follow. Not with ``power_of_two``.
    :param int8_constants: quantize the other constant inputs of quantized
            nodes (a LayerNorm scale, a Mul operand, ...) to per-tensor
            symmetric int8 like weights, instead of the activation dtype
    :param align_eltwise_dtype: with ``int8_constants``, constant operands of
            Add / Sub / Mul / Div / Min / Max keep the activation dtype
            (Quark's ``AlignEltwiseQuantType``, set by its ``A16W8`` presets)
    :param softmax_unit_range: calibrate every Softmax output to exactly
            ``(0, 1)`` instead of its observed range (what ONNX Runtime's QDQ
            quantizer, and so Quark's non-power-of-two presets, do)
    :param calibrate_options: extra keyword arguments for
            :func:`onnxsim.calibration.calibrate` (``range_symmetric``,
            ``moving_average``, ``quark_num_bins``, ...)
    :param remove_qdq_after: switches on Quark's rule for dropping the Q/DQ
            pair of an activation: the output of a ``remove_qdq_producers``
            node that has exactly one consumer, of one of these op types
            (``"Relu"``, ``"LeakyRelu"``, ``"PRelu"``, ``"Gelu"`` and
            ``"Clip"`` -- the latter only with constant bounds ``(0, 6)`` or
            ``(0, 1)``), stays float. ``None`` (default) keeps the older
            Relu-only behaviour of ``fold_relu``.
    :param remove_qdq_producers: producer op types of ``remove_qdq_after``
            (Quark's: Conv, Add, MaxPool, AveragePool, GlobalAveragePool,
            MatMul, Gemm, ConvTranspose; ``RemoveQDQInstanceNorm`` adds
            InstanceNormalization)
    :param fold_activation: with ``remove_qdq_after``: drop a Relu / Clip node
            that follows a quantized single-consumer producer, the producer
            quantizing straight to the node's output range (what ONNX Runtime's
            QDQ quantizer does for asymmetric activations). ``None``: only for
            non-symmetric activations, like ``fold_relu``
    :param adjust_activation_ranges: give the input of a single-consumer Relu /
            Clip the calibrated range of its output (ONNX Runtime's
            ``adjust_tensor_ranges``, which Quark inherits)
    :param quantize_prelu_slope: quantize a ``PRelu`` slope like a weight (int8,
            per tensor) even when the PRelu's input is left float (Quark does so
            for symmetric activations)
    :param align_ops: Quark's ``Align*`` options, by op type: after the
            parameters are chosen, ``Concat`` / ``Pad`` / ``Transpose`` /
            ``Reshape`` give their input the output's quantization parameters,
            and ``MaxPool`` / ``AveragePool`` / ``GlobalAveragePool`` /
            ``Slice`` give their output(s) the input's (up to five rounds, in
            Quark's order, so chains settle)
    :param unshared_ops: ops that would reuse their input's parameters (the
            data-movement ops) but calibrate their output on its own instead --
            Quark's behaviour for ``Slice`` and ``Split``
    :param quantize_bias: False leaves Conv / ConvTranspose / Gemm biases float
            (Quark's ``QuantizeBias=False``)
    :param weight_symmetric: False quantizes the Conv / ConvTranspose / Gemm /
            MatMul weights asymmetrically (scale over the observed range,
            non-zero zero point; Quark's ``WeightSymmetric=False``).
            ``weight_dtype="uint8"`` is asymmetric unsigned like Quark's
            ``U8U8_AAWA`` weights, or the centred unsigned grid when symmetric
    :param shared_ops: further op types whose output reuses the input's
            quantization parameters (on top of the data-movement ops), e.g.
            ``AveragePool`` under ONNX Runtime's plain QDQ quantizer
    :param float_clamp_input: leave the output of a quantized node float when
            its only consumer is a Relu / LeakyRelu / PRelu / Clip(0, 6 or 1)
            that is *not* itself quantized (outside ``op_types``) -- Quark's
            "remove Q/DQ between Gemm and Relu" in its NPU transformer
            scheme. (``fold_relu`` handles the quantized-Relu case.)
    :returns: the quantized ModelProto
    """
    if weight_dtype not in ("int8", "int16", "uint8"):
        raise ValueError(f"unsupported weight_dtype: {weight_dtype!r}")
    if activation_dtype not in _DTYPES:
        raise ValueError(f"unsupported activation_dtype: {activation_dtype!r}")
    if reduce_range and power_of_two:
        raise NotImplementedError("reduce_range with power-of-two weights")
    if pof2_mode not in ("ceil", "minmse"):
        raise ValueError(f"pof2_mode must be 'ceil' or 'minmse', got {pof2_mode!r}")
    act_type, act_np, qmin, qmax = _DTYPES[activation_dtype]
    if symmetric_activations is None:
        symmetric_activations = qmin < 0
    sym, p2 = symmetric_activations, power_of_two
    if method == "minmse_pof2" and not sym:
        raise ValueError("method 'minmse_pof2' needs symmetric activations")
    p2_search = p2 and pof2_mode == "minmse"
    tensor_symmetric = dict(tensor_symmetric or {})
    if isinstance(model, str):
        model = onnx.load(model)
    m = onnx.ModelProto()
    m.CopyFrom(model)
    _constants_to_initializers(m)
    m = onnx.shape_inference.infer_shapes(m)
    g = m.graph
    op_types = set(op_types) if op_types is not None else None
    exclude_op_types, exclude_nodes = set(exclude_op_types), set(exclude_nodes)
    skip_set = set(skip_nodes)

    floats = _float_tensor_names(m)
    inits = {i.name: i for i in g.initializer}
    graph_inputs = {i.name for i in g.input}
    graph_outputs = {o.name for o in g.output}
    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)

    qnodes = [
        n
        for n in g.node
        if _is_quantized_node(n, op_types, exclude_op_types, exclude_nodes, skip_set)
    ]
    qnode_ids = {id(n) for n in qnodes}

    # Activations to quantize: float, non-constant data inputs and outputs of quantized nodes.
    acts = []
    seen = set()
    for n in qnodes:
        for x in _data_inputs(n, inits) + [o for o in n.output if o]:
            if x in floats and x not in inits and x not in seen:
                seen.add(x)
                acts.append(x)
    # An explicitly excluded node whose every data input is dequantized and every output
    # quantized would itself form a QDQ node unit (and run quantized). Keep it float by leaving
    # its outputs unquantized; its consumers then read the float value (and run float too).
    # (A node merely outside ``op_types`` is left alone: sandwiched between quantized nodes it
    # runs quantized, which is what an op_types list asks for everywhere else.)
    for n in g.node:
        if id(n) in qnode_ids or not _is_quantized_node(
            n, op_types, set(), set(), skip_set
        ):
            continue
        ins = [x for x in _data_inputs(n, inits) if x in floats and x not in inits]
        outs = [o for o in n.output if o and o in floats]
        if (
            ins
            and outs
            and all(x in seen for x in ins)
            and all(o in seen for o in outs)
        ):
            for o in outs:
                seen.discard(o)
            acts = [a for a in acts if a not in outs]

    ranges = dict(ranges or {})
    missing = [a for a in acts if a not in ranges]
    if missing:
        if calibration_data is None:
            raise ValueError(
                f"no calibration data and no ranges for {len(missing)} tensors, e.g. {missing[:3]}"
            )
        # calibrate() reports every quantizable activation, not just the missing ones: the
        # caller's precomputed ranges must win over it
        calibrated = calibrate(
            m,
            calibration_data,
            providers=providers,
            method=method,
            extra_tensor_names=missing,
            activation_type=activation_dtype,
            tensor_dtypes=tensor_dtypes,
            **(calibrate_options or {}),  # type: ignore[arg-type]
        )
        ranges = {**calibrated, **ranges}

    if softmax_unit_range:
        act_set0 = set(acts)
        for n in g.node:
            # (only a Softmax the quantizer quantizes: ``should_quantize_node``)
            if n.op_type == "Softmax" and id(n) in qnode_ids:
                for o in n.output:
                    if o in act_set0:
                        ranges[o] = (0.0, 1.0)

    # Relu folding: producer -> Relu becomes producer -> Q(range of the Relu output, lo = 0).
    removed = set()
    unshared = set(unshared_ops)
    shared = set(shared_ops)
    folded = set()  # outputs of removed Relu / Clip nodes, now their producers'
    centred = sym or any(tensor_symmetric.values())
    quark_rules = remove_qdq_after is not None
    fold_node = (
        (fold_relu and not centred) if fold_activation is None else fold_activation
    )
    producer = {o: n for n in g.node for o in n.output}
    if adjust_activation_ranges:
        # (Quark runs ONNX Runtime's ``adjust_tensor_ranges`` twice -- once when
        # the quantizer is built, once when it quantizes -- so a chain of
        # Relu / Clip nodes propagates its range two steps)
        for _ in range(2):
            for r in qnodes:
                if (
                    r.op_type in ("Relu", "Clip")
                    and len(consumers[r.input[0]]) == 1
                    and r.input[0] in ranges
                    and r.output[0] in ranges
                ):
                    ranges[r.input[0]] = ranges[r.output[0]]
    if fold_node and (fold_relu or quark_rules):
        fold_ops = ("Relu", "Clip") if quark_rules else ("Relu",)
        folded_into: Dict[int, onnx.NodeProto] = {}  # removed activation -> producer
        for r in qnodes:
            if r.op_type not in fold_ops:
                continue
            src = r.input[0]
            p = producer.get(src)
            # (a Relu / Clip behind an already folded one folds into the same producer)
            while p is not None and id(p) in folded_into:
                p = folded_into[id(p)]
            if (
                p is None
                or id(p) not in qnode_ids
                or len(consumers[src]) != 1
                or src in graph_outputs
                or r.output[0] not in ranges
            ):
                continue
            for k, o in enumerate(p.output):
                if o == src:
                    p.output[k] = r.output[0]
            folded_into[id(r)] = p
            if not quark_rules:
                ranges[r.output[0]] = (0.0, max(ranges[r.output[0]][1], 0.0))
            elif r.op_type == "Relu" and ranges[r.output[0]][0] < 0:
                # Quark's own behaviour: a calibrator that reports a range
                # below zero for a Relu output (Distribution) leaves the
                # folded model without the clamp
                warnings.warn(
                    f"Relu {r.name!r} is folded onto the range "
                    f"{ranges[r.output[0]]}, which reaches below zero: the "
                    "quantized model no longer clamps negative values (as in "
                    "Quark)",
                    UserWarning,
                    stacklevel=2,
                )
            removed.add(id(r))
            folded.add(r.output[0])
        acts = [
            a
            for a in acts
            if not any(id(r) in removed and r.input[0] == a for r in consumers[a])
        ]

    if quark_rules:
        # Quark's "remove Q/DQ between producer and activation": the producer's
        # output stays float when its single consumer is one of these ops
        consumer_ops = set(remove_qdq_after or ())
        producer_ops = set(remove_qdq_producers)
        skip = set()
        for c in g.node:
            if c.op_type not in consumer_ops or not c.input or id(c) in removed:
                continue
            src = c.input[0]
            p = producer.get(src)
            if (
                p is not None
                and p.op_type in producer_ops
                and p.output[0] == src
                and len(consumers[src]) == 1
                # (a graph output's Q input is renamed: Quark's pair stays)
                and src not in graph_outputs
                and (c.op_type != "Clip" or _clip_bounds(c, inits) in _CLIP_BOUNDS)
            ):
                skip.add(src)
        # ... and so is a Pad's, when its only consumer is an (Average) pool
        for c in g.node:
            if c.op_type not in ("AveragePool", "GlobalAveragePool") or not c.input:
                continue
            p = producer.get(c.input[0])
            if (
                p is not None
                and p.op_type == "Pad"
                and p.output[0] == c.input[0]
                and len(consumers[c.input[0]]) == 1
                and c.input[0] not in graph_outputs
            ):
                skip.add(c.input[0])
        acts = [a for a in acts if a not in skip]
    elif fold_relu and centred:
        # A centred zero point cannot clamp at zero, so the Relu stays; instead
        # the tensor between a quantized producer and its Relu is left float
        # (Quark's "remove Q/DQ between conv and relu").
        skip = set()
        for r in qnodes:
            src = r.input[0] if r.op_type == "Relu" else None
            p = producer.get(src) if src else None
            if p is not None and len(consumers[src]) == 1 and src not in graph_outputs:
                skip.add(src)
        acts = [a for a in acts if a not in skip]

    if float_clamp_input:
        producer = {o: n for n in g.node for o in n.output}
        skip = set()
        for r in g.node:
            if id(r) in qnode_ids or not _is_clamp(r, inits):
                continue
            src = r.input[0]
            p = producer.get(src)
            if (
                p is not None
                and id(p) in qnode_ids
                and len(consumers[src]) == 1
                and src not in graph_outputs
            ):
                skip.add(src)
        acts = [a for a in acts if a not in skip]

    # Quantization parameters, propagating through data-movement ops in topological order.
    tensor_dtypes = dict(tensor_dtypes or {})
    for t in set(tensor_dtypes.values()) - set(_DTYPES):
        raise ValueError(f"unsupported dtype in tensor_dtypes: {t!r}")
    qp: Dict[str, Tuple[float, int]] = {}
    qdt: Dict[str, str] = {}
    # tensors whose parameters were taken from another tensor's (a data-movement
    # op's output): they use the same scale / zero-point initializers, as in
    # Quark's graphs, so a later position move reaches all of them
    share_root: Dict[str, str] = {}

    def set_qp(x: str) -> None:
        dt = tensor_dtypes.get(x, activation_dtype)
        qp[x] = _qparams(
            *ranges[x],
            *_DTYPES[dt][2:],
            tensor_symmetric.get(x, sym),
            p2,
            quark_rounding=p2_search,
        )
        qdt[x] = dt

    for x in graph_inputs:
        if x in seen and x in ranges:
            set_qp(x)
    # Quark's ``AlignEltwiseQuantType`` puts a ``TensorQuantOverrides`` entry on
    # every input of an eltwise op, and a tensor with an override (or whose
    # provider has one) is quantized with its own parameters instead of sharing
    override_tensors = (
        {x for n in g.node if n.op_type in _ELTWISE_OPS for x in n.input}
        if align_eltwise_dtype
        else set()
    )
    for n in g.node:
        if id(n) in removed:
            continue
        if (
            (n.op_type in _SHARED_QPARAM_OPS or n.op_type in shared)
            and id(n) in qnode_ids
            and n.input[0] in qp
        ):
            for o in n.output:
                # a folded Relu / Clip gives its producer's output the node's
                # own range: sharing the input's parameters would lose the clamp
                if (
                    o
                    and o in floats
                    and o not in tensor_dtypes
                    and o not in folded
                    and n.op_type not in unshared
                    and o not in override_tensors
                    and n.input[0] not in override_tensors
                ):
                    qp[o], qdt[o] = qp[n.input[0]], qdt[n.input[0]]
                    share_root[o] = share_root.get(n.input[0], n.input[0])
        for x in list(n.input) + list(n.output):
            if x in seen and x not in qp and x in ranges and x not in inits:
                set_qp(x)

    # the activation parameters after every round of Quark's alignment loop (the
    # first entry: as calibrated), which the int32 biases are re-quantized against
    qp_history: List[Dict[str, Tuple[float, int]]] = [dict(qp)]
    if align_ops:
        qp_history += _align_qparams(
            g, set(align_ops), set(acts) | graph_inputs, qp, qdt, share_root
        )

    opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
    if opset < 13:
        raise ValueError(
            "full-graph QDQ needs opset >= 13 (per-channel DequantizeLinear)"
        )

    def domain_of(dt: str) -> str:
        return "com.microsoft" if dt in _WIDE_DTYPES and opset < 21 else ""

    qdq_domain = domain_of(activation_dtype)
    if any(
        domain_of(d) for d in list(qdt.values()) + [activation_dtype, weight_dtype]
    ) and not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    new_inits: List[TensorProto] = []
    uid = [0]

    def fresh(base: str) -> str:
        uid[0] += 1
        return f"{base}/qdq{uid[0]}"

    def add_init(name: str, arr: np.ndarray) -> str:
        new_inits.append(numpy_helper.from_array(arr, name))
        return name

    # Activation Q -> DQ pairs. The producer's output is renamed and the DQ takes over the
    # original name, so every consumer (and a graph output) reads the dequantized value.
    act_nodes: List[onnx.NodeProto] = []
    rename: Dict[str, str] = {}
    shared_init_names: Dict[str, Tuple[str, str]] = {}
    for a in acts:
        if a not in qp:
            continue
        s, zp = qp[a]
        dom = domain_of(qdt[a])
        root = share_root.get(a, a)
        if root != a and qp.get(root) == qp[a] and root in shared_init_names:
            sn, zn = shared_init_names[root]
        else:
            sn = add_init(fresh(a) + "/scale", np.array(s, np.float32))
            zn = add_init(fresh(a) + "/zp", np.array(zp, _DTYPES[qdt[a]][1]))
            if qp.get(root) == qp[a]:
                shared_init_names.setdefault(root, (sn, zn))
        q_out = a + "/q"
        if a in graph_inputs:
            dq_out = a + "/dq"
            for c in consumers[a]:
                for k, x in enumerate(c.input):
                    if x == a:
                        c.input[k] = dq_out
            act_nodes += [
                helper.make_node(
                    "QuantizeLinear",
                    [a, sn, zn],
                    [q_out],
                    name=a + "/Q",
                    domain=dom,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [q_out, sn, zn],
                    [dq_out],
                    name=a + "/DQ",
                    domain=dom,
                ),
            ]
        else:
            pre = a + "/f"
            rename[a] = pre
            act_nodes += [
                helper.make_node(
                    "QuantizeLinear",
                    [pre, sn, zn],
                    [q_out],
                    name=a + "/Q",
                    domain=dom,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [q_out, sn, zn],
                    [a],
                    name=a + "/DQ",
                    domain=dom,
                ),
            ]

    # Mixed 8/16-bit activations: a quantized node computes in its output's dtype, so an input
    # of the other dtype is re-quantized for it (DQ -> Q' -> DQ', a "convert" the QNN EP maps
    # to its Convert op). GridSample's grid is exempt: its coordinates are the reason to mix.
    converted: Dict[Tuple[str, str], str] = {}
    acts_set = set(acts) | set(graph_inputs)
    act_extra: List[str] = []
    for n in qnodes if convert_inputs else ():
        if id(n) in removed:
            continue
        outs = [o for o in n.output if o in qdt]
        if not outs:
            continue
        node_dt = qdt[outs[0]]
        for k, x in enumerate(n.input):
            src = (
                x[: -len("/dq")]
                if x.endswith("/dq") and x[: -len("/dq")] in graph_inputs
                else x
            )
            if (
                src not in qdt
                or src not in acts_set  # left float (Relu-adjacent), nothing to convert
                or qdt[src] == node_dt
                or (n.op_type == "GridSample" and k == 1)
            ):
                continue
            ckey = (src, node_dt)
            if ckey not in converted:
                c = f"{src}/as_{node_dt}"
                sc, zc = _qparams(
                    *ranges[src],
                    *_DTYPES[node_dt][2:],
                    sym,
                    p2,
                    quark_rounding=p2_search,
                )
                qp[c], qdt[c] = (sc, zc), node_dt
                sn = add_init(fresh(c) + "/scale", np.array(sc, np.float32))
                zn = add_init(fresh(c) + "/zp", np.array(zc, _DTYPES[node_dt][1]))
                dom = domain_of(node_dt)
                act_nodes += [
                    helper.make_node(
                        "QuantizeLinear",
                        [x, sn, zn],
                        [c + "/q"],
                        name=c + "/Q",
                        domain=dom,
                    ),
                    helper.make_node(
                        "DequantizeLinear",
                        [c + "/q", sn, zn],
                        [c],
                        name=c + "/DQ",
                        domain=dom,
                    ),
                ]
                converted[ckey] = c
                act_extra.append(c)
            n.input[k] = converted[ckey]

    def w_range(dt: str) -> Tuple[int, int]:
        """``(qmin, qmax)`` of the weight grid (``reduce_range`` shrinks it)."""
        lo, hi = _REDUCED_RANGES[dt] if reduce_range else _DTYPES[dt][2:]
        return int(lo), int(hi)

    def int8_tensor_dq(x: str, w: np.ndarray, per_row: bool = False) -> str:
        """``int8`` + per-tensor symmetric scale (+ DQ) for a weight-like
        constant ``x``; returns the DQ output name. With asymmetric (or uint8)
        weights it is the weights' grid instead (Quark treats these constants
        as weights). ``per_row``: one scale per index of axis 0 instead (Quark's
        per-channel mode for a bias or a PReLU slope)."""
        if per_row and w.ndim >= 1 and w.shape[0] > 1:
            qm = w_range("int8")[1]
            rows = w.reshape(w.shape[0], -1)
            if p2_search:
                s = np.array([pof2_minmse_weight_scale(r) for r in rows], np.float32)
            else:
                s = (np.maximum(np.abs(rows).max(axis=1), 1e-12) / qm).astype(
                    np.float32
                )
                if p2:
                    s = (2.0 ** np.ceil(np.log2(s))).astype(np.float32)
            q = np.clip(
                np.round(w / s.reshape((-1,) + (1,) * (w.ndim - 1))), -qm, qm
            ).astype(np.int8)
            base = fresh(x)
            add_init(base + "/int8", q)
            add_init(base + "/scale", s)
            add_init(base + "/zp", np.zeros(s.shape, np.int8))
            out = base + "/dq"
            act_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [base + "/int8", base + "/scale", base + "/zp"],
                    [out],
                    name=out,
                    axis=0,
                )
            )
            return out
        if p2_search and not weight_symmetric and weight_dtype == "int8":
            s32, z = _pof2_minmse_asymmetric(w)
            q = np.clip(np.round(w / np.float32(s32)) + z, -127, 127).astype(np.int8)
            base = fresh(x)
            add_init(base + "/int8", q)
            add_init(base + "/scale", np.array(s32, np.float32))
            add_init(base + "/zp", np.array(z, np.int8))
            out = base + "/dq"
            act_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [base + "/int8", base + "/scale", base + "/zp"],
                    [out],
                    name=out,
                )
            )
            return out
        if not p2 and (not weight_symmetric or weight_dtype == "uint8"):
            dt = "uint8" if weight_dtype == "uint8" else "int8"
            lo, hi = w_range(dt)
            s32, z = _weight_qparams(w.min(), w.max(), lo, hi, weight_symmetric)
            # (Quark clips to the symmetric code range, so a grid's lowest code
            # -128 is never produced)
            q = np.clip(np.round(w / s32) + z, max(lo, -hi), hi).astype(_DTYPES[dt][1])
            base = fresh(x)
            add_init(base + f"/{dt}", q)
            add_init(base + "/scale", np.array(s32, np.float32))
            add_init(base + "/zp", np.array(z, _DTYPES[dt][1]))
            out = base + "/dq"
            act_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    [base + f"/{dt}", base + "/scale", base + "/zp"],
                    [out],
                    name=out,
                )
            )
            return out
        if p2_search:
            s = pof2_minmse_weight_scale(w)
        else:
            s = max(float(np.abs(w).max()), 1e-12) / w_range("int8")[1]
            s = _pof2(s) if p2 else s
        qm = w_range("int8")[1]
        q = np.clip(np.round(w / np.float32(s)), -qm, qm).astype(np.int8)
        base = fresh(x)
        add_init(base + "/int8", q)
        add_init(base + "/scale", np.array(s, np.float32))
        add_init(base + "/zp", np.array(0, np.int8))
        out = base + "/dq"
        act_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [base + "/int8", base + "/scale", base + "/zp"],
                [out],
                name=out,
            )
        )
        return out

    def act_scale(x: str, params: Optional[Dict] = None) -> Optional[float]:
        params = qp if params is None else params
        if x not in params and x.endswith("/dq"):
            x = x[: -len("/dq")]  # a graph input, rewired to its DQ above
        return params[x][0] if x in params else None

    # Constant inputs of quantized nodes that form a real QDQ unit (every float activation
    # input dequantized): a lone DQ on a weight of a float node would strand it on the CPU.
    cache: Dict[Tuple, str] = {}
    act_set = set(acts) | set(act_extra)
    before = {id(n): list(n.input) for n in qnodes}
    for n in qnodes:
        if id(n) in removed:
            continue
        slope_only = False
        if not all(
            x in act_set and x in qp
            for x in _data_inputs(n, inits)
            if x in floats and x not in inits
        ):
            if not (quantize_prelu_slope and n.op_type == "PRelu"):
                continue
            slope_only = True
        data = set(_data_inputs(n, inits))
        for k, x in enumerate(list(n.input)):
            if slope_only and k != 1:
                continue
            if (
                quark_rules
                and not quantize_prelu_slope
                and n.op_type == "PRelu"
                and k == 1
            ):
                continue  # Quark's plain quantizer leaves the slope float
            if (
                not quantize_bias
                and k == 2
                and (
                    n.op_type in ("Conv", "ConvTranspose", "Gemm")
                    or (n.op_type == "InstanceNormalization" and int8_constants)
                )
            ):
                continue  # the bias stays float
            if (
                x not in inits
                or x not in data
                or inits[x].data_type != TensorProto.FLOAT
            ):
                continue
            w = numpy_helper.to_array(inits[x]).astype(np.float32)
            if n.op_type in _WEIGHT_AXIS_OPS and k == 1:
                axis = _weight_axis(n, w.ndim) if per_channel else None
                key = ("w", x, axis)
                if key not in cache:
                    w_np = _DTYPES[weight_dtype][1]
                    wmin, wmax = w_range(weight_dtype)
                    w_max = wmax
                    if (
                        p2_search
                        and not weight_symmetric
                        and weight_dtype == "int8"
                        and axis is None
                    ):
                        s32, z = _pof2_minmse_asymmetric(w)
                        s, zp = np.array(s32, np.float32), np.array(z, w_np)
                        q = np.clip(np.round(w / s) + z, -w_max, w_max).astype(w_np)
                    elif not weight_symmetric or weight_dtype == "uint8":
                        clip_lo = max(wmin, -w_max)  # Quark: symmetric code range
                        if axis is None:
                            s32, z = _weight_qparams(
                                w.min(), w.max(), wmin, wmax, weight_symmetric
                            )
                            s, zp = np.array(s32, np.float32), np.array(z, w_np)
                            q = np.clip(np.round(w / s) + z, clip_lo, wmax).astype(w_np)
                        else:
                            chans = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)
                            sz = [
                                _weight_qparams(
                                    c.min(), c.max(), wmin, wmax, weight_symmetric
                                )
                                for c in chans
                            ]
                            s = np.array([a for a, _ in sz], np.float32)
                            zp = np.array([b for _, b in sz], w_np)
                            shape = [1] * w.ndim
                            shape[axis] = -1
                            q = np.clip(
                                np.round(w / s.reshape(shape)) + zp.reshape(shape),
                                clip_lo,
                                wmax,
                            ).astype(w_np)
                    elif axis is None:
                        s = np.array(max(np.abs(w).max(), 1e-12) / w_max, np.float32)
                        if p2_search and weight_dtype == "int8":
                            s = np.array(pof2_minmse_weight_scale(w), np.float32)
                        elif p2:
                            s = np.array(_pof2(float(s)), np.float32)
                        q = np.clip(np.round(w / s), -w_max, w_max).astype(w_np)
                        zp = np.array(0, w_np)
                    else:
                        red = tuple(i for i in range(w.ndim) if i != axis)
                        s = (np.maximum(np.abs(w).max(axis=red), 1e-12) / w_max).astype(
                            np.float32
                        )
                        if p2_search:
                            s = np.array(
                                [
                                    pof2_minmse_weight_scale(c)
                                    for c in np.moveaxis(w, axis, 0)
                                ],
                                np.float32,
                            )
                        elif p2:
                            s = (2.0 ** np.ceil(np.log2(s))).astype(np.float32)
                        shape = [1] * w.ndim
                        shape[axis] = -1
                        q = np.clip(
                            np.round(w / s.reshape(shape)), -w_max, w_max
                        ).astype(w_np)
                        zp = np.zeros(s.shape, w_np)
                    base = fresh(x)
                    add_init(base + f"/{weight_dtype}", q)
                    add_init(base + "/scale", s)
                    add_init(base + "/zp", zp)
                    out = base + "/dq"
                    attrs = {"axis": axis} if axis is not None else {}
                    act_nodes.append(
                        helper.make_node(
                            "DequantizeLinear",
                            [base + f"/{weight_dtype}", base + "/scale", base + "/zp"],
                            [out],
                            name=out,
                            domain=domain_of(weight_dtype),
                            **attrs,
                        )
                    )
                    cache[key] = out
                n.input[k] = cache[key]
            elif (
                (
                    n.op_type in ("Conv", "ConvTranspose", "Gemm")
                    or (n.op_type == "InstanceNormalization" and int8_constants)
                )
                and k == 2
                and w.ndim == 1
                and int8_bias
            ):
                key = ("b8", x, None)
                if key not in cache:
                    cache[key] = int8_tensor_dq(x, w, per_row=per_channel)
                n.input[k] = cache[key]
            elif (
                n.op_type in ("Conv", "ConvTranspose", "Gemm")
                or (n.op_type == "InstanceNormalization" and int8_constants)
            ) and (k == 2 and w.ndim == 1):
                if ("b32", x) in cache:
                    # a bias several nodes read is quantized once, with the scales
                    # of the first of them (Quark's ``bias_to_quantize``)
                    n.input[k] = cache[("b32", x)]
                    continue
                w_dq = n.input[1]
                sx = act_scale(n.input[0], qp_history[0])
                w_scale_name = (
                    w_dq[: -len("/dq")] + "/scale" if w_dq.endswith("/dq") else None
                )
                ws = next(
                    (
                        numpy_helper.to_array(t)
                        for t in new_inits
                        if t.name == w_scale_name
                    ),
                    None,
                )
                if sx is None or ws is None:
                    continue  # weight or input not quantized: keep a float bias
                s = (sx * np.broadcast_to(ws, w.shape)).astype(np.float32)
                s = np.maximum(s, 1e-30)
                # float64 division, as ONNX Runtime's quantize_bias_static does: a
                # float32 quotient loses integer precision above 2**24 (int16 x
                # int16 scales)
                q = np.clip(
                    np.round(w.astype(np.float64) / s.astype(np.float64)),
                    -(2**31),
                    2**31 - 1,
                ).astype(np.int32)
                # Quark's ``adjust_bias_scale``, after every round of its alignment
                # loop: a bias whose scale is no longer input scale * weight scale
                # (the input's parameters moved) is divided by the ratio and
                # truncated, and takes the new scale -- unless one element still
                # matches
                ws32 = np.broadcast_to(ws, w.shape).astype(np.float32)
                for params in qp_history[1:]:
                    sx_i = act_scale(n.input[0], params)
                    if sx_i is None:
                        break
                    prod = (np.float32(sx_i) * ws32).astype(np.float32)
                    if np.all(prod != s):
                        q = (q / (prod / s)).astype(np.int32)
                        s = prod
                base = fresh(x)
                add_init(base + "/int32", q)
                add_init(base + "/scale", s)
                add_init(base + "/zp", np.zeros(s.shape, np.int32))
                out = base + "/dq"
                act_nodes.append(
                    helper.make_node(
                        "DequantizeLinear",
                        [base + "/int32", base + "/scale", base + "/zp"],
                        [out],
                        name=out,
                        axis=0,
                    )
                )
                cache[("b32", x)] = out
                n.input[k] = out
            elif int8_constants and not (
                align_eltwise_dtype and n.op_type in _ELTWISE_OPS
            ):
                key = ("c8", x, None)
                if key not in cache:
                    cache[key] = int8_tensor_dq(
                        x,
                        w,
                        per_row=per_channel
                        and not p2
                        and n.op_type == "PRelu"
                        and w.ndim > 1,
                    )
                n.input[k] = cache[key]
            else:
                key = ("c", x, None)
                if key not in cache:
                    s, zp = _qparams(w.min(), w.max(), qmin, qmax, weight_symmetric, p2)
                    # (the symmetric code range, like Quark's weights: no -128 / -32768)
                    q = np.clip(np.round(w / s) + zp, max(qmin, -qmax), qmax)
                    q = q.astype(act_np)
                    base = fresh(x)
                    add_init(base + "/q", q)
                    add_init(base + "/scale", np.array(s, np.float32))
                    add_init(base + "/zp", np.array(zp, act_np))
                    out = base + "/dq"
                    act_nodes.append(
                        helper.make_node(
                            "DequantizeLinear",
                            [base + "/q", base + "/scale", base + "/zp"],
                            [out],
                            name=out,
                            domain=qdq_domain,
                        )
                    )
                    cache[key] = out
                n.input[k] = cache[key]

    # a quantized constant is replaced by its DQ in *every* node that reads it
    # (Quark's ``replace_input_of_all_nodes``): a shared initializer is not left
    # float for a node outside the QDQ unit, a Clip's bound or an excluded Conv
    const_dq: Dict[str, str] = {}
    for n in qnodes:
        for k, orig in enumerate(before[id(n)]):
            if orig in inits and n.input[k] != orig:
                const_dq.setdefault(orig, n.input[k])
    if const_dq:
        for n in g.node:
            for k, x in enumerate(n.input):
                if x in const_dq:
                    n.input[k] = const_dq[x]
    for n in g.node:
        for k, o in enumerate(n.output):
            if o in rename:
                n.output[k] = rename[o]
    nodes = [n for n in g.node if id(n) not in removed]
    del g.node[:]
    # Q/DQ first is fine topologically for constants; activation pairs must follow their
    # producer, so sort once at the end.
    g.node.extend(nodes + act_nodes)
    g.initializer.extend(new_inits)
    used = {x for n in g.node for x in n.input} | graph_outputs
    kept = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(kept)
    del g.value_info[:]
    _toposort(g)
    return m


# Quark's Align* passes, in its order: (op types, copy output -> inputs?)
_ALIGN_PASSES = (
    (("Concat",), True),
    (("MaxPool", "AveragePool", "GlobalAveragePool"), False),
    (("Pad",), True),
    (("Slice",), False),
    (("Transpose",), True),
    (("Reshape",), True),
)


def _align_qparams(
    g: onnx.GraphProto,
    ops: set,
    has_qdq: set,
    qp: Dict[str, Tuple[float, int]],
    qdt: Dict[str, str],
    share_root: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Tuple[float, int]]]:
    """Quark's ``align_quantize_info`` on the chosen parameters (see
    ``quantize_full_qdq``'s ``align_ops``). Tensors that share their parameter
    initializers (``share_root``: a data-movement op's output and its input) move
    together, as the initializer is rewritten in place in Quark. Returns the
    parameters after each round of the loop."""
    share_root = share_root or {}
    history: List[Dict[str, Tuple[float, int]]] = []
    groups: Dict[str, List[str]] = defaultdict(list)
    for t in list(qp):
        groups[share_root.get(t, t)].append(t)

    def copy(src: str, dst: str) -> bool:
        if (
            src not in has_qdq
            or dst not in has_qdq
            or src not in qp
            or dst not in qp
            or qdt[src] != qdt[dst]
            or qp[src] == qp[dst]
        ):
            return False
        for t in groups[share_root.get(dst, dst)]:
            qp[t] = qp[src]
        return True

    for _ in range(5):
        changed = False
        for pass_ops, out_to_in in _ALIGN_PASSES:
            for n in g.node:
                if n.op_type not in pass_ops or n.op_type not in ops:
                    continue
                if out_to_in:
                    if not n.output or n.output[0] not in has_qdq:
                        continue
                    ins = n.input if n.op_type == "Concat" else n.input[:1]
                    for x in ins:
                        changed |= copy(n.output[0], x)
                elif n.input and n.input[0] in has_qdq:
                    for o in n.output:
                        changed |= copy(n.input[0], o)
        history.append(dict(qp))
        if not changed:
            break
    return history


def _toposort(g: onnx.GraphProto) -> None:
    avail = {i.name for i in g.input} | {i.name for i in g.initializer} | {""}
    pending = list(g.node)
    order = []
    while pending:
        rest = []
        for n in pending:
            if all(x in avail for x in n.input):
                order.append(n)
                avail.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(pending):
            raise ValueError(
                f"graph has a cycle or dangling input at {rest[0].name}: {list(rest[0].input)}"
            )
        pending = rest
    del g.node[:]
    g.node.extend(order)


def quantized_io(
    model: onnx.ModelProto,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    nhwc_inputs: Iterable[str] = (),
) -> Tuple[onnx.ModelProto, Dict[str, Dict[str, Union[float, str]]]]:
    """
    Make a :func:`quantize_full_qdq` model take/return quantized tensors.

    A float graph input ``x`` whose only consumer is its ``QuantizeLinear``
    becomes the integer tensor ``x`` itself (the host quantizes it:
    ``round(x / scale) + zero_point``); a graph output ``y`` produced by a
    ``DequantizeLinear`` becomes the integer tensor before it (the host
    dequantizes, or feeds it to the next quantized model as-is). Both are
    lossless: the graph quantized/dequantized there anyway. ``nhwc_inputs``
    additionally take the input channels-last (``[N, H, W, C]``), transposed
    in the graph before the DequantizeLinear.

    :param inputs: input names to convert (default: every quantized input)
    :param outputs: output names to convert (default: every quantized output)
    :returns: ``(model, {tensor: {"scale": s, "zero_point": zp, "dtype": ...}})``
            (``"layout": "nhwc"`` added for ``nhwc_inputs``)
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)
    producer = {o: n for n in g.node for o in n.output}
    nhwc_inputs = set(nhwc_inputs)
    info: Dict[str, Dict[str, Union[float, str]]] = {}
    remove = set()
    extra_nodes = []
    for vi in g.input:
        if inputs is not None and vi.name not in inputs:
            continue
        cs = consumers[vi.name]
        if len(cs) != 1 or cs[0].op_type != "QuantizeLinear":
            continue
        q = cs[0]
        zp = inits[q.input[2]]
        info[vi.name] = {
            "scale": float(inits[q.input[1]]),
            "zero_point": int(zp),
            "dtype": str(zp.dtype),
        }
        remove.add(id(q))
        elem = helper.np_dtype_to_tensor_dtype(zp.dtype)
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim
        ]
        if vi.name in nhwc_inputs:
            assert len(dims) == 4, vi.name
            info[vi.name]["layout"] = "nhwc"
            dims = [dims[0], dims[2], dims[3], dims[1]]
            extra_nodes.append(
                helper.make_node(
                    "Transpose",
                    [vi.name],
                    [q.output[0]],
                    name=vi.name + "/to_nchw",
                    perm=[0, 3, 1, 2],
                )
            )
        else:
            for c in consumers[q.output[0]]:
                for k, x in enumerate(c.input):
                    if x == q.output[0]:
                        c.input[k] = vi.name
        vi.type.CopyFrom(helper.make_tensor_type_proto(elem, dims))
    for vi in g.output:
        if outputs is not None and vi.name not in outputs:
            continue
        dq = producer.get(vi.name)
        if dq is None or dq.op_type != "DequantizeLinear" or consumers[vi.name]:
            continue
        zp = inits[dq.input[2]]
        if zp.ndim != 0:
            continue
        info[vi.name] = {
            "scale": float(inits[dq.input[1]]),
            "zero_point": int(zp),
            "dtype": str(zp.dtype),
        }
        remove.add(id(dq))
        # The Q's output takes the graph output's name.
        q_out = dq.input[0]
        for n in g.node:
            for k, o in enumerate(n.output):
                if o == q_out:
                    n.output[k] = vi.name
            for k, x in enumerate(n.input):
                if x == q_out and id(n) != id(dq):
                    n.input[k] = vi.name
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in vi.type.tensor_type.shape.dim
        ]
        vi.type.CopyFrom(
            helper.make_tensor_type_proto(
                helper.np_dtype_to_tensor_dtype(zp.dtype), dims
            )
        )
    nodes = [n for n in g.node if id(n) not in remove] + extra_nodes
    del g.node[:]
    g.node.extend(nodes)
    _toposort(g)
    return m, info


def sampling_coordinate_tensors(
    model: onnx.ModelProto, stop_op_types: Iterable[str] = ("Gemm", "MatMul", "Conv")
) -> List[str]:
    """
    The tensors that compute ``GridSample``'s sampling grid: a backward slice
    from every GridSample's grid input through elementwise/data-movement ops,
    stopping at (and including) the outputs of ``stop_op_types`` (the offset
    projection) and graph inputs (host-computed reference points).

    Sampling coordinates need far more resolution than 8 bits over their
    range (a uint8 step of a [-1, 1] grid is ~0.4% of the feature map, i.e.
    most of a pixel on a 25-wide map), so these are the natural
    ``tensor_dtypes={t: "uint16" ...}`` of :func:`quantize_full_qdq` in a
    deformable-attention model.
    """
    producer = {o: n for n in model.graph.node for o in n.output}
    inits = {i.name for i in model.graph.initializer}
    stop = set(stop_op_types)
    out: List[str] = []
    todo = [n.input[1] for n in model.graph.node if n.op_type == "GridSample"]
    while todo:
        t = todo.pop()
        if t in out or t in inits or not t:
            continue
        out.append(t)
        p = producer.get(t)
        if p is None or p.op_type in stop or p.op_type in _NEVER_QUANTIZED:
            continue
        todo.extend(_data_inputs(p))
    return out
