"""Full-graph QDQ quantization: every float activation, not just MatMul/Conv inputs.

:func:`onnxsim.quantize_static`'s default scheme puts a QuantizeLinear/
DequantizeLinear pair only on the activation input of each MatMul/Gemm/Conv
(and dequantizes the INT8 weight): enough for a CPU runtime that fuses those
into integer kernels, but an NPU that runs *whole graphs* in integer -- the
Qualcomm HTP through ONNX Runtime's QNN EP, for example -- needs every op
wrapped. Its converter groups each ``DQ -> op -> Q`` "node unit" into one
quantized op, so a float edge anywhere (a SiLU's Sigmoid/Mul, a residual Add,
a Concat, a Resize) either falls back to the CPU or, in a strict all-NPU
session, fails to compile.

:func:`quantize_full_graph` places the pairs the way ONNX Runtime's own QDQ
quantizer does for such targets:

- **Activations** (graph inputs and every float node output): uint8 (or
  uint16, int8, int16) asymmetric, per tensor, from the calibrated ``(min, max)`` widened
  to include 0. Value-preserving ops (Reshape, Transpose, MaxPool, Split,
  Slice, Resize...) reuse their input's scale and zero point instead.
- **Conv/ConvTranspose/Gemm/MatMul weights** (constant input 1): INT8
  symmetric (zero point 0), per output channel by default.
- **Biases** (Conv/Gemm constant input 2): INT32 with scale
  ``activation_scale * weight_scale`` and zero point 0 -- the integer
  accumulator's own scale, so the bias adds in without a requantize.
- **Other float constants in data positions** (an Add's or Mul's constant
  operand, a Concat's constant piece...): quantized like an activation of
  their own range. Non-data inputs (Resize scales, Clip bounds, shape
  tensors) stay float.
- **Excluded nodes** (by name or op type) stay float: they read their inputs
  through the DequantizeLinear their quantized producers already emit, and
  their outputs are quantized only where a quantized node reads them. A tensor mixing two very different
  ranges (e.g. a detector head's final Concat of box pixels and 0..1 class
  scores) is the typical node to exclude.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Input slots that are not data: they stay float and are never QDQ'd.
_NON_DATA_INPUTS: Dict[str, Set[int]] = {
    "Resize": {1, 2, 3},
    "Upsample": {1},
    "Clip": {1, 2},
    "Pad": {1, 2, 3},
    "Reshape": {1},
    "Expand": {1},
    "Tile": {1},
    "Slice": {1, 2, 3, 4},
    "Split": {1},
    "Squeeze": {1},
    "Unsqueeze": {1},
    "TopK": {1},
    "Gather": {1},
    "GatherElements": {1},
    "GatherND": {1},
    "ScatterElements": {1},
    "ScatterND": {1},
    "ReduceMean": {1},
    "ReduceMax": {1},
    "ReduceMin": {1},
    "ReduceSum": {1},
    "ConstantOfShape": {0},
    "Range": {0, 1, 2},
    "Shape": {0},
    "Size": {0},
    "NonMaxSuppression": {0, 1, 2, 3, 4},
    "RoiAlign": {1, 2},
    "BatchNormalization": {1, 2, 3, 4},
    "Dropout": {1, 2},
}

# Ops whose own output is not quantized even when they are not excluded: the
# control/shape ops above consume float as data only in their slot 0, and
# these produce indices or booleans or are themselves Q/DQ.
_SKIP_OPS = {"QuantizeLinear", "DequantizeLinear", "Shape", "Size", "NonMaxSuppression"}

# Ops whose output values are a subset/rearrangement of their data input's
# (Resize: except cubic, which can overshoot).
_SAME_SCALE_OPS = {
    "Reshape",
    "Transpose",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "MaxPool",
    "Slice",
    "Split",
    "Gather",
    "Expand",
    "Tile",
    "DepthToSpace",
    "SpaceToDepth",
    "Identity",
    "Resize",
}


def _attr(node: onnx.NodeProto, name: str, default):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


# (weight slot, bias slot) for ops whose constant weight is quantized INT8
# symmetric (per channel) and whose bias is INT32 at x_scale * w_scale.
_WEIGHT_OPS = {
    "Conv": (1, 2),
    "ConvTranspose": (1, 2),
    "Gemm": (1, 2),
    "MatMul": (1, None),
}


def _weight_axis(node: onnx.NodeProto, w: np.ndarray) -> Optional[int]:
    """The output-channel axis of ``node``'s weight, or ``None`` when only a
    per-tensor scale is well defined."""
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if node.op_type == "Conv":
        return 0
    if node.op_type == "ConvTranspose":
        # [Cin, Cout/group, k...]: per output channel only without groups
        return 1 if attrs.get("group", 1) == 1 else None
    if node.op_type == "Gemm":
        return 0 if attrs.get("transB", 0) else 1
    if node.op_type == "MatMul":
        return w.ndim - 1 if w.ndim >= 2 else None
    return None


def _activation_qparams(
    lo: float, hi: float, qmin: int, qmax: int
) -> Tuple[float, int]:
    """Asymmetric ``(scale, zero_point)`` for the calibrated ``[lo, hi]`` (widened
    to contain 0) on the integer grid ``[qmin, qmax]``. ``qmin=0`` gives the
    unsigned grid, ``qmin=-128`` / ``-32768`` the signed int8 / int16 one."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo <= 0.0:
        return 1.0, 0
    scale = (hi - lo) / (qmax - qmin)
    zp = int(np.clip(np.round(qmin - lo / scale), qmin, qmax))
    return scale, zp


# Activation dtype -> (ONNX element type, numpy dtype, qmin, qmax).
_ACT_DTYPES = {
    "uint8": (TensorProto.UINT8, np.uint8, 0, 255),
    "int8": (TensorProto.INT8, np.int8, -128, 127),
    "uint16": (TensorProto.UINT16, np.uint16, 0, 65535),
    "int16": (TensorProto.INT16, np.int16, -32768, 32767),
}


# Ops whose output range is bounded to at most [-1, 1] by construction.
_BOUNDED_OUTPUT_OPS = {"Sigmoid", "HardSigmoid", "Softmax", "Tanh"}


def bounded_output_tensors(model: onnx.ModelProto) -> List[str]:
    """Outputs of Sigmoid/HardSigmoid/Softmax/Tanh, which calibration
    should never clip: their range is already at most 1 wide (a uint8 step
    of <= 1/255), so a percentile/entropy/mse clip saves almost no
    resolution, while the rare values it cuts off are what a classifier
    head's scores *are* -- YOLO11n's class Sigmoid is ~672k values per image,
    nearly all ~0, and clipping it at its 99.99th percentile pushes every
    score under the 0.25 detection threshold (0 detections)."""
    return [
        o
        for n in model.graph.node
        if n.op_type in _BOUNDED_OUTPUT_OPS
        for o in n.output
    ]


def list_full_graph_activations(
    model: onnx.ModelProto,
    nodes_to_exclude: Iterable[str] = (),
    op_types_to_exclude: Iterable[str] = (),
) -> List[str]:
    """The float activation tensors :func:`quantize_full_graph` will QDQ --
    what to pass :func:`onnxsim.calibrate` as ``tensor_names``."""
    model = _prepare(model)
    return sorted(_plan(model, set(nodes_to_exclude), set(op_types_to_exclude)))


def _prepare(model: onnx.ModelProto) -> onnx.ModelProto:
    """Shape-inferred copy with ``Constant`` nodes moved into initializers,
    so every constant is found the same way."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
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
    return onnx.shape_inference.infer_shapes(m)


def _elem_types(model: onnx.ModelProto) -> Dict[str, int]:
    g = model.graph
    types = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.HasField("tensor_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    for t in g.initializer:
        types[t.name] = t.data_type
    return types


def _excluded(n: onnx.NodeProto, nodes: Set[str], ops: Set[str]) -> bool:
    return n.name in nodes or n.op_type in ops or n.op_type in _SKIP_OPS


def _plan(model: onnx.ModelProto, nodes: Set[str], ops: Set[str]) -> Set[str]:
    g = model.graph
    types = _elem_types(model)
    inits = {t.name for t in g.initializer}
    wants: Set[str] = set()
    for n in g.node:
        if _excluded(n, nodes, ops):
            continue
        skip = _NON_DATA_INPUTS.get(n.op_type, set())
        for i, x in enumerate(n.input):
            if (
                x
                and i not in skip
                and x not in inits
                and types.get(x) == TensorProto.FLOAT
            ):
                wants.add(x)
        # every float output of a quantized node gets its Q, even when only
        # excluded nodes read it: the NPU's node unit is DQ -> op -> Q
        wants.update(o for o in n.output if o and types.get(o) == TensorProto.FLOAT)
    return wants


def quantize_full_graph(
    model: onnx.ModelProto,
    ranges: Dict[str, Tuple[float, float]],
    per_channel: bool = True,
    nodes_to_exclude: Iterable[str] = (),
    op_types_to_exclude: Iterable[str] = (),
    activation_type: str = "uint8",
) -> onnx.ModelProto:
    """
    QDQ every float activation of ``model`` with the calibrated ``ranges``
    (``{tensor: (min, max)}`` covering :func:`list_full_graph_activations`,
    e.g. from :func:`onnxsim.calibrate` with ``tensor_names=`` that list),
    with INT8 weights and INT32 biases -- see the module docstring.

    :param per_channel: per-output-channel weight scales (free on the HTP)
            rather than one per tensor
    :param nodes_to_exclude: node names left in float
    :param op_types_to_exclude: op types left in float
    :param activation_type: ``"uint8"`` (default), ``"int8"``, ``"uint16"`` or
            ``"int16"``. The signed types quantize onto the full signed grid
            ``[-128, 127]`` / ``[-32768, 32767]`` with the zero point fitted
            the same way (asymmetric); ``uint16`` and ``int16`` need opset >= 21
    :returns: the quantized model
    """
    if activation_type not in _ACT_DTYPES:
        raise ValueError(
            "activation_type must be one of uint8, int8, uint16, int16, "
            f"got {activation_type!r}"
        )
    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0
    )
    if opset < 13:
        raise ValueError(
            f"full-graph QDQ needs opset >= 13 (per-axis DequantizeLinear), got {opset}"
        )
    if activation_type in ("uint16", "int16") and opset < 21:
        raise ValueError(
            f"{activation_type} QuantizeLinear/DequantizeLinear needs opset >= 21"
        )
    _, act_np, qmin, qmax = _ACT_DTYPES[activation_type]

    nodes, ops = set(nodes_to_exclude), set(op_types_to_exclude)
    m = _prepare(model)
    g = m.graph
    acts = _plan(m, nodes, ops)
    missing = sorted(acts - set(ranges))
    if missing:
        raise ValueError(
            f"no calibrated range for {len(missing)} tensors, e.g. {missing[:5]}"
        )

    # Value-preserving ops share their input's scale/zero point (as ORT's QDQ
    # quantizer does): exact (their output needs no calibrated range of its
    # own) and no requantize op for the NPU to insert between them.
    ranges = dict(ranges)
    for n in g.node:
        if (
            n.op_type in _SAME_SCALE_OPS
            and not _excluded(n, nodes, ops)
            and n.input[0] in acts
            and not (n.op_type == "Resize" and _attr(n, "mode", b"nearest") == b"cubic")
        ):
            for o in n.output:
                if o in acts:
                    ranges[o] = ranges[n.input[0]]
    inits = {t.name: t for t in g.initializer}
    types = _elem_types(m)
    graph_inputs = {i.name for i in g.input}
    new_inits: List[onnx.TensorProto] = []
    act_scale: Dict[str, float] = {}

    def scalar(name: str, value, dtype) -> str:
        new_inits.append(numpy_helper.from_array(np.array(value, dtype=dtype), name))
        return name

    def qdq_nodes(src: str, q_name: str, dst: str) -> List[onnx.NodeProto]:
        scale, zp = _activation_qparams(*ranges[q_name], qmin, qmax)
        act_scale[q_name] = scale
        s = scalar(f"{q_name}_scale", scale, np.float32)
        z = scalar(f"{q_name}_zero_point", zp, act_np)
        return [
            helper.make_node(
                "QuantizeLinear",
                [src, s, z],
                [f"{q_name}_quantized"],
                name=f"{q_name}_QuantizeLinear",
            ),
            helper.make_node(
                "DequantizeLinear",
                [f"{q_name}_quantized", s, z],
                [dst],
                name=f"{q_name}_DequantizeLinear",
            ),
        ]

    # Activations: graph inputs are read through a new DQ output; a node
    # output keeps its name on the DQ, its producer writes `<name>_float`.
    input_rename = {x: f"{x}_dequantized" for x in acts if x in graph_inputs}
    out_nodes: List[onnx.NodeProto] = []
    for x in sorted(input_rename):
        out_nodes += qdq_nodes(x, x, input_rename[x])

    const_cache: Dict[Tuple, str] = {}

    def dq_const(name: str, key: Tuple, q: np.ndarray, scale, zp, axis, dtype) -> str:
        if key in const_cache:
            return const_cache[key]
        tag = "_".join(str(k) for k in key[1:])
        qn = f"{name}_{tag}_quantized"
        new_inits.append(numpy_helper.from_array(q.astype(dtype), qn))
        s = f"{qn}_scale"
        z = f"{qn}_zero_point"
        new_inits.append(
            numpy_helper.from_array(np.asarray(scale, dtype=np.float32), s)
        )
        new_inits.append(numpy_helper.from_array(np.asarray(zp, dtype=dtype), z))
        dst = f"{name}_{tag}_dequantized"
        attrs = {"axis": axis} if axis is not None else {}
        const_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [qn, s, z],
                [dst],
                name=f"{dst}_DequantizeLinear",
                **attrs,
            )
        )
        const_cache[key] = dst
        return dst

    def weight(node: onnx.NodeProto, name: str) -> Tuple[str, np.ndarray]:
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        axis = _weight_axis(node, w) if per_channel else None
        if axis is None:
            s = np.float32(max(float(np.abs(w).max()), 1e-12) / 127.0)
            q = np.clip(np.round(w / s), -127, 127)
            return dq_const(
                name, (name, "w"), q, s, np.int8(0), None, np.int8
            ), np.array([s])
        red = tuple(i for i in range(w.ndim) if i != axis)
        s = (np.maximum(np.abs(w).max(axis=red), 1e-12) / 127.0).astype(np.float32)
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / s.reshape(shape)), -127, 127)
        zp = np.zeros(s.shape, dtype=np.int8)
        return dq_const(name, (name, "w", axis), q, s, zp, axis, np.int8), s

    def bias(
        name: str, x_scale: float, w_scale: np.ndarray, consumer: str
    ) -> Optional[str]:
        b = numpy_helper.to_array(inits[name]).astype(np.float64)
        if b.ndim != 1 or w_scale.size not in (1, b.size):
            return None
        s = (x_scale * w_scale.astype(np.float64)).astype(np.float32)
        s = np.maximum(s, np.float32(1e-30))
        q = np.clip(np.round(b / s), -(2**31), 2**31 - 1)
        s_arr = s if s.size > 1 else s.reshape(())
        zp = np.zeros(s_arr.shape, dtype=np.int32)
        return dq_const(
            name,
            (name, "b", consumer),
            q,
            s_arr,
            zp,
            0 if s.size > 1 else None,
            np.int32,
        )

    def const_act(name: str) -> str:
        c = numpy_helper.to_array(inits[name]).astype(np.float32)
        scale, zp = _activation_qparams(
            c.min() if c.size else 0.0, c.max() if c.size else 0.0, qmin, qmax
        )
        q = np.clip(np.round(c / scale) + zp, qmin, qmax)
        return dq_const(
            name, (name, "c"), q, np.float32(scale), act_np(zp), None, act_np
        )

    for n in g.node:
        const_nodes: List[onnx.NodeProto] = []
        excluded = _excluded(n, nodes, ops)
        skip = _NON_DATA_INPUTS.get(n.op_type, set())
        wslot, bslot = _WEIGHT_OPS.get(n.op_type, (None, None))
        new_inputs = list(n.input)
        w_scale = None
        for i, x in enumerate(n.input):
            if x in input_rename:
                new_inputs[i] = input_rename[x]
        if not excluded:
            for i, x in enumerate(n.input):
                if (
                    not x
                    or x not in inits
                    or i in skip
                    or types.get(x) != TensorProto.FLOAT
                ):
                    continue
                if i == wslot:
                    new_inputs[i], w_scale = weight(n, x)
                elif i == bslot:
                    xs = act_scale.get(n.input[0]) if n.input[0] in acts else None
                    if xs is not None and w_scale is not None:
                        b = bias(x, xs, w_scale, n.name or n.output[0])
                        if b is not None:
                            new_inputs[i] = b
                            continue
                    new_inputs[i] = const_act(x)
                else:
                    new_inputs[i] = const_act(x)
        del n.input[:]
        n.input.extend(new_inputs)
        out_nodes += const_nodes
        tail: List[onnx.NodeProto] = []
        for j, o in enumerate(n.output):
            if o in acts:
                n.output[j] = f"{o}_float"
                tail += qdq_nodes(n.output[j], o, o)
        out_nodes.append(n)
        out_nodes += tail

    del g.node[:]
    g.node.extend(out_nodes)
    used = {x for n in g.node for x in n.input}
    kept = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(kept + new_inits)
    # value_info of renamed producer outputs is still right for the DQ that
    # took their name; drop inferred value_info so nothing stale lingers.
    del g.value_info[:]
    return m
