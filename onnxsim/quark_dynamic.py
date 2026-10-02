"""Dynamic-activation quantization, the graph Quark's ``UINT8_DYNAMIC_QUANT``
preset produces (ONNX Runtime's ``quantize_dynamic`` pattern): constant weights
become per-tensor integers, activations are quantized at run time.

- ``MatMul(x, W)`` -> ``DynamicQuantizeLinear(x)``, ``MatMulInteger``, ``Cast``,
  ``Mul(x_scale, w_scale)``, ``Mul`` -- the output is the int32 accumulator
  times both scales.
- ``Conv(x, W[, b])`` -> the same with ``ConvInteger``; the bias is added in
  float afterwards (reshaped to ``[1, C, 1, 1...]``). Only ``group == 1``.
- ``Gemm(x, W[, b])`` with ``alpha == beta == 1`` and ``transA == 0`` is first
  split into ``MatMul`` (+ ``Transpose`` of a constant ``W`` folded away) and
  ``Add``, as Quark does.

Weights: ``uint8`` asymmetric (Quark's default, zero point ``round(-min /
scale)``) or ``int8`` symmetric (zero point 0). ``DynamicQuantizeLinear`` is
uint8 asymmetric by definition. Below opset 11 (no ``DynamicQuantizeLinear``) the
scale and zero point are computed with ``ReduceMin`` / ``ReduceMax`` / ``Sub`` /
``Div`` / ``Floor`` / ``Cast`` nodes and a ``QuantizeLinear``, as ONNX Runtime's
quantizer (and Quark) do; ``MatMulInteger`` / ``ConvInteger`` need opset 10.

Independent implementation; the node pattern, naming and weight parameters
were checked against Quark's output (``tests/test_quark_parity.py``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _weight_params(w: np.ndarray, weight_dtype: str):
    if weight_dtype == "int8":
        absmax = max(float(np.abs(w).max()), 1e-12)
        scale = absmax / 127.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        return q, np.float32(scale), np.int8(0), TensorProto.INT8
    # (float32 arithmetic, as ONNX Runtime's ``compute_scale_zp`` does on float32
    # weights: the scale differs from a float64 quotient in the last bit)
    lo = np.float32(min(float(w.min()), 0.0))
    hi = np.float32(max(float(w.max()), 0.0))
    scale = np.float32((hi - lo) / np.float32(255.0))
    if not scale > 0:
        scale = np.float32(1.0)
    zp = int(np.clip(np.round(-lo / scale), 0, 255))
    q = np.clip(np.round(w / scale) + zp, 0, 255).astype(np.uint8)
    return q, scale, np.uint8(zp), TensorProto.UINT8


def quantize_dynamic_integer(
    model: onnx.ModelProto,
    weight_dtype: str = "uint8",
    exclude_nodes: Iterable[str] = (),
    op_types: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Return ``model`` with dynamic-activation integer quantization applied
    (see the module docstring).

    :param weight_dtype: ``"uint8"`` (asymmetric) or ``"int8"`` (symmetric)
    :param exclude_nodes: node names (or first-output names) kept in float
    :param op_types: restrict to these of ``MatMul`` / ``Conv`` / ``Gemm``
    """
    if weight_dtype not in ("uint8", "int8"):
        raise ValueError("weight_dtype must be 'uint8' or 'int8'")
    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0
    )
    # (below opset 11 there is no DynamicQuantizeLinear: ONNX Runtime's quantizer,
    # and Quark's, compute the scale and zero point with ordinary operators)
    fused = opset >= 11
    allowed = set(op_types) if op_types is not None else {"MatMul", "Conv", "Gemm"}
    skip: Set[str] = set(exclude_nodes)

    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits: Dict[str, onnx.TensorProto] = {t.name: t for t in g.initializer}
    users: Dict[str, int] = {}
    for n in g.node:
        for x in n.input:
            users[x] = users.get(x, 0) + 1

    def const_float(name: str) -> Optional[np.ndarray]:
        t = inits.get(name)
        if t is None or t.data_type != TensorProto.FLOAT:
            return None
        return numpy_helper.to_array(t)

    def attr(n: onnx.NodeProto, name: str, default):
        for a in n.attribute:
            if a.name == name:
                return helper.get_attribute_value(a)
        return default

    new_inits: List[onnx.TensorProto] = []
    dead: Set[str] = set()
    out_nodes: List[onnx.NodeProto] = []
    done_weights: Dict[str, tuple] = {}

    def quantized_weight(wname: str, w: np.ndarray):
        if wname not in done_weights:
            q, s, zp, dt = _weight_params(w, weight_dtype)
            for suffix, arr in (("_quantized", q), ("_scale", s), ("_zero_point", zp)):
                new_inits.append(
                    numpy_helper.from_array(np.asarray(arr), wname + suffix)
                )
            done_weights[wname] = (
                wname + "_quantized",
                wname + "_scale",
                wname + "_zero_point",
            )
        return done_weights[wname]

    unfused_consts: List[str] = []

    def unfused_scale_and_zero_point(x: str) -> List[onnx.NodeProto]:
        """ONNX Runtime's ``_get_dynamic_input_quantization_params_uint8``:
        ``scale = (max - min) / 255``, ``zero_point = cast(floor((0 - min) / scale))``."""
        if not unfused_consts:
            unfused_consts.extend(["fixed_quantization_range_uint8", "fixed_zero"])
            for name, value in zip(unfused_consts, (255.0, 0.0)):
                new_inits.append(
                    helper.make_tensor(name, TensorProto.FLOAT, [], [value])
                )
        scale, zero_point = x + "_scale", x + "_zero_point"
        rmin, rmax = x + "_ReduceMin", x + "_ReduceMax"
        scale_sub, zp_sub = x + "_scale_Sub", x + "_zero_point_Sub"
        zp_div, zp_floor = x + "_zero_point_Div", x + "_zero_point_Floor"
        return [
            helper.make_node("ReduceMin", [x], [rmin + ":0"], rmin, keepdims=0),
            helper.make_node("ReduceMax", [x], [rmax + ":0"], rmax, keepdims=0),
            helper.make_node(
                "Sub", [rmax + ":0", rmin + ":0"], [scale_sub + ":0"], scale_sub
            ),
            helper.make_node(
                "Div",
                [scale_sub + ":0", "fixed_quantization_range_uint8"],
                [scale],
                x + "_scale_Div",
            ),
            helper.make_node(
                "Sub", ["fixed_zero", rmin + ":0"], [zp_sub + ":0"], zp_sub
            ),
            helper.make_node("Div", [zp_sub + ":0", scale], [zp_div + ":0"], zp_div),
            helper.make_node("Floor", [zp_div + ":0"], [zp_floor + ":0"], zp_floor),
            helper.make_node(
                "Cast",
                [zp_floor + ":0"],
                [zero_point],
                x + "_zero_point_Cast",
                to=TensorProto.UINT8,
            ),
        ]

    def dynamic_input(x: str, cache: Dict[str, tuple]):
        if x not in cache:
            names = (x + "_quantized", x + "_scale", x + "_zero_point")
            if fused:
                out_nodes.append(
                    helper.make_node(
                        "DynamicQuantizeLinear",
                        [x],
                        list(names),
                        name=x + "_QuantizeLinear",
                    )
                )
            else:
                out_nodes.extend(unfused_scale_and_zero_point(x))
                out_nodes.append(
                    helper.make_node(
                        "QuantizeLinear",
                        [x, names[1], names[2]],
                        [names[0]],
                        name=x + "_QuantizeLinear",
                    )
                )
            cache[x] = names
        return cache[x]

    cache: Dict[str, tuple] = {}
    for n in g.node:
        label = n.name or (n.output[0] if n.output else "")
        if (
            n.op_type not in allowed
            or n.domain not in ("", "ai.onnx")
            or label in skip
            or (n.output and n.output[0] in skip)
        ):
            out_nodes.append(n)
            continue

        if n.op_type == "Gemm":
            w = const_float(n.input[1]) if len(n.input) > 1 else None
            if (
                w is None
                or w.ndim != 2
                or attr(n, "alpha", 1.0) != 1.0
                or attr(n, "beta", 1.0) != 1.0
                or attr(n, "transA", 0)
            ):
                out_nodes.append(n)
                continue
            if attr(n, "transB", 0):
                w = np.ascontiguousarray(w.T)
            wname = n.input[1] + ("_T" if attr(n, "transB", 0) else "")
            bias = n.input[2] if len(n.input) > 2 and n.input[2] else None
            x = n.input[0]
            mm_out = n.output[0] + "_MatMul"
            kind, wq_names = "MatMulInteger", quantized_weight(wname, w)
            main_out = mm_out
        elif n.op_type == "MatMul":
            w = const_float(n.input[1]) if len(n.input) > 1 else None
            if w is None or w.ndim < 2:
                out_nodes.append(n)
                continue
            wname, bias, x = n.input[1], None, n.input[0]
            kind, wq_names = "MatMulInteger", quantized_weight(wname, w)
            main_out = n.output[0]
        else:  # Conv
            w = const_float(n.input[1]) if len(n.input) > 1 else None
            if w is None or attr(n, "group", 1) != 1:
                out_nodes.append(n)
                continue
            wname, x = n.input[1], n.input[0]
            bias = n.input[2] if len(n.input) > 2 and n.input[2] else None
            kind, wq_names = "ConvInteger", quantized_weight(wname, w)
            main_out = n.output[0] + "quant_scaled_output" if bias else n.output[0]

        xq, xs, xz = dynamic_input(x, cache)
        wq, ws, wz = wq_names
        acc = n.output[0] + "_quantized"
        int_node = helper.make_node(
            kind, [xq, wq, xz, wz], [acc], name=(n.name + "_quant") if n.name else ""
        )
        if kind == "ConvInteger":
            for a in n.attribute:  # strides, pads, dilations, kernel_shape, auto_pad
                int_node.attribute.append(a)
        cast_out = acc + "_cast_output"
        scale_prod = f"{xs}_{ws}_mul:0"
        out_nodes += [
            int_node,
            helper.make_node("Cast", [acc], [cast_out], to=TensorProto.FLOAT),
            helper.make_node("Mul", [xs, ws], [scale_prod]),
            helper.make_node("Mul", [cast_out, scale_prod], [main_out]),
        ]
        if bias is not None:
            if n.op_type == "Conv":
                b = const_float(bias)
                if b is None:
                    raise ValueError(f"{label}: Conv bias must be constant")
                shape_name = n.output[0] + "_bias_reshape_shape"
                rank = w.ndim
                new_inits.append(
                    numpy_helper.from_array(
                        np.array([1, -1] + [1] * (rank - 2), np.int64), shape_name
                    )
                )
                reshaped = n.output[0] + "_bias_reshape_output"
                out_nodes += [
                    helper.make_node("Reshape", [bias, shape_name], [reshaped]),
                    helper.make_node("Add", [main_out, reshaped], [n.output[0]]),
                ]
            else:
                out_nodes.append(
                    helper.make_node("Add", [main_out, bias], [n.output[0]])
                )
        dead.add(n.input[1])

    # a weight read by an untouched node must stay
    still_used = {x for n in out_nodes for x in n.input}
    for t in list(g.initializer):
        if t.name in dead and t.name not in still_used:
            g.initializer.remove(t)
    del g.node[:]
    g.node.extend(out_nodes)
    g.initializer.extend(new_inits)
    return m


__all__ = ["quantize_dynamic_integer"]
