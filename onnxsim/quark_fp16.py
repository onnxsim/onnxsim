"""Quark's float16 detection and conversion of quantized FP16 graphs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxsim.quark_marking import quark_sorted


def is_fp16_model(model: onnx.ModelProto) -> bool:
    types = [
        v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.output)
    ]
    types.extend(t.data_type for t in model.graph.initializer)
    return TensorProto.FLOAT16 in types and TensorProto.FLOAT not in types


def convert_fp16_scale_to_fp32(
    model: onnx.ModelProto, include: Sequence[str] = (), exclude: Sequence[str] = ()
) -> onnx.ModelProto:
    """Convert internal FP16 tensors, retaining the FP16 interface with Casts.

    Quark's legacy UseFP32Scale option converts the whole selected computation,
    including FP16 zero points; BF16 type markers keep their original type.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    nodes = list(g.node)
    by_name = {n.name: n for n in nodes if n.name}
    excluded = set(exclude) & by_name.keys()
    included = set(include) & by_name.keys()
    converted = [n for n in nodes if n.name and n.name not in excluded]
    readers = {}
    producers = {y: n for n in nodes for y in n.output}
    for n in nodes:
        for x in n.input:
            readers.setdefault(x, []).append(n)
    inputs = {v.name for v in g.input}
    values = {v.name: v for v in (*g.input, *g.value_info, *g.output)}
    init_names = {x for n in converted for x in n.input}
    for t in g.initializer:
        if t.name in init_names and t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(
                numpy_helper.from_array(
                    numpy_helper.to_array(t).astype(np.float32), t.name
                )
            )
    for n in converted:
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(
                    numpy_helper.from_array(
                        numpy_helper.to_array(a.t).astype(np.float32), a.t.name
                    )
                )
        for x in n.input:
            if (
                x not in inputs
                and x in values
                and values[x].type.tensor_type.elem_type == TensorProto.FLOAT16
            ):
                values[x].type.tensor_type.elem_type = TensorProto.FLOAT
    for v in g.input:
        if v.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        for i, n in enumerate(readers.get(v.name, [])):
            if n.name in excluded:
                continue
            name = f"{v.name}_Cast_{i}"
            out = name + "_output"
            for j, x in enumerate(n.input):
                if x == v.name:
                    n.input[j] = out
            g.node.append(
                helper.make_node(
                    "Cast", [v.name], [out], name=name, to=TensorProto.FLOAT
                )
            )
    if not (included or excluded):
        for v in g.output:
            if (
                v.type.tensor_type.elem_type != TensorProto.FLOAT16
                or v.name not in producers
            ):
                continue
            name = v.name + "_Cast"
            src = name + "_input"
            n = producers[v.name]
            for j, y in enumerate(n.output):
                if y == v.name:
                    n.output[j] = src
            g.node.append(
                helper.make_node(
                    "Cast", [src], [v.name], name=name, to=TensorProto.FLOAT16
                )
            )
    else:
        for n in converted:
            if n.name not in included and n.op_type not in (
                "DequantizeLinear",
                "ExtendedDequantizeLinear",
            ):
                continue
            for y in n.output:
                if (
                    y in values
                    and values[y].type.tensor_type.elem_type != TensorProto.FLOAT16
                ):
                    continue
                for i, child in enumerate(readers.get(y, [])):
                    if excluded and child.name not in excluded:
                        continue
                    name = f"{y}_Cast_{i}"
                    out = name + "_output"
                    for j, x in enumerate(child.input):
                        if x == y:
                            child.input[j] = out
                    g.node.append(
                        helper.make_node(
                            "Cast", [y], [out], name=name, to=TensorProto.FLOAT16
                        )
                    )
    return quark_sorted(m)
