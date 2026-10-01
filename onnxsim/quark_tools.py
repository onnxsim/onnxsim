"""Post-quantization graph utilities named after the scripts in
``quark.onnx.tools`` (``remove_qdq``, ``convert_shared_initializer_to_unique``,
``convert_dynamic_to_fixed``, ``replace_inf_weights``). Independent
implementations: Quark's source was read for the names and intent only.

Every function takes and returns an ``onnx.ModelProto`` (the input is not
modified) and only rewrites the **top-level graph** -- nodes inside
control-flow subgraphs are left alone.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

_Q_OPS = {"QuantizeLinear"}
_DQ_OPS = {"DequantizeLinear"}
_QDQ_DOMAINS = ("", "ai.onnx", "com.microsoft")


def _copy(model: onnx.ModelProto) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _is(node: onnx.NodeProto, ops: set) -> bool:
    return node.op_type in ops and node.domain in _QDQ_DOMAINS


def _dequantize(
    q: np.ndarray, scale: np.ndarray, zp: Optional[np.ndarray], axis: int
) -> np.ndarray:
    x = q.astype(np.float32)
    z = np.zeros((), np.float32) if zp is None else zp.astype(np.float32)
    s = scale.astype(np.float32)
    if s.ndim == 1 and s.size > 1:
        shape = [1] * x.ndim
        shape[axis % x.ndim] = -1
        s = s.reshape(shape)
        if z.ndim == 1 and z.size > 1:
            z = z.reshape(shape)
    return ((x - z) * s).astype(np.float32)


def remove_qdq(model: onnx.ModelProto, fold_weights: bool = True) -> onnx.ModelProto:
    """Strip quantization from a QDQ model, returning a float32 graph.

    - ``QuantizeLinear -> DequantizeLinear`` pairs are removed and their
      consumers wired to the pair's float input (a pair whose DQ output is a
      graph output keeps that name through an ``Identity``).
    - With ``fold_weights``, a ``DequantizeLinear`` over constant
      initializers is evaluated and replaced by a float32 initializer of the
      same name as its output; the now-unused quantized tensor, scale and
      zero-point initializers are dropped.

    A Q whose output feeds anything other than ``DequantizeLinear``, and
    blocked (``block_size``) dequantization, are left untouched.
    """
    m = _copy(model)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    graph_outputs = {o.name for o in g.output}
    removed_inputs: List[str] = []
    drop = set()  # ids of nodes to remove
    replace_with: Dict[str, str] = {}  # DQ output name -> float source name
    identities: Dict[int, onnx.NodeProto] = {}  # id(DQ node) -> Identity to emit
    new_inits: List[onnx.TensorProto] = []

    consumers = defaultdict(list)
    for n in g.node:
        for x in n.input:
            consumers[x].append(n)

    if fold_weights:
        for n in g.node:
            if not _is(n, _DQ_OPS) or n.output[0] in graph_outputs:
                continue
            ins = list(n.input) + [""] * (3 - len(n.input))
            x, s, z = ins[:3]
            if x not in inits or s not in inits or (z and z not in inits):
                continue
            if any(a.name == "block_size" and a.i > 0 for a in n.attribute):
                continue
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            w = _dequantize(
                numpy_helper.to_array(inits[x]),
                numpy_helper.to_array(inits[s]),
                numpy_helper.to_array(inits[z]) if z else None,
                axis,
            )
            new_inits.append(numpy_helper.from_array(w, n.output[0]))
            drop.add(id(n))
            removed_inputs += [i for i in (x, s, z) if i]

    for q in g.node:
        if not _is(q, _Q_OPS) or q.output[0] in graph_outputs:
            continue
        users = consumers[q.output[0]]
        if not users or not all(_is(u, _DQ_OPS) and id(u) not in drop for u in users):
            continue
        drop.add(id(q))
        removed_inputs += [i for i in q.input[1:] if i]
        for dq in users:
            drop.add(id(dq))
            removed_inputs += [i for i in dq.input[1:] if i]
            if dq.output[0] in graph_outputs:
                identities[id(dq)] = onnx.helper.make_node(
                    "Identity", [q.input[0]], [dq.output[0]]
                )
            else:
                replace_with[dq.output[0]] = q.input[0]

    def resolve(name: str) -> str:
        while name in replace_with:
            name = replace_with[name]
        return name

    kept: List[onnx.NodeProto] = []
    for n in g.node:
        if id(n) in identities:
            kept.append(identities[id(n)])
        elif id(n) not in drop:
            for i, x in enumerate(n.input):
                if x in replace_with:
                    n.input[i] = resolve(x)
            kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    g.initializer.extend(new_inits)

    referenced = {x for n in g.node for x in n.input} | graph_outputs
    unused = {i for i in removed_inputs if i not in referenced}
    keep_inits = [i for i in g.initializer if i.name not in unused]
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    return m


def convert_shared_initializer_to_unique(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give each node its own copy of an initializer that several nodes use
    (the first consumer keeps the original name; the others get
    ``<name>_copy<k>``)."""
    m = _copy(model)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    seen: Dict[str, int] = defaultdict(int)
    for n in g.node:
        renamed: Dict[str, str] = {}
        for i, x in enumerate(n.input):
            if x not in inits:
                continue
            if x in renamed:  # same tensor twice in one node: share one copy
                n.input[i] = renamed[x]
                continue
            k = seen[x]
            seen[x] += 1
            if k == 0:
                renamed[x] = x
                continue
            new_name = f"{x}_copy{k}"
            dup = onnx.TensorProto()
            dup.CopyFrom(inits[x])
            dup.name = new_name
            g.initializer.append(dup)
            renamed[x] = new_name
            n.input[i] = new_name
    return m


def convert_dynamic_to_fixed(
    model: onnx.ModelProto, input_shapes: Dict[str, Sequence[int]]
) -> onnx.ModelProto:
    """Fix graph input shapes (``{name: [dims]}``), drop stale intermediate
    and output shape info, and re-run shape inference so static shapes
    propagate."""
    m = _copy(model)
    g = m.graph
    by_name = {i.name: i for i in g.input}
    for name, dims in input_shapes.items():
        if name not in by_name:
            raise ValueError(f"{name!r} is not a graph input")
        shape = by_name[name].type.tensor_type.shape
        if len(shape.dim) != len(dims):
            raise ValueError(
                f"{name!r} has rank {len(shape.dim)}, got {len(dims)} dims"
            )
        for d, v in zip(shape.dim, dims):
            d.ClearField("dim_param")
            d.dim_value = int(v)
    del g.value_info[:]
    for o in g.output:
        for d in o.type.tensor_type.shape.dim:
            d.Clear()
    return onnx.shape_inference.infer_shapes(m)


def replace_inf_weights(
    model: onnx.ModelProto, max_value: float = 3.0e38
) -> onnx.ModelProto:
    """Clamp +-inf in float initializers to +-``max_value`` (NaN is kept)."""
    m = _copy(model)
    for t in m.graph.initializer:
        if t.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
            continue
        arr = numpy_helper.to_array(t)
        if np.isfinite(arr).all() or not np.isinf(arr).any():
            continue
        fixed = np.where(np.isposinf(arr), max_value, arr)
        fixed = np.where(np.isneginf(arr), -max_value, fixed).astype(arr.dtype)
        t.CopyFrom(numpy_helper.from_array(fixed, t.name))
    return m


__all__ = [
    "convert_dynamic_to_fixed",
    "convert_shared_initializer_to_unique",
    "remove_qdq",
    "replace_inf_weights",
]
