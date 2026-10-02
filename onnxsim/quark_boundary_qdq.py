"""Quantizer pairs at precision boundaries -- a port of the post-processing AMD
Quark's AutoMixprecision applies with ``dual_quant_nodes=True``
(``insert_quant_nodes_at_boundaries``). Independent implementation: Quark's
source was read for the contract.

Quark's mixing leaves a node that is promoted to another precision reading
tensors that another precision produced (and the other way round): each
quantizer chain is edited in place, so the node consumes one precision and
produces the other. With ``dual_quant_nodes`` it then walks the *final* model
once (the candidates are scored without these nodes) and, for every node,

1. collects its *stages*: for every activation input, the Q/DQ pair (or the
   ``BFPQuantizeDequantize`` / ``MXQuantizeDequantize`` node) that produces it
   (constants and graph inputs without a quantizer are skipped), and for every
   output, the Q/DQ pair or block node that consumes it;
2. picks a *template* stage -- for a promoted node (or when no node is
   promoted) the first stage whose tensor is a promoted one, for an unpromoted
   node the first stage whose tensor is not; a node with no such stage borrows
   one from a neighbour that shares the tensor;
3. inserts, in front of every stage that differs from the template (different
   quantizer op, domain, zero point type or block attributes), a copy of the
   template's quantizer: a Q/DQ pair gets a scale and zero point recomputed
   from the tensor's calibrated range at the template's type (``int8`` /
   ``int16`` / half types symmetric, the unsigned ones asymmetric,
   ``compute_scale_zp`` / ``compute_scale_zp_fp``), a block node is copied
   as is.

So a tensor crossing a boundary is quantized twice, to the neighbour's
precision first and then to its own.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxsim.quark_mixing import (
    COP_DOMAIN,
    DQ_OPS,
    FN_OPS,
    Q_OPS,
    _toposort,
    compute_scale_zp,
    compute_scale_zp_fp,
)

_QTYPE_NAME = {
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.INT16: "int16",
    TensorProto.UINT16: "uint16",
    TensorProto.INT32: "int32",
    TensorProto.UINT32: "uint32",
    TensorProto.FLOAT16: "float16",
    TensorProto.BFLOAT16: "bfloat16",
}
_SYMMETRIC = {
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
}
_HALF = {TensorProto.FLOAT16, TensorProto.BFLOAT16}

Stage = Dict[str, Any]


def _canon(name: str) -> str:
    """The calibrated tensor behind onnxsim's ``<name>/f`` quantizer input."""
    return name[:-2] if name.endswith("/f") else name


def _unique(base: str, existing: Set[str]) -> str:
    if base not in existing:
        existing.add(base)
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    existing.add(f"{base}_{i}")
    return f"{base}_{i}"


def _attr_key(attr: onnx.AttributeProto) -> Tuple[str, Any]:
    if attr.type == onnx.AttributeProto.INT:
        return attr.name, attr.i
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.name, attr.f
    if attr.type == onnx.AttributeProto.STRING:
        return attr.name, attr.s
    return attr.name, repr(attr)


def insert_boundary_quant_nodes(
    model: onnx.ModelProto,
    ranges: Optional[Callable[[str], Any]],
    promoted_tensors: Iterable[str] = (),
    promoted_nodes: Iterable[str] = (),
    override_tensors: Iterable[str] = (),
    mixed_nodes: Iterable[str] = (),
) -> onnx.ModelProto:
    """``model`` with quantizer pairs inserted at its precision boundaries.

    :param ranges: ``f(float tensor name) -> (lo, hi)`` or ``None``: the
            calibrated range a boundary pair's scale / zero point come from (a
            tensor without one keeps the template's scale and zero point)
    :param promoted_tensors: float tensors the mixing step promoted (Quark's
            ``promoted_tensors``: they count as tensors with overrides)
    :param promoted_nodes: the promoted nodes' names
    :param override_tensors: tensors with quantization overrides of their own
            (Quark's ``TensorQuantOverrides``) besides the promoted ones
    :param mixed_nodes: nodes with a precision of their own (Quark's
            ``NodesWithMixedPrecision`` option), besides the promoted ones
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    nodes: List[onnx.NodeProto] = []
    for n in m.graph.node:
        c = onnx.NodeProto()
        c.CopyFrom(n)
        nodes.append(c)
    del m.graph.node[:]
    inits: Dict[str, TensorProto] = {t.name: t for t in m.graph.initializer}
    overrides = {_canon(t) for t in promoted_tensors} | {
        _canon(t) for t in override_tensors
    }
    mixed = set(mixed_nodes) | set(promoted_nodes)

    out_to_node: Dict[str, onnx.NodeProto] = {}
    in_to_nodes: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for o in n.output:
            out_to_node[o] = n
        for x in n.input:
            in_to_nodes.setdefault(x, []).append(n)

    tensor_names: Set[str] = set(inits)
    tensor_names.update(i.name for i in m.graph.input)
    tensor_names.update(o.name for o in m.graph.output)
    for n in nodes:
        tensor_names.update(x for x in n.input if x)
        tensor_names.update(x for x in n.output if x)
    node_names = {n.name for n in nodes if n.name}

    def has_override(name: str) -> bool:
        return bool(overrides) and _canon(name) in overrides

    def qtype_of(q: onnx.NodeProto) -> Optional[int]:
        if len(q.input) >= 3 and q.input[2] in inits:
            return inits[q.input[2]].data_type
        return None

    def quant_signature(q: onnx.NodeProto):
        return ("quant", f"{q.domain or 'ai.onnx'}::{q.op_type}", qtype_of(q))

    def fn_signature(n: onnx.NodeProto):
        attrs = tuple(sorted(_attr_key(a) for a in n.attribute))
        return ("fn", f"{n.domain or COP_DOMAIN}::{n.op_type}", attrs)

    def upstream(input_name: str):
        prod = out_to_node.get(input_name)
        if prod is not None and prod.op_type in DQ_OPS:
            q = out_to_node.get(prod.input[0])
            if q is not None and q.op_type in Q_OPS:
                src = q.input[0] if q.input and q.input[0] else input_name
                return "pair", quant_signature(q), (q, prod), src
        if prod is not None and prod.op_type in FN_OPS:
            src = prod.input[0] if prod.input and prod.input[0] else input_name
            return "fn", fn_signature(prod), (prod,), src
        return None

    def downstream(node: onnx.NodeProto):
        if node.op_type in Q_OPS:
            dq = next(
                (c for c in in_to_nodes.get(node.output[0], []) if c.op_type in DQ_OPS),
                None,
            )
            if dq is not None:
                target = dq.output[0] if dq.output and dq.output[0] else ""
                return "pair", quant_signature(node), (node, dq), target
        if node.op_type in FN_OPS:
            target = node.output[0] if node.output and node.output[0] else ""
            return "fn", fn_signature(node), (node,), target
        return None

    def override_name(name: str, other: str) -> Optional[str]:
        for cand in [name] + ([other] if other and other != name else []):
            if has_override(cand):
                return cand
        return None

    def scale_zp(tensor: str, qtype: int, scale_dtype: Any):
        r = ranges(tensor) if ranges is not None else None
        if r is None:
            return None
        if qtype not in _QTYPE_NAME:
            raise NotImplementedError(f"boundary quantizer of type {qtype}")
        dtype = _QTYPE_NAME[qtype]
        rmin, rmax = np.asarray(r[0]), np.asarray(r[1])
        if rmin.dtype != np.float16:
            rmin, rmax = rmin.astype(np.float32), rmax.astype(np.float32)
        symmetric = qtype in _SYMMETRIC
        if qtype in _HALF:
            zp, scale = compute_scale_zp_fp(rmin, rmax, dtype, symmetric)
            zp_dtype = helper.tensor_dtype_to_np_dtype(qtype)
        else:
            zp, scale = compute_scale_zp(rmin, rmax, dtype, symmetric, False)
            zp_dtype = helper.tensor_dtype_to_np_dtype(qtype)
        s_name = _unique(f"{tensor}_additional_scale", tensor_names)
        z_name = _unique(f"{tensor}_additional_zero_point", tensor_names)
        for name, arr in (
            (s_name, np.asarray(scale, dtype=scale_dtype).reshape(())),
            (z_name, np.asarray(zp, dtype=zp_dtype).reshape(())),
        ):
            t = numpy_helper.from_array(arr, name)
            inits[name] = t
            m.graph.initializer.append(t)
        return s_name, z_name

    def insert_between(
        source: str,
        target: onnx.NodeProto,
        kind: str,
        stage_nodes: Tuple[onnx.NodeProto, ...],
        qparam_tensor: str,
        scope: str,
    ) -> None:
        if kind == "pair":
            q_t, dq_t = stage_nodes
            new_q, new_dq = copy.deepcopy(q_t), copy.deepcopy(dq_t)
            q_base = f"{scope}_additional_{q_t.op_type}"
            dq_base = f"{scope}_additional_{dq_t.op_type}"
            new_q.name = _unique(q_base, node_names)
            new_dq.name = _unique(dq_base, node_names)
            q_out = _unique(f"{q_base}_output", tensor_names)
            dq_out = _unique(f"{dq_base}_output", tensor_names)
            qtype = qtype_of(q_t)
            scale_dtype: Any = np.float32
            if len(q_t.input) >= 2 and q_t.input[1] in inits:
                scale_dtype = helper.tensor_dtype_to_np_dtype(
                    inits[q_t.input[1]].data_type
                )
            if qtype is not None:
                made = scale_zp(qparam_tensor, qtype, scale_dtype)
                if made is not None and len(new_q.input) >= 3:
                    new_q.input[1], new_q.input[2] = made
                    if len(new_dq.input) >= 3:
                        new_dq.input[1], new_dq.input[2] = made
            new_q.input[0] = source
            new_q.output[0] = q_out
            new_dq.input[0] = q_out
            new_dq.output[0] = dq_out
            for j, x in enumerate(target.input):
                if x == source:
                    target.input[j] = dq_out
            nodes.extend([new_q, new_dq])
            return
        (fn_t,) = stage_nodes
        new_fn = copy.deepcopy(fn_t)
        base = f"{scope}_additional_{fn_t.op_type}"
        new_fn.name = _unique(base, node_names)
        out = _unique(f"{base}_output", tensor_names)
        new_fn.input[0] = source
        new_fn.output[0] = out
        for j, x in enumerate(target.input):
            if x == source:
                target.input[j] = out
        nodes.append(new_fn)

    # phase 1: every node's upstream / downstream stages
    infos: Dict[str, List[Stage]] = {}
    for idx, node in enumerate(list(nodes)):
        if node.op_type in Q_OPS + DQ_OPS + FN_OPS:
            continue
        node_name = node.name or node.op_type
        stages: List[Stage] = []
        infos[f"{node_name}_{idx}"] = stages
        for i, name in enumerate(node.input):
            if not name:
                continue
            up = upstream(name)
            if up is None:
                continue
            kind, sig, snodes, src = up
            if src in inits:
                continue
            stages.append(
                dict(
                    node_name=node_name,
                    tensor_type="input",
                    tensor_index=i,
                    tensor_name=name,
                    override=override_name(name, src),
                    calib=src,
                    kind=kind,
                    sig=sig,
                    stage_nodes=snodes,
                    target=node,
                    template_index=-1,
                )
            )
        for i, name in enumerate(node.output):
            for consumer in in_to_nodes.get(name, []):
                down = downstream(consumer)
                if down is None:
                    continue
                kind, sig, snodes, tgt = down
                stages.append(
                    dict(
                        node_name=node_name,
                        tensor_type="output",
                        tensor_index=i,
                        tensor_name=name,
                        override=override_name(name, tgt),
                        calib=tgt,
                        kind=kind,
                        sig=sig,
                        stage_nodes=snodes,
                        target=consumer,
                        template_index=-1,
                    )
                )

    def borrowed(own: List[Stage]) -> Optional[Stage]:
        """A template from a neighbour that shares the exact same tensor."""
        for st in own:
            for others in infos.values():
                for info in others:
                    if info["override"] == st["override"] and (
                        info["tensor_name"] == st["override"]
                        or info["override"] == st["tensor_name"]
                    ):
                        if info["template_index"] < 0:
                            continue
                        return others[info["template_index"]]
        return None

    # phase 2: a template per node, a copy of it in front of every other stage
    done: Set[Tuple[str, str, str]] = set()
    inserted = 0
    for stages in infos.values():
        if not stages:
            continue
        node_name = stages[0]["node_name"]
        tmpl_index = -1
        for i, info in enumerate(stages):
            if mixed and node_name not in mixed:
                if not info["override"]:
                    tmpl_index = i
                    break
            elif info["override"]:
                tmpl_index = i
                break
        tmpl = stages[tmpl_index] if tmpl_index >= 0 else borrowed(stages)
        if tmpl is None:
            continue
        for info in stages:
            if info is tmpl or info["sig"] == tmpl["sig"]:
                continue
            info["template_index"] = tmpl_index
            primary = info["override"] or info["tensor_name"]
            qparam = primary
            if not (ranges is None or ranges(primary) is not None or not info["calib"]):
                qparam = info["calib"]
            target_name = info["target"].name or info["target"].op_type
            key = (node_name, info["tensor_name"], target_name)
            if key in done:
                continue
            insert_between(
                info["tensor_name"],
                info["target"],
                tmpl["kind"],
                tmpl["stage_nodes"],
                qparam,
                f"{qparam}_{node_name}_{info['tensor_index']}_{target_name}",
            )
            done.add(key)
            inserted += 1

    if not inserted:
        m.graph.node.extend(nodes)
        return m
    known = set(inits) | {v.name for v in m.graph.input}
    ordered = _toposort(nodes, known)
    del m.graph.node[:]
    m.graph.node.extend(ordered)
    return m


__all__ = ["insert_boundary_quant_nodes"]
