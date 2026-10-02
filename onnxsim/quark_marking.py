"""Which nodes Quark's QDQ quantizer leaves alone, and in which order it visits
them, for :mod:`onnxsim.quark_compat`.

Quark's static quantizer (a ``QDQOperatorBase`` per node, in the node order of
the topologically sorted float graph) *marks* tensors for a Q/DQ pair. Three
things in that pass are visible in the quantized graph and were not reproduced
by looking at a node's neighbours alone:

- **the op-type list**: only the op types of Quark's op registries are quantized
  (:data:`QUARK_QDQ_OP_TYPES`, plus :data:`QUARK_NPU_CNN_OP_TYPES` for the NPU /
  extended quantizers). A ``Flatten`` -- or an ``Exp`` or a ``Neg`` -- marks
  nothing; only a consumer that is in the list gives its input (and so the
  output of the Flatten) a Q/DQ pair, and the Flatten's output is calibrated on
  its own instead of sharing its input's parameters;
- **order-dependent conditions**: a ``Relu`` / ``Clip`` is quantized only if its
  input has been marked by a node visited *before* it
  (``QDQRemovableActivation``), so a ``Relu`` / ``Clip`` fed straight by a graph
  input stays a plain float node whose *output* is quantized for its consumer
  (unless a ``Conv`` that reads the same input came first); the data-movement
  ops (``Reshape``, ``Transpose``, ...) behave alike without
  ``ForceQuantizeNoInputCheck``; a ``HardSigmoid`` that does not meet the DPU
  condition is never quantized;
- **the visiting order itself**: ``ONNXModel.topological_sort`` -- not a
  textbook Kahn sort: nodes without inputs first, then the nodes released by the
  graph inputs and initializers *in alphabetical order of their names*, then a
  breadth-first sweep over the outputs.

:func:`quark_node_order` reproduces the sort exactly, :func:`skipped_nodes` runs
the marking pass for the nodes that decide nothing by themselves.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import onnx

# the union of ``QLinearOpsRegistry`` and ``QDQRegistry`` (quark.onnx.quantizers.registry)
QUARK_QDQ_OP_TYPES = frozenset(
    {
        "Add",
        "ArgMax",
        "AveragePool",
        "Clip",
        "Concat",
        "Conv",
        "ConvTranspose",
        "EmbedLayerNormalization",
        "Gather",
        "Gelu",
        "Gemm",
        "GlobalAveragePool",
        "InstanceNormalization",
        "LayerNormalization",
        "LeakyRelu",
        "MatMul",
        "MaxPool",
        "Mul",
        "Pad",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
        "Softmax",
        "Split",
        "Squeeze",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)
#: ``NPUCnnRegistry`` -- added for the NPU CNN scheme and the extended quantizer
QUARK_NPU_CNN_OP_TYPES = frozenset(
    {
        "DepthToSpace",
        "Div",
        "Erf",
        "HardSigmoid",
        "LpNormalization",
        "Max",
        "Min",
        "PRelu",
        "ReduceMean",
        "Slice",
        "SpaceToDepth",
        "Sub",
    }
)
_DIRECT_OPS = ("Reshape", "Transpose", "Squeeze", "Unsqueeze")


def quark_op_types(npu_cnn_ops: bool, extra: Iterable[str] = ()) -> "frozenset[str]":
    """Quark's default ``op_types_to_quantize`` (``get_static_op_types``)."""
    types = set(QUARK_QDQ_OP_TYPES)
    if npu_cnn_ops:
        types |= QUARK_NPU_CNN_OP_TYPES
    types |= set(extra)
    return frozenset(types)


def _order_indices(model: onnx.ModelProto) -> List[int]:
    nodes = list(model.graph.node)
    deps_count = [0] * len(nodes)
    deps_to_nodes: Dict[str, List[int]] = {}
    ordered: List[int] = []
    for idx, node in enumerate(nodes):
        deps_count[idx] = sum(1 for x in node.input if x)
        if deps_count[idx] == 0:
            ordered.append(idx)
            continue
        for name in node.input:
            if name:
                deps_to_nodes.setdefault(name, []).append(idx)
    names = sorted(
        {t.name for t in model.graph.initializer} | {i.name for i in model.graph.input}
    )
    for name in names:
        for idx in deps_to_nodes.get(name, ()):
            deps_count[idx] -= 1
            if deps_count[idx] == 0:
                ordered.append(idx)
    start = 0
    while start < len(ordered):
        for out in nodes[ordered[start]].output:
            for idx in deps_to_nodes.get(out, ()):
                deps_count[idx] -= 1
                if deps_count[idx] == 0:
                    ordered.append(idx)
        start += 1
    if len(ordered) != len(nodes):
        raise ValueError("Graph is not a DAG")
    return ordered


def quark_node_order(model: onnx.ModelProto) -> List[onnx.NodeProto]:
    """The nodes of ``model`` in the order of Quark's (ONNX Runtime's)
    ``ONNXModel.topological_sort``. Raises ``ValueError`` on a cycle."""
    nodes = list(model.graph.node)
    return [nodes[i] for i in _order_indices(model)]


def quark_sort_inplace(model: onnx.ModelProto) -> None:
    """Reorder the nodes of ``model`` in place as :func:`quark_node_order` says
    (a cycle leaves it as it is)."""
    try:
        order = _order_indices(model)
    except ValueError:  # pragma: no cover - not a DAG
        return
    nodes = [onnx.NodeProto() for _ in order]
    for dst, i in zip(nodes, order):
        dst.CopyFrom(model.graph.node[i])
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def quark_sorted(model: onnx.ModelProto) -> onnx.ModelProto:
    """A copy of ``model`` with its nodes in :func:`quark_node_order`."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    try:
        order = quark_node_order(out)
    except ValueError:  # pragma: no cover - not a DAG
        return out
    nodes = [onnx.NodeProto() for _ in order]
    for dst, src in zip(nodes, order):
        dst.CopyFrom(src)
    del out.graph.node[:]
    out.graph.node.extend(nodes)
    return out


def _quark_names(model: onnx.ModelProto) -> Dict[str, str]:
    """``{initializer name: the name Quark gives it}`` for the scale, zero-point
    and quantized-data initializers of the Q/DQ pairs of a quantized ``model``
    built by :func:`onnxsim.full_qdq.quantize_full_qdq` (Quark names them after
    the tensor: ``<t>_scale`` / ``<t>_zero_point``, and ``<w>_quantized`` /
    ``<w>_scale`` / ``<w>_zero_point`` for a constant; an int32 bias's are
    ``<b>_quantized_scale`` / ``<b>_quantized_zero_point``)."""
    inits = {t.name for t in model.graph.initializer}
    rename: Dict[str, str] = {}

    def tensor_of(name: str) -> str:
        return name.split("/qdq")[0]

    for n in model.graph.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(n.input) < 3:
            continue
        data, scale, zp = n.input[0], n.input[1], n.input[2]
        if data in inits:  # a constant: DQ of the stored integer codes
            base = tensor_of(data)
            int32 = data.endswith("/int32")
            rename.setdefault(data, base + "_quantized")
            rename.setdefault(scale, base + ("_quantized_scale" if int32 else "_scale"))
            rename.setdefault(
                zp, base + ("_quantized_zero_point" if int32 else "_zero_point")
            )
        elif n.op_type == "QuantizeLinear":
            base = data[:-2] if data.endswith("/f") else data
            rename.setdefault(scale, base + "_scale")
            rename.setdefault(zp, base + "_zero_point")
    return rename


def quark_qdq_sorted(model: onnx.ModelProto) -> onnx.ModelProto:
    """A copy of a quantized ``model`` with its nodes in the order Quark's own
    ``topological_sort`` gives the equivalent graph: the sort seeds on the
    alphabetical order of the initializer and input names, so it is run on a
    copy whose Q/DQ parameters carry Quark's names."""
    tmp = onnx.ModelProto()
    tmp.CopyFrom(model)
    rename = _quark_names(tmp)
    for t in tmp.graph.initializer:
        t.name = rename.get(t.name, t.name)
    for n in tmp.graph.node:
        for k, x in enumerate(n.input):
            n.input[k] = rename.get(x, x)
    try:
        order = _order_indices(tmp)
    except ValueError:  # pragma: no cover - not a DAG
        return model
    out = onnx.ModelProto()
    out.CopyFrom(model)
    nodes = [onnx.NodeProto() for _ in order]
    for dst, i in zip(nodes, order):
        dst.CopyFrom(model.graph.node[i])
    del out.graph.node[:]
    out.graph.node.extend(nodes)
    return out


def _hard_sigmoid_ok(node: onnx.NodeProto) -> bool:
    attrs = {a.name: a.f for a in node.attribute}
    alpha = "alpha" in attrs and abs(attrs["alpha"] - 1.0 / 6.0) <= 1e-6
    beta = "beta" not in attrs or abs(attrs["beta"] - 0.5) <= 1e-6
    return bool(alpha and beta)


def skipped_nodes(
    model: onnx.ModelProto,
    op_types: "Optional[Iterable[str]]",
    excluded: Iterable[str] = (),
    force_no_input_check: bool = True,
    direct_pool: bool = False,
    order: Optional[Sequence[onnx.NodeProto]] = None,
    npu_registry: bool = True,
) -> Set[str]:
    """Names (first outputs, for unnamed nodes) of the nodes in ``op_types`` whose
    Quark op quantizer marks nothing, visiting the nodes in ``order`` (default:
    :func:`quark_node_order`):

    - a ``Relu`` / ``Clip`` whose input no earlier node marked;
    - a ``Reshape`` / ``Transpose`` / ``Squeeze`` / ``Unsqueeze`` / ``Resize`` /
      ``MaxPool`` / ``LayerNormalization`` (and ``AveragePool`` with ``direct_pool``, ONNX Runtime's plain
      scheme) whose input is unmarked, unless ``force_no_input_check``; likewise a
      ``Gather`` and a ``Where``;
    - a ``HardSigmoid`` that is not ``alpha = 1/6``, ``beta = 0.5`` (only with
      ``npu_registry``: Quark's ``QDQHardSigmoid`` belongs to the NPU CNN registry,
      the plain quantizer marks a HardSigmoid like any other op).

    Everything else marks its inputs and outputs."""
    types = None if op_types is None else set(op_types)
    excl = set(excluded)
    inits = {t.name for t in model.graph.initializer}
    nodes = list(order) if order is not None else quark_node_order(model)
    direct = set(_DIRECT_OPS) | {"Resize", "MaxPool", "LayerNormalization"}
    if direct_pool:
        direct.add("AveragePool")
    marked: Set[str] = set()
    skipped: Set[str] = set()

    def key(n: onnx.NodeProto) -> str:
        return n.name or (n.output[0] if n.output else "")

    for n in nodes:
        if n.domain not in ("", "ai.onnx") or n.op_type in (
            "QuantizeLinear",
            "DequantizeLinear",
        ):
            continue
        if types is not None and n.op_type not in types:
            continue
        if n.name in excl or (n.output and n.output[0] in excl):
            continue
        ins = [x for x in n.input if x]
        outs = [x for x in n.output if x]
        op = n.op_type
        if op in ("Relu", "Clip"):
            if not ins or ins[0] not in marked:
                skipped.add(key(n))
                continue
            marked.update(ins[:1] + outs)
        elif op in direct:
            if force_no_input_check:
                marked.update(ins[:1] + outs)
            elif ins and ins[0] in marked:
                marked.update(outs)
            else:
                skipped.add(key(n))
        elif op == "Gather":
            if (ins and ins[0] in inits) or force_no_input_check:
                marked.update(ins[:1] + outs[:1])
            elif ins and ins[0] in marked:
                marked.update(outs[:1])
            else:
                skipped.add(key(n))
        elif op == "Where":
            if force_no_input_check:
                marked.update(ins[1:3] + outs)
            elif len(ins) > 2 and ins[1] in marked and ins[2] in marked:
                marked.update(outs)
            else:
                skipped.add(key(n))
        elif op == "HardSigmoid" and npu_registry:
            if _hard_sigmoid_ok(n):
                marked.update(ins[:1] + outs[:1])
            else:
                skipped.add(key(n))
        elif op == "Split":
            marked.update(ins[:1] + outs)
        else:
            marked.update(ins + outs)
    return skipped


__all__: Any = [
    "QUARK_NPU_CNN_OP_TYPES",
    "QUARK_QDQ_OP_TYPES",
    "quark_node_order",
    "quark_op_types",
    "quark_qdq_sorted",
    "quark_sort_inplace",
    "quark_sorted",
    "skipped_nodes",
]
