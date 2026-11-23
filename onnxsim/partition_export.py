# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Export a device partition plan as a model with one ONNX function per region.

:func:`onnxsim.graph_partition.partition_graph` decides which device runs each
node. This module writes that decision out as a model: every *convex*
same-device region becomes one call to a model-local function in the
``onnxsim.partition`` domain, and the function body holds the region's original
nodes. A runtime or a reader then sees one node per device region, with the
region's boundary tensors as the call's inputs and outputs.

The planning step reads the protobuf model. The graph editing (building the
function bodies, replacing region nodes with calls, rewiring boundary tensors)
runs on :mod:`onnx_ir` objects, converted at entry and exit by
:mod:`onnxsim.ir_bridge`.

Rules:

* Only convex regions are outlined. A non-convex region would turn into a call
  that both feeds and consumes a node outside it, so it stays inline.
* Regions with fewer than ``min_nodes`` nodes stay inline, and so do regions
  containing a subgraph attribute (``If``, ``Loop``, ``Scan``), whose bodies
  reference outer-scope names that a function cannot capture.
* Collapsing regions can create a cycle between two regions that are each
  convex on their own. The highest-ordered region is dropped when this happens,
  until the outer graph is acyclic again.
* Tensor names are unchanged at the region boundaries, so graph inputs, outputs
  and initializers keep their names.

The output is a plain ONNX model with functions. It records the partition
structure so the plan can be inspected or verified (run with the reference
evaluator), or lowered by a backend that supports model-local functions.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import onnx
import onnx_ir as ir

from onnxsim.graph_partition import CPU, Partition, partition_graph
from onnxsim.ir_bridge import as_ir, as_proto

__all__ = [
    "EXPORT_DOMAIN",
    "PartitionExport",
    "PartitionFunction",
    "export_partitioned_model",
]

EXPORT_DOMAIN = "onnxsim.partition"
_MIN_IR_VERSION_FOR_FUNCTIONS = 8


@dataclasses.dataclass(frozen=True)
class PartitionFunction:
    """One outlined region.

    :param name: the function name (also the call node's op_type).
    :param device: the device the region runs on.
    :param node_count: how many original nodes the function body holds.
    :param node_indices: indices of those nodes in the input model's graph.
    """

    name: str
    device: str
    node_count: int
    node_indices: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PartitionExport:
    """Result of :func:`export_partitioned_model`.

    :param model: the exported model, with one call per outlined region.
    :param functions: the outlined regions, in graph order.
    :param inline_nodes: nodes left directly in the outer graph.
    :param skipped: regions that were not outlined, as
            ``(device, node_indices, reason)`` tuples.
    """

    model: onnx.ModelProto
    functions: Tuple[PartitionFunction, ...]
    inline_nodes: int
    skipped: Tuple[Tuple[str, Tuple[int, ...], str], ...]


@dataclasses.dataclass
class _Region:
    partition: Partition
    inputs: List[str]
    outputs: List[str]
    name: str


def _has_subgraph(node: onnx.NodeProto) -> bool:
    return any(a.HasField("g") or len(a.graphs) for a in node.attribute)


def _region_io(
    nodes: Sequence[onnx.NodeProto],
    members: Sequence[int],
    readers: Dict[str, Set[int]],
    graph_outputs: Set[str],
) -> Tuple[List[str], List[str]]:
    """External inputs and exported outputs of a region, in first-use order."""
    member_set = set(members)
    produced: Set[str] = set()
    for i in members:
        produced.update(name for name in nodes[i].output if name)

    inputs: List[str] = []
    seen: Set[str] = set()
    for i in members:
        for name in nodes[i].input:
            if name and name not in produced and name not in seen:
                seen.add(name)
                inputs.append(name)

    outputs: List[str] = []
    for i in members:
        for name in nodes[i].output:
            if not name or name in outputs:
                continue
            used_outside = any(r not in member_set for r in readers.get(name, ()))
            if used_outside or name in graph_outputs:
                outputs.append(name)
    return inputs, outputs


def _order(
    entries: List[Tuple[int, List[str], List[str], object]],
) -> Optional[List[object]]:
    """Topologically sort ``(key, inputs, outputs, payload)`` entries, keeping
    the original order where possible. Returns ``None`` on a cycle."""
    producer: Dict[str, int] = {}
    for idx, (_, _, outs, _) in enumerate(entries):
        for name in outs:
            if name:
                producer[name] = idx

    deps: List[Set[int]] = []
    dependents: Dict[int, List[int]] = {idx: [] for idx in range(len(entries))}
    for idx, (_, ins, _, _) in enumerate(entries):
        src = {
            producer[name] for name in ins if name in producer and producer[name] != idx
        }
        deps.append(src)
        for s in src:
            dependents[s].append(idx)

    remaining = [len(d) for d in deps]
    heap = [
        (entries[idx][0], idx) for idx in range(len(entries)) if remaining[idx] == 0
    ]
    heapq.heapify(heap)
    out: List[object] = []
    while heap:
        _, idx = heapq.heappop(heap)
        out.append(entries[idx][3])
        for nxt in dependents[idx]:
            remaining[nxt] -= 1
            if remaining[nxt] == 0:
                heapq.heappush(heap, (entries[nxt][0], nxt))
    if len(out) != len(entries):
        return None
    return out


def _outer_values(graph: ir.Graph) -> Dict[str, ir.Value]:
    """Every named value in the outer graph, by name."""
    values: Dict[str, ir.Value] = {}
    for v in graph.inputs:
        if v.name:
            values[v.name] = v
    for name, v in graph.initializers.items():
        values[name] = v
    for node in graph:
        for v in node.outputs:
            if v.name:
                values[v.name] = v
    return values


def _function_opsets(model: ir.Model, nodes: Sequence[ir.Node]) -> Dict[str, int]:
    """Opset versions for the domains the function body uses."""
    out: Dict[str, int] = {}
    for node in nodes:
        domain = "" if node.domain in ("", "ai.onnx") else node.domain
        if domain in out:
            continue
        version = model.opset_imports.get(domain)
        if version is None:
            raise ValueError(
                f"model has no opset import for domain {domain!r} used by the region"
            )
        out[domain] = version
    return out


def _build_body(
    region: _Region,
    ir_nodes: Sequence[ir.Node],
    outer: Dict[str, ir.Value],
    model: ir.Model,
) -> ir.Function:
    """The model-local function for ``region``, with a body copied from its nodes."""
    fresh: Dict[str, ir.Value] = {}
    formal_inputs: List[ir.Value] = []
    for name in region.inputs:
        src = outer[name]
        value = ir.Value(name=name, type=src.type, shape=src.shape)
        fresh[name] = value
        formal_inputs.append(value)

    body: List[ir.Node] = []
    for i in region.partition.node_indices:
        node = ir_nodes[i]
        inputs = [
            fresh[v.name] if v is not None and v.name else None for v in node.inputs
        ]
        outputs: List[ir.Value] = []
        for o in node.outputs:
            if o.name:
                value = ir.Value(name=o.name, type=o.type, shape=o.shape)
                fresh[o.name] = value
            else:
                value = ir.Value()
            outputs.append(value)
        body.append(
            ir.Node(
                node.domain,
                node.op_type,
                inputs,
                attributes=list(node.attributes.values()),
                outputs=outputs,
                overload=node.overload,
                name=node.name,
                doc_string=node.doc_string,
            )
        )

    graph = ir.Graph(
        inputs=formal_inputs,
        outputs=[fresh[name] for name in region.outputs],
        nodes=body,
        opset_imports=_function_opsets(model, body),
        name=region.name,
    )
    return ir.Function(EXPORT_DOMAIN, region.name, graph=graph, attributes=())


def export_partitioned_model(
    model: Union[str, onnx.ModelProto, ir.Model],
    primary: str,
    fallback: Sequence[str] = (CPU,),
    *,
    min_nodes: int = 2,
) -> PartitionExport:
    """Place ``model``'s nodes as :func:`partition_graph` does and export each
    convex device region as one model-local function call.

    :param model: the ONNX model, a file path, or an ``onnx_ir`` model. Run it on
            a simplified model.
    :param primary: the preferred device, one of ``"webnn"``, ``"webgpu"``.
    :param fallback: devices tried in order after ``primary``.
    :param min_nodes: smallest region to outline; smaller regions stay inline.
    :returns: a :class:`PartitionExport`. The input model is not modified.
    """
    if isinstance(model, ir.Model):
        model_proto = as_proto(model)
    elif isinstance(model, str):
        model_proto = onnx.load(model, load_external_data=False)
    else:
        model_proto = model

    plan = partition_graph(model_proto, primary, fallback)
    nodes = list(model_proto.graph.node)
    graph_outputs = {o.name for o in model_proto.graph.output}

    readers: Dict[str, Set[int]] = {}
    for i, node in enumerate(nodes):
        for name in node.input:
            if name:
                readers.setdefault(name, set()).add(i)

    skipped: List[Tuple[str, Tuple[int, ...], str]] = []
    candidates: List[_Region] = []
    for part in plan.partitions:
        members = part.node_indices
        if not part.convex:
            reason = "non-convex region"
        elif len(members) < min_nodes:
            reason = f"fewer than {min_nodes} nodes"
        elif any(_has_subgraph(nodes[i]) for i in members):
            reason = "contains a subgraph attribute"
        else:
            inputs, outputs = _region_io(nodes, members, readers, graph_outputs)
            if not outputs:
                reason = "region has no outputs"
            else:
                candidates.append(_Region(part, inputs, outputs, name=""))
                continue
        skipped.append((part.device, tuple(members), reason))

    ir_model = as_ir(model_proto)
    ir_nodes = list(ir_model.graph)
    if len(ir_nodes) != len(nodes):
        raise RuntimeError("onnx_ir changed the node count during conversion")

    # Drop the highest-ordered region until the collapsed graph is acyclic.
    while True:
        for k, region in enumerate(candidates):
            region.name = f"{region.partition.device}_partition_{k}"
        if _collapsed_order(candidates, nodes) is not None:
            break
        dropped = candidates.pop()
        skipped.append(
            (
                dropped.partition.device,
                tuple(dropped.partition.node_indices),
                "would create a cycle",
            )
        )

    if not candidates:
        return PartitionExport(as_proto(ir_model), (), len(nodes), tuple(skipped))

    graph = ir_model.graph
    outer = _outer_values(graph)
    functions: List[ir.Function] = []
    calls: Dict[str, ir.Node] = {}

    # 1. Replace each exported output with a fresh value, so the region's
    #    consumers now read the call's output. Values are replaced in order, so
    #    later regions see the already-replaced inputs.
    for region in candidates:
        for name in region.outputs:
            old = outer[name]
            new = ir.Value(name=name, type=old.type, shape=old.shape)
            old.replace_all_uses_with(new, replace_graph_outputs=True)
            outer[name] = new

    # 2. Build each function body from the original nodes, and the call node
    #    that replaces the region in the outer graph.
    for region in candidates:
        function = _build_body(region, ir_nodes, outer, ir_model)
        functions.append(function)
        call = ir.Node(
            EXPORT_DOMAIN,
            region.name,
            [outer[name] for name in region.inputs],
            outputs=[outer[name] for name in region.outputs],
            name=f"{region.name}_call",
            doc_string=f"device={region.partition.device}",
        )
        calls[region.name] = call

    # 3. Rebuild the outer graph: region members out, calls in, in dependency
    #    order.
    member_of: Dict[int, _Region] = {}
    for region in candidates:
        for i in region.partition.node_indices:
            member_of[i] = region
    entries: List[Tuple[int, List[str], List[str], object]] = []
    for i, node in enumerate(ir_nodes):
        owner = member_of.get(i)
        if owner is None:
            entries.append(
                (
                    i,
                    [v.name for v in node.inputs if v is not None and v.name],
                    [v.name for v in node.outputs if v.name],
                    node,
                )
            )
        elif i == owner.partition.node_indices[0]:
            entries.append(
                (i, list(owner.inputs), list(owner.outputs), calls[owner.name])
            )
    ordered = _order(entries)
    if ordered is None:
        raise RuntimeError("collapsed graph is cyclic after IR rewrite")

    graph.remove(list(graph), safe=False)
    graph.extend(ordered)  # type: ignore[arg-type]

    for function in functions:
        ir_model.functions[function.identifier()] = function
    if EXPORT_DOMAIN not in ir_model.opset_imports:
        ir_model.opset_imports[EXPORT_DOMAIN] = 1
    ir_model.ir_version = max(ir_model.ir_version, _MIN_IR_VERSION_FOR_FUNCTIONS)

    outlined = tuple(
        PartitionFunction(
            name=r.name,
            device=r.partition.device,
            node_count=len(r.partition.node_indices),
            node_indices=tuple(r.partition.node_indices),
        )
        for r in candidates
    )
    return PartitionExport(
        model=as_proto(ir_model),
        functions=outlined,
        inline_nodes=len(ordered) - len(candidates),
        skipped=tuple(skipped),
    )


def _collapsed_order(
    candidates: Sequence[_Region], nodes: Sequence[onnx.NodeProto]
) -> Optional[List[object]]:
    """Dependency order of the outer graph with ``candidates`` collapsed, or
    ``None`` on a cycle. Used to choose which regions to keep."""
    in_region: Dict[int, _Region] = {}
    for r in candidates:
        for i in r.partition.node_indices:
            in_region[i] = r
    entries: List[Tuple[int, List[str], List[str], object]] = []
    for i, node in enumerate(nodes):
        owner = in_region.get(i)
        if owner is None:
            entries.append((i, list(node.input), list(node.output), node))
        elif i == owner.partition.node_indices[0]:
            entries.append((i, list(owner.inputs), list(owner.outputs), owner))
    return _order(entries)
