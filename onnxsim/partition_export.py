# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Export a device partition plan as a model with one ONNX function per region.

:func:`onnxsim.graph_partition.partition_graph` decides which device runs each
node. This module writes that decision out as a model: every *convex*
same-device region becomes one call to a model-local ``FunctionProto`` in the
``onnxsim.partition`` domain, and the function body holds the region's original
nodes. A runtime or a reader then sees one node per device region, with the
region's boundary tensors as the call's inputs and outputs.

Rules:

* Only convex regions are outlined. A non-convex region would turn into a call
  that both feeds and consumes a node outside it, so it stays inline.
* Regions with fewer than ``min_nodes`` nodes stay inline, and so do regions
  containing a subgraph attribute (``If``, ``Loop``, ``Scan``), whose bodies
  reference outer-scope names that a function cannot capture.
* Collapsing regions can create a cycle between two regions that are each
  convex on their own. The lowest-ordered region is kept when this happens;
  the others are dropped until the outer graph is acyclic again.
* Tensor names are unchanged, so graph inputs, outputs and initializers keep
  their names. Each function's formal inputs and outputs use the outer names.

The output model is a plain ONNX model with functions. It is not the
device-specific runtime artifact; it records the partition structure so that
the plan can be inspected, verified (run with the reference evaluator), or
lowered by a backend that supports model-local functions.
"""

from __future__ import annotations

import copy
import dataclasses
import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import onnx
from onnx import helper

from onnxsim.graph_partition import CPU, Partition, partition_graph

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


def _opset_for_domain(model: onnx.ModelProto, domain: str) -> Optional[int]:
    wanted = "" if domain in ("", "ai.onnx") else domain
    for opset in model.opset_import:
        norm = "" if opset.domain in ("", "ai.onnx") else opset.domain
        if norm == wanted:
            return opset.version
    return None


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
        src = {producer[name] for name in ins if name in producer and producer[name] != idx}
        deps.append(src)
        for s in src:
            dependents[s].append(idx)

    remaining = [len(d) for d in deps]
    heap = [(entries[idx][0], idx) for idx in range(len(entries)) if remaining[idx] == 0]
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


def _build_function(
    model: onnx.ModelProto,
    region: _Region,
    nodes: Sequence[onnx.NodeProto],
) -> onnx.FunctionProto:
    body = [copy.deepcopy(nodes[i]) for i in region.partition.node_indices]
    domains = {node.domain for node in body}
    opset_imports = []
    for domain in sorted(domains):
        version = _opset_for_domain(model, domain)
        if version is None:
            raise ValueError(
                f"model has no opset import for domain {domain!r} used by the region"
            )
        norm_domain = "" if domain == "ai.onnx" else domain
        opset_imports.append(helper.make_opsetid(norm_domain, version))
    return helper.make_function(
        EXPORT_DOMAIN,
        region.name,
        region.inputs,
        region.outputs,
        body,
        opset_imports,
    )


def _assemble(
    regions: Sequence[_Region],
    nodes: Sequence[onnx.NodeProto],
) -> Optional[List[onnx.NodeProto]]:
    """Outer node list with ``regions`` collapsed to calls, or ``None`` on a cycle."""
    in_region: Dict[int, _Region] = {}
    for region in regions:
        for i in region.partition.node_indices:
            in_region[i] = region

    entries: List[Tuple[int, List[str], List[str], object]] = []
    for i, node in enumerate(nodes):
        region = in_region.get(i)
        if region is None:
            entries.append((i, list(node.input), list(node.output), node))
        elif i == region.partition.node_indices[0]:
            call = helper.make_node(
                region.name,
                region.inputs,
                region.outputs,
                name=f"{region.name}_call",
                domain=EXPORT_DOMAIN,
                doc_string=f"device={region.partition.device}",
            )
            entries.append((i, list(region.inputs), list(region.outputs), call))
    return _order(entries)  # type: ignore[return-value]


def export_partitioned_model(
    model: Union[str, onnx.ModelProto],
    primary: str,
    fallback: Sequence[str] = (CPU,),
    *,
    min_nodes: int = 2,
) -> PartitionExport:
    """Place ``model``'s nodes as :func:`partition_graph` does and export each
    convex device region as one model-local function call.

    :param model: the ONNX model, or a file path. Run it on a simplified model.
    :param primary: the preferred device, one of ``"webnn"``, ``"webgpu"``.
    :param fallback: devices tried in order after ``primary``.
    :param min_nodes: smallest region to outline; smaller regions stay inline.
    :returns: a :class:`PartitionExport`. The input model is not modified.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    plan = partition_graph(model, primary, fallback)
    graph = model.graph
    nodes = list(graph.node)
    graph_outputs = {o.name for o in graph.output}

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

    # Drop the highest-ordered region until the collapsed graph is acyclic.
    while True:
        for k, region in enumerate(candidates):
            region.name = f"{region.partition.device}_partition_{k}"
        if _assemble(candidates, nodes) is not None:
            break
        dropped = candidates.pop()
        skipped.append(
            (dropped.partition.device, tuple(dropped.partition.node_indices), "would create a cycle")
        )

    out = copy.deepcopy(model)
    if not candidates:
        return PartitionExport(out, (), len(nodes), tuple(skipped))

    new_nodes = _assemble(candidates, nodes)
    assert new_nodes is not None
    functions = [_build_function(model, r, nodes) for r in candidates]

    in_region = {i for r in candidates for i in r.partition.node_indices}
    internal = {
        name
        for i in in_region
        for name in nodes[i].output
        if name
    } - {name for r in candidates for name in r.outputs}

    del out.graph.node[:]
    out.graph.node.extend(new_nodes)
    kept_vi = [v for v in graph.value_info if v.name not in internal]
    del out.graph.value_info[:]
    out.graph.value_info.extend(kept_vi)
    out.functions.extend(functions)
    if not any(o.domain == EXPORT_DOMAIN for o in out.opset_import):
        out.opset_import.append(helper.make_opsetid(EXPORT_DOMAIN, 1))
    out.ir_version = max(out.ir_version, _MIN_IR_VERSION_FOR_FUNCTIONS)

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
        model=out,
        functions=outlined,
        inline_nodes=len(new_nodes) - len(candidates),
        skipped=tuple(skipped),
    )
