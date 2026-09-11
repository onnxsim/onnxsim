"""Shared graph-connectivity helper: given a set of node indices assumed to
fall back off a target execution provider (EP), estimate how many separate
"islands" that splits the rest of the graph's nodes into.

This matters because of how ONNX Runtime actually partitions a mixed-EP
graph (see the ``providers`` docs and ``GetCapability``): it's a one-time,
priority-ordered, per-node capability check at session creation, not a
runtime scheduler -- each EP claims the nodes it supports, remaining nodes
cascade to the next EP in priority order, and wherever two *adjacent* nodes
end up on different EPs, ORT inserts a device-to-device ``Memcpy`` node at
that boundary. A graph that alternates between "supported" and "fallback"
nodes doesn't just lose acceleration on the fallback nodes -- it also pays a
copy at every boundary and ends up as many small islands instead of one
large one, which is usually far slower than either EP running the whole
thing.

This module estimates that fragmentation from a caller-supplied set of
flagged (fallback) node indices: it does **not** know the target EP's full
operator/dtype support table itself (``onnxsim.webgpu_target`` /
``onnxsim.webnn_target`` each supply only the specific, currently-documented
gaps they check for) -- so the island count/boundary count returned here is
a **lower bound** on real fragmentation, not a full simulation of ORT's
partitioner. Any other op in the graph that isn't in the target EP's
operator table, but that this caller doesn't happen to check for, will
fragment the real session further without being reflected here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import onnx


@dataclass
class IslandReport:
    """:param island_count: number of connected groups the *non-flagged*
            nodes split into once every flagged node (and edges touching it)
            is cut out. ``0`` means there are no non-flagged compute nodes at
            all; ``1`` means the flagged nodes (if any) didn't fragment
            anything -- everything else stays one contiguous island.
    :param boundary_edge_count: number of producer/consumer edges directly
            connecting a flagged node to a non-flagged one -- each is a point
            ONNX Runtime would insert a ``Memcpy`` if the two sides land on
            different execution providers, so this approximates the number
            of extra device copies the flagged nodes introduce.
    :param flagged_node_names: the flagged nodes, for reference (name, or the
            first output name when unnamed).
    """

    island_count: int
    boundary_edge_count: int
    flagged_node_names: List[str] = field(default_factory=list)


def estimate_fragmentation(
    graph: onnx.GraphProto, flagged_node_indices: Set[int]
) -> IslandReport:
    """See this module's docstring for what "flagged" and "island" mean
    here. ``flagged_node_indices`` are indices into ``graph.node``.
    """
    nodes = list(graph.node)
    n = len(nodes)

    producers: Dict[str, List[int]] = {}
    consumers: Dict[str, List[int]] = {}
    for i, node in enumerate(nodes):
        for out in node.output:
            if out:
                producers.setdefault(out, []).append(i)
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(i)

    # Undirected adjacency between non-flagged node indices sharing a value
    # (Memcpy insertion doesn't care about direction); edges touching a
    # flagged node are counted as boundaries instead of adjacency.
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
    boundary_edge_count = 0
    for name in set(producers) | set(consumers):
        for p in producers.get(name, []):
            for c in consumers.get(name, []):
                if p == c:
                    continue
                p_flagged = p in flagged_node_indices
                c_flagged = c in flagged_node_indices
                if p_flagged != c_flagged:
                    boundary_edge_count += 1
                elif not p_flagged:  # neither flagged
                    adjacency[p].add(c)
                    adjacency[c].add(p)

    visited: Set[int] = set()
    island_count = 0
    for i in range(n):
        if i in flagged_node_indices or i in visited:
            continue
        island_count += 1
        stack = [i]
        visited.add(i)
        while stack:
            cur = stack.pop()
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

    flagged_node_names = [
        nodes[i].name or (nodes[i].output[0] if nodes[i].output else f"<node {i}>")
        for i in sorted(flagged_node_indices)
    ]
    return IslandReport(
        island_count=island_count,
        boundary_edge_count=boundary_edge_count,
        flagged_node_names=flagged_node_names,
    )
