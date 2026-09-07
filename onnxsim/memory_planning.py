import copy
import dataclasses
from typing import Dict, List, Tuple

import onnx

from onnxsim._rich_compat import Table, Text, print
from onnxsim.model_info import (
    METADATA_PREFIX,
    ModelInfo,
    _set_metadata,
    human_readable_size,
)

__all__ = [
    "MemoryPlan",
    "plan_activation_memory",
    "print_memory_plan",
    "annotate_memory_plan",
]


@dataclasses.dataclass(frozen=True)
class MemoryPlan:
    """A static byte-offset allocation plan for a model's activation tensors,
    computed by :func:`plan_activation_memory`.

    Every *planned* tensor (a graph input, node output, or graph output with a
    concretely known size) gets a ``(offset, size)`` pair in
    :attr:`tensor_offsets`, all within one shared arena of
    :attr:`arena_bytes`: two tensors only ever get overlapping offset ranges
    when their liveness intervals do not overlap, so a deployment target can
    allocate a single :attr:`arena_bytes`-sized buffer and hand each tensor
    its own slice, reused over the tensor's lifetime, instead of one
    permanent buffer per activation (:attr:`naive_bytes` -- the baseline
    :attr:`compression_ratio` is measured against).

    Weights (initializers) are never part of the plan: they stay resident for
    the whole graph, outside this arena. A tensor whose size cannot be
    resolved to a concrete byte count -- unknown shape/dtype, or a dynamic
    dimension -- is excluded from both :attr:`tensor_offsets` and
    :attr:`naive_bytes` and listed in :attr:`unplanned` instead of guessed, so
    a plan with a non-empty :attr:`unplanned` is a partial lower bound, not a
    complete plan.

    Control-flow (``If``/``Loop``/``Scan``) subgraph bodies are planned too,
    but independently: each gets its own nested :class:`MemoryPlan` in
    :attr:`subgraph_plans`, with its own arena starting at offset 0 -- never
    shared address space with this plan's own :attr:`tensor_offsets`, or with
    a sibling subgraph's. This plan instead reserves room for a subgraph's
    peak requirement inside *its own* arena for the span its owning node
    executes (:attr:`subgraph_reserved_bytes`, already folded into
    :attr:`arena_bytes`) -- the same "subgraph peak added on top of the live
    set at that node" convention :func:`onnxsim.model_info.ModelInfo`'s
    ``memory_footprint`` uses for its own peak figure.
    """

    arena_bytes: int
    naive_bytes: int
    tensor_offsets: Dict[str, Tuple[int, int]]
    unplanned: List[str]
    subgraph_reserved_bytes: int = 0
    subgraph_plans: Dict[str, "MemoryPlan"] = dataclasses.field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Fraction of :attr:`naive_bytes` the plan saves:
        ``1 - arena_bytes / naive_bytes``. ``0.0`` when there is nothing
        plannable (``naive_bytes == 0``).
        """
        if self.naive_bytes == 0:
            return 0.0
        return 1.0 - self.arena_bytes / self.naive_bytes


def _memory_plan_from_tuple(raw: tuple) -> MemoryPlan:
    # Mirrors MemoryPlanToPyTuple's recursive shape on the C++ side
    # (cpp2py_export.cc): (offsets, arena_bytes, naive_bytes, unplanned,
    # subgraph_reserved_bytes, subgraph_plans), with subgraph_plans itself a
    # dict of the same tuple shape, one level per control-flow subgraph.
    (
        offsets,
        arena_bytes,
        naive_bytes,
        unplanned,
        subgraph_reserved_bytes,
        subgraph_plans,
    ) = raw
    return MemoryPlan(
        arena_bytes=arena_bytes,
        naive_bytes=naive_bytes,
        tensor_offsets=dict(offsets),
        unplanned=list(unplanned),
        subgraph_reserved_bytes=subgraph_reserved_bytes,
        subgraph_plans={
            key: _memory_plan_from_tuple(sub) for key, sub in subgraph_plans.items()
        },
    )


def plan_activation_memory(
    model: onnx.ModelProto, run_shape_inference: bool = True
) -> MemoryPlan:
    """Compute a :class:`MemoryPlan` for ``model``'s top-level graph, plus a
    nested :class:`MemoryPlan` per control-flow subgraph body (see
    :attr:`MemoryPlan.subgraph_plans`).

    Delegates the allocation itself to the C++ core
    (``onnxsim/memory_planning.h``): a greedy best-fit placement over each
    tensor's liveness interval (from production to last use, the same
    liveness convention :class:`onnxsim.model_info.ModelInfo`'s
    ``memory_footprint`` uses), largest tensor first. ``model`` is not
    modified. Pass ``run_shape_inference=False`` when ``model`` already
    carries populated shapes (e.g. inferred with data propagation) to skip
    the extra pass.
    """
    from onnxsim import onnxsim_cpp2py_export as _C

    raw = _C._memory_plan(model.SerializeToString(), run_shape_inference)
    return _memory_plan_from_tuple(raw)


def print_memory_plan(
    plan: MemoryPlan, limit: int = 50, _title: str = "Activation Memory Plan"
) -> None:
    """Pretty-print ``plan`` as a table of tensor offsets (ordered by offset,
    then name), followed by the arena/naive totals and compression ratio, and
    -- when non-empty -- the list of tensors :func:`plan_activation_memory`
    could not place. Capped at ``limit`` rows (with a "... and N more" line)
    so a large model's plan stays readable. Recurses into
    :attr:`MemoryPlan.subgraph_plans`, printing each nested plan under its
    own heading after the top-level table.
    """
    table = Table(title=_title)
    table.add_column("Tensor")
    table.add_column("Offset")
    table.add_column("Size")

    ordered = sorted(plan.tensor_offsets.items(), key=lambda kv: (kv[1][0], kv[0]))
    for name, (offset, size) in ordered[:limit]:
        table.add_row(name, human_readable_size(offset), human_readable_size(size))
    print(table)
    if len(ordered) > limit:
        print(Text(f"... and {len(ordered) - limit} more", style="dim"))

    print(
        f"Arena: {human_readable_size(plan.arena_bytes)} "
        f"(naive: {human_readable_size(plan.naive_bytes)}, "
        f"{plan.compression_ratio * 100:.1f}% smaller)"
    )
    if plan.unplanned:
        shown = ", ".join(plan.unplanned[:10])
        more = ", ..." if len(plan.unplanned) > 10 else ""
        print(
            Text(
                f"{len(plan.unplanned)} tensor(s) could not be planned "
                f"(unknown shape/dtype or a dynamic dimension): {shown}{more}",
                style="yellow",
            )
        )
    if plan.subgraph_reserved_bytes:
        print(
            f"Reserved for control-flow subgraphs: "
            f"{human_readable_size(plan.subgraph_reserved_bytes)} "
            f"(already included in the arena above)"
        )
    for key, sub in plan.subgraph_plans.items():
        print()
        print_memory_plan(sub, limit=limit, _title=f"Subgraph '{key}'")


def _join_capped(items: List[str], limit: int) -> str:
    # Same "capped list, one line" shape as model_info.h's C++ JoinCapped /
    # RecordCappedListMetadata: join up to `limit` entries with ", ", appending
    # a "... (+N more)" marker when there were more, so a long unplanned list
    # doesn't blow up a single metadata_props value.
    joined = ", ".join(items[:limit])
    if len(items) > limit:
        joined += f", ... (+{len(items) - limit} more)"
    return joined


def _iter_node_subgraphs(node: onnx.NodeProto):
    # Same "singular g attribute, then each of a repeated graphs attribute,
    # per node.attribute in proto order" traversal onnxsim's C++ core
    # (model_info.cpp's BuildGraphView) uses to build a NodeView's
    # `subgraphs` vector, so the index handed out here lines up exactly with
    # the "<key_base>#<index>" keys ComputeActivationMemoryPlan
    # (memory_planning.cpp) used to key MemoryPlan.subgraph_plans.
    for attr in node.attribute:
        if attr.HasField("g"):
            yield attr.g
        for sub_graph in attr.graphs:
            yield sub_graph


def _annotate_subgraphs_of(
    graph: onnx.GraphProto, plan: MemoryPlan, prefix: str
) -> None:
    # Find each node's control-flow subgraph body/bodies (If/Loop/Scan) and
    # recurse into any this `plan` has a nested MemoryPlan for -- see
    # _iter_node_subgraphs and _annotate_subgraph_plan.
    for i, node in enumerate(graph.node):
        key_base = next((name for name in node.output if name), f"#node{i}")
        for sub_idx, sub_graph in enumerate(_iter_node_subgraphs(node)):
            sub_plan = plan.subgraph_plans.get(f"{key_base}#{sub_idx}")
            if sub_plan is not None:
                _annotate_subgraph_plan(sub_graph, sub_plan, prefix)


def _annotate_subgraph_plan(
    graph: onnx.GraphProto, plan: MemoryPlan, prefix: str
) -> None:
    """Write one nested :class:`MemoryPlan` (a value of
    :attr:`MemoryPlan.subgraph_plans`) onto its owning control-flow subgraph
    body: value-level ``mem_offset``/``mem_size`` the same way the top-level
    plan gets them, plus the same three totals :func:`annotate_memory_plan`
    puts on the model -- there being no model-level object to attach a
    subgraph body's own totals to instead. Then recurses into any subgraphs
    nested inside *this* one.
    """
    value_infos = {
        vi.name: vi
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info)
    }
    for name, (offset, size) in plan.tensor_offsets.items():
        vi = value_infos.get(name)
        if vi is not None:
            _set_metadata(vi, prefix + "mem_offset", str(offset))
            _set_metadata(vi, prefix + "mem_size", str(size))

    _set_metadata(graph, prefix + "memory_plan_arena_bytes", str(plan.arena_bytes))
    _set_metadata(graph, prefix + "memory_plan_naive_bytes", str(plan.naive_bytes))
    _set_metadata(
        graph,
        prefix + "memory_plan_compression_ratio",
        f"{plan.compression_ratio:.4f}",
    )
    _set_metadata(
        graph, prefix + "memory_plan_unplanned_count", str(len(plan.unplanned))
    )
    if plan.unplanned:
        _set_metadata(
            graph, prefix + "memory_plan_unplanned", _join_capped(plan.unplanned, 20)
        )
    if plan.subgraph_reserved_bytes:
        _set_metadata(
            graph,
            prefix + "memory_plan_subgraph_reserved_bytes",
            str(plan.subgraph_reserved_bytes),
        )

    _annotate_subgraphs_of(graph, plan, prefix)


def annotate_memory_plan(
    model: onnx.ModelProto, prefix: str = METADATA_PREFIX
) -> onnx.ModelProto:
    """Return a shape-inferred copy of ``model`` with :func:`plan_activation_memory`'s
    result stored in ``metadata_props``, so a downstream consumer -- an embedded
    runtime, a code generator, anything that doesn't have onnxsim installed --
    can read the plan directly off the model instead of recomputing it:

    - **Model**: ``<prefix>memory_plan_arena_bytes``, ``naive_bytes`` and
      ``compression_ratio`` (:class:`MemoryPlan`'s totals); ``unplanned_count``
      always, plus ``unplanned`` (a capped, comma-joined list of names) when
      non-empty.
    - **Value** (inputs, outputs, value_info -- i.e. every *planned* tensor):
      ``<prefix>mem_offset`` and ``<prefix>mem_size``, matching
      :attr:`MemoryPlan.tensor_offsets`. A tensor onnxsim could not plan (see
      :attr:`MemoryPlan.unplanned`) is simply left unannotated, the same way
      :func:`annotate_metadata` leaves an unknown-shape tensor's ``bytes``
      unset rather than guessing.
    - **Control-flow subgraph body** (an ``If``/``Loop``/``Scan`` node's
      ``then_branch``/``else_branch``/``body`` ``GraphProto``, recursively):
      the same value-level ``mem_offset``/``mem_size`` pairs from that
      subgraph's own nested :class:`MemoryPlan`
      (:attr:`MemoryPlan.subgraph_plans`), plus that subgraph's own
      ``memory_plan_arena_bytes``/``naive_bytes``/``compression_ratio``/
      ``unplanned_count``/``unplanned`` written directly onto the subgraph
      ``GraphProto`` itself (there is no model-level object a subgraph body
      could otherwise attach totals to).

    Values are strings. The input model is never mutated; the returned copy is
    shape-inferred so intermediate values carry a matching ``value_info`` entry
    to annotate. Weights are never annotated here -- they are outside the
    activation arena by construction (see :func:`plan_activation_memory`).
    """
    plan = plan_activation_memory(model)
    # Work on an inferred copy: infer_shapes returns a fresh model on success,
    # but the original object on failure -- deep-copy first so the caller's
    # model is never touched and the annotations land on populated value_info.
    # Shape inference runs a second time here (plan_activation_memory already
    # ran it once internally, in C++, to compute the plan itself) -- the same
    # duplication onnxsim.model_info.annotate_metadata already accepts, for
    # the same reason: the C++ side never hands its shape-inferred model back.
    work = ModelInfo._infer_shapes(copy.deepcopy(model))

    value_infos = {
        vi.name: vi
        for vi in list(work.graph.input)
        + list(work.graph.output)
        + list(work.graph.value_info)
    }
    for name, (offset, size) in plan.tensor_offsets.items():
        vi = value_infos.get(name)
        if vi is not None:
            _set_metadata(vi, prefix + "mem_offset", str(offset))
            _set_metadata(vi, prefix + "mem_size", str(size))

    _set_metadata(work, prefix + "memory_plan_arena_bytes", str(plan.arena_bytes))
    _set_metadata(work, prefix + "memory_plan_naive_bytes", str(plan.naive_bytes))
    _set_metadata(
        work,
        prefix + "memory_plan_compression_ratio",
        f"{plan.compression_ratio:.4f}",
    )
    _set_metadata(
        work, prefix + "memory_plan_unplanned_count", str(len(plan.unplanned))
    )
    if plan.unplanned:
        _set_metadata(
            work, prefix + "memory_plan_unplanned", _join_capped(plan.unplanned, 20)
        )
    _annotate_subgraphs_of(work.graph, plan, prefix)
    return work
