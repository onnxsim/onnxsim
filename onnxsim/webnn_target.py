"""Flags graph shapes that onnxruntime-web's WebNN execution provider does
not support, so a "webnn target" simplification doesn't silently lose
acceleration to onnxruntime-web's ``wasm`` fallback.

WebNN delegates compute to the platform's own ML stack (DirectML / Core ML /
the platform NN API, see ``docs/webnn.md``), which is considerably stricter
about graph shape than WebGPU or wasm. Two gaps are checked here, both
already visible in this codebase or documented upstream:

- **INT64 graph inputs/outputs.** ``scripts/convertmodel/ort_log_capture.mjs``
  already special-cases the exact failure this produces at runtime --
  ``WebNN backend does not support data type: int64`` -- surfaced as an
  *uncaught promise rejection* rather than a normal session error (see that
  file's own comment). Several WebNN backends (Core ML in particular) don't
  support INT64 tensors, so a model exposing one at its input/output boundary
  can fail to even build a WebNN graph, forcing the whole session onto the
  ``wasm`` fallback -- not just the one incompatible node.
- **Non-constant ``Reshape``/``Expand`` shape input.** onnxruntime-web's own
  WebNN operator table
  (``js/web/docs/webnn-operators.md``) documents both ops' shape input as
  required to be constant: Reshape's note reads "Input 'shape' should be a
  constant, 0 dimension value in 'shape' is not supported", Expand's reads
  "'shape' input should be a constant". Neither note says what actually
  happens if it isn't. Running the equivalent WebGPU/Attention gap
  (``onnxsim.webgpu_target``) against a real onnxruntime-web build found that
  kind of "unsupported input configuration, not unsupported op" gap tends to
  throw at kernel-execution time rather than gracefully fall back --
  ``GetCapability`` accepts the op by type alone and cannot see that a
  specific input isn't constant. A one-off local run of this exact Reshape
  case against an experimental WebNN backend (not on the target macOS/real-EP
  hardware -- see ``scripts/convertmodel/test/webnn_reshape_placement.test.mjs``'s
  own comment) reproduced a runtime failure for the dynamic-shape case and
  not the constant one, consistent with that mechanism, though with different
  wording ("MLTensor(s) doesn't match the expectation") than the upstream
  note -- plausibly because a dynamic shape needs an extra runtime tensor the
  WebNN graph builder didn't account for. Treat "falls back" below as
  optimistic phrasing pending a confirmed run on real hardware: the more
  likely outcome, by analogy, is the whole session failing.

Both checks are necessarily heuristic: WebNN's actual operator/dtype support
varies by backend (CPU/GPU/NPU), browser version, and is still evolving (see
``docs/webnn.md``), so this cannot promise a flagged model will fail, nor
that an unflagged one will fully run on WebNN -- it surfaces the two
concrete, currently-documented gaps most likely to silently defeat a "webnn
target" simplification.

Meant to be called on the *output* of
``onnxsim.simplify(model, gemm_fusion_backend="webnn")`` -- the point where
the model is about to be shipped to a WebNN-targeting caller -- not on the
input model, since simplification could in principle still add or remove
such a node.

:func:`estimate_webnn_islands` goes one step further for the ``Reshape``/
``Expand`` gap specifically: instead of just listing offending nodes, it
estimates how many separate contiguous "WebNN islands" they split the rest of
the graph into, and how many device-copy boundaries result -- see
``onnxsim._ep_fragmentation`` for what that means and its limits. The INT64
graph-boundary gap is deliberately excluded from that estimate: it is a
whole-session risk (WebNN graph construction can fail entirely), not a
single node falling back, so it doesn't fit the "one node cut out of an
otherwise-contiguous island" model the estimate uses.
"""

from __future__ import annotations

from typing import List, Set, Tuple, Union

import onnx

from onnxsim._ep_fragmentation import IslandReport, estimate_fragmentation


def _constant_producing_names(graph: onnx.GraphProto) -> Set[str]:
    names = {init.name for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            names.add(node.output[0])
    return names


def _int64_boundary_messages(graph: onnx.GraphProto) -> List[str]:
    messages = []
    for kind, values in (("input", graph.input), ("output", graph.output)):
        for value_info in values:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type == onnx.TensorProto.INT64:
                messages.append(
                    f"Graph {kind} {value_info.name!r} is INT64; several "
                    "WebNN backends (notably Core ML) do not support INT64 "
                    "tensors, which can fail WebNN graph construction "
                    "entirely and fall the whole session back to wasm "
                    "rather than just this value."
                )
    return messages


def _flagged_reshape_expand_nodes(
    graph: onnx.GraphProto,
) -> List[Tuple[int, str]]:
    """Every ``Reshape``/``Expand`` node with a non-constant shape input, as
    ``(node index in graph.node, message)`` pairs -- the single source of
    truth both :func:`check_webnn_support` and :func:`estimate_webnn_islands`
    build on.
    """
    constant_names = _constant_producing_names(graph)
    flagged = []
    for i, node in enumerate(graph.node):
        if node.op_type not in ("Reshape", "Expand") or len(node.input) < 2:
            continue
        shape_input = node.input[1]
        if shape_input and shape_input not in constant_names:
            node_label = node.name or (node.output[0] if node.output else "<unnamed>")
            flagged.append(
                (
                    i,
                    f"{node.op_type} node {node_label!r} has a non-constant "
                    f"shape input ({shape_input!r}); onnxruntime-web's WebNN "
                    f"operator table requires {node.op_type}'s shape input to "
                    "be constant. This may not gracefully fall back off "
                    "WebNN -- see this module's docstring for why the more "
                    "likely outcome (by analogy with a confirmed, similar "
                    "WebGPU gap) is the whole session failing at runtime.",
                )
            )
    return flagged


def check_webnn_support(model: Union[str, onnx.ModelProto]) -> List[str]:
    """Scans for the two documented WebNN gaps described in this module's
    docstring: INT64-typed graph inputs/outputs, and ``Reshape``/``Expand``
    nodes whose shape input isn't a constant.

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending input/output/node
            (empty if none). This is advisory only -- it does not modify
            ``model`` or raise, since every flagged graph is still perfectly
            valid ONNX, just not (fully) WebNN-accelerated.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    graph = model.graph
    return _int64_boundary_messages(graph) + [
        msg for _, msg in _flagged_reshape_expand_nodes(graph)
    ]


def estimate_webnn_islands(model: Union[str, onnx.ModelProto]) -> IslandReport:
    """Estimates how much the non-constant-shape ``Reshape``/``Expand`` nodes
    :func:`check_webnn_support` flags fragment the rest of the graph into
    separate WebNN-accelerated islands, *if* this gap gracefully falls back
    the way ``onnxsim._ep_fragmentation`` assumes (see this module's
    docstring: the more likely outcome, by analogy with a confirmed WebGPU
    gap, is the whole session failing instead -- which this estimate would
    then understate, the same way ``onnxsim.webgpu_target``'s equivalent
    does). This only accounts for that one gap either way, so it is a lower
    bound on real fragmentation, not a full simulation of ONNX Runtime's
    partitioner, and it does not factor in the separate INT64 graph-boundary
    gap at all.

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: an :class:`onnxsim._ep_fragmentation.IslandReport`
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    flagged = _flagged_reshape_expand_nodes(model.graph)
    return estimate_fragmentation(model.graph, {i for i, _ in flagged})
