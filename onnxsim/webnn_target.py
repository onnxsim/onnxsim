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
  "'shape' input should be a constant". A dynamically computed shape (common
  wherever a model isn't fully static) makes just that node fall back.

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
"""

from __future__ import annotations

from typing import List, Set, Union

import onnx


def _constant_producing_names(graph: onnx.GraphProto) -> Set[str]:
    names = {init.name for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            names.add(node.output[0])
    return names


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

    constant_names = _constant_producing_names(graph)
    for node in graph.node:
        if node.op_type not in ("Reshape", "Expand") or len(node.input) < 2:
            continue
        shape_input = node.input[1]
        if shape_input and shape_input not in constant_names:
            node_label = node.name or (node.output[0] if node.output else "<unnamed>")
            messages.append(
                f"{node.op_type} node {node_label!r} has a non-constant "
                f"shape input ({shape_input!r}); onnxruntime-web's WebNN "
                f"operator table requires {node.op_type}'s shape input to be "
                "constant, so this node will fall back off WebNN."
            )

    return messages
