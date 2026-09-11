"""Flags ``com.microsoft::Attention`` nodes that onnxruntime-web's WebGPU
execution provider cannot actually accelerate.

onnxruntime-web's WebGPU backend lists both ``Attention`` and
``MultiHeadAttention`` as supported ops, but its own operator table
(``js/web/docs/webgpu-operators.md``) annotates both with "need implementing
mask and past/present": the WebGPU kernel does not yet handle the
``mask_index`` or ``past``/``present`` KV-cache inputs. A node using either
one is still a *valid* graph -- ``onnx.checker`` has nothing to say about it
-- but at runtime onnxruntime-web falls back to a different execution
provider (wasm/CPU) for that node instead of running it on the GPU, silently
losing the acceleration a "webgpu target" simplification was meant to get.

:func:`onnxsim.fuse_attention <onnxsim.onnx_simplifier>`'s own fusion
(``onnxsim/passes/fuse_attention.h``) never produces this shape itself -- it
only matches self-attention with no mask and no past/present to begin with --
so this only ever fires on an ``Attention`` node that was already in the
input model (e.g. exported by another tool) before reaching onnxsim, and
survives simplification unchanged. ``MultiHeadAttention`` is not covered:
onnxsim has no pass that produces or consumes that op, so there is nothing in
this codebase to inspect it for.

Meant to be called on the *output* of
``onnxsim.simplify(model, gemm_fusion_backend="webgpu")`` -- the point where
the model is about to be shipped to a WebGPU-targeting caller -- not on the
input model, since simplification could in principle still add or remove
such a node.

:func:`estimate_webgpu_islands` goes one step further: instead of just
listing offending nodes, it estimates how many separate contiguous "WebGPU
islands" they split the rest of the graph into, and how many device-copy
boundaries result -- see ``onnxsim._ep_fragmentation`` for what that means
and, importantly, what it does *not* claim to know (it isn't a full
simulation of ONNX Runtime's partitioner -- only the specific gap this module
checks for).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import onnx

from onnxsim._ep_fragmentation import IslandReport, estimate_fragmentation

# com.microsoft::Attention's positional input order (ContribOperators.md):
# 0 input, 1 weights, 2 bias, 3 mask_index, 4 past, 5 attention_bias,
# 6 past_sequence_length. Only mask_index/past are checked here: they are the
# two onnxruntime-web's WebGPU Attention kernel documents as unimplemented.
_MASK_INDEX_INPUT_POSITION = 3
_PAST_INPUT_POSITION = 4


def _flagged_attention_nodes(graph: onnx.GraphProto) -> List[Tuple[int, str]]:
    """Every ``com.microsoft::Attention`` node using ``mask_index``/``past``,
    as ``(node index in graph.node, message)`` pairs -- the single source of
    truth both :func:`check_webgpu_attention_support` and
    :func:`estimate_webgpu_islands` build on.
    """
    flagged = []
    for i, node in enumerate(graph.node):
        if node.domain != "com.microsoft" or node.op_type != "Attention":
            continue
        unsupported = []
        if (
            len(node.input) > _MASK_INDEX_INPUT_POSITION
            and node.input[_MASK_INDEX_INPUT_POSITION]
        ):
            unsupported.append(f"mask_index={node.input[_MASK_INDEX_INPUT_POSITION]!r}")
        if len(node.input) > _PAST_INPUT_POSITION and node.input[_PAST_INPUT_POSITION]:
            unsupported.append(f"past={node.input[_PAST_INPUT_POSITION]!r}")
        if not unsupported:
            continue
        node_label = node.name or (node.output[0] if node.output else "<unnamed>")
        flagged.append(
            (
                i,
                f"Attention node {node_label!r} has {' and '.join(unsupported)} "
                "wired up; onnxruntime-web's WebGPU execution provider does not "
                "yet implement mask/past-present support for Attention, so this "
                "node will fall back to a different execution provider instead "
                "of running on the GPU.",
            )
        )
    return flagged


def check_webgpu_attention_support(
    model: Union[str, onnx.ModelProto],
) -> List[str]:
    """Scans for ``com.microsoft::Attention`` nodes wired up with a
    ``mask_index`` and/or ``past`` input -- the configuration
    onnxruntime-web's WebGPU execution provider does not accelerate (see this
    module's docstring).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending node (empty if none);
            each message names the node and which unsupported input(s) it
            uses. This is advisory only -- it does not modify ``model`` or
            raise, since the graph is still perfectly valid, just not
            GPU-accelerated for that node.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return [msg for _, msg in _flagged_attention_nodes(model.graph)]


def estimate_webgpu_islands(model: Union[str, onnx.ModelProto]) -> IslandReport:
    """Estimates how much the ``mask_index``/``past`` ``Attention`` nodes
    :func:`check_webgpu_attention_support` flags fragment the rest of the
    graph into separate WebGPU-accelerated islands (see this module's
    docstring and ``onnxsim._ep_fragmentation`` for the method and its
    limits: this only accounts for the specific gap this module checks for,
    so it is a lower bound on real fragmentation, not a full simulation of
    ONNX Runtime's partitioner).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: an :class:`onnxsim._ep_fragmentation.IslandReport`
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    flagged = _flagged_attention_nodes(model.graph)
    return estimate_fragmentation(model.graph, {i for i, _ in flagged})
