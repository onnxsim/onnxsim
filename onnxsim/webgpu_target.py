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
"""

from __future__ import annotations

from typing import List, Union

import onnx

# com.microsoft::Attention's positional input order (ContribOperators.md):
# 0 input, 1 weights, 2 bias, 3 mask_index, 4 past, 5 attention_bias,
# 6 past_sequence_length. Only mask_index/past are checked here: they are the
# two onnxruntime-web's WebGPU Attention kernel documents as unimplemented.
_MASK_INDEX_INPUT_POSITION = 3
_PAST_INPUT_POSITION = 4


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

    messages = []
    for node in model.graph.node:
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
        messages.append(
            f"Attention node {node_label!r} has {' and '.join(unsupported)} "
            "wired up; onnxruntime-web's WebGPU execution provider does not "
            "yet implement mask/past-present support for Attention, so this "
            "node will fall back to a different execution provider instead "
            "of running on the GPU."
        )
    return messages
