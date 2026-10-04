"""Rewrite integer ``BitwiseXor`` into ``And``/``Or``/``Sub``.

Why this exists
---------------
The Instant-NGP hash grid that nerfstudio-style NeRF encoders are built around
computes its space hash with ``torch.bitwise_xor``, which torch's dynamo
exporter lowers to ONNX ``BitwiseXor`` (opset 18). That is fine for
onnxruntime, but it has no kernel on several of the backends a NeRF graph would
otherwise be deployable to:

* **Qualcomm QNN HTP** -- the path every model in ``scripts/android/`` runs on
  -- has no known kernel for it and, as far as this repo is concerned, has never
  been probed. This is the backend the rewrite exists for.
* **TFLite** does have a ``BITWISE_XOR`` builtin (code 160), but only as a
  *builtin*, so it is unavailable to the int8/uint8 delegate graphs this repo
  builds. It has no integer ``BitwiseAnd``/``BitwiseOr`` either -- only the
  bool-only ``LOGICAL_AND``/``LOGICAL_OR`` -- so the rewrite also needs those
  two added to ``tflite_export`` to be usable there.
* **Core ML** cannot be helped by this at all: its MIL has no bitwise op of any
  kind (checked against ``SSAOpRegistry.core_ops``: no ``bitwise_and``,
  ``bitwise_or``, ``bitwise_xor`` or ``bitwise_not``). Rewriting ``BitwiseXor``
  into ``BitwiseOr``/``BitwiseAnd`` would only move the failure to the next
  unsupported op, so Core ML is deliberately *not* wired up to call this. An
  exact integer XOR needs some bitwise primitive; there is no arithmetic-only
  identity, because ``a - b - 2*min(a,b)`` and friends are wrong wherever the
  borrows interact.

The rewrite uses the standard exact identity

    a ^ b == (a | b) - (a & b)

because ``a | b >= a & b`` bitwise (every bit of the AND is also set in the OR),
so the subtraction never borrows across a bit and cannot wrap: the result is
exactly the XOR for every integer width, unsigned or signed, with no
intermediate overflow. The all-ones / all-zeros corners are covered in
``tests/test_bitwise_xor_decompose.py``.

This is a *legalisation* rewrite for a target that needs it, so it is not a
default ``simplify()`` pass. It is exposed as a function for a caller targeting
such a backend to apply explicitly, alongside :mod:`onnxsim.einsum_decompose`
and :mod:`onnxsim.gridsample_plan`, which are applied by the exporters
themselves.
"""

from __future__ import annotations

import copy
from typing import List, Set

import onnx
from onnx import helper

__all__ = ["decompose_bitwise_xor"]

# Integer tensor types ONNX allows for the bitwise ops. Restricted to these so
# a float-typed BitwiseXor (invalid, but representable in a malformed graph) is
# left alone rather than silently rewritten into arithmetic that would change a
# float NaN payload's bit pattern into a number.
_INTEGER_ELEM_TYPES = {
    onnx.TensorProto.UINT8,
    onnx.TensorProto.INT8,
    onnx.TensorProto.UINT16,
    onnx.TensorProto.INT16,
    onnx.TensorProto.UINT32,
    onnx.TensorProto.INT32,
    onnx.TensorProto.UINT64,
    onnx.TensorProto.INT64,
}


def _elem_types(graph: onnx.GraphProto) -> dict:
    """Map value name -> element type for every value the graph produces.

    Read from ``value_info``/``input``/``output``/``initializer``, then extended
    by walking the nodes so a tensor that is *only* a node output -- an
    intermediate with no declared ``value_info``, which is the common case in a
    graph that has not been through shape inference -- still gets a type. Without
    this, a chain ``x ^ y -> t; t ^ y -> z`` would only rewrite its first node,
    because ``t`` looks untyped and gets refused as "not an integer XOR".
    """
    types = {}
    for value in list(graph.value_info) + list(graph.input) + list(graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            types[value.name] = tensor_type.elem_type
    for initializer in graph.initializer:
        types[initializer.name] = initializer.data_type

    # Type-propagating ops, in the order they are walked, so a value's type is
    # known before the node consuming it is examined. Anything else is treated as
    # untyped, which makes the caller leave the node alone.
    _SAME_TYPE = {
        "BitwiseXor",
        "BitwiseAnd",
        "BitwiseOr",
        "Not",
        "Neg",
        "Abs",
        "Identity",
    }
    for node in graph.node:
        for name in node.output:
            if name and name not in types:
                if node.op_type in _SAME_TYPE and node.input:
                    source = types.get(node.input[0])
                    if source is not None:
                        types[name] = source
    return types


def _used_names(graph: onnx.GraphProto) -> Set[str]:
    names: Set[str] = set()
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(value.name)
    for initializer in graph.initializer:
        names.add(initializer.name)
    for node in graph.node:
        names.update(n for n in node.input if n)
        names.update(n for n in node.output if n)
    return names


def _rebuild_value_info(
    graph: onnx.GraphProto, derived: dict
) -> List[onnx.ValueInfoProto]:
    """Value infos for the tensors this pass introduces.

    The new intermediates carry the same element type and shape as the operands
    they are computed from, so each is declared as a copy of that operand's
    value info. Declaring them is not optional: an undeclared intermediate has
    no type, which ``onnx.checker`` rejects ("Field 'shape' of 'type' is
    required") and which also stops shape inference from checking the rewrite.
    """
    by_name = {
        v.name: v
        for v in list(graph.value_info) + list(graph.input) + list(graph.output)
    }
    out = []
    for name, source in derived.items():
        template = by_name.get(source)
        if template is None:
            continue
        new_value = onnx.ValueInfoProto()
        new_value.CopyFrom(template)
        new_value.name = name
        out.append(new_value)
    return out


def _rewrite_node(
    node: onnx.NodeProto, used: Set[str], counter: List[int]
) -> List[onnx.NodeProto]:
    """``a ^ b`` as ``(a | b) - (a & b)``; returns the replacement nodes."""
    if len(node.input) < 2 or not node.output or not node.output[0]:
        return [node]
    a, b = node.input[0], node.input[1]
    out = node.output[0]
    base = node.name or out

    def fresh(tag: str) -> str:
        counter[0] += 1
        name = f"{base}_{tag}_{counter[0]}"
        while name in used:
            counter[0] += 1
            name = f"{base}_{tag}_{counter[0]}"
        used.add(name)
        return name

    or_name = fresh("bor")
    and_name = fresh("band")
    return [
        helper.make_node("BitwiseOr", [a, b], [or_name], name=f"{base}_or"),
        helper.make_node("BitwiseAnd", [a, b], [and_name], name=f"{base}_and"),
        helper.make_node("Sub", [or_name, and_name], [out], name=node.name or base),
    ]


def decompose_bitwise_xor(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return ``model`` with integer ``BitwiseXor`` rewritten to And/Or/Sub.

    Only the two-operand form with a single output is rewritten. ``BitwiseXor``
    with an optional third ``direction`` attribute is left alone: that attribute
    selects bitwise vs logical behaviour, and ONNX defines it for the bool case
    only, so a non-default value must not be silently reinterpreted.

    The input is never mutated. A model with no rewritable node is returned
    unchanged and by identity.
    """
    graph = model.graph
    if not any(
        node.op_type == "BitwiseXor" and node.domain in ("", "ai.onnx")
        for node in graph.node
    ):
        return model

    out = copy.deepcopy(model)
    graph = out.graph
    types = _elem_types(graph)
    used = _used_names(graph)
    counter = [0]
    new_nodes: List[onnx.NodeProto] = []
    derived: dict = {}
    rewritten = False

    for node in graph.node:
        if node.op_type != "BitwiseXor" or node.domain not in ("", "ai.onnx"):
            new_nodes.append(node)
            continue
        # A non-integer operand means this is not the integer XOR the identity
        # is defined for; leave it for the target to report.
        if any(
            types.get(name, 0) not in _INTEGER_ELEM_TYPES for name in node.input[:2]
        ):
            new_nodes.append(node)
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            new_nodes.append(node)
            continue
        if any(attr.name == "direction" for attr in node.attribute):
            new_nodes.append(node)
            continue
        replacement = _rewrite_node(node, used, counter)
        # Remember what the two intermediates are, so they can be typed.
        derived[replacement[0].output[0]] = node.input[0]
        derived[replacement[1].output[0]] = node.input[1]
        new_nodes.extend(replacement)
        rewritten = True

    if not rewritten:
        return model

    del graph.node[:]
    graph.node.extend(new_nodes)
    existing = {v.name for v in graph.value_info}
    graph.value_info.extend(
        v for v in _rebuild_value_info(graph, derived) if v.name not in existing
    )
    return out
