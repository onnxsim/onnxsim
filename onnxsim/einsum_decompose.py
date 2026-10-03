"""Rewrite batched-matrix-multiply ``Einsum`` nodes into ``Transpose``/``MatMul``.

Every hand-written exporter in this package (Core ML, TFLite, WebNN) lowers one
ONNX op at a time out of an explicit handler table, and none of them has an
Einsum kernel. Each would otherwise refuse a graph containing the einsums torch
emits for linear layers and for ray/sample contractions -- ``ij,jd->id``,
``bij,bjk->bik``, ``nc,cd->nd`` -- which is most of what a NeRF, attention or
MLP graph is made of. Rather than duplicate that lowering three times, rewrite
the op once here at the ONNX level, on the way into every exporter.

Only the batched-matmul form is rewritten: exactly one axis shared by both
operands and summed over, exactly one free axis per side, and any remaining
shared axes in matching order on both sides and ahead of the free axes. A
diagonal, a trace, an outer product, a reduction to a scalar, or a batch axis
one operand carries and the other does not is left exactly as it was, so an
exporter that cannot handle it still raises its usual "unsupported op" error
naming the op -- never a silently wrong model. That is why the support is
deliberately narrower than a general einsum optimizer: it covers what real
graphs contain and refuses the rest.

The rewrite is exact, not approximate. ONNX MatMul computes
``A[batch, m, k] @ B[batch, k, n] -> [batch, m, n]``, and the operand
permutations below put each side in precisely that layout -- which is also the
einsum's own output order, so no output-side reshape or transpose is needed.
An operand already in that order is used as-is and contributes no Transpose.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Set, Tuple

import onnx
from onnx import helper

__all__ = ["decompose_einsum"]


def _equation(node: onnx.NodeProto) -> str:
    """The node's ``equation`` attribute, whitespace-stripped (ONNX allows it)."""
    for attr in node.attribute:
        if attr.name == "equation":
            # `attr.s` decodes to `str` on some protobuf builds and `bytes` on
            # others; ONNX permits surrounding whitespace in the equation.
            raw = attr.s
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return "".join(raw.split())
    return ""


def _is_label_run(term: str) -> bool:
    """True if ``term`` is a run of distinct lowercase ASCII letters."""
    return bool(term) and all("a" <= c <= "z" for c in term) and len(set(term)) == len(
        term
    )


def _batch_matmul_perms(
    term1: str, term2: str, out: str
) -> Optional[Tuple[List[int], List[int]]]:
    """Permutations that turn a two-operand einsum into one batched MatMul.

    Returns ``(perm1, perm2)``, where ``perm1``/``perm2`` reorder the two
    operands' axes into ``batch ++ [m, k]`` and ``batch ++ [k, n]``, or
    ``None`` when the equation is not exactly of that shape.
    """
    left, right, result = set(term1), set(term2), set(out)
    if not (_is_label_run(term1) and _is_label_run(term2) and _is_label_run(out)):
        return None

    # The single axis present in both operands that the result *drops* is the
    # contraction MatMul performs (it is summed over). An axis present in both
    # operands but *kept* in the result is a batch axis, broadcast rather than
    # contracted -- which is what separates `j` from `b` in `bij,bjk->bik`.
    contracted = (left & right) - result
    if len(contracted) != 1:
        return None
    # Exactly one free axis must survive on each side (MatMul's m and n).
    free_left, free_right = left - right, right - left
    if len(free_left) != 1 or len(free_right) != 1:
        return None
    k = next(iter(contracted))
    m = next(iter(free_left))
    n = next(iter(free_right))

    # The remaining shared-and-kept axes are batch axes. MatMul treats the
    # *last two* axes of each operand as the matrix and only broadcasts leading
    # ones, so every batch axis must come before this operand's two matrix axes.
    batch = (left & right) & result
    batch_seq = [c for c in out if c in batch]
    if [c for c in term1 if c in batch] != batch_seq:
        return None
    if [c for c in term2 if c in batch] != batch_seq:
        return None

    # MatMul emits ``batch ++ [m, n]`` in exactly that order, so an einsum whose
    # output interleaves those axes differently would need an output transpose.
    if out != "".join(batch_seq) + m + n:
        return None

    def operand_perm(term: str, matrix_axes: Sequence[str]) -> List[int]:
        # Every batch axis must precede the two matrix axes: ONNX MatMul reads
        # the *trailing* two axes as the matrix and broadcasts only leading
        # ones. Operand 1's matrix axes are ``[m, k]`` (free, then contracted);
        # operand 2's are ``[k, n]`` (contracted, then free) -- the asymmetry is
        # what makes ``A @ B`` contract k and is why an operand whose free axis
        # already sits left of k needs no transpose while the other does.
        return [term.index(c) for c in batch_seq + list(matrix_axes)]

    return operand_perm(term1, [m, k]), operand_perm(term2, [k, n])


def _used_tensor_names(graph: onnx.GraphProto) -> Set[str]:
    names: Set[str] = set()
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(value.name)
    for init in graph.initializer:
        names.add(init.name)
    for node in graph.node:
        names.update(n for n in node.input if n)
        names.update(n for n in node.output if n)
    return names


def _rewrite_einsum(
    node: onnx.NodeProto, used: Set[str], counter: List[int]
) -> Optional[List[onnx.NodeProto]]:
    """Nodes replacing ``node``, or ``None`` to leave it in place."""
    equation = _equation(node)
    if equation.count("->") != 1:
        return None
    lhs, out = equation.split("->", 1)
    terms = lhs.split(",")
    if len(terms) != 2:
        return None
    plan = _batch_matmul_perms(terms[0], terms[1], out)
    if plan is None:
        return None
    perm1, perm2 = plan

    base = node.name or (node.output[0] if node.output else node.op_type)

    def fresh(tag: str) -> str:
        counter[0] += 1
        name = f"{base}_{tag}_{counter[0]}"
        while name in used:
            counter[0] += 1
            name = f"{base}_{tag}_{counter[0]}"
        used.add(name)
        return name

    nodes: List[onnx.NodeProto] = []

    def operand(source: str, perm: Sequence[int], tag: str) -> str:
        if list(perm) == list(range(len(perm))):
            return source
        name = fresh(tag)
        nodes.append(helper.make_node("Transpose", [source], [name], perm=list(perm)))
        return name

    a = operand(node.input[0], perm1, "einsum_a")
    b = operand(node.input[1], perm2, "einsum_b")
    nodes.append(helper.make_node("MatMul", [a, b], list(node.output)))
    return nodes


def decompose_einsum(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return ``model`` with its batched-matmul ``Einsum`` nodes decomposed.

    The input is never mutated: a model with no rewritable Einsum is returned
    unchanged (and by identity), and otherwise a rewritten copy is returned.
    """
    graph = model.graph
    if not any(
        n.op_type == "Einsum" and n.domain in ("", "ai.onnx") for n in graph.node
    ):
        return model

    out = copy.deepcopy(model)
    used = _used_tensor_names(out.graph)
    counter = [0]
    new_nodes: List[onnx.NodeProto] = []
    rewritten = False
    for node in out.graph.node:
        if node.op_type == "Einsum" and node.domain in ("", "ai.onnx"):
            replacement = _rewrite_einsum(node, used, counter)
            if replacement is not None:
                new_nodes.extend(replacement)
                rewritten = True
                continue
        new_nodes.append(node)
    if not rewritten:
        return model
    del out.graph.node[:]
    out.graph.node.extend(new_nodes)
    return out