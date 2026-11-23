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

Three-or-more-operand einsums go through the same rewrite one pairwise step
at a time: when the optional ``opt_einsum`` dependency (the ``einsum`` extra)
is installed, its contraction path splits e.g. ``ij,jk,kl->il`` into binary
steps and each step is rewritten above. Without ``opt_einsum``, with unknown
input shapes, or when any step is not a batched matmul, the node is left
exactly as it was.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Set, Tuple

import onnx
from onnx import helper

try:
    import opt_einsum
except ImportError:  # optional dependency; see _split_einsum below.
    opt_einsum = None

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
    return (
        bool(term)
        and all("a" <= c <= "z" for c in term)
        and len(set(term)) == len(term)
    )


def _matmul_parts(
    term1: str, term2: str, out: str
) -> Optional[Tuple[str, str, str, List[int], List[int]]]:
    """MatMul-order parts and operand perms, or ``None`` if not a batched matmul.

    Returns ``(m, n, batch_seq, perm1, perm2)`` where the MatMul-order output
    is ``batch_seq + m + n`` even when ``out`` permutes those axes differently
    (the caller then appends an output transpose).
    """
    left, right, result = set(term1), set(term2), set(out)
    if not (_is_label_run(term1) and _is_label_run(term2) and _is_label_run(out)):
        return None

    # The single axis present in both operands that the result *drops* is the
    # contraction MatMul performs (it is summed over). An axis present in both
    # operands but *kept* in the result is a batch axis, broadcast rather than
    # contracted -- which is what separates `j` from `b` in `bij,bjk->bik`.
    #
    # A label on exactly one operand but missing from the output is that
    # operand's free axis only if the other operand broadcasts over it: e.g.
    # `bnc,cd->...` contracts `c` while `b` (rank-2 `cd` has no `b` axis) and
    # `n` survive. A label one operand carries that neither survives into the
    # output nor appears on the other side would be silently dropped.
    contracted = (left & right) - result
    if len(contracted) != 1:
        return None
    # MatMul's m is operand 1's free axis (carried there, kept in the result),
    # and n is operand 2's. A label on exactly one side that survives (e.g. `b`
    # in `bnc,cd->bnd`, where rank-2 `cd` has no `b` axis) broadcasts instead of
    # contracting, so each side's surviving labels are its broadcast axes plus
    # its one free axis -- exactly the axes MatMul's broadcasting carries.
    # Anything beyond that plus the contraction and the shared-and-kept batch
    # axes (e.g. the `j` of `ijk,jl->ikl`, kept on the other side) is a second
    # contraction no single MatMul performs. The per-side free axes must
    # survive into the result: a one-sided label that survives (e.g. `b` in
    # `bnc,cd->bnd`) is exactly operand 1's `m`, broadcast on operand 2.
    k = next(iter(contracted))
    # Operand 1 contributes ``[m, k]`` and operand 2 ``[k, n]``: m is operand
    # 1's surviving free axis (present there, absent from operand 2) and n is
    # operand 2's. Any other surviving one-sided label (e.g. `b` in
    # `bnc,cd->bnd`, where rank-2 `cd` has no `b` axis) broadcasts instead,
    # carried through by MatMul's own leading-dim broadcasting. Anything one
    # side carries beyond m/k, the batch axes, and its broadcast labels (e.g.
    # the `j` of `ijk,jl->ikl`, kept on the other side) is a second
    # contraction no single MatMul performs.
    m_only = (left - right) & result
    n_only = (right - left) & result
    if len(m_only) != 1 or len(n_only) != 1:
        # More than one surviving one-sided label per side (e.g. `b` AND `n`
        # in `bnc,...`): the extras broadcast. The matrix dimension is the
        # one adjacent to the contraction in that operand's term order; the
        # rest are leading broadcast axes MatMul carries through.
        m_cands = [c for c in term1 if c in m_only]
        n_cands = [c for c in term2 if c in n_only]
        if not m_cands or not n_cands:
            return None
        pos1, pos2 = term1.index(k), term2.index(k)
        # Operand 1 ends ``[..., m, k]``: among equidistant candidates the
        # LATER one keeps more of the suffix in place. Operand 2 starts
        # ``[k, n, ...]``: among equidistant candidates the EARLIER one does.
        m = min(m_cands, key=lambda c: (abs(term1.index(c) - pos1), -term1.index(c)))
        n = min(n_cands, key=lambda c: (abs(term2.index(c) - pos2), term2.index(c)))
    else:
        m = next(iter(m_only))
        n = next(iter(n_only))
    if m not in left or n not in right:
        return None
    # Every axis each operand carries must be a batch axis (shared AND kept),
    # the contraction, or that side's free axis. A one-sided label that
    # survives (e.g. `b` in `bnc,cd->...`, where rank-2 `cd` has no `b` axis)
    # broadcasts instead of contracting, so it sits ahead of the matrix axes
    # and MatMul's own broadcasting carries it through. A label one side
    # carries that neither survives nor appears on the other side would be
    # silently dropped; two shared-and-dropped axes (e.g. the `c` and `h` of
    # `nchw,ncd->ndh`) are a second contraction no single MatMul performs.
    batch = (left & right) & result
    # A one-sided label that survives (e.g. `b` in `bnc,cd->bnd`, where rank-2
    # `cd` has no `b` axis) broadcasts, so it is allowed; anything one side
    # carries beyond the contraction, its free axis, and the batch axes (e.g.
    # the `j` of `ijk,jl->ikl`, kept on the other side) is a second
    # contraction no single MatMul performs.
    if left - {m} - {k} - batch - (result & left):
        return None
    if right - {n} - {k} - batch - (result & right):
        return None
    # Every result label is batch, m, n, or a broadcast axis one side carries
    # and the other broadcasts over (e.g. `b` in `bnc,cd->bnd`).
    if result - batch - {m} - {n} - (result & (left | right)):
        return None
    batch_seq = [c for c in term1 if c in batch]
    if [c for c in term2 if c in batch] != batch_seq:
        return None
    if sorted(batch_seq) != sorted(c for c in out if c in batch):
        return None

    # Operand layout: batch axes, then any broadcast axes, then the two matrix
    # axes -- all must precede the matrix pair, since ONNX MatMul reads the
    # *trailing* two axes as the matrix and broadcasts only leading ones. The
    # MatMul-order output is batch ++ broadcast ++ [m, n], where the broadcast
    # axes of BOTH sides merge (each side's broadcast axes are its surviving
    # one-sided labels: e.g. operand 2's `b` in `cd,bnc->dbn`, where rank-2
    # `cd` has no `b` axis -- MatMul broadcasts rank-2 `cd` over it).
    bcast1 = [c for c in term1 if c in result and c not in batch and c != m]
    bcast2 = [c for c in term2 if c in result and c not in batch and c != n]
    bcast = bcast1 + [c for c in bcast2 if c not in bcast1]
    mm_batch = "".join(batch_seq + bcast)

    def operand_perm(
        term: str, own_bcast: List[str], matrix_axes: Sequence[str]
    ) -> Optional[List[int]]:
        # Every batch/broadcast axis must precede the two matrix axes: ONNX
        # MatMul reads the *trailing* two axes as the matrix and broadcasts
        # only leading ones. Operand 1's matrix axes are ``[m, k]`` (free, then
        # contracted); operand 2's are ``[k, n]`` (contracted, then free) --
        # the asymmetry is what makes ``A @ B`` contract k and is why an
        # operand whose free axis already sits left of k needs no transpose
        # while the other does. A side that lacks a merged broadcast axis
        # (e.g. rank-2 `cd` lacking `b`) broadcasts over it -- but only if it
        # is exactly the trailing matrix pair, i.e. rank 2 with no leading
        # axes of its own; anything else would need an Unsqueeze this rewrite
        # does not emit.
        if any(c not in term for c in bcast):
            if len(term) != 2:
                return None
        try:
            return [term.index(c) for c in batch_seq + own_bcast + list(matrix_axes)]
        except ValueError:
            return None

    perm1 = operand_perm(term1, bcast1, [m, k])
    perm2 = operand_perm(term2, bcast2, [k, n])
    if perm1 is None or perm2 is None:
        return None

    return (
        m,
        n,
        mm_batch,
        perm1,
        perm2,
    )


def _batch_matmul_perms(
    term1: str, term2: str, out: str
) -> Optional[Tuple[List[int], List[int]]]:
    """Permutations that turn a two-operand einsum into one batched MatMul.

    Returns ``(perm1, perm2)``, where ``perm1``/``perm2`` reorder the two
    operands' axes into ``batch ++ [m, k]`` and ``batch ++ [k, n]``, or
    ``None`` when the equation is not exactly of that shape.
    """
    parts = _matmul_parts(term1, term2, out)
    if parts is None:
        return None
    m, n, batch_seq, perm1, perm2 = parts
    # MatMul emits ``batch ++ [m, n]`` in exactly that order, so an einsum whose
    # output interleaves those axes differently would need an output transpose.
    if out != batch_seq + m + n:
        return None
    return perm1, perm2


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
    if plan is not None:
        perm1, perm2 = plan
        out_perm = None
    else:
        parts = _matmul_parts(terms[0], terms[1], out)
        if parts is None:
            return None
        m, n, batch_seq, perm1, perm2 = parts
        mm_out = batch_seq + m + n
        if mm_out == out:
            out_perm = None
        else:
            out_perm = [mm_out.index(c) for c in out]

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
    if out_perm is None:
        nodes.append(helper.make_node("MatMul", [a, b], list(node.output)))
    else:
        mm = fresh("einsum_mm")
        nodes.append(helper.make_node("MatMul", [a, b], [mm]))
        nodes.append(
            helper.make_node("Transpose", [mm], list(node.output), perm=out_perm)
        )
    return nodes


def _input_shapes(
    graph: onnx.GraphProto, node: onnx.NodeProto
) -> Optional[List[Tuple[int, ...]]]:
    """Concrete shapes of ``node``'s inputs, or ``None`` if any is unknown."""
    shapes: Dict[str, Tuple[int, ...]] = {}
    for value in list(graph.input) + list(graph.value_info):
        dims = value.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            shapes[value.name] = tuple(d.dim_value for d in dims)
    for init in graph.initializer:
        shapes[init.name] = tuple(init.dims)
    try:
        return [shapes[name] for name in node.input]
    except KeyError:
        return None


def _split_einsum(
    equation: str, shapes: List[Tuple[int, ...]]
) -> Optional[List[Tuple[Tuple[int, ...], str, Tuple[str, ...]]]]:
    """Binary sub-equations contracting the operands in path order.

    Returns a list of ``(inds, equation, remaining)`` where ``inds`` indexes
    the step's two operands in the CURRENT operand list, ``equation`` is the
    step's own two-operand equation, and ``remaining`` labels that list after
    the step (the next step's ``inds`` index into it). Step 0's list is the
    node's own input terms in order. Returns ``None`` when ``opt_einsum`` is
    not installed or the equation is not of the plain-label form the MatMul
    rewrite accepts.
    """
    if opt_einsum is None:
        return None
    try:
        expr = opt_einsum.contract_expression(equation, *shapes, shapes=True)
    except (ValueError, TypeError):
        return None
    steps = [(step[0], step[2], step[3]) for step in expr.contraction_list]
    if len(steps) != len(shapes) - 1:
        return None
    for _, step, _ in steps:
        lhs, _, _ = step.partition("->")
        if any("..." in term for term in lhs.split(",")):
            return None
    return steps


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
            if replacement is None and len(node.input) > 2:
                replacement = _rewrite_chain(out.graph, node, used, counter)
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


def _rewrite_chain(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    used: Set[str],
    counter: List[int],
) -> Optional[List[onnx.NodeProto]]:
    """Nodes replacing an N-operand ``Einsum`` via pairwise MatMuls."""
    equation = _equation(node)
    if equation.count("->") != 1:
        return None
    lhs, _ = equation.split("->", 1)
    if len(lhs.split(",")) != len(node.input):
        return None
    shapes = _input_shapes(graph, node)
    if shapes is None or any(len(t) != len(s) for t, s in zip(lhs.split(","), shapes)):
        return None
    steps = _split_einsum(equation, shapes)
    if steps is None:
        return None

    def fresh(tag: str) -> str:
        counter[0] += 1
        name = f"{node.output[0]}_{tag}_{counter[0]}"
        while name in used:
            counter[0] += 1
            name = f"{node.output[0]}_{tag}_{counter[0]}"
        used.add(name)
        return name

    live: List[str] = list(node.input)
    nodes: List[onnx.NodeProto] = []
    # The path's per-step `inds` index into the CURRENT operand list, ordered
    # as [previous intermediate, remaining originals in input order] (step 0
    # has no intermediate: just the inputs in order). The step equation's two
    # terms are in inds order, so align them to live tensors via live_terms.
    live_terms: List[str] = lhs.split(",")

    def emit_operand(source: str, perm: Sequence[int], tag: str) -> str:
        if list(perm) == list(range(len(perm))):
            return source
        name = fresh(tag)
        nodes.append(helper.make_node("Transpose", [source], [name], perm=list(perm)))
        return name

    for i, (inds, step, remaining_terms) in enumerate(steps):
        terms, _, step_out = step.partition("->")
        term1, term2 = terms.split(",")
        # The step equation's term order need not match the operand-list
        # order (e.g. `jk,ij->ki` with inds (1, 0) lists B=`jk` first while
        # the operand list holds [..., `ij`, `jk`]): the MatMul perms belong
        # to the equation's own term1/term2, so bind each equation term to
        # the operand-list slot whose labels equal it. Duplicate label terms
        # (e.g. two `bki`-shaped operands) bind by slot, not by label.
        if len(inds) != 2 or max(inds) >= len(live):
            return None
        slots = sorted(inds)
        have = sorted(live_terms[j] for j in slots)
        if have != sorted((term1, term2)):
            return None
        if term1 == term2:
            a_in, b_in = (live[slots[0]], live[slots[1]])
        elif live_terms[slots[0]] == term1:
            a_in, b_in = live[slots[0]], live[slots[1]]
        else:
            a_in, b_in = live[slots[1]], live[slots[0]]
        # The path's output order need not match MatMul's ``batch ++ [m, n]``
        # layout, so lower against the MatMul-order equation and transpose the
        # result back into the step's own output order when they differ.
        parts = _matmul_parts(term1, term2, step_out)
        if parts is None:
            return None
        m, n, batch_seq, perm1, perm2 = parts
        mm_out = batch_seq + m + n
        a = emit_operand(a_in, perm1, "einsum_a")
        b = emit_operand(b_in, perm2, "einsum_b")
        if mm_out == step_out and i == len(steps) - 1:
            out_name = node.output[0]
        else:
            out_name = fresh(f"einsum_t{i}")
        nodes.append(helper.make_node("MatMul", [a, b], [out_name]))
        if mm_out != step_out:
            fixed = fresh(f"einsum_o{i}")
            nodes.append(
                helper.make_node(
                    "Transpose",
                    [out_name],
                    [node.output[0] if i == len(steps) - 1 else fixed],
                    perm=[mm_out.index(c) for c in step_out],
                )
            )
            out_name = node.output[0] if i == len(steps) - 1 else fixed
        # The next step's `inds` index into the operand list AFTER this step,
        # whose order opt_einsum reports as `remaining_terms`: rebuild both
        # parallels from it rather than assuming [intermediate, ...originals].
        # `remaining_terms` holds label terms; map each back to its tensor:
        # originals keep their input tensor, the new intermediate is out_name.
        by_term: Dict[str, List[str]] = {}
        for t, name in zip(live_terms, live):
            by_term.setdefault(t, []).append(name)
        # Consumed slots are gone; the survivors are the slots NOT in inds.
        for d in sorted(inds, reverse=True):
            t = live_terms[d]
            by_term[t].remove(live[d])
        by_term.setdefault(step_out, []).append(out_name)
        live = [by_term[t].pop(0) for t in remaining_terms]
        live_terms = list(remaining_terms)
    return nodes
