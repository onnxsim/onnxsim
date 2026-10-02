"""The operator fusions Quark's ONNX pre-processing runs on the float model
(``FuseInstanceNorm``, ``FuseL2Norm``, ``FuseLayerNorm`` and ``FuseGelu``), for
:mod:`onnxsim.quark_compat`.

Where Quark runs them
---------------------

``quark.onnx`` pre-processes the float model in this order: ``ConvertOpsetVersion``,
fp16 -> fp32, onnxslim (``SimplifyModel``), ``RemoveInputInit``, ``CopyBiasInit``,
ONNX Runtime's basic graph optimizations (``OptimizeModel``), and then -- one call, all
four flags on by default, whatever the preset and whether or not the two optimizers
above ran -- the fusions, in this order, followed by ``FoldBatchNorm``:

1. ``FuseInstanceNorm`` (:func:`fuse_instance_norm`),
2. ``FuseL2Norm`` (:func:`fuse_l2_norm`),
3. ``FuseLayerNorm`` (:func:`fuse_layer_norm`; opset >= 17 only),
4. ``FuseGelu`` (:func:`fuse_gelu`; opset >= 20 only).

``SkipPreprocess`` skips all of it. The first two are Quark's own matchers; the last
two are ONNX Runtime's ``onnxruntime.transformers`` fusions (``FusionLayerNormalization``
and ``FusionGelu``). None of them runs again after the quantization algorithms
(``apply_pre_optimization_after_algo`` passes all four flags off), and
``apply_post_optimization_*`` on the quantized graph passes them off too.

Because ONNX Runtime's *own* basic optimizer (``OptimizeModel``, on in every preset
but ``VINT8``) already fuses a torch-exported LayerNorm at opset >= 17 and a Gelu at
opset >= 20, these passes matter in practice (a) when ``OptimizeModel`` is off,
(b) for the two Quark-only patterns (a TensorFlow-style InstanceNorm built from
``GlobalAveragePool`` / ``Reciprocal``, and an L2 normalization built from
``ReduceSum`` / ``Max`` / ``Reciprocal``), and (c) for what ONNX Runtime does not
match; below opset 17 / 20 nothing is fused, whatever the options are.

What they emit
--------------

- LayerNorm: ``LayerNormalization(x, scale, bias)``, ``epsilon`` only (the default
  ``axis=-1``; Quark does not look at the axes of the ``ReduceMean`` nodes it matches),
  named ``LayerNorm_<n>``. ONNX Runtime's own fusion (``style="ort"``, what the basic
  optimizer leaves) also writes ``axis=-1`` and ``stash_type=1``.
- Gelu: a ``com.microsoft`` ``Gelu`` with no attribute, named ``Gelu_<n>`` -- the
  contrib op, not the ``ai.onnx`` one; ONNX Runtime's own fusion (``style="ort"``)
  writes ``ai.onnx`` ``Gelu`` with ``approximate="none"``.
- InstanceNorm: ``InstanceNormalization(x, scale, bias)`` (the ``[1, C, 1, 1]``
  initializers become ``[C]`` in place), named after the final ``Add``.
- L2Norm: ``LpNormalization(x)`` with ``p=2`` and the default ``axis=-1`` (again no
  look at the ``ReduceSum`` axes), named after the final ``Mul``.

The matchers are walked as Quark walks them (graph order, the same loose checks: an
op type here and there, no look at most of the constants), so a pattern that Quark
fuses is fused here and one it does not is not. Two deliberate differences: a
subgraph of an ``If`` / ``Loop`` / ``Scan`` is left alone, and a fusion that would
delete a tensor that is a graph output is skipped (Quark would write a broken graph).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

__all__ = [
    "apply_fusions",
    "fuse_gelu",
    "fuse_instance_norm",
    "fuse_l2_norm",
    "fuse_layer_norm",
]

# Quark: ``optimize_model`` skips the pass below these ai.onnx opsets (the fused op
# does not exist before them)
LAYER_NORM_MIN_OPSET = 17
GELU_MIN_OPSET = 20


def _copy(model: onnx.ModelProto) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _opset(model: onnx.ModelProto) -> Optional[int]:
    versions = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    return versions[0] if len(versions) == 1 else None


def _standalone(nodes: Sequence[onnx.NodeProto]) -> List[onnx.NodeProto]:
    out = []
    for n in nodes:
        c = onnx.NodeProto()
        c.CopyFrom(n)
        out.append(c)
    return out


def _set_nodes(graph: onnx.GraphProto, nodes: Sequence[onnx.NodeProto]) -> None:
    # (copies first: clearing the graph invalidates messages that still belong to it)
    fresh = _standalone(nodes)
    del graph.node[:]
    graph.node.extend(fresh)


def _graph_attr_names(node: onnx.NodeProto) -> Set[str]:
    """Every tensor name the subgraphs of ``node`` read."""
    names: Set[str] = set()

    def walk(g: onnx.GraphProto) -> None:
        for n in g.node:
            names.update(x for x in n.input if x)
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sub in a.graphs:
                        walk(sub)

    for a in node.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            walk(a.g)
        elif a.type == onnx.AttributeProto.GRAPHS:
            for sub in a.graphs:
                walk(sub)
    return names


def _clean_initializers(graph: onnx.GraphProto) -> None:
    """Drop the initializers (and the graph inputs of the same name) that nothing
    reads: ONNX Runtime's ``ONNXModel.clean_initializers``."""
    used = {x for n in graph.node for x in n.input if x}
    for n in graph.node:
        used |= _graph_attr_names(n)
    used |= {o.name for o in graph.output if o.name}
    drop = {t.name for t in graph.initializer if t.name not in used}
    if not drop:
        return
    keep = []
    for t in graph.initializer:
        c = onnx.TensorProto()
        c.CopyFrom(t)
        if t.name not in drop:
            keep.append(c)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    inputs = []
    for i in graph.input:
        c = onnx.ValueInfoProto()
        c.CopyFrom(i)
        if i.name not in drop:
            inputs.append(c)
    del graph.input[:]
    graph.input.extend(inputs)


class _View:
    """The lookups of ONNX Runtime's ``OnnxModel`` over a standalone copy of the
    top-level nodes, computed once per pass (as ONNX Runtime's ``Fusion.apply``
    does: a fusion never sees what an earlier one of the same pass wrote)."""

    def __init__(self, model: onnx.ModelProto) -> None:
        g = model.graph
        self.nodes: List[onnx.NodeProto] = _standalone(g.node)
        self.graph_outputs = {o.name for o in g.output}
        self.consumers: Dict[str, List[onnx.NodeProto]] = {}
        self.producer: Dict[str, onnx.NodeProto] = {}
        # (a ``Constant`` node's tensor, else an initializer, by name; read into numpy
        # only when a matcher asks, as a model's weights are not)
        self._tensors: Dict[str, onnx.TensorProto] = {}
        self._values: Dict[str, Optional[np.ndarray]] = {}
        for n in self.nodes:
            for x in n.input:
                if x:
                    self.consumers.setdefault(x, []).append(n)
            for o in n.output:
                if o:
                    self.producer[o] = n
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self._tensors.setdefault(n.output[0], a.t)
        for t in g.initializer:
            self._tensors.setdefault(t.name, t)

    # -- navigation -------------------------------------------------------------

    def children(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        out: List[onnx.NodeProto] = []
        for o in node.output:
            out.extend(self.consumers.get(o, ()))
        return out

    def parent(self, node: onnx.NodeProto, i: int) -> Optional[onnx.NodeProto]:
        if len(node.input) <= i:
            return None
        return self.producer.get(node.input[i])

    def match_parent(
        self, node: onnx.NodeProto, op_type: str, i: int
    ) -> Optional[onnx.NodeProto]:
        p = self.parent(node, i)
        return p if p is not None and p.op_type == op_type else None

    def match_parent_path(
        self, node: onnx.NodeProto, op_types: Sequence[str], indices: Sequence[int]
    ) -> Optional[List[onnx.NodeProto]]:
        out: List[onnx.NodeProto] = []
        cur = node
        for op_type, i in zip(op_types, indices):
            cur_p = self.match_parent(cur, op_type, i)
            if cur_p is None:
                return None
            out.append(cur_p)
            cur = cur_p
        return out

    def first_child_of_type(
        self, node: onnx.NodeProto, op_type: str
    ) -> Optional[onnx.NodeProto]:
        # (ONNX Runtime pops its work list from the right: the last child wins)
        for c in reversed(self.children(node)):
            if c.op_type == op_type:
                return c
        return None

    def match_child_path(
        self, node: onnx.NodeProto, op_types: Sequence[str]
    ) -> Optional[List[onnx.NodeProto]]:
        out: List[onnx.NodeProto] = []
        cur = node
        for op_type in op_types:
            nxt = next((c for c in self.children(cur) if c.op_type == op_type), None)
            if nxt is None:
                return None
            out.append(nxt)
            cur = nxt
        return out

    # -- constants ----------------------------------------------------------------

    def _value(self, name: str) -> Optional[np.ndarray]:
        if name not in self._values:
            t = self._tensors.get(name)
            try:
                self._values[name] = None if t is None else numpy_helper.to_array(t)
            except Exception:  # e.g. external data that is not loaded
                self._values[name] = None
        return self._values[name]

    def constant_input(
        self, node: onnx.NodeProto
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """The first input that is a ``Constant`` node's output or an initializer."""
        for i, x in enumerate(node.input):
            if x in self._tensors:
                v = self._value(x)
                return (i, v) if v is not None else (None, None)
        return None, None

    def find_constant_input(
        self, node: onnx.NodeProto, expected: float, delta: float = 1e-6
    ) -> int:
        i, v = self.constant_input(node)
        # (in the constant's own dtype, as ONNX Runtime does: a float32 value
        # on the edge of ``delta`` is on one side of it only in float32)
        if v is not None and v.size == 1 and bool(np.all(np.abs(v - expected) < delta)):
            return i if i is not None else -1
        return -1

    def has_constant_input(
        self, node: onnx.NodeProto, expected: float, delta: float = 1e-6
    ) -> bool:
        return self.find_constant_input(node, expected, delta) >= 0

    def is_vector_constant(self, name: str) -> bool:
        t = self._tensors.get(name)
        return t is not None and len(t.dims) == 1

    # -- safety -------------------------------------------------------------------

    def safe_to_fuse(
        self, remove: Sequence[onnx.NodeProto], keep_outputs: Sequence[str]
    ) -> bool:
        """Every output of the removed nodes (but ``keep_outputs``) is read by
        removed nodes only, and none of them is a graph output."""
        ids = {id(n) for n in remove}
        for n in remove:
            for o in n.output:
                if o in keep_outputs:
                    continue
                if o in self.graph_outputs:
                    return False
                for c in self.consumers.get(o, ()):
                    if id(c) not in ids:
                        return False
        return True


def _new_name(
    nodes: Sequence[onnx.NodeProto], counters: Dict[str, int], prefix: str
) -> str:
    """ONNX Runtime's ``create_node_name``: ``<prefix>_<n>`` with ``n`` one past the
    highest suffix an existing node of that prefix has."""
    prefix = prefix if prefix.endswith("_") else prefix + "_"
    if prefix in counters:
        counters[prefix] += 1
    else:
        suffix = 0
        for n in nodes:
            if n.name and n.name.startswith(prefix):
                try:
                    suffix = max(int(n.name[len(prefix) :]) + 1, suffix)
                except ValueError:
                    continue
        counters[prefix] = suffix
    return prefix + str(counters[prefix])


def _finish_transformers_pass(
    model: onnx.ModelProto,
    view: _View,
    remove: Sequence[onnx.NodeProto],
    add: Sequence[onnx.NodeProto],
) -> None:
    """ONNX Runtime's ``Fusion.apply`` tail: remove, append the new nodes at the end
    of the graph and, when anything changed, ``update_graph`` (initializers nothing
    reads and ``Constant`` nodes nothing reads go)."""
    if not remove and not add:
        return
    gone = {id(n) for n in remove}
    nodes = [n for n in view.nodes if id(n) not in gone] + list(add)
    g = model.graph
    _set_nodes(g, nodes)
    read: Set[str] = set()
    for n in g.node:
        if n.op_type != "Constant":
            read.update(x for x in n.input if x)
        read |= _graph_attr_names(n)
    outs = {o.name for o in g.output}
    keep = []
    for t in g.initializer:
        if t.name in read or t.name in outs:
            c = onnx.TensorProto()
            c.CopyFrom(t)
            keep.append(c)
    del g.initializer[:]
    g.initializer.extend(keep)
    # (the second read set: what is left after the nodes above are gone)
    still = {x for n in g.node for x in n.input if x} | outs
    for n in g.node:
        still |= _graph_attr_names(n)
    unused = [n for n in g.node if n.op_type == "Constant" and n.output[0] not in still]
    if unused:
        ids = {id(n) for n in unused}
        _set_nodes(g, [n for n in g.node if id(n) not in ids])


# -- ONNX Runtime's LayerNormalization fusion ----------------------------------------


def fuse_layer_norm(
    model: onnx.ModelProto, style: str = "quark", _inplace: bool = False
) -> onnx.ModelProto:
    """``ReduceMean`` / ``Sub`` / ``Pow`` / ``ReduceMean`` / ``Add`` / ``Sqrt`` /
    ``Div`` / ``Mul`` / ``Add`` -> ``LayerNormalization`` (Quark: ONNX Runtime's
    ``FusionLayerNormalization``). Needs opset >= 17 (the caller checks).

    The pattern starts at each ``ReduceMean``: its consumers are one or two ``Sub``
    reading the same input (the second is the duplicate older torch exports leave),
    one of them -- or a ``Cast`` after it -- feeds a ``Div`` whose denominator is
    ``Sqrt(ReduceMean(Pow(Sub, 2)) + eps)`` (``Cast`` allowed between the ``Pow`` and
    the ``Sub``), with ``0 < eps <= 1e-4``; the ``Div`` feeds a ``Mul`` by a 1-D
    constant and that an ``Add`` of a 1-D constant (a ``Cast`` between ``Div`` and
    ``Mul`` is removed as well). The intermediate tensors must have no other reader.
    """
    out = model if _inplace else _copy(model)
    v = _View(out)
    remove: List[onnx.NodeProto] = []
    add: List[onnx.NodeProto] = []
    counters: Dict[str, int] = {}
    for node in v.nodes:
        if node.op_type != "ReduceMean":
            continue
        children = v.children(node)
        if not 1 <= len(children) <= 2:
            continue
        root = node.input[0]
        if children[0].op_type != "Sub" or children[0].input[0] != root:
            continue
        if len(children) == 2 and (
            children[1].op_type != "Sub" or children[1].input[0] != root
        ):
            continue
        div: Optional[onnx.NodeProto] = None
        for child in children:
            d = v.first_child_of_type(child, "Div")
            if d is not None:
                div = d
                break
            path = v.match_child_path(child, ["Cast", "Div"])
            if path is not None:
                div = path[-1]
                break
        if div is None:
            continue
        parents = None
        # (ONNX Runtime's own fusion also takes ``eps + variance``)
        for k in (0, 1) if style == "ort" else (0,):
            parents = v.match_parent_path(
                div, ["Sqrt", "Add", "ReduceMean", "Pow", "Sub"], [1, 0, k, 0, 0]
            ) or v.match_parent_path(
                div,
                ["Sqrt", "Add", "ReduceMean", "Pow", "Cast", "Sub"],
                [1, 0, k, 0, 0, 0],
            )
            if parents is not None:
                break
        if parents is None:
            continue
        sub_node = parents[-1]
        if not any(sub_node is c for c in children):
            continue
        add_eps = parents[1]
        _, eps = v.constant_input(add_eps)
        if eps is None or eps.size != 1:
            continue
        # (Quark: 0 < eps <= 1e-4; ONNX Runtime's own fusion takes any positive eps)
        if bool(np.all(eps <= 0)) or (style != "ort" and bool(np.all(eps > 1.0e-4))):
            continue
        epsilon = float(eps.reshape(-1)[0])
        if v.find_constant_input(parents[3], 2.0) != 1:
            continue
        if div.output[0] not in v.consumers:
            continue
        subgraph: List[onnx.NodeProto] = []
        for temp in v.consumers[div.output[0]]:
            if temp.op_type == "Cast":
                subgraph.append(temp)
                if temp.output[0] not in v.consumers:
                    continue
                mul = v.consumers[temp.output[0]][0]
            else:
                mul = temp
            if mul.op_type != "Mul" or mul.output[0] not in v.consumers:
                continue
            last_add = v.consumers[mul.output[0]][0]
            if last_add.op_type != "Add":
                continue
            subgraph.append(node)
            subgraph.extend(children)
            subgraph.extend(parents[:-1])
            subgraph.extend([last_add, mul, div])
            before = div if temp.op_type != "Cast" else temp
            try:
                weight = mul.input[1 - list(mul.input).index(before.output[0])]
                bias = last_add.input[1 - list(last_add.input).index(mul.output[0])]
            except (ValueError, IndexError):
                continue
            if not v.is_vector_constant(weight) or not v.is_vector_constant(bias):
                continue
            if not v.safe_to_fuse(subgraph, last_add.output):
                continue
            remove.extend(subgraph)
            if style == "ort":
                # (ONNX Runtime names it after the scale ``Mul``; an unnamed one is
                # named after its output so that the names stay unique)
                name = (mul.name or mul.output[0]) + "/LayerNormFusion/"
                attrs = {"stash_type": 1, "axis": -1, "epsilon": epsilon}
            else:
                name = _new_name(v.nodes, counters, "LayerNorm")
                attrs = {"epsilon": epsilon}
            add.append(
                helper.make_node(
                    "LayerNormalization",
                    [node.input[0], weight, bias],
                    [last_add.output[0]],
                    name=name,
                    **attrs,
                )
            )
    _finish_transformers_pass(out, v, remove, add)
    return out


# -- ONNX Runtime's Gelu fusion ------------------------------------------------------


def _only_child(
    v: _View, node: onnx.NodeProto, op_type: str
) -> Optional[onnx.NodeProto]:
    kids = v.consumers.get(node.output[0])
    if kids is None or len(kids) != 1 or kids[0].op_type != op_type:
        return None
    return kids[0]


def _gelu_erf_chain(
    v: _View, erf: onnx.NodeProto
) -> Tuple[Optional[onnx.NodeProto], Optional[onnx.NodeProto]]:
    """``erf`` -> ``Add`` (+1) -> ``Mul``: the two nodes after the ``Erf``."""
    add = _only_child(v, erf, "Add")
    if add is None or not v.has_constant_input(add, 1):
        return None, None
    return add, _only_child(v, add, "Mul")


def _fuse_gelu_pytorch(
    v: _View, erf: onnx.NodeProto
) -> Optional[Tuple[List[onnx.NodeProto], str, str]]:
    # x -> Div(sqrt 2) -> Erf -> Add(1) -> Mul(x or x * 0.5) [-> Mul(0.5)]
    add, mul_after = _gelu_erf_chain(v, erf)
    if add is None or mul_after is None:
        return None
    div = v.match_parent(erf, "Div", 0)
    if div is None or v.find_constant_input(div, 1.4142, 0.001) != 1:
        return None
    src = div.input[0]
    another = 1 if mul_after.input[0] == add.output[0] else 0
    if src == mul_after.input[another]:
        mul_half = _only_child(v, mul_after, "Mul")
        if mul_half is None or not v.has_constant_input(mul_half, 0.5):
            return None
        dst = mul_half.output[0]
    else:
        mul_half = v.match_parent(mul_after, "Mul", another)
        if mul_half is None or not v.has_constant_input(mul_half, 0.5):
            return None
        if src not in mul_half.input:
            return None
        dst = mul_after.output[0]
    return [div, erf, add, mul_after, mul_half], src, dst


def _fuse_gelu_keras(
    v: _View, erf: onnx.NodeProto
) -> Optional[Tuple[List[onnx.NodeProto], str, str]]:
    # root -> Div(sqrt 2 or Sqrt(2)) -> Erf -> Add(1) -> Mul(0.5) -> Mul(root)
    add, mul_after = _gelu_erf_chain(v, erf)
    if add is None or mul_after is None or not v.has_constant_input(mul_after, 0.5):
        return None
    mul = _only_child(v, mul_after, "Mul")
    if mul is None:
        return None
    div = v.match_parent(erf, "Div", 0)
    if div is None:
        return None
    sqrt = None
    if v.find_constant_input(div, 1.4142, 0.001) != 1:
        sqrt = v.match_parent(div, "Sqrt", 1)
        if sqrt is None or not v.has_constant_input(sqrt, 2.0):
            return None
    root = v.parent(div, 0)
    if root is None or root.output[0] not in mul.input:
        return None
    nodes = [div, erf, add, mul_after, mul] + ([sqrt] if sqrt is not None else [])
    return nodes, root.output[0], mul.output[0]


def _fuse_gelu_tf(
    v: _View, erf: onnx.NodeProto
) -> Optional[Tuple[List[onnx.NodeProto], str, str]]:
    # root -> Mul(1/sqrt 2) -> Erf -> Add(1) -> Mul(0.5) -> Mul(root)
    add = _only_child(v, erf, "Add")
    if add is None or not v.has_constant_input(add, 1):
        return None
    mul_half = _only_child(v, add, "Mul")
    if mul_half is None or not v.has_constant_input(mul_half, 0.5):
        return None
    first = v.match_parent(erf, "Mul", 0)
    if first is None:
        return None
    i = v.find_constant_input(first, 0.7071067690849304, 0.001)
    if i < 0:
        return None
    root = v.parent(first, 0 if i == 1 else 1)
    if root is None:
        return None
    last = _only_child(v, mul_half, "Mul")
    if last is None or root.output[0] not in last.input:
        return None
    return [first, erf, add, mul_half, last], root.output[0], last.output[0]


def fuse_gelu(
    model: onnx.ModelProto, style: str = "quark", _inplace: bool = False
) -> onnx.ModelProto:
    """The ``Erf`` form of Gelu -> one ``Gelu`` (Quark: ONNX Runtime's ``FusionGelu``).
    Needs opset >= 20 (the caller checks). Three shapes, each from an ``Erf``:

    - PyTorch: ``x / sqrt(2)`` -> ``Erf`` -> ``+ 1`` -> ``Mul`` by ``x`` and by 0.5
      (before, ``Mul(x, 0.5)`` feeding the last ``Mul``, or after it);
    - Keras: ``x / sqrt(2)`` (or ``/ Sqrt(2)``) -> ``Erf`` -> ``+ 1`` -> ``* 0.5`` -> ``* x``;
    - TensorFlow: ``x * 0.7071`` -> ``Erf`` -> ``+ 1`` -> ``* 0.5`` -> ``* x``;

    the last two want ``x`` to be the output of a node (not a graph input). The tanh
    form is not matched.
    """
    out = model if _inplace else _copy(model)
    v = _View(out)
    remove: List[onnx.NodeProto] = []
    add: List[onnx.NodeProto] = []
    counters: Dict[str, int] = {}
    for erf in v.nodes:
        if erf.op_type != "Erf":
            continue
        found = _fuse_gelu_pytorch(v, erf)
        if found is None and style != "ort":
            # (ONNX Runtime's own fusion knows the PyTorch shape only)
            found = _fuse_gelu_keras(v, erf) or _fuse_gelu_tf(v, erf)
        if found is None:
            continue
        nodes, src, dst = found
        if not v.safe_to_fuse(nodes, [dst]):
            continue
        remove.extend(nodes)
        if style == "ort":
            fused = helper.make_node(
                "Gelu",
                [src],
                [dst],
                name=_new_name(v.nodes, counters, "Gelu"),
                approximate="none",
            )
        else:
            fused = helper.make_node(
                "Gelu", [src], [dst], name=_new_name(v.nodes, counters, "Gelu")
            )
            fused.domain = "com.microsoft"
        add.append(fused)
    _finish_transformers_pass(out, v, remove, add)
    if (
        add
        and style != "ort"
        and not any(o.domain == "com.microsoft" for o in out.opset_import)
    ):
        # (the contrib op needs its domain imported; Quark's quantizer adds it)
        out.opset_import.add(domain="com.microsoft", version=1)
    return out


# -- Quark's own matchers --------------------------------------------------------------


def _sort(model: onnx.ModelProto) -> None:
    from onnxsim.quark_marking import quark_sort_inplace

    quark_sort_inplace(model)


def fuse_instance_norm(
    model: onnx.ModelProto, _inplace: bool = False
) -> onnx.ModelProto:
    """The TensorFlow-style InstanceNorm -> ``InstanceNormalization`` (Quark's
    ``Optimizer.fuse_instance_norm``). The pattern, from the final ``Add``::

        mean = GlobalAveragePool(x)          d = x - mean            (Sub)
        var  = GlobalAveragePool(d * d)      (Mul)
        s    = Reciprocal(Sqrt(var + eps)) * scale                   (Add, Sqrt, Reciprocal, Mul)
        y    = (x * s) + (bias - mean * s)                           (Mul, Mul, Sub, Add)

    The ``scale`` and ``bias`` initializers (``[1, C, 1, 1]``) become ``[C]`` in
    place; ``epsilon`` is the ``eps`` initializer's value. Quark checks the op types of
    the chain and where the ``Sub``'s second input, the ``GlobalAveragePool`` nodes and
    the scale / bias / eps come from -- not what the ``Mul`` feeding the final ``Add``
    reads -- and the model is topologically sorted afterwards, matched or not.
    """
    out = model if _inplace else _copy(model)
    g = out.graph
    nodes = _standalone(g.node)
    prod: Dict[str, onnx.NodeProto] = {o: n for n in nodes for o in n.output}
    inits = {t.name: t for t in g.initializer}
    graph_outputs = {o.name for o in g.output}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    remove: List[onnx.NodeProto] = []
    remove_inits: Set[str] = set()
    add: List[onnx.NodeProto] = []
    for node in nodes:
        if node.op_type != "Add":
            continue
        try:
            a0 = prod[node.input[0]]
            a1 = prod[node.input[1]]
            if not (a0.op_type == "Mul" and a1.op_type == "Sub"):
                continue
            sub0 = a1
            mul0 = prod[sub0.input[1]]
            if mul0.op_type != "Mul":
                continue
            gap_mean = prod[mul0.input[0]]
            mul1 = prod[mul0.input[1]]
            if not (gap_mean.op_type == "GlobalAveragePool" and mul1.op_type == "Mul"):
                continue
            rec = prod[mul1.input[0]]
            if rec.op_type != "Reciprocal":
                continue
            sqrt = prod[rec.input[0]]
            if sqrt.op_type != "Sqrt":
                continue
            add1 = prod[sqrt.input[0]]
            if add1.op_type != "Add":
                continue
            gap_var = prod[add1.input[0]]
            if gap_var.op_type != "GlobalAveragePool":
                continue
            mul2 = prod[gap_var.input[0]]
            if mul2.op_type != "Mul":
                continue
            sub1 = prod[mul2.input[0]]
            if sub1.op_type != "Sub":
                continue
            if prod[sub1.input[1]].op_type != "GlobalAveragePool":
                continue
            chain = [
                node,
                a0,
                a1,
                mul0,
                gap_mean,
                mul1,
                rec,
                sqrt,
                add1,
                gap_var,
                mul2,
                sub1,
            ]
            # (Quark would write a graph with a missing tensor)
            ids = {id(n) for n in chain}
            if any(
                o in graph_outputs
                or any(id(c) not in ids for c in consumers.get(o, ()))
                for n in chain
                if n is not node
                for o in n.output
            ):
                continue
            bias = inits[sub0.input[0]]
            bias.dims[:] = [bias.dims[1]]
            weight = inits[mul1.input[1]]
            weight.dims[:] = [weight.dims[1]]
            eps = inits[add1.input[1]]
            eps_value = float(numpy_helper.to_array(eps).item())
            add.append(
                helper.make_node(
                    "InstanceNormalization",
                    [sub1.input[0], mul1.input[1], sub0.input[0]],
                    list(node.output),
                    node.name,
                    epsilon=eps_value,
                )
            )
            remove.extend(chain)
            remove_inits.add(eps.name)
        except (KeyError, IndexError, ValueError):
            continue
    gone = {id(n) for n in remove}
    _set_nodes(g, [n for n in nodes if id(n) not in gone] + add)
    if remove_inits:
        keep = []
        for t in g.initializer:
            c = onnx.TensorProto()
            c.CopyFrom(t)
            if t.name not in remove_inits:
                keep.append(c)
        del g.initializer[:]
        g.initializer.extend(keep)
    _clean_initializers(g)
    _sort(out)
    return out


def fuse_l2_norm(model: onnx.ModelProto, _inplace: bool = False) -> onnx.ModelProto:
    """The L2 normalization built from ``ReduceSum`` -> ``LpNormalization`` (Quark's
    ``Optimizer.fuse_l2_norm``). From the final ``Mul``::

        y = Unsqueeze(..) * Reciprocal(Sqrt(Max(ReduceSum(Mul(Unsqueeze(x), ..)), eps)))

    becomes ``LpNormalization(Unsqueeze(x), p=2)`` -- with the default ``axis=-1``
    whatever the ``ReduceSum`` reduced over (Quark does not look). The model is
    topologically sorted afterwards, matched or not.
    """
    out = model if _inplace else _copy(model)
    g = out.graph
    nodes = _standalone(g.node)
    prod: Dict[str, onnx.NodeProto] = {o: n for n in nodes for o in n.output}
    graph_outputs = {o.name for o in g.output}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    inits = {t.name for t in g.initializer}
    remove: List[onnx.NodeProto] = []
    remove_inits: Set[str] = set()
    add: List[onnx.NodeProto] = []
    for node in nodes:
        if node.op_type != "Mul":
            continue
        try:
            if not (
                prod[node.input[0]].op_type == "Unsqueeze"
                and prod[node.input[1]].op_type == "Reciprocal"
            ):
                continue
            rec = prod[node.input[1]]
            sqrt = prod[rec.input[0]]
            if sqrt.op_type != "Sqrt":
                continue
            mx = prod[sqrt.input[0]]
            if mx.op_type != "Max":
                continue
            red = prod[mx.input[0]]
            if red.op_type != "ReduceSum":
                continue
            sq = prod[red.input[0]]
            if sq.op_type != "Mul":
                continue
            uns = prod[sq.input[0]]
            if uns.op_type != "Unsqueeze":
                continue
            chain = [node, rec, sqrt, mx, red, sq]
            ids = {id(n) for n in chain}
            if any(
                o in graph_outputs
                or any(id(c) not in ids for c in consumers.get(o, ()))
                for n in chain
                if n is not node
                for o in n.output
            ):
                continue
            remove.extend(chain)
            if mx.input[1] in inits:
                remove_inits.add(mx.input[1])
            add.append(
                helper.make_node(
                    "LpNormalization", [uns.output[0]], [node.output[0]], node.name, p=2
                )
            )
        except (KeyError, IndexError):
            continue
    gone = {id(n) for n in remove}
    _set_nodes(g, [n for n in nodes if id(n) not in gone] + add)
    if remove_inits:
        keep = []
        for t in g.initializer:
            c = onnx.TensorProto()
            c.CopyFrom(t)
            if t.name not in remove_inits:
                keep.append(c)
        del g.initializer[:]
        g.initializer.extend(keep)
    _clean_initializers(g)
    _sort(out)
    return out


def apply_fusions(
    model: onnx.ModelProto,
    instance_norm: bool = True,
    l2_norm: bool = True,
    layer_norm: bool = True,
    gelu: bool = True,
    style: str = "quark",
) -> onnx.ModelProto:
    """The four fusions in Quark's order (``style="ort"``: the fused nodes ONNX
    Runtime's own fusion writes, for a caller standing in for it). A copy is returned
    (the input itself when no pass applies)."""
    opset = _opset(model)
    run_ln = layer_norm and opset is not None and opset >= LAYER_NORM_MIN_OPSET
    run_gelu = gelu and opset is not None and opset >= GELU_MIN_OPSET
    if not (instance_norm or l2_norm or run_ln or run_gelu):
        return model
    out = _copy(model)
    if instance_norm:
        fuse_instance_norm(out, _inplace=True)
    if l2_norm:
        fuse_l2_norm(out, _inplace=True)
    if run_ln:
        fuse_layer_norm(out, style, _inplace=True)
    if run_gelu:
        fuse_gelu(out, style, _inplace=True)
    return out
