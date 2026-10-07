"""Exact structural pruning from provably stable ReLUs, over a declared input box.

Background (verification-aware-training literature, from memory, not re-checked here): given
an input box, a ReLU unit whose pre-activation *upper* bound is ``<= 0`` is **provably dead**
-- it outputs exactly 0 for every input in the box -- and a unit whose *lower* bound is
``>= 0`` is **provably always-on** -- the ReLU is the identity there. Tighter bounds
(CROWN, zonotopes) prove more units stable than plain intervals.

* A dead unit can be **removed** with zero output change inside the box: its producer
  column / output channel (and BatchNorm parameters) is dropped, and so is the matching
  input row / input channel of the consumer.
* A layer whose remaining units are *all* always-on has no nonlinearity left: with
  ``merge_active=True`` the two surrounding affine layers (MatMul/Gemm chains) are fused
  into one, ``W = W1 @ W2``, ``b = b1 @ W2 (+ b2)``.

``analyze`` reports how many units each bound provider proves dead / always-on, and what
removing them would save; ``apply`` performs the transformation and records the box it is
exact for in the output model.

THE RESULT IS EXACT ONLY INSIDE THE DECLARED BOX. Outside it the pruned model may differ
arbitrarily (a pruned unit might fire; a merged layer might need its ReLU). ``apply`` is
opt-in, refuses to run without a finite input range, and stores the precondition in
``metadata_props``: ``onnxsim.precondition.range.<input>`` (same JSON as
``onnxsim.range.*``) plus ``onnxsim.precondition.note``. A box derived from data (for
example per-pixel min/max of a training set) is a *statement about that data*, not a
certificate for all inputs: use it only if you accept that.

Float32 note: the bounds are real-arithmetic enclosures; float32 execution can differ from
them by rounding (~1e-6 relative). A unit is therefore called stable only with a margin
(``margin``, default 1e-5 in pre-activation units): ``hi <= -margin`` (dead) or
``lo >= margin`` (on). A removed unit that float32 would have fired by <= margin changes
the output by at most ``margin * |consumer weight|``; ``apply`` checks the pruned model
against the original on sampled inputs in the box and raises if they disagree beyond float
rounding.

Supported structure (anything else is left alone, with the reason in the report):

* pre-activation chain: ``MatMul(x, W) [+ Add(bias)]``, ``Gemm``, ``Conv`` (group = 1), each
  optionally followed by ``BatchNormalization`` (inference) / ``Add`` / ``Mul`` by a constant
  per-unit vector, then ``Relu``;
* post-activation chain: the ReLU output has exactly one consumer path through
  channel-preserving ops (``MaxPool``, ``AveragePool``, ``GlobalAveragePool``, ``Identity``,
  ``Dropout``; ``Flatten`` between a conv and a MatMul/Gemm) ending at a ``MatMul`` / ``Gemm``
  / ``Conv`` (group = 1) with a constant weight;
* the weights are initializers used by exactly one node.

Residual joins (the ReLU output also feeds an ``Add``), depthwise/grouped convs, ReLU6/Clip,
graph outputs and shared weights are reported as ``unprunable`` (their stable units are still
counted by ``analyze``).
"""

import copy
import dataclasses
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from . import interval as _interval
from . import ranges as _ranges

__all__ = ["LayerReport", "PruneReport", "analyze", "apply", "prune_units"]

DEFAULT_MARGIN = 1e-5
_PASS_THROUGH_CHANNEL = ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool")
_PASS_THROUGH_ANY = ("Identity", "Dropout")


# --------------------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class LayerReport:
    """Stability of one ReLU layer (units = neurons of an MLP layer / channels of a conv)."""

    relu: str
    pre_activation: str
    units: int
    dead: int  # proven dead by at least one bound provider
    always_on: int
    unstable: int
    by_method: Dict[str, Tuple[int, int]]  # method -> (dead, always_on)
    prunable: bool  # structure supported (see module doc)
    reason: str  # why not prunable ("" when prunable)
    params_removable: int = 0  # parameters freed by removing the dead units
    macs_removable: int = 0  # multiply-adds per sample freed by removing them
    dead_units: Optional[np.ndarray] = None  # indices, for apply()
    on_units: Optional[np.ndarray] = None
    note: str = ""


@dataclasses.dataclass
class PruneReport:
    layers: List[LayerReport]
    methods: List[str]  # bound providers that actually ran
    skipped_methods: Dict[str, str]  # provider -> reason it did not run
    precondition: Dict[str, Tuple[np.ndarray, np.ndarray]]
    applied: bool = False
    removed_units: int = 0
    # Combined savings of removing every dead unit of every prunable layer at once, computed
    # exactly from the pruned weight shapes. The per-layer figures are *standalone* (that layer
    # alone) and overlap where two adjacent layers both prune the weight between them, so they
    # do not add up to these.
    params_removed: int = 0
    macs_removed: int = 0
    merged_layers: int = 0
    verified_max_abs_diff: Optional[float] = None
    verified_samples: int = 0
    notes: List[str] = dataclasses.field(default_factory=list)

    @property
    def total_units(self) -> int:
        return sum(lay.units for lay in self.layers)

    def total(self, kind: str) -> int:
        return sum(getattr(lay, kind) for lay in self.layers)

    def total_by_method(self) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, Tuple[int, int]] = {}
        for m in self.methods:
            out[m] = (
                sum(lay.by_method.get(m, (0, 0))[0] for lay in self.layers),
                sum(lay.by_method.get(m, (0, 0))[1] for lay in self.layers),
            )
        return out

    def __str__(self) -> str:
        rows = [
            f"{'relu':44s} {'units':>6s} {'dead':>6s} {'on':>6s} {'unst.':>6s} {'prune':>5s} {'params':>9s} {'MACs':>11s}"
        ]
        for lay in self.layers:
            rows.append(
                f"{lay.relu[-44:]:44s} {lay.units:6d} {lay.dead:6d} {lay.always_on:6d} {lay.unstable:6d} "
                f"{'yes' if lay.prunable else 'no':>5s} {lay.params_removable:9d} {lay.macs_removable:11d}"
                + ("" if lay.prunable else f"  [{lay.reason}]")
            )
        tot = (
            f"{'total':44s} {self.total_units:6d} {self.total('dead'):6d} "
            f"{self.total('always_on'):6d} {self.total('unstable'):6d} {'':5s} "
            f"{self.total('params_removable'):9d} {self.total('macs_removable'):11d}"
        )
        rows.append(tot)
        rows.append(
            f"  all prunable dead units together: {self.params_removed} parameters, "
            f"{self.macs_removed} MACs per sample (exact; per-layer figures are standalone)"
        )
        for m, (d, o) in self.total_by_method().items():
            rows.append(f"  {m}: {d} dead, {o} always-on")
        for m, why in self.skipped_methods.items():
            rows.append(f"  {m}: skipped ({why})")
        return "\n".join(rows)


# --------------------------------------------------------------------------------------
# graph helpers
# --------------------------------------------------------------------------------------


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


class _Graph:
    """Producer/consumer maps, initializers and static shapes of one model."""

    def __init__(self, model: onnx.ModelProto, shapes: Dict[str, Tuple[int, ...]]):
        g = model.graph
        self.model = model
        self.init: Dict[str, np.ndarray] = {
            t.name: numpy_helper.to_array(t) for t in g.initializer
        }
        self.shapes = shapes
        self.producer: Dict[str, onnx.NodeProto] = {}
        self.consumers: Dict[str, List[onnx.NodeProto]] = {}
        self.uses: Dict[str, int] = {}
        for n in g.node:
            for o in n.output:
                self.producer[o] = n
            for i in n.input:
                if i:
                    self.consumers.setdefault(i, []).append(n)
                    self.uses[i] = self.uses.get(i, 0) + 1
        self.graph_outputs = {o.name for o in g.output}

    def is_weight(self, name: str) -> bool:
        return name in self.init and self.uses.get(name, 0) == 1

    def single_consumer(self, tensor: str) -> Optional[onnx.NodeProto]:
        c = self.consumers.get(tensor, [])
        if len(c) == 1 and tensor not in self.graph_outputs:
            return c[0]
        return None


@dataclasses.dataclass
class _Edit:
    name: str  # initializer
    axis: int
    expand: int = (
        1  # each unit owns `expand` consecutive entries (Flatten after a conv)
    )


@dataclasses.dataclass
class _Chain:
    relu: onnx.NodeProto
    pre_nodes: List[onnx.NodeProto]  # input -> relu order (relu excluded)
    post_nodes: List[onnx.NodeProto]  # relu output -> sink (sink last)
    units: int
    pre_edits: List[_Edit]
    post_edits: List[_Edit]
    kind: str  # "last" | "channel"
    macs_per_unit: int = 0
    params_per_unit: int = 0


def _relu_name(n: onnx.NodeProto) -> str:
    return n.name or n.output[0]


def _unit_axis_for_const(
    const: np.ndarray, out_rank: int, unit_axis: int, units: int
) -> Tuple[Optional[int], str]:
    """Axis of ``const`` that indexes the units (``None`` if it is a broadcast scalar)."""
    if const.size == 1:
        return None, ""
    if const.ndim > out_rank:
        return None, "constant has more dims than the tensor it joins"
    offset = out_rank - const.ndim
    axes = [a for a, d in enumerate(const.shape) if d != 1]
    if len(axes) != 1 or const.shape[axes[0]] != units or axes[0] + offset != unit_axis:
        return None, "per-unit constant is not aligned with the unit axis"
    return axes[0], ""


def _walk_chain(gr: _Graph, relu: onnx.NodeProto) -> Tuple[Optional[_Chain], str]:
    """Locate the producer and consumer structure around ``relu`` (see module doc)."""
    t = relu.input[0]
    in_shape = gr.shapes.get(t)
    if in_shape is None:
        return None, "unknown shape"
    rank = len(in_shape)
    pre_nodes: List[onnx.NodeProto] = []
    pre_edits: List[_Edit] = []
    kind = ""
    units = 0
    cur = relu
    # ---- pre-activation chain, walking backwards
    while True:
        p = gr.producer.get(t)
        if p is None:
            return None, "pre-activation is a graph input or initializer"
        nxt = gr.single_consumer(t)
        if nxt is None or nxt.output[0] != cur.output[0]:
            return None, f"{p.op_type} output feeds more than one consumer"
        pre_nodes.insert(0, p)
        op = p.op_type
        if op == "BatchNormalization":
            if _attr(p, "training_mode", 0):
                return None, "BatchNormalization in training mode"
            if len(p.output) > 1 and any(p.output[1:]):
                return None, "BatchNormalization training outputs"
            if not all(gr.is_weight(i) for i in p.input[1:5]):
                return (
                    None,
                    "BatchNormalization parameters are not unshared initializers",
                )
            u = gr.init[p.input[1]].shape[0]
            if units and u != units:
                return None, "inconsistent unit count"
            units = u
            pre_edits += [_Edit(i, 0) for i in p.input[1:5]]
            kind = kind or "channel"
            t, cur = p.input[0], p
            continue
        if op in ("Add", "Mul"):
            const = [i for i in p.input if i in gr.init]
            var = [i for i in p.input if i not in gr.init]
            if len(const) != 1 or len(var) != 1:
                return None, f"{op} is not a tensor-with-constant op"
            if not gr.is_weight(const[0]):
                return None, f"{op} constant is shared"
            t, cur = var[0], p
            # alignment is checked once the unit axis is known (below)
            pre_edits.append(_Edit(const[0], -1))
            continue
        if op == "MatMul":
            if not gr.is_weight(p.input[1]) or gr.init[p.input[1]].ndim != 2:
                return None, "MatMul weight is not an unshared 2-D initializer"
            w = gr.init[p.input[1]]
            if units and w.shape[1] != units:
                return None, "inconsistent unit count"
            units = w.shape[1]
            pre_edits.append(_Edit(p.input[1], 1))
            kind = "last"
            break
        if op == "Gemm":
            if not gr.is_weight(p.input[1]):
                return None, "Gemm weight is not an unshared initializer"
            w = gr.init[p.input[1]]
            tb = int(_attr(p, "transB", 0))
            axis = 0 if tb else 1
            if units and w.shape[axis] != units:
                return None, "inconsistent unit count"
            units = w.shape[axis]
            pre_edits.append(_Edit(p.input[1], axis))
            if len(p.input) > 2 and p.input[2]:
                c = p.input[2]
                if not gr.is_weight(c):
                    return None, "Gemm bias is shared / not an initializer"
                pre_edits.append(_Edit(c, -1))
            kind = "last"
            break
        if op == "Conv":
            if int(_attr(p, "group", 1)) != 1:
                return None, "grouped / depthwise Conv"
            if not gr.is_weight(p.input[1]):
                return None, "Conv weight is not an unshared initializer"
            w = gr.init[p.input[1]]
            if units and w.shape[0] != units:
                return None, "inconsistent unit count"
            units = w.shape[0]
            pre_edits.append(_Edit(p.input[1], 0))
            if len(p.input) > 2 and p.input[2]:
                if not gr.is_weight(p.input[2]):
                    return None, "Conv bias is shared"
                pre_edits.append(_Edit(p.input[2], 0))
            kind = "channel"
            break
        return None, f"unsupported producer {op}"
    if (
        any(n.op_type == "BatchNormalization" for n in pre_nodes)
        and kind == "last"
        and rank != 2
    ):
        return (
            None,
            f"BatchNormalization on a rank-{rank} MatMul output (channel axis is not the last)",
        )
    if units != in_shape[1 if kind == "channel" else -1]:
        return None, "unit count does not match the pre-activation shape"
    unit_axis = 1 if kind == "channel" else rank - 1
    # Resolve the Add/Mul constants' unit axis now that the unit axis is known.
    resolved: List[_Edit] = []
    for e in pre_edits:
        if e.axis != -1:
            resolved.append(e)
            continue
        const = gr.init[e.name]
        # Gemm bias (C) broadcasts against [N, units]: align as an Add constant
        const_axis, why = _unit_axis_for_const(const, rank, unit_axis, units)
        if why:
            return None, why
        if const_axis is not None:
            resolved.append(_Edit(e.name, const_axis))
    pre_edits = resolved
    # ---- post-activation chain, walking forwards
    post_nodes: List[onnx.NodeProto] = []
    post_edits: List[_Edit] = []
    t = relu.output[0]
    expand = 1
    cur_kind = kind
    shape_here = gr.shapes.get(t)
    while True:
        c = gr.single_consumer(t)
        if c is None:
            n = len(gr.consumers.get(t, []))
            return (
                None,
                "ReLU output is a graph output"
                if t in gr.graph_outputs
                else f"ReLU output feeds {n} consumers (residual / fan-out)"
                if n > 1
                else "ReLU output is unused",
            )
        post_nodes.append(c)
        op = c.op_type
        if op in _PASS_THROUGH_ANY and c.input[0] == t:
            t = c.output[0]
            continue
        if op in _PASS_THROUGH_CHANNEL and c.input[0] == t:
            if cur_kind != "channel":
                return None, f"{op} after an MLP layer"
            t = c.output[0]
            shape_here = gr.shapes.get(t)
            continue
        if op == "Flatten" and c.input[0] == t:
            if cur_kind != "channel" or int(_attr(c, "axis", 1)) != 1:
                return None, "Flatten that is not a conv->vector flatten at axis 1"
            if shape_here is None or len(shape_here) != 4:
                return None, "unknown conv output shape before Flatten"
            expand = int(np.prod(shape_here[2:]))
            cur_kind = "last"
            t = c.output[0]
            continue
        if op == "MatMul" and c.input[0] == t:
            if not gr.is_weight(c.input[1]) or gr.init[c.input[1]].ndim != 2:
                return None, "consumer MatMul weight is not an unshared 2-D initializer"
            if cur_kind != "last":
                return None, "MatMul after a conv feature map without Flatten"
            post_edits.append(_Edit(c.input[1], 0, expand))
            break
        if op == "Gemm" and c.input[0] == t:
            if int(_attr(c, "transA", 0)):
                return None, "consumer Gemm has transA"
            if not gr.is_weight(c.input[1]) or cur_kind != "last":
                return None, "consumer Gemm weight is not an unshared initializer"
            post_edits.append(
                _Edit(c.input[1], 1 if int(_attr(c, "transB", 0)) else 0, expand)
            )
            break
        if op == "Conv" and c.input[0] == t:
            if int(_attr(c, "group", 1)) != 1 or cur_kind != "channel":
                return None, "grouped / depthwise consumer Conv"
            if not gr.is_weight(c.input[1]):
                return None, "consumer Conv weight is not an unshared initializer"
            post_edits.append(_Edit(c.input[1], 1, expand))
            break
        return None, f"unsupported consumer {op}"
    return (
        _Chain(relu, pre_nodes, post_nodes, units, pre_edits, post_edits, kind),
        "",
    )


def _savings(gr: _Graph, ch: _Chain) -> Tuple[int, int]:
    """Parameters and per-sample MACs freed per removed unit."""
    params = 0
    macs = 0
    for e in ch.pre_edits:
        w = gr.init[e.name]
        params += w.size // w.shape[e.axis]
    for e in ch.post_edits:
        w = gr.init[e.name]
        params += (w.size // w.shape[e.axis]) * e.expand
    for node in ch.pre_nodes + ch.post_nodes:
        out = gr.shapes.get(node.output[0])
        if node.op_type == "Conv" and out is not None and len(out) == 4:
            w = gr.init.get(node.input[1])
            if w is None:
                continue
            spatial = int(np.prod(out[2:]))
            if node in ch.pre_nodes:  # removing an output channel
                macs += spatial * int(np.prod(w.shape[1:]))
            else:  # removing an input channel
                macs += out[1] * spatial * int(np.prod(w.shape[2:]))
        elif node.op_type in ("MatMul", "Gemm"):
            w = gr.init.get(node.input[1])
            if w is None or w.ndim != 2:
                continue
            xin = gr.shapes.get(node.input[0])
            rows = int(np.prod(xin[:-1])) if xin else 1
            if node in ch.pre_nodes:
                macs += rows * (
                    w.shape[0]
                    if node.op_type == "MatMul" or not int(_attr(node, "transB", 0))
                    else w.shape[1]
                )
            else:
                transb = node.op_type == "Gemm" and int(_attr(node, "transB", 0))
                macs += rows * (w.shape[0] if transb else w.shape[1]) * 1
    return params, macs


def _macs_of(gr: _Graph, node: onnx.NodeProto, wshape: Tuple[int, ...]) -> int:
    """Per-sample MACs of a Conv / MatMul / Gemm node whose weight has shape ``wshape``."""
    out = gr.shapes.get(node.output[0])
    if node.op_type == "Conv" and out is not None and len(out) == 4:
        return int(np.prod(out[2:])) * int(np.prod(wshape))
    if node.op_type in ("MatMul", "Gemm"):
        x = gr.shapes.get(node.input[0])
        rows = int(np.prod(x[:-1])) if x else 1
        return rows * int(np.prod(wshape))
    return 0


def _combined_savings(gr: _Graph, drops: List[Tuple["_Chain", int]]) -> Tuple[int, int]:
    """Exact parameters / MACs freed when every ``(chain, n_dropped)`` is applied together."""
    shape = {k: list(v.shape) for k, v in gr.init.items()}
    touched = set()
    for ch, d in drops:
        for e in ch.pre_edits:
            shape[e.name][e.axis] -= d
            touched.add(e.name)
        for e in ch.post_edits:
            shape[e.name][e.axis] -= d * e.expand
            touched.add(e.name)
    params = sum(
        int(np.prod(gr.init[n].shape)) - int(np.prod(shape[n])) for n in touched
    )
    macs = 0
    for node in gr.model.graph.node:
        if node.op_type in ("Conv", "MatMul", "Gemm") and node.input[1] in touched:
            macs += _macs_of(gr, node, gr.init[node.input[1]].shape) - _macs_of(
                gr, node, tuple(shape[node.input[1]])
            )
    return params, macs


# --------------------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------------------


def _pin_dynamic_dims(model: onnx.ModelProto) -> onnx.ModelProto:
    """Copy of ``model`` with every non-static graph-input dimension set to 1 (analysis only)."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    inits = {t.name for t in m.graph.initializer}
    for vi in m.graph.input:
        if vi.name in inits:
            continue
        for d in vi.type.tensor_type.shape.dim:
            if not (d.HasField("dim_value") and d.dim_value > 0):
                d.ClearField("dim_param")
                d.dim_value = 1
    del m.graph.value_info[:]
    return m


def _static_shapes(model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        inferred = model
    out: Dict[str, Tuple[int, ...]] = {}
    g = inferred.graph
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        if vi.type.HasField("tensor_type") and tt.HasField("shape"):
            dims = [d.dim_value if d.HasField("dim_value") else 0 for d in tt.shape.dim]
            if all(d > 0 for d in dims):
                out[vi.name] = tuple(dims)
    for t in g.initializer:
        out[t.name] = tuple(t.dims)
    return out


def _resolve_ranges(
    model: onnx.ModelProto, input_ranges: Optional[Dict[str, Tuple]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rg: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict(_ranges.get_ranges(model))
    for k, (lo, hi) in (input_ranges or {}).items():
        rg[k] = (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    inits = {t.name for t in model.graph.initializer}
    need = [
        vi.name
        for vi in model.graph.input
        if vi.name not in inits
        and vi.type.tensor_type.elem_type
        in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16)
    ]
    missing = [
        n
        for n in need
        if n not in rg
        or not (np.all(np.isfinite(rg[n][0])) and np.all(np.isfinite(rg[n][1])))
    ]
    if missing:
        raise ValueError(
            f"no finite input range for {missing}: stable-ReLU pruning is only meaningful over a "
            "declared box (an unbounded input proves nothing stable). Pass input_ranges={name: (lo, hi)} "
            "or annotate the model with onnxsim.ranges.set_range."
        )
    for n, (lo, hi) in rg.items():
        if np.any(lo > hi):
            raise ValueError(f"empty range for {n!r}")
    return rg


def _unit_bounds(
    lo: np.ndarray, hi: np.ndarray, unit_axis: int
) -> Tuple[np.ndarray, np.ndarray]:
    axes = tuple(a for a in range(lo.ndim) if a != unit_axis)
    return lo.min(axis=axes), hi.max(axis=axes)


def _collect_bounds(
    pinned: onnx.ModelProto,
    names: Sequence[str],
    ranges: Dict[str, Tuple[np.ndarray, np.ndarray]],
    methods: Sequence[str],
    max_elements: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], Dict[str, str]]:
    """``{method: {tensor: (lo, hi)}}`` for every pre-activation tensor, and skip reasons."""
    shapes = _static_shapes(pinned)
    total = sum(int(np.prod(shapes.get(n, (0,)))) for n in names)
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    skipped: Dict[str, str] = {}
    rg = {k: (v[0], v[1]) for k, v in ranges.items()}
    for m in methods:
        cap = max_elements.get(m)
        if cap is not None and total > cap:
            skipped[m] = (
                f"{total} pre-activation elements exceed the {m} budget ({cap})"
            )
            continue
        try:
            if m == "interval":
                res = _interval.propagate(pinned, rg)
                got = {
                    n: (
                        np.asarray(res.intervals[n][0]),
                        np.asarray(res.intervals[n][1]),
                    )
                    for n in names
                    if n in res.intervals
                }
            elif m == "crown":
                from . import crown as _crown

                cb = _crown.bounds(pinned, rg, output=list(names), method="crown")
                got = {n: (b.lo, b.hi) for n, b in cb.items()}
            elif m == "zonotope":
                from . import zonotope as _zono

                zr = _zono.propagate(pinned, rg)
                got = {n: zr.bounds(n) for n in names if n in zr.tensors}
            else:
                raise ValueError(f"unknown bound method {m!r}")
        except Exception as e:  # a provider failing must not break the analysis
            skipped[m] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        out[m] = got
    return out, skipped


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------


def analyze(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    methods: Sequence[str] = ("interval", "crown"),
    margin: float = DEFAULT_MARGIN,
    max_elements: Optional[Dict[str, int]] = None,
) -> PruneReport:
    """Count provably stable ReLU units over the input box and what pruning would save.

    :param input_ranges: ``{input: (lo, hi)}`` (scalars or arrays broadcastable to the input);
        merged over the model's ``onnxsim.range.*`` annotations. Raises ``ValueError`` when an
        input has no finite range.
    :param methods: bound providers, any of ``"interval"``, ``"crown"``, ``"zonotope"``. A unit
        is stable when *any* provider proves it (all are sound); ``by_method`` shows what each
        proved on its own. Providers over their size budget are skipped with a reason.
    :param margin: float32 safety margin in pre-activation units (see the module doc).
    :param max_elements: per-method cap on the total number of pre-activation elements
        (defaults: crown 60000, zonotope 4000; interval has none).
    """
    rg = _resolve_ranges(model, input_ranges)
    pinned = _pin_dynamic_dims(model)
    shapes = _static_shapes(pinned)
    gr = _Graph(pinned, shapes)
    relus = [n for n in pinned.graph.node if n.op_type == "Relu"]
    names = [r.input[0] for r in relus if r.input[0] in shapes]
    caps = {"crown": 60000, "zonotope": 4000}
    caps.update(max_elements or {})
    bounds, skipped = _collect_bounds(pinned, names, rg, list(methods), caps)
    layers: List[LayerReport] = []
    drops: List[Tuple[_Chain, int]] = []
    for r in relus:
        t = r.input[0]
        chain, reason = _walk_chain(gr, r)
        shape = shapes.get(t)
        if shape is None:
            layers.append(
                LayerReport(_relu_name(r), t, 0, 0, 0, 0, {}, False, "unknown shape")
            )
            continue
        unit_axis = 1 if (chain and chain.kind == "channel") else len(shape) - 1
        if chain is None:
            # still report stability, using the channel axis for 4-D and the last axis otherwise
            unit_axis = 1 if len(shape) == 4 else len(shape) - 1
        units = shape[unit_axis]
        dead = np.zeros(units, bool)
        on = np.zeros(units, bool)
        by_method: Dict[str, Tuple[int, int]] = {}
        for m, per in bounds.items():
            if t not in per:
                continue
            lo, hi = per[t]
            lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), shape)
            hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), shape)
            ulo, uhi = _unit_bounds(lo, hi, unit_axis)
            d = uhi <= -margin
            o = ulo >= margin
            by_method[m] = (int(d.sum()), int(o.sum()))
            dead |= d
            on |= o & ~d
        rep = LayerReport(
            relu=_relu_name(r),
            pre_activation=t,
            units=units,
            dead=int(dead.sum()),
            always_on=int(on.sum()),
            unstable=int(units - dead.sum() - on.sum()),
            by_method=by_method,
            prunable=chain is not None,
            reason=reason,
            dead_units=np.flatnonzero(dead),
            on_units=np.flatnonzero(on),
        )
        if chain is not None and rep.dead:
            keep_one = rep.dead == units
            n_removed = rep.dead - (1 if keep_one else 0)
            p, mc = _savings(gr, chain)
            rep.params_removable = p * n_removed
            rep.macs_removable = mc * n_removed
            if n_removed:
                drops.append((chain, n_removed))
            if keep_one:
                rep.note = "every unit is dead: the layer is constant; one unit is kept"
        layers.append(rep)
    params_all, macs_all = _combined_savings(gr, drops) if drops else (0, 0)
    return PruneReport(
        layers=layers,
        methods=list(bounds),
        skipped_methods=skipped,
        precondition=rg,
        params_removed=params_all,
        macs_removed=macs_all,
    )


# --------------------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------------------


def _delete(arr: np.ndarray, axis: int, drop: np.ndarray, expand: int) -> np.ndarray:
    idx = np.asarray(drop, dtype=np.int64)
    if expand > 1:
        idx = (idx[:, None] * expand + np.arange(expand)[None, :]).reshape(-1)
    return np.delete(arr, idx, axis=axis)


def _set_init(g: onnx.GraphProto, name: str, arr: np.ndarray) -> None:
    for i, t in enumerate(g.initializer):
        if t.name == name:
            g.initializer[i].CopyFrom(
                numpy_helper.from_array(np.ascontiguousarray(arr), name)
            )
            return
    raise KeyError(name)


def _precondition_metadata(
    m: onnx.ModelProto, rg: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> None:
    drop = [p for p in m.metadata_props if p.key.startswith("onnxsim.precondition.")]
    for p in drop:
        m.metadata_props.remove(p)
    inits = {t.name for t in m.graph.initializer}
    inputs = {vi.name for vi in m.graph.input if vi.name not in inits}
    for name, (lo, hi) in rg.items():
        if name not in inputs:
            continue
        e = m.metadata_props.add()
        e.key = f"onnxsim.precondition.range.{name}"
        e.value = json.dumps(
            {
                "min": np.asarray(lo).item()
                if np.asarray(lo).ndim == 0
                else np.asarray(lo).tolist(),
                "max": np.asarray(hi).item()
                if np.asarray(hi).ndim == 0
                else np.asarray(hi).tolist(),
            }
        )
    e = m.metadata_props.add()
    e.key = "onnxsim.precondition.note"
    e.value = (
        "stable_relu_prune: this model equals the original ONLY for inputs inside the ranges "
        "stored in onnxsim.precondition.range.*; outside them it may differ arbitrarily."
    )


def _affine_of(
    gr: _Graph, nodes: List[onnx.NodeProto]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(W [in, out], b [out])`` of a MatMul(+Add) or Gemm producer, else ``None``."""
    first = nodes[0] if len(nodes) == 1 else None
    if len(nodes) == 2 and nodes[0].op_type == "MatMul" and nodes[1].op_type == "Add":
        w = gr.init.get(nodes[0].input[1])
        cs = [i for i in nodes[1].input if i in gr.init]
        if w is None or w.ndim != 2 or len(cs) != 1:
            return None
        c = gr.init[cs[0]]
        if c.size not in (1, w.shape[1]) or c.ndim > 2:
            return None
        return w.astype(np.float64), np.broadcast_to(
            c.reshape(-1), (w.shape[1],)
        ).astype(np.float64)
    if first is not None and first.op_type == "Gemm":
        if int(_attr(first, "transA", 0)):
            return None
        w = gr.init.get(first.input[1])
        if w is None:
            return None
        w = w.T if int(_attr(first, "transB", 0)) else w
        alpha = float(_attr(first, "alpha", 1.0))
        beta = float(_attr(first, "beta", 1.0))
        out = w.shape[1]
        b = np.zeros(out)
        if len(first.input) > 2 and first.input[2]:
            c = gr.init.get(first.input[2])
            if c is None or c.size not in (1, out):
                return None
            b = beta * np.broadcast_to(c.reshape(-1), (out,)).astype(np.float64)
        return alpha * w.astype(np.float64), b
    return None


def _merge_layers(
    model: onnx.ModelProto, relu_name: str, shapes: Dict[str, Tuple[int, ...]]
) -> Tuple[bool, str]:
    """Fuse ``producer -> Relu -> consumer`` (all units on) into one Gemm, in place."""
    gr = _Graph(model, shapes)
    relu = next(
        (
            n
            for n in model.graph.node
            if n.op_type == "Relu" and _relu_name(n) == relu_name
        ),
        None,
    )
    if relu is None:
        return False, "ReLU not found"
    chain, why = _walk_chain(gr, relu)
    if chain is None:
        return False, why
    if len(chain.post_nodes) != 1 or chain.post_nodes[0].op_type not in (
        "MatMul",
        "Gemm",
    ):
        return False, "merge needs the consumer to follow the ReLU directly"
    aff = _affine_of(gr, chain.pre_nodes)
    if aff is None:
        return False, "producer is not MatMul(+Add) or Gemm"
    x_shape = gr.shapes.get(chain.pre_nodes[0].input[0])
    if x_shape is None or len(x_shape) != 2:
        return False, "merge needs a rank-2 input to the producer"
    cons = chain.post_nodes[0]
    w1, b1 = aff
    w2 = gr.init[cons.input[1]]
    c2 = None
    if cons.op_type == "Gemm":
        if int(_attr(cons, "transA", 0)):
            return False, "consumer Gemm has transA"
        w2 = w2.T if int(_attr(cons, "transB", 0)) else w2
        alpha2 = float(_attr(cons, "alpha", 1.0))
        beta2 = float(_attr(cons, "beta", 1.0))
        w2 = alpha2 * w2
        if len(cons.input) > 2 and cons.input[2]:
            c = gr.init.get(cons.input[2])
            if c is None or c.size not in (1, w2.shape[1]):
                return False, "consumer Gemm bias is not an initializer"
            c2 = beta2 * np.broadcast_to(c.reshape(-1), (w2.shape[1],)).astype(
                np.float64
            )
    w2 = w2.astype(np.float64)
    w_new = w1 @ w2
    b_new = b1 @ w2
    if c2 is not None:
        b_new = b_new + c2
    base = f"{relu_name}_merged".replace("/", "_")
    wn, bn = f"{base}_W", f"{base}_B"
    g = model.graph
    out_name = cons.output[0]
    x_name = chain.pre_nodes[0].input[0]
    for n in chain.pre_nodes + [relu, cons]:
        g.node.remove(n)
    # the merged Gemm replaces the consumer's position: keep topological order by inserting
    # it where the *first* removed node was
    gemm = onnx.helper.make_node(
        "Gemm", [x_name, wn, bn], [out_name], name=f"{base}_gemm"
    )
    g.initializer.append(numpy_helper.from_array(w_new.astype(np.float32), wn))
    g.initializer.append(numpy_helper.from_array(b_new.astype(np.float32), bn))
    # drop the now-unused originals
    keep = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    for t in [
        t for t in g.initializer if t.name not in keep and t.name not in (wn, bn)
    ]:
        g.initializer.remove(t)
    pos = 0
    for k, n in enumerate(g.node):
        if x_name in n.output:
            pos = k + 1
    g.node.insert(pos, gemm)
    return True, ""


def _apply_drops(
    out: onnx.ModelProto, gr: _Graph, items: List[Tuple[_Chain, np.ndarray]]
) -> int:
    """Delete ``drop`` units of each chain from ``out``'s initializers; returns units removed.

    Edits accumulate: one weight is the consumer of one layer and the producer of the next.
    """
    work: Dict[str, np.ndarray] = dict(gr.init)
    removed = 0
    for chain, drop in items:
        if len(drop) == 0:
            continue
        for e in chain.pre_edits:
            work[e.name] = _delete(work[e.name], e.axis, drop, 1)
        for e in chain.post_edits:
            work[e.name] = _delete(work[e.name], e.axis, drop, e.expand)
        removed += len(drop)
    for name, arr in work.items():
        if arr.shape != gr.init[name].shape:
            _set_init(out.graph, name, arr)
    return removed


def prune_units(
    model: onnx.ModelProto, units: Dict[str, Sequence[int]]
) -> Tuple[onnx.ModelProto, int, Dict[str, str]]:
    """Remove CALLER-CHOSEN units, ``{relu node name: unit indices}``. UNCHECKED.

    This is the structural surgery of :func:`apply` without any proof: nothing here says the
    units are dead. Use it for experiments (for example removing channels that were never
    positive on a data set, to compare that with the certified result) and treat the output as
    approximate. The model is marked ``onnxsim.precondition.note`` accordingly. Returns
    ``(model, units_removed, {relu: reason it was skipped})``.
    """
    pinned = _pin_dynamic_dims(model)
    gr = _Graph(pinned, _static_shapes(pinned))
    by_name = {_relu_name(n): n for n in pinned.graph.node if n.op_type == "Relu"}
    items: List[Tuple[_Chain, np.ndarray]] = []
    skipped: Dict[str, str] = {}
    for name, idx in units.items():
        if name not in by_name:
            skipped[name] = "no such ReLU"
            continue
        chain, why = _walk_chain(gr, by_name[name])
        if chain is None:
            skipped[name] = why
            continue
        drop = np.unique(np.asarray(idx, dtype=np.int64))
        if len(drop) and (drop.min() < 0 or drop.max() >= chain.units):
            skipped[name] = "unit index out of range"
            continue
        if len(drop) == chain.units:
            drop = drop[1:]  # a zero-width layer is not representable: keep one unit
        items.append((chain, drop))
    out = copy.deepcopy(model)
    removed = _apply_drops(out, gr, items)
    del out.graph.value_info[:]
    try:
        out = onnx.shape_inference.infer_shapes(out)
    except Exception:
        pass
    e = out.metadata_props.add()
    e.key = "onnxsim.precondition.note"
    e.value = (
        "stable_relu_prune.prune_units: units were removed on the caller's say-so and NOT proved "
        "dead; the output is approximate."
    )
    return out, removed, skipped


def apply(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    remove_dead: bool = True,
    merge_active: bool = False,
    methods: Sequence[str] = ("interval", "crown"),
    margin: float = DEFAULT_MARGIN,
    max_elements: Optional[Dict[str, int]] = None,
    verify_samples: int = 32,
    verify_atol: float = 1e-4,
    seed: int = 0,
) -> Tuple[onnx.ModelProto, PruneReport]:
    """Prune provably dead ReLU units (and optionally merge all-on layers); exact over the box.

    OPT-IN, and exact ONLY for inputs inside the declared ranges (see the module doc). The
    box is stored in the returned model's ``metadata_props``. Raises ``ValueError`` without a
    finite input range, and ``RuntimeError`` if the self-check on ``verify_samples`` random
    inputs inside the box finds a difference above ``verify_atol + 1e-4 * |reference|``
    (``verify_samples=0`` disables it).

    :param merge_active: also fuse ``Gemm/MatMul -> Relu -> Gemm/MatMul`` when every remaining
        unit of the layer is always-on (the ReLU is then the identity). Conv layers are never
        merged (composition would enlarge the kernel); partially active layers are left alone.
    """
    report = analyze(model, input_ranges, methods, margin, max_elements)
    rg = report.precondition
    out = copy.deepcopy(model)
    g = out.graph
    pinned = _pin_dynamic_dims(out)
    shapes = _static_shapes(pinned)
    gr = _Graph(pinned, shapes)
    all_on_after: List[str] = []
    items: List[Tuple[_Chain, np.ndarray]] = []
    if remove_dead:
        by_name = {_relu_name(n): n for n in pinned.graph.node if n.op_type == "Relu"}
        for lay in report.layers:
            if not lay.prunable or lay.dead_units is None:
                continue
            chain, _ = _walk_chain(gr, by_name[lay.relu])
            if chain is None:
                continue
            drop = lay.dead_units
            if len(drop) == lay.units:
                drop = drop[
                    1:
                ]  # keep one (dead) unit: a zero-width layer is not representable
            if len(drop):
                lay.note = (
                    lay.note + "; " if lay.note else ""
                ) + f"removed {len(drop)} units"
            items.append((chain, drop))
            if lay.always_on and lay.always_on == lay.units - lay.dead:
                all_on_after.append(lay.relu)
    elif merge_active:
        for lay in report.layers:
            if lay.prunable and lay.always_on == lay.units:
                all_on_after.append(lay.relu)
    removed = _apply_drops(out, gr, items)
    # refresh value_info so the model stays well-formed after the width changes
    del g.value_info[:]
    merged = 0
    if merge_active:
        for rname in all_on_after:
            ok, why = _merge_layers(out, rname, _static_shapes(_pin_dynamic_dims(out)))
            if ok:
                merged += 1
            else:
                report.notes.append(f"merge skipped at {rname}: {why}")
    report.applied = True
    report.removed_units = removed
    report.merged_layers = merged
    try:
        out = onnx.shape_inference.infer_shapes(out)
    except Exception:
        pass
    _precondition_metadata(out, rg)
    if verify_samples:
        _self_check(model, out, rg, report, verify_samples, verify_atol, seed)
    return out, report


def _self_check(
    orig: onnx.ModelProto,
    pruned: onnx.ModelProto,
    rg: Dict[str, Tuple[np.ndarray, np.ndarray]],
    report: PruneReport,
    n: int,
    atol: float,
    seed: int,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        report.notes.append("self-check skipped: onnxruntime is not installed")
        return
    rng = np.random.default_rng(seed)
    pin_a = _pin_dynamic_dims(orig)
    pin_b = _pin_dynamic_dims(pruned)
    inits = {t.name for t in pin_a.graph.initializer}
    feeds_spec = [vi for vi in pin_a.graph.input if vi.name not in inits]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sa = ort.InferenceSession(
        pin_a.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    sb = ort.InferenceSession(
        pin_b.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    for _ in range(n):
        feeds = {}
        for vi in feeds_spec:
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            np_t = onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
            if vi.name in rg and np.issubdtype(np_t, np.floating):
                feeds[vi.name] = _ranges.sample(rg[vi.name], shape, np_t, rng)
            elif np.issubdtype(np_t, np.floating):
                feeds[vi.name] = rng.standard_normal(shape).astype(np_t)
            else:
                feeds[vi.name] = np.zeros(shape, dtype=np_t)
        ra = sa.run(None, feeds)
        rb = sb.run(None, feeds)
        for a, b in zip(ra, rb):
            if a.dtype.kind != "f":
                continue
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            worst = max(worst, float(diff.max()) if diff.size else 0.0)
            if np.any(diff > atol + 1e-4 * np.abs(a)):
                raise RuntimeError(
                    f"stable_relu_prune self-check failed: pruned model differs by {float(diff.max()):.3g} "
                    "from the original on an input inside the declared box"
                )
    report.verified_max_abs_diff = worst
    report.verified_samples = n
