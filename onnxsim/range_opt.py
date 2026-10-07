"""Range-driven simplification and safe dtype narrowing -- strictly opt-in, precondition-tracked.

``onnxsim.simplify`` rewrites a graph so that it computes the same function *for every input*.
A declared **input range** ("this is an image in [0, 1]", "these are token ids below 30522")
justifies more: a ``Relu`` whose operand is provably non-negative is an identity, a ``Clip`` whose
bounds the operand already satisfies is an identity, a ``Where`` with a decided condition is one
of its branches, an ``int64`` index tensor whose values fit in ``int32`` can be an ``int32`` tensor.
This module finds those rewrites with interval analysis (:mod:`onnxsim.interval`, optionally
tightened by :mod:`onnxsim.crown`) and applies them.

**Every range-dependent rewrite is valid only inside the declared box.** Therefore:

* :func:`apply` never runs without ranges and never runs by default anywhere: it is an explicit call.
* The returned model records the box it relied on under ``onnxsim.precondition.range.<input>``
  (model-level ``metadata_props``, same JSON as :mod:`onnxsim.ranges`) and a machine-readable log
  under ``onnxsim.range_opt.log``. Deployments can call :func:`check_precondition` to reject
  inputs outside the box.
* Each rewrite carries its proof (the proving interval). A rewrite that would also be valid with
  every input unbounded is marked ``unconditional`` and needs no precondition.
* Rewrites that change the model's *interface* (an ``int64`` graph input becoming ``int32``)
  additionally require ``allow_interface_change=True``.

Rules
-----
``dead_relu``, ``dead_abs`` (also ``Abs`` of a non-positive operand becomes ``Neg``), ``dead_clip``
(whole ``Clip`` removed, or only the redundant bound dropped), ``dead_minmax`` (``Min``/``Max``
against a constant or another tensor), ``decided_where`` and ``decided_if`` (the condition is a
comparison the intervals decide), ``cast_noop`` / ``cast_roundtrip`` (a ``Cast`` pair that provably
loses nothing), ``narrow_const_index`` (constant ``int64`` index tensors of ``Gather`` /
``GatherElements`` / ``Slice`` become ``int32``; ``Slice`` bounds beyond ``int32`` are clamped, which
``Slice`` semantics make equivalent), ``narrow_input`` (an ``int64`` graph input used only as
indices / cast source becomes ``int32``; interface change), ``softmax_no_max`` (a decomposed softmax
``Exp(x - ReduceMax(x)) / ReduceSum(...)`` loses the max subtraction when the logits are in a range
where ``exp`` cannot overflow or underflow in float32).

Report-only (never applied): ``fp16_risk`` (tensors, accumulators and ``Exp`` operands whose proven
range exceeds float16) and ``int64_fits_int32`` (computed ``int64`` tensors that would fit).

What a proof means
------------------
Intervals enclose the **real-number** function; float32 execution can differ by rounding, so a
rewrite justified at the edge of a range (``lo`` exactly ``0`` for a ``Relu``) can change a value by
float-rounding magnitude. ``margin`` demands a gap. Outside the declared box the rewritten model can
differ arbitrarily from the original -- that is the point of the precondition.
"""

import argparse
import copy
import dataclasses
import json
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import interval as _interval
from . import ranges as _ranges

PRECONDITION_PREFIX = "onnxsim.precondition.range."
LOG_KEY = "onnxsim.range_opt.log"

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
FP16_MAX = 65504.0
# float32 exp(x) is finite for x <= 88.72 and a normal number for x >= -87.3; keep a margin.
_EXP_HI = 88.0
_EXP_LO = -80.0

ALL_RULES = (
    "dead_relu",
    "dead_abs",
    "dead_clip",
    "dead_minmax",
    "decided_where",
    "decided_if",
    "cast_noop",
    "cast_roundtrip",
    "narrow_const_index",
    "narrow_input",
    "softmax_no_max",
)
REPORT_RULES = ("fp16_risk", "int64_fits_int32")

Hull = Tuple[float, float]
HullFn = Callable[[str], Optional[Hull]]


class PreconditionViolation(ValueError):
    """Concrete inputs lie outside the box a rewritten model relies on."""


@dataclasses.dataclass
class Rewrite:
    """One proposed (or applied) rewrite with the proof that justifies it."""

    rule: str
    node: str
    op_type: str
    description: str
    proof: Dict[str, Any]
    applies: bool = True  # False: report only, never applied
    unconditional: bool = False  # also provable with every input unbounded
    interface_change: bool = False
    index: int = -1  # position in graph.node (-1 for model-level reports)
    action: Tuple[Any, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "node": self.node,
            "op_type": self.op_type,
            "description": self.description,
            "proof": self.proof,
            "applies": self.applies,
            "unconditional": self.unconditional,
            "interface_change": self.interface_change,
        }


# ------------------------------------------------------------------------------------------
# Graph facts
# ------------------------------------------------------------------------------------------

_INT_RANGE = {
    TensorProto.INT8: (-(2**7), 2**7 - 1),
    TensorProto.UINT8: (0, 2**8 - 1),
    TensorProto.INT16: (-(2**15), 2**15 - 1),
    TensorProto.UINT16: (0, 2**16 - 1),
    TensorProto.INT32: (INT32_MIN, INT32_MAX),
    TensorProto.UINT32: (0, 2**32 - 1),
    TensorProto.INT64: (-(2**63), 2**63 - 1),
    TensorProto.UINT64: (0, 2**64 - 1),
}
# integers a float type represents exactly: |v| <= 2**mantissa_bits
_FLOAT_EXACT_BITS = {
    TensorProto.FLOAT16: 11,
    TensorProto.BFLOAT16: 8,
    TensorProto.FLOAT: 24,
    TensorProto.DOUBLE: 53,
}
_FLOAT_WIDEN = {
    TensorProto.FLOAT16: {TensorProto.FLOAT, TensorProto.DOUBLE},
    TensorProto.BFLOAT16: {TensorProto.FLOAT, TensorProto.DOUBLE},
    TensorProto.FLOAT: {TensorProto.DOUBLE},
    TensorProto.DOUBLE: set(),
}


def _node_label(node: onnx.NodeProto, idx: int) -> str:
    return node.name or f"{node.op_type}#{idx}"


def _subgraph_inputs(node: onnx.NodeProto) -> Iterable[str]:
    """Every name read by the nodes of the subgraphs of ``node`` (outer references included)."""
    for a in node.attribute:
        graphs = []
        if a.HasField("g"):
            graphs.append(a.g)
        graphs.extend(a.graphs)
        for g in graphs:
            for n in g.node:
                yield from (x for x in n.input if x)
                yield from _subgraph_inputs(n)
            yield from (o.name for o in g.output)


class _Facts:
    """Producers, consumers, constants, element types and dims of a model."""

    def __init__(self, model: onnx.ModelProto):
        self.model = model
        g = model.graph
        self.nodes = list(g.node)
        self.producer: Dict[str, int] = {}
        for i, n in enumerate(self.nodes):
            for o in n.output:
                if o:
                    self.producer[o] = i
        self.consumers: Dict[str, List[Tuple[int, int]]] = {}
        for i, n in enumerate(self.nodes):
            for pos, x in enumerate(n.input):
                if x:
                    self.consumers.setdefault(x, []).append((i, pos))
            for x in _subgraph_inputs(n):
                self.consumers.setdefault(x, []).append((i, -1))
        self.outputs = [o.name for o in g.output]
        self.output_set = set(self.outputs)
        self.inputs = [
            i.name for i in g.input if i.name not in {t.name for t in g.initializer}
        ]
        self.const: Dict[str, np.ndarray] = {
            t.name: numpy_helper.to_array(t) for t in g.initializer
        }
        self.const_origin: Dict[str, Tuple[str, int]] = {
            t.name: ("init", k) for k, t in enumerate(g.initializer)
        }
        for i, n in enumerate(self.nodes):
            if n.op_type == "Constant" and not n.domain and n.output:
                for a in n.attribute:
                    if a.name == "value":
                        self.const[n.output[0]] = numpy_helper.to_array(a.t)
                        self.const_origin[n.output[0]] = ("node", i)
        self.dtype: Dict[str, int] = {}
        self.dims: Dict[str, Optional[Tuple[Any, ...]]] = {}
        try:
            inferred = onnx.shape_inference.infer_shapes(model).graph
        except Exception:
            inferred = g
        for vi in (
            list(inferred.input) + list(inferred.value_info) + list(inferred.output)
        ):
            tt = vi.type.tensor_type
            if vi.type.HasField("tensor_type") and tt.elem_type:
                self.dtype[vi.name] = tt.elem_type
                if tt.HasField("shape"):
                    self.dims[vi.name] = tuple(
                        ("v", d.dim_value)
                        if d.HasField("dim_value")
                        else (
                            ("p", d.dim_param) if d.HasField("dim_param") else ("?", 0)
                        )
                        for d in tt.shape.dim
                    )
        for t in g.initializer:
            self.dtype[t.name] = t.data_type
            self.dims[t.name] = tuple(("v", int(d)) for d in t.dims)

    def same_shape(self, a: str, b: str) -> bool:
        da, db = self.dims.get(a), self.dims.get(b)
        if da is None or db is None or len(da) != len(db):
            return False
        return all(x == y and x[0] != "?" for x, y in zip(da, db))

    def numel(self, name: str) -> Optional[int]:
        d = self.dims.get(name)
        if d is None or any(k != "v" for k, _ in d):
            return None
        return int(np.prod([v for _, v in d], dtype=np.int64)) if d else 1


# ------------------------------------------------------------------------------------------
# Rules. Each takes (facts, idx, node, H, margin) and returns a Rewrite or None. ``H`` maps a
# tensor name to a proven (lo, hi) hull or None; the same rule is evaluated again with every
# input unbounded to decide whether the rewrite is unconditional.
# ------------------------------------------------------------------------------------------


def _scalar_const(f: _Facts, name: str) -> Optional[float]:
    if not name:
        return None
    c = f.const.get(name)
    if c is None or c.size != 1 or c.dtype.kind not in "fiu":
        return None
    return float(c.reshape(-1)[0])


def _bypass(rule, f, idx, node, src, desc, proof) -> Rewrite:
    return Rewrite(
        rule,
        _node_label(node, idx),
        node.op_type,
        desc,
        proof,
        index=idx,
        action=("bypass", src),
    )


def _rule_dead_relu(f, idx, node, H, margin):
    if node.op_type != "Relu" or node.domain not in ("", "ai.onnx"):
        return None
    h = H(node.input[0])
    if h is None or not h[0] >= margin:
        return None
    return _bypass(
        "dead_relu",
        f,
        idx,
        node,
        node.input[0],
        f"Relu({node.input[0]}) is the identity: {node.input[0]} in [{h[0]:.6g}, {h[1]:.6g}] is non-negative",
        {"tensor": node.input[0], "hull": list(h), "needs": ">= 0"},
    )


def _rule_dead_abs(f, idx, node, H, margin):
    if node.op_type != "Abs" or node.domain not in ("", "ai.onnx"):
        return None
    h = H(node.input[0])
    if h is None:
        return None
    if h[0] >= margin:
        return _bypass(
            "dead_abs",
            f,
            idx,
            node,
            node.input[0],
            f"Abs({node.input[0]}) is the identity: {node.input[0]} in [{h[0]:.6g}, {h[1]:.6g}] is non-negative",
            {"tensor": node.input[0], "hull": list(h), "needs": ">= 0"},
        )
    if h[1] <= -margin:
        return Rewrite(
            "dead_abs",
            _node_label(node, idx),
            node.op_type,
            f"Abs({node.input[0]}) is Neg: {node.input[0]} in [{h[0]:.6g}, {h[1]:.6g}] is non-positive",
            {"tensor": node.input[0], "hull": list(h), "needs": "<= 0"},
            index=idx,
            action=("to_neg",),
        )
    return None


def _clip_bounds(f, node) -> Optional[Tuple[Optional[float], Optional[float], str]]:
    """(min, max, 'inputs'|'attrs') of a Clip, or None if they are not constants."""
    attrs = {a.name: a for a in node.attribute}
    if "min" in attrs or "max" in attrs:  # opset < 11
        lo = float(helper.get_attribute_value(attrs["min"])) if "min" in attrs else None
        hi = float(helper.get_attribute_value(attrs["max"])) if "max" in attrs else None
        return lo, hi, "attrs"
    lo = hi = None
    if len(node.input) > 1 and node.input[1]:
        lo = _scalar_const(f, node.input[1])
        if lo is None:
            return None
    if len(node.input) > 2 and node.input[2]:
        hi = _scalar_const(f, node.input[2])
        if hi is None:
            return None
    return lo, hi, "inputs"


def _rule_dead_clip(f, idx, node, H, margin):
    if node.op_type != "Clip" or node.domain not in ("", "ai.onnx"):
        return None
    b = _clip_bounds(f, node)
    h = H(node.input[0])
    if b is None or h is None:
        return None
    lo_b, hi_b, kind = b
    lo_ok = lo_b is None or h[0] >= lo_b + margin
    hi_ok = hi_b is None or h[1] <= hi_b - margin
    x = node.input[0]
    rng = f"[{h[0]:.6g}, {h[1]:.6g}]"
    proof = {"tensor": x, "hull": list(h), "clip": [lo_b, hi_b]}
    if lo_ok and hi_ok:
        return _bypass(
            "dead_clip",
            f,
            idx,
            node,
            x,
            f"Clip({x}, {lo_b}, {hi_b}) is the identity: {x} in {rng} already satisfies both bounds",
            proof,
        )
    if lo_b is not None and hi_b is not None and lo_ok != hi_ok:
        drop = "min" if lo_ok else "max"
        return Rewrite(
            "dead_clip",
            _node_label(node, idx),
            node.op_type,
            f"Clip({x}, {lo_b}, {hi_b}) keeps only its {'max' if lo_ok else 'min'} bound: "
            f"{x} in {rng} already satisfies the {drop} bound",
            {**proof, "dropped": drop},
            index=idx,
            action=("clip_drop", drop, kind),
        )
    return None


def _rule_dead_minmax(f, idx, node, H, margin):
    if node.op_type not in ("Min", "Max") or node.domain not in ("", "ai.onnx"):
        return None
    if len(node.input) != 2 or not all(node.input):
        return None
    a, b = node.input
    out = node.output[0]
    is_max = node.op_type == "Max"
    for keep, other in ((a, b), (b, a)):
        hk, ho = H(keep), H(other)
        if hk is None or ho is None or not f.same_shape(keep, out):
            continue
        # Max(keep, other) == keep when keep >= other everywhere; Min when keep <= other.
        if (is_max and hk[0] >= ho[1] + margin) or (
            not is_max and hk[1] <= ho[0] - margin
        ):
            op = ">=" if is_max else "<="
            return _bypass(
                "dead_minmax",
                f,
                idx,
                node,
                keep,
                f"{node.op_type}({a}, {b}) is {keep}: {keep} in [{hk[0]:.6g}, {hk[1]:.6g}] "
                f"is always {op} {other} in [{ho[0]:.6g}, {ho[1]:.6g}]",
                {
                    "kept": keep,
                    "kept_hull": list(hk),
                    "other": other,
                    "other_hull": list(ho),
                },
            )
    return None


_CMP = ("Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal")


def _decide(
    f: _Facts, name: str, H: HullFn, margin: float, depth: int = 0
) -> Optional[bool]:
    """True/False if the boolean tensor ``name`` is the same constant everywhere in the box."""
    if depth > 4:
        return None
    c = f.const.get(name)
    if c is not None and c.dtype == np.bool_ and c.size:
        return True if c.all() else (False if not c.any() else None)
    i = f.producer.get(name)
    if i is None:
        return None
    n = f.nodes[i]
    if n.domain not in ("", "ai.onnx"):
        return None
    if n.op_type == "Not":
        d = _decide(f, n.input[0], H, margin, depth + 1)
        return None if d is None else (not d)
    if n.op_type in ("And", "Or") and len(n.input) == 2:
        x = _decide(f, n.input[0], H, margin, depth + 1)
        y = _decide(f, n.input[1], H, margin, depth + 1)
        if n.op_type == "And":
            if x is False or y is False:
                return False
            return True if (x is True and y is True) else None
        if x is True or y is True:
            return True
        return False if (x is False and y is False) else None
    if n.op_type in _CMP and len(n.input) == 2:
        ha, hb = H(n.input[0]), H(n.input[1])
        if ha is None or hb is None:
            return None
        if n.op_type == "Greater":
            return (
                True
                if ha[0] > hb[1] + margin
                else (False if ha[1] <= hb[0] - margin else None)
            )
        if n.op_type == "Less":
            return (
                True
                if ha[1] < hb[0] - margin
                else (False if ha[0] >= hb[1] + margin else None)
            )
        if n.op_type == "GreaterOrEqual":
            return (
                True
                if ha[0] >= hb[1] + margin
                else (False if ha[1] < hb[0] - margin else None)
            )
        if n.op_type == "LessOrEqual":
            return (
                True
                if ha[1] <= hb[0] - margin
                else (False if ha[0] > hb[1] + margin else None)
            )
        if ha[1] < hb[0] - margin or hb[1] < ha[0] - margin:  # Equal
            return False
        if ha[0] == ha[1] == hb[0] == hb[1]:
            return True
    return None


def _rule_decided_where(f, idx, node, H, margin):
    if (
        node.op_type != "Where"
        or node.domain not in ("", "ai.onnx")
        or len(node.input) != 3
    ):
        return None
    d = _decide(f, node.input[0], H, margin)
    if d is None:
        return None
    keep = node.input[1] if d else node.input[2]
    if not f.same_shape(keep, node.output[0]):
        return None
    return _bypass(
        "decided_where",
        f,
        idx,
        node,
        keep,
        f"Where({node.input[0]}, ...) is its {'then' if d else 'else'} input {keep}: the condition is "
        f"{d} everywhere in the box",
        {"condition": node.input[0], "decided": d},
    )


def _rule_decided_if(f, idx, node, H, margin):
    if node.op_type != "If" or node.domain not in ("", "ai.onnx") or not node.input:
        return None
    d = _decide(f, node.input[0], H, margin)
    if d is None:
        return None
    attr = {a.name: a for a in node.attribute}
    br = attr["then_branch" if d else "else_branch"].g
    if any(
        a.HasField("g") or a.graphs for n in br.node for a in n.attribute
    ):  # nested subgraphs: name scoping is not handled
        return None
    if len(br.output) != len(node.output):
        return None
    return Rewrite(
        "decided_if",
        _node_label(node, idx),
        node.op_type,
        f"If({node.input[0]}) is inlined to its {'then' if d else 'else'} branch: the condition is {d} "
        f"everywhere in the box",
        {"condition": node.input[0], "decided": d},
        index=idx,
        action=("inline_if", d),
    )


def _rule_cast(f, idx, node, H, margin):
    if node.op_type != "Cast" or node.domain not in ("", "ai.onnx"):
        return None
    to = next((int(a.i) for a in node.attribute if a.name == "to"), None)
    x = node.input[0]
    src = f.dtype.get(x)
    if to is None or src is None:
        return None
    if src == to:
        return _bypass(
            "cast_noop",
            f,
            idx,
            node,
            x,
            f"Cast({x}) is the identity: it is already {onnx.TensorProto.DataType.Name(to)}",
            {"tensor": x, "dtype": onnx.TensorProto.DataType.Name(to)},
        )
    # Cast(to=B) of Cast(to=A) of a B tensor: the pair is the identity when A holds every value.
    p = f.producer.get(x)
    if p is None or f.nodes[p].op_type != "Cast":
        return None
    inner = f.nodes[p]
    a_type = next((int(a.i) for a in inner.attribute if a.name == "to"), None)
    root = inner.input[0]
    if a_type is None or f.dtype.get(root) != to:
        return None
    full = _INT_RANGE.get(to)
    h = H(root)
    if full is not None:
        # a value of an integer type always lies in that type's range, whatever the box says
        lo_t, hi_t = float(full[0]), float(full[1])
        h = (lo_t, hi_t) if h is None else (max(h[0], lo_t), min(h[1], hi_t))
    if not _lossless(to, a_type, h):
        return None
    names = onnx.TensorProto.DataType.Name
    return _bypass(
        "cast_roundtrip",
        f,
        idx,
        node,
        root,
        f"Cast({names(a_type)}) then Cast({names(to)}) of {root} is the identity: every value of "
        f"{root} (hull {None if h is None else [float(h[0]), float(h[1])]}) is exact in {names(a_type)}",
        {"tensor": root, "hull": None if h is None else list(h), "via": names(a_type)},
    )


def _lossless(src: int, mid: int, h: Optional[Hull]) -> bool:
    """Is every value of a ``src`` tensor with hull ``h`` exactly representable in ``mid``?"""
    if src == mid:
        return True
    if src in _INT_RANGE:
        lo, hi = _INT_RANGE[src] if h is None else (h[0], h[1])
        if mid in _INT_RANGE:
            m_lo, m_hi = _INT_RANGE[mid]
            return lo >= m_lo and hi <= m_hi
        if mid in _FLOAT_EXACT_BITS:
            bound = 2.0 ** _FLOAT_EXACT_BITS[mid]
            return lo >= -bound and hi <= bound
        return False
    if src in _FLOAT_WIDEN:
        return mid in _FLOAT_WIDEN[src]
    return False


# ---- integer narrowing ---------------------------------------------------------------------

# (op_type, input position) pairs that accept int32 or int64 indices (ONNX Tind).
_INDEX_USES = {("Gather", 1), ("GatherElements", 1)}
_SLICE_POS = (1, 2, 3, 4)  # starts, ends, axes, steps
_SLICE_BOUND_POS = (1, 2)


def _fits(v: np.ndarray) -> bool:
    return bool(v.size == 0 or (v.min() >= INT32_MIN and v.max() <= INT32_MAX))


def _slice_inputs(node: onnx.NodeProto) -> List[str]:
    return [node.input[p] for p in _SLICE_POS if len(node.input) > p and node.input[p]]


def _rule_narrow_const_index(f: _Facts) -> List[Rewrite]:
    """Constant int64 index tensors -> int32 (model-level: one proposal per tensor)."""
    out: List[Rewrite] = []
    for name, c in f.const.items():
        if c.dtype != np.int64 or name in f.output_set:
            continue
        uses = f.consumers.get(name, [])
        if not uses:
            continue
        ok = True
        clampable = True
        for i, pos in uses:
            n = f.nodes[i]
            if n.domain not in ("", "ai.onnx") or pos < 0:
                ok = False
                break
            if (n.op_type, pos) in _INDEX_USES:
                clampable = False
            elif n.op_type == "Slice" and pos in _SLICE_POS:
                clampable &= pos in _SLICE_BOUND_POS
                # every present index input of this Slice must itself be a constant int64:
                # starts/ends/axes/steps must share one type
                for other in _slice_inputs(n):
                    oc = f.const.get(other)
                    if oc is None or oc.dtype != np.int64:
                        ok = False
            else:
                ok = False
            if not ok:
                break
        if not ok:
            continue
        fits = _fits(c)
        if not fits and not clampable:
            continue
        i0 = uses[0][0]
        roles = sorted({f"{f.nodes[i].op_type}[{pos}]" for i, pos in uses})
        out.append(
            Rewrite(
                "narrow_const_index",
                _node_label(f.nodes[i0], i0),
                f.nodes[i0].op_type,
                f"constant int64 tensor {name} used as {', '.join(roles)} becomes int32"
                + (
                    ""
                    if fits
                    else " (out-of-range Slice bounds clamped to +-(2^31-1): Slice clamps to the dimension anyway)"
                ),
                {
                    "tensor": name,
                    "min": int(c.min()) if c.size else None,
                    "max": int(c.max()) if c.size else None,
                    "clamped": not fits,
                    "uses": roles,
                },
                unconditional=True,
                index=i0,
                action=("retype_const", name, not fits),
            )
        )
    return out


_CAST_OR_INDEX_OK = {("Gather", 1), ("GatherElements", 1)}


def _rule_narrow_input(f: _Facts, H: HullFn, allow: bool) -> List[Rewrite]:
    out: List[Rewrite] = []
    for name in f.inputs:
        if f.dtype.get(name) != TensorProto.INT64:
            continue
        h = H(name)
        uses = f.consumers.get(name, [])
        if h is None or not uses or not (h[0] >= INT32_MIN and h[1] <= INT32_MAX):
            continue
        if not all(
            pos >= 0
            and f.nodes[i].domain in ("", "ai.onnx")
            and (
                (f.nodes[i].op_type, pos) in _CAST_OR_INDEX_OK
                or f.nodes[i].op_type == "Cast"
            )
            for i, pos in uses
        ):
            continue
        roles = sorted({f"{f.nodes[i].op_type}[{pos}]" for i, pos in uses})
        out.append(
            Rewrite(
                "narrow_input",
                f"<input {name}>",
                "graph input",
                f"graph input {name} int64 -> int32 (used only as {', '.join(roles)}; hull "
                f"[{h[0]:.0f}, {h[1]:.0f}] fits int32). Callers must now feed int32.",
                {"tensor": name, "hull": list(h), "uses": roles},
                applies=allow,
                interface_change=True,
                index=uses[0][0],
                action=("retype_input", name),
            )
        )
    return out


# ---- decomposed softmax ----------------------------------------------------------------------


def _reduce_axes(
    f: _Facts, node: onnx.NodeProto, rank: int
) -> Optional[Tuple[int, ...]]:
    ax = None
    for a in node.attribute:
        if a.name == "axes":
            ax = [int(v) for v in a.ints]
    if ax is None and len(node.input) > 1 and node.input[1]:
        c = f.const.get(node.input[1])
        if c is None:
            return None
        ax = [int(v) for v in c.reshape(-1)]
    if ax is None:
        return tuple(range(rank))
    return tuple(sorted(v % rank for v in ax))


def _keepdims(node: onnx.NodeProto) -> int:
    return next((int(a.i) for a in node.attribute if a.name == "keepdims"), 1)


def _rule_softmax_no_max(f, idx, node, H, margin):
    if node.op_type != "Exp" or node.domain not in ("", "ai.onnx"):
        return None
    d = node.input[0]
    p = f.producer.get(d)
    if p is None or f.nodes[p].op_type != "Sub" or len(f.nodes[p].input) != 2:
        return None
    sub = f.nodes[p]
    x, m = sub.input
    pm = f.producer.get(m)
    if pm is None or f.nodes[pm].op_type != "ReduceMax" or f.nodes[pm].input[0] != x:
        return None
    rmax = f.nodes[pm]
    if [c for c in f.consumers.get(d, [])] != [(idx, 0)]:
        return None
    # Exp output e must feed exactly ReduceSum(e) and Div(e, sum): a scale-invariant normalisation.
    e = node.output[0]
    cons = f.consumers.get(e, [])
    kinds = sorted(f.nodes[i].op_type for i, _ in cons)
    if kinds != ["Div", "ReduceSum"]:
        return None
    rs_i = next(i for i, _ in cons if f.nodes[i].op_type == "ReduceSum")
    dv_i, dv_pos = next((i, pos) for i, pos in cons if f.nodes[i].op_type == "Div")
    rsum, div = f.nodes[rs_i], f.nodes[dv_i]
    if (
        dv_pos != 0
        or div.input[1] != rsum.output[0]
        or f.consumers.get(rsum.output[0], []) != [(dv_i, 1)]
    ):
        return None
    dims = f.dims.get(x)
    if f.dtype.get(x) != TensorProto.FLOAT or dims is None:
        return None
    rank = len(dims)
    ax_max, ax_sum = _reduce_axes(f, rmax, rank), _reduce_axes(f, rsum, rank)
    if (
        ax_max is None
        or ax_max != ax_sum
        or _keepdims(rmax) != 1
        or _keepdims(rsum) != 1
    ):
        return None
    if any(dims[a][0] != "v" for a in ax_max):
        return None
    n = int(np.prod([dims[a][1] for a in ax_max], dtype=np.int64))
    h = H(x)
    if h is None or n < 1:
        return None
    ln_n = math.log(n)
    if not (h[1] + ln_n <= _EXP_HI - margin and h[0] >= _EXP_LO + margin):
        return None
    return Rewrite(
        "softmax_no_max",
        _node_label(node, idx),
        node.op_type,
        f"decomposed softmax over {x}: the max subtraction is dropped; logits in [{h[0]:.6g}, {h[1]:.6g}] "
        f"over {n} elements keep exp(x) finite and exp(sum) normal in float32",
        {
            "tensor": x,
            "hull": list(h),
            "reduced_elements": n,
            "needs": f"hi + ln(n) <= {_EXP_HI} and lo >= {_EXP_LO}",
        },
        index=idx,
        action=("bypass_index", p, x),
    )


_NODE_RULES: Dict[str, Callable[..., Optional[Rewrite]]] = {
    "dead_relu": _rule_dead_relu,
    "dead_abs": _rule_dead_abs,
    "dead_clip": _rule_dead_clip,
    "dead_minmax": _rule_dead_minmax,
    "decided_where": _rule_decided_where,
    "decided_if": _rule_decided_if,
    "cast_noop": _rule_cast,
    "cast_roundtrip": _rule_cast,
    "softmax_no_max": _rule_softmax_no_max,
}


# ---- report-only ---------------------------------------------------------------------------


def _report_fp16(f: _Facts, H: HullFn) -> Optional[Rewrite]:
    # "risk" = the proven bound exceeds float16. Intervals over-approximate, so this means overflow
    # is NOT EXCLUDED, not that it happens; "unbounded" = no finite bound at all (unknown).
    unsafe: List[Dict[str, Any]] = []
    checked = safe = unbounded = 0
    for n in f.nodes:
        for o in n.output:
            if not o or f.dtype.get(o) != TensorProto.FLOAT:
                continue
            h = H(o)
            if h is None:
                continue
            checked += 1
            m = max(abs(h[0]), abs(h[1]))
            if math.isinf(m):
                unbounded += 1  # no finite bound: unknown, not proven risky
            elif m > FP16_MAX:
                unsafe.append(
                    {
                        "tensor": o,
                        "op": n.op_type,
                        "max_abs": m,
                        "why": "proven bound exceeds float16",
                    }
                )
            else:
                safe += 1
        if n.op_type == "Exp" and n.input:
            h = H(n.input[0])
            if h is not None and math.isfinite(h[1]) and h[1] > math.log(FP16_MAX):
                unsafe.append(
                    {
                        "tensor": n.output[0],
                        "op": "Exp",
                        "max_abs": h[1],
                        "why": "exp operand above ln(65504)",
                    }
                )
        if (
            n.op_type in ("MatMul", "Gemm", "Conv")
            and len(n.input) > 1
            and n.input[1] in f.const
        ):
            w = np.abs(f.const[n.input[1]].astype(np.float64))
            hx = H(n.input[0])
            if hx is None or w.ndim < 2:
                continue
            if n.op_type == "Conv":
                per_out = w.reshape(w.shape[0], -1).sum(axis=1)
            elif n.op_type == "Gemm" and any(
                a.name == "transB" and a.i for a in n.attribute
            ):
                per_out = w.sum(axis=1)
            else:
                per_out = w.reshape(-1, w.shape[-1]).sum(axis=0)
            bound = float(per_out.max()) * max(abs(hx[0]), abs(hx[1]))
            if math.isfinite(bound) and bound > FP16_MAX:
                unsafe.append(
                    {
                        "tensor": n.output[0],
                        "op": n.op_type,
                        "max_abs": bound,
                        "why": "accumulator bound sum|w|*max|x| exceeds float16",
                    }
                )
    if checked == 0:
        return None
    return Rewrite(
        "fp16_risk",
        "<model>",
        "report",
        f"of {checked} float32 tensors with a proven range: {safe} are proven inside float16; "
        f"{len(unsafe)} finding(s) have a proven bound beyond float16 (tensor range, Exp operand or "
        f"accumulator sum|w|*max|x|), so overflow is not excluded -- intervals over-approximate, so "
        f"this is a risk, not a certainty; {unbounded} have no finite bound (unknown) -- "
        f"informational, never applied",
        {
            "checked": checked,
            "safe_count": safe,
            "risk_count": len(unsafe),
            "unbounded_count": unbounded,
            "risk": unsafe[:25],
        },
        applies=False,
    )


def _report_int64(f: _Facts, H: HullFn) -> Optional[Rewrite]:
    fits: List[str] = []
    total = 0
    for n in f.nodes:
        for o in n.output:
            if o and f.dtype.get(o) == TensorProto.INT64 and o not in f.const:
                total += 1
                h = H(o)
                if h is not None and h[0] >= INT32_MIN and h[1] <= INT32_MAX:
                    fits.append(o)
    if total == 0:
        return None
    return Rewrite(
        "int64_fits_int32",
        "<model>",
        "report",
        f"{len(fits)} of {total} computed int64 tensors have a proven range that fits int32 "
        "(report only: ops such as Reshape/Expand/Tile/ConstantOfShape require int64 shape operands)",
        {"fits": len(fits), "computed_int64": total, "examples": fits[:15]},
        applies=False,
    )


# ------------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------------


def _hull_fn(res: "_interval.IntervalResult") -> HullFn:
    def H(name: str) -> Optional[Hull]:
        if name in res.intervals or name in res.ranged:
            try:
                lo, hi = res.hull(name)
            except Exception:
                return None
            return None if math.isnan(lo) or math.isnan(hi) else (lo, hi)
        return None

    return H


def _free_ranges(model: onnx.ModelProto) -> Dict[str, Tuple[float, float]]:
    """An 'every input unbounded' box, overriding any annotation."""
    inits = {t.name for t in model.graph.initializer}
    return {
        i.name: (-math.inf, math.inf) for i in model.graph.input if i.name not in inits
    }


def _merged_ranges(
    model: onnx.ModelProto, input_ranges: Optional[Dict[str, Tuple]]
) -> Dict[str, Tuple[Any, Any]]:
    merged: Dict[str, Tuple[Any, Any]] = dict(_ranges.get_ranges(model))
    for k, (lo, hi) in (input_ranges or {}).items():
        merged[k] = (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    inputs = {i.name for i in model.graph.input}
    return {k: v for k, v in merged.items() if k in inputs}


def _tight_hulls(
    model: onnx.ModelProto,
    ranges: Dict[str, Tuple[Any, Any]],
    names: Sequence[str],
    max_nodes: int,
) -> Dict[str, Hull]:
    """CROWN hulls for ``names`` (small models only); empty on any failure."""
    if not names or len(model.graph.node) > max_nodes:
        return {}
    try:
        from . import crown as _crown

        b = _crown.bounds(model, ranges, output=list(names), method="crown")
        return {
            n: (float(np.min(t.lo)), float(np.max(t.hi)))
            for n, t in b.items()
            if np.all(np.isfinite(t.lo)) and np.all(np.isfinite(t.hi))
        }
    except Exception:
        return {}


def _operand_names(f: _Facts, node: onnx.NodeProto) -> List[str]:
    t = node.op_type
    if t in ("Relu", "Abs", "Clip"):
        return [node.input[0]]
    if t in ("Min", "Max") and len(node.input) == 2:
        return list(node.input)
    if t in ("Where", "If") and node.input:
        out, stack = [], [node.input[0]]
        for _ in range(8):
            if not stack:
                break
            c = stack.pop()
            i = f.producer.get(c)
            if i is None:
                continue
            n = f.nodes[i]
            if n.op_type in _CMP:
                out += list(n.input)
            elif n.op_type in ("Not", "And", "Or"):
                stack += list(n.input)
        return out
    return []


def analyze(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    rules: Optional[Sequence[str]] = None,
    engine: str = "interval",
    allow_interface_change: bool = False,
    margin: float = 0.0,
    max_crown_nodes: int = 80,
    result: Optional["_interval.IntervalResult"] = None,
    check_unconditional: bool = True,
    input_shapes: Optional[Dict[str, Sequence[Any]]] = None,
    max_crown_elements: int = 4096,
) -> List[Rewrite]:
    """Find the rewrites the declared ranges justify, each with its proof. Changes nothing.

    :param input_ranges: ``{input: (lo, hi)}``, merged over the model's ``onnxsim.range.*``
        annotations. Inputs without a range stay unbounded (so only unconditional rewrites can
        be proved through them).
    :param rules: subset of :data:`ALL_RULES` (default: all) plus the report-only
        :data:`REPORT_RULES`.
    :param engine: ``"interval"`` or ``"crown"``. ``"crown"`` re-tries the rules the intervals could
        not decide with CROWN bounds of the operands (models of at most ``max_crown_nodes`` nodes).
    :param allow_interface_change: allow ``narrow_input`` to be marked applicable.
    :param margin: demand this absolute gap between a proven bound and the threshold it must clear
        (guards against float32 rounding at the edge of a range).
    :param check_unconditional: also analyse with every input unbounded to mark rewrites that need no
        precondition (doubles the analysis cost; without it nothing is marked unconditional except the
        constant-index narrowing).
    :param input_shapes: ``{input: [dim, ...]}`` forwarded to :func:`onnxsim.interval.propagate`; pin a
        dynamic batch dimension (``{"x": [1, 3, 224, 224]}``) to get per-element intervals instead of
        a value hull.
    :param max_crown_elements: ``engine="crown"`` only asks CROWN about operands with at most this
        many elements (its coefficient matrices are dense).
    """
    if engine not in ("interval", "crown"):
        raise ValueError('engine must be "interval" or "crown"')
    want = set(rules) if rules is not None else set(ALL_RULES) | set(REPORT_RULES)
    unknown = want - set(ALL_RULES) - set(REPORT_RULES)
    if unknown:
        raise ValueError(f"unknown rule(s): {sorted(unknown)}")
    f = _Facts(model)
    merged = _merged_ranges(model, input_ranges)
    res = result or _interval.propagate(model, merged, input_shapes)
    H_cond = _hull_fn(res)
    H_free: HullFn = (
        _hull_fn(_interval.propagate(model, _free_ranges(model), input_shapes))
        if check_unconditional
        else (lambda _n: None)
    )

    def run_node_rules(H: HullFn, skip: Set[int]) -> Dict[int, Rewrite]:
        found: Dict[int, Rewrite] = {}
        for idx, node in enumerate(f.nodes):
            if idx in skip or idx in found:
                continue
            for rule, fn in _NODE_RULES.items():
                if rule not in want:
                    continue
                rw = fn(f, idx, node, H, margin)
                if rw is not None and rw.rule in want:
                    found[idx] = rw
                    break
        return found

    found = run_node_rules(H_cond, set())
    if engine == "crown":
        names = sorted(
            {
                n
                for idx, node in enumerate(f.nodes)
                if idx not in found
                for n in _operand_names(f, node)
                if n in res.intervals
                and (f.numel(n) or max_crown_elements + 1) <= max_crown_elements
            }
        )
        tight = _tight_hulls(model, merged, names, max_crown_nodes)
        if tight:

            def H_tight(name: str, _t=tight) -> Optional[Hull]:
                h = H_cond(name)
                c = _t.get(name)
                if c is None:
                    return h
                if h is None:
                    return c
                return (max(h[0], c[0]), min(h[1], c[1]))

            for idx, rw in run_node_rules(H_tight, set(found)).items():
                rw.proof["engine"] = "crown"
                found[idx] = rw
    out: List[Rewrite] = []
    for idx, rw in sorted(found.items()):
        if "engine" not in rw.proof:
            rw.proof["engine"] = "interval"
        rw.unconditional = _rule_holds_unbounded(f, rw, H_free, margin, want)
        out.append(rw)
    if "narrow_const_index" in want:
        out.extend(_rule_narrow_const_index(f))
    if "narrow_input" in want:
        for rw in _rule_narrow_input(f, H_cond, allow_interface_change):
            rw.unconditional = bool(_rule_narrow_input(f, H_free, True))
            out.append(rw)
    if "fp16_risk" in want:
        rep = _report_fp16(f, H_cond)
        if rep is not None:
            out.append(rep)
    if "int64_fits_int32" in want:
        rep = _report_int64(f, H_cond)
        if rep is not None:
            out.append(rep)
    return out


def _rule_holds_unbounded(
    f, rw: Rewrite, H_free: HullFn, margin: float, want: Set[str]
) -> bool:
    if rw.index < 0 or rw.rule in ("narrow_const_index", "narrow_input"):
        return rw.unconditional
    node = f.nodes[rw.index]
    fn = _NODE_RULES.get(rw.rule)
    if fn is None:
        return False
    again = fn(f, rw.index, node, H_free, margin)
    # the SAME rewrite must be provable with every input unbounded: a weaker one under the same
    # rule name (dropping one Clip bound instead of removing the whole Clip) does not count
    return again is not None and again.rule == rw.rule and again.action == rw.action


# ------------------------------------------------------------------------------------------
# Applying
# ------------------------------------------------------------------------------------------


def _rename_refs(node: onnx.NodeProto, resolve: Callable[[str], str]) -> None:
    for k, x in enumerate(node.input):
        if x:
            r = resolve(x)
            if r != x:
                node.input[k] = r
    for a in node.attribute:
        graphs = [a.g] if a.HasField("g") else []
        graphs.extend(a.graphs)
        for g in graphs:
            for n in g.node:
                _rename_refs(n, resolve)
            for o in g.output:
                r = resolve(o.name)
                if r != o.name and o.name not in {x for n in g.node for x in n.output}:
                    o.name = r


def _all_reads(nodes: Iterable[onnx.NodeProto]) -> Set[str]:
    used: Set[str] = set()
    for n in nodes:
        used.update(x for x in n.input if x)
        used.update(_subgraph_inputs(n))
    return used


class _Editor:
    def __init__(self, model: onnx.ModelProto):
        self.model = copy.deepcopy(model)
        self.g = self.model.graph
        self.nodes = list(self.g.node)
        self.outputs = {o.name for o in self.g.output}
        self.alias: Dict[str, str] = {}
        self.deleted: Set[int] = set()
        self.replace: Dict[int, List[onnx.NodeProto]] = {}
        self.new_inits: List[onnx.TensorProto] = []
        self.candidates: Set[str] = set()

    def resolve(self, name: str) -> str:
        seen = 0
        while name in self.alias and seen < 10_000:
            name = self.alias[name]
            seen += 1
        return name

    def _orphan(self, node: onnx.NodeProto, keep: Optional[str] = None) -> None:
        self.candidates.update(x for x in node.input if x and x != keep)

    def bypass(self, idx: int, src: str) -> None:
        node = self.nodes[idx]
        y = node.output[0]
        s = self.resolve(src)
        self._orphan(node, keep=s)
        if y in self.outputs:
            self.replace[idx] = [
                helper.make_node(
                    "Identity",
                    [s],
                    [y],
                    name=(node.name or f"{node.op_type}_{idx}") + "_bypass",
                )
            ]
        else:
            self.alias[y] = s
            self.deleted.add(idx)

    def to_neg(self, idx: int) -> None:
        n = self.nodes[idx]
        self.replace[idx] = [
            helper.make_node("Neg", [n.input[0]], [n.output[0]], name=n.name)
        ]

    def clip_drop(self, idx: int, which: str, kind: str) -> None:
        n = onnx.NodeProto()
        n.CopyFrom(self.nodes[idx])
        if kind == "attrs":
            for a in list(n.attribute):
                if a.name == which:
                    n.attribute.remove(a)
        else:
            pos = 1 if which == "min" else 2
            if pos == 2:
                del n.input[2:]
            else:
                n.input[1] = ""
        self.replace[idx] = [n]

    def inline_if(self, idx: int, taken: bool) -> None:
        node = self.nodes[idx]
        br = {a.name: a for a in node.attribute}[
            "then_branch" if taken else "else_branch"
        ].g
        prefix = f"{node.name or 'If'}_{idx}_{'then' if taken else 'else'}__"
        defined = {t.name for t in br.initializer} | {
            o for n in br.node for o in n.output if o
        }
        ren = lambda x: (prefix + x) if x in defined else x  # noqa: E731
        new: List[onnx.NodeProto] = []
        for n in br.node:
            c = onnx.NodeProto()
            c.CopyFrom(n)
            for k, x in enumerate(c.input):
                if x:
                    c.input[k] = ren(x)
            for k, o in enumerate(c.output):
                if o:
                    c.output[k] = ren(o)
            if c.name:
                c.name = prefix + c.name
            new.append(c)
        for t in br.initializer:
            c = onnx.TensorProto()
            c.CopyFrom(t)
            c.name = ren(t.name)
            self.new_inits.append(c)
        self._orphan(node)
        for o_if, o_br in zip(node.output, br.output):
            src = ren(o_br.name)
            if o_if in self.outputs:
                new.append(
                    helper.make_node(
                        "Identity", [src], [o_if], name=prefix + "out_" + o_if
                    )
                )
            else:
                self.alias[o_if] = src
        self.replace[idx] = new

    def retype_const(self, name: str, origin: Tuple[str, int]) -> None:
        kind, k = origin
        if kind == "init":
            t = self.g.initializer[k]
            arr = np.clip(numpy_helper.to_array(t), -INT32_MAX, INT32_MAX).astype(
                np.int32
            )
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
        else:
            n = self.nodes[k]
            for a in n.attribute:
                if a.name == "value":
                    arr = np.clip(
                        numpy_helper.to_array(a.t), -INT32_MAX, INT32_MAX
                    ).astype(np.int32)
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
        for vi in list(self.g.value_info) + list(self.g.input):
            if vi.name == name:
                vi.type.tensor_type.elem_type = TensorProto.INT32

    def retype_input(self, name: str) -> None:
        for vi in self.g.input:
            if vi.name == name:
                vi.type.tensor_type.elem_type = TensorProto.INT32

    def finish(self) -> onnx.ModelProto:
        def copy_of(n: onnx.NodeProto) -> onnx.NodeProto:
            c = onnx.NodeProto()
            c.CopyFrom(n)
            return c

        final: List[onnx.NodeProto] = []
        for i, n in enumerate(self.nodes):
            if i in self.replace:
                final.extend(copy_of(r) for r in self.replace[i])
            elif i not in self.deleted:
                final.append(copy_of(n))
        for n in final:
            _rename_refs(n, self.resolve)
        # prune only what our rewrites orphaned
        cand = {self.resolve(c) for c in self.candidates} | set(self.candidates)
        changed = True
        while changed:
            changed = False
            used = _all_reads(final) | self.outputs
            for n in list(final):
                outs = [o for o in n.output if o]
                if outs and all(o in cand and o not in used for o in outs):
                    final.remove(n)
                    cand.update(x for x in n.input if x)
                    changed = True
        del self.g.node[:]
        self.g.node.extend(final)
        self.g.initializer.extend(self.new_inits)
        live = (
            {o for n in final for o in n.output}
            | {t.name for t in self.g.initializer}
            | {i.name for i in self.g.input}
        )
        keep = [
            vi
            for vi in self.g.value_info
            if vi.name in live and vi.name not in self.alias
        ]
        del self.g.value_info[:]
        self.g.value_info.extend(keep)
        return self.model


def apply(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    rules: Optional[Sequence[str]] = None,
    engine: str = "interval",
    allow_interface_change: bool = False,
    margin: float = 0.0,
    max_passes: int = 4,
    input_shapes: Optional[Dict[str, Sequence[Any]]] = None,
    check_unconditional: bool = True,
) -> Tuple[onnx.ModelProto, List[Dict[str, Any]]]:
    """Apply the rewrites the declared ranges justify and return ``(new_model, log)``.

    Explicit opt-in: raises ``ValueError`` when neither ``input_ranges`` nor the model's
    ``onnxsim.range.*`` annotations give any range. The new model's ``metadata_props`` carry the
    box it relies on (``onnxsim.precondition.range.<input>``) and the log (``onnxsim.range_opt.log``);
    ``log`` lists the applied rewrites with their proofs. Report-only findings (``fp16_risk``,
    ``int64_fits_int32``) appear in the log with ``applied: false``. See :func:`analyze` for arguments.
    """
    merged = _merged_ranges(model, input_ranges)
    if not merged:
        raise ValueError(
            "range_opt.apply needs at least one input range (input_ranges=... or an onnxsim.range.* "
            "annotation): every rewrite it makes is valid only inside a declared box"
        )
    current = model
    log: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    for pass_no in range(max_passes):
        props = analyze(
            current,
            merged,
            rules=rules,
            engine=engine,
            allow_interface_change=allow_interface_change,
            margin=margin,
            input_shapes=input_shapes,
            check_unconditional=check_unconditional,
        )
        if pass_no == 0:
            reports = [dict(r.as_dict(), applied=False) for r in props if not r.applies]
        todo = [r for r in props if r.applies and r.action]
        if not todo:
            break
        ed = _Editor(current)
        f = _Facts(current)
        touched: Set[int] = set()
        done: List[Rewrite] = []
        for rw in todo:
            if rw.index in touched and rw.action[0] not in (
                "retype_const",
                "retype_input",
            ):
                continue
            kind = rw.action[0]
            if kind == "bypass":
                ed.bypass(rw.index, rw.action[1])
            elif kind == "bypass_index":
                ed.bypass(rw.action[1], rw.action[2])
            elif kind == "to_neg":
                ed.to_neg(rw.index)
            elif kind == "clip_drop":
                ed.clip_drop(rw.index, rw.action[1], rw.action[2])
            elif kind == "inline_if":
                ed.inline_if(rw.index, rw.action[1])
            elif kind == "retype_const":
                ed.retype_const(rw.action[1], f.const_origin[rw.action[1]])
            elif kind == "retype_input":
                ed.retype_input(rw.action[1])
            else:  # pragma: no cover
                continue
            touched.add(rw.index)
            done.append(rw)
        new = ed.finish()
        for rw in done:
            log.append(dict(rw.as_dict(), applied=True, pass_no=pass_no))
        current = new
        if not done:
            break
    out = current if current is not model else copy.deepcopy(model)
    _stamp(out, merged, log)
    return out, log + reports


def _bound_json(a: np.ndarray) -> Any:
    """JSON form of one side of a box; ``None`` when that side is entirely infinite (unbounded)."""
    if a.size and np.all(np.isinf(a)):
        return None
    return a.item() if a.ndim == 0 else a.tolist()


def _stamp(
    model: onnx.ModelProto,
    merged: Dict[str, Tuple[Any, Any]],
    log: List[Dict[str, Any]],
) -> None:
    keys = [p.key for p in model.metadata_props]
    for k in list(keys):
        if k.startswith(PRECONDITION_PREFIX) or k == LOG_KEY:
            for p in list(model.metadata_props):
                if p.key == k:
                    model.metadata_props.remove(p)
    if any(not r["unconditional"] for r in log):
        for name, (lo, hi) in merged.items():
            lo_a, hi_a = (
                np.asarray(lo, dtype=np.float64),
                np.asarray(hi, dtype=np.float64),
            )
            e = model.metadata_props.add()
            e.key = PRECONDITION_PREFIX + name
            e.value = json.dumps({"min": _bound_json(lo_a), "max": _bound_json(hi_a)})
    e = model.metadata_props.add()
    e.key = LOG_KEY
    e.value = json.dumps(
        [
            {
                k: r[k]
                for k in (
                    "rule",
                    "node",
                    "op_type",
                    "description",
                    "unconditional",
                    "interface_change",
                )
            }
            for r in log
        ]
    )


# ------------------------------------------------------------------------------------------
# Deployment guard
# ------------------------------------------------------------------------------------------


def preconditions(model: onnx.ModelProto) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The input boxes a model produced by :func:`apply` relies on (empty if it needs none)."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for p in model.metadata_props:
        if p.key.startswith(PRECONDITION_PREFIX):
            d = json.loads(p.value)
            lo = -math.inf if d.get("min") is None else d["min"]
            hi = math.inf if d.get("max") is None else d["max"]
            out[p.key[len(PRECONDITION_PREFIX) :]] = (
                np.asarray(lo, dtype=np.float64),
                np.asarray(hi, dtype=np.float64),
            )
    return out


def check_precondition(
    model: onnx.ModelProto,
    inputs: Dict[str, np.ndarray],
    raise_on_violation: bool = True,
) -> List[str]:
    """Check concrete ``inputs`` against the box ``model`` relies on.

    Returns one message per violated input; raises :class:`PreconditionViolation` instead if
    ``raise_on_violation``. A model that relies on no box (every rewrite unconditional) never violates.
    """
    problems = []
    for name, (lo, hi) in preconditions(model).items():
        if name not in inputs:
            continue
        v = np.asarray(inputs[name], dtype=np.float64)
        below = v < np.broadcast_to(lo, v.shape)
        above = v > np.broadcast_to(hi, v.shape)
        if below.any() or above.any():
            problems.append(
                f"input {name!r}: observed [{np.nanmin(v):.6g}, {np.nanmax(v):.6g}] leaves the declared "
                f"box [{np.min(lo):.6g}, {np.max(hi):.6g}] the rewrites rely on "
                f"({int(below.sum() + above.sum())} of {v.size} elements)"
            )
    if problems and raise_on_violation:
        raise PreconditionViolation("; ".join(problems))
    return problems


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------


def _parse_range(spec: str) -> Tuple[str, Optional[float], Optional[float]]:
    name, _, rest = spec.rpartition("=")
    if not name or "," not in rest:
        raise argparse.ArgumentTypeError(f"expected NAME=LO,HI, got {spec!r}")
    lo, hi = rest.split(",", 1)
    cv = lambda s: None if s.strip().lower() in ("none", "") else float(s)  # noqa: E731
    return name, cv(lo), cv(hi)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m onnxsim.range_opt",
        description="Range-driven simplification: analyze (default) or --apply. Opt-in; every rewrite "
        "is valid only inside the declared --range box, which is recorded in the output model.",
    )
    ap.add_argument("model")
    ap.add_argument(
        "--range", action="append", default=[], type=_parse_range, metavar="NAME=LO,HI"
    )
    ap.add_argument(
        "--apply", action="store_true", help="write the rewritten model (needs -o)"
    )
    ap.add_argument("-o", "--output")
    ap.add_argument("--rules", help="comma-separated rule names")
    ap.add_argument("--engine", choices=("interval", "crown"), default="interval")
    ap.add_argument("--allow-interface-change", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args(argv)
    model = onnx.load(args.model)
    rng = {
        n: (-math.inf if lo is None else lo, math.inf if hi is None else hi)
        for n, lo, hi in args.range
    }
    rules = args.rules.split(",") if args.rules else None
    if args.apply:
        if not args.output:
            ap.error("--apply needs -o/--output")
        new, log = apply(
            model,
            rng,
            rules=rules,
            engine=args.engine,
            allow_interface_change=args.allow_interface_change,
            margin=args.margin,
        )
        onnx.save(new, args.output)
        for entry in log:
            flag = "applied" if entry["applied"] else "report "
            cond = (
                "informational"
                if not entry["applied"]
                else ("unconditional" if entry["unconditional"] else "needs box")
            )
            print(
                f"[{flag}] {entry['rule']:<20} {entry['node']}: {entry['description']} ({cond})"
            )
        print(
            f"wrote {args.output}: {sum(1 for entry in log if entry['applied'])} rewrite(s)"
        )
        return 0
    for rw in analyze(
        model,
        rng,
        rules=rules,
        engine=args.engine,
        allow_interface_change=args.allow_interface_change,
        margin=args.margin,
    ):
        flag = "apply " if rw.applies else "report"
        cond = (
            "informational"
            if not rw.applies
            else ("unconditional" if rw.unconditional else "needs box")
        )
        print(f"[{flag}] {rw.rule:<20} {rw.node}: {rw.description} ({cond})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
