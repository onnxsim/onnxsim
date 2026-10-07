"""Certified memory and compute bounds for models with dynamic or data-dependent shapes.

``onnxsim.memory_planning`` and ``onnxsim.model_info`` need static shapes or dim-symbol
polynomials. A model with a bounded dynamic batch (``batch <= 8``), a bounded sequence
(``seq <= 2048``) or a data-dependent op (``NonZero``, ``NonMaxSuppression``, ...) gets no
certified answer from them, although the question a deployment target asks is exactly
"does it provably fit?". :func:`bounds` answers it with **intervals**: for every tensor a
range of element counts and bytes, and for the whole graph a range of

* ``peak_live_bytes`` -- weights resident plus the activations live at once, with
  :mod:`onnxsim.model_info`'s liveness convention (live from production to last use, graph
  outputs to the end, an unconsumed tensor stays live);
* ``macs`` / ``flops`` -- the multiply-accumulates of the compute-dominant operators
  (``Conv``, ``ConvTranspose``, ``Gemm``, ``MatMul``, ``Attention`` and the quantized twins),
  ``flops = 2 * macs``;
* ``mem_access_bytes`` -- every node's inputs (weights included) plus outputs read/written;
* ``arena`` -- a static byte-offset allocation (below).

Shapes come from :func:`onnxsim.interval.propagate` (ranged shapes, shape arithmetic through
``Shape -> Gather -> Concat -> Reshape`` chains, the ``NonZero``/``TopK``/``Compress``/
``Unique``/``NonMaxSuppression`` rules of :mod:`onnxsim.shape_ranges`), extended with
interval shape rules for the ops it has no ranged rule for (``Conv``, pooling, ``Gemm``,
norms, ``Resize``, ``Split``, ``Einsum``, ...), and intersected with what ONNX's own shape
inference states. An op with no rule, a tensor whose dtype is unknown, a dimension without a
range, or a control-flow subgraph makes the affected quantity **unbounded with a note**; it
is never guessed, and a total that depends on one is unbounded too.

What each bound means
---------------------

*Per tensor and per node*: every valid execution with all dynamic dims inside the given
ranges produces tensors whose element counts lie in ``[lo, hi]``. The upper bound is attained
only if every dim sits at its upper bound at once (independent dims; a dim named by the same
``dim_param`` in two inputs is one number at run time, which the bound does not use).

*``peak_live_bytes``*: ``max_t sum_{i live at t} size_i`` is monotone in every size, and which
tensors are live at step ``t`` depends only on the graph and its (fixed) node order, not on
sizes. Evaluating it at the upper (lower) sizes therefore bounds the same quantity for every
shape in range. It is the peak of an *ideal* allocator with zero fragmentation; a real
allocator can need more (that is what ``arena`` is for).

*``arena``*: ``onnxsim.memory_planning``'s greedy planner is **not monotone in sizes** (see
``docs/shape-cost.md``: arena at smaller shapes exceeded arena at larger ones in 3 of 3,000
random graphs), so "plan at the upper-bound shapes" is *not* claimed to bound what the planner
would return at other shapes. What is claimed is smaller and checkable: the plan computed at
the upper-bound shapes assigns every tensor a slot of its *upper-bound* size, so it is a valid
static allocation for **every** shape in range, provided no two tensors that are live together
overlap in the arena. That is verified here independently of the planner (``ArenaPlan.verified``);
a plan that fails the check is dropped with a note.

Soundness is conditional on the model being valid for the sampled dims (an ONNX model that
errors at run time has no execution to bound), on the ranges being honest, and on
:mod:`onnxsim.interval`'s shape rules (cross-checked against onnxruntime in
``tests/test_shape_cost.py``).
"""

import argparse
import dataclasses
import json
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
from onnx import helper

from . import interval as _interval
from . import shape_ranges as _sr
from ._onnx_compat import INT4, UINT4
from .shape_ranges import Dim, Shape

_INF = float("inf")
_FLOAT4_E2M1 = 23  # TensorProto.FLOAT4E2M1 (not named here: absent from older onnx)
_FOUR_BIT = {UINT4, INT4, _FLOAT4_E2M1}


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Bound:
    """An inclusive integer range; ``hi=None`` means unbounded above."""

    lo: int
    hi: Optional[int]

    @property
    def bounded(self) -> bool:
        return self.hi is not None

    def __str__(self) -> str:
        return f"[{self.lo}, {'inf' if self.hi is None else self.hi}]"

    def __add__(self, other: "Bound") -> "Bound":
        return Bound(
            self.lo + other.lo,
            None if self.hi is None or other.hi is None else self.hi + other.hi,
        )

    def scale(self, k: int) -> "Bound":
        return Bound(self.lo * k, None if self.hi is None else self.hi * k)


ZERO = Bound(0, 0)
UNBOUNDED = Bound(0, None)


@dataclasses.dataclass
class TensorBound:
    name: str
    kind: str  # "input" | "weight" | "activation" | "output"
    shape: Optional[Shape]
    bits_per_element: Optional[int]
    numel: Bound
    bytes: Bound
    note: str = ""


@dataclasses.dataclass
class NodeCost:
    name: str
    op_type: str
    macs: Optional[
        Bound
    ]  # None: not a MAC op; UNBOUNDED when it is but cannot be bounded
    mem_access: Bound


@dataclasses.dataclass
class ArenaPlan:
    """A static allocation valid for every shape in range (see the module docstring)."""

    arena_bytes: int
    naive_bytes: int
    tensor_offsets: Dict[
        str, Tuple[int, int]
    ]  # name -> (offset, size at the upper bound)
    verified: bool
    note: str = ""


@dataclasses.dataclass
class CostBounds:
    tensors: Dict[str, TensorBound]
    nodes: List[NodeCost]
    weight_bytes: int
    peak_live_bytes: Bound  # weights + live activations (model_info convention)
    peak_activation_bytes: Bound  # live activations only
    macs: Bound
    mem_access_bytes: Bound
    arena: Optional[ArenaPlan]
    unbounded_tensors: List[str]
    notes: List[str]
    complete: bool  # False: a control-flow subgraph or similar was not analysed

    @property
    def flops(self) -> Bound:
        return self.macs.scale(2)

    def summary(self) -> Dict[str, Any]:
        def b(x: Bound) -> Dict[str, Any]:
            return {"lo": x.lo, "hi": x.hi}

        return {
            "weight_bytes": self.weight_bytes,
            "peak_live_bytes": b(self.peak_live_bytes),
            "peak_activation_bytes": b(self.peak_activation_bytes),
            "arena_bytes": None if self.arena is None else self.arena.arena_bytes,
            "arena_verified": None if self.arena is None else self.arena.verified,
            "macs": b(self.macs),
            "flops": b(self.flops),
            "mem_access_bytes": b(self.mem_access_bytes),
            "complete": self.complete,
            "unbounded_tensors": list(self.unbounded_tensors),
            "notes": list(self.notes),
        }


@dataclasses.dataclass
class BudgetCheck:
    metric: str
    limit: int
    bound: Bound
    status: str  # "proved" (hi <= limit) | "exceeds" (lo > limit) | "unknown"


@dataclasses.dataclass
class BudgetVerdict:
    checks: List[BudgetCheck]
    fits: Optional[
        bool
    ]  # True: every limit proved; False: some limit provably exceeded
    notes: List[str]

    def __str__(self) -> str:
        head = {True: "FITS", False: "DOES NOT FIT", None: "UNKNOWN"}[self.fits]
        lines = [f"budget: {head}"]
        for c in self.checks:
            lines.append(
                f"  {c.metric}: {c.status}  (bound {c.bound}, limit {c.limit})"
            )
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _const(iv: Dict[str, Any], node: onnx.NodeProto, idx: int) -> Optional[np.ndarray]:
    """The constant value of an operand (a point interval), or ``None``."""
    if idx >= len(node.input) or not node.input[idx]:
        return None
    v = iv.get(node.input[idx])
    if v is None or not np.array_equal(v[0], v[1]):
        return None
    return np.asarray(v[0])


def _ints(a: Optional[np.ndarray]) -> Optional[List[int]]:
    if a is None or not np.all(np.isfinite(a)):
        return None
    return [int(x) for x in np.asarray(a).reshape(-1)]


def _mono(d: Dim, f: Callable[[int], int]) -> Dim:
    """Apply a non-decreasing integer function to a dimension range."""
    return Dim(max(0, f(d.lo)), None if d.hi is None else max(0, f(d.hi)))


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _bits(elem_type: int) -> Optional[int]:
    """Bits per element, or ``None`` for types with no fixed width (strings)."""
    if elem_type in _FOUR_BIT:
        return 4
    try:
        dt = helper.tensor_dtype_to_np_dtype(elem_type)
    except Exception:
        return None
    if dt.kind in ("O", "U", "S", "V"):
        return None
    return int(dt.itemsize) * 8


def _bytes_of(numel: Bound, bits: Optional[int]) -> Bound:
    if bits is None:
        return UNBOUNDED
    return Bound(
        (numel.lo * bits) // 8,
        None if numel.hi is None else _ceil_div(numel.hi * bits, 8),
    )


def _numel_bound(shape: Optional[Shape]) -> Bound:
    if shape is None:
        return UNBOUNDED
    n = _sr.numel(shape)
    return Bound(n.lo, n.hi)


# --------------------------------------------------------------------------
# Interval shape rules for ops interval.propagate has no ranged rule for
# --------------------------------------------------------------------------

Rule = Callable[
    [onnx.NodeProto, List[Optional[Shape]], Dict[str, Any]], List[Optional[Shape]]
]
_RULES: Dict[str, Rule] = {}


def _rule(*ops: str) -> Callable[[Rule], Rule]:
    def deco(fn: Rule) -> Rule:
        for o in ops:
            _RULES[o] = fn
        return fn

    return deco


_SAME_SHAPE_UNARY = (
    "Relu LeakyRelu PRelu Elu Selu Celu Gelu Sigmoid Tanh Softplus Softsign HardSigmoid HardSwish "
    "Mish Swish Erf Exp Log Sqrt Reciprocal Neg Abs Sign Ceil Floor Round Sin Cos Tan Asin Acos "
    "Atan Sinh Cosh Asinh Acosh Atanh Not IsNaN IsInf Shrink ThresholdedRelu Clip Cast CastLike "
    "Identity Softmax LogSoftmax Hardmax BatchNormalization InstanceNormalization "
    "GroupNormalization LpNormalization LRN MeanVarianceNormalization QuantizeLinear "
    "DequantizeLinear CumSum Trilu Bernoulli RandomNormalLike RandomUniformLike "
    "SimplifiedLayerNormalization RMSNormalization BiasGelu FastGelu"
).split()


@_rule(*_SAME_SHAPE_UNARY)
def _r_same(node, ins, iv):
    # the first output keeps the input's shape; any extra output (training statistics) is unknown
    return [ins[0]] + [None] * (len(node.output) - 1)


@_rule("Dropout")
def _r_dropout(node, ins, iv):
    return [ins[0]] + [ins[0]] * (len(node.output) - 1)


@_rule("DynamicQuantizeLinear")
def _r_dql(node, ins, iv):
    return [ins[0], (), ()]


@_rule("LayerNormalization")
def _r_layernorm(node, ins, iv):
    x = ins[0]
    if x is None:
        return [None] * len(node.output)
    axis = int(_attrs(node).get("axis", -1))
    axis = axis + len(x) if axis < 0 else axis
    stat = tuple(x[:axis]) + tuple(_sr.exact(1) for _ in x[axis:])
    return [x, stat, stat][: len(node.output)]


@_rule("Pow")
def _r_broadcast(node, ins, iv):
    present = [i for i, name in zip(ins, node.input) if name]
    if not present or any(sh is None for sh in present):
        return [None]
    return [_sr.broadcast([sh for sh in present if sh is not None])]


for _op in (
    "Mod BitShift BitwiseAnd BitwiseOr BitwiseXor Xor And Or Equal Greater Less GreaterOrEqual "
    "LessOrEqual Sum Mean Where"
).split():
    _RULES[_op] = _r_broadcast


def _spatial_out(
    dims: Sequence[Dim],
    kernel: Sequence[int],
    strides: Sequence[int],
    dil: Sequence[int],
    pads: Sequence[int],
    auto_pad: str,
    ceil_mode: bool,
) -> List[Dim]:
    n = len(dims)
    out: List[Dim] = []
    for i in range(n):
        k_eff = dil[i] * (kernel[i] - 1) + 1
        s = max(1, strides[i])
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out.append(_mono(dims[i], _out_size_fn(0, s, True, same=True)))
            continue
        total = 0 if auto_pad == "VALID" else pads[i] + pads[i + n]
        out.append(_mono(dims[i], _out_size_fn(total - k_eff, s, ceil_mode)))
    return out


def _out_size_fn(
    offset: int, stride: int, ceil: bool, same: bool = False
) -> Callable[[int], int]:
    """Output size of a windowed op as a non-decreasing function of the input size.

    ``offset = pads - effective_kernel``: ``floor((n + offset) / stride) + 1`` (``ceil`` rounds the
    division up instead); ``same`` is ``ceil(n / stride)`` (auto_pad SAME_*).
    """

    def f(n: int) -> int:
        if same:
            return _ceil_div(n, stride)
        if ceil:
            return _ceil_div(n + offset, stride) + 1
        return (n + offset) // stride + 1

    return f


def _conv_like_params(
    node, x: Shape, kernel: Sequence[int]
) -> Tuple[List[int], List[int], List[int], str]:
    a = _attrs(node)
    n = len(x) - 2
    strides = [int(v) for v in a.get("strides", [1] * n)]
    dil = [int(v) for v in a.get("dilations", [1] * n)]
    pads = [int(v) for v in a.get("pads", [0] * (2 * n))]
    auto = a.get("auto_pad", b"NOTSET")
    auto = auto.decode() if isinstance(auto, bytes) else str(auto)
    return strides, dil, pads, auto


@_rule("Conv", "ConvInteger", "QLinearConv")
def _r_conv(node, ins, iv):
    w_idx = 3 if node.op_type == "QLinearConv" else 1
    x, w = ins[0], ins[w_idx] if len(ins) > w_idx else None
    if (
        x is None
        or w is None
        or len(x) < 3
        or len(w) != len(x)
        or not _sr.is_static(w[2:])
    ):
        return [None]
    kernel = list(_sr.static_dims(w[2:]))
    strides, dil, pads, auto = _conv_like_params(node, x, kernel)
    sp = _spatial_out(x[2:], kernel, strides, dil, pads, auto, False)
    return [(x[0], w[0]) + tuple(sp)]


@_rule("ConvTranspose")
def _r_convt(node, ins, iv):
    x, w = ins[0], ins[1] if len(ins) > 1 else None
    if (
        x is None
        or w is None
        or len(x) < 3
        or len(w) != len(x)
        or not _sr.is_static(w[2:])
    ):
        return [None]
    a = _attrs(node)
    group = int(a.get("group", 1))
    kernel = list(_sr.static_dims(w[2:]))
    n = len(x) - 2
    strides, dil, pads, _auto = _conv_like_params(node, x, kernel)
    if "output_shape" in a:
        sp = [_sr.exact(int(v)) for v in a["output_shape"]]
    else:
        opad = [int(v) for v in a.get("output_padding", [0] * n)]
        sp = [
            _mono(
                x[2 + i],
                lambda v, i=i: strides[i] * (v - 1)
                + opad[i]
                + (kernel[i] - 1) * dil[i]
                + 1
                - pads[i]
                - pads[i + n],
            )
            for i in range(n)
        ]
    if not w[1].exact:
        return [None]
    return [(x[0], _sr.exact(w[1].value * group)) + tuple(sp)]


@_rule("MaxPool", "AveragePool", "LpPool")
def _r_pool(node, ins, iv):
    x = ins[0]
    if x is None or len(x) < 3:
        return [None] * len(node.output)
    a = _attrs(node)
    kernel = [int(v) for v in a["kernel_shape"]]
    strides, dil, pads, auto = _conv_like_params(node, x, kernel)
    sp = _spatial_out(
        x[2:], kernel, strides, dil, pads, auto, bool(a.get("ceil_mode", 0))
    )
    y = (x[0], x[1]) + tuple(sp)
    return [y] + [y] * (len(node.output) - 1)


@_rule("GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool")
def _r_gpool(node, ins, iv):
    x = ins[0]
    if x is None or len(x) < 2:
        return [None]
    return [tuple(x[:2]) + tuple(_sr.exact(1) for _ in x[2:])]


@_rule("Gemm")
def _r_gemm(node, ins, iv):
    a, b = ins[0], ins[1] if len(ins) > 1 else None
    if a is None or b is None or len(a) != 2 or len(b) != 2:
        return [None]
    at = _attrs(node)
    m = a[1] if at.get("transA", 0) else a[0]
    n = b[0] if at.get("transB", 0) else b[1]
    return [(m, n)]


@_rule("MatMul", "MatMulInteger")
def _r_matmulint(node, ins, iv):
    return [
        _sr.matmul(ins[0], ins[1])
        if ins[0] is not None and ins[1] is not None
        else None
    ]


# Ops interval.propagate evaluates only while the *values* are finite; with unbounded values they
# fall through to here although their shapes are perfectly known (an attention MatMul, a reduction,
# an embedding Gather, ...). Without a rule the unknown output would cascade through the graph.


@_rule("Gather")
def _r_gather(node, ins, iv):
    if ins[0] is None or len(ins) < 2 or ins[1] is None:
        return [None]
    return [_sr.gather(ins[0], ins[1], int(_attrs(node).get("axis", 0)))]


@_rule("GatherND")
def _r_gather_nd(node, ins, iv):
    if ins[0] is None or len(ins) < 2 or ins[1] is None:
        return [None]
    return [_sr.gather_nd(ins[0], ins[1], int(_attrs(node).get("batch_dims", 0)))]


@_rule("Transpose")
def _r_transpose(node, ins, iv):
    if ins[0] is None:
        return [None]
    perm = _attrs(node).get("perm")
    return [_sr.transpose(ins[0], [int(p) for p in perm] if perm else None)]


@_rule("Concat")
def _r_concat(node, ins, iv):
    present = [s for s, name in zip(ins, node.input) if name]
    if not present or any(s is None for s in present):
        return [None]
    return [
        _sr.concat([s for s in present if s is not None], int(_attrs(node)["axis"]))
    ]


@_rule("Flatten")
def _r_flatten(node, ins, iv):
    if ins[0] is None:
        return [None]
    return [_sr.flatten(ins[0], int(_attrs(node).get("axis", 1)))]


@_rule("Squeeze", "Unsqueeze")
def _r_squeeze(node, ins, iv):
    if ins[0] is None:
        return [None]
    axes = _ints(_const(iv, node, 1)) if len(node.input) > 1 and node.input[1] else None
    if axes is None and "axes" in _attrs(node):
        axes = [int(a) for a in _attrs(node)["axes"]]
    if node.op_type == "Unsqueeze":
        return [_sr.unsqueeze(ins[0], axes)] if axes is not None else [None]
    return [_sr.squeeze(ins[0], axes)]


@_rule("QLinearMatMul")
def _r_qlmm(node, ins, iv):
    a, b = ins[0], ins[3] if len(ins) > 3 else None
    return [_sr.matmul(a, b) if a is not None and b is not None else None]


@_rule("Resize", "Upsample")
def _r_resize(node, ins, iv):
    x = ins[0]
    if x is None or "axes" in _attrs(node):
        return [None]
    n = len(x)
    sizes = _ints(_const(iv, node, 3)) if len(node.input) > 3 else None
    if sizes is not None and len(sizes) == n:
        return [tuple(_sr.exact(s) for s in sizes)]
    scales_idx = 1 if node.op_type == "Upsample" or len(node.input) == 2 else 2
    sc = _const(iv, node, scales_idx)
    if sc is None or sc.size != n:
        return [None]
    out = []
    for d, s in zip(x, np.asarray(sc, dtype=np.float64).reshape(-1)):
        if s < 0 or not np.isfinite(s):
            return [None]
        if float(s) == int(s):
            k = int(s)
            out.append(_mono(d, lambda v, k=k: v * k))
        else:  # float32 rounding of floor(in * scale): widen by one on both sides
            out.append(
                Dim(
                    max(0, int(math.floor(d.lo * s)) - 1),
                    None if d.hi is None else int(math.floor(d.hi * s)) + 1,
                )
            )
    return [tuple(out)]


@_rule("Split")
def _r_split(node, ins, iv):
    x = ins[0]
    nout = len([o for o in node.output])
    if x is None:
        return [None] * nout
    a = _attrs(node)
    axis = int(a.get("axis", 0))
    axis = axis + len(x) if axis < 0 else axis
    sizes = _ints(_const(iv, node, 1)) if len(node.input) > 1 else None
    if sizes is None and "split" in a:
        sizes = [int(v) for v in a["split"]]
    outs: List[Optional[Shape]] = []
    if sizes is not None and len(sizes) == nout:
        for s in sizes:
            outs.append(tuple(x[:axis]) + (_sr.exact(s),) + tuple(x[axis + 1 :]))
        return outs
    n = int(a.get("num_outputs", nout))
    d = x[axis]
    each = Dim(d.lo // n, None if d.hi is None else _ceil_div(d.hi, n))
    return [tuple(x[:axis]) + (each,) + tuple(x[axis + 1 :])] * nout


@_rule(
    "ReduceProd",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
)
def _r_reduce(node, ins, iv):
    x = ins[0]
    if x is None:
        return [None]
    a = _attrs(node)
    keep = bool(a.get("keepdims", 1))
    axes: Optional[List[int]]
    if len(node.input) > 1 and node.input[1]:
        axes = _ints(_const(iv, node, 1))
        if axes is None:
            return [None]
    else:
        axes = [int(v) for v in a["axes"]] if "axes" in a else None
    if axes is not None and len(axes) == 0 and a.get("noop_with_empty_axes", 0):
        return [x]
    return [_sr.reduce(x, axes, keep)]


@_rule("ArgMax", "ArgMin")
def _r_argmax(node, ins, iv):
    x = ins[0]
    if x is None:
        return [None]
    a = _attrs(node)
    axis = int(a.get("axis", 0))
    return [_sr.reduce(x, [axis], bool(a.get("keepdims", 1)))]


@_rule("ScatterND", "ScatterElements", "Scatter")
def _r_scatter(node, ins, iv):
    return [ins[0]]


@_rule("GatherElements")
def _r_gather_elements(node, ins, iv):
    return [ins[1] if len(ins) > 1 else None]


@_rule("DepthToSpace")
def _r_d2s(node, ins, iv):
    x = ins[0]
    if x is None or len(x) != 4:
        return [None]
    b = int(_attrs(node)["blocksize"])
    c = Dim(x[1].lo // (b * b), None if x[1].hi is None else _ceil_div(x[1].hi, b * b))
    return [(x[0], c, _mono(x[2], lambda v: v * b), _mono(x[3], lambda v: v * b))]


@_rule("SpaceToDepth")
def _r_s2d(node, ins, iv):
    x = ins[0]
    if x is None or len(x) != 4:
        return [None]
    b = int(_attrs(node)["blocksize"])
    return [
        (
            x[0],
            _mono(x[1], lambda v: v * b * b),
            Dim(x[2].lo // b, None if x[2].hi is None else _ceil_div(x[2].hi, b)),
            Dim(x[3].lo // b, None if x[3].hi is None else _ceil_div(x[3].hi, b)),
        )
    ]


@_rule("OneHot")
def _r_onehot(node, ins, iv):
    idx = ins[0]
    depth = _ints(_const(iv, node, 1))
    if idx is None or depth is None or len(depth) != 1:
        return [None]
    axis = int(_attrs(node).get("axis", -1))
    pos = axis + len(idx) + 1 if axis < 0 else axis
    return [tuple(idx[:pos]) + (_sr.exact(depth[0]),) + tuple(idx[pos:])]


@_rule("GridSample")
def _r_gridsample(node, ins, iv):
    x, g = ins[0], ins[1] if len(ins) > 1 else None
    if x is None or g is None or len(x) != 4 or len(g) != 4:
        return [None]
    return [(x[0], x[1], g[1], g[2])]


@_rule("RoiAlign")
def _r_roialign(node, ins, iv):
    x, rois = ins[0], ins[1] if len(ins) > 1 else None
    if x is None or rois is None or len(x) != 4 or len(rois) != 2:
        return [None]
    a = _attrs(node)
    return [
        (
            rois[0],
            x[1],
            _sr.exact(int(a.get("output_height", 1))),
            _sr.exact(int(a.get("output_width", 1))),
        )
    ]


@_rule("Einsum")
def _r_einsum(node, ins, iv):
    eq = _attrs(node).get("equation", b"")
    eq = (eq.decode() if isinstance(eq, bytes) else str(eq)).replace(" ", "")
    if "..." in eq or any(s is None for s in ins[: len(node.input)]):
        return [None]
    lhs, _, rhs = eq.partition("->")
    terms = lhs.split(",")
    if len(terms) != len(ins) or not rhs:
        return [None]
    size: Dict[str, Dim] = {}
    for term, shp in zip(terms, ins):
        if shp is None or len(term) != len(shp):
            return [None]
        for ch, d in zip(term, shp):
            # a label of size 1 broadcasts; otherwise take the hull of the stated sizes
            size[ch] = d if ch not in size else _sr.hull(size[ch], d)
    return [tuple(size[ch] for ch in rhs)]


def shape_fallback(
    node: onnx.NodeProto, in_shapes: List[Optional[Shape]], iv: Dict[str, Any]
) -> List[Optional[Shape]]:
    """Interval shape rule for ``node`` (the hook :func:`onnxsim.interval.propagate` accepts).

    Returns ``[None] * n_outputs`` for an op without a rule: the outputs stay unknown.
    """
    if node.domain not in ("", "ai.onnx"):
        return [None] * len(node.output)
    rule = _RULES.get(node.op_type)
    if rule is None:
        return [None] * len(node.output)
    out = rule(node, in_shapes, iv)
    return list(out) + [None] * (len(node.output) - len(out))


# --------------------------------------------------------------------------
# Compute (MACs) per node, model_info conventions, on interval shapes
# --------------------------------------------------------------------------

_MAC_OPS = {
    "Conv", "ConvInteger", "QLinearConv", "ConvTranspose", "Gemm", "MatMul", "MatMulInteger",
    "QLinearMatMul", "Attention",
}  # fmt: skip


def _dim_bound(d: Dim) -> Bound:
    return Bound(d.lo, d.hi)


def _prod_dims(ds: Sequence[Dim]) -> Dim:
    out = _sr.exact(1)
    for d in ds:
        out = out * d
    return out


def _node_macs(
    node: onnx.NodeProto, shape_of: Callable[[str], Optional[Shape]]
) -> Optional[Bound]:
    t = node.op_type
    if t not in _MAC_OPS:
        return None
    if node.domain not in ("", "ai.onnx"):
        return None

    def sh(i: int) -> Optional[Shape]:
        return (
            shape_of(node.input[i]) if i < len(node.input) and node.input[i] else None
        )

    out = shape_of(node.output[0]) if node.output and node.output[0] else None
    try:
        if t in ("Conv", "ConvInteger", "QLinearConv"):
            w = sh(3 if t == "QLinearConv" else 1)
            if w is None or out is None or len(w) < 2:
                return UNBOUNDED
            return _dim_bound(_prod_dims(out) * w[1] * _prod_dims(w[2:]))
        if t == "ConvTranspose":
            x, w = sh(0), sh(1)
            if x is None or w is None or len(w) < 2:
                return UNBOUNDED
            return _dim_bound(_prod_dims(x) * w[1] * _prod_dims(w[2:]))
        if t == "Gemm":
            a, b = sh(0), sh(1)
            if a is None or b is None or len(a) != 2 or len(b) != 2:
                return UNBOUNDED
            at = _attrs(node)
            m, k = (a[1], a[0]) if at.get("transA", 0) else (a[0], a[1])
            n = b[0] if at.get("transB", 0) else b[1]
            return _dim_bound(m * n * k)
        if t in ("MatMul", "MatMulInteger", "QLinearMatMul"):
            a = sh(0)
            if a is None or out is None or len(a) == 0:
                return UNBOUNDED
            return _dim_bound(_prod_dims(out) * a[-1])
        if t == "Attention":
            return _macs_attention(node, sh(0), sh(1), sh(2))
    except Exception:
        return UNBOUNDED
    return UNBOUNDED


def _macs_attention(
    node: onnx.NodeProto,
    q: Optional[Shape],
    k: Optional[Shape],
    v: Optional[Shape],
) -> Bound:
    """QK^T plus PV of ``Attention`` (opset 23), model_info's formula, on interval shapes."""
    if q is None or k is None or v is None:
        return UNBOUNDED
    if len(q) == 4 and len(k) == 4 and len(v) == 4:
        batch, heads, sq, dq = q
        skv, dv = k[2], v[3]
    elif len(q) == 3 and len(k) == 3 and len(v) == 3:
        at = _attrs(node)
        qh, kvh = int(at.get("q_num_heads", 0)), int(at.get("kv_num_heads", 0))
        if qh <= 0 or kvh <= 0 or not k[2].exact or not v[2].exact:
            return UNBOUNDED
        batch, sq, heads = q[0], q[1], _sr.exact(qh)
        skv = k[1]
        dq, dv = _sr.exact(k[2].value // kvh), _sr.exact(v[2].value // kvh)
    else:
        return UNBOUNDED
    base = batch * heads * sq * skv
    return _dim_bound(base * dq + base * dv)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _has_subgraph(node: onnx.NodeProto) -> bool:
    return any(a.HasField("g") or len(a.graphs) > 0 for a in node.attribute)


def _input_specs(
    model: onnx.ModelProto,
    dim_ranges: Dict[str, Any],
    overrides: Dict[str, Sequence[Any]],
    notes: List[str],
) -> Dict[str, Sequence[Any]]:
    """Per graph input, a dim list for :func:`onnxsim.interval.propagate`'s ``input_shapes``."""
    inits = {t.name for t in model.graph.initializer}
    specs: Dict[str, Sequence[Any]] = {}
    unconstrained: Set[str] = set()
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        if vi.name in overrides:
            specs[vi.name] = list(overrides[vi.name])
            continue
        if not (
            vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape")
        ):
            notes.append(
                f"input {vi.name!r} has no declared shape: its tensors are unbounded"
            )
            continue
        dims: List[Any] = []
        for i, d in enumerate(vi.type.tensor_type.shape.dim):
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(int(d.dim_value))
            elif d.dim_param and d.dim_param in dim_ranges:
                r = dim_ranges[d.dim_param]
                dims.append(
                    int(r)
                    if isinstance(r, (int, np.integer))
                    else (int(r[0]), None if r[1] is None else int(r[1]))
                )
            else:
                dims.append(None)
                unconstrained.add(d.dim_param or f"{vi.name}[{i}]")
        specs[vi.name] = dims
    if unconstrained:
        notes.append(
            "dimension(s) without a range, treated as unbounded: "
            + ", ".join(sorted(unconstrained))
        )
    return specs


def _onnx_static_dims(
    model: onnx.ModelProto, overridden: Optional[Dict[str, Sequence[Any]]] = None
) -> Dict[str, List[Optional[int]]]:
    """Dims ONNX shape inference states exactly (``None`` where it does not).

    ONNX infers from the *declared* input shapes. When the caller overrides an input's dims
    (``input_shapes=``), those declarations no longer describe the analysed model, so inference runs
    on a copy whose inputs carry the overridden dims (a range becomes an unnamed-symbol dimension);
    otherwise a model exported with static shapes would silently clamp every tensor back to them.
    """
    if overridden:
        model = _interval.declare_input_dims(model, overridden)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True).graph
    except Exception:
        return {}
    out: Dict[str, List[Optional[int]]] = {}
    for vi in list(inferred.value_info) + list(inferred.output):
        tt = vi.type.tensor_type
        if vi.type.HasField("tensor_type") and tt.HasField("shape"):
            out[vi.name] = [
                int(d.dim_value)
                if d.HasField("dim_value") and d.dim_value > 0
                else None
                for d in tt.shape.dim
            ]
    return out


def _elem_types(model: onnx.ModelProto) -> Dict[str, int]:
    try:
        g = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        g = model.graph
    types: Dict[str, int] = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type:
            types[vi.name] = int(vi.type.tensor_type.elem_type)
    for t in model.graph.initializer:
        types[t.name] = int(t.data_type)
    return types


def _tighten(
    shape: Optional[Shape], static: Optional[List[Optional[int]]]
) -> Optional[Shape]:
    """Intersect an interval-derived shape with the dims ONNX inference states exactly."""
    if shape is None or static is None or len(static) != len(shape):
        return shape
    out = []
    for d, s in zip(shape, static):
        if s is None:
            out.append(d)
            continue
        j = _sr.intersect(d, _sr.exact(s))
        out.append(
            j if j is not None else d
        )  # contradiction: keep the sound interval shape
    return tuple(out)


def bounds(
    model: Union[onnx.ModelProto, str],
    dim_ranges: Optional[Dict[str, Any]] = None,
    input_shapes: Optional[Dict[str, Sequence[Any]]] = None,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    plan_arena: bool = True,
) -> CostBounds:
    """Certified bounds on memory and compute for every shape allowed by the ranges.

    :param dim_ranges: ``{dim_param: (lo, hi)}`` (or an ``int``, or ``(lo, None)``) for the
        dynamic dimensions of the graph inputs, e.g. ``{"batch": (1, 8), "seq": (1, 2048)}``.
        A dimension with neither a static size nor a range is unbounded.
    :param input_shapes: per-input dim lists (``int``, ``(lo, hi)`` or ``None``), overriding
        the declaration, as in :func:`onnxsim.interval.propagate`.
    :param input_ranges: ``{input: (lo, hi)}`` *value* ranges, which matter for data-dependent
        shapes (a ``TopK`` ``K`` input, a ``NonZero`` over an input known to be positive).
    :param plan_arena: also compute (and verify) the static arena plan at the upper bounds.
    """
    if isinstance(model, str):
        model = onnx.load(model)
    notes: List[str] = []
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    specs = _input_specs(model, dict(dim_ranges or {}), dict(input_shapes or {}), notes)

    # Value ranges: unbounded unless given. Passing explicit infinite ranges also keeps the
    # value analysis on its cheap path (no reference-evaluator convolutions) even when the
    # model carries onnxsim.range.* annotations; the annotations are not needed for shapes.
    vr: Dict[str, Tuple[Any, Any]] = {}
    for vi in g.input:
        if vi.name not in inits:
            vr[vi.name] = (-np.inf, np.inf)
    for k, v in (input_ranges or {}).items():
        vr[k] = v

    res = _interval.propagate(model, vr, specs, shape_fallback=shape_fallback)
    static = _onnx_static_dims(model, dict(input_shapes or {}))
    etypes = _elem_types(model)

    complete = True
    for node in g.node:
        if _has_subgraph(node):
            complete = False
            notes.append(
                f"{node.op_type} {node.name or ''} has a control-flow subgraph: it is not "
                "analysed, so totals are unbounded"
            )
    producer: Dict[str, str] = {}
    for node in g.node:
        for o in node.output:
            if o:
                producer[o] = node.op_type

    def shape_of(name: str) -> Optional[Shape]:
        if name in inits:
            return _sr.from_ints(inits[name].dims)
        s: Optional[Shape] = None
        if name in res.ranged:
            s = res.ranged[name].shape
        elif name in res.intervals:
            s = _sr.from_ints(res.intervals[name][0].shape)
        return _tighten(s, static.get(name))

    out_names = {o.name for o in g.output}
    in_names = {i.name for i in g.input if i.name not in inits}
    tensors: Dict[str, TensorBound] = {}
    unbounded: List[str] = []

    def add(name: str, kind: str) -> None:
        if not name or name in tensors:
            return
        shape = shape_of(name)
        et = etypes.get(name)
        bits = _bits(et) if et is not None else None
        numel = _numel_bound(shape)
        nbytes = _bytes_of(numel, bits)
        note = ""
        if shape is None:
            note = "shape unknown" + (
                f" (output of {producer[name]})" if name in producer else ""
            )
        elif et is None:
            note = "element type unknown"
        elif bits is None:
            note = "element type has no fixed width"
        tensors[name] = TensorBound(name, kind, shape, bits, numel, nbytes, note)
        if not nbytes.bounded:
            unbounded.append(name)

    for n in inits:
        add(n, "weight")
    for n in in_names:
        add(n, "input")
    for node in g.node:
        for o in node.output:
            if o:
                add(o, "output" if o in out_names else "activation")
    for n in out_names:
        add(n, "output")
    if unbounded:
        # one grouped note per reason, naming at most a few tensors (a real model can have hundreds)
        by_reason: Dict[str, List[str]] = {}
        for n in unbounded:
            by_reason.setdefault(tensors[n].note or "unbounded dimension", []).append(n)
        for why, names in by_reason.items():
            shown = ", ".join(repr(n) for n in names[:6])
            more = f" and {len(names) - 6} more" if len(names) > 6 else ""
            notes.append(f"{len(names)} tensor(s) unbounded ({why}): {shown}{more}")

    weight_bytes = sum(int(tensors[n].bytes.lo) for n in inits)

    # ---- liveness peak, model_info convention
    weight_names = set(inits)
    last_use: Dict[str, int] = {}
    for i, node in enumerate(g.node):
        for x in node.input:
            if x:
                last_use[x] = i
    end = len(g.node)
    for o in out_names:
        last_use[o] = end

    def peak(pick: Callable[[Bound], Optional[int]]) -> Optional[int]:
        """Max over steps of the live activation total; ``None`` if some live size is unbounded."""
        live = set(in_names)
        best = sum(_pv(pick(tensors[n].bytes)) for n in live)
        for i, node in enumerate(g.node):
            for o in node.output:
                if o and o not in weight_names:
                    live.add(o)
            cur = 0.0
            for n in live:
                cur += _pv(pick(tensors[n].bytes))
            best = max(best, cur)
            for n in [n for n in live if last_use.get(n) == i]:
                live.discard(n)
        return None if best == _INF else int(best)

    def _pv(v: Optional[int]) -> float:
        return _INF if v is None else float(v)

    peak_lo = peak(lambda b: b.lo) or 0
    peak_hi = peak(lambda b: b.hi) if complete else None
    peak_act = Bound(peak_lo, peak_hi)
    peak_live = Bound(
        peak_lo + weight_bytes, None if peak_hi is None else peak_hi + weight_bytes
    )

    # ---- MACs and bytes moved
    nodes: List[NodeCost] = []
    macs = ZERO
    mem = ZERO
    for node in g.node:
        m = _node_macs(node, shape_of)
        access = ZERO
        for x in list(node.input) + list(node.output):
            if x and x in tensors:
                access = access + tensors[x].bytes
            elif x:
                access = access + UNBOUNDED
        label = node.name or (node.output[0] if node.output else "")
        nodes.append(NodeCost(label, node.op_type, m, access))
        mem = mem + access
        if m is not None:
            macs = macs + m
    if not complete:
        macs = Bound(macs.lo, None)
        mem = Bound(mem.lo, None)

    arena: Optional[ArenaPlan] = None
    if plan_arena:
        arena = _plan_arena(model, tensors, in_names, out_names, notes, complete)

    return CostBounds(
        tensors=tensors,
        nodes=nodes,
        weight_bytes=weight_bytes,
        peak_live_bytes=peak_live,
        peak_activation_bytes=peak_act,
        macs=macs,
        mem_access_bytes=mem,
        arena=arena,
        unbounded_tensors=unbounded,
        notes=notes,
        complete=complete,
    )


# --------------------------------------------------------------------------
# Static arena plan at the upper bounds, verified independently of the planner
# --------------------------------------------------------------------------


# The planner's in-place aliasing rules (onnxsim/memory_planning.cpp), restated here so the
# verifier does not depend on the planner's internals. Unary elementwise ops keep the shape of
# their input and view ops keep its element count, so aliasing is valid at *any* shape. A binary
# op is only safe when the aliased operand has the output's shape; the planner checks that with
# *byte sizes at the shapes it is given* (here the upper bounds), which does not imply it at a
# smaller shape (an input broadcast over a dimension that is larger in the other operand).
_UNARY_INPLACE = {
    "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu", "Softplus", "Softsign", "HardSigmoid",
    "Clip", "Neg", "Abs", "Sqrt", "Exp", "Log", "Reciprocal", "Identity", "Erf", "Celu", "Round",
    "Ceil", "Floor", "Sign",
}  # fmt: skip
_VIEW_OPS = {"Reshape", "Flatten", "Squeeze", "Unsqueeze"}
_BINARY_INPLACE = {"Add", "Sub", "Mul", "Div", "Max", "Min", "And", "Or", "Xor", "Mod"}
_NO_INPLACE_PREFIX = "NoInplace_"


def _same_static_shape(a: Optional[Shape], b: Optional[Shape]) -> bool:
    return (
        a is not None
        and b is not None
        and _sr.is_static(a)
        and _sr.is_static(b)
        and _sr.static_dims(a) == _sr.static_dims(b)
    )


def _alias_allowed(
    node: onnx.NodeProto, x: str, o: str, tensors: Dict[str, TensorBound]
) -> bool:
    """May output ``o`` share storage with input ``x`` of ``node`` at *every* shape in range?"""
    t = node.op_type
    if node.domain not in ("", "ai.onnx"):
        return False
    first = node.input[0] if node.input else ""
    if t in _UNARY_INPLACE or t in _VIEW_OPS:
        return x == first and o == (node.output[0] if node.output else "")
    if t in _BINARY_INPLACE:
        xt, ot = tensors.get(x), tensors.get(o)
        return (
            xt is not None and ot is not None and _same_static_shape(xt.shape, ot.shape)
        )
    return False


def _plan_arena(
    model: onnx.ModelProto,
    tensors: Dict[str, TensorBound],
    in_names: Set[str],
    out_names: Set[str],
    notes: List[str],
    complete: bool,
) -> Optional[ArenaPlan]:
    if not complete:
        notes.append("arena not planned: control-flow subgraph")
        return None
    acts = [n for n, t in tensors.items() if t.kind != "weight"]
    for n in acts:
        t = tensors[n]
        if t.shape is None or not t.bytes.bounded or any(d.hi is None for d in t.shape):
            notes.append("arena not planned: a tensor has no finite upper bound")
            return None
    try:
        from . import memory_planning as _mp
    except Exception:
        notes.append("arena not planned: onnxsim.memory_planning unavailable")
        return None

    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    g2 = m2.graph
    del g2.value_info[:]

    def hi_dims(n: str) -> List[int]:
        s = tensors[n].shape
        assert s is not None
        # The planner (like model_info) ignores rank-0 tensors; one element is one element, so
        # present a scalar as shape [1] -- the same size in bytes.
        return [int(d.hi) for d in s if d.hi is not None] or [1]

    def set_type(vi: onnx.ValueInfoProto, n: str, et: int) -> None:
        vi.type.tensor_type.elem_type = et
        vi.type.tensor_type.shape.SetInParent()  # a rank-0 tensor still has a (empty) shape
        del vi.type.tensor_type.shape.dim[:]
        for v in hi_dims(n):
            vi.type.tensor_type.shape.dim.add().dim_value = v

    # Disable the planner's in-place aliasing of a binary op unless every operand it could pick
    # (one whose size at the upper bounds equals the output's) provably has the output's shape.
    # Renaming the op in this throwaway copy changes nothing but that decision.
    for node in g2.node:
        if (
            node.op_type in _BINARY_INPLACE
            and node.domain in ("", "ai.onnx")
            and node.output
        ):
            out_t = tensors.get(node.output[0])
            risky = False
            for x in node.input:
                xt = tensors.get(x) if x else None
                if xt is None or out_t is None:
                    continue
                if xt.bytes.hi == out_t.bytes.hi and not _same_static_shape(
                    xt.shape, out_t.shape
                ):
                    risky = True
            if risky:
                node.op_type = _NO_INPLACE_PREFIX + node.op_type
    etypes = _elem_types(model)
    for vi in g2.input:
        if (
            vi.name in tensors
            and tensors[vi.name].kind == "input"
            and vi.name in etypes
        ):
            set_type(vi, vi.name, etypes[vi.name])
    for vi in g2.output:
        if vi.name in tensors and vi.name in etypes:
            set_type(vi, vi.name, etypes[vi.name])
    produced = set(out_names) | in_names
    for node in g2.node:
        for o in node.output:
            if o and o not in produced and o in tensors and o in etypes:
                set_type(g2.value_info.add(), o, etypes[o])
                g2.value_info[-1].name = o
    try:
        plan = _mp.plan_activation_memory(m2, run_shape_inference=False)
    except Exception as e:
        notes.append(f"arena not planned: planner failed ({type(e).__name__})")
        return None
    if plan.unplanned:
        notes.append(
            f"arena not planned: planner left {len(plan.unplanned)} tensor(s) unplanned"
        )
        return None

    ok, why = _verify_plan(
        model, plan.tensor_offsets, plan.arena_bytes, tensors, in_names, out_names
    )
    if not ok:
        notes.append(f"arena plan dropped: verification failed ({why})")
        return None
    return ArenaPlan(
        arena_bytes=int(plan.arena_bytes),
        naive_bytes=int(plan.naive_bytes),
        tensor_offsets={
            k: (int(v[0]), int(v[1])) for k, v in plan.tensor_offsets.items()
        },
        verified=True,
    )


def _verify_plan(
    model: onnx.ModelProto,
    offsets: Dict[str, Tuple[int, int]],
    arena_bytes: int,
    tensors: Dict[str, TensorBound],
    in_names: Set[str],
    out_names: Set[str],
) -> Tuple[bool, str]:
    """Check a plan against the model's liveness, independently of the planner.

    Passes iff (1) every activation has a slot at least as large as its *upper-bound* size
    and inside the arena, and (2) any two tensors that are live together occupy disjoint
    address ranges -- except an in-place pair: the input and output of one node, at the same
    offset, where the node is the input's last use (the planner's documented aliasing).
    """
    g = model.graph
    inits = {t.name for t in g.initializer}
    first: Dict[str, int] = {n: -1 for n in in_names}
    last: Dict[str, int] = {}
    for i, node in enumerate(g.node):
        for o in node.output:
            if o and o not in inits:
                first.setdefault(o, i)
        for x in node.input:
            if x:
                last[x] = i
    end = len(g.node)
    for o in out_names:
        last[o] = end
    names = [n for n, t in tensors.items() if t.kind != "weight"]
    spans: Dict[str, Tuple[int, int]] = {}
    for n in names:
        if n not in offsets:
            return False, f"tensor {n!r} has no slot"
        off, size = offsets[n]
        need = tensors[n].bytes.hi
        if need is None or size < need:
            return (
                False,
                f"slot of {n!r} ({size}) is smaller than its upper bound ({need})",
            )
        if off < 0 or off + size > arena_bytes:
            return False, f"slot of {n!r} leaves the arena"
        spans[n] = (first.get(n, 0), last.get(n, end))
    # in-place pairs: (input, output) of the same node where that node is the input's last use
    # and aliasing is valid at every shape in range (see _alias_allowed)
    inplace: Set[Tuple[str, str]] = set()
    for i, node in enumerate(g.node):
        for x in node.input:
            if x and last.get(x) == i and x in spans:
                for o in node.output:
                    if o and o in spans and _alias_allowed(node, x, o, tensors):
                        inplace.add((x, o))
    order = sorted(names, key=lambda n: spans[n][0])
    for a_i, a in enumerate(order):
        sa = spans[a]
        for b in order[a_i + 1 :]:
            sb = spans[b]
            if sb[0] > sa[1]:
                break
            if sa[0] <= sb[1] and sb[0] <= sa[1]:  # lifetimes overlap
                (oa, za), (ob, zb) = offsets[a], offsets[b]
                if oa < ob + zb and ob < oa + za:  # address ranges overlap
                    if oa == ob and ((a, b) in inplace or (b, a) in inplace):
                        continue
                    return False, f"{a!r} and {b!r} are live together and overlap"
    return True, ""


# --------------------------------------------------------------------------
# Budget check
# --------------------------------------------------------------------------


def check_budget(
    model: Union[onnx.ModelProto, str],
    dim_ranges: Optional[Dict[str, Any]] = None,
    memory_bytes: Optional[int] = None,
    macs: Optional[int] = None,
    arena_bytes: Optional[int] = None,
    mem_access_bytes: Optional[int] = None,
    input_shapes: Optional[Dict[str, Sequence[Any]]] = None,
    input_ranges: Optional[Dict[str, Tuple]] = None,
) -> BudgetVerdict:
    """Does the model *provably* fit a deployment target, for every shape in range?

    ``memory_bytes`` limits ``peak_live_bytes`` (weights + live activations, ideal allocator),
    ``arena_bytes`` limits weights-excluded static arena (a real single-buffer allocation),
    ``macs`` the multiply-accumulates, ``mem_access_bytes`` the bytes moved. Each metric is
    ``proved`` when its upper bound is within the limit, ``exceeds`` when even its *lower*
    bound is over it (the model cannot fit), ``unknown`` otherwise (including unbounded).
    ``fits`` is ``True`` only if every given limit is proved, ``False`` if any provably
    exceeds, ``None`` otherwise.
    """
    cb = bounds(model, dim_ranges, input_shapes, input_ranges)
    checks: List[BudgetCheck] = []
    notes = list(cb.notes)

    def judge(metric: str, limit: Optional[int], b: Bound) -> None:
        if limit is None:
            return
        if b.hi is not None and b.hi <= limit:
            st = "proved"
        elif b.lo > limit:
            st = "exceeds"
        else:
            st = "unknown"
        checks.append(BudgetCheck(metric, int(limit), b, st))

    judge("peak_live_bytes", memory_bytes, cb.peak_live_bytes)
    if arena_bytes is not None:
        # Any single-buffer allocator, in-place aliasing or not, must at least hold the largest
        # activation, so that is the only lower bound used for "provably exceeds".
        biggest = max(
            (t.bytes.lo for t in cb.tensors.values() if t.kind != "weight"), default=0
        )
        if cb.arena is not None:
            ab = Bound(biggest, cb.arena.arena_bytes)
        else:
            ab = Bound(biggest, None)
            notes.append("no verified arena plan: the arena budget cannot be proved")
        judge("arena_bytes", arena_bytes, ab)
    judge("macs", macs, cb.macs)
    judge("mem_access_bytes", mem_access_bytes, cb.mem_access_bytes)
    if any(c.status == "exceeds" for c in checks):
        fits: Optional[bool] = False
    elif checks and all(c.status == "proved" for c in checks):
        fits = True
    else:
        fits = None
    return BudgetVerdict(checks, fits, notes)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_range(text: str) -> Tuple[str, Any]:
    name, _, rest = text.partition("=")
    if not name or not rest:
        raise argparse.ArgumentTypeError(f"expected NAME=LO:HI or NAME=N, got {text!r}")
    if ":" not in rest:
        return name, int(rest)
    lo, _, hi = rest.partition(":")
    return name, (int(lo) if lo else 0, int(hi) if hi else None)


def _parse_value_range(text: str) -> Tuple[str, Tuple[float, float]]:
    name, _, rest = text.partition("=")
    lo, _, hi = rest.partition(":")
    if not name or not hi:
        raise argparse.ArgumentTypeError(f"expected NAME=LO:HI, got {text!r}")
    return name, (float(lo), float(hi))


def _fmt(b: Bound) -> str:
    return str(b)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m onnxsim.shape_cost",
        description="Certified memory/compute bounds for dynamic or data-dependent shapes.",
    )
    ap.add_argument("model")
    ap.add_argument(
        "--dim", action="append", default=[], type=_parse_range, metavar="NAME=LO:HI"
    )
    ap.add_argument(
        "--input-range",
        action="append",
        default=[],
        type=_parse_value_range,
        metavar="NAME=LO:HI",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--tensors", action="store_true", help="also list every tensor's bounds"
    )
    ap.add_argument("--budget-memory", type=int, help="limit for peak_live_bytes")
    ap.add_argument(
        "--budget-arena", type=int, help="limit for the static arena (weights excluded)"
    )
    ap.add_argument("--budget-macs", type=int)
    args = ap.parse_args(argv)
    dims = dict(args.dim)
    iranges = dict(args.input_range)
    cb = bounds(args.model, dims, input_ranges=iranges or None)
    verdict = None
    if args.budget_memory or args.budget_arena or args.budget_macs:
        verdict = check_budget(
            args.model, dims, memory_bytes=args.budget_memory, arena_bytes=args.budget_arena,
            macs=args.budget_macs, input_ranges=iranges or None,
        )  # fmt: skip
    if args.json:
        out = cb.summary()
        if args.tensors:
            out["tensors"] = {
                n: {
                    "kind": t.kind,
                    "shape": None if t.shape is None else _sr.shape_str(t.shape),
                    "bytes": {"lo": t.bytes.lo, "hi": t.bytes.hi},
                }
                for n, t in cb.tensors.items()
            }
        if verdict is not None:
            out["budget"] = {
                "fits": verdict.fits,
                "checks": [dataclasses.asdict(c) for c in verdict.checks],
            }
        json.dump(out, sys.stdout, indent=2)
        print()
    else:
        print(f"weights            : {cb.weight_bytes} B")
        print(
            f"peak live bytes    : {_fmt(cb.peak_live_bytes)}  (weights + live activations)"
        )
        print(f"peak activations   : {_fmt(cb.peak_activation_bytes)}")
        if cb.arena is not None:
            print(
                f"static arena       : {cb.arena.arena_bytes} B (verified, naive {cb.arena.naive_bytes} B)"
            )
        print(f"MACs               : {_fmt(cb.macs)}   FLOPs: {_fmt(cb.flops)}")
        print(f"memory access      : {_fmt(cb.mem_access_bytes)}")
        print(
            f"complete analysis  : {cb.complete};  unbounded tensors: {len(cb.unbounded_tensors)}"
        )
        for n in cb.notes:
            print(f"note: {n}")
        if args.tensors:
            for n, t in cb.tensors.items():
                print(
                    f"  {t.kind:10s} {n}: {'?' if t.shape is None else _sr.shape_str(t.shape)}  bytes {t.bytes}"
                )
        if verdict is not None:
            print(verdict)
    return 0 if verdict is None or verdict.fits is not False else 1


if __name__ == "__main__":
    sys.exit(main())
