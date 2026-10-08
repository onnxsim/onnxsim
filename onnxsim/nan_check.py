"""Static NaN and Inf hazard detection on top of interval propagation.

``interval.propagate`` bounds every tensor over the input boxes. A node is a *hazard*
when the operand intervals admit an input for which its IEEE operation yields NaN
(``kind="nan"``) or an infinity (``kind="inf"``), or when a float output may overflow
the range of its dtype. The analysis over-approximates: with no hazard, no NaN or Inf is
reachable inside the boxes; a reported hazard may be unreachable when the intervals are
loose. Inputs without a range are unbounded, so without ``input_ranges`` (or
``onnxsim.range.*`` annotations) nearly every arithmetic node is reported.

Ops fall in three groups. Ops with an exact rule (``_RULES``) are checked for their
specific domains and accumulation overflow. Ops in ``_INF_NAN`` are checked
conservatively: any infinite operand is a NaN hazard. Ops in ``_INF_SAFE`` never create
NaN from an infinite operand. Anything else is ``unmodelled``: its operands are checked
for infinities, but its own NaN behaviour is assumed.
"""

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnxsim import interval as _interval

Interval = Tuple[np.ndarray, np.ndarray]
Hit = Tuple[str, int, str]  # (kind, count of affected elements, detail)

_STANDARD_DOMAINS = ("", "ai.onnx")

_FLOAT_MAX = {
    TensorProto.FLOAT16: float(np.finfo(np.float16).max),
    TensorProto.FLOAT: float(np.finfo(np.float32).max),
    TensorProto.DOUBLE: float(np.finfo(np.float64).max),
    TensorProto.BFLOAT16: 3.3895313892515355e38,
}

_FIXED_OUT = {
    **dict.fromkeys(
        ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "Not", "And"],
        TensorProto.BOOL,
    ),
    **dict.fromkeys(["Or", "Xor", "IsNaN", "IsInf"], TensorProto.BOOL),
    **dict.fromkeys(
        ["Shape", "Size", "ArgMax", "ArgMin", "NonZero"], TensorProto.INT64
    ),
}

# Ops whose NaN behaviour is safe for infinite operands; outputs are checked for overflow.
_INF_SAFE = frozenset(
    {
        "Abs", "Acosh", "Asinh", "Atan", "Ceil", "Clip", "Concat", "Constant",
        "ConstantOfShape", "Cosh", "Celu", "Compress", "Dropout", "DepthToSpace", "Elu",
        "Erf", "Exp", "Expand", "Expm1", "Flatten", "Floor", "Gather", "GatherElements",
        "GatherND", "HardSigmoid", "Identity", "IsInf", "IsNaN", "LeakyRelu", "Max",
        "Min", "Neg", "NonZero", "Pad", "Range", "ReduceMax", "ReduceMin", "Relu",
        "Reshape", "Round", "Selu", "Shape", "Shrink", "Sigmoid", "Sign", "Sinh",
        "Size", "Slice", "SpaceToDepth", "Split", "Squeeze", "Tanh", "ThresholdedRelu",
        "Tile", "TopK", "Transpose", "Unique", "Unsqueeze", "Where", "ArgMax", "ArgMin",
        "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Not", "And", "Or",
        "Xor", "Softplus", "Trilu", "NonMaxSuppression",
    }
)  # fmt: skip

# Ops that yield NaN from any infinite operand (conservative: zeros or opposite-signed
# infinities are not tracked per element).
_INF_NAN = frozenset(
    {
        "Cos", "LogSoftmax", "LRN", "MeanVarianceNormalization", "PRelu", "ReduceLogSumExp",
        "Sin", "Softmax", "Softsign", "Tan",
    }
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Hazard:
    tensor: str  # first output of the node that may produce the value
    op_type: str
    kind: str  # "nan" or "inf"
    count: int  # elements that may be affected
    detail: str


@dataclasses.dataclass
class NanReport:
    hazards: List[Hazard]
    unmodelled: List[str]  # op types whose NaN behaviour is assumed, not checked
    unanalysed: List[str]  # output tensors of nodes whose operands have no interval

    @property
    def nan_free(self) -> bool:
        """No NaN is reachable through the checked ops, and every node was analysed."""
        return not self.unanalysed and not self._any("nan")

    @property
    def finite(self) -> bool:
        """No NaN or Inf is reachable through the checked ops, and every node was analysed."""
        return self.nan_free and not self._any("inf")

    @property
    def complete(self) -> bool:
        """Every op has a rule (no assumed NaN behaviour) and every node was analysed."""
        return not self.unmodelled and not self.unanalysed

    def _any(self, kind: str) -> bool:
        return any(h.kind == kind for h in self.hazards)


@dataclasses.dataclass(frozen=True)
class _Ctx:
    ins: List[Interval]
    in_shapes: List[Tuple[int, ...]]
    out_shape: Tuple[int, ...]
    attrs: Dict[str, Any]
    fmax: float  # largest finite value of the output dtype

    @property
    def size(self) -> int:
        return max(1, int(np.prod(self.out_shape, dtype=np.int64)))


Rule = Callable[[_Ctx], List[Hit]]


def _pos_inf(iv: Interval) -> np.ndarray:
    return iv[1] == np.inf


def _neg_inf(iv: Interval) -> np.ndarray:
    return iv[0] == -np.inf


def _inf(iv: Interval) -> np.ndarray:
    return _pos_inf(iv) | _neg_inf(iv)


def _zero(iv: Interval) -> np.ndarray:
    return (iv[0] <= 0) & (iv[1] >= 0)


def _maxabs(iv: Interval) -> float:
    return float(max(np.abs(iv[0]).max(initial=0.0), np.abs(iv[1]).max(initial=0.0)))


def _hit(kind: str, mask: np.ndarray, detail: str) -> Hit:
    return (kind, int(np.count_nonzero(mask)), detail)


def _inf_operand(c: _Ctx) -> List[Hit]:
    n = sum(int(np.count_nonzero(_inf(x))) for x in c.ins)
    return [("nan", n, "operand may be infinite")]


def _overflow(c: _Ctx, bound: float, detail: str) -> List[Hit]:
    if math.isfinite(bound) and bound > c.fmax:
        return [("inf", c.size, f"{detail} may overflow past {c.fmax:.3g}")]
    return []


def _sign_sum(c: _Ctx, bound: float) -> List[Hit]:
    """Sum-like reduction: +inf and -inf reaching one sum give NaN."""
    pos = any(bool(np.any(_pos_inf(x))) for x in c.ins)
    neg = any(bool(np.any(_neg_inf(x))) for x in c.ins)
    if pos and neg:
        return [("nan", c.size, "+inf and -inf may both reach a sum")]
    if pos or neg:
        return [("inf", c.size, "infinite operand")]
    return _overflow(c, bound, "partial sums")


def _in_size(c: _Ctx) -> int:
    return int(np.prod(c.in_shapes[0], dtype=np.int64))


def _sqrt(c: _Ctx) -> List[Hit]:
    return [_hit("nan", c.ins[0][0] < 0, "argument may be negative")]


def _log(c: _Ctx) -> List[Hit]:
    a = c.ins[0]
    return [
        _hit("nan", a[0] < 0, "argument may be negative"),
        _hit("inf", a[0] <= 0, "argument may be zero"),
    ]


def _log1p(c: _Ctx) -> List[Hit]:
    a = c.ins[0]
    return [
        _hit("nan", a[0] < -1, "argument may be below -1"),
        _hit("inf", (a[0] <= -1) & (a[1] >= -1), "argument may be -1"),
    ]


def _reciprocal(c: _Ctx) -> List[Hit]:
    return [_hit("inf", _zero(c.ins[0]), "argument may be zero")]


def _div(c: _Ctx) -> List[Hit]:
    a, b = c.ins[0], c.ins[1]
    return [
        _hit("nan", (_zero(a) & _zero(b)) | (_inf(a) & _inf(b)), "0/0 or inf/inf"),
        _hit("inf", _zero(b) | _inf(a), "division by zero or by infinity"),
    ]


def _add(c: _Ctx) -> List[Hit]:
    a, b = c.ins[0], c.ins[1]
    return [
        _hit(
            "nan",
            (_pos_inf(a) & _neg_inf(b)) | (_neg_inf(a) & _pos_inf(b)),
            "inf + -inf",
        ),
        _hit("inf", _inf(a) | _inf(b), "infinite operand"),
    ]


def _sub(c: _Ctx) -> List[Hit]:
    a, b = c.ins[0], c.ins[1]
    return [
        _hit(
            "nan",
            (_pos_inf(a) & _pos_inf(b)) | (_neg_inf(a) & _neg_inf(b)),
            "inf - inf",
        ),
        _hit("inf", _inf(a) | _inf(b), "infinite operand"),
    ]


def _mul(c: _Ctx) -> List[Hit]:
    a, b = c.ins[0], c.ins[1]
    return [
        _hit("nan", (_zero(a) & _inf(b)) | (_inf(a) & _zero(b)), "0 * inf"),
        _hit("inf", _inf(a) | _inf(b), "infinite operand"),
    ]


def _pow(c: _Ctx) -> List[Hit]:
    a, b = c.ins[0], c.ins[1]
    y_int = (b[0] == b[1]) & (b[0] == np.round(b[0])) & np.isfinite(b[0])
    return [
        _hit("nan", (a[0] < 0) & ~y_int, "negative base with a non-integer exponent"),
        _hit(
            "inf",
            (_zero(a) & (b[0] < 0)) | _inf(a) | _inf(b),
            "zero base or infinite operand",
        ),
    ]


def _mod(c: _Ctx) -> List[Hit]:
    a, b = c.ins[0], c.ins[1]
    return [_hit("nan", _inf(a) | _zero(b), "fmod(x, 0) or fmod(inf, y)")]


def _asin(c: _Ctx) -> List[Hit]:
    a = c.ins[0]
    return [_hit("nan", (a[0] < -1) | (a[1] > 1), "argument outside [-1, 1]")]


def _acosh(c: _Ctx) -> List[Hit]:
    return [_hit("nan", c.ins[0][0] < 1, "argument may be below 1")]


def _atanh(c: _Ctx) -> List[Hit]:
    a = c.ins[0]
    return [
        _hit("nan", (a[0] < -1) | (a[1] > 1), "argument outside [-1, 1]"),
        _hit(
            "inf",
            ((a[0] <= -1) & (a[1] >= -1)) | ((a[0] <= 1) & (a[1] >= 1)),
            "argument may be +-1",
        ),
    ]


def _neg_inf_nan(detail: str) -> Rule:
    """``x * f(x)`` with ``f(-inf) = 0`` (Gelu, HardSwish, Mish, Swish)."""

    def rule(c: _Ctx) -> List[Hit]:
        return [_hit("nan", _neg_inf(c.ins[0]), detail)]

    return rule


def _sum_rule(c: _Ctx) -> List[Hit]:
    return _sign_sum(c, sum(_maxabs(x) for x in c.ins))


def _reduce_sum(c: _Ctx) -> List[Hit]:
    n = _in_size(c) // c.size
    return _sign_sum(c, n * _maxabs(c.ins[0]))


def _reduce_prod(c: _Ctx) -> List[Hit]:
    a = c.ins[0]
    zero_any = bool(np.any(_zero(a)))
    inf_any = bool(np.any(_inf(a)))
    if zero_any and inf_any:
        return [("nan", c.size, "0 * inf may occur in a product")]
    if inf_any:
        return [("inf", c.size, "infinite operand")]
    m = _maxabs(a)
    n = _in_size(c) // c.size
    if m > 1 and n * math.log(m) > math.log(c.fmax):
        return [("inf", c.size, f"partial products may overflow past {c.fmax:.3g}")]
    return []


def _cumsum(c: _Ctx) -> List[Hit]:
    return _sign_sum(c, _in_size(c) * _maxabs(c.ins[0]))


def _reduce_l1(c: _Ctx) -> List[Hit]:
    n = _in_size(c) // c.size
    return _overflow(c, n * _maxabs(c.ins[0]), "absolute sums")


def _reduce_square_sum(c: _Ctx) -> List[Hit]:
    n = _in_size(c) // c.size
    m = _maxabs(c.ins[0])
    return _overflow(c, n * m * m, "sums of squares")


def _matmul(c: _Ctx) -> List[Hit]:
    k = c.in_shapes[0][-1] if c.in_shapes[0] else 1
    bound = k * _maxabs(c.ins[0]) * _maxabs(c.ins[1])
    return _inf_operand(c) + _overflow(c, bound, "dot products")


def _gemm(c: _Ctx) -> List[Hit]:
    a_shape = c.in_shapes[0]
    trans_a = int(c.attrs.get("transA", 0))
    k = a_shape[0] if trans_a else a_shape[1]
    alpha = abs(float(c.attrs.get("alpha", 1.0)))
    beta = abs(float(c.attrs.get("beta", 1.0)))
    bound = alpha * k * _maxabs(c.ins[0]) * _maxabs(c.ins[1])
    if len(c.ins) > 2:
        bound += beta * _maxabs(c.ins[2])
    return _inf_operand(c) + _overflow(c, bound, "dot products")


def _conv(c: _Ctx) -> List[Hit]:
    k = int(np.prod(c.in_shapes[1][1:], dtype=np.int64))
    bound = k * _maxabs(c.ins[0]) * _maxabs(c.ins[1])
    if len(c.ins) > 2:
        bound += _maxabs(c.ins[2])
    return _inf_operand(c) + _overflow(c, bound, "convolution sums")


def _conv_transpose(c: _Ctx) -> List[Hit]:
    w = c.in_shapes[1]
    k = w[0] * int(np.prod(w[2:], dtype=np.int64))
    bound = k * _maxabs(c.ins[0]) * _maxabs(c.ins[1])
    return _inf_operand(c) + _overflow(c, bound, "transposed convolution sums")


def _average_pool(c: _Ctx) -> List[Hit]:
    kernel = c.attrs.get("kernel_shape")
    k = int(np.prod(kernel, dtype=np.int64)) if kernel else 1
    return _inf_operand(c) + _overflow(c, k * _maxabs(c.ins[0]), "pooling sums")


def _global_average_pool(c: _Ctx) -> List[Hit]:
    k = int(np.prod(c.in_shapes[0][2:], dtype=np.int64))
    return _inf_operand(c) + _overflow(c, k * _maxabs(c.ins[0]), "pooling sums")


def _centred_square_sum(c: _Ctx, n: int) -> List[Hit]:
    """Sum of squared deviations over ``n`` values; each deviation is at most ``2 * max``."""
    m = _maxabs(c.ins[0])
    return _inf_operand(c) + _overflow(c, n * (2 * m) ** 2, "variance sums")


def _layer_norm(c: _Ctx) -> List[Hit]:
    shape = c.in_shapes[0]
    if not shape:
        return _inf_operand(c)
    axis = int(c.attrs.get("axis", -1)) % len(shape)
    return _centred_square_sum(c, int(np.prod(shape[axis:], dtype=np.int64)))


def _instance_norm(c: _Ctx) -> List[Hit]:
    return _centred_square_sum(c, int(np.prod(c.in_shapes[0][2:], dtype=np.int64)))


def _group_norm(c: _Ctx) -> List[Hit]:
    shape = c.in_shapes[0]
    groups = c.attrs.get("num_groups")
    channels = shape[1] // int(groups) if groups else shape[1]
    return _centred_square_sum(c, channels * int(np.prod(shape[2:], dtype=np.int64)))


def _batch_norm(c: _Ctx) -> List[Hit]:
    if len(c.ins) < 5:
        return _inf_operand(c)
    eps = float(c.attrs.get("epsilon", 1e-5))
    var = c.ins[4]
    return _inf_operand(c) + [
        _hit("nan", var[0] + eps < 0, "variance may be below -epsilon"),
        _hit(
            "inf",
            (var[0] + eps <= 0) & (var[1] + eps >= 0),
            "variance may equal -epsilon",
        ),
    ]


def _cast(c: _Ctx) -> List[Hit]:
    return _overflow(c, _maxabs(c.ins[0]), "cast")


def _resize(c: _Ctx) -> List[Hit]:
    mode = c.attrs.get("mode", b"nearest")
    if isinstance(mode, bytes):
        mode = mode.decode()
    return [] if mode == "nearest" else _inf_operand(c)


_RULES: Dict[str, Rule] = {
    "Sqrt": _sqrt,
    "Log": _log,
    "Log1p": _log1p,
    "Reciprocal": _reciprocal,
    "Div": _div,
    "Add": _add,
    "Sub": _sub,
    "Mul": _mul,
    "Pow": _pow,
    "Mod": _mod,
    "Asin": _asin,
    "Acos": _asin,
    "Acosh": _acosh,
    "Atanh": _atanh,
    "Gelu": _neg_inf_nan("x * erf(x) term with erf(-inf) = -1 gives -inf * 0"),
    "HardSwish": _neg_inf_nan("x * clip(x / 6 + 0.5) gives -inf * 0"),
    "Mish": _neg_inf_nan("x * tanh(softplus(x)) gives -inf * 0"),
    "Swish": _neg_inf_nan("x * sigmoid(x) gives -inf * 0"),
    "Sum": _sum_rule,
    "Mean": _sum_rule,
    "ReduceSum": _reduce_sum,
    "ReduceMean": _reduce_sum,
    "ReduceProd": _reduce_prod,
    "ReduceL1": _reduce_l1,
    "ReduceL2": _reduce_square_sum,
    "ReduceSumSquare": _reduce_square_sum,
    "CumSum": _cumsum,
    "MatMul": _matmul,
    "Gemm": _gemm,
    "Conv": _conv,
    "ConvTranspose": _conv_transpose,
    "AveragePool": _average_pool,
    "GlobalAveragePool": _global_average_pool,
    "LayerNormalization": _layer_norm,
    "InstanceNormalization": _instance_norm,
    "GroupNormalization": _group_norm,
    "BatchNormalization": _batch_norm,
    "Resize": _resize,
    "Cast": _cast,
}

_CHECKED = frozenset(_RULES) | _INF_SAFE | _INF_NAN


def _dtype_map(model: onnx.ModelProto) -> Dict[str, int]:
    dt: Dict[str, int] = {}
    for vi in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        et = vi.type.tensor_type.elem_type
        if et:
            dt[vi.name] = et
    for t in model.graph.initializer:
        dt[t.name] = t.data_type
    for node in model.graph.node:
        if node.op_type == "Cast":
            to = next(
                (a.i for a in node.attribute if a.name == "to"), TensorProto.FLOAT
            )
            inherited = to
        elif node.op_type in _FIXED_OUT:
            inherited = _FIXED_OUT[node.op_type]
        else:
            inherited = (
                dt.get(node.input[0], TensorProto.FLOAT)
                if node.input
                else TensorProto.FLOAT
            )
        for o in node.output:
            if o:
                dt.setdefault(o, inherited)
    return dt


def _shape(res: _interval.IntervalResult, name: str) -> Tuple[int, ...]:
    return tuple(np.shape(res.intervals[name][0])) if name in res.intervals else ()


def check_nan(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
) -> NanReport:
    """Report the NaN and Inf hazards of ``model`` over the boxes in ``input_ranges``.

    :param input_ranges: ``{input: (lo, hi)}``, merged over the model's ``onnxsim.range.*``
        annotations. An input with neither is unbounded.
    """
    res = _interval.propagate(model, input_ranges)
    dtypes = _dtype_map(model)
    hazards: List[Hazard] = []
    unmodelled: List[str] = []
    unanalysed: List[str] = []
    for node in model.graph.node:
        op = node.op_type
        outs = [o for o in node.output if o]
        tensor = outs[0] if outs else node.name
        standard = node.domain in _STANDARD_DOMAINS
        if not standard or op not in _CHECKED:
            if op not in unmodelled:
                unmodelled.append(op)
        if not standard:
            continue
        operands = [n for n in node.input if n]
        if any(n in res.ranged or n not in res.intervals for n in operands):
            unanalysed.append(tensor)
            continue
        if not operands or not outs:
            continue
        out_fmax = _FLOAT_MAX.get(dtypes.get(outs[0], TensorProto.FLOAT))
        if out_fmax is None:
            continue
        ins = [
            (
                np.asarray(res.intervals[n][0], np.float64),
                np.asarray(res.intervals[n][1], np.float64),
            )
            for n in operands
        ]
        ctx = _Ctx(
            ins=ins,
            in_shapes=[_shape(res, n) for n in operands],
            out_shape=_shape(res, outs[0]),
            attrs={a.name: helper.get_attribute_value(a) for a in node.attribute},
            fmax=out_fmax,
        )
        if op in _RULES:
            hits = _RULES[op](ctx)
        elif op in _INF_SAFE:
            hits = []
        else:
            hits = _inf_operand(ctx)
        found = [Hazard(tensor, op, k, n, d) for k, n, d in hits if n]
        hazards.extend(found)
        if any(h.kind == "inf" for h in found) or any(
            bool(np.any(_inf(x))) for x in ins
        ):
            continue
        for out in outs:
            if out not in res.intervals:
                continue
            omax = _FLOAT_MAX.get(dtypes.get(out, TensorProto.FLOAT))
            if omax is None:
                continue
            lo, hi = res.intervals[out]
            over = int(np.count_nonzero((np.abs(lo) > omax) | (np.abs(hi) > omax)))
            if over:
                hazards.append(
                    Hazard(out, op, "inf", over, f"may overflow past {omax:.3g}")
                )
    return NanReport(hazards, unmodelled, unanalysed)
