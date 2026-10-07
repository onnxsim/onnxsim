"""Interval analysis of an ONNX model, and quantization bounds built on it.

``propagate(model, input_ranges)`` computes, for every tensor, an elementwise
``[lo, hi]`` that encloses every value the tensor can take for inputs inside the
given box. ``quantization_bounds`` then answers, per MatMul/Gemm/Conv with
constant weights and *from the reachable activation range instead of the
scheme's full range*:

* the worst-case int32 accumulator magnitude, and whether it can overflow
  int32 or lose integer exactness in a float32 cast (2**24). This is the same
  question ``onnxsim.precision_estimator`` answers with ``K * 127 * 255``, which
  assumes the activation can use all 8 bits; the interval bound uses the
  activation range the model can actually produce, so it is never looser and is
  often several times tighter;
* a certified worst-case output error from rounding weights and activations to
  their grids (``eps_x*sum|w| + eps_w*sum|x| + K*eps_x*eps_w``, the lemma proved
  in ``tests/test_formal_verify_quantized_mac_bound.py``), as a fraction of the
  layer's output range;
* the activation scale / zero-point that range implies.

Transfer functions use midpoint-radius arithmetic (linear ops, products),
monotonicity (Relu, Sigmoid, Tanh, ...), and applying the op to ``lo`` and ``hi``
separately for data movement (Reshape, Slice, Concat, ...). An op without a rule
yields an unbounded interval for its outputs, never a wrong bound.

What an interval means, so it is not over-read:

* It encloses the **real-number** function, computed in float64 with a small
  relative widening. Running the model in float32 can exceed the enclosure by
  float32 rounding (about 1e-6 relative per op); ``slack`` in
  :func:`contains` is for that.
* Plain intervals lose correlations (``x - x`` is not 0), so bounds loosen with
  depth and with residual/branching structure. They are sound, not tight.
* The quantization numbers are worst-case over the box, not an estimate of
  accuracy on real data. A calibrated scale is usually far narrower than the
  interval range; use this to prove safety and spot risk, and calibration to
  choose scales.
"""

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from . import ranges as _ranges
from . import shape_ranges as _sr

Interval = Tuple[np.ndarray, np.ndarray]

_INF = np.inf
_DATA_MOVE = {
    "Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Concat",
    "Identity", "Expand", "Tile", "Pad", "MaxPool", "Split", "ReduceMax", "ReduceMin",
    "Dropout", "GlobalMaxPool", "DepthToSpace", "SpaceToDepth", "Max", "Min",
}  # fmt: skip
_MONOTONE = {
    "Relu", "Sigmoid", "Tanh", "Exp", "Erf", "Sqrt", "Softplus", "Clip", "Floor", "Ceil",
    "Round", "Log", "HardSigmoid",
}  # fmt: skip
_EXACT = {"Relu", "Clip", "Floor", "Ceil", "Round"}  # computed without rounding error
_CODOMAIN = {
    "Sigmoid": (0.0, 1.0),
    "Tanh": (-1.0, 1.0),
    "Relu": (0.0, _INF),
    "Exp": (0.0, _INF),
    "Softplus": (0.0, _INF),
    "Sqrt": (0.0, _INF),
    "Erf": (-1.0, 1.0),
}
_LINEAR_AVG = {"AveragePool", "GlobalAveragePool", "ReduceMean", "ReduceSum"}
_BILINEAR = {"MatMul", "Gemm", "Conv", "ConvTranspose"}


@dataclasses.dataclass
class RangedTensor:
    """A tensor whose *shape* is only known to lie in ranges (see :mod:`shape_ranges`).

    ``hull`` is one ``(lo, hi)`` enclosing **every** element: with a dynamic number of
    elements there is no fixed per-element array to hold, so the interval is the
    scalar hull. Such tensors live in :attr:`IntervalResult.ranged`, not in
    :attr:`IntervalResult.intervals`.
    """

    shape: _sr.Shape
    hull: Tuple[float, float]


@dataclasses.dataclass
class IntervalResult:
    intervals: Dict[str, Interval]
    unsupported: List[str]  # op types that fell back to "unbounded"
    #: Tensors whose shape is ranged (data-dependent ops such as NonZero/TopK, dynamic
    #: inputs, and anything computed from them) -- ``name -> RangedTensor``. Tensors
    #: with a fully known shape stay in ``intervals`` exactly as before.
    ranged: Dict[str, RangedTensor] = dataclasses.field(default_factory=dict)

    @property
    def ranged_names(self) -> List[str]:
        return sorted(self.ranged)

    def hull(self, name: str) -> Tuple[float, float]:
        if name in self.ranged:
            return self.ranged[name].hull
        lo, hi = self.intervals[name]
        return float(np.min(lo)), float(np.max(hi))

    def shape(self, name: str) -> _sr.Shape:
        """Ranged shape of a tensor (exact dims for tensors in ``intervals``)."""
        if name in self.ranged:
            return self.ranged[name].shape
        return _sr.from_ints(self.intervals[name][0].shape)

    def contains(self, name: str, value: np.ndarray, slack: float = 1e-4) -> bool:
        """Is ``value`` inside the interval of ``name``, up to relative+absolute ``slack``?

        For a ranged tensor this also requires the value's shape to be one of the
        shapes the ranged shape allows.
        """
        v = np.asarray(value, dtype=np.float64)
        pad = slack * (1.0 + np.abs(v))
        if name in self.ranged:
            r = self.ranged[name]
            if not _sr.contains_shape(r.shape, v.shape):
                return False
            return bool(np.all(v >= r.hull[0] - pad) and np.all(v <= r.hull[1] + pad))
        lo, hi = self.intervals[name]
        return bool(np.all(v >= lo - pad) and np.all(v <= hi + pad))


def _widen(lo: np.ndarray, hi: np.ndarray) -> Interval:
    eps = 64 * np.finfo(np.float64).eps
    with np.errstate(invalid="ignore"):
        lo = np.where(np.isfinite(lo), lo - eps * np.abs(lo), lo)
        hi = np.where(np.isfinite(hi), hi + eps * np.abs(hi), hi)
    return lo, hi


def _unbounded(shape) -> Interval:
    return np.full(shape, -_INF), np.full(shape, _INF)


def _point(a: np.ndarray) -> Interval:
    a = np.asarray(a)
    return a, a


def _is_point(iv: Interval) -> bool:
    return iv[0] is iv[1] or (
        iv[0].shape == iv[1].shape and np.array_equal(iv[0], iv[1])
    )


def _finite(iv: Interval) -> bool:
    return bool(np.all(np.isfinite(iv[0])) and np.all(np.isfinite(iv[1])))


def _midrad(iv: Interval) -> Tuple[np.ndarray, np.ndarray]:
    return (iv[0] + iv[1]) / 2.0, (iv[1] - iv[0]) / 2.0


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class _Unbounded(Exception):
    """An operand is unbounded: the output is too, which is a result, not an unsupported op."""


class _Runner:
    """Evaluate a single ONNX node on float64/int arrays via onnx's reference evaluator."""

    def __init__(self, model: onnx.ModelProto):
        self.opsets = list(model.opset_import)
        self.ir = model.ir_version

    def run(
        self, node: onnx.NodeProto, inputs: List[np.ndarray], **override
    ) -> List[np.ndarray]:
        from onnx.reference import ReferenceEvaluator

        # Interval arrays are float64 but constants (initializers) are usually float32,
        # and the reference evaluator rejects a binary op on mixed float dtypes (MatMul
        # with a float32 weight, for one) -- which would silently turn the op into
        # "unsupported". Promote every float operand to float64.
        inputs = [
            np.asarray(a, dtype=np.float64)
            if np.asarray(a).dtype.kind == "f"
            else np.asarray(a)
            for a in inputs
        ]
        n = onnx.NodeProto()
        n.CopyFrom(node)
        names = [f"i{k}" for k in range(len(inputs))]
        del n.input[:]
        n.input.extend(names)
        del n.output[:]
        n.output.extend(f"o{k}" for k in range(max(1, len(node.output))))
        for key, val in override.items():
            for a in list(n.attribute):
                if a.name == key:
                    n.attribute.remove(a)
            n.attribute.append(onnx.helper.make_attribute(key, val))
        g = onnx.helper.make_graph(
            [n], "n",
            [onnx.helper.make_tensor_value_info(nm, onnx.helper.np_dtype_to_tensor_dtype(np.asarray(a).dtype), np.asarray(a).shape)
             for nm, a in zip(names, inputs)],
            [onnx.helper.make_empty_tensor_value_info(o) for o in n.output],
        )  # fmt: skip
        m = onnx.helper.make_model(g, opset_imports=self.opsets)
        m.ir_version = self.ir
        return list(ReferenceEvaluator(m).run(None, dict(zip(names, inputs))))


def _bilinear(runner, node, ins: List[Interval]) -> Interval:
    """MatMul/Gemm/Conv(/ConvTranspose) of intervals: centre op(c1,c2), radius from |c|, r."""
    x, w = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    (c1, r1), (c2, r2) = _midrad(x), _midrad(w)
    centre = runner.run(
        node, [c1, c2] + ([_midrad(bias)[0]] if bias is not None else [])
    )[0]
    # The radius uses the same op without the bias term and with beta/alpha forced
    # non-negative so every term of the sum adds width.
    ov = {}
    if node.op_type == "Gemm":
        ov = {"alpha": abs(float(_attrs(node).get("alpha", 1.0))), "beta": 0.0}
    nb = lambda a, b: runner.run(node, [a, b], **ov)[0]  # noqa: E731
    rad = nb(np.abs(c1), r2) + nb(r1, np.abs(c2)) + nb(r1, r2)
    if bias is not None:
        rb = _midrad(bias)[1]
        beta = (
            abs(float(_attrs(node).get("beta", 1.0))) if node.op_type == "Gemm" else 1.0
        )
        rad = rad + (
            rb * beta
            if node.op_type == "Gemm"
            else rb.reshape([1, -1] + [1] * (rad.ndim - 2))
        )
    return _widen(centre - rad, centre + rad)


# --------------------------------------------------------------------------
# Tensors with ranged (data-dependent) shapes
# --------------------------------------------------------------------------

Hull = Tuple[float, float]
_FULL: Hull = (-_INF, _INF)
_BOOL_HULL: Hull = (0.0, 1.0)

# Ops that create a ranged shape from static inputs, handled here (after the exact
# all-constant path) instead of falling back to "unsupported".
_GENERATORS = {"NonZero", "TopK", "Unique", "NonMaxSuppression", "Compress"}
# Ops whose output shape comes from a shape-valued input; ranged when that input is an
# interval (e.g. derived from a ranged tensor's Shape) rather than a constant.
_SHAPE_CONSUMERS = {"Reshape": 1, "Expand": 1, "Tile": 1, "ConstantOfShape": 0}
_COMPARE_NAMES = {
    "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Xor",
}  # fmt: skip
_COMPARE = _COMPARE_NAMES
_HULL_OPS = {"Cast", "Abs"} | _COMPARE_NAMES
_INT_CAST = {
    onnx.TensorProto.INT8, onnx.TensorProto.INT16, onnx.TensorProto.INT32,
    onnx.TensorProto.INT64, onnx.TensorProto.UINT8, onnx.TensorProto.UINT16,
    onnx.TensorProto.UINT32, onnx.TensorProto.UINT64,
}  # fmt: skip


def _mul0(a: float, b: float) -> float:
    """``a * b`` with ``0 * inf = 0``: an element that is exactly 0 stays 0 whatever it multiplies."""
    return 0.0 if a == 0.0 or b == 0.0 else a * b


def _hull_of(iv: Interval) -> Hull:
    lo, hi = iv
    if np.asarray(lo).size == 0:
        return (0.0, 0.0)
    return float(np.min(lo)), float(np.max(hi))


def _hull_union(a: Hull, b: Hull) -> Hull:
    return min(a[0], b[0]), max(a[1], b[1])


def _hull_mul(a: Hull, b: Hull) -> Hull:
    p = [_mul0(x, y) for x in a for y in b]
    return min(p), max(p)


def _sum_hull(n: _sr.Dim, h: Hull) -> Hull:
    """Hull of a sum of ``n`` elements (``n`` ranging over ``[n.lo, n.hi]``) each in ``h``."""
    lo_c = [_mul0(float(n.lo), h[0])]
    hi_c = [_mul0(float(n.lo), h[1])]
    if n.hi is not None:
        lo_c.append(_mul0(float(n.hi), h[0]))
        hi_c.append(_mul0(float(n.hi), h[1]))
    else:  # unbounded count: the sum is unbounded in whichever direction the hull allows
        lo_c.append(-_INF if h[0] < 0 else lo_c[0])
        hi_c.append(_INF if h[1] > 0 else hi_c[0])
    return min(lo_c), max(hi_c)


def _scalar_arith(t: str, a: Hull, b: Hull, widen_int: bool) -> Hull:
    with np.errstate(all="ignore"):
        if t == "Add":
            lo, hi = a[0] + b[0], a[1] + b[1]
        elif t == "Sub":
            lo, hi = a[0] - b[1], a[1] - b[0]
        elif t == "Mul":
            lo, hi = _hull_mul(a, b)
        else:  # Div: only when the denominator excludes 0
            if b[0] > 0 or b[1] < 0:
                lo, hi = _hull_mul(a, (1.0 / b[1], 1.0 / b[0]))
                if widen_int and np.isfinite(lo) and np.isfinite(hi):
                    # integer division truncates: stay a superset of both semantics
                    lo, hi = math.floor(lo), math.ceil(hi)
            else:
                return _FULL
    return (-_INF if np.isnan(lo) else lo), (_INF if np.isnan(hi) else hi)


def _monotone_hull(runner: "_Runner", node: onnx.NodeProto, h: Hull, extra) -> Hull:
    t = node.op_type
    with np.errstate(all="ignore"):
        lo = float(runner.run(node, [np.float64(h[0])] + extra)[0])
        hi = float(runner.run(node, [np.float64(h[1])] + extra)[0])
    lo_a, hi_a = np.float64(lo), np.float64(hi)
    if t not in _EXACT:
        lo_a, hi_a = _widen(lo_a, hi_a)
    lo, hi = float(lo_a), float(hi_a)
    if t in _CODOMAIN:
        lo, hi = max(lo, _CODOMAIN[t][0]), min(hi, _CODOMAIN[t][1])
    return lo, hi


def _ints(iv: Interval) -> Optional[List[int]]:
    """A constant integer vector (a point interval), or ``None`` when it is ranged."""
    if not _is_point(iv):
        return None
    return [int(v) for v in np.asarray(iv[0]).reshape(-1)]


def _axes_of(node: onnx.NodeProto, ins: List[Any], pos: int) -> Optional[List[int]]:
    """``axes`` from an attribute (older opsets) or a constant input; ``None`` if absent."""
    a = _attrs(node).get("axes")
    if a is not None:
        return [int(v) for v in a]
    if len(ins) > pos and ins[pos] is not None:
        got = _ints(ins[pos])
        if got is None:
            raise KeyError(node.op_type)  # ranged axes: cannot say
        return got
    return None


def _dim_from_interval(iv: Interval) -> _sr.Dim:
    """A scalar/1-element integer interval (e.g. TopK's K) as a ranged dimension."""
    lo, hi = float(np.min(iv[0])), float(np.max(iv[1]))
    return _sr.Dim(
        max(0, int(math.floor(lo))) if np.isfinite(lo) else 0,
        int(math.ceil(hi)) if np.isfinite(hi) else None,
    )


def _ranged_node(
    node: onnx.NodeProto,
    ins: List[Any],
    iv: Dict[str, Interval],
    ranged: Dict[str, RangedTensor],
    runner: "_Runner",
) -> None:
    """Transfer function for a node with a ranged-shape operand, or a data-dependent op.

    Sets the node's outputs in ``iv`` (when their shape turns out fully known) or
    ``ranged``. Raises ``KeyError``/``ValueError`` for an op or situation without a
    sound rule; the caller then leaves the outputs unknown.

    Every rule states shapes through :mod:`shape_ranges` and values as one scalar hull
    over all elements; none guesses. An operand whose semantics depend on a dynamic
    dimension either folds the dimension's range into the result (reductions, MatMul)
    or yields the full hull.
    """
    t = node.op_type
    outs = list(node.output)
    attrs = _attrs(node)
    # The branches below reuse short names (``x``, ``h``, ``shape``, ...) for values of
    # different types; declare them loosely so each branch stays readable on its own.
    x: Any
    y: Any
    a: Any
    b: Any
    c: Any
    h: Any
    k: Any
    n: Any
    v: Any
    lo: Any
    hi: Any
    sl: Any
    rs: Any
    vh: Any
    idx: Any
    top: Any
    axis: Any
    axes: Any
    dims: Any
    shape: Any
    definitely: Any
    possibly: Any

    def R(k: int) -> RangedTensor:
        name = node.input[k]
        if name in ranged:
            return ranged[name]
        arr = iv[name]
        return RangedTensor(_sr.from_ints(np.asarray(arr[0]).shape), _hull_of(arr))

    def emit(k: int, shape: Optional[_sr.Shape], hull: Hull) -> None:
        if k >= len(outs) or not outs[k] or shape is None:
            return
        if _sr.is_static(shape):
            dims = _sr.static_dims(shape)
            iv[outs[k]] = (np.full(dims, hull[0]), np.full(dims, hull[1]))
        else:
            ranged[outs[k]] = RangedTensor(shape, hull)

    def static_in(k: int) -> Interval:
        name = node.input[k] if k < len(node.input) else ""
        if not name or name in ranged or name not in iv:
            raise KeyError(t)
        return iv[name]

    # ---- shape/size of a ranged tensor become ordinary (interval-valued) tensors
    if t == "Shape":
        shp = R(0).shape
        lo = np.array([d.lo for d in shp], dtype=np.float64)
        hi = np.array([_INF if d.hi is None else d.hi for d in shp], dtype=np.float64)
        s, e = int(attrs.get("start", 0)), attrs.get("end")
        sl = slice(s, None if e is None else int(e))
        iv[outs[0]] = (lo[sl], hi[sl])
        return
    if t == "Size":
        n = _sr.numel(R(0).shape)
        iv[outs[0]] = (np.float64(n.lo), np.float64(_INF if n.hi is None else n.hi))
        return

    # ---- data-dependent generators
    if t == "NonZero":
        name = node.input[0]
        if name in ranged:
            r = ranged[name]
            shape = _sr.nonzero(r.shape, value_hull=r.hull)
        else:
            lo_a, hi_a = iv[name]
            definitely = int(np.count_nonzero((lo_a > 0) | (hi_a < 0)))
            possibly = int(np.count_nonzero(~((lo_a == 0) & (hi_a == 0))))
            shape = _sr.nonzero(
                _sr.from_ints(np.asarray(lo_a).shape), definitely, possibly
            )
        in_shape = R(0).shape
        top = (
            max((_INF if d.hi is None else d.hi) for d in in_shape) - 1
            if in_shape
            else 0
        )
        emit(0, shape, (0.0, float(max(top, 0))))
        return

    if t == "TopK":
        xk = R(0)
        axis = int(attrs.get("axis", -1))
        k_iv = static_in(1)
        kd = _dim_from_interval(k_iv)
        shape = _sr.topk(xk.shape, axis, kd)
        ax = axis + len(xk.shape) if axis < 0 else axis
        d_axis = xk.shape[ax]
        idx_hi = _INF if d_axis.hi is None else float(max(d_axis.hi - 1, 0))
        name = node.input[0]
        if name not in ranged and _is_point(k_iv):
            # static input, constant K: per-position bounds from order statistics -- the
            # j-th largest value lies between the j-th largest lower bound and the j-th
            # largest upper bound (symmetrically for smallest)
            lo_a, hi_a = iv[name]
            largest = int(attrs.get("largest", 1)) != 0
            sl = [slice(None)] * lo_a.ndim
            sl[ax] = slice(0, kd.lo)
            if largest:
                v_lo = -np.sort(-lo_a, axis=ax)[tuple(sl)]
                v_hi = -np.sort(-hi_a, axis=ax)[tuple(sl)]
            else:
                v_lo = np.sort(lo_a, axis=ax)[tuple(sl)]
                v_hi = np.sort(hi_a, axis=ax)[tuple(sl)]
            iv[outs[0]] = (v_lo, v_hi)
            if len(outs) > 1 and outs[1]:
                iv[outs[1]] = (np.zeros(v_lo.shape), np.full(v_lo.shape, idx_hi))
            return
        emit(0, shape, xk.hull)
        emit(1, shape, (0.0, idx_hi))
        return

    if t == "Unique":
        xk = R(0)
        axis = attrs.get("axis")
        y, idx, inv, cnt = _sr.unique(xk.shape, None if axis is None else int(axis))
        n_hi = _sr.numel(xk.shape).hi
        top = _INF if n_hi is None else float(max(n_hi - 1, 0))
        emit(0, y, xk.hull)
        emit(1, idx, (0.0, top))
        emit(2, inv, (0.0, top))
        emit(3, cnt, (1.0, _INF if n_hi is None else float(max(n_hi, 1))))
        return

    if t == "NonMaxSuppression":
        boxes, scores = R(0), R(1)
        if len(node.input) > 2 and node.input[2]:
            mpc = _dim_from_interval(static_in(2))
        else:
            mpc = _sr.exact(
                0
            )  # default max_output_boxes_per_class is 0: selects nothing
        shape = _sr.nms(boxes.shape, scores.shape, mpc)
        dims = [boxes.shape[0], scores.shape[1], boxes.shape[1]]
        top = max((_INF if d.hi is None else d.hi) for d in dims) - 1
        emit(0, shape, (0.0, float(max(top, 0))))
        return

    if t == "Compress":
        xk = R(0)
        axis = attrs.get("axis")
        cname = node.input[1]
        cond_len = _sr.numel(R(1).shape)
        definitely = possibly = None
        if cname not in ranged:
            c_lo, c_hi = iv[cname]
            definitely = int(np.count_nonzero(c_lo > 0))
            possibly = int(np.count_nonzero(c_hi > 0))
        shape = _sr.compress(
            xk.shape,
            None if axis is None else int(axis),
            cond_len,
            definitely,
            possibly,
        )
        emit(0, shape, xk.hull)
        return

    if t == "Range":
        h = [_hull_of(static_in(k)) for k in range(3)]
        n = _sr.range_count(h[0], h[1], h[2])
        if h[2][0] > 0:
            vh = (h[0][0], h[1][1])
        elif h[2][1] < 0:
            vh = (h[1][0], h[0][1])
        else:
            vh = _FULL
        emit(0, (n,), vh)
        return

    if t == "ConstantOfShape":
        lo, hi = static_in(0)
        val = attrs.get("value")
        v = float(numpy_helper.to_array(val).reshape(-1)[0]) if val is not None else 0.0
        emit(0, _sr.constant_of_shape(lo, hi), (v, v))
        return

    # ---- everything below needs R(k) for its operands
    if t in _MONOTONE or (t == "LeakyRelu" and float(attrs.get("alpha", 0.01)) >= 0):
        extra = []
        for k in range(1, len(node.input)):
            if not node.input[k]:
                continue
            e = static_in(k)
            if not _is_point(e):
                raise KeyError(t)
            extra.append(e[0])
        x = R(0)
        emit(0, x.shape, _monotone_hull(runner, node, x.hull, extra))
        return
    if t in ("Identity", "Dropout"):
        if len([o for o in outs if o]) > 1:
            raise KeyError(t)
        x = R(0)
        emit(0, x.shape, x.hull)
        return
    if t == "Neg":
        x = R(0)
        emit(0, x.shape, (-x.hull[1], -x.hull[0]))
        return
    if t == "Abs":
        x = R(0)
        lo, hi = x.hull
        emit(
            0,
            x.shape,
            (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi))),
        )
        return
    if t == "Softmax":
        emit(0, R(0).shape, (0.0, 1.0))
        return
    if t == "Cast":
        x = R(0)
        to = int(attrs["to"])
        if to in _INT_CAST:
            h = (float(np.trunc(x.hull[0])), float(np.trunc(x.hull[1])))
        elif to == onnx.TensorProto.BOOL:
            if x.hull[0] > 0 or x.hull[1] < 0:
                h = (1.0, 1.0)
            elif x.hull == (0.0, 0.0):
                h = (0.0, 0.0)
            else:
                h = _BOOL_HULL
        else:
            h = x.hull
        emit(0, x.shape, h)
        return
    if t in _COMPARE:
        emit(0, _sr.broadcast([R(k).shape for k in range(len(node.input))]), _BOOL_HULL)
        return
    if t in ("Add", "Sub", "Mul", "Div") and len(node.input) == 2:
        a, b = R(0), R(1)
        widen = t == "Div" and all(
            float(v).is_integer() for v in (*a.hull, *b.hull) if np.isfinite(v)
        )
        emit(
            0,
            _sr.broadcast([a.shape, b.shape]),
            _scalar_arith(t, a.hull, b.hull, widen),
        )
        return
    if t == "Where":
        c, x, y = R(0), R(1), R(2)
        emit(0, _sr.broadcast([c.shape, x.shape, y.shape]), _hull_union(x.hull, y.hull))
        return
    if t in ("Max", "Min"):
        rs = [R(k) for k in range(len(node.input))]
        h = rs[0].hull
        for r in rs[1:]:
            h = _hull_union(h, r.hull)
        emit(0, _sr.broadcast([r.shape for r in rs]), h)
        return

    # ---- layout ops: the value hull is unchanged, only the shape moves
    if t == "Reshape":
        x = R(0)
        tlo, thi = static_in(1)
        emit(0, _sr.reshape(x.shape, tlo, thi, bool(attrs.get("allowzero", 0))), x.hull)
        return
    if t == "Flatten":
        x = R(0)
        emit(0, _sr.flatten(x.shape, int(attrs.get("axis", 1))), x.hull)
        return
    if t == "Transpose":
        x = R(0)
        emit(0, _sr.transpose(x.shape, attrs.get("perm")), x.hull)
        return
    if t == "Squeeze":
        x = R(0)
        emit(0, _sr.squeeze(x.shape, _axes_of(node, ins, 1)), x.hull)
        return
    if t == "Unsqueeze":
        x = R(0)
        axes = _axes_of(node, ins, 1)
        if axes is None:
            raise KeyError(t)
        emit(0, _sr.unsqueeze(x.shape, axes), x.hull)
        return
    if t == "Expand":
        x = R(0)
        lo, hi = static_in(1)
        emit(0, _sr.expand(x.shape, _sr.dims_from_vector(lo, hi)), x.hull)
        return
    if t == "Tile":
        x = R(0)
        lo, hi = static_in(1)
        emit(0, _sr.tile(x.shape, _sr.dims_from_vector(lo, hi)), x.hull)
        return
    if t == "Concat":
        rs = [R(k) for k in range(len(node.input))]
        h = rs[0].hull
        for r in rs[1:]:
            h = _hull_union(h, r.hull)
        emit(0, _sr.concat([r.shape for r in rs], int(attrs["axis"])), h)
        return
    if t == "Slice":
        x = R(0)
        vals: List[Optional[List[int]]] = []
        for k in (1, 2, 3, 4):
            if k < len(node.input) and node.input[k]:
                got = _ints(static_in(k))
                if got is None:
                    raise KeyError(t)
                vals.append(got)
            else:
                vals.append(None)
        starts, ends, axes, steps = vals
        if starts is None or ends is None:
            raise KeyError(t)
        emit(0, _sr.slice_(x.shape, starts, ends, axes, steps), x.hull)
        return
    if t == "Pad":
        x = R(0)
        if len(node.input) > 3 and node.input[3]:
            raise KeyError(t)  # explicit axes
        pads = _ints(static_in(1))
        if pads is None:
            raise KeyError(t)
        rank = len(x.shape)
        shape = tuple(
            d + _sr.exact(pads[i] + pads[i + rank]) for i, d in enumerate(x.shape)
        )
        mode = attrs.get("mode", b"constant")
        mode = mode.decode() if isinstance(mode, bytes) else str(mode)
        h = x.hull
        if mode == "constant":
            cv: Hull = (0.0, 0.0)
            if len(node.input) > 2 and node.input[2]:
                cv = _hull_of(static_in(2))
            h = _hull_union(h, cv)
        emit(0, shape, h)
        return
    if t == "Gather":
        x, idx = R(0), R(1)
        emit(0, _sr.gather(x.shape, idx.shape, int(attrs.get("axis", 0))), x.hull)
        return
    if t == "GatherND":
        x, idx = R(0), R(1)
        emit(
            0,
            _sr.gather_nd(x.shape, idx.shape, int(attrs.get("batch_dims", 0))),
            x.hull,
        )
        return

    # ---- reductions: a ranged reduced dim becomes a ranged element count
    if t in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"):
        x = R(0)
        axes = _axes_of(node, ins, 1)
        keep = bool(int(attrs.get("keepdims", 1)))
        shape = _sr.reduce(x.shape, axes, keep)
        count = _sr.reduced_count(x.shape, axes)
        if t == "ReduceSum":
            h = _sum_hull(count, x.hull)
        elif count.lo >= 1:
            h = x.hull  # mean/max/min of >= 1 elements stays inside the hull
        else:
            h = _FULL  # possibly an empty reduction: no meaningful value
        emit(0, shape, h)
        return
    if t == "MatMul":
        a, b = R(0), R(1)
        shape = _sr.matmul(a.shape, b.shape)
        ka = a.shape[-1]
        kb = b.shape[-2] if len(b.shape) > 1 else b.shape[0]
        k = _sr.intersect(ka, kb) or ka
        emit(0, shape, _sum_hull(k, _hull_mul(a.hull, b.hull)))
        return
    raise KeyError(t)


def _wants_ranged(node: onnx.NodeProto, ins: List[Any], derived: set) -> bool:
    """Static operands, but the op needs the ranged-shape machinery (see ``_GENERATORS``)."""
    t = node.op_type
    if t in _GENERATORS:
        return True
    if t in ("Gather", "GatherND") and len(ins) > 1 and ins[1] is not None:
        # indices produced by a data-dependent op (e.g. NonZero): not constants, but any
        # index selects some element of the data, so the data hull is exact for the values
        return node.input[1] in derived and not _is_point(ins[1])
    if t in _HULL_OPS:
        # casts/comparisons of values that came from a data-dependent op: the static path
        # has no rule for them, the hull rule does
        return any(x in derived for x in node.input if x)
    if t == "Range":
        return any(i is not None and not _is_point(i) for i in ins)
    pos = _SHAPE_CONSUMERS.get(t)
    return (
        pos is not None
        and pos < len(ins)
        and ins[pos] is not None
        and not _is_point(ins[pos])
    )


def _input_dims(
    vi: onnx.ValueInfoProto, spec: Optional[Sequence[Any]]
) -> Optional[_sr.Shape]:
    """Ranged shape of a graph input from its declaration (or an explicit override)."""
    if spec is not None:
        out = []
        for s in spec:
            if s is None:
                out.append(_sr.Dim(0, None))
            elif isinstance(s, (int, np.integer)):
                out.append(_sr.exact(int(s)))
            else:
                out.append(_sr.rng(int(s[0]), None if s[1] is None else int(s[1])))
        return tuple(out)
    if not (vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape")):
        return None
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(_sr.exact(d.dim_value))
        else:
            dims.append(_sr.Dim(0, None, d.dim_param or None))
    return tuple(dims)


def propagate(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    input_shapes: Optional[Dict[str, Sequence[Any]]] = None,
) -> IntervalResult:
    """Propagate input boxes through ``model``; see the module docstring for guarantees.

    :param input_ranges: ``{input: (lo, hi)}`` (scalars or arrays). Merged over the
        model's own ``onnxsim.range.*`` annotations; an input with neither is unbounded.
    :param input_shapes: optional ``{input: [dim, ...]}`` overriding an input's shape,
        each ``dim`` an ``int``, a ``(lo, hi)`` range, or ``None`` (unbounded). An input
        whose shape is not fully static (a ``dim_param`` or an unknown dim) becomes a
        :class:`RangedTensor` (value hull only) instead of being dropped; tensors that
        depend on a data-dependent op (``NonZero``, ``TopK``, ...) do too. See
        :mod:`onnxsim.shape_ranges`.
    """
    runner = _Runner(model)
    g = model.graph
    annotated = _ranges.get_ranges(model)
    for k, (lo, hi) in (input_ranges or {}).items():
        annotated[k] = (
            np.asarray(lo, dtype=np.float64),
            np.asarray(hi, dtype=np.float64),
        )
    iv: Dict[str, Interval] = {
        t.name: _point(numpy_helper.to_array(t)) for t in g.initializer
    }
    try:
        inferred = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        inferred = g
    shapes = {}
    for vi in list(inferred.input) + list(inferred.value_info) + list(inferred.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = [
                d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None
                for d in vi.type.tensor_type.shape.dim
            ]
            if None not in dims:
                shapes[vi.name] = tuple(dims)  # fmt: skip
    ranged: Dict[str, RangedTensor] = {}
    for vi in g.input:
        if vi.name in iv:
            continue
        shape = shapes.get(vi.name)
        spec = (input_shapes or {}).get(vi.name)
        in_dims = _input_dims(vi, spec)
        if in_dims is not None and not _sr.is_static(in_dims):
            lo, hi = annotated.get(vi.name, (np.asarray(-_INF), np.asarray(_INF)))
            ranged[vi.name] = RangedTensor(
                in_dims, (float(np.min(lo)), float(np.max(hi)))
            )
            continue
        if in_dims is not None and spec is not None:
            shape = _sr.static_dims(in_dims)
        if shape is None:
            continue
        lo, hi = annotated.get(vi.name, (np.asarray(-_INF), np.asarray(_INF)))
        iv[vi.name] = (
            np.broadcast_to(lo, shape).astype(np.float64),
            np.broadcast_to(hi, shape).astype(np.float64),
        )
    unsupported: List[str] = []
    # Tensors computed from a ranged-shape tensor: integer-valued ones among them may be
    # shape arithmetic (element counts), where Div truncates.
    derived: set = set()

    for node in g.node:
        ins: List[Any] = [iv.get(x) if x else None for x in node.input]
        present = [i for i in ins if i is not None]
        outs = [o for o in node.output if o]
        in_ranged = [x for x in node.input if x and x in ranged]
        if any(x and x in derived or x in ranged for x in node.input if x):
            derived.update(outs)
        if node.input and any(
            x and iv.get(x) is None and x not in ranged for x in node.input
        ):
            continue  # an operand has no known interval/shape: leave outputs unknown
        t = node.op_type
        try:
            if node.domain not in ("", "ai.onnx"):
                raise KeyError(t)
            if in_ranged:
                _ranged_node(node, ins, iv, ranged, runner)
                continue
            if (
                present and all(_is_point(i) for i in present) and "" not in node.input
            ) or not node.input:
                res = (
                    runner.run(node, [i[0] for i in present])
                    if node.input
                    else runner.run(node, [])
                )
                for o, r in zip(outs, res):
                    iv[o] = _point(np.asarray(r))
                continue
            if _wants_ranged(node, ins, derived):
                _ranged_node(node, ins, iv, ranged, runner)
                derived.update(outs)
                continue
            if t == "Shape":
                iv[outs[0]] = _point(np.array(ins[0][0].shape, dtype=np.int64))
                continue
            if t in ("Add", "Sub", "Mul", "Div") and len(ins) == 2:
                (a0, a1), (b0, b1) = ins
                if t == "Add":
                    lo, hi = a0 + b0, a1 + b1
                elif t == "Sub":
                    lo, hi = a0 - b1, a1 - b0
                else:
                    with np.errstate(all="ignore"):
                        if t == "Div":
                            ok = (b0 > 0) | (b1 < 0)
                            b0, b1 = (
                                np.where(ok, 1.0 / b1, np.nan),
                                np.where(ok, 1.0 / b0, np.nan),
                            )
                        p = np.stack([a0 * b0, a0 * b1, a1 * b0, a1 * b1])
                        lo, hi = np.nanmin(p, 0), np.nanmax(p, 0)
                        if t == "Div":
                            lo, hi = np.where(ok, lo, -_INF), np.where(ok, hi, _INF)
                with np.errstate(all="ignore"):
                    lo, hi = (
                        np.where(np.isnan(lo), -_INF, lo),
                        np.where(np.isnan(hi), _INF, hi),
                    )
                if t == "Div" and any(x in derived for x in node.input):
                    # element counts are integers and ONNX integer Div truncates:
                    # a superset of both the real and the truncated quotient
                    lo, hi = np.floor(lo), np.ceil(hi)
                iv[outs[0]] = _widen(lo, hi)
            elif t == "Neg":
                iv[outs[0]] = (-ins[0][1], -ins[0][0])
            elif t in _BILINEAR and len(present) >= 2:
                if not all(_finite(i) for i in present):
                    raise _Unbounded(t)
                iv[outs[0]] = _bilinear(runner, node, present)
            elif t == "BatchNormalization":
                scale, bias, mean, var = (
                    ins[k][0].astype(np.float64) for k in range(1, 5)
                )
                s = scale / np.sqrt(var + float(_attrs(node).get("epsilon", 1e-5)))
                shp = [1, -1] + [1] * (ins[0][0].ndim - 2)
                s, b0, mu = s.reshape(shp), bias.reshape(shp), mean.reshape(shp)
                with np.errstate(all="ignore"):
                    a, b = (ins[0][0] - mu) * s + b0, (ins[0][1] - mu) * s + b0
                iv[outs[0]] = _widen(np.minimum(a, b), np.maximum(a, b))
            elif t in _LINEAR_AVG:
                if not _finite(ins[0]):
                    raise _Unbounded(t)
                lo, hi = (runner.run(node, [a.astype(np.float64)])[0] for a in ins[0])
                iv[outs[0]] = _widen(lo, hi)  # non-negative weights: apply to lo and hi
            elif t in ("Softmax",):
                iv[outs[0]] = (np.zeros(ins[0][0].shape), np.ones(ins[0][0].shape))
            elif (
                t in _MONOTONE
                or t == "LeakyRelu"
                and float(_attrs(node).get("alpha", 0.01)) >= 0
            ):
                extra = [i[0] for i in ins[1:]]
                with np.errstate(all="ignore"):
                    lo = runner.run(node, [ins[0][0]] + extra)[0]
                    hi = runner.run(node, [ins[0][1]] + extra)[0]
                if t not in _EXACT:  # transcendental: widen for float64 rounding
                    lo, hi = _widen(lo, hi)
                if t in _CODOMAIN:  # then never exceed the function's true range
                    lo, hi = (
                        np.maximum(lo, _CODOMAIN[t][0]),
                        np.minimum(hi, _CODOMAIN[t][1]),
                    )
                iv[outs[0]] = (lo, hi)
            elif t in _DATA_MOVE:
                data_ins = [i for i in ins]
                idx_point = (
                    all(_is_point(i) for i in data_ins[1:])
                    if t not in ("Concat", "Max", "Min")
                    else True
                )
                if t in ("Concat", "Max", "Min"):
                    lo = runner.run(node, [i[0] for i in data_ins])
                    hi = runner.run(node, [i[1] for i in data_ins])
                elif idx_point:
                    lo = runner.run(
                        node, [data_ins[0][0]] + [i[0] for i in data_ins[1:]]
                    )
                    hi = runner.run(
                        node, [data_ins[0][1]] + [i[0] for i in data_ins[1:]]
                    )
                else:
                    raise KeyError(t)
                for o, a, b in zip(outs, lo, hi):
                    iv[o] = (a, b)
            elif t == "Gather":
                if _is_point(ins[1]):
                    lo = runner.run(node, [ins[0][0], ins[1][0]])[0]
                    hi = runner.run(node, [ins[0][1], ins[1][0]])[0]
                    iv[outs[0]] = (lo, hi)
                else:
                    raise KeyError(t)
            else:
                raise KeyError(t)
        except Exception as e:  # unbounded operand, unsupported op, or an evaluator limitation: stay sound
            if not isinstance(e, _Unbounded) and t not in unsupported:
                unsupported.append(t)
            for o in outs:
                shape = shapes.get(o)
                if shape is not None:
                    iv[o] = _unbounded(shape)
    return IntervalResult(iv, unsupported, ranged)


# --------------------------------------------------------------------------
# Quantization bounds
# --------------------------------------------------------------------------

INT32_MAX = 2**31 - 1
FP32_EXACT_INT = 2**24


@dataclasses.dataclass
class LayerQuantBound:
    node: str
    op_type: str
    reduction_depth: int
    act_range: Tuple[float, float]
    act_scale: float
    act_zero_point: int
    acc_bound: int  # worst-case |int32 accumulator| from the activation range
    acc_bound_full_range: int  # what K*qmax_w*qmax_act would give, for comparison
    int32_safe: bool
    fp32_cast_exact: bool
    max_abs_error: (
        float  # certified worst-case |float - quantized| of any output element
    )
    output_width: float  # width of the output interval hull
    notes: str = ""

    @property
    def relative_error(self) -> float:
        # An unbounded output interval means the error cannot be compared to the output
        # range at all: that is "unknown" (nan), not a tiny ratio.
        if not np.isfinite(self.output_width):
            return float("nan")
        return (
            self.max_abs_error / self.output_width
            if self.output_width > 0
            else float("inf")
        )

    @property
    def tightening(self) -> float:
        return self.acc_bound_full_range / max(1, self.acc_bound)


def _weight_matrix(node: onnx.NodeProto, w: np.ndarray) -> Optional[np.ndarray]:
    """Weights as ``(out_channels, K)``."""
    t = node.op_type
    if t == "Conv":
        return w.reshape(w.shape[0], -1)
    if t == "MatMul":
        return w.reshape(-1, w.shape[-1]).T if w.ndim == 2 else None
    if t == "Gemm":
        b = w.T if not _attrs(node).get("transB", 0) else w
        return b  # (N, K) after the transB handling above
    return None


def quantization_bounds(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    weight_bits: int = 8,
    act_bits: int = 8,
    result: Optional[IntervalResult] = None,
) -> List[LayerQuantBound]:
    """Per-layer int-quantization bounds from interval analysis (see module docstring).

    Weights: symmetric per-output-channel. Activations: asymmetric, scale and
    zero point from the interval hull of the layer input (extended to include 0,
    as ONNX's QuantizeLinear conventions do). Layers whose input interval is
    unbounded, or whose weight is not a constant, are omitted.
    """
    result = result or propagate(model, input_ranges)
    consts = {
        t.name: numpy_helper.to_array(t).astype(np.float64)
        for t in model.graph.initializer
    }
    qw, qa = 2 ** (weight_bits - 1) - 1, 2**act_bits - 1
    out: List[LayerQuantBound] = []
    for k, node in enumerate(model.graph.node):
        if node.op_type not in ("Conv", "MatMul", "Gemm") or len(node.input) < 2:
            continue
        x, wname = node.input[0], node.input[1]
        if (
            wname not in consts
            or x not in result.intervals
            or node.output[0] not in result.intervals
        ):
            continue
        xi = result.intervals[x]
        if not _finite(xi):
            continue
        W = _weight_matrix(node, consts[wname])
        if W is None:
            continue
        lo, hi = min(float(xi[0].min()), 0.0), max(float(xi[1].max()), 0.0)
        scale_x = (hi - lo) / qa if hi > lo else 1.0
        zp = int(round(-lo / scale_x))
        xq_max = max(zp, qa - zp)
        s_w = np.maximum(np.abs(W).max(axis=1), 1e-30) / qw
        wq = np.round(W / s_w[:, None])
        kdepth = W.shape[1]
        acc = int(np.abs(wq).sum(axis=1).max() * xq_max)
        eps_x, eps_w = scale_x / 2.0, s_w / 2.0
        max_x = max(abs(lo), abs(hi))
        err = (
            eps_x * np.abs(W).sum(axis=1)
            + eps_w * kdepth * max_x
            + kdepth * eps_x * eps_w
        )
        oi = result.intervals[node.output[0]]
        width = float(oi[1].max() - oi[0].min()) if _finite(oi) else float("inf")
        out.append(
            LayerQuantBound(
                node=node.name or f"{node.op_type}_{k}",
                op_type=node.op_type,
                reduction_depth=kdepth,
                act_range=(lo, hi),
                act_scale=scale_x,
                act_zero_point=zp,
                acc_bound=acc,
                acc_bound_full_range=kdepth * qw * qa,
                int32_safe=acc <= INT32_MAX,
                fp32_cast_exact=acc <= FP32_EXACT_INT,
                max_abs_error=float(err.max()),
                output_width=width,
            )
        )
    return out


def format_quantization_report(bounds: List[LayerQuantBound]) -> str:
    rows = [
        f"{'node':28s} {'K':>6s} {'act range':>20s} {'acc bound':>12s} {'x tighter':>9s} {'int32':>5s} {'fp32':>5s} {'rel err':>8s}"
    ]
    for b in bounds:
        rows.append(
            f"{b.node[:28]:28s} {b.reduction_depth:6d} {f'[{b.act_range[0]:.3g}, {b.act_range[1]:.3g}]':>20s} "
            f"{b.acc_bound:12d} {b.tightening:9.2f} {'ok' if b.int32_safe else 'OVER':>5s} "
            f"{'exact' if b.fp32_cast_exact else 'round':>5s} {b.relative_error:8.3f}"
        )  # fmt: skip
    return "\n".join(rows)
