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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from . import ranges as _ranges

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
class IntervalResult:
    intervals: Dict[str, Interval]
    unsupported: List[str]  # op types that fell back to "unbounded"

    def hull(self, name: str) -> Tuple[float, float]:
        lo, hi = self.intervals[name]
        return float(np.min(lo)), float(np.max(hi))

    def contains(self, name: str, value: np.ndarray, slack: float = 1e-4) -> bool:
        """Is ``value`` inside the interval of ``name``, up to relative+absolute ``slack``?"""
        lo, hi = self.intervals[name]
        v = np.asarray(value, dtype=np.float64)
        pad = slack * (1.0 + np.abs(v))
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


def propagate(
    model: onnx.ModelProto, input_ranges: Optional[Dict[str, Tuple]] = None
) -> IntervalResult:
    """Propagate input boxes through ``model``; see the module docstring for guarantees.

    :param input_ranges: ``{input: (lo, hi)}`` (scalars or arrays). Merged over the
        model's own ``onnxsim.range.*`` annotations; an input with neither is unbounded.
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
    for vi in g.input:
        if vi.name in iv:
            continue
        shape = shapes.get(vi.name)
        if shape is None:
            continue
        lo, hi = annotated.get(vi.name, (np.asarray(-_INF), np.asarray(_INF)))
        iv[vi.name] = (
            np.broadcast_to(lo, shape).astype(np.float64),
            np.broadcast_to(hi, shape).astype(np.float64),
        )
    unsupported: List[str] = []

    for node in g.node:
        ins: List[Any] = [iv.get(x) if x else None for x in node.input]
        present = [i for i in ins if i is not None]
        outs = [o for o in node.output if o]
        if node.input and any(x and iv.get(x) is None for x in node.input):
            continue  # an operand has no known interval/shape: leave outputs unknown
        t = node.op_type
        try:
            if node.domain not in ("", "ai.onnx"):
                raise KeyError(t)
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
    return IntervalResult(iv, unsupported)


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
