"""Backward linear bound propagation (CROWN / DeepPoly) and alpha-CROWN for ONNX models.

``onnxsim.interval`` gives each tensor an elementwise ``[lo, hi]`` by propagating
intervals forward. That is sound but loses every correlation: a residual branch
``x + f(x)`` or two neurons fed by the same input are treated as independent, so
bounds widen with depth. This module computes tighter, still sound, bounds by
the method of

* CROWN -- Zhang, Weng, Chen, Hsieh, Daniel, "Efficient Neural Network Robustness
  Certification with General Activation Functions", NeurIPS 2018 (DeepPoly, Singh
  et al. 2019, is the same backward scheme for ReLU), and
* alpha-CROWN -- Xu et al., "Fast and Complete: Enabling Complete Neural Network
  Verification with Rapid and Massively Parallel Incomplete Verifiers", ICLR 2021,

written here from those papers. Nothing is taken from Luna, auto_LiRPA or
alpha,beta-CROWN.

How it works. Each nonlinear op is replaced, over the box its input can reach, by
two lines that sandwich it (``a_l x + b_l <= f(x) <= a_u x + b_u``). The network
then is, for bound purposes, an affine map of the input, and a bound on any output
row is a linear form in the input that is concretised over the input box. The
linear forms are built *backward* from the output, one op at a time, choosing the
lower or upper line per neuron by the sign of the coefficient reaching it; a tensor
used by several ops (a residual connection) simply accumulates the coefficients of
all its consumers, which is exactly how correlations are kept.

Soundness, so the numbers are not over-read:

* Every relaxation is valid for all x in the box it was built for. ReLU uses the
  exact triangle. Sigmoid/Tanh lines are *constructed by a candidate search but
  certified* on a grid, with a margin ``sup|f''| h^2 / 8`` that makes the line valid
  between grid points (a linear function minus ``f`` has second derivative ``-f''``),
  so a poor candidate costs tightness, never validity.
* Intermediate pre-activation boxes start from ``interval.propagate`` and are then
  refined layer by layer with CROWN itself, always intersected (both are valid).
* An op without a backward rule, a tensor with an unknown shape, or an unbounded
  box is not guessed at: the tensor becomes an *interval leaf* and is concretised
  with its interval enclosure (``+-inf`` where nothing is known).
* Final bounds are intersected with the interval bounds, so they are never looser
  than ``interval.propagate``, and widened by a few float64 ulps.
* Like ``interval``, this encloses the *real-number* function in float64. float32
  execution can exceed it by float32 rounding; ``slack`` in the tests is for that.

alpha-CROWN optimises the slope of the ReLU lower line for every unstable neuron
(``alpha`` in ``[0, 1]``) by Adam on the concretised bound. It needs torch (an
optional dependency, imported lazily); every iterate is a valid bound, and the best
one per row is kept, starting from plain CROWN's own slopes, so alpha-CROWN is never
looser than CROWN.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from . import interval as _interval
from . import ranges as _ranges

_BIG = 1e30  # stand-in for +-inf inside arithmetic (inf * 0 would be nan)
_ROW_BUDGET = 4_000_000  # max elements of one coefficient array (rows x tensor size)
_ALPHA_BUDGET = 2_000_000  # max total elements of per-row alpha parameters


@dataclasses.dataclass
class TensorBound:
    """Bounds of one tensor and which method produced them.

    ``method`` is ``"ibp"`` (interval only: requested, or CROWN had nothing to add),
    ``"crown"``, or ``"alpha"``.
    """

    name: str
    lo: np.ndarray
    hi: np.ndarray
    method: str

    def hull(self) -> Tuple[float, float]:
        return float(np.min(self.lo)), float(np.max(self.hi))


# --------------------------------------------------------------------------
# Array backends: the backward pass is written once against these.
# --------------------------------------------------------------------------


class _NumpyOps:
    name = "numpy"

    def asarray(self, a: Any) -> Any:
        return np.asarray(a, dtype=np.float64)

    def zeros(self, shape: Sequence[int]) -> Any:
        return np.zeros(tuple(shape), dtype=np.float64)

    def pos(self, a: Any) -> Any:
        return np.maximum(a, 0.0)

    def neg(self, a: Any) -> Any:
        return np.minimum(a, 0.0)

    def sum(self, a: Any, axes: Sequence[int], keepdims: bool = False) -> Any:
        return a.sum(axis=tuple(axes), keepdims=keepdims)

    def reshape(self, a: Any, shape: Sequence[int]) -> Any:
        return a.reshape(tuple(shape))

    def transpose(self, a: Any, perm: Sequence[int]) -> Any:
        return np.transpose(a, tuple(perm))

    def einsum(self, eq: str, *ops: Any) -> Any:
        return np.einsum(eq, *ops, optimize=True)

    def to_numpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)


class _TorchOps(_NumpyOps):
    name = "torch"

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "alpha-CROWN needs torch (pip install torch); method='crown' and 'ibp' do not"
            ) from e
        self.t = torch

    def asarray(self, a: Any) -> Any:
        if isinstance(a, self.t.Tensor):
            return a
        return self.t.as_tensor(np.array(a, dtype=np.float64))

    def zeros(self, shape: Sequence[int]) -> Any:
        return self.t.zeros(tuple(shape), dtype=self.t.float64)

    def pos(self, a: Any) -> Any:
        return a.clamp(min=0.0)

    def neg(self, a: Any) -> Any:
        return a.clamp(max=0.0)

    def sum(self, a: Any, axes: Sequence[int], keepdims: bool = False) -> Any:
        return a.sum(dim=tuple(axes), keepdim=keepdims)

    def reshape(self, a: Any, shape: Sequence[int]) -> Any:
        return a.reshape(tuple(shape))

    def transpose(self, a: Any, perm: Sequence[int]) -> Any:
        return a.permute(*perm)

    def einsum(self, eq: str, *ops: Any) -> Any:
        return self.t.einsum(eq, *ops)

    def to_numpy(self, a: Any) -> np.ndarray:
        return a.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Relaxations of the nonlinear ops
# --------------------------------------------------------------------------


def _clip_box(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.clip(lo, -_BIG, _BIG), np.clip(hi, -_BIG, _BIG)


def _relu_relax(lo: np.ndarray, hi: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact ReLU triangle. The lower slope of unstable neurons is left to the caller."""
    active = lo >= 0
    inactive = hi <= 0
    unstable = ~(active | inactive)
    den = np.where(unstable, hi - lo, 1.0)
    return {
        "a_u": np.where(active, 1.0, np.where(unstable, hi / den, 0.0)),
        "b_u": np.where(unstable, -lo * hi / den, 0.0),
        "a_l_stable": np.where(active, 1.0, 0.0),
        "b_l": np.zeros_like(lo),
        "unstable": unstable.astype(np.float64),
        # CROWN's adaptive choice: slope 1 if the positive part dominates, else 0
        "alpha0": np.where(unstable & (hi >= -lo), 1.0, 0.0),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# name -> (f, f' as a function of f(x), sup |f''|)
_SCURVES = {
    "Sigmoid": (_sigmoid, lambda f: f * (1.0 - f), 1.0 / (6.0 * np.sqrt(3.0))),
    "Tanh": (np.tanh, lambda f: 1.0 - f * f, 4.0 / (3.0 * np.sqrt(3.0))),
}


def _scurve_relax(
    kind: str, lo: np.ndarray, hi: np.ndarray, grid: int = 33
) -> Dict[str, np.ndarray]:
    """Sound line sandwich for Sigmoid/Tanh on ``[lo, hi]`` (see the module docstring).

    Candidates (secant, tangents at the ends/middle, the tangent through an endpoint, a
    constant) are each made valid by touching the function on a grid and adding the
    inter-grid margin; the candidate with the smallest mean gap is kept.
    """
    f, df, m2 = _SCURVES[kind]
    shape = lo.shape
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    out = {k: np.empty(lo.size) for k in ("a_l", "b_l", "a_u", "b_u")}
    block = 20000
    for s in range(0, lo.size, block):
        lb, ub = lo[s : s + block], hi[s : s + block]
        t = np.linspace(0.0, 1.0, grid)[None, :]
        xs = lb[:, None] + (ub - lb)[:, None] * t
        fx = f(xs)
        margin = m2 * ((ub - lb) / (grid - 1)) ** 2 / 8.0
        fl, fu, mid = f(lb), f(ub), 0.5 * (lb + ub)

        def tangent(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            fd = f(d)
            slope = df(fd)
            return slope, fd - slope * d

        width = ub - lb
        secant = np.where(
            width > 1e-300, (fu - fl) / np.where(width > 1e-300, width, 1.0), 0.0
        )
        mid_slope, mid_b = tangent(mid)
        sec_slope = np.where(width > 1e-300, secant, mid_slope)
        sec_b = np.where(width > 1e-300, fl - sec_slope * lb, mid_b)

        def through(point: np.ndarray, d_lo: float, d_hi: float) -> np.ndarray:
            """Tangent point d in [d_lo, d_hi] whose tangent passes through (point, f(point))."""
            a = np.full(point.shape, d_lo)
            b = np.full(point.shape, d_hi)
            fp = f(point)
            for _ in range(50):
                d = 0.5 * (a + b)
                fd = f(d)
                g = fd + df(fd) * (point - d) - fp
                # g increases with d on both half-lines used below
                a, b = np.where(g < 0, d, a), np.where(g < 0, b, d)
            return 0.5 * (a + b)

        d_up = np.clip(through(lb, 0.0, 60.0), lb, ub)
        d_lo = np.clip(through(ub, -60.0, 0.0), lb, ub)

        def pick(upper: bool) -> Tuple[np.ndarray, np.ndarray]:
            cands = [(sec_slope, sec_b)]
            cands += [
                tangent(lb),
                tangent(ub),
                (mid_slope, mid_b),
                tangent(d_up if upper else d_lo),
            ]
            best_score = None
            best_a = best_b = None
            for a, b0 in cands:
                gap = fx - (a[:, None] * xs + b0[:, None])  # f - line
                if upper:
                    b = b0 + gap.max(axis=1) + margin
                    score = ((a[:, None] * xs + b[:, None]) - fx).mean(axis=1)
                else:
                    b = b0 + gap.min(axis=1) - margin
                    score = (fx - (a[:, None] * xs + b[:, None])).mean(axis=1)
                if best_score is None:
                    best_score, best_a, best_b = score, a, b
                else:
                    better = score < best_score
                    best_score = np.where(better, score, best_score)
                    best_a = np.where(better, a, best_a)
                    best_b = np.where(better, b, best_b)
            # the constant line is exactly valid by monotonicity: no margin needed
            b_const = fu if upper else fl
            score = (
                (b_const[:, None] - fx) if upper else (fx - b_const[:, None])
            ).mean(axis=1)
            better = score < best_score
            return (
                np.where(better, 0.0, best_a),
                np.where(better, b_const, best_b),
            )

        out["a_u"][s : s + block], out["b_u"][s : s + block] = pick(True)
        out["a_l"][s : s + block], out["b_l"][s : s + block] = pick(False)
    return {k: v.reshape(shape) for k, v in out.items()}


# --------------------------------------------------------------------------
# The analyser
# --------------------------------------------------------------------------

_RESHAPE_LIKE = {"Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity"}
_NONLINEAR = {"Relu", "Sigmoid", "Tanh"}


def _unbroadcast(
    ops: Any, a: Any, m: int, shape_in: Tuple[int, ...], shape_out: Tuple[int, ...]
) -> Any:
    """Sum coefficient array ``a`` (``(m, *shape_out)``) down to ``(m, *shape_in)``."""
    lead = len(shape_out) - len(shape_in)
    axes = list(range(1, 1 + lead))
    axes += [
        1 + lead + i
        for i, d in enumerate(shape_in)
        if d == 1 and shape_out[lead + i] != 1
    ]
    if axes:
        a = ops.sum(a, axes, keepdims=True)
    return ops.reshape(a, (m,) + tuple(shape_in))


class _Analyzer:
    def __init__(
        self, model: onnx.ModelProto, input_ranges: Optional[Dict[str, Tuple]]
    ):
        self.model = model
        self.ibp = _interval.propagate(model, input_ranges)
        self.ib: Dict[str, Tuple[np.ndarray, np.ndarray]] = dict(self.ibp.intervals)
        self.nodes = list(model.graph.node)
        self.producer: Dict[str, int] = {}
        for i, n in enumerate(self.nodes):
            for o in n.output:
                if o:
                    self.producer[o] = i
        self.leaf = {name for name in self.ib if self._is_leaf_name(name)}
        self._relax_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._refined = False
        self.maxel = max((int(v[0].size) for v in self.ib.values()), default=1)

    # -- classification -----------------------------------------------------
    def _point(self, name: str) -> bool:
        iv = self.ib.get(name)
        return iv is not None and _interval._is_point(iv)

    def _const(self, name: str) -> Optional[np.ndarray]:
        iv = self.ib.get(name)
        if iv is not None and _interval._is_point(iv):
            return np.asarray(iv[0], dtype=np.float64)
        return None

    def _supported(self, node: onnx.NodeProto) -> bool:
        t = node.op_type
        if node.domain not in ("", "ai.onnx") or not node.output or not node.output[0]:
            return False
        if len(node.output) > 1 and any(node.output[1:]):
            return False
        ins = list(node.input)
        if any(x and x not in self.ib for x in ins) or node.output[0] not in self.ib:
            return False
        a = _interval._attrs(node)
        shp = lambda n: self.ib[n][0].shape  # noqa: E731
        if t in ("Relu", "Sigmoid", "Tanh", "Neg", "Transpose") or t in _RESHAPE_LIKE:
            if t == "Reshape" and self._const(ins[1]) is None:
                return False
            return not self._point(ins[0])
        if t == "Gemm":
            return (
                len(ins) >= 2
                and not self._point(ins[0])
                and self._const(ins[1]) is not None
                and (len(ins) < 3 or not ins[2] or self._const(ins[2]) is not None)
                and len(shp(ins[0])) == 2
            )
        if t == "MatMul":
            w = self._const(ins[1])
            return not self._point(ins[0]) and w is not None and w.ndim == 2
        if t == "Conv":
            w = self._const(ins[1])
            return (
                not self._point(ins[0])
                and w is not None
                and w.ndim == 4
                and len(shp(ins[0])) == 4
                and (len(ins) < 3 or not ins[2] or self._const(ins[2]) is not None)
                and a.get("auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET")
            )
        if t == "BatchNormalization":
            return (
                not self._point(ins[0])
                and all(self._const(x) is not None for x in ins[1:5])
                and not a.get("training_mode", 0)
                and len(shp(ins[0])) >= 2
            )
        if t in ("Add", "Sub"):
            return len(ins) == 2 and not (self._point(ins[0]) and self._point(ins[1]))
        if t == "Mul":
            return len(ins) == 2 and (self._point(ins[0]) != self._point(ins[1]))
        if t == "AveragePool":
            nd = len(shp(ins[0])) - 2
            return (
                not self._point(ins[0])
                and nd == 2
                and not any(a.get("pads", [0] * 4))
                and not a.get("ceil_mode", 0)
                and not any(d != 1 for d in a.get("dilations", [1, 1]))
            )
        if t == "GlobalAveragePool":
            return not self._point(ins[0]) and len(shp(ins[0])) == 4
        return False

    def _is_leaf_name(self, name: str) -> bool:
        if name not in self.producer:
            return True  # graph input or initializer
        if self._point(name):
            return True
        return not self._supported(self.nodes[self.producer[name]])

    # -- relaxations --------------------------------------------------------
    def _relax(self, idx: int) -> Dict[str, np.ndarray]:
        if idx not in self._relax_cache:
            node = self.nodes[idx]
            lo, hi = _clip_box(*self.ib[node.input[0]])
            if node.op_type == "Relu":
                self._relax_cache[idx] = _relu_relax(lo, hi)
            else:
                self._relax_cache[idx] = _scurve_relax(node.op_type, lo, hi)
        return self._relax_cache[idx]

    # -- the backward pass ---------------------------------------------------
    def _lower(
        self, ops: Any, target: str, spec: Any, alpha: Optional[Dict[int, Any]] = None
    ) -> Any:
        """Lower bound of ``spec . target`` (``spec``: ``(m, *shape(target))``) over the box."""
        m = int(spec.shape[0])
        acc = [ops.zeros((m,))]
        coef: Dict[str, Any] = {}

        def row_sum(a: Any) -> Any:
            return ops.reshape(a, (m, -1)).sum(1)

        def add(name: str, a: Any) -> None:
            if name in self.leaf:
                lo, hi = _clip_box(*self.ib[name])
                acc[0] = acc[0] + row_sum(
                    ops.pos(a) * ops.asarray(lo) + ops.neg(a) * ops.asarray(hi)
                )
            elif name in coef:
                coef[name] = coef[name] + a
            else:
                coef[name] = a

        add(target, spec)
        start = self.producer.get(target, -1)
        for idx in range(start, -1, -1):
            node = self.nodes[idx]
            out = node.output[0] if node.output else ""
            if out not in coef:
                continue
            a = coef.pop(out)
            t = node.op_type
            ins = list(node.input)
            xs = lambda k: tuple(self.ib[ins[k]][0].shape)  # noqa: E731
            if t in ("Relu", "Sigmoid", "Tanh"):
                r = self._relax(idx)
                if t == "Relu":
                    base = ops.asarray(r["a_l_stable"])
                    if alpha is not None and idx in alpha:
                        a_l = base + ops.asarray(r["unstable"]) * alpha[idx]
                    else:
                        a_l = base + ops.asarray(r["unstable"] * r["alpha0"])
                else:
                    a_l = ops.asarray(r["a_l"])
                a_u, b_l, b_u = (
                    ops.asarray(r["a_u"]),
                    ops.asarray(r["b_l"]),
                    ops.asarray(r["b_u"]),
                )
                acc[0] = acc[0] + row_sum(ops.pos(a) * b_l + ops.neg(a) * b_u)
                add(ins[0], ops.pos(a) * a_l + ops.neg(a) * a_u)
            elif t in _RESHAPE_LIKE:
                add(ins[0], ops.reshape(a, (m,) + xs(0)))
            elif t == "Neg":
                add(ins[0], -a)
            elif t == "Transpose":
                perm = list(
                    _interval._attrs(node).get("perm") or reversed(range(len(xs(0))))
                )
                inv = [perm.index(i) for i in range(len(perm))]
                add(ins[0], ops.transpose(a, [0] + [1 + j for j in inv]))
            elif t == "Gemm":
                at = _interval._attrs(node)
                b = self._const(ins[1])
                assert b is not None
                bt = b.T if at.get("transB", 0) else b  # (K, N)
                ax = float(at.get("alpha", 1.0)) * (a @ ops.asarray(bt.T))  # (m, M, K)
                if at.get("transA", 0):
                    ax = ops.transpose(ax, [0, 2, 1])
                if len(ins) > 2 and ins[2]:
                    c = self._const(ins[2])
                    assert c is not None
                    cb = np.broadcast_to(c, tuple(self.ib[out][0].shape))
                    acc[0] = acc[0] + float(at.get("beta", 1.0)) * row_sum(
                        a * ops.asarray(cb)
                    )
                add(ins[0], ax)
            elif t == "MatMul":
                w = self._const(ins[1])
                assert w is not None
                add(ins[0], a @ ops.asarray(w.T))
            elif t == "Conv":
                self._conv_back(ops, node, a, m, acc, add, row_sum)
            elif t == "BatchNormalization":
                scale, bias, mean, var = (self._const(x) for x in ins[1:5])
                assert (
                    scale is not None
                    and bias is not None
                    and mean is not None
                    and var is not None
                )
                s = scale / np.sqrt(
                    var + float(_interval._attrs(node).get("epsilon", 1e-5))
                )
                shape = (1, 1, -1) + (1,) * (len(xs(0)) - 2)
                tb = (bias - mean * s).reshape(shape[1:])
                acc[0] = acc[0] + row_sum(
                    a * ops.asarray(np.broadcast_to(tb, xs(0))[None])
                )
                add(ins[0], a * ops.asarray(s.reshape(shape)))
            elif t in ("Add", "Sub"):
                sign = 1.0 if t == "Add" else -1.0
                oshape = tuple(self.ib[out][0].shape)
                for k, sg in ((0, 1.0), (1, sign)):
                    c = self._const(ins[k])
                    if c is not None:
                        acc[0] = acc[0] + sg * row_sum(
                            a * ops.asarray(np.broadcast_to(c, oshape)[None])
                        )
                    else:
                        add(ins[k], _unbroadcast(ops, sg * a, m, xs(k), oshape))
            elif t == "Mul":
                k = 0 if self._const(ins[0]) is None else 1
                c = self._const(ins[1 - k])
                assert c is not None
                oshape = tuple(self.ib[out][0].shape)
                add(ins[k], _unbroadcast(ops, a * ops.asarray(c), m, xs(k), oshape))
            elif t == "AveragePool":
                self._avgpool_back(ops, node, a, m, add)
            elif t == "GlobalAveragePool":
                n, c_, h, w_ = xs(0)
                full = ops.zeros((m, n, c_, h, w_)) + a / float(h * w_)
                add(ins[0], full)
            else:  # pragma: no cover - _supported() keeps these out of the graph walk
                raise AssertionError(f"no backward rule for {t}")
        return acc[0]

    def _conv_back(
        self,
        ops: Any,
        node: onnx.NodeProto,
        a: Any,
        m: int,
        acc: List[Any],
        add: Any,
        row_sum: Any,
    ) -> None:
        ins = list(node.input)
        at = _interval._attrs(node)
        w = self._const(ins[1])
        assert w is not None
        n, c, h, wd = self.ib[ins[0]][0].shape
        mo, cg, kh, kw = w.shape
        st = list(at.get("strides", [1, 1]))
        dl = list(at.get("dilations", [1, 1]))
        pd = list(at.get("pads", [0, 0, 0, 0]))
        group = int(at.get("group", 1))
        oh, ow = a.shape[-2], a.shape[-1]
        if len(ins) > 2 and ins[2]:
            b = self._const(ins[2])
            assert b is not None
            acc[0] = acc[0] + ops.einsum("mnqij,q->m", a, ops.asarray(b))
        hp, wp = h + pd[0] + pd[2], wd + pd[1] + pd[3]
        out = ops.zeros((m, n, c, hp, wp))
        mg = mo // group
        for g in range(group):
            ag = a[:, :, g * mg : (g + 1) * mg]
            wg = ops.asarray(w[g * mg : (g + 1) * mg])
            for ki in range(kh):
                for kj in range(kw):
                    contrib = ops.einsum("mnqij,qc->mncij", ag, wg[:, :, ki, kj])
                    i0, j0 = ki * dl[0], kj * dl[1]
                    out[
                        :,
                        :,
                        g * cg : (g + 1) * cg,
                        i0 : i0 + st[0] * (oh - 1) + 1 : st[0],
                        j0 : j0 + st[1] * (ow - 1) + 1 : st[1],
                    ] += contrib
        add(ins[0], out[:, :, :, pd[0] : pd[0] + h, pd[1] : pd[1] + wd])

    def _avgpool_back(
        self, ops: Any, node: onnx.NodeProto, a: Any, m: int, add: Any
    ) -> None:
        at = _interval._attrs(node)
        n, c, h, wd = self.ib[node.input[0]][0].shape
        kh, kw = at["kernel_shape"]
        st = list(at.get("strides", [1, 1]))
        oh, ow = a.shape[-2], a.shape[-1]
        out = ops.zeros((m, n, c, h, wd))
        for ki in range(kh):
            for kj in range(kw):
                out[
                    :,
                    :,
                    :,
                    ki : ki + st[0] * (oh - 1) + 1 : st[0],
                    kj : kj + st[1] * (ow - 1) + 1 : st[1],
                ] += a / float(kh * kw)
        add(node.input[0], out)

    # -- bounds of a tensor --------------------------------------------------
    def _rows_per_chunk(self) -> int:
        return max(1, _ROW_BUDGET // max(1, self.maxel))

    def tensor_bounds(
        self, name: str, ops: Any = None, alpha_cb: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CROWN bounds of ``name`` (identity spec, chunked over rows), as numpy arrays."""
        ops = ops or _NumpyOps()
        shape = tuple(self.ib[name][0].shape)
        size = int(np.prod(shape, dtype=np.int64))
        lo = np.empty(size)
        hi = np.empty(size)
        step = self._rows_per_chunk()
        for s in range(0, size, step):
            k = min(step, size - s)
            eye = np.zeros((k, size))
            eye[np.arange(k), s + np.arange(k)] = 1.0
            spec = ops.asarray(eye.reshape((k,) + shape))
            lo[s : s + k] = ops.to_numpy(self._lower(ops, name, spec))
            hi[s : s + k] = -ops.to_numpy(self._lower(ops, name, -spec))
        return lo.reshape(shape), hi.reshape(shape)

    def refine(self, max_elems: int = 4096) -> None:
        """Tighten every Relu/Sigmoid/Tanh input box with CROWN itself, layer by layer."""
        if self._refined:
            return
        self._refined = True
        for idx, node in enumerate(self.nodes):
            if node.op_type not in _NONLINEAR or not self._supported(node):
                continue
            x = node.input[0]
            if x in self.leaf or self.ib[x][0].size > max_elems:
                continue
            lo, hi = self.tensor_bounds(x)
            ilo, ihi = self.ib[x]
            self.ib[x] = (np.maximum(ilo, lo), np.minimum(ihi, hi))
            self._relax_cache.pop(idx, None)

    def final(self, name: str, ops: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """CROWN bounds of ``name`` intersected with the interval bounds, then widened."""
        ilo, ihi = self.ibp.intervals[name]
        if name in self.leaf:
            return np.array(ilo, dtype=np.float64), np.array(ihi, dtype=np.float64)
        lo, hi = self.tensor_bounds(name, ops)
        return _finish(lo, hi, ilo, ihi)


def _finish(
    lo: np.ndarray, hi: np.ndarray, ilo: np.ndarray, ihi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean ``+-BIG``/nan, intersect with the interval bounds, widen by a few ulps."""
    lo = np.where(np.isnan(lo) | (lo <= -_BIG * 0.1), -np.inf, lo)
    hi = np.where(np.isnan(hi) | (hi >= _BIG * 0.1), np.inf, hi)
    lo, hi = _interval._widen(lo, hi)
    lo, hi = np.maximum(lo, ilo), np.minimum(hi, ihi)
    return lo, hi


# --------------------------------------------------------------------------
# alpha-CROWN
# --------------------------------------------------------------------------


def _alpha_bounds(
    an: _Analyzer, name: str, iters: int, lr: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """alpha-CROWN bounds of ``name``, or None when the problem is too large for per-row slopes."""
    ops = _TorchOps()
    torch = ops.t
    shape = tuple(an.ib[name][0].shape)
    size = int(np.prod(shape, dtype=np.int64))
    if size * an.maxel > _ROW_BUDGET:
        return None
    relus = [
        i
        for i, n in enumerate(an.nodes)
        if n.op_type == "Relu" and an._supported(n) and i <= an.producer.get(name, -1)
    ]
    relus = [i for i in relus if an._relax(i)["unstable"].any()]
    total_unstable = sum(int(an._relax(i)["unstable"].size) for i in relus) * size
    per_row = total_unstable <= _ALPHA_BUDGET
    eye = np.eye(size).reshape((size,) + shape)
    results = []
    for side in (1.0, -1.0):
        spec = ops.asarray(side * eye)
        alpha = {}
        for i in relus:
            r = an._relax(i)
            init = (
                r["alpha0"]
                if not per_row
                else np.broadcast_to(r["alpha0"], (size,) + r["alpha0"].shape).copy()
            )
            alpha[i] = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            best = an._lower(ops, name, spec, alpha).clone()
        if alpha and iters > 0:
            opt = torch.optim.Adam(list(alpha.values()), lr=lr)
            for _ in range(iters):
                opt.zero_grad()
                lb = an._lower(ops, name, spec, alpha)
                (-lb.sum()).backward()
                opt.step()
                with torch.no_grad():
                    for a_ in alpha.values():
                        a_.clamp_(0.0, 1.0)
                    best = torch.maximum(best, lb.detach())
            with torch.no_grad():  # the final iterate too
                best = torch.maximum(best, an._lower(ops, name, spec, alpha))
        results.append(side * ops.to_numpy(best))
    lo, hi = results[0], results[1]
    return lo.reshape(shape), hi.reshape(shape)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def bounds(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    output: Union[None, str, Sequence[str]] = None,
    method: str = "crown",
    refine: bool = True,
    alpha_iters: int = 30,
    alpha_lr: float = 0.2,
) -> Dict[str, TensorBound]:
    """Sound bounds of ``output`` tensors (default: the graph outputs) over an input box.

    :param input_ranges: ``{input: (lo, hi)}`` overriding the model's ``onnxsim.range.*``
        annotations; an input with neither is unbounded.
    :param output: a tensor name or names; any tensor, not only graph outputs.
    :param method: ``"ibp"`` (interval propagation only), ``"crown"`` (default) or
        ``"alpha"`` (CROWN + optimised ReLU slopes; needs torch). Each result carries
        the method that actually produced it: a tensor CROWN cannot improve on (an
        interval leaf, or a problem too large for per-row slopes under ``alpha``)
        reports ``"ibp"`` / ``"crown"`` accordingly. The result is never looser than
        ``"ibp"``, and ``"alpha"`` is never looser than ``"crown"``.
    :param refine: tighten intermediate pre-activation boxes with CROWN first (slower,
        tighter). ``False`` uses interval boxes for every relaxation.
    """
    if method not in ("ibp", "crown", "alpha"):
        raise ValueError(f"method must be 'ibp', 'crown' or 'alpha', got {method!r}")
    names: List[str]
    if output is None:
        names = [o.name for o in model.graph.output]
    elif isinstance(output, str):
        names = [output]
    else:
        names = list(output)
    an = _Analyzer(model, input_ranges)
    if method == "alpha":
        _TorchOps()  # fail early, with a clear message, when torch is missing
    for n in names:
        if n not in an.ibp.intervals:
            raise ValueError(f"cannot analyse tensor {n!r}: its shape is unknown")
    result: Dict[str, TensorBound] = {}
    if method != "ibp" and refine:
        an.refine()
    for n in names:
        ilo, ihi = an.ibp.intervals[n]
        ilo, ihi = np.asarray(ilo, dtype=np.float64), np.asarray(ihi, dtype=np.float64)
        if method == "ibp" or n in an.leaf:
            result[n] = TensorBound(n, ilo, ihi, "ibp")
            continue
        lo, hi = an.final(n)
        used = "crown"
        if method == "alpha":
            alpha = _alpha_bounds(an, n, alpha_iters, alpha_lr)
            if alpha is not None:
                alo, ahi = _finish(alpha[0], alpha[1], lo, hi)
                lo, hi = np.maximum(lo, alo), np.minimum(hi, ahi)
                used = "alpha"
        result[n] = TensorBound(n, lo, hi, used)
    return result


@dataclasses.dataclass
class RangeVerdict:
    """Whether an annotated range is *proven* for the whole input box."""

    name: str
    proved: bool
    hull: Tuple[float, float]  # proven bounds of the output over the box
    annotated: Tuple[float, float]
    method: str


def verify_output_ranges(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    method: str = "crown",
    tol: float = 0.0,
) -> Dict[str, RangeVerdict]:
    """Check each annotated output range (``onnxsim.ranges``) holds for every input in the box.

    ``proved`` is True only when the *proven* bounds lie inside the annotation
    elementwise (up to ``tol``). ``False`` means *not proved* -- the bounds are sound
    but not necessarily tight, so the range may still hold; the ``hull`` shows what is
    proven. Outputs without an annotation are not reported.
    """
    ann = _ranges.get_ranges(model)
    outs = [o.name for o in model.graph.output if o.name in ann]
    if not outs:
        return {}
    got = bounds(model, input_ranges, outs, method)
    verdicts: Dict[str, RangeVerdict] = {}
    for n in outs:
        tb = got[n]
        rlo, rhi = ann[n]
        ok = bool(
            np.all(tb.lo >= np.broadcast_to(rlo, tb.lo.shape) - tol)
            and np.all(tb.hi <= np.broadcast_to(rhi, tb.hi.shape) + tol)
        )
        verdicts[n] = RangeVerdict(
            n, ok, tb.hull(), (float(np.min(rlo)), float(np.max(rhi))), tb.method
        )
    return verdicts


@dataclasses.dataclass
class QuantBoundComparison:
    """Interval vs CROWN-tightened quantization bounds of one layer."""

    node: str
    op_type: str
    interval: _interval.LayerQuantBound
    tight: _interval.LayerQuantBound

    @property
    def acc_tightening(self) -> float:
        return self.interval.acc_bound / max(1, self.tight.acc_bound)

    @property
    def error_tightening(self) -> float:
        t = self.tight.max_abs_error
        return self.interval.max_abs_error / t if t > 0 else float("inf")


def quantization_bounds_tight(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    weight_bits: int = 8,
    act_bits: int = 8,
    refine: bool = True,
) -> List[QuantBoundComparison]:
    """``interval.quantization_bounds`` fed with CROWN-tightened activation ranges.

    The int32 accumulator bound and the certified rounding error both scale with the
    activation range feeding a layer; correlated branches make the interval range
    loose, so tighter activation ranges tighten both. Returns one comparison per layer
    ``interval.quantization_bounds`` reports, with the interval numbers alongside.
    """
    an = _Analyzer(model, input_ranges)
    plain = _interval.quantization_bounds(
        model, input_ranges, weight_bits, act_bits, result=an.ibp
    )
    if refine:
        an.refine()
    intervals = dict(an.ibp.intervals)
    wanted = set()
    for node in model.graph.node:
        if (
            node.op_type in ("Conv", "MatMul", "Gemm")
            and len(node.input) >= 2
            and node.output
        ):
            wanted.update([node.input[0], node.output[0]])
    for n in wanted:
        if n in an.ibp.intervals and n not in an.leaf:
            lo, hi = an.final(n)
            intervals[n] = (lo, hi)
    tight_result = _interval.IntervalResult(intervals, list(an.ibp.unsupported))
    tight = _interval.quantization_bounds(
        model, input_ranges, weight_bits, act_bits, result=tight_result
    )
    by_node = {b.node: b for b in tight}
    return [
        QuantBoundComparison(p.node, p.op_type, p, by_node[p.node])
        for p in plain
        if p.node in by_node
    ]


def format_quantization_comparison(rows: List[QuantBoundComparison]) -> str:
    head = f"{'node':24s} {'act range (ibp)':>20s} {'act range (crown)':>20s} {'acc ibp':>10s} {'acc crown':>10s} {'x':>6s} {'err x':>7s}"
    lines = [head]
    for r in rows:
        lines.append(
            f"{r.node[:24]:24s} {f'[{r.interval.act_range[0]:.3g}, {r.interval.act_range[1]:.3g}]':>20s} "
            f"{f'[{r.tight.act_range[0]:.3g}, {r.tight.act_range[1]:.3g}]':>20s} "
            f"{r.interval.acc_bound:10d} {r.tight.acc_bound:10d} {r.acc_tightening:6.2f} {r.error_tightening:7.2f}"
        )  # fmt: skip
    return "\n".join(lines)
