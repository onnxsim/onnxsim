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

Beyond one linear relaxation per neuron:

* ``bab_bounds`` is branch and bound. It partitions the problem (bisect the input box with
  the ReluVal "smear" score, or branch on an unstable Relu neuron with the BaBSR score),
  bounds every region, and returns the union hull intersected with the root bound, so it
  is sound at any budget and never looser than the root. ``verify_output_ranges(method=
  "bab")`` uses it to prove ranges plain CROWN cannot. See the comment block above
  ``BabResult`` for why Relu branching without multipliers is sound but weak.
* ``leaf_method="beta"`` adds one multiplier per Relu branch constraint (the beta of
  beta-CROWN, Wang et al. 2021), optimised by gradient ascent together with the alpha
  slopes. A Lagrange multiplier on a constraint that holds on the true network, kept
  ``>= 0``, gives a valid bound for every value, so every iterate is sound and ``0`` is the
  plain bound (``_Analyzer._lower``'s ``extra`` argument carries the term).
* ``method="prima"`` / ``multi_neuron=k`` couples the ``k`` most influential unstable
  neurons of a layer pairwise through the facets of their *joint* hull (Singh et al.,
  "Beyond the Single Neuron Convex Barrier", POPL 2019; PRIMA, Müller et al. 2022), again
  as multipliers. The hull is taken over the polygon the two neurons can reach given CROWN
  bounds on ``z1 + z2`` and ``z1 - z2``; over a plain box the hull is a product of two
  triangles and there is nothing to add. The multi-neuron cuts here are pairs only
  (``k = 2``), with the facets found by brute force over the lifted cell vertices.

The multiplier-based methods need torch (optional, imported lazily); plain branch and
bound with CROWN leaves does not.
"""

import dataclasses
import itertools
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from . import interval as _interval
from . import ranges as _ranges

_BIG = 1e30  # stand-in for +-inf inside arithmetic (inf * 0 would be nan)
_HUGE = 1e30
_Range = Tuple[np.ndarray, np.ndarray]
_Forced = List[Tuple[int, int, int]]  # (Relu node index, flat neuron index, +1 / -1)
_ROW_BUDGET = 4_000_000  # max elements of one coefficient array (rows x tensor size)
_ALPHA_BUDGET = 2_000_000  # max total elements of per-row alpha parameters


@dataclasses.dataclass
class TensorBound:
    """Bounds of one tensor and which method produced them.

    ``method`` is ``"ibp"`` (interval only: requested, or CROWN had nothing to add),
    ``"crown"``, ``"alpha"`` or ``"bab"`` (branch-and-bound over CROWN/alpha bounds).
    """

    name: str
    lo: np.ndarray
    hi: np.ndarray
    method: str
    # branch-and-bound only (``method == "bab"``): did the budget run out while regions
    # could still be split, and into how many regions was the input box partitioned
    exhausted: bool = False
    regions: int = 1

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
        self.infeasible = (
            False  # a branch constraint was proven to exclude the whole box
        )
        self.maxel = max((int(v[0].size) for v in self.ib.values()), default=1)

    def clone(self) -> "_Analyzer":
        """A copy whose boxes/relaxations can be restricted without touching ``self``."""
        c = object.__new__(_Analyzer)
        c.__dict__.update(self.__dict__)
        c.ib = dict(self.ib)
        c._relax_cache = dict(self._relax_cache)
        c.infeasible = False
        return c

    def restrict(self, idx: int, flat: int, sign: int) -> bool:
        """Branch on one Relu neuron: ``sign > 0`` assumes its input >= 0, else <= 0.

        The neuron's pre-activation box is cut accordingly, so its relaxation becomes
        exact (identity or zero). Everything bounded afterwards is valid for the points
        of the original box that satisfy the branch constraint -- not for the whole box --
        which is all a branch-and-bound leaf needs. Returns False when the cut leaves an
        empty box (that branch has no points).
        """
        x = self.nodes[idx].input[0]
        lo, hi = self.ib[x]
        lo, hi = np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)
        if sign > 0:
            lo.flat[flat] = max(lo.flat[flat], 0.0)
        else:
            hi.flat[flat] = min(hi.flat[flat], 0.0)
        if lo.flat[flat] > hi.flat[flat]:
            return False
        self.ib[x] = (lo, hi)
        self._relax_cache.pop(idx, None)
        return True

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
        self,
        ops: Any,
        target: str,
        spec: Any,
        alpha: Optional[Dict[int, Any]] = None,
        leaf_cb: Optional[Dict[str, Any]] = None,
        relu_cb: Optional[Dict[int, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Lower bound of ``spec . target`` (``spec``: ``(m, *shape(target))``) over the box.

        ``leaf_cb`` / ``relu_cb`` (numpy backend only) receive, when given, the linear
        coefficients that reach each leaf tensor (``name -> (m, *shape)``) and the output
        of each Relu node (``node index -> (m, *shape)``): the sensitivities the
        branch-and-bound split heuristics rank by. They do not change the bound.

        ``extra`` carries Lagrange terms for constraints that hold on the true network
        (a Relu branch ``-s z <= 0``, a multi-neuron cut ``n . (z, y) <= d``): it adds a
        coefficient on the *input* ``extra["in"][idx]`` and on the *output*
        ``extra["out"][idx]`` of Relu node ``idx`` (each ``(m, *shape)``), plus a
        constant ``extra["const"]`` (``(m,)``). The constraint's multiplier must be
        non-negative: then ``spec . f >= spec . f + pi * (constraint)``, and the backward
        pass lower-bounds the right-hand side, so the result is a valid bound for every
        ``pi >= 0``, and ``pi = 0`` is the plain bound.
        """
        m = int(spec.shape[0])
        acc = [ops.zeros((m,))]
        if extra is not None and "const" in extra:
            acc[0] = acc[0] + extra["const"]
        coef: Dict[str, Any] = {}

        def row_sum(a: Any) -> Any:
            return ops.reshape(a, (m, -1)).sum(1)

        def add(name: str, a: Any) -> None:
            if name in self.leaf:
                if leaf_cb is not None:
                    leaf_cb[name] = leaf_cb[name] + a if name in leaf_cb else a
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
                if relu_cb is not None and t == "Relu":
                    relu_cb[idx] = a
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
                if extra is not None and t == "Relu" and idx in extra.get("out", {}):
                    a = a + extra["out"][idx]  # multiplier terms on the Relu output
                acc[0] = acc[0] + row_sum(ops.pos(a) * b_l + ops.neg(a) * b_u)
                nxt = ops.pos(a) * a_l + ops.neg(a) * a_u
                if extra is not None and t == "Relu" and idx in extra.get("in", {}):
                    nxt = nxt + extra["in"][idx]  # ... and on its input
                add(ins[0], nxt)
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

    def pair_bounds(
        self, idx: int, j1: int, j2: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """Bounds ``(ls, us, ld, ud)`` of ``z1 + z2`` and ``z1 - z2`` for two neurons of a Relu input.

        Computed by CROWN (backward through everything below), then intersected with what
        the two boxes imply, and widened by float slack. ``None`` when the Relu reads a
        leaf tensor directly: the two neurons are then independent and nothing couples them.
        """
        x = self.nodes[idx].input[0]
        if x in self.leaf:
            return None
        lo, hi = self.ib[x]
        shape = tuple(lo.shape)
        spec = np.zeros((4, int(np.prod(shape, dtype=np.int64))))
        spec[0, [j1, j2]] = 1.0
        spec[1, j1], spec[1, j2] = 1.0, -1.0
        spec[2], spec[3] = -spec[0], -spec[1]
        ops = _NumpyOps()
        got = ops.to_numpy(self._lower(ops, x, spec.reshape((4,) + shape)))
        l1, u1, l2, u2 = lo.flat[j1], hi.flat[j1], lo.flat[j2], hi.flat[j2]
        ls, us = max(got[0], l1 + l2), min(-got[2], u1 + u2)
        ld, ud = max(got[1], l1 - u2), min(-got[3], u1 - l2)
        vals = np.array([ls, us, ld, ud])
        if not np.all(np.isfinite(vals)) or ls > us or ld > ud:
            return None
        slack = 1e-9 * (1.0 + np.abs(vals))
        return (
            float(ls - slack[0]),
            float(us + slack[1]),
            float(ld - slack[2]),
            float(ud + slack[3]),
        )

    def refine(self, max_elems: int = 4096, force: bool = False) -> None:
        """Tighten every Relu/Sigmoid/Tanh input box with CROWN itself, layer by layer.

        ``force`` redoes it after branch restrictions. If a refined box comes out empty
        (clearly, beyond float rounding) the branch constraints exclude the whole input
        box: ``self.infeasible`` is set, and the caller may drop the branch.
        """
        if self._refined and not force:
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
            nlo, nhi = np.maximum(ilo, lo), np.minimum(ihi, hi)
            # conservative tolerance: only a clearly empty box counts as infeasible
            tol = 1e-7 * (1.0 + np.abs(nlo) + np.abs(nhi))
            if np.any(nlo > nhi + tol):
                self.infeasible = True
            self.ib[x] = (nlo, nhi)
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


_FACET_CACHE: Dict[Tuple[float, ...], Tuple[np.ndarray, np.ndarray]] = {}


def _pair_polygon(
    l1: float,
    u1: float,
    l2: float,
    u2: float,
    ls: float,
    us: float,
    ld: float,
    ud: float,
) -> np.ndarray:
    """Vertices (plus where the axes cut it) of ``{z in box, ls <= z1+z2 <= us, ld <= z1-z2 <= ud}``.

    The polygon is intersected with the lines ``z1 = 0`` and ``z2 = 0`` too, because the
    ReLU pair is affine inside each cell those axes cut it into. Candidates are all
    pairwise intersections of the boundary lines (and axes) that satisfy every constraint.
    """
    cons = np.array(
        [
            (-1.0, 0.0, -l1),
            (1.0, 0.0, u1),
            (0.0, -1.0, -l2),
            (0.0, 1.0, u2),
            (1.0, 1.0, us),
            (-1.0, -1.0, -ls),
            (1.0, -1.0, ud),
            (-1.0, 1.0, -ld),
        ]
    )
    lines = np.vstack([cons, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]])
    scale = max(1.0, float(np.abs(cons[:, 2]).max()))
    out: List[np.ndarray] = []
    for i, j in itertools.combinations(range(len(lines)), 2):
        det = lines[i, 0] * lines[j, 1] - lines[i, 1] * lines[j, 0]
        if abs(det) < 1e-12:
            continue
        z1 = (lines[i, 2] * lines[j, 1] - lines[i, 1] * lines[j, 2]) / det
        z2 = (lines[i, 0] * lines[j, 2] - lines[i, 2] * lines[j, 0]) / det
        if np.all(cons[:, 0] * z1 + cons[:, 1] * z2 <= cons[:, 2] + 1e-9 * scale):
            out.append(np.array([z1, z2]))
    if not out:
        return np.zeros((0, 2))
    pts = np.array(out)
    _, keep = np.unique(np.round(pts, 9), axis=0, return_index=True)
    return pts[np.sort(keep)]


def _pair_facets(
    l1: float,
    u1: float,
    l2: float,
    u2: float,
    ls: Optional[float] = None,
    us: Optional[float] = None,
    ld: Optional[float] = None,
    ud: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Facets ``n . (z1, z2, y1, y2) <= d`` of the hull of two Relus' joint graph.

    ``y_i = relu(z_i)`` and ``(z1, z2)`` ranges over the polygon of ``_pair_polygon``
    (the box when the sum/difference bounds are not given). The graph is piecewise linear
    on the cells the axes cut that polygon into and affine inside each, so it lies in the
    convex hull of the lifted cell vertices ``(z1, z2, relu z1, relu z2)``. Any halfspace
    containing those points therefore contains the whole graph, which is what makes every
    returned inequality valid regardless of how the normals were found. They are found by
    brute force: the hyperplane through every four of the points, kept when all points lie
    on one side (a facet of the hull). Only facets that couple the two neurons are kept.

    Over a plain box the hull is a product of two triangles and has no coupling facets:
    the gain of a multi-neuron relaxation comes entirely from the sum/difference bounds,
    i.e. from the two neurons being correlated through the layer below.
    """
    if ls is None:
        ls, us, ld, ud = l1 + l2, u1 + u2, l1 - u2, u1 - l2
    assert ls is not None and us is not None and ld is not None and ud is not None
    key = tuple(float(v) for v in (l1, u1, l2, u2, ls, us, ld, ud))
    if key in _FACET_CACHE:
        return _FACET_CACHE[key]
    q = _pair_polygon(l1, u1, l2, u2, ls, us, ld, ud)
    empty = (np.zeros((0, 4)), np.zeros(0))
    if len(q) < 4:
        _FACET_CACHE[key] = empty
        return empty
    pts = np.column_stack([q, np.maximum(q[:, 0], 0.0), np.maximum(q[:, 1], 0.0)])
    scale = max(1.0, float(np.abs(pts).max()))
    seen = set()
    facets = []
    for idx in itertools.combinations(range(len(pts)), 4):
        diff = pts[list(idx[1:])] - pts[idx[0]]
        _, sv, vt = np.linalg.svd(diff)
        if sv[2] <= 1e-9 * scale:  # the four points do not span a 3-D hyperplane
            continue
        n = vt[-1]
        vals = pts @ n
        c = float(vals[list(idx)].mean())
        tol = 1e-9 * scale
        if np.all(vals <= c + tol):
            pass
        elif np.all(vals >= c - tol):
            n, c = -n, -c
        else:
            continue
        n = n / np.linalg.norm(n)
        c = float(np.max(pts @ n))  # exact support value of the hull for this normal
        if max(abs(n[0]), abs(n[2])) < 1e-9 or max(abs(n[1]), abs(n[3])) < 1e-9:
            continue
        tag = tuple(np.round(np.append(n, c), 7))
        if tag in seen:
            continue
        seen.add(tag)
        # a margin covers the float error of the polygon vertices themselves
        facets.append((n, c + 1e-9 * scale))
    out = (
        np.array([f[0] for f in facets]).reshape(-1, 4),
        np.array([f[1] for f in facets]),
    )
    _FACET_CACHE[key] = out
    return out


def _alpha_bounds(
    an: _Analyzer,
    name: str,
    iters: int,
    lr: float,
    forced: Optional[_Forced] = None,
    pairs: Optional[List[Tuple[int, int, int]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """alpha-CROWN bounds of ``name``, or None when the problem is too large for per-row slopes.

    ``forced`` (Relu branches ``(node, neuron, sign)``) adds one multiplier per branch
    and row -- the beta of beta-CROWN -- and ``pairs`` (``(node, neuron1, neuron2)``)
    adds one multiplier per row for every facet of the two neurons' joint hull (a
    multi-neuron cut, PRIMA-style). Both are Lagrange multipliers of constraints that
    hold on the true network and are kept ``>= 0``, so every iterate is a valid bound
    (see ``_Analyzer._lower``), and the best one per row is kept, starting from zero
    multipliers: the result is never looser than without them.
    """
    ops = _TorchOps()
    torch = ops.t
    forced = list(forced or [])
    pairs = list(pairs or [])
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
    facets = []
    for idx, j1, j2 in pairs:
        lo, hi = an.ib[an.nodes[idx].input[0]]
        pb = an.pair_bounds(idx, j1, j2)
        if pb is None:
            continue
        n_, d_ = _pair_facets(lo.flat[j1], hi.flat[j1], lo.flat[j2], hi.flat[j2], *pb)
        if len(d_):
            facets.append(
                (
                    idx,
                    j1,
                    j2,
                    torch.tensor(n_, dtype=torch.float64),
                    torch.tensor(d_, dtype=torch.float64),
                )
            )
    n_facets = sum(int(f[4].shape[0]) for f in facets)

    def build_extra(beta: Any, pi: Any) -> Optional[Dict[str, Any]]:
        if not forced and not facets:
            return None
        ein: Dict[int, Any] = {}
        eout: Dict[int, Any] = {}
        const = torch.zeros(size, dtype=torch.float64)

        def slot(d: Dict[int, Any], idx: int) -> Any:
            if idx not in d:
                shp = tuple(an.ib[an.nodes[idx].input[0]][0].shape)
                d[idx] = torch.zeros((size,) + shp, dtype=torch.float64)
            return d[idx].view(size, -1)

        for c, (idx, flat, sign) in enumerate(forced):
            v = slot(ein, idx)
            v[:, flat] = v[:, flat] - float(sign) * beta[:, c]
        off = 0
        for idx, j1, j2, n_, d_ in facets:
            w = pi[:, off : off + n_.shape[0]]
            off += n_.shape[0]
            vi, vo = slot(ein, idx), slot(eout, idx)
            vi[:, j1] = vi[:, j1] + w @ n_[:, 0]
            vi[:, j2] = vi[:, j2] + w @ n_[:, 1]
            vo[:, j1] = vo[:, j1] + w @ n_[:, 2]
            vo[:, j2] = vo[:, j2] + w @ n_[:, 3]
            const = const - w @ d_
        return {"in": ein, "out": eout, "const": const}

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
        beta = torch.zeros((size, len(forced)), dtype=torch.float64, requires_grad=True)
        pi = torch.zeros((size, n_facets), dtype=torch.float64, requires_grad=True)
        mult = [t for t in (beta, pi) if t.numel()]

        def bound() -> Any:
            return an._lower(ops, name, spec, alpha, extra=build_extra(beta, pi))

        with torch.no_grad():
            best = bound().clone()
        if (alpha or mult) and iters > 0:
            groups: List[Dict[str, Any]] = []
            if alpha:
                groups.append({"params": list(alpha.values()), "lr": lr})
            if mult:  # multipliers live on the scale of the coefficients, not of [0, 1]
                groups.append({"params": mult, "lr": 5.0 * lr})
            opt = torch.optim.Adam(groups)
            for _ in range(iters):
                opt.zero_grad()
                lb = bound()
                if not lb.requires_grad:
                    # the parameters never reach this tensor (it sits below every Relu
                    # they belong to): the bound is constant, and already the best one
                    break
                (-lb.sum()).backward()
                opt.step()
                with torch.no_grad():
                    for a_ in alpha.values():
                        a_.clamp_(0.0, 1.0)
                    for m_ in mult:
                        m_.clamp_(min=0.0)
                    best = torch.maximum(best, lb.detach())
            with torch.no_grad():  # the final iterate too
                best = torch.maximum(best, bound())
        results.append(side * ops.to_numpy(best))
    lo, hi = results[0], results[1]
    return lo.reshape(shape), hi.reshape(shape)


# --------------------------------------------------------------------------
# Branch and bound
# --------------------------------------------------------------------------
#
# Plain CROWN relaxes every unstable Relu with a triangle, and the relaxation error
# does not shrink on its own. Branch and bound shrinks it by partitioning the problem
# into regions and bounding each one; the union of the regions' bounds is a bound for
# the whole box, and a region only has to be bounded as well as its own, smaller,
# relaxation allows. Two ways to partition, both sound:
#
# * input splitting (ReluVal, Wang et al. 2018; Neurify, 2018): bisect the input box
#   along the element whose width times influence (the "smear") is largest. Each half
#   is a smaller box, so every pre-activation box -- hence every relaxation -- shrinks.
# * Relu splitting (the branching of BaB, Bunel et al. 2018, with the BaBSR score; the
#   beta-CROWN paper, Wang et al. 2021, adds Lagrange multipliers that this module does
#   NOT implement): pick one unstable neuron and consider "input >= 0" and "input <= 0"
#   separately. In each branch the neuron is exact (identity or zero), so its triangle
#   gap disappears.
#
# Why Relu splitting without the multipliers is still sound. A branch's bound is a
# linear function of the input minimised over the *whole* input box, using lines that
# are valid only where the branch constraint holds (e.g. "relu(z) <= z" needs z >= 0).
# So the bound is valid for every point of the box that satisfies the branch
# constraint -- exactly the region the branch stands for -- and the two branches cover
# the box. What the multipliers would add is tightness (they put the constraint into the
# concretisation); what is lost without them is only that, never validity.
#
# Branches proven empty (a refined pre-activation box that is clearly empty) are
# dropped; an empty region contains no point, so dropping it cannot lose one. Every
# bound that is returned is the union hull over the surviving regions, intersected with
# the root bound, so it is never looser than plain CROWN (or alpha, with
# ``leaf_method="alpha"``) and is sound at any budget.


@dataclasses.dataclass
class BabResult:
    """Outcome of :func:`bab_bounds`.

    ``bounds`` is always a sound enclosure (the union hull over the final regions,
    intersected with the root CROWN/alpha bound). ``exhausted`` is True when the budget
    ran out while some region could still be split; ``evaluations`` counts bounded
    regions (the root and the infeasible branches included); ``regions`` is the size of
    the final partition; ``pruned`` is the number of branches proven empty. ``proved``
    is only set when a ``target`` was given: True when every region lies inside it.
    """

    bounds: Dict[str, TensorBound]
    exhausted: bool
    evaluations: int
    regions: int
    pruned: int
    proved: Optional[bool]
    split: str  # strategies actually used: "none", "input", "relu" or "relu+input"


class _Region:
    """One cell of the partition: an analyser restricted to it, and its bounds."""

    __slots__ = ("an", "ranges", "forced", "vals", "width", "violation", "stuck")

    def __init__(
        self,
        an: _Analyzer,
        ranges: Dict[str, _Range],
        forced: _Forced,
        vals: Dict[str, _Range],
        width: float,
        violation: float,
    ):
        self.an, self.ranges, self.forced = an, ranges, forced
        self.vals, self.width, self.violation = vals, width, violation
        self.stuck = False  # no split strategy applies to this region


def _finite_max(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.max(np.where(np.isfinite(a), a, _HUGE))) if a.size else 0.0


class _Bab:
    def __init__(
        self,
        model: onnx.ModelProto,
        input_ranges: Optional[Dict[str, Tuple]],
        names: List[str],
        split: str,
        leaf_method: str,
        alpha_iters: int,
        alpha_lr: float,
        target: Optional[Dict[str, _Range]],
        tol: float,
        multi_neuron: int = 0,
    ):
        self.model, self.names, self.split = model, names, split
        self.input_ranges = input_ranges
        self.leaf_method = leaf_method
        self.multi_neuron = multi_neuron
        self.alpha_iters, self.alpha_lr = alpha_iters, alpha_lr
        self.target, self.tol = target, tol
        self.used: set = set()
        inits = {t.name for t in model.graph.initializer}
        self.graph_inputs = [i.name for i in model.graph.input if i.name not in inits]

    # -- building and bounding a region ---------------------------------------
    def build(
        self, ranges: Optional[Dict[str, _Range]], forced: _Forced
    ) -> Optional[_Analyzer]:
        """An analyser for the input sub-box ``ranges`` with the Relu branches ``forced``."""
        an = _Analyzer(self.model, self.input_ranges if ranges is None else ranges)
        an.refine()
        for idx, flat, sign in forced:
            if not an.restrict(idx, flat, sign):
                return None
        if forced:
            an.refine(force=True)
        return None if an.infeasible else an

    def evaluate(self, an: _Analyzer, forced: _Forced) -> Dict[str, _Range]:
        out = {}
        for n in self.names:
            lo, hi = an.final(n)
            if self.leaf_method != "crown" and n not in an.leaf:
                pairs = None
                if self.multi_neuron >= 2:
                    pairs = self.choose_pairs(an, n, lo, hi)
                a = _alpha_bounds(
                    an,
                    n,
                    self.alpha_iters,
                    self.alpha_lr,
                    forced if self.leaf_method == "beta" else None,
                    pairs,
                )
                if a is not None:
                    alo, ahi = _finish(a[0], a[1], lo, hi)
                    lo, hi = np.maximum(lo, alo), np.minimum(hi, ahi)
            out[n] = (lo, hi)
        return out

    def choose_pairs(
        self, an: _Analyzer, n: str, lo: np.ndarray, hi: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Pairs among the most influential unstable neurons of the busiest Relu layer.

        Influence is the BaBSR-style |coefficient| x triangle gap for the widest output
        element; the ``multi_neuron`` best neurons of the layer with the largest total
        are coupled pairwise.
        """
        k = int(np.argmax(np.where(np.isfinite(hi - lo), hi - lo, _HUGE)))
        _, relu_infl = self._sensitivities(an, n, k)
        best: Optional[Tuple[float, int, np.ndarray]] = None
        for idx, infl in relu_infl.items():
            rx = an._relax(idx)
            score = (infl * rx["b_u"] * (rx["unstable"] > 0.5)).reshape(-1)
            top = np.argsort(-score)[: self.multi_neuron]
            top = top[score[top] > 0.0]
            total = float(score[top].sum())
            if len(top) >= 2 and (best is None or total > best[0]):
                best = (total, idx, top)
        if best is None:
            return []
        _, idx, top = best
        return [(idx, int(a), int(b)) for a, b in itertools.combinations(top, 2)]

    def _violation(self, n: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Elementwise amount by which ``[lo, hi]`` leaves the target (0 when inside)."""
        assert self.target is not None
        tlo, thi = self.target[n]
        return np.maximum(np.broadcast_to(tlo, lo.shape) - lo, 0.0) + np.maximum(
            hi - np.broadcast_to(thi, hi.shape), 0.0
        )

    def region(
        self, an: _Analyzer, ranges: Dict[str, _Range], forced: _Forced
    ) -> _Region:
        vals = self.evaluate(an, forced)
        width = max(_finite_max(hi - lo) for lo, hi in vals.values())
        viol = 0.0
        if self.target is not None:
            viol = max(
                _finite_max(self._violation(n, lo, hi)) for n, (lo, hi) in vals.items()
            )
        return _Region(an, ranges, forced, vals, width, viol)

    def undecided(self, r: _Region) -> bool:
        if self.target is None:
            return True  # tightening the hull: any splittable region is worth splitting
        return r.violation > self.tol

    # -- split heuristics -------------------------------------------------------
    def _worst(self, r: _Region) -> Optional[Tuple[str, int]]:
        """The output element to tighten: the widest (or most target-violating) one."""
        best: Optional[Tuple[float, str, int]] = None
        for n, (lo, hi) in r.vals.items():
            if n in r.an.leaf:
                continue
            v = (hi - lo) if self.target is None else self._violation(n, lo, hi)
            v = np.where(np.isfinite(v), v, _HUGE).reshape(-1)
            k = int(np.argmax(v))
            if best is None or v[k] > best[0]:
                best = (float(v[k]), n, k)
        if best is None or best[0] <= 0.0 or best[0] >= _HUGE:
            # nothing to tighten, or the worst element is unbounded (an unbounded input):
            # no amount of splitting the finite part can fix that
            return None
        return best[1], best[2]

    def _sensitivities(
        self, an: _Analyzer, n: str, k: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray]]:
        """|coefficient| reaching each input element / Relu neuron for output element ``k``.

        Summed over the lower and upper CROWN passes, so it ranks by how far the bound
        of that element can move, whichever side is the problem.
        """
        ops = _NumpyOps()
        shape = tuple(an.ib[n][0].shape)
        spec = np.zeros(int(np.prod(shape, dtype=np.int64)))
        spec[k] = 1.0
        spec = spec.reshape((1,) + shape)
        leaf_infl: Dict[str, np.ndarray] = {}
        relu_infl: Dict[int, np.ndarray] = {}
        for sign in (1.0, -1.0):
            leaf_cb: Dict[str, Any] = {}
            relu_cb: Dict[int, Any] = {}
            an._lower(ops, n, sign * spec, leaf_cb=leaf_cb, relu_cb=relu_cb)
            for name, a in leaf_cb.items():
                leaf_infl[name] = leaf_infl.get(name, 0.0) + np.abs(a[0])
            for idx, a in relu_cb.items():
                relu_infl[idx] = relu_infl.get(idx, 0.0) + np.abs(a[0])
        return leaf_infl, relu_infl

    def split_relu(self, r: _Region) -> Optional[Tuple[List[Optional[_Region]], int]]:
        """Branch on the unstable neuron with the largest |coefficient| x triangle gap.

        The BaBSR-style score: how much that neuron's relaxation gap can move the bound
        of the worst output element. Each branch fixes the neuron exactly.
        """
        worst = self._worst(r)
        if worst is None:
            return None
        _, relu_infl = self._sensitivities(r.an, *worst)
        best: Optional[Tuple[float, int, int]] = None
        for idx, infl in relu_infl.items():
            rx = r.an._relax(idx)
            score = (infl * rx["b_u"] * (rx["unstable"] > 0.5)).reshape(-1)
            j = int(np.argmax(score))
            if score[j] > 0.0 and (best is None or score[j] > best[0]):
                best = (float(score[j]), idx, j)
        if best is None:
            return None
        _, idx, j = best
        children: List[Optional[_Region]] = []
        for sign in (1, -1):
            an = r.an.clone()
            ok = an.restrict(idx, j, sign)
            if ok:
                an.refine(force=True)
            if not ok or an.infeasible:
                children.append(None)
                continue
            children.append(self.region(an, r.ranges, r.forced + [(idx, j, sign)]))
        self.used.add("relu")
        return children, 2

    def split_input(self, r: _Region) -> Optional[Tuple[List[Optional[_Region]], int]]:
        """Bisect the input element with the largest |coefficient| x width (ReluVal's smear)."""
        worst = self._worst(r)
        leaf_infl: Dict[str, np.ndarray] = {}
        if worst is not None:
            leaf_infl, _ = self._sensitivities(r.an, *worst)
        best: Optional[Tuple[float, str, int]] = None
        fallback: Optional[Tuple[float, str, int]] = None
        for name in self.graph_inputs:
            if name not in r.ranges:
                continue
            lo, hi = r.ranges[name]
            w = np.where(np.isfinite(hi - lo), hi - lo, -1.0).reshape(-1)
            if not np.any(w > 0.0):
                continue
            wj = int(np.argmax(w))
            if fallback is None or w[wj] > fallback[0]:
                fallback = (float(w[wj]), name, wj)
            infl = leaf_infl.get(name)
            if infl is not None:
                sc = np.where(
                    w > 0.0, np.broadcast_to(infl, hi.shape).reshape(-1) * w, -1.0
                )
                j = int(np.argmax(sc))
                if sc[j] > 0.0 and (best is None or sc[j] > best[0]):
                    best = (float(sc[j]), name, j)
        pick = best or fallback
        if pick is None:
            return None
        _, name, j = pick
        lo, hi = (np.array(a, dtype=np.float64) for a in r.ranges[name])
        mid = 0.5 * (lo.flat[j] + hi.flat[j])
        children: List[Optional[_Region]] = []
        for half in (0, 1):
            clo, chi = lo.copy(), hi.copy()
            if half == 0:
                chi.flat[j] = mid
            else:
                clo.flat[j] = mid
            ranges = dict(r.ranges)
            ranges[name] = (clo, chi)
            an = self.build(ranges, r.forced)
            children.append(None if an is None else self.region(an, ranges, r.forced))
        self.used.add("input")
        return children, 2

    def try_split(self, r: _Region) -> Optional[Tuple[List[Optional[_Region]], int]]:
        if self.split in ("relu", "auto"):
            got = self.split_relu(r)
            if got is not None:
                return got
        if self.split in ("input", "auto"):
            return self.split_input(r)
        return None

    # -- the search -----------------------------------------------------------
    def run(self, budget: int, time_limit: Optional[float]) -> BabResult:
        t0 = time.monotonic()
        root_an = self.build(None, [])
        assert root_an is not None
        ranges = {
            n: (
                np.array(root_an.ib[n][0], dtype=np.float64),
                np.array(root_an.ib[n][1], dtype=np.float64),
            )
            for n in self.graph_inputs
            if n in root_an.ib
        }
        root = self.region(root_an, ranges, [])
        regions = [root]
        evals, pruned = 1, 0

        def out_of_time() -> bool:
            return time_limit is not None and time.monotonic() - t0 > time_limit

        while evals + 2 <= budget and not out_of_time():  # a split bounds two regions
            cand = [r for r in regions if not r.stuck and self.undecided(r)]
            if not cand:
                break
            r = max(
                cand, key=lambda q: q.violation if self.target is not None else q.width
            )
            got = self.try_split(r)
            if got is None:
                r.stuck = True
                continue
            children, attempted = got
            evals += attempted
            regions.remove(r)
            for c in children:
                if c is None:
                    pruned += 1
                else:
                    regions.append(c)
        exhausted = (evals + 2 > budget or out_of_time()) and any(
            not r.stuck and self.undecided(r) for r in regions
        )
        result: Dict[str, TensorBound] = {}
        for n in self.names:
            rlo, rhi = root.vals[n]
            if regions:
                lo = np.min([r.vals[n][0] for r in regions], axis=0)
                hi = np.max([r.vals[n][1] for r in regions], axis=0)
                lo, hi = np.maximum(lo, rlo), np.minimum(hi, rhi)
            else:  # every branch empty: nothing is reachable, the root bound is vacuously fine
                lo, hi = rlo, rhi
            if self.used:
                label = "bab"
            elif self.multi_neuron >= 2:
                label = "prima"
            else:
                label = "crown" if self.leaf_method == "crown" else "alpha"
            result[n] = TensorBound(n, lo, hi, label, exhausted, max(1, len(regions)))
        proved: Optional[bool] = None
        if self.target is not None:
            proved = all(not self.undecided(r) for r in regions)
        used = "+".join(x for x in ("relu", "input") if x in self.used) or "none"
        return BabResult(result, exhausted, evals, len(regions), pruned, proved, used)


def _output_names(
    model: onnx.ModelProto, output: Union[None, str, Sequence[str]]
) -> List[str]:
    if output is None:
        return [o.name for o in model.graph.output]
    if isinstance(output, str):
        return [output]
    return list(output)


def bab_bounds(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    output: Union[None, str, Sequence[str]] = None,
    budget: int = 32,
    split: str = "auto",
    leaf_method: str = "crown",
    alpha_iters: int = 30,
    alpha_lr: float = 0.2,
    time_limit: Optional[float] = None,
    target: Optional[Dict[str, Tuple]] = None,
    tol: float = 0.0,
    multi_neuron: int = 0,
) -> BabResult:
    """Branch-and-bound bounds of ``output`` tensors; always sound, tighter with budget.

    :param budget: the maximum number of regions to bound. The root counts as one and
        every split bounds two more, so an even budget leaves one unused. Exhausting it is
        not an error: the result is the union over the regions reached so far, and
        ``exhausted`` says whether more splitting was still possible.
    :param split: ``"input"`` (bisect the input box, smear heuristic), ``"relu"`` (branch
        on unstable Relu neurons, BaBSR-style score), or ``"auto"`` (Relu splitting while
        a region has an unstable neuron to branch on, input splitting otherwise -- the
        only option for Sigmoid/Tanh nets, whose units cannot be branched on).
    :param leaf_method: how each region is bounded. ``"crown"``: plain CROWN, no
        multipliers (Relu branches then only fix one neuron's relaxation, which is sound
        but weak -- the branch constraint is not used in the concretisation). ``"alpha"``:
        optimised Relu slopes. ``"beta"``: slopes plus one multiplier per Relu branch
        constraint, the beta of beta-CROWN, which is what makes Relu splitting converge.
        ``"alpha"`` and ``"beta"`` need torch. The result is never looser than the root
        bound of the same method.
    :param multi_neuron: if ``>= 2``, additionally couple the ``multi_neuron`` most
        influential unstable neurons of one layer pairwise through the facets of their
        joint hull (PRIMA-style cuts, see ``_pair_facets``). Needs ``"alpha"`` or ``"beta"``.
    :param target: ``{output: (lo, hi)}`` to *prove*. Regions already inside it are not
        split, and the search stops as soon as every region is; ``proved`` reports it.
        Without a target the search tightens the hull until the budget is spent.
    :param tol: slack allowed when deciding a region lies inside the target.
    :param time_limit: optional wall-clock limit in seconds, checked between splits.
    """
    if split not in ("input", "relu", "auto"):
        raise ValueError(f"split must be 'input', 'relu' or 'auto', got {split!r}")
    if leaf_method not in ("crown", "alpha", "beta"):
        raise ValueError(
            f"leaf_method must be 'crown', 'alpha' or 'beta', got {leaf_method!r}"
        )
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if multi_neuron and (multi_neuron < 2 or leaf_method == "crown"):
        raise ValueError(
            "multi_neuron needs a value >= 2 and leaf_method 'alpha' or 'beta'"
        )
    if leaf_method != "crown":
        _TorchOps()  # fail early, with a clear message, when torch is missing
    names = _output_names(model, output)
    probe = _interval.propagate(model, input_ranges)
    for n in names:
        if n not in probe.intervals:
            raise ValueError(f"cannot analyse tensor {n!r}: its shape is unknown")
    tgt = None
    if target is not None:
        tgt = {
            n: (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
            for n, (lo, hi) in target.items()
        }
        missing = [n for n in names if n not in tgt]
        if missing:
            raise ValueError(f"target has no range for output(s) {missing}")
    bab = _Bab(
        model,
        input_ranges,
        names,
        split,
        leaf_method,
        alpha_iters,
        alpha_lr,
        tgt,
        tol,
        multi_neuron,
    )
    return bab.run(budget, time_limit)


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
    budget: int = 32,
    split: str = "auto",
    multi_neuron: int = 4,
) -> Dict[str, TensorBound]:
    """Sound bounds of ``output`` tensors (default: the graph outputs) over an input box.

    :param input_ranges: ``{input: (lo, hi)}`` overriding the model's ``onnxsim.range.*``
        annotations; an input with neither is unbounded.
    :param output: a tensor name or names; any tensor, not only graph outputs.
    :param method: ``"ibp"`` (interval propagation only), ``"crown"`` (default),
        ``"alpha"`` (CROWN + optimised ReLU slopes; needs torch) or ``"bab"`` (branch and
        bound over CROWN bounds, see :func:`bab_bounds`; ``budget`` and ``split`` apply) or
        ``"prima"`` (alpha-CROWN plus pairwise multi-neuron cuts on the ``multi_neuron``
        most influential unstable neurons; needs torch; no splitting).
        Each result carries the method that actually produced it: a tensor CROWN cannot
        improve on (an interval leaf, or a problem too large for per-row slopes under
        ``alpha``) reports ``"ibp"`` / ``"crown"`` accordingly, and ``"bab"`` reports
        ``"crown"`` when nothing could be split. The result is never looser than
        ``"ibp"``; ``"alpha"`` and ``"bab"`` are never looser than ``"crown"``.
    :param refine: tighten intermediate pre-activation boxes with CROWN first (slower,
        tighter). ``False`` uses interval boxes for every relaxation. Ignored by ``"bab"``,
        which always refines.
    """
    if method not in ("ibp", "crown", "alpha", "bab", "prima"):
        raise ValueError(
            f"method must be 'ibp', 'crown', 'alpha', 'bab' or 'prima', got {method!r}"
        )
    names = _output_names(model, output)
    if method == "prima":
        return bab_bounds(
            model,
            input_ranges,
            names,
            budget=1,
            leaf_method="alpha",
            alpha_iters=alpha_iters,
            alpha_lr=alpha_lr,
            multi_neuron=multi_neuron,
        ).bounds
    if method == "bab":
        return bab_bounds(
            model,
            input_ranges,
            names,
            budget=budget,
            split=split,
            alpha_iters=alpha_iters,
            alpha_lr=alpha_lr,
        ).bounds
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
    budget: int = 32,
    split: str = "auto",
) -> Dict[str, RangeVerdict]:
    """Check each annotated output range (``onnxsim.ranges``) holds for every input in the box.

    ``proved`` is True only when the *proven* bounds lie inside the annotation
    elementwise (up to ``tol``). ``False`` means *not proved* -- the bounds are sound
    but not necessarily tight, so the range may still hold; the ``hull`` shows what is
    proven. Outputs without an annotation are not reported.

    ``method="bab"`` tries to prove ranges that plain CROWN cannot, by splitting the
    problem into regions (see :func:`bab_bounds`): a region already inside the range is
    not split further and the search stops as soon as all of them are, or when
    ``budget`` regions have been bounded -- then ``proved`` is False, which still means
    only "not proved".
    """
    ann = _ranges.get_ranges(model)
    outs = [o.name for o in model.graph.output if o.name in ann]
    if not outs:
        return {}
    if method == "bab":
        got = bab_bounds(
            model,
            input_ranges,
            outs,
            budget=budget,
            split=split,
            target={n: ann[n] for n in outs},
            tol=tol,
        ).bounds
    else:
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
