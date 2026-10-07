"""Taylor-model arithmetic and the mean-value form for ONNX models.

Intervals, zonotopes and CROWN all enclose a nonlinear op by *linear* functions. For a
smooth op (Softmax, GELU, Sigmoid/Tanh, Exp/Log/Sqrt, LayerNorm, attention scores) over a
narrow box that wastes most of the information the op carries: the curvature. A
**Taylor model** (Makino & Berz) keeps a value as a low-degree polynomial in the input
perturbations plus a certified interval remainder,

    x_i = c_i + sum_j L[j, i] eps_j + sum_{j<=k < Q} Qd[(j,k), i] eps_j eps_k + [rlo_i, rhi_i],
    eps_j in [-1, 1],

and a smooth function is applied by expanding it around the centre and bounding the
Lagrange remainder over the argument's range. That is how Softmax / LayerNorm / attention
get *quadratically* tight enclosures where a linear relaxation is only first-order tight.

The **mean-value form** (``mean_value_bounds``) is the cheap relative: for a low-dimensional
input, ``f(x) in f(c) + J(box) (x - c)`` where ``J(box)`` is an interval enclosure of the
Jacobian over the whole box, computed by forward-mode interval automatic differentiation.

Soundness argument, piece by piece (and where it is not rigorous):

* **Polynomial range.** Over ``eps in [-1, 1]^n`` the range of the polynomial part is
  enclosed per element by ``c + sum_i range_i(L_i, Qd_ii) + sum_{i<j} [-|Q_ij|, |Q_ij|]``,
  where ``range_i`` is the exact range of the one-variable quadratic ``L_i e + Qd_ii e^2``
  on ``[-1, 1]`` (interior critical point included). A sum of independent per-variable
  ranges contains the joint range, so this is an enclosure.
* **Product** of two models: constant, linear and quadratic terms among the ``quad_vars``
  largest-radius symbols are kept exactly. Everything else -- products involving a symbol
  without a quadratic slot, and every term of degree >= 3 -- is bounded in magnitude by
  products of absolute coefficient sums (each ``|eps_i| <= 1``) and added to the remainder.
  Cross terms with the other factor's remainder use the interval product of the factor's
  polynomial range and remainder.
* **Smooth functions.** For ``x = c + D`` with ``D = x - c`` of range ``[dlo, dhi]``,
  ``f(c + D) = f(c) + f'(c) D + ... + f^(k)(c)/k! D^k + R`` with the Lagrange remainder
  ``|R| <= sup |f^(k+1)(xi)| / (k+1)! * max|D|^(k+1)`` for some ``xi`` between ``c`` and
  ``c + D``. The supremum is taken over the interval hull of ``{c} u [c+dlo, c+dhi]`` and
  is *exact* for these functions: every ``f^(m)`` used is either monotone in magnitude on
  its domain (Exp, Log, Sqrt, Reciprocal, Rsqrt) or has the explicit critical points
  listed in ``_FUNCS`` (Sigmoid, Tanh, Erf), so the supremum is the maximum over the two
  endpoints and the critical points inside the hull. ``tests/test_taylor.py`` checks each
  derivative formula and critical point against a dense grid.
* **Relu** is not smooth: ``relu(x) = lam x + rem`` with ``rem in [0, -lam l]`` and
  ``lam = u / (u - l)`` on an unstable neuron (the DeepZ parallelogram); exact when stable.
* **Interval box.** Every tensor also carries an independent interval enclosure maintained
  by plain interval arithmetic. The reported bounds are the intersection of the two
  enclosures, so a Taylor result is never looser than ``onnxsim.interval`` and keeps facts a
  polynomial cannot see (``exp(x) > 0``, ``sigmoid(x) in (0, 1)``), which Softmax's
  ``1 / sum`` needs.
* **Unsupported ops** are enclosed by an interval box from ``onnxsim.interval`` (a fresh
  remainder per element: sound, correlation through that op is lost, and a
  ``precision lost at <op>`` note is recorded). If even that is unbounded the tensor is
  *unbounded* and so is everything computed from it -- never a wrong bound, never a NaN.
* **Not rigorous:** arithmetic is float64, not directed rounding. After every operation the
  remainder is padded by ``8 eps_64`` times the magnitude of the terms, and the final
  bounds are widened by a relative ``1e-9``; that dwarfs float64 rounding at these sizes,
  but it is an allowance, not a proof. The bounds describe the *real-number* function;
  float32 execution can exceed them by float32 rounding (about 1e-6 relative per op).

Limits, stated plainly:

* **Curse of dimensionality.** Dense storage is ``n_symbols x tensor size`` for the linear
  part and ``quad_vars (quad_vars + 1) / 2 x tensor size`` for the quadratic part. Only the
  ``max_vars`` largest-radius input elements get a symbol (the rest enter as interval
  remainders, which is sound but loses their correlation); only the ``quad_vars``
  largest-radius symbols get quadratic terms. This targets small and medium windows, not
  full-size networks.
* **Degree.** ``degree`` is 1 or 2. Degree 1 is a zonotope with a Lagrange remainder;
  degree 2 adds the quadratic part. Higher degrees are not implemented.
* **Wide boxes.** The remainder grows like ``|D|^(degree+1)``. On a wide box (say
  ``exp`` over a width-10 input) it is huge and the interval box wins; the model never does
  worse than the interval, but it is only *better* on boxes where the curvature matters
  and the box is small enough for the Taylor expansion to converge.
"""

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper

from . import interval as _interval
from . import ranges as _ranges

DEFAULT_DEGREE = 2
DEFAULT_MAX_VARS = 64
DEFAULT_QUAD_VARS = 12
_EPS64 = float(np.finfo(np.float64).eps)
_PAD = 8.0 * _EPS64
_REL_SLACK = 1e-9


def _isc(x: Any) -> bool:
    """Is ``x`` a plain constant (NumPy array or scalar), as opposed to an abstract value?"""
    return isinstance(x, (np.ndarray, np.generic))


# --------------------------------------------------------------------------
# Scalar functions: value, derivatives, critical points
# --------------------------------------------------------------------------


def _erf(x: Any) -> Any:
    x = np.asarray(x, dtype=np.float64)
    return (
        np.vectorize(math.erf, otypes=[np.float64])(x)
        if x.ndim
        else np.float64(math.erf(float(x)))
    )


def _sigmoid(x: Any) -> Any:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


@dataclasses.dataclass
class _Fn:
    """A smooth scalar function with what a rigorous Lagrange remainder needs."""

    f: Callable[[Any], Any]
    d: Dict[int, Callable[[Any], Any]]  # k -> f^(k), k = 1..3
    crit: Dict[int, Tuple[float, ...]]  # k -> x where f^(k) has a local extremum
    domain: str = "all"  # "all" | "pos" | "nonzero"


def _logit(s: float) -> float:
    return math.log(s / (1.0 - s))


def _make_funcs() -> Dict[str, _Fn]:
    c2 = 2.0 / math.sqrt(math.pi)

    def sig1(x: Any) -> Any:
        s = _sigmoid(x)
        return s * (1.0 - s)

    def sig2(x: Any) -> Any:
        s = _sigmoid(x)
        return s * (1.0 - s) * (1.0 - 2.0 * s)

    def sig3(x: Any) -> Any:
        t = sig1(x)
        return t * (1.0 - 6.0 * t)

    def tanh1(x: Any) -> Any:
        u = np.tanh(x)
        return 1.0 - u * u

    def tanh2(x: Any) -> Any:
        u = np.tanh(x)
        return -2.0 * u * (1.0 - u * u)

    def tanh3(x: Any) -> Any:
        u = np.tanh(x)
        return -2.0 * (1.0 - u * u) * (1.0 - 3.0 * u * u)

    def erf1(x: Any) -> Any:
        x = np.asarray(x, dtype=np.float64)
        return c2 * np.exp(-x * x)

    def erf2(x: Any) -> Any:
        x = np.asarray(x, dtype=np.float64)
        return -2.0 * x * erf1(x)

    def erf3(x: Any) -> Any:
        x = np.asarray(x, dtype=np.float64)
        return (4.0 * x * x - 2.0) * erf1(x)

    # Critical points: x where f^(k) has a local extremum, i.e. f^(k+1)(x) = 0.
    s_a = 0.5 + math.sqrt(3.0) / 6.0  # sigmoid'' extrema: sigmoid''' = 0
    s_b = 0.5 + math.sqrt(1.0 / 6.0)  # sigmoid''' extrema (other than 0): t = 1/12
    t_a = math.atanh(1.0 / math.sqrt(3.0))  # tanh'' extrema: u^2 = 1/3
    t_b = math.atanh(math.sqrt(2.0 / 3.0))  # tanh''' extrema (other than 0): u^2 = 2/3
    return {
        "exp": _Fn(np.exp, {1: np.exp, 2: np.exp, 3: np.exp}, {}),
        "log": _Fn(
            np.log,
            {
                1: lambda x: 1.0 / x,
                2: lambda x: -1.0 / (x * x),
                3: lambda x: 2.0 / (x * x * x),
            },
            {},
            "pos",
        ),
        "sqrt": _Fn(
            np.sqrt,
            {
                1: lambda x: 0.5 * x**-0.5,
                2: lambda x: -0.25 * x**-1.5,
                3: lambda x: 0.375 * x**-2.5,
            },
            {},
            "pos",
        ),
        "recip": _Fn(
            lambda x: 1.0 / x,
            {
                1: lambda x: -1.0 / (x * x),
                2: lambda x: 2.0 / x**3,
                3: lambda x: -6.0 / x**4,
            },
            {},
            "nonzero",
        ),
        "rsqrt": _Fn(
            lambda x: x**-0.5,
            {
                1: lambda x: -0.5 * x**-1.5,
                2: lambda x: 0.75 * x**-2.5,
                3: lambda x: -1.875 * x**-3.5,
            },
            {},
            "pos",
        ),
        "sigmoid": _Fn(
            _sigmoid,
            {1: sig1, 2: sig2, 3: sig3},
            {
                1: (0.0,),
                2: (-_logit(s_a), _logit(s_a)),
                3: (0.0, -_logit(s_b), _logit(s_b)),
            },
        ),
        "tanh": _Fn(
            np.tanh,
            {1: tanh1, 2: tanh2, 3: tanh3},
            {1: (0.0,), 2: (-t_a, t_a), 3: (0.0, -t_b, t_b)},
        ),
        "erf": _Fn(
            _erf,
            {1: erf1, 2: erf2, 3: erf3},
            {
                1: (0.0,),
                2: (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
                3: (0.0, -math.sqrt(1.5), math.sqrt(1.5)),
            },
        ),
    }


_FUNCS = _make_funcs()


def _deriv(fn: _Fn, k: int, x: Any) -> Any:
    return fn.f(x) if k == 0 else fn.d[k](x)


def _fn_range(fn: _Fn, k: int, lo: Any, hi: Any) -> Tuple[Any, Any]:
    """Elementwise range of ``f^(k)`` over ``[lo, hi]`` (exact: endpoints + critical points)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    with np.errstate(all="ignore"):
        a, b = _deriv(fn, k, lo), _deriv(fn, k, hi)
        low, high = np.minimum(a, b), np.maximum(a, b)
        for x0 in fn.crit.get(k, ()):
            inside = (lo <= x0) & (x0 <= hi)
            if np.any(inside):
                v = _deriv(fn, k, np.float64(x0))
                low = np.where(inside, np.minimum(low, v), low)
                high = np.where(inside, np.maximum(high, v), high)
    return low, high


def _in_domain(fn: _Fn, lo: Any, hi: Any) -> Any:
    if fn.domain == "all":
        return np.ones(np.shape(lo), dtype=bool)
    if fn.domain == "pos":
        return lo > 0
    return (lo > 0) | (hi < 0)


def _taylor_remainder(
    fn: _Fn, order: int, c: Any, lo: Any, hi: Any, f0: Any, f1: Any, f2: Any
) -> Tuple[Any, Any]:
    """Interval ``[rlo, rhi]`` enclosing ``R(D) = f(c + D) - T_order(D)`` for ``c + D in [lo, hi]``.

    Two exact-in-the-limit cases, chosen per element:

    * ``f^(k)`` (``k = order + 1``) has a definite sign on the hull of ``{c} u [lo, hi]``.
      Then ``R^(k) = f^(k)(c + D)`` has that sign and ``R(0) = R'(0) = ... = 0``, so ``R`` is
      monotone (``order = 2``) or unimodal with its extremum at ``0`` (``order = 1``): its
      range is the hull of ``R`` at the two endpoints and at ``0`` when ``0`` is inside. The
      endpoint values are evaluated directly (and padded for float64 cancellation).
    * Otherwise a one-sided Lagrange bound: for ``D >= 0`` the remainder is
      ``f^(k)(xi) D^k / k!`` with ``xi in [c, c + dhi]``, and for ``D <= 0`` with
      ``xi in [c + dlo, c]``; each side uses its own range of ``f^(k)`` and the sign of
      ``D^k`` there. The union of the two sides is returned.
    """
    k = order + 1
    dlo, dhi = lo - c, hi - c
    hl, hh = np.minimum(c, lo), np.maximum(c, hi)
    glo, ghi = _fn_range(fn, k, hl, hh)
    definite = (glo >= 0) | (ghi <= 0)

    def r_at(d: Any) -> Tuple[Any, Any]:
        t = f1 * d + (f2 / 2.0 * d * d if order >= 2 else 0.0)
        val = fn.f(c + d) - f0 - t
        return val, _PAD * (np.abs(fn.f(c + d)) + np.abs(f0) + np.abs(t))

    ra, pa = r_at(dlo)
    rb, pb = r_at(dhi)
    zero_in = (dlo <= 0) & (dhi >= 0)
    ex_lo = np.minimum(np.minimum(ra, rb), np.where(zero_in, 0.0, np.inf)) - np.maximum(
        pa, pb
    )
    ex_hi = np.maximum(
        np.maximum(ra, rb), np.where(zero_in, 0.0, -np.inf)
    ) + np.maximum(pa, pb)

    fac = 1.0 / math.factorial(k)
    dp, dm = np.maximum(dhi, 0.0), np.minimum(dlo, 0.0)
    gpl, gph = _fn_range(fn, k, c, c + dp)
    gml, gmh = _fn_range(fn, k, c + dm, c)
    rl_r, rh_r = _imul(gpl * fac, gph * fac, 0.0, dp**k)
    dk = np.stack([np.zeros_like(dm), dm**k])
    rl_l, rh_l = _imul(gml * fac, gmh * fac, dk.min(axis=0), dk.max(axis=0))
    lg_lo = np.minimum(rl_r, rl_l)
    lg_hi = np.maximum(rh_r, rh_l)
    return np.where(definite, ex_lo, lg_lo), np.where(definite, ex_hi, lg_hi)


# --------------------------------------------------------------------------
# Plain interval helpers (the independent "box" enclosure, and the Dual values)
# --------------------------------------------------------------------------


def _imul(al: Any, ah: Any, bl: Any, bh: Any) -> Tuple[Any, Any]:
    with np.errstate(all="ignore"):
        p = np.stack(np.broadcast_arrays(al * bl, al * bh, ah * bl, ah * bh))
    p = np.where(np.isnan(p), 0.0, p)
    return p.min(axis=0), p.max(axis=0)


def _imatmul(al: Any, ah: Any, bl: Any, bh: Any) -> Tuple[Any, Any]:
    ca, ra = (al + ah) / 2.0, (ah - al) / 2.0
    cb, rb = (bl + bh) / 2.0, (bh - bl) / 2.0
    c = np.matmul(ca, cb)
    r = np.matmul(np.abs(ca), rb) + np.matmul(ra, np.abs(cb)) + np.matmul(ra, rb)
    pad = _PAD * (np.matmul(np.abs(ca) + ra, np.abs(cb) + rb))
    return c - r - pad, c + r + pad


def _widen(lo: Any, hi: Any, rel: float = _REL_SLACK) -> Tuple[Any, Any]:
    with np.errstate(invalid="ignore"):
        return (
            np.where(np.isfinite(lo), lo - rel * (1.0 + np.abs(lo)), lo),
            np.where(np.isfinite(hi), hi + rel * (1.0 + np.abs(hi)), hi),
        )


# --------------------------------------------------------------------------
# Taylor models
# --------------------------------------------------------------------------


class _Ctx:
    """Shared symbol layout: ``n`` noise symbols, the first ``pq`` have quadratic slots."""

    def __init__(self, n: int, pq: int) -> None:
        self.n = n
        self.pq = min(pq, n)
        pairs = [(i, j) for i in range(self.pq) for j in range(i, self.pq)]
        self.PI = np.array([p[0] for p in pairs], dtype=np.int64)
        self.PJ = np.array([p[1] for p in pairs], dtype=np.int64)
        self.npairs = len(pairs)
        self.diag = np.flatnonzero(self.PI == self.PJ)
        self.off = np.flatnonzero(self.PI != self.PJ)


def _resolve_shape(shape: Sequence[int], size: int) -> Tuple[int, ...]:
    """Replace a ``-1`` in ``shape`` using ``size`` (NumPy cannot infer it for empty arrays)."""
    out = [int(d) for d in shape]
    if -1 in out:
        known = int(np.prod([d for d in out if d != -1], dtype=np.int64))
        out[out.index(-1)] = size // known if known else 0
    return tuple(out)


def _bc_arr(a: Any, shape: Tuple[int, ...], lead: int) -> Any:
    """Broadcast ``a`` (with ``lead`` leading non-tensor axes) to ``shape``."""
    if tuple(a.shape[lead:]) == tuple(shape):
        return a
    k = len(shape) - (a.ndim - lead)
    a = a.reshape(a.shape[:lead] + (1,) * k + a.shape[lead:])
    return np.broadcast_to(a, a.shape[:lead] + tuple(shape))


def _quad_range(a: Any, q: Any) -> Tuple[Any, Any]:
    """Exact range of ``a e + q e^2`` over ``e in [-1, 1]``, elementwise."""
    with np.errstate(divide="ignore", invalid="ignore"):
        e0 = np.where(q != 0, -a / (2.0 * q), 2.0)
    vc = np.where(np.abs(e0) <= 1.0, a * e0 + q * e0 * e0, np.nan)
    ends = np.stack([-a + q, a + q])
    lo = np.fmin(ends.min(axis=0), vc)
    hi = np.fmax(ends.max(axis=0), vc)
    return lo, hi


class TaylorModel:
    """A tensor of Taylor models (see the module docstring)."""

    __slots__ = ("c", "L", "Q", "rlo", "rhi", "blo", "bhi", "ctx")

    def __init__(
        self, c: Any, L: Any, Q: Any, rlo: Any, rhi: Any, blo: Any, bhi: Any, ctx: _Ctx
    ) -> None:
        self.c, self.L, self.Q = c, L, Q
        self.rlo, self.rhi, self.blo, self.bhi = rlo, rhi, blo, bhi
        self.ctx = ctx

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.c.shape)

    @classmethod
    def constant(cls, arr: Any, ctx: _Ctx) -> "TaylorModel":
        a = np.asarray(arr, dtype=np.float64)
        z = np.zeros_like(a)
        return cls(
            a,
            np.zeros((ctx.n,) + a.shape),
            np.zeros((ctx.npairs,) + a.shape),
            z,
            z.copy(),
            a,
            a.copy(),
            ctx,
        )

    @classmethod
    def from_box(cls, lo: Any, hi: Any, ctx: _Ctx) -> "TaylorModel":
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        c = (lo + hi) / 2.0
        return cls(
            c,
            np.zeros((ctx.n,) + c.shape),
            np.zeros((ctx.npairs,) + c.shape),
            lo - c,
            hi - c,
            lo,
            hi,
            ctx,
        )

    def poly_range(self, zero_const: bool = False) -> Tuple[Any, Any]:
        """Enclosure of the polynomial part over ``eps in [-1, 1]^n``."""
        ctx = self.ctx
        base = np.zeros_like(self.c) if zero_const else self.c
        lo, hi = base.copy(), base.copy()
        pq = ctx.pq
        if pq > 0:
            ql, qh = _quad_range(self.L[:pq], self.Q[ctx.diag])
            lo = lo + ql.sum(axis=0)
            hi = hi + qh.sum(axis=0)
        if ctx.n > pq:
            tail = np.abs(self.L[pq:]).sum(axis=0)
            lo = lo - tail
            hi = hi + tail
        if len(ctx.off):
            off = np.abs(self.Q[ctx.off]).sum(axis=0)
            lo = lo - off
            hi = hi + off
        return lo, hi

    def range(self) -> Tuple[Any, Any]:
        """Enclosure of the value: polynomial + remainder, intersected with the interval box."""
        plo, phi = self.poly_range()
        lo = np.maximum(plo + self.rlo, self.blo)
        hi = np.minimum(phi + self.rhi, self.bhi)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def bounds(self, rel_slack: float = _REL_SLACK) -> Tuple[Any, Any]:
        """Elementwise ``(lo, hi)`` enclosing the value for every input in the box."""
        lo, hi = self.range()
        return _widen(lo, hi, rel_slack)

    def remainder(self) -> Tuple[Any, Any]:
        return self.rlo, self.rhi

    def evaluate(self, eps: Any) -> Any:
        """The polynomial part at noise values ``eps`` (shape ``(n,)``, each in ``[-1, 1]``).

        For every ``eps`` in the box the true value lies in
        ``evaluate(eps) + [rlo, rhi]`` -- the property the tests check per function.
        """
        eps = np.asarray(eps, dtype=np.float64)
        ctx = self.ctx
        out = self.c + np.tensordot(eps, self.L, axes=(0, 0))
        if ctx.npairs:
            out = out + np.tensordot(eps[ctx.PI] * eps[ctx.PJ], self.Q, axes=(0, 0))
        return out

    def is_exact(self) -> bool:
        return bool(
            not np.any(self.L)
            and not np.any(self.Q)
            and not np.any(self.rlo)
            and not np.any(self.rhi)
        )


def _pad(tm: TaylorModel) -> TaylorModel:
    """Allow for float64 rounding of the operation that produced ``tm``."""
    mag = np.abs(tm.c) + np.abs(tm.L).sum(axis=0) + np.abs(tm.Q).sum(axis=0)
    p = _PAD * mag
    tm.rlo = tm.rlo - p
    tm.rhi = tm.rhi + p
    return tm


def _bc(tm: TaylorModel, shape: Tuple[int, ...]) -> TaylorModel:
    if tm.shape == tuple(shape):
        return tm
    return TaylorModel(
        _bc_arr(tm.c, shape, 0),
        _bc_arr(tm.L, shape, 1),
        _bc_arr(tm.Q, shape, 1),
        _bc_arr(tm.rlo, shape, 0),
        _bc_arr(tm.rhi, shape, 0),
        _bc_arr(tm.blo, shape, 0),
        _bc_arr(tm.bhi, shape, 0),
        tm.ctx,
    )


def _tm_add(a: TaylorModel, b: TaylorModel, sign: float = 1.0) -> TaylorModel:
    shape = tuple(np.broadcast_shapes(a.shape, b.shape))
    a, b = _bc(a, shape), _bc(b, shape)
    if sign >= 0:
        rlo, rhi = a.rlo + b.rlo, a.rhi + b.rhi
        blo, bhi = a.blo + b.blo, a.bhi + b.bhi
    else:
        rlo, rhi = a.rlo - b.rhi, a.rhi - b.rlo
        blo, bhi = a.blo - b.bhi, a.bhi - b.blo
    return _pad(
        TaylorModel(
            a.c + sign * b.c,
            a.L + sign * b.L,
            a.Q + sign * b.Q,
            rlo,
            rhi,
            blo,
            bhi,
            a.ctx,
        )
    )


def _tm_scale(a: TaylorModel, k: Any) -> TaylorModel:
    k = np.asarray(k, dtype=np.float64)
    shape = tuple(np.broadcast_shapes(a.shape, k.shape))
    a = _bc(a, shape)
    k = np.broadcast_to(k, shape)
    r = np.stack([k * a.rlo, k * a.rhi])
    b = np.stack([k * a.blo, k * a.bhi])
    return _pad(
        TaylorModel(
            a.c * k,
            a.L * k,
            a.Q * k,
            r.min(axis=0),
            r.max(axis=0),
            b.min(axis=0),
            b.max(axis=0),
            a.ctx,
        )
    )


def _tm_add_const(a: TaylorModel, k: Any) -> TaylorModel:
    k = np.asarray(k, dtype=np.float64)
    shape = tuple(np.broadcast_shapes(a.shape, k.shape))
    a = _bc(a, shape)
    return _pad(
        TaylorModel(a.c + k, a.L, a.Q, a.rlo, a.rhi, a.blo + k, a.bhi + k, a.ctx)
    )


def _tm_mul(a: TaylorModel, b: TaylorModel) -> TaylorModel:
    ctx = a.ctx
    shape = tuple(np.broadcast_shapes(a.shape, b.shape))
    a, b = _bc(a, shape), _bc(b, shape)
    pq = ctx.pq
    c = a.c * b.c
    L = a.c * b.L + b.c * a.L
    Q = a.c * b.Q + b.c * a.Q
    if pq > 0:
        la, lb = a.L[:pq], b.L[:pq]
        cross = la[ctx.PI] * lb[ctx.PJ]
        offmask = (ctx.PI != ctx.PJ).reshape((-1,) + (1,) * len(shape))
        Q = Q + cross + np.where(offmask, la[ctx.PJ] * lb[ctx.PI], 0.0)
    # Everything not kept: |eps| <= 1 bounds every monomial by 1.
    s_al, s_bl = np.abs(a.L).sum(axis=0), np.abs(b.L).sum(axis=0)
    s_aq_l, s_bq_l = np.abs(a.L[:pq]).sum(axis=0), np.abs(b.L[:pq]).sum(axis=0)
    s_aQ, s_bQ = np.abs(a.Q).sum(axis=0), np.abs(b.Q).sum(axis=0)
    trunc = (
        np.maximum(s_al * s_bl - s_aq_l * s_bq_l, 0.0)
        + s_al * s_bQ
        + s_aQ * s_bl
        + s_aQ * s_bQ
    )
    pa_lo, pa_hi = a.poly_range()
    pb_lo, pb_hi = b.poly_range()
    r1l, r1h = _imul(pa_lo, pa_hi, b.rlo, b.rhi)
    r2l, r2h = _imul(pb_lo, pb_hi, a.rlo, a.rhi)
    r3l, r3h = _imul(a.rlo, a.rhi, b.rlo, b.rhi)
    blo, bhi = _imul(a.blo, a.bhi, b.blo, b.bhi)
    return _pad(
        TaylorModel(
            c, L, Q, r1l + r2l + r3l - trunc, r1h + r2h + r3h + trunc, blo, bhi, ctx
        )
    )


# --------------------------------------------------------------------------
# Algebras: the same op code runs on numbers, Taylor models and interval-AD duals
# --------------------------------------------------------------------------


class _Unsupported(Exception):
    """This op/shape has no rule here; the caller falls back to an interval box."""


class _Top:
    """An unbounded tensor (shape may be unknown). Anything computed from it is Top."""

    def __init__(self, shape: Optional[Tuple[int, ...]] = None) -> None:
        self.shape = shape


class _Lin:
    """A linear map with constant weights, and its positive/negative parts.

    ``pos``/``neg`` are the same op with weights ``max(W, 0)`` / ``min(W, 0)``; they give the
    exact image of an interval: ``[pos(lo) + neg(hi), pos(hi) + neg(lo)]``.
    """

    def __init__(
        self, fwd: Callable, pos: Callable, neg: Callable, bias: Any = None
    ) -> None:
        self.fwd, self.pos, self.neg, self.bias = fwd, pos, neg, bias

    @staticmethod
    def matmul_right(w: Any) -> "_Lin":
        wp, wn = np.maximum(w, 0.0), np.minimum(w, 0.0)
        return _Lin(
            lambda t: np.matmul(t, w),
            lambda t: np.matmul(t, wp),
            lambda t: np.matmul(t, wn),
        )

    @staticmethod
    def matmul_left(w: Any) -> "_Lin":
        wp, wn = np.maximum(w, 0.0), np.minimum(w, 0.0)
        return _Lin(
            lambda t: np.matmul(w, t),
            lambda t: np.matmul(wp, t),
            lambda t: np.matmul(wn, t),
        )

    def image(self, lo: Any, hi: Any) -> Tuple[Any, Any]:
        out_lo = self.pos(lo) + self.neg(hi)
        out_hi = self.pos(hi) + self.neg(lo)
        if self.bias is not None:
            out_lo, out_hi = out_lo + self.bias, out_hi + self.bias
        return out_lo, out_hi

    def batched(self, fn: Callable, arr: Any, per_slice: bool) -> Any:
        """Apply ``fn`` to every leading slice of ``arr`` (shape ``(k,) + tensor``)."""
        if not per_slice:
            return fn(arr)
        if arr.shape[0] == 0:
            # No slices (no noise symbols, or no quadratic slots): the result must still have
            # the *output* tensor shape, which only the op knows -- ask it with a zero slice.
            return np.zeros((0,) + np.shape(fn(np.zeros(arr.shape[1:]))))
        return np.stack([fn(arr[i]) for i in range(arr.shape[0])])


def _conv2d(
    x: Any,
    w: Any,
    strides: Sequence[int],
    pads: Sequence[int],
    dil: Sequence[int],
    group: int,
) -> Any:
    """2-D convolution (NCHW) with stride, padding, dilation and groups."""
    n, c, h, wd = x.shape
    m, cg, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    eh, ew = dil[0] * (kh - 1) + 1, dil[1] * (kw - 1) + 1
    win = np.lib.stride_tricks.sliding_window_view(xp, (eh, ew), axis=(2, 3))
    win = win[:, :, :: strides[0], :: strides[1], :: dil[0], :: dil[1]]
    oh, ow = win.shape[2], win.shape[3]
    out = np.empty((n, m, oh, ow), dtype=np.float64)
    mg = m // group
    for g in range(group):
        out[:, g * mg : (g + 1) * mg] = np.einsum(
            "bchwij,mcij->bmhw", win[:, g * cg : (g + 1) * cg], w[g * mg : (g + 1) * mg]
        )
    return out


class _Alg:
    """Base class: shared glue, with const-only paths handled by the subclass."""

    def matmul(self, a: Any, b: Any) -> Any:
        ca, cb = _isc(a), _isc(b)
        if ca and cb:
            return np.matmul(a, b)
        if cb:
            return self.lin(a, _Lin.matmul_right(np.asarray(b, dtype=np.float64)), True)
        if ca:
            if np.ndim(self.shape(b)) and len(self.shape(b)) < 2:
                raise _Unsupported(
                    "MatMul with a constant left operand and a 1-D right operand"
                )
            return self.lin(b, _Lin.matmul_left(np.asarray(a, dtype=np.float64)), True)
        return self.matmul_general(a, b)

    def sub(self, a: Any, b: Any) -> Any:
        return self.add(a, self.neg(b))

    def square(self, a: Any) -> Any:
        return self.mul(a, a)  # overridden where the sign information matters

    # The interface every algebra implements (the evaluator is written against it once).
    def shape(self, a: Any) -> Tuple[int, ...]:
        raise NotImplementedError

    def add(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def neg(self, a: Any) -> Any:
        raise NotImplementedError

    def mul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def func(self, name: str, x: Any) -> Any:
        raise NotImplementedError

    def relu(self, x: Any) -> Any:
        raise NotImplementedError

    def lin(self, x: Any, lin: "_Lin", per_slice: bool) -> Any:
        raise NotImplementedError

    def matmul_general(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def sum(self, x: Any, axes: Sequence[int], keepdims: bool) -> Any:
        raise NotImplementedError

    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        raise NotImplementedError

    def transpose(self, x: Any, perm: Optional[Sequence[int]]) -> Any:
        raise NotImplementedError

    def concat(self, xs: List[Any], axis: int) -> Any:
        raise NotImplementedError

    def bounds(self, x: Any) -> Tuple[Any, Any]:
        raise NotImplementedError

    def from_box(self, lo: Any, hi: Any) -> Any:
        raise NotImplementedError

    def lift(self, x: Any) -> Any:
        raise NotImplementedError


class _NumAlg(_Alg):
    """Plain float64 arrays (used when every operand of a node is a constant)."""

    def shape(self, a: Any) -> Tuple[int, ...]:
        return tuple(np.shape(a))

    def add(self, a: Any, b: Any) -> Any:
        return np.asarray(a) + np.asarray(b)

    def neg(self, a: Any) -> Any:
        return -np.asarray(a)

    def mul(self, a: Any, b: Any) -> Any:
        return np.asarray(a) * np.asarray(b)

    def func(self, name: str, x: Any) -> Any:
        with np.errstate(all="ignore"):
            out = _FUNCS[name].f(np.asarray(x, dtype=np.float64))
        if not np.all(np.isfinite(out)):
            raise _Unsupported(f"{name} is not finite on its constant argument")
        return out

    def relu(self, x: Any) -> Any:
        return np.maximum(x, 0.0)

    def lin(self, x: Any, lin: _Lin, per_slice: bool) -> Any:
        out = lin.fwd(np.asarray(x, dtype=np.float64))
        return out if lin.bias is None else out + lin.bias

    def matmul_general(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def sum(self, x: Any, axes: Sequence[int], keepdims: bool) -> Any:
        return np.sum(x, axis=tuple(axes), keepdims=keepdims)

    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        return np.reshape(x, shape)

    def transpose(self, x: Any, perm: Optional[Sequence[int]]) -> Any:
        return np.transpose(x, perm)

    def concat(self, xs: List[Any], axis: int) -> Any:
        return np.concatenate(xs, axis=axis)

    def bounds(self, x: Any) -> Tuple[Any, Any]:
        return np.asarray(x), np.asarray(x)

    def from_box(self, lo: Any, hi: Any) -> Any:
        raise _Unsupported("no interval fallback for constants")

    def lift(self, x: Any) -> Any:
        return np.asarray(x, dtype=np.float64)


class _TMAlg(_Alg):
    """Taylor models over a shared symbol context."""

    def __init__(self, ctx: _Ctx, degree: int) -> None:
        self.ctx = ctx
        self.degree = degree

    # -- helpers
    def lift(self, x: Any) -> TaylorModel:
        return x if isinstance(x, TaylorModel) else TaylorModel.constant(x, self.ctx)

    def shape(self, a: Any) -> Tuple[int, ...]:
        return a.shape if isinstance(a, TaylorModel) else tuple(np.shape(a))

    def bounds(self, x: Any) -> Tuple[Any, Any]:
        if isinstance(x, TaylorModel):
            return x.range()
        return np.asarray(x), np.asarray(x)

    def from_box(self, lo: Any, hi: Any) -> TaylorModel:
        return TaylorModel.from_box(lo, hi, self.ctx)

    # -- arithmetic
    def add(self, a: Any, b: Any) -> Any:
        if _isc(a) and _isc(b):
            return a + b
        if _isc(b):
            return _tm_add_const(a, b)
        if _isc(a):
            return _tm_add_const(b, a)
        return _tm_add(a, b)

    def neg(self, a: Any) -> Any:
        return -a if _isc(a) else _tm_scale(a, -1.0)

    def mul(self, a: Any, b: Any) -> Any:
        if _isc(a) and _isc(b):
            return a * b
        if _isc(b):
            return _tm_scale(a, b)
        if _isc(a):
            return _tm_scale(b, a)
        return _tm_mul(a, b)

    def square(self, a: Any) -> Any:
        if _isc(a):
            return a * a
        res = _tm_mul(a, a)
        lo, hi = a.range()
        sl, sh = lo * lo, hi * hi
        res.blo = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(sl, sh))
        res.bhi = np.maximum(sl, sh)
        return res

    def relu(self, x: Any) -> Any:
        if _isc(x):
            return np.maximum(x, 0.0)
        lo, hi = x.range()
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise _Unsupported("Relu of an unbounded tensor")
        unstable = (lo < 0) & (hi > 0)
        with np.errstate(all="ignore"):
            lam = np.where(unstable, hi / (hi - lo), np.where(lo >= 0, 1.0, 0.0))
        rem_hi = np.where(unstable, -lam * lo, 0.0)
        res = _tm_scale(x, lam)
        res.rhi = res.rhi + rem_hi
        res.blo, res.bhi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        return res

    def func(self, name: str, x: Any) -> Any:
        fn = _FUNCS[name]
        if _isc(x):
            return NUM.func(name, x)
        lo, hi = x.range()
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise _Unsupported(f"{name} of an unbounded tensor")
        c = x.c
        hl, hh = np.minimum(c, lo), np.maximum(c, hi)
        if not np.all(_in_domain(fn, hl, hh)):
            if not np.all(_in_domain(fn, lo, hi)):
                raise _Unsupported(f"{name} outside its domain on part of the box")
            flo, fhi = _fn_range(fn, 0, lo, hi)
            return self.from_box(flo, fhi)
        order = self.degree
        f0, f1 = fn.f(c), fn.d[1](c)
        f2 = fn.d[2](c)
        rl, rh = _taylor_remainder(fn, order, c, lo, hi, f0, f1, f2)
        d = TaylorModel(np.zeros_like(c), x.L, x.Q, x.rlo, x.rhi, lo - c, hi - c, x.ctx)
        res = _tm_add_const(_tm_scale(d, f1), f0)
        if order >= 2:
            res = _tm_add(res, _tm_scale(_tm_mul(d, d), f2 / 2.0))
        res.rlo = res.rlo + rl
        res.rhi = res.rhi + rh
        res.blo, res.bhi = _fn_range(fn, 0, lo, hi)
        return _pad(res)

    # -- structure
    def lin(self, x: Any, lin: _Lin, per_slice: bool) -> Any:
        if _isc(x):
            return NUM.lin(x, lin, per_slice)
        c = lin.fwd(x.c)
        if lin.bias is not None:
            c = c + lin.bias
        L = lin.batched(lin.fwd, x.L, per_slice)
        Q = lin.batched(lin.fwd, x.Q, per_slice)
        rlo = lin.pos(x.rlo) + lin.neg(x.rhi)
        rhi = lin.pos(x.rhi) + lin.neg(x.rlo)
        blo, bhi = lin.image(x.blo, x.bhi)
        return _pad(TaylorModel(c, L, Q, rlo, rhi, blo, bhi, x.ctx))

    def matmul_general(self, a: Any, b: Any) -> Any:
        sa, sb = self.shape(a), self.shape(b)
        if len(sa) < 2 or len(sb) < 2:
            raise _Unsupported("MatMul of tensors with a rank below 2")
        ea = self.reshape(a, sa + (1,))
        eb = self.reshape(b, sb[:-2] + (1,) + sb[-2:])
        prod = self.mul(ea, eb)
        return self.sum(prod, [len(self.shape(prod)) - 2], False)

    def sum(self, x: Any, axes: Sequence[int], keepdims: bool) -> Any:
        if _isc(x):
            return np.sum(x, axis=tuple(axes), keepdims=keepdims)
        nd = len(x.shape)
        ax = tuple(a % nd for a in axes)
        ax1 = tuple(a + 1 for a in ax)
        return _pad(
            TaylorModel(
                x.c.sum(axis=ax, keepdims=keepdims),
                x.L.sum(axis=ax1, keepdims=keepdims),
                x.Q.sum(axis=ax1, keepdims=keepdims),
                x.rlo.sum(axis=ax, keepdims=keepdims),
                x.rhi.sum(axis=ax, keepdims=keepdims),
                x.blo.sum(axis=ax, keepdims=keepdims),
                x.bhi.sum(axis=ax, keepdims=keepdims),
                x.ctx,
            )
        )

    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        if _isc(x):
            return np.reshape(x, shape)
        shape = _resolve_shape(shape, x.c.size)
        n, p = x.ctx.n, x.ctx.npairs
        return TaylorModel(
            x.c.reshape(shape),
            x.L.reshape((n,) + shape),
            x.Q.reshape((p,) + shape),
            x.rlo.reshape(shape),
            x.rhi.reshape(shape),
            x.blo.reshape(shape),
            x.bhi.reshape(shape),
            x.ctx,
        )

    def transpose(self, x: Any, perm: Optional[Sequence[int]]) -> Any:
        if _isc(x):
            return np.transpose(x, perm)
        nd = len(x.shape)
        pm = list(perm) if perm is not None else list(range(nd))[::-1]
        pm1 = [0] + [p + 1 for p in pm]
        return TaylorModel(
            np.transpose(x.c, pm),
            np.transpose(x.L, pm1),
            np.transpose(x.Q, pm1),
            np.transpose(x.rlo, pm),
            np.transpose(x.rhi, pm),
            np.transpose(x.blo, pm),
            np.transpose(x.bhi, pm),
            x.ctx,
        )

    def concat(self, xs: List[Any], axis: int) -> Any:
        if all(_isc(x) for x in xs):
            return np.concatenate(xs, axis=axis)
        ts = [self.lift(x) for x in xs]
        nd = len(ts[0].shape)
        ax = axis % nd
        cat = np.concatenate
        return TaylorModel(
            cat([t.c for t in ts], axis=ax),
            cat([t.L for t in ts], axis=ax + 1),
            cat([t.Q for t in ts], axis=ax + 1),
            cat([t.rlo for t in ts], axis=ax),
            cat([t.rhi for t in ts], axis=ax),
            cat([t.blo for t in ts], axis=ax),
            cat([t.bhi for t in ts], axis=ax),
            ts[0].ctx,
        )


NUM = _NumAlg()


# --------------------------------------------------------------------------
# Forward-mode interval automatic differentiation (the mean-value form)
# --------------------------------------------------------------------------


class _Dual:
    """Interval value ``[vlo, vhi]`` and interval derivative ``[dlo, dhi]`` w.r.t. each input."""

    __slots__ = ("vlo", "vhi", "dlo", "dhi")

    def __init__(self, vlo: Any, vhi: Any, dlo: Any, dhi: Any) -> None:
        self.vlo, self.vhi, self.dlo, self.dhi = vlo, vhi, dlo, dhi

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.vlo.shape)


def _dbc(x: _Dual, shape: Tuple[int, ...]) -> _Dual:
    if x.shape == tuple(shape):
        return x
    return _Dual(
        _bc_arr(x.vlo, shape, 0),
        _bc_arr(x.vhi, shape, 0),
        _bc_arr(x.dlo, shape, 1),
        _bc_arr(x.dhi, shape, 1),
    )


class _DualAlg(_Alg):
    def __init__(self, n: int) -> None:
        self.n = n

    def lift(self, x: Any) -> _Dual:
        if isinstance(x, _Dual):
            return x
        a = np.asarray(x, dtype=np.float64)
        z = np.zeros((self.n,) + a.shape)
        return _Dual(a, a.copy(), z, z.copy())

    def shape(self, a: Any) -> Tuple[int, ...]:
        return a.shape if isinstance(a, _Dual) else tuple(np.shape(a))

    def bounds(self, x: Any) -> Tuple[Any, Any]:
        if isinstance(x, _Dual):
            return x.vlo, x.vhi
        return np.asarray(x), np.asarray(x)

    def from_box(self, lo: Any, hi: Any) -> Any:
        raise _Unsupported("no derivative through an unsupported op")

    def add(self, a: Any, b: Any) -> Any:
        if _isc(a) and _isc(b):
            return a + b
        a, b = self.lift(a), self.lift(b)
        shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        a, b = _dbc(a, shape), _dbc(b, shape)
        return _Dual(a.vlo + b.vlo, a.vhi + b.vhi, a.dlo + b.dlo, a.dhi + b.dhi)

    def neg(self, a: Any) -> Any:
        if _isc(a):
            return -a
        return _Dual(-a.vhi, -a.vlo, -a.dhi, -a.dlo)

    def mul(self, a: Any, b: Any) -> Any:
        if _isc(a) and _isc(b):
            return a * b
        a, b = self.lift(a), self.lift(b)
        shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        a, b = _dbc(a, shape), _dbc(b, shape)
        vlo, vhi = _imul(a.vlo, a.vhi, b.vlo, b.vhi)
        t1l, t1h = _imul(a.dlo, a.dhi, b.vlo, b.vhi)
        t2l, t2h = _imul(b.dlo, b.dhi, a.vlo, a.vhi)
        return _Dual(vlo, vhi, t1l + t2l, t1h + t2h)

    def func(self, name: str, x: Any) -> Any:
        fn = _FUNCS[name]
        if _isc(x):
            return NUM.func(name, x)
        lo, hi = x.vlo, x.vhi
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise _Unsupported(f"{name} of an unbounded tensor")
        if not np.all(_in_domain(fn, lo, hi)):
            raise _Unsupported(f"{name} outside its domain on part of the box")
        vlo, vhi = _fn_range(fn, 0, lo, hi)
        gl, gh = _fn_range(fn, 1, lo, hi)
        dl, dh = _imul(x.dlo, x.dhi, gl, gh)
        return _Dual(vlo, vhi, dl, dh)

    def square(self, a: Any) -> Any:
        if _isc(a):
            return a * a
        sl, sh = a.vlo * a.vlo, a.vhi * a.vhi
        vlo = np.where((a.vlo <= 0) & (a.vhi >= 0), 0.0, np.minimum(sl, sh))
        dl, dh = _imul(a.dlo, a.dhi, 2.0 * a.vlo, 2.0 * a.vhi)
        return _Dual(vlo, np.maximum(sl, sh), dl, dh)

    def relu(self, x: Any) -> Any:
        if _isc(x):
            return np.maximum(x, 0.0)
        lo, hi = x.vlo, x.vhi
        gl = np.where(lo >= 0, 1.0, 0.0)
        gh = np.where(hi > 0, 1.0, 0.0)
        dl, dh = _imul(x.dlo, x.dhi, gl, gh)
        return _Dual(np.maximum(lo, 0.0), np.maximum(hi, 0.0), dl, dh)

    def lin(self, x: Any, lin: _Lin, per_slice: bool) -> Any:
        if _isc(x):
            return NUM.lin(x, lin, per_slice)
        vlo, vhi = lin.image(x.vlo, x.vhi)
        dl = lin.batched(lambda t: lin.pos(t), x.dlo, per_slice) + lin.batched(
            lambda t: lin.neg(t), x.dhi, per_slice
        )
        dh = lin.batched(lambda t: lin.pos(t), x.dhi, per_slice) + lin.batched(
            lambda t: lin.neg(t), x.dlo, per_slice
        )
        return _Dual(vlo, vhi, dl, dh)

    def matmul_general(self, a: Any, b: Any) -> Any:
        vlo, vhi = _imatmul(a.vlo, a.vhi, b.vlo, b.vhi)
        t1l, t1h = _imatmul(a.dlo, a.dhi, b.vlo, b.vhi)  # (n, ..., I, J) @ (..., J, K)
        t2l, t2h = _imatmul(a.vlo, a.vhi, b.dlo, b.dhi)  # (..., I, J) @ (n, ..., J, K)
        return _Dual(vlo, vhi, t1l + t2l, t1h + t2h)

    def sum(self, x: Any, axes: Sequence[int], keepdims: bool) -> Any:
        if _isc(x):
            return np.sum(x, axis=tuple(axes), keepdims=keepdims)
        nd = len(x.shape)
        ax = tuple(a % nd for a in axes)
        ax1 = tuple(a + 1 for a in ax)
        return _Dual(
            x.vlo.sum(axis=ax, keepdims=keepdims),
            x.vhi.sum(axis=ax, keepdims=keepdims),
            x.dlo.sum(axis=ax1, keepdims=keepdims),
            x.dhi.sum(axis=ax1, keepdims=keepdims),
        )

    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        if _isc(x):
            return np.reshape(x, shape)
        shape = _resolve_shape(shape, x.vlo.size)
        n = self.n
        return _Dual(
            x.vlo.reshape(shape),
            x.vhi.reshape(shape),
            x.dlo.reshape((n,) + shape),
            x.dhi.reshape((n,) + shape),
        )

    def transpose(self, x: Any, perm: Optional[Sequence[int]]) -> Any:
        if _isc(x):
            return np.transpose(x, perm)
        nd = len(x.shape)
        pm = list(perm) if perm is not None else list(range(nd))[::-1]
        pm1 = [0] + [p + 1 for p in pm]
        return _Dual(
            np.transpose(x.vlo, pm),
            np.transpose(x.vhi, pm),
            np.transpose(x.dlo, pm1),
            np.transpose(x.dhi, pm1),
        )

    def concat(self, xs: List[Any], axis: int) -> Any:
        if all(_isc(x) for x in xs):
            return np.concatenate(xs, axis=axis)
        ds = [self.lift(x) for x in xs]
        ax = axis % len(ds[0].shape)
        cat = np.concatenate
        return _Dual(
            cat([d.vlo for d in ds], axis=ax),
            cat([d.vhi for d in ds], axis=ax),
            cat([d.dlo for d in ds], axis=ax + 1),
            cat([d.dhi for d in ds], axis=ax + 1),
        )


# --------------------------------------------------------------------------
# Graph evaluation (shared by Taylor models and the mean-value form)
# --------------------------------------------------------------------------

Value = Union[np.ndarray, TaylorModel, _Dual, _Top]


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _ref_run(node: onnx.NodeProto, arrays: List[Any], opsets: Any) -> List[Any]:
    from onnx.reference import ReferenceEvaluator

    n = onnx.NodeProto()
    n.CopyFrom(node)
    names = [f"i{k}" for k in range(len(arrays))]
    del n.input[:]
    n.input.extend(names)
    del n.output[:]
    n.output.extend(f"o{k}" for k in range(max(1, len(node.output))))
    arrays = [
        np.asarray(a, dtype=np.float64)
        if np.asarray(a).dtype.kind == "f"
        else np.asarray(a)
        for a in arrays
    ]
    g = onnx.helper.make_graph(
        [n],
        "n",
        [
            onnx.helper.make_tensor_value_info(
                nm, onnx.helper.np_dtype_to_tensor_dtype(a.dtype), a.shape
            )
            for nm, a in zip(names, arrays)
        ],
        [onnx.helper.make_empty_tensor_value_info(o) for o in n.output],
    )
    m = onnx.helper.make_model(g, opset_imports=list(opsets))
    m.ir_version = 8
    return list(ReferenceEvaluator(m).run(None, dict(zip(names, arrays))))


class _Evaluator:
    """Run an ONNX graph on abstract values; one set of op rules for every algebra."""

    def __init__(self, model: onnx.ModelProto, alg: _Alg) -> None:
        self.model = model
        self.alg = alg
        self.opsets = list(model.opset_import)
        self.opset = next(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 1
        )
        self.notes: List[str] = []
        self.consts: Dict[str, np.ndarray] = {}
        for t in model.graph.initializer:
            a = numpy_helper.to_array(t)
            self.consts[t.name] = a.astype(np.float64) if a.dtype.kind == "f" else a

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Value]:
        values: Dict[str, Value] = dict(self.consts)
        values.update(inputs)
        for node in self.model.graph.node:
            args = [values[x] if x else None for x in node.input]
            outs = self._apply(node, args)
            for name, v in zip(node.output, outs):
                if name:
                    values[name] = v
        return values

    # -- dispatch with sound fallbacks
    def _apply(self, node: onnx.NodeProto, args: List[Any]) -> List[Value]:
        present = [a for a in args if a is not None]
        n_out = max(1, len([o for o in node.output if o]))
        if any(isinstance(a, _Top) for a in present):
            return [_Top() for _ in range(n_out)]
        allconst = all(_isc(a) for a in present)
        alg: _Alg = NUM if allconst else self.alg
        fn = (
            getattr(self, "op_" + node.op_type, None)
            if node.domain in ("", "ai.onnx", "com.microsoft")
            else None
        )
        if fn is not None:
            try:
                out = fn(node, args, alg)
                return out if isinstance(out, list) else [out]
            except _Unsupported as e:
                if not allconst:
                    self.note(f"precision lost at {node.op_type}: {e}")
        elif not allconst:
            self.note(f"precision lost at {node.op_type}: no rule")
        if allconst:
            try:
                return list(_ref_run(node, args, self.opsets))
            except Exception:
                return [_Top() for _ in range(n_out)]
        return self._interval_fallback(node, args, n_out)

    def _interval_fallback(
        self, node: onnx.NodeProto, args: List[Any], n_out: int
    ) -> List[Value]:
        alg = self.alg
        try:
            ranges: Dict[str, Tuple[Any, Any]] = {}
            inits: List[onnx.TensorProto] = []
            ins: List[onnx.ValueInfoProto] = []
            names: List[str] = []
            for k, a in enumerate(args):
                if a is None:
                    names.append("")
                    continue
                nm = f"a{k}"
                names.append(nm)
                if _isc(a):
                    inits.append(numpy_helper.from_array(a, nm))
                else:
                    lo, hi = alg.bounds(a)
                    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                        return [_Top() for _ in range(n_out)]
                    ins.append(
                        onnx.helper.make_tensor_value_info(
                            nm, onnx.TensorProto.DOUBLE, list(np.shape(lo))
                        )
                    )
                    ranges[nm] = (lo, hi)
            n = onnx.NodeProto()
            n.CopyFrom(node)
            del n.input[:]
            n.input.extend(names)
            outs = [f"o{k}" for k in range(n_out)]
            del n.output[:]
            n.output.extend(outs)
            g = onnx.helper.make_graph(
                [n],
                "n",
                ins,
                [onnx.helper.make_empty_tensor_value_info(o) for o in outs],
                inits,
            )
            m = onnx.helper.make_model(g, opset_imports=self.opsets)
            m.ir_version = 8
            res = _interval.propagate(m, ranges)
            boxes: List[Value] = []
            for o in outs:
                if o not in res.intervals:
                    return [_Top() for _ in range(n_out)]
                lo, hi = res.intervals[o]
                lo, hi = (
                    np.asarray(lo, dtype=np.float64),
                    np.asarray(hi, dtype=np.float64),
                )
                if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                    return [_Top() for _ in range(n_out)]
                boxes.append(alg.from_box(lo, hi))
            return boxes
        except Exception:
            return [_Top() for _ in range(n_out)]

    # -- op rules
    def _axes(
        self, node: onnx.NodeProto, args: List[Any], nd: int, default_all: bool = True
    ) -> List[int]:
        at = _attrs(node)
        if "axes" in at:
            axes = [int(a) for a in at["axes"]]
        elif len(args) > 1 and args[1] is not None and _isc(args[1]):
            axes = [int(a) for a in args[1].reshape(-1)]
        else:
            if not default_all or at.get("noop_with_empty_axes", 0):
                return []
            axes = list(range(nd))
        return [a % nd for a in axes]

    def op_Identity(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return args[0]

    op_Dropout = op_Identity

    def op_Cast(self, node: Any, args: List[Any], A: _Alg) -> Any:
        if _attrs(node).get("to") in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
            return args[0]
        raise _Unsupported("Cast to a non-float type")

    def op_Neg(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.neg(args[0])

    def op_Add(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.add(args[0], args[1])

    def op_Sub(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.sub(args[0], args[1])

    def op_Mul(self, node: Any, args: List[Any], A: _Alg) -> Any:
        if (
            args[0] is args[1]
        ):  # x * x: the square rule knows the result is non-negative
            return A.square(args[0])
        return A.mul(args[0], args[1])

    def op_Div(self, node: Any, args: List[Any], A: _Alg) -> Any:
        b = args[1]
        if _isc(b):
            with np.errstate(all="ignore"):
                inv = 1.0 / b
            if not np.all(np.isfinite(inv)):
                raise _Unsupported("division by a constant that contains zero")
            return A.mul(args[0], inv)
        return A.mul(args[0], A.func("recip", b))

    def op_Reciprocal(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("recip", args[0])

    def op_Sqrt(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("sqrt", args[0])

    def op_Exp(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("exp", args[0])

    def op_Log(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("log", args[0])

    def op_Erf(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("erf", args[0])

    def op_Sigmoid(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("sigmoid", args[0])

    def op_Tanh(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.func("tanh", args[0])

    def op_Relu(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.relu(args[0])

    def op_Pow(self, node: Any, args: List[Any], A: _Alg) -> Any:
        e = args[1]
        if not _isc(e) or e.size != 1:
            raise _Unsupported("Pow with a non-constant or non-scalar exponent")
        p = float(e.reshape(-1)[0])
        x = args[0]
        if p == 2.0:
            return A.square(x)
        if p == 3.0:
            return A.mul(A.mul(x, x), x)
        if p == 1.0:
            return x
        if p == 0.5:
            return A.func("sqrt", x)
        if p == -1.0:
            return A.func("recip", x)
        if p == -0.5:
            return A.func("rsqrt", x)
        raise _Unsupported(f"Pow with exponent {p}")

    def op_MatMul(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.matmul(args[0], args[1])

    def op_Gemm(self, node: Any, args: List[Any], A: _Alg) -> Any:
        at = _attrs(node)
        a, b = args[0], args[1]
        if at.get("transA", 0):
            a = A.transpose(a, None)
        if at.get("transB", 0):
            b = A.transpose(b, None)
        y = A.matmul(a, b)
        alpha, beta = float(at.get("alpha", 1.0)), float(at.get("beta", 1.0))
        if alpha != 1.0:
            y = A.mul(y, np.float64(alpha))
        if len(args) > 2 and args[2] is not None:
            c = args[2]
            y = A.add(y, A.mul(c, np.float64(beta)) if beta != 1.0 else c)
        return y

    def op_Conv(self, node: Any, args: List[Any], A: _Alg) -> Any:
        x, w = args[0], args[1]
        if not _isc(w) or (len(args) > 2 and args[2] is not None and not _isc(args[2])):
            raise _Unsupported("Conv with non-constant weights")
        at = _attrs(node)
        if at.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise _Unsupported("Conv auto_pad")
        if w.ndim != 4:
            raise _Unsupported(f"Conv with {w.ndim - 2} spatial dims")
        strides = list(at.get("strides", [1, 1]))
        dil = list(at.get("dilations", [1, 1]))
        pads = list(at.get("pads", [0, 0, 0, 0]))
        group = int(at.get("group", 1))
        wp, wn = np.maximum(w, 0.0), np.minimum(w, 0.0)
        bias = None
        if len(args) > 2 and args[2] is not None:
            bias = np.asarray(args[2], dtype=np.float64).reshape(1, -1, 1, 1)

        def mk(ww: Any) -> Callable:
            return lambda t: _conv2d(t, ww, strides, pads, dil, group)

        return A.lin(x, _Lin(mk(w), mk(wp), mk(wn), bias), True)

    def op_GlobalAveragePool(self, node: Any, args: List[Any], A: _Alg) -> Any:
        x = args[0]
        nd = len(A.shape(x))
        axes = list(range(2, nd))
        n = float(np.prod([A.shape(x)[a] for a in axes]))
        return A.mul(A.sum(x, axes, True), np.float64(1.0 / n))

    def op_AveragePool(self, node: Any, args: List[Any], A: _Alg) -> Any:
        at = _attrs(node)
        k = list(at.get("kernel_shape", []))
        pads = list(at.get("pads", [0, 0, 0, 0]))
        if (
            len(k) != 2
            or any(pads)
            or at.get("ceil_mode", 0)
            or at.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET")
        ):
            raise _Unsupported("AveragePool with padding, ceil_mode or non-2-D kernel")
        x = args[0]
        c = A.shape(x)[1]
        w = np.zeros((c, 1, k[0], k[1])) + 1.0 / (k[0] * k[1])
        strides = list(at.get("strides", [1, 1]))
        wf = lambda t: _conv2d(t, w, strides, [0, 0, 0, 0], [1, 1], c)  # noqa: E731
        zero = lambda t: np.zeros_like(wf(t))  # noqa: E731
        return A.lin(x, _Lin(wf, wf, zero), True)

    def op_BatchNormalization(self, node: Any, args: List[Any], A: _Alg) -> Any:
        at = _attrs(node)
        if at.get("training_mode", 0):
            raise _Unsupported("BatchNormalization training_mode=1")
        x = args[0]
        if not all(_isc(a) for a in args[1:5]):
            raise _Unsupported("BatchNormalization with non-constant statistics")
        scale, bias, mean, var = (np.asarray(a, dtype=np.float64) for a in args[1:5])
        s = scale / np.sqrt(var + float(at.get("epsilon", 1e-5)))
        shp = [1, -1] + [1] * (len(A.shape(x)) - 2)
        return A.add(
            A.mul(A.add(x, (-mean).reshape(shp)), s.reshape(shp)), bias.reshape(shp)
        )

    def _softmax_core(self, x: Any, axis: int, A: _Alg) -> Any:
        nd = len(A.shape(x))
        axis %= nd
        _, hi = A.bounds(x)
        m = np.max(
            hi, axis=axis, keepdims=True
        )  # a constant shift: softmax is shift-invariant
        e = A.func("exp", A.add(x, -m))
        s = A.sum(e, [axis], True)
        return A.mul(e, A.func("recip", s))

    def op_Softmax(self, node: Any, args: List[Any], A: _Alg) -> Any:
        if self.opset < 13:
            raise _Unsupported("Softmax before opset 13 flattens to 2-D")
        return self._softmax_core(args[0], int(_attrs(node).get("axis", -1)), A)

    def op_Gelu(self, node: Any, args: List[Any], A: _Alg) -> Any:
        x = args[0]
        approx = _attrs(node).get("approximate", b"none")
        if approx in (b"tanh", "tanh"):
            x3 = A.mul(A.mul(x, x), x)
            inner = A.mul(
                A.add(x, A.mul(x3, np.float64(0.044715))),
                np.float64(math.sqrt(2.0 / math.pi)),
            )
            return A.mul(
                A.mul(x, A.add(A.func("tanh", inner), np.float64(1.0))), np.float64(0.5)
            )
        inner = A.func("erf", A.mul(x, np.float64(1.0 / math.sqrt(2.0))))
        return A.mul(A.mul(x, A.add(inner, np.float64(1.0))), np.float64(0.5))

    def _norm(self, node: Any, args: List[Any], A: _Alg, center: bool) -> Any:
        at = _attrs(node)
        x = args[0]
        nd = len(A.shape(x))
        axis = int(at.get("axis", -1)) % nd
        axes = list(range(axis, nd))
        n = float(np.prod([A.shape(x)[a] for a in axes]))
        eps = float(at.get("epsilon", 1e-5))
        xm = (
            A.add(x, A.neg(A.mul(A.sum(x, axes, True), np.float64(1.0 / n))))
            if center
            else x
        )
        var = A.mul(A.sum(A.square(xm), axes, True), np.float64(1.0 / n))
        inv = A.func("rsqrt", A.add(var, np.float64(eps)))
        y = A.mul(xm, inv)
        scale = args[1] if len(args) > 1 else None
        bias = args[2] if len(args) > 2 else None
        if scale is not None:
            y = A.mul(y, scale)
        if bias is not None:
            y = A.add(y, bias)
        return y

    def op_LayerNormalization(self, node: Any, args: List[Any], A: _Alg) -> Any:
        if len([o for o in node.output if o]) > 1:
            raise _Unsupported("LayerNormalization with Mean/InvStdDev outputs")
        return self._norm(node, args, A, True)

    def op_RMSNormalization(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return self._norm(node, args, A, False)

    op_SimplifiedLayerNormalization = op_RMSNormalization

    def op_ReduceMean(self, node: Any, args: List[Any], A: _Alg) -> Any:
        x = args[0]
        axes = self._axes(node, args, len(A.shape(x)))
        n = float(np.prod([A.shape(x)[a] for a in axes])) if axes else 1.0
        keep = bool(_attrs(node).get("keepdims", 1))
        return A.mul(A.sum(x, axes, keep), np.float64(1.0 / n)) if axes else x

    def op_ReduceSum(self, node: Any, args: List[Any], A: _Alg) -> Any:
        x = args[0]
        axes = self._axes(node, args, len(A.shape(x)))
        return A.sum(x, axes, bool(_attrs(node).get("keepdims", 1))) if axes else x

    def op_Transpose(self, node: Any, args: List[Any], A: _Alg) -> Any:
        perm = _attrs(node).get("perm")
        return A.transpose(args[0], list(perm) if perm else None)

    def op_Reshape(self, node: Any, args: List[Any], A: _Alg) -> Any:
        if not _isc(args[1]):
            raise _Unsupported("Reshape with a non-constant shape")
        if _attrs(node).get("allowzero", 0):
            raise _Unsupported("Reshape allowzero=1")
        old = A.shape(args[0])
        new = [
            old[i] if int(s) == 0 else int(s) for i, s in enumerate(args[1].reshape(-1))
        ]
        return A.reshape(args[0], new)

    def op_Flatten(self, node: Any, args: List[Any], A: _Alg) -> Any:
        x = args[0]
        old = A.shape(x)
        axis = int(_attrs(node).get("axis", 1)) % (len(old) + 1)
        return A.reshape(x, [int(np.prod(old[:axis], dtype=np.int64)), -1])

    def op_Squeeze(self, node: Any, args: List[Any], A: _Alg) -> Any:
        old = A.shape(args[0])
        axes = self._axes(node, args, len(old), default_all=False)
        if not axes:
            axes = [i for i, d in enumerate(old) if d == 1]
        return A.reshape(args[0], [d for i, d in enumerate(old) if i not in axes])

    def op_Unsqueeze(self, node: Any, args: List[Any], A: _Alg) -> Any:
        old = list(A.shape(args[0]))
        at = _attrs(node)
        raw = at["axes"] if "axes" in at else args[1].reshape(-1)
        nd = len(old) + len(raw)
        new = list(old)
        for a in sorted(int(a) % nd for a in raw):
            new.insert(a, 1)
        return A.reshape(args[0], new)

    def op_Concat(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return A.concat([a for a in args if a is not None], int(_attrs(node)["axis"]))

    def op_Shape(self, node: Any, args: List[Any], A: _Alg) -> Any:
        return np.array(A.shape(args[0]), dtype=np.int64)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


@dataclasses.dataclass
class TaylorResult:
    """Per-tensor Taylor models of one model (see :func:`propagate`)."""

    tensors: Dict[str, Value]
    notes: List[str]
    degree: int
    n_symbols: int

    def bounds(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Elementwise ``(lo, hi)`` enclosing ``name`` for every input in the box."""
        v = self.tensors[name]
        if isinstance(v, TaylorModel):
            return v.bounds()
        if isinstance(v, _Top):
            shape = v.shape if v.shape is not None else ()
            return np.full(shape, -np.inf), np.full(shape, np.inf)
        a = np.asarray(v, dtype=np.float64)
        return a, a

    def hull(self, name: str) -> Tuple[float, float]:
        lo, hi = self.bounds(name)
        return float(np.min(lo)), float(np.max(hi))


def _static_inputs(
    model: onnx.ModelProto, input_ranges: Optional[Dict[str, Tuple]]
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    inits = {t.name for t in model.graph.initializer}
    ranges = dict(_ranges.get_ranges(model))
    ranges.update(input_ranges or {})
    out = []
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        tt = vi.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else 0 for d in tt.shape.dim]
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"input {vi.name!r} needs a fully static shape")
        if vi.name not in ranges:
            raise ValueError(
                f"input {vi.name!r} has no finite range; pass input_ranges"
            )
        lo, hi = (
            np.broadcast_to(np.asarray(b, dtype=np.float64), dims).copy()
            for b in ranges[vi.name]
        )
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError(f"input {vi.name!r} has an unbounded range")
        if np.any(lo > hi):
            raise ValueError(f"input {vi.name!r} has an empty range")
        out.append((vi.name, lo, hi))
    return out


def propagate(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    degree: int = DEFAULT_DEGREE,
    max_vars: int = DEFAULT_MAX_VARS,
    quad_vars: int = DEFAULT_QUAD_VARS,
) -> TaylorResult:
    """Propagate the input box through ``model`` as Taylor models.

    :param input_ranges: ``{input: (lo, hi)}`` (scalars or arrays broadcastable to the input
        shape), merged over the model's own ``onnxsim.range.*`` annotations. Every graph input
        needs a finite, static-shape range, else ``ValueError``.
    :param degree: 1 (linear + Lagrange remainder) or 2 (adds the quadratic part).
    :param max_vars: how many of the largest-radius input elements get a noise symbol; the rest
        enter as interval remainders (sound, but their correlation is lost).
    :param quad_vars: how many of those symbols get quadratic terms (the largest-radius ones).
    """
    if degree not in (1, 2):
        raise ValueError(f"degree must be 1 or 2, got {degree!r}")
    inputs = _static_inputs(model, input_ranges)
    radii = (
        np.concatenate([((hi - lo) / 2.0).reshape(-1) for _, lo, hi in inputs])
        if inputs
        else np.zeros(0)
    )
    order = np.argsort(-radii, kind="stable")
    nsym = int(min(max_vars, int(np.count_nonzero(radii > 0))))
    sym_of = np.full(radii.shape, -1, dtype=np.int64)
    sym_of[order[:nsym]] = np.arange(nsym)
    ctx = _Ctx(nsym, quad_vars if degree >= 2 else 0)
    tms: Dict[str, Any] = {}
    offset = 0
    for name, lo, hi in inputs:
        size = lo.size
        c = (lo + hi) / 2.0
        rad = ((hi - lo) / 2.0).reshape(-1)
        smap = sym_of[offset : offset + size]
        L = np.zeros((nsym, size))
        has = smap >= 0
        L[smap[has], np.flatnonzero(has)] = rad[has]
        rr = np.where(has, 0.0, rad).reshape(lo.shape)
        tms[name] = TaylorModel(
            c,
            L.reshape((nsym,) + lo.shape),
            np.zeros((ctx.npairs,) + lo.shape),
            -rr,
            rr.copy(),
            lo,
            hi,
            ctx,
        )
        offset += size
    ev = _Evaluator(model, _TMAlg(ctx, degree))
    values = ev.run(tms)
    return TaylorResult(dict(values), ev.notes, degree, nsym)


@dataclasses.dataclass
class MeanValueResult:
    """Mean-value-form bounds of the requested tensors (see :func:`mean_value_bounds`)."""

    bounds: Dict[str, Tuple[np.ndarray, np.ndarray]]
    notes: List[str]


def mean_value_bounds(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    output: Union[None, str, Sequence[str]] = None,
    max_inputs: int = 256,
) -> MeanValueResult:
    """Bounds from the mean-value form ``f(x) in f(c) + J(box) (x - c)``.

    ``J(box)`` is an interval enclosure of the Jacobian over the *whole* box, obtained by
    forward-mode interval automatic differentiation of the supported ops, so the enclosure is
    rigorous (for the real-number function; ``f(c)`` is evaluated in float64 and the result
    widened by a relative ``1e-9``). The result is intersected with the interval enclosure
    the same pass produces, so it is never looser than plain interval propagation.

    The Jacobian has one row per *input element*, so this is for low-dimensional inputs:
    more than ``max_inputs`` input elements raises ``ValueError``. An op without a
    differentiation rule makes every tensor computed from it unbounded.
    """
    inputs = _static_inputs(model, input_ranges)
    n = sum(lo.size for _, lo, _ in inputs)
    if n > max_inputs:
        raise ValueError(
            f"{n} input elements exceed max_inputs={max_inputs}; use propagate() instead"
        )
    alg = _DualAlg(n)
    duals: Dict[str, Any] = {}
    centres: Dict[str, Any] = {}
    offset = 0
    for name, lo, hi in inputs:
        d = np.zeros((n,) + lo.shape)
        flat = d.reshape(n, -1)
        flat[offset + np.arange(lo.size), np.arange(lo.size)] = 1.0
        duals[name] = _Dual(lo.copy(), hi.copy(), d, d.copy())
        centres[name] = (lo + hi) / 2.0
        offset += lo.size
    ev = _Evaluator(model, alg)
    values = ev.run(duals)
    # f(c): the same graph on plain numbers at the centre of the box.
    cvals = _Evaluator(model, NUM).run(centres)
    names: List[str]
    if output is None:
        names = [o.name for o in model.graph.output]
    elif isinstance(output, str):
        names = [output]
    else:
        names = list(output)
    res: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for nm in names:
        v = values[nm]
        if _isc(v):
            res[nm] = (v, v)
            continue
        if isinstance(v, _Top):
            shape = v.shape if v.shape is not None else ()
            res[nm] = (np.full(shape, -np.inf), np.full(shape, np.inf))
            continue
        if not isinstance(v, _Dual):
            raise TypeError(f"unexpected abstract value for {nm!r}: {type(v).__name__}")
        fc = cvals[nm]
        if _isc(fc) and np.all(np.isfinite(fc)):
            dx_lo = np.concatenate(
                [(lo - centres[k]).reshape(-1) for k, lo, _ in inputs]
            )
            dx_hi = np.concatenate(
                [(hi - centres[k]).reshape(-1) for k, _, hi in inputs]
            )
            shp = (n,) + (1,) * (v.dlo.ndim - 1)
            tl, th = _imul(v.dlo, v.dhi, dx_lo.reshape(shp), dx_hi.reshape(shp))
            lo = fc + tl.sum(axis=0)
            hi = fc + th.sum(axis=0)
            pad = _PAD * (np.abs(fc) + np.maximum(np.abs(tl), np.abs(th)).sum(axis=0))
            lo, hi = np.maximum(lo - pad, v.vlo), np.minimum(hi + pad, v.vhi)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        else:
            lo, hi = v.vlo, v.vhi
        res[nm] = _widen(lo, hi)
    return MeanValueResult(res, ev.notes)


# --------------------------------------------------------------------------
# Comparing methods
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Comparison:
    """Output bounds of several methods on one model, plus sampled ground truth."""

    outputs: List[str]
    bounds: Dict[
        str, Dict[str, Tuple[np.ndarray, np.ndarray]]
    ]  # method -> output -> (lo, hi)
    errors: Dict[str, str]  # method -> why it produced nothing
    notes: List[str]

    def width(self, method: str, output: str) -> float:
        lo, hi = self.bounds[method][output]
        return float(np.mean(hi - lo))

    def table(self) -> str:
        return format_comparison(self)


def _sampled_bounds(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]],
    samples: int,
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    import onnxruntime as ort

    inputs = _static_inputs(model, input_ranges)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(seed)
    outs = [o.name for o in model.graph.output]
    lo: Dict[str, Any] = {}
    hi: Dict[str, Any] = {}
    elem = {i.name: i.type.tensor_type.elem_type for i in model.graph.input}
    for s in range(samples):
        feed = {}
        for name, a, b in inputs:
            u = rng.random(a.shape)
            if s < 2:  # the two extreme corners
                u = np.full(a.shape, float(s))
            elif (
                s % 2 == 0
            ):  # every other sample: a random vertex (extremes sit near vertices)
                u = (rng.random(a.shape) < 0.5).astype(np.float64)
            feed[name] = (a + (b - a) * u).astype(
                onnx.helper.tensor_dtype_to_np_dtype(elem[name])
            )
        for nm, v in zip(outs, sess.run(outs, feed)):
            v = np.asarray(v, dtype=np.float64)
            lo[nm] = v if nm not in lo else np.minimum(lo[nm], v)
            hi[nm] = v if nm not in hi else np.maximum(hi[nm], v)
    return {nm: (lo[nm], hi[nm]) for nm in outs}


def compare_methods(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    degree: int = DEFAULT_DEGREE,
    samples: int = 256,
    seed: int = 0,
    methods: Sequence[str] = ("interval", "zonotope", "crown", "taylor", "mean_value"),
) -> Comparison:
    """Output bounds from interval, zonotope, CROWN, Taylor and mean-value methods, side by side.

    ``"sampled"`` (random, random-vertex and corner inputs through onnxruntime, when it is
    importable) is added as a reference: it is an *inner* approximation of the true range, so a sound method's
    width should sit just above it. A method that cannot handle the model (an unbounded
    input, an unsupported shape) is reported in ``errors`` instead of aborting the comparison.
    """
    outs = [o.name for o in model.graph.output]
    bounds: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    errors: Dict[str, str] = {}
    notes: List[str] = []
    for m in methods:
        try:
            if m == "interval":
                r = _interval.propagate(model, input_ranges)
                bounds[m] = {
                    o: (np.asarray(r.intervals[o][0]), np.asarray(r.intervals[o][1]))
                    for o in outs
                }
            elif m == "zonotope":
                from . import zonotope as _zono

                zr = _zono.propagate(model, input_ranges)
                bounds[m] = {o: zr.bounds(o) for o in outs}
                notes += [f"zonotope: {x}" for x in zr.notes]
            elif m == "crown":
                from . import crown as _crown

                cr = _crown.bounds(model, input_ranges, method="crown")
                bounds[m] = {o: (cr[o].lo, cr[o].hi) for o in outs}
            elif m == "taylor":
                tr = propagate(model, input_ranges, degree=degree)
                bounds[m] = {o: tr.bounds(o) for o in outs}
                notes += [f"taylor: {x}" for x in tr.notes]
            elif m == "mean_value":
                mv = mean_value_bounds(model, input_ranges)
                bounds[m] = {o: mv.bounds[o] for o in outs}
                notes += [f"mean_value: {x}" for x in mv.notes]
            else:
                raise ValueError(f"unknown method {m!r}")
        except Exception as e:
            errors[m] = f"{type(e).__name__}: {e}"
    try:
        bounds["sampled"] = _sampled_bounds(model, input_ranges, samples, seed)
    except Exception as e:
        errors["sampled"] = f"{type(e).__name__}: {e}"
    return Comparison(outs, bounds, errors, notes)


def format_comparison(cmp: Comparison) -> str:
    """Mean output width per method, as a ratio to the sampled width when it is available."""
    ref = cmp.bounds.get("sampled")
    head = f"{'method':12s}" + "".join(f"{o[:16]:>20s}" for o in cmp.outputs)
    rows = [head]
    for m, per in cmp.bounds.items():
        cells = []
        for o in cmp.outputs:
            w = float(np.mean(per[o][1] - per[o][0]))
            if ref is not None and m != "sampled":
                rw = float(np.mean(ref[o][1] - ref[o][0]))
                cells.append(f"{w:11.4g} ({w / rw:6.2f}x)" if rw > 0 else f"{w:11.4g}")
            else:
                cells.append(f"{w:11.4g}")
        rows.append(f"{m:12s}" + "".join(f"{c:>20s}" for c in cells))
    for m, e in cmp.errors.items():
        rows.append(f"{m:12s}  unavailable: {e[:80]}")
    return "\n".join(rows)
