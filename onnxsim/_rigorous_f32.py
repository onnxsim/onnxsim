"""Rigorous float32: values that carry a bound on their own rounding error.

``onnxsim.crown`` can run its backward pass in float32 on a GPU. A float32 bound is only a
*bound* if the rounding error of computing it is accounted for, so every tensor in that mode
is a :class:`P` (a float32 value ``v`` plus an elementwise error bound ``e``): the real number
the computation stands for lies in ``[v - e, v + e]``. :class:`K` is the same for a constant
(a weight, a relaxation line): its ``e`` is the error of having rounded it to float32.

The error model (Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 3):

* one float32 operation ``fl(x op y) = (x op y)(1 + t)`` with ``|t| <= u = 2**-24``
  (IEEE round-to-nearest; a fused multiply-add rounds less, so the bound still holds);
* a sum or dot product of ``n`` terms computed in *any* order has error at most
  ``gamma_n * sum |x_i y_i|`` with ``gamma_n = n u / (1 - n u)``;
* errors of the inputs propagate through a linear map ``M`` as ``|M| e`` (a matrix product with
  the absolute values), and through a product with a constant ``c`` as ``|c| e``;
* ``pos`` / ``neg`` (the sign selection of CROWN) are 1-Lipschitz, so a wrong sign caused by
  a rounded value moves the result by at most ``|error|`` times the slope.

Everything is done with *upper* bounds only, and the bound computed for the error is itself
computed in float32, so every error array is inflated by ``1 + 2 gamma_{n+8}`` (the relative
error of the nonnegative sums that formed it) and floored by the smallest subnormal ``eta``
(absolute error of an operation that underflows). A result that overflows to ``inf`` / ``nan``
is reported as ``-inf`` (a vacuous but sound lower bound), never as a finite number.

Large reductions are summed block by block (``BLOCK`` terms at a time, then the block sums, ...)
so the number of terms ``n`` in ``gamma_n`` is ``levels * (BLOCK - 1)`` instead of the full
element count: a 800k-element row sum costs ``gamma_{384}`` (2e-5) instead of ``gamma_{800k}``
(5e-2). Any order *within* a block is covered by the ``gamma`` bound.

Assumptions the proof rests on (they are about the hardware / library, not about this code):
IEEE-754 float32 multiply and add; no TF32 or other reduced-precision matrix units (see
``_device.strict_float32``); matrix products by BLAS-style multiply-accumulate (no Strassen /
Winograd / FFT variants, which have different error behaviour); no flush-to-zero beyond the
``eta`` floor.

Imported lazily: it needs torch.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

U = 2.0**-24  # unit roundoff of float32
UP = U / (1.0 - U)  # bound of |fl(x) - x| / |fl(x)|
ETA = 2.0**-149  # smallest positive float32 subnormal
BLOCK = 128  # terms summed in one go; larger sums are summed hierarchically
_EPS64 = 2.0**-52


def gamma(n: int) -> float:
    """Higham's gamma_n; ``inf`` when ``n u >= 1/2`` (too many terms for the model)."""
    nu = n * U
    if nu >= 0.5:
        return math.inf
    return nu / (1.0 - nu)


def _infl(n: int) -> float:
    return 1.0 + 2.0 * gamma(n + 8)


def _fin(e: Any, n: int) -> Any:
    """Make a computed error bound safe: inflate for its own rounding, floor for underflow."""
    return e * _infl(n) + ETA


# --------------------------------------------------------------------------
# directed rounding of float64 numbers to float32
# --------------------------------------------------------------------------


def up32(x: np.ndarray) -> np.ndarray:
    """Smallest float32 ``>= x``, returned as float64 (so it can be used on the host)."""
    x = np.asarray(x, dtype=np.float64)
    y = x.astype(np.float32)
    below = y.astype(np.float64) < x
    return np.where(below, np.nextafter(y, np.float32(np.inf)), y).astype(np.float64)


def down32(x: np.ndarray) -> np.ndarray:
    """Largest float32 ``<= x``, returned as float64."""
    x = np.asarray(x, dtype=np.float64)
    y = x.astype(np.float32)
    above = y.astype(np.float64) > x
    return np.where(above, np.nextafter(y, np.float32(-np.inf)), y).astype(np.float64)


# --------------------------------------------------------------------------
# hierarchical sum
# --------------------------------------------------------------------------


def tree_sum(x: Any, dims: Tuple[int, ...], keepdim: bool = False) -> Tuple[Any, int]:
    """Sum ``x`` over ``dims`` block by block; returns ``(sum, n_eff)``.

    ``n_eff`` is the ``n`` of the ``gamma_n`` that bounds the relative error of the result
    against ``sum |x|``.
    """
    nd = x.dim()
    dims = tuple(sorted(d % nd for d in dims))
    keep = [d for d in range(nd) if d not in dims]
    n = 1
    for d in dims:
        n *= int(x.shape[d])
    if n <= BLOCK:
        return x.sum(dim=dims, keepdim=keepdim), max(n - 1, 1)
    xp = x.permute(*keep, *dims)
    lead = tuple(int(s) for s in xp.shape[: len(keep)])
    flat = xp.reshape(-1, n)
    levels = 0
    while flat.shape[1] > BLOCK:
        k = int(flat.shape[1])
        pad = (-k) % BLOCK
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        flat = flat.reshape(flat.shape[0], -1, BLOCK).sum(-1)
        levels += 1
    out = flat.sum(1)
    if keepdim:
        shape = list(x.shape)
        for d in dims:
            shape[d] = 1
        out = out.reshape(shape)
    else:
        out = out.reshape(lead)
    return out, (levels + 1) * (BLOCK - 1)


def _contraction_size(eq: str, shapes: Sequence[Tuple[int, ...]]) -> int:
    """Number of terms summed per output element of ``einsum(eq, ...)``."""
    ins, out = eq.replace(" ", "").split("->")
    specs = ins.split(",")
    size: Dict[str, int] = {}
    for sp, sh in zip(specs, shapes):
        for ch, d in zip(sp, sh):
            size[ch] = max(size.get(ch, 1), int(d))
    n = 1
    for ch in set("".join(specs)) - set(out):
        n *= size[ch]
    return n


# --------------------------------------------------------------------------
# constants and values
# --------------------------------------------------------------------------


class K:
    """A float32 constant: value ``v``, ``a = |v|``, and ``e`` = |intended - v| (None: exact)."""

    __slots__ = ("v", "a", "e")

    def __init__(self, v: Any, e: Optional[Any] = None, a: Optional[Any] = None):
        self.v = v
        self.a = v.abs() if a is None else a
        self.e = e

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.v.shape)

    def __getitem__(self, idx: Any) -> "K":
        return K(self.v[idx], None if self.e is None else self.e[idx], self.a[idx])

    @staticmethod
    def from_numpy(x: Any, device: Any) -> "K":
        """Round a float64 array to float32 and record the cast error (None when exact)."""
        x64 = np.asarray(x, dtype=np.float64)
        v32 = x64.astype(np.float32)
        err = np.abs(x64 - v32.astype(np.float64))
        v = torch.as_tensor(np.ascontiguousarray(v32), device=device)
        if not np.any(err):
            return K(v, None)
        return K(v, torch.as_tensor(up32(err).astype(np.float32), device=device))

    @staticmethod
    def exact(x: Any, device: Any) -> "K":
        """A float32 array that already *is* the intended value (relaxation lines, 0/1 masks)."""
        if isinstance(x, torch.Tensor):
            return K(x.to(device=device, dtype=torch.float32))
        x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float64).astype(np.float32))
        return K(torch.as_tensor(x32, device=device))


class P:
    """A float32 value ``v`` and an elementwise bound ``e`` on its distance to the real number."""

    __slots__ = ("v", "e")

    def __init__(self, v: Any, e: Optional[Any] = None):
        self.v = v
        self.e = e  # None: exact

    # -- plumbing ---------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.v.shape)

    def _err(self) -> Any:
        return self.e if self.e is not None else torch.zeros_like(self.v)

    def reshape(self, shape: Sequence[int]) -> "P":
        shape = tuple(shape)
        return P(
            self.v.reshape(shape), None if self.e is None else self.e.reshape(shape)
        )

    def permute(self, *perm: int) -> "P":
        return P(
            self.v.permute(*perm), None if self.e is None else self.e.permute(*perm)
        )

    def __neg__(self) -> "P":
        return P(-self.v, self.e)

    def __getitem__(self, idx: Any) -> "P":
        return P(self.v[idx], None if self.e is None else self.e[idx])

    def __setitem__(self, idx: Any, value: "P") -> None:
        tgt = self.v[idx]
        if tgt.data_ptr() == value.v.data_ptr() and tgt.stride() == value.v.stride():
            return  # the in-place `+=` already updated the view
        self.v[idx] = value.v
        assert self.e is not None
        self.e[idx] = value._err()

    # -- arithmetic -------------------------------------------------------
    @staticmethod
    def _coerce(o: Any, like: "P") -> "P":
        if isinstance(o, P):
            return o
        if isinstance(o, torch.Tensor):  # an exact float32 number, e.g. a multiplier
            return P(o.to(device=like.v.device, dtype=torch.float32))
        raise TypeError(f"cannot add {type(o).__name__} to a rigorous float32 value")

    def __add__(self, o: Any) -> "P":
        o = P._coerce(o, self)
        v = self.v + o.v
        if self.e is None and o.e is None:
            e = UP * v.abs()
        elif o.e is None:
            e = self.e + UP * v.abs()
        elif self.e is None:
            e = o.e + UP * v.abs()
        else:
            e = (self.e + o.e) + UP * v.abs()
        return P(v, _fin(e, 2))

    __radd__ = __add__

    def __iadd__(self, o: Any) -> "P":
        o = P._coerce(o, self)
        assert self.e is not None, "in-place add needs an error array (use ops.zeros)"
        self.v.add_(o.v)
        self.e.add_(UP * self.v.abs())
        if o.e is not None:
            self.e.add_(o.e)
        self.e.mul_(_infl(2)).add_(ETA)
        return self

    def __mul__(self, o: Any) -> "P":
        if isinstance(o, (int, float)):
            f = float(o)
            if f == 1.0:
                return self
            if f == -1.0:
                return -self
            k32 = np.float32(f)
            o = K(
                torch.as_tensor(k32, device=self.v.device),
                None
                if float(k32) == f
                else torch.as_tensor(
                    np.float32(up32(np.array(abs(f - float(k32))))),
                    device=self.v.device,
                ),
            )
        elif isinstance(o, torch.Tensor):
            o = K(o.to(device=self.v.device, dtype=torch.float32))
        if not isinstance(o, K):
            raise TypeError(f"cannot multiply a rigorous value by {type(o).__name__}")
        v = self.v * o.v
        terms = UP * v.abs()
        if self.e is not None:
            terms = terms + o.a * self.e
        if o.e is not None:
            terms = terms + self.v.abs() * o.e
            if self.e is not None:
                terms = terms + self.e * o.e
        return P(v, _fin(terms, 3))

    __rmul__ = __mul__

    def __truediv__(self, f: float) -> "P":
        f = float(f)
        f32 = np.float32(f)
        v = self.v / float(f32)
        e = UP * v.abs()
        if self.e is not None:
            e = e + self.e / abs(float(f32))
        rel = abs(f - float(f32)) / (abs(f) * abs(float(f32)))
        if rel:
            e = e + self.v.abs() * float(up32(np.array(rel)))
        return P(v, _fin(e, 3))

    def __matmul__(self, k: K) -> "P":
        n = int(self.v.shape[-1])
        v = self.v @ k.v
        lhs = self.v.abs() * gamma(n)
        if self.e is not None:
            lhs = lhs + self.e
        e = lhs @ k.a
        if k.e is not None:
            rhs = self.v.abs() if self.e is None else self.v.abs() + self.e
            e = e + rhs @ k.e
        return P(v, _fin(e, n + 2))

    def sum(self, dim: Any = None, keepdim: bool = False) -> "P":
        if dim is None:
            dims: Tuple[int, ...] = tuple(range(self.v.dim()))
        elif isinstance(dim, int):
            dims = (dim,)
        else:
            dims = tuple(dim)
        v, n_eff = tree_sum(self.v, dims, keepdim)
        absum, _ = tree_sum(self.v.abs(), dims, keepdim)
        e = absum * gamma(n_eff)
        if self.e is not None:
            es, _ = tree_sum(self.e, dims, keepdim)
            e = e + es
        return P(v, _fin(e, n_eff + 2))

    # -- sign selection of CROWN ------------------------------------------
    def pos(self) -> "P":
        return P(self.v.clamp(min=0.0), self.e)

    def neg(self) -> "P":
        return P(self.v.clamp(max=0.0), self.e)


def einsum(eq: str, a: P, k: K) -> P:
    """``einsum(eq, a, k)`` with the error of the contraction accounted for."""
    n = _contraction_size(eq, [tuple(a.v.shape), tuple(k.v.shape)])
    v = torch.einsum(eq, a.v, k.v)
    lhs = a.v.abs() * gamma(n)
    if a.e is not None:
        lhs = lhs + a.e
    e = torch.einsum(eq, lhs, k.a)
    if k.e is not None:
        rhs = a.v.abs() if a.e is None else a.v.abs() + a.e
        e = e + torch.einsum(eq, rhs, k.e)
    return P(v, _fin(e, n + 2))


def lower_to_numpy(p: P) -> np.ndarray:
    """The sound lower bound ``v - e`` as float64; ``-inf`` wherever anything overflowed."""
    v = p.v.detach().to(torch.float64)
    if p.e is None:
        e = torch.zeros_like(v)
    else:
        e = p.e.detach().to(torch.float64)
    ok = torch.isfinite(v) & torch.isfinite(e)
    out = torch.where(ok, v - e, torch.full_like(v, -math.inf))
    return np.asarray(out.cpu().numpy(), dtype=np.float64)


def sound_lines_f32(kind: str, r: dict, lo: np.ndarray, hi: np.ndarray) -> dict:
    """float32-valid versions of float64 relaxation lines on the box ``[lo, hi]``.

    A line ``a x + b`` that sandwiches ``f`` on the box stays valid when ``a`` is rounded to
    ``a32`` if ``b`` absorbs ``|a - a32| * max|x|`` in the safe direction (up for an upper
    line, down for a lower one) and is then rounded in that direction too. The returned
    arrays hold float32-representable values, so they can be used as exact constants.
    Slopes and intercepts that are already exact (0, 1) are left alone.
    """
    out = dict(r)
    rad = np.maximum(np.abs(lo), np.abs(hi))

    def slope(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a32 = np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)
        return a32, np.abs(a - a32) * rad

    def widen(b: np.ndarray, extra: np.ndarray, sign: float) -> np.ndarray:
        raw = b + sign * extra
        return raw + sign * 4.0 * _EPS64 * (np.abs(b) + extra)

    if kind == "Relu":
        # lower line is alpha * x with b_l = 0: valid for every alpha in [0, 1], whatever it rounds to
        a32, shift = slope(r["a_u"])
        out["a_u"] = a32
        out["b_u"] = up32(widen(r["b_u"], shift, +1.0))
    else:
        a32, shift = slope(r["a_u"])
        out["a_u"] = a32
        out["b_u"] = up32(widen(r["b_u"], shift, +1.0))
        a32, shift = slope(r["a_l"])
        out["a_l"] = a32
        out["b_l"] = down32(widen(r["b_l"], shift, -1.0))
    return out


__all__: List[str] = [
    "U",
    "UP",
    "ETA",
    "BLOCK",
    "gamma",
    "K",
    "P",
    "einsum",
    "tree_sum",
    "lower_to_numpy",
    "sound_lines_f32",
    "up32",
    "down32",
]
