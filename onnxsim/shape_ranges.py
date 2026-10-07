"""Ranged tensor shapes: dimensions that are known only to lie in an interval.

ONNX shape inference gives up on data-dependent ops. ``NonZero`` of a ``float[3, 4]``
comes out as ``[2, unk__0]`` and ``TopK`` with a runtime ``K`` as all-unknown dims.
But the dynamic dimension is *bounded*: ``NonZero`` returns between 0 and
``numel(x)`` columns, ``TopK`` returns ``K`` of them, ``NonMaxSuppression`` at most
``batches * classes * max_output_boxes_per_class`` rows. A :class:`Dim` carries that
bound, so analyses that need a shape (interval propagation, memory planning) can
keep going instead of discarding the tensor.

This module is pure shape arithmetic -- no values, no numpy tensors -- except where a
rule takes a *shape vector* (the value of a shape-valued input such as ``Reshape``'s
second operand), given as an elementwise ``(lo, hi)`` pair.

Soundness contract: every rule returns dimensions that contain the dimension of
every *valid* execution consistent with its inputs. Where a rule cannot say, the
dimension is ``[0, unbounded)``; ``None`` is returned only when no output shape can
be stated at all (for example an unknown output rank).
"""

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class Dim:
    """One dimension: an integer in ``[lo, hi]``; ``hi=None`` means unbounded above.

    ``sym`` optionally names the dimension (an ONNX ``dim_param`` such as ``batch`` or
    ``unk__0``). Two dims with the same name are the same number at run time, which
    :func:`intersect` and the broadcast rule use.
    """

    lo: int
    hi: Optional[int]
    sym: Optional[str] = None

    def __post_init__(self):
        if self.lo < 0:
            object.__setattr__(self, "lo", 0)
        if self.hi is not None and self.hi < self.lo:
            raise ValueError(f"empty dimension range [{self.lo}, {self.hi}]")

    @property
    def exact(self) -> bool:
        return self.hi is not None and self.lo == self.hi

    @property
    def value(self) -> int:
        if not self.exact:
            raise ValueError(f"{self} is not a single value")
        return self.lo

    def contains(self, n: int) -> bool:
        return n >= self.lo and (self.hi is None or n <= self.hi)

    def __add__(self, other: "Dim") -> "Dim":
        return Dim(self.lo + other.lo, _add_hi(self.hi, other.hi))

    def __mul__(self, other: "Dim") -> "Dim":
        return Dim(self.lo * other.lo, _mul_hi(self.lo, self.hi, other.lo, other.hi))

    def __str__(self) -> str:
        name = f"{self.sym}:" if self.sym else ""
        if self.exact:
            return f"{name}{self.lo}"
        return f"{name}[{self.lo},{'inf' if self.hi is None else self.hi}]"


Shape = Tuple[Dim, ...]


def exact(n: int, sym: Optional[str] = None) -> Dim:
    return Dim(int(n), int(n), sym)


def rng(lo: int, hi: Optional[int], sym: Optional[str] = None) -> Dim:
    return Dim(int(lo), None if hi is None else int(hi), sym)


def from_ints(dims: Sequence[int]) -> Shape:
    return tuple(exact(d) for d in dims)


def is_static(shape: Shape) -> bool:
    return all(d.exact for d in shape)


def static_dims(shape: Shape) -> Tuple[int, ...]:
    return tuple(d.value for d in shape)


def shape_str(shape: Shape) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"


def contains_shape(shape: Shape, actual: Sequence[int]) -> bool:
    """Is a concrete shape one of those ``shape`` allows?"""
    return len(shape) == len(actual) and all(
        d.contains(int(a)) for d, a in zip(shape, actual)
    )


def _add_hi(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if a is None or b is None else a + b


def _mul_hi(
    alo: int, ahi: Optional[int], blo: int, bhi: Optional[int]
) -> Optional[int]:
    # 0 * unbounded is 0, not unbounded: a dimension of exactly 0 empties the tensor.
    if (ahi == 0) or (bhi == 0):
        return 0
    if ahi is None or bhi is None:
        return None
    return ahi * bhi


def _min_hi(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def intersect(a: Dim, b: Dim) -> Optional[Dim]:
    """The values both allow; ``None`` when none (an invalid model)."""
    lo, hi = max(a.lo, b.lo), _min_hi(a.hi, b.hi)
    if hi is not None and hi < lo:
        return None
    return Dim(lo, hi, a.sym if a.sym is not None else b.sym)


def hull(a: Dim, b: Dim) -> Dim:
    """Smallest range containing both."""
    hi = None if a.hi is None or b.hi is None else max(a.hi, b.hi)
    return Dim(min(a.lo, b.lo), hi, a.sym if a.sym == b.sym else None)


def numel(shape: Shape) -> Dim:
    out = exact(1)
    for d in shape:
        out = out * d
    return out


def _vec_dims(lo: np.ndarray, hi: np.ndarray) -> List[Dim]:
    """Shape vector given as elementwise (lo, hi) -> dims; infinities become unbounded."""
    out = []
    for a, b in zip(np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)):
        lo_i = 0 if not np.isfinite(a) else max(0, int(math.floor(a)))
        hi_i = None if not np.isfinite(b) else max(lo_i, int(math.ceil(b)))
        out.append(Dim(lo_i, hi_i))
    return out


def dims_from_vector(lo, hi) -> Shape:
    return tuple(_vec_dims(np.asarray(lo), np.asarray(hi)))


# --------------------------------------------------------------------------
# Broadcasting and layout rules
# --------------------------------------------------------------------------


def broadcast2(a: Shape, b: Shape) -> Optional[Shape]:
    """NumPy broadcasting of two ranged shapes.

    Per aligned pair the result is ``b`` when ``a`` is 1, ``a`` when ``b`` is 1, or
    their common value; with ranges any of those may apply, so the result is the hull
    of the cases that are possible.
    """
    n = max(len(a), len(b))
    pa = (exact(1),) * (n - len(a)) + tuple(a)
    pb = (exact(1),) * (n - len(b)) + tuple(b)
    out = []
    for da, db in zip(pa, pb):
        cases: List[Dim] = []
        if da.contains(1):
            cases.append(db)
        if db.contains(1):
            cases.append(da)
        both = intersect(da, db)
        if both is not None:
            cases.append(both)
        if not cases:
            return None
        d = cases[0]
        for c in cases[1:]:
            d = hull(d, c)
        out.append(d)
    return tuple(out)


def broadcast(shapes: Sequence[Shape]) -> Optional[Shape]:
    out: Optional[Shape] = shapes[0]
    for s in shapes[1:]:
        if out is None:
            return None
        out = broadcast2(out, s)
    return out


def _norm_axis(axis: int, rank: int) -> int:
    a = axis + rank if axis < 0 else axis
    if not 0 <= a < rank:
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    return a


def concat(shapes: Sequence[Shape], axis: int) -> Optional[Shape]:
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        return None
    ax = _norm_axis(axis, rank)
    out: List[Dim] = []
    for i in range(rank):
        if i == ax:
            tot = exact(0)
            for s in shapes:
                tot = tot + s[i]
            out.append(tot)
        else:
            d: Optional[Dim] = shapes[0][i]
            for s in shapes[1:]:
                d = intersect(d, s[i]) if d is not None else None
            if d is None:
                return None
            out.append(d)
    return tuple(out)


def gather(data: Shape, indices: Shape, axis: int = 0) -> Shape:
    ax = _norm_axis(axis, len(data))
    return tuple(data[:ax]) + tuple(indices) + tuple(data[ax + 1 :])


def gather_nd(data: Shape, indices: Shape, batch_dims: int = 0) -> Optional[Shape]:
    """``indices[-1]`` must be an exact number of leading data dims to index."""
    if batch_dims != 0 or not indices or not indices[-1].exact:
        return None
    k = indices[-1].value
    if k > len(data):
        return None
    return tuple(indices[:-1]) + tuple(data[k:])


def squeeze(shape: Shape, axes: Optional[Sequence[int]]) -> Optional[Shape]:
    rank = len(shape)
    if axes is None:
        # Which dims are 1 must be known exactly, otherwise the output rank is unknown.
        if any(d.contains(1) and not d.exact for d in shape):
            return None
        return tuple(d for d in shape if not (d.exact and d.value == 1))
    drop = {_norm_axis(a, rank) for a in axes}
    return tuple(d for i, d in enumerate(shape) if i not in drop)


def unsqueeze(shape: Shape, axes: Sequence[int]) -> Shape:
    rank = len(shape) + len(axes)
    ax = sorted(_norm_axis(a, rank) for a in axes)
    out: List[Dim] = []
    it = iter(shape)
    for i in range(rank):
        out.append(exact(1) if i in ax else next(it))
    return tuple(out)


def transpose(shape: Shape, perm: Optional[Sequence[int]]) -> Shape:
    return tuple(shape[i] for i in (perm if perm else reversed(range(len(shape)))))


def flatten(shape: Shape, axis: int) -> Shape:
    ax = axis + len(shape) if axis < 0 else axis
    return (numel(tuple(shape[:ax])), numel(tuple(shape[ax:])))


def expand(shape: Shape, target: Shape) -> Optional[Shape]:
    return broadcast2(shape, target)


def tile(shape: Shape, repeats: Shape) -> Optional[Shape]:
    if len(repeats) != len(shape):
        return None
    return tuple(d * r for d, r in zip(shape, repeats))


def reshape(
    shape: Shape, target_lo: np.ndarray, target_hi: np.ndarray, allowzero: bool = False
) -> Optional[Shape]:
    """``Reshape`` with the target given as an elementwise (lo, hi) vector.

    ``0`` copies the input dim (unless ``allowzero``) and ``-1`` is inferred from the
    element count. The target must have exact ``0``/``-1`` markers; a ranged entry
    that *might* be one of them leaves the output unknown rather than guessed.
    """
    lo, hi = np.asarray(target_lo).reshape(-1), np.asarray(target_hi).reshape(-1)
    out: List[Optional[Dim]] = []
    infer_at = -1
    for i, (a, b) in enumerate(zip(lo, hi)):
        if a == b and a == -1:
            if infer_at >= 0:
                return None
            infer_at = i
            out.append(None)
        elif a == b and a == 0 and not allowzero:
            if i >= len(shape):
                return None
            out.append(shape[i])
        elif a < 0 or (a <= 0 <= b and not allowzero and not (a == b)):
            return None  # a ranged entry that may be 0 or -1: cannot say
        else:
            out.append(
                Dim(
                    max(0, int(math.floor(a))),
                    None if not np.isfinite(b) else int(math.ceil(b)),
                )
            )
    n = numel(shape)
    if infer_at >= 0:
        rest = exact(1)
        for d in out:
            if d is not None:
                rest = rest * d
        if rest.lo == 0:
            # a zero in the product makes the inferred dim ill-defined; allow anything up to numel
            out[infer_at] = Dim(0, n.hi)
        else:
            hi_ = None if n.hi is None else n.hi // rest.lo
            lo_ = 0 if rest.hi is None else -(-n.lo // rest.hi)  # ceil(n.lo / rest.hi)
            out[infer_at] = Dim(lo_, hi_)
    return tuple(d for d in out if d is not None)  # type: ignore[misc]


def slice_(
    shape: Shape,
    starts: Sequence[int],
    ends: Sequence[int],
    axes: Optional[Sequence[int]] = None,
    steps: Optional[Sequence[int]] = None,
    enumerate_limit: int = 4096,
) -> Optional[Shape]:
    """``Slice`` with constant ``starts/ends/axes/steps`` on a ranged shape.

    The output length is *not* monotone in the input length (``starts=0, ends=-3``
    shrinks as the dim grows), so each possible length is evaluated and the extremes
    taken; an unbounded or very large dim gives ``[0, hi]``.
    """
    rank = len(shape)
    axes_l = (
        list(range(len(starts)))
        if axes is None
        else [_norm_axis(a, rank) for a in axes]
    )
    steps_l = [1] * len(starts) if steps is None else list(steps)
    out = list(shape)
    for st, en, ax, sp in zip(starts, ends, axes_l, steps_l):
        d = shape[ax]
        if sp == 0:
            return None
        if d.hi is None or d.hi - d.lo > enumerate_limit:
            out[ax] = Dim(0, d.hi if sp == 1 and d.hi is not None else None)
            continue
        lens = [_slice_len(n, st, en, sp) for n in range(d.lo, d.hi + 1)]
        out[ax] = Dim(min(lens), max(lens))
    return tuple(out)


def _slice_len(n: int, start: int, end: int, step: int) -> int:
    if step > 0:
        s = min(max(start + n if start < 0 else start, 0), n)
        e = min(max(end + n if end < 0 else end, 0), n)
        return max(0, -(-(e - s) // step))
    s = min(max(start + n if start < 0 else start, -1), n - 1)
    e = min(max(end + n if end < 0 else end, -1), n - 1)
    return max(0, -(-(s - e) // -step))


def reduce(shape: Shape, axes: Optional[Sequence[int]], keepdims: bool) -> Shape:
    rank = len(shape)
    red = set(range(rank)) if axes is None else {_norm_axis(a, rank) for a in axes}
    if keepdims:
        return tuple(exact(1) if i in red else d for i, d in enumerate(shape))
    return tuple(d for i, d in enumerate(shape) if i not in red)


def reduced_count(shape: Shape, axes: Optional[Sequence[int]]) -> Dim:
    """How many elements are folded into each output element of a reduction."""
    rank = len(shape)
    red = set(range(rank)) if axes is None else {_norm_axis(a, rank) for a in axes}
    return numel(tuple(d for i, d in enumerate(shape) if i in red))


def matmul(a: Shape, b: Shape) -> Optional[Shape]:
    if len(a) == 0 or len(b) == 0:
        return None
    a2 = (exact(1),) + tuple(a) if len(a) == 1 else tuple(a)
    b2 = tuple(b) + (exact(1),) if len(b) == 1 else tuple(b)
    batch = broadcast2(a2[:-2], b2[:-2])
    if batch is None:
        return None
    out = tuple(batch) + (a2[-2], b2[-1])
    if len(a) == 1:
        out = out[:-2] + out[-1:]
    if len(b) == 1:
        out = out[:-1]
    return out


# --------------------------------------------------------------------------
# Data-dependent ops
# --------------------------------------------------------------------------


def nonzero(
    shape: Shape,
    definitely: Optional[int] = None,
    possibly: Optional[int] = None,
    value_hull: Optional[Tuple[float, float]] = None,
) -> Shape:
    """Output shape of ``NonZero``: ``[rank, N]``.

    ``N`` is the number of non-zero elements. Without value information
    ``N in [0, numel]``. ``definitely``/``possibly`` (counts of elements that are
    certainly / possibly non-zero, from an elementwise value interval of a static
    input) pin it to ``[definitely, possibly]``. A scalar ``value_hull`` sharpens it
    when it excludes zero (every element non-zero: ``N = numel``) or is exactly zero
    (``N = 0``).
    """
    n = numel(shape)
    rank = exact(len(shape))
    if possibly is not None and definitely is not None:
        return (rank, Dim(definitely, possibly))
    if value_hull is not None:
        lo, hi = value_hull
        if lo > 0 or hi < 0:
            return (rank, Dim(n.lo, n.hi))
        if lo == 0 and hi == 0:
            return (rank, exact(0))
    return (rank, Dim(0, n.hi))


def topk(shape: Shape, axis: int, k: Dim) -> Shape:
    """Output shape of ``TopK`` for both outputs; ``k`` is a (possibly ranged) dim."""
    ax = _norm_axis(axis, len(shape))
    d = shape[ax]
    hi = _min_hi(k.hi, d.hi)
    lo = min(k.lo, hi) if hi is not None else k.lo
    out = list(shape)
    out[ax] = Dim(lo, hi)
    return tuple(out)


def compress(
    shape: Shape,
    axis: Optional[int],
    cond_len: Dim,
    definitely: Optional[int] = None,
    possibly: Optional[int] = None,
) -> Shape:
    """``Compress``: the kept length is the number of true condition entries (at most
    ``min(dim, len(condition))``); ``definitely``/``possibly`` sharpen it when the
    condition's values are known."""
    if axis is None:
        d = numel(shape)
    else:
        ax = _norm_axis(axis, len(shape))
        d = shape[ax]
    hi = _min_hi(d.hi, cond_len.hi)
    lo = 0
    if definitely is not None and possibly is not None:
        lo, hi = definitely, _min_hi(hi, possibly)
        if hi is not None and lo > hi:
            lo = hi
    kept = Dim(lo, hi)
    if axis is None:
        return (kept,)
    out = list(shape)
    out[ax] = kept
    return tuple(out)


def unique(shape: Shape, axis: Optional[int]) -> Tuple[Shape, Shape, Shape, Shape]:
    """Shapes of ``Unique``'s outputs ``(Y, indices, inverse_indices, counts)``."""
    if axis is None:
        n = numel(shape)
        nu = Dim(min(1, n.lo), n.hi)
        return ((nu,), (nu,), (Dim(n.lo, n.hi),), (nu,))
    ax = _norm_axis(axis, len(shape))
    d = shape[ax]
    nu = Dim(min(1, d.lo), d.hi)
    y = list(shape)
    y[ax] = nu
    return (tuple(y), (nu,), (Dim(d.lo, d.hi),), (nu,))


def nms(boxes: Shape, scores: Shape, max_per_class: Dim) -> Optional[Shape]:
    """``NonMaxSuppression``: ``[num_selected, 3]`` with ``num_selected`` at most
    ``batches * classes * min(max_output_boxes_per_class, num_boxes)``."""
    if len(boxes) != 3 or len(scores) != 3:
        return None
    b, c, n = boxes[0], scores[1], boxes[1]
    per = Dim(0, _min_hi(max_per_class.hi, n.hi))
    total = b * c * per
    return (Dim(0, total.hi), exact(3))


def range_count(
    start: Tuple[float, float], limit: Tuple[float, float], delta: Tuple[float, float]
) -> Dim:
    """Length of ``Range(start, limit, delta)``: ``max(0, ceil((limit-start)/delta))``."""
    (s0, s1), (l0, l1), (d0, d1) = start, limit, delta
    if not all(np.isfinite(v) for v in (s0, s1, l0, l1, d0, d1)):
        return Dim(0, None)
    if d0 > 0:
        lo = max(0, math.ceil((l0 - s1) / d1))
        hi = max(0, math.ceil((l1 - s0) / d0))
        return Dim(lo, hi)
    if d1 < 0:
        lo = max(0, math.ceil((l1 - s0) / d0))
        hi = max(0, math.ceil((l0 - s1) / d1))
        return Dim(lo, hi)
    return Dim(0, None)  # delta may be 0 or change sign: no useful bound


def constant_of_shape(vec_lo, vec_hi) -> Shape:
    return dims_from_vector(vec_lo, vec_hi)
