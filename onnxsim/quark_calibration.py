"""AMD Quark's histogram calibrators, reimplemented from their observable
behaviour: ``Entropy``, ``Distribution``, ``Percentile`` and
``LayerwisePercentile`` (what ``quark.onnx.CalibMethod`` selects).

They share one histogram *layout* -- which is what makes the scales agree with
Quark's to the bit rather than to binning accuracy:

- **signed** (``absolute=False``; Entropy, Distribution, asymmetric
  Percentile): the first batch builds ``num_bins`` bins over ``[-t, t]``,
  ``t = max(|min|, |max|)`` of that batch; a later batch with a larger ``t'``
  grows the histogram by ``h`` whole bins of the same width on *each* side
  (``h = (t' - t) // stride + 1``), so the bin count -- and hence the entropy
  search space -- depends on the data;
- **absolute** (symmetric Percentile, the default): the first batch builds
  ``num_bins`` bins over ``[min|x|, max|x|]`` of that batch, later batches that
  exceed it append bins of that width on the right. Values below the first
  batch's smallest magnitude are dropped, as in Quark.

float32 data, float32 edges and the same ``numpy.histogram`` calls: the
arithmetic is done in the dtypes Quark uses so a tie or a bin edge falls the
same way.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["QuarkHistogram", "lwp_select"]

_F32 = np.float32


def _as_f32(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).ravel()
    return a if a.dtype in (np.float32, np.float16) else a.astype(np.float32)


class QuarkHistogram:
    """One tensor's histogram in Quark's layout (see the module docstring)."""

    def __init__(self, num_bins: int, absolute: bool):
        self.num_bins = num_bins
        self.absolute = absolute
        self.hist: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.rmin = _F32(0)
        self.rmax = _F32(0)
        self.threshold = _F32(0)  # signed layout only

    # -- collection ------------------------------------------------------------

    def add(self, arr: np.ndarray) -> None:
        data = _as_f32(arr)
        if not data.size:
            return
        lo, hi = np.nanmin(data), np.nanmax(data)
        if self.absolute:
            self._add_absolute(data, lo, hi)
        else:
            self._add_signed(data, lo, hi)

    def _add_absolute(self, data: np.ndarray, lo: Any, hi: Any) -> None:
        orig_dtype = data.dtype
        mag = np.absolute(data).astype(np.float32)
        if self.hist is None or self.edges is None:
            hist, edges = np.histogram(mag, bins=self.num_bins)
            self.hist, self.edges = hist, edges.astype(orig_dtype)
            self.rmin, self.rmax = lo, hi
            return
        edges = self.edges
        top = np.nanmax(mag)
        if top > edges[-1]:
            width = edges[1] - edges[0]
            extra = np.arange(edges[-1] + width, top + width, width)
            edges = np.hstack((edges, extra))
        hist, new_edges = np.histogram(mag, bins=edges)
        hist[: len(self.hist)] += self.hist
        self.hist, self.edges = hist, new_edges.astype(orig_dtype)
        self.rmin, self.rmax = min(self.rmin, lo), max(self.rmax, hi)

    def _add_signed(self, data: np.ndarray, lo: Any, hi: Any) -> None:
        thr = _F32(max(abs(lo), abs(hi)))
        if self.hist is None or self.edges is None:
            hist, edges = np.histogram(data, self.num_bins, range=(-thr, thr))
            self.hist, self.edges = hist, edges
            self.rmin, self.rmax, self.threshold = lo, hi, thr
            return
        old_n, old_thr = len(self.hist), self.threshold
        if thr <= old_thr:
            new, _ = np.histogram(data, old_n, range=(-old_thr, old_thr))
            self.hist = new + self.hist
        elif old_thr == 0:
            hist, edges = np.histogram(data, old_n, range=(-thr, thr))
            self.hist, self.edges, self.threshold = hist + self.hist, edges, thr
        else:
            stride = _F32(2) * old_thr / _F32(old_n)
            half = int((thr - old_thr) // stride + 1)
            n = old_n + 2 * half
            thr = _F32(half) * stride + old_thr
            hist, edges = np.histogram(data, n, range=(-thr, thr))
            hist[half : n - half] += self.hist
            self.hist, self.edges, self.threshold = hist, edges, thr
        self.rmin, self.rmax = min(self.rmin, lo), max(self.rmax, hi)

    # -- ranges ----------------------------------------------------------------

    def _clip(self, lo: Any, hi: Any) -> Tuple[float, float]:
        if lo < self.rmin:
            lo = self.rmin
        if hi > self.rmax:
            hi = self.rmax
        return float(lo), float(hi)

    def percentile_range(
        self, percentile: float, symmetric: bool = True
    ) -> Tuple[float, float]:
        """Quark's ``compute_percentile``: ``(-e, e)`` at the first bin edge
        whose cumulative share reaches ``percentile`` (``symmetric``, on the
        absolute histogram), or the two one-sided cut edges (signed histogram),
        then clipped to the observed ``(min, max)``."""
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        assert self.hist is not None and self.edges is not None
        if symmetric != self.absolute:
            raise ValueError(
                "symmetric percentiles need the absolute histogram (and the "
                "asymmetric ones the signed histogram)"
            )
        cdf = np.cumsum(self.hist / self.hist.sum())
        if symmetric:
            edge = self.edges[np.searchsorted(cdf, percentile / 100.0)]
            return self._clip(-edge, edge)
        cut = (100.0 - percentile) / 200.0
        return self._clip(
            self.edges[np.searchsorted(cdf, cut)],
            self.edges[np.searchsorted(cdf, 1.0 - cut)],
        )

    def distribution_range(self) -> Tuple[float, float]:
        """Quark's ``Distribution``: the histogram's own extent (the observed
        threshold rounded up to a whole bin), not clipped to ``(min, max)``."""
        assert self.edges is not None and not self.absolute
        return float(self.edges.min()), float(self.edges.max())

    def entropy_range(self, num_quantized_bins: int = 128) -> Tuple[float, float]:
        """Quark's ``Entropy``: the symmetric window around the zero bin, at
        least ``num_quantized_bins`` wide, that minimises the KL divergence
        between the clipped histogram (outliers folded into the end bins) and
        its quantization to ``num_quantized_bins`` levels; then clipped to
        ``(min, max)``. With the default 128 bins and 128 quantized bins the
        window can only start as the whole range, so it is the histogram's
        growth over the batches that opens the search."""
        assert self.hist is not None and self.edges is not None and not self.absolute
        hist = self.hist
        n = hist.size
        zero = n // 2
        half_q = num_quantized_bins // 2
        total = hist.sum()
        cum = np.concatenate(([0], np.cumsum(hist)))  # cum[k] = sum(hist[:k])
        best_kl, best = None, (self.edges[0], self.edges[0])
        for i in range(half_q, zero + 1):
            start, end = zero - i, min(zero + i + 1, n)
            kl = _window_kl(
                hist, start, end, cum[start], total - cum[end], num_quantized_bins
            )
            if best_kl is None or kl < best_kl:
                best_kl, best = kl, (self.edges[start], self.edges[end])
        return self._clip(*best)

    # -- layerwise percentile helper ---------------------------------------------

    def centres_and_counts(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.hist is not None and self.edges is not None
        e = self.edges.astype(np.float64)
        return (e[:-1] + e[1:]) / 2, self.hist.astype(np.float64)


def _smooth(p: np.ndarray, eps: float = 1e-4) -> Optional[np.ndarray]:
    """Quark's smoothing: zeros become ``eps``, the non-zeros give back
    ``eps * zeros / nonzeros`` each (float32); ``None`` if that is impossible."""
    is_zero = (p == 0).astype(np.float32)
    nonzero = (p != 0).astype(np.float32)
    n_zero = is_zero.sum()
    n_nonzero = p.size - n_zero
    if not n_nonzero:
        return None
    eps1 = eps * float(n_zero) / float(n_nonzero)
    if eps1 >= 1.0:
        return None
    out = p.astype(np.float32)
    out += eps * is_zero + (-eps1) * nonzero
    if (out <= 0).any():
        return None
    return out


def _kl(p: np.ndarray, q: np.ndarray) -> np.float32:
    """``sum(p * log(p / q))`` of two normalised float32 distributions."""
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q)).astype(np.float32)


def _window_kl(
    hist: np.ndarray,
    start: int,
    end: int,
    left_out: Any,
    right_out: Any,
    num_q: int,
) -> float:
    window = hist[start:end]
    p = window.copy()
    p[0] += left_out
    p[-1] += right_out
    nonzero = (p != 0).astype(np.int64)
    size = window.size
    merged = size // num_q
    main = merged * num_q
    qb = window[:main].reshape(num_q, merged).sum(axis=1)
    qb[-1] += window[main:].sum()
    norm = nonzero[:main].reshape(num_q, merged).sum(axis=1)
    per = np.where(norm > 0, qb // np.maximum(norm, 1), 0)
    q = np.zeros(size, dtype=np.int64)
    q[:main] = np.repeat(per, merged)
    ps, qs = _smooth(p), _smooth(q)
    if ps is None or qs is None:
        return float("inf")
    return float(_kl(ps, qs))


# -- LayerwisePercentile ------------------------------------------------------------

_QRANGES = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
}


def _affine(rmin: float, rmax: float, qmin: int, qmax: int) -> Tuple[float, int]:
    """``(scale, zero_point)`` of Quark's plain MinMax grid, in its convention
    ``r = scale * (q + zp)`` (``zp`` the negated zero point)."""
    lo = np.minimum(_F32(rmin), _F32(0))
    hi = np.maximum(_F32(rmax), _F32(0))
    scale = np.array(np.float64(hi - lo) / np.float64(qmax - qmin))
    if scale < np.finfo(np.float32).tiny:
        return 1.0, 0
    zp = int(np.round(np.int32(qmin) - lo / scale))
    return float(scale.astype(np.float32)), -zp


def lwp_select(
    hist: QuarkHistogram,
    candidates: Sequence[float],
    dtype: str,
    metric: str = "mae",
    symmetric: bool = True,
) -> Tuple[float, float]:
    """Quark's ``LayerwisePercentile``: of the ``candidates`` percentiles, the
    range whose quantize-dequantize error on the histogram (bin centres
    weighted by counts) is smallest -- ``metric`` ``"mae"`` or ``"mse"``,
    ties to the first."""
    if metric not in ("mae", "mse"):
        raise ValueError(f"unknown lwp metric {metric!r}, expected 'mae' or 'mse'")
    qmin, qmax = _QRANGES[dtype]
    centres, counts = hist.centres_and_counts()
    total = counts.sum()
    ranges: List[Tuple[float, float]] = [
        hist.percentile_range(p, symmetric) for p in candidates
    ]
    if total == 0:
        return ranges[0]
    scores = []
    for lo, hi in ranges:
        scale, zp = _affine(lo, hi, qmin, qmax)
        q = np.clip(np.round(centres / scale - zp), qmin, qmax)
        diff = centres - (q + zp) * scale
        err = diff * diff if metric == "mse" else np.abs(diff)
        scores.append(float(np.sum(counts * err) / total))
    return ranges[int(np.argmin(scores))]


def histograms_by_name(
    names: Sequence[str], num_bins: int, absolute: bool
) -> Dict[str, QuarkHistogram]:
    return {n: QuarkHistogram(num_bins, absolute) for n in names}
