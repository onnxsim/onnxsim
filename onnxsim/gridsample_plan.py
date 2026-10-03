"""The host-side plan for a ``GridSample``: which cells to tap and how to weight them.

None of the hand-written exporters has a native grid-sample kernel -- Core ML's
MIL has no grid-sample op at all, and TFLite's converter has no ONNX-to-TFLite
translation for one -- so each lowers the op itself as a handful of corner
``gather``/``gather_nd`` taps plus elementwise weighting. That means the
*arithmetic* (mapping normalized grid coordinates to source pixels, choosing the
neighbouring corners, and weighting them) is pure numpy and identical everywhere,
while only the emission differs: one target builds TensorFlow ops, another builds
MIL ops.

This module holds that shared arithmetic. ``grid_sample_plan`` reduces a
``(x_shape, grid_shape, grid, mode, padding_mode, align_corners)`` tuple to the
list of taps, each with its own gather indices, gather weight and validity mask;
every exporter lowers that plan with its own ops. Keeping one copy is what makes
the Core ML and TFLite lowerings equivalent to each other (and to ONNX's
reference implementation, ``onnx/reference/ops/op_grid_sample.py``) instead of
two lookalike reimplementations that can drift apart.

Everything here is computed on the host because the exporters' target ops carry
static shapes: the gather indices and the corner weights are compile-time
constants of the emitted graph, not runtime tensors.
"""

from __future__ import annotations

import itertools
from typing import List, NamedTuple, Sequence

import numpy as np

__all__ = ["grid_sample_plan", "GridSampleTap"]


class GridSampleTap(NamedTuple):
    """One corner tap of a grid sample.

    Attributes:
        indices: ``(N, flat, 1 + xd)`` int32 gather indices -- the batch index
            stacked in front of one index per spatial axis, i.e. the layout
            ``gather_nd`` consumes on a channel-last view of ``x``.
        gather_weight: ``(N, flat)`` float32 bilinear weight of this corner.
        valid: ``(N, flat)`` bool, False for corners that fall outside ``x``
            under ``padding_mode="zeros"`` (they still contribute a clamped
            gather, which the mask then zeroes out).
    """

    indices: np.ndarray
    gather_weight: np.ndarray
    valid: np.ndarray


def _source_coordinates(
    grid: np.ndarray, dims: Sequence[int], xd: int, align_corners: bool
) -> List[np.ndarray]:
    """Normalized ``[-1, 1]`` grid coordinates as source-space floats.

    The grid's last axis is ``(x, y[, z])``, i.e. reversed with respect to the
    tensor layout, so spatial axis ``j`` reads coordinate ``xd - 1 - j``. Output
    axis order is ``(N, *out)`` per spatial axis.
    """
    scoord = []
    for j in range(xd):
        c = grid[..., xd - 1 - j]
        dim = dims[j]
        if align_corners:
            scoord.append((c + 1) / 2 * (dim - 1))
        else:
            scoord.append(((c + 1) * dim - 1) / 2)
    return scoord


def _build_tap(
    corners_flat: Sequence[np.ndarray],
    dims: Sequence[int],
    xd: int,
    n: int,
    flat: int,
    padding_mode: str,
    gather_weight: np.ndarray,
    valid: np.ndarray,
) -> GridSampleTap:
    """Build one tap's gather indices, weight and mask.

    ``corners_flat`` are the integer corner coordinates already flattened to the
    ``(N, flat)`` row layout every emitted constant uses.
    """
    idx = np.zeros((n, flat, xd + 1), dtype=np.int32)
    idx[..., 0] = np.broadcast_to(
        np.arange(n, dtype=np.int32).reshape(n, 1), (n, flat)
    )
    if padding_mode == "zeros":
        # Validity is decided by the *raw* coordinate; the gather below reads a
        # clamped one, so an out-of-range corner contributes a real (clamped)
        # value that the exporter multiplies by this zero mask.
        for ix, dim in zip(corners_flat, dims):
            valid = valid & (ix >= 0) & (ix < dim)
    for j, dim in enumerate(dims):
        idx[..., j + 1] = np.clip(corners_flat[j], 0, dim - 1).astype(np.int32)
    return GridSampleTap(idx, gather_weight, valid)


def grid_sample_plan(
    x_shape: Sequence[int],
    grid_shape: Sequence[int],
    grid: np.ndarray,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> List[GridSampleTap]:
    """Reduce a ``GridSample`` to its list of corner taps.

    ``x_shape`` is ``(N, C, *spatial)`` and ``grid_shape`` ``(N, *out, d)``;
    ``grid`` is the grid's value (shape ``grid_shape``), needed because the
    sampling coordinates are compile-time constants of the emitted graph.

    Returns 4 taps for 2-D bilinear, 8 for 3-D trilinear, or 1 for ``nearest``
    (whatever the spatial rank), each carrying that corner's gather indices,
    bilinear weight and out-of-range mask.
    """
    x_shape = [int(d) for d in x_shape]
    grid_shape = [int(d) for d in grid_shape]
    xd = len(x_shape) - 2
    if xd not in (2, 3):
        raise ValueError(f"GridSample supports 2-D/3-D only, got rank {len(x_shape)}")
    if grid_shape[-1] != xd or len(grid_shape) != xd + 2:
        raise ValueError(
            f"GridSample grid shape {grid_shape} does not match "
            f"{xd}-D input of shape {x_shape}"
        )
    if padding_mode not in ("zeros", "border"):
        raise ValueError(f"unsupported GridSample padding_mode {padding_mode!r}")
    if mode not in ("bilinear", "linear", "nearest"):
        raise ValueError(f"unsupported GridSample mode {mode!r}")

    n = x_shape[0]
    dims = x_shape[2:]
    out_shape = grid_shape[1:-1]
    flat = int(np.prod(out_shape)) if out_shape else 1
    grid = np.asarray(grid, dtype=np.float32).reshape([n] + out_shape + [xd])
    scoord = _source_coordinates(grid, dims, xd, align_corners)

    bilinear = mode in ("bilinear", "linear")
    if bilinear:
        base = [np.floor(c) for c in scoord]
        frac = [c - b for c, b in zip(scoord, base)]
        combos = list(itertools.product([0, 1], repeat=xd))
    else:
        # nearest: a single tap at the rounded coordinate (ties-to-even, per ONNX).
        base = [np.rint(c) for c in scoord]
        frac = None
        combos = [()]

    plan: List[GridSampleTap] = []
    for combo in combos:
        # `combo` is empty for nearest, where `base` already holds the single
        # (rounded) corner and there is nothing to add to it.
        corners = base if not combo else [b + c for b, c in zip(base, combo)]
        corners_flat = [c.reshape(n, flat) for c in corners]
        # A single tap (nearest) carries no bilinear weighting; a corner of a
        # bilinear tap weighs the product of, per axis, `frac` (upper corner) or
        # `1 - frac` (lower corner).
        weight = np.ones((n, flat), dtype=np.float32)
        if bilinear:
            for j in range(xd):
                weight = weight * (
                    frac[j] if combo[j] else 1 - frac[j]
                ).reshape(n, flat)
        valid = np.ones((n, flat), dtype=bool)
        plan.append(
            _build_tap(corners_flat, dims, xd, n, flat, padding_mode, weight, valid)
        )
    return plan