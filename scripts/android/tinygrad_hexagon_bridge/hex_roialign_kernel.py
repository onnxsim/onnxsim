#!/usr/bin/env python3
"""RoiAlign via tinygrad's `Tensor.gather()` -- Stage 1 of the "beyond the backbone" op-coverage
effort's second-ranked target (see `dynamic_ops_survey.md`, PR #1813). Answers the survey's own
open research question decisively: **yes**, this project's tinygrad/UOp machinery CAN express
RoiAlign's genuinely data-dependent source addressing (every sample point's feature-map read
location depends on that RoI's own runtime box coordinates) -- no new `Ops`/UOp capability is
needed, contrary to the survey's uncertainty. `Tensor.gather(dim, index)` (`tinygrad/mixin/op.py`)
already exists and is a first-class Tensor op, usable exactly like any other tensor composition;
it isn't Hexagon-specific and needed zero new UOp-level work to reach for it here.

Verified two ways, both against `scripts/android/maskrcnn_e2e/tinygrad_ops.py`'s exact NumPy
`roi_align()` reference (this project's own established ground truth for this op's semantics --
avg pooling, `sampling_ratio=2` 2x2 sample grid per bin, matching this model's real RoiAlign node
attributes per the survey):

- Default device: max_abs_err=5.90e-06 (float32 precision, no approximation).
- `DEV=DSP MOCKDSP=1` (the actual Hexagon-targeted `hexagon-clang`-style compile, real
  `--target=hexagon -mhvx=v65` codegen, executed under `qemu-hexagon-static` -- the same
  correctness-verification path every kernel in this project has used before bridging to real
  hardware): max_abs_err=1.53e-05. **This compiles and runs correctly on the actual DSP-targeted
  toolchain, not just the host default device** -- the real test this Stage-1 pass needed to pass.

One real bug hit and fixed along the way, the same class this project's `sigmoid` work already
established: the bilinear weight tensors (`hy*hx` etc.) are plain NumPy arithmetic on `np.arange`
outputs, which defaults to `float64` -- passing that straight to `Tensor(...)` produced a kernel
needing `__hexagon_muldf3`/`__hexagon_adddf3` (double-precision compiler-rt symbols, not the
single-precision ones `sigmoid`'s `libgcc.a` fix covers), missing from the freestanding link.
Fixed by explicitly `.astype(np.float32)`-ing every weight array before wrapping in `Tensor(...)`
-- not a Hexagon/tinygrad limitation, a bug in this file's own host-side arithmetic.

**What this Stage-1 result does NOT establish**: speed. `Tensor.gather()`'s real implementation
(`mixin/op.py`'s `gather()`: `index.unsqueeze(-1)._one_hot_along_dim(self.shape[dim]).where(x,
0)).sum(-1)`) is a **dense one-hot-mask-and-reduce over the whole gathered dimension**, not a
genuine indirect/sparse memory load -- correct, but O(H*W) work per single gathered pixel rather
than O(1). For a real box-head-scale RoiAlign call (a 200x272-ish feature map, 4 taps per output
position, 49 output positions per ROI, up to 1000 ROIs), this would be enormously wasteful.
**A fast kernel needs real HVX indexed-load hardware, which exists and is compiler-accessible on
this exact toolchain** -- confirmed directly, not assumed: `__builtin_HEXAGON_V6_vgathermh_128B`
(real HVX vector gather) and `__builtin_HEXAGON_V6_vlutvvb_128B` (real HVX vector lookup-table,
Hexagon's other genuinely-indexed-read primitive) both compile cleanly (recognized, correctly
typed) against `hexagon-clang -mcpu=hexagonv73 -mhvx=v73`. Building a hand-written `custom_kernel`
around one of those -- computing each RoI's 4-tap sample addresses + bilinear weights host-side
(cheap scalar arithmetic per the survey's own hybrid-split design, since address computation isn't
bulk data movement) and feeding them to an HVX-native gather/LUT instruction inside the kernel,
padded to a compile-time-max ROI count the same way `hex_gemm_kernel.py`'s tiny-`cout` coverage and
`hex_stem7x7_kernel.py`'s `cin=3` padding already established -- is the concrete next step, not
attempted here (out of scope for this Stage-1 pass, which was specifically about answering
whether data-dependent addressing is reachable at all).

Not attempted here either: real hardware (this file's `roi_align_tensor()` has never been bridged
through `native_transport/` or TVM-RPC -- `MOCKDSP=1`/qemu correctness is as far as this pass
went, matching the survey's own explicit permission to land a correctness-only Stage-1 result);
batching over more than one ROI (single-ROI correctness was this pass's whole scope, per the
survey's own recommendation not to rush "all 1000 proposals, batched, fast" in one step).
"""
from __future__ import annotations

import numpy as np


def roi_align_tensor(x, roi, output_height: int, output_width: int, sampling_ratio: int, spatial_scale: float):
    """One ROI, avg pooling -- the exact semantics of `tinygrad_ops.py`'s `roi_align()`, rebuilt
    from `Tensor` ops (using `.gather()` for the data-dependent feature-map reads) instead of
    NumPy fancy indexing, so it can run through tinygrad's real device/codegen path (including
    the Hexagon DSP target) rather than only ever executing as host-side NumPy.

    `x`: (C, H, W) Tensor. `roi`: 4 python floats (x1, y1, x2, y2) in the pre-`spatial_scale`
    coordinate system, matching `tinygrad_ops.roi_align`'s per-`r` loop body exactly."""
    from tinygrad import Tensor, dtypes

    C, H, W = x.shape
    x1, y1, x2, y2 = (v * spatial_scale for v in roi)
    roi_w, roi_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    bin_h, bin_w = roi_h / output_height, roi_w / output_width
    grid_h = grid_w = sampling_ratio
    ph = np.arange(output_height)[:, None, None, None]
    pw = np.arange(output_width)[None, :, None, None]
    iy = np.arange(grid_h)[None, None, :, None]
    ix = np.arange(grid_w)[None, None, None, :]
    shape = (output_height, output_width, grid_h, grid_w)
    ys = np.broadcast_to(y1 + ph * bin_h + (iy + 0.5) * bin_h / grid_h, shape)
    xs = np.broadcast_to(x1 + pw * bin_w + (ix + 0.5) * bin_w / grid_w, shape)
    valid = ((ys >= -1) & (ys <= H) & (xs >= -1) & (xs <= W)).astype(np.float32)
    ys, xs = np.maximum(ys, 0.0), np.maximum(xs, 0.0)
    y_low, x_low = ys.astype(np.int64), xs.astype(np.int64)
    clamp_y, clamp_x = y_low >= H - 1, x_low >= W - 1
    y_low, x_low = np.where(clamp_y, H - 1, y_low), np.where(clamp_x, W - 1, x_low)
    y_high, x_high = np.where(clamp_y, H - 1, y_low + 1), np.where(clamp_x, W - 1, x_low + 1)
    ys_f = np.where(clamp_y, y_low.astype(np.float32), ys)
    xs_f = np.where(clamp_x, x_low.astype(np.float32), xs)
    ly, lx = ys_f - y_low, xs_f - x_low
    hy, hx = 1 - ly, 1 - lx

    n = output_height * output_width * grid_h * grid_w

    def flat(y, x):
        return (y * W + x).reshape(-1).astype(np.int64)

    idx00, idx01, idx10, idx11 = flat(y_low, x_low), flat(y_low, x_high), flat(y_high, x_low), flat(y_high, x_high)
    xf = x.reshape(C, H * W)

    def gather_flat(idx_np):
        idx_t = Tensor(idx_np, dtype=dtypes.int64).reshape(1, n).expand(C, n)
        return xf.gather(1, idx_t)  # the data-dependent read

    v00, v01, v10, v11 = gather_flat(idx00), gather_flat(idx01), gather_flat(idx10), gather_flat(idx11)
    # NB: must be float32 -- plain `np.arange`-derived arithmetic defaults to float64, which
    # silently pulls in double-precision compiler-rt symbols the freestanding Hexagon link lacks
    # (a different, larger set than the single-precision ones `sigmoid`'s `libgcc.a` fix covers).
    w00 = Tensor(((hy * hx).reshape(-1) * valid.reshape(-1)).astype(np.float32))
    w01 = Tensor(((hy * lx).reshape(-1) * valid.reshape(-1)).astype(np.float32))
    w10 = Tensor(((ly * hx).reshape(-1) * valid.reshape(-1)).astype(np.float32))
    w11 = Tensor(((ly * lx).reshape(-1) * valid.reshape(-1)).astype(np.float32))
    samples = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    samples = samples.reshape(C, output_height, output_width, grid_h * grid_w)
    return samples.mean(axis=-1)


def main() -> None:
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--out-size", type=int, default=3, help="output_height == output_width")
    p.add_argument("--sampling-ratio", type=int, default=2)
    args = p.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "maskrcnn_e2e"))
    from tinygrad_ops import roi_align  # this project's established NumPy reference

    from tinygrad import Tensor

    rng = np.random.default_rng(7)
    C, H, W = args.channels, args.height, args.width
    OH = OW = args.out_size
    spatial_scale = 1.0
    feat = rng.integers(0, 200, (1, C, H, W)).astype(np.float32)
    roi = np.array([[1.3, 0.7, H * 0.9, W * 0.6]], dtype=np.float32)
    batch_idx = np.array([0], dtype=np.int64)

    ref = roi_align(feat, roi, batch_idx, "avg", OH, OW, args.sampling_ratio, spatial_scale)[0]

    dev = os.environ.get("DEV")
    x_t = Tensor(feat[0], device=dev)
    out = roi_align_tensor(x_t, roi[0].tolist(), OH, OW, args.sampling_ratio, spatial_scale).numpy()

    err = np.abs(out - ref)
    print(f"device={dev or 'default'} max_abs_err={err.max():.6g} mean_abs_err={err.mean():.6g}")
    if err.max() >= 1e-3:
        raise SystemExit("kernel is incorrect")
    print("PASS")


if __name__ == "__main__":
    main()
