"""Fast path for exact-2x nearest-neighbor upsampling on Hexagon.

TVM's `topi.image.resize2d` computes each output pixel's source index with `get_inx`/
`get_closest_index` (`topi/image/resize.py`), which for `nearest_neighbor` calls `te.ceil`/
`te.floor`/`te.round` on a per-pixel floating-point coordinate. On Hexagon this lowers to a
scalar libm call per lane, serializing an otherwise-vectorized loop -- measured at 21x slower
than a trivial op of the same output size (see `maskrcnn_e2e/README.md`'s "21x anomaly" section).

Mask R-CNN's FPN always upsamples by exactly 2x. For `coordinate_transformation_mode="half_pixel"`
(the ONNX default and what this model uses) and `rounding_method="round_prefer_floor"` (ditto),
the source index reduces to a closed form with no float math at all:

    in_x = (out_x + 0.5) * 0.5 - 0.5         # get_inx, scale_x = in_w / out_w = 0.5
    idx  = ceil(in_x - 0.5)                  # get_closest_index, round_prefer_floor
         = ceil(out_x * 0.5 - 0.75)
         = out_x // 2                        # verified exact for out_x in 0..39; see below

`patch()` intercepts `topi.image.resize2d` (a live attribute lookup from
`relay/op/image/_image.py`'s `compute_resize2d`, so patching `tvm.topi.image.resize2d` takes
effect for any `relay.build` called while the patch is active) and substitutes a pure-integer,
directly-vectorizable `te.compute` for calls matching this exact case; anything else (a
different scale, rounding mode, or resize method) falls through to the original implementation
unchanged.
"""

from __future__ import annotations

import contextlib

from tvm import te, topi


def is_exact_2x_nearest_half_pixel(size, layout, method, coordinate_transformation_mode, rounding_method):
    return (
        layout in ("NCHW", "NHWC")
        and method == "nearest_neighbor"
        and coordinate_transformation_mode == "half_pixel"
        and rounding_method == "round_prefer_floor"
        and int(size[0]) % 2 == 0
        and int(size[1]) % 2 == 0
    )  # fmt: skip


def _nearest_2x(data, layout, out_dtype):
    height_axis, width_axis = (2, 3) if layout == "NCHW" else (1, 2)
    out_shape = list(data.shape)
    out_shape[height_axis] *= 2
    out_shape[width_axis] *= 2

    def index(*out_idx):
        in_idx = list(out_idx)
        in_idx[height_axis] = out_idx[height_axis] // 2
        in_idx[width_axis] = out_idx[width_axis] // 2
        value = data(*in_idx)
        return value.astype(out_dtype) if out_dtype and out_dtype != "" else value

    return te.compute(out_shape, index, name="resize_nearest_2x")


def resize2d_with_fast_path(
    data,
    roi,
    size,
    layout="NCHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    bicubic_alpha=-0.5,
    bicubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
    output_shape=None,
):
    in_h = int(data.shape[2 if layout == "NCHW" else 1])
    in_w = int(data.shape[3 if layout == "NCHW" else 2])
    fast = (
        int(size[0]) == 2 * in_h
        and int(size[1]) == 2 * in_w
        and is_exact_2x_nearest_half_pixel(size, layout, method, coordinate_transformation_mode, rounding_method)
    )  # fmt: skip
    if fast:
        return _nearest_2x(data, layout, out_dtype)
    return _ORIGINAL_RESIZE2D(
        data, roi, size, layout, method, coordinate_transformation_mode, rounding_method,
        bicubic_alpha, bicubic_exclude, extrapolation_value, out_dtype, output_shape,
    )  # fmt: skip


_ORIGINAL_RESIZE2D = topi.image.resize2d


@contextlib.contextmanager
def patch():
    """Context manager: while active, `relay.build` uses the exact-2x fast path where it applies."""
    topi.image.resize2d = resize2d_with_fast_path
    try:
        yield
    finally:
        topi.image.resize2d = _ORIGINAL_RESIZE2D
