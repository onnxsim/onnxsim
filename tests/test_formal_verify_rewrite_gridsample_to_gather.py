"""Formal check for RewriteGridSampleToGather (opt-in; onnxsim's own
``onnxsim/passes/rewrite_gridsample_to_gather.h``), restricted here to
``mode="nearest", padding_mode="zeros"`` -- the tractable case; bilinear
interpolation and the reflection/border padding modes are not covered (see
scoping note below).

For nearest+zeros, the rewrite (after denormalizing the grid's [-1, 1]
coordinates to pixel coordinates ``coordx``/``coordy`` -- an
``align_corners``-dependent arithmetic step this proof does not re-derive,
scoped out the same way test_formal_verify_fuse_pad_into_conv.py scopes out
output-length arithmetic) computes, per output pixel::

    xr = Round(coordx); yr = Round(coordy)          # rewrite_gridsample_to_gather.h:552-553
    ix = Cast(Clip(xr, 0, W-1)); iy = Cast(Clip(yr, 0, H-1))   # BuildIndex, :406-407
    gathered = X[iy, ix]                             # GatherPixel, always in-bounds
    valid = (0 <= xr <= W-1) and (0 <= yr <= H-1)     # InRange, :412-413 -- tested
                                                       # against the *rounded* xr/yr,
                                                       # not the pre-round coordinate
    result = gathered * valid

Soundness: this is exactly the same zero-fill boundary rule as
test_formal_verify_fuse_pad_into_conv.py's own ``_read`` helper (clip for
the lookup, but test validity separately), just in two dimensions and with
``round`` in place of a pad-shift. That the validity test uses the
*rounded* coordinate ``xr``/``yr`` -- not the raw pre-round coordinate --
is exactly what makes clip-then-mask reproduce true zero-fill sampling:
had it tested the raw coordinate instead, a coordinate like -0.3 (which
rounds to the valid index 0) would be masked to zero even though its
nearest pixel is in-bounds. Modeling ``round`` the same way as
test_formal_verify_quantize_round_trip.py (some integer within 0.5 of its
argument, not one specific tie rule) and reusing
test_formal_verify_fuse_pad_into_conv.py's zero-fill-vs-clip proof
structure turns "clip-then-mask equals true zero-fill nearest sampling"
into a direct case-split Z3 can check.
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser


def test_rewrite_gridsample_to_gather_nearest_zeros_is_sound():
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    H, W = z3.Ints("H W")
    coordx, coordy = z3.Reals("coordx coordy")
    nx, ny = z3.Ints("nx ny")  # Round(coordx), Round(coordy)
    half = z3.RealVal(1) / 2

    domain = z3.And(
        H > 0,
        W > 0,
        nx - coordx <= half,
        coordx - nx <= half,
        ny - coordy <= half,
        coordy - ny <= half,
    )

    def clip(v, hi):
        return z3.If(v < 0, 0, z3.If(v > hi, hi, v))

    ix = clip(nx, W - 1)
    iy = clip(ny, H - 1)
    valid = z3.And(nx >= 0, nx <= W - 1, ny >= 0, ny <= H - 1)
    result = z3.If(valid, X(iy, ix), z3.RealVal(0))

    # The zero-fill target: read X at the rounded index directly, zero if
    # either axis is out of [0, dim).
    zero_fill = z3.If(
        z3.And(ny >= 0, ny < H, nx >= 0, nx < W), X(ny, nx), z3.RealVal(0)
    )

    prove(z3.Implies(domain, result == zero_fill))


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def test_rewrite_gridsample_to_gather_pass_matches():
    # X is 4x4; grid mixes in-range normalized coordinates (0.0 -> the
    # image center) with clearly out-of-range ones (+-3.0, well outside
    # [-1, 1]) so the differential check (via simplify_isolated_extra's own
    # check_n, which randomizes X) exercises both the plain-gather branch
    # and the zero-fill branch against the real compiled pass.
    grid = np.array(
        [[[[0.0, 0.0], [3.0, 0.0]], [[-3.0, -3.0], [0.5, -0.5]]]], dtype=np.float32
    )
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 16]
        >
        g (float[1,1,4,4] X) => (float[1,1,2,2] Y)
        {
          Y = GridSample<mode = "nearest", padding_mode = "zeros", align_corners = 0>(X, grid)
        }
        """
    )
    model.graph.initializer.append(_f32(grid, "grid"))
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gridsample_to_gather")
    assert ops["GridSample"] == 0
