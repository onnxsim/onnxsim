"""Formal check for RewriteDeformConvToGather (opt-in; onnxsim's own
``onnxsim/passes/rewrite_deform_conv_to_gather.h``).

Unlike almost every other file in this suite, this is **not** a
numeric-error-bound pass (there is no quantization, no rounding, no
approximation anywhere in this rewrite -- it is pure float32 arithmetic
restructuring of one custom op into standard ONNX ops) and it is **not** a
node-count-reduction pass either: one ``MMCVDeformConv2d``/
``MMCVModulatedDeformConv2d`` node explodes into dozens-to-low-hundreds of
standard nodes (see the pass header's own comment). The right formal claim
here is therefore *algebraic equivalence* of the modulated-deformable-
convolution-v2 formula (Zhu et al.), decomposed into small, honestly-scoped,
separately-stated lemmas about the two genuinely interesting pieces of
algebra the rewrite performs -- **not** one monolithic symbolic proof of the
whole generated subgraph, which would be neither tractable nor informative.

The formula, restated from the pass header (``groups == 1`` throughout, its
only supported case)::

    out[n, cout, ho, wo] =
        bias[cout] +
        sum over (cin_local, i, j) of
            weight[cout, cin_local, i, j] *
            mask[n, dg*kh*kw + k, ho, wo] *              # 1 if unmodulated
            bilinear_zero_pad(X[n, cin_global], y, x)

    y = ho*sh - ph + i*dh + offset[n, dg*2*kh*kw + 2*k,     ho, wo]
    x = wo*sw - pw + j*dw + offset[n, dg*2*kh*kw + 2*k + 1, ho, wo]

with ``bilinear_zero_pad`` the standard 4-corner bilinear interpolation,
zero-padded outside ``[0, H-1]``/``[0, W-1]`` -- exactly GridSample's
``padding_mode="zeros"`` rule (see
``test_formal_verify_rewrite_gridsample_to_gather.py``, whose nearest-mode
version of this same "clamp the index for the gather, but test *validity*
against the raw, unclamped coordinate, and multiply it in as a 0/1 mask"
trick this pass reuses -- generalized here from one nearest-neighbour tap to
four bilinear corners). Confirmed against the code
(``rewrite_deform_conv_to_gather.h:439-470`` -- ``ClampIndex``, ``InRange``,
``GatherPixel``, the ``corner`` lambda in ``runTransform``): each corner's
*validity* is checked on the raw (pre-clamp) floor/floor+1 coordinate, the
*index* fed to ``GatherND`` is separately clamped into range so the gather
itself never reads out of bounds, and the corner's weight is multiplied by
``Cast<float>(valid_x AND valid_y)`` -- i.e. an invalid corner is clamped to
some (arbitrary, in-bounds) pixel and then zeroed out by the mask multiply,
never actually contributing whatever value clamping happened to read.

Two things are deliberately **not** attempted here:

* A symbolic proof of the whole generated subgraph (Shape/Range/Gather
  chains for Hout/Wout/base coordinates, the per-deform-group Slice, the
  Concat/Transpose/Reshape "unfold" that lines sampled taps up with
  weight's own flattened layout, ...). That plumbing is exercised instead by
  the differential tests below (and, more exhaustively for
  groups/deform_groups/stride/padding/dilation/dynamic-shape variety, by
  the pre-existing ``tests/test_deform_conv_to_gather.py``, which this file
  does not duplicate).
* Bounded-error reasoning of any kind: since this pass introduces no
  rounding, the accumulation lemma below is an *exact* real-arithmetic
  equality, not a tolerance bound -- the "direct-error-variable-free exact
  equality idiom" this suite otherwise reserves for non-quantization passes.
  (The differential tests still compare float32 execution with a small
  numeric tolerance, because float32 op-by-op execution -- Floor/Cast/Clip
  chains, MatMul reduction order -- is not bit-exact with a float64 NumPy
  reference; that is an execution-precision fact, not evidence the rewrite's
  *algebra* is approximate.)
"""

import numpy as np
import onnx
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None


# --------------------------------------------------------------------------- #
# Part 1: Z3 lemmas about the bilinear-sampling formula.
# --------------------------------------------------------------------------- #


def test_bilinear_weights_sum_to_one():
    """(1-fy)(1-fx) + (1-fy)fx + fy(1-fx) + fy*fx == 1 for any fractional
    offset (fy, fx) in [0,1) x [0,1) -- the weights the pass computes as
    ``wx0/wx1/wy0/wy1`` (``Sub(one, wx1)`` etc., runTransform lines 731-734)
    always redistribute the corner's full mass, never gain or lose any."""
    fy, fx = z3.Reals("fy fx")
    domain = z3.And(fy >= 0, fy < 1, fx >= 0, fx < 1)
    wx1, wx0 = fx, 1 - fx
    wy1, wy0 = fy, 1 - fy
    total = wx0 * wy0 + wx1 * wy0 + wx0 * wy1 + wx1 * wy1
    prove(z3.Implies(domain, total == 1))


def _graph_sampled(X, y0, x0, wx0, wx1, wy0, wy1, H, W):
    """The exact Z3 transcription of runTransform's per-tap sampling
    formula (lines 727-760): given the floor coordinates ``y0``/``x0`` (Z3
    Ints -- what ``Floor`` produces, always integral) and the four corner
    weights (Z3 Reals), builds the masked-and-clamped 4-corner sum in the
    same grouping the generated graph uses
    (``Add(Add(c00, c10), Add(c01, c11))``).

    ``X`` is an uninterpreted ``Int, Int -> Real`` function standing in for
    one input channel -- full generality, since nothing in this formula
    depends on the channel's actual pixel values.
    """
    y1, x1 = y0 + 1, x0 + 1

    def clip(v, hi):
        return z3.If(v < 0, 0, z3.If(v > hi, hi, v))

    def valid(v, hi):
        return z3.And(v >= 0, v <= hi)

    def corner(iy, ix, wy, wx):
        m = z3.If(
            z3.And(valid(iy, H - 1), valid(ix, W - 1)), z3.RealVal(1), z3.RealVal(0)
        )
        gathered = X(clip(iy, H - 1), clip(ix, W - 1))
        return gathered * wx * wy * m

    c00 = corner(y0, x0, wy0, wx0)
    c10 = corner(y0, x1, wy0, wx1)
    c01 = corner(y1, x0, wy1, wx0)
    c11 = corner(y1, x1, wy1, wx1)
    return (c00 + c10) + (c01 + c11)


def test_bilinear_exact_at_integer_grid_point():
    """fy=0, fx=0 (the sample point lands exactly on grid point (y0, x0)):
    bilinear(feat, y0, x0) == feat[y0, x0] exactly, when (y0, x0) is
    in-bounds. wx1=wy1=0 makes every corner but c00 vanish regardless of its
    own validity; c00's own mask is 1 since (y0, x0) is in-range by
    assumption."""
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    H, W = z3.Ints("H W")
    y0, x0 = z3.Ints("y0 x0")
    domain = z3.And(H > 0, W > 0, y0 >= 0, y0 <= H - 1, x0 >= 0, x0 <= W - 1)

    sampled = _graph_sampled(
        X, y0, x0, z3.RealVal(1), z3.RealVal(0), z3.RealVal(1), z3.RealVal(0), H, W
    )
    prove(z3.Implies(domain, sampled == X(y0, x0)))


def test_invalid_corner_contributes_exactly_zero():
    """A corner whose (y, x) is out-of-range contributes exactly 0 to the
    sum, *no matter what pixel value clamping happens to read there* -- this
    is the crux of the "clamp the index (so GatherND never actually reads
    out of bounds), but gate its contribution with a separately-computed
    validity mask" design: the clamped-to-some-real-pixel gathered value is
    multiplied by a 0 mask, not used. Universally quantifying the gathered
    value itself (via an uninterpreted ``X``, rather than assuming it is 0)
    is exactly what makes this check meaningful."""
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    H, W = z3.Ints("H W")
    iy, ix = z3.Ints("iy ix")
    wy, wx = z3.Reals("wy wx")
    domain = z3.And(H > 0, W > 0, z3.Or(iy < 0, iy > H - 1, ix < 0, ix > W - 1))

    def clip(v, hi):
        return z3.If(v < 0, 0, z3.If(v > hi, hi, v))

    def valid(v, hi):
        return z3.And(v >= 0, v <= hi)

    m = z3.If(z3.And(valid(iy, H - 1), valid(ix, W - 1)), z3.RealVal(1), z3.RealVal(0))
    corner = X(clip(iy, H - 1), clip(ix, W - 1)) * wx * wy * m
    prove(z3.Implies(domain, corner == 0))


def test_fully_in_range_reduces_to_plain_four_corner_formula():
    """When all four corners are in-bounds (an interior sample point), the
    masked-and-clamped formula the graph computes reduces exactly to the
    textbook, unmasked 4-corner bilinear formula reading real pixel values
    -- i.e. the zero-padding machinery has *no effect* away from the
    boundary, for arbitrary corner weights (this does not depend on the
    weights summing to 1; that is the separate, already-proven Lemma
    above)."""
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    H, W = z3.Ints("H W")
    y0, x0 = z3.Ints("y0 x0")
    wx0, wx1, wy0, wy1 = z3.Reals("wx0 wx1 wy0 wy1")
    domain = z3.And(H > 0, W > 0, y0 >= 0, y0 + 1 <= H - 1, x0 >= 0, x0 + 1 <= W - 1)

    sampled = _graph_sampled(X, y0, x0, wx0, wx1, wy0, wy1, H, W)
    plain = (
        X(y0, x0) * wx0 * wy0
        + X(y0, x0 + 1) * wx1 * wy0
        + X(y0 + 1, x0) * wx0 * wy1
        + X(y0 + 1, x0 + 1) * wx1 * wy1
    )
    prove(z3.Implies(domain, sampled == plain))


def test_kernel_tap_accumulation_matches_mathematical_sum():
    """The per-output-pixel accumulation (bias + sum over taps of
    weight*mask*sampled) is a straightforward weighted sum -- no rounding or
    reordering hazard the way a quantized pass's accumulation would have, so
    an *exact* equality is the right (and easy) claim, unlike this suite's
    quantization files' bounded-error ones.

    Modeled here for a concrete 1-channel/group, 2x2-kernel tap (k=0..3, as
    ``rewrite_deform_conv_to_gather.h``'s per-tap loop over ``(i,j)`` would
    unroll it, single input channel so there is no cin_local sum to worry
    about -- that is covered separately below): the per-tap sampled values
    are left as opaque symbolic reals (each already proven, by the lemmas
    above, to equal the true masked bilinear sample at that tap's
    coordinate), and this lemma checks only that combining them --
    multiplying each by its tap's ``weight``/``mask`` and summing, plus
    ``bias`` -- reproduces the header's own closed-form sum term for term,
    with none dropped, double-counted, or mis-signed."""
    s0, s1, s2, s3 = z3.Reals("s0 s1 s2 s3")  # per-tap bilinear samples
    w0, w1, w2, w3 = z3.Reals("w0 w1 w2 w3")  # weight[cout, 0, i, j]
    m0, m1, m2, m3 = z3.Reals("m0 m1 m2 m3")  # mask[dg, k] (1 if unmodulated)
    bias = z3.Real("bias")

    # What runTransform builds: MatMul's implicit dot-product sum (folded
    # here as a plain sum -- Real addition is associative/commutative, so no
    # particular Add-tree shape changes the *value*, only float32 rounding
    # would, which is out of scope per this file's docstring) over the
    # already-masked per-tap samples, then a bias add.
    masked = [s0 * m0, s1 * m1, s2 * m2, s3 * m3]
    graph_result = bias + sum(w * s for w, s in zip([w0, w1, w2, w3], masked))

    # The header's own closed form.
    target = bias + w0 * m0 * s0 + w1 * m1 * s1 + w2 * m2 * s2 + w3 * m3 * s3

    prove(graph_result == target)


def test_kernel_tap_accumulation_flatten_matches_double_sum():
    """The real risk in "flatten taps, reshape, MatMul" (runTransform steps
    5-6: stack the ``kh*kw`` per-tap ``(..., Cin_dg)`` tensors, transpose,
    and reshape to ``(..., Cin_dg*kh*kw)`` in *row-major*
    ``cin_local`-major, tap-minor`` order, to match how ``weight``'s own
    ``(Cout, Cin, kh, kw) -> (Cout, Cin*kh*kw)`` reshape flattens) is a
    channel/tap mixup: if the two flattens didn't use the same
    (outer, inner) convention, MatMul's dot product would pair the wrong
    weight with the wrong sample.

    Checked here for a concrete Cin_dg=2, kh*kw=2 (4 flattened positions):
    with ``W``/``S`` modeled as uninterpreted ``(cin_local, k) -> Real``
    functions and the flatten index built the same way the pass's Reshape
    does (``m = cin_local*K + k``), summing over the flat index ``m`` in
    ``[0, Cin_dg*kh*kw)`` gives exactly the same total as the nested
    mathematical double sum over ``(cin_local, k)`` -- confirming the
    flatten/index arithmetic itself (not just the already-trivial fact that
    real addition can be grouped either way) is a correct bijection between
    the flat MatMul-contraction axis and the ``(cin_local, k)`` pairs it is
    supposed to represent."""
    W = z3.Function("W", z3.IntSort(), z3.IntSort(), z3.RealSort())
    S = z3.Function("S", z3.IntSort(), z3.IntSort(), z3.RealSort())
    K = 2  # kh*kw
    C = 2  # Cin_dg

    def flat_term(m):
        cin_local, k = m // K, m % K
        return W(cin_local, k) * S(cin_local, k)

    flat_sum = sum(flat_term(m) for m in range(C * K))
    double_sum = sum(
        W(cin_local, k) * S(cin_local, k) for cin_local in range(C) for k in range(K)
    )
    prove(flat_sum == double_sum)


# --------------------------------------------------------------------------- #
# Part 2: independent from-scratch NumPy reimplementation of modulated
# deformable convolution v2, built fresh from the pass header's own formula
# (not imported from tests/test_deform_conv_to_gather.py's own reference,
# and deliberately structured differently -- tap-major/deform-group-major
# loop nesting here, versus that file's channel-major nesting -- so a bug
# shared by "the two ways someone might transcribe this formula" is
# unlikely to slip through both).
# --------------------------------------------------------------------------- #


def _bilinear_zero_pad(feat, y, x):
    """4-corner bilinear sample of ``feat`` (H, W) at float coordinates
    ``y``/``x`` (broadcastable arrays), zero outside [0,H-1]/[0,W-1]."""
    H, W = feat.shape
    y0 = np.floor(y)
    x0 = np.floor(x)
    y1, x1 = y0 + 1, x0 + 1
    fy, fx = y - y0, x - x0

    def px(yy, xx):
        in_range = (yy >= 0) & (yy <= H - 1) & (xx >= 0) & (xx <= W - 1)
        yy_c = np.clip(yy, 0, H - 1).astype(np.int64)
        xx_c = np.clip(xx, 0, W - 1).astype(np.int64)
        return np.where(in_range, feat[yy_c, xx_c], 0.0)

    return (
        px(y0, x0) * (1 - fy) * (1 - fx)
        + px(y0, x1) * (1 - fy) * fx
        + px(y1, x0) * fy * (1 - fx)
        + px(y1, x1) * fy * fx
    )


def _independent_deform_conv2d(
    X, offset, weight, mask, bias, stride, padding, dilation, deform_groups
):
    """``groups == 1`` modulated deformable convolution v2, built directly
    from the formula in ``rewrite_deform_conv_to_gather.h``'s header
    comment. ``mask=None`` gives DCNv1 (implicit all-ones mask)."""
    N, Cin, H, W = X.shape
    Cout, Cin_g, kh, kw = weight.shape
    assert Cin_g == Cin, "groups == 1 only"
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    Hout, Wout = offset.shape[2], offset.shape[3]
    Cin_dg = Cin // deform_groups
    khkw = kh * kw

    ho = np.arange(Hout, dtype=np.float64).reshape(-1, 1)
    wo = np.arange(Wout, dtype=np.float64).reshape(1, -1)

    out = np.zeros((N, Cout, Hout, Wout), dtype=np.float64)
    for n in range(N):
        acc = np.zeros((Cout, Hout, Wout), dtype=np.float64)
        for dg in range(deform_groups):
            m_scale = 1.0 if mask is not None else None
            for i in range(kh):
                for j in range(kw):
                    k = i * kw + j
                    dy = offset[n, dg * 2 * khkw + 2 * k].astype(np.float64)
                    dx = offset[n, dg * 2 * khkw + 2 * k + 1].astype(np.float64)
                    y = ho * sh - ph + i * dh + dy
                    x = wo * sw - pw + j * dw + dx
                    m_scale = (
                        mask[n, dg * khkw + k].astype(np.float64)
                        if mask is not None
                        else 1.0
                    )
                    for cin_local in range(Cin_dg):
                        cin = dg * Cin_dg + cin_local
                        sampled = m_scale * _bilinear_zero_pad(
                            X[n, cin].astype(np.float64), y, x
                        )
                        acc += (
                            weight[:, cin, i, j].astype(np.float64).reshape(-1, 1, 1)
                            * sampled
                        )
        if bias is not None:
            acc += bias.astype(np.float64).reshape(-1, 1, 1)
        out[n] = acc
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Part 3: differential/structural tests against the real compiled pass.
#
# MMCVDeformConv2d/MMCVModulatedDeformConv2d have no ONNX Runtime kernel, so
# onnxsim's own random-sample equivalence check (what `simplify_isolated_extra`
# runs internally via its `check_n`) cannot execute the *original* graph --
# there is nothing to compare against. So, mirroring
# tests/test_deform_conv_to_gather.py's approach: `check_n=0` here (skips
# that internal check, which only leaves onnx.checker's structural
# acceptance -- happy with an unrecognized op in a non-standard domain), and
# the *simplified* graph (now pure standard ops) is executed independently
# below and compared against `_independent_deform_conv2d`.
# --------------------------------------------------------------------------- #


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=17, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "mmdeploy": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _run_model(model, feeds):
    if _ort is not None:
        sess = _ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (out,) = sess.run(None, feeds)
        return out
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    (out,) = evaluator.run(None, feeds)
    return out


def _simplify_and_run(model, feeds):
    sim_model, ops = simplify_isolated_extra(
        model, "rewrite_deform_conv_to_gather", check_n=0
    )
    assert ops["MMCVDeformConv2d"] == 0
    assert ops["MMCVModulatedDeformConv2d"] == 0
    assert ops["GatherND"] > 0
    return _run_model(sim_model, feeds), ops


# ----- small (2 in/out channels, 1 group, 3x3 kernel, 4x4 spatial) case,
# matching the dimensions the task itself calls out as tractable, with a
# fixed (deterministic, non-random) offset/mask -- exercising exact-integer
# taps, genuine non-integer (bilinear) taps, and out-of-range taps all at
# once, per output pixel. ----- #

_N, _CIN, _COUT, _H, _W, _KH, _KW = 1, 2, 2, 4, 4, 3, 3
_PAD = (1, 1)


def _fixed_small_tensors(with_mask):
    # Deterministic (index-formula-derived, not RNG-seeded) so results are
    # exactly reproducible and the offset tensor is "fixed" as required for
    # the interpolation to be traceable by hand if desired.
    c, h, w = np.indices((_CIN, _H, _W))
    X = ((c + 1) * 10 + h * 4 + w).astype(np.float32)  # (Cin,H,W) -> add N axis
    X = X[None, ...]

    cout, cin, i, j = np.indices((_COUT, _CIN, _KH, _KW))
    weight = ((cout - cin) + 0.1 * (i * _KW + j)).astype(np.float32)

    bias = np.array([0.5, -0.25], dtype=np.float32)

    off_c, off_h, off_w = np.indices((2 * _KH * _KW, _H, _W))
    # Values in {-2.0, -1.5, ..., +2.0}: half-integer steps (genuine
    # bilinear taps) mixed with whole-integer ones (exact-grid taps), large
    # enough in magnitude that some taps land out of [0, H-1]/[0, W-1] given
    # padding=1.
    offset = ((((off_c * 7 + off_h * 5 + off_w * 3) % 9) - 4) * 0.5).astype(np.float32)
    offset = offset[None, ...]
    assert np.any(offset % 1.0 != 0.0), "fixture must exercise genuine bilinear taps"
    assert np.any((offset > _H - 1) | (offset < -1)), (
        "fixture must exercise the zero-pad boundary"
    )

    mask = None
    if with_mask:
        mask_c, mask_h, mask_w = np.indices((_KH * _KW, _H, _W))
        mask = (0.2 + 0.1 * ((mask_c + mask_h + mask_w) % 8)).astype(np.float32)
        mask = mask[None, ...]

    return X, weight, bias, offset, mask


def _small_model(op_type, with_bias=True):
    modulated = op_type == "MMCVModulatedDeformConv2d"
    X, weight, bias, offset, mask = _fixed_small_tensors(with_mask=modulated)

    inputs_txt = (
        f"float[{_N},{_CIN},{_H},{_W}] X, float[{_N},{2 * _KH * _KW},{_H},{_W}] offset"
    )
    call_inputs = "X, offset"
    initializer = [_f32(weight, "weight")]
    if modulated:
        inputs_txt += f", float[{_N},{_KH * _KW},{_H},{_W}] mask"
        call_inputs += ", mask"
    call_inputs += ", weight"
    if with_bias:
        call_inputs += ", bias"
        initializer.append(_f32(bias, "bias"))

    body = f"""
    agraph ({inputs_txt}) => (float[{_N},{_COUT},{_H},{_W}] Y)
    {{
      Y = mmdeploy.{op_type} <stride=[1,1], padding=[{_PAD[0]},{_PAD[1]}], dilation=[1,1], groups=1, deform_groups=1> ({call_inputs})
    }}
    """
    model = _model(body, initializer=initializer)

    feeds = {"X": X, "offset": offset}
    if modulated:
        feeds["mask"] = mask

    expected = _independent_deform_conv2d(
        X,
        offset,
        weight,
        mask,
        bias if with_bias else None,
        stride=(1, 1),
        padding=_PAD,
        dilation=(1, 1),
        deform_groups=1,
    )
    return model, feeds, expected


def test_dcnv1_matches_independent_reference():
    model, feeds, expected = _small_model("MMCVDeformConv2d")
    actual, _ops = _simplify_and_run(model, feeds)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_dcnv2_matches_independent_reference():
    model, feeds, expected = _small_model("MMCVModulatedDeformConv2d")
    actual, _ops = _simplify_and_run(model, feeds)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


# ----- dedicated hand-verifiable single-output-element case: 1 channel, one
# 2x2-kernel deform-conv, small enough that output[0,0,0,0] can be derived
# term by term with pencil-and-paper arithmetic (worked out in comments
# below), independent of both `_independent_deform_conv2d` above and the
# pass itself -- catching a bug that a shared misreading of the header
# formula could otherwise let slip past both. ----- #


def test_manual_single_pixel_hand_verification():
    # X (1,1,3,3):
    #   [[1, 2, 3],
    #    [4, 5, 6],
    #    [7, 8, 9]]
    # weight[0,0]: w(0,0)=1, w(0,1)=2, w(1,0)=3, w(1,1)=4. bias=0.5.
    # mask (DCNv2): m(k=0)=1.0, m(k=1)=0.5, m(k=2)=1.0, m(k=3)=2.0 -- constant
    # across the (only-computed-by-hand) output pixel (ho=0, wo=0).
    # stride=1, padding=0, dilation=1, deform_groups=1 -> Hout=Wout=2.
    #
    # Offsets, per tap k=i*2+j (channels dy=2k, dx=2k+1), also constant
    # across output pixels:
    #   k=0 (i=0,j=0): dy=0,    dx=0     -> y=0,    x=0     (exact grid point)
    #   k=1 (i=0,j=1): dy=0,    dx=0.5   -> y=0,    x=1.5   (genuine bilinear)
    #   k=2 (i=1,j=0): dy=-1.5, dx=0     -> y=-0.5, x=0     (1 corner OOB)
    #   k=3 (i=1,j=1): dy=0,    dx=-3    -> y=1,    x=-2    (fully OOB)
    #
    # At (ho=0, wo=0): base_y = base_x = 0.
    #
    # k=0: y=0, x=0 -- lands exactly on grid point (0,0): sample = X[0,0] = 1.
    # k=1: y=0, x=1.5 -- y is exact (fy=0), x interpolates columns 1,2 of
    #      row 0: sample = 0.5*X[0,1] + 0.5*X[0,2] = 0.5*2 + 0.5*3 = 2.5.
    # k=2: y=-0.5, x=0 -- y0=floor(-0.5)=-1 (OOB, contributes 0), y1=0
    #      (in range), fy = -0.5-(-1) = 0.5; x is exact (fx=0, x0=x1... no,
    #      x0=0 in range, x1=1 in range, but fx=0 so weight on x1 is 0):
    #      only the (y1=0, x0=0) corner survives, weight = fy*(1-fx) = 0.5*1:
    #      sample = 0.5 * X[0,0] = 0.5 * 1 = 0.5.
    # k=3: y=1, x=-2 -- x0=-2, x1=-1: both OOB regardless of y -> sample = 0.
    #
    # out[0,0,0,0] (DCNv1, mask implicitly 1):
    #   bias + w(0,0)*s0 + w(0,1)*s1 + w(1,0)*s2 + w(1,1)*s3
    #   = 0.5 + 1*1 + 2*2.5 + 3*0.5 + 4*0 = 0.5 + 1 + 5 + 1.5 + 0 = 8.0
    #
    # out[0,0,0,0] (DCNv2, with the mask above):
    #   bias + w(0,0)*m0*s0 + w(0,1)*m1*s1 + w(1,0)*m2*s2 + w(1,1)*m3*s3
    #   = 0.5 + 1*1.0*1 + 2*0.5*2.5 + 3*1.0*0.5 + 4*2.0*0
    #   = 0.5 + 1.0 + 2.5 + 1.5 + 0 = 5.5

    X = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32
    )
    weight = np.array(
        [[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32
    )  # (Cout=1,Cin=1,2,2)
    bias = np.array([0.5], dtype=np.float32)

    # offset channels: dy0,dx0,dy1,dx1,dy2,dx2,dy3,dx3, broadcast over the
    # (Hout=2, Wout=2) spatial grid (only (0,0) is hand-checked below).
    off_vals = [0.0, 0.0, 0.0, 0.5, -1.5, 0.0, 0.0, -3.0]
    offset = np.array(off_vals, dtype=np.float32).reshape(1, 8, 1, 1) * np.ones(
        (1, 8, 2, 2), dtype=np.float32
    )

    mask_vals = [1.0, 0.5, 1.0, 2.0]
    mask = np.array(mask_vals, dtype=np.float32).reshape(1, 4, 1, 1) * np.ones(
        (1, 4, 2, 2), dtype=np.float32
    )

    def build(op_type, with_mask):
        inputs_txt = "float[1,1,3,3] X, float[1,8,2,2] offset"
        call_inputs = "X, offset"
        if with_mask:
            inputs_txt += ", float[1,4,2,2] mask"
            call_inputs += ", mask"
        call_inputs += ", weight, bias"
        body = f"""
        agraph ({inputs_txt}) => (float[1,1,2,2] Y)
        {{
          Y = mmdeploy.{op_type} <stride=[1,1], padding=[0,0], dilation=[1,1], groups=1, deform_groups=1> ({call_inputs})
        }}
        """
        model = _model(body, initializer=[_f32(weight, "weight"), _f32(bias, "bias")])
        feeds = {"X": X, "offset": offset}
        if with_mask:
            feeds["mask"] = mask
        return model, feeds

    dcnv1_model, dcnv1_feeds = build("MMCVDeformConv2d", with_mask=False)
    actual_v1, _ = _simplify_and_run(dcnv1_model, dcnv1_feeds)
    np.testing.assert_allclose(actual_v1[0, 0, 0, 0], 8.0, rtol=1e-5, atol=1e-5)

    dcnv2_model, dcnv2_feeds = build("MMCVModulatedDeformConv2d", with_mask=True)
    actual_v2, _ = _simplify_and_run(dcnv2_model, dcnv2_feeds)
    np.testing.assert_allclose(actual_v2[0, 0, 0, 0], 5.5, rtol=1e-5, atol=1e-5)

    # Cross-check against the (independently written) general reference too,
    # for the full 2x2 output, not just the hand-derived corner.
    expected_v1 = _independent_deform_conv2d(
        X, offset, weight, None, bias, (1, 1), (0, 0), (1, 1), deform_groups=1
    )
    np.testing.assert_allclose(actual_v1, expected_v1, rtol=1e-4, atol=1e-4)
    expected_v2 = _independent_deform_conv2d(
        X, offset, weight, mask, bias, (1, 1), (0, 0), (1, 1), deform_groups=1
    )
    np.testing.assert_allclose(actual_v2, expected_v2, rtol=1e-4, atol=1e-4)
