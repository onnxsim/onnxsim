"""Tests for the opt-in ``rewrite_deform_conv_to_gather`` pass.

``MMCVDeformConv2d``/``MMCVModulatedDeformConv2d`` (mmdeploy/mmcv's exported
deformable-convolution ops, DCNv1/DCNv2) have no ONNX Runtime kernel, so
onnxsim's usual ``simplify(..., check_n=N)`` equivalence check -- which
compares the simplified graph's output against the *original* graph's own
output, executed via onnxruntime or onnx's reference evaluator -- cannot even
load the original graph, let alone run it. There is nothing to compare
against.

So, mirroring ``tests/test_gridsample_to_gather.py``'s approach for a
different not-quite-standard op: every test here calls ``simplify`` with
``check_n=0`` (the default -- this only runs the graph through
``onnx.checker``, which happily accepts an unrecognized op in a non-standard
domain like ``mmdeploy``, and never tries to execute the original node), then
independently verifies the *simplified* graph -- now built entirely from
standard ops -- against a from-scratch NumPy implementation of modulated
deformable convolution v2 (Zhu et al.; DCNv1 is the same algorithm with an
implicit all-ones mask), run on the same random inputs. This is a
well-established, standard algorithm (the same one mmcv's CUDA kernel and
torchvision's ``deform_conv2d`` both implement) -- see
``_deform_conv2d_reference`` below -- not something op-specific invented for
this test.

The simplified graph itself is executed with onnxruntime when available,
falling back to ``onnx.reference.ReferenceEvaluator`` otherwise (every op the
rewrite emits -- ``Range``, ``Gather``, ``GatherND``, ``Slice``, ``Cast``,
``Reshape``, ``Transpose``, ``Concat``, ``Unsqueeze``, ``MatMul`` and plain
arithmetic/comparison ops -- is supported by both).
"""

import collections

import numpy as np
import onnx
from onnx import parser

import onnxsim

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None


def _model(body, initializer=(), opset=17, ir_version=10, custom_domain="mmdeploy"):
    # `body` is just the graph declaration (`agraph (...) => (...) { ... }`),
    # not a full text-format model -- the ir_version/opset_import header is
    # added here, once, per CLAUDE.md's `_model` helper convention.
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "{custom_domain}": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


# --------------------------------------------------------------------------- #
# NumPy reference implementation of modulated deformable convolution v2.
# --------------------------------------------------------------------------- #


def _bilinear_sample_zero_pad(img, y, x):
    """4-corner bilinear sample of ``img`` (H,W) at float coords ``y``/``x``
    (broadcastable arrays), zero-padded outside ``[0,H-1]``/``[0,W-1]`` --
    the same rule as GridSample's ``padding_mode="zeros"``."""
    H, W = img.shape
    x0 = np.floor(x)
    x1 = x0 + 1
    y0 = np.floor(y)
    y1 = y0 + 1
    wx1 = x - x0
    wx0 = 1 - wx1
    wy1 = y - y0
    wy0 = 1 - wy1

    def get(yy, xx):
        valid = (yy >= 0) & (yy <= H - 1) & (xx >= 0) & (xx <= W - 1)
        yyc = np.clip(yy, 0, H - 1).astype(np.int64)
        xxc = np.clip(xx, 0, W - 1).astype(np.int64)
        val = img[yyc, xxc]
        return np.where(valid, val, 0.0)

    return (
        get(y0, x0) * wx0 * wy0
        + get(y0, x1) * wx1 * wy0
        + get(y1, x0) * wx0 * wy1
        + get(y1, x1) * wx1 * wy1
    )


def _deform_conv2d_reference(
    X,
    offset,
    weight,
    mask,
    bias,
    stride,
    padding,
    dilation,
    deform_groups,
):
    """Standard modulated deformable convolution v2 (Zhu et al.), groups=1
    only (matching this pass's own scope limit). ``mask=None`` gives plain
    (DCNv1) unmodulated deformable convolution."""
    N, Cin, H, W = X.shape
    Cout, Cin_g, kh, kw = weight.shape
    assert Cin_g == Cin, "reference only supports groups == 1"
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    Hout, Wout = offset.shape[2], offset.shape[3]
    Cin_dg = Cin // deform_groups

    ho = np.arange(Hout, dtype=np.float64).reshape(-1, 1)
    wo = np.arange(Wout, dtype=np.float64).reshape(1, -1)

    out = np.zeros((N, Cout, Hout, Wout), dtype=np.float64)
    for n in range(N):
        for cin in range(Cin):
            dg = cin // Cin_dg
            for i in range(kh):
                for j in range(kw):
                    k = i * kw + j
                    dy = offset[n, dg * 2 * kh * kw + 2 * k].astype(np.float64)
                    dx = offset[n, dg * 2 * kh * kw + 2 * k + 1].astype(np.float64)
                    y = ho * sh - ph + i * dh + dy
                    x = wo * sw - pw + j * dw + dx
                    sampled = _bilinear_sample_zero_pad(
                        X[n, cin].astype(np.float64), y, x
                    )
                    if mask is not None:
                        sampled = sampled * mask[n, dg * kh * kw + k].astype(np.float64)
                    # Accumulate this input channel's contribution into every
                    # output channel at once via the weight column.
                    out[n, :, :, :] += (
                        weight[:, cin, i, j].astype(np.float64).reshape(-1, 1, 1)
                        * sampled[None, :, :]
                    )
    if bias is not None:
        out += bias.astype(np.float64).reshape(1, -1, 1, 1)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Simplify + run-and-compare harness.
# --------------------------------------------------------------------------- #


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


def _simplify_and_check(model, feeds, expected):
    sim_model, check_ok = onnxsim.simplify(
        model,
        extra_optimizers=["rewrite_deform_conv_to_gather"],
    )
    assert check_ok, "rewritten graph failed onnxsim's checker-only check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "MMCVDeformConv2d" not in op_types, op_types
    assert "MMCVModulatedDeformConv2d" not in op_types, op_types
    assert "GatherND" in op_types, op_types

    actual = _run_model(sim_model, feeds)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    return sim_model, op_types


def _offset_shape(deform_groups, kh, kw, n, hout, wout):
    return [n, deform_groups * 2 * kh * kw, hout, wout]


def _mask_shape(deform_groups, kh, kw, n, hout, wout):
    return [n, deform_groups * kh * kw, hout, wout]


def _conv_out_hw(h, w, kh, kw, stride, padding, dilation):
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    hout = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wout = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return hout, wout


def _build_and_check(
    op_type,
    n,
    cin,
    h,
    w,
    cout,
    kh,
    kw,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    deform_groups=1,
    with_bias=False,
    dynamic_n=False,
    dynamic_hw=False,
    seed=0,
):
    """Builds a single-node model for ``op_type``, runs the pass, and checks
    the simplified graph's output against the NumPy reference. ``dynamic_n``/
    ``dynamic_hw`` substitute symbolic dims for the batch/spatial axes (the
    actual numpy arrays are still concrete -- only the graph's declared
    shapes go symbolic) to prove the rewrite doesn't assume them static."""
    rng = np.random.RandomState(seed)
    modulated = op_type == "MMCVModulatedDeformConv2d"
    hout, wout = _conv_out_hw(h, w, kh, kw, stride, padding, dilation)

    X = rng.randn(n, cin, h, w).astype(np.float32)
    # Offsets: mostly small (in-range) but with enough spread that some taps
    # land out-of-bounds, exercising the zeros-padding path.
    offset = rng.uniform(
        -2.0, 2.0, size=_offset_shape(deform_groups, kh, kw, n, hout, wout)
    ).astype(np.float32)
    weight = rng.randn(cout, cin, kh, kw).astype(np.float32) * 0.5
    mask = None
    if modulated:
        mask = rng.uniform(
            0.0, 1.0, size=_mask_shape(deform_groups, kh, kw, n, hout, wout)
        ).astype(np.float32)
    bias = rng.randn(cout).astype(np.float32) if with_bias else None

    expected = _deform_conv2d_reference(
        X,
        offset,
        weight,
        mask,
        bias,
        stride,
        padding,
        dilation,
        deform_groups,
    )

    x_n = "N" if dynamic_n else str(n)
    x_h = "H" if dynamic_hw else str(h)
    x_w = "W" if dynamic_hw else str(w)
    out_h = "Hout" if dynamic_hw else str(hout)
    out_w = "Wout" if dynamic_hw else str(wout)

    x_shape_txt = f"[{x_n},{cin},{x_h},{x_w}]"
    off_shape_txt = f"[{x_n},{deform_groups * 2 * kh * kw},{out_h},{out_w}]"
    mask_shape_txt = f"[{x_n},{deform_groups * kh * kw},{out_h},{out_w}]"
    out_shape_txt = f"[{x_n},{cout},{out_h},{out_w}]"

    inputs_txt = f"float{x_shape_txt} X, float{off_shape_txt} offset"
    call_inputs = "X, offset"
    if modulated:
        inputs_txt += f", float{mask_shape_txt} mask"
        call_inputs += ", mask"
    call_inputs += ", weight"
    if with_bias:
        call_inputs += ", bias"

    body = f"""
    agraph ({inputs_txt}) => (float{out_shape_txt} Y)
    {{
      Y = mmdeploy.{op_type} <stride=[{stride[0]},{stride[1]}], padding=[{padding[0]},{padding[1]}], dilation=[{dilation[0]},{dilation[1]}], groups=1, deform_groups={deform_groups}> ({call_inputs})
    }}
    """
    initializer = [_f32(weight, "weight")]
    if with_bias:
        initializer.append(_f32(bias, "bias"))
    model = _model(body, initializer=initializer)

    feeds = {"X": X, "offset": offset}
    if modulated:
        feeds["mask"] = mask

    return _simplify_and_check(model, feeds, expected)


# --------------------------------------------------------------------------- #
# Unmodulated (DCNv1, MMCVDeformConv2d).
# --------------------------------------------------------------------------- #


def test_dcnv1_basic():
    _build_and_check(
        "MMCVDeformConv2d",
        n=2,
        cin=3,
        h=6,
        w=7,
        cout=4,
        kh=3,
        kw=3,
        padding=(1, 1),
    )


def test_dcnv1_with_bias():
    _build_and_check(
        "MMCVDeformConv2d",
        n=1,
        cin=2,
        h=5,
        w=5,
        cout=3,
        kh=3,
        kw=3,
        padding=(1, 1),
        with_bias=True,
    )


# --------------------------------------------------------------------------- #
# Modulated (DCNv2, MMCVModulatedDeformConv2d).
# --------------------------------------------------------------------------- #


def test_dcnv2_basic_no_bias():
    _build_and_check(
        "MMCVModulatedDeformConv2d",
        n=2,
        cin=3,
        h=6,
        w=7,
        cout=4,
        kh=3,
        kw=3,
        padding=(1, 1),
    )


def test_dcnv2_with_bias():
    _build_and_check(
        "MMCVModulatedDeformConv2d",
        n=2,
        cin=3,
        h=6,
        w=7,
        cout=4,
        kh=3,
        kw=3,
        padding=(1, 1),
        with_bias=True,
    )


# --------------------------------------------------------------------------- #
# deform_groups > 1 (groups == 1) -- the common mmdetection DCNv2 config.
# --------------------------------------------------------------------------- #


def test_dcnv2_deform_groups_4():
    _build_and_check(
        "MMCVModulatedDeformConv2d",
        n=1,
        cin=8,
        h=6,
        w=6,
        cout=6,
        kh=3,
        kw=3,
        padding=(1, 1),
        deform_groups=4,
        with_bias=True,
    )


def test_dcnv1_deform_groups_2():
    _build_and_check(
        "MMCVDeformConv2d",
        n=1,
        cin=4,
        h=6,
        w=6,
        cout=3,
        kh=3,
        kw=3,
        padding=(1, 1),
        deform_groups=2,
    )


# --------------------------------------------------------------------------- #
# Non-trivial stride / padding / dilation.
# --------------------------------------------------------------------------- #


def test_dcnv2_stride_padding_dilation():
    _build_and_check(
        "MMCVModulatedDeformConv2d",
        n=1,
        cin=3,
        h=11,
        w=13,
        cout=4,
        kh=3,
        kw=3,
        stride=(2, 2),
        padding=(2, 2),
        dilation=(2, 2),
        deform_groups=1,
        with_bias=True,
    )


def test_dcnv1_asymmetric_kernel_and_stride():
    _build_and_check(
        "MMCVDeformConv2d",
        n=1,
        cin=2,
        h=9,
        w=10,
        cout=3,
        kh=3,
        kw=5,
        stride=(1, 2),
        padding=(1, 2),
        dilation=(1, 1),
    )


# --------------------------------------------------------------------------- #
# Dynamic (symbolic) N/H/W -- the pass must derive H/W from Shape(input) and
# Hout/Wout from Shape(offset) at runtime, never assume they're static.
# --------------------------------------------------------------------------- #


def test_dynamic_batch_and_spatial_dims():
    _build_and_check(
        "MMCVModulatedDeformConv2d",
        n=2,
        cin=4,
        h=6,
        w=7,
        cout=4,
        kh=3,
        kw=3,
        padding=(1, 1),
        deform_groups=2,
        dynamic_n=True,
        dynamic_hw=True,
    )


# --------------------------------------------------------------------------- #
# Scope limit: groups > 1 is declined (not silently mis-computed). This pass
# only supports groups == 1 -- see rewrite_deform_conv_to_gather.h's header
# comment for why.
# --------------------------------------------------------------------------- #


def test_declines_groups_greater_than_one():
    n, cin, h, w, cout, kh, kw, groups = 1, 4, 6, 6, 4, 3, 3, 2
    stride, padding, dilation, deform_groups = (1, 1), (1, 1), (1, 1), 1
    hout, wout = _conv_out_hw(h, w, kh, kw, stride, padding, dilation)
    cin_g = cin // groups

    rng = np.random.RandomState(0)
    weight = rng.randn(cout, cin_g, kh, kw).astype(np.float32)

    body = f"""
    agraph (float[{n},{cin},{h},{w}] X, float[{n},{deform_groups * 2 * kh * kw},{hout},{wout}] offset) => (float[{n},{cout},{hout},{wout}] Y)
    {{
      Y = mmdeploy.MMCVDeformConv2d <stride=[{stride[0]},{stride[1]}], padding=[{padding[0]},{padding[1]}], dilation=[{dilation[0]},{dilation[1]}], groups={groups}, deform_groups={deform_groups}> (X, offset, weight)
    }}
    """
    model = _model(body, initializer=[_f32(weight, "weight")])

    sim_model, check_ok = onnxsim.simplify(
        model,
        extra_optimizers=["rewrite_deform_conv_to_gather"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    # The predicate must decline (groups != 1): the custom op survives
    # untouched rather than being silently (and incorrectly) rewritten as if
    # groups == 1.
    assert op_types["MMCVDeformConv2d"] == 1, op_types


# --------------------------------------------------------------------------- #
# Non-mmdeploy/"" domain and unrelated ops are left alone.
# --------------------------------------------------------------------------- #


def test_declines_unrelated_domain():
    body = """
    agraph (float[1,3,4,4] X, float[1,8,4,4] offset, float[6,3,3,3] weight) => (float[1,6,4,4] Y)
    {
      Y = some_other_vendor.MMCVDeformConv2d <stride=[1,1], padding=[1,1], dilation=[1,1], groups=1, deform_groups=1> (X, offset, weight)
    }
    """
    model = _model(body, custom_domain="some_other_vendor")
    sim_model, check_ok = onnxsim.simplify(
        model,
        extra_optimizers=["rewrite_deform_conv_to_gather"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types["MMCVDeformConv2d"] == 1, op_types
