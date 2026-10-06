"""Tests for onnxsim.certify (Z3 translation validation of simplify()).

Two kinds of test: (1) the op encodings a proof trusts are checked against
onnx's reference evaluator on concrete inputs, and (2) end-to-end verdicts --
real ``simplify()`` output is proved, and deliberately wrong rewrites (a bad
bias, an RGB<->BGR channel swap) are refuted with a counterexample that is then
replayed on the reference evaluator to confirm it is a genuine difference.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

import onnxsim

pytest.importorskip("z3", reason="onnxsim.certify needs the 'verify' extra (z3-solver)")
from onnxsim import certify as C  # noqa: E402


def _model(body, initializer=None, opset=13, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _f32(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _conv_bn_relu(rng, k=4, relu=True):
    body = f"""
    m (float[1,3,6,6] x) => (float[1,{k},4,4] y) {{
      c = Conv(x, W, B)
      b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
      {"y = Relu(b)" if relu else "y = Identity(b)"}
    }}"""
    return _model(
        body,
        dict(
            W=_f32(rng, k, 3, 3, 3),
            B=_f32(rng, k),
            g=rng.uniform(0.5, 1.5, k).astype(np.float32),
            be=_f32(rng, k),
            mu=_f32(rng, k),
            var=rng.uniform(0.5, 2, k).astype(np.float32),
        ),
    )


# ---- (1) encodings vs the reference evaluator ------------------------------

_ENCODING_CASES = {
    "conv_stride_pad_dilation": (
        "m (float[1,2,7,7] x) => (float[1,3,3,3] y) "
        "{ y = Conv<strides=[2,2], pads=[1,1,1,1], dilations=[2,2]>(x, W, B) }",
        dict(W=(3, 2, 3, 3), B=(3,)),
        (1, 2, 7, 7),
    ),
    "conv_group": (
        "m (float[1,4,5,5] x) => (float[1,4,3,3] y) { y = Conv<group=2>(x, W) }",
        dict(W=(4, 2, 3, 3)),
        (1, 4, 5, 5),
    ),
    "gemm_trans_alpha_beta": (
        "m (float[3,5] x) => (float[3,4] y) "
        "{ y = Gemm<transB=1, alpha=0.5, beta=2.0>(x, W, C) }",
        dict(W=(4, 5), C=(4,)),
        (3, 5),
    ),
    "matmul_batched": (
        "m (float[2,3,4] x) => (float[2,3,5] y) { y = MatMul(x, W) }",
        dict(W=(4, 5)),
        (2, 3, 4),
    ),
    "batchnorm": (
        "m (float[1,3,2,2] x) => (float[1,3,2,2] y) "
        "{ y = BatchNormalization<epsilon=1e-3>(x, g, be, mu, var) }",
        dict(g=(3,), be=(3,), mu=(3,), var=(3,)),
        (1, 3, 2, 2),
    ),
    "relu_clip_sub_mul": (
        "m (float[6] x) => (float[6] y) "
        "{ a = Relu(x)  lo = Constant<value=float {-0.5}>()  hi = Constant<value=float {0.7}>() "
        "b = Clip(x, lo, hi)  c = Sub(a, b)  y = Mul(c, x) }",
        {},
        (6,),
    ),
    "reshape_transpose_flatten": (
        "m (float[2,3,4] x) => (float[3,8] y) "
        "{ t = Transpose<perm=[1,0,2]>(x)  y = Flatten<axis=1>(t) }",
        {},
        (2, 3, 4),
    ),
}


@pytest.mark.parametrize("case", sorted(_ENCODING_CASES))
def test_encoding_matches_reference_evaluator(case):
    body, shapes, in_shape = _ENCODING_CASES[case]
    rng = np.random.default_rng(0)
    inits = {
        k: rng.uniform(0.5, 1.5, s).astype(np.float32) if k == "var" else _f32(rng, *s)
        for k, s in shapes.items()
    }
    # opset 15, not 13: onnx's ReferenceEvaluator disagrees with onnxruntime and with
    # this encoding on opset-13 BatchNormalization (off by ~0.09 whatever epsilon is),
    # and matches both from opset 15.
    model = _model(body, inits, opset=15)
    x = _f32(rng, *in_shape)
    want = ReferenceEvaluator(model).run(None, {"x": x})[0]
    got = C._concrete_eval(model, {"x": x})["y"]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


# ---- (2) end-to-end verdicts -----------------------------------------------

_BOX = {"x": (-1.0, 1.0)}


def test_proves_real_simplify_conv_bn_relu():
    model = _conv_bn_relu(np.random.default_rng(1))
    sim, _ = onnxsim.simplify(model)
    assert "BatchNormalization" not in {n.op_type for n in sim.graph.node}
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.ok, str(report)
    # Relu is peeled by congruence; only the Conv/BN window reaches the solver.
    assert report.outputs["y"] == C.PROVED_CONGRUENCE
    assert any(w.status == C.PROVED_SMT for w in report.windows)


def test_proves_real_simplify_matmul_add_to_gemm():
    rng = np.random.default_rng(2)
    model = _model(
        """
        m (float[1,16] x) => (float[1,4] y) {
          a = MatMul(x, W1)
          b = Add(a, B1)
          r = Relu(b)
          c = MatMul(r, W2)
          y = Add(c, B2)
        }""",
        dict(W1=_f32(rng, 16, 8), B1=_f32(rng, 8), W2=_f32(rng, 8, 4), B2=_f32(rng, 4)),
    )
    sim, _ = onnxsim.simplify(model)
    assert "Gemm" in {n.op_type for n in sim.graph.node}
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.ok, str(report)


def test_identical_models_are_structural():
    model = _conv_bn_relu(np.random.default_rng(3))
    report = C.certify(model, model)
    assert report.ok and report.outputs["y"] == C.PROVED_STRUCTURAL
    assert report.windows == []


def test_wrong_bias_is_refuted_and_counterexample_is_real():
    rng = np.random.default_rng(4)
    model = _conv_bn_relu(rng, relu=False)
    bad = onnx.ModelProto()
    bad.CopyFrom(model)
    for t in bad.graph.initializer:
        if t.name == "be":
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t) + 0.25, "be"))
    report = C.certify(model, bad, input_ranges=_BOX)
    assert not report.ok and report.outputs["y"] == C.REFUTED
    cex = report.windows[0].counterexample
    x = cex["x"].astype(np.float32)
    a = ReferenceEvaluator(model).run(None, {"x": x})[0]
    b = ReferenceEvaluator(bad).run(None, {"x": x})[0]
    assert np.abs(a - b).max() > 0.2


def _conv_with_weights(w):
    return _model(
        "m (float[1,3,6,6] x) => (float[1,4,4,4] y) { y = Conv(x, W, B) }",
        dict(W=w, B=np.zeros(4, np.float32)),
    )


def test_rgb_bgr_swap_is_refuted_but_symmetric_weights_are_not():
    rng = np.random.default_rng(5)
    w = _f32(rng, 4, 3, 3, 3)
    swapped = C.certify(
        _conv_with_weights(w),
        _conv_with_weights(w[:, ::-1].copy()),
        input_ranges={"x": (0.0, 1.0)},
    )
    assert swapped.outputs["y"] == C.REFUTED

    sym = np.repeat(w[:, :1], 3, axis=1)  # identical across channels: a swap is a no-op
    harmless = C.certify(
        _conv_with_weights(sym),
        _conv_with_weights(sym[:, ::-1].copy()),
        input_ranges={"x": (0.0, 1.0)},
    )
    assert harmless.ok, str(harmless)


def test_unbounded_refutation_explains_missing_ranges():
    model = _conv_bn_relu(np.random.default_rng(6), relu=False)
    sim, _ = onnxsim.simplify(model)
    report = C.certify(model, sim)  # no input_ranges
    if (
        not report.ok
    ):  # folded constants differ by float rounding, amplified by a huge input
        assert "input_ranges" in report.windows[0].detail


def test_unsupported_op_is_skipped_not_proved():
    model = _model("m (float[4] x) => (float[4] y) { y = Abs(x) }")
    other = _model(
        "m (float[4] x) => (float[4] y) { a = Relu(x)  n = Neg(x)  b = Relu(n)  y = Add(a, b) }"
    )
    report = C.certify(model, other)
    assert not report.ok and report.outputs["y"] == C.SKIPPED
    assert "Abs" in report.windows[0].detail


def test_large_window_is_skipped():
    rng = np.random.default_rng(7)
    model = _conv_bn_relu(rng, relu=False)
    sim, _ = onnxsim.simplify(model)
    report = C.certify(model, sim, input_ranges=_BOX, max_work=100)
    assert report.outputs["y"] == C.SKIPPED and "too large" in report.windows[0].detail


def test_mismatched_inputs_raise():
    a = _model("m (float[4] x) => (float[4] y) { y = Relu(x) }")
    b = _model("m (float[4] z) => (float[4] y) { y = Relu(z) }")
    with pytest.raises(ValueError, match="inputs differ"):
        C.certify(a, b)


# ---- (3) spatial shrink of windows too large for the full-size encoding ------
#
# A window of spatially local ops is re-encoded at a smaller spatial size that keeps
# all of its border behaviour (see the "Spatial shrink" notes in onnxsim/certify.py).


def _conv_bn_model(rng, h, w, strides=(1, 1), pads=(1, 1, 1, 1), k=4, relu=False):
    oh = (h + pads[0] + pads[2] - 3) // strides[0] + 1
    ow = (w + pads[1] + pads[3] - 3) // strides[1] + 1
    last = "b = BatchNormalization" if relu else "y = BatchNormalization"
    return _model(
        f"""
        m (float[1,3,{h},{w}] x) => (float[1,{k},{oh},{ow}] y) {{
          c = Conv<strides={list(strides)}, pads={list(pads)}>(x, W, B)
          {last}<epsilon=1e-5>(c, g, be, mu, var)
          {"y = Relu(b)" if relu else ""}
        }}""",
        dict(
            W=_f32(rng, k, 3, 3, 3),
            B=_f32(rng, k),
            g=rng.uniform(0.5, 1.5, k).astype(np.float32),
            be=_f32(rng, k),
            mu=_f32(rng, k),
            var=rng.uniform(0.5, 2, k).astype(np.float32),
        ),
        opset=15,
    )


def _with_conv_bias_shifted(model, delta):
    bad = onnx.ModelProto()
    bad.CopyFrom(model)
    conv = next(n for n in bad.graph.node if n.op_type == "Conv")
    bias_name = conv.input[2]
    for t in bad.graph.initializer:
        if t.name == bias_name:
            t.CopyFrom(
                numpy_helper.from_array(numpy_helper.to_array(t) + delta, bias_name)
            )
    return bad


def _run(model, x):
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x.astype(np.float32)})[0]


@pytest.mark.parametrize(
    "h,w,strides,pads",
    [
        (64, 64, (1, 1), (1, 1, 1, 1)),
        (128, 128, (2, 2), (1, 1, 1, 1)),
        (129, 127, (2, 2), (1, 1, 1, 1)),  # odd sizes: the residue mod stride is kept
        (100, 100, (1, 1), (0, 0, 0, 0)),  # no padding at all: every position interior
        (128, 96, (2, 1), (1, 0, 0, 1)),  # asymmetric strides and pads
    ],
)
def test_shrink_proves_real_simplify_at_realistic_sizes(h, w, strides, pads):
    import re

    model = _conv_bn_model(np.random.default_rng(10), h, w, strides, pads)
    sim, _ = onnxsim.simplify(model, certify=False)
    assert "BatchNormalization" not in {n.op_type for n in sim.graph.node}
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.ok, str(report)
    win = next(x for x in report.windows if x.status == C.PROVED_REDUCED)
    # "<H>x<W> -> <h>x<w> spatial": smaller, with the same residue modulo the stride
    m = re.search(r"(\d+)x(\d+) -> (\d+)x(\d+) spatial", win.detail)
    fh, fw, rh, rw = map(int, m.groups())
    assert (fh, fw) == (h, w) and rh < h and rw < w
    assert (h - rh) % strides[0] == 0 and (w - rw) % strides[1] == 0


def test_shrink_through_two_stacked_convs():
    # Two 3x3 convs compose to a 5x5 receptive field: two top and two bottom border
    # positions, so the reduced extent must still contain all of them.
    rng = np.random.default_rng(11)
    k = 2

    def bn(tag):
        return {
            "g" + tag: rng.uniform(0.5, 1.5, k).astype(np.float32),
            "be" + tag: _f32(rng, k),
            "mu" + tag: _f32(rng, k),
            "var" + tag: rng.uniform(0.5, 2, k).astype(np.float32),
        }

    model = _model(
        """
        m (float[1,3,32,32] x) => (float[1,2,32,32] y) {
          c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
          b1 = BatchNormalization<epsilon=1e-5>(c1, g1, be1, mu1, var1)
          c2 = Conv<pads=[1,1,1,1]>(b1, W2, B2)
          y = BatchNormalization<epsilon=1e-5>(c2, g2, be2, mu2, var2)
        }""",
        dict(
            W1=_f32(rng, k, 3, 3, 3),
            B1=_f32(rng, k),
            W2=_f32(rng, k, k, 3, 3),
            B2=_f32(rng, k),
            **bn("1"),
            **bn("2"),
        ),
        opset=15,
    )
    sim, _ = onnxsim.simplify(model, certify=False)
    # a small max_work forces the shrink at this (cheap-to-test) size
    report = C.certify(model, sim, input_ranges=_BOX, max_work=20_000)
    assert report.ok, str(report)
    win = next(w for w in report.windows if w.status == C.PROVED_REDUCED)
    assert "32x32 -> 5x5" in win.detail  # 2 top + 1 interior + 2 bottom positions


def test_shrink_peels_a_relu_on_top_of_a_shrunk_window():
    model = _conv_bn_model(np.random.default_rng(18), 64, 64, relu=True)
    sim, _ = onnxsim.simplify(model, certify=False)
    report = C.certify(model, sim, input_ranges=_BOX)
    assert report.ok, str(report)
    assert report.outputs["y"] == C.PROVED_CONGRUENCE  # Relu peeled by congruence
    assert any(w.status == C.PROVED_REDUCED for w in report.windows)


def test_shrink_wrong_rewrite_is_refuted_with_a_full_size_counterexample():
    model = _conv_bn_model(np.random.default_rng(12), 64, 64)
    sim, _ = onnxsim.simplify(model, certify=False)
    bad = _with_conv_bias_shifted(sim, 0.25)
    report = C.certify(model, bad, input_ranges=_BOX)
    assert not report.ok and report.outputs["y"] == C.REFUTED
    win = report.windows[0]
    assert "reduced shape" in win.detail
    assert "confirmed on the original models" in win.detail
    x = win.counterexample["x"]
    assert x.shape == (1, 3, 64, 64)  # the ORIGINAL size, not the reduced one
    assert x.min() >= -1.0 and x.max() <= 1.0  # still inside the declared box
    # independent replay of the counterexample on the two real models
    assert np.abs(_run(model, x) - _run(bad, x)).max() > 0.2


def test_shrink_border_only_bug_is_found_and_lifted_to_the_bottom_right():
    # A: Conv(x) + d*sum(W);  B: Conv(x + d).  Equal in the interior, different only
    # where the window touches the (bottom/right) zero padding -- which the lift must
    # reproduce by moving the violating position to the real bottom-right edge.
    rng = np.random.default_rng(13)
    w = _f32(rng, 4, 3, 3, 3)
    d = np.float32(0.5)
    shift = (d * w.sum(axis=(1, 2, 3))).reshape(1, 4, 1, 1).astype(np.float32)
    a = _model(
        """
        m (float[1,3,64,64] x) => (float[1,4,64,64] y) {
          c = Conv<pads=[0,0,2,2]>(x, W)
          y = Add(c, S)
        }""",
        dict(W=w, S=shift),
        opset=15,
    )
    b = _model(
        """
        m (float[1,3,64,64] x) => (float[1,4,64,64] y) {
          xd = Add(x, D)
          y = Conv<pads=[0,0,2,2]>(xd, W)
        }""",
        dict(W=w, D=np.asarray(d)),
        opset=15,
    )
    report = C.certify(a, b, input_ranges=_BOX)
    assert not report.ok and report.outputs["y"] == C.REFUTED, str(report)
    win = report.windows[0]
    assert "confirmed on the original models" in win.detail
    x = win.counterexample["x"]
    assert x.shape == (1, 3, 64, 64)
    diff = np.abs(_run(a, x) - _run(b, x))
    assert diff.max() > 0.05  # a real difference at full size ...
    assert diff[:, :, :62, :62].max() < 1e-4  # ... only along the bottom/right border


def _forced_shrink_refutation(monkeypatch, replay):
    model = _conv_bn_model(np.random.default_rng(14), 64, 64)
    sim, _ = onnxsim.simplify(model, certify=False)
    bad = _with_conv_bias_shifted(sim, 0.25)
    monkeypatch.setattr(C, "_make_replay", lambda *a, **k: replay)
    return C.certify(model, bad, input_ranges=_BOX)


def test_shrink_replay_that_disagrees_downgrades_refuted_to_skipped(monkeypatch):
    report = _forced_shrink_refutation(monkeypatch, lambda a, b, lifted: False)
    assert report.outputs["y"] == C.SKIPPED and not report.ok
    assert "did not reproduce" in report.windows[0].detail


def test_shrink_without_a_replay_still_refutes_but_says_so(monkeypatch):
    report = _forced_shrink_refutation(monkeypatch, None)
    assert report.outputs["y"] == C.REFUTED
    assert "not replayed" in report.windows[0].detail


def _skipped_with_reason(a, b, reason):
    report = C.certify(a, b, max_work=500)
    assert not report.ok and report.outputs["y"] == C.SKIPPED, str(report)
    detail = report.windows[0].detail
    assert "not shrinkable" in detail and reason in detail, detail


def test_shrink_refuses_a_non_spatially_local_op():
    rng = np.random.default_rng(15)
    conv = dict(W=_f32(rng, 4, 3, 3, 3), B=_f32(rng, 4))
    a = _model(
        """
        m (float[1,3,16,16] x) => (float[1,1024] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          y = Flatten(c)
        }""",
        conv,
        opset=15,
    )
    b = _model(
        """
        m (float[1,3,16,16] x) => (float[1,1024] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          y = Reshape(c, S)
        }""",
        dict(conv, S=np.array([1, -1], dtype=np.int64)),
        opset=15,
    )
    _skipped_with_reason(a, b, "Flatten")


def test_shrink_refuses_a_positional_constant():
    rng = np.random.default_rng(16)
    conv = dict(W=_f32(rng, 4, 3, 3, 3), B=_f32(rng, 4))
    pos = _f32(rng, 1, 4, 16, 16)  # a different value at every position
    a = _model(
        """
        m (float[1,3,16,16] x) => (float[1,4,16,16] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          y = Add(c, P)
        }""",
        dict(conv, P=pos),
        opset=15,
    )
    b = _model(
        """
        m (float[1,3,16,16] x) => (float[1,4,16,16] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          y = Sub(c, N)
        }""",
        dict(conv, N=-pos),
        opset=15,
    )
    _skipped_with_reason(a, b, "varies over space")


def test_shrink_refuses_an_input_range_that_varies_over_space():
    model = _conv_bn_model(np.random.default_rng(17), 64, 64)
    sim, _ = onnxsim.simplify(model, certify=False)
    lo = -np.ones((1, 3, 64, 64), dtype=np.float32)
    hi = np.ones((1, 3, 64, 64), dtype=np.float32)
    report = C.certify(model, sim, input_ranges={"x": (lo, hi)})
    assert report.outputs["y"] == C.SKIPPED and not report.ok
    assert "input range" in report.windows[0].detail
