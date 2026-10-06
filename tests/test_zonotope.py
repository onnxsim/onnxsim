"""Tests for onnxsim.zonotope (DeepZ-style zonotope analysis + certified difference bounds).

Central properties:

* Soundness -- every value onnxruntime produces for an input inside the box, for *every*
  intermediate tensor, lies inside the zonotope's concretised bounds (float32 slack).
* Usefulness -- on REAL ``onnxsim.simplify`` output, ``bound_difference`` is orders of
  magnitude tighter than plain interval propagation on the product graph, and is small
  enough to prove the (exact) rewrite equal; wrong rewrites are never proved equal.
* Direction -- the zonotope never proves what Z3 (``onnxsim.certify``) refutes.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import interval
from onnxsim import zonotope as Z

BOX = {"x": (-1.0, 1.0)}
# float32 execution can exceed a real-arithmetic bound by float32 rounding.
SLACK = 1e-4


def _model(body, initializer=None, opset=15, ir_version=8):
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


def _simplify(model):
    # certify=False: these tests are about zonotopes, not the default-on certify hook.
    return onnxsim.simplify(model, certify=False)[0]


def _conv_bn_relu(rng, k=4, relu=True):
    return _model(
        f"""
        m (float[1,3,6,6] x) => (float[1,{k},4,4] y) {{
          c = Conv(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          {"y = Relu(b)" if relu else "y = Identity(b)"}
        }}""",
        dict(
            W=_f32(rng, k, 3, 3, 3), B=_f32(rng, k),
            g=rng.uniform(0.5, 1.5, k).astype(np.float32), be=_f32(rng, k),
            mu=_f32(rng, k), var=rng.uniform(0.5, 2, k).astype(np.float32),
        ),
    )  # fmt: skip


def _mlp(rng):
    return _model(
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


def _all_tensors(model, x):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = [o for n in m.graph.node for o in n.output if o]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in produced)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip(produced, sess.run(None, x)))


def _contains(res, name, value):
    lo, hi = res.bounds(name)
    v = np.asarray(value, dtype=np.float64)
    pad = SLACK * (1.0 + np.abs(v))
    return bool(np.all(v >= lo - pad) and np.all(v <= hi + pad))


def _assert_sound(model, rng, input_shape, box=(-1.0, 1.0), n=40, **kw):
    res = Z.propagate(model, {"x": box}, **kw)
    for _ in range(n):
        x = rng.uniform(box[0], box[1], input_shape).astype(np.float32)
        for name, val in _all_tensors(model, {"x": x}).items():
            assert name in res.tensors
            assert _contains(res, name, val), f"{name} escaped its zonotope bounds"
    return res


# ---- algebra ------------------------------------------------------------------


def test_cancellation_is_exact():
    space = Z._Space()
    z = Z._box_zonotope(np.array([-1.0, 0.0]), np.array([1.0, 3.0]), space)
    d = Z._add(z, z, -1.0)
    assert np.all(d.bounds(rel_slack=0.0)[0] == 0.0) and np.all(
        d.bounds(rel_slack=0.0)[1] == 0.0
    )
    two = Z._add(z, z, 1.0)
    np.testing.assert_allclose(two.bounds(rel_slack=0.0)[0], [-2.0, 0.0])
    np.testing.assert_allclose(two.bounds(rel_slack=0.0)[1], [2.0, 6.0])


def test_relu_stable_neurons_are_exact_and_unstable_get_a_fresh_symbol():
    space = Z._Space()
    z = Z._box_zonotope(np.array([1.0, -3.0, -1.0]), np.array([2.0, -2.0, 1.0]), space)
    r = Z._relu(z, space)
    lo, hi = r.bounds(rel_slack=0.0)
    np.testing.assert_allclose(lo[:2], [1.0, 0.0])  # positive: identity; negative: zero
    np.testing.assert_allclose(hi[:2], [2.0, 0.0])
    assert (
        lo[2] <= 0.0 <= hi[2] and hi[2] >= 1.0
    )  # unstable one encloses relu on [-1, 1]
    assert (
        len(r.ids) == len(z.ids) + 1
    )  # exactly one fresh symbol (the unstable neuron)


# ---- soundness vs onnxruntime -------------------------------------------------


def test_sound_mlp():
    _assert_sound(_mlp(np.random.default_rng(0)), np.random.default_rng(10), (1, 16))


def test_sound_conv_bn_relu():
    _assert_sound(
        _conv_bn_relu(np.random.default_rng(1)), np.random.default_rng(11), (1, 3, 6, 6)
    )


def test_sound_residual_add_and_gemm_transb():
    rng = np.random.default_rng(2)
    m = _model(
        """
        m (float[1,3,6,6] x) => (float[1,5] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          r = Relu(c)
          a = Add(r, c)
          p = GlobalAveragePool(a)
          f = Flatten(p)
          y = Gemm<transB=1, alpha=0.5, beta=2.0>(f, W2, B2)
        }""",
        dict(
            W=_f32(rng, 4, 3, 3, 3), B=_f32(rng, 4), W2=_f32(rng, 5, 4), B2=_f32(rng, 5)
        ),
    )
    _assert_sound(m, np.random.default_rng(12), (1, 3, 6, 6))


def test_sound_sigmoid_tanh_chain():
    rng = np.random.default_rng(3)
    m = _model(
        """
        m (float[1,8] x) => (float[1,3] y) {
          a = MatMul(x, W1)
          s = Sigmoid(a)
          b = MatMul(s, W2)
          t = Tanh(b)
          y = Add(t, B)
        }""",
        dict(W1=_f32(rng, 8, 6), W2=_f32(rng, 6, 3), B=_f32(rng, 3)),
    )
    res = _assert_sound(m, np.random.default_rng(13), (1, 8), box=(-3.0, 3.0))
    lo, hi = res.bounds("s")
    assert (
        lo.min() >= 0.0 - 1e-6 and hi.max() <= 1.0 + 1e-6
    )  # Sigmoid stays in its codomain


def test_sound_shape_ops():
    rng = np.random.default_rng(4)
    m = _model(
        """
        m (float[2,3,4] x) => (float[3,8] y) {
          t = Transpose<perm=[1,0,2]>(x)
          r = Relu(t)
          u = Unsqueeze(r, axes)
          q = Squeeze(u, axes)
          f = Flatten<axis=1>(q)
          y = Add(f, B)
        }""",
        dict(axes=np.array([0], dtype=np.int64), B=_f32(rng, 3, 8)),
        opset=15,
    )
    _assert_sound(m, np.random.default_rng(14), (2, 3, 4))


def test_symbol_cap_consolidation_stays_sound_and_never_tightens():
    rng = np.random.default_rng(5)
    m = _mlp(rng)
    full = Z.propagate(m, BOX)
    capped = _assert_sound(m, np.random.default_rng(15), (1, 16), max_symbols=6)
    assert any("symbol cap (6) reached" in n for n in capped.notes)
    assert not any("symbol cap" in n for n in full.notes)
    for name in ("r", "y"):
        flo, fhi = full.bounds(name)
        clo, chi = capped.bounds(name)
        assert np.all(clo <= flo + 1e-9) and np.all(chi >= fhi - 1e-9)


# ---- usefulness: real onnxsim output ------------------------------------------


def _product(a, b):
    def prefixed(m, p):
        g = m.graph
        keep = {i.name for i in g.input} - {i.name for i in g.initializer}
        ren = lambda n: n if n in keep or n == "" else p + n  # noqa: E731
        nodes = []
        for k, n in enumerate(g.node):
            c = onnx.NodeProto()
            c.CopyFrom(n)
            c.input[:] = [ren(x) for x in n.input]
            c.output[:] = [ren(x) for x in n.output]
            c.name = p + (n.name or f"n{k}")
            nodes.append(c)
        inits = []
        for t in g.initializer:
            c = onnx.TensorProto()
            c.CopyFrom(t)
            c.name = ren(t.name)
            inits.append(c)
        return nodes, inits, ren(g.output[0].name), g.input[0], g.output[0]

    na, ia, oa, inp, outa = prefixed(a, "a_")
    nb, ib, ob, _, _ = prefixed(b, "b_")
    sub = onnx.helper.make_node("Sub", [oa, ob], ["diff"], name="diff")
    out = onnx.helper.make_tensor_value_info(
        "diff",
        outa.type.tensor_type.elem_type,
        [d.dim_value for d in outa.type.tensor_type.shape.dim],
    )
    g = onnx.helper.make_graph(na + nb + [sub], "product", [inp], [out], ia + ib)
    m = onnx.helper.make_model(g, opset_imports=list(a.opset_import))
    m.ir_version = a.ir_version
    return m


def _interval_baseline(orig, simp):
    res = interval.propagate(_product(orig, simp), BOX)
    if "diff" not in res.intervals:
        return np.inf
    lo, hi = res.intervals["diff"]
    return float(max(np.abs(lo).max(), np.abs(hi).max()))


@pytest.mark.parametrize(
    "build", [_conv_bn_relu, _mlp], ids=["conv_bn_relu", "mlp_gemm"]
)
def test_real_simplify_is_proved_equal_and_far_tighter_than_intervals(build):
    orig = build(np.random.default_rng(21))
    simp = _simplify(orig)
    assert {n.op_type for n in simp.graph.node} != {n.op_type for n in orig.graph.node}
    d = Z.bound_difference(orig, simp, BOX)
    assert d.bounded and d.worst < 1e-5, d.worst
    assert d.within(atol=1e-5, rtol=1e-4) and Z.proves_equal(orig, simp, BOX)
    baseline = _interval_baseline(orig, simp)
    # The baseline is large (or unbounded) because each branch's Relu relaxation is independent.
    assert baseline > 1.0, baseline
    ratio = baseline / max(d.worst, 1e-300)
    assert ratio > 1e4, f"zonotope {d.worst:.3e} vs interval {baseline:.3e}"
    # And the bound really bounds float32 execution of both models.
    rng = np.random.default_rng(31)
    shape = tuple(d_.dim_value for d_ in orig.graph.input[0].type.tensor_type.shape.dim)
    sa = ort.InferenceSession(orig.SerializeToString())
    sb = ort.InferenceSession(simp.SerializeToString())
    worst_seen = 0.0
    for _ in range(40):
        x = rng.uniform(-1, 1, shape).astype(np.float32)
        worst_seen = max(
            worst_seen,
            float(np.abs(sa.run(None, {"x": x})[0] - sb.run(None, {"x": x})[0]).max()),
        )
    assert worst_seen <= d.worst + 1e-5  # float32 evaluation noise budget


def test_pairing_is_what_makes_it_tight():
    orig = _conv_bn_relu(np.random.default_rng(22))
    simp = _simplify(orig)
    space = Z._Space()
    inputs = Z._input_zonotopes([orig, simp], BOX, space)
    ev_a = Z._Evaluator(orig, space, 4096)
    ev_b = Z._Evaluator(simp, space, 4096)  # independent fresh symbols: no pairing
    za, zb = ev_a._as_z(ev_a.run(inputs)["y"]), ev_b._as_z(ev_b.run(inputs)["y"])
    lo, hi = Z._add(za, zb, -1.0).bounds()
    unpaired = float(max(np.abs(lo).max(), np.abs(hi).max()))
    paired = Z.bound_difference(orig, simp, BOX).worst
    assert unpaired > 1.0 and paired < 1e-5 and unpaired / paired > 1e5


# ---- wrong rewrites are not proved --------------------------------------------


def test_wrong_bias_is_not_proved_and_bound_is_clearly_large():
    orig = _conv_bn_relu(np.random.default_rng(23), relu=False)
    bad = onnx.ModelProto()
    bad.CopyFrom(orig)
    for t in bad.graph.initializer:
        if t.name == "be":
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t) + 0.25, "be"))
    d = Z.bound_difference(orig, bad, BOX)
    assert not d.within() and not Z.proves_equal(orig, bad, BOX)
    assert d.worst >= 0.2  # the injected 0.25 shift is certified, not hidden


def test_wrong_bias_behind_a_relu_is_not_proved():
    orig = _conv_bn_relu(np.random.default_rng(24))
    bad = onnx.ModelProto()
    bad.CopyFrom(orig)
    for t in bad.graph.initializer:
        if t.name == "be":
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t) + 0.25, "be"))
    d = Z.bound_difference(orig, bad, BOX)
    assert (
        not d.within() and d.worst > 0.05
    )  # pairing must not let the Relu hide the shift


def _conv_with_weights(w):
    return _model(
        "m (float[1,3,6,6] x) => (float[1,4,4,4] y) { y = Conv(x, W, B) }",
        dict(W=w, B=np.zeros(4, np.float32)),
    )


def test_rgb_bgr_swap_is_not_proved():
    w = _f32(np.random.default_rng(25), 4, 3, 3, 3)
    box = {"x": (0.0, 1.0)}
    swapped = Z.bound_difference(
        _conv_with_weights(w), _conv_with_weights(w[:, ::-1].copy()), box
    )
    assert not swapped.within() and swapped.worst > 0.5
    sym = np.repeat(w[:, :1], 3, axis=1)  # identical channels: swapping them is a no-op
    harmless = Z.bound_difference(
        _conv_with_weights(sym), _conv_with_weights(sym[:, ::-1].copy()), box
    )
    assert harmless.within()


def test_unrelated_models_are_sound_whatever_the_pairing():
    rng = np.random.default_rng(26)
    a = _mlp(rng)
    b = _model(
        """
        m (float[1,16] x) => (float[1,4] y) {
          a = MatMul(x, W1)
          r = Sigmoid(a)
          c = MatMul(r, W2)
          y = Add(c, B2)
        }""",
        dict(W1=_f32(rng, 16, 8), W2=_f32(rng, 8, 4), B2=_f32(rng, 4)),
    )
    d = Z.bound_difference(a, b, BOX)
    sa, sb = (ort.InferenceSession(m.SerializeToString()) for m in (a, b))
    for _ in range(60):
        x = rng.uniform(-1, 1, (1, 16)).astype(np.float32)
        diff = np.abs(sa.run(None, {"x": x})[0] - sb.run(None, {"x": x})[0])
        assert np.all(diff <= d.max_abs["y"] + SLACK)


# ---- fallback and unbounded handling ------------------------------------------


def test_op_without_a_rule_falls_back_to_an_interval_box_with_a_note():
    rng = np.random.default_rng(27)
    m = _model(
        """
        m (float[1,6] x) => (float[1,3] y) {
          a = MatMul(x, W)
          s = Softmax<axis=1>(a)
          y = Add(s, B)
        }""",
        dict(W=_f32(rng, 6, 3), B=_f32(rng, 3)),
    )
    res = _assert_sound(m, np.random.default_rng(16), (1, 6))
    assert any("precision lost at Softmax" in n for n in res.notes)
    lo, hi = res.bounds("s")
    assert lo.min() >= -1e-9 and hi.max() <= 1.0 + 1e-6


def test_unbounded_operand_propagates_as_unbounded_not_wrong():
    m = _model("m (float[1,6] x) => (float[1,6] y) { a = Abs(x)  y = Relu(a) }")
    res = Z.propagate(m, {"x": (-1.0, 1.0)})
    assert any("precision lost at Abs" in n or "unbounded" in n for n in res.notes)
    lo, hi = res.bounds("y")
    assert not np.isnan(lo).any() and not np.isnan(hi).any()
    assert np.all(
        np.isposinf(hi)
    )  # nothing is known about y, so it must not claim a finite bound
    # Honest limitation: even two *identical* models cannot be proved equal through an op with
    # no rule, because an unbounded tensor carries no symbols that could cancel. The answer is
    # inf (sound), not a guess.
    d = Z.bound_difference(m, m, {"x": (-1.0, 1.0)})
    assert d.worst == np.inf and not d.within()
    assert any("unbounded operand reaches" in n for n in d.notes)


def test_unbounded_input_gives_infinite_bound_no_nan_no_false_proof():
    orig = _conv_bn_relu(np.random.default_rng(28))
    simp = _simplify(orig)
    d = Z.bound_difference(orig, simp)  # no input_ranges at all
    assert d.worst == np.inf and not d.bounded and not d.within()
    assert all(not np.isnan(v).any() for v in d.max_abs.values())
    assert not Z.proves_equal(orig, simp, {"x": (-np.inf, np.inf)})
    with pytest.raises(ValueError):
        Z.propagate(orig)


def test_annotated_range_is_used_without_explicit_input_ranges():
    from onnxsim import ranges

    orig = _conv_bn_relu(np.random.default_rng(29))
    simp = _simplify(orig)
    ranges.set_range(orig, "x", -1.0, 1.0)
    assert Z.proves_equal(orig, simp)


def test_mismatched_graph_io_raises():
    a = _model("m (float[4] x) => (float[4] y) { y = Relu(x) }")
    b = _model("m (float[4] z) => (float[4] y) { y = Relu(z) }")
    with pytest.raises(ValueError, match="inputs differ"):
        Z.bound_difference(a, b, {"x": (0.0, 1.0)})


# ---- direction vs Z3 -----------------------------------------------------------


def test_zonotope_never_proves_what_z3_refutes():
    pytest.importorskip("z3", reason="needs the 'verify' extra (z3-solver)")
    from onnxsim import certify

    rng = np.random.default_rng(30)
    good = _conv_bn_relu(rng, relu=False)  # linear: Z3 handles it exactly
    folded = _simplify(good)
    bad = onnx.ModelProto()
    bad.CopyFrom(good)
    for t in bad.graph.initializer:
        if t.name == "be":
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t) + 0.25, "be"))
    for orig, other in ((good, folded), (good, bad)):
        z_ok = Z.proves_equal(orig, other, BOX)
        report = certify.certify(orig, other, input_ranges=BOX)
        z3_refuted = any(s == certify.REFUTED for s in report.outputs.values())
        assert not (z_ok and z3_refuted)
        if z3_refuted:
            assert not z_ok
    # and on the exact fold both agree it is equal
    assert Z.proves_equal(good, folded, BOX)
    assert certify.certify(good, folded, input_ranges=BOX).ok


def test_symbol_cap_costs_precision_and_says_so():
    # Two stacked Conv->BN->Relu at 16x16: the default cap keeps the shared symbols (tight);
    # a cap below the symbol count merges them into boxes (sound, but loose) and reports it.
    rng = np.random.default_rng(32)
    f = lambda *s: _f32(rng, *s)  # noqa: E731
    u = lambda n: rng.uniform(0.5, 1.5, n).astype(np.float32)  # noqa: E731
    m = _model(
        """
        m (float[1,3,16,16] x) => (float[1,8,12,12] y) {
          c1 = Conv(x, W1, B1)
          b1 = BatchNormalization<epsilon=1e-5>(c1, g1, be1, mu1, var1)
          r1 = Relu(b1)
          c2 = Conv(r1, W2, B2)
          b2 = BatchNormalization<epsilon=1e-5>(c2, g2, be2, mu2, var2)
          y = Relu(b2)
        }""",
        dict(
            W1=f(8, 3, 3, 3), B1=f(8), g1=u(8), be1=f(8), mu1=f(8), var1=u(8),
            W2=f(8, 8, 3, 3) * 0.2, B2=f(8), g2=u(8), be2=f(8), mu2=f(8), var2=u(8),
        ),
    )  # fmt: skip
    simp = _simplify(m)
    tight = Z.bound_difference(m, simp, BOX)
    assert tight.worst < 1e-3 and not any("symbol cap" in n for n in tight.notes)
    loose = Z.bound_difference(m, simp, BOX, max_symbols=1024)
    assert loose.worst > 100 * tight.worst  # sound but much looser
    assert any("symbol cap (1024) reached" in n for n in loose.notes)
