"""Tests for onnxsim.crown (CROWN / alpha-CROWN bounds).

What is checked, in order of how much each one would catch:

* Exactness on affine nets: for a model that is affine in its input, CROWN must equal
  the true min/max, which pins down every backward rule (Conv stride/pad/dilation/group,
  BatchNorm, AveragePool, Transpose, Gemm attributes, Mul/Sub/Add, Reshape) without any
  relaxation to hide behind.
* Soundness: sample inputs in the box, run onnxruntime with *every* intermediate tensor
  exposed, and require each observed value inside the bounds.
* Ordering: CROWN is never looser than interval bounds and strictly tighter where
  correlations matter; alpha-CROWN is never looser than CROWN.
* The nonlinear relaxations on a dense grid, the interval-leaf fallback, unbounded inputs,
  ``verify_output_ranges`` and the quantization wrapper.
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import crown
from onnxsim import interval as I
from onnxsim import ranges as R

HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="alpha-CROWN needs torch")


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


def _all_tensors(model, feeds):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = [o for n in m.graph.node for o in n.output if o]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in produced)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip(produced, sess.run(None, feeds)))


def _produced(model):
    return [o for n in model.graph.node for o in n.output if o]


def _mlp(rng, act="Relu", widths=(6, 10, 8, 3)):
    body = [f"m (float[1,{widths[0]}] x) => (float[1,{widths[-1]}] y) {{"]
    inits, prev = {}, "x"
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        inits[f"W{i}"], inits[f"B{i}"] = _f32(rng, a, b), _f32(rng, b)
        last = i == len(widths) - 2
        body.append(f"  g{i} = Gemm({prev}, W{i}, B{i})")
        prev = f"g{i}"
        if not last:
            kind = act if isinstance(act, str) else act[i % len(act)]
            body.append(f"  a{i} = {kind}(g{i})")
            prev = f"a{i}"
    body.append(f"  y = Identity({prev})\n}}")
    return _model("\n".join(body), inits)


def _conv_bn_relu_net(rng):
    k = 4
    return _model(
        """
        m (float[1,3,8,8] x) => (float[1,5] y) {
          c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
          b1 = BatchNormalization<epsilon=1e-5>(c1, g, be, mu, var)
          r1 = Relu(b1)
          c2 = Conv<strides=[2,2], pads=[1,1,1,1]>(r1, W2, B2)
          r2 = Relu(c2)
          p = GlobalAveragePool(r2)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W3, B3)
        }""",
        dict(
            W1=_f32(rng, k, 3, 3, 3), B1=_f32(rng, k), g=rng.uniform(0.5, 1.5, k).astype(np.float32),
            be=_f32(rng, k), mu=_f32(rng, k), var=rng.uniform(0.5, 2, k).astype(np.float32),
            W2=_f32(rng, 6, k, 3, 3), B2=_f32(rng, 6), W3=_f32(rng, 5, 6), B3=_f32(rng, 5),
        ),
    )  # fmt: skip


def _residual_net(rng):
    return _model(
        """
        m (float[1,3,6,6] x) => (float[1,4] y) {
          c = Conv<pads=[1,1,1,1]>(x, W1, B1)
          r = Relu(c)
          s = Add(c, r)
          d = Conv<pads=[1,1,1,1]>(s, W2, B2)
          t = Tanh(d)
          p = AveragePool<kernel_shape=[2,2], strides=[2,2]>(t)
          f = Flatten(p)
          y = Gemm(f, W3, B3)
        }""",
        dict(W1=_f32(rng, 4, 3, 3, 3), B1=_f32(rng, 4), W2=_f32(rng, 4, 4, 3, 3), B2=_f32(rng, 4),
             W3=_f32(rng, 36, 4), B3=_f32(rng, 4)),
    )  # fmt: skip


def _check_sound(model, box, got, rng, n=60, slack=1e-4):
    (name,) = {t.name for t in model.graph.input} - {
        t.name for t in model.graph.initializer
    }
    shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    lo, hi = box
    for _ in range(n):
        x = rng.uniform(lo, hi, shape).astype(np.float32)
        for tname, val in _all_tensors(model, {name: x}).items():
            tb = got[tname]
            pad = slack * (1.0 + np.abs(val))
            assert np.all(val >= tb.lo - pad) and np.all(val <= tb.hi + pad), (
                f"{tname} escaped ({tb.method})"
            )


# ---- exactness on affine nets -------------------------------------------------


def _affine_net(rng):
    return _model(
        """
        m (float[1,4,7,7] x) => (float[1,5] y) {
          c1 = Conv<strides=[2,2], pads=[1,1,1,1]>(x, W1, B1)
          b1 = BatchNormalization<epsilon=1e-5>(c1, g, be, mu, var)
          c2 = Conv<group=2, dilations=[2,2], pads=[2,2,2,2]>(b1, W2)
          p = AveragePool<kernel_shape=[2,2], strides=[2,2]>(c2)
          s = Sub(p, Cs)
          t = Transpose<perm=[0,2,3,1]>(s)
          f = Flatten(t)
          h = Mul(f, half)
          d = Sub(f, h)
          z = Gemm<transB=1, alpha=0.7, beta=1.3>(d, Wg, Bg)
          y = Add(z, Cadd)
        }""",
        dict(
            W1=_f32(rng, 6, 4, 3, 3), B1=_f32(rng, 6), g=rng.uniform(0.5, 1.5, 6).astype(np.float32),
            be=_f32(rng, 6), mu=_f32(rng, 6), var=rng.uniform(0.5, 2, 6).astype(np.float32),
            W2=_f32(rng, 6, 3, 3, 3), Cs=_f32(rng, 1, 6, 1, 1), half=np.float32([0.5]),
            Wg=_f32(rng, 5, 24), Bg=_f32(rng, 5), Cadd=_f32(rng, 1, 5),
        ),
    )  # fmt: skip


def test_crown_is_exact_on_an_affine_net_and_interval_is_not():
    rng = np.random.default_rng(0)
    model = _affine_net(rng)
    box = {"x": (-1.0, 2.0)}
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    n = 4 * 7 * 7
    base = (
        sess.run(None, {"x": np.zeros((1, 4, 7, 7), np.float32)})[0]
        .reshape(-1)
        .astype(np.float64)
    )
    jac = np.empty((n, base.size))
    for i in range(n):
        e = np.zeros(n, np.float32)
        e[i] = 1.0
        jac[i] = sess.run(None, {"x": e.reshape(1, 4, 7, 7)})[0].reshape(-1) - base
    lo_x, hi_x = -1.0, 2.0
    exact_lo = base + np.minimum(jac * lo_x, jac * hi_x).sum(0)
    exact_hi = base + np.maximum(jac * lo_x, jac * hi_x).sum(0)

    got = crown.bounds(model, box, method="crown")["y"]
    ibp = I.propagate(model, box).intervals["y"]
    np.testing.assert_allclose(got.lo.reshape(-1), exact_lo, rtol=1e-4, atol=2e-4)
    np.testing.assert_allclose(got.hi.reshape(-1), exact_hi, rtol=1e-4, atol=2e-4)
    # the Sub(f, 0.5 f) correlation makes plain intervals strictly looser
    assert float((ibp[1] - ibp[0]).mean()) > 1.2 * float((got.hi - got.lo).mean())


# ---- soundness on nonlinear nets ----------------------------------------------


@pytest.mark.parametrize(
    "build, box",
    [
        (lambda r: _mlp(r), (-1.0, 1.0)),
        (lambda r: _mlp(r, act="Sigmoid"), (-2.0, 2.0)),
        (lambda r: _mlp(r, act=("Tanh", "Relu")), (-1.5, 1.5)),
        (_conv_bn_relu_net, (-1.0, 1.0)),
        (_residual_net, (-1.0, 1.0)),
    ],
    ids=["mlp-relu", "mlp-sigmoid", "mlp-tanh-relu", "conv-bn-relu", "residual-tanh"],
)
def test_bounds_enclose_every_intermediate_tensor(build, box):
    rng = np.random.default_rng(1)
    model = build(rng)
    (iname,) = {t.name for t in model.graph.input} - {
        t.name for t in model.graph.initializer
    }
    got = crown.bounds(model, {iname: box}, output=_produced(model), method="crown")
    _check_sound(model, box, got, rng)


@needs_torch
def test_alpha_bounds_enclose_every_intermediate_tensor():
    rng = np.random.default_rng(2)
    model = _mlp(rng)
    got = crown.bounds(
        model, {"x": (-1.0, 1.0)}, output=_produced(model), method="alpha"
    )
    assert any(b.method == "alpha" for b in got.values())
    _check_sound(model, (-1.0, 1.0), got, rng)


# ---- ordering: ibp >= crown >= alpha -----------------------------------------------


def _width(tb):
    return float((tb.hi - tb.lo).mean())


def test_crown_never_looser_than_interval_and_tighter_on_relu_nets():
    for build in (lambda r: _mlp(r), _conv_bn_relu_net, _residual_net):
        model = build(np.random.default_rng(3))
        (iname,) = {t.name for t in model.graph.input} - {
            t.name for t in model.graph.initializer
        }
        box = {iname: (-1.0, 1.0)}
        names = _produced(model)
        ibp = crown.bounds(model, box, output=names, method="ibp")
        cr = crown.bounds(model, box, output=names, method="crown")
        for n in names:
            assert np.all(cr[n].lo >= ibp[n].lo) and np.all(cr[n].hi <= ibp[n].hi), n
    model = _mlp(np.random.default_rng(3))
    box = {"x": (-1.0, 1.0)}
    ratio = _width(crown.bounds(model, box, method="crown")["y"]) / _width(
        crown.bounds(model, box, method="ibp")["y"]
    )
    assert ratio < 0.8, (
        ratio
    )  # a 3-layer ReLU MLP: a measurable gain, not a rounding artifact


@needs_torch
def test_alpha_never_looser_than_crown_and_tighter_somewhere():
    model = _mlp(np.random.default_rng(4))
    box = {"x": (-1.0, 1.0)}
    cr = crown.bounds(model, box, output=["g1", "y"], method="crown")
    al = crown.bounds(model, box, output=["g1", "y"], method="alpha")
    for n in ("g1", "y"):
        assert np.all(al[n].lo >= cr[n].lo) and np.all(al[n].hi <= cr[n].hi), n
    assert _width(al["y"]) < 0.95 * _width(cr["y"])


@needs_torch
def test_alpha_on_a_conv_net_is_sound_and_not_looser():
    rng = np.random.default_rng(5)
    model = _conv_bn_relu_net(rng)
    box = {"x": (-1.0, 1.0)}
    cr = crown.bounds(model, box, method="crown")["y"]
    al = crown.bounds(model, box, method="alpha", alpha_iters=15)["y"]
    assert np.all(al.lo >= cr.lo) and np.all(al.hi <= cr.hi)
    _check_sound(
        model,
        (-1.0, 1.0),
        {**crown.bounds(model, box, output=_produced(model)), "y": al},
        rng,
        n=20,
    )


def test_alpha_without_torch_raises_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="torch"):
        crown.bounds(_mlp(np.random.default_rng(0)), {"x": (-1.0, 1.0)}, method="alpha")
    # and the numpy methods do not need it
    assert (
        crown.bounds(_mlp(np.random.default_rng(0)), {"x": (-1.0, 1.0)})["y"].method
        == "crown"
    )


def test_invalid_method_and_unknown_tensor_are_rejected():
    model = _mlp(np.random.default_rng(0))
    with pytest.raises(ValueError, match="method"):
        crown.bounds(model, {"x": (-1.0, 1.0)}, method="nope")
    with pytest.raises(ValueError, match="unknown"):
        crown.bounds(model, {"x": (-1.0, 1.0)}, output="does_not_exist")


def test_residual_branch_correlation_is_exploited_when_tanh_does_not_saturate():
    # Add(c, relu(c)) reuses the same pre-activation twice: intervals treat the operands as
    # independent, CROWN accumulates both consumers' coefficients on the shared tensor.
    model = _residual_net(np.random.default_rng(3))
    box = {"x": (-0.05, 0.05)}
    ibp = crown.bounds(model, box, method="ibp")["y"]
    cr = crown.bounds(model, box, method="crown")["y"]
    assert np.all(cr.lo >= ibp.lo) and np.all(cr.hi <= ibp.hi)
    assert _width(cr) < 0.9 * _width(ibp)
    _check_sound(
        model,
        (-0.05, 0.05),
        crown.bounds(model, box, output=_produced(model)),
        np.random.default_rng(3),
        n=30,
    )


# ---- fallback, unbounded inputs ----------------------------------------------------


def test_op_without_a_rule_falls_back_to_its_interval_and_stays_sound():
    rng = np.random.default_rng(6)
    model = _model(
        """
        m (float[1,6] x) => (float[1,3] y) {
          a = Gemm(x, W0, B0)
          lo = Constant<value=float {-0.5}>()
          hi = Constant<value=float {0.5}>()
          c = Clip(a, lo, hi)
          g = Gemm(c, W1, B1)
          r = Relu(g)
          y = Gemm(r, W2, B2)
        }""",
        dict(W0=_f32(rng, 6, 8), B0=_f32(rng, 8), W1=_f32(rng, 8, 8), B1=_f32(rng, 8),
             W2=_f32(rng, 8, 3), B2=_f32(rng, 3)),
    )  # fmt: skip
    an = crown._Analyzer(model, {"x": (-1.0, 1.0)})
    assert "c" in an.leaf  # Clip has no backward rule: an interval leaf, not a guess
    box = {"x": (-1.0, 1.0)}
    got = crown.bounds(model, box, output=_produced(model))
    ibp = crown.bounds(model, box, output=_produced(model), method="ibp")
    assert np.all(np.isfinite(got["y"].lo)) and np.all(got["y"].hi <= ibp["y"].hi)
    _check_sound(model, (-1.0, 1.0), got, rng)


def test_unbounded_input_gives_unbounded_not_wrong():
    rng = np.random.default_rng(7)
    model = _mlp(rng)
    got = crown.bounds(model, None, output=_produced(model))
    assert np.all(np.isinf(got["g0"].lo)) and np.all(np.isinf(got["g0"].hi))
    assert np.all(
        got["a0"].lo >= 0.0
    )  # a Relu output is bounded below whatever the input
    finite = crown.bounds(
        model, {"x": (-1.0, 1.0)}
    )  # the same model with a finite box is bounded
    assert np.all(np.isfinite(finite["y"].lo)) and np.all(np.isfinite(finite["y"].hi))


# ---- the nonlinear relaxations, directly --------------------------------------------


@pytest.mark.parametrize("kind", ["Sigmoid", "Tanh"])
def test_scurve_lines_are_valid_on_a_dense_grid(kind):
    rng = np.random.default_rng(8)
    f = crown._SCURVES[kind][0]
    lo = np.concatenate(
        [rng.uniform(-8, 8, 400), [-1e30, -3.0, 0.0, 0.0, 2.0, -5.0, 1e-9]]
    )
    width = np.concatenate(
        [rng.exponential(2.0, 400), [1e30, 0.0, 0.0, 6.0, 1e-3, 5.0, 0.0]]
    )
    hi = np.minimum(lo + width, 1e30)
    r = crown._scurve_relax(kind, lo, hi)
    for i in range(lo.size):
        xs = np.linspace(max(lo[i], -60.0), min(hi[i], 60.0), 4001)
        if xs.size == 0 or lo[i] > 60 or hi[i] < -60:
            continue
        fx = f(xs)
        assert np.all(r["a_l"][i] * xs + r["b_l"][i] <= fx + 1e-12), (
            kind,
            lo[i],
            hi[i],
            "lower",
        )
        assert np.all(r["a_u"][i] * xs + r["b_u"][i] >= fx - 1e-12), (
            kind,
            lo[i],
            hi[i],
            "upper",
        )


def test_relu_triangle_is_valid_and_exact_when_stable():
    rng = np.random.default_rng(9)
    lo = rng.uniform(-3, 1, 200)
    hi = lo + rng.exponential(1.5, 200)
    r = crown._relu_relax(lo, hi)
    for i in range(lo.size):
        xs = np.linspace(lo[i], hi[i], 501)
        a_l = r["a_l_stable"][i] + r["unstable"][i] * r["alpha0"][i]
        assert np.all(a_l * xs + r["b_l"][i] <= np.maximum(xs, 0) + 1e-12)
        assert np.all(r["a_u"][i] * xs + r["b_u"][i] >= np.maximum(xs, 0) - 1e-12)
        if r["unstable"][i] == 0:  # stable: both lines are the function itself
            assert np.allclose(a_l * xs + r["b_l"][i], np.maximum(xs, 0))


# ---- verify_output_ranges ----------------------------------------------------------


def test_verify_output_ranges_proves_true_and_rejects_too_tight():
    model = _mlp(np.random.default_rng(10))
    box = {"x": (-1.0, 1.0)}
    hull = crown.bounds(model, box)["y"].hull()
    R.set_range(model, "y", hull[0] - 1.0, hull[1] + 1.0)
    ok = crown.verify_output_ranges(model, box)
    assert ok["y"].proved and ok["y"].hull == pytest.approx(hull)
    assert ok["y"].annotated == (
        pytest.approx(hull[0] - 1.0),
        pytest.approx(hull[1] + 1.0),
    )

    R.set_range(
        model, "y", hull[0] + 0.5 * (hull[1] - hull[0]) * 0.2, hull[1]
    )  # cuts the proven low end
    bad = crown.verify_output_ranges(model, box)
    assert not bad["y"].proved and bad["y"].hull == pytest.approx(hull)


def test_verify_output_ranges_handles_one_sided_and_missing_annotations():
    model = _mlp(np.random.default_rng(11))
    assert crown.verify_output_ranges(model, {"x": (-1.0, 1.0)}) == {}
    R.set_range(model, "y", None, 1e6)
    assert crown.verify_output_ranges(model, {"x": (-1.0, 1.0)})["y"].proved


def test_verify_output_ranges_uses_the_models_own_input_annotation():
    model = _mlp(np.random.default_rng(12))
    R.set_range(model, "x", -1.0, 1.0)
    hull = crown.bounds(model)[
        "y"
    ].hull()  # the annotation is the box when none is passed
    assert np.isfinite(hull[0]) and np.isfinite(hull[1])
    R.set_range(model, "y", hull[0] - 1e-3, hull[1] + 1e-3)
    assert crown.verify_output_ranges(model)["y"].proved


# ---- quantization wrapper -----------------------------------------------------------


def _conv_relu_conv(rng):
    return _model(
        """
        m (float[1,3,8,8] x) => (float[1,4,8,8] y) {
          c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)
          r1 = Relu(c1)
          s = Sub(c1, r1)
          c2 = Conv<pads=[1,1,1,1]>(s, W2, B2)
          y = Relu(c2)
        }""",
        dict(
            W1=_f32(rng, 6, 3, 3, 3),
            B1=_f32(rng, 6),
            W2=_f32(rng, 4, 6, 3, 3),
            B2=_f32(rng, 4),
        ),
    )


def test_tight_quantization_bounds_are_never_worse_and_better_with_correlation():
    model = _conv_relu_conv(np.random.default_rng(13))
    rows = crown.quantization_bounds_tight(model, {"x": (-1.0, 1.0)})
    assert len(rows) == 2
    for r in rows:
        assert r.tight.acc_bound <= r.interval.acc_bound
        assert r.tight.max_abs_error <= r.interval.max_abs_error * (1 + 1e-9)
        assert r.tight.act_range[0] >= r.interval.act_range[0] - 1e-9
        assert r.tight.act_range[1] <= r.interval.act_range[1] + 1e-9
    first, second = rows
    assert first.acc_tightening == pytest.approx(
        1.0
    )  # its input is the graph input: nothing to tighten
    # Sub(c1, relu(c1)): the interval range treats the two operands as independent, CROWN does not
    assert second.tight.act_range[0] > second.interval.act_range[0] + 1.0
    assert second.tight.act_range[1] < second.interval.act_range[1]
    assert second.acc_tightening > 1.1 and second.error_tightening > 1.3
    text = crown.format_quantization_comparison(rows)
    assert "acc crown" in text and "Conv" not in text.splitlines()[0]
    assert text.count("\n") == 2


def test_tight_quantization_bounds_skip_layers_without_a_bounded_input():
    model = _conv_relu_conv(np.random.default_rng(14))
    assert crown.quantization_bounds_tight(model) == []
