"""Tests for onnxsim.taylor: Taylor models with certified remainders, and the mean-value form.

Layers, from the bottom up: (1) the scalar functions' derivative formulas and critical points
against dense grids; (2) the Lagrange remainder of each function against the true remainder;
(3) polynomial range and product soundness, pointwise; (4) whole models against onnxruntime with
*every* intermediate tensor exposed (the property that matters); (5) tightness, fallbacks,
unbounded inputs, the mean-value form and ``compare_methods``.

Models are built with ``onnx.parser`` (CLAUDE.md) and numpy weights attached programmatically.
Nothing here calls ``onnxsim.simplify``.
"""

import contextlib
import io
import math
import pathlib
import re

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

from onnxsim import interval, ranges
from onnxsim import taylor as T


def _model(body, initializer=None, opset=17, ir_version=9):
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


def _run_all(model, feed):
    """Every node output of ``model`` for ``feed``: onnxruntime, else the reference evaluator."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = [o for n in m.graph.node for o in n.output if o]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in produced)
    try:
        sess = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        vals = sess.run(None, feed)
    except Exception:
        vals = ReferenceEvaluator(m).run(None, feed)
    return dict(zip(produced, vals))


def _box(model, input_ranges):
    out = {}
    for vi in model.graph.input:
        if vi.name in {t.name for t in model.graph.initializer}:
            continue
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        lo, hi = (
            np.broadcast_to(np.asarray(b, dtype=np.float64), shape)
            for b in input_ranges[vi.name]
        )
        out[vi.name] = (lo, hi)
    return out


def _assert_sound(
    model, input_ranges, degree=2, n_random=40, n_vertices=24, slack=1e-4, seed=0
):
    """Every observed value of every tensor lies inside its Taylor bounds."""
    res = T.propagate(model, input_ranges, degree=degree)
    rng = np.random.default_rng(seed)
    box = _box(model, input_ranges)
    elem = {i.name: i.type.tensor_type.elem_type for i in model.graph.input}
    checked = 0
    for k in range(n_random + n_vertices):
        feed = {}
        for name, (lo, hi) in box.items():
            u = (
                rng.random(lo.shape)
                if k < n_random
                else (rng.random(lo.shape) < 0.5).astype(float)
            )
            feed[name] = (lo + (hi - lo) * u).astype(
                onnx.helper.tensor_dtype_to_np_dtype(elem[name])
            )
        for name, val in _run_all(model, feed).items():
            val = np.asarray(val)
            if val.dtype.kind != "f" or name not in res.tensors:
                continue
            lo, hi = res.bounds(name)
            v = val.astype(np.float64)
            pad = slack * (1.0 + np.abs(v))
            assert np.all(np.isfinite(v))
            assert np.all(v >= lo - pad) and np.all(v <= hi + pad), (
                f"{name} escaped its Taylor bounds (degree {degree}): "
                f"observed [{v.min():.6g}, {v.max():.6g}] vs bounds [{np.min(lo):.6g}, {np.max(hi):.6g}]"
            )
            checked += 1
    assert checked > 0
    return res


def _width(res, name):
    lo, hi = res.bounds(name)
    return float(np.mean(hi - lo))


def _interval_width(model, input_ranges, name):
    lo, hi = interval.propagate(model, input_ranges).intervals[name]
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


# ---- (1) scalar functions ------------------------------------------------------------

_DOMAINS = {
    "exp": (-3.0, 3.0),
    "log": (0.2, 6.0),
    "sqrt": (0.2, 6.0),
    "recip": (0.3, 6.0),
    "rsqrt": (0.2, 6.0),
    "sigmoid": (-7.0, 7.0),
    "tanh": (-4.0, 4.0),
    "erf": (-3.5, 3.5),
}


@pytest.mark.parametrize("name", sorted(T._FUNCS))
def test_derivative_formulas_match_finite_differences(name):
    fn = T._FUNCS[name]
    lo, hi = _DOMAINS[name]
    x = np.linspace(lo, hi, 61)
    h = 1e-5
    for k in (1, 2, 3):
        prev = fn.f if k == 1 else fn.d[k - 1]
        numeric = (prev(x + h) - prev(x - h)) / (2 * h)
        np.testing.assert_allclose(fn.d[k](x), numeric, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("name", sorted(T._FUNCS))
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_derivative_range_is_exact_on_dense_grid(name, k):
    """``_fn_range`` encloses a dense grid and is within grid resolution of its extremes."""
    fn = T._FUNCS[name]
    dlo, dhi = _DOMAINS[name]
    rng = np.random.default_rng(1)
    for _ in range(25):
        a, b = np.sort(rng.uniform(dlo, dhi, 2))
        if b - a < 1e-3:
            continue
        grid = np.linspace(a, b, 20001)
        g = T._deriv(fn, k, grid)
        lo, hi = T._fn_range(fn, k, np.float64(a), np.float64(b))
        tol = 1e-9 * (1.0 + np.max(np.abs(g)))
        assert lo <= g.min() + tol and hi >= g.max() - tol, (name, k, a, b)
        assert hi - g.max() <= 1e-6 * (1.0 + abs(g.max())), (
            f"{name} k={k} upper bound not tight on [{a}, {b}]"
        )
        assert g.min() - lo <= 1e-6 * (1.0 + abs(g.min())), (
            f"{name} k={k} lower bound not tight on [{a}, {b}]"
        )


# ---- (2) Lagrange remainder ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(T._FUNCS))
@pytest.mark.parametrize("order", [1, 2])
def test_remainder_encloses_the_true_remainder(name, order):
    fn = T._FUNCS[name]
    dlo, dhi = _DOMAINS[name]
    rng = np.random.default_rng(2)
    kinds = {"definite": 0, "lagrange": 0}
    for trial in range(60):
        a, b = np.sort(rng.uniform(dlo, dhi, 2))
        if trial % 3 == 0:  # narrow boxes: the exact endpoint branch
            w = 0.05 + 0.3 * rng.random()
            m = rng.uniform(dlo + w, dhi - w)
            a, b = m - w, m + w
        c = rng.uniform(a, b)
        f0, f1, f2 = fn.f(c), fn.d[1](c), fn.d[2](c)
        rl, rh = T._taylor_remainder(
            fn,
            order,
            np.float64(c),
            np.float64(a),
            np.float64(b),
            np.float64(f0),
            np.float64(f1),
            np.float64(f2),
        )
        glo, ghi = T._fn_range(
            fn, order + 1, np.float64(min(a, c)), np.float64(max(b, c))
        )
        kinds["definite" if (glo >= 0 or ghi <= 0) else "lagrange"] += 1
        d = np.linspace(a - c, b - c, 4001)
        r = fn.f(c + d) - f0 - f1 * d - (f2 / 2.0 * d * d if order >= 2 else 0.0)
        tol = 1e-12 * (1.0 + np.max(np.abs(fn.f(c + d))))
        assert rl <= r.min() + tol and rh >= r.max() - tol, (name, order, a, b, c)
    # the branch that matters for tightness must actually be exercised for most functions
    assert kinds["definite"] > 0


def test_exact_endpoint_remainder_is_close_to_the_true_range():
    """On a sign-definite f''' the remainder interval is within float padding of the true range."""
    fn = T._FUNCS["recip"]
    c, a, b = 2.735, 1.47, 4.0
    f0, f1, f2 = fn.f(c), fn.d[1](c), fn.d[2](c)
    rl, rh = T._taylor_remainder(
        fn,
        2,
        np.float64(c),
        np.float64(a),
        np.float64(b),
        np.float64(f0),
        np.float64(f1),
        np.float64(f2),
    )
    d = np.linspace(a - c, b - c, 100001)
    r = fn.f(c + d) - f0 - f1 * d - f2 / 2.0 * d * d
    assert abs(rl - r.min()) < 1e-9 and abs(rh - r.max()) < 1e-9
    # and nowhere near the symmetric Lagrange bound sup|f'''| |D|^3 / 6 (about 0.43 here)
    assert rh - rl < 0.1


# ---- (3) polynomial range and products ------------------------------------------------


def _random_tm(rng, ctx, shape, rem=0.0):
    c = rng.standard_normal(shape)
    L = 0.3 * rng.standard_normal((ctx.n,) + shape)
    Q = 0.1 * rng.standard_normal((ctx.npairs,) + shape)
    r = np.full(shape, rem)
    return T.TaylorModel(
        c, L, Q, -r, 1.5 * r, np.full(shape, -1e9), np.full(shape, 1e9), ctx
    )


def test_poly_range_encloses_the_polynomial():
    rng = np.random.default_rng(3)
    ctx = T._Ctx(7, 4)
    tm = _random_tm(rng, ctx, (5,))
    lo, hi = tm.poly_range()
    for _ in range(3000):
        v = tm.evaluate(rng.uniform(-1, 1, ctx.n))
        assert np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12)
    for _ in range(500):  # vertices
        v = tm.evaluate(rng.choice([-1.0, 1.0], ctx.n))
        assert np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12)


def test_product_encloses_pointwise_including_remainders():
    rng = np.random.default_rng(4)
    ctx = T._Ctx(
        6, 3
    )  # symbols 3..5 have no quadratic slot: their products go to the remainder
    a, b = _random_tm(rng, ctx, (4,), rem=0.05), _random_tm(rng, ctx, (4,), rem=0.02)
    p = T._tm_mul(a, b)
    for _ in range(2000):
        eps = rng.uniform(-1, 1, ctx.n)
        ra = rng.uniform(a.rlo, a.rhi)
        rb = rng.uniform(b.rlo, b.rhi)
        true = (a.evaluate(eps) + ra) * (b.evaluate(eps) + rb)
        approx = p.evaluate(eps)
        assert np.all(true >= approx + p.rlo - 1e-10) and np.all(
            true <= approx + p.rhi + 1e-10
        )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("name", sorted(T._FUNCS))
def test_function_of_a_taylor_model_encloses_pointwise(name, degree):
    """For every noise value, f(x(eps)) lies in the polynomial plus the remainder."""
    fn = T._FUNCS[name]
    dlo, dhi = _DOMAINS[name]
    rng = np.random.default_rng(5)
    for trial in range(12):
        w = rng.uniform(0.05, 0.4) * (dhi - dlo) / 4
        m = rng.uniform(dlo + w + 0.1, dhi - w - 0.1)
        ctx = T._Ctx(2, 2 if degree == 2 else 0)
        # x = m + w*eps0 + 0.3 w*eps1 (two symbols, so the quadratic terms are exercised)
        L = np.zeros((2, 1))
        L[0, 0], L[1, 0] = w, 0.3 * w
        lo, hi = np.array([m - 1.3 * w]), np.array([m + 1.3 * w])
        x = T.TaylorModel(
            np.array([m]),
            L,
            np.zeros((ctx.npairs, 1)),
            np.zeros(1),
            np.zeros(1),
            lo,
            hi,
            ctx,
        )
        y = T._TMAlg(ctx, degree).func(name, x)
        for e0 in np.linspace(-1, 1, 41):
            for e1 in (-1.0, 0.0, 1.0):
                eps = np.array([e0, e1])
                true = fn.f(x.evaluate(eps))
                approx = y.evaluate(eps)
                assert np.all(true >= approx + y.rlo - 1e-10) and np.all(true <= approx + y.rhi + 1e-10), (
                    name, degree, m, w, e0, e1,
                )  # fmt: skip


def test_degree_two_is_tighter_than_degree_one_on_a_curved_function():
    """Compared on polynomial + remainder, i.e. before the interval box is intersected in:
    for ``exp`` of one variable the box is already exact, so ``range()`` would hide the effect."""
    widths = {}
    for deg, ctx in ((1, T._Ctx(1, 0)), (2, T._Ctx(1, 1))):
        x = T.TaylorModel(
            np.array([0.5]),
            np.array([[0.4]]),
            np.zeros((ctx.npairs, 1)),
            np.zeros(1),
            np.zeros(1),
            np.array([0.1]),
            np.array([0.9]),
            ctx,
        )
        y = T._TMAlg(ctx, deg).func("exp", x)
        plo, phi = y.poly_range()
        widths[deg] = float(np.max((phi + y.rhi) - (plo + y.rlo)))
    true_width = math.exp(0.9) - math.exp(0.1)
    assert widths[2] < widths[1]
    assert widths[2] < 1.05 * true_width and widths[1] > 1.05 * true_width


# ---- (4) whole models against onnxruntime ------------------------------------------------

_UNARY = {
    "Exp": (-1.0, 1.0),
    "Log": (0.5, 2.5),
    "Sqrt": (0.5, 2.5),
    "Reciprocal": (1.0, 3.0),
    "Sigmoid": (-2.0, 2.0),
    "Tanh": (-1.5, 1.5),
    "Erf": (-1.5, 1.5),
}


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("op", sorted(_UNARY))
def test_unary_ops_are_sound(op, degree):
    m = _model(f"m (float[2,5] x) => (float[2,5] y) {{ y = {op}(x) }}")
    _assert_sound(m, {"x": _UNARY[op]}, degree=degree)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("axis", [-1, 0])
def test_softmax_is_sound(axis, degree):
    m = _model(f"m (float[3,4] x) => (float[3,4] y) {{ y = Softmax<axis={axis}>(x) }}")
    _assert_sound(m, {"x": (-0.6, 0.4)}, degree=degree)


@pytest.mark.parametrize("approx", ["none", "tanh"])
@pytest.mark.parametrize("degree", [1, 2])
def test_gelu_is_sound(approx, degree):
    m = _model(
        f'm (float[1,6] x) => (float[1,6] y) {{ y = Gelu<approximate="{approx}">(x) }}',
        opset=20,
    )
    _assert_sound(m, {"x": (-0.4, 1.2)}, degree=degree)


def test_decomposed_gelu_is_sound_and_matches_the_fused_op():
    body = """
    m (float[1,6] x) => (float[1,6] y) {
      s = Constant<value=float {1.4142135}>()
      one = Constant<value=float {1.0}>()
      half = Constant<value=float {0.5}>()
      t = Div(x, s)
      e = Erf(t)
      u = Add(e, one)
      v = Mul(x, u)
      y = Mul(v, half)
    }"""
    m = _model(body, opset=17)
    res = _assert_sound(m, {"x": (-0.4, 1.2)})
    fused = T.propagate(
        _model("m (float[1,6] x) => (float[1,6] y) { y = Gelu(x) }", opset=20),
        {"x": (-0.4, 1.2)},
    )
    assert _width(res, "y") == pytest.approx(_width(fused, "y"), rel=0.05)


_CENTRES = np.linspace(-1.5, 1.5, 6).astype(np.float32)


@pytest.mark.parametrize("degree", [1, 2])
def test_layernorm_is_sound(degree):
    g = np.linspace(0.8, 1.2, 6).astype(np.float32)
    b = np.linspace(-0.1, 0.1, 6).astype(np.float32)
    m = _model(
        "m (float[2,6] x) => (float[2,6] y) { y = LayerNormalization<axis=-1, epsilon=1e-5>(x, g, b) }",
        {"g": g, "b": b},
    )
    _assert_sound(m, {"x": (_CENTRES - 0.1, _CENTRES + 0.1)}, degree=degree)


def test_rmsnorm_is_sound():
    g = np.linspace(0.8, 1.2, 6).astype(np.float32)
    m = _model(
        "m (float[2,6] x) => (float[2,6] y) { y = RMSNormalization<axis=-1, epsilon=1e-5>(x, g) }",
        {"g": g},
        opset=23,
    )
    _assert_sound(m, {"x": (_CENTRES - 0.1, _CENTRES + 0.1)})


def test_decomposed_layernorm_is_sound():
    rng = np.random.default_rng(6)
    gamma, beta = rng.uniform(0.8, 1.2, 6).astype(np.float32), _f32(rng, 6) * 0.1
    body = """
    m (float[2,6] x) => (float[2,6] y) {
      mu = ReduceMean<axes=[-1], keepdims=1>(x)
      d = Sub(x, mu)
      sq = Mul(d, d)
      var = ReduceMean<axes=[-1], keepdims=1>(sq)
      eps = Constant<value=float {1e-5}>()
      ve = Add(var, eps)
      sd = Sqrt(ve)
      n = Div(d, sd)
      gn = Mul(n, gamma)
      y = Add(gn, beta)
    }"""
    m = _model(body, {"gamma": gamma, "beta": beta}, opset=13)
    _assert_sound(m, {"x": (_CENTRES - 0.1, _CENTRES + 0.1)})


def test_mlp_with_relu_and_const_matmuls_is_sound():
    rng = np.random.default_rng(7)
    m = _model(
        """
        m (float[2,6] x) => (float[2,5] y) {
          a = MatMul(x, W1)
          b = Add(a, B1)
          r = Relu(b)
          c = Gemm<transB=1, alpha=0.5, beta=2.0>(r, W2, B2)
          t = Tanh(c)
          y = MatMul(W3, t)
        }""",
        {
            "W1": _f32(rng, 6, 8),
            "B1": _f32(rng, 8),
            "W2": _f32(rng, 5, 8),
            "B2": _f32(rng, 5),
            "W3": _f32(rng, 2, 2),
        },
    )
    _assert_sound(m, {"x": (-0.5, 0.5)})


@pytest.mark.parametrize("degree", [1, 2])
def test_conv_bn_sigmoid_pool_net_is_sound(degree):
    rng = np.random.default_rng(8)
    k = 4
    m = _model(
        """
        m (float[1,2,6,6] x) => (float[1,3] y) {
          c = Conv<pads=[1,1,1,1], strides=[1,1]>(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          s = Sigmoid(b)
          p = AveragePool<kernel_shape=[2,2], strides=[2,2]>(s)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W2, B2)
        }""",
        {
            "W": 0.5 * _f32(rng, k, 2, 3, 3), "B": _f32(rng, k),
            "g": rng.uniform(0.5, 1.5, k).astype(np.float32), "be": _f32(rng, k), "mu": _f32(rng, k),
            "var": rng.uniform(0.5, 2, k).astype(np.float32),
            "W2": _f32(rng, 3, k * 9), "B2": _f32(rng, 3),
        },
        opset=15,
    )  # fmt: skip
    _assert_sound(m, {"x": (-0.3, 0.3)}, degree=degree)


def test_grouped_strided_dilated_conv_is_sound():
    rng = np.random.default_rng(9)
    m = _model(
        "m (float[1,4,7,7] x) => (float[1,4,3,3] y) "
        "{ y = Conv<group=2, strides=[2,2], dilations=[2,2], pads=[1,1,1,1]>(x, W, B) }",
        {"W": _f32(rng, 4, 2, 3, 3), "B": _f32(rng, 4)},
    )
    res = _assert_sound(m, {"x": (-0.5, 0.5)})
    # an affine map: the Taylor bounds are exact up to padding, so equal to the interval image
    assert _width(res, "y") == pytest.approx(
        _interval_width(m, {"x": (-0.5, 0.5)}, "y"), rel=1e-6
    )


def test_structure_ops_and_reductions_are_sound():
    m = _model(
        """
        m (float[2,3,4] x) => (float[3] y) {
          t = Transpose<perm=[1,0,2]>(x)
          r = Reshape(t, shp)
          s = Sigmoid(r)
          u = Unsqueeze(s, ax)
          q = Squeeze(u, ax)
          c = Concat<axis=1>(q, q)
          m1 = ReduceMean<keepdims=0>(c, rax)
          z = Exp(m1)
          z2 = Reshape(z, shp2)
          y = ReduceSum<keepdims=0>(z2, rax)
        }""",
        {
            "shp": np.array([3, 8], dtype=np.int64),
            "ax": np.array([0], dtype=np.int64),
            "rax": np.array([1], dtype=np.int64),
            "shp2": np.array([3, 1], dtype=np.int64),
        },
        opset=18,
    )
    _assert_sound(m, {"x": (-1.0, 1.0)})


def test_division_of_two_taylor_models_is_sound():
    m = _model(
        """
        m (float[1,5] x) => (float[1,5] y) {
          three = Constant<value=float {3.0}>()
          d = Add(x, three)
          y = Div(x, d)
        }"""
    )
    _assert_sound(m, {"x": (0.0, 1.0)})


def test_integer_powers_are_sound():
    m = _model(
        """
        m (float[1,5] x) => (float[1,5] y) {
          two = Constant<value=float {2.0}>()
          three = Constant<value=float {3.0}>()
          a = Pow(x, two)
          b = Pow(x, three)
          y = Add(a, b)
        }"""
    )
    _assert_sound(m, {"x": (-0.7, 0.9)})


def _attention_model(rng, tokens=4, d=8, h=8):
    s = np.float32(1.0 / math.sqrt(h))
    return _model(
        f"""
        m (float[{tokens},{d}] x) => (float[{tokens},{h}] y) {{
          q = MatMul(x, Wq)
          k = MatMul(x, Wk)
          v = MatMul(x, Wv)
          kt = Transpose<perm=[1,0]>(k)
          sc = MatMul(q, kt)
          ss = Mul(sc, scale)
          a = Softmax<axis=-1>(ss)
          y = MatMul(a, v)
        }}""",
        {
            "Wq": 0.4 * _f32(rng, d, h), "Wk": 0.4 * _f32(rng, d, h), "Wv": 0.4 * _f32(rng, d, h),
            "scale": np.array(s, dtype=np.float32),
        },
    )  # fmt: skip


@pytest.mark.parametrize("degree", [1, 2])
def test_attention_block_is_sound(degree):
    m = _attention_model(np.random.default_rng(11))
    _assert_sound(m, {"x": (-0.15, 0.15)}, degree=degree, n_random=25, n_vertices=15)


def test_taylor_is_never_looser_than_intervals():
    rng = np.random.default_rng(12)
    cases = [
        (
            _model("m (float[3,4] x) => (float[3,4] y) { y = Softmax(x) }"),
            {"x": (-0.5, 0.5)},
        ),
        (
            _model("m (float[2,5] x) => (float[2,5] y) { y = Sigmoid(x) }"),
            {"x": (-3.0, 3.0)},
        ),
        (_attention_model(rng), {"x": (-0.2, 0.2)}),
    ]
    for m, rg in cases:
        res = T.propagate(m, rg)
        iv = interval.propagate(m, rg)
        for o in (o.name for o in m.graph.output):
            tlo, thi = res.bounds(o)
            ilo, ihi = np.asarray(iv.intervals[o][0]), np.asarray(iv.intervals[o][1])
            ok = np.isfinite(ilo) & np.isfinite(ihi)
            assert np.all(tlo[ok] >= ilo[ok] - 1e-6) and np.all(
                thi[ok] <= ihi[ok] + 1e-6
            )


# ---- (5) tightness ------------------------------------------------------------------


def test_softmax_is_much_tighter_than_intervals_on_a_narrow_box():
    m = _model("m (float[2,4] x) => (float[2,4] y) { y = Softmax(x) }")
    rg = {"x": (-0.1, 0.1)}
    res = _assert_sound(m, rg)
    assert _width(res, "y") < 0.35 * _interval_width(m, rg, "y")


def test_decomposed_gelu_is_tighter_than_intervals():
    body = """
    m (float[1,6] x) => (float[1,6] y) {
      s = Constant<value=float {1.4142135}>()
      one = Constant<value=float {1.0}>()
      half = Constant<value=float {0.5}>()
      t = Div(x, s)
      e = Erf(t)
      u = Add(e, one)
      v = Mul(x, u)
      y = Mul(v, half)
    }"""
    m = _model(body)
    # A box that straddles 0: there x and 1 + erf(x / sqrt 2) are strongly correlated and plain
    # intervals lose it (on an all-positive monotone box they would already be exact).
    rg = {"x": (-0.5, 0.5)}
    res = _assert_sound(m, rg)
    assert _width(res, "y") < 0.85 * _interval_width(m, rg, "y")


def test_decomposed_layernorm_is_tighter_than_intervals():
    rng = np.random.default_rng(13)
    gamma, beta = np.ones(6, np.float32), np.zeros(6, np.float32)
    body = """
    m (float[1,6] x) => (float[1,6] y) {
      mu = ReduceMean<axes=[-1], keepdims=1>(x)
      d = Sub(x, mu)
      sq = Mul(d, d)
      var = ReduceMean<axes=[-1], keepdims=1>(sq)
      eps = Constant<value=float {1e-5}>()
      ve = Add(var, eps)
      sd = Sqrt(ve)
      n = Div(d, sd)
      gn = Mul(n, gamma)
      y = Add(gn, beta)
    }"""
    m = _model(body, {"gamma": gamma, "beta": beta}, opset=13)
    rg = {"x": (_CENTRES - 0.05, _CENTRES + 0.05)}
    res = _assert_sound(m, rg)
    assert _width(res, "y") < 0.5 * _interval_width(m, rg, "y")
    assert rng is not None


def test_dependency_error_vanishes_with_box_width():
    """``Sigmoid(x) * Sigmoid(-x)``: intervals lose the correlation by a constant factor, a
    degree-1 model's excess over the true width halves with the box, degree 2's quarters."""
    m = _model(
        "m (float[1,4] x) => (float[1,4] y) { a = Sigmoid(x)  n = Neg(x)  b = Sigmoid(n)  y = Mul(a, b) }"
    )
    excess = {1: {}, 2: {}, "interval": {}}
    for r in (0.4, 0.2, 0.1):
        rg = {"x": (0.8 - r, 0.8 + r)}
        xs = np.linspace(0.8 - r, 0.8 + r, 20001)
        f = 1.0 / (1.0 + np.exp(-xs)) / (1.0 + np.exp(xs))
        true = f.max() - f.min()
        for deg in (1, 2):
            excess[deg][r] = _width(T.propagate(m, rg, degree=deg), "y") / true - 1.0
        excess["interval"][r] = _interval_width(m, rg, "y") / true - 1.0
    assert all(
        e > 1.5 for e in excess["interval"].values()
    )  # intervals: +165% at every width
    assert all(excess[2][r] < excess[1][r] for r in excess[2])
    assert excess[2][0.4] < 0.15 and excess[2][0.2] < 0.04 and excess[2][0.1] < 0.012
    assert (
        excess[2][0.2] / excess[2][0.4] < 0.35
        and excess[2][0.1] / excess[2][0.2] < 0.35
    )  # second order
    assert (
        excess[1][0.2] / excess[1][0.4] < 0.6 and excess[1][0.1] / excess[1][0.2] < 0.6
    )  # first order


# ---- fallbacks, unbounded inputs, errors ---------------------------------------------------


def test_unsupported_op_falls_back_to_an_interval_box_with_a_note():
    rng = np.random.default_rng(14)
    m = _model(
        """
        m (float[1,2,6,6] x) => (float[1,2,3,3] y) {
          c = Conv<pads=[1,1,1,1]>(x, W)
          p = MaxPool<kernel_shape=[2,2], strides=[2,2]>(c)
          y = Sigmoid(p)
        }""",
        {"W": _f32(rng, 2, 2, 3, 3)},
    )
    res = _assert_sound(m, {"x": (-0.4, 0.4)})
    assert any("precision lost at MaxPool" in n for n in res.notes)
    lo, hi = res.bounds("y")
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_op_with_no_interval_rule_gives_unbounded_not_a_wrong_bound():
    m = _model("m (float[4] x) => (float[4] y) { a = Abs(x)  y = Sigmoid(a) }")
    res = T.propagate(m, {"x": (-1.0, 1.0)})
    assert any("Abs" in n for n in res.notes)
    lo, hi = res.bounds("a")
    assert np.all(lo == -np.inf) and np.all(hi == np.inf)
    ylo, yhi = res.bounds("y")
    assert not np.any(np.isnan(ylo)) and not np.any(np.isnan(yhi))


def test_function_outside_its_domain_is_unbounded_not_nan():
    m = _model("m (float[1,3] x) => (float[1,3] y) { y = Reciprocal(x) }")
    res = T.propagate(m, {"x": (-1.0, 1.0)})
    lo, hi = res.bounds("y")
    assert np.all(lo == -np.inf) and np.all(hi == np.inf) and not np.any(np.isnan(lo))
    # log of a box that contains zero: same
    m2 = _model("m (float[1,3] x) => (float[1,3] y) { y = Log(x) }")
    lo2, hi2 = T.propagate(m2, {"x": (-0.5, 1.0)}).bounds("y")
    assert np.all(np.isinf(lo2)) and not np.any(np.isnan(hi2))


def test_unbounded_or_missing_input_ranges_are_rejected():
    m = _model("m (float[1,3] x) => (float[1,3] y) { y = Sigmoid(x) }")
    with pytest.raises(ValueError, match="no finite range"):
        T.propagate(m)
    with pytest.raises(ValueError, match="unbounded"):
        T.propagate(m, {"x": (-np.inf, 1.0)})
    with pytest.raises(ValueError, match="empty range"):
        T.propagate(m, {"x": (1.0, 0.0)})
    with pytest.raises(ValueError, match="degree"):
        T.propagate(m, {"x": (0.0, 1.0)}, degree=3)


def test_ranges_come_from_model_annotations():
    m = _model("m (float[1,3] x) => (float[1,3] y) { y = Sigmoid(x) }")
    ranges.set_range(m, "x", -1.0, 1.0)
    res = T.propagate(m)
    assert res.hull("y")[0] > 0.2 and res.hull("y")[1] < 0.8


def test_inputs_beyond_max_vars_enter_as_interval_remainders_soundly():
    m = _model("m (float[1,40] x) => (float[1,40] y) { y = Sigmoid(x) }")
    res = _assert_sound(m, {"x": (-0.5, 0.5)})
    tight = T.propagate(m, {"x": (-0.5, 0.5)}, max_vars=64)
    capped = T.propagate(m, {"x": (-0.5, 0.5)}, max_vars=8)
    assert capped.n_symbols == 8 and tight.n_symbols == 40
    _assert_sound_with(m, {"x": (-0.5, 0.5)}, capped)
    assert res is not None


def _assert_sound_with(model, input_ranges, res, n=30):
    rng = np.random.default_rng(15)
    box = _box(model, input_ranges)
    for _ in range(n):
        feed = {
            k: (lo + (hi - lo) * rng.random(lo.shape)).astype(np.float32)
            for k, (lo, hi) in box.items()
        }
        for name, val in _run_all(model, feed).items():
            lo, hi = res.bounds(name)
            v = np.asarray(val, dtype=np.float64)
            assert np.all(v >= lo - 1e-4 * (1 + np.abs(v))) and np.all(
                v <= hi + 1e-4 * (1 + np.abs(v))
            )


# ---- mean-value form -----------------------------------------------------------------


def test_mean_value_is_exact_on_an_affine_map():
    rng = np.random.default_rng(16)
    m = _model(
        "m (float[1,3] x) => (float[1,2] y) { a = MatMul(x, W)  y = Add(a, B) }",
        {"W": _f32(rng, 3, 2), "B": _f32(rng, 2)},
    )
    rg = {"x": (-0.5, 1.0)}
    mv = T.mean_value_bounds(m, rg).bounds["y"]
    lo, hi = interval.propagate(m, rg).intervals["y"]
    np.testing.assert_allclose(mv[0], lo, atol=1e-6)
    np.testing.assert_allclose(mv[1], hi, atol=1e-6)


def test_mean_value_is_sound_and_never_looser_than_intervals():
    rng = np.random.default_rng(17)
    m = _model(
        """
        m (float[1,4] x) => (float[1,2] y) {
          a = MatMul(x, W1)
          b = Tanh(a)
          c = MatMul(b, W2)
          y = Sigmoid(c)
        }""",
        # modest weights: with unit-variance ones the net saturates and every method says [0, 1]
        {"W1": 0.4 * _f32(rng, 4, 5), "W2": 0.4 * _f32(rng, 5, 2)},
    )
    rg = {"x": (-0.3, 0.3)}
    mv = T.mean_value_bounds(m, rg).bounds["y"]
    iv = interval.propagate(m, rg).intervals["y"]
    # up to the 1e-9 relative widening the mean-value form applies and plain intervals do not
    assert np.all(mv[0] >= iv[0] - 1e-8) and np.all(mv[1] <= iv[1] + 1e-8)
    assert float(np.mean(mv[1] - mv[0])) < float(np.mean(iv[1] - iv[0]))
    box = _box(m, rg)
    for _ in range(300):
        x = (box["x"][0] + (box["x"][1] - box["x"][0]) * rng.random((1, 4))).astype(
            np.float32
        )
        y = _run_all(m, {"x": x})["y"].astype(np.float64)
        assert np.all(y >= mv[0] - 1e-5) and np.all(y <= mv[1] + 1e-5)


def test_mean_value_handles_softmax_gelu_and_layernorm_soundly():
    cases = [
        (
            _model("m (float[1,4] x) => (float[1,4] y) { y = Softmax(x) }"),
            {"x": (-0.3, 0.3)},
        ),
        (
            _model("m (float[1,4] x) => (float[1,4] y) { y = Gelu(x) }", opset=20),
            {"x": (0.0, 0.8)},
        ),
        (
            _model(
                "m (float[1,6] x) => (float[1,6] y) { y = LayerNormalization<epsilon=1e-5>(x, g, b) }",
                {"g": np.ones(6, np.float32), "b": np.zeros(6, np.float32)},
            ),
            {"x": (_CENTRES - 0.05, _CENTRES + 0.05)},
        ),
    ]
    for m, rg in cases:
        mv = T.mean_value_bounds(m, rg).bounds["y"]
        box = _box(m, rg)
        rng = np.random.default_rng(18)
        for _ in range(100):
            x = {
                k: (lo + (hi - lo) * rng.random(lo.shape)).astype(np.float32)
                for k, (lo, hi) in box.items()
            }
            y = _run_all(m, x)["y"].astype(np.float64)
            assert np.all(y >= mv[0] - 1e-5) and np.all(y <= mv[1] + 1e-5)
        assert np.all(np.isfinite(mv[0]))


def test_mean_value_unsupported_op_is_unbounded_and_large_inputs_are_rejected():
    m = _model("m (float[4] x) => (float[4] y) { y = Abs(x) }")
    lo, hi = T.mean_value_bounds(m, {"x": (-1.0, 1.0)}).bounds["y"]
    assert np.all(np.isinf(lo)) and not np.any(np.isnan(hi))
    big = _model("m (float[1,300] x) => (float[1,300] y) { y = Sigmoid(x) }")
    with pytest.raises(ValueError, match="max_inputs"):
        T.mean_value_bounds(big, {"x": (0.0, 1.0)})
    with pytest.raises(ValueError, match="no finite range"):
        T.mean_value_bounds(m)


# ---- compare_methods -----------------------------------------------------------------------


def test_compare_methods_reports_every_method_and_a_sampled_reference():
    m = _model("m (float[1,4] x) => (float[1,4] y) { y = Softmax(x) }")
    cmp = T.compare_methods(m, {"x": (-0.1, 0.1)}, samples=100)
    assert set(cmp.bounds) >= {"interval", "taylor", "mean_value", "sampled"}
    assert cmp.width("taylor", "y") < cmp.width("interval", "y")
    assert (
        cmp.width("sampled", "y") <= cmp.width("taylor", "y") + 1e-6
    )  # sampled is an inner bound
    assert "taylor" in cmp.table() and "sampled" in cmp.table()


def test_compare_methods_reports_failures_instead_of_raising():
    m = _model("m (float[1,3] x) => (float[1,3] y) { y = Sigmoid(x) }")
    cmp = T.compare_methods(m, None, samples=5)
    assert "taylor" in cmp.errors and "no finite range" in cmp.errors["taylor"]
    assert "unavailable" in cmp.table()


def test_compare_methods_rejects_unknown_method_names_into_errors():
    m = _model("m (float[1,3] x) => (float[1,3] y) { y = Sigmoid(x) }")
    cmp = T.compare_methods(
        m, {"x": (0.0, 1.0)}, methods=("taylor", "nonsense"), samples=5
    )
    assert "nonsense" in cmp.errors and "taylor" in cmp.bounds


def test_degree_one_and_zero_width_boxes_go_through_convolutions():
    """No quadratic slots (degree 1) and no noise symbols at all (a point box) are empty leading
    axes: they once made a Conv return an array of the *input* shape and fail to broadcast."""
    rng = np.random.default_rng(19)
    m = _model(
        """
        m (float[1,2,5,5] x) => (float[1,3,5,5] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          s = Sigmoid(c)
          y = Conv<pads=[1,1,1,1]>(s, W2)
        }""",
        {"W": _f32(rng, 4, 2, 3, 3), "B": _f32(rng, 4), "W2": _f32(rng, 3, 4, 3, 3)},
    )
    for degree in (1, 2):
        _assert_sound(m, {"x": (-0.2, 0.2)}, degree=degree, n_random=10, n_vertices=6)
    res = T.propagate(m, {"x": (0.3, 0.3)})  # a point: zero noise symbols
    assert res.n_symbols == 0
    lo, hi = res.bounds("y")
    feed = {"x": np.full((1, 2, 5, 5), 0.3, dtype=np.float32)}
    y = _run_all(m, feed)["y"].astype(np.float64)
    assert np.all(y >= lo - 1e-4) and np.all(y <= hi + 1e-4)
    assert float(np.max(hi - lo)) < 1e-6  # a point in, a point out


def test_mean_value_through_convolutions_and_flatten_is_sound():
    rng = np.random.default_rng(20)
    m = _model(
        """
        m (float[1,2,4,4] x) => (float[1,3] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          s = Sigmoid(c)
          p = GlobalAveragePool(s)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W2)
        }""",
        {"W": 0.5 * _f32(rng, 4, 2, 3, 3), "B": _f32(rng, 4), "W2": _f32(rng, 3, 4)},
    )
    rg = {"x": (-0.2, 0.2)}
    mv = T.mean_value_bounds(m, rg).bounds["y"]
    iv = interval.propagate(m, rg).intervals["y"]
    assert np.all(mv[0] >= iv[0] - 1e-8) and np.all(mv[1] <= iv[1] + 1e-8)
    box = _box(m, rg)
    for k in range(150):
        u = (
            rng.random((1, 2, 4, 4))
            if k % 2
            else (rng.random((1, 2, 4, 4)) < 0.5).astype(float)
        )
        x = (box["x"][0] + (box["x"][1] - box["x"][0]) * u).astype(np.float32)
        y = _run_all(m, {"x": x})["y"].astype(np.float64)
        assert np.all(y >= mv[0] - 1e-5) and np.all(y <= mv[1] + 1e-5)


# ---- the documentation's examples ----------------------------------------------------------

_DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "taylor-models.md"
_NUM = r"-?\d+\.?\d*(?:e-?\d+)?"


def test_documentation_examples_run_and_print_what_the_doc_quotes():
    text = _DOC.read_text()
    code = re.findall(r"<!-- doctest -->\n```python\n(.*?)```", text, re.S)
    quoted = re.findall(r"<!-- output -->\n```text\n(.*?)```", text, re.S)
    assert code and len(code) == len(quoted)
    namespace = {}
    for src, want in zip(code, quoted):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, str(_DOC), "exec"), namespace)
        got = buf.getvalue()
        # same text once numbers and spacing are ignored ...
        strip = lambda s: re.sub(r"\s+", "", re.sub(_NUM, "#", s))  # noqa: E731
        assert strip(got) == strip(want), f"output of a doc example changed:\n{got}"
        # ... and the numbers agree (sampling and float32 leave a little room)
        a = [float(x) for x in re.findall(_NUM, got)]
        b = [float(x) for x in re.findall(_NUM, want)]
        np.testing.assert_allclose(
            a, b, rtol=0.03, atol=2e-4, err_msg=f"numbers in:\n{got}"
        )
