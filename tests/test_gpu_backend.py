"""Tests for the opt-in torch / GPU backend of onnxsim.crown and onnxsim.backward_diff.

CI has no GPU, so everything that matters runs on ``device="torch-cpu"`` -- the same code path as
``"cuda"`` (device-resident tensors, the same ops), only on the CPU. The tests that need an
accelerator are skipped when ``torch.cuda.is_available()`` is False and are run by hand on every
device (CUDA and ROCm) -- see docs/gpu-backend.md.

What is checked:

* the default device is the numpy float64 path and nothing imports torch for it;
* the torch float64 backend gives the same bounds as numpy for crown / alpha / bab / prima;
* **float32 is sound**: the primitive error model (``_rigorous_f32``) against a float64 reference
  including adversarial cancellation, a mutation check that this test *fails* when the widening
  is removed, end-to-end containment on an affine net where float32 rounding is the only slack,
  the same-parameters comparison against float64, and sampled soundness against onnxruntime;
* device memory: a chunk that does not fit is halved and retried, then a clear error.
"""

import subprocess
import sys

import numpy as np
import pytest
from onnx import numpy_helper, parser

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from onnxsim import (  # noqa: E402
    _device,  # noqa: E402
    backward_diff,
    crown,
    quant_verify,
)
from onnxsim import _rigorous_f32 as R  # noqa: E402
from onnxsim import ranges as ranges_mod  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA / ROCm device"
)

# float32 execution can exceed a real-arithmetic bound by float32 rounding
SLACK = 1e-4


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _model(body, initializer=None, opset=13, ir_version=8):
    m = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    m.graph.initializer.extend(
        numpy_helper.from_array(np.asarray(v, dtype=np.float32), k)
        for k, v in (initializer or {}).items()
    )
    return m


def _mlp(seed=0, n_in=4, widths=(10, 10), n_out=3, relu=True, dw=0.0):
    rng = np.random.default_rng(seed)
    rng_dw = np.random.default_rng(seed + 1000)
    dims = [n_in, *widths, n_out]
    lines, prev, inits = [], "x", {}
    for i in range(len(dims) - 1):
        inits[f"W{i}"] = rng.standard_normal((dims[i], dims[i + 1])) * 0.8
        if dw:
            w_i = inits[f"W{i}"]
            inits[f"W{i}"] = w_i + rng_dw.standard_normal(w_i.shape) * dw
        inits[f"B{i}"] = rng.standard_normal(dims[i + 1]) * 0.2
        lines.append(f"m{i} = MatMul({prev}, W{i})\n a{i} = Add(m{i}, B{i})")
        prev = f"a{i}"
        if relu and i < len(dims) - 2:
            lines.append(f"r{i} = Relu({prev})")
            prev = f"r{i}"
    lines.append(f"y = Identity({prev})")
    body = (
        f"g (float[1,{n_in}] x) => (float[1,{n_out}] y) {{\n" + "\n".join(lines) + "\n}"
    )
    return _model(body, inits)


def _convnet(seed=0, h=6, c=3, k=4):
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.standard_normal(s)  # noqa: E731
    inits = dict(
        W1=f(k, c, 3, 3) * 0.3,
        B1=f(k) * 0.1,
        g=np.abs(f(k)) + 0.5,
        be=f(k) * 0.1,
        mu=f(k) * 0.1,
        var=np.abs(f(k)) + 0.5,
        W2=f(k, k, 3, 3) * 0.3,
        B2=f(k) * 0.1,
        W3=f(5, k) * 0.5,
        B3=f(5) * 0.1,
    )
    return _model(
        f"g (float[1,{c},{h},{h}] x) => (float[1,5] y) {{\n"
        "c1 = Conv<pads=[1,1,1,1]>(x, W1, B1)\n"
        "b1 = BatchNormalization<epsilon=1e-5>(c1, g, be, mu, var)\n"
        "r1 = Relu(b1)\n c2 = Conv<strides=[2,2], pads=[1,1,1,1]>(r1, W2, B2)\n"
        "r2 = Relu(c2)\n p = GlobalAveragePool(r2)\n f = Flatten(p)\n"
        "y = Gemm<transB=1>(f, W3, B3)\n}",
        inits,
    )


def _resnet_block(seed=0, n=6):
    rng = np.random.default_rng(seed)
    return _model(
        f"g (float[1,{n}] x) => (float[1,{n}] y) {{\n"
        "a = MatMul(x, W)\n r = Relu(a)\n s = Sigmoid(r)\n y = Add(s, x)\n}",
        {"W": rng.standard_normal((n, n)) * 0.5},
        ir_version=8,
    )


def _sample_inside(model, bounds, box, n=300, seed=0):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    lo, hi = box
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.uniform(lo, hi, shape).astype(np.float32)
        y = sess.run(None, {model.graph.input[0].name: x})[0]
        pad = SLACK * (1 + np.abs(y))
        assert np.all(y >= bounds.lo - pad) and np.all(y <= bounds.hi + pad)


BOX = {"x": (-1.0, 1.0)}


# --------------------------------------------------------------------------
# device resolution
# --------------------------------------------------------------------------


def test_default_is_the_numpy_float64_path():
    r = _device.resolve(None, None)
    assert r.is_numpy and r.precision == "float64" and r.torch_device is None
    assert _device.resolve("cpu", "float64") is _device.NUMPY
    assert crown._Backend().numpy


def test_bad_arguments_are_rejected_clearly():
    with pytest.raises(ValueError, match="device must be one of"):
        _device.resolve("tpu")
    with pytest.raises(ValueError, match="precision must be"):
        _device.resolve("torch-cpu", "float16")
    with pytest.raises(ValueError, match="needs a torch device"):
        _device.resolve("cpu", "float32")
    with pytest.raises(ValueError, match="needs a torch device"):
        crown.bounds(_mlp(), BOX, precision="float32")
    with pytest.raises(ValueError):
        _device.resolve("cuda:x" if torch.cuda.is_available() else "gpu")


def test_a_missing_accelerator_raises_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never falls back"):
        crown.bounds(_mlp(), BOX, device="cuda")
    assert _device.resolve_device("auto") == "cpu"  # 'auto' may choose, 'cuda' may not


def test_cpu_numpy_path_does_not_import_torch():
    code = (
        "import sys\n"
        "import onnxsim.crown, onnxsim._device\n"
        "from onnxsim import crown\n"
        "from onnx import parser\n"
        "before = 'torch' in sys.modules\n"  # the package itself may import torch elsewhere
        "m = parser.parse_model('<ir_version: 8, opset_import: [\"\" : 13]> g (float[1,2] x) => (float[1,2] y) { y = Relu(x) }')\n"
        "crown.bounds(m, {'x': (-1.0, 1.0)}, method='crown')\n"
        "assert before or 'torch' not in sys.modules, 'the default path imported torch'\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_device_cpu_equals_default_exactly():
    m = _convnet()
    a = crown.bounds(m, BOX)["y"]
    b = crown.bounds(m, BOX, device="cpu", precision="float64")["y"]
    assert np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)


# --------------------------------------------------------------------------
# float64 on the torch backend == numpy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ibp", "crown", "alpha", "bab", "prima"])
@pytest.mark.parametrize("net", ["mlp", "conv", "residual"])
def test_torch_float64_matches_numpy(net, method):
    m = {"mlp": _mlp(), "conv": _convnet(), "residual": _resnet_block()}[net]
    kw = dict(budget=8, alpha_iters=15)
    a = crown.bounds(m, BOX, method=method, **kw)["y"]
    b = crown.bounds(m, BOX, method=method, device="torch-cpu", **kw)["y"]
    np.testing.assert_allclose(b.lo, a.lo, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(b.hi, a.hi, rtol=1e-8, atol=1e-8)
    assert a.method == b.method


def test_verify_output_ranges_agrees_across_backends():
    m = _mlp()
    ranges_mod.set_range(m, "y", -50.0, 50.0)
    a = crown.verify_output_ranges(m, BOX)
    b = crown.verify_output_ranges(m, BOX, device="torch-cpu")
    c = crown.verify_output_ranges(m, BOX, device="torch-cpu", precision="float32")
    assert a["y"].proved and b["y"].proved and c["y"].proved
    tight = _mlp()
    ranges_mod.set_range(
        tight, "y", -1.0, 1.0
    )  # too tight to hold: nobody may prove it
    for kw in (
        {},
        {"device": "torch-cpu"},
        {"device": "torch-cpu", "precision": "float32"},
    ):
        assert not crown.verify_output_ranges(tight, BOX, **kw)["y"].proved


def test_chunking_does_not_change_the_result(monkeypatch):
    m = _convnet()
    base = crown.bounds(m, BOX, device="torch-cpu")["y"]
    monkeypatch.setattr(crown, "_ROW_BUDGET", 300)  # a handful of rows per chunk
    small = crown.bounds(m, BOX, device="torch-cpu")["y"]
    np.testing.assert_allclose(small.lo, base.lo, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(small.hi, base.hi, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# the float32 error model, primitive by primitive
# --------------------------------------------------------------------------


def _wide(rng, shape):
    """Values spanning ~7 orders of magnitude with random signs: cancellation-prone."""
    return (
        rng.standard_normal(shape) * 10.0 ** rng.integers(-3, 4, size=shape)
    ).astype(np.float32)


def _perturbed(rng, p_v, p_e):
    """A 'true' float64 value inside [v - e, v + e]."""
    return p_v.astype(np.float64) + rng.uniform(-1, 1, p_v.shape) * p_e.astype(
        np.float64
    )


def _primitive_violations(trials=6):
    """Cases where the computed error bound is smaller than the real float32 error."""
    bad = []
    rng = np.random.default_rng(1234)
    dev = "cpu"

    def check(tag, got_v, got_e, truth, scale):
        v = got_v.detach().double().numpy()
        e = 0.0 if got_e is None else got_e.detach().double().numpy()
        slack = 1e-12 * scale  # the float64 reference's own rounding
        miss = np.abs(v - truth) - (e + slack)
        if np.any(miss > 0):
            bad.append((tag, float(miss.max())))

    for t in range(trials):
        # ---- add (broadcast), both sides carrying input error
        a = _wide(rng, (7, 9))
        ae = np.abs(a) * 1e-6
        b = _wide(rng, (9,))
        be = np.abs(b) * 1e-6
        pa = R.P(torch.as_tensor(a), torch.as_tensor(ae.astype(np.float32)))
        pb = R.P(torch.as_tensor(b), torch.as_tensor(be.astype(np.float32)))
        truth = _perturbed(rng, a, ae) + _perturbed(rng, b, be)
        s = pa + pb
        check("add", s.v, s.e, truth, np.abs(a) + np.abs(b) + ae + be)

        # ---- multiply by a constant that needs rounding (cast error)
        c64 = rng.standard_normal((7, 9)) * 3.3
        k = R.K.from_numpy(c64, dev)
        truth = _perturbed(rng, a, ae) * c64
        m_ = pa * k
        check("mul", m_.v, m_.e, truth, np.abs(a * c64) * 2 + ae * np.abs(c64))

        # ---- division by a float
        d = pa / 7.3
        check("div", d.v, d.e, _perturbed(rng, a, ae) / 7.3, np.abs(a) / 7.3 * 2 + ae)

        # ---- matmul with a long inner dimension and cancellation
        n = int(rng.integers(40, 400))
        x = _wide(rng, (5, n))
        w64 = rng.standard_normal((n, 6)) * 10.0 ** rng.integers(-2, 3, (n, 6))
        xe = np.abs(x) * 1e-7
        kx = R.K.from_numpy(w64, dev)
        px = R.P(torch.as_tensor(x), torch.as_tensor(xe.astype(np.float32)))
        y = px @ kx
        truth = _perturbed(rng, x, xe) @ w64
        check("matmul", y.v, y.e, truth, (np.abs(x) + xe) @ np.abs(w64))

        # ---- einsum used by the conv backward pass
        a5 = _wide(rng, (3, 2, 5, 4, 4))
        w2 = rng.standard_normal((5, 6)) * 2
        pe = R.P(torch.as_tensor(a5))
        out = R.einsum("mnqij,qc->mncij", pe, R.K.from_numpy(w2, dev))
        truth = np.einsum("mnqij,qc->mncij", a5.astype(np.float64), w2)
        check(
            "einsum",
            out.v,
            out.e,
            truth,
            np.einsum("mnqij,qc->mncij", np.abs(a5.astype(np.float64)), np.abs(w2)),
        )

        # ---- a long block-wise sum (more terms than BLOCK, so several levels)
        big = _wide(rng, (3, 6000))
        ps = R.P(torch.as_tensor(big))
        r = ps.sum(1)
        check(
            "sum",
            r.v,
            r.e,
            big.astype(np.float64).sum(1),
            np.abs(big.astype(np.float64)).sum(1),
        )

        # ---- accumulation with += (the conv backward scatter)
        acc = R.P(torch.zeros(5), torch.zeros(5))
        truth = np.zeros(5)
        for _ in range(9):
            term = _wide(rng, (5,))
            acc += R.P(torch.as_tensor(term))
            truth += term.astype(np.float64)
        check("iadd", acc.v, acc.e, truth, np.full(5, 1e4 * 9))
    return bad


def test_float32_error_model_covers_the_real_rounding_error():
    assert _primitive_violations() == []


def test_the_primitive_check_has_teeth_it_fails_when_the_widening_is_removed(
    monkeypatch,
):
    """Mutation: with no rounding terms (gamma = 0, u = 0) the check must catch real violations."""
    monkeypatch.setattr(R, "gamma", lambda n: 0.0)
    monkeypatch.setattr(R, "UP", 0.0)
    monkeypatch.setattr(R, "_infl", lambda n: 1.0)
    assert _primitive_violations() != []


def test_overflow_is_minus_infinity_never_a_finite_bound():
    huge = R.P(torch.full((2,), 3e38))
    k = R.K.exact(np.array([10.0, 10.0]), "cpu")
    out = huge * k  # overflows float32
    low = R.lower_to_numpy(out)
    assert np.all(np.isneginf(low))
    nan_case = R.P(torch.tensor([float("nan"), 1.0]))
    assert np.isneginf(R.lower_to_numpy(nan_case)[0])


def test_directed_rounding_helpers():
    x = np.array([0.1, -0.1, 1e-3, 3.0000000001, -7.7e5])
    assert np.all(R.up32(x) >= x) and np.all(R.down32(x) <= x)
    assert np.all(R.up32(x) - R.down32(x) <= 2 * np.abs(x) * 2.0**-23 + 1e-45)
    exact = np.array([1.0, 0.5, -2.0])
    assert np.array_equal(R.up32(exact), exact) and np.array_equal(
        R.down32(exact), exact
    )


def test_tree_sum_uses_few_terms_in_the_error_bound():
    x = torch.ones(1, 1_000_000)
    _, n_eff = R.tree_sum(x, (1,))
    assert n_eff < 1000  # block-wise: not a million-term gamma
    assert R.gamma(1_000_000) > 0.01 > R.gamma(n_eff)
    assert R.gamma(10_000_000) == float("inf")  # beyond the model: vacuous, not wrong


# --------------------------------------------------------------------------
# float32 end to end
# --------------------------------------------------------------------------


def test_float32_bound_contains_the_float64_bound_on_an_affine_net():
    """No Relu: both passes are exact linear algebra, so float32 rounding is the *only* difference.

    The float64 result is (to 1e-16) the exact bound; the float32 result must therefore lie
    outside it by at least the float32 rounding it cannot see -- i.e. contain it.
    """
    m = _mlp(seed=3, n_in=8, widths=(64, 64), n_out=16, relu=False)
    exact = crown.bounds(m, BOX, method="crown")["y"]
    got = crown.bounds(m, BOX, method="crown", device="torch-cpu", precision="float32")[
        "y"
    ]
    assert np.all(got.lo <= exact.lo) and np.all(got.hi >= exact.hi)
    assert np.all(got.hi - got.lo >= exact.hi - exact.lo)
    # ... and is not absurdly loose
    assert float((got.hi - got.lo).sum()) <= 1.001 * float((exact.hi - exact.lo).sum())


def test_mutation_without_the_widening_float32_would_not_contain_float64(monkeypatch):
    m = _mlp(seed=3, n_in=8, widths=(64, 64), n_out=16, relu=False)
    exact = crown.bounds(m, BOX, method="crown")["y"]
    monkeypatch.setattr(R, "gamma", lambda n: 0.0)
    monkeypatch.setattr(R, "UP", 0.0)
    monkeypatch.setattr(R, "_infl", lambda n: 1.0)
    monkeypatch.setattr(R, "ETA", 0.0)
    got = crown.bounds(m, BOX, method="crown", device="torch-cpu", precision="float32")[
        "y"
    ]
    assert not (np.all(got.lo <= exact.lo) and np.all(got.hi >= exact.hi))


@pytest.mark.parametrize("method", ["crown", "alpha", "bab", "prima"])
@pytest.mark.parametrize("net", ["mlp", "conv", "residual"])
def test_float32_bounds_enclose_every_sampled_output(net, method):
    m = {"mlp": _mlp(seed=2), "conv": _convnet(seed=2), "residual": _resnet_block()}[
        net
    ]
    b = crown.bounds(
        m,
        BOX,
        method=method,
        budget=8,
        alpha_iters=15,
        device="torch-cpu",
        precision="float32",
    )["y"]
    _sample_inside(m, b, BOX[next(iter(BOX))])


@pytest.mark.parametrize("net", ["mlp", "conv"])
def test_float32_costs_little_tightness(net):
    m = {"mlp": _mlp(), "conv": _convnet()}[net]
    a = crown.bounds(m, BOX, method="crown")["y"]
    b = crown.bounds(m, BOX, method="crown", device="torch-cpu", precision="float32")[
        "y"
    ]
    wa, wb = float((a.hi - a.lo).sum()), float((b.hi - b.lo).sum())
    assert wa <= wb <= 1.01 * wa


def test_float32_at_the_same_alpha_is_never_tighter_than_float64():
    """Same parameters, same boxes: the rigorous float32 lower bound sits below float64's."""
    m = _mlp(seed=5, widths=(16, 16))
    an64 = crown._Analyzer(m, BOX)
    an32 = crown._Analyzer(m, BOX, crown._Backend("torch-cpu", "float32"))
    an64.refine()
    an32.ib = dict(an64.ib)  # identical boxes: only the arithmetic differs
    an32._relax_cache.clear()
    an32._relax32_cache.clear()
    rng = np.random.default_rng(0)
    relus = [i for i, n in enumerate(an64.nodes) if n.op_type == "Relu"]
    alpha = {
        i: torch.as_tensor(
            rng.uniform(0, 1, an64.ib[an64.nodes[i].input[0]][0].shape)
            .astype(np.float32)
            .astype(np.float64)
        )
        for i in relus
    }
    shape = tuple(an64.ib["y"][0].shape)
    size = int(np.prod(shape))
    ops64, ops32 = crown._TorchOps("cpu"), an32.backend.ops()
    for side in (1.0, -1.0):
        eye = np.eye(size).reshape((size,) + shape) * side
        lo64 = ops64.to_numpy(an64._lower(ops64, "y", ops64.asarray(eye), alpha))
        with an32.backend.context():
            lo32 = ops32.to_numpy(an32._lower(ops32, "y", ops32.asarray(eye), alpha))
        assert np.all(lo32 <= lo64 + 1e-12 * (1 + np.abs(lo64)))


def _cut_violations(facets_f32, trials=200):
    """Worst violation of float32-rounded cuts at the exact vertices of each pair hull."""
    rng = np.random.default_rng(10)
    worst, checked = 0.0, 0
    for _ in range(trials):
        l1, l2 = -rng.uniform(0.05, 3, 2)
        u1, u2 = rng.uniform(0.05, 3, 2)
        t = rng.uniform(0.0, 0.45, 4)  # correlated neurons: a polygon, not the box
        w_s, w_d = (u1 + u2) - (l1 + l2), (u1 - l2) - (l1 - u2)
        poly = (
            l1 + l2 + t[0] * w_s, u1 + u2 - t[1] * w_s,
            l1 - u2 + t[2] * w_d, u1 - l2 - t[3] * w_d,
        )  # fmt: skip
        n_, d_ = crown._pair_facets(l1, u1, l2, u2, *poly)
        if not len(d_):
            continue
        n32, d32 = facets_f32(
            n_,
            d_,
            [max(abs(l1), abs(u1)), max(abs(l2), abs(u2)), max(0, u1), max(0, u2)],
        )
        q = crown._pair_polygon(
            l1, u1, l2, u2, *poly
        )  # the cell vertices: the cuts are tight here
        v = np.column_stack([q, np.maximum(q, 0.0)])
        worst = max(worst, float((v @ n32.T - d32[None, :]).max()))
        checked += 1
    return worst, checked


def test_rounded_cuts_stay_valid_at_every_hull_vertex():
    worst, checked = _cut_violations(crown._facets_f32)
    assert checked > 80
    assert worst <= 0.0


def test_mutation_rounding_a_cut_without_raising_d_breaks_validity():
    def naive(n_, d_, mags):
        return n_.astype(np.float32).astype(np.float64), d_  # forgot the rounding of n

    worst, checked = _cut_violations(naive)
    assert checked > 80
    assert worst > 0.0  # the vertex check does see the invalidity


def test_float32_valid_cuts_are_exactly_representable():
    rng = np.random.default_rng(2)
    n_ = rng.standard_normal((6, 4))
    n32, d32 = crown._facets_f32(n_, rng.standard_normal(6), [1.0, 2.0, 0.5, 3.0])
    assert np.array_equal(n32.astype(np.float32).astype(np.float64), n32)
    assert np.array_equal(d32.astype(np.float32).astype(np.float64), d32)


def test_relaxation_lines_rounded_to_float32_still_sandwich_the_function():
    rng = np.random.default_rng(1)
    lo = -rng.uniform(0.01, 6, 400)
    hi = rng.uniform(0.01, 6, 400)
    for kind, f in (
        ("Relu", lambda x: np.maximum(x, 0)),
        ("Sigmoid", crown._sigmoid),
        ("Tanh", np.tanh),
    ):
        r = (
            crown._relu_relax(lo, hi)
            if kind == "Relu"
            else crown._scurve_relax(kind, lo, hi)
        )
        s = R.sound_lines_f32(kind, r, lo, hi)
        for key in ("a_u", "b_u") + (("a_l", "b_l") if kind != "Relu" else ()):
            assert np.all(s[key].astype(np.float32).astype(np.float64) == s[key])
        t = rng.uniform(0, 1, (50, lo.size))
        x = lo + (hi - lo) * t
        assert np.all(s["a_u"] * x + s["b_u"] >= f(x) - 1e-12)
        if kind != "Relu":
            assert np.all(s["a_l"] * x + s["b_l"] <= f(x) + 1e-12)


# --------------------------------------------------------------------------
# device memory handling
# --------------------------------------------------------------------------


def test_out_of_memory_halves_the_chunk_and_still_gives_the_same_answer(monkeypatch):
    m = _convnet()
    base = crown.bounds(m, BOX, device="torch-cpu")["y"]
    real = crown._Analyzer._lower
    seen = []

    def flaky(self, ops, target, spec, *a, **k):
        seen.append(int(spec.shape[0]))
        if spec.shape[0] > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return real(self, ops, target, spec, *a, **k)

    monkeypatch.setattr(crown._Analyzer, "_lower", flaky)
    got = crown.bounds(m, BOX, device="torch-cpu")["y"]
    np.testing.assert_allclose(got.lo, base.lo, rtol=1e-9, atol=1e-9)
    assert max(seen) > 2 and min(seen) <= 2  # it did retry with smaller chunks


def test_out_of_memory_on_a_single_row_is_a_clear_error(monkeypatch):
    def always(self, ops, target, spec, *a, **k):
        raise RuntimeError("HIP out of memory")

    monkeypatch.setattr(crown._Analyzer, "_lower", always)
    with pytest.raises(crown.DeviceMemoryError, match="out of memory bounding"):
        crown.bounds(_convnet(), BOX, device="torch-cpu")


def test_other_errors_are_not_swallowed_as_oom(monkeypatch):
    def broken(self, ops, target, spec, *a, **k):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(crown._Analyzer, "_lower", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        crown.bounds(_convnet(), BOX, device="torch-cpu")


def test_strict_float32_restores_the_backend_flags():
    before = torch.get_float32_matmul_precision()
    with _device.strict_float32():
        assert torch.get_float32_matmul_precision() == "highest"
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == before


# --------------------------------------------------------------------------
# backward_diff and quant_verify
# --------------------------------------------------------------------------


def _pair(seed=0, dw=0.01):
    kw = dict(seed=seed, n_in=8, widths=(16, 16), n_out=4)
    return _mlp(**kw), _mlp(**kw, dw=dw)


def test_backward_diff_float64_matches_numpy_and_float32_is_sound():
    a, b = _pair()
    box = {"x": (-1.0, 1.0)}
    base = backward_diff.bound_difference(a, b, box)
    f64 = backward_diff.bound_difference(a, b, box, device="torch-cpu")
    f32 = backward_diff.bound_difference(
        a, b, box, device="torch-cpu", precision="float32"
    )
    for o in base.max_abs:
        np.testing.assert_allclose(
            f64.max_abs[o], base.max_abs[o], rtol=1e-8, atol=1e-10
        )
        assert np.all(f32.max_abs[o] >= base.max_abs[o] * (1 - 1e-9))
        assert np.all(f32.max_abs[o] <= base.max_abs[o] * 1.01 + 1e-6)
    # sound: sampled differences stay below the float32 bound
    sa = ort.InferenceSession(a.SerializeToString(), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(b.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.uniform(-1, 1, (1, 8)).astype(np.float32)
        d = np.abs(sa.run(None, {"x": x})[0] - sb.run(None, {"x": x})[0])
        assert np.all(d <= f32.max_abs["y"] + SLACK * (1 + d))


def _qdq_pair():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 8)).astype(np.float32) * 0.5
    ref = _model(
        "g (float[1,8] x) => (float[1,8] y) { a = MatMul(x, W) y = Relu(a) }", {"W": w}
    )
    scale = np.float32(0.02)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    quant = parser.parse_model(
        '<ir_version: 8, opset_import: ["" : 13]> g (float[1,8] x) => (float[1,8] y) {'
        "xq = QuantizeLinear(x, sx, zx)\n xd = DequantizeLinear(xq, sx, zx)\n"
        "wd = DequantizeLinear(Wq, sw, zw)\n a = MatMul(xd, wd)\n y = Relu(a) }"
    )
    quant.graph.initializer.extend(
        [
            numpy_helper.from_array(np.float32(1 / 127), "sx"),
            numpy_helper.from_array(np.int8(0), "zx"),
            numpy_helper.from_array(q, "Wq"),
            numpy_helper.from_array(scale, "sw"),
            numpy_helper.from_array(np.int8(0), "zw"),
        ]
    )
    return ref, quant


def test_quant_verify_backward_engine_device_option():
    ref, quant = _qdq_pair()
    box = {"x": (-1.0, 1.0)}
    base = quant_verify.verify(ref, quant, box, breakdown=False, engine="backward")
    f64 = quant_verify.verify(
        ref, quant, box, breakdown=False, engine="backward", device="torch-cpu"
    )
    f32 = quant_verify.verify(
        ref,
        quant,
        box,
        breakdown=False,
        engine="backward",
        device="torch-cpu",
        precision="float32",
    )
    assert f64.worst == pytest.approx(base.worst, rel=1e-8)
    assert base.worst <= f32.worst <= base.worst * 1.01
    obs = quant_verify.observed_error(ref, quant, box, n=150)
    assert obs <= f32.worst + SLACK * (1 + f32.worst)


def test_quant_verify_zonotope_engine_rejects_a_device():
    ref, quant = _qdq_pair()
    with pytest.raises(ValueError, match="engine='backward' only"):
        quant_verify.verify(ref, quant, {"x": (-1.0, 1.0)}, device="torch-cpu")
    with pytest.raises(ValueError, match="engine='backward' only"):
        quant_verify.verify(ref, quant, {"x": (-1.0, 1.0)}, precision="float32")


# --------------------------------------------------------------------------
# real accelerators (skipped without one; run by hand on CUDA and on ROCm)
# --------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("method", ["crown", "alpha", "bab"])
def test_accelerator_matches_numpy_and_is_sound(method, precision):
    m = _convnet(seed=4, h=8)
    a = crown.bounds(m, BOX, method=method, budget=8, alpha_iters=15)["y"]
    b = crown.bounds(
        m,
        BOX,
        method=method,
        budget=8,
        alpha_iters=15,
        device="cuda",
        precision=precision,
    )["y"]
    if precision == "float64":
        np.testing.assert_allclose(b.lo, a.lo, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(b.hi, a.hi, rtol=1e-8, atol=1e-8)
    else:
        assert float((b.hi - b.lo).sum()) <= 1.02 * float((a.hi - a.lo).sum())
    _sample_inside(m, b, BOX["x"])


@needs_cuda
def test_accelerator_float32_contains_float64_on_an_affine_net():
    m = _mlp(seed=3, n_in=8, widths=(64, 64), n_out=16, relu=False)
    exact = crown.bounds(m, BOX, method="crown")["y"]
    got = crown.bounds(m, BOX, method="crown", device="cuda", precision="float32")["y"]
    assert np.all(got.lo <= exact.lo) and np.all(got.hi >= exact.hi)
