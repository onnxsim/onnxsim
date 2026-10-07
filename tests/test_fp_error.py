"""Tests for onnxsim.fp_error (certified floating-point roundoff bounds).

Soundness is checked empirically against real executions: onnxruntime in float32 (graph
optimisations both off and on) and numpy float16 (exact per-op rounding, where roundoff is
large enough to see) against a float64 reference of the same graph. A certified bound that is
ever exceeded is a bug; a bound that is absurdly loose on a single op is also tested for.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

import onnxsim
from onnxsim import fp_error

# opset 15, not 13: onnx's ReferenceEvaluator disagrees with onnxruntime on opset-13
# BatchNormalization (see tests/test_certify.py), and Softmax below needs opset >= 13.
OPSET = 15


def _model(body, init=None, dtype=np.float32, opset=OPSET):
    m = parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')
    m.graph.initializer.extend(
        numpy_helper.from_array(v.astype(dtype), k) for k, v in (init or {}).items()
    )
    return m


def _f64(m):
    """The same graph evaluated in float64 -- the real-number reference."""
    c = onnx.ModelProto()
    c.CopyFrom(m)
    for t in c.graph.initializer:
        if t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            t.CopyFrom(
                numpy_helper.from_array(
                    numpy_helper.to_array(t).astype(np.float64), t.name
                )
            )
    for vi in list(c.graph.input) + list(c.graph.output):
        if vi.type.tensor_type.elem_type in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
        ):
            vi.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    del c.graph.value_info[:]
    return c


def _ort(m, x, optimize=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    s = ort.InferenceSession(
        m.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return dict(zip([o.name for o in s.get_outputs()], s.run(None, {"x": x})))


def _worst_ratio(m, bound, sample, n=60, optimize=False, ref_run=None):
    """max over samples/outputs/elements of observed_error / certified_bound (must stay <= 1)."""
    ref = ReferenceEvaluator(_f64(m))
    names = [o.name for o in m.graph.output]
    worst = 0.0
    in_dtype = onnx.helper.tensor_dtype_to_np_dtype(
        m.graph.input[0].type.tensor_type.elem_type
    )
    for _ in range(n):
        # The bound's contract: inputs are numbers already representable in the model's dtype.
        # Quantise the sample first so the reference sees exactly what the model sees.
        x = sample().astype(in_dtype)
        got = _ort(m, x, optimize) if ref_run is None else ref_run(m, x)
        want = dict(zip(names, ref.run(None, {"x": x.astype(np.float64)})))
        for k in names:
            err = np.abs(np.asarray(got[k], dtype=np.float64) - want[k])
            b = bound.outputs[k]
            with np.errstate(all="ignore"):
                r = np.where(b > 0, err / b, np.where(err > 0, np.inf, 0.0))
            worst = max(worst, float(np.max(r)))
    return worst


def _w(rng, *shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# ---- constants ---------------------------------------------------------------


def test_precision_constants_and_gamma():
    assert fp_error.PRECISIONS["fp32"].u == 2.0**-24
    assert (
        fp_error.PRECISIONS["fp16"].u == 2.0**-11
        and fp_error.PRECISIONS["fp16"].fmax == 65504.0
    )
    assert fp_error.PRECISIONS["bf16"].u == 2.0**-8
    assert fp_error._gamma(100, 2.0**-24) == pytest.approx(100 * 2.0**-24, rel=1e-5)
    assert fp_error._gamma(2**24, 2.0**-24) == float("inf")  # n*u >= 1: no bound
    with pytest.raises(ValueError, match="unknown precision"):
        fp_error.roundoff_bound(
            _model("m (float[2] x) => (float[2] y) { y = Relu(x) }"), precision="fp8"
        )


# ---- single ops: near-tight and never exceeded (float16 emulated exactly) ---------


def _f16_run(m, x):
    return dict(
        zip(
            [o.name for o in m.graph.output],
            ReferenceEvaluator(m).run(None, {"x": x.astype(np.float16)}),
        )
    )


def _f16_model(body, init):
    return _model(body.replace("float[", "float16["), init, dtype=np.float16)


@pytest.mark.parametrize(
    "body,init",
    [
        (
            "m (float[64] x) => (float[64] y) { y = Add(x, C) }",
            {"C": np.linspace(-1, 1, 64)},
        ),
        (
            "m (float[64] x) => (float[64] y) { y = Mul(x, C) }",
            {"C": np.linspace(0.3, 1.7, 64)},
        ),
        (
            "m (float[64] x) => (float[64] y) { y = Sub(x, C) }",
            {"C": np.linspace(-1, 1, 64)},
        ),
        (
            "m (float[64] x) => (float[64] y) { y = Div(x, C) }",
            {"C": np.linspace(0.5, 2.0, 64)},
        ),
    ],
)
def test_elementwise_ops_are_sound_and_nearly_tight_in_fp16(body, init):
    m = _f16_model(body, init)
    rng = np.random.default_rng(0)
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, precision="fp16")
    ratio = _worst_ratio(m, b, lambda: rng.uniform(-1, 1, 64), n=400, ref_run=_f16_run)
    assert ratio <= 1.0, f"bound violated: observed/bound = {ratio}"
    assert ratio > 0.2, (
        f"a single rounding should come within ~5x of the bound, got {ratio}"
    )


def test_dot_product_bound_holds_for_sequential_and_wider_accumulation_in_fp16():
    # K=256 fp16 MatMul: numpy may accumulate in a wider format than fp16; the any-order bound
    # must hold either way. Roundoff is ~1e-3 here, so a missing factor would be visible.
    rng = np.random.default_rng(1)
    k = 256
    w = (rng.standard_normal((k, 8)) / np.sqrt(k)).astype(np.float16)
    m = _f16_model(
        f"m (float[1,{k}] x) => (float[1,8] y) {{ y = MatMul(x, W) }}", {"W": w}
    )
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, precision="fp16")
    ratio = _worst_ratio(
        m, b, lambda: rng.uniform(-1, 1, (1, k)), n=200, ref_run=_f16_run
    )
    assert ratio <= 1.0, ratio

    # And a genuinely sequential fp16 accumulation (every add rounds to fp16), the textbook case.
    ref = ReferenceEvaluator(_f64(m))
    worst = 0.0
    for _ in range(100):
        x = rng.uniform(-1, 1, (1, k)).astype(np.float16)
        acc = np.zeros(8, dtype=np.float16)
        for i in range(k):
            acc = (acc + (x[0, i] * w[i]).astype(np.float16)).astype(np.float16)
        want = ref.run(None, {"x": x.astype(np.float64)})[0][0]
        worst = max(
            worst,
            float(np.max(np.abs(acc.astype(np.float64) - want) / b.outputs["y"][0])),
        )
    assert worst <= 1.0, worst
    # Sequential random-sign accumulation is ~sqrt(K) better than the any-order worst case, so the
    # observed error is only a small fraction of the bound -- but a visible one (not 1e-6).
    assert worst > 1e-3


def test_fp16_overflow_is_reported_as_unbounded():
    m = _f16_model("m (float[4] x) => (float[4] y) { y = Mul(x, x) }", {})
    b = fp_error.roundoff_bound(
        m, {"x": (-300.0, 300.0)}, precision="fp16"
    )  # 300^2 = 90000 > 65504
    assert not b.bounded and any("exceed the largest finite" in n for n in b.notes)
    ok = fp_error.roundoff_bound(
        m, {"x": (-200.0, 200.0)}, precision="fp16"
    )  # 40000 < 65504
    assert ok.bounded


def test_division_by_a_range_containing_zero_is_unbounded():
    m = _model("m (float[4] x) => (float[4] y) { y = Div(x, x) }")
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)})
    assert not b.bounded and any("contain 0" in n for n in b.notes)
    assert fp_error.roundoff_bound(m, {"x": (1.0, 2.0)}).bounded


# ---- library functions: the documented hypothesis, checked against ORT -------------


@pytest.mark.parametrize(
    "op,lo,hi",
    [
        ("Sigmoid", -20.0, 20.0),
        ("Tanh", -10.0, 10.0),
        ("Exp", -20.0, 20.0),
        ("Erf", -6.0, 6.0),
        ("Sqrt", 1e-6, 1e4),
        ("Log", 1e-3, 1e4),
        ("Softplus", -20.0, 20.0),
    ],
)
def test_default_libm_model_covers_onnxruntime(op, lo, hi):
    n = 20000
    m = _model(f"m (float[{n}] x) => (float[{n}] y) {{ y = {op}(x) }}")
    # one pass over a dense grid is enough: the library error is a function of the argument only
    x = np.linspace(lo, hi, n).astype(np.float32)
    b = fp_error.roundoff_bound(m, {"x": (lo, hi)}, tight=False)
    ref = ReferenceEvaluator(_f64(m))
    err = np.abs(
        _ort(m, x)["y"].astype(np.float64)
        - ref.run(None, {"x": x.astype(np.float64)})[0]
    )
    assert np.all(err <= b.outputs["y"]), (
        f"{op}: observed {err.max():.3e} > bound {b.outputs['y'].max():.3e}"
    )
    # (the bound is built from the ORT measurement x ~3, so a gross loss of tightness is a regression)
    assert np.max(err) > 0 and np.max(b.outputs["y"]) < 1e3 * np.max(err) + 1e-6


def test_custom_libm_model_is_honoured():
    m = _model("m (float[8] x) => (float[8] y) { y = Exp(x) }")
    loose = fp_error.roundoff_bound(
        m,
        {"x": (-1.0, 1.0)},
        lib=fp_error.LibmModel(rel={**fp_error.DEFAULT_LIBM.rel, "Exp": 400.0}),
        tight=False,
    )
    tight = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, tight=False)
    assert loose.worst > 50 * tight.worst


# ---- whole graphs vs onnxruntime ---------------------------------------------------


def _cnn(rng, k=4):
    return _model(
        f"""m (float[1,3,12,12] x) => (float[1,{k},10,10] y) {{
          c = Conv(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          y = Relu(b) }}""",
        dict(
            W=_w(rng, k, 3, 3, 3, scale=0.3),
            B=_w(rng, k),
            g=rng.uniform(0.5, 1.5, k),
            be=_w(rng, k),
            mu=_w(rng, k),
            var=rng.uniform(0.5, 2, k),
        ),
    )


def _mlp(rng, widths):
    body, init, prev = [], {}, "x"
    for i in range(len(widths) - 1):
        init[f"W{i}"] = _w(rng, widths[i], widths[i + 1], scale=1 / np.sqrt(widths[i]))
        init[f"B{i}"] = _w(rng, widths[i + 1], scale=0.1)
        body.append(f"a{i} = MatMul({prev}, W{i})\n b{i} = Add(a{i}, B{i})")
        prev = f"b{i}"
        if i < len(widths) - 2:
            body.append(f"r{i} = Relu(b{i})")
            prev = f"r{i}"
    body.append(f"y = Identity({prev})")
    return _model(
        f"m (float[1,{widths[0]}] x) => (float[1,{widths[-1]}] y) {{ "
        + "\n".join(body)
        + " }",
        init,
    )


def _residual(rng, k=4):
    return _model(
        f"""m (float[1,{k},10,10] x) => (float[1,5] y) {{
          c = Conv<pads=[1,1,1,1]>(x, W1, B1)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          r = Relu(b)
          c2 = Conv<pads=[1,1,1,1]>(r, W2, B2)
          s = Add(c2, x)
          r2 = Relu(s)
          p = GlobalAveragePool(r2)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W3, B3) }}""",
        dict(W1=_w(rng, k, k, 3, 3, scale=0.2), B1=_w(rng, k, scale=0.1), g=rng.uniform(0.5, 1.5, k), be=_w(rng, k, scale=0.1), mu=_w(rng, k, scale=0.1),
             var=rng.uniform(0.5, 2, k), W2=_w(rng, k, k, 3, 3, scale=0.2), B2=_w(rng, k, scale=0.1), W3=_w(rng, 5, k), B3=_w(rng, 5, scale=0.1)),
    )  # fmt: skip


def _smooth(rng):
    return _model(
        """m (float[1,32] x) => (float[1,10] y, float[1,10] p) {
          a = Gemm(x, W1, B1)
          t = Tanh(a)
          b = Gemm(t, W2, B2)
          y = Sigmoid(b)
          p = Softmax(b) }""",
        dict(
            W1=_w(rng, 32, 64, scale=0.2),
            B1=_w(rng, 64, scale=0.1),
            W2=_w(rng, 64, 10, scale=0.2),
            B2=_w(rng, 10, scale=0.1),
        ),
    )


_GRAPHS = {
    "conv_bn_relu": (_cnn, (1, 3, 12, 12)),
    "mlp": (lambda r: _mlp(r, [64, 96, 96, 10]), (1, 64)),
    "residual": (_residual, (1, 4, 10, 10)),
    "tanh_sigmoid_softmax": (_smooth, (1, 32)),
}


@pytest.mark.parametrize("name", sorted(_GRAPHS))
@pytest.mark.parametrize("optimize", [False, True])
def test_graph_bound_is_never_exceeded_by_onnxruntime(name, optimize):
    build, shape = _GRAPHS[name]
    rng = np.random.default_rng(7)
    m = build(rng)
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)})
    assert b.bounded, b.notes

    def sample():
        # a third of the draws are +-1 corners: they maximise accumulation
        return (
            rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
            if rng.random() < 1 / 3
            else rng.uniform(-1, 1, shape).astype(np.float32)
        )

    ratio = _worst_ratio(m, b, sample, n=45, optimize=optimize)
    assert ratio <= 1.0, (
        f"{name}: bound violated (observed/bound = {ratio:.3f}, ort optimisations={optimize})"
    )


def test_single_layer_bound_is_within_two_orders_of_magnitude():
    # One Conv+BN+Relu: the worst case is not much worse than what is observed.
    rng = np.random.default_rng(3)
    m = _cnn(rng)
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)})
    ref = ReferenceEvaluator(_f64(m))
    obs = 0.0
    for _ in range(40):
        x = rng.choice([-1.0, 1.0], size=(1, 3, 12, 12)).astype(np.float32)
        obs = max(
            obs,
            float(
                np.max(
                    np.abs(
                        _ort(m, x)["y"] - ref.run(None, {"x": x.astype(np.float64)})[0]
                    )
                )
            ),
        )
    assert 0 < obs <= b.worst < 100 * obs


# ---- the tight (zonotope-difference) pass ------------------------------------------


def test_tight_pass_never_loosens_and_tightens_a_deep_net():
    m = _mlp(np.random.default_rng(5), [48, 64, 64, 64, 8])
    fwd = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, tight=False)
    tgt = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, tight=True)
    assert np.all(tgt.outputs["y"] <= fwd.outputs["y"])
    assert tgt.worst < 0.9 * fwd.worst, (fwd.worst, tgt.worst)
    assert any("tight pass" in n for n in tgt.notes)


def test_tight_pass_is_skipped_with_a_note_when_not_applicable():
    rng = np.random.default_rng(6)
    m = _model(
        "m (float[1,4] x) => (float[1,3] y) { y = Gemm<alpha=2.0>(x, W, C) }",
        {"W": _w(rng, 4, 3), "C": _w(rng, 3)},
    )
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)})
    assert b.bounded and any("tight pass skipped" in n for n in b.notes)
    # fp16 analysis of fp32 weights rounds the weights: that is propagation, not noise -> skipped too
    h = fp_error.roundoff_bound(
        _mlp(rng, [8, 8, 2]), {"x": (-1.0, 1.0)}, precision="fp16"
    )
    assert h.bounded and any("not exactly representable" in n for n in h.notes)


# ---- ranges, unsupported ops, unbounded inputs -------------------------------------


def test_unbounded_input_gives_unbounded_error_not_a_wrong_one():
    b = fp_error.roundoff_bound(_mlp(np.random.default_rng(0), [4, 4, 2]))  # no box
    assert not b.bounded


def test_unsupported_op_is_infinite_with_a_note():
    m = _model("m (float[4] x) => (float[4] y) { a = Add(x, x)  y = Sign(a) }")
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)})
    assert (
        not b.bounded
        and "Sign" in b.unsupported
        and any("no roundoff rule for Sign" in n for n in b.notes)
    )


def test_exact_ops_have_no_roundoff():
    m = _model("m (float[2,3] x) => (float[3,2] y) { a = Transpose(x)  y = Relu(a) }")
    assert fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}).worst == 0.0


def test_softmax_before_opset_13_is_not_modelled():
    m = _model("m (float[1,4] x) => (float[1,4] y) { y = Softmax(x) }", opset=11)
    b = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)})
    assert not b.bounded and "Softmax" in b.unsupported


def test_ranges_option_validation_and_zonotope_ranges_are_tighter():
    m = _mlp(np.random.default_rng(9), [32, 48, 48, 4])
    with pytest.raises(ValueError, match="unknown ranges"):
        fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, ranges="magic")
    a = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, ranges="interval", tight=False)
    z = fp_error.roundoff_bound(m, {"x": (-1.0, 1.0)}, ranges="zonotope", tight=False)
    assert z.worst <= a.worst


# ---- tolerance_for -----------------------------------------------------------------


def test_tolerance_for_covers_a_real_simplify_and_rejects_a_wrong_rewrite():
    rng = np.random.default_rng(11)
    orig = _cnn(rng)
    sim, _ = onnxsim.simplify(orig, certify=False)
    assert "BatchNormalization" not in {n.op_type for n in sim.graph.node}
    box = {"x": (-1.0, 1.0)}
    tol = fp_error.tolerance_for(orig, sim, box)
    assert tol.bounded and tol.atol["y"] == pytest.approx(
        tol.real_difference["y"] + tol.roundoff_orig["y"] + tol.roundoff_simplified["y"]
    )
    worst = 0.0
    for _ in range(40):
        x = rng.choice([-1.0, 1.0], size=(1, 3, 12, 12)).astype(np.float32)
        worst = max(
            worst, float(np.max(np.abs(_ort(orig, x)["y"] - _ort(sim, x)["y"])))
        )
    assert worst <= tol.atol["y"]
    # the tolerance is a real, certified replacement for an arbitrary constant: it is small
    assert tol.atol["y"] < 1e-3

    bad = onnx.ModelProto()
    bad.CopyFrom(orig)
    for t in bad.graph.initializer:
        if t.name == "be":
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t) + 0.25, "be"))
    wrong = fp_error.tolerance_for(orig, bad, box)
    assert (
        wrong.real_difference["y"] >= 0.2
    )  # the rewrite is wrong; roundoff does not hide it
