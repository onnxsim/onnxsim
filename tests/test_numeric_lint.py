"""Tests for onnxsim.numeric_lint (interval-based numerical-safety linter).

Each rule has a positive and a negative case; witnesses are replayed on onnxruntime; and a property
test checks that the bounds the linter used (including the ones it derives by *cutting* the graph at
Gather / LayerNormalization / Cast / a decomposed softmax) enclose every value onnxruntime produces.
"""

import json

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim import numeric_lint as L
from onnxsim import ranges as R


def _model(body, initializer=None, opset=17, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _rules(report, severity=None):
    return sorted(f.rule for f in report.by(severity))


def _serious(report):
    """Findings that are not merely informational."""
    return [f for f in report.findings if f.severity != L.INFO]


def _assert_replays(report):
    """Every confirmed finding's stored witness re-runs on onnxruntime and breaks its condition."""
    assert report.confirmed
    for f in report.confirmed:
        i = report.findings.index(f)
        assert f._check is not None
        confirmed, _, _ = f._check(report.replay(i))
        assert confirmed, (
            f"finding {f.rule} was reported confirmed but its witness does not replay"
        )


# ---- div-by-zero --------------------------------------------------------------


def test_div_by_zero_positive_with_confirmed_witness():
    m = _model("g (float[4] a, float[4] b) => (float[4] y) { y = Div(a, b) }")
    rep = L.lint(m, {"a": (1.0, 2.0), "b": (-1.0, 1.0)}, witness=30)
    assert _rules(rep) == ["div-by-zero"]
    f = rep.findings[0]
    assert f.severity == L.CAN_FAIL and f.interval[0] <= 0 <= f.interval[1]
    assert f.assumption["b"] == (-1.0, 1.0) and f.fix
    assert (
        f.certainty == L.CONFIRMED
    )  # the box midpoint makes the denominator exactly 0
    _assert_replays(rep)


def test_div_negative_case_is_clean_even_with_witness_search():
    m = _model("g (float[4] a, float[4] b) => (float[4] y) { y = Div(a, b) }")
    rep = L.lint(m, {"a": (1.0, 2.0), "b": (1.0, 2.0)}, witness=30)
    assert rep.findings == [] and rep.ok


def test_possible_stays_possible_when_the_failure_input_is_not_reached():
    # den = x - 0.37 is 0 only at one float; no sampled/mid/corner point lands exactly there
    m = _model(
        "g (float[4] x) => (float[4] y) { c = Constant<value = float {0.37}>() d = Sub(x, c) "
        "one = Constant<value = float {1.0}>() y = Div(one, d) }"
    )
    rep = L.lint(m, {"x": (-1.0, 1.0)}, witness=40)
    (f,) = rep.by(L.CAN_FAIL)
    assert f.rule == "div-by-zero" and f.certainty == L.POSSIBLE
    assert (
        f.observed is not None and f.observed > 0
    )  # closest approach seen, still not zero
    assert not rep.confirmed


def test_witness_data_confirms_what_random_search_cannot():
    m = _model(
        "g (float[4] x) => (float[4] y) { c = Constant<value = float {0.37}>() d = Sub(x, c) "
        "one = Constant<value = float {1.0}>() y = Div(one, d) }"
    )
    # a real sample that happens to hit the singularity (float32(0.37) - float32(0.37) == 0)
    sample = {"x": np.full(4, 0.37, np.float32)}
    rep = L.lint(m, {"x": (-1.0, 1.0)}, witness=10, witness_data=[sample])
    (f,) = rep.by(L.CAN_FAIL)
    assert f.certainty == L.CONFIRMED
    _assert_replays(rep)
    # data that does not break anything confirms nothing
    harmless = {"x": np.full(4, 0.9, np.float32)}
    rep2 = L.lint(m, {"x": (-1.0, 1.0)}, witness=10, witness_data=[harmless])
    assert rep2.confirmed == []


def test_div_confirmed_through_floor_denominator():
    m = _model(
        "g (float[4] x) => (float[4] y) { d = Floor(x) one = Constant<value = float {1.0}>() "
        "y = Div(one, d) }"
    )
    rep = L.lint(m, {"x": (-1.0, 1.0)}, witness=30)
    assert "div-by-zero" in _rules(rep) and rep.confirmed
    _assert_replays(rep)


def test_a_denominator_interval_that_merely_touches_zero_is_flagged():
    m = _model("g (float[4] a, float[4] b) => (float[4] y) { y = Div(a, b) }")
    for box in ((0.0, 1.0), (-1.0, 0.0)):
        rep = L.lint(m, {"a": (1.0, 2.0), "b": box}, witness=20)
        assert _rules(rep) == ["div-by-zero"], box
    assert L.lint(m, {"a": (1.0, 2.0), "b": (0.01, 1.0)}).findings == []


def test_reciprocal_zero_case():
    m = _model("g (float[4] x) => (float[4] y) { y = Reciprocal(x) }")
    assert _rules(L.lint(m, {"x": (-1.0, 1.0)})) == ["div-by-zero"]
    assert L.lint(m, {"x": (0.5, 1.0)}).findings == []


# ---- log / sqrt / pow ----------------------------------------------------------


def test_log_domain():
    m = _model("g (float[4] x) => (float[4] y) { y = Log(x) }")
    rep = L.lint(m, {"x": (-1.0, 1.0)}, witness=30)
    assert _rules(rep) == ["log-domain"] and rep.findings[0].certainty == L.CONFIRMED
    _assert_replays(rep)
    assert L.lint(m, {"x": (0.1, 2.0)}, witness=30).findings == []
    # touching 0 is still a hazard: log(0) = -inf
    assert _rules(L.lint(m, {"x": (0.0, 2.0)})) == ["log-domain"]


def test_sqrt_domain():
    m = _model("g (float[4] x) => (float[4] y) { y = Sqrt(x) }")
    rep = L.lint(m, {"x": (-1.0, 1.0)}, witness=30)
    assert _rules(rep) == ["sqrt-domain"] and rep.findings[0].certainty == L.CONFIRMED
    _assert_replays(rep)
    assert L.lint(m, {"x": (0.0, 2.0)}, witness=30).findings == []


def test_pow_domain_fractional_and_negative_exponents():
    frac = _model(
        "g (float[4] x) => (float[4] y) { e = Constant<value = float {0.5}>() y = Pow(x, e) }"
    )
    assert _rules(L.lint(frac, {"x": (-1.0, 1.0)}, witness=30)) == ["pow-domain"]
    assert L.lint(frac, {"x": (0.0, 1.0)}).findings == []
    neg = _model(
        "g (float[4] x) => (float[4] y) { e = Constant<value = float {-1.0}>() y = Pow(x, e) }"
    )
    rep = L.lint(neg, {"x": (-1.0, 1.0)}, witness=30)
    assert _rules(rep) == ["pow-domain"] and rep.findings[0].certainty == L.CONFIRMED
    assert L.lint(neg, {"x": (1.0, 2.0)}).findings == []
    # an integer exponent on a negative base is fine
    ok = _model(
        "g (float[4] x) => (float[4] y) { e = Constant<value = float {2.0}>() y = Pow(x, e) }"
    )
    assert L.lint(ok, {"x": (-3.0, 3.0)}).findings == []


# ---- exp / softmax / saturation --------------------------------------------------


def test_exp_overflow_depends_on_dtype():
    m = _model("g (float[4] x) => (float[4] y) { y = Exp(x) }")
    assert L.lint(m, {"x": (0.0, 12.0)}, dtype="fp32").findings == []
    assert L.lint(m, {"x": (0.0, 12.0)}, dtype="bf16").findings == []
    rep16 = L.lint(m, {"x": (0.0, 12.0)}, dtype="fp16", witness=30)
    assert (
        _rules(rep16) == ["exp-overflow"] and rep16.findings[0].certainty == L.CONFIRMED
    )
    _assert_replays(rep16)
    rep32 = L.lint(m, {"x": (0.0, 100.0)}, dtype="fp32", witness=30)
    assert (
        _rules(rep32) == ["exp-overflow"] and rep32.findings[0].certainty == L.CONFIRMED
    )
    assert L.lint(m, {"x": (0.0, 10.0)}, dtype="fp16").findings == []


_SOFTMAX_PLAIN = "g (float[1,8] x) => (float[1,8] y) { e = Exp(x) s = ReduceSum<keepdims=1>(e, ax) y = Div(e, s) }"
_SOFTMAX_STABLE = (
    "g (float[1,8] x) => (float[1,8] y) { m = ReduceMax<keepdims=1, axes=[1]>(x) d = Sub(x, m) "
    "e = Exp(d) s = ReduceSum<keepdims=1>(e, ax) y = Div(e, s) }"
)


def test_decomposed_softmax_without_max_subtraction_gets_a_tailored_finding():
    ax = {"ax": np.array([1], dtype=np.int64)}
    rep = L.lint(
        _model(_SOFTMAX_PLAIN, ax), {"x": (-50.0, 50.0)}, dtype="fp16", witness=30
    )
    (f,) = rep.by(L.CAN_FAIL)
    assert f.rule == "exp-overflow" and "softmax" in f.message and "max" in f.fix
    assert f.certainty == L.CONFIRMED
    # the normalised output is still bounded to [0, 1], so the analysis continues past it
    assert rep.hull("y") == (0.0, 1.0)


def test_decomposed_softmax_with_max_subtraction_is_clean():
    ax = {"ax": np.array([1], dtype=np.int64)}
    for dtype in ("fp32", "fp16"):
        rep = L.lint(
            _model(_SOFTMAX_STABLE, ax), {"x": (-50.0, 50.0)}, dtype=dtype, witness=30
        )
        assert rep.by(L.CAN_FAIL) == [], rep
        assert rep.hull("y") == (0.0, 1.0)


def test_native_softmax_is_info_only_in_fp16():
    m = _model("g (float[1,8] x) => (float[1,8] y) { y = Softmax<axis=-1>(x) }")
    rep = L.lint(m, {"x": (-20.0, 20.0)}, dtype="fp16")
    assert [f.severity for f in rep.findings] == [L.INFO] and rep.findings[
        0
    ].implementation_dependent
    assert L.lint(m, {"x": (-20.0, 20.0)}, dtype="fp32").findings == []
    assert L.lint(m, {"x": (-2.0, 2.0)}, dtype="fp16").findings == []


def test_sigmoid_tanh_softplus_saturation_is_implementation_dependent():
    for op in ("Sigmoid", "Tanh", "Softplus"):
        m = _model(f"g (float[4] x) => (float[4] y) {{ y = {op}(x) }}")
        rep = L.lint(m, {"x": (-20.0, 20.0)}, dtype="fp16")
        assert _rules(rep) == ["saturation"], op
        assert rep.findings[0].severity == L.CAN_LOSE_PRECISION
        assert rep.findings[0].implementation_dependent
        assert L.lint(m, {"x": (-5.0, 5.0)}, dtype="fp16").findings == []
        assert L.lint(m, {"x": (-20.0, 20.0)}, dtype="fp32").findings == []


# ---- normalisation -------------------------------------------------------------


def _layernorm(n, eps=1e-5):
    body = (
        f"g (float[2,{n}] x) => (float[2,{n}] y) "
        f"{{ y = LayerNormalization<axis=-1, epsilon={eps}>(x, s, b) }}"
    )
    return _model(body, {"s": np.ones(n, np.float32), "b": np.zeros(n, np.float32)})


def test_layernorm_variance_overflow_in_fp16():
    rep = L.lint(_layernorm(8), {"x": (-300.0, 300.0)}, dtype="fp16", witness=30)
    (f,) = [f for f in rep.by(L.CAN_FAIL) if f.rule == "norm-variance"]
    assert f.severity == L.CAN_FAIL and not f.implementation_dependent
    assert f.interval[1] == pytest.approx(300.0**2) and f.certainty == L.CONFIRMED
    _assert_replays(rep)
    assert _serious(L.lint(_layernorm(8), {"x": (-300.0, 300.0)}, dtype="fp32")) == []
    assert _serious(L.lint(_layernorm(8), {"x": (-10.0, 10.0)}, dtype="fp16")) == []


def test_layernorm_sum_of_squares_overflow_is_flagged_as_implementation_dependent():
    # variance <= 1e4 fits fp16, but N * variance = 7.7e6 does not: only a kernel that sums first fails
    rep = L.lint(_layernorm(768), {"x": (-100.0, 100.0)}, dtype="fp16")
    (f,) = [f for f in rep.by(L.CAN_FAIL) if f.rule == "norm-variance"]
    assert f.implementation_dependent and "sum" in f.message


def test_layernorm_eps_underflow_with_constant_rows():
    # x in [-1, 1] lets a row be constant (variance 0); eps = 0 then gives 0 * inf
    rep = L.lint(_layernorm(8, eps=0.0), {"x": (-1.0, 1.0)}, dtype="fp32", witness=30)
    (f,) = [f for f in rep.findings if f.rule == "norm-variance"]
    assert f.severity == L.CAN_FAIL and "epsilon" in f.message
    assert (
        L.lint(_layernorm(8, eps=1e-5), {"x": (-1.0, 1.0)}, dtype="fp32").findings == []
    )
    # a tiny eps underflows to 0 in fp16 (smallest subnormal 6e-8)
    rep16 = L.lint(_layernorm(8, eps=1e-9), {"x": (-1.0, 1.0)}, dtype="fp16")
    assert any(
        f.rule == "norm-variance" and f.severity == L.CAN_FAIL for f in rep16.findings
    )
    # rows whose per-element boxes are disjoint can never be constant, so epsilon = 0 is safe there
    lo = np.arange(8, dtype=np.float64) * 10.0
    assert (
        _serious(L.lint(_layernorm(8, eps=0.0), {"x": (lo, lo + 1.0)}, dtype="fp32"))
        == []
    )


def test_rmsnorm_variance_overflow():
    m = _model(
        "g (float[2,8] x) => (float[2,8] y) { y = RMSNormalization<axis=-1, epsilon=1e-5>(x, s) }",
        {"s": np.ones(8, np.float32)},
        opset=23,
    )
    rep = L.lint(m, {"x": (-300.0, 300.0)}, dtype="fp16")
    assert any(
        f.rule == "norm-variance" and f.severity == L.CAN_FAIL for f in rep.findings
    )
    assert L.lint(m, {"x": (-10.0, 10.0)}, dtype="fp16").by(L.CAN_FAIL) == []


# ---- generic range rules ---------------------------------------------------------


def _matmul_ones(k):
    return _model(
        f"g (float[1,{k}] x) => (float[1,1] y) {{ a = MatMul(x, W) b = Relu(a) y = Mul(b, two) }}",
        {"W": np.ones((k, 1), np.float32), "two": np.array(2.0, np.float32)},
    )


def test_range_overflow_is_reported_where_it_arises_not_downstream():
    rep = L.lint(_matmul_ones(64), {"x": (0.0, 2000.0)}, dtype="fp16", witness=30)
    over = [f for f in rep.findings if f.rule == "range-overflow"]
    assert (
        len(over) == 1 and over[0].tensor == "a"
    )  # MatMul output; Relu/Mul are consequences
    assert over[0].certainty == L.CONFIRMED and rep.consequences >= 1
    _assert_replays(rep)
    # the same graph is fine in fp32, and in fp16 for a smaller box
    assert _serious(L.lint(_matmul_ones(64), {"x": (0.0, 2000.0)}, dtype="fp32")) == []
    assert _serious(L.lint(_matmul_ones(64), {"x": (0.0, 100.0)}, dtype="fp16")) == []


def test_const_overflow_catches_a_minus_3e38_mask_in_fp16():
    m = _model(
        "g (float[1,4] x, bool[1,4] keep) => (float[1,4] y) "
        "{ neg = Constant<value = float {-3.4028235e38}>() y = Where(keep, x, neg) }"
    )
    rep16 = L.lint(m, {"x": (-1.0, 1.0), "keep": (0, 1)}, dtype="fp16")
    assert "const-overflow" in _rules(rep16, L.CAN_FAIL)
    # no input is involved: the constant is read straight from the model, so it is confirmed
    # by construction (there is nothing to replay)
    (const,) = [f for f in rep16.findings if f.rule == "const-overflow"]
    assert const.certainty == L.CONFIRMED and rep16.confirmed == [const]
    with pytest.raises(KeyError):
        rep16.replay(rep16.findings.index(const))
    assert (
        "mask" in [f for f in rep16.findings if f.rule == "const-overflow"][0].message
    )
    assert "const-overflow" not in _rules(
        L.lint(m, {"x": (-1.0, 1.0), "keep": (0, 1)}, dtype="fp32")
    )
    ok = _model(
        "g (float[1,4] x, bool[1,4] keep) => (float[1,4] y) "
        "{ neg = Constant<value = float {-1e4}>() y = Where(keep, x, neg) }"
    )
    assert "const-overflow" not in _rules(
        L.lint(ok, {"x": (-1.0, 1.0), "keep": (0, 1)}, dtype="fp16")
    )


def test_a_negligible_fraction_of_flushed_weights_is_only_info():
    w = np.ones(5000, np.float32)
    w[0] = 1e-9  # 1 of 5000 nonzero values flushes (0.02%), below the warning fraction
    m = _model("g (float[1,5000] x) => (float[1,5000] y) { y = Mul(x, w) }", {"w": w})
    rep = L.lint(m, {"x": (-1.0, 1.0)}, dtype="fp16")
    assert [(f.rule, f.severity) for f in rep.findings] == [("range-underflow", L.INFO)]
    w[:100] = 1e-9  # 2% flush: now a real warning
    m = _model("g (float[1,5000] x) => (float[1,5000] y) { y = Mul(x, w) }", {"w": w})
    assert [
        f.severity for f in L.lint(m, {"x": (-1.0, 1.0)}, dtype="fp16").findings
    ] == [L.CAN_LOSE_PRECISION]


def test_const_underflow_in_fp16():
    m = _model(
        "g (float[1,3] x) => (float[1,3] y) { y = Mul(x, w) }",
        {"w": np.array([1e-9, 0.5, 2.0], np.float32)},
    )
    rep = L.lint(m, {"x": (-1.0, 1.0)}, dtype="fp16")
    (f,) = [f for f in rep.findings if f.rule == "range-underflow"]
    assert f.severity == L.CAN_LOSE_PRECISION and "flush" in f.message
    assert [f for f in L.lint(m, {"x": (-1.0, 1.0)}, dtype="fp32").findings] == []
    sub = _model(
        "g (float[1,2] x) => (float[1,2] y) { y = Mul(x, w) }",
        {"w": np.array([1e-5, 0.5], np.float32)},
    )
    assert [
        f.severity for f in L.lint(sub, {"x": (-1.0, 1.0)}, dtype="fp16").findings
    ] == [L.INFO]
    fine = _model(
        "g (float[1,2] x) => (float[1,2] y) { y = Mul(x, w) }",
        {"w": np.array([0.25, 0.5], np.float32)},
    )
    assert L.lint(fine, {"x": (-1.0, 1.0)}, dtype="fp16").findings == []


# ---- integer / cast ------------------------------------------------------------


def _matmul_integer(k):
    return _model(
        f"g (uint8[1,{k}] a) => (int32[1,1] y) {{ y = MatMulInteger(a, W) }}",
        {"W": np.full((k, 1), 127, np.int8)},
    )


def test_int32_wrap_positive_confirmed_by_exact_arithmetic_and_negative():
    rep = L.lint(_matmul_integer(70000), {"a": (0, 255)}, witness=20)
    (f,) = rep.findings
    assert f.rule == "int32-wrap" and f.severity == L.CAN_FAIL
    assert f.interval[1] == pytest.approx(70000 * 127 * 255)
    # confirmed from the *input* (replayed in numpy int64), never from the runtime's own overflow
    assert f.certainty == L.CONFIRMED
    _assert_replays(rep)
    assert L.lint(_matmul_integer(100), {"a": (0, 255)}, witness=20).findings == []
    # the declared activation range matters: a narrow range cannot wrap
    assert L.lint(_matmul_integer(70000), {"a": (0, 100)}).findings == []


def test_cast_to_a_narrower_integer_and_to_fp16():
    to8 = _model("g (float[4] x) => (int8[4] y) { y = Cast<to=3>(x) }")
    rep = L.lint(to8, {"x": (-200.0, 200.0)}, witness=20)
    assert _rules(rep) == ["cast-range"] and rep.findings[0].certainty == L.CONFIRMED
    _assert_replays(rep)
    assert L.lint(to8, {"x": (-100.0, 100.0)}, witness=20).findings == []
    to16 = _model("g (float[4] x) => (float16[4] y) { y = Cast<to=10>(x) }")
    assert _rules(L.lint(to16, {"x": (-1e5, 1e5)})) == ["cast-range"]
    assert L.lint(to16, {"x": (-1e3, 1e3)}).findings == []


# ---- info rules ------------------------------------------------------------------


def test_dead_clip_relu_and_decided_where():
    clip = _model(
        "g (float[4] x) => (float[4] y) { lo = Constant<value = float {0.0}>() "
        "hi = Constant<value = float {6.0}>() y = Clip(x, lo, hi) }"
    )
    rep = L.lint(clip, {"x": (1.0, 2.0)})
    assert [(f.rule, f.severity) for f in rep.findings] == [("dead-op", L.INFO)]
    assert L.lint(clip, {"x": (-1.0, 10.0)}).findings == []
    sat = L.lint(clip, {"x": (7.0, 9.0)})
    assert [f.rule for f in sat.findings] == [
        "dead-op"
    ] and "saturates" in sat.findings[0].message

    relu = _model("g (float[4] x) => (float[4] y) { y = Relu(x) }")
    assert "identity" in L.lint(relu, {"x": (0.5, 1.0)}).findings[0].message
    assert "always outputs 0" in L.lint(relu, {"x": (-2.0, -1.0)}).findings[0].message
    assert L.lint(relu, {"x": (-1.0, 1.0)}).findings == []

    where = _model(
        "g (float[4] a, float[4] b) => (float[4] y) { c = Greater(a, b) y = Where(c, a, b) }"
    )
    rep = L.lint(where, {"a": (2.0, 3.0), "b": (0.0, 1.0)})
    assert [f.rule for f in rep.findings] == [
        "dead-op"
    ] and "always true" in rep.findings[0].message
    assert L.lint(where, {"a": (0.0, 3.0), "b": (1.0, 2.0)}).findings == []


# ---- assumptions, unbounded inputs, annotations ------------------------------------


def test_unannotated_input_suppresses_findings_that_rest_on_it():
    m = _model("g (float[4] x) => (float[4] y) { y = Log(x) }")
    rep = L.lint(m)
    assert (
        rep.findings == []
        and rep.suppressed_unbounded == 1
        and rep.unannotated_inputs == ["x"]
    )
    shown = L.lint(m, include_unbounded=True)
    (f,) = shown.findings
    assert f.rule == "log-domain" and f.unbounded
    assert "no range" in str(rep).lower() or "without a range" in str(rep).lower()


def test_ranges_come_from_model_annotations():
    m = _model("g (float[4] x) => (float[4] y) { y = Log(x) }")
    R.set_range(m, "x", -1.0, 1.0)
    rep = L.lint(m)
    assert _rules(rep) == ["log-domain"] and rep.unannotated_inputs == []
    assert rep.findings[0].assumption == {"x": (-1.0, 1.0)}
    # an explicit argument overrides the annotation
    assert L.lint(m, {"x": (1.0, 2.0)}).findings == []


def test_bad_dtype_is_rejected():
    m = _model("g (float[4] x) => (float[4] y) { y = Relu(x) }")
    with pytest.raises(ValueError, match="dtype"):
        L.lint(m, {"x": (0, 1)}, dtype="fp8")


# ---- soundness of the cuts: bounds enclose what onnxruntime produces ---------------


def _transformer_ish():
    rng = np.random.default_rng(0)
    n, d = 6, 8
    emb = (rng.standard_normal((10, d)) * 2).astype(np.float32)
    return _model(
        f"""
        g (int64[1,{n}] ids) => (float[1,{n},{d}] y) {{
          h = Gather<axis=0>(emb, ids)
          ln = LayerNormalization<axis=-1, epsilon=1e-5>(h, s, b)
          f = Cast<to=1>(ids)
          f3 = Unsqueeze(f, ax)
          sc = Mul(ln, f3)
          q = MatMul(sc, W)
          e = Exp(q)
          z = ReduceSum<keepdims=1>(e, ax2)
          p = Div(e, z)
          gt = Greater(f, three)
          cond = Unsqueeze(gt, ax)
          y = Where(cond, p, neg)
        }}""",
        {
            "emb": emb,
            "s": (rng.standard_normal(d) * 0.5 + 1).astype(np.float32),
            "b": rng.standard_normal(d).astype(np.float32),
            "ax": np.array([2], dtype=np.int64),
            "three": np.array(3.0, dtype=np.float32),
            "neg": np.array(-50.0, dtype=np.float32),
            "W": (rng.standard_normal((d, d)) * 0.3).astype(np.float32),
            "ax2": np.array([2], dtype=np.int64),
        },
    )


def test_cut_bounds_enclose_every_value_onnxruntime_produces():
    m = _transformer_ish()
    rep = L.lint(m, {"ids": (0, 9)}, dtype="fp32", include_unbounded=True)
    # Gather, LayerNorm, Cast, Greater, Where and the softmax Div
    assert rep.cuts >= 5 and rep.unanalysed == []
    produced = [o for n in m.graph.node for o in n.output]
    ex = L._Exposed(m, produced)
    rng = np.random.default_rng(1)
    for _ in range(40):
        vals = ex.run({"ids": rng.integers(0, 10, (1, 6)).astype(np.int64)})
        for name in produced:
            hull = rep.hull(name)
            assert hull is not None, name
            v = np.asarray(vals[name], dtype=np.float64)
            pad = 1e-4 * (1.0 + np.abs(v))
            assert np.all(v >= hull[0] - pad) and np.all(v <= hull[1] + pad), (
                f"{name}: observed [{v.min():.4g}, {v.max():.4g}] escapes {hull}"
            )
    assert rep.hull("p") == (0.0, 1.0)  # the decomposed softmax
    assert rep.hull("y") == (-50.0, 1.0)  # Where: the union of both branches


# ---- witnesses never falsely confirm ------------------------------------------------


def test_no_finding_on_a_safe_graph_is_ever_confirmed():
    safe = [
        (
            _model("g (float[4] x) => (float[4] y) { y = Sqrt(x) }"),
            {"x": (0.5, 4.0)},
            "fp32",
        ),
        (
            _model("g (float[4] x) => (float[4] y) { y = Log(x) }"),
            {"x": (0.5, 4.0)},
            "fp16",
        ),
        (_layernorm(8), {"x": (-5.0, 5.0)}, "fp16"),
        (_matmul_ones(64), {"x": (0.0, 100.0)}, "fp16"),
    ]
    for m, rng, dtype in safe:
        rep = L.lint(m, rng, dtype=dtype, witness=40)
        assert rep.confirmed == [] and rep.by(L.CAN_FAIL) == []


def test_a_loose_bound_is_never_confirmed_however_long_the_search_runs():
    # true maximum 6e4 fits fp16; the interval bound 1.2e5 does not. No input can break it.
    m = _model(
        "g (float[1,2] x) => (float[1,1] y) { z = MatMul(x, W1) h = Relu(z) y = MatMul(h, W2) }",
        {
            "W1": np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32),
            "W2": np.array([[100.0], [100.0]], np.float32),
        },
    )
    rep = L.lint(m, {"x": (-300.0, 300.0)}, dtype="fp16", witness=200)
    (f,) = rep.by(L.CAN_FAIL)
    assert f.rule == "range-overflow" and f.certainty == L.POSSIBLE
    assert rep.confirmed == [] and f.witness_runs > 0
    assert f.observed is not None and f.observed <= 6.0e4 + 1e-3 < 65504


def test_possible_findings_are_not_upgraded_without_a_witness_and_observed_is_recorded():
    m = _matmul_ones(64)
    plain = L.lint(m, {"x": (0.0, 2000.0)}, dtype="fp16")
    (p,) = plain.by(L.CAN_FAIL)
    assert plain.confirmed == [] and plain.witness_runs == 0 and p.observed is None
    searched = L.lint(m, {"x": (0.0, 2000.0)}, dtype="fp16", witness=20)
    (f,) = searched.by(L.CAN_FAIL)
    assert f.certainty == L.CONFIRMED and f.observed is not None and f.observed > 65504
    assert f.witness_runs > 0 and searched.witness_runs == f.witness_runs


# ---- output formats and CLI ------------------------------------------------------------


def test_json_report_roundtrips_and_has_the_documented_fields():
    m = _model("g (float[4] a, float[4] b) => (float[4] y) { y = Div(a, b) }")
    rep = L.lint(m, {"a": (1.0, 2.0), "b": (-1.0, 1.0)}, witness=10)
    d = json.loads(rep.to_json())
    assert d["dtype"] == "fp32" and d["ok"] is False and d["counts"]["can-fail"] == 1
    assert d["counts"]["confirmed"] == 1
    (f,) = d["findings"]
    for key in (
        "rule",
        "severity",
        "node",
        "tensor",
        "message",
        "interval",
        "assumption",
        "fix",
        "certainty",
        "observed",
        "implementation_dependent",
    ):
        assert key in f
    assert f["assumption"]["b"] == [-1.0, 1.0]
    assert "CONFIRMED" in str(rep)


def test_cli_exit_status_and_json(tmp_path, capsys):
    bad = _model("g (float[4] x) => (float[4] y) { y = Log(x) }")
    good = _model("g (float[4] x) => (float[4] y) { y = Relu(x) }")
    pb, pg = tmp_path / "bad.onnx", tmp_path / "good.onnx"
    onnx.save(bad, str(pb))
    onnx.save(good, str(pg))
    assert L.main([str(pb), "--range", "x=-1,1", "--witness", "10"]) == 1
    assert "log-domain" in capsys.readouterr().out
    assert L.main([str(pg), "--range", "x=-1,1"]) == 0
    capsys.readouterr()
    assert L.main([str(pb), "--range", "x=-1,1", "--dtype", "fp16", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["dtype"] == "fp16"
    with pytest.raises(SystemExit):
        L.main([str(pb), "--range", "x=oops"])


# ---- CROWN refinement ------------------------------------------------------------------


def test_refine_with_crown_drops_a_finding_intervals_cannot_rule_out():
    # z2 = -z1, so y = 100 * (relu(z) + relu(-z)) = 100 * |z1| <= 100 * 600 = 6e4 fits fp16; plain
    # intervals treat the two relus independently (each up to 600) and report 1.2e5 > 65504
    m = _model(
        "g (float[1,2] x) => (float[1,1] y) { z = MatMul(x, W1) h = Relu(z) y = MatMul(h, W2) }",
        {
            "W1": np.array([[1.0, -1.0], [-1.0, 1.0]], np.float32),
            "W2": np.array([[100.0], [100.0]], np.float32),
        },
    )
    box = {"x": (-300.0, 300.0)}
    plain = L.lint(m, box, dtype="fp16")
    assert "range-overflow" in _rules(plain)
    refined = L.lint(m, box, dtype="fp16", refine=True)
    assert "range-overflow" not in _rules(refined)
    assert refined.refined_away >= 1 and all(f.refined for f in refined.findings)
