"""``certify(atol="certified")``: a certified fp32 tolerance next to the real-arithmetic proof.

The default mode is untouched (checked here); the certified mode must never prove more than the
real-arithmetic proof, never prove without a finite tolerance, and its tolerance must cover the
differences actually observed between the two models' fp32 executions in onnxruntime.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import certify as C

pytest.importorskip("z3", reason="needs the 'verify' extra (z3-solver)")
ort = pytest.importorskip("onnxruntime")

_BOX = {"x": (-1.0, 1.0)}


def _model(body, initializer=None, opset=13):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _f32(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _conv_bn_relu(seed=0, size=6, k=4):
    rng = np.random.default_rng(seed)
    return _model(
        f"""
        m (float[1,3,{size},{size}] x) => (float[1,{k},{size - 2},{size - 2}] y) {{
          c = Conv(x, W, B)
          b = BatchNormalization<epsilon=1e-5>(c, g, be, mu, var)
          y = Relu(b)
        }}""",
        dict(
            W=_f32(rng, k, 3, 3, 3),
            B=_f32(rng, k),
            g=rng.uniform(0.5, 1.5, k).astype(np.float32),
            be=_f32(rng, k),
            mu=_f32(rng, k),
            var=rng.uniform(0.5, 2, k).astype(np.float32),
        ),
    )


def _two_conv_bn_relu(seed=0, size=8, ch=4):
    rng = np.random.default_rng(seed)
    inits = {}
    for i, cin in enumerate((3, ch)):
        inits[f"W{i}"] = _f32(rng, ch, cin, 3, 3)
        inits[f"B{i}"] = _f32(rng, ch)
        inits[f"g{i}"] = rng.uniform(0.5, 1.5, ch).astype(np.float32)
        inits[f"be{i}"] = _f32(rng, ch)
        inits[f"mu{i}"] = _f32(rng, ch)
        inits[f"var{i}"] = rng.uniform(0.5, 2, ch).astype(np.float32)
    o = size - 4
    return _model(
        f"""
        m (float[1,3,{size},{size}] x) => (float[1,{ch},{o},{o}] y) {{
          c0 = Conv(x, W0, B0)
          b0 = BatchNormalization<epsilon=1e-5>(c0, g0, be0, mu0, var0)
          r0 = Relu(b0)
          c1 = Conv(r0, W1, B1)
          b1 = BatchNormalization<epsilon=1e-5>(c1, g1, be1, mu1, var1)
          y = Relu(b1)
        }}""",
        inits,
    )


def _pair(seed=0, size=6):
    model = _conv_bn_relu(seed, size)
    simplified, _ = onnxsim.simplify(model, certify=False)
    return model, simplified


def _observed_difference(a, b, shape, n=60, seed=1):
    rng = np.random.default_rng(seed)
    # optimisations off: onnxruntime otherwise fuses Conv+BN in the ORIGINAL model too, both
    # sessions run the same kernel and the difference is exactly 0 (a vacuous check)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sa = ort.InferenceSession(
        a.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    sb = ort.InferenceSession(
        b.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    for _ in range(n):
        x = rng.uniform(-1, 1, shape).astype(np.float32)
        worst = max(
            worst,
            float(np.abs(sa.run(None, {"x": x})[0] - sb.run(None, {"x": x})[0]).max()),
        )
    return worst


def test_default_mode_is_unchanged():
    model, simplified = _pair()
    report = C.certify(model, simplified, input_ranges=_BOX)
    assert report.tolerance is None and report.real_outputs is None
    assert report.ok and all(v != C.PROVED_FP32 for v in report.outputs.values())
    explicit = C.certify(model, simplified, input_ranges=_BOX, atol=1e-5, rtol=1e-4)
    assert explicit.outputs == report.outputs


def test_certified_mode_reports_both_the_real_proof_and_the_fp32_tolerance():
    model, simplified = _pair()
    report = C.certify(model, simplified, input_ranges=_BOX, atol="certified")
    assert report.ok and report.outputs == {"y": C.PROVED_FP32}
    assert report.real_outputs is not None and report.real_outputs["y"].startswith(
        "proved"
    )
    tol = report.tolerance
    assert tol is not None and tol.precision == "fp32"
    x = tol.atol["y"]
    assert np.isfinite(x) and x >= tol.real_bound["y"] >= 0.0
    assert x == pytest.approx(
        tol.real_bound["y"] + tol.roundoff_orig["y"] + tol.roundoff_simplified["y"]
    )
    detail = next(w.detail for w in report.windows if w.status == C.PROVED_FP32)
    assert "proved equal in real arithmetic" in detail
    assert "certified fp32 tolerance" in detail
    assert report.within(x) and not report.within(x / 10)


def test_certified_tolerance_is_above_the_default_threshold_for_a_bn_fold():
    # the point of the exercise: "proved at 1e-5 in real arithmetic" does not mean the fp32
    # executions agree to 1e-5
    model, simplified = _pair()
    report = C.certify(model, simplified, input_ranges=_BOX, atol="certified")
    assert report.tolerance.atol["y"] > 1e-5


def test_certified_tolerance_covers_the_executed_difference():
    model, simplified = _pair(seed=3)
    report = C.certify(model, simplified, input_ranges=_BOX, atol="certified")
    observed = _observed_difference(model, simplified, (1, 3, 6, 6))
    assert observed > 0.0  # not vacuous: the fold really changes the fp32 result
    assert observed <= report.tolerance.atol["y"]


def test_never_looser_than_fp_error_tolerance_for():
    from onnxsim import fp_error

    model, simplified = _pair(seed=2)
    report = C.certify(model, simplified, input_ranges=_BOX, atol="certified")
    reference = fp_error.tolerance_for(model, simplified, _BOX)
    assert report.tolerance.atol["y"] <= reference.atol["y"] * (1 + 1e-12)


def test_unbounded_inputs_never_prove_a_tolerance():
    model, simplified = _pair()
    folded = C.certify(model, simplified, atol="certified")  # no input_ranges
    assert not folded.ok and C.PROVED_FP32 not in folded.outputs.values()
    # an exactly identical pair IS proved in real arithmetic with no range, but no finite fp32
    # tolerance exists without one: skipped with the reason, never proved
    same = C.certify(model, model, atol="certified")
    assert same.real_outputs == {"y": C.PROVED_STRUCTURAL}
    assert same.outputs == {"y": C.SKIPPED} and not same.ok
    assert "has no range" in " ".join(w.detail for w in same.windows)
    assert same.tolerance is None


@pytest.mark.parametrize("op", ["Sin", "Floor"])
def test_op_without_a_roundoff_rule_is_skipped_with_the_reason(op):
    model = _model(f"m (float[4] x) => (float[4] y) {{ y = {op}(x) }}")
    report = C.certify(model, model, input_ranges=_BOX, atol="certified")
    assert report.real_outputs == {"y": C.PROVED_STRUCTURAL}
    assert report.outputs == {"y": C.SKIPPED}
    detail = " ".join(w.detail for w in report.windows)
    assert "no certified fp32 tolerance" in detail and op in detail


def test_identical_models_get_only_the_roundoff_terms():
    model = _conv_bn_relu()
    report = C.certify(model, model, input_ranges=_BOX, atol="certified")
    tol = report.tolerance
    assert report.outputs == {"y": C.PROVED_FP32}
    assert tol.real_bound["y"] == 0.0  # a structural proof gives t = 0
    assert tol.atol["y"] == pytest.approx(
        tol.roundoff_orig["y"] + tol.roundoff_simplified["y"]
    )


def test_a_wrong_rewrite_is_never_proved_in_certified_mode():
    model, simplified = _pair()
    bad = onnx.ModelProto()
    bad.CopyFrom(simplified)
    for t in bad.graph.initializer:
        if t.dims and t.data_type == onnx.TensorProto.FLOAT and t.dims == [4]:
            arr = numpy_helper.to_array(t).copy()
            arr += 0.25
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
            break
    report = C.certify(model, bad, input_ranges=_BOX, atol="certified")
    assert not report.ok and C.PROVED_FP32 not in report.outputs.values()
    assert report.real_outputs["y"] in (C.REFUTED, C.SKIPPED)


def test_congruence_is_not_used_to_build_the_real_bound():
    # a congruence proof is not a tolerance-level statement about the outputs (an op with gain
    # above 1 amplifies an input gap), so the real bound must come from the zonotope alone;
    # a structural proof, which is exact, contributes 0
    model, simplified = _pair()
    # thresholds of 0 make the proof-derived bound (atol + rtol * max|y|) exactly 0, i.e.
    # smaller than any zonotope bound, so a status that is allowed to use it shows up as 0
    kw = dict(input_ranges=_BOX, atol=0.0, rtol=0.0, precision="fp32", deadline=None)

    def real_bound(status, **override):
        tol, why = C._certified_tolerance(
            model, simplified, real_status={"y": status}, **{**kw, **override}
        )
        assert tol is not None, why
        return tol.real_bound["y"]

    assert real_bound(C.PROVED_STRUCTURAL) == 0.0  # exact proof: t = 0
    # ... whatever the threshold: a structural proof is not "within the threshold", it is exact
    assert real_bound(C.PROVED_STRUCTURAL, atol=1e-5, rtol=1e-4) == 0.0
    assert real_bound(C.PROVED_SMT) == 0.0  # direct comparison of the output pair
    assert real_bound(C.PROVED_REDUCED) == 0.0
    zonotope_only = real_bound(C.PROVED_CONGRUENCE)
    assert 0.0 < zonotope_only < np.inf  # the zonotope bound, not the proof's tolerance
    assert (
        real_bound(C.PROVED_AFFINE) == zonotope_only
    )  # that proof IS the zonotope bound


def test_a_skipped_output_still_reports_its_tolerance_for_information_only():
    # two folded layers: the zonotope's real-arithmetic bound is a few 1e-5, above the default
    # threshold, so the real-arithmetic proof is not available and the verdict stays skipped.
    # The certified tolerance is still finite and is reported -- as information, not as a proof:
    # a bound X is not an equality (a wrong rewrite has a finite X too).
    model = _two_conv_bn_relu()
    simplified, _ = onnxsim.simplify(model, certify=False)
    report = C.certify(
        model,
        simplified,
        input_ranges=_BOX,
        atol="certified",
        rtol=0.0,
        max_work=10,  # keep the SMT step out of it: this is about the verdict wording
    )
    assert report.real_outputs == {"y": C.SKIPPED}
    assert report.outputs == {"y": C.SKIPPED} and not report.ok
    assert report.tolerance is not None and np.isfinite(report.tolerance.atol["y"])
    assert not report.within(1.0)  # never "within": the output is not proved
    info = [w.detail for w in report.windows if "for information" in w.detail]
    assert info and "at most" in info[0]
    observed = _observed_difference(model, simplified, (1, 3, 8, 8))
    assert observed <= report.tolerance.atol["y"]


def test_fp16_tolerance_is_larger_than_fp32():
    model, simplified = _pair()
    x32 = C.certify(model, simplified, input_ranges=_BOX, atol="certified").tolerance
    x16 = C.certify(
        model, simplified, input_ranges=_BOX, atol="certified", precision="fp16"
    ).tolerance
    assert x16.precision == "fp16" and x16.atol["y"] > x32.atol["y"]


def test_exhausted_time_budget_skips_instead_of_proving():
    model = _conv_bn_relu()
    report = C.certify(
        model, model, input_ranges=_BOX, atol="certified", total_timeout_ms=0
    )
    assert report.real_outputs == {"y": C.PROVED_STRUCTURAL}
    assert report.outputs == {"y": C.SKIPPED}
    assert "time budget" in " ".join(w.detail for w in report.windows)


def test_an_unknown_atol_string_raises():
    model = _conv_bn_relu()
    with pytest.raises(ValueError, match="certified"):
        C.certify(model, model, atol="tight")


def test_report_text_mentions_the_certified_tolerance():
    model, simplified = _pair()
    text = str(C.certify(model, simplified, input_ranges=_BOX, atol="certified"))
    assert "proved-fp32" in text and "certified fp32 tolerance" in text
