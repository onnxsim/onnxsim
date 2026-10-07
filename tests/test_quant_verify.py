"""Tests for onnxsim.quant_verify: certified bounds on |quantized - float|.

The model of a quantizer is checked against hand-computable cases (one Q->DQ site, one saturating
site), then the verifier is run on REAL quantized models produced by onnxsim's own quantizers
(static / static_int16 / qoperator / weight_only int8+int4 / dynamic) and the certified bound is
required to dominate the error observed in onnxruntime, including a corner-search adversary. A
certified bound below an observed error is a soundness bug, so those tests are the point.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

pytest.importorskip("onnxruntime")

from onnxsim import accuracy, quant_verify  # noqa: E402
from onnxsim import ranges as R  # noqa: E402


def _model(body, initializer=None, opset=21, ir_version=9, extra_opsets=""):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}{extra_opsets}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _f32(rng, *shape, scale=0.5):
    return rng.standard_normal(shape).astype(np.float32) * scale


def _cnn2():
    rng = np.random.default_rng(1)
    return _model(
        "m (float[1,3,8,8] x) => (float[1,4,4,4] y) "
        "{ c = Conv(x, W1, B1) r = Relu(c) y = Conv(r, W2, B2) }",
        dict(
            W1=_f32(rng, 6, 3, 3, 3),
            B1=_f32(rng, 6),
            W2=_f32(rng, 4, 6, 3, 3),
            B2=_f32(rng, 4),
        ),
    )


def _mlp3():
    rng = np.random.default_rng(1)
    return _model(
        "m (float[2,32] x) => (float[2,8] y) "
        "{ a = MatMul(x, W1) b = Add(a, B1) r = Relu(b) c = MatMul(r, W2) d = Relu(c) y = MatMul(d, W3) }",
        dict(
            W1=_f32(rng, 32, 64),
            B1=_f32(rng, 64),
            W2=_f32(rng, 64, 64),
            W3=_f32(rng, 64, 8),
        ),
    )


def _residual():
    rng = np.random.default_rng(1)
    return _model(
        "m (float[1,4,6,6] x) => (float[1,4,6,6] y) "
        "{ c = Conv<pads=[1,1,1,1]>(x, W1, B1) r = Relu(c) d = Conv<pads=[1,1,1,1]>(r, W2, B2) "
        "a = Add(d, x) y = Relu(a) }",
        dict(
            W1=_f32(rng, 4, 4, 3, 3),
            B1=_f32(rng, 4),
            W2=_f32(rng, 4, 4, 3, 3),
            B2=_f32(rng, 4),
        ),
    )


_MODELS = {"cnn2": _cnn2, "mlp3": _mlp3, "residual": _residual}


def _quantize(model, scheme, dtype):
    cfg = accuracy.QuantizationConfig(
        scheme=scheme, dtype=dtype, num_calibration_samples=16
    )
    return accuracy.quantize(model, cfg)


def _within(observed, report, fp32_slack=1e-4):
    """The certified bound is in real arithmetic; float32 execution adds a little roundoff."""
    return observed <= report.worst + fp32_slack * (1.0 + report.worst)


# ---- the quantizer model, on hand-computable cases ---------------------------------------------


def _qdq(scale, zp_dtype=np.int8, zp=0):
    return _model(
        "m (float[1,64] x) => (float[1,64] y) { q = QuantizeLinear(x, s, z)  y = DequantizeLinear(q, s, z) }",
        dict(s=np.array(scale, np.float32), z=np.array(zp, zp_dtype)),
    )


def _identity():
    return _model("m (float[1,64] x) => (float[1,64] y) { y = Identity(x) }")


def test_round_trip_site_is_exactly_half_a_step_inside_the_range():
    scale = 0.05
    q = _qdq(scale)  # int8, zero point 0: representable range is [-6.4, 6.35]
    box = {"x": (-3.0, 3.0)}
    rep = quant_verify.verify(_identity(), q, box)
    assert rep.worst == pytest.approx(scale / 2, rel=1e-6)
    assert [s.kind for s in rep.sites] == ["QuantizeLinear"] and not rep.sites[
        0
    ].clipped
    # random inputs attain almost exactly that error, so the bound is tight, not just sound
    observed = quant_verify.observed_error(_identity(), q, {"x": (-3.0, 3.0)}, n=300)
    assert 0.9 * (scale / 2) <= observed <= scale / 2 + 1e-6


def test_saturation_is_charged_not_assumed_away():
    scale = 0.05
    q = _qdq(scale)
    rmax = 127 * scale
    rep = quant_verify.verify(_identity(), q, {"x": (-10.0, 10.0)})
    # error at x = 10 is 10 - rmax; above the range the error is one-sided, below likewise
    assert rep.worst == pytest.approx(10.0 - rmax, rel=1e-6)
    assert rep.sites[0].clipped and any(
        "exceeds the representable range" in h for h in rep.hazards
    )
    observed = quant_verify.observed_error(
        _identity(), q, {"x": (-10.0, 10.0)}, n=200, adversarial=50
    )
    assert observed <= rep.worst + 1e-6 and observed > 0.9 * (10.0 - rmax)


def test_asymmetric_uint8_zero_point_range_is_used():
    scale, zp = 0.02, 100  # representable [-2.0, 3.1]
    q = _qdq(scale, np.uint8, zp)
    inside = quant_verify.verify(_identity(), q, {"x": (-1.0, 3.0)})
    assert (
        inside.worst == pytest.approx(scale / 2, rel=1e-6)
        and not inside.sites[0].clipped
    )
    below = quant_verify.verify(_identity(), q, {"x": (-4.0, 3.0)})
    assert below.worst == pytest.approx(-2.0 - (-4.0), rel=1e-6)  # Rmin - lo


def test_noise_and_clamp_clipping_models_are_both_sound():
    model, q = _mlp3(), _quantize(_mlp3(), "qoperator", "int8")
    box = {"x": (-1.0, 1.0)}
    observed = quant_verify.observed_error(model, q, box, n=60, adversarial=150)
    for clipping in ("noise", "clamp"):
        rep = quant_verify.verify(model, q, box, clipping=clipping, breakdown=False)
        assert _within(observed, rep), clipping


# ---- real quantizers: the certified bound must dominate what onnxruntime does ------------------

_SCHEMES = [
    ("static", "int8"),
    ("static_int16", "int16"),
    ("qoperator", "int8"),
    ("weight_only", "int8"),
    ("weight_only", "int4"),
    ("dynamic", "int8"),
]


@pytest.mark.parametrize("box", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("scheme,dtype", _SCHEMES)
@pytest.mark.parametrize("name", sorted(_MODELS))
def test_certified_bound_dominates_observed_error(name, scheme, dtype, box):
    model = _MODELS[name]()
    q = _quantize(model, scheme, dtype)
    if q.SerializeToString() == model.SerializeToString():
        pytest.skip("quantizer left this model unchanged")
    ranges = {"x": (-box, box)}
    rep = quant_verify.verify(model, q, ranges, breakdown=False)
    assert rep.bounded, rep.hazards
    observed = quant_verify.observed_error(model, q, ranges, n=60, adversarial=150)
    assert _within(observed, rep), (observed, rep.worst)


def test_bound_grows_when_the_box_exceeds_the_calibrated_range():
    model, q = _mlp3(), _quantize(_mlp3(), "static", "int8")
    narrow = quant_verify.verify(model, q, {"x": (-0.25, 0.25)}, breakdown=False)
    wide = quant_verify.verify(model, q, {"x": (-4.0, 4.0)}, breakdown=False)
    assert not any(s.clipped for s in narrow.sites) and any(
        s.clipped for s in wide.sites
    )
    assert wide.worst > 5 * narrow.worst
    assert any("exceeds the representable range" in h for h in wide.hazards)


def test_weight_only_models_have_no_activation_sites_and_a_weights_floor():
    model = _mlp3()
    q8, q4 = (
        _quantize(model, "weight_only", "int8"),
        _quantize(model, "weight_only", "int4"),
    )
    r8 = quant_verify.verify(model, q8, {"x": (-1.0, 1.0)})
    r4 = quant_verify.verify(model, q4, {"x": (-1.0, 1.0)})
    assert r8.sites == [] and r4.sites == []
    assert (
        r4.worst > 5 * r8.worst
    )  # int4 blockwise is far coarser than per-channel int8


def test_per_site_breakdown_is_consistent():
    model, q = _mlp3(), _quantize(_mlp3(), "qoperator", "int8")
    rep = quant_verify.verify(model, q, {"x": (-0.25, 0.25)})
    assert len(rep.sites) == 2 and all(s.contribution is not None for s in rep.sites)
    assert all(s.contribution <= rep.worst * (1 + 1e-9) for s in rep.sites)
    assert rep.weights_only is not None and rep.weights_only <= rep.worst * (1 + 1e-9)
    assert "quantization verify" in str(rep)


def test_dynamic_quantization_is_handled_by_bounding_the_scale():
    model, q = _mlp3(), _quantize(_mlp3(), "dynamic", "int8")
    rep = quant_verify.verify(model, q, {"x": (-1.0, 1.0)}, breakdown=False)
    assert [s.kind for s in rep.sites] == ["DynamicQuantizeLinear"]
    # scale <= (max(hi, 0) - min(lo, 0)) / 255 = 2 / 255, and |error| <= scale / 2
    assert rep.sites[0].radius == pytest.approx(2.0 / 255 / 2, rel=1e-3)
    fused = _quantize(model, "dynamic_fused", "int8")
    rep2 = quant_verify.verify(model, fused, {"x": (-1.0, 1.0)}, breakdown=False)
    assert rep2.worst == pytest.approx(rep.worst, rel=1e-6)


def test_within_checks_atol_and_rtol_soundly():
    model, q = _mlp3(), _quantize(_mlp3(), "weight_only", "int8")
    rep = quant_verify.verify(model, q, {"x": (-1.0, 1.0)})
    assert rep.within(rep.worst * 1.01) and not rep.within(rep.worst * 0.5)


# ---- refusals and hazards: never a guessed finite bound ----------------------------------------


def test_unsupported_quantized_op_gives_an_infinite_bound_with_a_reason():
    rng = np.random.default_rng(0)
    float_model = _model(
        "m (float[1,4] x) => (float[1,4] y) { y = Add(x, B) }", dict(B=_f32(rng, 4))
    )
    q = _model(
        "m (float[1,4] x) => (float[1,4] y) { y = com.microsoft.QLinearAdd(x, x, x, x, x, x, x, x) }",
        extra_opsets=', "com.microsoft" : 1',
    )
    rep = quant_verify.verify(float_model, q, {"x": (-1.0, 1.0)})
    assert not rep.bounded and not rep.within(1e9)
    assert any("QLinearAdd" in h for h in rep.hazards)


def test_leaked_integer_accumulator_is_refused():
    # DynamicQuantizeLinear -> MatMulInteger whose accumulator is consumed by something other than
    # the recognised Cast -> Mul(scale) pattern: unsound to rewrite, so it must be refused.
    rng = np.random.default_rng(0)
    float_model = _model(
        "m (float[1,4] x) => (float[1,3] y) { y = MatMul(x, W) }",
        dict(W=_f32(rng, 4, 3)),
    )
    q = _model(
        "m (float[1,4] x) => (float[1,3] y) "
        "{ xq, xs, xz = DynamicQuantizeLinear(x)  acc = MatMulInteger(xq, Wq, xz)  accf = Cast<to=1>(acc) "
        "y = Add(accf, accf) }",
        dict(Wq=rng.integers(-100, 100, (4, 3)).astype(np.int8)),
    )
    rep = quant_verify.verify(float_model, q, {"x": (-1.0, 1.0)})
    assert not rep.bounded and any("integer-domain" in h for h in rep.hazards)


def test_missing_or_unbounded_input_range_is_reported_not_guessed():
    model, q = _mlp3(), _quantize(_mlp3(), "static", "int8")
    rep = quant_verify.verify(model, q)  # no input ranges at all
    assert not rep.bounded and any("no range" in h for h in rep.hazards)


def test_mismatched_graph_inputs_raise():
    model = _mlp3()
    other = _model(
        "m (float[2,32] z) => (float[2,8] y) { y = MatMul(z, W) }",
        dict(W=np.zeros((32, 8), np.float32)),
    )
    with pytest.raises(ValueError, match="inputs differ"):
        quant_verify.verify(model, other, {"x": (-1.0, 1.0)})


def test_accumulator_overflow_is_flagged_as_a_hazard():
    # K = 70000, uint8 activations (|q - zp| <= 255) and int8 weights of 127: 70000 * 255 * 127 = 2.27e9
    # exceeds 2**31 - 1 = 2.15e9 (with weights of 100 it would not: 1.79e9)
    k = 70000
    q = _model(
        "m (float[1,70000] x) => (float[1,1] y) "
        "{ xq = QuantizeLinear(x, xs, xz)  yq = QLinearMatMul(xq, xs, xz, Wq, ws, wz, ys, yz)  y = DequantizeLinear(yq, ys, yz) }",
        dict(
            xs=np.array(0.01, np.float32),
            xz=np.array(0, np.uint8),
            Wq=np.full((k, 1), 127, np.int8),
            ws=np.array(0.01, np.float32),
            wz=np.array(0, np.int8),
            ys=np.array(1.0, np.float32),
            yz=np.array(0, np.uint8),
        ),
    )
    conv = quant_verify._Converter(q)
    conv.run()
    assert any("int32 accumulator" in h for h in conv.hazards)


# ---- output range annotations ------------------------------------------------------------------


def test_annotated_output_ranges_are_checked_on_the_quantized_graph():
    q = _quantize(_mlp3(), "static", "int8")
    loose = onnx.ModelProto()
    loose.CopyFrom(q)
    R.set_range(loose, "y", -1e4, 1e4)
    verdicts = quant_verify.verify_against_annotation(loose, {"x": (-1.0, 1.0)})
    assert verdicts["y"].proved
    tight = onnx.ModelProto()
    tight.CopyFrom(q)
    R.set_range(tight, "y", -0.01, 0.01)
    assert not quant_verify.verify_against_annotation(tight, {"x": (-1.0, 1.0)})[
        "y"
    ].proved


def test_output_rename_by_position():
    # the quantized model may name its output differently; outputs are matched by position
    rng = np.random.default_rng(0)
    w = _f32(rng, 4, 3)
    f = _model("m (float[1,4] x) => (float[1,3] y) { y = MatMul(x, W) }", dict(W=w))
    g = _model("m (float[1,4] x) => (float[1,3] out) { out = MatMul(x, W) }", dict(W=w))
    rep = quant_verify.verify(f, g, {"x": (-1.0, 1.0)})
    assert rep.worst == 0.0 or rep.worst < 1e-6
