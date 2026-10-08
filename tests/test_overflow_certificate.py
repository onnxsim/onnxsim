"""onnxsim.overflow_certificate: no reachable value leaves a signed word or a field window."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import ranges
from onnxsim.overflow_certificate import certify_no_overflow


def _model(body, initializer=None):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["" : 17]> {body}')
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _accumulators(rep):
    return [f for f in rep.findings if f.kind == "accumulator"]


def test_small_integer_matmul_fits_int32_with_its_bound():
    model = _model(
        "agraph (int8[1, 4] x) => (int32[1, 2] y) { y = MatMulInteger(x, W) }",
        {"W": np.full((4, 2), 127, np.int8)},
    )
    rep = certify_no_overflow(model, bits=32)
    assert rep.fits
    (acc,) = _accumulators(rep)
    # sum_k |w| * max|x| = 4 * 127 * 128
    assert acc.hi == 4 * 127 * 128
    assert acc.required_bits == 17


def test_accumulator_overflows_int32_at_its_bound():
    k = 2**18
    model = _model(
        f"agraph (int8[1, {k}] x) => (int32[1, 1] y) {{ y = MatMulInteger(x, W) }}",
        {"W": np.full((k, 1), 127, np.int8)},
    )
    rep = certify_no_overflow(model, bits=32)
    assert not rep.fits
    (violation,) = rep.violations
    assert violation.kind == "accumulator"
    assert violation.hi == k * 127 * 128
    assert violation.required_bits == 33
    assert certify_no_overflow(model, bits=33).fits


def test_unbounded_float_input_is_never_certified():
    model = _model("agraph (float[4] x) => (int8[4] y) { y = Cast<to = 3>(x) }")
    rep = certify_no_overflow(model, bits=32)
    assert not rep.fits
    assert rep.unanalysed == ["Cast_0: unbounded x"]


@pytest.mark.parametrize(
    "lo, hi, fits, kind", [(-100, 100, True, None), (-200, 200, False, "cast")]
)
def test_cast_to_int8_is_checked_against_the_dtype_range(lo, hi, fits, kind):
    model = _model("agraph (float[4] x) => (int8[4] y) { y = Cast<to = 3>(x) }")
    ranges.set_range(model, "x", lo, hi)
    rep = certify_no_overflow(model, bits=32)
    assert rep.fits is fits
    assert [f.kind for f in rep.violations] == ([] if kind is None else [kind])


def test_field_window_rejects_values_above_half_the_modulus():
    model = _model("agraph (float[4] x) => (float[4] y) { y = Mul(x, x) }")
    ranges.set_range(model, "x", 0.0, 2.0)
    # window [-3, 3] for p = 7: y in [0, 4] escapes it
    assert not certify_no_overflow(model, field_modulus=7).fits
    # window [-5, 5] for p = 11
    assert certify_no_overflow(model, field_modulus=11).fits


def test_integer_input_defaults_to_its_dtype_range():
    model = _model(
        "agraph (int8[4] x) => (int8[4] y) { y = Add(x, C) }",
        {"C": np.ones(4, np.int8)},
    )
    # x in [-128, 127] so x + 1 reaches 128, which int8 cannot hold
    assert not certify_no_overflow(model, bits=8).fits
    assert certify_no_overflow(model, bits=9).fits


def test_window_arguments_are_validated():
    model = _model("agraph (int8[4] x) => (int8[4] y) { y = Identity(x) }")
    with pytest.raises(ValueError):
        certify_no_overflow(model)
    with pytest.raises(ValueError):
        certify_no_overflow(model, bits=8, field_modulus=7)
    with pytest.raises(ValueError):
        certify_no_overflow(model, field_modulus=8)


def test_accumulator_bound_covers_onnxruntime_outputs():
    rng = np.random.default_rng(0)
    W = rng.integers(-128, 128, size=(8, 3), dtype=np.int8)
    model = _model(
        "agraph (int8[2, 8] x) => (int32[2, 3] y) { y = MatMulInteger(x, W) }",
        {"W": W},
    )
    rep = certify_no_overflow(model, bits=32)
    assert rep.fits
    (acc,) = _accumulators(rep)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for _ in range(200):
        x = rng.integers(-128, 128, size=(2, 8), dtype=np.int8)
        y = sess.run(None, {"x": x})[0]
        assert acc.lo <= y.min() and y.max() <= acc.hi
