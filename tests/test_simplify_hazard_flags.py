"""``simplify(check_hazards=..., check_gradients=...)``: opt-in NaN/Inf and gradient-health
checks of the simplified model, with verdicts recorded in ``metadata_props``."""

import onnx
import pytest
from onnx import parser

import onnxsim
from onnxsim import nan_check, ranges

_HEADER = '<ir_version: 8, opset_import: ["": 17]>'


def _model(body: str) -> onnx.ModelProto:
    return parser.parse_model(f"{_HEADER}\n{body}")


def _meta(model, key):
    return {p.key: p.value for p in model.metadata_props}.get(key)


def _log_model(lo, hi):
    model = _model(
        """
        agraph (float[4] x) => (float[4] y)
        {
            y = Log(x)
        }
        """
    )
    ranges.set_range(model, "x", lo, hi)
    return model


def test_nan_hazard_is_recorded_when_opted_in():
    sim, _ = onnxsim.simplify(_log_model(-1.0, 1.0), certify=False, check_hazards=True)
    assert _meta(sim, "onnxsim.nan_check").startswith("nan reachable")


def test_positive_domain_is_recorded_as_finite():
    sim, _ = onnxsim.simplify(_log_model(0.5, 2.0), certify=False, check_hazards=True)
    assert _meta(sim, "onnxsim.nan_check").startswith("finite")


def test_checks_are_off_by_default():
    sim, _ = onnxsim.simplify(_log_model(-1.0, 1.0), certify=False)
    assert _meta(sim, "onnxsim.nan_check") is None
    assert _meta(sim, "onnxsim.gradient_health") is None


def test_gradient_flush_is_recorded_per_precision():
    body = """
    agraph (float[4] dY, float[4] W) => (float[4] dX) {
        dX = Mul(dY, W)
    }
    """

    def run(precision):
        model = _model(body)
        ranges.set_range(model, "dY", 1e-9, 1e-8)
        ranges.set_range(model, "W", 0.5, 1.0)
        sim, _ = onnxsim.simplify(
            model,
            certify=False,
            check_gradients=True,
            gradient_precision=precision,
        )
        return _meta(sim, "onnxsim.gradient_health")

    assert run("fp16").startswith("unhealthy")
    assert "flushed" in run("fp16")
    assert run("fp32").startswith("healthy")


def test_checker_failure_is_reported_and_does_not_fail_simplify(monkeypatch):
    def boom(model, input_ranges=None):
        raise RuntimeError("synthetic checker failure")

    monkeypatch.setattr(nan_check, "check_nan", boom)
    sim, _ = onnxsim.simplify(_log_model(-1.0, 1.0), certify=False, check_hazards=True)
    assert _meta(sim, "onnxsim.nan_check").startswith("error:")
    assert "synthetic checker failure" in _meta(sim, "onnxsim.nan_check")


def test_gradient_precision_must_be_known():
    with pytest.raises(ValueError, match="fp8"):
        onnxsim.simplify(
            _log_model(0.5, 2.0),
            certify=False,
            check_gradients=True,
            gradient_precision="fp8",
        )
