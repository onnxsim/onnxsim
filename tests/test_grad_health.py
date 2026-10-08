import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim.grad_health import check_gradient_health


def _model(body: str, opset: int = 15) -> onnx.ModelProto:
    return parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')


def _kinds(rep):
    return {(f.tensor, f.kind) for f in rep.findings}


def test_tiny_upstream_gradient_is_flushed_in_fp16_but_not_fp32():
    body = """
agraph (float[4] dY, float[4] W) => (float[4] dX) {
    dX = Mul(dY, W)
}
"""
    ranges = {"dY": (1e-9, 1e-8), "W": (0.5, 1.0)}
    rep16 = check_gradient_health(_model(body), ranges, precision="fp16")
    assert ("dX", "flushed") in _kinds(rep16)
    assert not rep16.vanishing_free
    rep32 = check_gradient_health(_model(body), ranges, precision="fp32")
    assert rep32.vanishing_free


def test_gradient_that_is_identically_zero_is_dead():
    body = """
agraph (float[4] dY, float[4] W) => (float[4] dX) {
    dX = Mul(dY, W)
}
"""
    rep = check_gradient_health(
        _model(body), {"dY": (0.0, 0.0), "W": (0.5, 1.0)}, precision="fp32"
    )
    assert ("dX", "dead") in _kinds(rep)


def test_cancellation_loses_precision_in_fp16_only():
    body = """
agraph (float[4] A, float[4] B) => (float[4] G) {
    G = Sub(A, B)
}
"""
    ranges = {"A": (1000.0, 1000.01), "B": (999.99, 1000.0)}
    rep16 = check_gradient_health(_model(body), ranges, precision="fp16")
    assert ("G", "imprecise") in _kinds(rep16)
    assert not rep16.precise
    rep32 = check_gradient_health(_model(body), ranges, precision="fp32")
    assert rep32.precise


def test_well_scaled_gradient_is_healthy():
    body = """
agraph (float[4] dY, float[4] W) => (float[4] dX) {
    dX = Mul(dY, W)
}
"""
    rep = check_gradient_health(
        _model(body), {"dY": (0.5, 1.0), "W": (0.5, 1.0)}, precision="fp16"
    )
    assert rep.healthy and rep.findings == []


def test_unknown_precision_is_rejected():
    body = """
agraph (float[4] X) => (float[4] Y) {
    Y = Relu(X)
}
"""
    with pytest.raises(ValueError):
        check_gradient_health(_model(body), {"X": (0.0, 1.0)}, precision="fp8")


def test_magnitude_bound_contains_onnxruntime_gradients():
    ort = pytest.importorskip("onnxruntime")
    body = """
agraph (float[4] dY, float[4] W) => (float[4] dX) {
    dX = Mul(dY, W)
}
"""
    model = _model(body)
    ranges = {"dY": (-1e-3, 1e-3), "W": (-2.0, 2.0)}
    from onnxsim import interval

    res = interval.propagate(model, ranges)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(3)
    lo, hi = res.intervals["dX"]
    for _ in range(200):
        feed = {
            k: rng.uniform(a, b, size=4).astype(np.float32)
            for k, (a, b) in ranges.items()
        }
        out = sess.run(None, feed)[0].astype(np.float64)
        assert np.all(out >= lo - 1e-12) and np.all(out <= hi + 1e-12)
