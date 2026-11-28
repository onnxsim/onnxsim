"""Signed activation dtypes for ``quantize_static(full_graph=True)``.

The whole-graph QDQ scheme used to quantize activations only onto the unsigned
grid (uint8 / uint16). ``activation_type="int8"`` and ``"int16"`` now fit the
same asymmetric scale and zero point onto the full signed grid, so a zero
point can be negative and the calibrated extremes land on the grid's ends.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim.qdq_full_graph import _activation_qparams


def _model(body, initializer=(), opset=17, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _conv_block(opset=17):
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(
        rng.standard_normal((4, 4, 3, 3)).astype(np.float32) * 0.3, "w"
    )
    b = numpy_helper.from_array(rng.standard_normal(4).astype(np.float32) * 0.1, "b")
    return _model(
        "g (float[1, 4, 8, 8] x) => (float[1, 4, 8, 8] y) {"
        "  c = Conv<pads = [1, 1, 1, 1]>(x, w, b)"
        "  r = Relu(c)"
        "  y = Add(r, x)"
        "}",
        initializer=[w, b],
        opset=opset,
    )


def _calib(n=4, seed=1):
    rng = np.random.default_rng(seed)
    return [
        {"x": rng.standard_normal((1, 4, 8, 8)).astype(np.float32)} for _ in range(n)
    ]


def _quantize_zero_point_dtypes(model):
    """Dtypes of every QuantizeLinear zero point (the activation Q nodes)."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    return {
        inits[n.input[2]].dtype
        for n in model.graph.node
        if n.op_type == "QuantizeLinear"
    }


def _run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)[0]


def _cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize(
    "lo, hi, qmin, qmax",
    [(-1.0, 3.0, 0, 255), (-1.0, 3.0, -128, 127), (-0.5, 2.0, -32768, 32767)],
)
def test_calibrated_extremes_land_on_grid_ends(lo, hi, qmin, qmax):
    scale, zp = _activation_qparams(lo, hi, qmin, qmax)
    assert qmin <= zp <= qmax
    # lo and hi quantize to (within one step of) the grid's first and last code
    assert abs(round(lo / scale) + zp - qmin) <= 1
    assert abs(round(hi / scale) + zp - qmax) <= 1


def test_unsigned_qparams_match_the_original_formula():
    # qmin=0 is the old unsigned formula: scale = (hi - lo) / qmax, zp = round(-lo / scale)
    scale, zp = _activation_qparams(-1.0, 3.0, 0, 255)
    assert scale == pytest.approx(4.0 / 255)
    assert zp == round(1.0 / (4.0 / 255))


def test_positive_range_gets_the_lowest_signed_zero_point():
    # [0.5, 4] widens to [0, 4]: 0 sits at the bottom of the range, so with a
    # signed grid the zero point is qmin itself
    _, zp = _activation_qparams(0.5, 4.0, -128, 127)
    assert zp == -128


def test_int8_activations_quantize_with_int8_zero_points():
    model = _conv_block()
    x = _calib()[0]["x"]
    ref = _run(model, {"x": x})
    q = onnxsim.quantize_static(
        model, _calib(), full_graph=True, activation_type="int8"
    )
    onnx.checker.check_model(q)
    # QuantizeLinear nodes are the activations only (weights are pre-quantized)
    assert _quantize_zero_point_dtypes(q) == {np.dtype(np.int8)}
    assert _cos(_run(q, {"x": x}), ref) > 0.99


def test_uint8_default_is_unchanged():
    q = onnxsim.quantize_static(_conv_block(), _calib(), full_graph=True)
    assert _quantize_zero_point_dtypes(q) == {np.dtype(np.uint8)}


def test_int16_needs_opset_21():
    with pytest.raises(ValueError, match="needs opset >= 21"):
        onnxsim.quantize_static(
            _conv_block(opset=17),
            _calib(),
            full_graph=True,
            activation_type="int16",
        )


def test_unknown_activation_type_is_rejected():
    with pytest.raises(ValueError, match="activation_type must be one of"):
        onnxsim.quantize_static(
            _conv_block(), _calib(), full_graph=True, activation_type="int4"
        )


def test_int16_activations_at_opset_21():
    model = _conv_block(opset=21)
    x = _calib()[0]["x"]
    ref = _run(model, {"x": x})
    q = onnxsim.quantize_static(
        model, _calib(), full_graph=True, activation_type="int16"
    )
    onnx.checker.check_model(q)
    assert _quantize_zero_point_dtypes(q) == {np.dtype(np.int16)}
    assert _cos(_run(q, {"x": x}), ref) > 0.999
