"""Tests for onnxsim.quark_dynamic (dynamic-activation integer quantization)."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim.quark_dynamic import quantize_dynamic_integer


def _model(body, inits, shape="[3,8]"):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g (float{shape} x) => (float y) {{ {body} }}'
    )
    m.graph.initializer.extend(
        numpy_helper.from_array(np.asarray(v, np.float32), k) for k, v in inits.items()
    )
    return m


def _run(m, x):
    return ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]


RNG = np.random.default_rng(0)
X = RNG.standard_normal((3, 8)).astype(np.float32)
W = RNG.standard_normal((8, 5)).astype(np.float32)
B = RNG.standard_normal(5).astype(np.float32)


def _ops(m):
    return [n.op_type for n in m.graph.node]


def test_matmul_becomes_matmulinteger_and_stays_close():
    m = _model("y = MatMul(x, w)", {"w": W})
    q = quantize_dynamic_integer(m, weight_dtype="uint8")
    assert _ops(q) == ["DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul", "Mul"]
    ref, got = _run(m, X), _run(q, X)
    assert np.linalg.norm(ref - got) / np.linalg.norm(ref) < 0.03
    assert not any(i.name == "w" for i in q.graph.initializer)  # replaced


def test_int8_weights_are_symmetric_and_match_an_integer_emulation():
    # Not run through ONNX Runtime: its u8 x s8 MatMulInteger kernel saturates
    # on x86 CPUs without VNNI (CI runners), so the result would depend on the
    # host. The integer arithmetic is emulated in numpy instead.
    m = _model("y = MatMul(x, w)", {"w": W})
    q = quantize_dynamic_integer(m, weight_dtype="int8")
    assert _ops(q) == ["DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul", "Mul"]
    inits = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    wq, ws, wz = inits["w_quantized"], inits["w_scale"], inits["w_zero_point"]
    assert wq.dtype == np.int8 and wz == 0 and np.abs(wq).max() == 127
    # DynamicQuantizeLinear: uint8 asymmetric over [min(x, 0), max(x, 0)]
    lo, hi = min(float(X.min()), 0.0), max(float(X.max()), 0.0)
    xs = (hi - lo) / 255.0
    xz = np.clip(np.round(-lo / xs), 0, 255)
    xq = np.clip(np.round(X / xs) + xz, 0, 255)
    got = ((xq - xz) @ wq.astype(np.float64)) * (xs * ws)
    ref = X @ W
    assert np.linalg.norm(ref - got) / np.linalg.norm(ref) < 0.03


@pytest.mark.parametrize("trans_b", [0, 1])
def test_gemm_is_split_into_matmulinteger_and_add(trans_b):
    w = W.T.copy() if trans_b else W
    m = _model(f"y = Gemm<transB={trans_b}>(x, w, b)", {"w": w, "b": B})
    q = quantize_dynamic_integer(m)
    assert _ops(q)[-1] == "Add" and "MatMulInteger" in _ops(q)
    ref, got = _run(m, X), _run(q, X)
    assert np.linalg.norm(ref - got) / np.linalg.norm(ref) < 0.03


def test_conv_uses_convinteger_with_reshaped_bias():
    m = _model(
        "y = Conv<pads=[1,1,1,1]>(x, w, b)",
        {"w": RNG.standard_normal((4, 3, 3, 3)), "b": RNG.standard_normal(4)},
        shape="[1,3,6,6]",
    )
    q = quantize_dynamic_integer(m)
    assert "ConvInteger" in _ops(q) and "Reshape" in _ops(q)
    x = RNG.standard_normal((1, 3, 6, 6)).astype(np.float32)
    ref, got = _run(m, x), _run(q, x)
    assert np.linalg.norm(ref - got) / np.linalg.norm(ref) < 0.03


def test_untouched_cases_stay_float():
    # non-constant weight, grouped conv, Gemm with alpha != 1, excluded node
    m = _model("y = MatMul(x, x2) x2 = Transpose(x)", {})
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(reversed(nodes))
    assert _ops(quantize_dynamic_integer(m)) == _ops(m)
    g = _model("y = Gemm<alpha=2.0>(x, w, b)", {"w": W, "b": B})
    assert _ops(quantize_dynamic_integer(g)) == ["Gemm"]
    e = _model("y = MatMul(x, w)", {"w": W})
    e.graph.node[0].name = "keep"
    assert _ops(quantize_dynamic_integer(e, exclude_nodes=["keep"])) == ["MatMul"]


def test_preset_is_wired_through_the_shim():
    m = _model("y = MatMul(x, w)", {"w": W})
    out = qc.ModelQuantizer(
        qc.QConfig.get_default_config("UINT8_DYNAMIC_QUANT")
    ).quantize_model(m)
    assert "MatMulInteger" in _ops(out)
    cfg = qc.QConfig.get_default_config("UINT8_DYNAMIC_QUANT")
    cfg.algo_config = [qc.CLEConfig()]
    with pytest.raises(NotImplementedError, match="dynamic"):
        qc.ModelQuantizer(cfg).quantize_model(m)


@pytest.mark.parametrize("opset", [9, 10])
def test_below_opset_11_the_scale_and_zero_point_are_computed_by_nodes(opset):
    # no DynamicQuantizeLinear (an opset 11 operator): ONNX Runtime's unfused
    # pattern, which is what Quark emits (tests/test_quark_low_opset_parity.py)
    m = _model("y = MatMul(x, w)", {"w": W})
    m.opset_import[0].version = opset
    q = quantize_dynamic_integer(m)
    ops = _ops(q)
    assert "DynamicQuantizeLinear" not in ops
    assert ops[:9] == [
        "ReduceMin",
        "ReduceMax",
        "Sub",
        "Div",
        "Sub",
        "Div",
        "Floor",
        "Cast",
        "QuantizeLinear",
    ]
    assert ops.count("MatMulInteger") == 1
    inits = {t.name: numpy_helper.to_array(t) for t in q.graph.initializer}
    assert inits["fixed_quantization_range_uint8"] == 255.0
    assert inits["fixed_zero"] == 0.0


def test_input_model_is_not_modified_and_arguments_checked():
    m = _model("y = MatMul(x, w)", {"w": W})
    before = m.SerializeToString()
    quantize_dynamic_integer(m)
    assert m.SerializeToString() == before
    with pytest.raises(ValueError, match="weight_dtype"):
        quantize_dynamic_integer(
            _model("y = MatMul(x, w)", {"w": W}), weight_dtype="int4"
        )
