import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim.nan_check import check_nan


def _model(body: str, opset: int = 15) -> onnx.ModelProto:
    return parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')


_SQRT = """
agraph (float[4] X) => (float[4] Y) {
    Y = Sqrt(X)
}
"""

_LOG = """
agraph (float[4] X) => (float[4] Y) {
    Y = Log(X)
}
"""

_DIV = """
agraph (float[4] X, float[4] Z) => (float[4] Y) {
    Y = Div(X, Z)
}
"""

_EXP = """
agraph (float[4] X) => (float[4] Y) {
    Y = Exp(X)
}
"""

_SQRT_SIGMOID = """
agraph (float[4] X) => (float[4] Y) {
    S = Sigmoid(X)
    Y = Sqrt(S)
}
"""


def test_sqrt_of_range_crossing_zero_is_nan_hazard():
    rep = check_nan(_model(_SQRT), {"X": (-1.0, 4.0)})
    assert [(h.op_type, h.kind) for h in rep.hazards] == [("Sqrt", "nan")]
    assert rep.hazards[0].count == 4
    assert not rep.nan_free


def test_sqrt_of_nonnegative_range_is_nan_free():
    rep = check_nan(_model(_SQRT), {"X": (0.0, 4.0)})
    assert rep.nan_free and rep.finite and rep.complete


def test_log_at_zero_is_inf_hazard_but_nan_free():
    rep = check_nan(_model(_LOG), {"X": (0.0, 1.0)})
    assert [(h.op_type, h.kind) for h in rep.hazards] == [("Log", "inf")]
    assert rep.nan_free
    assert not rep.finite


def test_div_zero_by_zero_is_nan_hazard():
    rep = check_nan(_model(_DIV), {"X": (0.0, 1.0), "Z": (-1.0, 1.0)})
    assert any(h.kind == "nan" and h.op_type == "Div" for h in rep.hazards)


def test_div_by_range_crossing_zero_is_inf_hazard_only():
    rep = check_nan(_model(_DIV), {"X": (1.0, 2.0), "Z": (-1.0, 1.0)})
    assert [(h.kind) for h in rep.hazards] == ["inf"]
    assert rep.nan_free


def test_div_by_strictly_positive_is_finite():
    rep = check_nan(_model(_DIV), {"X": (1.0, 2.0), "Z": (0.5, 1.0)})
    assert rep.finite and rep.complete


def test_exp_overflow_in_float32_is_inf_hazard():
    rep = check_nan(_model(_EXP), {"X": (0.0, 100.0)})
    assert [(h.op_type, h.kind) for h in rep.hazards] == [("Exp", "inf")]
    assert rep.nan_free and not rep.finite


def test_exp_within_float32_range_is_finite():
    rep = check_nan(_model(_EXP), {"X": (0.0, 10.0)})
    assert rep.finite and rep.complete


def test_interval_propagates_through_sigmoid_to_sqrt():
    rep = check_nan(_model(_SQRT_SIGMOID), {"X": (-10.0, 10.0)})
    assert rep.nan_free and rep.complete


def test_unbounded_input_flags_arithmetic():
    rep = check_nan(_model(_SQRT))
    assert [h.op_type for h in rep.hazards] == ["Sqrt"]
    assert not rep.nan_free


def test_unmodelled_op_is_listed_and_does_not_block_nan_free():
    body = """
agraph (float[4] X) => (float[4] Y) {
    Y = Softmax(X)
}
"""
    rep = check_nan(_model(body), {"X": (-1.0, 1.0)})
    assert rep.unmodelled == ["Softmax"]
    assert rep.nan_free and not rep.complete


def test_unmodelled_op_with_unbounded_operand_is_nan_hazard():
    body = """
agraph (float[4] X) => (float[4] Y) {
    Y = Softmax(X)
}
"""
    rep = check_nan(_model(body))
    assert any(h.op_type == "Softmax" and h.kind == "nan" for h in rep.hazards)


def test_shape_ranged_tensor_is_unanalysed():
    body = """
agraph (float[4] X) => (float[N] Y) {
    I = NonZero(X)
    C = Cast<to=1>(I)
    Y = Sqrt(C)
}
"""
    rep = check_nan(_model(body), {"X": (0.0, 1.0)})
    assert rep.unanalysed
    assert not rep.nan_free


def test_nan_free_verdict_matches_onnxruntime_on_sampled_inputs():
    ort = pytest.importorskip("onnxruntime")
    body = """
agraph (float[4] X, float[4] Z) => (float[4] Y) {
    S = Sigmoid(X)
    L = Log(S)
    M = Neg(L)
    Q = Div(M, Z)
    Y = Sqrt(Q)
}
"""
    model = _model(body)
    rng = np.random.default_rng(0)
    cases = [
        ({"X": (-3.0, 3.0), "Z": (0.5, 2.0)}, True),
        ({"X": (-3.0, 3.0), "Z": (-2.0, 2.0)}, False),
    ]
    for ranges, expect_nan_free in cases:
        rep = check_nan(model, ranges)
        assert rep.nan_free is expect_nan_free
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        saw_nan = False
        for _ in range(200):
            feed = {
                k: rng.uniform(lo, hi, size=4).astype(np.float32)
                for k, (lo, hi) in ranges.items()
            }
            (out,) = sess.run(None, feed)
            saw_nan = saw_nan or bool(np.isnan(out).any())
        if rep.nan_free:
            assert not saw_nan
