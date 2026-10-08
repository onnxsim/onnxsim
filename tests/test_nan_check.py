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
    Y = Bernoulli(X)
}
"""
    rep = check_nan(_model(body), {"X": (-1.0, 1.0)})
    assert rep.unmodelled == ["Bernoulli"]
    assert rep.nan_free and not rep.complete


def test_unmodelled_op_with_unbounded_operand_is_nan_hazard():
    body = """
agraph (float[4] X) => (float[4] Y) {
    Y = Bernoulli(X)
}
"""
    rep = check_nan(_model(body))
    assert any(h.op_type == "Bernoulli" and h.kind == "nan" for h in rep.hazards)


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


def _kinds(rep):
    return sorted({(h.op_type, h.kind) for h in rep.hazards})


def test_matmul_partial_sums_overflow_float32():
    body = """
agraph (float[1,4] X, float[4,1] W) => (float[1,1] Y) {
    Y = MatMul(X, W)
}
"""
    rep = check_nan(_model(body), {"X": (-1e38, 1e38), "W": (-10.0, 10.0)})
    assert ("MatMul", "inf") in _kinds(rep)
    small = check_nan(_model(body), {"X": (-1.0, 1.0), "W": (-10.0, 10.0)})
    assert small.finite and small.complete


def test_softmax_is_modelled_and_flags_unbounded_input():
    body = """
agraph (float[4] X) => (float[4] Y) {
    Y = Softmax(X)
}
"""
    rep = check_nan(_model(body), {"X": (-1.0, 1.0)})
    assert rep.complete and rep.finite
    rep = check_nan(_model(body))
    assert ("Softmax", "nan") in _kinds(rep)


def test_gelu_nan_only_for_unbounded_below():
    body = """
agraph (float[4] X) => (float[4] Y) {
    Y = Gelu(X)
}
"""
    assert check_nan(_model(body, opset=20), {"X": (-5.0, 5.0)}).finite
    assert ("Gelu", "nan") in _kinds(check_nan(_model(body, opset=20)))


def test_variadic_sum_of_opposite_infinities_is_nan():
    body = """
agraph (float[4] X, float[4] Z) => (float[4] Y) {
    Y = Sum(X, Z)
}
"""
    assert ("Sum", "nan") in _kinds(check_nan(_model(body)))
    assert check_nan(_model(body), {"X": (0.0, 1.0), "Z": (0.0, 1.0)}).finite


def test_reduce_sum_overflow_uses_element_count():
    body = """
agraph (float[1000] X) => (float[1] Y) {
    Y = ReduceSum<keepdims=1>(X)
}
"""
    rep = check_nan(_model(body), {"X": (-1e36, 1e36)})
    assert ("ReduceSum", "inf") in _kinds(rep)
    assert check_nan(_model(body), {"X": (-1.0, 1.0)}).finite


def test_cast_to_float16_overflows_at_fp16_max():
    body = """
agraph (float[4] X) => (float16[4] Y) {
    Y = Cast<to=10>(X)
}
"""
    assert ("Cast", "inf") in _kinds(check_nan(_model(body), {"X": (0.0, 1e5)}))
    assert check_nan(_model(body), {"X": (0.0, 100.0)}).finite


def test_float16_model_overflows_at_fp16_max():
    body = """
agraph (float16[4] X) => (float16[4] Y) {
    Y = Mul(X, X)
}
"""
    assert ("Mul", "inf") in _kinds(check_nan(_model(body), {"X": (0.0, 300.0)}))
    assert check_nan(_model(body), {"X": (0.0, 10.0)}).finite


def test_domain_rules_for_asin_atanh_log1p():
    for body, rng, expected in [
        ("Y = Asin(X)", (-2.0, 0.5), ("Asin", "nan")),
        ("Y = Atanh(X)", (-1.0, 0.5), ("Atanh", "inf")),
        ("Y = Log1p(X)", (-1.0, 1.0), ("Log1p", "inf")),
    ]:
        text = f"agraph (float[4] X) => (float[4] Y) {{ {body} }}"
        rep = check_nan(_model(text), {"X": rng})
        assert expected in _kinds(rep), body
        assert ("Log1p", "nan") not in _kinds(rep)


def test_batch_norm_negative_variance_is_nan():
    body = """
agraph (float[1,2,3] X, float[2] S, float[2] B, float[2] M, float[2] V)
    => (float[1,2,3] Y) {
    Y = BatchNormalization(X, S, B, M, V)
}
"""
    ranges = {"X": (-1.0, 1.0), "S": (1.0, 1.0), "B": (0.0, 0.0), "M": (0.0, 0.0)}
    assert ("BatchNormalization", "nan") in _kinds(
        check_nan(_model(body), {**ranges, "V": (-1.0, 1.0)})
    )
    assert check_nan(_model(body), {**ranges, "V": (0.5, 1.0)}).finite


def test_layer_norm_squared_deviation_overflow():
    body = """
agraph (float[4] X, float[4] S) => (float[4] Y) {
    Y = LayerNormalization<axis=-1>(X, S)
}
"""
    scale = {"S": (1.0, 1.0)}
    rep = check_nan(_model(body, opset=17), {"X": (-1e19, 1e19), **scale})
    assert ("LayerNormalization", "inf") in _kinds(rep)
    assert check_nan(_model(body, opset=17), {"X": (-1.0, 1.0), **scale}).finite


def test_layer_norm_output_is_bounded_by_scale_and_bias():
    body = """
agraph (float[4] X, float[4] S, float[4] B) => (float[4] Y) {
    Y = LayerNormalization<axis=-1>(X, S, B)
}
"""
    ranges = {"X": (-1e6, 1e6), "S": (0.5, 2.0), "B": (-1.0, 1.0)}
    rep = check_nan(_model(body, opset=17), ranges)
    assert rep.finite and rep.complete


def test_expanded_ops_nan_free_verdict_matches_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    body = """
agraph (float[2,3] X, float[3,3] W, float[3] S, float[3] B, float[3] M, float[3] V)
    => (float[2,3] Y) {
    H = MatMul(X, W)
    G = Tanh(H)
    N = BatchNormalization(G, S, B, M, V)
    Y = Softmax(N)
}
"""
    model = _model(body)
    ranges = {
        "X": (-2.0, 2.0),
        "W": (-1.0, 1.0),
        "S": (0.5, 1.5),
        "B": (-1.0, 1.0),
        "M": (-1.0, 1.0),
        "V": (0.5, 2.0),
    }
    rep = check_nan(model, ranges)
    assert rep.nan_free and rep.complete
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(1)
    for _ in range(200):
        feed = {
            k: rng.uniform(lo, hi, size=s).astype(np.float32)
            for (k, (lo, hi)), s in zip(
                ranges.items(), [(2, 3), (3, 3), (3,), (3,), (3,), (3,)]
            )
        }
        (out,) = sess.run(None, feed)
        assert not np.isnan(out).any()


def test_new_interval_transfers_contain_onnxruntime_outputs():
    ort = pytest.importorskip("onnxruntime")
    from onnxsim import interval

    body = """
agraph (float[4] X, float[4] S, float[4] B) => (float[4] G, float[4] Z, float16[4] Q, float[4] L) {
    G = Gelu(X)
    Z = Sum(X, S)
    Q = Cast<to=10>(Z)
    L = LayerNormalization<axis=-1>(X, S, B)
}
"""
    model = _model(body, opset=20)
    ranges = {"X": (-30.0, 30.0), "S": (-2.0, 2.0), "B": (-1.0, 1.0)}
    res = interval.propagate(model, ranges)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2)
    for _ in range(200):
        feed = {
            k: rng.uniform(lo, hi, size=4).astype(np.float32)
            for k, (lo, hi) in ranges.items()
        }
        for name, out in zip(["G", "Z", "Q", "L"], sess.run(None, feed)):
            lo, hi = res.intervals[name]
            v = out.astype(np.float64)
            assert np.all(v >= lo - 1e-5 * (1 + np.abs(lo))), name
            assert np.all(v <= hi + 1e-5 * (1 + np.abs(hi))), name


def test_dequantize_linear_overflow_uses_integer_range():
    body = """
agraph (int8[4] X, float[1] S) => (float[4] Y) {
    Y = DequantizeLinear(X, S)
}
"""
    rep = check_nan(_model(body, opset=15), {"S": (1e38, 1e38)})
    assert any(h.op_type == "DequantizeLinear" and h.kind == "inf" for h in rep.hazards)


def test_dequantize_linear_with_small_scale_is_finite():
    body = """
agraph (int8[4] X, float[1] S) => (float[4] Y) {
    Y = DequantizeLinear(X, S)
}
"""
    rep = check_nan(_model(body, opset=15), {"S": (0.001, 0.001)})
    assert rep.finite and rep.complete


def test_quantize_linear_zero_over_zero_is_nan_hazard():
    body = """
agraph (float[4] X, float[4] S) => (uint8[4] Y) {
    Y = QuantizeLinear(X, S)
}
"""
    assert ("QuantizeLinear", "nan") in _kinds(check_nan(_model(body, opset=15)))
    rep = check_nan(_model(body, opset=15), {"X": (1.0, 2.0), "S": (0.5, 1.0)})
    assert rep.nan_free and rep.complete


def test_max_pool_and_hardmax_are_modelled():
    body = """
agraph (float[4] X) => (float[4] Y, float[4] Z) {
    Y = Hardmax(X)
    Z = Relu(X)
}
"""
    rep = check_nan(_model(body), {"X": (-1.0, 1.0)})
    assert rep.complete and rep.finite


_EINSUM = """
agraph (float[2,3] A, float[3,2] B) => (float[2,2] Y) {
    Y = Einsum<equation="ij,jk->ik">(A, B)
}
"""

_ATTENTION = """
agraph (float[1,1,2,4] Q, float[1,1,2,4] K, float[1,1,2,4] V) => (float[1,1,2,4] Y) {
    Y = Attention(Q, K, V)
}
"""


def test_einsum_bounded_inputs_are_modelled_and_finite():
    rep = check_nan(_model(_EINSUM, opset=15), {"A": (-1, 1), "B": (-1, 1)})
    assert rep.complete and rep.finite


def test_einsum_unbounded_operand_may_be_infinite():
    rep = check_nan(_model(_EINSUM, opset=15))
    assert not rep.nan_free


def test_einsum_sum_of_products_may_overflow_float32():
    rep = check_nan(_model(_EINSUM, opset=15), {"A": (-1e20, 1e20), "B": (-1e20, 1e20)})
    assert any(h.kind == "inf" and h.op_type == "Einsum" for h in rep.hazards)


def test_attention_bounded_inputs_are_modelled_and_finite():
    ranges = {"Q": (-1, 1), "K": (-1, 1), "V": (-1, 1)}
    rep = check_nan(_model(_ATTENTION, opset=23), ranges)
    assert rep.complete and rep.finite


def test_attention_logits_overflow_leaves_nan_in_softmax():
    ranges = {"Q": (-1e20, 1e20), "K": (-1e20, 1e20), "V": (-1, 1)}
    rep = check_nan(_model(_ATTENTION, opset=23), ranges)
    assert any("logits" in h.detail for h in rep.hazards)
    assert not rep.nan_free


def test_attention_softcap_bounds_logits():
    body = """
agraph (float[1,1,2,4] Q, float[1,1,2,4] K, float[1,1,2,4] V) => (float[1,1,2,4] Y) {
    Y = Attention<softcap=30.0>(Q, K, V)
}
"""
    ranges = {"Q": (-1e20, 1e20), "K": (-1e20, 1e20), "V": (-1, 1)}
    rep = check_nan(_model(body, opset=23), ranges)
    assert not any("logits" in h.detail for h in rep.hazards)
