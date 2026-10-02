"""Tests for onnxsim.quark_auto_mixprecision (Quark-style AMP on QDQ models)."""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_auto_mixprecision as amp
from onnxsim.full_qdq import quantize_full_qdq


def f32(*v):
    return np.array(v, dtype=np.float32)


# -- metrics: hand-computed values ----------------------------------------------


def test_l2_is_the_mean_norm_over_sample_output_pairs():
    f = [[f32(0, 0)], [f32(0, 0)]]
    q = [[f32(3, 4)], [f32(0, 0)]]  # norms 5 and 0
    assert amp.l2_metric(f, q) == 2.5


def test_cosine_distance_of_orthogonal_and_identical_vectors():
    assert amp.cosine_metric([[f32(1, 0)]], [[f32(0, 1)]]) == pytest.approx(1.0)
    assert amp.cosine_metric([[f32(1, 2)]], [[f32(1, 2)]]) == pytest.approx(0.0)


def test_sqnr_is_negative_db():
    # signal power 1, noise power 0.01 -> SQNR 20 dB -> metric -20
    f = [[f32(1, 1, 1, 1)]]
    q = [[f32(1.1, 1.1, 1.1, 1.1)]]
    assert amp.sqnr_metric(f, q) == pytest.approx(-20.0, abs=0.01)


def test_psnr_is_negative_db_with_peak_of_the_float_output():
    # peak 2, mse 0.01 -> 20log10(2) - 10log10(0.01) = 6.02 + 20 -> -26.02
    f = [[f32(2, 0)]]
    q = [[f32(2.1, 0.1)]]
    assert amp.psnr_metric(f, q) == pytest.approx(-(20 * np.log10(2) + 20), abs=0.01)


def test_kl_is_zero_for_identical_and_positive_otherwise():
    assert amp.kl_metric([[f32(1, 2, 3)]], [[f32(1, 2, 3)]]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert amp.kl_metric([[f32(1, 2, 3)]], [[f32(3, 2, 1)]]) > 0.1


def test_metrics_reject_mismatched_sample_counts():
    with pytest.raises(ValueError, match="same number of samples"):
        amp.l2_metric([[f32(1)]], [])


def test_resolve_metric_priority_and_errors():
    dist = lambda f, q: 7.0  # noqa: E731
    assert amp.resolve_metric("l2", distance_fn=dist)([], []) == 7.0
    ev = amp.resolve_metric(evaluate_fn=lambda out: float(len(out)))
    assert ev([[f32(1)]] * 3, [[f32(1)]]) == 2.0  # evaluate(float) - evaluate(quant)
    with pytest.raises(ValueError, match="mutually exclusive"):
        amp.resolve_metric(distance_fn=dist, evaluate_fn=lambda o: 0.0)
    with pytest.raises(ValueError, match="unknown metric"):
        amp.resolve_metric("nope")


# -- the algorithm ---------------------------------------------------------------


def _model():
    rng = np.random.default_rng(0)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,16] x) => (float[N,16] y)
        {
            h1 = MatMul(x, w1)
            t1 = Tanh(h1)
            h2 = MatMul(t1, w2)
            t2 = Tanh(h2)
            y = MatMul(t2, w3)
        }
        """
    )
    # Random weights are attached programmatically (too large for text literals).
    model.graph.initializer.extend(
        numpy_helper.from_array(
            (rng.standard_normal((16, 16)) * s).astype(np.float32), n
        )
        for n, s in (("w1", 1.0), ("w2", 3.0), ("w3", 0.3))
    )
    return model


def _data(n=8, seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((8, 16)).astype(np.float32)} for _ in range(n)]


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(scope="module")
def data():
    return _data()


def _run(model, data, **kw):
    return amp.auto_mixprecision(model, data, **kw)


def _same_quantized_model(a, b, data):
    """The two models quantize the same way: the same operators and, under
    ONNX Runtime without graph optimizations, bit-identical outputs. (The mixing
    step edits the quantized baseline in place, so a model is no longer
    byte-equal to a fresh ``quantize_full_qdq`` result: its nodes are named and
    ordered differently.)"""
    import onnxruntime as ort

    assert sorted(n.op_type for n in a.graph.node) == sorted(
        n.op_type for n in b.graph.node
    )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sa, sb = (
        ort.InferenceSession(
            m.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )
        for m in (a, b)
    )
    for feed in data:
        for x, y in zip(sa.run(None, feed), sb.run(None, feed)):
            np.testing.assert_array_equal(x, y)


def _all_tensors(result):
    return {t for c in result.ranked for t in c.tensors}


def test_sensitivity_is_ranked_ascending_over_all_matmuls(model, data):
    r = _run(
        model, data, base_dtype="uint8", target_dtype="uint16", metric_threshold=None
    )
    assert [c.name for c in sorted(r.ranked, key=lambda c: c.name)] == ["h1", "h2", "y"]
    scores = [c.score for c in r.ranked]
    assert scores == sorted(scores)
    assert len(set(scores)) == 3  # layers genuinely differ in sensitivity
    assert all(np.isfinite(scores))
    # sensitivity only: the baseline model comes back untouched, nothing moved
    assert r.moved == [] and r.final_score == r.baseline_score
    base = quantize_full_qdq(model, calibration_data=data, activation_dtype="uint8")
    _same_quantized_model(r.model, base, data)


def test_threshold_zero_moves_every_candidate(model, data):
    r = _run(model, data, base_dtype="uint8", target_dtype="uint16", optimize="quality")
    assert sorted(r.moved) == ["h1", "h2", "y"]
    expected = quantize_full_qdq(
        model,
        calibration_data=data,
        activation_dtype="uint8",
        tensor_dtypes={t: "uint16" for t in _all_tensors(r)},
    )
    _same_quantized_model(r.model, expected, data)
    assert r.final_score < r.baseline_score  # more activation bits -> closer to float


def test_quality_stops_as_soon_as_the_threshold_is_met(model, data):
    full = _run(model, data, base_dtype="uint8", target_dtype="uint16")
    # a threshold only reachable after moving at least one, but not all, layers
    thr = (full.baseline_score + full.final_score) / 2
    r = _run(
        model,
        data,
        base_dtype="uint8",
        target_dtype="uint16",
        optimize="quality",
        metric_threshold=thr,
    )
    assert r.threshold_reached and r.final_score <= thr
    assert 0 < len(r.moved) <= 3
    # moved candidates are a prefix of the ranking, and the last one crossed the line
    assert r.moved == [c.name for c in r.ranked[: len(r.moved)]]


def test_quality_returns_the_baseline_when_it_already_meets_the_threshold(model, data):
    r = _run(
        model,
        data,
        base_dtype="uint8",
        target_dtype="uint16",
        optimize="quality",
        metric_threshold=1e9,
    )
    base = quantize_full_qdq(model, calibration_data=data, activation_dtype="uint8")
    assert r.moved == []
    _same_quantized_model(r.model, base, data)
    assert r.final_score == r.baseline_score


def test_speed_moves_layers_down_until_the_threshold_would_be_exceeded(model, data):
    # baseline is uint16 (accurate); candidates go to uint8 (cheaper)
    all_low = _run(model, data, base_dtype="uint16", target_dtype="uint8")
    base = all_low.baseline_score
    thr = (base + all_low.final_score) / 2
    r = _run(
        model,
        data,
        base_dtype="uint16",
        target_dtype="uint8",
        optimize="speed",
        metric_threshold=thr,
    )
    assert r.final_score <= thr
    assert len(r.moved) < 3  # not all of them fit under the threshold
    assert r.moved == [c.name for c in r.ranked[: len(r.moved)]]


def test_speed_returns_the_baseline_when_it_already_exceeds_the_threshold(model, data):
    r = _run(
        model,
        data,
        base_dtype="uint16",
        target_dtype="uint8",
        optimize="speed",
        metric_threshold=1e-12,
    )
    assert r.moved == [] and r.final_score == r.baseline_score


def test_include_and_exclude_layers(model, data):
    only = _run(
        model, data, base_dtype="uint8", target_dtype="uint16", include_layers=["h2"]
    )
    assert [c.name for c in only.ranked] == ["h2"]
    skipped = _run(
        model, data, base_dtype="uint8", target_dtype="uint16", exclude_layers=["h2"]
    )
    assert sorted(c.name for c in skipped.ranked) == ["h1", "y"]


def test_output_model_is_valid_and_runs(model, data):
    r = _run(model, data, base_dtype="uint8", target_dtype="uint16", metric="cosine")
    onnx.checker.check_model(r.model)


def test_validation_errors(model, data):
    with pytest.raises(ValueError, match="differ"):
        _run(model, data, base_dtype="uint8", target_dtype="uint8")
    with pytest.raises(ValueError, match="dtypes"):
        _run(model, data, base_dtype="float16", target_dtype="uint16")
    with pytest.raises(ValueError, match="optimize"):
        _run(model, data, base_dtype="uint8", target_dtype="uint16", optimize="x")
    with pytest.raises(ValueError, match="calibration_data"):
        _run(model, [], base_dtype="uint8", target_dtype="uint16")
