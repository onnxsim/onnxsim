"""Batched min/max calibration: the per-tensor (min, max) reduction runs inside
ONNX Runtime, and ``calibration_batch_size`` merges consecutive samples into
one run.

The reductions are exact (min and max commute with batching for a graph that
treats samples independently), so every test compares against a plain numpy
reference computed from the full activation tensors.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim.calibration import _batched_feeds, calibrate


def _model(body, initializer=(), opset=17, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _conv_mlp():
    """Batch-independent: a conv, a ReLU and a MatMul over a dynamic spatial size."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(
        rng.standard_normal((4, 3, 3, 3)).astype(np.float32), "w"
    )
    v = numpy_helper.from_array(rng.standard_normal((4, 5)).astype(np.float32), "v")
    return _model(
        "g (float[N, 3, H, W] x) => (float[N, 5] y) {"
        "  c = Conv<pads = [1, 1, 1, 1]>(x, w)"
        "  r = Relu(c)"
        "  p = GlobalAveragePool(r)"
        "  f = Flatten(p)"
        "  y = MatMul(f, v)"
        "}",
        initializer=[w, v],
    )


def _samples(n, seed=0, shape=(1, 3, 6, 6)):
    rng = np.random.default_rng(seed)
    return [
        {"x": (rng.standard_normal(shape) * (1 + i)).astype(np.float32)}
        for i in range(n)
    ]


def _reference_minmax(model, names, samples):
    """Per-tensor (min, max) over every sample, from the full tensors."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    existing = {o.name for o in m.graph.output}
    for name in names:
        if name not in existing:
            m.graph.output.append(onnx.ValueInfoProto(name=name))
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    ranges = {}
    for feed in samples:
        for name, value in zip(out_names, sess.run(out_names, feed)):
            if name not in names:
                continue
            lo, hi = float(value.min()), float(value.max())
            prev = ranges.get(name, (lo, hi))
            ranges[name] = (min(prev[0], lo), max(prev[1], hi))
    return ranges


class _CountingSession(ort.InferenceSession):
    runs = 0

    def run(self, output_names, input_feed, run_options=None):
        _CountingSession.runs += 1
        return super().run(output_names, input_feed, run_options)


@pytest.fixture
def count_runs(monkeypatch):
    _CountingSession.runs = 0
    monkeypatch.setattr(ort, "InferenceSession", _CountingSession)
    return _CountingSession


NAMES = ["x", "c", "r", "p", "f"]


def test_ort_reduction_matches_full_tensor_reference():
    model = _conv_mlp()
    samples = _samples(6)
    got = calibrate(model, samples, tensor_names=NAMES)
    want = _reference_minmax(model, NAMES, samples)
    assert got.keys() == want.keys()
    for name in NAMES:
        assert got[name] == pytest.approx(want[name], rel=0, abs=0), name


def test_batched_run_is_exact_and_runs_fewer_times(count_runs):
    model = _conv_mlp()
    samples = _samples(8)
    unbatched = calibrate(model, samples, tensor_names=NAMES)
    runs_unbatched = count_runs.runs

    count_runs.runs = 0
    batched = calibrate(model, samples, tensor_names=NAMES, calibration_batch_size=4)
    runs_batched = count_runs.runs

    assert batched == unbatched  # min/max are exact under batching
    assert runs_unbatched == 8 and runs_batched == 2


def test_batched_run_with_uneven_tail():
    model = _conv_mlp()
    samples = _samples(7)
    got = calibrate(model, samples, tensor_names=NAMES, calibration_batch_size=3)
    want = _reference_minmax(model, NAMES, samples)
    for name in NAMES:
        assert got[name] == pytest.approx(want[name], rel=0, abs=0), name


def test_mismatched_shapes_are_not_merged():
    # a spatial-size change breaks the merge: those samples run on their own
    model = _conv_mlp()
    samples = _samples(3, shape=(1, 3, 6, 6)) + _samples(2, seed=9, shape=(1, 3, 4, 8))
    got = calibrate(model, samples, tensor_names=NAMES, calibration_batch_size=8)
    want = _reference_minmax(model, NAMES, samples)
    for name in NAMES:
        assert got[name] == pytest.approx(want[name], rel=0, abs=0), name


def test_feed_merging_groups_by_shape_and_batch_size():
    a = [{"x": np.zeros((1, 2), np.float32)} for _ in range(3)]
    b = [{"x": np.zeros((1, 5), np.float32)} for _ in range(2)]
    feeds = list(_batched_feeds(a + b, batch_size=2))
    assert [f["x"].shape for f in feeds] == [(2, 2), (1, 2), (2, 5)]


def test_batch_size_one_yields_samples_unchanged():
    samples = _samples(3)
    feeds = list(_batched_feeds(samples, batch_size=1))
    assert feeds == samples


def test_empty_tensor_is_left_out_of_the_ranges():
    # a zero-length slice has no extremes; it must not appear in the ranges
    model = _model(
        "g (float[N, 4] x) => (float[N, 4] y) {  e = Slice(x, s, t, a)  y = Relu(x)}",
        initializer=[
            numpy_helper.from_array(np.array([0], np.int64), "s"),
            numpy_helper.from_array(np.array([0], np.int64), "t"),
            numpy_helper.from_array(np.array([1], np.int64), "a"),
        ],
    )
    samples = [{"x": np.ones((2, 4), np.float32)}]
    got = calibrate(model, samples, tensor_names=["x", "e", "y"])
    assert "e" not in got
    assert got["x"] == (1.0, 1.0) and got["y"] == (1.0, 1.0)


@pytest.mark.parametrize("method", ["minmax_mean", "entropy", "percentile", "mse"])
def test_batching_rejected_for_methods_it_would_change(method):
    with pytest.raises(ValueError, match="needs method='minmax'"):
        calibrate(
            _conv_mlp(),
            _samples(4),
            tensor_names=NAMES,
            method=method,
            calibration_batch_size=2,
        )


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="must be >= 1"):
        calibrate(
            _conv_mlp(), _samples(2), tensor_names=NAMES, calibration_batch_size=0
        )


def test_quantize_static_batched_matches_unbatched():
    model = _conv_mlp()
    samples = _samples(6)
    a = onnxsim.quantize_static(model, samples)
    b = onnxsim.quantize_static(model, samples, calibration_batch_size=3)
    assert a.SerializeToString() == b.SerializeToString()
