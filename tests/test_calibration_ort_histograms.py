"""Pass 2 of calibration counts histograms inside ONNX Runtime.

The per-tensor histograms that the entropy / mse / percentile / auto methods
read used to be filled by ``np.histogram`` on every activation tensor, which
returned each full tensor to Python. Now the bin index and its numpy-style
edge correction are ordinary ONNX nodes, and only the ``2 * num_bins`` counts
come back. These tests pin that the counts equal ``_StreamingHistogram``'s
``np.histogram`` counts exactly, including NaN, infinities and values sitting
on or one float32 ulp away from a bin edge.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import parser

from onnxsim.calibration import (
    _accumulate_histograms_in_ort,
    _StreamingHistogram,
    collect_calibration_stats,
)


def _model(body, opset=17):
    return parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')


def _identity_model(opset=17, dtype=np.float32):
    t = "double" if np.dtype(dtype) == np.float64 else "float"
    return _model(f"g ({t}[N] x) => ({t}[N] y) {{ y = Identity(x) }}", opset)


def _ort_counts(abs_max, num_bins, feeds, dtype=np.float32):
    hist = _StreamingHistogram(abs_max, num_bins)
    _accumulate_histograms_in_ort(
        _identity_model(dtype=dtype),
        {"x": hist},
        {"x": np.dtype(dtype)},
        feeds,
        ort.SessionOptions(),
        None,
    )
    return hist.counts


def _numpy_counts(abs_max, num_bins, feeds):
    hist = _StreamingHistogram(abs_max, num_bins)
    for feed in feeds:
        hist.add(feed["x"])
    return hist.counts


def test_counts_match_numpy_on_mixed_values():
    rng = np.random.default_rng(0)
    A, B = 3.0, 64
    edges = np.linspace(-A, A, B + 1)
    x = (rng.standard_normal(20000) * 1.5).astype(np.float32)
    on_edges = edges[rng.integers(0, B + 1, 500)].astype(np.float32)
    special = np.array(
        [np.nan, np.inf, -np.inf, A, -A, A + 1e-3, -A - 1e-3, 0.0], np.float32
    )
    feeds = [
        {"x": np.concatenate([x, on_edges, special])},
        {"x": np.concatenate([x[:300], special[::-1]])},
    ]
    np.testing.assert_array_equal(_ort_counts(A, B, feeds), _numpy_counts(A, B, feeds))


@pytest.mark.parametrize("num_bins, A", [(32, 3.0), (2048, 6.0)])
def test_every_float32_neighbourhood_of_an_edge_is_counted_like_numpy(num_bins, A):
    # all float32 values within 50 ulps of every edge (both sides), in range
    edges = np.linspace(-A, A, num_bins + 1).astype(np.float32)
    up = edges.copy()
    down = edges.copy()
    chunks = [edges]
    for _ in range(50):
        up = np.nextafter(up, np.float32(np.inf))
        down = np.nextafter(down, np.float32(-np.inf))
        chunks += [up, down]
    x = np.concatenate(chunks).astype(np.float32)
    x = x[(x >= -A) & (x <= A)]
    feeds = [{"x": x}]
    np.testing.assert_array_equal(
        _ort_counts(A, num_bins, feeds), _numpy_counts(A, num_bins, feeds)
    )


def test_float64_samples_use_float64_edges():
    rng = np.random.default_rng(3)
    A, B = 2.5, 128
    x = rng.standard_normal(30000).astype(np.float64) * 1.2
    feeds = [{"x": x}]
    hist = _StreamingHistogram(A, B)
    for f in feeds:
        hist.add(f["x"])
    got = _ort_counts(A, B, feeds, dtype=np.float64)
    np.testing.assert_array_equal(got, hist.counts)


def test_counts_add_across_feeds_and_keep_dtype_int64():
    feeds = [{"x": np.full(5, 0.5, np.float32)}, {"x": np.full(7, -0.5, np.float32)}]
    counts = _ort_counts(1.0, 4, feeds)
    assert counts.dtype == np.int64
    assert counts.sum() == 12
    np.testing.assert_array_equal(counts, _numpy_counts(1.0, 4, feeds))


def _conv_model(opset=17):
    rng = np.random.default_rng(0)
    from onnx import numpy_helper

    w = numpy_helper.from_array(
        rng.standard_normal((4, 3, 3, 3)).astype(np.float32), "w"
    )
    v = numpy_helper.from_array(rng.standard_normal((4, 5)).astype(np.float32), "v")
    m = _model(
        "g (float[N, 3, H, W] x) => (float[N, 5] y) {"
        "  c = Conv<pads = [1, 1, 1, 1]>(x, w)"
        "  r = Relu(c)"
        "  p = GlobalAveragePool(r)"
        "  f = Flatten(p)"
        "  y = MatMul(f, v)"
        "}",
        opset=opset,
    )
    m.graph.initializer.extend([w, v])
    return m


def _samples(n, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"x": (rng.standard_normal((1, 3, 6, 6)) * (1 + i)).astype(np.float32)}
        for i in range(n)
    ]


@pytest.fixture
def ort_histograms(monkeypatch):
    import onnxsim.calibration as calibration

    monkeypatch.setattr(calibration, "_COUNT_HISTOGRAMS_IN_ORT", True)


def test_collect_pass2_matches_streamed_numpy_histograms(ort_histograms):
    model = _conv_model()
    samples = _samples(5)
    names = ["x", "c", "r", "p"]
    stats = collect_calibration_stats(
        model, samples, tensor_names=names, histograms=True, num_bins=32
    )
    assert stats.histograms.keys() == set(names)

    # reference: the same fixed ranges, counted by numpy over the full tensors
    sess = ort.InferenceSession(
        _with_outputs(model, names).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    ref = {n: _StreamingHistogram(stats.histograms[n].abs_max, 32) for n in names}
    out_names = [o.name for o in sess.get_outputs()]
    for feed in samples:
        for name, value in zip(out_names, sess.run(out_names, feed)):
            if name in ref:
                ref[name].add(value)
    for name in names:
        np.testing.assert_array_equal(
            stats.histograms[name].counts, ref[name].counts, err_msg=name
        )


def _with_outputs(model, names):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    existing = {o.name for o in m.graph.output}
    for n in names:
        if n not in existing:
            m.graph.output.append(onnx.ValueInfoProto(name=n))
    return m


def test_default_pass2_is_numpy_and_matches_ort_counts(monkeypatch):
    import onnxsim.calibration as calibration

    samples = _samples(3)
    names = ["x", "c", "r"]
    default = collect_calibration_stats(
        _conv_model(), samples, tensor_names=names, histograms=True, num_bins=32
    )
    monkeypatch.setattr(calibration, "_COUNT_HISTOGRAMS_IN_ORT", True)
    in_ort = collect_calibration_stats(
        _conv_model(), samples, tensor_names=names, histograms=True, num_bins=32
    )
    for name in names:
        np.testing.assert_array_equal(
            default.histograms[name].counts,
            in_ort.histograms[name].counts,
            err_msg=name,
        )


def test_opset_below_16_falls_back_to_numpy_and_agrees(monkeypatch):
    import onnxsim.calibration as calibration

    monkeypatch.setattr(calibration, "_COUNT_HISTOGRAMS_IN_ORT", True)
    # ScatterElements' reduction="add" needs opset 16; older models use numpy
    samples = _samples(3)
    names = ["x", "c", "r"]
    new = collect_calibration_stats(
        _conv_model(17), samples, tensor_names=names, histograms=True, num_bins=32
    )
    old = collect_calibration_stats(
        _conv_model(13), samples, tensor_names=names, histograms=True, num_bins=32
    )
    for name in names:
        np.testing.assert_array_equal(
            new.histograms[name].counts, old.histograms[name].counts, err_msg=name
        )
