"""Streaming histogram calibration (entropy/mse/percentile) and full-graph QDQ.

``onnxsim.calibrate``'s histogram methods used to keep every observed
activation value until the end (memory = #images x activations -- ORT's own
percentile calibrator OOMs at 16 GB on YOLO11n at 640). They now accumulate
a fixed-size histogram per tensor over a second pass, so memory is #tensors x
bins; these tests pin that the streamed thresholds match the exact
(all-values) searches, that ``"percentile"`` finds the right quantiles, that
``"minmax"`` stays exact, and that memory does not grow with the data.

``quantize_static(full_graph=True)`` is the whole-graph QDQ scheme an
integer NPU (the Qualcomm HTP through ORT's QNN EP) needs.
"""

import tracemalloc

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim.calibration import _entropy_threshold, _mse_threshold
from onnxsim.qdq_full_graph import list_full_graph_activations


def _model(body, initializer=(), opset=17, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _matmul_model(k=64, n=32):
    # list_quantizable_activations picks up x (the MatMul's activation input)
    w = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((k, n)).astype(np.float32), "w"
    )
    return _model(
        f"g (float[1, {k}] x) => (float[1, {n}] y) {{ y = MatMul(x, w) }}",
        initializer=[w],
    )


def _heavy_tailed_batches(num_batches, k=64, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_batches):
        x = rng.standard_normal((1, k)).astype(np.float32)
        # ~0.08% outliers: rare enough for entropy's 99.9% coverage floor
        if rng.random() < 0.05:
            x[0, rng.integers(k)] = rng.choice([-1.0, 1.0]) * 40.0
        batches.append({"x": x})
    return batches


def test_minmax_is_exact():
    batches = _heavy_tailed_batches(50)
    ranges = onnxsim.calibrate(_matmul_model(), batches, method="minmax")
    allx = np.concatenate([b["x"].ravel() for b in batches])
    assert ranges["x"] == (float(allx.min()), float(allx.max()))


@pytest.mark.parametrize("pct", [99.0, 99.9, 99.99])
def test_percentile_matches_numpy_quantiles(pct):
    rng = np.random.default_rng(1)
    batches = [
        {"x": rng.standard_normal((1, 64)).astype(np.float32) * 2.0 + 0.5}
        for _ in range(400)
    ]
    allx = np.concatenate([b["x"].ravel() for b in batches])
    lo, hi = onnxsim.calibrate(
        _matmul_model(), batches, method="percentile", percentile=pct
    )["x"]
    # by rank, not value: in a sparse tail the in-bin interpolation and
    # numpy's order-statistic interpolation can sit a couple of bins apart
    # while clipping the same number of samples, give or take a few
    expected = (100 - pct) / 100 * allx.size
    for clipped in (np.sum(allx < lo), np.sum(allx > hi)):
        assert abs(clipped - expected) <= 0.2 * expected + 2, (clipped, expected)
    assert lo < np.percentile(allx, 50) < hi


def test_percentile_keeps_one_sided_range():
    # post-ReLU-like: 70% zeros, the rest positive -- lower quantile stays 0
    rng = np.random.default_rng(2)
    batches = []
    for _ in range(200):
        x = np.abs(rng.standard_normal((1, 64))).astype(np.float32)
        x[x < 1.0] = 0.0
        batches.append({"x": x})
    lo, hi = onnxsim.calibrate(
        _matmul_model(), batches, method="percentile", percentile=99.9
    )["x"]
    # the zeros sit in the [0, bin) bin; in-bin interpolation may land a
    # hair above 0 (the quantizer widens every range to include 0 anyway)
    assert 0.0 <= lo < 1e-3
    assert 0.0 < hi < max(float(b["x"].max()) for b in batches)


def test_streaming_entropy_matches_exact_search():
    batches = _heavy_tailed_batches(400)
    allx = np.concatenate([b["x"].ravel() for b in batches])
    exact = _entropy_threshold(allx, num_bins=2048, num_quantized_bins=128)
    lo, hi = onnxsim.calibrate(_matmul_model(), batches, method="entropy")["x"]
    # the streamed histogram's |x| bins are the same width as the exact
    # search's (A / 2048); allow a couple of bins for the coverage floor
    bin_width = float(np.abs(allx).max()) / 2048
    assert max(-lo, hi) == pytest.approx(exact, abs=3 * bin_width)
    assert exact < float(np.abs(allx).max())  # it did clip


def test_streaming_mse_matches_exact_search():
    batches = _heavy_tailed_batches(400)
    allx = np.concatenate([b["x"].ravel() for b in batches])
    exact = _mse_threshold(allx, num_candidates=100, min_coverage=0.5)
    lo, hi = onnxsim.calibrate(_matmul_model(), batches, method="mse")["x"]
    # candidates are 1/99 of the search span apart; the histogram version
    # may land on a neighbouring one
    step = (float(np.abs(allx).max()) - np.percentile(np.abs(allx), 50)) / 99
    # a Python float: pytest 9.2 refuses a numpy float32 tolerance
    assert max(-lo, hi) == pytest.approx(exact, abs=float(2 * step))


@pytest.mark.parametrize("method", ["entropy", "percentile"])
def test_histogram_memory_does_not_grow_with_data(method):
    # one 256 KiB activation per batch: keeping every value (the old
    # behaviour) would retain 4x more at 32 batches than at 8
    k = 65536
    model = _matmul_model(k=k, n=1)
    rng = np.random.default_rng(3)
    data = [{"x": rng.standard_normal((1, k)).astype(np.float32)} for _ in range(32)]

    def peak(batches):
        tracemalloc.start()
        onnxsim.calibrate(model, batches, method=method)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return p

    small, large = peak(data[:8]), peak(data)
    assert large < small * 1.5, (small, large)
    assert large - small < k * 4  # < one activation's worth for 24 more


def test_one_shot_iterator_is_materialized():
    batches = _heavy_tailed_batches(20)
    a = onnxsim.calibrate(_matmul_model(), iter(batches), method="percentile")
    b = onnxsim.calibrate(_matmul_model(), batches, method="percentile")
    assert a == b


def _full_graph_model(seed=0):
    rng = np.random.default_rng(seed)
    w = numpy_helper.from_array(
        rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.3, "w"
    )
    b = numpy_helper.from_array(rng.standard_normal(8).astype(np.float32) * 0.1, "b")
    return _model(
        """
        g (float[1, 3, 16, 16] x) => (float[1, 16, 16, 16] y)
          <float[1] c = {0.5}, int64[4] flat_shape = {1, 8, 256, 1}, int64[4] shape = {1, 8, 16, 16}> {
            conv = Conv <pads = [1, 1, 1, 1]> (x, w, b)
            sig = Sigmoid(conv)
            act = Mul(conv, sig)
            scaled = Add(act, c)
            flat = Reshape(scaled, flat_shape)
            back = Reshape(flat, shape)
            y = Concat <axis = 1> (back, sig)
        }
        """,
        initializer=[w, b],
    )


def _calib(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return [{"x": rng.random((1, 3, 16, 16)).astype(np.float32)} for _ in range(n)]


def test_full_graph_qdq_structure():
    model = _full_graph_model()
    q = onnxsim.quantize_static(model, _calib(), full_graph=True, method="percentile")
    onnx.checker.check_model(q)
    g = q.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}

    conv = next(n for n in g.node if n.op_type == "Conv")
    x_dq, w_dq, b_dq = (producer[i] for i in conv.input)
    assert all(n.op_type == "DequantizeLinear" for n in (x_dq, w_dq, b_dq))
    w_q, w_s = inits[w_dq.input[0]], inits[w_dq.input[1]]
    assert w_q.dtype == np.int8 and w_s.shape == (8,)
    assert dict((a.name, a.i) for a in w_dq.attribute)["axis"] == 0
    assert not inits[w_dq.input[2]].any()
    b_q, b_s = inits[b_dq.input[0]], inits[b_dq.input[1]]
    assert b_q.dtype == np.int32
    np.testing.assert_allclose(b_s, inits[x_dq.input[1]] * w_s, rtol=1e-6)
    # graph input keeps its exact range under percentile: [0, max] -> zp 0
    assert int(inits[x_dq.input[2]]) == 0

    # every non-Q/DQ op reads its data inputs from a DequantizeLinear and
    # writes through a QuantizeLinear: complete DQ -> op -> Q node units
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        data_inputs = n.input[:1] if n.op_type == "Reshape" else n.input
        assert all(producer[i].op_type == "DequantizeLinear" for i in data_inputs), (
            n.name
        )
        for o in n.output:
            assert [c.op_type for c in consumers[o]] == ["QuantizeLinear"], n.name
    # the Add's constant operand is quantized as uint8 of its own range
    add = next(n for n in g.node if n.op_type == "Add")
    c_dq = producer[add.input[1]]
    assert inits[c_dq.input[0]].dtype == np.uint8
    # Reshape's shape input stays int64
    reshape = next(n for n in g.node if n.op_type == "Reshape")
    assert inits[reshape.input[1]].dtype == np.int64
    # value-preserving ops reuse their input's scale/zero point: no requantize
    for n in g.node:
        if n.op_type == "Reshape":
            dq_in = producer[n.input[0]]
            (q_out,) = consumers[n.output[0]]
            assert inits[q_out.input[1]] == inits[dq_in.input[1]]
            assert inits[q_out.input[2]] == inits[dq_in.input[2]]


def test_full_graph_exclusion_and_accuracy():
    model = _full_graph_model()
    concat_name = "final_concat"
    for n in model.graph.node:
        if n.op_type == "Concat":
            n.name = concat_name
    q = onnxsim.quantize_static(
        model, _calib(), full_graph=True, nodes_to_exclude=[concat_name]
    )
    g = q.graph
    producer = {o: n for n in g.node for o in n.output}
    # the excluded Concat is float: its output is the graph output directly
    assert producer["y"].op_type == "Concat"
    # ... reading its inputs through the quantized producers' DQs
    assert all(producer[i].op_type == "DequantizeLinear" for i in producer["y"].input)

    x = _calib(1, seed=9)[0]
    ref = ort.InferenceSession(model.SerializeToString()).run(None, x)[0]
    # No graph optimizations: ORT would otherwise fuse the QDQ Conv into a
    # u8s8 integer kernel, which saturates on x86 CPUs without VNNI (11%
    # error on the Windows CI runner) -- this checks the quantization
    # parameters, not ORT's kernels.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    got = ort.InferenceSession(q.SerializeToString(), so).run(None, x)[0]
    assert np.abs(ref - got).max() < 0.05 * np.abs(ref).max()


def test_full_graph_per_tensor_and_op_type_exclusion():
    model = _full_graph_model()
    q = onnxsim.quantize_static(
        model,
        _calib(),
        full_graph=True,
        per_channel=False,
        op_types_to_exclude=["Sigmoid"],
    )
    g = q.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    conv = next(n for n in g.node if n.op_type == "Conv")
    assert inits[producer[conv.input[1]].input[1]].shape == ()
    sig = next(n for n in g.node if n.op_type == "Sigmoid")
    # the Sigmoid reads the conv output through its DQ and is itself float;
    # its output is still quantized, since the (quantized) Mul reads it
    assert producer[sig.input[0]].op_type == "DequantizeLinear"
    assert "sig" in list_full_graph_activations(model, op_types_to_exclude=["Sigmoid"])


def test_full_graph_only_options_rejected_on_default_scheme():
    with pytest.raises(ValueError, match="full_graph"):
        onnxsim.quantize_static(
            _matmul_model(), _heavy_tailed_batches(4), per_channel=False
        )
    with pytest.raises(ValueError, match="percentile"):
        onnxsim.calibrate(
            _matmul_model(),
            _heavy_tailed_batches(4),
            method="percentile",
            percentile=10,
        )
