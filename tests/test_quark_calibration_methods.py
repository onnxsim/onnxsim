"""Quark's histogram calibrators (Entropy / Distribution / Percentile /
LayerwisePercentile), its Q/DQ-removal and ``Align*`` options, and the
``ActivationSymmetric`` / ``WeightSymmetric`` / ``QuantizeBias`` options -- the
parts of :mod:`onnxsim.quark_compat` that do not need AMD Quark installed.
Their parity with Quark is checked in ``tests/test_quark_parity.py``."""

import warnings

import numpy as np
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim.calibration import calibrate
from onnxsim.full_qdq import quantize_full_qdq
from onnxsim.quark_calibration import QuarkHistogram, lwp_select

# -- histograms ----------------------------------------------------------------


def test_signed_histogram_grows_by_whole_bins_on_both_sides():
    h = QuarkHistogram(num_bins=8, absolute=False)
    h.add(np.array([-1.0, 0.0, 0.5, 1.0], np.float32))
    assert h.hist.size == 8 and float(h.edges[-1]) == 1.0
    h.add(np.array([-2.5, 2.5], np.float32))
    # stride 0.25: (2.5 - 1) // 0.25 + 1 = 7 extra bins per side
    assert h.hist.size == 8 + 14
    assert float(h.edges[-1]) == pytest.approx(2.75) == -float(h.edges[0])
    assert int(h.hist.sum()) == 6
    assert (float(h.rmin), float(h.rmax)) == (-2.5, 2.5)


def test_signed_histogram_keeps_its_layout_for_smaller_batches():
    h = QuarkHistogram(num_bins=16, absolute=False)
    h.add(np.linspace(-4, 4, 100).astype(np.float32))
    edges = h.edges.copy()
    h.add(np.linspace(-1, 1, 50).astype(np.float32))
    np.testing.assert_array_equal(h.edges, edges)
    assert int(h.hist.sum()) == 150


def test_absolute_histogram_appends_bins_to_the_right():
    h = QuarkHistogram(num_bins=10, absolute=True)
    h.add(np.array([-1.0, 0.5, 1.0], np.float32))
    assert (float(h.edges[0]), float(h.edges[-1])) == (0.5, 1.0)
    h.add(np.array([2.0], np.float32))
    width = 0.05
    assert h.hist.size >= 10 + 19  # (2 - 1) / 0.05 bins appended
    assert float(h.edges[1] - h.edges[0]) == pytest.approx(width)
    assert float(h.edges[-1]) == pytest.approx(2.0, abs=1e-4)
    # a magnitude below the first batch's smallest is dropped, as in Quark
    before = int(h.hist.sum())
    h.add(np.array([0.01], np.float32))
    assert int(h.hist.sum()) == before


def test_percentile_range_is_symmetric_and_clipped_to_the_observed_range():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000).astype(np.float32)
    x[0] = 40.0  # one outlier
    h = QuarkHistogram(2048, absolute=True)
    h.add(x)
    lo, hi = h.percentile_range(99.9)
    assert lo == -hi and 2.5 < hi < 4.5
    lo2, hi2 = h.percentile_range(100.0)
    assert hi2 == pytest.approx(float(x.max()), rel=1e-5)  # clipped to max
    assert lo2 == float(x.min())  # ... and to the observed min, not -max
    with pytest.raises(ValueError):
        h.percentile_range(100.5)


def test_asymmetric_percentile_needs_the_signed_histogram():
    x = np.linspace(-1, 3, 1000).astype(np.float32)
    h = QuarkHistogram(2048, absolute=False)
    h.add(x)
    lo, hi = h.percentile_range(99.0, symmetric=False)
    assert -1.0 <= lo < -0.9 and 2.9 < hi <= 3.0
    with pytest.raises(ValueError):
        h.percentile_range(99.0, symmetric=True)


def test_distribution_range_is_the_histogram_extent_not_clipped():
    h = QuarkHistogram(1024, absolute=False)
    h.add(np.array([0.0, 1.0, 3.0], np.float32))
    lo, hi = h.distribution_range()
    assert (lo, hi) == (-3.0, 3.0)  # symmetric, although the data is >= 0
    h.add(np.array([3.5], np.float32))
    lo, hi = h.distribution_range()
    assert hi == -lo and hi >= 3.5  # grown to whole bins past the new max


def test_entropy_default_bins_can_only_pick_the_whole_range_until_it_grows():
    rng = np.random.default_rng(1)
    h = QuarkHistogram(128, absolute=False)
    h.add(rng.standard_normal(5000).astype(np.float32))
    # 128 bins, 128 quantized bins: a single candidate -- the full range
    assert h.entropy_range() == (float(h.rmin), float(h.rmax))
    h.add((3 * rng.standard_normal(5000)).astype(np.float32))
    assert h.hist.size > 128  # the growth opens a search window
    lo, hi = h.entropy_range()
    assert float(h.rmin) <= lo < 0 < hi <= float(h.rmax)


def test_entropy_clips_a_heavy_tail():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200_000).astype(np.float32)
    x[:20] = 60.0  # a handful of far outliers
    h = QuarkHistogram(2048, absolute=False)
    h.add(x)
    lo, hi = h.entropy_range()
    assert hi < 30.0 and hi > 2.0  # tighter than the max, not absurdly tight
    assert lo >= float(x.min())


def test_layerwise_percentile_picks_per_error_metric():
    rng = np.random.default_rng(3)
    heavy = rng.standard_normal(200_000).astype(np.float32)
    heavy[:30] = 500.0
    h = QuarkHistogram(2048, absolute=True)
    h.add(heavy)
    cands = [99.0, 99.99, 100.0]
    got = lwp_select(h, cands, "int8")
    assert got in [h.percentile_range(p) for p in cands]
    # the outliers make the full range far worse than a clipped one
    assert got[1] < 100.0
    # a uniform tensor wants the whole range
    h2 = QuarkHistogram(2048, absolute=True)
    h2.add(np.linspace(-1, 1, 100_000).astype(np.float32))
    assert lwp_select(h2, cands, "int8")[1] == pytest.approx(1.0, rel=1e-3)
    with pytest.raises(ValueError):
        lwp_select(h, cands, "int8", metric="l1")


# -- calibrate() ----------------------------------------------------------------


def _mlp():
    rng = np.random.default_rng(0)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[4,16] x) => (float[4,6] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }
        """
    )
    for name, shape in (("w1", (16, 24)), ("b1", (24,)), ("w2", (24, 6)), ("b2", (6,))):
        m.graph.initializer.append(
            numpy_helper.from_array(
                (rng.standard_normal(shape) * 0.5).astype(np.float32), name
            )
        )
    return m


def _data(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((4, 16)).astype(np.float32)} for _ in range(n)]


@pytest.mark.parametrize(
    "method",
    [
        "quark_percentile",
        "quark_percentile:99.99",
        "quark_entropy",
        "quark_distribution",
        "quark_layerwise_percentile",
    ],
)
def test_quark_methods_return_ranges_inside_or_around_the_observed(method):
    model, data = _mlp(), _data()
    plain = calibrate(model, data, method="minmax")
    got = calibrate(model, data, method=method, activation_type="int8")
    assert set(got) == set(plain)
    for name, (lo, hi) in got.items():
        olo, ohi = plain[name]
        assert lo <= 0 <= hi or lo <= hi
        if method != "quark_distribution":  # the others never widen the range
            assert olo - 1e-6 <= lo and hi <= ohi + 1e-6
        else:  # whole histogram bins past the observed extent
            assert lo <= olo + 1e-6 and hi >= ohi - 1e-6


def test_quark_methods_validate_their_arguments():
    model, data = _mlp(), _data(2)
    with pytest.raises(ValueError):
        calibrate(model, data, method="quark_percentile:150")
    with pytest.raises(ValueError):
        calibrate(model, data, method="quark_entropy:3")
    with pytest.raises(ValueError):
        calibrate(model, data, method="quark_distribution", quark_num_bins=100)
    with pytest.raises(ValueError):
        calibrate(model, data, method="quark_bogus")


def test_range_symmetric_option_symmetrizes_minmax_and_percentile():
    model, data = _mlp(), _data()
    for method in ("minmax", "minmax_mean"):
        for lo, hi in calibrate(
            model, data, method=method, range_symmetric=True
        ).values():
            assert lo == -hi
    asym = calibrate(model, data, method="quark_percentile", range_symmetric=False)
    sym = calibrate(model, data, method="quark_percentile", range_symmetric=True)
    assert any(asym[k] != sym[k] for k in asym)


def test_layerwise_percentile_candidates_and_metric_are_honoured():
    model = _mlp()
    rng = np.random.default_rng(5)
    data = [
        {"x": (rng.standard_t(1.5, (4, 16)) * 2).astype(np.float32)} for _ in range(8)
    ]
    one = calibrate(
        model,
        data,
        method="quark_layerwise_percentile",
        percentile_candidates=(99.9,),
        activation_type="int8",
    )
    pct = calibrate(model, data, method="quark_percentile:99.9")
    assert one == pct  # a single candidate is just that percentile
    default = calibrate(
        model, data, method="quark_layerwise_percentile", activation_type="int8"
    )
    mse = calibrate(
        model,
        data,
        method="quark_layerwise_percentile",
        lwp_metric="mse",
        activation_type="int8",
    )
    assert set(default) == set(mse)


# -- quark_compat -----------------------------------------------------------------


def _quantize(cfg, data=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            _mlp(), calibration_data_reader=list(data or _data())
        )


def _act_scales(model):
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    return {
        n.input[0]: float(inits[n.input[1]])
        for n in model.graph.node
        if n.op_type == "QuantizeLinear" and n.input[1] in inits
    }


@pytest.mark.parametrize("member", list(qc.CalibMethod))
def test_every_calib_method_quantizes(member):
    cfg = qc.QConfig(
        qc.QLayerConfig(
            activation=qc.Int8Spec(calibration_method=member), weight=qc.Int8Spec()
        )
    )
    out = _quantize(cfg)
    assert sum(n.op_type == "QuantizeLinear" for n in out.graph.node) >= 2


def test_calib_methods_choose_different_scales():
    scales = {}
    for member in (
        qc.CalibMethod.MinMax,
        qc.CalibMethod.Entropy,
        qc.CalibMethod.Distribution,
    ):
        cfg = qc.QConfig(
            qc.QLayerConfig(
                activation=qc.Int8Spec(calibration_method=member),
                weight=qc.Int8Spec(),
            )
        )
        scales[member] = _act_scales(_quantize(cfg))
    assert scales[qc.CalibMethod.MinMax] != scales[qc.CalibMethod.Entropy]
    assert scales[qc.CalibMethod.MinMax] != scales[qc.CalibMethod.Distribution]


def test_calibration_extra_options_reach_the_calibrator():
    def scales(**opts):
        cfg = qc.QConfig(
            qc.QLayerConfig(
                activation=qc.Int8Spec(calibration_method=qc.CalibMethod.Percentile),
                weight=qc.Int8Spec(),
            ),
            **opts,
        )
        return _act_scales(_quantize(cfg))

    assert scales() != scales(Percentile=99.0)
    assert scales() != scales(CalibTensorRangeSymmetric=False)
    assert scales(CalibDataSize=1) != scales()


def test_calibration_args_mapping():
    f = qc._calibration_args
    assert f("entropy", {}) == ("quark_entropy", {})
    assert f("percentile:99.9", {}) == ("quark_percentile:99.9", {})
    assert f("percentile:99.9", {"Percentile": 99.99})[0] == "quark_percentile:99.99"
    assert f("onnxsim:entropy", {}) == ("entropy", {})
    assert f("minmax", {"CalibTensorRangeSymmetric": True})[1] == {
        "range_symmetric": True
    }
    assert f("distribution", {"NumBins": 1024})[1] == {"quark_num_bins": 1024}
    assert f("layerwise_percentile", {"LWPMetric": "mse"})[1] == {"lwp_metric": "mse"}
    with pytest.raises(ValueError):
        f("bogus", {})


# -- Q/DQ placement in full_qdq (Quark mode) --------------------------------------


def _model(body, initializer=(), opset=17):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": {opset}]> g (float[2,8] x) => (float y) {{ {body} }}'
    )
    m.graph.initializer.extend(initializer)
    return m


def _f(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name)


def _gemm_then(consumer, extra_inits=()):
    rng = np.random.default_rng(0)
    return _model(
        f"t = Gemm(x, w, b)  r = {consumer}  y = Gemm(r, w2, b2)",
        [
            _f("w", rng.standard_normal((8, 8))),
            _f("b", rng.standard_normal(8)),
            _f("w2", rng.standard_normal((8, 3))),
            _f("b2", rng.standard_normal(3)),
            *extra_inits,
        ],
    )


def _ranges(model, lo=-2.0, hi=3.0):
    out = {}
    for n in model.graph.node:
        for t in list(n.input) + list(n.output):
            if t and t not in {i.name for i in model.graph.initializer}:
                out[t] = (lo, hi)
    return out


def _qdq(model, **kw):
    kw.setdefault("activation_dtype", "uint8")
    kw.setdefault("symmetric_activations", False)
    ranges = _ranges(model)
    return quantize_full_qdq(model, ranges=ranges, **kw)


def _tensors_with_q(out):
    return {
        n.input[0].removesuffix("/f")
        for n in out.graph.node
        if n.op_type == "QuantizeLinear"
    }


def _ops(out):
    return [n.op_type for n in out.graph.node if n.op_type not in ("DequantizeLinear",)]


@pytest.mark.parametrize(
    "consumer, extra",
    [
        ("Relu(t)", ()),
        ("LeakyRelu<alpha=0.1>(t)", ()),
        ("PRelu(t, sl)", [_f("sl", [0.25])]),
        ("Clip(t, lo, hi)", [_f("lo", 0.0), _f("hi", 6.0)]),
        ("Clip(t, lo, hi)", [_f("lo", 0.0), _f("hi", 1.0)]),
    ],
)
def test_listed_consumers_leave_the_producer_output_float(consumer, extra):
    model = _gemm_then(consumer, extra)
    kept = _tensors_with_q(_qdq(model, remove_qdq_after=[], fold_activation=False))
    assert "t" in kept
    dropped = _tensors_with_q(
        _qdq(
            model,
            remove_qdq_after=["Relu", "LeakyRelu", "PRelu", "Clip"],
            fold_activation=False,
        )
    )
    assert "t" not in dropped and "r" in dropped


def test_clip_needs_relu_like_constant_bounds():
    model = _gemm_then("Clip(t, lo, hi)", [_f("lo", -1.0), _f("hi", 1.0)])
    out = _qdq(model, remove_qdq_after=["Clip"], fold_activation=False)
    assert "t" in _tensors_with_q(out)


def test_only_listed_op_types_count():
    model = _gemm_then("LeakyRelu<alpha=0.1>(t)")
    out = _qdq(model, remove_qdq_after=["Relu", "PRelu"], fold_activation=False)
    assert "t" in _tensors_with_q(out)
    out = _qdq(model, remove_qdq_after=["LeakyRelu"], fold_activation=False)
    assert "t" not in _tensors_with_q(out)


def test_producer_must_be_listed_and_the_consumer_unique():
    model = _model(
        "t = Gemm(x, w, b)  r = Relu(t)  y = Add(r, t)",
        [_f("w", np.eye(8)), _f("b", np.zeros(8))],
    )
    out = _qdq(model, remove_qdq_after=["Relu"], fold_activation=False)
    assert "t" in _tensors_with_q(out)  # t has two consumers: kept
    model = _model("t = Mul(x, k)  r = Relu(t)  y = Relu(r)", [_f("k", 2.0)])
    out = _qdq(model, remove_qdq_after=["Relu"], fold_activation=False)
    assert "t" in _tensors_with_q(out)  # Mul is not a listed producer
    out = _qdq(
        model,
        remove_qdq_after=["Relu"],
        remove_qdq_producers=["Mul"],
        fold_activation=False,
    )
    assert "t" not in _tensors_with_q(out)


def test_fold_activation_removes_relu_and_clip_nodes():
    model = _gemm_then("Relu(t)")
    out = _qdq(model, remove_qdq_after=["Relu"], fold_activation=True)
    assert "Relu" not in _ops(out)
    out = _qdq(model, remove_qdq_after=["Relu"], fold_activation=False)
    assert "Relu" in _ops(out)
    clip = _gemm_then("Clip(t, lo, hi)", [_f("lo", -1.0), _f("hi", 1.0)])
    assert "Clip" not in _ops(_qdq(clip, remove_qdq_after=[], fold_activation=True))
    # default: only for non-symmetric activations
    out = _qdq(model, remove_qdq_after=["Relu"], symmetric_activations=True)
    assert "Relu" in _ops(out)


def test_adjust_activation_ranges_gives_the_input_the_output_range():
    model = _gemm_then("Relu(t)")
    ranges = {"x": (-1.0, 1.0), "t": (-9.0, 9.0), "r": (0.0, 2.0), "y": (-1.0, 1.0)}
    out = quantize_full_qdq(
        model,
        ranges=ranges,
        activation_dtype="int8",
        symmetric_activations=True,
        remove_qdq_after=[],
        adjust_activation_ranges=True,
    )
    inits = {i.name: numpy_helper.to_array(i) for i in out.graph.initializer}
    scale = {
        n.input[0].removesuffix("/f"): float(inits[n.input[1]])
        for n in out.graph.node
        if n.op_type == "QuantizeLinear"
    }
    assert scale["t"] == scale["r"] == pytest.approx(2.0 / 127)


def test_quantize_bias_false_leaves_the_bias_float():
    model = _gemm_then("Relu(t)")
    with_bias = _qdq(model)
    without = _qdq(model, quantize_bias=False)
    ints = lambda m: {  # noqa: E731
        numpy_helper.to_array(i).dtype.name for i in m.graph.initializer
    }
    assert "int32" in ints(with_bias) and "int32" not in ints(without)


def test_weight_symmetric_false_and_uint8_weights():
    model = _gemm_then("Relu(t)")

    def weights(**kw):
        out = _qdq(model, per_channel=False, **kw)
        inits = {i.name: numpy_helper.to_array(i) for i in out.graph.initializer}
        return [
            (inits[n.input[0]], inits[n.input[2]])
            for n in out.graph.node
            if n.op_type == "DequantizeLinear"
            and n.input[0] in inits
            and inits[n.input[0]].ndim == 2
        ]

    (q, zp), *_ = weights()
    assert q.dtype == np.int8 and int(zp) == 0
    (q, zp), *_ = weights(weight_symmetric=False)
    assert q.dtype == np.int8 and int(zp) != 0
    (q, zp), *_ = weights(weight_dtype="uint8", weight_symmetric=False)
    assert q.dtype == np.uint8 and int(q.min()) == 0 and int(q.max()) == 255
    (q, zp), *_ = weights(weight_dtype="uint8", weight_symmetric=True)
    assert q.dtype == np.uint8 and int(zp) == 128


def test_unsigned_symmetric_activations_span_the_whole_code_range():
    model = _gemm_then("Relu(t)")
    out = _qdq(
        model,
        symmetric_activations=True,
        remove_qdq_after=[],
        fold_activation=False,
    )
    inits = {i.name: numpy_helper.to_array(i) for i in out.graph.initializer}
    q = next(n for n in out.graph.node if n.op_type == "QuantizeLinear")
    # absmax 3, scale 2 * 3 / 255 (not 3 / 127), zero point 128
    assert float(inits[q.input[1]]) == pytest.approx(6.0 / 255, rel=1e-6)
    assert int(inits[q.input[2]]) == 128


def _concat_model():
    return _model("a = Sigmoid(x)  b = Tanh(x)  y = Concat<axis=1>(a, b)")


def test_align_concat_copies_the_output_parameters_to_the_inputs():
    model = _concat_model()
    ranges = {"x": (-1.0, 1.0), "a": (0.0, 1.0), "b": (-1.0, 1.0), "y": (-1.0, 1.0)}
    kw = dict(
        ranges=ranges,
        activation_dtype="int8",
        symmetric_activations=True,
        remove_qdq_after=[],
    )

    def scales(out):
        inits = {i.name: numpy_helper.to_array(i) for i in out.graph.initializer}
        return {
            n.input[0].removesuffix("/f"): float(inits[n.input[1]])
            for n in out.graph.node
            if n.op_type == "QuantizeLinear"
        }

    plain = scales(quantize_full_qdq(model, **kw))
    aligned = scales(quantize_full_qdq(model, align_ops=["Concat"], **kw))
    assert plain["a"] == plain["b"] == plain["y"]  # absmax 1 everywhere
    ranges["a"] = (0.0, 0.5)
    plain = scales(quantize_full_qdq(model, **{**kw, "ranges": ranges}))
    aligned = scales(
        quantize_full_qdq(model, align_ops=["Concat"], **{**kw, "ranges": ranges})
    )
    assert plain["a"] != plain["y"]
    assert aligned["a"] == aligned["b"] == aligned["y"]


def test_pool_and_slice_alignment_copy_input_to_output():
    model = _model(
        "x1 = Mul(x, k)  s = Slice(x1, st, en, ax)  y = Sigmoid(s)",
        [
            _f("k", 3.0),
            numpy_helper.from_array(np.array([0], np.int64), "st"),
            numpy_helper.from_array(np.array([4], np.int64), "en"),
            numpy_helper.from_array(np.array([1], np.int64), "ax"),
        ],
    )
    ranges = {"x": (-1, 1), "x1": (-3, 3), "s": (-1, 2), "y": (0, 1)}
    kw = dict(
        ranges=ranges,
        activation_dtype="int8",
        symmetric_activations=True,
        remove_qdq_after=[],
        unshared_ops=["Slice"],
    )

    def scale_of(out, name):
        inits = {i.name: numpy_helper.to_array(i) for i in out.graph.initializer}
        q = next(
            n
            for n in out.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0].removesuffix("/f") == name
        )
        return float(inits[q.input[1]])

    own = quantize_full_qdq(model, **kw)
    aligned = quantize_full_qdq(model, align_ops=["Slice"], **kw)
    assert scale_of(own, "s") == pytest.approx(2 / 127)  # calibrated on its own
    assert scale_of(aligned, "s") == scale_of(aligned, "x1") == pytest.approx(3 / 127)
    shared = quantize_full_qdq(
        model, **{**kw, "unshared_ops": []}
    )  # the default NPU behaviour: Slice reuses its input's parameters
    assert scale_of(shared, "s") == pytest.approx(3 / 127)


def test_prelu_slope_is_quantized_when_asked():
    model = _gemm_then("PRelu(t, sl)", [_f("sl", [0.25])])
    kw = dict(
        remove_qdq_after=["PRelu"],
        fold_activation=False,
        int8_constants=True,
    )

    def has_slope_dq(out):
        return any(
            n.op_type == "DequantizeLinear" and "sl" in n.input[0]
            for n in out.graph.node
        )

    assert not has_slope_dq(_qdq(model, **kw))
    assert has_slope_dq(_qdq(model, quantize_prelu_slope=True, **kw))


# -- quark_compat option plumbing -----------------------------------------------------


@pytest.mark.parametrize(
    "preset, extra, relu_stays",
    [
        ("U8S8_AAWS", {}, False),  # plain quantizer, asymmetric: Relu folded
        ("U8S8_AAWS", {"ActivationSymmetric": True}, True),
        ("A8W8", {"ActivationSymmetric": False}, False),  # extended, FoldRelu=True
        ("A8W8", {"ActivationSymmetric": False, "FoldRelu": False}, True),
        ("U16S8_AAWS", {}, True),  # extended without FoldRelu
        ("U16S8_AAWS", {"FoldRelu": True}, False),
    ],
)
def test_relu_folding_rule_per_quantizer_class(preset, extra, relu_stays):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra)
    out = _quantize(cfg)
    assert ("Relu" in [n.op_type for n in out.graph.node]) is relu_stays


def test_remove_qdq_extra_options_drive_the_removal():
    def has_q(extra):
        cfg = qc.QConfig.get_default_config("A8W8")
        cfg.extra_options.update(extra)
        return "h0" in _tensors_with_q(_quantize(cfg))

    assert not has_q({})
    assert has_q({"RemoveQDQConvRelu": False})
    assert not has_q({"RemoveQDQConvRelu": True, "RemoveQDQConvClip": False})


def test_presets_carry_quarks_extra_options():
    assert qc.QConfig.get_default_config("A8W8").extra_options == {
        "ActivationSymmetric": True,
        "FoldRelu": True,
        "AlignConcat": True,
        "AlignSlice": False,
        # the input-marking default Quark's legacy presets carry
        "ForceQuantizeNoInputCheck": True,
    }
    a16 = qc.QConfig.get_default_config("A16W8")
    assert a16.extra_options["AlignEltwiseQuantType"] is True
    assert a16.extra_options["AlignConcat"] is True
    assert qc.QConfig.get_default_config("S16S8_ASWS").extra_options == {
        "ActivationSymmetric": True,
        "ForceQuantizeNoInputCheck": True,
    }
