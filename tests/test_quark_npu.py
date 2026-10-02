"""Quark's XINT8 NPU rewrites (``onnxsim.quark_npu`` / ``onnxsim.quark_convert``)
and its power-of-two BiasCorrection, without Quark. The parity of every rewrite
against the real package is ``tests/test_quark_xint8_parity.py``."""

import warnings

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim import calibration as cal
from onnxsim import quark_bias_correction as qbc
from onnxsim import quark_compat as qc
from onnxsim import quark_convert as conv
from onnxsim import quark_npu as npu

SHAPE = (1, 3, 12, 12)


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _model(body, initializer=(), opset=17, ir_version=9, inp=f"float{list(SHAPE)} x"):
    model = parser.parse_model(
        f"""<ir_version: {ir_version}, opset_import: ["": {opset}]>
        g ({inp}) => (float y) {{ {body} }}"""
    )
    model.graph.initializer.extend(initializer)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return onnx.shape_inference.infer_shapes(model)


def _convs(rng=None):
    rng = rng or np.random.default_rng(0)
    return [
        numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1"),
        numpy_helper.from_array(_w(rng, 8), "b1"),
        numpy_helper.from_array(_w(rng, 2, 8, 1, 1), "w2"),
        numpy_helper.from_array(_w(rng, 2), "b2"),
        numpy_helper.from_array(_w(rng, 8, 8, 1, 1), "w5"),
        numpy_helper.from_array(_w(rng, 8), "b5"),
    ]


def _data(shape=SHAPE, n=4, seed=3):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


class _Reader:
    def __init__(self, data):
        self._it = iter(data)

    def get_next(self):
        return next(self._it, None)


def _xint8(model, extra=None, algos=(), preset="XINT8", data=None):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    cfg.algo_config = list(cfg.algo_config) + list(algos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(data or _data())
        )


def _ops(model):
    return [
        n.op_type
        for n in model.graph.node
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear", "Constant")
    ]


def _consts(model):
    return {
        n.output[0]: float(numpy_helper.to_array(n.attribute[0].t))
        for n in model.graph.node
        if n.op_type == "Constant"
    }


def _run(model, x):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


# -- the DPU constants ----------------------------------------------------------------


def test_positions_and_scales():
    assert npu.scale2pos(0.125) == 3 and npu.pos2scale(3) == 0.125
    assert npu.scale2pos(4.0) == -2 and npu.pos2scale(-2) == 4.0
    assert npu.scale2pos(2.0**-200) == 127  # clamped, no infinity


@pytest.mark.parametrize(
    "kh, kw, expected",
    [
        (3, 3, 9 * 7 / 64),
        (5, 5, 25 * 10 / 256),
        (6, 6, 36 * 7 / 256),
        (7, 7, 49 * 21 / 1024),
        (14, 14, 196 * 21 / 4096),
        (2, 2, 1.0),  # 1/4 is exact
        (4, 4, 1.0),
        (256, 3, 1.0),  # beyond the NPU's window: left alone
    ],
)
def test_average_pool_dpu_scale(kh, kw, expected):
    assert npu.avg_pool_dpu_scale(kh, kw) == expected


def test_reciprocal_scale_is_the_closest_fixed_point_reciprocal():
    for rec in (6, 10, 12, 25, 49, 100):
        s = npu.reciprocal_dpu_scale(rec)
        # k / 2**n is within 1% of 1 / rec for these
        assert abs(s - 1.0) < 0.05, (rec, s)
    assert npu.dpu_leaky_relu_alpha(0.1) == 26 / 256
    assert npu.dpu_leaky_relu_alpha(0.01) == 3 / 256


# -- position refinement on hand-built Q/DQ graphs ---------------------------------------


def _qdq(body, positions, dims=None):
    """A Q/DQ graph from ``body`` (text); ``positions`` maps each ``s_<name>``
    scale initializer to a position (scale ``2**-pos``)."""
    model = parser.parse_model(
        f"""<ir_version: 9, opset_import: ["": 17]>
        g (float[1,4,4,4] x) => (float[1,4,4,4] y) {{ {body} }}"""
    )
    for name, pos in positions.items():
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(2.0**-pos, np.float32), name)
        )
    model.graph.initializer.append(numpy_helper.from_array(np.array(0, np.int8), "z"))
    for name in ("w", "b"):
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.ones((4, 4, 1, 1) if name == "w" else 4, np.int8), name
            )
        )
    return model


def _pos_of(model, name):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    return npu.scale2pos(float(inits[name]))


_CONV_QDQ = """
    xq = QuantizeLinear(x, s_x, z)
    xd = DequantizeLinear(xq, s_x, z)
    wd = DequantizeLinear(w, s_w, z)
    bd = DequantizeLinear(b, s_b, z)
    c = Conv(xd, wd, bd)
    cq = QuantizeLinear(c, s_c, z)
    y = DequantizeLinear(cq, s_c, z)
"""


def _refine(model, **flags):
    return npu.adjust_quantize_info(model, lambda n: n.op_type == "Relu", **flags)


def test_shift_cut_moves_the_weight_position_into_0_16():
    # wpos + ipos - opos = 17 + 5 - 3 = 19 > 16: wpos -> 16 + 3 - 5 = 14
    m = _refine(_qdq(_CONV_QDQ, dict(s_x=5, s_w=17, s_b=22, s_c=3)))
    assert _pos_of(m, "s_w") == 14
    # ... and below 0: wpos -> 0 + opos - ipos
    m = _refine(_qdq(_CONV_QDQ, dict(s_x=0, s_w=0, s_b=0, s_c=4)))
    assert _pos_of(m, "s_w") == 4
    # in range: untouched
    m = _refine(_qdq(_CONV_QDQ, dict(s_x=5, s_w=8, s_b=13, s_c=3)))
    assert (_pos_of(m, "s_w"), _pos_of(m, "s_c")) == (8, 3)


def test_shift_bias_moves_the_bias_position():
    # shift_cut = 6 + 4 - 4 = 6, so shift_bias must be in [min(0, 6 - 16), 15]
    # = [-10, 15]; bpos 30 gives -20 -> bpos = wpos + ipos + 10 = 20
    m = _refine(_qdq(_CONV_QDQ, dict(s_x=4, s_w=6, s_b=30, s_c=4)))
    assert _pos_of(m, "s_b") == 20
    # bpos 0 gives 10: fine
    m = _refine(_qdq(_CONV_QDQ, dict(s_x=4, s_w=6, s_b=-8, s_c=4)))
    assert _pos_of(m, "s_b") == -5  # shift_bias 18 > 15 -> bpos = 10 - 15


def test_flags_switch_every_pass_off():
    pos = dict(s_x=5, s_w=17, s_b=22, s_c=3)
    assert _pos_of(_refine(_qdq(_CONV_QDQ, pos), AdjustShiftCut=False), "s_w") == 17
    assert _pos_of(_refine(_qdq(_CONV_QDQ, pos), AdjustShiftBias=False), "s_b") == 22
    assert _pos_of(_refine(_qdq(_CONV_QDQ, pos), max_loop_num=0), "s_w") == 17


_CONCAT_QDQ = """
    aq = QuantizeLinear(x, s_a, z)
    ad = DequantizeLinear(aq, s_a, z)
    bq = QuantizeLinear(x, s_b, z)
    bd = DequantizeLinear(bq, s_b, z)
    k = Concat<axis=1>(ad, bd)
    kq = QuantizeLinear(k, s_k, z)
    y = DequantizeLinear(kq, s_k, z)
"""


def test_align_concat_takes_the_smallest_position():
    m = _refine(_qdq(_CONCAT_QDQ, dict(s_a=5, s_b=3, s_k=4)))
    assert [_pos_of(m, n) for n in ("s_a", "s_b", "s_k")] == [3, 3, 3]
    m = _refine(_qdq(_CONCAT_QDQ, dict(s_a=5, s_b=3, s_k=4)), AlignConcat=False)
    assert [_pos_of(m, n) for n in ("s_a", "s_b", "s_k")] == [5, 3, 4]


_POOL_QDQ = """
    xq = QuantizeLinear(x, s_x, z)
    xd = DequantizeLinear(xq, s_x, z)
    p = MaxPool<kernel_shape=[2,2], strides=[1,1], pads=[0,0,1,1]>(xd)
    pq = QuantizeLinear(p, s_p, z)
    y = DequantizeLinear(pq, s_p, z)
"""


def test_align_pool_lowers_the_larger_position_of_input_and_output():
    m = _refine(_qdq(_POOL_QDQ, dict(s_x=3, s_p=5)))
    assert (_pos_of(m, "s_x"), _pos_of(m, "s_p")) == (3, 3)
    m = _refine(_qdq(_POOL_QDQ, dict(s_x=6, s_p=2)))
    assert (_pos_of(m, "s_x"), _pos_of(m, "s_p")) == (2, 2)


_ADD_QDQ = """
    aq = QuantizeLinear(x, s_a, z)
    ad = DequantizeLinear(aq, s_a, z)
    bq = QuantizeLinear(x, s_b, z)
    bd = DequantizeLinear(bq, s_b, z)
    k = Add(ad, bd)
    kq = QuantizeLinear(k, s_k, z)
    y = DequantizeLinear(kq, s_k, z)
"""


def test_add_shift_read_and_write():
    # inputs 12 and 3 apart by 9 > 7: the larger one drops to 3 + 7
    m = _refine(_qdq(_ADD_QDQ, dict(s_a=12, s_b=3, s_k=3)))
    assert (_pos_of(m, "s_a"), _pos_of(m, "s_b")) == (10, 3)
    # min(ipos) - opos = 3 - 30 = -27 < -7: the output position -> 3 + 7 = 10
    m = _refine(_qdq(_ADD_QDQ, dict(s_a=4, s_b=3, s_k=30)))
    assert _pos_of(m, "s_k") == 10
    # > 25 the other way
    m = _refine(_qdq(_ADD_QDQ, dict(s_a=40, s_b=36, s_k=-2)))
    assert _pos_of(m, "s_a") == 43 or _pos_of(m, "s_k") <= 36 + 7


def test_hard_sigmoid_positions_are_limited():
    body = """
        xq = QuantizeLinear(x, s_x, z)
        xd = DequantizeLinear(xq, s_x, z)
        h = HardSigmoid<alpha=0.16666667, beta=0.5>(xd)
        hq = QuantizeLinear(h, s_h, z)
        y = DequantizeLinear(hq, s_h, z)
    """
    # input position clamped to [0, 15], output to >= 7
    m = _refine(_qdq(body, dict(s_x=-3, s_h=2)))
    assert (_pos_of(m, "s_x"), _pos_of(m, "s_h")) == (0, 7)
    m = _refine(_qdq(body, dict(s_x=20, s_h=9)))
    assert _pos_of(m, "s_x") == 15


# -- through the preset ---------------------------------------------------------------------


def _avg_model():
    return _model(
        "c0 = Conv(x, w1, b1)\n p = AveragePool<kernel_shape=[3,3]>(c0)\n"
        " y = Conv(p, w2, b2)",
        _convs(),
    )


def test_average_pool_gets_the_dpu_mul_before_its_quantizer():
    q = _xint8(_avg_model())
    pool = next(n for n in q.graph.node if n.op_type == "AveragePool")
    mul = next(n for n in q.graph.node if n.op_type == "Mul")
    assert mul.input[0] == pool.output[0]
    assert _consts(q)[mul.input[1]] == 9 * 7 / 64
    # pool input and output share a grid (AlignPool)
    inits = {t.name: numpy_helper.to_array(t) for t in q.graph.initializer}
    dq_in = next(n for n in q.graph.node if n.output and n.output[0] == pool.input[0])
    q_out = next(
        n
        for n in q.graph.node
        if n.op_type == "QuantizeLinear" and n.input[0] == mul.output[0]
    )
    assert inits[dq_in.input[1]] == inits[q_out.input[1]]


@pytest.mark.parametrize(
    "extra",
    [
        {"SimulateDPU": False},
        {"ConvertAvgPoolToDPUVersion": False},
        {"EnableNPUCnn": False},
    ],
)
def test_the_dpu_mul_can_be_switched_off(extra):
    assert "Mul" not in _ops(_xint8(_avg_model(), extra))


def test_other_presets_do_not_get_the_npu_rewrites():
    assert "Mul" not in _ops(_xint8(_avg_model(), preset="U8S8_AAWS"))


def test_sigmoid_becomes_hard_sigmoid_and_leaky_relu_alpha_is_rounded():
    sig = _model(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n y = Conv(s, w2, b2)", _convs()
    )
    q = _xint8(sig)
    ops = _ops(q)
    assert "Sigmoid" not in ops and ops.count("HardSigmoid") == 1
    hs = next(n for n in q.graph.node if n.op_type == "HardSigmoid")
    alpha = next(a.f for a in hs.attribute if a.name == "alpha")
    assert abs(alpha - 1 / 6) < 1e-6
    assert list(_consts(q).values()) == [(2731.0 / 16384.0) / (1.0 / 6.0)]
    kept = _xint8(sig, {"ConvertSigmoidToHardSigmoid": False})
    assert "Sigmoid" in _ops(kept) and "HardSigmoid" not in _ops(kept)

    leaky = _model(
        "c0 = Conv(x, w1, b1)\n l = LeakyRelu<alpha=0.1>(c0)\n y = Conv(l, w2, b2)",
        _convs(),
    )
    q = _xint8(leaky)
    node = next(n for n in q.graph.node if n.op_type == "LeakyRelu")
    assert next(a.f for a in node.attribute if a.name == "alpha") == 26 / 256


def test_hard_swish_is_inlined_like_onnx_runtime_does():
    body = "c0 = Conv(x, w1, b1)\n h = HardSwish(c0)\n y = Conv(h, w2, b2)"
    q = _xint8(_model(body, _convs()))
    assert "HardSwish" not in _ops(q)
    assert _ops(q).count("HardSigmoid") == 1 and _ops(q).count("Mul") == 2
    kept = _xint8(_model(body, _convs()), {"OptimizeModel": False})
    assert "HardSwish" in _ops(kept)


def test_quantized_xint8_model_still_computes_the_float_function():
    model = _model(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n p = AveragePool<kernel_shape=[3,3]>(r)\n"
        " c1 = Conv(p, w5, b5)\n y = Conv(c1, w2, b2)",
        _convs(),
    )
    q = _xint8(model)
    x = _data(n=1, seed=11)[0]["x"]
    want = _run(model, x)
    got = _run(q, x)
    assert np.abs(got - want).max() < 0.25 * np.abs(want).max()


# -- the float-graph conversions -----------------------------------------------------------


def _bn_inits(rng, c=8, prefix="b"):
    return [
        numpy_helper.from_array(
            (1 + _w(rng, c, scale=0.3)).astype(np.float32), prefix + "s"
        ),
        numpy_helper.from_array(_w(rng, c), prefix + "b"),
        numpy_helper.from_array(_w(rng, c), prefix + "m"),
        numpy_helper.from_array(
            (1 + np.abs(_w(rng, c))).astype(np.float32), prefix + "v"
        ),
    ]


def test_batch_norm_folds_into_the_convolution():
    rng = np.random.default_rng(1)
    model = _model(
        "c0 = Conv(x, w1, b1)\n b = BatchNormalization(c0, bs, bb, bm, bv)\n"
        " y = Conv(b, w2, b2)",
        _convs(rng) + _bn_inits(rng),
    )
    out = conv.graph_cleanup(model)
    assert [n.op_type for n in out.graph.node] == ["Conv", "Conv"]
    x = _data(n=1)[0]["x"]
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-4, atol=1e-5)


def test_batch_norm_after_a_relu_becomes_a_depthwise_conv():
    rng = np.random.default_rng(2)
    model = _model(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n b = BatchNormalization(r, bs, bb, bm, bv)\n"
        " y = Conv(b, w5, b5)",
        _convs(rng) + _bn_inits(rng),
    )
    out = conv.convert_for_npu(model, {})
    assert [n.op_type for n in out.graph.node] == ["Conv", "Relu", "Conv", "Conv"]
    dw = out.graph.node[2]
    assert next(a.i for a in dw.attribute if a.name == "group") == 8
    # BatchNorm's epsilon defaults to 1e-5, Quark's conversion uses 1e-10 when the
    # attribute is absent: so the two differ at the 1e-5 relative level
    x = _data(n=1)[0]["x"]
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-3, atol=1e-4)
    kept = conv.convert_for_npu(model, {"ConvertBNToConv": False})
    assert "BatchNormalization" in [n.op_type for n in kept.graph.node]


def test_reduce_mean_over_the_spatial_axes_becomes_a_global_average_pool():
    body = (
        "c0 = Conv(x, w1, b1)\n r = ReduceMean<axes=[2,3], keepdims=1>(c0)\n"
        " y = Conv(r, w2, b2)"
    )
    out = conv.convert_for_npu(_model(body, _convs()), {})
    assert [n.op_type for n in out.graph.node] == ["Conv", "GlobalAveragePool", "Conv"]
    other = body.replace("[2,3]", "[3]")
    out = conv.convert_for_npu(_model(other, _convs()), {})
    assert "ReduceMean" in [n.op_type for n in out.graph.node]


def test_large_global_pool_is_split_in_two():
    shape = (1, 3, 40, 40)
    rng = np.random.default_rng(3)
    inits = [
        numpy_helper.from_array(_w(rng, 4, 3, 1, 1), "w1"),
        numpy_helper.from_array(_w(rng, 4), "b1"),
    ]
    model = _model(
        "c0 = Conv(x, w1, b1)\n y = GlobalAveragePool(c0)",
        inits,
        inp=f"float{list(shape)} x",
    )
    out = conv.convert_for_npu(model, {})
    pool = [n for n in out.graph.node if n.op_type == "AveragePool"]
    assert len(pool) == 1
    ks = next(list(a.ints) for a in pool[0].attribute if a.name == "kernel_shape")
    assert ks == [5, 5]  # sqrt-ish factor of 40 -> 8 x 8 left for the global pool
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-4, atol=1e-5)
    assert [
        n.op_type
        for n in conv.convert_for_npu(model, {"SplitLargeKernelPool": False}).graph.node
    ] == ["Conv", "GlobalAveragePool"]


def test_split_with_sizes_becomes_slices():
    rng = np.random.default_rng(4)
    inits = _convs(rng) + [
        numpy_helper.from_array(_w(rng, 2, 4, 1, 1), "w8"),
        numpy_helper.from_array(_w(rng, 2), "b8"),
        numpy_helper.from_array(np.array([4, 4], np.int64), "sp"),
    ]
    model = _model(
        "c0 = Conv(x, w1, b1)\n s0, s1 = Split<axis=1>(c0, sp)\n"
        " y0 = Conv(s0, w8, b8)\n y1 = Conv(s1, w8, b8)\n y = Add(y0, y1)",
        inits,
    )
    out = conv.convert_for_npu(model, {})
    assert [n.op_type for n in out.graph.node if n.op_type in ("Split", "Slice")] == [
        "Slice",
        "Slice",
    ]
    x = _data(n=1)[0]["x"]
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-5, atol=1e-6)


def test_convert_clip_to_relu_drops_the_upper_bound():
    rng = np.random.default_rng(5)
    inits = _convs(rng) + [
        numpy_helper.from_array(np.array(0.0, np.float32), "lo"),
        numpy_helper.from_array(np.array(6.0, np.float32), "hi"),
        numpy_helper.from_array(np.array(-1.0, np.float32), "neg"),
    ]
    body = "c0 = Conv(x, w1, b1)\n c = Clip(c0, lo, hi)\n y = Conv(c, w2, b2)"
    out = conv.convert_clip_to_relu(_model(body, inits))
    assert [n.op_type for n in out.graph.node] == ["Conv", "Relu", "Conv"]
    assert {t.name for t in out.graph.initializer}.isdisjoint({"lo", "hi"})
    neg = conv.convert_clip_to_relu(_model(body.replace("lo", "neg"), inits))
    assert "Clip" in [n.op_type for n in neg.graph.node]
    # through the preset: VINT8 turns it on, XINT8 leaves Clips alone
    q = _xint8(_model(body, inits), preset="VINT8")
    assert "Relu" in _ops(q) and "Clip" not in _ops(q)
    assert "Clip" in _ops(_xint8(_model(body, inits)))
    assert "Relu" in _ops(_xint8(_model(body, inits), {"ConvertClipToRelu": True}))


def test_clip_bounds_can_be_rounded_for_the_dpu():
    rng = np.random.default_rng(6)
    inits = _convs(rng) + [
        numpy_helper.from_array(np.array(-0.4, np.float32), "lo"),
        numpy_helper.from_array(np.array(6.4, np.float32), "hi"),
    ]
    body = "c0 = Conv(x, w1, b1)\n c = Clip(c0, lo, hi)\n y = Conv(c, w2, b2)"
    q = _xint8(_model(body, inits), {"ConvertClipToDPUVersion": True})
    vals = {
        t.name: float(numpy_helper.to_array(t))
        for t in q.graph.initializer
        if t.name in ("lo", "hi")
    }
    assert vals == {"lo": 0.0, "hi": 6.0}


# -- BiasCorrection's power-of-two bias re-quantization ----------------------------------------


@pytest.mark.parametrize("symmetric", [True, False])
def test_pof2_requantization_properties(symmetric):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(16) * 0.3).astype(np.float32)
    codes, scale, zp = qbc.quark_pof2_quantize(data, "int8", symmetric)
    assert codes.dtype == np.int8 and np.log2(scale) == round(np.log2(scale))
    assert np.abs(codes.astype(np.int32)).max() <= 127
    if symmetric:
        assert zp == 0
        # the chosen grid is the least-squares one among the five candidates
        errs = [
            np.sum(
                (
                    (np.clip(np.round(data / np.float32(2.0**-p)), -127, 127))
                    * np.float32(2.0**-p)
                    - data
                )
                ** 2
            )
            for p in range(-3, 12)
        ]
        got = np.sum((codes.astype(np.float32) * scale - data) ** 2)
        assert got <= min(errs) * 4


def _bc_model():
    rng = np.random.default_rng(8)
    inits = [
        numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w0"),
        numpy_helper.from_array(_w(rng, 8, scale=2), "b0"),
        numpy_helper.from_array(_w(rng, 8, 8, 1, 1), "w1"),
        numpy_helper.from_array(_w(rng, 8), "b1"),
        numpy_helper.from_array(_w(rng, 4, 8, 1, 1), "w2"),
        numpy_helper.from_array(_w(rng, 4, scale=0.1), "b2"),
    ]
    return _model(
        "c0 = Conv<pads=[1,1,1,1]>(x, w0, b0)\n r0 = Relu(c0)\n c1 = Conv(r0, w1, b1)\n"
        " r1 = Relu(c1)\n y = Conv(r1, w2, b2)",
        inits,
        inp="float[1,3,8,8] x",
    )


def _bias_dqs(model):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    return [
        (inits[n.input[0]], float(inits[n.input[1]].ravel()[0]))
        for n in model.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in inits
        and inits[n.input[0]].ndim == 1
    ]


def test_xint8_bias_correction_keeps_quarks_scale_quirk_unless_asked():
    model = _bc_model()
    data = _data((1, 3, 8, 8))
    base = _xint8(model, data=data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = qc.QConfig.get_default_config("XINT8")
        cfg.algo_config = [qc.BiasCorrectionConfig()]
        quirk = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(data)
        )
    assert any("power-of-two flow" in str(w.message) for w in caught)
    plain = _xint8(
        model,
        {"BiasCorrectionStoredScale": True},
        [qc.BiasCorrectionConfig()],
        data=data,
    )
    # the stored scales never change ...
    for (_, s0), (_, s1), (_, s2) in zip(
        _bias_dqs(base), _bias_dqs(quirk), _bias_dqs(plain)
    ):
        assert s0 == s1 == s2
    # ... but the quirk writes codes of a different grid where the fresh scale differs
    assert any(
        not np.array_equal(a, b)
        for (a, _), (b, _) in zip(_bias_dqs(quirk), _bias_dqs(plain))
    )
    # StoredScale keeps every code in range and consistent with the stored scale
    assert all(np.abs(c.astype(np.int32)).max() <= 127 for c, _ in _bias_dqs(plain))


def test_bias_correction_methods_follow_quarks_dispatch():
    model = _bc_model()
    data = _data((1, 3, 8, 8))
    cfg = qc.QConfig.get_default_config("INT8_CNN_DEFAULT")  # MinMax -> int32 codes
    cfg.algo_config = [qc.BiasCorrectionConfig()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(data)
        )
    assert all(c.dtype == np.int32 for c, _ in _bias_dqs(mm))
    # a histogram calibrator is not one Quark's BiasCorrection writes biases for
    cfg = qc.QConfig.get_default_config("INT8_CNN_DEFAULT")
    cfg.global_config.activation.calibration_method = qc.CalibMethod.Entropy
    plain = qc.ModelQuantizer(cfg).quantize_model(
        model, calibration_data_reader=_Reader(data)
    )
    cfg.algo_config = [qc.BiasCorrectionConfig()]
    corrected = qc.ModelQuantizer(cfg).quantize_model(
        model, calibration_data_reader=_Reader(data)
    )
    for (a, _), (b, _) in zip(_bias_dqs(plain), _bias_dqs(corrected)):
        np.testing.assert_array_equal(a, b)


# -- ReduceRange, Pad fusion, Identity -------------------------------------------------------


def _weight_codes(model):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    return [
        int(np.abs(inits[n.input[0]].astype(np.int32)).max())
        for n in model.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in inits
        and inits[n.input[0]].ndim == 4
    ]


def test_reduce_range_keeps_weights_to_the_reduced_grid():
    model = _model(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n y = Conv(r, w5, b5)", _convs()
    )
    full = _xint8(model, preset="U8S8_AAWS")
    reduced = _xint8(model, {"ReduceRange": True}, preset="U8S8_AAWS")
    assert max(_weight_codes(full)) == 127 and max(_weight_codes(reduced)) == 64
    # activations are untouched, the weights' scale doubles
    a = {t.name: numpy_helper.to_array(t) for t in full.graph.initializer}
    b = {t.name: numpy_helper.to_array(t) for t in reduced.graph.initializer}
    scales = [k for k in a if k.endswith("/scale") and a[k].size == 1]
    ratios = {round((b[k] / a[k]).item(), 3) for k in scales}
    assert ratios == {1.0, round(127 / 64, 3)}


def test_reduce_range_is_refused_where_quark_refuses_or_has_no_equivalent():
    model = _model("c0 = Conv(x, w1, b1)\n y = Conv(c0, w5, b5)", _convs())
    with pytest.raises(ValueError, match="ReduceRange"):
        _xint8(model, {"ReduceRange": True})  # NPU CNN: Quark raises as well
    with pytest.raises(NotImplementedError, match="reduce_range"):
        _xint8(model, {"ReduceRange": True}, preset="VINT8")


def test_pad_fuses_into_the_convolution_and_the_pool():
    rng = np.random.default_rng(7)
    pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")
    for body, op in (
        ("c0 = Conv(x, w1, b1)\n p = Pad(c0, pads)\n y = Conv(p, w2, b2)", "Conv"),
        (
            "c0 = Conv(x, w1, b1)\n p = Pad(c0, pads)\n"
            " a = AveragePool<kernel_shape=[3,3]>(p)\n y = Conv(a, w2, b2)",
            "AveragePool",
        ),
    ):
        model = _model(body, _convs(rng) + [pads])
        fused = conv.graph_cleanup(model)
        assert "Pad" not in [n.op_type for n in fused.graph.node]
        x = _data(n=1)[0]["x"]
        np.testing.assert_allclose(_run(fused, x), _run(model, x), rtol=1e-5, atol=1e-5)
    # onnxslim fuses into a Conv, only ONNX Runtime also into a pool
    kept = conv.graph_cleanup(model, optimize=False)
    assert "Pad" in [n.op_type for n in kept.graph.node]
    # a non-zero pad value, or padding of the batch / channel dims, is not fused
    value = numpy_helper.from_array(np.array(1.0, np.float32), "pv")
    model = _model(
        "c0 = Conv(x, w1, b1)\n p = Pad(c0, pads, pv)\n y = Conv(p, w2, b2)",
        _convs(rng) + [pads, value],
    )
    assert "Pad" in [n.op_type for n in conv.graph_cleanup(model).graph.node]


def test_identity_nodes_are_removed():
    model = _model(
        "c0 = Conv(x, w1, b1)\n i = Identity(c0)\n y = Conv(i, w2, b2)", _convs()
    )
    assert [n.op_type for n in conv.graph_cleanup(model).graph.node] == ["Conv", "Conv"]
    kept = _xint8(model, {"OptimizeModel": False, "SimplifyModel": False})
    assert "Identity" in _ops(kept)


def test_pad_before_an_average_pool_loses_its_quantizer_pair():
    """Quark drops the Q/DQ between a Pad and its (Average)Pool consumer."""
    pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")
    model = _model(
        "c0 = Conv(x, w1, b1)\n p = Pad(c0, pads)\n"
        " a = AveragePool<kernel_shape=[3,3]>(p)\n y = Conv(a, w2, b2)",
        _convs() + [pads],
    )
    q = _xint8(model, {"OptimizeModel": False, "SimplifyModel": False})
    ops = [n.op_type for n in q.graph.node]
    assert "Pad" in ops and "AveragePool" in ops
    by_out = {o: n for n in q.graph.node for o in n.output}
    pad = next(n for n in q.graph.node if n.op_type == "Pad")
    pool = next(n for n in q.graph.node if n.op_type == "AveragePool")
    assert by_out[pool.input[0]] is pad  # no Q/DQ between them


def test_batch_norm_that_stays_is_not_quantized():
    rng = np.random.default_rng(2)
    model = _model(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n b = BatchNormalization(r, bs, bb, bm, bv)\n"
        " y = Conv(b, w5, b5)",
        _convs(rng) + _bn_inits(rng),
    )
    q = _xint8(model, {"ConvertBNToConv": False}, preset="U8S8_AAWS")
    bn = next(n for n in q.graph.node if n.op_type == "BatchNormalization")
    inits = {t.name for t in q.graph.initializer}
    assert all(x in inits for x in bn.input[1:])  # float scale / bias / mean / var
    assert _ops(q).count("Conv") == 2  # the Conv after it is still quantized
    converted = _xint8(model, preset="XINT8")
    assert "BatchNormalization" not in _ops(converted)


# == Edge cases: Quark's op-type list and node order, optimizer passes, DPU nodes ==========

from onnxsim import quark_marking as marking  # noqa: E402


def _graph(body, inits=(), opset=17, inp="float[1,4,4,4] x"):
    model = parser.parse_model(
        f"""<ir_version: 9, opset_import: ["": {opset}]>
        g ({inp}) => (float y) {{ {body} }}"""
    )
    model.graph.initializer.extend(inits)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return model


def test_quarks_op_type_lists():
    base = marking.quark_op_types(False)
    cnn = marking.quark_op_types(True, extra=["Exp"])
    assert {"Conv", "Gemm", "Relu", "Clip", "Reshape", "MaxPool", "Softmax"} <= base
    assert not {"Flatten", "Neg", "Exp", "Sub", "Slice", "PRelu"} & base
    assert {"Sub", "Slice", "PRelu", "HardSigmoid", "ReduceMean", "Exp"} <= cnn
    assert "Flatten" not in cnn


def test_node_order_is_the_topological_sort_of_onnx_runtime():
    """Quark sorts with ``ONNXModel.topological_sort`` -- constants first, then the
    nodes the *alphabetically sorted* inputs and initializers release, then breadth
    first."""
    from onnxruntime.quantization.onnx_model import ONNXModel

    rng = np.random.default_rng(0)
    for seed in range(12):
        r = np.random.default_rng(seed)
        names = ["zw", "aw", "mw", "bb"]
        lines = ["k = Constant<value_float=2.0>()"]
        tensors = ["x", "k"]
        for i in range(int(r.integers(6, 14))):
            a, b = r.choice(tensors, 2)
            kind = int(r.integers(0, 3))
            if kind == 0:
                lines.append(f"t{i} = Add({a}, {b})")
            elif kind == 1:
                lines.append(f"t{i} = Mul({a}, {r.choice(names)})")
            else:
                lines.append(f"t{i} = Relu({a})")
            tensors.append(f"t{i}")
        lines.append(f"y = Add({tensors[-1]}, aw)")
        inits = [
            numpy_helper.from_array(
                rng.standard_normal((1, 4, 4, 4)).astype(np.float32), n
            )
            for n in names
        ]
        model = _graph("\n".join(lines), inits)
        for i, n in enumerate(model.graph.node):  # a different file order
            n.name = f"n{i}"
        ref = onnx.ModelProto()
        ref.CopyFrom(model)
        wrapped = ONNXModel(ref)
        wrapped.topological_sort()
        want = [(n.op_type, n.output[0]) for n in ref.graph.node]
        got = [(n.op_type, n.output[0]) for n in marking.quark_node_order(model)]
        assert got == want
        assert [n.output[0] for n in marking.quark_sorted(model).graph.node] == [
            o for _, o in want
        ]


def test_a_relu_or_clip_is_skipped_when_no_earlier_node_marked_its_input():
    lo = numpy_helper.from_array(np.array(0.0, np.float32), "lo")
    hi = numpy_helper.from_array(np.array(6.0, np.float32), "hi")
    w = numpy_helper.from_array(np.ones((4, 4, 1, 1), np.float32), "w")
    types = marking.quark_op_types(True)

    def skipped(body, **kw):
        model = _graph(body, [lo, hi, w])
        return marking.skipped_nodes(model, types, **kw)

    assert skipped("r = Relu(x)\n y = Conv(r, w)") == {"n0_Relu"}
    assert skipped("c = Clip(x, lo, hi)\n y = Conv(c, w)") == {"n0_Clip"}
    # a Conv visited first marks x for the Relu (the sort releases the readers of an
    # input in file order: whichever is listed first is visited first)
    assert skipped("c = Conv(x, w)\n r = Relu(x)\n y = Add(c, r)") == set()
    assert skipped("r = Relu(x)\n c = Conv(x, w)\n y = Add(c, r)") == {"n0_Relu"}
    # an op outside the registries marks nothing: its consumer finds an unmarked input
    assert skipped("n = Neg(x)\n r = Relu(n)\n y = Conv(r, w)") == {"n1_Relu"}
    # data movement: skipped without ForceQuantizeNoInputCheck when unmarked
    body = "t = Transpose<perm=[0,1,3,2]>(x)\n y = Conv(t, w)"
    assert skipped(body) == set()
    assert skipped(body, force_no_input_check=False) == {"n0_Transpose"}
    # a HardSigmoid that is not 1/6, 0.5 is never quantized
    hs = "h = HardSigmoid<alpha=0.2>(x)\n y = Conv(h, w)"
    assert skipped(hs) == {"n0_HardSigmoid"}


def test_xint8_leaves_the_input_of_an_unmarked_clip_and_of_a_flatten_float():
    lo = numpy_helper.from_array(np.array(0.0, np.float32), "lo")
    hi = numpy_helper.from_array(np.array(6.0, np.float32), "hi")
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(_w(rng, 4, 3, 3, 3), "w")
    b = numpy_helper.from_array(_w(rng, 4), "b")
    wg = numpy_helper.from_array(_w(rng, 432, 4), "wg")
    shape = (1, 3, 12, 12)
    for body, op in (
        ("c = Clip(x, lo, hi)\n y = Conv(c, w, b)", "Clip"),
        ("f = Flatten(x)\n y = Gemm(f, wg)", "Flatten"),
    ):
        model = _graph(body, [lo, hi, w, b, wg], inp=f"float{list(shape)} x")
        q = _xint8(model)
        node = next(n for n in q.graph.node if n.op_type == op)
        assert node.input[0] == "x", "no Q/DQ pair on the graph input"
        # its *output* is quantized, for the consumer
        reader = next(n for n in q.graph.node if node.output[0] in n.input)
        assert reader.op_type == "QuantizeLinear"
    # ... while a Conv reading the same input marks it for everyone
    model = _graph(
        "c = Conv<pads=[1,1,1,1]>(x, w, b)\n r = Relu(x)\n y = Concat<axis=1>(c, r)",
        [w, b],
        inp=f"float{list(shape)} x",
    )
    q = _xint8(model)
    relu = next(n for n in q.graph.node if n.op_type == "Relu")
    assert relu.input[0] != "x"


def test_pad_constant_value_is_quantized_and_align_pad_follows_it():
    rng = np.random.default_rng(1)
    pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")
    big = numpy_helper.from_array(np.array(100.0, np.float32), "big")
    inits = [
        pads,
        big,
        numpy_helper.from_array(_w(rng, 4, 3, 3, 3), "w1"),
        numpy_helper.from_array(_w(rng, 4), "b1"),
        numpy_helper.from_array(_w(rng, 2, 4, 3, 3), "w2"),
        numpy_helper.from_array(_w(rng, 2), "b2"),
    ]
    model = _graph(
        "c0 = Conv(x, w1, b1)\n p = Pad(c0, pads, big)\n y = Conv(p, w2, b2)",
        inits,
        inp=f"float{list(SHAPE)} x",
    )
    on = _xint8(model)
    off = _xint8(model, {"AlignPad": False})
    # the value is an int8 constant (its DQ reads codes, scale 2**-k), not float
    pad = next(n for n in on.graph.node if n.op_type == "Pad")
    by_out = {o: n for n in on.graph.node for o in n.output}
    assert by_out[pad.input[2]].op_type == "DequantizeLinear"

    def scale(model, tensor_prefix):
        inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        q = next(
            n
            for n in model.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0].startswith(tensor_prefix)
        )
        return float(inits[q.input[1]])

    # Pad's output grid is coarse (100), its input's was fine: AlignPad moves the input
    assert scale(off, "c0") < scale(off, "p") == scale(on, "p")
    assert scale(on, "c0") == scale(on, "p")


def test_all_zero_unsigned_activation_scale():
    h = cal._Pof2Histogram()
    h.add(np.zeros(64, np.float32))
    assert h.scale("uint8") == 4.0  # Quark: threshold [0, 255 * 2], read as 2 * 2
    assert h.scale("int8") == 2.0


def test_calibration_session_runs_without_graph_optimizations(monkeypatch):
    import onnxruntime as ort

    seen = []
    real = ort.InferenceSession

    def spy(model, sess_options=None, *a, **k):
        if not sess_options.optimized_model_filepath:  # (not the cleanup session)
            seen.append(sess_options.graph_optimization_level)
        return real(model, sess_options, *a, **k)

    monkeypatch.setattr(ort, "InferenceSession", spy)
    rng = np.random.default_rng(0)
    model = _model(
        "c0 = Conv(x, w1, b1)\n y = Conv(c0, w2, b2)",
        _convs(rng)[:2] + _convs(rng)[2:4],
    )
    _xint8(model)
    assert seen and all(
        level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL for level in seen
    )


def test_softmax_dpu_nodes():
    node = parser.parse_node("s = Softmax<axis=1>(c)")
    node.name = "sm"
    new = npu.softmax_dpu_nodes(node, 13)
    ops = [n.op_type for n in new]
    assert ops.count("Floor") == 1 and ops.count("Pow") == 1 and ops.count("Div") == 1
    assert "Softmax" not in ops and ops[-1] == "Cast"
    assert new[-1].output == ["s"] and new[0].input == ["c"]
    consts = [n for n in new if n.op_type == "Constant"]
    assert {c.attribute[0].t.data_type for c in consts} == {
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.INT64,
    }
    # before opset 13 the axes are an attribute; below opset 7 nothing is converted
    old = npu.softmax_dpu_nodes(node, 11)
    rs = next(n for n in old if n.op_type == "ReduceSum")
    assert (
        len(rs.input) == 1
        and next(a.ints[0] for a in rs.attribute if a.name == "axes") == 1
    )
    assert npu.softmax_dpu_nodes(node, 6) == []


def test_softmax_and_instance_norm_dpu_versions_are_opt_in():
    rng = np.random.default_rng(0)
    inits = _convs(rng) + [
        numpy_helper.from_array((1 + _w(rng, 8)).astype(np.float32), "sc"),
        numpy_helper.from_array(_w(rng, 8), "bi"),
    ]
    sm = _model(
        "c0 = Conv(x, w1, b1)\n s = Softmax<axis=1>(c0)\n y = Conv(s, w2, b2)",
        inits,
        opset=13,
    )
    inn = _model(
        "c0 = Conv(x, w1, b1)\n s = InstanceNormalization<epsilon=0.001>(c0, sc, bi)\n y = Conv(s, w2, b2)",
        inits,
        opset=13,
    )
    assert "Softmax" in _ops(_xint8(sm))
    q = _xint8(sm, {"ConvertSoftmaxToDPUVersion": True})
    assert "Softmax" not in _ops(q) and "Floor" in _ops(q)
    assert "InstanceNormalization" in _ops(_xint8(inn))
    q = _xint8(inn, {"ConvertInstanceNormToDPUVersion": True})
    ext = next(n for n in q.graph.node if n.op_type == "ExtendedInstanceNormalization")
    assert ext.domain == "com.amd.quark"
    assert next(a.f for a in ext.attribute if a.name == "epsilon") == pytest.approx(
        0.001
    )
    assert any(o.domain == "com.amd.quark" for o in q.opset_import)


def test_instance_norm_bias_is_an_int8_constant_under_xint8():
    rng = np.random.default_rng(0)
    inits = _convs(rng) + [
        numpy_helper.from_array((1 + _w(rng, 8)).astype(np.float32), "sc"),
        numpy_helper.from_array(_w(rng, 8), "bi"),
    ]
    model = _model(
        "c0 = Conv(x, w1, b1)\n s = InstanceNormalization(c0, sc, bi)\n y = Conv(s, w2, b2)",
        inits,
        opset=13,
    )
    q = _xint8(model)
    codes = {
        t.name: t.data_type
        for t in q.graph.initializer
        if t.name.startswith("bi") and t.name.endswith(("/int8", "/int32"))
    }
    assert set(codes.values()) == {onnx.TensorProto.INT8}
    with_int32 = _xint8(model, {"Int32Bias": True})
    assert any(
        t.data_type == onnx.TensorProto.INT32
        for t in with_int32.graph.initializer
        if t.name.startswith("bi")
    )


def test_dpu_nodes_are_appended_and_pool_shapes_come_from_the_unrewritten_graph():
    """Quark's ``insert_mul`` appends the ``Constant`` and ``Mul`` at the end of the
    node list, and a ``Sigmoid`` becomes a ``HardSigmoid`` appended there too; the
    shape of a pool's input is looked up in the graph as it was before."""
    rng = np.random.default_rng(0)
    model = _model(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n p = GlobalAveragePool(s)\n"
        " m = Mul(s, p)\n y = Conv(m, w5, b5)",
        _convs(rng),
    )
    q = _xint8(model, {"OnnxsimKeepQuarkNodeOrder": True})
    ops = [n.op_type for n in q.graph.node]
    assert ops[-4:] == ["HardSigmoid", "Constant", "Mul", "Constant"] or ops[-5:] == [
        "HardSigmoid",
        "Constant",
        "Mul",
        "Constant",
        "Mul",
    ]
    # both the HardSigmoid and the pool got their DPU Mul
    assert [
        n.op_type
        for n in q.graph.node
        if n.op_type == "Mul" and n.input[1].endswith("_Scale")
    ] == ["Mul", "Mul"]
    ordered = _xint8(model)  # default: topologically sorted again
    assert "HardSigmoid" in _ops(ordered)
    onnx.checker.check_model(ordered)


def test_quark_qdq_sorted_follows_quarks_parameter_names():
    """The sort seeds on the alphabetical order of the initializer names, so it is
    run on a copy with Quark's names (``<w>_scale`` ...) for the Q/DQ parameters."""
    rng = np.random.default_rng(0)
    twin = [
        numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1b"),
        numpy_helper.from_array(_w(rng, 8), "b1b"),
    ]
    model = _model(
        "c0 = Conv(x, w1, b1)\n c1 = Conv(x, w1b, b1b)\n y = Add(c0, c1)",
        _convs(rng) + twin,
    )
    cfg = qc.QConfig.get_default_config("XINT8")
    cfg.extra_options["OnnxsimKeepQuarkNodeOrder"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data())
        )
    again = marking.quark_qdq_sorted(q)
    assert [n.output[0] for n in again.graph.node] == [
        n.output[0] for n in q.graph.node
    ]
    assert marking._quark_names(q)  # every Q/DQ parameter has a Quark-style name
    names = set(marking._quark_names(q).values())
    assert {"w1_quantized", "w1_scale", "x_scale", "x_zero_point"} <= names


def test_a_shared_int32_bias_is_quantized_once_with_the_first_readers_scales():
    rng = np.random.default_rng(2)
    inits = [
        numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1"),
        numpy_helper.from_array(_w(rng, 8), "b1"),
        numpy_helper.from_array(_w(rng, 8, 8, 1, 1), "w2"),
        numpy_helper.from_array(_w(rng, 8, 8, 1, 1, scale=0.1), "w3"),
        numpy_helper.from_array(_w(rng, 8, scale=3), "bs"),
    ]
    model = _model(
        "a = Conv(x, w1, b1)\n r = Relu(a)\n y1 = Conv(r, w2, bs)\n z = Conv(r, w3, bs)\n"
        " y = Add(y1, z)",
        inits,
    )
    q = _xint8(model, {"Int32Bias": True})
    convs = [n for n in q.graph.node if n.op_type == "Conv"]
    assert convs[1].input[2] == convs[2].input[2], "both read one bias DQ"
    # (a copy per Conv under non-power-of-two calibrations, Quark's CopyBiasInit)
    a8 = _xint8(model, preset="A8W8")
    convs = [n for n in a8.graph.node if n.op_type == "Conv"]
    assert convs[1].input[2] != convs[2].input[2]


# -- the float-graph optimizer steps ------------------------------------------------


def test_remove_input_init_and_duplicate_shared_biases():
    rng = np.random.default_rng(0)
    model = _model(
        "a = Conv(x, w1, b1)\n y = Conv(a, w3, b1)",
        [
            numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1"),
            numpy_helper.from_array(_w(rng, 8), "b1"),
            numpy_helper.from_array(_w(rng, 8, 8, 1, 1), "w3"),
        ],
    )
    model.graph.input.append(onnx.helper.make_tensor_value_info("w1", 1, [8, 3, 3, 3]))
    out = conv.remove_input_init(model)
    assert [i.name for i in out.graph.input] == ["x"]
    dup = conv.duplicate_shared_biases(model)
    convs = [n for n in dup.graph.node if n.op_type == "Conv"]
    assert convs[0].input[2] == "b1" and convs[1].input[2] == "duplicatedb12"
    inits = {t.name: numpy_helper.to_array(t) for t in dup.graph.initializer}
    np.testing.assert_array_equal(inits["b1"], inits["duplicatedb12"])


def test_onnx_runtime_optimizer_folds_what_the_reproductions_do_not():
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    extra = [
        numpy_helper.from_array(_w(rng, 1, 8, 1, 1), "ca"),
        numpy_helper.from_array(np.array(0.0, np.float32), "lo"),
        numpy_helper.from_array(np.array(6.0, np.float32), "hi"),
    ]
    for body, want in (
        (
            "c0 = Conv(x, w1, b1)\n a = Add(c0, ca)\n y = Conv(a, w2, b2)",
            ["Conv", "Conv"],
        ),
        (
            "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n c = Clip(r, lo, hi)\n y = Conv(c, w2, b2)",
            ["Conv", "Clip", "Conv"],
        ),
        (
            "c0 = Conv(x, w1, b1)\n r1 = Relu(c0)\n r2 = Relu(c0)\n s = Add(r1, r2)\n y = Conv(s, w2, b2)",
            ["Conv", "Relu", "Add", "Conv"],
        ),
    ):
        model = _model(body, _convs() + extra)
        got = conv.graph_cleanup(model, runtime=True)
        assert sorted(n.op_type for n in got.graph.node) == sorted(want), body
        python_only = conv.graph_cleanup(model)
        assert len(python_only.graph.node) > len(got.graph.node)
        x = _data(n=1)[0]["x"]
        np.testing.assert_allclose(_run(got, x), _run(model, x), rtol=1e-5, atol=1e-5)


def test_batch_norm_after_a_concat_folds_into_each_convolution():
    rng = np.random.default_rng(3)
    inits = [
        numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1"),
        numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w3"),
        numpy_helper.from_array(_w(rng, 8), "b3"),
        numpy_helper.from_array(_w(rng, 2, 16, 1, 1), "w4"),
        numpy_helper.from_array(_w(rng, 2), "b4"),
    ] + _bn_inits(rng, c=16)
    model = _model(
        "a = Conv(x, w1)\n b = Conv(x, w3, b3)\n k = Concat<axis=1>(a, b)\n"
        " n = BatchNormalization(k, bs, bb, bm, bv)\n y = Conv(n, w4, b4)",
        inits,
    )
    out = conv.fold_batch_norm_after_concat(model)
    assert [n.op_type for n in out.graph.node] == ["Conv", "Conv", "Concat", "Conv"]
    x = _data(n=1)[0]["x"]
    # (Quark's epsilon default is 1e-10 against ONNX's 1e-5)
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-3, atol=1e-4)
    assert len(out.graph.node[0].input) == 3, "the bias-free Conv got one"
    # an unfoldable parent (a Relu) leaves the BatchNormalization
    blocked = _model(
        "a = Conv(x, w1)\n r = Relu(a)\n b = Conv(x, w3, b3)\n k = Concat<axis=1>(r, b)\n"
        " n = BatchNormalization(k, bs, bb, bm, bv)\n y = Conv(n, w4, b4)",
        inits,
    )
    assert "BatchNormalization" in [
        n.op_type for n in conv.fold_batch_norm_after_concat(blocked).graph.node
    ]


def test_the_python_reproductions_stand_in_for_a_missing_optimizer(monkeypatch):
    """Without ONNX Runtime (or onnxslim) -- or with ``UseRuntimeOptimizers=False`` --
    the BatchNorm / Pad / Identity / HardSwish reproductions are used."""
    rng = np.random.default_rng(1)
    model = _model(
        "c0 = Conv(x, w1, b1)\n b = BatchNormalization(c0, bs, bb, bm, bv)\n"
        " h = HardSwish(b)\n i = Identity(h)\n y = Conv(i, w2, b2)",
        _convs(rng) + _bn_inits(rng),
    )
    twice = _model(
        "c0 = Conv(x, w1, b1)\n r1 = Relu(c0)\n r2 = Relu(c0)\n s = Add(r1, r2)\n y = Conv(s, w2, b2)",
        _convs(rng),
    )
    deduped = conv.eliminate_duplicate_nodes(twice)
    assert [n.op_type for n in deduped.graph.node] == ["Conv", "Relu", "Add", "Conv"]
    assert list(deduped.graph.node[2].input) == ["r1", "r1"]
    monkeypatch.setattr(conv, "ort_basic_optimize", lambda m: None)
    monkeypatch.setattr(conv, "onnxslim_simplify", lambda m, c=None: None)
    got = conv.graph_cleanup(model, runtime=True)
    ops = [n.op_type for n in got.graph.node]
    assert "HardSwish" not in ops and "Identity" not in ops
    assert "BatchNormalization" not in ops and ops.count("Conv") == 2
    q = _xint8(model, {"UseRuntimeOptimizers": False})
    assert "HardSwish" not in _ops(q) and "Identity" not in _ops(q)
    x = _data(n=1)[0]["x"]
    np.testing.assert_allclose(_run(got, x), _run(model, x), rtol=1e-3, atol=1e-4)
