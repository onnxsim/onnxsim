"""The Quark-compat integer flow on models whose default-domain opset is below 13.

Quark quantizes such models in place: it never converts the opset (only an FP8
request bumps it), per-tensor ``QuantizeLinear`` / ``DequantizeLinear`` carry no
``axis``, and ONNX Runtime's QDQ operator quantizers skip ``MaxPool`` below
opset 12 and ``Resize`` below opset 11. A per-channel configuration raises there.
``tests/test_quark_low_opset_parity.py`` compares all of that with the real Quark;
the tests here need nothing but ONNX Runtime and also own the model zoo.

Models are built in the ONNX text format; the attribute-vs-input forms of Clip,
Pad, Split, Squeeze / Unsqueeze, Slice and Resize follow the opset they declare.
"""

import warnings

import numpy as np
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc
from onnxsim.full_qdq import quantize_full_qdq

OPSETS = (9, 10, 11, 12)
QUARK_ERROR = (
    "Per-Channel support with QDQ format requires onnx opset version 13 or above."
)


def _w(rng, *shape):
    return (rng.standard_normal(shape) * 0.5).astype(np.float32)


def _init(array, name):
    return numpy_helper.from_array(np.asarray(array), name)


def _model(
    body, opset, initializer=(), inputs="float[1,4,8,8] x", outputs="float[?] y"
):
    model = parser.parse_model(
        f'<ir_version: 7, opset_import: ["": {opset}]> '
        f"g ({inputs}) => ({outputs}) {{ {body} }}"
    )
    model.graph.initializer.extend(initializer)
    return model


def _conv_weights(rng, out=8, cin=4, k=3):
    return [_init(_w(rng, out, cin, k, k), "w1"), _init(_w(rng, out), "b1")]


# Every builder takes the opset and returns ``(model, input shape)``.


def conv_relu_pool(opset):
    rng = np.random.default_rng(1)
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        r0 = Relu(c0)
        y = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r0)"""
    m = _model(body, opset, _conv_weights(rng), outputs="float[1,8,4,4] y")
    return m, (1, 4, 8, 8)


def mlp(opset):
    rng = np.random.default_rng(0)
    body = """
        h0 = Gemm(x, w1, b1)
        h1 = Relu(h0)
        y = Gemm(h1, w2, b2)"""
    inits = [
        _init(_w(rng, 16, 32), "w1"),
        _init(_w(rng, 32), "b1"),
        _init(_w(rng, 32, 8), "w2"),
        _init(_w(rng, 8), "b2"),
    ]
    return _model(body, opset, inits, "float[3,16] x", "float[3,8] y"), (3, 16)


def matmul_add(opset):
    rng = np.random.default_rng(2)
    body = """
        h0 = MatMul(x, w1)
        h1 = Add(h0, b1)
        y = MatMul(h1, w2)"""
    inits = [
        _init(_w(rng, 16, 32), "w1"),
        _init(_w(rng, 32), "b1"),
        _init(_w(rng, 32, 8), "w2"),
    ]
    return _model(body, opset, inits, "float[3,16] x", "float[3,8] y"), (3, 16)


def clip6(opset):
    # Clip: min / max attributes below opset 11, inputs from 11 on
    rng = np.random.default_rng(3)
    inits = _conv_weights(rng) + [
        _init(_w(rng, 4, 8, 1, 1), "w2"),
        _init(_w(rng, 4), "b2"),
    ]
    if opset >= 11:
        clip = "k = Clip(c0, lo, hi)"
        inits += [_init(np.float32(0.0), "lo"), _init(np.float32(6.0), "hi")]
    else:
        clip = "k = Clip<min=0.0, max=6.0>(c0)"
    body = f"""
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        {clip}
        y = Conv(k, w2, b2)"""
    return _model(body, opset, inits, outputs="float[1,4,8,8] y"), (1, 4, 8, 8)


def clip_generic(opset):
    rng = np.random.default_rng(4)
    inits = _conv_weights(rng)
    if opset >= 11:
        clip = "k = Clip(c0, lo, hi)"
        inits += [_init(np.float32(-1.0), "lo"), _init(np.float32(2.0), "hi")]
    else:
        clip = "k = Clip<min=-1.0, max=2.0>(c0)"
    body = f"""
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        {clip}
        y = MaxPool<kernel_shape=[2,2], strides=[2,2]>(k)"""
    return _model(body, opset, inits, outputs="float[1,8,4,4] y"), (1, 4, 8, 8)


def softmax(opset):
    # Softmax's default axis is 1 below opset 13 and -1 from there on
    rng = np.random.default_rng(5)
    body = f"""
        h0 = Gemm(x, w1, b1)
        s = Softmax<axis={1 if opset < 13 else -1}>(h0)
        y = MatMul(s, w2)"""
    inits = [
        _init(_w(rng, 16, 32), "w1"),
        _init(_w(rng, 32), "b1"),
        _init(_w(rng, 32, 8), "w2"),
    ]
    return _model(body, opset, inits, "float[3,16] x", "float[3,8] y"), (3, 16)


def concat_split(opset):
    # Split: ``split`` attribute below opset 13, input from 13 on
    rng = np.random.default_rng(6)
    inits = [
        _init(_w(rng, 4, 4, 3, 3), "wa"),
        _init(_w(rng, 4), "ba"),
        _init(_w(rng, 4, 4, 1, 1), "wb"),
        _init(_w(rng, 4), "bb"),
        _init(_w(rng, 4, 4, 3, 3), "wc"),
        _init(_w(rng, 4), "bc"),
    ]
    split = "s0, s1 = Split<axis=1, split=[4,4]>(c)"
    body = f"""
        a = Conv<pads=[1,1,1,1]>(x, wa, ba)
        b0 = Conv(x, wb, bb)
        c = Concat<axis=1>(a, b0)
        {split}
        d = Add(s0, s1)
        y = Conv<pads=[1,1,1,1]>(d, wc, bc)"""
    return _model(body, opset, inits, outputs="float[1,4,8,8] y"), (1, 4, 8, 8)


def reshape_transpose(opset):
    rng = np.random.default_rng(7)
    inits = _conv_weights(rng) + [
        _init(np.array([1, 8, 64], np.int64), "shp"),
        _init(_w(rng, 8, 5), "w2"),
    ]
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        r = Reshape(c0, shp)
        t = Transpose<perm=[0,2,1]>(r)
        y = MatMul(t, w2)"""
    return _model(body, opset, inits, outputs="float[1,64,5] y"), (1, 4, 8, 8)


def pool(opset):
    rng = np.random.default_rng(8)
    inits = _conv_weights(rng) + [_init(_w(rng, 8, 5), "w2"), _init(_w(rng, 5), "b2")]
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        p0 = AveragePool<kernel_shape=[2,2], strides=[2,2]>(c0)
        p1 = GlobalAveragePool(p0)
        f = Flatten(p1)
        y = Gemm(f, w2, b2)"""
    return _model(body, opset, inits, outputs="float[1,5] y"), (1, 4, 8, 8)


def pad_conv(opset):
    # Pad: pads / value attributes below opset 11, inputs from 11 on
    rng = np.random.default_rng(9)
    inits = _conv_weights(rng)
    if opset >= 11:
        pad = "p = Pad(x, pads)"
        inits.append(_init(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads"))
    else:
        pad = "p = Pad<pads=[0,0,1,1,0,0,1,1]>(x)"
    body = f"""
        {pad}
        c0 = Conv(p, w1, b1)
        y = Relu(c0)"""
    return _model(body, opset, inits, outputs="float[1,8,8,8] y"), (1, 4, 8, 8)


def pad_value(opset):
    rng = np.random.default_rng(17)
    inits = _conv_weights(rng)
    if opset >= 11:
        pad = "p = Pad(x, pads, cv)"
        inits += [
            _init(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads"),
            _init(np.float32(0.5), "cv"),
        ]
    else:
        pad = "p = Pad<pads=[0,0,1,1,0,0,1,1], value=0.5>(x)"
    body = f"""
        {pad}
        c0 = Conv(p, w1, b1)
        y = Relu(c0)"""
    return _model(body, opset, inits, outputs="float[1,8,8,8] y"), (1, 4, 8, 8)


def reducemean_squeeze(opset):
    # ReduceMean / Squeeze / Unsqueeze: ``axes`` attribute (ReduceMean below
    # opset 18, Squeeze / Unsqueeze below 13)
    rng = np.random.default_rng(10)
    inits = _conv_weights(rng) + [
        _init(_w(rng, 8, 5), "w2"),
        _init(_w(rng, 5), "b2"),
        _init(np.array([1, 8], np.int64), "shp"),
    ]
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        rm = ReduceMean<axes=[2,3], keepdims=1>(c0)
        sq = Squeeze<axes=[2,3]>(rm)
        us = Unsqueeze<axes=[0]>(sq)
        f = Reshape(us, shp)
        y = Gemm(f, w2, b2)"""
    return _model(body, opset, inits, outputs="float[1,5] y"), (1, 4, 8, 8)


def slice_conv(opset):
    # Slice: attributes at opset 9, inputs from 10 on
    rng = np.random.default_rng(11)
    inits = _conv_weights(rng)
    if opset >= 10:
        sl = "s = Slice(c0, st, en, axs)"
        inits += [
            _init(np.array([0], np.int64), "st"),
            _init(np.array([4], np.int64), "en"),
            _init(np.array([1], np.int64), "axs"),
        ]
    else:
        sl = "s = Slice<starts=[0], ends=[4], axes=[1]>(c0)"
    body = f"""
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        {sl}
        y = Relu(s)"""
    return _model(body, opset, inits, outputs="float[1,4,8,8] y"), (1, 4, 8, 8)


def residual(opset):
    rng = np.random.default_rng(12)
    inits = [_init(_w(rng, 4, 4, 3, 3), "w1"), _init(_w(rng, 4), "b1")]
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        r0 = Relu(c0)
        a = Add(r0, x)
        y = Sigmoid(a)"""
    return _model(body, opset, inits, outputs="float[1,4,8,8] y"), (1, 4, 8, 8)


def resize_conv(opset):
    # Resize: no such op at opset 9 (Upsample), scales only at 10, roi + scales from 11
    rng = np.random.default_rng(13)
    inits = [_init(_w(rng, 4, 4, 3, 3), "w1"), _init(_w(rng, 4), "b1")]
    scales = _init(np.array([1, 1, 2, 2], np.float32), "sc")
    if opset >= 11:
        rs = 'r = Resize<mode="nearest">(c0, roi, sc)'
        inits += [_init(np.array([], np.float32), "roi"), scales]
    elif opset == 10:
        rs = 'r = Resize<mode="nearest">(c0, sc)'
        inits.append(scales)
    else:
        rs = 'r = Upsample<mode="nearest">(c0, sc)'
        inits.append(scales)
    body = f"""
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        {rs}
        y = Relu(r)"""
    return _model(body, opset, inits, outputs="float[1,4,16,16] y"), (1, 4, 8, 8)


def bn_conv(opset):
    rng = np.random.default_rng(14)
    inits = _conv_weights(rng) + [
        _init(np.abs(_w(rng, 8)) + 0.5, "g"),
        _init(_w(rng, 8), "be"),
        _init(_w(rng, 8), "mu"),
        _init(np.abs(_w(rng, 8)) + 0.5, "var"),
    ]
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        n = BatchNormalization<epsilon=1e-5>(c0, g, be, mu, var)
        y = Relu(n)"""
    return _model(body, opset, inits, outputs="float[1,8,8,8] y"), (1, 4, 8, 8)


def eltwise(opset):
    rng = np.random.default_rng(15)
    inits = _conv_weights(rng) + [
        _init(_w(rng, 1, 8, 1, 1), "m"),
        _init(_w(rng, 1, 8, 1, 1), "a"),
    ]
    body = """
        c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
        t = Mul(c0, m)
        u = Add(t, a)
        s = Sigmoid(u)
        l = LeakyRelu<alpha=0.1>(c0)
        z = Sub(s, l)
        y = Tanh(z)"""
    return _model(body, opset, inits, outputs="float[1,8,8,8] y"), (1, 4, 8, 8)


def convt(opset):
    rng = np.random.default_rng(18)
    inits = [
        _init(_w(rng, 4, 4, 2, 2), "w1"),
        _init(_w(rng, 4), "b1"),
        _init(_w(rng, 4, 4, 3, 3), "w2"),
        _init(_w(rng, 4), "b2"),
    ]
    body = """
        c0 = ConvTranspose<kernel_shape=[2,2], strides=[2,2]>(x, w1, b1)
        r = Relu(c0)
        y = Conv<pads=[1,1,1,1]>(r, w2, b2)"""
    return _model(body, opset, inits, outputs="float[1,4,16,16] y"), (1, 4, 8, 8)


MODELS = {
    f.__name__: f
    for f in (
        conv_relu_pool,
        mlp,
        matmul_add,
        clip6,
        clip_generic,
        softmax,
        concat_split,
        reshape_transpose,
        pool,
        pad_conv,
        pad_value,
        reducemean_squeeze,
        slice_conv,
        residual,
        resize_conv,
        bn_conv,
        eltwise,
        convt,
    )
}


# -- helpers ----------------------------------------------------------------------


def _data(shape, n=4, seed=3):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


class _Reader:
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        return next(self.it, None)


def _quantize(model, preset, shape, extra=None):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data(shape))
        )


def _run(model, x):
    so = ort.SessionOptions()
    # (no graph optimizations: fused integer kernels depend on the host CPU)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


def _default_opset(model):
    return next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))


def _qdq(model):
    return [
        n
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    ]


def _op_types(model):
    return [n.op_type for n in model.graph.node]


# -- quantize_full_qdq ------------------------------------------------------------


@pytest.mark.parametrize("opset", OPSETS)
@pytest.mark.parametrize("name", ["conv_relu_pool", "mlp", "concat_split", "pad_conv"])
def test_per_tensor_full_qdq_keeps_the_opset_and_has_no_axis(name, opset):
    model, shape = MODELS[name](opset)
    q = quantize_full_qdq(model, _data(shape), per_channel=False)
    assert _default_opset(q) == opset  # quantized in place, never converted
    qdq = _qdq(q)
    assert qdq and all(n.domain == "" for n in qdq)
    assert not any(a.name == "axis" for n in qdq for a in n.attribute)
    inits = {t.name: numpy_helper.to_array(t) for t in q.graph.initializer}
    for n in qdq:  # per-tensor: scalar (or one-element) scale and zero point
        assert inits[n.input[1]].size == 1 and inits[n.input[2]].size == 1
    if opset >= 10:  # (a Q/DQ node is an opset 10 operator)
        x = _data(shape)[0]["x"]
        want = _run(model, x)
        got = _run(q, x)
        assert got.shape == want.shape
        assert np.abs(got - want).max() < 0.1 * np.abs(want).max() + 1e-6


@pytest.mark.parametrize("opset", OPSETS)
def test_per_channel_full_qdq_raises_quarks_error_below_opset_13(opset):
    model, shape = conv_relu_pool(opset)
    with pytest.raises(ValueError, match=QUARK_ERROR.replace(".", r"\.")):
        quantize_full_qdq(model, _data(shape), per_channel=True)


def test_per_channel_full_qdq_still_works_at_opset_13():
    model, shape = conv_relu_pool(13)
    q = quantize_full_qdq(model, _data(shape), per_channel=True)
    axes = [a.i for n in _qdq(q) for a in n.attribute if a.name == "axis"]
    assert axes and set(axes) == {0}


@pytest.mark.parametrize("opset", [12, 13])
def test_int32_bias_dequantizer_form_follows_the_weight_granularity(opset):
    model, shape = mlp(opset)

    def bias_dq(per_channel):
        q = quantize_full_qdq(model, _data(shape), per_channel=per_channel)
        inits = {t.name: numpy_helper.to_array(t) for t in q.graph.initializer}
        return next(
            (n, inits[n.input[1]], inits[n.input[2]])
            for n in q.graph.node
            if n.op_type == "DequantizeLinear"
            and n.input[0] in inits
            and inits[n.input[0]].dtype == np.int32
        )

    node, scale, zp = bias_dq(False)  # Quark: a one-element scale, scalar zero point
    assert scale.shape == (1,) and zp.shape == () and not node.attribute
    if opset >= 13:
        node, scale, zp = bias_dq(True)
        assert scale.shape == (32,) and zp.shape == (32,)
        assert [a.i for a in node.attribute] == [0]


# -- the compat presets -----------------------------------------------------------

_INT_PRESETS = ["A8W8", "U8S8_AAWS", "S8S8_AAWS", "XINT8", "INT8_CNN_DEFAULT", "A16W8"]


@pytest.mark.parametrize("opset", OPSETS)
@pytest.mark.parametrize("preset", _INT_PRESETS)
def test_presets_quantize_in_place_and_register_quarks_operator_sets(preset, opset):
    model, shape = conv_relu_pool(opset)
    q = _quantize(model, preset, shape)
    assert _default_opset(q) == opset
    domains = {o.domain for o in q.opset_import}
    assert {"com.microsoft", "com.amd.quark"} <= domains
    # Quark's extended quantizer (A8W8 and the 16-bit presets) ends with its
    # custom Q/DQ converted to the com.microsoft ones, the plain one stays standard
    ms = preset in ("A8W8", "A16W8")
    assert {n.domain for n in _qdq(q)} == {"com.microsoft" if ms else ""}
    if opset >= 10 or ms:  # (a default-domain Q/DQ node is an opset 10 operator)
        x = _data(shape)[0]["x"]
        assert _run(q, x).shape == _run(model, x).shape


@pytest.mark.parametrize("opset", [10, 11, 12])
def test_maxpool_is_quantized_from_opset_12_only(opset):
    model, shape = conv_relu_pool(opset)
    q = _quantize(model, "U8S8_AAWS", shape)
    pool_out = next(n.output[0] for n in q.graph.node if n.op_type == "MaxPool")
    quantized = any(
        n.op_type == "QuantizeLinear" and n.input[0] == pool_out for n in q.graph.node
    )
    assert quantized == (opset >= 12)


@pytest.mark.parametrize("opset", [10, 11])
def test_resize_is_quantized_from_opset_11_only(opset):
    model, shape = resize_conv(opset)
    q = _quantize(model, "U8S8_AAWS", shape)
    producers = {o: n for n in q.graph.node for o in n.output}
    # (the graph output is the Relu's below opset 11, the Resize's output gets
    # a Q/DQ pair from there on)
    assert (producers["y"].op_type == "DequantizeLinear") == (opset >= 11)


@pytest.mark.parametrize("opset", OPSETS)
@pytest.mark.parametrize("preset", ["U8S8_AAWS", "A8W8", "INT8_CNN_DEFAULT"])
def test_per_channel_option_raises_quarks_error(preset, opset):
    model, shape = conv_relu_pool(opset)
    with pytest.raises(ValueError, match=QUARK_ERROR.replace(".", r"\.")):
        _quantize(model, preset, shape, {"PerChannel": True})


@pytest.mark.parametrize("opset", [11, 13])
def test_xint8_refuses_per_channel_like_quarks_npu_quantizer(opset):
    model, shape = conv_relu_pool(opset)
    with pytest.raises(ValueError, match="Only per-tensor quantization is supported"):
        _quantize(model, "XINT8", shape, {"PerChannel": True})


@pytest.mark.parametrize("opset", [11, 12])
def test_gptq_does_not_need_per_channel_dequantizers(opset):
    from onnxsim.quark_compat import GPTQConfig

    model, shape = mlp(opset)
    cfg = qc.QConfig.get_default_config("A8W8")
    cfg.algo_config = [GPTQConfig(per_channel=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data(shape))
        )
    assert not any(a.name == "axis" for n in _qdq(q) for a in n.attribute)
    x = _data(shape)[0]["x"]
    assert np.abs(_run(q, x) - _run(model, x)).max() < 0.2


@pytest.mark.parametrize("opset", [10, 11, 12])
def test_xint8_converts_split_with_the_split_attribute_to_slice(opset):
    model, shape = concat_split(opset)
    ops = _op_types(_quantize(model, "XINT8", shape))
    assert "Split" not in ops and ops.count("Slice") == 2


@pytest.mark.parametrize("opset", OPSETS)
def test_xint8_converts_reducemean_with_the_axes_attribute(opset):
    # ReduceMean(axes=[2,3]) -> GlobalAveragePool, whose DPU compensation needs the
    # shapes (inferred on the float graph: its Q/DQ nodes are opset 10 operators)
    model, shape = reducemean_squeeze(opset)
    ops = _op_types(_quantize(model, "XINT8", shape))
    assert "ReduceMean" not in ops and "GlobalAveragePool" in ops
    assert "Mul" in ops


@pytest.mark.parametrize("opset", OPSETS)
def test_clip6_pair_is_dropped_only_for_clip_with_bound_inputs(opset):
    # Quark's RemoveQDQConvClip reads the bounds from the Clip's inputs (opset >= 11);
    # the min / max attributes of older Clips are not recognised and the pair stays
    model, shape = clip6(opset)
    q = _quantize(model, "A8W8", shape)
    clip = next(n for n in q.graph.node if n.op_type == "Clip")
    producers = {o: n for n in q.graph.node for o in n.output}
    assert (producers[clip.input[0]].op_type == "Conv") == (opset >= 11)


@pytest.mark.parametrize("opset", [9, 10, 11])
def test_dynamic_quantization_below_opset_11_computes_scale_and_zero_point(opset):
    model, shape = mlp(opset)
    q = _quantize(model, "UINT8_DYNAMIC_QUANT", shape)
    ops = _op_types(q)
    assert _default_opset(q) == opset
    assert ("DynamicQuantizeLinear" in ops) == (opset >= 11)
    if opset < 11:
        # ReduceMin / ReduceMax / Sub / Div / Floor / Cast + QuantizeLinear, per input
        assert ops.count("ReduceMin") == ops.count("QuantizeLinear") == 2
        assert ops.count("Floor") == 2 and "MatMulInteger" in ops
    if opset >= 10:  # (MatMulInteger and QuantizeLinear are opset 10 operators)
        x = _data(shape)[0]["x"]
        want = _run(model, x)
        assert np.abs(_run(q, x) - want).max() < 0.05 * np.abs(want).max()


@pytest.mark.parametrize("target", [13, 17])
def test_convert_opset_version_option_converts_before_quantizing(target):
    # Quark's pre-processing converts first when asked, which also opens the
    # per-channel mode that a model below opset 13 cannot have
    model, shape = conv_relu_pool(11)
    q = _quantize(
        model, "U8S8_AAWS", shape, {"ConvertOpsetVersion": target, "PerChannel": True}
    )
    assert _default_opset(q) == target
    assert 0 in {a.i for n in _qdq(q) for a in n.attribute if a.name == "axis"}


def test_failed_opset_conversion_is_skipped_with_a_warning():
    model, shape = softmax(11)
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.extra_options["ConvertOpsetVersion"] = 99
    with pytest.warns(UserWarning, match="skipping the conversion"):
        q = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data(shape))
        )
    assert _default_opset(q) == 11
