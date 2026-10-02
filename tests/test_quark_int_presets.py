"""The integer presets other than XINT8 (A8W8, A16W8, VINT8, U8S8_AAWS, S8S8_AAWS,
INT8_CNN_DEFAULT, INT8_TRANSFORMER_DEFAULT, ...) -- the properties of the Quark
behaviours ``test_quark_int_presets_parity.py`` checks against the real package,
without needing it: models are built with ``onnx.parser`` and quantized by
:mod:`onnxsim.quark_compat`; ONNX Runtime runs them with its graph optimizations off.
"""

import re
import warnings
import zlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc

_POOL = "MaxPool<kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]>"


def _init(name, scale=1.0):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    shape = {"w": (8, 8, 1, 1), "b": (8,)}[name[0]]
    return (
        rng.standard_normal(shape) * (0.5 if name[0] == "b" else 0.4) * scale
    ).astype(np.float32)


def _model(body, shape=(1, 8, 8, 8), **over):
    """``w<k>`` / ``b<k>`` names in ``body`` get a deterministic initializer
    (``over[name]`` replaces it: a factor or an array)."""
    dims = ",".join(map(str, shape))
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g (float[{dims}] x) => (float y) {{'
        + body
        + "}"
    )
    consts = dict(
        lo=np.array(0.0, np.float32),
        hi6=np.array(6.0, np.float32),
        hi1=np.array(1.0, np.float32),
        sp44=np.array([4, 4], np.int64),
        sl8=np.linspace(0.05, 0.4, 8, dtype=np.float32).reshape(8, 1, 1),
    )
    for name in dict.fromkeys(re.findall(r"\b[a-z]\w*\b", body)):
        if name in over and not np.isscalar(over[name]):
            value = over[name]
        elif re.fullmatch(r"[wb]\d+", name):
            value = _init(name, over.get(name, 1.0))
        elif name in consts:
            value = consts[name]
        else:
            continue
        m.graph.initializer.append(numpy_helper.from_array(value, name))
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return onnx.shape_inference.infer_shapes(m)


def _data(shape=(1, 8, 8, 8), n=4, seed=3):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


class _Reader:
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        return next(self.it, None)

    def reset_iter(self):
        pass


def _quantize(model, preset, data=None, extra=None):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(data or _data())
        )


def _run(model, x):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]


def _q_params(model):
    """``{float tensor a QuantizeLinear reads: (scale, zero point)}``."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear":
            t = n.input[0][:-2] if n.input[0].endswith("/f") else n.input[0]
            out[t] = (float(inits[n.input[1]]), int(inits[n.input[2]]))
    return out


def _bias_codes(model):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    return [
        inits[n.input[0]]
        for n in model.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in inits
        and inits[n.input[0]].dtype == np.int32
    ]


def _well_formed(model):
    """Every node input is a graph input, an initializer or an earlier output."""
    known = {i.name for i in model.graph.input} | {
        t.name for t in model.graph.initializer
    }
    for n in model.graph.node:
        assert all(x in known for x in n.input if x), (n.op_type, list(n.input))
        known.update(n.output)
    assert all(o.name in known for o in model.graph.output)


_ALIGNED = f"""
    c0 = Conv(x, w1, b1)
    c1 = Conv(x, w2, b2)
    p = {_POOL}(c0)
    k = Concat<axis=1>(c1, c0)
    f = Conv(k, w3, b3)
    e = Conv(p, w4, b4)
    d = Conv(c0, w5, b5)
    fe = Add(f, e)
    s = Relu(fe)
    sd = Add(s, d)
    y = Conv(sd, w6, b6)
"""
_ALIGNED_INITS = dict(w2=4.0, w3=np.tile(_init("w3"), (1, 2, 1, 1)))


@pytest.mark.parametrize("preset", ["A8W8", "A16W8"])
def test_a_pool_output_moves_with_its_inputs_concat_alignment(preset):
    q = _quantize(_model(_ALIGNED, **_ALIGNED_INITS), preset)
    params = _q_params(q)
    assert params["c0"] == params["p"] == params["k"]
    assert params["c1"] == params["k"]
    # (without the alignment the Concat input keeps its own range)
    plain = _q_params(
        _quantize(
            _model(_ALIGNED, **_ALIGNED_INITS), preset, extra={"AlignConcat": False}
        )
    )
    assert plain["c0"] == plain["p"] != plain["k"]


@pytest.mark.parametrize("preset", ["A8W8", "A16W8"])
def test_the_int32_bias_is_requantized_with_truncation_when_the_scale_moves(preset):
    model = _model(_ALIGNED, **_ALIGNED_INITS)
    aligned = _quantize(model, preset)
    plain = _quantize(model, preset, extra={"AlignConcat": False})
    # the Conv reading c0 (w5, b5): bias scale = input scale * weight scale, both ways
    s_old, s_new = _q_params(plain)["c0"][0], _q_params(aligned)["c0"][0]
    assert s_old != s_new
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    by_out = {o: n for n in aligned.graph.node for o in n.output}
    conv = next(
        n for n in aligned.graph.node if n.op_type == "Conv" and "c0" == n.input[0]
    )
    bias_dq = by_out[conv.input[2]]
    a_inits = {t.name: numpy_helper.to_array(t) for t in aligned.graph.initializer}
    w_scale = a_inits[by_out[conv.input[1]].input[1]]
    got = a_inits[bias_dq.input[0]]
    bias = inits["b5"].astype(np.float64)
    s0 = (np.float32(s_old) * w_scale).astype(np.float32)
    s1 = (np.float32(s_new) * w_scale).astype(np.float32)
    first = np.round(bias / np.float64(s0)).astype(np.int32)
    np.testing.assert_array_equal(got, (first / (s1 / s0)).astype(np.int32))
    np.testing.assert_array_equal(a_inits[bias_dq.input[1]], s1)


@pytest.mark.parametrize("preset", ["A8W8", "S8S8_AAWS"])
def test_a_bias_beyond_the_int32_range_saturates_at_the_int32_limits(preset):
    big = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    q = _quantize(
        _model(
            "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n y = Conv(r, w2, b2)",
            w1=_init("w1", 1e-7),
            b1=big,
        ),
        preset,
    )
    codes = np.concatenate([c.ravel() for c in _bias_codes(q)])
    assert codes.min() == -(2**31) and codes.max() == 2**31 - 1


@pytest.mark.parametrize("preset", ["S8S8_AAWS", "INT8_CNN_DEFAULT", "U8S8_AAWS"])
def test_an_all_zero_activation_is_scale_one_zero_point_zero(preset):
    model = _model(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n y = Conv(r, w2, b2)",
        b1=np.full(8, -60.0, np.float32),
    )
    q = _quantize(model, preset)
    assert (1.0, 0) in _q_params(q).values()
    _well_formed(q)
    _run(q, _data()[0]["x"])


@pytest.mark.parametrize("preset", ["U8S8_AAWS", "INT8_CNN_DEFAULT"])
def test_two_clips_in_a_row_fold_into_the_producer(preset):
    model = _model(
        "c0 = Conv(x, w1, b1)\n c = Clip(c0, lo, hi6)\n d = Clip(c, lo, hi6)\n"
        " y = Conv(d, w2, b2)"
    )
    q = _quantize(model, preset)
    _well_formed(q)
    assert "Clip" not in [n.op_type for n in q.graph.node]
    scale, zp = _q_params(q)["d"]
    assert (
        zp == 0 and 0 < scale * 255 <= 6.0 + 1e-4
    )  # (the observed range of the Clips)
    x = _data()[0]["x"]
    assert np.abs(_run(q, x) - _run(model, x)).max() < 0.5


def test_align_eltwise_quant_type_gives_a_pool_feeding_an_add_its_own_range():
    body = (
        f"c0 = Conv(x, w1, b1)\n p = {_POOL}(c0)\n a = Add(p, c0)\n y = Conv(a, w2, b2)"
    )
    model = _model(body, b1=np.full(8, -3.0, np.float32))
    shared = _q_params(_quantize(model, "A8W8"))
    own = _q_params(_quantize(model, "A16W8"))
    assert shared["p"] == shared["c0"]  # (no override without the option)
    assert own["p"] != own["c0"]


_VINT8_X = """
    l = LeakyRelu<alpha=0.1>(x)
    a, b = Split<axis=1>(x, sp44)
    c = Concat<axis=1>(b, a)
    d = Conv(c, w1, b1)
    m = Min(x, d)
    ml = Add(m, l)
    y = Conv(ml, w2, b2)
"""


def test_vint8_slices_from_a_split_read_the_float_input_and_readers_get_a_pair_each():
    q = _quantize(_model(_VINT8_X), "VINT8")
    _well_formed(q)
    slices = [n for n in q.graph.node if n.op_type == "Slice"]
    assert len(slices) == 2 and all(s.input[0] == "x" for s in slices)
    # LeakyRelu and Min each read their own Q/DQ pair of the graph input
    pairs = [
        n for n in q.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == "x"
    ]
    assert len(pairs) == 2


def test_vint8_gives_each_input_slot_of_a_quantized_reader_a_pair():
    q = _quantize(
        _model(
            "k = Concat<axis=1>(x, x)\n s = Sigmoid(x)\n d = Conv(k, wk, bk)\n"
            " ds = Add(d, s)\n y = Conv(ds, w2, b2)",
            wk=np.tile(_init("w1"), (1, 2, 1, 1)),
            bk=_init("b1"),
        ),
        "VINT8",
    )
    _well_formed(q)
    pairs = [
        n for n in q.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == "x"
    ]
    assert len(pairs) == 3  # (the Concat reads x twice: its second pair has no reader)


@pytest.mark.parametrize("slope", [None, "sl8"])
def test_vint8_quantizes_the_prelu_slope(slope):
    if slope is None:
        slope = "sl1"
        extra = dict(sl1=np.array([0.25], np.float32))
    else:
        extra = {}
    q = _quantize(
        _model(
            f"c0 = Conv(x, w1, b1)\n r = PRelu(c0, {slope})\n y = Conv(r, w2, b2)",
            **extra,
        ),
        "VINT8",
    )
    by_out = {o: n for n in q.graph.node for o in n.output}
    prelu = next(n for n in q.graph.node if n.op_type == "PRelu")
    assert by_out[prelu.input[1]].op_type == "DequantizeLinear"


def test_vint8_does_not_force_ops_on_an_unmarked_input_to_quantize():
    model = _model(
        f"r = Relu(x)\n p = {_POOL}(x)\n c = Conv(r, w1, b1)\n d = Conv(p, w2, b2)\n"
        " cd = Add(c, d)\n y = Conv(cd, w3, b3)"
    )
    q = _quantize(model, "VINT8")
    ops = {n.op_type: n for n in q.graph.node}
    assert ops["Relu"].input[0] == ops["MaxPool"].input[0] == "x"
    # ... while A8W8, which forces them to, puts a Q/DQ pair in front
    forced = _quantize(model, "A8W8")
    assert "x" in _q_params(forced)


def test_vint8_marks_what_a_hard_sigmoid_with_other_constants_reads():
    q = _quantize(
        _model(
            "r = Relu(x)\n h = HardSigmoid<alpha=0.2, beta=0.5>(r)\n y = Conv(h, w1, b1)"
        ),
        "VINT8",
    )
    assert "r" in _q_params(q) and "x" not in _q_params(q)


def _gemm_model(body):
    rng = np.random.default_rng(0)
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float[4,16] x) => (float y) {'
        + body
        + "}"
    )
    shapes = {"w1": (16, 16), "b1": (16,), "w2": (16, 4), "b2": (4,)}
    for name in dict.fromkeys(re.findall(r"\b[wb]\d\b", body)):
        m.graph.initializer.append(
            numpy_helper.from_array(
                (rng.standard_normal(shapes[name]) * 0.5).astype(np.float32), name
            )
        )
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return onnx.shape_inference.infer_shapes(m)


def test_a_softmax_outside_the_transformer_scheme_keeps_its_observed_range():
    """The (0, 1) range of a Softmax output is applied to the Softmax nodes the
    quantizer quantizes; the transformer scheme quantizes Gemm / MatMul only."""
    model = _gemm_model("s = Softmax<axis=-1>(x)\n y = Gemm(s, w1, b1)")
    q = _quantize(model, "INT8_TRANSFORMER_DEFAULT", _data((4, 16)))
    scale = _q_params(q)["s"][0]
    assert scale < 1 / 255 - 1e-6


def test_the_batch_extremes_are_averaged_in_float32():
    """``CalibMovingAverage`` (the transformer preset): ``np.nanmean`` of the
    per-batch maxima in float32, which is not the float64 mean of the same numbers
    (2.1265609264... against 2.1265608072...)."""
    from onnxsim.calibration import calibrate

    maxima = np.array(
        [2.5566258, 2.4513779, 2.5653403, 1.6667643, 0.9052895, 2.664465, 2.076063],
        np.float32,
    )
    data = []
    for v in maxima:
        x = np.zeros((4, 16), np.float32)
        x[1, 3] = v
        data.append({"x": x})
    model = _gemm_model("y = Relu(x)")
    ranges = calibrate(model, data, method="minmax_mean", extra_tensor_names=["y"])
    assert ranges["y"] == (0.0, float(np.nanmean(maxima)))
    assert ranges["y"][1] == 2.126560926437378
    assert ranges["y"][1] != float(maxima.astype(np.float64).mean())


def test_a_relu_clip_chain_propagates_the_range_two_steps():
    """Quark runs ONNX Runtime's ``adjust_tensor_ranges`` twice: the input of
    ``Relu -> Clip(0, 1)`` takes the range of the Clip's output."""
    model = _model(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n k = Clip(r, lo, hi1)\n y = Conv(k, w2, b2)",
        w1=20.0,
        b1=np.full(8, 2.0, np.float32),
    )
    params = _q_params(_quantize(model, "VINT8"))
    assert params["c0"] == params["k"]  # (one step: the Relu's range, reaching past 1)


def _dq_of(model, name):
    by_out = {o: n for n in model.graph.node for o in n.output}
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    dq = by_out[name]
    return dq, inits[dq.input[0]], inits[dq.input[1]], inits[dq.input[2]]


@pytest.mark.parametrize("preset", ["A16W8", "A8W8"])
def test_asymmetric_eltwise_constants_clip_to_the_symmetric_code_range(preset):
    model = _model(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n m = Mul(s, cm)\n y = Conv(m, w2, b2)",
        cm=np.linspace(-1.0, 3.0, 8, dtype=np.float32).reshape(1, 8, 1, 1),
    )
    q = _quantize(
        model, preset, extra={"WeightSymmetric": False, "AlignEltwiseQuantType": True}
    )
    mul = next(n for n in q.graph.node if n.op_type == "Mul")
    _, codes, _, zp = _dq_of(q, mul.input[1])
    assert codes.min() == -np.iinfo(codes.dtype).max and zp != 0


def test_align_eltwise_quant_type_is_ignored_outside_the_extended_quantizer():
    model = _model(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n m = Mul(s, cm)\n y = Conv(m, w2, b2)",
        cm=np.linspace(-1.0, 3.0, 8, dtype=np.float32).reshape(1, 8, 1, 1),
    )
    on = _quantize(model, "S8S8_AAWS", extra={"AlignEltwiseQuantType": True})
    off = _quantize(model, "S8S8_AAWS")
    assert on.SerializeToString() == off.SerializeToString()


def test_int8_bias_per_channel_has_a_scale_for_every_element():
    model = _model("c0 = Conv(x, w1, b1)\n y = Conv(c0, w2, b2)")
    q = _quantize(model, "A8W8", extra={"Int32Bias": False, "PerChannel": True})
    conv = [n for n in q.graph.node if n.op_type == "Conv"][0]
    dq, codes, scale, zp = _dq_of(q, conv.input[2])
    assert codes.dtype == np.int8 and scale.shape == (8,) == zp.shape
    assert [a.i for a in dq.attribute if a.name == "axis"] == [0]
    assert (np.abs(codes) == 127).all()  # (each element is its own channel)


def test_int8_bias_asymmetric_has_a_zero_point():
    model = _model(
        "c0 = Conv(x, w1, b1)\n y = Conv(c0, w2, b2)", b1=np.full(8, 1.0, np.float32)
    )
    q = _quantize(model, "A8W8", extra={"Int32Bias": False, "WeightSymmetric": False})
    conv = [n for n in q.graph.node if n.op_type == "Conv"][0]
    _, codes, scale, zp = _dq_of(q, conv.input[2])
    assert codes.dtype == np.int8 and int(zp) != 0


def test_vint8_asymmetric_weights_take_the_power_of_two_zero_point():
    model = _model("c0 = Conv(x, w1, b1)\n y = Conv(c0, w2, b2)", w1=1.5, b1=1.5)
    q = _quantize(model, "VINT8", extra={"WeightSymmetric": False})
    conv = [n for n in q.graph.node if n.op_type == "Conv"][0]
    for k in (1, 2):
        _, codes, scale, zp = _dq_of(q, conv.input[k])
        assert float(scale) == 2.0 ** round(np.log2(float(scale)))
        assert int(zp) != 0 and codes.min() >= -127


def test_per_channel_gives_a_prelu_slope_a_scale_per_row():
    model = _model("c0 = Conv(x, w1, b1)\n r = PRelu(c0, sl8)\n y = Conv(r, w2, b2)")
    q = _quantize(model, "A8W8", extra={"PerChannel": True})
    prelu = next(n for n in q.graph.node if n.op_type == "PRelu")
    dq, codes, scale, zp = _dq_of(q, prelu.input[1])
    assert scale.shape == (8,) and [a.i for a in dq.attribute if a.name == "axis"] == [
        0
    ]
    # ... but not under the power-of-two schemes (VINT8)
    q = _quantize(model, "VINT8", extra={"PerChannel": True})
    prelu = next(n for n in q.graph.node if n.op_type == "PRelu")
    _, _, scale, _ = _dq_of(q, prelu.input[1])
    assert scale.shape == ()


def test_flatten_keeps_its_own_range_when_a_pool_alignment_moves_its_input():
    model = _model(
        "c0 = Conv(x, w1, b1)\n g = GlobalAveragePool(c0)\n f = Flatten(g)\n y = Gemm(f, wg, bg)",
        wg=np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4),
        bg=np.linspace(-0.5, 0.5, 4, dtype=np.float32),
    )
    q = _quantize(model, "A8W8", extra={"AlignPool": True, "QuantizeAllOpTypes": True})
    params = _q_params(q)
    assert params["g"] == params["c0"]  # (AlignPool: the pool takes its input's)
    assert "f" in params and params["f"] != params["g"]
