"""Parity of onnxsim's XINT8 (power-of-two) flow against the real AMD Quark ONNX
package: the BiasCorrection scale quirk and the NPU graph rewrites Quark's
``XINT8`` preset runs. Skipped unless ``quark.onnx`` is importable.

Each test runs Quark and onnxsim on the same parser-built graph and compares the
emitted graphs node by node (op types, attributes, scales, zero points,
constants) and, with ONNX Runtime's graph optimizations off, what they compute.
"""

import contextlib
import copy
import io
import os
import warnings

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_bias_correction as qbc  # noqa: E402
from onnxsim import quark_compat as qc  # noqa: E402


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files into the current directory."""
    monkeypatch.chdir(tmp_path)


# -- helpers --------------------------------------------------------------------


def _data(shape, n=4, seed=3, name="x"):
    rng = np.random.default_rng(seed)
    return [{name: rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


def _reader(data):
    from onnxruntime.quantization import CalibrationDataReader

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

        def reset_iter(self):
            self.it = iter(data)

    return R()


def _named(model):
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return onnx.shape_inference.infer_shapes(model)


def _xint8_quark_config(algos=(), extra=None):
    from quark.onnx import QConfig, QLayerConfig, XInt8Spec

    return QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        algo_config=list(algos),
        extra_options={"ForceQuantizeNoInputCheck": True, **(extra or {})},
    )


def _quark(model, data, tmp_path, algos=(), extra=None):
    from quark.onnx import ModelQuantizer

    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ModelQuantizer(_xint8_quark_config(algos, extra)).quantize_model(
            src, dst, _reader(data)
        )
    return onnx.load(dst)


def _mine(model, data, algos=(), extra=None, quiet=True):
    cfg = qc.QConfig(
        global_config=qc.QLayerConfig(activation=qc.XInt8Spec(), weight=qc.XInt8Spec()),
        algo_config=list(algos),
        extra_options=dict(extra or {}),
    )
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(data)
        )


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


def _bias_codes(model):
    """Per bias DequantizeLinear in node order: (codes, scale, zero point)."""
    inits = _inits(model)
    out = []
    for n in model.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            if q.ndim == 1:
                # (an int32 bias carries one scale per element in onnxsim, a
                # single per-tensor one in Quark; compare the distinct values)
                scale = sorted(set(inits[n.input[1]].ravel().tolist()))
                zp = sorted(set(inits[n.input[2]].ravel().tolist()))
                out.append((q.tolist(), scale, zp))
    return out


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _with(text, inits):
    m = parser.parse_model(text)
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    return _named(m)


# -- BiasCorrection: Quark's power-of-two bias scale quirk --------------------------


def _bc_conv(seed=8):
    rng = np.random.default_rng(seed)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,4,8,8] y) {
            c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
            r0 = Relu(c0)
            c1 = Conv<group=1>(r0, w1, b1)
            r1 = Relu(c1)
            y = Conv<group=1>(r1, w2, b2)
        }""",
        dict(
            w0=_w(rng, 8, 3, 3, 3),
            b0=_w(rng, 8, scale=2),
            w1=_w(rng, 8, 8, 1, 1),
            b1=_w(rng, 8),
            w2=_w(rng, 4, 8, 1, 1),
            b2=_w(rng, 4, scale=0.1),
        ),
    ), (1, 3, 8, 8)


def _bc_mlp(seed=9):
    rng = np.random.default_rng(seed)
    return _with(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }""",
        dict(w1=_w(rng, 16, 32), b1=_w(rng, 32), w2=_w(rng, 32, 8), b2=_w(rng, 8)),
    ), (3, 16)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_pof2_requantization_matches_quarks_quantize_data(dtype, symmetric, seed):
    """The re-derivation Quark's BiasCorrection runs on a corrected bias
    (power-of-two scale + MinMSE over five candidates + zero point + clipped
    codes) -- codes, scale and zero point bit for bit."""
    from onnx import TensorProto
    from quark.onnx.calibration import PowerOfTwoMethod
    from quark.onnx.quantization.quant_utils import quantize_data

    rng = np.random.default_rng(seed)
    for scale in (0.01, 0.3, 4.0):
        data = (rng.standard_normal(rng.integers(1, 40)) * scale).astype(np.float32)
        if seed == 3:
            data[0] = 17 * scale  # an outlier
        qtype = {"int8": TensorProto.INT8, "int16": TensorProto.INT16}[dtype]
        *_, zp, sc, codes = quantize_data(
            data, qtype, symmetric, method=PowerOfTwoMethod.MinMSE
        )
        got, got_scale, got_zp = qbc.quark_pof2_quantize(data, dtype, symmetric)
        np.testing.assert_array_equal(got, codes)
        assert np.float32(got_scale) == np.float32(sc)
        assert got_zp == int(zp)


@pytest.mark.parametrize("build", [_bc_conv, _bc_mlp])
def test_xint8_bias_correction_writes_quarks_integer_biases(build, tmp_path):
    """XINT8 + BiasCorrection: the corrected int8 bias codes -- re-derived
    through the power-of-two quantizer, scale left as stored -- equal Quark's."""
    model, shape = build()
    data = _data(shape)
    algos = [quark_onnx.BiasCorrectionConfig()]
    q = _quark(model, data, tmp_path, algos)
    m = _mine(model, data, [qc.BiasCorrectionConfig()])
    plain = _mine(model, data)
    assert _bias_codes(m) == _bias_codes(q)
    assert _bias_codes(q) != _bias_codes(plain), "bias correction changed nothing"


def test_the_quirk_is_visible_and_can_be_switched_off(tmp_path):
    """Quark's BiasCorrection can leave a bias whose int8 codes belong to a
    different scale than the stored one. onnxsim reproduces it, warns, and with
    ``BiasCorrectionStoredScale`` writes codes for the stored scale instead
    (so the dequantized bias is the corrected float bias)."""
    model, shape = _bc_conv()
    data = _data(shape)
    q = _quark(model, data, tmp_path, [quark_onnx.BiasCorrectionConfig()])
    base = _mine(model, data)
    with pytest.warns(UserWarning, match="power-of-two flow"):
        quirk = _mine(model, data, [qc.BiasCorrectionConfig()], quiet=False)
    assert _bias_codes(quirk) == _bias_codes(q)
    consistent = _mine(
        model,
        data,
        [qc.BiasCorrectionConfig()],
        {"BiasCorrectionStoredScale": True},
    )
    # same stored scales, but the codes follow the corrected float bias
    for (cq, sq, _), (cb, sb, _), (cc, sc, _) in zip(
        _bias_codes(quirk), _bias_codes(base), _bias_codes(consistent)
    ):
        assert sq == sb == sc
    assert _bias_codes(consistent) != _bias_codes(quirk)
    # the last layer is where the stored scale and the fresh one disagree
    assert (
        max(
            abs(np.array(a) - np.array(b)).max()
            for (a, _, _), (b, _, _) in zip(_bias_codes(quirk), _bias_codes(consistent))
        )
        > 1
    )


def test_xint8_bias_correction_from_the_same_quantized_model(tmp_path):
    """The algorithm alone: Quark's ``bias_correction`` and onnxsim's on the
    same Quark-quantized XINT8 model (symmetric and asymmetric)."""
    from onnxruntime.quantization import QuantType
    from quark.onnx.algorithm.bc.bias_correction import bias_correction
    from quark.onnx.calibration import CachedDataReader, PowerOfTwoMethod

    model, shape = _bc_conv(11)
    data = _data(shape, seed=5)
    quant = _quark(model, data, tmp_path)
    for symmetric in (True, False):
        theirs = bias_correction(
            copy.deepcopy(model),
            copy.deepcopy(quant),
            False,
            CachedDataReader(_reader(data), None),
            QuantType.QInt8,
            PowerOfTwoMethod.MinMSE,
            {"ActivationSymmetric": symmetric},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = qbc.correct_bias_quark(
                model, quant, data, activation_symmetric=symmetric, method="pof2"
            )
        assert _bias_codes(ours) == _bias_codes(theirs), symmetric


def test_xint8_int32_bias_correction_follows_quarks_requantization(tmp_path):
    """``Int32Bias=True``: Quark's re-derivation runs on the int32 biases too
    (scale about 2**-24, zero point 1 from float32 rounding of the int32 range);
    the codes are around 1e8, so only float32 noise in the measured mean error
    separates them."""
    model, shape = _bc_conv()
    data = _data(shape)
    extra = {"Int32Bias": True}
    q = _quark(model, data, tmp_path, [quark_onnx.BiasCorrectionConfig()], extra)
    m = _mine(model, data, [qc.BiasCorrectionConfig()], extra)
    for (cq, sq, zq), (cm, sm, zm) in zip(_bias_codes(q), _bias_codes(m)):
        assert (sq, zq) == (sm, zm)
        assert max(abs(c) for c in cq) > 1e6  # the fresh 2**-24 grid
        np.testing.assert_allclose(cm, cq, rtol=3e-5, atol=1e3)


def test_xint8_preset_bias_correction_matches_quark(tmp_path):
    """The same through the ``XINT8`` preset (uint8 activations, the legacy
    ``BiasCorrection`` option on Quark's side)."""
    model, shape = _bc_conv(5)
    data = _data(shape)
    q = _quark_preset(model, data, tmp_path, extra={"BiasCorrection": True})
    m = _mine_preset(model, data, algos=[qc.BiasCorrectionConfig()])
    base = _mine_preset(model, data)
    assert _bias_codes(m) == _bias_codes(q)
    assert _bias_codes(q) != _bias_codes(base)


# == the NPU graph rewrites ==================================================================
#
# Quark's XINT8 quantizer simulates the DPU (AveragePool / HardSigmoid ``Mul``,
# Sigmoid -> HardSigmoid, LeakyRelu alpha) and moves the power-of-two positions
# of the Q/DQ pairs to meet the compiler's limits; the float graph is first
# converted (BN folding / conversion, ReduceMean -> GlobalAveragePool, Split ->
# Slice, Pad fusion, HardSwish inlining). Each graph below is quantized by both
# and the results compared node for node -- op types, attributes, every scale,
# zero point and constant, and the wiring -- then run in ONNX Runtime.

import hashlib  # noqa: E402
import re  # noqa: E402


def _attr_key(a):
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, onnx.TensorProto):
        v = numpy_helper.to_array(v)
    if isinstance(v, bytes):
        v = v.decode()
    if hasattr(v, "tolist"):
        v = v.tolist()
    return (a.name, tuple(v) if isinstance(v, (list, tuple)) else v)


_DEFAULT_ATTRS = {
    ("auto_pad", "NOTSET"),
    ("group", 1),
    ("ceil_mode", 0),
    ("count_include_pad", 0),
    ("storage_order", 0),
}


def _node_attrs(n):
    return tuple(
        sorted(k for k in map(_attr_key, n.attribute) if k not in _DEFAULT_ATTRS)
    )


def _uniform_qdq(model):
    """Per-axis Q/DQ parameters that are uniform become per-tensor scalars (an
    int32 bias carries one scale per element in onnxsim, one in Quark)."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    inits = {t.name: t for t in out.graph.initializer}
    for n in out.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) > 2:
            s, z = inits.get(n.input[1]), inits.get(n.input[2])
            if s is None or z is None:
                continue
            sa, za = numpy_helper.to_array(s), numpy_helper.to_array(z)
            if (
                (sa.size > 1 or za.size > 1 or sa.ndim)
                and np.all(sa == sa.flat[0])
                and np.all(za == za.flat[0])
            ):
                s.CopyFrom(
                    numpy_helper.from_array(np.array(sa.flat[0], sa.dtype), s.name)
                )
                z.CopyFrom(
                    numpy_helper.from_array(np.array(za.flat[0], za.dtype), z.name)
                )
                for a in list(n.attribute):
                    if a.name == "axis":
                        n.attribute.remove(a)
    return out


def _values(model):
    """Initializers plus Constant nodes' values (onnxsim folds a Constant into an
    initializer, Quark keeps the node)."""
    vals = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            vals[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return vals


def _arr_key(a):
    a = np.ascontiguousarray(a)
    return f"{a.dtype}{list(a.shape)}:{hashlib.md5(a.tobytes()).hexdigest()[:12]}"


def _graph_keys(model):
    """Name-independent signature of every graph output (hash of the whole
    producing sub-graph) and the multiset of local node signatures."""
    model = _uniform_qdq(model)
    vals = _values(model)
    prod = {
        o: n
        for n in model.graph.node
        if not (n.op_type == "Constant" and n.output[0] in vals)
        for o in n.output
    }
    memo = {}

    def sig(t):
        if t not in memo:
            if t == "":
                memo[t] = "-"
            elif t in vals:
                memo[t] = "init:" + _arr_key(vals[t])
            elif t in prod:
                n = prod[t]
                body = f"{n.op_type}{_node_attrs(n)}#{list(n.output).index(t)}("
                body += ",".join(sig(x) for x in n.input) + ")"
                memo[t] = hashlib.md5(body.encode()).hexdigest()[:12]
            else:
                memo[t] = "in:" + t
        return memo[t]

    outs = [sig(o.name) for o in model.graph.output]
    local = []
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] in vals:
            continue
        ins = tuple(
            "I" + _arr_key(vals[x])
            if x in vals
            else "T:" + prod[x].op_type
            if x in prod
            else "in"
            for x in n.input
        )
        local.append((n.op_type, _node_attrs(n), ins))
    return outs, local


def _graph_diff(q, m):
    """``[]`` when the two graphs are the same graph, else the nodes only one of
    them has."""
    from collections import Counter

    sq, lq = _graph_keys(q)
    sm, lm = _graph_keys(m)
    if sq == sm:
        return []
    cq, cm = Counter(lq), Counter(lm)
    return [("Quark only", x) for x in (cq - cm).elements()] + [
        ("onnxsim only", x) for x in (cm - cq).elements()
    ] or [("wiring differs", None)]


def _assert_same_graph(q, m, msg=""):
    diff = _graph_diff(q, m)
    assert not diff, f"{msg}: graphs differ: {diff}"


def _ort(model, x):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    lib = os.environ.get("QUARK_ONNX_OPS_LIB")
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


def _quark_preset(
    model, data, tmp_path, preset="XINT8", extra=None, reduce_range=False
):
    """Quark's preset as ``QConfig.get_default_config`` hands it out (a private
    copy; CLE off, which the presets enable implicitly and onnxsim only runs when
    configured)."""
    from quark.onnx import ModelQuantizer, QConfig

    cfg = copy.deepcopy(QConfig.get_default_config(preset))
    cfg.global_quant_config.include_cle = False
    cfg.global_quant_config.reduce_range = reduce_range
    cfg.global_quant_config.extra_options.update(extra or {})
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ModelQuantizer(cfg).quantize_model(src, dst, _reader(data))
    return onnx.load(dst)


def _mine_preset(model, data, preset="XINT8", extra=None, algos=()):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    cfg.algo_config = list(cfg.algo_config) + list(algos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(data)
        )


SHAPE = (1, 3, 12, 12)


def _case_inits(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *s, sc=0.5: _w(rng, *s, scale=sc)  # noqa: E731
    const = lambda v: np.array(v, np.float32)  # noqa: E731
    return dict(
        w1=f(8, 3, 3, 3),
        b1=f(8),
        w2=f(2, 8, 1, 1),
        b2=f(2),
        w3=f(8, 3, 3, 3),
        b3=f(8, sc=5),
        w4=f(2, 16, 1, 1),
        b4=f(2),
        w5=f(8, 8, 1, 1),
        b5=f(8),
        w8=f(2, 4, 1, 1),
        b8=f(2),
        wt=f(3, 8, 3, 3),
        bt=f(8),
        wg=f(8, 2),
        wg1=f(8, 16),
        bg1=f(8),
        wg2=f(16, 8),
        bg2=f(8),
        wg3=f(8, 4),
        bg3=f(4),
        bg=f(2),
        bns=(1 + f(8, sc=0.3)).astype(np.float32),
        bnb=f(8),
        bnm=f(8),
        bnv=(1 + np.abs(f(8))).astype(np.float32),
        lo=const(0.0),
        hi6=const(6.0),
        hi1=const(1.0),
        lom=const(-1.0),
        sl=const([0.25]),
        pads=np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64),
        sp=np.array([4, 4], np.int64),
        starts=np.array([0], np.int64),
        ends=np.array([4], np.int64),
        axes1=np.array([1], np.int64),
        w6=f(8, 8, 1, 1),
        b6=f(8),
    )


def _build(body, overrides=None, opset=17, shape=SHAPE):
    """A parser-built model; every ``_case_inits`` name the body mentions is
    attached as an initializer (``overrides`` replace their values)."""
    inits = _case_inits()
    inits.update(overrides or {})
    m = parser.parse_model(
        f"""<ir_version: 9, opset_import: ["": {opset}]>
        g (float{list(shape)} x) => (float y) {{ {body} }}"""
    )
    for k, v in inits.items():
        if re.search(rf"\b{k}\b", body):
            m.graph.initializer.append(numpy_helper.from_array(v, k))
    return _named(m)


_CONV1 = "c0 = Conv(x, w1, b1)\n"
_POOL = "p = MaxPool<kernel_shape=[2,2],strides=[2,2]>(c0)\n"
CASES = {
    "avg3": _CONV1 + "p = AveragePool<kernel_shape=[3,3]>(c0)\n y = Conv(p, w2, b2)",
    "avg2": _CONV1
    + "p = AveragePool<kernel_shape=[2,2],strides=[2,2]>(c0)\n y = Conv(p, w2, b2)",
    "avg4": _CONV1 + "p = AveragePool<kernel_shape=[4,4]>(c0)\n y = Conv(p, w2, b2)",
    "avg5": _CONV1 + "p = AveragePool<kernel_shape=[5,5]>(c0)\n y = Conv(p, w2, b2)",
    "avg3x2": _CONV1 + "p = AveragePool<kernel_shape=[3,2]>(c0)\n y = Conv(p, w2, b2)",
    "gap": _CONV1 + "p = GlobalAveragePool(c0)\n y = Conv(p, w2, b2)",
    "gap_after_relu": _CONV1
    + "r = Relu(c0)\n p = GlobalAveragePool(r)\n y = Conv(p, w2, b2)",
    "avg_then_relu": _CONV1
    + "p = AveragePool<kernel_shape=[3,3]>(c0)\n r = Relu(p)\n y = Conv(r, w2, b2)",
    "gap_then_relu": _CONV1
    + "p = GlobalAveragePool(c0)\n r = Relu(p)\n y = Conv(r, w2, b2)",
    "avg_then_clip6": _CONV1
    + "p = AveragePool<kernel_shape=[3,3]>(c0)\n r = Clip(p, lo, hi6)\n y = Conv(r, w2, b2)",
    "maxpool": _CONV1 + _POOL + "y = Conv(p, w2, b2)",
    "maxpool_then_relu": _CONV1 + _POOL + "r = Relu(p)\n y = Conv(r, w2, b2)",
    "maxpool_then_clip1": _CONV1 + _POOL + "r = Clip(p, lo, hi1)\n y = Conv(r, w2, b2)",
    "maxpool_then_clip6": _CONV1 + _POOL + "r = Clip(p, lo, hi6)\n y = Conv(r, w2, b2)",
    "maxpool_then_clip_m1_1": _CONV1
    + _POOL
    + "r = Clip(p, lom, hi1)\n y = Conv(r, w2, b2)",
    "concat": _CONV1
    + "c1 = Conv(x, w3, b3)\n k = Concat<axis=1>(c0, c1)\n y = Conv(k, w4, b4)",
    "add": _CONV1 + "c1 = Conv(x, w3, b3)\n a = Add(c0, c1)\n y = Conv(a, w5, b5)",
    "sigmoid": _CONV1 + "s = Sigmoid(c0)\n y = Conv(s, w2, b2)",
    "swish": _CONV1 + "s = Sigmoid(c0)\n m = Mul(c0, s)\n y = Conv(m, w2, b2)",
    "hard_sigmoid": _CONV1
    + "h = HardSigmoid<alpha=0.1666667, beta=0.5>(c0)\n y = Conv(h, w2, b2)",
    "hard_swish": _CONV1 + "h = HardSwish(c0)\n y = Conv(h, w2, b2)",
    "leaky": _CONV1 + "l = LeakyRelu<alpha=0.1>(c0)\n y = Conv(l, w2, b2)",
    "leaky_0_3": _CONV1 + "l = LeakyRelu<alpha=0.3>(c0)\n y = Conv(l, w2, b2)",
    "prelu": _CONV1 + "h = PRelu(c0, sl)\n y = Conv(h, w2, b2)",
    "clip6": _CONV1 + "c = Clip(c0, lo, hi6)\n y = Conv(c, w2, b2)",
    "softmax": _CONV1 + "s = Softmax<axis=1>(c0)\n y = Conv(s, w2, b2)",
    "bn_after_conv": _CONV1
    + "b = BatchNormalization(c0, bns, bnb, bnm, bnv)\n y = Conv(b, w2, b2)",
    "bn_after_conv_relu": _CONV1
    + "b = BatchNormalization(c0, bns, bnb, bnm, bnv)\n r = Relu(b)\n y = Conv(r, w2, b2)",
    "bn_after_relu": _CONV1
    + "r = Relu(c0)\n b = BatchNormalization(r, bns, bnb, bnm, bnv)\n y = Conv(b, w5, b5)",
    "reduce_mean": _CONV1
    + "r = ReduceMean<axes=[2,3], keepdims=1>(c0)\n y = Conv(r, w2, b2)",
    "reduce_mean_keepdims0": _CONV1
    + "r = ReduceMean<axes=[2,3], keepdims=0>(c0)\n y = Gemm(r, wg, bg)",
    "reduce_mean_one_axis": _CONV1
    + "r = ReduceMean<axes=[3], keepdims=1>(c0)\n y = Conv(r, w2, b2)",
    "split_sizes": _CONV1
    + "s0, s1 = Split<axis=1>(c0, sp)\n y0 = Conv(s0, w8, b8)\n y1 = Conv(s1, w8, b8)\n y = Add(y0, y1)",
    "slice": _CONV1 + "s = Slice(c0, starts, ends, axes1)\n y = Conv(s, w8, b8)",
    "pad_into_conv": _CONV1 + "p = Pad(c0, pads)\n y = Conv(p, w2, b2)",
    "pad_into_avg_pool": _CONV1
    + "p = Pad(c0, pads)\n a = AveragePool<kernel_shape=[3,3]>(p)\n y = Conv(a, w2, b2)",
    "pad_into_max_pool": _CONV1
    + "p = Pad(c0, pads)\n a = MaxPool<kernel_shape=[3,3]>(p)\n y = Conv(a, w2, b2)",
    "pad_into_two_convs": _CONV1
    + "p = Pad(c0, pads)\n a = Conv(p, w5, b5)\n b = Conv(p, w6, b6)\n"
    " s = Add(a, b)\n y = Conv(s, w2, b2)",
    "pad_into_conv_and_add": _CONV1
    + "p = Pad(c0, pads)\n a = Conv(p, w5, b5)\n s = Add(a, p)\n y = Conv(s, w5, b5)",
    "convt_bn": "c0 = ConvTranspose(x, wt, bt)\n"
    " b = BatchNormalization(c0, bns, bnb, bnm, bnv)\n y = Conv(b, w5, b5)",
    "conv_transpose": "c0 = ConvTranspose(x, wt, bt)\n r = Relu(c0)\n y = Conv(r, w5, b5)",
    "identity": _CONV1 + "i = Identity(c0)\n y = Conv(i, w2, b2)",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_xint8_graph_matches_quark(name, tmp_path):
    model = _build(CASES[name])
    data = _data(SHAPE)
    q = _quark_preset(model, data, tmp_path)
    m = _mine_preset(model, data)
    _assert_same_graph(q, m, name)
    x = data[0]["x"]
    np.testing.assert_array_equal(_ort(m, x), _ort(q, x))


_MLP = {
    # a BatchNorm after a Gemm folds when transB = 1 ...
    "gemm_bn_folds": "g = Gemm<transB=1>(x, wg1, bg1)\n"
    " b = BatchNormalization(g, bns, bnb, bnm, bnv)\n y = Gemm(b, wg3, bg3)",
    # ... and otherwise stays, quantized like any other op
    "gemm_bn_stays": "g = Gemm(x, wg2, bg2)\n"
    " b = BatchNormalization(g, bns, bnb, bnm, bnv)\n y = Gemm(b, wg3, bg3)",
}


@pytest.mark.parametrize("name", sorted(_MLP))
@pytest.mark.parametrize("extra", [{}, {"ConvertBNToConv": False}])
def test_batch_norm_after_a_gemm_matches_quark(name, extra, tmp_path):
    model = _build(_MLP[name], shape=(3, 16))
    data = _data((3, 16))
    q = _quark_preset(model, data, tmp_path, extra=extra)
    m = _mine_preset(model, data, extra=extra)
    _assert_same_graph(q, m, f"{name} {extra}")
    x = data[0]["x"]
    np.testing.assert_array_equal(_ort(m, x), _ort(q, x))
    ops = [n.op_type for n in q.graph.node]
    assert ("BatchNormalization" in ops) == (name == "gemm_bn_stays")


def test_the_rewrites_are_visible_in_the_graphs(tmp_path):
    """The parity above is not vacuous: Quark's graph really has the rewritten
    nodes."""
    data = _data(SHAPE)
    ops = {
        k: [
            n.op_type
            for n in _quark_preset(_build(CASES[k]), data, tmp_path).graph.node
        ]
        for k in (
            "avg3",
            "sigmoid",
            "reduce_mean",
            "split_sizes",
            "bn_after_relu",
            "hard_swish",
        )
    }
    assert ops["avg3"].count("Mul") == 1
    assert "HardSigmoid" in ops["sigmoid"] and "Sigmoid" not in ops["sigmoid"]
    assert "GlobalAveragePool" in ops["reduce_mean"] and "Mul" in ops["reduce_mean"]
    assert "Slice" in ops["split_sizes"] and "Split" not in ops["split_sizes"]
    assert ops["bn_after_relu"].count("Conv") == 3
    assert "HardSwish" not in ops["hard_swish"]


def test_split_without_sizes_stays_and_shares_its_input_grid(tmp_path):
    """Quark cannot convert a Split without explicit sizes (opset 18's
    ``num_outputs``): it stays, its outputs on the input's quantization."""
    body = (
        _CONV1 + "s0, s1 = Split<axis=1, num_outputs=2>(c0)\n y0 = Conv(s0, w8, b8)\n"
        " y1 = Conv(s1, w8, b8)\n y = Add(y0, y1)"
    )
    model = _build(body, opset=18)
    data = _data(SHAPE)
    q = _quark_preset(model, data, tmp_path)
    m = _mine_preset(model, data)
    assert "Split" in [n.op_type for n in q.graph.node]
    _assert_same_graph(q, m, "split")


def test_a_large_global_pool_is_split_like_quarks(tmp_path):
    shape = (1, 3, 42, 42)
    model = _build(
        "c0 = Conv(x, w1, b1)\n p = GlobalAveragePool(c0)\n y = Conv(p, w2, b2)",
        shape=shape,
    )
    data = _data(shape)
    q = _quark_preset(model, data, tmp_path)
    m = _mine_preset(model, data)
    assert [n.op_type for n in q.graph.node].count("AveragePool") == 1
    _assert_same_graph(q, m, "large pool")


# -- the position adjustments ---------------------------------------------------------------
#
# On the graphs above the positions already meet the NPU's limits. Scaling the
# weights and biases by random powers of two pushes the shifts out of range, so
# the Align* / Adjust* passes have something to do.

SWEEP = {
    "chain": _CONV1 + "r = Relu(c0)\n c1 = Conv(r, w5, b5)\n y = Conv(c1, w2, b2)",
    "chain_no_relu": _CONV1 + "c1 = Conv(c0, w5, b5)\n y = Conv(c1, w2, b2)",
    "add": CASES["add"],
    "sub": _CONV1 + "c1 = Conv(x, w3, b3)\n a = Sub(c0, c1)\n y = Conv(a, w5, b5)",
    "add_relu": _CONV1
    + "c1 = Conv(x, w3, b3)\n a = Add(c0, c1)\n r = Relu(a)\n y = Conv(r, w5, b5)",
    "concat": CASES["concat"],
    "swish": CASES["swish"],
    "sigmoid": CASES["sigmoid"],
    "avg_pool": _CONV1
    + "r = Relu(c0)\n p = AveragePool<kernel_shape=[3,3]>(r)\n y = Conv(p, w2, b2)",
    "global_pool": _CONV1
    + "p = GlobalAveragePool(c0)\n c1 = Conv(p, w5, b5)\n y = Conv(c1, w2, b2)",
    "max_pool": _CONV1 + _POOL + "c1 = Conv(p, w5, b5)\n y = Conv(c1, w2, b2)",
    "mul": _CONV1 + "c1 = Conv(x, w3, b3)\n a = Mul(c0, c1)\n y = Conv(a, w5, b5)",
    "leaky": _CONV1
    + "l = LeakyRelu<alpha=0.1>(c0)\n c1 = Conv(l, w5, b5)\n y = Conv(c1, w2, b2)",
    "pool_concat": _CONV1
    + "p = MaxPool<kernel_shape=[1,1]>(c0)\n c1 = Conv(x, w3, b3)\n"
    " k = Concat<axis=1>(p, c1)\n y = Conv(k, w4, b4)",
}
_WEIGHTS = ("w1", "w2", "w3", "w4", "w5", "w8")
_BIASES = ("b1", "b2", "b3", "b4", "b5", "b8")


def _scaled_case(pattern, seed):
    """``SWEEP[pattern]`` with every weight (bias) multiplied by ``2**k``, ``k``
    random in [-14, 8] ([-16, 9])."""
    rng = np.random.default_rng(1000 + seed)
    base = _case_inits()
    over = {k: base[k] * np.float32(2.0 ** rng.integers(-14, 9)) for k in _WEIGHTS}
    over.update(
        {k: base[k] * np.float32(2.0 ** rng.integers(-16, 10)) for k in _BIASES}
    )
    return _build(SWEEP[pattern], over)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("pattern", sorted(SWEEP))
def test_scaled_xint8_graphs_match_quark(pattern, seed, tmp_path):
    model = _scaled_case(pattern, seed)
    data = _data(SHAPE)
    q = _quark_preset(model, data, tmp_path)
    m = _mine_preset(model, data)
    _assert_same_graph(q, m, f"{pattern} #{seed}")
    x = data[0]["x"]
    np.testing.assert_array_equal(_ort(m, x), _ort(q, x))


def _craft_cancelling_add():
    """Two convolutions that cancel: the Add's output is far finer than its
    inputs (AdjustShiftWrite)."""
    rng = np.random.default_rng(0)
    w1 = _w(rng, 8, 3, 3, 3)
    b1 = _w(rng, 8)
    w1n = (-w1 + _w(rng, 8, 3, 3, 3, scale=1e-3)).astype(np.float32)
    return _build(
        _CONV1 + "c1 = Conv(x, w1n, b1n)\n a = Add(c0, c1)\n y = Conv(a, w5, b5)",
        dict(w1=w1, b1=b1, w1n=w1n, b1n=-b1),
    )


def _craft_small_slice():
    """A Slice of the channels with small weights: its own range is finer than
    its input's (AlignSlice)."""
    rng = np.random.default_rng(0)
    w1 = _w(rng, 8, 3, 3, 3)
    w1[:4] *= 0.01
    return _build(CASES["slice"], dict(w1=w1))


def _craft_swish(k):
    rng = np.random.default_rng(0)
    return _build(
        CASES["swish"],
        dict(
            w1=_w(rng, 8, 3, 3, 3) * np.float32(2.0**k),
            b1=_w(rng, 8) * np.float32(2.0**k),
        ),
    )


# (option, models on which switching it off changes Quark's graph)
_PASSES = [
    ("AdjustShiftCut", lambda s: _scaled_case("chain_no_relu", s)),
    ("AdjustShiftBias", lambda s: _scaled_case("chain_no_relu", s)),
    ("AdjustShiftRead", lambda s: _scaled_case("add", s)),
    ("AdjustHardSigmoid", lambda s: _scaled_case("sigmoid", s)),
    ("AlignConcat", lambda s: _scaled_case("concat", s)),
    ("AlignPool", lambda s: _scaled_case("avg_pool", s)),
    ("AdjustShiftWrite", lambda s: _craft_cancelling_add()),
    ("AlignSlice", lambda s: _craft_small_slice()),
    ("AdjustShiftSwish", lambda s: _craft_swish(12 + 2 * (s % 2))),
]


@pytest.mark.parametrize("option, build", _PASSES, ids=[p[0] for p in _PASSES])
def test_each_adjustment_pass_matches_quark_on_and_off(option, build, tmp_path):
    """Per pass: find a graph where switching it off changes what Quark emits,
    then onnxsim must emit Quark's graph with the pass on *and* with it off."""
    data = _data(SHAPE)
    for seed in range(12):
        model = build(seed)
        on = _quark_preset(model, data, tmp_path)
        off = _quark_preset(model, data, tmp_path, extra={option: False})
        if _graph_diff(on, off):
            break
    else:
        pytest.fail(f"no graph found where {option} matters")
    _assert_same_graph(on, _mine_preset(model, data), f"{option} on")
    _assert_same_graph(
        off, _mine_preset(model, data, extra={option: False}), f"{option} off"
    )


# every option of the NPU stages, switched off or changed, on a graph it affects
_OPTION_CASES = [
    ({"NPULimitationCheck": False}, "chain_no_relu"),
    ({"MaxLoopNum": 1}, "add"),
    ({"MaxLoopNum": 1}, "concat"),
    ({"AdjustShiftRead": False, "AdjustShiftWrite": False}, "add"),
    ({"AdjustHardSigmoid": False, "AdjustShiftSwish": False}, "swish"),
    ({"SimulateDPU": False}, "avg_pool"),
    ({"SimulateDPU": False}, "swish"),
    ({"ConvertSigmoidToHardSigmoid": False}, "swish"),
    ({"ConvertSigmoidToHardSigmoid": False}, "sigmoid"),
    ({"ConvertHardSigmoidToDPUVersion": False}, "sigmoid"),
    ({"ConvertAvgPoolToDPUVersion": False}, "avg_pool"),
    ({"ConvertAvgPoolToDPUVersion": False}, "global_pool"),
    ({"ConvertLeakyReluToDPUVersion": False}, "leaky"),
    ({"ConvertReduceMeanToGlobalAvgPool": False}, "reduce_mean"),
    (
        {
            "ConvertReduceMeanToGlobalAvgPool": False,
            "ConvertReduceMeanToDPUVersion": False,
        },
        "reduce_mean",
    ),
    ({"ConvertReduceMeanToDPUVersion": False}, "reduce_mean"),
    ({"ConvertBNToConv": False}, "bn_after_relu"),
    ({"ConvertSplitToSlice": False}, "split_sizes"),
    ({"SplitLargeKernelPool": False}, "global_pool"),
    ({"ConvertClipToDPUVersion": True}, "clip6"),
    ({"ConvertClipToRelu": True}, "clip6"),
    ({"RemoveQDQConvRelu": False}, "chain"),
    ({"RemoveQDQConvClip": False}, "clip6"),
    ({"AlignPool": False}, "max_pool"),
]


def _option_model(name):
    if name in SWEEP:
        return _scaled_case(name, 3)
    return _build(CASES[name])


@pytest.mark.parametrize(
    "extra, name",
    _OPTION_CASES,
    ids=[f"{'+'.join(f'{k}={v}' for k, v in e.items())}-{n}" for e, n in _OPTION_CASES],
)
def test_xint8_options_match_quark(extra, name, tmp_path):
    model = _option_model(name)
    data = _data(SHAPE)
    q = _quark_preset(model, data, tmp_path, extra=extra)
    m = _mine_preset(model, data, extra=extra)
    _assert_same_graph(q, m, f"{extra} {name}")
    x = data[0]["x"]
    np.testing.assert_array_equal(_ort(m, x), _ort(q, x))


def test_the_options_change_the_graph():
    """... and are not vacuous: on onnxsim's side they change what is emitted."""
    data = _data(SHAPE)
    model = _build(CASES["avg3"])
    on = _mine_preset(model, data)
    assert _graph_diff(on, _mine_preset(model, data, extra={"SimulateDPU": False}))
    model = _scaled_case("concat", 1)
    on = _mine_preset(model, data)
    assert _graph_diff(on, _mine_preset(model, data, extra={"AlignConcat": False}))
    model = _build(CASES["clip6"])
    assert _graph_diff(
        _mine_preset(model, data),
        _mine_preset(model, data, extra={"ConvertClipToRelu": True}),
    )


# NPU CNN off / the float optimizers off: Quark takes these from its config
# object, not from extra_options, so they are compared on the new QConfig API.

_SPEC_OPTIONS = [
    {"EnableNPUCnn": False},
    {"OptimizeModel": False},
    {"SimplifyModel": False},
    {"OptimizeModel": False, "SimplifyModel": False},
]
_SPEC_CASES = [
    "avg3",
    "gap",
    "swish",
    "hard_swish",
    "bn_after_conv",
    "bn_after_relu",
    "pad_into_conv",
    "pad_into_avg_pool",
    "pad_into_max_pool",
    "pad_into_two_convs",
    "convt_bn",
    "identity",
    "split_sizes",
    "leaky",
    "concat",
    "prelu",
]


@pytest.mark.parametrize(
    "extra",
    _SPEC_OPTIONS,
    ids=["+".join(sorted(o)) + str(i) for i, o in enumerate(_SPEC_OPTIONS)],
)
@pytest.mark.parametrize("name", _SPEC_CASES)
def test_xint8_spec_options_match_quark(extra, name, tmp_path):
    model = _build(CASES[name])
    data = _data(SHAPE)
    q = _quark(model, data, tmp_path, extra=extra)
    m = _mine(model, data, extra=extra)
    _assert_same_graph(q, m, f"{extra} {name}")


# VINT8: the VAIML flavour -- no NPU CNN rewrites, but its own conversions.


@pytest.mark.parametrize(
    "name",
    [
        "clip6",
        "avg3",
        "sigmoid",
        "leaky",
        "bn_after_relu",
        "bn_after_conv",
        "concat",
        "hard_swish",
        "reduce_mean",
        "split_sizes",
        "add",
        "maxpool_then_relu",
        "pad_into_conv",
    ],
)
def test_vint8_graph_matches_quark(name, tmp_path):
    model = _build(CASES[name])
    data = _data(SHAPE)
    q = _quark_preset(model, data, tmp_path, preset="VINT8")
    m = _mine_preset(model, data, preset="VINT8")
    _assert_same_graph(q, m, name)


def test_vint8_turns_clip_into_relu_and_xint8_does_not(tmp_path):
    model = _build(CASES["clip6"])
    data = _data(SHAPE)
    v = _quark_preset(model, data, tmp_path, preset="VINT8")
    x = _quark_preset(model, data, tmp_path)
    assert "Relu" in [n.op_type for n in v.graph.node]
    assert "Clip" in [n.op_type for n in x.graph.node]


# == ReduceRange ===============================================================================
#
# Quark's legacy ``reduce_range`` (an attribute of its ``QuantizationConfig``, not an
# option): weights -- and constants quantized like weights -- keep to the reduced
# code range ([-64, 64] int8, [0, 127] uint8, [-16384, 16384] int16); activations are
# untouched and the biases follow the weights' scale. onnxsim reads it from
# ``extra_options["ReduceRange"]``. Refused with the NPU CNN scheme by both.
# (Non-power-of-two scales: compared to float32 rounding, like the other presets'
# tests in ``test_quark_parity.py``.)


def _close_summary(model):
    """Op types, the activation quantizers (scale, zero point, dtype) and the
    dequantized constants of a Q/DQ graph, independent of names and order."""
    vals = _values(model)
    ops, acts, consts = [], [], []
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in vals:
            zp = vals[n.input[2]]
            acts.append((str(zp.dtype), int(zp), float(vals[n.input[1]])))
        elif n.op_type == "DequantizeLinear" and n.input[0] in vals:
            q = vals[n.input[0]].astype(np.float64)
            z = vals[n.input[2]].astype(np.float64)
            s = vals[n.input[1]].astype(np.float64)
            if s.ndim:
                s = s.reshape([-1] + [1] * (q.ndim - 1)) if q.ndim else s
            consts.append(((q - z) * s, str(vals[n.input[0]].dtype)))
        elif n.op_type not in ("QuantizeLinear", "DequantizeLinear", "Constant"):
            ops.append(n.op_type)
    acts.sort(key=lambda a: (a[0], a[1], round(a[2], 3)))
    consts.sort(key=lambda c: (c[0].size, c[1], round(float(np.abs(c[0]).sum()), 1)))
    return sorted(ops), acts, consts


def _assert_close_graph(q, m, msg=""):
    oq, aq, cq = _close_summary(q)
    om, am, cm = _close_summary(m)
    assert om == oq, msg
    assert [a[:2] for a in am] == [a[:2] for a in aq], msg
    np.testing.assert_allclose(
        [a[2] for a in am], [a[2] for a in aq], rtol=1e-5, err_msg=msg
    )
    assert len(cm) == len(cq), msg
    for (xm, dm), (xq, dq) in zip(cm, cq):
        assert dm == dq, msg
        assert xm.shape == xq.shape, msg
        # int32 biases: a code or two of calibration noise
        atol = 1e-3 * max(float(np.abs(xq).max()), 1e-6) if dq == "int32" else 1e-5
        np.testing.assert_allclose(xm, xq, rtol=1e-5, atol=atol, err_msg=msg)


_RR_PRESETS = [
    "A8W8",
    "U8S8_AAWS",
    "S8S8_AAWS",
    "INT8_CNN_DEFAULT",
    "A16W8",
    "U16S8_AAWS",
    "U8U8_AAWA",
]
_RR_CASES = ["avg3", "add", "maxpool_then_relu", "leaky"]


@pytest.mark.parametrize("name", _RR_CASES)
@pytest.mark.parametrize("preset", _RR_PRESETS)
def test_reduce_range_matches_quark(preset, name, tmp_path):
    model = _build(CASES[name])
    data = _data(SHAPE)
    q = _quark_preset(model, data, tmp_path, preset=preset, reduce_range=True)
    full = _quark_preset(model, data, tmp_path, preset=preset)
    m = _mine_preset(model, data, preset=preset, extra={"ReduceRange": True})
    _assert_close_graph(q, m, f"{preset} {name}")

    # ... and it is not vacuous: Quark's weights really are on the reduced grid
    def weight_codes(model):
        vals = _values(model)
        return max(
            int(
                np.abs(
                    vals[n.input[0]].astype(np.int32) - int(vals[n.input[2]].ravel()[0])
                ).max()
            )
            for n in model.graph.node
            if n.op_type == "DequantizeLinear"
            and n.input[0] in vals
            and vals[n.input[0]].ndim == 4
        )

    assert weight_codes(q) <= (127 if preset == "U8U8_AAWA" else 64)
    assert weight_codes(full) > weight_codes(q)
    assert weight_codes(m) == weight_codes(q)


@pytest.mark.parametrize(
    "extra", [{"WeightSymmetric": False}, {"ActivationSymmetric": True}]
)
def test_reduce_range_with_other_quantizer_options(extra, tmp_path):
    model = _build(CASES["avg3"])
    data = _data(SHAPE)
    q = _quark_preset(
        model, data, tmp_path, preset="U8S8_AAWS", extra=extra, reduce_range=True
    )
    m = _mine_preset(
        model, data, preset="U8S8_AAWS", extra={**extra, "ReduceRange": True}
    )
    _assert_close_graph(q, m, str(extra))


def test_reduce_range_is_refused_with_the_npu_cnn_scheme(tmp_path):
    model = _build(CASES["avg3"])
    data = _data(SHAPE)
    with pytest.raises(Exception, match="reduce_range"):
        _quark_preset(model, data, tmp_path, reduce_range=True)
    with pytest.raises(ValueError, match="ReduceRange"):
        _mine_preset(model, data, extra={"ReduceRange": True})
