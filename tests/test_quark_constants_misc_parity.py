"""Quark's ``Constant`` nodes, the DPU simulation of the extended quantizer and the NPU
transformer scheme's conversion scope -- onnxsim against the real AMD Quark ONNX
package (the parity tests, skipped unless ``quark.onnx`` is importable) and the same
properties without it (always run).

- ``SimplifyModel`` and ``OptimizeModel`` both off (or ``SkipPreprocess``): Quark keeps
  the ``Constant`` nodes of the float graph and its quantizers treat their outputs as
  activations -- a Conv weight Constant gets a calibrated ``Q -> DQ`` pair (not an
  integer initializer), a Conv / Gemm / InstanceNormalization bias Constant stays raw,
  Add / Mul / Pad / Clip operands are quantized, Split / Slice / Reshape parameters
  (integers) are not. With either pass on, the Constants are initializers.
- the ``Constant`` nodes Quark's own conversions add (``Split`` -> ``Slice``) stay in its
  graph whatever the optimizers did before.
- the extended quantizer (``A8W8``, the 16-bit presets) runs the DPU simulation too, all
  of it opt-in (``ConvertInstanceNormToDPUVersion``, ...), without ``value_info``, so a
  ``GlobalAveragePool`` / ``ReduceMean`` window is never found; a ``Split`` that stays a
  ``Split`` shares its input's parameters there as everywhere.
- Quark's pre-optimizer converts only nodes whose op type the quantizer takes: the NPU
  transformer scheme (Gemm / MatMul) leaves a ``ReduceMean`` / ``Split`` / ``Clip`` alone,
  a plain quantizer (no ``ReduceMean`` in its registry) too.

Models are built with ``onnx.parser``; the weights, which the text format cannot spell
out, are attached as numpy arrays after parsing (some of them as ``Constant`` nodes).
"""

import contextlib
import copy
import io
import os
import warnings
import zlib
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
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

needs_quark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

import test_quark_xint8_parity as P  # noqa: E402  (its Quark import is guarded too)

from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim.full_qdq import quantize_full_qdq  # noqa: E402

INT_PRESETS = (
    "A8W8",
    "A16W8",
    "VINT8",
    "U8S8_AAWS",
    "S8S8_AAWS",
    "U16S8_AAWS",
    "S16S8_ASWS",
    "INT8_CNN_DEFAULT",
    "INT8_TRANSFORMER_DEFAULT",
    "XINT8",
)
#: the presets whose quantizer is Quark's extended one
EXTENDED = ("A8W8", "A16W8", "U16S8_AAWS", "S16S8_ASWS")
NO_FOLD = {"SimplifyModel": False}  # (+ ``optimize_model=False``: see ``_opts``)


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files into the current directory."""
    monkeypatch.chdir(tmp_path)


# -- model construction ----------------------------------------------------------------


def _array(name, shape, scale=0.5):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _model(body, shape, outputs="float y", consts=(), arrays=None, opset=17):
    """``body`` in the ONNX text format over one input ``x``. ``arrays`` (name -> numpy
    array) are attached as initializers after parsing; the names in ``consts`` become
    ``Constant`` nodes instead (the text format cannot spell out a weight tensor)."""
    dims = ",".join(map(str, shape))
    model = parser.parse_model(
        f'<ir_version: 8, opset_import: ["": {opset}]> '
        f"g (float[{dims}] x) => ({outputs}) {{\n{body}\n}}"
    )
    nodes = []
    for name, arr in (arrays or {}).items():
        t = numpy_helper.from_array(arr, name)
        if name in consts:
            nodes.append(onnx.helper.make_node("Constant", [], [name], value=t))
        else:
            model.graph.initializer.append(t)
    old = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes + old)
    return onnx.shape_inference.infer_shapes(model)


def _data(shape, n=4, seed=3):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


class _Reader:
    def __init__(self, data):
        self.it = iter(data)

    def get_next(self):
        return next(self.it, None)

    def reset_iter(self):
        pass


def _mine(model, data, preset, extra=None):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            copy.deepcopy(model), calibration_data_reader=_Reader(data)
        )


def _quark(model, data, tmp_path, preset, extra=None, optimize=None):
    """Quark's preset as ``QConfig.get_default_config`` hands it out (a private copy; CLE
    off). Its presets that are legacy ``QuantizationConfig`` objects ignore the
    ``OptimizeModel`` option and read the config's ``optimize_model`` field, hence the
    separate argument (onnxsim reads the option)."""
    from quark.onnx import ModelQuantizer, QConfig

    cfg = copy.deepcopy(QConfig.get_default_config(preset))
    cfg.global_quant_config.include_cle = False
    if optimize is not None:
        cfg.global_quant_config.optimize_model = optimize
    cfg.global_quant_config.extra_options.update(
        {k: v for k, v in (extra or {}).items() if not k.startswith("Onnxsim")}
    )
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ModelQuantizer(cfg).quantize_model(src, dst, P._reader(data))
    return onnx.load(dst)


def _both(model, data, tmp_path, preset, extra=None, optimize=None):
    q = _quark(model, data, tmp_path, preset, extra, optimize)
    mine_extra = dict(extra or {})
    if optimize is not None:
        mine_extra["OptimizeModel"] = optimize
    return q, _mine(model, data, preset, mine_extra)


def _no_fold(preset):
    """The options of a flow that keeps the ``Constant`` nodes (XINT8's DPU nodes sit at
    the end of Quark's node list; onnxsim sorts them in unless asked not to)."""
    extra = dict(NO_FOLD)
    if preset == "XINT8":
        extra["OnnxsimKeepQuarkNodeOrder"] = True
    return extra


# -- graph comparison ----------------------------------------------------------------------


def _digest(arr):
    return (str(arr.dtype), arr.shape, arr.tobytes())


def _wired(model):
    """Name-independent, ordered description of the graph: per node its op type,
    domain, attributes and inputs (the producing node's position, an initializer's /
    Constant's value, a graph input's name), then the graph outputs."""
    model = P._uniform_qdq(model)  # (an int32 bias's scale: one per element / one)
    inits = {t.name: _digest(numpy_helper.to_array(t)) for t in model.graph.initializer}
    produced = {}
    for i, n in enumerate(model.graph.node):
        for k, o in enumerate(n.output):
            produced[o] = (i, k)
    graph_inputs = {i.name for i in model.graph.input}

    def describe(t):
        if t == "":
            return ("empty",)
        if t in inits:
            return ("init", inits[t])
        if t in produced:
            i, k = produced[t]
            n = model.graph.node[i]
            if n.op_type == "Constant":
                return ("const", _digest(numpy_helper.to_array(n.attribute[0].t)))
            return ("node", i, k)
        assert t in graph_inputs, t
        return ("input", t)

    nodes = [
        (
            n.op_type,
            n.domain,
            repr(P._node_attrs(n)) if n.op_type != "Constant" else "",
            tuple(describe(t) for t in n.input),
        )
        for n in model.graph.node
    ]
    return nodes, [describe(o.name) for o in model.graph.output]


def _same_wiring(q, m, msg):
    wq, wm = _wired(q), _wired(m)
    ops_q = [(n[0], n[1]) for n in wq[0]]
    ops_m = [(n[0], n[1]) for n in wm[0]]
    assert ops_q == ops_m, (
        f"{msg}: node order differs\nQuark:   {ops_q}\nonnxsim: {ops_m}"
    )
    assert wq == wm, f"{msg}: wiring differs"


def _same_graph(q, m, msg):
    """Same graph up to node order (the value-based comparison of the Quark parity tests)
    and the same multiset of op types, ``Constant`` nodes included."""
    P._assert_same_graph(q, m, msg)
    assert Counter(n.op_type for n in q.graph.node) == Counter(
        n.op_type for n in m.graph.node
    ), msg


def _constants(model):
    return sorted(
        _digest(numpy_helper.to_array(n.attribute[0].t))
        for n in model.graph.node
        if n.op_type == "Constant"
    )


def _run(model, data):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    lib = os.environ.get("QUARK_ONNX_OPS_LIB")
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, data)


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _compute_ops(model):
    return [n.op_type for n in model.graph.node if "Linear" not in n.op_type]


# -- the models ---------------------------------------------------------------------------------

SHAPE4 = (1, 3, 8, 8)


def _conv_gemm():
    body = """
    c = Conv<pads=[1,1,1,1]>(x, cw, cb)
    r = Relu(c)
    p = GlobalAveragePool(r)
    f = Flatten(p)
    y = Gemm(f, gw, gb)
    """
    arrays = dict(
        cw=_array("cw", (4, 3, 3, 3)),
        cb=_array("cb", (4,)),
        gw=_array("gw", (4, 5)),
        gb=_array("gb", (5,)),
    )
    return _model(body, SHAPE4, "float[1,5] y", tuple(arrays), arrays), SHAPE4


def _gemm_transb():
    arrays = dict(gw=_array("gw", (5, 6)), gb=_array("gb", (5,)))
    body = "y = Gemm<transB=1>(x, gw, gb)"
    return _model(body, (2, 6), "float[2,5] y", tuple(arrays), arrays), (2, 6)


def _matmul_add_mul():
    arrays = dict(mw=_array("mw", (16, 8)), ca=_array("ca", (8,)))
    body = """
    ck = Constant<value = float {1.5}>()
    m = MatMul(x, mw)
    a = Add(m, ca)
    y = Mul(a, ck)
    """
    return _model(body, (4, 16), "float[4,8] y", tuple(arrays), arrays), (4, 16)


def _matmul_const_a():
    arrays = dict(ma=_array("ma", (4, 16)))
    body = "y = MatMul(ma, x)"
    return _model(body, (16, 8), "float[4,8] y", tuple(arrays), arrays), (16, 8)


def _elementwise():
    arrays = dict(a1=_array("a1", (1, 4, 1, 1)), a2=_array("a2", (1, 4, 1, 1)))
    body = """
    a3 = Constant<value = float {2.0}>()
    t1 = Add(x, a1)
    t2 = Mul(t1, a2)
    t3 = Sub(t2, a3)
    y = Div(t3, a3)
    """
    shape = (1, 4, 6, 6)
    return _model(body, shape, "float[1,4,6,6] y", tuple(arrays), arrays), shape


def _split_slice_pad():
    body = """
    sp = Constant<value = int64[2] {4, 4}>()
    st = Constant<value = int64[1] {0}>()
    en = Constant<value = int64[1] {2}>()
    ax = Constant<value = int64[1] {1}>()
    sts = Constant<value = int64[1] {1}>()
    pd = Constant<value = int64[8] {0, 0, 0, 0, 0, 0, 0, 0}>()
    pv = Constant<value = float {0.0}>()
    a, b = Split<axis=1>(x, sp)
    s = Slice(a, st, en, ax, sts)
    pp = Pad(b, pd, pv)
    y = Concat<axis=1>(s, pp)
    """
    shape = (1, 8, 8, 8)
    return _model(body, shape, "float[1,6,8,8] y"), shape


def _clip_reshape():
    body = """
    mn = Constant<value = float {0.0}>()
    mx = Constant<value = float {6.0}>()
    sh = Constant<value = int64[2] {1, 144}>()
    c = Clip(x, mn, mx)
    y = Reshape(c, sh)
    """
    shape = (1, 4, 6, 6)
    return _model(body, shape, "float[1,144] y"), shape


def _conv_clip_relu():
    arrays = dict(cw=_array("cw", (4, 3, 3, 3)), cb=_array("cb", (4,)))
    body = """
    mn = Constant<value = float {0.0}>()
    mx = Constant<value = float {6.0}>()
    c = Conv<pads=[1,1,1,1]>(x, cw, cb)
    k = Clip(c, mn, mx)
    y = Relu(k)
    """
    m = _model(body, SHAPE4, "float[1,4,8,8] y", tuple(arrays), arrays)
    return m, SHAPE4


def _convt_prelu():
    arrays = dict(
        cw=_array("cw", (3, 4, 3, 3)), cb=_array("cb", (4,)), sl=_array("sl", (4, 1, 1))
    )
    body = """
    t = ConvTranspose<pads=[1,1,1,1]>(x, cw, cb)
    y = PRelu(t, sl)
    """
    m = _model(body, SHAPE4, "float[1,4,8,8] y", tuple(arrays), arrays)
    return m, SHAPE4


def _layer_norm():
    arrays = dict(sc=_array("sc", (16,), 1.0), bi=_array("bi", (16,)))
    body = "y = LayerNormalization<axis=-1>(x, sc, bi)"
    shape = (2, 8, 16)
    return _model(body, shape, "float[2,8,16] y", tuple(arrays), arrays), shape


def _instance_norm():
    arrays = dict(sc=_array("sc", (4,), 1.0), bi=_array("bi", (4,)))
    body = "y = InstanceNormalization(x, sc, bi)"
    shape = (1, 4, 6, 6)
    return _model(body, shape, "float[1,4,6,6] y", tuple(arrays), arrays), shape


def _bn_after_conv():
    arrays = dict(
        cw=_array("cw", (4, 3, 3, 3)),
        cb=_array("cb", (4,)),
        bs=np.array([1.0, 1.1, 0.9, 1.2], np.float32),
        bb=np.array([0.1, 0.2, 0.3, 0.4], np.float32),
        bm=np.array([0.0, 0.1, 0.2, 0.3], np.float32),
        bv=np.array([1.0, 1.5, 0.5, 2.0], np.float32),
    )
    body = """
    c = Conv<pads=[1,1,1,1]>(x, cw, cb)
    y = BatchNormalization(c, bs, bb, bm, bv)
    """
    m = _model(body, SHAPE4, "float[1,4,8,8] y", tuple(arrays), arrays)
    return m, SHAPE4


def _shared_and_output():
    arrays = dict(a1=_array("a1", (1, 4, 1, 1)))
    body = """
    t1 = Add(x, a1)
    y = Mul(t1, a1)
    z = Identity(a1)
    """
    shape = (1, 4, 6, 6)
    m = _model(body, shape, "float[1,4,6,6] y, float[1,4,1,1] z", tuple(arrays), arrays)
    return m, shape


MODELS = {
    "conv_gemm": _conv_gemm,
    "gemm_transb": _gemm_transb,
    "matmul_add_mul": _matmul_add_mul,
    "matmul_const_a": _matmul_const_a,
    "elementwise": _elementwise,
    "split_slice_pad": _split_slice_pad,
    "clip_reshape": _clip_reshape,
    "conv_clip_relu": _conv_clip_relu,
    "convt_prelu": _convt_prelu,
    "layer_norm": _layer_norm,
    "instance_norm": _instance_norm,
    "bn_after_conv": _bn_after_conv,
    "shared_and_output": _shared_and_output,
}


# -- Constant nodes: parity with Quark --------------------------------------------------------


@needs_quark
@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("preset", INT_PRESETS)
def test_constants_stay_in_the_graph_like_quark(name, preset, tmp_path):
    """Both optimizers off: the graph, node for node and in Quark's order, with the
    same Constant nodes, Q/DQ pairs, scales and zero points; and the same results."""
    model, shape = MODELS[name]()
    data = _data(shape)
    q, m = _both(model, data, tmp_path, preset, _no_fold(preset), optimize=False)
    _same_wiring(q, m, f"{name} {preset}")
    assert _constants(q) == _constants(m)
    for got, want in zip(_run(m, data[0]), _run(q, data[0])):
        np.testing.assert_array_equal(got, want)


@needs_quark
@pytest.mark.parametrize("preset", ["A8W8", "VINT8", "U8S8_AAWS", "XINT8"])
def test_skip_preprocess_keeps_the_constants_too(preset, tmp_path):
    model, shape = _conv_gemm()
    data = _data(shape)
    extra = {"SkipPreprocess": True}
    if preset == "XINT8":
        extra["OnnxsimKeepQuarkNodeOrder"] = True
    q, m = _both(model, data, tmp_path, preset, extra)
    assert "Constant" in _ops(q)
    _same_wiring(q, m, f"SkipPreprocess {preset}")


@needs_quark
@pytest.mark.parametrize("simplify", [True, False])
@pytest.mark.parametrize("optimize", [True, False])
def test_either_optimizer_folds_the_constants_like_quark(simplify, optimize, tmp_path):
    model, shape = _conv_gemm()
    data = _data(shape)
    q, m = _both(
        model, data, tmp_path, "A8W8", {"SimplifyModel": simplify}, optimize=optimize
    )
    assert ("Constant" in _ops(q)) == (not simplify and not optimize)
    assert ("Constant" in _ops(m)) == ("Constant" in _ops(q))
    _same_graph(q, m, f"simplify={simplify} optimize={optimize}")


@needs_quark
@pytest.mark.parametrize("preset", ["A8W8", "A16W8", "U8S8_AAWS", "XINT8"])
def test_constants_of_a_converted_split_stay_in_the_graph(preset, tmp_path):
    """The Constant nodes Quark's ``Split`` -> ``Slice`` conversion adds are in its
    graph even though ONNX Runtime and onnxslim ran before (they fold the model's
    own)."""
    arrays = dict(w=_array("w", (4, 4, 3, 3)), b=_array("b", (4,)))
    body = """
    sp = Constant<value = int64[2] {2, 2}>()
    r = Conv<pads=[1,1,1,1]>(x, w, b)
    y, z = Split<axis=1>(r, sp)
    """
    shape = (1, 4, 8, 8)
    model = _model(body, shape, "float[1,2,8,8] y, float[1,2,8,8] z", ("sp",), arrays)
    data = _data(shape)
    extra = {"ConvertSplitToSlice": True}
    q, m = _both(model, data, tmp_path, preset, extra)
    assert _ops(q).count("Constant") == 8 and "Split" not in _ops(q)
    _same_graph(q, m, f"{preset} split")


# -- Constant nodes: without Quark ----------------------------------------------------------


def _conv_mul_model():
    arrays = dict(w=_array("w", (4, 3, 3, 3)), b=_array("b", (4,)))
    body = """
    k = Constant<value = float {2.0}>()
    c = Conv<pads=[1,1,1,1]>(x, w, b)
    y = Mul(c, k)
    """
    return _model(body, SHAPE4, "float[1,4,8,8] y", ("w", "b"), arrays)


def _by_output(model):
    return {o: n for n in model.graph.node for o in n.output}


@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "INT8_CNN_DEFAULT"])
def test_constant_weights_are_activations_when_nothing_folds(preset):
    model = _conv_mul_model()
    q = _mine(model, _data(SHAPE4), preset, {"OptimizeModel": False, **NO_FOLD})
    by_out = _by_output(q)
    ops = _ops(q)
    assert ops.count("Constant") == 3
    conv = next(n for n in q.graph.node if n.op_type == "Conv")
    # the weight: a Constant -> Q -> DQ chain with calibrated float scale / zero point
    dq = by_out[conv.input[1]]
    assert dq.op_type == "DequantizeLinear"
    quant = by_out[dq.input[0]]
    assert quant.op_type == "QuantizeLinear"
    assert by_out[quant.input[0]].op_type == "Constant"
    # the bias: the raw Constant, unquantized (Quark's ``quantize_bias_tensor`` only
    # takes an initializer)
    assert by_out[conv.input[2]].op_type == "Constant"
    mul = next(n for n in q.graph.node if n.op_type == "Mul")
    assert by_out[mul.input[1]].op_type == "DequantizeLinear"
    assert all(t.data_type != onnx.TensorProto.INT32 for t in q.graph.initializer)


@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "VINT8"])
def test_the_optimizers_fold_the_constants(preset):
    model = _conv_mul_model()
    for extra in (
        {"OptimizeModel": True, "SimplifyModel": False},
        {"OptimizeModel": False, "SimplifyModel": True},
        {},
    ):
        q = _mine(model, _data(SHAPE4), preset, extra)
        assert "Constant" not in _ops(q), (preset, extra)


def test_skip_preprocess_keeps_the_constants():
    q = _mine(_conv_mul_model(), _data(SHAPE4), "A8W8", {"SkipPreprocess": True})
    assert _ops(q).count("Constant") == 3


def test_constants_lead_the_node_list_like_quarks():
    q = _mine(
        _conv_mul_model(), _data(SHAPE4), "A8W8", {"OptimizeModel": False, **NO_FOLD}
    )
    ops = _ops(q)
    assert ops[:3] == ["Constant"] * 3
    # a valid topological order
    seen = {i.name for i in q.graph.input} | {t.name for t in q.graph.initializer}
    for n in q.graph.node:
        assert all(x in seen or not x for x in n.input)
        seen.update(n.output)


def test_vint8_quantizes_every_float_constant_as_an_op():
    """``QuantizeAllOpTypes`` lists ``Constant`` among the op types: the bias Constant
    gets a Q/DQ pair too (the plain presets leave it raw)."""
    model = _conv_mul_model()
    q = _mine(model, _data(SHAPE4), "VINT8", NO_FOLD)
    by_out = _by_output(q)
    conv = next(n for n in q.graph.node if n.op_type == "Conv")
    assert by_out[conv.input[2]].op_type == "DequantizeLinear"


def test_an_integer_constant_parameter_is_never_quantized():
    body = """
    sh = Constant<value = int64[2] {1, 144}>()
    mn = Constant<value = float {0.0}>()
    mx = Constant<value = float {6.0}>()
    c = Clip(x, mn, mx)
    y = Reshape(c, sh)
    """
    model = _model(body, (1, 4, 6, 6), "float[1,144] y")
    q = _mine(model, _data((1, 4, 6, 6)), "A8W8", {"OptimizeModel": False, **NO_FOLD})
    by_out = _by_output(q)
    reshape = next(n for n in q.graph.node if n.op_type == "Reshape")
    assert by_out[reshape.input[1]].op_type == "Constant"
    clip = next(n for n in q.graph.node if n.op_type == "Clip")
    assert by_out[clip.input[1]].op_type == "Constant"  # (the bounds stay raw)


def test_a_transformer_matmul_with_a_constant_b_is_not_a_weight_matmul():
    """Quark's ``MatMulConstBOnly`` looks for an *initializer*: with the Constants kept
    the MatMul is left alone, with them folded it is quantized."""
    arrays = dict(mw=_array("mw", (16, 8)))
    body = "y = MatMul(x, mw)"
    model = _model(body, (4, 16), "float[4,8] y", ("mw",), arrays)
    data = _data((4, 16))
    kept = _mine(
        model, data, "INT8_TRANSFORMER_DEFAULT", {**NO_FOLD, "OptimizeModel": False}
    )
    assert _ops(kept) == ["Constant", "MatMul"]
    folded = _mine(model, data, "INT8_TRANSFORMER_DEFAULT")
    assert "QuantizeLinear" in _ops(folded) and "Constant" not in _ops(folded)


def test_instance_norm_bias_constant_stays_raw():
    arrays = dict(sc=_array("sc", (4,), 1.0), bi=_array("bi", (4,)))
    body = "y = InstanceNormalization(x, sc, bi)"
    model = _model(body, (1, 4, 6, 6), "float[1,4,6,6] y", ("sc", "bi"), arrays)
    q = _mine(model, _data((1, 4, 6, 6)), "A8W8", {"OptimizeModel": False, **NO_FOLD})
    by_out = _by_output(q)
    norm = next(n for n in q.graph.node if n.op_type == "InstanceNormalization")
    assert by_out[norm.input[1]].op_type == "DequantizeLinear"
    assert by_out[norm.input[2]].op_type == "Constant"


def test_quantize_full_qdq_keeps_named_constants():
    model = _conv_mul_model()
    data = _data(SHAPE4)
    q = quantize_full_qdq(
        model, data, activation_dtype="int8", keep_constants={"k"}, op_types=None
    )
    consts = [
        n.output[0].split("/")[0] for n in q.graph.node if n.op_type == "Constant"
    ]
    assert consts == ["k"]
    q = quantize_full_qdq(model, data, activation_dtype="int8", keep_constants=True)
    assert len([n for n in q.graph.node if n.op_type == "Constant"]) == 3
    q = quantize_full_qdq(model, data, activation_dtype="int8")
    assert "Constant" not in _ops(q)


def test_the_conversion_constants_stay_when_the_optimizers_ran():
    arrays = dict(w=_array("w", (4, 4, 3, 3)), b=_array("b", (4,)))
    body = """
    sp = Constant<value = int64[2] {2, 2}>()
    r = Conv<pads=[1,1,1,1]>(x, w, b)
    y, z = Split<axis=1>(r, sp)
    """
    shape = (1, 4, 8, 8)
    model = _model(body, shape, "float[1,2,8,8] y, float[1,2,8,8] z", ("sp",), arrays)
    q = _mine(model, _data(shape), "A8W8")
    ops = _ops(q)
    assert "Split" not in ops and ops.count("Slice") == 2
    assert ops.count("Constant") == 8  # starts / ends / axes / steps of each Slice


# -- ExtendedInstanceNormalization and the other DPU conversions of the extended quantizer ---


def _in_model():
    arrays = dict(sc=_array("sc", (4,), 1.0), bi=_array("bi", (4,)))
    body = "y = InstanceNormalization(x, sc, bi)"
    return _model(body, (1, 4, 6, 6), "float[1,4,6,6] y", (), arrays), (1, 4, 6, 6)


@needs_quark
@pytest.mark.parametrize("preset", INT_PRESETS)
@pytest.mark.parametrize("flag", [None, True, False])
def test_instance_norm_dpu_version_follows_quarks_quantizer(preset, flag, tmp_path):
    model, shape = _in_model()
    data = _data(shape)
    extra = {} if flag is None else {"ConvertInstanceNormToDPUVersion": flag}
    q, m = _both(model, data, tmp_path, preset, extra)
    kinds = [(n.op_type, n.domain) for n in q.graph.node if "Linear" not in n.op_type]
    converted = ("ExtendedInstanceNormalization", "com.amd.quark") in kinds
    assert converted == (bool(flag) and preset in (*EXTENDED, "XINT8"))
    assert [
        (n.op_type, n.domain) for n in m.graph.node if "Linear" not in n.op_type
    ] == kinds
    P._assert_same_graph(q, m, f"{preset} {flag}")


@pytest.mark.parametrize("preset", [*EXTENDED, "XINT8"])
def test_the_extended_quantizer_converts_instance_norm_on_request(preset):
    model, shape = _in_model()
    data = _data(shape)
    plain = _mine(model, data, preset)
    assert "InstanceNormalization" in _ops(plain)
    q = _mine(model, data, preset, {"ConvertInstanceNormToDPUVersion": True})
    node = next(n for n in q.graph.node if n.op_type == "ExtendedInstanceNormalization")
    assert node.domain == "com.amd.quark"
    assert any(o.domain == "com.amd.quark" for o in q.opset_import)
    assert [n.op_type for n in q.graph.node].count("InstanceNormalization") == 0


@pytest.mark.parametrize(
    "preset", ["U8S8_AAWS", "S8S8_AAWS", "INT8_CNN_DEFAULT", "VINT8"]
)
def test_the_plain_quantizers_ignore_the_dpu_conversions(preset):
    model, shape = _in_model()
    q = _mine(model, _data(shape), preset, {"ConvertInstanceNormToDPUVersion": True})
    assert "ExtendedInstanceNormalization" not in _ops(q)


def _dpu_model():
    arrays = dict(
        w=_array("w", (4, 4, 3, 3)),
        b=_array("b", (4,)),
        sc=_array("sc", (4,), 1.0),
        bi=_array("bi", (4,)),
        mn=np.array(-3.3, np.float32),
        mx=np.array(5.7, np.float32),
    )
    body = """
    c = Conv<pads=[1,1,1,1]>(x, w, b)
    l = LeakyRelu<alpha=0.1>(c)
    s = Sigmoid(l)
    a = AveragePool<kernel_shape=[3,3], pads=[1,1,1,1]>(s)
    n = InstanceNormalization(a, sc, bi)
    k = Clip(n, mn, mx)
    sm = Softmax<axis=1>(k)
    r = ReduceMean<axes=[2,3], keepdims=1>(sm)
    y = GlobalAveragePool(sm)
    y2 = Identity(r)
    """
    shape = (1, 4, 8, 8)
    out = "float[1,4,1,1] y, float[1,4,1,1] y2"
    return _model(body, shape, out, (), arrays), shape


DPU_OPTIONS = [
    "ConvertLeakyReluToDPUVersion",
    "ConvertSigmoidToHardSigmoid",
    "ConvertHardSigmoidToDPUVersion",
    "ConvertAvgPoolToDPUVersion",
    "ConvertReduceMeanToDPUVersion",
    "ConvertSoftmaxToDPUVersion",
    "ConvertInstanceNormToDPUVersion",
    "ConvertClipToDPUVersion",
]


@needs_quark
@pytest.mark.parametrize("preset", EXTENDED)
@pytest.mark.parametrize("option", [None, *DPU_OPTIONS, "ALL"])
def test_the_extended_quantizers_dpu_simulation_matches_quark(preset, option, tmp_path):
    model, shape = _dpu_model()
    data = _data(shape)
    extra = {}
    if option == "ALL":
        extra = {o: True for o in DPU_OPTIONS}
    elif option:
        extra = {option: True}
    q, m = _both(model, data, tmp_path, preset, extra)
    P._assert_same_graph(q, m, f"{preset} {option}")
    assert Counter(_ops(q)) == Counter(_ops(m)), (preset, option)


_CONV = "c = Conv<pads=[1,1,1,1]>(x, w, b)\n"
POOL_CASES = {
    "gap": (_CONV + "p = GlobalAveragePool(c)\n y = Flatten(p)", "float[1,4] y"),
    "gap_relu": (
        _CONV + "r = Relu(c)\n p = GlobalAveragePool(r)\n y = Flatten(p)",
        "float[1,4] y",
    ),
    "gap_input": ("p = GlobalAveragePool(x)\n y = Flatten(p)", "float[1,4] y"),
    "gap_output": (_CONV + "y = GlobalAveragePool(c)", "float[1,4,1,1] y"),
    "mean": (_CONV + "y = ReduceMean<axes=[2,3], keepdims=0>(c)", "float[1,4] y"),
}


@needs_quark
@pytest.mark.parametrize("preset", ["A8W8", "XINT8"])
@pytest.mark.parametrize("case", sorted(POOL_CASES))
def test_the_extended_quantizer_never_finds_a_pool_window(preset, case, tmp_path):
    """The extended quantizer hands ``simulate_transforms`` a graph without
    ``value_info``: ``GlobalAveragePool`` / ``ReduceMean`` keep their output (the NPU CNN
    quantizer, which has it, rescales them)."""
    arrays = dict(w=_array("w", (4, 4, 3, 3)), b=_array("b", (4,)))
    body, outputs = POOL_CASES[case]
    model = _model(body, (1, 4, 8, 8), outputs, (), arrays)
    data = _data((1, 4, 8, 8))
    extra = {"ConvertAvgPoolToDPUVersion": True, "ConvertReduceMeanToDPUVersion": True}
    q, m = _both(model, data, tmp_path, preset, extra)
    assert Counter(_compute_ops(q)) == Counter(_compute_ops(m)), (preset, case)
    P._assert_same_graph(q, m, f"{preset} {case}")
    # (a graph input has no value_info for the NPU CNN quantizer either)
    assert ("Mul" in _ops(q)) == (preset == "XINT8" and case != "gap_input")


def test_the_extended_dpu_simulation_converts_an_average_pool_only():
    model, shape = _dpu_model()
    q = _mine(
        model,
        _data(shape),
        "A8W8",
        {"ConvertAvgPoolToDPUVersion": True, "ConvertReduceMeanToDPUVersion": True},
    )
    ops = _ops(q)
    assert ops.count("Mul") == 1  # the 3x3 AveragePool; no value_info for the rest
    # (the ReduceMean is a GlobalAveragePool by then, the pre-optimizer's conversion:
    # neither of the two is rescaled)
    assert ops.count("GlobalAveragePool") == 2 and "ReduceMean" not in ops
    xint = _mine(model, _data(shape), "XINT8")
    assert _ops(xint).count("Mul") >= 2  # the NPU CNN quantizer has the shapes


def test_the_dpu_simulation_is_opt_in_for_the_extended_quantizer():
    model, shape = _dpu_model()
    q = _mine(model, _data(shape), "A8W8")
    ops = _ops(q)
    assert "Mul" not in ops and "HardSigmoid" not in ops and "Sigmoid" in ops
    off = _mine(
        model,
        _data(shape),
        "A8W8",
        {"SimulateDPU": False, "ConvertSigmoidToHardSigmoid": True},
    )
    assert "HardSigmoid" not in _ops(off)
    on = _mine(model, _data(shape), "A8W8", {"ConvertSigmoidToHardSigmoid": True})
    assert "HardSigmoid" in _ops(on) and "Sigmoid" not in _ops(on)


# -- a Split that stays a Split shares its input's parameters (the extended quantizer too) --


def _split_model():
    body = """
    sp = Constant<value = int64[2] {3, 5}>()
    a, b = Split<axis=1>(x, sp)
    y = Sigmoid(a)
    z = Sigmoid(b)
    """
    shape = (1, 8, 8, 8)
    return _model(body, shape, "float[1,3,8,8] y, float[1,5,8,8] z"), shape


def _scale_of(model, tensor):
    by_out = _by_output(model)
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] in (tensor, tensor + "/f"):
            return float(inits[n.input[1]])
    raise AssertionError(tensor, by_out.keys())


@needs_quark
@pytest.mark.parametrize("preset", [*EXTENDED, "XINT8", "U8S8_AAWS"])
def test_an_unconverted_split_shares_its_inputs_parameters(preset, tmp_path):
    model, shape = _split_model()
    data = _data(shape)
    q, m = _both(model, data, tmp_path, preset, {"ConvertSplitToSlice": False})
    P._assert_same_graph(q, m, f"{preset} split")
    assert "Split" in _ops(q) and "Split" in _ops(m)
    for graph in (q, m):
        assert _scale_of(graph, "a") == _scale_of(graph, "x")
        assert _scale_of(graph, "b") == _scale_of(graph, "x")


@pytest.mark.parametrize("preset", EXTENDED)
def test_the_extended_quantizer_shares_a_split_that_is_not_converted(preset):
    model, shape = _split_model()
    data = _data(shape)
    kept = _mine(model, data, preset, {"ConvertSplitToSlice": False})
    assert "Split" in _ops(kept)
    assert _scale_of(kept, "a") == _scale_of(kept, "x")
    converted = _mine(model, data, preset)
    assert "Split" not in _ops(converted) and _ops(converted).count("Slice") == 2


# -- Quark's pre-optimizer converts only the op types the quantizer takes ---------------------


def _transformer_tail(tail, outputs, extra_arrays=None):
    arrays = dict(w=_array("w", (16, 16)), b=_array("b", (16,)))
    arrays.update(extra_arrays or {})
    body = "g = Gemm(x, w, b)\n r = Reshape(g, shp)\n" + tail
    model = _model(body, (4, 16), outputs, (), arrays)
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1, 4, 4, 4], np.int64), "shp")
    )
    return onnx.shape_inference.infer_shapes(model)


TRANSFORMER_TAILS = {
    "reduce_mean": ("y = ReduceMean<axes=[2,3], keepdims=0>(r)", "float[1,4] y", {}),
    "global_pool": ("y = GlobalAveragePool(r)", "float[1,4,1,1] y", {}),
    "average_pool": ("y = AveragePool<kernel_shape=[2,2]>(r)", "float[1,4,2,2] y", {}),
    "clip": (
        "y = Clip(r, mn, mx)",
        "float[1,4,4,4] y",
        dict(mn=np.array(0.0, np.float32), mx=np.array(6.0, np.float32)),
    ),
    "split": (
        "y, z = Split<axis=1>(r, sp)",
        "float[1,2,4,4] y, float[1,2,4,4] z",
        dict(sp=np.array([2, 2], np.int64)),
    ),
    "batch_norm": (
        "y = BatchNormalization(r, bs, bb, bm, bv)",
        "float[1,4,4,4] y",
        dict(
            bs=np.array([1.0, 1.1, 0.9, 1.2], np.float32),
            bb=np.array([0.1, 0.2, 0.3, 0.4], np.float32),
            bm=np.array([0.0, 0.1, 0.2, 0.3], np.float32),
            bv=np.array([1.0, 1.5, 0.5, 2.0], np.float32),
        ),
    ),
}


@needs_quark
@pytest.mark.parametrize(
    "preset", ["INT8_TRANSFORMER_DEFAULT", "INT16_TRANSFORMER_DEFAULT"]
)
@pytest.mark.parametrize("tail", sorted(TRANSFORMER_TAILS))
def test_the_transformer_scheme_converts_only_what_it_quantizes(preset, tail, tmp_path):
    body, outputs, arrays = TRANSFORMER_TAILS[tail]
    model = _transformer_tail(body, outputs, arrays)
    data = _data((4, 16), n=4, seed=5)
    q, m = _both(model, data, tmp_path, preset)
    assert Counter(_compute_ops(q)) == Counter(_compute_ops(m)), tail
    P._assert_same_graph(q, m, f"{preset} {tail}")
    ops = _ops(q)
    # (only the BatchNorm -> Conv conversion adds its op type to the list first)
    assert ("Conv" in ops) == (tail == "batch_norm")
    assert "ReduceMean" in ops or tail != "reduce_mean"
    assert "Split" in ops or tail != "split"


@pytest.mark.parametrize("tail", ["reduce_mean", "split", "clip", "batch_norm"])
def test_the_transformer_scheme_leaves_unquantized_ops_unconverted(tail):
    body, outputs, arrays = TRANSFORMER_TAILS[tail]
    model = _transformer_tail(body, outputs, arrays)
    ops = _ops(_mine(model, _data((4, 16)), "INT8_TRANSFORMER_DEFAULT"))
    assert "GlobalAveragePool" not in ops and "Slice" not in ops
    assert ("Conv" in ops) == (tail == "batch_norm")
    assert "BatchNormalization" not in ops


PLAIN_CASES = {
    "reduce_mean": ("y = ReduceMean<axes=[2,3], keepdims=0>(r)", "float[1,4] y"),
    "split": ("y, z = Split<axis=1>(r, sp)", "float[1,2,8,8] y, float[1,2,8,8] z"),
    "clip": ("y = Clip(r, mn, mx)", "float[1,4,8,8] y"),
}


def _plain_model(case):
    body, outputs = PLAIN_CASES[case]
    arrays = dict(
        w=_array("w", (4, 4, 3, 3)),
        b=_array("b", (4,)),
        sp=np.array([2, 2], np.int64),
        mn=np.array(0.0, np.float32),
        mx=np.array(6.0, np.float32),
    )
    body = "r = Conv<pads=[1,1,1,1]>(x, w, b)\n" + body
    return _model(body, (1, 4, 8, 8), outputs, (), arrays), (1, 4, 8, 8)


@needs_quark
@pytest.mark.parametrize("preset", ["U8S8_AAWS", "INT8_CNN_DEFAULT", "A8W8"])
@pytest.mark.parametrize("case", sorted(PLAIN_CASES))
def test_a_conversion_asked_for_covers_the_quantizers_op_types(preset, case, tmp_path):
    """``ConvertReduceMeanToGlobalAvgPool`` / ``ConvertSplitToSlice`` /
    ``ConvertClipToRelu`` switched on in a plain quantizer: ``ReduceMean`` is not in
    its registry, ``Split`` and ``Clip`` are."""
    model, shape = _plain_model(case)
    data = _data(shape, seed=5)
    extra = {
        "ConvertReduceMeanToGlobalAvgPool": True,
        "ConvertSplitToSlice": True,
        "ConvertClipToRelu": True,
    }
    q, m = _both(model, data, tmp_path, preset, extra)
    assert Counter(_compute_ops(q)) == Counter(_compute_ops(m)), (preset, case)
    P._assert_same_graph(q, m, f"{preset} {case}")


@pytest.mark.parametrize("case", ["reduce_mean"])
def test_a_plain_quantizer_does_not_convert_a_reduce_mean(case):
    model, shape = _plain_model(case)
    q = _mine(
        model, _data(shape), "U8S8_AAWS", {"ConvertReduceMeanToGlobalAvgPool": True}
    )
    assert "ReduceMean" in _ops(q) and "GlobalAveragePool" not in _ops(q)


# -- a BatchNorm that stays, an all-zero constant, a model with nothing to quantize ----------


def _gemm_bn_gemm(consts=()):
    """``Gemm -> BatchNormalization -> Gemm`` on 2-D tensors: the BatchNorm cannot be
    folded or turned into a Conv (and with Constant parameters it cannot either)."""
    arrays = dict(
        w=_array("w", (16, 16)),
        b=_array("b", (16,)),
        w2=_array("w2", (16, 8)),
        b2=_array("b2", (8,)),
        bs=np.ones(16, np.float32),
        bb=np.full(16, 0.1, np.float32),
        bm=np.zeros(16, np.float32),  # (an all-zero constant)
        bv=np.ones(16, np.float32),
    )
    body = """
    g = Gemm(x, w, b)
    n = BatchNormalization(g, bs, bb, bm, bv)
    y = Gemm(n, w2, b2)
    """
    return _model(body, (4, 16), "float[4,8] y", consts, arrays), (4, 16)


@needs_quark
@pytest.mark.parametrize("preset", INT_PRESETS)
@pytest.mark.parametrize("constants", [False, True])
def test_a_batch_norm_that_stays_matches_quark(preset, constants, tmp_path):
    """Quark's BatchNorm conversion adds its op type to the list the quantizer takes
    even when it converts nothing, so a left-over BatchNorm is quantized (in the NPU
    transformer scheme too); an all-zero parameter (the mean) gets scale 1."""
    model, shape = _gemm_bn_gemm(("bs", "bb", "bm", "bv") if constants else ())
    data = _data(shape, seed=5)
    extra = _no_fold(preset) if constants else {}
    q, m = _both(model, data, tmp_path, preset, extra, False if constants else None)
    assert "BatchNormalization" in _ops(q)
    if constants:
        _same_wiring(q, m, f"{preset} constants")
    else:
        P._assert_same_graph(q, m, preset)
    for got, want in zip(_run(m, data[0]), _run(q, data[0])):
        np.testing.assert_array_equal(got, want)


def test_a_left_over_batch_norm_is_quantized_in_the_transformer_scheme():
    model, shape = _gemm_bn_gemm()
    q = _mine(model, _data(shape), "INT8_TRANSFORMER_DEFAULT")
    by_out = _by_output(q)
    bn = next(n for n in q.graph.node if n.op_type == "BatchNormalization")
    assert all(by_out[i].op_type == "DequantizeLinear" for i in bn.input)


@pytest.mark.parametrize("preset", ["A8W8", "A16W8", "U16S8_AAWS", "S16S8_ASWS"])
def test_an_all_zero_constant_has_scale_one(preset):
    """ONNX Runtime's ``compute_scale_zp``: no range, scale 1 (not ``1e-12 / 127``)."""
    model, shape = _gemm_bn_gemm()
    q = _mine(model, _data(shape), preset)
    inits = {t.name: numpy_helper.to_array(t) for t in q.graph.initializer}
    zero = [
        n
        for n in q.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in inits
        and inits[n.input[0]].size == 16
        and not inits[n.input[0]].any()
    ]
    assert len(zero) == 1
    assert float(inits[zero[0].input[1]]) == 1.0


@needs_quark
def test_a_transformer_model_with_nothing_to_quantize_comes_back_in_quarks_order(
    tmp_path,
):
    """Quark: "No quantizable ops in this model" -- the float graph (its Constant nodes
    unfolded) in Quark's topological order."""
    model, shape = _shared_and_output()
    data = _data(shape)
    q, m = _both(model, data, tmp_path, "INT8_TRANSFORMER_DEFAULT")
    assert "QuantizeLinear" not in _ops(q) and "Constant" in _ops(q)
    _same_wiring(q, m, "nothing to quantize")
    q, m = _both(model, data, tmp_path, "INT8_TRANSFORMER_DEFAULT", NO_FOLD, False)
    _same_wiring(q, m, "nothing to quantize, nothing folded")


# -- Conv -> Clip / Relu tails -------------------------------------------------------------------

CLIP_TAILS = {
    "clip6_relu": "k = Clip(c, mn, mx6)\n y = Relu(k)",
    "clip1_relu": "k = Clip(c, mn, mx1)\n y = Relu(k)",
    "relu_clip6": "k = Relu(c)\n y = Clip(k, mn, mx6)",
    "clip6_clip6_relu": "k = Clip(c, mn, mx6)\n j = Clip(k, mn, mx6)\n y = Relu(j)",
}


def _clip_tail(tail, consts=()):
    arrays = dict(
        cw=_array("cw", (4, 3, 3, 3)),
        cb=_array("cb", (4,)),
        mn=np.array(0.0, np.float32),
        mx6=np.array(6.0, np.float32),
        mx1=np.array(1.0, np.float32),
    )
    body = "c = Conv<pads=[1,1,1,1]>(x, cw, cb)\n" + CLIP_TAILS[tail]
    return _model(body, SHAPE4, "float[1,4,8,8] y", consts, arrays)


def _valid(model):
    """The graph loads (ONNX Runtime checks the wiring and the topological order)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    lib = os.environ.get("QUARK_ONNX_OPS_LIB")
    if lib:
        so.register_custom_ops_library(lib)
    ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )


@needs_quark
@pytest.mark.parametrize(
    "preset",
    [
        "A8W8",
        "U8S8_AAWS",
        "S8S8_AAWS",
        "INT8_CNN_DEFAULT",
        "U16S8_AAWS",
        "XINT8",
        "VINT8",
    ],
)
@pytest.mark.parametrize("tail", sorted(CLIP_TAILS))
@pytest.mark.parametrize("constants", [False, True])
def test_a_conv_clip_relu_tail_is_a_valid_graph_and_matches_quark(
    preset, tail, constants, tmp_path
):
    """The tails Quark folds into the Conv's quantizer (a ``Clip`` to [0, 6] or [0, 1],
    a ``Relu``), in a row: each one that is folded renames its producer's output once,
    the graph stays valid and is Quark's."""
    model = _clip_tail(tail, ("cw", "cb", "mn", "mx6", "mx1") if constants else ())
    data = _data(SHAPE4)
    extra = _no_fold(preset) if constants else {}
    q, m = _both(model, data, tmp_path, preset, extra, False if constants else None)
    _valid(m)
    if constants:
        _same_wiring(q, m, f"{preset} {tail}")
    else:
        _same_graph(q, m, f"{preset} {tail}")
    for got, want in zip(_run(m, data[0]), _run(q, data[0])):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "preset", ["A8W8", "U8S8_AAWS", "INT8_CNN_DEFAULT", "U16S8_AAWS"]
)
@pytest.mark.parametrize("tail", sorted(CLIP_TAILS))
def test_a_conv_clip_relu_tail_is_valid_without_quark(preset, tail):
    q = _mine(_clip_tail(tail), _data(SHAPE4), preset)
    _valid(q)
    produced = {o for n in q.graph.node for o in n.output}
    known = produced | {t.name for t in q.graph.initializer} | {"x"}
    assert all(i in known for n in q.graph.node for i in n.input if i)
    assert q.graph.output[0].name in produced
