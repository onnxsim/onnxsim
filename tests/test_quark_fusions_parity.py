"""Parity of onnxsim's reproduction of Quark's operator fusions against the real AMD
Quark ONNX package (``FuseInstanceNorm``, ``FuseL2Norm``, ``FuseLayerNorm``,
``FuseGelu``; :mod:`onnxsim.quark_fusions`). Skipped unless ``quark.onnx`` is importable.

What Quark runs, and when (probed on 0.13; the module docstring of
:mod:`onnxsim.quark_fusions` has the details):

- the four passes run in one ``optimize_model`` call after onnxslim and ONNX Runtime's
  basic optimizer, with all four flags on by default, in every preset -- also when
  ``OptimizeModel`` is off (VINT8), where they are the only thing fusing a LayerNorm /
  Gelu: ONNX Runtime's own optimizer fuses a torch-exported LayerNorm (opset >= 17) and
  Gelu (opset >= 20) first otherwise;
- LayerNorm needs opset >= 17 and Gelu opset >= 20; below, nothing is fused;
- ``SkipPreprocess`` skips them; ``ConvertOpsetVersion`` runs before them;
- the InstanceNorm and L2Norm matchers are Quark's own and want TensorFlow-style
  decompositions (``GlobalAveragePool`` / ``Reciprocal``, ``ReduceSum`` / ``Max``);
  the tanh form of Gelu and the other torch InstanceNorm decompositions match nothing.

The float tests compare Quark's pre-processing function with onnxslim and ONNX Runtime
off -- the fusions alone -- node by node (op types, domains, names, wiring, attributes)
and initializers. The quantized tests compare the graphs the quantizers emit (op types,
attributes, Q/DQ placement, scales, zero points, constants) and, with ONNX Runtime's
graph optimizations off, what they compute.
"""

import contextlib
import copy
import io
import os
import warnings

import numpy as np
import onnx
import pytest
from _quark_fusion_common import (
    _GELU,
    _GELU_NOT,
    _IN,
    _L2,
    F32,
    _assert_close_graph,
    _gelu_inits,
    _gelu_model,
    _graph_diff,
    _in_model,
    _l2_model,
    _ln,
    _ln_inits,
    _ln_model,
    _model,
    _ops,
    _t,
)
from onnx import numpy_helper

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

from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim import quark_fusions as qf  # noqa: E402


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files into the current directory."""
    monkeypatch.chdir(tmp_path)


# -- running Quark -----------------------------------------------------------------------


def _quark_float(model, tmp_path, **flags):
    """Quark's float pre-processing before the algorithms with onnxslim and ONNX
    Runtime's optimizer off: the four fusions (``flags``: ``FuseGelu=False``, ...)."""
    from quark.onnx.preprocess.preproc import apply_pre_optimization_before_algo

    path = tmp_path / "float.onnx"
    onnx.save(model, str(path))
    extra = {"SimplifyModel": False, **flags}
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return apply_pre_optimization_before_algo(
            copy.deepcopy(model),
            path,
            nodes_to_quantize=[],
            nodes_to_exclude=[],
            op_types_to_quantize=None,
            optimize_model_flag=False,
            extra_options=extra,
        )


def _signature(model):
    """Everything a fusion decides: every node (op, domain, name, wiring, attributes)
    in order, the initializers and the graph inputs and outputs."""
    nodes = [
        (
            n.op_type,
            n.domain,
            n.name,
            tuple(n.input),
            tuple(n.output),
            tuple(
                sorted(
                    (a.name, str(onnx.helper.get_attribute_value(a)))
                    for a in n.attribute
                )
            ),
        )
        for n in model.graph.node
    ]
    inits = sorted(
        (t.name, tuple(t.dims), t.data_type, numpy_helper.to_array(t).tobytes())
        for t in model.graph.initializer
    )
    return (
        nodes,
        inits,
        [i.name for i in model.graph.input],
        [o.name for o in model.graph.output],
    )


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


def _data(model, n=4, seed=3):
    shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(F32)} for _ in range(n)]


def _quark_quantize(model, data, tmp_path, preset="XINT8", extra=None, optimize=None):
    """Quark's preset as ``QConfig.get_default_config`` hands it out (a private copy;
    CLE off); ``optimize`` is its ``optimize_model`` field -- the ``OptimizeModel``
    extra option is ignored by the presets that are legacy ``QuantizationConfig``s."""
    from quark.onnx import ModelQuantizer, QConfig

    cfg = copy.deepcopy(QConfig.get_default_config(preset))
    cfg.global_quant_config.include_cle = False
    if optimize is not None:
        cfg.global_quant_config.optimize_model = optimize
    cfg.global_quant_config.extra_options.update(extra or {})
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ModelQuantizer(cfg).quantize_model(src, dst, _reader(data))
    return onnx.load(dst)


def _mine_quantize(model, data, preset="XINT8", extra=None, optimize=None):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    if optimize is not None:
        cfg.extra_options["OptimizeModel"] = optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            onnx.shape_inference.infer_shapes(model),
            calibration_data_reader=_reader(data),
        )


def _ort(model, x):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    lib = os.environ.get("QUARK_ONNX_OPS_LIB")
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


# -- the patterns --------------------------------------------------------------------------

_ERF_BODY = (
    "{p}h = Mul({x}, half)\n{p}d = Div({x}, rt2)\n{p}e = Erf({p}d)\n"
    "{p}a = Add({p}e, one)\n{y} = Mul({p}h, {p}a)\n"
)


def _block_model(opset=17):
    """A transformer-block-shaped graph: LayerNorm, MatMul + bias, Gelu (erf), MatMul +
    bias, residual Add, LayerNorm."""
    rng = np.random.default_rng(5)
    body = (
        _ln(opset, "x", "h1", "a_")
        + "f1 = MatMul(h1, w1)\nf1b = Add(f1, bb1)\n"
        + _ERF_BODY.format(p="g_", x="f1b", y="g")
        + "f2 = MatMul(g, w2)\nf2b = Add(f2, bb2)\nr = Add(x, f2b)\n"
        + _ln(opset, "r", "y", "b_")
    )
    inits = (
        _ln_inits(opset, "a_")
        + _ln_inits(opset, "b_")
        + _gelu_inits()
        + [
            _t("w1", rng.standard_normal((16, 32)) * 0.3),
            _t("bb1", rng.standard_normal(32) * 0.1),
            _t("w2", rng.standard_normal((32, 16)) * 0.3),
            _t("bb2", rng.standard_normal(16) * 0.1),
        ]
    )
    return _model(body, inits, opset)


def _conv_in_model(opset=13):
    """Conv -> TensorFlow-style InstanceNorm -> Relu -> Conv."""
    rng = np.random.default_rng(6)
    body = (
        "c = Conv(x, cw)\n"
        + _IN.replace("GlobalAveragePool(x)", "GlobalAveragePool(c)")
        .replace("Sub(x, mean)", "Sub(c, mean)")
        .replace("Mul(x, s)", "Mul(c, s)")
        .replace("y = Add(xs, b2)", "z = Add(xs, b2)")
        + "r = Relu(z)\ny = Conv(r, cw2)\n"
    )
    scale = (1 + rng.random((1, 4, 1, 1))).astype(F32)
    return _model(
        body,
        [
            _t("eps", 1e-3),
            _t("scale", scale),
            _t("bias", rng.standard_normal((1, 4, 1, 1))),
            _t("cw", rng.standard_normal((4, 4, 1, 1)) * 0.4),
            _t("cw2", rng.standard_normal((4, 4, 1, 1)) * 0.4),
        ],
        opset,
        shape=(2, 4, 6, 6),
    )


def _ln_variants():
    out = {}
    for opset in (13, 16, 17, 18, 20, 21):
        out[f"ln_base@{opset}"] = _ln_model(opset)
        out[f"ln_consts_as_nodes@{opset}"] = _ln_model(opset, consts="node")
    for v in ("dup_sub", "cast_pow", "cast_after_div", "pow3", "square_by_mul"):
        out[f"ln_{v}@17"] = _ln_model(17, variant=v)
    for v in ("no_affine", "no_bias", "extra_consumer", "comm_affine"):
        out[f"ln_{v}@18"] = _ln_model(18, variant=v)
    for eps in (1e-6, 1e-4, 1.0001e-4, 1e-3, 0.0):
        out[f"ln_eps={eps:g}@17"] = _ln_model(17, eps=eps)
    out["ln_eps_first@17"] = _ln_model(17, eps_first=True)
    out["ln_2d_weight@17"] = _ln_model(17, w_shape=(1, 16))
    return out


def _gelu_variants():
    out = {}
    for name in sorted(_GELU):
        for opset in (13, 19, 20, 21):
            out[f"gelu_{name}@{opset}"] = _gelu_model(name, opset)
    for name in sorted(_GELU_NOT):
        out[f"gelu_{name}@20"] = _gelu_model(name, 20, _GELU_NOT)
    return out


def _norm_variants():
    out = {
        "instance_norm@13": _in_model(opset=13),
        "instance_norm@18": _in_model(opset=18),
    }
    out["instance_norm_flat_scale"] = _in_model(flat=True)
    out["instance_norm_wrong_xs"] = _in_model(
        _IN.replace("xs = Mul(x, s)", "xs = Mul(x, x)")
    )
    out["instance_norm_swapped_add"] = _in_model(
        _IN.replace("y = Add(xs, b2)", "y = Add(b2, xs)")
    )
    out["instance_norm_neg"] = _in_model(
        _IN.replace("rc = Reciprocal(sd)", "rc = Neg(sd)")
    )
    out["l2@13"] = _l2_model()
    out["l2@13_axis_1"] = _l2_model(rax=1)
    out["l2_min"] = _l2_model(_L2.replace("mx = Max(rs, eps)", "mx = Min(rs, eps)"))
    return out


_FLOAT_CASES = {**_ln_variants(), **_gelu_variants(), **_norm_variants()}


@pytest.mark.parametrize("name", sorted(_FLOAT_CASES))
def test_float_fusions_match_quark(name, tmp_path):
    """The graph Quark's four fusions leave -- op types, domains, names, wiring,
    attributes, initializers -- is the one onnxsim leaves, pattern by pattern."""
    model = _FLOAT_CASES[name]
    q = _quark_float(model, tmp_path)
    m = qf.apply_fusions(model)
    sq, sm = _signature(q), _signature(m)
    assert sm[0] == sq[0], f"{name}: nodes differ"
    assert sm[1:] == sq[1:], f"{name}: initializers or graph inputs / outputs differ"


_FUSED_OPS = ("LayerNormalization", "Gelu", "InstanceNormalization", "LpNormalization")


def test_the_cases_cover_fusions_and_non_fusions():
    fused = {
        name
        for name, m in _FLOAT_CASES.items()
        if any(op in _FUSED_OPS for op in _ops(qf.apply_fusions(m)))
    }
    assert {
        op for name in fused for op in _ops(qf.apply_fusions(_FLOAT_CASES[name]))
    } >= set(_FUSED_OPS)
    assert len(fused) >= 25 and len(_FLOAT_CASES) - len(fused) >= 30


@pytest.mark.parametrize(
    "flags",
    [
        dict(FuseLayerNorm=False),
        dict(FuseGelu=False),
        dict(FuseInstanceNorm=False),
        dict(FuseL2Norm=False),
        dict(FuseLayerNorm=False, FuseGelu=False),
        dict(
            FuseInstanceNorm=False,
            FuseL2Norm=False,
            FuseGelu=False,
            FuseLayerNorm=False,
        ),
    ],
    ids=lambda f: "+".join(sorted(k for k in f)),
)
def test_each_flag_switches_its_pass_off_like_quark(flags, tmp_path):
    body = (
        _ln(20, "x", "h", "a_")
        + "d = Div(h, rt2)\ne = Erf(d)\nad = Add(e, one)\nmh = Mul(h, half)\ny = Mul(mh, ad)\n"
    )
    model = _model(body, _ln_inits(20, "a_") + _gelu_inits(), 20)
    kw = dict(
        instance_norm=flags.get("FuseInstanceNorm", True),
        l2_norm=flags.get("FuseL2Norm", True),
        layer_norm=flags.get("FuseLayerNorm", True),
        gelu=flags.get("FuseGelu", True),
    )
    q = _quark_float(model, tmp_path, **flags)
    m = qf.apply_fusions(model, **kw)
    assert _signature(m) == _signature(q)


@pytest.mark.parametrize("opset", [13, 16, 17, 19, 20])
def test_the_opset_gates_match_quark(opset, tmp_path):
    body = (
        _ln(opset, "x", "h", "a_")
        + "d = Div(h, rt2)\ne = Erf(d)\nad = Add(e, one)\nmh = Mul(h, half)\ny = Mul(mh, ad)\n"
    )
    model = _model(body, _ln_inits(opset, "a_") + _gelu_inits(), opset)
    q = _quark_float(model, tmp_path)
    m = qf.apply_fusions(model)
    assert _signature(m) == _signature(q)
    ops = set(_ops(q))
    assert ("LayerNormalization" in ops) == (opset >= 17)
    assert ("Gelu" in ops) == (opset >= 20)


def test_a_pattern_whose_intermediate_has_other_readers_is_not_fused_here(tmp_path):
    """Quark deletes the matched nodes whatever else reads them and then fails to sort
    a graph with a missing tensor; onnxsim keeps the pattern."""
    m = _in_model(_IN + "z = Add(y, s)\n")
    m.graph.node[-1].output[0] = "z"
    m.graph.node[-2].output[0] = "y"
    m.graph.output[0].name = "z"
    with pytest.raises(Exception):
        _quark_float(m, tmp_path)
    assert "InstanceNormalization" not in _ops(qf.apply_fusions(m))


# -- quantized graphs --------------------------------------------------------------------------

_Q_MODELS = {
    "ln@13": lambda: _ln_model(13),
    "ln@17": lambda: _ln_model(17),
    "ln@18_nodes": lambda: _ln_model(18, consts="node"),
    "ln@20_dup_sub": lambda: _ln_model(20, variant="dup_sub"),
    "gelu_torch@20": lambda: _gelu_model("torch_mul_half_last", 20),
    "gelu_torch@21": lambda: _gelu_model("torch_mul_half_first", 21),
    "gelu_torch@19": lambda: _gelu_model("torch_mul_half_first", 19),
    "gelu_keras@20": lambda: _gelu_model("keras", 20),
    "gelu_tf@20": lambda: _gelu_model("tf", 20),
    "instance_norm": lambda: _in_model(),
    "conv_instance_norm": lambda: _conv_in_model(),
    "l2": lambda: _l2_model(),
    "block@13": lambda: _block_model(13),
    "block@17": lambda: _block_model(17),
    "block@20": lambda: _block_model(20),
}


def _same_as_quark(
    model, tmp_path, preset="XINT8", extra=None, optimize=None, mine_extra=None
):
    data = _data(model)
    q = _quark_quantize(model, data, tmp_path, preset, extra, optimize)
    m = _mine_quantize(
        model, data, preset, {**(extra or {}), **(mine_extra or {})}, optimize
    )
    return q, m, data


def _assert_same(q, m, msg):
    diff = _graph_diff(q, m)
    assert not diff, f"{msg}: graphs differ: {diff}"


@pytest.mark.parametrize("preset", ["XINT8", "VINT8"])
@pytest.mark.parametrize("name", sorted(_Q_MODELS))
def test_pof2_quantized_graph_matches_quark(name, preset, tmp_path):
    """XINT8 (ONNX Runtime's optimizer on) and VINT8 (off; the fusions are the only
    thing fusing a LayerNorm / Gelu): the same graph, bit for bit, and the same output."""
    model = _Q_MODELS[name]()
    q, m, data = _same_as_quark(model, tmp_path, preset)
    _assert_same(q, m, f"{name} {preset}")
    x = data[0]["x"]
    np.testing.assert_array_equal(_ort(m, x), _ort(q, x))


def test_the_fused_ops_are_in_the_quantized_graphs(tmp_path):
    expect = {
        "ln@17": "LayerNormalization",
        "gelu_torch@20": "Gelu",
        "gelu_keras@20": "Gelu",
        "instance_norm": "InstanceNormalization",
        "l2": "LpNormalization",
    }
    for name, op in expect.items():
        model = _Q_MODELS[name]()
        q = _quark_quantize(model, _data(model), tmp_path, "XINT8")
        assert op in _ops(q), name
    # not vacuous: decomposed where Quark does not fuse
    for name, op in (("ln@13", "Pow"), ("gelu_torch@19", "Erf")):
        model = _Q_MODELS[name]()
        q = _quark_quantize(model, _data(model), tmp_path, "XINT8")
        assert op in _ops(q), name


@pytest.mark.parametrize(
    "extra, optimize",
    [
        ({"FuseLayerNorm": False}, None),
        ({"FuseGelu": False}, None),
        ({"FuseInstanceNorm": False}, None),
        ({"FuseL2Norm": False}, None),
        ({"SkipPreprocess": True}, None),
        ({"ConvertOpsetVersion": 17}, None),
        ({"ConvertOpsetVersion": 20}, None),
        ({"ConvertOpsetVersion": 18, "FuseLayerNorm": False}, None),
        ({}, False),
        ({"FuseLayerNorm": False, "FuseGelu": False}, False),
        ({"SkipPreprocess": True}, False),
        ({"SimplifyModel": False}, None),
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize(
    "name", ["ln@13", "ln@17", "gelu_torch@20", "gelu_keras@20", "instance_norm", "l2"]
)
def test_options_match_quark(name, extra, optimize, tmp_path):
    model = _Q_MODELS[name]()
    q, m, data = _same_as_quark(model, tmp_path, "XINT8", extra, optimize)
    _assert_same(q, m, f"{name} {extra} optimize={optimize}")


@pytest.mark.parametrize("preset", ["A8W8", "A16W8", "U8U8_AAWA", "S8S8_AAWS"])
@pytest.mark.parametrize(
    "name", ["ln@17", "gelu_torch@20", "gelu_keras@20", "block@17"]
)
def test_float_scale_presets_match_quark(name, preset, tmp_path):
    """The presets with float scales: equal graphs up to float32 rounding of the
    scales (and, on aarch64, of the calibration)."""
    model = _Q_MODELS[name]()
    q, m, data = _same_as_quark(model, tmp_path, preset)
    _assert_close_graph(q, m, f"{name} {preset}")
    x = data[0]["x"]
    ref, got = _ort(q, x), _ort(m, x)
    np.testing.assert_allclose(got, ref, atol=0.1 * float(np.abs(ref).max()))


def test_vint8_does_not_quantize_what_its_registries_lack(tmp_path):
    """VINT8's op list is the model's own op types (taken before the fusions) plus the
    registry's: an LpNormalization only the fusion introduces is not quantized, and a
    LayerNormalization fed by a graph input is not either (no
    ForceQuantizeNoInputCheck)."""
    for name, quantized in (("l2", False), ("ln@17", False)):
        model = _Q_MODELS[name]()
        q, m, _ = _same_as_quark(model, tmp_path, "VINT8")
        _assert_same(q, m, name)
        assert any(n.op_type == "QuantizeLinear" for n in q.graph.node) is quantized
    model = _Q_MODELS["l2"]()
    q = _quark_quantize(model, _data(model), tmp_path, "XINT8")
    assert any(n.op_type == "QuantizeLinear" for n in q.graph.node)


@pytest.mark.parametrize(
    "name", ["ln@17", "ln@18_nodes", "ln@20_dup_sub", "gelu_torch@20", "gelu_torch@21"]
)
def test_without_the_runtime_optimizers_onnxsim_stands_in_for_onnx_runtime(
    name, tmp_path
):
    """``UseRuntimeOptimizers=False``: no onnxslim / ONNX Runtime run here, so ONNX
    Runtime's own LayerNorm / Gelu fusions are reproduced; the graph is Quark's with
    them on. (Its other fusions -- MatMul + Add into a Gemm -- are not.)"""
    model = _Q_MODELS[name]()
    data = _data(model)
    q = _quark_quantize(model, data, tmp_path, "XINT8")
    m = _mine_quantize(model, data, "XINT8", {"UseRuntimeOptimizers": False})
    _assert_same(q, m, name)


def test_the_stand_in_leaves_the_keras_gelu_to_quarks_own_pass(tmp_path):
    """ONNX Runtime's optimizer fuses the PyTorch Gelu shape only; a Keras one reaches
    Quark's own pass, whose Gelu is the contrib op."""
    model = _Q_MODELS["gelu_keras@20"]()
    data = _data(model)
    q = _quark_quantize(model, data, tmp_path, "XINT8")
    m = _mine_quantize(model, data, "XINT8", {"UseRuntimeOptimizers": False})
    _assert_same(q, m, "keras")
    (g,) = [n for n in q.graph.node if n.op_type == "Gelu"]
    assert g.domain == "com.microsoft"


def test_the_fused_float_graph_computes_what_the_decomposed_one_does(tmp_path):
    x = np.random.default_rng(9).standard_normal((2, 4, 16)).astype(F32)
    for name in (
        "ln@17",
        "ln@18_nodes",
        "gelu_torch@20",
        "gelu_keras@20",
        "gelu_tf@20",
        "block@17",
    ):
        model = _Q_MODELS[name]()
        q = _quark_float(model, tmp_path)
        m = qf.apply_fusions(model)
        ref = _ort(model, x)
        np.testing.assert_allclose(_ort(q, x), ref, rtol=1e-4, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(_ort(m, x), ref, rtol=1e-4, atol=1e-5, err_msg=name)
