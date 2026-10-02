"""Quark's operator fusions (``FuseInstanceNorm``, ``FuseL2Norm``, ``FuseLayerNorm`` and
``FuseGelu``; :mod:`onnxsim.quark_fusions`) and their place in the compat flow, without
Quark. The parity of every pattern against the real package is
``tests/test_quark_fusions_parity.py``; the models are ``tests/_quark_fusion_common.py``.
"""

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
    _attrs,
    _gelu_inits,
    _gelu_model,
    _in_model,
    _l2_model,
    _ln,
    _ln_inits,
    _ln_model,
    _model,
    _ops,
    _run,
    _t,
)

from onnxsim import quark_compat as qc
from onnxsim import quark_fusions as qf
from onnxsim import quark_marking as qm

# -- LayerNorm -------------------------------------------------------------------------


@pytest.mark.parametrize("consts", ["init", "node"])
@pytest.mark.parametrize("opset", [13, 16, 17, 18, 20, 21])
def test_layer_norm_is_fused_from_opset_17(opset, consts):
    m = qf.apply_fusions(_ln_model(opset, consts=consts))
    if opset < 17:
        assert "LayerNormalization" not in _ops(m)
        assert "Pow" in _ops(m)
        return
    assert _ops(m) == ["LayerNormalization"]
    (n,) = m.graph.node
    # epsilon only (the default axis -1), no stash_type; the affine constants are the
    # fused node's inputs and what nothing reads any more is gone
    assert _attrs(n) == {"epsilon": pytest.approx(1e-5, rel=1e-6)}
    assert (n.name, n.domain) == ("LayerNorm_0", "")
    assert list(n.input) == ["x", "w", "b"] and list(n.output) == ["y"]
    assert {t.name for t in m.graph.initializer} == {"w", "b"}


@pytest.mark.parametrize("variant", ["dup_sub", "cast_pow", "cast_after_div"])
def test_layer_norm_variants_older_exporters_emit_are_fused(variant):
    m = qf.apply_fusions(_ln_model(17, variant=variant))
    assert _ops(m) == ["LayerNormalization"]


@pytest.mark.parametrize(
    "kw",
    [
        dict(eps=1e-3),
        dict(eps=0.0),
        dict(eps=1.0001e-4),
        dict(variant="pow3"),
        dict(variant="square_by_mul"),
        dict(variant="no_affine"),
        dict(variant="no_bias"),
        dict(variant="extra_consumer"),
        dict(w_shape=(1, 16)),
        # (Quark reads the epsilon off the Add's first input only; ONNX Runtime's own
        # fusion takes the other order too)
        dict(eps_first=True),
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()),
)
def test_layer_norm_patterns_quark_does_not_match(kw):
    base = _ln_model(17, **kw)
    m = qf.apply_fusions(base)
    assert "LayerNormalization" not in _ops(m)
    assert sorted(_ops(m)) == sorted(_ops(base))


def test_layer_norm_commuted_scale_and_bias_are_fused():
    assert _ops(qf.apply_fusions(_ln_model(17, variant="comm_affine"))) == [
        "LayerNormalization"
    ]


def test_layer_norm_epsilon_limit_is_float32_1e_minus_4():
    assert "LayerNormalization" in _ops(qf.apply_fusions(_ln_model(17, eps=1e-4)))
    assert "LayerNormalization" not in _ops(
        qf.apply_fusions(_ln_model(17, eps=1.0001e-4))
    )


def test_layer_norm_epsilon_may_be_a_one_element_vector():
    m = qf.apply_fusions(_ln_model(17, eps=1e-5))
    assert _attrs(m.graph.node[0])["epsilon"] == pytest.approx(1e-5, rel=1e-6)
    v = _ln_model(17)
    for t in v.graph.initializer:
        if t.name == "eps":
            t.CopyFrom(_t("eps", [1e-6]))
    assert _attrs(qf.apply_fusions(v).graph.node[0])["epsilon"] == pytest.approx(
        1e-6, rel=1e-6
    )


def test_a_layer_norm_whose_intermediate_is_a_graph_output_is_kept():
    # (Quark would delete the tensor and write a broken graph)
    m = _ln_model(17)
    m.graph.output.append(onnx.helper.make_tensor_value_info("n", 1, None))
    assert "LayerNormalization" not in _ops(qf.apply_fusions(m))


def test_two_layer_norms_are_numbered_and_appended_after_the_rest():
    body = (
        _ln(17, "x", "h", "a_")
        + "r = Relu(h)\n"
        + _ln(17, "r", "y", "b_")
        + "z = Sigmoid(y)"
    )
    m = _model(body, _ln_inits(17, "a_") + _ln_inits(17, "b_"))
    m.graph.output[0].name = "z"
    out = qf.apply_fusions(m)
    # Quark appends each fused node to the end of the graph (the next sort moves it)
    assert _ops(out) == ["Relu", "Sigmoid", "LayerNormalization", "LayerNormalization"]
    assert [n.name for n in out.graph.node if n.op_type == "LayerNormalization"] == [
        "LayerNorm_0",
        "LayerNorm_1",
    ]
    assert qm.quark_sorted(out).graph.node[-1].op_type == "Sigmoid"


@pytest.mark.parametrize("opset", [17, 18])
def test_the_fused_layer_norm_computes_the_decomposed_one(opset):
    m = _ln_model(opset)
    x = np.random.default_rng(1).standard_normal((2, 4, 16)).astype(F32)
    np.testing.assert_allclose(
        _run(qf.apply_fusions(m), x), _run(m, x), rtol=1e-4, atol=1e-5
    )


def test_the_layer_norm_axis_is_not_looked_at():
    # Quark matches the pattern, not the reduced axes: a normalization over axis 1
    # becomes a LayerNormalization over the last axis, and so does it here
    m = _ln_model(17)
    for n in m.graph.node:
        if n.op_type == "ReduceMean":
            del n.attribute[:]
            n.attribute.append(onnx.helper.make_attribute("axes", [1]))
    out = qf.apply_fusions(m)
    assert _ops(out) == ["LayerNormalization"]
    assert "axis" not in _attrs(out.graph.node[0])


# -- Gelu ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_GELU))
def test_gelu_is_fused_to_the_contrib_op(name):
    m = qf.apply_fusions(_gelu_model(name))
    gelus = [n for n in m.graph.node if n.op_type == "Gelu"]
    assert len(gelus) == 1
    (g,) = gelus
    # com.microsoft Gelu without attributes (not the ai.onnx one), named Gelu_<n>
    assert (g.domain, g.name, len(g.attribute)) == ("com.microsoft", "Gelu_0", 0)
    assert list(g.output) == ["y"]
    assert list(g.input) == (["x"] if name.startswith("torch") else ["r"])
    assert {o.domain for o in m.opset_import} >= {"", "com.microsoft"}
    for op in ("Erf", "Div"):
        assert op not in _ops(m)


@pytest.mark.parametrize("name", sorted(_GELU_NOT))
def test_gelu_patterns_quark_does_not_match(name):
    m = _gelu_model(name, table=_GELU_NOT)
    out = qf.apply_fusions(m)
    assert "Gelu" not in _ops(out)
    # (Quark's InstanceNorm pass sorts the graph, which moves the Constants first)
    assert sorted(_ops(out)) == sorted(_ops(m))


@pytest.mark.parametrize("opset", [13, 17, 19])
def test_gelu_is_left_alone_below_opset_20(opset):
    m = _gelu_model("torch_mul_half_first", opset)
    assert _ops(qf.apply_fusions(m)) == _ops(m)


def test_gelu_constants_are_matched_with_a_tolerance():
    for kw, fused in (
        (dict(rt2=1.4142), True),
        (dict(rt2=1.41), False),
        (dict(half=0.5001), False),
        (dict(one=1.0000001), True),
        (dict(one=1.001), False),
    ):
        m = _gelu_model("torch_mul_half_first")
        for t in m.graph.initializer:
            if t.name in kw:
                t.CopyFrom(_t(t.name, kw[t.name]))
        assert ("Gelu" in _ops(qf.apply_fusions(m))) is fused, kw


def test_the_fused_gelu_computes_the_decomposed_one():
    x = np.random.default_rng(2).standard_normal((2, 4, 16)).astype(F32)
    for name in ("torch_mul_half_first", "torch_mul_half_last", "keras", "tf"):
        m = _gelu_model(name)
        np.testing.assert_allclose(
            _run(qf.apply_fusions(m), x), _run(m, x), rtol=1e-4, atol=1e-5, err_msg=name
        )


def test_two_gelus_in_a_row():
    body = (
        "h = Mul(x, half)\nd = Div(x, rt2)\ne = Erf(d)\na = Add(e, one)\ng = Mul(h, a)\n"
        "h2 = Mul(g, half)\nd2 = Div(g, rt2)\ne2 = Erf(d2)\na2 = Add(e2, one)\ny = Mul(h2, a2)"
    )
    m = qf.apply_fusions(_model(body, _gelu_inits(), 20))
    assert _ops(m) == ["Gelu", "Gelu"]
    assert [n.name for n in m.graph.node] == ["Gelu_0", "Gelu_1"]


# -- InstanceNorm ------------------------------------------------------------------------


def test_tensorflow_instance_norm_becomes_instance_normalization():
    m = _in_model()
    out = qf.apply_fusions(m)
    assert _ops(out) == ["InstanceNormalization"]
    (n,) = out.graph.node
    assert list(n.input) == ["x", "scale", "bias"] and list(n.output) == ["y"]
    assert n.name == ""  # (named after the final Add, which had none)
    assert _attrs(n)["epsilon"] == pytest.approx(1e-3, rel=1e-6)
    inits = {t.name: t for t in out.graph.initializer}
    assert set(inits) == {"scale", "bias"}  # eps goes
    assert list(inits["scale"].dims) == [4] and list(inits["bias"].dims) == [4]
    x = np.random.default_rng(3).standard_normal((2, 4, 6, 6)).astype(F32)
    np.testing.assert_allclose(_run(out, x), _run(m, x), rtol=1e-3, atol=1e-4)


def test_instance_norm_node_keeps_the_name_of_the_final_add():
    m = _in_model()
    m.graph.node[-1].name = "inorm"
    assert qf.apply_fusions(m).graph.node[0].name == "inorm"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.replace("rc = Reciprocal(sd)", "rc = Neg(sd)"),
        lambda b: b.replace("sd = Sqrt(ve)", "sd = Relu(ve)"),
        lambda b: b.replace("var = GlobalAveragePool(sq)", "var = ReduceMean(sq)"),
        lambda b: b.replace("y = Add(xs, b2)", "y = Add(b2, xs)"),
        # one more reader of the scale: Quark deletes the node and breaks the graph
        lambda b: b + "z = Add(y, s)\n",
    ],
    ids=["neg", "relu", "reduce_mean", "swapped_add", "shared_scale"],
)
def test_instance_norm_patterns_quark_does_not_match(mutate):
    body = mutate(_IN)
    m = _in_model(body)
    if "z = Add" in body:
        m.graph.node[-1].output[0] = "z"
        m.graph.output[0].name = "z"
        m.graph.node[-2].output[0] = "y"
    out = qf.apply_fusions(m)
    assert "InstanceNormalization" not in _ops(out)


def test_instance_norm_with_flat_scale_is_not_fused():
    # the scale is reshaped by dims[1]: a vector has no such dim
    assert "InstanceNormalization" not in _ops(qf.apply_fusions(_in_model(flat=True)))


def test_instance_norm_does_not_look_at_the_multiplication_by_x():
    # Quark checks the op type of the Mul feeding the final Add, not its inputs
    body = _IN.replace("xs = Mul(x, s)", "xs = Mul(x, x)")
    assert _ops(qf.apply_fusions(_in_model(body))) == ["InstanceNormalization"]


# -- L2 normalization ----------------------------------------------------------------------


def test_l2_normalization_becomes_lp_normalization():
    m = _l2_model()
    out = qf.apply_fusions(m)
    assert _ops(out) == ["Unsqueeze", "LpNormalization"]
    n = out.graph.node[1]
    assert _attrs(n) == {"p": 2}  # no axis: the default, the last
    assert list(n.input) == ["u"] and list(n.output) == ["y"]
    assert {t.name for t in out.graph.initializer} == {"ax"}
    x = np.random.default_rng(4).standard_normal((3, 8)).astype(F32)
    np.testing.assert_allclose(_run(out, x), _run(m, x), rtol=1e-4, atol=1e-6)


def test_l2_normalization_axis_is_not_looked_at():
    out = qf.apply_fusions(_l2_model(rax=1))
    assert _ops(out) == ["Unsqueeze", "LpNormalization"]
    assert "axis" not in _attrs(out.graph.node[1])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.replace("mx = Max(rs, eps)", "mx = Min(rs, eps)"),
        lambda b: b.replace("rc = Reciprocal(sr)", "rc = Neg(sr)"),
        lambda b: b.replace("sq = Mul(u, u)", "sq = Add(u, u)"),
    ],
    ids=["min", "neg", "add"],
)
def test_l2_patterns_quark_does_not_match(mutate):
    out = qf.apply_fusions(_l2_model(mutate(_L2)))
    assert "LpNormalization" not in _ops(out)


# -- the passes and their switches ----------------------------------------------------------


def _all_patterns_model():
    """A Conv-free graph with the four patterns in a row (opset 20)."""
    rng = np.random.default_rng(0)
    body = (
        _ln(20, "x", "h", "a_")
        + "d = Div(h, rt2)\ne = Erf(d)\nad = Add(e, one)\nmh = Mul(h, half)\ny = Mul(mh, ad)\n"
    )
    inits = _ln_inits(20, "a_") + _gelu_inits()
    del rng
    return _model(body, inits, 20)


def test_every_flag_switches_its_pass_off():
    m = _all_patterns_model()
    assert sorted(_ops(qf.apply_fusions(m))) == ["Gelu", "LayerNormalization"]
    assert "Gelu" not in _ops(qf.apply_fusions(m, gelu=False))
    assert "LayerNormalization" not in _ops(qf.apply_fusions(m, layer_norm=False))
    assert sorted(_ops(qf.apply_fusions(m, gelu=False, layer_norm=False))) == sorted(
        _ops(m)
    )


def test_the_input_model_is_not_modified():
    m = _all_patterns_model()
    before = m.SerializeToString()
    qf.apply_fusions(m)
    qf.fuse_instance_norm(_in_model())
    assert m.SerializeToString() == before


def test_ort_style_writes_what_the_runtime_optimizer_writes():
    ln = qf.apply_fusions(_ln_model(17), style="ort")
    (n,) = ln.graph.node
    assert (n.op_type, n.domain) == ("LayerNormalization", "")
    assert _attrs(n) == {
        "stash_type": 1,
        "axis": -1,
        "epsilon": pytest.approx(1e-5, rel=1e-6),
    }
    g = qf.apply_fusions(_gelu_model("torch_mul_half_first"), style="ort")
    (n,) = g.graph.node
    assert (n.op_type, n.domain) == ("Gelu", "")
    assert _attrs(n) == {"approximate": b"none"}
    # (it fuses the PyTorch shape only, and takes a larger epsilon)
    assert "Gelu" not in _ops(qf.apply_fusions(_gelu_model("keras"), style="ort"))
    assert "LayerNormalization" in _ops(
        qf.apply_fusions(_ln_model(17, eps=1e-3), style="ort")
    )
    assert "LayerNormalization" in _ops(
        qf.apply_fusions(_ln_model(17, eps_first=True), style="ort")
    )


# -- the compat flow --------------------------------------------------------------------------


class _Reader:
    def __init__(self, data):
        self._it = iter(data)

    def get_next(self):
        return next(self._it, None)


def _data(shape, n=4, seed=3):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal(shape).astype(F32)} for _ in range(n)]


def _quantize(model, preset="XINT8", **extra):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra)
    shape = tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    model = onnx.shape_inference.infer_shapes(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data(shape))
        )


def _core(m):
    return [n.op_type for n in m.graph.node if "Linear" not in n.op_type]


# (no onnxslim / ONNX Runtime graph optimizers: what is left is Quark's own fusions)
_NO_OPT = dict(OptimizeModel=False, SimplifyModel=False)


@pytest.mark.parametrize("preset", ["XINT8", "A8W8", "VINT8"])
def test_a_decomposed_layer_norm_is_quantized_as_one_op(preset):
    m = _ln_model(17)
    q = _quantize(m, preset, **_NO_OPT)
    assert _core(q) == ["LayerNormalization"]
    x = _data((2, 4, 16))[0]["x"]
    ref, got = _run(m, x), _run(q, x)
    assert np.abs(got - ref).max() < 0.15 * np.abs(ref).max()


def test_vint8_leaves_a_layer_norm_fed_by_a_graph_input_in_float():
    # VINT8 is the one preset without ForceQuantizeNoInputCheck: QDQLayerNorm touches
    # nothing unless its input was marked by a node before it
    q = _quantize(_ln_model(17), "VINT8", **_NO_OPT)
    assert [n.op_type for n in q.graph.node] == ["LayerNormalization"]
    inits = {t.name for t in q.graph.initializer}
    assert {"w", "b"} <= inits
    # ... and quantizes it behind a node that marked its input
    body = "r = Tanh(x)\n" + _ln(17, "r", "y")
    q = _quantize(_model(body, _ln_inits(17)), "VINT8", **_NO_OPT)
    (ln,) = [n for n in q.graph.node if n.op_type == "LayerNormalization"]
    # scale and bias are dequantized constants, the output goes through a Q/DQ pair
    assert not {"w", "b"} & set(ln.input) and ln.output[0] != "y"
    assert q.graph.node[-1].op_type == "DequantizeLinear"


def test_the_fused_gelu_is_quantized_like_the_ai_onnx_one():
    m = _gelu_model("torch_mul_half_first")
    q = _quantize(m, "XINT8", **_NO_OPT)
    assert _core(q) == ["Gelu"]
    (g,) = [n for n in q.graph.node if n.op_type == "Gelu"]
    assert g.domain == "com.microsoft"
    assert g.input[0].endswith("_DequantizeLinear_Output") or "dq" in g.input[0]
    x = _data((2, 4, 16))[0]["x"]
    ref, got = _run(m, x), _run(q, x)
    assert np.abs(got - ref).max() < 0.15 * np.abs(ref).max()


@pytest.mark.parametrize("preset", ["XINT8", "A8W8"])
def test_instance_norm_and_l2_norm_are_fused_before_quantization(preset):
    q = _quantize(_in_model(), preset, **_NO_OPT)
    assert _core(q) == ["InstanceNormalization"]
    q = _quantize(_l2_model(), preset, **_NO_OPT)
    assert "LpNormalization" in _core(q) and "ReduceSum" not in _core(q)


def test_an_op_only_a_fusion_introduces_is_not_on_vint8s_list():
    # QuantizeAllOpTypes lists the op types of the model as given (before the
    # fusions): the LpNormalization no registry of VINT8 has stays float
    q = _quantize(_l2_model(), "VINT8", **_NO_OPT)
    assert [n.op_type for n in q.graph.node if n.op_type != "Unsqueeze"] == [
        "LpNormalization"
    ]
    # (and in XINT8, whose registry has it, it is quantized)
    q = _quantize(_l2_model(), "XINT8", **_NO_OPT)
    assert any(n.op_type == "QuantizeLinear" for n in q.graph.node)


@pytest.mark.parametrize(
    "extra, left",
    [
        (dict(FuseLayerNorm=False), "ReduceMean"),
        (dict(SkipPreprocess=True), "ReduceMean"),
    ],
    ids=["FuseLayerNorm=False", "SkipPreprocess"],
)
def test_the_switches_keep_the_decomposed_layer_norm(extra, left):
    q = _quantize(_ln_model(17), "XINT8", **_NO_OPT, **extra)
    assert left in _core(q) and "LayerNormalization" not in _core(q)


def test_flags_off_for_the_other_passes_do_not_stop_the_layer_norm():
    q = _quantize(
        _ln_model(17),
        "XINT8",
        FuseGelu=False,
        FuseInstanceNorm=False,
        FuseL2Norm=False,
        **_NO_OPT,
    )
    assert _core(q) == ["LayerNormalization"]


def test_nothing_is_fused_below_the_opsets():
    q = _quantize(_ln_model(13), "XINT8", **_NO_OPT)
    assert "LayerNormalization" not in _core(q) and "Pow" in _core(q)
    q = _quantize(_gelu_model("torch_mul_half_first", 19), "XINT8", **_NO_OPT)
    assert "Gelu" not in _core(q) and "Erf" in _core(q)


@pytest.mark.parametrize("target", [17, 20])
def test_convert_opset_version_makes_a_layer_norm_fusable(target):
    m = _ln_model(13)
    q = _quantize(m, "XINT8", ConvertOpsetVersion=target, **_NO_OPT)
    assert _core(q) == ["LayerNormalization"]
    assert {o.version for o in q.opset_import if o.domain == ""} == {target}


def test_a_failed_opset_conversion_is_a_warning_not_an_error():
    cfg = qc.QConfig.get_default_config("XINT8")
    cfg.extra_options.update(ConvertOpsetVersion=3, **_NO_OPT)
    m = onnx.shape_inference.infer_shapes(_ln_model(17))
    with pytest.warns(UserWarning, match="opset conversion skipped"):
        qc.ModelQuantizer(cfg).quantize_model(
            m, calibration_data_reader=_Reader(_data((2, 4, 16)))
        )


def test_without_the_runtime_optimizers_the_ort_fusions_are_stood_in_for():
    # UseRuntimeOptimizers=False: no ONNX Runtime run, so its own LayerNorm fusion
    # (stash_type / axis written) is reproduced with OptimizeModel on, and Quark's
    # (epsilon only) with it off
    q = _quantize(_ln_model(17), "XINT8", UseRuntimeOptimizers=False)
    (ln,) = [n for n in q.graph.node if n.op_type == "LayerNormalization"]
    assert set(_attrs(ln)) == {"stash_type", "axis", "epsilon"}
    q = _quantize(
        _ln_model(17), "XINT8", UseRuntimeOptimizers=False, OptimizeModel=False
    )
    (ln,) = [n for n in q.graph.node if n.op_type == "LayerNormalization"]
    assert set(_attrs(ln)) == {"epsilon"}
    q = _quantize(
        _gelu_model("torch_mul_half_first"), "XINT8", UseRuntimeOptimizers=False
    )
    (g,) = [n for n in q.graph.node if n.op_type == "Gelu"]
    assert (g.domain, set(_attrs(g))) == ("", {"approximate"})


# -- marking ------------------------------------------------------------------------------------


def _ln_node_model(first_op=None):
    body = "y = LayerNormalization<axis=-1, epsilon=1e-5>(x, w, b)"
    if first_op:
        body = (
            f"r = {first_op}(x)\ny = LayerNormalization<axis=-1, epsilon=1e-5>(r, w, b)"
        )
    return _model(body, [_t("w", np.ones(16)), _t("b", np.zeros(16))])


def test_layer_normalization_marks_nothing_unless_forced_or_fed_by_a_marked_tensor():
    types = {"LayerNormalization", "Relu"}
    m = _ln_node_model()
    assert qm.skipped_nodes(m, types, force_no_input_check=False) == {"y"}
    assert qm.skipped_nodes(m, types, force_no_input_check=True) == set()
    # behind a Relu fed by a graph input nothing is marked either; behind a Gelu is
    m = _ln_node_model("Relu")
    assert qm.skipped_nodes(m, types, force_no_input_check=False) == {"r", "y"}
    m = _ln_node_model("Tanh")
    assert qm.skipped_nodes(m, types | {"Tanh"}, force_no_input_check=False) == set()


def test_the_contrib_gelu_is_visited_like_the_ai_onnx_one():
    body = "g = Gelu(x)\nr = Relu(g)\ny = Sigmoid(r)"
    m = _model(body)
    m.graph.node[0].domain = "com.microsoft"
    # a Relu is quantized only behind a marked tensor: the Gelu marks its output
    assert qm.skipped_nodes(m, {"Gelu", "Relu"}, force_no_input_check=False) == set()
    plain = _model(body)
    plain.graph.node[0].domain = "com.example"
    assert qm.skipped_nodes(plain, {"Gelu", "Relu"}, force_no_input_check=False) == {
        "r"
    }
