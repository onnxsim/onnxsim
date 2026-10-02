"""Quark-free tests of the Quark-compat algorithms: cross-layer equalization
(``onnxsim.quark_equalization``), SmoothQuant (``onnxsim.quark_smoothquant``),
Quark-style bias correction (``onnxsim.quark_bias_correction``) and the
AutoMixprecision forms (``onnxsim.quark_auto_mixprecision``). Their parity with
the real Quark is in ``tests/test_quark_algo_parity.py``."""

import json
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_auto_mixprecision as amp
from onnxsim import quark_compat as qc
from onnxsim.quark_bias_correction import correct_bias_quark
from onnxsim.quark_equalization import (
    apply_cle_config,
    equalize,
    replace_clip6_with_relu,
    stem_equalize,
)
from onnxsim.quark_smoothquant import apply_smooth_quant_config, smooth_quant


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _model(body, inits, inputs="float[1,3,8,8] x", outputs="float[1,4,8,8] y"):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g ({inputs}) => ({outputs}) {{ {body} }}'
    )
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return m


def _run(m, x):
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    return s.run(None, {"x": x})[0]


def _arr(m, name):
    return numpy_helper.to_array(next(i for i in m.graph.initializer if i.name == name))


def _same(a, b):
    return all(
        np.array_equal(numpy_helper.to_array(x), numpy_helper.to_array(y))
        for x, y in zip(a.graph.initializer, b.graph.initializer)
    )


# -- CLE -------------------------------------------------------------------------


def _conv_chain():
    rng = np.random.default_rng(0)
    return _model(
        """c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
        r0 = Relu(c0)
        c1 = Conv<group=8, pads=[1,1,1,1]>(r0, w1, b1)
        r1 = Relu(c1)
        c2 = Conv<group=1>(r1, w2, b2)
        r2 = Relu(c2)
        y = Conv<group=1>(r2, w3)""",
        dict(
            w0=_w(rng, 8, 3, 3, 3, scale=2.0),
            b0=_w(rng, 8),
            w1=_w(rng, 8, 1, 3, 3),
            b1=_w(rng, 8),
            w2=_w(rng, 4, 8, 1, 1, scale=0.3),
            b2=_w(rng, 4),
            w3=_w(rng, 4, 4, 1, 1, scale=0.2),
        ),
    )


X = np.random.default_rng(1).standard_normal((1, 3, 8, 8)).astype(np.float32)


@pytest.mark.parametrize("steps", [1, 3, -1])
@pytest.mark.parametrize("append_bias", [True, False])
def test_cle_preserves_the_function_and_moves_the_weights(steps, append_bias):
    m = _conv_chain()
    out = equalize(m, steps=steps, scale_append_bias=append_bias)
    assert not _same(m, out)
    np.testing.assert_allclose(_run(out, X), _run(m, X), rtol=1e-4, atol=1e-4)


def test_cle_depthwise_triple_balances_the_three_ranges():
    m = _conv_chain()
    out = equalize(m, steps=1)
    r0 = np.abs(_arr(out, "w0")).reshape(8, -1).max(1)
    r1 = np.abs(_arr(out, "w1")).reshape(8, -1).max(1)
    r2 = np.abs(_arr(out, "w2")).max(axis=(0, 2, 3))
    spread = lambda *rs: np.max(np.max(rs, 0) / np.min(rs, 0))  # noqa: E731
    before = spread(
        np.abs(_arr(m, "w0")).reshape(8, -1).max(1),
        np.abs(_arr(m, "w1")).reshape(8, -1).max(1),
        np.abs(_arr(m, "w2")).max(axis=(0, 2, 3)),
    )
    assert spread(r0, r1, r2) < before


def _pair():
    rng = np.random.default_rng(12)
    return _model(
        "c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0) r0 = Relu(c0) y = Conv<group=1>(r0, w1)",
        dict(
            w0=_w(rng, 8, 3, 3, 3, scale=2),
            b0=_w(rng, 8, scale=30),
            w1=_w(rng, 4, 8, 1, 1),
        ),
    )


def test_cle_options_change_the_result():
    m = _pair()
    base = equalize(m, steps=1)
    assert not _same(m, base)
    # channels whose two ranges sum to less than the threshold are left alone
    assert _same(m, equalize(m, steps=1, weight_threshold=1e6))
    assert not _same(
        m, equalize(m, steps=1, weight_threshold=1e6, scale_use_threshold=False)
    )
    assert not _same(base, equalize(m, steps=1, scale_append_bias=False))
    assert not _same(base, equalize(m, steps=-1))
    assert _same(m, equalize(m, steps=0))
    with pytest.raises(ValueError, match="balance"):
        equalize(m, balance_method="mean")


def test_cle_skips_nodes_it_may_not_touch():
    m = _conv_chain()
    names = [n.name for n in m.graph.node]
    assert _same(m, equalize(m, nodes_to_exclude=[n for n in names if "Conv" in n]))
    only = equalize(m, nodes_to_quantize=["n0_Conv", "n2_Conv"], steps=1)
    assert not _same(m, only)
    # a Conv without an explicit ``group`` attribute is not supported (Quark)
    rng = np.random.default_rng(2)
    nog = _model(
        "c0 = Conv(x, w0, b0) r0 = Relu(c0) y = Conv(r0, w1)",
        dict(w0=_w(rng, 8, 3, 3, 3, scale=2), b0=_w(rng, 8), w1=_w(rng, 4, 8, 1, 1)),
    )
    assert _same(nog, equalize(nog))
    # an intermediate tensor that is also a graph output must keep its value
    shared = _model(
        "c0 = Conv<group=1>(x, w0, b0) r0 = Relu(c0) y = Conv<group=1>(r0, w1)",
        dict(w0=_w(rng, 8, 3, 3, 3, scale=2), b0=_w(rng, 8), w1=_w(rng, 4, 8, 1, 1)),
        outputs="float[1,4,8,8] y, float[1,8,8,8] r0",
    )
    assert _same(shared, equalize(shared))


def test_cle_replace_clip6_relu_enables_clip_chains():
    rng = np.random.default_rng(3)
    m = _model(
        "c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0) r0 = Clip(c0, lo, hi) y = Conv<group=1>(r0, w1)",
        dict(
            w0=_w(rng, 8, 3, 3, 3, scale=2),
            b0=_w(rng, 8),
            w1=_w(rng, 4, 8, 1, 1),
            lo=np.float32(0),
            hi=np.float32(6),
        ),
    )
    assert _same(m, equalize(m))
    out = equalize(m, replace_clip6=True)
    assert [n.op_type for n in out.graph.node] == ["Conv", "Relu", "Conv"]
    assert not _same(replace_clip6_with_relu(m), out)
    assert {i.name for i in out.graph.initializer} == {"w0", "b0", "w1"}


def _stem(activation="Relu"):
    rng = np.random.default_rng(4)
    c = 8
    ramp = np.linspace(0.05, 2, c).astype(np.float32)
    return _model(
        f"""c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0)
        r0 = {activation}(c0)
        bn = BatchNormalization(r0, g, be, mu, var)
        y = Conv<group=1>(bn, w1)""",
        dict(
            w0=_w(rng, c, 3, 3, 3) * ramp[:, None, None, None],
            b0=_w(rng, c),
            g=np.abs(_w(rng, c)) + 0.5,
            be=_w(rng, c),
            mu=_w(rng, c),
            var=np.abs(_w(rng, c)) + 0.5,
            w1=_w(rng, 4, c, 1, 1),
        ),
    )


def test_stem_equalization_scales_weak_channels_up_and_folds_into_bn():
    m = _stem()
    out = stem_equalize(m, ["Conv"])
    np.testing.assert_allclose(_run(out, X), _run(m, X), rtol=1e-4, atol=1e-4)
    w0, w = _arr(out, "w0"), _arr(m, "w0")
    s = np.abs(w0).reshape(8, -1).max(1) / np.abs(w).reshape(8, -1).max(1)
    assert s.min() >= 1 - 1e-6 and s.max() <= 16 + 1e-4 and s.max() > 2
    absmax = np.abs(w).reshape(8, -1).max(1)
    np.testing.assert_allclose(s, np.clip(absmax.max() / absmax, 1, 16), rtol=1e-5)
    assert s[absmax.argmax()] == pytest.approx(1.0) and s.max() == pytest.approx(16.0)
    np.testing.assert_allclose(_arr(out, "g") * s, _arr(m, "g"), rtol=1e-5)
    np.testing.assert_allclose(_arr(out, "mu"), _arr(m, "mu") * s, rtol=1e-5)


def test_stem_equalization_only_passes_positively_homogeneous_ops():
    assert _same(_stem("Sigmoid"), stem_equalize(_stem("Sigmoid"), ["Conv"]))
    # nodes the op filter does not name are not a stem
    assert _same(_stem(), stem_equalize(_stem(), ["Gemm"]))


def test_apply_cle_config_takes_options_from_params_and_extra_options():
    m = _conv_chain()
    default = apply_cle_config(m, {}, {}, [])
    from_params = apply_cle_config(m, {"cle_steps": -1}, {}, [])
    from_extra = apply_cle_config(m, {}, {"CLESteps": -1}, [])
    assert not _same(default, from_params)
    assert _same(from_params, from_extra)
    # extra_options win over the config's fields
    assert _same(apply_cle_config(m, {"cle_steps": -1}, {"CLESteps": 1}, []), default)
    # explicit exclusion of every Conv is a no-op, and ReplaceClip6Relu is read
    assert _same(m, apply_cle_config(m, {}, {}, [n.name for n in m.graph.node]))


def test_cle_runs_before_quantization_through_the_compat_layer():
    rng = np.random.default_rng(5)
    m = _model(
        "c0 = Conv<group=1, pads=[1,1,1,1]>(x, w0, b0) r0 = Relu(c0) y = Conv<group=1>(r0, w1, b1)",
        dict(
            w0=_w(rng, 8, 3, 3, 3, scale=2),
            b0=_w(rng, 8),
            w1=_w(rng, 4, 8, 1, 1, scale=0.2),
            b1=_w(rng, 4),
        ),
    )
    data = [
        {"x": rng.standard_normal((1, 3, 8, 8)).astype(np.float32)} for _ in range(3)
    ]

    def quantize(**opts):
        cfg = qc.QConfig.get_default_config("S8S8_AAWS")
        cfg.algo_config = [qc.CLEConfig()]
        cfg.extra_options.update(opts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qc.ModelQuantizer(cfg).quantize_model(
                m, calibration_data_reader=data
            )

    default, none = quantize(), quantize(CLESteps=0)
    assert default.SerializeToString() != none.SerializeToString()


# -- SmoothQuant -------------------------------------------------------------------


def _calibration(shape, hot, n=4):
    rng = np.random.default_rng(0)
    boost = 1 + 4 * (np.arange(shape[-1]) == hot)
    return [
        {"x": (rng.standard_normal(shape) * boost).astype(np.float32)} for _ in range(n)
    ]


def _transformerish():
    rng = np.random.default_rng(6)
    d = 8
    return _model(
        """ln = LayerNormalization<axis=-1>(x, gam, bet)
        q = MatMul(ln, wq)
        k = MatMul(ln, wk)
        a = Add(q, k)
        y = MatMul(a, wo)""",
        dict(
            gam=np.ones(d, np.float32),
            bet=np.zeros(d, np.float32),
            wq=_w(rng, d, d),
            wk=_w(rng, d, d),
            wo=_w(rng, d, d),
        ),
        inputs="float[2,5,8] x",
        outputs="float[2,5,8] y",
    )


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.9])
def test_smooth_quant_follows_the_formula_on_3d_activations(alpha):
    m = _transformerish()
    data = _calibration((2, 5, 8), hot=2)
    out = smooth_quant(m, data, alpha=alpha)
    assert [n.op_type for n in out.graph.node].count("Mul") == 3  # one per MatMul
    ln = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    # act_scale of each MatMul input, measured on the float model
    probe = onnx.ModelProto()
    probe.CopyFrom(m)
    for t in ("ln", "a"):
        probe.graph.output.append(onnx.ValueInfoProto(name=t))
    sess = ort.InferenceSession(probe.SerializeToString())
    seen: dict = {}
    for b in data:
        for name, v in zip(("y", "ln", "a"), sess.run(None, b)):
            if name != "y":
                seen[name] = np.maximum(
                    seen.get(name, 0), np.abs(v.reshape(-1, 8)).max(0)
                )
    for wname, act in (("wq", "ln"), ("wk", "ln"), ("wo", "a")):
        w = _arr(m, wname)
        s = np.power(seen[act], alpha) / np.power(np.abs(w).max(1) + 1e-9, 1 - alpha)
        np.testing.assert_allclose(_arr(out, wname), s[:, None] * w, rtol=1e-5)
    del ln
    x = data[0]["x"]
    np.testing.assert_allclose(_run(out, x), _run(m, x), rtol=1e-4, atol=1e-4)


def test_smooth_quant_gemm_and_unusable_matmuls_are_left_alone():
    rng = np.random.default_rng(7)
    gemm = _model(
        "y = Gemm(x, w, b)",
        dict(w=_w(rng, 8, 8), b=_w(rng, 8)),
        inputs="float[3,8] x",
        outputs="float[3,8] y",
    )
    data = _calibration((3, 8), 1)
    out = smooth_quant(gemm, data)
    assert [n.op_type for n in out.graph.node] == ["Gemm"] and _same(gemm, out)
    dynamic = _model(
        "y = MatMul(x, x2)",
        {},
        inputs="float[3,8] x, float[8,8] x2",
        outputs="float[3,8] y",
    )
    assert [n.op_type for n in smooth_quant(dynamic, data).graph.node] == ["MatMul"]
    assert _same(gemm, smooth_quant(gemm, []))


def test_smooth_quant_zero_activation_channel_zeroes_the_weight_row():
    rng = np.random.default_rng(8)
    m = _model(
        "y = MatMul(x, w)",
        dict(w=_w(rng, 4, 4)),
        inputs="float[3,4] x",
        outputs="float[3,4] y",
    )
    data = [
        {
            "x": np.where(np.arange(4) == 1, 0, 1.0).astype(np.float32)
            * np.ones((3, 4), np.float32)
        }
    ]
    out = smooth_quant(m, data)
    # no epsilon floor on the activation range, as in Quark: scale 0, factor 1/1e-9
    assert not _arr(out, "w")[1].any()
    assert _arr(out, "x_n0_MatMul_smooth_scale")[1] == pytest.approx(1e9)


def test_smooth_alpha_from_extra_options_wins():
    m = _transformerish()
    data = _calibration((2, 5, 8), 2)
    a = apply_smooth_quant_config(m, {"alpha": 0.9}, {}, data)
    b = apply_smooth_quant_config(m, {"alpha": 0.1}, {"SmoothAlpha": 0.9}, data)
    c = apply_smooth_quant_config(m, {}, {}, data)
    assert _same(a, b) and not _same(a, c)
    assert _same(c, smooth_quant(m, data, alpha=0.5))


# -- BiasCorrection ----------------------------------------------------------------


def _mlp():
    rng = np.random.default_rng(9)
    return _model(
        "h0 = Gemm(x, w1, b1) h1 = Relu(h0) y = Gemm(h1, w2, b2)",
        dict(w1=_w(rng, 16, 32), b1=_w(rng, 32), w2=_w(rng, 32, 8), b2=_w(rng, 8)),
        inputs="float[3,16] x",
        outputs="float[3,8] y",
    )


def _quantized(m, data, preset="S8S8_AAWS", algos=()):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config = list(algos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(m, calibration_data_reader=data)


def _int32_biases(q):
    return {
        i.name: numpy_helper.to_array(i)
        for i in q.graph.initializer
        if i.data_type == onnx.TensorProto.INT32 and i.name.endswith("/int32")
    }


def test_bias_correction_rewrites_quantized_biases_and_shrinks_the_mean_error():
    rng = np.random.default_rng(10)
    m = _mlp()
    data = [{"x": rng.standard_normal((3, 16)).astype(np.float32)} for _ in range(8)]
    base = _quantized(m, data)
    fixed = correct_bias_quark(m, base, data)
    before, after = _int32_biases(base), _int32_biases(fixed)
    assert set(before) == set(after) and len(before) == 2
    assert any(not np.array_equal(before[k], after[k]) for k in before)

    def mean_error(q):
        err = np.stack([_run(q, b["x"]) - _run(m, b["x"]) for b in data])
        return float(np.abs(err.mean(axis=(0, 1))).mean())

    assert mean_error(fixed) <= mean_error(base) * 1.05 + 1e-6
    # only the quantized biases change
    other = lambda q: {  # noqa: E731
        i.name: numpy_helper.to_array(i).tobytes()
        for i in q.graph.initializer
        if i.name not in before
    }
    assert other(base) == other(fixed)
    # the compat layer runs the same thing
    via = _quantized(m, data, algos=[qc.BiasCorrectionConfig()])
    assert all(np.array_equal(_int32_biases(via)[k], after[k]) for k in after)


def test_bias_correction_leaves_layers_without_a_bias_alone():
    rng = np.random.default_rng(11)
    m = _model(
        "h = MatMul(x, w1) r = Relu(h) y = MatMul(r, w2)",
        dict(w1=_w(rng, 16, 16), w2=_w(rng, 16, 8)),
        inputs="float[3,16] x",
        outputs="float[3,8] y",
    )
    data = [{"x": rng.standard_normal((3, 16)).astype(np.float32)} for _ in range(4)]
    base = _quantized(m, data)
    assert (
        correct_bias_quark(m, base, data).SerializeToString()
        == base.SerializeToString()
    )
    assert (
        correct_bias_quark(
            _mlp(),
            _quantized(_mlp(), data[:0] or [{"x": np.zeros((3, 16), np.float32)}]),
            [],
        ).SerializeToString()
        == _quantized(
            _mlp(), [{"x": np.zeros((3, 16), np.float32)}]
        ).SerializeToString()
    )


# -- AutoMixprecision --------------------------------------------------------------


def _amp_model():
    rng = np.random.default_rng(0)
    d = 16
    return _model(
        """h1 = Gemm(x, w1, b1)
        t1 = Tanh(h1)
        h2 = Gemm(t1, w2, b2)
        t2 = Tanh(h2)
        h3 = Gemm(t2, w3, b3)
        skip = Gemm(x, w4, b4)
        y = Add(h3, skip)""",
        dict(
            w1=_w(rng, d, d, scale=1.5),
            b1=_w(rng, d),
            w2=_w(rng, d, d, scale=0.3),
            b2=_w(rng, d),
            w3=_w(rng, d, d),
            b3=_w(rng, d),
            w4=_w(rng, d, d),
            b4=_w(rng, d),
        ),
        inputs="float[3,16] x",
        outputs="float[3,16] y",
    )


def _amp_data(n=6):
    return [
        {"x": np.random.default_rng(i).standard_normal((3, 16)).astype(np.float32)}
        for i in range(n)
    ]


def _zero_point_dtypes(model):
    inits = {t.name: t for t in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits:
            t = n.input[0].removesuffix("/f")
            dt = onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type)
            if dt != "INT32":
                out.setdefault(t, set()).add(dt)
    return out


def _amp(**kw):
    kw.setdefault("base_dtype", "uint8")
    return amp.auto_mixprecision(_amp_model(), _amp_data(), **kw)


def test_amp_several_targets_pick_the_best_scoring_one_per_candidate():
    res = _amp(targets=[("uint8", None), ("uint16", None)])
    for c in res.ranked:
        assert len(c.all_config_scores) == 2
        assert c.score == min(c.all_config_scores)
        assert c.best_config_index == int(np.argmin(c.all_config_scores))
    # (which config wins a layer is a near tie for some layers and varies with
    # the platform's floating-point rounding, so only the structure is asserted)
    assert {c.best_config_index for c in res.ranked} <= {0, 1}
    swapped = _amp(targets=[("uint16", None), ("uint8", None)])
    assert {c.name: c.best_config_index for c in swapped.ranked} == {
        c.name: 1 - c.best_config_index for c in res.ranked
    }
    assert swapped.model.SerializeToString() == res.model.SerializeToString()


def test_amp_pinned_targets_apply_to_the_named_candidates_only():
    res = _amp(
        targets=[("uint16", None)],
        candidate_targets={"n2_Gemm": ("int16", True)},
    )
    dts = _zero_point_dtypes(res.model)
    assert dts["t1"] == dts["h2"] == {"INT16"}  # n2's input and output
    assert dts["h1"] == {"UINT16"}
    with pytest.raises(ValueError, match="dtypes"):
        _amp(targets=[("uint16", None)], candidate_targets={"n0_Gemm": ("int4", None)})
    with pytest.raises(ValueError, match="target_dtype or targets"):
        _amp()


def test_amp_boundaries_are_dual_only_when_asked():
    kw = dict(targets=[("uint16", None)], include_layers=["n2_Gemm"])
    single = {n.name for n in _amp(**kw).model.graph.node if "_additional_" in n.name}
    dual = {
        n.name
        for n in _amp(dual_quant_nodes=True, **kw).model.graph.node
        if "_additional_" in n.name
    }
    assert not single and dual


def test_amp_no_input_qdq_shared_keeps_shared_inputs_at_base_precision():
    plain = _amp(targets=[("uint16", None)])
    keep = _amp(targets=[("uint16", None)], no_input_qdq_shared=True)
    assert set(plain.moved) == {"n0_Gemm", "n2_Gemm", "n4_Gemm", "n5_Gemm"}
    # x feeds n0 and n5, so those two stay out of the mixing step
    assert set(keep.moved) == {"n2_Gemm", "n4_Gemm"}


def test_amp_worker_threads_do_not_change_the_result():
    one = _amp(targets=[("uint16", None)], worker_num=1)
    many = _amp(targets=[("uint16", None)], worker_num=4)
    assert [(c.name, c.score) for c in one.ranked] == [
        (c.name, c.score) for c in many.ranked
    ]
    assert one.model.SerializeToString() == many.model.SerializeToString()


def _write_subgraphs(path, **overrides):
    doc = {
        "quantized": False,
        "num_subgraphs": 2,
        "subgraphs": [
            {"name": "front", "start_nodes": ["n0_Gemm"], "end_nodes": ["n1_Tanh"]},
            {"name": "back", "start_nodes": ["n2_Gemm"], "end_nodes": ["n3_Tanh"]},
        ],
    }
    doc.update(overrides)
    path.write_text(json.dumps(doc))
    return str(path)


def test_amp_subgraphs_are_scored_and_moved_as_groups(tmp_path):
    m = _amp_model()
    specs = amp.parse_subgraph_json(_write_subgraphs(tmp_path / "s.json"), m, m)
    assert [s.name for s in specs] == ["front", "back", "__ungrouped__"]
    assert specs[0].resolved_nodes == ["n0_Gemm", "n1_Tanh"]
    assert "n4_Gemm" in specs[2].resolved_nodes
    res = _amp(
        targets=[("uint16", None)],
        subgraphs=[(s.name, s.resolved_nodes) for s in specs],
    )
    assert {c.name for c in res.ranked} == {"front", "back", "__ungrouped__"}
    front = next(c for c in res.ranked if c.name == "front")
    assert front.nodes == ["n0_Gemm"]  # only the target ops of the group
    for bad, match in (
        ({"num_subgraphs": 3}, "num_subgraphs"),
        (
            {
                "subgraphs": [{"name": "a", "start_nodes": ["nope"], "end_nodes": []}],
                "num_subgraphs": 1,
            },
            "not found",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            amp.parse_subgraph_json(_write_subgraphs(tmp_path / "b.json", **bad), m, m)


def test_amp_subgraph_overlaps_go_to_the_first_group(tmp_path):
    m = _amp_model()
    path = _write_subgraphs(
        tmp_path / "o.json",
        num_subgraphs=2,
        subgraphs=[
            {"name": "a", "start_nodes": ["n0_Gemm"], "end_nodes": ["n2_Gemm"]},
            {"name": "b", "start_nodes": ["n1_Tanh"], "end_nodes": ["n3_Tanh"]},
        ],
    )
    a, b, *_ = amp.parse_subgraph_json(path, m, m)
    assert "n1_Tanh" in a.resolved_nodes and "n1_Tanh" not in b.resolved_nodes


def test_amp_sensitivity_cache_is_reused_pinned_and_invalidated(tmp_path):
    cache = tmp_path / "c.json"
    first = _amp(targets=[("uint16", None)], cache_file=cache)
    doc = json.loads(cache.read_text())
    assert set(doc) == {"version", "cache_key", "results"}
    assert [r["name"] for r in doc["results"]] == [c.name for c in first.ranked]

    # a cached ranking is used as is: reverse it and disable one layer
    for i, r in enumerate(reversed(doc["results"])):
        r["score"] = float(i)
    doc["results"].reverse()
    doc["results"][0]["enabled"] = False
    pinned = doc["results"][0]["name"]
    cache.write_text(json.dumps(doc))
    reused = _amp(targets=[("uint16", None)], cache_file=cache)
    assert [c.name for c in reused.ranked] == [r["name"] for r in doc["results"]]
    assert pinned not in reused.moved and len(reused.moved) == len(doc["results"]) - 1

    # a changed configuration makes the file stale: recomputed, with a warning
    with pytest.warns(UserWarning, match="stale"):
        again = _amp(targets=[("int16", None)], cache_file=cache)
    assert all(c.score == c.score for c in again.ranked)
    assert json.loads(cache.read_text())["cache_key"] != doc["cache_key"]


def test_amp_through_the_compat_layer_accepts_the_multi_config_forms(tmp_path):
    u16 = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec())
    s16 = qc.QLayerConfig(activation=qc.Int16Spec(), weight=qc.Int8Spec())
    u8 = qc.QLayerConfig(activation=qc.UInt8Spec(), weight=qc.Int8Spec())
    sg = _write_subgraphs(tmp_path / "s.json")
    forms = {
        "list": dict(target_layer_config=[u8, u16]),
        "dict": dict(target_layer_config={s16: ["n2_Gemm"], u16: []}),
        "dict-no-fallback": dict(
            target_layer_config={s16: ["n2_Gemm"], u16: ["n0_Gemm"]}
        ),
        "subgraph": dict(target_layer_config=u16, subgraph_json=sg),
        "cache": dict(
            target_layer_config=u16, sensitivity_cache_file=str(tmp_path / "c.json")
        ),
    }
    results = {}
    for name, params in forms.items():
        cfg = qc.QConfig(
            global_config=qc.QLayerConfig(
                activation=qc.UInt8Spec(), weight=qc.Int8Spec()
            ),
            algo_config=[qc.AutoMixprecisionConfig(**params)],
        )
        q = qc.ModelQuantizer(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[name] = _zero_point_dtypes(
                q.quantize_model(_amp_model(), calibration_data_reader=_amp_data())
            )
        assert q.last_auto_mixprecision.ranked, name
    assert results["dict"]["t1"] == results["dict"]["h2"] == {"INT16"}
    assert results["dict"]["h1"] == {"UINT16"}
    # no entry maps to []: the first entry is the fallback for everything else
    assert results["dict-no-fallback"]["h3"] == {"INT16"}
    assert results["dict-no-fallback"]["h1"] == {"UINT16"}
    assert (tmp_path / "c.json").exists()
