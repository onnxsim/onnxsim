"""Quark-free tests of the AutoMixprecision pieces that mirror Quark 0.13's
``MixingStrategy``: weight / bias precision mixing, the scoring conventions
(graph optimizations off, ``data_size`` counting one batch more), the block
formats' candidate scoring on the ONNX reference path, and the options both
paths share (``subgraph_json``, ``sensitivity_cache_file``, ...). Parity with
the real Quark is in ``tests/test_quark_amp_parity.py``."""

import json
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quark_auto_mixprecision as amp
from onnxsim import quark_block_formats as qbf
from onnxsim import quark_compat as qc
from onnxsim.quark_fakequant_eval import run_fake_quantized
from onnxsim.quark_preset_graphs import dedicate_dq_nodes

D = 16


def _w(rng, *shape, scale=0.5):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _model(body, inits, inputs=f"float[3,{D}] x", outputs=f"float[3,{D}] y"):
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g ({inputs}) => ({outputs}) {{ {body} }}'
    )
    m.graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in inits.items())
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return m


def _mlp():
    rng = np.random.default_rng(0)
    return _model(
        """h1 = Gemm(x, w1, b1)  t1 = Tanh(h1)  h2 = Gemm(t1, w2, b2)
           t2 = Tanh(h2)  y = Gemm(t2, w3, b3)""",
        dict(
            w1=_w(rng, D, D, scale=1.5),
            b1=_w(rng, D),
            w2=_w(rng, D, D, scale=0.3),
            b2=_w(rng, D),
            w3=_w(rng, D, D),
            b3=_w(rng, D),
        ),
    )


def _data(n=6):
    return [
        {"x": np.random.default_rng(i).standard_normal((3, D)).astype(np.float32)}
        for i in range(n)
    ]


def _dq_const(model, node_name, slot):
    """``(codes, scale, zero_point, DequantizeLinear)`` feeding input ``slot`` of
    ``node_name`` when it is a quantized constant."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    node = next(n for n in model.graph.node if n.name == node_name)
    dq = prod[node.input[slot]]
    assert dq.op_type == "DequantizeLinear"
    return inits[dq.input[0]], inits[dq.input[1]], inits[dq.input[2]], dq


def _dq_scale(model, tensor):
    """Scale of the DequantizeLinear that produces ``tensor``."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    dq = next(n for n in model.graph.node if tensor in n.output)
    return inits[dq.input[1]]


def _baseline(**kw):
    return amp.auto_mixprecision(
        _mlp(), _data(), base_dtype="uint8", metric_threshold=None, **kw
    ).model


# -- weights and biases move with the layer -------------------------------------------


def test_weight_moves_to_int16_from_its_quantized_values():
    spec = amp.TargetSpec(
        inputs=("uint16", None),
        outputs=("uint16", None),
        weight=("int16", True),
    )
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="uint8",
        targets=[spec],
        include_layers=["n2_Gemm"],
    )
    base = _baseline(targets=[spec])
    codes8, scale8, zp8, _ = _dq_const(base, "n2_Gemm", 1)
    codes16, scale16, zp16, dq16 = _dq_const(res.model, "n2_Gemm", 1)
    assert codes8.dtype == np.int8 and codes16.dtype == np.int16
    # the int8 resolution survives: dequantized int8 values, one per-tensor
    # scale from their range (symmetric: max |w| / 32767), rounded
    deq = (codes8.astype(np.float32) - zp8.astype(np.float32)) * scale8
    want_scale = np.float32(np.abs(deq).max() / 32767)
    assert scale16.shape == () and np.isclose(scale16, want_scale, rtol=1e-6)
    np.testing.assert_array_equal(codes16, np.round(deq / scale16).astype(np.int16))
    assert dq16.domain == "com.microsoft"  # int16 below opset 21
    assert any(o.domain == "com.microsoft" for o in res.model.opset_import)
    # the other layers keep their int8 weights
    for node in ("n0_Gemm", "n4_Gemm"):
        assert _dq_const(res.model, node, 1)[0].dtype == np.int8


def test_int32_bias_scale_follows_the_moved_input_and_weight_scales():
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="uint8",
        targets=[("uint16", None)],
        include_layers=["n2_Gemm"],
    )
    base = _baseline(targets=[("uint16", None)])
    # n2: its input t1 is now uint16 (Quark refreshes the bias scale to
    # input_scale * weight_scale) and the codes are rescaled from the baseline
    # codes -- truncated, not rounded, in float32
    b_codes, b_scale, _, _ = _dq_const(base, "n2_Gemm", 2)
    codes, scale, _, _ = _dq_const(res.model, "n2_Gemm", 2)
    in_scale = _dq_scale(res.model, "t1")
    w_scale = _dq_const(res.model, "n2_Gemm", 1)[1]
    want_scale = (in_scale * w_scale).astype(np.float32)
    np.testing.assert_array_equal(scale, want_scale)
    want = (
        b_codes.astype(np.float32) * b_scale.astype(np.float32) / want_scale
    ).astype(np.int32)
    np.testing.assert_array_equal(codes, want)
    # layers that did not move keep their baseline bias (Quark edits the
    # quantized baseline in place and refreshes only the layers it moves)
    for node in ("n0_Gemm", "n4_Gemm"):
        np.testing.assert_array_equal(
            _dq_const(res.model, node, 2)[0], _dq_const(base, node, 2)[0]
        )
        np.testing.assert_array_equal(
            _dq_const(res.model, node, 2)[1].ravel()[:1],
            _dq_const(base, node, 2)[1].ravel()[:1],
        )


def test_bias_spec_requantizes_the_bias_and_skips_the_scale_refresh():
    spec = amp.TargetSpec(
        inputs=("int8", False), weight=("int8", True), bias=("int8", True)
    )
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="int16",
        targets=[spec],
        include_layers=["n2_Gemm"],
        quantize_kwargs=dict(weight_dtype="int16"),
    )
    codes, scale, zp, dq = _dq_const(res.model, "n2_Gemm", 2)
    assert codes.dtype == np.int8 and zp.dtype == np.int8
    assert dq.domain == ""
    # input tensors move, outputs (no ``outputs`` precision) stay int16
    inits = {t.name: t for t in res.model.graph.initializer}
    q_zp = {
        n.input[0].removesuffix("/f"): inits[n.input[2]].data_type
        for n in res.model.graph.node
        if n.op_type == "QuantizeLinear" and len(n.input) > 2
    }
    assert q_zp["t1"] == onnx.TensorProto.INT8
    assert q_zp["h2"] == onnx.TensorProto.INT16


def test_a_target_equal_to_the_base_precision_still_requantizes_weights():
    # Quark does not refuse "nothing to mix": the weights become per-tensor and
    # the bias scales are refreshed
    spec = amp.TargetSpec(
        inputs=("uint8", None), outputs=("uint8", None), weight=("int8", True)
    )
    res = amp.auto_mixprecision(
        _mlp(),
        _data(),
        base_dtype="uint8",
        targets=[spec],
        quantize_kwargs=dict(per_channel=True),
    )
    base = _baseline(targets=[spec], quantize_kwargs=dict(per_channel=True))
    assert _dq_const(base, "n0_Gemm", 1)[1].shape != ()  # per channel
    assert _dq_const(res.model, "n0_Gemm", 1)[1].shape == ()  # per tensor
    assert res.moved


def test_unsupported_precisions_are_refused():
    with pytest.raises(ValueError, match="dtypes"):
        amp.auto_mixprecision(
            _mlp(),
            _data(),
            base_dtype="uint8",
            targets=[amp.TargetSpec(weight=("int4", None))],
        )


# -- the compat layer reads a QLayerConfig like Quark's MixingStrategy --------------


def _amp_config(base_act, base_wt, target, **params):
    return qc.QConfig(
        global_config=qc.QLayerConfig(activation=base_act(), weight=base_wt()),
        algo_config=[qc.AutoMixprecisionConfig(target_layer_config=target, **params)],
    )


def _zp_types(model):
    inits = {t.name: t for t in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits:
            dt = onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type)
            if dt != "INT32":
                out[n.input[0].removesuffix("/f")] = dt
    return out


def _quantize(cfg, **kw):
    q = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return q, q.quantize_model(_mlp(), calibration_data_reader=_data(), **kw)


def test_activation_target_moves_inputs_and_outputs_but_input_tensors_only_inputs():
    both = qc.QLayerConfig(activation=qc.Int16Spec(), weight=qc.Int8Spec())
    ins = qc.QLayerConfig(input_tensors=qc.Int16Spec(), weight=qc.Int8Spec())
    kw = dict(include_layers=["n2_Gemm"])
    _, m_both = _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, both, **kw))
    _, m_ins = _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, ins, **kw))
    assert _zp_types(m_both)["t1"] == _zp_types(m_both)["h2"] == "INT16"
    assert _zp_types(m_ins)["t1"] == "INT16" and _zp_types(m_ins)["h2"] == "UINT8"


def test_target_weight_and_bias_specs_reach_the_model():
    target = qc.QLayerConfig(
        activation=qc.UInt16Spec(),
        weight=qc.Int16Spec(),
        bias=qc.Int16Spec(),
    )
    cfg = _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    _, out = _quantize(cfg)
    assert _dq_const(out, "n2_Gemm", 1)[0].dtype == np.int16
    assert _dq_const(out, "n2_Gemm", 2)[0].dtype == np.int16
    assert _dq_const(out, "n0_Gemm", 1)[0].dtype == np.int8
    assert _dq_const(out, "n0_Gemm", 2)[0].dtype == np.int32
    onnx.checker.check_model(out)
    ort.InferenceSession(out.SerializeToString(), providers=["CPUExecutionProvider"])


def test_target_symmetry_follows_the_global_specs_like_quark():
    # the global activation spec is asymmetric, so the int16 target is too even
    # though Int16Spec() alone is symmetric (Quark's ActivationSymmetric wins)
    target = qc.QLayerConfig(activation=qc.Int16Spec(), weight=qc.Int8Spec())
    cfg = _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    _, out = _quantize(cfg)
    inits = {t.name: numpy_helper.to_array(t) for t in out.graph.initializer}
    q = next(
        n
        for n in out.graph.node
        if n.op_type == "QuantizeLinear" and n.input[0].startswith("h2")
    )
    assert inits[q.input[2]] != 0  # an asymmetric int16 zero point


def _pof2(x):
    return float(np.log2(x)) == int(np.log2(x))


def test_power_of_two_target_weights_round_the_scale_to_a_power_of_two():
    target = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.XInt8Spec())
    _, out = _quantize(
        _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    )
    codes, scale, zp, _ = _dq_const(out, "n2_Gemm", 1)
    assert codes.dtype == np.int8 and scale.shape == () and _pof2(float(scale))
    # Quark rounds in the log domain: to the *nearest* power of two, which can be
    # below the scale that just covers the range
    _, plain = _quantize(
        _amp_config(
            qc.UInt8Spec,
            qc.Int8Spec,
            qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec()),
            include_layers=["n2_Gemm"],
        )
    )
    raw = float(_dq_const(plain, "n2_Gemm", 1)[1])
    assert float(scale) == 2.0 ** -int(np.rint(-np.log2(raw)))
    # the other layers keep their (non power-of-two) int8 weights
    assert not _pof2(float(_dq_const(out, "n0_Gemm", 1)[1]))


def test_power_of_two_target_activations_are_rounded_too():
    target = qc.QLayerConfig(activation=qc.XInt8Spec(), weight=qc.Int8Spec())
    _, out = _quantize(
        _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    )
    assert _pof2(float(_dq_scale(out, "t1"))) and _pof2(float(_dq_scale(out, "h2")))
    assert not _pof2(float(_dq_scale(out, "h1")))


def _chains(model):
    """``{node: ([quantizer op types behind each input], [after each output])}``."""
    prod = {o: n for n in model.graph.node for o in n.output}
    cons = {}
    for n in model.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    quant = {
        "QuantizeLinear",
        "DequantizeLinear",
        "ExtendedQuantizeLinear",
        "ExtendedDequantizeLinear",
        "BFPQuantizeDequantize",
        "MXQuantizeDequantize",
    }
    out = {}
    for n in model.graph.node:
        if n.op_type in quant:
            continue
        ins = []
        for x in n.input:
            chain, cur = [], x
            while cur in prod and prod[cur].op_type in quant:
                chain.append(prod[cur].op_type)
                cur = prod[cur].input[0]
            ins.append(chain)
        outs = []
        for o in n.output:
            chain, cur = [], o
            while True:
                nxt = [c for c in cons.get(cur, []) if c.op_type in quant]
                if not nxt:
                    break
                chain.append(nxt[0].op_type)
                cur = nxt[0].output[0]
            outs.append(chain)
        out[n.name] = (ins, outs)
    return out


def test_block_and_half_targets_over_an_integer_base_edit_the_quantizers_in_place():
    for dtype, op in (
        (qc.BFP16Spec, "BFPQuantizeDequantize"),
        (qc.MXInt8Spec, "MXQuantizeDequantize"),
    ):
        target = qc.QLayerConfig(activation=dtype(), weight=dtype())
        _, out = _quantize(
            _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
        )
        chains = _chains(out)
        # the moved layer: every activation and its weight go through one block node
        assert chains["n2_Gemm"][0][:2] == [[op], [op]]
        assert chains["n2_Gemm"][1] == [[op]]
        # the bias keeps its (int32) quantizer; the neighbours are untouched
        assert chains["n2_Gemm"][0][2] == ["DequantizeLinear"]
        assert chains["n0_Gemm"][0][0] == ["DequantizeLinear", "QuantizeLinear"]
        assert chains["n0_Gemm"][0][1] == ["DequantizeLinear"]
        onnx.checker.check_model(out)
        # the weight is dequantized from its int8 codes first, then block-quantized
        w = next(t for t in out.graph.initializer if t.name.endswith("_float_Mixed"))
        assert numpy_helper.to_array(w).dtype == np.float32
    for dtype in (qc.BFloat16Spec, qc.Float16Spec):
        target = qc.QLayerConfig(activation=dtype(), weight=dtype())
        _, out = _quantize(
            _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
        )
        ext = ["ExtendedDequantizeLinear", "ExtendedQuantizeLinear"]
        chains = _chains(out)
        assert chains["n2_Gemm"][0][0] == ext and chains["n2_Gemm"][1] == [ext[::-1]]
        assert chains["n2_Gemm"][0][1] == ["ExtendedDequantizeLinear"]  # folded weight
        assert chains["n0_Gemm"][0][0] == ["DequantizeLinear", "QuantizeLinear"]
        inits = {t.name: t for t in out.graph.initializer}
        dq = next(
            n
            for n in out.graph.node
            if n.op_type == "ExtendedDequantizeLinear"
            and inits.get(n.input[0]) is not None
        )
        half = {
            qc.BFloat16Spec: onnx.TensorProto.BFLOAT16,
            qc.Float16Spec: onnx.TensorProto.FLOAT16,
        }
        assert inits[dq.input[0]].data_type == half[dtype]  # the weight codes
        assert float(numpy_helper.to_array(inits[dq.input[1]])) == 1.0  # scale 1
        assert any(o.domain == "com.amd.quark" for o in out.opset_import)


def test_half_and_block_baselines_mix_into_integer_layers_without_calibration_data():
    # no calibration is run for these formats: Quark's fake range [0, 1] sizes
    # the integer activations, the (float) weights their own range
    for base in (qc.BFloat16Spec, qc.BFP16Spec, qc.MXInt8Spec):
        cfg = qc.QConfig(
            global_config=qc.QLayerConfig(activation=base(), weight=base()),
            algo_config=[
                qc.AutoMixprecisionConfig(
                    target_layer_config=qc.QLayerConfig(
                        activation=qc.UInt8Spec(), weight=qc.Int8Spec()
                    ),
                    include_layers=["n2_Gemm"],
                )
            ],
        )
        _, out = _quantize(cfg)
        chains = _chains(out)
        assert chains["n2_Gemm"][0][0] == ["DequantizeLinear", "QuantizeLinear"]
        # (the baseline's weight keeps its Q on the float constant)
        assert chains["n2_Gemm"][0][1] == ["DequantizeLinear", "QuantizeLinear"]
        assert chains["n0_Gemm"][0][0][0] != "DequantizeLinear"  # still the base
        inits = {t.name: numpy_helper.to_array(t) for t in out.graph.initializer}
        q = next(
            n
            for n in out.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0].startswith("t1")
        )
        # range [0, 1], symmetric like the global activation spec (Quark's
        # ActivationSymmetric): a uint8 grid centred on 128 spanning +-1
        assert float(inits[q.input[1]]) == np.float32(2.0 / 255)
        assert int(inits[q.input[2]]) == 128
        out_vals = run_fake_quantized(out, _data())
        assert all(np.isfinite(o[0]).all() for o in out_vals)


def _extra_pairs(model):
    return sorted(n.name for n in model.graph.node if "_additional_" in n.name)


@pytest.mark.parametrize(
    "target_spec",
    [qc.BFP16Spec, qc.BFloat16Spec, qc.Float16Spec, qc.MXInt8Spec, qc.XInt8Spec],
    ids=["bfp16", "bfloat16", "float16", "mxint8", "pof2"],
)
def test_dual_quant_nodes_cover_half_block_and_pof2_mixes(target_spec):
    """Quark inserts the boundary quantizers into the *final* mixed model for
    every kind of mix (candidates are scored without them)."""
    target = qc.QLayerConfig(activation=target_spec(), weight=target_spec())
    kw = dict(include_layers=["n2_Gemm"])
    _, single = _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, target, **kw))
    q, dual = _quantize(
        _amp_config(qc.UInt8Spec, qc.Int8Spec, target, dual_quant_nodes=True, **kw)
    )
    assert not _extra_pairs(single)
    # t1 (into n2) and h2 (out of n2) cross the boundary
    assert len(_extra_pairs(dual)) >= 2
    assert q.last_auto_mixprecision.moved == ["n2_Gemm"]
    # the model is otherwise the one without boundary nodes
    plain = [n.op_type for n in single.graph.node]
    kept = [n.op_type for n in dual.graph.node if "_additional_" not in n.name]
    assert sorted(kept) == sorted(plain)


def test_dual_quant_nodes_over_a_float_baseline():
    target = qc.QLayerConfig(activation=qc.Int8Spec(), weight=qc.Int8Spec())
    _, dual = _quantize(
        _amp_config(
            qc.BFloat16Spec,
            qc.BFloat16Spec,
            target,
            include_layers=["n2_Gemm"],
            dual_quant_nodes=True,
        )
    )
    assert _extra_pairs(dual)


def test_dual_boundary_pair_is_requantized_to_the_neighbour_precision():
    """A tensor that goes from a uint8 producer to a uint16 consumer is first
    quantized to uint8 again (its own calibrated range), then to uint16."""
    target = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec())
    _, dual = _quantize(
        _amp_config(
            qc.UInt8Spec,
            qc.Int8Spec,
            target,
            include_layers=["n2_Gemm"],
            dual_quant_nodes=True,
        )
    )
    inits = {t.name: numpy_helper.to_array(t) for t in dual.graph.initializer}
    extra = [
        n
        for n in dual.graph.node
        if "_additional_" in n.name and n.op_type == "QuantizeLinear"
    ]
    assert len(extra) == 2
    for q in extra:
        assert inits[q.input[2]].dtype == np.uint8
        assert inits[q.input[1]].dtype == np.float32 and inits[q.input[1]].shape == ()


# -- scoring conventions ----------------------------------------------------------------


def test_candidates_are_scored_with_graph_optimizations_off(monkeypatch):
    levels = []
    real = ort.InferenceSession

    def spy(model, sess_options=None, *a, **k):
        if sess_options is not None:
            levels.append(sess_options.graph_optimization_level)
        return real(model, sess_options, *a, **k)

    monkeypatch.setattr(ort, "InferenceSession", spy)
    amp.auto_mixprecision(
        _mlp(), _data(), base_dtype="uint8", targets=[("uint16", None)]
    )
    # float model + baseline + three candidates + the mixed trials (the
    # calibration pass runs its own sessions)
    assert levels.count(ort.GraphOptimizationLevel.ORT_DISABLE_ALL) >= 8


@pytest.mark.parametrize("data_size, scored", [(0, 1), (1, 2), (3, 4), (5, 6), (50, 6)])
def test_data_size_counts_one_batch_more_like_quark(data_size, scored):
    # Quark's inference_model stops once len(results) > data_size: the default 0
    # scores only the first batch (the documented "0 = all" does not hold)
    seen = []

    def metric(float_out, quant_out):
        seen.append(len(float_out))
        return float(len(float_out))

    cfg = _amp_config(
        qc.UInt8Spec,
        qc.Int8Spec,
        qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec()),
        metric_distance_fn=metric,
        data_size=data_size,
    )
    _quantize(cfg)
    assert set(seen) == {scored}


def test_missing_subgraph_json_is_ignored_with_a_note():
    cfg = _amp_config(
        qc.UInt8Spec,
        qc.Int8Spec,
        qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec()),
        subgraph_json="/no/such/file.json",
    )
    q, out = _quantize(cfg)
    assert any("does not exist" in a for a in q.last_approximations)
    assert q.last_auto_mixprecision.moved


# -- the int16 -> int8 promotion goes through the same machinery ----------------------


def _s16_config(**params):
    target = qc.QLayerConfig(
        input_tensors=qc.Int8Spec(symmetric=False),
        weight=qc.Int8Spec(),
        bias=qc.Int8Spec(),
    )
    return qc.QConfig(
        global_config=qc.QLayerConfig(
            activation=qc.Int16Spec(symmetric=False), weight=qc.Int16Spec()
        ),
        algo_config=[qc.AutoMixprecisionConfig(target_layer_config=target, **params)],
        Int32Bias=False,
    )


def _write_subgraphs(path):
    path.write_text(
        json.dumps(
            {
                "quantized": False,
                "num_subgraphs": 2,
                "subgraphs": [
                    {
                        "name": "front",
                        "start_nodes": ["n0_Gemm"],
                        "end_nodes": ["n1_Tanh"],
                    },
                    {
                        "name": "back",
                        "start_nodes": ["n2_Gemm"],
                        "end_nodes": ["n3_Tanh"],
                    },
                ],
            }
        )
    )
    return str(path)


def test_int16_to_int8_accepts_subgraph_json_and_cache_pins(tmp_path):
    sg, cache = _write_subgraphs(tmp_path / "sg.json"), tmp_path / "cache.json"
    q, full = _quantize(
        _s16_config(subgraph_json=sg, sensitivity_cache_file=str(cache))
    )
    doc = json.loads(cache.read_text())
    assert {r["name"] for r in doc["results"]} == {"front", "back", "__ungrouped__"}
    assert len(q.last_auto_mixprecision.moved) == 3
    for n in ("n0_Gemm", "n2_Gemm", "n4_Gemm"):
        assert _dq_const(full, n, 1)[0].dtype == np.int8
        assert _dq_const(full, n, 2)[0].dtype == np.int8
    # pin "back": its layer keeps int16 weights and an int16 bias, and its
    # input / output stay int16
    for r in doc["results"]:
        r["enabled"] = r["name"] != "back"
    cache.write_text(json.dumps(doc))
    q, pinned = _quantize(
        _s16_config(subgraph_json=sg, sensitivity_cache_file=str(cache))
    )
    assert (
        q.last_auto_mixprecision.moved and "back" not in q.last_auto_mixprecision.moved
    )
    assert _dq_const(pinned, "n2_Gemm", 1)[0].dtype == np.int16
    assert _dq_const(pinned, "n2_Gemm", 2)[0].dtype == np.int16
    assert _dq_const(pinned, "n0_Gemm", 1)[0].dtype == np.int8
    assert _zp_types(pinned)["t1"] == "INT16" and _zp_types(pinned)["x"] == "INT8"


def test_int16_unpromoted_layers_keep_int16_weights_and_biases():
    q, out = _quantize(_s16_config(include_layers=["n2_Gemm"]))
    assert _dq_const(out, "n2_Gemm", 1)[0].dtype == np.int8
    for n in ("n0_Gemm", "n4_Gemm"):
        assert _dq_const(out, n, 1)[0].dtype == np.int16
        codes, scale, _, _ = _dq_const(out, n, 2)
        assert codes.dtype == np.int16  # Int32Bias=False: quantized like a weight


# -- block formats: candidates scored on the reference path ---------------------------


def _bf16_mixed(**params):
    cfg = qc.QConfig.get_default_config("BF16_MIXED_BFP16")
    cfg.algo_config[0].params.update(params)
    return cfg


def _block_run(**params):
    q = qc.ModelQuantizer(_bf16_mixed(**params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = q.quantize_model(_mlp(), calibration_data_reader=_data())
    return q, out


def _block_weights(model):
    """Gemm names whose weight reads through a BFP node."""
    prod = {o: n for n in model.graph.node for o in n.output}
    return {
        n.name
        for n in model.graph.node
        if n.op_type == "Gemm" and prod[n.input[1]].op_type == "BFPQuantizeDequantize"
    }


def test_block_cache_is_written_ranked_and_pins(tmp_path):
    cache = tmp_path / "c.json"
    q, full = _block_run(sensitivity_cache_file=str(cache))
    doc = json.loads(cache.read_text())
    scores = [r["score"] for r in doc["results"]]
    assert scores == sorted(scores) and len(scores) == 3
    assert {r["name"] for r in doc["results"]} == {"n0_Gemm", "n2_Gemm", "n4_Gemm"}
    assert _block_weights(full) == {"n0_Gemm", "n2_Gemm", "n4_Gemm"}
    assert q.last_auto_mixprecision.ranked
    # reuse the file with one candidate pinned: it stays bfloat16
    for r in doc["results"]:
        r["enabled"] = r["name"] != "n2_Gemm"
    cache.write_text(json.dumps(doc))
    _, pinned = _block_run(sensitivity_cache_file=str(cache))
    assert _block_weights(pinned) == {"n0_Gemm", "n4_Gemm"}
    _, excluded = _block_run(exclude_layers=["n2_Gemm"])
    assert pinned.SerializeToString() == excluded.SerializeToString()


def test_block_subgraph_json_groups_the_candidates(tmp_path):
    cache = tmp_path / "c.json"
    sg = _write_subgraphs(tmp_path / "sg.json")
    _, grouped = _block_run(subgraph_json=sg)
    _, plain = _block_run()
    # every candidate moves anyway (threshold 0): same model as without it
    assert grouped.SerializeToString() == plain.SerializeToString()
    _block_run(subgraph_json=sg, sensitivity_cache_file=str(cache))
    names = [
        (r["name"], r["candidate_nodes"])
        for r in json.loads(cache.read_text())["results"]
    ]
    assert sorted(names) == [
        ("__ungrouped__", ["n4_Gemm"]),
        ("back", ["n2_Gemm"]),
        ("front", ["n0_Gemm"]),
    ]
    q, _ = _block_run(subgraph_json="/no/such.json")
    assert any("does not exist" in a for a in q.last_approximations)


def test_block_thresholds_stop_the_mixing_like_the_integer_path():
    q, none = _block_run(metric_threshold=None)
    assert not q.last_auto_mixprecision.moved and not _block_weights(none)
    q, tight = _block_run(metric_threshold=1e-9)  # the baseline is already worse
    assert not q.last_auto_mixprecision.moved and not _block_weights(tight)
    q, loose = _block_run(metric_threshold=1e9)
    assert len(q.last_auto_mixprecision.moved) == 3
    assert _block_weights(loose) == {"n0_Gemm", "n2_Gemm", "n4_Gemm"}
    # "quality": the baseline already meeting the threshold needs no work ...
    q, _ = _block_run(metric_threshold=1e9, metric_optimize_object="quality")
    assert not q.last_auto_mixprecision.moved
    # ... and one that does not moves candidates until the score drops to it
    baseline = q.last_auto_mixprecision.baseline_score
    q, _ = _block_run(
        metric_threshold=baseline * 0.999, metric_optimize_object="quality"
    )
    assert q.last_auto_mixprecision.moved


def test_block_dual_nodes_only_when_asked():
    _, single = _block_run(dual_quant_nodes=False)
    _, dual = _block_run(dual_quant_nodes=True)
    extra = lambda m: {n.name for n in m.graph.node if "_additional_" in n.name}  # noqa: E731
    assert not extra(single) and extra(dual)


def test_block_candidates_are_scored_on_the_fake_quantized_models():
    from onnxsim.quark_preset_graphs import apply_mixed_block_format

    model, data = _mlp(), _data()
    res = amp.auto_mixprecision_blocks(
        model, data, "bfp16", metric_threshold=None, data_size=len(data)
    )
    sess = ort.InferenceSession(model.SerializeToString())
    base = apply_mixed_block_format(
        model, "bfp16", include_layers=["no-such-layer"], dual_nodes=False
    )
    ref = [sess.run(None, d)[0] for d in data]
    got = [o[0] for o in run_fake_quantized(base, data)]
    want = float(np.mean([np.linalg.norm(a - b) for a, b in zip(ref, got)]))
    assert res.baseline_score == pytest.approx(want, rel=1e-6)
    assert [c.score for c in res.ranked] == sorted(c.score for c in res.ranked)
    # moving a candidate changes its score: the block format really is applied
    assert all(c.score != res.baseline_score for c in res.ranked)


def test_reference_evaluator_runs_the_custom_ops_bit_exactly():
    x = np.random.default_rng(1).standard_normal((2, 32)).astype(np.float32)
    g = parser.parse_model(
        """<ir_version: 9, opset_import: ["": 17, "com.amd.quark": 1]>
        g (float[2,32] x) => (float[2,32] y) {
            a = com.amd.quark.BFPQuantizeDequantize<bfp_method="to_bfp", axis=1,
                bit_width=16, block_size=8, rounding_mode=2>(x)
            y = com.amd.quark.MXQuantizeDequantize<element_dtype="int8", axis=1,
                block_size=32, rounding_mode=2>(a)
        }"""
    )
    (y,) = run_fake_quantized(g, [{"x": x}])[0]
    np.testing.assert_array_equal(y, qbf.mx(qbf.bfp16(x, axis=1), "int8", axis=1))


# -- DedicateDQNode ---------------------------------------------------------------------


def test_dedicate_dq_nodes_copies_a_shared_dequantizer_per_reader():
    m = parser.parse_model(
        """<ir_version: 9, opset_import: ["": 17, "com.amd.quark": 1]>
        g (float[2] x) => (float[2] a, float[2] b, float[2] c) {
            q = com.amd.quark.ExtendedQuantizeLinear(x, s, z)
            d = com.amd.quark.ExtendedDequantizeLinear(q, s, z)
            a = Relu(d)
            b = Neg(d)
            c = Identity(d)
        }"""
    )
    m.graph.initializer.append(numpy_helper.from_array(np.float32(1.0), "s"))
    m.graph.initializer.append(numpy_helper.from_array(np.float32(0.0), "z"))
    out = dedicate_dq_nodes(m)
    dqs = [n for n in out.graph.node if n.op_type == "ExtendedDequantizeLinear"]
    assert len(dqs) == 3
    assert [n.output[0] for n in dqs] == ["d", "d_1", "d_2"]
    reads = {
        n.op_type: n.input[0]
        for n in out.graph.node
        if n.op_type in ("Relu", "Neg", "Identity")
    }
    assert reads == {"Relu": "d", "Neg": "d_1", "Identity": "d_2"}


# -- unpromoted layers keep Quark's bias scale shape ----------------------------------


def test_unpromoted_int32_bias_scale_is_a_one_element_vector():
    target = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec())
    _, out = _quantize(
        _amp_config(qc.UInt8Spec, qc.Int8Spec, target, include_layers=["n2_Gemm"])
    )
    shapes = {}
    for node in ("n0_Gemm", "n2_Gemm", "n4_Gemm"):
        _, scale, zp, _ = _dq_const(out, node, 2)
        shapes[node] = (scale.shape, zp.shape)
    # Quark: a baseline per-tensor bias scale is a one-element vector, the
    # refreshed scale of a promoted layer (input scale * weight scale) a scalar
    assert shapes == {"n0_Gemm": ((1,), ()), "n2_Gemm": ((), ()), "n4_Gemm": ((1,), ())}


# -- Quark's scale / zero point arithmetic -------------------------------------------------


def test_compute_scale_zp_follows_quarks_formulas():
    from onnxsim.quark_mixing import compute_scale_zp, compute_scale_zp_fp

    f32 = lambda v: np.array(v, np.float32)  # noqa: E731
    zp, scale = compute_scale_zp(f32(-1.0), f32(2.0), "uint8", False)
    assert scale == np.float32(3.0 / 255) and zp == 85 and zp.dtype == np.uint8
    zp, scale = compute_scale_zp(f32(-1.0), f32(2.0), "int8", True)
    assert scale == np.float32(4.0 / 254) and zp == 0 and zp.dtype == np.int8
    # symmetric uint8: the zero point 127 is bumped to 128
    zp, _ = compute_scale_zp(f32(-1.0), f32(1.0), "uint8", True)
    assert zp == 128
    # power-of-two scales round in the log domain (to the nearest, not the next)
    zp, scale = compute_scale_zp(f32(-1.0), f32(2.0), "int8", True, pof2=True)
    assert scale == np.float32(2.0**-6) and zp == 0
    zp, scale = compute_scale_zp(f32(0.0), f32(3.0), "uint8", False, pof2=True)
    assert scale == np.float32(2.0**-6)  # 3 / 255 = 0.0118 -> 2**-6.4 -> 2**-6
    assert zp == 0
    # half types: scale 1, zero point 0 (symmetric) or the type's lowest value
    zp, scale = compute_scale_zp_fp(f32(-3.0), f32(5.0), "bfloat16", True)
    assert scale == 1.0 and float(np.asarray(zp, np.float32)) == 0.0
    zp, scale = compute_scale_zp_fp(f32(-3.0), f32(5.0), "float16", False)
    assert scale == 1.0 and float(zp) == -65504.0 + 3.0 or float(zp) == -65504.0
    zp, _ = compute_scale_zp_fp(f32(0.0), f32(5.0), "bfloat16", False)
    assert float(np.asarray(zp, np.float32)) == float(np.float32(-3.38953139e38))


# -- shared scale / zero point initializers -------------------------------------------------


def _shared_qdq_graph():
    """x -> Q/DQ(s, z) -> Transpose -> Q/DQ(s, z: the same initializers) -> MatMul(w):
    what ONNX Runtime's quantizer builds around a pass-through op."""
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[4,4] x) => (float[4,4] y)
        <float s = {0.0625}, uint8 z = {128}>
        {
            qa = QuantizeLinear(x, s, z)
            da = DequantizeLinear(qa, s, z)
            t = Transpose<perm=[1,0]>(da)
            qb = QuantizeLinear(t, s, z)
            db = DequantizeLinear(qb, s, z)
            y = MatMul(db, w)
        }
        """
    )
    m.graph.initializer.append(
        numpy_helper.from_array(np.eye(4, dtype=np.float32) * 0.5, "w")
    )
    for n, name in zip(m.graph.node, ("qa", "da", "tr", "qb", "db", "mm")):
        n.name = name
    return m


@pytest.mark.parametrize("mode", ["propagate", "unshare"])
def test_shared_params_follow_the_promoted_pair_only_in_propagate_mode(mode):
    from onnxsim.quark_mixing import QuarkMixer

    ranges = {"t": (-2.0, 3.0), "x": (-1.0, 1.0)}
    mixer = QuarkMixer(_shared_qdq_graph(), ranges, mode)
    mixer.promote_node("mm", amp.TargetSpec(inputs=("uint16", False)))
    out = mixer.result()
    inits = {t.name: numpy_helper.to_array(t) for t in out.graph.initializer}
    nodes = {n.name: n for n in out.graph.node}
    for name in ("qb", "db"):  # the promoted pair, whatever the mode
        assert nodes[name].domain == "com.microsoft"
        assert inits[nodes[name].input[2]].dtype == np.uint16
    want = (-2.0, 3.0)  # the promoted tensor's range sets the new scale
    assert inits[nodes["db"].input[1]] == np.float32((want[1] - want[0]) / 65535)
    if mode == "propagate":
        # the other pair reads the same (rewritten) initializers: same domain,
        # same new values
        assert nodes["qa"].input[1:] == nodes["qb"].input[1:]
        assert nodes["qa"].domain == nodes["da"].domain == "com.microsoft"
    else:
        # the pair got its own copy; the first one still reads the originals
        assert nodes["qb"].input[1:] != nodes["qa"].input[1:]
        assert nodes["qa"].domain == nodes["da"].domain == ""
        assert inits[nodes["qa"].input[2]].dtype == np.uint8
        assert inits[nodes["qa"].input[1]] == np.float32(0.0625)
    onnx.checker.check_model(out)


def test_baseline_pass_through_quantizers_share_initializers_like_quarks():
    # conv -> MaxPool: ONNX Runtime's quantizer gives the pool's output quantizer
    # the conv output's scale / zero point initializers
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[1,1,6,6] x) => (float[1,1,3,3] y)
        {
            c = Conv<pads=[1,1,1,1]>(x, w)
            y = MaxPool<kernel_shape=[2,2],strides=[2,2]>(c)
        }
        """
    )
    m.graph.initializer.append(
        numpy_helper.from_array(
            np.random.default_rng(0).standard_normal((1, 1, 3, 3)).astype(np.float32),
            "w",
        )
    )
    for n in m.graph.node:
        n.name = n.op_type.lower()
    data = [
        {"x": np.random.default_rng(i).standard_normal((1, 1, 6, 6)).astype(np.float32)}
        for i in range(3)
    ]
    res = amp.auto_mixprecision(
        m,
        data,
        base_dtype="uint8",
        target_dtype="uint16",
        target_op_types=("Conv",),
        metric_threshold=0,
    )
    nodes = {n.name: n for n in res.model.graph.node if n.op_type == "MaxPool"}
    prod = {o: n for n in res.model.graph.node for o in n.output}
    cons = {x: n for n in res.model.graph.node for x in n.input}
    pool = nodes["maxpool"]
    pair_after = cons[pool.output[0]]  # the pool's output Q
    pair_before = prod[prod[pool.input[0]].input[0]]  # the conv output's Q
    assert pair_after.op_type == pair_before.op_type == "QuantizeLinear"
    assert pair_after.input[1:] == pair_before.input[1:]
    # propagate (Quark's default): promoting the conv output moved the pool's too
    zp = {t.name: t for t in res.model.graph.initializer}[pair_after.input[2]]
    assert zp.data_type == onnx.TensorProto.UINT16
    assert pair_after.domain == "com.microsoft"
    unshared = amp.auto_mixprecision(
        m,
        data,
        base_dtype="uint8",
        target_dtype="uint16",
        target_op_types=("Conv",),
        metric_threshold=0,
        shared_param_mode="unshare",
    )
    cons_u = {x: n for n in unshared.model.graph.node for x in n.input}
    pool_u = next(n for n in unshared.model.graph.node if n.op_type == "MaxPool")
    assert cons_u[pool_u.output[0]].domain == ""  # the pool's quantizer stays uint8


def test_shared_param_mode_is_validated():
    with pytest.raises(ValueError, match="shared_param_mode"):
        amp.auto_mixprecision(
            _mlp(),
            _data(),
            base_dtype="uint8",
            target_dtype="uint16",
            shared_param_mode="share",
        )


# -- the sensitivity cache: Quark's key, Quark's schema -------------------------------------

# Quark 0.13's cache_key of its quantized baseline of ``_mlp`` for these requests
# (read from the file its AutoMixprecision wrote; pins the naming reproduced in
# onnxsim.quark_amp_cache)
_QUARK_KEYS = {
    "u8-u16": (
        (qc.UInt8Spec, qc.Int8Spec),
        (qc.UInt16Spec, qc.Int8Spec),
        "685165232fb00e448cddbaf9ef05ed26369387301100fed4ed63d6ce3318c141",
    ),
    "bf16-int8": (
        (qc.BFloat16Spec, qc.BFloat16Spec),
        (qc.Int8Spec, qc.Int8Spec),
        "20c66bdab16aa60a49a0d101097ebdf4069cd2903bc83a09750a14b3583736c8",
    ),
    "bfp16-mxint8": (
        (qc.BFP16Spec, qc.BFP16Spec),
        (qc.MXInt8Spec, qc.MXInt8Spec),
        "1b65a86389e566b9602aea38dfacc14bae72b91774091a38214dc6369df4fb54",
    ),
    "u8-xint8": (
        (qc.UInt8Spec, qc.Int8Spec),
        (qc.XInt8Spec, qc.XInt8Spec),
        "0d6f28c894cabc339c685489212925a702f7c5a2fe1c917604089b66248ae477",
    ),
}


def _cfg_of(case, **params):
    (a, w), (ta, tw), _ = _QUARK_KEYS[case]
    return _amp_config(
        a, w, qc.QLayerConfig(activation=ta(), weight=tw()), data_size=3, **params
    )


@pytest.mark.parametrize("case", sorted(_QUARK_KEYS))
def test_cache_is_written_under_quarks_key(case, tmp_path):
    cache = tmp_path / "c.json"
    _quantize(_cfg_of(case, metric_threshold=None, sensitivity_cache_file=str(cache)))
    doc = json.loads(cache.read_text())
    assert doc["cache_key"] == _QUARK_KEYS[case][2]
    assert {"name", "candidate_nodes", "score", "all_config_scores", "enabled"} <= set(
        doc["results"][0]
    )


@pytest.mark.parametrize("case", sorted(_QUARK_KEYS))
def test_a_cache_quark_wrote_is_read_not_recomputed(case, tmp_path):
    """A file in Quark's schema and under Quark's key drives the mixing: its
    ranking (not a recomputed one) and its ``enabled`` pins."""
    rows = [
        # scores no computation reproduces: only a read of the file puts n4 first
        dict(name="n4_Gemm", candidate_nodes=["n4_Gemm"], score=0.001),
        dict(name="n0_Gemm", candidate_nodes=["n0_Gemm"], score=0.002, enabled=False),
        dict(name="n2_Gemm", candidate_nodes=["n2_Gemm"], score=0.003),
    ]
    doc = dict(
        version="0.13",
        cache_key=_QUARK_KEYS[case][2],
        results=[
            {
                "all_config_scores": [r["score"]],
                "best_config_index": 0,
                "enabled": True,
                **r,
            }
            for r in rows
        ],
    )
    cache = tmp_path / "quark.json"
    cache.write_text(json.dumps(doc))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q, _ = _quantize(_cfg_of(case, sensitivity_cache_file=str(cache)))
    assert not [w for w in caught if "stale" in str(w.message)]
    res = q.last_auto_mixprecision
    assert [r.name for r in res.ranked] == ["n4_Gemm", "n0_Gemm", "n2_Gemm"]
    assert res.moved == ["n4_Gemm", "n2_Gemm"]  # n0 is pinned
    assert json.loads(cache.read_text()) == doc  # read, not rewritten


def test_a_cache_under_another_key_is_stale_and_recomputed(tmp_path):
    cache = tmp_path / "c.json"
    doc = dict(
        version="0.13",
        cache_key="0" * 64,
        results=[
            dict(
                name="n4_Gemm",
                candidate_nodes=["n4_Gemm"],
                score=0.0,
                all_config_scores=[0.0],
                best_config_index=0,
                enabled=False,
            )
        ],
    )
    cache.write_text(json.dumps(doc))
    q = qc.ModelQuantizer(_cfg_of("u8-u16", sensitivity_cache_file=str(cache)))
    with pytest.warns(UserWarning, match="stale"):
        q.quantize_model(_mlp(), calibration_data_reader=_data())
    assert len(q.last_auto_mixprecision.ranked) == 3
    assert "n4_Gemm" in q.last_auto_mixprecision.moved  # the bogus pin is gone
    assert json.loads(cache.read_text())["cache_key"] == _QUARK_KEYS["u8-u16"][2]


def test_a_changed_request_changes_the_key():
    from onnxsim.quark_amp_cache import quark_cache_key

    base = _baseline(targets=[("uint16", None)])
    t16 = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec())
    t8 = qc.QLayerConfig(activation=qc.Int8Spec(), weight=qc.Int8Spec())
    ops = ("Conv", "ConvTranspose", "Gemm", "MatMul")
    keys = {
        quark_cache_key(base, t16, ops),
        quark_cache_key(base, t8, ops),
        quark_cache_key(base, t16, ops[:2]),
        quark_cache_key(base, t16, ops, include_layers=["n0_Gemm"]),
        quark_cache_key(base, t16, ops, exclude_layers=["n0_Gemm"]),
        quark_cache_key(base, [t16, t8], ops),
        quark_cache_key(base, {t16: []}, ops),
    }
    assert len(keys) == 7
    assert quark_cache_key(base, t16, ops[::-1]) == quark_cache_key(base, t16, ops)
    # the spelling of the activation spec matters (``input_tensors`` is not
    # ``activation``), as in Quark's ``QLayerConfig.to_dict``
    spelled = qc.QLayerConfig(input_tensors=qc.UInt16Spec(), weight=qc.Int8Spec())
    assert quark_cache_key(base, spelled, ops) != quark_cache_key(base, t16, ops)


# -- the boundary quantizers of ``dual_quant_nodes`` (Quark's post-processing) ------------------


def _boundary_model():
    """``x -> Q/DQ(uint8) -> Neg -> Q/DQ(uint16) -> Relu``: Neg is the layer
    that was promoted from uint8 to uint16 outputs."""
    m = parser.parse_model(
        """<ir_version: 9, opset_import: ["": 13]>
        g (float[2,4] x) => (float[2,4] y) {
            xq = QuantizeLinear(x, s8, z8)
            xd = DequantizeLinear(xq, s8, z8)
            a = Neg(xd)
            aq = QuantizeLinear(a, s16, z16)
            ad = DequantizeLinear(aq, s16, z16)
            y = Relu(ad)
        }"""
    )
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}_{n.op_type}"
    m.graph.initializer.extend(
        [
            numpy_helper.from_array(np.float32(0.01), "s8"),
            numpy_helper.from_array(np.uint8(3), "z8"),
            numpy_helper.from_array(np.float32(0.0002), "s16"),
            numpy_helper.from_array(np.uint16(7), "z16"),
        ]
    )
    return m


def test_boundary_pass_inserts_the_neighbours_precision_in_front_of_a_promoted_node():
    from onnxsim.quark_boundary_qdq import insert_boundary_quant_nodes
    from onnxsim.quark_mixing import compute_scale_zp

    ranges = {"xd": (-1.0, 3.0), "a": (-3.0, 1.0)}
    out = insert_boundary_quant_nodes(
        _boundary_model(),
        ranges.get,
        promoted_tensors={"a"},
        promoted_nodes={"n2_Neg"},
    )
    extra = [n for n in out.graph.node if "_additional_" in n.name]
    assert [n.op_type for n in extra] == ["QuantizeLinear", "DequantizeLinear"]
    inits = {t.name: numpy_helper.to_array(t) for t in out.graph.initializer}
    q, dq = extra
    # the template is Neg's promoted output quantizer: a uint16 pair, with a
    # scale / zero point computed from the range of the tensor in front of Neg
    zp, scale = compute_scale_zp(
        np.float32(-1.0), np.float32(3.0), "uint16", symmetric=False
    )
    assert inits[q.input[2]].dtype == np.uint16
    assert inits[q.input[1]] == scale and inits[q.input[2]] == zp
    assert (dq.input[1], dq.input[2]) == (q.input[1], q.input[2])
    neg = next(n for n in out.graph.node if n.op_type == "Neg")
    assert neg.input[0] == dq.output[0] and q.input[0] == "xd"
    # the other quantizers are untouched
    assert len(out.graph.node) == len(_boundary_model().graph.node) + 2


def test_boundary_pass_does_nothing_without_a_boundary_or_a_promotion():
    from onnxsim.quark_boundary_qdq import insert_boundary_quant_nodes

    model = _boundary_model()
    # nothing promoted: no tensor has an override, so no template is found
    same = insert_boundary_quant_nodes(model, {}.get)
    assert [n.name for n in same.graph.node] == [n.name for n in model.graph.node]
    # a node whose quantizers all agree needs no extra pair
    flat = _boundary_model()
    for n in flat.graph.node:
        n.input[:] = [
            "s8" if x == "s16" else "z8" if x == "z16" else x for x in n.input
        ]
    out = insert_boundary_quant_nodes(
        flat, {}.get, promoted_tensors={"a"}, promoted_nodes={"n2_Neg"}
    )
    assert not [n for n in out.graph.node if "_additional_" in n.name]


def test_boundary_pass_keeps_the_template_scale_for_a_tensor_without_a_range():
    from onnxsim.quark_boundary_qdq import insert_boundary_quant_nodes

    out = insert_boundary_quant_nodes(
        _boundary_model(),
        {}.get,
        promoted_tensors={"a"},
        promoted_nodes={"n2_Neg"},
    )
    q = next(n for n in out.graph.node if n.name.endswith("_additional_QuantizeLinear"))
    assert (q.input[1], q.input[2]) == ("s16", "z16")  # shared with the template


def test_dual_quant_nodes_leave_the_scored_models_alone():
    """Quark scores the candidates without the boundary pairs and inserts them
    into the final model only."""
    target = qc.QLayerConfig(activation=qc.UInt16Spec(), weight=qc.Int8Spec())
    kw = dict(include_layers=["n2_Gemm"])
    q, _ = _quantize(_amp_config(qc.UInt8Spec, qc.Int8Spec, target, **kw))
    q2, dual = _quantize(
        _amp_config(qc.UInt8Spec, qc.Int8Spec, target, dual_quant_nodes=True, **kw)
    )
    assert q.last_auto_mixprecision.baseline_score == pytest.approx(
        q2.last_auto_mixprecision.baseline_score
    )
    assert [c.score for c in q.last_auto_mixprecision.ranked] == [
        c.score for c in q2.last_auto_mixprecision.ranked
    ]
    assert _extra_pairs(dual)


def test_dual_quant_nodes_are_not_added_to_a_threshold_free_analysis():
    target = qc.QLayerConfig(activation=qc.BFloat16Spec(), weight=qc.BFloat16Spec())
    _, out = _quantize(
        _amp_config(
            qc.UInt8Spec,
            qc.Int8Spec,
            target,
            metric_threshold=None,
            dual_quant_nodes=True,
        )
    )
    assert not _extra_pairs(out)  # Quark returns the baseline before mixing


# -- half / block activations over any constant format --------------------------------------


def _generic(act, wt, **extra):
    cfg = qc.QConfig(
        global_config=qc.QLayerConfig(activation=act(), weight=wt()), **extra
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            _mlp(), calibration_data_reader=_data()
        )


def _fn_nodes(model):
    return [
        n
        for n in model.graph.node
        if n.op_type in ("BFPQuantizeDequantize", "MXQuantizeDequantize")
    ]


def _fn_attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


@pytest.mark.parametrize(
    "wt, op",
    [
        (qc.BFP16Spec, "BFPQuantizeDequantize"),
        (qc.MX4Spec, "BFPQuantizeDequantize"),
        (qc.MX9Spec, "BFPQuantizeDequantize"),
        (qc.MXInt8Spec, "MXQuantizeDequantize"),
        (qc.MXFP4E2M1Spec, "MXQuantizeDequantize"),
    ],
)
def test_block_constants_under_half_activations_use_quarks_default_attributes(wt, op):
    """Quark's quantizer picks the node by the constant's format family and
    fills in its plain defaults (``rounding_mode`` 0, 16-bit ``to_bfp`` /
    ``int8`` MX) -- the format-specific attributes only reach a pair of the
    *same* ``MX*`` format."""
    out = _generic(qc.BFloat16Spec, wt)
    consts = [n for n in _fn_nodes(out) if n.input[0] in {"w1", "w2", "w3"}]
    assert consts and {n.op_type for n in consts} == {op}
    for n in consts:
        a = _fn_attrs(n)
        assert a["rounding_mode"] == 0
        if op == "BFPQuantizeDequantize":
            assert (a["bfp_method"], a["bit_width"], a["block_size"]) == (
                b"to_bfp",
                16,
                8,
            )
        else:
            assert (a["element_dtype"], a["block_size"]) == (b"int8", 32)


def test_a_pair_of_the_same_mx_format_gets_its_own_attributes():
    out = _generic(qc.MX4Spec, qc.MX4Spec)
    a = _fn_attrs(_fn_nodes(out)[0])
    assert (a["bfp_method"], a["bit_width"], a["rounding_mode"]) == (
        b"to_bfp_prime",
        11,
        2,
    )
    mixed = _generic(qc.MX4Spec, qc.MX6Spec)
    a = _fn_attrs(_fn_nodes(mixed)[0])
    assert (a["bfp_method"], a["bit_width"], a["rounding_mode"]) == (b"to_bfp", 16, 0)


def test_an_explicit_bfp_attributes_option_wins_over_the_defaults():
    out = _generic(
        qc.BFloat16Spec,
        qc.BFP16Spec,
        BFPAttributes=dict(bit_width=12, rounding_mode=2),
    )
    consts = [n for n in _fn_nodes(out) if n.input[0] in {"w1", "w2", "w3"}]
    got = {(_fn_attrs(n)["bit_width"], _fn_attrs(n)["rounding_mode"]) for n in consts}
    assert got == {(12, 2)}


@pytest.mark.parametrize("wt", [qc.Int8Spec, qc.UInt8Spec])
@pytest.mark.parametrize("act", [qc.BFloat16Spec, qc.Float16Spec, qc.BFP16Spec])
def test_integer_constants_under_half_and_block_activations(act, wt):
    """Weights are offline per-tensor integer codes behind a ``DequantizeLinear``;
    a bias is int32 on the weight's scale behind half-precision activations
    (their scale is 1.0) and quantized like a weight behind block ones."""
    out = _generic(act, wt)
    inits = {t.name: numpy_helper.to_array(t) for t in out.graph.initializer}
    want = np.int8 if wt is qc.Int8Spec else np.uint8
    assert inits["w1_quantized"].dtype == want
    assert inits["w1_scale"].shape == () and inits["w1_scale"].dtype == np.float32
    if act is qc.BFP16Spec:
        assert inits["b1_quantized"].dtype == want
    else:
        assert inits["b1_quantized"].dtype == np.int32
        np.testing.assert_array_equal(
            inits["b1_quantized_scale"], np.array([inits["w1_scale"]], np.float32)
        )
        assert inits["b1_quantized_zero_point"] == 0
    # the codes are the float weights on that grid
    w = _w(np.random.default_rng(0), D, D, scale=1.5)
    deq = (
        inits["w1_quantized"].astype(np.float32)
        - inits["w1_zero_point"].astype(np.float32)
    ) * inits["w1_scale"]
    assert np.abs(deq - w).max() <= float(inits["w1_scale"]) / 2 + 1e-6


@pytest.mark.parametrize("act", [qc.Int8Spec, qc.UInt8Spec, qc.Int16Spec])
def test_integer_activations_over_half_or_block_constants_run(act):
    """Integer activations keep their calibrated Q/DQ pairs; the half-format
    constants get an ``ExtendedQuantizeLinear`` / ``ExtendedDequantizeLinear``
    pair each, so the graph runs through Quark's extended Q/DQ ops."""
    out = _generic(act, qc.BFloat16Spec)
    assert not _fn_nodes(out)
    # every constant of the graph (the three weights and the three biases)
    const_q = {
        n.output[0] for n in out.graph.node if n.op_type == "ExtendedQuantizeLinear"
    }
    assert const_q == {
        f"{c}_QuantizeLinear_Output" for c in ("w1", "b1", "w2", "b2", "w3", "b3")
    }
    # the activations keep the plain, calibrated integer Q/DQ pairs
    for n in out.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            assert all(not x.endswith("_QuantizeLinear_Output") for x in n.input)


@pytest.mark.parametrize(
    "act, wt",
    [
        (qc.BFloat16Spec, qc.MX6Spec),
        (qc.BFloat16Spec, qc.Float16Spec),
        (qc.Float16Spec, qc.BFloat16Spec),
        (qc.BFP16Spec, qc.MXInt8Spec),
        (qc.MX4Spec, qc.BFloat16Spec),
        (qc.BFP16Spec, qc.Int8Spec),
    ],
)
def test_auto_mixprecision_over_a_mixed_format_baseline_runs(act, wt):
    target = qc.QLayerConfig(activation=qc.Int8Spec(), weight=qc.Int8Spec())
    q, out = _quantize(
        _amp_config(act, wt, target, include_layers=["n2_Gemm"], dual_quant_nodes=True)
    )
    assert q.last_auto_mixprecision.moved == ["n2_Gemm"]
    assert _extra_pairs(out)
