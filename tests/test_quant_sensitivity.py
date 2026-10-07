"""Tests for onnxsim.quant_sensitivity (small synthetic nets; no downloads, no Z3).

The real-model evaluation (does any estimator actually rank layers well?) is in
scripts/quant_sensitivity_bench.py and docs/quant-sensitivity.md; these tests pin the
machinery: sites, quantizers, the differentiable executor, estimator definitions, the
brute-force oracle, rank correlations and the decision helper.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import quant_sensitivity as qs

torch = pytest.importorskip("torch")


def _model(body, initializer=None, opset=17, ir_version=9):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(np.asarray(v), k)
        for k, v in (initializer or {}).items()
    )
    return model


def _mlp(seed, scales=(1.0, 1.0, 1.0), width=12, classes=5):
    """MLP classifier with per-layer weight scales (so layers differ in sensitivity)."""
    rng = np.random.default_rng(seed)
    dims = [width, width, width, classes]
    init, lines, prev = {}, [], "x"
    for i, s in enumerate(scales):
        init[f"W{i}"] = (
            rng.standard_normal((dims[i], dims[i + 1])) * s / np.sqrt(dims[i])
        ).astype(np.float32)
        init[f"B{i}"] = (rng.standard_normal(dims[i + 1]) * 0.1).astype(np.float32)
        out = "logits" if i == len(scales) - 1 else f"r{i}"
        lines.append(f"h{i} = MatMul({prev}, W{i})")
        lines.append(f"a{i} = Add(h{i}, B{i})")
        lines.append(
            f"{out} = Identity(a{i})" if out == "logits" else f"{out} = Relu(a{i})"
        )
        prev = out
    body = (
        f"m (float[N,{width}] x) => (float[N,{classes}] logits) {{ "
        + " ".join(lines)
        + " }"
    )
    return _model(body, init), rng


def _conv_net(seed):
    rng = np.random.default_rng(seed)
    f = lambda *s: (rng.standard_normal(s) * 0.3).astype(np.float32)  # noqa: E731
    body = """
    m (float[N,3,8,8] x) => (float[N,4] logits) {
      c0 = Conv<pads=[1,1,1,1]>(x, W0, B0)
      r0 = Relu(c0)
      p0 = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r0)
      c1 = Conv<pads=[1,1,1,1]>(p0, W1, B1)
      s1 = Add(c1, p0)
      r1 = Relu(s1)
      g = GlobalAveragePool(r1)
      f = Flatten(g)
      logits = Gemm<transB=1>(f, W2, B2)
    }"""
    return _model(
        body,
        dict(W0=f(6, 3, 3, 3), B0=f(6), W1=f(6, 6, 3, 3), B1=f(6), W2=f(4, 6), B2=f(4)),
    ), rng


def _data(rng, shape, n=48):
    return rng.standard_normal((n,) + shape).astype(np.float32)


# ---- sites, quantizers, model surgery ---------------------------------------------------------


def test_find_sites_covers_conv_gemm_matmul_and_skips_dynamic_weights():
    m, _ = _conv_net(0)
    sites = qs.find_sites(m)
    assert [s.op_type for s in sites] == ["Conv", "Conv", "Gemm"]
    assert [s.channel_axis for s in sites] == [
        0,
        0,
        0,
    ]  # Gemm transB=1: rows are output channels
    mlp, _ = _mlp(0)
    assert [(s.op_type, s.channel_axis) for s in qs.find_sites(mlp)] == [
        ("MatMul", 1)
    ] * 3
    attn = _model(
        "m (float[1,4,8] a, float[1,8,4] b) => (float[1,4,4] y) { y = MatMul(a, b) }"
    )  # both operands are activations: not a site
    assert qs.find_sites(attn) == []


def test_weight_quantizer_is_per_channel_symmetric_and_within_half_a_step():
    rng = np.random.default_rng(0)
    w = (
        rng.standard_normal((6, 5)).astype(np.float32)
        * np.array([1, 10, 0.1, 3, 1, 1], np.float32)[:, None]
    )
    for bits in (3, 4, 8):
        q = qs.quantize_weights(w, bits, axis=0)
        step = np.abs(w).max(axis=1, keepdims=True) / (2 ** (bits - 1) - 1)
        assert np.all(np.abs(q - w) <= step / 2 + 1e-6)
        assert np.array_equal(qs.quantize_weights(q, bits, axis=0), q)  # idempotent
        assert all(
            len(np.unique(row)) <= 2**bits for row in q
        )  # at most 2^bits levels per channel


def test_with_quantized_weights_touches_only_the_chosen_sites():
    m, _ = _conv_net(1)
    sites = qs.find_sites(m)
    q = qs.with_quantized_weights(m, [sites[1]], bits=4)
    before, after = qs._initializers(m), qs._initializers(q)
    assert not np.array_equal(before["W1"], after["W1"])
    for k in ("W0", "W2", "B0", "B1", "B2"):
        assert np.array_equal(before[k], after[k])
    onnx.checker.check_model(q)


@pytest.mark.parametrize("bits", [8, 4])
def test_activation_quantizer_matches_numpy_fake_quant(bits):
    m, rng = _mlp(2)
    site = qs.find_sites(m)[1]
    x = _data(rng, (12,))
    quants = qs.calibrate_activations(m, [site], {"x": x}, bits)
    aq = quants[site.name]
    q = qs.with_quantized_activations(m, [site], quants)
    onnx.checker.check_model(q)
    has_qdq = any(n.op_type == "QuantizeLinear" for n in q.graph.node)
    assert has_qdq == (bits == 8)  # real QDQ at 8 bits, float emulation otherwise
    # reference: run the float model up to the site input, fake-quantize in numpy, finish by hand
    consts = qs._initializers(m)
    a = np.maximum(x @ consts["W0"] + consts["B0"], 0)
    ref = (aq.apply(a) @ consts["W1"] + consts["B1"]).clip(0) @ consts["W2"] + consts[
        "B2"
    ]
    got = qs.run_logits(q, {"x": x})
    np.testing.assert_allclose(got, ref, atol=1e-4)


# ---- differentiable executor ------------------------------------------------------------------


def test_torch_graph_matches_onnxruntime_on_a_conv_net():
    m, rng = _conv_net(3)
    x = _data(rng, (3, 8, 8), 4)
    ref = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]
    out, _ = qs.TorchGraph(m).run({"x": torch.from_numpy(x)})
    np.testing.assert_allclose(out["logits"].detach().numpy(), ref, atol=1e-5)


def test_torch_graph_matches_onnxruntime_on_a_transformer_style_block():
    rng = np.random.default_rng(4)
    f = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)  # noqa: E731
    body = """
    m (int64[1,6] ids) => (float[1,3] logits) {
      e = Gather<axis=0>(E, ids)
      n = LayerNormalization<axis=-1, epsilon=1e-5>(e, G, Bt)
      q = MatMul(n, Wq)
      k = MatMul(n, Wk)
      v = MatMul(n, Wv)
      kt = Transpose<perm=[0,2,1]>(k)
      s = MatMul(q, kt)
      sc = Div(s, Sc)
      p = Softmax<axis=-1>(sc)
      c = MatMul(p, v)
      o = MatMul(c, Wo)
      r = Add(o, n)
      pooled = ReduceMean<axes=[1], keepdims=0>(r)
      logits = Gemm(pooled, Wc, Bc)
    }"""
    m = _model(
        body,
        dict(E=f(20, 8), G=np.ones(8, np.float32), Bt=np.zeros(8, np.float32), Wq=f(8, 8), Wk=f(8, 8),
             Wv=f(8, 8), Wo=f(8, 8), Sc=np.float32(2.8), Wc=f(8, 3), Bc=f(3)),
    )  # fmt: skip
    ids = rng.integers(0, 20, (1, 6)).astype(np.int64)
    ref = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"ids": ids})[0]
    out, _ = qs.TorchGraph(m).run({"ids": torch.from_numpy(ids)})
    np.testing.assert_allclose(out["logits"].detach().numpy(), ref, atol=1e-5)
    assert [s.op_type for s in qs.find_sites(m)] == ["MatMul"] * 4 + [
        "Gemm"
    ]  # q, k, v, o, classifier


def test_torch_graph_rejects_an_unsupported_op_by_name():
    m = _model("m (float[1,4] x) => (float[1,4] y) { y = Abs(x) }")
    with pytest.raises(NotImplementedError, match="Abs"):
        qs.TorchGraph(m).run({"x": torch.ones(1, 4)})


# ---- rank correlation --------------------------------------------------------------------------


def test_spearman_and_kendall_known_values():
    assert qs.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert qs.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert qs.kendall([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert qs.kendall([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(
        4 / 6
    )  # 5 concordant, 1 discordant
    assert np.isnan(qs.spearman([1, 1, 1], [1, 2, 3]))  # a constant has no ranking
    # ties get average ranks: both vectors tie the same pair, so the correlation is exactly 1
    assert qs.spearman([1, 1, 2, 3], [5, 5, 6, 7]) == pytest.approx(1.0)


# ---- brute-force oracle and the estimators -----------------------------------------------------


def test_measure_reports_zero_for_an_untouched_model_and_positive_otherwise():
    m, rng = _mlp(5)
    x = _data(rng, (12,), 64)
    groups = qs.as_groups(qs.find_sites(m))
    gt = qs.measure(m, groups, "weights", 4, {"x": x}, labels=None)
    assert set(gt) == set(groups)
    assert all(
        v["kl"] > 0 and 0 <= v["agreement"] <= 1 and len(v["kl_samples"]) == 64
        for v in gt.values()
    )
    assert gt[next(iter(gt))]["kl_samples"].min() >= -1e-12  # KL is non-negative
    # quantizing nothing changes nothing
    ref = qs.run_logits(m, {"x": x})
    assert qs.logit_metrics(ref, ref)["kl"] == pytest.approx(0.0, abs=1e-12)


def test_fisher_ranks_layers_like_the_brute_force_oracle():
    """Over several nets whose layers differ in scale, the Fisher score tracks measured KL.

    Threshold: the mean Spearman over 8 nets must be >= 0.5. Measured when this test was
    written: about 0.8 (8 nets x 3 layers; chance is 0), so 0.5 leaves a wide margin for
    platform noise while still failing if the estimator stopped being data-aware.
    """
    rhos = []
    for seed in range(8):
        scales = tuple(np.random.default_rng(100 + seed).uniform(0.3, 3.0, 3))
        m, rng = _mlp(seed, scales)
        calib, ev = _data(rng, (12,), 128), _data(rng, (12,), 400)
        groups = qs.as_groups(qs.find_sites(m))
        gt = qs.measure(m, groups, "weights", 3, {"x": ev})
        rep = qs.rank(
            m,
            calib={"x": calib},
            kind="weights",
            bits=3,
            methods=("fisher",),
            n_samples=128,
        )
        rhos.append(qs.spearman(rep.scores["fisher"], [gt[g]["kl"] for g in groups]))
    assert np.nanmean(rhos) >= 0.5, rhos


def test_estimators_agree_with_brute_force_on_a_conv_net():
    m, rng = _conv_net(6)
    calib, ev = _data(rng, (3, 8, 8), 96), _data(rng, (3, 8, 8), 300)
    groups = qs.as_groups(qs.find_sites(m))
    gt = np.array(
        [v["kl"] for v in qs.measure(m, groups, "weights", 3, {"x": ev}).values()]
    )
    rep = qs.rank(
        m,
        calib={"x": calib},
        kind="weights",
        bits=3,
        methods=("weight_err", "fisher", "taylor", "hessian_trace"),
        n_samples=96,
    )
    assert set(rep.scores) == {"weight_err", "fisher", "taylor", "hessian_trace"}
    assert all(np.isfinite(s).all() and (s >= 0).all() for s in rep.scores.values())
    assert (
        qs.spearman(rep.scores["fisher"], gt) >= 0.5
    )  # 3 sites: 0.5 means at most one pair swapped
    assert int(np.argmax(rep.scores["fisher"])) == int(np.argmax(gt))


def test_hessian_trace_matches_the_analytic_value_for_softmax_regression():
    """For logits = x @ W the cross-entropy Hessian trace is mean_i |x_i|^2 sum_c p_ic (1 - p_ic)."""
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((6, 4)) * 0.5).astype(np.float32)
    m = _model(
        "m (float[N,6] x) => (float[N,4] logits) { logits = MatMul(x, W) }", {"W": w}
    )
    x = rng.standard_normal((64, 6)).astype(np.float32)
    logits = x @ w
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    trace = float(np.mean((x**2).sum(1) * (p * (1 - p)).sum(1)))
    dnorm = float(np.sum((qs.quantize_weights(w, 3, 1) - w) ** 2))
    expected = trace / w.size * dnorm
    got = qs.rank(
        m,
        calib={"x": x},
        kind="weights",
        bits=3,
        methods=("hessian_trace",),
        n_samples=64,
        probes=400,
        hessian_batch=64,
    ).scores["hessian_trace"][0]
    assert got == pytest.approx(expected, rel=0.2)  # Hutchinson noise at 400 probes


def test_hessian_trace_respects_a_graph_exported_with_a_fixed_batch():
    """A graph with a baked-in batch of 1 cannot take the 16-sample batch the estimator asks for."""
    rng = np.random.default_rng(21)
    w = (rng.standard_normal((6, 4)) * 0.5).astype(np.float32)
    m = _model(
        "m (float[1,6] x) => (float[1,4] logits) { f = MatMul(x, W)  r = Reshape(f, S)  logits = Identity(r) }",
        {"W": w, "S": np.array([1, 4], np.int64)},
    )
    x = rng.standard_normal((12, 6)).astype(np.float32)
    rep = qs.rank(
        m,
        calib={"x": x},
        kind="weights",
        bits=3,
        methods=("hessian_trace", "fisher"),
        n_samples=12,
        probes=4,
        hessian_batch=8,
    )
    assert (
        np.isfinite(rep.scores["hessian_trace"]).all()
        and rep.scores["hessian_trace"][0] > 0
    )


def test_activation_edge_gradient_is_per_consumer_and_first_order_correct():
    """q/k/v-style sites share one input tensor; the edge leaf must isolate each consumer."""
    rng = np.random.default_rng(8)
    f = lambda *s: (rng.standard_normal(s) * 0.5).astype(np.float32)  # noqa: E731
    body = """
    m (float[N,6] x) => (float[N,3] logits) {
      a = MatMul(x, Wa)
      b = MatMul(x, Wb)
      s = Add(a, b)
      r = Relu(s)
      logits = MatMul(r, Wc)
    }"""
    m = _model(body, dict(Wa=f(6, 5), Wb=f(6, 5), Wc=f(5, 3)))
    sites = qs.find_sites(m)
    x = rng.standard_normal((1, 6)).astype(np.float32)
    g = qs.TorchGraph(m)
    out, _ = g.run_with_edge_eps(
        {"x": torch.from_numpy(x)}, [sites[0].node_index, sites[1].node_index]
    )
    logp = torch.log_softmax(out["logits"], -1)[0, 1]
    ga, gb = torch.autograd.grad(
        logp, [g.edge_eps[sites[0].node_index], g.edge_eps[sites[1].node_index]]
    )
    assert not torch.allclose(ga, gb)  # different consumers, different gradients
    # first-order check by finite difference on the real model: perturb ONLY site 0's edge by d
    d = (rng.standard_normal((1, 6)) * 1e-3).astype(np.float32)
    consts = qs._initializers(m)

    def logp_of(da, db):
        h = (
            np.maximum((x + da) @ consts["Wa"] + (x + db) @ consts["Wb"], 0)
            @ consts["Wc"]
        )
        h = h - h.max()
        return float(h[0, 1] - np.log(np.exp(h).sum()))

    fd = logp_of(d, 0 * d) - logp_of(0 * d, 0 * d)
    assert float((ga.detach().numpy() * d).sum()) == pytest.approx(
        fd, rel=0.05, abs=1e-7
    )


def test_activation_fisher_of_a_site_does_not_depend_on_which_other_sites_are_scored():
    m, rng = _mlp(9)
    calib = {"x": _data(rng, (12,), 64)}
    sites = qs.find_sites(m)
    quants = qs.calibrate_activations(m, sites, calib, 4)
    both = qs.rank(
        m,
        sites,
        None,
        calib,
        "activations",
        4,
        ("fisher",),
        n_samples=64,
        quants=quants,
    )
    alone = qs.rank(
        m,
        [sites[1]],
        None,
        calib,
        "activations",
        4,
        ("fisher",),
        n_samples=64,
        quants=quants,
    )
    assert alone.scores["fisher"][0] == pytest.approx(
        both.scores["fisher"][both.groups.index(sites[1].name)], rel=1e-5
    )


def test_group_score_is_the_square_of_the_summed_first_order_change():
    """Grouped sites cancel/reinforce before squaring (one block = one perturbation)."""
    m, rng = _mlp(10)
    calib = {"x": _data(rng, (12,), 48)}
    sites = qs.find_sites(m)
    names = [s.name for s in sites]
    single = qs.rank(
        m, sites, None, calib, "weights", 3, ("fisher",), n_samples=48
    ).scores["fisher"]
    block = qs.rank(
        m, sites, {"all": names}, calib, "weights", 3, ("fisher",), n_samples=48
    ).scores["fisher"][0]
    assert block >= 0 and not np.isclose(
        block, single.sum()
    )  # not a plain sum of the parts


# ---- certified method and guards ---------------------------------------------------------------


def test_certified_bound_dominates_the_measured_error_for_every_site():
    m, rng = _mlp(11, width=6)
    box = {"x": (-1.0, 1.0)}
    rep = qs.rank(m, kind="weights", bits=3, methods=("certified",), input_ranges=box)
    # finite and positive first: `inf >= measured` would make the domination check below vacuous
    assert (
        np.isfinite(rep.scores["certified"]).all()
        and (rep.scores["certified"] > 0).all()
    )
    groups = qs.as_groups(qs.find_sites(m))
    x = rng.uniform(-1, 1, (4000, 6)).astype(np.float32)
    ref = qs.run_logits(m, {"x": x})
    for i, (g, members) in enumerate(groups.items()):
        v = qs.with_quantized_weights(m, members, 3)
        measured = float(np.abs(qs.run_logits(v, {"x": x}) - ref).max())
        assert rep.scores["certified"][i] >= measured, (
            g,
            rep.scores["certified"][i],
            measured,
        )


def test_certified_is_skipped_with_a_reason_when_too_large_or_unbounded():
    m, rng = _mlp(12, width=6)
    big = qs.rank(
        m,
        kind="weights",
        bits=3,
        methods=("weight_err", "certified"),
        input_ranges={"x": (-1.0, 1.0)},
        max_certified_elements=10,
    )
    assert "certified" not in big.scores and "too large" in big.skipped["certified"]
    unbounded = qs.rank(m, kind="weights", bits=3, methods=("certified",))
    assert (
        "certified" not in unbounded.scores
        and "input_ranges" in unbounded.skipped["certified"]
    )


def test_certified_activation_bound_dominates_measured_error_with_interval_calibration():
    """8-bit activation quantizers calibrated by interval analysis never clip inside the box,
    so quant_verify's bound is finite and must dominate what sampled inputs actually show."""
    m, rng = _mlp(13, width=6)
    box = {"x": (-1.0, 1.0)}
    sites = qs.find_sites(m)
    quants = qs.calibrate_activations_interval(m, sites, box, 8)
    rep = qs.rank(
        m,
        kind="activations",
        act_bits=8,
        methods=("certified",),
        input_ranges=box,
        quants=quants,
    )
    assert (
        np.isfinite(rep.scores["certified"]).all()
        and (rep.scores["certified"] > 0).all()
    )
    x = rng.uniform(-1, 1, (4000, 6)).astype(np.float32)
    ref = qs.run_logits(m, {"x": x})
    for i, s in enumerate(sites):
        v = qs.with_quantized_activations(m, [s], quants)
        measured = float(np.abs(qs.run_logits(v, {"x": x}) - ref).max())
        assert rep.scores["certified"][i] >= measured, (
            s.name,
            rep.scores["certified"][i],
            measured,
        )


def test_interval_calibrated_quantizers_do_not_clip_and_cover_the_worst_case_range():
    m, rng = _mlp(18, width=6)
    box = {"x": (-1.0, 1.0)}
    sites = qs.find_sites(m)
    quants = qs.calibrate_activations_interval(m, sites, box, 8)
    x = np.concatenate(
        [rng.uniform(-1, 1, (500, 6)), rng.choice([-1.0, 1.0], (500, 6))]
    ).astype(np.float32)
    consts = qs._initializers(m)
    a = x
    for i, s in enumerate(sites):
        aq = quants[s.name]
        assert (
            np.abs(aq.apply(a) - a).max() <= aq.scale / 2 + 1e-6
        )  # rounding only: no saturation
        a = a @ consts[f"W{i}"] + consts[f"B{i}"]
        a = np.maximum(a, 0) if i < len(sites) - 1 else a


def test_certified_activation_path_needs_8_bit_quantizers():
    m, rng = _mlp(19, width=6)
    box = {"x": (-1.0, 1.0)}
    four = qs.calibrate_activations_interval(m, qs.find_sites(m), box, 4)
    rep = qs.rank(
        m,
        kind="activations",
        act_bits=4,
        methods=("certified",),
        input_ranges=box,
        quants=four,
    )
    assert "certified" not in rep.scores and any(
        "8-bit" in v for v in rep.skipped.values()
    )


def test_static_batch_copy_pins_symbolic_dims_and_leaves_the_original_alone():
    m, _ = _mlp(20)
    s = qs.static_batch_copy(m, 3)
    dims = s.graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [3, 12] and not dims[0].HasField("dim_param")
    orig = m.graph.input[0].type.tensor_type.shape.dim[0]
    assert (
        orig.dim_param == "N" and orig.dim_value == 0
    )  # the original keeps its symbolic batch


# ---- decision helper and report ----------------------------------------------------------------


def test_select_float_sites_returns_the_most_sensitive_first_and_validates_the_method():
    m, rng = _mlp(14, scales=(0.3, 3.0, 1.0))
    rep = qs.rank(
        m,
        calib={"x": _data(rng, (12,), 64)},
        kind="weights",
        bits=3,
        methods=("weight_err", "fisher"),
        n_samples=64,
    )
    order = rep.ranking("fisher")
    assert sorted(order) == sorted(rep.groups)
    assert qs.select_float_sites(rep, 2, "fisher") == order[:2]
    assert qs.select_float_sites(rep, 0, "fisher") == []
    assert len(qs.select_float_sites(rep, 99, "fisher")) == len(rep.groups)
    with pytest.raises(KeyError, match="hessian_trace"):
        qs.select_float_sites(rep, 1, "hessian_trace")
    assert "fisher" in rep.table() and rep.groups[0] in rep.table()


def test_ranking_puts_unscored_groups_last():
    rep = qs.SensitivityReport(
        groups=["a", "b", "c"], members={}, kind="weights", bits=4,
        scores={"m": np.array([1.0, np.nan, 3.0])}, skipped={}, seconds={}, notes=[],
    )  # fmt: skip
    assert rep.ranking("m") == ["c", "a", "b"]


def test_unknown_method_and_kind_are_rejected():
    m, _ = _mlp(15)
    with pytest.raises(ValueError, match="unknown methods"):
        qs.rank(m, methods=("nope",))
    with pytest.raises(ValueError, match="kind"):
        qs.rank(m, kind="gradients")


def test_methods_that_need_data_are_skipped_not_faked_without_calibration():
    m, _ = _mlp(16)
    rep = qs.rank(
        m, kind="weights", bits=4, methods=("weight_err", "fisher", "hessian_trace")
    )
    assert set(rep.scores) == {"weight_err"}
    assert "fisher" in rep.skipped and "hessian_trace" in rep.skipped


def test_evaluate_selection_keeping_everything_float_is_exact():
    m, rng = _mlp(17)
    x = {"x": _data(rng, (12,), 32)}
    groups = qs.as_groups(qs.find_sites(m))
    kept_all = qs.evaluate_selection(m, groups, list(groups), "weights", 3, x)
    assert (
        kept_all["kl"] == pytest.approx(0.0, abs=1e-12) and kept_all["agreement"] == 1.0
    )
    kept_none = qs.evaluate_selection(m, groups, [], "weights", 3, x)
    assert kept_none["kl"] > 0


# ---- the documentation example ------------------------------------------------------------------


def test_doc_example_runs_and_its_deterministic_line_is_quoted_correctly(capsys):
    """docs/quant-sensitivity.md must stay true: run its example and compare with what it quotes.

    The ``||W-Q(W)||`` ranking is deterministic and is checked exactly. The ``fisher`` ranking
    depends on seeded random labels and the measured KL on float rounding, so only their form
    is checked (and the numbers the page quotes are not asserted).
    """
    import os
    import re

    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "docs", "quant-sensitivity.md"
    )
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r"<!-- doctest -->\n```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) == 1
    exec(compile(blocks[0], "docs/quant-sensitivity.md", "exec"), {})
    out = capsys.readouterr().out
    quoted = re.search(
        r"```\n(most sensitive first \(fisher\).*?)```", text, flags=re.DOTALL
    )
    assert quoted, "the page no longer quotes the example's output"
    want = next(ln for ln in quoted.group(1).splitlines() if "||W-Q(W)||" in ln)
    assert want in out.splitlines()
    names = re.findall(
        r"'(MatMul_\d+)'", next(ln for ln in out.splitlines() if "(fisher)" in ln)
    )
    assert sorted(names) == ["MatMul_0", "MatMul_3", "MatMul_6"]
    assert re.search(r"keep in float \(k=1, fisher\): +\['MatMul_\d+'\]", out)
    kls = [
        float(v) for v in re.findall(r"'MatMul_\d+': ([0-9.]+)", out.splitlines()[-1])
    ]
    assert len(kls) == 3 and all(v > 0 for v in kls)
