"""Tests for onnxsim.stable_relu_prune: exact pruning of provably stable ReLU units.

The structure of the checks: (1) the counts of dead / always-on / unstable units equal an
INDEPENDENT numpy interval reference on networks whose stability is constructed through the
biases; (2) the pruned / merged model equals the original on many sampled inputs inside the
box (onnxruntime), changes shape as expected, and records its precondition; (3) outside the
box the pruned model is allowed to differ (and the test shows that it does); (4) the
structures it must NOT touch are reported with a reason and left byte-identical.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import ranges as R
from onnxsim import stable_relu_prune as S


def _model(body, initializer=None, opset=13, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _run(model, x, name="x"):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {name: x})[0]


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _inits(model):
    return {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}


# ---- an MLP whose stability is constructed through the biases ------------------------

_D, _H1, _H2, _O = 6, 12, 10, 4


def _mlp(seed=0):
    rng = np.random.default_rng(seed)
    w1 = _f32(rng.standard_normal((_D, _H1)) * 0.3)
    b1 = _f32(rng.standard_normal(_H1) * 0.1)
    b1[:4] = -10.0  # dead over [0, 1]^6
    b1[4:7] = 10.0  # always on
    w2 = _f32(rng.standard_normal((_H1, _H2)) * 0.3)
    b2 = _f32(rng.standard_normal(_H2) * 0.1)
    b2[:2] = -50.0
    w3 = _f32(rng.standard_normal((_H2, _O)) * 0.3)
    b3 = _f32(rng.standard_normal(_O))
    body = (
        f"m (float[N,{_D}] x) => (float[N,{_O}] y) "
        "{ a = MatMul(x, W1)  b = Add(a, B1)  r = Relu(b)  c = MatMul(r, W2)  d = Add(c, B2)  "
        "e = Relu(d)  y0 = MatMul(e, W3)  y = Add(y0, B3) }"
    )
    init = dict(W1=w1, B1=b1, W2=w2, B2=b2, W3=w3, B3=b3)
    return _model(body, init), init


_BOX = {"x": (0.0, 1.0)}


def _ibp_mlp(init, lo, hi):
    """Independent reference: plain interval arithmetic through the 3-layer MLP."""

    def affine(a, b_, w, bias):
        wp, wn = np.maximum(w, 0), np.minimum(w, 0)
        return a @ wp + b_ @ wn + bias, b_ @ wp + a @ wn + bias

    lo_v, hi_v = np.full(_D, lo), np.full(_D, hi)
    lo_v, hi_v = affine(lo_v, hi_v, init["W1"], init["B1"])
    pre1 = (lo_v.copy(), hi_v.copy())
    lo_v, hi_v = np.maximum(lo_v, 0), np.maximum(hi_v, 0)
    lo_v, hi_v = affine(lo_v, hi_v, init["W2"], init["B2"])
    return pre1, (lo_v, hi_v)


def test_counts_match_an_independent_interval_reference():
    m, init = _mlp()
    rep = S.analyze(m, _BOX, methods=("interval",))
    (l1, u1), (l2, u2) = _ibp_mlp(init, 0.0, 1.0)
    for layer, (lo, hi) in zip(rep.layers, ((l1, u1), (l2, u2))):
        assert set(layer.dead_units) == set(np.flatnonzero(hi <= -S.DEFAULT_MARGIN))
        assert set(layer.on_units) == set(np.flatnonzero(lo >= S.DEFAULT_MARGIN))
        assert layer.dead + layer.always_on + layer.unstable == layer.units
    assert [layer.prunable for layer in rep.layers] == [True, True]
    assert rep.methods == ["interval"]


def test_crown_proves_at_least_what_intervals_prove():
    m, _ = _mlp(3)
    rep = S.analyze(m, _BOX, methods=("interval", "crown"))
    d_i, o_i = rep.total_by_method()["interval"]
    d_c, o_c = rep.total_by_method()["crown"]
    assert d_c >= d_i and o_c >= o_i  # sound and never looser
    assert rep.total("dead") >= d_i


def test_zonotope_provider_runs_and_never_proves_less_than_intervals():
    m, _ = _mlp(4)
    rep = S.analyze(m, _BOX, methods=("interval", "zonotope"))
    assert "zonotope" in rep.methods
    assert rep.total_by_method()["zonotope"][0] >= rep.total_by_method()["interval"][0]


def test_provider_over_its_budget_is_skipped_with_a_reason_not_hidden():
    m, _ = _mlp()
    rep = S.analyze(m, _BOX, methods=("interval", "crown"), max_elements={"crown": 1})
    assert "crown" in rep.skipped_methods and "budget" in rep.skipped_methods["crown"]
    assert rep.methods == ["interval"]
    assert "skipped" in str(rep)


# ---- apply: exact inside the box ------------------------------------------------------


def test_apply_is_exact_inside_the_box_and_removes_the_units():
    m, init = _mlp()
    before = onnx.ModelProto()
    before.CopyFrom(m)
    pruned, rep = S.apply(m, _BOX)
    assert m == before  # the input model is untouched
    assert rep.applied and rep.removed_units == rep.total("dead") > 0
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1, (500, _D)).astype(np.float32)
    assert np.abs(_run(m, xs) - _run(pruned, xs)).max() < 1e-5
    assert rep.verified_max_abs_diff is not None and rep.verified_max_abs_diff < 1e-4
    new = _inits(pruned)
    assert new["W1"].shape[1] == _H1 - rep.layers[0].dead
    assert new["W2"].shape == (_H1 - rep.layers[0].dead, _H2 - rep.layers[1].dead)
    assert new["W3"].shape[0] == _H2 - rep.layers[1].dead
    assert [o.name for o in pruned.graph.output] == ["y"]
    assert rep.total("params_removable") > 0 and rep.total("macs_removable") > 0
    # the COMBINED figures are exact; the per-layer ones are standalone and overlap on W2
    saved = sum(v.size for v in init.values()) - sum(v.size for v in new.values())
    assert saved == rep.params_removed
    assert (
        rep.total("params_removable")
        == rep.params_removed + rep.layers[0].dead * rep.layers[1].dead
    )
    dims_before = [_D, _H1, _H2, _O]
    dims_after = [_D, _H1 - rep.layers[0].dead, _H2 - rep.layers[1].dead, _O]

    def macs(d):  # batch 1, one MAC per weight
        return sum(d[i] * d[i + 1] for i in range(3))

    assert macs(dims_before) - macs(dims_after) == rep.macs_removed


def test_the_result_is_only_exact_inside_the_box_and_says_so():
    m, _ = _mlp()
    pruned, _ = S.apply(m, _BOX)
    rng = np.random.default_rng(1)
    outside = rng.uniform(-30, 30, (300, _D)).astype(np.float32)
    assert np.abs(_run(m, outside) - _run(pruned, outside)).max() > 0.1
    meta = {p.key: p.value for p in pruned.metadata_props}
    assert "onnxsim.precondition.range.x" in meta
    assert "ONLY for inputs inside" in meta["onnxsim.precondition.note"]
    import json

    box = json.loads(meta["onnxsim.precondition.range.x"])
    assert box == {"min": 0.0, "max": 1.0}


def test_refuses_without_a_finite_input_range_and_accepts_an_annotation():
    m, _ = _mlp()
    for fn in (S.analyze, S.apply):
        with pytest.raises(ValueError, match="no finite input range"):
            fn(m)
    R.set_range(m, "x", 0.0, 1.0)
    pruned, rep = S.apply(m)
    assert rep.removed_units > 0
    # the contract annotation survives next to the precondition
    assert "x" in R.get_ranges(pruned)


def test_a_unit_barely_below_zero_is_not_called_dead():
    # pre-activation upper bound is exactly -1e-7: inside the float32 margin, so not provable
    w = _f32([[1.0, 1.0]])
    b = _f32([-1.0 - 1e-7, -5.0])
    m = _model(
        "m (float[N,1] x) => (float[N,2] y) { a = MatMul(x, W)  c = Add(a, B)  y = Relu(c) }",
        dict(W=w, B=b),
    )
    rep = S.analyze(m, {"x": (0.0, 1.0)}, methods=("interval",))
    assert rep.layers[0].dead == 1  # only the -5 unit; the one at ~-1e-7 stays
    rep2 = S.analyze(m, {"x": (0.0, 1.0)}, methods=("interval",), margin=0.0)
    assert rep2.layers[0].dead == 2


def test_an_all_dead_layer_keeps_one_unit_and_is_still_exact():
    w1 = _f32(np.ones((2, 3)))
    b1 = _f32([-9.0, -9.0, -9.0])
    w2 = _f32(np.arange(6).reshape(3, 2))
    b2 = _f32([0.5, -0.25])
    m = _model(
        "m (float[N,2] x) => (float[N,2] y) { a = MatMul(x, W1)  b = Add(a, B1)  r = Relu(b)  c = MatMul(r, W2)  y = Add(c, B2) }",
        dict(W1=w1, B1=b1, W2=w2, B2=b2),
    )
    pruned, rep = S.apply(m, {"x": (0.0, 1.0)})
    assert rep.layers[0].dead == 3 and rep.removed_units == 2
    assert "constant" in rep.layers[0].note
    xs = np.random.default_rng(0).uniform(0, 1, (50, 2)).astype(np.float32)
    assert np.abs(_run(m, xs) - _run(pruned, xs)).max() < 1e-6


def test_remove_dead_false_leaves_the_model_unchanged():
    m, _ = _mlp()
    pruned, rep = S.apply(m, _BOX, remove_dead=False)
    assert rep.removed_units == 0
    assert _inits(pruned).keys() == _inits(m).keys()
    for k, v in _inits(m).items():
        assert np.array_equal(_inits(pruned)[k], v)


# ---- Gemm / BatchNormalization / dynamic batch -----------------------------------------


def test_gemm_with_transB_and_batchnorm_are_pruned_exactly():
    rng = np.random.default_rng(5)
    w1 = _f32(rng.standard_normal((8, 4)) * 0.2)  # transB=1: [out, in]
    c1 = _f32(rng.standard_normal(8) * 0.1)
    c1[:3] = -8.0
    scale = _f32(rng.uniform(0.5, 1.5, 8))
    bias = _f32(rng.standard_normal(8) * 0.1)
    mean = _f32(rng.standard_normal(8) * 0.1)
    var = _f32(rng.uniform(0.5, 1.5, 8))
    w2 = _f32(rng.standard_normal((3, 8)) * 0.3)  # transB=1: [out, in]
    c2 = _f32(rng.standard_normal(3))
    m = _model(
        "m (float[N,4] x) => (float[N,3] y) "
        "{ a = Gemm<transB=1>(x, W1, C1)  n = BatchNormalization<epsilon=1e-5>(a, S, Bb, Mu, Va)  "
        "r = Relu(n)  y = Gemm<transB=1>(r, W2, C2) }",
        dict(W1=w1, C1=c1, S=scale, Bb=bias, Mu=mean, Va=var, W2=w2, C2=c2),
        opset=15,
    )
    pruned, rep = S.apply(m, {"x": (0.0, 1.0)})
    d = rep.layers[0].dead
    assert d >= 3 and rep.layers[0].prunable
    new = _inits(pruned)
    assert new["W1"].shape == (8 - d, 4) and new["S"].shape == (8 - d,)
    assert new["W2"].shape == (3, 8 - d)
    xs = np.random.default_rng(0).uniform(0, 1, (200, 4)).astype(np.float32)
    assert np.abs(_run(m, xs) - _run(pruned, xs)).max() < 1e-5


# ---- conv -------------------------------------------------------------------------------


def _conv_net(seed=0, pool=False, group=1):
    rng = np.random.default_rng(seed)
    w1 = _f32(rng.standard_normal((6, 2, 3, 3)) * 0.2)
    b1 = _f32(rng.standard_normal(6) * 0.05)
    b1[:3] = -20.0  # three dead channels over [0,1]
    w2 = _f32(rng.standard_normal((4, 6 // group, 3, 3)) * 0.2)
    b2 = _f32(rng.standard_normal(4) * 0.1)
    mid = "p = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r)  " if pool else ""
    src = "p" if pool else "r"
    body = (
        "m (float[1,2,8,8] x) => (float[1,4,?,?] y) "
        f"{{ a = Conv<pads=[1,1,1,1]>(x, W1, B1)  r = Relu(a)  {mid}y = Conv<pads=[1,1,1,1], group={group}>({src}, W2, B2) }}"
    )
    return _model(body, dict(W1=w1, B1=b1, W2=w2, B2=b2))


@pytest.mark.parametrize("pool", [False, True])
def test_conv_relu_conv_prunes_dead_channels_exactly(pool):
    m = _conv_net(pool=pool)
    pruned, rep = S.apply(m, {"x": (0.0, 1.0)})
    lay = rep.layers[0]
    assert lay.prunable and lay.dead >= 3 and lay.macs_removable > 0
    new = _inits(pruned)
    assert new["W1"].shape[0] == 6 - lay.dead and new["W2"].shape[1] == 6 - lay.dead
    xs = np.random.default_rng(0).uniform(0, 1, (16, 1, 2, 8, 8)).astype(np.float32)
    for x in xs:
        assert np.abs(_run(m, x) - _run(pruned, x)).max() < 1e-5


def test_conv_flatten_gemm_maps_channels_to_all_their_columns():
    rng = np.random.default_rng(2)
    w1 = _f32(rng.standard_normal((4, 1, 3, 3)) * 0.2)
    b1 = _f32([-20.0, 0.1, -20.0, 0.2])
    # conv output [1,4,4,4] -> flatten -> 64 features; Gemm weight [64, 3]
    wg = _f32(rng.standard_normal((64, 3)) * 0.2)
    bg = _f32(rng.standard_normal(3))
    m = _model(
        "m (float[1,1,4,4] x) => (float[1,3] y) "
        "{ a = Conv<pads=[1,1,1,1]>(x, W1, B1)  r = Relu(a)  f = Flatten(r)  y = Gemm(f, Wg, Bg) }",
        dict(W1=w1, B1=b1, Wg=wg, Bg=bg),
    )
    pruned, rep = S.apply(m, {"x": (0.0, 1.0)})
    assert rep.layers[0].dead == 2
    assert _inits(pruned)["Wg"].shape == (32, 3)  # 2 channels x 16 positions dropped
    for x in (
        np.random.default_rng(0).uniform(0, 1, (20, 1, 1, 4, 4)).astype(np.float32)
    ):
        assert np.abs(_run(m, x) - _run(pruned, x)).max() < 1e-5


# ---- structures that must be left alone, with a reason ----------------------------------


def test_residual_fan_out_is_counted_but_not_pruned():
    w = _f32(np.eye(3))
    b = _f32([-9.0, 0.5, -9.0])
    m = _model(
        "m (float[N,3] x) => (float[N,3] y) { a = MatMul(x, W)  c = Add(a, B)  r = Relu(c)  y = Add(r, x) }",
        dict(W=w, B=b),
    )
    before = onnx.ModelProto()
    before.CopyFrom(m)
    pruned, rep = S.apply(m, {"x": (0.0, 1.0)})
    lay = rep.layers[0]
    assert lay.dead == 2 and not lay.prunable and "consumer" in lay.reason
    assert (
        rep.removed_units == 0 and pruned.graph.initializer == before.graph.initializer
    )


def test_relu_that_is_a_graph_output_shared_or_grouped_weights_are_reported():
    # graph output
    m = _model(
        "m (float[N,2] x) => (float[N,2] y) { a = MatMul(x, W)  y = Relu(a) }",
        dict(W=_f32(np.eye(2))),
    )
    assert "graph output" in S.analyze(m, {"x": (0.0, 1.0)}).layers[0].reason
    # shared weight: the same initializer feeds two MatMuls
    m = _model(
        "m (float[N,2] x) => (float[N,2] y) { a = MatMul(x, W)  r = Relu(a)  y = MatMul(r, W) }",
        dict(W=_f32(np.eye(2))),
    )
    assert "unshared" in S.analyze(m, {"x": (0.0, 1.0)}).layers[0].reason
    # depthwise / grouped consumer
    g = _conv_net(group=2)
    assert "grouped" in S.analyze(g, {"x": (0.0, 1.0)}).layers[0].reason


# ---- merge_active -----------------------------------------------------------------------


def _all_on_mlp(gemm=False):
    rng = np.random.default_rng(7)
    w1 = _f32(rng.standard_normal((4, 5)) * 0.2)
    b1 = _f32(np.full(5, 20.0))  # always on
    w2 = _f32(rng.standard_normal((5, 3)) * 0.3)
    b2 = _f32(rng.standard_normal(3))
    if gemm:
        body = (
            "m (float[N,4] x) => (float[N,3] y) "
            "{ a = Gemm<alpha=0.5, beta=2.0>(x, W1, B1)  r = Relu(a)  y = Gemm<alpha=2.0, beta=0.25>(r, W2, B2) }"
        )
    else:
        body = (
            "m (float[N,4] x) => (float[N,3] y) "
            "{ a = MatMul(x, W1)  c = Add(a, B1)  r = Relu(c)  d = MatMul(r, W2)  y = Add(d, B2) }"
        )
    return _model(body, dict(W1=w1, B1=b1, W2=w2, B2=b2))


@pytest.mark.parametrize("gemm", [False, True])
def test_merge_active_fuses_an_all_on_layer_exactly(gemm):
    m = _all_on_mlp(gemm)
    merged, rep = S.apply(m, {"x": (0.0, 1.0)}, merge_active=True)
    assert rep.merged_layers == 1
    ops = [n.op_type for n in merged.graph.node]
    assert "Relu" not in ops and ops.count("Gemm") + ops.count("MatMul") == 1 + (
        0 if gemm else 0
    )
    xs = np.random.default_rng(0).uniform(0, 1, (300, 4)).astype(np.float32)
    assert np.abs(_run(m, xs) - _run(merged, xs)).max() < 1e-4
    assert rep.verified_max_abs_diff is not None and rep.verified_max_abs_diff < 1e-4


def test_merge_is_off_by_default_and_skips_partially_active_layers():
    m = _all_on_mlp()
    same, rep = S.apply(m, {"x": (0.0, 1.0)})
    assert rep.merged_layers == 0 and "Relu" in [n.op_type for n in same.graph.node]
    mixed, _ = _mlp()
    out, rep2 = S.apply(mixed, _BOX, merge_active=True)
    assert rep2.merged_layers == 0  # layers keep unstable units: no merge, only pruning
    assert rep2.removed_units > 0


def test_prune_then_merge_collapses_a_layer_whose_remaining_units_are_all_on():
    rng = np.random.default_rng(9)
    b1 = _f32(np.concatenate([np.full(3, -30.0), np.full(4, 30.0)]))  # dead + on only
    w1 = _f32(rng.standard_normal((3, 7)) * 0.2)
    w2 = _f32(rng.standard_normal((7, 2)) * 0.3)
    b2 = _f32([0.1, -0.2])
    m = _model(
        "m (float[N,3] x) => (float[N,2] y) "
        "{ a = MatMul(x, W1)  c = Add(a, B1)  r = Relu(c)  d = MatMul(r, W2)  y = Add(d, B2) }",
        dict(W1=w1, B1=b1, W2=w2, B2=b2),
    )
    out, rep = S.apply(m, {"x": (0.0, 1.0)}, merge_active=True)
    assert rep.removed_units == 3 and rep.merged_layers == 1
    assert "Relu" not in [n.op_type for n in out.graph.node]
    xs = np.random.default_rng(0).uniform(0, 1, (200, 3)).astype(np.float32)
    assert np.abs(_run(m, xs) - _run(out, xs)).max() < 1e-4


# ---- an independent proof -----------------------------------------------------------------


def test_certify_proves_the_pruned_model_equal_over_the_box():
    pytest.importorskip("z3")
    from onnxsim import certify

    rng = np.random.default_rng(11)
    w1 = _f32(rng.standard_normal((2, 5)) * 0.3)
    b1 = _f32([-9.0, 0.2, -9.0, 0.1, 9.0])
    w2 = _f32(rng.standard_normal((5, 2)) * 0.3)
    b2 = _f32([0.1, 0.2])
    m = _model(
        "m (float[1,2] x) => (float[1,2] y) "
        "{ a = MatMul(x, W1)  c = Add(a, B1)  r = Relu(c)  d = MatMul(r, W2)  y = Add(d, B2) }",
        dict(W1=w1, B1=b1, W2=w2, B2=b2),
    )
    pruned, rep = S.apply(m, {"x": (0.0, 1.0)})
    assert rep.removed_units == 2
    verdict = certify.certify(
        m,
        pruned,
        input_ranges={"x": (0.0, 1.0)},
        timeout_ms=20000,
        total_timeout_ms=60000,
    )
    assert verdict.ok, str(verdict)
    # and the same claim over a WIDER box is not provable: the pruning is conditional
    wide = certify.certify(
        m,
        pruned,
        input_ranges={"x": (-50.0, 50.0)},
        timeout_ms=20000,
        total_timeout_ms=60000,
    )
    assert not wide.ok


def test_self_check_raises_on_a_pruning_that_is_not_exact(monkeypatch):
    m, _ = _mlp()
    # sabotage the analysis: claim every unit is dead, so the pruned model must disagree
    real = S.analyze

    def lying(*a, **k):
        rep = real(*a, **k)
        for lay in rep.layers:
            lay.dead_units = np.arange(lay.units)
            lay.dead = lay.units
        return rep

    monkeypatch.setattr(S, "analyze", lying)
    with pytest.raises(RuntimeError, match="self-check failed"):
        S.apply(m, _BOX)


def test_report_text_lists_layers_methods_and_reasons():
    m, _ = _mlp()
    text = str(S.analyze(m, _BOX))
    assert "total" in text and "interval" in text and "crown" in text


def test_prune_units_matches_apply_for_the_same_units_and_is_marked_unchecked():
    m, _ = _mlp()
    rep = S.analyze(m, _BOX)
    chosen = {layer.relu: layer.dead_units for layer in rep.layers}
    manual, removed, skipped = S.prune_units(m, chosen)
    auto, arep = S.apply(m, _BOX, verify_samples=0)
    assert removed == arep.removed_units and skipped == {}
    for k, v in _inits(auto).items():
        assert np.array_equal(_inits(manual)[k], v)
    note = next(
        p.value for p in manual.metadata_props if p.key == "onnxsim.precondition.note"
    )
    assert "NOT proved" in note and "onnxsim.precondition.range.x" not in {
        p.key for p in manual.metadata_props
    }


def test_prune_units_reports_bad_requests_instead_of_guessing():
    m, _ = _mlp()
    out, removed, skipped = S.prune_units(m, {"nope": [0], "r": [999]})
    assert removed == 0
    assert skipped["nope"] == "no such ReLU" and "out of range" in skipped["r"]
    res = _model(
        "m (float[N,3] x) => (float[N,3] y) { a = MatMul(x, W)  c = Add(a, B)  r = Relu(c)  y = Add(r, x) }",
        dict(W=_f32(np.eye(3)), B=_f32([0.0, 0.0, 0.0])),
    )
    _, removed, skipped = S.prune_units(res, {"r": [0]})
    assert removed == 0 and "unsupported consumer Add" in skipped["r"]


# ---- the doc must stay true ---------------------------------------------------------------


def test_the_doc_example_prints_what_the_doc_says(capsys):
    import pathlib
    import re

    doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "stable-relu-prune.md"
    blocks = re.findall(
        r"<!-- doctest -->\n```python\n(.*?)```\n```text\n(.*?)```",
        doc.read_text(),
        re.DOTALL,
    )
    assert len(blocks) == 1
    code, expected = blocks[0]
    exec(compile(code, doc.name, "exec"), {"__name__": "__doc__"})
    assert capsys.readouterr().out.strip() == expected.strip()
