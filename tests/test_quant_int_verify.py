"""Tests for onnxsim.quant_int_verify (integer-pipeline verifier for quantized graphs).

The strongest tests are the exactness ones: the closed-form accumulator range against
brute-force enumeration, the tie-window set against exhaustive evaluation of *every*
accumulator, and counterexamples replayed on onnxruntime at the exact reduction depth where
int32 first overflows. Z3 is used only in small, bounded checks, each with a timeout and a
fresh context (see the module docstring for why).
"""

import itertools
from fractions import Fraction

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import quant_int_verify as Q

INT32_MAX = Q.INT32_MAX

# Some CPUs' onnxruntime u8 x s8 kernels sum adjacent products in saturating int16 (AVX2
# vpmaddubsw without VNNI). On such a host onnxruntime is NOT an oracle for exact int32
# arithmetic: CI's runner returned 1086422270 for the K=66311 extreme case where exact
# two's-complement arithmetic gives 2147481735 (about half: 33155 pair sums saturated at
# 32767). The verifier models exact int32 semantics and flags the hazard via
# probe_u8s8_saturation(), so the tests that compare it with onnxruntime's output must not
# run there; they still run on every host whose kernel is exact.
requires_exact_u8s8 = pytest.mark.skipif(
    Q.probe_u8s8_saturation() is True,
    reason="this host's onnxruntime u8xs8 kernel saturates int16 pair sums, so it is not an "
    "oracle for exact int32 arithmetic (see quant_int_verify.probe_u8s8_saturation)",
)


def _model(body, initializer=None, opset=13, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return model


def _matmul_integer(k, w, za=0, a_type="uint8"):
    n = w.shape[1]
    azp = np.array(za, dtype=np.uint8 if a_type == "uint8" else np.int8)
    return _model(
        f"m ({a_type}[1,{k}] a) => (int32[1,{n}] y) {{ y = MatMulInteger(a, W, azp) }}",
        dict(W=w.astype(np.int8), azp=azp),
    )


def _only(model):
    consts = Q._consts(model)
    node = next(
        n
        for n in model.graph.node
        if n.op_type in ("MatMulInteger", "ConvInteger", "QLinearMatMul", "QLinearConv")
    )
    return node, consts


def _ort(model, feeds):
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feeds)


# ---- exact arithmetic ---------------------------------------------------------


def test_accumulator_range_is_exact_against_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        k = int(rng.integers(1, 5))
        w = rng.integers(-6, 7, size=k)
        d_lo, d_hi = sorted(int(v) for v in rng.integers(-4, 5, size=2))
        bias = int(rng.integers(-5, 6))
        vals = [
            bias + int(np.dot(w, d))
            for d in itertools.product(range(d_lo, d_hi + 1), repeat=k)
        ]
        assert Q.accumulator_range(w, d_lo, d_hi, bias) == (min(vals), max(vals))


def test_rne_and_fp32_requant_match_numpy_and_exact_reference():
    assert [Q.rne_fraction(Fraction(v, 2)) for v in (-5, -3, -1, 1, 3, 5, 7)] == [
        -2,
        -2,
        0,
        0,
        2,
        2,
        4,
    ]
    m_real, m32 = Q.requant_multiplier(
        np.float32(0.0173), np.float32(0.0091), np.float32(0.05)
    )
    assert m32 == np.float32(np.float32(0.0173) * np.float32(0.0091)) / np.float32(0.05)
    assert Q.real_requant(0, m_real, 130, 0, 255) == 130


@pytest.mark.parametrize(
    "sa,sb,sy,zy,dtype",
    [
        (0.0173, 0.0091, 0.05, 130, "u8"),
        (0.02, 0.013, 0.021, 7, "u8"),
        (0.011, 0.007, 0.0093, 0, "i8"),
        (0.1, 0.1, 0.01, 128, "u8"),
    ],
)
def test_tie_windows_are_complete_against_exhaustive_evaluation(sa, sb, sy, zy, dtype):
    qmin, qmax = (0, 255) if dtype == "u8" else (-128, 127)
    zy = zy if dtype == "u8" else 0
    m_real, m32 = Q.requant_multiplier(np.float32(sa), np.float32(sb), np.float32(sy))
    lo, hi = -60000, 60000
    brute = [
        a
        for a in range(lo, hi + 1)
        if Q.fp32_requant(a, m32, zy, qmin, qmax)
        != Q.real_requant(a, m_real, zy, qmin, qmax)
    ]
    assert Q.requant_differences(m_real, m32, zy, qmin, qmax, lo, hi) == brute
    # and the 1-LSB claim: every difference is exactly one step
    assert all(
        abs(
            Q.fp32_requant(a, m32, zy, qmin, qmax)
            - Q.real_requant(a, m_real, zy, qmin, qmax)
        )
        == 1
        for a in brute
    )


def test_requant_differences_gives_up_instead_of_guessing():
    m_real, m32 = Q.requant_multiplier(
        np.float32(1e-9), np.float32(1.0), np.float32(1.0)
    )
    assert (
        Q.requant_differences(
            m_real, m32, 0, 0, 255, -(2**31), 2**31 - 1, max_candidates=10
        )
        is None
    )


def test_dynamic_rel_error_bound_dominates_sampled_error():
    rng = np.random.default_rng(1)
    for acc_max in (1000, 2**24, 2**30):
        bound = Q.dynamic_rel_error_bound(acc_max)
        worst = Fraction(0)
        for _ in range(400):
            acc = int(rng.integers(-acc_max, acc_max + 1)) or 1
            xs, ws = np.float32(rng.uniform(1e-3, 1)), np.float32(rng.uniform(1e-3, 1))
            got = np.float32(np.float32(acc) * np.float32(xs * ws))
            exact = Fraction(acc) * Fraction(float(xs)) * Fraction(float(ws))
            worst = max(worst, abs(Fraction(float(got)) - exact) / abs(exact))
        assert worst <= bound
    assert Q.dynamic_rel_error_bound(1000) < Q.dynamic_rel_error_bound(
        2**30
    )  # exact cast helps


# ---- no-wrap: the headline result ---------------------------------------------


@requires_exact_u8s8
def test_no_wrap_boundary_is_exact_and_counterexample_replays_on_onnxruntime():
    k_safe = INT32_MAX // (
        255 * 127
    )  # the largest uint8 x int8 reduction that cannot wrap
    assert k_safe * 255 * 127 <= INT32_MAX < (k_safe + 1) * 255 * 127
    safe = _matmul_integer(k_safe, np.full((k_safe, 1), 127))
    node, consts = _only(safe)
    rep = Q.verify_layer(node, consts)
    assert rep.proved and rep.of("no-wrap")[0].verdict == Q.PROVED
    # the proved bound really is the maximum: run the extreme input, ORT agrees with exact arithmetic
    ext = np.full((1, k_safe), 255, np.uint8)
    assert int(_ort(safe, {"a": ext})[0][0, 0]) == k_safe * 255 * 127

    bad = _matmul_integer(k_safe + 1, np.full((k_safe + 1, 1), 127))
    node, consts = _only(bad)
    rep = Q.verify_layer(node, consts)
    f = rep.of("no-wrap")[0]
    assert f.verdict == Q.REFUTED and not rep.sound
    observed, exact = Q.replay_no_wrap(bad, rep.name, f)
    assert int(exact[0, 0]) == (k_safe + 1) * 255 * 127 > INT32_MAX
    assert int(observed[0, 0]) != int(exact[0, 0])  # the runtime wrapped
    assert (
        int(observed[0, 0]) == ((int(exact[0, 0]) - 2**31) % 2**32) - 2**31
    )  # exactly two's complement


def test_narrower_activation_range_proves_what_the_full_range_cannot():
    k = 70000
    bad = _matmul_integer(k, np.full((k, 1), 127))
    node, consts = _only(bad)
    assert Q.verify_layer(node, consts).of("no-wrap")[0].verdict == Q.REFUTED
    rep = Q.verify_layer(node, consts, act_range=(0, 200))  # 70000*200*127 < 2**31
    assert rep.of("no-wrap")[0].verdict == Q.PROVED
    assert "restricted" in rep.of("note")[0].detail


def test_signed_activation_with_zero_point():
    k = 40000
    w = np.full((k, 1), 127)
    m = _model(
        f"m (int8[1,{k}] a) => (int32[1,1] y) {{ y = MatMulInteger(a, W, azp) }}",
        dict(W=w.astype(np.int8), azp=np.array(-128, np.int8)),
    )
    node, consts = _only(m)
    rep = Q.verify_layer(node, consts)
    # d in [0, 255] -> same 255*127 per tap as uint8: 40000*255*127 = 1.295e9, safe
    assert rep.of("no-wrap")[0].verdict == Q.PROVED
    assert "[0, 1295400000]" in rep.of("no-wrap")[0].detail


def test_mixed_sign_weights_reach_both_extremes():
    w = np.array([[100], [-100], [100], [-100]] * 5000, dtype=np.int8)
    m = _matmul_integer(len(w), w, za=128)
    node, consts = _only(m)
    layer, _ = Q._extract_layer(node, consts, {}, {}, None)
    lo, hi = Q.accumulator_range(layer.wq[0], layer.d_lo, layer.d_hi)
    assert (lo, hi) == (
        -10000 * 100 * 127 - 10000 * 100 * 128,
        10000 * 100 * 127 + 10000 * 100 * 128,
    )
    assert (
        Q.verify_layer(node, consts).of("no-wrap")[0].verdict == Q.PROVED
    )  # |acc| <= ~2.5e8, far below 2**31


def test_z3_agrees_with_closed_form_and_catches_a_wrap():
    pytest.importorskip("z3")
    assert Q.z3_no_wrap([3, -2, 5], 0, 255)[0] == Q.PROVED
    verdict, d = Q.z3_no_wrap(
        [2**30, 2**30, 2**30], 0, 3
    )  # force a wrap with absurd weights
    assert verdict == Q.REFUTED and d is not None
    assert sum(wk * dk for wk, dk in zip([2**30] * 3, d)) > INT32_MAX
    # tiny layer through verify_layer: closed form and Z3 are cross-checked, no inconsistency
    rep = Q.verify_layer(
        *_only(_matmul_integer(8, np.arange(-4, 4).reshape(8, 1) * 30))
    )
    assert (
        not rep.of("internal-inconsistency")
        and rep.of("no-wrap-z3")[0].verdict == Q.PROVED
    )


def test_disagreement_between_routes_is_reported_not_swallowed(monkeypatch):
    pytest.importorskip("z3")
    monkeypatch.setattr(
        Q, "z3_no_wrap", lambda *a, **k: (Q.REFUTED, [0])
    )  # a lying solver
    rep = Q.verify_layer(*_only(_matmul_integer(4, np.ones((4, 1)))))
    assert rep.of("internal-inconsistency")[0].verdict == Q.REFUTED and not rep.sound


def test_reachability_witnesses_are_verified_and_unreachable_targets_are_rejected():
    w = np.array([3, 5])
    # greedy finds it constructively, and the witness really hits the target
    d = Q._greedy_reach(w, 0, 10, 13)
    assert d is not None and int(w @ d) == 13 and d.min() >= 0 and d.max() <= 10
    # a target outside the exact range is unreachable without any search
    assert Q._reach(w, 0, 10, 81, 0, 24, 5000)[0] == "unreachable"
    # inside the range but not representable: 3*d0 + 5*d1 with d in 0..2 cannot make 2 (min 0, then 3, 5, 6, ...)
    pytest.importorskip("z3")
    assert Q.z3_reach([3, 5], 0, 2, 2)[0] == "unreachable"
    status, d = Q.z3_reach([3, 5], 0, 2, 11)
    assert status == "reachable" and 3 * d[0] + 5 * d[1] == 11
    # the combined entry point agrees, and returns a full-length witness
    status, full = Q._reach(np.array([3, 0, 5]), 0, 2, 11, 0, 24, 5000)
    assert status == "reachable" and int(np.array([3, 0, 5]) @ full) == 11


def test_z3_timeout_is_a_skip_not_a_hang():
    pytest.importorskip("z3")
    verdict, _ = Q.z3_no_wrap(list(range(1, 200)), 0, 255, timeout_ms=1)
    assert verdict in (Q.PROVED, Q.REFUTED, Q.SKIPPED)  # bounded either way


# ---- convolution ----------------------------------------------------------------


def _conv_integer(cin, cout, k, wval, za=0, pads=(0, 0, 0, 0), x_hw=None, group=1):
    h, w_ = x_hw if x_hw else (k, k)
    wq = np.full((cout, cin // group, k, k), wval, dtype=np.int8)
    return _model(
        f"m (uint8[1,{cin},{h},{w_}] x) => (int32[1,{cout},?,?] y) "
        f"{{ y = ConvInteger<pads=[{','.join(map(str, pads))}], group={group}>(x, W, azp) }}",
        dict(W=wq, azp=np.array(za, np.uint8)),
    )


def test_conv_overflow_is_refuted_and_replays_on_onnxruntime():
    cin = 7400  # 7400*9 = 66600 taps > 66310
    m = _conv_integer(cin, 1, 3, 127)
    node, consts = _only(m)
    rep = Q.verify_layer(node, consts, x_shape=(1, cin, 3, 3))
    f = rep.of("no-wrap")[0]
    assert f.verdict == Q.REFUTED
    observed, exact = Q.replay_no_wrap(m, rep.name, f)
    assert int(exact.reshape(-1)[0]) == 66600 * 255 * 127 and int(
        observed.reshape(-1)[0]
    ) != int(exact.reshape(-1)[0])


def test_conv_below_the_boundary_is_proved_and_matches_exact_arithmetic():
    m = _conv_integer(3, 4, 3, 100, za=17)
    node, consts = _only(m)
    rep = Q.verify_layer(node, consts, x_shape=(1, 3, 3, 3))
    assert rep.proved
    rng = np.random.default_rng(2)
    x = rng.integers(0, 256, (1, 3, 3, 3), dtype=np.uint8)
    got = _ort(m, {"x": x})[0]
    assert np.array_equal(
        got.astype(np.int64),
        Q._exact_conv(
            x.astype(np.int64) - 17, consts["W"].astype(np.int64), Q._attrs(node)
        ),
    )


def test_conv_with_padding_is_conservative_without_geometry():
    cin = 7400
    m = _conv_integer(cin, 1, 3, 127, pads=(1, 1, 1, 1), x_hw=(3, 3))
    node, consts = _only(m)
    rep = Q.verify_layer(
        node, consts
    )  # no shape -> cannot confirm the padded bound is attained
    assert (
        rep.of("no-wrap")[0].verdict == Q.SKIPPED and rep.sound
    )  # nothing claimed either way
    rep = Q.verify_layer(
        node, consts, x_shape=(1, cin, 5, 5)
    )  # a fully interior position exists
    assert rep.of("no-wrap")[0].verdict == Q.REFUTED
    obs, exact = Q.replay_no_wrap(m, rep.name, rep.of("no-wrap")[0])
    assert int(exact.max()) == 66600 * 255 * 127 and int(obs.max()) != int(exact.max())


def test_grouped_conv_counts_only_its_own_group():
    m = _conv_integer(
        4, 2, 3, 127, group=2
    )  # each output channel sees 2 input channels = 18 taps
    node, consts = _only(m)
    layer, why = Q._extract_layer(node, consts, {}, {}, None)
    assert why is None and layer.wq.shape == (2, 2 * 3 * 3)


# ---- QLinear* ---------------------------------------------------------------------


def _qlinear_matmul(k, n, w, sa, sb, sy, za, zy, y_type="uint8"):
    yt = np.uint8 if y_type == "uint8" else np.int8
    return _model(
        f"m (uint8[1,{k}] a) => ({y_type}[1,{n}] y) {{ y = QLinearMatMul(a, sa, za, W, sb, zb, sy, zy) }}",
        dict(
            W=w.astype(np.int8),
            sa=np.array(sa, np.float32),
            sb=np.array(sb, np.float32),
            sy=np.array(sy, np.float32),
            za=np.array(za, np.uint8),
            zb=np.array(0, np.int8),
            zy=np.array(zy, yt),
        ),
    )


def test_qlinear_matmul_hazards_match_onnxruntime_exactly():
    # K = 1, weight = 1  =>  accumulator = a - za for a in 0..255: the whole function is observable.
    sa, sb, sy, za, zy = 0.0173, 0.0091, 0.0013, 100, 0
    m = _qlinear_matmul(1, 1, np.array([[1]]), sa, sb, sy, za, zy)
    got = np.array(
        [int(_ort(m, {"a": np.array([[a]], np.uint8)})[0][0, 0]) for a in range(256)]
    )
    m_real, m32 = Q.requant_multiplier(np.float32(sa), np.float32(sb), np.float32(sy))
    fp32 = np.array([Q.fp32_requant(a - za, m32, zy, 0, 255) for a in range(256)])
    ideal = np.array([Q.real_requant(a - za, m_real, zy, 0, 255) for a in range(256)])
    assert np.array_equal(got, fp32)  # the runtime IS the fp32 pipeline we model
    node, consts = _only(m)
    rep = Q.verify_layer(node, consts)
    assert rep.of("requant-1lsb")[0].verdict == Q.PROVED and rep.proved
    hz = rep.of("requant-bit-exact")[0]
    assert (hz.verdict == Q.REFUTED) == bool(
        (fp32 != ideal).any()
    )  # reported iff a real difference exists
    assert int(np.abs(fp32 - ideal).max()) <= 1


def test_qlinear_matmul_with_no_hazard_is_proved_bit_exact():
    m = _qlinear_matmul(
        1, 1, np.array([[1]]), 0.5, 0.25, 0.25, 0, 0
    )  # multiplier exactly 0.5, fp32-exact
    rep = Q.verify_layer(*_only(m))
    assert rep.of("requant-bit-exact")[0].verdict == Q.PROVED


def test_z3_floating_point_encoding_matches_runtime_requantisation():
    pytest.importorskip("z3")
    m_real, m32 = Q.requant_multiplier(
        np.float32(0.0173), np.float32(0.0091), np.float32(0.0013)
    )
    rng = np.random.default_rng(3)
    for acc in [
        0,
        1,
        -1,
        127,
        -128,
        24000,
        -24000,
        2**24 + 1,
        *map(int, rng.integers(-30000, 30000, 12)),
    ]:
        assert Q.z3_eval_requant(acc, m32, 130, 0, 255) == Q.fp32_requant(
            acc, m32, 130, 0, 255
        )


def test_qlinear_requant_findings_on_signed_output_and_overflow_interplay():
    m = _qlinear_matmul(
        70000, 1, np.full((70000, 1), 127), 0.01, 0.01, 40.0, 0, 0, y_type="int8"
    )
    rep = Q.verify_layer(*_only(m))
    assert rep.of("no-wrap")[0].verdict == Q.REFUTED
    assert rep.of("float-cast-exact")[0].verdict == Q.REFUTED  # |acc| > 2**24


def test_float_cast_exactness_is_informational_only():
    rep = Q.verify_layer(
        *_only(_qlinear_matmul(4, 1, np.ones((4, 1)), 0.02, 0.02, 0.05, 0, 0))
    )
    assert all(
        f.severity == Q.INFORMATIONAL
        for f in rep.of("float-cast-exact") + rep.of("requant-bit-exact")
    )


# ---- skips are honest ----------------------------------------------------------------


def test_unsupported_configurations_are_skipped_with_a_reason():
    # per-row zero point on the weight
    m = _model(
        "m (uint8[1,4] a) => (int32[1,2] y) { y = MatMulInteger(a, W, azp, wzp) }",
        dict(
            W=np.ones((4, 2), np.int8),
            azp=np.array(0, np.uint8),
            wzp=np.zeros(4, np.int8),
        ),
    )
    rep = Q.verify_layer(*_only(m))
    assert (
        rep.of("no-wrap")[0].verdict == Q.SKIPPED
        and "per-row" in rep.of("no-wrap")[0].detail
    )
    # non-constant weight
    m = _model(
        "m (uint8[1,4] a, int8[4,2] b) => (int32[1,2] y) { y = MatMulInteger(a, b) }"
    )
    rep = Q.verify_layer(m.graph.node[0], Q._consts(m))
    assert (
        rep.of("no-wrap")[0].verdict == Q.SKIPPED
        and "constant" in rep.of("no-wrap")[0].detail
    )
    assert not rep.proved  # a skipped soundness finding is never "proved"


# ---- onnxsim's own quantizers ----------------------------------------------------------


def _float_matmul(k, n, seed=0):
    rng = np.random.default_rng(seed)
    return _model(
        f"m (float[2,{k}] x) => (float[2,{n}] y) {{ y = MatMul(x, W) }}",
        dict(W=rng.standard_normal((k, n)).astype(np.float32)),
    )


def test_dynamic_chain_from_onnxsim_quantize_dynamic_is_proved_and_matches_execution():
    ref = _float_matmul(48, 6)
    qm = onnxsim.quantize_dynamic(ref)
    rep = Q.verify_model(qm, reference_model=ref)
    assert rep.ok, str(rep)
    layer = rep.layers[0]
    assert {f.kind for f in layer.findings} >= {
        "no-wrap",
        "dynamic-rel-error",
        "weight-fidelity",
    }
    assert layer.of("weight-fidelity")[0].verdict == Q.PROVED
    # execution check: the certified relative error holds on the real intermediates
    m = onnx.ModelProto()
    m.CopyFrom(qm)
    mm = next(n for n in m.graph.node if n.op_type == "MatMulInteger")
    dql = next(n for n in m.graph.node if n.op_type == "DynamicQuantizeLinear")
    cast = next(n for n in m.graph.node if n.op_type == "Cast")
    del m.graph.output[:]
    for name, t in (
        (mm.output[0], 6),
        (dql.output[1], 1),
        (qm.graph.output[0].name, 1),
    ):
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, t, None))
    consts = Q._consts(qm)
    ws = next(
        consts[i]
        for n in qm.graph.node
        if n.op_type == "Mul"
        for i in n.input
        if i in consts and consts[i].dtype == np.float32 and consts[i].size == 6
    )
    bound = float(Q.dynamic_rel_error_bound(10**6))
    rng = np.random.default_rng(4)
    for _ in range(20):
        x = rng.standard_normal((2, 48)).astype(np.float32)
        acc, xs, y = _ort(m, {"x": x})
        exact = acc.astype(np.float64) * float(xs) * ws.astype(np.float64)
        err = np.abs(y - exact) / np.maximum(np.abs(exact), 1e-30)
        assert err.max() <= bound
    del cast


def _dynamic_chain(k, wq):
    """The chain onnxsim's quantize_dynamic emits, assembled by hand (it refuses to emit it past the safe depth)."""
    return _model(
        f"""
        m (float[1,{k}] x) => (float[1,1] y) {{
          xq, xs, xzp = DynamicQuantizeLinear(x)
          acc = MatMulInteger(xq, Wq, xzp)
          accf = Cast<to=1>(acc)
          sc = Mul(xs, Ws)
          y = Mul(accf, sc)
        }}""",
        dict(Wq=wq.astype(np.int8), Ws=np.array([0.01], np.float32)),
    )


def test_dynamic_chain_overflow_is_refuted_and_safe_k_is_proved():
    k_safe = INT32_MAX // (255 * 127)
    # onnxsim's own quantizer refuses to quantize past the boundary: no integer layer is emitted
    big = _model(
        f"m (float[1,{k_safe + 1}] x) => (float[1,1] y) {{ y = MatMul(x, W) }}",
        dict(W=np.full((k_safe + 1, 1), 127, np.float32)),
    )
    assert not any(
        n.op_type == "MatMulInteger" for n in onnxsim.quantize_dynamic(big).graph.node
    )
    # ... and at the boundary it does, and the verifier proves it
    ok = onnxsim.quantize_dynamic(
        _model(
            f"m (float[1,{k_safe}] x) => (float[1,1] y) {{ y = MatMul(x, W) }}",
            dict(W=np.full((k_safe, 1), 127, np.float32)),
        )
    )
    assert Q.verify_model(ok).layers[0].of("no-wrap")[0].verdict == Q.PROVED
    # a hand-built chain just past it is refuted with a float input that reproduces the wrap on onnxruntime
    qm = _dynamic_chain(k_safe + 1, np.full((k_safe + 1, 1), 127))
    rep = Q.verify_model(qm)
    f = rep.layers[0].of("no-wrap")[0]
    assert f.verdict == Q.REFUTED and not rep.ok
    observed, exact = Q.replay_dynamic_no_wrap(qm, rep.layers[0].name, f)
    assert int(exact[0, 0]) == (k_safe + 1) * 255 * 127 > INT32_MAX and int(
        observed[0, 0]
    ) != int(exact[0, 0])
    # a non-negative input range pins the zero point to 0, which does not help here: the vertex is zp=0 already
    assert (
        Q.verify_model(qm, input_ranges={"x": (0.0, 1.0)})
        .layers[0]
        .of("no-wrap")[0]
        .verdict
        == Q.REFUTED
    )
    # but an all-negative-weight layer with non-negative input never goes above 0 and stays within int32 on the low side
    neg = _dynamic_chain(k_safe, np.full((k_safe, 1), -127))
    assert (
        Q.verify_model(neg, input_ranges={"x": (0.0, 1.0)})
        .layers[0]
        .of("no-wrap")[0]
        .verdict
        == Q.PROVED
    )


def test_dynamic_zero_point_coupling_is_tighter_than_the_naive_bound():
    w = np.array([100, -90] * 100, dtype=np.int64)
    lo, hi, _, _ = Q._dynamic_accumulator_range(w, 0, 255)
    naive = 255 * int(np.abs(w).sum())
    assert (
        hi == 255 * 100 * 100 and hi < naive
    )  # the coupled bound is exactly 255*max(P, N)
    assert Q._dynamic_zero_points((0.0, 5.0)) == (0, 0) and Q._dynamic_zero_points(
        (-3.0, 0.0)
    ) == (255, 255)
    assert Q._dynamic_zero_points((-1.0, 1.0)) == (0, 255) and Q._dynamic_zero_points(
        None
    ) == (0, 255)


def test_weight_fidelity_catches_a_tampered_weight():
    ref = _float_matmul(16, 4)
    qm = onnxsim.quantize_dynamic(ref)
    consts = Q._consts(qm)
    mm = next(n for n in qm.graph.node if n.op_type == "MatMulInteger")
    for t in qm.graph.initializer:
        if t.name == mm.input[1]:
            w = numpy_helper.to_array(t).copy()
            w[3, 2] = np.int8(np.clip(int(w[3, 2]) + 9, -127, 127))
            t.CopyFrom(numpy_helper.from_array(w, t.name))
    rep = Q.verify_model(qm, reference_model=ref)
    f = rep.layers[0].of("weight-fidelity")[0]
    assert f.verdict == Q.REFUTED and "(3, 2)" in f.detail and not rep.ok
    del consts


def test_weight_fidelity_function_directly():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((6, 3)).astype(np.float32)
    s = (np.abs(w).max(axis=0) / 127).astype(np.float32)
    q = np.rint(w / s).astype(np.int8)
    assert Q.weight_fidelity(w, q, s, 0, axis=1)[0] == Q.PROVED
    q2 = q.copy()
    q2[0, 0] += 3
    verdict, detail, idx = Q.weight_fidelity(w, q2, s, 0, axis=1)
    assert verdict == Q.REFUTED and idx == (0, 0)
    assert Q.weight_fidelity(w, q[:3], s, 0, axis=1)[0] == Q.SKIPPED


def test_weight_fidelity_slack_scales_with_the_weight_magnitude():
    """Regression for a real false alarm: a fixed 2**-22 slack flagged correct weights near a tie
    at large |w/s| (found on a 4096 x 4096 layer). fp32 division error is relative, not absolute."""
    s = np.array([0.01], dtype=np.float32)
    q = np.array([127], dtype=np.int8)
    deq = float(s[0]) * 127
    just_inside = np.array(
        [deq + float(s[0]) * (0.5 + 4e-6)]
    )  # 4e-6 steps past the tie: fp32-division noise at |w/s|=127
    beyond = np.array([deq + float(s[0]) * (0.5 + 1e-4)])  # a real violation
    assert Q.weight_fidelity(just_inside, q, s, 0, axis=0)[0] == Q.PROVED
    assert Q.weight_fidelity(beyond, q, s, 0, axis=0)[0] == Q.REFUTED
    # at small |w/s| the allowance is correspondingly tiny
    small_q = np.array([1], dtype=np.int8)
    assert (
        Q.weight_fidelity(
            np.array([float(s[0]) * (1 + 0.5 + 4e-6)]), small_q, s, 0, axis=0
        )[0]
        == Q.REFUTED
    )


def test_big_layer_weight_fidelity_has_no_false_alarms_near_ties():
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((1500, 600)) * 0.05).astype(np.float32)
    ref = _model(
        "m (float[1,1500] x) => (float[1,600] y) { y = MatMul(x, W) }", dict(W=w)
    )
    rep = Q.verify_model(onnxsim.quantize_dynamic(ref), reference_model=ref)
    assert rep.layers[0].of("weight-fidelity")[0].verdict == Q.PROVED and rep.ok, str(
        rep
    )


def test_qoperator_model_from_onnxsim_is_verified_against_its_float_reference():
    rng = np.random.default_rng(6)
    ref = _model(
        "m (float[4,32] x) => (float[4,5] y) { y = MatMul(x, W) }",
        dict(W=rng.standard_normal((32, 5)).astype(np.float32)),
        opset=13,
    )
    calib = [{"x": rng.standard_normal((4, 32)).astype(np.float32)} for _ in range(4)]
    try:
        qdq = onnxsim.quantize_static(ref, calibration_data=calib)
        qop = onnxsim.quantize_qoperator(ref, calibration_data=calib)
    except Exception as e:  # pragma: no cover - API drift
        pytest.skip(f"cannot build a QOperator model here: {e}")
    del qdq
    ops = {n.op_type for n in qop.graph.node}
    if "QLinearMatMul" not in ops:
        pytest.skip(f"quantize_qoperator produced {sorted(ops)}")
    rep = Q.verify_model(qop, reference_model=ref, input_ranges={"x": (-4.0, 4.0)})
    layer = next(layer for layer in rep.layers if layer.op_type == "QLinearMatMul")
    assert layer.of("no-wrap")[0].verdict == Q.PROVED
    assert layer.of("requant-1lsb")[0].verdict == Q.PROVED
    assert layer.of("weight-fidelity")[0].verdict == Q.PROVED
    # no PROVED layer is contradicted by execution: integer model vs float reference stays near
    x = rng.standard_normal((4, 32)).astype(np.float32)
    a, b = _ort(ref, {"x": x})[0], _ort(qop, {"x": x})[0]
    assert np.abs(a - b).max() < 0.5 * np.abs(a).max() + 1.0


def _qlinear_conv(cin, cout, k, wq, bias, sa, sb, sy, za, zy, x_hw):
    return _model(
        f"m (uint8[1,{cin},{x_hw},{x_hw}] x) => (uint8[1,{cout},?,?] y) "
        f"{{ y = QLinearConv(x, sa, za, W, sb, zb, sy, zy, B) }}",
        dict(
            W=wq.astype(np.int8),
            B=bias.astype(np.int32),
            sa=np.array(sa, np.float32),
            sb=np.array(sb, np.float32),
            sy=np.array(sy, np.float32),
            za=np.array(za, np.uint8),
            zb=np.array(0, np.int8),
            zy=np.array(zy, np.uint8),
        ),
    )


@requires_exact_u8s8
def test_qlinear_conv_with_int32_bias_matches_the_fp32_pipeline_at_the_extremes():
    rng = np.random.default_rng(7)
    cin, cout, k = 2, 3, 3
    wq = rng.integers(-100, 101, (cout, cin, k, k))
    bias = rng.integers(-3000, 3000, cout)
    sa, sb, sy, za, zy = 0.02, 0.011, 0.9, 90, 128
    m = _qlinear_conv(cin, cout, k, wq, bias, sa, sb, sy, za, zy, k)
    node, consts = _only(m)
    rep = Q.verify_layer(node, consts, x_shape=(1, cin, k, k))
    assert (
        rep.of("no-wrap")[0].verdict == Q.PROVED
        and rep.of("requant-1lsb")[0].verdict == Q.PROVED
    )
    layer, _ = Q._extract_layer(node, consts, {}, {"x": (1, cin, k, k)}, None)
    mins, maxs, _, _ = Q._channel_ranges(layer)
    m_real, m32 = Q.requant_multiplier(np.float32(sa), np.float32(sb), np.float32(sy))
    # drive each channel to its exact accumulator extremes through onnxruntime and compare
    for c in range(cout):
        for maximise, want in ((True, maxs[c]), (False, mins[c])):
            d = Q._vertex_input(layer.wq[c], layer.d_lo, layer.d_hi, maximise)
            x = (d.reshape(1, cin, k, k) + za).astype(np.uint8)
            exact = int(
                Q._exact_conv(
                    x.astype(np.int64) - za, wq.astype(np.int64), Q._attrs(node)
                )[0, c, 0, 0]
            ) + int(bias[c])
            assert exact == want  # the closed form (with bias) is attained, exactly
            got = int(_ort(m, {"x": x})[0][0, c, 0, 0])
            assert got == Q.fp32_requant(exact, m32, zy, 0, 255)


@requires_exact_u8s8
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_requant_emulation_is_bit_exact_against_onnxruntime_with_per_channel_scales(
    seed,
):
    """The model every verdict rests on: ORT's QLinearMatMul output == our fp32 pipeline, bit for bit."""
    rng = np.random.default_rng(seed)
    k, n = 96, 8
    wq = rng.integers(-127, 128, (k, n)).astype(np.int8)
    sa, sy, za, zy = (
        np.float32(rng.uniform(0.005, 0.05)),
        np.float32(rng.uniform(0.05, 0.5)),
        110,
        120,
    )
    sb = rng.uniform(0.002, 0.02, n).astype(np.float32)
    m = _model(
        f"m (uint8[8,{k}] a) => (uint8[8,{n}] y) {{ y = QLinearMatMul(a, sa, za, W, sb, zb, sy, zy) }}",
        dict(
            W=wq,
            sa=np.array(sa),
            sb=sb,
            sy=np.array(sy),
            za=np.array(za, np.uint8),
            zb=np.zeros(n, np.int8),
            zy=np.array(zy, np.uint8),
        ),
    )
    node, consts = _only(m)
    rep = Q.verify_layer(node, consts)
    assert (
        rep.of("no-wrap")[0].verdict == Q.PROVED
        and rep.of("requant-1lsb")[0].verdict == Q.PROVED
    )
    layer, _ = Q._extract_layer(node, consts, {}, {}, None)
    mins, maxs, _, _ = Q._channel_ranges(layer)
    mismatches = 0
    for _ in range(25):
        a = rng.integers(0, 256, (8, k), dtype=np.uint8)
        got = _ort(m, {"a": a})[0]
        acc = (a.astype(np.int64) - za) @ wq.astype(np.int64)
        for j in range(n):
            assert (acc[:, j] >= mins[j]).all() and (
                acc[:, j] <= maxs[j]
            ).all()  # a PROVED range is never contradicted
            _, m32 = Q.requant_multiplier(sa, sb[j], sy)
            want = np.array(
                [Q.fp32_requant(int(v), m32, zy, 0, 255) for v in acc[:, j]]
            )
            mismatches += int((want != got[:, j]).sum())
    assert mismatches == 0


def test_activation_range_from_quantize_linear_narrows_the_proof():
    k = 70000
    m = _model(
        f"""
        m (float[1,{k}] x) => (uint8[1,1] y) {{
          a = QuantizeLinear(x, qs, qzp)
          y = QLinearMatMul(a, sa, za, W, sb, zb, sy, zy)
        }}""",
        dict(
            W=np.full((k, 1), 127, np.int8),
            qs=np.array(0.01, np.float32),
            qzp=np.array(0, np.uint8),
            sa=np.array(0.01, np.float32),
            za=np.array(0, np.uint8),
            sb=np.array(0.01, np.float32),
            zb=np.array(0, np.int8),
            sy=np.array(40.0, np.float32),
            zy=np.array(0, np.uint8),
        ),
    )
    nowrap = lambda r: r.layers[0].of("no-wrap")[0]  # noqa: E731
    assert (
        nowrap(Q.verify_model(m)).verdict == Q.REFUTED
    )  # full uint8 range: 70000*255*127 wraps
    narrow = Q.verify_model(
        m, input_ranges={"x": (0.0, 1.5)}
    )  # codes <= 150: 70000*150*127 < 2**31
    assert nowrap(narrow).verdict == Q.PROVED
    assert "restricted to [0, 150]" in narrow.layers[0].of("note")[0].detail


def test_integer_graph_input_range_is_taken_as_codes():
    k = 70000
    m = _matmul_integer(k, np.full((k, 1), 127))
    assert Q.verify_model(m).layers[0].of("no-wrap")[0].verdict == Q.REFUTED
    assert (
        Q.verify_model(m, input_ranges={"a": (0, 100)})
        .layers[0]
        .of("no-wrap")[0]
        .verdict
        == Q.PROVED
    )


def test_probe_reports_a_boolean_or_none():
    assert Q.probe_u8s8_saturation() in (True, False, None)
    rep = Q.verify_model(_matmul_integer(4, np.ones((4, 1))), probe_runtime=True)
    assert any("probe" in n or "u8 x s8" in n for n in rep.notes)


def test_report_text_and_properties():
    rep = Q.verify_model(_matmul_integer(8, np.ones((8, 2))))
    assert rep.ok and "no-wrap" in str(rep) and "OK" in str(rep)
    assert Q.verify_model(
        _model("m (float[2] x) => (float[2] y) { y = Relu(x) }")
    ).notes == ["no integer layers found"]
