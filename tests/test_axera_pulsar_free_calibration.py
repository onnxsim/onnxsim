"""No-device checks for ``scripts/axera/pulsar_free_calibration.py``: the
scales, zero points and signedness computed from a float graph and its
calibration samples against committed Pulsar2 oracles (the resolved
``quant_axmodel.json`` entries of native builds), and calibrate + stitch
against the native fused builds."""

import gzip
import hashlib
import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import emitter  # noqa: E402
import graph_stitch as gs  # noqa: E402
import pulsar_free_calibration as pfc  # noqa: E402

FIXTURES = os.path.join(AXERA, "fixtures", "pulsar_free_calibration")
with open(os.path.join(FIXTURES, "index.json")) as _f:
    INDEX = json.load(_f)
CASES = INDEX["cases"]

V = "float[1,64]"
# the float graphs Pulsar2 was given (ONNX text; random constants are attached
# as numpy initializers by ``_float_model``)
GRAPHS = {
    "sigmoid": f"g ({V} x) => ({V} y) {{ y = Sigmoid(x) }}",
    "mul": f"g ({V} a, {V} b) => ({V} y) {{ y = Mul(a, b) }}",
    "softmax": f"g ({V} x) => ({V} y) {{ y = Softmax<axis=-1>(x) }}",
    "silu": f"g ({V} x) => ({V} y) {{ s = Sigmoid(x) y = Mul(x, s) }}",
    "rmsnorm": (
        f"g ({V} x) => ({V} y) <float eps = {{0.00001}}> {{"
        " sq = Mul(x, x) ms = ReduceMean<axes=[-1], keepdims=1>(sq)"
        " me = Add(ms, eps) rt = Sqrt(me) n = Div(x, rt) y = Mul(n, gain) }"
    ),
    "attention": (
        "g (float[8,64] q, float[64,8] kt, float[8,64] v) => (float[8,64] y) {"
        " s = MatMul(q, kt) p = Softmax<axis=-1>(s) y = MatMul(p, v) }"
    ),
    "linear64": f"g ({V} x) => ({V} y) {{ y = MatMul(x, w) }}",
}


def _model(body, initializer=(), opset=17, ir_version=9):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _linear_weight(seed: int) -> np.ndarray:
    """The 64x64 weights of native build ``linear64_w<seed>`` ([in, out])."""
    return np.random.default_rng(seed).uniform(-0.1, 0.1, (64, 64)).astype(np.float32)


def _float_model(case: str) -> onnx.ModelProto:
    graph = case.rsplit("_", 1)[0]
    init = []
    if graph == "rmsnorm":
        gain = gs.load_index()["graphs"]["rmsnorm"]["constants"]["gain"]
        init = [numpy_helper.from_array(np.array([gain], np.float32), "gain")]
    elif graph == "linear64":
        w = _linear_weight(CASES[case]["weight_seed"])
        init = [numpy_helper.from_array(w, "w")]
    return _model(GRAPHS[graph], init)


def _samples(case: str) -> dict[str, np.ndarray]:
    out = {}
    for name, file in CASES[case]["samples"].items():
        with gzip.open(os.path.join(FIXTURES, file), "rb") as f:
            out[name] = np.load(f)
    return out


def _ulps(a, b) -> int:
    a = np.atleast_1d(np.asarray(a, np.float32)).view(np.int32).astype(np.int64)
    b = np.atleast_1d(np.asarray(b, np.float32)).view(np.int32).astype(np.int64)
    assert a.shape == b.shape
    return int(np.abs(a - b).max())


def _calibrate(case: str) -> dict[str, pfc.TensorQuant]:
    return pfc.calibrate(_float_model(case), _samples(case))


def test_linear_weights_are_the_native_builds_weights():
    # the weights are regenerated, not committed: numpy must still draw the
    # stream the native builds were given
    with open(os.path.join(AXERA, "fixtures", "linear_emit", "index.json")) as f:
        builds = json.load(f)["builds"]
    for seed in range(1, 7):
        digest = hashlib.sha256(_linear_weight(seed).tobytes()).hexdigest()
        assert digest == builds[f"linear64_w{seed}"]["weight_sha256"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_calibration_matches_pulsar2(case):
    got = _calibrate(case)
    oracle = CASES[case]["tensors"]
    off = CASES[case]["scale_ulps"]  # tensors the comparison found one ulp off
    assert set(off) <= set(oracle)
    for name, want in oracle.items():
        # Pulsar2 names the fully connected weight w_trans
        q = got["w" if name == "w_trans" else name]
        assert q.zero_point == want["zero_point"], name
        assert q.signed == want["signed"], name
        ulps = _ulps(q.scale, want["scale"])
        if name in off:
            assert 0 < ulps <= off[name] == 1, (name, ulps)
        else:
            assert ulps == 0, (name, ulps)
    # nothing is quantized that Pulsar2 does not quantize
    assert {"w_trans" if k == "w" else k for k in got} == set(oracle)


def test_each_rule_is_named():
    att = _calibrate("attention_pm1")
    # both operands of a MatMul of two activations are signed, a Softmax
    # output read by one included
    assert {k for k, q in att.items() if q.rule == "SIGNED_ACT"} == {
        "q",
        "kt",
        "p",
        "v",
    }
    assert {k for k, q in att.items() if q.rule == "UNSIGNED"} == {"s", "y"}
    rms = _calibrate("rmsnorm_pm1")
    assert rms["eps"].rule == rms["gain"].rule == "CONST_U8"
    # a positive constant: range widened to 0, so eps gets code 255
    assert rms["eps"].scale == np.float32(float(np.float32(1e-5)) / 255.0)
    assert rms["eps"].zero_point == 0
    lin = _calibrate("linear64_w1")
    assert lin["w"].rule == "WEIGHT_S8_PC" and lin["w"].per_channel
    assert lin["w"].scale.shape == (64,) and lin["w"].scale.dtype == np.float32


def test_formulas():
    s, zp = pfc.unsigned_minmax(-1.0, 3.0)
    assert s == np.float32(4.0 / 255.0) and zp == round(1.0 / float(s))
    s, zp = pfc.unsigned_minmax(0.5, 2.0)  # the range always includes 0
    assert s == np.float32(2.0 / 255.0) and zp == 0
    assert pfc.unsigned_minmax(0.0, 0.0) == (np.float32(0), 0)
    s, zp = pfc.signed_symmetric(1.5)
    assert s == np.float32(1.5) / np.float32(127.5) and zp == 0


@pytest.mark.parametrize("seed", range(1, 7))
def test_weight_codes_divide_in_float32(seed):
    # the calibration forward pass must see the codes the compiler stores
    # (emitter.codes_of, byte-exact against the weight tables); a float64
    # division rounds some of them the other way
    w = _linear_weight(seed)
    codes = pfc.weight_codes(w)
    assert np.array_equal(codes + 128, emitter.codes_of(w.T, axis=0).T)
    wide = np.rint(w.astype(np.float64) / pfc.weight_scales(w).astype(np.float64))
    assert 0 < int((np.clip(wide, -128, 127) != codes).sum()) < 32
    # a channel's peak is +-127.5: -128 is kept, +128 clips to 127
    assert codes.min() == -128 and codes.max() == 127


def test_constants_are_baked_into_the_forward_pass():
    # without fake-quantized weights the output scale is far off (thousands
    # of ulps), with them it is the oracle's
    case = "linear64_w4"
    model, samples = _float_model(case), _samples(case)
    want = CASES[case]["tensors"]["y"]["scale"]
    assert _ulps(pfc.calibrate(model, samples)["y"].scale, want) == 0
    w = numpy_helper.to_array(model.graph.initializer[0])
    y = np.concatenate([x.reshape(1, 64) @ w for x in samples["x"]])
    assert _ulps(pfc.unsigned_minmax(y.min(), y.max())[0], want) > 100


def test_stitch_calibration_gives_graph_stitch_arguments():
    _, scales, zps, signed, _ = gs.fixture_case("attention", "pm1")
    got = pfc.calibrate_for_stitch(
        _float_model("attention_pm1"), _samples("attention_pm1"), scales
    )
    assert list(got[0]) == list(scales)
    assert got[1] == zps and got[2] == sorted(signed)
    # per-channel weights are left out by default and refused by name
    quants = _calibrate("linear64_w1")
    assert set(pfc.stitch_calibration(quants)[0]) == {"x", "y"}
    with pytest.raises(ValueError, match="no single-scale"):
        pfc.stitch_calibration(quants, ["x", "w"])
    with pytest.raises(ValueError, match="no single-scale"):
        pfc.stitch_calibration(quants, ["nope"])


# ---- calibrate + stitch, no quant json ---------------------------------------------
LANES = set(gs.LANES_LO) | set(gs.LANES_HI)


def _stitched(graph, cal):
    case = f"{graph}_{cal}"
    wiring, scales, _, _, oracle = gs.fixture_case(graph, cal)
    sc, zp, signed = pfc.calibrate_for_stitch(
        _float_model(case), _samples(case), scales
    )
    return gs.stitch_model(wiring, sc, zp, signed), oracle


def _segments(model):
    mc = bytes(gs.mre.mcode_initializer(model).raw_data)
    return [gs.records(r) for r in gs.suc.decode_segments(mc)]


def _params(model):
    init = next(i for i in model.graph.initializer if i.name == "npu_params")
    return bytes(init.raw_data)


@pytest.mark.parametrize("cal", ["pm1", "pm4"])
def test_silu_from_samples_equals_the_native_build(cal):
    # every scale is bit-exact, so the whole model is the native one
    assert not CASES[f"silu_{cal}"]["scale_ulps"]
    model, oracle = _stitched("silu", cal)
    assert gs.compare_models(model, oracle) == []


@pytest.mark.parametrize(
    "graph,cal,records",
    [("rmsnorm", "pm1", 16), ("rmsnorm", "pm4", 16), ("attention", "pm1", 16)],
)
def test_one_ulp_scales_change_float_lanes_only(graph, cal, records):
    # a scale one ulp off moves the float32 lane words derived from it, and
    # nothing else: no zero point, address, table or record count
    assert CASES[f"{graph}_{cal}"]["scale_ulps"]
    model, oracle = _stitched(graph, cal)
    got, want = _segments(model), _segments(oracle)
    assert [len(s) for s in got] == [len(s) for s in want]
    changed = 0
    for si in range(1, 5):
        for a, b in zip(got[si], want[si]):
            if a != b:
                changed += 1
                assert a[:4] == b[:4] and gs._reg(a) in LANES
                assert (
                    _ulps(*(np.uint32(gs._val(w)).view(np.float32) for w in (a, b)))
                    <= 2
                )
    assert changed == records
    pa, pb = _params(model), _params(oracle)
    assert len(pa) == len(pb)
    if graph == "attention":
        # MatMul's npu_params constant holds s_a * s_b / s_y as float32 lanes
        n = len(pa) // 4 * 4
        fa, fb = (np.frombuffer(p[:n], "<f4") for p in (pa, pb))
        assert 0 < int((fa != fb).sum()) <= 72 and _ulps(fa, fb) == 1
        assert pa[n:] == pb[n:]
    else:
        assert pa == pb


# ---- refused -------------------------------------------------------------------
def _x16(shape=(1, 64)):
    return {"x": [np.zeros(shape, np.float32)] * 2}


def test_unsupported_op_is_refused():
    model = _model(f"g ({V} x) => ({V} y) {{ y = Tanh(x) }}")
    with pytest.raises(NotImplementedError, match="Tanh"):
        pfc.calibrate(model, _x16())
    # Pulsar2 leaves a standalone Concat float: there is nothing to reproduce
    model = _model(f"g ({V} x) => (float[1,128] y) {{ y = Concat<axis=1>(x, x) }}")
    with pytest.raises(NotImplementedError, match="Concat"):
        pfc.calibrate(model, _x16())


def test_unmeasured_initializer_uses_are_refused():
    w = numpy_helper.from_array(np.ones((64, 64), np.float32), "w")
    model = _model("g (float[64,1] x) => (float[64,1] y) { y = MatMul(w, x) }", [w])
    with pytest.raises(NotImplementedError, match="first operand"):
        pfc.calibrate(model, _x16((64, 1)))
    c = numpy_helper.from_array(np.ones((1, 64), np.float32), "c")
    model = _model(f"g ({V} x) => ({V} y) {{ s = Sigmoid(c) y = Mul(x, s) }}", [c])
    with pytest.raises(NotImplementedError, match="Sigmoid reading initializer"):
        pfc.calibrate(model, _x16())


def test_bad_samples_are_refused():
    model = _model(GRAPHS["mul"])
    a = [np.zeros((1, 64), np.float32)] * 2
    with pytest.raises(ValueError, match="no calibration samples"):
        pfc.calibrate(model, {"a": a})
    with pytest.raises(ValueError, match="same, nonzero sample count"):
        pfc.calibrate(model, {"a": a, "b": a[:1]})
    with pytest.raises(ValueError, match="elements"):
        pfc.calibrate(model, {"a": a, "b": [np.zeros((1, 8), np.float32)] * 2})
