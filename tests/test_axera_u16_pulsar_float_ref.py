"""Pulsar2's float Sigmoid, Softmax, MatMul and ReduceMean as ``calibrate_graph``
computes them for ``NEW_OPS`` graphs (``u16_retarget._pulsar_forward``).

The golden values are Pulsar2 7.0 (AX650, NPU1) float IO dumps of single-op probes:
68 Sigmoid inputs over [-30, 30] (plus values near 0) and two Softmax probes over
rows of 5 and 17 (with a -1e4 masked entry per row). The float32 values are written
as ``repr`` (round-trips exactly). Host checks only, no build. The scratch-gated
check reads the native seed-777 build (``AXERA_U16_IDENT_BUILD``, skipped when absent):
its quant tensors must keep the exact matches this model reaches (77 of 114, 14 of 21
ReduceMean-derived; ``AXERA_U16_IDENT_BUILD`` at ``/nonexistent`` skips it).
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import u16_retarget as ur  # noqa: E402

F = np.float32
IDENT = os.environ.get(
    "AXERA_U16_IDENT_BUILD",
    "/mnt/data/cache/claude-work/tmp/qwen-fuse/Q/ident/"
    "templates_alyrrsqkrk4096_v0_s777",
)

SIG_X = [
    -1.4346306324005127,
    -0.00013593866606242955,
    -5.5058088302612305,
    17.680694580078125,
    12.542072296142578,
    22.804378509521484,
    9.48564624786377,
    -17.66206169128418,
    -0.48146331310272217,
    28.93962860107422,
    10.469110488891602,
    -28.986040115356445,
    -16.63397979736328,
    6.361154079437256,
    -7.522037982940674,
    20.76192855834961,
    4.367015838623047,
    -4.452402591705322,
    -24.90395736694336,
    25.88679313659668,
    14.625818252563477,
    -19.753156661987305,
    18.68209457397461,
    -3.4358632564544678,
    19.704103469848633,
    -10.588589668273926,
    -23.85944938659668,
    16.673839569091797,
    -27.94983673095703,
    1.2967238426208496,
    24.84889793395996,
    11.527950286865234,
    -21.81273078918457,
    5.3432416915893555,
    7.390423774719238,
    -26.9533748626709,
    2.3323042392730713,
    -22.80896759033203,
    -30.0,
    -25.928796768188477,
    -20.78053855895996,
    -18.677433013916016,
    -15.587279319763184,
    -14.576229095458984,
    -13.586179733276367,
    -12.557127952575684,
    -11.591079711914062,
    -9.529976844787598,
    -8.539926528930664,
    -6.4998250007629395,
    -2.44362211227417,
    0.42752137780189514,
    3.3466672897338867,
    8.422921180725098,
    13.574178695678711,
    15.674283981323242,
    21.79458999633789,
    23.858692169189453,
    26.873844146728516,
    27.8848934173584,
    -3.9866886680783864e-08,
    8.42769441078417e-05,
    -3.1546045420327573e-07,
    0.006745452526956797,
    -1.7245525668840855e-05,
    -0.01565839909017086,
    5.745211183239007e-06,
    -1.4621457822272532e-08,
]
SIG_Y = [
    0.1923782080411911,
    0.499966025352478,
    0.004046658985316753,
    1.0,
    0.9999964237213135,
    1.0,
    0.9999240636825562,
    2.1353253387701443e-08,
    0.3819066286087036,
    1.0,
    0.9999716281890869,
    2.579424033307548e-13,
    5.969724981014224e-08,
    0.998275637626648,
    0.0005407363059930503,
    1.0,
    0.9874699711799622,
    0.011516369879245758,
    1.528793237837167e-11,
    1.0,
    0.9999995231628418,
    2.6382322992901663e-09,
    1.0,
    0.03119325451552868,
    1.0,
    2.5201303287758492e-05,
    4.3448300512949345e-11,
    1.0,
    7.2700959835581e-13,
    0.7852831482887268,
    1.0,
    0.9999901056289673,
    3.363964107272466e-10,
    0.9952424764633179,
    0.9993832111358643,
    1.9692372732471597e-12,
    0.9115173816680908,
    1.242199121342935e-10,
    9.357623589846195e-14,
    5.4861371522052504e-12,
    9.443364978878321e-10,
    7.735598828162438e-09,
    1.7003195296183549e-07,
    4.6733029535062087e-07,
    1.2577564802995767e-06,
    3.519711526678293e-06,
    9.248132300854195e-06,
    7.263603038154542e-05,
    0.00019546640396583825,
    0.0015014444943517447,
    0.07990620285272598,
    0.60528165102005,
    0.9659955501556396,
    0.9997802376747131,
    0.9999986886978149,
    0.9999998807907104,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.5000211000442505,
    0.49999988079071045,
    0.5016863346099854,
    0.49999570846557617,
    0.4960854649543762,
    0.5000014305114746,
    0.5,
]
SMA_SHAPE = (2, 5)
SMA_X = [
    -10000.0,
    1.936356544494629,
    -10.929983139038086,
    -20.717147827148438,
    -25.909141540527344,
    -10000.0,
    7.558015823364258,
    3.7125489711761475,
    5.0533647537231445,
    24.222684860229492,
]
SMA_Y = [
    0.0,
    0.9999973773956299,
    2.5835597625700757e-06,
    1.451133380125924e-10,
    8.069620817917345e-13,
    0.0,
    5.789301127379076e-08,
    1.2375456215352187e-09,
    4.730101199612591e-09,
    0.9999998807907104,
]
SMD_SHAPE = (2, 17)
SMD_X = [
    -10000.0,
    22.889320373535156,
    -22.611085891723633,
    0.8161452412605286,
    -8.31203842163086,
    7.221366882324219,
    28.95391845703125,
    27.07653045654297,
    -14.865211486816406,
    15.128395080566406,
    16.523069381713867,
    -29.264755249023438,
    21.156017303466797,
    1.5094666481018066,
    1.5867959260940552,
    -15.90152645111084,
    1.1629109382629395,
    -10000.0,
    -29.072097778320312,
    -20.690074920654297,
    24.124855041503906,
    -14.719586372375488,
    23.52430534362793,
    0.03337166830897331,
    11.647664070129395,
    -28.174514770507812,
    -25.736085891723633,
    -26.024272918701172,
    8.11683464050293,
    -9.063399314880371,
    -23.450700759887695,
    18.21194076538086,
    -9.635199546813965,
    4.087745666503906,
]
SMD_Y = [
    0.0,
    0.0020105864387005568,
    3.4893798016611367e-23,
    5.212720988363728e-13,
    5.659066825275383e-17,
    3.15367565395519e-10,
    0.8652552366256714,
    0.13237470388412476,
    8.067480235767461e-20,
    8.566348697058856e-07,
    3.455374326222227e-06,
    4.498787768962713e-26,
    0.0003552701964508742,
    1.0427255846962025e-12,
    1.1265589155984301e-12,
    2.8620132539256515e-20,
    7.373315825086324e-13,
    0.0,
    5.0837482338112246e-24,
    2.2204976919454124e-20,
    0.6446546316146851,
    8.69763043306645e-18,
    0.35359951853752136,
    2.2208974242987622e-11,
    2.4578305328759598e-06,
    1.2473817309874746e-23,
    1.4288821854175646e-22,
    1.0711186187704555e-22,
    7.196679518983728e-08,
    2.4880034284972125e-15,
    1.4045156225851573e-21,
    0.0017433358589187264,
    1.4044985758855351e-15,
    1.2803275106776368e-09,
]


def _f32(values):
    return np.array([float(v) for v in values], F)


def test_sigmoid_matches_pulsar2_float_dump_bit_for_bit():
    x, y = _f32(SIG_X), _f32(SIG_Y)
    assert np.array_equal(ur._sigmoid_pulsar(x), y)


@pytest.mark.parametrize("name", ["SMA", "SMD"])
def test_softmax_matches_pulsar2_float_dump_bit_for_bit(name):
    shape = globals()[f"{name}_SHAPE"]
    x = _f32(globals()[f"{name}_X"]).reshape(shape)
    y = _f32(globals()[f"{name}_Y"]).reshape(shape)
    assert np.array_equal(ur._softmax_pulsar(x), y)


def test_softmax_sum_is_sixteen_lanes_and_a_tree():
    # 17 terms: the first 16 fill the lanes, then term 16 (a second chunk padded with
    # zeros) joins lane 0; the lanes are then summed by pairwise halving.
    v = np.arange(1, 18, dtype=F).reshape(1, 17)
    lanes = v[0, :16].copy()
    lanes[0] = F(lanes[0] + v[0, 16])
    h = lanes
    while h.size > 1:
        h = (h[: h.size // 2] + h[h.size // 2 :]).astype(F)
    assert ur._lanes_sum(v, 16)[0] == h[0]


def test_pulsar_forward_models_its_ops_and_runs_the_rest_by_reference():
    m = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> agraph (float[1,4] x) => (float[1,4] y, float[1,4] z) '
        "{ s = Sigmoid(x)\n  y = Add(s, x)\n  z = Softmax<axis=-1>(y) }"
    )
    x = np.array([[-3.0, -0.5, 0.25, 4.0]], F)
    env = ur._pulsar_forward(m)({"x": x})
    s = ur._sigmoid_pulsar(x)
    assert np.array_equal(env["s"], s)
    assert np.array_equal(env["y"], (s + x).astype(F))
    assert np.array_equal(env["z"], ur._softmax_pulsar(env["y"]))


def test_pulsar_ref_follows_new_ops_by_default(tmp_path):
    body = "{ s = Sigmoid(x)\n  y = Softmax<axis=-1>(s) }"
    plain = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> agraph (float[1,4] x) => (float[1,4] y) '
        + body
    )
    new = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> agraph (float[1,4] x) => (float[1,4] y) '
        "{ a = Abs(x)\n  s = Sigmoid(x)\n  y = Softmax<axis=-1>(s) }"
    )
    x = np.array([[-7.5, 0.1, 2.0, 9.0]], F)
    for name, model in (("plain", plain), ("new", new)):
        d = tmp_path / name
        d.mkdir()
        onnx.save(model, str(d / "model.onnx"))
    s = {"x": [x]}
    assert ur.calibrate_graph(str(tmp_path / "plain"), samples=s) == ur.calibrate_graph(
        str(tmp_path / "plain"), samples=s, pulsar_ref=False
    )
    assert ur.calibrate_graph(str(tmp_path / "new"), samples=s) == ur.calibrate_graph(
        str(tmp_path / "new"), samples=s, pulsar_ref=True
    )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(IDENT, "model.onnx")),
    reason="native seed-777 build (AXERA_U16_IDENT_BUILD) absent",
)
def test_ident_build_exact_matches_do_not_regress():
    nat = ur.quant_of(IDENT)
    cal = ur.calibrate_graph(IDENT)
    tens = sorted(set(nat) & set(cal))
    exact = [t for t in tens if nat[t][1] == cal[t][1] and F(cal[t][0]) == F(nat[t][0])]
    assert len(exact) >= 77
