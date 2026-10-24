"""No-device checks for ``scripts/axera/u16_retarget.py`` against native Pulsar2
U16 single-op builds.

The builds are local scratch (not committed), from ``AXERA_U16_PROGRAM_BUILDS``
(default: the pulsar-free scratch tree); tests whose build is missing are
skipped. ``recal_l0/*`` and ``u16_elementwise/*`` are the SmolLM2-135M layer-0
programs (real-activation calibration); ``u16_pf/<op>/v*`` are calibration
variants of one op and shape (inputs uniform over chosen ranges plus the two
range ends).
"""

import itertools
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import graph_stitch as gs  # noqa: E402
import u16_retarget as ur  # noqa: E402

B = os.environ.get(
    "AXERA_U16_PROGRAM_BUILDS", "/mnt/data/cache/claude-work/tmp/llm-pulsar-free"
)
RC, EW, V = f"{B}/recal_l0", f"{B}/u16_elementwise", f"{B}/u16_pf"


def _v(name, n):
    return [(f"{V}/{name}/v{i}", "y") for i in range(n)]


# group -> (op, inputs, attrs, [(build, output name)]): one op and shape each
GROUPS = {
    "mulxx576": (
        "Mul",
        ("x", "x"),
        None,
        [
            (f"{RC}/n1_sq", "y"),
            (f"{RC}/nf_sq", "y"),
            (f"{EW}/u16_ew_n_sq_1x576_2x", "z"),
            (f"{EW}/u16_ew_n_sq_1x576_real", "z"),
        ],
    ),
    "mean576": (
        "ReduceMean",
        ("x",),
        {"count": 576},
        [
            (f"{RC}/n1_m", "y"),
            (f"{RC}/nf_m", "y"),
            (f"{EW}/u16_ew_mean_1x576_2x", "z"),
            (f"{EW}/u16_ew_mean_1x576_real", "z"),
        ],
    ),
    "div576": (
        "Div",
        ("x", "r"),
        None,
        [
            (f"{RC}/n1_d", "y"),
            (f"{RC}/nf_d", "y"),
            (f"{EW}/u16_ew_n_d_1x576_2x", "z"),
            (f"{EW}/u16_ew_n_d_1x576_real", "z"),
        ],
    ),
    "mulg576": (
        "Mul",
        ("x", "g"),
        None,
        [
            (f"{RC}/n1_g", "y"),
            (f"{RC}/nf_g", "y"),
            (f"{EW}/u16_ew_m_1x576_2x", "z"),
            (f"{EW}/u16_ew_m_1x576_real", "z"),
        ],
    ),
    "addeps": (
        "Add",
        ("x", "eps"),
        None,
        [(f"{RC}/n1_e", "y"), (f"{RC}/n2_e", "y"), (f"{RC}/nf_e", "y")],
    ),
    "sqrt": (
        "Sqrt",
        ("x",),
        None,
        [(f"{RC}/n1_r", "y"), (f"{RC}/n2_r", "y"), (f"{RC}/nf_r", "y")],
    ),
    "add576": (
        "Add",
        ("a", "b"),
        None,
        [(f"{RC}/res1", "y"), (f"{RC}/res2", "y"), (f"{RC}/rq_add16", "y")]
        + _v("add576", 4)[1:],
    ),
    "add192": ("Add", ("a", "b"), None, [(f"{RC}/rk_add16", "y")] + _v("add192", 4)),
    "mul576": (
        "Mul",
        ("a", "b"),
        None,
        [(f"{RC}/rq_a16", "y"), (f"{RC}/rq_b16", "y")] + _v("mul576", 4)[1:],
    ),
    "mul192": ("Mul", ("a", "b"), None, [(f"{RC}/rk_a16", "y"), (f"{RC}/rk_b16", "y")]),
    "mul1536": (
        "Mul",
        ("a", "b"),
        None,
        [(f"{RC}/silu16", "y"), (f"{RC}/act16", "y")] + _v("mul1536", 1),
    ),
    # the attention-mask Add between score and softmax (mask [-32, 0] -> zero point 65535)
    "madd129": ("Add", ("x", "mask"), None, _v("madd129", 4)),
    "sig1536": ("Sigmoid", ("x",), None, [(f"{RC}/sig16", "y")] + _v("sig1536", 4)),
    "smx129": ("Softmax", ("x",), None, [(f"{RC}/smx16", "y")] + _v("smx129", 4)),
    "score": ("MatMul", ("a", "b"), None, [(f"{RC}/score16", "y")] + _v("score", 4)),
    "pv": ("MatMul", ("a", "b"), None, [(f"{RC}/pv16", "y")] + _v("pv", 4)),
}
PAIRS = [
    (g, t, n)
    for g, (_, _, _, members) in GROUPS.items()
    for t, n in itertools.permutations(range(len(members)), 2)
]
REFUSALS = ("cannot be located", "ambiguous template", "repeats a write")


def _need(directory):
    path = os.path.join(directory, "out", "compiled.axmodel")
    if not os.path.exists(path):
        pytest.skip(f"native build not present: {directory}")
    return path


def _q(directory, out):
    return {("y" if k == out else k): v for k, v in ur.quant_of(directory).items()}


@pytest.mark.parametrize("group,t,n", PAIRS)
def test_retarget_equals_the_native_build_or_refuses(group, t, n):
    """Every template -> target pair of a group: the retargeted program equals
    the native build of the target calibration, or the template is refused
    for a stated reason (a zero point below 256, two tensors sharing a zero
    point, a write the template already left out). Never a silent difference."""
    op, ins, attrs, members = GROUPS[group]
    (td, to), (nd, no) = members[t], members[n]
    template, want = _need(td), gs.load_model(_need(nd))
    try:
        got = ur.retarget(template, op, _q(td, to), _q(nd, no), ins, attrs=attrs)
    except ValueError as e:
        assert any(r in str(e) for r in REFUSALS), e
        return
    assert ur.compare(got, want) == []


@pytest.mark.parametrize("group", list(GROUPS))
def test_every_group_has_a_working_template(group):
    op, ins, attrs, members = GROUPS[group]
    ok = 0
    for (td, to), (nd, no) in itertools.permutations(members, 2):
        _need(td), _need(nd)
        try:
            got = ur.retarget(
                f"{td}/out/compiled.axmodel",
                op,
                _q(td, to),
                _q(nd, no),
                ins,
                attrs=attrs,
            )
        except ValueError:
            continue
        ok += ur.compare(got, gs.load_model(f"{nd}/out/compiled.axmodel")) == []
    assert ok >= 1


def test_elision_of_a_zero_zero_point():
    """act16 (both input zero points nonzero) to silu16's calibration, whose
    sigmoid input has zero point 0: the core and DEQUANT writes of 0x1ad0 are
    left out and the segment loses its padding, as in the native build."""
    act, silu = _need(f"{RC}/act16"), f"{RC}/silu16"
    got = ur.retarget(act, "Mul", _q(f"{RC}/act16", "y"), _q(silu, "y"), ("a", "b"))
    want = gs.load_model(_need(silu))
    assert ur.compare(got, want) == []
    assert len(ur._segs(got)[gs.MAIN]) < len(ur._segs(gs.load_model(act))[gs.MAIN])
    with pytest.raises(ValueError, match="cannot be located"):
        ur.retarget(
            _need(silu), "Mul", _q(silu, "y"), _q(f"{RC}/act16", "y"), ("a", "b")
        )


CALIBRATED = [
    (f"{RC}/{k}", op)
    for k, op in [
        ("n1_sq", "Mul"),
        ("n1_m", "ReduceMean"),
        ("n1_e", "Add"),
        ("nf_e", "Add"),
        ("n1_r", "Sqrt"),
        ("n1_d", "Div"),
        ("nf_d", "Div"),
        ("n1_g", "Mul"),
        ("nf_g", "Mul"),
        ("res1", "Add"),
        ("res2", "Add"),
        ("rq_a16", "Mul"),
        ("rq_b16", "Mul"),
        ("rk_b16", "Mul"),
        ("rq_add16", "Add"),
        ("score16", "MatMul"),
        ("smx16", "Softmax"),
        ("pv16", "MatMul"),
        ("sig16", "Sigmoid"),
        ("silu16", "Mul"),
        ("act16", "Mul"),
    ]
] + [(f"{EW}/u16_ew_mean_1x576_2x", "ReduceMean"), (f"{EW}/u16_ew_n_d_1x576_2x", "Div")]
CALIBRATED += [
    (f"{V}/madd129/v{i}", "Add") for i in range(4)
]  # v3: an exact .5 at an asymmetric range


@pytest.mark.parametrize("directory,op", CALIBRATED)
def test_calibrate_reproduces_pulsar2(directory, op):
    _need(directory)
    assert ur.calibrate(directory, op) == ur.quant_of(directory)


def test_calibrate_reduce_mean_output_within_two_ulps():
    # the one known difference: Pulsar2's float forward reduces in an order not reproduced
    d = f"{RC}/nf_m"
    _need(d)
    mine, native = ur.calibrate(d, "ReduceMean"), ur.quant_of(d)
    assert mine["x"] == native["x"]
    assert mine["y"][1] == native["y"][1]
    assert (
        abs(
            np.float32(mine["y"][0]).view(np.int32)
            - np.float32(native["y"][0]).view(np.int32)
        )
        <= 2
    )


def test_symmetric_range_zero_point():
    s, zp = ur.unsigned_minmax(-0.1751, 0.1751)
    assert zp == 32767  # rint(-lo / s) = rint(32767.5...) would give 32768
    assert (
        ur.unsigned_minmax(-36.0, 4.0)[1] == 58982
    )  # an asymmetric exact .5 rounds up
    assert ur.unsigned_minmax(0.0, 2.0)[1] == 0
    assert ur.unsigned_minmax(-2.0, 0.0)[1] == 65535


def test_add_shift_follows_the_ratio_constant():
    q = {"a": (2e-4, 300), "b": (5e-5, 400), "y": (1.5e-4, 500)}  # s_a / s_y > 1
    c = ur.candidates("Add", q, ("a", "b"))
    assert c["add_shift@0x1ea0"] == 14
    lo, hi = c["add_const"] & 0xFFFF, c["add_const"] >> 16
    assert (lo, hi) == (round(2e-4 / 1.5e-4 * 2**14), round(5e-5 / 1.5e-4 * 2**14))


def test_sigmoid_table_is_monotonic_and_saturates():
    c = ur.candidates("Sigmoid", {"x": (1e-4, 30000), "y": (1.5e-5, 0)}, ("x",))
    t = []
    for k in range(128):
        v = c[f"sig_t{k}@{0x1050 + 16 * k:#06x}"]
        t += [v & 0xFFFF, v >> 16]
    assert all(a <= b for a, b in zip(t, t[1:]))
    assert c["sig_tail@0x1850"] >= t[-1]


def test_unknown_op_refused():
    with pytest.raises(ValueError, match="not a retargetable"):
        ur.candidates("Conv", {"x": (1.0, 300), "y": (1.0, 300)}, ("x",))
