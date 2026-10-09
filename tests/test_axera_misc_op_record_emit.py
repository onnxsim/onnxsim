"""No-device checks for ``scripts/axera/misc_op_record_emit.py`` on committed
compiler-built fixtures: emitted records must equal held-out native builds."""

import gzip
import itertools
import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import misc_op_record_emit as mre  # noqa: E402

FIXTURES = os.path.join(AXERA, "fixtures")
with open(os.path.join(FIXTURES, "misc_op_record_emit", "held_out.json")) as _f:
    HELD = json.load(_f)
INDEX = mre.load_index()


def _mcode(rel: str) -> bytes:
    with gzip.open(os.path.join(FIXTURES, rel), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return bytes(mre.mcode_initializer(model).raw_data)


@pytest.mark.parametrize(
    "key", sorted(k for k, v in INDEX.items() if v["op"] not in mre.CALIBRATION_FREE)
)
def test_own_calibration_is_identity(key):
    # Each template's lanes and zero-point records are exactly what the
    # formulas give for its own calibration.
    meta = INDEX[key]
    mc = _mcode(meta["file"])
    sc, zp, n = meta["scales"], meta["zero_points"], meta.get("reduce_count")
    # MaxPool and Neg share one scale; Neg's shift stays inside its program
    x = 1.1 if meta["op"] == "Neg" else 1.5
    y = x if meta["op"] in ("MaxPool", "Neg") else 0.75
    shifted = {"x": sc["x"] * x, "y": sc["y"] * y}
    moved = mre.retarget(mc, meta["op"], sc, shifted, zp, zp, n)
    assert mre.normalized_records(moved) != mre.normalized_records(mc)
    back = mre.retarget(moved, meta["op"], shifted, sc, zp, zp, n)
    assert mre.normalized_records(back) == mre.normalized_records(mc)
    assert mre.retarget(mc, meta["op"], sc, sc, zp, zp, n) == mc


SQRT = sorted(k for k in HELD if "sqrt_512x512x3x3" in k)
REDUCESUM = sorted(k for k in HELD if "reducesum" in k)


@pytest.mark.parametrize("src,dst", list(itertools.permutations(SQRT, 2)))
def test_sqrt_512x512x3x3_matches_held_out_builds(src, dst):
    a, b = HELD[src], HELD[dst]
    got = mre.retarget(_mcode(src), "Sqrt", a["scales"], b["scales"])
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(dst))


@pytest.mark.parametrize("src,dst", list(itertools.permutations(REDUCESUM, 2)))
def test_reducesum_matches_held_out_builds(src, dst):
    # s1 -> asym moves a 0x1b10 record between stages (zp_y == zp_x vs 0).
    a, b = HELD[src], HELD[dst]
    got = mre.retarget(
        _mcode(src),
        "ReduceSum",
        a["scales"],
        b["scales"],
        a["zero_points"],
        b["zero_points"],
    )
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(dst))


SOFTMAX_1X64 = sorted(k for k in HELD if "softmax_1x64" in k)
SOFTMAX_1X64_KEY = "Softmax:1x64:axis1"


@pytest.mark.parametrize("src,dst", list(itertools.permutations(SOFTMAX_1X64, 2)))
def test_softmax_1x64_matches_held_out_builds(src, dst):
    # Last-axis Softmax at an LLM-like shape. The builds cover zp_x 64, 128
    # and 232, so pairs across them move the 0x1b10 = zp_x write's value.
    a, b = HELD[src], HELD[dst]
    got = mre.retarget(
        _mcode(src),
        "Softmax",
        a["scales"],
        b["scales"],
        a["zero_points"],
        b["zero_points"],
    )
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(dst))


@pytest.mark.parametrize("dst", SOFTMAX_1X64)
def test_softmax_1x64_emit_spec_matches_held_out_build(dst):
    # axis -1 resolves to the template's axis1 key
    b = HELD[dst]
    model = mre.emit_spec(
        "Softmax",
        (1, 64),
        attrs={"axis": -1},
        scales=b["scales"],
        zero_points=b["zero_points"],
    )
    got = bytes(mre.mcode_initializer(model).raw_data)
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(dst))


@pytest.mark.parametrize("dst", SOFTMAX_1X64)
def test_softmax_calibration_follows_minmax_rules(dst):
    # The output scale is the calibration's largest probability / 255, not 1/255.
    meta, cal = HELD[dst], HELD[dst]["calibration"]
    rng = np.random.default_rng(cal["seed"])
    samples = [
        rng.uniform(cal["range"][0], cal["range"][1], (1, 64)).astype(np.float32)
        for _ in range(cal["count"])
    ]
    scale, zp = mre.minmax_input_calibration(samples)
    assert np.float32(scale) == np.float32(meta["scales"]["x"])
    assert zp == meta["zero_points"]["x"]
    assert mre.softmax_output_scale(samples) == pytest.approx(
        meta["scales"]["y"], rel=1e-6
    )


def _softmax_1x64_calibration(name):
    if name == "pm4":
        meta = INDEX[SOFTMAX_1X64_KEY]
    else:
        meta = HELD[f"misc_op_record_emit/softmax_1x64_{name}.axmodel.gz"]
    return meta["scales"], meta["zero_points"]


_SOFTMAX_DEVICE = np.load(
    os.path.join(FIXTURES, "misc_op_record_emit", "softmax_1x64_device.npz")
)


@pytest.mark.parametrize("name", ["pm4", "pm1", "pm8", "pm0p5", "asym", "asym2"])
def test_softmax_codes_match_device_capture(name):
    # AX8850 outputs (codes = y / s_y) for the native build at each calibration.
    scales, zps = _softmax_1x64_calibration(name)
    want = _SOFTMAX_DEVICE[f"{name}_codes"].astype(np.int64)
    got = mre.softmax_codes(_SOFTMAX_DEVICE[f"{name}_x"], scales, zps)
    assert np.abs(got - want).max() <= 1
    assert (got == want).mean() > 0.96


def test_softmax_device_output_is_shift_invariant():
    # Rows shifted by whole input codes give bit-identical device outputs:
    # the op depends on code distances from the row maximum only.
    codes = _SOFTMAX_DEVICE["shift_codes"]
    assert list(_SOFTMAX_DEVICE["shift_steps"]) == [0, 16, 32, -16, -32]
    for step in range(1, codes.shape[1]):
        assert np.array_equal(codes[:, step], codes[:, 0])
    scales, zps = _softmax_1x64_calibration("pm4")
    x = _SOFTMAX_DEVICE["shift_x"]
    got = mre.softmax_codes(x.reshape(-1, 64), scales, zps).reshape(codes.shape)
    assert np.array_equal(got[:, 1:], np.repeat(got[:, :1], 4, axis=1))


STEP_PAIRS = sorted(
    (k, k.replace(".axmodel.gz", "__v2.axmodel.gz"))
    for k in HELD
    if k.startswith("misc_op_step_templates/") and "__v2" not in k
)


@pytest.mark.parametrize("a,b", [p for pair in STEP_PAIRS for p in (pair, pair[::-1])])
def test_step_template_matches_second_calibration_build(a, b):
    # ReduceSum pairs include zp_y = 0 targets, where elided 0x1b10 writes
    # remove records and the segment is re-padded to whole 4-record groups
    # (and the reverse). Softmax's pair also moves zp_x; Log's also rewrites
    # its lookup table.
    op = HELD[a].get("op", "ReduceSum")
    meta = next(
        (m for m in INDEX.values() if m["file"] in (a, b.replace("__v2", ""))), {}
    )
    got = mre.retarget(
        _mcode(a),
        op,
        HELD[a]["scales"],
        HELD[b]["scales"],
        HELD[a]["zero_points"],
        HELD[b]["zero_points"],
        meta.get("reduce_count"),
    )
    assert mre.normalized_records(got) == mre.normalized_records(_mcode(b))


def test_record_removing_zero_point_change_is_measured():
    # Both directions change the record count; the fixtures prove it is exact.
    counts = {
        len(mre._chunks(mre.suc.decode_segments(_mcode(k))[2]))
        for pair in STEP_PAIRS
        if "784" in pair[0] and "ReduceSumReshape" not in pair[0]
        for k in pair
    }
    assert len(counts) == 2


def test_log_table_formula():
    meta = INDEX["Log:16x1000"]
    t = mre.log_table(meta["scales"], meta["zero_points"])
    assert len(t) == 258 and t[0] == 0 and t[255] == meta["zero_points"]["y"]
    assert t[256] == t[255] and t[257] == 0
    assert t == sorted(t[:256]) + t[256:]  # log is monotonic


@pytest.mark.parametrize(
    "key,scales,zps",
    [
        # MaxPool shares one scale between input and output
        ("MaxPool:16x64x112x112:k3x3:s2x2:p1,1,1,1", {"x": 0.02, "y": 0.03}, None),
        # Softmax's output zero point is fixed by the template
        ("Softmax:16x1000:axis1", None, {"x": 127, "y": 3}),
        # Log's zero points are fixed by the template
        ("Log:16x1000", None, {"x": 1, "y": 255}),
        ("ReduceMean:16x512x7x7:axes2,3:k1", None, {"x": 5, "y": 0}),
    ],
)
def test_tail_ops_refuse_unmeasured_targets(key, scales, zps):
    with pytest.raises(ValueError):
        mre.emit_model(key, scales, zps)


def test_greater_cast_is_calibration_free():
    s1 = _mcode("teng_register_census/gtcast_s1.axmodel.gz")
    asym = _mcode("teng_register_census/gtcast_asym.axmodel.gz")
    assert s1 != asym
    assert mre.normalized_records(s1) == mre.normalized_records(asym)
    model = mre.emit_model("GreaterCast:1x64x56x56")
    assert bytes(mre.mcode_initializer(model).raw_data) == s1
    with pytest.raises(ValueError):
        mre.emit_model("GreaterCast:1x64x56x56", scales={"x": 0.1})


def test_emit_model_retargets_template():
    key = "ReduceSum:16x1x64x576:axes0:k0"
    meta = INDEX[key]
    scales = {"x": 0.01, "y": 0.2}
    zps = {"x": 120, "y": 135}
    mc = bytes(mre.mcode_initializer(mre.emit_model(key, scales, zps)).raw_data)
    want = mre.lane_values("ReduceSum", scales)
    runs = {v for _, v in mre.lane_runs(mre._chunks(mre.suc.decode_segments(mc)[2]))}
    assert set(want.values()) <= runs
    assert not set(mre.lane_values("ReduceSum", meta["scales"]).values()) & runs


@pytest.mark.parametrize(
    "scales,zps",
    [
        ({"x": 0.5, "y": 2.0}, None),  # 1/s_x == s_y: lanes indistinguishable
        ({"x": 0.02, "y": 0.02}, {"x": 0, "y": 0}),  # zp_x == 0 is unmeasured
    ],
)
def test_reducesum_refuses_unmeasured_targets(scales, zps):
    meta = INDEX["ReduceSum:1x1x64x3136:axes2:k0"]
    mc = _mcode(meta["file"])
    with pytest.raises(ValueError):
        mre.retarget(
            mc,
            "ReduceSum",
            meta["scales"],
            scales,
            meta["zero_points"],
            zps or meta["zero_points"],
        )


def test_sqrt_refuses_zero_point_change():
    meta = INDEX["Sqrt:512x512x3x3"]
    with pytest.raises(ValueError):
        mre.retarget(
            _mcode(meta["file"]),
            "Sqrt",
            meta["scales"],
            meta["scales"],
            meta["zero_points"],
            {"x": 3, "y": 0},
        )


def test_unknown_template_is_an_error():
    with pytest.raises(ValueError):
        mre.emit_model("ReduceSum:3x5x7:axes0:k0")


def test_step_node_keys(tmp_path):
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 13]>
        g (float[16,1,64,576] a, float[16,64,56,56] x, float[512,512,3,3] w)
            => (float[1,64,576] r, float[16,64,56,56] c, float[512,512,3,3] s) {
            ax = Constant <value = int64[1] {0}> ()
            r = ReduceSum <keepdims = 0> (a, ax)
            zero = Constant <value = float {0.0}> ()
            g = Greater (x, zero)
            c = Cast <to = 1> (g)
            s = Sqrt (w)
        }
        """
    )
    path = str(tmp_path / "step.onnx")
    onnx.save(model, path)
    assert mre.step_node_keys(path) == [
        ("ReduceSum", "ReduceSum:16x1x64x576:axes0:k0"),
        ("Greater", "GreaterCast:16x64x56x56"),
        ("Cast", "GreaterCast:16x64x56x56"),
        ("Sqrt", "Sqrt:512x512x3x3"),
    ]
    cov = mre.coverage(path)
    assert cov["ReduceSum"] == {"nodes": 1, "covered": 1, "missing": {}}
    assert cov["Sqrt"]["covered"] == 1
    assert cov["Greater"] == {"nodes": 1, "covered": 1, "missing": {}}


with open(os.path.join(FIXTURES, "misc_op_neg", "sweep.json")) as _f:
    NEG = json.load(_f)
# a template must write every zero-point record (both zero points nonzero);
# any build of the same program is a target
NEG_PAIRS = [
    (a, b)
    for a, b in itertools.permutations(sorted(NEG), 2)
    if NEG[a]["program"] == NEG[b]["program"]
    and 0 not in NEG[a]["zero_points"].values()
]


def test_neg_program_is_picked_by_the_scale():
    # 1/s_x >= 64 compiles the small program, below it the large one; zero
    # points 0..255 occur on both sides
    for meta in NEG.values():
        assert mre.neg_program(meta["scales"]["x"]) == meta["program"]
    zps = {
        p: {m["zero_points"]["x"] for m in NEG.values() if m["program"] == p}
        for p in ("small", "large")
    }
    assert {0, 255} <= zps["small"] and {0, 255} <= zps["large"]


@pytest.mark.parametrize("src,dst", NEG_PAIRS)
def test_neg_matches_every_build_of_its_program(src, dst):
    # lanes 1/s_x and s_x; zero points on 0x1a90/0x1ad0/0x1b10, with writes of
    # an unchanged value omitted (zp 0 and 255 targets drop records) and the
    # segment re-padded
    a, b = NEG[src], NEG[dst]
    got = mre.retarget(
        _mcode(f"misc_op_neg/{src}"),
        "Neg",
        a["scales"],
        b["scales"],
        a["zero_points"],
        b["zero_points"],
    )
    assert mre.normalized_records(got) == mre.normalized_records(
        _mcode(f"misc_op_neg/{dst}")
    )


def test_neg_emit_model_picks_the_program():
    # the step's Neg: a cross-entropy input (<= 0) calibrates to zp_x = 255
    for name, meta in NEG.items():
        if meta["zero_points"]["x"] != 255:
            continue
        model = mre.emit_model("Neg:1x1", meta["scales"], meta["zero_points"])
        got = bytes(mre.mcode_initializer(model).raw_data)
        want = _mcode(f"misc_op_neg/{name}")
        assert mre.normalized_records(got) == mre.normalized_records(want)


@pytest.mark.parametrize(
    "scales,zps",
    [
        ({"x": 0.02, "y": 0.02}, {"x": 100, "y": 100}),  # zp_y != 255 - zp_x
        ({"x": 0.02, "y": 0.03}, {"x": 100, "y": 155}),  # s_y != s_x
    ],
)
def test_neg_refuses_non_neg_calibrations(scales, zps):
    with pytest.raises(ValueError):
        mre.emit_model("Neg:1x1", scales, zps)


def test_neg_program_boundary_is_one_sixty_fourth():
    assert mre.neg_program(0.01562) == "small"
    assert mre.neg_program(2.0**-6) == "large"
    for meta in NEG.values():
        s = meta["scales"]["x"]
        assert (meta["program"] == "small") == (s < 2.0**-6)


@pytest.mark.parametrize("key,alt", sorted(mre.REDUCESUM_EQUIVALENTS.items()))
def test_reducesum_equivalent_reduces_the_same_bytes(key, alt):
    # Pulsar2 can't tile the node as written (also not inside the step's own
    # Reshape -> ReduceSum -> Reshape chain); the equivalent reduces the same
    # contiguous elements into the same contiguous output
    def parts(k):
        _, shape, axes, _ = k.split(":")
        dims = [int(d) for d in shape.split("x")]
        red = [int(a) for a in axes[4:].split(",")]
        kept = [d for i, d in enumerate(dims) if i not in red]
        return dims, red, kept

    import numpy as np

    d0, r0, k0 = parts(key)
    d1, r1, k1 = parts(alt)
    x = np.arange(int(np.prod(d0)), dtype=np.int64) % 97
    assert np.array_equal(
        x.reshape(d0).sum(axis=tuple(r0)).ravel(),
        x.reshape(d1).sum(axis=tuple(r1)).ravel(),
    )
    assert mre.equivalent_key(key) == alt
    meta = INDEX[alt]
    got = mre.emit_model(key, meta["scales"], meta["zero_points"])
    want = _mcode(meta["file"])
    assert mre.normalized_records(bytes(mre.mcode_initializer(got).raw_data)) == (
        mre.normalized_records(want)
    )


_RELAYOUT = os.path.join(FIXTURES, "misc_op_record_emit", "relayout")
_NOISE = range(301, 400)  # segment 0 slot table: a per-build permutation


@pytest.mark.parametrize(
    "name", sorted(json.load(open(os.path.join(_RELAYOUT, "index.json"))))
)
def test_fused_chain_at_an_extreme_ratio_matches_native_bytes(name):
    # The C64 ReduceSum -> flatten chain at the step's s_y/s_x of ~460 re-encodes
    # 32 bytes shorter. relayout_segment used to shift the scalar header word
    # 0x1000 at byte 72 as if it were an offset; the model then ran 48-71 LSB
    # off (or faulted) on the device. Pulsar2 keeps it:
    # docs/axera-reshape-signed-templates.md.
    meta = json.load(open(os.path.join(_RELAYOUT, "index.json")))[name]
    with gzip.open(os.path.join(_RELAYOUT, name)) as f:
        native = onnx.load_model_from_string(f.read())
    got = bytes(
        mre.mcode_initializer(
            mre.emit_model(meta["key"], meta["scales"], meta["zero_points"])
        ).raw_data
    )
    want = bytes(mre.mcode_initializer(native).raw_data)
    assert len(got) == len(want)
    assert [i for i in range(len(got)) if got[i] != want[i] and i not in _NOISE] == []
