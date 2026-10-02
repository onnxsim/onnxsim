"""No-device checks for matmul_record_emit: recalibrating a live-operand
MatMul template from scales alone must reproduce a native Pulsar2 build at
the target calibration, record for record and npu_params byte for byte, and
must refuse what it cannot explain."""

import gzip
import json
import os
import struct
import sys

import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import matmul_record_emit as mre  # noqa: E402
import short_unit_codec as codec  # noqa: E402
import step_recalibrate as sr  # noqa: E402

FIX = os.path.join(_AXERA, "fixtures")
HERE = os.path.join(FIX, "matmul_record_emit")
GATHER = os.path.join(FIX, "gather_aggregate_real")


def _build(model, quant):
    return mre.load_model(model), mre.load_scales(quant)


SMALL = {
    name: _build(
        os.path.join(HERE, f"matmul_2x4x8x8_{name}.axmodel.gz"),
        os.path.join(HERE, f"matmul_2x4x8x8_{name}.quant.json.gz"),
    )
    for name in ("acal_narrow", "acal_wide")
}
# The step's conv5 im2col MatMul, [1,1,512,4608] x [16,1,4608,49], in its
# real Gather -> Mul -> Reshape -> MatMul chain, at two calibrations.
STEP_CHAIN = {
    "a": _build(
        os.path.join(GATHER, "a_reference_narrow.axmodel.gz"),
        os.path.join(HERE, "gather_a_reference.quant.json.gz"),
    ),
    "c": _build(
        os.path.join(GATHER, "c_wide_reference.axmodel.gz"),
        os.path.join(HERE, "gather_c_wide_reference.quant.json.gz"),
    ),
}
# The step's classifier: MatMul [16,512] x [512,1000] + live bias [1000].
GEMM = _build(
    os.path.join(HERE, "matmul_add_16x512x1000.axmodel.gz"),
    os.path.join(HERE, "matmul_add_16x512x1000.quant.json.gz"),
)


def _exact(report):
    assert report["structure"]
    assert report["record_diffs"] == []
    assert report["params_diff_bytes"] == 0


@pytest.mark.parametrize(
    "src,dst", [("acal_narrow", "acal_wide"), ("acal_wide", "acal_narrow")]
)
def test_small_matmul_recalibrates_to_native_build(src, dst):
    (model, old), (want, new) = SMALL[src], SMALL[dst]
    assert old != new
    out, report = mre.recalibrate(model, old, new)
    assert report["segments_changed"] == [2]
    _exact(mre.compare(out, want))


@pytest.mark.parametrize("src,dst", [("a", "c"), ("c", "a")])
def test_step_shape_chain_recalibrates_to_native_build(src, dst):
    (model, old), (want, new) = STEP_CHAIN[src], STEP_CHAIN[dst]
    assert old["y"] != new["y"]
    out, report = mre.recalibrate(model, old, new)
    assert report["records"] == 890
    assert report["params_words"] == 98
    got = mre.compare(out, want)
    _exact(got)
    # Only segment 0's tail rotation differs, as between any two rebuilds.
    assert got["noise_diffs"] == 4


def test_gemm_template_every_lane_explained():
    model, scales = GEMM
    found = mre.locate(model, scales)
    assert all(roles for *_, roles in found["records"])
    kinds = {
        min((r for r in roles), key=lambda r: mre.PRECEDENCE.index(r[0]))
        for *_, roles in found["records"]
    }
    assert {("inv32", "a"), ("inv32", "b"), ("inv32", "c"), ("s", "z")} <= kinds
    assert {("zp", "z"), ("zp", "c")} <= kinds
    # The live bias Add's zero-point offset lanes and Q-format (#1869's
    # zp_offset, here with k = 0), and its Q15 header word in npu_params.
    # (x and z commute in both, so both orders explain them.)
    every = {r for *_, roles in found["records"] for r in roles}
    assert {("zpoff", "t", "c", "z"), ("qshift", "t", "c", "z")} <= every
    lanes = {tuple(roles[:1]) for _, _, roles in found["params"]}
    assert lanes == {
        (("zpf", "t"),),
        (("mult", "a", "b", "t"),),
        (("q15", "t", "c", "z"),),
    }
    assert len(found["params"]) == 2 * 520 + 1


def test_gemm_template_identity_and_round_trip():
    model, scales = GEMM
    same, _ = mre.recalibrate(model, scales, scales)
    assert sr.get_mcode(same) == sr.get_mcode(model)
    assert mre.params_of(same) == mre.params_of(model)

    moved = {t: (s * 1.37, z) for t, (s, z) in scales.items()}
    moved["z"] = (moved["z"][0], 120.0)
    out, _ = mre.recalibrate(model, scales, moved)
    lanes = mre.locate(out, moved)
    assert all(roles for *_, roles in lanes["records"])
    zp = [v for _, _, reg, v, _ in lanes["records"] if reg == 0x1A90]
    assert zp == [120]
    back, _ = mre.recalibrate(out, moved, scales)
    assert codec.decode_segments(sr.get_mcode(back)) == codec.decode_segments(
        sr.get_mcode(model)
    )
    assert mre.params_of(back) == mre.params_of(model)


def test_refuses_zero_point_crossing_and_missing_tensor():
    model, scales = GEMM
    to_zero = dict(scales, z=(scales["z"][0], 0.0))
    with pytest.raises(mre.CalibrationError, match="zero and nonzero"):
        mre.recalibrate(model, scales, to_zero)
    with pytest.raises(mre.CalibrationError, match="miss"):
        mre.recalibrate(model, scales, {t: v for t, v in scales.items() if t != "c"})


def test_refuses_unexplained_and_ambiguous_values():
    scales = {"p": (0.5, 0.0), "q": (0.5, 0.0)}
    with pytest.raises(mre.CalibrationError, match="no scale formula"):
        mre._new_value("x", 0x12345678, [], scales)
    # 1/s_p and 1/s_q agree at the template and part at the new scales.
    old = struct.unpack("<I", struct.pack("<f", 2.0))[0]
    roles = [("inv32", "p"), ("inv32", "q")]
    assert mre._new_value("x", old, roles, scales) == old
    with pytest.raises(mre.CalibrationError, match="ambiguous"):
        mre._new_value("x", old, roles, {"p": (0.5, 0.0), "q": (0.25, 0.0)})


def test_step_shape_tables_cover_the_step():
    assert sum(n for n, _ in mre.STEP_SHAPES.values()) == 42  # 41 MatMul + Gemm
    assert sum(mre.STEP_CONV_MATMULS.values()) == 20


def test_requantize_offset_words():
    # A weight slice (asymmetric, zp 127) into its taps' Concat (symmetric):
    # at the same scale the shift stays at 15 (0x8f) and the offset is
    # -127 * 2**15, as the step's stage4 conv1 build stores them.
    same = {"w": (0.0019607842, 127.0), "cat": (0.0019607842, 0.0)}
    assert mre.evaluate(("rqshift", "w", "cat"), same) == 0x8F
    assert mre.evaluate(("rqoff", "w", "cat"), same) == 0xFFC08000
    above = {"w": (0.0031372542, 128.0), "cat": (0.0031372522, 0.0)}
    assert mre.evaluate(("rqshift", "w", "cat"), above) == 0x8E


def test_concat_header_half_above_one_is_shifted():
    # stage2 conv1's taps are an unfused Relu's output: they share the wider
    # pre-activation quantization, so their ratio into the Concat is 1.0991
    # and the header's activation half is Q14 (18008); the weight half is
    # still Q15. Build: 0x708d4658 at npu_params offset 32.
    sc = {
        "x": (0.024485019966959953, 139.0),
        "xcat": (0.022276567295193672, 0.0),
        "w": (0.0056243580766022205, 110.0),
        "wcat": (0.006396328564733267, 0.0),
    }
    assert mre._q15_shift("x", "xcat", sc) == 1
    assert mre.evaluate(("cat15", "x", "xcat", "w", "wcat", 1, 0), sc) == 0x708D4658
    # Shifts 1 and 2 are one program (the sweep in
    # docs/axera-conv-concat-shift.md); across 1 the record count changes.
    wider = {**sc, "xcat": (sc["x"][0] / 2.3, 0.0)}
    assert mre._q15_shift("x", "xcat", wider) == 2
    mre.evaluate(("cat15", "x", "xcat", "w", "wcat", 1, 0), wider)
    below = {**sc, "xcat": (sc["x"][0] / 0.9, 0.0)}
    with pytest.raises(mre.CalibrationError, match="other side of 1"):
        mre.evaluate(("cat15", "x", "xcat", "w", "wcat", 1, 0), below)


def test_concat_header_found_once_in_every_conv_template():
    man = mre.step_manifest()
    for name, meta in man["templates"].items():
        if meta["kind"] != "conv":
            continue
        d = mre.STEP_TEMPLATE_DIR
        model = mre.load_model(os.path.join(d, meta["axmodel"]))
        scales = mre.load_scales(os.path.join(d, meta["quant"]))
        found = mre.locate(model, scales)
        cats = [
            r for _, _, roles in found["params"] for r in roles[:1] if r[0] == "cat15"
        ]
        assert len(cats) <= 1, name
        want = meta.get("concat_shift_class")
        if want and cats:
            assert (cats[0][5] == 0) == (want == "k0"), name


# A bare live-operand [1,64,128] x [1,128,64] MatMul at layer precision U16
# (``pilot_matmul_u16.py``), built at two calibrations. The 16-bit path adds
# fixed 256.0/1.0 lanes and scales the npu_params multiplier lane by 256.
U16 = {
    name: _build(
        os.path.join(HERE, f"matmul_1x64x128x64_u16_{name}.axmodel.gz"),
        os.path.join(HERE, f"matmul_1x64x128x64_u16_{name}.quant.json.gz"),
    )
    for name in ("a", "b")
}


@pytest.mark.parametrize("src,dst", [("a", "b"), ("b", "a")])
def test_u16_matmul_recalibrates_to_the_native_build(src, dst):
    out, _ = mre.recalibrate(U16[src][0], U16[src][1], U16[dst][1])
    diff = mre.compare(out, U16[dst][0])
    assert diff["record_diffs"] == []
    assert diff["params_diff_bytes"] == 0


# Real step chains at layer precision U16, each built at two calibrations
# (``probe_u16_recalibrate.py``): a bare MatMul (dX_MatMul_240) and the forward
# Conv chain (Transpose, Slice, Reshape, MatMul, bias Add) of stage2_conv2.
U16_CHAINS = {
    chain: {
        name: _build(
            os.path.join(HERE, f"u16chain_{chain}_{name}.axmodel.gz"),
            os.path.join(HERE, f"u16chain_{chain}_{name}.quant.json.gz"),
        )
        for name in ("A", "B")
    }
    for chain in ("dx240", "convfwd")
}


@pytest.mark.parametrize("chain", ["dx240", "convfwd"])
@pytest.mark.parametrize("src,dst", [("A", "B"), ("B", "A")])
def test_u16_step_chain_recalibrates_to_the_native_build(chain, src, dst):
    builds = U16_CHAINS[chain]
    out, _ = mre.recalibrate(builds[src][0], builds[src][1], builds[dst][1])
    diff = mre.compare(out, builds[dst][0])
    assert diff["record_diffs"] == []
    assert diff["params_diff_bytes"] == 0


# stage2_conv0's forward chain (a strided 3x3 Conv: Pad, nine taps concatenated,
# a weight requantized from asymmetric to symmetric, MatMul, bias Add) at U16, built
# at several calibrations (``/tmp``-free: the builds are fixtures). v0/v1/x3/x4 have
# nonzero MatMul and output zero points (v0 and v1 share the MatMul and output scale,
# the case that tied roles), v2/v4 have zero ones, x0/x1 an asymmetric activation
# (16-bit zero points stored twice in each word of ``npu_params``).
CONV3X3 = {
    name: _build(
        os.path.join(HERE, f"u16conv3x3_{name}.axmodel.gz"),
        os.path.join(HERE, f"u16conv3x3_{name}.quant.json.gz"),
    )
    for name in ("v0", "v1", "v2", "v4", "x0", "x1", "x3", "x4")
}


@pytest.mark.parametrize(
    "src,dst",
    [
        ("v0", "v1"),
        ("v1", "v0"),
        ("v0", "x3"),  # a template with a tied scale moves to an untied one
        ("v1", "x4"),
        ("x3", "x4"),
        ("x4", "v0"),
        ("v2", "v4"),  # zero MatMul and output zero points
        ("v4", "v2"),
        ("x0", "x1"),  # an asymmetric activation: zp16 words in npu_params
        ("x1", "x0"),
    ],
)
def test_u16_conv3x3_chain_recalibrates_to_the_native_build(src, dst):
    out, _ = mre.recalibrate(CONV3X3[src][0], CONV3X3[src][1], CONV3X3[dst][1])
    diff = mre.compare(out, CONV3X3[dst][0])
    assert diff["record_diffs"] == []
    assert diff["params_diff_bytes"] == 0


def test_u16_conv3x3_chain_refuses_a_zero_point_layout_change():
    with pytest.raises(mre.CalibrationError, match="zero point goes between"):
        mre.recalibrate(CONV3X3["v0"][0], CONV3X3["v0"][1], CONV3X3["v2"][1])


def test_derive_ranges_gives_every_member_of_a_measured_group_its_range():
    """A measured tensor sets the range of every tensor Pulsar2 groups with it (a
    Pad, Transpose or Slice passes the same values on), so the scales predicted for
    the group move together and by the measured factor."""
    import onnx

    import u16_chain

    quant = os.path.join(HERE, "u16conv3x3_v0.quant.json.gz")
    q = json.loads(gzip.open(quant).read())
    scales = CONV3X3["v0"][1]
    sub = onnx.ModelProto()  # no nodes: only the grouping is exercised
    act = next(t for t in scales if t.endswith("stage1_activation1"))
    group = [t for t in scales if scales[t] == scales[act]]
    assert len(group) > 3  # the activation, its Pad/Transpose/Slice outputs
    lo, hi = (
        -2.0 * 32767.5 * scales[act][0],
        2.0 * 32767.5 * scales[act][0],
    )  # 2x its range
    derived = u16_chain.derive_ranges(sub, q, scales, {act: (lo, hi)})
    assert all(derived[t] == (lo, hi) for t in group)
    pred = u16_chain.predict_scales16(q, derived)
    for t in group:
        assert pred[t][0] == pytest.approx(2.0 * scales[t][0], rel=1e-6)
    # a group with no measured member keeps its template scale
    rest = [t for t in scales if scales[t] != scales[act] and t in pred]
    assert rest and all(
        pred[t][0] == pytest.approx(scales[t][0], rel=1e-6) for t in rest
    )
