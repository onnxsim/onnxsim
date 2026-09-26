"""No-device tests for step_calibration.py: offline prediction of Pulsar2's
per-tensor activation quantization, and coverage settled against it.

The fixtures are real Pulsar2 7.0-lite builds: each graph (weights dropped),
its MinMax ranges over its own calibration set, and the build's
``quant_axmodel.json``. ``assign`` must reproduce every tensor Pulsar2
quantized, which is what makes a predicted step calibration trustworthy.
"""

import gzip
import json
import os
import sys

import onnx
import pytest
from onnx import parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import step_calibration as sc  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures", "step_calibration")
_STEP_OPS = os.path.join(
    _AXERA_DIR, "fixtures", "tinygrad_ax_backend", "resnet18_step_ops.json.gz"
)
_STEP_CALIB = os.path.join(_FIX, "resnet18_step_calibration.json.gz")

with open(os.path.join(_FIX, "ranges.json")) as _f:
    _RANGES = json.load(_f)


def _model(body, opset=13):
    return parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')


def _probe(name):
    with gzip.open(os.path.join(_FIX, f"{name}.onnx.gz"), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    with gzip.open(os.path.join(_FIX, f"{name}.quant.json.gz"), "rt") as f:
        quant = json.load(f)
    ranges = {k: tuple(v) for k, v in _RANGES[name]["ranges"].items()}
    return model, ranges, quant


def test_qparams_minmax():
    # asymmetric uint8 over the range widened to include 0
    assert sc.qparams(-1.0, 2.0) == pytest.approx((3.0 / 255, 85))
    assert sc.qparams(0.5, 2.0) == pytest.approx((2.0 / 255, 0))
    # symmetric int8 (MatMul inputs): max|x| / 127.5, zero point 0
    assert sc.qparams(-0.35, 0.5, symmetric=True) == pytest.approx((0.5 / 127.5, 0))


@pytest.mark.parametrize("name", sorted(_RANGES))
def test_assign_reproduces_pulsar2(name):
    model, ranges, quant = _probe(name)
    compared, misses = sc.compare_to_pulsar2(model, ranges, quant)
    assert compared > 0
    assert misses == []


def _assign(body, ranges):
    return sc.assign(_model(body), ranges)


def test_tensor_feeding_only_matmuls_is_symmetric_int8():
    q = _assign(
        """g (float[4,8] a, float[4,8] b, float[8,2] w) => (float[4,2] y) {
            m = Mul(a, b)
            y = MatMul(m, w)
        }""",
        {"a": (-1, 2), "b": (-1, 1), "w": (-1, 1), "m": (-0.5, 1.0), "y": (-3, 1)},
    )
    assert q["m"] == {
        "scale": pytest.approx(1.0 / 127.5),
        "zero_point": 0,
        "signed": True,
    }
    assert not q["y"]["signed"]


def test_mixed_use_tensor_stays_uint8_with_a_consumer_requant():
    q = _assign(
        """g (float[4,8] a, float[4,8] b, float[8,2] w) => (float[4,2] y, float[8] r) {
            m = Mul(a, b)
            y = MatMul(m, w)
            ax = Constant <value = int64[1] {0}> ()
            r = ReduceSum <keepdims = 0> (m, ax)
        }""",
        {
            "a": (-1, 2),
            "b": (-1, 1),
            "w": (-1, 1),
            "m": (-0.5, 1.0),
            "y": (-3, 1),
            "r": (-2, 4),
        },
    )
    assert not q["m"]["signed"] and q["m"]["zero_point"] == 85
    assert q["m"]["consumer_int8_scale"] == pytest.approx(1.0 / 127.5)


def test_passive_op_into_matmuls_requantizes_a_mixed_use_input():
    # m has a ReduceSum use, so it stays uint8; the Reshape between m and the
    # MatMul only feeds the MatMul, so Pulsar2 requantizes there: the Reshape
    # output gets its own symmetric int8 parameters instead of overlapping m
    # (the mixed_reshape_mm build).
    q = _assign(
        """g (float[16,64] a, float[16,64] b, float[64,32] w) => (float[4,4,32] y1, float[64] y2) {
            m = Mul(a, b)
            s = Constant <value = int64[3] {4, 4, 64}> ()
            r = Reshape(m, s)
            y1 = MatMul(r, w)
            ax = Constant <value = int64[1] {0}> ()
            y2 = ReduceSum <keepdims = 0> (m, ax)
        }""",
        {
            "a": (-1, 2),
            "b": (-1, 1),
            "w": (-1, 1),
            "m": (-0.5, 1.0),
            "r": (-0.5, 1.0),
            "y1": (-3, 1),
            "y2": (-2, 4),
        },
    )
    assert not q["m"]["signed"] and q["m"]["zero_point"] == 85
    assert q["r"] == {
        "scale": pytest.approx(1.0 / 127.5),
        "zero_point": 0,
        "signed": True,
    }


def test_relu_shares_its_input_quantization_unless_it_fuses():
    body = """g (float[8] a, float[8] b) => (float[8] r{extra_out}) {{
        s = Add(a, b)
        r = Relu(s){extra}
    }}"""
    ranges = {"a": (-1, 2), "b": (-1, 1), "s": (-2, 3), "r": (0, 3), "c": (0, 1)}
    fused = _assign(body.format(extra_out="", extra=""), ranges)
    # the Add's only consumer is the Relu: both take the Relu output's range
    assert fused["s"]["zero_point"] == fused["r"]["zero_point"] == 0
    shared = _assign(
        body.format(
            extra_out=", float[8] c",
            extra="\n        z = Constant <value = float {0.0}> ()"
            "\n        gt = Greater(s, z)\n        c = Cast <to = 1> (gt)",
        ),
        ranges,
    )
    # a second consumer (the backward mask) keeps them apart: Relu is passive
    assert shared["r"]["zero_point"] == shared["s"]["zero_point"] == 102


def _step_records():
    with gzip.open(_STEP_OPS, "rt") as f:
        return json.load(f)


def _record(name):
    return next(r for r in _step_records() if r["name"] == name)


def _calib(zp, signed=False, scale=0.02):
    """Every tensor at one quantization (a synthetic calibration)."""

    class _All(dict):
        def get(self, _key, _default=None):
            return {"scale": scale, "zero_point": zp, "signed": signed}

    return {"tensors": _All()}


def test_plan_at_calibration_settles_zero_point_classes():
    relu = _record("resnetv15_stage4_relu0_fwd")
    assert axb.plan_node(relu)[0] == "conditional"
    assert axb.plan_at_calibration(relu, _calib(128))[0] == "covered"
    status, detail = axb.plan_at_calibration(relu, _calib(128, signed=True))
    assert status == "refused" and "symmetric int8" in detail
    reshape = next(
        r
        for r in _step_records()
        if r["op"] == "Reshape" and axb.plan_node(r)[1].startswith("Reshape step")
    )
    assert axb.plan_at_calibration(reshape, _calib(97))[0] == "covered"
    status, detail = axb.plan_at_calibration(reshape, _calib(0))
    assert status == "refused" and "zero point is 0" in detail


def _gz_mcode(path):
    with gzip.open(path, "rb") as f:
        return axb.rre.mcode_of(onnx.load_model_from_string(f.read()))


_RELU_PROBES = os.path.join(
    _AXERA_DIR, "fixtures", "elementwise_scale_emit", "zero_point_probes"
)
with open(os.path.join(_RELU_PROBES, "index.json")) as _f:
    _RELU = json.load(_f)


@pytest.mark.parametrize("name", sorted(_RELU))
def test_relu_zero_point_retarget_matches_native_builds(name):
    # The x128,y128 template moved to a native build's (scale, zero point)
    # is that build, record for record, and back again.
    p = _RELU[name]
    model, meta = axb.ew.load_template("Relu", p["shape"], {"x": 128, "y": 128})
    tmpl = axb.rre.mcode_of(model)
    want = _gz_mcode(os.path.join(_RELU_PROBES, name))
    got = axb.ew.retarget_relu_records(tmpl, p["scale"], p["zero_point"])
    cmp = axb.rre.compare_mcode(got, want)
    assert cmp["same_length"] and not cmp["record_diffs"] and not cmp["tail_byte_diffs"]
    back = axb.ew.retarget_relu_records(want, meta["scales"]["x"], 128)
    cmp = axb.rre.compare_mcode(back, tmpl)
    assert cmp["same_length"] and not cmp["record_diffs"] and not cmp["tail_byte_diffs"]
    with pytest.raises(ValueError):
        axb.ew.retarget_relu_records(tmpl, p["scale"], 0)


@pytest.mark.parametrize("src,dst", [("a", "b"), ("b", "a")])
def test_reshape_zero_point_0_template_matches_second_build(src, dst):
    entry = axb.rre.step_template_zp0([1024, 28224], [1024, 9, 3136])
    files = {"a": entry, "b": entry["check"]}
    d = os.path.dirname(entry["axmodel"])
    a = _gz_mcode(os.path.join(d, os.path.basename(files[src]["axmodel"])))
    b = _gz_mcode(os.path.join(d, os.path.basename(files[dst]["axmodel"])))
    cmp = axb.rre.compare_mcode(axb.rre.retarget_scale(a, files[dst]["scale"]), b)
    assert cmp["same_length"] and not cmp["record_diffs"] and not cmp["tail_byte_diffs"]


def test_misc_zero_point_variants_are_picked_by_calibration():
    maxpool = _record("resnetv15_pool0_fwd")
    status, detail = axb.plan_at_calibration(maxpool, _calib(108))
    assert status == "covered" and ":zp " in detail
    assert "p1,1,1,1 (" in axb.plan_at_calibration(maxpool, _calib(0))[1]
    rs = _record("ReduceSum_452")
    assert ":zp0" in axb.plan_at_calibration(rs, _calib(0))[1]
    assert ":zp0" not in axb.plan_at_calibration(rs, _calib(120))[1]


def test_coverage_at_the_predicted_step_calibration():
    calib = axb.load_calibration(_STEP_CALIB)
    report = axb.coverage_report(_step_records(), calibration=calib)
    assert report["nodes"] == 1104
    # docs/axera-step-real-calibration.md
    assert report["totals"] == _EXPECTED_TOTALS
    for op, counts in _EXPECTED_PER_OP.items():
        assert report["per_op"][op] == counts, op


_EXPECTED_TOTALS = {"covered": 570, "refused": 534}
_EXPECTED_PER_OP = {
    "Add": {"covered": 43, "refused": 101},
    "Conv": {"covered": 20},
    "Div": {"covered": 1, "refused": 51},
    "Gemm": {"covered": 1},
    "Log": {"covered": 2},
    "MatMul": {"covered": 41},
    "MaxPool": {"covered": 1},
    # Live broadcast binary templates cover the 42 calibrated broadcast Muls
    # that were previously refused by the standalone-shape check.
    "Mul": {"covered": 61, "refused": 336},
    "ReduceMean": {"covered": 1},
    "Neg": {"covered": 2},
    "ReduceSum": {"covered": 44},
    "Relu": {"covered": 17},
    "Reshape": {"covered": 170},
    "Softmax": {"covered": 3},
    "Sqrt": {"covered": 42},
    "Sub": {"refused": 46},
}


def test_bias_flatten_reshapes_settle_through_their_reducesum_chain():
    # Each [1,C] -> [C] bias-gradient Reshape compiles fused with the
    # ReduceSum producing it; its template is that chain's build (or, for
    # Reshape_475, ReduceSum_474's same-bytes equivalent, which already writes
    # the flattened [64])
    calib = axb.load_calibration(_STEP_CALIB)
    flat = [
        r for r in _step_records() if r["op"] == "Reshape" and "fused_key" in r["attrs"]
    ]
    assert len(flat) == 18
    for rec in flat:
        status, detail = axb.plan_at_calibration(rec, calib)
        assert status == "covered", (rec["name"], detail)
        assert "fused chain" in detail
