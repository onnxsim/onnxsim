"""No-device checks for the Add/Sub/Mul/Div scale-retarget emitter.

Each committed oracle is a native Pulsar2 build at a calibration held out from
its template. Emitting that calibration from the template must reproduce the
oracle's ``npu_params`` and every decompressed MCode record of segments 1..
(segment 0 is rebuild noise; the node input order is normalized). Where the
oracle was compiled with the template's input order and Pulsar2's LZ77 parse
matches the codec's, the whole blob outside segment 0 must match byte for
byte too."""

import gzip
import json
import os
import sys

import onnx
import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import binary_op_scale_emit as B  # noqa: E402
import binary_op_scale_validate as V  # noqa: E402

_ORACLES = os.path.join(B.TEMPLATE_DIR, "oracles")
with open(os.path.join(_ORACLES, "index.json")) as _f:
    _ORACLE_INDEX = json.load(_f)
with open(os.path.join(B.TEMPLATE_DIR, "index.json")) as _f:
    _TEMPLATE_INDEX = json.load(_f)


def _load_gz(path):
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


@pytest.mark.parametrize("oracle", sorted(_ORACLE_INDEX))
def test_emit_reproduces_held_out_native_build(tmp_path, oracle):
    meta = _ORACLE_INDEX[oracle]
    out = tmp_path / "emitted.axmodel"
    B.emit(meta["op"], meta["shape"], meta["scales"], meta["zero_points"], str(out))
    emitted = onnx.load(str(out), load_external_data=False)
    native = _load_gz(os.path.join(_ORACLES, oracle))
    r = V.compare(emitted, native)
    assert r["params"] and r["segments"], r
    if meta["byte_exact_outside_segment0"]:
        assert r["stream_equal_len"] and r["blob_diff_outside_seg0"] == 0, r
        assert r["dims"]


@pytest.mark.parametrize("key", sorted(_TEMPLATE_INDEX))
def test_every_template_has_an_oracle(key):
    meta = _TEMPLATE_INDEX[key]
    assert any(
        o["op"] == meta["op"]
        and o["shape"] == meta["shape"]
        and o["zero_points"] == meta["zero_points"]
        for o in _ORACLE_INDEX.values()
    )


# (op, scales, zero points, npu_params header words, nonzero 0x1ea0 values,
#  nonzero 0x1ef0 offset values) read from native Pulsar2 builds
_NATIVE = [
    # a ratio just below 1 keeps k = 0 and stores 32768
    (
        "Sub",
        {
            "x": 0.007843137718737125,
            "z": 0.01568627543747425,
            "y": 0.015686485916376114,
        },
        {"x": 64, "z": 191, "y": 53},
        [16384, 32768],
        15,
        6946746,
    ),
    (
        "Sub",
        {
            "x": 0.01568627543747425,
            "z": 0.003921568859368563,
            "y": 0.019127050414681435,
        },
        {"x": 128, "z": 128, "y": 127},
        [26873, 6718],
        15,
        1581694,
    ),
    # negative offsets are two's complement int32
    (
        "Add",
        {
            "x": 0.007058156654238701,
            "z": 0.0007841793121770024,
            "y": 0.007791136857122183,
        },
        {"x": 127, "z": 128, "y": 127},
        [29685, 3298],
        15,
        -30645,
    ),
    (
        "Add",
        {
            "x": 0.007058057934045792,
            "z": 0.00705726258456707,
            "y": 0.013801783323287964,
        },
        {"x": 127, "z": 128, "y": 127},
        [16757, 16755],
        15,
        -111293,
    ),
    # ratios above 1 shift: k = 3, 0x1ea0 = 12; zero points 0 give no offset
    (
        "Sub",
        {
            "x": 0.0014509804314002395,
            "z": 0.0011372548760846257,
            "y": 0.00031372555531561375,
        },
        {"x": 0, "z": 0, "y": 0},
        [18944, 14848],
        12,
        0,
    ),
]


@pytest.mark.parametrize("op,scales,zps,words,shift_reg,offset", _NATIVE)
def test_add_sub_side_values_match_native(op, scales, zps, words, shift_reg, offset):
    got_words, k = B.q15_header(op, scales)
    assert got_words == words
    assert 15 - k == shift_reg
    assert B.zp_offset(op, scales, zps) == offset


def _any_template(op):
    for meta in _TEMPLATE_INDEX.values():
        if meta["op"] == op:
            return meta
    pytest.skip(f"no {op} template committed")


def test_self_retarget_is_identity():
    for op in B.OPS:
        meta = _any_template(op)
        model, _ = B.load_template(op, meta["shape"], meta["zero_points"])
        out = B.emit_model(
            model, op, meta["scales"], meta["scales"], meta["zero_points"]
        )
        assert out.SerializeToString() == model.SerializeToString()


def test_refuses_coinciding_scale_values():
    meta = _any_template("Mul")
    model, _ = B.load_template("Mul", meta["shape"], meta["zero_points"])
    same = {"x": 0.01, "z": 0.01, "y": 0.02}  # 1/s_x == 1/s_z
    with pytest.raises(ValueError, match="coincide"):
        B.emit_model(model, "Mul", meta["scales"], same, meta["zero_points"])


def test_add_needs_template_zero_points():
    meta = _any_template("Add")
    model, _ = B.load_template("Add", meta["shape"], meta["zero_points"])
    target = {k: v * 1.5 for k, v in meta["scales"].items()}
    with pytest.raises(ValueError, match="zero points"):
        B.emit_model(model, "Add", meta["scales"], target)


def test_refuses_unknown_template(tmp_path):
    with pytest.raises(ValueError, match="no validated template"):
        B.emit(
            "Add",
            [3, 5, 7],
            {"x": 0.01, "z": 0.02, "y": 0.03},
            {"x": 0, "z": 0, "y": 0},
            str(tmp_path / "x.axmodel"),
        )


def test_rank1_shape_is_served_by_the_1xc_template():
    assert B.template_shape([64]) == [1, 64]
    assert B.template_shape([16, 1000]) == [16, 1000]
