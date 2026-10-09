"""No-device checks for ``scripts/axera/linear_emit.py`` on committed native
builds: a linear layer emitted for held-out weights from another build of the
same shape must equal the native build of those weights (``npu_params`` byte
for byte, every record outside segment 0's slot table)."""

import gzip
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import emitter  # noqa: E402
import graph_stitch as gs  # noqa: E402
import linear_emit as le  # noqa: E402

FIXTURES = os.path.join(AXERA, "fixtures", "linear_emit")
with open(os.path.join(FIXTURES, "index.json")) as _f:
    BUILDS = json.load(_f)["builds"]
SMALL = [f"linear64_w{k}" for k in range(1, 7)]
LARGE = ["linear576_w1_pm1", "linear576_w100_pm4"]
CALIBRATION = os.path.join(AXERA, "fixtures", "pulsar_free_calibration")


def _model(name) -> onnx.ModelProto:
    return gs.load_model(os.path.join(FIXTURES, BUILDS[name]["file"]))


def _weights(name) -> np.ndarray:
    """The float weights ``[in, out]`` of a native build. They are regenerated
    from the build's seed (576x576 float32 is 1.3 MB) and checked against the
    committed digest, so a numpy that draws another stream fails here."""
    b = BUILDS[name]
    n = b["width"]
    w = np.random.default_rng(b["weight_seed"]).uniform(-0.1, 0.1, (n, n))
    w = w.astype(np.float32)
    assert hashlib.sha256(w.tobytes()).hexdigest() == b["weight_sha256"], name
    return w


def _params(model) -> bytes:
    init = next(i for i in model.graph.initializer if i.name == "npu_params")
    return bytes(init.raw_data)


def _segments(model):
    mc = bytes(gs.mre.mcode_initializer(model).raw_data)
    return [gs.records(r) for r in gs.suc.decode_segments(mc)]


def _assert_equals_native(got, want):
    assert _params(got) == _params(want)
    a, b = _segments(got), _segments(want)
    assert a[1:] == b[1:]
    # segment 0's slot table is a per-build permutation of the same records
    lo, hi = gs.slot_table(b[0])
    assert a[0][:lo] == b[0][:lo] and a[0][hi:] == b[0][hi:]
    assert sorted(a[0][lo:hi]) == sorted(b[0][lo:hi])
    assert gs.compare_models(got, want) == []


def _emit(template, held):
    b = BUILDS[held]
    return le.emit_linear(
        _model(template), _weights(held), b["scales"], b["zero_points"]
    )


@pytest.mark.parametrize("template,held", list(itertools.permutations(SMALL, 2)))
def test_linear_64_for_held_out_weights(template, held):
    _assert_equals_native(_emit(template, held), _model(held))


@pytest.mark.parametrize("template,held", list(itertools.permutations(LARGE, 2)))
def test_linear_576_for_held_out_weights(template, held):
    # the two committed builds differ in weights and in calibration (inputs of
    # about +-1 and +-4), so this also moves the QUANT job's lanes
    assert BUILDS[template]["scales"]["x"] != BUILDS[held]["scales"]["x"]
    _assert_equals_native(_emit(template, held), _model(held))


@pytest.mark.parametrize("name", SMALL + LARGE)
def test_own_weights_reproduce_the_build(name):
    native = _model(name)
    assert _emit(name, name).SerializeToString() == native.SerializeToString()


@pytest.mark.parametrize("name", SMALL + LARGE)
def test_weight_table_layout(name):
    n = BUILDS[name]["width"]
    table = np.frombuffer(_params(_model(name)), np.uint8)
    assert len(table) == le.params_bytes(n)
    addr = le.weight_addresses(n)
    codes, s_w = le.weight_codes(_weights(name))
    # every code has its own byte (64) or two nibble-plane bytes per input pair
    assert addr.shape == ((64, 64) if n == 64 else (576, 288, 2))
    assert len(np.unique(addr)) == addr.size == n * n
    assert np.array_equal(table[addr], le.weight_bytes(codes))
    # each tile: weight block, then the requantization block
    q = BUILDS[name]
    for t in range(n // le.TILE):
        base = t * le.tile_bytes(n)
        ch = slice(64 * t, 64 * t + 64)
        in_tile = addr[ch]
        assert base <= in_tile.min() and in_tile.max() < base + le.WEIGHT_BYTES[n]
        block = emitter.requant_block(
            codes[ch],
            q["scales"]["x"],
            q["zero_points"]["x"],
            q["scales"]["y"],
            q["zero_points"]["y"],
            s_w[ch],
        )
        at = base + le.WEIGHT_BYTES[n]
        assert table[at : at + le.BLOCK].tobytes() == block.tobytes()
    assert not table[-2 * le.IO_BYTES :].any()


def test_bytes_outside_the_codes_do_not_depend_on_the_weights():
    # 64x64: 512 of the 4608 weight-block bytes hold no code; they are equal
    # in all six builds, so keeping the template's is exact
    tables = [np.frombuffer(_params(_model(n)), np.uint8) for n in SMALL]
    rest = np.ones(le.WEIGHT_BYTES[64], bool)
    rest[le.weight_addresses(64).ravel()] = False
    assert int(rest.sum()) == 512
    assert all(np.array_equal(t[:4608][rest], tables[0][:4608][rest]) for t in tables)
    # 576x576: the two nibble planes fill the weight blocks completely
    assert le.weight_addresses(576).size == 9 * le.WEIGHT_BYTES[576]


def test_a_weight_set_changes_nine_records():
    a, b = _segments(_model(SMALL[0])), _segments(_model(SMALL[1]))
    changed = [
        (si, gs._reg(x)) for si in range(1, 5) for x, y in zip(a[si], b[si]) if x != y
    ]
    assert changed == [(2, gs.REG_ZP_A), *((2, r) for r in gs.LANES_LO)]


def test_any_template_gives_the_same_layer_for_unseen_weights():
    # weights no native build has: the result does not depend on the template
    w = np.random.default_rng(4242).uniform(-0.1, 0.1, (64, 64)).astype(np.float32)
    scales, zps = {"x": 0.0078, "y": 0.0071}, {"x": 128, "y": 131}
    models = [le.emit_linear(_model(n), w, scales, zps) for n in SMALL]
    assert len({_params(m) for m in models}) == 1
    assert all(gs.compare_models(m, models[0]) == [] for m in models)
    table = np.frombuffer(_params(models[0]), np.uint8)
    codes, _ = le.weight_codes(w)
    assert np.array_equal(table[le.weight_addresses(64)], codes)


# ---- calibration from samples ------------------------------------------------------
def _samples():
    with open(os.path.join(CALIBRATION, "index.json")) as f:
        file = json.load(f)["cases"]["linear64_w1"]["samples"]["x"]
    with gzip.open(os.path.join(CALIBRATION, file), "rb") as f:
        return list(np.load(f))


@pytest.mark.parametrize("held", ["linear64_w4", "linear64_w5", "linear64_w6"])
def test_linear_64_from_samples_equals_the_native_build(held):
    # no quant json: the calibration is derived from the samples; on these
    # three builds every scale is bit-exact
    template = "linear64_w1"
    model, scales, zps = le.emit_linear_from_samples(
        _model(template), _weights(held), _samples()
    )
    assert zps == BUILDS[held]["zero_points"]
    assert scales == {k: gs._f32(v) for k, v in BUILDS[held]["scales"].items()}
    _assert_equals_native(model, _model(held))


@pytest.mark.parametrize("held", ["linear64_w1", "linear64_w2", "linear64_w3"])
def test_linear_64_from_samples_with_a_one_ulp_output_scale(held):
    # here the derived s_y is one float32 ulp from Pulsar2's: the eight
    # DEQUANT lanes and the bias/multiplier floats move by an ulp or two
    model, scales, zps = le.emit_linear_from_samples(
        _model("linear64_w4"), _weights(held), _samples()
    )
    want = _model(held)
    assert zps == BUILDS[held]["zero_points"]
    assert scales["x"] == gs._f32(BUILDS[held]["scales"]["x"])
    bits = [gs._f32bits(s) for s in (scales["y"], BUILDS[held]["scales"]["y"])]
    assert abs(bits[0] - bits[1]) == 1
    a, b = _segments(model), _segments(want)
    changed = [gs._reg(x) for x, y in zip(a[2], b[2]) if x != y]
    assert changed == list(gs.LANES_LO) and a[1] == b[1] and a[3:] == b[3:]
    pa, pb = (np.frombuffer(_params(m), np.uint8) for m in (model, want))
    assert np.array_equal(pa[:4608], pb[:4608])  # the weight codes
    fa, fb = (p[4608 : 4608 + le.BLOCK].view("<f4") for p in (pa, pb))
    # multipliers s_x s_w / s_y: an ulp or two; biases zp_y - zp_x sum(q) M
    # (magnitudes up to a few hundred, with cancellation): below 1e-4
    ulps = np.abs(fa[64:].view("<i4").astype(np.int64) - fb[64:].view("<i4"))
    assert 0 < int(ulps.max()) <= 2
    assert 0 < float(np.abs(fa[:64] - fb[:64]).max()) < 1e-4


# ---- refused -------------------------------------------------------------------
def _args(name="linear64_w1"):
    b = BUILDS[name]
    return _model(name), _weights(name), dict(b["scales"]), dict(b["zero_points"])


def test_other_shapes_are_refused():
    model, w, scales, zps = _args()
    with pytest.raises(ValueError, match="the weights are"):
        le.emit_linear(model, w[:32, :32], scales, zps)
    with pytest.raises(ValueError, match="the weights are"):
        le.emit_linear(model, _weights("linear576_w1_pm1"), scales, zps)
    for n in (128, 512, 1024):
        with pytest.raises(ValueError, match="only 64x64 and 576x576 were built"):
            le.weight_addresses(n)
    wide = onnx.ModelProto()
    wide.CopyFrom(model)
    for v in [*wide.graph.input, *wide.graph.output]:
        v.type.tensor_type.shape.dim[1].dim_value = 128
    with pytest.raises(ValueError, match="only 64x64 and 576x576 were built"):
        le.emit_linear(wide, np.ones((128, 128), np.float32), scales, zps)


def test_a_template_that_is_not_a_linear_layer_is_refused():
    width = os.path.join(AXERA, "fixtures", "graph_stitch", "op_sigmoid_pm1.axmodel.gz")
    _, w, scales, zps = _args()
    with pytest.raises(ValueError, match="npu_params hold 40 bytes"):
        le.emit_linear(width, w, scales, zps)


def test_unbuilt_calibrations_and_weights_are_refused():
    model, w, scales, zps = _args()
    for name in ("x", "y"):
        with pytest.raises(ValueError, match="both must be in 1..255"):
            le.emit_linear(model, w, scales, dict(zps, **{name: 0}))
    with pytest.raises(ValueError, match="scales must be positive"):
        le.emit_linear(model, w, dict(scales, y=0.0), zps)
    with pytest.raises(ValueError, match="no scale or zero point"):
        le.emit_linear(model, w, {"x": scales["x"]}, zps)
    dead = w.copy()
    dead[:, 5] = 0
    with pytest.raises(ValueError, match="all zero"):
        le.emit_linear(model, dead, scales, zps)
    bad = w.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        le.emit_linear(model, bad, scales, zps)
