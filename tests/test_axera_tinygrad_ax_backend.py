"""No-device tests for the template+patch tinygrad backend skeleton.

Every comparison is against committed Pulsar2-built fixtures: the backend's
output must equal what the underlying emitter already reproduces byte for byte
(or within the tolerance that emitter's own tests document).
"""

import gzip
import json
import os
import struct
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import binary_op_scale_emit as bse  # noqa: E402
import binary_op_scale_validate as bsv  # noqa: E402
import elementwise_scale_emit as ew  # noqa: E402
import tinygrad_ax_backend as axb  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures")
_STEP_OPS = os.path.join(_FIX, "tinygrad_ax_backend", "resnet18_step_ops.json.gz")


def _load_gz(path):
    with gzip.open(path, "rb") as f:
        return onnx.load_model_from_string(f.read())


def _init(model, name):
    return bytes(next(i for i in model.graph.initializer if i.name == name).raw_data)


def _mcode(model):
    return bytes(
        next(i for i in model.graph.initializer if i.name.endswith("_neu")).raw_data
    )


def _outside_noise(mc):
    data = bytearray(mc)
    data[301:326] = bytes(25)
    return bytes(data)


def _gather_key():
    return axb.TemplateKey("Gather", ((1, 1, 4, 16),), (("axis", 3), ("indices", 8)))


def test_template_key_json_round_trip():
    key = axb.TemplateKey(
        "Transpose", ((16, 512),), (("perm", (1, 0)),), calibration_class=""
    )
    assert axb.TemplateKey.from_json(json.loads(json.dumps(key.to_json()))) == key


@pytest.mark.parametrize(
    "key",
    [
        axb.TemplateKey("Mul", ((1, 64, 56, 56),)),
        axb.TemplateKey("Gather", ((1, 1, 4, 16),), (("axis", 3), ("indices", 7))),
        axb.TemplateKey("Transpose", ((16, 513),), (("perm", (1, 0)),)),
        axb.TemplateKey("Relu", ((16, 64, 56, 56),), calibration_class="x1,y1"),
        axb.TemplateKey("Relu", ((16, 64, 56, 56),), dtypes=("int8",)),
    ],
)
def test_cache_refuses_unmeasured_keys(key):
    with pytest.raises(ValueError):
        axb.TemplateCache().lookup(key)


def test_cache_miss_is_not_built():
    with pytest.raises(NotImplementedError, match="Pulsar2 build"):
        axb.TemplateCache().get_or_build(axb.TemplateKey("Mul", ((1, 64, 56, 56),)))


def test_gather_edit_writes_indices_and_keeps_mcode():
    key = _gather_key()
    template = axb.TemplateCache().load(key)
    indices = [15, 0, 7, 7, 3, 12, 1, 14]
    model = axb.EditSet([axb.GatherIndexEdit(indices)]).build(key)
    table = _init(model, "npu_params")
    assert list(struct.unpack("<8I", table[:32])) == indices
    assert table[32:] == _init(template, "npu_params")[32:]
    assert _mcode(model) == _mcode(template)


@pytest.mark.parametrize("indices", [[0] * 7, [16] + [0] * 7, [-1] + [0] * 7])
def test_gather_edit_refuses_bad_indices(indices):
    with pytest.raises(ValueError):
        axb.EditSet([axb.GatherIndexEdit(indices)]).build(_gather_key())


def test_transpose_template_only_is_the_committed_template():
    key = axb.TemplateKey("Transpose", ((16, 512),), (("perm", (1, 0)),))
    model = axb.EditSet([axb.TemplateOnly()]).build(key)
    template = _load_gz(
        os.path.join(_FIX, "transpose_real", "transpose_real_16x512_perm10.axmodel.gz")
    )
    assert model.SerializeToString() == template.SerializeToString()


def test_template_only_refused_where_an_edit_is_required():
    with pytest.raises(ValueError):
        axb.EditSet([axb.TemplateOnly()]).build(_gather_key())


_ORACLES = os.path.join(ew.TEMPLATE_DIR, "oracles")
with open(os.path.join(_ORACLES, "index.json")) as _f:
    _ORACLE_INDEX = json.load(_f)


@pytest.mark.parametrize(
    "oracle",
    [n for n in sorted(_ORACLE_INDEX) if n.startswith("relu_16x512")]
    + ["sqrt_64x64x3x3_zp0_h2.axmodel.gz"],
)
def test_elementwise_scale_edit_reproduces_held_out_native_build(oracle):
    meta = _ORACLE_INDEX[oracle]
    zp = meta["zero_points"]
    key = axb.TemplateKey(
        meta["op"],
        (tuple(meta["shape"]),),
        calibration_class=f"x{zp['x']},y{zp['y']}",
    )
    model = axb.EditSet([axb.ElementwiseScaleEdit(meta["scales"])]).build(key)
    native = _load_gz(os.path.join(_ORACLES, oracle))
    assert _outside_noise(_mcode(model)) == _outside_noise(_mcode(native))
    assert _init(model, "npu_params") == _init(native, "npu_params")


_BINARY_ORACLES = os.path.join(bse.TEMPLATE_DIR, "oracles")
with open(os.path.join(_BINARY_ORACLES, "index.json")) as _f:
    _BINARY_ORACLE_INDEX = json.load(_f)


@pytest.mark.parametrize(
    "oracle",
    [
        n
        for n in sorted(_BINARY_ORACLE_INDEX)
        if _BINARY_ORACLE_INDEX[n]["shape"] in ([16, 64, 56, 56], [16, 128, 28, 28])
    ],
)
def test_elementwise_scale_edit_reproduces_held_out_binary_build(oracle):
    meta = _BINARY_ORACLE_INDEX[oracle]
    zp = meta["zero_points"]
    key = axb.TemplateKey(
        meta["op"],
        (tuple(meta["shape"]),),
        calibration_class=",".join(f"{k}{v}" for k, v in sorted(zp.items())),
    )
    model = axb.EditSet([axb.ElementwiseScaleEdit(meta["scales"])]).build(key)
    native = _load_gz(os.path.join(_BINARY_ORACLES, oracle))
    r = bsv.compare(model, native)
    assert r["params"] and r["segments"], r


def test_relu_tile_prediction_matches_template():
    key = axb.TemplateKey("Relu", ((16, 128, 28, 28),), calibration_class="x0,y0")
    template = axb.TemplateCache().load(key)
    assert axb.predicted_npu_params("Relu", (16, 128, 28, 28)) == _init(
        template, "npu_params"
    )


def _npy_gz(name):
    with gzip.open(os.path.join(_FIX, "conv_weight_learn", name), "rb") as f:
        return np.load(f)


def test_conv_weight_edit_stage1_matches_native_code_region():
    key = axb.TemplateKey(
        "Conv",
        ((16, 64, 56, 56),),
        (("pads", (1, 1, 1, 1)), ("strides", (1, 1)), ("w", (64, 64, 3, 3))),
    )
    w, b = _npy_gz("holdout_w.npy.gz"), _npy_gz("holdout_b.npy.gz")
    # holdout calibration recorded in docs/axera-conv-weight-learn-stem.md
    edit = axb.ConvWeightEdit(
        w, b, 0.007058821618556976, 127.0, 0.03239550068974495, 125.0
    )
    model = axb.EditSet([edit]).build(key)
    native = _load_gz(
        os.path.join(_FIX, "conv_weight_learn", "holdout_native.axmodel.gz")
    )
    got = np.frombuffer(_init(model, "npu_params"), np.uint8)
    want = np.frombuffer(_init(native, "npu_params"), np.uint8)
    assert np.array_equal(got[: w.size], want[: w.size])
    block = slice(36864, 36864 + 512)
    assert (
        np.abs(got[block].view(np.float32) - want[block].view(np.float32)).max() < 2e-4
    )


def _quant(name, tensor):
    with gzip.open(os.path.join(_FIX, "conv_learn_downsample", name), "rt") as f:
        doc = json.load(f)
    tc = list(doc["tensor_configs"].values())[0][tensor]
    v = doc["values"][str(tc["hash"])]
    return float(v["scale"][0]), float(v["zero_point"][0])


def test_conv_weight_edit_1x1_downsample_matches_native_outside_block():
    key = axb.TemplateKey(
        "Conv",
        ((16, 64, 56, 56),),
        (("pads", (0, 0, 0, 0)), ("strides", (2, 2)), ("w", (128, 64, 1, 1))),
    )
    wts = np.load(os.path.join(_FIX, "conv_learn_downsample", "s1_holdout_weights.npz"))
    xs, xz = _quant("s1_reference_quant.json.gz", "x")
    ys, yz = _quant("s1_holdout_quant.json.gz", "y")
    model = axb.EditSet([axb.ConvWeightEdit(wts["w"], wts["b"], xs, xz, ys, yz)]).build(
        key
    )
    native = _load_gz(
        os.path.join(_FIX, "conv_learn_downsample", "s1_holdout_native.axmodel.gz")
    )
    got = np.frombuffer(_init(model, "npu_params"), np.uint8)
    want = np.frombuffer(_init(native, "npu_params"), np.uint8)
    outside = np.ones(len(got), bool)
    outside[9216 : 9216 + 1024] = False
    assert np.array_equal(got[outside], want[outside])


def test_conv_weight_edit_refuses_wrong_weight_shape():
    key = axb.TemplateKey(
        "Conv",
        ((16, 64, 56, 56),),
        (("pads", (1, 1, 1, 1)), ("strides", (1, 1)), ("w", (64, 64, 3, 3))),
    )
    w = np.zeros((64, 64, 1, 1), np.float32)
    with pytest.raises(ValueError):
        axb.EditSet(
            [axb.ConvWeightEdit(w, np.zeros(64, np.float32), 1, 0, 1, 0)]
        ).build(key)


def test_compile_request_round_trip():
    key = _gather_key()
    src = axb.build_request(key, [axb.GatherIndexEdit(list(range(8)))])
    out = axb.compile_request(src)
    assert axb.gather_indices_of(out, 8) == list(range(8))


def _step_records():
    with gzip.open(_STEP_OPS, "rt") as f:
        return json.load(f)


def test_coverage_report_on_the_resnet18_step():
    report = axb.coverage_report(_step_records())
    assert report["nodes"] == 1104
    assert report["per_op"]["Gather"] == {"covered": 41}
    assert report["per_op"]["Transpose"] == {"covered": 41}
    assert report["per_op"]["Relu"] == {"conditional": 17}
    assert report["per_op"]["Sqrt"] == {"conditional": 39, "refused": 3}
    assert report["per_op"]["Conv"] == {"refused": 20}
    # same-shape binary ops: ElementwiseScaleEdit (binary_op_scale_emit.py);
    # constant and broadcast operands stay refused
    assert report["per_op"]["Add"] == {"conditional": 101, "refused": 43}
    assert report["per_op"]["Sub"] == {"conditional": 42, "refused": 4}
    assert report["per_op"]["Mul"] == {"conditional": 63, "refused": 334}
    assert report["per_op"]["Div"] == {"conditional": 44, "refused": 8}
    assert report["totals"] == {"covered": 82, "conditional": 324, "refused": 698}


def test_trainable_conv_is_refused_even_with_a_template():
    rec = next(
        r
        for r in _step_records()
        if r["op"] == "Conv" and r["attrs"]["w"] == [64, 64, 3, 3]
    )
    status, detail = axb.plan_node(rec)
    assert status == "refused" and "graph input" in detail


def test_tinygrad_compiler_seam():
    pytest.importorskip("tinygrad")
    classes = axb.tinygrad_classes()
    src = axb.build_request(_gather_key(), [axb.GatherIndexEdit(list(range(8)))])
    out = classes["AXCompiler"]().compile_cached(src)
    assert out == axb.compile_request(src)
    with pytest.raises(NotImplementedError):
        classes["AXProgram"](None, None)
