"""Quark's early return keeps the unoptimized float graph."""

import contextlib
import copy
import io

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim import quark_compat as qc
from onnxsim.quark_marking import quark_sorted

PRESETS = [
    "A8W8",
    "U8S8_AAWS",
    "A16W8",
    "XINT8",
    "VINT8",
    "FP16",
    "BF16",
    "BFP16",
    "MX9_INT8",
    "MATMUL_NBITS",
    "INT8_TRANSFORMER_DEFAULT",
]


def _model():
    model = parser.parse_model(
        """<ir_version: 9, opset_import: ["": 17]>
        g (float[2,4] x, float[2,4] a) => (float[2,4] y)
        <float[4,4] w = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},
         float unused = {3.0}> {
            c = Constant<value = float[4] {0.1,0.2,-0.3,0.4}>()
            i = Identity(x)
            m = MatMul(a, w)
            b = Add(m, c)
            y = Add(b, i)
        }"""
    )
    for i, node in enumerate(model.graph.node):
        node.name = f"n{i}_{node.op_type}"
    return model


def _cfg(preset, exclude):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.exclude = list(exclude)
    cfg.extra_options["ConvertOpsetVersion"] = 20
    return cfg


class _UnusedReader:
    def get_next(self):
        raise AssertionError("Nothing is quantizable: calibration must not run")


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("reader", [None, _UnusedReader()])
def test_all_excluded_nodes_return_before_optimization_and_calibration(preset, reader):
    model = _model()
    expected = quark_sorted(model) if preset == "INT8_TRANSFORMER_DEFAULT" else model
    before = model.SerializeToString()
    out = qc.ModelQuantizer(_cfg(preset, ["^n.*"])).quantize_model(
        model, calibration_data_reader=reader
    )
    assert out.SerializeToString() == expected.SerializeToString()
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    "preset", ["A8W8", "U8S8_AAWS", "XINT8", "FP16", "BFP16", "MATMUL_NBITS"]
)
def test_unsupported_ops_return_without_calibration_or_new_domains(preset):
    model = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> '
        "g (float[2,4] x) => (float[2,4] y) { y = Abs(x) }"
    )
    out = qc.ModelQuantizer(_cfg(preset, [])).quantize_model(model)
    assert out.SerializeToString() == quark_sorted(model).SerializeToString()


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("convert", [False, True])
def test_all_excluded_nodes_match_quark(preset, convert, tmp_path, monkeypatch):
    quark = pytest.importorskip("quark.onnx")
    monkeypatch.chdir(tmp_path)
    model = _model()
    onnx.save(model, "src.onnx")
    cfg = copy.deepcopy(quark.QConfig.get_default_config(preset))
    cfg.global_quant_config.nodes_to_exclude = ["^n.*"]
    if convert:
        cfg.global_quant_config.extra_options["ConvertOpsetVersion"] = 20
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        quark.ModelQuantizer(cfg).quantize_model("src.onnx", "dst.onnx", None)
    ours_cfg = _cfg(preset, ["^n.*"])
    if not convert:
        ours_cfg.extra_options.pop("ConvertOpsetVersion")
    ours = qc.ModelQuantizer(ours_cfg).quantize_model(model)
    theirs = onnx.load("dst.onnx")
    assert list(ours.graph.node) == list(theirs.graph.node)
    assert list(ours.graph.initializer) == list(theirs.graph.initializer)
    assert list(ours.opset_import) == list(theirs.opset_import)
    assert [n.op_type for n in theirs.graph.node].count("Identity") == 1
    assert [n.op_type for n in theirs.graph.node].count("Constant") == 1
    # Execute both models with optimizations disabled, including the two inputs.
    import onnxruntime as ort

    data = {
        name: np.random.default_rng(i).standard_normal((2, 4)).astype(np.float32)
        for i, name in enumerate(("x", "a"))
    }
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    outputs = [
        ort.InferenceSession(
            m.SerializeToString(), so, providers=["CPUExecutionProvider"]
        ).run(None, data)[0]
        for m in (theirs, ours)
    ]
    np.testing.assert_array_equal(outputs[0], outputs[1])
