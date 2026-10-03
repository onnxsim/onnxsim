"""Generic QConfig input marking versus the named Quark preset options."""

import contextlib
import io
import warnings

import numpy as np
import onnx
import pytest
import test_quark_low_opset_parity as parity
from onnx import parser

from onnxsim import quark_compat as qc


def _model(kind):
    body = {
        "transpose": "y = Transpose<perm=[1,0]>(x)",
        "marked": "a = Add(x, bias) y = Transpose<perm=[1,0]>(a)",
        "layernorm": "y = LayerNormalization<axis=-1>(x, scale, bias)",
        "abs": "y = Abs(x)",
    }[kind]
    out_shape = "[4,2]" if kind in ("transpose", "marked") else "[2,4]"
    model = parser.parse_model(
        f"""<ir_version: 9, opset_import: ["": 17]>
        g (float[2,4] x) => (float{out_shape} y)
        <float[4] scale = {{1,1,1,1}}, float[4] bias = {{0.1,0.2,-0.3,0.4}}>
        {{ {body} }}"""
    )
    for i, node in enumerate(model.graph.node):
        node.name = f"n{i}_{node.op_type}"
    return model


def _data():
    rng = np.random.default_rng(11)
    return [{"x": rng.standard_normal((2, 4)).astype(np.float32)} for _ in range(3)]


def _config(lib, spec, force=None):
    opts = {"SkipPreprocess": True}
    if force is not None:
        opts["ForceQuantizeNoInputCheck"] = force
    return lib.QConfig(
        global_config=lib.QLayerConfig(
            activation=getattr(lib, spec)(),
            weight=lib.Int8Spec() if spec == "Int16Spec" else getattr(lib, spec)(),
        ),
        **opts,
    )


def _mine(model, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_data()
        )


def _has_quantizers(model):
    return any("Quantize" in n.op_type for n in model.graph.node)


@pytest.mark.parametrize(
    "spec", ["Int8Spec", "Int16Spec", "Float16Spec", "BFloat16Spec", "BFP16Spec"]
)
@pytest.mark.parametrize("kind", ["transpose", "layernorm"])
def test_generic_config_leaves_unmarked_inputs_float_unless_forced(spec, kind):
    model = _model(kind)
    assert not _has_quantizers(_mine(model, _config(qc, spec)))
    assert not _has_quantizers(_mine(model, _config(qc, spec, False)))
    assert _has_quantizers(_mine(model, _config(qc, spec, True)))


@pytest.mark.parametrize(
    "preset", ["A8W8", "A16W8", "XINT8", "U8S8_AAWS", "FP16", "BF16"]
)
def test_named_presets_still_quantize_unmarked_inputs(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options["SkipPreprocess"] = True
    assert _has_quantizers(_mine(_model("transpose"), cfg))


def test_generic_bf16_does_not_inherit_the_presets_all_op_types_option():
    assert not _has_quantizers(_mine(_model("abs"), _config(qc, "BFloat16Spec")))
    cfg = qc.QConfig.get_default_config("BF16")
    cfg.extra_options["SkipPreprocess"] = True
    assert _has_quantizers(_mine(_model("abs"), cfg))


@pytest.mark.parametrize(
    "spec, kind",
    [
        (s, k)
        for s in ("Int8Spec", "Int16Spec", "Float16Spec", "BFloat16Spec", "BFP16Spec")
        for k in ("transpose", "marked", "layernorm")
    ]
    + [("BFloat16Spec", "abs")],
)
@pytest.mark.parametrize("force", [None, False, True])
def test_generic_input_marking_matches_quark(spec, kind, force, tmp_path, monkeypatch):
    quark = pytest.importorskip("quark.onnx")
    monkeypatch.chdir(tmp_path)
    model = _model(kind)
    onnx.save(model, "src.onnx")

    # Use the same calibration samples on both sides.
    class Reader:
        def __init__(self):
            self.it = iter(_data())

        def get_next(self):
            return next(self.it, None)

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        quark.ModelQuantizer(_config(quark, spec, force)).quantize_model(
            "src.onnx", "dst.onnx", Reader()
        )
    theirs = onnx.load("dst.onnx")
    ours = _mine(model, _config(qc, spec, force))
    msg = f"{spec} {kind} ForceQuantizeNoInputCheck={force}"
    parity.assert_same_graph(theirs, ours, msg=msg)
    parity.assert_same_outputs(theirs, ours, (2, 4), msg=msg)
