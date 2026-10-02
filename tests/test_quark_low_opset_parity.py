"""Parity of onnxsim's Quark-compat flow on models with a default-domain opset
below 13 against the real AMD Quark ONNX package (skipped unless ``quark.onnx`` is
importable).

What Quark does with an opset 9..12 model (probed with Quark 0.13):

- it quantizes in place -- the default-domain opset is never converted (only an
  FP8 request bumps it) and Quark's own ``com.microsoft`` / ``com.amd.quark``
  operator sets are registered on the model;
- per-tensor ``QuantizeLinear`` / ``DequantizeLinear`` carry no ``axis`` (a
  per-channel configuration raises below opset 13);
- ONNX Runtime's QDQ operator quantizers, which its registries reuse, leave
  ``MaxPool`` alone below opset 12 and ``Resize`` below opset 11;
- the attribute-vs-input forms of Clip / Pad / Split / Squeeze / Unsqueeze /
  Slice / ReduceMean follow the opset of the model and are handled like the
  input forms.

Each test runs Quark and onnxsim on the same parser-built graph and compares the
whole emitted graph (``opset_import``, node types, domains, attributes, wiring,
scales, zero points and constants, every float bit for bit) and, with ONNX Runtime's
graph optimizations off, what the two graphs compute. A default-domain
``QuantizeLinear`` is an opset 10 operator: Quark's plain-QDQ presets emit models
ONNX Runtime refuses to load at opset 9, and the tests then require onnxsim's to
be refused as well. Known, deliberate differences are the ``KNOWN_*`` constants.
"""

import contextlib
import copy
import io
import os
import warnings
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from test_quark_low_opset import (  # noqa: E402
    MODELS,
    OPSETS,
    QUARK_ERROR,
    _data,
    _quantize,
    _Reader,
)

from onnxsim import quark_compat as qc  # noqa: E402

# Graphs that INT8_TRANSFORMER_DEFAULT quantizes as Quark does only with a
# tolerance: the INT8_TRANSFORMER_DEFAULT preset's moving-average calibration
# differs from Quark's by one float32 ulp in some activation scales (at every
# opset), and onnxsim keeps the 1/255 range of a Softmax output where Quark's
# transformer quantizer calibrates it; Quark's ReduceMean -> GlobalAveragePool
# rewrite is not applied by that preset.
KNOWN_TRANSFORMER_DIFF = {"softmax", "reducemean_squeeze"}
#: the Split -> Slice conversion of these presets writes the opset 10 form of
#: Slice (five inputs), which ONNX Runtime cannot load below opset 10: both fail
#: (see ``test_split_to_slice_at_opset_9_fails_in_quark_and_here``)
KNOWN_UNLOADABLE = {
    ("concat_split", 9, p): "Split -> Slice conversion"
    for p in ("A8W8", "A16W8", "XINT8")
}

_PRESETS = ["A8W8", "U8S8_AAWS", "S8S8_AAWS", "XINT8", "INT8_CNN_DEFAULT", "A16W8"]


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files (``quantized_info.csv`` ...) into the current
    directory; keep them out of the checkout."""
    monkeypatch.chdir(tmp_path)


# -- helpers --------------------------------------------------------------------


def _reader(shape):
    from onnxruntime.quantization import CalibrationDataReader

    data = _data(shape)

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

    return R


def quark_quantize(model, preset, shape, tmp_path, extra=None, per_channel=False):
    from quark.onnx import ModelQuantizer, QConfig

    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    if os.path.exists(dst):
        os.remove(dst)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        # (``get_default_config`` hands out a shared object: work on a copy)
        cfg = copy.deepcopy(QConfig.get_default_config(preset))
        g = cfg.global_quant_config
        g.include_cle = False  # (every Quark preset runs CLE; onnxsim's does not)
        g.per_channel = per_channel
        g.extra_options.update(extra or {})
        ModelQuantizer(cfg).quantize_model(src, dst, _reader(shape)())
    # (Quark logs an error and writes nothing when ONNX Runtime refuses its graph)
    return onnx.load(dst) if os.path.exists(dst) else None


def _outputs(model, x):
    """The model's output, or None when ONNX Runtime refuses to load it."""
    so = ort.SessionOptions()
    # No graph optimizations: ORT would otherwise fuse DQ -> MatMul/Conv -> Q into
    # integer kernels that saturate on x86 CPUs without VNNI (CI runners)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    lib = os.environ.get("QUARK_ONNX_OPS_LIB")
    if lib:
        so.register_custom_ops_library(lib)
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )
    except Exception:
        return None
    return sess.run(None, {"x": x})[0]


def _attr(a):
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, onnx.TensorProto):
        return ("tensor", numpy_helper.to_array(v).tobytes())
    return tuple(v) if hasattr(v, "__iter__") else v


def _const(arr, exact):
    arr = np.asarray(arr)
    if arr.dtype.kind == "f" and not exact:
        arr = np.array([float(f"{x:.5g}") for x in arr.ravel()]).reshape(arr.shape)
    return (str(arr.dtype), arr.shape, arr.tobytes())


def _signatures(model, exact=True):
    """``(opset_import, [Merkle hash of every node])``: a node's hash covers its
    op type, domain, attributes and, recursively, what feeds it (a constant by
    value: a ``Constant`` node counts as an initializer, which is where
    onnxsim's rewrites put what Quark leaves in ``Constant`` nodes)."""
    consts = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = []
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute[0].name == "value":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
        else:
            nodes.append(n)
    prod = {o: n for n in nodes for o in n.output}
    memo = {}

    def h(tensor):
        if not tensor:
            return "-"
        if tensor not in memo:
            if tensor in prod:
                n = prod[tensor]
                memo[tensor] = (
                    n.op_type,
                    n.domain,
                    tuple(sorted((a.name, _attr(a)) for a in n.attribute)),
                    tuple(h(i) for i in n.input),
                    list(n.output).index(tensor),
                )
            elif tensor in consts:
                memo[tensor] = ("const", _const(consts[tensor], exact))
            else:
                memo[tensor] = ("input", tensor)
        return memo[tensor]

    opsets = {o.domain: o.version for o in model.opset_import}
    return opsets, sorted(
        (repr(h(o)) for n in nodes for o in n.output[:1]),
    )


def _brief(model):
    return Counter(
        f"{n.op_type}:{n.domain}" for n in model.graph.node if n.op_type != "Constant"
    )


def assert_same_graph(q, m, exact=True, msg=""):
    qo, qs = _signatures(q, exact)
    mo, ms = _signatures(m, exact)
    assert mo == qo, f"{msg}: opset_import"
    assert _brief(m) == _brief(q), f"{msg}: node types"
    assert ms == qs, f"{msg}: wiring / attributes / constants"


def assert_same_outputs(q, m, shape, exact=True, msg=""):
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    want, got = _outputs(q, x), _outputs(m, x)
    # (a model ONNX Runtime refuses to load must be refused in both cases)
    assert (want is None) == (got is None), f"{msg}: loadable by ONNX Runtime"
    if want is not None:
        if exact:
            np.testing.assert_array_equal(got, want, err_msg=msg)
        else:
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=msg)


# -- the main presets ---------------------------------------------------------------


@pytest.mark.parametrize("opset", OPSETS)
@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("preset", _PRESETS)
def test_main_presets_match_quark_below_opset_13(preset, name, opset, tmp_path):
    if (name, opset, preset) in KNOWN_UNLOADABLE:
        pytest.skip(KNOWN_UNLOADABLE[(name, opset, preset)])
    model, shape = MODELS[name](opset)
    q = quark_quantize(model, preset, shape, tmp_path)
    m = _quantize(model, preset, shape)
    msg = f"{name}@{opset}/{preset}"
    assert_same_graph(q, m, msg=msg)
    assert_same_outputs(q, m, shape, msg=msg)


@pytest.mark.parametrize("opset", OPSETS)
@pytest.mark.parametrize(
    "name", sorted(set(MODELS) - KNOWN_TRANSFORMER_DIFF - {"softmax"})
)
def test_transformer_default_matches_quark_below_opset_13(name, opset, tmp_path):
    # (floats to 5 digits, outputs to 1e-5: see KNOWN_TRANSFORMER_DIFF)
    model, shape = MODELS[name](opset)
    q = quark_quantize(model, "INT8_TRANSFORMER_DEFAULT", shape, tmp_path)
    m = _quantize(model, "INT8_TRANSFORMER_DEFAULT", shape)
    msg = f"{name}@{opset}/INT8_TRANSFORMER_DEFAULT"
    assert_same_graph(q, m, exact=False, msg=msg)
    assert_same_outputs(q, m, shape, exact=False, msg=msg)


@pytest.mark.parametrize("opset", OPSETS)
def test_quark_never_converts_the_opset(opset, tmp_path):
    model, shape = MODELS["mlp"](opset)
    for preset in ("A8W8", "U8S8_AAWS", "INT8_TRANSFORMER_DEFAULT"):
        q = quark_quantize(model, preset, shape, tmp_path)
        assert {o.version for o in q.opset_import if o.domain == ""} == {opset}


# -- the errors -----------------------------------------------------------------------


@pytest.mark.parametrize("opset", OPSETS + (13,))
@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "INT8_CNN_DEFAULT", "A16W8"])
def test_per_channel_raises_below_opset_13_like_quark(preset, opset, tmp_path):
    model, shape = MODELS["conv_relu_pool"](opset)
    expected = None if opset >= 13 else QUARK_ERROR
    with pytest.raises(ValueError) if expected else contextlib.nullcontext() as quark:
        quark_quantize(model, preset, shape, tmp_path, per_channel=True)
    with pytest.raises(ValueError) if expected else contextlib.nullcontext() as mine:
        _quantize(model, preset, shape, {"PerChannel": True})
    if expected:
        assert str(quark.value) == expected
        assert str(mine.value) == expected


@pytest.mark.parametrize("opset", [11, 13])
def test_xint8_refuses_per_channel_like_quark(opset, tmp_path):
    model, shape = MODELS["conv_relu_pool"](opset)
    with pytest.raises(ValueError) as quark:
        quark_quantize(model, "XINT8", shape, tmp_path, per_channel=True)
    with pytest.raises(ValueError) as mine:
        _quantize(model, "XINT8", shape, {"PerChannel": True})
    assert str(mine.value) == str(quark.value)


@pytest.mark.parametrize("opset", OPSETS)
@pytest.mark.parametrize(
    "name",
    ["mlp", "matmul_add", "conv_relu_pool", "softmax", "pool", "eltwise", "bn_conv"],
)
def test_dynamic_quantization_matches_quark_below_opset_13(name, opset, tmp_path):
    # below opset 11 (no DynamicQuantizeLinear) both compute the scale and zero
    # point with ReduceMin / ReduceMax / Sub / Div / Floor / Cast nodes and
    # quantize with QuantizeLinear
    model, shape = MODELS[name](opset)
    q = quark_quantize(model, "UINT8_DYNAMIC_QUANT", shape, tmp_path)
    m = _quantize(model, "UINT8_DYNAMIC_QUANT", shape)
    msg = f"{name}@{opset}/UINT8_DYNAMIC_QUANT"
    ops = {n.op_type for n in q.graph.node}
    assert ("DynamicQuantizeLinear" in ops) == (opset >= 11), msg
    assert_same_graph(q, m, msg=msg)
    assert_same_outputs(q, m, shape, msg=msg)


# -- ConvertOpsetVersion: Quark's opt-in conversion --------------------------------

_CONVERTED = [
    "mlp",
    "softmax",
    "clip6",
    "concat_split",
    "pad_conv",
    "slice_conv",
    "reducemean_squeeze",
    "resize_conv",
    "pool",
    "residual",
]


@pytest.mark.parametrize("target", [13, 17])
@pytest.mark.parametrize("opset", [9, 12])
@pytest.mark.parametrize("name", _CONVERTED)
@pytest.mark.parametrize("preset", ["U8S8_AAWS", "A8W8", "XINT8"])
def test_convert_opset_version_matches_quark(preset, name, opset, target, tmp_path):
    model, shape = MODELS[name](opset)
    extra = {"ConvertOpsetVersion": target}
    q = quark_quantize(model, preset, shape, tmp_path, extra)
    m = _quantize(model, preset, shape, extra)
    msg = f"{name}@{opset}->{target}/{preset}"
    assert {o.version for o in m.opset_import if o.domain == ""} == {target}
    assert_same_graph(q, m, msg=msg)
    assert_same_outputs(q, m, shape, msg=msg)


@pytest.mark.parametrize("target", [13, 17])
@pytest.mark.parametrize("name", ["conv_relu_pool", "mlp", "concat_split"])
@pytest.mark.parametrize("preset", ["U8S8_AAWS", "A8W8", "INT8_CNN_DEFAULT"])
def test_per_channel_works_after_converting_the_opset(preset, name, target, tmp_path):
    model, shape = MODELS[name](11)
    extra = {"ConvertOpsetVersion": target}
    q = quark_quantize(model, preset, shape, tmp_path, extra, per_channel=True)
    m = _quantize(model, preset, shape, {**extra, "PerChannel": True})
    msg = f"{name}@11->{target}/{preset} per channel"
    assert_same_graph(q, m, msg=msg)
    assert_same_outputs(q, m, shape, msg=msg)


@pytest.mark.parametrize("target", [99, 8])
def test_failed_opset_conversion_is_skipped_like_quark(target, tmp_path):
    # Quark warns and quantizes the model as it is
    model, shape = MODELS["softmax"](11)
    extra = {"ConvertOpsetVersion": target}
    q = quark_quantize(model, "U8S8_AAWS", shape, tmp_path, extra)
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.extra_options.update(extra)
    with pytest.warns(UserWarning, match="skipping the conversion"):
        m = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_Reader(_data(shape))
        )
    assert {o.version for o in m.opset_import if o.domain == ""} == {11}
    assert_same_graph(q, m)
    assert_same_outputs(q, m, shape)


@pytest.mark.parametrize("preset", ["A8W8", "XINT8"])
def test_split_to_slice_at_opset_9_fails_in_quark_and_here(preset, tmp_path):
    # Both convert Split to the opset 10 form of Slice (five inputs), which ONNX
    # Runtime cannot load in an opset 9 model: Quark logs the error and writes no
    # file, onnxsim raises it
    model, shape = MODELS["concat_split"](9)
    assert quark_quantize(model, preset, shape, tmp_path) is None
    with pytest.raises(Exception, match="Slice"):
        _quantize(model, preset, shape)


# -- half precision and block formats: the MaxPool / Resize gates -------------------


def _ext_ops(model):
    return Counter(
        f"{n.op_type}:{n.domain}"
        for n in model.graph.node
        if n.domain == "com.amd.quark"
        or n.op_type in ("QuantizeLinear", "DequantizeLinear")
    )


@pytest.mark.parametrize("opset", [10, 11, 12])
@pytest.mark.parametrize("name", ["conv_relu_pool", "resize_conv"])
@pytest.mark.parametrize("preset", ["FP16", "BF16", "BFP16", "MX9", "BF16_BFP16"])
def test_half_and_block_formats_skip_maxpool_and_resize_like_quark(
    preset, name, opset, tmp_path
):
    model, shape = MODELS[name](opset)
    q = quark_quantize(model, preset, shape, tmp_path)
    m = _quantize(model, preset, shape)
    assert _ext_ops(m) == _ext_ops(q), f"{name}@{opset}/{preset}"
    assert _brief(m) == _brief(q)
