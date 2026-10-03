"""Tests for onnxsim.quark_compat -- the Quark-ONNX-API-shaped shim backed by
onnxsim's own quantizers (see that module's docstring for scope)."""

import os
import warnings

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim import quark_compat as qc


def _model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,4] x) => (float[N,4] y)
        <float[4,4] w = {1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0}>
        {
            y = MatMul(x, w)
        }
        """
    )


class _Reader:
    """onnxruntime-style CalibrationDataReader."""

    def __init__(self, n=4):
        rng = np.random.default_rng(0)
        self._it = iter(
            [{"x": rng.standard_normal((2, 4)).astype(np.float32)} for _ in range(n)]
        )

    def get_next(self):
        return next(self._it, None)


def _ops(model):
    return {n.op_type for n in model.graph.node}


def test_default_config_lookup_and_unknown():
    assert (
        qc.QConfig.get_default_config("A8W8").global_config.activation.dtype == "int8"
    )
    assert (
        qc.QConfig.get_default_config("U16S8_AAWS").global_config.activation.dtype
        == "uint16"
    )
    with pytest.raises(ValueError, match="unknown preset"):
        qc.QConfig.get_default_config("NOPE")


def test_adaround_preset_carries_algo():
    cfg = qc.QConfig.get_default_config("A8W8_ADAROUND")
    assert [a.name for a in cfg.algo_config] == ["adaround"]


def test_config_wrapper_and_extra_options():
    cfg = qc.QConfig(
        qc.QLayerConfig(qc.Int8Spec(), qc.Int8Spec()),
        extra_options={"FoldRelu": True},
    )
    assert cfg.extra_options == {"FoldRelu": True}
    q = qc.ModelQuantizer(qc.Config(global_quant_config=cfg))
    assert q.config is cfg


def test_u8_preset_quantizes_to_qdq_without_approximation(tmp_path):
    out = tmp_path / "q.onnx"
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("U8S8_AAWS"))
    model = q.quantize_model(_model(), str(out), _Reader())
    assert {"QuantizeLinear", "DequantizeLinear"} <= _ops(model)
    assert q.last_approximations == []
    onnx.checker.check_model(onnx.load(str(out)))


def test_signed_int8_activations_are_symmetric_with_zero_point_zero():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("A8W8"))
    model = q.quantize_model(_model(), calibration_data_reader=_Reader())
    assert not any("mapped to uint" in m for m in q.last_approximations)
    zps = [
        i
        for i in model.graph.initializer
        if i.data_type == onnx.TensorProto.INT8 and i.name.endswith("/zp")
    ]
    assert zps and all(not onnx.numpy_helper.to_array(z).any() for z in zps)
    assert not any(
        i.data_type == onnx.TensorProto.UINT8 for i in model.graph.initializer
    )


def test_a16w8_uses_int16_activations():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("A16W8"))
    model = q.quantize_model(_model(), calibration_data_reader=_Reader())
    types = {i.data_type for i in model.graph.initializer}
    assert onnx.TensorProto.INT16 in types


def test_xint8_scales_are_powers_of_two_with_centred_uint8_zero_point():
    model = qc.ModelQuantizer(qc.QConfig.get_default_config("XINT8")).quantize_model(
        _model(), calibration_data_reader=_Reader()
    )
    scales = [
        float(onnx.numpy_helper.to_array(i))
        for i in model.graph.initializer
        if i.name.endswith("/scale") and not onnx.numpy_helper.to_array(i).ndim
    ]
    assert scales and all(np.log2(s) == round(np.log2(s)) for s in scales)
    zps = [
        int(onnx.numpy_helper.to_array(i))
        for i in model.graph.initializer
        if i.name.endswith("/zp") and i.data_type == onnx.TensorProto.UINT8
    ]
    assert zps and set(zps) == {128}


@pytest.mark.parametrize("preset, half", [("FP16", "FLOAT16"), ("BF16", "BFLOAT16")])
def test_float_presets_fake_quantize_every_tensor_like_quark(preset, half):
    # Quark's FP16 / BF16 presets are per-tensor rounding (Extended Q/DQ with scale
    # 1.0 and a half-precision zero point), not a whole-graph dtype conversion --
    # which is also why the result still runs on ONNX Runtime CPU.
    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    with pytest.warns(UserWarning, match="custom-op library"):
        out = q.quantize_model(_model())  # no calibration data needed
    ops = [n.op_type for n in out.graph.node]
    assert ops.count("ExtendedQuantizeLinear") == ops.count("ExtendedDequantizeLinear")
    assert {n.domain for n in out.graph.node if n.op_type.startswith("Extended")} == {
        "com.amd.quark"
    }
    zps = [t for t in out.graph.initializer if t.name.endswith("_zero_point")]
    assert zps and {onnx.TensorProto.DataType.Name(t.data_type) for t in zps} == {half}
    scales = [t for t in out.graph.initializer if t.name.endswith("_scale")]
    assert all(float(onnx.numpy_helper.to_array(t)) == 1.0 for t in scales)
    onnx.checker.check_model(out)


@pytest.mark.parametrize("preset", ["FP16", "BF16"])
def test_convert_to_half_option_keeps_the_whole_graph_conversion(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options["ConvertToHalf"] = True
    out = qc.ModelQuantizer(cfg).quantize_model(_model())
    assert "Cast" in _ops(out) and "ExtendedQuantizeLinear" not in _ops(out)


def _wide_matmul_model():
    rng = np.random.default_rng(5)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,32] x) => (float[N,8] y)
        {
            y = MatMul(x, w)
        }
        """
    )
    # Random weights are attached programmatically (too large for text literals).
    w = (rng.standard_normal((32, 8)) * np.exp2(rng.integers(-3, 3, (32, 8)))).astype(
        np.float32
    )
    model.graph.initializer.append(onnx.numpy_helper.from_array(w, "w"))
    return model


_BLOCK_PRESETS = [
    ("BFP16", "bfp16"),
    ("MX4", "mx4"),
    ("MX9", "mx9"),
    ("MXINT8", "mxint8"),
    ("MXFP8E4M3", "mxfp8_e4m3"),
    ("MXFP4E2M1", "mxfp4_e2m1"),
]


def _expected_weights(fmt, w):
    from onnxsim import quark_block_formats as bf

    return {
        "bfp16": lambda a: bf.bfp16(a, axis=0),
        "mx4": lambda a: bf.bfp_prime(a, bit_width=11, axis=0),
        "mx9": lambda a: bf.bfp_prime(a, bit_width=16, axis=0),
        "mxint8": lambda a: bf.mx(a, element_dtype="int8", axis=0),
        "mxfp8_e4m3": lambda a: bf.mx(a, element_dtype="fp8_e4m3", axis=0),
        "mxfp4_e2m1": lambda a: bf.mx(a, element_dtype="fp4_e2m1", axis=0),
    }[fmt](w)


def _weights_only(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options["BlockFormatActivations"] = False
    return qc.ModelQuantizer(cfg)


@pytest.mark.parametrize("preset, fmt", _BLOCK_PRESETS)
def test_block_format_presets_fake_quantize_weights(preset, fmt):
    q = _weights_only(preset)
    model = _wide_matmul_model()
    with pytest.warns(UserWarning, match="BlockFormatActivations=False"):
        out = q.quantize_model(model)
    w_in = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w_out = onnx.numpy_helper.to_array(out.graph.initializer[0])
    assert not np.array_equal(w_in, w_out)
    # blocks run along the reduction axis (K = axis 0 of a MatMul weight)
    np.testing.assert_array_equal(w_out, _expected_weights(fmt, w_in))
    assert [n.op_type for n in out.graph.node] == ["MatMul"]  # graph untouched
    onnx.checker.check_model(out)


def _attrs(node):
    out = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    return {k: v.decode() if isinstance(v, bytes) else v for k, v in out.items()}


def _fn_nodes(model):
    """``{quantized tensor: node}`` -- the tensor a block node quantizes."""
    out = {}
    for n in model.graph.node:
        if n.domain == "com.amd.quark":
            src = n.input[0]
            out[src.removesuffix("_QuantizeLinear_Input")] = n
    return out


@pytest.mark.parametrize("preset, fmt", _BLOCK_PRESETS)
def test_block_format_presets_follow_quarks_graph_layout(preset, fmt):
    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    model = _wide_matmul_model()
    with pytest.warns(UserWarning, match="custom-op library"):
        out = q.quantize_model(model)
    fns = _fn_nodes(out)
    # Quark quantizes the constant, the activation input and the layer output
    assert set(fns) == {"w", "x", "y"}
    ops = {n.op_type for n in fns.values()}
    assert ops == {
        "BFPQuantizeDequantize"
        if fmt.startswith(("bfp", "mx4", "mx9"))
        else "MXQuantizeDequantize"
    }
    assert {t: _attrs(n)["axis"] for t, n in fns.items()} == {"w": 0, "x": -1, "y": -1}
    assert all(n.domain == "com.amd.quark" for n in fns.values())
    mm = next(n for n in out.graph.node if n.op_type == "MatMul")
    assert list(mm.input) == ["x_DequantizeLinear_Output", "w_DequantizeLinear_Output"]
    assert list(mm.output) == ["y_QuantizeLinear_Input"]
    assert list(fns["y"].output) == ["y"]  # the graph output keeps its name
    # (Quark registers its two operator sets after the model's own, which include
    # every domain ONNX Runtime's optimizer writes when it runs)
    domains = [(o.domain, o.version) for o in out.opset_import if o.domain]
    # (an ONNX Runtime that loaded Quark's op library in this process lists
    # ``com.amd.quark`` at version 1000, in its own place)
    names = [d for d, _ in domains]
    assert "com.amd.quark" in names and "com.microsoft" in names
    # the constant is untouched (the node quantizes it at run time, as in Quark)
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(out.graph.initializer[0]),
        onnx.numpy_helper.to_array(model.graph.initializer[0]),
    )
    onnx.checker.check_model(out)


_EXPECTED_ATTRS = {
    "BFP16": dict(bfp_method="to_bfp", bit_width=16, block_size=8),
    "MX4": dict(bfp_method="to_bfp_prime", bit_width=11, block_size=16),
    "MX9": dict(bfp_method="to_bfp_prime", bit_width=16, block_size=16),
    "MXINT8": dict(element_dtype="int8", block_size=32),
    "MXFP8E4M3": dict(element_dtype="fp8_e4m3", block_size=32),
    "MXFP4E2M1": dict(element_dtype="fp4_e2m1", block_size=32),
}


@pytest.mark.parametrize("preset, fmt", _BLOCK_PRESETS)
def test_block_format_node_attributes_match_quarks_defaults(preset, fmt):
    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    with pytest.warns(UserWarning):
        out = q.quantize_model(_wide_matmul_model())
    attrs = _attrs(_fn_nodes(out)["x"])
    attrs.pop("axis")
    expected = {**_EXPECTED_ATTRS[preset], "rounding_mode": 2}
    if "bfp_method" in expected:  # the BFP op carries its sub-block attributes too
        expected |= dict(
            sub_block_size=2, sub_block_shift_bits=1, convert_to_bfloat_before_bfp=0
        )
    assert attrs == expected


def test_block_format_fold_weights_option_quantizes_constants_offline():
    cfg = qc.QConfig.get_default_config("BFP16")
    cfg.extra_options["BlockFormatFoldWeights"] = True
    model = _wide_matmul_model()
    with pytest.warns(UserWarning):
        out = qc.ModelQuantizer(cfg).quantize_model(model)
    assert set(_fn_nodes(out)) == {"x", "y"}
    w_in = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w_out = onnx.numpy_helper.to_array(out.graph.initializer[0])
    np.testing.assert_array_equal(w_out, _expected_weights("bfp16", w_in))


def test_block_format_conv_graph_quantizes_bias_and_relu_output():
    model = _conv_model()  # conv1(x) -> relu -> conv2, both with a bias
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("BFP16"))
    with pytest.warns(UserWarning):
        out = q.quantize_model(model)
    fns = _fn_nodes(out)
    assert set(fns) == {"x", "w1", "b1", "h", "r", "w2", "b2", "y"}
    axes = {t: _attrs(n)["axis"] for t, n in fns.items()}
    assert axes["b1"] == axes["b2"] == 0  # a 1-D constant
    assert axes["w1"] == axes["w2"] == axes["x"] == axes["h"] == 1  # Conv: channels
    # a tensor feeding two target layers gets a single node
    shared = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,16] x) => (float[N,8] y1, float[N,8] y2)
        {
            y1 = MatMul(x, wa)
            y2 = MatMul(x, wb)
        }
        """
    )
    rng = np.random.default_rng(0)
    for n in ("wa", "wb"):
        shared.graph.initializer.append(
            onnx.numpy_helper.from_array(
                rng.standard_normal((16, 8)).astype(np.float32), n
            )
        )
    with pytest.warns(UserWarning):
        out = qc.ModelQuantizer(qc.QConfig.get_default_config("BFP16")).quantize_model(
            shared
        )
    # x (once, despite two consumers) + two weights + two outputs
    assert sum(n.op_type == "BFPQuantizeDequantize" for n in out.graph.node) == 5
    assert len([n for n in out.graph.node if n.input[0] == "x"]) == 1


def test_excluded_layers_get_no_block_quantization():
    cfg = qc.QConfig.get_default_config("BFP16")
    cfg.exclude = ["y"]
    model = _wide_matmul_model()
    # Quark: "No quantizable ops in this model" -- the model comes back as given
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = qc.ModelQuantizer(cfg).quantize_model(model)
    assert out.SerializeToString() == model.SerializeToString()


@pytest.mark.skipif(
    not os.environ.get("QUARK_ONNX_OPS_LIB"),
    reason="set QUARK_ONNX_OPS_LIB to a build of Quark's ONNX custom-op library",
)
@pytest.mark.parametrize("preset, fmt", _BLOCK_PRESETS)
def test_emitted_models_run_in_quarks_op_library_and_match_numpy(preset, fmt):
    """End to end through ONNX Runtime with Quark's own compiled op library.
    (``tests/test_quark_parity.py`` additionally compares against the models
    Quark itself produces.)"""
    import onnxruntime as ort

    from onnxsim import quark_block_formats as bf

    model = _wide_matmul_model()
    q = qc.ModelQuantizer(qc.QConfig.get_default_config(preset))
    with pytest.warns(UserWarning):
        out = q.quantize_model(model)
    so = ort.SessionOptions()
    so.register_custom_ops_library(os.environ["QUARK_ONNX_OPS_LIB"])
    sess = ort.InferenceSession(
        out.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(1).standard_normal((5, 32)).astype(np.float32) * 3
    got = sess.run(None, {"x": x})[0]

    def fq(a, axis):
        return {
            "bfp16": lambda: bf.bfp16(a, axis=axis),
            "mx4": lambda: bf.bfp_prime(a, bit_width=11, axis=axis),
            "mx9": lambda: bf.bfp_prime(a, bit_width=16, axis=axis),
            "mxint8": lambda: bf.mx(a, element_dtype="int8", axis=axis),
            "mxfp8_e4m3": lambda: bf.mx(a, element_dtype="fp8_e4m3", axis=axis),
            "mxfp4_e2m1": lambda: bf.mx(a, element_dtype="fp4_e2m1", axis=axis),
        }[fmt]()

    w = onnx.numpy_helper.to_array(model.graph.initializer[0])
    expected = fq(fq(x, -1) @ fq(w, 0), -1)  # quantized x, w and the output
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_block_format_with_algo_config_is_refused():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("BFP16_ADAQUANT"))
    with pytest.raises(NotImplementedError, match="block formats"):
        q.quantize_model(_wide_matmul_model())


def test_no_adaround_variant_for_block_presets():
    qc.QConfig.get_default_config("MX9_ADAQUANT")
    with pytest.raises(ValueError):
        qc.QConfig.get_default_config("MX9_ADAROUND")


def test_unknown_algo_config_is_not_silently_dropped():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.AlgoConfig(name="some_new_algo")]
    q = qc.ModelQuantizer(cfg)
    with pytest.raises(NotImplementedError, match="some_new_algo"):
        q.quantize_model(_model(), calibration_data_reader=_Reader())
    q.quantize_model(
        _model(), calibration_data_reader=_Reader(), ignore_unsupported_algos=True
    )


def test_integer_preset_requires_reader():
    q = qc.ModelQuantizer(qc.QConfig.get_default_config("U8S8_AAWS"))
    with pytest.raises(ValueError, match="calibration_data_reader"):
        q.quantize_model(_model())


def _two_layer_model():
    rng = np.random.default_rng(1)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,8] x) => (float[N,8] y)
        {
            h = MatMul(x, w1)
            r = Relu(h)
            y = MatMul(r, w2)
        }
        """
    )
    # Random weights are attached programmatically (too large for text literals).
    model.graph.initializer.extend(
        onnx.numpy_helper.from_array(rng.standard_normal((8, 8)).astype(np.float32), n)
        for n in ("w1", "w2")
    )
    return model


def _conv_model():
    rng = np.random.default_rng(2)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 17]>
        agraph (float[1,3,8,8] x) => (float[1,4,8,8] y)
        {
            h = Conv<group=1, pads=[1,1,1,1]>(x, w1, b1)
            r = Relu(h)
            y = Conv<group=1, pads=[1,1,1,1]>(r, w2, b2)
        }
        """
    )
    scale = np.array([1.0, 10.0, 0.1, 3.0], dtype=np.float32)  # uneven channel ranges
    for name, shape in (
        ("w1", (4, 3, 3, 3)),
        ("b1", (4,)),
        ("w2", (4, 4, 3, 3)),
        ("b2", (4,)),
    ):
        arr = rng.standard_normal(shape).astype(np.float32)
        arr *= scale.reshape(-1, *[1] * (len(shape) - 1))
        model.graph.initializer.append(onnx.numpy_helper.from_array(arr, name))
    return model


def _quantize(model, algos, batches):
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = algos
    return qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=batches)


def _batches(shape, n=4):
    rng = np.random.default_rng(3)
    return [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]


def test_smooth_quant_changes_the_quantized_model():
    batches = _batches((4, 8))
    base = _quantize(_two_layer_model(), [], batches)
    out = _quantize(_two_layer_model(), [qc.SmoothQuantConfig(alpha=0.5)], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_bias_correction_changes_the_quantized_gemm_biases():
    # like Quark's, it rewrites the quantized bias of Conv / Gemm layers
    # (MatMul without a bias is left alone)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,8] x) => (float[N,8] y)
        {
            h = Gemm(x, w1, b1)
            r = Relu(h)
            y = Gemm(r, w2, b2)
        }
        """
    )
    rng = np.random.default_rng(1)
    model.graph.initializer.extend(
        onnx.numpy_helper.from_array(rng.standard_normal(s).astype(np.float32), n)
        for n, s in (("w1", (8, 8)), ("b1", (8,)), ("w2", (8, 8)), ("b2", (8,)))
    )
    batches = _batches((4, 8))
    base = _quantize(model, [], batches)
    out = _quantize(model, [qc.BiasCorrectionConfig()], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_cle_changes_a_conv_relu_conv_model():
    batches = _batches((1, 3, 8, 8))
    base = _quantize(_conv_model(), [], batches)
    out = _quantize(_conv_model(), [qc.CLEConfig()], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_cle_also_equalizes_a_gemm_relu_gemm_chain():
    rng = np.random.default_rng(0)
    model = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[4,8] x) => (float[4,4] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }
        """
    )
    for name, shape, k in (
        ("w1", (8, 6), 2.0),
        ("b1", (6,), 1.0),
        ("w2", (6, 4), 0.3),
        ("b2", (4,), 1.0),
    ):
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(
                (rng.standard_normal(shape) * k).astype(np.float32), name
            )
        )
    batches = _batches((4, 8))
    base = _quantize(model, [], batches)
    out = _quantize(model, [qc.CLEConfig()], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_adaquant_runs_and_changes_the_quantized_model():
    batches = _batches((4, 8))
    base = _quantize(_two_layer_model(), [], batches)
    algo = qc.AdaQuantConfig(num_iterations=20)
    out = _quantize(_two_layer_model(), [algo], batches)
    assert out.SerializeToString() != base.SerializeToString()


def test_adaquant_preset_runs_end_to_end():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS_ADAQUANT")
    cfg.algo_config[0].params["num_iterations"] = 20
    out = qc.ModelQuantizer(cfg).quantize_model(
        _two_layer_model(), calibration_data_reader=_batches((4, 8))
    )
    onnx.checker.check_model(out)


@pytest.mark.parametrize("preset", ["U8S8_AAWS_ADAROUND"])
def test_adaround_preset_runs_and_reports_layers(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config[0].params["num_iterations"] = 30
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning, match="numpy port of Quark.s FastFinetune"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    reports = q.last_weight_rounding["adaround"]
    assert reports and all(r.error_after <= r.error_before for r in reports)
    base = _quantize(_two_layer_model(), [], _batches((4, 8)))
    assert out.SerializeToString() != base.SerializeToString()
    onnx.checker.check_model(out)


def test_gptq_runs_through_the_compat_layer():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.GPTQConfig(act_order=True, perc_damp=0.02)]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning, match="GPTQ keeps"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    reports = q.last_weight_rounding["gptq"]
    assert reports and all(r.error_after <= r.error_before for r in reports)
    onnx.checker.check_model(out)


@pytest.mark.parametrize(
    "algo, match",
    [
        (qc.GPTQConfig(group_size=4, act_order=True), "act_order with group_size"),
    ],
)
def test_weight_rounding_options_that_change_the_meaning_are_refused(algo, match):
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [algo]
    with pytest.raises(NotImplementedError, match=match):
        qc.ModelQuantizer(cfg).quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )


def test_weight_rounding_state_is_reset_between_runs():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.GPTQConfig()]
    q = qc.ModelQuantizer(cfg)
    with pytest.warns(UserWarning):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert "gptq" in q.last_weight_rounding
    q.config.algo_config = []
    q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_weight_rounding == {}


def _amp_config(**params):
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    target = params.pop(
        "target_layer_config", qc.QLayerConfig(qc.UInt16Spec(), qc.Int8Spec())
    )
    cfg.algo_config = [qc.AutoMixprecisionConfig(target_layer_config=target, **params)]
    return cfg


def test_auto_mixprecision_moves_layers_to_the_target_activation_precision():
    q = qc.ModelQuantizer(
        _amp_config(metric_optimize_object="quality", metric_threshold=0)
    )
    with pytest.warns(UserWarning, match="AutoMixprecision"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    res = q.last_auto_mixprecision
    assert res is not None and res.moved  # threshold 0: every candidate moves
    assert res.final_score < res.baseline_score
    # the moved layers' activations really are uint16 now
    assert onnx.TensorProto.UINT16 in {i.data_type for i in out.graph.initializer}
    onnx.checker.check_model(out)


def test_auto_mixprecision_threshold_none_is_sensitivity_only():
    q = qc.ModelQuantizer(_amp_config(metric_threshold=None))
    with pytest.warns(UserWarning, match="AutoMixprecision"):
        out = q.quantize_model(
            _two_layer_model(), calibration_data_reader=_batches((4, 8))
        )
    assert q.last_auto_mixprecision.ranked and not q.last_auto_mixprecision.moved
    assert onnx.TensorProto.UINT16 not in {i.data_type for i in out.graph.initializer}


@pytest.mark.parametrize(
    "params, error, match",
    [
        ({"target_layer_config": {}}, ValueError, "dict must not be empty"),
        ({"target_layer_config": []}, ValueError, "list must not be empty"),
        ({"target_layer_config": "uint16"}, TypeError, "must be a QLayerConfig"),
        ({"shared_param_mode": "nope"}, ValueError, "shared_param_mode"),
    ],
)
def test_auto_mixprecision_invalid_forms_are_refused(params, error, match):
    q = qc.ModelQuantizer(_amp_config(**params))
    with pytest.raises(error, match=match):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))


def test_auto_mixprecision_missing_subgraph_json_is_ignored_like_quark():
    # Quark checks ``Path(subgraph_json).exists()`` and falls back to layer-wise
    # candidates (the same result as no ``subgraph_json`` at all)
    model, data = _two_layer_model(), _batches((4, 8))
    q = qc.ModelQuantizer(_amp_config(subgraph_json="does-not-exist.json"))
    with pytest.warns(UserWarning, match="does not exist"):
        out = q.quantize_model(model, calibration_data_reader=data)
    ref = qc.ModelQuantizer(_amp_config())
    with pytest.warns(UserWarning):
        want = ref.quantize_model(model, calibration_data_reader=_batches((4, 8)))
    assert out.SerializeToString() == want.SerializeToString()


def test_auto_mixprecision_same_precision_target_is_not_an_error_like_quark():
    # Quark never rejects it: the layers are still re-quantized (per-tensor
    # weights, refreshed bias scales)
    q = qc.ModelQuantizer(
        _amp_config(target_layer_config=qc.QLayerConfig(qc.UInt8Spec(), qc.Int8Spec()))
    )
    with pytest.warns(UserWarning, match="AutoMixprecision"):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_auto_mixprecision is not None


def test_auto_mixprecision_state_is_reset_between_runs():
    q = qc.ModelQuantizer(_amp_config(metric_threshold=None))
    with pytest.warns(UserWarning):
        q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_auto_mixprecision is not None
    q.config.algo_config = []
    q.quantize_model(_two_layer_model(), calibration_data_reader=_batches((4, 8)))
    assert q.last_auto_mixprecision is None


def _stream_model():
    # x -> w_in (writes the stream) -> per-channel scale (the "norm") -> w_out (reads it)
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        agraph (float[N,8] x) => (float[N,8] y)
        {
            s = MatMul(x, w_in)
            n = Mul(s, g)
            y = MatMul(n, w_out)
        }
        """
    )
    rng = np.random.default_rng(7)
    # Random weights are attached programmatically (too large for text literals).
    for name, shape, scale in (
        ("w_in", (8, 16), 1.0),
        ("g", (16,), 0.5),
        ("w_out", (16, 8), 0.3),
    ):
        arr = rng.standard_normal(shape).astype(np.float32) * scale
        if name == "g":
            arr += 1.0
        model.graph.initializer.append(onnx.numpy_helper.from_array(arr, name))
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def _quarot_cfg(tmp_path, **params):
    import json

    path = tmp_path / "rot.json"
    path.write_text(
        json.dumps(
            {"R1_pairs": [{"prev_nodes": ["s"], "next_nodes": ["y"], "norm_node": "n"}]}
        )
    )
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [
        qc.QuarotConfig(r_matrix_dim=16, r_config_path=str(path), **params)
    ]
    return cfg


def test_quarot_rotates_the_float_weights_before_quantization(tmp_path):
    import onnxruntime as ort

    batches = _batches((6, 8))
    plain = _quantize(_stream_model(), [], batches)
    q = qc.ModelQuantizer(_quarot_cfg(tmp_path))
    with pytest.warns(UserWarning, match="Quarot"):
        out = q.quantize_model(_stream_model(), calibration_data_reader=batches)
    codes = lambda m: {  # noqa: E731
        t.name: onnx.numpy_helper.to_array(t)
        for t in m.graph.initializer
        if t.name.endswith("/int8")
    }
    assert any(not np.array_equal(codes(out)[k], codes(plain)[k]) for k in codes(out))

    def run(m, x):
        sess = ort.InferenceSession(m.SerializeToString())
        return sess.run(None, {"x": x})[0]

    x = batches[0]["x"]
    ref = run(_stream_model(), x)
    err = lambda m: float(np.abs(run(m, x) - ref).max())  # noqa: E731
    assert err(out) < 5 * err(plain) + 0.05  # a different basis, same function


def test_quarot_requires_a_config_path():
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.algo_config = [qc.QuarotConfig()]
    with pytest.raises(ValueError, match="r_config_path"):
        qc.ModelQuantizer(cfg).quantize_model(
            _stream_model(), calibration_data_reader=_batches((6, 8))
        )


@pytest.mark.parametrize("preset", ["FP16", "BF16"])
def test_float_presets_refuse_an_algo_config_instead_of_dropping_it(preset):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.algo_config = [qc.SmoothQuantConfig()]
    q = qc.ModelQuantizer(cfg)
    with pytest.raises(NotImplementedError, match="float presets"):
        q.quantize_model(_model())
    with pytest.warns(UserWarning):
        out = q.quantize_model(_model(), ignore_unsupported_algos=True)
    assert "ExtendedQuantizeLinear" in _ops(out)


# -- per-layer / per-type overrides ------------------------------------------------


def _named_chain():
    model = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,8] x) => (float[3,4] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }
        """
    )
    for n, name in zip(model.graph.node, ("g1", "relu", "g2")):
        n.name = name
    rng = np.random.default_rng(0)
    for name, shape in (("w1", (8, 6)), ("b1", (6,)), ("w2", (6, 4)), ("b2", (4,))):
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(
                rng.standard_normal(shape).astype(np.float32), name
            )
        )
    return model


def _int16_layer():
    return qc.QLayerConfig(
        input_tensors=qc.Int16Spec(),
        weight=qc.Int8Spec(),
        output_tensors=qc.Int16Spec(),
    )


def _q_zp_dtypes(model):
    inits = {i.name: i for i in model.graph.initializer}
    return {
        n.input[0]: onnx.TensorProto.DataType.Name(inits[n.input[2]].data_type)
        for n in model.graph.node
        if n.op_type == "QuantizeLinear"
    }


def _quantize_with(**kwargs):
    cfg = qc.QConfig(qc.QLayerConfig(qc.Int8Spec(), qc.Int8Spec()), **kwargs)
    return qc.ModelQuantizer(cfg).quantize_model(
        _named_chain(), calibration_data_reader=_batches((3, 8))
    )


def test_qlayerconfig_input_tensors_is_the_activation_spec():
    spec = qc.Int16Spec()
    assert qc.QLayerConfig(input_tensors=spec).activation is spec
    with pytest.raises(ValueError, match="input_tensors"):
        qc.QLayerConfig(activation=qc.Int8Spec(), input_tensors=qc.Int8Spec())


def test_specific_layer_config_changes_that_layers_tensor_dtypes():
    out = _quantize_with(specific_layer_config={_int16_layer(): ["g2"]})
    dtypes = _q_zp_dtypes(out)
    assert dtypes["x/f" if "x/f" in dtypes else "x"] == "INT8"
    assert {dtypes["h1/f"], dtypes["y/f"]} == {"INT16"}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(layer_type_config={None: ["Gemm", "Relu"]}),
        dict(exclude=["^g.*", "relu"]),
    ],
)
def test_excluded_layers_stay_float(kwargs):
    out = _quantize_with(**kwargs)
    assert not [n for n in out.graph.node if n.op_type == "QuantizeLinear"]


def test_regex_and_type_overrides_and_specific_wins():
    out = _quantize_with(
        layer_type_config={_int16_layer(): ["Gemm"]},
        specific_layer_config={
            qc.QLayerConfig(
                input_tensors=qc.Int8Spec(),
                weight=qc.Int8Spec(),
                output_tensors=qc.Int8Spec(),
            ): ["^g1.*"]
        },
    )
    dtypes = _q_zp_dtypes(out)
    # g1 (specific: int8 in and out) beats the type config on its own tensors
    assert dtypes["y/f"] == "INT16"
    assert dtypes[next(k for k in dtypes if k.startswith("x"))] == "INT8"


def test_override_errors():
    with pytest.raises(ValueError, match="no node named"):
        _quantize_with(specific_layer_config={_int16_layer(): ["nope"]})
    with pytest.raises(ValueError, match="matches no node"):
        _quantize_with(specific_layer_config={_int16_layer(): ["^zzz.*"]})
    with pytest.raises(NotImplementedError, match="subgraph"):
        _quantize_with(specific_layer_config={_int16_layer(): [(["g1"],)]})
    wide_weight = qc.QLayerConfig(
        input_tensors=qc.Int8Spec(), weight=qc.Int16Spec(), output_tensors=None
    )
    with pytest.raises(NotImplementedError, match="weight dtype"):
        _quantize_with(specific_layer_config={wide_weight: ["g1"]})
