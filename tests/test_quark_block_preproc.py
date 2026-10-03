"""Quark's float pre-processing in the block-format (BFP / MX), bfloat16 / float16 and
``MATMUL_NBITS`` flows of :mod:`onnxsim.quark_compat`, without Quark. The parity of the
quantized graphs against the real package is
``tests/test_quark_block_preproc_parity.py``; the models are
``tests/_quark_block_common.py``.

The runtime optimizers (onnxslim, ONNX Runtime's basic level) are switched off
(``UseRuntimeOptimizers=False``) where a test wants onnxsim's own reproductions of the
passes, so the result does not depend on what is installed.
"""

import warnings

import numpy as np
import onnx
import pytest
from _quark_block_common import (
    PATTERNS,
    block13,
    block17,
    block20,
    bn_concat,
    calibration_data,
    identity_only,
    ops,
)

from onnxsim import quark_compat as qc
from onnxsim.quark_marking import quark_sorted

_BLOCK = ["BFP16", "MX4", "MX9", "MXFP4E2M1", "MXFP8E4M3", "MXINT8"]
_HALF = ["BF16", "FP16"]


def _quantize(preset, model, runtime=False, **extra):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update({"UseRuntimeOptimizers": runtime, **extra})
    if preset == "MATMUL_NBITS":
        cfg.extra_options["MatMulNBitsParams"] = {
            "GroupSize": 16,
            "Symmetric": True,
            "Bits": 4,
            "AccuracyLevel": 1,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=calibration_data(model)
        )


def _domains(model):
    return [(o.domain, o.version) for o in model.opset_import]


def _quantized_tensors(model):
    """The tensors that carry a block node or a Q/DQ pair."""
    out = set()
    for n in model.graph.node:
        if n.op_type in ("BFPQuantizeDequantize", "MXQuantizeDequantize"):
            out.add(n.input[0].removesuffix("_QuantizeLinear_Input"))
        elif n.op_type == "ExtendedQuantizeLinear":
            out.add(n.input[0].removesuffix("_QuantizeLinear_Input"))
    return out


# -- the float graph Quark quantizes --------------------------------------------------------


@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_batch_norm_folds_into_the_conv_before_quantization(preset):
    out = _quantize(preset, PATTERNS["conv_bn"]())
    assert "BatchNormalization" not in ops(out)
    assert ops(out).count("Conv") == 2
    # the folded bias is quantized with the rest: Conv, Conv, Relu and their tensors
    assert {"w1", "b1", "w2", "b2"} <= _quantized_tensors(out)
    # the BatchNorm's own constants are gone with it
    assert not {"s", "bb", "mu", "var"} & {t.name for t in out.graph.initializer}


@pytest.mark.parametrize(
    "name, gone, now",
    [
        ("identity", "Identity", "Conv"),
        ("pad_conv", "Pad", "Conv"),
        ("hardswish", "HardSwish", "HardSigmoid"),
    ],
)
@pytest.mark.parametrize("preset", ["BFP16", "MXINT8", "BF16", "FP16", "MATMUL_NBITS"])
def test_identity_pad_and_hardswish_are_rewritten_first(name, gone, now, preset):
    out = _quantize(preset, PATTERNS[name]())
    assert gone not in ops(out) and now in ops(out)
    if name == "pad_conv":
        (conv,) = [n for n in out.graph.node if n.op_type == "Conv"]
        pads = next(a for a in conv.attribute if a.name == "pads")
        assert list(pads.ints) == [1, 1, 1, 1]


@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_reduce_mean_becomes_a_global_average_pool(preset):
    out = _quantize(preset, PATTERNS["reducemean"]())
    assert "GlobalAveragePool" in ops(out) and "ReduceMean" not in ops(out)


def test_matmul_nbits_leaves_reduce_mean_alone():
    """Its plain ``QDQ`` format does not switch the extended format's conversions on."""
    out = _quantize("MATMUL_NBITS", PATTERNS["reducemean"]())
    assert "ReduceMean" in ops(out) and "GlobalAveragePool" not in ops(out)


def test_conversions_follow_their_options():
    out = _quantize(
        "BFP16",
        PATTERNS["reducemean"](),
        ConvertReduceMeanToGlobalAvgPool=False,
    )
    assert "ReduceMean" in ops(out)
    out = _quantize("BFP16", PATTERNS["split_concat"]())
    assert "Split" not in ops(out) and ops(out).count("Slice") == 2
    out = _quantize("BFP16", PATTERNS["split_concat"](), ConvertSplitToSlice=False)
    assert "Split" in ops(out)


@pytest.mark.parametrize("preset", ["BFP16", "MX6", "BF16", "FP16", "MATMUL_NBITS"])
def test_layer_norm_and_gelu_fuse_by_opset(preset):
    for build, norm, gelu in ((block13, False, False), (block17, True, False)):
        out = _quantize(preset, build())
        assert ("LayerNormalization" in ops(out)) is norm
        assert ("Gelu" in ops(out)) is gelu
        assert ("Pow" in ops(out)) is not norm
    out = _quantize(preset, block20())
    assert {"LayerNormalization", "Gelu"} <= set(ops(out))
    assert "Erf" not in ops(out) and "Pow" not in ops(out)


def test_fusion_flags_switch_the_fusions_off():
    # (without ONNX Runtime's own optimizer, which fuses a torch LayerNorm / Gelu first)
    out = _quantize(
        "BFP16", block20(), FuseLayerNorm=False, FuseGelu=False, OptimizeModel=False
    )
    assert {"Pow", "Erf"} <= set(ops(out))
    assert not {"LayerNormalization", "Gelu"} & set(ops(out))


@pytest.mark.parametrize("preset", ["BFP16", "BF16", "MATMUL_NBITS"])
def test_skip_preprocess_skips_all_of_it(preset):
    model = block17()
    out = _quantize(preset, model, SkipPreprocess=True, ConvertOpsetVersion=20)
    assert {"Pow", "ReduceMean", "Erf"} <= set(ops(out))
    assert "LayerNormalization" not in ops(out)
    # (its opset is not converted either)
    assert next(o.version for o in out.opset_import if o.domain == "") == 17
    for name in ("conv_bn", "identity", "pad_conv"):
        out = _quantize(preset, PATTERNS[name](), SkipPreprocess=True)
        assert {"BatchNormalization", "Identity", "Pad"} & set(ops(out))


def test_opset_conversion_comes_first():
    out = _quantize("BFP16", block17(), ConvertOpsetVersion=20)
    assert next(o.version for o in out.opset_import if o.domain == "") == 20
    # ... so the Gelu that only fuses from opset 20 on is fused
    assert "Gelu" in ops(out)
    out = _quantize("MATMUL_NBITS", block17(), ConvertOpsetVersion=20)
    assert "Gelu" in ops(out)


def test_bn_quark_is_asked_to_quantize_is_not_folded_by_its_own_pass():
    """With ``QuantizeAllOpTypes`` the BatchNormalization is on Quark's op list: its own
    folding passes (after a Concat here) leave it, and the extended format's conversion
    turns it into the depthwise 1x1 Conv that is then quantized."""
    folded = _quantize("BFP16", bn_concat())
    assert ops(folded).count("Conv") == 2 and "BatchNormalization" not in ops(folded)
    kept = _quantize("BFP16", bn_concat(), QuantizeAllOpTypes=True)
    assert ops(kept).count("Conv") == 3 and "BatchNormalization" not in ops(kept)
    conv = [n for n in kept.graph.node if n.op_type == "Conv"][-1]
    assert next(a.i for a in conv.attribute if a.name == "group") == 8


def test_shared_bias_is_copied_for_matmul_nbits_only():
    """``CopyBiasInit`` runs when both Quark types are integer (MATMUL_NBITS's int8 ones);
    the block and half formats are not."""
    nbits = _quantize("MATMUL_NBITS", PATTERNS["shared_bias"](), runtime=True)
    assert any(t.name.startswith("duplicated") for t in nbits.graph.initializer)
    for preset in ("BFP16", "BF16"):
        out = _quantize(preset, PATTERNS["shared_bias"](), runtime=True)
        assert not any(t.name.startswith("duplicated") for t in out.graph.initializer)


# -- the quantized graph -------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["BFP16", "MXFP8E4M3", "BF16", "FP16"])
def test_a_bare_clip_output_is_quantized(preset):
    """Quark's Clip quantizer marks its output whatever follows (here: nothing)."""
    out = _quantize(preset, PATTERNS["clip_bare"]())
    assert {"y"} <= _quantized_tensors(out)
    assert out.graph.output[0].name == "y"


@pytest.mark.parametrize("preset", ["BFP16", "BF16"])
def test_relu6_clip_between_convolutions(preset):
    out = _quantize(preset, PATTERNS["clip_relu6"]())
    quantized = _quantized_tensors(out)
    assert {"x", "w1", "w2"} <= quantized
    # the Clip's own output feeds a Conv: the Conv marks it
    assert any(t not in {"x", "w1", "b1", "w2", "b2", "y"} for t in quantized)


def test_half_precision_pairs_of_a_data_movement_chain_share_the_scale():
    out = _quantize("BF16", PATTERNS["movement_chain"]())
    pairs = [n for n in out.graph.node if n.op_type == "ExtendedQuantizeLinear"]
    scales = {n.input[1] for n in pairs}
    # x, w, b, c, then t / r / u / y read c's scale and zero point
    assert len(pairs) == 8 and len(scales) == 4
    assert {n.input[2] for n in pairs} == {
        s.replace("_scale", "_zero_point") for s in scales
    }
    # block formats have no such parameters
    out = _quantize("BFP16", PATTERNS["movement_chain"]())
    assert not any(n.op_type == "ExtendedQuantizeLinear" for n in out.graph.node)


def test_pad_before_an_average_pool_goes_without_a_pair_in_the_half_formats():
    out = _quantize("FP16", PATTERNS["pad_avgpool"](), OptimizeModel=False)
    assert "p" not in _quantized_tensors(out)
    out = _quantize("BFP16", PATTERNS["pad_avgpool"](), OptimizeModel=False)
    assert "p" in _quantized_tensors(out)


def test_remove_qdq_options_keep_the_pair_between_conv_and_relu():
    model = PATTERNS["conv_bn"]()
    default = _quantize("FP16", model)
    kept = _quantize("FP16", model, RemoveQDQConvRelu=False)
    assert len(_quantized_tensors(kept)) == len(_quantized_tensors(default)) + 1


@pytest.mark.parametrize("name", sorted(PATTERNS))
@pytest.mark.parametrize("preset", ["BFP16", "BF16", "MATMUL_NBITS"])
def test_the_graph_is_in_quarks_topological_order(name, preset):
    """Quark's quantizers end with their own topological sort; sorting the result
    again changes nothing."""
    out = _quantize(preset, PATTERNS[name]())
    again = quark_sorted(out)
    assert [n.name or n.output[0] for n in out.graph.node] == [
        n.name or n.output[0] for n in again.graph.node
    ]
    onnx.checker.check_model(out)


# -- domains and the models Quark leaves alone -----------------------------------------------------


@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_domains_follow_quark_with_the_pre_processing_off(preset):
    out = _quantize(preset, PATTERNS["conv_bn"](), SkipPreprocess=True)
    assert _domains(out) == [("", 17), ("com.microsoft", 1), ("com.amd.quark", 1)]


def test_matmul_nbits_registers_the_contrib_domain_only():
    out = _quantize("MATMUL_NBITS", PATTERNS["matmul_add"](), SkipPreprocess=True)
    assert _domains(out) == [("", 17), ("com.microsoft", 1)]


@pytest.mark.parametrize("preset", ["BFP16", "MXFP4E2M1", "FP16", "MATMUL_NBITS"])
@pytest.mark.parametrize("target", [None, 20])
def test_nothing_to_quantize_returns_the_model_as_given(preset, target):
    extra = {} if target is None else {"ConvertOpsetVersion": target}
    model = identity_only()
    out = _quantize(preset, model, **extra)
    assert ops(out) == ["Identity", "Identity"]
    assert _domains(out) == [("", 17)]
    assert out.SerializeToString() == model.SerializeToString()


def test_bfloat16_quantizes_every_op_type_of_the_model():
    out = _quantize("BF16", identity_only())
    assert ops(out).count("ExtendedQuantizeLinear") == 2
    assert _domains(out) == [("", 17), ("com.microsoft", 1), ("com.amd.quark", 1)]


def test_constants_the_pre_processing_left_behind_are_dropped():
    """Quark's quantizers end with ``clean_initializers``."""
    for preset in ("BFP16", "BF16", "MATMUL_NBITS"):
        out = _quantize(preset, block17(), SkipPreprocess=True)
        used = {x for n in out.graph.node for x in n.input}
        assert {t.name for t in out.graph.initializer} <= used | {
            o.name for o in out.graph.output
        }
        assert "a_three" not in {t.name for t in out.graph.initializer}


def test_a_leftover_op_outside_the_registries_is_left_alone():
    """The block formats quantize the op types of Quark's registries only (BF16 every
    op type of the model): an Abs stays as it is."""
    model = onnx.parser.parse_model(
        """<ir_version: 8, opset_import: ["": 17]>
        g (float[2,4] x) => (float y) { a = Abs(x)\n y = Relu(a) }"""
    )
    block = _quantize("BFP16", model)
    assert not {"a"} & _quantized_tensors(block)
    half = _quantize("BF16", model)
    assert {"x", "a", "y"} <= _quantized_tensors(half)


def test_outputs_of_the_rewritten_float_graph_are_unchanged():
    """The passes are exact rewrites of the float model: with the block format's
    rounding left out (BlockFormatActivations off, weights folded as given), the model
    computes what the input did."""
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    for name in (
        "conv_bn",
        "identity",
        "pad_conv",
        "reducemean",
        "hardswish",
        "block20",
    ):
        model = PATTERNS[name]()
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        x = rng.standard_normal(shape).astype(np.float32)

        def run(m):
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.log_severity_level = 4
            s = ort.InferenceSession(
                m.SerializeToString(), so, providers=["CPUExecutionProvider"]
            )
            return s.run(None, {"x": x})[0]

        cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
        cfg.extra_options["UseRuntimeOptimizers"] = False
        cfg.extra_options["MatMulNBitsParams"] = {"GroupSize": 16, "Bits": 4}
        cfg.exclude = [n.name for n in model.graph.node if n.op_type == "MatMul"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = qc.ModelQuantizer(cfg).quantize_model(model)
        np.testing.assert_allclose(
            run(out), run(model), rtol=1e-4, atol=1e-5, err_msg=name
        )


def test_bfloat16_constants_outside_the_normal_range_are_clipped_like_quark():
    out = _quantize("BF16", PATTERNS["tiny_constants"]())
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in out.graph.initializer}
    lo = np.float32(1.17549435e-38)
    np.testing.assert_array_equal(inits["w"].reshape(-1)[:3], [lo, 0.0, -lo])
    np.testing.assert_array_equal(inits["b"][1:3], [lo, 0.0])
    assert inits["b"][3] == np.float32(3.38953139e38)
    for preset in ("FP16", "BFP16"):
        out = _quantize(preset, PATTERNS["tiny_constants"]())
        w = {t.name: onnx.numpy_helper.to_array(t) for t in out.graph.initializer}["w"]
        assert w.reshape(-1)[0] == np.float32(1e-39)
