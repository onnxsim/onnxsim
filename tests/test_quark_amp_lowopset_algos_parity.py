"""Parity with the real AMD Quark (0.13) for: AutoMixprecision ``dual_quant_nodes``
on every kind of mix, half / block activations over any constant format, and the
algorithm flows (CLE, SmoothQuant, BiasCorrection, AdaRound, AdaQuant) on models
below opset 13. Skipped unless ``quark.onnx`` (and, for fine-tuning, ``torch``)
imports. Quark-free counterparts: ``test_quark_amp_mixing.py``,
``test_quark_low_opset.py``, ``test_quark_finetune.py``.

Findings pinned here (probed with Quark 0.13):

* ``dual_quant_nodes`` is a post-processing of the *final* mixed model; a threshold
  walk's demoted candidate keeps its tensors in Quark's "promoted" set.
* Quark's AdaQuant / AdaRound quantizers call ``torch.clamp`` with tensor bounds,
  whose backward gives half the gradient to a value exactly on a bound. With that
  rule AdaQuant codes are bit-identical to Quark's on MatMul / Gemm, grouped and
  1-D Conv, LayerNorm, Gelu and Tanh layers. The remainder is 2-D ``Conv``:
  oneDNN picks its accumulation order by shape (a single sequential FMA chain
  over ``(kh, kw, ic)`` for N=2, 6x6, ``cin <= 16``; 16-channel partial sums for
  larger ``cin``; yet another order for e.g. N=1, 5x5, 16->16), so no single
  numpy order reproduces it; with torch's own conv forward patched in (test time
  only) the codes are identical again. Long runs on such layers are therefore
  compared with a tolerance relative to the code range.
"""

import contextlib
import io
import os
import warnings

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

warnings.filterwarnings("ignore")

import test_quark_amp_parity as AP  # noqa: E402  (also sets up the Quark imports)
import test_quark_finetune_coverage_parity as C  # noqa: E402
import test_quark_low_opset_parity as L  # noqa: E402
from test_quark_low_opset import MODELS, _data, _Reader  # noqa: E402

from onnxsim import quark_compat as qc  # noqa: E402

P = C.P
quark_onnx = AP.quark_onnx
pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)
needs_lib = AP._needs_lib
needs_torch = pytest.mark.skipif(P.torch is None, reason="torch is not installed")


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _at(model, opset):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for o in m.opset_import:
        if o.domain == "":
            o.version = opset
    return m


# == dual_quant_nodes =========================================================


def _dual(name, base, target, kw, **extra):
    quark, mine, data = AP._strict_case(
        name, base, target, dict(kw, dual_quant_nodes=True), **extra
    )
    extra_nodes = lambda m: sum("_additional_" in n.name for n in m.graph.node)  # noqa: E731
    assert extra_nodes(mine) == extra_nodes(quark)
    return extra_nodes(quark)


@pytest.mark.parametrize("case", sorted(AP._MIX))
def test_dual_quant_nodes_integer_mixes_match_quark(case):
    name, base, target, kw = AP._MIX[case]
    _dual(name, base, target, kw)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(k, marks=needs_lib) if AP._XMIX[k][4] else k
        for k in sorted(AP._XMIX)
    ],
)
def test_dual_quant_nodes_half_block_and_pof2_mixes_match_quark(case):
    name, base, target, kw, _ = AP._XMIX[case]
    _dual(name, base, target, kw)


@needs_lib
@pytest.mark.parametrize("case", sorted(AP._FORMS))
def test_dual_quant_nodes_target_forms_match_quark(case):
    name, base, target, kw = AP._FORMS[case]
    _dual(name, base, target, kw)


def test_the_dual_cases_really_insert_boundary_pairs():
    for case in (
        "act-u16-first",
        "weights-int16-second-only",
        "conv-signed-act-second",
    ):
        name, base, target, kw = AP._MIX[case]
        assert _dual(name, base, target, kw) > 0
    name, base, target, kw, _ = AP._XMIX["u8-bfp16-second"]
    assert _dual(name, base, target, kw) > 0


@needs_lib
@pytest.mark.parametrize(
    "table, case, amp",
    [
        ("mix", "s16-to-s8", dict(metric_threshold=0.2)),
        ("x", "u8-mxfp8-all", dict(metric_threshold=0.5)),
        ("x", "mxint8-mxfp8", dict(metric_threshold=0.5)),
        ("x", "u8-pof2-all", dict(metric_threshold=0.5)),
        ("x", "u8-bfp16-second", dict(metric_threshold=0.3)),
        ("x", "bf16-fp16", dict(metric_threshold=0.3)),
        ("x", "xint8-bf16-conv", dict(metric_threshold=0.3)),
        (
            "x",
            "u8-bf16-fanout",
            dict(metric_threshold=0.15, metric_optimize_object="quality"),
        ),
    ],
)
def test_dual_quant_nodes_after_a_threshold_walk_match_quark(table, case, amp):
    """(A demoted candidate's tensors stay "promoted" in Quark.)"""
    spec = (AP._MIX if table == "mix" else AP._XMIX)[case]
    name, base, target, kw = spec[:4]
    _dual(name, base, target, {**kw, **amp})


_P = lambda s: (s, s)  # noqa: E731
_MODEL_CASES = [
    (AP.U8, AP.U16),
    (AP.U8, ("UInt16Spec", "Int16Spec")),
    (AP.U8, _P("BFloat16Spec")),
    (AP.U8, _P("XInt8Spec")),
    (_P("BFP16Spec"), _P("BFloat16Spec")),
]


@needs_lib
@pytest.mark.parametrize("name", ["conv", "fanout"])
@pytest.mark.parametrize("base, target", _MODEL_CASES)
@pytest.mark.parametrize(
    "kw",
    [
        {"no_input_qdq_shared": True},
        {"shared_param_mode": "unshare", "include_layers": ["n0_Conv", "n0_Gemm"]},
    ],
    ids=["shared", "unshare"],
)
def test_dual_quant_nodes_on_conv_and_fanout_graphs_match_quark(name, base, target, kw):
    _dual(name, base, target, kw)


@pytest.mark.parametrize("name", ["mlp", "conv", "fanout"])
@pytest.mark.parametrize(
    "amp",
    [
        {},
        {"MetricThreshold": 0.3},
        {"MetricThreshold": 0.1},
        {"IncludeLayers": "first"},
    ],
    ids=["all", "t0.3", "t0.1", "first"],
)
def test_s16s16_mixed_s8s8_with_dual_nodes_matches_quark(name, amp):
    model = AP.MODELS[name]()
    data = AP._data(model)
    amp = dict(amp, DualQuantNodes=True)
    if amp.get("IncludeLayers") == "first":
        amp["IncludeLayers"] = [AP._first_layer(name)]
    quark = AP._quark_preset(model, data, "S16S16_MIXED_S8S8", amp)
    mine = AP._mine_preset(model, data, "S16S16_MIXED_S8S8", amp)
    AP._assert_same_strict(quark, mine, data)


# == half / block activations over any constant format =========================

_ACTS = [
    "BFloat16Spec",
    "Float16Spec",
    "BFP16Spec",
    "MX4Spec",
    "MX9Spec",
    "MXInt8Spec",
    "MXFP8E4M3Spec",
]
_WTS = [
    "BFloat16Spec", "Float16Spec", "BFP16Spec", "MX4Spec", "MX6Spec", "MX9Spec",
    "MXFP4E2M1Spec", "MXFP8E5M2Spec", "MXInt8Spec", "Int8Spec", "UInt8Spec",
]  # fmt: skip


def _plain(model, data, act, wt):
    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx.quantization.config import spec as qspec

    it = iter(data)

    class R(CalibrationDataReader):
        def get_next(self):
            return next(it, None)

    onnx.save(model, "float.onnx")
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        cfg = quark_onnx.QConfig(
            global_config=quark_onnx.QLayerConfig(
                activation=getattr(qspec, act)(), weight=getattr(qspec, wt)()
            )
        )
        quark_onnx.ModelQuantizer(cfg).quantize_model("float.onnx", "q.onnx", R())
    cfg = qc.QConfig(
        global_config=qc.QLayerConfig(
            activation=getattr(qc, act)(), weight=getattr(qc, wt)()
        )
    )
    mine = qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=data)
    return onnx.load("q.onnx"), mine


@needs_lib
@pytest.mark.parametrize("wt", _WTS)
@pytest.mark.parametrize("act", _ACTS)
def test_generic_half_block_pairs_match_quark_on_an_mlp(act, wt):
    model = AP._mlp()
    data = AP._data(model)
    q, m = _plain(model, data, act, wt)
    AP._assert_same_strict(q, m, data)


@needs_lib
@pytest.mark.parametrize(
    "wt", ["Float16Spec", "MX6Spec", "MXInt8Spec", "Int8Spec", "UInt8Spec"]
)
@pytest.mark.parametrize("act", ["BFloat16Spec", "BFP16Spec", "MX4Spec", "MXInt8Spec"])
def test_generic_half_block_pairs_match_quark_on_a_conv_net(act, wt):
    model = AP._convnet()
    data = AP._data(model)
    q, m = _plain(model, data, act, wt)
    AP._assert_same_strict(q, m, data)


_BASES = {
    "bf16-bfp16": ("BFloat16Spec", "BFP16Spec"),
    "bf16-mx4": ("BFloat16Spec", "MX4Spec"),
    "bf16-int8": ("BFloat16Spec", "Int8Spec"),
    "bf16-uint8": ("BFloat16Spec", "UInt8Spec"),
    "bf16-fp16": ("BFloat16Spec", "Float16Spec"),
    "fp16-mx6": ("Float16Spec", "MX6Spec"),
    "bfp16-mxint8": ("BFP16Spec", "MXInt8Spec"),
    "bfp16-int8": ("BFP16Spec", "Int8Spec"),
    "mx4-bf16": ("MX4Spec", "BFloat16Spec"),
    "mxint8-mxfp8": ("MXInt8Spec", "MXFP8E4M3Spec"),
}
_TARGETS = {
    "int8": _P("Int8Spec"),
    "bf16": _P("BFloat16Spec"),
    "mxint8": _P("MXInt8Spec"),
}


@needs_lib
@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("target", sorted(_TARGETS))
@pytest.mark.parametrize("base", sorted(_BASES))
def test_amp_over_mixed_format_baselines_matches_quark(base, target, dual):
    AP._strict_case(
        "mlp",
        _BASES[base],
        _TARGETS[target],
        {"include_layers": ["n2_Gemm"], "dual_quant_nodes": dual},
    )


# == algorithm flows below opset 13 ============================================

_ALGOS = {
    "sq": (lambda: quark_onnx.SmoothQuantConfig(alpha=0.5), lambda: qc.SmoothQuantConfig(alpha=0.5)),
    "cle": (lambda: quark_onnx.CLEConfig(), lambda: qc.CLEConfig()),
    "bc": (lambda: quark_onnx.BiasCorrectionConfig(), lambda: qc.BiasCorrectionConfig()),
}  # fmt: skip


def _named(model):
    for i, n in enumerate(model.graph.node):  # (Quark keys several algorithms by name)
        n.name = f"n{i}_{n.op_type}"
    return onnx.shape_inference.infer_shapes(model)


def _algo_pair(model, shape, algo, spec):
    from onnxruntime.quantization import CalibrationDataReader

    data = _data(shape)
    it = iter(data)

    class R(CalibrationDataReader):
        def get_next(self):
            return next(it, None)

        def reset_iter(self):
            pass

    onnx.save(model, "src.onnx")
    if os.path.exists("dst.onnx"):
        os.remove("dst.onnx")
    # (the generic config does not set Quark's ForceQuantizeNoInputCheck, which
    # every preset does: set on both sides)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        cfg = quark_onnx.QConfig(
            global_config=quark_onnx.QLayerConfig(
                activation=getattr(quark_onnx, spec[0])(),
                weight=getattr(quark_onnx, spec[1])(),
            ),
            algo_config=[_ALGOS[algo][0]()],
            ForceQuantizeNoInputCheck=True,
        )
        quark_onnx.ModelQuantizer(cfg).quantize_model("src.onnx", "dst.onnx", R())
    cfg = qc.QConfig(
        global_config=qc.QLayerConfig(
            activation=getattr(qc, spec[0])(), weight=getattr(qc, spec[1])()
        ),
        algo_config=[_ALGOS[algo][1]()],
        ForceQuantizeNoInputCheck=True,
    )
    mine = qc.ModelQuantizer(cfg).quantize_model(
        model, calibration_data_reader=_Reader(data)
    )
    return onnx.load("dst.onnx"), mine, shape


@pytest.mark.parametrize("opset", [11, 12])
@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize(
    "algo, spec",
    [
        ("cle", ("UInt8Spec", "Int8Spec")),
        ("sq", ("UInt8Spec", "Int8Spec")),
        ("bc", ("UInt8Spec", "Int8Spec")),
        ("bc", ("XInt8Spec", "XInt8Spec")),
    ],
    ids=["cle", "smoothquant", "bias-correction", "bias-correction-xint8"],
)
def test_cle_smoothquant_and_bias_correction_match_quark_below_opset_13(
    algo, spec, name, opset
):
    model, shape = MODELS[name](opset)
    q, m, shape = _algo_pair(_named(model), shape, algo, spec)
    msg = f"{algo}/{spec[0]} {name}@{opset}"
    L.assert_same_graph(q, m, msg=msg)
    L.assert_same_outputs(q, m, shape, msg=msg)


# -- fine-tuning ---------------------------------------------------------------


@needs_torch
@pytest.mark.parametrize("opset", [11, 12])
@pytest.mark.parametrize("kind", ["A", "B", "D", "E", "F"])
@pytest.mark.parametrize("algo", ["adaround", "adaquant"])
def test_finetuning_codes_equal_quarks_below_opset_13(kind, algo, opset):
    """Given Quark's torch stream (AdaQuant: a short run, see the module doc)."""
    model, data = P._build(kind)
    model = _at(model, opset)
    q = P._quark_quantize(model, data, "A8W8")[0]
    extra = (
        dict(UpdateBias=True, LearningRate=1e-3, NumIterations=30, BatchSize=2)
        if algo == "adaquant"
        else {}
    )
    quark_out, mine, q, *_ = C._both(model, data, P._ff(algo, **extra), q=q)
    assert C._changed(q, quark_out)
    assert max(C._mismatch(quark_out, mine).values()) == 0.0


def test_a_weight_on_its_grid_edge_trains_like_quark():
    """The clamp-tie rule: the largest weight of a per-tensor grid with a
    power-of-two scale sits exactly on code 127 (a Gemm: bit-exact class)."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    w = np.round(rng.standard_normal((16, 8)) * 20).astype(np.float32)
    w = np.clip(w, -126, 126) / 64
    w[0, 0] = 127 / 64
    model = onnx.parser.parse_model(
        '<ir_version: 10, opset_import: ["": 13]> g (float[N,16] x) => (float[N,8] y)'
        " { y = Gemm(x, w) }"
    )
    model.graph.initializer.append(numpy_helper.from_array(w, "w"))
    model.graph.node[0].name = "n0"
    data = [{"x": rng.standard_normal((4, 16)).astype(np.float32)} for _ in range(4)]
    q = P._quark_quantize(model, data, "A8W8")[0]
    ff = P._ff("adaquant", LearningRate=1e-2, NumIterations=20, BatchSize=2)
    quark_out, mine, q, *_ = C._both(model, data, ff, q=q)
    assert C._changed(q, quark_out)
    assert max(C._mismatch(quark_out, mine).values()) == 0.0


@needs_torch
@pytest.mark.parametrize("opset", [11, 12])
def test_long_adaquant_on_2d_conv_layers_stays_within_a_few_codes(opset):
    """2-D Conv forward accumulation order is oneDNN's (module doc): compare
    relative to the 256-code range."""
    model, data = P._build("A")
    model = _at(model, opset)
    q = P._quark_quantize(model, data, "A8W8")[0]
    ff = P._ff(
        "adaquant", UpdateBias=True, LearningRate=1e-3, NumIterations=100, BatchSize=2
    )
    quark_out, mine, q, *_ = C._both(model, data, ff, q=q)
    ca, cb = C._codes_all(quark_out), C._codes_all(mine)
    for k in ca:
        # (a platform-dependent gap -- a few codes on x86, more on aarch64 BLAS --
        # so bound it relative to the 8-bit code range, not by a fixed count)
        assert np.abs(ca[k] - cb[k]).max() <= 0.05 * 256, k
        assert np.mean(ca[k] != cb[k]) <= 0.75, k


@needs_torch
def test_with_torchs_conv_forward_the_conv_chain_is_bit_identical():
    """Test-time evidence for the module doc: replacing the numpy conv forward by
    torch's own makes AdaQuant on the 2-D conv chain bit-identical."""
    import torch
    import torch.nn.functional as F

    from onnxsim import quark_finetune as qf

    original = qf._ConvOp.forward

    def forward(self, x, w):
        y, ctx = original(self, x, w)
        if x.dtype == np.float32 and self.group == 1 and self.pad_layer is None:
            y = F.conv2d(
                torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)),
                None,
                stride=self.strides,
                padding=[b for b, _ in self.widths],
                dilation=self.dil,
            ).numpy()
        return y, ctx

    model, data = P._build("A")
    q = P._quark_quantize(model, data, "A8W8")[0]
    ff = P._ff(
        "adaquant", UpdateBias=True, LearningRate=1e-3, NumIterations=100, BatchSize=2
    )
    qf._ConvOp.forward = forward
    try:
        quark_out, mine, q, *_ = C._both(model, data, ff, q=q)
    finally:
        qf._ConvOp.forward = original
    assert max(C._mismatch(quark_out, mine).values()) == 0.0


# -- end to end through the presets, Quark's torch stream replayed -----------------


def _e2e_finetune(model, shape, preset, **extra):
    from onnxsim import quark_finetune as qf

    data = _data(shape)
    algo = P.ff_algorithm(preset)
    ff = P._ff(algo, **{"NumIterations": 40, **extra})
    if algo == "adaquant":
        ff.update(UpdateBias=True, LearningRate=1e-3)
    original = qf.finetune

    def patched(float_model, quantized, calibration, opt, **kw):
        kw.update(C._replay(float_model, calibration, quantized, ff))
        return original(float_model, quantized, calibration, opt, **kw)

    qf.finetune = patched
    try:  # (ours first: Quark's fine-tuning leaves custom-op domains registered
        cfg = qc.QConfig.get_default_config(preset)  # that ONNX Runtime's optimizer
        cfg.extra_options["FastFinetune"] = dict(ff)  # would then add to our model)
        for a in cfg.algo_config:
            a.params["guard"] = False
        mine = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=P._Reader(data)
        )
    finally:
        qf.finetune = original
    quark = P._quark_quantize(model, data, preset, **ff)[0]
    for m in (quark, mine):  # (version-1000 domains are Quark's registered custom ops)
        keep = [o for o in m.opset_import if o.version != 1000]
        del m.opset_import[:]
        m.opset_import.extend(keep)
    return quark, mine


@needs_torch
@pytest.mark.parametrize("opset", [11, 12])
@pytest.mark.parametrize("name", ["mlp", "conv_relu_pool", "residual"])
@pytest.mark.parametrize(
    "preset",
    [
        "A8W8_ADAROUND",
        "INT8_CNN_ACCURATE",
        "XINT8_ADAROUND",
        "U8S8_AAWS_ADAROUND",
        "INT8_TRANSFORMER_ACCURATE",
    ],
)
def test_adaround_presets_match_quark_below_opset_13(preset, name, opset):
    model, shape = MODELS[name](opset)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    q, m = _e2e_finetune(model, shape, preset)
    L.assert_same_graph(q, m, msg=f"{preset} {name}@{opset}")
    L.assert_same_outputs(q, m, shape, msg=f"{preset} {name}@{opset}")


@needs_torch
@pytest.mark.parametrize("opset", [11, 12])
@pytest.mark.parametrize("name", ["mlp", "matmul_add", "convt"])
@pytest.mark.parametrize(
    "preset", ["A8W8_ADAQUANT", "S8S8_AAWS_ADAQUANT", "XINT8_ADAQUANT"]
)
def test_short_adaquant_presets_match_quark_below_opset_13(preset, name, opset):
    model, shape = MODELS[name](opset)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    q, m = _e2e_finetune(model, shape, preset, NumIterations=4)
    L.assert_same_graph(q, m, msg=f"{preset} {name}@{opset}")
    L.assert_same_outputs(q, m, shape, msg=f"{preset} {name}@{opset}")


@needs_torch
@pytest.mark.parametrize("opset", [11, 12])
@pytest.mark.parametrize("name", ["mlp", "matmul_add", "bn_conv"])
@pytest.mark.parametrize("preset", ["A16W8_ADAROUND", "INT16_CNN_ACCURATE"])
def test_16_bit_adaround_presets_stay_within_a_code_of_quark(preset, name, opset):
    """float32 vs float64 decides single 16-bit-grid codes (opset-independent)."""
    model, shape = MODELS[name](opset)
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    q, m = _e2e_finetune(model, shape, preset)
    assert L._brief(q) == L._brief(m)
    ca, cb = C._codes_all(q), C._codes_all(m)
    assert len(ca) == len(cb)
    for a, b in zip(
        sorted(ca.values(), key=lambda v: v.shape),
        sorted(cb.values(), key=lambda v: v.shape),
    ):
        assert a.shape == b.shape
        # (float32 vs float64 flips single codes on a 65536-code grid; leave room
        # for platforms whose BLAS flips a code or two more)
        assert np.abs(a - b).max() <= 4
        assert np.mean(a != b) <= 0.1
