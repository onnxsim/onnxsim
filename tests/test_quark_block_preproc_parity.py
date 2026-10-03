"""Parity of onnxsim's reproduction of Quark's float pre-processing for the block-format
(``BFP16``, ``MX4`` / ``MX6`` / ``MX9``, ``MXFP4`` / ``6`` / ``8``, ``MXINT8``),
bfloat16 / float16 and ``MATMUL_NBITS`` flows against the real AMD Quark ONNX package.
Skipped unless ``quark.onnx`` is importable.

What Quark does (probed on 0.13): these presets go through the same
``quantize_static`` as the integer ones, so ``apply_pre_process`` runs before the
quantizer -- onnxslim, ONNX Runtime's basic graph optimizer (BatchNorm / Pad / Identity /
MatMul + Add folding, HardSwish expansion), its own BatchNorm folding, the operator
fusions (``FuseLayerNorm``, ``FuseGelu``, ...), then the extended ``QDQ`` format's
conversions (``ReduceMean`` -> ``GlobalAveragePool``, a leftover ``BatchNormalization``
-> ``Conv``, ``Split`` -> ``Slice``; not for ``MATMUL_NBITS``). ``SkipPreprocess`` skips
all of it. Before any of it Quark checks that the model has something to quantize and
returns it as given when it has not. Which tensors the block / half quantizers touch
follows its op-type registry and its Q/DQ-removal rules; the float16 / bfloat16 pairs of
a data-movement op's output read its input's scale and zero point.

Every test compares the quantized graph node by node (op type, domain, name, wiring,
attributes, in order), the initializers (values to float rounding) and the
``opset_import`` list, and -- with Quark's custom-op library where the graph needs it --
what ONNX Runtime computes with its graph optimizations off.
"""

import contextlib
import copy
import io
import os
import warnings

import numpy as np
import onnx
import pytest
from _quark_block_common import (
    PATTERNS,
    block13,
    block17,
    block20,
    calibration_data,
    identity_only,
    ops,
)
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

from onnxsim import quark_compat as qc  # noqa: E402


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark writes scratch files into the current directory."""
    monkeypatch.chdir(tmp_path)


# -- running both -------------------------------------------------------------------------


def _reader(data):
    from onnxruntime.quantization import CalibrationDataReader

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

        def reset_iter(self):
            self.it = iter(data)

        def __iter__(self):
            return iter(data)

    return R()


_MATMUL_NBITS = {
    "MatMulNBitsParams": {
        "GroupSize": 16,
        "Symmetric": True,
        "Bits": 4,
        "AccuracyLevel": 1,
    }
}


def _quark(model, data, tmp_path, preset, extra=None, optimize=None, exclude=()):
    """Quark's preset as ``QConfig.get_default_config`` hands it out (a private copy;
    CLE off, which onnxsim leaves to a ``CLEConfig``). ``optimize`` is its
    ``optimize_model`` field (the ``OptimizeModel`` extra option is ignored by the
    presets, which are legacy ``QuantizationConfig`` objects)."""
    from quark.onnx import ModelQuantizer, QConfig

    cfg = copy.deepcopy(QConfig.get_default_config(preset))
    g = cfg.global_quant_config
    g.include_cle = False
    if optimize is not None:
        g.optimize_model = optimize
    g.extra_options = {**g.extra_options, **(extra or {})}
    g.nodes_to_exclude = list(exclude)
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ModelQuantizer(cfg).quantize_model(src, dst, _reader(data))
    return onnx.load(dst)


def _mine(model, data, preset, extra=None, optimize=None, exclude=()):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.exclude = list(exclude)
    cfg.extra_options.update(extra or {})
    if optimize is not None:
        cfg.extra_options["OptimizeModel"] = optimize
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            onnx.shape_inference.infer_shapes(model),
            calibration_data_reader=_reader(data),
        )


def _both(name_or_model, preset, tmp_path, extra=None, optimize=None, exclude=()):
    model = (
        PATTERNS[name_or_model]() if isinstance(name_or_model, str) else name_or_model
    )
    data = calibration_data(model)
    extra = dict(extra or {})
    if preset == "MATMUL_NBITS":
        extra = {**_MATMUL_NBITS, **extra}
    q = _quark(model, data, tmp_path, preset, extra, optimize, exclude)
    m = _mine(model, data, preset, extra, optimize, exclude)
    return model, data, q, m


# -- comparing -----------------------------------------------------------------------------


def _attr(a):
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, onnx.TensorProto):
        v = numpy_helper.to_array(v).tolist()
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, (list, tuple)):
        v = tuple(v)
    return (a.name, v)


def _nodes(model):
    return [
        (
            n.op_type,
            n.domain,
            n.name,
            tuple(n.input),
            tuple(n.output),
            tuple(sorted(_attr(a) for a in n.attribute)),
        )
        for n in model.graph.node
    ]


def _assert_same(q, m, msg):
    """The same graph: ``opset_import`` (order included), nodes in order, initializers
    (integers exact, floats to rounding), graph inputs and outputs."""
    opsets = lambda x: [(o.domain, o.version) for o in x.opset_import]  # noqa: E731
    assert opsets(m) == opsets(q), f"{msg}: opset_import"
    assert m.ir_version == q.ir_version, f"{msg}: ir_version"
    nq, nm = _nodes(q), _nodes(m)
    if nq != nm:
        first = next((i for i, (a, b) in enumerate(zip(nq, nm)) if a != b), None)
        detail = (
            f"{len(nq)} vs {len(nm)} nodes: Quark {[n[0] for n in nq]} "
            f"vs onnxsim {[n[0] for n in nm]}"
            if first is None
            else f"node {first}: Quark {nq[first]} vs onnxsim {nm[first]}"
        )
        pytest.fail(f"{msg}: nodes differ, {detail}")
    iq = {t.name: t for t in q.graph.initializer}
    im = {t.name: t for t in m.graph.initializer}
    assert set(im) == set(iq), f"{msg}: initializers {set(im) ^ set(iq)}"
    for name, tq in iq.items():
        tm = im[name]
        assert (tm.data_type, tuple(tm.dims)) == (tq.data_type, tuple(tq.dims)), name
        a, b = numpy_helper.to_array(tq), numpy_helper.to_array(tm)
        if a.dtype.kind == "f":
            np.testing.assert_allclose(
                b.astype(np.float64),
                a.astype(np.float64),
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"{msg}: {name}",
            )
        else:
            np.testing.assert_array_equal(b, a, err_msg=f"{msg}: {name}")
    for kind in ("input", "output"):
        ins = lambda x: [  # noqa: E731
            (v.name, v.type.tensor_type.elem_type) for v in getattr(x.graph, kind)
        ]
        assert ins(m) == ins(q), f"{msg}: graph {kind}s"


def _ops_lib():
    path = os.environ.get("QUARK_ONNX_OPS_LIB")
    if path:
        return path
    try:
        from quark.onnx.operators.custom_ops import get_library_path

        return get_library_path()
    except Exception:  # pragma: no cover - environment dependent
        return None


def _ort(model, x):
    import onnxruntime as ort

    so = ort.SessionOptions()
    # no graph optimizations: ORT would fuse DQ -> MatMul / Conv -> Q into integer
    # kernels that saturate on some CPUs, making results depend on the host
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    lib = _ops_lib()
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


def _assert_same_outputs(q, m, data, msg):
    needs_lib = any(n.domain == "com.amd.quark" for n in q.graph.node)
    if needs_lib and _ops_lib() is None:
        return  # the custom ops cannot run: the graphs were compared above
    x = data[0]["x"]
    try:
        ref = _ort(q, x)
    except Exception:  # pragma: no cover - a build of the op library that cannot load
        return
    np.testing.assert_allclose(_ort(m, x), ref, rtol=1e-5, atol=1e-5, err_msg=msg)


# -- every preset family on the patterns ------------------------------------------------------

_BLOCK = [
    "BFP16",
    "MX4",
    "MX6",
    "MX9",
    "MXFP4E2M1",
    "MXFP6E2M3",
    "MXFP6E3M2",
    "MXFP8E4M3",
    "MXFP8E5M2",
    "MXINT8",
]
_HALF = ["BF16", "FP16"]
#: bfloat16 activations over block-format constants, block-format activations over
#: int8 constants
_MIXED = ["BF16_BFP16", "BF16_MXINT8", "MX9_INT8"]
_ALL = _BLOCK + _HALF + _MIXED + ["MATMUL_NBITS"]
#: one preset per family for the rest of the patterns
_FAMILIES = ["BFP16", "MXINT8", "BF16", "FP16", "MATMUL_NBITS"]
_CORE = ["conv_bn", "block20", "clip_bare", "reducemean"]

_CASES = [(p, n) for n in _CORE for p in _ALL] + [
    (p, n) for n in PATTERNS if n not in _CORE for p in _FAMILIES
]


@pytest.mark.parametrize("preset, name", _CASES, ids=[f"{n}-{p}" for p, n in _CASES])
def test_quantized_graph_matches_quark(preset, name, tmp_path):
    """Every preset (all of them on four patterns, one per family on the rest): the
    same graph as Quark's, and the same outputs."""
    _, data, q, m = _both(name, preset, tmp_path)
    _assert_same(q, m, f"{name} {preset}")
    _assert_same_outputs(q, m, data, f"{name} {preset}")


def test_the_pre_processing_is_in_the_graphs(tmp_path):
    """Not vacuous: what Quark's pre-processing does is what the graphs above show."""
    expect = {
        "conv_bn": ("BatchNormalization", "BFP16"),
        "identity": ("Identity", "BF16"),
        "pad_conv": ("Pad", "FP16"),
        "reducemean": ("ReduceMean", "MXINT8"),
        "hardswish": ("HardSwish", "BFP16"),
        "matmul_add": ("Add", "BF16"),
        "block17": ("Pow", "BFP16"),
        "block20": ("Erf", "MXFP8E4M3"),
    }
    for name, (gone, preset) in expect.items():
        _, _, q, _ = _both(name, preset, tmp_path)
        assert gone not in ops(q), f"{name} {preset}: {gone} survived Quark"
    _, _, q, _ = _both("block20", "BFP16", tmp_path)
    assert {"LayerNormalization", "Gelu"} <= set(ops(q))
    _, _, q, _ = _both("reducemean", "BFP16", tmp_path)
    assert "GlobalAveragePool" in ops(q)
    # (the plain ``QDQ`` format of MATMUL_NBITS leaves a ReduceMean alone)
    _, _, q, _ = _both("reducemean", "MATMUL_NBITS", tmp_path)
    assert "ReduceMean" in ops(q)


# -- options -------------------------------------------------------------------------------------

_OPTION_CASES = [
    ({"SkipPreprocess": True}, None, "block17"),
    ({"SkipPreprocess": True}, None, "conv_bn"),
    ({"SkipPreprocess": True}, None, "matmul_add"),
    ({"SkipPreprocess": True}, None, "identity"),
    ({"ConvertOpsetVersion": 20}, None, "block17"),
    ({"ConvertOpsetVersion": 17}, None, "block13"),
    ({"SimplifyModel": False}, None, "identity"),
    ({"SimplifyModel": False}, None, "pad_conv"),
    ({"FoldBatchNorm": False}, None, "gemm_bn"),
    ({"FoldBatchNorm": False}, None, "bn_concat"),
    ({"FuseLayerNorm": False}, None, "block20"),
    ({"FuseGelu": False}, None, "block20"),
    (
        {
            "ConvertBNToConv": False,
            "ConvertReduceMeanToGlobalAvgPool": False,
            "ConvertSplitToSlice": False,
        },
        None,
        "reducemean",
    ),
    ({"ConvertSplitToSlice": False}, None, "split_concat"),
    ({"QuantizeAllOpTypes": True}, None, "bn_concat"),
    ({"QuantizeAllOpTypes": True}, None, "convt_bn"),
    ({"RemoveQDQConvRelu": False, "RemoveQDQConvClip": False}, None, "clip_relu6"),
    ({"RemoveQDQConvRelu": False}, None, "conv_bn"),
    ({}, False, "conv_bn"),
    ({}, False, "matmul_add"),
    ({}, False, "hardswish"),
    ({"ForceQuantizeNoInputCheck": True}, None, "movement_chain"),
    ({"ForceQuantizeNoInputCheck": False}, None, "movement_chain"),
]
_OPTION_PRESETS = ["BFP16", "BF16", "MATMUL_NBITS"]


@pytest.mark.parametrize("preset", _OPTION_PRESETS)
@pytest.mark.parametrize(
    "extra, optimize, name",
    _OPTION_CASES,
    ids=[f"{n}-{e or 'opt'}-{o}" for e, o, n in _OPTION_CASES],
)
def test_options_match_quark(extra, optimize, name, preset, tmp_path):
    _, data, q, m = _both(name, preset, tmp_path, extra, optimize)
    _assert_same(q, m, f"{name} {preset} {extra} optimize={optimize}")
    _assert_same_outputs(q, m, data, f"{name} {preset} {extra}")


@pytest.mark.parametrize("preset", ["BFP16", "MXFP4E2M1", "FP16", "MATMUL_NBITS"])
@pytest.mark.parametrize("target", [None, 20])
def test_a_model_with_nothing_to_quantize_comes_back_as_given(preset, target, tmp_path):
    """Quark checks for quantizable ops before pre-processing: this model is returned
    unchanged -- Identity nodes kept, opset not converted, no ``com.amd.quark``
    domain."""
    extra = {} if target is None else {"ConvertOpsetVersion": target}
    model, _, q, m = _both(identity_only(), preset, tmp_path, extra)
    assert ops(q) == ["Identity", "Identity"]
    assert [(o.domain, o.version) for o in q.opset_import] == [("", 17)]
    _assert_same(q, m, f"identity_only {preset} {extra}")


@pytest.mark.parametrize("target", [None, 20])
def test_bfloat16_quantizes_every_op_type_so_identity_is_quantizable(target, tmp_path):
    """(``QuantizeAllOpTypes`` puts the model's own op types on the list.)"""
    extra = {} if target is None else {"ConvertOpsetVersion": target}
    _, data, q, m = _both(identity_only(), "BF16", tmp_path, extra)
    assert "ExtendedQuantizeLinear" in ops(q)
    _assert_same(q, m, f"identity_only BF16 {extra}")
    _assert_same_outputs(q, m, data, "identity_only BF16")


def test_the_shared_scale_of_a_data_movement_output_matches_quark(tmp_path):
    """float16 / bfloat16: the output of a Transpose / Reshape / Unsqueeze / Squeeze
    reads its input's scale and zero point initializers."""
    _, _, q, m = _both("movement_chain", "BF16", tmp_path)
    reads = {n.output[0]: n.input[1:] for n in q.graph.node if "Quantize" in n.op_type}
    shared = [i for i in q.graph.initializer if i.name.endswith("_scale")]
    assert len(shared) < sum(
        1 for n in q.graph.node if n.op_type == "ExtendedQuantizeLinear"
    )
    assert reads
    _assert_same(q, m, "movement_chain BF16")


@pytest.mark.parametrize(
    "build", [block13, block17, block20], ids=["opset13", "opset17", "opset20"]
)
@pytest.mark.parametrize("preset", ["BFP16", "BF16", "MATMUL_NBITS"])
def test_decomposed_norm_and_gelu_follow_the_opset(build, preset, tmp_path):
    """LayerNorm fuses from opset 17 and Gelu from opset 20, before the quantizer
    sees them (so a fused LayerNormalization is quantized as one)."""
    _, data, q, m = _both(build(), preset, tmp_path)
    _assert_same(q, m, f"{build.__name__} {preset}")
    got = set(ops(q))
    assert ("LayerNormalization" in got) == (build is not block13)
    assert ("Gelu" in got) == (build is block20)
    _assert_same_outputs(q, m, data, build.__name__)


@pytest.mark.parametrize("name", ["bn_concat", "convt_bn", "gemm_bn"])
def test_int_flow_leaves_a_quantized_batch_norm_to_the_quantizer(name, tmp_path):
    """The same rule in the integer flows: with ``QuantizeAllOpTypes`` the
    BatchNormalization is on Quark's op list, so its own folding passes skip it."""
    from _quark_fusion_common import _graph_diff

    _, data, q, m = _both(name, "XINT8", tmp_path, {"QuantizeAllOpTypes": True})
    # (the integer flows' names are not Quark's: compared by content, as in the
    # fusions' parity tests)
    assert not _graph_diff(q, m), f"{name}: {_graph_diff(q, m)}"
    assert "BatchNormalization" not in ops(q) or name == "gemm_bn"
    _assert_same_outputs(q, m, data, name)


def _named(model):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    for i, n in enumerate(out.graph.node):
        n.name = f"n{i}_{n.op_type}"
    return out


@pytest.mark.parametrize("preset", ["BFP16", "BF16", "FP16"])
@pytest.mark.parametrize(
    "name",
    ["conv_bn", "matmul_add", "clip_relu6", "hardswish", "reducemean", "pad_conv"],
)
@pytest.mark.parametrize("which", ["first", "last", "first_two", "middle"])
def test_excluded_nodes_match_quark(name, which, preset, tmp_path):
    """An excluded node quantizes none of its tensors, but the structural rules of
    Quark's post-processing (the Q/DQ pair before a ReLU, the block axis of a MatMul)
    still read it."""
    model = _named(PATTERNS[name]())
    names = [n.name for n in model.graph.node]
    exclude = {
        "first": names[:1],
        "last": names[-1:],
        "first_two": names[:2],
        "middle": [names[len(names) // 2]],
    }[which]
    _, data, q, m = _both(model, preset, tmp_path, exclude=exclude)
    _assert_same(q, m, f"{name} {preset} exclude {exclude}")
