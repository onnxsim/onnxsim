"""Parity of onnxsim's Quark-compat layer against the real AMD Quark ONNX
package. Skipped unless ``quark.onnx`` is importable (the
``quark-parity`` workflow installs it).

Quark is the ground truth: each test runs Quark and onnxsim on the same graph
and compares what they emit (op placement, attributes, axes) and, where the
ONNX Runtime custom-op library Quark JIT-builds is available, what the emitted
graphs compute. Differences that are known and deliberate are listed in the
``KNOWN_*`` constants so a *new* difference fails the test.

Run with ``-s`` (or see ``QUARK_PARITY_REPORT``) for the quality report.
"""

import contextlib
import io
import json
import os
import warnings

import numpy as np
import onnx
import pytest
from onnx import parser

warnings.filterwarnings("ignore")

with (
    contextlib.redirect_stdout(io.StringIO()),
    contextlib.redirect_stderr(io.StringIO()),
):
    try:
        import quark.onnx as quark_onnx
        from quark.onnx.quantization.config import DefaultConfigMapping
    except Exception as e:  # pragma: no cover - environment dependent
        quark_onnx = None
        _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    quark_onnx is None, reason="AMD Quark (amd-quark) is not installed"
)

from onnxsim import quark_auto_mixprecision as amp  # noqa: E402
from onnxsim import quark_compat as qc  # noqa: E402
from onnxsim import quark_tools  # noqa: E402
from onnxsim.quark_fakequant_graph import apply_fake_quant_format  # noqa: E402

# Quark presets onnxsim does not implement (NPU CNN/transformer quantizers,
# MatMulNBits, dynamic/VINT8, mixed block formats, ...).
KNOWN_MISSING: set = set()
# onnxsim-only presets (Quark has no ADAROUND/ADAQUANT variant for U8U8_AAWA).
KNOWN_EXTRA = {"U8U8_AAWA_ADAQUANT", "U8U8_AAWA_ADAROUND"}
# Ops whose single-op graph Quark rewrites before quantizing (ReduceMean ->
# GlobalAveragePool); onnxsim quantizes the graph as given.
KNOWN_GRAPH_DIFF = {"ReduceMean"}

_BLOCK = ["BFP16", "MX4", "MX9", "MXINT8", "MXFP8E4M3", "MXFP4E2M1"]
_HALF = ["FP16", "BF16"]


# -- helpers ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Quark's fine-tuning presets and ``quantized_info.csv`` write scratch files
    into the current directory; keep them out of the checkout."""
    monkeypatch.chdir(tmp_path)


def _reader(shape, n=4, seed=3, name="x"):
    from onnxruntime.quantization import CalibrationDataReader

    rng = np.random.default_rng(seed)
    data = [{name: rng.standard_normal(shape).astype(np.float32)} for _ in range(n)]

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

    return R


def quark_quantize(model, preset, shape, tmp_path, tag="m", cle=False):
    """Run Quark. Every Quark preset enables cross-layer equalization
    (``include_cle``) implicitly, which rewrites weights; onnxsim only runs CLE
    when the config lists a ``CLEConfig``. It is therefore off here unless a
    test asks for it."""
    from quark.onnx import ModelQuantizer, QConfig

    src, dst = str(tmp_path / f"{tag}.onnx"), str(tmp_path / f"{tag}_q.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        cfg = QConfig.get_default_config(preset)
        cfg.global_quant_config.include_cle = cle
        ModelQuantizer(cfg).quantize_model(src, dst, _reader(shape)())
    return onnx.load(dst)


def mine_quantize(model, preset, shape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(qc.QConfig.get_default_config(preset)).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )


def _attrs(n):
    out = {}
    for a in n.attribute:
        v = onnx.helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else v
    return out


def _cop_map(model):
    """tensor -> (op_type, attrs) for every com.amd.quark node."""
    out = {}
    for n in model.graph.node:
        if n.domain == "com.amd.quark":
            src = n.input[0].removesuffix("_QuantizeLinear_Input")
            out[src] = (n.op_type, tuple(sorted(_attrs(n).items())))
    return out


def _ext_map(model):
    """tensor -> op type, for every (Extended)QuantizeLinear."""
    return {
        n.input[0].removesuffix("_QuantizeLinear_Input"): n.op_type
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "ExtendedQuantizeLinear")
    }


def _op_counts(model):
    counts = {}
    for n in model.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return counts


def _ops_lib():
    path = os.environ.get("QUARK_ONNX_OPS_LIB")
    if path:
        return path
    try:
        from quark.onnx.operators.custom_ops import get_library_path

        return get_library_path()
    except Exception:  # pragma: no cover - environment dependent
        return None


def _run(model, x):
    import onnxruntime as ort

    so = ort.SessionOptions()
    # No graph optimizations: ORT would otherwise fuse DQ -> MatMul/Conv -> Q into
    # integer kernels that saturate on x86 CPUs without VNNI (CI runners), making
    # results depend on the host rather than on the quantization.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    lib = _ops_lib()
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})[0]


# -- models ---------------------------------------------------------------------


def _w(rng, *shape):
    return (rng.standard_normal(shape) * 0.5).astype(np.float32)


def _mlp():
    # Gemm, not MatMul+Add: Quark's preprocessing fuses the latter into a Gemm
    # (with renamed tensors) before quantizing, which onnxsim does not do.
    rng = np.random.default_rng(0)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            y = Gemm(h1, w2, b2)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 16, 32), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 32), "b1"),
            onnx.numpy_helper.from_array(_w(rng, 32, 8), "w2"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b2"),
        ]
    )
    return m, (3, 16)


def _conv():
    rng = np.random.default_rng(1)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[1,3,8,8] x) => (float[1,8,4,4] y) {
            c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
            r0 = Relu(c0)
            y = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r0)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 8, 3, 3, 3), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b1"),
        ]
    )
    return m, (1, 3, 8, 8)


def _gemm_transb():
    rng = np.random.default_rng(2)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            y = Gemm<transB=1>(x, w, b)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 8, 16), "w"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b"),
        ]
    )
    return m, (3, 16)


MODELS = {"mlp": _mlp, "conv": _conv, "gemm_transb": _gemm_transb}


# -- presets ----------------------------------------------------------------------


def test_preset_coverage_matches_documented_gaps():
    quark_names, mine = set(DefaultConfigMapping), set(qc._PRESETS)
    assert quark_names - mine == KNOWN_MISSING, "Quark gained/lost presets"
    assert mine - quark_names == KNOWN_EXTRA


_DTYPE = {
    "QInt8": "int8",
    "QUInt8": "uint8",
    "QInt16": "int16",
    "QUInt16": "uint16",
    "QFloat16": "float16",
    "QBFloat16": "bfloat16",
}


def _shared_presets():
    return sorted(set(DefaultConfigMapping) & set(qc._PRESETS)) if quark_onnx else []


@pytest.mark.parametrize("preset", _shared_presets())
def test_preset_dtypes_match(preset):
    from quark.onnx import QConfig

    with contextlib.redirect_stdout(io.StringIO()):
        q = QConfig.get_default_config(preset).global_quant_config
    mine = qc._PRESETS[preset]().global_config
    for quark_t, spec in (
        (q.activation_type, mine.activation),
        (q.weight_type, mine.weight),
    ):
        if quark_t.name in _DTYPE:
            assert spec.dtype == _DTYPE[quark_t.name]
        else:  # QBFP / QMX: a block format on both sides
            assert spec.dtype.startswith(("bfp", "mx"))


# -- block formats and half precision: graph parity --------------------------------


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("preset", _BLOCK)
def test_block_format_graph_matches_quark(preset, model_name, tmp_path):
    model, shape = MODELS[model_name]()
    q = quark_quantize(model, preset, shape, tmp_path)
    m = mine_quantize(model, preset, shape)
    assert _cop_map(m) == _cop_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("preset", _HALF)
def test_half_precision_graph_matches_quark(preset, model_name, tmp_path):
    model, shape = MODELS[model_name]()
    q = quark_quantize(model, preset, shape, tmp_path)
    m = mine_quantize(model, preset, shape)
    assert _ext_map(m) == _ext_map(q)
    assert _op_counts(m) == _op_counts(q)


def _single_op_cases():
    X = [1, 4, 6, 6]
    rng = np.random.default_rng(0)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    return {
        **{
            op: dict(shapes=[X])
            for op in (
                "Relu",
                "Sigmoid",
                "Tanh",
                "LeakyRelu",
                "Erf",
                "Abs",
                "Neg",
                "Exp",
                "Sqrt",
                "Flatten",
                "GlobalAveragePool",
            )
        },
        "Softmax": dict(shapes=[X], attrs=dict(axis=1)),
        "Add": dict(shapes=[X, X]),
        "Mul": dict(shapes=[X, X]),
        "AddConst": dict(op="Add", shapes=[X], consts={"c": f(1, 4, 1, 1)}),
        "Concat": dict(shapes=[X, X], attrs=dict(axis=1)),
        "MaxPool": dict(shapes=[X], attrs=dict(kernel_shape=[2, 2])),
        "AveragePool": dict(shapes=[X], attrs=dict(kernel_shape=[2, 2])),
        "ReduceMean": dict(shapes=[X], attrs=dict(axes=[2, 3])),
        "Reshape": dict(shapes=[X], consts={"s": np.array([1, 144], np.int64)}),
        "Transpose": dict(shapes=[X], attrs=dict(perm=[0, 2, 3, 1])),
        "Resize": dict(
            shapes=[X],
            consts={
                "roi": np.array([], np.float32),
                "sc": np.array([1, 1, 2, 2], np.float32),
            },
        ),
        "BatchNormalization": dict(
            shapes=[X],
            consts={"s": f(4), "b": f(4), "m": f(4), "v": np.abs(f(4)) + 1},
        ),
        "InstanceNormalization": dict(shapes=[X], consts={"s": f(4), "b": f(4)}),
        "ConvTranspose": dict(
            shapes=[X], consts={"w": f(4, 4, 2, 2)}, attrs=dict(strides=[2, 2])
        ),
        "Gemm": dict(shapes=[[3, 8]], consts={"w": f(8, 5)}),
    }


def _single_op_model(case):
    from onnx import TensorProto as T
    from onnx import helper as H

    op = case.get("op")
    shapes = case["shapes"]
    names = ["x"] + [f"in{i}" for i in range(1, len(shapes))]
    inputs = [H.make_tensor_value_info(n, T.FLOAT, s) for n, s in zip(names, shapes)]
    inits = []
    for nm, arr in (case.get("consts") or {}).items():
        inits.append(onnx.numpy_helper.from_array(arr, nm))
        names.append(nm)
    node = H.make_node(op, names, ["y"], **(case.get("attrs") or {}))
    g = H.make_graph(
        [node],
        "g",
        inputs,
        [H.make_tensor_value_info("y", T.FLOAT, None)],
        initializer=inits,
    )
    return H.make_model(g, opset_imports=[H.make_opsetid("", 17)], ir_version=9)


def _single_op_params():
    return sorted(_single_op_cases()) if quark_onnx else []


@pytest.mark.parametrize(
    "dtype, preset", [("bfp16", "BFP16"), ("float16", "FP16"), ("bfloat16", "BF16")]
)
def test_single_op_placement_matches_quark(dtype, preset, tmp_path):
    """Quark's op coverage, op by op (``KNOWN_GRAPH_DIFF`` aside)."""
    diffs = []
    for name, case in _single_op_cases().items():
        case = dict(case, op=case.get("op", name))
        model = _single_op_model(case)
        try:
            q = quark_quantize(model, preset, case["shapes"][0], tmp_path, name)
        except Exception:  # Quark itself rejects this graph
            continue
        m = apply_fake_quant_format(model, dtype)
        got, want = _op_counts(m), _op_counts(q)
        if got != want and name not in KNOWN_GRAPH_DIFF:
            diffs.append((name, got, want))
    assert not diffs, json.dumps(diffs, indent=1)


# -- numerics --------------------------------------------------------------------


@pytest.mark.skipif(_ops_lib() is None, reason="Quark's custom-op library is not built")
@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_outputs_match_quark(preset, model_name, tmp_path):
    model, shape = MODELS[model_name]()
    q = quark_quantize(model, preset, shape, tmp_path)
    m = mine_quantize(model, preset, shape)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32) * 2
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-5, atol=1e-5)


# -- tools ------------------------------------------------------------------------


def test_remove_qdq_matches_quarks_convert_quant_to_float(tmp_path):
    from quark.onnx.tools.convert_quant_to_float import (
        convert_quant_to_float as quark_to_float,
    )

    model, shape = _mlp()
    q = quark_quantize(model, "A8W8", shape, tmp_path)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        theirs = quark_to_float(q)
    # Quark converts the int32 bias initializers to float64 (which ORT rejects
    # next to float32 Gemm operands); cast them back before comparing.
    for t in theirs.graph.initializer:
        if t.data_type == onnx.TensorProto.DOUBLE:
            t.CopyFrom(
                onnx.numpy_helper.from_array(
                    onnx.numpy_helper.to_array(t).astype(np.float32), t.name
                )
            )
    ours = quark_tools.convert_quant_to_float(q)
    x = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(ours, x), _run(theirs, x), rtol=1e-5, atol=1e-5)
    assert not any(
        n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in ours.graph.node
    )


# -- auto mixed precision metrics -------------------------------------------------


def test_amp_metrics_match_quark():
    from quark.onnx.algorithm.mprecision import metric_funcs as qm

    rng = np.random.default_rng(0)
    f = [[rng.standard_normal((4, 6)).astype(np.float32)] for _ in range(5)]
    q = [[a[0] + 0.1 * rng.standard_normal((4, 6)).astype(np.float32)] for a in f]
    for name, theirs in (
        ("l2", qm.l2_metric),
        ("cosine", qm.cosine_metric),
        ("sqnr", qm.sqnr_metric),
        ("psnr", qm.psnr_metric),
        ("kl", qm.kl_divergence_metric),
    ):
        np.testing.assert_allclose(
            amp.resolve_metric(name)(f, q), theirs(f, q), rtol=1e-5, err_msg=name
        )


# -- cross-layer equalization ------------------------------------------------------


def _mlp3():
    rng = np.random.default_rng(6)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,8] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            h2 = Gemm(h1, w2, b2)
            h3 = LeakyRelu<alpha=0.1>(h2)
            y = Gemm(h3, w3, b3)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 16, 32) * 3, "w1"),
            onnx.numpy_helper.from_array(_w(rng, 32), "b1"),
            onnx.numpy_helper.from_array(_w(rng, 32, 24), "w2"),
            onnx.numpy_helper.from_array(_w(rng, 24), "b2"),
            onnx.numpy_helper.from_array(_w(rng, 24, 8) * 0.2, "w3"),
            onnx.numpy_helper.from_array(_w(rng, 8), "b3"),
        ]
    )
    return m, (3, 16)


@pytest.mark.parametrize("build", [_mlp, _mlp3])
def test_cle_matches_quarks_equalization(build, tmp_path):
    """FP16 leaves the weights as initializers, so Quark's implicit CLE shows
    up as changed ``w*`` / ``b*``."""
    from onnxsim.quark_cle import equalize_linear_layers

    model, shape = build()
    q = quark_quantize(model, "FP16", shape, tmp_path, "cle", cle=True)
    ours = equalize_linear_layers(model)
    theirs = {i.name: onnx.numpy_helper.to_array(i) for i in q.graph.initializer}
    mine = {i.name: onnx.numpy_helper.to_array(i) for i in ours.graph.initializer}
    orig = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    assert not np.allclose(theirs["w1"], orig["w1"]), "Quark did not equalize"
    for name in orig:
        np.testing.assert_allclose(
            mine[name], theirs[name], rtol=1e-3, atol=1e-5, err_msg=name
        )
    x = np.random.default_rng(2).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(ours, x), _run(model, x), rtol=1e-4, atol=1e-4)


# -- quarot ------------------------------------------------------------------------


def _torch_style_llm():
    rng = np.random.default_rng(4)
    d = 16
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,{d}] x) => (float[3,{d}] y) {{
            h = Gemm<alpha=1.0, beta=1.0, transB=1>(x, w_in, b_in)
            y = Gemm<alpha=1.0, beta=1.0, transB=1>(h, w_out, b_out)
        }}
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, d, d), "w_in"),
            onnx.numpy_helper.from_array(_w(rng, d), "b_in"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "w_out"),
            onnx.numpy_helper.from_array(_w(rng, d), "b_out"),
        ]
    )
    return m, (3, d)


def test_quarot_preserves_function_like_quark():
    from onnxsim.quark_quarot import rotate_model

    model, shape = _torch_style_llm()
    for n, name in zip(model.graph.node, ("g_in", "g_out")):
        n.name = name
    cfg = {"R1_pairs": [{"prev_nodes": ["g_in"], "next_nodes": ["g_out"]}]}
    out = rotate_model(model, cfg, r_matrix_dim=16)
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    # the rotation changes the hidden basis only; the first Gemm's output is
    # rotated, the final output is unchanged only when the last layer is
    # un-rotated by its reader -- compare the whole function.
    np.testing.assert_allclose(_run(out, x), _run(model, x), rtol=1e-4, atol=1e-4)


# -- quality report (non-asserting) ------------------------------------------------


def test_quality_report_against_quark(tmp_path):
    """Records, per preset, each side's output error versus float. No
    assertion on who is better -- onnxsim's calibration differs from Quark's
    (MinMSE power-of-two scales) -- only that both produce a usable model."""
    rows = []
    for mname, build in MODELS.items():
        model, shape = build()
        x = np.random.default_rng(9).standard_normal(shape).astype(np.float32)
        ref = _run(model, x)
        for preset in (
            "A8W8",
            "XINT8",
            "U8S8_AAWS",
            "A16W8",
            "FP16",
            "BF16",
            "BFP16",
            "MX9",
        ):
            try:
                q = _run(
                    quark_quantize(model, preset, shape, tmp_path, f"{mname}_{preset}"),
                    x,
                )
                m = _run(mine_quantize(model, preset, shape), x)
            except Exception as e:  # pragma: no cover - op library missing etc.
                rows.append(dict(model=mname, preset=preset, error=str(e)[:80]))
                continue

            def rel(a):
                return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-12))

            rows.append(
                dict(
                    model=mname,
                    preset=preset,
                    quark_rel_err=rel(q),
                    onnxsim_rel_err=rel(m),
                )
            )
            assert np.isfinite(rel(m)) and rel(m) < 1.0
    path = os.environ.get("QUARK_PARITY_REPORT")
    text = "\n".join(
        f"{r['model']:12s} {r['preset']:10s} "
        + (
            r["error"]
            if "error" in r
            else f"quark {r['quark_rel_err']:.4f}  onnxsim {r['onnxsim_rel_err']:.4f}"
        )
        for r in rows
    )
    print("\n" + text)
    if path:
        with open(path, "w") as fh:
            fh.write(
                "| model | preset | Quark rel. err | onnxsim rel. err |\n|---|---|---|---|\n"
            )
            for r in rows:
                if "error" not in r:
                    fh.write(
                        f"| {r['model']} | {r['preset']} | {r['quark_rel_err']:.4f} | {r['onnxsim_rel_err']:.4f} |\n"
                    )


# -- integer presets: quantization parameters ------------------------------------


def _int_params(model):
    """Sorted ``(scale, zero_point, dtype)`` of every activation QuantizeLinear
    and the scales of the weight DequantizeLinear nodes (per-tensor)."""
    inits = {i.name: i for i in model.graph.initializer}

    def arr(name):
        return onnx.numpy_helper.to_array(inits[name])

    acts, weights = [], []
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = arr(n.input[2])
            acts.append((float(arr(n.input[1])), int(zp), str(zp.dtype)))
        elif (
            n.op_type == "DequantizeLinear"
            and n.input[0] in inits
            and arr(n.input[0]).dtype == np.int8
            and arr(n.input[0]).ndim >= 2
        ):
            weights.append(float(np.max(arr(n.input[1]))))
    return sorted(acts), sorted(weights)


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize(
    "preset",
    ["A8W8", "S8S8_AAWS", "U8S8_AAWS", "A16W8", "S16S8_ASWS", "U16S8_AAWS", "XINT8"],
)
def test_integer_preset_quantization_parameters_match_quark(
    preset, model_name, tmp_path
):
    """Same scale / zero point / dtype on every activation Q node and the same
    per-tensor weight scales. (``U8S8_AAWS``-style presets calibrate with
    percentiles; ours uses the same percentile, so the parameters agree up to
    histogram binning.)"""
    model, shape = MODELS[model_name]()
    q_acts, q_w = _int_params(quark_quantize(model, preset, shape, tmp_path))
    m_acts, m_w = _int_params(mine_quantize(model, preset, shape))
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    # histogram binning moves a percentile's zero point by a few codes of 16
    np.testing.assert_allclose(
        [a[1] for a in m_acts],
        [a[1] for a in q_acts],
        atol=40 if "16" in preset else 1,
    )
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose(m_w, q_w, rtol=2e-3)


# -- per-layer / per-type overrides ------------------------------------------------


def _named_mlp():
    model, shape = _mlp()
    for n, name in zip(model.graph.node, ("g1", "relu", "g2")):
        n.name = name
    return model, shape


def _layer_config(api, case):
    """``QConfig`` for the override ``case``, built from ``api`` (either
    ``quark.onnx`` or ``onnxsim.quark_compat`` -- the class names match)."""

    def glob():
        return api.QLayerConfig(activation=api.Int8Spec(), weight=api.Int8Spec())

    def int16():
        return api.QLayerConfig(
            input_tensors=api.Int16Spec(),
            weight=api.Int8Spec(),
            output_tensors=api.Int16Spec(),
        )

    kwargs = {
        "specific_name": dict(specific_layer_config={int16(): ["g2"]}),
        "specific_regex": dict(specific_layer_config={int16(): ["^g.*"]}),
        "type_gemm": dict(layer_type_config={int16(): ["Gemm"]}),
        "exclude_node": dict(exclude=["g1"]),
        "specific_over_type": dict(
            layer_type_config={int16(): ["Gemm"]},
            specific_layer_config={
                api.QLayerConfig(
                    input_tensors=api.Int8Spec(),
                    weight=api.Int8Spec(),
                    output_tensors=api.Int8Spec(),
                ): ["g1"]
            },
        ),
    }[case]
    return api.QConfig(global_config=glob(), **kwargs)


_LAYER_CASES = [
    "specific_name",
    "specific_regex",
    "type_gemm",
    "exclude_node",
    "specific_over_type",
]


@pytest.mark.parametrize("case", _LAYER_CASES)
def test_layer_overrides_match_quark(case, tmp_path):
    from quark.onnx import ModelQuantizer

    model, shape = _named_mlp()
    src, dst = str(tmp_path / "m.onnx"), str(tmp_path / "m_q.onnx")
    onnx.save(model, src)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ModelQuantizer(_layer_config(quark_onnx, case)).quantize_model(
            src, dst, _reader(shape)()
        )
    q_acts, q_w = _int_params(onnx.load(dst))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = qc.ModelQuantizer(_layer_config(qc, case)).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )
    m_acts, m_w = _int_params(mine)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose(m_w, q_w, rtol=2e-3)


# -- dynamic quantization ------------------------------------------------------------


@pytest.mark.parametrize("build", [_mlp, _conv, _gemm_transb])
def test_dynamic_quantization_matches_quark(build, tmp_path):
    model, shape = build()
    q = quark_quantize(model, "UINT8_DYNAMIC_QUANT", shape, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = qc.ModelQuantizer(
            qc.QConfig.get_default_config("UINT8_DYNAMIC_QUANT")
        ).quantize_model(model)
    assert [n.op_type for n in mine.graph.node] == [n.op_type for n in q.graph.node]
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(mine, x), _run(q, x), rtol=1e-4, atol=1e-5)


# =============================================================================
# quark_tools_extra: the rest of quark.onnx.tools and the model_utils helpers.
# Each test runs Quark's own tool and onnxsim's on the same graph. Deliberate
# differences are documented next to the test that sees them.
# =============================================================================


def _quiet(fn, *args, **kwargs):
    # (logging is disabled too: some Quark helpers call logger.info with
    # extra positional args, which blows up under pytest's log capture)
    import logging

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        logging.disable(logging.CRITICAL)
        try:
            return fn(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)


def _copy_model(model):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _tx_ort(model, feed):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 4
    lib = _ops_lib()
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feed)


def _tx_inits(model):
    from onnx import numpy_helper

    return {
        t.name: (t.data_type, numpy_helper.to_array(t)) for t in model.graph.initializer
    }


def _tx_nodes(model):
    return [(n.op_type, n.domain) for n in model.graph.node]


def _tx_sorted_nodes(model):
    return sorted(_tx_nodes(model))


def _tx_same_inits(a_model, b_model):
    a, b = _tx_inits(a_model), _tx_inits(b_model)
    assert set(a) == set(b)
    for k in a:
        assert a[k][0] == b[k][0], k
        np.testing.assert_array_equal(a[k][1], b[k][1], err_msg=k)


def _tx_conv_qdq(bias_dtype="int8", bias_scale=0.0007, act_zp_type="int8"):
    """x -> Q/DQ -> Conv(w DQ, bias DQ) -> Q/DQ -> y. Scalars come from the
    parser (``float_data``); the integer weights from numpy (``raw_data``),
    the form Quark's A8W8 converter reads. The bias / weight DQs are listed
    before the activation Q/DQ because Quark's converter assumes that graph
    order (it pairs Conv inputs with producers by node position)."""
    from onnx import numpy_helper

    rng = np.random.default_rng(0)
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y)
        <float xs = {{0.02}}, {act_zp_type} xz = {{0}}, float ws = {{0.01}},
         int8 wz = {{0}}, float bs = {{{bias_scale}}}, {bias_dtype} bz = {{0}},
         float ys = {{0.05}}, {act_zp_type} yz = {{0}}>
        {{
            bd = DequantizeLinear(bq, bs, bz)
            wd = DequantizeLinear(wq, ws, wz)
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            c = Conv(xd, wd, bd)
            cq = QuantizeLinear(c, ys, yz)
            y = DequantizeLinear(cq, ys, yz)
        }}
        """
    )
    bq = rng.integers(-100, 100, (4,)).astype(
        np.int8 if bias_dtype == "int8" else np.int32
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                rng.integers(-100, 100, (4, 3, 3, 3)).astype(np.int8), "wq"
            ),
            numpy_helper.from_array(bq, "bq"),
        ]
    )
    return model


def _tx_x(shape=(1, 3, 8, 8), seed=1):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_tools_a8w8_npu_to_cpu_matches_quark():
    from quark.onnx.tools.convert_a8w8_npu_to_a8w8_cpu import (
        convert_a8w8_npu_to_a8w8_cpu as q_fn,
    )

    model = _tx_conv_qdq()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_a8w8_npu_to_a8w8_cpu(model)
    _tx_same_inits(ours, theirs)
    assert _tx_inits(ours)["bq"][1].dtype == np.int32
    x = _tx_x()
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def test_tools_bias_int32_to_int16_matches_quark():
    from quark.onnx.tools.convert_bias_int32_to_int16 import (
        convert_bias_int32_to_int16 as q_fn,
    )

    model = _tx_conv_qdq(bias_dtype="int32")
    theirs, t_flag = _quiet(q_fn, _copy_model(model))
    ours, o_flag = quark_tools.convert_bias_int32_to_int16(model)
    assert o_flag is True and bool(t_flag) is True
    _tx_same_inits(ours, theirs)
    assert _tx_inits(ours)["bq"][1].dtype == np.int16
    assert _tx_inits(ours)["bz"][1].dtype == np.int16
    # nothing to convert -> flag False on both
    plain = _tx_conv_qdq()
    assert not _quiet(q_fn, _copy_model(plain))[1]
    assert quark_tools.convert_bias_int32_to_int16(plain)[1] is False


def test_tools_customqdq_to_qdq_matches_quark():
    from quark.onnx.tools.convert_customqdq_to_qdq import (
        convert_customqdq_to_qdq as q_fn,
    )

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[4] x) => (float[4] y, float[4] z)
        <float s = {0.1}, uint16 z16 = {32768}, int8 z8 = {0}, bfloat16 zb = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, z16)
            y = com.amd.quark.ExtendedDequantizeLinear(a, s, z16)
            b = com.amd.quark.ExtendedQuantizeLinear(x, s, z8)
            c = com.amd.quark.ExtendedDequantizeLinear(b, s, z8)
            d = com.amd.quark.ExtendedQuantizeLinear(c, s, zb)
            z = com.amd.quark.ExtendedDequantizeLinear(d, s, zb)
        }
        """
    )
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_customqdq_to_qdq(model)
    assert _tx_nodes(ours) == _tx_nodes(theirs)
    assert [n.op_type for n in ours.graph.node] == [
        "QuantizeLinear",
        "DequantizeLinear",
        "QuantizeLinear",
        "DequantizeLinear",
        "ExtendedQuantizeLinear",
        "ExtendedDequantizeLinear",
    ]
    # deliberate: we also register the com.microsoft opset so the model loads
    assert "com.microsoft" in {o.domain for o in ours.opset_import}


@pytest.mark.parametrize("reverse", [False, True])
def test_tools_convert_custom_ops_matches_quark(reverse):
    from quark.onnx.tools import convert_custom_ops as q_mod

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[4] x) => (float[4] y)
        <float s = {0.1}, int8 z = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, z)
            b = com.amd.quark.ExtendedDequantizeLinear(a, s, z)
            y = Relu(b)
        }
        """
    )
    if reverse:
        model = _quiet(
            q_mod.convert_custom_ops,
            _copy_model(model),
            q_mod.OLD_DOMAIN,
            q_mod.NAME_MAPPING,
        )
        domain = q_mod.NEW_DOMAIN
        mapping = {v: k for k, v in q_mod.NAME_MAPPING.items()}
        ours_map = {v: k for k, v in quark_tools.CUSTOM_OP_NAME_MAPPING.items()}
    else:
        domain, mapping = q_mod.OLD_DOMAIN, q_mod.NAME_MAPPING
        ours_map = quark_tools.CUSTOM_OP_NAME_MAPPING
    assert ours_map == mapping
    theirs = _quiet(q_mod.convert_custom_ops, _copy_model(model), domain, mapping)
    ours = quark_tools.convert_custom_ops(model, domain, ours_map)
    assert _tx_nodes(ours) == _tx_nodes(theirs)
    assert {(o.domain, o.version) for o in ours.opset_import} == {
        (o.domain, o.version) for o in theirs.opset_import
    }


def test_tools_fp16_to_bf16_matches_quarks_bf16_format():
    from quark.onnx.quantization.quant_utils import convert_to_bf16

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float16[2,4] x) => (float16[2,4] y)
        {
            t = Cast<to = 1>(x)
            u = Add(x, w)
            y = Relu(u)
        }
        """
    )
    # float16 / bfloat16 literals are not parseable: attach programmatically
    model.graph.initializer.append(
        onnx.helper.make_tensor(
            "w", onnx.TensorProto.FLOAT16, [4], [0.1, -2.5, 3.14159, 1000.0]
        )
    )
    theirs = _quiet(convert_to_bf16, _copy_model(model), onnx.TensorProto.BFLOAT16, 10)
    ours = quark_tools.convert_fp16_to_bf16(model)

    # deliberate: Quark appends the boundary casts at the *end* of the node
    # list (not topologically sorted) and adds one input Cast per consuming
    # node, so an input read twice gets two identical Casts writing the same
    # `<in>_cast` (an invalid graph). We put one Cast first. Same nodes and
    # wiring once the duplicates are folded.
    def sig(m):
        return sorted(
            {
                (
                    n.op_type,
                    tuple(n.input),
                    tuple(n.output),
                    tuple((t.name, t.i) for t in n.attribute),
                )
                for n in m.graph.node
            }
        )

    assert sig(ours) == sig(theirs)
    onnx.checker.check_model(ours)
    wa = {t.name: t for t in ours.graph.initializer}
    wb = {t.name: t for t in theirs.graph.initializer}
    assert set(wa) == set(wb)
    for k in wa:
        assert wa[k].data_type == wb[k].data_type == onnx.TensorProto.BFLOAT16
        assert wa[k].raw_data == wb[k].raw_data
    assert [o.type.tensor_type.elem_type for o in ours.graph.output] == [
        o.type.tensor_type.elem_type for o in theirs.graph.output
    ]


def test_tools_nchw_to_nhwc_matches_quark():
    from quark.onnx.utils.model_utils import convert_nchw_to_nhwc as q_fn

    plain = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,3,8,8] y)
        { y = Relu(x) }
        """
    )
    quant = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,3,8,8] y)
        <float s = {0.1}, int8 z = {0}>
        {
            r = Relu(x)
            q = QuantizeLinear(r, s, z)
            y = DequantizeLinear(q, s, z)
        }
        """
    )
    flat = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,5] x) => (float[1,5] y)
        { y = Relu(x) }
        """
    )

    def sig(m):
        return (
            sorted(
                (n.op_type, n.name, list(n.input), list(n.output)) for n in m.graph.node
            ),
            [o.name for o in m.graph.output],
            [
                [d.dim_value for d in v.type.tensor_type.shape.dim]
                for v in list(m.graph.input) + list(m.graph.output)
            ],
        )

    for model in (plain, quant, flat):
        theirs = _quiet(q_fn, _copy_model(model))
        ours = quark_tools.convert_nchw_to_nhwc(model)
        assert sig(ours) == sig(theirs)
        if model is not flat:
            x = _tx_x((1, 8, 8, 3))
            out_o = _tx_ort(ours, {"x": x})[0]
            np.testing.assert_array_equal(out_o, _tx_ort(theirs, {"x": x})[0])
            assert out_o.shape == (1, 8, 8, 3)


def test_tools_qdq_to_qop_matches_quark():
    from quark.onnx.tools.convert_qdq_to_qop import convert_qdq_to_qop as q_fn

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[2,4] x, float[2,4] u) => (float[2,4] y)
        <float s = {0.1}, uint8 z = {128}, float sw = {0.05}, uint8 zw = {120},
         uint8[4,4] wq = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            uq = QuantizeLinear(u, s, z)
            ud = DequantizeLinear(uq, s, z)
            wd = DequantizeLinear(wq, sw, zw)
            m = MatMul(xd, wd)
            mq = QuantizeLinear(m, s, z)
            md = DequantizeLinear(mq, s, z)
            a = Add(md, ud)
            aq = QuantizeLinear(a, s, z)
            ad = DequantizeLinear(aq, s, z)
            p = Mul(ad, xd)
            pq = QuantizeLinear(p, s, z)
            pd = DequantizeLinear(pq, s, z)
            g1 = Sigmoid(pd)
            gq = QuantizeLinear(g1, s, z)
            y = DequantizeLinear(gq, s, z)
        }
        """
    )
    # Quark's CLI names every node (and un-shares DQs) before converting
    from quark.onnx.utils.model_utils import copy_shared_nodes

    model = _quiet(copy_shared_nodes, model)
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_qdq_to_qop(model)

    def sig(m):
        return sorted(
            (n.op_type, n.domain, list(n.input), list(n.output)) for n in m.graph.node
        )

    assert sig(ours) == sig(theirs)
    assert {"QLinearMatMul", "QLinearAdd", "QLinearMul", "QLinearSigmoid"} <= {
        n.op_type for n in ours.graph.node
    }
    feed = {"x": _tx_x((2, 4)), "u": _tx_x((2, 4), 2)}
    np.testing.assert_array_equal(_tx_ort(ours, feed)[0], _tx_ort(theirs, feed)[0])
    # fused integer kernels agree with the QDQ graph to a few quantization steps
    np.testing.assert_allclose(
        _tx_ort(ours, feed)[0], _tx_ort(model, feed)[0], atol=0.3
    )


def test_tools_resize_fs_to_pof2s_matches_quark():
    from quark.onnx.tools.convert_resize_fs_to_pof2s import (
        convert_resize_fs_to_pof2s as q_fn,
    )

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[1,1,4,4] x) => (float[1,1,8,8] y)
        <float s1 = {0.037}, int8 z1 = {3}, float s2 = {0.0123}, int8 z2 = {-5},
         float[4] scales = {1.0, 1.0, 2.0, 2.0}>
        {
            q1 = QuantizeLinear(x, s1, z1)
            d1 = DequantizeLinear(q1, s1, z1)
            r = Resize<mode = "nearest">(d1, , scales)
            q2 = QuantizeLinear(r, s2, z2)
            y = DequantizeLinear(q2, s2, z2)
        }
        """
    )
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_resize_fs_to_pof2s(model)
    _tx_same_inits(ours, theirs)
    a = _tx_inits(ours)
    assert a["z1"][1] == 0 and np.log2(float(a["s1"][1])) % 1 == 0
    x = _tx_x((1, 1, 4, 4))
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def _tx_u16_model():
    return parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,3] y)
        <float s = {0.0001}, uint16 z = {32768}, float sw = {0.0001}, uint16 zw = {32768},
         uint16[4,3] wq = {7068, 33000, 32768, 33025, 19918, 32768, 32768, 55879,
                            31997, 35338, 50798, 27628},
         float sb = {0.00001}, int32 zb = {0}, int32[3] bq = {5, -7, 100}>
        {
            xq = QuantizeLinear(x, s, z)
            xd = DequantizeLinear(xq, s, z)
            wd = DequantizeLinear(wq, sw, zw)
            m = MatMul(xd, wd)
            bd = DequantizeLinear(bq, sb, zb)
            a = Add(m, bd)
            aq = QuantizeLinear(a, s, z)
            y = DequantizeLinear(aq, s, z)
        }
        """
    )


def test_tools_u16s8_to_s16s8_matches_quark():
    from quark.onnx.tools.convert_u16s8_to_s16s8 import convert_u16s8_to_s16s8 as q_fn

    model = _tx_u16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_u16s8_to_s16s8(model)
    # the activation zero point becomes int16 0; the weight DQ is untouched
    assert [n.input[2] for n in ours.graph.node if len(n.input) > 2] == [
        n.input[2] for n in theirs.graph.node if len(n.input) > 2
    ]
    _tx_same_inits(ours, theirs)
    x = _tx_x((2, 4))
    np.testing.assert_allclose(
        _tx_ort(ours, {"x": x})[0], _tx_ort(model, {"x": x})[0], atol=1e-6
    )
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def test_tools_u16u8_to_u8u8_matches_quark():
    from quark.onnx.tools.convert_u16u8_to_u8u8 import convert_u16u8_to_u8u8 as q_fn

    model = _tx_u16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.convert_u16u8_to_u8u8(model)
    a, b = _tx_inits(ours), _tx_inits(theirs)
    assert set(a) == set(b)
    for k in a:
        assert a[k][0] == b[k][0], k
        if k == "wq":
            # deliberate differences on the re-quantized uint16 constant:
            # (1) we round to nearest, Quark truncates toward zero -> one
            # code apart where both are right; (2) Quark computes `q - zp` in
            # uint16, which wraps for q < zp and saturates those weights to
            # 255 -- we dequantize them correctly.
            src = _tx_inits(model)["wq"][1].astype(np.int64)
            ok = src >= 32768
            assert np.abs(a[k][1].astype(int) - b[k][1].astype(int))[ok].max() <= 1
            exact = np.clip(
                np.rint((src - 32768) * 0.0001 / (0.0001 * 65535 / 255) + 128), 0, 255
            )
            assert np.abs(a[k][1].astype(int) - exact).max() <= 1
        else:
            np.testing.assert_array_equal(a[k][1], b[k][1], err_msg=k)
    x = _tx_x((2, 4))
    ref = _tx_ort(model, {"x": x})[0]
    # 8-bit activations: coarser by 257x, so allow a few steps of 0.0257
    np.testing.assert_allclose(_tx_ort(ours, {"x": x})[0], ref, atol=0.5)


def test_tools_fix_shapes_matches_quark():
    from quark.onnx.tools import fix_shapes as q_mod

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[N,3] x) => (float[N,2] y)
        <float[3,2] w = {1,2,3,4,5,6}>
        {
            a = Relu(x)
            y = MatMul(a, w)
        }
        """
    )
    spec = "x:[4,3];y:[4,2]"
    t = _quiet(q_mod.fix_input_and_output_shapes, _copy_model(model), spec)
    o = quark_tools.fix_input_and_output_shapes(model, spec)

    def dims(m):
        return [
            [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(m.graph.input) + list(m.graph.output)
        ]

    assert dims(o) == dims(t) == [[4, 3], [4, 2]]
    assert quark_tools.parse_input_and_output_shapes(
        spec
    ) == q_mod.parse_input_and_output_shapes(spec)
    # intermediate tensors: Quark runs the model; the result must agree
    inferred = onnx.shape_inference.infer_shapes(o)
    shapes = _quiet(q_mod.infer_all_tensors_shape, inferred)
    t_full = _quiet(q_mod.save_all_tensors_shape, inferred, shapes)
    o_full = quark_tools.fix_shapes(model, spec)

    def vi(m):
        return {
            v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in m.graph.value_info
        }

    assert vi(o_full)["a"] == vi(t_full)["a"] == [4, 3]


def test_tools_a16w8_a8w8_nodes_match_quark(tmp_path):
    from onnx import numpy_helper
    from quark.onnx.tools.print_a16w8_a8w8_nodes import a16w8_a8w8_nodes as q_fn

    m8 = _tx_conv_qdq()
    m16 = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x) => (float[1,4,6,6] y)
        <float xs = {0.02}, int16 xz = {0}, float ws = {0.01}, int8 wz = {0}>
        {
            xq = QuantizeLinear(x, xs, xz)
            xd = DequantizeLinear(xq, xs, xz)
            wd = DequantizeLinear(wq, ws, wz)
            y = Conv(xd, wd)
        }
        """
    )
    m16.graph.initializer.append(
        numpy_helper.from_array(np.ones((4, 3, 3, 3), np.int8), "wq")
    )
    m8.graph.node[4].name = "conv8"
    m16.graph.node[3].name = "conv16"
    none = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2] x) => (float[2] y)
        { y = Relu(x) }
        """
    )
    for model, want in (
        (m8, (["conv8"], [])),
        (m16, ([], ["conv16"])),
        (none, ([], [])),
    ):
        path = str(tmp_path / "m.onnx")
        onnx.save(model, path)
        assert quark_tools.a16w8_a8w8_nodes(model) == want
        assert tuple(_quiet(q_fn, path)) == want


def _tx_bf16_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21, "com.amd.quark": 1]>
        g (float[2,4] x) => (float[2,4] y)
        <float s = {0.5}, float one = {1.0}, bfloat16 zb = {0}, int8 z8 = {0}>
        {
            a = com.amd.quark.ExtendedQuantizeLinear(x, s, zb)
            b = com.amd.quark.ExtendedDequantizeLinear(a, s, zb)
            c = com.amd.quark.ExtendedQuantizeLinear(b, one, zb)
            d = com.amd.quark.ExtendedDequantizeLinear(c, one, zb)
            e = com.amd.quark.ExtendedQuantizeLinear(d, s, z8)
            y = com.amd.quark.ExtendedDequantizeLinear(e, s, z8)
        }
        """
    )
    for i, n in enumerate(model.graph.node):
        n.name = f"n{i}"
    return model


def test_tools_replace_bfloat16_qdq_cast_matches_quark():
    from quark.onnx.tools.replace_bfloat16_qdq_cast import (
        replace_bfloat16_qdq_cast as q_fn,
    )

    model = _tx_bf16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.replace_bfloat16_qdq_cast(model)

    def sig(m):
        return [
            (n.op_type, n.domain, list(n.input), list(n.output)) for n in m.graph.node
        ]

    assert sorted(sig(ours)) == sorted(sig(theirs))
    assert sorted(n.op_type for n in ours.graph.node) == sorted(
        ["Mul", "Cast", "Cast", "Mul", "Cast", "Cast", "ExtendedQuantizeLinear"]
        + ["ExtendedDequantizeLinear"]
    )
    _tx_same_inits(ours, theirs)
    assert {k for k in _tx_inits(ours) if k.endswith("_scale")} == {
        "n0_scale",
        "n1_scale",
    }


def test_tools_insert_clip_bfloat16_qdq_matches_quark():
    from quark.onnx.tools.insert_clip_bfloat16_qdq import (
        insert_clip_bfloat16_qdq as q_fn,
    )

    model = _tx_bf16_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.insert_clip_bfloat16_qdq(model)
    assert _tx_sorted_nodes(ours) == _tx_sorted_nodes(theirs)

    def clip_inits(m):
        return {k: v for k, v in _tx_inits(m).items() if "clip" in k}

    a, b = clip_inits(ours), clip_inits(theirs)
    assert set(a) == set(b) and len(a) == 4
    for k in a:
        assert a[k][0] == b[k][0]
        np.testing.assert_array_equal(a[k][1], b[k][1], err_msg=k)

    def fed_by_clip(m):
        prod = {o: n for n in m.graph.node for o in n.output}
        return sorted(
            n.output[0]
            for n in m.graph.node
            if n.op_type == "ExtendedQuantizeLinear"
            and n.input[0] in prod
            and prod[n.input[0]].op_type == "Clip"
        )

    assert fed_by_clip(ours) == fed_by_clip(theirs) == ["a", "c"]


def _tx_cast_model():
    # Quark reconnects the consumers of the second cast only when the first
    # node's output name is a *substring* of that cast's output name (it tests
    # `a in b` on strings), hence the a / a_c1 / a_c2 naming. onnxsim rewires
    # unconditionally.
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float[4] w = {0.1234567, -2.7182818, 3.14159265, 1000.123}>
        {
            a = Relu(x)
            a_c1 = Cast<to = 16>(a)
            a_c2 = Cast<to = 1>(a_c1)
            wb = Cast<to = 16>(w)
            wf = Cast<to = 1>(wb)
            m = Add(a_c2, wf)
            n = Mul(m, a_c2)
            o1 = Cast<to = 16>(n)
            y = Cast<to = 1>(o1)
        }
        """
    )
    return model


def test_tools_remove_bf16_cast_matches_quark():
    from quark.onnx.tools.remove_bf16_cast import remove_bf16_cast as q_fn

    base = _tx_cast_model()
    theirs = _quiet(q_fn, _copy_model(base))
    ours = quark_tools.remove_bf16_cast(base)
    assert [n.op_type for n in ours.graph.node] == [
        n.op_type for n in theirs.graph.node
    ]
    assert [n.op_type for n in ours.graph.node] == ["Relu", "Add", "Mul"]
    a, b = _tx_inits(ours), _tx_inits(theirs)
    assert set(a) == set(b) == {"w_bf16"}
    np.testing.assert_array_equal(a["w_bf16"][1], b["w_bf16"][1])
    x = _tx_x((2, 4))
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def _tx_between_model():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[1,3,8,8] x, float[1,4,6,6] u) => (float[1,4,6,6] y)
        <float s = {0.1}, int8 z = {0}>
        {
            c = Conv(x, wf)
            cq = QuantizeLinear(c, s, z)
            cd = DequantizeLinear(cq, s, z)
            r = Relu(cd)
            rq = QuantizeLinear(r, s, z)
            rd = DequantizeLinear(rq, s, z)
            mu = Mul(rd, u)
            mq = QuantizeLinear(mu, s, z)
            md = DequantizeLinear(mq, s, z)
            y = Add(md, u)
        }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.full((4, 3, 3, 3), 0.5, np.float32), "wf")
    )
    return model


@pytest.mark.parametrize(
    "between",
    [[("Conv", "Relu")], [("Relu", "Mul"), ("Mul", "Add")], [("Mul", "Add")]],
)
def test_tools_remove_qdq_between_ops_matches_quark(between):
    from quark.onnx.tools.remove_qdq_between_ops import remove_qdq_between_ops as q_fn

    model = _tx_between_model()
    theirs = _quiet(q_fn, _copy_model(model), between)
    ours = quark_tools.remove_qdq_between_ops(model, between)
    assert sorted(n.op_type for n in ours.graph.node) == sorted(
        n.op_type for n in theirs.graph.node
    )
    assert _tx_inits(ours).keys() == _tx_inits(theirs).keys()
    assert len(ours.graph.node) == len(model.graph.node) - 2 * len(between)
    feed = {"x": _tx_x(), "u": _tx_x((1, 4, 6, 6), 3)}
    np.testing.assert_array_equal(_tx_ort(ours, feed)[0], _tx_ort(theirs, feed)[0])


def test_tools_remove_qdq_mul_add_matches_quark():
    from quark.onnx.tools.remove_qdq_mul_add import remove_qdq_mul_add as q_fn

    model = _tx_between_model()
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.remove_qdq_mul_add(model)
    assert sorted(n.op_type for n in ours.graph.node) == sorted(
        n.op_type for n in theirs.graph.node
    )
    feed = {"x": _tx_x(), "u": _tx_x((1, 4, 6, 6), 3)}
    np.testing.assert_array_equal(_tx_ort(ours, feed)[0], _tx_ort(theirs, feed)[0])


def test_tools_onnxtxt_roundtrip_matches_quark():
    from google.protobuf import text_format

    model = _tx_conv_qdq()
    text = quark_tools.convert_onnx_to_onnxtxt(model)
    assert text == text_format.MessageToString(model)  # what Quark's CLI writes
    back = quark_tools.convert_onnxtxt_to_onnx(text)
    assert back == model
    parsed = onnx.ModelProto()
    text_format.Parse(text.encode(), parsed)  # Quark's CLI reads bytes
    assert parsed == back


def _tx_shared_models():
    shared_dq = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float s = {0.1}, int8 z = {0},
         int8[4,4] wq = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            w = DequantizeLinear(wq, s, z)
            a = MatMul(x, w)
            y = MatMul(a, w)
        }
        """
    )
    shared_init = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,4] x) => (float[2,4] y)
        <float[4,4] w = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}>
        {
            a = MatMul(x, w)
            y = MatMul(a, w)
        }
        """
    )
    return shared_dq, shared_init


@pytest.mark.parametrize("which", [0, 1])
def test_tools_copy_shared_nodes_matches_quark(which):
    from quark.onnx.utils.model_utils import check_shared_initializers
    from quark.onnx.utils.model_utils import copy_shared_nodes as q_fn

    model = _tx_shared_models()[which]
    # a DQ-shared model has no shared *initializer* (wq is read once)
    expect = bool(which)
    assert check_shared_initializers(model) is quark_tools.check_shared_initializers(
        model
    )
    assert quark_tools.check_shared_initializers(model) is expect
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.copy_shared_nodes(model)

    def sig(m):
        return (
            sorted(n.op_type for n in m.graph.node),
            sorted(t.name for t in m.graph.initializer),
            sorted(n.name for n in m.graph.node),
        )

    assert sig(ours) == sig(theirs)
    assert not quark_tools.check_shared_initializers(ours)
    x = _tx_x((2, 4))
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(model, {"x": x})[0]
    )
    np.testing.assert_array_equal(
        _tx_ort(ours, {"x": x})[0], _tx_ort(theirs, {"x": x})[0]
    )


def test_tools_clean_initializer_in_input_matches_quark():
    from quark.onnx.utils.model_utils import clean_initializer_in_input as q_fn

    model = parser.parse_model(
        """
        <ir_version: 3, opset_import: ["": 9]>
        g (float[2] x, float[2] w) => (float[2] y)
        <float[2] w = {1.0, 2.0}>
        { y = Add(x, w) }
        """
    )
    theirs = _quiet(q_fn, _copy_model(model))
    ours = quark_tools.clean_initializer_in_input(model)
    assert [i.name for i in ours.graph.input] == [i.name for i in theirs.graph.input]
    assert ours.ir_version == theirs.ir_version == 4
    assert model.ir_version == 3  # ours does not mutate the argument


def test_tools_save_with_external_data_matches_quark(tmp_path):
    from quark.onnx.utils.model_utils import (
        save_onnx_model_with_external_data as q_fn,
    )

    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 21]>
        g (float[2,64] x) => (float[2,64] y)
        { y = MatMul(x, w) }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32), "w"
        )
    )
    for tag, fn in (
        ("q", q_fn),
        ("o", quark_tools.save_onnx_model_with_external_data),
    ):
        path = str(tmp_path / f"{tag}.onnx")
        _quiet(fn, _copy_model(model), path, True)
        assert (tmp_path / f"{tag}.onnx.data").exists()
        loaded = onnx.load(path)
        assert _tx_inits(loaded).keys() == _tx_inits(model).keys()
        for k, v in _tx_inits(loaded).items():
            np.testing.assert_array_equal(v[1], _tx_inits(model)[k][1])


# == calibration / scale parity (power-of-two MinMSE, int8 biases, methods) =======


def _heavy(rng, *shape):
    """Student-t weights: heavy tails make clipping beat ``ceil(log2)`` scales."""
    return (rng.standard_t(2.5, shape) * 0.3).astype(np.float32)


def _cal_inits(pairs):
    return [onnx.numpy_helper.from_array(a, n) for n, a in pairs]


def _cal_mlp(seed):
    rng = np.random.default_rng(seed)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[4,24] x) => (float[4,10] y) {
            h0 = Gemm(x, w1, b1)
            h1 = Relu(h0)
            h2 = Gemm(h1, w2, b2)
            h3 = Relu(h2)
            y = Gemm(h3, w3, b3)
        }
        """
    )
    m.graph.initializer.extend(
        _cal_inits(
            [
                ("w1", _heavy(rng, 24, 48)),
                ("b1", _heavy(rng, 48)),
                ("w2", _heavy(rng, 48, 32)),
                ("b2", _heavy(rng, 32)),
                ("w3", _heavy(rng, 32, 10)),
                ("b3", _heavy(rng, 10)),
            ]
        )
    )
    return m, (4, 24)


def _cal_conv(seed):
    rng = np.random.default_rng(seed)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[2,3,12,12] x) => (float[2,6,6,6] y) {
            c0 = Conv<pads=[1,1,1,1]>(x, w1, b1)
            r0 = Relu(c0)
            c1 = Conv<pads=[1,1,1,1]>(r0, w2, b2)
            r1 = Relu(c1)
            y = MaxPool<kernel_shape=[2,2], strides=[2,2]>(r1)
        }
        """
    )
    m.graph.initializer.extend(
        _cal_inits(
            [
                ("w1", _heavy(rng, 8, 3, 3, 3)),
                ("b1", _heavy(rng, 8)),
                ("w2", _heavy(rng, 6, 8, 3, 3)),
                ("b2", _heavy(rng, 6)),
            ]
        )
    )
    return m, (2, 3, 12, 12)


def _cal_attn(seed):
    """LayerNorm / Softmax / residual Add around four projections."""
    rng = np.random.default_rng(seed)
    d = 16
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 17]>
        g (float[1,6,{d}] x) => (float[1,6,{d}] y) {{
            n = LayerNormalization<axis=-1, epsilon=1e-5>(x, ln_s, ln_b)
            q = MatMul(n, wq)
            k = MatMul(n, wk)
            kt = Transpose<perm=[0,2,1]>(k)
            s = MatMul(q, kt)
            sc = Mul(s, c)
            p = Softmax<axis=-1>(sc)
            v = MatMul(n, wv)
            a = MatMul(p, v)
            o = MatMul(a, wo)
            y = Add(x, o)
        }}
        """
    )
    m.graph.initializer.extend(
        _cal_inits(
            [
                ("ln_s", (1 + 0.2 * rng.standard_normal(d)).astype(np.float32)),
                ("ln_b", (0.1 * rng.standard_normal(d)).astype(np.float32)),
                ("wq", _heavy(rng, d, d)),
                ("wk", _heavy(rng, d, d)),
                ("wv", _heavy(rng, d, d)),
                ("wo", _heavy(rng, d, d)),
                ("c", np.array(0.25, np.float32)),
            ]
        )
    )
    return m, (1, 6, d)


CAL_MODELS = {"mlp": _cal_mlp, "conv": _cal_conv, "attn": _cal_attn}


def _norm_name(name):
    for cut in ("_QuantizeLinear_Input", "_quantized", "/f", "/dq"):
        name = name.removesuffix(cut)
    return name.split("/qdq")[0]


def _qparam_map(model):
    """``({tensor: (scale, zero_point, dtype)}, {initializer: (dequantized,
    int8-code abs-sum)})``: every activation Q node, and every DQ that reads
    an initializer."""
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    acts, consts = {}, {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = inits[n.input[2]]
            acts[_norm_name(n.input[0])] = (
                float(inits[n.input[1]]),
                int(zp),
                str(zp.dtype),
            )
        elif n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            deq = (q.astype(np.float64) - inits[n.input[2]]) * inits[n.input[1]]
            consts[_norm_name(n.input[0])] = (
                deq,
                q.dtype.name,
                float(np.max(inits[n.input[1]])),
            )
    return acts, consts


def _check_cal_parity(model, shape, preset, tmp_path, rtol, zp_atol=0, constants=True):
    q_acts, q_consts = _qparam_map(quark_quantize(model, preset, shape, tmp_path))
    m_acts, m_consts = _qparam_map(mine_quantize(model, preset, shape))
    assert q_acts and set(q_acts) == set(m_acts)
    for name, (scale, zp, dt) in q_acts.items():
        ms, mz, mdt = m_acts[name]
        assert mdt == dt, name
        np.testing.assert_allclose(ms, scale, rtol=rtol, err_msg=name)
        assert abs(mz - zp) <= zp_atol, (name, mz, zp)
    if not constants:
        return
    assert set(q_consts) == set(m_consts)
    for name, (want, dt, scale) in q_consts.items():
        got, mdt, _ = m_consts[name]
        assert mdt == dt, name
        # int8 weights / biases dequantize identically; an int32 bias carries
        # the activation scale's (percentile-binning) difference: a code or two
        atol = 2 * scale if dt == "int32" else 1e-4
        np.testing.assert_allclose(
            got, want, rtol=max(rtol, 1e-6), atol=atol, err_msg=name
        )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
def test_xint8_minmse_pof2_matches_quark(model_name, seed, tmp_path):
    """XINT8: power-of-two scales picked by MinMSE -- activations (histogram
    search), weights, and the int8 biases / constants -- are identical to
    Quark's: same scale, zero point and dtype per tensor, same int8 codes."""
    model, shape = CAL_MODELS[model_name](seed)
    _check_cal_parity(model, shape, "XINT8", tmp_path, rtol=0)


def _drain(reader):
    out = []
    while (b := reader.get_next()) is not None:
        out.append(b)
    return out


def test_xint8_cases_where_ceil_log2_differs_from_quark(tmp_path):
    """The previous ``ceil(log2(scale))`` / int32-bias rule lands on different
    scales than Quark on these heavy-tailed models, so the parity above is not
    vacuous."""
    from onnxsim.full_qdq import quantize_full_qdq

    differing = 0
    for name, build in sorted(CAL_MODELS.items()):
        for seed in range(4):
            model, shape = build(seed)
            q_acts, q_consts = _qparam_map(
                quark_quantize(model, "XINT8", shape, tmp_path)
            )
            old = quantize_full_qdq(
                model,
                _drain(_reader(shape)()),
                activation_dtype="uint8",
                method="minmax",
                symmetric_activations=True,
                power_of_two=True,
                per_channel=False,
            )
            o_acts, o_consts = _qparam_map(old)
            differing += any(
                o_acts[k][0] != v[0] for k, v in q_acts.items() if k in o_acts
            )
    assert differing >= 4


@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
def test_xint8_int32_bias_option_keeps_int32(model_name):
    model, shape = CAL_MODELS[model_name](0)
    cfg = qc.QConfig.get_default_config("XINT8")
    cfg.extra_options["Int32Bias"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(shape)()
        )
    dtypes = {onnx.numpy_helper.to_array(i).dtype for i in out.graph.initializer}
    has_bias = any(n.op_type in ("Gemm", "Conv") for n in model.graph.node)
    assert (np.dtype(np.int32) in dtypes) == has_bias


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
@pytest.mark.parametrize(
    "preset, rtol, zp_atol",
    [
        # MinMax: the same range -> the same scale
        ("A8W8", 1e-6, 0),
        ("A16W8", 1e-6, 0),
        # Percentile: histogram binning moves a scale by a few 1e-4
        ("U8S8_AAWS", 2e-3, 1),
        ("S8S8_AAWS", 2e-3, 1),
        ("U8U8_AAWA", 2e-3, 1),
        ("S16S8_ASWS", 2e-3, 40),
        ("U16S8_AAWS", 2e-3, 40),
    ],
)
def test_calibration_methods_match_quark_per_preset(
    preset, rtol, zp_atol, model_name, seed, tmp_path
):
    """Per-tensor scale / zero point / dtype and dequantized weights, biases
    and constants of each preset's calibration method (MinMax, Percentile
    99.999 / 99.9999, symmetric or not), including Softmax's fixed (0, 1)
    output range and the int8 weight treatment of non-weight constants."""
    model, shape = CAL_MODELS[model_name](seed)
    # U8U8_AAWA's uint8 asymmetric weights are a documented approximation
    # (int8 symmetric here): only its activations are compared
    _check_cal_parity(
        model, shape, preset, tmp_path, rtol, zp_atol, constants=preset != "U8U8_AAWA"
    )


# -- preset combinations: mixed formats and mixed precision -------------------------
#
# BF16_BFP16 / BF16_MXINT8 (bfloat16 activations over block-format constants),
# MX9_INT8 (block-format activations over int8 constants) and the
# BF16_MIXED_BFP16 / BF16_MIXED_MXINT8 AutoMixprecision presets. Quark's mixed
# presets pick candidate layers by *node name*, so these models name every node.


def _named(model):
    for i, n in enumerate(model.graph.node):
        n.name = f"{n.op_type}_{i}"
    return model


def _transformer():
    """Attention + MLP block. No ``MatMul`` -> ``Add(const)`` (Quark's
    pre-processing would fuse that into a ``Gemm``)."""
    rng = np.random.default_rng(11)
    d = 16
    m = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": 20]>
        g (float[1,4,{d}] x) => (float[1,4,{d}] y) {{
            q = MatMul(x, wq)
            k = MatMul(x, wk)
            v = MatMul(x, wv)
            kt = Transpose<perm=[0,2,1]>(k)
            s0 = MatMul(q, kt)
            s1 = Mul(s0, scale)
            p = Softmax<axis=-1>(s1)
            c = MatMul(p, v)
            o = MatMul(c, wo)
            r = Add(x, o)
            n = LayerNormalization<axis=-1, epsilon=1e-5>(r, g1, b1)
            h = MatMul(n, w1)
            ge = Gelu(h)
            f = MatMul(ge, w2)
            y = Add(n, f)
        }}
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, d, d), "wq"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "wk"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "wv"),
            onnx.numpy_helper.from_array(_w(rng, d, d), "wo"),
            onnx.numpy_helper.from_array(np.array(0.25, np.float32), "scale"),
            onnx.numpy_helper.from_array(np.ones(d, np.float32), "g1"),
            onnx.numpy_helper.from_array(np.zeros(d, np.float32), "b1"),
            onnx.numpy_helper.from_array(_w(rng, d, 32), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 32, d), "w2"),
        ]
    )
    return m, (1, 4, d)


def _branchy():
    """A residual ``Add`` reading a tensor that also feeds a ``Gemm``, a
    Gemm -> Gemm chain without activation in between, and a ``MatMul`` last."""
    rng = np.random.default_rng(12)
    m = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["": 17]>
        g (float[3,16] x) => (float[3,16] y) {
            a = Gemm(x, w1, b1)
            b = Gemm(a, w2, b2)
            r = Add(x, b)
            y = MatMul(r, w3)
        }
        """
    )
    m.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(_w(rng, 16, 16), "w1"),
            onnx.numpy_helper.from_array(_w(rng, 16), "b1"),
            onnx.numpy_helper.from_array(_w(rng, 16, 16), "w2"),
            onnx.numpy_helper.from_array(_w(rng, 16), "b2"),
            onnx.numpy_helper.from_array(_w(rng, 16, 16), "w3"),
        ]
    )
    return m, (3, 16)


MIXED_MODELS = {**MODELS, "branchy": _branchy, "transformer": _transformer}
_MIXED_FORMATS = ["BF16_BFP16", "BF16_MXINT8", "MX9_INT8"]
_MIXED_PRECISION = ["BF16_MIXED_BFP16", "BF16_MIXED_MXINT8"]


def _mixed_pair(preset, model_name, tmp_path):
    model, shape = MIXED_MODELS[model_name]()
    _named(model)
    return (
        quark_quantize(model, preset, shape, tmp_path, f"{model_name}_{preset}"),
        mine_quantize(model, preset, shape),
        shape,
    )


@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
@pytest.mark.parametrize("preset", _MIXED_FORMATS + _MIXED_PRECISION)
def test_mixed_preset_graph_matches_quark(preset, model_name, tmp_path):
    """Same custom-op / (Extended)Q/DQ placement, attributes, block axes and
    even the names of the dual nodes Quark inserts at precision boundaries."""
    q, m, _ = _mixed_pair(preset, model_name, tmp_path)
    assert _cop_map(m) == _cop_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.skipif(_ops_lib() is None, reason="Quark's custom-op library is not built")
@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
@pytest.mark.parametrize("preset", _MIXED_FORMATS + _MIXED_PRECISION)
def test_mixed_preset_outputs_match_quark(preset, model_name, tmp_path):
    q, m, shape = _mixed_pair(preset, model_name, tmp_path)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32) * 2
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_name", ["mlp", "conv", "transformer"])
def test_mx9_int8_constants_are_bit_identical(model_name, tmp_path):
    """The int8 codes, scales and zero points of the constants (weights *and*
    biases: symmetric per tensor, ``max|w| / 127``) equal Quark's."""
    q, m, _ = _mixed_pair("MX9_INT8", model_name, tmp_path)

    def consts(model):
        return {
            i.name: onnx.numpy_helper.to_array(i)
            for i in model.graph.initializer
            if i.name.endswith(("_quantized", "_scale", "_zero_point"))
        }

    theirs, ours = consts(q), consts(m)
    assert theirs and set(ours) == set(theirs)
    for name in theirs:
        assert ours[name].dtype == theirs[name].dtype, name
        np.testing.assert_array_equal(ours[name], theirs[name], err_msg=name)


@pytest.mark.parametrize("preset", _MIXED_FORMATS + _MIXED_PRECISION)
def test_mixed_preset_single_op_placement_matches_quark(preset, tmp_path):
    """Quark's op coverage, op by op, for the combined presets: the op counts
    (nodes, custom ops, Q/DQ) agree on every single-op graph Quark accepts."""
    diffs = []
    for name, case in _single_op_cases().items():
        case = dict(case, op=case.get("op", name))
        model = _named(_single_op_model(case))
        try:
            q = quark_quantize(model, preset, case["shapes"][0], tmp_path, name)
        except Exception:  # Quark itself rejects this graph
            continue
        m = mine_quantize(model, preset, case["shapes"][0])
        got, want = _op_counts(m), _op_counts(q)
        if got != want and name not in KNOWN_GRAPH_DIFF:
            diffs.append((name, got, want))
    assert not diffs, json.dumps(diffs, indent=1)


def test_mixed_precision_keeps_biases_float_and_matches_quark(tmp_path):
    """Quark's default ``metric_threshold=0`` promotes every Conv / Gemm /
    MatMul; the biases stay float constants (``QuantizeBias=False``)."""
    model, shape = _mlp()
    _named(model)
    q = quark_quantize(model, "BF16_MIXED_BFP16", shape, tmp_path, "promote")
    m = mine_quantize(model, "BF16_MIXED_BFP16", shape)
    for out in (q, m):
        reads = {i for n in out.graph.node if n.op_type == "Gemm" for i in n.input}
        assert {"b1", "b2"} <= reads  # read straight from the float initializers
    assert _cop_map(m) == _cop_map(q)


@pytest.mark.parametrize("preset", ["FP16_ADAQUANT", "BF16_ADAQUANT"])
def test_half_adaquant_graph_matches_quark(preset, tmp_path):
    """Quark's FP16/BF16 AdaQuant presets emit the plain FP16/BF16 graph (the
    algorithm only retunes initializer values; it also fails outright on a
    graph whose last op is a pooling, so only the MLP is probed). onnxsim has
    no AdaQuant for float formats: the preset exists, and runs only when told
    to ignore the algorithm."""
    model, shape = _mlp()
    q = quark_quantize(model, preset, shape, tmp_path, preset)
    cfg = qc.QConfig.get_default_config(preset)
    with pytest.raises(NotImplementedError):
        qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=None, ignore_unsupported_algos=True
        )
    assert _ext_map(m) == _ext_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.parametrize(
    "preset", ["BF16_MIXED_BFP16_ADAQUANT", "BF16_MIXED_MXINT8_ADAQUANT"]
)
def test_mixed_adaquant_graph_matches_quark(preset, tmp_path):
    model, shape = _mlp()
    _named(model)
    q = quark_quantize(model, preset, shape, tmp_path, preset)
    cfg = qc.QConfig.get_default_config(preset)
    with pytest.raises(NotImplementedError):
        qc.ModelQuantizer(cfg).quantize_model(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(model, ignore_unsupported_algos=True)
    assert _cop_map(m) == _cop_map(q)
    assert _op_counts(m) == _op_counts(q)


@pytest.mark.parametrize("model_name", ["branchy", "transformer"])
@pytest.mark.parametrize("preset", _BLOCK + _HALF)
def test_block_and_half_presets_match_quark_on_transformer_graphs(
    preset, model_name, tmp_path
):
    """The single-format presets on the attention block (``LayerNormalization``
    is quantized end to end when its input already is, scale / bias included)
    and the residual graph."""
    q, m, _ = _mixed_pair(preset, model_name, tmp_path)
    assert _cop_map(m) == _cop_map(q)
    assert _ext_map(m) == _ext_map(q)
    assert _op_counts(m) == _op_counts(q)


# -- integer presets: the CNN presets and the mixed int16 / int8 preset ---------------


def _qdq_params(model):
    """``(activations, weights, int8_biases)``: every activation
    ``QuantizeLinear``'s ``(scale, zero_point, dtype)``, each weight's
    ``(max scale, dtype)`` (int8 / int16 codes of rank >= 2) and the int8
    bias codes by name."""
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    acts, weights, biases = [], [], {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = inits[n.input[2]]
            acts.append((float(inits[n.input[1]]), int(zp), str(zp.dtype)))
        elif n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = inits[n.input[0]]
            if q.dtype in (np.int8, np.int16) and q.ndim >= 2:
                weights.append((float(np.max(inits[n.input[1]])), str(q.dtype)))
            elif q.dtype == np.int8:
                key = n.input[0].replace("_quantized", "").split("/")[0]
                biases[key] = (q, float(inits[n.input[1]]))
    return sorted(acts), sorted(weights), biases


def _quantize_int_pair(preset, model_name, tmp_path):
    model, shape = MIXED_MODELS[model_name]()
    _named(model)
    q = quark_quantize(model, preset, shape, tmp_path, f"{model_name}_{preset}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(qc.QConfig.get_default_config(preset)).quantize_model(
            model,
            calibration_data_reader=_reader(shape)(),
        )
    return model, q, m, shape


@pytest.mark.parametrize("model_name", ["mlp", "conv", "gemm_transb", "branchy"])
@pytest.mark.parametrize("preset", ["INT8_CNN_DEFAULT", "INT16_CNN_DEFAULT"])
def test_cnn_default_presets_match_quark_exactly(preset, model_name, tmp_path):
    """Plain min/max calibration, asymmetric uint8 / uint16 activations,
    per-tensor int8 / int16 weights: the same quantization parameters, and the
    same outputs, as Quark."""
    _, q, m, shape = _quantize_int_pair(preset, model_name, tmp_path)
    q_acts, q_w, _ = _qdq_params(q)
    m_acts, m_w, _ = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_array_equal([a[1] for a in m_acts], [a[1] for a in q_acts])
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=1e-5
    )
    assert [w[1] for w in m_w] == [w[1] for w in q_w]
    np.testing.assert_allclose([w[0] for w in m_w], [w[0] for w in q_w], rtol=1e-6)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("model_name", ["mlp", "conv", "gemm_transb"])
@pytest.mark.parametrize("preset", ["INT8_CNN_ACCURATE", "INT16_CNN_ACCURATE"])
def test_cnn_accurate_presets_match_quark_parameters(preset, model_name, tmp_path):
    """Percentile 99.9999 calibration: activation parameters agree up to
    histogram binning, weight scales exactly. AdaRound only changes weight
    codes (for int8 and int16 weights alike, as Quark's FastFinetune does)."""
    model, q, m, shape = _quantize_int_pair(preset, model_name, tmp_path)
    q_acts, q_w, _ = _qdq_params(q)
    m_acts, m_w, _ = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose(
        [a[1] for a in m_acts], [a[1] for a in q_acts], atol=40 if "16" in preset else 2
    )
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose([w[0] for w in m_w], [w[0] for w in q_w], rtol=1e-6)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)

    def rel(a):
        return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

    assert rel(_run(m, x)) < max(3 * rel(_run(q, x)), 0.05)


def test_int16_cnn_accurate_runs_adaround_on_int16_weights():
    model, shape = _mlp()
    cfg = qc.QConfig.get_default_config("INT16_CNN_ACCURATE")
    quantizer = qc.ModelQuantizer(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = quantizer.quantize_model(model, calibration_data_reader=_reader(shape)())
    assert quantizer.last_weight_rounding["adaround"]
    assert {
        t.data_type
        for t in out.graph.initializer
        if t.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.INT16)
        and len(t.dims) == 2
    } == {onnx.TensorProto.INT16}


@pytest.mark.parametrize("model_name", ["mlp", "conv", "gemm_transb", "branchy"])
def test_s16s16_mixed_s8s8_matches_quark(model_name, tmp_path):
    """int16 everywhere, every Conv / Gemm / MatMul promoted to int8 inputs,
    weights and (int8, per-tensor) biases; the promoted layers' outputs stay
    int16 and no convert pairs appear."""
    model, q, m, shape = _quantize_int_pair("S16S16_MIXED_S8S8", model_name, tmp_path)
    # (Quark keeps a Relu behind its producer; onnxsim folds it into the Q)
    assert {k: v for k, v in _op_counts(m).items() if k != "Relu"} == {
        k: v for k, v in _op_counts(q).items() if k != "Relu"
    }
    q_acts, q_w, q_b = _qdq_params(q)
    m_acts, m_w, m_b = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose([a[1] for a in m_acts], [a[1] for a in q_acts], atol=40)
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    assert m_w == q_w
    assert set(m_b) == set(q_b) and q_b
    for k in q_b:  # biases: scale max|b|/127, codes equal (up to a rounding tie)
        np.testing.assert_allclose(m_b[k][1], q_b[k][1], rtol=1e-6)
        np.testing.assert_allclose(m_b[k][0], q_b[k][0], atol=1)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)

    def rel(a):
        return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

    assert rel(_run(m, x)) < max(2 * rel(_run(q, x)), 0.02)
    assert float(np.linalg.norm(_run(m, x) - _run(q, x)) / np.linalg.norm(ref)) < 0.02


@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
def test_vint8_matches_quark(model_name, tmp_path):
    """Signed power-of-two int8 everywhere: Quark's ``VINT8`` quantizes every
    activation (no Relu folding, one dedicated Q/DQ pair per consumer) and
    stores weights *and biases* as per-tensor int8. Same graph structure and
    weight / bias parameters; activation scales are powers of two that may sit
    one octave off (Quark searches the MSE-best power of two, onnxsim rounds
    up)."""
    model, q, m, shape = _quantize_int_pair("VINT8", model_name, tmp_path)
    assert _op_counts(m) == _op_counts(q)
    q_acts, q_w, q_b = _qdq_params(q)
    m_acts, m_w, m_b = _qdq_params(m)
    assert [(a[1], a[2]) for a in m_acts] == [(0, "int8")] * len(q_acts)
    assert len(m_acts) == len(q_acts)
    log2 = lambda acts: np.log2([a[0] for a in acts])  # noqa: E731
    assert np.all(log2(m_acts) == np.round(log2(m_acts)))
    assert np.all(np.abs(log2(m_acts) - log2(q_acts)) <= 1)
    assert m_w == q_w
    assert set(m_b) == set(q_b)
    if model_name != "transformer":  # (there the constants are activation-path ones)
        for k in q_b:
            assert m_b[k][1] == q_b[k][1], k
            np.testing.assert_array_equal(m_b[k][0], q_b[k][0], err_msg=k)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)
    err = float(np.linalg.norm(_run(m, x) - _run(q, x)) / np.linalg.norm(ref))
    assert err < 0.1


# -- INT{8,16}_TRANSFORMER_{DEFAULT,ACCURATE}: Quark's NPU transformer quantizer ------
# (``enable_npu_transformer``): only Gemm and MatMul-with-constant-B nodes are
# quantized (Q/DQ on their inputs and outputs, int8 / int16 per-tensor weights,
# int32 biases); every other op -- Softmax, LayerNormalization, Gelu, Add, Mul, a
# MatMul of two activations -- stays float; a model with no such node comes back
# unchanged. DEFAULT calibrates with the mean of the per-batch min / max.

_TRANSFORMER_PRESETS = []


@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
@pytest.mark.parametrize(
    "preset", ["INT8_TRANSFORMER_DEFAULT", "INT16_TRANSFORMER_DEFAULT"]
)
def test_transformer_default_presets_match_quark_exactly(preset, model_name, tmp_path):
    """Same Q/DQ placement and graph, the same activation / weight
    parameters, the same outputs."""
    _, q, m, shape = _quantize_int_pair(preset, model_name, tmp_path)
    assert _op_counts(m) == _op_counts(q)
    q_acts, q_w, _ = _qdq_params(q)
    m_acts, m_w, _ = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_array_equal([a[1] for a in m_acts], [a[1] for a in q_acts])
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=1e-5
    )
    assert [w[1] for w in m_w] == [w[1] for w in q_w]
    np.testing.assert_allclose([w[0] for w in m_w], [w[0] for w in q_w], rtol=1e-6)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    np.testing.assert_allclose(_run(m, x), _run(q, x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("model_name", sorted(MIXED_MODELS))
@pytest.mark.parametrize(
    "preset", ["INT8_TRANSFORMER_ACCURATE", "INT16_TRANSFORMER_ACCURATE"]
)
def test_transformer_accurate_presets_match_quark_parameters(
    preset, model_name, tmp_path
):
    """Percentile 99.9999 calibration (activation parameters agree up to
    histogram binning, weight scales exactly) and AdaRound, which only changes
    weight codes -- and runs for int8 weights only."""
    model, q, m, shape = _quantize_int_pair(preset, model_name, tmp_path)
    assert _op_counts(m) == _op_counts(q)
    q_acts, q_w, _ = _qdq_params(q)
    m_acts, m_w, _ = _qdq_params(m)
    assert [a[2] for a in m_acts] == [a[2] for a in q_acts]
    np.testing.assert_allclose(
        [a[1] for a in m_acts], [a[1] for a in q_acts], atol=40 if "16" in preset else 2
    )
    np.testing.assert_allclose(
        [a[0] for a in m_acts], [a[0] for a in q_acts], rtol=2e-3
    )
    np.testing.assert_allclose([w[0] for w in m_w], [w[0] for w in q_w], rtol=1e-6)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)

    def rel(a):
        return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))

    assert rel(_run(m, x)) < max(3 * rel(_run(q, x)), 0.05)


# =============================================================================
# MATMUL_NBITS: weight-only MatMulNBits (onnxsim.quark_matmul_nbits)
# =============================================================================
# Quark (like onnxsim) rewrites constant-weight MatMuls to com.microsoft::
# MatMulNBits. Quark first runs ONNX Runtime graph optimizations that fuse
# MatMul + Add into an unquantized Gemm (``SkipPreprocess``); the parity runs set
# ``SkipPreprocess=True``, which is what onnxsim reproduces.


def _nb_model(body, shapes, inputs, seed=0):
    model = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g ({inputs}) => (float y) '
        f"{{ {body} }}"
    )
    rng = np.random.default_rng(seed)
    model.graph.initializer.extend(
        onnx.numpy_helper.from_array(
            (rng.standard_normal(s) * 0.5).astype(np.float32), n
        )
        for n, s in shapes
    )
    for i, node in enumerate(model.graph.node):
        node.name = f"n{i}"
    return model


def _nb_models():
    return {
        "mlp": (
            _nb_model(
                "h = MatMul(x, w1)\n h2 = Relu(h)\n h3 = MatMul(h2, w2)\n"
                " h4 = Relu(h3)\n y = MatMul(h4, w3)",
                [("w1", (64, 96)), ("w2", (96, 64)), ("w3", (64, 8))],
                "float[3,64] x",
            ),
            (3, 64),
        ),
        "attention": (
            _nb_model(
                "q = MatMul(x, wq)\n k = MatMul(x, wk)\n v = MatMul(x, wv)\n"
                " kt = Transpose<perm=[1,0]>(k)\n s = MatMul(q, kt)\n"
                " p = Softmax<axis=-1>(s)\n a = MatMul(p, v)\n"
                " o = MatMul(a, wo)\n y = Relu(o)",
                [(n, (48, 48)) for n in ("wq", "wk", "wv", "wo")],
                "float[6,48] x",
            ),
            (6, 48),
        ),
        # K and N not multiples of the block size, odd block counts
        "ragged": (
            _nb_model(
                "h = MatMul(x, w1)\n h2 = Relu(h)\n y = MatMul(h2, w2)",
                [("w1", (100, 130)), ("w2", (130, 20))],
                "float[3,100] x",
            ),
            (3, 100),
        ),
        "batched": (
            _nb_model(
                "h = MatMul(x, w1)\n h2 = Relu(h)\n y = MatMul(h2, w2)",
                [("w1", (64, 160)), ("w2", (160, 32))],
                "float[2,5,64] x",
            ),
            (2, 5, 64),
        ),
    }


_NB_MODELS = _nb_models()


def _nb_quark(model, shape, tmp_path, mm=None, gptq=None, algo=None, skip=True):
    import copy

    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx import ModelQuantizer, QConfig

    rng = np.random.default_rng(1)
    data = [{"x": rng.standard_normal(shape).astype(np.float32)} for _ in range(4)]

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

        def __iter__(self):
            return iter(data)

    src, dst = str(tmp_path / "nb.onnx"), str(tmp_path / "nb_q.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        # get_default_config hands out one shared object: never edit it in place
        cfg = copy.deepcopy(QConfig.get_default_config("MATMUL_NBITS"))
        g = cfg.global_quant_config
        g.include_cle = False
        eo = dict(g.extra_options)
        eo["SkipPreprocess"] = skip
        eo["MatMulNBitsParams"] = {**eo["MatMulNBitsParams"], **(mm or {})}
        if gptq is not None:
            eo["GPTQParams"] = dict(gptq)
        g.extra_options = eo
        if algo:
            g.algo_config = algo
        ModelQuantizer(cfg).quantize_model(src, dst, R())
    return onnx.load(dst), data


def _nb_mine(model, mm=None, gptq=None, data=None, algo=None, **opts):
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"].update(mm or {})
    cfg.extra_options["SkipPreprocess"] = True
    cfg.extra_options.update(opts)
    if gptq is not None:
        cfg.extra_options["GPTQParams"] = dict(gptq)
    if algo:
        cfg.algo_config = algo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=data
        )


def _nb_nodes(model):
    return [
        (n.op_type, n.domain, n.name, tuple(n.input), tuple(n.output), _attrs(n))
        for n in model.graph.node
    ]


def _nb_inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def _assert_nb_equal(q, m, zero_point_atol=None):
    """Same nodes / attributes / tensors; ``zero_point_atol`` relaxes float
    zero points (HQQ) to a tolerance, everything else must be bit-identical."""
    assert _nb_nodes(m) == _nb_nodes(q)
    assert ("com.microsoft", 1) in {(o.domain, o.version) for o in m.opset_import}
    qi, mi = _nb_inits(q), _nb_inits(m)
    assert set(mi) == set(qi)
    for name, a in qi.items():
        b = mi[name]
        assert (a.shape, a.dtype) == (b.shape, b.dtype), name
        if zero_point_atol is not None and name.endswith("_zero_points"):
            np.testing.assert_allclose(b, a, atol=zero_point_atol, err_msg=name)
        else:
            np.testing.assert_array_equal(b, a, err_msg=name)


def _assert_nb_outputs_equal(q, m, shape):
    sizes = [
        _attrs(n)["block_size"] for n in q.graph.node if n.op_type == "MatMulNBits"
    ]
    if any(s < 16 or s & (s - 1) for s in sizes):
        return  # GPTQ's default block is K; MatMulNBits only runs power-of-two blocks
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    np.testing.assert_array_equal(_run(m, x), _run(q, x))


def _nb_id(d):
    return "-".join(f"{k}{v}" for k, v in d.items()) or "defaults"


def test_matmul_nbits_preset_registered_like_quark():
    q = DefaultConfigMapping["MATMUL_NBITS"].extra_options
    m = qc.QConfig.get_default_config("MATMUL_NBITS").extra_options
    assert m["UseMatMulNBits"] == q["UseMatMulNBits"]
    assert m["MatMulNBitsParams"] == q["MatMulNBitsParams"]


@pytest.mark.parametrize("model_name", sorted(_NB_MODELS))
@pytest.mark.parametrize(
    "mm",
    [
        {},
        {"GroupSize": 32},
        {"GroupSize": 64, "Symmetric": False},
        {"GroupSize": 32, "Symmetric": False, "AccuracyLevel": 4},
        {"GroupSize": 16, "AccuracyLevel": 0},
    ],
    ids=_nb_id,
)
def test_matmul_nbits_default_matches_quark(model_name, mm, tmp_path):
    """Identical node placement, attributes, packed weights, scales and zero
    points, and (so) identical ONNX Runtime outputs."""
    model, shape = _NB_MODELS[model_name]
    q, data = _nb_quark(model, shape, tmp_path, mm)
    m = _nb_mine(model, mm, data=data)
    assert "MatMulNBits" in [n.op_type for n in q.graph.node]
    _assert_nb_equal(q, m)
    _assert_nb_outputs_equal(q, m, shape)


@pytest.mark.parametrize("model_name", ["mlp", "ragged", "batched"])
def test_matmul_nbits_hqq_matches_quark(model_name, tmp_path):
    """HQQ: float (unpacked) zero points, no accuracy_level. Quark runs torch
    float32; the zero points agree to float rounding."""
    model, shape = _NB_MODELS[model_name]
    mm = {"GroupSize": 32, "Algorithm": "HQQ"}
    q, data = _nb_quark(model, shape, tmp_path, mm)
    m = _nb_mine(model, mm, data=data)
    assert "accuracy_level" not in _attrs(m.graph.node[0])
    _assert_nb_equal(q, m, zero_point_atol=1e-5)
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    ref = _run(model, x)
    assert np.linalg.norm(_run(m, x) - _run(q, x)) / np.linalg.norm(ref) < 0.01


_NB_GPTQ = [
    {},
    {"GroupSize": 32},
    {"GroupSize": 32, "PerChannel": True},
    {"PerChannel": True},
    {"WeightSymmetric": False, "PerChannel": True},
    {"WeightSymmetric": False, "GroupSize": 32},
    {"ActOrder": True, "PerChannel": True},
    {"ActOrder": True, "GroupSize": 32, "PerChannel": True},
    {"MSE": True, "PerChannel": True},
    {"MSE": True, "GroupSize": 64, "WeightSymmetric": False},
    {"BlockSize": 32, "PercDamp": 0.1, "GroupSize": 32},
]


@pytest.mark.parametrize("model_name", ["mlp", "ragged", "batched"])
@pytest.mark.parametrize("gptq", _NB_GPTQ, ids=_nb_id)
def test_matmul_nbits_gptq_matches_quark(model_name, gptq, tmp_path):
    """Quark's GPTQ path (Hessian from the first batch, Quark's grid, then a
    re-derived block grid is packed; error propagation off, as in Quark 0.13):
    bit-identical tensors."""
    model, shape = _NB_MODELS[model_name]
    mm = {"Algorithm": "GPTQ"}
    q, data = _nb_quark(model, shape, tmp_path, mm, gptq)
    m = _nb_mine(model, mm, gptq, data=data)
    _assert_nb_equal(q, m)
    _assert_nb_outputs_equal(q, m, shape)


def test_matmul_nbits_gptq_config_alone_does_not_select_gptq(tmp_path):
    """Quark: a GPTQConfig only feeds GPTQParams; MatMulNBitsParams.Algorithm
    decides, so this is the plain DEFAULT quantization."""
    from quark.onnx import GPTQConfig as QuarkGPTQConfig

    model, shape = _NB_MODELS["mlp"]
    mm = {"GroupSize": 32}
    q, data = _nb_quark(
        model, shape, tmp_path, mm, algo=[QuarkGPTQConfig(bits=4, group_size=32)]
    )
    m = _nb_mine(model, mm, data=data, algo=[qc.GPTQConfig(bits=4, group_size=32)])
    _assert_nb_equal(q, m)
    plain, _ = _nb_quark(model, shape, tmp_path, mm)
    _assert_nb_equal(plain, m)


def test_matmul_nbits_gptq_config_is_ignored_like_quark(tmp_path):
    """Even with ``Algorithm=GPTQ`` Quark does not read a GPTQConfig (only
    ``GPTQParams``): same tensors as the GPTQ defaults."""
    from quark.onnx import GPTQConfig as QuarkGPTQConfig

    model, shape = _NB_MODELS["mlp"]
    mm = {"Algorithm": "GPTQ"}
    kw = dict(bits=4, group_size=32, per_channel=True, act_order=True)
    q, data = _nb_quark(model, shape, tmp_path, mm, algo=[QuarkGPTQConfig(**kw)])
    m = _nb_mine(model, mm, data=data, algo=[qc.GPTQConfig(**kw)])
    _assert_nb_equal(q, m)
    plain, _ = _nb_quark(model, shape, tmp_path, mm)
    _assert_nb_equal(plain, q)


def test_matmul_nbits_shared_weight_matches_quark(tmp_path):
    model = _nb_model(
        "h = MatMul(x, w1)\n h2 = Relu(h)\n h3 = MatMul(h2, w2)\n y = MatMul(h3, w1)",
        [("w1", (64, 64)), ("w2", (64, 64))],
        "float[3,64] x",
    )
    q, data = _nb_quark(model, (3, 64), tmp_path, {"GroupSize": 32})
    m = _nb_mine(model, {"GroupSize": 32}, data=data)
    _assert_nb_equal(q, m)


def test_matmul_nbits_selection_matches_quark(tmp_path):
    """Gemm, 3-D-weight and non-constant MatMuls are left alone."""
    model = _nb_model(
        "a = MatMul(x, w)\n"
        "b = MatMul(a, w3)\n"
        "g = Gemm(a, gw)\n"
        "at = Transpose<perm=[1,0]>(a)\n"
        "c = MatMul(g, at)\n"
        "y = Add(b, c)",
        [("w", (64, 16)), ("w3", (2, 16, 3)), ("gw", (16, 16))],
        "float[3,64] x",
    )
    q, data = _nb_quark(model, (3, 64), tmp_path, {"GroupSize": 32})
    m = _nb_mine(model, {"GroupSize": 32}, data=data)
    assert [n.op_type for n in m.graph.node] == [n.op_type for n in q.graph.node]
    _assert_nb_equal(q, m)


def test_matmul_nbits_quark_default_fuses_matmul_add_into_gemm(tmp_path):
    """The documented difference: Quark's default pre-processing (ORT
    MatMulAddFusion) turns MatMul + Add into a Gemm that is not quantized;
    onnxsim converts the MatMul and keeps the Add. Everything else matches."""
    model = _nb_model(
        "h = MatMul(x, w1)\n h2 = Add(h, b)\n h3 = Relu(h2)\n y = MatMul(h3, w2)",
        [("w1", (64, 96)), ("b", (96,)), ("w2", (96, 8))],
        "float[3,64] x",
    )
    q, data = _nb_quark(model, (3, 64), tmp_path, {"GroupSize": 32}, skip=False)
    assert [n.op_type for n in q.graph.node] == ["Gemm", "Relu", "MatMulNBits"]
    cfg = qc.QConfig.get_default_config("MATMUL_NBITS")
    cfg.extra_options["MatMulNBitsParams"]["GroupSize"] = 32
    # onnxsim runs the same pre-processing now (ONNX Runtime's MatMulAddFusion)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=data)
    assert [n.op_type for n in m.graph.node] == ["Gemm", "Relu", "MatMulNBits"]
    for k in ("w2_Q4", "w2_scales"):
        np.testing.assert_array_equal(_nb_inits(m)[k], _nb_inits(q)[k])
    # without ONNX Runtime's optimizer the pair stays: the MatMul is converted, with
    # a note, and its last layer is Quark's, tensor for tensor
    cfg.extra_options["UseRuntimeOptimizers"] = False
    with pytest.warns(UserWarning, match="Gemm"):
        m = qc.ModelQuantizer(cfg).quantize_model(model, calibration_data_reader=data)
    assert [n.op_type for n in m.graph.node] == [
        "MatMulNBits",
        "Add",
        "Relu",
        "MatMulNBits",
    ]
    x = np.random.default_rng(7).standard_normal((3, 64)).astype(np.float32)
    ref = _run(model, x)
    assert np.linalg.norm(_run(m, x) - ref) / np.linalg.norm(ref) < 0.3


def test_matmul_nbits_bits_other_than_four_is_a_quark_bug(tmp_path):
    """Quark packs 4-bit codes whatever ``Bits`` says; onnxsim refuses."""
    model, shape = _NB_MODELS["mlp"]
    q, data = _nb_quark(model, shape, tmp_path, {"GroupSize": 32, "Bits": 8})
    assert _attrs(q.graph.node[0])["bits"] == 8
    assert _nb_inits(q)["w1_Q4"].shape == (96, 2, 16)  # still 4-bit sized
    with pytest.raises(NotImplementedError, match="4-bit"):
        _nb_mine(model, {"GroupSize": 32, "Bits": 8}, data=data)


# == Quark's calibrators, Q/DQ removal and quantizer-level options ================
#
# Entropy / Distribution / Percentile / LayerwisePercentile ranges, the
# RemoveQDQ* / FoldRelu / Align* / ActivationSymmetric / WeightSymmetric /
# QuantizeBias options: same activation Q/DQ placement and the same scales,
# zero points and constants as Quark, on parser-built models. Known, deliberate
# deviations are spelled out in the tests that touch them.

import copy as _copy  # noqa: E402


def _quark_reset_globals():
    """``RemoveQDQInstanceNorm`` appends to a module-level list Quark never
    clears; restore its default so one test cannot leak into the next."""
    import quark.onnx.quantization.quant_utils as qu

    qu.annotate_op_type[:] = [
        "Conv",
        "Add",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "MatMul",
        "Gemm",
        "ConvTranspose",
    ]


def _quark_cfg(preset, extra=None, method=None):
    """A private copy of Quark's preset (``get_default_config`` hands out a
    shared object whose options would otherwise leak between tests)."""
    from quark.onnx import QConfig

    cfg = _copy.deepcopy(QConfig.get_default_config(preset))
    g = cfg.global_quant_config
    g.include_cle = False
    if method is not None:
        g.calibrate_method = method
    g.extra_options.update(extra or {})
    return cfg


def _quark_run(model, preset, shape, tmp_path, extra=None, method=None, n=4, seed=3):
    from quark.onnx import ModelQuantizer

    _quark_reset_globals()
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ModelQuantizer(_quark_cfg(preset, extra, method)).quantize_model(
            src, dst, _reader(shape, n=n, seed=seed)()
        )
    return onnx.load(dst)


def _mine_run(model, preset, shape, extra=None, method=None, n=4, seed=3):
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    if method is not None:
        cfg.global_config.activation.calibration_method = method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(shape, n=n, seed=seed)()
        )


def _placement(model):
    """Graph structure around the activations: the compute / activation nodes
    in order, with every activation ``QuantizeLinear`` as ``(tensor, scale,
    zero_point, dtype)`` (weight / bias DequantizeLinear nodes left out)."""
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[1] in inits:
            zp = inits[n.input[2]]
            out.append(
                (
                    _norm_name(n.input[0]),
                    float(inits[n.input[1]]),
                    int(zp),
                    str(zp.dtype),
                )
            )
        elif n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            out.append(n.op_type)
    return out


def _assert_same_placement(q, m, rtol=1e-5, msg=""):
    pq, pm = _placement(q), _placement(m)
    assert [e if isinstance(e, str) else e[0] for e in pm] == [
        e if isinstance(e, str) else e[0] for e in pq
    ], (msg, pq, pm)
    for a, b in zip(pq, pm):
        if not isinstance(a, str):
            assert (b[2], b[3]) == (a[2], a[3]), (msg, a, b)
            np.testing.assert_allclose(b[1], a[1], rtol=rtol, err_msg=f"{msg} {a[0]}")


def _assert_same_constants(q, m, rtol=1e-5):
    _, qc_ = _qparam_map(q)
    _, mc_ = _qparam_map(m)
    assert set(qc_) == set(mc_)
    for name, (want, dt, scale) in qc_.items():
        got, mdt, _ = mc_[name]
        assert mdt == dt, name
        atol = 2 * scale if dt == "int32" else 1e-4
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=name)


def _assert_parity(q, m, rtol=1e-5, msg=""):
    _assert_same_placement(q, m, rtol, msg)
    _assert_same_constants(q, m, rtol)


# -- calibrators: the ranges themselves ---------------------------------------------

_CALIB_CASES = {
    "entropy": ("Entropy", "quark_entropy", {}, {}),
    "entropy_512": (
        "Entropy",
        "quark_entropy",
        {"NumBins": 512},
        {"quark_num_bins": 512},
    ),
    "distribution": ("Distribution", "quark_distribution", {}, {}),
    "distribution_1024": (
        "Distribution",
        "quark_distribution",
        {"NumBins": 1024},
        {"quark_num_bins": 1024},
    ),
    "percentile": ("Percentile", "quark_percentile", {}, {}),
    "percentile_asym": (
        "Percentile",
        "quark_percentile:99.99",
        {"CalibTensorRangeSymmetric": False, "Percentile": 99.99},
        {"range_symmetric": False},
    ),
    "lwp": ("LayerwisePercentile", "quark_layerwise_percentile", {}, {}),
    "lwp_mse": (
        "LayerwisePercentile",
        "quark_layerwise_percentile",
        {"LWPMetric": "mse", "PercentileCandidates": [99.9, 99.99, 99.999]},
        {"lwp_metric": "mse", "percentile_candidates": (99.9, 99.99, 99.999)},
    ),
}


def _quark_method(name):
    from onnxruntime.quantization import CalibrationMethod
    from quark.onnx.calibration.methods import LayerWiseMethod

    if name == "LayerwisePercentile":
        return LayerWiseMethod.LayerWisePercentile
    return getattr(CalibrationMethod, name)


def _heavy_reader(shape, n, seed, df):
    """Student-t calibration batches (heavy tails make the clipping choices
    of Entropy / LayerwisePercentile matter), as a Quark reader + a list."""
    from onnxruntime.quantization import CalibrationDataReader

    rng = np.random.default_rng(seed)
    data = [{"x": (rng.standard_t(df, shape) * 2).astype(np.float32)} for _ in range(n)]

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

    return R(), data


def _quark_ranges(model, reader, case):
    from quark.onnx.calibration import interface
    from quark.onnx.calibration.data_readers import CachedDataReader

    name, _, extra, _ = _CALIB_CASES[case]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        res = interface.run_calibration(
            model,
            CachedDataReader(reader, None),
            None,
            calibrate_method=_quark_method(name),
            extra_options=extra,
        )
    return {k: tuple(float(v) for v in res[k].range_value) for k in res.keys()}


def _compare_ranges(want, got):
    for k, (lo, hi) in want.items():
        scale = max(abs(lo), abs(hi), 1e-12)
        assert abs(got[k][0] - lo) / scale < 1e-6, (k, (lo, hi), got[k])
        assert abs(got[k][1] - hi) / scale < 1e-6, (k, (lo, hi), got[k])


@pytest.mark.parametrize("data_kind", ["normal", "heavy"])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
@pytest.mark.parametrize("case", sorted(_CALIB_CASES))
def test_quark_calibrator_ranges_match_quark(case, model_name, seed, data_kind):
    """Entropy, Distribution, Percentile and LayerwisePercentile compute the
    same ``(min, max)`` per tensor as Quark's calibrators -- bit for bit up to
    float32 rounding: the same histogram layout (growing bins), search and
    clipping."""
    from onnxsim.calibration import calibrate

    model, shape = CAL_MODELS[model_name](seed)
    model = onnx.shape_inference.infer_shapes(model)
    n = 8
    if data_kind == "normal":
        reader, data = _heavy_reader(shape, n, seed + 5, 1e9)  # ~ Gaussian
    else:
        reader, data = _heavy_reader(shape, n, seed + 5, 1.5)
    want = _quark_ranges(model, reader, case)
    _, method, _, kw = _CALIB_CASES[case]
    got = calibrate(
        model,
        data,
        method=method,
        tensor_names=list(want),
        activation_type="int8",
        **kw,
    )
    _compare_ranges(want, got)


def test_layerwise_percentile_really_differs_from_percentile():
    """The parity above is not vacuous: on heavy-tailed data LayerwisePercentile
    picks a different candidate than plain Percentile for many tensors, and the
    ranges still agree with Quark's."""
    from onnxsim.calibration import calibrate

    differing = 0
    for model_name in sorted(CAL_MODELS):
        model, shape = CAL_MODELS[model_name](0)
        model = onnx.shape_inference.infer_shapes(model)
        reader, data = _heavy_reader(shape, 8, 5, 1.5)
        lwp = _quark_ranges(model, reader, "lwp")
        reader, data = _heavy_reader(shape, 8, 5, 1.5)
        pct = _quark_ranges(model, reader, "percentile")
        differing += sum(lwp[k] != pct[k] for k in lwp)
        mine = calibrate(
            model,
            data,
            method="quark_layerwise_percentile",
            tensor_names=list(lwp),
            activation_type="int8",
        )
        _compare_ranges(lwp, mine)
    assert differing >= 4


# -- calibrators through the presets ----------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "A16W8"])
@pytest.mark.parametrize(
    "member", ["Entropy", "Distribution", "LayerwisePercentile", "Percentile"]
)
def test_calibrators_through_presets_match_quark_scale_for_scale(
    member, preset, model_name, tmp_path
):
    """``CalibMethod.<member>`` on a preset's activations: identical scale,
    zero point and dtype per tensor and identical weights / biases /
    constants. (Quark's Distribution reports the symmetric histogram extent
    even for a post-Relu tensor; with uint8 activations its Relu fold then
    sits on a centred grid -- reproduced, see the test below.)"""
    model, shape = CAL_MODELS[model_name](1)
    q = _quark_run(
        model, preset, shape, tmp_path, method=_quark_method(member), n=4, seed=3
    )
    m = _mine_run(model, preset, shape, method=qc.CalibMethod[member], n=4, seed=3)
    _assert_parity(q, m, rtol=1e-5, msg=f"{preset} {member} {model_name}")


def test_distribution_with_unsigned_relu_reproduces_quarks_centred_fold(tmp_path):
    """Quark: Distribution + uint8 + Relu folds the Relu node onto the
    symmetric ``(-T, T)`` grid (zero point 128), so negative pre-activations
    survive. onnxsim reproduces that graph -- and says so."""
    model, shape = CAL_MODELS["mlp"](1)
    q = _quark_run(
        model, "U8S8_AAWS", shape, tmp_path, method=_quark_method("Distribution")
    )
    cfg = qc.QConfig.get_default_config("U8S8_AAWS")
    cfg.global_config.activation.calibration_method = qc.CalibMethod.Distribution
    with pytest.warns(UserWarning, match="no longer clamps"):
        m = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=_reader(shape, n=4, seed=3)()
        )
    _assert_parity(q, m, msg="U8S8_AAWS Distribution")
    assert "Relu" not in [n.op_type for n in m.graph.node]
    assert _qparam_map(m)[0]["h1"][1] in (127, 128)


_CALIB_OPTION_CASES = [
    ("A8W8", "MinMax", {"CalibMovingAverage": True}),
    ("INT8_CNN_DEFAULT", "MinMax", {"CalibTensorRangeSymmetric": True}),
    ("U8S8_AAWS", "Percentile", {"CalibTensorRangeSymmetric": False}),
    ("U8S8_AAWS", "Percentile", {"Percentile": 99.9}),
    ("A8W8", "Percentile", {"Percentile": 99.99}),
    ("A8W8", "Entropy", {"NumBins": 512, "NumQuantizedBins": 128}),
    ("U8S8_AAWS", "Entropy", {"NumBins": 2048}),
    ("A8W8", "Distribution", {"NumBins": 1024}),
    ("A8W8", "LayerwisePercentile", {"LWPMetric": "mse"}),
    (
        "U8S8_AAWS",
        "LayerwisePercentile",
        {"PercentileCandidates": [99.9, 99.99, 99.999]},
    ),
    ("A8W8", "MinMax", {"CalibDataSize": 2}),
]


@pytest.mark.parametrize("model_name", ["mlp", "conv", "attn"])
@pytest.mark.parametrize("preset, member, extra", _CALIB_OPTION_CASES)
def test_calibration_extra_options_match_quark(
    preset, member, extra, model_name, tmp_path
):
    """CalibMovingAverage, CalibTensorRangeSymmetric, Percentile, NumBins,
    NumQuantizedBins, LWPMetric, PercentileCandidates and CalibDataSize."""
    model, shape = CAL_MODELS[model_name](0)
    if member == "MinMax" and "CalibMovingAverage" in extra:
        quark_member = _quark_method("MinMax")
    else:
        quark_member = _quark_method(member)
    q = _quark_run(
        model, preset, shape, tmp_path, extra=extra, method=quark_member, n=6, seed=7
    )
    m = _mine_run(
        model, preset, shape, extra=extra, method=qc.CalibMethod[member], n=6, seed=7
    )
    q_acts, _ = _qparam_map(q)
    m_acts, _ = _qparam_map(m)
    assert set(q_acts) == set(m_acts)
    for k, (scale, zp, dt) in q_acts.items():
        assert m_acts[k][1:] == (zp, dt), k
        np.testing.assert_allclose(m_acts[k][0], scale, rtol=1e-5, err_msg=k)


# -- Q/DQ removal around activations ----------------------------------------------------


def _removal_model(prod, cons):
    """``x -> producer -> consumer -> (Gemm | Conv)``: the consumer's output
    feeds a quantized node, so every placement choice is visible."""
    rng = np.random.default_rng(0)

    def w(name, *shape):
        return onnx.numpy_helper.from_array(
            (rng.standard_normal(shape) * 0.5).astype(np.float32), name
        )

    shapes = {
        "Conv": (1, 3, 8, 8),
        "ConvTranspose": (1, 3, 8, 8),
        "MaxPool": (1, 3, 8, 8),
        "AveragePool": (1, 3, 8, 8),
        "GlobalAveragePool": (1, 3, 8, 8),
        "InstanceNormalization": (1, 3, 8, 8),
        "Gemm": (3, 16),
        "MatMul": (3, 16),
        "Add": (3, 16),
    }
    producers = {
        "Conv": (
            "t = Conv<pads=[1,1,1,1]>(x, w, b)",
            [w("w", 4, 3, 3, 3), w("b", 4)],
            4,
        ),
        "ConvTranspose": (
            "t = ConvTranspose(x, w, b)",
            [w("w", 3, 4, 3, 3), w("b", 4)],
            4,
        ),
        "Gemm": ("t = Gemm(x, w, b)", [w("w", 16, 8), w("b", 8)], 8),
        "MatMul": ("t = MatMul(x, w)", [w("w", 16, 8)], 8),
        "Add": ("t = Add(x, w)", [w("w", 16)], 16),
        "MaxPool": ("t = MaxPool<kernel_shape=[2,2], strides=[2,2]>(x)", [], 3),
        "AveragePool": ("t = AveragePool<kernel_shape=[2,2], strides=[2,2]>(x)", [], 3),
        "GlobalAveragePool": ("t = GlobalAveragePool(x)", [], 3),
        "InstanceNormalization": (
            "t = InstanceNormalization(x, w, b)",
            [w("w", 3), w("b", 3)],
            3,
        ),
    }
    consumers = {
        "Relu": "r = Relu(t)",
        "LeakyRelu": "r = LeakyRelu<alpha=0.1>(t)",
        "Clip6": "r = Clip(t, lo, hi6)",
        "Clip1": "r = Clip(t, lo, hi1)",
        "ClipM": "r = Clip(t, lom, hi1)",
        "PRelu": "r = PRelu(t, sl)",
        "Gelu": "r = Gelu(t)",
    }
    pdef, inits, cin = producers[prod]
    twod = prod in ("Gemm", "MatMul", "Add")
    tail = "y = Gemm(r, w2, b2)" if twod else "y = Conv(r, w2, b2)"
    m = parser.parse_model(
        f"""<ir_version: 9, opset_import: ["": {20 if cons == "Gelu" else 17}]>
        g (float{list(shapes[prod])} x) => (float y) {{ {pdef}  {consumers[cons]}  {tail} }}"""
    )
    m.graph.initializer.extend(inits)
    for n, v in (("lo", 0.0), ("hi6", 6.0), ("hi1", 1.0), ("lom", -1.0)):
        m.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array(v, np.float32), n)
        )
    m.graph.initializer.append(w("sl", 1))
    m.graph.initializer.extend(
        [w("w2", cin, 5), w("b2", 5)] if twod else [w("w2", 2, cin, 1, 1), w("b2", 2)]
    )
    return onnx.shape_inference.infer_shapes(m), shapes[prod]


_PRODUCERS = [
    "Conv",
    "ConvTranspose",
    "Gemm",
    "MatMul",
    "Add",
    "MaxPool",
    "GlobalAveragePool",
    "InstanceNormalization",
]
_CONSUMERS = ["Relu", "LeakyRelu", "Clip6", "Clip1", "ClipM", "PRelu", "Gelu"]


@pytest.mark.parametrize("cons", _CONSUMERS)
@pytest.mark.parametrize("prod", _PRODUCERS)
@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS"])
def test_qdq_removal_between_producer_and_activation_matches_quark(
    preset, prod, cons, tmp_path
):
    """With the default options Quark drops the Q/DQ between a Conv / Add /
    MaxPool / AveragePool / GlobalAveragePool / MatMul / Gemm / ConvTranspose and
    a following Relu / LeakyRelu / PRelu / Clip(0, 6 | 0, 1) (not Gelu, not
    InstanceNormalization, not Clip(-1, 1)) and, for asymmetric activations
    under the plain quantizer, folds the Relu / Clip node itself. Same nodes,
    scales, zero points and constants (the Relu input takes the output's
    range where its Q/DQ stays)."""
    model, shape = _removal_model(prod, cons)
    q = _quark_run(model, preset, shape, tmp_path)
    m = _mine_run(model, preset, shape)
    _assert_parity(q, m, msg=f"{preset} {prod} {cons}")


@pytest.mark.parametrize("cons", ["Relu", "LeakyRelu", "Clip6", "PRelu"])
@pytest.mark.parametrize("prod", ["Conv", "Gemm", "InstanceNormalization"])
@pytest.mark.parametrize("preset", ["A16W8", "U16S8_AAWS", "S8S8_AAWS"])
def test_qdq_removal_matches_quark_for_the_other_quantizer_classes(
    preset, prod, cons, tmp_path
):
    """The extended (16-bit) quantizer folds a Relu node only under
    ``FoldRelu`` (so U16S8_AAWS keeps it); the plain one folds it for any
    asymmetric preset (S8S8_AAWS as well)."""
    model, shape = _removal_model(prod, cons)
    q = _quark_run(model, preset, shape, tmp_path)
    m = _mine_run(model, preset, shape)
    _assert_parity(q, m, msg=f"{preset} {prod} {cons}")


@pytest.mark.parametrize("prod", ["Conv", "Gemm", "Add"])
@pytest.mark.parametrize("cons", ["Relu", "LeakyRelu", "Clip6", "PRelu"])
def test_xint8_qdq_removal_matches_quark(prod, cons, tmp_path):
    model, shape = _removal_model(prod, cons)
    q = _quark_run(model, "XINT8", shape, tmp_path)
    m = _mine_run(model, "XINT8", shape)
    _assert_parity(q, m, rtol=0, msg=f"XINT8 {prod} {cons}")


_REMOVAL_OPTIONS = [
    ({"RemoveQDQConvRelu": False}, "Conv", "Relu"),
    ({"RemoveQDQConvRelu": False}, "Gemm", "Relu"),
    ({"RemoveQDQConvClip": False}, "Gemm", "Clip6"),
    ({"RemoveQDQConvClip": False}, "Conv", "Relu"),  # unrelated: Relu unchanged
    ({"RemoveQDQConvLeakyRelu": False}, "Conv", "LeakyRelu"),
    ({"RemoveQDQConvLeakyRelu": False}, "Conv", "Relu"),
    ({"RemoveQDQConvPRelu": False}, "Gemm", "PRelu"),
    ({"RemoveQDQConvGelu": True}, "Conv", "Gelu"),
    ({"RemoveQDQConvGelu": True}, "Gemm", "Gelu"),
    ({"RemoveQDQInstanceNorm": True}, "InstanceNormalization", "Relu"),
    ({"RemoveQDQInstanceNorm": True}, "InstanceNormalization", "LeakyRelu"),
    ({"RemoveQDQInstanceNorm": False}, "InstanceNormalization", "Relu"),
    ({"FoldRelu": True}, "Conv", "Relu"),
    ({"FoldRelu": False}, "Conv", "Relu"),
    ({"FoldRelu": True}, "Gemm", "Clip6"),
]


@pytest.mark.parametrize("extra, prod, cons", _REMOVAL_OPTIONS)
@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "U16S8_AAWS"])
def test_qdq_removal_options_match_quark(extra, prod, cons, preset, tmp_path):
    """Each ``RemoveQDQ*`` / ``FoldRelu`` option, off or on, against Quark's
    default (``RemoveQDQConvGelu`` and ``RemoveQDQInstanceNorm`` are opt-in)."""
    model, shape = _removal_model(prod, cons)
    q = _quark_run(model, preset, shape, tmp_path, extra=extra)
    m = _mine_run(model, preset, shape, extra=extra)
    _assert_parity(q, m, msg=f"{preset} {extra} {prod} {cons}")


def test_removal_options_change_the_graph():
    """... and are not vacuous: switching an option changes onnxsim's graph."""
    model, shape = _removal_model("Gemm", "LeakyRelu")
    default = _placement(_mine_run(model, "A8W8", shape))
    off = _placement(_mine_run(model, "A8W8", shape, {"RemoveQDQConvLeakyRelu": False}))
    assert len(off) == len(default) + 1  # the Q on the producer output stays
    model, shape = _removal_model("Conv", "Gelu")
    on = _placement(_mine_run(model, "A8W8", shape, {"RemoveQDQConvGelu": True}))
    assert len(on) == len(_placement(_mine_run(model, "A8W8", shape))) - 1


# -- ActivationSymmetric / WeightSymmetric / QuantizeBias ---------------------------------


@pytest.mark.parametrize("model_spec", [("Gemm", "Relu"), ("Conv", "PRelu")])
@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    "preset", ["A8W8", "U8S8_AAWS", "A16W8", "U16S8_AAWS", "S8S8_AAWS"]
)
def test_activation_symmetric_option_matches_quark(preset, value, model_spec, tmp_path):
    """``ActivationSymmetric`` overrides the preset's symmetry: signed types
    centre at 0, unsigned ones at 128 / 32768 (scale ``2 * absmax / 255``),
    and the Relu-folding rule follows it."""
    model, shape = _removal_model(*model_spec)
    extra = {"ActivationSymmetric": value}
    q = _quark_run(model, preset, shape, tmp_path, extra=extra)
    m = _mine_run(model, preset, shape, extra=extra)
    _assert_parity(q, m, msg=f"{preset} {extra}")


@pytest.mark.parametrize(
    "model_spec", [("Gemm", "Relu"), ("Conv", "Relu"), ("Add", "Relu")]
)
@pytest.mark.parametrize(
    "preset", ["A8W8", "U8U8_AAWA", "U8S8_AAWS", "A16W8", "S16S8_ASWS"]
)
def test_weight_symmetric_false_matches_quark(preset, model_spec, tmp_path):
    """Asymmetric weights (int8, int16 or uint8 per the preset), biases scaled
    ``input * weight`` and the constant operand of an Add."""
    model, shape = _removal_model(*model_spec)
    extra = {"WeightSymmetric": False}
    q = _quark_run(model, preset, shape, tmp_path, extra=extra)
    m = _mine_run(model, preset, shape, extra=extra)
    _assert_parity(q, m, msg=f"{preset} {model_spec}")


@pytest.mark.parametrize("model_name", sorted(CAL_MODELS))
def test_u8u8_aawa_uint8_weights_match_quark(model_name, tmp_path):
    """U8U8_AAWA's asymmetric uint8 weights (no longer an approximation)."""
    model, shape = CAL_MODELS[model_name](0)
    _check_cal_parity(model, shape, "U8U8_AAWA", tmp_path, rtol=1e-5, zp_atol=1)


@pytest.mark.parametrize("prod", ["Gemm", "Conv"])
@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "A16W8", "XINT8"])
def test_quantize_bias_false_matches_quark(preset, prod, tmp_path):
    model, shape = _removal_model(prod, "Relu")
    extra = {"QuantizeBias": False}
    q = _quark_run(model, preset, shape, tmp_path, extra=extra)
    m = _mine_run(model, preset, shape, extra=extra)
    _assert_parity(q, m, rtol=0 if preset == "XINT8" else 1e-5, msg=preset)
    assert not any(
        onnx.numpy_helper.to_array(i).dtype in (np.int32,) for i in m.graph.initializer
    )


# -- Align* options ---------------------------------------------------------------------


def _align_model(kind):
    """Models where an alignment is observable: its input / output ranges
    differ, so copying the parameters changes a scale."""
    consts = {
        "k3": np.array(3.0, np.float32),
        "st": np.array([0], np.int64),
        "en": np.array([2], np.int64),
        "ax": np.array([3], np.int64),
        "pd": np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64),
        "c5": np.array(5.0, np.float32),
        "shp": np.array([1, 3, 4, 16], np.int64),
    }
    bodies = {
        "Concat": "a = Sigmoid(x)  b = Tanh(x)  c0 = Mul(x, k3)  y = Concat<axis=3>(a, b, c0)",
        "Slice": "x1 = Mul(x, k3)  s = Slice(x1, st, en, ax)  y = Sigmoid(s)",
        "Pad": "x1 = Mul(x, k3)  s = Pad(x1, pd, c5)  y = Sigmoid(s)",
        "MaxPool": "x1 = Mul(x, k3)  s = MaxPool<kernel_shape=[2,2], strides=[2,2]>(x1)  y = Sigmoid(s)",
        "AveragePool": "x1 = Mul(x, k3)  s = AveragePool<kernel_shape=[2,2], strides=[2,2]>(x1)  y = Sigmoid(s)",
        "GlobalAveragePool": "x1 = Mul(x, k3)  s = GlobalAveragePool(x1)  y = Sigmoid(s)",
        "Transpose": "x1 = Mul(x, k3)  s = Transpose<perm=[0,1,3,2]>(x1)  y = Sigmoid(s)",
        "Reshape": "x1 = Mul(x, k3)  s = Reshape(x1, shp)  y = Sigmoid(s)",
    }
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float[1,3,8,8] x) => (float y) { '
        + bodies[kind]
        + " }"
    )
    for k, v in consts.items():
        m.graph.initializer.append(onnx.numpy_helper.from_array(v, k))
    return onnx.shape_inference.infer_shapes(m), (1, 3, 8, 8)


_ALIGN_OPTION = {
    "Concat": "AlignConcat",
    "Slice": "AlignSlice",
    "Pad": "AlignPad",
    "MaxPool": "AlignPool",
    "AveragePool": "AlignPool",
    "GlobalAveragePool": "AlignPool",
    "Transpose": "AlignTranspose",
    "Reshape": "AlignReshape",
}


@pytest.mark.parametrize("value", [None, True, False])
@pytest.mark.parametrize("kind", sorted(_ALIGN_OPTION))
@pytest.mark.parametrize(
    "preset", ["A16W8", "U16S8_AAWS", "A8W8", "U8S8_AAWS", "S8S8_AAWS"]
)
def test_align_options_match_quark(preset, kind, value, tmp_path):
    """``Align{Concat,Slice,Pad,Pool,Transpose,Reshape}``: Concat / Pad /
    Transpose / Reshape inputs take their output's quantization parameters,
    Pool / Slice outputs their input's -- but only the extended quantizer runs
    these passes (A8W8, A16W8, U16S8_AAWS; A8W8 sets AlignConcat itself). The
    plain quantizer's Slice is calibrated on its own, its AveragePool shares
    its input's parameters."""
    model, shape = _align_model(kind)
    extra = {} if value is None else {_ALIGN_OPTION[kind]: value}
    q = _quark_run(model, preset, shape, tmp_path, extra=extra)
    m = _mine_run(model, preset, shape, extra=extra)
    _assert_same_placement(q, m, msg=f"{preset} {kind} {extra}")


def test_align_concat_is_observable():
    """Not vacuous: aligning a Concat moves its inputs' scales onto the
    output's."""
    model, shape = _align_model("Concat")
    on = dict(
        _scales_by_tensor(_mine_run(model, "A16W8", shape, {"AlignConcat": True}))
    )
    off = dict(
        _scales_by_tensor(_mine_run(model, "A16W8", shape, {"AlignConcat": False}))
    )
    assert on["a"] == on["b"] == on["y"] and off["a"] != off["y"]


def _scales_by_tensor(model):
    return [(e[0], e[1]) for e in _placement(model) if not isinstance(e, str)]
