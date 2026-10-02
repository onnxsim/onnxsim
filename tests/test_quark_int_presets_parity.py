"""Parity of onnxsim's integer presets other than XINT8 against the real AMD Quark ONNX
package: ``A8W8``, ``A16W8``, ``VINT8``, ``U8S8_AAWS``, ``S8S8_AAWS``, ``U16S8_AAWS``,
``S16S8_ASWS``, ``INT8_CNN_DEFAULT`` and ``INT8_TRANSFORMER_DEFAULT``. Skipped unless
``quark.onnx`` is importable.

Each test runs Quark and onnxsim on the same parser-built graph and compares the
emitted graphs node by node (op types, attributes, scales, zero points, constants,
wiring) and, with ONNX Runtime's graph optimizations off, what they compute. The
helpers are the ones of ``test_quark_xint8_parity.py``.
"""

import random
import re
import warnings
import zlib

import numpy as np
import onnx
import pytest
import test_quark_xint8_parity as P  # noqa: E402  (also sets up the Quark imports)
from onnx import numpy_helper, parser

from onnxsim import quark_compat as qc  # noqa: E402

pytestmark = P.pytestmark

#: presets the random-graph sweeps cover (name -> shuffle the node order too?)
PRESETS = (
    "A8W8",
    "A16W8",
    "VINT8",
    "U8S8_AAWS",
    "S8S8_AAWS",
    "U16S8_AAWS",
    "S16S8_ASWS",
    "INT8_CNN_DEFAULT",
)

_run_in_tmp_dir = P._run_in_tmp_dir  # the autouse fixture (Quark writes scratch files)


def _quark_run(model, data, tmp_path, preset, extra=None, exclude=()):
    """Quark's preset (``PerChannel`` is an attribute of its quantization config, not
    an option; onnxsim reads it from ``extra_options``)."""
    import contextlib
    import copy
    import io

    from quark.onnx import ModelQuantizer, QConfig

    extra = dict(extra or {})
    cfg = copy.deepcopy(QConfig.get_default_config(preset))
    cfg.global_quant_config.include_cle = False
    cfg.global_quant_config.per_channel = bool(extra.pop("PerChannel", False))
    cfg.global_quant_config.nodes_to_exclude = list(exclude)
    cfg.global_quant_config.extra_options.update(extra)
    src, dst = str(tmp_path / "src.onnx"), str(tmp_path / "dst.onnx")
    onnx.save(model, src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        ModelQuantizer(cfg).quantize_model(src, dst, P._reader(data))
    return onnx.load(dst)


def _same(model, data, tmp_path, preset, extra=None, exclude=()):
    q = _quark_run(model, data, tmp_path, preset, extra, exclude)
    cfg = qc.QConfig.get_default_config(preset)
    cfg.extra_options.update(extra or {})
    cfg.exclude = list(exclude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = qc.ModelQuantizer(cfg).quantize_model(
            model, calibration_data_reader=P._reader(data)
        )
    return q, m


def _run_all(model, x):
    """Every output of ``model`` in ONNX Runtime with its graph optimizations off."""
    import os

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    lib = os.environ.get("QUARK_ONNX_OPS_LIB")
    if lib:
        so.register_custom_ops_library(lib)
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {"x": x})


def _assert_same_outputs(m, q, x):
    got, want = _run_all(m, x), _run_all(q, x)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-5)


def _check(model, data, tmp_path, preset, msg="", extra=None, exclude=()):
    q, m = _same(model, data, tmp_path, preset, extra, exclude)
    P._assert_same_graph(q, m, f"{preset} {msg}")
    _assert_same_outputs(m, q, data[0]["x"])
    return q, m


# -- a random-graph generator with a wider operator mix ----------------------------


def _random_graph_ext(seed, big=False, rich=False):
    """A random DAG on ``[1, 8, 8, 8]`` tensors: convolutions, activations (PRelu,
    HardSigmoid, HardSwish, Clip, Softmax, ...), element-wise ops, pools, Pad, Slice,
    Split, Transpose, Reshape, BatchNormalization and a Gemm tail. ``rich`` adds
    bias-less, dilated, transposed and weight-sharing convolutions and extra graph
    outputs."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    inits = {
        "lo": np.array(0.0, np.float32),
        "hi6": np.array(6.0, np.float32),
        "hi1": np.array(1.0, np.float32),
        "lom": np.array(-1.0, np.float32),
        "pads": np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64),
        "sh_flat": np.array([1, 8, 64], np.int64),
        "sh_back": np.array([1, 8, 8, 8], np.int64),
        "st0": np.array([0], np.int64),
        "en4": np.array([4], np.int64),
        "st4": np.array([4], np.int64),
        "en8": np.array([8], np.int64),
        "ax1": np.array([1], np.int64),
        "sp44": np.array([4, 4], np.int64),
    }
    lines, tensors, uses = [], ["x"], {"x": 0}
    counter = [0]

    def name(prefix="t"):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def pow2(lo=-5, hi=3):
        return np.float32(2.0 ** rng.randint(lo, hi))

    def const(shape, scale=0.4, lo=-5, hi=3):
        n = name("c")
        inits[n] = (nrng.standard_normal(shape) * scale * pow2(lo, hi)).astype(
            np.float32
        )
        return n

    def weight(*shape):
        return const(shape)

    def bias():
        return const((8,), 0.5, -7, 3)

    shared = {}

    def shared_w():
        if "w" not in shared:
            shared["w"] = weight(8, 8, 3, 3)
        return shared["w"]

    def shared_b():
        if "b" not in shared:
            shared["b"] = bias()
        return shared["b"]

    def pick():
        t = rng.choice(tensors[-4:] if rng.random() < 0.7 else tensors)
        uses[t] = uses.get(t, 0) + 1
        return t

    def emit(op, ins, attrs="", register=True):
        out = name()
        lines.append(f"{out} = {op}{attrs}({', '.join(ins)})")
        if register:
            tensors.append(out)
            uses[out] = 0
        return out

    ops = [
        "conv3",
        "conv1",
        "dwconv",
        *(["conv_nobias", "conv_shared", "convt", "dilconv"] if rich else []),
        "relu",
        "add",
        "sub",
        "mul",
        "min",
        "max",
        "concat",
        "maxpool",
        "avgpool",
        "gap",
        "sigmoid",
        "tanh",
        "swish",
        "leaky",
        "prelu",
        "hardsigmoid",
        "hardswish",
        "clip6",
        "clip1",
        "clip11",
        "softmax",
        "gap_mul",
        "mul_const",
        "add_const",
        "pad_conv",
        "slice_cat",
        "split_cat",
        "transpose",
        "reshape",
        "bn",
        "reducemean",
        "identity",
    ]
    for _ in range(rng.randint(9, 18) if big else rng.randint(5, 11)):
        k = rng.choice(ops)
        if k == "conv3":
            emit("Conv", [pick(), weight(8, 8, 3, 3), bias()], "<pads=[1,1,1,1]>")
        elif k == "conv1":
            emit("Conv", [pick(), weight(8, 8, 1, 1), bias()])
        elif k == "conv_nobias":
            emit("Conv", [pick(), weight(8, 8, 1, 1)])
        elif k == "conv_shared":
            emit("Conv", [pick(), shared_w(), shared_b()], "<pads=[1,1,1,1]>")
        elif k == "convt":
            emit(
                "ConvTranspose",
                [pick(), weight(8, 8, 3, 3), bias()],
                "<pads=[1,1,1,1]>",
            )
        elif k == "dilconv":
            emit(
                "Conv",
                [pick(), weight(8, 8, 3, 3), bias()],
                "<pads=[2,2,2,2], dilations=[2,2]>",
            )
        elif k == "dwconv":
            emit(
                "Conv",
                [pick(), weight(8, 1, 3, 3), bias()],
                "<pads=[1,1,1,1], group=8>",
            )
        elif k in ("relu", "sigmoid", "tanh", "identity"):
            emit(k.capitalize(), [pick()])
        elif k in ("add", "sub", "mul", "min", "max"):
            a = pick()
            others = [t for t in tensors if t != a]
            if not others:
                emit("Relu", [a])
                continue
            emit(k.capitalize(), [a, rng.choice(others)])
        elif k == "concat":
            ins = [pick() for _ in range(rng.choice((2, 2, 3)))]
            c = emit("Concat", ins, "<axis=1>", register=False)
            emit("Conv", [c, weight(8, 8 * len(ins), 1, 1), bias()])
        elif k == "maxpool":
            emit(
                "MaxPool", [pick()], "<kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]>"
            )
        elif k == "avgpool":
            emit(
                "AveragePool",
                [pick()],
                "<kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]>",
            )
        elif k == "gap":
            # (not a Sub: Quark's InstanceNormalization fusion chokes on x - mean(y))
            emit("Add", [pick(), emit("GlobalAveragePool", [pick()], register=False)])
        elif k == "swish":
            a = pick()
            emit("Mul", [a, emit("Sigmoid", [a], register=False)])
        elif k == "leaky":
            emit("LeakyRelu", [pick()], "<alpha=0.1>")
        elif k == "prelu":
            slope = const((8, 1, 1) if rng.random() < 0.5 else (1,), 0.3, -3, 0)
            emit("PRelu", [pick(), slope])
        elif k == "hardsigmoid":
            alpha = rng.choice(("0.1666667", "0.2"))
            emit("HardSigmoid", [pick()], f"<alpha={alpha}, beta=0.5>")
        elif k == "hardswish":
            emit("HardSwish", [pick()])
        elif k == "clip6":
            emit("Clip", [pick(), "lo", "hi6"])
        elif k == "clip1":
            emit("Clip", [pick(), "lo", "hi1"])
        elif k == "clip11":
            emit("Clip", [pick(), "lom", "hi1"])
        elif k == "softmax":
            emit("Softmax", [pick()], "<axis=1>")
        elif k == "gap_mul":
            a = pick()
            emit("Mul", [a, emit("GlobalAveragePool", [a], register=False)])
        elif k in ("mul_const", "add_const"):
            emit("Mul" if k == "mul_const" else "Add", [pick(), const((1, 8, 1, 1))])
        elif k == "pad_conv":
            p = emit("Pad", [pick(), "pads"], register=False)
            emit("Conv", [p, weight(8, 8, 3, 3), bias()])
        elif k == "slice_cat":
            a = pick()
            s0 = emit("Slice", [a, "st0", "en4", "ax1"], register=False)
            s1 = emit("Slice", [a, "st4", "en8", "ax1"], register=False)
            c = emit("Concat", [s1, s0], "<axis=1>", register=False)
            emit("Conv", [c, weight(8, 8, 1, 1), bias()])
        elif k == "split_cat":
            a = pick()
            s0, s1 = name(), name()
            lines.append(f"{s0}, {s1} = Split<axis=1>({a}, sp44)")
            c = emit("Concat", [s1, s0], "<axis=1>", register=False)
            emit("Conv", [c, weight(8, 8, 1, 1), bias()])
        elif k == "transpose":
            emit("Transpose", [pick()], "<perm=[0,1,3,2]>")
        elif k == "reshape":
            r = emit("Reshape", [pick(), "sh_flat"], register=False)
            emit("Reshape", [r, "sh_back"])
        elif k == "bn":
            b = [name("bn") for _ in range(4)]
            inits[b[0]] = (1 + nrng.standard_normal(8) * 0.3).astype(np.float32)
            inits[b[1]] = (nrng.standard_normal(8) * 0.3).astype(np.float32)
            inits[b[2]] = (nrng.standard_normal(8) * 0.3).astype(np.float32)
            inits[b[3]] = (1 + np.abs(nrng.standard_normal(8))).astype(np.float32)
            c = emit("Conv", [pick(), weight(8, 8, 1, 1), bias()], register=False)
            emit("BatchNormalization", [c] + b)
        elif k == "reducemean":
            r = emit("ReduceMean", [pick()], "<axes=[2,3], keepdims=1>", register=False)
            emit("Add", [pick(), r])
    leaves = [t for t in tensors[1:] if uses.get(t, 0) == 0]
    cur = leaves[0]
    for leaf in leaves[1:]:
        if rng.random() < 0.5:
            cur = emit("Add", [cur, leaf])
        else:
            cat = emit("Concat", [cur, leaf], "<axis=1>", register=False)
            cur = emit("Conv", [cat, weight(8, 16, 1, 1), bias()])
    if rng.random() < 0.3:
        g = emit("GlobalAveragePool", [cur], register=False)
        f = emit("Flatten", [g], register=False)
        lines.append(f"y = Gemm({f}, wg, bg)")
        inits["wg"] = (nrng.standard_normal((8, 4)) * 0.4).astype(np.float32)
        inits["bg"] = (nrng.standard_normal(4) * 0.4).astype(np.float32)
    else:
        lines.append(f"y = Conv({cur}, wf, bf)")
        inits["wf"] = (nrng.standard_normal((4, 8, 1, 1)) * 0.4 * pow2()).astype(
            np.float32
        )
        inits["bf"] = (nrng.standard_normal(4) * 0.5 * pow2()).astype(np.float32)
    extra_outs = []
    if rich:
        # a few intermediate tensors are graph outputs too
        cands = [t for t in tensors[1:] if t != cur]
        extra_outs = rng.sample(cands, min(len(cands), rng.randint(0, 2)))
    outs = ", ".join(["float y"] + [f"float[1,8,8,8] {t}" for t in extra_outs])
    m = parser.parse_model(
        f'<ir_version: 9, opset_import: ["": 17]> g (float[1,8,8,8] x) => ({outs}) {{'
        + "\n".join(lines)
        + "}"
    )
    words = set(re.findall(r"\w+", "\n".join(lines)))
    m.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in inits.items() if k in words
    )
    return P._named(m), (1, 8, 8, 8)


# -- transformer-style graphs (INT8_TRANSFORMER_DEFAULT) -------------------------------


def _random_gemm_graph(seed):
    """A random chain/DAG of Gemm, weight MatMul (+ Add bias), activation MatMul,
    LayerNormalization, Softmax, Relu, residual Add, Mul and Transpose on ``[4, 16]``
    tensors."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    inits = {}
    lines, tensors, counter = [], ["x"], [0]

    def name(p="t"):
        counter[0] += 1
        return f"{p}{counter[0]}"

    def const(*shape, scale=0.4):
        n = name("c")
        inits[n] = (
            nrng.standard_normal(shape) * scale * 2.0 ** rng.randint(-3, 2)
        ).astype(np.float32)
        return n

    def emit(op, ins, attrs="", register=True):
        out = name()
        lines.append(f"{out} = {op}{attrs}({', '.join(ins)})")
        if register:
            tensors.append(out)
        return out

    def pick():
        return rng.choice(tensors[-3:] if rng.random() < 0.7 else tensors)

    for _ in range(rng.randint(4, 10)):
        k = rng.choice(
            [
                "gemm",
                "gemm_t",
                "matmul",
                "matmul_add",
                "matmul_act",
                "relu",
                "add",
                "mul",
                "softmax",
                "layernorm",
                "transpose2",
                "sigmoid",
                "gemm_nobias",
                "gemm_alpha",
                "matmul_left",
                "gemm_shared",
            ]
        )
        if k == "gemm_nobias":
            emit("Gemm", [pick(), const(16, 16)])
        elif k == "gemm_alpha":
            emit(
                "Gemm",
                [pick(), const(16, 16), const(16)],
                "<alpha=0.5, beta=2.0>",
            )
        elif k == "matmul_left":
            # a constant first operand: [4, 4] x [4, 16]
            emit("MatMul", [const(4, 4), pick()])
        elif k == "gemm_shared":
            if "ws" not in inits:
                inits["ws"] = (nrng.standard_normal((16, 16)) * 0.4).astype(np.float32)
            emit("Gemm", [pick(), "ws"])
        elif k == "gemm":
            emit("Gemm", [pick(), const(16, 16), const(16)])
        elif k == "gemm_t":
            emit("Gemm", [pick(), const(16, 16), const(16)], "<transB=1>")
        elif k == "matmul":
            emit("MatMul", [pick(), const(16, 16)])
        elif k == "matmul_add":
            emit(
                "Add",
                [emit("MatMul", [pick(), const(16, 16)], register=False), const(16)],
            )
        elif k == "matmul_act":
            a, b, c = pick(), pick(), pick()
            bt = emit("Transpose", [b], "<perm=[1,0]>", register=False)
            s = emit("MatMul", [a, bt], register=False)
            emit("MatMul", [emit("Softmax", [s], "<axis=-1>", register=False), c])
        elif k == "relu":
            emit("Relu", [pick()])
        elif k == "sigmoid":
            emit("Sigmoid", [pick()])
        elif k == "add":
            emit("Add", [pick(), pick()])
        elif k == "mul":
            emit("Mul", [pick(), const(16)])
        elif k == "softmax":
            emit("Softmax", [pick()], "<axis=-1>")
        elif k == "layernorm":
            emit(
                "LayerNormalization",
                [pick(), const(16, scale=1.0), const(16)],
                "<axis=-1>",
            )
        elif k == "transpose2":
            t1 = emit("Transpose", [pick()], "<perm=[1,0]>", register=False)
            emit("Transpose", [t1], "<perm=[1,0]>")
    lines.append(f"y = Gemm({tensors[-1]}, wy, by)")
    inits["wy"] = (nrng.standard_normal((16, 4)) * 0.4).astype(np.float32)
    inits["by"] = (nrng.standard_normal(4) * 0.4).astype(np.float32)
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float[4,16] x) => (float y) {'
        + "\n".join(lines)
        + "}"
    )
    words = set(re.findall(r"\w+", "\n".join(lines)))
    m.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in inits.items() if k in words
    )
    return P._named(m), (4, 16)


# -- small graphs: one test per class of difference found -------------------------------

_POOL = "MaxPool<kernel_shape=[3,3],pads=[1,1,1,1],strides=[1,1]>"


def _named_init(name, scale=1.0):
    """A deterministic initializer for ``w<k>`` (1x1 weight), ``u<k>`` (3x3 weight)
    and ``b<k>`` (bias) on 8 channels."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    shape = {"w": (8, 8, 1, 1), "u": (8, 8, 3, 3), "b": (8,)}[name[0]]
    return (
        rng.standard_normal(shape) * (0.5 if name[0] == "b" else 0.4) * scale
    ).astype(np.float32)


def _model8(body, outputs=("y",), **over):
    """A parser-built model on ``[1, 8, 8, 8]``. ``w<k>`` / ``u<k>`` / ``b<k>`` names
    in the body get a deterministic initializer (``over[name]`` is a factor to scale
    it by, or the array itself); the usual constants (``lo``, ``hi6``, ...) too."""
    consts = dict(
        lo=np.array(0.0, np.float32),
        hi6=np.array(6.0, np.float32),
        sp44=np.array([4, 4], np.int64),
        sl8=np.linspace(0.05, 0.4, 8, dtype=np.float32).reshape(8, 1, 1),
        sl1=np.array([0.25], np.float32),
    )
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float[1,8,8,8] x) => ('
        + ", ".join(f"float {o}" for o in outputs)
        + ") {"
        + body
        + "}"
    )
    for name in dict.fromkeys(re.findall(r"\b[wub]\d+\b|\b[a-z]\w*\b", body)):
        if name in over and not np.isscalar(over[name]):
            value = over[name]
        elif re.fullmatch(r"[wub]\d+", name):
            value = _named_init(name, over.get(name, 1.0))
        elif name in consts:
            value = consts[name]
        elif name in over:
            value = over[name]
        else:
            continue
        m.graph.initializer.append(numpy_helper.from_array(value, name))
    return P._named(m)


def _q_scales(model):
    """``{tensor read by a QuantizeLinear: (scale, zero point)}`` (the float tensor's
    own name, without the quantizers' suffixes)."""
    vals = P._values(model)
    out = {}
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear":
            t = n.input[0]
            for suffix in ("/f", "_QuantizeLinear_Input"):
                t = t[: -len(suffix)] if t.endswith(suffix) else t
            out[t] = (float(vals[n.input[1]]), int(vals[n.input[2]]))
    return out


def _both(
    body, preset, tmp_path, extra=None, batches=4, outputs=("y",), exclude=(), **over
):
    model = _model8(body, outputs, **over)
    data = P._data((1, 8, 8, 8), n=batches)
    q, m = _same(model, data, tmp_path, preset, extra, exclude)
    P._assert_same_graph(q, m, f"{preset}")
    _assert_same_outputs(m, q, data[0]["x"])
    return q, m


_ALIGNED = f"""
    c0 = Conv(x, w1, b1)
    c1 = Conv(x, w2, b2)
    p = {_POOL}(c0)
    k = Concat<axis=1>(c1, c0)
    f = Conv(k, w3, b3)
    e = Conv(p, w4, b4)
    d = Conv(c0, w5, b5)
    fe = Add(f, e)
    s = Relu(fe)
    sd = Add(s, d)
    y = Conv(sd, w6, b6)
"""


_ALIGNED_INITS = dict(w2=4.0, w3=np.tile(_named_init("w3"), (1, 2, 1, 1)))


@pytest.mark.parametrize("preset", ["A8W8", "A16W8"])
def test_a_pool_output_follows_the_concat_alignment_of_its_input(preset, tmp_path):
    """The extended quantizer's ``AlignConcat`` rewrites a Concat input's scale and
    zero-point initializers *in place*, and a MaxPool output shares its input's
    initializers: it moves with them. (``A16W8`` keeps the pool's own range: the
    pool output feeds no element-wise op here, so nothing overrides the sharing.)"""
    q, m = _both(_ALIGNED, preset, tmp_path, **_ALIGNED_INITS)
    for model in (q, m):
        scales = _q_scales(model)
        assert scales["c0"] == scales["p"] == scales["k"]


@pytest.mark.parametrize("preset", ["A8W8", "A16W8"])
def test_the_int32_bias_is_requantized_when_the_input_scale_moves(preset, tmp_path):
    """Quark's ``adjust_bias_scale`` divides the int32 bias codes by the ratio of the
    new to the old bias scale and *truncates* them after the Concat alignment moved
    the scale of the Conv's input."""
    q, m = _both(_ALIGNED, preset, tmp_path, **_ALIGNED_INITS)
    # (the same flow without the alignment rounds the codes at the original scale)
    _, plain = _both(
        _ALIGNED, preset, tmp_path, extra={"AlignConcat": False}, **_ALIGNED_INITS
    )
    codes = lambda model: sorted(  # noqa: E731
        tuple(c.tolist()) for c in _bias_codes(model)
    )
    assert codes(m) == codes(q) and codes(m) != codes(plain)


def _bias_codes(model):
    inits = P._inits(model)
    return [
        inits[n.input[0]]
        for n in model.graph.node
        if n.op_type == "DequantizeLinear"
        and n.input[0] in inits
        and inits[n.input[0]].dtype == np.int32
    ]


@pytest.mark.parametrize("preset", ["A8W8", "S8S8_AAWS"])
def test_an_int32_bias_beyond_the_int32_range_saturates_at_int32_min(preset, tmp_path):
    """A bias this large for its scale clips to ``[-2**31, 2**31 - 1]`` (not to
    ``-2**31 + 1``)."""
    tiny = _named_init("w1", 1e-7)
    big = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n y = Conv(r, w2, b2)",
        preset,
        tmp_path,
        w1=tiny,
        b1=big,
    )
    codes = np.concatenate([c.ravel() for c in _bias_codes(m)])
    assert codes.min() == -(2**31) and codes.max() == 2**31 - 1


@pytest.mark.parametrize("preset", ["S8S8_AAWS", "INT8_CNN_DEFAULT", "U8S8_AAWS"])
def test_an_all_zero_activation_gets_scale_one_and_zero_point_zero(preset, tmp_path):
    """A Conv whose output is negative everywhere, behind a Relu: the folded
    activation's range is ``[0, 0]``, which Quark quantizes with scale 1 and zero
    point 0 whatever the integer type (an int8 asymmetric grid is not at ``-128``)."""
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n y = Conv(r, w2, b2)",
        preset,
        tmp_path,
        b1=np.full(8, -60.0, np.float32),
    )
    for model in (q, m):
        assert (1.0, 0) in _q_scales(model).values()


@pytest.mark.parametrize("preset", ["U8S8_AAWS", "INT8_CNN_DEFAULT", "U16S8_AAWS"])
def test_two_clips_in_a_row_fold_into_the_producer(preset, tmp_path):
    """``Conv -> Clip(0, 6) -> Clip(0, 6)``: both Clips fold into the Conv's output
    (onnxsim once renamed the first producer's output twice and left the
    consumer's input dangling)."""
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n c = Clip(c0, lo, hi6)\n d = Clip(c, lo, hi6)\n"
        " y = Conv(d, w2, b2)",
        preset,
        tmp_path,
    )
    ops = lambda model: [n.op_type for n in model.graph.node]  # noqa: E731
    assert ops(m).count("Clip") == ops(q).count("Clip")
    # (the 16-bit quantizer folds an activation only under FoldRelu)
    assert (preset == "U16S8_AAWS") == ("Clip" in ops(q))


@pytest.mark.parametrize("preset", ["A16W8"])
def test_eltwise_inputs_do_not_share_parameters_under_align_eltwise_quant_type(
    preset, tmp_path
):
    """``AlignEltwiseQuantType`` puts a tensor-quantization override on every input of
    an Add / Mul / ..., and a tensor with an override (or whose provider has one)
    is quantized with its own parameters: the MaxPool output here does not reuse
    its input's."""
    q, m = _both(
        f"c0 = Conv(x, w1, b1)\n p = {_POOL}(c0)\n a = Add(p, c0)\n y = Conv(a, w2, b2)",
        preset,
        tmp_path,
        b1=np.full(8, -3.0, np.float32),
    )
    for model in (q, m):
        scales = _q_scales(model)
        assert scales["p"] != scales["c0"]


_VINT8_X = """
    l = LeakyRelu<alpha=0.1>(x)
    a, b = Split<axis=1>(x, sp44)
    c = Concat<axis=1>(b, a)
    d = Conv(c, w1, b1)
    m = Min(x, d)
    ml = Add(m, l)
    y = Conv(ml, w2, b2)
"""


def test_vint8_slices_made_from_a_split_read_the_float_graph_input(tmp_path):
    """VINT8 quantizes the op types of the model it is handed (plus the registry),
    so the Slices its ``ConvertSplitToSlice`` makes are not quantized; the graph
    input has two quantized readers, so it gets one Q/DQ pair for each of them (the
    ``DedicatedQDQPair`` option) and the Slices read it as it is."""
    q, m = _both(_VINT8_X, "VINT8", tmp_path)
    for model in (q, m):
        slices = [n for n in model.graph.node if n.op_type == "Slice"]
        assert len(slices) == 2 and all(s.input[0] == "x" for s in slices)
        assert sum(1 for n in model.graph.node if n.op_type == "QuantizeLinear") >= 3


def test_vint8_gives_every_input_slot_of_a_reader_a_pair(tmp_path):
    """A node reading a tensor twice counts twice for ``DedicatedQDQPair``: it takes
    the first pair and the second one is left without a reader."""
    q, m = _both(
        "k = Concat<axis=1>(x, x)\n s = Sigmoid(x)\n d = Conv(k, wk, bk)\n"
        " ds = Add(d, s)\n y = Conv(ds, w2, b2)",
        "VINT8",
        tmp_path,
        wk=_named_init("w1").repeat(2, axis=1),
        bk=_named_init("b1"),
    )
    for model in (q, m):
        pairs = [
            n
            for n in model.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0] == "x"
        ]
        assert len(pairs) == 3


def test_vint8_graph_output_read_by_several_nodes_stays_float(tmp_path):
    """A graph output that two quantized nodes read: each reader gets its own Q/DQ
    pair, the graph output itself is the float tensor (not a DQ)."""
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n r = Relu(c0)\n y = Add(s, r)",
        "VINT8",
        tmp_path,
        outputs=("y", "c0"),
    )
    for model in (q, m):
        by_out = {o: n for n in model.graph.node for o in n.output}
        assert by_out["c0"].op_type == "Conv"
        quantizers = [
            n
            for n in model.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0] == "c0"
        ]
        assert len(quantizers) == 2


@pytest.mark.parametrize("preset", ["VINT8", "U8S8_AAWS", "S8S8_AAWS"])
def test_a_relu_clip_chain_propagates_the_range_two_steps(preset, tmp_path):
    """Quark runs ONNX Runtime's ``adjust_tensor_ranges`` twice, so the input of a
    ``Relu -> Clip(0, 6)`` pair takes the *Clip* output's range, not the Relu's."""
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n r = Relu(c0)\n k = Clip(r, lo, hi6)\n y = Conv(k, w2, b2)",
        preset,
        tmp_path,
        extra={
            "RemoveQDQConvRelu": False,
            "RemoveQDQConvClip": False,
            "FoldRelu": False,
        },
        w1=6.0,
        b1=np.full(8, 2.0, np.float32),
    )
    if preset == "VINT8":  # (the others fold or drop the Q/DQ pairs in between)
        for model in (q, m):
            scales = _q_scales(model)
            assert scales["c0"] == scales["k"]


@pytest.mark.parametrize("preset", ["A16W8", "A8W8"])
def test_asymmetric_eltwise_constants_keep_to_the_symmetric_code_range(
    preset, tmp_path
):
    """With ``WeightSymmetric=False`` the constants an eltwise op reads (quantized
    like weights, in the activation type) are asymmetric: the smallest element maps
    to ``-qmax``, not to the type's ``-qmax - 1`` (Quark clips them like its
    weights)."""
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n m = Mul(s, cm)\n y = Conv(m, w2, b2)",
        preset,
        tmp_path,
        extra={"WeightSymmetric": False, "AlignEltwiseQuantType": True},
        cm=np.linspace(-1.0, 3.0, 8, dtype=np.float32).reshape(1, 8, 1, 1),
    )
    inits = P._inits(q)
    by_out = {o: n for n in q.graph.node for o in n.output}
    mul = next(n for n in q.graph.node if n.op_type == "Mul")
    codes = inits[by_out[mul.input[1]].input[0]]
    assert codes.min() == -np.iinfo(codes.dtype).max


@pytest.mark.parametrize("preset", ["U16S8_AAWS", "S8S8_AAWS"])
def test_eltwise_constants_are_symmetric_whatever_the_activation_symmetry(
    preset, tmp_path
):
    """... and the constants of an eltwise op follow ``WeightSymmetric``, not the
    (asymmetric) activations'; ``AlignEltwiseQuantType`` acts under the extended
    quantizer only (``U16S8_AAWS``), elsewhere Quark warns and ignores it."""
    _both(
        "c0 = Conv(x, w1, b1)\n s = Sigmoid(c0)\n m = Mul(s, cm)\n y = Conv(m, w2, b2)",
        preset,
        tmp_path,
        extra={"AlignEltwiseQuantType": True},
        cm=np.linspace(-1.0, 3.0, 8, dtype=np.float32).reshape(1, 8, 1, 1),
    )


@pytest.mark.parametrize("preset", ["A8W8", "U8S8_AAWS", "A16W8"])
@pytest.mark.parametrize(
    "extra",
    [
        {"Int32Bias": False, "PerChannel": True},
        {"Int32Bias": False, "WeightSymmetric": False},
    ],
    ids=["per_channel", "asymmetric"],
)
def test_int8_biases_follow_per_channel_and_weight_symmetry(preset, extra, tmp_path):
    """``Int32Bias=False``: the bias is quantized like a weight -- one scale per
    element with ``PerChannel``, the asymmetric grid with ``WeightSymmetric=False``."""
    _both(
        "c0 = Conv(x, w1, b1)\n y = Conv(c0, w2, b2)",
        preset,
        tmp_path,
        extra=extra,
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"PerChannel": True},
        {"WeightSymmetric": False},
        {"PerChannel": True, "WeightSymmetric": False},
        {"ActivationSymmetric": False},
    ],
    ids=[
        "per_channel",
        "asymmetric",
        "per_channel_asymmetric",
        "asymmetric_activations",
    ],
)
def test_vint8_int8_biases_and_weights_follow_per_channel_and_weight_symmetry(
    extra, tmp_path
):
    """The power-of-two flavour: a per-channel int8 bias takes one MinMSE scale per
    element; asymmetric weights and biases take the zero point of the min / max
    scale at the nearest position and the best scale around it at that zero point."""
    _both(
        "c0 = Conv(x, w1, b1)\n y = Conv(c0, w2, b2)",
        "VINT8",
        tmp_path,
        extra=extra,
        w1=1.5,
        b1=1.5,
    )


def test_a_plain_quantizer_quantizes_a_prelu_slope_per_tensor_even_per_channel(
    tmp_path,
):
    """Only the extended quantizer's ``QDQPRelu`` goes per row; the default operator
    quantizer ``QuantizeAllOpTypes`` hands a PRelu to quantizes the slope per tensor."""
    _both(
        "c0 = Conv(x, w1, b1)\n r = PRelu(c0, sl8)\n y = Conv(r, w2, b2)",
        "S8S8_AAWS",
        tmp_path,
        extra={"PerChannel": True, "QuantizeAllOpTypes": True},
    )


def test_per_channel_quantizes_a_prelu_slope_per_row_unless_power_of_two(tmp_path):
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n r = PRelu(c0, sl8)\n y = Conv(r, w2, b2)",
        "A8W8",
        tmp_path,
        extra={"PerChannel": True},
    )
    for model in (q, m):
        by_out = {o: n for n in model.graph.node for o in n.output}
        prelu = next(n for n in model.graph.node if n.op_type == "PRelu")
        slope = by_out[prelu.input[1]]
        assert [a.i for a in slope.attribute if a.name == "axis"] == [0]
    _both(
        "c0 = Conv(x, w1, b1)\n r = PRelu(c0, sl8)\n y = Conv(r, w2, b2)",
        "VINT8",
        tmp_path,
        extra={"PerChannel": True},
    )


@pytest.mark.parametrize("preset", ["A8W8", "A16W8"])
def test_ops_without_a_quantizer_of_their_own_calibrate_their_output_alone(
    preset, tmp_path
):
    """Flatten (not in Quark's registries; ``QuantizeAllOpTypes`` hands it to the
    default operator quantizer) keeps its own range: a pool alignment that moves its
    input's parameters leaves its output's alone."""
    _both(
        "c0 = Conv(x, w1, b1)\n g = GlobalAveragePool(c0)\n f = Flatten(g)\n"
        " y = Gemm(f, wg, bg)",
        preset,
        tmp_path,
        extra={"AlignPool": True, "QuantizeAllOpTypes": True},
        wg=np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4),
        bg=np.linspace(-0.5, 0.5, 4, dtype=np.float32),
    )


@pytest.mark.parametrize("preset", ["A8W8", "S8S8_AAWS", "VINT8"])
def test_an_excluded_node_between_quantized_ones_is_wrapped_not_kept_float(
    preset, tmp_path
):
    """The quantizer only skips marking an excluded node; its quantized neighbours
    still put Q/DQ pairs around it (the next Conv is quantized in full)."""
    q, m = _both(
        "c0 = Conv(x, w1, b1)\n c1 = Conv(c0, w2, b2)\n y = Conv(c1, w3, b3)",
        preset,
        tmp_path,
        exclude=["n1_Conv"],
    )
    for model in (q, m):
        scales = _q_scales(model)
        assert {"c0", "c1"} <= set(scales)


def test_a_matmul_without_a_constant_b_is_excluded_from_the_transformer_scheme(
    tmp_path,
):
    """... which is how a ``MatMul`` with a constant *first* operand (not a weight)
    is left out of ``INT8_TRANSFORMER_DEFAULT``: the Gemm behind it is quantized."""
    model = _gemm_model("g = Gemm(x, w1, b1)\n t = MatMul(a1, g)\n y = Gemm(t, w3, b1)")
    data = P._data((4, 16))
    q, m = _same(model, data, tmp_path, "INT8_TRANSFORMER_DEFAULT")
    P._assert_same_graph(q, m, "constant-A MatMul")
    for graph in (q, m):
        assert "t" in _q_scales(graph)


def test_vint8_layer_normalization_needs_a_marked_input(tmp_path):
    """Like the data-movement ops, ``QDQLayerNorm`` with ``ForceQuantizeNoInputCheck``
    off (VINT8) quantizes nothing -- not even its scale and bias -- when its input
    is unmarked (here the graph input)."""
    model = _gemm_model("y = LayerNormalization<axis=-1>(x, s1, b1)")
    data = P._data((4, 16))
    q, m = _same(model, data, tmp_path, "VINT8")
    P._assert_same_graph(q, m, "LayerNormalization on the graph input")
    for graph in (q, m):
        ln = next(n for n in graph.graph.node if n.op_type == "LayerNormalization")
        assert ln.input[0] == "x"


@pytest.mark.parametrize("slope", ["sl8", "sl1"])
def test_vint8_quantizes_the_prelu_slope(slope, tmp_path):
    """With ``QuantizeAllOpTypes`` a PRelu reaches the plain quantizer, which
    quantizes all its inputs: the slope becomes an int8 constant like a weight."""
    q, m = _both(
        f"c0 = Conv(x, w1, b1)\n r = PRelu(c0, {slope})\n y = Conv(r, w2, b2)",
        "VINT8",
        tmp_path,
    )
    for model in (q, m):
        by_out = {o: n for n in model.graph.node for o in n.output}
        prelu = next(n for n in model.graph.node if n.op_type == "PRelu")
        assert by_out[prelu.input[1]].op_type == "DequantizeLinear"


def test_vint8_marks_the_input_of_a_hard_sigmoid_with_other_constants(tmp_path):
    """``QDQHardSigmoid`` (NPU CNN registry: the extended quantizer) marks nothing for
    a HardSigmoid that is not ``alpha = 1/6``, ``beta = 0.5``; VINT8's plain quantizer
    has no special case, so the Relu output it reads is quantized (the Relu itself,
    on the unmarked graph input, is not)."""
    q, m = _both(
        "r = Relu(x)\n h = HardSigmoid<alpha=0.2, beta=0.5>(r)\n y = Conv(h, w1, b1)",
        "VINT8",
        tmp_path,
    )
    for model in (q, m):
        assert "r" in _q_scales(model) and "x" not in _q_scales(model)


def test_vint8_leaves_a_relu_on_an_unmarked_input_alone(tmp_path):
    """The VINT8 preset does not set ``ForceQuantizeNoInputCheck``: a Relu / MaxPool
    reading the (unmarked) graph input is not quantized."""
    q, m = _both(
        f"r = Relu(x)\n p = {_POOL}(x)\n c = Conv(r, w1, b1)\n d = Conv(p, w2, b2)\n"
        " cd = Add(c, d)\n y = Conv(cd, w3, b3)",
        "VINT8",
        tmp_path,
    )
    for model in (q, m):
        ops = {n.op_type: n for n in model.graph.node}
        assert ops["Relu"].input[0] == ops["MaxPool"].input[0] == "x"


@pytest.mark.parametrize("batches", [3, 5, 9])
def test_the_transformer_preset_averages_the_batch_extremes_like_numpy(
    batches, tmp_path
):
    """``CalibMovingAverage``: the mean over batches of every batch's min / max, in
    float32 (``np.nanmean``), which is not the float64 mean of the same numbers."""
    rng = np.random.default_rng(batches)
    model = _gemm_model(
        "a = Relu(x)\n g = Gemm(a, w1, b1)\n y = Gemm(g, w2, b2)", rng=rng
    )
    data = P._data((4, 16), n=batches, seed=batches)
    q = P._quark_preset(model, data, tmp_path, preset="INT8_TRANSFORMER_DEFAULT")
    m = P._mine_preset(model, data, preset="INT8_TRANSFORMER_DEFAULT")
    P._assert_same_graph(q, m, f"{batches} batches")


def _gemm_model(body, rng=None):
    rng = rng or np.random.default_rng(0)
    m = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float[4,16] x) => (float y) {'
        + body
        + "}"
    )
    shapes = {
        "w1": (16, 16),
        "b1": (16,),
        "w2": (16, 4),
        "b2": (4,),
        "w3": (16, 16),
        "s1": (16,),
        "a1": (4, 4),
    }
    for name in dict.fromkeys(re.findall(r"\b[wbsa]\d\b", body)):
        m.graph.initializer.append(
            numpy_helper.from_array(
                (rng.standard_normal(shapes[name]) * 0.5).astype(np.float32), name
            )
        )
    return P._named(m)


@pytest.mark.parametrize(
    "preset",
    ["A8W8", "A16W8", "VINT8", "U8S8_AAWS", "INT8_TRANSFORMER_DEFAULT", "XINT8"],
)
def test_gemm_beta_moves_into_the_bias_scale(preset, tmp_path):
    """ONNX Runtime's ``QDQGemm`` sets ``beta`` to 1 and folds it into the int32 bias's
    scale (``input scale * weight scale * beta``) -- on the extended quantizer's
    ``adjust_bias_scale`` that is then undone again, by a truncating requantization."""
    model = _gemm_model("y = Gemm<alpha=0.5, beta=2.0>(x, w1, b1)")
    data = P._data((4, 16))
    q = P._quark_preset(model, data, tmp_path, preset=preset)
    m = P._mine_preset(model, data, preset=preset)
    P._assert_same_graph(q, m, f"{preset} Gemm beta")
    for graph in (q, m):
        gemm = next(n for n in graph.graph.node if n.op_type == "Gemm")
        assert {a.name: a.f for a in gemm.attribute if a.name in ("alpha", "beta")} == {
            "alpha": 0.5,
            "beta": 1.0,
        }


def test_a_softmax_the_transformer_preset_does_not_quantize_keeps_its_range(tmp_path):
    """ONNX Runtime's unit range for a Softmax output applies to the Softmax nodes
    the quantizer quantizes; in the transformer scheme (Gemm / MatMul only) the
    Softmax feeding a Gemm is calibrated like any other tensor."""
    model = _gemm_model("s = Softmax<axis=-1>(x)\n y = Gemm(s, w3, b1)")
    data = P._data((4, 16))
    q = P._quark_preset(model, data, tmp_path, preset="INT8_TRANSFORMER_DEFAULT")
    m = P._mine_preset(model, data, preset="INT8_TRANSFORMER_DEFAULT")
    P._assert_same_graph(q, m, "softmax into a Gemm")
    for model_ in (q, m):
        scale = next(iter(_q_scales(model_).values()))[0]
        assert scale < 1 / 255 - 1e-6  # (not the unit range's 1 / 255)


# -- random graphs -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("preset", PRESETS)
def test_random_graphs_match_quark(preset, seed, tmp_path):
    model, shape = _random_graph_ext(seed, big=bool(seed % 2))
    data = P._data(shape, seed=seed + 7)
    _check(model, data, tmp_path, preset, f"random graph #{seed}")


@pytest.mark.parametrize("seed", range(101, 104))
@pytest.mark.parametrize("preset", PRESETS)
def test_random_graphs_with_odd_convolutions_and_extra_outputs_match_quark(
    preset, seed, tmp_path
):
    model, shape = _random_graph_ext(seed, big=True, rich=True)
    data = P._data(shape, seed=seed + 7)
    _check(model, data, tmp_path, preset, f"rich random graph #{seed}")


@pytest.mark.parametrize("seed", range(6))
def test_random_transformer_graphs_match_quark(seed, tmp_path):
    model, shape = _random_gemm_graph(seed)
    data = P._data(shape, seed=seed + 7)
    _check(model, data, tmp_path, "INT8_TRANSFORMER_DEFAULT", f"gemm graph #{seed}")
