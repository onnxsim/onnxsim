"""Float models shared by ``test_quark_block_preproc.py`` (Quark-free) and
``test_quark_block_preproc_parity.py`` (against the real package): the patterns
Quark's float pre-processing rewrites before the block-format (BFP / MX), bfloat16 /
float16 and ``MATMUL_NBITS`` flows quantize -- BatchNorm after a Conv / ConvTranspose /
Gemm / Concat, Identity, Pad, ReduceMean, HardSwish, MatMul + Add, the decomposed
LayerNorm and Gelu, shared biases -- plus a few Quark leaves alone (a bare Clip output,
data-movement chains). Every model is written in the ONNX text format; random weights
are attached afterwards, as numpy-built initializers.
"""

from typing import Callable, Dict, List, Sequence

import numpy as np
import onnx
from _quark_fusion_common import _gelu_inits, _ln, _ln_inits
from onnx import numpy_helper, parser

F32 = np.float32


def _w(name: str, rng: np.random.Generator, *shape: int, scale: float = 0.4):
    return numpy_helper.from_array(
        (rng.standard_normal(shape) * scale).astype(F32), name
    )


def _pos(name: str, rng: np.random.Generator, *shape: int):
    return numpy_helper.from_array((1 + rng.random(shape)).astype(F32), name)


def _c(name: str, value, dtype=F32):
    return numpy_helper.from_array(np.array(value, dtype), name)


def _model(
    body: str,
    initializer: Sequence[onnx.TensorProto] = (),
    shape: Sequence[int] = (1, 3, 8, 8),
    opset: int = 17,
) -> onnx.ModelProto:
    m = parser.parse_model(
        f"""<ir_version: 8, opset_import: ["": {opset}]>
        g (float{list(shape)} x) => (float y) {{ {body} }}"""
    )
    m.graph.initializer.extend(initializer)
    return m


def _bn(rng: np.random.Generator, channels: int, p: str = ""):
    return [
        _pos(p + "s", rng, channels),
        _w(p + "bb", rng, channels),
        _w(p + "mu", rng, channels),
        _pos(p + "var", rng, channels),
    ]


def conv_bn() -> onnx.ModelProto:
    rng = np.random.default_rng(1)
    return _model(
        "c = Conv(x, w1, b1)\n bn = BatchNormalization(c, s, bb, mu, var)\n"
        " r = Relu(bn)\n y = Conv(r, w2, b2)",
        [
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
            *_bn(rng, 4),
            _w("w2", rng, 4, 4, 1, 1),
            _w("b2", rng, 4),
        ],
    )


def convt_bn() -> onnx.ModelProto:
    rng = np.random.default_rng(2)
    return _model(
        "c = ConvTranspose(x, w1, b1)\n y = BatchNormalization(c, s, bb, mu, var)",
        [_w("w1", rng, 3, 4, 3, 3), _w("b1", rng, 4), *_bn(rng, 4)],
    )


def gemm_bn() -> onnx.ModelProto:
    rng = np.random.default_rng(3)
    return _model(
        "g = Gemm<transB=1>(x, w, b)\n y = BatchNormalization(g, s, bb, mu, var)",
        [_w("w", rng, 8, 16), _w("b", rng, 8), *_bn(rng, 8)],
        shape=(3, 16),
    )


def bn_concat() -> onnx.ModelProto:
    rng = np.random.default_rng(4)
    return _model(
        "c1 = Conv(x, w1, b1)\n c2 = Conv(x, w2, b2)\n cc = Concat<axis=1>(c1, c2)\n"
        " y = BatchNormalization(cc, s, bb, mu, var)",
        [
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
            _w("w2", rng, 4, 3, 3, 3),
            _w("b2", rng, 4),
            *_bn(rng, 8),
        ],
    )


def identity() -> onnx.ModelProto:
    rng = np.random.default_rng(5)
    return _model(
        "i = Identity(x)\n c = Conv(i, w1, b1)\n j = Identity(c)\n r = Relu(j)\n"
        " y = Conv(r, w2, b2)",
        [
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
            _w("w2", rng, 4, 4, 1, 1),
            _w("b2", rng, 4),
        ],
    )


def pad_conv() -> onnx.ModelProto:
    rng = np.random.default_rng(6)
    return _model(
        "p = Pad(x, pads)\n c = Conv(p, w1, b1)\n y = Relu(c)",
        [
            _c("pads", [0, 0, 1, 1, 0, 0, 1, 1], np.int64),
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
        ],
    )


def pad_avgpool() -> onnx.ModelProto:
    return _model(
        "p = Pad(x, pads)\n y = AveragePool<kernel_shape=[3,3]>(p)",
        [_c("pads", [0, 0, 1, 1, 0, 0, 1, 1], np.int64)],
    )


def reducemean() -> onnx.ModelProto:
    rng = np.random.default_rng(7)
    return _model(
        "c = Conv(x, w1, b1)\n r = Relu(c)\n m = ReduceMean<axes=[2,3], keepdims=1>(r)\n"
        " f = Flatten(m)\n y = Gemm(f, w2, b2)",
        [
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
            _w("w2", rng, 4, 5),
            _w("b2", rng, 5),
        ],
    )


def hardswish() -> onnx.ModelProto:
    rng = np.random.default_rng(8)
    return _model(
        "c = Conv(x, w1, b1)\n h = HardSwish(c)\n y = Conv(h, w2, b2)",
        [
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
            _w("w2", rng, 4, 4, 1, 1),
            _w("b2", rng, 4),
        ],
    )


def matmul_add() -> onnx.ModelProto:
    rng = np.random.default_rng(9)
    return _model(
        "h = MatMul(x, w1)\n ha = Add(h, b1)\n r = Relu(ha)\n y = MatMul(r, w2)",
        [_w("w1", rng, 16, 32), _w("b1", rng, 32), _w("w2", rng, 32, 8)],
        shape=(3, 16),
    )


def clip_bare() -> onnx.ModelProto:
    rng = np.random.default_rng(10)
    return _model(
        "c = Conv(x, w1, b1)\n y = Clip(c, lo, hi)",
        [_w("w1", rng, 4, 3, 3, 3), _w("b1", rng, 4), _c("lo", -1.0), _c("hi", 1.0)],
    )


def clip_relu6() -> onnx.ModelProto:
    rng = np.random.default_rng(11)
    return _model(
        "c = Conv(x, w1, b1)\n k = Clip(c, lo, hi)\n y = Conv(k, w2, b2)",
        [
            _w("w1", rng, 4, 3, 3, 3),
            _w("b1", rng, 4),
            _c("lo", 0.0),
            _c("hi", 6.0),
            _w("w2", rng, 4, 4, 1, 1),
            _w("b2", rng, 4),
        ],
    )


def shared_bias() -> onnx.ModelProto:
    rng = np.random.default_rng(12)
    return _model(
        "c1 = Conv(x, w1, b)\n c2 = Conv(c1, w2, b)\n y = Relu(c2)",
        [_w("w1", rng, 4, 3, 3, 3), _w("b", rng, 4), _w("w2", rng, 4, 4, 3, 3)],
    )


def tiny_constants() -> onnx.ModelProto:
    """Constants outside bfloat16's normal range: Quark's BF16 preset clips them into it
    (a zero stays zero); the other formats leave them alone."""
    rng = np.random.default_rng(17)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.4).astype(F32)
    w.reshape(-1)[:3] = [1e-39, 0.0, -3e-39]
    b = np.array([0.5, 1e-40, 0.0, 3.4e38], F32)
    return _model(
        "y = Conv(x, w, b)",
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )


def split_concat() -> onnx.ModelProto:
    rng = np.random.default_rng(13)
    return _model(
        "c = Conv(x, w1, b1)\n a, b = Split<axis=1>(c, sp)\n y = Concat<axis=1>(b, a)",
        [_w("w1", rng, 4, 3, 3, 3), _w("b1", rng, 4), _c("sp", [2, 2], np.int64)],
    )


def leaky_conv() -> onnx.ModelProto:
    rng = np.random.default_rng(14)
    return _model(
        "c = Conv(x, w, b)\n l = LeakyRelu<alpha=0.1>(c)\n y = Sigmoid(l)",
        [_w("w", rng, 4, 3, 3, 3), _w("b", rng, 4)],
    )


def movement_chain() -> onnx.ModelProto:
    """Data-movement ops after a Conv: their outputs share its quantization parameters
    (float16 / bfloat16 pairs read one scale and zero point)."""
    rng = np.random.default_rng(15)
    return _model(
        "c = Conv(x, w, b)\n t = Transpose<perm=[0,1,3,2]>(c)\n"
        " r = Reshape(t, shp)\n u = Unsqueeze(r, ax)\n y = Squeeze(u, ax)",
        [
            _w("w", rng, 4, 3, 3, 3),
            _w("b", rng, 4),
            _c("shp", [1, 4, 8, 8], np.int64),
            _c("ax", [0], np.int64),
        ],
    )


_ERF_GELU = (
    "gh = Mul(f1b, half)\n gd = Div(f1b, rt2)\n ge = Erf(gd)\n ga = Add(ge, one)\n"
    " g = Mul(gh, ga)\n"
)


def transformer_block(opset: int = 17) -> onnx.ModelProto:
    """LayerNorm, MatMul + bias, erf Gelu, MatMul + bias, residual, LayerNorm --
    decomposed as torch exports it."""
    rng = np.random.default_rng(16)
    body = (
        _ln(opset, "x", "h1", "a_")
        + "f1 = MatMul(h1, w1)\n f1b = Add(f1, bb1)\n"
        + _ERF_GELU
        + "f2 = MatMul(g, w2)\n f2b = Add(f2, bb2)\n r = Add(x, f2b)\n"
        + _ln(opset, "r", "y", "b_")
    )
    return _model(
        body,
        _ln_inits(opset, "a_")
        + _ln_inits(opset, "b_")
        + _gelu_inits()
        + [
            _w("w1", rng, 16, 32, scale=0.3),
            _w("bb1", rng, 32, scale=0.1),
            _w("w2", rng, 32, 16, scale=0.3),
            _w("bb2", rng, 16, scale=0.1),
        ],
        shape=(2, 4, 16),
        opset=opset,
    )


def block13() -> onnx.ModelProto:
    return transformer_block(13)


def block17() -> onnx.ModelProto:
    return transformer_block(17)


def block20() -> onnx.ModelProto:
    return transformer_block(20)


def identity_only() -> onnx.ModelProto:
    """Nothing Quark quantizes: it returns the model as given."""
    return _model("a = Identity(x)\n y = Identity(a)")


PATTERNS: Dict[str, Callable[[], onnx.ModelProto]] = {
    f.__name__: f
    for f in (
        conv_bn,
        convt_bn,
        gemm_bn,
        bn_concat,
        identity,
        pad_conv,
        pad_avgpool,
        reducemean,
        hardswish,
        matmul_add,
        clip_bare,
        clip_relu6,
        shared_bias,
        tiny_constants,
        split_concat,
        leaky_conv,
        movement_chain,
        block13,
        block17,
        block20,
    )
}


def input_shape(model: onnx.ModelProto) -> List[int]:
    return [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]


def calibration_data(model: onnx.ModelProto, n: int = 4, seed: int = 3):
    rng = np.random.default_rng(seed)
    shape = tuple(input_shape(model))
    return [{"x": rng.standard_normal(shape).astype(F32)} for _ in range(n)]


def ops(model: onnx.ModelProto) -> List[str]:
    return [n.op_type for n in model.graph.node]
