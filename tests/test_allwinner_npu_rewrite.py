"""scripts/allwinner/npu_rewrite.py replaces LayerNormalization, Gelu and Div with operators the Allwinner NPU toolchain documents.

Each test builds a small graph in the ONNX text format, rewrites it, and checks (1) the result only uses documented operators and
(2) it computes the same values as the original on random inputs, using ONNX's reference evaluator (no onnxruntime needed).
"""

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "allwinner" / "npu_rewrite.py"
)
spec = importlib.util.spec_from_file_location("npu_rewrite", SCRIPT)
npu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu)


def _model(body, initializers=(), opset=17):
    model = parser.parse_model(f'<ir_version: 9, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializers)
    return model


def _init(name, array):
    return numpy_helper.from_array(np.asarray(array, np.float32), name)


def _run(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def _check_equivalent(model, feeds, atol=1e-5):
    rewritten, stats = npu.rewrite(model)
    onnx.checker.check_model(rewritten)
    assert not npu.undocumented(rewritten), npu.undocumented(rewritten)
    for want, got in zip(_run(model, feeds), _run(rewritten, feeds)):
        np.testing.assert_allclose(got, want, atol=atol, rtol=1e-5)
    return rewritten, stats


RNG = np.random.default_rng(0)


@pytest.mark.parametrize("opset", [17, 18, 20])
@pytest.mark.parametrize("with_bias", [True, False])
def test_layer_normalization(opset, with_bias):
    # ReduceMean takes `axes` as an attribute up to opset 17 and as an input from 18; the rewrite must follow the model's opset.
    inputs = "(x, s, b)" if with_bias else "(x, s)"
    model = _model(
        f"g (float[2, 5, 8] x) => (float[2, 5, 8] y) {{ y = LayerNormalization<axis = -1, epsilon = 1e-5>{inputs} }}",
        [
            _init("s", RNG.standard_normal(8) + 1),
            *([_init("b", RNG.standard_normal(8))] if with_bias else []),
        ],
        opset,
    )
    rewritten, stats = _check_equivalent(
        model, {"x": RNG.standard_normal((2, 5, 8)).astype(np.float32) * 3 + 1}
    )
    assert stats["LayerNormalization"] == 1
    assert "LayerNormalization" not in npu.op_histogram(rewritten)
    reduce_nodes = [n for n in rewritten.graph.node if n.op_type == "ReduceMean"]
    assert len(reduce_nodes) == 2
    assert all((len(n.input) == 2) == (opset >= 18) for n in reduce_nodes)


def test_layer_normalization_over_two_axes_and_custom_epsilon():
    model = _model(
        "g (float[3, 4, 6] x) => (float[3, 4, 6] y) { y = LayerNormalization<axis = -2, epsilon = 0.1>(x, s) }",
        [_init("s", RNG.standard_normal((4, 6)) + 1)],
    )
    _check_equivalent(model, {"x": RNG.standard_normal((3, 4, 6)).astype(np.float32)})


def test_layer_normalization_squares_stay_inside_fp16_range():
    # DistilBERT's residual stream has outlier channels; with one channel at 600 the deviation is ~525 and its square ~2.8e5, above fp16's
    # 65504. A decomposed LayerNorm that squares the raw deviation overflows to inf in fp16 (a 6% logit error on SST-2 in the accuracy
    # study); the pre-scaled form must keep every squared tensor in range and still give the same answer.
    model = _model(
        "g (float[2, 3, 8] x) => (float[2, 3, 8] y) { y = LayerNormalization<axis = -1, epsilon = 1e-12>(x, s) }",
        [_init("s", np.ones(8))],
    )
    x = RNG.standard_normal((2, 3, 8)).astype(np.float32)
    x[..., 0] = 600.0
    assert (
        (x - x.mean(-1, keepdims=True)) ** 2
    ).max() > 65504  # the raw squares really would overflow
    rewritten, _ = _check_equivalent(model, {"x": x}, atol=1e-4)
    squares = [n.output[0] for n in rewritten.graph.node if "__sq" in n.output[0]]
    assert (
        len(squares) == 1
    )  # the rewriter's name for the squared (pre-scaled) deviation
    probe = onnx.ModelProto.FromString(rewritten.SerializeToString())
    probe.graph.ClearField("output")
    probe.graph.output.extend(
        onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
        for t in squares
    )
    for value in ReferenceEvaluator(probe).run(None, {"x": x}):
        assert np.abs(value).max() < 65504 / 8


def test_layer_normalization_rejects_a_positive_axis():
    model = _model(
        "g (float[2, 8] x) => (float[2, 8] y) { y = LayerNormalization<axis = 1>(x, s) }",
        [_init("s", np.ones(8))],
    )
    with pytest.raises(ValueError, match="positive axis"):
        npu.rewrite(model)


@pytest.mark.skipif(onnx.defs.onnx_opset_version() < 20, reason="Gelu needs opset 20")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(approximate):
    model = _model(
        f'g (float[4, 16] x) => (float[4, 16] y) {{ y = Gelu<approximate = "{approximate}">(x) }}',
        opset=20,
    )
    # large magnitudes included: erf/tanh saturate and the identity x * (1 + t) / 2 must still hold
    x = (RNG.standard_normal((4, 16)) * 4).astype(np.float32)
    rewritten, stats = _check_equivalent(model, {"x": x}, atol=2e-6)
    assert stats["Gelu"] == 1
    ops = npu.op_histogram(rewritten)
    assert ops["Erf" if approximate == "none" else "Tanh"] == 1


def test_div_by_constant_becomes_a_multiplication():
    model = _model(
        "g (float[3, 4] x) => (float[3, 4] y) { y = Div(x, d) }",
        [_init("d", np.full(4, 8.0) + np.arange(4))],
    )
    rewritten, stats = _check_equivalent(
        model, {"x": RNG.standard_normal((3, 4)).astype(np.float32)}
    )
    assert stats["Div(const)"] == 1
    assert npu.op_histogram(rewritten) == {
        "Mul": 1
    }  # no Reciprocal node: 1/d is folded into the initializer


def test_div_by_a_tensor_uses_reciprocal():
    model = _model(
        "g (float[3, 4] x, float[3, 4] z) => (float[3, 4] y) { y = Div(x, z) }"
    )
    feeds = {
        "x": RNG.standard_normal((3, 4)).astype(np.float32),
        "z": (RNG.random((3, 4)) + 0.5).astype(np.float32),
    }
    rewritten, stats = _check_equivalent(model, feeds)
    assert stats["Div"] == 1
    assert npu.op_histogram(rewritten) == {"Reciprocal": 1, "Mul": 1}


def test_div_by_a_constant_containing_zero_keeps_ieee_semantics_via_reciprocal():
    # 1/0 folded into an initializer would turn x/0 into x*inf: fine for finite x, but 0/0 must stay NaN and signs must follow
    # IEEE, so a constant with a zero is left to the Reciprocal path rather than folded.
    model = _model(
        "g (float[4] x) => (float[4] y) { y = Div(x, d) }",
        [_init("d", [1.0, 0.0, 2.0, 4.0])],
    )
    rewritten, _ = npu.rewrite(model)
    assert "Reciprocal" in npu.op_histogram(rewritten)
    x = np.array([2.0, 0.0, 1.0, -8.0], np.float32)
    with np.errstate(all="ignore"):
        want, got = _run(model, {"x": x})[0], _run(rewritten, {"x": x})[0]
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(want)], want[~np.isnan(want)])


def test_integer_division_is_left_alone():
    # shape arithmetic such as Div on int64 is not a tensor op the NPU runs; rewriting it would be wrong
    model = _model("g (int64[2] a, int64[2] b) => (int64[2] y) { y = Div(a, b) }")
    rewritten, stats = npu.rewrite(model)
    assert npu.op_histogram(rewritten) == {"Div": 1} and not stats


def test_models_without_the_operators_are_unchanged():
    model = _model("g (float[2, 3] x) => (float[2, 3] y) { t = Relu(x) y = Neg(t) }")
    rewritten, stats = npu.rewrite(model)
    assert not stats and npu.op_histogram(rewritten) == npu.op_histogram(model)


def test_unhandled_undocumented_operators_are_reported_not_guessed():
    # Identity is normally removed by onnxsim before this tool runs; if one survives it is reported, not silently dropped.
    model = _model("g (float[2, 3] x) => (float[2, 3] y) { y = Identity(x) }")
    assert npu.undocumented(npu.rewrite(model)[0]) == {"Identity": 1}


def test_large_dimensions_are_flagged_for_the_size_limited_operators():
    # A LLaMA-style vocabulary projection (hidden 64 -> 32000 outputs) and a long-sequence softmax exceed the documented 8191 limit.
    model = _model(
        """g (float[1, 4, 64] x, float[64, 32000] w, float[1, 2, 9000] scores) => (float[1, 4, 32000] logits, float[1, 2, 9000] p) {
             logits = MatMul(x, w)
             p = Softmax<axis = -1>(scores)
           }"""
    )
    found = {(op, tuple(shape)) for _, op, _, shape in npu.large_dims(model)}
    assert ("MatMul", (64, 32000)) in found and ("MatMul", (1, 4, 32000)) in found
    assert ("Softmax", (1, 2, 9000)) in found
    # an ordinary encoder-sized graph raises nothing, and non-limited operators are never flagged
    small = _model(
        "g (float[1, 4, 64] x, float[64, 256] w) => (float[1, 4, 256] y) { y = MatMul(x, w) }"
    )
    assert npu.large_dims(small) == []
    relu = _model("g (float[1, 20000] x) => (float[1, 20000] y) { y = Relu(x) }")
    assert npu.large_dims(relu) == []


def test_documented_set_covers_what_a_transformer_needs():
    needed = "MatMul Gemm Softmax Erf Tanh Sigmoid Silu Exp Sqrt Pow Reciprocal ReduceMean Where Cast Gather Transpose Reshape Slice Split".split()
    assert all(op in npu.DOCUMENTED_ONNX_OPS for op in needed)
    assert not {"LayerNormalization", "Gelu", "Div"} & npu.DOCUMENTED_ONNX_OPS
