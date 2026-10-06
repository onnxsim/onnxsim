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


def _rms_body(opset, reciprocal="Div", axes="-1", exponent="two", extra=""):
    mean = (
        f"ReduceMean<keepdims = 1, axes = [{axes}]>(p)"
        if opset < 18
        else "ReduceMean<keepdims = 1>(p, ax)"
    )
    recip = "Div(one, s)" if reciprocal == "Div" else "Reciprocal(s)"
    return f"""g (float[2, 3, 8] x) => (float[2, 3, 8] y{", float[2, 3, 8] n" if extra == "n_output" else ""}) {{
        p = Pow(x, {exponent})
        m = {mean}
        e = Add(m, eps)
        s = Sqrt(e)
        r = {recip}
        n = Mul(x, r)
        y = Mul(n, w)
    }}"""


def _rms_inits(weights=None):
    return [
        _init("two", np.float32(2.0)),
        _init("three", np.float32(3.0)),
        _init("one", np.float32(1.0)),
        _init("eps", np.float32(1e-5)),
        _init("w", np.ones(8) if weights is None else weights),
        numpy_helper.from_array(np.array([-1], np.int64), "ax"),
    ]


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("reciprocal", ["Div", "Reciprocal"])
def test_rmsnorm_is_rewritten_exactly(opset, reciprocal):
    w = RNG.standard_normal(8) + 1
    model = _model(_rms_body(opset, reciprocal), _rms_inits(w), opset)
    x = (RNG.standard_normal((2, 3, 8)) * 3).astype(np.float32)
    rewritten, stats = _check_equivalent(model, {"x": x}, atol=2e-5)
    assert stats["RMSNorm"] == 1
    ops = npu.op_histogram(rewritten)
    assert (
        "Pow" not in ops
        and "Div" not in ops
        and ops["ReduceMax"] == 1
        and ops["ReduceMean"] == 1
    )


def test_rmsnorm_handles_activations_whose_squares_overflow_fp16():
    # SmolLM2-135M's residual stream reaches ~2e4, so x*x is ~4e8 (fp16 max 65504): torch's formula cannot be held in fp16 at all.
    # Dividing by the row's max |x| first keeps every squared value at most 1 and still gives the same answer.
    model = _model(_rms_body(17), _rms_inits(), 17)
    x = RNG.standard_normal((2, 3, 8)).astype(np.float32)
    x[0, 0, 3] = 25000.0
    x[1, 2, 5] = -31000.0
    assert (x**2).max() > 6e8  # the raw squares are ~1e4 times fp16's range
    rewritten, _ = _check_equivalent(model, {"x": x}, atol=1e-5)
    probe = onnx.ModelProto.FromString(rewritten.SerializeToString())
    internal = [
        n.output[0]
        for n in probe.graph.node
        if "__" in n.output[0] and n.op_type in ("Mul", "Add", "ReduceMean")
    ]
    probe.graph.ClearField("output")
    probe.graph.output.extend(
        onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
        for t in internal
    )
    for value in ReferenceEvaluator(probe).run(None, {"x": x}):
        assert np.abs(value).max() < 65504 / 8, "an interior tensor can overflow fp16"


def test_rmsnorm_rewrite_is_finite_on_an_all_zero_row_and_matches_epsilon_on_tiny_rows():
    model = _model(_rms_body(17), _rms_inits(), 17)
    x = RNG.standard_normal((2, 3, 8)).astype(np.float32)
    x[0, 0] = 0.0  # torch gives 0 * rsqrt(eps) = 0
    x[1, 1] *= (
        1e-4  # rows whose energy is comparable to eps, where the epsilon term matters
    )
    rewritten, _ = _check_equivalent(model, {"x": x}, atol=1e-6)
    out = ReferenceEvaluator(rewritten).run(None, {"x": x})[0]
    assert np.isfinite(out).all() and (out[0, 0] == 0).all()


@pytest.mark.parametrize(
    "body_kwargs,why",
    [
        ({"exponent": "three"}, "a cube is not a mean square"),
        ({"axes": "1"}, "normalising over another axis is not RMSNorm over the last"),
        (
            {"extra": "n_output"},
            "the pre-weight tensor is also a graph output, so it cannot be replaced by an interior one",
        ),
    ],
)
def test_near_miss_graphs_are_left_alone(body_kwargs, why):
    model = _model(_rms_body(17, **body_kwargs), _rms_inits(), 17)
    rewritten, stats = npu.rewrite(model)
    assert not stats.get("RMSNorm"), why
    assert npu.op_histogram(rewritten)["Pow"] == 1


def test_rmsnorm_with_an_extra_consumer_of_an_intermediate_is_left_alone():
    body = """g (float[2, 3, 8] x) => (float[2, 3, 8] y, float[2, 3, 1] z) {
        p = Pow(x, two)
        m = ReduceMean<keepdims = 1, axes = [-1]>(p)
        e = Add(m, eps)
        s = Sqrt(e)
        r = Div(one, s)
        n = Mul(x, r)
        y = Mul(n, w)
        z = Neg(s)
    }"""
    rewritten, stats = npu.rewrite(_model(body, _rms_inits(), 17))
    assert not stats.get("RMSNorm") and npu.op_histogram(rewritten)["Pow"] == 1


def test_rewrite_never_reads_weights_or_chokes_on_a_malformed_initializer():
    # The pattern matchers only need scalars, so a large (or, as here, malformed: 36 elements declared, one stored) initializer is
    # never converted to an array. Reading every weight would copy a whole LLM just to look at a few constants.
    model = _model(
        "g (float[1, 3, 8, 8] x) => (float[1, 4, 8, 8] y) <float[4, 3, 3, 3] W = {0.1}> { y = Conv<pads = [1, 1, 1, 1]>(x, W) }",
        opset=13,
    )
    rewritten, stats = npu.rewrite(model)
    assert not stats and npu.op_histogram(rewritten) == {"Conv": 1}
