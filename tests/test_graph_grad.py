"""Tests for ``onnxsim.graph_grad`` -- reverse-mode differentiation of an ONNX
slice, emitted as ONNX nodes.

A gradient is the one kind of code whose bugs do not announce themselves: a
wrong VJP still produces a finite number of the right shape, the optimizer
still runs, and the only symptom is that training converges somewhere
slightly worse. So almost every test here is the same experiment -- build a
small forward graph, emit its backward with
:func:`onnxsim.graph_grad.build_backward`, run both through onnxruntime, and
check the result against a central finite difference of the same forward
evaluated in float64 by onnx's reference evaluator. The finite difference is
an independent implementation in a different precision through a different
runtime, which is what makes it worth comparing against; the tolerances are
set by the float32 the backward graph itself computes in, not by the
reference.

Two things beyond correctness are also checked, for the same reason
``tests/test_qat_graph.py`` checks them: that the emitted graph stays inside
the operator allowlist the accelerator execution providers actually
implement, and that an op with no rule is refused loudly instead of silently
skipped.
"""

import math

import numpy as np
import onnx
import onnx.parser
import onnx.shape_inference
import pytest
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

from onnxsim import graph_grad, qat_graph

ort = pytest.importorskip("onnxruntime")


class Erf(OpRun):  # noqa: D101 -- the class name is how the evaluator binds it
    """A float64 ``Erf`` for the reference evaluator.

    onnx's own reference implementation rounds to float32
    (``np.vectorize(erf, otypes=["f"])``) whatever it is given, which leaves
    the finite difference below with ~1e-7 of noise over a 1e-5 step -- a 1%
    error, an order of magnitude worse than the float32 gradient it is meant
    to be the reference for. This one keeps the precision the rest of the
    reference forward runs in.
    """

    op_domain = ""

    def _run(self, x):
        return (np.vectorize(math.erf, otypes=[np.float64])(x).astype(x.dtype),)


# Same opset/IR pairing qat_graph builds its step graphs with.
_HEADER = '<ir_version: 8, opset_import: ["": 17]>'

# Step for the central difference. The reference forward runs in float64, so
# this can be small enough for the O(h^2) truncation error to fall well below
# the float32 noise in the graph being tested without round-off taking over.
_H = 1e-5


def _model(body: str) -> onnx.ModelProto:
    return onnx.parser.parse_model(f"{_HEADER}\n{body}")


def _static_shapes(model: onnx.ModelProto) -> dict:
    """Every tensor's static shape -- what ``build_backward`` asks its caller
    for, obtained the way a real caller would."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
    for initializer in inferred.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _as_double(model: onnx.ModelProto) -> onnx.ModelProto:
    """The same forward, retyped float64.

    The point of the reference is to be *more* accurate than the thing under
    test; running the finite difference in the float32 the backward graph
    uses would measure the two against the same round-off and prove nothing.
    """
    doubled = onnx.ModelProto()
    doubled.CopyFrom(model)
    for value in list(doubled.graph.input) + list(doubled.graph.output):
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            value.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    converted = []
    for initializer in doubled.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        if array.dtype == np.float32:
            array = array.astype(np.float64)
        converted.append(onnx.numpy_helper.from_array(array, initializer.name))
    del doubled.graph.initializer[:]
    doubled.graph.initializer.extend(converted)
    return doubled


def _backward_model(model: onnx.ModelProto, targets, seed_name="dY"):
    """``model``'s forward nodes followed by the emitted backward, as one
    graph whose outputs are the requested gradients.

    Wiring the backward into the *same* graph as the forward is not a test
    convenience -- it is the contract: the rules read forward tensors
    (including intermediate node outputs) by name, so they only mean anything
    where those tensors exist.
    """
    shapes = _static_shapes(model)
    output = model.graph.output[0].name
    b = qat_graph.GraphBuilder("bw_")
    grads = graph_grad.build_backward(
        b, list(model.graph.node), shapes, {output: seed_name}, targets
    )

    emitted = {node.op_type for node in b.nodes}
    assert emitted <= graph_grad.BACKWARD_OPS, (
        f"backward graph reached outside the allowlist: "
        f"{sorted(emitted - graph_grad.BACKWARD_OPS)}"
    )

    # A returned gradient can be an alias of an existing tensor (Identity
    # emits no node at all), and a graph output has to be produced by a node,
    # so each one is copied out under a stable name.
    nodes = list(model.graph.node) + list(b.nodes)
    outputs = []
    for target in targets:
        name = f"grad_{target}"
        nodes.append(onnx.helper.make_node("Identity", [grads[target]], [name]))
        outputs.append(
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, shapes[target]
            )
        )
    inputs = list(model.graph.input) + [
        onnx.helper.make_tensor_value_info(
            seed_name, onnx.TensorProto.FLOAT, shapes[output]
        )
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "backward",
        inputs,
        outputs,
        initializer=list(model.graph.initializer) + list(b.initializer),
    )
    built = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    built.ir_version = 8
    onnx.checker.check_model(built)
    return built


def _feeds(model: onnx.ModelProto, rng, overrides=None) -> dict:
    """Random float32 values for every graph input.

    float32 rather than float64 so the analytic and the finite-difference
    paths start from bit-identical inputs -- otherwise the comparison would
    also be measuring the rounding of the inputs themselves.
    """
    overrides = overrides or {}
    feeds = {}
    for value in model.graph.input:
        name = value.name
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if name in overrides:
            feeds[name] = np.asarray(overrides[name](rng, shape), dtype=np.float32)
        else:
            feeds[name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def _finite_difference(evaluator, feeds64, target, seed):
    """d/d(target) of ``sum(forward(feeds) * seed)``, by central differences.

    That scalar is exactly the quantity a VJP computes: seeding the backward
    with ``seed`` and asking for ``target``'s gradient must give the same
    thing, for every ``seed``.
    """
    working = {k: np.array(v, dtype=np.float64) for k, v in feeds64.items()}
    flat = working[target].reshape(-1)
    result = np.empty_like(flat)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + _H
        plus = float((evaluator.run(None, working)[0] * seed).sum())
        flat[i] = original - _H
        minus = float((evaluator.run(None, working)[0] * seed).sum())
        flat[i] = original
        result[i] = (plus - minus) / (2.0 * _H)
    return result.reshape(working[target].shape)


def _check(model, targets=None, overrides=None, seed=0, rtol=2e-3, atol=2e-4):
    """The whole experiment: emitted gradients against finite differences."""
    rng = np.random.default_rng(seed)
    targets = targets or [value.name for value in model.graph.input]
    feeds = _feeds(model, rng, overrides)

    backward = _backward_model(model, targets)
    output_shape = _static_shapes(model)[model.graph.output[0].name]
    grad_seed = rng.standard_normal(output_shape).astype(np.float32)
    session = ort.InferenceSession(
        backward.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    analytic = session.run([f"grad_{t}" for t in targets], dict(feeds, dY=grad_seed))

    evaluator = ReferenceEvaluator(_as_double(model), new_ops=[Erf])
    feeds64 = {k: v.astype(np.float64) for k, v in feeds.items()}
    for target, got in zip(targets, analytic):
        expected = _finite_difference(
            evaluator, feeds64, target, grad_seed.astype(np.float64)
        )
        assert got.shape == expected.shape, f"{target}: {got.shape} != {expected.shape}"
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
    return backward


def _positive(rng, shape):
    return np.abs(rng.standard_normal(shape)) + 0.5


def _away_from_zero(rng, shape):
    x = rng.standard_normal(shape)
    return np.where(x >= 0, x + 0.5, x - 0.5)


# Every rule, one small graph each. Shapes are kept tiny because the finite
# difference costs two full forward passes per element.
_CASES = {
    "matmul": (
        """
        g (float[3,4] A, float[4,5] B) => (float[3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
        None,
    ),
    "matmul_batched": (
        """
        g (float[2,3,4] A, float[2,4,5] B) => (float[2,3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
        None,
    ),
    "matmul_broadcast_batch": (
        """
        g (float[1,3,4] A, float[2,4,5] B) => (float[2,3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
        None,
    ),
    "matmul_rank_mix": (
        """
        g (float[2,3,4] A, float[4,5] B) => (float[2,3,5] Y) {
          Y = MatMul(A, B)
        }
        """,
        None,
    ),
    "add": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Add(A, B)
        }
        """,
        None,
    ),
    "sub": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Sub(A, B)
        }
        """,
        None,
    ),
    "mul": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Mul(A, B)
        }
        """,
        None,
    ),
    "div": (
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Div(A, B)
        }
        """,
        {"B": _away_from_zero},
    ),
    "neg": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Neg(A)
        }
        """,
        None,
    ),
    "identity": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Identity(A)
        }
        """,
        None,
    ),
    "relu": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Relu(A)
        }
        """,
        None,
    ),
    "sigmoid": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Sigmoid(A)
        }
        """,
        None,
    ),
    "tanh": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Tanh(A)
        }
        """,
        None,
    ),
    "erf": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Erf(A)
        }
        """,
        None,
    ),
    "exp": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Exp(A)
        }
        """,
        None,
    ),
    "sqrt": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Sqrt(A)
        }
        """,
        {"A": _positive},
    ),
    "transpose_default": (
        """
        g (float[2,3,4] A) => (float[4,3,2] Y) {
          Y = Transpose(A)
        }
        """,
        None,
    ),
    "transpose_perm": (
        """
        g (float[2,3,4] A) => (float[3,2,4] Y) {
          Y = Transpose <perm = [1, 0, 2]> (A)
        }
        """,
        None,
    ),
    "reshape": (
        """
        g (float[2,3,4] A) => (float[6,4] Y)
        <int64[2] target = {6, 4}>
        {
          Y = Reshape(A, target)
        }
        """,
        None,
    ),
    "reducesum_keepdims": (
        """
        g (float[2,3,4] A) => (float[2,1,4] Y)
        <int64[1] axes = {1}>
        {
          Y = ReduceSum <keepdims = 1> (A, axes)
        }
        """,
        None,
    ),
    "reducesum_dropdims": (
        """
        g (float[2,3,4] A) => (float[2,4] Y)
        <int64[1] axes = {1}>
        {
          Y = ReduceSum <keepdims = 0> (A, axes)
        }
        """,
        None,
    ),
    "reducesum_all": (
        """
        g (float[2,3] A) => (float Y) {
          Y = ReduceSum <keepdims = 0> (A)
        }
        """,
        None,
    ),
    # ReduceMean only takes its axes as a tensor input from opset 18; at the
    # opset 17 this whole module targets they are still an attribute, so both
    # spellings of "which axes" get exercised across these two cases.
    "reducemean_axes_attribute": (
        """
        g (float[2,3,4] A) => (float[2,3,1] Y) {
          Y = ReduceMean <axes = [2], keepdims = 1> (A)
        }
        """,
        None,
    ),
    "reducemean_dropdims": (
        """
        g (float[2,3,4] A) => (float[2,4] Y) {
          Y = ReduceMean <axes = [1], keepdims = 0> (A)
        }
        """,
        None,
    ),
    "reducemean_all": (
        """
        g (float[3,4] A) => (float Y) {
          Y = ReduceMean <keepdims = 0> (A)
        }
        """,
        None,
    ),
    "softmax_last_axis": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Softmax (A)
        }
        """,
        None,
    ),
    # LayerNormalization: the one op a pre-norm transformer block needs that
    # plain arithmetic does not provide. Its dx depends on every element in
    # the normalization group through both the mean and the variance, so a
    # rule that dropped either mean term would still look plausible and would
    # still be wrong -- which is what these finite differences are for.
    "layer_norm": (
        """
        g (float[2,3,4] A, float[4] S, float[4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization (A, S, B)
        }
        """,
        None,
    ),
    "layer_norm_no_bias": (
        """
        g (float[2,3,4] A, float[4] S) => (float[2,3,4] Y) {
          Y = LayerNormalization (A, S)
        }
        """,
        None,
    ),
    "layer_norm_axis_1": (
        """
        g (float[2,3,4] A, float[3,4] S, float[3,4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization <axis = 1> (A, S, B)
        }
        """,
        None,
    ),
    "layer_norm_epsilon": (
        """
        g (float[2,3,4] A, float[4] S, float[4] B) => (float[2,3,4] Y) {
          Y = LayerNormalization <epsilon = 0.001> (A, S, B)
        }
        """,
        None,
    ),
    # ...and composed, since a block never contains a bare LayerNorm: the
    # gradient has to flow through it into an upstream MatMul's weight.
    "layer_norm_in_a_chain": (
        """
        g (float[2,4] A, float[4,4] W, float[4] S, float[4] B) => (float[2,4] Y) {
          H = MatMul (A, W)
          N = LayerNormalization (H, S, B)
          Y = Relu (N)
        }
        """,
        None,
    ),
    "softmax_first_axis": (
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Softmax <axis = 0> (A)
        }
        """,
        None,
    ),
    # Conv, whose rule is written as im2col rather than as the ConvTranspose
    # a convolution's gradient naturally is (see _grad_conv for why). Every
    # attribute below changes the index tables that stand in for the
    # convolution, and a table that is wrong produces a gradient that is
    # finite, correctly shaped and quietly wrong -- so each combination is
    # differenced separately rather than trusted to the plain case.
    "conv": (
        """
        g (float[1,2,4,4] A, float[3,2,3,3] B) => (float[1,3,2,2] Y) {
          Y = Conv(A, B)
        }
        """,
        None,
    ),
    "conv_bias": (
        """
        g (float[1,1,4,4] A, float[2,1,3,3] B, float[2] C)
            => (float[1,2,2,2] Y) {
          Y = Conv(A, B, C)
        }
        """,
        None,
    ),
    "conv_pointwise": (
        """
        g (float[1,2,3,3] A, float[3,2,1,1] B) => (float[1,3,3,3] Y) {
          Y = Conv(A, B)
        }
        """,
        None,
    ),
    "conv_strided_padded": (
        """
        g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
          Y = Conv <strides = [2, 2], pads = [1, 1, 1, 1]> (A, B)
        }
        """,
        None,
    ),
    "conv_dilated": (
        """
        g (float[1,1,5,5] A, float[1,1,2,2] B) => (float[1,1,3,3] Y) {
          Y = Conv <dilations = [2, 2]> (A, B)
        }
        """,
        None,
    ),
    "conv_grouped": (
        """
        g (float[1,4,3,3] A, float[4,2,2,2] B) => (float[1,4,2,2] Y) {
          Y = Conv <group = 2> (A, B)
        }
        """,
        None,
    ),
    "conv_depthwise": (
        """
        g (float[1,3,3,3] A, float[3,1,2,2] B, float[3] C)
            => (float[1,3,2,2] Y) {
          Y = Conv <group = 3> (A, B, C)
        }
        """,
        None,
    ),
    "conv_1d": (
        """
        g (float[1,2,7] A, float[2,2,3] B) => (float[1,2,3] Y) {
          Y = Conv <strides = [2]> (A, B)
        }
        """,
        None,
    ),
    "conv_3d": (
        """
        g (float[1,1,3,3,3] A, float[2,1,2,2,2] B) => (float[1,2,2,2,2] Y) {
          Y = Conv(A, B)
        }
        """,
        None,
    ),
    # The two ``auto_pad`` spellings whose padding is asymmetric, which is
    # what makes them worth differencing rather than only checking
    # structurally: 5 and 6 against a 3-wide kernel at stride 2 need two and
    # one pad respectively, so SAME_UPPER and SAME_LOWER put them in
    # different places and a rule that confused the two would still produce
    # the right *shape*. Both are two-input-channel on purpose -- see
    # ``test_auto_pad_resolves_to_the_padding_the_spec_asks_for`` for the
    # reference evaluator bug that makes the single-channel case unusable as
    # a reference here.
    "conv_same_upper": (
        """
        g (float[1,2,5,6] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
          Y = Conv <strides = [2, 2], auto_pad = "SAME_UPPER"> (A, B)
        }
        """,
        None,
    ),
    "conv_same_lower": (
        """
        g (float[1,2,5,6] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
          Y = Conv <strides = [2, 2], auto_pad = "SAME_LOWER"> (A, B)
        }
        """,
        None,
    ),
    "clip": (
        """
        g (float[3,4] A) => (float[3,4] Y)
        <float lo = {-0.5}, float hi = {0.5}>
        {
          Y = Clip(A, lo, hi)
        }
        """,
        None,
    ),
    "clip_lower_only": (
        """
        g (float[3,4] A) => (float[3,4] Y)
        <float lo = {-0.25}>
        {
          Y = Clip(A, lo)
        }
        """,
        None,
    ),
}


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", sorted(_CASES), ids=sorted(_CASES))
def test_rule_matches_finite_differences(case, seed):
    # Two draws rather than one: a single random point can miss a sign error
    # on a term that happens to be small there, and the finite difference is
    # cheap at these sizes.
    body, overrides = _CASES[case]
    _check(_model(body), overrides=overrides, seed=seed)


@pytest.mark.parametrize(
    "attributes",
    [
        "",
        "<alpha = 0.75>",
        "<beta = 0.5>",
        "<alpha = 1.5, beta = -0.25>",
        "<transB = 1>",
        "<transA = 1>",
        "<transA = 1, transB = 1, alpha = 0.5, beta = 2.0>",
    ],
    ids=["plain", "alpha", "beta", "alpha_beta", "transB", "transA", "everything"],
)
def test_gemm_honours_its_attributes(attributes):
    """Gemm's four attributes each change the gradient, and getting one of
    them wrong is invisible in the forward -- the Y a wrongly-differentiated
    Gemm produces is still correct."""
    trans_a = "transA = 1" in attributes
    trans_b = "transB = 1" in attributes
    a_shape = "4,3" if trans_a else "3,4"
    b_shape = "5,4" if trans_b else "4,5"
    _check(
        _model(
            f"""
            g (float[{a_shape}] A, float[{b_shape}] B, float[5] C)
                => (float[3,5] Y) {{
              Y = Gemm {attributes} (A, B, C)
            }}
            """
        )
    )


@pytest.mark.parametrize(
    "op", ["Add", "Sub", "Mul", "Div"], ids=["add", "sub", "mul", "div"]
)
@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [
        ("4,3", "3", "4,3"),
        ("4,1", "1,3", "4,3"),
        ("3", "4,3", "4,3"),
        ("2,3,4", "4", "2,3,4"),
        ("1,3,1", "2,1,4", "2,3,4"),
    ],
    ids=["row", "outer", "left_smaller", "batched", "both_broadcast"],
)
def test_broadcasting_gradients_are_reduced_to_the_input_shape(
    op, a_shape, b_shape, out_shape
):
    """The single most common way a VJP goes quietly wrong: a broadcast
    operand's gradient must be summed over the axes that were replicated, and
    reshaped back to the operand's own shape. A gradient of the wrong shape
    does not fail -- it broadcasts again inside the optimizer and applies a
    multiple of the intended step."""
    overrides = {"B": _away_from_zero} if op == "Div" else None
    _check(
        _model(
            f"""
            g (float[{a_shape}] A, float[{b_shape}] B) => (float[{out_shape}] Y) {{
              Y = {op}(A, B)
            }}
            """
        ),
        overrides=overrides,
    )


@pytest.mark.parametrize(
    "grad_shape,target_shape",
    [
        ((4, 3), (3,)),
        ((4, 3), (1, 3)),
        ((4, 3), (4, 1)),
        ((2, 3, 4), (4,)),
        ((2, 3, 4), (1, 3, 1)),
        ((2, 3, 4), ()),
        ((2, 3, 4), (2, 3, 4)),
    ],
)
def test_reduce_to_matches_a_numpy_broadcast_reduction(grad_shape, target_shape):
    """The broadcast-undoing helper on its own, against the numpy sum it is
    supposed to be."""
    b = qat_graph.GraphBuilder()
    ctx = graph_grad._Backward(b, {})
    out = ctx.reduce_to("G", grad_shape, target_shape)
    reduced = onnx.helper.make_node("Identity", [out], ["R"])
    graph = onnx.helper.make_graph(
        b.nodes + [reduced],
        "reduce_to",
        [
            onnx.helper.make_tensor_value_info(
                "G", onnx.TensorProto.FLOAT, list(grad_shape)
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "R", onnx.TensorProto.FLOAT, list(target_shape)
            )
        ],
        initializer=b.initializer,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    g = np.random.default_rng(0).standard_normal(grad_shape).astype(np.float32)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    got = session.run(None, {"G": g})[0]

    offset = len(grad_shape) - len(target_shape)
    axes = tuple(range(offset)) + tuple(
        offset + i
        for i, d in enumerate(target_shape)
        if d == 1 and grad_shape[offset + i] != 1
    )
    expected = g.sum(axis=axes).reshape(target_shape)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# Blocks rather than single ops: gradient accumulation, and the two halves of
# a transformer layer. Kept beside `_CASES` because the allowlist test needs
# both corpora -- accumulation is the only thing that emits an `Add`.
_BLOCKS = {
    "shared_input": """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          T = Mul(A, B)
          U = Add(T, A)
          Y = Mul(U, A)
        }
        """,
    "gelu_block": """
        g (float[4,6] X, float[6,8] W1, float[8] B1, float[8,6] W2, float[6] B2)
            => (float[4,6] Y)
        <float half = {0.5}, float one = {1.0}, float inv_sqrt2 = {0.70710678}>
        {
          H = MatMul(X, W1)
          Hb = Add(H, B1)
          S = Mul(Hb, inv_sqrt2)
          E = Erf(S)
          Ep = Add(E, one)
          Hh = Mul(half, Ep)
          G = Mul(Hb, Hh)
          P = MatMul(G, W2)
          Pb = Add(P, B2)
          Y = Add(Pb, X)
        }
        """,
    "attention_block": """
        g (float[2,3,4] Q, float[2,3,4] K, float[2,3,4] V) => (float[2,3,4] Y)
        <float scale = {0.5}>
        {
          Kt = Transpose <perm = [0, 2, 1]> (K)
          L = MatMul(Q, Kt)
          Ls = Mul(L, scale)
          P = Softmax (Ls)
          Y = MatMul(P, V)
        }
        """,
}


def test_a_tensor_consumed_twice_sums_its_gradients():
    """``A`` reaches the output down two paths, so its gradient is the sum of
    both. Dropping one is the classic residual-connection bug, and it is
    invisible unless the two paths are compared against a reference."""
    backward = _check(_model(_BLOCKS["shared_input"]))
    # The sum is a real Add in the graph, not an accident of the arithmetic.
    assert sum(node.op_type == "Add" for node in backward.graph.node) >= 1


def test_a_gradient_seeded_on_an_intermediate_tensor_is_added_in():
    """Seeding an intermediate value adds to what the slice itself
    contributes to it, which is what an auxiliary loss on an activation
    means. Checked against the equivalent single-seed graph rather than
    against a number, so the claim being tested is the additivity itself."""
    model = _model(
        """
        g (float[3,4] A) => (float[3,4] Y) {
          H = Mul(A, A)
          Y = Tanh(H)
        }
        """
    )
    shapes = _static_shapes(model)
    b = qat_graph.GraphBuilder("bw_")
    grads = graph_grad.build_backward(
        b, list(model.graph.node), shapes, {"Y": "dY", "H": "dH"}, ["A"]
    )
    nodes = list(model.graph.node) + list(b.nodes)
    nodes.append(onnx.helper.make_node("Identity", [grads["A"]], ["grad_A"]))
    graph = onnx.helper.make_graph(
        nodes,
        "seeded",
        list(model.graph.input)
        + [
            onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [3, 4])
            for n in ("dY", "dH")
        ],
        [onnx.helper.make_tensor_value_info("grad_A", onnx.TensorProto.FLOAT, [3, 4])],
        initializer=list(b.initializer),
    )
    model2 = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model2.ir_version = 8
    onnx.checker.check_model(model2)

    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    dy = rng.standard_normal((3, 4)).astype(np.float32)
    dh = rng.standard_normal((3, 4)).astype(np.float32)
    session = ort.InferenceSession(
        model2.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    both = session.run(None, {"A": a, "dY": dy, "dH": dh})[0]
    only_y = session.run(None, {"A": a, "dY": dy, "dH": np.zeros_like(dh)})[0]
    only_h = session.run(None, {"A": a, "dY": np.zeros_like(dy), "dH": dh})[0]
    np.testing.assert_allclose(both, only_y + only_h, rtol=1e-5, atol=1e-6)
    # dL/dH = dh flowing through H = A*A gives 2*A*dh; a sanity anchor on the
    # intermediate seed actually being used at all.
    np.testing.assert_allclose(only_h, 2.0 * a * dh, rtol=1e-5, atol=1e-6)


def test_a_transformer_style_block_end_to_end():
    """Two linear layers, a GELU in its exact ``erf`` form, and a residual
    Add -- the shape :mod:`onnxsim.brecq` cannot currently reconstruct
    because its block discovery only walks a linear MatMul chain, and the
    reason this module exists (``docs/qat.md``, deliverable B).

    Note ``X`` feeds both the first projection and the residual, so this also
    exercises gradient accumulation through a whole block rather than through
    a three-node toy.
    """
    _check(_model(_BLOCKS["gelu_block"]))


def test_an_attention_style_block_end_to_end():
    """The other half of a transformer block: a scaled dot product, a
    Softmax, and a second MatMul, batched over heads."""
    _check(_model(_BLOCKS["attention_block"]))


def test_the_emitted_backward_stays_inside_the_operator_allowlist():
    """Every rule at once, checked against :data:`graph_grad.BACKWARD_OPS`.

    The allowlist is not decoration: this machinery exists so a training step
    can run on WebGPU and NPU execution providers, and a backward graph using
    an op none of them implement would pass every numerical test above and
    still be useless for what it is for.
    """
    emitted = set()
    bodies = [body for body, _ in _CASES.values()] + list(_BLOCKS.values())
    for body in bodies:
        model = _model(body)
        b = qat_graph.GraphBuilder()
        graph_grad.build_backward(
            b,
            list(model.graph.node),
            _static_shapes(model),
            {model.graph.output[0].name: "dY"},
            [value.name for value in model.graph.input],
        )
        emitted |= {node.op_type for node in b.nodes}

    # Exact equality, both ways: nothing escapes the allowlist, and the
    # allowlist is not padded with ops no rule can actually produce.
    assert emitted == set(graph_grad.BACKWARD_OPS)

    # And this module's own allowlist has to sit inside the package-wide one a
    # step graph is held to -- an emitted backward is appended to the same
    # builder as the forward, so anything it can produce is something the
    # execution provider has to be able to run.
    assert graph_grad.BACKWARD_OPS <= qat_graph.EP_FRIENDLY_OPS


def test_an_unsupported_op_is_refused():
    """No rule means no gradient -- not a zero, not a straight-through
    approximation. Refusing is recoverable; a quietly wrong gradient is
    not."""
    model = _model(
        """
        g (float[3,4] A) => (float[3,4] Y) {
          Y = Sin(A)
        }
        """
    )
    b = qat_graph.GraphBuilder()
    with pytest.raises(graph_grad.UnsupportedOpError, match="Sin"):
        graph_grad.build_backward(
            b, list(model.graph.node), _static_shapes(model), {"Y": "dY"}, ["A"]
        )


def test_an_unsupported_op_is_refused_even_when_no_gradient_reaches_it():
    """The refusal is a property of the slice, not of where the seed happens
    to reach: a caller should learn its block is out of scope from the block,
    not from which tensor it asked about."""
    model = _model(
        """
        g (float[3,4] A) => (float[3,4] Y, float[3,4] Z) {
          Y = Relu(A)
          Z = Sin(A)
        }
        """
    )
    b = qat_graph.GraphBuilder()
    with pytest.raises(graph_grad.UnsupportedOpError, match="Sin"):
        graph_grad.build_backward(
            b, list(model.graph.node), _static_shapes(model), {"Y": "dY"}, ["A"]
        )


def test_a_matmul_with_a_1d_operand_is_refused():
    """A covered op type in a configuration the rule does not handle is
    refused the same way an uncovered op type is -- the caller's remedy is
    the same either way."""
    model = _model(
        """
        g (float[3,4] A, float[4] B) => (float[3] Y) {
          Y = MatMul(A, B)
        }
        """
    )
    b = qat_graph.GraphBuilder()
    with pytest.raises(graph_grad.UnsupportedOpError, match="1-D"):
        graph_grad.build_backward(
            b, list(model.graph.node), _static_shapes(model), {"Y": "dY"}, ["A"]
        )


def _emitted(model: onnx.ModelProto):
    """The backward a model's nodes emit, as comparable plain data.

    Node op types, names and attributes plus every initializer's name, shape
    and values -- the same things ``onnx/qat_parity_fixtures.txt`` compares,
    and for the same reason: the builder's counter makes the names, so two
    emissions with equal names ran the same operations in the same order.
    """
    shapes = _static_shapes(model)
    b = qat_graph.GraphBuilder("bw_")
    graph_grad.build_backward(
        b,
        list(model.graph.node),
        shapes,
        {model.graph.output[0].name: "dY"},
        [value.name for value in model.graph.input],
    )
    nodes = [
        (
            n.op_type,
            list(n.input),
            list(n.output),
            [onnx.helper.printable_attribute(a) for a in n.attribute],
        )
        for n in b.nodes
    ]
    initializers = [
        (t.name, list(t.dims), t.data_type, onnx.numpy_helper.to_array(t).tolist())
        for t in b.initializer
    ]
    return nodes, initializers


@pytest.mark.parametrize(
    "auto_pad,pads,in_shape,out_shape",
    [
        ("VALID", "0, 0, 0, 0", "1,2,5,5", "1,2,3,3"),
        ("SAME_UPPER", "1, 0, 1, 1", "1,2,5,6", "1,2,3,3"),
        ("SAME_LOWER", "1, 1, 1, 0", "1,2,5,6", "1,2,3,3"),
        ("SAME_UPPER", "1, 0, 1, 1", "1,1,5,6", "1,2,3,3"),
    ],
    ids=["valid", "same_upper", "same_lower", "same_upper_single_channel"],
)
def test_auto_pad_resolves_to_the_padding_the_spec_asks_for(
    auto_pad, pads, in_shape, out_shape
):
    """``auto_pad`` is resolved at build time, so the gradient of a
    ``SAME_UPPER`` convolution must be *the same graph* as the gradient of the
    explicitly padded one it stands for.

    The expected padding is written out here rather than recomputed: 5 and 6
    against a 3-wide kernel at stride 2 need 2 and 1 pads, and SAME_UPPER puts
    the odd one at the end where SAME_LOWER puts it at the beginning. This is
    the check that a misreading of the spec cannot pass.

    It is also the only check ``VALID`` gets, and the reason is worth
    recording: onnx's reference evaluator computes ``auto_pad="VALID"`` as if
    it were ``SAME`` (a 5x5 input through a 3x3 kernel comes back 5x5 rather
    than 3x3), and it disagrees with onnxruntime on ``SAME_UPPER`` for a
    single-channel input as well. Both were checked against onnxruntime and
    against a hand-written convolution when this rule was written; the
    finite-difference cases above therefore avoid exactly those two shapes,
    and the equivalence here covers them instead.
    """
    strides = "" if auto_pad == "VALID" else "strides = [2, 2], "
    channels = in_shape.split(",")[1]
    features = out_shape.split(",")[1]
    automatic = _model(
        f"""
        g (float[{in_shape}] A, float[{features},{channels},3,3] B)
            => (float[{out_shape}] Y) {{
          Y = Conv <{strides}auto_pad = "{auto_pad}"> (A, B)
        }}
        """
    )
    explicit = _model(
        f"""
        g (float[{in_shape}] A, float[{features},{channels},3,3] B)
            => (float[{out_shape}] Y) {{
          Y = Conv <{strides}pads = [{pads}]> (A, B)
        }}
        """
    )
    assert _emitted(automatic) == _emitted(explicit)


def test_a_convolutions_gradient_contains_no_convolution():
    """The design claim of :func:`onnxsim.graph_grad._grad_conv`, checked
    rather than only argued.

    ``dX`` is naturally a ``ConvTranspose`` and ``dW`` a ``Conv``, and neither
    is in :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS` -- the note beside that
    set records what their coverage on the WebGPU and WebNN backends actually
    is and why it was not enough. So the rule is written as im2col instead,
    and what it emits has to stay inside the allowlist like every other rule.
    """
    model = _model(_CASES["conv_grouped"][0])
    b = qat_graph.GraphBuilder()
    graph_grad.build_backward(
        b,
        list(model.graph.node),
        _static_shapes(model),
        {"Y": "dY"},
        ["A", "B"],
    )
    emitted = {node.op_type for node in b.nodes}
    assert not emitted & {"Conv", "ConvTranspose"}
    assert emitted <= graph_grad.BACKWARD_OPS
    # And the one operator this rule needed that arithmetic does not give: a
    # Gather with a constant index, the same shape of thing gather_rows was
    # admitted to EP_FRIENDLY_OPS for.
    assert "Gather" in emitted


@pytest.mark.parametrize(
    "body,fragment",
    [
        (
            """
            g (float[2,3] A, float[3,4] B) => (float[2,4] Y) {
              Y = Conv(A, B)
            }
            """,
            "spatial dimension",
        ),
        (
            """
            g (float[1,4,5,5] A, float[4,2,3,3] B) => (float[1,4,3,3] Y) {
              Y = Conv <group = 3> (A, B)
            }
            """,
            "group=3",
        ),
        (
            """
            g (float[1,4,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
              Y = Conv(A, B)
            }
            """,
            "channels per group",
        ),
        (
            """
            g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
              Y = Conv <kernel_shape = [2, 2]> (A, B)
            }
            """,
            "kernel_shape",
        ),
        (
            """
            g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
              Y = Conv <auto_pad = "SAME"> (A, B)
            }
            """,
            "auto_pad",
        ),
        (
            """
            g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
              Y = Conv <strides = [2]> (A, B)
            }
            """,
            "one entry per spatial axis",
        ),
        (
            """
            g (float[1,2,4,4] A, float[3,2,3,3] B, float[1,3] C)
                => (float[1,3,2,2] Y) {
              Y = Conv(A, B, C)
            }
            """,
            "B has shape",
        ),
    ],
    ids=[
        "no_spatial_axis",
        "group_does_not_divide",
        "weight_channel_mismatch",
        "kernel_shape_disagrees",
        "unknown_auto_pad",
        "wrong_strides_length",
        "bias_not_rank_1",
    ],
)
def test_a_conv_this_rule_cannot_invert_is_refused(body, fragment):
    """A convolution whose geometry does not add up is refused by name.

    This is the boundary that matters most for this rule, because its whole
    method is to precompute *where every element came from*: a misread
    attribute does not produce an error at build time or a wrong shape at run
    time, it produces index tables that gather the wrong elements, and the
    gradient that comes out is finite, correctly shaped and wrong. So each
    disagreement between the attributes, the weight and the declared output
    is refused rather than resolved in favour of one of them.
    """
    model = _model(body)
    # The parser accepts these; onnx's own shape inference is not asked,
    # because several of them are exactly the case where it would object
    # first and the point is what this module does with them.
    shapes = {
        value.name: [d.dim_value for d in value.type.tensor_type.shape.dim]
        for value in list(model.graph.input) + list(model.graph.output)
    }
    with pytest.raises(graph_grad.UnsupportedOpError, match=fragment):
        graph_grad.build_backward(
            qat_graph.GraphBuilder(),
            list(model.graph.node),
            shapes,
            {"Y": "dY"},
            ["A"],
        )


def test_a_conv_whose_output_shape_does_not_follow_is_refused():
    """The catch-all the rest of the refusals lean on: whatever the
    attributes say, the geometry they resolve to has to reproduce the output
    shape the node itself declares.

    Written with the shapes passed in by hand rather than inferred, since a
    model this inconsistent is one onnx's shape inference rejects outright --
    which is the point: ``build_backward`` is given shapes by its caller, and
    a caller that got them from somewhere else must not be trusted to have
    got them right.
    """
    model = _model(
        """
        g (float[1,2,5,5] A, float[2,2,3,3] B) => (float[1,2,3,3] Y) {
          Y = Conv(A, B)
        }
        """
    )
    shapes = {"A": [1, 2, 5, 5], "B": [2, 2, 3, 3], "Y": [1, 2, 4, 4]}
    with pytest.raises(graph_grad.UnsupportedOpError, match="does not follow from"):
        graph_grad.build_backward(
            qat_graph.GraphBuilder(),
            list(model.graph.node),
            shapes,
            {"Y": "dY"},
            ["A"],
        )


def test_ambiguous_reduced_axes_are_refused():
    """``[3, 3] -> [3]`` with ``keepdims=0`` could have reduced either axis,
    and the two answers put the gradient in different places. The axes are a
    tensor input the slice does not carry, so this is refused rather than
    guessed."""
    model = _model(
        """
        g (float[3,3] A) => (float[3] Y)
        <int64[1] axes = {0}>
        {
          Y = ReduceSum <keepdims = 0> (A, axes)
        }
        """
    )
    b = qat_graph.GraphBuilder()
    with pytest.raises(graph_grad.UnsupportedOpError, match="ambiguous"):
        graph_grad.build_backward(
            b, list(model.graph.node), _static_shapes(model), {"Y": "dY"}, ["A"]
        )


def test_a_disconnected_target_is_refused():
    """Returning a zero gradient for a target nothing reaches would hide a
    wrong slice or a typo in a parameter name."""
    model = _model(
        """
        g (float[3,4] A, float[3,4] B) => (float[3,4] Y) {
          Y = Relu(A)
        }
        """
    )
    b = qat_graph.GraphBuilder()
    with pytest.raises(ValueError, match="no gradient reaches"):
        graph_grad.build_backward(
            b, list(model.graph.node), _static_shapes(model), {"Y": "dY"}, ["A", "B"]
        )


def test_a_missing_shape_is_refused():
    """Shapes are what makes undoing a broadcast possible at build time;
    a missing one has to be an error, not a shrug."""
    model = _model(
        """
        g (float[4,3] A, float[3] B) => (float[4,3] Y) {
          Y = Mul(A, B)
        }
        """
    )
    b = qat_graph.GraphBuilder()
    with pytest.raises(ValueError, match="no static shape"):
        graph_grad.build_backward(
            b, list(model.graph.node), {"Y": [4, 3], "A": [4, 3]}, {"Y": "dY"}, ["A"]
        )


def test_the_backward_composes_into_a_step_graph_that_trains():
    """The point of emitting ONNX rather than computing numbers: the result
    drops straight into :mod:`onnxsim.qat_graph`'s Adam step and optimizes,
    on whatever execution provider the caller names.

    The task is deliberately one no hand-derived gradient in this repo covers
    -- fitting a two-layer block with a GELU in the middle against a teacher's
    output, which is exactly ``docs/qat.md``'s block reconstruction in
    miniature.
    """
    rows, k, hidden = 16, 4, 6
    forward = _model(
        f"""
        g (float[{rows},{k}] x, float[{k},{hidden}] w1, float[{hidden},{k}] w2)
            => (float[{rows},{k}] y_hat)
        <float half = {{0.5}}, float one = {{1.0}}, float inv_sqrt2 = {{0.70710678}}>
        {{
          h = MatMul(x, w1)
          s = Mul(h, inv_sqrt2)
          e = Erf(s)
          ep = Add(e, one)
          hh = Mul(half, ep)
          gelu = Mul(h, hh)
          y_hat = MatMul(gelu, w2)
        }}
        """
    )
    shapes = _static_shapes(forward)

    b = qat_graph.GraphBuilder()
    b.nodes.extend(forward.graph.node)
    b.initializer.extend(forward.graph.initializer)
    diff = b.sub("y_hat", "y")
    dl_dy = b.mul(diff, b.const(2.0 / (rows * k)))
    grads = graph_grad.build_backward(
        b, list(forward.graph.node), shapes, {"y_hat": dl_dy}, ["w2"]
    )
    w2_next, m_next, v_next = qat_graph.adam_update(
        b, "w2", grads["w2"], "m", "vv", "lr", "m_correction", "v_correction"
    )
    step = qat_graph.make_step_graph(
        b,
        constants={"x": [rows, k], "y": [rows, k], "w1": [k, hidden]},
        state={
            "w2": ([hidden, k], w2_next),
            "m": ([hidden, k], m_next),
            "vv": ([hidden, k], v_next),
        },
        scalars=["lr", "m_correction", "v_correction"],
        loss=b.mean_square(diff),
    )
    onnx.checker.check_model(step.model)

    rng = np.random.default_rng(21)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    w1 = rng.standard_normal((k, hidden)).astype(np.float32)
    w2_true = rng.standard_normal((hidden, k)).astype(np.float32)
    teacher = ReferenceEvaluator(_as_double(forward), new_ops=[Erf])
    y = teacher.run(
        None,
        {
            "x": x.astype(np.float64),
            "w1": w1.astype(np.float64),
            "w2": w2_true.astype(np.float64),
        },
    )[0].astype(np.float32)

    losses: list = []
    final = qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y, "w1": w1},
        state={
            "w2": np.zeros_like(w2_true),
            "m": np.zeros_like(w2_true),
            "vv": np.zeros_like(w2_true),
        },
        num_steps=1500,
        scalars=lambda t: dict(lr=0.05, **qat_graph.adam_bias_corrections(t)),
        losses=losses,
    )
    assert losses[-1] < losses[0] * 1e-4
    np.testing.assert_allclose(final["w2"], w2_true, rtol=1e-2, atol=1e-2)
