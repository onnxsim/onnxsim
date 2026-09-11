"""Legalizing a *training* graph for the AX650, checked offline.

`scripts/axera/legalize.py`'s `TRAINING_RULES` exist because a graph that
computes gradients breaks Pulsar2 in ways an inference graph never does: the
weight is a live tensor rather than a constant, the loss is a scalar, and the
backward pass emits ops and function calls the compiler has never seen. Every
rule here answers a failure a real `pulsar2 build` produced, and every test
checks the same two things -- it fires where it should, and onnxruntime agrees
the graph still computes what it did.

Neither needs Docker nor a card.
"""

import os
import sys

import numpy as np
import onnx
import onnx.parser
from onnx import TensorProto, helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import legalize  # noqa: E402


def _model(nodes, inputs, outputs, initializer=(), opset=17, functions=()):
    graph = helper.make_graph(
        nodes,
        "g",
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
            for n, s in inputs.items()
        ],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
            for n, s in outputs.items()
        ],
        initializer=list(initializer),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    model.functions.extend(functions)
    return model


def _run(model, feeds):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _snr(ref, got):
    ref = np.asarray(ref, np.float64)
    err = ref - np.asarray(got, np.float64)
    return 10 * np.log10((ref**2).sum() / max((err**2).sum(), 1e-30))


def _conv(cin, cout, k, size, stride, pad, nd):
    """A convolution whose weight arrives at runtime -- what training makes."""
    out = (size + 2 * pad - k) // stride + 1
    node = helper.make_node(
        "Conv",
        ["x", "w"],
        ["y"],
        kernel_shape=[k] * nd,
        strides=[stride] * nd,
        pads=[pad] * (2 * nd),
        name="c",
    )
    return _model(
        [node],
        {"x": [1, cin] + [size] * nd, "w": [cout, cin] + [k] * nd},
        {"y": [1, cout] + [out] * nd},
    )


def test_a_live_weight_convolution_becomes_matmuls_in_1d_and_2d():
    """Pulsar2 reaches `AxQuantizedActWeightConv` and fails its shape function
    with the weight still FP32; the 2-D case dies in the backend instead. So
    the convolution is spelled as one MatMul per tap."""
    rng = np.random.default_rng(0)
    for nd, k, stride, pad in (
        (1, 3, 1, 1),
        (2, 3, 1, 1),
        (2, 3, 2, 1),
        (2, 1, 2, 0),
        (2, 7, 2, 3),
    ):
        model = _conv(8, 6, k, 16, stride, pad, nd)
        feeds = {
            "x": rng.standard_normal([1, 8] + [16] * nd).astype(np.float32),
            "w": (rng.standard_normal([6, 8] + [k] * nd) * 0.2).astype(np.float32),
        }
        ref = _run(model, feeds)[0]
        after = onnx.ModelProto.FromString(model.SerializeToString())
        assert legalize.act_weight_conv_to_matmul(after) == 1, (nd, k, stride)
        onnx.checker.check_model(after)
        got = _run(after, feeds)[0]
        assert got.shape == ref.shape
        assert _snr(ref, got) > 100, (nd, k, stride, _snr(ref, got))
        assert {n.op_type for n in after.graph.node} <= {
            "Pad",
            "Transpose",
            "Slice",
            "Reshape",
            "MatMul",
            "Add",
        }


def test_a_convolution_whose_weight_is_constant_is_left_alone():
    """Only a *live* weight is a problem; an ordinary inference convolution
    must keep its fast path."""
    w = numpy_helper.from_array(
        np.random.default_rng(1).standard_normal((6, 8, 3)).astype(np.float32), "w"
    )
    model = _model(
        [helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3], pads=[1, 1])],
        {"x": [1, 8, 16]},
        {"y": [1, 6, 16]},
        [w],
    )
    assert legalize.act_weight_conv_to_matmul(model) == 0
    assert [n.op_type for n in model.graph.node] == ["Conv"]


def test_dilation_and_groups_are_declined_rather_than_approximated():
    model = _conv(8, 8, 3, 16, 1, 1, 1)
    model.graph.node[0].attribute.append(helper.make_attribute("dilations", [2]))
    assert legalize.act_weight_conv_to_matmul(model) == 0
    grouped = _conv(8, 8, 3, 16, 1, 1, 1)
    grouped.graph.node[0].attribute.append(helper.make_attribute("group", 2))
    assert legalize.act_weight_conv_to_matmul(grouped) == 0


def test_gemm_with_two_live_inputs_becomes_matmul():
    """The compiler asks for this by name: `NotImplementedError('Should fuse
    Gemm (two non-parameter inputs) to MatMul.')`."""
    rng = np.random.default_rng(2)
    model = _model(
        [
            helper.make_node(
                "Gemm", ["a", "b", "c"], ["y"], transB=1, alpha=0.5, beta=2.0, name="g"
            )
        ],
        {"a": [5, 4], "b": [6, 4], "c": [5, 6]},
        {"y": [5, 6]},
    )
    feeds = {
        n: rng.standard_normal(s).astype(np.float32)
        for n, s in (("a", (5, 4)), ("b", (6, 4)), ("c", (5, 6)))
    }
    ref = _run(model, feeds)[0]
    assert legalize.gemm_to_matmul(model) == 1
    onnx.checker.check_model(model)
    assert _snr(ref, _run(model, feeds)[0]) > 100
    assert "Gemm" not in {n.op_type for n in model.graph.node}


def test_a_scalar_loss_grows_an_axis_and_names_its_reduction():
    """Two separate failures: Pulsar2's calibration cannot concatenate a rank-0
    tensor across samples, and it reduces only the last axis when a reduction
    names none -- where ONNX reduces all of them."""
    rng = np.random.default_rng(3)
    model = _model(
        [
            helper.make_node("Mul", ["x", "x"], ["sq"]),
            helper.make_node("ReduceMean", ["sq"], ["loss"], keepdims=0),
        ],
        {"x": [1, 4, 8]},
        {"loss": []},
    )
    feeds = {"x": rng.standard_normal((1, 4, 8)).astype(np.float32)}
    ref = _run(model, feeds)[0]
    assert legalize.rank0_to_rank1(model) == 1
    onnx.checker.check_model(model)
    got = _run(model, feeds)[0]
    assert np.shape(got) == (1,)
    assert abs(float(ref) - float(got[0])) < 1e-6
    reduce_node = next(n for n in model.graph.node if n.op_type == "ReduceMean")
    axes = [list(a.ints) for a in reduce_node.attribute if a.name == "axes"]
    assert axes == [[0, 1, 2]], axes


def test_the_reduction_axes_move_to_an_input_at_opset_18():
    """`ReduceMean`'s `axes` became an input at opset 18 and `onnx.checker`
    rejects the other form, so the rule has to know which era it is in."""
    model = _model(
        [
            helper.make_node("Mul", ["x", "x"], ["sq"]),
            helper.make_node("ReduceMean", ["sq"], ["loss"], keepdims=0),
        ],
        {"x": [1, 4, 8]},
        {"loss": []},
        opset=18,
    )
    assert legalize.rank0_to_rank1(model) == 1
    onnx.checker.check_model(model)
    node = next(n for n in model.graph.node if n.op_type == "ReduceMean")
    assert len(node.input) == 2 and node.input[1]
    axes = next(i for i in model.graph.initializer if i.name == node.input[1])
    assert list(numpy_helper.to_array(axes)) == [0, 1, 2]


def test_neg_becomes_mul_by_minus_one():
    """`Neg` is the one op a backward pass emits that the AX650 list lacks."""
    rng = np.random.default_rng(4)
    model = _model(
        [helper.make_node("Neg", ["x"], ["y"])], {"x": [2, 3]}, {"y": [2, 3]}
    )
    feeds = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    ref = _run(model, feeds)[0]
    assert legalize.neg_to_mul(model) == 1
    onnx.checker.check_model(model)
    assert np.allclose(ref, _run(model, feeds)[0])
    assert [n.op_type for n in model.graph.node] == ["Mul"]


def test_ceil_mode_is_cleared_only_where_it_changes_nothing():
    """`graph_grad` declines a pool with `ceil_mode=1`, which stops resnet18d
    dead on its downsample pools. Where the stride divides evenly, ceil and
    floor agree exactly and the flag is decoration."""
    even = _model(
        [
            helper.make_node(
                "AveragePool",
                ["x"],
                ["y"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                ceil_mode=1,
            )
        ],
        {"x": [1, 3, 8, 8]},
        {"y": [1, 3, 4, 4]},
    )
    assert legalize.avgpool_ceil_to_floor(even) == 1
    assert all(
        a.i == 0 for n in even.graph.node for a in n.attribute if a.name == "ceil_mode"
    )
    # kernel 3 stride 2 over 8 leaves a span of 5, which 2 does not divide --
    # ceil keeps a final ragged window that floor drops, so the flag is real
    ragged = _model(
        [
            helper.make_node(
                "AveragePool",
                ["x"],
                ["y"],
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=1,
            )
        ],
        {"x": [1, 3, 8, 8]},
        {"y": [1, 3, 4, 4]},
    )
    assert legalize.avgpool_ceil_to_floor(ragged) == 0


def test_flatten_and_global_pool_become_ops_that_have_gradients():
    """Neither has a rule in `graph_grad`, and neither needs one: a `Flatten`
    is a `Reshape` and a `GlobalAveragePool` is a `ReduceMean`."""
    rng = np.random.default_rng(5)
    model = _model(
        [
            helper.make_node("GlobalAveragePool", ["x"], ["p"]),
            helper.make_node("Flatten", ["p"], ["y"], axis=1),
        ],
        {"x": [2, 3, 4, 4]},
        {"y": [2, 3]},
    )
    feeds = {"x": rng.standard_normal((2, 3, 4, 4)).astype(np.float32)}
    ref = _run(model, feeds)[0]
    assert legalize.global_pool_to_reduce(model) == 1
    assert legalize.flatten_to_reshape(model) == 1
    onnx.checker.check_model(model)
    got = _run(model, feeds)[0]
    assert got.shape == ref.shape
    assert _snr(ref, got) > 100
    kinds = {n.op_type for n in model.graph.node}
    assert kinds == {"ReduceMean", "Reshape"}
    import onnxsim.graph_grad as graph_grad

    assert kinds <= graph_grad.SUPPORTED_OPS


def test_a_local_function_call_is_inlined_into_real_ops():
    """Pulsar2 dispatches on op type, and a call to a locally-defined function
    is not one. `graph_grad`'s templated rules emit exactly eight of them for
    resnet18 -- one per residual connection -- so they are not optional."""
    fn = helper.make_function(
        "onnxsim.grad",
        "Twice",
        ["a"],
        ["b"],
        [
            helper.make_node("Add", ["a", "a"], ["b"]),
        ],
        [helper.make_opsetid("", 17)],
    )
    model = _model(
        [helper.make_node("Twice", ["x"], ["y"], domain="onnxsim.grad")],
        {"x": [2, 3]},
        {"y": [2, 3]},
        functions=[fn],
    )
    assert legalize.inline_local_functions(model) == 1
    onnx.checker.check_model(model)
    assert [n.op_type for n in model.graph.node] == ["Add"]
    assert not model.functions
    rng = np.random.default_rng(6)
    feeds = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    assert np.allclose(_run(model, feeds)[0], 2 * feeds["x"])


def test_inlining_a_graph_with_no_functions_is_a_no_op():
    model = _model([helper.make_node("Relu", ["x"], ["y"])], {"x": [2]}, {"y": [2]})
    assert legalize.inline_local_functions(model) == 0


def test_training_rules_run_in_an_order_that_works():
    """`inline_local_functions` has to come first -- every later rule inspects
    op types and would look straight past a function call."""
    assert legalize.TRAINING_RULES[0] == "inline_local_functions"
    for name in legalize.TRAINING_RULES:
        assert name in legalize.RULES
