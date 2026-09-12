"""AX650 legalization rewrites, checked offline.

`scripts/axera/legalize.py` holds rewrites that make a graph acceptable to
Pulsar2. Each exists because a real build refused a real model without it, so
the tests here check the two properties that matter: the rewrite fires where
it should, and it does not change what the graph computes.

Neither needs Docker or a card.
"""

import importlib.util
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

# scripts/axera/legalize.py and scripts/axelera/legalize.py are two
# different, same-named modules -- a plain `import legalize` here would
# share one `sys.modules["legalize"]` entry with whichever of the two test
# files collects first, silently handing this one the wrong module. Load
# this one under a private key instead, so the two never collide regardless
# of collection order (see tests/test_axelera_legalize.py, which already
# does this on its own side of the same collision).
_spec = importlib.util.spec_from_file_location(
    "axera_legalize", os.path.join(_AXERA_DIR, "legalize.py")
)
legalize = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legalize
_spec.loader.exec_module(legalize)


def _snake_model(dtype=TensorProto.FLOAT, exponent_as_initializer=True):
    """`x + sin(alpha*x)**2 / alpha` -- the Snake activation, spelled the way
    a real neural-codec export spells it."""
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    alpha = numpy_helper.from_array(np.array([2.0], np_dtype), "alpha")
    two = numpy_helper.from_array(np.array(2.0, np_dtype), "two")
    nodes = [
        helper.make_node("Mul", ["x", "alpha"], ["ax"]),
        helper.make_node("Sin", ["ax"], ["s"]),
        helper.make_node("Pow", ["s", "two"], ["s2"]),
        helper.make_node("Div", ["s2", "alpha"], ["d"]),
        helper.make_node("Add", ["x", "d"], ["y"]),
    ]
    initializer = [alpha, two]
    if not exponent_as_initializer:
        initializer = [alpha]
        nodes.insert(2, helper.make_node("Constant", [], ["two"], value=two))
    graph = helper.make_graph(
        nodes,
        "snake",
        [helper.make_tensor_value_info("x", dtype, [1, 4, 8])],
        [helper.make_tensor_value_info("y", dtype, [1, 4, 8])],
        initializer=initializer,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def test_pow2_becomes_mul_and_computes_the_same_thing():
    """`Pow(x, 2)` -> `Mul(x, x)` is exact, and it is what stops Pulsar2 fusing
    a Snake activation into the native op that then fails to build."""
    ort = __import__("onnxruntime")
    before = _snake_model()
    after = _snake_model()
    assert legalize.pow2_to_mul(after) == 1
    assert [n.op_type for n in after.graph.node].count("Pow") == 0
    onnx.checker.check_model(after)

    x = np.random.RandomState(0).randn(1, 4, 8).astype(np.float32)
    runs = []
    for model in (before, after):
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        runs.append(session.run(None, {"x": x})[0])
    assert np.allclose(runs[0], runs[1], atol=1e-6), np.abs(runs[0] - runs[1]).max()


def test_pow2_finds_the_exponent_however_it_is_stored():
    """An export may put the exponent in an initializer or in a `Constant`
    node; a rule that only looks in one place fires on half the graphs."""
    for as_init in (True, False):
        model = _snake_model(exponent_as_initializer=as_init)
        assert legalize.pow2_to_mul(model) == 1, as_init


def test_pow2_leaves_other_exponents_alone():
    """Only the exponent 2 is exact as a self-multiply."""
    model = _snake_model()
    for init in model.graph.initializer:
        if init.name == "two":
            init.CopyFrom(numpy_helper.from_array(np.array(3.0, np.float32), "two"))
    assert legalize.pow2_to_mul(model) == 0
    assert [n.op_type for n in model.graph.node].count("Pow") == 1


def test_float16_graph_is_retyped_everywhere_it_matters():
    """Constants live in three places -- initializers, `Constant` attributes
    and `Cast` targets -- and converting only the first leaves a graph that
    mixes precisions inside a single op, which onnxruntime rejects outright."""
    model = _snake_model(dtype=TensorProto.FLOAT16, exponent_as_initializer=False)
    model.graph.node.append(
        helper.make_node("Cast", ["y"], ["y16"], to=TensorProto.FLOAT16)
    )
    model.graph.output[0].name = "y16"

    assert legalize.float16_to_float32(model) > 0
    assert all(t.data_type != TensorProto.FLOAT16 for t in model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value":
                assert attr.t.data_type != TensorProto.FLOAT16
            if node.op_type == "Cast" and attr.name == "to":
                assert attr.i != TensorProto.FLOAT16
    for value in list(model.graph.input) + list(model.graph.output):
        assert value.type.tensor_type.elem_type != TensorProto.FLOAT16

    ort = __import__("onnxruntime")
    ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )  # loads, which the half-converted graph does not


def _dilated_conv_model(pads, dilation, k=3, C=4, L=16):
    """A dilated 1-D convolution with shapes resolved, the way an export that
    has been through shape inference looks."""
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(C, C, k).astype(np.float32), "w"
    )
    b = numpy_helper.from_array(
        np.random.RandomState(1).randn(C).astype(np.float32), "b"
    )
    out_len = L + pads[0] + pads[1] - ((k - 1) * dilation + 1) + 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["x", "w", "b"],
                ["y"],
                kernel_shape=[k],
                pads=list(pads),
                dilations=[dilation],
                strides=[1],
            )
        ],
        "dilated",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, L])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, out_len])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def _causal_conv_model():
    """A convolution padded only on the left -- the causal form a streaming
    codec uses, and the one Pulsar2's backend refuses."""
    w = numpy_helper.from_array(
        np.random.RandomState(0).randn(4, 4, 3).astype(np.float32), "w"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "w"], ["y"], kernel_shape=[3], pads=[4, 0], dilations=[2]
            )
        ],
        "causal",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def test_asymmetric_padding_is_hoisted_into_a_pad_node():
    """Asymmetric `pads` become an explicit `Pad`, leaving the convolution
    with the symmetric padding every other convolution already has."""
    ort = __import__("onnxruntime")
    before = _causal_conv_model()
    after = _causal_conv_model()
    assert legalize.explicit_conv_padding(after) == 1
    assert [n.op_type for n in after.graph.node] == ["Pad", "Conv"]
    conv = after.graph.node[1]
    pads = next(a.ints for a in conv.attribute if a.name == "pads")
    assert list(pads) == [0, 0]
    onnx.checker.check_model(after)

    x = np.random.RandomState(1).randn(1, 4, 16).astype(np.float32)
    runs = [
        __import__("onnxruntime")
        .InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        .run(None, {"x": x})[0]
        for m in (before, after)
    ]
    assert np.allclose(runs[0], runs[1], atol=1e-5), np.abs(runs[0] - runs[1]).max()
    assert ort is not None


def test_symmetric_padding_is_left_alone():
    """Only asymmetry needs hoisting; rewriting every convolution would add a
    node per layer for nothing."""
    model = _causal_conv_model()
    for attr in model.graph.node[0].attribute:
        if attr.name == "pads":
            del attr.ints[:]
            attr.ints.extend([2, 2])
    assert legalize.explicit_conv_padding(model) == 0
    assert [n.op_type for n in model.graph.node] == ["Conv"]


def test_dilated_conv_becomes_one_convolution_per_tap():
    """`y[t] = sum_j w[:,:,j] . xp[t + j*d]` -- slicing the padded input at each
    tap and convolving with a kernel of one is the same function, with the
    dilation gone."""
    ort = __import__("onnxruntime")
    for pads, dilation in (((4, 0), 2), ((2, 2), 2), ((0, 0), 3)):
        before = _dilated_conv_model(pads, dilation)
        after = _dilated_conv_model(pads, dilation)
        assert legalize.dilated_conv_to_taps(after) == 1, (pads, dilation)
        kinds = [n.op_type for n in after.graph.node]
        assert kinds.count("Conv") == 3 and "Pad" in kinds and "Slice" in kinds
        onnx.checker.check_model(after)

        x = np.random.RandomState(3).randn(1, 4, 16).astype(np.float32)
        runs = [
            ort.InferenceSession(
                m.SerializeToString(), providers=["CPUExecutionProvider"]
            ).run(None, {"x": x})[0]
            for m in (before, after)
        ]
        assert np.allclose(runs[0], runs[1], atol=1e-5), (
            pads,
            dilation,
            np.abs(runs[0] - runs[1]).max(),
        )


def test_dilated_conv_rule_needs_shapes_and_says_so_by_not_firing():
    """The rule sizes each slice from the convolution's output length. A graph
    cut out of a larger one carries no `value_info`, and an earlier version
    skipped every convolution in silence because of it -- so the rule now runs
    shape inference itself, and this is the case that regressed."""
    model = _dilated_conv_model((4, 0), 2)
    del model.graph.value_info[:]
    assert legalize.dilated_conv_to_taps(model) == 1


def test_io_names_are_made_filename_safe():
    """`axcl_run_model` writes one `<tensor name>.bin` per input, so an
    exporter's `/Add_10_output_0` becomes an absolute path and the run dies
    with `PermissionError`."""
    model = _dilated_conv_model((4, 0), 2)
    model.graph.input[0].name = "/Add_10_output_0"
    model.graph.node[0].input[0] = "/Add_10_output_0"
    assert legalize.filename_safe_io_names(model) == 1
    assert model.graph.input[0].name == "Add_10_output_0"
    assert model.graph.node[0].input[0] == "Add_10_output_0"
    onnx.checker.check_model(model)


def test_clean_io_names_are_left_alone():
    model = _dilated_conv_model((4, 0), 2)
    assert legalize.filename_safe_io_names(model) == 0


def test_legalize_reports_what_each_rule_changed():
    model = _snake_model(dtype=TensorProto.FLOAT16)
    applied = legalize.legalize(model)
    assert set(applied) == set(legalize.RULES)
    assert applied["pow2_to_mul"] == 1
    assert applied["float16_to_float32"] > 0
