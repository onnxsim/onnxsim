import os
import tempfile

import numpy as np
import onnx
import onnx.defs
import pytest
import torch
import torchvision as tv

import onnxsim
from onnxsim.test_utils import export_simplify_and_check_by_python_api


def str_is_logical_positive(x: str) -> bool:
    return x.lower() in ["1", "on", "true"]


def skip_in_ci():
    return pytest.mark.skipif(
        str_is_logical_positive(os.getenv("CI", "")), reason="memory limited"
    )


def test_just_reshape():
    class JustReshape(torch.nn.Module):
        def __init__(self):
            super(JustReshape, self).__init__()

        def forward(self, x):
            return x.view((x.shape[0], x.shape[1], x.shape[3] * x.shape[2]))

    net = JustReshape()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net, dummy_input, export_kwargs={"do_constant_folding": False}
    )
    assert len(sim_model.graph.node) == 1


def test_a_model_not_need_simplification():
    class ModelNotNeedSimplification(torch.nn.Module):
        def __init__(self):
            super(ModelNotNeedSimplification, self).__init__()

        def forward(self, x):
            return x + 1

    net = ModelNotNeedSimplification()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(net, dummy_input)
    assert len(sim_model.graph.node) == 1


def test_exprimental_simplify_subgraph():
    class WithSubGraph(torch.nn.Module):
        def __init__(self):
            super(WithSubGraph, self).__init__()

        def forward(self, x):
            if x.sum() > 1.0:
                # NOTE: even onnxsim cannot simplify it,
                # a canonical pass in onnx-optimizer is needed for it.
                # so this test only tests that include_subgraph doesn't
                # result in invalid model in this case
                return 3 + x + 3
            else:
                return x + 4

    net = torch.jit.script(WithSubGraph())
    dummy_input = torch.randn(2)
    sim_model = export_simplify_and_check_by_python_api(
        net, dummy_input, simplify_kwargs={"include_subgraph": True}
    )
    assert len(sim_model.graph.node) == 3
    assert len(sim_model.graph.node[2].attribute[0].g.node) == 2
    assert len(sim_model.graph.node[2].attribute[1].g.node) == 1


def test_dynamic_batch_size():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            return x + 2

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "dynamic_axes": {"input": {0: "batch_size"}},
        },
        simplify_kwargs={"test_input_shapes": {"input": [2, 3, 4, 5]}},
    )
    assert len(sim_model.graph.node) == 1


def test_dynamic_axes_preserve_dynamic_dimension():
    # Regression test for GitHub issue #299. When a dimension of the input is
    # dynamic, the shape computation that reads that dimension at runtime must
    # NOT be constant-folded away, otherwise the simplified model bakes in the
    # dummy batch size and breaks for every other input size.
    #
    # onnxsim only folds a node when *all* of its inputs are constants
    # (initializers or the outputs of already-folded nodes). A graph input is
    # not a constant, so a "Shape" op reading a dynamic input is never folded.
    # This test locks that in behaviourally: the simplified model must still
    # run correctly at a batch size different from the one used at export time.
    class DynamicReshape(torch.nn.Module):
        def __init__(self):
            super(DynamicReshape, self).__init__()

        def forward(self, x):
            # Keep the dynamic batch dim, merge the two static trailing dims.
            return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

    net = DynamicReshape()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
        },
        simplify_kwargs={"test_input_shapes": {"input": [2, 3, 4, 5]}},
    )

    # The simplified model must still expose the batch dimension as dynamic
    # rather than hardcoding the dummy value of 2.
    in_dim0 = sim_model.graph.input[0].type.tensor_type.shape.dim[0]
    out_dim0 = sim_model.graph.output[0].type.tensor_type.shape.dim[0]
    assert in_dim0.dim_value == 0 and in_dim0.dim_param == "batch"
    assert out_dim0.dim_value == 0 and out_dim0.dim_param == "batch"

    # And it must actually run for a batch size other than the export dummy of
    # 2. If the shape computation had been folded to a constant, this would
    # raise or produce the wrong output shape.
    for batch_size in (1, 2, 7):
        x = np.random.rand(batch_size, 3, 4, 5).astype(np.float32)
        outputs = onnxsim.backend.run_model(sim_model, {"input": x})
        (result,) = outputs.values()
        assert result.shape == (batch_size, 3, 20)
        np.testing.assert_allclose(
            result, x.reshape(batch_size, 3, 20), rtol=1e-5, atol=1e-6
        )


# NOTE: `include_subgraph` makes this test fail
@skip_in_ci()
def test_torchvision_fasterrcnn_fpn():
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# maskrcnn is only supported in opset 11 and higher
@skip_in_ci()
def test_torchvision_maskrcnn_fpn_opset11():
    model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# keypointrcnn is only supported in opset 11 and higher
@skip_in_ci()
def test_torchvision_keypointrcnn_fpn():
    model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# shufflenet and mnasnet causes segfault in CI (perhaps because of memory limit)
# but works locally
@skip_in_ci()
def test_torchvision_shufflenet_v2():
    model = tv.models.shufflenet_v2_x1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@skip_in_ci()
def test_torchvision_mnasnet():
    model = tv.models.mnasnet1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@skip_in_ci()
def test_torchvision_deeplabv3():
    model = tv.models.segmentation.deeplabv3_resnet50(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


def test_unused_output():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            x1 = x + 2
            x1 = x1 - 2
            x1 = x1 * 2
            x1 = x1 / 2
            y1 = x1
            x2 = x + 2
            x2 = x2 - 2
            x2 = x2 * 2
            x2 = x2 / 2
            y2 = x2
            x3 = x + 2
            x3 = x3 - 2
            x3 = x3 * 2
            x3 = x3 / 2
            y3 = x3
            return y1, y2, y3

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "output_names": ["output0", "output1", "output2"],
        },
        simplify_kwargs={"unused_output": ["output1", "output2"]},
    )
    assert len(sim_model.graph.node) == 4


def test_remove_unused_initializer():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.w = torch.nn.Parameter(torch.ones(5, 4))

        def forward(self, x):
            return x + torch.transpose(self.w, 0, 1)

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        is_model_valid=lambda model: any(
            node.op_type == "Transpose" for node in model.graph.node
        ),
        export_kwargs={"do_constant_folding": False},
    )
    assert len(sim_model.graph.node) == 1
    assert len(sim_model.graph.initializer) == 1


@skip_in_ci()
def test_model_larger_than_2gb():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            # a parameter is 500MB
            self.w1 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w2 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w3 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w4 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w5 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))

        def forward(self, x):
            return x + (self.w1 + self.w2 + self.w3 + self.w4 + self.w5)

    net = SimpleModel()
    dummy_input = torch.randn(125 * 1024 * 1024)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        is_model_valid=lambda model: (
            sum(node.op_type == "Add" for node in model.graph.node) == 5
        ),
        export_kwargs={"do_constant_folding": False},
    )
    assert len(sim_model.graph.node) == 1
    assert sim_model.graph.node[0].op_type == "Add"


def test_unset_optional_input():
    fmap = []
    nodes = []
    initializers = []

    fmap.append(
        onnx.helper.make_tensor_value_info(
            "y", onnx.TensorProto.FLOAT, shape=(1, 3, 4, 4)
        )
    )

    X = np.random.rand(1, 3, 2, 2).astype(np.float32)
    initializers.append(
        onnx.helper.make_tensor(
            "X", onnx.TensorProto.FLOAT, X.shape, X.copy().tobytes(), raw=True
        )
    )
    sizes = np.asarray([1, 3, 4, 4]).astype(np.int64)
    initializers.append(
        onnx.helper.make_tensor(
            "sizes",
            onnx.TensorProto.INT64,
            sizes.shape,
            sizes.copy().tobytes(),
            raw=True,
        )
    )

    nodes.append(
        onnx.helper.make_node(
            "Resize", inputs=["X", "", "", "sizes"], outputs=["y"], mode="linear"
        )
    )

    graph_def = onnx.helper.make_graph(
        nodes,
        "test_unset_optional_input",
        [],
        [fmap[-1]],
        value_info=fmap,
        initializer=initializers,
    )

    opset_imports = [onnx.helper.make_opsetid("", 14)]

    model = onnx.helper.make_model(
        graph_def, opset_imports=opset_imports, ir_version=10
    )
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    assert len(model.graph.node) == 1
    assert len(model.graph.initializer) == 2
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1


def test_fold_deterministic_op():
    # An op that the operator schema marks as deterministic and whose inputs are
    # all constants should be constant-folded away.
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    initializers = [
        onnx.helper.make_tensor(
            "a", onnx.TensorProto.FLOAT, a.shape, a.tobytes(), raw=True
        ),
        onnx.helper.make_tensor(
            "b", onnx.TensorProto.FLOAT, b.shape, b.tobytes(), raw=True
        ),
    ]
    node = onnx.helper.make_node("Add", inputs=["a", "b"], outputs=["y"])
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 3))
    graph_def = onnx.helper.make_graph(
        [node], "test_fold_deterministic_op", [], [out], initializer=initializers
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    # The Add node is folded into a single constant initializer.
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1


def test_do_not_fold_random_op():
    # RandomUniform is non-deterministic according to the operator schema
    # determinism attribute, so it must not be constant-folded even though it
    # has no non-constant inputs.
    node = onnx.helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["y"],
        shape=[2, 3],
        dtype=onnx.TensorProto.FLOAT,
    )
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 3))
    graph_def = onnx.helper.make_graph([node], "test_do_not_fold_random_op", [], [out])
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    sim_model, _ = onnxsim.simplify(model, check_n=0)
    assert len(sim_model.graph.node) == 1
    assert sim_model.graph.node[0].op_type == "RandomUniform"
    assert len(sim_model.graph.initializer) == 0


def test_do_not_fold_random_like_op():
    # RandomNormalLike is non-deterministic; it must not be folded even when its
    # input is a constant.
    x = np.zeros((2, 3), dtype=np.float32)
    initializers = [
        onnx.helper.make_tensor(
            "x", onnx.TensorProto.FLOAT, x.shape, x.tobytes(), raw=True
        ),
    ]
    node = onnx.helper.make_node("RandomNormalLike", inputs=["x"], outputs=["y"])
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 3))
    graph_def = onnx.helper.make_graph(
        [node], "test_do_not_fold_random_like_op", [], [out], initializer=initializers
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    sim_model, _ = onnxsim.simplify(model, check_n=0)
    assert any(n.op_type == "RandomNormalLike" for n in sim_model.graph.node)


def test_overwrite_input_shape_ignores_non_positive():
    # A non-positive value in overwrite_input_shapes must not be written to the
    # graph as a literal (e.g. 0) dimension; the original dimension should be
    # kept instead so the simplified model stays runnable (GitHub issue #237).
    x = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, ["N", 3, "H", "W"]
    )
    y = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, ["N", 3, "H", "W"]
    )
    node = onnx.helper.make_node("Relu", ["input"], ["output"])
    graph_def = onnx.helper.make_graph(
        [node], "test_overwrite_input_shape_ignores_non_positive", [x], [y]
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    sim_model, _ = onnxsim.simplify(
        model, overwrite_input_shapes={"input": [1, 3, 0, 0]}
    )
    dims = sim_model.graph.input[0].type.tensor_type.shape.dim
    # The positive value is applied, the non-positive ones are left untouched
    # (the original dynamic dim params are kept, never set to 0).
    assert dims[0].dim_value == 1
    assert dims[2].dim_param == "H"
    assert dims[3].dim_param == "W"


def test_preserve_doc_strings():
    # onnxsim must not drop the doc_string fields of the model / graph / inputs
    # / outputs while simplifying (GitHub issue #428).
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    x.doc_string = "input documentation"
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    y.doc_string = "output documentation"
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph_def = onnx.helper.make_graph([node], "test_preserve_doc_strings", [x], [y])
    graph_def.doc_string = "graph documentation"
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.doc_string = "model documentation"

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert sim_model.doc_string == "model documentation"
    assert sim_model.graph.doc_string == "graph documentation"
    assert sim_model.graph.input[0].doc_string == "input documentation"
    assert sim_model.graph.output[0].doc_string == "output documentation"


def _make_scalar_initializer(name: str, value, dtype) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _quant_params():
    return [
        _make_scalar_initializer("s", 0.01, np.float32),
        _make_scalar_initializer("zp", 128, np.uint8),
    ]


def _build_contrib_model(nodes, inputs, outputs, initializer):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer=initializer)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 13),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 9
    return model


def _value_info_shape(model: onnx.ModelProto, name: str):
    for vi in model.graph.value_info:
        if vi.name == name:
            tensor_type = vi.type.tensor_type
            if not tensor_type.HasField("shape"):
                return None
            return [d.dim_value for d in tensor_type.shape.dim]
    return None


def test_qlinear_add_shape_inference():
    # QLinearAdd is an ONNX Runtime "com.microsoft" contrib op. Without a schema
    # registered for it, ONNX shape inference stops and the intermediate tensor
    # never gets a shape (GitHub issue #245).
    nodes = [
        onnx.helper.make_node(
            "QLinearAdd",
            ["A", "s", "zp", "B", "s", "zp", "s", "zp"],
            ["C"],
            domain="com.microsoft",
        ),
        onnx.helper.make_node("DequantizeLinear", ["C", "s", "zp"], ["out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
        onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "out", onnx.TensorProto.FLOAT, [1, 3, 16, 16]
        )
    ]
    model = _build_contrib_model(nodes, inputs, outputs, _quant_params())
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 3, 16, 16]


def test_qlinear_concat_shape_inference():
    nodes = [
        onnx.helper.make_node(
            "QLinearConcat",
            ["s", "zp", "A", "s", "zp", "B", "s", "zp"],
            ["C"],
            domain="com.microsoft",
            axis=1,
        ),
        onnx.helper.make_node("DequantizeLinear", ["C", "s", "zp"], ["out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
        onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 5, 16, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "out", onnx.TensorProto.FLOAT, [1, 8, 16, 16]
        )
    ]
    model = _build_contrib_model(nodes, inputs, outputs, _quant_params())
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 8, 16, 16]


def test_unknown_contrib_op_is_tolerated():
    # Registering schemas for the supported quantized ops must not make the
    # checker reject other, unregistered "com.microsoft" contrib operators.
    nodes = [
        onnx.helper.make_node(
            "QLinearAdd",
            ["A", "s", "zp", "B", "s", "zp", "s", "zp"],
            ["C"],
            domain="com.microsoft",
        ),
        onnx.helper.make_node(
            "SomeUnknownContribOp", ["C"], ["D"], domain="com.microsoft"
        ),
        onnx.helper.make_node("DequantizeLinear", ["C", "s", "zp"], ["out"]),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
        onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "out", onnx.TensorProto.FLOAT, [1, 3, 16, 16]
        ),
        onnx.helper.make_tensor_value_info("D", onnx.TensorProto.UINT8, [1, 3, 16, 16]),
    ]
    model = _build_contrib_model(nodes, inputs, outputs, _quant_params())
    sim_model, check_ok = onnxsim.simplify(model, skip_constant_folding=True)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 3, 16, 16]


def test_run_coerces_non_ndarray_output():
    # Regression test for GitHub PR #249. The inference backend returns a
    # non-ndarray value for a sequence output: a SequenceEmpty op produces an
    # empty Python list rather than a numpy array. Passing that straight to
    # onnx.numpy_helper.from_array used to crash the executor with
    #     AttributeError: 'list' object has no attribute 'shape'
    # The executor must coerce such a value into an (empty) numpy array so the
    # serialization keeps working.
    from onnxsim import onnx_simplifier

    node = onnx.helper.make_node(
        "SequenceEmpty", [], ["seq"], dtype=onnx.TensorProto.FLOAT
    )
    seq_out = onnx.helper.make_value_info(
        "seq",
        onnx.helper.make_sequence_type_proto(
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, None)
        ),
    )
    graph = onnx.helper.make_graph([node], "g", [], [seq_out])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )

    # Drive the executor with the real backend: SequenceEmpty yields an empty
    # list, exercising the exact code path that used to raise.
    executor = onnx_simplifier.PyModelExecutor()
    outputs = executor.Run(model.SerializeToString(), [])

    assert len(outputs) == 1
    assert outputs[0] == []

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok


def _make_batched_nms_trt_model():
    # A model whose only compute node is the TensorRT plugin ``BatchedNMS_TRT``
    # exported into the *default* ONNX domain, exactly as reported in GitHub
    # issue #107 ("No Op registered for BatchedNMS_TRT with domain_version of 9").
    boxes = onnx.helper.make_tensor_value_info(
        "boxes", onnx.TensorProto.FLOAT, [1, 100, 1, 4]
    )
    scores = onnx.helper.make_tensor_value_info(
        "scores", onnx.TensorProto.FLOAT, [1, 100, 5]
    )
    num_detections = onnx.helper.make_tensor_value_info(
        "num_detections", onnx.TensorProto.INT32, [1, 1]
    )
    nmsed_boxes = onnx.helper.make_tensor_value_info(
        "nmsed_boxes", onnx.TensorProto.FLOAT, [1, 20, 4]
    )
    nmsed_scores = onnx.helper.make_tensor_value_info(
        "nmsed_scores", onnx.TensorProto.FLOAT, [1, 20]
    )
    nmsed_classes = onnx.helper.make_tensor_value_info(
        "nmsed_classes", onnx.TensorProto.FLOAT, [1, 20]
    )
    node = onnx.helper.make_node(
        "BatchedNMS_TRT",
        ["boxes", "scores"],
        ["num_detections", "nmsed_boxes", "nmsed_scores", "nmsed_classes"],
        # plugin-specific attributes of assorted types
        shareLocation=1,
        backgroundLabelId=-1,
        numClasses=5,
        topK=100,
        keepTopK=20,
        scoreThreshold=0.3,
        iouThreshold=0.5,
        isNormalized=1,
        clipBoxes=1,
    )
    graph = onnx.helper.make_graph(
        [node],
        "batched_nms_trt",
        [boxes, scores],
        [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 9)]
    )
    model.ir_version = 6
    return model


def test_custom_trt_op_in_default_domain_is_simplified():
    # Regression test for GitHub issues #107 and #220. A custom TensorRT plugin
    # op (``BatchedNMS_TRT``) exported into the default ONNX domain used to make
    # onnxsim fail in onnx.checker.check_model with
    #     No Op registered for BatchedNMS_TRT with domain_version of 9
    # onnxsim now registers a permissive placeholder schema for such ops so the
    # model passes validation and is simplified with the op preserved.
    model = _make_batched_nms_trt_model()

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok

    nms_nodes = [n for n in sim_model.graph.node if n.op_type == "BatchedNMS_TRT"]
    assert len(nms_nodes) == 1
    # The op stays in the default domain and keeps its attributes.
    assert nms_nodes[0].domain in ("", "ai.onnx")
    attr_names = {a.name for a in nms_nodes[0].attribute}
    assert {"numClasses", "keepTopK", "iouThreshold"} <= attr_names


def test_custom_trt_op_does_not_block_surrounding_simplification():
    # The presence of a default-domain custom op must not prevent onnxsim from
    # simplifying the rest of the graph. Here a redundant Identity feeding the
    # plugin should be eliminated while the custom op survives (issues #107/#220).
    boxes = onnx.helper.make_tensor_value_info(
        "boxes", onnx.TensorProto.FLOAT, [1, 100, 1, 4]
    )
    scores = onnx.helper.make_tensor_value_info(
        "scores", onnx.TensorProto.FLOAT, [1, 100, 5]
    )
    out = onnx.helper.make_tensor_value_info(
        "num_detections", onnx.TensorProto.INT32, [1, 1]
    )
    nodes = [
        onnx.helper.make_node("Identity", ["boxes"], ["boxes_id"]),
        onnx.helper.make_node(
            "BatchedNMS_TRT",
            ["boxes_id", "scores"],
            ["num_detections"],
            numClasses=5,
            topK=100,
            keepTopK=20,
        ),
    ]
    graph = onnx.helper.make_graph(nodes, "g", [boxes, scores], [out])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 11)]
    )
    model.ir_version = 6

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Identity" not in op_types
    assert op_types.count("BatchedNMS_TRT") == 1


def _register_custom_onnx_schema(op_type, domain, since_version=1):
    # Register a custom operator schema in the Python ``onnx`` module the same
    # way a user would to teach onnx about their custom operator (GitHub issue
    # #326). The schema has a single float input/output and one optional float
    # attribute so the round-trip through onnxsim's importer covers inputs,
    # outputs, attributes with defaults and type constraints.
    OpSchema = onnx.defs.OpSchema
    schema = OpSchema(
        op_type,
        domain,
        since_version,
        inputs=[OpSchema.FormalParameter("X", "T", "the input")],
        outputs=[OpSchema.FormalParameter("Y", "T", "the output")],
        type_constraints=[("T", ["tensor(float)"], "Constrain to float tensors.")],
        attributes=[
            OpSchema.Attribute(
                "alpha", OpSchema.AttrType.FLOAT, "slope", required=False
            ),
        ],
    )
    onnx.defs.register_schema(schema)


def test_import_onnx_schemas_bridges_registry():
    # A schema registered in the Python ``onnx`` module lives in a different
    # registry from onnxsim's statically linked one. ``import_onnx_schemas``
    # must copy it across so onnxsim's registry learns about the custom op.
    from onnxsim import onnx_simplifier

    op_type = "OnnxsimBridgeTestOp"
    domain = "onnxsim.bridge.test"

    C = onnx_simplifier.C
    assert not C._has_schema(op_type, domain)

    _register_custom_onnx_schema(op_type, domain)
    # Registering in ``onnx`` alone must not affect onnxsim's separate registry.
    assert not C._has_schema(op_type, domain)

    imported = onnxsim.import_onnx_schemas()
    assert imported >= 1
    assert C._has_schema(op_type, domain)

    # Idempotent: a second call imports nothing new for this op (it is already
    # known) and does not raise.
    onnxsim.import_onnx_schemas()
    assert C._has_schema(op_type, domain)


def test_custom_op_with_registered_schema_is_simplified():
    # End-to-end: a model using a custom operator whose schema was registered via
    # ``onnx.defs.register_schema`` must simplify successfully -- the custom op is
    # preserved (with its attribute) and a redundant Identity feeding it is
    # eliminated -- instead of failing validation (GitHub issue #326).
    op_type = "OnnxsimCustomLeakyRelu"
    domain = "onnxsim.custom.ops"
    _register_custom_onnx_schema(op_type, domain)

    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    nodes = [
        onnx.helper.make_node("Identity", ["X"], ["X_id"]),
        onnx.helper.make_node(op_type, ["X_id"], ["Y"], domain=domain, alpha=0.1),
    ]
    graph = onnx.helper.make_graph(nodes, "custom_op_graph", [x], [y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 13),
            onnx.helper.make_opsetid(domain, 1),
        ],
    )
    model.ir_version = 9

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok

    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Identity" not in op_types
    assert op_types.count(op_type) == 1
    custom_node = next(n for n in sim_model.graph.node if n.op_type == op_type)
    assert custom_node.domain == domain
    assert any(a.name == "alpha" for a in custom_node.attribute)


def test_import_custom_schemas_can_be_disabled():
    # ``import_custom_schemas=False`` must leave onnxsim's schema registry
    # untouched, while the default (True) bridges the schema across.
    from onnxsim import onnx_simplifier

    C = onnx_simplifier.C
    op_type = "OnnxsimDisableTestOp"
    domain = "onnxsim.disable.test"
    _register_custom_onnx_schema(op_type, domain)
    assert not C._has_schema(op_type, domain)

    # A trivial model that does not even use the custom op.
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph = onnx.helper.make_graph([node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    # With the import disabled, onnxsim's registry stays untouched.
    onnxsim.simplify(model, import_custom_schemas=False)
    assert not C._has_schema(op_type, domain)

    # With the import enabled (the default), the schema is bridged into onnxsim.
    onnxsim.simplify(model)
    assert C._has_schema(op_type, domain)


def test_custom_op_shape_inference_via_python_trampoline():
    # When a custom operator's Python schema carries a type/shape inference
    # function, onnxsim registers a trampoline that runs that Python function
    # during its own shape inference (GitHub issue #326). The distinctive
    # intermediate dimension 99 -- taken from the ``pad`` attribute -- can only
    # appear in ``t``'s inferred shape if the user's Python inference function
    # actually ran inside onnxsim's shape inference.
    op_type = "OnnxsimShapeInferOp"
    domain = "onnxsim.infer.test"
    OpSchema = onnx.defs.OpSchema
    schema = OpSchema(
        op_type,
        domain,
        1,
        inputs=[OpSchema.FormalParameter("X", "T")],
        outputs=[OpSchema.FormalParameter("Y", "T")],
        type_constraints=[("T", ["tensor(float)"], "Constrain to float tensors.")],
        attributes=[
            OpSchema.Attribute(
                "pad", OpSchema.AttrType.INT, "extra dim", required=False
            ),
        ],
    )

    def infer(ctx):
        # Output type == input type with one extra trailing dimension whose size
        # is the ``pad`` attribute.
        output_type = onnx.TypeProto()
        output_type.CopyFrom(ctx.get_input_type(0))
        pad_attr = ctx.get_attribute("pad")
        pad = pad_attr.i if pad_attr is not None else 0
        output_type.tensor_type.shape.dim.add().dim_value = pad
        ctx.set_output_type(0, output_type)

    schema.set_type_and_shape_inference_function(infer)
    onnx.defs.register_schema(schema)
    try:
        # The custom op feeds an ``Add`` (which survives simplification), so the
        # intermediate ``t`` keeps a value_info entry whose shape is produced only
        # by the custom operator's inference function.
        x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])
        y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3, 99])
        nodes = [
            onnx.helper.make_node(op_type, ["X"], ["t"], domain=domain, pad=99),
            onnx.helper.make_node("Add", ["t", "t"], ["Y"]),
        ]
        graph = onnx.helper.make_graph(nodes, "shape_infer_graph", [x], [y])
        model = onnx.helper.make_model(
            graph,
            opset_imports=[
                onnx.helper.make_opsetid("", 13),
                onnx.helper.make_opsetid(domain, 1),
            ],
        )
        model.ir_version = 9

        sim_model, check_ok = onnxsim.simplify(model)
        assert check_ok
        assert _value_info_shape(sim_model, "t") == [2, 3, 99]
    finally:
        # Deregister the Python inference function so onnx's global registry does
        # not segfault while tearing it down at interpreter shutdown.
        onnx.defs.deregister_schema(op_type, 1, domain)


def test_nameless_nodes_get_names():
    # Regression test for GitHub issue #269. Nodes that have no name in the input
    # model (and nodes left nameless by onnx-optimizer passes) must be assigned
    # unique names during simplification, otherwise downstream tools that key on
    # node names break.
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    # Both nodes are created without a name (the `name` argument is omitted) and
    # operate on the non-constant graph input, so they survive simplification.
    nodes = [
        onnx.helper.make_node("Abs", ["X"], ["t"]),
        onnx.helper.make_node("Relu", ["t"], ["Y"]),
    ]
    graph_def = onnx.helper.make_graph(nodes, "test_nameless_nodes", [x], [y])
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    # Sanity check: the input model really has nameless nodes.
    assert all(node.name == "" for node in model.graph.node)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(sim_model.graph.node) == 2

    names = [node.name for node in sim_model.graph.node]
    # Every surviving node has a non-empty, unique name.
    assert all(name != "" for name in names)
    assert len(set(names)) == len(names)


def test_simplify_path_with_external_data():
    # When ``simplify`` is given a *file path* to a model whose weights live in a
    # separate external-data file, it defers loading that (potentially huge)
    # external data until right before the model is serialized for the C++
    # simplifier -- every graph-metadata phase in between runs without the
    # weights resident. This regression test makes sure that deferral still
    # produces a correct result: the external tensor data must be materialized so
    # the constant fold below can happen and the values are preserved.
    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)
    initializers = [
        onnx.numpy_helper.from_array(a, "a"),
        onnx.numpy_helper.from_array(b, "b"),
    ]
    node = onnx.helper.make_node("Add", ["a", "b"], ["y"])
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (64, 64))
    graph_def = onnx.helper.make_graph(
        [node],
        "test_simplify_path_with_external_data",
        [],
        [out],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        # The .onnx file itself carries no raw tensor data -- it is all in the
        # external .data file, so this really exercises the deferred-load path.
        assert os.path.exists(os.path.join(tmpdir, "model.data"))

        sim_model, check_ok = onnxsim.simplify(model_path, check_n=1)

    assert check_ok
    # Add of two constants is folded into a single initializer holding a + b.
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1
    folded = onnx.numpy_helper.to_array(sim_model.graph.initializer[0])
    np.testing.assert_allclose(folded, a + b, rtol=1e-5, atol=1e-6)


def test_load_model_hydrates_external_data():
    # ``onnxsim.load_model`` must produce a ModelProto functionally identical
    # to ``onnx.load`` -- including for a model whose weights live in a
    # separate classic external-data file -- even though it hydrates that
    # data via onnxsim's memory-mapped TensorPool loader internally instead
    # of ``onnx.load_external_data_for_model``.
    a = np.random.rand(8, 8).astype(np.float32)
    initializer = onnx.numpy_helper.from_array(a, "a")
    node = onnx.helper.make_node("Identity", ["a"], ["y"])
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (8, 8))
    graph_def = onnx.helper.make_graph(
        [node],
        "test_load_model_hydrates_external_data",
        [],
        [out],
        initializer=[initializer],
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        assert os.path.exists(os.path.join(tmpdir, "model.data"))

        loaded = onnxsim.load_model(model_path)
        expected = onnx.load(model_path)

    assert len(loaded.graph.initializer) == 1
    assert loaded.graph.initializer[0].data_location == onnx.TensorProto.DEFAULT
    loaded_array = onnx.numpy_helper.to_array(loaded.graph.initializer[0])
    np.testing.assert_array_equal(loaded_array, a)
    assert loaded.SerializeToString() == expected.SerializeToString()


def test_model_info_size_counts_external_data_without_loading():
    # ModelInfo must report a model's size from external-data metadata, so a
    # model whose weights live on disk can be measured without loading them --
    # and the number must match what a fully-loaded model reports.
    from onnxsim import model_info

    w = np.random.rand(256, 256).astype(np.float32)  # 256 KiB of weights
    initializer = onnx.numpy_helper.from_array(w, "w")
    node = onnx.helper.make_node("Identity", ["w"], ["y"])
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (256, 256))
    graph_def = onnx.helper.make_graph(
        [node], "g", [], [out], initializer=[initializer]
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    full_size = model_info.ModelInfo(model).model_size
    # The weights dominate the reported size.
    assert full_size >= w.nbytes

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        meta_only = onnx.load(model_path, load_external_data=False)

    # The metadata-only model carries no raw tensor bytes...
    assert len(meta_only.graph.initializer[0].raw_data) == 0
    meta_size = model_info.ModelInfo(meta_only).model_size
    # ...yet its reported size still counts the external weights and matches the
    # fully-loaded size (to within the few bytes of external-data bookkeeping).
    assert meta_size >= w.nbytes
    assert abs(meta_size - full_size) < 1024


def test_model_info_size_does_not_double_count_subgraphs():
    # graph.ByteSize() already includes nested subgraphs, so the size must be
    # taken once at the top -- not summed per subgraph (which double-counted).
    from onnxsim import model_info

    def _branch(name, out_name):
        c = onnx.numpy_helper.from_array(np.zeros(1024, dtype=np.float32), name + "_c")
        n = onnx.helper.make_node("Identity", [name + "_c"], [out_name])
        return onnx.helper.make_graph(
            [n],
            name,
            [],
            [
                onnx.helper.make_tensor_value_info(
                    out_name, onnx.TensorProto.FLOAT, (1024,)
                )
            ],
            initializer=[c],
        )

    if_node = onnx.helper.make_node(
        "If",
        ["cond"],
        ["y"],
        then_branch=_branch("then", "ty"),
        else_branch=_branch("else", "ey"),
    )
    graph_def = onnx.helper.make_graph(
        [if_node],
        "g",
        [onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1024,))],
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )

    # With no external data, the reported size is exactly the graph's serialized
    # size -- the subgraph bytes are counted once, not twice.
    assert model_info.ModelInfo(model).model_size == model.graph.ByteSize()


def _make_lstm_model_with_dynamic_zero_state(
    initial_state_value: float = 0.0, hidden_size: int = 4, input_size: int = 3
):
    """Build the LSTM graph paddle2onnx emits for a zero initial state.

    The state's shape is [num_directions, batch_size, hidden_size], so a
    converter that cannot assume a static batch size materializes it as
    ``Shape -> Slice -> Concat -> Tile -> Transpose -> Slice``, reading the
    batch size off the input at runtime. ``X`` here is the ONNX LSTM layout
    [seq_length, batch_size, input_size] with a dynamic batch.
    """
    x = onnx.helper.make_tensor_value_info(
        "X", onnx.TensorProto.FLOAT, ["seq", "batch", input_size]
    )
    y = onnx.helper.make_tensor_value_info(
        "Y", onnx.TensorProto.FLOAT, ["seq", 1, "batch", hidden_size]
    )

    w = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    r = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    # The tiled state seed: [1, 2, hidden_size], holding initial_h and
    # initial_c stacked along axis 1.
    state = np.full((1, 2, hidden_size), initial_state_value, dtype=np.float32)
    initializers = [
        onnx.numpy_helper.from_array(w, "W"),
        onnx.numpy_helper.from_array(r, "R"),
        onnx.numpy_helper.from_array(state, "state"),
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
        onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "two"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "zero"),
        onnx.numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "ones2"),
    ]
    nodes = [
        onnx.helper.make_node("Shape", ["X"], ["shape"]),
        # shape[1:2] == [batch]
        onnx.helper.make_node("Slice", ["shape", "one", "two", "zero"], ["batch"]),
        onnx.helper.make_node("Concat", ["batch", "ones2"], ["repeats"], axis=0),
        # [1, 2, hidden] -> [batch, 2, hidden] -> [2, batch, hidden]
        onnx.helper.make_node("Tile", ["state", "repeats"], ["tiled"]),
        onnx.helper.make_node("Transpose", ["tiled"], ["states"], perm=[1, 0, 2]),
        onnx.helper.make_node("Slice", ["states", "zero", "one", "zero"], ["h0"]),
        onnx.helper.make_node("Slice", ["states", "one", "two", "zero"], ["c0"]),
        onnx.helper.make_node(
            "LSTM",
            ["X", "W", "R", "", "", "h0", "c0"],
            ["Y"],
            hidden_size=hidden_size,
        ),
    ]
    graph_def = onnx.helper.make_graph(
        nodes, "lstm_zero_state", [x], [y], initializer=initializers
    )
    return onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )


def test_eliminate_zero_lstm_initial_state():
    # Regression test for GitHub issue #314. An LSTM whose initial_h/initial_c
    # are provably all zeros must have those inputs unset (the ONNX spec
    # defaults them to zero), which makes the batch-dependent subgraph that
    # computed them dead. Without this the model keeps Shape/Slice/Concat/Tile
    # ops that downstream converters such as onnx2ncnn cannot handle.
    model = _make_lstm_model_with_dynamic_zero_state()
    sim_model, check_ok = onnxsim.simplify(
        model, test_input_shapes={"X": [5, 2, 3]}, check_n=3
    )
    assert check_ok

    op_types = [node.op_type for node in sim_model.graph.node]
    assert op_types == ["LSTM"]
    lstm = sim_model.graph.node[0]
    # initial_h/initial_c (indices 5 and 6) are gone, and the trailing empty
    # inputs are trimmed away rather than left dangling.
    assert all(name == "" for name in lstm.input[5:])

    # The simplified model still computes the same thing, for a batch size
    # other than the one used to probe shapes.
    for batch_size in (1, 2, 7):
        x = np.random.rand(5, batch_size, 3).astype(np.float32)
        (before,) = onnxsim.backend.run_model(model, {"X": x}).values()
        (after,) = onnxsim.backend.run_model(sim_model, {"X": x}).values()
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


def test_eliminate_zero_gru_initial_state_from_constant_of_shape():
    # The same elimination for GRU (which has initial_h but no initial_c), with
    # the zero state produced by a bare ConstantOfShape whose `value` attribute
    # is omitted and therefore defaults to zero.
    hidden_size, input_size = 4, 3
    x = onnx.helper.make_tensor_value_info(
        "X", onnx.TensorProto.FLOAT, ["seq", "batch", input_size]
    )
    y = onnx.helper.make_tensor_value_info(
        "Y", onnx.TensorProto.FLOAT, ["seq", 1, "batch", hidden_size]
    )
    initializers = [
        onnx.numpy_helper.from_array(
            np.random.rand(1, 3 * hidden_size, input_size).astype(np.float32), "W"
        ),
        onnx.numpy_helper.from_array(
            np.random.rand(1, 3 * hidden_size, hidden_size).astype(np.float32), "R"
        ),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "zero"),
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
        onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), "two"),
        onnx.numpy_helper.from_array(np.array([hidden_size], dtype=np.int64), "hidden"),
    ]
    nodes = [
        onnx.helper.make_node("Shape", ["X"], ["shape"]),
        onnx.helper.make_node("Slice", ["shape", "one", "two", "zero"], ["batch"]),
        # [1, batch, hidden_size]
        onnx.helper.make_node(
            "Concat", ["one", "batch", "hidden"], ["state_shape"], axis=0
        ),
        onnx.helper.make_node("ConstantOfShape", ["state_shape"], ["h0"]),
        onnx.helper.make_node(
            "GRU", ["X", "W", "R", "", "", "h0"], ["Y"], hidden_size=hidden_size
        ),
    ]
    graph_def = onnx.helper.make_graph(
        nodes, "gru_zero_state", [x], [y], initializer=initializers
    )
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )

    sim_model, check_ok = onnxsim.simplify(
        model, test_input_shapes={"X": [5, 2, 3]}, check_n=3
    )
    assert check_ok
    assert [node.op_type for node in sim_model.graph.node] == ["GRU"]
    assert all(name == "" for name in sim_model.graph.node[0].input[5:])


def test_keep_nonzero_lstm_initial_state():
    # The counterpart of the test above: an initial state that is *not* zero
    # carries real information and must be preserved verbatim.
    model = _make_lstm_model_with_dynamic_zero_state(initial_state_value=0.5)
    sim_model, check_ok = onnxsim.simplify(
        model, test_input_shapes={"X": [5, 2, 3]}, check_n=3
    )
    assert check_ok

    (lstm,) = [node for node in sim_model.graph.node if node.op_type == "LSTM"]
    assert len(lstm.input) == 7
    assert lstm.input[5] != "" and lstm.input[6] != ""
    # The state is still computed from the input's dynamic batch size.
    assert "Tile" in [node.op_type for node in sim_model.graph.node]


def test_target_opset_version_upgrades_model():
    # ``target_opset_version`` must upgrade the model's default-domain opset
    # during simplification, using onnx's version converter.
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph_def = onnx.helper.make_graph([node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 11)]
    )

    def _default_opset(m):
        return next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))

    assert _default_opset(model) == 11

    sim_model, check_ok = onnxsim.simplify(model, target_opset_version=18)
    assert check_ok
    assert _default_opset(sim_model) == 18


def test_target_opset_version_none_keeps_opset():
    # The default (None) must leave the model's opset version untouched.
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph_def = onnx.helper.make_graph([node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 11)]
    )

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    default_opset = next(
        o.version for o in sim_model.opset_import if o.domain in ("", "ai.onnx")
    )
    assert default_opset == 11


def test_perform_optimization_false():
    def _create_dummy_model():
        class MockModel(torch.nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = MockModel()
        dummy_input = torch.randn(1, 10)
        onnx_file = "dummy_model.onnx"
        torch.onnx.export(model, dummy_input, onnx_file, dynamo=False)
        return onnx_file

    onnx_model_path = _create_dummy_model()
    onnx_model = onnx.load(onnx_model_path)
    simple_model, _ = onnxsim.simplify(
        onnx_model, perform_optimization=False, skip_shape_inference=True
    )
    assert simple_model is not None


def _add_const_model(delta: float) -> onnx.ModelProto:
    """A minimal model computing ``y = x + delta`` (delta baked as initializer)."""
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])
    const = onnx.helper.make_tensor("c", onnx.TensorProto.FLOAT, [1], [delta])
    node = onnx.helper.make_node("Add", inputs=["x", "c"], outputs=["y"])
    graph = onnx.helper.make_graph([node], "g", [x], [y], initializer=[const])
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )


def test_compare_respects_check_tolerance():
    # Two models whose outputs differ by a known 5e-4 offset -- above the
    # default check tolerance (rtol=1e-4, atol=1e-5 -> ~1.1e-4 at |y|~1) but
    # well within a looser tolerance. This is the RF-DETR XLarge situation in
    # miniature: a correct-but-not-bit-identical simplified graph.
    from onnxsim import model_checking

    ori = _add_const_model(0.0)
    opt = _add_const_model(5e-4)
    data = {"x": np.ones((1, 4), dtype=np.float32)}

    # Strict default: flagged as changed.
    assert (
        model_checking.compare(opt, ori, n_times=1, input_data=data, verbose=False)
        is False
    )
    # Looser tolerance: accepted.
    assert (
        model_checking.compare(
            opt, ori, n_times=1, input_data=data, verbose=False, rtol=1e-2, atol=1e-2
        )
        is True
    )


def test_compare_input_fill_modes():
    # The random test inputs can be filled several ways. Capture what compare()
    # actually feeds the model by intercepting the backend run.
    from onnxsim import backend, model_checking

    model = _add_const_model(0.0)
    captured = {}

    def fake_run_model(m, inputs, custom_lib=None):
        captured["inputs"] = inputs
        return {"y": inputs["x"]}

    for fill, expected in (
        ("ones", np.ones((1, 4), dtype=np.float32)),
        ("zeros", np.zeros((1, 4), dtype=np.float32)),
        ("arange", np.arange(4, dtype=np.float32).reshape(1, 4)),
    ):
        import unittest.mock

        with unittest.mock.patch.object(backend, "run_model", fake_run_model):
            assert model_checking.compare(
                model, model, n_times=1, verbose=False, input_fill=fill
            )
        np.testing.assert_array_equal(captured["inputs"]["x"], expected)


def test_compare_rejects_unknown_input_fill():
    from onnxsim import model_checking

    with pytest.raises(ValueError):
        model_checking.compare(
            _add_const_model(0.0), _add_const_model(0.0), n_times=1, input_fill="bogus"
        )


def test_simplify_threads_input_fill(monkeypatch):
    # simplify() must forward input_fill into model_checking.compare.
    from onnxsim import model_checking

    captured = {}

    def fake_compare(model_opt, model_ori, n_times, *args, **kwargs):
        captured["input_fill"] = kwargs.get("input_fill")
        return True

    monkeypatch.setattr(model_checking, "compare", fake_compare)

    onnxsim.simplify(_add_const_model(0.0), check_n=1, input_fill="arange")
    assert captured == {"input_fill": "arange"}


def test_simplify_threads_check_tolerance(monkeypatch):
    # simplify() must forward check_rtol/check_atol into model_checking.compare.
    from onnxsim import model_checking

    captured = {}

    def fake_compare(model_opt, model_ori, n_times, *args, **kwargs):
        captured["rtol"] = kwargs.get("rtol")
        captured["atol"] = kwargs.get("atol")
        return True

    monkeypatch.setattr(model_checking, "compare", fake_compare)

    onnxsim.simplify(
        _add_const_model(0.0),
        check_n=1,
        check_rtol=0.123,
        check_atol=0.456,
    )
    assert captured == {"rtol": 0.123, "atol": 0.456}


def _mul_init_by_const_model() -> onnx.ModelProto:
    """A model ``y = (W * K) + x`` where ``W`` is an initializer and ``K`` is a
    ``Constant`` node. ``W * K`` is foldable only when initializers count as
    constants; ``x`` is a genuine graph input so the ``Add`` never folds."""
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])
    w = onnx.helper.make_tensor("W", onnx.TensorProto.FLOAT, [1, 3], [1.0, 2.0, 3.0])
    const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["K"],
        value=onnx.helper.make_tensor(
            "value", onnx.TensorProto.FLOAT, [1, 3], [2.0, 2.0, 2.0]
        ),
    )
    mul = onnx.helper.make_node("Mul", inputs=["W", "K"], outputs=["M"])
    add = onnx.helper.make_node("Add", inputs=["x", "M"], outputs=["y"])
    graph = onnx.helper.make_graph(
        [const_node, mul, add], "g", [x], [y], initializer=[w]
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10
    )


def test_initializers_as_constants_default_folds_initializer():
    # By default the initializer W is constant, so W * K is constant-folded away
    # and only the Add on the real input survives.
    model = _mul_init_by_const_model()
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Mul" not in op_types
    assert op_types.count("Add") == 1


def test_initializers_as_non_constants_keeps_initializer_node():
    # Treating initializers as non-constant leaves the Mul on the initializer in
    # the graph; only the Constant node (K) is still folded.
    model = _mul_init_by_const_model()
    sim_model, ok = onnxsim.simplify(model, initializers_as_constants=False)
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Mul" in op_types
    assert "Add" in op_types
    # W stays an initializer, and K has been folded into one too.
    init_names = {i.name for i in sim_model.graph.initializer}
    assert "W" in init_names


def _model_with_local_function() -> onnx.ModelProto:
    # A model whose main graph calls a single model-defined (local) function
    # ``custom.AddRelu`` -- authored via the ONNX text form so the FunctionProto
    # rides along in ``model.functions``.
    from onnx import parser

    return parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 18, "custom": 1]
        >
        agraph (float[N] X) => (float[N] Y) {
          Y = custom.AddRelu(X)
        }
        <
          domain: "custom",
          opset_import: ["": 18]
        >
        AddRelu (x) => (y) {
          one = Constant<value = float {1.0}>()
          shifted = Add(x, one)
          y = Relu(shifted)
        }
        """
    )


def test_inline_functions_disabled_by_default_keeps_function():
    # Without inline_functions the local function is preserved: the graph still
    # calls custom.AddRelu and the FunctionProto stays in model.functions.
    model = _model_with_local_function()
    sim_model, ok = onnxsim.simplify(model, check_n=0)
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types == ["AddRelu"]
    assert sim_model.graph.node[0].domain == "custom"
    assert [f.name for f in sim_model.functions] == ["AddRelu"]


def test_inline_functions_flattens_local_function():
    # With inline_functions the call site is replaced by the function body (Add +
    # Relu, with the folded constant), and the function is removed from the model
    # so the optimizer and constant folding can act on the flattened graph.
    model = _model_with_local_function()
    sim_model, ok = onnxsim.simplify(model, inline_functions=True, check_n=0)
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "AddRelu" not in op_types
    # The inlined body survives as plain ops (the +1 bias folds into an Add
    # initializer/Constant, followed by Relu).
    assert "Relu" in op_types
    assert len(sim_model.functions) == 0
