import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnx.defs
import pytest
import torch
import torchvision as tv
from onnx import parser

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
    # The exporter emits the literal `1` as a Constant node; onnxsim now leaves
    # a genuine Constant node as-is rather than baking it into an initializer
    # (only a fold's *result* gets that treatment), so it survives alongside
    # the (unfoldable, `x` being a real input) Add.
    assert len(sim_model.graph.node) == 2


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
    # The exporter's literal constants (the `1.0` comparison threshold, and the
    # `3`s / `4` added to `x`) are each a genuine Constant node; onnxsim leaves
    # them as-is rather than baking them into initializers, so they now show up
    # as their own nodes (one at the top level, two in then_branch, one in
    # else_branch) alongside the previously-counted ops.
    assert len(sim_model.graph.node) == 4
    assert len(sim_model.graph.node[3].attribute[0].g.node) == 4
    assert len(sim_model.graph.node[3].attribute[1].g.node) == 2


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
    # The exporter emits the literal `2` as a Constant node, which onnxsim now
    # leaves as-is (see test_a_model_not_need_simplification).
    assert len(sim_model.graph.node) == 2


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
    # The exporter emits one shared Constant node for the literal `2` reused by
    # all four ops; onnxsim now leaves it as-is (see
    # test_a_model_not_need_simplification) instead of baking it into an
    # initializer, so it survives alongside them.
    assert len(sim_model.graph.node) == 5


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


def test_cli_large_model_save_fallback_mutates_in_place():
    # Regression test for the CLI's >2GB save fallback in onnx_simplifier.main()
    # (GitHub PR #730), reproduced exporting a real multi-gigabyte TTS model. Two
    # distinct bugs lived in the same try/except:
    #
    # 1. onnx.save() raises google.protobuf.message.EncodeError (not ValueError)
    #    once the serialized proto exceeds 2GB, so the external-data fallback
    #    never triggered and the crash propagated straight out of main().
    # 2. The fallback used to save a deepcopy of model_opt as external data while
    #    leaving the original model_opt (with data still inline) around for the
    #    subsequent model_info diff-printing step, which re-serializes model_opt
    #    and would hit that same EncodeError on the very model just saved.
    #
    # Both are exercised here without needing an actual >2GB fixture: onnx.save
    # is mocked to raise EncodeError on its first call (matching the real >2GB
    # failure mode), and model_info.print_simplifying_info is wrapped to capture
    # the exact model_opt object it receives so its data_location can be
    # inspected afterward.
    import sys

    from google.protobuf.message import EncodeError

    from onnxsim import model_info, onnx_simplifier

    # A weight large enough (4 MiB) to actually be moved to external storage --
    # onnx.save's external-data conversion leaves tensors below its size
    # threshold inline regardless of save_as_external_data, which would make
    # the data_location assertion below meaningless.
    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["": 17]
        >
        g (float[4,1024] x) => (float[4,1024] y)
        {
          y = MatMul(x, w)
        }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.random.rand(1024, 1024).astype(np.float32), name="w"
        )
    )
    onnx.checker.check_model(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, input_path)

        real_save = onnx.save
        call_count = 0

        def fake_save(proto, path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # The exact failure onnx.save raises past the 2GB protobuf
                # limit -- simulated here so the test stays fast and small.
                raise EncodeError("Message larger than 2GiB")
            return real_save(proto, path, *args, **kwargs)

        captured = {}
        real_print = model_info.print_simplifying_info

        def capturing_print(ori, opt):
            captured["opt"] = opt
            return real_print(ori, opt)

        argv = sys.argv
        try:
            onnx_simplifier.onnx.save = fake_save
            onnx_simplifier.model_info.print_simplifying_info = capturing_print
            sys.argv = ["onnxsim", input_path, output_path]
            onnx_simplifier.main()  # must not raise EncodeError (bug 1)
        finally:
            onnx_simplifier.onnx.save = real_save
            onnx_simplifier.model_info.print_simplifying_info = real_print
            sys.argv = argv

        assert call_count == 2  # the initial attempt, then the external-data save
        assert os.path.exists(output_path)
        assert os.path.exists(output_path + ".data")

        # model_opt was mutated in place, not deep-copied (bug 2): the object
        # model_info was handed after the fallback already carries external
        # data references rather than inline bytes.
        opt = captured["opt"]
        assert opt.graph.initializer[0].data_location == onnx.TensorProto.EXTERNAL


def test_cli_external_data_threshold_forces_external_data():
    # --external-data-threshold lets a model below the 2GB protobuf limit (and
    # below onnxsim's own 100MB default) still be forced to external data
    # without --save-as-external-data. Weights need to individually clear
    # onnx.save's own default per-tensor size_threshold (1024 bytes) too, or
    # onnx keeps them inline regardless (see
    # test_cli_large_model_save_fallback_mutates_in_place's own note on this).
    from onnxsim import onnx_simplifier

    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g () => (float[64,64] y)
        {
          y = Add(a, b)
        }
        """
    )
    model.graph.initializer.extend(
        [onnx.numpy_helper.from_array(a, "a"), onnx.numpy_helper.from_array(b, "b")]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, input_path)

        argv = sys.argv
        try:
            sys.argv = [
                "onnxsim",
                input_path,
                output_path,
                "--external-data-threshold",
                "1KB",
            ]
            onnx_simplifier.main()
        finally:
            sys.argv = argv

        assert os.path.exists(output_path + ".data")
        saved = onnx.load(output_path, load_external_data=False)
        assert saved.graph.initializer[0].data_location == onnx.TensorProto.EXTERNAL
        hydrated, _pool = onnxsim.load_model(output_path)
        folded = onnx.numpy_helper.to_array(hydrated.graph.initializer[0])
        np.testing.assert_allclose(folded, a + b, rtol=1e-5, atol=1e-6)
        # _pool mmaps output_path + ".data" -- on Windows an open mapping
        # blocks deleting the file, so it must not outlive this block.
        del _pool


def test_unset_optional_input():
    # A Resize with its unused optional "roi"/"scales" inputs left empty (""),
    # only "sizes" provided.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        test_unset_optional_input () => (float[1,3,4,4] y)
        <
          float[1,3,2,2] X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
          int64[4] sizes = {1, 3, 4, 4}
        >
        {
          y = Resize<mode = "linear">(X, , , sizes)
        }
        """
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
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        test_fold_deterministic_op () => (float[2,3] y)
        <
          float[2,3] a = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
          float[2,3] b = {0.6, 0.5, 0.4, 0.3, 0.2, 0.1}
        >
        {
          y = Add(a, b)
        }
        """
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
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        test_do_not_fold_random_op () => (float[2,3] y)
        {
          y = RandomUniform<shape = [2, 3], dtype = 1>()
        }
        """
    )

    sim_model, _ = onnxsim.simplify(model, check_n=0)
    assert len(sim_model.graph.node) == 1
    assert sim_model.graph.node[0].op_type == "RandomUniform"
    assert len(sim_model.graph.initializer) == 0


def test_do_not_fold_random_like_op():
    # RandomNormalLike is non-deterministic; it must not be folded even when its
    # input is a constant.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        test_do_not_fold_random_like_op () => (float[2,3] y)
        <float[2,3] x = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}>
        {
          y = RandomNormalLike(x)
        }
        """
    )

    sim_model, _ = onnxsim.simplify(model, check_n=0)
    assert any(n.op_type == "RandomNormalLike" for n in sim_model.graph.node)


def test_overwrite_input_shape_ignores_non_positive():
    # A non-positive value in overwrite_input_shapes must not be written to the
    # graph as a literal (e.g. 0) dimension; the original dimension should be
    # kept instead so the simplified model stays runnable (GitHub issue #237).
    model = parser.parse_model(
        """
        <
          opset_import: ["": 13]
        >
        test_overwrite_input_shape_ignores_non_positive (float[N,3,H,W] input) => (float[N,3,H,W] output)
        {
          output = Relu(input)
        }
        """
    )
    model.ir_version = onnx.IR_VERSION

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
    # / outputs while simplifying (GitHub issue #428). doc_string isn't part of
    # the ONNX text grammar, so it's set programmatically after parsing.
    model = parser.parse_model(
        """
        <
          opset_import: ["": 13]
        >
        test_preserve_doc_strings (float[1,4] X) => (float[1,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    model.ir_version = onnx.IR_VERSION
    model.doc_string = "model documentation"
    model.graph.doc_string = "graph documentation"
    model.graph.input[0].doc_string = "input documentation"
    model.graph.output[0].doc_string = "output documentation"

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert sim_model.doc_string == "model documentation"
    assert sim_model.graph.doc_string == "graph documentation"
    assert sim_model.graph.input[0].doc_string == "input documentation"
    assert sim_model.graph.output[0].doc_string == "output documentation"


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
    model = parser.parse_model(
        """
        <
          ir_version: 9,
          opset_import: ["": 13, "com.microsoft": 1]
        >
        g (uint8[1,3,16,16] A, uint8[1,3,16,16] B) => (float[1,3,16,16] out)
        <float s = {0.01}, uint8 zp = {128}>
        {
          C = com.microsoft.QLinearAdd(A, s, zp, B, s, zp, s, zp)
          out = DequantizeLinear(C, s, zp)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 3, 16, 16]


def test_qlinear_concat_shape_inference():
    model = parser.parse_model(
        """
        <
          ir_version: 9,
          opset_import: ["": 13, "com.microsoft": 1]
        >
        g (uint8[1,3,16,16] A, uint8[1,5,16,16] B) => (float[1,8,16,16] out)
        <float s = {0.01}, uint8 zp = {128}>
        {
          C = com.microsoft.QLinearConcat<axis = 1>(s, zp, A, s, zp, B, s, zp)
          out = DequantizeLinear(C, s, zp)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert _value_info_shape(sim_model, "C") == [1, 8, 16, 16]


def test_unknown_contrib_op_is_tolerated():
    # Registering schemas for the supported quantized ops must not make the
    # checker reject other, unregistered "com.microsoft" contrib operators.
    model = parser.parse_model(
        """
        <
          ir_version: 9,
          opset_import: ["": 13, "com.microsoft": 1]
        >
        g (uint8[1,3,16,16] A, uint8[1,3,16,16] B) => (float[1,3,16,16] out, uint8[1,3,16,16] D)
        <float s = {0.01}, uint8 zp = {128}>
        {
          C = com.microsoft.QLinearAdd(A, s, zp, B, s, zp, s, zp)
          D = com.microsoft.SomeUnknownContribOp(C)
          out = DequantizeLinear(C, s, zp)
        }
        """
    )
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

    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g () => (seq(float) seq)
        {
          seq = SequenceEmpty<dtype = 1>()
        }
        """
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
    model = parser.parse_model(
        """
        <
          ir_version: 6,
          opset_import: ["": 9]
        >
        batched_nms_trt (float[1,100,1,4] boxes, float[1,100,5] scores) => (int32[1,1] num_detections, float[1,20,4] nmsed_boxes, float[1,20] nmsed_scores, float[1,20] nmsed_classes)
        {
          num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = BatchedNMS_TRT<
            shareLocation = 1,
            backgroundLabelId = -1,
            numClasses = 5,
            topK = 100,
            keepTopK = 20,
            scoreThreshold = 0.3,
            iouThreshold = 0.5,
            isNormalized = 1,
            clipBoxes = 1
          >(boxes, scores)
        }
        """
    )
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
    model = parser.parse_model(
        """
        <
          ir_version: 6,
          opset_import: ["": 11]
        >
        g (float[1,100,1,4] boxes, float[1,100,5] scores) => (int32[1,1] num_detections)
        {
          boxes_id = Identity(boxes)
          num_detections = BatchedNMS_TRT<numClasses = 5, topK = 100, keepTopK = 20>(boxes_id, scores)
        }
        """
    )

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


def test_export_onnx_schemas_bridges_registry():
    # The reverse direction of ``import_onnx_schemas``: a schema onnxsim's
    # internal (statically linked) registry knows about -- registered here the
    # same way onnxsim's own built-in ONNX Runtime contrib-op schemas are --
    # is invisible to the Python ``onnx`` module's separate registry until
    # ``export_onnx_schemas`` copies it across.
    from onnxsim import onnx_simplifier

    op_type = "OnnxsimExportTestOp"
    domain = "onnxsim.export.test"

    C = onnx_simplifier.C
    assert not onnx.defs.has(op_type, domain=domain)

    C._register_schema(
        op_type,
        domain,
        1,
        "a test op",
        [("X", "the input", "T", 0, True, 1)],
        [("Y", "the output", "T", 0, True, 1)],
        [
            (
                "alpha",
                "slope",
                int(onnx.AttributeProto.FLOAT),
                False,
                onnx.AttributeProto(
                    name="alpha", f=0.1, type=onnx.AttributeProto.FLOAT
                ),
            )
        ],
        [("T", ["tensor(float)"], "Constrain to float tensors.")],
        False,
    )
    # Registering in onnxsim alone must not affect onnx's separate registry.
    assert not onnx.defs.has(op_type, domain=domain)

    exported = onnxsim.export_onnx_schemas()
    assert exported >= 1
    assert onnx.defs.has(op_type, domain=domain)

    schema = onnx.defs.get_schema(op_type, domain=domain)
    assert [p.name for p in schema.inputs] == ["X"]
    assert [p.name for p in schema.outputs] == ["Y"]
    assert "alpha" in schema.attributes

    # Idempotent: a second call exports nothing new for this op (it is
    # already known) and does not raise.
    onnxsim.export_onnx_schemas()
    assert onnx.defs.has(op_type, domain=domain)


def test_custom_op_with_registered_schema_is_simplified():
    # End-to-end: a model using a custom operator whose schema was registered via
    # ``onnx.defs.register_schema`` must simplify successfully -- the custom op is
    # preserved (with its attribute) and a redundant Identity feeding it is
    # eliminated -- instead of failing validation (GitHub issue #326).
    op_type = "OnnxsimCustomLeakyRelu"
    domain = "onnxsim.custom.ops"
    _register_custom_onnx_schema(op_type, domain)

    model = parser.parse_model(
        f"""
        <
          ir_version: 9,
          opset_import: ["": 13, "{domain}": 1]
        >
        custom_op_graph (float[1,3,8,8] X) => (float[1,3,8,8] Y)
        {{
          X_id = Identity(X)
          Y = {domain}.{op_type}<alpha = 0.1>(X_id)
        }}
        """
    )

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
    model = parser.parse_model(
        """
        <
          opset_import: ["": 13]
        >
        g (float[1,4] X) => (float[1,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    model.ir_version = onnx.IR_VERSION

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
        model = parser.parse_model(
            f"""
            <
              ir_version: 9,
              opset_import: ["": 13, "{domain}": 1]
            >
            shape_infer_graph (float[2,3] X) => (float[2,3,99] Y)
            {{
              t = {domain}.{op_type}<pad = 99>(X)
              Y = Add(t, t)
            }}
            """
        )

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
    # Nodes written in the ONNX text form (the `name` argument is omitted) and
    # operate on the non-constant graph input, so they survive simplification.
    model = parser.parse_model(
        """
        <
          opset_import: ["": 13]
        >
        test_nameless_nodes (float[1,4] X) => (float[1,4] Y)
        {
          t = Abs(X)
          Y = Relu(t)
        }
        """
    )
    model.ir_version = onnx.IR_VERSION
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
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        test_simplify_path_with_external_data () => (float[64,64] y)
        {
          y = Add(a, b)
        }
        """
    )
    model.graph.initializer.extend(
        [onnx.numpy_helper.from_array(a, "a"), onnx.numpy_helper.from_array(b, "b")]
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


def test_load_model_hydrates_classic_external_data():
    # onnxsim.load_model mmaps a model's classic ONNX external data (through
    # tensor_pool_bridge.h's LoadModelWithTensorPool) instead of using onnx's
    # own per-tensor loader -- verify it round-trips a model saved with
    # save_as_external_data=True back to plain in-memory tensors carrying the
    # original values, and that the returned TensorPool holds the same bytes.
    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g () => (float[64,64] y)
        {
          y = Add(a, b)
        }
        """
    )
    model.graph.initializer.extend(
        [onnx.numpy_helper.from_array(a, "a"), onnx.numpy_helper.from_array(b, "b")]
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
        # All the data really is external -- this exercises the mmap'd path,
        # not a no-op passthrough.
        assert os.path.exists(os.path.join(tmpdir, "model.data"))

        loaded, pool = onnxsim.load_model(model_path)

        # Read everything needed from the pool while it's still alive: its
        # classic-external-data entries mmap model.data, and on Windows an
        # open mapping blocks deleting the file, so `pool` must not outlive
        # this block's directory cleanup. The values captured here (ints,
        # strs, bytes) are plain copies, independent of the mapping.
        pool_len = len(pool)
        pool_names = set(pool.names())
        pool_bytes_a = pool.bytes("a")
        pool_bytes_b = pool.bytes("b")
        pool_dtype_a = pool.dtype("a")
        pool_shape_a = pool.shape("a")
        pool_hash_a = pool.content_hash("a")
        del pool

    for init in loaded.graph.initializer:
        assert init.data_location == onnx.TensorProto.DEFAULT
    values = {
        init.name: onnx.numpy_helper.to_array(init) for init in loaded.graph.initializer
    }
    np.testing.assert_allclose(values["a"], a)
    np.testing.assert_allclose(values["b"], b)

    assert pool_len == 2
    assert pool_names == {"a", "b"}
    assert pool_bytes_a == a.tobytes()
    assert pool_bytes_b == b.tobytes()
    assert pool_dtype_a == onnx.TensorProto.FLOAT
    assert pool_shape_a == [64, 64]
    assert len(pool_hash_a) == 64  # hex-encoded BLAKE3 digest


def test_load_model_hydrates_unnamed_attribute_tensor():
    # hydrate_all=True's Python-side hydration (_hydrate_graph_tensors_from_pool
    # in onnx_simplifier.py) re-derives each pooled tensor's key the same way
    # tensor_pool_bridge.h's ForEachTensor does in C++: a tensor's own name, or
    # -- for an unnamed node-attribute tensor -- a positional fallback
    # ("node<i>/attr<j>/t"). This only round-trips correctly if the two
    # independent implementations agree on that key, so exercise it directly
    # with a Constant node whose `value` tensor has no name of its own.
    #
    # onnx.save's external-data converter doesn't externalize attribute
    # tensors in the onnx version this repo pins (only initializers), so the
    # text/onnx.parser form can't produce this fixture -- the EXTERNAL
    # TensorProto has to be hand-built and pointed at a real data file.
    c = np.random.rand(64, 64).astype(np.float32)
    c_bytes = c.tobytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "attr.data")
        with open(data_path, "wb") as f:
            f.write(c_bytes)

        model = onnx.ModelProto()
        model.ir_version = 9
        model.opset_import.add(domain="", version=17)
        graph = model.graph
        graph.name = "g"
        graph.input.add(
            name="x",
            type=onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [64, 64]),
        )
        graph.output.add(
            name="y",
            type=onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [64, 64]),
        )

        const_node = graph.node.add()
        const_node.op_type = "Constant"
        const_node.output.append("cst")
        attr = const_node.attribute.add()
        attr.name = "value"
        attr.type = onnx.AttributeProto.TENSOR
        attr.t.data_type = onnx.TensorProto.FLOAT
        attr.t.dims.extend([64, 64])
        attr.t.data_location = onnx.TensorProto.EXTERNAL
        # attr.t.name deliberately left empty.
        for key, value in (
            ("location", data_path),
            ("offset", "0"),
            ("length", str(len(c_bytes))),
        ):
            entry = attr.t.external_data.add()
            entry.key = key
            entry.value = value

        add_node = graph.node.add()
        add_node.op_type = "Add"
        add_node.input.extend(["x", "cst"])
        add_node.output.append("y")

        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save(model, model_path)

        loaded, pool = onnxsim.load_model(model_path)
        pool_names = pool.names()
        # pool's entry mmaps attr.data inside tmpdir -- on Windows an open
        # mapping blocks deleting the file, so it must not outlive this
        # block's cleanup (see onnxsim.load_model's docstring).
        del pool

    assert pool_names == ["node0/attr0/t"]
    loaded_const = [n for n in loaded.graph.node if n.op_type == "Constant"][0]
    (value_attr,) = [a for a in loaded_const.attribute if a.name == "value"]
    assert value_attr.t.data_location == onnx.TensorProto.DEFAULT
    np.testing.assert_allclose(onnx.numpy_helper.to_array(value_attr.t), c)


def test_load_model_hydrate_all_false_leaves_tensors_external():
    # hydrate_all=False leaves the model's tensors as lazy EXTERNAL
    # references -- the pool already holds their bytes, so nothing is lost,
    # but the model itself needs an explicit hydrate to use those values.
    a = np.random.rand(64, 64).astype(np.float32)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g () => (float[64,64] y)
        {
          y = Identity(a)
        }
        """
    )
    model.graph.initializer.append(onnx.numpy_helper.from_array(a, "a"))

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        loaded, pool = onnxsim.load_model(model_path, hydrate_all=False)

        # See test_load_model_hydrates_classic_external_data's comment on
        # why `pool` must not outlive this block.
        pool_has_a = "a" in pool
        pool_bytes_a = pool.bytes("a")
        del pool

    assert loaded.graph.initializer[0].data_location == onnx.TensorProto.EXTERNAL
    assert loaded.graph.initializer[0].raw_data == b""
    assert pool_has_a
    np.testing.assert_allclose(
        np.frombuffer(pool_bytes_a, dtype=np.float32).reshape(64, 64), a
    )


def test_load_model_passes_through_inline_model():
    # A model with no external data at all must still load correctly (no
    # EXTERNAL tensors for LoadModelWithTensorPool to resolve) and the
    # returned pool is empty -- nothing needed resolving.
    model, a, b = _make_add_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        onnx.save(model, model_path)
        loaded, pool = onnxsim.load_model(model_path)

    assert loaded.graph.initializer[0].data_location == onnx.TensorProto.DEFAULT
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(loaded.graph.initializer[0]), a
    )
    assert len(pool) == 0


def test_load_model_dispatches_to_safetensors_archive():
    # A ".safetensors" path is treated as one of onnxsim's own
    # self-describing archives (export_safetensors's own format), not a
    # plain .onnx file with classic external data.
    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "model.safetensors")
        onnxsim.export_safetensors(model, archive_path)

        loaded, pool = onnxsim.load_model(archive_path)

    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(loaded.graph.initializer[0]), a
    )
    assert set(pool.names()) >= {"a", "b"}


def test_load_model_dispatches_to_gguf_archive():
    # Same dispatch, for onnxsim's own self-describing GGUF archives.
    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "model.gguf")
        onnxsim.export_gguf(model, archive_path)

        loaded, pool = onnxsim.load_model(archive_path)

    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(loaded.graph.initializer[0]), a
    )
    assert set(pool.names()) >= {"a", "b"}


@pytest.mark.parametrize(
    "export_fn,import_fn,ext",
    [
        (onnxsim.export_safetensors, onnxsim.import_safetensors, "safetensors"),
        (onnxsim.export_gguf, onnxsim.import_gguf, "gguf"),
    ],
)
def test_export_archive_leaves_model_unchanged(export_fn, import_fn, ext):
    # export_safetensors/export_gguf used to cross the whole model (tensor
    # bytes included) into C++ via a single SerializeToString()/ParseFromString()
    # round trip -- the same double-encode pattern load_model's hydrate_all=True
    # path had (see that function's docstring). The fix pulls each tensor's
    # raw_data out into a separate dict up front (so the accompanying
    # model-structure serialize is cheap) and puts it back once the C++ call
    # returns -- this must be transparent to the caller: `model` compares
    # byte-equal to a snapshot taken before the call, even though it was
    # mutated and restored in between.
    model, a, b = _make_add_model()
    before = onnx.ModelProto()
    before.CopyFrom(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, f"model.{ext}")
        export_fn(model, archive_path)
        assert model == before

        loaded = import_fn(archive_path)

    onnx.checker.check_model(loaded)
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(loaded.graph.initializer[0]), a
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(loaded.graph.initializer[1]), b
    )


@pytest.mark.parametrize(
    "export_fn,import_fn",
    [
        (onnxsim.export_safetensors, onnxsim.import_safetensors),
        (onnxsim.export_gguf, onnxsim.import_gguf),
    ],
)
def test_export_archive_roundtrips_unnamed_attribute_tensor(export_fn, import_fn):
    # The extraction side of the same fix (_extract_graph_tensors_to_dict)
    # must key an unnamed node-attribute tensor the same positional way
    # (`node<i>/attr<j>/t`) its hydration counterpart does, or the C++ side's
    # external_tensor_bytes lookup misses and the tensor is silently dropped
    # from the archive instead of exported.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[2,2] x) => (float[2,2] y)
        {
          c = Constant<value = float[2,2] {0.0, 1.0, 2.0, 3.0}>()
          y = Add(x, c)
        }
        """
    )
    (const_node,) = [n for n in model.graph.node if n.op_type == "Constant"]
    (value_attr,) = [a for a in const_node.attribute if a.name == "value"]
    value_attr.t.ClearField("name")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "model.archive")
        export_fn(model, archive_path)
        loaded = import_fn(archive_path)

    onnx.checker.check_model(loaded)
    (loaded_const,) = [n for n in loaded.graph.node if n.op_type == "Constant"]
    (loaded_value,) = [a for a in loaded_const.attribute if a.name == "value"]
    assert loaded_value.t.data_location == onnx.TensorProto.DEFAULT
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(loaded_value.t),
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )


@skip_in_ci()
@pytest.mark.skipif(sys.platform == "win32", reason="resource.getrusage is POSIX-only")
def test_simplify_path_peak_memory_stays_near_model_size():
    # Regression test for the root cause documented in
    # bench/RESULTS_synthetic_decoder_oom.md / bench/TODO_large_decoder_submodule_oom.md:
    # Simplify() used to unconditionally deep-copy its whole input model into a
    # mutable working copy (`sim_model = model` in onnxsim.cpp), so peak RSS for a
    # large external-data model was ~1.9-2x its own size. SimplifyConsumeInput
    # (wired into SimplifyPath, which onnxsim.simplify(path, check_n=0)'s fast path
    # calls) moves tensor data into the working copy instead of copying it,
    # bringing peak RSS down to approximately 1x model size.
    #
    # This only shows up **above the 2GB protobuf limit**: below it, SimplifyPath's
    # own C++ side still has to inline-serialize the *output* model into one
    # contiguous buffer (onnxsim.cpp's `needs_external_data` only trips past
    # kProtobufSizeLimit), and that serialize buffer's own size dominates enough to
    # mask the fix at smaller scales -- measured empirically while writing this
    # test: at 196 MiB-1.5 GiB, the pre-fix vs post-fix delta was a near-constant
    # ~28 MiB regardless of model size (not a ratio), whereas at ~2.2+ GiB (crossing
    # the threshold) it was a clean ~2.06x (pre-fix) vs ~1.07x (post-fix) at every
    # size tried. So this reuses bench/decoder_oom_repro.py's own generator sized to
    # land just above that threshold (11 layers, ~2.3 GiB) rather than
    # reimplementing the same decoder-block shape here at a size that was never
    # actually measured against pre-fix behavior.
    #
    # This is why it's gated behind @skip_in_ci() like this file's other
    # multi-hundred-MB+ tests (e.g. test_model_larger_than_2gb): building and
    # simplifying a ~2.3 GiB model takes real time and memory. Run locally with
    # CI unset (skip_in_ci only skips when CI is a truthy env var).
    bench_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench")
    sys.path.insert(0, bench_dir)
    try:
        import decoder_oom_repro
    finally:
        sys.path.remove(bench_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, total_bytes = decoder_oom_repro.gen(
            tmpdir,
            layers=11,
            hidden=decoder_oom_repro.DEFAULT_HIDDEN,
            ffn=decoder_oom_repro.DEFAULT_FFN,
            seq_len=8,
            layout="single",
            seed=0,
        )
        total_mib = total_bytes / 1024 / 1024

        # Measured in a fresh child process: resource.getrusage(RUSAGE_SELF)'s
        # ru_maxrss is a process-lifetime high-water mark, so it must be read from
        # a process that only ever does this one simplify() call.
        child_script = os.path.join(tmpdir, "child.py")
        with open(child_script, "w") as f:
            f.write(
                "import resource, sys\n"
                "import onnx, onnxsim\n"
                "model_opt, ok = onnxsim.simplify(sys.argv[1], check_n=0)\n"
                "assert ok\n"
                "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
            )
        proc = subprocess.run(
            [sys.executable, child_script, model_path],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"child process failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        peak_kib = int(proc.stdout.strip().splitlines()[-1])

    peak_mib = peak_kib / 1024
    # Empirically: ~1.07x post-fix, ~2.06x pre-fix at this size (see comment
    # above). 1.5x sits cleanly between the two.
    assert peak_mib < total_mib * 1.5, (
        f"peak RSS ({peak_mib:.0f} MiB) for a {total_mib:.0f} MiB external-data "
        "model is too high -- this is the double-materialization regression "
        "SimplifyConsumeInput fixed (see bench/RESULTS_synthetic_decoder_oom.md)"
    )


def _make_add_model():
    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g () => (float[64,64] y)
        {
          y = Add(a, b)
        }
        """
    )
    model.graph.initializer.extend(
        [onnx.numpy_helper.from_array(a, "a"), onnx.numpy_helper.from_array(b, "b")]
    )
    return model, a, b


def test_output_path_fast_path_saves_directly_and_skips_reload():
    # Regression test for a real, if secondary, inefficiency documented in
    # bench/RESULTS_synthetic_decoder_oom.md: on the check_n=0 fast path,
    # simplify() used to always call onnx.load(fast_out_path) (full data inline)
    # purely to satisfy its return contract, even when the caller's very next
    # step is to save the result again (as onnxsim's own CLI does).
    # ``output_path`` lets the C++ core write the final result directly, so the
    # returned model can stay structure-only. (That doc's real headline fix --
    # the dominant peak-memory cost, inside the C++ core's own working copy --
    # is separate, in onnxsim.cpp's SimplifyConsumeInput; this reload is real
    # but turned out not to be what was driving the original OOM report.)
    #
    # The C++ core only actually externalizes a saved model's data past the 2GB
    # protobuf limit (onnxsim.cpp's SimplifyPath: ``needs_external_data =
    # model.ByteSizeLong() >= kProtobufSizeLimit``), so a small test model's
    # output is always inline regardless of output_path -- there is no
    # multi-GB fixture to assert "raw_data is empty" against here. What *is*
    # testable at this scale is the mechanism itself: with output_path set,
    # simplify() must read the result back with ``load_external_data=False``
    # instead of the eager default, which is exactly the reload this test
    # guards against reintroducing.
    from onnxsim import onnx_simplifier

    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, model_path)

        real_load = onnx.load
        load_calls = []

        def spying_load(path, *args, **kwargs):
            load_calls.append((path, args, kwargs))
            return real_load(path, *args, **kwargs)

        try:
            onnx_simplifier.onnx.load = spying_load
            sim_model, check_ok = onnxsim.simplify(
                model_path, check_n=0, output_path=output_path
            )
        finally:
            onnx_simplifier.onnx.load = real_load

        assert check_ok
        # The result was saved directly to output_path by simplify() itself.
        assert os.path.exists(output_path)
        saved, _pool = onnxsim.load_model(output_path)
        assert len(saved.graph.node) == 0
        assert len(saved.graph.initializer) == 1
        folded = onnx.numpy_helper.to_array(saved.graph.initializer[0])
        np.testing.assert_allclose(folded, a + b, rtol=1e-5, atol=1e-6)

    # Exactly one load, of output_path itself (never a throwaway temp file),
    # with load_external_data explicitly disabled -- the actual fix.
    assert len(load_calls) == 1
    (loaded_path, load_args, load_kwargs) = load_calls[0]
    assert loaded_path == output_path
    assert load_kwargs.get("load_external_data") is False


def test_output_path_requires_str_model():
    model, _, _ = _make_add_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "out.onnx")
        with pytest.raises(ValueError, match="output_path"):
            onnxsim.simplify(model, output_path=output_path)


def test_output_path_off_fast_path_still_saves_full_model():
    # check_n > 0 takes the slow path (it needs the full model in memory
    # regardless, to run the correctness check), so output_path can't skip the
    # reload there -- but the file must still end up saved, and the returned
    # model must carry real data (unlike the fast-path case above), since a
    # caller who asked for check_n > 0 is presumably going to use it.
    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, model_path)

        sim_model, check_ok = onnxsim.simplify(
            model_path, check_n=1, output_path=output_path
        )

        assert check_ok
        assert os.path.exists(output_path)
        saved, _pool = onnxsim.load_model(output_path)
        folded = onnx.numpy_helper.to_array(saved.graph.initializer[0])
        np.testing.assert_allclose(folded, a + b, rtol=1e-5, atol=1e-6)

    # Unlike the fast-path case, the returned model actually has data: check_n > 0
    # already required materializing it, so there is nothing left to save by
    # deferring the load.
    assert len(sim_model.graph.initializer[0].raw_data) > 0
    folded_returned = onnx.numpy_helper.to_array(sim_model.graph.initializer[0])
    np.testing.assert_allclose(folded_returned, a + b, rtol=1e-5, atol=1e-6)


def test_output_path_falls_back_to_external_data_past_2gb():
    # Same >2GB save fallback the CLI relies on (see
    # test_cli_large_model_save_fallback_mutates_in_place), exercised here for
    # output_path's own fallback save at the end of simplify() -- reached when
    # output_path is set but the fast path doesn't apply (check_n > 0 here).
    from google.protobuf.message import EncodeError

    from onnxsim import onnx_simplifier

    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, model_path)

        real_save = onnx.save
        call_count = 0

        def fake_save(proto, path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EncodeError("Message larger than 2GiB")
            return real_save(proto, path, *args, **kwargs)

        try:
            onnx_simplifier.onnx.save = fake_save
            sim_model, check_ok = onnxsim.simplify(
                model_path, check_n=1, output_path=output_path
            )
        finally:
            onnx_simplifier.onnx.save = real_save

        assert check_ok
        assert call_count == 2  # the initial (faked-failing) attempt, then the fallback
        assert os.path.exists(output_path)
        assert os.path.exists(output_path + ".data")
        saved, _pool = onnxsim.load_model(output_path)
        folded = onnx.numpy_helper.to_array(saved.graph.initializer[0])
        np.testing.assert_allclose(folded, a + b, rtol=1e-5, atol=1e-6)
        # _pool's classic-external-data entries mmap output_path + ".data";
        # on Windows an open mapping blocks deleting the file, so it must not
        # outlive this block's directory cleanup.
        del _pool


def test_output_path_external_data_threshold_default_keeps_small_model_inline():
    # The default external_data_threshold (100MB) leaves a small model inline
    # with no extra argument needed, matching pre-existing behavior.
    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, model_path)

        sim_model, check_ok = onnxsim.simplify(
            model_path, check_n=1, output_path=output_path
        )

        assert check_ok
        assert not os.path.exists(output_path + ".data")
        saved = onnx.load(output_path, load_external_data=False)
        assert saved.graph.initializer[0].data_location == onnx.TensorProto.DEFAULT


def test_output_path_external_data_threshold_forces_external_data():
    # A low external_data_threshold forces external data even for a model far
    # below the 2GB protobuf limit and the 100MB default -- exercised at this
    # scale by passing an explicit threshold. Weights need to individually
    # clear onnx.save's own default per-tensor size_threshold (1024 bytes)
    # too, or onnx keeps them inline regardless of save_as_external_data
    # (_make_add_model's 64x64 float32 initializers, 16KB each, clear it).
    model, a, b = _make_add_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        output_path = os.path.join(tmpdir, "out.onnx")
        onnx.save(model, model_path)

        sim_model, check_ok = onnxsim.simplify(
            model_path,
            check_n=1,
            output_path=output_path,
            external_data_threshold="1KB",
        )

        assert check_ok
        assert os.path.exists(output_path + ".data")
        saved = onnx.load(output_path, load_external_data=False)
        assert saved.graph.initializer[0].data_location == onnx.TensorProto.EXTERNAL
        hydrated, _pool = onnxsim.load_model(output_path)
        folded = onnx.numpy_helper.to_array(hydrated.graph.initializer[0])
        np.testing.assert_allclose(folded, a + b, rtol=1e-5, atol=1e-6)
        # _pool mmaps output_path + ".data" -- on Windows an open mapping
        # blocks deleting the file, so it must not outlive this block.
        del _pool


def test_model_info_size_counts_external_data_without_loading():
    # ModelInfo must report a model's size from external-data metadata, so a
    # model whose weights live on disk can be measured without loading them --
    # and the number must match what a fully-loaded model reports.
    from onnxsim import model_info

    w = np.random.rand(256, 256).astype(np.float32)  # 256 KiB of weights
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g () => (float[256,256] y)
        {
          y = Identity(w)
        }
        """
    )
    model.graph.initializer.append(onnx.numpy_helper.from_array(w, "w"))

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
    w = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    r = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    # The tiled state seed: [1, 2, hidden_size], holding initial_h and
    # initial_c stacked along axis 1.
    state = np.full((1, 2, hidden_size), initial_state_value, dtype=np.float32)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        lstm_zero_state (float[seq,batch,{input_size}] X) => (float[seq,1,batch,{hidden_size}] Y)
        <int64[1] one = {{1}}, int64[1] two = {{2}}, int64[1] zero = {{0}}, int64[2] ones2 = {{1, 1}}>
        {{
          shape = Shape(X)
          # shape[1:2] == [batch]
          batch = Slice(shape, one, two, zero)
          repeats = Concat<axis = 0>(batch, ones2)
          # [1, 2, hidden] -> [batch, 2, hidden] -> [2, batch, hidden]
          tiled = Tile(state, repeats)
          states = Transpose<perm = [1, 0, 2]>(tiled)
          h0 = Slice(states, zero, one, zero)
          c0 = Slice(states, one, two, zero)
          Y = LSTM<hidden_size = {hidden_size}>(X, W, R, , , h0, c0)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(w, "W"),
            onnx.numpy_helper.from_array(r, "R"),
            onnx.numpy_helper.from_array(state, "state"),
        ]
    )
    return model


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
    w = np.random.rand(1, 3 * hidden_size, input_size).astype(np.float32)
    r = np.random.rand(1, 3 * hidden_size, hidden_size).astype(np.float32)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        gru_zero_state (float[seq,batch,{input_size}] X) => (float[seq,1,batch,{hidden_size}] Y)
        <int64[1] zero = {{0}}, int64[1] one = {{1}}, int64[1] two = {{2}}, int64[1] hidden = {{{hidden_size}}}>
        {{
          shape = Shape(X)
          batch = Slice(shape, one, two, zero)
          # [1, batch, hidden_size]
          state_shape = Concat<axis = 0>(one, batch, hidden)
          h0 = ConstantOfShape(state_shape)
          Y = GRU<hidden_size = {hidden_size}>(X, W, R, , , h0)
        }}
        """
    )
    model.graph.initializer.extend(
        [onnx.numpy_helper.from_array(w, "W"), onnx.numpy_helper.from_array(r, "R")]
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
    model = parser.parse_model(
        """
        <
          opset_import: ["": 11]
        >
        g (float[1,4] X) => (float[1,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    model.ir_version = onnx.IR_VERSION

    def _default_opset(m):
        return next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))

    assert _default_opset(model) == 11

    sim_model, check_ok = onnxsim.simplify(model, target_opset_version=18)
    assert check_ok
    assert _default_opset(sim_model) == 18


def test_target_opset_version_none_keeps_opset():
    # The default (None) must leave the model's opset version untouched.
    model = parser.parse_model(
        """
        <
          opset_import: ["": 11]
        >
        g (float[1,4] X) => (float[1,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    model.ir_version = onnx.IR_VERSION

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
    onnx_model, _pool = onnxsim.load_model(onnx_model_path)
    simple_model, _ = onnxsim.simplify(
        onnx_model, perform_optimization=False, skip_shape_inference=True
    )
    assert simple_model is not None


def _add_const_model(delta: float) -> onnx.ModelProto:
    """A minimal model computing ``y = x + delta`` (delta baked as initializer)."""
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[1,4] x) => (float[1,4] y)
        <float[1] c = {{{delta}}}>
        {{
          y = Add(x, c)
        }}
        """
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
    return parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[1,3] x) => (float[1,3] y)
        <float[1,3] W = {1.0, 2.0, 3.0}>
        {
          K = Constant<value = float[1,3] {2.0, 2.0, 2.0}>()
          M = Mul(W, K)
          y = Add(x, M)
        }
        """
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
    # the graph; K is a Constant node already, so folding leaves it untouched.
    model = _mul_init_by_const_model()
    sim_model, ok = onnxsim.simplify(model, initializers_as_constants=False)
    assert ok
    op_types = [n.op_type for n in sim_model.graph.node]
    assert "Mul" in op_types
    assert "Add" in op_types
    # W stays an initializer.
    init_names = {i.name for i in sim_model.graph.initializer}
    assert "W" in init_names


def test_fold_not_purely_from_initializer_becomes_constant_node():
    # W * K -- W an initializer, K a Constant node -- is still folded away (both
    # are constant), but since the fold consumed a Constant node's value rather
    # than tracing back purely to graph initializers, the result must itself be
    # materialized as a Constant node rather than baked into a plain
    # initializer, so a value the graph actually computed stays visually
    # distinct from literal weight data.
    model = _mul_init_by_const_model()
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    init_names = {i.name for i in sim_model.graph.initializer}
    assert "M" not in init_names
    (m_node,) = [n for n in sim_model.graph.node if "M" in n.output]
    assert m_node.op_type == "Constant"
    value = onnx.numpy_helper.to_array(m_node.attribute[0].t)
    np.testing.assert_array_equal(value, np.array([[2.0, 4.0, 6.0]], dtype=np.float32))


def test_fold_purely_from_initializer_stays_initializer():
    # A * B, where A and B are both plain initializers, is a fold rooted purely
    # in initializer data (no Constant node anywhere upstream), so it must still
    # collapse into a plain initializer, exactly as before this behavior was
    # made to depend on provenance.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        test_fold_purely_from_initializer_stays_initializer () => (float[3] y)
        <float[3] A = {1.0, 2.0, 3.0}, float[3] B = {4.0, 5.0, 6.0}>
        {
          y = Mul(A, B)
        }
        """
    )
    sim_model, ok = onnxsim.simplify(model)
    assert ok
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1
    assert sim_model.graph.initializer[0].name == "y"


def _model_with_local_function() -> onnx.ModelProto:
    # A model whose main graph calls a single model-defined (local) function
    # ``custom.AddRelu`` -- authored via the ONNX text form so the FunctionProto
    # rides along in ``model.functions``.
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
