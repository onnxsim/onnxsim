import tempfile
import torch
import onnx
import onnxruntime
import onnxsim
import os

from onnx import helper, TensorProto
from typing import Optional
from onnxsim.test_utils import export_simplify_and_check_by_python_api


def test_onnx_simplifier():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    export_simplify_and_check_by_python_api(MockModel(), torch.randn(1, 10))


def test_mg():
    class MG(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x, b):
            x = x.float()
            b = b.float()
            sh = x.shape
            x = x.view(sh[0], sh[1], -1)
            b = b.squeeze(-1)
            b = b.squeeze(-1)
            a = torch.matmul(b, x)
            preds = a.view(1, 100, sh[2], sh[3])
            return preds

    x = torch.randn([1, 256, 160, 184])
    b = torch.randn([100, 256, 1, 1])
    opt = export_simplify_and_check_by_python_api(MG(), (x, b), export_kwargs={"dynamo": True})
    sess = onnxruntime.InferenceSession(opt.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(out_names, { opt.graph.input[0].name: x.numpy(), opt.graph.input[1].name: b.numpy() })
    assert outs[0].shape == MG()(x, b).shape


def test_transformer():
    model = torch.nn.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1)
    model.to('cpu').to(torch.float32)
    model.eval()

    inputs = (
        torch.rand((100, 2, 256), dtype=torch.float32),
        torch.rand((15, 2, 256), dtype=torch.float32),
    )
    export_simplify_and_check_by_python_api(model, inputs, export_kwargs={"dynamo": True})


def test_upsample():
    import torch.nn.functional as F

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1_1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(16)
            self.maxpool = torch.nn.MaxPool2d(2, stride=2)
            self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.conv3_1 = torch.nn.Conv2d(32, 16, 3, padding=1)
            self.conv3_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
            self.conv3_3 = torch.nn.Conv2d(16, 3, 3, padding=1)
            self.bn3 = torch.nn.BatchNorm2d(3)

        def forward(self, x):
            x1 = F.relu(self.bn1(self.conv1_2(F.relu(self.conv1_1(x)))))
            x2 = self.maxpool(x1)
            xup = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3 = self.bn2(self.conv2_2(F.relu(self.conv2_1(xup))))
            x4 = F.relu(self.conv3_3(self.conv3_2(F.relu(self.conv3_1(x3)))))
            x5 = self.bn3(x4)
            return F.softsign(x5)


    inp = torch.rand(1, 3, 96, 96)
    net = Net()
    opt = export_simplify_and_check_by_python_api(net, (inp,), export_kwargs={"opset_version": 9})

    u_out = None
    for n in opt.graph.node:
        if n.op_type == "Upsample":
            u_out = n.output[0]
    assert u_out is not None
    u_info = None
    for v in opt.graph.value_info:
        if v.name == u_out:
            u_info = v
    assert u_info is not None
    assert [i.dim_value for i in v.type.tensor_type.shape.dim] == [1, 3, 96, 96]


def test_concat_squeese():
    # test for https://github.com/onnxsim/onnxsim/issues/46
    class Model(torch.nn.Module):
        def forward(self, x):
            # return torch.cat((torch.mean(x, 1, keepdim=True), torch.mean(x, 1, keepdim=True)), dim=1)
            return torch.cat((torch.mean(x, 1).unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    export_simplify_and_check_by_python_api(Model(), (torch.rand(20, 20),), export_kwargs={"opset_version": 9})


def test_trilinear():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, input_tensor):
            return torch.nn.functional.interpolate(input_tensor, scale_factor=[4, 4, 4], mode='trilinear')

    x = torch.rand(1, 8, 20, 120, 120)
    opt = export_simplify_and_check_by_python_api(
        Model(),
        (x,),
        export_kwargs={
            "opset_version": 11,
            "export_params": True,
        })
    sess = onnxruntime.InferenceSession(opt.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [i.name for i in sess.get_outputs()]
    outs = sess.run(out_names, { opt.graph.input[0].name: x.numpy() })
    assert outs[0].shape == (1, 8, 80, 480, 480)


def test_optional_type():
    class MyModule(torch.nn.Module):
        def forward(self, x: Optional[torch.Tensor]):
            return x is not None

    module = MyModule()
    inputs = (torch.tensor(1),)
    module = torch.jit.script(module, example_inputs=[inputs, (None,)])
    export_simplify_and_check_by_python_api(
        module,
        inputs,
        simplify_kwargs={"input_data": {"i": inputs[0].numpy()}},
        export_kwargs={"opset_version": 15, "input_names": ["i"]})


def test_ext():
    class MyMod(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.param = torch.nn.Parameter(torch.rand(1024, 1024, 1024, dtype=torch.float16))

        def forward(self, x):
            return self.param * x

    module = MyMod()
    inputs = (torch.tensor(1),)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_fn = os.path.join(tmpdirname, "tmp.onnx")
        torch.onnx.export(module, inputs, model_fn, dynamo=False, input_names=["x"])
        model = onnx.load(model_fn)
        sim_model, check_ok = onnxsim.simplify(model_fn, check_n=0)
        module = None
        assert check_ok


def test_unfoldable_const_node_keeps_topological_order():
    # A const node (all-constant inputs) that fails to fold must keep its
    # original position. Here SequenceEmpty is treated as a const node but can't
    # be folded into an initializer; its output feeds a non-const consumer
    # (SequenceInsert). If the failed node were moved to the end of the graph it
    # would land after its consumer and break topological sorting, making the
    # output fail onnx's checker (issues #238, #335, #352).
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    nodes = [
        helper.make_node("SequenceEmpty", [], ["seq"]),
        helper.make_node("SequenceInsert", ["seq", "x"], ["seq2"]),
        helper.make_node("ConcatFromSequence", ["seq2"], ["y"], axis=0, new_axis=0),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    # Output must remain a valid, topologically sorted graph.
    onnx.checker.check_model(sim_model)
    op_types = [n.op_type for n in sim_model.graph.node]
    assert op_types.index("SequenceEmpty") < op_types.index("SequenceInsert")


def test_if_with_const_cond_does_not_segfault():
    # An `If` whose condition is a constant, with branches that each just
    # produce a constant, used to crash simplify() with a segfault (exit 139).
    # onnxsim's constant folding turns the branch `Constant` nodes into subgraph
    # initializers, and the onnxoptimizer "eliminate_if_with_const_cond" pass
    # then dereferenced a null value while inlining the taken branch. simplify()
    # must instead complete and return a valid model (GitHub issue #452).
    tv = helper.make_tensor("tv", TensorProto.FLOAT, [2], [1.0, 2.0])
    fv = helper.make_tensor("fv", TensorProto.FLOAT, [2], [3.0, 4.0])
    then_b = helper.make_graph(
        [helper.make_node("Constant", [], ["to"], value=tv)], "tb", [],
        [helper.make_tensor_value_info("to", TensorProto.FLOAT, [2])])
    else_b = helper.make_graph(
        [helper.make_node("Constant", [], ["fo"], value=fv)], "fb", [],
        [helper.make_tensor_value_info("fo", TensorProto.FLOAT, [2])])
    if_node = helper.make_node(
        "If", ["c"], ["Y"], then_branch=then_b, else_branch=else_b)
    graph = helper.make_graph(
        [if_node], "g", [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        [helper.make_tensor("c", TensorProto.BOOL, [], [True])])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok
    onnx.checker.check_model(sim_model)
