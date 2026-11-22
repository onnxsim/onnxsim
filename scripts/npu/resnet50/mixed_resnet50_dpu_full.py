"""ResNet50 with every op on the DPU where possible: Conv(+Relu) and Gemm via dpu_layers/dpu_relu, Add(+Relu),
MaxPool and GlobalAveragePool via dpu_ops. Flatten is a view. Counts the nodes per route and compares with onnxruntime.
Needs dpu_layers.py, dpu_relu.py, dpu_ops.py and mixed_resnet50_dpu.py in the same directory."""
import os
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mixed_resnet50_dpu as w  # noqa: E402
import dpu_relu  # noqa: E402
import dpu_ops  # noqa: E402
from onnx import numpy_helper  # noqa: E402

base = w.base
MODEL = w.MODEL


def walk(model, x):
    cons = {}
    for n in model.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    fused = {}          # producer node name -> relu node output name (producer output becomes the relu result)
    passthrough = set()  # relu node names whose work is done by the producer
    for n in model.graph.node:
        if n.op_type in ("Conv", "Add") and len(cons.get(n.output[0], [])) == 1 and cons[n.output[0]][0].op_type == "Relu":
            r = cons[n.output[0]][0]
            fused[n.name] = r.output[0]
            passthrough.add(r.name)
    counts = {}
    env = {init.name: base.Tensor(numpy_helper.to_array(init).astype(np.float32)) for init in model.graph.initializer}
    env[model.graph.input[0].name] = base.Tensor(x)
    for node in model.graph.node:
        ins = [env[n] if n else None for n in node.input]
        a = base.attrs_of(node)
        if node.name in passthrough:
            outs = [ins[0]]
            route = "fused-into-producer"
        elif node.op_type == "Conv":
            relu = node.name in fused
            if relu:
                W = ins[1].numpy().astype(np.float32)
                B = ins[2].numpy().astype(np.float32).reshape(-1) if len(ins) > 2 and ins[2] is not None else np.zeros(W.shape[0], np.float32)
                y = dpu_relu.conv_r(W, B, ins[0].numpy().astype(np.float32), a["pads"][0], a["strides"][0], True)
                outs = [base.Tensor(y.astype(np.float32))]
                route = "DPU conv+relu"
            else:
                outs = [base.NPU_IMPL["Conv"](ins, a)]
                route = "DPU conv"
        elif node.op_type == "Add":
            relu = node.name in fused
            y = dpu_ops.add(ins[0].numpy().astype(np.float32), ins[1].numpy().astype(np.float32), relu)
            outs = [base.Tensor(y.astype(np.float32))]
            route = "DPU add+relu" if relu else "DPU add"
        elif node.op_type == "MaxPool":
            y = dpu_ops.maxpool(ins[0].numpy().astype(np.float32), a["kernel_shape"][0], a["strides"][0], a["pads"][0])
            outs = [base.Tensor(y.astype(np.float32))]
            route = "DPU maxpool"
        elif node.op_type == "GlobalAveragePool":
            y = dpu_ops.global_avg_pool(ins[0].numpy().astype(np.float32))
            outs = [base.Tensor(y.astype(np.float32))]
            route = "DPU gap"
        elif node.op_type in base.NPU_IMPL:
            outs = [base.NPU_IMPL[node.op_type](ins, a)]
            route = "NPU other (%s)" % node.op_type
        else:
            raise RuntimeError("no route for " + node.op_type)
        counts[route] = counts.get(route, 0) + 1
        for name, t in zip(node.output, outs):
            env[name] = t
    return env[model.graph.output[0].name].numpy(), counts


if __name__ == "__main__":
    model = onnx.load(MODEL)
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    ref = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(None, {model.graph.input[0].name: x})[0]
    walk(model, x)  # warm: compiles
    t = time.time()
    got, counts = walk(model, x)
    dt = time.time() - t
    print("routes:", counts, flush=True)
    print("warm walk %.2fs | max diff %.4g (ref max %.3g) | top-1 %d vs %d" % (
        dt, np.abs(got - ref.reshape(got.shape)).max(), np.abs(ref).max(), int(got.argmax()), int(ref.argmax())), flush=True)
