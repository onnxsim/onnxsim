"""ResNet50 with Conv and Gemm on the Intel NPU DPU (see dpu_chunked.py). Other ops are as in mixed_resnet50.py.

Usage: python mixed_resnet50_dpu.py
Compares the logits with onnxruntime on one random 224x224 input and prints the walk time and the node counts.
"""
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dpu_chunked as dpu  # noqa: E402
import mixed_resnet50 as base  # noqa: E402

MODEL = base.MODEL


def npu_conv_dpu(ins, attrs):
    assert attrs.get("group", 1) == 1 and all(d == 1 for d in attrs.get("dilations", [1, 1]))
    pads = attrs.get("pads", [0, 0, 0, 0])
    strides = attrs.get("strides", [1, 1])
    assert pads == [pads[0]] * 4 and strides == [strides[0]] * 2
    x = ins[0].numpy().astype(np.float32)
    W = ins[1].numpy().astype(np.float32)
    y = dpu.conv(W, x, pads[0], strides[0])
    if len(ins) > 2 and ins[2] is not None:
        y = y + ins[2].numpy().astype(np.float32).reshape(1, -1, 1, 1)
    return base.Tensor(y.astype(np.float32))


def npu_gemm_dpu(ins, attrs):
    assert attrs.get("transB", 0) == 1 and attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
    x = ins[0].numpy().astype(np.float32)
    W = ins[1].numpy().astype(np.float32)
    bias = ins[2].numpy().astype(np.float32).reshape(-1) if len(ins) > 2 and ins[2] is not None else None
    return base.Tensor(dpu.gemm(W, x, bias).astype(np.float32))


base.NPU_IMPL["Conv"] = npu_conv_dpu
base.NPU_IMPL["Gemm"] = npu_gemm_dpu


def walk(model, x):
    from onnx import numpy_helper
    from tinygrad.nn.onnx import get_onnx_ops
    tg_ops = get_onnx_ops()
    env = {init.name: base.Tensor(numpy_helper.to_array(init).astype(np.float32)) for init in model.graph.initializer}
    env[model.graph.input[0].name] = base.Tensor(x)
    counts = {"npu": 0, "cpu": 0}
    for node in model.graph.node:
        ins = [env[n] if n else None for n in node.input]
        a = base.attrs_of(node)
        if node.op_type in base.NPU_IMPL:
            outs = [base.NPU_IMPL[node.op_type](ins, a)]
            counts["npu"] += 1
        else:
            impl = tg_ops[node.op_type]
            impl = impl.get(max(impl)) if isinstance(impl, dict) else impl
            res = impl(*ins, **a)
            outs = list(res) if isinstance(res, tuple) else [res]
            counts["cpu"] += 1
        for name, t in zip(node.output, outs):
            env[name] = t
    print("nodes on NPU: %d | on tinygrad CPU: %d" % (counts["npu"], counts["cpu"]), flush=True)
    return env[model.graph.output[0].name].numpy()


def main():
    model = onnx.load(MODEL)
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    t0 = time.time()
    got = walk(model, x)
    print("walk time %.1fs" % (time.time() - t0), flush=True)
    ref = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(None, {model.graph.input[0].name: x})[0]
    d = np.abs(got - ref)
    print("max abs diff vs onnxruntime %.4g (ref max %.4g) | top-1 %d vs %d, match %s"
          % (d.max(), np.abs(ref).max(), int(got.argmax()), int(ref.argmax()), int(got.argmax()) == int(ref.argmax())), flush=True)


if __name__ == "__main__":
    main()
