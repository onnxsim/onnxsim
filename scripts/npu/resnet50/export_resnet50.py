"""Export torchvision ResNet50 to ONNX (opset 17, static 1x3x224x224) and check it against torch."""
import os, sys
import numpy as np
import torch
import torchvision
import onnx
import onnxruntime as ort

out_dir = os.path.expanduser("~/models")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "resnet50_opset17.onnx")

try:
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    weights = "IMAGENET1K_V1"
except Exception as e:  # network or weight download failure: fall back to random init (structure is what matters here)
    print("pretrained weights unavailable, using random init:", type(e).__name__, str(e)[:120])
    model = torchvision.models.resnet50(weights=None)
    weights = "random"
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    ref = model(x).numpy()

torch.onnx.export(model, x, out, opset_version=17, input_names=["input"], output_names=["logits"],
                  dynamo=False)
m = onnx.load(out)
onnx.checker.check_model(m)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("weights:", weights, "| nodes:", len(m.graph.node), "| initializers:", len(m.graph.initializer))
print("op counts:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))

sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
got = sess.run(None, {"input": x.numpy()})[0]
print("max abs diff torch vs onnxruntime:", float(np.abs(got - ref).max()))
print("saved:", out, os.path.getsize(out) // 1024, "KiB")
