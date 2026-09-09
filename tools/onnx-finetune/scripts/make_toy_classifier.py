#!/usr/bin/env python3
"""Write a tiny MLP classifier to demo/test knowledge distillation against.

Unlike make_toy_model.py's regression toy (a 2-layer MLP fitting
``y = sum(x)``, used with ``--loss mse``), distillation needs a
classification-shaped model: soft-target cross-entropy only makes sense
comparing two logit distributions over the same classes. --hidden-dim
controls how big a model this produces -- give the teacher a bigger one than
the student (see ../README.md's distillation section) so there's actually
something for the student to learn from.
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

p = argparse.ArgumentParser()
p.add_argument("-o", "--output", default="toy_classifier.onnx")
p.add_argument("--input-dim", type=int, default=8)
p.add_argument("--hidden-dim", type=int, default=16)
p.add_argument("--num-classes", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

rng = np.random.default_rng(args.seed)


def linear(name, in_dim, out_dim):
    w = (rng.standard_normal((in_dim, out_dim)) * 0.1).astype(np.float32)
    b = np.zeros(out_dim, dtype=np.float32)
    return numpy_helper.from_array(w, f"{name}.weight"), numpy_helper.from_array(b, f"{name}.bias")


w1, b1 = linear("fc1", args.input_dim, args.hidden_dim)
w2, b2 = linear("fc2", args.hidden_dim, args.num_classes)

nodes = [
    helper.make_node("MatMul", ["input", "fc1.weight"], ["mm1"]),
    helper.make_node("Add", ["mm1", "fc1.bias"], ["add1"]),
    helper.make_node("Relu", ["add1"], ["relu1"]),
    helper.make_node("MatMul", ["relu1", "fc2.weight"], ["mm2"]),
    helper.make_node("Add", ["mm2", "fc2.bias"], ["logits"]),
]

graph = helper.make_graph(
    nodes,
    "toy_classifier",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, args.input_dim])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, args.num_classes])],
    initializer=[w1, b1, w2, b2],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, args.output)
print(
    f"wrote {args.output} (input_dim={args.input_dim} hidden_dim={args.hidden_dim} "
    f"num_classes={args.num_classes}, {sum(np.prod(i.dims) for i in model.graph.initializer):.0f} params)"
)
