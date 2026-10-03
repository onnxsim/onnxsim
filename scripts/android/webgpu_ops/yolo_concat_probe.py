"""Timing-only probe: how much wall-clock do the Concat dispatches cost *in situ*?  (outputs are wrong on purpose)

    python yolo_concat_probe.py in.onnx outX.onnx outZ.onnx

For every Concat(axis=1) that feeds a single 1x1 Conv, both variants feed the Conv from the Concat's first input only
(weights sliced along Cin, so both variants run the same, cheaper Conv).  Variant Z additionally keeps the Concat alive
(its output goes to a ReduceMean that becomes an extra graph output), variant X drops it.  time(Z) - time(X) is the cost
of the Concat dispatches plus 17 small ReduceMeans (read-only, ~half a Concat each, so this slightly over-counts).
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

from yolo_graph_opt import Graph, attr, shape_inference


def build(model, keep):
    G = Graph(model)
    G.refresh()
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    shapes = {}
    for v in list(shape_inference.infer_shapes(m2).graph.value_info) + list(model.graph.input):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    extra_outputs = []
    for cat in [n for n in G.nodes if n.op_type == "Concat"]:
        if attr(cat, "axis") != 1 or len(G.cons[cat.output[0]]) != 1 or G.cons[cat.output[0]][0] is None:
            continue
        conv = G.cons[cat.output[0]][0]
        if conv.op_type != "Conv" or conv.input[0] != cat.output[0] or list(attr(conv, "kernel_shape", [])) != [1, 1]:
            continue
        if attr(conv, "group", 1) != 1 or any(i not in shapes for i in cat.input):
            continue
        ca = shapes[cat.input[0]][1]
        w = G.arr(conv.input[1])
        wn = G.name(conv.input[1] + "_probe")
        G.add_init(wn, w[:, :ca])
        conv.input[0] = cat.input[0]
        conv.input[1] = wn
        if keep:
            red = G.name("probe_red")
            rn = helper.make_node("ReduceMean", [cat.output[0]], [red], keepdims=0, name=G.name("probe_mean"))
            G.nodes.insert(G.nodes.index(cat) + 1, rn)
            extra_outputs.append(red)
        else:
            G.nodes.remove(cat)
        G.refresh()
    out = G.finish()
    for o in extra_outputs:
        out.graph.output.append(helper.make_tensor_value_info(o, 1, None))
    return out


if __name__ == "__main__":
    src, x, z = sys.argv[1:4]
    onnx.save(build(onnx.load(src), False), x)
    onnx.save(build(onnx.load(src), True), z)
