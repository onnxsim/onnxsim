"""Build chains of stand-in ops to calibrate their cost: standin_cost.py OUTDIR
Models: X[shape] -> (Mul by 1 | Slice-channels | Concat x4) repeated N times -> Y. Compare N=2 and N=22 per op/shape to get ms per stand-in."""
import sys, os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
shapes = {"s256x32": [1, 256, 256, 32], "s128x256": [1, 128, 128, 256], "s64x512": [1, 64, 64, 512]}


def build(name, shape, kind, n):
    nodes, inits = [], []
    cur = "X"
    shp = list(shape)
    c = shp[-1]
    for i in range(n):
        o = f"t{i}"
        if kind == "mul":
            inits.append(numpy_helper.from_array(np.float32(1.0), f"one{i}"))
            nodes.append(helper.make_node("Mul", [cur, f"one{i}"], [o]))
        elif kind == "slice_cat":  # Slice channels c -> c/4 then Concat x4 back (the expand stand-in, 2 ops)
            for nm, arr in (("st", [0]), ("en", [c // 4]), ("ax", [len(shp) - 1])):
                inits.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), f"{nm}{i}"))
            nodes.append(helper.make_node("Slice", [cur, f"st{i}", f"en{i}", f"ax{i}"], [o + "s"]))
            nodes.append(helper.make_node("Concat", [o + "s"] * 4, [o], axis=len(shp) - 1))
        cur = o
    nodes.append(helper.make_node("Identity", [cur], ["Y"]))
    g = helper.make_graph(nodes, name, [helper.make_tensor_value_info("X", TensorProto.FLOAT, shp)],
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, shp)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, os.path.join(out, f"{name}.onnx"))


for sn, shp in shapes.items():
    for kind in ("mul", "slice_cat"):
        for n in (2, 22):
            build(f"sc_{sn}_{kind}_{n}", shp, kind, n)
print(os.listdir(out))
