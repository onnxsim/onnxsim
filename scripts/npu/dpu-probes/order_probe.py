"""Ordering probe: the P2 graph (main chain + internal side branch, C=16) with the node order changed, sizes unchanged.
Order A: main chain first. Order B: side branch first. Lists literal, non-relocated words that differ between A and B.
Compile only. Run on H8 after reloc.py and litdiff.py."""
import os
import subprocess
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
sys.path.insert(0, os.path.expanduser("~"))
import litdiff as L

OV = os.path.expanduser("~/openvino_git/bin/intel64/Release")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/ze-loader/lib") + ":" + OV)
D = os.path.expanduser("~/npu-dpu/roadmap/order"); os.makedirs(D, exist_ok=True)
C = 16


def init(name, shape):
    return numpy_helper.from_array((np.arange(int(np.prod(shape))).reshape(shape) * 0.01).astype(np.float16), name)


def compile_order(name, side_first):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8, 16, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 16, 16, 16])
    main = [helper.make_node("Conv", ["x", "W1"], ["m"], kernel_shape=[1, 1]),
            helper.make_node("Conv", ["m", "W2"], ["p"], kernel_shape=[1, 1])]
    side = [helper.make_node("Conv", ["x", "Ws"], ["s"], kernel_shape=[1, 1]),
            helper.make_node("Conv", ["s", "Wt"], ["q"], kernel_shape=[1, 1])]
    add = [helper.make_node("Add", ["p", "q"], ["y"])]
    nodes = (side + main if side_first else main + side) + add
    inits = [init("W1", (16, 8, 1, 1)), init("W2", (16, 16, 1, 1)), init("Ws", (C, 8, 1, 1)), init("Wt", (16, C, 1, 1))]
    g = helper.make_graph(nodes, "order", [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 9
    p = os.path.join(D, name + ".onnx"); onnx.save(m, p)
    blob = p[:-5] + ".blob"
    r = subprocess.run([os.path.join(OV, "compile_tool"), "-m", p, "-d", "NPU", "-o", blob, "-ip", "FP16", "-op", "FP16"],
                       env=ENV, cwd=OV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return blob


a_blob = compile_order("order_A", False)
b_blob = compile_order("order_B", True)
wa, ra = L.sec_words(a_blob), L.reloc_words(a_blob)
wb, rb = L.sec_words(b_blob), L.reloc_words(b_blob)
print("== literal words that differ between order A (main first) and order B (side first)", flush=True)
for name in sorted(set(wa) & set(wb)):
    a, b = wa[name], wb[name]
    n = min(len(a), len(b))
    d = [j for j in range(n) if a[j] != b[j] and j not in ra.get(name, set()) and j not in rb.get(name, set())]
    print("  %-24s words %d/%d  literal diffs %d" % (name, len(a), len(b), len(d)), flush=True)
    for j in d[:20]:
        print("     word %4d  A 0x%08x  B 0x%08x" % (j, a[j], b[j]), flush=True)
