"""Turn every ONNX Constant node into an initializer (host-side, before export_cl.py on the phone).

tinygrad's OnnxRunner realizes a Constant node's value with a kernel on its CPU device, which needs a C compiler;
the Android CPython the phone-side export runs under has none. Initializers load as plain bytes instead.
  python const_to_init.py in.onnx out.onnx
"""
import sys
import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1])
keep = []
for n in m.graph.node:
  if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
    t = numpy_helper.to_array(n.attribute[0].t)
    m.graph.initializer.append(numpy_helper.from_array(t, n.output[0]))
  else: keep.append(n)
print(f"{len(m.graph.node) - len(keep)} Constant nodes -> initializers")
del m.graph.node[:]
m.graph.node.extend(keep)
onnx.save(m, sys.argv[2])
