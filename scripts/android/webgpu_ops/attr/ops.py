"""List op classes of a saved optimized graph: counts, shapes (from value_info/initializers), bytes moved.
   python ops.py model_opt.onnx"""
import sys, collections
import onnx, numpy as np
from onnx import numpy_helper

m = onnx.load(sys.argv[1], load_external_data=False)
g = m.graph
shape = {}
for v in list(g.value_info) + list(g.input) + list(g.output):
    t = v.type.tensor_type
    if t.HasField("shape"):
        shape[v.name] = [d.dim_value if d.HasField("dim_value") else -1 for d in t.shape.dim]
for i in g.initializer:
    shape[i.name] = list(i.dims)
attrs = lambda n: {a.name: (list(a.ints) if a.ints else (a.i if a.type == 2 else (a.s.decode() if a.type == 3 else None))) for a in n.attribute}
cnt = collections.Counter()
rows = []
for idx, n in enumerate(g.node):
    key = (n.domain, n.op_type)
    cnt[key] += 1
    ins = [shape.get(i) for i in n.input]
    outs = [shape.get(o) for o in n.output]
    rows.append((idx, n.domain, n.op_type, n.name, ins, outs, attrs(n)))
print("nodes", len(g.node))
for k, v in cnt.most_common():
    print(f"{v:5d} {k[0] or 'onnx'}::{k[1]}")
if len(sys.argv) > 2:
    for r in rows:
        if sys.argv[2] in (r[2], "all"):
            print(r[0], r[1] or "onnx", r[2], r[4], "->", r[5], r[6])
