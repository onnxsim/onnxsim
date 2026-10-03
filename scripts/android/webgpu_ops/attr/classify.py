"""Classify the nodes of a saved optimized graph into op classes with FLOPs and bytes.
   classify.py model_opt.onnx shapes.json  -> prints a table and writes classes.json {node_name: class}"""
import sys, json, collections
import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1], load_external_data=False)
sh = json.load(open(sys.argv[2]))
init = {i.name: list(i.dims) for i in m.graph.initializer}
prod = lambda l: __import__("math").prod(l) if l else 1
rows = []
for n in m.graph.node:
    s = sh.get(n.name)
    if not s:
        continue
    at = {a.name: (list(a.ints) if a.ints else (a.i if a.type == 2 else None)) for a in n.attribute}
    ins, outs = s["in"], s["out"]
    op = n.op_type
    cls, flops = op, 0
    nbytes = 4 * (sum(prod(x) for x in ins if x is not None) + sum(prod(o) for o in outs if o is not None))
    if op in ("Conv", "NhwcFusedConv"):
        w = init.get(n.input[1])
        g = at.get("group", 1)
        k = at.get("kernel_shape")
        st = at.get("strides") or [1, 1]
        o = outs[0]
        flops = 2 * prod(o) * (w[1] * w[2] * w[3])
        if g == 1 and k == [1, 1]:
            cls = "Conv 1x1" + (" s2" if st[0] == 2 else "")
        elif g == 1:
            cls = f"Conv {k[0]}x{k[1]} dense" + (" s2" if st[0] == 2 else "")
        elif g == w[0]:
            cls = f"Conv dw {k[0]}x{k[1]}" + (" s2" if st[0] == 2 else "")
        else:
            cls = f"Conv grouped(g={g}) {k[0]}x{k[1]}"
        if at.get("activation"):
            cls += "+act"
    elif op in ("MatMul", "Gemm"):
        a, b = ins[0], ins[1]
        flops = 2 * prod(outs[0]) * (a[-1] if op == "MatMul" else a[-1])
        cls = f"{op} {a}x{b}"
    rows.append(dict(name=n.name, idx=s["idx"], op=op, cls=cls, flops=flops, bytes=nbytes, ins=ins, outs=outs))
json.dump({r["name"]: r["cls"] for r in rows}, open(sys.argv[2].replace(".json", "_classes.json"), "w"))
agg = collections.OrderedDict()
for r in rows:
    key = r["cls"] if not r["cls"].startswith(("MatMul", "Gemm")) else r["op"] + " " + str(r["ins"][0][-2:]) + "x" + str(r["ins"][1][-1:])
    a = agg.setdefault(key, [0, 0, 0])
    a[0] += 1; a[1] += r["flops"]; a[2] += r["bytes"]
print(f"{'class':45s} {'n':>4s} {'GFLOP':>8s} {'MB moved':>9s}")
for k, (c, f, b) in sorted(agg.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
    print(f"{k:45s} {c:4d} {f/1e9:8.2f} {b/1e6:9.1f}")
print("total GFLOP", sum(a[1] for a in agg.values()) / 1e9, "MB", sum(a[2] for a in agg.values()) / 1e6)
json.dump(rows, open(sys.argv[2].replace(".json", "_rows.json"), "w"))
