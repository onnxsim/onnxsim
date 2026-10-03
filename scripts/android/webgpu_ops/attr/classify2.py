"""Finer conv classes for the current ORT WebGPU paths.  classify2.py model_opt.onnx shapes.json
Writes <shapes>_classes.json (node name -> class) used by ablate.py, and prints GFLOP / MB / count per class.
Conv classes carry the code path they take on the current stack:
  W    Winograd F(2,3): 3x3 s1 d1 g1, Cin,Cout >= 64 and multiples of 4
  T    texture-weight direct conv (ORT_WEBGPU_CONV_TEXDIRECT=1): other dense non-1x1 convs, channels <= 1024
  M    1x1 stride-1 convs (MatMul shared-memory path)
  M2   1x1 stride-2 (Conv2dMM path)
  S    stem (Cin = 3)
  D    depthwise (vec4 kernel)
The resolution (output width) is appended so classes split per feature-map size."""
import sys, json, collections, math
import onnx

m = onnx.load(sys.argv[1], load_external_data=False)
sh = json.load(open(sys.argv[2]))
init = {i.name: list(i.dims) for i in m.graph.initializer}
prod = lambda l: math.prod(l) if l else 1
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
        k = at.get("kernel_shape") or w[2:]
        st = at.get("strides") or [1, 1]
        o = outs[0]  # NHWC
        cin, cout = w[1] * g, w[0]
        flops = 2 * prod(o) * (w[1] * w[2] * w[3])
        ow = o[2]
        if g == 1 and k == [1, 1]:
            kind = "M2" if st[0] == 2 else "M"
        elif g == 1 and cin == 3:
            kind = "S"
        elif g == 1 and k == [3, 3] and st[0] == 1 and cin >= 64 and cout >= 64 and cin % 4 == 0 and cout % 4 == 0:
            kind = "W"
        elif g == 1:
            kind = "T"
        elif g == cin == cout:
            kind = "D"
        else:
            kind = "G"
        res = f" ow{ow}"
        cls = f"Conv {kind} {k[0]}x{k[1]} s{st[0]} {cin}>{cout}{res}"
        if at.get("activation"):
            cls += "+act"
        if op == "NhwcFusedConv" and len(n.input) > 3:
            cls += "+res"
    rows.append(dict(name=n.name, idx=s["idx"], op=op, cls=cls, flops=flops, bytes=nbytes))
json.dump({r["name"]: r["cls"] for r in rows}, open(sys.argv[2].replace(".json", "_classes.json"), "w"))
json.dump(rows, open(sys.argv[2].replace(".json", "_rows.json"), "w"))
agg = collections.OrderedDict()
for r in rows:
    key = r["cls"] if r["op"] not in ("MatMul", "Gemm") else r["op"]
    a = agg.setdefault(key, [0, 0, 0])
    a[0] += 1; a[1] += r["flops"]; a[2] += r["bytes"]
print(f"{'class':46s} {'n':>3s} {'GFLOP':>7s} {'MB':>7s}")
for k, (c, f, b) in sorted(agg.items(), key=lambda kv: -(kv[1][1] * 1e-9 * 5 + kv[1][2] * 1e-6 * 0.1)):
    print(f"{k:46s} {c:3d} {f/1e9:7.2f} {b/1e6:7.1f}")
