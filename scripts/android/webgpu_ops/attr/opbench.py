"""Isolated per-op timing at the real shapes of a saved optimized graph.
   opbench.py model_opt.onnx shapes.json outdir tag
Builds, for every distinct (domain, op, attrs, input shapes) of the standard-domain ops of the graph (plus com.microsoft ones), two models
with K=2 and K=32 independent copies of the node (same inputs, separate outputs, each output consumed by a Reshape->Slice[0:1] graph output so nothing is pruned
and nothing big is read back). per-op ms = (median(K=32) - median(K=2)) / 30 (includes one tiny keeper Slice per copy, ~0.02 ms).
Writes outdir/manifest.json: {model_file: {"op":..., "ins": [...], "n": occurrences in the graph}}."""
import sys, json, os, collections, math
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

mp, shp, outdir, tag = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)
m = onnx.load(mp, load_external_data=False)
sh = json.load(open(shp))
DT = {"float": TensorProto.FLOAT, "int64": TensorProto.INT64, "int32": TensorProto.INT32, "bool": TensorProto.BOOL}
init = {i.name: i for i in m.graph.initializer}
groups = collections.OrderedDict()
for n in m.graph.node:
    s = sh.get(n.name)
    if not s or n.domain not in ("", "com.microsoft") or n.op_type in ("NhwcFusedConv",):
        continue
    if any(x is None for x in s["in"]) or not s["out"]:
        continue
    key = (n.domain, n.op_type, json.dumps([[a.name, helper.get_attribute_value(a) if a.type not in (3, 4, 6, 7, 8, 9, 10) else str(a)] for a in n.attribute], default=str),
           json.dumps(s["in"]), json.dumps(s["dtype"]), tuple(i in init for i in n.input))
    groups.setdefault(key, []).append(n)
manifest = {}
# calibration entry: a tiny Relu, its per-copy cost is the harness floor (keeper Slice + graph output copy)
fake = helper.make_node("Relu", ["fx"], ["fy"], name="floor_relu")
sh["floor_relu"] = dict(op="Relu", **{"in": [[1, 4]], "out": [[1, 4]], "dtype": ["float"], "odtype": ["float"], "idx": -1})
groups[("", "Relu", "[]", "floor", "", ())] = [fake]
for gi, (key, nodes) in enumerate(groups.items()):
    n = nodes[0]
    s = sh[n.name]
    for K in (2, 32):
        gin, gout, gnodes, ginit = [], [], [], []
        names = []
        for k, nm in enumerate(n.input):
            if not nm:
                names.append("")
                continue
            if nm in init:
                t = onnx.TensorProto(); t.CopyFrom(init[nm]); t.name = f"c{k}"; ginit.append(t); names.append(f"c{k}")
            elif s["dtype"][k] != "float":
                arr = np.zeros(s["in"][k], dtype={"int64": np.int64, "int32": np.int32, "bool": np.bool_}.get(s["dtype"][k], np.int64))
                ginit.append(numpy_helper.from_array(arr, f"c{k}")); names.append(f"c{k}")
            else:
                gin.append(helper.make_tensor_value_info(f"i{k}", DT[s["dtype"][k]], s["in"][k])); names.append(f"i{k}")
        for c in range(K):
            outs = [f"o{c}_{j}" for j in range(len(n.output))]
            nn = helper.make_node(n.op_type, names, outs, domain=n.domain, name=f"op{c}")
            nn.attribute.extend(n.attribute)
            gnodes.append(nn)
            # keeper on the first output
            ginit.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), f"kf{c}"))
            for nmx, arr in (("ks", [0]), ("ke", [1]), ("ka", [0])):
                ginit.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), f"{nmx}{c}"))
            gnodes.append(helper.make_node("Reshape", [outs[0], f"kf{c}"], [f"kr{c}"], name=f"kr{c}"))
            gnodes.append(helper.make_node("Slice", [f"kr{c}", f"ks{c}", f"ke{c}", f"ka{c}"], [f"keep{c}"], name=f"ks{c}"))
            odt = DT.get(s["odtype"][0], TensorProto.FLOAT)
            gout.append(helper.make_tensor_value_info(f"keep{c}", odt, [1]))
        g = helper.make_graph(gnodes, "ob", gin, gout, ginit)
        opsets = [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
        mm = helper.make_model(g, opset_imports=opsets)
        mm.ir_version = 8
        fn = f"ob_{tag}_{gi}_K{K}.onnx"
        onnx.save(mm, os.path.join(outdir, fn))
    manifest[f"ob_{tag}_{gi}"] = dict(op=n.op_type, domain=n.domain, ins=s["in"], outs=s["out"], dtype=s["dtype"], n=len(nodes))
json.dump(manifest, open(os.path.join(outdir, f"manifest_{tag}.json"), "w"), indent=1)
print(len(manifest), "distinct op/shape combos")
