"""Node shapes of a saved optimized graph from an ORT profile json: shapes.py model_opt.onnx profile.json out.json"""
import sys, json
import onnx
m = onnx.load(sys.argv[1], load_external_data=False)
names = {n.name: i for i, n in enumerate(m.graph.node)}
ev = json.load(open(sys.argv[2]))
res = {}
for e in ev:
    if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
        continue
    a = e.get("args", {})
    if "input_type_shape" not in a:
        continue
    nm = e["name"][: -len("_kernel_time")]
    if nm in names:
        res[nm] = {"op": a["op_name"], "in": [list(d.values())[0] for d in a["input_type_shape"]],
                   "out": [list(d.values())[0] for d in a["output_type_shape"]], "idx": names[nm],
                   "dtype": [list(d.keys())[0] for d in a["input_type_shape"]], "odtype": [list(d.keys())[0] for d in a["output_type_shape"]]}
print(len(res), "of", len(names), "nodes have shapes")
json.dump(res, open(sys.argv[3], "w"))
