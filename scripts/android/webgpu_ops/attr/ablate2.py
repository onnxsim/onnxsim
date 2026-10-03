"""Replace selected nodes of a saved optimized graph with cheap shape-preserving stand-ins (for timing ablations).

   ablate.py model_opt.onnx shapes.json out.onnx SELECTOR [--identity]

SELECTOR: comma separated list of  op=<op_type>  |  cls=<class substring from classify.py>  |  name=<node name>  |  idx=<a-b>
A node is replaced only if it has one output with a known static shape:
  - same shape in/out  -> Mul(x, 1)  (one elementwise pass; with --identity: Identity)
  - NHWC-like (rank>=3, batch equal, spatial dims divide): strided Slice for stride, channel Slice for fewer channels, Concat of copies (+Slice) for more
  - otherwise          -> Reshape(x,[-1]) -> Slice/Pad to the output size -> Reshape(out shape)  (flat; one pass over the smaller side)
The stand-in therefore costs about one copy; subtract nothing -- the report shows both 'removed' and the pure copy cost class by class.
Values are not meaningful (timing only) but stay finite: Mul by 1 and slices/zero pads of finite data."""
import sys, json, math
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

model_path, shapes_path, out_path, selector = sys.argv[1:5]
use_identity = "--identity" in sys.argv
m = onnx.load(model_path, load_external_data=False)
sh = json.load(open(shapes_path))
cls = json.load(open(shapes_path.replace(".json", "_classes.json")))
sels = [s.split("=", 1) for s in selector.split(",")]


def matches(i, n):
    c = cls.get(n.name, "")
    for k, v in sels:
        if k == "op" and n.op_type == v:
            return True
        if k == "clsx" and v == c:
            return True
        if k == "cls" and v in c:
            return True
        if k == "name" and n.name == v:
            return True
        if k == "idx":
            a, b = (int(x) for x in v.split("-"))
            if a <= i <= b:
                return True
    return False


init_names = {i.name for i in m.graph.initializer}
keepers = []
new_nodes, new_inits, replaced = [], [], 0
uid = 0
for i, n in enumerate(m.graph.node):
    s = sh.get(n.name)
    if not (matches(i, n) and s and len(n.output) == 1 and s["out"] and s["out"][0] and s["in"] and s["in"][0] and n.input[0]
            and s.get("odtype", ["float"])[0] == "float" and "float" in s.get("dtype", ["float"])):
        new_nodes.append(n)
        continue
    # main input: the dynamic input whose shape equals the output shape (largest first), else input 0
    # (an extra input without a profile shape entry, e.g. the fused residual Z of NhwcFusedConv, is kept alive but never the main input)
    dyn_all = [k for k in range(len(n.input)) if n.input[k] and n.input[k] not in init_names and (k >= len(s["in"]) or s["in"][k] is not None)]
    dyn = [k for k in dyn_all if k < len(s["in"])]
    cand = [k for k in dyn if s["in"][k] == s["out"][0]]
    mk = cand[0] if cand else (dyn[0] if dyn else 0)
    if len(cand) > 1:
        mk = max(cand, key=lambda k: math.prod(s["in"][k]))
    elif not cand and dyn:
        mk = max(dyn, key=lambda k: math.prod(s["in"][k]))
    a, b = s["in"][mk], s["out"][0]
    x, y = n.input[mk], n.output[0]
    # keep the other dynamic inputs alive (otherwise ORT prunes their producers): Reshape -> Slice[0:1] -> graph output
    for k in dyn_all:
        if k == mk:
            continue
        uidk = len(keepers) + 1
        kp = f"keep{uidk}"
        new_inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), kp + "_flat"))
        for nm, arr in (("st", [0]), ("en", [1]), ("ax", [0])):
            new_inits.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), f"{kp}_{nm}"))
        new_nodes.append(helper.make_node("Reshape", [n.input[k], kp + "_flat"], [kp + "_f"], name=kp + "_r"))
        new_nodes.append(helper.make_node("Slice", [kp + "_f", kp + "_st", kp + "_en", kp + "_ax"], [kp], name=kp + "_s"))
        keepers.append((kp, (s.get("dtype", ["float"] * len(n.input)) + ["float"] * 8)[k]))
    uid += 1
    p = f"abl{uid}"
    if a == b:
        if use_identity:
            new_nodes.append(helper.make_node("Identity", [x], [y], name=p))
        else:
            one = numpy_helper.from_array(np.float32(1.0), p + "_one")
            new_inits.append(one)
            new_nodes.append(helper.make_node("Mul", [x, p + "_one"], [y], name=p))
    elif len(a) == len(b) >= 3 and a[0] == b[0] and (a[1] % b[1] == 0 and a[2] % b[2] == 0):
        # NHWC-like: optional spatial stride (strided Slice), then channel reduce (Slice) or expand (Concat of copies + Slice)
        ca, cb = a[-1], b[-1]
        sy, sx = a[1] // b[1], a[2] // b[2]
        cur = x
        def ini(nm, arr):
            new_inits.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), f"{p}_{nm}"))
            return f"{p}_{nm}"
        if sy != 1 or sx != 1 or cb < ca:
            keep = min(ca, cb)
            st, en, ax, stp = ini("st", [0, 0, 0]), ini("en", [b[1] * sy, b[2] * sx, keep]), ini("ax", [1, 2, len(a) - 1]), ini("stp", [sy, sx, 1])
            out_name = y if cb <= ca else p + "_s"
            new_nodes.append(helper.make_node("Slice", [cur, st, en, ax, stp], [out_name], name=p + "_sl"))
            cur = out_name
        if cb > ca:
            reps = -(-cb // ca)
            cat = p + "_cat"
            new_nodes.append(helper.make_node("Concat", [cur] * reps, [cat], name=p + "_cc", axis=len(a) - 1))
            if reps * ca == cb:
                new_nodes[-1].output[0] = y
            else:
                st2, en2, ax2 = ini("st2", [0]), ini("en2", [cb]), ini("ax2", [len(a) - 1])
                new_nodes.append(helper.make_node("Slice", [cat, st2, en2, ax2], [y], name=p + "_sl2"))
    else:
        na, nb = math.prod(a), math.prod(b)
        shp = numpy_helper.from_array(np.array([-1], dtype=np.int64), p + "_flat")
        oshp = numpy_helper.from_array(np.array(b, dtype=np.int64), p + "_oshape")
        new_inits += [shp, oshp]
        new_nodes.append(helper.make_node("Reshape", [x, p + "_flat"], [p + "_f"], name=p + "_r1"))
        cur = p + "_f"
        if nb <= na:
            for nm, arr in (("st", [0]), ("en", [nb]), ("ax", [0])):
                new_inits.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), f"{p}_{nm}"))
            new_nodes.append(helper.make_node("Slice", [cur, p + "_st", p + "_en", p + "_ax"], [p + "_s"], name=p + "_sl"))
        else:
            new_inits.append(numpy_helper.from_array(np.array([0, nb - na], dtype=np.int64), p + "_pads"))
            new_nodes.append(helper.make_node("Pad", [cur, p + "_pads"], [p + "_s"], name=p + "_pad", mode="constant"))
        new_nodes.append(helper.make_node("Reshape", [p + "_s", p + "_oshape"], [y], name=p + "_r2"))
    replaced += 1

del m.graph.node[:]
m.graph.node.extend(new_nodes)
m.graph.initializer.extend(new_inits)
DT = {"float": TensorProto.FLOAT, "int64": TensorProto.INT64, "int32": TensorProto.INT32, "bool": TensorProto.BOOL, "float16": TensorProto.FLOAT16}
for kp, dt in keepers:
    m.graph.output.append(helper.make_tensor_value_info(kp, DT.get(dt, TensorProto.FLOAT), [1]))
# value_info of replaced outputs stays valid; drop stale ones for safety
onnx.save(m, out_path)
print(f"{replaced} nodes replaced -> {out_path}")
