import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto
D = "/mnt/data/cache/claude-work/models/demo/float/"
def conv_only(src, dst):
    m = onnx.load(src); g = m.graph
    inits = {i.name: i for i in g.initializer}
    done = {}
    def f16_init(name):
        if name in done: return done[name]
        t = inits[name]
        if t.data_type != TensorProto.FLOAT: return None
        a = numpy_helper.to_array(t).astype(np.float16)
        nn = name + "_h"; g.initializer.append(numpy_helper.from_array(a, nn)); done[name] = nn; return nn
    out = []
    k = 0
    for n in g.node:
        if n.op_type == "Conv":
            ins = list(n.input)
            w = f16_init(ins[1]) if ins[1] in inits else None
            b = (f16_init(ins[2]) if len(ins) > 2 and ins[2] in inits else None) if len(ins) > 2 else None
            ok = w is not None and (len(ins) < 3 or b is not None)
            if ok:
                xh = ins[0] + "__h%d" % k; yh = n.output[0] + "__h%d" % k
                out.append(helper.make_node("Cast", [ins[0]], [xh], to=TensorProto.FLOAT16, name="cin%d" % k))
                nn = onnx.NodeProto(); nn.CopyFrom(n); nn.ClearField("input"); nn.ClearField("output")
                nn.input.extend([xh, w] + ([b] if b else [])); nn.output.append(yh)
                out.append(nn)
                out.append(helper.make_node("Cast", [yh], [n.output[0]], to=TensorProto.FLOAT, name="cout%d" % k))
                k += 1; continue
        out.append(n)
    del g.node[:]; g.node.extend(out)
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used or any(o.name == i.name for o in g.output)]
    del g.initializer[:]; g.initializer.extend(keep)
    onnx.save(m, dst); return k
for k in ("sam_l0_enc", "rtdetr_pre"):
    print(k, "convs:", conv_only(D + k + ".onnx", f"m/{k}_f16conv.onnx"))
