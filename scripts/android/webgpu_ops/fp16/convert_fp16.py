import sys, onnx, numpy as np
from onnx import numpy_helper, TensorProto, helper
from onnxconverter_common import float16

def to_fp16(src, dst):
    m = onnx.load(src)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False)
    onnx.save(m16, dst, save_as_external_data=False)

def weight_only(src, dst):
    """Conv/MatMul/Gemm float weight initializers stored as fp16 + runtime Cast (initializer kept as graph input so ORT cannot constant-fold the Cast back to fp32)."""
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    targets = set()
    for n in g.node:
        if n.op_type in ("Conv", "MatMul", "Gemm") and len(n.input) > 1 and n.input[1] in inits:
            targets.add(n.input[1])
    new_nodes = []
    for name in targets:
        t = inits[name]
        if t.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(t).astype(np.float16)
        t16 = numpy_helper.from_array(arr, name + "_f16")
        g.initializer.remove(t)
        g.initializer.append(t16)
        g.input.append(helper.make_tensor_value_info(name + "_f16", TensorProto.FLOAT16, list(arr.shape)))
        new_nodes.append(helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT, name="cast_" + name))
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    if m.ir_version < 4: m.ir_version = 8
    onnx.save(m, dst)
    return len(new_nodes)

D = "/mnt/data/cache/claude-work/models/demo/float/"
srcs = {"resnet50": "/mnt/data/cache/claude-work/models/resnet50/resnet50.onnx", "yolo11n": D + "yolo11n.onnx", "sam_l0_enc": D + "sam_l0_enc.onnx", "rtdetr_pre": D + "rtdetr_pre.onnx"}
for k, v in srcs.items():
    try:
        to_fp16(v, f"m/{k}_f16.onnx"); print(k, "f16 ok")
    except Exception as e:
        print(k, "f16 FAIL", repr(e)[:200])
    try:
        print(k, "weight-only casts:", weight_only(v, f"m/{k}_w16.onnx"))
    except Exception as e:
        print(k, "w16 FAIL", repr(e)[:200])
