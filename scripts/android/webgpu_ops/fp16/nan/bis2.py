import onnx, sys, numpy as np
from onnx import helper, TensorProto
name, tag = sys.argv[1], sys.argv[2]
ref = onnx.load(f'/mnt/data/cache/claude-work/models/demo/float/{name}.onnx')
m = onnx.load(f'm/{name}_{tag}.onnx')
have = {o for n in m.graph.node for o in n.output}
want = []
for n in ref.graph.node:
    if n.op_type in ('Constant', 'Shape'): continue
    t = n.output[0]
    if t in have: want.append((n.op_type, n.name, t))
del m.graph.output[:]
new = []
for k, (op, nm, t) in enumerate(want):
    c = f'{t}__dbg'
    new.append(helper.make_node('Cast', [t], [c], to=TensorProto.FLOAT, name=f'dbgcast{k}'))
    m.graph.output.append(helper.make_tensor_value_info(c, TensorProto.FLOAT, None))
m.graph.node.extend(new)
onnx.save(m, f'm/{name}_{tag}_bis.onnx')
open(f'{name}_{tag}_bis.txt', 'w').write('\n'.join(f'{k}\t{op}\t{nm}\t{t}' for k, (op, nm, t) in enumerate(want)))
print(len(want), 'outputs')
