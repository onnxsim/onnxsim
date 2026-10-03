import onnx, numpy as np, onnxruntime as ort, sys
name=sys.argv[1]; txt=sys.argv[2]
rows=[l.split('\t') for l in open(txt).read().split('\n')]
m=onnx.load(f'/mnt/data/cache/claude-work/models/demo/float/{name}.onnx')
m2=onnx.shape_inference.infer_shapes(m)
vi={v.name:v for v in list(m2.graph.value_info)+list(m2.graph.output)}
del m.graph.output[:]
names=[r[3] for r in rows]
for t in names: m.graph.output.append(vi[t])
x=np.load(f'{name}_{sys.argv[3] if len(sys.argv)>3 else "noise"}_in.npy')
so=ort.SessionOptions(); so.log_severity_level=3
s=ort.InferenceSession(m.SerializeToString(),so,providers=['CPUExecutionProvider'])
res=s.run(names,{m.graph.input[0].name:x})
import os; os.makedirs(f'hostref_{name}',exist_ok=True)
for k,r in enumerate(res): np.asarray(r,np.float32).tofile(f'hostref_{name}/{k}.bin')
print(len(res))
