import onnx, numpy as np, onnxruntime as ort, sys, collections, json
name = sys.argv[1]; inp = sys.argv[2]   # noise | photo
src = f'/mnt/data/cache/claude-work/models/demo/float/{name}.onnx'
m = onnx.load(src)
def lcg(n, seed):
    out = np.empty(n, np.float32); r = np.uint32(seed)
    # same LCG as bench.cc: rng = rng*1664525+1013904223 ; u = (rng>>8)/2^24
    r = int(seed)
    for j in range(n):
        r = (r * 1664525 + 1013904223) & 0xffffffff
        out[j] = (r >> 8) / 16777216.0
    return out
i0 = m.graph.input[0]
shape = [d.dim_value for d in i0.type.tensor_type.shape.dim]
et = i0.type.tensor_type.elem_type
n = int(np.prod(shape))
if inp == 'photo':
    x = np.load(f'{name}_photo_in.npy')
elif inp == 'noise':
    u = lcg(n, 12345)
    x = (u * 255).astype(np.uint8) if et == 2 else u
    x = x.reshape(shape)
else:
    p = np.load('/mnt/data/cache/claude-work/models/photo640.npy')
    print('photo', p.shape, p.dtype, p.min(), p.max())
    # resize to model input H,W
    raise SystemExit('use mkphoto')
    a = p if p.ndim == 3 else p[0]
    if a.shape[0] == 3: a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8: a = np.clip(a * (255 if a.max() <= 1.5 else 1), 0, 255).astype(np.uint8)
    im = Image.fromarray(a).resize((shape[2], shape[1]))
    a = np.asarray(im)
    x = a.astype(np.float32) / 255 if et == 1 else a
    x = x.reshape(shape)
if inp == 'noise': np.save(f'{name}_{inp}_in.npy', x)
names = []
for nd in m.graph.node:
    for o in nd.output:
        if o and o not in [g.name for g in m.graph.output]:
            m.graph.output.append(onnx.helper.make_value_info(o, onnx.TypeProto())) if False else None
# add outputs with unknown type via shape inference
m2 = onnx.shape_inference.infer_shapes(m)
vi = {v.name: v for v in list(m2.graph.value_info) + list(m2.graph.output)}
outs = []
for nd in m.graph.node:
    if nd.op_type in ('Constant', 'Shape'): continue
    o = nd.output[0]
    if o in vi and vi[o].type.tensor_type.elem_type in (1,):
        outs.append((nd, o))
del m.graph.output[:]
for nd, o in outs: m.graph.output.append(vi[o])
so = ort.SessionOptions(); so.log_severity_level = 3
s = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
res = s.run([o for _, o in outs], {i0.name: x})
rows = []
for (nd, o), r in zip(outs, res):
    rows.append(dict(op=nd.op_type, name=nd.name, out=o, maxabs=float(np.abs(r).max()), finite=bool(np.isfinite(r).all()), shape=list(r.shape)))
json.dump(rows, open(f'{name}_{inp}_ranges.json', 'w'))
by = collections.defaultdict(float)
for r in rows: by[r['op']] = max(by[r['op']], r['maxabs'])
print(name, inp, 'per-op max |output|:')
for k, v in sorted(by.items(), key=lambda t: -t[1]): print(f'  {k:22s} {v:12.1f}')
big = [r for r in rows if r['maxabs'] > 2000]
print('tensors > 2000:', len(big))
for r in sorted(big, key=lambda r: -r['maxabs'])[:15]: print(f"  {r['maxabs']:12.1f} {r['op']:10s} {r['name']}")
