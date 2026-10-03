import onnx, numpy as np, onnxruntime as ort, sys, json
name=sys.argv[1]; inp=sys.argv[2]
m=onnx.load(f'/mnt/data/cache/claude-work/models/demo/float/{name}.onnx')
init={i.name:onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
convs=[n for n in m.graph.node if n.op_type=='Conv' and n.input[1] in init]
need=set()
for n in convs: need.add(n.input[0])
m2=onnx.shape_inference.infer_shapes(m)
vi={v.name:v for v in list(m2.graph.value_info)}
x=np.load(f'{name}_{inp}_in.npy')
# graph input tensors might be consumed directly
outs=[t for t in need if t in vi]
del m.graph.output[:]
for t in outs: m.graph.output.append(vi[t])
so=ort.SessionOptions(); so.log_severity_level=3
s=ort.InferenceSession(m.SerializeToString(),so,providers=['CPUExecutionProvider'])
res=dict(zip(outs,s.run(outs,{m.graph.input[0].name:x})))
rows=[]
for n in convs:
    if n.input[0] not in res: continue
    W=init[n.input[1]]; g=[a.i for a in n.attribute if a.name=='group']; g=g[0] if g else 1
    B=init[n.input[2]] if len(n.input)>2 else np.zeros(W.shape[0],np.float32)
    X=res[n.input[0]]
    if W.shape[2:]!=(1,1) or g!=1: 
        rows.append(dict(name=n.name,kind=f'{W.shape[2]}x{W.shape[3]} g{g}',K=int(W.shape[1]*W.shape[2]*W.shape[3]),err=None)); continue
    Xm=X.reshape(X.shape[1],-1) if X.ndim==4 and X.shape[0]==1 else None
    Wm=W[:,:,0,0].astype(np.float64)
    ref=Wm@Xm.astype(np.float64)+B[:,None]
    W16=W[:,:,0,0].astype(np.float16).astype(np.float64); X16=Xm.astype(np.float16).astype(np.float64); B16=B.astype(np.float16).astype(np.float64)
    a32=W16@X16+B16[:,None]
    def acc16(chunk=4):
        out=np.zeros(ref.shape,np.float16); Wf=W16.astype(np.float32); Xf=X16.astype(np.float32)
        for k0 in range(0,Wf.shape[1],chunk):
            out=(out.astype(np.float32)+(Wf[:,k0:k0+chunk]@Xf[k0:k0+chunk]).astype(np.float16).astype(np.float32)).astype(np.float16)
        return out.astype(np.float64)+B16[:,None]
    r=lambda a: float(np.sqrt(((a-ref)**2).sum()/(ref**2).sum()))
    S=np.abs(Wm)@np.abs(Xm.astype(np.float64))
    rows.append(dict(name=n.name,kind='1x1',K=int(W.shape[1]),err=r(a32),err16=r(acc16()),cancel=float(np.median(S/(np.abs(ref)+1e-9))),maxpartial=float(S.max())))
json.dump(rows,open(f'{name}_rank.json','w'))
r1=[r for r in rows if r['err'] is not None]
print(len(rows),'weight convs;',len(r1),'group-1 1x1 convs')
for r in sorted(r1,key=lambda r:-r['err'])[:14]: print(f"  f16-operand err {r['err']:.1e}  f16-accum err {r['err16']:.1e}  cancel {r['cancel']:7.0f}  K {r['K']:5d}  maxpartial {r['maxpartial']:7.1f}  {r['name'].split('/enc/')[-1][-55:]}")
print('median err f16-operand',np.median([r['err'] for r in r1]),' median f16-accum',np.median([r['err16'] for r in r1]))
