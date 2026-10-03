import onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper
rng=np.random.default_rng(3)
def mk(name,K,N,H,bias=True,dtype='f16'):
    W=(rng.standard_normal((N,K,1,1))*0.02).astype(np.float32)
    B=(rng.standard_normal(N)*0.1).astype(np.float32)
    x=(rng.standard_normal((1,K,H,H))*1.0).astype(np.float32)
    x.tofile(f'iso/{name}_in.bin')
    # fp32 reference output (float64 accumulate)
    ref=np.einsum('nk,bkhw->bnhw',W[:,:,0,0].astype(np.float64),x.astype(np.float64))+B[None,:,None,None]
    ref.astype(np.float32).tofile(f'iso/{name}_ref.bin')
    nodes=[]; inits=[]
    if dtype=='f16':
        nodes.append(helper.make_node('Cast',['X'],['Xh'],to=TensorProto.FLOAT16))
        inits+= [numpy_helper.from_array(W.astype(np.float16),'W'),numpy_helper.from_array(B.astype(np.float16),'B')]
        nodes.append(helper.make_node('Conv',['Xh','W','B'],['Yh'],kernel_shape=[1,1]))
        nodes.append(helper.make_node('Cast',['Yh'],['Y'],to=TensorProto.FLOAT))
    else:
        inits+= [numpy_helper.from_array(W,'W'),numpy_helper.from_array(B,'B')]
        nodes.append(helper.make_node('Conv',['X','W','B'],['Y'],kernel_shape=[1,1]))
    g=helper.make_graph(nodes,'g',[helper.make_tensor_value_info('X',TensorProto.FLOAT,[1,K,H,H])],[helper.make_tensor_value_info('Y',TensorProto.FLOAT,[1,N,H,H])],inits)
    m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)]); m.ir_version=8
    onnx.save(m,f'iso/{name}.onnx')
cases={'k256':(256,256,32),'k512':(512,256,32),'k1024':(1024,256,32),'k2048':(2048,256,32),'k2048n2048':(2048,2048,16),'k128':(128,2048,64),'k2048h64':(2048,256,64)}
for n,(K,N,H) in cases.items():
    mk(n+'_h',K,N,H,dtype='f16'); mk(n+'_f',K,N,H,dtype='f32')
    # f32 twin reuses the same data: regenerate with same rng state is not needed; the f32 model has its own weights, so skip comparing across
print(list(cases))
