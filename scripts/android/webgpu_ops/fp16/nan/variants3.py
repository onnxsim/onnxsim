import onnx, json, warnings, collections
warnings.filterwarnings('ignore')
from onnxconverter_common import float16
D="/mnt/data/cache/claude-work/models/demo/float/"
rank=json.load(open('sam_l0_enc_rank.json'))
bad=set(r['name'] for r in rank if r['err'] is not None and r['err']>6e-3)
m0=onnx.load(D+'sam_l0_enc.onnx')
prod={o:n for n in m0.graph.node for o in n.output}
def chain(name,depth):
    n=[x for x in m0.graph.node if x.name==name][0]; out=[]; t=n.input[0]
    for _ in range(depth):
        p=prod.get(t)
        if p is None or p.op_type in ('Add','Sub','Slice','Reshape','MatMul','Concat','Div','Relu','Transpose','Pad','DepthToSpace'): break
        out.append(p.name); t=p.input[0]
    return out
def attn_tail(n): return n.op_type=='Div' or (n.op_type=='Add' and '/'.join(n.name.split('/')[-3:])=='context_module/main/Add')
def mk(tag,names):
    m=onnx.load(D+'sam_l0_enc.onnx'); names=set(names)
    blk=[n.name for n in m.graph.node if attn_tail(n) or n.name in names]
    m16=float16.convert_float_to_float16(m,keep_io_types=True,disable_shape_infer=False,node_block_list=blk)
    onnx.save(m16,f'm/sam_l0_enc_{tag}.onnx'); print(tag,len(blk),collections.Counter(n.op_type for n in m.graph.node if n.name in set(blk)))
d2=set(bad); [d2.update(chain(b,2)) for b in bad]     # + producer Gelu + depth conv
mk('pc10d',d2)
d3=set(bad); [d3.update(chain(b,4)) for b in bad]     # + Gelu, depth conv, Gelu, inverted_conv
mk('pc10e',d3)
