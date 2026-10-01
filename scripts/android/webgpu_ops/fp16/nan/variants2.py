import onnx, json, warnings, collections
warnings.filterwarnings('ignore')
from onnxconverter_common import float16
D="/mnt/data/cache/claude-work/models/demo/float/"
rank=json.load(open('sam_l0_enc_rank.json'))
bad=[r['name'] for r in rank if r['err'] is not None and r['err']>6e-3]
bad_k=[r['name'] for r in rank if r['err'] is not None and r['err']>1.5e-2]
print(len(bad),len(bad_k))
def attn_tail(n): return n.op_type=='Div' or (n.op_type=='Add' and '/'.join(n.name.split('/')[-3:])=='context_module/main/Add')
def mk(tag,pred):
    m=onnx.load(D+'sam_l0_enc.onnx'); blk=[n.name for n in m.graph.node if pred(n)]
    m16=float16.convert_float_to_float16(m,keep_io_types=True,disable_shape_infer=False,node_block_list=blk)
    onnx.save(m16,f'm/sam_l0_enc_{tag}.onnx'); print(tag,len(blk),collections.Counter(n.op_type for n in m.graph.node if n.name in set(blk)))
mk('pc10',lambda n: attn_tail(n) or n.name in set(bad))
mk('pc9',lambda n: attn_tail(n) or n.name in set(bad_k))
mk('pc10g',lambda n: attn_tail(n) or n.name in set(bad) or n.op_type=='Gelu')
