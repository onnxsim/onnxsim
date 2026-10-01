import onnx, json, collections, warnings
warnings.filterwarnings('ignore')
from onnxconverter_common import float16
D="/mnt/data/cache/claude-work/models/demo/float/"
rows=json.load(open('rtdetr_pre_photo_ranges.json'))
rn=json.load(open('rtdetr_pre_noise_ranges.json'))
def big(th):
    s=set(r['name'] for r in rows if r['maxabs']>th)|set(r['name'] for r in rn if r['maxabs']>th)
    return s
def mk(tag, names):
    m=onnx.load(D+'rtdetr_pre.onnx')
    names=set(names)
    blk=[n.name for n in m.graph.node if n.name in names]
    m16=float16.convert_float_to_float16(m,keep_io_types=True,disable_shape_infer=False,node_block_list=blk)
    onnx.save(m16,f'm/rtdetr_pre_{tag}.onnx')
    print(tag,'blocked',len(blk),collections.Counter(n.op_type for n in m.graph.node if n.name in names))
# the overflow region + the LayerNormalization that renormalises it
b=big(20000)
m=onnx.load(D+'rtdetr_pre.onnx')
ln_after=[n.name for n in m.graph.node if n.op_type=='LayerNormalization'][:1]
mk('r_base', [])
mk('r_big', b|set(ln_after))
mk('r_big_ln', b|set(n.name for n in m.graph.node if n.op_type=='LayerNormalization'))
mk('r_big_sm', b|set(ln_after)|set(n.name for n in m.graph.node if n.op_type=='Softmax'))
# contiguous island: from the first overflowing node to the LayerNormalization that renormalises (nodes 37..65 of the fp32 float-output node list)
mk('r_isl', [r['name'] for r in rows[37:66]])
