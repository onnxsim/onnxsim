import onnx, sys, re, collections
from onnxconverter_common import float16
D = "/mnt/data/cache/claude-work/models/demo/float/"
def mk(name, tag, pred):
    m = onnx.load(D + name + ".onnx")
    block = [n.name for n in m.graph.node if pred(n)]
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False, node_block_list=block)
    onnx.save(m16, f"m/{name}_{tag}.onnx")
    print(name, tag, "blocked", len(block), collections.Counter(n.op_type for n in m.graph.node if n.name in set(block)))
def attn_tail(n):   # linear-attention normalisation: Add(den, eps) and Div
    return n.op_type == 'Div' or (n.op_type == 'Add' and 'context_module/main/Add' == '/'.join(n.name.split('/')[-3:]))
V = {
 'base': lambda n: False,
 'div': lambda n: n.op_type == 'Div',
 'attn': attn_tail,
 'gelu': lambda n: n.op_type == 'Gelu',
 'attn_gelu': lambda n: attn_tail(n) or n.op_type == 'Gelu',
}
for name in ('sam_l0_enc',):
    for tag, p in V.items(): mk(name, 'n_' + tag, p)
