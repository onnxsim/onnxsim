"""Exact graph rewrites for the float YOLO11n / YOLO26n twins that remove Split / Concat dispatches.

    python yolo_graph_opt.py in.onnx out.onnx [--rules split_conv,head,concat_conv]

Rules (all mathematically exact, applied in this order):

split_conv    Conv -> SiLU(Sigmoid, Mul) -> Split(axis=1, 2 outputs)   (the C2f / C3k2 cv1 of every block)
              => two Convs on the split halves of the weights (+ their own SiLU). Channels are independent,
              so the result is identical; one Split dispatch (a full read+write of the tensor) disappears.
head          YOLO11n detect head: per scale Concat(box, cls) -> Reshape, Concat over scales, Split(box, cls)
              => Reshape each branch, Concat box and cls over scales separately (3 Concat + 1 Split fewer).
resize_convt  nearest 2x Resize => depthwise ConvTranspose (has an NHWC kernel; Resize has none, so ORT wraps it in Transposes)
concat_conv   Concat(axis=1) whose only consumer is a 1x1 stride-1 Conv (group 1)
              => sum of per-input partial Convs (weight sliced along Cin), chained with Add so that ORT's
              Conv+Add fusion (NhwcFusedConv with residual) folds the sums into the convs. No Concat copy,
              but one more Conv dispatch per extra input; useful only when the copy is more expensive than that.

The default rule set is the one that measured as a win on the phone (see WEBGPU_SURVEY.md): split_conv,head,resize_convt.
"""

import argparse
import collections
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


class Graph:
    def __init__(self, model):
        self.model = model
        self.g = model.graph
        self.inits = {i.name: i for i in self.g.initializer}
        self.nodes = list(self.g.node)
        self.uid = 0

    def refresh(self):
        self.prod = {o: n for n in self.nodes for o in n.output}
        self.cons = collections.defaultdict(list)
        for n in self.nodes:
            for i in n.input:
                self.cons[i].append(n)
        for o in self.g.output:
            self.cons[o.name].append(None)  # graph outputs count as consumers

    def arr(self, name):
        return numpy_helper.to_array(self.inits[name])

    def add_init(self, name, array):
        self.inits[name] = numpy_helper.from_array(np.asarray(array), name)

    def name(self, base):
        self.uid += 1
        return f"{base}__opt{self.uid}"

    def finish(self):
        used = {i for n in self.nodes for i in n.input}
        keep = [t for n, t in self.inits.items() if n in used]
        del self.g.node[:]
        self.g.node.extend(self.nodes)
        del self.g.initializer[:]
        self.g.initializer.extend(keep)
        del self.g.value_info[:]
        return self.model


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def silu_chain(G, node):
    """Mul(c, Sigmoid(c)) with c = Conv output used only by the Sigmoid and the Mul -> (conv, sigmoid, mul)."""
    mul = node
    if mul.op_type != "Mul":
        return None
    sig = next((G.prod.get(i) for i in mul.input if G.prod.get(i) is not None and G.prod[i].op_type == "Sigmoid"), None)
    if sig is None:
        return None
    other = [i for i in mul.input if i != sig.output[0]]
    if len(other) != 1 or sig.input[0] != other[0]:
        return None
    conv = G.prod.get(other[0])
    if conv is None or conv.op_type != "Conv" or len(G.cons[conv.output[0]]) != 2 or len(G.cons[sig.output[0]]) != 1:
        return None
    return conv, sig, mul


def rule_split_conv(G):
    n_done = 0
    G.refresh()
    for split in [n for n in G.nodes if n.op_type == "Split"]:
        if attr(split, "axis", 0) != 1 or len(split.output) != 2:
            continue
        src = G.prod.get(split.input[0])
        if src is None or len(G.cons[split.input[0]]) != 1:
            continue
        chain = silu_chain(G, src)
        if chain is None:
            continue
        conv, sig, mul = chain
        if attr(conv, "group", 1) != 1 or conv.input[1] not in G.inits:
            continue
        sizes = attr(split, "split")
        if sizes is None and len(split.input) > 1 and split.input[1] in G.inits:
            sizes = G.arr(split.input[1]).tolist()
        if sizes is None:
            sizes = [G.arr(conv.input[1]).shape[0] // 2] * 2
        w = G.arr(conv.input[1])
        b = G.arr(conv.input[2]) if len(conv.input) > 2 and conv.input[2] else None
        if sum(sizes) != w.shape[0]:
            continue
        new, off = [], 0
        for k, sz in enumerate(sizes):
            wn = G.name(conv.input[1] + f"_s{k}")
            G.add_init(wn, w[off : off + sz])
            ins = [conv.input[0], wn]
            if b is not None:
                bn = G.name(conv.input[2] + f"_s{k}")
                G.add_init(bn, b[off : off + sz])
                ins.append(bn)
            off += sz
            c_out, s_out = G.name(split.output[k] + "_c"), G.name(split.output[k] + "_s")
            new.append(helper.make_node("Conv", ins, [c_out], name=G.name(conv.name + f"_s{k}"), **{a.name: helper.get_attribute_value(a) for a in conv.attribute}))
            new.append(helper.make_node("Sigmoid", [c_out], [s_out], name=G.name(sig.name + f"_s{k}")))
            new.append(helper.make_node("Mul", [c_out, s_out], [split.output[k]], name=G.name(mul.name + f"_s{k}")))
        idx = G.nodes.index(split)
        G.nodes[idx : idx + 1] = new
        for dead in (conv, sig, mul):
            G.nodes.remove(dead)
        G.refresh()
        n_done += 1
    return n_done


def rule_head(G):
    G.refresh()
    done = 0
    for split in [n for n in G.nodes if n.op_type == "Split"]:
        if attr(split, "axis", 0) != 1 or len(split.output) != 2 or len(split.input) < 2 or split.input[1] not in G.inits:
            continue
        sizes = G.arr(split.input[1]).tolist()
        c3 = G.prod.get(split.input[0])
        if c3 is None or c3.op_type != "Concat" or attr(c3, "axis") != 2:
            continue
        reshapes = [G.prod.get(i) for i in c3.input]
        if any(r is None or r.op_type != "Reshape" or len(G.cons[r.output[0]]) != 1 for r in reshapes):
            continue
        cats = [G.prod.get(r.input[0]) for r in reshapes]
        if any(c is None or c.op_type != "Concat" or attr(c, "axis") != 1 or len(c.input) != 2 or len(G.cons[c.output[0]]) != 1 for c in cats):
            continue
        boxes, clss = [], []
        new = []
        for k, (r, c) in enumerate(zip(reshapes, cats)):
            for j, (src, ch) in enumerate(zip(c.input, sizes)):
                sn = G.name(f"head_shape{ch}")
                G.add_init(sn, np.array([1, ch, -1], dtype=np.int64))
                out = G.name(f"head_r{k}_{j}")
                new.append(helper.make_node("Reshape", [src, sn], [out], name=G.name(f"head_reshape{k}_{j}")))
                (boxes if j == 0 else clss).append(out)
        new.append(helper.make_node("Concat", boxes, [split.output[0]], axis=2, name=G.name("head_concat_box")))
        new.append(helper.make_node("Concat", clss, [split.output[1]], axis=2, name=G.name("head_concat_cls")))
        idx = G.nodes.index(split)
        G.nodes[idx : idx + 1] = new
        for dead in [c3] + reshapes + cats:
            G.nodes.remove(dead)
        G.refresh()
        done += 1
    return done


def rule_concat_conv(G, shapes):
    G.refresh()
    done = 0
    for cat in [n for n in G.nodes if n.op_type == "Concat"]:
        if attr(cat, "axis") != 1 or len(G.cons[cat.output[0]]) != 1 or G.cons[cat.output[0]][0] is None:
            continue
        conv = G.cons[cat.output[0]][0]
        if conv.op_type != "Conv" or conv.input[0] != cat.output[0] or attr(conv, "group", 1) != 1:
            continue
        if list(attr(conv, "kernel_shape", [])) != [1, 1] or list(attr(conv, "strides", [1, 1])) != [1, 1] or any(attr(conv, "pads", [0] * 4)):
            continue
        if conv.input[1] not in G.inits or any(i not in shapes for i in cat.input):
            continue
        w = G.arr(conv.input[1])
        b = G.arr(conv.input[2]) if len(conv.input) > 2 and conv.input[2] else np.zeros(w.shape[0], w.dtype)
        chans = [shapes[i][1] for i in cat.input]
        if sum(chans) != w.shape[1]:
            continue
        new, off, partial = [], 0, []
        for k, (src, ch) in enumerate(zip(cat.input, chans)):
            wn = G.name(conv.input[1] + f"_c{k}")
            G.add_init(wn, w[:, off : off + ch])
            bn = G.name(conv.input[2] + f"_c{k}") if len(conv.input) > 2 else G.name("zbias")
            G.add_init(bn, b if k == 0 else np.zeros_like(b))
            off += ch
            out = G.name(conv.output[0] + f"_p{k}")
            new.append(helper.make_node("Conv", [src, wn, bn], [out], name=G.name(conv.name + f"_c{k}"), **{a.name: helper.get_attribute_value(a) for a in conv.attribute}))
            partial.append(out)
        acc = partial[0]
        for k, p in enumerate(partial[1:]):
            out = conv.output[0] if k == len(partial) - 2 else G.name(conv.output[0] + f"_acc{k}")
            new.append(helper.make_node("Add", [acc, p], [out], name=G.name(conv.name + f"_add{k}")))
            acc = out
        idx = G.nodes.index(conv)
        G.nodes[idx : idx + 1] = new
        G.nodes.remove(cat)
        G.refresh()
        done += 1
    return done


def rule_resize_convt(G, shapes):
    """Nearest 2x Resize (asymmetric, floor) => depthwise ConvTranspose(kernel 2, stride 2) with all-ones weights.
    Exact (every output pixel copies one input pixel). ConvTranspose has an NHWC kernel in the WebGPU EP, Resize does not, so
    this removes the NCHW<->NHWC Transposes ORT wraps around each Resize."""
    G.refresh()
    done = 0
    for rs in [n for n in G.nodes if n.op_type == "Resize"]:
        if attr(rs, "mode", b"nearest") not in (b"nearest", "nearest") or rs.input[0] not in shapes:
            continue
        if attr(rs, "coordinate_transformation_mode", b"half_pixel") not in (b"asymmetric", "asymmetric"):
            continue
        if attr(rs, "nearest_mode", b"round_prefer_floor") not in (b"floor", "floor"):
            continue
        sc = rs.input[2] if len(rs.input) > 2 else ""
        if sc not in G.inits or G.arr(sc).tolist() != [1.0, 1.0, 2.0, 2.0]:
            continue
        c = shapes[rs.input[0]][1]
        wn = G.name("upsample_w")
        G.add_init(wn, np.ones((c, 1, 2, 2), dtype=np.float32))
        ct = helper.make_node("ConvTranspose", [rs.input[0], wn], [rs.output[0]], kernel_shape=[2, 2], strides=[2, 2], group=c, name=G.name(rs.name + "_ct"))
        G.nodes[G.nodes.index(rs)] = ct
        done += 1
    G.refresh()
    return done


def _shapes(G):
    m2 = onnx.ModelProto()
    m2.CopyFrom(G.model)
    del m2.graph.node[:]
    m2.graph.node.extend(G.nodes)
    del m2.graph.initializer[:]
    m2.graph.initializer.extend(G.inits.values())
    inferred = shape_inference.infer_shapes(m2)
    out = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input):
        out[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    return out


def optimize(model, rules):
    G = Graph(model)
    counts = {}
    for r in rules:
        if r == "split_conv":
            counts[r] = rule_split_conv(G)
        elif r == "head":
            counts[r] = rule_head(G)
        elif r == "concat_conv":
            counts[r] = rule_concat_conv(G, _shapes(G))
        elif r == "resize_convt":
            counts[r] = rule_resize_convt(G, _shapes(G))
        else:
            raise SystemExit(f"unknown rule {r}")
    return G.finish(), counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rules", default="split_conv,head,resize_convt")
    a = ap.parse_args()
    m = onnx.load(a.src)
    m, counts = optimize(m, [r for r in a.rules.split(",") if r])
    onnx.checker.check_model(m)
    onnx.save(m, a.dst)
    print(a.src, "->", a.dst, counts, collections.Counter(n.op_type for n in m.graph.node).most_common(6), file=sys.stderr)
