#!/usr/bin/env python3
"""Lower a static QDQ ONNX graph (onnxsim full_qdq + quantized_io form) to the DSP runner's program, and emulate
that program exactly in numpy (the reference the DSP output must match, itself checked against ORT CPU).

Supported: Conv (k x k, stride 1/2, symmetric pad k//2, group 1, per-channel int8 weights, int32 bias) with its
output Q (a Relu folded into Q, i.e. zero point 0, is implicit), Add of two quantized tensors (ORT: QLinearAdd),
MaxPool (k x k, stride, pad; same scale in and out), and the NHWC->NCHW Transpose quantized_io puts after a uint8
NHWC graph input. Anything else stops with an error naming the node.

usage: qdq_graph.py <model.onnx> <outdir> [input.bin ref.bin]
writes <outdir>/program.txt (one op per line) + weights.bin; with input.bin/ref.bin also runs the emulation and
compares with ORT's output.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


class Tensor:
    def __init__(self, name, c, h, w, scale, zp):
        self.name, self.c, self.h, self.w, self.scale, self.zp = (
            name,
            c,
            h,
            w,
            np.float32(scale),
            int(zp),
        )


def lower(model):
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

    def dq_params(name):
        """DequantizeLinear output -> (quantized source, scale, zp)"""
        n = prod[name]
        if n.op_type != "DequantizeLinear":
            raise ValueError(
                f"{name}: expected a DequantizeLinear, got {n.op_type} ({n.name})"
            )
        src = n.input[0]
        return src, inits[n.input[1]], inits[n.input[2]] if len(n.input) > 2 else None

    tensors, ops, blob = {}, [], bytearray()

    def put(a):
        off = len(blob)
        blob.extend(np.ascontiguousarray(a).tobytes())
        while len(blob) % 128:
            blob.append(0)
        return off

    gin = g.input[0].name
    xs = [n for n in cons[gin]]
    if (
        len(xs) != 1
        or xs[0].op_type != "Transpose"
        or list(xs[0].attribute[0].ints) != [0, 3, 1, 2]
    ):
        raise ValueError(
            "expected a uint8 NHWC graph input followed by Transpose(0,3,1,2) (quantized_io nhwc)"
        )
    nchw = xs[0].output[0]
    dqin = cons[nchw][0]
    _, h, w, c = shapes[gin]
    tensors[nchw] = Tensor(nchw, c, h, w, inits[dqin.input[1]], inits[dqin.input[2]])
    done = {xs[0].name}

    def qtensor(qout_name, c, h, w):
        q = prod[qout_name]
        if q.op_type != "QuantizeLinear":
            raise ValueError(f"{qout_name}: not a QuantizeLinear output")
        t = Tensor(qout_name, c, h, w, inits[q.input[1]], inits[q.input[2]])
        tensors[qout_name] = t
        return t

    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") or n.name in done:
            continue
        if n.op_type == "Conv":
            at = {a.name: a for a in n.attribute}
            k = list(at["kernel_shape"].ints)
            st = list(at["strides"].ints) if "strides" in at else [1, 1]
            pads = list(at["pads"].ints) if "pads" in at else [0, 0, 0, 0]
            grp = at["group"].i if "group" in at else 1
            if (
                k[0] != k[1]
                or st[0] != st[1]
                or st[0] not in (1, 2)
                or len(set(pads)) != 1
                or pads[0] != k[0] // 2
                or grp != 1
            ):
                raise ValueError(
                    f"unsupported Conv {n.name}: k {k} strides {st} pads {pads} group {grp}"
                )
            xsrc, sx, zx = dq_params(n.input[0])
            wq, sw, _ = dq_params(n.input[1])
            bq, _, _ = dq_params(n.input[2]) if len(n.input) > 2 else (None, None, None)
            y = n.output[0]
            qn = cons[y]
            if len(qn) != 1 or qn[0].op_type != "QuantizeLinear":
                raise ValueError(
                    f"Conv {n.name}: output must go straight into one QuantizeLinear"
                )
            xt = tensors[xsrc]
            wa = inits[wq]
            ho, wo = (xt.h - 1) // st[0] + 1, (xt.w - 1) // st[0] + 1
            yt = qtensor(qn[0].output[0], wa.shape[0], ho, wo)
            ba = inits[bq] if bq is not None else np.zeros(wa.shape[0], np.int32)
            swa = np.broadcast_to(np.asarray(sw, np.float32), (wa.shape[0],)).copy()
            ops.append(
                dict(
                    op="conv",
                    x=xt,
                    y=yt,
                    k=k[0],
                    s=st[0],
                    w=put(wa.astype(np.int8)),
                    wshape=wa.shape,
                    b=put(ba.astype(np.int32)),
                    sw=put(swa),
                    wq=wa,
                    bq=ba,
                    swa=swa,
                )
            )
        elif n.op_type == "Add":
            a, sa, za = dq_params(n.input[0])
            b, sb, zb = dq_params(n.input[1])
            qn = cons[n.output[0]]
            if len(qn) != 1 or qn[0].op_type != "QuantizeLinear":
                raise ValueError(
                    f"Add {n.name}: output must go straight into one QuantizeLinear"
                )
            at_, bt = tensors[a], tensors[b]
            yt = qtensor(qn[0].output[0], at_.c, at_.h, at_.w)
            ops.append(dict(op="add", a=at_, b=bt, y=yt))
        elif n.op_type == "MaxPool":
            at = {a.name: a for a in n.attribute}
            k, st = list(at["kernel_shape"].ints), list(at["strides"].ints)
            pads = list(at["pads"].ints) if "pads" in at else [0, 0, 0, 0]
            xsrc, sx, zx = dq_params(n.input[0])
            qn = cons[n.output[0]]
            if (
                len(qn) != 1
                or qn[0].op_type != "QuantizeLinear"
                or k[0] != 3
                or st[0] != 2
                or pads[0] != 1
            ):
                raise ValueError(f"unsupported MaxPool {n.name}")
            xt = tensors[xsrc]
            yt = qtensor(
                qn[0].output[0], xt.c, (xt.h - 1) // 2 + 1, (xt.w - 1) // 2 + 1
            )
            if yt.scale != xt.scale or yt.zp != xt.zp:
                raise ValueError(
                    f"MaxPool {n.name}: requantizing MaxPool not supported"
                )
            ops.append(dict(op="maxpool", x=xt, y=yt, k=3, s=2))
        else:
            raise ValueError(f"unsupported op {n.op_type} ({n.name})")
    out = g.output[0].name
    if out not in tensors:
        raise ValueError(
            f"graph output {out} is not a quantized tensor produced by a supported op"
        )
    return tensors, ops, bytes(blob), tensors[nchw], tensors[out]


# ---------------------------------------------------------------- exact emulation (ORT CPU semantics)


def conv(xq, zx, wq, bq, k, s):
    c, h, w = xq.shape
    p = k // 2
    x = np.pad(xq.astype(np.int64) - zx, ((0, 0), (p, p), (p, p)))
    ho, wo = (h - 1) // s + 1, (w - 1) // s + 1
    acc = np.zeros((wq.shape[0], ho, wo), np.int64)
    for dy in range(k):
        for dx in range(k):
            patch = x[:, dy : dy + s * ho : s, dx : dx + s * wo : s].reshape(c, -1)
            acc += (wq[:, :, dy, dx].astype(np.int64) @ patch).reshape(-1, ho, wo)
    return acc + bq.astype(np.int64)[:, None, None]


def requant(acc, sx, sw, sy, zy):
    m = (np.float32(sx) * sw.astype(np.float32) / np.float32(sy)).astype(np.float32)
    return np.clip(
        np.rint(acc.astype(np.float32) * m[:, None, None]).astype(np.int64) + zy, 0, 255
    ).astype(np.uint8)


def add_consts(ta, tb, ty):
    """ORT's QLinearAdd (MLAS): ra = sa/sy, rb = sb/sy, fixed = zy - (ra*za + rb*zb), all fp32"""
    f = np.float32
    ra, rb = f(ta.scale / ty.scale), f(tb.scale / ty.scale)
    return ra, rb, f(f(ty.zp) - f(f(ra * f(ta.zp)) + f(rb * f(tb.zp))))


def add(a, ta, b, tb, ty):
    """y = clamp(rne(rb*b + (ra*a + fixed))), separate (unfused) fp32 multiplies and adds in exactly this order:
    matches ORT CPU on every Add of the ResNet-18 graph over 3 inputs (a fused multiply-add, or another order, is
    1 LSB off on a few elements, and ReLU nets amplify one LSB into ~20% of the final output)"""
    f = np.float32
    ra, rb, fixed = add_consts(ta, tb, ty)
    v = f(rb * b.astype(f)) + f(f(ra * a.astype(f)) + fixed)
    return np.clip(np.rint(v).astype(np.int64), 0, 255).astype(np.uint8)


def maxpool(x):
    c, h, w = x.shape
    ho, wo = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    y = np.zeros((c, ho, wo), np.uint8)
    for dy in range(3):
        for dx in range(3):
            y = np.maximum(y, xp[:, dy : dy + 2 * ho : 2, dx : dx + 2 * wo : 2])
    return y


def emulate(tensors, ops, xin, xq_nhwc):
    vals = {xin.name: xq_nhwc[0].transpose(2, 0, 1)}
    for o in ops:
        if o["op"] == "conv":
            acc = conv(vals[o["x"].name], o["x"].zp, o["wq"], o["bq"], o["k"], o["s"])
            vals[o["y"].name] = requant(
                acc, o["x"].scale, o["swa"], o["y"].scale, o["y"].zp
            )
        elif o["op"] == "add":
            vals[o["y"].name] = add(
                vals[o["a"].name], o["a"], vals[o["b"].name], o["b"], o["y"]
            )
        else:
            vals[o["y"].name] = maxpool(vals[o["x"].name])
    return vals


def write_program(tensors, ops, blob, xin, yout, out):
    names = {t: i for i, t in enumerate(tensors)}
    lines = [
        f"tensors {len(tensors)} ops {len(ops)} input {names[xin.name]} output {names[yout.name]}"
    ]
    for t in tensors.values():
        lines.append(
            f"T {names[t.name]} {t.c} {t.h} {t.w} {float(t.scale).hex()} {t.zp}"
        )
    for o in ops:
        if o["op"] == "conv":
            n, c, _, _ = o["wshape"]
            lines.append(
                f"conv {names[o['x'].name]} {names[o['y'].name]} {o['k']} {o['s']} {n} {c} {o['w']} {o['b']} {o['sw']}"
            )
        elif o["op"] == "add":
            lines.append(
                f"add {names[o['a'].name]} {names[o['b'].name]} {names[o['y'].name]}"
            )
        else:
            lines.append(
                f"maxpool {names[o['x'].name]} {names[o['y'].name]} {o['k']} {o['s']}"
            )
    (out / "program.txt").write_text("\n".join(lines) + "\n")
    (out / "weights.bin").write_bytes(blob)


def main():
    model = onnx.shape_inference.infer_shapes(onnx.load(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tensors, ops, blob, xin, yout = lower(model)
    write_program(tensors, ops, blob, xin, yout, out)
    kinds = {}
    for o in ops:
        key = o["op"] if o["op"] != "conv" else f"conv{o['k']}x{o['k']}s{o['s']}"
        kinds[key] = kinds.get(key, 0) + 1
    print(f"{len(ops)} ops {kinds}, {len(tensors)} tensors, {len(blob)} weight bytes")
    if len(sys.argv) > 4:
        x = np.fromfile(sys.argv[3], np.uint8).reshape(1, xin.h, xin.w, xin.c)
        ref = np.fromfile(sys.argv[4], np.uint8)
        y = emulate(tensors, ops, xin, x)[yout.name].ravel()
        d = y.astype(int) - ref
        print(
            f"emulation vs ORT: {int((d != 0).sum())} mismatches of {d.size}",
            {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))},
        )


if __name__ == "__main__":
    main()
