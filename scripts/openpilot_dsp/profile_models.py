"""Op histogram, per-op MACs and I/O of openpilot's ONNX models (driving_supercombo, dmonitoring)."""

import argparse
import collections
import json

import numpy as np
import onnx
from onnx import shape_inference


def dims(vi):
    return [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in vi.type.tensor_type.shape.dim
    ]


def profile(path):
    m = onnx.load(path)
    m = shape_inference.infer_shapes(m)
    g = m.graph
    shapes = {
        v.name: dims(v) for v in list(g.value_info) + list(g.input) + list(g.output)
    }
    dtypes = {
        v.name: v.type.tensor_type.elem_type
        for v in list(g.value_info) + list(g.input) + list(g.output)
    }
    inits = {i.name: i for i in g.initializer}
    for i in g.initializer:
        shapes[i.name] = list(i.dims)
        dtypes[i.name] = i.data_type
    hist = collections.Counter(n.op_type for n in g.node)
    macs = collections.Counter()
    convs = []
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            w = shapes.get(n.input[1])
            y = shapes.get(n.output[0])
            if not w or not y or not all(isinstance(d, int) for d in y):
                continue
            k = int(np.prod(w[1:]))
            mac = (
                int(np.prod(y)) * k
                if n.op_type == "Conv"
                else int(np.prod(shapes[n.input[0]])) * int(np.prod(w[1:]))
            )
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            macs[n.op_type] += mac
            convs.append(
                dict(
                    name=n.name,
                    x=shapes.get(n.input[0]),
                    w=w,
                    y=y,
                    group=attrs.get("group", 1),
                    stride=attrs.get("strides"),
                    macs=mac,
                )
            )
        elif n.op_type in ("MatMul", "Gemm"):
            a, b = shapes.get(n.input[0]), shapes.get(n.input[1])
            y = shapes.get(n.output[0])
            if a and b and y and all(isinstance(d, int) for d in y):
                kdim = (
                    a[-1]
                    if not (
                        n.op_type == "Gemm"
                        and any(x.name == "transA" and x.i for x in n.attribute)
                    )
                    else a[0]
                )
                mac = int(np.prod(y)) * kdim
                macs[n.op_type] += mac
                convs.append(
                    dict(
                        name=n.name,
                        x=a,
                        w=b,
                        y=y,
                        group=1,
                        stride=None,
                        macs=mac,
                        op=n.op_type,
                    )
                )
    params = sum(int(np.prod(i.dims)) for i in g.initializer)
    init_types = collections.Counter(
        onnx.TensorProto.DataType.Name(i.data_type) for i in g.initializer
    )
    io = dict(
        inputs=[
            (
                v.name,
                dims(v),
                onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
            )
            for v in g.input
            if v.name not in inits
        ],
        outputs=[
            (
                v.name,
                dims(v),
                onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
            )
            for v in g.output
        ],
    )
    return dict(
        path=path,
        opset=[(o.domain, o.version) for o in m.opset_import],
        nodes=len(g.node),
        params=params,
        init_types=dict(init_types),
        hist=dict(hist.most_common()),
        macs=dict(macs),
        total_gmac=sum(macs.values()) / 1e9,
        io=io,
        heavy=sorted(convs, key=lambda c: -c["macs"]),
        metadata={p.key: p.value[:200] for p in m.metadata_props},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    out = []
    for p in args.models:
        r = profile(p)
        out.append(r)
        print(
            f"== {p}: {r['nodes']} nodes, {r['params'] / 1e6:.2f} M params {r['init_types']}, {r['total_gmac']:.3f} GMAC, opset {r['opset']}"
        )
        print("  inputs:", r["io"]["inputs"])
        print("  outputs:", r["io"]["outputs"])
        print("  ops:", r["hist"])
        print("  MACs by op (G):", {k: round(v / 1e9, 3) for k, v in r["macs"].items()})
        print("  metadata:", r["metadata"])
        for c in r["heavy"][: args.top]:
            print(
                f"    {c['macs'] / 1e6:9.1f} MMAC  x={c['x']} w={c['w']} y={c['y']} g={c['group']} s={c['stride']} {c.get('op', '')}"
            )
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=str)
