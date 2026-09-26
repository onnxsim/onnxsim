"""Static int8 PTQ of openpilot's driving / DM models with onnxsim's whole-graph QDQ quantizer.

Calibration samples come from real route frames (`prepare_inputs.py`), run through the fp32 model in
modeld order so the driving model's recurrent state inputs are realistic too; every `--stride`-th
frame's full input dict is kept. The quantized model is QDQ (uint8 activations, int8 per-channel
weights, int32 biases): exactly the integer contract an HVX `vrmpy` kernel implements.
"""

import argparse
import json
import time

import numpy as np
import onnx
import run_models

from onnxsim.full_qdq import quantize_full_qdq


def backbone_nodes(model):
    """Nodes the Convs depend on (the conv backbone and its input preprocessing), in graph order.

    Everything else -- the Gemm/MatMul heads, the recurrent state bookkeeping (`next_state_*` Concats,
    which sit early in node order but never feed a Conv) -- stays float: quantizing a recurrent state
    tensor re-quantizes it every frame and the error compounds.
    """
    nodes = list(model.graph.node)
    prod = {o: i for i, n in enumerate(nodes) for o in n.output}
    seen, stack = set(), [i for i, n in enumerate(nodes) if n.op_type == "Conv"]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack += [prod[x] for x in nodes[i].input if x in prod]
    return [nodes[i] for i in sorted(seen)]


def non_backbone_node_names(model):
    bb = {n.name or n.output[0] for n in backbone_nodes(model)}
    return [
        n.name or n.output[0]
        for n in model.graph.node
        if (n.name or n.output[0]) not in bb
    ]


def exact_input_ranges(model):
    """Activation ranges that follow exactly from the uint8 camera inputs, not from calibration.

    The pixel normalization in front of each backbone ((x - 127.5) / 63.75 for driving, x / 255 for DM)
    is interval-propagated from [0, 255] through data movement (Cast, Slice, Gather, Reshape, Transpose,
    Unsqueeze, Concat) and elementwise ops with a constant operand (Sub, Add, Mul, Div). Calibration
    only sees the pixel values present in its frames (e.g. no pixel below 22 in the driving calibration
    set), so every darker pixel in another segment would be clipped at the network's very first tensor.
    """
    from onnx import numpy_helper

    consts = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    r = {
        i.name: (0.0, 255.0)
        for i in model.graph.input
        if i.type.tensor_type.elem_type == onnx.TensorProto.UINT8
    }
    move = {
        "Cast",
        "Slice",
        "Gather",
        "Reshape",
        "Transpose",
        "Unsqueeze",
        "Squeeze",
        "Flatten",
        "Identity",
    }
    for n in model.graph.node:
        ins = [x for x in n.input if x]
        if n.op_type in move and ins[0] in r:
            r[n.output[0]] = r[ins[0]]
        elif n.op_type == "Concat" and all(x in r for x in ins):
            r[n.output[0]] = (min(r[x][0] for x in ins), max(r[x][1] for x in ins))
        elif (
            n.op_type in ("Sub", "Add", "Mul", "Div")
            and len(ins) == 2
            and ins[0] in r
            and ins[1] in consts
        ):
            c = consts[ins[1]].astype(np.float64)
            lo, hi = r[ins[0]]
            f = {
                "Sub": lambda v: v - c,
                "Add": lambda v: v + c,
                "Mul": lambda v: v * c,
                "Div": lambda v: v / c,
            }[n.op_type]
            e = np.concatenate([np.ravel(f(lo)), np.ravel(f(hi))])
            r[n.output[0]] = (float(e.min()), float(e.max()))
    return r


def driving_samples(model, inputs, n, stride):
    sess = run_models.session(model)
    ins = {i.name: i for i in sess.get_inputs()}
    road = np.concatenate([np.load(f)["road"] for f in inputs.split(",")])
    state = {
        k: np.zeros(ins[k].shape, np.uint8 if "uint8" in ins[k].type else np.float32)
        for k in ins
        if k.startswith("state_")
    }
    fixed = {
        "desire": np.zeros(ins["desire"].shape, np.float32),
        "traffic_convention": np.array([[1, 0]], np.float32),
        "action_t": run_models.ACTION_T,
    }
    names = [o.name for o in sess.get_outputs()]
    out = []
    for i in range(min(n * stride, len(road))):
        feed = {"new_img": road[i], **state, **fixed}
        if i % stride == stride - 1:
            out.append({k: v.copy() for k, v in feed.items()})
        res = dict(zip(names, sess.run(None, feed)))
        for k in state:
            state[k] = res["next_" + k]
    return out


def dm_samples(inputs, n, stride, calib):
    drv = np.concatenate([np.load(f)["driver"] for f in inputs.split(",")])
    c = np.array([calib], np.float32)
    return [
        {"input_img": drv[i], "calib": c}
        for i in range(stride - 1, min(n * stride, len(drv)), stride)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["driving", "dm"])
    ap.add_argument("model", help="fp32 model (run_models.py to-fp32)")
    ap.add_argument(
        "inputs",
        help="calibration frames .npz, comma list (segments other than the eval one)",
    )
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--activation-dtype", default="uint8")
    ap.add_argument("--exclude-op-types", default="", help="comma list kept in float")
    ap.add_argument(
        "--exclude-nodes-prefix",
        default="",
        help="comma list of node-name prefixes kept in float",
    )
    ap.add_argument(
        "--float-after-last-conv",
        action="store_true",
        help="keep every node after the last Conv (the Gemm/MatMul heads) in float",
    )
    ap.add_argument(
        "--policy",
        help="JSON {tensor_dtypes: {tensor: dtype}, exclude_nodes: [...]} "
        "(e.g. from sweep_driving_groups.py), merged with the flags above",
    )
    ap.add_argument("--calib", default="0,0.164,0.005")
    args = ap.parse_args()
    t = time.time()
    data = (
        driving_samples(args.model, args.inputs, args.samples, args.stride)
        if args.kind == "driving"
        else dm_samples(
            args.inputs,
            args.samples,
            args.stride,
            [float(v) for v in args.calib.split(",")],
        )
    )
    m = onnx.load(args.model)
    prefixes = [p for p in args.exclude_nodes_prefix.split(",") if p]
    excl_nodes = [
        n.name for n in m.graph.node if any(n.name.startswith(p) for p in prefixes)
    ]
    if args.float_heads:
        excl_nodes += non_backbone_node_names(m)
    tensor_dtypes = {}
    if args.policy:
        pol = json.load(open(args.policy))
        tensor_dtypes = pol.get("tensor_dtypes", {})
        excl_nodes += pol.get("exclude_nodes", [])
    q = quantize_full_qdq(
        m,
        data,
        activation_dtype=args.activation_dtype,
        method=args.method,
        exclude_op_types=[o for o in args.exclude_op_types.split(",") if o],
        exclude_nodes=excl_nodes,
        tensor_dtypes=tensor_dtypes,
    )
    onnx.save(q, args.out)
    nq = sum(n.op_type == "QuantizeLinear" for n in q.graph.node)
    print(
        f"{args.out}: {len(data)} calibration samples, {nq} QuantizeLinear, {len(excl_nodes)} nodes kept float, {time.time() - t:.0f}s"
    )


if __name__ == "__main__":
    main()
