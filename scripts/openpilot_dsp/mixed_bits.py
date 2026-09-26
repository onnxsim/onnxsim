"""Per-conv weight precision (int8 / int16) on top of `quantize_full_qdq`, and a sensitivity sweep.

`quantize_full_qdq` always makes Conv weights int8. This rewrites the weight `DequantizeLinear` of chosen
convs to an int16 per-channel one (scale max|W| / 32767, `com.microsoft` DQ, since the models are opset 20)
from the *original* float weights. Biases keep their int32 / (s_x * s_w8) encoding, which stays valid.
Int8 convs can come from an AdaRound'ed model (`adaround_conv.py`): its weights lie exactly on the
int8 grid, so `quantize_full_qdq` reproduces the rounded codes.

  build:  python mixed_bits.py build <fp32> <int8-weight source> <calib npz> out.onnx --w16 <conv node names>
  sweep:  python mixed_bits.py sweep <fp32> <int8-weight source> <calib npz> <rank npz> sweep.json
  actsweep: the same arguments; every conv int16, one window of backbone nodes at a time kept float

The sweep ranks convs by the damage of making *only that conv* int8 (every other conv int16, A16
activations, float heads), scored free-running on `--rank-frames` frames of a *calibration* segment by
plan lateral error, so the held-out segments stay held out.
"""

import argparse
import json
import time

import numpy as np
import onnx
import quantize
import run_models
from onnx import TensorProto, helper, numpy_helper

from onnxsim.full_qdq import quantize_full_qdq


def head_nodes(m):
    return quantize.non_backbone_node_names(m)


def to_w16(q, fp32, convs16):
    """Rewrite the int8 weight DQ of every conv (by node name) in convs16 to int16."""
    fw = {i.name: numpy_helper.to_array(i) for i in fp32.graph.initializer}
    prod = {o: n for n in q.graph.node for o in n.output}
    inits = {i.name: k for k, i in enumerate(q.graph.initializer)}
    orig_w = {n.name: n.input[1] for n in fp32.graph.node if n.op_type == "Conv"}
    done = 0
    for n in q.graph.node:
        if n.op_type != "Conv" or n.name not in convs16:
            continue
        dq = prod.get(n.input[1])
        if (
            dq is None or dq.op_type != "DequantizeLinear"
        ):  # left float (a float neighbour declined the QDQ unit)
            done += 1
            continue
        w = fw[orig_w[n.name]].astype(np.float32)
        s = np.abs(w.reshape(w.shape[0], -1)).max(1) / 32767
        s[s == 0] = 1
        codes = np.clip(np.round(w / s.reshape(-1, 1, 1, 1)), -32767, 32767).astype(
            np.int16
        )
        xq, sc, zp = (
            dq.input[0],
            dq.input[1],
            dq.input[2] if len(dq.input) > 2 else None,
        )
        q.graph.initializer[inits[xq]].CopyFrom(numpy_helper.from_array(codes, xq))
        q.graph.initializer[inits[sc]].CopyFrom(
            numpy_helper.from_array(s.astype(np.float32), sc)
        )
        if zp:
            q.graph.initializer[inits[zp]].CopyFrom(
                numpy_helper.from_array(np.zeros(w.shape[0], np.int16), zp)
            )
        dq.domain = "com.microsoft"
        done += 1
    if not any(o.domain == "com.microsoft" for o in q.opset_import):
        q.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    assert done == len(convs16), (done, len(convs16))
    return q


def per_channel_acts(q, fp32, data, tensors, dtype="uint16"):
    """Give the listed activation tensors (original fp32 names) per-channel (axis 1) Q/DQ parameters.

    For a DSP kernel this is free wherever a per-channel scale can be folded: into the producing conv's
    per-output-channel requantization and the consuming conv's weights (1x1: w[:, c] *= s_c; depthwise:
    per channel anyway). Ranges are per-channel min/max of the fp32 model on the calibration data.
    """
    import onnxruntime as ort

    m = onnx.ModelProto()
    m.CopyFrom(fp32)
    del m.graph.output[:]
    for t in tensors:
        m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        m.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    lo = {t: None for t in tensors}
    hi = dict(lo)
    for feed in data:
        for t, v in zip(tensors, sess.run(tensors, feed)):
            a, b = v.min(axis=(0, 2, 3)), v.max(axis=(0, 2, 3))
            lo[t] = a if lo[t] is None else np.minimum(lo[t], a)
            hi[t] = b if hi[t] is None else np.maximum(hi[t], b)
    qmax, npt, tp = (
        (65535, np.uint16, TensorProto.UINT16)
        if dtype == "uint16"
        else (255, np.uint8, TensorProto.UINT8)
    )
    cons = {}
    for n in q.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    done = 0
    for n in list(q.graph.node):
        if n.op_type != "QuantizeLinear":
            continue
        t = n.input[0][:-2] if n.input[0].endswith("/f") else n.input[0]
        if t not in tensors:
            continue
        lo_t, h = np.minimum(lo[t], 0.0), np.maximum(hi[t], 0.0)
        sc = np.maximum(h - lo_t, 1e-8) / qmax
        zp = np.clip(np.round(-lo_t / sc), 0, qmax).astype(npt)
        sn, zn = t + "/pc_scale", t + "/pc_zp"
        q.graph.initializer.extend(
            [
                numpy_helper.from_array(sc.astype(np.float32), sn),
                numpy_helper.from_array(zp, zn),
            ]
        )
        for node in [n] + [
            d for d in cons.get(n.output[0], []) if d.op_type == "DequantizeLinear"
        ]:
            node.input[1], node.input[2] = sn, zn
            del node.attribute[:]
            node.attribute.append(helper.make_attribute("axis", 1))
        done += 1
    return q, done


def build(
    fp32,
    src,
    data,
    convs16,
    act="uint16",
    tensor_dtypes=None,
    ranges=None,
    float_nodes=(),
    method="minmax",
):
    ranges = {**(ranges or {}), **quantize.exact_input_ranges(src)}
    q = quantize_full_qdq(
        src,
        data,
        activation_dtype=act,
        method=method,
        exclude_nodes=head_nodes(src) + list(float_nodes),
        tensor_dtypes=tensor_dtypes or {},
        ranges=ranges,
    )
    return to_w16(q, fp32, set(convs16) - set(float_nodes))


def plan_err(model, road, ref_plan, slices):
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser

    y = run_models.run_driving(
        run_models.session(model.SerializeToString()), road, len(road)
    )
    p = Parser().parse_outputs({k: y[:, v] for k, v in slices.items()})["plan"]
    return float(np.abs(p[:, :, 1] - ref_plan[:, :, 1]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "sweep", "actsweep"])
    ap.add_argument("fp32")
    ap.add_argument("src", help="int8 weight source (fp32 or AdaRound'ed fp32 model)")
    ap.add_argument("calib", help="calibration frames .npz, comma list")
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--kind", choices=["driving", "dm"], default="driving")
    ap.add_argument(
        "--dm-calib",
        default="-0.0003,0.1643,0.005",
        help="DM calib input for calibration samples",
    )
    ap.add_argument("--w16", default="")
    ap.add_argument(
        "--per-channel-acts",
        default="",
        help="comma list of fp32 activation tensors (or 'all4d') "
        "to quantize per channel",
    )
    ap.add_argument(
        "--float-nodes", default="", help="comma list of backbone node names kept float"
    )
    ap.add_argument(
        "--window", type=int, default=8, help="actsweep: backbone nodes per window"
    )
    ap.add_argument(
        "--only",
        action="store_true",
        help="actsweep: quantize only the window instead of floating it",
    )
    ap.add_argument("--policy", help="JSON with tensor_dtypes (activation overrides)")
    ap.add_argument("--act", default="uint16")
    ap.add_argument(
        "--method",
        default="minmax",
        help="activation calibration method (minmax, percentile, mse, ...)",
    )
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--stride", type=int, default=18)
    ap.add_argument("--rank-frames", type=int, default=200)
    ap.add_argument(
        "--range-margin",
        type=float,
        default=1.0,
        help="widen every calibrated activation range by this factor (headroom for frames "
        "outside the calibration set; cheap at 16 bits)",
    )
    args = ap.parse_args()
    fp32, src = onnx.load(args.fp32), onnx.load(args.src)
    data = (
        quantize.driving_samples(args.fp32, args.calib, args.samples, args.stride)
        if args.kind == "driving"
        else quantize.dm_samples(
            args.calib,
            args.samples,
            args.stride,
            [float(v) for v in args.dm_calib.split(",")],
        )
    )
    td = json.load(open(args.policy)).get("tensor_dtypes", {}) if args.policy else {}
    ranges = None
    if args.range_margin != 1.0:
        from onnxsim.calibration import calibrate

        ranges = {
            k: (lo * args.range_margin, hi * args.range_margin)
            for k, (lo, hi) in calibrate(src, data).items()
            if np.isfinite(lo) and np.isfinite(hi)
        }
    if args.cmd == "build":
        q = build(
            fp32,
            src,
            data,
            [c for c in args.w16.split(",") if c],
            args.act,
            td,
            ranges,
            [c for c in args.float_nodes.split(",") if c],
            args.method,
        )
        if args.per_channel_acts:
            import onnx.shape_inference as si

            vi = {v.name: v for v in si.infer_shapes(fp32).graph.value_info}
            ts = (
                [
                    k
                    for k, v in vi.items()
                    if len(v.type.tensor_type.shape.dim) == 4
                    and v.type.tensor_type.elem_type == TensorProto.FLOAT
                ]
                if args.per_channel_acts == "all4d"
                else args.per_channel_acts.split(",")
            )
            q, k = per_channel_acts(q, fp32, data, ts, args.act)
            print(f"{k} activations per channel")
        onnx.save(q, args.rest[0])
        print("wrote", args.rest[0])
        return
    rank_npz, out_json = args.rest
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser

    slices = run_models.output_slices("driving_supercombo.onnx")
    road = np.load(rank_npz)["road"][: args.rank_frames]
    yref = run_models.run_driving(run_models.session(args.fp32), road, len(road))
    ref_plan = Parser().parse_outputs({k: yref[:, v] for k, v in slices.items()})[
        "plan"
    ]
    convs = [n.name for n in fp32.graph.node if n.op_type == "Conv"]
    # calibrate activation ranges once, reuse for every variant
    from onnxsim.calibration import calibrate

    ranges = ranges or calibrate(src, data)
    if args.cmd == "actsweep":
        # every conv W16; one window of backbone nodes at a time kept float: which activations cost most
        ranges = ranges or calibrate(src, data)
        bb = [
            n.name
            for n in quantize.backbone_nodes(fp32)
            if n.op_type in ("Conv", "Gelu", "Add", "Mul", "Sub", "Div", "Concat")
        ]
        res = {
            "base": plan_err(
                build(fp32, src, data, convs, args.act, td, ranges),
                road,
                ref_plan,
                slices,
            ),
            "windows": [],
        }
        print("W16A16 base:", res["base"], flush=True)
        for i in range(0, len(bb), args.window):
            w = bb[i : i + args.window]
            # "only": quantize just this window (every other backbone node float); else keep just it float
            fl = [x for x in bb if x not in w] if args.only else w
            e = plan_err(
                build(fp32, src, data, convs, args.act, td, ranges, fl),
                road,
                ref_plan,
                slices,
            )
            res["windows"].append({"nodes": w, "plan_err": e})
            print(
                f"  {'only' if args.only else 'float'} {w[0]} .. {w[-1]}: {e:.5f}",
                flush=True,
            )
            json.dump(res, open(out_json, "w"), indent=1)
        return
    res = {
        "all16": plan_err(
            build(fp32, src, data, convs, args.act, td, ranges), road, ref_plan, slices
        ),
        "all8": plan_err(
            build(fp32, src, data, [], args.act, td, ranges), road, ref_plan, slices
        ),
        "only8": {},
    }
    print("all w16:", res["all16"], "all w8:", res["all8"], flush=True)
    for c in convs:
        t = time.time()
        res["only8"][c] = plan_err(
            build(fp32, src, data, [x for x in convs if x != c], args.act, td, ranges),
            road,
            ref_plan,
            slices,
        )
        print(
            f"  only {c} int8: {res['only8'][c]:.5f} ({time.time() - t:.0f}s)",
            flush=True,
        )
        json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
