"""Per-group int8 / int16 weights for the heads (Gemm/MatMul outside the conv backbone), fake-quantized.

  sweep: python head_bits.py sweep <fp32> <base model> <rank npz> out.json   (rank each head group by the
         damage of making only it int8, everything else int16, free-running plan lateral error)
  build: python head_bits.py build <fp32> <base model> out.onnx --w8 <group,group,...> (the rest int16)

Groups are weight-name prefixes (3 dotted components, e.g. `model/vision_model.policy.hydra`). Weights
are rounded per output channel onto the int8 / int16 grid and stored as float, so the heads keep float
activations here: this measures weight precision only, which is what the head GEMVs stream (§8).
"""

import argparse
import json

import numpy as np
import onnx
import quantize
from mixed_bits import plan_err
from onnx import numpy_helper


def head_weights(fp32):
    bb = {n.name for n in quantize.backbone_nodes(fp32)}
    init = {i.name for i in fp32.graph.initializer}
    out = {}
    for n in fp32.graph.node:
        if n.op_type in ("Gemm", "MatMul") and n.name not in bb and n.input[1] in init:
            tb = any(a.name == "transB" and a.i for a in n.attribute)
            out[n.input[1]] = (
                0 if (n.op_type == "Gemm" and tb) else 1
            )  # output-channel axis
    return out


def group(name):
    return ".".join(name.split(".")[:3])


def apply(base, fp32, w8_groups):
    m = onnx.ModelProto()
    m.CopyFrom(base)
    fw = {i.name: numpy_helper.to_array(i) for i in fp32.graph.initializer}
    inits = {i.name: k for k, i in enumerate(m.graph.initializer)}
    for name, ax in head_weights(fp32).items():
        a = fw[name].astype(np.float32)
        if a.ndim != 2 or name not in inits:
            continue
        qmax = 127 if group(name) in w8_groups else 32767
        s = np.abs(a).max(axis=1 - ax, keepdims=True) / qmax
        s[s == 0] = 1
        m.graph.initializer[inits[name]].CopyFrom(
            numpy_helper.from_array(
                (np.clip(np.round(a / s), -qmax, qmax) * s).astype(np.float32), name
            )
        )
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["sweep", "build"])
    ap.add_argument("fp32")
    ap.add_argument("base")
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--w8", default="")
    ap.add_argument("--rank-frames", type=int, default=600)
    ap.add_argument(
        "--metric",
        choices=["plan", "lead"],
        default="plan",
        help="rank by plan lateral error or by lead probability error (p95)",
    )
    args = ap.parse_args()
    fp32, base = onnx.load(args.fp32), onnx.load(args.base)
    groups = sorted({group(n) for n in head_weights(fp32)})
    if args.cmd == "build":
        onnx.save(apply(base, fp32, set(args.w8.split(","))), args.rest[0])
        return
    import run_models
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser

    slices = run_models.output_slices("driving_supercombo.onnx")
    road = np.load(args.rest[0])["road"][: args.rank_frames]
    y = run_models.run_driving(run_models.session(args.fp32), road, len(road))
    if args.metric == "plan":
        ref = Parser().parse_outputs({k: y[:, v] for k, v in slices.items()})["plan"]

        def err(m):
            return plan_err(m, road, ref, slices)
    else:
        lref = Parser().parse_outputs({k: y[:, v] for k, v in slices.items()})[
            "lead_prob"
        ][:, 0]

        def err(m):
            yy = run_models.run_driving(
                run_models.session(m.SerializeToString()), road, len(road)
            )
            lp = Parser().parse_outputs({k: yy[:, v] for k, v in slices.items()})[
                "lead_prob"
            ][:, 0]
            return float(np.percentile(np.abs(lp - lref), 95))

    res = {
        "metric": args.metric,
        "all16": err(apply(base, fp32, set())),
        "all8": err(apply(base, fp32, set(groups))),
        "only8": {},
    }
    print("all int16:", res["all16"], "all int8:", res["all8"], flush=True)
    for g in groups:
        res["only8"][g] = err(apply(base, fp32, {g}))
        print(f"  only {g} int8: {res['only8'][g]:.5f}", flush=True)
        json.dump(res, open(args.rest[1], "w"), indent=1)


if __name__ == "__main__":
    main()
