"""Score quantized openpilot models against the fp32 model on held-out route segments.

  python evaluate.py driving name=model.onnx [name2=...] --segments 8,5 [--json out.json]

Each model is run sequentially over each segment's 600 frames (recurrent state fed back, as modeld does)
and compared with the fp32 model's run (cached as `ref_<kind>_fp32_s<seg>.npy`) through openpilot's
own output parser. Metrics (lower is better unless noted):

- driving: plan lateral error (m, mean over the 33 plan points), plan lateral error at 10 s (p95),
  lead probability error (mean, p95), lead decision agreement (higher is better), lead distance error
  (m, frames where both see a lead)
- dm: worst mean |probability error| over the face/eye/blink/phone/sleep heads (LHD+RHD), L/R blink decision
  agreement (higher is better)

`fp16=<path to the original fp16 model>` in the model list gives the deployed precision's own error,
the honest floor for a quantized model.
"""

import argparse
import json
import os

import numpy as np
import run_models

SEG_CALIB = {  # extrinsicsCalibration.rpyCalib means from each segment's rlog (read_rlog.py)
    "3": "-0.00041,0.16457,0.00449",
    "5": "-0.00039,0.16419,0.00458",
    "8": "-0.00028,0.16415,0.00528",
    "12": "-0.00023,0.164,0.0056",
}
DM_HEADS = [
    "face_prob",
    "left_eye_prob",
    "right_eye_prob",
    "left_blink_prob",
    "right_blink_prob",
    "using_phone_prob",
    "sleep_prob",
]


def run(kind, model, seg, inputs_dir, threads):
    d = np.load(os.path.join(inputs_dir, f"inputs_seg{seg}.npz"))
    s = run_models.session(model, threads)
    if kind == "driving":
        return run_models.run_driving(s, d["road"], len(d["road"]))
    calib = [float(v) for v in SEG_CALIB[seg].split(",")]
    return run_models.run_dm(s, d["driver"], len(d["driver"]), calib)


def driving_metrics(ref, out, slices):
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser

    def parse(x):
        return Parser().parse_outputs({k: x[:, v] for k, v in slices.items()})

    a, b = parse(ref), parse(out)
    lat = np.abs(a["plan"][:, :, 1] - b["plan"][:, :, 1])
    la, lb = a["lead_prob"][:, 0], b["lead_prob"][:, 0]
    both = (la > 0.5) & (lb > 0.5)
    return {
        "plan_lat_m": float(lat.mean()),
        "plan_lat_10s_p95_m": float(np.percentile(lat[:, -1], 95)),
        "lead_prob_err": float(np.abs(la - lb).mean()),
        "lead_prob_err_p95": float(np.percentile(np.abs(la - lb), 95)),
        "lead_agree": float(((la > 0.5) == (lb > 0.5)).mean()),
        "lead_x_err_m": float(
            np.abs(a["lead"][:, 0, 0, 0] - b["lead"][:, 0, 0, 0])[both].mean()
        )
        if both.any()
        else None,
    }


def dm_metrics(ref, out, slices):
    idx = [slices[f"{h}_{s}"].start for h in DM_HEADS for s in ("lhd", "rhd")]
    pa, pb = run_models.sigmoid(ref[:, idx]), run_models.sigmoid(out[:, idx])
    bl = [slices[f"{h}_lhd"].start for h in ("left_blink_prob", "right_blink_prob")]
    ba, bb = run_models.sigmoid(ref[:, bl]), run_models.sigmoid(out[:, bl])
    agree = ((ba > 0.5) == (bb > 0.5)).mean(0)
    return {
        "worst_head_err": float(np.abs(pa - pb).mean(0).max()),
        "blink_agree_l": float(agree[0]),
        "blink_agree_r": float(agree[1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["driving", "dm"])
    ap.add_argument("models", nargs="+", help="name=path.onnx")
    ap.add_argument(
        "--fp32", help="fp32 reference model (default driving_fp32.onnx / dm_fp32.onnx)"
    )
    ap.add_argument(
        "--slices-model",
        help="original model carrying output_slices (default: the openpilot one)",
    )
    ap.add_argument("--segments", default="8,5")
    ap.add_argument("--inputs-dir", default=".")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", help="append results to this JSON file")
    args = ap.parse_args()
    fp32 = args.fp32 or (
        "driving_fp32.onnx" if args.kind == "driving" else "dm_fp32.onnx"
    )
    slices = run_models.output_slices(
        args.slices_model
        or (
            "driving_supercombo.onnx"
            if args.kind == "driving"
            else "dmonitoring_model.onnx"
        )
    )
    metric = driving_metrics if args.kind == "driving" else dm_metrics
    results = (
        json.load(open(args.json)) if args.json and os.path.exists(args.json) else {}
    )
    for spec in args.models:
        name, path = spec.split("=", 1)
        row = {}
        for seg in args.segments.split(","):
            refp = os.path.join(args.inputs_dir, f"ref_{args.kind}_fp32_s{seg}.npy")
            if not os.path.exists(refp):
                np.save(refp, run(args.kind, fp32, seg, args.inputs_dir, args.threads))
            row[seg] = metric(
                np.load(refp),
                run(args.kind, path, seg, args.inputs_dir, args.threads),
                slices,
            )
        results[name] = row
        print(
            name,
            " | ".join(
                f"s{s}: "
                + ", ".join(f"{k} {v:.4g}" for k, v in m.items() if v is not None)
                for s, m in row.items()
            ),
            flush=True,
        )
        if args.json:
            json.dump(results, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
