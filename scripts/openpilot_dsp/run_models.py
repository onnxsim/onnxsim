"""Run openpilot's driving/DM ONNX models on prepared route frames with ONNX Runtime (CPU).

- `to-fp32`: rewrites a fp16 model (fp16 initializers, Cast(to=fp16)) as fp32, so PTQ tools and a
  fp32 reference can use it.
- `run`: runs a model over `prepare_inputs.py` frames. The driving model is run the way modeld runs
  it: one call per 20 Hz frame with its recurrent `state_*` inputs fed back from its own `next_state_*`
  outputs (zeros at start), desire 0, traffic_convention LHD, a fixed action_t. Saves raw outputs.
- `compare`: compares two runs' outputs with openpilot's own output parser: plan position/velocity in
  metres (m/s), lead probability and distance, lane-line probabilities; DM face/eye probabilities.
"""

import argparse
import base64
import pickle
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

ACTION_T = np.array(
    [[0.25, 0.55]], np.float32
)  # nominal lat/long action_t (s); the route's delays are not read


def to_fp32(src, dst):
    m = onnx.load(src)
    g = m.graph
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT16:
            a = numpy_helper.to_array(t).astype(np.float32)
            g.initializer[i].CopyFrom(numpy_helper.from_array(a, t.name))
    for n in g.node:
        for a in n.attribute:
            if n.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(
                    numpy_helper.from_array(
                        numpy_helper.to_array(a.t).astype(np.float32), a.t.name
                    )
                )
    for v in list(g.value_info) + list(g.input) + list(g.output):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def output_slices(path):
    m = onnx.load(path, load_external_data=False)
    md = {p.key: p.value for p in m.metadata_props}
    return pickle.loads(base64.b64decode(md["output_slices"]))


def session(path, threads=8):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    # basic level: QDQ stays simulated in float, instead of ORT's fused u8s8 kernels (which saturate on
    # CPUs without VNNI and requantize differently from the reference Q/DQ semantics)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def tinygrad_session(path, device="DSP", target="snapdragon845"):
    """Create the generic tinygrad session used as the SNPE replacement path."""
    from tinygrad_runner import TinygradSession

    return TinygradSession(path, device, target)


def run_driving(sess, road, n):
    ins = {i.name: i for i in sess.get_inputs()}
    ft = lambda name: np.float16 if "float16" in ins[name].type else np.float32  # noqa: E731
    state = {
        k: np.zeros(ins[k].shape, np.uint8 if "uint8" in ins[k].type else ft(k))
        for k in ins
        if k.startswith("state_")
    }
    fixed = {
        "desire": np.zeros(ins["desire"].shape, ft("desire")),
        "traffic_convention": np.array([[1, 0]], ft("traffic_convention")),
        "action_t": ACTION_T.astype(ft("action_t")),
    }
    names = [o.name for o in sess.get_outputs()]
    outs = []
    for i in range(n):
        res = dict(zip(names, sess.run(None, {"new_img": road[i], **state, **fixed})))
        for k in state:
            state[k] = res["next_" + k]
        outs.append(res["outputs"].astype(np.float32)[0])
    return np.stack(outs)


def run_dm(sess, driver, n, calib):
    ins = {i.name: i for i in sess.get_inputs()}
    c = np.array([calib], np.float16 if "float16" in ins["calib"].type else np.float32)
    return np.stack(
        [
            sess.run(None, {"input_img": driver[i], "calib": c})[0].astype(np.float32)[
                0
            ]
            for i in range(n)
        ]
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compare_driving(a, b, slices):
    sys.path.insert(0, ".")
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser

    p = Parser()

    def parse(x):
        return p.parse_outputs({k: x[:, v] for k, v in slices.items()})

    pa, pb = parse(a), parse(b)
    plan_a, plan_b = pa["plan"][:, :, :], pb["plan"][:, :, :]  # [N, 33, 15]
    pos_err = np.abs(plan_a[:, :, 0:3] - plan_b[:, :, 0:3])
    vel_err = np.abs(plan_a[:, :, 3:6] - plan_b[:, :, 3:6])
    lead_a, lead_b = pa["lead_prob"][:, 0], pb["lead_prob"][:, 0]
    lead_x_a, lead_x_b = pa["lead"][:, 0, 0, 0], pb["lead"][:, 0, 0, 0]
    lead_mask = (lead_a > 0.5) & (lead_b > 0.5)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "raw_cos_mean": float(cos.mean()),
        "raw_cos_min": float(cos.min()),
        "plan_lateral_y_err_m_mean": float(pos_err[:, :, 1].mean()),
        "plan_lateral_y_err_m_at_10s_p95": float(np.percentile(pos_err[:, -1, 1], 95)),
        "plan_long_x_err_m_at_10s_mean": float(pos_err[:, -1, 0].mean()),
        "plan_vel_err_mps_mean": float(vel_err[:, :, 0].mean()),
        "lead_prob_absdiff_mean": float(np.abs(lead_a - lead_b).mean()),
        "lead_detect_agree": float(((lead_a > 0.5) == (lead_b > 0.5)).mean()),
        "lead_x_err_m_mean": float(np.abs(lead_x_a - lead_x_b)[lead_mask].mean())
        if lead_mask.any()
        else None,
        "lead_x_ref_m_mean": float(lead_x_a[lead_mask].mean())
        if lead_mask.any()
        else None,
        "lane_prob_absdiff_mean": float(
            np.abs(pa["lane_lines_prob"] - pb["lane_lines_prob"]).mean()
        ),
    }


def compare_dm(a, b, slices):
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    r = {"raw_cos_mean": float(cos.mean()), "raw_cos_min": float(cos.min())}
    for k, s in slices.items():
        if "prob" in k and (s.stop - (s.start or 0)) == 1:
            pa, pb = sigmoid(a[:, s]), sigmoid(b[:, s])
            r[k + "_absdiff_mean"] = float(np.abs(pa - pb).mean())
            r[k + "_decision_agree"] = float(((pa > 0.5) == (pb > 0.5)).mean())
    fd = [k for k in slices if k.startswith("face_descs")]
    for k in fd:
        r[k + "_absdiff_mean"] = float(
            np.abs(a[:, slices[k]] - b[:, slices[k]])[:, :6].mean()
        )
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("to-fp32")
    c.add_argument("src")
    c.add_argument("dst")
    r = sp.add_parser("run")
    r.add_argument("kind", choices=["driving", "dm"])
    r.add_argument("model")
    r.add_argument("inputs")
    r.add_argument("out")
    r.add_argument("--frames", type=int, default=600)
    r.add_argument("--calib", default="0,0.164,0.005")
    r.add_argument("--threads", type=int, default=8)
    m = sp.add_parser("compare")
    m.add_argument("kind", choices=["driving", "dm"])
    m.add_argument("slices_model")
    m.add_argument("ref")
    m.add_argument("test")
    args = ap.parse_args()
    if args.cmd == "to-fp32":
        to_fp32(args.src, args.dst)
    elif args.cmd == "run":
        d = np.load(args.inputs)
        s = session(args.model, args.threads)
        n = min(args.frames, len(d["road"]))
        y = (
            run_driving(s, d["road"], n)
            if args.kind == "driving"
            else run_dm(s, d["driver"], n, [float(v) for v in args.calib.split(",")])
        )
        np.save(args.out, y)
        print(args.out, y.shape)
    else:
        sl = output_slices(args.slices_model)
        a, b = np.load(args.ref), np.load(args.test)
        res = (
            compare_driving(a, b, sl)
            if args.kind == "driving"
            else compare_dm(a, b, sl)
        )
        for k, v in res.items():
            print(f"  {k}: {v:.5g}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
