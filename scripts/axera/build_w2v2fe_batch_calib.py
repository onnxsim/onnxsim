#!/usr/bin/env python3
"""Build+calibrate the wav2vec2 feature-extractor training step at a given
batch size, reusing PR #1370's real-data calibration fix (measure the real
host float32 trajectory, jitter lr) rather than PR #1367's arbitrary
x_scale/weight_scale defaults that produced a dead gradient.

**Confirmed real on hardware at batch=1 only.** batch=4/8 compile and run,
but their real trajectory-derived calibration still leaves the gradient
completely dead (loss bit-identical across all 8 real steps) -- a new
calibration-range regression at batch>1, not yet root-caused. See
`docs/axera-on-device-training-handoff.md`'s wav2vec2 batching section.
Treat batch>1 output here as "compiles and times, correctness unconfirmed,"
not as working training.

Usage: build_w2v2fe_batch_calib.py OUT_DIR BATCH
Writes OUT_DIR/step.onnx, OUT_DIR/calib/{dataset,config,step.onnx} (via
make_training_calib.make_work_dir -- pass OUT_DIR/calib as pulsar2_docker.
build()'s own work_dir, config_path="config/step.json"), and
OUT_DIR/step.onnx.state0 / .x0 / .y0 for the resident runner's seeding
convention (a single real sample, regardless of batch -- the runner reads
exactly `in_sz[x_in]` bytes, which already reflects the compiled model's
own batch size).
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_w2v2_feature_extractor_step as m  # noqa: E402
import make_training_calib as mtc  # noqa: E402

WNAME = "fe.conv_layers.0.conv.weight"


def run(path, feeds):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def main():
    out_dir, batch = sys.argv[1], int(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    step_path = os.path.join(out_dir, "step.onnx")

    torch.manual_seed(42)
    step_model, state = m.build(step_path, batch=batch)
    onnx.save(step_model, step_path)
    state_out = state[WNAME]

    rng = np.random.default_rng(0)
    w_shape = tuple(
        d.dim_value
        for d in next(
            i for i in step_model.graph.input if i.name == WNAME
        ).type.tensor_type.shape.dim
    )
    w0 = (rng.standard_normal(w_shape) * 0.36).astype(np.float32)
    x_shape = tuple(
        d.dim_value
        for d in next(
            i for i in step_model.graph.input if i.name == "x"
        ).type.tensor_type.shape.dim
    )
    y_shape = tuple(
        d.dim_value
        for d in next(
            i for i in step_model.graph.input if i.name == "y"
        ).type.tensor_type.shape.dim
    )
    # real host float32 trajectory: 8 real SGD steps at lr=100 (PR #1370's
    # own working config), recording w/x/y at each step for calibration.
    ws, xs, ys = [w0.copy()], [], []
    w = w0.copy()
    lr = 100.0
    for _ in range(8):
        x = rng.standard_normal(x_shape).astype(np.float32)
        y = rng.standard_normal(y_shape).astype(np.float32)
        xs.append(x)
        ys.append(y)
        feeds = {
            "x": x,
            "y": y,
            "lr": np.array([lr], np.float32),
            "grad_seed": np.array([1.0], np.float32),
            WNAME: w,
        }
        out = run(step_path, feeds)
        w = out[state_out]
        ws.append(w.copy())

    print("host trajectory: mean|w| ->", [float(np.abs(v).mean()) for v in ws])

    work_dir = os.path.join(out_dir, "calib")
    mtc.make_work_dir(
        step_path,
        work_dir,
        n=8,
        label_inputs=(),
        real_data={
            WNAME: ws[:-1],
            "x": xs,
            "y": ys,
            "lr": [
                np.array([v], np.float32)
                for v in [0.01, 0.1, 1.0, 10.0, 100.0, 1.0, 50.0, 100.0]
            ],
        },
    )
    print("wrote calib work dir:", work_dir)

    # runner companion files: initial weight + one real x/y sample.
    w0.tofile(step_path + ".state0")
    xs[0].tofile(step_path + ".x0")
    ys[0].tofile(step_path + ".y0")
    print("wrote", step_path + ".state0/.x0/.y0")


if __name__ == "__main__":
    main()
