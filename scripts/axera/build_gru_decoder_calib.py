#!/usr/bin/env python3
"""Build+calibrate `build_gru_decoder_train_step.py`'s training-step graph
-- the first real-hardware test of `scripts/axera/legalize.py`'s
`unroll_gru`, the GRU counterpart of `build_parakeet_lstm_calib.py`.

Follows that module's established real-data-calibration pattern exactly
(measure a real host float32 SGD trajectory for every trainable tensor,
jitter `lr` around the value that trajectory actually used, give
`grad_seed` real values near 1.0).

Usage: build_gru_decoder_calib.py OUT_DIR
Writes OUT_DIR/step.onnx, OUT_DIR/calib/{dataset,config,step.onnx}.
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_gru_decoder_train_step as m  # noqa: E402
import make_training_calib as mtc  # noqa: E402


def run(path, feeds):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    step_path = os.path.join(out_dir, "step.onnx")

    forward = m.build_forward()
    with_loss = m.add_loss_3d(forward, "decoder_output")
    params = m.layer0_param_names()
    step_model, state = m.brts.build_resident_step(
        with_loss, params, loss_output="loss"
    )
    onnx.checker.check_model(step_model)
    onnx.save(step_model, step_path)
    print(f"step graph: {len(step_model.graph.node)} nodes, {len(params)} trainable")

    x_name = m._EMBEDDED_INPUT
    shapes = {
        inp.name: tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        for inp in step_model.graph.input
    }

    rng = np.random.default_rng(0)
    w0s = {p: (rng.standard_normal(shapes[p]) * 0.1).astype(np.float32) for p in params}
    ws = {p: [w0s[p].copy()] for p in params}
    xs, ys = [], []
    w = dict(w0s)
    # lr chosen the same way build_parakeet_lstm_calib.py's own comment
    # explains: measure the real gradient scale at a small lr first, then
    # pick a value that clears this tensor's own INT8 quantization step.
    lr = 200.0
    for _ in range(8):
        x = (rng.standard_normal(shapes[x_name]) * 0.3).astype(np.float32)
        y = (rng.standard_normal(shapes["y"]) * 0.3).astype(np.float32)
        xs.append(x)
        ys.append(y)
        feeds = {
            x_name: x,
            "y": y,
            "lr": np.array([lr], np.float32),
            "grad_seed": np.array([1.0], np.float32),
            **w,
        }
        out = run(step_path, feeds)
        w = {p: out[state[p]] for p in params}
        for p in params:
            ws[p].append(w[p].copy())

    print(
        "host trajectory: mean|w0| ->", [float(np.abs(v).mean()) for v in ws[params[0]]]
    )

    real_data = {
        **{p: ws[p][:-1] for p in params},
        x_name: xs,
        "y": ys,
        "lr": [
            np.array([v], np.float32)
            for v in [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 200.0, 200.0]
        ],
        "grad_seed": [
            np.array([v], np.float32)
            for v in [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 1.0]
        ],
    }

    work_dir = os.path.join(out_dir, "calib")
    mtc.make_work_dir(step_path, work_dir, n=8, label_inputs=(), real_data=real_data)
    print("wrote calib work dir:", work_dir)


if __name__ == "__main__":
    main()
