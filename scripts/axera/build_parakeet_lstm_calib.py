#!/usr/bin/env python3
"""Build+calibrate `build_parakeet_lstm_train_step.py`'s training-step graph
-- the first real-hardware test of `scripts/axera/legalize.py`'s
`unroll_lstm`.

Follows `build_w2v2_encoder_attn_calib.py`'s established real-data-
calibration pattern exactly (measure a real host float32 SGD trajectory for
every trainable tensor, jitter `lr` around the value that trajectory
actually used, give `grad_seed` real values near 1.0) -- the same recipe
that has turned every other "compiles but nothing moves" build in this
project into "trains correctly."

Usage: build_parakeet_lstm_calib.py OUT_DIR
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

import build_parakeet_lstm_train_step as m  # noqa: E402
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
    # lr=200: a first real host step at lr=0.1 measured this weight's
    # gradient at ~5.3e-6/element (two real per-gate MatMuls plus a full
    # second LSTM layer and the output projector between this tensor and
    # the loss) -- comparable in scale to wav2vec2's own "two real encoder
    # layers deep" gradient (docs/axera-audio-speech-op-coverage.md's
    # w2v2_encoder_attn_calib section, which needed lr=2000 for a ~1e-6
    # gradient). At this scale, lr=200 gives a real per-step update
    # (~1.1e-3) comfortably above an ~8e-4 INT8 quantization step for a
    # weight calibrated to this tensor's own ~0.1 range -- lr=0.1 alone
    # would round to zero under real quantization, the same "gradient real
    # but below the resolution the deployed range can represent" ceiling
    # this project has hit repeatedly.
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
