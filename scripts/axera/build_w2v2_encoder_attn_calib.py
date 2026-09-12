#!/usr/bin/env python3
"""Build+calibrate `build_w2v2_encoder_attn_step.py`'s training-step graph --
the wav2vec2 attention-output tail
`docs/axera-audio-speech-op-coverage.md` flagged as "not yet done" once
`onnxsim.graph_grad._grad_where`/`_grad_is_nan` closed the backward-rule gap.

Follows `build_w2v2fe_batch_calib.py`'s established real-data-calibration
pattern exactly (measure a real host float32 SGD trajectory, jitter `lr`,
give `grad_seed` real values near 1.0 rather than letting it fall through to
`make_training_calib`'s generic `weight_scale=0.05` random-draw branch --
PR #1373's confirmed bug class), since that pattern is what turned wav2vec2's
feature-extractor build from "compiles but the gradient is dead" into "trains
correctly."

Usage: build_w2v2_encoder_attn_calib.py OUT_DIR
Writes OUT_DIR/step.onnx, OUT_DIR/calib/{dataset,config,step.onnx}, and
OUT_DIR/step.onnx.state0 / .x0 / .y0 for the resident runner's seeding
convention.
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_w2v2_encoder_attn_step as m  # noqa: E402
import make_training_calib as mtc  # noqa: E402


def run(path, feeds):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    step_path = os.path.join(out_dir, "step.onnx")

    step_model, state, wname = m.build(step_path, layer=0, batch=1)
    onnx.save(step_model, step_path)
    state_out = state[wname]
    print(f"trainable param: {wname} -> {state_out}")

    rng = np.random.default_rng(0)
    w_shape = tuple(
        d.dim_value
        for d in next(
            i for i in step_model.graph.input if i.name == wname
        ).type.tensor_type.shape.dim
    )
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

    # real host float32 trajectory: 8 real SGD steps, recording w/x/y at each
    # step for calibration -- same recipe as build_w2v2fe_batch_calib.py's
    # own docstring. lr chosen small (1.0, not 100.0): this graph's gradient
    # w.r.t. one 128x128 attention weight buried behind two real encoder
    # layers' worth of LayerNorm/Softmax/Where is orders of magnitude smaller
    # than the feature-extractor's own conv-weight gradient (confirmed on
    # host: ~1e-6 per element vs. that script's much larger scale), so a
    # large lr would blow the weight trajectory's own scale up unrealistically
    # over 8 steps rather than calibrate the range the real deployed loop
    # would actually see.
    w0 = (rng.standard_normal(w_shape) * 0.1).astype(np.float32)
    ws, xs, ys = [w0.copy()], [], []
    w = w0.copy()
    # 2000, not 1.0: this weight's real host gradient is ~1e-6/element (two
    # real encoder layers deep, one of many weights), far below its own
    # ~8e-4 INT8 quantization step at this ~0.1 scale -- lr*grad needs to
    # clear that step to survive quantization at all, matching this
    # project's own established fix (PR #1370's feature-extractor lr=100,
    # scaled up here for a ~100-1000x smaller gradient).
    lr = 2000.0
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
            wname: w,
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
            wname: ws[:-1],
            "x": xs,
            "y": ys,
            "lr": [
                np.array([v], np.float32)
                for v in [1.0, 10.0, 100.0, 500.0, 1000.0, 2000.0, 1500.0, 2000.0]
            ],
            "grad_seed": [
                np.array([v], np.float32)
                for v in [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 1.0]
            ],
        },
    )
    print("wrote calib work dir:", work_dir)

    w0.tofile(step_path + ".state0")
    xs[0].tofile(step_path + ".x0")
    ys[0].tofile(step_path + ".y0")
    print("wrote", step_path + ".state0/.x0/.y0")


if __name__ == "__main__":
    main()
