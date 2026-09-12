#!/usr/bin/env python3
"""Phase 3 of the wav2vec2 multi-phase calibration swap
(`docs/axera-audio-speech-op-coverage.md`'s "Multi-phase calibration swap"
section): same mechanism as `build_w2v2fe_mp_swap_phase2.py`, one step
further -- recalibrate against phase 2's *own* real late-training
trajectory, captured directly off the AX650N running the phase-2 compile
(`w2v2fe_runner_capture.c` again), instead of phase 1's.

Direct evidence this is the same class of problem, not a new one: phase 2's
`quant_axmodel.json` calibrates `loss` to `scale=0.0034355`, a representable
span of `255*scale=0.876`. A fresh real capture off `w2v2_phase2.axmodel`
(steps 5-24, squarely inside its plateau) confirms the real observed
trajectory alternates between exactly two adjacent quantized codes,
`0.862318` and `0.865753` -- a span of `0.0034356`, matching the calibrated
`scale` itself almost exactly. So phase 2's calibration is still ~252x wider
than what the real post-swap trajectory actually uses: the same "MinMax
calibrates to the captures' own min/max, not the much tighter band a
continuing run settles into" mechanism as phase 1's ~7x-too-wide span, just
compounded -- not a different, harder (e.g. SNR-floor) problem.

Usage: build_w2v2fe_mp_swap_phase3.py OUT_DIR CAPTURE_DIR N_CAPTURES
       X0_PATH Y0_PATH FINAL_STATE_PATH
Writes OUT_DIR/step.onnx, OUT_DIR/calib/{dataset,config,step.onnx}, and
OUT_DIR/step.onnx.state0 (seeded from FINAL_STATE_PATH -- phase 2's own real
final weight, for a same-state continuation run against the phase-3 compile)
/ .x0 / .y0 (byte-identical to phase 1/2's, passed in as X0_PATH/Y0_PATH so
all three phases see the same repeating batch).
"""

import os
import shutil
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
    out_dir, capture_dir, n_captures = sys.argv[1], sys.argv[2], int(sys.argv[3])
    x0_path, y0_path, final_state_path = sys.argv[4], sys.argv[5], sys.argv[6]
    os.makedirs(out_dir, exist_ok=True)
    step_path = os.path.join(out_dir, "step.onnx")

    torch.manual_seed(42)
    step_model, state = m.build(step_path, batch=4)
    onnx.save(step_model, step_path)
    state_out = state[WNAME]

    w_shape = tuple(
        d.dim_value
        for d in next(
            i for i in step_model.graph.input if i.name == WNAME
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

    # Real late-stage trajectory, captured off the AX650N running phase 2
    # (see this module's docstring) -- phase 2's own plateau, not phase 1's.
    w_captures = []
    for i in range(n_captures):
        arr = np.fromfile(
            os.path.join(capture_dir, f"w_capture_{i}.bin"), dtype=np.float32
        ).reshape(w_shape)
        w_captures.append(arr)
    print(
        "real phase-2 trajectory: mean|w| ->",
        [float(np.abs(v).mean()) for v in w_captures],
    )

    # x/y: the runner's own fixed, repeating batch, byte-identical across
    # all three phases -- see build_w2v2fe_mp_swap_phase2.py's docstring for
    # why this (not a diverse trajectory) is the real runtime distribution.
    x0 = np.fromfile(x0_path, dtype=np.float32).reshape(x_shape)
    y0 = np.fromfile(y0_path, dtype=np.float32).reshape(y_shape)

    n = len(w_captures)
    host_loss = []
    for w in w_captures:
        feeds = {
            "x": x0,
            "y": y0,
            "lr": np.array([100.0], np.float32),
            "grad_seed": np.array([1.0], np.float32),
            WNAME: w,
        }
        out = run(step_path, feeds)
        host_loss.append(float(np.asarray(out["loss"]).ravel()[0]))
    print("host loss at each real phase-2 capture (x0/y0 fixed):", host_loss)

    work_dir = os.path.join(out_dir, "calib")
    mtc.make_work_dir(
        step_path,
        work_dir,
        n=n,
        label_inputs=(),
        real_data={
            WNAME: w_captures,
            "x": [x0] * n,
            "y": [y0] * n,
            "lr": [
                np.array([v], np.float32)
                for v in [0.01, 0.1, 1.0, 10.0, 100.0, 1.0, 50.0, 100.0]
            ],
            "grad_seed": [
                np.array([v], np.float32)
                for v in [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 1.0]
            ],
        },
    )
    print("wrote calib work dir:", work_dir)

    # Runner companion files: seed from phase 2's real plateaued state (the
    # same weight its own run actually ended at), and phase 1/2's own x/y
    # batch byte-for-byte -- a genuine same-state continuation.
    final_w = np.fromfile(final_state_path, dtype=np.float32)
    final_w.tofile(step_path + ".state0")
    shutil.copyfile(x0_path, step_path + ".x0")
    shutil.copyfile(y0_path, step_path + ".y0")
    print("wrote", step_path + ".state0/.x0/.y0 (seeded from real phase-2 plateau)")


if __name__ == "__main__":
    main()
