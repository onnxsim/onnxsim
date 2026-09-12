#!/usr/bin/env python3
"""Phase 2 of a multi-phase calibration swap for wav2vec2's real plateau
(`docs/axera-audio-speech-op-coverage.md`'s "2,000-step plateau" section):
same step graph as `build_w2v2fe_batch_calib.py`'s batch=4 build, but the
trainable weight's calibration data is a real late-training trajectory
captured directly off the AX650N (`w2v2fe_runner_capture.c`, 10 snapshots
at steps 700-1600 of the real PR #1376 run) instead of an 8-step host
simulation seeded from a fresh random init.

Direct evidence this narrows what matters: the phase-1 `quant_axmodel.json`
calibrates `loss` to `scale=0.0044175` (`zero_point=0`), a representable
span of `255*scale=1.126` -- but the real observed loss trajectory across
the whole 2,000-step run only ever occupies `[0.8702, 1.0293]`, a span of
`0.159`, ~7x narrower. The weight tensor calibrated against this script's
captured real late-stage values should produce a correspondingly narrower
`loss` calibration, giving the quantizer more resolution exactly where the
plateau lives -- the untested follow-on the plateau writeup named.

`x`/`y` are the runner's own fixed, repeating `.x0`/`.y0` batch -- not a
diverse trajectory -- because that IS the real runtime distribution here:
`w2v2fe_runner_capture.c` reads `.x0`/`.y0` once and reuses the identical
host buffer every step, so calibrating against anything else would
reintroduce the exact "calibration data uncorrelated with real runtime
values" bug class this project has hit four times before. A host check
(`(w_capture[i], x0, y0)` through the exported step graph on CPU) confirms
this combination reproduces the real observed plateau loss almost exactly
(0.8717-0.8761 across the 10 captures, against the real hardware's
0.8702-0.8791 for the same step window) -- the earlier version of this
script paired the late-stage weight captures with an unrelated early-
training x/y trajectory and left the loss tensor's calibrated range
unchanged (`scale` moved by <1%), because the computed calibration loss
values never actually fell in the real plateau band either. `lr`/
`grad_seed` keep phase 1's real_data (already covers each tensor's real
constant runtime value, 100.0 and 1.0 respectively).

Usage: build_w2v2fe_mp_swap_phase2.py OUT_DIR CAPTURE_DIR N_CAPTURES
Writes OUT_DIR/step.onnx, OUT_DIR/calib/{dataset,config,step.onnx}, and
OUT_DIR/step.onnx.state0 (seeded from CAPTURE_DIR/final.state0 -- the real
plateaued weight, for a same-state continuation run against the phase-2
compile) / .x0 / .y0 (byte-identical to phase 1's, passed in as
X0_PATH/Y0_PATH so both phases see the same repeating batch).
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

    # Real late-stage trajectory, captured directly off the AX650N running
    # phase 1 (see this module's docstring) -- not a host simulation.
    w_captures = []
    for i in range(n_captures):
        arr = np.fromfile(
            os.path.join(capture_dir, f"w_capture_{i}.bin"), dtype=np.float32
        ).reshape(w_shape)
        w_captures.append(arr)
    print(
        "real late-stage trajectory: mean|w| ->",
        [float(np.abs(v).mean()) for v in w_captures],
    )

    # x/y: the runner's own fixed, repeating batch -- see this module's
    # docstring for why this (not a diverse trajectory) is the real
    # runtime distribution paired with the captured weight trajectory.
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
    print("host loss at each real capture (x0/y0 fixed):", host_loss)

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

    # Runner companion files: seed from the real plateaued state (the same
    # weight phase 1's own 1,700-step run actually ended at), and phase 1's
    # own x/y batch byte-for-byte -- a genuine same-state continuation.
    final_w = np.fromfile(final_state_path, dtype=np.float32)
    final_w.tofile(step_path + ".state0")
    import shutil

    shutil.copyfile(x0_path, step_path + ".x0")
    shutil.copyfile(y0_path, step_path + ".y0")
    print("wrote", step_path + ".state0/.x0/.y0 (seeded from real plateau state)")


if __name__ == "__main__":
    main()
