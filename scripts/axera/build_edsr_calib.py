#!/usr/bin/env python3
"""Build+calibrate `build_edsr_train_step.py`'s training-step graph -- the
real-hardware test of `onnxsim.graph_grad._grad_depth_to_space`
(`docs/axera-super-resolution-op-coverage.md`'s real-hardware section).

Follows `build_parakeet_lstm_calib.py`'s established real-data-calibration
pattern exactly (measure a real host float32 SGD trajectory for every
trainable tensor, jitter `lr` around the value that trajectory actually
used, give `grad_seed` real values near 1.0) -- the same recipe that has
turned every other "compiles but nothing moves" build in this project into
"trains correctly."

Defaults to `--scope tail --weights-only`, the configuration confirmed
first on real hardware. A rank-1 (bias) tensor's in-graph SGD update used
to crash Pulsar2's NPU backend tiler (`TileFailException("AxQuantizedSub,
tuple index out of range")`, confirmed on two different bias shapes here)
-- **fixed generically in `build_resident_train_step.build_resident_step()`
itself** (reshape the whole per-step update to rank-2 around the `Sub` and
back), so `--scope tail --no-weights-only` (all four tensors, both `Conv`
weights and both biases) now also compiles and trains correctly; see
`docs/axera-super-resolution-op-coverage.md`'s real-hardware section for
the numbers.

A real, distinct calibration-degeneracy trap this domain's own input names
surfaced (not present in any earlier model): `make_training_calib.
make_work_dir`'s generic fallback treats the *first* graph input as the
image-scale ("x") input and any literally-named `"y"` input as a
classification label -- neither matches EDSR's own `lowres`/`hr` naming,
so an unwary call would silently calibrate `hr` (a real image-scale
target) with `weight_scale`-scaled noise instead of `x_scale`-scaled noise,
the same magnitude of mismatch PR #1367 found between wav2vec2's own
`x_scale`/`weight_scale`. Passed explicitly via `real_data` below instead.

Usage: build_edsr_calib.py OUT_DIR [--scope head|tail|all] [--weights-only]
       [--lr LR]
Writes OUT_DIR/edsr_tiny.onnx, OUT_DIR/edsr_step.onnx,
OUT_DIR/calib/{dataset,config,step.onnx}.
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_edsr_train_step as m  # noqa: E402
import make_training_calib as mtc  # noqa: E402


def run(path, feeds):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--scope", choices=["head", "tail", "all"], default="tail")
    parser.add_argument("--weights-only", action="store_true", default=True)
    parser.add_argument("--no-weights-only", dest="weights_only", action="store_false")
    # 1.0: the confirmed real-hardware working value (see the module
    # docstring) -- lr=0.1 rounds to zero under real INT8 quantization
    # (a real, non-tiny gradient, ~3.5e-4/2.1e-3 mean|grad| for the two
    # tail tensors, just still below this scope's resolution at that lr),
    # and lr=10 diverges to NaN within a handful of real host SGD steps.
    parser.add_argument("--lr", type=float, default=1.0)
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    enc_path = os.path.join(args.out_dir, "edsr_tiny.onnx")
    m.export_edsr(enc_path)
    fwd = onnx.shape_inference.infer_shapes(onnx.load(enc_path))
    sr_name = fwd.graph.output[0].name
    fwd = m.add_mse_loss_nchw(fwd, sr_name)

    params = m.trainable_scope(fwd, args.scope, weights_only=args.weights_only)
    step_model, state = m.brts.build_resident_step(fwd, params, loss_output="loss")
    onnx.checker.check_model(step_model)
    step_path = os.path.join(args.out_dir, "edsr_step.onnx")
    onnx.save(step_model, step_path)
    print(f"step graph: {len(step_model.graph.node)} nodes, {len(params)} trainable")
    for p in params:
        print(" ", p)

    shapes = {
        inp.name: tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        for inp in step_model.graph.input
    }

    rng = np.random.default_rng(0)
    w0s = {
        p: (rng.standard_normal(shapes[p]) * 0.05).astype(np.float32) for p in params
    }
    ws = {p: [w0s[p].copy()] for p in params}
    lowres_list, hr_list = [], []
    w = dict(w0s)
    lr = args.lr
    for _ in range(8):
        lowres = (rng.standard_normal(shapes["lowres"]) * 0.3).astype(np.float32)
        hr = (rng.standard_normal(shapes["hr"]) * 0.3).astype(np.float32)
        lowres_list.append(lowres)
        hr_list.append(hr)
        feeds = {
            "lowres": lowres,
            "hr": hr,
            "lr": np.array([lr], np.float32),
            "grad_seed": np.array([1.0], np.float32),
            **w,
        }
        out = run(step_path, feeds)
        w = {p: out[state[p]] for p in params}
        for p in params:
            ws[p].append(w[p].copy())

    print(
        "host trajectory: mean|",
        params[0],
        "| ->",
        [float(np.abs(v).mean()) for v in ws[params[0]]],
    )
    print("loss ->", np.asarray(out["loss"]).flatten()[0])

    real_data = {
        **{p: ws[p][:-1] for p in params},
        "lowres": lowres_list,
        "hr": hr_list,
        "lr": [
            np.array([v], np.float32)
            for v in [lr * m_ for m_ in (0.5, 1, 1.5, 1, 0.5, 1.5, 1, 1)]
        ],
        "grad_seed": [
            np.array([v], np.float32)
            for v in [1.0, 0.9, 1.1, 1.0, 0.95, 1.05, 1.0, 1.0]
        ],
    }

    work_dir = os.path.join(args.out_dir, "calib")
    mtc.make_work_dir(step_path, work_dir, n=8, label_inputs=(), real_data=real_data)
    print("wrote calib work dir:", work_dir)


if __name__ == "__main__":
    main()
