#!/usr/bin/env python3
"""Calibration for testing the FP32 `grad_seed` / `finetune.LossScaler`
mechanism against Whisper's real `last_half` training step, not just the
small isolated probes it was previously validated on.

Real captured initializer values (+0.1% jitter) for the 14 trainable state
tensors, matching "Attempt 1" in `docs/axera-on-device-training-handoff.md`'s
"Recalibrating Whisper for its real gradient scale" section. `grad_seed`
gets a wide `real_data` *list* (not a single jittered sample) spanning the
sweep an experiment intends to run at runtime -- the "`real_data={...}`
remains the right tool for" widening pattern `make_training_calib.py`'s own
`lr` handling documents, applied here to `grad_seed` instead; without it,
`grad_seed` falls into that module's generic `weight_scale`-random fallback,
the same narrow/uncorrelated-range calibration-degeneracy bug class this
project has hit repeatedly for other scalar inputs.

Usage::

    build_whisper_train_step.py OUT_DIR   # writes whisper_base_enc.onnx,
                                           # whisper_step_last_half.onnx, etc.
    build_whisper_grad_seed_calib.py OUT_DIR OUT_DIR/work

writes `OUT_DIR/work/step.onnx`, `.../dataset/*.tar` and
`.../config/step.json`, ready for `pulsar2_docker.build(work_dir,
"step.onnx", ..., config_path="config/step.json")`. See
`docs/axera-on-device-training-handoff.md`'s "The multi-thousand-step
LossScaler run, attempted on Whisper's real graph" section for what this
was built to test and what it found (a real Pulsar2 NPU-backend
`ddr_allocate` crash on any `layer_configs` FP32 override at this graph's
scale, on every `op_types` combination tried -- the experiment itself was
blocked before a `LossScaler` loop could run against it).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_resident_train_step as brts  # noqa: E402
import build_whisper_train_step as bwts  # noqa: E402
import make_training_calib as mtc  # noqa: E402

#: The small Conv/Conv/Gemm probe's own tested range was 1 -> 2**24; anchored
#: lower here since Whisper's true gradient is 4-5 orders of magnitude
#: smaller than that probe's (see the handoff doc's "SNR floor" finding).
DEFAULT_GRAD_SEED_SWEEP = (1.0, 1e2, 1e4, 1e6, 1e8, 1e10)


def real_trainable_state(out_dir: str, params: "list[str]") -> "dict[str, np.ndarray]":
    """The real (pre-scope-promotion) initializer value for each of
    `params` -- the same folded model `build_whisper_train_step.main`
    derives its trainable scopes from, so names match exactly.
    """
    model = onnx.shape_inference.infer_shapes(
        onnx.load(os.path.join(out_dir, "whisper_base_enc.onnx"))
    )
    fwd = bwts.add_loss_3d(model, model.graph.output[0].name)
    fwd = brts._fold_constants(fwd)
    init_by_name = {
        t.name: onnx.numpy_helper.to_array(t) for t in fwd.graph.initializer
    }
    return {p: init_by_name[p].astype(np.float32) for p in params}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="build_whisper_train_step.py's own OUT_DIR")
    parser.add_argument("work_dir")
    parser.add_argument("--scope", default="last_half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument(
        "--grad-seed-sweep",
        type=float,
        nargs="+",
        default=list(DEFAULT_GRAD_SEED_SWEEP),
    )
    args = parser.parse_args(argv)

    params = [
        line.strip()
        for line in open(os.path.join(args.out_dir, f"{args.scope}.params.txt"))
        if line.strip()
    ]
    real_data = real_trainable_state(args.out_dir, params)
    real_data["grad_seed"] = [
        np.array([v], dtype=np.float32) for v in args.grad_seed_sweep
    ]

    work_dir = mtc.make_work_dir(
        os.path.join(args.out_dir, f"whisper_step_{args.scope}.onnx"),
        args.work_dir,
        seed=args.seed,
        n=args.n,
        real_data=real_data,
        label_inputs=(),  # y is a dense MSE target, not a one-hot label
    )
    print("wrote", work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
