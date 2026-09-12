#!/usr/bin/env python3
"""Build a `pulsar2_docker.build()` work directory for a training-step graph
(`build_resident_train_step.py`'s output): per-input calibration `.tar`s of
`.npy` samples, plus the `config/*.json` Pulsar2 quantization config that
names them.

This exists because every training-step compile so far (the resnet18 speed
work, resnet50's first compile, the batch-scaling and batch+vNPU sweeps) grew
its own ad-hoc, uncommitted copy of this generator in a session scratchpad --
and one bug in that copy shipped unnoticed across three separate compiles
before being tracked down here (`docs/axera-on-device-training-handoff.md`'s
"The loss=0 finding" section has the full story): the one-hot `y` label was
scattered into the **flattened** batch tensor via a single random index
(`arr.reshape(-1)[rng.integers(0, arr.size)] = 1.0`), which at batch size 1
is indistinguishable from a correct per-row one-hot but at batch>1 leaves
`(batch-1)/batch` of the rows an all-zero "label" in every calibration
sample. That degenerate calibration data miscalibrated the loss output's
quantization range enough to clip a real, non-degenerate runtime loss down to
exactly 0 -- confirmed by comparing the per-sample squared error (tapped as
an extra debug output before the batch-mean reduction, which read correctly
non-zero and identical across rows) against the final scalar loss (which
read exactly 0) on real AX650N hardware, then fixing the generator and
confirming the same build now reads a real, consistent, non-zero loss.

Kept intentionally small and specific to this repo's training-step graphs
(a `y` one-hot label, an `lr` scalar, everything else a plain random-normal
weight or activation) rather than generalized into a configurable calibration
framework -- expand it if a genuinely different loss/label shape needs it,
not speculatively.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from typing import Sequence

import numpy as np
import onnx


def make_work_dir(
    step_onnx_path: str,
    work_dir: str,
    seed: int = 0,
    n: int = 4,
    label_inputs: Sequence[str] = ("y",),
    x_scale: float = 0.3,
    weight_scale: float = 0.05,
    real_data: dict = None,
) -> str:
    """Writes `work_dir/step.onnx`, `work_dir/dataset/*.tar` and
    `work_dir/config/*.json` for `pulsar2_docker.build(work_dir, "step.onnx",
    ..., config_path="config/step.json")`.

    :param real_data: `{input_name: array}` or `{input_name: [array, ...]}`
            overrides for specific inputs, instead of a fresh
            `weight_scale`-scaled random draw. A single array is calibrated
            with a small per-sample jitter, `n` times (only safe when the
            tensor genuinely doesn't move much across steps); a list of
            arrays is used directly as the `n` calibration samples, cycling
            if shorter than `n` -- the right choice for a tensor whose real
            trajectory across a few real training steps is known (see this
            function's own note below on why a too-narrow single-point
            jitter silently miscalibrates the range).
            Exists because a random draw only matches a real model's *own*
            calibration-relevant statistics (the resulting gradient's own
            magnitude, downstream of the whole forward+backward pass, per
            `docs/axera-quantizer-reverse-engineering.md`) when the model's
            weights are themselves i.i.d. random and uncorrelated across
            layers -- not true of a real trained/initialized network, where
            structured (not random) weights can produce a very different
            gradient magnitude than random noise at the same scale. Pass the
            model's own real initializer values here to calibrate against
            what the deployed model will actually see.
    :param label_inputs: input names to fill with a one-hot label, placed
            **once per batch row** (`arr[row, rng.integers(0, classes)] =
            1.0`) rather than once in the flattened tensor -- see this
            module's docstring for why that distinction matters at batch>1.
    :param n: calibration sample count. `x`/other inputs get a fresh random
            draw per sample; `lr` is constant; label inputs get a fresh
            per-row one-hot per sample.
    """
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(work_dir + "/dataset", exist_ok=True)
    os.makedirs(work_dir + "/config", exist_ok=True)
    model = onnx.load(step_onnx_path)
    onnx.save(model, work_dir + "/step.onnx")

    rng = np.random.default_rng(seed)
    real_data = real_data or {}
    x_name = model.graph.input[0].name
    input_configs = []
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        tar_path = f"dataset/{inp.name.replace('/', '_').replace(':', '_')}.tar"
        with tarfile.open(work_dir + "/" + tar_path, "w") as tf:
            for i in range(n):
                if inp.name in real_data:
                    entry = real_data[inp.name]
                    if isinstance(entry, (list, tuple)):
                        # a real multi-step trajectory (e.g. this tensor's
                        # own value at steps 0..k of an actual host-side
                        # training loop) -- use the samples directly, cycling
                        # if there are fewer than `n`. This is the fix for a
                        # real, confirmed failure mode: jittering a *single*
                        # snapshot by a small relative amount (the base-case
                        # branch below) calibrates a range too narrow to
                        # survive even one real SGD step's own movement,
                        # rounding every later step's gradient to zero
                        # (`docs/axera-on-device-training-handoff.md`'s
                        # Whisper section has the confirmed real-hardware
                        # evidence this was fixed against).
                        arr = np.asarray(entry[i % len(entry)], dtype=np.float32)
                    else:
                        base = np.asarray(entry, dtype=np.float32)
                        jitter_scale = 1e-3 * (np.abs(base).max() + 1e-12)
                        arr = (
                            base + rng.standard_normal(base.shape) * jitter_scale
                        ).astype(np.float32)
                elif inp.name in label_inputs:
                    arr = np.zeros(dims, dtype=np.float32)
                    batch, classes = dims[0], dims[1]
                    for row in range(batch):
                        arr[row, rng.integers(0, classes)] = 1.0
                elif inp.name == "lr":
                    arr = np.array([1e-4], dtype=np.float32)
                elif inp.name == x_name:
                    arr = (rng.standard_normal(dims) * x_scale).astype(np.float32)
                else:
                    # a trainable weight's state input: a plausible-scale
                    # random draw, not the model's own initializer -- the
                    # state inputs of a resident step graph carry no
                    # initializer of their own (see build_resident_step),
                    # so there is nothing else to center calibration on.
                    arr = (rng.standard_normal(dims) * weight_scale).astype(np.float32)
                buf = io.BytesIO()
                np.save(buf, arr)
                data = buf.getvalue()
                ti = tarfile.TarInfo(name=f"{i}.npy")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        input_configs.append(
            {
                "tensor_name": inp.name,
                "calibration_dataset": f"./{tar_path}",
                "calibration_format": "Numpy",
                "calibration_size": n,
            }
        )

    config = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": input_configs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    with open(work_dir + "/config/step.json", "w") as f:
        json.dump(config, f, indent=2)
    return work_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_onnx")
    parser.add_argument("work_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument(
        "--label-input",
        action="append",
        dest="label_inputs",
        help="an input name that carries a one-hot label; repeat for more "
        "than one. Defaults to just 'y'.",
    )
    args = parser.parse_args(argv)
    label_inputs = args.label_inputs or ["y"]
    out = make_work_dir(
        args.step_onnx,
        args.work_dir,
        seed=args.seed,
        n=args.n,
        label_inputs=label_inputs,
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
