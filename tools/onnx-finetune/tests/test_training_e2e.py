"""End-to-end test of onnx-finetune's actual training path: the half
test_lora.py's own docstring says it deliberately does not cover, because it
needs a training-enabled onnxruntime build (see ../README.md). This is that
coverage -- it runs the README's own toy end-to-end example for real:
scripts/generate_artifacts.py (needs onnxruntime.training.artifacts) followed
by the onnx-finetune C++ binary's actual train loop.

Skipped, with a reason, unless the environment provides a training-enabled
ONNX Runtime build -- see .github/workflows/onnx-finetune-training.yml, the
only place that sets ONNX_FINETUNE_BIN/ORT_TRAINING_PYTHONPATH today. Under
the plain-onnxruntime jobs in onnx-finetune-lora.yml this shows as SKIPPED
rather than simply not existing, so the coverage boundary stays visible there
too.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

ONNX_FINETUNE_BIN = os.environ.get("ONNX_FINETUNE_BIN")
ORT_TRAINING_PYTHONPATH = os.environ.get("ORT_TRAINING_PYTHONPATH")

pytestmark = pytest.mark.skipif(
    not ONNX_FINETUNE_BIN or not ORT_TRAINING_PYTHONPATH,
    reason=(
        "needs a training-enabled ONNX Runtime build; set ONNX_FINETUNE_BIN and "
        "ORT_TRAINING_PYTHONPATH (see tools/onnx-finetune/README.md and "
        ".github/workflows/onnx-finetune-training.yml)"
    ),
)

INPUT_DIM = 4
TARGET_DIM = 1
NUM_SAMPLES = 2048

_LOSS_RE = re.compile(r"loss ([\d.]+)")


def _run(script, *args, env=None):
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True, env=env)


def test_toy_mlp_loss_drops_and_finetuned_model_is_usable(tmp_path):
    toy_model = tmp_path / "toy_model.onnx"
    train_input = tmp_path / "train_input.bin"
    train_target = tmp_path / "train_target.bin"
    artifacts_dir = tmp_path / "artifacts"
    finetuned_model = tmp_path / "finetuned.onnx"

    _run("make_toy_model.py", "-o", str(toy_model))
    _run(
        "make_synthetic_data.py",
        "--num-samples",
        str(NUM_SAMPLES),
        "--input-out",
        str(train_input),
        "--target-out",
        str(train_target),
    )

    # generate_artifacts.py needs onnxruntime.training, which this test's
    # own process does not import (see module docstring) -- only the
    # subprocess gets the training-enabled PYTHONPATH.
    artifacts_env = {**os.environ, "PYTHONPATH": ORT_TRAINING_PYTHONPATH}
    _run(
        "generate_artifacts.py",
        str(toy_model),
        "-o",
        str(artifacts_dir),
        env=artifacts_env,
    )

    result = subprocess.run(
        [
            ONNX_FINETUNE_BIN,
            "--artifacts-dir",
            str(artifacts_dir),
            "--train-input",
            str(train_input),
            "--train-target",
            str(train_target),
            "--input-dim",
            str(INPUT_DIM),
            "--target-dim",
            str(TARGET_DIM),
            "--num-samples",
            str(NUM_SAMPLES),
            "--batch-size",
            "32",
            "--epochs",
            "20",
            "--lr",
            "0.01",
            "--output-model",
            str(finetuned_model),
            "--output-names",
            "output",
            "--log-every",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Secondary check: the tool's own reported loss trend.
    losses = [float(m) for m in _LOSS_RE.findall(result.stdout)]
    assert losses, f"no loss values parsed from onnx-finetune output:\n{result.stdout}"
    assert losses[0] > 1.0, (
        f"first loss {losses[0]} lower than expected for an untrained model"
    )
    assert losses[-1] < losses[0] * 0.1, (
        f"loss did not drop as expected: first={losses[0]} last={losses[-1]}"
    )

    # Primary check, independent of the tool's own log: evaluate the
    # exported model with a *plain* onnxruntime install (this is the process's
    # own import, never given the training PYTHONPATH -- see module docstring)
    # against the raw synthetic data, proving both the numeric claim and the
    # README's "loads with any onnxruntime build" claim in one step.
    x = np.fromfile(train_input, dtype=np.float32).reshape(NUM_SAMPLES, INPUT_DIM)
    y = np.fromfile(train_target, dtype=np.float32).reshape(NUM_SAMPLES, TARGET_DIM)
    session = onnxruntime.InferenceSession(str(finetuned_model))
    (pred,) = session.run(["output"], {"input": x})
    mse = float(np.mean((pred - y) ** 2))
    assert mse < 0.05, f"finetuned model's MSE against training data too high: {mse}"
