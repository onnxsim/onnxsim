"""End-to-end tests for onnx-finetune's distillation support:
scripts/distillation_loss.py's custom onnxblock KD loss, wired through
scripts/generate_artifacts.py --loss distillation.

Needs a training-enabled onnxruntime build (onnxruntime.training.artifacts/
api), same as generate_artifacts.py itself -- see ../README.md. Unlike
test_lora.py (pure graph surgery, only onnx + numpy), this file skips
without that build rather than failing, matching how heavy/optional
dependencies are handled elsewhere in this repo (see e.g.
tests/test_export_transformers.py in the repo root).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

pytest.importorskip("onnxruntime.training.artifacts")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True)


def _clamp_ir_version(path, max_version=10):
    # The training runtime's bundled C++ engine (as of the onnxruntime-
    # training version this was tested against) rejects any IR version
    # newer than this; the installed onnx package's own default can exceed
    # it. See generate_web_artifacts.py in examples/llm_distillation/wasm_demo/
    # for the same fix, applied there for the same reason.
    model = onnx.load(str(path))
    if model.ir_version > max_version:
        model.ir_version = max_version
        onnx.save(model, str(path))


@pytest.fixture
def toy_models(tmp_path):
    teacher_path = tmp_path / "teacher.onnx"
    student_path = tmp_path / "student.onnx"
    _run("make_toy_classifier.py", "-o", str(teacher_path),
         "--input-dim", "8", "--hidden-dim", "32", "--num-classes", "4", "--seed", "1")
    _run("make_toy_classifier.py", "-o", str(student_path),
         "--input-dim", "8", "--hidden-dim", "8", "--num-classes", "4", "--seed", "2")
    return teacher_path, student_path


@pytest.fixture
def synthetic_data(tmp_path):
    input_path = tmp_path / "train_input.bin"
    labels_path = tmp_path / "train_labels.bin"
    _run(
        "make_synthetic_classification_data.py",
        "--input-dim", "8", "--num-classes", "4", "--num-samples", "512",
        "--input-out", str(input_path), "--labels-out", str(labels_path),
    )
    return input_path, labels_path


def test_generate_artifacts_distillation_mode(tmp_path, toy_models):
    _teacher_path, student_path = toy_models
    artifacts_dir = tmp_path / "artifacts"
    _run(
        "generate_artifacts.py", str(student_path), "-o", str(artifacts_dir),
        "--loss", "distillation", "--distill-temperature", "2.0", "--distill-alpha", "0.5",
    )

    for name in ("checkpoint", "training_model.onnx", "eval_model.onnx", "optimizer_model.onnx"):
        assert (artifacts_dir / name).exists()

    model = onnx.load(str(artifacts_dir / "training_model.onnx"))
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    # The model's own input, plus the loss's two extra inputs, in the order
    # DistillationLoss.build() appends them -- see distillation_loss.py's
    # module docstring on why this exact order/naming is a documented
    # contract with onnx-finetune's --teacher-model mode.
    assert input_names[:3] == ["input", "teacher_logits", "labels"]
    # Combined loss first, then the soft/hard breakdown (additional_output_names).
    assert output_names[1:3] == ["kd_soft_loss", "kd_hard_loss"]


def test_generate_artifacts_rejects_distill_flags_without_distillation_loss(tmp_path, toy_models):
    _teacher_path, student_path = toy_models
    result = subprocess.run(
        [
            sys.executable, str(SCRIPTS / "generate_artifacts.py"), str(student_path),
            "-o", str(tmp_path / "artifacts"), "--loss", "mse", "--distill-alpha", "0.9",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "only apply with --loss distillation" in result.stderr


def test_distillation_training_step_reduces_loss(tmp_path, toy_models, synthetic_data):
    ort = pytest.importorskip("onnxruntime")
    api = pytest.importorskip("onnxruntime.training.api")

    teacher_path, student_path = toy_models
    input_path, labels_path = synthetic_data
    artifacts_dir = tmp_path / "artifacts"
    _run(
        "generate_artifacts.py", str(student_path), "-o", str(artifacts_dir),
        "--loss", "distillation", "--distill-temperature", "2.0", "--distill-alpha", "0.5",
    )
    _clamp_ir_version(artifacts_dir / "training_model.onnx")
    _clamp_ir_version(artifacts_dir / "optimizer_model.onnx")

    state = api.CheckpointState.load_checkpoint(str(artifacts_dir / "checkpoint"))
    module = api.Module(str(artifacts_dir / "training_model.onnx"), state)
    optimizer = api.Optimizer(str(artifacts_dir / "optimizer_model.onnx"), module)
    teacher_session = ort.InferenceSession(str(teacher_path))

    input_data = np.fromfile(input_path, dtype=np.float32).reshape(-1, 8)
    labels_data = np.fromfile(labels_path, dtype=np.int64)
    # A single fixed batch, trained repeatedly: the point is to prove the
    # gradients are correct (loss on this exact batch goes down), not to
    # demonstrate generalization -- same rationale as
    # tests/test_llm_distillation_hub_checkpoint.py's fixed-batch check.
    batch_input = input_data[:32]
    batch_labels = labels_data[:32]
    teacher_logits = teacher_session.run(None, {"input": batch_input})[0]

    module.train()
    losses = []
    for _ in range(50):
        loss, soft_loss, hard_loss = module(batch_input, teacher_logits, batch_labels)
        assert np.isfinite(loss) and np.isfinite(soft_loss) and np.isfinite(hard_loss)
        losses.append(float(loss))
        optimizer.step()
        module.lazy_reset_grad()

    assert losses[-1] < losses[0]
