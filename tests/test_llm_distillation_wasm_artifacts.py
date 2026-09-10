"""Regression test for examples/llm_distillation/wasm_demo/generate_web_artifacts.py.

Loaded by file path rather than import, same reason as
tests/test_llm_distillation_demo.py: examples/ sits outside onnxsim's own
package and outside pytest's tests/ sys.path entry.

Only covers the Python-side artifact generation (export + custom
onnxblock loss + onnxruntime.training.artifacts.generate_artifacts): the
browser-side training loop in wasm_demo/index.html needs a real
ort.TrainingSession running in a JS engine, which pytest can't exercise --
that was validated by hand (see wasm_demo/README.md's "Validation notes")
via headless Chromium.
"""

import importlib.util
import os

import onnx
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime.training.artifacts")

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "llm_distillation",
    "wasm_demo",
    "generate_web_artifacts.py",
)
_spec = importlib.util.spec_from_file_location("generate_web_artifacts", _SCRIPT_PATH)
generate_web_artifacts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_web_artifacts)


def test_generated_artifacts_load_and_have_clamped_ir_version(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_web_artifacts, "ASSETS_DIR", str(tmp_path))

    generate_web_artifacts.main()

    expected_files = {
        "teacher_model.onnx",
        "checkpoint",
        "training_model.onnx",
        "optimizer_model.onnx",
    }
    assert expected_files <= set(os.listdir(tmp_path))
    assert not (tmp_path / "eval_model.onnx").exists()
    assert not (tmp_path / "student_export.onnx").exists()

    for name in ("teacher_model.onnx", "training_model.onnx", "optimizer_model.onnx"):
        model = onnx.load(str(tmp_path / name))
        assert model.ir_version <= generate_web_artifacts.MAX_SUPPORTED_IR_VERSION


def test_training_session_runs_a_real_step(tmp_path, monkeypatch):
    ort = pytest.importorskip(
        "onnxruntime.training.api",
        reason="needs onnxruntime-training's Python training API",
    )
    import numpy as np

    monkeypatch.setattr(generate_web_artifacts, "ASSETS_DIR", str(tmp_path))
    generate_web_artifacts.main()

    state = ort.CheckpointState.load_checkpoint(str(tmp_path / "checkpoint"))
    module = ort.Module(str(tmp_path / "training_model.onnx"), state)
    optimizer = ort.Optimizer(str(tmp_path / "optimizer_model.onnx"), module)

    vocab, seq = generate_web_artifacts.VOCAB_SIZE, 8
    input_ids = np.random.randint(0, vocab, (1, seq)).astype(np.int64)
    attention_mask = np.ones((1, seq), dtype=np.int64)
    teacher_logits = np.random.randn(1, seq, vocab).astype(np.float32)
    labels = np.random.randint(0, vocab, (1, seq)).astype(np.int64)

    module.train()
    loss = module(input_ids, attention_mask, teacher_logits, labels)
    optimizer.step()
    module.lazy_reset_grad()

    assert np.isfinite(loss)
