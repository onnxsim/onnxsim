"""Dependency-light tests for the Qualcomm compatibility harness driver."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "qualcomm" / "run_qnn_compat.py"
COMPILER = Path(__file__).parents[1] / "scripts" / "qualcomm" / "qnn_compile.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("qnn_compat_driver", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_model_path_is_forwarded(monkeypatch, tmp_path):
    driver = load_driver()
    model = tmp_path / "mobilenet-qdq.onnx"
    model.write_bytes(b"not parsed by the driver")
    seen = {}

    class Completed:
        stdout = '__RESULT__{"model":"mobilenet-qdq","status":"unsupported"}\n'
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = driver.run_one("mobilenet-qdq", 17, str(model))

    assert result["status"] == "unsupported"
    assert seen["command"][-1] == str(model)
    assert seen["kwargs"]["timeout"] == 17


def test_driver_help_does_not_import_onnx():
    # The driver intentionally imports the model suite lazily so --help and
    # path validation remain usable in a minimal environment.
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--model MODEL.onnx" in completed.stdout


def test_qnn_compiler_help_does_not_import_qnn():
    completed = subprocess.run(
        ["python3", str(COMPILER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--input INPUT" in completed.stdout
    assert "--target TARGET" in completed.stdout
