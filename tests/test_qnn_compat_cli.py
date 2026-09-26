"""Dependency-light tests for the Qualcomm compatibility harness driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
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


def test_qnn_compiler_configures_embedded_context(monkeypatch, tmp_path):
    compiler = importlib.util.spec_from_file_location("qnn_compile", COMPILER)
    assert compiler and compiler.loader
    module = importlib.util.module_from_spec(compiler)
    compiler.loader.exec_module(module)

    backend = tmp_path / "libQnnHtp.so"
    backend.write_bytes(b"mock backend")
    input_model = tmp_path / "model.onnx"
    input_model.write_bytes(b"mock model")
    output_model = tmp_path / "nested" / "model_ctx.onnx"
    manifest_path = tmp_path / "nested" / "manifest.json"
    captured = {}

    class SessionOptions:
        def __init__(self):
            self.entries = {}
            self.log_severity_level = None

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    def session(path, sess_options, providers, provider_options):
        captured["path"] = path
        captured["options"] = sess_options
        captured["providers"] = providers
        captured["provider_options"] = provider_options
        output_model.parent.mkdir(parents=True, exist_ok=True)
        output_model.write_bytes(b"mock embedded context")

    fake_ort = types.SimpleNamespace(
        __version__="mock-ort",
        SessionOptions=SessionOptions,
        InferenceSession=session,
        register_execution_provider_library=lambda name, path: captured.setdefault(
            "registered", (name, path)
        ),
    )
    fake_qnn = types.SimpleNamespace(
        get_qnn_htp_path=lambda: str(backend),
        get_library_path=lambda: "libonnxruntime_providers_qnn.so",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", fake_qnn)

    module.compile_qnn(input_model, output_model, manifest_path, "qnn-htp", True)

    assert captured["path"] == str(input_model)
    assert captured["providers"] == ["QNNExecutionProvider"]
    assert captured["provider_options"][0]["backend_path"] == str(backend)
    assert captured["options"].entries["ep.context_enable"] == "1"
    assert captured["options"].entries["ep.context_embed_mode"] == "1"
    assert captured["options"].entries["ep.context_file_path"] == str(output_model)
    assert captured["options"].entries["session.disable_cpu_ep_fallback"] == "1"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact"]["format"] == "onnxruntime-ep-context"
    assert manifest["target"]["device"] == "qnn-htp"
