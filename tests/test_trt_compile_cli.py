"""Dependency-light tests for the TensorRT compiler adapter."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "nvidia" / "trt_compile.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("trt_compile", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_does_not_import_tensorrt():
    completed = subprocess.run(
        ["python3", str(SCRIPT), "--help"], check=True, capture_output=True, text=True
    )
    assert "--workspace-mb" in completed.stdout


def test_compile_writes_engine_and_profile_manifest(monkeypatch, tmp_path):
    adapter = load_adapter()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"mock model")
    output = tmp_path / "nested" / "model.engine"
    manifest = tmp_path / "nested" / "manifest.json"

    fake_harness = types.SimpleNamespace(
        build_engine=lambda path, fp16, int8, workspace_mb: (
            b"serialized-engine",
            {
                "build_s": 0.12,
                "n_layers": 2,
                "n_int8_layers": 1,
                "engine_bytes": 17,
                "int8_effective": True,
                "layers": [{"name": "conv", "tactic": "mock"}],
            },
        )
    )
    monkeypatch.setitem(sys.modules, "trt_harness", fake_harness)
    monkeypatch.setitem(
        sys.modules, "tensorrt", types.SimpleNamespace(__version__="10.mock")
    )

    adapter.compile_tensorrt(model, output, manifest, "trt-sm87", True, True, 256)

    assert output.read_bytes() == b"serialized-engine"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["artifact"]["format"] == "tensorrt-engine"
    assert data["target"]["device"] == "trt-sm87"
    assert data["compiler"]["version"] == "10.mock"
    assert data["profiling"]["n_layers"] == 2
    assert data["profiling"]["layers"][0]["tactic"] == "mock"
    assert data["options"]["workspace_mb"] == 256


def test_large_layer_profile_is_bounded(monkeypatch, tmp_path):
    adapter = load_adapter()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"mock model")
    output = tmp_path / "model.engine"
    manifest = tmp_path / "manifest.json"
    layers = [{"name": f"layer-{i}", "detail": "x" * 800} for i in range(500)]
    monkeypatch.setitem(
        sys.modules,
        "trt_harness",
        types.SimpleNamespace(
            build_engine=lambda *args, **kwargs: (
                b"engine",
                {"n_layers": len(layers), "layers": layers},
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules, "tensorrt", types.SimpleNamespace(__version__="mock")
    )

    adapter.compile_tensorrt(model, output, manifest, "trt", False, False, 1)

    assert manifest.stat().st_size <= adapter.MAX_MANIFEST_BYTES
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["profiling"]["layers_total"] == 500
    assert data["profiling"]["layers_truncated"] is True
