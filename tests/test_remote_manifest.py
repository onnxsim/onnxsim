"""Dependency-light tests for remote compiler preflight and trace summaries."""

from types import SimpleNamespace

import importlib.util
from pathlib import Path
import sys


def load_module(name):
    path = Path(__file__).parents[1] / "onnxsim" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile_merge = load_module("profile_merge")
remote_manifest = load_module("remote_manifest")
legalize_for_manifest = remote_manifest.legalize_for_manifest
preflight_model = remote_manifest.preflight_model


def model_with_ops(*ops):
    return SimpleNamespace(graph=SimpleNamespace(node=[SimpleNamespace(op_type=op) for op in ops]))


def test_preflight_checks_identity_and_supported_ops():
    manifest = {
        "compiler": {"id": "trt-10"},
        "target": {"device": "sm87"},
        "artifact": {"format": "tensorrt-engine"},
        "capabilities": {"ops": ["Conv", "Relu"]},
    }
    report = preflight_model(
        model_with_ops("Conv", "MatMul"), manifest,
        target="sm87", artifact_format="tensorrt-engine", compiler_id="trt-10"
    )
    assert not report.ok
    assert report.unsupported_ops == ["MatMul"]


def test_legalization_reaches_manifest_supported_form():
    model = model_with_ops("Unsupported")
    manifest = {"capabilities": {"ops": ["Relu"]}}
    report = legalize_for_manifest(
        model, manifest,
        [("rewrite", lambda m: (
            setattr(m.graph.node[0], "op_type", "Relu") or 1
            if m.graph.node[0].op_type != "Relu" else 0
        )),
         ("stable", lambda _m: 0)],
    )
    assert report.ok


def test_empty_capabilities_are_unknown():
    report = preflight_model(model_with_ops("Anything"), {"capabilities": {"ops": []}})
    assert report.ok


def test_remote_profile_summary_groups_phases():
    trace = {"traceEvents": [
        {"name": "RemoteRPC/compile", "ph": "X", "dur": 4},
        {"name": "worker", "ph": "X", "dur": 9, "args": {"phase": "execute"}},
        {"name": "RemoteArtifactReady", "ph": "X", "dur": 0},
    ]}
    summary = profile_merge.summarize_remote_events(trace)
    assert summary["compile"] == {"count": 1, "duration_us": 4}
    assert summary["execute"] == {"count": 1, "duration_us": 9}
    assert summary["other"] == {"count": 1, "duration_us": 0}
