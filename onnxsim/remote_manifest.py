"""Manifest-driven preflight and legalization helpers for remote compilers.

The transport deliberately treats manifests as opaque.  This module provides
the optional policy layer used by applications that want to inspect the JSON
manifest before sending a fold group to a constrained runner.  It depends only
on the standard library and accepts any model-like object with
``model.graph.node[*].op_type`` attributes, so it is usable without ONNX in
small compiler-side tools and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass
class RemotePreflightReport:
    """Result of checking a model against a compiler manifest."""

    ok: bool
    unsupported_ops: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    target: str | None = None
    artifact_format: str | None = None


def load_manifest(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a manifest from JSON text, a file path, or an already parsed map."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        return json.loads(value.read_text(encoding="utf-8"))
    text = str(value)
    candidate = Path(text)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("remote compiler manifest must be a JSON object")
    return parsed


def _model_ops(model: Any) -> list[str]:
    graph = getattr(model, "graph", None)
    nodes = getattr(graph, "node", ()) if graph is not None else ()
    return [str(getattr(node, "op_type", "")) for node in nodes]


def preflight_model(
    model: Any,
    manifest: str | Path | Mapping[str, Any],
    *,
    target: str | None = None,
    artifact_format: str | None = None,
    compiler_id: str | None = None,
) -> RemotePreflightReport:
    """Check target/compiler/artifact identity and advertised operator support.

    Empty capability lists are treated as unknown (rather than "supports no
    operators") because the reference QNN adapter intentionally leaves them
    empty until its backend has compiled the graph.
    """
    data = load_manifest(manifest)
    errors: list[str] = []
    target_data = data.get("target") if isinstance(data.get("target"), Mapping) else {}
    artifact_data = data.get("artifact") if isinstance(data.get("artifact"), Mapping) else {}
    compiler_data = data.get("compiler") if isinstance(data.get("compiler"), Mapping) else {}
    actual_target = target_data.get("device") or target_data.get("name")
    actual_format = artifact_data.get("format")
    if target and actual_target and target != actual_target:
        errors.append(f"target mismatch: required {target!r}, manifest has {actual_target!r}")
    if artifact_format and actual_format and artifact_format != actual_format:
        errors.append(
            f"artifact format mismatch: required {artifact_format!r}, manifest has {actual_format!r}"
        )
    actual_compiler = compiler_data.get("id")
    if compiler_id and actual_compiler and compiler_id != actual_compiler:
        errors.append(
            f"compiler mismatch: required {compiler_id!r}, manifest has {actual_compiler!r}"
        )

    capabilities = data.get("capabilities")
    supported = capabilities.get("ops") if isinstance(capabilities, Mapping) else None
    supported_set = {str(op) for op in supported} if supported else set()
    unsupported = sorted({op for op in _model_ops(model) if supported_set and op not in supported_set})
    if unsupported:
        errors.append("unsupported operators: " + ", ".join(unsupported))
    return RemotePreflightReport(
        ok=not errors,
        unsupported_ops=unsupported,
        errors=errors,
        target=str(actual_target) if actual_target is not None else None,
        artifact_format=str(actual_format) if actual_format is not None else None,
    )


def legalize_for_manifest(
    model: Any,
    manifest: str | Path | Mapping[str, Any],
    passes: Sequence[tuple[str, Callable[[Any], Any]]] = (),
    *,
    target: str | None = None,
    max_rounds: int = 8,
) -> RemotePreflightReport:
    """Apply target-specific passes to a model, then run manifest preflight.

    Each pass mutates the model and returns a truthy change count/flag. Passes
    are revisited until a fixed point or ``max_rounds``. The report records a
    clear error if a pass keeps changing the graph beyond the bound.
    """
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    for _round in range(max_rounds):
        round_changed = False
        for _name, apply_pass in passes:
            result = apply_pass(model)
            round_changed = round_changed or bool(result)
        if not round_changed:
            return preflight_model(model, manifest, target=target)
    report = preflight_model(model, manifest, target=target)
    report.ok = False
    report.errors.append(f"legalization did not converge within {max_rounds} rounds")
    return report
