"""Project directory operations for the versioning workflow.

A project directory holds the graph text and a manifest. It refers to the original
``.onnx`` by path and digest, and never copies weights into itself::

    <dir>/model.txt       canonical graph text, initializers as references
    <dir>/manifest.json   base file and digest, plus one entry per recorded step

Weights are resolved from the original file, plus any extra ``.onnx`` files the
caller passes, by the digests in the text. Test cases come from a JSON file whose
supplied-input paths are relative to that file.
"""

import inspect
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import onnx
from onnx import ModelProto

from onnxsim.versioning import (
    MANIFEST_VERSION,
    Backend,
    Case,
    CaseReport,
    ChangeUnit,
    GeneratedCase,
    OnnxTestDataCase,
    ReproducibilityError,
    SuppliedCase,
    TensorIndex,
    TensorSpec,
    bisect_failure,
    check_equivalent,
    executor_name,
    file_digest,
    graph_hash,
    graph_text,
    load_graph_text,
    load_manifest,
    record_step,
    save_snapshot,
)
from onnxsim.versioning_manifest import validate_manifest

MODEL_FILE = "model.txt"
MANIFEST_FILE = "manifest.json"


def _model_path(directory: str) -> str:
    return os.path.join(directory, MODEL_FILE)


def _manifest_path(directory: str) -> str:
    return os.path.join(directory, MANIFEST_FILE)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def init_project(original: str, directory: str) -> dict:
    """Create a project in ``directory`` from the original ``.onnx`` file."""
    if os.path.exists(_model_path(directory)) or os.path.exists(
        _manifest_path(directory)
    ):
        raise FileExistsError(f"{directory} already holds a project")
    os.makedirs(directory, exist_ok=True)
    text = graph_text(onnx.load(original, load_external_data=True))
    with open(_model_path(directory), "w", encoding="utf-8") as f:
        f.write(text)
    manifest = {
        "version": MANIFEST_VERSION,
        "base": {
            "file": os.path.relpath(original, directory),
            "file_digest": file_digest(original),
            "graph": graph_hash(text),
        },
        "steps": [],
    }
    validate_manifest(manifest)
    _write_json(_manifest_path(directory), manifest)
    return manifest


def read_project(directory: str) -> Tuple[dict, str]:
    """The manifest and graph text of an initialized project."""
    manifest = load_manifest(_manifest_path(directory))
    if "base" not in manifest:
        raise ValueError(f"{directory} is not an initialized project (run init first)")
    with open(_model_path(directory), encoding="utf-8") as f:
        return manifest, f.read()


def _load_base(directory: str, manifest: dict) -> ModelProto:
    original = os.path.normpath(os.path.join(directory, manifest["base"]["file"]))
    if file_digest(original) != manifest["base"]["file_digest"]:
        raise ReproducibilityError(
            f"{original} no longer matches the digest recorded at init"
        )
    return onnx.load(original, load_external_data=True)


def build_project(
    directory: str, weights: Sequence[str] = ()
) -> Tuple[ModelProto, ModelProto, str]:
    """Return ``(base, candidate, graph text)`` for a project.

    The candidate is the graph text rebuilt with weights from the original file
    and from any extra ``weights`` files.
    """
    manifest, text = read_project(directory)
    base = _load_base(directory, manifest)
    index = TensorIndex()
    index.add_model(base)
    for path in weights:
        index.add_file(path)
    return base, load_graph_text(text, index), text


def project_status(directory: str) -> dict:
    manifest, text = read_project(directory)
    return {
        "base": manifest["base"],
        "graph": graph_hash(text),
        "steps": [
            {
                "label": s["label"],
                "verdict": s["verdict"],
                "output_graph": s["output_graph"],
            }
            for s in manifest["steps"]
        ],
    }


def case_from_dict(data: dict, base_dir: str = ".") -> Case:
    """A test case from its JSON form. Supplied paths resolve against ``base_dir``."""
    kind = data.get("kind")
    if kind == "generated":
        specs = tuple(
            TensorSpec(
                name=s["name"],
                dtype=s["dtype"],
                shape=tuple(s["shape"]),
                low=float(s.get("low", 0.0)),
                high=float(s.get("high", 1.0)),
            )
            for s in data["specs"]
        )
        return GeneratedCase(
            data["name"], int(data["seed"]), specs, data.get("digests") or None
        )
    if kind == "supplied":
        return SuppliedCase(
            data["name"], os.path.join(base_dir, data["path"]), data["digests"]
        )
    if kind == "onnx_test_data":
        return OnnxTestDataCase(
            data["name"],
            os.path.join(base_dir, data["directory"]),
            int(data.get("test_set", 0)),
        )
    raise ValueError(f"case {data.get('name', '?')}: unknown kind {kind!r}")


def load_cases(path: str) -> List[Case]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(path))
    return [case_from_dict(c, base_dir) for c in data["cases"]]


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    reports: Tuple[CaseReport, ...]
    culprit: Optional[ChangeUnit]
    entry: Optional[dict]


def verify_project(
    directory: str,
    cases: Sequence[Case],
    *,
    weights: Sequence[str] = (),
    label: str = "",
    command: str = "",
    atol: float = 1e-5,
    rtol: float = 1e-4,
    fusion: str = "default",
    record: bool = True,
    backend: Optional[Backend] = None,
) -> VerifyResult:
    """Check the project's graph against its base on ``cases``.

    On failure the failing change is bisected, and the culprit is stored with the
    step. With ``record`` the step is appended to the manifest.
    """
    manifest, text = read_project(directory)
    base, candidate, _ = build_project(directory, weights)
    reports = check_equivalent(base, candidate, cases, atol, rtol, backend)
    passed = all(r.ok for r in reports)

    culprit: Optional[ChangeUnit] = None
    culprit_data: Optional[Dict] = None
    if not passed:
        found = bisect_failure(
            base, candidate, cases, atol, rtol, fusion=fusion, backend=backend
        )
        if found is not None:
            culprit = found.culprit
            culprit_data = {
                "fusion": fusion,
                "nodes": list(culprit.nodes),
                "blocks": list(culprit.blocks),
                "initializers": list(culprit.initializers),
                "evaluations": found.evaluations,
            }

    entry = None
    if record:
        previous = (
            manifest["steps"][-1]["output_graph"]
            if manifest["steps"]
            else manifest["base"]["graph"]
        )
        entry = record_step(
            _manifest_path(directory),
            label=label,
            command=command,
            base_graph=previous,
            output_graph=graph_hash(text),
            cases=cases,
            reports=reports,
            executor=executor_name(),
            culprit=culprit_data,
        )
    return VerifyResult(passed, tuple(reports), culprit, entry)


def simplify_options_check(options: dict) -> None:
    """Reject option names that ``onnxsim.simplify`` does not take, before any work."""
    from onnxsim import simplify

    accepted = set(inspect.signature(simplify).parameters) - {"model"}
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise ValueError(f"unknown simplify option(s): {', '.join(unknown)}")


def build_output(
    directory: str,
    output: str,
    weights: Sequence[str] = (),
    simplify_options: Optional[dict] = None,
) -> dict:
    """Write the project's graph to ``output``, optionally simplified, and record how.

    With ``simplify_options`` (possibly empty), ``onnxsim.simplify`` runs on the
    rebuilt graph with those keyword options. The build's entry in the manifest
    records the options, the simplify version, the executor and the output's digest.
    Returns the entry; its ``simplify_checked`` is ``False`` when simplify's own
    output check failed.
    """
    import onnxsim
    from onnxsim import simplify

    if simplify_options is not None:
        simplify_options_check(simplify_options)
    _, candidate, text = build_project(directory, weights)
    checked: Optional[bool] = None
    if simplify_options is not None:
        candidate, checked = simplify(candidate, **simplify_options)
        checked = bool(checked)
    digest = save_snapshot(candidate, output)
    entry = {
        "output": os.path.basename(output),
        "file_digest": digest,
        "source_graph": graph_hash(text),
        "simplify": simplify_options,
        "simplify_checked": checked,
        "onnxsim_version": onnxsim.__version__,
        "executor": executor_name(),
    }
    path = _manifest_path(directory)
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("builds", []).append(entry)
    validate_manifest(manifest)
    _write_json(path, manifest)
    return entry
