"""Tests for the versioning manifest: version 1 validation and the JSON Schema."""

import copy
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim.versioning import (
    CaseReport,
    GeneratedCase,
    TensorSpec,
    load_manifest,
    record_step,
)
from onnxsim.versioning_manifest import validate_manifest
from onnxsim.versioning_project import build_output, init_project, verify_project

SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "schemas"
    / "versioning-manifest.v1.schema.json"
)
W = np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0


@pytest.fixture
def manifest_path(tmp_path):
    """A project with an init entry, a passing verify, a failing verify and a build."""
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["" : 21]>
        g (float[1,4] x) => (float[1,4] y)
        {
          h = MatMul (x, W)
          y = Relu (h)
        }
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(W, name="W"))
    original = tmp_path / "model.onnx"
    onnx.save(model, str(original))
    project = tmp_path / "proj"
    init_project(str(original), str(project))
    cases = [
        GeneratedCase(
            "smoke", seed=0, specs=(TensorSpec("x", "float32", (1, 4), 0.0, 1.0),)
        )
    ]
    verify_project(str(project), cases, label="noop")
    text_path = project / "model.txt"
    text_path.write_text(text_path.read_text().replace("Relu (h)", "Sigmoid (h)"))
    verify_project(str(project), cases, label="bad")
    build_output(str(project), str(tmp_path / "out.onnx"), simplify_options={})
    return project / "manifest.json"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_a_manifest_written_by_the_code_validates(manifest_path):
    validate_manifest(_load(manifest_path))
    manifest = load_manifest(str(manifest_path))
    assert [s["verdict"] for s in manifest["steps"]] == ["pass", "fail"]
    assert manifest["steps"][1]["culprit"]["blocks"] == ["MatMul+Relu"]
    assert manifest["builds"][0]["simplify"] == {}


def test_a_version_other_than_one_is_rejected(manifest_path):
    data = _load(manifest_path)
    data["version"] = 2
    with pytest.raises(ValueError, match="unsupported manifest version 2"):
        validate_manifest(data)


def test_an_unknown_top_level_key_is_rejected(manifest_path):
    data = _load(manifest_path)
    data["extra"] = True
    with pytest.raises(ValueError, match="unexpected extra"):
        validate_manifest(data)


def test_a_bad_verdict_names_its_location(manifest_path):
    data = _load(manifest_path)
    data["steps"][0]["verdict"] = "maybe"
    with pytest.raises(ValueError, match=r"steps\[0\]\.verdict: must be one of"):
        validate_manifest(data)


def test_a_malformed_digest_is_rejected(manifest_path):
    data = _load(manifest_path)
    data["steps"][0]["output_graph"] = "sha256:abc"
    with pytest.raises(ValueError, match=r"steps\[0\]\.output_graph"):
        validate_manifest(data)


def test_a_report_missing_its_status_is_rejected(manifest_path):
    data = _load(manifest_path)
    del data["steps"][0]["reports"][0]["status"]
    with pytest.raises(ValueError, match="missing status"):
        validate_manifest(data)


def test_a_bad_case_kind_is_rejected(manifest_path):
    data = _load(manifest_path)
    data["steps"][0]["cases"][0]["kind"] = "random"
    with pytest.raises(ValueError, match="must be one of"):
        validate_manifest(data)


def test_an_invalid_manifest_is_refused_on_load(manifest_path):
    data = _load(manifest_path)
    data["steps"][0]["verdict"] = "maybe"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=str(manifest_path.name)):
        load_manifest(str(manifest_path))


def test_a_non_finite_difference_is_stored_as_null(tmp_path):
    path = str(tmp_path / "manifest.json")
    report = CaseReport("c", float("inf"), float("nan"), False, status="fail")
    cases = [
        GeneratedCase("c", seed=0, specs=(TensorSpec("x", "float32", (1,), 0.0, 1.0),))
    ]
    record_step(
        path,
        label="shape",
        command="test",
        base_graph="sha256:" + "0" * 64,
        output_graph="sha256:" + "1" * 64,
        cases=cases,
        reports=[report],
        executor="test",
    )
    raw = Path(path).read_text()
    assert "Infinity" not in raw and "NaN" not in raw
    stored = json.loads(raw)["steps"][0]["reports"][0]
    assert stored["max_abs_diff"] is None and stored["max_rel_diff"] is None
    validate_manifest(json.loads(raw))


def test_the_json_schema_agrees_with_the_validator(manifest_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    good = _load(manifest_path)
    validator.validate(good)

    bad_cases = {
        "version": lambda m: m.__setitem__("version", 2),
        "verdict": lambda m: m["steps"][0].__setitem__("verdict", "maybe"),
        "digest": lambda m: m["steps"][0].__setitem__("output_graph", "sha256:abc"),
        "extra key": lambda m: m.__setitem__("extra", 1),
        "report status": lambda m: m["steps"][0]["reports"][0].pop("status"),
        "case kind": lambda m: m["steps"][0]["cases"][0].__setitem__("kind", "random"),
    }
    for name, break_it in bad_cases.items():
        data = copy.deepcopy(good)
        break_it(data)
        with pytest.raises(ValueError):
            validate_manifest(data)
        assert not validator.is_valid(data), (
            f"schema accepted a manifest with a bad {name}"
        )


def test_the_other_case_kinds_pass_both_validators(manifest_path):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        json.loads(SCHEMA.read_text(encoding="utf-8"))
    )
    data = _load(manifest_path)
    digest = "sha256:" + "a" * 64
    data["steps"][0]["cases"] += [
        {
            "name": "recorded",
            "kind": "supplied",
            "path": "in.npz",
            "digests": {"x": digest},
        },
        {
            "name": "official",
            "kind": "onnx_test_data",
            "directory": "case",
            "test_set": 0,
        },
    ]
    validate_manifest(data)
    validator.validate(data)
