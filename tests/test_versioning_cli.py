"""End-to-end tests for the versioning CLI: init, status, build and verify."""

import json

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim.versioning import graph_hash, graph_text
from onnxsim.versioning_cli import EXIT_ERROR, EXIT_FAIL, EXIT_PASS, main
from onnxsim.versioning_project import load_cases, read_project

W = np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0


def _original(path, relu="Relu"):
    model = parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["" : 21]>
        g (float[1,4] x) => (float[1,4] y)
        {{
          h = MatMul (x, W)
          y = {relu} (h)
        }}
        """
    )
    model.graph.initializer.append(numpy_helper.from_array(W, name="W"))
    onnx.save(model, str(path))
    return str(path)


def _cases_file(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "smoke",
                        "kind": "generated",
                        "seed": 0,
                        "specs": [
                            {
                                "name": "x",
                                "dtype": "float32",
                                "shape": [1, 4],
                                "low": 0.0,
                                "high": 1.0,
                            }
                        ],
                    }
                ]
            }
        )
    )
    return str(path)


@pytest.fixture
def project(tmp_path):
    original = _original(tmp_path / "model.onnx")
    directory = str(tmp_path / "proj")
    assert main(["init", original, directory]) == EXIT_PASS
    return directory


def test_init_writes_graph_text_and_base_manifest(tmp_path, capsys):
    original = _original(tmp_path / "model.onnx")
    directory = str(tmp_path / "proj")
    assert main(["init", original, directory]) == EXIT_PASS
    manifest, text = read_project(directory)
    assert manifest["base"]["graph"] == graph_hash(text)
    assert "W = {" not in text
    assert "base graph" in capsys.readouterr().out


def test_init_refuses_an_existing_project(tmp_path, project):
    original = str(tmp_path / "model.onnx")
    assert main(["init", original, project]) == EXIT_ERROR


def test_status_reports_graph_and_steps(project, capsys):
    assert main(["status", project]) == EXIT_PASS
    out = capsys.readouterr().out
    assert "base graph" in out and "steps:      0" in out


def test_build_writes_a_loadable_onnx_matching_the_text(tmp_path, project):
    out = str(tmp_path / "built.onnx")
    assert main(["build", project, "-o", out]) == EXIT_PASS
    built = onnx.load(out)
    _, text = read_project(project)
    assert graph_text(built) == text


def test_verify_passes_and_records_an_unchanged_graph(project, tmp_path):
    cases = _cases_file(tmp_path)
    assert main(["verify", project, "--cases", cases, "--label", "noop"]) == EXIT_PASS
    manifest, _ = read_project(project)
    assert [s["verdict"] for s in manifest["steps"]] == ["pass"]
    assert manifest["steps"][0]["label"] == "noop"


def test_verify_fails_on_an_edit_and_bisects_to_the_fused_block(
    project, tmp_path, capsys
):
    text_path = f"{project}/model.txt"
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text.replace("Relu (h)", "Sigmoid (h)"))
    cases = _cases_file(tmp_path)
    assert main(["verify", project, "--cases", cases, "--label", "bad"]) == EXIT_FAIL
    out = capsys.readouterr().out
    assert "verdict: fail" in out
    assert "MatMul+Relu" in out
    manifest, _ = read_project(project)
    assert manifest["steps"][-1]["verdict"] == "fail"
    assert manifest["steps"][-1]["culprit"]["blocks"] == ["MatMul+Relu"]


def test_verify_without_record_leaves_the_manifest_alone(project, tmp_path):
    cases = _cases_file(tmp_path)
    assert main(["verify", project, "--cases", cases, "--no-record"]) == EXIT_PASS
    manifest, _ = read_project(project)
    assert manifest["steps"] == []


def test_verify_detects_an_edited_original(tmp_path, project):
    original = tmp_path / "model.onnx"
    _original(original, relu="Sigmoid")
    cases = _cases_file(tmp_path)
    assert main(["verify", project, "--cases", cases]) == EXIT_ERROR


def test_missing_cases_file_is_a_usage_error(project, tmp_path):
    assert (
        main(["verify", project, "--cases", str(tmp_path / "nope.json")]) == EXIT_ERROR
    )


def test_load_cases_resolves_supplied_paths_next_to_the_file(tmp_path):
    x = np.full((1, 4), 0.5, np.float32)
    np.savez(tmp_path / "in.npz", x=x)
    from onnxsim.versioning import tensor_digest

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "real",
                        "kind": "supplied",
                        "path": "in.npz",
                        "digests": {"x": tensor_digest(x)},
                    }
                ]
            }
        )
    )
    (case,) = load_cases(str(cases_path))
    assert case.path == str(tmp_path / "in.npz")
