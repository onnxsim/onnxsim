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


def refuses_relu(node, inputs):
    """A backend that cannot run Relu. Used with --backend below."""
    if node.op_type == "Relu":
        raise NotImplementedError("Relu is not supported")
    if node.op_type == "MatMul":
        return [inputs[0] @ inputs[1]]
    raise NotImplementedError(node.op_type)


def test_verify_with_a_backend_reports_partial_and_exits_zero(
    project, tmp_path, capsys
):
    cases = _cases_file(tmp_path)
    code = main(
        [
            "verify",
            project,
            "--cases",
            cases,
            "--backend",
            "test_versioning_cli:refuses_relu",
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_PASS
    assert "partial" in out
    assert "Relu:y" in out
    manifest, _ = read_project(project)
    assert manifest["steps"][-1]["verdict"] == "partial"


def test_backend_spec_without_a_function_is_a_usage_error(project, tmp_path):
    cases = _cases_file(tmp_path)
    assert (
        main(["verify", project, "--cases", cases, "--backend", "no_colon"])
        == EXIT_ERROR
    )


def test_build_with_simplify_records_its_configuration(project, tmp_path, capsys):
    out = str(tmp_path / "simplified.onnx")
    assert main(["build", project, "-o", out, "--simplify"]) == EXIT_PASS
    assert "simplify check: pass" in capsys.readouterr().out
    assert onnx.load(out).graph.node
    manifest, _ = read_project(project)
    (build,) = manifest["builds"]
    assert build["output"] == "simplified.onnx"
    assert build["simplify"] == {}
    assert build["simplify_checked"] is True
    assert build["onnxsim_version"]
    assert build["executor"]


def test_simplify_options_are_parsed_and_recorded(project, tmp_path):
    out = str(tmp_path / "opt.onnx")
    code = main(
        [
            "build",
            project,
            "-o",
            out,
            "--simplify-opt",
            "skip_fuse_bn=true",
            "--simplify-opt",
            "tensor_size_threshold=4KB",
        ]
    )
    assert code == EXIT_PASS
    manifest, _ = read_project(project)
    assert manifest["builds"][-1]["simplify"] == {
        "skip_fuse_bn": True,
        "tensor_size_threshold": "4KB",
    }


def test_an_unknown_simplify_option_fails_before_anything_is_written(
    project, tmp_path, capsys
):
    out = tmp_path / "never.onnx"
    assert (
        main(["build", project, "-o", str(out), "--simplify-opt", "no_such_option=1"])
        == EXIT_ERROR
    )
    assert "no_such_option" in capsys.readouterr().err
    assert not out.exists()
    manifest, _ = read_project(project)
    assert "builds" not in manifest


def test_a_simplify_option_without_a_value_is_a_usage_error(project, tmp_path):
    out = str(tmp_path / "bad.onnx")
    assert (
        main(["build", project, "-o", out, "--simplify-opt", "skip_fuse_bn"])
        == EXIT_ERROR
    )
