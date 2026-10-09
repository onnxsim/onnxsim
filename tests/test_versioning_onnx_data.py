"""Tests for ONNX backend test data (``test_data_set_N/input_i.pb``) as versioning cases."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

from onnxsim.versioning import (
    OnnxTestDataCase,
    check_equivalent,
    load_onnx_test_set,
    resolve_feeds,
    run_model,
)
from onnxsim.versioning_project import case_from_dict

ONNX_DATA = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "onnx"
    / "onnx"
    / "backend"
    / "test"
    / "data"
)


def _write_case(root: Path) -> Path:
    """A two-output model with one input, laid out like an ONNX backend test case."""
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["" : 21]>
        g (float[2] x) => (float[2] y, float[2] z)
        {
          y = Relu (x)
          z = Add (x, x)
        }
        """
    )
    case = root / "case"
    data = case / "test_data_set_0"
    data.mkdir(parents=True)
    onnx.save(model, str(case / "model.onnx"))
    x = np.array([-1.0, 2.0], np.float32)
    (data / "input_0.pb").write_bytes(numpy_helper.from_array(x).SerializeToString())
    (data / "output_0.pb").write_bytes(
        numpy_helper.from_array(np.maximum(x, 0)).SerializeToString()
    )
    (data / "output_1.pb").write_bytes(
        numpy_helper.from_array(x + x).SerializeToString()
    )
    return case


def test_inputs_are_matched_to_graph_inputs_and_read_as_arrays(tmp_path):
    case = _write_case(tmp_path)
    model = onnx.load(str(case / "model.onnx"))
    feeds, expected = load_onnx_test_set(str(case), model)
    assert list(feeds) == ["x"]
    np.testing.assert_array_equal(feeds["x"], [-1.0, 2.0])
    assert list(expected) == ["y", "z"]
    np.testing.assert_array_equal(expected["z"], [-2.0, 4.0])


def test_a_file_count_that_does_not_match_the_graph_is_an_error(tmp_path):
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["" : 21]>
        g (float[2] a, float[2] b) => (float[2] y)
        {
          y = Add (a, b)
        }
        """
    )
    case = tmp_path / "case"
    data = case / "test_data_set_0"
    data.mkdir(parents=True)
    (data / "input_0.pb").write_bytes(
        numpy_helper.from_array(np.zeros(2, np.float32)).SerializeToString()
    )
    with pytest.raises(ValueError, match="1 input_\\*.pb files for 2 graph inputs"):
        load_onnx_test_set(str(case), model)


def test_an_onnx_test_case_passes_an_identical_candidate(tmp_path):
    case = _write_case(tmp_path)
    model = onnx.load(str(case / "model.onnx"))
    (report,) = check_equivalent(model, model, [OnnxTestDataCase("c", str(case))])
    assert report.status == "pass" and report.ok


def test_a_test_case_without_a_model_cannot_resolve_its_inputs(tmp_path):
    case = _write_case(tmp_path)
    with pytest.raises(ValueError, match="needs the model"):
        resolve_feeds(OnnxTestDataCase("c", str(case)))


def test_onnx_test_case_round_trips_through_json(tmp_path):
    from onnxsim.versioning import case_to_dict

    case = OnnxTestDataCase("c", str(tmp_path / "case"), test_set=1)
    data = case_to_dict(case)
    assert data == {
        "name": "c",
        "kind": "onnx_test_data",
        "directory": "case",
        "test_set": 1,
    }
    assert case_from_dict(data, str(tmp_path)) == case


def _real_cases():
    """Up to five backend cases from the ONNX submodule, across its test-data groups."""
    found = []
    for group in ("pytorch-operator", "pytorch-converted", "simple", "real"):
        root = ONNX_DATA / group
        if not root.is_dir():
            continue
        found += sorted(
            d
            for d in root.iterdir()
            if (d / "model.onnx").is_file() and (d / "test_data_set_0").is_dir()
        )
    return found[:5]


@pytest.mark.skipif(
    not _real_cases(), reason="ONNX backend test data submodule not checked out"
)
@pytest.mark.parametrize("case_dir", _real_cases(), ids=lambda d: d.name)
def test_onnx_backend_case_matches_its_official_outputs(case_dir):
    """Official outputs come from ONNX's own reference evaluator, which generated them."""
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(str(case_dir / "model.onnx"))
    feeds, expected = load_onnx_test_set(str(case_dir), model)
    if expected is None:
        pytest.skip("case has no expected outputs")
    got = ReferenceEvaluator(model).run(None, feeds)
    for name, actual in zip([o.name for o in model.graph.output], got):
        np.testing.assert_allclose(actual, expected[name], rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(
    not _real_cases(), reason="ONNX backend test data submodule not checked out"
)
@pytest.mark.parametrize("case_dir", _real_cases(), ids=lambda d: d.name)
def test_onnx_backend_case_passes_an_identical_candidate_on_onnxruntime(case_dir):
    model = onnx.load(str(case_dir / "model.onnx"))
    try:
        run_model(model, load_onnx_test_set(str(case_dir), model)[0])
    except (
        Exception
    ) as e:  # ONNX Runtime rejects some old opsets; that is not ours to fix
        pytest.skip(f"onnxruntime cannot run this model: {type(e).__name__}")
    (report,) = check_equivalent(
        model, model, [OnnxTestDataCase("official", str(case_dir))]
    )
    assert report.ok
