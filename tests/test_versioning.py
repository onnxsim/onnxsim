"""Tests for onnxsim.versioning: hash-referenced graph text, reproducible test
cases, equivalence checks, and the manifest."""

import json

import numpy as np
import pytest
from onnx import numpy_helper, parser

from onnxsim.versioning import (
    GeneratedCase,
    ReproducibilityError,
    SuppliedCase,
    TensorIndex,
    TensorSpec,
    case_set_id,
    check_equivalent,
    file_digest,
    generate_feeds,
    graph_hash,
    graph_text,
    load_graph_text,
    load_manifest,
    record_step,
    resolve_feeds,
    save_snapshot,
    tensor_digest,
)


def _model(body, opset=21):
    return parser.parse_model(f'<ir_version: 10, opset_import: ["" : {opset}]> {body}')


def _with_weight(model, name, arr):
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))
    return model


W = np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0


def _matmul_model(weight=W):
    model = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          h = MatMul (x, W)
          y = Relu (h)
        }
        """
    )
    return _with_weight(model, "W", weight)


def _index(*models):
    index = TensorIndex()
    for m in models:
        index.add_model(m)
    return index


def test_tensor_digest_depends_on_dtype_shape_and_bytes():
    a = np.zeros((2, 3), np.float32)
    assert tensor_digest(a) == tensor_digest(a.copy())
    assert tensor_digest(a) != tensor_digest(a.astype(np.float64))
    assert tensor_digest(a) != tensor_digest(a.reshape(3, 2))
    assert tensor_digest(a) != tensor_digest(a + 1)


def test_graph_text_has_no_weight_values_and_parses_back_to_the_same_graph():
    model = _matmul_model()
    text = graph_text(model)
    assert "W = {" not in text
    assert "onnxsim-init" in text
    assert str(tensor_digest(W)) in text

    rebuilt = load_graph_text(text, _index(model))
    assert rebuilt.graph.node == model.graph.node
    assert rebuilt.graph.initializer[0].name == "W"
    np.testing.assert_array_equal(
        numpy_helper.to_array(rebuilt.graph.initializer[0]), W
    )


def test_graph_text_is_a_fixed_point_of_print_parse_print():
    text = graph_text(_matmul_model())
    assert graph_text(load_graph_text(text, _index(_matmul_model()))) == text


def test_graph_hash_changes_when_weight_changes():
    h1 = graph_hash(graph_text(_matmul_model()))
    h2 = graph_hash(graph_text(_matmul_model(W + 1.0)))
    assert h1 != h2
    assert h1 == graph_hash(graph_text(_matmul_model()))


def test_load_fails_when_digest_is_not_indexed():
    text = graph_text(_matmul_model())
    with pytest.raises(KeyError, match="no indexed initializer"):
        load_graph_text(text, TensorIndex())


def test_load_fails_when_indexed_weight_has_wrong_shape():
    text = graph_text(_matmul_model())
    index = TensorIndex()
    index._by_digest[tensor_digest(W)] = W.reshape(2, 8)
    with pytest.raises(ReproducibilityError, match="shape"):
        load_graph_text(text, index)


def test_snapshot_round_trips_weights_and_reports_file_digest(tmp_path):
    model = _matmul_model()
    path = str(tmp_path / "snap.onnx")
    digest = save_snapshot(model, path)
    assert digest == file_digest(path)
    index = TensorIndex()
    index.add_file(path)
    assert len(index) == 1
    rebuilt = load_graph_text(graph_text(model), index)
    np.testing.assert_array_equal(
        numpy_helper.to_array(rebuilt.graph.initializer[0]), W
    )


def _spec():
    # Non-negative inputs keep the MatMul output positive, so Relu does not clip
    # the perturbation below to zero.
    return (TensorSpec("x", "float32", (1, 4), 0.0, 1.0),)


def test_generated_inputs_are_deterministic_per_seed():
    case = GeneratedCase("a", seed=7, specs=_spec())
    a = generate_feeds(case)["x"]
    b = generate_feeds(case)["x"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(
        a, generate_feeds(GeneratedCase("a", seed=8, specs=_spec()))["x"]
    )
    assert a.dtype == np.float32 and a.shape == (1, 4)


def test_generated_case_digest_mismatch_is_reported():
    good = tensor_digest(generate_feeds(GeneratedCase("a", seed=7, specs=_spec()))["x"])
    resolve_feeds(GeneratedCase("a", seed=7, specs=_spec(), digests={"x": good}))
    with pytest.raises(ReproducibilityError, match="digest"):
        resolve_feeds(
            GeneratedCase("a", seed=7, specs=_spec(), digests={"x": "sha256:00"})
        )


def test_supplied_case_is_checked_against_its_digests(tmp_path):
    x = np.full((1, 4), 0.5, np.float32)
    path = str(tmp_path / "in.npz")
    np.savez(path, x=x)
    case = SuppliedCase("s", path, {"x": tensor_digest(x)})
    np.testing.assert_array_equal(resolve_feeds(case)["x"], x)
    bad = SuppliedCase("s", path, {"x": tensor_digest(x + 1)})
    with pytest.raises(ReproducibilityError):
        resolve_feeds(bad)


def test_mixed_test_set_has_a_stable_identity(tmp_path):
    x = np.full((1, 4), 0.5, np.float32)
    path = str(tmp_path / "in.npz")
    np.savez(path, x=x)
    cases = [
        GeneratedCase("gen", seed=1, specs=_spec()),
        SuppliedCase("sup", path, {"x": tensor_digest(x)}),
    ]
    assert case_set_id(cases) == case_set_id(list(cases))
    assert case_set_id(cases) != case_set_id(cases[::-1])


def test_identical_graphs_are_equivalent():
    cases = [GeneratedCase(f"c{i}", seed=i, specs=_spec()) for i in range(3)]
    reports = check_equivalent(_matmul_model(), _matmul_model(), cases)
    assert all(r.ok for r in reports)
    assert all(r.max_abs_diff == 0.0 for r in reports)


def test_perturbed_weight_fails_and_reports_the_difference():
    cases = [GeneratedCase("c", seed=0, specs=_spec())]
    reports = check_equivalent(_matmul_model(), _matmul_model(W + 0.5), cases)
    assert not reports[0].ok
    assert reports[0].max_abs_diff > 0.1


def test_output_name_mismatch_is_rejected():
    other = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 21]> '
        "g (float[1,4] x) => (float[1,4] z) { z = Relu (x) }"
    )
    with pytest.raises(ValueError, match="output names differ"):
        check_equivalent(_matmul_model(), other, [GeneratedCase("c", 0, _spec())])


def test_record_step_appends_and_reloads(tmp_path):
    path = str(tmp_path / "manifest.json")
    cases = [GeneratedCase("c", seed=0, specs=_spec())]
    reports = check_equivalent(_matmul_model(), _matmul_model(), cases)
    entry = record_step(
        path,
        label="identity",
        command="none",
        base_graph="sha256:base",
        output_graph="sha256:out",
        cases=cases,
        reports=reports,
        executor="test",
    )
    record_step(
        path,
        label="perturbed",
        command="none",
        base_graph="sha256:out",
        output_graph="sha256:out2",
        cases=cases,
        reports=check_equivalent(_matmul_model(), _matmul_model(W + 0.5), cases),
        executor="test",
    )
    manifest = load_manifest(path)
    assert [s["verdict"] for s in manifest["steps"]] == ["pass", "fail"]
    assert manifest["steps"][0] == entry
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["version"] == 1


def test_manifest_rejects_unknown_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 99, "steps": []}))
    with pytest.raises(ValueError, match="unsupported manifest version"):
        load_manifest(str(path))


def test_int64_initializer_used_as_reshape_shape_round_trips():
    model = _model(
        """
        g (float[2,4] x) => (float[4,2] y)
        {
          y = Reshape (x, shape)
        }
        """
    )
    _with_weight(model, "shape", np.array([4, 2], dtype=np.int64))
    text = graph_text(model)
    rebuilt = load_graph_text(text, _index(model))
    np.testing.assert_array_equal(
        numpy_helper.to_array(rebuilt.graph.initializer[0]), [4, 2]
    )
    assert (
        rebuilt.graph.initializer[0].data_type == model.graph.initializer[0].data_type
    )
