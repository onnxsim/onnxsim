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
    bisect_failure,
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
    run_partial,
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
        base_graph="sha256:" + "0" * 64,
        output_graph="sha256:" + "1" * 64,
        cases=cases,
        reports=reports,
        executor="test",
    )
    record_step(
        path,
        label="perturbed",
        command="none",
        base_graph="sha256:" + "1" * 64,
        output_graph="sha256:" + "2" * 64,
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


def _chain(body, extra=()):
    model = _model(
        f"""
        g (float[1,4] x) => (float[1,4] y)
        {{
          {body}
        }}
        """
    )
    for name, arr in extra:
        _with_weight(model, name, arr)
    return model


_BASE_CHAIN = """
  a = Identity (x)
  b = Neg (a)
  c = Abs (b)
  d = Sigmoid (c)
  y = Identity (d)
"""


def test_bisect_returns_none_when_candidate_passes():
    base = _chain(_BASE_CHAIN)
    assert (
        bisect_failure(base, _chain(_BASE_CHAIN), [GeneratedCase("c", 0, _spec())])
        is None
    )


def test_bisect_finds_the_harmful_change_among_benign_ones():
    base = _chain(_BASE_CHAIN)
    # Benign: Identity -> Mul by one. Harmful: Abs -> Add 0.5 (changes the output).
    candidate = _chain(
        """
        a = Mul (x, one)
        b = Neg (a)
        c = Add (b, half)
        d = Sigmoid (c)
        y = Identity (d)
        """,
        extra=[
            ("one", np.array([1.0], np.float32)),
            ("half", np.array([0.5], np.float32)),
        ],
    )
    result = bisect_failure(base, candidate, [GeneratedCase("c", 0, _spec())])
    assert result is not None
    assert len(result.units) == 2
    assert result.culprit.index == 1
    assert "Add:c" in result.culprit.nodes
    assert "Abs:c" in result.culprit.nodes
    assert result.culprit.initializers == ("half",)
    assert result.evaluations == 1


def _matmul_relu(weight_name, tail):
    model = _model(
        f"""
        g (float[1,4] x) => (float[1,4] y)
        {{
          h = MatMul (x, {weight_name})
          y = {tail}
        }}
        """
    )
    return model


def test_fusion_block_names_the_head_and_tail_it_belongs_to():
    base = _with_weight(_matmul_relu("W", "Relu (h)"), "W", W)
    candidate = _with_weight(_matmul_relu("W2", "Relu (h)"), "W2", W + 1.0)
    cases = [GeneratedCase("c", seed=0, specs=_spec())]

    fused = bisect_failure(base, candidate, cases, fusion="default")
    assert set(fused.culprit.nodes) == {"MatMul:h"}
    assert fused.culprit.blocks == ("MatMul+Relu",)

    per_node = bisect_failure(base, candidate, cases, fusion="node")
    assert per_node.culprit.blocks == ("MatMul",)


def test_unknown_fusion_preset_is_rejected():
    base = _chain(_BASE_CHAIN)
    candidate = _chain(_BASE_CHAIN.replace("Abs", "Neg"))
    with pytest.raises(ValueError, match="unknown fusion preset"):
        bisect_failure(
            base, candidate, [GeneratedCase("c", 0, _spec())], fusion="bogus"
        )


def _numpy_backend(node, inputs):
    """A backend that runs three ops and refuses Sigmoid."""
    if node.op_type == "MatMul":
        return [inputs[0] @ inputs[1]]
    if node.op_type == "Relu":
        return [np.maximum(inputs[0], 0)]
    if node.op_type == "Identity":
        return [inputs[0]]
    raise NotImplementedError(f"{node.op_type} is not supported")


def _two_outputs(second):
    model = _model(
        f"""
        g (float[1,4] x) => (float[1,4] y, float[1,4] z)
        {{
          h = MatMul (x, W)
          y = Sigmoid (h)
          z = {second} (h)
        }}
        """
    )
    # Negative weights make MatMul's output negative, so Relu and Identity differ.
    return _with_weight(model, "W", -np.eye(4, dtype=np.float32))


def test_partial_run_judges_outputs_it_can_and_skips_the_rest():
    cases = [GeneratedCase("c", seed=0, specs=_spec())]
    (report,) = check_equivalent(
        _two_outputs("Relu"), _two_outputs("Relu"), cases, backend=_numpy_backend
    )
    assert report.status == "partial"
    assert report.ok
    assert report.skipped == ("y",)
    assert report.failed_nodes == ("Sigmoid:y",)


def test_a_mismatch_is_still_caught_on_an_output_that_can_be_judged():
    cases = [GeneratedCase("c", seed=0, specs=_spec())]
    (report,) = check_equivalent(
        _two_outputs("Relu"), _two_outputs("Identity"), cases, backend=_numpy_backend
    )
    assert report.status == "fail"
    assert not report.ok
    assert report.skipped == ("y",)


def test_partial_run_is_reproducible_for_the_same_seed():
    cand = _two_outputs("Relu")
    feeds = {"x": np.full((1, 4), 0.5, np.float32)}
    sizes = {"h": np.zeros((1, 4), np.float32), "y": np.zeros((1, 4), np.float32)}
    a = run_partial(cand, feeds, _numpy_backend, sizes, seed=3)
    b = run_partial(cand, feeds, _numpy_backend, sizes, seed=3)
    assert a.failed_nodes == b.failed_nodes == ("Sigmoid:y",)
    assert a.skipped == b.skipped == ("y",)


def test_nan_and_infinity_outputs_match_themselves():
    sqrt = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 21]> g (float[4] x) => (float[4] y) { y = Sqrt (x) }'
    )
    cases = [
        GeneratedCase("c", seed=0, specs=(TensorSpec("x", "float32", (4,), -1.0, 1.0),))
    ]
    (report,) = check_equivalent(sqrt, sqrt, cases)
    assert report.ok and report.status == "pass"
