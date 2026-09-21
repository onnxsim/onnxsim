"""Regression coverage for the evidence-scoped AX650 Slice model emitter."""

import gzip
import json
import os
import struct
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from memory_emit import emit_gather_axmodel, emit_slice_axmodel  # noqa: E402

_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "axera",
    "fixtures",
    "slice_1x8_axis1_step1_len4.axmodel.gz",
)
_GATHER_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "axera",
    "fixtures",
    "gather_1x8_axis1_even4.axmodel.gz",
)
_SLICE_STEP2_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "axera",
    "fixtures",
    "slice_1x8_axis1_step2_len4.axmodel.gz",
)
_SLICE_STEP3_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "axera",
    "fixtures",
    "slice_1x8_axis1_step3_len3.axmodel.gz",
)
_SLICE_STEP4_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "axera",
    "fixtures",
    "slice_1x8_axis1_step4_len2.axmodel.gz",
)


def _fixture_path(tmp_path):
    path = tmp_path / "slice_reference.axmodel"
    with gzip.open(_FIXTURE, "rb") as source, path.open("wb") as target:
        target.write(source.read())
    return str(path)


def _init(model, name):
    return next(item for item in model.graph.initializer if item.name == name)


def _attr(node, name):
    return next(
        onnx.helper.get_attribute_value(item)
        for item in node.attribute
        if item.name == name
    )


@pytest.mark.parametrize(
    "start,end,expected_params,expected_shape",
    [
        (0, 3, 0, [1, 3]),
        (0, 4, 0, [1, 4]),
        (1, 4, 4, [1, 3]),
        (1, 5, 4, [1, 4]),
        (2, 5, 8, [1, 3]),
        (3, 6, 12, [1, 3]),
        (4, 7, 16, [1, 3]),
        (4, 8, 16, [1, 4]),
    ],
)
def test_emit_slice_retargets_offset_and_shape_but_preserves_mcode(
    tmp_path, start, end, expected_params, expected_shape
):
    reference = _fixture_path(tmp_path)
    target = tmp_path / "slice_target.axmodel"
    source_model = onnx.load(reference, load_external_data=False)
    source_mcode = bytes(_init(source_model, "subgraph_npu_0_b1_neu").raw_data)

    emit_slice_axmodel(reference, str(target), start=start, end=end)

    emitted = onnx.load(str(target), load_external_data=False)
    assert (
        struct.unpack("<5Q", _init(emitted, "npu_params").raw_data)
        == (expected_params,) * 5
    )
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == source_mcode
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == expected_shape
    assert json.loads(_attr(emitted.graph.node[0], "outputs_info")) == {
        "y": ["FP32", expected_shape]
    }


@pytest.mark.parametrize(
    "start,end",
    [(-1, 2), (2, 2), (5, 9), (1.0, 4), (0, 2), (2, 7), (0, 8)],
)
def test_emit_slice_rejects_out_of_scope_ranges(tmp_path, start, end):
    reference = _fixture_path(tmp_path)
    with pytest.raises(ValueError):
        emit_slice_axmodel(
            reference, str(tmp_path / "bad.axmodel"), start=start, end=end
        )


def _gather_fixture_path(tmp_path):
    path = tmp_path / "gather_reference.axmodel"
    with gzip.open(_GATHER_FIXTURE, "rb") as source, path.open("wb") as target:
        target.write(source.read())
    return str(path)


def _slice_step2_fixture_path(tmp_path):
    path = tmp_path / "slice_step2_reference.axmodel"
    with gzip.open(_SLICE_STEP2_FIXTURE, "rb") as source, path.open("wb") as target:
        target.write(source.read())
    return str(path)


def _slice_step3_fixture_path(tmp_path):
    path = tmp_path / "slice_step3_reference.axmodel"
    with gzip.open(_SLICE_STEP3_FIXTURE, "rb") as source, path.open("wb") as target:
        target.write(source.read())
    return str(path)


def _slice_step4_fixture_path(tmp_path):
    path = tmp_path / "slice_step4_reference.axmodel"
    with gzip.open(_SLICE_STEP4_FIXTURE, "rb") as source, path.open("wb") as target:
        target.write(source.read())
    return str(path)


@pytest.mark.parametrize(
    "indices,expected_words",
    [
        ([0, 2, 4, 6], (0, 2, 4, 6) + (0,) * 10),
        ([1, 3, 5, 7], (1, 3, 5, 7) + (0,) * 10),
        ([0, 3, 5, 7], (0, 3, 5, 7) + (0,) * 10),
        ([7, 6, 5, 4], (7, 6, 5, 4) + (0,) * 10),
        ([7, 7, 0, 0], (7, 7, 0, 0) + (0,) * 10),
    ],
)
def test_emit_gather_retargets_measured_indices_and_preserves_mcode(
    tmp_path, indices, expected_words
):
    reference = _gather_fixture_path(tmp_path)
    target = tmp_path / "gather_target.axmodel"
    source_model = onnx.load(reference, load_external_data=False)
    source_mcode = bytes(_init(source_model, "subgraph_npu_0_b1_neu").raw_data)

    emit_gather_axmodel(reference, str(target), indices=indices)

    emitted = onnx.load(str(target), load_external_data=False)
    assert struct.unpack("<14I", _init(emitted, "npu_params").raw_data) == (
        expected_words
    )
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == source_mcode
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == [1, 4]
    assert json.loads(_attr(emitted.graph.node[0], "outputs_info")) == {
        "y": ["FP32", [1, 4]]
    }


@pytest.mark.parametrize(
    "indices", [[0, 2, 4], [0, 2, 4, 8], [0, 2, 4, True], [0, 2, 4, 5, 6]]
)
def test_emit_gather_rejects_invalid_index_vectors(tmp_path, indices):
    reference = _gather_fixture_path(tmp_path)
    with pytest.raises(ValueError):
        emit_gather_axmodel(reference, str(tmp_path / "bad.axmodel"), indices=indices)


def test_emit_gather_rejects_reference_with_unmeasured_parameter_table(tmp_path):
    reference = _gather_fixture_path(tmp_path)
    model = onnx.load(reference, load_external_data=False)
    table = _init(model, "npu_params")
    table.raw_data = struct.pack("<14I", *range(14))
    onnx.save(model, reference)

    with pytest.raises(ValueError, match="padding"):
        emit_gather_axmodel(
            reference, str(tmp_path / "bad.axmodel"), indices=[1, 3, 5, 7]
        )


@pytest.mark.parametrize("start", [0, 1])
def test_emit_slice_step2_retargets_offset_and_preserves_mcode(tmp_path, start):
    reference = _slice_step2_fixture_path(tmp_path)
    target = tmp_path / "slice_step2_target.axmodel"
    source_model = onnx.load(reference, load_external_data=False)
    source_mcode = bytes(_init(source_model, "subgraph_npu_0_b1_neu").raw_data)

    emit_slice_axmodel(reference, str(target), start=start, end=8, step=2)

    emitted = onnx.load(str(target), load_external_data=False)
    assert (
        struct.unpack("<5Q", _init(emitted, "npu_params").raw_data) == (start * 4,) * 5
    )
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == source_mcode
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == [1, 4]
    assert json.loads(_attr(emitted.graph.node[0], "outputs_info")) == {
        "y": ["FP32", [1, 4]]
    }


@pytest.mark.parametrize(
    "start,end,step",
    [
        (2, 8, 2),
        (0, 7, 2),
        (0, 8, 0),
        (0, 8, True),
        (2, 8, 3),
        (0, 7, 3),
        (2, 8, 4),
        (0, 7, 4),
    ],
)
def test_emit_slice_rejects_unmeasured_nonunit_ranges(tmp_path, start, end, step):
    reference_for_step = {
        1: _fixture_path,
        2: _slice_step2_fixture_path,
        3: _slice_step3_fixture_path,
        4: _slice_step4_fixture_path,
    }
    reference = reference_for_step.get(step, _fixture_path)(tmp_path)
    with pytest.raises(ValueError):
        emit_slice_axmodel(
            reference,
            str(tmp_path / "bad_step2.axmodel"),
            start=start,
            end=end,
            step=step,
        )


@pytest.mark.parametrize("start", [0, 1])
def test_emit_slice_step3_retargets_offset_and_preserves_mcode(tmp_path, start):
    reference = _slice_step3_fixture_path(tmp_path)
    target = tmp_path / "slice_step3_target.axmodel"
    source_model = onnx.load(reference, load_external_data=False)
    source_mcode = bytes(_init(source_model, "subgraph_npu_0_b1_neu").raw_data)

    emit_slice_axmodel(reference, str(target), start=start, end=8, step=3)

    emitted = onnx.load(str(target), load_external_data=False)
    assert (
        struct.unpack("<5Q", _init(emitted, "npu_params").raw_data) == (start * 4,) * 5
    )
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == source_mcode
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == [1, 3]
    assert json.loads(_attr(emitted.graph.node[0], "outputs_info")) == {
        "y": ["FP32", [1, 3]]
    }


@pytest.mark.parametrize("start", [0, 1])
def test_emit_slice_step4_retargets_offset_and_preserves_mcode(tmp_path, start):
    reference = _slice_step4_fixture_path(tmp_path)
    target = tmp_path / "slice_step4_target.axmodel"
    source_model = onnx.load(reference, load_external_data=False)
    source_mcode = bytes(_init(source_model, "subgraph_npu_0_b1_neu").raw_data)

    emit_slice_axmodel(reference, str(target), start=start, end=8, step=4)

    emitted = onnx.load(str(target), load_external_data=False)
    assert (
        struct.unpack("<5Q", _init(emitted, "npu_params").raw_data) == (start * 4,) * 5
    )
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == source_mcode
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == [1, 2]
    assert json.loads(_attr(emitted.graph.node[0], "outputs_info")) == {
        "y": ["FP32", [1, 2]]
    }
