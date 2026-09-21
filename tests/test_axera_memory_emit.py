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

from memory_emit import (  # noqa: E402
    _GATHER_LAST_AXIS_TEMPLATES,
    _NOISE_END,
    _NOISE_START,
    emit_gather_axmodel,
    emit_gather_last_axis_axmodel,
    emit_slice_axmodel,
)

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


_FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "axera",
    "fixtures",
)


def _unzip_fixture(tmp_path, name):
    path = tmp_path / name.removesuffix(".gz")
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as source:
        path.write_bytes(source.read())
    return str(path)


def _words(model):
    table = bytes(_init(model, "npu_params").raw_data)
    return struct.unpack(f"<{len(table) // 4}I", table)


def _mcode_outside_noise(model):
    data = bytearray(_init(model, "subgraph_npu_0_b1_neu").raw_data)
    data[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    return bytes(data)


@pytest.mark.parametrize(
    "key", sorted(_GATHER_LAST_AXIS_TEMPLATES), ids=lambda k: f"{k[0]}-n{k[1]}"
)
def test_emit_gather_last_axis_retargets_every_measured_template(tmp_path, key):
    shape, count = key
    reference = _unzip_fixture(tmp_path, _GATHER_LAST_AXIS_TEMPLATES[key])
    source = onnx.load(reference, load_external_data=False)
    # Descending indices ending in duplicates exercise ordering and repeats.
    width = shape[-1]
    indices = [(width - 1 - 3 * i) % width for i in range(count - 2)] + [0, 0]
    target = tmp_path / "gather_target.axmodel"

    emit_gather_last_axis_axmodel(reference, str(target), indices=indices)

    emitted = onnx.load(str(target), load_external_data=False)
    assert _words(emitted) == tuple(indices) + _words(source)[count:]
    assert bytes(_init(emitted, "subgraph_npu_0_b1_neu").raw_data) == bytes(
        _init(source, "subgraph_npu_0_b1_neu").raw_data
    )
    expected_shape = list(shape[:-1]) + [count]
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == expected_shape
    assert json.loads(_attr(emitted.graph.node[0], "outputs_info")) == {
        "y": ["FP32", expected_shape]
    }


@pytest.mark.parametrize(
    "template,oracle",
    [
        (
            "gather_1x1x4x16_axis3_n8.axmodel.gz",
            "gather_1x1x4x16_axis3_n8_oracle_odd.axmodel.gz",
        ),
        (
            "gather_1x1x8x196_axis3_n1764.axmodel.gz",
            "gather_1x1x8x196_axis3_n1764_oracle_b.axmodel.gz",
        ),
    ],
)
def test_emit_gather_last_axis_reproduces_compiler_built_variant(
    tmp_path, template, oracle
):
    reference = _unzip_fixture(tmp_path, template)
    built = onnx.load(_unzip_fixture(tmp_path, oracle), load_external_data=False)
    count = json.loads(_attr(built.graph.node[0], "outputs_info"))["y"][1][-1]
    target = tmp_path / "emitted.axmodel"

    emit_gather_last_axis_axmodel(
        reference, str(target), indices=list(_words(built)[:count])
    )

    emitted = onnx.load(str(target), load_external_data=False)
    assert _words(emitted) == _words(built)
    assert _mcode_outside_noise(emitted) == _mcode_outside_noise(built)


@pytest.mark.parametrize(
    "indices",
    [
        [0, 2, 4],  # wrong count for the [1,1,4,16] x8 template
        [0, 2, 4, 6, 8, 10, 12, 16],  # out of range for width 16
        [0, 2, 4, 6, 8, 10, 12, -1],
        [0, 2, 4, 6, 8, 10, 12, True],
        "01234567",
    ],
)
def test_emit_gather_last_axis_rejects_invalid_indices(tmp_path, indices):
    reference = _unzip_fixture(tmp_path, "gather_1x1x4x16_axis3_n8.axmodel.gz")
    with pytest.raises(ValueError):
        emit_gather_last_axis_axmodel(
            reference, str(tmp_path / "bad.axmodel"), indices=indices
        )


def test_emit_gather_last_axis_rejects_unmeasured_shape(tmp_path):
    reference = _unzip_fixture(tmp_path, "gather_1x1x4x16_axis3_n8.axmodel.gz")
    model = onnx.load(reference, load_external_data=False)
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 32
    onnx.save(model, reference)
    with pytest.raises(ValueError, match="unmeasured"):
        emit_gather_last_axis_axmodel(
            reference, str(tmp_path / "bad.axmodel"), indices=list(range(8))
        )


def test_emit_gather_last_axis_rejects_altered_table_tail_and_mcode(tmp_path):
    reference = _unzip_fixture(tmp_path, "gather_1x1x4x16_axis3_n8.axmodel.gz")
    model = onnx.load(reference, load_external_data=False)
    words = list(_words(model))
    words[-1] = 1
    _init(model, "npu_params").raw_data = struct.pack(f"<{len(words)}I", *words)
    onnx.save(model, reference)
    with pytest.raises(ValueError, match="tail"):
        emit_gather_last_axis_axmodel(
            reference, str(tmp_path / "bad.axmodel"), indices=list(range(8))
        )

    reference = _unzip_fixture(tmp_path, "gather_1x1x4x16_axis3_n8.axmodel.gz")
    model = onnx.load(reference, load_external_data=False)
    mcode = _init(model, "subgraph_npu_0_b1_neu")
    data = bytearray(mcode.raw_data)
    data[1000] ^= 0xFF
    mcode.raw_data = bytes(data)
    onnx.save(model, reference)
    with pytest.raises(ValueError, match="MCode"):
        emit_gather_last_axis_axmodel(
            reference, str(tmp_path / "bad.axmodel"), indices=list(range(8))
        )
