"""Regression coverage for the evidence-scoped AX650 Slice model emitter."""

import gzip
import json
import os
import struct

import onnx
import pytest

from scripts.axera.memory_emit import emit_slice_axmodel

_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "axera",
    "fixtures",
    "slice_1x8_axis1_step1_len4.axmodel.gz",
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
    [(0, 4, 0, [1, 4]), (1, 5, 4, [1, 4]), (2, 5, 8, [1, 3]), (4, 8, 16, [1, 4])],
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
    [(-1, 2), (2, 2), (5, 9), (1.0, 4), (0, 3), (3, 6), (0, 8)],
)
def test_emit_slice_rejects_out_of_scope_ranges(tmp_path, start, end):
    reference = _fixture_path(tmp_path)
    with pytest.raises(ValueError):
        emit_slice_axmodel(
            reference, str(tmp_path / "bad.axmodel"), start=start, end=end
        )
