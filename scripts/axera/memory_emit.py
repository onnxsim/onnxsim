"""Small, evidence-scoped emitters for Axera memory-only operators.

Pulsar2 7.0-lite emits the same MCode program for the tested AX650 ``Slice``
shape ``X[1, 8] -> Y[1, n]`` (axis 1, step 1), for intervals ``[0:4]``,
``[1:5]``, ``[2:6]``, ``[3:7]``, ``[4:8]``, and ``[2:5]``. The operation's
byte offset is stored in five repeated
uint64 entries in ``npu_params`` and the output shape is stored in the
``neu mode`` node's ``outputs_info`` attribute plus ONNX output metadata.
This module patches those fields on a compiled template. It deliberately
does not generalize to other Slice ranks, axes, steps, or input shapes.

The evidence and hardware check are recorded in the Axera MCode coverage
notes. The reference fixture was built with start=2/end=6; builds with
start=0..4 and lengths 3/4, including an exact-config rebuild, changed no
MCode bytes outside the known compiler-noise region at offsets 301..325.
"""

from __future__ import annotations

import gzip
import json
import os
import struct

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_SLICE_TEMPLATE = os.path.join(
    _HERE, "fixtures", "slice_1x8_axis1_step1_len4.axmodel.gz"
)
_NOISE_START = 301
_NOISE_END = 326


def _supported_interval(start: int, end: int) -> bool:
    return (end - start == 4 and 0 <= start <= 4) or (start == 2 and end == 5)


def _initializer(model: onnx.ModelProto, name: str):
    matches = [item for item in model.graph.initializer if item.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {name!r} initializer, found {len(matches)}"
        )
    return matches[0]


def _output_dims(value_info, name: str) -> list[int]:
    if (
        value_info.name != name
        or value_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError(f"expected float32 output {name!r}")
    dims = value_info.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") for dim in dims):
        raise ValueError(f"{name!r} must have a static shape")
    return [dim.dim_value for dim in dims]


def _set_dims(value_info, shape: list[int]) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for size in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = size


def _mcode(model: onnx.ModelProto) -> bytes:
    matches = [item for item in model.graph.initializer if item.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *_neu MCode initializer, found {len(matches)}"
        )
    return bytes(matches[0].raw_data)


def _validate_slice_template(model: onnx.ModelProto) -> tuple[onnx.NodeProto, int]:
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("reference must contain exactly one compiled 'neu mode' node")
    if list(model.graph.node[0].input) != ["x"] or list(model.graph.node[0].output) != [
        "y"
    ]:
        raise ValueError("reference NPU node must map input 'x' to output 'y'")
    if [value.name for value in model.graph.input] != ["x"]:
        raise ValueError("reference must have exactly one input named 'x'")
    if [value.name for value in model.graph.output] != ["y"]:
        raise ValueError("reference must have exactly one output named 'y'")
    if [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] != [1, 8]:
        raise ValueError("reference input must have shape [1, 8]")

    output_shape = _output_dims(model.graph.output[0], "y")
    if len(output_shape) != 2 or output_shape[0] != 1:
        raise ValueError(f"reference output shape is unsupported: {output_shape}")

    table = bytes(_initializer(model, "npu_params").raw_data)
    if len(table) != 40:
        raise ValueError(
            f"expected the Slice template's 40-byte npu_params, got {len(table)}"
        )
    offsets = struct.unpack("<5Q", table)
    if len(set(offsets)) != 1 or offsets[0] % 4:
        raise ValueError(
            f"npu_params is not five equal float32 byte offsets: {offsets}"
        )
    source_start = offsets[0] // 4
    source_end = source_start + output_shape[1]
    if not _supported_interval(source_start, source_end):
        raise ValueError("reference slice interval is outside the tested cases")

    dynamic = _initializer(model, "npu_dyn_params")
    if dynamic.raw_data or dynamic.dims != [0]:
        raise ValueError("reference must have the empty npu_dyn_params initializer")

    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in model.graph.node[0].attribute
    }
    if "outputs_info" not in attrs:
        raise ValueError("reference node has no outputs_info metadata")
    outputs_info = json.loads(attrs["outputs_info"])
    if outputs_info != {"y": ["FP32", output_shape]}:
        raise ValueError(f"outputs_info disagrees with output shape: {outputs_info!r}")

    with gzip.open(_SLICE_TEMPLATE, "rb") as f:
        expected = onnx.load_model_from_string(f.read())
    template_mcode = _mcode(expected)
    actual_mcode = _mcode(model)
    if len(actual_mcode) != len(template_mcode):
        raise ValueError("reference MCode size does not match the known Slice template")
    actual = bytearray(actual_mcode)
    wanted = bytearray(template_mcode)
    actual[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    wanted[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    if actual != wanted:
        raise ValueError(
            "reference MCode does not match the characterized Slice template"
        )

    return model.graph.node[0], output_shape[1]


def emit_slice_axmodel(
    reference_path: str, output_path: str, *, start: int, end: int
) -> str:
    """Retarget a compiled ``Slice(x[1,8], axis=1, step=1)`` template.

    ``start`` and ``end`` follow ONNX's non-negative, end-exclusive semantics.
    The target output has shape ``[1, end - start]``. Only the five repeated
    offset words in ``npu_params`` and the output shape metadata are changed;
    the characterized MCode stream is retained byte-for-byte.

    This API refuses unsupported layouts instead of extrapolating the
    observed byte-offset rule to another rank/dtype/axis/step.
    """
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in (start, end)
    ):
        raise ValueError("start and end must be integers")
    if not 0 <= start < end <= 8 or not _supported_interval(start, end):
        raise ValueError(
            "supported intervals are [0:4], [1:5], [2:6], [3:7], [4:8], "
            f"and [2:5]; got [{start}, {end})"
        )

    model = onnx.load(reference_path, load_external_data=False)
    node, _ = _validate_slice_template(model)
    table = _initializer(model, "npu_params")
    table.raw_data = struct.pack("<5Q", *([start * 4] * 5))
    target_shape = [1, end - start]

    outputs_info = next(
        (attr for attr in node.attribute if attr.name == "outputs_info"), None
    )
    if outputs_info is None:
        raise ValueError("reference node has no outputs_info metadata")
    outputs_info.s = json.dumps({"y": ["FP32", target_shape]}).encode()

    _set_dims(model.graph.output[0], target_shape)
    for value_info in model.graph.value_info:
        if value_info.name == "y":
            _set_dims(value_info, target_shape)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
