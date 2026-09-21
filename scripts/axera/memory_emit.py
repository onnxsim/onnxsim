"""Small, evidence-scoped emitters for Axera memory-only operators.

Pulsar2 7.0-lite emits measured AX650 ``Slice`` templates for ``X[1, 8]``
(axis 1), covering step-one lengths 3/4 and steps two through four. The
operation's byte offset is stored in five repeated uint64 entries in
``npu_params``; the output shape is stored in the ``neu mode`` node's
``outputs_info`` attribute plus ONNX output metadata. Step-one length-three
models use the measured length-four template; compiler-built length-three
MCode has extra shape-specific bytes, but the patched template ran correctly
for every tested start. Steps two through four each have their own compiled template; each
supports only starts 0 and 1 with end 8. Other ranks, axes, steps, and input
shapes remain out of scope.

It also retargets measured static ``Gather`` index vectors for
``X[1,8] -> Y[1,4]``. Their indices are stored in the compiled model's
``npu_params`` table; other Gather shapes are out of scope.

``emit_gather_last_axis_axmodel`` extends that to the training-graph shape
family: ``Gather(x[..., W], axis=-1)`` on float32 inputs of rank >= 2 (the
im2col tap gather of a legalized convolution backward). Each measured
``(input shape, index count)`` pair has its own compiled template, because the
MCode depends on the shape and count but not on index values.

The evidence and hardware check are recorded in the Axera MCode coverage
notes. The reference fixture was built with start=2/end=6. Length-four builds
with starts 0..4 changed no MCode bytes outside the known compiler-noise
region at offsets 301..325. Compiler-built length-three variants also change
bytes 988, 1172, and 1584; emitted length-three models retain the length-four
template and have been hardware-verified. Step-two starts 0 and 1 shared a
second MCode template, with `npu_params` offsets 0 and 4 respectively; see the
separate step-two fixture. Step-three starts 0 and 1 shared a third MCode
template, again with offsets 0 and 4; see the step-three fixture. Step-four
starts 0 and 1 share a fourth template with offsets 0 and 4.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
from collections.abc import Sequence

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_SLICE_TEMPLATE = os.path.join(
    _HERE, "fixtures", "slice_1x8_axis1_step1_len4.axmodel.gz"
)
_SLICE_STEP2_TEMPLATE = os.path.join(
    _HERE, "fixtures", "slice_1x8_axis1_step2_len4.axmodel.gz"
)
_SLICE_STEP3_TEMPLATE = os.path.join(
    _HERE, "fixtures", "slice_1x8_axis1_step3_len3.axmodel.gz"
)
_SLICE_STEP4_TEMPLATE = os.path.join(
    _HERE, "fixtures", "slice_1x8_axis1_step4_len2.axmodel.gz"
)
_GATHER_TEMPLATE = os.path.join(_HERE, "fixtures", "gather_1x8_axis1_even4.axmodel.gz")
# Measured last-axis Gather templates, keyed by (input shape, index count).
# Each is a real Pulsar2 7.0-lite AX650 build of ``Gather(x, idx, axis=-1)``.
_GATHER_LAST_AXIS_TEMPLATES = {
    ((1, 1, 4, 16), 8): "gather_1x1x4x16_axis3_n8.axmodel.gz",
    ((1, 1, 4, 16), 16): "gather_1x1x4x16_axis3_n16.axmodel.gz",
    ((1, 1, 4, 256), 8): "gather_1x1x4x256_axis3_n8.axmodel.gz",
    ((2, 1, 4, 16), 8): "gather_2x1x4x16_axis3_n8.axmodel.gz",
    ((1, 1, 8, 196), 1764): "gather_1x1x8x196_axis3_n1764.axmodel.gz",
    ((1, 1, 4, 70000), 8): "gather_1x1x4x70000_axis3_n8.axmodel.gz",
}
_NOISE_START = 301
_NOISE_END = 326
_GATHER_INDEX_COUNT = 4
_GATHER_INPUT_WIDTH = 8


def _supported_slice(start: int, end: int, step: int) -> bool:
    if step == 1:
        return end - start in (3, 4) and 0 <= start <= 4
    return step in (2, 3, 4) and end == 8 and start in (0, 1)


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


def _validate_slice_template(
    model: onnx.ModelProto, template_path: str = _SLICE_TEMPLATE
) -> tuple[onnx.NodeProto, int]:
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
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT or [
        d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
    ] != [1, 8]:
        raise ValueError("reference input must be float32 with shape [1, 8]")

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
    if not 0 <= source_start <= 4:
        raise ValueError("reference slice start is outside the tested cases")

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

    with gzip.open(template_path, "rb") as f:
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
    reference_path: str, output_path: str, *, start: int, end: int, step: int = 1
) -> str:
    """Retarget a compiled ``Slice(x[1,8], axis=1)`` template.

    ``start``, ``end``, and ``step`` follow ONNX's positive-step, end-exclusive
    semantics. Step 1 supports output lengths 3/4 with starts 0..4. Step 2
    supports only ``[0:8:2]`` and ``[1:8:2]``; step 3 supports only
    ``[0:8:3]`` and ``[1:8:3]``; step 4 supports only ``[0:8:4]`` and
    ``[1:8:4]``. The target output shape is
    ``[1, ceil((end - start) / step)]``. The five repeated offset words in
    ``npu_params`` and output shape metadata are updated; the matching
    characterized MCode template is retained byte-for-byte.

    This API refuses unsupported layouts instead of extrapolating the
    observed encoding to another rank/dtype/axis/step.
    """
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in (start, end)
    ):
        raise ValueError("start and end must be integers")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError("step must be an integer")
    if not 0 <= start < end <= 8 or not _supported_slice(start, end, step):
        raise ValueError(
            "step 1 supports length 3/4 with start 0..4; step 2 supports "
            "[0:8:2]/[1:8:2]; step 3 supports [0:8:3]/[1:8:3]; "
            "step 4 supports [0:8:4]/[1:8:4]; "
            f"got [{start}, {end}:{step}]"
        )

    model = onnx.load(reference_path, load_external_data=False)
    template_path = {
        1: _SLICE_TEMPLATE,
        2: _SLICE_STEP2_TEMPLATE,
        3: _SLICE_STEP3_TEMPLATE,
        4: _SLICE_STEP4_TEMPLATE,
    }[step]
    node, _ = _validate_slice_template(model, template_path)
    table = _initializer(model, "npu_params")
    table.raw_data = struct.pack("<5Q", *([start * 4] * 5))
    target_shape = [1, (end - start + step - 1) // step]

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


def _gather_indices_from_params(model: onnx.ModelProto) -> tuple[int, ...]:
    table = bytes(_initializer(model, "npu_params").raw_data)
    if len(table) != 56:
        raise ValueError(f"expected Gather's 56-byte npu_params, got {len(table)}")
    words = struct.unpack("<14I", table)
    if any(words[4:]):
        raise ValueError("Gather parameter padding must contain only zeros")
    indices = tuple(words[:_GATHER_INDEX_COUNT])
    if any(index >= _GATHER_INPUT_WIDTH for index in indices):
        raise ValueError(f"reference Gather index is out of bounds: {indices}")
    return indices


def emit_gather_axmodel(
    reference_path: str, output_path: str, *, indices: Sequence[int]
) -> str:
    """Retarget the measured float32 ``Gather(x[1,8], axis=1)`` template.

    The four indices may be any values in ``[0,7]``, including duplicates.
    They occupy the first four little-endian uint32 words of a 56-byte
    ``npu_params`` table; the remaining ten words are zero padding. This
    encoding was checked against varied, descending, and duplicate index
    vectors on the NPU. The compiled output shape and MCode do not change.
    Other ranks, axes, and output lengths are rejected.
    """
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise ValueError("indices must be a sequence of four integers")
    target_indices = tuple(indices)
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in target_indices
    ):
        raise ValueError("indices must contain integers")
    if len(target_indices) != _GATHER_INDEX_COUNT or any(
        not 0 <= value < _GATHER_INPUT_WIDTH for value in target_indices
    ):
        raise ValueError("indices must contain four integers in the range [0, 7]")

    model = onnx.load(reference_path, load_external_data=False)
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
    if model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT or [
        d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
    ] != [1, 8]:
        raise ValueError("reference input must be float32 with shape [1, 8]")
    if _output_dims(model.graph.output[0], "y") != [1, 4]:
        raise ValueError("reference output must have shape [1, 4]")
    _gather_indices_from_params(model)

    dynamic = _initializer(model, "npu_dyn_params")
    if dynamic.raw_data or dynamic.dims != [0]:
        raise ValueError("reference must have the empty npu_dyn_params initializer")

    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in model.graph.node[0].attribute
    }
    outputs_info = json.loads(attrs.get("outputs_info", b"{}"))
    if outputs_info != {"y": ["FP32", [1, 4]]}:
        raise ValueError(f"outputs_info disagrees with output shape: {outputs_info!r}")

    with gzip.open(_GATHER_TEMPLATE, "rb") as f:
        expected = onnx.load_model_from_string(f.read())
    template_mcode = _mcode(expected)
    actual_mcode = _mcode(model)
    if len(actual_mcode) != len(template_mcode):
        raise ValueError(
            "reference MCode size does not match the known Gather template"
        )
    actual = bytearray(actual_mcode)
    wanted = bytearray(template_mcode)
    actual[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    wanted[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    if actual != wanted:
        raise ValueError(
            "reference MCode does not match the characterized Gather template"
        )

    table = _initializer(model, "npu_params")
    table.raw_data = struct.pack("<14I", *target_indices, *([0] * 10))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def _load_gather_last_axis_template(key) -> onnx.ModelProto:
    with gzip.open(
        os.path.join(_HERE, "fixtures", _GATHER_LAST_AXIS_TEMPLATES[key]), "rb"
    ) as f:
        return onnx.load_model_from_string(f.read())


def _param_words(model: onnx.ModelProto) -> tuple[int, ...]:
    table = bytes(_initializer(model, "npu_params").raw_data)
    if len(table) % 4:
        raise ValueError(f"npu_params is not whole uint32 words: {len(table)} bytes")
    return struct.unpack(f"<{len(table) // 4}I", table)


def emit_gather_last_axis_axmodel(
    reference_path: str, output_path: str, *, indices: Sequence[int]
) -> str:
    """Retarget a measured float32 ``Gather(x[..., W], axis=-1)`` template.

    ``indices`` may be any values in ``[0, W)`` (duplicates and descending
    order included) but must have exactly the index count the reference was
    compiled for: the MCode is specific to the input shape and that count. The
    first ``len(indices)`` little-endian uint32 words of ``npu_params`` hold
    the indices; the words after them are shape-dependent (ten zeros for most
    shapes, a tiling table for a very wide axis) and are preserved from the
    reference. Only ``(input shape, count)`` pairs in
    ``_GATHER_LAST_AXIS_TEMPLATES`` are accepted; the reference's normalized
    MCode and table tail must match that pair's compiled fixture.
    """
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise ValueError("indices must be a sequence of integers")
    target = tuple(indices)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in target):
        raise ValueError("indices must contain integers")

    model = onnx.load(reference_path, load_external_data=False)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("reference must contain exactly one compiled 'neu mode' node")
    if list(model.graph.node[0].input) != ["x"] or list(model.graph.node[0].output) != [
        "y"
    ]:
        raise ValueError("reference NPU node must map input 'x' to output 'y'")
    if [v.name for v in model.graph.input] != ["x"]:
        raise ValueError("reference must have exactly one input named 'x'")
    if [v.name for v in model.graph.output] != ["y"]:
        raise ValueError("reference must have exactly one output named 'y'")
    x_type = model.graph.input[0].type.tensor_type
    if x_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("reference input must be float32")
    in_shape = tuple(d.dim_value for d in x_type.shape.dim)
    out_shape = _output_dims(model.graph.output[0], "y")
    if len(in_shape) < 2 or out_shape[:-1] != list(in_shape[:-1]):
        raise ValueError(
            f"reference is not a last-axis Gather: {in_shape}->{out_shape}"
        )
    count = out_shape[-1]

    key = (in_shape, count)
    if key not in _GATHER_LAST_AXIS_TEMPLATES:
        measured = sorted(_GATHER_LAST_AXIS_TEMPLATES)
        raise ValueError(
            f"unmeasured Gather (input shape, index count) {key}; measured: {measured}"
        )
    if len(target) != count or any(not 0 <= v < in_shape[-1] for v in target):
        raise ValueError(
            f"indices must be {count} integers in the range [0, {in_shape[-1] - 1}]"
        )

    dynamic = _initializer(model, "npu_dyn_params")
    if dynamic.raw_data or dynamic.dims != [0]:
        raise ValueError("reference must have the empty npu_dyn_params initializer")
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in model.graph.node[0].attribute
    }
    outputs_info = json.loads(attrs.get("outputs_info", b"{}"))
    if outputs_info != {"y": ["FP32", out_shape]}:
        raise ValueError(f"outputs_info disagrees with output shape: {outputs_info!r}")

    expected = _load_gather_last_axis_template(key)
    template_words = _param_words(expected)
    words = _param_words(model)
    if len(words) != len(template_words) or words[count:] != template_words[count:]:
        raise ValueError("reference npu_params tail does not match the measured layout")
    if any(v >= in_shape[-1] for v in words[:count]):
        raise ValueError("reference Gather index is out of bounds")

    template_mcode = _mcode(expected)
    actual_mcode = _mcode(model)
    if len(actual_mcode) != len(template_mcode):
        raise ValueError(
            "reference MCode size does not match the known Gather template"
        )
    actual = bytearray(actual_mcode)
    wanted = bytearray(template_mcode)
    actual[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    wanted[_NOISE_START:_NOISE_END] = bytes(_NOISE_END - _NOISE_START)
    if actual != wanted:
        raise ValueError(
            "reference MCode does not match the characterized Gather template"
        )

    table = _initializer(model, "npu_params")
    table.raw_data = struct.pack(f"<{len(words)}I", *target, *words[count:])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    return output_path
