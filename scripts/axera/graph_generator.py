"""Small graph-level AX model generator built from validated templates.

This is the first scheduler-owned layer above the operation emitters.  It
recognizes measured fused Gather and Reshape -> Relu families, collapses each
to one AX program, and retargets its decoded fields without a Pulsar2
invocation.  Unknown graphs are rejected deliberately: a template emitter
cannot safely be promoted to a general graph compiler without knowing the
fused MCode and memory schedule.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from collections.abc import Mapping, Sequence

import binary_op_scale_emit
import compose_emit
import elementwise_scale_emit
import matmul_record_emit
import misc_op_record_emit
import onnx
import reshape_emit
import transpose_real_shapes
from onnx import numpy_helper


@dataclasses.dataclass(frozen=True)
class GraphSegment:
    """One scheduled fused segment and its externally visible tensors."""

    chain: str
    inputs: tuple[str, ...]
    output: str
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    position: str = ""
    operand_shape: tuple[int, ...] = ()
    perm: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class GraphPlan:
    """A complete plan for the currently supported single-segment generator."""

    segments: tuple[GraphSegment, ...]

    @property
    def chain(self) -> str:
        if len(self.segments) != 1:
            raise ValueError("the current generator only emits one fused segment")
        return self.segments[0].chain


def _retarget_schedule_names(
    schedule: dict, model: onnx.ModelProto, source: onnx.ModelProto
) -> None:
    """Make schedule buffer names match a measured model's external names.

    The measured emitters preserve the tensor names from the training-step
    build.  The schedule, however, is built from the source graph (including
    ONNX-to-UOp graphs whose inputs are conventionally named ``x``/``y``).
    Retarget the sidecar by IO position while leaving the model and its MCode
    untouched.
    """
    if len(model.graph.input) != len(source.graph.input) or len(
        model.graph.output
    ) != len(source.graph.output):
        # Calibration-free comparison templates may expose an additional
        # constant/runtime input that is not present in the source UOp graph.
        # Their generated sidecar is still useful for inspection, but cannot
        # be positionally retargeted.
        return
    rename = {
        new.name: old.name
        for old, new in zip(model.graph.input, source.graph.input)
        if old.name != new.name
    }
    rename.update(
        {
            new.name: old.name
            for old, new in zip(model.graph.output, source.graph.output)
            if old.name != new.name
        }
    )

    def replace(value):
        if isinstance(value, str):
            return rename.get(value, value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    retargeted = replace(dict(schedule))
    for kind, specs in (("inputs", model.graph.input), ("outputs", model.graph.output)):
        for entry, spec in zip(retargeted.get(kind, ()), specs):
            entry["name"] = spec.name
            entry["shape"] = [
                int(dim.dim_value) for dim in spec.type.tensor_type.shape.dim
            ]
            # AXCL reports float32 model IO as runtime type 15, while ONNX
            # stores it as TensorProto.FLOAT (1).
            entry["elem_type"] = (
                15
                if spec.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
                else int(spec.type.tensor_type.elem_type)
            )
            entry["nbytes"] = math.prod(entry["shape"]) * 4
    schedule.clear()
    schedule.update(retargeted)


def _shape(value) -> tuple[int, ...]:
    return tuple(int(d.dim_value) for d in value.type.tensor_type.shape.dim)


def _attrs(node: onnx.NodeProto) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {item.name: item for item in model.graph.initializer}


def schedule_graph(model: onnx.ModelProto) -> GraphPlan:
    """Recognize and schedule one measured composed graph.

    Accepted source graph forms are prefixes of the measured chain.  The
    returned chain names match :func:`compose_emit.measured_chains`, so the
    scheduler's output is a single fused AX segment rather than five host
    launches.  Inputs ``x``, ``w`` and ``b`` are required to remain runtime
    inputs; constant folding them would select a different compiled family.
    """
    model = onnx.shape_inference.infer_shapes(model)
    nodes = list(model.graph.node)
    values = {
        v.name: _shape(v)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    if len(nodes) == 2 and [node.op_type for node in nodes] in (
        ["Reshape", "Relu"],
        ["Relu", "Reshape"],
    ):
        reshape = nodes[0] if nodes[0].op_type == "Reshape" else nodes[1]
        if len(reshape.input) != 2 or reshape.input[1] not in _initializer_map(model):
            raise ValueError("fused Reshape must use a constant shape initializer")
        source = values.get(reshape.input[0], ())
        target = tuple(
            int(v)
            for v in numpy_helper.to_array(
                _initializer_map(model)[reshape.input[1]]
            ).reshape(-1)
        )
        position = "before" if nodes[0].op_type == "Reshape" else "after"
        if not source or not target:
            raise ValueError("fused Reshape shapes must be statically known")
        if (source, target) not in (
            reshape_emit.FUSED_BEFORE | reshape_emit.FUSED_AFTER
        ):
            # Let the emitter provide the more specific measured/not-fused
            # refusal when generation is attempted, but do not schedule an
            # unrelated shape as if it were a known fused segment.
            raise ValueError(f"unmeasured fused Reshape pair {source} -> {target}")
        if position == "before" and (source, target) not in reshape_emit.FUSED_BEFORE:
            raise ValueError("Reshape -> Relu pair is not measured as fused")
        if position == "after" and (source, target) not in reshape_emit.FUSED_AFTER:
            raise ValueError("Relu -> Reshape pair is not measured as fused")
        if [v.name for v in model.graph.input] != ["x"]:
            raise ValueError(
                "fused Reshape generator requires one runtime input named x"
            )
        return GraphPlan(
            (
                GraphSegment(
                    "reshape_relu",
                    ("x",),
                    model.graph.output[0].name,
                    source,
                    target,
                    position,
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type == "Relu":
        relu = nodes[0]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError("standalone Relu requires one runtime input named x")
        shape = values.get("x", ())
        output_shape = (
            values.get(model.graph.output[0].name, ()) if model.graph.output else ()
        )
        if not shape or output_shape != shape or list(relu.input) != ["x"]:
            raise ValueError("standalone Relu requires matching static x/y shapes")
        return GraphPlan(
            (GraphSegment("relu", ("x",), model.graph.output[0].name, shape, shape),)
        )
    if len(nodes) == 1 and nodes[0].op_type == "Transpose":
        transpose = nodes[0]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError("standalone Transpose requires one runtime input named x")
        shape = values.get("x", ())
        output_shape = (
            values.get(model.graph.output[0].name, ()) if model.graph.output else ()
        )
        perm = tuple(_attrs(transpose).get("perm", ()))
        if not shape or not output_shape or len(perm) != len(shape):
            raise ValueError(
                "standalone Transpose requires static shapes and a complete permutation"
            )
        if output_shape != tuple(shape[axis] for axis in perm):
            raise ValueError("standalone Transpose output shape does not match perm")
        try:
            transpose_real_shapes.template_path(shape, perm)
        except ValueError as exc:
            raise ValueError(
                f"standalone Transpose has no verified template for {shape} perm {perm}"
            ) from exc
        return GraphPlan(
            (
                GraphSegment(
                    "transpose",
                    ("x",),
                    model.graph.output[0].name,
                    shape,
                    output_shape,
                    perm=perm,
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type == "MatMul":
        matmul = nodes[0]
        if [item.name for item in model.graph.input] != ["x", "z"]:
            raise ValueError("standalone MatMul requires runtime inputs named x and z")
        input_shapes = tuple(values.get(name, ()) for name in matmul.input)
        output_shape = (
            values.get(model.graph.output[0].name, ()) if model.graph.output else ()
        )
        if any(not shape for shape in input_shapes) or not output_shape:
            raise ValueError(
                "standalone MatMul requires static operand and output shapes"
            )
        if len(input_shapes[0]) < 2 or len(input_shapes[1]) < 2:
            raise ValueError("standalone MatMul requires rank-2-or-higher operands")
        if input_shapes[0][-1] != input_shapes[1][-2]:
            raise ValueError(
                "standalone MatMul operands have incompatible contraction dimensions"
            )
        return GraphPlan(
            (
                GraphSegment(
                    "matmul",
                    ("x", "z"),
                    model.graph.output[0].name,
                    input_shapes[0],
                    output_shape,
                    "",
                    input_shapes[1],
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type in ("Neg", "Sqrt", "Log", "Softmax"):
        neg = nodes[0]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError(
                f"standalone {neg.op_type} generator requires one runtime input named x"
            )
        shape = values.get("x", ())
        if not shape or len(neg.input) != 1 or neg.input[0] != "x":
            raise ValueError(
                f"standalone {neg.op_type} requires a static input named x"
            )
        if not model.graph.output or values.get(model.graph.output[0].name) != shape:
            raise ValueError(
                f"standalone {neg.op_type} output shape must match its input"
            )
        return GraphPlan(
            (
                GraphSegment(
                    neg.op_type.lower(),
                    ("x",),
                    model.graph.output[0].name,
                    shape,
                    shape,
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type == "MaxPool":
        max_pool = nodes[0]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError("standalone MaxPool requires one runtime input named x")
        shape = values.get("x", ())
        output_shape = (
            values.get(model.graph.output[0].name, ()) if model.graph.output else ()
        )
        attrs = _attrs(max_pool)
        if (
            shape != (16, 64, 112, 112)
            or output_shape != (16, 64, 56, 56)
            or tuple(attrs.get("kernel_shape", ())) != (3, 3)
            or tuple(attrs.get("strides", ())) != (2, 2)
            or tuple(attrs.get("pads", ())) != (1, 1, 1, 1)
        ):
            raise ValueError(
                "standalone MaxPool requires the measured [16,64,112,112] k3/s2/p1 form"
            )
        return GraphPlan(
            (
                GraphSegment(
                    "maxpool", ("x",), model.graph.output[0].name, shape, output_shape
                ),
            )
        )
    if (
        len(nodes) == 2
        and nodes[0].op_type in ("Greater", "Less")
        and nodes[1].op_type == "Cast"
    ):
        comparison, cast = nodes
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError(
                "standalone comparison cast requires one runtime input named x"
            )
        shape = values.get("x", ())
        if (
            shape not in ((16, 64, 112, 112), (1024, 9, 3136))
            or values.get(cast.output[0], ()) != shape
            or comparison.input[0] != "x"
            or comparison.input[1] not in _initializer_map(model)
            or cast.input[0] != comparison.output[0]
        ):
            raise ValueError("standalone comparison cast requires a measured form")
        return GraphPlan(
            (
                GraphSegment(
                    comparison.op_type.lower() + "cast",
                    ("x",),
                    model.graph.output[0].name,
                    shape,
                    shape,
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type == "ReduceSum":
        reduce_sum = nodes[0]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError("standalone ReduceSum requires one runtime input named x")
        shape = values.get("x", ())
        output_shape = (
            values.get(model.graph.output[0].name, ()) if model.graph.output else ()
        )
        attrs = _attrs(reduce_sum)
        key_by_signature = {
            (
                (16, 64, 112, 112),
                (64,),
                (0, 2, 3),
                0,
            ): "ReduceSum:16x64x112x112:axes0,2,3:k0",
            (
                (16, 1, 64, 3136),
                (1, 64),
                (0, 3),
                0,
            ): "ReduceSum:16x1x64x3136:axes0,3:k0",
            ((16, 1, 512, 49), (1, 512), (0, 3), 0): "ReduceSum:16x1x512x49:axes0,3:k0",
            ((16, 1000), (1, 1000), (0,), 1): "ReduceSum:16x1000:axes0:k1",
            ((16, 1000), (16, 1), (1,), 1): "ReduceSum:16x1000:axes1:k1",
        }
        axes = tuple(attrs.get("axes", ()))
        keepdims = attrs.get("keepdims")
        dynamic_key = misc_op_record_emit.template_key(
            "ReduceSum", shape, axes, keepdims
        )
        key = dynamic_key
        try:
            misc_op_record_emit.load_template(key)
        except ValueError:
            key = key_by_signature.get((shape, output_shape, axes, keepdims))
        if key is None:
            raise ValueError(
                "standalone ReduceSum has no validated template for "
                f"shape={shape}, axes={axes}, keepdims={keepdims}"
            )
        return GraphPlan(
            (
                GraphSegment(
                    "reducesum",
                    ("x",),
                    model.graph.output[0].name,
                    shape,
                    output_shape,
                    key,
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type == "ReduceMean":
        reduce_mean = nodes[0]
        if len(model.graph.input) != 1 or model.graph.input[0].name != "x":
            raise ValueError("standalone ReduceMean requires one runtime input named x")
        shape = values.get("x", ())
        output_shape = (
            values.get(model.graph.output[0].name, ()) if model.graph.output else ()
        )
        attrs = _attrs(reduce_mean)
        if (
            shape != (16, 512, 7, 7)
            or output_shape != (16, 512, 1, 1)
            or tuple(attrs.get("axes", ())) != (2, 3)
            or attrs.get("keepdims") != 1
        ):
            raise ValueError(
                "standalone ReduceMean requires the measured [16,512,7,7] axes [2,3] form"
            )
        return GraphPlan(
            (
                GraphSegment(
                    "reducemean",
                    ("x",),
                    model.graph.output[0].name,
                    shape,
                    output_shape,
                ),
            )
        )
    if len(nodes) == 1 and nodes[0].op_type in ("Add", "Sub", "Mul", "Div"):
        add = nodes[0]
        if len(model.graph.input) != 2 or [item.name for item in model.graph.input] != [
            "x",
            "z",
        ]:
            raise ValueError(
                f"standalone {add.op_type} generator requires runtime inputs named x and z"
            )
        if len(add.input) != 2 or tuple(values.get(name, ()) for name in add.input) != (
            values.get("x", ()),
            values.get("z", ()),
        ):
            raise ValueError(
                f"standalone {add.op_type} inputs must be the graph inputs"
            )
        shape = values.get("x", ())
        if not shape or values.get("z", ()) != shape:
            raise ValueError(
                f"standalone {add.op_type} requires equal static input shapes"
            )
        if not model.graph.output or values.get(model.graph.output[0].name) != shape:
            raise ValueError(
                f"standalone {add.op_type} output shape must match its inputs"
            )
        return GraphPlan(
            (
                GraphSegment(
                    add.op_type.lower(),
                    ("x", "z"),
                    model.graph.output[0].name,
                    shape,
                    shape,
                ),
            )
        )
    if not nodes or nodes[0].op_type != "Gather":
        raise ValueError("graph must start with the measured Gather family")
    init = _initializer_map(model)
    gather = nodes[0]
    gather_attrs = _attrs(gather)
    if gather_attrs.get("axis", 0) != 3:
        raise ValueError("Gather axis must be 3")
    if len(gather.input) != 2 or gather.input[1] not in init:
        raise ValueError("Gather indices must be a constant initializer")
    indices = numpy_helper.to_array(init[gather.input[1]])
    if indices.size != 8:
        raise ValueError("the measured composed family requires eight indices")
    values = {
        v.name: _shape(v)
        for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    if values.get(gather.input[0]) != (1, 1, 4, 16):
        raise ValueError("Gather input must have shape [1,1,4,16]")

    chain = "gather_reshape"
    if len(nodes) >= 2 and nodes[1].op_type == "Reshape":
        chain = "gather_reshape"
    else:
        raise ValueError("measured family requires Gather -> Reshape")

    if len(nodes) >= 3 and nodes[2].op_type == "MatMul":
        chain = "gather_reshape_matmul"
    if len(nodes) >= 4 and nodes[3].op_type == "Transpose":
        if tuple(_attrs(nodes[3]).get("perm", ())) != (0, 1, 3, 2):
            raise ValueError("measured MatMul chain requires Transpose perm [0,1,3,2]")
        chain = "gather_reshape_matmul_transpose"
    if len(nodes) >= 5 and nodes[4].op_type == "Add":
        chain = "gather_reshape_matmul_transpose_add"

    if (
        len(nodes)
        != {
            "gather_reshape": 2,
            "gather_reshape_matmul": 3,
            "gather_reshape_matmul_transpose": 4,
            "gather_reshape_matmul_transpose_add": 5,
        }[chain]
    ):
        raise ValueError("graph has unsupported nodes after the measured chain")
    input_names = tuple(item.name for item in model.graph.input)
    expected_inputs = {
        "gather_reshape": ("x",),
        "gather_reshape_matmul": ("x", "w"),
        "gather_reshape_matmul_transpose": ("x", "w"),
        "gather_reshape_matmul_transpose_add": ("x", "w", "b"),
    }[chain]
    if input_names != expected_inputs:
        raise ValueError(f"runtime inputs must be {expected_inputs}, got {input_names}")
    output = model.graph.output[0].name if model.graph.output else nodes[-1].output[0]
    return GraphPlan((GraphSegment(chain, expected_inputs, output),))


def generate(
    source_path: str,
    output_path: str,
    *,
    indices: Sequence[int] | None = None,
    schedule_path: str | None = None,
    calibration: Mapping[str, Mapping[str, float | int]] | None = None,
) -> GraphPlan:
    """Generate an AX model from a supported ONNX graph without Pulsar2.

    The source graph supplies the schedule/signature.  The checked-in fused
    template supplies the already validated MCode and IO metadata; only the
    measured Gather index table is edited.  ``indices`` defaults to the source
    initializer and must stay in the template's calibrated range.
    """
    model = onnx.load(source_path, load_external_data=False)
    plan = schedule_graph(model)
    if schedule_path is not None:
        # Keep schedule generation on the same validated source model and
        # avoid making the schedule a second, independently maintained plan.
        import json

        import schedule_ir

        schedule = schedule_ir.build(model)
        with open(schedule_path, "w", encoding="utf-8") as stream:
            json.dump(schedule.to_json(), stream, indent=2, sort_keys=True)
            stream.write("\n")
    if plan.chain == "relu":
        if calibration is None:
            raise ValueError("standalone Relu generation requires explicit calibration")
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                "Relu calibration requires scales and zero_points mappings"
            )
        elementwise_scale_emit.emit(
            "Relu", plan.segments[0].input_shape, scales, zero_points, output_path
        )
    elif plan.chain == "reshape_relu":
        segment = plan.segments[0]
        reshape_emit.emit_fused_reshape_axmodel(
            segment.input_shape,
            segment.output_shape,
            output_path,
            position=segment.position,
        )
    elif plan.chain == "transpose":
        with open(output_path, "wb") as stream:
            stream.write(
                transpose_real_shapes.load_template_bytes(
                    plan.segments[0].input_shape, plan.segments[0].perm
                )
            )
    elif plan.chain == "matmul":
        if calibration is None:
            raise ValueError(
                "standalone MatMul generation requires explicit calibration"
            )
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                "MatMul calibration requires scales and zero_points mappings"
            )
        segment = plan.segments[0]
        model = matmul_record_emit.emit_standalone_matmul(
            segment.input_shape,
            segment.operand_shape,
            scales,
            zero_points,
        )
        onnx.save(model, output_path)
    elif plan.chain == "reducemean":
        if calibration is None:
            raise ValueError(
                "standalone ReduceMean generation requires explicit calibration"
            )
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                "ReduceMean calibration requires scales and zero_points mappings"
            )
        model = misc_op_record_emit.emit_spec(
            "ReduceMean",
            plan.segments[0].input_shape,
            axes=(2, 3),
            keepdims=1,
            scales=scales,
            zero_points=zero_points,
        )
        onnx.save(model, output_path)
    elif plan.chain == "reducesum":
        if calibration is None:
            raise ValueError(
                "standalone ReduceSum generation requires explicit calibration"
            )
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                "ReduceSum calibration requires scales and zero_points mappings"
            )
        segment = plan.segments[0]
        attrs = _attrs(model.graph.node[0])
        shape = segment.input_shape
        axes = tuple(attrs.get("axes", ()))
        keepdims = attrs.get("keepdims")
        dynamic_key = misc_op_record_emit.template_key(
            "ReduceSum", shape, axes, keepdims
        )
        if plan.segments[0].position == dynamic_key:
            model = misc_op_record_emit.emit_spec(
                "ReduceSum",
                shape,
                axes=axes,
                keepdims=keepdims,
                scales=scales,
                zero_points=zero_points,
            )
        else:
            model = misc_op_record_emit.emit_model(
                plan.segments[0].position, scales, zero_points
            )
        onnx.save(model, output_path)
    elif plan.chain == "maxpool":
        if calibration is None:
            raise ValueError(
                "standalone MaxPool generation requires explicit calibration"
            )
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                "MaxPool calibration requires scales and zero_points mappings"
            )
        model = misc_op_record_emit.emit_spec(
            "MaxPool",
            plan.segments[0].input_shape,
            attrs={"kernel_shape": (3, 3), "strides": (2, 2), "pads": (1, 1, 1, 1)},
            scales=scales,
            zero_points=zero_points,
        )
        onnx.save(model, output_path)
    elif plan.chain in ("greatercast", "lesscast"):
        model = misc_op_record_emit.emit_spec(
            plan.chain.title().replace("cast", "Cast"),
            plan.segments[0].input_shape,
        )
        onnx.save(model, output_path)
    elif plan.chain in ("neg", "sqrt", "log", "softmax"):
        if calibration is None:
            raise ValueError(
                f"standalone {plan.chain.title()} generation requires explicit calibration"
            )
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                f"{plan.chain.title()} calibration requires scales and zero_points mappings"
            )
        if plan.chain == "softmax":
            model = misc_op_record_emit.emit_spec(
                "Softmax",
                plan.segments[0].input_shape,
                attrs={"axis": 1},
                scales=scales,
                zero_points=zero_points,
            )
        else:
            model = misc_op_record_emit.emit_spec(
                plan.chain.title(),
                plan.segments[0].input_shape,
                scales=scales,
                zero_points=zero_points,
            )
        onnx.save(model, output_path)
    elif plan.chain in ("add", "sub", "mul", "div"):
        if calibration is None:
            raise ValueError(
                f"standalone {plan.chain.title()} generation requires explicit calibration"
            )
        scales = calibration.get("scales")
        zero_points = calibration.get("zero_points")
        if not isinstance(scales, Mapping) or not isinstance(zero_points, Mapping):
            raise ValueError(
                f"{plan.chain.title()} calibration requires scales and zero_points mappings"
            )
        binary_op_scale_emit.emit(
            plan.chain.title(),
            plan.segments[0].input_shape,
            scales,
            zero_points,
            output_path,
        )
    else:
        if indices is None:
            init = _initializer_map(model)
            indices = (
                numpy_helper.to_array(init[model.graph.node[0].input[1]])
                .reshape(-1)
                .tolist()
            )
        compose_emit.emit_gather_in_graph(plan.chain, output_path, indices=indices)
    if not os.path.exists(output_path):
        raise RuntimeError(f"generator did not produce {output_path}")
    if schedule_path is not None:
        with open(schedule_path, encoding="utf-8") as stream:
            schedule = json.load(stream)
        _retarget_schedule_names(
            schedule,
            onnx.load(output_path, load_external_data=False),
            onnx.load(source_path, load_external_data=False),
        )
        with open(schedule_path, "w", encoding="utf-8") as stream:
            json.dump(schedule, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--schedule", dest="schedule_path")
    parser.add_argument("--indices", nargs=8, type=int)
    args = parser.parse_args(argv)
    plan = generate(
        args.source,
        args.output,
        indices=args.indices,
        schedule_path=args.schedule_path,
    )
    print(f"chain={plan.chain} segments={len(plan.segments)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
