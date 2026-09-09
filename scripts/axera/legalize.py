#!/usr/bin/env python3
"""Rewrites that make an ONNX graph acceptable to Pulsar2 for the AX650N.

Every rule here exists because a real `pulsar2 build` refused a real model
without it, and each records which failure it answers. They are semantics-
preserving: a legalized graph computes the same function, it is only spelled
in a way the compiler can lower.

This is deliberately separate from op *coverage* (`op_coverage.py`). Coverage
asks whether the ONNX op types are on the vendor's list, which is necessary
and, as the Audio8 codec decoder showed, nowhere near sufficient -- that graph
reaches 99.3% eligible and still fails, on an op the compiler *fuses into
existence itself*. A legalizer is how you act on that gap.

Usage::

    legalize.py in.onnx out.onnx            # apply every rule
    legalize.py --rules pow2_to_mul in.onnx out.onnx
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import onnx
import onnx.shape_inference
from onnx import AttributeProto, TensorProto, helper, numpy_helper


def float16_to_float32(model):
    """Retype an all-float16 graph to float32.

    Pulsar2 takes float32. An fp16 export has its constants in three places,
    and missing any one leaves a graph that mixes precisions: the initializer
    list, the `value` attribute of `Constant` nodes, and the `to` attribute of
    `Cast`. Converting only the initializers -- 214 of them in the Audio8
    codec decoder, against 1,174 `Constant` nodes -- produces a model ONNX
    Runtime rejects with "Type parameter (T) of Optype (Div) bound to
    different types (tensor(float) and tensor(float16))".
    """
    changed = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
            changed += 1
    for node in model.graph.node:
        for attr in node.attribute:
            if (
                node.op_type == "Cast"
                and attr.name == "to"
                and attr.i == TensorProto.FLOAT16
            ):
                attr.i = TensorProto.FLOAT
                changed += 1
            if (
                attr.type == AttributeProto.TENSOR
                and attr.t.data_type == TensorProto.FLOAT16
            ):
                arr = numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
                changed += 1
            if attr.type == AttributeProto.TENSORS:
                for t in attr.tensors:
                    if t.data_type == TensorProto.FLOAT16:
                        arr = numpy_helper.to_array(t).astype(np.float32)
                        t.CopyFrom(numpy_helper.from_array(arr, t.name))
                        changed += 1
    for value in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
            changed += 1
    return changed


def _scalar_constant(model, name):
    """The scalar value of `name` if it is a constant, else None."""
    for init in model.graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            return float(arr.reshape(-1)[0]) if arr.size == 1 else None
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    return float(arr.reshape(-1)[0]) if arr.size == 1 else None
    return None


def pow2_to_mul(model):
    """`Pow(x, 2)` becomes `Mul(x, x)`.

    Exact for floats, and one fewer transcendental op. It also stops Pulsar2
    matching the Snake activation `x + sin(alpha*x)**2 / alpha`, which it
    otherwise fuses into a native `AxQuantizedSnake` that then fails to build
    at every size tried: `NoTilerException` on a `(1,384,28160)` tensor, and
    `OpBuildException: broadcast dim 2: 32 1536 mismatch` on a small one. The
    unfused `Sin`/`Mul`/`Div`/`Add` are each on the supported list.
    """
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Pow" or len(node.input) != 2:
            continue
        if _scalar_constant(model, node.input[1]) != 2.0:
            continue
        base = node.input[0]
        del node.input[:]
        node.input.extend([base, base])
        node.op_type = "Mul"
        changed += 1
    return changed


def explicit_conv_padding(model):
    """A convolution with asymmetric padding gets an explicit `Pad` instead.

    The Audio8 vocoder's causal convolutions carry `pads=(54, 0)` -- all of it
    on the left -- alongside `dilation=9`, and Pulsar2's backend refuses the
    fused `AxQuantizedConv` for one. Hoisting the padding into a `Pad` node
    leaves the convolution with symmetric (zero) padding, which is the form
    every other convolution in the graph already has.

    Semantics are unchanged: zero-padding explicitly and then convolving with
    no padding is what the attribute means.
    """
    changed = 0
    nodes = list(model.graph.node)
    out = []
    for node in nodes:
        pads = None
        for attr in node.attribute:
            if attr.name == "pads":
                pads = list(attr.ints)
        if node.op_type != "Conv" or not pads or len(pads) % 2:
            out.append(node)
            continue
        half = len(pads) // 2
        begin, end = pads[:half], pads[half:]
        if begin == end:
            out.append(node)
            continue
        name = node.input[0] + f"_padded_{changed}"
        full = [0, 0] + begin + [0, 0] + end
        pads_init = numpy_helper.from_array(
            np.array(full, np.int64), node.name + "_pads"
        )
        model.graph.initializer.append(pads_init)
        out.append(
            helper.make_node(
                "Pad",
                [node.input[0], pads_init.name],
                [name],
                name=node.name + "_explicit_pad",
                mode="constant",
            )
        )
        node.input[0] = name
        for attr in node.attribute:
            if attr.name == "pads":
                del attr.ints[:]
                attr.ints.extend([0] * len(pads))
        out.append(node)
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def _initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def dilated_conv_to_taps(model, min_dilation=2):
    """A dilated 1-D convolution becomes one 1x1 convolution per tap, summed.

    `y[t] = sum_j w[:, :, j] . xp[t + j*d]` is the definition, so slicing the
    padded input at each tap offset and convolving with a kernel of one is
    exactly the same function -- with `dilation` gone and the padding consumed
    by an explicit `Pad` that no longer sits against a convolution, so the
    frontend's `Pad`-into-`Conv` fusion cannot put it back.

    This is also the shape the hardware wants: the weight table stores a widely
    dilated convolution as one block per tap already (see "A widely dilated
    convolution is K convolutions"), so the rewrite moves the graph towards
    what the compiler does internally rather than away from it.
    """
    # Shapes are needed to size each tap's slice, and a graph that was cut out
    # of a larger one carries no `value_info` at all -- which made an earlier
    # version of this rule skip every convolution in silence.
    try:
        shaped = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        known = {
            v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(shaped.graph.value_info) + list(shaped.graph.output)
        }
    except Exception:  # noqa: BLE001 -- shape inference is best-effort here
        known = {}

    changed = 0
    out = []
    for node in model.graph.node:
        attrs = {a.name: a for a in node.attribute}
        dil = list(attrs["dilations"].ints) if "dilations" in attrs else []
        weight = _initializer(model, node.input[1]) if len(node.input) > 1 else None
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1]
        group = attrs["group"].i if "group" in attrs else 1
        shape = known.get(node.output[0], [])
        if (
            node.op_type != "Conv"
            or weight is None
            or len(dil) != 1
            or dil[0] < min_dilation
            or strides != [1]
            or group != 1
            or len(shape) != 3
            or not shape[2]
        ):
            out.append(node)
            continue

        w = numpy_helper.to_array(weight)
        taps, d, length = w.shape[2], dil[0], shape[2]
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0]
        stem = node.name or node.output[0]

        pad_name = f"{stem}_pads"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array([0, 0, pads[0], 0, 0, pads[1]], np.int64), pad_name
            )
        )
        padded = f"{stem}_padded"
        out.append(
            helper.make_node(
                "Pad",
                [node.input[0], pad_name],
                [padded],
                name=f"{stem}_pad",
                mode="constant",
            )
        )

        partials = []
        for j in range(taps):
            tap_w = f"{stem}_w{j}"
            model.graph.initializer.append(
                numpy_helper.from_array(np.ascontiguousarray(w[:, :, j : j + 1]), tap_w)
            )
            starts, ends, axes = (f"{stem}_s{j}", f"{stem}_e{j}", f"{stem}_a{j}")
            for name, value in ((starts, j * d), (ends, j * d + length), (axes, 2)):
                model.graph.initializer.append(
                    numpy_helper.from_array(np.array([value], np.int64), name)
                )
            sliced = f"{stem}_x{j}"
            out.append(
                helper.make_node(
                    "Slice",
                    [padded, starts, ends, axes],
                    [sliced],
                    name=f"{stem}_slice{j}",
                )
            )
            inputs = [sliced, tap_w]
            if j == 0 and len(node.input) > 2:
                inputs.append(node.input[2])
            partial = f"{stem}_y{j}"
            out.append(
                helper.make_node(
                    "Conv",
                    inputs,
                    [partial],
                    name=f"{stem}_tap{j}",
                    kernel_shape=[1],
                    pads=[0, 0],
                    dilations=[1],
                    strides=[1],
                )
            )
            partials.append(partial)

        acc = partials[0]
        for j, part in enumerate(partials[1:], start=1):
            nxt = node.output[0] if j == len(partials) - 1 else f"{stem}_acc{j}"
            out.append(
                helper.make_node("Add", [acc, part], [nxt], name=f"{stem}_add{j}")
            )
            acc = nxt
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def filename_safe_io_names(model):
    """Graph inputs and outputs get names that can be a file name.

    `axcl_run_model` feeds a compiled model by writing one `<tensor name>.bin`
    per input, and Pulsar2 carries an ONNX name through to the `.axmodel`
    unchanged. Exporters routinely emit names like `/Add_10_output_0`, and a
    leading slash turns that path into an absolute one -- the runner then tries
    to write `/Add_10_output_0.bin` and fails with `PermissionError`. Renaming
    is safe: only the graph's own boundary names change, and nothing outside
    the model refers to them.
    """
    renamed = {}
    for value in list(model.graph.input) + list(model.graph.output):
        if "/" in value.name or value.name.startswith("."):
            clean = value.name.strip("/").replace("/", "_").lstrip(".")
            renamed[value.name] = clean or "tensor"
            value.name = renamed[value.name]
    if not renamed:
        return 0
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in renamed:
                node.input[i] = renamed[name]
        for i, name in enumerate(node.output):
            if name in renamed:
                node.output[i] = renamed[name]
    for value in model.graph.value_info:
        if value.name in renamed:
            value.name = renamed[value.name]
    return len(renamed)


#: Order matters. `dilated_conv_to_taps` consumes a convolution's `pads`
#: attribute, so it has to run before `explicit_conv_padding` zeroes it.
RULES = {
    "float16_to_float32": float16_to_float32,
    "pow2_to_mul": pow2_to_mul,
    "dilated_conv_to_taps": dilated_conv_to_taps,
    "explicit_conv_padding": explicit_conv_padding,
    "filename_safe_io_names": filename_safe_io_names,
}


def legalize(model, rules=None):
    """Apply the named rules in order; returns `{rule: sites changed}`."""
    applied = collections.OrderedDict()
    for name in rules or RULES:
        applied[name] = RULES[name](model)
    return applied


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--rules", nargs="*", choices=sorted(RULES))
    args = parser.parse_args(argv)

    model = onnx.load(args.input)
    for name, n in legalize(model, args.rules).items():
        print(f"  {name}: {n} sites")
    onnx.save(
        model,
        args.output,
        save_as_external_data=True,
        location=args.output.rsplit("/", 1)[-1] + ".data",
        size_threshold=1024,
    )
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
