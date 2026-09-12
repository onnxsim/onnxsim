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

    Also has a target-agnostic C++ counterpart in onnxsim's own core
    (`onnxsim/passes/float16_to_float32.h`), usable from any binding via
    `onnxsim.simplify(model, extra_optimizers=["float16_to_float32"])` --
    see `tests/test_float16_to_float32.py`. This module's own version stays:
    it needs nothing beyond the `onnx` package (no onnxsim build), which
    `legalize.py in.onnx out.onnx`'s standalone-script usage depends on.
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

    Also has a target-agnostic C++ counterpart in onnxsim's own core
    (`onnxsim/passes/pow2_to_mul.h`), usable from any binding via
    `onnxsim.simplify(model, extra_optimizers=["pow2_to_mul"])` -- see
    `tests/test_pow2_to_mul.py`. This module's own version stays: it needs
    nothing beyond the `onnx` package (no onnxsim build), which
    `legalize.py in.onnx out.onnx`'s standalone-script usage depends on. The
    core pass is scoped to `float32` and a constant single-element exponent
    (this Python version checks the exponent value the same way but doesn't
    restrict the element type) -- fine for this project's own use, where
    every Pulsar2-bound graph is float32 throughout (`float16_to_float32`
    runs first in `TRAINING_RULES`/`RULES` when it doesn't).
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

    This is not the same fix as `explicit_auto_pad`: that rule turns a
    *symbolic* `auto_pad` mode (`SAME_UPPER`/`SAME_LOWER`/`VALID`) into
    explicit `pads`; this rule starts from `pads` that are *already*
    explicit and only acts when they are asymmetric. A `Conv` needs at most
    one of the two -- `explicit_auto_pad` first if `auto_pad` is symbolic,
    then this rule if what it produces (or what the graph already had) is
    asymmetric.

    Also has a target-agnostic C++ counterpart in onnxsim's own core
    (`onnxsim/passes/explicit_conv_padding.h`), usable from any binding via
    `onnxsim.simplify(model, extra_optimizers=["explicit_conv_padding"])` --
    see `tests/test_explicit_conv_padding.py`. This module's own version
    stays: it needs nothing beyond the `onnx` package (no onnxsim build),
    which `legalize.py in.onnx out.onnx`'s standalone-script usage depends
    on.
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

    Also has a target-agnostic C++ counterpart in onnxsim's own core
    (`onnxsim/passes/dilated_conv_to_taps.h`), usable from any binding via
    `onnxsim.simplify(model, extra_optimizers=["dilated_conv_to_taps"])` --
    see `tests/test_dilated_conv_to_taps.py`. This module's own version
    stays: it needs nothing beyond the `onnx` package (no onnxsim build),
    which `legalize.py in.onnx out.onnx`'s standalone-script usage depends
    on. The core pass fixes `min_dilation` at 2 rather than exposing it as a
    parameter -- every call site in this project (and the vendor rule's own
    tests) uses the default.
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


def neg_to_mul(model):
    """`Neg(x)` becomes `Mul(x, -1)`.

    `Neg` is the one op `onnxsim.graph_grad` emits that is absent from
    `AX650_SUPPORTED_OPS` -- differentiating a subtraction produces exactly
    one of them, so every backward pass hits it. The rewrite is exact.

    Also has a target-agnostic C++ counterpart in onnxsim's own core
    (`onnxsim/passes/neg_to_mul.h`), usable from any binding via
    `onnxsim.simplify(model, extra_optimizers=["neg_to_mul"])` -- see
    `tests/test_neg_to_mul.py`. This module's own version stays: it needs
    nothing beyond the `onnx` package (no onnxsim build), which
    `legalize.py in.onnx out.onnx`'s standalone-script usage depends on.
    Both versions assume `float32` (this one emits a `float32` constant
    unconditionally; the core pass checks and declines other element types
    rather than emitting a mismatched one) -- fine for this project's own
    float32 training graphs, not a general-dtype guarantee either way.
    """
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Neg":
            continue
        name = _unique_name(model, f"{node.name or node.output[0]}_minus_one")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([-1.0], np.float32), name)
        )
        node.op_type = "Mul"
        node.input.append(name)
        changed += 1
    return changed


def _unique_name(model, stem):
    taken = (
        {i.name for i in model.graph.initializer}
        | {n.name for n in model.graph.node if n.name}
        | {o for n in model.graph.node for o in n.output}
    )
    name, k = stem, 0
    while name in taken:
        k += 1
        name = f"{stem}_{k}"
    return name


def rank0_to_rank1(model):
    """Graph outputs of rank 0 are given a trailing axis.

    Pulsar2's calibration concatenates each tensor across the calibration
    samples, and a rank-0 tensor cannot be concatenated -- the build dies with
    `RuntimeError: zero-dimensional tensor (at position 0) cannot be
    concatenated`. A scalar loss is the obvious way to hit this, and a
    training graph always has one.

    Only the declared rank changes; `ReduceMean`/`ReduceSum` grow a
    `keepdims=1` instead of reducing away, which is the same number in a
    1-element tensor.

    Also has a target-agnostic C++ counterpart in onnxsim's own core
    (`onnxsim/passes/rank0_to_rank1.h`), usable from any binding via
    `onnxsim.simplify(model, extra_optimizers=["rank0_to_rank1"])` -- see
    `tests/test_rank0_to_rank1.py`. This module's own version stays: it
    needs nothing beyond the `onnx` package (no onnxsim build), which
    `legalize.py in.onnx out.onnx`'s standalone-script usage depends on.
    """
    scalars = {
        v.name
        for v in model.graph.output
        if v.type.tensor_type.HasField("shape")
        and len(v.type.tensor_type.shape.dim) == 0
    }
    if not scalars:
        return 0
    shapes = _value_shapes(model)
    extra, changed = [], 0
    for node in model.graph.node:
        if not node.output or node.output[0] not in scalars:
            continue
        name = node.output[0]
        if node.op_type in (
            "ReduceMean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
        ):
            for attr in node.attribute:
                if attr.name == "keepdims":
                    attr.i = 1
                    break
            else:
                node.attribute.append(helper.make_attribute("keepdims", 1))
            # ONNX reduces *every* axis when none is named. Pulsar2 does not:
            # a `ReduceMean` over (1, 16, 32) with no `axes` came back shaped
            # (1, 16, 1), which is the last axis alone. Name them, so the
            # graph says what it means to both.
            named = any(a.name == "axes" for a in node.attribute) or (
                len(node.input) > 1 and node.input[1]
            )
            rank = len(shapes.get(node.input[0], ()))
            if not named and rank:
                _set_axes(model, node, range(rank))
        # `keepdims=1` keeps *every* reduced axis, so the rank depends on the
        # input's. Reshaping to [1] is exact whatever that turns out to be,
        # and it is one more NPU-legal op.
        inner = _unique_name(model, f"{name}_kept")
        node.output[0] = inner
        shape_name = _unique_name(model, f"{name}_shape1")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([1], np.int64), shape_name)
        )
        extra.append(
            (
                node,
                helper.make_node(
                    "Reshape",
                    [inner, shape_name],
                    [name],
                    name=_unique_name(model, f"{name}_R1"),
                ),
            )
        )
        changed += 1
    if extra:
        out = []
        after = {id(producer): reshape for producer, reshape in extra}
        for node in model.graph.node:
            out.append(node)
            if id(node) in after:
                out.append(after[id(node)])
        del model.graph.node[:]
        model.graph.node.extend(out)
    for value in model.graph.output:
        if value.name in scalars:
            value.type.tensor_type.shape.dim.add().dim_value = 1
    return changed


def _is_initializer(model, name):
    return any(i.name == name for i in model.graph.initializer)


def _opset(model, domain=""):
    for entry in model.opset_import:
        if (entry.domain or "") in (domain, "ai.onnx" if domain == "" else domain):
            return entry.version
    return 0


def _set_axes(model, node, axes):
    """Name a reduction's axes, as an attribute or an input by opset.

    `axes` moved from attribute to input at opset 18, and `onnx.checker`
    rejects the wrong one. Both forms matter here: resnet18d is opset 18,
    the training graphs built by hand are opset 17.
    """
    node.attribute.extend([a for a in ()])
    if _opset(model) >= 18:
        name = _unique_name(model, f"{node.name or node.output[0]}_axes")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(list(axes), np.int64), name)
        )
        while len(node.input) < 2:
            node.input.append("")
        node.input[1] = name
    else:
        node.attribute.append(helper.make_attribute("axes", list(axes)))


def inline_local_functions(model):
    """Expand the graph's own `FunctionProto` calls into ordinary nodes.

    `onnxsim.graph_grad`'s templated rules emit calls to checked-in local
    functions -- `GradAdd` and friends -- rather than open-coding them. Pulsar2
    parses op *types*, and a call to a locally-defined function is not one:
    eight `GradAdd` nodes are the only thing off the AX650 list in a legalized
    resnet18 training step, and they are eight residual connections' gradient
    accumulations, so they are not optional.

    `onnx.inliner` expands them into what they always were -- here, `Identity`,
    which the AX650 does support. Returns the number of function definitions
    that were inlined away.
    """
    if not model.functions:
        return 0
    import onnx.inliner

    # A locally-defined function lives in its own domain, and the model has to
    # import that domain before anything -- checker or inliner -- will look at
    # it. A graph assembled by hand from `graph_grad`'s output does not have
    # the import, so add it rather than failing on "No opset import for domain".
    have = {entry.domain: entry for entry in model.opset_import}
    for fn in model.functions:
        if fn.domain not in have:
            entry = helper.make_opsetid(fn.domain, 1)
            model.opset_import.append(entry)
            have[fn.domain] = entry
        # The inliner silently declines a function whose standard-domain opset
        # disagrees with the model's -- it inlined nothing and left eight
        # GradAdd calls for Pulsar2 to reject with "dont support GradAdd opr".
        # The functions `graph_grad` ships declare opset 17; a graph built from
        # a modern export declares 18.
        for imp in fn.opset_import:
            standard = have.get(imp.domain or "")
            if standard is not None and (imp.domain or "") == "":
                imp.version = standard.version
    before = len(model.functions)
    inlined = onnx.inliner.inline_local_functions(model)
    model.CopyFrom(inlined)
    return before - len(model.functions)


def avgpool_ceil_to_floor(model):
    """Clear `ceil_mode` on a pool where it changes nothing.

    `onnxsim.graph_grad` declines `AveragePool`/`MaxPool` with `ceil_mode=1`
    rather than approximating its ragged edge window -- which stops resnet18d
    dead, because its downsample pools carry the flag. On an input the stride
    divides evenly, ceil and floor agree exactly, so the flag is decoration
    and dropping it is a no-op. Where they genuinely differ the node is left
    alone and `graph_grad` still refuses, which is the honest outcome.
    """
    shapes = _value_shapes(model)
    changed = 0
    for node in model.graph.node:
        if node.op_type not in ("AveragePool", "MaxPool"):
            continue
        attrs = {a.name: a for a in node.attribute}
        if not (attrs.get("ceil_mode") and attrs["ceil_mode"].i):
            continue
        shape = shapes.get(node.input[0])
        if shape is None or "kernel_shape" not in attrs:
            continue
        nd = len(attrs["kernel_shape"].ints)
        kernel = list(attrs["kernel_shape"].ints)
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1] * nd
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0] * (2 * nd)
        same = True
        for j in range(nd):
            span = shape[2 + j] + pads[j] + pads[j + nd] - kernel[j]
            if span % strides[j]:
                same = False  # ceil would keep a ragged final window
                break
        if not same:
            continue
        attrs["ceil_mode"].i = 0
        changed += 1
    return changed


def flatten_to_reshape(model):
    """`Flatten` becomes `Reshape`.

    `onnxsim.graph_grad` has no rule for `Flatten` and does not need one: it
    is a `Reshape`, which does have one. resnet18 has exactly one, between the
    pooling and the classifier, and without this the whole graph is
    undifferentiable for the sake of a no-op.
    """
    shapes = _value_shapes(model)
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Flatten":
            continue
        shape = shapes.get(node.input[0])
        if shape is None:
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        axis = axis if axis >= 0 else len(shape) + axis
        head = int(np.prod(shape[:axis])) if axis else 1
        name = _unique_name(model, f"{node.name or node.output[0]}_shape")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([head, -1], np.int64), name)
        )
        node.op_type = "Reshape"
        del node.attribute[:]
        node.input.append(name)
        changed += 1
    return changed


def global_pool_to_reduce(model):
    """`GlobalAveragePool` becomes `ReduceMean` over the spatial axes.

    Same reason as `flatten_to_reshape`: no gradient rule for the global form,
    a perfectly good one for the reduction it is. The axes are named rather
    than defaulted, because this hardware reduces only the last axis when a
    reduction names none (see `rank0_to_rank1`).
    """
    shapes = _value_shapes(model)
    changed = 0
    for node in model.graph.node:
        if node.op_type != "GlobalAveragePool":
            continue
        shape = shapes.get(node.input[0])
        if shape is None or len(shape) < 3:
            continue
        node.op_type = "ReduceMean"
        del node.attribute[:]
        node.attribute.append(helper.make_attribute("keepdims", 1))
        _set_axes(model, node, range(2, len(shape)))
        changed += 1
    return changed


def _unfusable_bias(model, name, width):
    """A copy of bias `name` shaped `[1, width]` instead of `[width]`.

    `MatMul(live) + Add(constant 1-D)` is exactly what
    `fuse_matmul_add_bias_into_gemm` matches, and skipping that pass in
    onnxsim is not enough -- **Pulsar2 runs its own optimizer and fuses it
    back**, into a `Gemm` whose weight is live, which it then cannot lower.
    It does not say so: the build dies inside PPQ's calibrator with
    `ValueError('The truth value of an array with more than one element is
    ambiguous')`, naming no node.

    Measured on the four-node repro this was bisected down to: a 1-D constant
    bias fails, the same values shaped `[1, N]` build, and so does an
    `Identity` wedged between the `MatMul` and the `Add`. The reshape is the
    better of the two -- it adds no node, and an `Identity` is exactly what a
    later dead-code pass would remove, putting the bug back.
    """
    for init in model.graph.initializer:
        if init.name != name:
            continue
        values = numpy_helper.to_array(init)
        if values.ndim != 1:
            return name
        wide = _unique_name(model, f"{name}_rank2")
        model.graph.initializer.append(
            numpy_helper.from_array(values.reshape(1, -1), wide)
        )
        return wide
    return name


def gemm_to_matmul(model):
    """A `Gemm` whose `B` is a live tensor becomes `MatMul` (plus what it drops).

    Pulsar2 asks for this by name: a `Gemm` with two non-parameter inputs
    fails the build with `NotImplementedError('Should fuse Gemm (two
    non-parameter inputs) to MatMul.')`. A training graph produces them
    wherever a fully-connected layer's weight is being learned rather than
    baked in.

    `transA`/`transB` become `Transpose`, `alpha`/`beta` become `Mul`, and `C`
    becomes `Add` -- all NPU-legal, and all no-ops when left at their defaults.
    """
    out, changed = [], 0
    for node in model.graph.node:
        if (
            node.op_type != "Gemm"
            or len(node.input) < 2
            or _is_initializer(model, node.input[1])
        ):
            out.append(node)
            continue
        attrs = {a.name: a for a in node.attribute}
        alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
        beta = attrs["beta"].f if "beta" in attrs else 1.0
        stem = node.name or node.output[0]
        a, b = node.input[0], node.input[1]
        made = []
        if "transA" in attrs and attrs["transA"].i:
            t = _unique_name(model, f"{stem}_at")
            made.append(
                helper.make_node(
                    "Transpose",
                    [a],
                    [t],
                    perm=[1, 0],
                    name=_unique_name(model, f"{stem}_TA"),
                )
            )
            a = t
        if "transB" in attrs and attrs["transB"].i:
            t = _unique_name(model, f"{stem}_bt")
            made.append(
                helper.make_node(
                    "Transpose",
                    [b],
                    [t],
                    perm=[1, 0],
                    name=_unique_name(model, f"{stem}_TB"),
                )
            )
            b = t
        tail = node.output[0]
        cur = (
            tail
            if (alpha == 1.0 and len(node.input) < 3)
            else _unique_name(model, f"{stem}_mm")
        )
        made.append(
            helper.make_node(
                "MatMul", [a, b], [cur], name=_unique_name(model, f"{stem}_MM")
            )
        )
        if alpha != 1.0:
            k = _unique_name(model, f"{stem}_alpha")
            model.graph.initializer.append(
                numpy_helper.from_array(np.array([alpha], np.float32), k)
            )
            nxt = tail if len(node.input) < 3 else _unique_name(model, f"{stem}_sc")
            made.append(
                helper.make_node(
                    "Mul", [cur, k], [nxt], name=_unique_name(model, f"{stem}_A")
                )
            )
            cur = nxt
        if len(node.input) >= 3:
            c = _unfusable_bias(model, node.input[2], 0)
            if beta != 1.0:
                k = _unique_name(model, f"{stem}_beta")
                model.graph.initializer.append(
                    numpy_helper.from_array(np.array([beta], np.float32), k)
                )
                scaled = _unique_name(model, f"{stem}_cb")
                made.append(
                    helper.make_node(
                        "Mul", [c, k], [scaled], name=_unique_name(model, f"{stem}_B")
                    )
                )
                c = scaled
            made.append(
                helper.make_node(
                    "Add", [cur, c], [tail], name=_unique_name(model, f"{stem}_C")
                )
            )
        out.extend(made)
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def act_weight_conv_to_matmul(model, fuse=True):
    """A `Conv` whose weight is a live tensor becomes one `MatMul` per tap.

    Pulsar2 has an op for a runtime weight -- the error names
    `AxQuantizedActWeightConv` -- but it fails its own shape function with the
    weight still `FP32` while the activation is already `U8`, in 1-D, in 1x1
    and with `data_type: U8` forced; the 2-D case gets past the frontend and
    dies in the backend instead. So the convolution has to be spelled as
    matrix multiplies, which is the shape the compiler asks for elsewhere --
    "Should fuse Gemm (two non-parameter inputs) to MatMul".

    ``y[n,o,p] = sum_{i,k} w[o,i,k] * xpad[n,i,p*stride+k]``, so with the
    activation transposed to `[N, spatial..., Cin]` each tap is one `MatMul`
    against `w[..., k]` reshaped to `[Cin, Cout]`, and the taps are added.
    Stride becomes the tap slice's `step`. Handles 1-D and 2-D, any stride,
    dilation 1, any `group`; anything else is declined rather than
    approximated.

    `group > 1` (a depthwise or otherwise grouped Conv, e.g. the gated conv
    module in a Conformer-style audio-model block) runs the exact same
    per-tap matmul once per group, against that group's own channel slice of
    the (already-transposed-once) activation and weight, and concatenates
    the per-group outputs back onto the `Cout` axis before the shared bias
    add -- ONNX's own grouping semantics, since each output channel only
    ever contracts against its own group's input channels. `group == 1`
    (every model this rule has run on before grouped support was added,
    including resnet18) takes the untouched original path: no extra
    Slice/Concat, byte-identical node sequence to before.
    """
    shapes = _value_shapes(model)
    out, changed = [], 0
    for node in model.graph.node:
        w = node.input[1] if len(node.input) > 1 else None
        if node.op_type != "Conv" or w is None or _is_initializer(model, w):
            out.append(node)
            continue
        wshape, xshape = shapes.get(w), shapes.get(node.input[0])
        yshape = shapes.get(node.output[0])
        attrs = {a.name: a for a in node.attribute}
        if not wshape or not xshape or len(wshape) != len(xshape):
            out.append(node)
            continue
        nd = len(wshape) - 2
        if nd not in (1, 2):
            out.append(node)
            continue
        dil = list(attrs["dilations"].ints) if "dilations" in attrs else [1] * nd
        if any(d != 1 for d in dil):
            out.append(node)
            continue
        group = attrs["group"].i if "group" in attrs else 1
        if group == 0:  # ONNX default; a handful of exporters emit this literally
            group = 1
        cout, cin = wshape[0], wshape[1]
        if (
            group < 1
            or xshape[1] % group != 0
            or cout % group != 0
            or xshape[1] // group != cin
        ):
            out.append(node)  # W's own channel count disagrees with group; do not guess
            continue
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1] * nd
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0] * (2 * nd)
        ksize = wshape[2:]
        spatial = xshape[2:]
        outdim = [
            (spatial[j] + pads[j] + pads[j + nd] - ksize[j]) // strides[j] + 1
            for j in range(nd)
        ]
        if yshape and list(yshape[2:]) != outdim:
            out.append(node)  # our geometry disagrees; do not guess
            continue
        stem = node.name or node.output[0]
        made = []

        src = node.input[0]
        if any(pads):
            padded = _unique_name(model, f"{stem}_pad")
            pname = _unique_name(model, f"{stem}_pads")
            begins = [0, 0] + pads[:nd]
            ends = [0, 0] + pads[nd:]
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(begins + ends, np.int64), pname)
            )
            made.append(
                helper.make_node(
                    "Pad", [src, pname], [padded], name=_unique_name(model, f"{stem}_P")
                )
            )
            src = padded
        # [N, C, spatial...] -> [N, spatial..., C]
        xt = _unique_name(model, f"{stem}_xt")
        made.append(
            helper.make_node(
                "Transpose",
                [src],
                [xt],
                perm=[0] + list(range(2, nd + 2)) + [1],
                name=_unique_name(model, f"{stem}_XT"),
            )
        )
        # [Cout, Cin, k...] -> [k..., Cin, Cout]
        wt = _unique_name(model, f"{stem}_wt")
        made.append(
            helper.make_node(
                "Transpose",
                [w],
                [wt],
                perm=list(range(2, nd + 2)) + [1, 0],
                name=_unique_name(model, f"{stem}_WT"),
            )
        )

        taps = [()]
        for j in range(nd):
            taps = [t + (k,) for t in taps for k in range(ksize[j])]

        def taps_matmul(x_src, w_src, cout_g, label_prefix):
            """One group's `sum_tap X_tap @ W_tap` over `x_src`/`w_src`
            (already sliced to this group's channels, `cin` wide on `x_src`'s
            last axis and `w_src`'s second-to-last), producing `[N,
            spatial..., cout_g]`. Identical to the whole (group=1) computation
            below, just parameterized so `group > 1` can call it once per
            group over group-sliced inputs -- see the call sites below."""
            wshape_name = _unique_name(model, f"{label_prefix}_wshape")
            model.graph.initializer.append(
                numpy_helper.from_array(np.array([cin, cout_g], np.int64), wshape_name)
            )
            acc = None
            tap_x, tap_w = [], []
            for tap in taps:
                label = "_".join(str(k) for k in tap)
                cur_x, cur_w = x_src, w_src
                for j, k in enumerate(tap):
                    nxt = _unique_name(model, f"{label_prefix}_x{label}_{j}")
                    made.append(
                        _slice(
                            model,
                            cur_x,
                            nxt,
                            1 + j,
                            k,
                            k + (outdim[j] - 1) * strides[j] + 1,
                            f"{label_prefix}_SX{label}_{j}",
                            step=strides[j],
                        )
                    )
                    cur_x = nxt
                    nxw = _unique_name(model, f"{label_prefix}_w{label}_{j}")
                    made.append(
                        _slice(
                            model,
                            cur_w,
                            nxw,
                            j,
                            k,
                            k + 1,
                            f"{label_prefix}_SW{label}_{j}",
                        )
                    )
                    cur_w = nxw
                flat = _unique_name(model, f"{label_prefix}_wf{label}")
                made.append(
                    helper.make_node(
                        "Reshape",
                        [cur_w, wshape_name],
                        [flat],
                        name=_unique_name(model, f"{label_prefix}_WR{label}"),
                    )
                )
                tap_x.append(cur_x)
                tap_w.append(flat)

            # sum_k X_k @ W_k is one matmul over the concatenation:
            #   [X_0 | ... | X_{K-1}] @ [W_0 ; ... ; W_{K-1}]
            # exactly, because the taps share an output and differ only along
            # the reduction axis. For a 3x3 that is 9 MatMuls and 8 Adds
            # replaced by two Concats and one MatMul nine times deeper --
            # fewer nodes to schedule, and an arithmetic intensity the matrix
            # unit can actually use. `fuse` exists so the unfused form stays
            # reachable for bisecting a compiler that dislikes one of them.
            if fuse and len(tap_x) > 1:
                xcat = _unique_name(model, f"{label_prefix}_xcat")
                wcat = _unique_name(model, f"{label_prefix}_wcat")
                made.append(
                    helper.make_node(
                        "Concat",
                        tap_x,
                        [xcat],
                        axis=-1,
                        name=_unique_name(model, f"{label_prefix}_XC"),
                    )
                )
                made.append(
                    helper.make_node(
                        "Concat",
                        tap_w,
                        [wcat],
                        axis=0,
                        name=_unique_name(model, f"{label_prefix}_WC"),
                    )
                )
                acc = _unique_name(model, f"{label_prefix}_mm")
                made.append(
                    helper.make_node(
                        "MatMul",
                        [xcat, wcat],
                        [acc],
                        name=_unique_name(model, f"{label_prefix}_MM"),
                    )
                )
            else:
                for i, (xk, wk) in enumerate(zip(tap_x, tap_w)):
                    prod = _unique_name(model, f"{label_prefix}_m{i}")
                    made.append(
                        helper.make_node(
                            "MatMul",
                            [xk, wk],
                            [prod],
                            name=_unique_name(model, f"{label_prefix}_MM{i}"),
                        )
                    )
                    if acc is None:
                        acc = prod
                    else:
                        nxt = _unique_name(model, f"{label_prefix}_a{i}")
                        made.append(
                            helper.make_node(
                                "Add",
                                [acc, prod],
                                [nxt],
                                name=_unique_name(model, f"{label_prefix}_AD{i}"),
                            )
                        )
                        acc = nxt
            return acc

        if group == 1:
            # Exactly the pre-group-support node sequence -- no extra
            # Slice/Concat for the common case, so this path (resnet18's, and
            # every model without a grouped Conv) is untouched.
            acc = taps_matmul(xt, wt, cout, stem)
        else:
            # `x`'s channels split into `group` equal ranges on `xt`'s last
            # axis; `w`'s *output* channels (the only axis `wt` has not
            # already reduced to one group's worth -- its Cin axis is a
            # group's `cin` by construction, since ONNX's Conv spec makes W's
            # second dimension `Cin/group` regardless of `group`) split the
            # same way on `wt`'s last axis. Each group only ever contracts
            # against its own `cin` input channels, matching ONNX Conv's own
            # grouping semantics; the per-group outputs are independent and
            # simply concatenate back into the full `Cout` axis, which is
            # exactly what a depthwise/grouped Conv computes.
            cout_g = cout // group
            group_accs = []
            for g in range(group):
                gstem = f"{stem}_g{g}"
                xt_g = _unique_name(model, f"{gstem}_xg")
                made.append(
                    _slice(
                        model, xt, xt_g, nd + 1, g * cin, (g + 1) * cin, f"{gstem}_SXG"
                    )
                )
                wt_g = _unique_name(model, f"{gstem}_wg")
                made.append(
                    _slice(
                        model,
                        wt,
                        wt_g,
                        nd + 1,
                        g * cout_g,
                        (g + 1) * cout_g,
                        f"{gstem}_SWG",
                    )
                )
                group_accs.append(taps_matmul(xt_g, wt_g, cout_g, gstem))
            acc = _unique_name(model, f"{stem}_gcat")
            made.append(
                helper.make_node(
                    "Concat",
                    group_accs,
                    [acc],
                    axis=-1,
                    name=_unique_name(model, f"{stem}_GC"),
                )
            )
        # A bias is [Cout], which broadcasts onto the trailing axis exactly
        # where the taps land -- before the output is transposed back. The
        # 1x1 downsample convolutions in resnet18 carry one, and skipping
        # them left a live-weight Conv for Pulsar2 to reject with
        # "Hardware Op spec error ActWeightConv ... list index out of range".
        if len(node.input) > 2 and node.input[2]:
            biased = _unique_name(model, f"{stem}_biased")
            made.append(
                helper.make_node(
                    "Add",
                    [acc, _unfusable_bias(model, node.input[2], cout)],
                    [biased],
                    name=_unique_name(model, f"{stem}_B"),
                )
            )
            acc = biased
        # [N, spatial..., Cout] -> [N, Cout, spatial...]
        made.append(
            helper.make_node(
                "Transpose",
                [acc],
                [node.output[0]],
                perm=[0, nd + 1] + list(range(1, nd + 1)),
                name=_unique_name(model, f"{stem}_YT"),
            )
        )
        out.extend(made)
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def _slice(model, src, dst, axis, start, end, stem, step=1):
    names = []
    fields = [("starts", start), ("ends", end), ("axes", axis)]
    if step != 1:
        fields.append(("steps", step))
    for label, value in fields:
        name = _unique_name(model, f"{stem}_{label}")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([value], np.int64), name)
        )
        names.append(name)
    return helper.make_node(
        "Slice", [src] + names, [dst], name=_unique_name(model, stem)
    )


def _value_shapes(model):
    inferred = model
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    shapes = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        dims = value.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") for d in dims):
            shapes[value.name] = [d.dim_value for d in dims]
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def _const_i64(model, stem, value):
    name = _unique_name(model, stem)
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(value, dtype=np.int64), name)
    )
    return name


def _gate_weight(model, stem, arr):
    """Stores a precomputed, already-transposed 2-D gate weight (or bias) as
    a new initializer, for `MatMul(x, w)` (not `MatMul(x, w.T)`) directly."""
    name = _unique_name(model, stem)
    model.graph.initializer.append(
        numpy_helper.from_array(np.ascontiguousarray(arr), name)
    )
    return name


def _unroll_recurrent_node(model, node, shapes, n_gates, step_fn):
    """Shared scaffolding for `unroll_lstm`/`unroll_gru`: validates the
    node is in the supported shape (forward direction, static shapes,
    initializer weights, no peephole/variable-length inputs), splits `W`/
    `R`/`B` into per-gate 2-D matrices, builds the `Gather`-per-timestep
    loop, and wires up whichever of the node's own outputs are actually
    used. `step_fn(gates, h, c, t_nodes)` computes one timestep's `(h, c)`
    from that step's per-gate `x @ Wg + h @ Rg + bias_g` pre-activations
    (`c` is `None` for GRU) and appends any nodes it creates to `t_nodes`;
    returns `None` for a shape this rule declines to handle (left
    untouched, the same precedent every other shape-scoped rule here
    follows -- e.g. `act_weight_conv_to_matmul`'s `group=1, dilation=1`).
    """
    attrs = {a.name: a for a in node.attribute}
    direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
    layout = attrs["layout"].i if "layout" in attrs else 0
    if direction != "forward" or layout != 0:
        return None  # bidirectional / reverse / batch-first: not seen, declined

    x_name = node.input[0]
    w_init = _initializer(model, node.input[1]) if len(node.input) > 1 else None
    r_init = _initializer(model, node.input[2]) if len(node.input) > 2 else None
    if w_init is None or r_init is None:
        return None  # a live/trainable W or R: no target model needs this yet

    b_init = None
    if len(node.input) > 3 and node.input[3]:
        b_init = _initializer(model, node.input[3])
        if b_init is None:
            return None  # a live/trainable bias: same, not a seen pattern
    if len(node.input) > 4 and node.input[4]:
        return None  # sequence_lens: variable-length batches, declined
    if len(node.input) > 4 + n_gates:  # LSTM's peephole `P`, GRU has no 8th input
        return None

    x_shape = shapes.get(x_name)
    if not x_shape or len(x_shape) != 3:
        return None
    seq_len, batch, _ = x_shape
    hidden_size = attrs["hidden_size"].i if "hidden_size" in attrs else None
    w = numpy_helper.to_array(w_init)[0]  # [n_gates*H, input_size]
    r = numpy_helper.to_array(r_init)[0]  # [n_gates*H, H]
    if hidden_size is None:
        hidden_size = w.shape[0] // n_gates
    h = hidden_size

    def gate(mat, i):
        return np.ascontiguousarray(mat[i * h : (i + 1) * h].T)  # already x@w form

    w_gates = [gate(w, i) for i in range(n_gates)]
    r_gates = [gate(r, i) for i in range(n_gates)]
    if b_init is not None:
        b = numpy_helper.to_array(b_init)[0]  # [2*n_gates*H] = Wb..., Rb...
        wb_gates = [b[i * h : (i + 1) * h] for i in range(n_gates)]
        rb_gates = [
            b[(n_gates + i) * h : (n_gates + i + 1) * h] for i in range(n_gates)
        ]
    else:
        wb_gates = rb_gates = [np.zeros(h, dtype=np.float32)] * n_gates

    stem = node.name or node.output[0]
    new_nodes = []

    def initial_state(idx, label):
        if len(node.input) > idx and node.input[idx]:
            squeezed = _unique_name(model, f"{stem}_{label}0")
            new_nodes.append(
                helper.make_node(
                    "Reshape",
                    [
                        node.input[idx],
                        _const_i64(model, f"{stem}_{label}0_shape", [batch, h]),
                    ],
                    [squeezed],
                    name=squeezed,
                )
            )
            return squeezed
        zero_name = _unique_name(model, f"{stem}_{label}0_zero")
        model.graph.initializer.append(
            numpy_helper.from_array(np.zeros((batch, h), dtype=np.float32), zero_name)
        )
        return zero_name

    h_state = initial_state(5, "h")
    c_state = initial_state(6, "c") if n_gates == 4 else None

    w_names = [_gate_weight(model, f"{stem}_w{i}", w_gates[i]) for i in range(n_gates)]
    r_names = [_gate_weight(model, f"{stem}_r{i}", r_gates[i]) for i in range(n_gates)]
    wb_names = [
        _gate_weight(model, f"{stem}_wb{i}", wb_gates[i]) for i in range(n_gates)
    ]
    rb_names = [
        _gate_weight(model, f"{stem}_rb{i}", rb_gates[i]) for i in range(n_gates)
    ]

    per_step_h = []
    for t in range(seq_len):
        t_idx = _const_i64(model, f"{stem}_t{t}", t)
        xt = _unique_name(model, f"{stem}_x{t}")
        new_nodes.append(
            helper.make_node("Gather", [x_name, t_idx], [xt], name=xt, axis=0)
        )
        gates = []
        for i in range(n_gates):
            xw = _unique_name(model, f"{stem}_xw{t}_{i}")
            hr = _unique_name(model, f"{stem}_hr{t}_{i}")
            pre = _unique_name(model, f"{stem}_pre{t}_{i}")
            new_nodes.append(
                helper.make_node("MatMul", [xt, w_names[i]], [xw], name=xw)
            )
            new_nodes.append(
                helper.make_node("MatMul", [h_state, r_names[i]], [hr], name=hr)
            )
            gates.append((xw, hr, wb_names[i], rb_names[i], pre))
        h_state, c_state = step_fn(model, stem, t, gates, h_state, c_state, new_nodes)
        per_step_h.append(h_state)

    outs = list(node.output) + [""] * (3 - len(node.output))
    if outs[0]:
        unsq = []
        for t, hv in enumerate(per_step_h):
            u = _unique_name(model, f"{stem}_yu{t}")
            new_nodes.append(
                helper.make_node(
                    "Reshape",
                    [hv, _const_i64(model, f"{stem}_yu{t}_shape", [1, 1, batch, h])],
                    [u],
                    name=u,
                )
            )
            unsq.append(u)
        new_nodes.append(
            helper.make_node("Concat", unsq, [outs[0]], name=f"{stem}_y", axis=0)
        )
    if outs[1]:
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [h_state, _const_i64(model, f"{stem}_yh_shape", [1, batch, h])],
                [outs[1]],
                name=f"{stem}_yh",
            )
        )
    if n_gates == 4 and outs[2]:
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [c_state, _const_i64(model, f"{stem}_yc_shape", [1, batch, h])],
                [outs[2]],
                name=f"{stem}_yc",
            )
        )

    # W/R/B's own values are already extracted into per-gate initializers
    # above; the original packed tensors are now unused (split, not
    # referenced by any remaining node).
    orphaned = {w_init.name, r_init.name} | (
        {b_init.name} if b_init is not None else set()
    )
    kept = [init for init in model.graph.initializer if init.name not in orphaned]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    return new_nodes


def _lstm_step(model, stem, t, gates, h, c, new_nodes):
    """ONNX `LSTM` semantics, gate order `i, o, f, c` (verified directly
    against real `torch.onnx.export`-produced `LSTM` nodes)."""

    def activate(op, idx):
        xw, hr, wb, rb, pre = gates[idx]
        s = _unique_name(model, f"{pre}_sum")
        new_nodes.append(helper.make_node("Add", [xw, hr], [s], name=s))
        s2 = _unique_name(model, f"{pre}_sum2")
        new_nodes.append(helper.make_node("Add", [s, wb], [s2], name=s2))
        s3 = _unique_name(model, f"{pre}_sum3")
        new_nodes.append(helper.make_node("Add", [s2, rb], [s3], name=s3))
        new_nodes.append(helper.make_node(op, [s3], [pre], name=pre))
        return pre

    it = activate("Sigmoid", 0)
    ot = activate("Sigmoid", 1)
    ft = activate("Sigmoid", 2)
    ct_tilde = activate("Tanh", 3)

    fc = _unique_name(model, f"{stem}_t{t}_fc")
    new_nodes.append(helper.make_node("Mul", [ft, c], [fc], name=fc))
    ic = _unique_name(model, f"{stem}_t{t}_ic")
    new_nodes.append(helper.make_node("Mul", [it, ct_tilde], [ic], name=ic))
    c_new = _unique_name(model, f"{stem}_t{t}_c")
    new_nodes.append(helper.make_node("Add", [fc, ic], [c_new], name=c_new))
    tanh_c = _unique_name(model, f"{stem}_t{t}_tanhc")
    new_nodes.append(helper.make_node("Tanh", [c_new], [tanh_c], name=tanh_c))
    h_new = _unique_name(model, f"{stem}_t{t}_h")
    new_nodes.append(helper.make_node("Mul", [ot, tanh_c], [h_new], name=h_new))
    return h_new, c_new


def unroll_lstm(model):
    """Replaces a forward, single-direction `LSTM` node with its own
    per-timestep gate arithmetic -- `MatMul`/`Add`/`Sigmoid`/`Tanh`/`Mul`,
    every one of which already has both a `graph_grad` backward rule and
    NPU support (`AX650_SUPPORTED_OPS`), unlike the opaque `LSTM` op itself
    (NPU-executable but with no backward rule at all --
    `docs/axera-audio-speech-op-coverage.md`'s LSTM row).

    Requires: static `[seq_length, batch, input_size]` input shape (shapes
    are inferred internally, the same `dilated_conv_to_taps` precedent),
    forward direction only, `W`/`R`/`B` as initializers, no peephole (`P`)
    input, no `sequence_lens` input. A node outside this scope is left
    untouched.

    Numerically exact to float32 precision: verified directly against real
    `torch.onnx.export`-produced `LSTM` nodes (gate order `i, o, f, c`,
    bias layout `[Wb(i,o,f,c), Rb(i,o,f,c)]`), not derived from the ONNX
    spec text alone.
    """
    shapes = _value_shapes(model)
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type != "LSTM":
            out.append(node)
            continue
        replacement = _unroll_recurrent_node(model, node, shapes, 4, _lstm_step)
        if replacement is None:
            out.append(node)
            continue
        out.extend(replacement)
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def _gru_step(model, stem, t, gates, h, _c, new_nodes):
    """ONNX `GRU` semantics with `linear_before_reset=1` (what real
    `torch.onnx.export` always emits for `nn.GRU`), gate order `z, r, h` --
    verified directly against a real export, not the spec text alone."""
    zxw, zhr, zwb, zrb, zpre = gates[0]
    rxw, rhr, rwb, rrb, rpre = gates[1]
    nxw, nhr, nwb, nrb, npre = gates[2]

    def gate_sum(op, xw, hr, wb, rb, pre):
        s = _unique_name(model, f"{pre}_sum")
        new_nodes.append(helper.make_node("Add", [xw, hr], [s], name=s))
        s2 = _unique_name(model, f"{pre}_sum2")
        new_nodes.append(helper.make_node("Add", [s, wb], [s2], name=s2))
        s3 = _unique_name(model, f"{pre}_sum3")
        new_nodes.append(helper.make_node("Add", [s2, rb], [s3], name=s3))
        new_nodes.append(helper.make_node(op, [s3], [pre], name=pre))
        return pre

    zt = gate_sum("Sigmoid", zxw, zhr, zwb, zrb, zpre)
    rt = gate_sum("Sigmoid", rxw, rhr, rwb, rrb, rpre)

    # linear_before_reset=1: the reset gate scales (h @ Rh + Rbh) as a
    # whole, added to the already-computed Wh/x term -- not `(rt*h) @ Rh`.
    rc = _unique_name(model, f"{stem}_t{t}_rc")
    new_nodes.append(helper.make_node("Add", [nhr, nrb], [rc], name=rc))
    nxw_full = _unique_name(model, f"{stem}_t{t}_nxwfull")
    new_nodes.append(helper.make_node("Add", [nxw, nwb], [nxw_full], name=nxw_full))
    rn = _unique_name(model, f"{stem}_t{t}_rn")
    new_nodes.append(helper.make_node("Mul", [rt, rc], [rn], name=rn))
    n_pre = _unique_name(model, f"{stem}_t{t}_npre")
    new_nodes.append(helper.make_node("Add", [nxw_full, rn], [n_pre], name=n_pre))
    nt = npre
    new_nodes.append(helper.make_node("Tanh", [n_pre], [nt], name=nt))

    one_minus_z = _unique_name(model, f"{stem}_t{t}_1mz")
    ones = _const_ones_like(model, f"{stem}_t{t}_ones", zt)
    new_nodes.append(
        helper.make_node("Sub", [ones, zt], [one_minus_z], name=one_minus_z)
    )
    a = _unique_name(model, f"{stem}_t{t}_a")
    new_nodes.append(helper.make_node("Mul", [one_minus_z, nt], [a], name=a))
    b = _unique_name(model, f"{stem}_t{t}_b")
    new_nodes.append(helper.make_node("Mul", [zt, h], [b], name=b))
    h_new = _unique_name(model, f"{stem}_t{t}_h")
    new_nodes.append(helper.make_node("Add", [a, b], [h_new], name=h_new))
    return h_new, None


def _const_ones_like(model, stem, ref):
    # `ref` is a runtime tensor (an activation), not a static shape -- a
    # constant `1.0` broadcasts against it in `Sub` without needing its
    # actual shape known ahead of time.
    name = _unique_name(model, stem)
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name)
    )
    return name


def unroll_gru(model):
    """Replaces a forward, single-direction `GRU` node with its own
    per-timestep gate arithmetic, the `GRU` counterpart of `unroll_lstm`.

    A strictly bigger win than the LSTM case: `GRU` is not in
    `AX650_SUPPORTED_OPS` at all (unlike `LSTM`, which at least runs at
    inference), so this doesn't just add a backward rule -- it is the only
    way a `GRU`-containing model runs on this hardware at all, training or
    not. Same scope restrictions as `unroll_lstm` (forward direction,
    static shapes, `W`/`R`/`B` as initializers, no `sequence_lens`).

    Numerically exact to float32 precision: verified directly against a
    real `torch.onnx.export`-produced `GRU` node, `linear_before_reset=1`
    (what `nn.GRU` always exports), gate order `z, r, h`.
    """
    shapes = _value_shapes(model)
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type != "GRU":
            out.append(node)
            continue
        replacement = _unroll_recurrent_node(model, node, shapes, 3, _gru_step)
        if replacement is None:
            out.append(node)
            continue
        out.extend(replacement)
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


#: Order matters. `dilated_conv_to_taps` consumes a convolution's `pads`
#: attribute, so it has to run before `explicit_conv_padding` zeroes it.
RULES = {
    "float16_to_float32": float16_to_float32,
    "pow2_to_mul": pow2_to_mul,
    "neg_to_mul": neg_to_mul,
    "rank0_to_rank1": rank0_to_rank1,
    "inline_local_functions": inline_local_functions,
    "avgpool_ceil_to_floor": avgpool_ceil_to_floor,
    "flatten_to_reshape": flatten_to_reshape,
    "global_pool_to_reduce": global_pool_to_reduce,
    "gemm_to_matmul": gemm_to_matmul,
    "act_weight_conv_to_matmul": act_weight_conv_to_matmul,
    "dilated_conv_to_taps": dilated_conv_to_taps,
    "explicit_conv_padding": explicit_conv_padding,
    "filename_safe_io_names": filename_safe_io_names,
    "unroll_lstm": unroll_lstm,
    "unroll_gru": unroll_gru,
}

#: The rules a graph needs to be a *training* step rather than an inference
#: model: a live weight, a scalar loss, and the ops a backward pass emits.
#: `onnxsim.graph_grad` produces all three.
TRAINING_RULES = (
    "inline_local_functions",
    "avgpool_ceil_to_floor",
    "flatten_to_reshape",
    "global_pool_to_reduce",
    "neg_to_mul",
    "rank0_to_rank1",
    "gemm_to_matmul",
    "act_weight_conv_to_matmul",
    "unroll_lstm",
    "unroll_gru",
)


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
