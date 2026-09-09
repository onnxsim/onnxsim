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


RULES = {
    "float16_to_float32": float16_to_float32,
    "pow2_to_mul": pow2_to_mul,
    "explicit_conv_padding": explicit_conv_padding,
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
