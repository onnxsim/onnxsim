#!/usr/bin/env python3
"""Rewrites that make a graph's use of `auto_pad`/`transA`/`storage_order`
explicit, matching Voyager SDK's own opset-17 AIPU acceleration constraints
(`voyager_ops.py`, scraped from its docs by `scrape_onnx_support_docs.py`).

`voyager_simulator.evaluate_constraints()` is a docs-derived, no-hardware
estimate of those constraints (see its docstring for exactly what it does
and does not claim) -- op *coverage* the same way `pulsar2_ops.py` is for
Axera. For three of its documented rules, the fix is not "avoid this op",
it is a rewrite that is exact, not an approximation:

- **`Conv`/`AveragePool`/`MaxPool` need `auto_pad == "NOTSET"`.** ONNX
  defines each `auto_pad` mode (`SAME_UPPER`, `SAME_LOWER`, `VALID`) as a
  formula over the input's spatial shape (see the ONNX operator spec for
  `Conv`). Computing that formula once and writing the result out as
  explicit `pads` with `auto_pad = "NOTSET"` changes nothing about what the
  op computes, only how the padding amount is spelled -- the two forms are
  defined to agree.
- **`Gemm` needs `transA == 0`.** `transA = 1` means "transpose A before the
  matmul"; inserting an explicit `Transpose` node ahead of the `Gemm` and
  clearing the flag is the same operation, spelled as two nodes instead of
  one attribute.
- **`MaxPool` needs `storage_order == 0`.** The attribute only orders the
  *second*, optional `Indices` output; a graph that never produces that
  output computes an identical first output (`Y`) whichever way
  `storage_order` is set, so forcing it to 0 there is a free rewrite rather
  than a behavior change.

Unlike `scripts/axera/legalize.py`'s rules -- each bisected against a real
`pulsar2 build` failure on real hardware -- nothing here has been checked
against an actual Voyager SDK compile. `axelera-rt`/`axelera-devkit`
(`voyager_backend.py`) do install from a public index per Voyager SDK's own
docs, but it's a large, optional pull (torch, CUDA toolkit packages, TVM)
not exercised for this file. What backs these three rules instead is
`voyager_ops.py`'s own scraped constraint text and
`voyager_simulator.evaluate_constraints()`, derived from the same docs and
itself partially cross-checked against the real compiler (see
`voyager_backend.py`'s docstring) -- each rule's test checks that the
target node's verdict against that checker moves from `"violated"` to
`"ok"`, and separately that `onnx.reference.ReferenceEvaluator` gives the
same output before and after. That is real, mechanical verification that
the graph now reads as compliant and still computes the same thing; it is
not a substitute for an actual `axelera.compiler` run.

Usage::

    legalize.py in.onnx out.onnx
    legalize.py --rules gemm_transA_to_transpose in.onnx out.onnx
"""

from __future__ import annotations

import argparse
import collections

import onnx
import onnx.shape_inference
from onnx import helper


def _attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def _attr_int(node, name, default):
    a = _attr(node, name)
    return a.i if a is not None else default


def _attr_ints(node, name, default=None):
    a = _attr(node, name)
    return list(a.ints) if a is not None else default


def _attr_str(node, name, default):
    a = _attr(node, name)
    return a.s.decode("utf-8") if a is not None else default


def _set_attr(node, name, value):
    """Replace (or add) one attribute, inferring its kind from `value`."""
    existing = _attr(node, name)
    if existing is not None:
        node.attribute.remove(existing)
    node.attribute.append(helper.make_attribute(name, value))


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


def _value_shapes(model):
    """{tensor name: [dims]} for every value whose shape is fully static --
    graph inputs/outputs/value_info with every dim resolved, plus
    initializers. Runs shape inference itself so this works even on a graph
    fragment that carries no `value_info` of its own.
    """
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


def _same_pads(auto_pad, spatial_in, kernel, strides, dilations):
    """ONNX's own `auto_pad` formula (see the `Conv` operator spec), as
    explicit `(begin, end)` pads per spatial axis. `VALID` is zero padding
    by definition; `SAME_UPPER`/`SAME_LOWER` split the padding the output
    needs to reach `ceil(input / stride)`, the extra pixel (if any) going to
    the end (`SAME_UPPER`) or the start (`SAME_LOWER`).
    """
    nd = len(kernel)
    if auto_pad == "VALID":
        return [0] * nd, [0] * nd
    begin, end = [0] * nd, [0] * nd
    for i in range(nd):
        out = -(-spatial_in[i] // strides[i])  # ceil division
        needed = max(
            0,
            (out - 1) * strides[i]
            + ((kernel[i] - 1) * dilations[i] + 1)
            - spatial_in[i],
        )
        if auto_pad == "SAME_UPPER":
            begin[i] = needed // 2
            end[i] = needed - begin[i]
        else:  # SAME_LOWER
            end[i] = needed // 2
            begin[i] = needed - end[i]
    return begin, end


def explicit_auto_pad(model):
    """`Conv`/`AveragePool`/`MaxPool` with `auto_pad` in `SAME_UPPER`,
    `SAME_LOWER` or `VALID` get `auto_pad = "NOTSET"` and the equivalent
    explicit `pads`, computed from the ONNX spec's own formula.

    Needs the input's spatial shape to be statically known -- the formula
    is defined in terms of it -- so a node whose input shape isn't resolved
    (a dynamic axis, or a graph fragment with no shape info at all) is left
    alone rather than guessed at.
    """
    shapes = _value_shapes(model)
    changed = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "AveragePool", "MaxPool"):
            continue
        auto_pad = _attr_str(node, "auto_pad", "NOTSET")
        if auto_pad == "NOTSET":
            continue
        kernel = _attr_ints(node, "kernel_shape")
        if kernel is None and node.op_type == "Conv" and len(node.input) > 1:
            # Conv's kernel_shape is optional -- inferrable from W -- and
            # Voyager's own Conv rule reads it from W for the same reason.
            wshape = shapes.get(node.input[1])
            kernel = wshape[2:] if wshape else None
        xshape = shapes.get(node.input[0])
        if kernel is None or xshape is None or len(xshape) != len(kernel) + 2:
            continue
        nd = len(kernel)
        strides = _attr_ints(node, "strides", [1] * nd)
        dilations = _attr_ints(node, "dilations", [1] * nd)
        begin, end = _same_pads(auto_pad, xshape[2:], kernel, strides, dilations)
        _set_attr(node, "auto_pad", "NOTSET")
        _set_attr(node, "pads", begin + end)
        changed += 1
    return changed


def gemm_transA_to_transpose(model):
    """A `Gemm` with `transA = 1` gets an explicit `Transpose` on `A`
    instead, with `transA` cleared. `A` is required to be rank 2 by the
    `Gemm` spec, so `perm=[1, 0]` is always the right transpose.

    This satisfies Gemm's own documented rule, but does not by itself make
    the rewritten graph fully Voyager-legal: `voyager_ops.py`'s scraped
    `Transpose` rule (`perm == [0, 1, 2, 3]`) is written for 4D feature
    maps and has no rank-2 case, so the inserted `Transpose` reads as its
    own violation under `evaluate_constraints()` -- a real gap in what the
    docs (and this transcription of them) cover for a bare `Gemm`, not a
    correctness problem with the rewrite, which is exact either way.
    """
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type != "Gemm" or _attr_int(node, "transA", 0) == 0:
            out.append(node)
            continue
        stem = node.name or node.output[0]
        transposed = _unique_name(model, f"{stem}_A_T")
        out.append(
            helper.make_node(
                "Transpose",
                [node.input[0]],
                [transposed],
                perm=[1, 0],
                name=_unique_name(model, f"{stem}_transA"),
            )
        )
        node.input[0] = transposed
        _set_attr(node, "transA", 0)
        out.append(node)
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def maxpool_rowmajor_when_indices_unused(model):
    """A `MaxPool` with `storage_order = 1` (column-major) gets it cleared
    to 0 (row-major) whenever the optional `Indices` output isn't actually
    produced -- the attribute only orders that output, so with no `Indices`
    consumer the two settings compute the identical `Y`.
    """
    changed = 0
    for node in model.graph.node:
        if node.op_type != "MaxPool" or _attr_int(node, "storage_order", 0) == 0:
            continue
        if len(node.output) > 1 and node.output[1]:
            continue  # Indices is produced; its ordering is observable
        _set_attr(node, "storage_order", 0)
        changed += 1
    return changed


#: Order doesn't matter between these three -- each only touches its own
#: op_type's attributes/inputs, and none produces a node another consumes.
RULES = {
    "explicit_auto_pad": explicit_auto_pad,
    "gemm_transA_to_transpose": gemm_transA_to_transpose,
    "maxpool_rowmajor_when_indices_unused": maxpool_rowmajor_when_indices_unused,
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
