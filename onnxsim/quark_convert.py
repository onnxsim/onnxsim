"""The float-graph conversions Quark's quantization pipeline runs around the
quantizer for NPU targets, for :mod:`onnxsim.quark_compat`:

before the quantization algorithms (Quark: ORT's graph optimizer, ``OptimizeModel``)

- :func:`graph_cleanup` (Quark: onnxslim's ``SimplifyModel`` and ONNX Runtime's
  ``OptimizeModel``, either one): :func:`remove_identity`,
  :func:`fold_batch_norm` (``Conv`` -> ``BatchNormalization`` with a single
  consumer becomes one ``Conv``: ``W * s``, ``beta + (b - mean) * s`` with
  ``s = gamma / sqrt(var + eps)``) and :func:`fuse_pad` (a zero ``Pad`` into the
  following ``Conv`` / ``AveragePool``); and, ``OptimizeModel`` only,
  :func:`expand_hardswish` (``HardSigmoid`` then ``Mul``);

after them (Quark's ``optimize_model`` flags, defaults on under ``EnableNPUCnn``)

- ``ConvertReduceMeanToGlobalAvgPool``: ``ReduceMean`` over axes [2, 3] with
  ``keepdims=1`` -> ``GlobalAveragePool``;
- ``SplitLargeKernelPool``: a ``GlobalAveragePool`` over more than 512
  positions -> ``AveragePool`` (the largest square-ish factor of the height and
  width as kernel and stride) then ``GlobalAveragePool``;
- ``ConvertSplitToSlice``: ``Split`` with explicit sizes -> one ``Slice`` per
  output;
- ``ConvertBNToConv``: a ``BatchNormalization`` on a 4-D tensor (one with a
  known shape, i.e. not a graph input) -> a 1x1 depthwise ``Conv``
  (``gamma / sqrt(var + eps)`` and ``beta - mean * that``; Quark's ``eps`` default
  is 1e-10 here, not ONNX's 1e-5).

and, after quantization (``ConvertClipToRelu``, off unless asked, on in
``VINT8``): :func:`convert_clip_to_relu` turns each ``Clip`` whose lower bound
is >= 0 into a ``Relu`` -- dropping the upper bound, so ``Clip(0, 6)`` stops
clipping at 6.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _shapes(model: onnx.ModelProto) -> Dict[str, List[int]]:
    """Tensor name -> dims (0 for unknown) of the intermediate tensors (graph
    inputs and outputs excluded, as in Quark's ``value_info`` lookups)."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # pragma: no cover - shape inference failure
        return {}
    return {
        vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
        for vi in inferred.graph.value_info
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape")
    }


def _copy(model: onnx.ModelProto) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    return out


def _consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    out: Dict[str, List[onnx.NodeProto]] = {}
    for n in graph.node:
        for x in n.input:
            out.setdefault(x, []).append(n)
    return out


def _clean_initializers(graph: onnx.GraphProto) -> None:
    used = {x for n in graph.node for x in n.input}
    used |= {o.name for o in graph.output}
    keep = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)


def _fold_into_transposed_or_gemm(
    g: onnx.GraphProto,
    inits: Dict[str, Any],
    uses: Dict[str, int],
    cons: Dict[str, List[onnx.NodeProto]],
    bn: onnx.NodeProto,
    target: onnx.NodeProto,
    skip: Set[str],
) -> bool:
    """Quark's ``fold_batch_norm`` for a ``ConvTranspose`` / ``Gemm`` target;
    True once the BN has been folded (the caller drops the node)."""
    attrs = {a.name: a for a in target.attribute}
    if target.name in skip or target.output[0] in {o.name for o in g.output}:
        return False
    if len(cons.get(target.output[0], [])) != 1 or len(target.input) < 2:
        return False
    if (
        target.op_type == "Gemm"
        and (attrs["transB"].i if "transB" in attrs else 0) != 1
    ):
        return False
    if (
        target.op_type == "ConvTranspose"
        and (attrs["group"].i if "group" in attrs else 1) != 1
    ):
        return False
    has_bias = len(target.input) > 2 and target.input[2] != ""
    needed = (
        list(bn.input[1:]) + [target.input[1]] + ([target.input[2]] if has_bias else [])
    )
    if any(x not in inits for x in needed) or any(
        uses.get(x, 0) != 1 for x in needed[4:]
    ):
        return False
    eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-10)
    gamma, beta, mean, var = (numpy_helper.to_array(inits[x]) for x in bn.input[1:])
    w = numpy_helper.to_array(inits[target.input[1]])
    mult = gamma / np.sqrt(var + eps)
    bn_bias = beta + (-mean) * mult
    if has_bias:
        b = numpy_helper.to_array(inits[target.input[2]])
    else:
        b = np.zeros(w.shape[0] if target.op_type == "Gemm" else w.shape[1])
    if target.op_type == "Gemm":
        diag = np.diag(mult)
        new_w = np.dot(diag, w)
        new_b = np.dot(diag, b) + bn_bias
    else:
        scale = mult.reshape(1, len(mult), 1, 1)
        new_w = scale * w
        new_b = (scale.reshape(1, -1) * b + bn_bias).reshape(-1)
    inits[target.input[1]].CopyFrom(
        numpy_helper.from_array(new_w.astype(np.float32), target.input[1])
    )
    if has_bias:
        inits[target.input[2]].CopyFrom(
            numpy_helper.from_array(new_b.astype(np.float32), target.input[2])
        )
    else:
        name = (target.name or target.output[0]) + "_bias_4bn"
        g.initializer.append(numpy_helper.from_array(new_b.astype(np.float32), name))
        while len(target.input) < 3:
            target.input.append("")
        target.input[2] = name
    target.output[0] = bn.output[0]
    return True


def fold_batch_norm(
    model: onnx.ModelProto,
    skip: Optional[Set[str]] = None,
    transposed_and_gemm: bool = False,
) -> onnx.ModelProto:
    """Fold each ``Conv`` -> ``BatchNormalization`` pair (constant weights, the
    convolution's output used only by the BN) into the convolution. With
    ``transposed_and_gemm`` also a ``ConvTranspose`` (``group=1``) or a ``Gemm``
    (``transB=1``) with its BN, the way Quark's own optimizer does (a BN epsilon
    attribute missing is 1e-10 there)."""
    skip = skip or set()
    out = _copy(model)
    g = out.graph
    inits = {t.name: t for t in g.initializer}
    cons = _consumers(g)
    graph_outputs = {o.name for o in g.output}
    producers = {o: n for n in g.node for o in n.output}
    uses: Dict[str, int] = {}
    for n in g.node:
        for x in n.input:
            uses[x] = uses.get(x, 0) + 1
    remove: List[onnx.NodeProto] = []
    for bn in list(g.node):
        if bn.op_type != "BatchNormalization" or len(bn.input) != 5:
            continue
        if len([o for o in bn.output if o]) != 1 or bn.name in skip:
            continue
        conv = producers.get(bn.input[0])
        if (
            transposed_and_gemm
            and conv is not None
            and conv.op_type in ("ConvTranspose", "Gemm")
            and _fold_into_transposed_or_gemm(g, inits, uses, cons, bn, conv, skip)
        ):
            remove.append(bn)
            continue
        if (
            conv is None
            or conv.op_type != "Conv"
            or conv.name in skip
            or conv.output[0] in graph_outputs
            or len(cons.get(conv.output[0], [])) != 1
            or conv.input[1] not in inits
            or any(x not in inits for x in bn.input[1:])
            or uses.get(conv.input[1], 0) != 1
        ):
            continue
        has_bias = len(conv.input) > 2 and conv.input[2] != ""
        if has_bias and (conv.input[2] not in inits or uses.get(conv.input[2], 0) != 1):
            continue
        w = numpy_helper.to_array(inits[conv.input[1]])
        if w.dtype != np.float32:
            continue
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        gamma, beta, mean, var = (
            numpy_helper.to_array(inits[x]).astype(np.float32) for x in bn.input[1:]
        )
        s = gamma / np.sqrt(var + np.float32(eps))
        b = (
            numpy_helper.to_array(inits[conv.input[2]]).astype(np.float32)
            if has_bias
            else np.zeros(w.shape[0], np.float32)
        )
        new_w = (w * s.reshape([-1] + [1] * (w.ndim - 1))).astype(np.float32)
        new_b = (beta + (b - mean) * s).astype(np.float32)
        inits[conv.input[1]].CopyFrom(numpy_helper.from_array(new_w, conv.input[1]))
        if has_bias:
            inits[conv.input[2]].CopyFrom(numpy_helper.from_array(new_b, conv.input[2]))
        else:
            name = (conv.name or conv.output[0]) + "_bias_bn"
            g.initializer.append(numpy_helper.from_array(new_b, name))
            while len(conv.input) < 3:
                conv.input.append("")
            conv.input[2] = name
        conv.output[0] = bn.output[0]
        remove.append(bn)
    for n in remove:
        g.node.remove(n)
    _clean_initializers(g)
    return out


def expand_hardswish(model: onnx.ModelProto) -> onnx.ModelProto:
    """``HardSwish(x)`` -> ``HardSigmoid(x, alpha=1/6, beta=0.5)`` then ``Mul(x,
    that)``: what ONNX Runtime's function inlining leaves of it."""
    out = _copy(model)
    g = out.graph
    for n in list(g.node):
        if n.op_type != "HardSwish":
            continue
        hs = "_inlfunc_HardSwish_" + n.output[0] + "_HS_X"
        idx = list(g.node).index(n)
        g.node.remove(n)
        g.node.insert(
            idx,
            helper.make_node(
                "HardSigmoid",
                [n.input[0]],
                [hs],
                alpha=float(np.float32(1.0 / 6.0)),
                beta=0.5,
                name=hs,
            ),
        )
        g.node.insert(
            idx + 1,
            helper.make_node("Mul", [n.input[0], hs], list(n.output), name=n.name),
        )
    return out


def fuse_pad(
    model: onnx.ModelProto, pools: bool = True, shared: bool = True
) -> onnx.ModelProto:
    """Fold a constant zero ``Pad`` of the spatial dims into each ``Conv`` (and,
    with ``pools``, ``AveragePool`` / ``MaxPool``) that reads it (``pads`` added; an
    ``AveragePool`` then counts the padding, ``count_include_pad=1``), like ONNX
    Runtime's pad fusion. The ``Pad`` goes once nothing reads it any more.
    ``shared=False`` fuses only a ``Pad`` that has a single reader (ONNX Runtime;
    onnxslim fuses it into every ``Conv`` that reads it)."""
    out = _copy(model)
    g = out.graph
    inits = {t.name: t for t in g.initializer}
    graph_outputs = {o.name for o in g.output}
    producers = {o: n for n in g.node for o in n.output}
    fused: List[onnx.NodeProto] = []
    for child in list(g.node):
        if child.op_type not in ("Conv", "AveragePool", "MaxPool") or not child.input:
            continue
        if child.op_type != "Conv" and not pools:
            continue
        pad = producers.get(child.input[0])
        if pad is None or pad.op_type != "Pad" or child.input[0] != pad.output[0]:
            continue
        if not shared and (
            sum(child.input[0] in n.input for n in g.node) != 1
            or pad.output[0] in graph_outputs
        ):
            continue
        attrs = {a.name: a for a in child.attribute}
        if "auto_pad" in attrs and attrs["auto_pad"].s not in (b"NOTSET", b""):
            continue
        if any(a.name == "mode" and a.s != b"constant" for a in pad.attribute):
            continue
        if len(pad.input) < 2 or pad.input[1] not in inits:
            continue
        if len(pad.input) > 2 and pad.input[2]:
            if pad.input[2] not in inits:
                continue
            if numpy_helper.to_array(inits[pad.input[2]]).ravel()[0] != 0:
                continue
        if len(pad.input) > 3 and pad.input[3]:
            continue  # explicit axes
        pads = numpy_helper.to_array(inits[pad.input[1]]).tolist()
        rank = len(pads) // 2
        if rank < 3 or any(pads[i] or pads[rank + i] for i in (0, 1)):
            continue
        if any(p < 0 for p in pads):
            continue
        spatial = rank - 2
        old = list(attrs["pads"].ints) if "pads" in attrs else [0] * (2 * spatial)
        new = [old[i] + pads[2 + i] for i in range(spatial)] + [
            old[spatial + i] + pads[rank + 2 + i] for i in range(spatial)
        ]
        if "pads" in attrs:
            del attrs["pads"].ints[:]
            attrs["pads"].ints.extend(new)
        else:
            child.attribute.append(helper.make_attribute("pads", new))
        if child.op_type == "AveragePool":
            cip = [a for a in child.attribute if a.name == "count_include_pad"]
            if cip:
                cip[0].i = 1
            else:
                child.attribute.append(helper.make_attribute("count_include_pad", 1))
        child.input[0] = pad.input[0]
        fused.append(pad)
    read = {x for n in g.node for x in n.input} | graph_outputs
    for pad in dict.fromkeys(map(id, fused)):
        node = next(n for n in fused if id(n) == pad)
        if node.output[0] not in read:
            g.node.remove(node)
    _clean_initializers(g)
    return out


def remove_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop ``Identity`` nodes whose output is not a graph output (consumers read
    the input directly)."""
    out = _copy(model)
    g = out.graph
    graph_outputs = {o.name for o in g.output}
    for n in list(g.node):
        if n.op_type != "Identity" or n.output[0] in graph_outputs:
            continue
        for m in g.node:
            for k, x in enumerate(m.input):
                if x == n.output[0]:
                    m.input[k] = n.input[0]
        g.node.remove(n)
    return out


def graph_cleanup(
    model: onnx.ModelProto, optimize: bool = True, simplify: bool = True
) -> onnx.ModelProto:
    """What Quark's float-model optimizers (onnxslim's ``SimplifyModel`` and ONNX
    Runtime's ``OptimizeModel``, ``optimize``) do that changes what is
    quantized: ``Identity`` removal, Pad fusion into a ``Conv`` and BatchNorm
    folding into a ``Conv`` / ``ConvTranspose`` / ``Gemm``. (Pad fusion into an ``AveragePool`` and HardSwish
    inlining, :func:`expand_hardswish`, are ONNX Runtime's alone.)"""
    out = remove_identity(model)
    if simplify:
        out = fuse_pad(out, pools=False, shared=True)
    if optimize:
        out = fuse_pad(out, pools=True, shared=False)
    return fold_batch_norm(out, transposed_and_gemm=True)


def _reduce_mean_to_gap(n: onnx.NodeProto, inits: Dict[str, Any]) -> bool:
    attrs = {a.name: a for a in n.attribute}
    keep_ok = "keepdims" not in attrs or attrs["keepdims"].i == 1
    if "axes" in attrs:
        return list(attrs["axes"].ints) == [2, 3] and keep_ok
    if keep_ok and len(n.input) == 2 and n.input[1] in inits:
        return numpy_helper.to_array(inits[n.input[1]]).tolist() == [2, 3]
    return False


def _factors(num: int) -> "tuple[int, int]":
    f1 = int(math.sqrt(num))
    while f1 > 1:
        if num % f1 == 0:
            return f1, int(num / f1)
        f1 -= 1
    return f1, num


def convert_for_npu(
    model: onnx.ModelProto,
    options: Dict[str, Any],
    should_convert: Optional[Callable[[onnx.NodeProto], bool]] = None,
    default: bool = True,
) -> onnx.ModelProto:
    """Quark's ``apply_pre_optimization_after_algo`` conversions (see the module
    docstring) on a copy of ``model``; each is switched by its Quark option
    (``ConvertReduceMeanToGlobalAvgPool``, ``SplitLargeKernelPool``,
    ``ConvertSplitToSlice``, ``ConvertBNToConv``), which defaults to ``default``
    (on for the NPU / extended / transformer flows, off otherwise)."""
    ok = should_convert or (lambda n: True)

    def on(key: str) -> bool:
        return bool(options.get(key, default))

    out = _copy(model)
    g = out.graph
    inits = {t.name: t for t in g.initializer}

    if on("ConvertReduceMeanToGlobalAvgPool"):
        for i, n in enumerate(list(g.node)):
            if n.op_type == "ReduceMean" and _reduce_mean_to_gap(n, inits) and ok(n):
                new = helper.make_node(
                    "GlobalAveragePool", [n.input[0]], list(n.output), name=n.name
                )
                idx = list(g.node).index(n)
                g.node.remove(n)
                g.node.insert(idx, new)
        _clean_initializers(g)

    if on("SplitLargeKernelPool"):
        shapes = _shapes(out)
        i = 0
        while i < len(g.node):
            n = g.node[i]
            i += 1
            if n.op_type != "GlobalAveragePool" or not ok(n):
                continue
            shape = shapes.get(n.input[0])
            if not shape or len(shape) != 4 or not shape[2] or not shape[3]:
                continue
            kh, kw = shape[2], shape[3]
            if kh * kw <= 512:
                continue
            kh1, kh2 = _factors(kh)
            kw1, kw2 = _factors(kw)
            if kh1 * kw1 > 512 or kh2 * kw2 > 512:
                continue
            split = n.input[0] + "_Split"
            pool = helper.make_node(
                "AveragePool",
                [n.input[0]],
                [split],
                kernel_shape=[kh1, kw1],
                strides=[kh1, kw1],
                name=split,
            )
            if not n.name:
                n.name = n.output[0]
            n.input[0] = split
            g.node.insert(i - 1, pool)
            i += 1

    if on("ConvertSplitToSlice"):
        remove: List[onnx.NodeProto] = []
        for n in list(g.node):
            if n.op_type != "Split" or not ok(n):
                continue
            axis = next((a.i for a in n.attribute if a.name == "axis"), None)
            if axis is None:
                break
            if len(n.input) == 2:
                if n.input[1] not in inits:
                    break
                splits = numpy_helper.to_array(inits[n.input[1]]).tolist()
            elif len(n.input) == 1:
                attr = next((a for a in n.attribute if a.name == "split"), None)
                if attr is None:
                    break
                splits = list(attr.ints)
            else:
                break
            starts = [sum(splits[:k]) for k in range(len(splits))]
            ends = [sum(splits[: k + 1]) for k in range(len(splits))]
            idx = list(g.node).index(n)
            slices: List[onnx.NodeProto] = []
            for k, name in enumerate(n.output):
                parts = {}
                for key, val in (
                    ("starts", starts[k]),
                    ("ends", ends[k]),
                    ("axes", axis),
                    ("steps", 1),
                ):
                    t = f"{name}_{key}_{k}"
                    parts[key] = t
                    slices.append(
                        helper.make_node(
                            "Constant",
                            [],
                            [t],
                            value=helper.make_tensor(t, TensorProto.INT64, [1], [val]),
                        )
                    )
                slices.append(
                    helper.make_node(
                        "Slice",
                        [
                            n.input[0],
                            parts["starts"],
                            parts["ends"],
                            parts["axes"],
                            parts["steps"],
                        ],
                        [name],
                        name=f"{name}_{k}",
                    )
                )
            for k, node in enumerate(slices):
                g.node.insert(idx + k, node)
            remove.append(n)
        for n in remove:
            g.node.remove(n)
        _clean_initializers(g)

    if on("ConvertBNToConv"):
        shapes = _shapes(out)
        inits = {t.name: t for t in g.initializer}
        remove = []
        for n in list(g.node):
            if n.op_type != "BatchNormalization" or not ok(n):
                continue
            shape = shapes.get(n.input[0], [])
            if len(n.input) != 5 or len(shape) != 4:
                continue
            if any(x not in inits for x in n.input[1:]):
                continue
            eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-10)
            gamma, beta, mean, var = (
                numpy_helper.to_array(inits[x]) for x in n.input[1:]
            )
            mult = gamma / np.sqrt(var + eps)
            weights = mult.astype(np.float32).reshape([mean.shape[0], 1, 1, 1])
            bias = (beta + (-mean) * mult).astype(np.float32)
            wname, bname = n.output[0] + "weights", n.output[0] + "bias"
            g.initializer.extend(
                [
                    numpy_helper.from_array(weights, wname),
                    numpy_helper.from_array(bias, bname),
                ]
            )
            conv = helper.make_node(
                "Conv",
                [n.input[0], wname, bname],
                [n.output[0]],
                group=int(mean.shape[0]),
                kernel_shape=[1, 1],
                strides=[1, 1],
                name=n.name,
            )
            idx = list(g.node).index(n)
            g.node.insert(idx, conv)
            remove.append(n)
        for n in remove:
            g.node.remove(n)
        _clean_initializers(g)
    return out


def _clip_bound(
    graph: onnx.GraphProto, inits: Dict[str, Any], node: onnx.NodeProto, k: int
) -> Optional[float]:
    name = node.input[k] if len(node.input) > k else ""
    for a in node.attribute:  # opset < 11: attributes
        if a.name == ("min" if k == 1 else "max"):
            return float(a.f)
    if name in inits:
        return float(numpy_helper.to_array(inits[name]).ravel()[0])
    for n in graph.node:
        if name and name in n.output:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        return float(numpy_helper.to_array(a.t).ravel()[0])
            if n.op_type == "Identity" and n.input and n.input[0] in inits:
                return float(numpy_helper.to_array(inits[n.input[0]]).ravel()[0])
    return None


def convert_clip_to_relu(
    model: onnx.ModelProto,
    should_convert: Optional[Callable[[onnx.NodeProto], bool]] = None,
) -> onnx.ModelProto:
    """``Clip`` with a lower bound >= 0 (given as an initializer or a constant;
    the bounds are dropped) -> ``Relu``, same name and tensors."""
    ok = should_convert or (lambda n: True)
    out = _copy(model)
    g = out.graph
    inits = {t.name: t for t in g.initializer}
    for n in list(g.node):
        if n.op_type != "Clip" or not ok(n):
            continue
        lo = _clip_bound(g, inits, n, 1)
        if lo is None or lo < 0:
            continue
        bounds = {x for x in n.input[1:] if x}
        relu = helper.make_node("Relu", [n.input[0]], list(n.output), name=n.name)
        idx = list(g.node).index(n)
        g.node.remove(n)
        g.node.insert(idx, relu)
        for b in bounds:  # a Constant / Identity feeding only this Clip
            users = [m for m in g.node if b in m.input]
            for p in list(g.node):
                if (
                    b in p.output
                    and p.op_type in ("Constant", "Identity")
                    and not users
                ):
                    g.node.remove(p)
    _clean_initializers(g)
    return out


__all__: Any = [
    "convert_clip_to_relu",
    "convert_for_npu",
    "expand_hardswish",
    "fold_batch_norm",
    "fuse_pad",
    "graph_cleanup",
    "remove_identity",
]
