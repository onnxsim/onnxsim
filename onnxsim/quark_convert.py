"""The float-graph conversions Quark's quantization pipeline runs around the
quantizer for NPU targets, for :mod:`onnxsim.quark_compat`:

before the quantization algorithms (Quark: ORT's graph optimizer, ``OptimizeModel``)

- :func:`graph_cleanup` (Quark: onnxslim's ``SimplifyModel`` and ONNX Runtime's
  ``OptimizeModel``, either one): :func:`remove_identity`,
  :func:`fold_batch_norm` (``Conv`` -> ``BatchNormalization`` with a single
  consumer becomes one ``Conv``: ``W * s``, ``beta + (b - mean) * s`` with
  ``s = gamma / sqrt(var + eps)``) and :func:`fuse_pad` (a zero ``Pad`` into the
  following ``Conv`` / ``AveragePool``); and, ``OptimizeModel`` only,
  :func:`expand_hardswish` (``HardSigmoid`` then ``Mul``); with ``runtime=True``
  the real optimizers run instead where installed (:func:`onnxslim_simplify`,
  :func:`remove_input_init`, :func:`duplicate_shared_biases`,
  :func:`ort_basic_optimize`) and these Python reproductions only stand in for a
  missing one; Quark's own :func:`fold_batch_norm` (ConvTranspose / Gemm) and
  :func:`fold_batch_norm_after_concat` follow either way;

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

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


def fold_batch_norm_after_concat(
    model: onnx.ModelProto, skip: Optional[Set[str]] = None
) -> onnx.ModelProto:
    """Quark's ``fold_batch_norm_after_concat``: a ``BatchNormalization`` whose
    input is a ``Concat`` of ``Conv`` / ``ConvTranspose`` / ``Gemm`` outputs (all
    with constant weights) is folded into those, each taking its slice of the
    channels, and the ``Concat`` takes over the BN's output. Like Quark's, it checks
    neither the ``Concat`` axis nor how many nodes read the convolutions, and uses the
    last producer's op type for every slice. ``skip`` names BNs it leaves alone."""
    skip = skip or set()
    out = _copy(model)
    g = out.graph
    inits = {t.name: t for t in g.initializer}
    remove: List[onnx.NodeProto] = []
    for bn in list(g.node):
        if bn.op_type != "BatchNormalization" or len(bn.input) != 5:
            continue
        if bn.name in skip:
            continue
        producers = {o: n for n in g.node for o in n.output}
        concat = producers.get(bn.input[0])
        if concat is None or concat.op_type != "Concat":
            continue
        parents = [producers[x] for x in concat.input if x in producers]
        foldable = bool(parents)
        for t in parents:
            attrs = {a.name: a for a in t.attribute}
            if t.op_type not in ("ConvTranspose", "Gemm", "Conv"):
                foldable = False
                break
            if (
                t.op_type == "Gemm"
                and (attrs["transB"].i if "transB" in attrs else 0) == 0
            ):
                foldable = False
                if len(t.input) > 1 and t.input[1] in inits:
                    dims = inits[t.input[1]].dims
                    foldable = len(dims) == 2 and dims[0] == dims[1]
                break
            if (
                t.op_type == "ConvTranspose"
                and (attrs["group"].i if "group" in attrs else 1) != 1
            ):
                foldable = False
                break
            if len(t.input) < 2 or t.input[1] not in inits:
                foldable = False
                break
        if not foldable:
            continue
        if any(x not in inits for x in bn.input[3:5]):
            continue
        gamma = (
            numpy_helper.to_array(inits[bn.input[1]]) if bn.input[1] in inits else None
        )
        beta = (
            numpy_helper.to_array(inits[bn.input[2]]) if bn.input[2] in inits else None
        )
        mean = numpy_helper.to_array(inits[bn.input[3]])
        var = numpy_helper.to_array(inits[bn.input[4]])
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-10)
        target_type = parents[-1].op_type
        start = end = 0
        for t in parents:
            w_init = inits[t.input[1]]
            w = numpy_helper.to_array(w_init)
            has_bias = len(t.input) > 2 and t.input[2] in inits
            if has_bias:
                b_init = inits[t.input[2]]
                b = numpy_helper.to_array(b_init)
            else:
                b = np.zeros(w.shape[1] if t.op_type == "ConvTranspose" else w.shape[0])
                name = (t.name or t.output[0]) + "_bias_4bn"
                b_init = numpy_helper.from_array(b.astype(np.float32), name)
                g.initializer.append(b_init)
                inits[name] = g.initializer[-1]
                b_init = inits[name]
                while len(t.input) < 3:
                    t.input.append("")
                t.input[2] = name
            end += b.shape[0]
            sl = slice(start, end)
            mult = (
                gamma[sl] / np.sqrt(var[sl] + eps)
                if gamma is not None
                else 1 / np.sqrt(var[sl] + eps)
            )
            bn_bias = (
                beta[sl] + (-mean[sl]) * mult
                if beta is not None
                else (-mean[sl]) * mult
            )
            if target_type == "Gemm":
                diag = np.diag(mult)
                new_w = np.dot(diag, w)
                new_b = np.dot(diag, b) + bn_bias
            elif target_type == "ConvTranspose":
                scale = mult.reshape(1, len(mult), 1, 1)
                new_w = scale * w
                new_b = (scale.reshape(1, -1) * b + bn_bias).reshape(-1)
            else:  # Conv
                scale = mult.reshape(len(mult), 1, 1, 1)
                new_w = scale * w
                new_b = scale.reshape(-1) * b + bn_bias
            start += b.shape[0]
            w_init.CopyFrom(
                numpy_helper.from_array(new_w.astype(np.float32), w_init.name)
            )
            b_init.CopyFrom(
                numpy_helper.from_array(new_b.astype(np.float32), b_init.name)
            )
        for child in g.node:
            if child is bn:
                continue
            for k, x in enumerate(child.input):
                if x == concat.output[0]:
                    child.input[k] = bn.output[0]
        concat.output[0] = bn.output[0]
        remove.append(bn)
        for n in remove:
            if n in g.node:
                g.node.remove(n)
        remove = []
    _clean_initializers(g)
    return out


def eliminate_duplicate_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """ONNX Runtime's common sub-expression elimination: of two nodes with the same
    op type, domain, inputs and attributes (and a single output each that is not a
    graph output) the later one goes and its consumers read the first's output."""
    out = _copy(model)
    g = out.graph
    graph_outputs = {o.name for o in g.output}
    seen: Dict[Any, onnx.NodeProto] = {}
    rename: Dict[str, str] = {}
    drop: List[onnx.NodeProto] = []
    for n in list(g.node):
        for k, x in enumerate(n.input):
            if x in rename:
                n.input[k] = rename[x]
        if n.op_type == "Constant" or len(n.output) != 1:
            continue
        attrs = tuple(sorted(a.SerializeToString() for a in n.attribute))
        key = (n.op_type, n.domain, tuple(n.input), attrs)
        first = seen.get(key)
        if first is None:
            seen[key] = n
        elif n.output[0] not in graph_outputs:
            rename[n.output[0]] = first.output[0]
            drop.append(n)
    for n in drop:
        g.node.remove(n)
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


def remove_input_init(model: onnx.ModelProto) -> onnx.ModelProto:
    """Quark's ``RemoveInputInit``: initializers that are also listed as graph
    inputs stop being inputs (ONNX Runtime does not fold what is overridable)."""
    out = _copy(model)
    g = out.graph
    names = {t.name for t in g.initializer}
    keep = [i for i in g.input if i.name not in names]
    del g.input[:]
    g.input.extend(keep)
    if out.ir_version < 4:
        out.ir_version = 7
    return out


def duplicate_shared_biases(
    model: onnx.ModelProto,
    op_types: Sequence[str] = ("Conv", "ConvTranspose", "Gemm"),
) -> onnx.ModelProto:
    """Quark's ``CopyBiasInit``: a bias initializer read by several ``Conv`` /
    ``ConvTranspose`` / ``Gemm`` nodes is copied, so that each of them owns one
    (the first reader keeps the original, later ones read ``duplicated<name><k>``)
    and each is quantized on its own."""
    out = _copy(model)
    g = out.graph
    inits = {t.name: t for t in g.initializer}
    used: Dict[str, int] = {}
    for n in g.node:
        if n.op_type not in op_types or len(n.input) < 3:
            continue
        name = n.input[2]
        if name not in inits:
            continue
        if name in used:
            used[name] += 1
            new = onnx.TensorProto()
            new.CopyFrom(inits[name])
            new.name = f"duplicated{name}{used[name]}"
            n.input[2] = new.name
            g.initializer.append(new)
        else:
            used[name] = 1
    return out


def onnxslim_simplify(
    model: onnx.ModelProto, config: Optional[Dict[str, Any]] = None
) -> Optional[onnx.ModelProto]:
    """Quark's ``SimplifyModel``: ``onnxslim.slim`` on the model, ``None`` when
    onnxslim is not installed or fails (Quark then keeps the model as it is)."""
    try:
        from onnxslim import slim
    except Exception:
        return None
    try:
        result = slim(model, **(config or {}))
    except Exception:
        return None
    return result if isinstance(result, onnx.ModelProto) else None


def ort_basic_optimize(model: onnx.ModelProto) -> Optional[onnx.ModelProto]:
    """Quark's ``OptimizeModel`` first stage: ONNX Runtime's basic graph
    optimizations (constant folding, Conv + Add / Mul / BatchNorm folding, Relu
    + Clip, redundant-node elimination, common sub-expression elimination, Pad
    fusion, ...; ``ConstantSharing`` off, as Quark runs it). ``None`` when
    onnxruntime is missing or cannot load the model."""
    try:
        import onnxruntime as ort
    except Exception:
        return None
    import os
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "optimized_model.onnx")
            so = ort.SessionOptions()
            so.optimized_model_filepath = path
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            so.log_severity_level = 4
            try:
                ort.InferenceSession(
                    model.SerializeToString(),
                    so,
                    providers=["CPUExecutionProvider"],
                    disabled_optimizers=["ConstantSharing"],
                )
            except TypeError:  # pragma: no cover - onnxruntime < 1.10
                ort.InferenceSession(
                    model.SerializeToString(), so, providers=["CPUExecutionProvider"]
                )
            return onnx.load(path)
    except Exception:
        return None


def graph_cleanup(
    model: onnx.ModelProto,
    optimize: bool = True,
    simplify: bool = True,
    runtime: bool = False,
    slim_config: Optional[Dict[str, Any]] = None,
    copy_bias_ops: Optional[Sequence[str]] = ("Conv", "ConvTranspose", "Gemm"),
    fold_bn: Optional[bool] = None,
    fuse: Optional[Dict[str, Any]] = None,
    keep_bn: Optional[Set[str]] = None,
) -> onnx.ModelProto:
    """What Quark's float-model optimizers (onnxslim's ``SimplifyModel`` and ONNX
    Runtime's ``OptimizeModel``, ``optimize``) do that changes what is
    quantized: ``Identity`` removal, Pad fusion into a ``Conv`` and BatchNorm
    folding into a ``Conv`` / ``ConvTranspose`` / ``Gemm``. (Pad fusion into an ``AveragePool`` and HardSwish
    inlining, :func:`expand_hardswish`, are ONNX Runtime's alone.)

    ``runtime=True`` runs the real optimizers where they are installed, in
    Quark's order (onnxslim, then ``RemoveInputInit`` and ``CopyBiasInit``, then
    ONNX Runtime's basic level) and falls back to the Python reproductions above
    for the one that is missing -- ONNX Runtime also folds constants, merges
    ``Conv`` + ``Add`` / ``Mul``, fuses ``Relu`` + ``Clip``, drops no-op nodes
    and merges duplicated nodes, which the reproductions do not.
    ``copy_bias_ops`` are the op types whose shared bias is copied per node
    (Quark's ``CopyBiasInit``; it does so for the min / max, entropy, percentile
    and distribution calibrations only -- not for the power-of-two ones, where a
    shared bias is quantized once, with its first reader's scales -- so the caller
    passes ``None`` there).

    ``fuse`` (keyword flags of :func:`onnxsim.quark_fusions.apply_fusions`:
    ``instance_norm`` / ``l2_norm`` / ``layer_norm`` / ``gelu``) runs Quark's own
    operator fusions after the two optimizers and before its BatchNorm folding, where
    Quark runs them; ``None`` leaves them out. Where ONNX Runtime's basic optimizer is
    only reproduced (not run), its LayerNormalization / Gelu fusions are reproduced
    too, with the nodes it writes.

    ``keep_bn`` names the ``BatchNormalization`` nodes Quark's quantizer is asked to
    quantize (their op type is on its list: ``QuantizeAllOpTypes``, an extra op type):
    its own folding passes leave exactly those alone -- ONNX Runtime's still fold a
    Conv + BN."""
    from onnxsim.quark_fusions import apply_fusions

    if not runtime:
        out = remove_identity(model)
        if simplify:
            out = fuse_pad(out, pools=False, shared=True)
        if optimize:
            out = fuse_pad(out, pools=True, shared=False)
            out = apply_fusions(out, instance_norm=False, l2_norm=False, style="ort")
        if fuse is not None:
            out = apply_fusions(out, **fuse)
        if keep_bn:
            # (ONNX Runtime's Conv + BN fusion is not Quark's pass)
            out = fold_batch_norm(out)
        out = fold_batch_norm(out, keep_bn, transposed_and_gemm=True)
        if optimize if fold_bn is None else fold_bn:
            out = fold_batch_norm_after_concat(out, keep_bn)
        return out
    out = model
    if simplify:
        slimmed = onnxslim_simplify(out, slim_config)
        if slimmed is None:
            out = fuse_pad(remove_identity(out), pools=False, shared=True)
        else:
            out = slimmed
    out = remove_input_init(out)
    if copy_bias_ops:
        out = duplicate_shared_biases(out, copy_bias_ops)
    if optimize:
        optimized = ort_basic_optimize(out)
        if optimized is None:
            out = fuse_pad(
                remove_identity(expand_hardswish(out)), pools=True, shared=False
            )
            out = fold_batch_norm(out)
            out = eliminate_duplicate_nodes(out)
            out = apply_fusions(out, instance_norm=False, l2_norm=False, style="ort")
        else:
            out = optimized
    if fuse is not None:
        out = apply_fusions(out, **fuse)
    # Quark's own optimizer: BN after a ConvTranspose / Gemm, and after a Concat of
    # convolutions, when ``FoldBatchNorm`` (default: ``OptimizeModel``) is on
    if optimize if fold_bn is None else fold_bn:
        out = fold_batch_norm(out, keep_bn, transposed_and_gemm=True)
        out = fold_batch_norm_after_concat(out, keep_bn)
    return out


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

    # (Quark's conversions append their new nodes at the end of the list, remove the
    # old ones and sort the graph with its own ``topological_sort``: the order of
    # the siblings in the final graph, which the quantizer and the position passes
    # visit, depends on it)
    from onnxsim.quark_marking import quark_sort_inplace

    if on("ConvertReduceMeanToGlobalAvgPool"):
        for n in list(g.node):
            if n.op_type == "ReduceMean" and _reduce_mean_to_gap(n, inits) and ok(n):
                new = helper.make_node(
                    "GlobalAveragePool", [n.input[0]], list(n.output), name=n.name
                )
                g.node.remove(n)
                g.node.append(new)
        _clean_initializers(g)
        quark_sort_inplace(out)

    if on("SplitLargeKernelPool"):
        shapes = _shapes(out)
        for n in list(g.node):
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
            g.node.append(pool)
        quark_sort_inplace(out)

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
            for k, name in enumerate(n.output):
                parts = {}
                consts: List[onnx.NodeProto] = []
                for key, val in (
                    ("starts", starts[k]),
                    ("ends", ends[k]),
                    ("axes", axis),
                    ("steps", 1),
                ):
                    t = f"{name}_{key}_{k}"
                    parts[key] = t
                    consts.append(
                        helper.make_node(
                            "Constant",
                            [],
                            [t],
                            value=helper.make_tensor(t, TensorProto.INT64, [1], [val]),
                        )
                    )
                g.node.append(
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
                g.node.extend(consts)
            remove.append(n)
        for n in remove:
            g.node.remove(n)
        _clean_initializers(g)
        quark_sort_inplace(out)

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
            g.node.append(conv)
            remove.append(n)
        for n in remove:
            g.node.remove(n)
        _clean_initializers(g)
        quark_sort_inplace(out)
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
    "duplicate_shared_biases",
    "eliminate_duplicate_nodes",
    "expand_hardswish",
    "fold_batch_norm",
    "fold_batch_norm_after_concat",
    "fuse_pad",
    "graph_cleanup",
    "onnxslim_simplify",
    "ort_basic_optimize",
    "remove_identity",
    "remove_input_init",
]
