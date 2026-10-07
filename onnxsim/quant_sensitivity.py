"""Which layers is quantization sensitive to? Estimators, a brute-force oracle, and a decision helper.

Choosing which layers to leave in float is the practical question behind mixed-precision
quantization. This module puts several *estimators* of a layer's quantization sensitivity behind
one API, together with the brute-force measurement they are trying to predict, so they can be
compared honestly (``scripts/quant_sensitivity_bench.py`` does that on real models; the measured
results are in ``docs/quant-sensitivity.md``).

A *site* is a MatMul/Gemm/Conv node with a constant weight. Quantizing a site means either its
**weights** (per-output-channel symmetric, ``bits``) or its **input activation** (per-tensor
affine, calibrated). Sites can be grouped (all linear layers of one transformer block); every
estimator scores *groups*, and a singleton group is a site.

Estimators (``methods=``); higher score = more sensitive:

``weight_err`` / ``weight_err_rel`` (weights)
    ``||W - Q(W)||_F`` and the same divided by ``||W||_F``. Free; ignores the data and the rest of
    the network.
``act_scale`` (activations)
    The calibrated quantization step of the site's input. Free.
``fisher``
    ``mean_i (g_i . delta)^2`` where ``g_i`` is the gradient of ``log p(y_i | x_i)`` with respect to
    the site's weight (or input activation) and ``delta`` is the quantization perturbation
    (``Q(W) - W``; or the activation's actual rounding error on sample ``i``). With ``y_i`` drawn
    from the model's own distribution this is twice the diagonal-Fisher estimate of the expected
    KL divergence a perturbation ``delta`` causes. One backward pass per calibration sample gives
    every site at once, which is what makes it cheaper than quantizing sites one by one.
``taylor``
    ``mean_i |g_i . delta|``: the first-order change of ``log p(y_i | x_i)`` itself (same
    gradients and ``y_i`` as ``fisher``, absolute value instead of square, so a perturbation that
    leaves the loss unchanged to first order scores zero).
``hessian_trace`` (weights)
    HAWQ-V2: ``tr(H) / n * ||delta||^2`` with ``tr(H)`` from Hutchinson's estimator (Rademacher
    probes, Hessian-vector products by double backward) of the cross-entropy Hessian.
``certified``
    A *sound* bound on ``max |float(x) - quantized(x)|`` over an input box, from
    :mod:`onnxsim.zonotope` (weights) or :mod:`onnxsim.quant_verify` (8-bit activations). Needs
    ``input_ranges`` and only fits small graphs (``max_certified_elements``); skipped sites are
    reported, never silently scored.

What this module deliberately does *not* claim: that any estimator ranks well. The point is to
measure that. Worst-case certified bounds and typical-case estimates answer different questions;
see the docs.

The gradient-based estimators run the ONNX graph through a small differentiable executor
(:class:`TorchGraph`, the ops CNNs and transformer encoders use); an unsupported op raises an
error naming it. ``onnxsim.to_torch`` is not used: it targets LLM graphs and has no Conv rule.
"""

import copy
import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

Feeds = Dict[str, np.ndarray]

_SITE_OPS = ("Conv", "Gemm", "MatMul")


# --------------------------------------------------------------------------
# Sites and quantizers
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Site:
    """One quantizable MatMul/Gemm/Conv with a constant weight."""

    name: str
    node_index: int
    op_type: str
    weight: str  # initializer name
    input: str  # activation tensor name feeding the node
    channel_axis: int  # output-channel axis of the weight (per-channel quantization)
    n_weight: int = 0


def _initializers(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    out = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output:
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = numpy_helper.to_array(a.t)
    return out


def find_sites(model: onnx.ModelProto) -> List[Site]:
    """The quantizable layers of ``model``, in graph order."""
    consts = _initializers(model)
    sites: List[Site] = []
    for k, n in enumerate(model.graph.node):
        if n.op_type not in _SITE_OPS or len(n.input) < 2 or n.input[1] not in consts:
            continue
        w = consts[n.input[1]]
        if w.dtype.kind != "f" or w.ndim < 2:
            continue
        if n.op_type == "Conv":
            axis = 0
        elif n.op_type == "Gemm":
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            axis = 0 if trans_b else 1
        else:  # MatMul(x, W[K, N])
            if w.ndim != 2:
                continue
            axis = 1
        sites.append(
            Site(
                name=n.name or f"{n.op_type}_{k}",
                node_index=k,
                op_type=n.op_type,
                weight=n.input[1],
                input=n.input[0],
                channel_axis=axis,
                n_weight=int(w.size),
            )
        )
    return sites


def quantize_weights(w: np.ndarray, bits: int, axis: int) -> np.ndarray:
    """Per-output-channel symmetric min-max fake quantization of ``w`` (returns float32)."""
    q = 2 ** (bits - 1) - 1
    red = tuple(i for i in range(w.ndim) if i != axis)
    scale = np.maximum(np.abs(w).max(axis=red, keepdims=True), 1e-12) / q
    return (np.clip(np.round(w / scale), -q - 1, q) * scale).astype(np.float32)


@dataclasses.dataclass
class ActQuant:
    """Per-tensor affine activation quantizer."""

    scale: float
    zero_point: int
    bits: int

    @property
    def qmax(self) -> int:
        return 2**self.bits - 1

    def apply(self, a: np.ndarray) -> np.ndarray:
        q = np.clip(np.round(a / self.scale) + self.zero_point, 0, self.qmax)
        return ((q - self.zero_point) * self.scale).astype(a.dtype)


def calibrate_activation(
    values: np.ndarray, bits: int, percentile: float = 99.99
) -> ActQuant:
    """Affine quantizer from the ``percentile`` range of calibration activations (0 included)."""
    lo = min(float(np.percentile(values, 100 - percentile)), 0.0)
    hi = max(float(np.percentile(values, percentile)), 0.0)
    scale = (hi - lo) / (2**bits - 1) if hi > lo else 1.0
    zp = int(np.clip(round(-lo / scale), 0, 2**bits - 1))
    return ActQuant(scale, zp, bits)


def _replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for t in model.graph.initializer:
        if t.name == name:
            t.CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))
            return
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    a.t.CopyFrom(
                        numpy_helper.from_array(value.astype(np.float32), name)
                    )
                    return
    raise KeyError(name)


def with_quantized_weights(
    model: onnx.ModelProto, sites: Sequence[Site], bits: int
) -> onnx.ModelProto:
    """A copy of ``model`` whose ``sites`` have fake-quantized (dequantized float) weights."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    consts = _initializers(model)
    for s in sites:
        _replace_initializer(
            out, s.weight, quantize_weights(consts[s.weight], bits, s.channel_axis)
        )
    return out


def with_quantized_activations(
    model: onnx.ModelProto, sites: Sequence[Site], quants: Dict[str, ActQuant]
) -> onnx.ModelProto:
    """A copy of ``model`` with an activation fake-quantizer on each site's input.

    8-bit quantizers are real ``QuantizeLinear``/``DequantizeLinear`` pairs (what
    :mod:`onnxsim.quant_verify` analyses); other bit widths are emulated with
    ``Div``/``Round``/``Clip``/``Mul`` in float. The quantizer feeds *only* the site's node.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    new_nodes: Dict[int, List[onnx.NodeProto]] = {}
    for si, s in enumerate(sites):
        aq = quants[s.name]
        tag = f"__qs{si}_"
        h = onnx.helper
        if aq.bits == 8:
            g.initializer.append(
                numpy_helper.from_array(np.float32(aq.scale), tag + "s")
            )
            g.initializer.append(
                numpy_helper.from_array(np.uint8(aq.zero_point), tag + "z")
            )
            nodes = [
                h.make_node(
                    "QuantizeLinear",
                    [s.input, tag + "s", tag + "z"],
                    [tag + "q"],
                    name=tag + "Q",
                ),
                h.make_node(
                    "DequantizeLinear",
                    [tag + "q", tag + "s", tag + "z"],
                    [tag + "d"],
                    name=tag + "D",
                ),
            ]
        else:
            for nm, v in (
                ("s", aq.scale),
                ("z", aq.zero_point),
                ("lo", 0.0),
                ("hi", float(aq.qmax)),
            ):
                g.initializer.append(numpy_helper.from_array(np.float32(v), tag + nm))
            nodes = [
                h.make_node("Div", [s.input, tag + "s"], [tag + "a"], name=tag + "div"),
                h.make_node("Round", [tag + "a"], [tag + "b"], name=tag + "rnd"),
                h.make_node(
                    "Add", [tag + "b", tag + "z"], [tag + "c"], name=tag + "addz"
                ),
                h.make_node(
                    "Clip",
                    [tag + "c", tag + "lo", tag + "hi"],
                    [tag + "e"],
                    name=tag + "clip",
                ),
                h.make_node(
                    "Sub", [tag + "e", tag + "z"], [tag + "f"], name=tag + "subz"
                ),
                h.make_node(
                    "Mul", [tag + "f", tag + "s"], [tag + "d"], name=tag + "mul"
                ),
            ]
        new_nodes[s.node_index] = nodes
        node = g.node[s.node_index]
        node.input[0] = tag + "d"
    rebuilt: List[onnx.NodeProto] = []
    for k, n in enumerate(g.node):
        for q in new_nodes.get(k, []):
            rebuilt.append(q)
        rebuilt.append(n)
    # the inserted nodes consume tensors produced earlier in the graph and are emitted right
    # before their consumer, so topological order is preserved
    nodes_copy = [copy.deepcopy(n) for n in rebuilt]
    del g.node[:]
    g.node.extend(nodes_copy)
    return out


# --------------------------------------------------------------------------
# Differentiable executor
# --------------------------------------------------------------------------


def _torch():
    import torch

    return torch


class TorchGraph:
    """Run an ONNX graph with torch ops so gradients flow (CNN and transformer-encoder op set).

    ``run(feeds, capture=..., grad_weights=...)`` returns ``(outputs, captured)``; weights named in
    ``grad_weights`` are fresh leaf tensors with ``requires_grad`` set (returned in
    ``self.weights``), and tensors named in ``capture`` keep their graph (``retain_grad``) so
    ``.grad`` is available after ``backward``.
    """

    def __init__(self, model: onnx.ModelProto):
        torch = _torch()
        self.model = model
        self.consts: Dict[str, Any] = {
            k: torch.from_numpy(np.ascontiguousarray(v).copy())
            for k, v in _initializers(model).items()
        }
        self.weights: Dict[str, Any] = {}
        inits = {t.name for t in model.graph.initializer}
        self.input_names = [i.name for i in model.graph.input if i.name not in inits]
        self.output_names = [o.name for o in model.graph.output]

    # -- helpers
    @staticmethod
    def _attr(n: onnx.NodeProto, name: str, default: Any = None) -> Any:
        for a in n.attribute:
            if a.name == name:
                return onnx.helper.get_attribute_value(a)
        return default

    def run_with_edge_eps(
        self, feeds: Dict[str, Any], node_indices: Sequence[int]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run with a zero-valued gradient leaf added to the first input of each listed node.

        Several nodes can share one input tensor (q/k/v projections), so the gradient of the
        shared tensor mixes their paths. A leaf on each consuming *edge* gives the per-site
        gradient; ``self.edge_eps[k]`` is the leaf and ``self.edge_acts[k]`` the activation.
        """
        return self.run(feeds, edge_eps=node_indices)

    def run(
        self,
        feeds: Dict[str, Any],
        capture: Sequence[str] = (),
        grad_weights: Sequence[str] = (),
        edge_eps: Sequence[int] = (),
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        torch = _torch()
        env: Dict[str, Any] = dict(self.consts)
        self.weights = {}
        self.edge_eps: Dict[int, Any] = {}
        self.edge_acts: Dict[int, Any] = {}
        eps_nodes = set(edge_eps)
        for w in grad_weights:
            t = self.consts[w].detach().clone().requires_grad_(True)
            env[w] = t
            self.weights[w] = t
        for k, v in feeds.items():
            env[k] = v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v))
        captured: Dict[str, Any] = {}
        want = set(capture)
        for ni, n in enumerate(self.model.graph.node):
            args = [env[i] if i else None for i in n.input]
            if ni in eps_nodes:
                eps = torch.zeros_like(args[0]).requires_grad_(True)
                self.edge_eps[ni] = eps
                self.edge_acts[ni] = args[0]
                args[0] = args[0] + eps
            res = self._op(n, args)
            if not isinstance(res, tuple):
                res = (res,)
            for o, r in zip(n.output, res):
                if o:
                    env[o] = r
        for name in want:
            t = env[name]
            if t.requires_grad:
                t.retain_grad()
            captured[name] = t
        return {o: env[o] for o in self.output_names}, captured

    def _op(self, n: onnx.NodeProto, a: List[Any]) -> Any:  # noqa: C901 - one dispatch table
        torch = _torch()
        F = torch.nn.functional
        t = n.op_type
        at = lambda k, d=None: self._attr(n, k, d)  # noqa: E731
        if t == "Conv":
            x, w = a[0], a[1]
            b = a[2] if len(a) > 2 else None
            pads = list(at("pads", [0] * (2 * (x.ndim - 2))))
            nd = x.ndim - 2
            if nd != 2:
                raise NotImplementedError(
                    f"Conv with {nd} spatial dims (node {n.name!r})"
                )
            if pads[0] != pads[2] or pads[1] != pads[3]:
                x = F.pad(x, [pads[1], pads[3], pads[0], pads[2]])
                pads = [0, 0, 0, 0]
            return F.conv2d(
                x, w, b, stride=tuple(at("strides", [1, 1])), padding=(pads[0], pads[1]),
                dilation=tuple(at("dilations", [1, 1])), groups=int(at("group", 1)),
            )  # fmt: skip
        if t == "Gemm":
            x, w = a[0], a[1]
            if at("transA", 0):
                x = x.T
            if at("transB", 0):
                w = w.T
            y = float(at("alpha", 1.0)) * (x @ w)
            if len(a) > 2 and a[2] is not None:
                y = y + float(at("beta", 1.0)) * a[2]
            return y
        if t == "MatMul":
            return torch.matmul(a[0], a[1])
        if t == "Add":
            return a[0] + a[1]
        if t == "Sub":
            return a[0] - a[1]
        if t == "Mul":
            return a[0] * a[1]
        if t == "Div":
            if not a[0].is_floating_point() and not a[1].is_floating_point():
                return torch.div(a[0], a[1], rounding_mode="trunc")
            return a[0] / a[1]
        if t == "Pow":
            return torch.pow(a[0], a[1])
        if t == "Neg":
            return -a[0]
        if t == "Sqrt":
            return torch.sqrt(a[0])
        if t == "Exp":
            return torch.exp(a[0])
        if t == "Erf":
            return torch.erf(a[0])
        if t == "Relu":
            return torch.relu(a[0])
        if t == "Sigmoid":
            return torch.sigmoid(a[0])
        if t == "Tanh":
            return torch.tanh(a[0])
        if t == "Clip":
            lo = a[1] if len(a) > 1 and a[1] is not None else at("min", None)
            hi = a[2] if len(a) > 2 and a[2] is not None else at("max", None)
            return torch.clamp(a[0], min=lo, max=hi)
        if t == "HardSigmoid":
            return torch.clamp(
                float(at("alpha", 0.2)) * a[0] + float(at("beta", 0.5)), 0.0, 1.0
            )
        if t == "HardSwish":
            return F.hardswish(a[0])
        if t == "Softmax":
            return torch.softmax(a[0], dim=int(at("axis", -1)))
        if t == "LogSoftmax":
            return torch.log_softmax(a[0], dim=int(at("axis", -1)))
        if t == "LayerNormalization":
            x = a[0]
            axis = int(at("axis", -1))
            axis = axis + x.ndim if axis < 0 else axis
            b = a[2] if len(a) > 2 else None
            return F.layer_norm(
                x, tuple(x.shape[axis:]), a[1], b, float(at("epsilon", 1e-5))
            )
        if t == "BatchNormalization":
            x = a[0]
            shp = [1, -1] + [1] * (x.ndim - 2)
            s = a[1] / torch.sqrt(a[4] + float(at("epsilon", 1e-5)))
            return (x - a[3].reshape(shp)) * s.reshape(shp) + a[2].reshape(shp)
        if t in ("MaxPool", "AveragePool"):
            x = a[0]
            k = tuple(at("kernel_shape"))
            s = tuple(at("strides", [1] * len(k)))
            p = list(at("pads", [0] * (2 * len(k))))
            if p[0] != p[2] or p[1] != p[3]:
                x = F.pad(
                    x,
                    [p[1], p[3], p[0], p[2]],
                    value=float("-inf") if t == "MaxPool" else 0.0,
                )
                p = [0, 0, 0, 0]
            if t == "MaxPool":
                return F.max_pool2d(
                    x, k, s, (p[0], p[1]), ceil_mode=bool(at("ceil_mode", 0))
                )
            return F.avg_pool2d(
                x, k, s, (p[0], p[1]), ceil_mode=bool(at("ceil_mode", 0)),
                count_include_pad=bool(at("count_include_pad", 0)),
            )  # fmt: skip
        if t == "GlobalAveragePool":
            return a[0].mean(dim=tuple(range(2, a[0].ndim)), keepdim=True)
        if t == "Flatten":
            ax = int(at("axis", 1))
            ax = ax + a[0].ndim if ax < 0 else ax
            return a[0].reshape(int(np.prod(a[0].shape[:ax])) if ax else 1, -1)
        if t == "Reshape":
            shape = [int(v) for v in a[1].tolist()]
            shape = [a[0].shape[i] if v == 0 else v for i, v in enumerate(shape)]
            return a[0].reshape(shape)
        if t == "Transpose":
            perm = at("perm")
            return (
                a[0].permute(*perm)
                if perm
                else a[0].permute(*reversed(range(a[0].ndim)))
            )
        if t == "Concat":
            return torch.cat(a, dim=int(at("axis")))
        if t in ("Identity", "Dropout"):
            return a[0]
        if t == "Cast":
            return a[0].to(_ONNX_TO_TORCH[int(at("to"))](torch))
        if t == "Constant":
            return torch.from_numpy(numpy_helper.to_array(at("value")).copy())
        if t == "Shape":
            return torch.tensor(list(a[0].shape), dtype=torch.long)
        if t == "Unsqueeze":
            axes = [int(v) for v in (a[1].tolist() if len(a) > 1 else at("axes"))]
            x = a[0]
            out_rank = x.ndim + len(axes)  # negative axes count from the *output* rank
            for ax in sorted(v + out_rank if v < 0 else v for v in axes):
                x = x.unsqueeze(ax)
            return x
        if t == "Squeeze":
            sq_axes = a[1].tolist() if len(a) > 1 and a[1] is not None else at("axes")
            return (
                a[0].squeeze()
                if sq_axes is None
                else a[0].squeeze(tuple(int(v) for v in sq_axes))
            )
        if t == "ReduceMean":
            rm_axes = (
                at("axes")
                if at("axes") is not None
                else (a[1].tolist() if len(a) > 1 else None)
            )
            return (
                a[0].mean(dim=tuple(rm_axes), keepdim=bool(at("keepdims", 1)))
                if rm_axes
                else a[0].mean()
            )
        if t == "Gather":
            ax = int(at("axis", 0))
            idx = a[1].long()
            out = torch.index_select(a[0], ax, idx.reshape(-1))
            return out.reshape(*a[0].shape[:ax], *idx.shape, *a[0].shape[ax + 1 :])
        if t == "Slice":
            x = a[0]
            starts, ends = a[1].tolist(), a[2].tolist()
            axes = (
                a[3].tolist()
                if len(a) > 3 and a[3] is not None
                else list(range(len(starts)))
            )
            steps = (
                a[4].tolist() if len(a) > 4 and a[4] is not None else [1] * len(starts)
            )
            for ax, st, en, sp in zip(axes, starts, ends, steps):
                idx = [slice(None)] * x.ndim
                idx[ax] = slice(int(st), int(min(en, 2**62)), int(sp))
                x = x[tuple(idx)]
            return x
        if t == "Expand":
            shp = torch.broadcast_shapes(
                tuple(a[0].shape), tuple(int(v) for v in a[1].tolist())
            )
            return a[0].expand(shp)
        if t == "Where":
            return torch.where(a[0].bool(), a[1], a[2])
        if t == "Equal":
            return a[0] == a[1]
        if t == "Greater":
            return a[0] > a[1]
        if t == "GreaterOrEqual":
            return a[0] >= a[1]
        if t == "Less":
            return a[0] < a[1]
        if t == "LessOrEqual":
            return a[0] <= a[1]
        if t == "And":
            return a[0].bool() & a[1].bool()
        if t == "Or":
            return a[0].bool() | a[1].bool()
        if t == "Not":
            return ~a[0].bool()
        if t == "ConstantOfShape":
            v = at("value")
            fill = (
                numpy_helper.to_array(v).reshape(-1)[0]
                if v is not None
                else np.float32(0)
            )
            return torch.full(
                [int(s) for s in a[0].tolist()],
                fill.item(),
                dtype=torch.from_numpy(np.asarray(fill)).dtype,
            )
        if t == "Split":
            axis = int(at("axis", 0))
            sizes = a[1].tolist() if len(a) > 1 and a[1] is not None else at("split")
            return tuple(torch.split(a[0], [int(s) for s in sizes], dim=axis))
        raise NotImplementedError(
            f"quant_sensitivity: ONNX op {t} (node {n.name!r}) has no torch rule"
        )


def _onnx_dtype_table() -> Dict[int, Any]:
    P = onnx.TensorProto
    return {
        P.FLOAT: lambda torch: torch.float32,
        P.DOUBLE: lambda torch: torch.float64,
        P.INT64: lambda torch: torch.long,
        P.INT32: lambda torch: torch.int32,
        P.BOOL: lambda torch: torch.bool,
        P.FLOAT16: lambda torch: torch.float16,
    }


_ONNX_TO_TORCH = _onnx_dtype_table()


# --------------------------------------------------------------------------
# Brute-force measurement (the quantity the estimators try to predict)
# --------------------------------------------------------------------------


def _ort_session(model: onnx.ModelProto):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )


def _static_batch(model: onnx.ModelProto) -> Optional[int]:
    """The batch size the graph is fixed to, or ``None`` when the first axis is dynamic."""
    inits = {t.name for t in model.graph.initializer}
    for i in model.graph.input:
        if i.name in inits:
            continue
        dims = i.type.tensor_type.shape.dim
        if dims and dims[0].HasField("dim_value") and dims[0].dim_value > 0:
            return int(dims[0].dim_value)
        return None
    return None


def run_logits(
    model: onnx.ModelProto,
    data: Feeds,
    output: Optional[str] = None,
    batch: int = 64,
    sess=None,
) -> np.ndarray:
    """Output ``output`` (default: the first) of ``model`` over every sample of ``data``."""
    sess = sess or _ort_session(model)
    name = output or sess.get_outputs()[0].name
    n = len(next(iter(data.values())))
    fixed = _static_batch(model)
    b = fixed or batch
    parts = []
    for i in range(0, n, b):
        parts.append(sess.run([name], {k: v[i : i + b] for k, v in data.items()})[0])
    return np.concatenate(parts, axis=0)


def logit_metrics(
    ref: np.ndarray, got: np.ndarray, labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Mean KL(softmax(ref) || softmax(got)), top-1 agreement, and accuracy when ``labels`` given."""
    ref = ref.astype(np.float64).reshape(len(ref), -1)
    got = got.astype(np.float64).reshape(len(got), -1)

    def logsm(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    lr, lg = logsm(ref), logsm(got)
    kl = (np.exp(lr) * (lr - lg)).sum(axis=1)
    out = {
        "kl": float(kl.mean()),
        "agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
    }
    if labels is not None:
        out["accuracy"] = float((got.argmax(1) == labels).mean())
    return out


def per_sample_kl(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    ref = ref.astype(np.float64).reshape(len(ref), -1)
    got = got.astype(np.float64).reshape(len(got), -1)

    def logsm(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    lr, lg = logsm(ref), logsm(got)
    return (np.exp(lr) * (lr - lg)).sum(axis=1)


def as_groups(
    sites: Sequence[Site], groups: Optional[Dict[str, Sequence[str]]] = None
) -> Dict[str, List[Site]]:
    """``{group name: [sites]}``; without ``groups`` every site is its own group."""
    by_name = {s.name: s for s in sites}
    if groups is None:
        return {s.name: [s] for s in sites}
    return {g: [by_name[n] for n in members] for g, members in groups.items()}


def calibrate_activations(
    model: onnx.ModelProto,
    sites: Sequence[Site],
    calib: Feeds,
    bits: int,
    percentile: float = 99.99,
    batch: int = 64,
    max_values: int = 2_000_000,
    seed: int = 0,
) -> Dict[str, ActQuant]:
    """A per-tensor affine quantizer for each site's input, from ``calib`` activations."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    wanted = sorted({s.input for s in sites})
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(w) for w in wanted)
    sess = _ort_session(m)
    n = len(next(iter(calib.values())))
    b = _static_batch(model) or batch
    rng = np.random.default_rng(seed)
    pool: Dict[str, List[np.ndarray]] = {w: [] for w in wanted}
    per_batch = max(1, max_values // max(1, math.ceil(n / b)))
    for i in range(0, n, b):
        outs = sess.run(wanted, {k: v[i : i + b] for k, v in calib.items()})
        for w, o in zip(wanted, outs):
            flat = np.asarray(o).reshape(-1)
            take = (
                flat
                if flat.size <= per_batch
                else flat[rng.integers(0, flat.size, per_batch)]
            )
            pool[w].append(take)
    cal = {
        w: calibrate_activation(np.concatenate(v), bits, percentile)
        for w, v in pool.items()
    }
    return {s.name: cal[s.input] for s in sites}


def _variant(
    model: onnx.ModelProto,
    members: Sequence[Site],
    kind: str,
    bits: int,
    quants: Optional[Dict[str, ActQuant]],
) -> onnx.ModelProto:
    if kind == "weights":
        return with_quantized_weights(model, members, bits)
    if quants is None:
        raise ValueError(
            "activation quantization needs calibrated quantizers (calibrate_activations)"
        )
    return with_quantized_activations(model, members, quants)


def measure(
    model: onnx.ModelProto,
    groups: Dict[str, List[Site]],
    kind: str,
    bits: int,
    eval_data: Feeds,
    labels: Optional[np.ndarray] = None,
    quants: Optional[Dict[str, ActQuant]] = None,
    output: Optional[str] = None,
    batch: int = 64,
    ref: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Brute-force sensitivity: quantize ONE group at a time and measure it on ``eval_data``.

    Returns ``{group: {"kl", "agreement", ["accuracy"], "kl_samples"}}`` (``kl_samples`` is the
    per-sample KL, for bootstrap intervals). This is the ground truth the estimators approximate.
    """
    ref = run_logits(model, eval_data, output, batch) if ref is None else ref
    out: Dict[str, Dict[str, Any]] = {}
    for g, members in groups.items():
        v = _variant(model, members, kind, bits, quants)
        got = run_logits(v, eval_data, output, batch)
        m: Dict[str, Any] = dict(logit_metrics(ref, got, labels))
        m["kl_samples"] = per_sample_kl(ref, got)
        out[g] = m
    return out


def evaluate_selection(
    model: onnx.ModelProto,
    groups: Dict[str, List[Site]],
    keep_float: Sequence[str],
    kind: str,
    bits: int,
    eval_data: Feeds,
    labels: Optional[np.ndarray] = None,
    quants: Optional[Dict[str, ActQuant]] = None,
    output: Optional[str] = None,
    batch: int = 64,
    ref: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Quantize every group *except* ``keep_float`` at once and report the metrics."""
    ref = run_logits(model, eval_data, output, batch) if ref is None else ref
    members = [s for g, ms in groups.items() if g not in set(keep_float) for s in ms]
    v = _variant(model, members, kind, bits, quants) if members else model
    return logit_metrics(ref, run_logits(v, eval_data, output, batch), labels)


# --------------------------------------------------------------------------
# Rank correlation
# --------------------------------------------------------------------------


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties); ``nan`` if either is constant."""
    ra, rb = _avg_ranks(np.asarray(a)), _avg_ranks(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def kendall(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall tau-b; ``nan`` if either input is constant."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / denom if denom else float("nan")


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


@dataclasses.dataclass
class SensitivityReport:
    """Per-group sensitivity scores from each requested estimator (higher = more sensitive)."""

    groups: List[str]
    members: Dict[str, List[str]]
    kind: str
    bits: int
    scores: Dict[str, np.ndarray]
    skipped: Dict[str, str]  # method (or "method:group") -> why it produced no score
    seconds: Dict[str, float]
    notes: List[str]

    def ranking(self, method: str) -> List[str]:
        """Groups from most to least sensitive by ``method`` (unscored groups last)."""
        s = self.scores[method]
        order = sorted(
            range(len(s)),
            key=lambda i: (math.isnan(s[i]), -s[i] if not math.isnan(s[i]) else 0),
        )
        return [self.groups[i] for i in order]

    def table(self, top: Optional[int] = None) -> str:
        meths = list(self.scores)
        rows = [f"{'group':34s} " + " ".join(f"{m[:13]:>13s}" for m in meths)]
        order = self.ranking(meths[0]) if meths else self.groups
        for g in order[:top]:
            i = self.groups.index(g)
            rows.append(
                f"{g[:34]:34s} " + " ".join(f"{self.scores[m][i]:13.4g}" for m in meths)
            )
        for k, v in self.skipped.items():
            rows.append(f"  skipped {k}: {v}")
        return "\n".join(rows)


def select_float_sites(report: SensitivityReport, k: int, method: str) -> List[str]:
    """The ``k`` groups ``method`` says are most sensitive: the ones to leave in float."""
    if method not in report.scores:
        raise KeyError(
            f"method {method!r} was not computed; have {sorted(report.scores)}"
        )
    return report.ranking(method)[: max(0, k)]


def _site_perturbations(
    model: onnx.ModelProto, sites: Sequence[Site], bits: int
) -> Dict[str, np.ndarray]:
    consts = _initializers(model)
    return {
        s.name: quantize_weights(consts[s.weight], bits, s.channel_axis)
        - consts[s.weight].astype(np.float32)
        for s in sites
    }


def _sample_labels(logits, mode: str, labels, gen):
    torch = _torch()
    if mode == "true" and labels is not None:
        return int(labels)
    p = torch.softmax(logits.detach(), dim=-1)
    if mode == "argmax":
        return int(p.argmax(-1).reshape(-1)[0])
    return int(torch.multinomial(p.reshape(-1, p.shape[-1])[0], 1, generator=gen))


def _flat_logits(out: Any) -> Any:
    return out.reshape(out.shape[0], -1)


def _estimate_fisher_taylor(
    model: onnx.ModelProto,
    sites: List[Site],
    groups: Dict[str, List[Site]],
    kind: str,
    bits: int,
    quants: Optional[Dict[str, ActQuant]],
    calib: Feeds,
    labels: Optional[np.ndarray],
    output: Optional[str],
    n_samples: int,
    label_mode: str,
    seed: int,
    want: Sequence[str],
) -> Dict[str, np.ndarray]:
    """``fisher`` (per-sample squared first-order change of log p) and ``taylor`` (batch |g . delta|)."""
    torch = _torch()
    graph = TorchGraph(model)
    out_name = output or graph.output_names[0]
    gen = torch.Generator().manual_seed(seed)
    n = min(n_samples, len(next(iter(calib.values()))))
    delta_w = (
        {
            k: torch.from_numpy(v)
            for k, v in _site_perturbations(model, sites, bits).items()
        }
        if kind == "weights"
        else {}
    )
    fisher = {g: 0.0 for g in groups}
    taylor_batch = {g: 0.0 for g in groups}
    wnames = sorted({s.weight for s in sites})
    for i in range(n):
        feeds = {k: torch.from_numpy(v[i : i + 1]) for k, v in calib.items()}
        if kind == "weights":
            outs, _ = graph.run(feeds, grad_weights=wnames)
            leaves = graph.weights
        else:
            outs, _ = graph.run_with_edge_eps(feeds, [s.node_index for s in sites])
            leaves = {}
        logits = _flat_logits(outs[out_name])
        y = _sample_labels(
            logits, label_mode, None if labels is None else labels[i], gen
        )
        logp = torch.log_softmax(logits, dim=-1)[0, y]
        params = (
            list(leaves.values())
            if kind == "weights"
            else [e for e in graph.edge_eps.values()]
        )
        grads = torch.autograd.grad(logp, params, allow_unused=True)
        gmap = dict(
            zip(leaves.keys() if kind == "weights" else graph.edge_eps.keys(), grads)
        )
        for g, members in groups.items():
            inner = 0.0
            for s in members:
                if kind == "weights":
                    gr = gmap.get(s.weight)
                    if gr is not None:
                        inner += float((gr * delta_w[s.name]).sum())
                else:
                    gr = gmap.get(s.node_index)
                    if gr is not None:
                        a = graph.edge_acts[s.node_index].detach()
                        aq = quants[s.name]  # type: ignore[index]
                        q = torch.clamp(
                            torch.round(a / aq.scale) + aq.zero_point, 0, aq.qmax
                        )
                        inner += float(
                            (gr * ((q - aq.zero_point) * aq.scale - a)).sum()
                        )
            fisher[g] += inner * inner
            taylor_batch[g] += abs(inner)
    res: Dict[str, np.ndarray] = {}
    if "fisher" in want:
        res["fisher"] = np.array([fisher[g] / n for g in groups])
    if "taylor" in want:
        res["taylor"] = np.array([taylor_batch[g] / n for g in groups])
    return res


def _estimate_hessian_trace(
    model: onnx.ModelProto,
    sites: List[Site],
    groups: Dict[str, List[Site]],
    bits: int,
    calib: Feeds,
    labels: Optional[np.ndarray],
    output: Optional[str],
    n_samples: int,
    probes: int,
    batch: int,
    seed: int,
) -> np.ndarray:
    """HAWQ-V2: ``tr(H_site) / n_site * ||Q(W) - W||^2``, Hutchinson trace of the CE Hessian."""
    torch = _torch()
    graph = TorchGraph(model)
    out_name = output or graph.output_names[0]
    gen = torch.Generator().manual_seed(seed)
    wnames = sorted({s.weight for s in sites})
    dnorm = {
        s.name: float((torch.from_numpy(v) ** 2).sum())
        for s, v in zip(sites, _site_perturbations(model, sites, bits).values())
    }
    n = min(n_samples, len(next(iter(calib.values()))))
    trace = {w: 0.0 for w in wnames}
    nb = 0
    batch = (
        _static_batch(model) or batch
    )  # a graph exported with a fixed batch cannot take more
    for i in range(0, n, batch):
        feeds = {
            k: torch.from_numpy(v[i : min(i + batch, n)]) for k, v in calib.items()
        }
        outs, _ = graph.run(feeds, grad_weights=wnames)
        logits = _flat_logits(outs[out_name])
        y = (
            torch.from_numpy(labels[i : min(i + batch, n)]).long()
            if labels is not None
            else logits.detach().argmax(-1)
        )
        loss = torch.nn.functional.cross_entropy(logits, y)
        ws = [graph.weights[w] for w in wnames]
        grads = torch.autograd.grad(loss, ws, create_graph=True)
        for _ in range(probes):
            vs = [
                (torch.randint(0, 2, w.shape, generator=gen) * 2 - 1).to(w.dtype)
                for w in ws
            ]
            hv = torch.autograd.grad(
                grads, ws, grad_outputs=vs, retain_graph=True, allow_unused=True
            )
            for w, v, h in zip(wnames, vs, hv):
                if h is not None:
                    trace[w] += float((v * h).sum()) / probes
        nb += 1
    by_name = {s.name: s for s in sites}
    out = []
    for g, members in groups.items():
        total = 0.0
        for s in members:
            total += trace[s.weight] / nb / by_name[s.name].n_weight * dnorm[s.name]
        out.append(total)
    return np.array(out)


def static_batch_copy(model: onnx.ModelProto, batch: int = 1) -> onnx.ModelProto:
    """A copy of ``model`` with every dynamic (symbolic) dimension of the graph I/O set to ``batch``.

    The bound-based analyses (:mod:`onnxsim.zonotope`, :mod:`onnxsim.quant_verify`,
    :mod:`onnxsim.interval`) need static shapes; a symbolic leading axis would otherwise make them
    give up (an ``inf`` bound with a "tensor shape is not static" hazard).
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    del out.graph.value_info[
        :
    ]  # stale symbolic intermediates; shape inference refills them
    for vi in list(out.graph.input) + list(out.graph.output):
        for d in vi.type.tensor_type.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                d.dim_value = batch
    return out


def calibrate_activations_interval(
    model: onnx.ModelProto,
    sites: Sequence[Site],
    input_ranges: Dict[str, Tuple],
    bits: int,
) -> Dict[str, ActQuant]:
    """Activation quantizers whose range is the *interval-analysis* range of each site input.

    The range is guaranteed for every input in the box, so these quantizers can never clip
    inside it (and :mod:`onnxsim.quant_verify` charges no saturation). The price is a coarser
    step than a data-calibrated percentile range: worst-case ranges are wider than real ones.
    """
    from . import interval

    res = interval.propagate(static_batch_copy(model), input_ranges)
    quants: Dict[str, ActQuant] = {}
    for s in sites:
        if s.input not in res.intervals:
            raise ValueError(f"no finite interval range for the input of {s.name!r}")
        lo, hi = res.hull(s.input)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"unbounded interval range for the input of {s.name!r}")
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        scale = (hi - lo) / (2**bits - 1) if hi > lo else 1.0
        quants[s.name] = ActQuant(
            scale, int(np.clip(round(-lo / scale), 0, 2**bits - 1)), bits
        )
    return quants


def _elements_estimate(model: onnx.ModelProto) -> Tuple[int, int]:
    """(largest activation element count, number of input elements) at batch 1."""
    inferred = onnx.shape_inference.infer_shapes(model)
    inits = {t.name for t in model.graph.initializer}
    biggest, n_in = 0, 0
    for vi in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        if vi.name in inits or not vi.type.HasField("tensor_type"):
            continue
        size = 1
        for d in vi.type.tensor_type.shape.dim:
            size *= d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
        biggest = max(biggest, size)
    for i in model.graph.input:
        if i.name in inits:
            continue
        size = 1
        for d in i.type.tensor_type.shape.dim:
            size *= d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
        n_in += size
    return biggest, n_in


def _estimate_certified(
    model: onnx.ModelProto,
    groups: Dict[str, List[Site]],
    kind: str,
    bits: int,
    quants: Optional[Dict[str, ActQuant]],
    input_ranges: Optional[Dict[str, Tuple]],
    max_elements: int,
    skipped: Dict[str, str],
) -> np.ndarray:
    scores = np.full(len(groups), np.nan)
    if input_ranges is None:
        skipped["certified"] = "needs input_ranges (a finite box for every real input)"
        return scores
    model = static_batch_copy(model)
    biggest, n_in = _elements_estimate(model)
    if biggest * n_in > max_elements:
        skipped["certified"] = (
            f"graph too large for dense zonotope analysis: {biggest} activation elements x "
            f"{n_in} input symbols = {biggest * n_in:.2e} > max_certified_elements={max_elements:.0e}"
        )
        return scores
    from . import quant_verify, zonotope

    for gi, (g, members) in enumerate(groups.items()):
        try:
            if kind == "weights":
                v = with_quantized_weights(model, members, bits)
                scores[gi] = max(
                    float(np.max(b))
                    for b in zonotope.bound_difference(
                        model, v, input_ranges
                    ).max_abs.values()
                )
            else:
                if quants is None or any(quants[s.name].bits != 8 for s in members):
                    skipped[f"certified:{g}"] = (
                        "activation quantizers must be 8-bit (QuantizeLinear/DequantizeLinear)"
                    )
                    continue
                v = with_quantized_activations(model, members, quants)
                scores[gi] = float(
                    quant_verify.verify(model, v, input_ranges, breakdown=False).worst
                )
        except (
            Exception
        ) as e:  # a sound analysis that cannot run is reported, not scored
            skipped[f"certified:{g}"] = f"{type(e).__name__}: {str(e)[:100]}"
    return scores


METHODS = (
    "weight_err",
    "weight_err_rel",
    "act_scale",
    "fisher",
    "taylor",
    "hessian_trace",
    "certified",
)


def rank(
    model: onnx.ModelProto,
    sites: Optional[Sequence[Site]] = None,
    groups: Optional[Dict[str, Sequence[str]]] = None,
    calib: Optional[Feeds] = None,
    kind: str = "weights",
    bits: int = 4,
    methods: Sequence[str] = ("weight_err", "fisher"),
    labels: Optional[np.ndarray] = None,
    output: Optional[str] = None,
    act_bits: int = 8,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    n_samples: int = 128,
    label_mode: str = "sample",
    probes: int = 8,
    hessian_batch: int = 16,
    max_certified_elements: int = 200_000_000,
    percentile: float = 99.99,
    quants: Optional[Dict[str, ActQuant]] = None,
    seed: int = 0,
) -> SensitivityReport:
    """Score how sensitive each site/group is to quantization, with each of ``methods``.

    :param calib: ``{input name: array}`` with a leading sample axis (needed by ``fisher``,
        ``taylor``, ``hessian_trace`` and for activation quantizers; ``label_mode="true"`` and
        ``taylor``/``hessian_trace`` also use ``labels`` when given).
    :param kind: ``"weights"`` (per-channel symmetric, ``bits``) or ``"activations"`` (per-tensor
        affine, ``act_bits``, calibrated from ``calib``).
    :param label_mode: how ``fisher`` draws ``y`` for ``log p(y | x)``: ``"sample"`` from the
        model's own distribution (true Fisher), ``"argmax"``, or ``"true"`` (needs ``labels``).
    """
    if kind not in ("weights", "activations"):
        raise ValueError("kind must be 'weights' or 'activations'")
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {METHODS}")
    all_sites = list(sites) if sites is not None else find_sites(model)
    grp = as_groups(all_sites, groups)
    used = [s for ms in grp.values() for s in ms]
    scores: Dict[str, np.ndarray] = {}
    skipped: Dict[str, str] = {}
    seconds: Dict[str, float] = {}
    notes: List[str] = []
    if kind == "activations" and quants is None and calib is not None:
        quants = calibrate_activations(model, used, calib, act_bits, percentile)
    bits_used = bits if kind == "weights" else act_bits

    def timed(name, fn):
        t0 = time.time()
        r = fn()
        seconds[name] = time.time() - t0
        return r

    consts = _initializers(model)
    for m in methods:
        if m in ("weight_err", "weight_err_rel"):
            if kind != "weights":
                skipped[m] = "weights only"
                continue
            pert = _site_perturbations(model, used, bits)

            def werr(m=m, pert=pert):
                out = []
                for ms in grp.values():
                    tot = 0.0
                    for s in ms:
                        d = float(np.linalg.norm(pert[s.name]))
                        tot += (
                            d / float(np.linalg.norm(consts[s.weight]))
                            if m == "weight_err_rel"
                            else d
                        )
                    out.append(tot)
                return np.array(out)

            scores[m] = timed(m, werr)
        elif m == "act_scale":
            if kind != "activations" or quants is None:
                skipped[m] = "activations only (needs calib)"
                continue
            scores[m] = timed(
                m,
                lambda: np.array(
                    [sum(quants[s.name].scale for s in ms) for ms in grp.values()]
                ),
            )
        elif m in ("fisher", "taylor"):
            if calib is None:
                skipped[m] = "needs calib"
                continue
            if kind == "activations" and quants is None:
                skipped[m] = "needs activation quantizers"
                continue
            if m == "taylor" and "fisher" in methods:
                continue  # computed together with fisher below
            want = [x for x in ("fisher", "taylor") if x in methods]
            try:
                res = timed(
                    "+".join(want),
                    lambda: _estimate_fisher_taylor(
                        model, used, grp, kind, bits, quants, calib, labels, output,
                        n_samples, label_mode, seed, want,
                    ),
                )  # fmt: skip
            except NotImplementedError as e:
                skipped[m] = str(e)
                continue
            scores.update(res)
        elif m == "hessian_trace":
            if kind != "weights" or calib is None:
                skipped[m] = "weights only; needs calib"
                continue
            try:
                scores[m] = timed(
                    m,
                    lambda: _estimate_hessian_trace(
                        model,
                        used,
                        grp,
                        bits,
                        calib,
                        labels,
                        output,
                        n_samples,
                        probes,
                        hessian_batch,
                        seed,
                    ),
                )
            except NotImplementedError as e:
                skipped[m] = str(e)
        elif m == "certified":
            scores[m] = timed(
                m,
                lambda: _estimate_certified(
                    model,
                    grp,
                    kind,
                    bits_used,
                    quants,
                    input_ranges,
                    max_certified_elements,
                    skipped,
                ),
            )
            if np.all(np.isnan(scores[m])):
                del scores[m]
    return SensitivityReport(
        groups=list(grp),
        members={g: [s.name for s in ms] for g, ms in grp.items()},
        kind=kind,
        bits=bits_used,
        scores=scores,
        skipped=skipped,
        seconds=seconds,
        notes=notes,
    )
