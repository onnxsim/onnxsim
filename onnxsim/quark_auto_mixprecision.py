"""Automatic mixed precision for QDQ models, shaped after AMD Quark's
``AutoMixprecisionConfig`` flow (``quark.onnx.algorithm.mprecision``).
Independent implementation: Quark's source was read for the contract and the
metric definitions, not copied.

Given a float model, :func:`auto_mixprecision` quantizes it with
:func:`onnxsim.full_qdq.quantize_full_qdq` at a *base* activation precision,
then moves layers ("candidates") to a *target* precision one at a time:

1. **Baseline score**: ``metric(float_out, quantized_out)``; lower is better.
2. **Sensitivity**: each candidate is moved to the target precision *on its
   own*; its score is the metric of that model. Candidates are ranked
   ascending (closest to float first), for either objective.
3. **Greedy mixing**, walking the ranking and keeping candidates moved
   cumulatively. ``metric_threshold``:

   - ``None``: sensitivity analysis only, the baseline model is returned;
   - ``0`` (Quark's default): the threshold is disabled, every candidate moves;
   - ``optimize="speed"``: the baseline is the *higher* precision and the
     target the lower one -- keep moving candidates while the score stays
     ``<= threshold``, and undo the first one that pushes it above (stop);
   - ``optimize="quality"``: the baseline is the *lower* precision and the
     target the higher one -- keep moving candidates until the score drops
     to ``<= threshold`` (stop).

   With a non-zero threshold, "speed" returns the baseline unchanged if its
   score already exceeds the threshold, and "quality" if it already meets it.

A candidate is a node of one of ``target_op_types``; moving it to the target
precision sets the dtype of its float activation inputs and its output (and of
a directly-following Relu's output, which :func:`quantize_full_qdq` folds into
the output quantizer). Tensors at precision boundaries simply keep their own
Q/DQ pair, so no extra boundary nodes are needed.

**Forms of the target.** ``targets`` is a list of ``(dtype, symmetric)``
precisions; with one entry every candidate moves to it, with several each
candidate is scored under every one and moves to the best-scoring (Quark's
list-of-``QLayerConfig`` mode); ``candidate_targets`` pins named candidates to
an entry (Quark's ``{QLayerConfig: [names]}`` form -- candidates it does not
name use entry 0). ``subgraphs`` (Quark's ``subgraph_json``, see
:func:`parse_subgraph_json`) makes each group of nodes one candidate that is
scored and moved together. ``cache_file`` stores the sensitivity ranking in
Quark's JSON schema (a candidate's ``"enabled": false`` pins it) and reuses it
while the model / configuration fingerprint matches; ``worker_num`` scores
candidates on that many threads; ``no_input_qdq_shared`` keeps nodes whose
input activation is read by several nodes out of the mixing step.

**Scope.** What a candidate moves is a :class:`TargetSpec`, read from Quark's
``QLayerConfig``: the activation inputs / outputs, the constant weight and the
constant bias, each at a precision of any kind -- an integer type (a
power-of-two scale included), ``float16`` / ``bfloat16``, or a BFP / MX block
format. As in Quark, the candidate is not re-quantized: the *quantized
baseline* is edited in place by :class:`onnxsim.quark_mixing.QuarkMixer` (a
port of ``MixingStrategy``: new scales and zero points from the calibrated
range, the float / dequantized constant, or Quark's fake ``[0, 1]`` range for
float / block baselines; a Q/DQ pair swapped for a ``BFPQuantizeDequantize`` /
``MXQuantizeDequantize`` node and back; int32 bias scales refreshed to
``input_scale * weight_scale`` with truncated codes), which is what makes the
result bit-identical to Quark's whichever way the precisions go.
:func:`auto_mixprecision_from_baseline` runs the same flow on a baseline that
is not integer QDQ (``float16`` / ``bfloat16`` / BFP / MX fake-quantized
models from :mod:`onnxsim.quark_fakequant_graph`). The baseline is shaped like
Quark's where the mixing step can tell: an unpromoted layer keeps its
per-tensor int32 bias scale as a one-element vector, and the output quantizer
of a pass-through op (``Transpose``, ``Reshape``, ``MaxPool``, ``Split``, ...)
reads the *same* scale / zero point initializers as the quantizer behind its
input, so ``shared_param_mode`` (``"propagate"``: the partner quantizer follows
the promoted one; ``"unshare"``: the promoted pair gets its own copy) behaves
as in Quark. Candidates are scored with ONNX Runtime's graph optimizations off
(as Quark does -- it fuses QDQ into integer kernels otherwise); models with
``com.amd.quark`` custom ops run on
:func:`onnxsim.quark_fakequant_eval.run_fake_quantized`. ``dual_quant_nodes`` is
Quark's post-processing of the final mixed model
(:func:`onnxsim.quark_boundary_qdq.insert_boundary_quant_nodes`): the candidates
are scored without it and an extra Q/DQ pair (or block node) goes in front of
every stage that differs from its node's template stage, for every kind of mix.
(:func:`apply_layer_mixing` is the older, re-quantizing way to edit weights and
biases; the mixer above does not use it.)

**Sensitivity cache.** ``cache_file`` is Quark's JSON. Through
``quark_compat`` it is written under Quark's own key (a digest of the quantized
baseline's graph, the target config and the layer filters, see
:mod:`onnxsim.quark_amp_cache`), so Quark reads a ranking onnxsim wrote and
onnxsim reads one Quark wrote; called directly (``cache_key_fn=None``) the
key is this module's own fingerprint.

:func:`auto_mixprecision_blocks` is the same flow for a bfloat16 model whose
candidates move to a block format (``BF16_MIXED_BFP16`` / ``_MXINT8``).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_EPS = 1e-10
_DTYPES = ("int8", "uint8", "int16", "uint16")

#: ``outputs[sample][output_index]`` arrays; what the metric functions score.
Outputs = List[List[np.ndarray]]
MetricFn = Callable[[Outputs, Outputs], float]


# -- metrics (lower is better) ---------------------------------------------------


def _pairs(float_out: Outputs, quant_out: Outputs):
    if len(float_out) != len(quant_out):
        raise ValueError(
            "float_out and quant_out must have the same number of samples, "
            f"got {len(float_out)} vs {len(quant_out)}"
        )
    for f_sample, q_sample in zip(float_out, quant_out):
        for f, q in zip(f_sample, q_sample):
            yield f, q


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def l2_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean L2 norm of the difference over all (sample, output) pairs."""
    return _mean(
        [
            float(np.linalg.norm(np.asarray(f, np.float32) - np.asarray(q, np.float32)))
            for f, q in _pairs(float_out, quant_out)
        ]
    )


def kl_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean KL(P_float || P_quant); each output is shifted to be non-negative
    and normalized to a distribution."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float64).ravel()
        q = np.asarray(q, np.float64).ravel()
        f = f - min(f.min(), 0.0)
        q = q - min(q.min(), 0.0)
        p = f / f.sum() if f.sum() > 0 else np.ones_like(f) / f.size
        r = q / q.sum() if q.sum() > 0 else np.ones_like(q) / q.size
        vals.append(float(np.sum(p * np.log((p + _EPS) / (r + _EPS)))))
    return _mean(vals)


def cosine_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean cosine distance ``1 - cos_sim`` (0 when either norm is ~0)."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float32).ravel()
        q = np.asarray(q, np.float32).ravel()
        nf, nq = float(np.linalg.norm(f)), float(np.linalg.norm(q))
        sim = 1.0 if nf < _EPS or nq < _EPS else float(np.dot(f, q) / (nf * nq))
        vals.append(1.0 - sim)
    return _mean(vals)


def sqnr_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean *negative* SQNR in dB (use a negative threshold, e.g. ``-30``)."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float32)
        q = np.asarray(q, np.float32)
        signal = max(float(np.mean(f**2)), _EPS)
        noise = max(float(np.mean((f - q) ** 2)), _EPS)
        vals.append(-10.0 * float(np.log10(signal / noise)))
    return _mean(vals)


def psnr_metric(float_out: Outputs, quant_out: Outputs) -> float:
    """Mean *negative* PSNR in dB (peak = max |float output|)."""
    vals = []
    for f, q in _pairs(float_out, quant_out):
        f = np.asarray(f, np.float32)
        q = np.asarray(q, np.float32)
        mse = float(np.mean((f - q) ** 2)) or _EPS
        peak = float(np.max(np.abs(f))) or _EPS
        vals.append(-(20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(mse))))
    return _mean(vals)


BUILTIN_METRICS: Dict[str, MetricFn] = {
    "l2": l2_metric,
    "kl": kl_metric,
    "cosine": cosine_metric,
    "sqnr": sqnr_metric,
    "psnr": psnr_metric,
}


def resolve_metric(
    metric: str = "l2",
    distance_fn: Optional[MetricFn] = None,
    evaluate_fn: Optional[Callable[[Outputs], float]] = None,
) -> MetricFn:
    """``distance_fn`` (lower is better) wins over ``evaluate_fn`` (higher is
    better, adapted to ``evaluate(float) - evaluate(quant)``) over the named
    built-in ``metric``. Giving both callables is an error."""
    if distance_fn is not None and evaluate_fn is not None:
        raise ValueError("distance_fn and evaluate_fn are mutually exclusive")
    if distance_fn is not None:
        return distance_fn
    if evaluate_fn is not None:
        return lambda f, q: float(evaluate_fn(f)) - float(evaluate_fn(q))
    if metric not in BUILTIN_METRICS:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(BUILTIN_METRICS)}")
    return BUILTIN_METRICS[metric]


# -- the algorithm -----------------------------------------------------------------


#: ``(activation dtype, symmetric or None to keep the model's setting)``
Precision = Tuple[str, Optional[bool]]


@dataclass(frozen=True)
class TargetSpec:
    """What a candidate is moved to -- Quark's ``target_layer_config`` of one
    layer: a precision per slot, ``None`` leaving that slot untouched.

    ``inputs`` / ``outputs`` are the activation inputs / outputs of the node,
    ``weight`` its constant second operand and ``bias`` its constant bias. A
    weight (or bias) moved to a precision is re-quantized from the *already
    quantized* values (dequantized, then one per-tensor scale from their
    range), and the int32 bias scale is re-derived as ``input_scale *
    weight_scale`` afterwards (see :func:`apply_layer_mixing`)."""

    inputs: Optional[Precision] = None
    outputs: Optional[Precision] = None
    weight: Optional[Precision] = None
    bias: Optional[Precision] = None

    @property
    def as_json(self) -> list:
        return [self.inputs, self.outputs, self.weight, self.bias]


#: a bare ``(dtype, symmetric)`` moves the activation inputs and outputs only
Target = Union[Precision, TargetSpec]


def as_target_spec(t: Target) -> TargetSpec:
    if isinstance(t, TargetSpec):
        return t
    return TargetSpec(inputs=t, outputs=t)


@dataclass
class SensitivityResult:
    """One candidate's score when moved to a target precision on its own."""

    name: str
    nodes: List[str]
    tensors: List[str]
    score: float
    enabled: bool = True
    #: score under each entry of ``targets`` (``score`` is their minimum)
    all_config_scores: List[float] = field(default_factory=list)
    best_config_index: int = 0


@dataclass
class AutoMixprecisionResult:
    model: onnx.ModelProto
    baseline_score: float
    final_score: float
    ranked: List[SensitivityResult] = field(default_factory=list)
    moved: List[str] = field(default_factory=list)  # candidate names, in order
    #: the nodes of those candidates (``no_input_qdq_shared`` skips some), in order
    moved_nodes: List[str] = field(default_factory=list)
    threshold_reached: bool = False


# -- subgraph partitions (Quark's ``subgraph_json``) ---------------------------------


@dataclass
class SubgraphSpec:
    name: str
    start_nodes: List[str]
    end_nodes: List[str]
    resolved_nodes: List[str] = field(default_factory=list)


def _reach(
    model: onnx.ModelProto, starts: Sequence[str], ends: Sequence[str]
) -> List[str]:
    """Nodes reachable from ``starts``, not walking past an end node."""
    nodes = {n.name: n for n in model.graph.node}
    readers: Dict[str, List[str]] = {}
    for n in model.graph.node:
        for x in n.input:
            readers.setdefault(x, []).append(n.name)
    stop = set(ends)
    seen: List[str] = []
    todo = list(starts)
    while todo:
        name = todo.pop(0)
        if name in seen or name not in nodes:
            continue
        seen.append(name)
        if name in stop:
            continue
        for o in nodes[name].output:
            todo += [r for r in readers.get(o, []) if r not in seen]
    return seen


def parse_subgraph_json(
    path: Union[str, Path], float_model: onnx.ModelProto, quant_model: onnx.ModelProto
) -> List[SubgraphSpec]:
    """Quark's subgraph partition file::

        {"quantized": false, "num_subgraphs": 2,
         "subgraphs": [{"name": "a", "start_nodes": [...], "end_nodes": [...]}]}

    Each subgraph is every node reachable from its start nodes up to its end
    nodes (on ``float_model``, or on ``quant_model`` when ``"quantized"`` is
    true); nodes of a later subgraph already claimed by an earlier one are
    dropped from it; every remaining node forms a final ``__ungrouped__``
    entry. Unknown boundary nodes and a wrong ``num_subgraphs`` raise."""
    data = json.loads(Path(path).read_text())
    entries = data.get("subgraphs", [])
    declared = data.get("num_subgraphs")
    if declared is not None and declared != len(entries):
        raise ValueError(
            f"num_subgraphs={declared} does not match the number of subgraph "
            f"entries ({len(entries)})"
        )
    quantized = bool(data.get("quantized", False))
    f_names = {n.name for n in float_model.graph.node}
    q_names = {n.name for n in quant_model.graph.node}
    specs: List[SubgraphSpec] = []
    assigned: Set[str] = set()
    for e in entries:
        starts, ends = list(e["start_nodes"]), list(e["end_nodes"])
        known = q_names if quantized else f_names
        for n in starts + ends:
            if n not in known:
                raise ValueError(
                    f"subgraph {e['name']!r}: node {n!r} not found in the "
                    f"{'quantized' if quantized else 'float'} model"
                )
        resolved = _reach(quant_model if quantized else float_model, starts, ends)
        resolved = [n for n in resolved if n in q_names or quantized]
        resolved = [n for n in resolved if n not in assigned]
        assigned.update(resolved)
        specs.append(SubgraphSpec(e["name"], starts, ends, resolved))
    rest = [n.name for n in float_model.graph.node if n.name not in assigned]
    if rest:
        specs.append(SubgraphSpec("__ungrouped__", [], [], rest))
    return specs


# -- candidates ----------------------------------------------------------------------


def _node_key(n: onnx.NodeProto) -> str:
    return n.name or (n.output[0] if n.output else "")


def _node_tensors(model: onnx.ModelProto) -> Dict[str, Tuple[List[str], List[str]]]:
    """Per node: ``(activation inputs, outputs)`` -- the float tensors whose
    precision moves with it (the outputs include a directly-following Relu's
    output, which :func:`quantize_full_qdq` folds into the output quantizer)."""
    g = model.graph
    inits = {t.name for t in g.initializer}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    outputs = {o.name for o in g.output}
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for n in g.node:
        if not n.output:
            continue
        ins = [x for x in n.input if x and x not in inits]
        outs = [n.output[0]]
        users = consumers.get(n.output[0], [])
        if (
            len(users) == 1
            and users[0].op_type == "Relu"
            and n.output[0] not in outputs
        ):
            outs.append(users[0].output[0])
        out[_node_key(n)] = (ins, outs)
    return out


def _name_unnamed_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """``model`` with every unnamed node named after its first output (its
    candidate key): the in-place edits find a candidate in the quantized
    baseline by name, and the quantizer keeps node names."""
    if all(n.name for n in model.graph.node):
        return model
    m = onnx.ModelProto()
    m.CopyFrom(model)
    taken = {n.name for n in m.graph.node if n.name}
    for n in m.graph.node:
        if not n.name and n.output:
            name = n.output[0]
            while name in taken:
                name += "_"
            n.name = name
            taken.add(name)
    return m


def _candidate_nodes(
    model: onnx.ModelProto,
    target_op_types: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
    restrict: Optional[Sequence[str]] = None,
) -> List[str]:
    out: List[str] = []
    for n in model.graph.node:
        if n.op_type not in target_op_types or not n.output:
            continue
        ids = {n.name, n.output[0]} - {""}
        if restrict is not None and _node_key(n) not in restrict:
            continue
        if include and not (ids & set(include)):
            continue
        if ids & set(exclude):
            continue
        out.append(_node_key(n))
    return out


def _fingerprint(
    model: onnx.ModelProto,
    base_dtype: str,
    targets: Sequence[Target],
    candidate_targets: Dict[str, Target],
    target_op_types: Sequence[str],
    include: Sequence[str],
    exclude: Sequence[str],
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]],
) -> str:
    """Hash of everything a ranking depends on structurally (graph topology,
    precisions, op filter, partition) -- not the weight values."""
    parts = [
        f"{n.name}|{n.op_type}|{','.join(n.input)}|{','.join(n.output)}"
        for n in sorted(model.graph.node, key=lambda n: (n.name, n.op_type))
    ]
    parts.append(f"base:{base_dtype}")
    parts.append(f"targets:{json.dumps([as_target_spec(t).as_json for t in targets])}")
    parts.append(
        "pinned:"
        + json.dumps(
            {k: as_target_spec(v).as_json for k, v in candidate_targets.items()},
            sort_keys=True,
        )
    )
    parts.append("ops:" + ",".join(sorted(target_op_types)))
    parts.append("include:" + ",".join(sorted(include)))
    parts.append("exclude:" + ",".join(sorted(exclude)))
    if subgraphs is not None:
        parts.append(json.dumps([[a, list(b)] for a, b in subgraphs]))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def save_sensitivity(
    ranked: Sequence[SensitivityResult], path: Union[str, Path], key: str
) -> None:
    """Write ``ranked`` in Quark's sensitivity-cache JSON schema."""
    payload = {
        "version": "onnxsim",
        "cache_key": key,
        "results": [
            {
                "name": r.name,
                "candidate_nodes": r.nodes,
                "score": r.score,
                "all_config_scores": r.all_config_scores,
                "best_config_index": r.best_config_index,
                "enabled": r.enabled,
            }
            for r in ranked
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_sensitivity(
    path: Union[str, Path],
    key: Union[str, Sequence[str]],
    node_tensors: Dict[str, Tuple[List[str], List[str]]],
) -> Optional[List[SensitivityResult]]:
    """Read a cache written by :func:`save_sensitivity` (or by Quark); ``None``
    -- with a warning -- when its fingerprint is not ``key`` (or one of the
    ``key`` strings). Quark's own key (see :mod:`onnxsim.quark_amp_cache`) is
    among those the ``quark_compat`` flows pass, so a ranking Quark wrote for
    the same quantized baseline and configuration is read instead of
    recomputed."""
    payload = json.loads(Path(path).read_text())
    keys = [key] if isinstance(key, str) else list(key)
    if payload.get("cache_key") not in keys:
        warnings.warn(
            f"sensitivity cache {path} is stale (model or configuration "
            "changed); recomputing",
            UserWarning,
            stacklevel=3,
        )
        return None
    out = []
    for e in payload["results"]:
        tensors: List[str] = []
        for n in e["candidate_nodes"]:
            ins, outs = node_tensors.get(n, ([], []))
            tensors += [t for t in ins + outs if t not in tensors]
        out.append(
            SensitivityResult(
                e["name"],
                list(e["candidate_nodes"]),
                tensors,
                float(e["score"]),
                bool(e.get("enabled", True)),
                [float(x) for x in e.get("all_config_scores", [])],
                int(e.get("best_config_index", 0)),
            )
        )
    return out


# -- layer mixing: weights and biases (Quark's ``MixingStrategy``) -------------------

_NP_DTYPES = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
}
#: Quark's ``ONNX_INT_TYPE_RANGE`` / ``ONNX_INT_TYPE_SYMMETRIC_RANGE``
_RANGES = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
}
_SYMMETRIC_RANGES = {"int8": (-127, 127), "int16": (-32767, 32767)}
_WIDE = ("int16", "uint16")


def _dtype_name_of(inits: Dict[str, onnx.TensorProto], name: str) -> str:
    t = inits.get(name)
    if t is None:
        return "other"
    return {
        int(TensorProto.INT8): "int8",
        int(TensorProto.UINT8): "uint8",
        int(TensorProto.INT16): "int16",
        int(TensorProto.UINT16): "uint16",
        int(TensorProto.INT32): "int32",
    }.get(int(t.data_type), "other")


def _requantize_constant(
    values: np.ndarray, dtype: str, symmetric: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(codes, scale, zero_point)`` of the float32 ``values`` on ``dtype``'s
    grid with one per-tensor scale from their range -- Quark's
    ``_compute_scale_zp`` + ``_apply_scale_zp`` for a folded constant."""
    from onnxsim.full_qdq import _weight_qparams

    qmin, qmax = (_SYMMETRIC_RANGES if symmetric else {}).get(dtype) or _RANGES[dtype]
    np_dt = _NP_DTYPES[dtype]
    scale, zp = _weight_qparams(values.min(), values.max(), qmin, qmax, symmetric)
    scale_np = np.asarray(scale, dtype=np.float32).reshape(())
    zp_np = np.asarray(zp, dtype=np_dt).reshape(())
    codes = np.clip(np.round(values / scale_np + zp_np), qmin, qmax).astype(np_dt)
    return codes, scale_np, zp_np


def _check_target(spec: TargetSpec) -> None:
    """Refuse precisions neither the integer re-quantization nor
    :class:`onnxsim.quark_mixing.QuarkMixer` knows."""
    from onnxsim import quark_mixing as qm

    for prec in (spec.inputs, spec.outputs, spec.weight, spec.bias):
        if prec is None:
            continue
        try:
            kind = qm.kind_of(prec[0])
        except ValueError:
            kind = ""
        if kind == "" or (kind == "int" and prec[0] not in _NP_DTYPES):
            raise ValueError(
                f"dtypes must be {tuple(_NP_DTYPES)}, float16 / bfloat16 or a "
                f"block format (bfp16, mx4/6/9, mxint8, mxfp*), got {prec[0]!r}"
            )


def _needs_mixer(spec: TargetSpec) -> bool:
    """Whether a target is outside what re-quantizing with another integer
    dtype does: a half / block precision, or a power-of-two scale."""
    from onnxsim import quark_mixing as qm

    for prec in (spec.inputs, spec.outputs, spec.weight, spec.bias):
        if prec is None:
            continue
        if qm.kind_of(prec[0]) != "int" or qm.unpack(prec)[2]:
            return True
    return False


def _resolve_symmetry(spec: TargetSpec, act: bool, wt: bool) -> TargetSpec:
    """``spec`` with every ``symmetric=None`` replaced (``act`` for the
    activation slots, ``wt`` for weight and bias)."""

    def fix(prec: Optional[Precision], default: bool) -> Optional[Precision]:
        if prec is None or prec[1] is not None:
            return prec
        return (prec[0], default, *prec[2:])

    return TargetSpec(
        inputs=fix(spec.inputs, act),
        outputs=fix(spec.outputs, act),
        weight=fix(spec.weight, wt),
        bias=fix(spec.bias, wt),
    )


def _range_lookup(ranges: Mapping[str, Tuple[float, float]]):
    """``f(float tensor name) -> range``: the quantizer's float tensor is the
    calibrated tensor's ``<name>/f`` copy."""

    def lookup(name: str):
        r = ranges.get(name)
        if r is None and name.endswith("/f"):
            r = ranges.get(name[:-2])
        return r

    return lookup


_BIAS_NODES = ("Conv", "ConvTranspose", "Gemm")

#: ops whose output quantizer ONNX Runtime's QDQ quantizer (hence Quark's
#: baseline) points at the scale / zero point initializers of the quantizer
#: behind their input (probed; ``AveragePool`` joins them for the plain
#: quantizer, see ``_activation_rules``'s ``shared_ops``)
_QDQ_SHARING_OPS = frozenset(
    {
        "Reshape",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Split",
        "Resize",
        "Gather",
        "MaxPool",
    }
)


def _normalize_bias_params(
    model: onnx.ModelProto, wide: bool = False
) -> onnx.ModelProto:
    """Quark stores a per-tensor int32 bias scale as a one-element vector and
    its zero point as a scalar (the layers AutoMixprecision does not touch keep
    that); a per-channel one stays a vector. With 16-bit activations or weights
    (``wide``, opset below 21) its int32 bias dequantizer is the
    ``com.microsoft`` one. Edits ``model`` in place."""
    inits = {t.name: t for t in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    for n in model.graph.node:
        if n.op_type not in _BIAS_NODES or len(n.input) < 3 or not n.input[2]:
            continue
        bdq, wdq = prod.get(n.input[2]), prod.get(n.input[1])
        if (
            bdq is None
            or wdq is None
            or "DequantizeLinear"
            not in (
                bdq.op_type,
                wdq.op_type,
            )
        ):
            continue
        if bdq.op_type != "DequantizeLinear" or wdq.op_type != "DequantizeLinear":
            continue
        if len(bdq.input) < 3 or bdq.input[0] not in inits:
            continue
        if _dtype_name_of(inits, bdq.input[2]) != "int32":
            continue
        if wide:
            bdq.domain = "com.microsoft"
        if wdq.input[1] not in inits or inits[wdq.input[1]].dims not in ([], [1]):
            continue
        scale = numpy_helper.to_array(inits[bdq.input[1]])
        zp = numpy_helper.to_array(inits[bdq.input[2]])
        if scale.size < 1 or not np.all(scale == scale.reshape(-1)[0]):
            continue
        if not np.all(zp == zp.reshape(-1)[0]):
            continue
        inits[bdq.input[1]].CopyFrom(
            numpy_helper.from_array(scale.reshape(-1)[:1].copy(), bdq.input[1])
        )
        inits[bdq.input[2]].CopyFrom(
            numpy_helper.from_array(zp.reshape(-1)[0].copy(), bdq.input[2])
        )
    if wide and not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return model


def _share_qparams(model: onnx.ModelProto, ops: Sequence[str]) -> onnx.ModelProto:
    """Make the output quantizer of a pass-through op (``Transpose``,
    ``Reshape``, ``MaxPool``, ...) read the scale / zero point *initializers* of
    the quantizer behind its input, as ONNX Runtime's QDQ quantizer (hence
    Quark's baseline) does: the quantizers then share them, which is what
    ``shared_param_mode`` of the mixing step is about. Only quantizers whose
    parameters are already equal are rewired. Edits ``model`` in place."""
    inits = {t.name: t for t in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    cons: Dict[str, List[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)

    def pair_behind(tensor: str):
        dq = prod.get(tensor)
        if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 3:
            return None
        q = prod.get(dq.input[0])
        if q is None or q.op_type != "QuantizeLinear" or len(q.input) < 3:
            return None
        return q, dq

    def same(a: str, b: str) -> bool:
        ta, tb = inits.get(a), inits.get(b)
        return (
            ta is not None
            and tb is not None
            and ta.data_type == tb.data_type
            and numpy_helper.to_array(ta).shape == numpy_helper.to_array(tb).shape
            and bool(np.all(numpy_helper.to_array(ta) == numpy_helper.to_array(tb)))
        )

    for n in model.graph.node:
        if n.op_type not in ops or not n.input or not n.input[0]:
            continue
        src = pair_behind(n.input[0])
        if src is None:
            continue
        q_a = src[0]
        for o in n.output:
            users = cons.get(o, [])
            if len(users) != 1 or users[0].op_type != "QuantizeLinear":
                continue
            q_o = users[0]
            dqs = cons.get(q_o.output[0], [])
            if (
                len(dqs) != 1
                or dqs[0].op_type != "DequantizeLinear"
                or len(q_o.input) < 3
            ):
                continue
            dq_o = dqs[0]
            if len(dq_o.input) < 3 or q_o.input[1:3] != dq_o.input[1:3]:
                continue
            if not (
                same(q_o.input[1], q_a.input[1]) and same(q_o.input[2], q_a.input[2])
            ):
                continue
            for node in (q_o, dq_o):
                node.input[1], node.input[2] = q_a.input[1], q_a.input[2]
    used = {x for n in model.graph.node for x in n.input}
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return model


def _has_custom_ops(model: onnx.ModelProto) -> bool:
    return any(n.domain == "com.amd.quark" for n in model.graph.node)


def _make_runner(
    eval_data: Sequence[Dict[str, np.ndarray]],
    metric_output_index: Optional[int],
    providers: Sequence[str],
) -> Callable[[onnx.ModelProto], Outputs]:
    """``run(model) -> outputs[batch][output]`` the way Quark scores a model:
    ONNX Runtime with every graph optimization off (it otherwise fuses QDQ
    into integer kernels, which changes a quantized model's numerics); a model
    with ``com.amd.quark`` custom ops runs on
    :func:`onnxsim.quark_fakequant_eval.run_fake_quantized` (the same ONNX
    Runtime kernels, with Quark's ops evaluated bit-exactly in numpy)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    def pick(outs: List[np.ndarray]) -> List[np.ndarray]:
        return [outs[metric_output_index]] if metric_output_index is not None else outs

    def run(m: onnx.ModelProto) -> Outputs:
        if _has_custom_ops(m):
            from onnxsim.quark_fakequant_eval import run_fake_quantized

            return [pick(o) for o in run_fake_quantized(m, eval_data)]
        sess = ort.InferenceSession(m.SerializeToString(), so, providers=providers)
        return [pick(sess.run(None, batch)) for batch in eval_data]

    return run


def apply_layer_mixing(
    model: onnx.ModelProto,
    base: onnx.ModelProto,
    steps: Sequence[Tuple[str, TargetSpec]],
) -> onnx.ModelProto:
    """Move the *constants* of the nodes in ``steps`` to their target
    precision, the way Quark's ``MixingStrategy.promote`` does, on top of
    ``model`` (whose activation Q/DQ pairs already carry the target
    precisions; ``base`` is the same model with nothing moved).

    For each ``(node, spec)``, in order:

    1. **weight** (``spec.weight``): the weight's DequantizeLinear is
       re-quantized from its *dequantized* codes -- one per-tensor scale from
       their range (symmetric per ``spec.weight``) -- so moving int8 weights to
       int16 keeps their int8 resolution, and moving them to the same dtype
       turns per-channel scales into one per-tensor scale.
    2. **bias** (``spec.bias``): likewise, from the dequantized bias.
    3. **bias scale refresh**, always, for a node with an int32 constant bias:
       the scale becomes ``input_scale * weight_scale`` (the node's current
       scales) and the int32 codes are rescaled from the base bias,
       ``trunc(codes * old_scale / new_scale)`` in float32 -- Quark truncates
       rather than rounds. The refresh only reads scales at the moment the
       node moves, so a later change of its input tensor's scale is not
       reflected (as in Quark).

    Nodes without the expected Q/DQ structure are skipped."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    base_inits = {t.name: t for t in base.graph.initializer}
    prod = {o: n for n in g.node for o in n.output}
    base_prod = {o: n for n in base.graph.node for o in n.output}
    nodes = {_node_key(n): n for n in g.node}
    base_nodes = {_node_key(n): n for n in base.graph.node}
    opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
    touched: Set[str] = set()  # DequantizeLinear nodes already rewritten here
    need_ms = [False]

    def dq_of(table: Dict[str, onnx.NodeProto], name: str):
        p = table.get(name)
        return p if p is not None and p.op_type == "DequantizeLinear" else None

    def put(name: str, value: np.ndarray) -> None:
        new = numpy_helper.from_array(np.asarray(value), name)
        if name in inits:
            inits[name].CopyFrom(new)
        else:
            g.initializer.append(new)
            inits[name] = g.initializer[-1]

    def arr(table: Dict[str, onnx.TensorProto], name: str) -> np.ndarray:
        return numpy_helper.to_array(table[name])

    def source(dq: onnx.NodeProto, bdq: Optional[onnx.NodeProto]):
        """The DQ's ``(codes, scale, zero_point)`` -- the current ones once this
        call rewrote it, else the base model's."""
        if bdq is not None and dq.name not in touched:
            tab, use = base_inits, bdq
        else:
            tab, use = inits, dq
        zp = arr(tab, use.input[2]) if len(use.input) > 2 else np.int8(0)
        return arr(tab, use.input[0]), arr(tab, use.input[1]), zp

    def requantize_slot(
        dq: onnx.NodeProto, bdq: Optional[onnx.NodeProto], prec: Precision
    ) -> None:
        dtype, sym = prec
        if dtype not in _NP_DTYPES:
            raise ValueError(f"unsupported constant precision {dtype!r}")
        q, scale, zp = source(dq, bdq)
        deq = (q.astype(np.float32) - zp.astype(np.float32)) * scale.astype(np.float32)
        codes, scale_np, zp_np = _requantize_constant(deq, dtype, bool(sym))
        put(dq.input[0], codes)
        put(dq.input[1], scale_np)
        if len(dq.input) < 3:
            dq.input.append(dq.input[0] + "_zero_point")
        put(dq.input[2], zp_np)
        wide = dtype in _WIDE and opset < 21
        dq.domain = "com.microsoft" if wide else ""
        need_ms[0] = need_ms[0] or wide
        touched.add(dq.name)

    for key, spec in steps:
        n, bn = nodes.get(key), base_nodes.get(key)
        if n is None or bn is None:
            continue
        slots = []
        for i, x in enumerate(n.input):
            dq = dq_of(prod, x)
            bdq = dq_of(base_prod, bn.input[i]) if i < len(bn.input) else None
            slots.append((dq, bdq))

        def const(i: int) -> bool:
            return (
                i < len(slots)
                and slots[i][0] is not None
                and slots[i][0].input[0] in inits
                and len(slots[i][0].input) > 1
            )

        if spec.weight is not None and const(1):
            requantize_slot(slots[1][0], slots[1][1], spec.weight)
        if spec.bias is not None and const(2):
            requantize_slot(slots[2][0], slots[2][1], spec.bias)
        # bias scale refresh (Quark's ``_refine_bias_scale``)
        if len(n.input) != 3 or not const(2):
            continue
        bdq, bbdq = slots[2]
        if len(bdq.input) < 3 or _dtype_name_of(inits, bdq.input[2]) != "int32":
            continue
        idq, wdq = slots[0][0], slots[1][0]
        if idq is None or wdq is None:
            continue
        input_scale = arr(inits, idq.input[1])
        weight_scale = arr(inits, wdq.input[1])
        new_scale = (input_scale * weight_scale).astype(input_scale.dtype)
        codes, old_scale, _ = source(bdq, bbdq)
        with np.errstate(invalid="ignore", over="ignore"):
            new_codes = (
                codes.astype(np.float32) * old_scale.astype(np.float32) / new_scale
            ).astype(np.int32)
        put(bdq.input[0], new_codes)
        put(bdq.input[1], new_scale)
        if arr(inits, bdq.input[2]).shape != new_scale.shape:
            put(bdq.input[2], np.zeros(new_scale.shape, np.int32))
        touched.add(bdq.name)
    # A node that did not move keeps its baseline bias even when an activation
    # input's scale changed with a neighbour (Quark edits the quantized
    # baseline in place and refreshes biases only for the nodes it moves).
    moved_keys = {k for k, _ in steps}
    for key, n in nodes.items():
        bn = base_nodes.get(key)
        if key in moved_keys or bn is None or len(n.input) < 3 or len(bn.input) < 3:
            continue
        bdq, bbdq = dq_of(prod, n.input[2]), dq_of(base_prod, bn.input[2])
        if bdq is None or bbdq is None or len(bdq.input) < 3 or len(bbdq.input) < 3:
            continue
        if bdq.input[0] not in inits or bbdq.input[0] not in base_inits:
            continue
        if _dtype_name_of(inits, bdq.input[2]) != "int32":
            continue
        for i in range(3):
            put(bdq.input[i], arr(base_inits, bbdq.input[i]))
    if need_ms[0] and not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return m


def auto_mixprecision(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    base_dtype: str,
    target_dtype: Optional[str] = None,
    target_op_types: Sequence[str] = ("Conv", "Gemm", "MatMul"),
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
    exclude_nodes: Sequence[str] = (),
    metric: str = "l2",
    metric_distance_fn: Optional[MetricFn] = None,
    metric_evaluate_fn: Optional[Callable[[Outputs], float]] = None,
    metric_threshold: Optional[float] = 0.0,
    optimize: str = "speed",
    metric_output_index: Optional[int] = 0,
    data_size: int = 0,
    per_channel: bool = True,
    method: str = "minmax",
    providers: Optional[Sequence[str]] = None,
    targets: Optional[Sequence[Target]] = None,
    candidate_targets: Optional[Mapping[str, Target]] = None,
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    cache_file: Optional[Union[str, Path]] = None,
    worker_num: int = 1,
    no_input_qdq_shared: bool = False,
    dual_quant_nodes: bool = False,
    quantize_kwargs: Optional[Dict[str, Any]] = None,
    calibrate_options: Optional[Dict[str, Any]] = None,
    post_quantize: Optional[Callable[[onnx.ModelProto], onnx.ModelProto]] = None,
    shared_param_mode: str = "propagate",
    cache_key_fn: Optional[Callable[[onnx.ModelProto], str]] = None,
) -> AutoMixprecisionResult:
    """Mixed-precision quantization of ``model`` (see the module docstring).

    :param exclude_nodes: nodes kept in float by the quantizer itself (they are
            not quantized at either precision, unlike ``exclude_layers``, which
            only stops a layer from being a mixing candidate)
    :param base_dtype: activation precision of the starting model
    :param target_dtype: precision candidates move to (a single target; give
            ``targets`` instead for several)
    :param targets: ``(dtype, symmetric)`` entries (``symmetric=None`` keeps
            the model's own setting; the activation inputs and outputs move) or
            :class:`TargetSpec` entries (inputs / outputs / weight / bias
            separately); several entries -> each candidate takes the
            best-scoring one
    :param candidate_targets: ``{candidate node name: target}`` pinned
            targets (either form), never scored; other nodes use their
            candidate's best entry of ``targets``
    :param subgraphs: ``[(name, [node names])]``: each group is one candidate
    :param cache_file: JSON file for the sensitivity ranking: reused when its
            fingerprint matches, written otherwise
    :param worker_num: threads used to score candidates
    :param dual_quant_nodes: insert Quark's boundary quantizers into the final
            model (an extra Q/DQ pair, or block node, in front of every stage
            whose precision differs from its node's template stage); off, a
            promoted node simply consumes one precision and produces the other
    :param quantize_kwargs: keyword arguments of the :func:`quantize_full_qdq`
            call that makes the baseline (``calibration_data`` / ``activation_dtype``
            / ``ranges`` / ``convert_inputs`` are taken from here); its own
            ``tensor_dtypes`` / ``tensor_symmetric`` apply to every trial. Default:
            ``per_channel`` / ``exclude_nodes`` / ``method`` only
    :param no_input_qdq_shared: skip nodes whose first activation input is
            read by more than one node in the mixing step
    :param metric_output_index: model output scored by the metric; ``None``
            scores every output
    :param data_size: use only the first ``data_size`` calibration batches for
            scoring (``0`` = all)
    :param post_quantize: applied to every :func:`quantize_full_qdq` result
            (baseline and trials) before mixing, e.g. a bias re-quantization
    :param shared_param_mode: Quark's ``"propagate"`` / ``"unshare"`` for the
            scale / zero point initializers a promoted Q/DQ pair shares with
            other nodes
    :param cache_key_fn: ``f(baseline model) -> str``, the key a
            ``cache_file`` is written under and also read with (besides this
            module's own fingerprint) -- Quark's, from
            :func:`onnxsim.quark_amp_cache.quark_cache_key`
    """
    if shared_param_mode not in ("propagate", "unshare"):
        raise ValueError("shared_param_mode must be 'propagate' or 'unshare'")
    if targets is None:
        if target_dtype is None:
            raise ValueError("target_dtype or targets is required")
        targets = [(target_dtype, None)]
        if base_dtype == target_dtype:
            raise ValueError("base_dtype and target_dtype must differ")
    targets = list(targets)
    if not targets:
        raise ValueError("targets must not be empty")
    if base_dtype not in _DTYPES:
        raise ValueError(f"dtypes must be in {_DTYPES}")
    pinned: Dict[str, Target] = dict(candidate_targets or {})
    for t in [*targets, *pinned.values()]:
        _check_target(as_target_spec(t))
    if optimize not in ("speed", "quality"):
        raise ValueError("optimize must be 'speed' or 'quality'")
    if not calibration_data:
        raise ValueError("calibration_data is required")
    metric_fn = resolve_metric(metric, metric_distance_fn, metric_evaluate_fn)
    model = _name_unnamed_nodes(model)

    from onnxsim.calibration import calibrate
    from onnxsim.full_qdq import quantize_full_qdq

    prov = list(providers) if providers else ["CPUExecutionProvider"]
    eval_data = list(calibration_data)[: data_size or None]

    run = _make_runner(eval_data, metric_output_index, prov)

    float_out = run(model)

    # Calibrate once: every float activation tensor, reused by every trial.
    inits = {t.name for t in model.graph.initializer}
    acts = [i.name for i in model.graph.input if i.name not in inits]
    acts += [o for n in model.graph.node for o in n.output]
    if calibrate_options is None:
        calibrate_options = (quantize_kwargs or {}).get("calibrate_options")
    ranges = calibrate(
        model,
        calibration_data,
        providers=providers,
        method=(quantize_kwargs or {}).get("method", method),
        extra_tensor_names=acts,
        **{"activation_type": base_dtype, **(calibrate_options or {})},
    )

    node_tensors = _node_tensors(model)

    base_kw: Dict[str, Any] = dict(
        quantize_kwargs
        if quantize_kwargs is not None
        else dict(per_channel=per_channel, exclude_nodes=exclude_nodes, method=method)
    )
    for k in ("calibration_data", "activation_dtype", "ranges", "convert_inputs"):
        base_kw.pop(k, None)
    base_td = dict(base_kw.pop("tensor_dtypes", None) or {})
    base_ts = dict(base_kw.pop("tensor_symmetric", None) or {})

    def quantize_acts(moved: Dict[str, TargetSpec]) -> onnx.ModelProto:
        """The model with the activations of the ``moved`` nodes at their
        target precisions (weights and biases still at the baseline's)."""
        dts, sym = dict(base_td), dict(base_ts)
        for node, spec in moved.items():
            ins, outs = node_tensors.get(node, ([], []))
            for prec, tensors in ((spec.inputs, ins), (spec.outputs, outs)):
                if prec is None:
                    continue
                for t in tensors:
                    dts[t] = prec[0]
                    if prec[1] is not None:
                        sym[t] = prec[1]
        q = quantize_full_qdq(
            model,
            calibration_data=calibration_data,
            activation_dtype=base_dtype,
            providers=providers,
            ranges=ranges,
            tensor_dtypes=dts or None,
            tensor_symmetric=sym or None,
            **base_kw,
        )
        return post_quantize(q) if post_quantize is not None else q

    baseline_model = quantize_acts({})
    # A Relu after a candidate only shares its output quantizer when the
    # quantizer folded it (not so for symmetric activations): the baseline
    # tells which Relu nodes survived
    kept_relus = {
        n.output[0].removesuffix("/f")
        for n in baseline_model.graph.node
        if n.op_type == "Relu"
    }
    node_tensors = {
        k: (ins, outs[:1] + [r for r in outs[1:] if r not in kept_relus])
        for k, (ins, outs) in node_tensors.items()
    }

    opset = next(
        (o.version for o in baseline_model.opset_import if o.domain in ("", "ai.onnx")),
        0,
    )
    baseline_model = _normalize_bias_params(
        baseline_model,
        wide=opset < 21
        and (base_dtype in _WIDE or base_kw.get("weight_dtype") in _WIDE),
    )
    baseline_model = _share_qparams(
        baseline_model,
        sorted(_QDQ_SHARING_OPS | set(base_kw.get("shared_ops", ()))),
    )
    act_sym = base_kw.get("symmetric_activations")
    if act_sym is None:
        act_sym = not base_dtype.startswith("u")
    wt_sym = bool(base_kw.get("weight_symmetric", True))
    range_of = _range_lookup(ranges)
    # tensors / nodes the baseline's own per-layer overrides gave a precision
    # (Quark's ``TensorQuantOverrides`` / ``NodesWithMixedPrecision``)
    override_tensors = set(base_td) | set(base_ts)
    mixed_nodes = [
        k
        for k, (ins, outs) in node_tensors.items()
        if override_tensors & set(ins + outs)
    ]

    def score_of(moved: Dict[str, TargetSpec]):
        if not moved:
            return baseline_model, metric_fn(float_out, run(baseline_model)), set()
        # Quark edits the quantized baseline in place (no re-quantization)
        from onnxsim.quark_mixing import QuarkMixer

        mixer = QuarkMixer(baseline_model, range_of, shared_param_mode)
        for node, spec in moved.items():
            mixer.promote_node(node, _resolve_symmetry(spec, act_sym, wt_sym))
        q = mixer.result()
        return q, metric_fn(float_out, run(q)), mixer.promoted_tensors

    def finalize(q: onnx.ModelProto, moved_nodes: List[str], tensors: Set[str]):
        from onnxsim.quark_boundary_qdq import insert_boundary_quant_nodes

        return insert_boundary_quant_nodes(
            q, range_of, tensors, moved_nodes, override_tensors, mixed_nodes
        )

    return _search(
        model,
        base_dtype,
        targets,
        pinned,
        score_of,
        node_tensors,
        target_op_types=target_op_types,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
        subgraphs=subgraphs,
        cache_file=cache_file,
        worker_num=worker_num,
        no_input_qdq_shared=no_input_qdq_shared,
        metric_threshold=metric_threshold,
        optimize=optimize,
        cache_key_fn=cache_key_fn,
        finalize=finalize if dual_quant_nodes else None,
    )


def _search(
    model: onnx.ModelProto,
    base_dtype: str,
    targets: Sequence[Target],
    pinned: Dict[str, Target],
    score_of: Callable[
        [Dict[str, TargetSpec]], Tuple[onnx.ModelProto, float, Set[str]]
    ],
    node_tensors: Dict[str, Tuple[List[str], List[str]]],
    *,
    target_op_types: Sequence[str],
    include_layers: Sequence[str],
    exclude_layers: Sequence[str],
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]],
    cache_file: Optional[Union[str, Path]],
    worker_num: int,
    no_input_qdq_shared: bool,
    metric_threshold: Optional[float],
    optimize: str,
    cache_key_fn: Optional[Callable[[onnx.ModelProto], str]] = None,
    finalize: Optional[
        Callable[[onnx.ModelProto, List[str], Set[str]], onnx.ModelProto]
    ] = None,
) -> AutoMixprecisionResult:
    """Quark's AMP driver over ``score_of(moved) -> (model, score, promoted
    tensors)``: baseline score, the threshold pre-check, the (cached)
    sensitivity ranking, then the greedy mixing walk (see the module
    docstring). ``finalize(model, moved nodes, promoted tensors)`` post-processes
    the mixed model once the walk is done (Quark's ``dual_quant_nodes``); the
    tensors are those of the *last* trial, a demoted candidate's included (as
    in Quark, whose set of promoted tensors is never reduced)."""

    def assign(nodes: Sequence[str], default: int) -> Dict[str, TargetSpec]:
        """``{node: target}`` for moving ``nodes`` (pinned nodes use their entry)."""
        return {
            n: as_target_spec(pinned[n] if n in pinned else targets[default])  # type: ignore[index]
            for n in nodes
        }

    baseline, baseline_score, _ = score_of({})
    result = AutoMixprecisionResult(baseline, baseline_score, baseline_score)
    # Quark decides on the threshold before it analyses anything (a model that
    # is already past it never gets a sensitivity cache written)
    if metric_threshold is not None and metric_threshold != 0:
        if optimize == "speed" and baseline_score > metric_threshold:
            return result  # already past the threshold: no room to optimize
        if optimize == "quality" and baseline_score <= metric_threshold:
            return result  # already good enough

    # -- sensitivity: cached, or each candidate under each target
    ops = tuple(target_op_types)
    if subgraphs is not None:
        groups = [
            (name, _candidate_nodes(model, ops, include_layers, exclude_layers, nodes))
            for name, nodes in subgraphs
        ]
    else:
        groups = [
            (n, [n])
            for n in _candidate_nodes(model, ops, include_layers, exclude_layers)
        ]
    groups = [(name, nodes) for name, nodes in groups if nodes]
    key = _fingerprint(
        model,
        base_dtype,
        targets,
        pinned,
        ops,
        include_layers,
        exclude_layers,
        subgraphs,
    )
    # ``cache_key_fn(baseline)`` is Quark's own key for the quantized baseline:
    # a cache is written with it (so Quark reads it) and read under it or under
    # this module's fingerprint
    quark_key = cache_key_fn(baseline) if cache_key_fn is not None else None
    write_key = quark_key or key
    accepted = [k for k in (quark_key, key) if k]
    ranked: Optional[List[SensitivityResult]] = None
    if cache_file is not None and Path(cache_file).exists():
        ranked = load_sensitivity(cache_file, accepted, node_tensors)
    if ranked is None:

        def score_group(item: Tuple[str, List[str]]) -> SensitivityResult:
            name, nodes = item
            scores = [score_of(assign(nodes, i))[1] for i in range(len(targets))]
            best = min(range(len(scores)), key=lambda i: scores[i])
            tensors = list(
                dict.fromkeys(
                    t
                    for n in nodes
                    for t in (lambda io: io[0] + io[1])(node_tensors.get(n, ([], [])))
                )
            )
            return SensitivityResult(
                name, nodes, tensors, scores[best], True, scores, best
            )

        workers = max(int(worker_num), 1)
        if workers > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                scored = list(ex.map(score_group, groups))
        else:
            scored = [score_group(g) for g in groups]
        ranked = sorted(scored, key=lambda c: c.score)
        if cache_file is not None:
            save_sensitivity(ranked, cache_file, write_key)
    result.ranked = ranked

    if metric_threshold is None or not result.ranked:
        return result

    shared = _shared_inputs(model) if no_input_qdq_shared else set()
    moved: Dict[str, TargetSpec] = {}
    cur_model, cur_score = baseline, baseline_score
    tensors: Set[str] = set()
    for c in result.ranked:
        if not c.enabled:
            continue
        nodes = [n for n in c.nodes if n not in shared]
        if not nodes:
            continue
        trial_model, score, tensors = score_of(
            {**moved, **assign(nodes, c.best_config_index)}
        )
        if metric_threshold == 0:
            keep, stop = True, False
        elif optimize == "speed":
            keep, stop = score <= metric_threshold, score > metric_threshold
        else:
            keep, stop = True, score <= metric_threshold
        if keep:
            moved.update(assign(nodes, c.best_config_index))
            result.moved.append(c.name)
            result.moved_nodes.extend(nodes)
            cur_model, cur_score = trial_model, score
        if stop:
            result.threshold_reached = optimize == "quality"
            break
    if finalize is not None:
        cur_model = finalize(cur_model, list(result.moved_nodes), tensors)
    result.model, result.final_score = cur_model, cur_score
    return result


def auto_mixprecision_blocks(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    block_dtype: str,
    exclude_nodes: Sequence[str] = (),
    target_op_types: Sequence[str] = ("Conv", "ConvTranspose", "Gemm", "MatMul"),
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
    metric: str = "l2",
    metric_distance_fn: Optional[MetricFn] = None,
    metric_evaluate_fn: Optional[Callable[[Outputs], float]] = None,
    metric_threshold: Optional[float] = 0.0,
    optimize: str = "speed",
    metric_output_index: Optional[int] = 0,
    data_size: int = 0,
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    cache_file: Optional[Union[str, Path]] = None,
    worker_num: int = 1,
    no_input_qdq_shared: bool = False,
    dual_quant_nodes: bool = True,
    cache_key_fn: Optional[Callable[[onnx.ModelProto], str]] = None,
    marking: Optional[Mapping[str, object]] = None,
    remove_after: Optional[Iterable[str]] = None,
) -> AutoMixprecisionResult:
    """Quark's AutoMixprecision for a bfloat16 model whose candidates move to a
    block format (``BF16_MIXED_BFP16`` / ``BF16_MIXED_MXINT8``): the same
    ranking and greedy mixing as :func:`auto_mixprecision` (``subgraphs``,
    ``cache_file`` with ``"enabled": false`` pins, ``worker_num``,
    ``no_input_qdq_shared``, thresholds, ...), over
    :func:`onnxsim.quark_preset_graphs.apply_mixed_block_format`.

    A candidate is scored on the bfloat16 model with only its own slots swapped
    to the block format (no boundary nodes, like Quark's analysis) and the
    models run on :func:`onnxsim.quark_fakequant_eval.run_fake_quantized`, so a
    score can differ from Quark's by float32 accumulation noise. ``metric_*``
    of the result are those of that model; the returned ``model`` carries the
    boundary nodes of ``dual_quant_nodes``."""
    from onnxsim.quark_fakequant_eval import run_fake_quantized
    from onnxsim.quark_preset_graphs import _with_node_names, apply_mixed_block_format

    if optimize not in ("speed", "quality"):
        raise ValueError("optimize must be 'speed' or 'quality'")
    if not calibration_data:
        raise ValueError("calibration_data is required")
    metric_fn = resolve_metric(metric, metric_distance_fn, metric_evaluate_fn)
    import onnxruntime as ort

    model = _with_node_names(model)
    eval_data = list(calibration_data)[: data_size or None]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )

    def pick(outs: List[np.ndarray]) -> List[np.ndarray]:
        return [outs[metric_output_index]] if metric_output_index is not None else outs

    float_out: Outputs = [pick(sess.run(None, b)) for b in eval_data]
    ops = tuple(target_op_types)

    def build(nodes: Sequence[str], dual: bool) -> onnx.ModelProto:
        # an include list with no match promotes nothing: the bfloat16 baseline
        return apply_mixed_block_format(
            model,
            block_dtype,
            exclude=list(exclude_nodes),
            target_ops=ops,
            include_layers=list(nodes) or ["\0no-such-layer"],
            dual_nodes=dual,
            marking=marking,
            remove_after=remove_after,
        )

    def score_of(moved: Dict[str, TargetSpec]):
        q = build(list(moved), False)
        outs = run_fake_quantized(q, eval_data)
        return q, metric_fn(float_out, [pick(o) for o in outs]), set()

    # the block format has no per-tensor targets: one placeholder precision
    node_tensors: Dict[str, Tuple[List[str], List[str]]] = {
        _node_key(n): ([], []) for n in model.graph.node
    }
    result = _search(
        model,
        f"bfloat16->{block_dtype}",
        [TargetSpec()],
        {},
        score_of,
        node_tensors,
        target_op_types=ops,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
        subgraphs=subgraphs,
        cache_file=cache_file,
        worker_num=worker_num,
        no_input_qdq_shared=no_input_qdq_shared,
        metric_threshold=metric_threshold,
        optimize=optimize,
        cache_key_fn=cache_key_fn,
    )
    if result.moved_nodes:
        result.model = build(result.moved_nodes, dual_quant_nodes)
    return result


def auto_mixprecision_from_baseline(
    model: onnx.ModelProto,
    calibration_data: Sequence[Dict[str, np.ndarray]],
    baseline: onnx.ModelProto,
    targets: Sequence[Target],
    candidate_targets: Optional[Mapping[str, Target]] = None,
    ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    base_label: str = "baseline",
    target_op_types: Sequence[str] = ("Conv", "ConvTranspose", "Gemm", "MatMul"),
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
    metric: str = "l2",
    metric_distance_fn: Optional[MetricFn] = None,
    metric_evaluate_fn: Optional[Callable[[Outputs], float]] = None,
    metric_threshold: Optional[float] = 0.0,
    optimize: str = "speed",
    metric_output_index: Optional[int] = 0,
    data_size: int = 0,
    subgraphs: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
    cache_file: Optional[Union[str, Path]] = None,
    worker_num: int = 1,
    no_input_qdq_shared: bool = False,
    activation_symmetric: bool = True,
    weight_symmetric: bool = True,
    shared_param_mode: str = "propagate",
    cache_key_fn: Optional[Callable[[onnx.ModelProto], str]] = None,
    dual_quant_nodes: bool = False,
    override_tensors: Sequence[str] = (),
    mixed_nodes: Sequence[str] = (),
) -> AutoMixprecisionResult:
    """Quark's AutoMixprecision over a *given* quantized baseline of any kind --
    a ``float16`` / ``bfloat16`` / BFP / MX fake-quantized model (see
    :mod:`onnxsim.quark_fakequant_graph`) or an integer QDQ one -- with targets
    of any kind (:class:`TargetSpec` entries): the same ranking and greedy
    mixing as :func:`auto_mixprecision`, the edits those of
    :class:`onnxsim.quark_mixing.QuarkMixer` on ``baseline``.

    :param model: the float model (its outputs are the reference)
    :param baseline: the quantized model to promote candidates in; its node
            names must be those of ``model``
    :param ranges: calibrated ranges of the activations; ``None`` is Quark's
            fake calibration ``[0, 1]`` of every tensor, which it uses (instead
            of calibrating) whenever the baseline is a float / block format
    :param base_label: names the baseline in the cache fingerprint
    :param activation_symmetric: Quark's ``ActivationSymmetric`` (the global
            activation spec's symmetry, which wins over a target spec's own)
    :param weight_symmetric: likewise ``WeightSymmetric``
    :param dual_quant_nodes: Quark's boundary quantizers, inserted into the
            final model (:func:`onnxsim.quark_boundary_qdq.insert_boundary_quant_nodes`)
    :param override_tensors: tensors with quantization overrides of their own
            (only matters to ``dual_quant_nodes``)
    :param mixed_nodes: nodes with a precision of their own (likewise)
    """
    from onnxsim.quark_mixing import QuarkMixer, fake_ranges

    if optimize not in ("speed", "quality"):
        raise ValueError("optimize must be 'speed' or 'quality'")
    if shared_param_mode not in ("propagate", "unshare"):
        raise ValueError("shared_param_mode must be 'propagate' or 'unshare'")
    if not calibration_data:
        raise ValueError("calibration_data is required")
    targets = list(targets)
    if not targets:
        raise ValueError("targets must not be empty")
    pinned: Dict[str, Target] = dict(candidate_targets or {})
    for t in [*targets, *pinned.values()]:
        _check_target(as_target_spec(t))
    metric_fn = resolve_metric(metric, metric_distance_fn, metric_evaluate_fn)
    eval_data = list(calibration_data)[: data_size or None]
    run = _make_runner(eval_data, metric_output_index, ["CPUExecutionProvider"])
    float_out = run(model)
    range_of = (
        _range_lookup(ranges)
        if ranges is not None
        else _range_lookup(fake_ranges(model))
    )

    def score_of(moved: Dict[str, TargetSpec]):
        if not moved:
            return baseline, metric_fn(float_out, run(baseline)), set()
        mixer = QuarkMixer(baseline, range_of, shared_param_mode)
        for node, spec in moved.items():
            mixer.promote_node(
                node, _resolve_symmetry(spec, activation_symmetric, weight_symmetric)
            )
        q = mixer.result()
        return q, metric_fn(float_out, run(q)), mixer.promoted_tensors

    def finalize(q: onnx.ModelProto, moved_nodes: List[str], tensors: Set[str]):
        from onnxsim.quark_boundary_qdq import insert_boundary_quant_nodes

        return insert_boundary_quant_nodes(
            q, range_of, tensors, moved_nodes, override_tensors, mixed_nodes
        )

    node_tensors: Dict[str, Tuple[List[str], List[str]]] = {
        _node_key(n): ([], []) for n in model.graph.node
    }
    return _search(
        model,
        base_label,
        targets,
        pinned,
        score_of,
        node_tensors,
        target_op_types=tuple(target_op_types),
        include_layers=include_layers,
        exclude_layers=exclude_layers,
        subgraphs=subgraphs,
        cache_file=cache_file,
        worker_num=worker_num,
        no_input_qdq_shared=no_input_qdq_shared,
        metric_threshold=metric_threshold,
        optimize=optimize,
        cache_key_fn=cache_key_fn,
        finalize=finalize if dual_quant_nodes else None,
    )


def _shared_inputs(model: onnx.ModelProto) -> Set[str]:
    """Nodes whose first activation input is read by more than one node."""
    inits = {t.name for t in model.graph.initializer}
    readers: Dict[str, int] = {}
    for n in model.graph.node:
        for x in set(n.input):
            readers[x] = readers.get(x, 0) + 1
    out: Set[str] = set()
    for n in model.graph.node:
        first = next((x for x in n.input if x and x not in inits), None)
        if first is not None and readers.get(first, 0) > 1:
            out.add(_node_key(n))
    return out


__all__: Any = [
    "AutoMixprecisionResult",
    "BUILTIN_METRICS",
    "SensitivityResult",
    "SubgraphSpec",
    "auto_mixprecision",
    "auto_mixprecision_blocks",
    "auto_mixprecision_from_baseline",
    "apply_layer_mixing",
    "TargetSpec",
    "cosine_metric",
    "kl_metric",
    "l2_metric",
    "load_sensitivity",
    "parse_subgraph_json",
    "psnr_metric",
    "resolve_metric",
    "save_sensitivity",
    "sqnr_metric",
]
