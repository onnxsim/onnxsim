"""The sensitivity-cache key of AMD Quark's AutoMixprecision, reproduced.

Quark (``quark.onnx.algorithm.mprecision.sensitivity_analyzer``) writes its
sensitivity ranking as JSON ``{"version", "cache_key", "results"}`` and reads
it back only when ``cache_key`` equals the SHA-256 of

* every node of its *quantized baseline model* -- ``name|op_type|inputs|outputs``,
  sorted by ``(name, op_type)``,
* the candidate op types (sorted),
* the target ``QLayerConfig`` (s) as ``to_dict()`` JSON,
* ``include_layers`` and ``exclude_layers`` (sorted).

To read a cache Quark wrote -- or write one Quark reads -- onnxsim has to
compute that digest for *its* baseline, so :func:`quark_named_nodes` renames
the tensors and nodes of an onnxsim integer-QDQ baseline the way Quark's
quantizer names them (``<t>_QuantizeLinear`` / ``<t>_DequantizeLinear`` pairs
around the original tensor names, ``<w>_quantized`` ... for folded constants,
the scale / zero point initializers named after the tensor that owns them).
Baselines built by :mod:`onnxsim.quark_fakequant_graph` already use Quark's
names. Graph shapes outside what the probes covered may still rename
differently; a key that does not match is then treated as a stale cache, which
is the safe direction.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import onnx

_DQ_NAME = re.compile(r"^(?P<base>.*)/qdq\d+/")

#: onnxsim dtype -> the ``data_type`` Quark's ``QTensorConfig.to_dict`` shows
_DATA_TYPE = {
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "float16": "Float16",
    "bfloat16": "BFloat16",
    "bfp16": "BFP16",
    "mx4": "MX4",
    "mx6": "MX6",
    "mx9": "MX9",
    "mxfp4_e2m1": "MXFP4E2M1",
    "mxfp6_e3m2": "MXFP6E3M2",
    "mxfp6_e2m3": "MXFP6E2M3",
    "mxfp8_e5m2": "MXFP8E5M2",
    "mxfp8_e4m3": "MXFP8E4M3",
    "mxint8": "MXInt8",
}
_CALIB = {
    "minmax": "MinMax",
    "minmax_mean": "MinMax",
    "percentile": "Percentile",
    "quark_percentile": "Percentile",
    "entropy": "Entropy",
    "quark_entropy": "Entropy",
    "distribution": "Distribution",
    "quark_distribution": "Distribution",
    "layerwise_percentile": "LayerwisePercentile",
    "quark_layerwise_percentile": "LayerwisePercentile",
    "minmse_pof2": "MinMSE",
}


def spec_dict(spec: Any) -> Dict[str, Any]:
    """Quark's ``QTensorConfig.to_dict()`` for a ``quark_compat`` spec."""
    calib = str(spec.calibration_method).split(":")[0]
    return {
        "symmetric": bool(spec.symmetric),
        "scale_type": "ScaleType.PowerOf2" if spec.pof2 else "ScaleType.Float32",
        "calibration_method": "CalibMethod." + _CALIB.get(calib, "MinMax"),
        "quant_granularity": "QuantGranularity.Tensor",
        "data_type": _DATA_TYPE[spec.dtype],
    }


def layer_config_dict(cfg: Any) -> Dict[str, Any]:
    """Quark's ``QLayerConfig.to_dict()`` (set fields only, under the names
    the config was spelled with)."""
    out: Dict[str, Any] = {}
    spelled_activation = getattr(cfg, "_activation_spelled", True)
    for field, spec in (
        ("input_tensors", None if spelled_activation else cfg.input_tensors),
        ("activation", cfg.activation if spelled_activation else None),
        ("weight", cfg.weight),
        ("bias", cfg.bias),
        ("output_tensors", cfg.output_tensors),
    ):
        if spec is not None:
            out[field] = spec_dict(spec)
    return out


def target_config_repr(target: Any) -> str:
    """The ``target_layer_config`` part of Quark's key: a ``QLayerConfig``, a
    list of them, or ``{QLayerConfig: [node names]}``."""
    if isinstance(target, list):
        return json.dumps([layer_config_dict(c) for c in target], sort_keys=True)
    if isinstance(target, dict):
        return json.dumps(
            {
                json.dumps(layer_config_dict(k), sort_keys=True): v
                for k, v in target.items()
            },
            sort_keys=True,
        )
    return json.dumps(layer_config_dict(target), sort_keys=True)


def cache_key(
    nodes: Iterable[Tuple[str, str, Sequence[str], Sequence[str]]],
    target_op_types: Sequence[str],
    config_repr: str,
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
) -> str:
    """Quark's ``compute_cache_key`` over ``(name, op_type, inputs, outputs)``
    node tuples."""
    parts: List[str] = [
        f"{name}|{op}|{','.join(ins)}|{','.join(outs)}"
        for name, op, ins, outs in sorted(nodes, key=lambda n: (n[0], n[1]))
    ]
    parts.append("target_op_type:" + ",".join(sorted(target_op_types)))
    parts.append("target_layer_config:" + config_repr)
    parts.append("include_layers:" + ",".join(sorted(include_layers)))
    parts.append("exclude_layers:" + ",".join(sorted(exclude_layers)))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def quark_named_nodes(
    model: onnx.ModelProto,
) -> List[Tuple[str, str, List[str], List[str]]]:
    """``(name, op_type, inputs, outputs)`` of every node of ``model`` with
    Quark's quantizer naming (see the module docstring). A model that is
    already Quark-named (float / block baselines) comes back unchanged."""
    g = model.graph
    nodes = list(g.node)
    inits = {t.name: t for t in g.initializer}
    graph_outs = {o.name for o in g.output}
    cons: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for x in n.input:
            cons.setdefault(x, []).append(n)

    tensor: Dict[str, str] = {}  # old tensor name -> Quark's
    node_name: Dict[int, str] = {}
    init_name: Dict[str, str] = {}  # scale / zero point initializer -> Quark's

    def qdq_nodes():
        for n in nodes:
            if n.op_type == "QuantizeLinear" and n.input[0] not in inits:
                dq = next(
                    (
                        c
                        for c in cons.get(n.output[0], [])
                        if c.op_type == "DequantizeLinear"
                    ),
                    None,
                )
                if dq is not None:
                    yield n, dq

    # activation pairs -- the tensor ``A`` is the Q's data input without onnxsim's
    # ``/f`` suffix; the pair is named after it
    for q, dq in qdq_nodes():
        data = q.input[0]
        a = (
            data[:-2]
            if data.endswith("/f") and data not in {i.name for i in g.input}
            else data
        )
        is_out = a in graph_outs
        if data != a or is_out:
            tensor[data] = a + "_QuantizeLinear_Input" if is_out else a
        q_out = a + "_QuantizeLinear_Output"
        tensor[q.output[0]] = q_out
        tensor[dq.output[0]] = a if is_out else a + "_DequantizeLinear_Output"
        node_name[id(q)] = a + "_QuantizeLinear"
        node_name[id(dq)] = a + "_DequantizeLinear"
        # the pair's parameters belong to its tensor unless an earlier pair
        # already owns them (pass-through ops share them)
        for idx, suffix in ((1, "_scale"), (2, "_zero_point")):
            if len(q.input) > idx:
                init_name.setdefault(q.input[idx], a + suffix)
                init_name.setdefault(dq.input[idx], init_name[q.input[idx]])
    # folded constants: a DequantizeLinear over initializer codes
    for n in nodes:
        if n.op_type != "DequantizeLinear" or n.input[0] not in inits:
            continue
        m = _DQ_NAME.match(n.output[0])
        base = m.group("base") if m else n.output[0]
        zp = inits.get(n.input[2]) if len(n.input) > 2 else None
        is_bias = zp is not None and zp.data_type == onnx.TensorProto.INT32
        node_name[id(n)] = base + "_DequantizeLinear"
        if is_bias:
            tensor[n.output[0]] = base
            tensor[n.input[0]] = base + "_quantized"
            if len(n.input) > 1:
                tensor[n.input[1]] = base + "_quantized_scale"
            if len(n.input) > 2:
                tensor[n.input[2]] = base + "_quantized_zero_point"
        else:
            tensor[n.output[0]] = base + "_DequantizeLinear_Output"
            tensor[n.input[0]] = base + "_quantized"
            if len(n.input) > 1:
                tensor[n.input[1]] = base + "_scale"
            if len(n.input) > 2:
                tensor[n.input[2]] = base + "_zero_point"
    for old, new in init_name.items():
        tensor[old] = new

    def plain(name: str) -> str:
        # nodes onnxsim inserts after a float tensor ``t/f`` (the NPU CNN
        # rewrites: ``t_Mul`` and its ``t_Scale``) are named after ``t`` in Quark
        return name.replace("/f_", "_")

    out = []
    for n in nodes:
        out.append(
            (
                node_name.get(id(n)) or plain(n.name),
                n.op_type,
                [tensor[x] if x in tensor else plain(x) for x in n.input],
                [tensor[o] if o in tensor else plain(o) for o in n.output],
            )
        )
    return out


def quark_cache_key(
    model: onnx.ModelProto,
    target: Any,
    target_op_types: Sequence[str],
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
) -> str:
    """Quark's cache key of ``model`` (an onnxsim baseline) for a
    ``target_layer_config`` of ``quark_compat`` objects."""
    return cache_key(
        quark_named_nodes(model),
        target_op_types,
        target_config_repr(target),
        include_layers,
        exclude_layers,
    )


__all__: Any = [
    "cache_key",
    "layer_config_dict",
    "quark_cache_key",
    "quark_named_nodes",
    "spec_dict",
    "target_config_repr",
]
