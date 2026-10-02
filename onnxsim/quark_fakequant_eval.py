"""Run models that use Quark's ``com.amd.quark`` fake-quantization custom ops
(``BFPQuantizeDequantize``, ``MXQuantizeDequantize`` and the bfloat16 /
float16 ``ExtendedQuantizeLinear`` / ``ExtendedDequantizeLinear`` pair) without
Quark's compiled operator library.

ONNX Runtime cannot execute those ops on its own, so :func:`run_fake_quantized`
cuts the graph at them: every run of ordinary ops between two custom ops is an
ONNX Runtime session of its own (so Gemm / Conv / ... keep ONNX Runtime's
kernels and float32 accumulation order, graph optimizations off) and the custom
ops themselves are evaluated with :mod:`onnxsim.quark_block_formats`, which is
bit-exact against Quark's kernels. The outputs therefore match running the
whole model in ONNX Runtime with Quark's library.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxsim import quark_block_formats as bf

COP_DOMAIN = "com.amd.quark"
_ROUNDING = {0: "std", 1: "dpu", 2: "py3"}
_HALF = {int(TensorProto.FLOAT16): "float16", int(TensorProto.BFLOAT16): "bfloat16"}


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    out = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    return {k: v.decode() if isinstance(v, bytes) else v for k, v in out.items()}


def _bfp(node: onnx.NodeProto, x: np.ndarray) -> np.ndarray:
    a = _attrs(node)
    x = np.asarray(x, np.float32)
    if a.get("convert_to_bfloat_before_bfp", 0):
        x = bf.bf16_round(x)
    mode = _ROUNDING[int(a.get("rounding_mode", 2))]
    if a.get("bfp_method", "to_bfp") == "to_bfp_prime":
        return bf.bfp_prime(
            x,
            bit_width=int(a.get("bit_width", 13)),
            block_size=int(a.get("block_size", 16)),
            sub_block_size=int(a.get("sub_block_size", 2)),
            sub_block_shift_bits=int(a.get("sub_block_shift_bits", 1)),
            axis=int(a.get("axis", 1)),
            rounding=mode,
        )
    return bf.bfp16(
        x,
        bit_width=int(a.get("bit_width", 16)),
        block_size=int(a.get("block_size", 8)),
        axis=int(a.get("axis", 1)),
        rounding=mode,
    )


def _mx(node: onnx.NodeProto, x: np.ndarray) -> np.ndarray:
    a = _attrs(node)
    return bf.mx(
        np.asarray(x, np.float32),
        element_dtype=a.get("element_dtype", "int8"),
        block_size=int(a.get("block_size", 32)),
        axis=int(a.get("axis", 1)),
        rounding=_ROUNDING[int(a.get("rounding_mode", 2))],
    )


def _scalar(model_inits: Dict[str, TensorProto], name: str, default: float) -> float:
    t = model_inits.get(name)
    if t is None:
        return default
    try:
        return float(np.asarray(numpy_helper.to_array(t), np.float32).ravel()[0])
    except Exception:  # an exotic half-type tensor numpy cannot hold
        return default


def _custom(
    node: onnx.NodeProto,
    env: Dict[str, np.ndarray],
    inits: Dict[str, TensorProto],
) -> List[np.ndarray]:
    op = node.op_type
    if op == "BFPQuantizeDequantize":
        return [_bfp(node, env[node.input[0]]).astype(np.float32)]
    if op == "MXQuantizeDequantize":
        return [_mx(node, env[node.input[0]]).astype(np.float32)]
    if op in ("ExtendedQuantizeLinear", "ExtendedDequantizeLinear"):
        x = np.asarray(env[node.input[0]], np.float32)
        scale = np.asarray(
            env[node.input[1]]
            if node.input[1] in env
            else numpy_helper.to_array(inits[node.input[1]]),
            np.float32,
        )
        zp_t = inits.get(node.input[2]) if len(node.input) > 2 else None
        zp = _scalar(inits, node.input[2], 0.0) if zp_t is not None else 0.0
        if op == "ExtendedDequantizeLinear":
            return [((x - np.float32(zp)) * scale).astype(np.float32)]
        # the half type of the quantizer is its zero point's type
        half = (
            _HALF.get(int(zp_t.data_type), "bfloat16")
            if zp_t is not None
            else "bfloat16"
        )
        v = x / scale + np.float32(zp)
        return [
            (bf.fp16_round(v) if half == "float16" else bf.bf16_round(v)).astype(
                np.float32
            )
        ]
    raise NotImplementedError(f"custom op {COP_DOMAIN}::{op} is not supported")


def _toposort(nodes: Sequence[onnx.NodeProto], known: Set[str]) -> List[onnx.NodeProto]:
    done = set(known)
    todo = list(nodes)
    out: List[onnx.NodeProto] = []
    while todo:
        ready = [n for n in todo if all(not x or x in done for x in n.input)]
        if not ready:
            raise ValueError("the graph is not a DAG (or an input is undefined)")
        for n in ready:
            out.append(n)
            done.update(o for o in n.output if o)
        ids = {id(n) for n in ready}
        todo = [n for n in todo if id(n) not in ids]
    return out


class _Segment:
    """A run of ordinary nodes executed by one ONNX Runtime session."""

    def __init__(
        self,
        nodes: List[onnx.NodeProto],
        consumed_later: Set[str],
        model: onnx.ModelProto,
        types: Dict[str, int],
        inits: Dict[str, TensorProto],
    ) -> None:
        import onnxruntime as ort

        produced = {o for n in nodes for o in n.output}
        self.inputs = [
            x
            for x in dict.fromkeys(i for n in nodes for i in n.input)
            if x and x not in produced and x not in inits
        ]
        self.outputs = [o for o in dict.fromkeys(produced) if o in consumed_later]
        g = helper.make_graph(
            nodes,
            "segment",
            [
                helper.make_tensor_value_info(x, types.get(x, TensorProto.FLOAT), None)
                for x in self.inputs
            ],
            [
                helper.make_tensor_value_info(o, types.get(o, TensorProto.FLOAT), None)
                for o in self.outputs
            ],
            [
                inits[x]
                for x in dict.fromkeys(i for n in nodes for i in n.input)
                if x in inits
            ],
        )
        seg = helper.make_model(g, opset_imports=list(model.opset_import))
        seg.ir_version = model.ir_version
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(
            seg.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )

    def run(self, env: Dict[str, np.ndarray]) -> None:
        outs = self.sess.run(self.outputs, {x: env[x] for x in self.inputs})
        env.update(zip(self.outputs, outs))


def run_fake_quantized(
    model: onnx.ModelProto,
    batches: Sequence[Dict[str, np.ndarray]],
    output_names: Optional[Sequence[str]] = None,
) -> List[List[np.ndarray]]:
    """``outputs[batch][output index]`` of ``model`` on each feed dict
    (``output_names`` defaults to the graph outputs)."""
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    feeds = {v.name for v in g.input} - set(inits)
    nodes = _toposort(g.node, feeds | set(inits))
    wanted = list(output_names) if output_names else [o.name for o in g.output]

    types: Dict[str, int] = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        for v in [
            *inferred.graph.value_info,
            *inferred.graph.input,
            *inferred.graph.output,
        ]:
            types[v.name] = v.type.tensor_type.elem_type
    except Exception:  # types are only hints for the segment boundaries
        pass
    # shape inference knows no ``com.microsoft`` QuantizeLinear (16-bit codes):
    # a quantizer's output is typed by its zero point (uint8 without one)
    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.output:
            zp = inits.get(n.input[2]) if len(n.input) > 2 else None
            types[n.output[0]] = zp.data_type if zp is not None else TensorProto.UINT8

    # plan: ordinary runs and single custom nodes, in order
    plan: List[Any] = []
    run_nodes: List[onnx.NodeProto] = []
    for n in nodes:
        if n.domain == COP_DOMAIN:
            if run_nodes:
                plan.append(run_nodes)
                run_nodes = []
            plan.append(n)
        else:
            run_nodes.append(n)
    if run_nodes:
        plan.append(run_nodes)

    steps: List[Any] = []
    for i, item in enumerate(plan):
        if isinstance(item, onnx.NodeProto):
            steps.append(item)
            continue
        later = {x for later_item in plan[i + 1 :] for x in _inputs_of(later_item)}
        later |= set(wanted)
        steps.append(_Segment(item, later, model, types, inits))

    consts: Dict[str, np.ndarray] = {}  # initializers the custom ops read
    for step in steps:
        if isinstance(step, onnx.NodeProto):
            for x in step.input:
                if x in inits and x not in consts:
                    try:
                        consts[x] = numpy_helper.to_array(inits[x])
                    except Exception:  # a half-type zero point: read by _scalar
                        pass

    results: List[List[np.ndarray]] = []
    for batch in batches:
        env: Dict[str, np.ndarray] = {**consts}
        env.update({k: np.asarray(v) for k, v in batch.items()})
        for step in steps:
            if isinstance(step, _Segment):
                step.run(env)
            else:
                env.update(zip(step.output, _custom(step, env, inits)))
        results.append([env[o] for o in wanted])
    return results


def _inputs_of(item: Any) -> List[str]:
    if isinstance(item, onnx.NodeProto):
        return [x for x in item.input if x]
    return [x for n in item for x in n.input if x]


__all__ = ["run_fake_quantized"]
