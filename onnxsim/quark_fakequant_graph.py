"""Insert Quark-style fake-quantization nodes into a float ONNX graph -- the
graph-level half of Quark's ``BFP16`` / ``MX*`` (block formats) and ``FP16`` /
``BF16`` presets. The numerics of the block formats are in
:mod:`onnxsim.quark_block_formats`; for ``float16`` / ``bfloat16`` the "fake
quantization" is rounding the tensor to that dtype while the computation stays
float32 (so the model runs on any runtime that has Quark's ops, e.g. ONNX
Runtime CPU, which has no bf16 MatMul).

The placement rules were derived by running Quark's own quantizer on probe
graphs (see ``tests/test_quark_parity.py``, which re-checks them against the
installed ``amd-quark`` in CI) and reading its source for the axis rule:

- **Active ops** quantize all of their own float tensors -- activation inputs,
  activation outputs and constant (weight / bias) inputs: ``Conv``,
  ``ConvTranspose``, ``MatMul``, ``Gemm``, ``Add`` / ``Sub`` / ``Mul`` / ``Div``,
  ``Concat``, ``Pad``, ``Sigmoid``, ``Tanh``, ``LeakyRelu``, ``Softmax``,
  ``Erf``, ``Gelu``, ``AveragePool``, ``GlobalAveragePool``,
  ``InstanceNormalization``.
- **Pass-through ops** (``Relu``, ``MaxPool``, ``Flatten``, ``Reshape``,
  ``Transpose``, ``Squeeze``, ``Unsqueeze``, ``Resize``) get their output
  quantized only when their input already is -- so a lone ``Relu`` is left
  alone but ``Conv -> Relu`` is quantized end to end.
- ``LayerNormalization`` is quantized end to end (scale and bias too) when its
  input already is; otherwise left alone.
- Everything else (``Abs``, ``Exp``, ``Clip``, ``BatchNormalization``, ...) is
  not quantized.

Block formats get one ``com.amd.quark`` node per tensor; ``float16`` /
``bfloat16`` get an ``ExtendedQuantizeLinear`` / ``ExtendedDequantizeLinear``
pair (scale 1.0, a zero point of 0 *in that dtype* as the type marker). Each
quantized tensor ``t`` gets that; its consumers read
``t_DequantizeLinear_Output`` (a graph output ``t`` keeps its name and its
producer writes ``t_QuantizeLinear_Input`` -- Quark's naming). The block axis
is 1 by default and then refined, in graph order, to the reduction (``K``)
dimension: ``MatMul`` A -> ``-1`` (its output follows), ``MatMul`` B -> ``-2``
(or ``rank - 2`` for a constant), ``Gemm`` A / B by ``transA`` / ``transB``,
``Softmax`` input -> the softmax axis; a 1-D constant (bias) uses axis 0.

With ``marking`` (what :class:`onnxsim.quark_compat.ModelQuantizer` passes) the
tensors are picked by Quark's own marking walk
(:func:`onnxsim.quark_marking.skipped_nodes`) instead of these lists, and the
result goes through Quark's topological sort, ``clean_initializers`` and -- for the
float16 / bfloat16 pairs -- Quark's scale / zero point sharing (a data-movement op's
output reads its input's) and its Q/DQ removal rules. The model pre-processing
(BatchNormalization folding, ``ReduceMean`` -> ``GlobalAveragePool``, ...) is
:meth:`ModelQuantizer._preprocess_block_flow`'s, before this module sees the graph;
CLE is left to a ``CLEConfig``.
"""

from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import onnx
from onnx import numpy_helper

from onnxsim.quark_marking import opset_unquantized_ops, quark_sorted, skipped_nodes

COP_DOMAIN = "com.amd.quark"
DQ_SUFFIX = "_DequantizeLinear_Output"
Q_INPUT_SUFFIX = "_QuantizeLinear_Input"

ACTIVE_OPS = {
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Gemm",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Concat",
    "Pad",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Softmax",
    "Erf",
    "Gelu",
    "AveragePool",
    "GlobalAveragePool",
    "InstanceNormalization",
}
# Quantized end to end (inputs, constants, outputs) when -- and only when --
# their data input already is quantized.
FOLLOWING_ACTIVE_OPS = {"LayerNormalization"}
PASS_THROUGH_OPS = {
    "Relu",
    "MaxPool",
    "Flatten",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Resize",
}


# Ops whose quantizer gives the output the input's quantization parameters
# (ONNX Runtime's QDQDirect8BitOp / QDQResize / QDQMaxPool / QDQSplit, which
# Quark's registry reuses)
_SHARING_OPS = {
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Resize",
    "MaxPool",
    "Split",
    "Gather",
}


def node_spec(dtype: str, axis: int = 1) -> Tuple[str, Dict[str, object]]:
    """``(op_type, attributes)`` of the custom op that fake-quantizes in the
    block format ``dtype`` (a ``quark_compat`` spec dtype such as ``"bfp16"``,
    ``"mx6"``, ``"mxfp8_e4m3"``), with Quark's default attributes."""
    if dtype == "bfp16" or dtype in ("mx4", "mx6", "mx9"):
        prime = dtype != "bfp16"
        return "BFPQuantizeDequantize", {
            "bfp_method": "to_bfp_prime" if prime else "to_bfp",
            "axis": axis,
            "bit_width": 16
            if dtype == "bfp16"
            else {"mx4": 11, "mx6": 13, "mx9": 16}[dtype],
            "block_size": 16 if prime else 8,
            "rounding_mode": 2,
            "sub_block_size": 2,
            "sub_block_shift_bits": 1,
            "convert_to_bfloat_before_bfp": 0,
        }
    if dtype == "mxint8" or dtype.startswith("mxfp"):
        elem = "int8" if dtype == "mxint8" else dtype.replace("mxfp", "fp", 1)
        return "MXQuantizeDequantize", {
            "element_dtype": elem,
            "axis": axis,
            "block_size": 32,
            "rounding_mode": 2,
        }
    raise ValueError(f"not a block format: {dtype!r}")


# Quark's FP16/BF16 presets quantize a wider op set than its block formats:
# the shape-only ops are quantized even when they start a chain, and BF16
# additionally covers the elementwise math ops and BatchNormalization.
_HALF_SHAPE_OPS = {
    "MaxPool",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Resize",
    "LayerNormalization",
}
HALF_EXTRA_OPS = {
    "float16": _HALF_SHAPE_OPS,
    "bfloat16": _HALF_SHAPE_OPS
    | {"Flatten", "Abs", "Neg", "Exp", "Sqrt", "BatchNormalization"},
}

HALF_DTYPES = {
    "float16": onnx.TensorProto.FLOAT16,
    "bfloat16": onnx.TensorProto.BFLOAT16,
}


def _make_node(
    dtype: str,
    x: str,
    y: str,
    axis: int,
    name: str,
    overrides: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> onnx.NodeProto:
    op, attrs = node_spec(dtype, axis)
    # Quark's ``BFPAttributes`` / ``MXAttributes`` extra options update the
    # node's default attributes (the block axis is refined afterwards anyway)
    attrs.update(
        {k: v for k, v in (overrides or {}).get(op, {}).items() if k != "axis"}
    )
    return onnx.helper.make_node(
        op,
        [x],
        [y],
        name=name,
        domain=COP_DOMAIN,
        **attrs,  # type: ignore[arg-type]
    )


def _make_half_pair(
    dtype: str, x: str, y: str, tensor: str, share: Optional[str] = None
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
    """Quark's fp16 / bf16 fake quantization of ``tensor``: Q then DQ with
    scale 1.0 and a zero point of 0 in ``dtype``; ``x`` -> ``y``. ``share`` is the
    tensor whose scale and zero-point initializers ``tensor`` reads instead of
    its own (a data-movement op's output), which then add none."""
    scale, zp = (share or tensor) + "_scale", (share or tensor) + "_zero_point"
    qout = tensor + "_QuantizeLinear_Output"
    nodes = [
        onnx.helper.make_node(
            "ExtendedQuantizeLinear",
            [x, scale, zp],
            [qout],
            name=tensor + "_QuantizeLinear",
            domain=COP_DOMAIN,
        ),
        onnx.helper.make_node(
            "ExtendedDequantizeLinear",
            [qout, scale, zp],
            [y],
            name=tensor + "_DequantizeLinear",
            domain=COP_DOMAIN,
        ),
    ]
    inits = (
        []
        if share
        else [
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), scale),
            onnx.helper.make_tensor(zp, HALF_DTYPES[dtype], [], [0.0]),
        ]
    )
    return nodes, inits


def _attr(node: onnx.NodeProto, name: str, default):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


class _Plan:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.inits = {t.name: t for t in model.graph.initializer}
        # (ONNX Runtime's QDQ MaxPool / Resize quantizers, which Quark's flows
        # reuse, do nothing below opset 12 / 11)
        self.gated = opset_unquantized_ops(model)
        self.skipped: Set[str] = set()
        inferred = onnx.shape_inference.infer_shapes(model)
        self.elem: Dict[str, int] = {}
        for vi in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        ):
            self.elem[vi.name] = vi.type.tensor_type.elem_type
        for name, t in self.inits.items():
            self.elem[name] = t.data_type

    def is_float(self, name: str) -> bool:
        return bool(name) and self.elem.get(name) == onnx.TensorProto.FLOAT

    def quantized_tensors(
        self, model: onnx.ModelProto, extra_active: Optional[Set[str]] = None
    ) -> List[str]:
        """Tensors that get a node, in first-use order (activations and
        constants alike)."""
        q: List[str] = []
        seen: Set[str] = set()

        def add(name: str) -> None:
            if name not in seen and self.is_float(name):
                seen.add(name)
                q.append(name)

        for n in model.graph.node:
            if n.op_type in self.gated:
                continue
            if n.op_type in ACTIVE_OPS or n.op_type in (extra_active or ()):
                # Resize's roi / scales / sizes are parameters, not data.
                for x in n.input[:1] if n.op_type == "Resize" else n.input:
                    add(x)
                for y in n.output:
                    add(y)
            elif n.op_type in PASS_THROUGH_OPS and n.input and n.input[0] in seen:
                for y in n.output:
                    add(y)
            elif n.op_type in FOLLOWING_ACTIVE_OPS and n.input and n.input[0] in seen:
                # active, but only behind an already quantized activation
                for x in n.input:
                    add(x)
                for y in n.output:
                    add(y)
        # A pass-through op feeding an active op: its output was added above as
        # that op's input; nothing to do for its own input (not quantized).
        return q

    def share_roots(
        self,
        model: onnx.ModelProto,
        quantized: Sequence[str],
        marking: Mapping[str, object],
    ) -> Dict[str, str]:
        """``{tensor: the tensor whose quantization parameters it reuses}``: the
        output of a data-movement op (Quark's ``quantize_output_same_as_input``,
        the QDQ direct / resize / pool / split quantizers) whose input is quantized
        too reads that input's scale and zero point, the chain's first tensor's."""
        qset = set(quantized)
        types = marking.get("op_types")
        roots: Dict[str, str] = {}
        for n in model.graph.node:
            if (
                n.op_type in _SHARING_OPS
                and n.input
                and n.input[0] in qset
                and (n.name or (n.output[0] if n.output else "")) not in self.skipped
                and (types is None or n.op_type in types)  # type: ignore[operator]
            ):
                for o in n.output:
                    if o in qset:
                        roots[o] = roots.get(n.input[0], n.input[0])
        return roots

    def marked_tensors(
        self,
        model: onnx.ModelProto,
        op_types: Optional[Iterable[str]],
        force_no_input_check: bool,
    ) -> List[str]:
        """Quark's marking (:func:`onnxsim.quark_marking.skipped_nodes`, the walk its
        op quantizers make over the nodes of the sorted graph): the float tensors of
        every node whose op type is in ``op_types`` and that does not skip itself,
        in the order the nodes marked them."""
        order: List[str] = []
        self.skipped = skipped_nodes(
            model,
            op_types,
            force_no_input_check=force_no_input_check,
            order=list(model.graph.node),
            unquantized_ops=self.gated,
            marked_out=order,
        )
        out: List[str] = []
        seen: Set[str] = set()
        for name in order:
            if name not in seen and self.is_float(name):
                seen.add(name)
                out.append(name)
        return out


# Quark drops the fake-quant pair between a producer and a directly following
# ReLU-like activation in its Q/DQ-based flows (FP16/BF16 here), since the
# pair is fused into one kernel on its target; the BFP/MX custom ops stay.
_FUSE_PRODUCERS = {
    "Conv",
    "Add",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "MatMul",
    "Gemm",
    "ConvTranspose",
}
_FUSE_ACTIVATIONS = {"Relu", "LeakyRelu", "PRelu"}


def _is_relu_clip(node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]) -> bool:
    if node.op_type != "Clip":
        return False
    bounds = []
    for name in node.input[1:3]:
        t = inits.get(name)
        if t is None or t.data_type != onnx.TensorProto.FLOAT:
            return False
        bounds.append(float(numpy_helper.to_array(t).reshape(-1)[0]))
    return len(bounds) == 2 and bounds[0] == 0.0 and bounds[1] in (1.0, 6.0)


def _fused_activation_inputs(
    model: onnx.ModelProto,
    inits: Dict[str, onnx.TensorProto],
    remove_after: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Tensors Quark leaves without a Q/DQ pair (``get_annotate_tensors``): one
    written by a producer and read only by one ReLU-like node (``remove_after``:
    the op types among ``Relu`` / ``Clip`` / ``LeakyRelu`` / ``PRelu`` its
    ``RemoveQDQConv*`` options keep on, default all), and a ``Pad``'s read only by
    an (Average) pool."""
    activations = (
        set(_FUSE_ACTIVATIONS | {"Clip"}) if remove_after is None else set(remove_after)
    )
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    graph_outputs = {o.name for o in model.graph.output}
    producer_out = {
        n.output[0]
        for n in model.graph.node
        if n.op_type in _FUSE_PRODUCERS and n.output
    }
    out: Set[str] = set()
    for n in model.graph.node:
        if not n.input or n.input[0] not in producer_out:
            continue
        if not (
            (n.op_type in _FUSE_ACTIVATIONS and n.op_type in activations)
            or ("Clip" in activations and _is_relu_clip(n, inits))
        ):
            continue
        t = n.input[0]
        if len(consumers.get(t, [])) == 1 and t not in graph_outputs:
            out.add(t)
    pad_out = {n.output[0] for n in model.graph.node if n.op_type == "Pad" and n.output}
    for n in model.graph.node:
        if (
            n.op_type in ("AveragePool", "GlobalAveragePool")
            and n.input
            and n.input[0] in pad_out
        ):
            t = n.input[0]
            if len(consumers.get(t, [])) == 1 and t not in graph_outputs:
                out.add(t)
    return out


def _refine_axes(
    model: onnx.ModelProto, quantized: Set[str], inits: Dict[str, onnx.TensorProto]
) -> Dict[str, int]:
    """Block axis per quantized tensor (Quark's ``refine_block_axis``)."""
    axis: Dict[str, int] = {}
    for t in quantized:
        axis[t] = 0 if t in inits and len(inits[t].dims) <= 1 else 1
    for n in model.graph.node:
        if n.op_type == "MatMul" and len(n.input) == 2:
            a, b = n.input
            if a in quantized:
                ax = len(inits[a].dims) - 1 if a in inits and len(inits[a].dims) else -1
                axis[a] = ax
                if n.output[0] in quantized:
                    axis[n.output[0]] = ax
            if b in quantized:
                axis[b] = (
                    len(inits[b].dims) - 2 if b in inits and len(inits[b].dims) else -2
                )
        elif n.op_type == "Gemm" and len(n.input) >= 2:
            a, b = n.input[0], n.input[1]
            if a in quantized:
                axis[a] = 0 if _attr(n, "transA", 0) else 1
            if b in quantized:
                axis[b] = 1 if _attr(n, "transB", 0) else 0
        elif n.op_type == "Softmax" and n.input and n.input[0] in quantized:
            axis[n.input[0]] = int(_attr(n, "axis", -1))
    return axis


def drop_unused_initializers(model: onnx.ModelProto) -> None:
    """Quark's ``clean_initializers``: drop the constants (and graph inputs of the
    same name) nothing reads."""
    used = {o.name for o in model.graph.output}

    def walk(g: onnx.GraphProto) -> None:
        for n in g.node:
            used.update(x for x in n.input if x)
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sub in a.graphs:
                        walk(sub)

    walk(model.graph)
    keep = [t for t in model.graph.initializer if t.name in used]
    if len(keep) != len(model.graph.initializer):
        gone = {t.name for t in model.graph.initializer} - {t.name for t in keep}
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
        inputs = [i for i in model.graph.input if i.name not in gone]
        del model.graph.input[:]
        model.graph.input.extend(inputs)


def apply_fake_quant_format(
    model: onnx.ModelProto,
    dtype: str,
    activations: bool = True,
    fold_weights: bool = False,
    fold_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    exclude: Sequence[str] = (),
    const_dtype: Optional[str] = None,
    quantize_all_ops: bool = True,
    attr_overrides: Optional[Mapping[str, Mapping[str, object]]] = None,
    marking: Optional[Mapping[str, object]] = None,
    remove_after: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Return ``model`` with fake-quantization nodes inserted (see the module
    docstring).

    :param dtype: a block format (``"bfp16"`` / ``"mx9"`` / ``"mxfp8_e4m3"`` ...)
            or ``"float16"`` / ``"bfloat16"``
    :param activations: insert nodes on activation tensors; when False only the
            constants are quantized (always folded offline, since no custom op
            may be emitted)
    :param fold_weights: fake-quantize constants offline (needs ``fold_fn``)
            instead of inserting a node on them
    :param fold_fn: ``f(array, axis) -> array``, the numpy fake-quantizer
    :param exclude: node names / first-output names left entirely alone
    :param const_dtype: a block format for the *constants* (weights and biases)
            while the activations use ``dtype`` -- Quark's ``BF16_BFP16`` /
            ``BF16_MXINT8`` (bfloat16 activations, block-format constants).
            Axes are refined as for that block format.
    :param quantize_all_ops: for ``float16`` / ``bfloat16``: also quantize the
            wider op set of Quark's ``FP16`` / ``BF16`` presets
            (``QuantizeAllOpTypes``); False gives the block formats' op
            coverage with half-precision quantizers. Always off with
            ``const_dtype``.
    :param attr_overrides: ``{op type: {attribute: value}}`` applied to the
            block-format nodes (Quark's ``BFPAttributes`` / ``MXAttributes``)
    :param marking: ``{"op_types": ..., "force_no_input_check": bool}``: pick the
            tensors with Quark's own marking walk (the op types its quantizer is
            asked to quantize, whether its direct ops go without an input check)
            instead of the fixed op lists above
    :param remove_after: for the half-precision pairs, the activation op types
            (``Relu`` / ``Clip`` / ``LeakyRelu`` / ``PRelu``) whose producer's output
            goes without a pair (Quark's ``RemoveQDQConv*`` options); default all
    """
    half = dtype in HALF_DTYPES
    if not half:
        node_spec(dtype)  # validates the block format
    if const_dtype is not None:
        node_spec(const_dtype)
        if fold_weights or not activations:
            raise ValueError("const_dtype nodes cannot be combined with folding")
    m = onnx.ModelProto()
    m.CopyFrom(model)
    fold = fold_weights or not activations
    if fold and fold_fn is None:
        raise ValueError("fold_fn is required to fold constants offline")
    g = m.graph
    excluded = set(exclude)
    work = onnx.ModelProto()
    work.CopyFrom(m)
    if excluded:
        keep = [
            n
            for n in work.graph.node
            if n.name not in excluded and (not n.output or n.output[0] not in excluded)
        ]
        del work.graph.node[:]
        work.graph.node.extend(keep)

    plan = _Plan(m)
    extra = HALF_EXTRA_OPS.get(dtype) if quantize_all_ops and not const_dtype else None
    if marking is not None:
        quantized = plan.marked_tensors(
            work,
            marking.get("op_types"),  # type: ignore[arg-type]
            bool(marking.get("force_no_input_check", True)),
        )
    else:
        quantized = plan.quantized_tensors(work, extra)
    if half:
        # (structural rules of Quark's post-processing: they read the whole graph, the
        # excluded nodes included)
        fused = _fused_activation_inputs(m, plan.inits, remove_after)
        quantized = [t for t in quantized if t not in fused]
    axes = _refine_axes(m, set(quantized), plan.inits)
    roots = plan.share_roots(work, quantized, marking) if marking is not None else {}
    consts = [t for t in quantized if t in plan.inits]
    acts = [t for t in quantized if t not in plan.inits]
    if marking is not None and dtype == "bfloat16" and const_dtype is None:
        # Quark's bfloat16 constants "avoid the NaN issue due to overflow": a tensor
        # with a magnitude outside bfloat16's normal range is clipped into it (a zero
        # stays zero)
        for c in consts:
            w = numpy_helper.to_array(plan.inits[c])
            if (
                w.dtype.kind == "f"
                and w.size
                and (
                    np.max(np.abs(w)) > 3.38953139e38
                    or np.min(np.abs(w)) < 1.17549435e-38
                )
            ):
                clipped = (
                    np.sign(w) * np.clip(np.abs(w), 1.17549435e-38, 3.38953139e38)
                ).astype(w.dtype)
                plan.inits[c].CopyFrom(numpy_helper.from_array(clipped, c))

    def fake_quant(t: str, src: str, dst: str) -> List[onnx.NodeProto]:
        """The node(s) quantizing tensor ``t``: ``src`` -> ``dst``."""
        if const_dtype is not None and t in plan.inits:
            return [
                _make_node(
                    const_dtype,
                    src,
                    dst,
                    axes[t],
                    t + "_DequantizeLinear",
                    attr_overrides,
                )
            ]
        if half:
            nodes, extra = _make_half_pair(dtype, src, dst, t, roots.get(t))
            g.initializer.extend(extra)
            return nodes
        return [
            _make_node(
                dtype, src, dst, axes[t], t + "_DequantizeLinear", attr_overrides
            )
        ]

    in_rename: Dict[str, str] = {}  # tensor -> what its consumers read instead
    out_rename: Dict[str, str] = {}  # graph output -> what its producer writes
    const_nodes: List[onnx.NodeProto] = []
    for c in consts:
        if fold:
            assert fold_fn is not None
            arr = numpy_helper.to_array(plan.inits[c])
            plan.inits[c].CopyFrom(
                numpy_helper.from_array(fold_fn(arr, axes[c]).astype(np.float32), c)
            )
        else:
            in_rename[c] = c + DQ_SUFFIX
            const_nodes += fake_quant(c, c, in_rename[c])

    graph_outputs = {o.name for o in g.output}
    act_nodes: Dict[str, List[onnx.NodeProto]] = {}  # tensor -> its node(s)
    if activations:
        for t in acts:
            if t in graph_outputs:
                out_rename[t] = t + Q_INPUT_SUFFIX
                act_nodes[t] = fake_quant(t, out_rename[t], t)
            else:
                in_rename[t] = t + DQ_SUFFIX
                act_nodes[t] = fake_quant(t, t, in_rename[t])

    # -- rewire (every consumer reads the dequantized tensor; a graph output's
    # producer writes the pre-quantization name) ---------------------------------
    nodes = list(g.node)
    produced: Set[str] = {o for n in nodes for o in n.output}
    originals: Dict[int, List[str]] = {}
    for n in nodes:
        originals[id(n)] = list(n.output)
        for i, x in enumerate(n.input):
            if x in in_rename:
                n.input[i] = in_rename[x]
        for i, o in enumerate(n.output):
            if o in out_rename:
                n.output[i] = out_rename[o]

    # -- order: constants' nodes and graph-input nodes first, every other
    # activation's node(s) right after its producer ------------------------------
    ordered: List[onnx.NodeProto] = list(const_nodes)
    for t in acts:
        if t in act_nodes and t not in produced:  # a graph input
            ordered += act_nodes[t]
    for n in nodes:
        ordered.append(n)
        for o in originals[id(n)]:
            if o in act_nodes:
                ordered += act_nodes[o]
    del g.node[:]
    g.node.extend(ordered)
    if (const_nodes or act_nodes) and not any(
        o.domain == COP_DOMAIN for o in m.opset_import
    ):
        m.opset_import.append(onnx.helper.make_opsetid(COP_DOMAIN, 1))
    if marking is not None:
        # (Quark's quantizers end with ``clean_initializers``: a constant nothing
        # reads any more -- one the pre-processing left behind -- is dropped)
        drop_unused_initializers(m)
    # (Quark's quantizers end with its own topological sort)
    return quark_sorted(m)


__all__ = [
    "ACTIVE_OPS",
    "COP_DOMAIN",
    "PASS_THROUGH_OPS",
    "HALF_DTYPES",
    "HALF_EXTRA_OPS",
    "apply_fake_quant_format",
    "drop_unused_initializers",
    "node_spec",
]
