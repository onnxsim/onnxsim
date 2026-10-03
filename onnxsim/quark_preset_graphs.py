"""Graph builders for the Quark presets that combine two number formats
(``MX9_INT8`` and the ``BF16_MIXED_*`` presets), on top of
:mod:`onnxsim.quark_fakequant_graph`.

Everything here was derived by running AMD Quark on probe graphs and is
re-checked against the installed ``amd-quark`` by ``tests/test_quark_parity.py``
(the "preset combinations" section).

``MX9_INT8``
    Activations get the same ``BFPQuantizeDequantize`` (``to_bfp_prime``, 16
    bits) nodes as the plain ``MX9`` preset; every constant that preset would
    quantize (weights *and* biases) is instead stored as an ``int8`` initializer
    with a symmetric per-tensor scale ``max|w| / 127`` and read through a
    ``com.microsoft`` ``DequantizeLinear`` -- the offline-quantized form Quark's
    int8 path emits.

``BF16_MIXED_BFP16`` / ``BF16_MIXED_MXINT8``
    A bfloat16 model (weights, activations; biases are *not* quantized) in which
    every ``Conv`` / ``ConvTranspose`` / ``Gemm`` / ``MatMul`` is promoted to a
    block format (Quark runs its AutoMixprecision with the threshold disabled,
    so every candidate is promoted -- there is no sensitivity ranking to
    reproduce). Promoting a node replaces the bfloat16 quantizers on its
    activation inputs and weight with block-format nodes (default block axis 1,
    no axis refinement); its outputs stay bfloat16. Wherever a promoted node
    meets a bfloat16 quantizer on the other side of a tensor ("precision
    boundary") an extra node of the promoted format is inserted -- Quark's
    ``DualQuantNodes``. See :func:`apply_mixed_block_format` for the exact rule.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnxsim.quark_fakequant_graph import (
    COP_DOMAIN,
    DQ_SUFFIX,
    _make_node,
    _Plan,
    apply_fake_quant_format,
)
from onnxsim.quark_marking import quark_sorted

MS_DOMAIN = "com.microsoft"
PROMOTABLE_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")


def _add_opset(model: onnx.ModelProto, domain: str) -> None:
    if not any(o.domain == domain for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid(domain, 1))


# -- MX9_INT8 --------------------------------------------------------------------


def apply_block_activations_int8_constants(
    model: onnx.ModelProto,
    act_dtype: str,
    exclude: Sequence[str] = (),
    marking: Optional[Mapping[str, object]] = None,
) -> onnx.ModelProto:
    """``act_dtype`` (a block format) fake-quantization on the activations,
    int8 symmetric per-tensor constants (see the module docstring); ``marking``
    as for :func:`apply_fake_quant_format`."""
    return apply_fake_quant_int_constants(
        model, act_dtype, "int8", exclude, marking=marking
    )


_INT_WEIGHT_OPS = ("Conv", "ConvTranspose", "Gemm")


def apply_fake_quant_int_constants(
    model: onnx.ModelProto,
    act_dtype: str,
    const_dtype: str = "int8",
    exclude: Sequence[str] = (),
    attr_overrides: Optional[Dict[str, Dict[str, object]]] = None,
    weight_symmetric: Optional[bool] = None,
    int32_bias: bool = True,
    quantize_bias: bool = True,
    marking: Optional[Mapping[str, object]] = None,
) -> onnx.ModelProto:
    """Fake-quantization in ``act_dtype`` (a block format or ``float16`` /
    ``bfloat16``) on the activations, ``int8`` / ``uint8`` per-tensor constants
    read through a ``com.microsoft`` ``DequantizeLinear`` (the offline-quantized
    form Quark's integer weight path emits).

    The weights take ``compute_scale_zp`` over their own range (symmetric by
    default for int8, asymmetric for uint8). A bias is stored int32 with scale
    ``weight scale * 1.0`` -- the half-precision activations' scale is 1.0 --
    when ``int32_bias`` and the activations are ``float16`` / ``bfloat16``;
    behind block-format activations (which have no scale) it is quantized like a
    weight, from its own range. ``quantize_bias=False`` leaves biases float.
    """
    from onnxsim.quark_fakequant_graph import HALF_DTYPES
    from onnxsim.quark_mixing import compute_scale_zp

    if const_dtype not in ("int8", "uint8"):
        raise ValueError("const_dtype must be int8 or uint8")
    sym = const_dtype == "int8" if weight_symmetric is None else weight_symmetric
    plan = _Plan(model)
    work = onnx.ModelProto()
    work.CopyFrom(model)
    if marking is not None:
        consts = [
            t
            for t in plan.marked_tensors(
                work,
                marking.get("op_types"),  # type: ignore[arg-type]
                bool(marking.get("force_no_input_check", False)),
            )
            if t in plan.inits
        ]
    else:
        consts = [t for t in plan.quantized_tensors(work) if t in plan.inits]
    m = apply_fake_quant_format(
        model,
        act_dtype,
        activations=True,
        fold_weights=True,
        fold_fn=lambda a, ax: a,  # constants are handled below, not folded
        exclude=exclude,
        attr_overrides=attr_overrides,
        marking=marking,
    )
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    half_act = act_dtype in HALF_DTYPES
    np_dtype = np.int8 if const_dtype == "int8" else np.uint8
    qmin, qmax = (-128, 127) if const_dtype == "int8" else (0, 255)
    bias_of: Dict[str, str] = {}  # bias constant -> its weight constant
    for n in model.graph.node:
        if n.op_type in _INT_WEIGHT_OPS and len(n.input) > 2 and n.input[2]:
            bias_of.setdefault(n.input[2], n.input[1])
    rename: Dict[str, str] = {}
    new_nodes: List[onnx.NodeProto] = []
    weight_scale: Dict[str, np.float32] = {}

    def emit(
        c: str,
        codes: np.ndarray,
        scale: np.ndarray,
        zp: np.ndarray,
        tag: str = "",
    ) -> None:
        """The codes ``<c>_quantized`` read through a DequantizeLinear (the
        int32 biases name their scale / zero point ``<c>_quantized_*``)."""
        names = [c + "_quantized", c + tag + "_scale", c + tag + "_zero_point"]
        for name, arr in zip(names, (codes, scale, zp)):
            g.initializer.append(numpy_helper.from_array(arr, name))
        rename[c] = c + DQ_SUFFIX
        new_nodes.append(
            onnx.helper.make_node(
                "DequantizeLinear",
                names,
                [rename[c]],
                name=c + "_DequantizeLinear",
                domain=MS_DOMAIN,
            )
        )

    def own_range(w: np.ndarray):
        if not w.size:
            return np.array(0, np_dtype), np.array(1.0, np.float32)
        return compute_scale_zp(
            np.asarray(w.min(), np.float32),
            np.asarray(w.max(), np.float32),
            const_dtype,
            bool(sym),
        )

    # weights first: an int32 bias takes its weight's scale
    for c in sorted(consts, key=lambda t: t in bias_of):
        w = numpy_helper.to_array(inits[c]).astype(np.float32)
        if c in bias_of:
            if not quantize_bias:
                continue
            if half_act and int32_bias:
                if bias_of[c] not in weight_scale:
                    continue  # its weight stays float
                scale = np.float32(np.float32(1.0) * weight_scale[bias_of[c]])
                q = np.clip(np.round(w / scale), -(2**31), 2**31 - 1).astype(np.int32)
                emit(
                    c,
                    q,
                    np.array([scale], np.float32),
                    np.array(0, np.int32),
                    "_quantized",
                )
                continue
        zp, scale = own_range(w)
        scale = np.asarray(scale, np.float32).reshape(())
        zp = np.asarray(zp, np_dtype).reshape(())
        q = np.clip(np.round(w / scale) + zp.astype(np.float32), qmin, qmax)
        if c not in bias_of:
            weight_scale[c] = np.float32(scale)
        emit(c, q.astype(np_dtype), scale, zp)
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    used = {x for n in g.node for x in n.input} | {o.name for o in g.output}
    keep = [t for t in g.initializer if t.name not in rename or t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    if new_nodes:
        _add_opset(m, MS_DOMAIN)
    # (Quark's quantizers end with its own topological sort)
    return quark_sorted(m) if marking is not None else m


# -- BF16_MIXED_* -----------------------------------------------------------------

_Q_OPS = ("QuantizeLinear", "ExtendedQuantizeLinear")
_DQ_OPS = ("DequantizeLinear", "ExtendedDequantizeLinear")
_FN_OPS = ("BFPQuantizeDequantize", "MXQuantizeDequantize")
_BIAS_OPS = ("Conv", "ConvTranspose", "Gemm", "InstanceNormalization")


def _drop_bias_quantizers(model: onnx.ModelProto) -> None:
    """Remove the bfloat16 Q/DQ pairs on Conv / ConvTranspose / Gemm bias
    constants (Quark's mixed presets set ``QuantizeBias=False``)."""
    g = model.graph
    nodes = list(g.node)
    inits = {t.name for t in g.initializer}
    biases = {  # tensors read in a bias slot (the DQ outputs of the bias constants)
        n.input[2] for n in nodes if n.op_type in _BIAS_OPS and len(n.input) > 2
    }
    q_by_out = {n.output[0]: n for n in nodes if n.op_type in _Q_OPS}
    drop: Set[int] = set()
    rename: Dict[str, str] = {}
    gone: Set[str] = set()
    for n in nodes:
        if n.op_type in _DQ_OPS and n.input[0] in q_by_out:
            q = q_by_out[n.input[0]]
            if n.output[0] in biases and q.input[0] in inits:
                rename[n.output[0]] = q.input[0]
                drop.update((id(n), id(q)))
                gone.update(n.input[1:3])
    if not drop:
        return
    keep = [n for n in nodes if id(n) not in drop]
    for n in keep:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    del g.node[:]
    g.node.extend(keep)
    used = {x for n in keep for x in n.input}
    kept_inits = [t for t in g.initializer if t.name not in gone or t.name in used]
    del g.initializer[:]
    g.initializer.extend(kept_inits)


def _node_attrs(n: onnx.NodeProto) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        sorted((a.name, repr(onnx.helper.get_attribute_value(a))) for a in n.attribute)
    )


def _zp_dtype(inits: Dict[str, onnx.TensorProto], n: onnx.NodeProto) -> int:
    if len(n.input) >= 3 and n.input[2] in inits:
        return inits[n.input[2]].data_type
    return 0


Stage = Tuple[str, tuple, Tuple[onnx.NodeProto, ...]]


def _with_node_names(model: onnx.ModelProto) -> onnx.ModelProto:
    if all(n.name for n in model.graph.node):
        return model
    m = onnx.ModelProto()
    m.CopyFrom(model)
    taken = {n.name for n in m.graph.node if n.name}
    for i, n in enumerate(m.graph.node):
        if not n.name:
            name = f"{n.op_type}_{i}"
            while name in taken:
                name += "_"
            taken.add(name)
            n.name = name
    return m


def apply_mixed_block_format(
    model: onnx.ModelProto,
    block_dtype: str,
    exclude: Sequence[str] = (),
    target_ops: Sequence[str] = PROMOTABLE_OPS,
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
    dual_nodes: bool = True,
) -> onnx.ModelProto:
    """Quark's ``BF16_MIXED_<block_dtype>``: a bfloat16 fake-quantized model
    (biases left alone) whose ``Conv`` / ``ConvTranspose`` / ``Gemm`` /
    ``MatMul`` nodes are promoted to ``block_dtype`` (``"bfp16"`` or
    ``"mxint8"``), with dual nodes at the precision boundaries.

    Promotion: for each target node, the quantizer feeding input 0 and input 1
    (weight or second activation) is replaced by a block-format node (default
    block axis, attributes of the format); the bias and the outputs are left.
    Boundaries: for every other node, each tensor edge whose quantizer differs
    from the node's *template* gets a copy of the template inserted --
    the template of a promoted node is its first edge that carries a promoted
    tensor, the template of any other node is its first edge that does not.
    So a promoted node's output into a bfloat16 quantizer gets a block node in
    front of it, and a bfloat16 node reading a promoted tensor gets a
    bfloat16 pair behind the block node. ``dual_nodes=False`` (Quark's
    ``DualQuantNodes=False``, and the model its sensitivity analysis scores)
    skips the boundaries and only swaps the promoted nodes' own slots.
    """
    # Quark identifies candidate layers by node name (and misbehaves on unnamed
    # nodes); give every unnamed node a unique name so that all are promoted.
    model = _with_node_names(model)
    float_names = (
        {o for n in model.graph.node for o in n.output}
        | {v.name for v in model.graph.input}
        | {v.name for v in model.graph.output}
    )
    m = apply_fake_quant_format(
        model, "bfloat16", exclude=exclude, quantize_all_ops=False
    )
    _drop_bias_quantizers(m)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    nodes: List[onnx.NodeProto] = list(g.node)
    node_names = {n.name for n in nodes if n.name}
    tensor_names = (
        {x for n in nodes for x in list(n.input) + list(n.output) if x}
        | set(inits)
        | {v.name for v in g.input}
        | {v.name for v in g.output}
    )
    excluded = set(exclude)

    def unique(base: str, existing: Set[str]) -> str:
        name, i = base, 1
        while name in existing:
            name = f"{base}_{i}"
            i += 1
        existing.add(name)
        return name

    producer = {o: n for n in nodes for o in n.output}
    promoted_nodes: Set[str] = set()
    promoted_tensors: Set[str] = set()
    removed: Set[int] = set()
    replaced_at: Dict[int, onnx.NodeProto] = {}  # id(old DQ) -> its block node

    # -- promotion -----------------------------------------------------------------
    for n in nodes:
        if n.op_type not in target_ops or n.name in excluded:
            continue
        if (
            include_layers and n.name not in include_layers
        ) or n.name in exclude_layers:
            continue
        promoted_nodes.add(n.name or n.op_type)
        for slot in (0, 1):
            if slot >= len(n.input) or not n.input[slot]:
                continue
            prod = producer.get(n.input[slot])
            if prod is None:
                continue
            if prod.op_type in _DQ_OPS:
                q = producer.get(prod.input[0])
                if q is None or q.op_type not in _Q_OPS or id(prod) in removed:
                    continue
                fn = _make_node(block_dtype, q.input[0], prod.output[0], 1, "")
                fn.name = prod.name + "_Mixed_fn"
                removed.update((id(q), id(prod)))
                replaced_at[id(prod)] = fn
                producer[prod.output[0]] = fn
                promoted_tensors.add(q.input[0])
            elif prod.op_type in _FN_OPS and prod.domain == COP_DOMAIN:
                # a second consumer promoting a shared tensor swaps the block
                # node for a fresh one (Quark names it ``<old name>_Mixed``)
                prod.name += "_Mixed"
                promoted_tensors.add(prod.input[0])
    if not promoted_nodes:
        return m
    nodes = [
        replaced_at.get(id(n), n)
        for n in nodes
        if id(n) not in removed or id(n) in replaced_at
    ]
    if not dual_nodes:  # Quark's ``DualQuantNodes=False``: the promoted slots only
        del g.node[:]
        g.node.extend(nodes)
        return m

    # -- dual nodes at the precision boundaries ------------------------------------
    producer = {o: n for n in nodes for o in n.output}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)

    def q_sig(q: onnx.NodeProto) -> tuple:
        return ("quant", f"{q.domain or 'ai.onnx'}::{q.op_type}", _zp_dtype(inits, q))

    def fn_sig(f: onnx.NodeProto) -> tuple:
        return ("fn", f"{f.domain}::{f.op_type}", _node_attrs(f))

    def upstream(x: str) -> Optional[Stage]:
        """The quantizer behind tensor ``x`` (``nodes`` = its Q+DQ, or the block node)."""
        prod = producer.get(x)
        if prod is not None and prod.op_type in _DQ_OPS:
            q = producer.get(prod.input[0])
            if q is not None and q.op_type in _Q_OPS:
                return "pair", q_sig(q), (q, prod)
        elif prod is not None and prod.op_type in _FN_OPS and prod.domain == COP_DOMAIN:
            return "fn", fn_sig(prod), (prod,)
        return None

    def downstream(c: onnx.NodeProto) -> Optional[Stage]:
        """The quantizer ``c`` starts, if it is one."""
        if c.op_type in _Q_OPS:
            dq = next(
                (d for d in consumers.get(c.output[0], []) if d.op_type in _DQ_OPS),
                None,
            )
            return ("pair", q_sig(c), (c, dq)) if dq is not None else None
        if c.op_type in _FN_OPS and c.domain == COP_DOMAIN:
            return "fn", fn_sig(c), (c,)
        return None

    def override(*names: str) -> Optional[str]:
        return next((x for x in names if x and x in promoted_tensors), None)

    infos_by_node: Dict[str, List[Dict]] = {}
    for idx, n in enumerate(nodes):
        if n.op_type in _Q_OPS + _DQ_OPS + _FN_OPS:
            continue
        infos: List[Dict] = infos_by_node.setdefault(f"{n.name or n.op_type}_{idx}", [])
        disp = n.name or n.op_type
        for ii, x in enumerate(n.input):
            st = upstream(x) if x else None
            if st is None:
                continue
            source = st[2][0].input[0]  # the float tensor the quantizer reads
            if source in inits:
                continue
            infos.append(
                dict(
                    node=disp,
                    kind=st[0],
                    sig=st[1],
                    nodes=st[2],
                    tensor=x,
                    override=override(x, source),
                    calib=source,
                    index=ii,
                    target=n,
                    template=-1,
                )
            )
        for oi, y in enumerate(n.output):
            for c in consumers.get(y, []):
                st = downstream(c)
                if st is None:
                    continue
                after = st[2][-1].output[0]
                infos.append(
                    dict(
                        node=disp,
                        kind=st[0],
                        sig=st[1],
                        nodes=st[2],
                        tensor=y,
                        override=override(y, after),
                        calib=after,
                        index=oi,
                        target=c,
                        template=-1,
                    )
                )

    def find_template(cur: List[Dict]) -> Optional[Dict]:
        """A node surrounded only by promoted tensors borrows the template of
        a neighbour that shares the tensor."""
        for st in cur:
            for infos in infos_by_node.values():
                for info in infos:
                    if info["override"] == st["override"] and (
                        info["tensor"] == st["override"]
                        or info["override"] == st["tensor"]
                    ):
                        if info["template"] < 0:
                            continue
                        return infos[info["template"]]
        return None

    inserts: Dict[int, List[onnx.NodeProto]] = {}  # id(target) -> nodes in front of it
    done: Set[Tuple[str, str, str]] = set()
    for infos in infos_by_node.values():
        if not infos:
            continue
        nname = infos[0]["node"]
        t_idx = -1
        for k, info in enumerate(infos):
            if nname not in promoted_nodes:
                if not info["override"]:
                    t_idx = k
                    break
            elif info["override"]:
                t_idx = k
                break
        tmpl = infos[t_idx] if t_idx >= 0 else find_template(infos)
        if tmpl is None:
            continue
        for info in infos:
            if info is tmpl or info["sig"] == tmpl["sig"]:
                continue
            info["template"] = t_idx
            tgt = info["target"]
            tname = tgt.name or tgt.op_type
            if (nname, info["tensor"], tname) in done:
                continue
            done.add((nname, info["tensor"], tname))
            primary = info["override"] or info["tensor"]
            # Quark names the extra pair's parameters after a calibrated (i.e.
            # original float-model) tensor
            qparam = primary if primary in float_names else info["calib"]
            scope = f"{qparam}_{nname}_{info['index']}_{tname}"
            if tmpl["kind"] == "pair":
                tq, tdq = tmpl["nodes"]
                nq, ndq = onnx.NodeProto(), onnx.NodeProto()
                nq.CopyFrom(tq)
                ndq.CopyFrom(tdq)
                nq.name = unique(f"{scope}_additional_{tq.op_type}", node_names)
                ndq.name = unique(f"{scope}_additional_{tdq.op_type}", node_names)
                qout = unique(f"{nq.name}_output", tensor_names)
                last = unique(f"{ndq.name}_output", tensor_names)
                sc = unique(f"{qparam}_additional_scale", tensor_names)
                zp = unique(f"{qparam}_additional_zero_point", tensor_names)
                g.initializer.append(
                    numpy_helper.from_array(np.array(1.0, np.float32), sc)
                )
                g.initializer.append(
                    onnx.helper.make_tensor(zp, inits[tq.input[2]].data_type, [], [0.0])
                )
                nq.input[:] = [info["tensor"], sc, zp]
                nq.output[0] = qout
                ndq.input[:] = [qout, sc, zp]
                ndq.output[0] = last
                new = [nq, ndq]
            else:
                fnode = onnx.NodeProto()
                fnode.CopyFrom(tmpl["nodes"][0])
                fnode.name = unique(f"{scope}_additional_{fnode.op_type}", node_names)
                last = unique(f"{fnode.name}_output", tensor_names)
                fnode.input[0] = info["tensor"]
                fnode.output[0] = last
                new = [fnode]
            for i, x in enumerate(tgt.input):
                if x == info["tensor"]:
                    tgt.input[i] = last
            inserts.setdefault(id(tgt), []).extend(new)
    final: List[onnx.NodeProto] = []
    for n in nodes:
        final.extend(inserts.get(id(n), []))
        final.append(n)
    del g.node[:]
    g.node.extend(final)
    return m


# -- Int32Bias=False (S16S16_MIXED_S8S8, VINT8) -------------------------------------


def requantize_biases_int8(
    model: onnx.ModelProto,
    float_model: onnx.ModelProto,
    target_ops: Sequence[str] = PROMOTABLE_OPS,
    include_layers: Sequence[str] = (),
    exclude_layers: Sequence[str] = (),
    power_of_two: bool = False,
    dtype: str = "int8",
    per_channel: bool = False,
    symmetric: bool = True,
) -> onnx.ModelProto:
    """Replace the int32 bias (``input_scale * weight_scale``) of every
    promoted node by Quark's ``Int32Bias=False`` form: the bias quantized like
    a weight of ``dtype`` (``"int8"`` or ``"int16"``), symmetric per tensor,
    scale ``max|b| / qmax`` (rounded up to a power of two with
    ``power_of_two``, as Quark's ``VINT8``), zero point 0. The codes come from
    the float model's bias (the int32 form is too coarse to recover them
    from). ``per_channel``: one scale per element (``axis=0``); a not
    ``symmetric`` bias takes the asymmetric grid over its range."""
    qmax = {"int8": 127, "int16": 32767}[dtype]
    np_dt = {"int8": np.int8, "int16": np.int16}[dtype]
    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0
    )
    ms = dtype == "int16" and opset < 21
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    float_inits = {t.name: t for t in float_model.graph.initializer}
    dq_by_out = {
        n.output[0]: n for n in g.node if n.op_type == "DequantizeLinear" and n.output
    }
    drop: Set[str] = set()
    for n in g.node:
        if n.op_type not in target_ops or len(n.input) < 3:
            continue
        if (
            include_layers and n.name not in include_layers
        ) or n.name in exclude_layers:
            continue
        dq = dq_by_out.get(n.input[2])
        if dq is None or dq.input[0] not in inits:
            continue
        q = inits[dq.input[0]]
        if q.data_type != onnx.TensorProto.INT32:
            continue
        base = dq.input[0].rsplit("/", 2)[0]  # "b1/qdq8/int32" -> "b1"
        if base in float_inits:
            b = numpy_helper.to_array(float_inits[base]).astype(np.float64)
        else:
            b = numpy_helper.to_array(q).astype(np.float64) * numpy_helper.to_array(
                inits[dq.input[1]]
            ).astype(np.float64)
        names = (base + "_quantized", base + "_scale", base + "_zero_point")
        if per_channel and b.ndim == 1 and b.size > 1:
            scale = (np.maximum(np.abs(b), 1e-12) / qmax).astype(np.float32)
            if power_of_two:
                scale = (2.0 ** np.ceil(np.log2(scale))).astype(np.float32)
            codes = np.clip(np.round(b / scale), -qmax, qmax).astype(np_dt)
            zero = np.zeros(scale.shape, np_dt)
        elif not symmetric and not power_of_two:
            from onnxsim.full_qdq import _weight_qparams

            s32, z = _weight_qparams(b.min(), b.max(), -qmax - 1, qmax, False)
            scale = np.float32(s32)
            codes = np.clip(np.round(b / scale) + z, -qmax, qmax).astype(np_dt)
            zero = np.array(z, np_dt)
        else:
            amax = float(np.max(np.abs(b))) if b.size else 0.0
            scale = np.float32(amax / qmax) if amax > 0 else np.float32(1.0)
            if power_of_two and amax > 0:
                scale = np.float32(2.0 ** np.ceil(np.log2(float(scale))))
            codes = np.clip(np.round(b / scale), -qmax - 1, qmax).astype(np_dt)
            zero = np.array(0, np_dt)
        g.initializer.extend(
            [
                numpy_helper.from_array(codes, names[0]),
                numpy_helper.from_array(np.asarray(scale, np.float32), names[1]),
                numpy_helper.from_array(zero, names[2]),
            ]
        )
        drop.update(dq.input)
        dq.input[:] = list(names)
        del dq.attribute[:]
        if per_channel and b.ndim == 1 and b.size > 1:
            dq.attribute.append(onnx.helper.make_attribute("axis", 0))
        dq.domain = "com.microsoft" if ms else ""
    if ms and not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    used = {x for node in g.node for x in node.input} | {o.name for o in g.output}
    kept = [t for t in g.initializer if t.name in used or t.name not in drop]
    del g.initializer[:]
    g.initializer.extend(kept)
    return m


# -- VINT8 ---------------------------------------------------------------------------


def dedicate_qdq_pairs(
    model: onnx.ModelProto, receivers: Optional[Set[str]] = None
) -> onnx.ModelProto:
    """Quark's ``DedicatedQDQPair``: an activation ``Q -> DQ`` pair read by
    several nodes is replaced by one pair (same scale and zero point) per
    consumer, so each consumer owns the quantizer in front of it. The DQ of a
    graph output stays as it is.

    Quark counts the *quantized* nodes that read the tensor (``receivers``: their
    names, default every named node; a node reading the tensor twice counts
    twice, and takes the first pair -- the other is left unused): with more than
    one, only those get a pair and any other reader sees the float tensor; with
    one or none the single pair stays in front of every reader."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    nodes = list(g.node)
    inits = {t.name for t in g.initializer}
    outputs = {o.name for o in g.output}
    q_by_out = {
        n.output[0]: n
        for n in nodes
        if n.op_type == "QuantizeLinear" and n.input[0] not in inits
    }
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    taken = {x for n in nodes for x in list(n.input) + list(n.output)} | inits
    drop: Set[int] = set()
    inserts: Dict[int, List[onnx.NodeProto]] = {}  # id(consumer) -> nodes before it

    def receiving(u: onnx.NodeProto) -> bool:
        return receivers is None or not u.name or u.name in receivers

    for dq in nodes:
        if dq.op_type != "DequantizeLinear" or dq.input[0] not in q_by_out:
            continue
        q = q_by_out[dq.input[0]]
        readers = consumers.get(dq.output[0], [])  # one entry per input slot
        users = list({id(u): u for u in readers}.values())
        # (Quark's list: one entry per input slot of a node it quantizes)
        slots = [u for u in readers if receiving(u)]
        if len(slots) < 2:
            continue
        if dq.output[0] in outputs:
            # a graph output read by several quantized nodes: each gets its own
            # pair and the graph output stays the float tensor (Quark's
            # dedicated branch comes before its graph-output one)
            raw = q.input[0]
            for n in nodes:
                if n is dq:
                    continue
                for i, x in enumerate(n.input):
                    if x == raw:
                        n.input[i] = dq.output[0]
                for i, x in enumerate(n.output):
                    if x == raw:
                        n.output[i] = dq.output[0]
        drop.update((id(q), id(dq)))
        first_slot: Dict[int, int] = {}
        pairs: List[onnx.NodeProto] = []
        for k, u in enumerate(slots, 1):
            qn, dqn = onnx.NodeProto(), onnx.NodeProto()
            qn.CopyFrom(q)
            dqn.CopyFrom(dq)
            qn.name, dqn.name = f"{q.name}_{k}", f"{dq.name}_{k}"
            qn.output[0] = f"{q.output[0]}_{k}"
            dqn.input[0] = qn.output[0]
            dqn.output[0] = f"{dq.output[0]}_{k}"
            if qn.output[0] in taken or dqn.output[0] in taken:
                raise ValueError(f"tensor name clash while duplicating {dq.name}")
            if id(u) not in first_slot:
                first_slot[id(u)] = k
                for i, x in enumerate(u.input):
                    if x == dq.output[0]:
                        u.input[i] = dqn.output[0]
                inserts.setdefault(id(u), []).extend([qn, dqn])
            else:
                pairs.extend([qn, dqn])  # a second slot of the same node: unused
        for u in users:
            if not receiving(u):
                for i, x in enumerate(u.input):
                    if x == dq.output[0]:
                        u.input[i] = q.input[0]
        if pairs:
            # (placed in front of the first node that reads the tensor)
            inserts.setdefault(id(users[0]), []).extend(pairs)
    if not drop:
        return m
    final: List[onnx.NodeProto] = []
    for n in nodes:
        if id(n) in drop:
            continue
        # the new pairs go in front of their consumer; the source tensor is
        # produced earlier (the dropped Q sat between producer and consumer)
        final.extend(inserts.get(id(n), []))
        final.append(n)
    del g.node[:]
    g.node.extend(final)
    return m


def dedicate_dq_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Quark's ``DedicateDQNode`` post-processing: a DequantizeLinear read by
    several nodes (a graph output counts as a reader) is copied so that each
    reader has its own (``<name>_1``, ``<name>_2``, ... -- the first reader
    keeps the original); the Q of a constant (a shared weight) is copied too.
    Only block / half-type pairs matter here, but any Q/DQ pair is handled."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    nodes = list(g.node)
    inits = {t.name for t in g.initializer}
    outputs = {o.name for o in g.output}
    producer = {o: n for n in nodes for o in n.output}
    readers: Dict[str, List[onnx.NodeProto]] = {}
    for n in nodes:
        for x in n.input:
            readers.setdefault(x, []).append(n)
    after: Dict[int, List[onnx.NodeProto]] = {}  # id(node) -> copies to put behind it
    for dq in nodes:
        if dq.op_type not in _DQ_OPS or not dq.output:
            continue
        children = readers.get(dq.output[0])
        if not children:
            continue
        users: List[Optional[onnx.NodeProto]] = (
            [None] if dq.output[0] in outputs else []
        )
        users += children
        if len(users) < 2:
            continue
        parent = producer.get(dq.input[0])
        if parent is None or parent.op_type not in _Q_OPS:
            continue
        copy_q = parent.input[0] in inits
        for index, user in enumerate(users):
            if index == 0:
                continue
            post = f"_{index}"
            new_dq = onnx.NodeProto()
            new_dq.CopyFrom(dq)
            new_dq.name = dq.name + post
            new_dq.output[0] = dq.output[0] + post
            if copy_q:
                new_q = onnx.NodeProto()
                new_q.CopyFrom(parent)
                new_q.name = parent.name + post
                new_q.output[0] = parent.output[0] + post
                new_dq.input[0] = new_q.output[0]
                after.setdefault(id(parent), []).append(new_q)
            after.setdefault(id(dq), []).append(new_dq)
            if user is not None:
                for i, x in enumerate(user.input):
                    if x == dq.output[0]:
                        user.input[i] = new_dq.output[0]
    if not after:
        return m
    final: List[onnx.NodeProto] = []
    for n in nodes:
        final.append(n)
        final.extend(after.get(id(n), []))
    del g.node[:]
    g.node.extend(final)
    return m


__all__ = [
    "PROMOTABLE_OPS",
    "dedicate_dq_nodes",
    "dedicate_qdq_pairs",
    "requantize_biases_int8",
    "apply_block_activations_int8_constants",
    "apply_mixed_block_format",
]
