"""Transformer fixtures for the operator re-fusion survey.

The models here are built with ``onnx.helper`` only (no torch / transformers
dependency) but their graph topology is the one real exporters emit, so they
exercise the same pattern matchers that ONNX Runtime's fusion passes use:

``bert_layer``
    Post-LayerNorm BERT encoder layers, attention spelled out as
    ``MatMul+Add`` projections -> ``Reshape`` -> ``Transpose`` -> ``MatMul`` ->
    ``Div`` -> ``Add`` (mask) -> ``Softmax`` -> ``MatMul`` -> ``Transpose`` ->
    ``Reshape`` -> ``MatMul+Add``, an erf ``Gelu`` written out, and
    ``LayerNormalization`` nodes.  This is what ORT's ``AttentionFusion`` /
    ``SkipLayerNormalization`` / ``BiasGelu`` passes expect.

``bert_layer_decomposed_ln``
    Same, with LayerNormalization itself decomposed
    (``ReduceMean``/``Sub``/``Pow``/``ReduceMean``/``Add``/``Sqrt``/``Div``),
    the shape opset<17 exporters produce.  Tests whether onnxsim keeps that
    subgraph in a form ORT's ``LayerNormFusion`` still recognises -- LayerNorm
    fusion is the gateway to attention fusion, which starts its pattern walk at
    a LayerNormalization node.

``llama_layer``
    LLaMA-style decoder layer: decomposed RMSNorm, rotary embeddings applied to
    Q and K, additive causal mask, SwiGLU MLP, no biases.

``fused_onnx``
    ai.onnx opset 23 ``Attention`` + ``RMSNormalization`` + ``RotaryEmbedding``
    (the standardised fused ops).

``fused_contrib``
    ``com.microsoft`` ``MultiHeadAttention`` + ``SkipLayerNormalization`` +
    ``FastGelu`` -- what ORT's own optimizer emits, i.e. what a model looks like
    when a user hands an *already fused* model to onnxsim.

Every builder returns ``(ModelProto, {input_name: numpy array})``.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BATCH = 1
SEQ = 128
HIDDEN = 256
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
FFN = 4 * HIDDEN
LAYERS = 2

_RNG = np.random.default_rng(0)

# Named sizes.  The default is deliberately tiny so the CPU survey runs in
# seconds; ``bert-base`` is the size worth timing on a GPU.
PRESETS = {
    "tiny": dict(batch=1, seq=128, hidden=256, heads=4, layers=2),
    "bert-base": dict(batch=1, seq=384, hidden=768, heads=12, layers=12),
    "bert-large": dict(batch=1, seq=384, hidden=1024, heads=16, layers=24),
}


def configure(batch=None, seq=None, hidden=None, heads=None, layers=None, preset=None):
    """Resize the fixtures (globals are read at build time, not import time)."""
    global BATCH, SEQ, HIDDEN, HEADS, HEAD_DIM, FFN, LAYERS
    if preset:
        values = dict(PRESETS[preset])
        values.update(
            {
                k: v
                for k, v in dict(
                    batch=batch, seq=seq, hidden=hidden, heads=heads, layers=layers
                ).items()
                if v is not None
            }
        )
    else:
        values = dict(
            batch=batch if batch is not None else BATCH,
            seq=seq if seq is not None else SEQ,
            hidden=hidden if hidden is not None else HIDDEN,
            heads=heads if heads is not None else HEADS,
            layers=layers if layers is not None else LAYERS,
        )
    BATCH, SEQ, HIDDEN, HEADS, LAYERS = (
        values["batch"],
        values["seq"],
        values["hidden"],
        values["heads"],
        values["layers"],
    )
    HEAD_DIM = HIDDEN // HEADS
    FFN = 4 * HIDDEN
    for name in NUM_HEADS:
        NUM_HEADS[name] = HEADS
        HIDDEN_SIZE[name] = HIDDEN
    return values


def describe() -> str:
    return (
        f"batch={BATCH} seq={SEQ} hidden={HIDDEN} heads={HEADS} "
        f"head_dim={HEAD_DIM} ffn={FFN} layers={LAYERS}"
    )


def _f32(array, name):
    # Keep rank-0 scalars rank-0 (np.ascontiguousarray would promote them to
    # rank 1): exporters emit scalar epsilons/scales, and ORT's fusion passes
    # read them with ``float()``, which rejects a shape-(1,) array on numpy 2.
    array = np.asarray(array, dtype=np.float32)
    if array.ndim:
        array = np.ascontiguousarray(array)
    return numpy_helper.from_array(array, name)


def _i64(array, name):
    return numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def _vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _weight(shape, name, inits, scale=0.02):
    inits.append(_f32(_RNG.standard_normal(shape) * scale, name))
    return name


def _const(value, name, inits):
    inits.append(_f32(np.asarray(value, dtype=np.float32), name))
    return name


def _model(nodes, inputs, outputs, inits, opset=17, extra_opsets=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, inits)
    opset_imports = [helper.make_opsetid("", opset)]
    opset_imports += [helper.make_opsetid(d, v) for d, v in extra_opsets]
    model = helper.make_model(graph, opset_imports=opset_imports, ir_version=10)
    return model


def _linear(nodes, inits, x, in_dim, out_dim, prefix, bias=True):
    """MatMul (+ Add bias) -- how exporters spell ``nn.Linear``."""
    w = _weight((in_dim, out_dim), f"{prefix}.weight", inits)
    out = f"{prefix}.matmul"
    nodes.append(helper.make_node("MatMul", [x, w], [out], name=f"{prefix}/MatMul"))
    if not bias:
        return out
    b = _weight((out_dim,), f"{prefix}.bias", inits, scale=0.01)
    biased = f"{prefix}.out"
    nodes.append(helper.make_node("Add", [out, b], [biased], name=f"{prefix}/Add"))
    return biased


def _layer_norm(nodes, inits, x, prefix, decomposed=False, eps=1e-5):
    # Trained norm weights are near 1 but never exactly 1: an all-ones gamma
    # would make ``Mul(x, gamma)`` a no-op that onnxsim legitimately deletes,
    # which is a property of the fixture rather than of real models.
    w = f"{prefix}.gamma"
    inits.append(_f32(1.0 + 0.02 * _RNG.standard_normal(HIDDEN), w))
    b = _weight((HIDDEN,), f"{prefix}.beta", inits, scale=0.01)
    out = f"{prefix}.out"
    if not decomposed:
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [x, w, b],
                [out],
                axis=-1,
                epsilon=eps,
                name=f"{prefix}/LN",
            )
        )
        return out
    # The pre-opset-17 spelling every exporter used to emit.
    two = _const(2.0, f"{prefix}.two", inits)
    epsv = _const(eps, f"{prefix}.eps", inits)
    n = lambda op, i, o, **kw: nodes.append(  # noqa: E731
        helper.make_node(op, i, o, name=f"{prefix}/{op}_{o[0]}", **kw)
    )
    # opset 17: ReduceMean takes `axes` as an attribute (it became an input in 18).
    n("ReduceMean", [x], [f"{prefix}.mean"], axes=[-1], keepdims=1)
    n("Sub", [x, f"{prefix}.mean"], [f"{prefix}.d"])
    n("Pow", [f"{prefix}.d", two], [f"{prefix}.d2"])
    n("ReduceMean", [f"{prefix}.d2"], [f"{prefix}.var"], axes=[-1], keepdims=1)
    n("Add", [f"{prefix}.var", epsv], [f"{prefix}.vare"])
    n("Sqrt", [f"{prefix}.vare"], [f"{prefix}.std"])
    n("Div", [f"{prefix}.d", f"{prefix}.std"], [f"{prefix}.norm"])
    n("Mul", [f"{prefix}.norm", w], [f"{prefix}.scaled"])
    n("Add", [f"{prefix}.scaled", b], [out])
    return out


def _gelu_erf(nodes, inits, x, prefix):
    """The erf spelling of GELU (ORT fuses it into BiasGelu/FastGelu)."""
    half = _const(0.5, f"{prefix}.half", inits)
    one = _const(1.0, f"{prefix}.one", inits)
    sqrt2 = _const(1.4142135381698608, f"{prefix}.sqrt2", inits)
    nodes += [
        helper.make_node("Div", [x, sqrt2], [f"{prefix}.div"], name=f"{prefix}/Div"),
        helper.make_node(
            "Erf", [f"{prefix}.div"], [f"{prefix}.erf"], name=f"{prefix}/Erf"
        ),
        helper.make_node(
            "Add", [f"{prefix}.erf", one], [f"{prefix}.a"], name=f"{prefix}/Add"
        ),
        helper.make_node(
            "Mul", [x, f"{prefix}.a"], [f"{prefix}.m"], name=f"{prefix}/Mul"
        ),
        helper.make_node(
            "Mul", [f"{prefix}.m", half], [f"{prefix}.out"], name=f"{prefix}/Mul2"
        ),
    ]
    return f"{prefix}.out"


def _mha_decomposed(nodes, inits, x, mask, prefix, heads=None, kv_heads=None):
    """Multi-head attention written out node by node, exporter style."""
    heads = heads or HEADS
    kv_heads = kv_heads or heads
    head_dim = HIDDEN // heads
    q = _linear(nodes, inits, x, HIDDEN, heads * head_dim, f"{prefix}.q_proj")
    k = _linear(nodes, inits, x, HIDDEN, kv_heads * head_dim, f"{prefix}.k_proj")
    v = _linear(nodes, inits, x, HIDDEN, kv_heads * head_dim, f"{prefix}.v_proj")

    inits.append(_i64([0, 0, heads, head_dim], f"{prefix}.qshape"))
    inits.append(_i64([0, 0, kv_heads, head_dim], f"{prefix}.kvshape"))
    inits.append(_i64([0, 0, heads * head_dim], f"{prefix}.oshape"))

    def split_heads(t, name, shape_init, perm):
        nodes.append(
            helper.make_node(
                "Reshape", [t, shape_init], [f"{name}.r"], name=f"{name}/Reshape"
            )
        )
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{name}.r"],
                [f"{name}.t"],
                perm=perm,
                name=f"{name}/Transpose",
            )
        )
        return f"{name}.t"

    qh = split_heads(q, f"{prefix}.q", f"{prefix}.qshape", [0, 2, 1, 3])
    kh = split_heads(k, f"{prefix}.k", f"{prefix}.kvshape", [0, 2, 3, 1])
    vh = split_heads(v, f"{prefix}.v", f"{prefix}.kvshape", [0, 2, 1, 3])

    scale = _const(float(np.sqrt(head_dim)), f"{prefix}.scale", inits)
    nodes += [
        helper.make_node("MatMul", [qh, kh], [f"{prefix}.qk"], name=f"{prefix}/QK"),
        helper.make_node(
            "Div", [f"{prefix}.qk", scale], [f"{prefix}.qks"], name=f"{prefix}/Scale"
        ),
        helper.make_node(
            "Add", [f"{prefix}.qks", mask], [f"{prefix}.qkm"], name=f"{prefix}/Mask"
        ),
        helper.make_node(
            "Softmax",
            [f"{prefix}.qkm"],
            [f"{prefix}.p"],
            axis=-1,
            name=f"{prefix}/Softmax",
        ),
        helper.make_node(
            "MatMul", [f"{prefix}.p", vh], [f"{prefix}.ctx"], name=f"{prefix}/PV"
        ),
        helper.make_node(
            "Transpose",
            [f"{prefix}.ctx"],
            [f"{prefix}.ctxt"],
            perm=[0, 2, 1, 3],
            name=f"{prefix}/CtxTranspose",
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}.ctxt", f"{prefix}.oshape"],
            [f"{prefix}.merged"],
            name=f"{prefix}/CtxReshape",
        ),
    ]
    return _linear(
        nodes, inits, f"{prefix}.merged", heads * head_dim, HIDDEN, f"{prefix}.o_proj"
    )


def _hf_extended_mask(nodes, inits, seq):
    """HuggingFace's ``get_extended_attention_mask`` as exporters emit it.

    ``mask[B, S]`` (int64) -> Unsqueeze -> Unsqueeze -> Cast -> ``1 - m`` ->
    ``* -10000`` -> additive mask ``[B, 1, 1, S]``.

    ORT's ``FusionAttention`` only fuses attention when it can walk this exact
    chain back to a 2-D graph input -- it needs the raw 2-D mask to build the
    ``mask_index`` input of ``com.microsoft.Attention``.  Any rewrite of these
    five nodes costs the model its Attention fusion, which is why the fixture
    spells them out instead of feeding a ready-made additive mask.
    """
    inits.append(_i64([1], "mask.axes1"))
    inits.append(_i64([2], "mask.axes2"))
    _const(1.0, "mask.one", inits)
    _const(-10000.0, "mask.neg", inits)
    nodes += [
        helper.make_node(
            "Unsqueeze", ["mask", "mask.axes1"], ["mask.u1"], name="mask/Unsqueeze1"
        ),
        helper.make_node(
            "Unsqueeze", ["mask.u1", "mask.axes2"], ["mask.u2"], name="mask/Unsqueeze2"
        ),
        helper.make_node(
            "Cast", ["mask.u2"], ["mask.cast"], to=TensorProto.FLOAT, name="mask/Cast"
        ),
        helper.make_node(
            "Sub", ["mask.one", "mask.cast"], ["mask.sub"], name="mask/Sub"
        ),
        helper.make_node(
            "Mul", ["mask.sub", "mask.neg"], ["mask.add"], name="mask/Mul"
        ),
    ]
    return "mask.add"


def bert_layer(layers=None, decomposed_ln=False, seq=None):
    layers, seq = layers or LAYERS, seq or SEQ
    """Post-LayerNorm BERT encoder layers (the classic fusable shape)."""
    nodes, inits = [], []
    x = "hidden"
    mask = _hf_extended_mask(nodes, inits, seq)
    for i in range(layers):
        p = f"layer{i}"
        attn = _mha_decomposed(nodes, inits, x, mask, f"{p}.attn")
        nodes.append(
            helper.make_node("Add", [attn, x], [f"{p}.res0"], name=f"{p}/Residual0")
        )
        ln0 = _layer_norm(nodes, inits, f"{p}.res0", f"{p}.ln0", decomposed_ln)
        ff = _linear(nodes, inits, ln0, HIDDEN, FFN, f"{p}.ffn.up")
        act = _gelu_erf(nodes, inits, ff, f"{p}.ffn.act")
        down = _linear(nodes, inits, act, FFN, HIDDEN, f"{p}.ffn.down")
        nodes.append(
            helper.make_node("Add", [down, ln0], [f"{p}.res1"], name=f"{p}/Residual1")
        )
        x = _layer_norm(nodes, inits, f"{p}.res1", f"{p}.ln1", decomposed_ln)

    nodes.append(helper.make_node("Identity", [x], ["output"], name="out"))
    model = _model(
        nodes,
        [
            _vi("hidden", [BATCH, seq, HIDDEN]),
            _vi("mask", [BATCH, seq], TensorProto.INT64),
        ],
        [_vi("output", [BATCH, seq, HIDDEN])],
        inits,
    )
    mask_values = np.ones((BATCH, seq), dtype=np.int64)
    mask_values[:, seq // 2 :] = 0  # half the sequence padded, as in a real batch
    feed = {
        "hidden": _RNG.standard_normal((BATCH, seq, HIDDEN)).astype(np.float32),
        "mask": mask_values,
    }
    return model, feed


def bert_layer_decomposed_ln():
    return bert_layer(decomposed_ln=True)


def _rope_caches(inits, seq, head_dim):
    """The shared ``cos_cache``/``sin_cache`` initializers, ``[S, D]``."""
    if any(init.name == "rope.cos_cache" for init in inits):
        return "rope.cos_cache", "rope.sin_cache"
    pos = np.arange(seq)[:, None]
    inv = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
    emb = np.concatenate([pos * inv[None, :]] * 2, axis=-1)
    inits.append(_f32(np.cos(emb), "rope.cos_cache"))
    inits.append(_f32(np.sin(emb), "rope.sin_cache"))
    return "rope.cos_cache", "rope.sin_cache"


def _rope(nodes, inits, x, prefix, seq=None, head_dim=None):
    """Rotary embedding on a ``[B, H, S, D]`` tensor, HuggingFace export shape.

    ``cos = cos_cache[position_ids].unsqueeze(1)``,
    ``x_embed = x * cos + rotate_half(x) * sin`` with ``rotate_half`` spelled as
    ``Slice``/``Neg``/``Concat`` -- the node sequence ORT's
    ``FusionRotaryEmbeddings`` looks for.
    """
    cos_cache, sin_cache = _rope_caches(inits, seq, head_dim)
    half = head_dim // 2
    if not any(init.name == "rope.zero" for init in inits):
        inits.append(_i64([0], "rope.zero"))
        inits.append(_i64([half], "rope.half"))
        inits.append(_i64([head_dim], "rope.dim"))
        inits.append(_i64([-1], "rope.last_axis"))
        inits.append(_i64([1], "rope.unsq_axis"))
    for name, cache in (("cos", cos_cache), ("sin", sin_cache)):
        if any(n.output and n.output[0] == f"rope.{name}" for n in nodes):
            continue
        nodes += [
            helper.make_node(
                "Gather",
                [cache, "position_ids"],
                [f"rope.{name}.g"],
                axis=0,
                name=f"rope/{name}Gather",
            ),
            helper.make_node(
                "Unsqueeze",
                [f"rope.{name}.g", "rope.unsq_axis"],
                [f"rope.{name}"],
                name=f"rope/{name}Unsqueeze",
            ),
        ]
    nodes += [
        helper.make_node(
            "Slice",
            [x, "rope.zero", "rope.half", "rope.last_axis"],
            [f"{prefix}.x1"],
            name=f"{prefix}/Slice1",
        ),
        helper.make_node(
            "Slice",
            [x, "rope.half", "rope.dim", "rope.last_axis"],
            [f"{prefix}.x2"],
            name=f"{prefix}/Slice2",
        ),
        helper.make_node(
            "Neg", [f"{prefix}.x2"], [f"{prefix}.nx2"], name=f"{prefix}/Neg"
        ),
        helper.make_node(
            "Concat",
            [f"{prefix}.nx2", f"{prefix}.x1"],
            [f"{prefix}.rot"],
            axis=-1,
            name=f"{prefix}/Concat",
        ),
        helper.make_node(
            "Mul", [x, "rope.cos"], [f"{prefix}.mc"], name=f"{prefix}/MulCos"
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}.rot", "rope.sin"],
            [f"{prefix}.ms"],
            name=f"{prefix}/MulSin",
        ),
        helper.make_node(
            "Add",
            [f"{prefix}.mc", f"{prefix}.ms"],
            [f"{prefix}.out"],
            name=f"{prefix}/AddRope",
        ),
    ]
    return f"{prefix}.out"


def _rms_norm(nodes, inits, x, prefix, eps=1e-6):
    w = f"{prefix}.weight"
    inits.append(_f32(1.0 + 0.02 * _RNG.standard_normal(HIDDEN), w))
    two = _const(2.0, f"{prefix}.two", inits)
    epsv = _const(eps, f"{prefix}.eps", inits)
    n = lambda op, i, o: nodes.append(  # noqa: E731
        helper.make_node(op, i, o, name=f"{prefix}/{op}_{o[0]}")
    )
    n("Pow", [x, two], [f"{prefix}.sq"])
    nodes.append(
        helper.make_node(
            "ReduceMean",
            [f"{prefix}.sq"],
            [f"{prefix}.var"],
            axes=[-1],
            keepdims=1,
            name=f"{prefix}/RM",
        )
    )
    n("Add", [f"{prefix}.var", epsv], [f"{prefix}.vare"])
    nodes.append(
        helper.make_node(
            "Sqrt", [f"{prefix}.vare"], [f"{prefix}.std"], name=f"{prefix}/Sqrt"
        )
    )
    n("Div", [x, f"{prefix}.std"], [f"{prefix}.norm"])
    n("Mul", [f"{prefix}.norm", w], [f"{prefix}.out"])
    return f"{prefix}.out"


def llama_layer(layers=None, seq=None, kv_heads=None):
    layers, seq = layers or LAYERS, seq or SEQ
    kv_heads = kv_heads or HEADS
    """LLaMA-style decoder layers: RMSNorm + RoPE attention + SwiGLU."""
    nodes, inits = [], []
    x = "hidden"
    head_dim = HEAD_DIM
    for i in range(layers):
        p = f"layer{i}"
        norm = _rms_norm(nodes, inits, x, f"{p}.input_norm")
        q = _linear(nodes, inits, norm, HIDDEN, HIDDEN, f"{p}.attn.q_proj", bias=False)
        k = _linear(
            nodes,
            inits,
            norm,
            HIDDEN,
            kv_heads * head_dim,
            f"{p}.attn.k_proj",
            bias=False,
        )
        v = _linear(
            nodes,
            inits,
            norm,
            HIDDEN,
            kv_heads * head_dim,
            f"{p}.attn.v_proj",
            bias=False,
        )
        inits.append(_i64([0, 0, HEADS, head_dim], f"{p}.qshape"))
        inits.append(_i64([0, 0, kv_heads, head_dim], f"{p}.kvshape"))
        inits.append(_i64([0, 0, HIDDEN], f"{p}.oshape"))

        def heads(t, name, shape_init, perm=(0, 2, 1, 3)):
            nodes.append(
                helper.make_node(
                    "Reshape", [t, shape_init], [f"{name}.r"], name=f"{name}/R"
                )
            )
            nodes.append(
                helper.make_node(
                    "Transpose",
                    [f"{name}.r"],
                    [f"{name}.t"],
                    perm=list(perm),
                    name=f"{name}/T",
                )
            )
            return f"{name}.t"

        qh = heads(q, f"{p}.q", f"{p}.qshape")
        kh = heads(k, f"{p}.k", f"{p}.kvshape")
        vh = heads(v, f"{p}.v", f"{p}.kvshape")
        qr = _rope(nodes, inits, qh, f"{p}.q_rope", seq, head_dim)
        kr = _rope(nodes, inits, kh, f"{p}.k_rope", seq, head_dim)
        if kv_heads != HEADS:
            # HuggingFace's ``repeat_kv``: Unsqueeze -> Expand -> Reshape.
            rep = HEADS // kv_heads
            inits.append(_i64([2], f"{p}.rep_axis"))
            inits.append(_i64([BATCH, kv_heads, rep, seq, head_dim], f"{p}.rep_shape"))
            inits.append(_i64([BATCH, HEADS, seq, head_dim], f"{p}.rep_out"))
            expanded = []
            for name, t in (("k", kr), ("v", vh)):
                nodes += [
                    helper.make_node(
                        "Unsqueeze",
                        [t, f"{p}.rep_axis"],
                        [f"{p}.{name}.u"],
                        name=f"{p}/{name}RepUnsqueeze",
                    ),
                    helper.make_node(
                        "Expand",
                        [f"{p}.{name}.u", f"{p}.rep_shape"],
                        [f"{p}.{name}.e"],
                        name=f"{p}/{name}RepExpand",
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"{p}.{name}.e", f"{p}.rep_out"],
                        [f"{p}.{name}.rep"],
                        name=f"{p}/{name}RepReshape",
                    ),
                ]
                expanded.append(f"{p}.{name}.rep")
            kr, vh = expanded
        nodes.append(
            helper.make_node(
                "Transpose", [kr], [f"{p}.kt"], perm=[0, 1, 3, 2], name=f"{p}/KT"
            )
        )
        scale = _const(float(np.sqrt(head_dim)), f"{p}.scale", inits)
        nodes += [
            helper.make_node("MatMul", [qr, f"{p}.kt"], [f"{p}.qk"], name=f"{p}/QK"),
            helper.make_node(
                "Div", [f"{p}.qk", scale], [f"{p}.qks"], name=f"{p}/Scale"
            ),
            helper.make_node(
                "Add", [f"{p}.qks", "mask"], [f"{p}.qkm"], name=f"{p}/Mask"
            ),
            helper.make_node(
                "Softmax", [f"{p}.qkm"], [f"{p}.prob"], axis=-1, name=f"{p}/Softmax"
            ),
            helper.make_node("MatMul", [f"{p}.prob", vh], [f"{p}.ctx"], name=f"{p}/PV"),
            helper.make_node(
                "Transpose",
                [f"{p}.ctx"],
                [f"{p}.ctxt"],
                perm=[0, 2, 1, 3],
                name=f"{p}/CT",
            ),
            helper.make_node(
                "Reshape", [f"{p}.ctxt", f"{p}.oshape"], [f"{p}.merged"], name=f"{p}/CR"
            ),
        ]
        o = _linear(
            nodes, inits, f"{p}.merged", HIDDEN, HIDDEN, f"{p}.attn.o_proj", bias=False
        )
        nodes.append(helper.make_node("Add", [x, o], [f"{p}.res0"], name=f"{p}/Res0"))

        pn = _rms_norm(nodes, inits, f"{p}.res0", f"{p}.post_norm")
        gate = _linear(nodes, inits, pn, HIDDEN, FFN, f"{p}.mlp.gate", bias=False)
        up = _linear(nodes, inits, pn, HIDDEN, FFN, f"{p}.mlp.up", bias=False)
        nodes += [
            helper.make_node("Sigmoid", [gate], [f"{p}.sig"], name=f"{p}/Sigmoid"),
            helper.make_node(
                "Mul", [gate, f"{p}.sig"], [f"{p}.silu"], name=f"{p}/SiLU"
            ),
            helper.make_node(
                "Mul", [f"{p}.silu", up], [f"{p}.swiglu"], name=f"{p}/SwiGLU"
            ),
        ]
        down = _linear(
            nodes, inits, f"{p}.swiglu", FFN, HIDDEN, f"{p}.mlp.down", bias=False
        )
        x = f"{p}.res1"
        nodes.append(
            helper.make_node("Add", [f"{p}.res0", down], [x], name=f"{p}/Res1")
        )

    final = _rms_norm(nodes, inits, x, "final_norm")
    nodes.append(helper.make_node("Identity", [final], ["output"], name="out"))
    model = _model(
        nodes,
        [
            _vi("hidden", [BATCH, seq, HIDDEN]),
            _vi("mask", [BATCH, 1, seq, seq]),
            _vi("position_ids", [BATCH, seq], TensorProto.INT64),
        ],
        [_vi("output", [BATCH, seq, HIDDEN])],
        inits,
    )
    causal = np.triu(np.full((seq, seq), -1e4, dtype=np.float32), k=1)
    feed = {
        "hidden": _RNG.standard_normal((BATCH, seq, HIDDEN)).astype(np.float32) * 0.1,
        "mask": causal[None, None].repeat(1, axis=0),
        "position_ids": np.arange(seq, dtype=np.int64)[None].repeat(BATCH, axis=0),
    }
    return model, feed


def llama_layer_gqa():
    """LLaMA layer with grouped-query attention (kv_heads < heads)."""
    return llama_layer(kv_heads=max(1, HEADS // 2))


def fused_onnx(seq=None):
    seq = seq or SEQ
    """ai.onnx opset 23 fused ops: Attention + RMSNormalization + RotaryEmbedding."""
    nodes, inits = [], []
    inits.append(_f32(1.0 + 0.02 * _RNG.standard_normal(HIDDEN), "rms.weight"))
    nodes.append(
        helper.make_node(
            "RMSNormalization",
            ["hidden", "rms.weight"],
            ["normed"],
            axis=-1,
            epsilon=1e-6,
        )
    )
    q = _linear(nodes, inits, "normed", HIDDEN, HIDDEN, "q_proj", bias=False)
    k = _linear(nodes, inits, "normed", HIDDEN, HIDDEN, "k_proj", bias=False)
    v = _linear(nodes, inits, "normed", HIDDEN, HIDDEN, "v_proj", bias=False)
    inits.append(_i64([0, 0, HEADS, HEAD_DIM], "qkv_shape"))
    for name, t in (("q", q), ("k", k), ("v", v)):
        nodes += [
            helper.make_node("Reshape", [t, "qkv_shape"], [f"{name}.r"]),
            helper.make_node(
                "Transpose", [f"{name}.r"], [f"{name}.h"], perm=[0, 2, 1, 3]
            ),
        ]
    pos = np.arange(seq)[:, None]
    inv = 1.0 / (10000 ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    ang = pos * inv[None, :]
    inits += [_f32(np.cos(ang), "rope.cos"), _f32(np.sin(ang), "rope.sin")]
    inits.append(_i64(np.arange(seq)[None], "position_ids"))
    for name in ("q", "k"):
        nodes.append(
            helper.make_node(
                "RotaryEmbedding",
                [f"{name}.h", "rope.cos", "rope.sin", "position_ids"],
                [f"{name}.rope"],
            )
        )
    nodes.append(
        helper.make_node(
            "Attention",
            ["q.rope", "k.rope", "v.h"],
            ["attn"],
            q_num_heads=HEADS,
            kv_num_heads=HEADS,
        )
    )
    inits.append(_i64([0, 0, HIDDEN], "merge_shape"))
    nodes += [
        helper.make_node("Transpose", ["attn"], ["attn.t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["attn.t", "merge_shape"], ["merged"]),
    ]
    out = _linear(nodes, inits, "merged", HIDDEN, HIDDEN, "o_proj", bias=False)
    nodes.append(helper.make_node("Identity", [out], ["output"]))
    for i, node in enumerate(nodes):
        if not node.name:
            node.name = f"node_{i}"
    model = _model(
        nodes,
        [_vi("hidden", [BATCH, seq, HIDDEN])],
        [_vi("output", [BATCH, seq, HIDDEN])],
        inits,
        opset=23,
    )
    feed = {
        "hidden": _RNG.standard_normal((BATCH, seq, HIDDEN)).astype(np.float32) * 0.1
    }
    return model, feed


def fused_contrib(seq=None):
    seq = seq or SEQ
    """com.microsoft fused ops: MultiHeadAttention + SkipLayerNormalization + FastGelu."""
    nodes, inits = [], []
    q = _linear(nodes, inits, "hidden", HIDDEN, HIDDEN, "q_proj")
    k = _linear(nodes, inits, "hidden", HIDDEN, HIDDEN, "k_proj")
    v = _linear(nodes, inits, "hidden", HIDDEN, HIDDEN, "v_proj")
    nodes.append(
        helper.make_node(
            "MultiHeadAttention",
            [q, k, v],
            ["attn"],
            num_heads=HEADS,
            domain="com.microsoft",
            name="mha",
        )
    )
    o = _linear(nodes, inits, "attn", HIDDEN, HIDDEN, "o_proj")
    inits.append(_f32(1.0 + 0.02 * _RNG.standard_normal(HIDDEN), "ln.gamma"))
    inits.append(_f32(0.01 * _RNG.standard_normal(HIDDEN), "ln.beta"))
    nodes.append(
        helper.make_node(
            "SkipLayerNormalization",
            [o, "hidden", "ln.gamma", "ln.beta"],
            ["ln"],
            epsilon=1e-5,
            domain="com.microsoft",
            name="skipln",
        )
    )
    ff = _linear(nodes, inits, "ln", HIDDEN, FFN, "ffn.up")
    nodes.append(
        helper.make_node(
            "FastGelu", [ff], ["act"], domain="com.microsoft", name="fastgelu"
        )
    )
    down = _linear(nodes, inits, "act", FFN, HIDDEN, "ffn.down")
    nodes.append(helper.make_node("Identity", [down], ["output"], name="out"))
    model = _model(
        nodes,
        [_vi("hidden", [BATCH, seq, HIDDEN])],
        [_vi("output", [BATCH, seq, HIDDEN])],
        inits,
        extra_opsets=[("com.microsoft", 1)],
    )
    feed = {
        "hidden": _RNG.standard_normal((BATCH, seq, HIDDEN)).astype(np.float32) * 0.1
    }
    return model, feed


FIXTURES = {
    "bert_layer": bert_layer,
    "bert_layer_decomposed_ln": bert_layer_decomposed_ln,
    "llama_layer": llama_layer,
    "llama_layer_gqa": llama_layer_gqa,
    "fused_onnx_attention": fused_onnx,
    "fused_contrib_mha": fused_contrib,
}

# ORT's transformer optimizer needs to know which fusion set to try.
MODEL_TYPE = {
    "bert_layer": "bert",
    "bert_layer_decomposed_ln": "bert",
    "llama_layer": "gpt2",
    "llama_layer_gqa": "gpt2",
    "fused_onnx_attention": "bert",
    "fused_contrib_mha": "bert",
}

NUM_HEADS = {name: HEADS for name in FIXTURES}
HIDDEN_SIZE = {name: HIDDEN for name in FIXTURES}


def build(name):
    model, feed = FIXTURES[name]()
    onnx.checker.check_model(model, full_check=False)
    return model, feed


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import collections

    for name in FIXTURES:
        model, feed = build(name)
        ops = collections.Counter(n.op_type for n in model.graph.node)
        print(f"{name}: {sum(ops.values())} nodes, {dict(ops)}")
