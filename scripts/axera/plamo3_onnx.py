"""ONNX graphs of PLaMo 3 built from a ``plamo3_reference.Plamo3Reference`` model.

``build_decode``: the whole model for one decode step with a KV cache of dynamic length.
    Inputs : input_ids int64 [1], position_ids int64 [1],
             past_k_{i}, past_v_{i} float32 [nkv, P, hd] (P dynamic)
    Outputs: logits float32 [1, vocab], present_k_{i}, present_v_{i} float32 [nkv, P+1, hd]

``build_layer``: one decoder layer with a static cache (for a fixed-shape compiler).
    Inputs : hidden float32 [1, H], position_ids int64 [1],
             past_k, past_v float32 [nkv, C, hd], attn_mask float32 [1, C+1]
             (0 for valid slots and the new token, a large negative for padding)
    Outputs: hidden_out float32 [1, H], k_new, v_new float32 [nkv, 1, hd]

Weights are float32 initializers. RMSNorm gains are folded (offset + weight). Large models are
saved with external data next to the graph.
"""

import math
import os

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

OPSET = 17
MAX_POS = 4096


class _Builder:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def tmp(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def const(self, name, arr):
        arr = np.asarray(arr)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        self.inits.append(nh.from_array(arr, name))
        return name

    def node(self, op, inputs, **attrs):
        out = self.tmp(op.lower())
        self.nodes.append(h.make_node(op, inputs, [out], **attrs))
        return out

    def node_multi(self, op, inputs, n_out, **attrs):
        outs = [self.tmp(op.lower()) for _ in range(n_out)]
        self.nodes.append(h.make_node(op, inputs, outs, **attrs))
        return outs

    def rms(self, x, gain, eps):
        """x / sqrt(mean(x^2) + eps) * gain (the gain already includes its offset)."""
        sq = self.node("Mul", [x, x])
        m = self.node("ReduceMean", [sq], axes=[-1], keepdims=1)
        r = self.node("Sqrt", [self.node("Add", [m, eps])])
        return self.node("Mul", [self.node("Div", [x, r]), gain])


def _common(b, model):
    """Constants shared by every layer: eps, attention scale and the slice bounds for RoPE."""
    hd = model.hd
    return {
        "eps": b.const("eps", np.array(1e-6, np.float32)),
        "scale": b.const("inv_sqrt_hd", np.array(1.0 / math.sqrt(hd), np.float32)),
        "s0": b.const("s_h0", np.array([0], np.int64)),
        "sh": b.const("s_hh", np.array([hd // 2], np.int64)),
        "shd": b.const("s_hd", np.array([hd], np.int64)),
        "axl": b.const("ax_last", np.array([-1], np.int64)),
        "ax1": b.const("ax1", np.array([1], np.int64)),
    }


def _rope_tables(model, full, max_pos):
    inv = model.inv_full if full else model.inv_local
    conc = model.conc if full else 1.0
    freqs = np.outer(np.arange(max_pos, dtype=np.float64), inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return (np.cos(emb) * conc).astype(np.float32), (np.sin(emb) * conc).astype(
        np.float32
    )


def _qconst(b, name, B):
    """A [K, N] float matrix as per-column symmetric int8 (an int8 initializer and an fp32 scale), dequantized in the graph:
    Cast -> Mul -> the MatMul's B. The initializer is what the program stores (a quarter of the fp32 bytes)."""
    s = np.abs(B).max(axis=0, keepdims=True) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.round(B / s), -127, 127).astype(np.int8)
    c = b.node("Cast", [b.const(name + "_q", q)], to=T.FLOAT)
    return b.node("Mul", [c, b.const(name + "_s", s.astype(np.float32))])


def _qconst4(b, name, B):
    """A [K, N] float matrix (K even) as per-column symmetric int4 packed two per byte along K: byte j of column n holds
    q[2j] + 16 * q[2j+1] (q + 8 in 0..15). The graph unpacks with Mod / Div on int32 (the importer has no bit shifts),
    dequantizes with the fp32 scale, and the MatMul sees the usual fp32 B. The initializer is half the int8 bytes."""
    K, N = B.shape
    assert K % 2 == 0
    s = np.abs(B).max(axis=0, keepdims=True) / 7.0
    s[s == 0] = 1.0
    q = (np.clip(np.round(B / s), -8, 7) + 8).astype(np.uint8)
    packed = (q[0::2] + 16 * q[1::2]).astype(np.uint8)  # [K/2, N]
    x = b.node("Cast", [b.const(name + "_p", packed)], to=T.INT32)
    lo = b.node("Mod", [x, b.const(name + "_c16", np.array(16, np.int32))])
    hi = b.node("Div", [x, b.const(name + "_c16b", np.array(16, np.int32))])
    ax = b.const(name + "_ax1", np.array([1], np.int64))
    stacked = b.node(
        "Concat", [b.node("Unsqueeze", [lo, ax]), b.node("Unsqueeze", [hi, ax])], axis=1
    )  # [K/2, 2, N]
    w = b.node(
        "Reshape", [stacked, b.const(name + "_shape", np.array([K, N], np.int64))]
    )
    f = b.node(
        "Sub",
        [
            b.node("Cast", [w], to=T.FLOAT),
            b.const(name + "_c8", np.array(8.0, np.float32)),
        ],
    )
    return b.node("Mul", [f, b.const(name + "_s", s.astype(np.float32))])


def _layer_weights(b, model, i, int8=False, int4=False, split=None):
    """split: {"qkv"|"o"|"gu"|"d": n} column slices per matrix; each slice is quantized and multiplied on its own, so the HMX
    fp16 copy of a weight is 1/n of the matrix (the per-program scratch that bounds how many programs stay resident).
    Do not split "o": on the phone's HMX path the split O projection (its input is a transposed attention view used by
    two matmuls) gave a 24% error at layer 0, while the qkv, gu and d splits matched the unsplit program exactly."""
    p = f"model.layers.layers.{i}."
    base = (
        (lambda name, B: _qconst4(b, name, B))
        if int4
        else (lambda name, B: _qconst(b, name, B))
        if int8
        else (lambda name, B: b.const(name, B))
    )
    split = split or {}

    def mat(name, B):
        n = split.get(name.rsplit("_", 1)[-1], 1)
        if n == 1:
            return base(name, B)
        return [
            base(f"{name}_c{j}", part)
            for j, part in enumerate(np.array_split(B, n, axis=1))
        ]

    return {
        "g_pre_mix": b.const(
            f"l{i}_g_pre_mix", 1.0 + model.w(p + "pre_mixer_norm.weight")
        ),
        "g_post_mix": b.const(
            f"l{i}_g_post_mix", 0.2 + model.w(p + "post_mixer_norm.weight")
        ),
        "g_pre_mlp": b.const(
            f"l{i}_g_pre_mlp", 1.0 + model.w(p + "pre_mlp_norm.weight")
        ),
        "g_post_mlp": b.const(
            f"l{i}_g_post_mlp", 0.2**1.5 + model.w(p + "post_mlp_norm.weight")
        ),
        "g_q": b.const(f"l{i}_g_q", 1.0 + model.w(p + "mixer.q_norm.weight")),
        "g_k": b.const(f"l{i}_g_k", 1.0 + model.w(p + "mixer.k_norm.weight")),
        "w_qkv": mat(f"l{i}_w_qkv", model.w(p + "mixer.qkv_proj.weight").T.copy()),
        "w_o": mat(f"l{i}_w_o", model.w(p + "mixer.o_proj.weight").T.copy()),
        "w_gu": mat(f"l{i}_w_gu", model.w(p + "mlp.gate_up_proj.weight").T.copy()),
        "w_d": mat(f"l{i}_w_d", model.w(p + "mlp.down_proj.weight").T.copy()),
    }


def _layer_body(
    b, model, c, i, hidden, cos_p, sin_p, past_k, past_v, mask, int8=False, int4=False
):
    """One decoder layer on a single token. Returns (hidden_out, k_new, v_new) tensor names."""
    inter, nh, nkv, hd = model.I, model.nh, model.nkv, model.hd
    g = nh // nkv
    q_dim, k_dim = nh * hd, nkv * hd
    W = _layer_weights(b, model, i, int8, int4)
    shape_q = b.const(f"l{i}_shape_q", np.array([1, nh, hd], np.int64))
    shape_kv = b.const(f"l{i}_shape_kv", np.array([1, nkv, hd], np.int64))
    shape_qg = b.const(f"l{i}_shape_qg", np.array([nkv, g, 1, hd], np.int64))
    shape_att = b.const(f"l{i}_shape_att", np.array([1, q_dim], np.int64))
    split_qkv = b.const(f"l{i}_split_qkv", np.array([q_dim, k_dim, k_dim], np.int64))
    split_mlp = b.const(f"l{i}_split_mlp", np.array([inter, inter], np.int64))

    def rope(x):
        x1 = b.node("Slice", [x, c["s0"], c["sh"], c["axl"]])
        x2 = b.node("Slice", [x, c["sh"], c["shd"], c["axl"]])
        rot = b.node("Concat", [b.node("Neg", [x2]), x1], axis=-1)
        return b.node("Add", [b.node("Mul", [x, cos_p]), b.node("Mul", [rot, sin_p])])

    x = b.rms(hidden, W["g_pre_mix"], c["eps"])
    qkv = b.node("MatMul", [x, W["w_qkv"]])
    q, k, v = b.node_multi("Split", [qkv, split_qkv], 3, axis=-1)
    q = b.node("Transpose", [b.node("Reshape", [q, shape_q])], perm=[1, 0, 2])
    k = b.node("Transpose", [b.node("Reshape", [k, shape_kv])], perm=[1, 0, 2])
    v = b.node("Transpose", [b.node("Reshape", [v, shape_kv])], perm=[1, 0, 2])
    q = b.rms(q, W["g_q"], c["eps"])
    k = b.rms(k, W["g_k"], c["eps"])
    q = rope(q)
    k = rope(k)
    kall = b.node("Concat", [past_k, k], axis=1)
    vall = b.node("Concat", [past_v, v], axis=1)
    qg = b.node("Reshape", [q, shape_qg])
    kt = b.node("Transpose", [b.node("Unsqueeze", [kall, c["ax1"]])], perm=[0, 1, 3, 2])
    vg = b.node("Unsqueeze", [vall, c["ax1"]])
    s = b.node("Mul", [b.node("MatMul", [qg, kt]), c["scale"]])
    if mask is not None:
        s = b.node("Add", [s, mask])
    pr = b.node("Softmax", [s], axis=-1)
    att = b.node("MatMul", [pr, vg])
    att = b.node("Reshape", [b.node("Transpose", [att], perm=[2, 0, 1, 3]), shape_att])
    o = _mm(b, att, W["w_o"], "o")
    o = b.rms(o, W["g_post_mix"], c["eps"])
    h1 = b.node("Add", [hidden, o])
    x = b.rms(h1, W["g_pre_mlp"], c["eps"])
    gu = _mm(b, x, W["w_gu"], "gu")
    gate, up = b.node_multi("Split", [gu, split_mlp], 2, axis=-1)
    a = b.node("Mul", [b.node("Mul", [gate, b.node("Sigmoid", [gate])]), up])
    mo = _mm(b, a, W["w_d"], "d")
    mo = b.rms(mo, W["g_post_mlp"], c["eps"])
    return b.node("Add", [h1, mo]), k, v, kall, vall


def _save(model_proto, out_path, external):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if external:
        onnx.save_model(
            model_proto,
            out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(out_path) + ".data",
            size_threshold=1024,
        )
    else:
        onnx.save_model(model_proto, out_path)
    onnx.checker.check_model(out_path)
    return out_path


def build_decode(model, out_path, max_pos=MAX_POS, external=True):
    """The whole model, one token, KV cache of dynamic length (see the module docstring)."""
    b = _Builder()
    c = _common(b, model)
    tables = {}
    for full in (False, True):
        cos, sin = _rope_tables(model, full, max_pos)
        tables[full] = (
            b.const(f"cos_tab_{int(full)}", cos),
            b.const(f"sin_tab_{int(full)}", sin),
        )
    emb = b.const("embed_tokens", model.w("model.embed_tokens.weight"))
    hidden = b.node("Gather", [emb, "input_ids"], axis=0)
    pos_cs = {}
    for full in (False, True):
        cos_t, sin_t = tables[full]
        pos_cs[full] = (
            b.node("Gather", [cos_t, "position_ids"], axis=0),
            b.node("Gather", [sin_t, "position_ids"], axis=0),
        )
    inputs = [
        h.make_tensor_value_info("input_ids", T.INT64, [1]),
        h.make_tensor_value_info("position_ids", T.INT64, [1]),
    ]
    outputs = [h.make_tensor_value_info("logits", T.FLOAT, [1, model.c["vocab_size"]])]
    for i in range(model.L):
        full = (i + 1) % model.pattern == 0
        cos_p, sin_p = pos_cs[full]
        pk, pv = f"past_k_{i}", f"past_v_{i}"
        inputs.append(h.make_tensor_value_info(pk, T.FLOAT, [model.nkv, "P", model.hd]))
        inputs.append(h.make_tensor_value_info(pv, T.FLOAT, [model.nkv, "P", model.hd]))
        hidden, k, v, kall, vall = _layer_body(
            b, model, c, i, hidden, cos_p, sin_p, pk, pv, None
        )
        b.nodes.append(h.make_node("Identity", [kall], [f"present_k_{i}"]))
        b.nodes.append(h.make_node("Identity", [vall], [f"present_v_{i}"]))
        outputs.append(
            h.make_tensor_value_info(
                f"present_k_{i}", T.FLOAT, [model.nkv, "Pnext", model.hd]
            )
        )
        outputs.append(
            h.make_tensor_value_info(
                f"present_v_{i}", T.FLOAT, [model.nkv, "Pnext", model.hd]
            )
        )
    gn = b.const("g_norm", 1.0 + model.w("model.norm.weight"))
    hidden = b.rms(hidden, gn, c["eps"])
    w_lm = b.const("lm_head_T", model.w("lm_head.weight").T.copy())
    b.nodes.append(h.make_node("MatMul", [hidden, w_lm], ["logits"]))
    graph = h.make_graph(b.nodes, "plamo3_decode", inputs, outputs, b.inits)
    proto = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)])
    proto.ir_version = 9
    return _save(proto, out_path, external)


def build_layer(model, layer_idx, out_path, cache_len=1023, external=False):
    """One decoder layer with a static cache of ``cache_len`` slots plus the new token."""
    b = _Builder()
    c = _common(b, model)
    full = (layer_idx + 1) % model.pattern == 0
    inv = model.inv_full if full else model.inv_local
    conc = model.conc if full else 1.0
    freqs = np.outer(np.arange(MAX_POS, dtype=np.float64), inv)
    emb_ = np.concatenate([freqs, freqs], axis=-1)
    cos_t = b.const("cos_tab", (np.cos(emb_) * conc).astype(np.float32))
    sin_t = b.const("sin_tab", (np.sin(emb_) * conc).astype(np.float32))
    cos_p = b.node("Gather", [cos_t, "position_ids"], axis=0)
    sin_p = b.node("Gather", [sin_t, "position_ids"], axis=0)
    hidden, k, v, _, _ = _layer_body(
        b, model, c, layer_idx, "hidden", cos_p, sin_p, "past_k", "past_v", "attn_mask"
    )
    b.nodes.append(h.make_node("Identity", [hidden], ["hidden_out"]))
    b.nodes.append(h.make_node("Identity", [k], ["k_new"]))
    b.nodes.append(h.make_node("Identity", [v], ["v_new"]))
    nkv, hd, H = model.nkv, model.hd, model.H
    inputs = [
        h.make_tensor_value_info("hidden", T.FLOAT, [1, H]),
        h.make_tensor_value_info("position_ids", T.INT64, [1]),
        h.make_tensor_value_info("past_k", T.FLOAT, [nkv, cache_len, hd]),
        h.make_tensor_value_info("past_v", T.FLOAT, [nkv, cache_len, hd]),
        h.make_tensor_value_info("attn_mask", T.FLOAT, [1, cache_len + 1]),
    ]
    outputs = [
        h.make_tensor_value_info("hidden_out", T.FLOAT, [1, H]),
        h.make_tensor_value_info("k_new", T.FLOAT, [nkv, 1, hd]),
        h.make_tensor_value_info("v_new", T.FLOAT, [nkv, 1, hd]),
    ]
    graph = h.make_graph(b.nodes, f"plamo3_layer{layer_idx}", inputs, outputs, b.inits)
    proto = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)])
    proto.ir_version = 9
    return _save(proto, out_path, external)


def build_chunk(
    model, layers, out_path, cache_len=15, external=False, int8=False, int4=False
):
    """Consecutive decoder layers ``layers`` (a range) with static caches, one token.

    Used to split the model into phone-sized programs: the host runs the embedding and feeds each
    chunk's hidden state into the next one, keeping every layer's cache on the host.
    Inputs : hidden float32 [1, H], position_ids int64 [1], attn_mask float32 [1, C+1],
             past_k_{i}, past_v_{i} float32 [nkv, C, hd] for each layer i in the chunk
    Outputs: hidden_out float32 [1, H], k_new_{i}, v_new_{i} float32 [nkv, 1, hd]
    """
    layers = list(layers)
    b = _Builder()
    c = _common(b, model)
    tabs = {}
    for full in (False, True):
        cos, sin = _rope_tables(model, full, MAX_POS)
        tabs[full] = (
            b.const(f"cos_tab_{int(full)}", cos),
            b.const(f"sin_tab_{int(full)}", sin),
        )
    pos_cs = {}
    for full in (False, True):
        cos_t, sin_t = tabs[full]
        pos_cs[full] = (
            b.node("Gather", [cos_t, "position_ids"], axis=0),
            b.node("Gather", [sin_t, "position_ids"], axis=0),
        )
    nkv, hd, H = model.nkv, model.hd, model.H
    inputs = [
        h.make_tensor_value_info("hidden", T.FLOAT, [1, H]),
        h.make_tensor_value_info("position_ids", T.INT64, [1]),
        h.make_tensor_value_info("attn_mask", T.FLOAT, [1, cache_len + 1]),
    ]
    outputs = []
    hidden = "hidden"
    for i in layers:
        full = (i + 1) % model.pattern == 0
        cos_p, sin_p = pos_cs[full]
        pk, pv = f"past_k_{i}", f"past_v_{i}"
        inputs.append(h.make_tensor_value_info(pk, T.FLOAT, [nkv, cache_len, hd]))
        inputs.append(h.make_tensor_value_info(pv, T.FLOAT, [nkv, cache_len, hd]))
        hidden, k, v, _, _ = _layer_body(
            b, model, c, i, hidden, cos_p, sin_p, pk, pv, "attn_mask", int8, int4
        )
        b.nodes.append(h.make_node("Identity", [k], [f"k_new_{i}"]))
        b.nodes.append(h.make_node("Identity", [v], [f"v_new_{i}"]))
        outputs.append(h.make_tensor_value_info(f"k_new_{i}", T.FLOAT, [nkv, 1, hd]))
        outputs.append(h.make_tensor_value_info(f"v_new_{i}", T.FLOAT, [nkv, 1, hd]))
    b.nodes.append(h.make_node("Identity", [hidden], ["hidden_out"]))
    outputs.insert(0, h.make_tensor_value_info("hidden_out", T.FLOAT, [1, H]))
    graph = h.make_graph(
        b.nodes, f"plamo3_chunk_{layers[0]}_{layers[-1]}", inputs, outputs, b.inits
    )
    proto = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)])
    proto.ir_version = 9
    return _save(proto, out_path, external)


def build_head(model, out_path, external=False):
    """Final RMSNorm and the LM head: hidden float32 [1, H] -> logits float32 [1, vocab]."""
    b = _Builder()
    c = _common(b, model)
    gn = b.const("g_norm", 1.0 + model.w("model.norm.weight"))
    x = b.rms("hidden", gn, c["eps"])
    w_lm = b.const("lm_head_T", model.w("lm_head.weight").T.copy())
    b.nodes.append(h.make_node("MatMul", [x, w_lm], ["logits"]))
    H, V = model.H, model.c["vocab_size"]
    inputs = [h.make_tensor_value_info("hidden", T.FLOAT, [1, H])]
    outputs = [h.make_tensor_value_info("logits", T.FLOAT, [1, V])]
    graph = h.make_graph(b.nodes, "plamo3_head", inputs, outputs, b.inits)
    proto = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)])
    proto.ir_version = 9
    return _save(proto, out_path, external)


def _mm(b, x, entry, name):
    """x @ W for a weight entry: one node, or a list of per-column slices whose products are concatenated."""
    if isinstance(entry, str):
        return b.node("MatMul", [x, entry])
    return b.node("Concat", [b.node("MatMul", [x, e]) for e in entry], axis=-1)


def _layer_body_rows(
    b,
    model,
    c,
    i,
    hidden,
    cos_p,
    sin_p,
    past_k,
    past_v,
    mask,
    rows,
    int8=False,
    int4=False,
    split=None,
):
    """One decoder layer over ``rows`` tokens at once (a block of prompt positions, or one token padded to the block): the
    keys and values of the block are appended after the cache, and the mask ([rows, C + rows]) carries causality and the
    padding. Returns (hidden_out, k_new, v_new) as [rows, H], [nkv, rows, hd], [nkv, rows, hd]."""
    inter, nh, nkv, hd = model.I, model.nh, model.nkv, model.hd
    g = nh // nkv
    q_dim, k_dim = nh * hd, nkv * hd
    W = _layer_weights(b, model, i, int8, int4, split=split)
    shape_q = b.const(f"l{i}_rq", np.array([rows, nh, hd], np.int64))
    shape_kv = b.const(f"l{i}_rkv", np.array([rows, nkv, hd], np.int64))
    shape_qg = b.const(f"l{i}_rqg", np.array([nkv, g, rows, hd], np.int64))
    shape_att = b.const(f"l{i}_ratt", np.array([rows, q_dim], np.int64))
    split_qkv = b.const(f"l{i}_rsplit", np.array([q_dim, k_dim, k_dim], np.int64))
    split_mlp = b.const(f"l{i}_rsplit_mlp", np.array([inter, inter], np.int64))

    def rope(x):
        x1 = b.node("Slice", [x, c["s0"], c["sh"], c["axl"]])
        x2 = b.node("Slice", [x, c["sh"], c["shd"], c["axl"]])
        rot = b.node("Concat", [b.node("Neg", [x2]), x1], axis=-1)
        return b.node("Add", [b.node("Mul", [x, cos_p]), b.node("Mul", [rot, sin_p])])

    x = b.rms(hidden, W["g_pre_mix"], c["eps"])
    qkv = _mm(b, x, W["w_qkv"], "qkv")
    q, k, v = b.node_multi("Split", [qkv, split_qkv], 3, axis=-1)
    q = b.node(
        "Transpose", [b.node("Reshape", [q, shape_q])], perm=[1, 0, 2]
    )  # [nh, rows, hd]
    k = b.node(
        "Transpose", [b.node("Reshape", [k, shape_kv])], perm=[1, 0, 2]
    )  # [nkv, rows, hd]
    v = b.node("Transpose", [b.node("Reshape", [v, shape_kv])], perm=[1, 0, 2])
    q = b.rms(q, W["g_q"], c["eps"])
    k = b.rms(k, W["g_k"], c["eps"])
    q = rope(q)
    k = rope(k)
    kall = b.node("Concat", [past_k, k], axis=1)  # [nkv, C + rows, hd]
    vall = b.node("Concat", [past_v, v], axis=1)
    qg = b.node("Reshape", [q, shape_qg])  # [nkv, g, rows, hd]
    kt = b.node(
        "Transpose", [b.node("Unsqueeze", [kall, c["ax1"]])], perm=[0, 1, 3, 2]
    )  # [nkv, 1, hd, C + rows]
    vg = b.node("Unsqueeze", [vall, c["ax1"]])  # [nkv, 1, C + rows, hd]
    s = b.node(
        "Mul", [b.node("MatMul", [qg, kt]), c["scale"]]
    )  # [nkv, g, rows, C + rows]
    s = b.node("Add", [s, mask])
    pr = b.node("Softmax", [s], axis=-1)
    att = b.node("MatMul", [pr, vg])  # [nkv, g, rows, hd]
    att = b.node("Reshape", [b.node("Transpose", [att], perm=[2, 0, 1, 3]), shape_att])
    o = _mm(b, att, W["w_o"], "o")
    o = b.rms(o, W["g_post_mix"], c["eps"])
    h1 = b.node("Add", [hidden, o])
    x = b.rms(h1, W["g_pre_mlp"], c["eps"])
    gu = _mm(b, x, W["w_gu"], "gu")
    gate, up = b.node_multi("Split", [gu, split_mlp], 2, axis=-1)
    a = b.node("Mul", [b.node("Mul", [gate, b.node("Sigmoid", [gate])]), up])
    mo = _mm(b, a, W["w_d"], "d")
    mo = b.rms(mo, W["g_post_mlp"], c["eps"])
    return b.node("Add", [h1, mo]), k, v


def build_block(
    model,
    layers,
    out_path,
    cache_len=128,
    rows=32,
    external=False,
    int8=False,
    int4=False,
    split=None,
    state=False,
):
    """Consecutive decoder layers over a block of ``rows`` token positions with static caches (the prefill graph; a decode
    step is a block with one real row). Inputs: hidden [rows, H], position_ids int64 [rows], attn_mask [rows, C + rows],
    past_k_{i}, past_v_{i} [nkv, C, hd]. Outputs: hidden_out [rows, H], k_new_{i}, v_new_{i} [nkv, rows, hd].

    state=True keeps the caches on the device (onnx-remote's recurrent state: program.txt "state"): the inputs
    past_k_{i} / past_v_{i} are the caches, the outputs next_past_k_{i} / next_past_v_{i} the caches after this block
    (the old slots, and the block's rows written to the slots given by wsel [C, rows] / wmask [C, 1]). The caller sends
    the caches in full once, then sends them empty; no cache crosses the wire after that."""
    layers = list(layers)
    b = _Builder()
    c = _common(b, model)
    tabs = {}
    for full in (False, True):
        cos, sin = _rope_tables(model, full, MAX_POS)
        tabs[full] = (
            b.const(f"cos_tab_{int(full)}", cos),
            b.const(f"sin_tab_{int(full)}", sin),
        )
    pos_cs = {}
    for full in (False, True):
        cos_t, sin_t = tabs[full]
        pos_cs[full] = (
            b.node("Gather", [cos_t, "position_ids"], axis=0),
            b.node("Gather", [sin_t, "position_ids"], axis=0),
        )
    nkv, hd, H = model.nkv, model.hd, model.H
    inputs = [
        h.make_tensor_value_info("hidden", T.FLOAT, [rows, H]),
        h.make_tensor_value_info("position_ids", T.INT64, [rows]),
        h.make_tensor_value_info("attn_mask", T.FLOAT, [rows, cache_len + rows]),
    ]
    if state:
        inputs.append(h.make_tensor_value_info("wsel", T.FLOAT, [cache_len, rows]))
        inputs.append(h.make_tensor_value_info("wmask", T.FLOAT, [cache_len, 1]))
        one = b.const("one_c", np.array(1.0, np.float32))
        keep_m = b.node(
            "Sub", [one, "wmask"]
        )  # 1 - wmask: the slots this block leaves alone
    outputs = []
    hidden = "hidden"
    for i in layers:
        full = (i + 1) % model.pattern == 0
        cos_p, sin_p = pos_cs[full]
        pk, pv = f"past_k_{i}", f"past_v_{i}"
        inputs.append(h.make_tensor_value_info(pk, T.FLOAT, [nkv, cache_len, hd]))
        inputs.append(h.make_tensor_value_info(pv, T.FLOAT, [nkv, cache_len, hd]))
        hidden, k, v = _layer_body_rows(
            b,
            model,
            c,
            i,
            hidden,
            cos_p,
            sin_p,
            pk,
            pv,
            "attn_mask",
            rows,
            int8,
            int4,
            split,
        )
        if state:
            for past, new, name in ((pk, k, f"next_{pk}"), (pv, v, f"next_{pv}")):
                kept = b.node("Mul", [past, keep_m])
                written = b.node("MatMul", ["wsel", new])  # [nkv, C, hd]
                b.nodes.append(h.make_node("Add", [kept, written], [name]))
                outputs.append(
                    h.make_tensor_value_info(name, T.FLOAT, [nkv, cache_len, hd])
                )
        else:
            b.nodes.append(h.make_node("Identity", [k], [f"k_new_{i}"]))
            b.nodes.append(h.make_node("Identity", [v], [f"v_new_{i}"]))
            outputs.append(
                h.make_tensor_value_info(f"k_new_{i}", T.FLOAT, [nkv, rows, hd])
            )
            outputs.append(
                h.make_tensor_value_info(f"v_new_{i}", T.FLOAT, [nkv, rows, hd])
            )
    b.nodes.append(h.make_node("Identity", [hidden], ["hidden_out"]))
    outputs.insert(0, h.make_tensor_value_info("hidden_out", T.FLOAT, [rows, H]))
    graph = h.make_graph(
        b.nodes, f"plamo3_block_{layers[0]}_{layers[-1]}", inputs, outputs, b.inits
    )
    proto = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)])
    proto.ir_version = 9
    return _save(proto, out_path, external)


#: the residual channel with massive activations in PLaMo 3 2B (layers 7-23); see build_layer_split
MASSIVE_CHANNEL = 380


def build_layer_split(
    model, layer_idx, out_path, cache_len=1023, external=False, massive=MASSIVE_CHANNEL
):
    """One decoder layer with the residual stream split into the ordinary channels and one
    ``massive`` channel, so that a quantized compile gives each its own range.

    Inputs : hidden_m float32 [1, H-1] (all channels but ``massive``), hidden_c float32 [1, 1]
             (the ``massive`` channel), position_ids int64 [1], past_k / past_v float32
             [nkv, C, hd], attn_mask float32 [1, C+1] (0 for valid slots and the new token)
    Outputs: hidden_m_out float32 [1, H-1], hidden_c_out float32 [1, 1],
             k_new / v_new float32 [nkv, 1, hd]

    Every norm is computed from the full mean square (the two parts added), and every linear
    that reads the residual is the ordinary-channel rows plus the massive channel's row, so the
    result equals ``build_layer``'s up to float rounding.
    """
    b = _Builder()
    c = _common(b, model)
    H, inter, nh, nkv, hd = model.H, model.I, model.nh, model.nkv, model.hd
    g = nh // nkv
    q_dim, k_dim = nh * hd, nkv * hd
    keep = np.array([i for i in range(H) if i != massive])
    full = (layer_idx + 1) % model.pattern == 0
    inv = model.inv_full if full else model.inv_local
    conc = model.conc if full else 1.0
    freqs = np.outer(np.arange(MAX_POS, dtype=np.float64), inv)
    emb_ = np.concatenate([freqs, freqs], axis=-1)
    cos_p = b.node(
        "Gather",
        [b.const("sp_cos", (np.cos(emb_) * conc).astype(np.float32)), "position_ids"],
        axis=0,
    )
    sin_p = b.node(
        "Gather",
        [b.const("sp_sin", (np.sin(emb_) * conc).astype(np.float32)), "position_ids"],
        axis=0,
    )
    p = f"model.layers.layers.{layer_idx}."
    Hm = float(H - 1)
    hm_c = b.const("sp_Hm", np.array(Hm, np.float32))
    hh_c = b.const("sp_H", np.array(float(H), np.float32))

    def gain_split(name, offset, tag):
        gg = offset + model.w(p + name)
        return b.const(f"sp_{tag}_m", gg[keep].copy()), b.const(
            f"sp_{tag}_c", gg[massive : massive + 1].copy()
        )

    def rms_split(xm, xc, gm, gc):
        """RMS-normalize the split vector with the full-vector mean square; returns the parts."""
        msm = b.node("ReduceMean", [b.node("Mul", [xm, xm])], axes=[-1], keepdims=1)
        ms = b.node(
            "Div",
            [
                b.node("Add", [b.node("Mul", [msm, hm_c]), b.node("Mul", [xc, xc])]),
                hh_c,
            ],
        )
        r = b.node("Sqrt", [b.node("Add", [ms, c["eps"]])])
        return b.node("Mul", [b.node("Div", [xm, r]), gm]), b.node(
            "Mul", [b.node("Div", [xc, r]), gc]
        )

    def cut(x, lo, hi):
        tag = b.tmp("sp_cut")
        return b.node(
            "Slice",
            [
                x,
                b.const(f"{tag}_s", np.array([lo], np.int64)),
                b.const(f"{tag}_e", np.array([hi], np.int64)),
                c["axl"],
            ],
        )

    def split_full(x):
        """A full-width tensor (no massive channel in it) split into the two parts."""
        xm = b.node("Concat", [cut(x, 0, massive), cut(x, massive + 1, H)], axis=-1)
        return xm, cut(x, massive, massive + 1)

    hm_in, hc_in = "hidden_m", "hidden_c"
    gpre_m, gpre_c = gain_split("pre_mixer_norm.weight", 1.0, "gpre")
    xm, xc = rms_split(hm_in, hc_in, gpre_m, gpre_c)
    Wqkv = model.w(p + "mixer.qkv_proj.weight").T
    qkv = b.node(
        "Add",
        [
            b.node("MatMul", [xm, b.const("sp_Wqkv_m", Wqkv[keep].copy())]),
            b.node(
                "Mul", [xc, b.const("sp_Wqkv_c", Wqkv[massive : massive + 1].copy())]
            ),
        ],
    )
    q, k, v = b.node_multi(
        "Split",
        [qkv, b.const("sp_split_qkv", np.array([q_dim, k_dim, k_dim], np.int64))],
        3,
        axis=-1,
    )
    shape_q = b.const("sp_shape_q", np.array([1, nh, hd], np.int64))
    shape_kv = b.const("sp_shape_kv", np.array([1, nkv, hd], np.int64))
    q = b.node("Transpose", [b.node("Reshape", [q, shape_q])], perm=[1, 0, 2])
    k = b.node("Transpose", [b.node("Reshape", [k, shape_kv])], perm=[1, 0, 2])
    v = b.node("Transpose", [b.node("Reshape", [v, shape_kv])], perm=[1, 0, 2])
    q = b.rms(q, b.const("sp_gq", 1.0 + model.w(p + "mixer.q_norm.weight")), c["eps"])
    k = b.rms(k, b.const("sp_gk", 1.0 + model.w(p + "mixer.k_norm.weight")), c["eps"])

    def rope(x):
        x1 = b.node("Slice", [x, c["s0"], c["sh"], c["axl"]])
        x2 = b.node("Slice", [x, c["sh"], c["shd"], c["axl"]])
        rot = b.node("Concat", [b.node("Neg", [x2]), x1], axis=-1)
        return b.node("Add", [b.node("Mul", [x, cos_p]), b.node("Mul", [rot, sin_p])])

    q, k = rope(q), rope(k)
    kall = b.node("Concat", ["past_k", k], axis=1)
    vall = b.node("Concat", ["past_v", v], axis=1)
    qg = b.node(
        "Reshape", [q, b.const("sp_shape_qg", np.array([nkv, g, 1, hd], np.int64))]
    )
    kt = b.node("Transpose", [b.node("Unsqueeze", [kall, c["ax1"]])], perm=[0, 1, 3, 2])
    vg = b.node("Unsqueeze", [vall, c["ax1"]])
    s = b.node(
        "Add", [b.node("Mul", [b.node("MatMul", [qg, kt]), c["scale"]]), "attn_mask"]
    )
    att = b.node(
        "Reshape",
        [
            b.node(
                "Transpose",
                [b.node("MatMul", [b.node("Softmax", [s], axis=-1), vg])],
                perm=[2, 0, 1, 3],
            ),
            b.const("sp_shape_att", np.array([1, q_dim], np.int64)),
        ],
    )
    o = b.node(
        "MatMul", [att, b.const("sp_Wo", model.w(p + "mixer.o_proj.weight").T.copy())]
    )
    on = b.rms(
        o, b.const("sp_gpm", 0.2 + model.w(p + "post_mixer_norm.weight")), c["eps"]
    )
    on_m, on_c = split_full(on)
    h1m = b.node("Add", [hm_in, on_m])
    h1c = b.node("Add", [hc_in, on_c])
    gpm_m, gpm_c = gain_split("pre_mlp_norm.weight", 1.0, "gpre2")
    xm2, xc2 = rms_split(h1m, h1c, gpm_m, gpm_c)
    Wgu = model.w(p + "mlp.gate_up_proj.weight").T
    gu = b.node(
        "Add",
        [
            b.node("MatMul", [xm2, b.const("sp_Wgu_m", Wgu[keep].copy())]),
            b.node(
                "Mul", [xc2, b.const("sp_Wgu_c", Wgu[massive : massive + 1].copy())]
            ),
        ],
    )
    gate, up = b.node_multi(
        "Split",
        [gu, b.const("sp_split_gu", np.array([inter, inter], np.int64))],
        2,
        axis=-1,
    )
    a = b.node("Mul", [b.node("Mul", [gate, b.node("Sigmoid", [gate])]), up])
    mo = b.node(
        "MatMul", [a, b.const("sp_Wd", model.w(p + "mlp.down_proj.weight").T.copy())]
    )
    mo = b.rms(
        mo, b.const("sp_gpM", 0.2**1.5 + model.w(p + "post_mlp_norm.weight")), c["eps"]
    )
    mo_m, mo_c = split_full(mo)
    b.nodes.append(h.make_node("Add", [h1m, mo_m], ["hidden_m_out"]))
    b.nodes.append(h.make_node("Add", [h1c, mo_c], ["hidden_c_out"]))
    b.nodes.append(h.make_node("Identity", [k], ["k_new"]))
    b.nodes.append(h.make_node("Identity", [v], ["v_new"]))
    inputs = [
        h.make_tensor_value_info("hidden_m", T.FLOAT, [1, H - 1]),
        h.make_tensor_value_info("hidden_c", T.FLOAT, [1, 1]),
        h.make_tensor_value_info("position_ids", T.INT64, [1]),
        h.make_tensor_value_info("past_k", T.FLOAT, [nkv, cache_len, hd]),
        h.make_tensor_value_info("past_v", T.FLOAT, [nkv, cache_len, hd]),
        h.make_tensor_value_info("attn_mask", T.FLOAT, [1, cache_len + 1]),
    ]
    outputs = [
        h.make_tensor_value_info("hidden_m_out", T.FLOAT, [1, H - 1]),
        h.make_tensor_value_info("hidden_c_out", T.FLOAT, [1, 1]),
        h.make_tensor_value_info("k_new", T.FLOAT, [nkv, 1, hd]),
        h.make_tensor_value_info("v_new", T.FLOAT, [nkv, 1, hd]),
    ]
    graph = h.make_graph(
        b.nodes, f"plamo3_layer{layer_idx}_split", inputs, outputs, b.inits
    )
    proto = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)])
    proto.ir_version = 9
    return _save(proto, out_path, external)
