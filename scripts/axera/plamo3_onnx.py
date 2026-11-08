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


def _layer_weights(b, model, i, int8=False, int4=False):
    p = f"model.layers.layers.{i}."
    mat = (
        (lambda name, B: _qconst4(b, name, B))
        if int4
        else (lambda name, B: _qconst(b, name, B))
        if int8
        else (lambda name, B: b.const(name, B))
    )
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
    o = b.node("MatMul", [att, W["w_o"]])
    o = b.rms(o, W["g_post_mix"], c["eps"])
    h1 = b.node("Add", [hidden, o])
    x = b.rms(h1, W["g_pre_mlp"], c["eps"])
    gu = b.node("MatMul", [x, W["w_gu"]])
    gate, up = b.node_multi("Split", [gu, split_mlp], 2, axis=-1)
    a = b.node("Mul", [b.node("Mul", [gate, b.node("Sigmoid", [gate])]), up])
    mo = b.node("MatMul", [a, W["w_d"]])
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
