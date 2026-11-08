"""PLaMo 3 float reference and its ONNX graphs, on a small random model (no checkpoint, no Docker).

The reference is checked for causality. The decode graph (KV cache in and out) and the static
layer graph are checked token by token against the reference, on both kinds of layer (plain
RoPE and YaRN full attention).
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

ort = pytest.importorskip("onnxruntime")
import plamo3_onnx as O  # noqa: E402
import plamo3_reference as R  # noqa: E402

CFG = {
    "num_hidden_layers": 4,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 40,
    "sliding_window_pattern": 2,
    "window_size": 2048,
    "rope_theta": 1000000,
    "rope_local_theta": 10000,
    "rope_scaling_factor": 4.0,
    "rope_parameters": {
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 1000000,
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
    },
}


def _weights(seed=0):
    rng = np.random.RandomState(seed)
    H, inter, hd = CFG["hidden_size"], CFG["intermediate_size"], CFG["head_dim"]
    nh, nkv, V = (
        CFG["num_attention_heads"],
        CFG["num_key_value_heads"],
        CFG["vocab_size"],
    )

    def r(*shape, scale=0.2):
        return (rng.randn(*shape) * scale).astype(np.float32)

    t = {
        "model.embed_tokens.weight": r(V, H),
        "model.norm.weight": r(H, scale=0.1),
        "lm_head.weight": r(V, H),
    }
    for i in range(CFG["num_hidden_layers"]):
        p = f"model.layers.layers.{i}."
        t[p + "pre_mixer_norm.weight"] = r(H, scale=0.1)
        t[p + "post_mixer_norm.weight"] = r(H, scale=0.1)
        t[p + "pre_mlp_norm.weight"] = r(H, scale=0.1)
        t[p + "post_mlp_norm.weight"] = r(H, scale=0.1)
        t[p + "mixer.qkv_proj.weight"] = r((nh + 2 * nkv) * hd, H)
        t[p + "mixer.o_proj.weight"] = r(H, nh * hd)
        t[p + "mixer.q_norm.weight"] = r(hd, scale=0.1)
        t[p + "mixer.k_norm.weight"] = r(hd, scale=0.1)
        t[p + "mlp.gate_up_proj.weight"] = r(2 * inter, H)
        t[p + "mlp.down_proj.weight"] = r(H, inter)
    return t


@pytest.fixture(scope="module")
def model():
    return R.Plamo3Reference(CFG, R.DictWeights(_weights()))


IDS = [3, 17, 5, 29, 8, 11, 2, 35]


def test_reference_is_causal(model):
    full = model.logits(IDS)
    for n in (1, 3, 6):
        np.testing.assert_allclose(
            model.logits(IDS[:n]), full[:n], rtol=1e-4, atol=1e-5
        )


def test_decode_graph_matches_reference(model, tmp_path):
    path = O.build_decode(
        model, str(tmp_path / "decode.onnx"), max_pos=64, external=False
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = model.logits(IDS)
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    past = {}
    for i in range(CFG["num_hidden_layers"]):
        past[f"past_k_{i}"] = np.zeros((nkv, 0, hd), np.float32)
        past[f"past_v_{i}"] = np.zeros((nkv, 0, hd), np.float32)
    names = [o.name for o in sess.get_outputs()]
    got = []
    for pos, tok in enumerate(IDS):
        feed = dict(past)
        feed["input_ids"] = np.array([tok], np.int64)
        feed["position_ids"] = np.array([pos], np.int64)
        out = sess.run(None, feed)
        got.append(out[0][0])
        for j, name in enumerate(names[1:]):
            past[name.replace("present_", "past_")] = out[1 + j]
    got = np.stack(got)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("layer", [0, 1])  # 0: plain RoPE, 1: YaRN full attention
def test_layer_graph_matches_reference(model, tmp_path, layer):
    cap = {}
    h_in = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for i in range(layer):
        h_in = model.layer(i, h_in)
    ref_out = model.layer(layer, h_in, capture=cap)
    path = O.build_layer(
        model, layer, str(tmp_path / f"layer{layer}.onnx"), cache_len=16
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    pk = np.zeros((nkv, 16, hd), np.float32)
    pv = np.zeros((nkv, 16, hd), np.float32)
    outs = []
    for pos in range(len(IDS)):
        mask = np.full((1, 17), -1000.0, np.float32)
        mask[0, :pos] = 0.0
        mask[0, 16] = 0.0
        res = sess.run(
            None,
            {
                "hidden": h_in[pos : pos + 1],
                "position_ids": np.array([pos], np.int64),
                "past_k": pk,
                "past_v": pv,
                "attn_mask": mask,
            },
        )
        outs.append(res[0][0])
        np.testing.assert_allclose(
            res[1][:, 0, :], cap["k"][:, pos, :], rtol=1e-4, atol=1e-4
        )
        pk[:, pos, :] = res[1][:, 0, :]
        pv[:, pos, :] = res[2][:, 0, :]
    np.testing.assert_allclose(np.stack(outs), ref_out, rtol=1e-4, atol=1e-4)


def _quantized_model(model, bits):
    """The same reference with each layer matrix rounded the way the graph stores it: per output column, symmetric int8 or
    int4 (rounded through the grid, then back to float)."""
    qmax = {8: 127.0, 4: 7.0}[bits]
    lo = {8: -127.0, 4: -8.0}[bits]
    tensors = {}
    for name in model.src.keys():
        a = model.src.f32(name)
        if a.ndim == 2 and "layers" in name:
            B = a.T
            sc = np.abs(B).max(axis=0, keepdims=True) / qmax
            sc[sc == 0] = 1.0
            a = (np.clip(np.round(B / sc), lo, qmax) * sc).T
        tensors[name] = a
    return R.Plamo3Reference(CFG, R.DictWeights(tensors))


@pytest.mark.parametrize("bits, fidelity", [(8, 2e-2), (4, 5e-1)])
def test_quantized_chunk_matches_reference(model, tmp_path, bits, fidelity):
    layers = range(0, 2)
    qmodel = _quantized_model(model, bits)
    h = qmodel.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for i in layers:
        h = qmodel.layer(i, h)
    quant_ref = h
    h = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for i in layers:
        h = model.layer(i, h)
    float_ref = h
    path = O.build_chunk(
        model,
        layers,
        str(tmp_path / f"chunk{bits}.onnx"),
        cache_len=16,
        int8=bits == 8,
        int4=bits == 4,
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    emb = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    past = {
        i: (np.zeros((nkv, 16, hd), np.float32), np.zeros((nkv, 16, hd), np.float32))
        for i in layers
    }
    outs = []
    for pos in range(len(IDS)):
        mask = np.full((1, 17), -1000.0, np.float32)
        mask[0, :pos] = 0.0
        mask[0, 16] = 0.0
        feeds = {
            "hidden": emb[pos : pos + 1],
            "position_ids": np.array([pos], np.int64),
            "attn_mask": mask,
        }
        for i in layers:
            feeds[f"past_k_{i}"], feeds[f"past_v_{i}"] = past[i]
        res = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        outs.append(res["hidden_out"][0])
        for i in layers:
            past[i][0][:, pos, :] = res[f"k_new_{i}"][:, 0, :]
            past[i][1][:, pos, :] = res[f"v_new_{i}"][:, 0, :]
    got = np.stack(outs)
    # the graph is the quantized reference (packing and dequantization exact up to float rounding)
    np.testing.assert_allclose(got, quant_ref, rtol=1e-4, atol=1e-4)
    # and the quantized model stays a usable approximation of the float one
    err = np.max(np.abs(got - float_ref)) / np.max(np.abs(float_ref))
    assert err < fidelity, f"int{bits} chunk relative error vs float {err:.3e}"


def test_block_graph_matches_reference(model, tmp_path):
    """The row-block graph (a prompt block padded to the block size, cache empty) equals the reference layer on the real
    rows; the padding rows attend one slot only and are ignored."""
    layer, rows, cache = 1, 8, 16
    ids = IDS[:6]
    h = model.w("model.embed_tokens.weight")[np.asarray(ids)].astype(np.float32)
    ref_out = model.layer(layer, h)
    path = O.build_block(
        model, [layer], str(tmp_path / "block.onnx"), cache_len=cache, rows=rows
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    hid = np.zeros((rows, CFG["hidden_size"]), np.float32)
    hid[: len(ids)] = h
    pos = np.array([min(r, len(ids) - 1) for r in range(rows)], np.int64)
    mask = np.full((rows, cache + rows), -1000.0, np.float32)
    for r in range(rows):
        if r < len(ids):
            mask[r, cache : cache + r + 1] = 0.0
        else:
            mask[r, cache] = 0.0
    zeros = np.zeros((nkv, cache, hd), np.float32)
    out = sess.run(
        None,
        {
            "hidden": hid,
            "position_ids": pos,
            "attn_mask": mask,
            f"past_k_{layer}": zeros,
            f"past_v_{layer}": zeros,
        },
    )
    np.testing.assert_allclose(out[0][: len(ids)], ref_out, rtol=1e-4, atol=1e-4)


def test_resident_state_block_matches_host_cache(model, tmp_path):
    """build_block(state=True) keeps the KV caches in its own outputs: a prompt block then a decode block give the same
    result as the host-managed cache path (same graph without state), slot for slot."""
    layer, rows, cache = 0, 8, 16
    ids = IDS[:7]
    emb = model.w("model.embed_tokens.weight")
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]

    def run(state):
        path = O.build_block(
            model,
            [layer],
            str(tmp_path / f"block_{state}.onnx"),
            cache_len=cache,
            rows=rows,
            state=state,
        )
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        kc = np.zeros((nkv, cache, hd), np.float32)
        vc = np.zeros((nkv, cache, hd), np.float32)
        outs = []
        for p0, toks in [(0, ids[:6]), (6, ids[6:7])]:
            n = len(toks)
            hid = np.zeros((rows, CFG["hidden_size"]), np.float32)
            hid[:n] = emb[np.asarray(toks)]
            pos = np.array([p0 + min(r, n - 1) for r in range(rows)], np.int64)
            mask = np.full((rows, cache + rows), -1000.0, np.float32)
            for r in range(rows):
                if r < n:
                    mask[r, :p0] = 0.0
                    mask[r, cache : cache + r + 1] = 0.0
                else:
                    mask[r, cache] = 0.0
            feeds = {"hidden": hid, "position_ids": pos, "attn_mask": mask}
            if state:
                wsel = np.zeros((cache, rows), np.float32)
                wmask = np.zeros((cache, 1), np.float32)
                for r in range(n):
                    wsel[p0 + r, r] = 1.0
                    wmask[p0 + r, 0] = 1.0
                feeds.update(wsel=wsel, wmask=wmask)
                feeds[f"past_k_{layer}"], feeds[f"past_v_{layer}"] = kc, vc
                res = dict(
                    zip([o.name for o in sess.get_outputs()], sess.run(None, feeds))
                )
                kc, vc = res[f"next_past_k_{layer}"], res[f"next_past_v_{layer}"]
            else:
                feeds[f"past_k_{layer}"], feeds[f"past_v_{layer}"] = kc, vc
                res = dict(
                    zip([o.name for o in sess.get_outputs()], sess.run(None, feeds))
                )
                kc = kc.copy()
                vc = vc.copy()
                kc[:, p0 : p0 + n] = res[f"k_new_{layer}"][:, :n]
                vc[:, p0 : p0 + n] = res[f"v_new_{layer}"][:, :n]
            outs.append(res["hidden_out"][:n])
        return np.concatenate(outs)

    np.testing.assert_allclose(run(True), run(False), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("layer", [0, 1])  # 0: plain RoPE, 1: YaRN full attention
def test_split_layer_graph_matches_reference(model, tmp_path, layer):
    """The split-residual layer (ordinary channels + one massive channel) equals the reference."""
    massive = (
        3  # the random test model has 32 channels; the real model uses MASSIVE_CHANNEL
    )
    cap = {}
    h_in = model.w("model.embed_tokens.weight")[np.asarray(IDS)].astype(np.float32)
    for i in range(layer):
        h_in = model.layer(i, h_in)
    ref_out = model.layer(layer, h_in, capture=cap)
    path = O.build_layer_split(
        model,
        layer,
        str(tmp_path / f"split{layer}.onnx"),
        cache_len=16,
        massive=massive,
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nkv, hd = CFG["num_key_value_heads"], CFG["head_dim"]
    keep = np.arange(CFG["hidden_size"]) != massive
    pk = np.zeros((nkv, 16, hd), np.float32)
    pv = np.zeros((nkv, 16, hd), np.float32)
    outs = []
    for pos in range(len(IDS)):
        mask = np.full((1, 17), -1000.0, np.float32)
        mask[0, :pos] = 0.0
        mask[0, 16] = 0.0
        hin = h_in[pos : pos + 1]
        res = sess.run(
            None,
            {
                "hidden_m": hin[:, keep].copy(),
                "hidden_c": hin[:, massive : massive + 1].copy(),
                "position_ids": np.array([pos], np.int64),
                "past_k": pk,
                "past_v": pv,
                "attn_mask": mask,
            },
        )
        full = np.concatenate([res[0][0, :massive], res[1][0], res[0][0, massive:]])
        outs.append(full)
        np.testing.assert_allclose(
            res[2][:, 0, :], cap["k"][:, pos, :], rtol=1e-4, atol=1e-4
        )
        pk[:, pos, :] = res[2][:, 0, :]
        pv[:, pos, :] = res[3][:, 0, :]
    np.testing.assert_allclose(np.stack(outs), ref_out, rtol=1e-4, atol=1e-4)
